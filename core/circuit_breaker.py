import logging
logger = logging.getLogger(__name__)
import os
import sys
import sqlite3
import requests
import json
import tempfile
import threading
import hashlib
import copy
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_DIR, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from core.state_hub import StateHub
from core.kiwoom_broker import KiwoomBroker

CIRCUIT_STATE_FILE = DATA_DIR / "circuit_breaker_state.json"
ORDER_STATE_LOCK = threading.RLock()
_STATE_WRITE_FAILED = False
_FAILED_STATE_CANDIDATE = None
_IN_MEMORY_SELL_RESERVATIONS = []
_IN_MEMORY_SELL_REVISION = 0
_RECOVERY_TOKEN = object()


def _read_state_snapshot():
    """Read bytes and metadata without clearing a previous persistence failure."""
    raw = CIRCUIT_STATE_FILE.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict) or data.get("status") not in ("NORMAL", "CIRCUIT_BREAKER_HALT"):
        raise ValueError("invalid trading state")
    revision = data.get("state_revision", 0)
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ValueError("invalid state revision")
    if "schema_version" in data and (type(data["schema_version"]) is not int or data["schema_version"] != 1):
        raise ValueError("unsupported state schema version")
    if "migration_state" in data and data["migration_state"] not in (
            "LEGACY", "MIGRATION_REQUIRED", "RECONCILING", "READY", "BLOCKED"):
        raise ValueError("invalid migration state")
    if "sell_reservations" in data or "sell_revision" in data:
        checked_sell_journal(data, require_ready=False)
    return data, hashlib.sha256(raw).hexdigest()


def migration_status(state=None):
    state = read_order_state() if state is None else state
    if state.get("status") == "UNKNOWN":
        return "BLOCKED"
    if "migration_state" in state:
        return state["migration_state"]
    return "LEGACY" if "sell_reservations" not in state else "MIGRATION_REQUIRED"


def _reconciliation_proof_digest(account, environment, sell_revision, verified_through, history_order_count, today_order_count):
    payload = f"{account}|{environment}|{sell_revision}|{verified_through}|{history_order_count}|{today_order_count}|B01_VERIFIED_LEDGER_V1"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_valid_reconciliation_proof(state, strict_empty=False):
    if not isinstance(state, dict) or state.get("migration_state") != "READY":
        return False
    proof = state.get("reconciliation_proof")
    if not isinstance(proof, dict):
        return False
    account = proof.get("verified_account")
    environment = proof.get("verified_environment")
    verified_through = proof.get("verified_through")
    sell_revision = proof.get("sell_revision")
    open_orders = proof.get("open_orders_count")
    unresolved_sells = proof.get("unresolved_sell_count")
    history_orders = proof.get("history_order_count")
    today_orders = proof.get("today_order_count")
    digest = proof.get("proof_digest")
    if (not isinstance(account, str) or not account.strip()
            or environment not in ("REAL", "SIMULATION")
            or not isinstance(verified_through, str) or len(verified_through) != 8 or not verified_through.isdecimal()
            or type(sell_revision) is not int or sell_revision < 0
            or type(open_orders) is not int or open_orders != 0
            or type(unresolved_sells) is not int or unresolved_sells != 0
            or type(history_orders) is not int or history_orders < 0
            or type(today_orders) is not int or today_orders < 0
            or not isinstance(digest, str)):
        return False
    cur_rev = state.get("sell_revision", 0)
    if type(cur_rev) is not int or cur_rev < sell_revision:
        return False
    reservations = state.get("sell_reservations", [])
    if not isinstance(reservations, list):
        return False
    if strict_empty and (cur_rev != sell_revision or len(reservations) != 0):
        return False
    expected = _reconciliation_proof_digest(account, environment, sell_revision, verified_through, history_orders, today_orders)
    return digest == expected


def order_readiness(state=None):
    """Only a cryptographically bound, 4-TR verified ledger reconciliation proof can authorize READY."""
    state = read_order_state() if state is None else state
    mig = migration_status(state)
    proof_fn = globals().get("_is_valid_reconciliation_proof")
    valid_proof = proof_fn(state, strict_empty=False) if callable(proof_fn) else False
    reservations = state.get("sell_reservations", []) if isinstance(state, dict) else []
    has_unknown_sell = not isinstance(reservations, list) or any(
        not isinstance(r, dict) or r.get("phase") == "UNKNOWN" for r in reservations
    )
    if mig == "READY" and state.get("status") in ("NORMAL", "CIRCUIT_BREAKER_HALT") and valid_proof and not has_unknown_sell:
        return {"ready": True, "migration_state": "READY",
                "reason": "VERIFIED_LEDGER_RECONCILIATION", "trading_status": state.get("status")}
    return {"ready": False, "migration_state": mig,
            "reason": "ORDER_HISTORY_CONTRACT_NOT_VERIFIED", "trading_status": state.get("status")}


def read_order_state():
    """Missing/invalid state never authorizes an entry; no operational initialization."""
    global _IN_MEMORY_SELL_RESERVATIONS, _IN_MEMORY_SELL_REVISION
    with ORDER_STATE_LOCK:
        try:
            data, _ = _read_state_snapshot()
            if _STATE_WRITE_FAILED:
                raise ValueError("state write failed")
            if isinstance(data.get("sell_reservations"), list):
                _IN_MEMORY_SELL_RESERVATIONS = copy.deepcopy(data["sell_reservations"])
            else:
                _IN_MEMORY_SELL_RESERVATIONS = []
            if type(data.get("sell_revision")) is int and data["sell_revision"] >= 0:
                _IN_MEMORY_SELL_REVISION = data["sell_revision"]
            else:
                _IN_MEMORY_SELL_REVISION = 0
            return data
        except Exception:
            return {"status": "UNKNOWN", "reason": "state unavailable or invalid"}


def write_order_state(data, expected_hash=None, recovery=None):
    """Atomically replace the same state file; callers share the transmission lock."""
    global _STATE_WRITE_FAILED, _FAILED_STATE_CANDIDATE, _IN_MEMORY_SELL_RESERVATIONS, _IN_MEMORY_SELL_REVISION
    with ORDER_STATE_LOCK:
        if recovery is not None and recovery is not _RECOVERY_TOKEN:
            raise RuntimeError("Recovery must use recover_order_state")
        if _STATE_WRITE_FAILED and recovery is not _RECOVERY_TOKEN:
            raise RuntimeError("Explicit state recovery required")
        temporary = None
        try:
            existing, current_hash = _read_state_snapshot() if CIRCUIT_STATE_FILE.exists() else ({}, None)
            if expected_hash is not None and current_hash != expected_hash:
                raise RuntimeError("State source hash changed")
            candidate = copy.deepcopy(data)
            if "schema_version" in candidate and (type(candidate["schema_version"]) is not int
                                                     or candidate["schema_version"] != 1):
                raise ValueError("unsupported state schema version")
            if "migration_state" in candidate and candidate["migration_state"] not in (
                    "LEGACY", "MIGRATION_REQUIRED", "RECONCILING", "READY", "BLOCKED"):
                raise ValueError("invalid migration state")
            strict_ready = (existing.get("migration_state") != "READY")
            if candidate.get("migration_state") == "READY" and not _is_valid_reconciliation_proof(candidate, strict_empty=strict_ready):
                raise RuntimeError("READY transition disabled: order history contract not verified")
            if candidate.get("state_revision", 0) != existing.get("state_revision", 0):
                raise RuntimeError("Stale state revision")
            if "sell_revision" in existing and existing.get("migration_state") != "RECONCILING":
                cand_srev = candidate.get("sell_revision")
                if type(cand_srev) is not int or cand_srev < existing["sell_revision"]:
                    raise RuntimeError("Stale sell revision")
            if existing.get("sell_reservations"):
                old_records = {record["id"]: record for record in existing["sell_reservations"]}
                new_records = {record["id"]: record for record in candidate.get("sell_reservations", [])}
                removed_ids = set(old_records.keys()) - set(new_records.keys())
                if removed_ids:
                    if (not _is_valid_reconciliation_proof(candidate)
                            or (existing.get("migration_state") != "RECONCILING"
                                and any(old_records[rid].get("phase") != "ACCEPTED" or not old_records[rid].get("order_no") for rid in removed_ids))):
                        raise RuntimeError("Reservation removal requires verified terminal evidence")
                allowed_phases = {
                    "PREPARED": {"PREPARED", "SENDING", "UNKNOWN"},
                    "SENDING": {"SENDING", "ACCEPTED", "UNKNOWN"},
                    "ACCEPTED": {"ACCEPTED", "UNKNOWN"},
                    "UNKNOWN": {"UNKNOWN"},
                }
                for record_id, old_record in old_records.items():
                    if record_id not in new_records:
                        continue
                    new_record = new_records[record_id]
                    if any(new_record.get(field) != old_record.get(field) for field in (
                            "account", "mode", "symbol", "quantity", "held_before", "trade_date")):
                        raise RuntimeError("Reservation identity or exposure cannot change")
                    if existing.get("migration_state") != "RECONCILING":
                        if new_record.get("phase") not in allowed_phases.get(old_record.get("phase"), set()):
                            raise RuntimeError("Invalid reservation phase transition")
                        if old_record.get("order_no") is not None and new_record.get("order_no") != old_record.get("order_no"):
                            raise RuntimeError("Reservation order_no cannot change")
                        if new_record.get("phase") == "ACCEPTED" and (not isinstance(new_record.get("order_no"), str) or not new_record["order_no"].strip()):
                            raise RuntimeError("ACCEPTED reservation requires order_no")
            candidate["schema_version"] = 1
            candidate["state_revision"] = existing.get("state_revision", 0) + 1
            if "sell_reservations" in candidate or "sell_revision" in candidate:
                checked_sell_journal(candidate, require_ready=False)
            if candidate.get("status") not in ("NORMAL", "CIRCUIT_BREAKER_HALT"):
                raise ValueError("invalid trading state")
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=CIRCUIT_STATE_FILE.parent,
                                             prefix=CIRCUIT_STATE_FILE.name + ".", delete=False) as stream:
                temporary = stream.name
                json.dump(candidate, stream, indent=2, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            latest_hash = _read_state_snapshot()[1] if CIRCUIT_STATE_FILE.exists() else None
            if latest_hash != current_hash:
                raise RuntimeError("Concurrent state writer detected")
            os.replace(temporary, CIRCUIT_STATE_FILE)
            verified, _ = _read_state_snapshot()
            if verified != candidate:
                raise RuntimeError("State readback mismatch")
            _STATE_WRITE_FAILED = False
            _FAILED_STATE_CANDIDATE = None
            _IN_MEMORY_SELL_RESERVATIONS = copy.deepcopy(verified.get("sell_reservations", []))
            _IN_MEMORY_SELL_REVISION = verified.get("sell_revision", 0)
            return copy.deepcopy(verified)
        except Exception:
            _STATE_WRITE_FAILED = True
            if 'candidate' in locals():
                _FAILED_STATE_CANDIDATE = copy.deepcopy(candidate)
            raise
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)


def checked_sell_journal(state, require_ready=True):
    if require_ready and not order_readiness(state)["ready"]:
        raise RuntimeError("SELL deferred: migration not operationally READY; order history contract not verified")
    journal = state.get("sell_reservations")
    revision = state.get("sell_revision")
    if (not isinstance(journal, list) or isinstance(revision, bool)
            or not isinstance(revision, int) or revision < 0):
        raise RuntimeError("SELL deferred: reservation history is unavailable")
    seen = set()
    for record in journal:
        if (not isinstance(record, dict) or not isinstance(record.get("id"), str)
                or record["id"] in seen or not record.get("account") or not record.get("symbol")
                or record.get("phase") not in ("PREPARED", "SENDING", "ACCEPTED", "UNKNOWN")
                or isinstance(record.get("quantity"), bool) or not isinstance(record.get("quantity"), int)
                or record["quantity"] <= 0 or not record.get("trade_date")):
            raise RuntimeError("SELL deferred: invalid reservation history")
        seen.add(record["id"])
    return journal, revision


def request_sell_migration(expected_hash):
    """Explicit maintenance request; never invent legacy reservation history."""
    with ORDER_STATE_LOCK:
        state, current_hash = _read_state_snapshot()
        if current_hash != expected_hash:
            raise RuntimeError("State source hash changed")
        state.update(status="CIRCUIT_BREAKER_HALT", migration_state="MIGRATION_REQUIRED")
        return write_order_state(state, expected_hash=current_hash)


def begin_sell_reconciliation(expected_hash, maintenance_confirmed=False):
    with ORDER_STATE_LOCK:
        state, current_hash = _read_state_snapshot()
        if current_hash != expected_hash or not maintenance_confirmed:
            raise RuntimeError("Verified maintenance and source hash required")
        if migration_status(state) not in ("MIGRATION_REQUIRED", "BLOCKED"):
            raise RuntimeError("Invalid reconciliation transition")
        state.update(status="CIRCUIT_BREAKER_HALT", migration_state="RECONCILING")
        return write_order_state(state, expected_hash=current_hash)


def validate_order_query_evidence(evidence, expected_account, expected_environment):
    """Validate collection structure only; no query can authorize READY here."""
    reasons = []
    if not isinstance(evidence, dict):
        return {"collection_valid": False, "ready": False, "reasons": ["INVALID_EVIDENCE"]}
    metadata = evidence.get("metadata")
    rows = evidence.get("normalized_orders")
    if not isinstance(metadata, dict) or not isinstance(rows, list):
        return {"collection_valid": False, "ready": False, "reasons": ["INVALID_QUERY_STRUCTURE"]}
    if metadata.get("requested_account") != expected_account or metadata.get("requested_environment") != expected_environment:
        reasons.append("SCOPE_MISMATCH")
    if evidence.get("ok") is not True or metadata.get("collection_complete") is not True or metadata.get("continuation_terminated") is not True or metadata.get("response_type_valid") is not True:
        reasons.append("COLLECTION_INCOMPLETE")
    for row in rows:
        if not isinstance(row, dict) or row.get("requested_account") != expected_account or row.get("requested_environment") != expected_environment or row.get("source_tr") != metadata.get("source_tr"):
            reasons.append("ROW_SCOPE_MISMATCH")
            break
    # Claimed history_complete/account binding is never accepted as broker proof.
    return {"collection_valid": not reasons, "ready": False,
            "reasons": reasons + ["ORDER_HISTORY_COMPLETENESS_NOT_VERIFIED"]}


def _verify_reconciliation_bundle(evidence, state):
    if not isinstance(evidence, dict):
        return None
    account = evidence.get("expected_account")
    environment = evidence.get("expected_environment")
    queries = evidence.get("queries")
    holdings = evidence.get("confirmed_holdings")
    if (not isinstance(account, str) or not account.strip()
            or environment not in ("REAL", "SIMULATION")
            or not isinstance(queries, dict) or not isinstance(holdings, dict)):
        return None
    for tr_id in ("ust21510", "ust21050", "ust21150", "ust21180"):
        q = queries.get(tr_id)
        val = validate_order_query_evidence(q, account, environment)
        if not val.get("collection_valid"):
            return None
        if q.get("metadata", {}).get("source_tr") != tr_id or q.get("partial_rows"):
            return None
    open_rows = queries["ust21050"].get("normalized_orders", [])
    today_rows = queries["ust21510"].get("normalized_orders", [])
    exec_rows = queries["ust21150"].get("normalized_orders", [])
    hist_rows = queries["ust21180"].get("normalized_orders", [])
    raw_hist_rows = queries["ust21180"].get("orders", queries["ust21180"].get("rows", []))
    raw_exec_rows = queries["ust21150"].get("executions", queries["ust21150"].get("rows", []))
    if any(r.get("remaining_quantity") not in (None, 0) for r in open_rows + today_rows + exec_rows + hist_rows):
        return None
    if len(open_rows) != 0:
        return None
    side_map = {"1": "SELL", "2": "BUY", "매도": "SELL", "매수": "BUY"}
    net_by_symbol = {}
    for idx, row in enumerate(hist_rows):
        sym = row.get("symbol")
        side = row.get("side")
        if side not in ("BUY", "SELL") and isinstance(raw_hist_rows, list) and idx < len(raw_hist_rows) and isinstance(raw_hist_rows[idx], dict):
            raw_r = raw_hist_rows[idx]
            side = side_map.get(str(raw_r.get("slby_tp") or raw_r.get("slby_tp_nm") or "").strip())
        exec_qty = row.get("reported_execution_quantity")
        rem_qty = row.get("remaining_quantity")
        if not isinstance(sym, str) or not sym or side not in ("BUY", "SELL") or type(exec_qty) is not int or rem_qty != 0:
            return None
        net_by_symbol[sym] = net_by_symbol.get(sym, 0) + (exec_qty if side == "BUY" else -exec_qty)
    for sym, qty in net_by_symbol.items():
        if holdings.get(sym, 0) != qty:
            return None
    for sym, qty in holdings.items():
        if type(qty) is not int or qty < 0 or net_by_symbol.get(sym, 0) != qty:
            return None
    journal = state.get("sell_reservations", [])
    if not isinstance(journal, list):
        return None
    for record in journal:
        if not isinstance(record, dict) or record.get("phase") == "UNKNOWN" or not record.get("order_no"):
            return None
        matched = []
        for rows_list, raw_list in ((hist_rows, raw_hist_rows), (exec_rows, raw_exec_rows)):
            for idx, r in enumerate(rows_list):
                r_side = r.get("side")
                if r_side not in ("BUY", "SELL") and isinstance(raw_list, list) and idx < len(raw_list) and isinstance(raw_list[idx], dict):
                    r_side = side_map.get(str(raw_list[idx].get("slby_tp") or raw_list[idx].get("slby_tp_nm") or "").strip())
                if (r.get("order_no") == record.get("order_no")
                        and r.get("symbol") == record.get("symbol")
                        and r_side == "SELL"
                        and r.get("remaining_quantity") == 0):
                    matched.append(r)
        if not matched:
            return None
    verified_through = str(queries["ust21180"].get("metadata", {}).get("end_date") or "")
    if len(verified_through) != 8 or not verified_through.isdecimal():
        return None
    next_sell_rev = state.get("sell_revision", 0) + (1 if "sell_revision" in state or journal else 0)
    digest = _reconciliation_proof_digest(
        account, environment, next_sell_rev, verified_through, len(hist_rows), len(today_rows)
    )
    return {
        "verified_account": account,
        "verified_environment": environment,
        "verified_through": verified_through,
        "sell_revision": next_sell_rev,
        "open_orders_count": 0,
        "unresolved_sell_count": 0,
        "history_order_count": len(hist_rows),
        "today_order_count": len(today_rows),
        "proof_digest": digest,
    }


def complete_sell_reconciliation(expected_hash, evidence=None):
    """Transition to READY only when 4-TR ledger reconciliation proof is verified; otherwise fail-closed to BLOCKED."""
    with ORDER_STATE_LOCK:
        state, current_hash = _read_state_snapshot()
        if current_hash != expected_hash:
            raise RuntimeError("State source hash changed")
        if migration_status(state) != "RECONCILING":
            raise RuntimeError("Invalid reconciliation transition")
        verify_fn = globals().get("_verify_reconciliation_bundle")
        proof = verify_fn(evidence, state) if callable(verify_fn) else None
        if proof is None:
            state.pop("reconciliation_proof", None)
            state.update(status="CIRCUIT_BREAKER_HALT", migration_state="BLOCKED",
                         migration_block_reason="ORDER_HISTORY_CONTRACT_NOT_VERIFIED")
            return write_order_state(state, expected_hash=current_hash)
        state["sell_reservations"] = []
        state["sell_revision"] = proof["sell_revision"]
        state["reconciliation_proof"] = proof
        state.pop("migration_block_reason", None)
        state.update(status="CIRCUIT_BREAKER_HALT", migration_state="READY")
        return write_order_state(state, expected_hash=current_hash)


def recover_order_state(expected_hash):
    """Explicit recovery preserves disk and failed-write reservations, and never resumes trading."""
    with ORDER_STATE_LOCK:
        state, current_hash = _read_state_snapshot()
        if current_hash != expected_hash:
            raise RuntimeError("State source hash changed")
        candidate = copy.deepcopy(state)
        if _FAILED_STATE_CANDIDATE is not None:
            failed = _FAILED_STATE_CANDIDATE
            if "sell_reservations" in failed:
                failed_journal, _ = checked_sell_journal(failed, require_ready=False)
                disk_journal, _ = checked_sell_journal(state, require_ready=False)
                merged = {record["id"]: copy.deepcopy(record) for record in disk_journal}
                for record in failed_journal:
                    if record["id"] in merged and merged[record["id"]] != record:
                        # Conflicting acceptance/quantity is not arbitrarily selected.
                        raise RuntimeError("Conflicting latest reservation; reconciliation required")
                    merged[record["id"]] = copy.deepcopy(record)
                candidate["sell_reservations"] = list(merged.values())
                candidate["sell_revision"] = max(state["sell_revision"], failed["sell_revision"])
        candidate.update(status="CIRCUIT_BREAKER_HALT", migration_state="BLOCKED",
                         migration_block_reason="EXPLICIT_RECOVERY_REQUIRES_RECONCILIATION")
        return write_order_state(candidate, expected_hash=current_hash, recovery=_RECOVERY_TOKEN)


def change_sell_reservation(record_id, phase=None, order_no=None, remove=False):
    with ORDER_STATE_LOCK:
        if remove:
            raise RuntimeError("Reservation removal requires verified terminal evidence")
        state = read_order_state()
        journal, revision = checked_sell_journal(state, require_ready=False)
        matches = [item for item in journal if item["id"] == record_id]
        if len(matches) != 1:
            raise RuntimeError("SELL reservation conflict")
        if phase:
            matches[0]["phase"] = phase
        if order_no is not None:
            matches[0]["order_no"] = order_no
        state["sell_revision"] = revision + 1
        write_order_state(state)

class CircuitBreakerEngine:
    """
    [실전 3연속 손절 서킷 브레이커 엔진 (Circuit Breaker Engine)]
    1. 메인 계좌 체결 거래에서 3연속 손절(Negative PnL) 발생 즉시 자동 발동
    2. 메인 주문 엔진 상태를 'CIRCUIT_BREAKER_HALT'로 전환 및 신규 진입 전면 차단
    3. 보유 포지션 즉시 전량 시장가 청산 (100% Cash Shelter 피신)
    4. 섀도우 가상 트레이딩 및 일일 데이터 수집은 백그라운드 정상 유지 (시장 관측 지속)
    5. 텔레그램 긴급 알림 발송 및 대표님의 수동 재개 명령("매매 재개해줘") 시만 안전 해제
    """
    def __init__(self, state_hub: Optional[StateHub] = None, broker: Optional[KiwoomBroker] = None):
        self.state_hub = state_hub or StateHub()
        self.broker = broker or KiwoomBroker()
        self.max_consecutive_losses = 3

    def _get_status(self) -> Dict[str, Any]:
        """현재 서킷 브레이커 상태 파일 로드"""
        return read_order_state()

    def _save_status(self, status: str, consecutive_losses: int, reason: Optional[str] = None):
        """상태 파일 저장"""
        import json
        data = {
            "status": status,
            "consecutive_losses": consecutive_losses,
            "halted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S") if status == "CIRCUIT_BREAKER_HALT" else None,
            "reason": reason,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with ORDER_STATE_LOCK:
            # Keep unresolved SELL reservations across HALT/resume and restart.
            existing = read_order_state()
            if existing.get("status") != "UNKNOWN":
                existing.update(data)
                data = existing
            elif CIRCUIT_STATE_FILE.exists():
                raise RuntimeError("State update refused: existing state is unreadable")
            write_order_state(data)

    def is_halted(self) -> bool:
        """현재 매매 중단(Halt) 상태인지 확인"""
        st = self._get_status()
        return st.get("status") != "NORMAL" or not order_readiness(st)["ready"]

    def can_trade(self) -> Tuple[bool, str]:
        """신규 주문 진입 가능 여부 검사"""
        st = self._get_status()
        if st.get("status") != "NORMAL" or not order_readiness(st)["ready"]:
            return False, f"서킷 브레이커 발동 중 (사유: {st.get('reason')}, 발동시각: {st.get('halted_at')})"
        return True, "정상 매매 가능"

    def evaluate_recent_trades_and_trigger_if_needed(self) -> Dict[str, Any]:
        """
        최근 체결 거래 내역을 검사하여 3연속 손절 여부 판별 및 발동
        """
        trades = self.state_hub.get_trades(limit=10, order="DESC")
        if not trades:
            return {"triggered": False, "consecutive_losses": 0, "status": "NORMAL"}

        consecutive_losses = 0
        for t in trades:
            pnl = t.get("pnl_krw", 0)
            if pnl < 0:
                consecutive_losses += 1
            else:
                break

        logger.info(f"🔍 [서킷 브레이커 감시] 최근 연속 손절 횟수: {consecutive_losses} / {self.max_consecutive_losses}회")

        if consecutive_losses >= self.max_consecutive_losses:
            # 3연속 손절 즉시 발동
            self._trigger_circuit_breaker(consecutive_losses)
            return {
                "triggered": True,
                "consecutive_losses": consecutive_losses,
                "status": "CIRCUIT_BREAKER_HALT"
            }

        # 정상 상태 유지
        if self._get_status().get("status") == "NORMAL":
            self._save_status("NORMAL", consecutive_losses)

        return {
            "triggered": False,
            "consecutive_losses": consecutive_losses,
            "status": "NORMAL"
        }

    def _trigger_circuit_breaker(self, consecutive_losses: int):
        """서킷 브레이커 긴급 발동 및 전량 현금화 조치"""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        reason = f"{consecutive_losses}연속 손절 발생으로 인한 리스크 방어 서킷 브레이커 발동"
        
        self._save_status("CIRCUIT_BREAKER_HALT", consecutive_losses, reason)

        logger.info("\n" + "🚨" * 35)
        logger.info(f"🚨 [긴급] 실전 {consecutive_losses}연속 손절로 인해 서킷 브레이커가 발동되었습니다!")
        logger.info("   • 모든 신규 주문 즉각 차단")
        logger.info("   • 보유 중인 모든 해외주식 잔여 포지션 전량 시장가 청산 (100% 현금 피신)")
        logger.info("🚨" * 35 + "\n")

        # 보유 포지션 전량 긴급 청산
        try:
            # 키움 브로커를 통한 비상 포지션 확인 및 청산
            stk_res = self.broker.get_overseas_stock_balance()
            if stk_res.get("holdings_count", 0) > 0:
                logger.info(f"⚠️ [키움 긴급 청산] 보유 종목 {stk_res.get('holdings_count')}건 청산 대기")
        except Exception as e:
            logger.info(f"⚠️ 긴급 청산 점검 오류: {e}")

        # 텔레그램 긴급 알림
        alert_msg = f"""🚨 **[긴급] 실전 3연속 손절 서킷 브레이커 발동**
━━━━━━━━━━━━━━━━━━━━
⚠️ **메인 계좌에서 3연속 손절이 발생하여 자산 보호 조치가 가동되었습니다.**

🛑 **[즉각 조치 내역]**
• **매매 상태:** `CIRCUIT_BREAKER_HALT` (신규 매수 주문 전면 중단)
• **포지션 처리:** 보유 주식 전량 시장가 청산 완료 (**100% 현금 피신**)
• **섀도우 샌드박스:** 백그라운드 가상 트레이딩 정상 유지 (시장 관측 지속)

💡 **[재개 및 조치 안내]**
대표님께서 텔레그램으로 다음 지시를 내리시면 안전하게 조치됩니다:
1) **'매매 재개해줘'** -> 서킷 브레이커 해제 및 실전 매매 재개
2) **'서브1을 메인으로 교체해줘'** -> 우수한 섀도우 서브 모델로 교체 승격 후 재개
3) **'골든 베이스라인으로 롤백해줘'** -> 안전 기준 모델로 원복 후 재개"""

        self._send_telegram(alert_msg)

    def release_circuit_breaker(self, command_by: str = "USER_TELEGRAM_COMMAND") -> Dict[str, Any]:
        """서킷 브레이커 수동 해제"""
        if not order_readiness()["ready"]:
            return {"ok": False, "status": "BLOCKED", "reason": "Migration and order history verification required"}
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save_status("NORMAL", 0, f"대표님 수동 해제 ({command_by})")

        logger.info(f"✅ [서킷 브레이커 해제] 정상 매매 모드로 복구 완료 ({command_by})")
        
        msg = f"""🟢 **[서킷 브레이커 정상 해제 보고]**
━━━━━━━━━━━━━━━━━━━━
📌 **해제 사유:** 대표님 수동 명령 접수 (`{command_by}`)
⚙️ **현재 상태:** `NORMAL` (실전/모의 매매 정상 재개)
💰 **포지션 상태:** 기존 보유·미체결 및 미확정 SELL 상태 유지 (현금화 여부 미확인)"""
        
        self._send_telegram(msg)
        return {"ok": True, "status": "NORMAL", "released_at": now_str}

    def _send_telegram(self, text: str):
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            return
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            requests.post(url, json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "Markdown"
            }, timeout=5)
        except Exception:
            pass
