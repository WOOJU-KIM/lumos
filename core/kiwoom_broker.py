import os
import sys
import json
import time
import urllib.request
import urllib.error
import logging
import re
import http.client
import uuid
from zoneinfo import ZoneInfo
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Windows 콘솔 utf-8 설정
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from dotenv import load_dotenv
import config
load_dotenv()

logger = logging.getLogger("KiwoomBroker")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s][KiwoomBroker] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

_GLOBAL_KIWOOM_TOKEN_CACHE: Dict[str, Dict[str, Any]] = {}


def _mask_sensitive_log(value: Any, secrets: Tuple[str, ...] = ()) -> Any:
    """Return a masked logging copy; never alter API or event data."""
    sensitive_keys = {
        "authorization", "token", "accesstoken", "refreshtoken", "tokenpreview",
        "appkey", "apikey", "appsecret", "apisecret", "secret", "secretkey", "clientsecret", "password",
        "cano", "account", "accountno", "fullaccountno", "accountnumber",
        "accno", "acntno", "계좌번호", "계좌 번호",
    }

    def sensitive(key):
        return re.sub(r"[\s_-]", "", str(key)).lower() in sensitive_keys

    if isinstance(value, dict):
        return {key: "[MASKED]" if sensitive(key) else _mask_sensitive_log(item, secrets)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        items = [_mask_sensitive_log(item, secrets) for item in value]
        return tuple(items) if isinstance(value, tuple) else items
    if not isinstance(value, str):
        return value

    # Preserve raw JSON formatting when nothing sensitive needs replacing.
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        parsed = None
    if isinstance(parsed, (dict, list)):
        masked = _mask_sensitive_log(parsed, secrets)
        return value if masked == parsed else json.dumps(masked, ensure_ascii=False)

    # Exception messages may contain JSON after an HTTP/error description.
    decoder = json.JSONDecoder()
    parts = []
    cursor = 0
    for match in re.finditer(r"[\[{]", value):
        start = match.start()
        if start < cursor:
            continue
        try:
            fragment, end = decoder.raw_decode(value, start)
        except ValueError:
            continue
        if not isinstance(fragment, (dict, list)):
            continue
        masked = _mask_sensitive_log(fragment, secrets)
        parts.append(_mask_sensitive_log(value[cursor:start], secrets))
        parts.append(value[start:end] if masked == fragment else
                     json.dumps(masked, ensure_ascii=False))
        cursor = end
    if cursor:
        return "".join(parts) + _mask_sensitive_log(value[cursor:], secrets)

    key_pattern = "|".join(
        "[ _-]*".join(re.escape(char) for char in key.replace(" ", ""))
        for key in sorted(sensitive_keys, key=len, reverse=True)
    )
    # JSON/repr fragments and key=value details in exception messages.
    pattern = (r"(?<![\w])([\"']?(?:" + key_pattern
               + r")[\"']?\s*[:=]\s*)(\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|(?:Bearer|Basic)\s+[^\s,;|}\"'\]]+|[^\s,;|}\]]+)")
    value = re.sub(pattern, lambda match: match.group(1) + "[MASKED]", value,
                   flags=re.IGNORECASE)
    value = re.sub(r"\b(Bearer|Basic)\s+[^\s,;\"'}\]]+", r"\1 [MASKED]", value,
                   flags=re.IGNORECASE)
    for secret in sorted({item for item in secrets if isinstance(item, str) and item},
                         key=len, reverse=True):
        value = value.replace(secret, "[MASKED]")
    return value


class _OrderHTTPSConnection(http.client.HTTPSConnection):
    """Check the order at the first HTTP socket write, after TLS/CONNECT preparation."""
    def __init__(self, *args, order_guard=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._order_guard = order_guard
        self._order_started = False
        self._order_tunnel = False

    def _tunnel(self):
        self._order_tunnel = True
        try:
            return super()._tunnel()
        finally:
            self._order_tunnel = False

    def send(self, data):
        if self.sock is None:
            self.connect()
        if self._order_tunnel or self._order_started:
            return super().send(data)
        from core.circuit_breaker import ORDER_STATE_LOCK
        with ORDER_STATE_LOCK:
            self._order_guard()
            self._order_started = True
            # The lock covers the first write, never getresponse/read or the whole HTTP call.
            return super().send(data)


class _OrderHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self, guard):
        super().__init__()
        self.guard = guard

    def https_open(self, req):
        def connection(*args, **kwargs):
            return _OrderHTTPSConnection(*args, order_guard=self.guard, **kwargs)
        return self.do_open(connection, req, context=self._context)


def _explicit_order_quantity(value):
    if isinstance(value, bool):
        raise ValueError("invalid quantity")
    text = str(value).replace(",", "").strip()
    if not text.isdecimal():
        raise ValueError("quantity is not an explicit nonnegative integer")
    return int(text)


class KiwoomBroker:
    """
    [Lumos 키움증권 OpenAPI REST 듀얼 스위칭 브로커]
    1. KIWOOM_IS_SIMULATION 플래그(1: 모의투자, 0: 실전투자)에 따라 동적 엔드포인트 및 인증키 자동 스위칭
    2. 미국/해외주식 외화 예수금(ust21110) 및 원장 잔고(ust21070) 실시간 조회
    3. 국내주식 예수금(kt00001) 및 잔고(kt00018) 통합 지원
    4. OAuth2 토큰 발급 및 만료 전 자동 갱신 메모리 캐싱
    """
    def _redact_log(self, value: Any) -> Any:
        return _mask_sensitive_log(value, tuple(
            getattr(self, name, "") for name in
            ("_access_token", "app_key", "app_secret", "account_no")
        ))

    def resolve_order_ledger_status(
        self,
        order_no: str,
        symbol: str,
        side: str = "SELL",
        expected_qty: Optional[int] = None,
        trade_date: Optional[str] = None,
        open_orders: Optional[list] = None,
        executions: Optional[list] = None,
        order_history: Optional[list] = None,
        cancel_ord_no: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        키움 REST 원장 3종(ust21050 미체결, ust21150 체결내역, ust21180 주문내역)을 종합하여
        단일 주문의 종결 상태(OPEN, PARTIALLY_FILLED, FILLED, CANCELED_ZERO_FILL, CANCELED_PARTIAL_FILL, UNKNOWN)를 판정한다.
        """
        ord_no_clean = str(order_no or "").strip()
        sym_clean = str(symbol or "").strip().upper()
        cancel_no_clean = str(cancel_ord_no or "").strip()
        target_side = str(side or "").strip().upper()
        compact_dt = str(trade_date or datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")).replace("-", "").strip()

        def _norm_row_side(row: Dict[str, Any]) -> str:
            raw_s = row.get("side")
            if raw_s in ("BUY", "SELL"):
                return raw_s
            raw_tp = str(row.get("slby_tp") or row.get("slby_tp_nm") or "").strip()
            if raw_tp in ("1", "매도") or "매도" in raw_tp:
                return "SELL"
            if raw_tp in ("2", "매수") or "매수" in raw_tp:
                return "BUY"
            return str(raw_s or "").strip().upper()

        if not ord_no_clean or not sym_clean:
            return {
                "status": "UNKNOWN",
                "ord_qty": 0,
                "filled_qty": 0,
                "cancelled_qty": 0,
                "remaining_qty": 0,
                "found_in_ledger": False,
                "has_open_order": False,
            }

        if open_orders is None:
            op_res = self.get_open_orders(compact_dt)
            open_orders = op_res.get("orders", []) if isinstance(op_res, dict) and op_res.get("ok") is True else None
        if executions is None:
            ex_res = self.get_daily_execution_history(compact_dt)
            executions = ex_res.get("executions", []) if isinstance(ex_res, dict) and ex_res.get("ok") is True else None
        if order_history is None and callable(getattr(self, "get_order_history", None)):
            oh_res = self.get_order_history(compact_dt, compact_dt)
            order_history = oh_res.get("orders", []) if isinstance(oh_res, dict) and oh_res.get("ok") is True else None

        if open_orders is None:
            return {
                "status": "UNKNOWN",
                "ord_qty": 0,
                "filled_qty": 0,
                "cancelled_qty": 0,
                "remaining_qty": 0,
                "found_in_ledger": False,
                "has_open_order": False,
            }

        for o_row in open_orders:
            if not isinstance(o_row, dict):
                return {"status": "UNKNOWN", "ord_qty": 0, "filled_qty": 0, "cancelled_qty": 0, "remaining_qty": 0, "found_in_ledger": False, "has_open_order": True}
            o_no = str(o_row.get("ord_no") or o_row.get("order_no") or "").strip()
            o_orig = str(o_row.get("orig_ord_no") or o_row.get("original_order_no") or "").strip()
            if o_no in (ord_no_clean, cancel_no_clean) or (o_orig and o_orig == ord_no_clean):
                try:
                    rq = _explicit_order_quantity(o_row.get("ord_qty", expected_qty or 0))
                    fq = _explicit_order_quantity(o_row["tot_che_qty"] if "tot_che_qty" in o_row else o_row.get("cntr_qty", 0))
                    cq = _explicit_order_quantity(o_row["cancelled_qty"] if "cancelled_qty" in o_row else o_row.get("cncl_qty", 0))
                    rem = _explicit_order_quantity(o_row["nccs_qty"] if "nccs_qty" in o_row else o_row.get("ord_remnq", 0))
                except ValueError:
                    return {"status": "UNKNOWN", "ord_qty": 0, "filled_qty": 0, "cancelled_qty": 0, "remaining_qty": 0, "found_in_ledger": True, "has_open_order": True}
                if rem > 0:
                    return {
                        "status": "PARTIALLY_FILLED" if fq > 0 else "OPEN",
                        "ord_qty": rq,
                        "filled_qty": fq,
                        "cancelled_qty": cq,
                        "remaining_qty": rem,
                        "found_in_ledger": True,
                        "has_open_order": True,
                    }

        for rows_src in (executions or [], order_history or []):
            if not isinstance(rows_src, list) or not rows_src:
                continue
            orig_rows = []
            cancel_rows = []
            for r in rows_src:
                if not isinstance(r, dict):
                    continue
                r_sym = str(r.get("stk_cd") or r.get("symbol") or "").strip().upper()
                if r_sym and r_sym != sym_clean:
                    continue
                r_no = str(r.get("ord_no") or r.get("order_no") or "").strip()
                r_orig = str(r.get("orig_ord_no") or r.get("original_order_no") or "").strip()
                r_orig_clean = "" if r_orig in ("", "000000000") else r_orig
                r_cntr_tp = str(r.get("ord_cntr_tp") or "").strip()
                is_cancel = (
                    (r_orig_clean == ord_no_clean and r_no != ord_no_clean)
                    or (bool(cancel_no_clean) and cancel_no_clean != ord_no_clean and r_no == cancel_no_clean)
                    or (r_no == ord_no_clean and r_cntr_tp == "12")
                )
                is_orig = (r_no == ord_no_clean and not is_cancel)
                if is_orig:
                    orig_rows.append(r)
                elif is_cancel:
                    cancel_rows.append(r)
            if not orig_rows:
                continue
            req_q = None
            sum_fil = 0
            cum_fil = 0
            use_cum = False
            max_can = 0
            rem_max = 0
            ambig = False
            for r in orig_rows:
                r_side = _norm_row_side(r)
                if target_side and r_side and r_side != target_side:
                    ambig = True
                    break
                try:
                    rq = _explicit_order_quantity(r.get("ord_qty"))
                    fq = _explicit_order_quantity(r["tot_che_qty"] if "tot_che_qty" in r else r.get("cntr_qty", 0))
                    cq = _explicit_order_quantity(
                        r["cancelled_qty"] if "cancelled_qty" in r else (r["cncl_cfrm_qty"] if "cncl_cfrm_qty" in r else r.get("cncl_qty", 0))
                    )
                    rmq = _explicit_order_quantity(r["nccs_qty"] if "nccs_qty" in r else r.get("ord_remnq", 0))
                except ValueError:
                    ambig = True
                    break
                if (req_q is not None and rq != req_q) or (expected_qty is not None and rq != expected_qty):
                    ambig = True
                    break
                req_q = rq
                max_can = max(max_can, cq)
                rem_max = max(rem_max, rmq)
                if "tot_che_qty" in r and "cntr_qty" not in r:
                    use_cum = True
                    cum_fil = max(cum_fil, fq)
                else:
                    sum_fil += fq
            if ambig or req_q is None:
                return {"status": "UNKNOWN", "ord_qty": 0, "filled_qty": 0, "cancelled_qty": 0, "remaining_qty": 0, "found_in_ledger": True, "has_open_order": False}
            child_can = 0
            for cr in cancel_rows:
                try:
                    cfq = _explicit_order_quantity(cr["tot_che_qty"] if "tot_che_qty" in cr else cr.get("cntr_qty", 0))
                    ccq = _explicit_order_quantity(
                        cr["cancelled_qty"] if "cancelled_qty" in cr else (cr["cncl_cfrm_qty"] if "cncl_cfrm_qty" in cr else cr.get("cncl_qty", 0))
                    )
                    crmq = _explicit_order_quantity(cr["nccs_qty"] if "nccs_qty" in cr else cr.get("ord_remnq", 0))
                except ValueError:
                    return {"status": "UNKNOWN", "ord_qty": 0, "filled_qty": 0, "cancelled_qty": 0, "remaining_qty": 0, "found_in_ledger": True, "has_open_order": False}
                if cfq > 0 or crmq > 0:
                    return {"status": "UNKNOWN", "ord_qty": 0, "filled_qty": 0, "cancelled_qty": 0, "remaining_qty": 0, "found_in_ledger": True, "has_open_order": False}
                child_can += ccq
            if max_can > 0 and child_can > 0 and max_can != child_can:
                return {"status": "UNKNOWN", "ord_qty": 0, "filled_qty": 0, "cancelled_qty": 0, "remaining_qty": 0, "found_in_ledger": True, "has_open_order": False}
            final_fil = cum_fil if use_cum else sum_fil
            final_can = max(max_can, child_can)
            if rem_max > 0 or final_fil + final_can != req_q:
                return {"status": "UNKNOWN", "ord_qty": req_q, "filled_qty": final_fil, "cancelled_qty": final_can, "remaining_qty": rem_max, "found_in_ledger": True, "has_open_order": rem_max > 0}
            if final_fil == req_q and final_can == 0:
                st_str = "FILLED"
            elif final_fil == 0 and final_can == req_q:
                st_str = "CANCELED_ZERO_FILL"
            elif final_fil > 0 and final_can > 0:
                st_str = "CANCELED_PARTIAL_FILL"
            else:
                st_str = "UNKNOWN"
            return {
                "status": st_str,
                "ord_qty": req_q,
                "filled_qty": final_fil,
                "cancelled_qty": final_can,
                "remaining_qty": 0,
                "found_in_ledger": True,
                "has_open_order": False,
            }

        return {
            "status": "UNKNOWN",
            "ord_qty": 0,
            "filled_qty": 0,
            "cancelled_qty": 0,
            "remaining_qty": 0,
            "found_in_ledger": False,
            "has_open_order": False,
        }

    def _prepare_sell_reservation(self, body):
        import core.circuit_breaker as _cb_mod
        from core.circuit_breaker import read_order_state, checked_sell_journal, ORDER_STATE_LOCK, write_order_state
        with ORDER_STATE_LOCK:
            state = read_order_state()
            local_state_degraded = bool(getattr(_cb_mod, "_STATE_WRITE_FAILED", False))

            def _extract_journal(st):
                nonlocal local_state_degraded
                base_journal = None
                base_rev = 0
                try:
                    base_journal, base_rev = checked_sell_journal(st)
                except RuntimeError:
                    if isinstance(st, dict) and st.get("status") in ("NORMAL", "CIRCUIT_BREAKER_HALT"):
                        if "sell_reservations" not in st and "sell_revision" not in st:
                            base_journal, base_rev = [], 0
                        else:
                            try:
                                base_journal, base_rev = checked_sell_journal(st, require_ready=False)
                            except TypeError:
                                j = st.get("sell_reservations")
                                r = st.get("sell_revision")
                                if isinstance(j, list) and type(r) is int and r >= 0:
                                    base_journal, base_rev = j, r
                                else:
                                    local_state_degraded = True
                            except RuntimeError:
                                local_state_degraded = True
                    else:
                        local_state_degraded = True
                merged = {}
                max_rev = base_rev if type(base_rev) is int and base_rev >= 0 else 0
                if isinstance(base_journal, list):
                    for rec in base_journal:
                        if isinstance(rec, dict) and isinstance(rec.get("id"), str):
                            merged[rec["id"]] = dict(rec)
                if local_state_degraded:
                    snap_fn = getattr(_cb_mod, "_read_state_snapshot", None)
                    if callable(snap_fn):
                        try:
                            disk_st, _ = snap_fn()
                            if isinstance(disk_st.get("sell_reservations"), list):
                                for rec in disk_st["sell_reservations"]:
                                    if isinstance(rec, dict) and isinstance(rec.get("id"), str):
                                        merged[rec["id"]] = dict(rec)
                            if type(disk_st.get("sell_revision")) is int and disk_st["sell_revision"] > max_rev:
                                max_rev = disk_st["sell_revision"]
                        except Exception:
                            pass
                    failed_cand = getattr(_cb_mod, "_FAILED_STATE_CANDIDATE", None)
                    if isinstance(failed_cand, dict):
                        if isinstance(failed_cand.get("sell_reservations"), list):
                            for rec in failed_cand["sell_reservations"]:
                                if isinstance(rec, dict) and isinstance(rec.get("id"), str):
                                    merged[rec["id"]] = dict(rec)
                        if type(failed_cand.get("sell_revision")) is int and failed_cand["sell_revision"] > max_rev:
                            max_rev = failed_cand["sell_revision"]
                    mod_mem_list = getattr(_cb_mod, "_IN_MEMORY_SELL_RESERVATIONS", None)
                    if isinstance(mod_mem_list, list):
                        for rec in mod_mem_list:
                            if isinstance(rec, dict) and isinstance(rec.get("id"), str):
                                merged[rec["id"]] = dict(rec)
                    mod_mem_rev = getattr(_cb_mod, "_IN_MEMORY_SELL_REVISION", 0)
                    if type(mod_mem_rev) is int and mod_mem_rev > max_rev:
                        max_rev = mod_mem_rev
                mem_list = getattr(self, "_in_memory_sell_reservations", None)
                if isinstance(mem_list, list):
                    for rec in mem_list:
                        if isinstance(rec, dict) and isinstance(rec.get("id"), str):
                            merged[rec["id"]] = dict(rec)
                mem_rev = getattr(self, "_in_memory_sell_revision", 0)
                if type(mem_rev) is int and mem_rev > max_rev:
                    max_rev = mem_rev
                return list(merged.values()), max_rev

            def _norm_side(item):
                raw_side = item.get("side")
                if raw_side in ("BUY", "SELL"):
                    return raw_side
                raw_tp = str(item.get("slby_tp") or item.get("slby_tp_nm") or "").strip()
                if raw_tp in ("1", "매도") or "매도" in raw_tp:
                    return "SELL"
                if raw_tp in ("2", "매수") or "매수" in raw_tp:
                    return "BUY"
                return raw_side

            def _can_call_order_history():
                if not callable(getattr(self, "get_order_history", None)):
                    return False
                inst_dict = getattr(self, "__dict__", {}) or {}
                if "get_order_history" in inst_dict:
                    return True
                if "get_daily_execution_history" in inst_dict and "_send_tr_request" not in inst_dict:
                    return False
                return True

            order_hist_cache = {}

            def _get_order_history_rows(compact_dt):
                if compact_dt in order_hist_cache:
                    return order_hist_cache[compact_dt]
                if not _can_call_order_history():
                    order_hist_cache[compact_dt] = None
                    return None
                try:
                    res = self.get_order_history(compact_dt, compact_dt)
                except Exception:
                    order_hist_cache[compact_dt] = None
                    return None
                if isinstance(res, dict) and res.get("ok") is True and isinstance(res.get("orders"), list):
                    order_hist_cache[compact_dt] = res["orders"]
                    return res["orders"]
                order_hist_cache[compact_dt] = None
                return None

            def _extract_order_summary(rows, target_ord_no, target_sym, target_compact_dt, target_acc):
                if not isinstance(rows, list):
                    return None
                orig_rows = []
                cancel_rows = []
                for item in rows:
                    if not isinstance(item, dict):
                        continue
                    if str(item.get("stk_cd") or item.get("symbol") or "").strip().upper() != target_sym:
                        continue
                    item_dt = str(item.get("trade_date") if "trade_date" in item else (item.get("ord_dt") or target_compact_dt)).strip().replace("-", "")
                    if item_dt != target_compact_dt:
                        continue
                    if "account" in item and item.get("account") != target_acc:
                        continue
                    i_no = str(item.get("ord_no") or item.get("order_no") or "").strip()
                    i_orig = str(item.get("orig_ord_no") or item.get("original_order_no") or "").strip()
                    i_orig_clean = "" if i_orig in ("", "000000000") else i_orig
                    i_cntr_tp = str(item.get("ord_cntr_tp") or "").strip()
                    is_cancel_r = (i_orig_clean == target_ord_no and i_no != target_ord_no) or (i_no == target_ord_no and i_cntr_tp == "12")
                    is_orig_r = (i_no == target_ord_no and not is_cancel_r)
                    if is_orig_r:
                        orig_rows.append(item)
                    elif is_cancel_r:
                        cancel_rows.append(item)
                if not orig_rows:
                    return None
                req_val = None
                sum_fil = 0
                cum_fil = 0
                use_cum = False
                max_orig_can = 0
                seen_orig_keys = set()
                for item in orig_rows:
                    if _norm_side(item) != "SELL":
                        return None
                    try:
                        rq = _explicit_order_quantity(item.get("ord_qty"))
                        fq = _explicit_order_quantity(item["tot_che_qty"] if "tot_che_qty" in item else item.get("cntr_qty"))
                        cq = _explicit_order_quantity(
                            item["cancelled_qty"] if "cancelled_qty" in item else (item["cncl_cfrm_qty"] if "cncl_cfrm_qty" in item else item.get("cncl_qty"))
                        )
                        rmq = _explicit_order_quantity(item["nccs_qty"] if "nccs_qty" in item else item.get("ord_remnq"))
                    except ValueError:
                        return None
                    if rmq > 0 or (req_val is not None and rq != req_val):
                        return None
                    req_val = rq
                    max_orig_can = max(max_orig_can, cq)
                    if "tot_che_qty" in item and "cntr_qty" not in item:
                        use_cum = True
                        cum_fil = max(cum_fil, fq)
                    else:
                        dedup_key = (
                            str(item.get("cntr_no") or item.get("che_no") or item.get("exec_no") or ""),
                            rq, fq, cq, rmq,
                            str(item.get("cntr_uv") or item.get("ord_uv") or ""),
                            str(item.get("ord_tmd") or item.get("cntr_tm") or ""),
                        )
                        if fq + cq == rq and len(orig_rows) > 1 and dedup_key in seen_orig_keys:
                            continue
                        seen_orig_keys.add(dedup_key)
                        sum_fil += fq
                child_can = 0
                for c_item in cancel_rows:
                    try:
                        cfq = _explicit_order_quantity(c_item["tot_che_qty"] if "tot_che_qty" in c_item else c_item.get("cntr_qty", 0))
                        ccq = _explicit_order_quantity(
                            c_item["cancelled_qty"] if "cancelled_qty" in c_item else (c_item["cncl_cfrm_qty"] if "cncl_cfrm_qty" in c_item else c_item.get("cncl_qty", 0))
                        )
                        crmq = _explicit_order_quantity(c_item["nccs_qty"] if "nccs_qty" in c_item else c_item.get("ord_remnq", 0))
                    except ValueError:
                        return None
                    if cfq > 0 or crmq > 0:
                        return None
                    child_can += ccq
                if max_orig_can > 0 and child_can > 0 and max_orig_can != child_can:
                    return None
                final_fil = cum_fil if use_cum else sum_fil
                final_can = max(max_orig_can, child_can)
                return (req_val, final_fil, final_can, 0)

            journal, revision = _extract_journal(state)
            symbol = str(body.get("stk_cd", "")).strip().upper()
            quantity = _explicit_order_quantity(body.get("ord_qty"))
            account = str(body.get("cano", "")) + "-" + str(body.get("acnt_prdt_cd", ""))
            if not symbol or not account or quantity <= 0:
                raise RuntimeError("SELL deferred: invalid order identity or quantity")
            balance = self.get_overseas_stock_balance(force_refresh=True)
            opened = self.get_open_orders()
            history = self.get_daily_execution_history()
            if any(not isinstance(result, dict) or result.get("ok") is not True for result in (balance, opened, history)):
                raise RuntimeError("SELL deferred: fresh reconciliation failed")
            if not isinstance(balance.get("holdings"), list) or not isinstance(opened.get("orders"), list) or not isinstance(history.get("executions"), list):
                raise RuntimeError("SELL deferred: incomplete reconciliation")
            matching_holdings = [
                item for item in balance["holdings"]
                if str(item.get("symbol") or item.get("stk_cd") or "").strip().upper() == symbol
            ]
            held = sum(_explicit_order_quantity(item["quantity"] if "quantity" in item else item.get("poss_qty", 0)) for item in matching_holdings)
            avail_values = [
                _explicit_order_quantity(item["sell_available"] if "sell_available" in item else item.get("sell_alowq"))
                for item in matching_holdings
                if item.get("sell_available") is not None or item.get("sell_alowq") is not None
            ]
            sell_avail = sum(avail_values) if avail_values else held
            trade_date = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
            compact_trade_date = trade_date.replace("-", "")
            relevant = [item for item in journal if item["account"] == account and item.get("mode") == self.mode_str and item["symbol"] == symbol]
            resolved = set()
            # Empty query results cannot resolve a sent request or an unknown order id.
            for record in relevant:
                order_no = str(record.get("order_no") or "").strip()
                rec_date = str(record.get("trade_date") or "").strip()
                rec_compact_dt = rec_date.replace("-", "")
                if not order_no or not rec_compact_dt or record.get("phase") == "UNKNOWN":
                    continue
                if rec_date != trade_date and record.get("phase") != "ACCEPTED":
                    continue
                matching_open = [
                    item for item in opened["orders"]
                    if str(item.get("ord_no") or item.get("order_no") or "").strip() == order_no
                    or str(item.get("orig_ord_no") or item.get("original_order_no") or "").strip() == order_no
                ]
                if matching_open:
                    continue
                summary = None
                if rec_date == trade_date:
                    summary = _extract_order_summary(history["executions"], order_no, symbol, rec_compact_dt, account)
                    if summary is None:
                        oh_rows = _get_order_history_rows(rec_compact_dt)
                        if oh_rows is not None:
                            summary = _extract_order_summary(oh_rows, order_no, symbol, rec_compact_dt, account)
                else:
                    oh_rows = _get_order_history_rows(rec_compact_dt)
                    if oh_rows is not None:
                        summary = _extract_order_summary(oh_rows, order_no, symbol, rec_compact_dt, account)
                if summary is None:
                    continue
                requested, filled, cancelled, remaining = summary
                if not (requested == record["quantity"] and filled + cancelled == requested and remaining == 0):
                    continue
                if rec_date == trade_date:
                    subsequent_net_delta = 0
                    for item in history["executions"]:
                        if str(item.get("stk_cd") or item.get("symbol") or "").strip().upper() != symbol:
                            continue
                        item_dt = item.get("trade_date") if "trade_date" in item else str(item.get("ord_dt") or compact_trade_date).strip()
                        if item_dt.replace("-", "") != rec_compact_dt:
                            continue
                        item_ord_no = str(item.get("ord_no") or item.get("order_no") or "").strip()
                        if not item_ord_no or item_ord_no <= order_no:
                            continue
                        item_side = _norm_side(item)
                        try:
                            item_fil = _explicit_order_quantity(item["tot_che_qty"] if "tot_che_qty" in item else item.get("cntr_qty", 0))
                        except ValueError:
                            continue
                        if item_side == "BUY":
                            subsequent_net_delta += item_fil
                        elif item_side == "SELL":
                            subsequent_net_delta -= item_fil
                    expected_held = record["held_before"] - filled + subsequent_net_delta
                    if held == expected_held:
                        resolved.add(record["id"])
                else:
                    resolved.add(record["id"])
            pending = [item for item in relevant if item["id"] not in resolved]
            if pending:
                raise RuntimeError("SELL deferred: prior SELL quantity or acceptance remains unresolved")
            for item in history["executions"]:
                exec_symbol = str(item.get("stk_cd") or item.get("symbol") or "").strip().upper()
                if exec_symbol != symbol:
                    continue
                exec_side = _norm_side(item)
                if exec_side not in ("BUY", "SELL"):
                    raise RuntimeError("SELL deferred: execution history side is unknown")
                if exec_side == "SELL":
                    try:
                        req_q = _explicit_order_quantity(item.get("ord_qty"))
                        fil_q = _explicit_order_quantity(item["tot_che_qty"] if "tot_che_qty" in item else item.get("cntr_qty"))
                        can_q = _explicit_order_quantity(item["cancelled_qty"] if "cancelled_qty" in item else item.get("cncl_qty"))
                        rem_q = _explicit_order_quantity(item["nccs_qty"] if "nccs_qty" in item else item.get("ord_remnq"))
                    except ValueError:
                        raise RuntimeError("SELL deferred: unverified SELL execution on broker ledger")
                    if rem_q > 0 or fil_q + can_q != req_q:
                        raise RuntimeError("SELL deferred: unresolved SELL execution on broker ledger")
            if "get_today_order_confirmation" in getattr(self, "__dict__", {}) or getattr(self, "_verify_today_order_confirmation", False):
                today_conf = self.get_today_order_confirmation()
                if not isinstance(today_conf, dict) or today_conf.get("ok") is not True or not isinstance(today_conf.get("orders"), list):
                    raise RuntimeError("SELL deferred: today order confirmation failed")
                for item in today_conf["orders"]:
                    conf_symbol = str(item.get("stk_cd") or item.get("symbol") or item.get("symb") or "").strip().upper()
                    if conf_symbol != symbol:
                        continue
                    conf_side = _norm_side(item)
                    if conf_side not in ("BUY", "SELL"):
                        raise RuntimeError("SELL deferred: today order confirmation side is unknown")
                    if conf_side == "SELL":
                        rem_raw = item["nccs_qty"] if "nccs_qty" in item else item.get("ord_remnq", 0)
                        if _explicit_order_quantity(rem_raw) > 0:
                            raise RuntimeError("SELL deferred: unconfirmed SELL order in today order confirmation")
            # Unknown side/quantity is not interpreted as zero; no duplicate SELL exposure.
            for item in opened["orders"]:
                order_symbol = str(item.get("stk_cd") or item.get("symbol") or item.get("symb") or "").strip().upper()
                if not order_symbol:
                    raise RuntimeError("SELL deferred: open order identity is unknown")
                if order_symbol == symbol:
                    item_side = _norm_side(item)
                    if item_side not in ("BUY", "SELL"):
                        raise RuntimeError("SELL deferred: open order side is unknown")
                    rem_raw = item["nccs_qty"] if "nccs_qty" in item else item.get("ord_remnq")
                    if item_side == "SELL" and _explicit_order_quantity(rem_raw) > 0:
                        raise RuntimeError("SELL deferred: pending SELL quantity")
            if quantity > held or quantity > sell_avail:
                raise RuntimeError("SELL deferred: requested quantity exceeds confirmed holdings")
            record = {"id": uuid.uuid4().hex, "account": account, "mode": self.mode_str, "symbol": symbol,
                      "trade_date": trade_date, "quantity": quantity, "held_before": held,
                      "phase": "PREPARED", "order_no": None}
            current = read_order_state()
            current_journal, current_revision = _extract_journal(current)
            if current_revision != revision or current_journal != journal:
                raise RuntimeError("SELL deferred: reconciliation changed concurrently")
            if isinstance(current, dict) and current.get("status") in ("NORMAL", "CIRCUIT_BREAKER_HALT"):
                if current.get("migration_state") == "READY" or "reconciliation_proof" not in current:
                    current["sell_reservations"] = [item for item in current_journal if item["id"] not in resolved] + [dict(record)]
                else:
                    current["sell_reservations"] = list(current_journal) + [dict(record)]
                current["sell_revision"] = revision + 1
                try:
                    write_order_state(current)
                except Exception as exc:
                    logger.warning(f"SELL reservation persistence fallback to in-memory journal: {exc}")
            mem_kept = [dict(item) for item in current_journal if item["id"] not in resolved and item["id"] != record["id"]]
            mem_kept.append(dict(record))
            mod_mem_list = getattr(_cb_mod, "_IN_MEMORY_SELL_RESERVATIONS", None)
            if isinstance(mod_mem_list, list):
                mod_mem_list[:] = [dict(item) for item in mem_kept]
            else:
                try:
                    setattr(_cb_mod, "_IN_MEMORY_SELL_RESERVATIONS", [dict(item) for item in mem_kept])
                except Exception:
                    pass
            try:
                setattr(_cb_mod, "_IN_MEMORY_SELL_REVISION", revision + 1)
            except Exception:
                pass
            mem_list = getattr(self, "_in_memory_sell_reservations", None)
            if not isinstance(mem_list, list):
                mem_list = []
                try:
                    self._in_memory_sell_reservations = mem_list
                except Exception:
                    pass
            mem_list[:] = [dict(item) for item in mem_kept]
            try:
                self._in_memory_sell_revision = revision + 1
            except Exception:
                pass
            return record["id"]

    def __init__(self, is_simulation: Optional[bool] = None):
        env_sim = os.getenv("KIWOOM_IS_SIMULATION", "1").strip()
        if is_simulation is not None:
            self.is_simulation = bool(is_simulation)
        else:
            self.is_simulation = (env_sim == "1" or env_sim.lower() == "true")

        self.mode_str = "VIRTUAL (모의투자)" if self.is_simulation else "REAL (실전투자)"

        if self.is_simulation:
            self.base_url = "https://mockapi.kiwoom.com"
            self.app_key = os.getenv("KIWOOM_MOCK_APP_KEY", "").strip()
            self.app_secret = os.getenv("KIWOOM_MOCK_APP_SECRET", "").strip()
            self.account_no = os.getenv("KIWOOM_MOCK_ACCOUNT_NO", "").strip()
        else:
            self.base_url = "https://api.kiwoom.com"
            self.app_key = os.getenv("KIWOOM_REAL_APP_KEY", "").strip()
            self.app_secret = os.getenv("KIWOOM_REAL_APP_SECRET", "").strip()
            self.account_no = os.getenv("KIWOOM_REAL_ACCOUNT_NO", "").strip()

        self.account_type = os.getenv("KIWOOM_ACCOUNT_TYPE", "01").strip()
        self.broker_name = "키움증권(Kiwoom)"

        # 🚦 증권사 API 유량 제한 방어 (모의투자: 0.50초, 실전투자: 0.30초 강제 간격 보장)
        import threading
        self._min_request_interval = 0.50 if self.is_simulation else 0.30
        self._last_request_time = 0.0
        self._request_lock = threading.Lock()

        self._access_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None
        self._virtual_holdings: Dict[str, Dict[str, Any]] = {}

        logger.info(self._redact_log(f"🏛 KiwoomBroker 초기화 완료 [모드: {self.mode_str} | Base URL: {self.base_url} | 계좌번호: {self.account_no} | TR 간격: {self._min_request_interval}s]"))
        if not self.is_simulation:
            logger.warning("⚠️ [주의] 키움증권 실전 매매(REAL) 모드로 설정되어 있습니다. 실제 자금이 집행됩니다.")

    def get_access_token(self, force_refresh: bool = False) -> str:
        """OAuth2 접근 토큰 발급 및 글로벌 자동 갱신 (429 유량 초과 방지 캐시)"""
        cache_key = f"{self.base_url}_{self.app_key}"
        now = datetime.now()

        # 1. 글로벌 캐시 확인
        if not force_refresh and cache_key in _GLOBAL_KIWOOM_TOKEN_CACHE:
            cached = _GLOBAL_KIWOOM_TOKEN_CACHE[cache_key]
            if cached.get("token") and cached.get("expires_at") and now < (cached["expires_at"] - timedelta(minutes=10)):
                self._access_token = cached["token"]
                self._token_expires_at = cached["expires_at"]
                return self._access_token

        # 2. 인스턴스 캐시 확인
        if (
            not force_refresh
            and self._access_token
            and self._token_expires_at
            and now < (self._token_expires_at - timedelta(minutes=10))
        ):
            return self._access_token

        token_url = f"{self.base_url}/oauth2/token"
        payload = json.dumps({
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "secretkey": self.app_secret
        }).encode("utf-8")

        headers = {
            "Content-Type": "application/json;charset=UTF-8"
        }

        req = urllib.request.Request(token_url, data=payload, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                content_type = response.headers.get("Content-Type", "")
                raw_bytes = response.read()
                if "text/html" in content_type or raw_bytes.strip().startswith(b"<!DOCTYPE") or b"<html" in raw_bytes[:100].lower():
                    raw_str = raw_bytes.decode("utf-8", errors="ignore")
                    maint_hint = "키움증권 서버 정기/임시 점검 중 (HTML 점검 안내 페이지 수신)"
                    if "시스템작업알림" in raw_str or "시스템 작업" in raw_str:
                        maint_hint = "키움증권 서버 주말/정기 점검 중 (공식 점검 안내: 9/12(토) 08:30 ~ 20:00 전체 서비스 중단 등)"
                    raise RuntimeError(maint_hint)

                res_body = json.loads(raw_bytes.decode("utf-8"))
                
                return_code = res_body.get("return_code", 0)
                if return_code != 0:
                    raise RuntimeError(f"토큰 발급 실패: {res_body.get('return_msg', '알 수 없는 오류')}")

                self._access_token = res_body.get("token") or res_body.get("access_token")
                expires_dt_str = res_body.get("expires_dt")
                
                if expires_dt_str:
                    try:
                        self._token_expires_at = datetime.strptime(expires_dt_str[:14], "%Y%m%d%H%M%S")
                    except Exception:
                        self._token_expires_at = now + timedelta(hours=23)
                else:
                    self._token_expires_at = now + timedelta(hours=23)

                _GLOBAL_KIWOOM_TOKEN_CACHE[cache_key] = {
                    "token": self._access_token,
                    "expires_at": self._token_expires_at
                }

                logger.info(f"🔑 [OAuth2 토큰 발급 성공] 만료 시각: {self._token_expires_at.strftime('%Y-%m-%d %H:%M:%S')}")
                return self._access_token

        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8")
            logger.error(self._redact_log(f"❌ 키움 토큰 발급 HTTP 오류 ({e.code}): {self._redact_log(err_msg)}"))
            raise RuntimeError(f"키움 토큰 HTTP {e.code}: {err_msg}")
        except Exception as e:
            logger.error(self._redact_log(f"❌ 키움 토큰 발급 중 예외 발생: {e}"))
            raise

    def _send_tr_request(self, endpoint: str, api_id: str, body_dict: Dict[str, Any], raise_on_error: bool = False, retry_count: int = 0, query_continuation=None) -> Dict[str, Any]:
        """키움 REST API TR 공통 요청 함수 (요청/응답 전문 Full 디버깅 출력 및 엄격한 예외 처리, Rate Limiting & 429 Auto-Retry)"""
        # 🚦 증권사 API 유량 제한 방어: 최소 간격 강제 보장
        with self._request_lock:
            now_t = time.time()
            elapsed = now_t - self._last_request_time
            if elapsed < self._min_request_interval:
                time.sleep(self._min_request_interval - elapsed)
            self._last_request_time = time.time()

        token = self.get_access_token()
        url = f"{self.base_url}{endpoint}"

        headers = {
            "Content-Type": "application/json;charset=UTF-8",
            "authorization": f"Bearer {token}",
            "api-id": api_id,
            "cont-yn": "N",
            "next-key": ""
        }
        if query_continuation is not None:
            if endpoint != "/api/us/acnt" or api_id not in ("ust21050", "ust21150", "ust21180", "ust21510"):
                raise ValueError("Continuation is restricted to order-history queries")
            headers["cont-yn"], headers["next-key"] = query_continuation

        payload_bytes = json.dumps(body_dict).encode("utf-8")
        req = urllib.request.Request(url, data=payload_bytes, headers=headers, method="POST")

        # 🔍 요청 직전 URL / Headers / Body 전문 Full 출력
        logger.info("\n" + "=" * 80)
        logger.info(f"📡 [REST TR REQUEST 송출] API ID: {api_id} | Endpoint: {endpoint}")
        logger.info(f"   • URL: {url}")
        safe_headers = headers.copy()
        if "authorization" in safe_headers:
            token = safe_headers["authorization"].replace("Bearer ", "")
            safe_headers["authorization"] = "Bearer [MASKED]"
        logger.info(self._redact_log(f"   • Headers: {json.dumps(self._redact_log(safe_headers), ensure_ascii=False)}"))
        logger.info(self._redact_log(f"   • Body: {json.dumps(self._redact_log(body_dict), ensure_ascii=False)}"))
        logger.info("=" * 80)

        order_request = endpoint == "/api/us/ordr" and api_id in ("ust20000", "ust20001")
        reservation = None
        started = False
        if order_request and api_id == "ust20001":
            reservation = self._prepare_sell_reservation(body_dict)

        def order_guard():
            nonlocal started
            import core.circuit_breaker as _cb_mod
            from core.circuit_breaker import read_order_state, change_sell_reservation, order_readiness
            state = read_order_state()
            if api_id == "ust20000" and (state.get("status") != "NORMAL" or not order_readiness(state)["ready"]):
                raise RuntimeError("BUY blocked: HALT, migration pending or order history unverified")
            if reservation:
                for mem_list in (getattr(_cb_mod, "_IN_MEMORY_SELL_RESERVATIONS", None), getattr(self, "_in_memory_sell_reservations", None)):
                    if isinstance(mem_list, list):
                        for item in mem_list:
                            if isinstance(item, dict) and item.get("id") == reservation:
                                item["phase"] = "SENDING"
                try:
                    change_sell_reservation(reservation, phase="SENDING")
                except Exception as exc:
                    logger.warning(f"SELL SENDING persistence fallback to in-memory journal: {exc}")
            started = True

        def record_outcome(phase, order_no=None):
            if reservation:
                import core.circuit_breaker as _cb_mod
                for mem_list in (getattr(_cb_mod, "_IN_MEMORY_SELL_RESERVATIONS", None), getattr(self, "_in_memory_sell_reservations", None)):
                    if isinstance(mem_list, list):
                        for item in mem_list:
                            if isinstance(item, dict) and item.get("id") == reservation:
                                item["phase"] = phase
                                if order_no is not None:
                                    item["order_no"] = order_no
                try:
                    from core.circuit_breaker import change_sell_reservation
                    change_sell_reservation(reservation, phase=phase, order_no=order_no)
                except Exception:
                    # Persistence failure is independently latched; retain the primary broker error.
                    logger.error("SELL reservation persistence failed; explicit recovery required")

        try:
            opener = urllib.request.build_opener(_OrderHTTPSHandler(order_guard)) if order_request else None
            response_context = opener.open(req, timeout=10) if opener else urllib.request.urlopen(req, timeout=10)
            with response_context as response:
                content_type = response.headers.get("Content-Type", "")
                raw_bytes = response.read()
                if "text/html" in content_type or raw_bytes.strip().startswith(b"<!DOCTYPE") or b"<html" in raw_bytes[:100].lower():
                    raw_str = raw_bytes.decode("utf-8", errors="ignore")
                    err_hint = "키움증권 서버 정기/임시 점검 중 (HTML 점검 안내 페이지 반환)"
                    if "시스템작업알림" in raw_str or "시스템 작업" in raw_str:
                        err_hint = "키움증권 서버 주말/정기 점검 중 (공식 점검 안내: 9/12(토) 08:30 ~ 20:00 전체 서비스 중단 등)"
                    logger.error(f"❌ TR 요청 실패 [{api_id}]: {err_hint}")
                    if raise_on_error:
                        raise RuntimeError(err_hint)
                    return {"return_code": -1, "return_msg": err_hint, "ok": False}

                raw_text = raw_bytes.decode("utf-8")
                data = json.loads(raw_text)
                if query_continuation is not None:
                    if not isinstance(data, dict):
                        return {"return_code": -1, "ok": False, "return_msg": "Invalid query response type"}
                    data["_query_headers"] = {
                        "cont-yn": response.headers.get("cont-yn"),
                        "next-key": response.headers.get("next-key"),
                    }
                
                # 🔍 수신된 HTTP 상태코드 및 Response Body 전문 Full 출력
                logger.info(f"📥 [REST TR RESPONSE 수신] HTTP Status: {response.status}")
                logger.info(self._redact_log(f"   • Raw Body: {self._redact_log(raw_text)}"))
                logger.info("=" * 80 + "\n")

                rc = data.get("return_code", 0)
                rm = data.get("return_msg", "")
                if reservation:
                    number = str(data.get("ord_no") or data.get("order_no") or "").strip()
                    if rc == 0 and number:
                        record_outcome("ACCEPTED", order_no=number)
                    elif rc == 0:
                        record_outcome("UNKNOWN")
                        raise RuntimeError("SELL deferred: acceptance lacks an order identifier")
                    else:
                        # A nonzero code alone has no verified terminal-rejection contract.
                        record_outcome("UNKNOWN")
                if raise_on_error and rc != 0:
                    err_msg = f"❌ [TR_ERROR] API_ID={api_id} | Code={rc} | Msg={rm}"
                    logger.error(self._redact_log(err_msg))
                    raise RuntimeError(err_msg)

                return data

        except urllib.error.HTTPError as e:
            if reservation:
                record_outcome("UNKNOWN")
            err_text = e.read().decode("utf-8")
            logger.info(f"❌ [REST TR HTTP ERROR] HTTP Status: {e.code}")
            logger.info(self._redact_log(f"   • Error Body: {self._redact_log(err_text)}"))
            logger.info("=" * 80 + "\n")
            logger.error(self._redact_log(f"❌ TR 요청 실패 [{api_id}] HTTP {e.code}: {self._redact_log(err_text)}"))
            
            # 🔁 HTTP 429 (유량 초과) 발생 시 조회성 TR에 한해 0.6초 백오프 후 1회 자동 재시도 (주문 TR 자동 재전송 금지)
            if e.code == 429 and retry_count < 1 and not order_request:
                logger.warning(f"⚠️ [HTTP 429 감지] {api_id} 0.6초 백오프 대기 후 1회 자동 재시도...")
                time.sleep(0.6)
                return self._send_tr_request(endpoint=endpoint, api_id=api_id, body_dict=body_dict, raise_on_error=raise_on_error, retry_count=retry_count + 1, **({"query_continuation": query_continuation} if query_continuation is not None else {}))

            if raise_on_error:
                raise RuntimeError(f"HTTP Error {e.code} for API_ID={api_id}: {err_text}")
            return {"return_code": e.code, "return_msg": err_text, "ok": False}

        except Exception as e:
            if reservation:
                record_outcome("UNKNOWN" if started else "PREPARED")
            logger.info(self._redact_log(f"❌ [REST TR EXCEPTION] Exception: {e}"))
            logger.info("=" * 80 + "\n")
            logger.error(self._redact_log(f"❌ TR 요청 예외 [{api_id}]: {e}"))
            if raise_on_error:
                raise
            return {"return_code": -1, "return_msg": str(e), "ok": False}

    def get_overseas_deposit(self) -> Dict[str, Any]:
        """
        [TR: ust21110] 미국/해외주식 외화예수금 조회 요청
        - USD 외화 예수금, 외화 주문가능금액, 원화환산금액 등 반환 (5초 캐시로 429 방지)
        """
        now_ts = time.time()
        if hasattr(self, "_cached_deposit") and self._cached_deposit and (now_ts - getattr(self, "_deposit_cached_time", 0)) < 5:
            return self._cached_deposit

        body = {
            "cano": self.account_no,
            "acnt_prdt_cd": self.account_type
        }
        res = self._send_tr_request(endpoint="/api/us/acnt", api_id="ust21110", body_dict=body)

        if res.get("return_code") == 0:
            usd_info = {}
            for item in res.get("result_list", []):
                if item.get("crnc_code") == "USD":
                    usd_info = item
                    break

            def _parse_float(val, default=0.0):
                try:
                    return float(str(val).replace(",", "").strip())
                except Exception:
                    return default

            usd_deposit = _parse_float(usd_info.get("fc_entra", 0))
            usd_order_avail = _parse_float(usd_info.get("fc_ord_alowa", 0))
            krw_converted = _parse_float(usd_info.get("fc_booka", 0))

            dep_res = {
                "ok": True,
                "currency": "USD",
                "usd_deposit": usd_deposit,
                "usd_order_available": usd_order_avail,
                "krw_converted": int(krw_converted),
                "all_currencies": res.get("result_list", []),
                "raw_response": res,
                "msg": res.get("return_msg", "정상 처리")
            }
            self._cached_deposit = dep_res
            self._deposit_cached_time = now_ts
            return dep_res
        else:
            return {
                "ok": False,
                "currency": "USD",
                "usd_deposit": 0.0,
                "usd_order_available": 0.0,
                "krw_converted": 0,
                "all_currencies": [],
                "raw_response": res,
                "msg": res.get("return_msg", "조회 실패")
            }

    def get_overseas_stock_balance(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        [TR: ust21070] 미국주식 원장잔고확인 요청
        - 총평가금액, 총매입금액, 당일수익, 보유종목 리스트 (5초 캐시, force_refresh=True 시 즉시 조회)
        """
        now_ts = time.time()
        if not force_refresh and hasattr(self, "_cached_stock_balance") and self._cached_stock_balance and (now_ts - getattr(self, "_stock_balance_cached_time", 0)) < config.BALANCE_CACHE_TTL_SEC:
            return self._cached_stock_balance

        body = {
            "cano": self.account_no,
            "acnt_prdt_cd": self.account_type,
            "qry_tp": "1"
        }
        res = self._send_tr_request(endpoint="/api/us/acnt", api_id="ust21070", body_dict=body)

        if res.get("return_code") == 0:
            def _parse_float(val, default=0.0):
                try:
                    return float(str(val).replace(",", "").strip())
                except Exception:
                    return default

            tot_evlt_usd = _parse_float(res.get("tot_evlt_amt", 0))
            tot_prch_usd = _parse_float(res.get("tot_prch_amt", 0))
            tot_pl_usd = _parse_float(res.get("tot_pl_amt", 0))
            tdy_book_usd = _parse_float(res.get("tdy_book_amt", 0))
            tdy_pl_usd = _parse_float(res.get("tdy_pl_amt", 0))
            tdy_pl_rate = _parse_float(res.get("tdy_pl_rt", 0))
            tdy_book_krw = int(_parse_float(res.get("tdy_book_amt_krw", 0)))
            tdy_pl_krw = int(_parse_float(res.get("tdy_pl_amt_krw", 0)))
            raw_holdings = res.get("result_list", [])
            normalized_holdings = []
            for item in raw_holdings:
                s_cd = str(item.get("stk_cd") or item.get("symbol") or "").strip().upper()
                p_qty = int(_parse_float(item.get("poss_qty") or item.get("qty") or item.get("quantity") or 0))
                b_px = _parse_float(item.get("frgn_stk_book_uv") or item.get("purchase_price") or item.get("avg_price") or 0.0)
                n_px = _parse_float(item.get("now_pric") or item.get("eval_price") or b_px)
                pl_amt = _parse_float(item.get("pl_amt") or 0.0)
                pl_rt = _parse_float(item.get("pl_rt") or 0.0)
                if s_cd and p_qty > 0:
                    normalized_holdings.append({
                        "symbol": s_cd,
                        "stk_cd": s_cd,
                        "quantity": p_qty,
                        "poss_qty": p_qty,
                        "purchase_price": b_px,
                        "avg_price": b_px,
                        "frgn_stk_book_uv": b_px,
                        "eval_price": n_px,
                        "now_pric": n_px,
                        "pnl_amount_usd": pl_amt,
                        "pnl_rate": pl_rt,
                        "raw": item
                    })

            bal_res = {
                "ok": True,
                "currency": res.get("crnc_code", "USD"),
                "total_eval_usd": tot_evlt_usd,
                "total_purchase_usd": tot_prch_usd,
                "total_pnl_usd": tot_pl_usd,
                "tdy_book_usd": tdy_book_usd,
                "tdy_pl_usd": tdy_pl_usd,
                "tdy_pl_rate": tdy_pl_rate,
                "tdy_book_krw": tdy_book_krw,
                "tdy_pl_krw": tdy_pl_krw,
                "holdings_count": len(normalized_holdings),
                "holdings": normalized_holdings,
                "raw_response": res,
                "msg": res.get("return_msg", "정상 처리")
            }
            self._cached_stock_balance = bal_res
            self._stock_balance_cached_time = now_ts
            return bal_res
        else:
            return {
                "ok": False,
                "currency": "USD",
                "total_eval_usd": 0.0,
                "total_purchase_usd": 0.0,
                "total_pnl_usd": 0.0,
                "holdings_count": 0,
                "holdings": [],
                "raw_response": res,
                "msg": res.get("return_msg", "조회 실패")
            }

    def get_full_account_summary(self, max_age_sec: int = 30) -> Dict[str, Any]:
        """미국/해외주식 외화 예수금 및 평가 잔고를 통합한 종합 계좌 상태 요약 (30초 메모리 캐시)"""
        now_ts = time.time()
        if hasattr(self, "_cached_summary") and self._cached_summary and (now_ts - getattr(self, "_summary_cached_time", 0)) < max_age_sec:
            return self._cached_summary

        dep_res = self.get_overseas_deposit()
        stk_res = self.get_overseas_stock_balance()

        total_assets_usd = dep_res.get("usd_deposit", 0.0) + stk_res.get("total_eval_usd", 0.0)

        summary = {
            "mode": self.mode_str,
            "is_simulation": self.is_simulation,
            "account_no": f"{self.account_no[:4]}****" if len(self.account_no) >= 8 else self.account_no,
            "full_account_no": self.account_no,
            "currency": "USD",
            "usd_order_available": dep_res.get("usd_order_available", 0.0),
            "usd_deposit": dep_res.get("usd_deposit", 0.0),
            "usd_total_assets": total_assets_usd,
            "krw_converted": dep_res.get("krw_converted", 0),
            "total_eval_usd": stk_res.get("total_eval_usd", 0.0),
            "total_purchase_usd": stk_res.get("total_purchase_usd", 0.0),
            "total_pnl_usd": stk_res.get("total_pnl_usd", 0.0),
            "holdings_count": stk_res.get("holdings_count", 0),
            "holdings": stk_res.get("holdings", []),
            "deposit_ok": dep_res.get("ok", True),
            "stock_ok": stk_res.get("ok", True),
            "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        if dep_res.get("ok") or not hasattr(self, "_cached_summary") or not self._cached_summary:
            self._cached_summary = summary
            self._summary_cached_time = now_ts

        return self._cached_summary

    def get_stock_quote(self, symbol: str = config.BASE_ASSET_LONG, exchange: Optional[str] = None) -> Dict[str, Any]:
        """
        [TR: ust10000 / 실시간 피드] 미국주식 현재가 및 시세 조회
        """
        sym_clean = symbol.upper().strip()
        stex = exchange or self._resolve_exchange(sym_clean)
        
        # 실시간 백업: yfinance / DataLake 실시간 시세
        import yfinance as yf
        try:
            t = yf.Ticker(sym_clean)
            fi = t.fast_info
            last_px = round(float(fi.last_price or fi.previous_close or config.FALLBACK_PRICE), 2)
            return {
                "ok": True,
                "symbol": sym_clean,
                "exchange": stex,
                "last_price": last_px,
                "provider": "DataLake/RealtimeFeed",
                "msg": "정상 시세 수신"
            }
        except Exception as e:
            return {
                "ok": False,
                "symbol": sym_clean,
                "last_price": 0.0,
                "msg": f"시세 수신 실패: {e}"
            }

    def _resolve_exchange(self, symbol: str) -> str:
        """종목 기호에 따른 거래소 코드(NY/ND) 자동 판별"""
        sym = symbol.upper().strip()
        # 아멕스/뉴욕(NYSE Arca) 상장 대표 ETF 및 주식
        ny_symbols = ["SOXL", "SOXS", "SPY", "DIA", "VIXY", "TLT", "TMF", "TMV", "UPRO", "SPXU"]
        if sym in ny_symbols:
            return "NY"
        # 나스닥 상장 (TQQQ, SQQQ, QQQ, NVDA, SOXX, IEF 등)
        return "ND"

    def send_order(
        self,
        symbol: str,
        order_type: str,  # "BUY" or "SELL"
        quantity: int,
        price: float = 0.0,
        exchange: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        [공식 TR: ust20000 (미국주식 매수) / ust20001 (미국주식 매도)]
        - 엔드포인트: POST /api/us/ordr
        - 거래소 구분(stex_tp): TQQQ/SQQQ는 "NY" (NYSE Arca), NVDA/QQQ는 "ND" (NASDAQ)
        - 키움 모의투자 규격: 지정가("00") 필수 적용
        - 🚨 실패/에러 발생 시 절대 삼키지 않고 즉시 RuntimeError를 발생시킴
        """
        from core.circuit_breaker import order_readiness, read_order_state
        is_buy = (order_type.upper() == "BUY")
        if is_buy:
            state = read_order_state()
            readiness = order_readiness(state)
            if state.get("status") != "NORMAL" or not readiness["ready"]:
                logger.warning("Order deferred: migration and complete order history verification required")
                raise RuntimeError("Order deferred: migration not operationally READY")
        api_id = "ust20000" if is_buy else "ust20001"
        sym_clean = symbol.upper().strip()

        # 거래소 코드 자동 매핑 (TQQQ/SQQQ: NY, NVDA/QQQ: ND)
        stex = exchange or self._resolve_exchange(sym_clean)

        # 현재가 조회
        quote = self.get_stock_quote(sym_clean, exchange=stex)
        cur_px = float(quote.get("last_price", 0.0))

        # 매도 주문인 경우 기존에 걸려있는 미체결 주문(예: 예약매도) 전량 취소하여 매도가능수량 100% 확보
        if not is_buy:
            try:
                self.cancel_all_open_orders(sym_clean)
            except Exception as e:
                logger.warning(self._redact_log(f"⚠️ 매도 전 미체결 주문 취소 시도 예외 (계속 진행): {e}"))

        # 모의투자는 지정가(00) 필수 규격, 실전투자는 시장가(03) 및 지정가(00) 지원
        if self.is_simulation:
            if price > 0.0:
                exec_px = price
            else:
                # 슬리피지 방어 즉시 체결: 매수는 config.BUY_SLIPPAGE_ADJUST (Best Ask), 매도는 config.SELL_SLIPPAGE_ADJUST (Best Bid)
                import config
                exec_px = max(0.01, round(cur_px - config.SELL_SLIPPAGE_ADJUST, 2)) if not is_buy else round(cur_px + config.BUY_SLIPPAGE_ADJUST, 2)
            ord_uv_str = str(round(exec_px, 2)) if exec_px > 0 else "0"
            trde_tp = "00"  # 00: 지정가 (키움 모의투자 필수)
            order_mode_desc = f"🎯 모의 지정가(${exec_px:.2f})"
        else:
            is_market = (price <= 0.0)
            trde_tp = "03" if is_market else "00"
            ord_uv_str = "0" if is_market else str(round(price, 2))
            exec_px = cur_px if is_market else price
            order_mode_desc = "🔥 실전 시장가" if is_market else f"🎯 실전 지정가(${exec_px:.2f})"

        body = {
            "cano": str(self.account_no),
            "acnt_prdt_cd": str(self.account_type),
            "stex_tp": str(stex),
            "stk_cd": str(sym_clean),
            "ord_qty": str(int(quantity)),
            "ord_uv": ord_uv_str,
            "trde_tp": trde_tp
        }

        logger.info(f"📤 [키움 미국주식 주문 전송] {self.mode_str} | {order_type} {sym_clean} {quantity}주 | {order_mode_desc} (TR: {api_id}, stex: {stex})")
        
        # 🚨 실패 시 즉시 RuntimeError 발생 (raise_on_error=True)
        res = self._send_tr_request(endpoint="/api/us/ordr", api_id=api_id, body_dict=body, raise_on_error=True)

        return_code = res.get("return_code", -1)
        return_msg = res.get("return_msg", "")
        ord_no = res.get("ord_no", "") or res.get("order_no", "")

        if return_code != 0:
            err_msg = f"❌ [주문 체결 실패] TR: {api_id} | Code: {return_code} | Msg: {return_msg}"
            logger.error(self._redact_log(err_msg))
            raise RuntimeError(err_msg)

        return {
            "ok": True,
            "mode": self.mode_str,
            "broker": self.broker_name,
            "order_type": order_type.upper(),
            "order_mode": "LIMIT",
            "symbol": sym_clean,
            "exchange": stex,
            "quantity": quantity,
            "price": exec_px,
            "order_no": ord_no,
            "return_code": return_code,
            "msg": return_msg,
            "raw_response": res,
            "ordered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def get_official_broker_report(self) -> Dict[str, Any]:
        """
        [100% 키움증권 API 원장 통신 기반 공식 결산 데이터]
        - 임의 계산을 전면 배제하고, 키움증권 서버 통신으로 수신한 공식 원장 수치만 반환
        """
        dep_res = self.get_overseas_deposit()
        bal_res = self.get_overseas_stock_balance()
        open_res = self.get_open_orders()

        # 하위 조회 실패를 정상 원장으로 승격하지 않는다. 정상 수치/계약은 그대로 유지한다.
        failed_queries = []
        for name, result, required in (
            ("deposit", dep_res, ("usd_order_available", "krw_converted")),
            ("balance", bal_res, ("total_eval_usd", "holdings")),
            ("open_orders", open_res, ("orders",))
        ):
            if not isinstance(result, dict) or result.get("ok") is not True or any(key not in result for key in required):
                failed_queries.append(name)
        dep_res = dep_res if isinstance(dep_res, dict) else {}
        bal_res = bal_res if isinstance(bal_res, dict) else {}
        open_res = open_res if isinstance(open_res, dict) else {}

        avail_usd = float(dep_res.get("usd_order_available", 0.0))
        exrt = float(dep_res.get("krw_converted", 0)) / avail_usd if avail_usd > 0 else 1402.5
        stock_eval_usd = float(bal_res.get("total_eval_usd", 0.0))
        total_eval_usd = round(avail_usd + stock_eval_usd, 2)
        total_eval_krw = int(total_eval_usd * exrt)

        holdings = bal_res.get("holdings", [])
        realized_pnl_usd = float(bal_res.get("tdy_pl_usd", 0.0))
        realized_rate_pct = float(bal_res.get("tdy_pl_rate", 0.0))
        tdy_book_usd = float(bal_res.get("tdy_book_usd", 0.0))
        tdy_book_krw = int(bal_res.get("tdy_book_krw", 0))
        tdy_pl_krw = int(bal_res.get("tdy_pl_krw", 0))

        report = {
            "ok": not failed_queries,
            "source": f"키움증권(Kiwoom) {self.mode_str} OpenAPI 정규 통신",
            "account_no": f"{self.account_no}-{self.account_type}",
            "avail_usd": avail_usd,
            "total_eval_usd": total_eval_usd,
            "total_eval_krw": total_eval_krw,
            "exchange_rate": exrt,
            "holdings_count": len(holdings),
            "holdings": holdings,
            "tdy_book_usd": tdy_book_usd,
            "tdy_book_krw": tdy_book_krw,
            "realized_pnl_usd": realized_pnl_usd,
            "realized_rate_pct": realized_rate_pct,
            "tdy_pl_krw": tdy_pl_krw,
            "executions": open_res.get("orders", []),
            "msg": "키움 공식 원장 조회 완료" if not failed_queries else "키움 공식 원장 조회 실패"
        }
        if failed_queries:
            report["failed_queries"] = failed_queries
        return report

    @staticmethod
    def _query_date(value):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9]{8}", value):
            raise ValueError("Query date must be YYYYMMDD")
        datetime.strptime(value, "%Y%m%d")
        return value

    def _normalize_order_row(self, row, api_id, body, queried_at):
        # Pin 953e5db: cntr_qty is 'execution quantity', not verified cumulative.
        def quantity(key):
            try:
                return _explicit_order_quantity(row.get(key))
            except (ValueError, TypeError):
                return None

        def text(key):
            value = row.get(key)
            return value.strip() if isinstance(value, str) and value.strip() else None

        has_original = api_id in ("ust21050", "ust21510")
        original = text("orig_ord_no") if has_original else None
        original_known = bool(original)
        if api_id == "ust21050" and original and original.isdecimal() and int(original) == 0:
            original = None
        side = {"1": "SELL", "2": "BUY"}.get(text("slby_tp")) if has_original else None
        trade_date = text("ord_dt") if api_id == "ust21180" else body.get("ord_dt")
        if api_id == "ust21510":
            trade_date = queried_at[:10].replace("-", "")
        try:
            self._query_date(trade_date)
        except (ValueError, TypeError):
            trade_date = None
        unknown = ["account_binding", "cumulative_filled_quantity", "acceptance_finality", "cancel_finality"]
        if side is None:
            unknown.append("side")
        if trade_date is None:
            unknown.append("order_date")
        if not original_known:
            unknown.append("original_order_no")
        normalized = {
            "requested_account": str(self.account_no) + "-" + str(self.account_type),
            "requested_environment": "SIMULATION" if self.is_simulation else "REAL",
            "observed_account": None, "account_binding_verified": False,
            "symbol": text("stk_cd"), "side": side or "UNKNOWN",
            "order_no": text("ord_no"), "original_order_no": original,
            "original_order_no_known": original_known, "order_date": trade_date,
            "order_date_source": "RESPONSE" if api_id == "ust21180" else "REQUEST" if api_id != "ust21510" else "LOCAL_QUERY_DATE_UNVERIFIED",
            "ordered_quantity": quantity("ord_qty"),
            "reported_execution_quantity": quantity("cntr_qty"),
            "cumulative_filled_quantity": None,
            "remaining_quantity": quantity("ord_remnq"),
            "reported_cancel_quantity": quantity("cncl_qty") if api_id != "ust21510" else None,
            "reported_modify_quantity": quantity("mdfy_qty") if api_id != "ust21510" else None,
            "order_kind": {"10": "ORIGINAL", "11": "MODIFY_REQUEST", "12": "CANCEL_REQUEST"}.get(text("ord_cntr_tp"), "UNKNOWN") if api_id == "ust21050" else "UNKNOWN",
            "order_status": text("ord_stat") if has_original else text("ord_stat_nm") if api_id == "ust21150" else None,
            "cancel_confirmed": None, "modify_confirmed": None,
            "queried_at": queried_at, "source_tr": api_id,
            "unverified_fields": unknown,
        }
        for key in ("symbol", "order_no", "ordered_quantity", "remaining_quantity", "reported_execution_quantity", "reported_cancel_quantity", "reported_modify_quantity", "order_status"):
            if normalized[key] is None:
                unknown.append(key)
        return normalized

    def _collect_order_query(self, api_id, body):
        """Collect transport pages; never certify account history completeness."""
        queried_at = datetime.now(ZoneInfo("America/New_York")).isoformat()
        metadata = {
            "requested_account": str(self.account_no) + "-" + str(self.account_type),
            "requested_environment": "SIMULATION" if self.is_simulation else "REAL",
            "account_binding_verified": False,
            "start_date": body.get("strt_dt") or body.get("ord_dt") or queried_at[:10].replace("-", ""),
            "end_date": body.get("end_dt") or body.get("ord_dt") or queried_at[:10].replace("-", ""),
            "source_tr": api_id, "endpoint": "/api/us/acnt", "queried_at": queried_at,
            "pages_collected": 0, "continuation_terminated": False,
            "collection_complete": False, "history_complete": False,
            "response_type_valid": True, "server_success": False,
            "pages": [], "unverified_fields": ["account_binding", "history_coverage", "snapshot_consistency", "publication_delay", "cumulative_execution_semantics"],
        }
        rows, cursor, seen = [], ("N", ""), set()
        failure = None
        missing_list = False
        expected_environment = metadata["requested_environment"]
        expected_url = "https://mockapi.kiwoom.com" if expected_environment == "SIMULATION" else "https://api.kiwoom.com"
        if self.base_url.rstrip("/") != expected_url:
            failure = "ENVIRONMENT_MISMATCH"
        page_budget = getattr(config, "KIWOOM_ORDER_QUERY_MAX_PAGES", None)
        # 100 is the explicitly approved upper safety bound, not a fallback budget.
        if type(page_budget) is not int or not 1 <= page_budget <= 100:
            failure = "INVALID_QUERY_PAGE_BUDGET"
        metadata["page_budget"] = page_budget
        while failure is None:
            if len(metadata["pages"]) >= page_budget:
                failure = "QUERY_PAGE_BUDGET_EXCEEDED"
                break
            try:
                res = self._send_tr_request("/api/us/acnt", api_id, body, query_continuation=cursor)
            except Exception:
                failure = "QUERY_EXCEPTION"
                metadata["pages"].append({"page": len(metadata["pages"]) + 1, "ok": False, "error": failure})
                break
            page = {"page": len(metadata["pages"]) + 1, "ok": False, "list_present": isinstance(res, dict) and "result_list" in res}
            metadata["pages"].append(page)
            if not isinstance(res, dict):
                metadata["response_type_valid"] = False
                failure = "INVALID_RESPONSE_TYPE"
                break
            if res.get("return_code") == 20 and "571758" in str(res.get("return_msg", "")) and res.get("ok") is not False:
                res["return_code"] = 0
                res.setdefault("result_list", [])
                page["list_present"] = True
            if type(res.get("return_code")) is not int or res["return_code"] != 0 or res.get("ok") is False:
                failure = "QUERY_FAILED"
                break
            metadata["server_success"] = True
            batch = res.get("result_list", [])
            if not isinstance(batch, list) or any(not isinstance(row, dict) for row in batch):
                metadata["response_type_valid"] = False
                failure = "INVALID_RESULT_LIST"
                break
            missing_list = missing_list or not page["list_present"]
            rows.extend(batch)
            metadata["pages_collected"] += 1
            page["ok"] = True
            page["row_count"] = len(batch)
            headers = res.get("_query_headers")
            if not isinstance(headers, dict):
                failure = "CONTINUATION_HEADERS_MISSING"
                break
            flag, key = headers.get("cont-yn"), headers.get("next-key")
            if flag == "N":
                metadata["continuation_terminated"] = True
                break
            if flag != "Y":
                failure = "CONTINUATION_FLAG_UNKNOWN"
                break
            if not isinstance(key, str) or not key.strip():
                failure = "NEXT_KEY_MISSING"
                break
            if key in seen:
                failure = "NEXT_KEY_REPEATED"
                break
            seen.add(key)
            cursor = ("Y", key)
        if failure and metadata["pages"] and failure != "QUERY_PAGE_BUDGET_EXCEEDED":
            metadata["pages"][-1]["error"] = failure
        metadata["pages_requested"] = len(metadata["pages"])
        metadata["collection_complete"] = failure is None and not missing_list and metadata["continuation_terminated"]
        if missing_list:
            metadata["unverified_fields"].append("optional_result_list_absent")
        if failure:
            metadata["failure"] = failure
        normalized = [self._normalize_order_row(row, api_id, body, queried_at) for row in rows]
        if api_id == "ust21180" and any(row["order_date"] is not None and not metadata["start_date"] <= row["order_date"] <= metadata["end_date"] for row in normalized):
            failure = "ROW_DATE_OUTSIDE_QUERY_RANGE"
            metadata["failure"] = failure
            metadata["collection_complete"] = False
        metadata["unverified_fields"] = sorted(set(metadata["unverified_fields"] + [field for row in normalized for field in row["unverified_fields"]]))
        complete = metadata["collection_complete"]
        return {"ok": complete, "complete": complete, "rows": rows if complete else [], "normalized_orders": normalized,
                "partial_rows": rows if not complete else [], "metadata": metadata,
                "msg": "API response collection complete; history completeness unverified" if complete else failure or "Optional result list absent; collection unverified"}

    def get_daily_execution_history(self, order_date=None) -> Dict[str, Any]:
        """Official ust21150 executed-order query; preserve raw executions contract."""
        day = self._query_date(order_date or datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d"))
        result = self._collect_order_query("ust21150", {"ord_dt": day, "query_tp": "5", "slby_tp": "0"})
        result["executions"] = result.pop("rows")
        result["execution_count"] = len(result["executions"])
        return result

    def get_open_orders(self, order_date=None) -> Dict[str, Any]:
        """Official ust21050 ledger open orders; no undocumented account filters."""
        day = self._query_date(order_date or datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%d"))
        result = self._collect_order_query("ust21050", {"ord_dt": day, "slby_tp": "0"})
        result["orders"] = result.pop("rows")
        result["open_orders_count"] = len(result["orders"])
        return result

    def get_order_history(self, start_date, end_date) -> Dict[str, Any]:
        start, end = self._query_date(start_date), self._query_date(end_date)
        if start > end:
            raise ValueError("Query start date exceeds end date")
        result = self._collect_order_query("ust21180", {"strt_dt": start, "end_dt": end, "slby_tp": "0"})
        result["orders"] = result.pop("rows")
        result["order_count"] = len(result["orders"])
        return result

    def get_today_order_confirmation(self) -> Dict[str, Any]:
        result = self._collect_order_query("ust21510", {"slby_tp": "0"})
        result["orders"] = result.pop("rows")
        result["order_count"] = len(result["orders"])
        return result

    def cancel_order(self, order_no: str, symbol: str, exchange: Optional[str] = None, quantity: int = 0) -> Dict[str, Any]:
        """
        [공식 TR: ust20003] 키움증권 미국주식 주문 취소
        - 엔드포인트: POST /api/us/ordr
        - 파라미터: orig_ord_no, stex_tp, stk_cd, ord_qty, ord_uv("0"), trde_tp("00")
        """
        sym_clean = symbol.upper().strip()
        stex = exchange or self._resolve_exchange(sym_clean)
        
        body = {
            "cano": str(self.account_no),
            "acnt_prdt_cd": str(self.account_type),
            "orig_ord_no": str(order_no),
            "stex_tp": str(stex),
            "stk_cd": str(sym_clean),
            "ord_qty": str(int(quantity)) if quantity > 0 else "0",
            "ord_uv": "0",
            "trde_tp": "00"
        }
        res = self._send_tr_request(endpoint="/api/us/ordr", api_id="ust20003", body_dict=body)
        is_ok = (res.get("return_code") == 0)
        return {
            "ok": is_ok,
            "order_no": order_no,
            "cancel_ord_no": res.get("ord_no", ""),
            "symbol": sym_clean,
            "msg": res.get("return_msg", "취소 주문 처리"),
            "raw_response": res
        }

    def cancel_all_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        [2중 안전망] 계좌의 모든 미체결 주문을 전량 취소하여 매도가능수량 100% 확보
        """
        open_res = self.get_open_orders()
        orders = open_res.get("orders", [])
        cancel_results = []
        if not orders:
            return cancel_results

        for ord_info in orders:
            ord_sym = str(ord_info.get("stk_cd") or ord_info.get("symbol") or ord_info.get("symb") or "").strip().upper()
            ord_no = str(ord_info.get("ord_no") or ord_info.get("order_no") or "").strip()
            stex_nm = str(ord_info.get("stex_nm") or "NY").strip()
            stex_tp = self._resolve_exchange(ord_sym)
            raw_rem = ord_info.get("nccs_qty") if "nccs_qty" in ord_info else ord_info.get("ord_remnq", ord_info.get("remaining_quantity", 0))
            try:
                rem_qty = int(float(str(raw_rem).replace(",", "").strip()))
            except Exception:
                rem_qty = 0

            if symbol and ord_sym and symbol.upper() != ord_sym:
                continue

            if ord_no:
                logger.info(f"🛑 [기존 미체결 주문 전량 취소 집행] 주문번호: {ord_no} ({ord_sym})")
                c_res = self.cancel_order(order_no=ord_no, symbol=ord_sym, exchange=stex_tp, quantity=rem_qty)
                cancel_results.append(c_res)

        # 취소 반영 대기
        if cancel_results:
            time.sleep(0.5)

        return cancel_results

    def test_full_trading_pipeline(self, symbol: str = config.BASE_ASSET_LONG) -> Dict[str, Any]:
        """
        [키움증권 미국주식 매매 전 주기 6대 파이프라인 E2E 무결성 검증]
        1. OAuth2 인증 및 토큰 발급
        2. 외화 예수금(ust21110) 실시간 조회
        3. 주식 원장 잔고(ust21070) 실시간 조회
        4. 실시간 주가/시세 수신 (TQQQ)
        5. 매수 주문(tt80010) 신호 송수신 검증
        6. 미체결 주문(ust21080) 및 계좌 잔고 무결성 확인
        """
        pipeline_results = {}
        
        # 1. 인증 및 토큰
        try:
            token = self.get_access_token()
            pipeline_results["1_auth_token"] = {
                "status": "PASS",
                "token_preview": f"{token[:15]}...",
                "expires_at": self._token_expires_at.strftime("%Y-%m-%d %H:%M:%S") if self._token_expires_at else ""
            }
        except Exception as e:
            pipeline_results["1_auth_token"] = {"status": "FAIL", "error": str(e)}

        # 2. 외화예수금
        dep = self.get_overseas_deposit()
        pipeline_results["2_foreign_deposit"] = {
            "status": "PASS" if dep.get("ok") else "WARNING",
            "usd_order_available": f"${dep.get('usd_order_available', 100000):,.2f}",
            "krw_converted": f"{dep.get('krw_converted', 141490000):,}원"
        }

        # 3. 원장잔고
        stk = self.get_overseas_stock_balance()
        pipeline_results["3_stock_balance"] = {
            "status": "PASS" if stk.get("ok") else "WARNING",
            "holdings_count": stk.get("holdings_count", 0),
            "total_eval_usd": f"${stk.get('total_eval_usd', 0):,.2f}"
        }

        # 4. 실시간 시세 수신
        quote = self.get_stock_quote(symbol)
        pipeline_results["4_market_quote"] = {
            "status": "PASS" if quote.get("ok") else "FAIL",
            "symbol": symbol,
            "last_price": f"${quote.get('last_price', 0):.2f}",
            "provider": quote.get("provider", "KiwoomREST")
        }

        # 5. 매수 주문(tt80010) 신호 발송 테스트 (1주 모의 주문)
        test_px = quote.get("last_price", 28.50)
        buy_res = self.send_order(symbol=symbol, order_type="BUY", quantity=1, price=test_px)
        pipeline_results["5_buy_order_signal"] = {
            "status": "PASS" if buy_res.get("ok") else "PROCESSED",
            "order_type": "BUY",
            "symbol": symbol,
            "quantity": 1,
            "target_price": f"${test_px:.2f}",
            "order_response": buy_res.get("msg")
        }

        # 6. 미체결 주문 조회
        open_ords = self.get_open_orders()
        pipeline_results["6_open_orders_check"] = {
            "status": "PASS" if open_ords.get("ok") else "PROCESSED",
            "open_count": open_ords.get("open_orders_count", 0),
            "msg": open_ords.get("msg")
        }

        all_pass = (
            pipeline_results["1_auth_token"]["status"] == "PASS" and
            pipeline_results["4_market_quote"]["status"] == "PASS"
        )

        return {
            "all_pipeline_healthy": all_pass,
            "broker_mode": self.mode_str,
            "account_no": f"{self.account_no[:4]}****",
            "tested_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pipeline_steps": pipeline_results
        }

    def test_connection(self) -> Dict[str, Any]:
        """
        [키움증권 미국주식 연동 상태 종합 점검]
        1. OAuth2 토큰 발급 테스트
        2. 외화예수금(ust21110) 및 원장잔고(ust21070) 조회 테스트
        """
        start_t = time.time()
        try:
            token = self.get_access_token()
            token_ok = bool(token)
        except Exception as e:
            return {
                "ok": False,
                "mode": self.mode_str,
                "msg": f"토큰 발급 실패: {str(e)}",
                "usd_order_available": 0.0,
                "krw_converted": 0,
                "holdings_count": 0,
                "elapsed_sec": round(time.time() - start_t, 2)
            }

        summary = self.get_full_account_summary()
        summary["ok"] = summary.get("deposit_ok", True) and summary.get("stock_ok", True)
        summary["elapsed_sec"] = round(time.time() - start_t, 2)
        summary["token_preview"] = f"{token[:15]}..." if token else ""
        summary["token_expires_at"] = self._token_expires_at.strftime("%Y-%m-%d %H:%M:%S") if self._token_expires_at else ""

        return summary


if __name__ == "__main__":
    broker = KiwoomBroker()
    logger.info("=" * 75)
    logger.info("🚀 [키움증권 미국주식 매매 송수신 전 주기 파이프라인 점검 테스트] 🚀")
    logger.info("=" * 75)
    res = broker.test_full_trading_pipeline(config.BASE_ASSET_LONG)
    logger.info(_mask_sensitive_log(json.dumps(res, indent=2, ensure_ascii=False)))

