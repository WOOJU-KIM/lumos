import copy
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
import core.circuit_breaker as cb_mod
import core.kiwoom_broker as kb_mod
from core.kiwoom_broker import KiwoomBroker
from core.live_experience_logger import DEFAULT_CSV_PATH, DEFAULT_DB_PATH, LiveExperienceLogger
from core.live_runner import KiwoomLiveRunner

WATCHED_PROD_FILES = [
    PROJECT_ROOT / "data" / "market_data.db",
    PROJECT_ROOT / "data" / "live_experience.db",
    PROJECT_ROOT / "data" / "circuit_breaker_state.json",
    PROJECT_ROOT / "data" / "live_trades.csv",
    PROJECT_ROOT / "data" / "active_position.json",
    PROJECT_ROOT / "models" / "active_model_manifest.json",
    PROJECT_ROOT / "models" / "model_normalized_72.pkl",
    PROJECT_ROOT / "models" / "model_champion.pkl",
    PROJECT_ROOT / "models" / "m_moe_orchestrator.pkl",
]


def snapshot_prod_hashes():
    out = {}
    for p in WATCHED_PROD_FILES:
        rel = str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
        if not p.exists():
            out[rel] = ("MISSING", 0)
        else:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            out[rel] = (h.hexdigest(), p.stat().st_size)
    return out


class RealContractFakeKiwoomSetup:
    """
    Isolated sandbox wiring a real KiwoomLiveRunner + real KiwoomBroker
    against a realistic Kiwoom REST API ledger state (ust21070, ust21050, ust21150, ust21180, ust21510)
    with zero real network calls and zero writes to production data/* or models/*.
    """

    _cached_moe = None

    def __init__(self, tmp_dir: Path):
        self.tmp_dir = tmp_dir
        self.cb_file = tmp_dir / "circuit_breaker_state.json"
        self.pos_file = tmp_dir / "active_position.json"
        self.exp_db = tmp_dir / "live_experience.db"
        self.exp_csv = tmp_dir / "live_trades.csv"
        self.dedup_file = tmp_dir / "telegram_model_alert_state.json"
        self.ny_today = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        self.ny_compact = self.ny_today.replace("-", "")
        self.ny_yesterday = (datetime.now(ZoneInfo("America/New_York")) - timedelta(days=1)).strftime("%Y-%m-%d")
        self.ny_yesterday_compact = self.ny_yesterday.replace("-", "")
        self.telegram_messages = []

        self.ledger = {
            "holdings": {},  # sym -> {"qty": int, "sell_avail": int, "avg_px": float, "cur_px": float}
            "open_orders": [],        # ust21050 rows
            "executions": [],         # ust21150 rows (ONLY cntr_qty > 0 in real Kiwoom)
            "order_history": [],      # ust21180 rows (ALL orders including 0-fill cancels & cancel request rows)
            "today_confirmations": [],
            "balance_ok": True,
            "open_orders_ok": True,
            "executions_ok": True,
            "order_history_ok": True,
            "cancel_ok": True,
            "next_ord_seq": 58100,
            "sent_orders": [],
            "sent_cancels": [],
            "on_send_order_hook": None,
            "on_cancel_order_hook": None,
        }

    def write_cb_state(
        self,
        status="NORMAL",
        migration_state="READY",
        sell_reservations=None,
        sell_revision=0,
        account="56910000-01",
        env="REAL",
    ):
        cb_mod._IN_MEMORY_SELL_RESERVATIONS.clear()
        cb_mod._IN_MEMORY_SELL_REVISION = 0
        cb_mod._STATE_WRITE_FAILED = False
        cb_mod._FAILED_STATE_CANDIDATE = None
        digest = cb_mod._reconciliation_proof_digest(account, env, sell_revision, self.ny_compact, 0, 0)
        state = {
            "schema_version": 1,
            "state_revision": 1,
            "status": status,
            "migration_state": migration_state,
            "consecutive_losses": 0,
            "sell_revision": sell_revision,
            "sell_reservations": list(sell_reservations) if sell_reservations else [],
            "reconciliation_proof": {
                "verified_account": account,
                "verified_environment": env,
                "verified_through": self.ny_compact,
                "sell_revision": sell_revision,
                "open_orders_count": 0,
                "unresolved_sell_count": 0,
                "history_order_count": 0,
                "today_order_count": 0,
                "proof_digest": digest,
            },
        }
        self.cb_file.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        return state

    def build_runner_and_broker(self):
        if RealContractFakeKiwoomSetup._cached_moe is None:
            from core.moe_orchestrator import MoEMetaOrchestrator
            RealContractFakeKiwoomSetup._cached_moe = MoEMetaOrchestrator(
                confidence_threshold=config.GBDT_CONFIDENCE_THRESHOLD,
                gbdt_threshold=config.GBDT_CONFIDENCE_THRESHOLD,
            )

        with patch("core.circuit_breaker.CIRCUIT_STATE_FILE", self.cb_file), \
             patch("core.live_runner.MoEMetaOrchestrator", return_value=RealContractFakeKiwoomSetup._cached_moe):
            runner = KiwoomLiveRunner(is_simulation=False)

        broker = runner.broker
        broker.is_simulation = False
        broker.mode_str = "REAL (실전투자)"
        broker.base_url = "https://api.kiwoom.com"
        broker.account_no = "56910000"
        broker.account_type = "01"
        broker._min_request_interval = 0.0
        broker.get_access_token = lambda force_refresh=False: "FAKE_TOKEN"

        runner.pos_file_path = str(self.pos_file)
        runner._active_position = None
        runner._empty_balance_streak = 0
        runner._order_fills = {}
        runner._is_order_in_progress = False
        runner._last_wait_fill_stk_bal = None
        runner._unreconciled_holding_mismatch = None
        runner._unconfirmed_buy_order = None
        runner.order_timeout_buy_sec = 0.02
        runner.order_timeout_sell_sec = 0.02
        runner.experience_logger.db_path = self.exp_db
        runner.experience_logger.csv_path = self.exp_csv
        runner.experience_logger._init_db()
        runner.notifier.state_file_path = self.dedup_file
        runner.notifier.async_mode = False
        runner.notifier._send_http_request = lambda text: (self.telegram_messages.append(text), {"ok": True})[1]
        runner.dispatcher.send_telegram_message = lambda msg: (self.telegram_messages.append(msg), True)[1]

        def sync_dispatch_sell(**kwargs):
            if not runner._acquire_order_lock():
                return False
            kwargs["_already_locked"] = True
            return runner._execute_sell_with_10s_chase(**kwargs)

        runner._dispatch_sell = sync_dispatch_sell

        def fake_balance(force_refresh=False):
            if callable(self.ledger["balance_ok"]):
                return self.ledger["balance_ok"](force_refresh=force_refresh)
            if not self.ledger["balance_ok"]:
                return {"ok": False, "holdings": [], "holdings_count": 0, "msg": "Balance query failed"}
            h_list = []
            for sym, info in self.ledger["holdings"].items():
                q = int(info.get("qty", 0))
                if q > 0:
                    sa = int(info.get("sell_avail", q))
                    apx = float(info.get("avg_px", 100.0))
                    cpx = float(info.get("cur_px", apx))
                    h_list.append({
                        "symbol": sym,
                        "stk_cd": sym,
                        "quantity": q,
                        "poss_qty": q,
                        "sell_available": sa,
                        "sell_alowq": sa,
                        "purchase_price": apx,
                        "avg_price": apx,
                        "frgn_stk_book_uv": apx,
                        "eval_price": cpx,
                        "now_pric": cpx,
                    })
            return {"ok": True, "holdings_count": len(h_list), "holdings": h_list}

        def fake_open_orders(order_date=None):
            if not self.ledger["open_orders_ok"]:
                return {"ok": False, "orders": []}
            return {
                "ok": True,
                "complete": True,
                "open_orders_count": len(self.ledger["open_orders"]),
                "orders": copy.deepcopy(self.ledger["open_orders"]),
            }

        def fake_exec_history(order_date=None):
            if not self.ledger["executions_ok"]:
                return {"ok": False, "executions": []}
            dt = order_date or self.ny_compact
            rows = [
                copy.deepcopy(r)
                for r in self.ledger["executions"]
                if str(r.get("ord_dt") or self.ny_compact).replace("-", "") == str(dt).replace("-", "")
            ]
            return {
                "ok": True,
                "complete": True,
                "execution_count": len(rows),
                "executions": rows,
                "metadata": {"start_date": str(dt), "end_date": str(dt), "source_tr": "ust21150"},
            }

        def fake_order_history(start_date, end_date):
            if not self.ledger["order_history_ok"]:
                return {"ok": False, "orders": []}
            s_dt = str(start_date).replace("-", "")
            e_dt = str(end_date).replace("-", "")
            rows = [
                copy.deepcopy(r)
                for r in self.ledger["order_history"]
                if s_dt <= str(r.get("ord_dt") or self.ny_compact).replace("-", "") <= e_dt
            ]
            return {
                "ok": True,
                "complete": True,
                "order_count": len(rows),
                "orders": rows,
                "metadata": {"start_date": s_dt, "end_date": e_dt, "source_tr": "ust21180"},
            }

        def fake_cancel_order(order_no, symbol, exchange=None, quantity=0):
            rec = {"order_no": str(order_no), "symbol": symbol, "quantity": int(quantity)}
            self.ledger["sent_cancels"].append(rec)
            if callable(self.ledger["on_cancel_order_hook"]):
                return self.ledger["on_cancel_order_hook"](self.ledger, rec)
            if not self.ledger["cancel_ok"]:
                return {"ok": False, "order_no": str(order_no), "cancel_ord_no": "", "symbol": symbol, "msg": "Cancel failed"}
            self.ledger["next_ord_seq"] += 1
            c_ord_no = f"0000{self.ledger['next_ord_seq']}"
            return {"ok": True, "order_no": str(order_no), "cancel_ord_no": c_ord_no, "symbol": symbol, "msg": "정상 취소"}

        broker.get_overseas_stock_balance = fake_balance
        broker.get_open_orders = fake_open_orders
        broker.get_daily_execution_history = fake_exec_history
        broker.get_order_history = fake_order_history
        broker.cancel_order = fake_cancel_order
        broker.cancel_all_open_orders = lambda symbol=None: []
        broker.get_stock_quote = lambda sym, exchange=None: {
            "ok": True,
            "symbol": sym,
            "exchange": exchange or "NY",
            "last_price": runner.ws_streamer.get_latest_price(sym, 100.0) or 100.0,
        }
        return runner, broker

    def make_opener_factory(self):
        ledger = self.ledger

        class _FakeHTTPResp:
            def __init__(self, payload):
                self.status = 200
                self.headers = {"Content-Type": "application/json"}
                self._bytes = json.dumps(payload).encode("utf-8")

            def read(self):
                return self._bytes

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class _FakeOpener:
            def __init__(self, handler):
                self.guard = getattr(handler, "guard", None)

            def open(self, req, timeout=10):
                if callable(self.guard):
                    with cb_mod.ORDER_STATE_LOCK:
                        self.guard()
                body = json.loads(req.data.decode("utf-8"))
                api_id = req.headers.get("Api-id") or req.headers.get("api-id")
                ledger["next_ord_seq"] += 1
                ord_no = f"0000{ledger['next_ord_seq']}"
                call_rec = {
                    "api_id": api_id,
                    "body": body,
                    "ord_no": ord_no,
                    "symbol": body.get("stk_cd"),
                    "quantity": int(body.get("ord_qty", 0)),
                    "price": float(body.get("ord_uv", 0.0)),
                }
                ledger["sent_orders"].append(call_rec)
                if callable(ledger["on_send_order_hook"]):
                    custom = ledger["on_send_order_hook"](ledger, call_rec)
                    if custom is not None:
                        return _FakeHTTPResp(custom)
                return _FakeHTTPResp({"return_code": 0, "return_msg": "정상 주문 접수", "ord_no": ord_no})

        return lambda handler: _FakeOpener(handler)


class TestB03OrderLedgerIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._hashes_before = snapshot_prod_hashes()

    @classmethod
    def tearDownClass(cls):
        if LiveExperienceLogger._instance is not None:
            LiveExperienceLogger._instance.db_path = DEFAULT_DB_PATH
            LiveExperienceLogger._instance.csv_path = DEFAULT_CSV_PATH
        hashes_after = snapshot_prod_hashes()
        assert cls._hashes_before == hashes_after, (
            f"CRITICAL: Operational/model files modified during TestB03OrderLedgerIntegrity!\n"
            f"Before: {cls._hashes_before}\nAfter: {hashes_after}"
        )

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)
        self._orig_cb_file = cb_mod.CIRCUIT_STATE_FILE
        self.env = RealContractFakeKiwoomSetup(self.tmp_path)
        cb_mod.CIRCUIT_STATE_FILE = self.env.cb_file
        self.env.write_cb_state()

    def tearDown(self):
        cb_mod.CIRCUIT_STATE_FILE = self._orig_cb_file
        cb_mod._IN_MEMORY_SELL_RESERVATIONS.clear()
        cb_mod._IN_MEMORY_SELL_REVISION = 0
        cb_mod._STATE_WRITE_FAILED = False
        cb_mod._FAILED_STATE_CANDIDATE = None
        if LiveExperienceLogger._instance is not None:
            LiveExperienceLogger._instance.db_path = DEFAULT_DB_PATH
            LiveExperienceLogger._instance.csv_path = DEFAULT_CSV_PATH
        self.tmpdir.cleanup()

    # =========================================================================
    # STEP 3: Defect #1 — 1차 SELL 취소 후 2차 SELL 추격 보장 (7 Mandatory Tests)
    # =========================================================================
    def test_step3_1_sell_1st_timeout_0fill_cancel_in_ust21180_chases_2nd_sell_and_clears_pos(self):
        """[STEP 3-1] 1차 SELL 3초 타임아웃 -> ust20003 취소 -> ust21050 빈 목록 -> ust21150 빈 목록 -> ust21180 취소완료 -> 2차 SELL 4주 정상 발주 및 체결 -> active_position 삭제"""
        runner, broker = self.env.build_runner_and_broker()
        self.env.ledger["holdings"]["SOXL"] = {"qty": 4, "sell_avail": 4, "avg_px": 100.0, "cur_px": 97.0}
        runner.ws_streamer.latest_prices["SOXL"] = 97.0
        runner._save_active_position({
            "symbol": "SOXL", "quantity": 4, "price": 100.0,
            "buy_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "order_no": "000010001", "tp_pct": 10.0, "sl_pct": 2.5,
        })

        def on_send(ledger, rec):
            if len(ledger["sent_orders"]) == 1:
                # 1st SELL: 0 fill, stays open until cancel_order is called
                ledger["open_orders"] = [{
                    "ord_dt": self.env.ny_compact, "ord_no": rec["ord_no"], "orig_ord_no": "000000000",
                    "stk_cd": "SOXL", "slby_tp": "1", "ord_cntr_tp": "10",
                    "ord_qty": "000000000004", "cntr_qty": "000000000000",
                    "cncl_qty": "000000000000", "ord_remnq": "000000000004",
                }]
            else:
                # 2nd SELL: 100% filled immediately!
                ledger["holdings"].clear()
                runner._order_fills[rec["ord_no"]] = rec["quantity"]
                ledger["executions"].append({
                    "ord_dt": self.env.ny_compact, "ord_no": rec["ord_no"],
                    "stk_cd": "SOXL", "slby_tp": "1",
                    "ord_qty": "000000000004", "cntr_qty": "000000000004",
                    "cncl_qty": "000000000000", "ord_remnq": "000000000000",
                    "cntr_uv": str(rec["price"]),
                })
            return None

        def on_cancel(ledger, rec):
            ledger["open_orders"] = []
            # Real Kiwoom contract: 0-fill cancel does NOT appear in ust21150 (executions)!
            # It ONLY appears in ust21180 (order_history) as original row (cncl_qty=4) + cancel request row (cncl_qty=0, ord_uv=0.0000)
            c_no = "000058180"
            ledger["order_history"] = [
                {
                    "ord_dt": self.env.ny_compact, "ord_no": rec["order_no"],
                    "stk_cd": "SOXL", "slby_tp": "1",
                    "ord_qty": "000000000004", "cntr_qty": "000000000000",
                    "cncl_qty": "000000000004", "ord_remnq": "000000000000",
                    "ord_uv": "96.9500", "ord_stat": "취소완료",
                },
                {
                    "ord_dt": self.env.ny_compact, "ord_no": c_no, "orig_ord_no": rec["order_no"],
                    "stk_cd": "SOXL", "slby_tp": "1",
                    "ord_qty": "000000000004", "cntr_qty": "000000000000",
                    "cncl_qty": "000000000000", "ord_remnq": "000000000000",
                    "ord_uv": "0.0000", "ord_stat": "취소완료",
                },
            ]
            return {"ok": True, "order_no": rec["order_no"], "cancel_ord_no": c_no, "symbol": "SOXL"}

        self.env.ledger["on_send_order_hook"] = on_send
        self.env.ledger["on_cancel_order_hook"] = on_cancel

        with patch("core.circuit_breaker.CIRCUIT_STATE_FILE", self.env.cb_file), \
             patch("urllib.request.build_opener", side_effect=self.env.make_opener_factory()), \
             patch.object(config, "CANCEL_ORDER_WAIT_TIME", 0.005), \
             patch.object(config, "WS_FILL_POLL_INTERVAL", 0.005):
            ok = runner._execute_sell_with_10s_chase("SOXL", 4, "ATR 손절", buy_px=100.0, cur_px=97.0, is_stoploss=True)

        self.assertTrue(ok, "1차 SELL 미체결 취소 후 2차 SELL 추격이 성공해야 합니다.")
        self.assertEqual(len(self.env.ledger["sent_orders"]), 2, "총 2회 SELL 발주가 수행되어야 합니다.")
        self.assertEqual(self.env.ledger["sent_orders"][0]["quantity"], 4)
        self.assertEqual(self.env.ledger["sent_orders"][1]["quantity"], 4)
        self.assertIsNone(runner._get_active_position(), "2차 SELL 체결 완료 후 active_position이 삭제되어야 합니다.")

    def test_step3_2_sell_partial_1_of_4_then_cancel_3_chases_exact_3(self):
        """[STEP 3-2] 1차 SELL 4주 중 1주 체결 후 3주 취소 완료 -> 2차 SELL 정확히 3주 발주"""
        runner, broker = self.env.build_runner_and_broker()
        self.env.ledger["holdings"]["SOXL"] = {"qty": 4, "sell_avail": 4, "avg_px": 100.0, "cur_px": 97.0}
        runner.ws_streamer.latest_prices["SOXL"] = 97.0
        runner._save_active_position({
            "symbol": "SOXL", "quantity": 4, "price": 100.0,
            "buy_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "order_no": "000010001", "tp_pct": 10.0, "sl_pct": 2.5,
        })

        def on_send(ledger, rec):
            if len(ledger["sent_orders"]) == 1:
                # 1 share filled out of 4 during 1st SELL
                ledger["holdings"]["SOXL"]["qty"] = 3
                ledger["holdings"]["SOXL"]["sell_avail"] = 3
            else:
                # 2nd SELL: remaining 3 shares filled
                ledger["holdings"].clear()
                runner._order_fills[rec["ord_no"]] = rec["quantity"]
                ledger["executions"].append({
                    "ord_dt": self.env.ny_compact, "ord_no": rec["ord_no"],
                    "stk_cd": "SOXL", "slby_tp": "1",
                    "ord_qty": str(rec["quantity"]), "cntr_qty": str(rec["quantity"]),
                    "cncl_qty": "0", "ord_remnq": "0",
                })
            return None

        def on_cancel(ledger, rec):
            ledger["open_orders"] = []
            row_partial = {
                "ord_dt": self.env.ny_compact, "ord_no": rec["order_no"],
                "stk_cd": "SOXL", "slby_tp": "1",
                "ord_qty": "4", "cntr_qty": "1", "cncl_qty": "3", "ord_remnq": "0",
            }
            ledger["executions"] = [dict(row_partial)]
            ledger["order_history"] = [dict(row_partial)]
            return {"ok": True, "order_no": rec["order_no"], "cancel_ord_no": "000058199", "symbol": "SOXL"}

        self.env.ledger["on_send_order_hook"] = on_send
        self.env.ledger["on_cancel_order_hook"] = on_cancel

        with patch("core.circuit_breaker.CIRCUIT_STATE_FILE", self.env.cb_file), \
             patch("urllib.request.build_opener", side_effect=self.env.make_opener_factory()), \
             patch.object(config, "CANCEL_ORDER_WAIT_TIME", 0.005), \
             patch.object(config, "WS_FILL_POLL_INTERVAL", 0.005):
            ok = runner._execute_sell_with_10s_chase("SOXL", 4, "ATR 손절", buy_px=100.0, cur_px=97.0, is_stoploss=True)

        self.assertTrue(ok)
        self.assertEqual(len(self.env.ledger["sent_orders"]), 2)
        self.assertEqual(self.env.ledger["sent_orders"][0]["quantity"], 4)
        self.assertEqual(self.env.ledger["sent_orders"][1]["quantity"], 3, "2차 SELL은 정확히 잔여 3주만 발주해야 합니다.")
        self.assertIsNone(runner._get_active_position())

    def test_step3_3_sell_cancel_unconfirmed_open_order_remains_blocks_2nd_sell(self):
        """[STEP 3-3] 1차 SELL 취소 미확인(ust21050에 잔량 존재) -> 2차 SELL 차단"""
        runner, broker = self.env.build_runner_and_broker()
        self.env.ledger["holdings"]["SOXL"] = {"qty": 4, "sell_avail": 4, "avg_px": 100.0, "cur_px": 97.0}
        runner.ws_streamer.latest_prices["SOXL"] = 97.0
        runner._save_active_position({"symbol": "SOXL", "quantity": 4, "price": 100.0, "buy_time": "2026-10-15 10:00:00"})

        def on_cancel(ledger, rec):
            # Still open in ust21050 with ord_remnq = 4!
            ledger["open_orders"] = [{
                "ord_dt": self.env.ny_compact, "ord_no": rec["order_no"],
                "stk_cd": "SOXL", "slby_tp": "1",
                "ord_qty": "4", "cntr_qty": "0", "cncl_qty": "0", "ord_remnq": "4",
            }]
            return {"ok": True, "order_no": rec["order_no"], "cancel_ord_no": "000058180", "symbol": "SOXL"}

        self.env.ledger["on_cancel_order_hook"] = on_cancel

        with patch("core.circuit_breaker.CIRCUIT_STATE_FILE", self.env.cb_file), \
             patch("urllib.request.build_opener", side_effect=self.env.make_opener_factory()), \
             patch.object(config, "CANCEL_ORDER_WAIT_TIME", 0.005), \
             patch.object(config, "WS_FILL_POLL_INTERVAL", 0.005):
            ok = runner._execute_sell_with_10s_chase("SOXL", 4, "ATR 손절", buy_px=100.0, cur_px=97.0, is_stoploss=True)

        self.assertFalse(ok)
        self.assertEqual(len(self.env.ledger["sent_orders"]), 1, "ust21050 미체결 잔량 존재 시 2차 SELL은 차단되어야 합니다.")
        self.assertIsNotNone(runner._get_active_position())

    def test_step3_4_sell_cancel_missing_from_all_ledgers_blocks_2nd_sell(self):
        """[STEP 3-4] 1차 SELL 취소 후 ust21050, ust21150, ust21180 모두에서 주문 상태 확인 불가 -> 2차 SELL 차단"""
        runner, broker = self.env.build_runner_and_broker()
        self.env.ledger["holdings"]["SOXL"] = {"qty": 4, "sell_avail": 4, "avg_px": 100.0, "cur_px": 97.0}
        runner.ws_streamer.latest_prices["SOXL"] = 97.0
        runner._save_active_position({"symbol": "SOXL", "quantity": 4, "price": 100.0, "buy_time": "2026-10-15 10:00:00"})

        def on_cancel(ledger, rec):
            # All ledgers empty — no evidence of cancel completion anywhere!
            ledger["open_orders"] = []
            ledger["executions"] = []
            ledger["order_history"] = []
            return {"ok": True, "order_no": rec["order_no"], "cancel_ord_no": "000058180", "symbol": "SOXL"}

        self.env.ledger["on_cancel_order_hook"] = on_cancel

        with patch("core.circuit_breaker.CIRCUIT_STATE_FILE", self.env.cb_file), \
             patch("urllib.request.build_opener", side_effect=self.env.make_opener_factory()), \
             patch.object(config, "CANCEL_ORDER_WAIT_TIME", 0.005), \
             patch.object(config, "WS_FILL_POLL_INTERVAL", 0.005):
            ok = runner._execute_sell_with_10s_chase("SOXL", 4, "ATR 손절", buy_px=100.0, cur_px=97.0, is_stoploss=True)

        self.assertFalse(ok)
        self.assertEqual(len(self.env.ledger["sent_orders"]), 1, "원장 전체에서 주문 종결 근거가 없으면 2차 SELL은 차단되어야 합니다.")
        self.assertIsNotNone(runner._get_active_position())

    def test_step3_5_sell_filled_right_before_cancel_skips_2nd_sell_and_finishes(self):
        """[STEP 3-5] 1차 SELL 취소 후 잔고 0주 확인(ust21070=0, 체결 이력 일치) -> 2차 SELL 미발주 및 정상 종료"""
        runner, broker = self.env.build_runner_and_broker()
        self.env.ledger["holdings"]["SOXL"] = {"qty": 4, "sell_avail": 4, "avg_px": 100.0, "cur_px": 97.0}
        runner.ws_streamer.latest_prices["SOXL"] = 97.0
        runner._save_active_position({"symbol": "SOXL", "quantity": 4, "price": 100.0, "buy_time": "2026-10-15 10:00:00"})

        def on_cancel(ledger, rec):
            # Right before cancel, all 4 shares actually filled!
            ledger["holdings"].clear()
            ledger["open_orders"] = []
            filled_row = {
                "ord_dt": self.env.ny_compact, "ord_no": rec["order_no"],
                "stk_cd": "SOXL", "slby_tp": "1",
                "ord_qty": "4", "cntr_qty": "4", "cncl_qty": "0", "ord_remnq": "0",
            }
            ledger["executions"] = [dict(filled_row)]
            ledger["order_history"] = [dict(filled_row)]
            return {"ok": True, "order_no": rec["order_no"], "cancel_ord_no": "000058180", "symbol": "SOXL"}

        self.env.ledger["on_cancel_order_hook"] = on_cancel

        with patch("core.circuit_breaker.CIRCUIT_STATE_FILE", self.env.cb_file), \
             patch("urllib.request.build_opener", side_effect=self.env.make_opener_factory()), \
             patch.object(config, "CANCEL_ORDER_WAIT_TIME", 0.005), \
             patch.object(config, "WS_FILL_POLL_INTERVAL", 0.005):
            ok = runner._execute_sell_with_10s_chase("SOXL", 4, "ATR 손절", buy_px=100.0, cur_px=97.0, is_stoploss=True)

        self.assertTrue(ok)
        self.assertEqual(len(self.env.ledger["sent_orders"]), 1, "이미 전량 체결되어 잔고 0주이면 2차 SELL을 발주하면 안 됩니다.")
        self.assertIsNone(runner._get_active_position())

    def test_step3_6_sell_partial_fill_ledger_balance_contradiction_aborts_2nd_sell(self):
        """[STEP 3-6] 1차 SELL 중 부분체결 발생했으나 ust21070과 ust21150/ust21180 수량이 모순될 때 -> 추가 SELL 중단 및 경고"""
        runner, broker = self.env.build_runner_and_broker()
        self.env.ledger["holdings"]["SOXL"] = {"qty": 4, "sell_avail": 4, "avg_px": 100.0, "cur_px": 97.0}
        runner.ws_streamer.latest_prices["SOXL"] = 97.0
        runner._save_active_position({"symbol": "SOXL", "quantity": 4, "price": 100.0, "buy_time": "2026-10-15 10:00:00"})

        def on_cancel(ledger, rec):
            # ust21150 says 1 filled + 3 cancelled (so 3 should remain), but ust21070 still says 4 held (contradiction!)
            ledger["holdings"]["SOXL"]["qty"] = 4
            ledger["holdings"]["SOXL"]["sell_avail"] = 4
            row = {
                "ord_dt": self.env.ny_compact, "ord_no": rec["order_no"],
                "stk_cd": "SOXL", "slby_tp": "1",
                "ord_qty": "4", "cntr_qty": "1", "cncl_qty": "3", "ord_remnq": "0",
            }
            ledger["executions"] = [dict(row)]
            ledger["order_history"] = [dict(row)]
            return {"ok": True, "order_no": rec["order_no"], "cancel_ord_no": "000058180", "symbol": "SOXL"}

        self.env.ledger["on_cancel_order_hook"] = on_cancel

        with patch("core.circuit_breaker.CIRCUIT_STATE_FILE", self.env.cb_file), \
             patch("urllib.request.build_opener", side_effect=self.env.make_opener_factory()), \
             patch.object(config, "CANCEL_ORDER_WAIT_TIME", 0.005), \
             patch.object(config, "WS_FILL_POLL_INTERVAL", 0.005):
            ok = runner._execute_sell_with_10s_chase("SOXL", 4, "ATR 손절", buy_px=100.0, cur_px=97.0, is_stoploss=True)

        self.assertFalse(ok)
        self.assertEqual(len(self.env.ledger["sent_orders"]), 1, "잔고와 체결원장 수량 모순 시 2차 SELL은 차단되어야 합니다.")
        self.assertIsNotNone(runner._get_active_position())

    def test_step3_7_order_accepted_without_fill_never_deletes_active_position_early(self):
        """[STEP 3-7] 주문 접수만 되고 체결이 없는 상태에서 active_position이 조기 삭제되지 않음"""
        runner, broker = self.env.build_runner_and_broker()
        self.env.ledger["holdings"]["SOXL"] = {"qty": 1, "sell_avail": 1, "avg_px": 100.0, "cur_px": 97.0}
        runner.ws_streamer.latest_prices["SOXL"] = 97.0
        runner._save_active_position({"symbol": "SOXL", "quantity": 1, "price": 100.0, "buy_time": "2026-10-15 10:00:00"})

        # Simulate a WS message with nccs_qty=0 but tot_che_qty=0 and che_qty=0 (e.g. cancel/non-fill event)
        runner._on_websocket_execution({"ord_no": "000058101", "tot_che_qty": "0", "che_qty": "0", "nccs_qty": "0"})
        self.assertEqual(runner._order_fills.get("000058101", 0), 0, "체결수량 0인 WS 이벤트가 가짜 1주 체결을 만들면 안 됩니다.")

        self.env.ledger["cancel_ok"] = False
        with patch("core.circuit_breaker.CIRCUIT_STATE_FILE", self.env.cb_file), \
             patch("urllib.request.build_opener", side_effect=self.env.make_opener_factory()), \
             patch.object(config, "CANCEL_ORDER_WAIT_TIME", 0.005), \
             patch.object(config, "WS_FILL_POLL_INTERVAL", 0.005):
            ok = runner._execute_sell_with_10s_chase("SOXL", 1, "ATR 손절", buy_px=100.0, cur_px=97.0, is_stoploss=True)

        self.assertFalse(ok)
        self.assertIsNotNone(runner._get_active_position(), "체결이 없는 상태에서 active_position이 삭제되면 안 됩니다.")

    # =========================================================================
    # STEP 4: Defect #5 — 전일 sell_reservations 교차 일자 정리 (6 Mandatory Tests)
    # =========================================================================
    def test_step4_1_prior_day_filled_sell_reservation_resolved_allows_today_sell(self):
        """[STEP 4-1] 전일 ACCEPTED SELL 예약이 ust21180에서 전량 체결 완료된 경우 -> 당일 SELL 정상 허용 및 이전 예약 정리"""
        runner, broker = self.env.build_runner_and_broker()
        self.env.write_cb_state(sell_reservations=[{
            "id": "prev_filled_1", "account": "56910000-01", "mode": "REAL (실전투자)",
            "symbol": "SOXL", "trade_date": self.env.ny_yesterday, "quantity": 4,
            "held_before": 4, "phase": "ACCEPTED", "order_no": "000040001",
        }], sell_revision=1)
        self.env.ledger["holdings"]["SOXL"] = {"qty": 4, "sell_avail": 4, "avg_px": 100.0, "cur_px": 97.0}
        self.env.ledger["order_history"] = [{
            "ord_dt": self.env.ny_yesterday_compact, "ord_no": "000040001",
            "stk_cd": "SOXL", "slby_tp": "1",
            "ord_qty": "4", "cntr_qty": "4", "cncl_qty": "0", "ord_remnq": "0",
        }]

        with patch("core.circuit_breaker.CIRCUIT_STATE_FILE", self.env.cb_file), \
             patch("urllib.request.build_opener", side_effect=self.env.make_opener_factory()):
            res = broker.send_order("SOXL", "SELL", 4, 97.0)

        self.assertTrue(res["ok"])
        st_after = cb_mod.read_order_state()
        ids_after = [r["id"] for r in st_after.get("sell_reservations", [])]
        self.assertNotIn("prev_filled_1", ids_after, "종결 확인된 전일 예약은 제거되어야 합니다.")
        self.assertTrue(cb_mod.order_readiness(st_after)["ready"])

    def test_step4_2_prior_day_canceled_zero_fill_reservation_resolved_allows_today_sell(self):
        """[STEP 4-2] 전일 ACCEPTED SELL 예약이 ust21180에서 전량 취소 완료된 경우 -> 당일 SELL 정상 허용 및 이전 예약 정리"""
        runner, broker = self.env.build_runner_and_broker()
        self.env.write_cb_state(sell_reservations=[{
            "id": "prev_canc_1", "account": "56910000-01", "mode": "REAL (실전투자)",
            "symbol": "SOXL", "trade_date": self.env.ny_yesterday, "quantity": 4,
            "held_before": 4, "phase": "ACCEPTED", "order_no": "000040002",
        }], sell_revision=1)
        self.env.ledger["holdings"]["SOXL"] = {"qty": 4, "sell_avail": 4, "avg_px": 100.0, "cur_px": 97.0}
        self.env.ledger["order_history"] = [{
            "ord_dt": self.env.ny_yesterday_compact, "ord_no": "000040002",
            "stk_cd": "SOXL", "slby_tp": "1",
            "ord_qty": "4", "cntr_qty": "0", "cncl_qty": "4", "ord_remnq": "0",
        }]

        with patch("core.circuit_breaker.CIRCUIT_STATE_FILE", self.env.cb_file), \
             patch("urllib.request.build_opener", side_effect=self.env.make_opener_factory()):
            res = broker.send_order("SOXL", "SELL", 4, 97.0)

        self.assertTrue(res["ok"])
        st_after = cb_mod.read_order_state()
        self.assertNotIn("prev_canc_1", [r["id"] for r in st_after.get("sell_reservations", [])])

    def test_step4_3_prior_day_partial_fill_canceled_reservation_resolved_allows_today_sell(self):
        """[STEP 4-3] 전일 ACCEPTED SELL 예약이 부분체결 후 잔량 취소 완료된 경우 -> 당일 잔여 보유분 SELL 정상 허용"""
        runner, broker = self.env.build_runner_and_broker()
        self.env.write_cb_state(sell_reservations=[{
            "id": "prev_part_1", "account": "56910000-01", "mode": "REAL (실전투자)",
            "symbol": "SOXL", "trade_date": self.env.ny_yesterday, "quantity": 4,
            "held_before": 4, "phase": "ACCEPTED", "order_no": "000040003",
        }], sell_revision=1)
        self.env.ledger["holdings"]["SOXL"] = {"qty": 2, "sell_avail": 2, "avg_px": 100.0, "cur_px": 97.0}
        self.env.ledger["order_history"] = [{
            "ord_dt": self.env.ny_yesterday_compact, "ord_no": "000040003",
            "stk_cd": "SOXL", "slby_tp": "1",
            "ord_qty": "4", "cntr_qty": "2", "cncl_qty": "2", "ord_remnq": "0",
        }]

        with patch("core.circuit_breaker.CIRCUIT_STATE_FILE", self.env.cb_file), \
             patch("urllib.request.build_opener", side_effect=self.env.make_opener_factory()):
            res = broker.send_order("SOXL", "SELL", 2, 97.0)

        self.assertTrue(res["ok"])
        st_after = cb_mod.read_order_state()
        self.assertNotIn("prev_part_1", [r["id"] for r in st_after.get("sell_reservations", [])])

    def test_step4_4_prior_day_unknown_reservation_blocks_sell(self):
        """[STEP 4-4] 전일 UNKNOWN SELL 예약이 미해결인 경우 -> 자동 삭제 금지 및 차단 유지"""
        runner, broker = self.env.build_runner_and_broker()
        self.env.write_cb_state(sell_reservations=[{
            "id": "prev_unk_1", "account": "56910000-01", "mode": "REAL (실전투자)",
            "symbol": "SOXL", "trade_date": self.env.ny_yesterday, "quantity": 4,
            "held_before": 4, "phase": "UNKNOWN", "order_no": None,
        }], sell_revision=1)
        self.env.ledger["holdings"]["SOXL"] = {"qty": 4, "sell_avail": 4, "avg_px": 100.0, "cur_px": 97.0}

        with patch("core.circuit_breaker.CIRCUIT_STATE_FILE", self.env.cb_file), \
             patch("urllib.request.build_opener", side_effect=self.env.make_opener_factory()):
            with self.assertRaises(RuntimeError) as ctx:
                broker.send_order("SOXL", "SELL", 4, 97.0)

        self.assertIn("unresolved", str(ctx.exception))
        st_after = cb_mod.read_order_state()
        self.assertIn("prev_unk_1", [r["id"] for r in st_after.get("sell_reservations", [])])

    def test_step4_5_prior_day_reservation_ust21180_query_fails_blocks_sell(self):
        """[STEP 4-5] 전일 예약 조회 중 ust21180 실패 -> 자동 삭제 금지 및 차단 유지"""
        runner, broker = self.env.build_runner_and_broker()
        self.env.write_cb_state(sell_reservations=[{
            "id": "prev_qfail_1", "account": "56910000-01", "mode": "REAL (실전투자)",
            "symbol": "SOXL", "trade_date": self.env.ny_yesterday, "quantity": 4,
            "held_before": 4, "phase": "ACCEPTED", "order_no": "000040005",
        }], sell_revision=1)
        self.env.ledger["holdings"]["SOXL"] = {"qty": 4, "sell_avail": 4, "avg_px": 100.0, "cur_px": 97.0}
        self.env.ledger["order_history_ok"] = False

        with patch("core.circuit_breaker.CIRCUIT_STATE_FILE", self.env.cb_file), \
             patch("urllib.request.build_opener", side_effect=self.env.make_opener_factory()):
            with self.assertRaises(RuntimeError) as ctx:
                broker.send_order("SOXL", "SELL", 4, 97.0)

        self.assertIn("unresolved", str(ctx.exception))
        st_after = cb_mod.read_order_state()
        self.assertIn("prev_qfail_1", [r["id"] for r in st_after.get("sell_reservations", [])])

    def test_step4_6_same_day_unresolved_sell_reservation_still_blocks_duplicate_sell(self):
        """[STEP 4-6] 당일 미종결 SELL 예약은 절대 삭제되지 않고 중복 SELL 차단 유지"""
        runner, broker = self.env.build_runner_and_broker()
        self.env.write_cb_state(sell_reservations=[{
            "id": "today_open_1", "account": "56910000-01", "mode": "REAL (실전투자)",
            "symbol": "SOXL", "trade_date": self.env.ny_today, "quantity": 4,
            "held_before": 4, "phase": "ACCEPTED", "order_no": "000050001",
        }], sell_revision=1)
        self.env.ledger["holdings"]["SOXL"] = {"qty": 4, "sell_avail": 4, "avg_px": 100.0, "cur_px": 97.0}
        self.env.ledger["order_history"] = [{
            "ord_dt": self.env.ny_compact, "ord_no": "000050001",
            "stk_cd": "SOXL", "slby_tp": "1",
            "ord_qty": "4", "cntr_qty": "0", "cncl_qty": "0", "ord_remnq": "4",
        }]

        with patch("core.circuit_breaker.CIRCUIT_STATE_FILE", self.env.cb_file), \
             patch("urllib.request.build_opener", side_effect=self.env.make_opener_factory()):
            with self.assertRaises(RuntimeError) as ctx:
                broker.send_order("SOXL", "SELL", 4, 97.0)

        self.assertIn("unresolved", str(ctx.exception))

    # =========================================================================
    # STEP 5: Defect #2 — 1차 BUY 취소 후 2차 BUY 추격 복원 (7 Mandatory Tests)
    # =========================================================================
    def test_step5_1_buy_1st_timeout_0fill_cancel_in_ust21180_chases_2nd_buy(self):
        """[STEP 5-1] 1차 BUY 3초 타임아웃 -> ust20003 취소 -> ust21050 빈 목록 -> ust21150 빈 목록 -> ust21180 취소완료 -> 2차 BUY 10주 정상 발주"""
        runner, broker = self.env.build_runner_and_broker()
        runner.ws_streamer.latest_prices["SOXL"] = 40.0

        def on_send(ledger, rec):
            if len(ledger["sent_orders"]) == 1:
                ledger["open_orders"] = [{
                    "ord_dt": self.env.ny_compact, "ord_no": rec["ord_no"], "orig_ord_no": "000000000",
                    "stk_cd": "SOXL", "slby_tp": "2", "ord_cntr_tp": "10",
                    "ord_qty": "000000000010", "cntr_qty": "000000000000",
                    "cncl_qty": "000000000000", "ord_remnq": "000000000010",
                }]
            else:
                # 2nd BUY fills all 10 shares
                ledger["holdings"]["SOXL"] = {"qty": 10, "sell_avail": 10, "avg_px": rec["price"], "cur_px": rec["price"]}
                runner._order_fills[rec["ord_no"]] = rec["quantity"]
                ledger["executions"].append({
                    "ord_dt": self.env.ny_compact, "ord_no": rec["ord_no"],
                    "stk_cd": "SOXL", "slby_tp": "2",
                    "ord_qty": "000000000010", "cntr_qty": "000000000010",
                    "cncl_qty": "000000000000", "ord_remnq": "000000000000",
                    "cntr_uv": str(rec["price"]),
                })
            return None

        def on_cancel(ledger, rec):
            ledger["open_orders"] = []
            # Real Kiwoom contract: 0-fill BUY cancel does NOT appear in ust21150!
            # Appears in ust21180 as original row (cncl_qty=10) + separate cancel request row (cncl_qty=0, ord_uv=0.0000)
            c_no = "000058180"
            ledger["order_history"] = [
                {
                    "ord_dt": self.env.ny_compact, "ord_no": rec["order_no"],
                    "stk_cd": "SOXL", "slby_tp": "2",
                    "ord_qty": "000000000010", "cntr_qty": "000000000000",
                    "cncl_qty": "000000000010", "ord_remnq": "000000000000",
                    "ord_uv": "40.0300", "ord_stat": "취소완료",
                },
                {
                    "ord_dt": self.env.ny_compact, "ord_no": c_no, "orig_ord_no": rec["order_no"],
                    "stk_cd": "SOXL", "slby_tp": "2",
                    "ord_qty": "000000000010", "cntr_qty": "000000000000",
                    "cncl_qty": "000000000000", "ord_remnq": "000000000000",
                    "ord_uv": "0.0000", "ord_stat": "취소완료",
                },
            ]
            return {"ok": True, "order_no": rec["order_no"], "cancel_ord_no": c_no, "symbol": "SOXL"}

        self.env.ledger["on_send_order_hook"] = on_send
        self.env.ledger["on_cancel_order_hook"] = on_cancel

        with patch("core.circuit_breaker.CIRCUIT_STATE_FILE", self.env.cb_file), \
             patch("urllib.request.build_opener", side_effect=self.env.make_opener_factory()), \
             patch.object(config, "CANCEL_ORDER_WAIT_TIME", 0.005), \
             patch.object(config, "WS_FILL_POLL_INTERVAL", 0.005):
            ok = runner._execute_buy_with_10s_chase(
                "SOXL", 10, 40.0,
                {"gating_confidence": 0.78, "cross_dir": "LONG", "active_model_id": "B_Normalized_72"},
                {"dynamic_tp_px": 44.0, "dynamic_sl_px": 39.0, "tp_pct": 10.0, "sl_pct": 2.5, "time_stop_minutes": 90},
            )

        self.assertTrue(ok, "1차 BUY 0주 체결 취소 완료(ust21180) 후 2차 BUY가 발주 및 체결되어야 합니다.")
        self.assertEqual(len(self.env.ledger["sent_orders"]), 2)
        self.assertEqual(self.env.ledger["sent_orders"][0]["quantity"], 10)
        self.assertEqual(self.env.ledger["sent_orders"][1]["quantity"], 10)
        pos = runner._get_active_position()
        self.assertIsNotNone(pos)
        self.assertEqual(pos["quantity"], 10)

    def test_step5_2_buy_partial_4_of_10_then_cancel_6_chases_exact_6(self):
        """[STEP 5-2] 1차 BUY 10주 중 4주 체결 후 6주 취소 완료 -> 2차 BUY 정확히 6주 발주 및 active_position 보존"""
        runner, broker = self.env.build_runner_and_broker()
        runner.ws_streamer.latest_prices["SOXL"] = 40.0

        def on_send(ledger, rec):
            if len(ledger["sent_orders"]) == 1:
                ledger["holdings"]["SOXL"] = {"qty": 4, "sell_avail": 4, "avg_px": 40.03, "cur_px": 40.03}
            else:
                ledger["holdings"]["SOXL"] = {"qty": 4 + rec["quantity"], "sell_avail": 4 + rec["quantity"], "avg_px": 40.03, "cur_px": 40.03}
                runner._order_fills[rec["ord_no"]] = rec["quantity"]
            return None

        def on_cancel(ledger, rec):
            ledger["open_orders"] = []
            c_no = "000058180"
            row_orig = {
                "ord_dt": self.env.ny_compact, "ord_no": rec["order_no"],
                "stk_cd": "SOXL", "slby_tp": "2",
                "ord_qty": "10", "cntr_qty": "4", "cncl_qty": "6", "ord_remnq": "0",
                "cntr_uv": "40.03", "ord_uv": "40.03",
            }
            ledger["executions"] = [dict(row_orig)]
            ledger["order_history"] = [
                dict(row_orig),
                {
                    "ord_dt": self.env.ny_compact, "ord_no": c_no, "orig_ord_no": rec["order_no"],
                    "stk_cd": "SOXL", "slby_tp": "2",
                    "ord_qty": "6", "cntr_qty": "0", "cncl_qty": "0", "ord_remnq": "0",
                    "ord_uv": "0.0000", "ord_stat": "취소완료",
                },
            ]
            return {"ok": True, "order_no": rec["order_no"], "cancel_ord_no": c_no, "symbol": "SOXL"}

        self.env.ledger["on_send_order_hook"] = on_send
        self.env.ledger["on_cancel_order_hook"] = on_cancel

        with patch("core.circuit_breaker.CIRCUIT_STATE_FILE", self.env.cb_file), \
             patch("urllib.request.build_opener", side_effect=self.env.make_opener_factory()), \
             patch.object(config, "CANCEL_ORDER_WAIT_TIME", 0.005), \
             patch.object(config, "WS_FILL_POLL_INTERVAL", 0.005):
            ok = runner._execute_buy_with_10s_chase(
                "SOXL", 10, 40.0,
                {"gating_confidence": 0.78, "cross_dir": "LONG", "active_model_id": "B_Normalized_72"},
                {"dynamic_tp_px": 44.0, "dynamic_sl_px": 39.0, "tp_pct": 10.0, "sl_pct": 2.5, "time_stop_minutes": 90},
            )

        self.assertTrue(ok)
        self.assertEqual(len(self.env.ledger["sent_orders"]), 2)
        self.assertEqual(self.env.ledger["sent_orders"][1]["quantity"], 6)
        self.assertEqual(runner._get_active_position()["quantity"], 10)

    def test_step5_3_buy_cancel_fails_blocks_2nd_buy(self):
        """[STEP 5-3] 1차 BUY 취소 요청 실패(cancel_res.ok=False) -> 2차 BUY 차단"""
        runner, broker = self.env.build_runner_and_broker()
        self.env.ledger["cancel_ok"] = False

        with patch("core.circuit_breaker.CIRCUIT_STATE_FILE", self.env.cb_file), \
             patch("urllib.request.build_opener", side_effect=self.env.make_opener_factory()), \
             patch.object(config, "CANCEL_ORDER_WAIT_TIME", 0.005), \
             patch.object(config, "WS_FILL_POLL_INTERVAL", 0.005):
            ok = runner._execute_buy_with_10s_chase(
                "SOXL", 10, 40.0, {"gating_confidence": 0.78},
                {"dynamic_tp_px": 44.0, "dynamic_sl_px": 39.0},
            )

        self.assertFalse(ok)
        self.assertEqual(len(self.env.ledger["sent_orders"]), 1)

    def test_step5_4_buy_cancel_unconfirmed_open_order_remains_blocks_2nd_buy(self):
        """[STEP 5-4] 1차 BUY 취소 후 ust21050에 미체결 잔량 존재 -> 2차 BUY 차단"""
        runner, broker = self.env.build_runner_and_broker()

        def on_cancel(ledger, rec):
            ledger["open_orders"] = [{
                "ord_dt": self.env.ny_compact, "ord_no": rec["order_no"],
                "stk_cd": "SOXL", "slby_tp": "2",
                "ord_qty": "10", "cntr_qty": "0", "cncl_qty": "0", "ord_remnq": "10",
            }]
            return {"ok": True, "order_no": rec["order_no"], "cancel_ord_no": "000058180", "symbol": "SOXL"}

        self.env.ledger["on_cancel_order_hook"] = on_cancel

        with patch("core.circuit_breaker.CIRCUIT_STATE_FILE", self.env.cb_file), \
             patch("urllib.request.build_opener", side_effect=self.env.make_opener_factory()), \
             patch.object(config, "CANCEL_ORDER_WAIT_TIME", 0.005), \
             patch.object(config, "WS_FILL_POLL_INTERVAL", 0.005):
            ok = runner._execute_buy_with_10s_chase(
                "SOXL", 10, 40.0, {"gating_confidence": 0.78},
                {"dynamic_tp_px": 44.0, "dynamic_sl_px": 39.0},
            )

        self.assertFalse(ok)
        self.assertEqual(len(self.env.ledger["sent_orders"]), 1)

    def test_step5_5_buy_cancel_missing_from_all_ledgers_blocks_2nd_buy(self):
        """[STEP 5-5] ust21050 빈 목록 + ust21150 빈 목록 + ust21180에도 취소완료 근거 없음 -> 2차 BUY 차단"""
        runner, broker = self.env.build_runner_and_broker()

        def on_cancel(ledger, rec):
            ledger["open_orders"] = []
            ledger["executions"] = []
            ledger["order_history"] = []
            return {"ok": True, "order_no": rec["order_no"], "cancel_ord_no": "000058180", "symbol": "SOXL"}

        self.env.ledger["on_cancel_order_hook"] = on_cancel

        with patch("core.circuit_breaker.CIRCUIT_STATE_FILE", self.env.cb_file), \
             patch("urllib.request.build_opener", side_effect=self.env.make_opener_factory()), \
             patch.object(config, "CANCEL_ORDER_WAIT_TIME", 0.005), \
             patch.object(config, "WS_FILL_POLL_INTERVAL", 0.005):
            ok = runner._execute_buy_with_10s_chase(
                "SOXL", 10, 40.0, {"gating_confidence": 0.78},
                {"dynamic_tp_px": 44.0, "dynamic_sl_px": 39.0},
            )

        self.assertFalse(ok)
        self.assertEqual(len(self.env.ledger["sent_orders"]), 1)

    def test_step5_6_buy_ust21180_contradictory_rows_blocks_2nd_buy(self):
        """[STEP 5-6] ust21180 원주문 행과 취소요청 행 수량/원주문번호가 모순될 때 -> 2차 BUY 차단"""
        runner, broker = self.env.build_runner_and_broker()

        def on_cancel(ledger, rec):
            ledger["open_orders"] = []
            c_no = "000058180"
            ledger["order_history"] = [
                {
                    "ord_dt": self.env.ny_compact, "ord_no": rec["order_no"],
                    "stk_cd": "SOXL", "slby_tp": "2",
                    "ord_qty": "10", "cntr_qty": "0", "cncl_qty": "6", "ord_remnq": "0",
                },
                {
                    "ord_dt": self.env.ny_compact, "ord_no": c_no, "orig_ord_no": "999999999",
                    "stk_cd": "SOXL", "slby_tp": "2",
                    "ord_qty": "6", "cntr_qty": "0", "cncl_qty": "6", "ord_remnq": "0",
                },
            ]
            return {"ok": True, "order_no": rec["order_no"], "cancel_ord_no": c_no, "symbol": "SOXL"}

        self.env.ledger["on_cancel_order_hook"] = on_cancel

        with patch("core.circuit_breaker.CIRCUIT_STATE_FILE", self.env.cb_file), \
             patch("urllib.request.build_opener", side_effect=self.env.make_opener_factory()), \
             patch.object(config, "CANCEL_ORDER_WAIT_TIME", 0.005), \
             patch.object(config, "WS_FILL_POLL_INTERVAL", 0.005):
            ok = runner._execute_buy_with_10s_chase(
                "SOXL", 10, 40.0, {"gating_confidence": 0.78},
                {"dynamic_tp_px": 44.0, "dynamic_sl_px": 39.0},
            )

        self.assertFalse(ok)
        self.assertEqual(len(self.env.ledger["sent_orders"]), 1)

    def test_step5_7_buy_filled_right_before_cancel_skips_2nd_buy_and_saves_pos(self):
        """[STEP 5-7] 1차 BUY가 취소 직전 전량 체결된 경우 -> 2차 BUY 미발주 및 1차 체결 포지션 정상 저장"""
        runner, broker = self.env.build_runner_and_broker()

        def on_cancel(ledger, rec):
            ledger["holdings"]["SOXL"] = {"qty": 10, "sell_avail": 10, "avg_px": 40.02, "cur_px": 40.02}
            ledger["open_orders"] = []
            row = {
                "ord_dt": self.env.ny_compact, "ord_no": rec["order_no"],
                "stk_cd": "SOXL", "slby_tp": "2",
                "ord_qty": "10", "cntr_qty": "10", "cncl_qty": "0", "ord_remnq": "0",
                "cntr_uv": "40.02",
            }
            ledger["executions"] = [dict(row)]
            ledger["order_history"] = [dict(row)]
            return {"ok": True, "order_no": rec["order_no"], "cancel_ord_no": "000058180", "symbol": "SOXL"}

        self.env.ledger["on_cancel_order_hook"] = on_cancel

        with patch("core.circuit_breaker.CIRCUIT_STATE_FILE", self.env.cb_file), \
             patch("urllib.request.build_opener", side_effect=self.env.make_opener_factory()), \
             patch.object(config, "CANCEL_ORDER_WAIT_TIME", 0.005), \
             patch.object(config, "WS_FILL_POLL_INTERVAL", 0.005):
            ok = runner._execute_buy_with_10s_chase(
                "SOXL", 10, 40.0, {"gating_confidence": 0.78},
                {"dynamic_tp_px": 44.0, "dynamic_sl_px": 39.0},
            )

        self.assertTrue(ok)
        self.assertEqual(len(self.env.ledger["sent_orders"]), 1)
        pos = runner._get_active_position()
        self.assertIsNotNone(pos)
        self.assertEqual(pos["quantity"], 10)

    # =========================================================================
    # STEP 6: Defect #3 — _verify_pre_buy_ledger_clean 과거 체결 오탐 제거 (6 Mandatory Tests)
    # =========================================================================
    def test_step6_1_prior_completed_buy_sell_trade_and_canceled_unconfirmed_buy_allows_new_buy(self):
        """[STEP 6-1] 당일 오전 1차 매매(BUY 체결 -> SELL 전량 청산 완료) 후, 오후 미확정 BUY가 ust21180에서 취소완료 확인되면 신규 BUY 허용"""
        runner, broker = self.env.build_runner_and_broker()
        self.env.ledger["executions"] = [
            {"ord_dt": self.env.ny_compact, "ord_no": "000010001", "stk_cd": "SOXL", "slby_tp": "2", "ord_qty": "4", "cntr_qty": "4", "cncl_qty": "0", "ord_remnq": "0"},
            {"ord_dt": self.env.ny_compact, "ord_no": "000010002", "stk_cd": "SOXL", "slby_tp": "1", "ord_qty": "4", "cntr_qty": "4", "cncl_qty": "0", "ord_remnq": "0"},
        ]
        self.env.ledger["order_history"] = [
            {"ord_dt": self.env.ny_compact, "ord_no": "000010001", "stk_cd": "SOXL", "slby_tp": "2", "ord_qty": "4", "cntr_qty": "4", "cncl_qty": "0", "ord_remnq": "0"},
            {"ord_dt": self.env.ny_compact, "ord_no": "000010002", "stk_cd": "SOXL", "slby_tp": "1", "ord_qty": "4", "cntr_qty": "4", "cncl_qty": "0", "ord_remnq": "0"},
            {"ord_dt": self.env.ny_compact, "ord_no": "000020001", "stk_cd": "SOXL", "slby_tp": "2", "ord_qty": "4", "cntr_qty": "0", "cncl_qty": "4", "ord_remnq": "0"},
        ]
        runner._unconfirmed_buy_order = {"symbol": "SOXL", "order_no": "000020001", "target_qty": 4}

        clean = runner._verify_pre_buy_ledger_clean("SOXL")
        self.assertTrue(clean, "이미 SELL로 청산된 오전 매매 때문에 오후 신규 BUY가 영구 차단되면 안 됩니다.")
        self.assertIsNone(runner._unconfirmed_buy_order)

    def test_step6_2_unconfirmed_buy_still_open_in_ust21050_blocks_new_buy(self):
        """[STEP 6-2] 미확정 BUY 주문이 ust21050에 미체결로 남아 있으면 신규 BUY 차단"""
        runner, broker = self.env.build_runner_and_broker()
        self.env.ledger["open_orders"] = [
            {"ord_dt": self.env.ny_compact, "ord_no": "000020001", "stk_cd": "SOXL", "slby_tp": "2", "ord_qty": "4", "cntr_qty": "0", "ord_remnq": "4"}
        ]
        runner._unconfirmed_buy_order = {"symbol": "SOXL", "order_no": "000020001", "target_qty": 4}

        self.assertFalse(runner._verify_pre_buy_ledger_clean("SOXL"))
        self.assertIsNotNone(runner._unconfirmed_buy_order)

    def test_step6_3_unconfirmed_buy_partially_filled_with_balance_lag_recovers_and_blocks_new_buy(self):
        """[STEP 6-3] 미확정 BUY 주문이 부분체결되었고 잔고 반영 지연으로 ust21070=0이지만 ust21150에 체결이 있으면 포지션 복구/보존 및 신규 BUY 차단"""
        runner, broker = self.env.build_runner_and_broker()
        self.env.ledger["executions"] = [
            {"ord_dt": self.env.ny_compact, "ord_no": "000020001", "stk_cd": "SOXL", "slby_tp": "2", "ord_qty": "10", "cntr_qty": "4", "cncl_qty": "6", "ord_remnq": "0", "cntr_uv": "40.00"}
        ]
        self.env.ledger["order_history"] = list(self.env.ledger["executions"])
        runner._unconfirmed_buy_order = {"symbol": "SOXL", "order_no": "000020001", "target_qty": 10}

        self.assertFalse(runner._verify_pre_buy_ledger_clean("SOXL"), "미청산 체결분이 체결원장에 존재하면 신규 BUY를 차단해야 합니다.")

    def test_step6_4_unconfirmed_buy_without_order_no_and_unverified_blocks_new_buy(self):
        """[STEP 6-4] 미확정 BUY 주문의 주문번호가 없고(응답 유실) 원장에서 종결 여부를 확인할 수 없으면 신규 BUY 차단"""
        runner, broker = self.env.build_runner_and_broker()
        runner._unconfirmed_buy_order = {"symbol": "SOXL", "target_qty": 10, "price": 40.03, "timestamp": "2026-10-15 10:00:00"}

        self.assertFalse(runner._verify_pre_buy_ledger_clean("SOXL"))
        self.assertIsNotNone(runner._unconfirmed_buy_order)

    def test_step6_5_soxl_trade_closed_then_soxs_opposite_buy_with_resolved_unconfirmed_allowed(self):
        """[STEP 6-5] SOXL 매매 종료 후 SOXS 반대 방향 진입 시 과거 SOXL 체결 기록 때문에 차단되지 않음"""
        runner, broker = self.env.build_runner_and_broker()
        self.env.ledger["executions"] = [
            {"ord_dt": self.env.ny_compact, "ord_no": "000010001", "stk_cd": "SOXL", "slby_tp": "2", "ord_qty": "4", "cntr_qty": "4", "cncl_qty": "0", "ord_remnq": "0"},
            {"ord_dt": self.env.ny_compact, "ord_no": "000010002", "stk_cd": "SOXL", "slby_tp": "1", "ord_qty": "4", "cntr_qty": "4", "cncl_qty": "0", "ord_remnq": "0"},
        ]
        self.env.ledger["order_history"] = [
            {"ord_dt": self.env.ny_compact, "ord_no": "000010001", "stk_cd": "SOXL", "slby_tp": "2", "ord_qty": "4", "cntr_qty": "4", "cncl_qty": "0", "ord_remnq": "0"},
            {"ord_dt": self.env.ny_compact, "ord_no": "000010002", "stk_cd": "SOXL", "slby_tp": "1", "ord_qty": "4", "cntr_qty": "4", "cncl_qty": "0", "ord_remnq": "0"},
            {"ord_dt": self.env.ny_compact, "ord_no": "000020001", "stk_cd": "SOXL", "slby_tp": "2", "ord_qty": "4", "cntr_qty": "0", "cncl_qty": "4", "ord_remnq": "0"},
        ]
        runner._unconfirmed_buy_order = {"symbol": "SOXL", "order_no": "000020001", "target_qty": 4}

        self.assertTrue(runner._verify_pre_buy_ledger_clean("SOXS"))
        self.assertIsNone(runner._unconfirmed_buy_order)

    def test_step6_6_real_account_holds_soxl_or_soxs_blocks_new_buy(self):
        """[STEP 6-6] 실제로 SOXL 또는 SOXS 보유가 남아 있으면 신규 BUY 차단"""
        runner, broker = self.env.build_runner_and_broker()
        self.env.ledger["holdings"]["SOXL"] = {"qty": 4, "sell_avail": 4, "avg_px": 100.0, "cur_px": 100.0}
        self.assertFalse(runner._verify_pre_buy_ledger_clean("SOXS"))

    # =========================================================================
    # STEP 7: Defect #4 — 부분체결 후 잔고 반영 지연 시 포지션 보존 (7 Mandatory Tests)
    # =========================================================================
    def test_step7_1_partial_fill_4_with_balance_0_lag_preserves_4_shares_from_ust21150(self):
        """[STEP 7-1] 1차 BUY 10주 중 4주 체결, ust21070=0주(지연), ust21150=4주 체결(ord_remnq=0) -> active_position 4주 생성 및 SL/TP 보호"""
        runner, broker = self.env.build_runner_and_broker()
        self.env.ledger["executions"] = [{
            "ord_dt": self.env.ny_compact, "ord_no": "000030001",
            "stk_cd": "SOXL", "slby_tp": "2",
            "ord_qty": "10", "cntr_qty": "4", "cncl_qty": "6", "ord_remnq": "0", "cntr_uv": "40.00",
        }]
        saved = runner._preserve_verified_partial_buy(
            symbol="SOXL", order_no="000030001", buy_time_str="2026-10-15 10:00:00",
            tp_pct=10.0, sl_pct=2.5, tp_px=44.0, sl_px=39.0, time_stop_min=90,
            strategy_tag="Lumos V4", ref_price=40.0, moe_res={"gating_confidence": 0.78},
            stk_bal_snapshot={"ok": True, "holdings": [], "holdings_count": 0},
        )
        self.assertTrue(saved, "ust21070=0주(지연)여도 ust21150에서 검증된 4주 부분체결은 active_position으로 보존되어야 합니다.")
        pos = runner._get_active_position()
        self.assertIsNotNone(pos)
        self.assertEqual(pos["quantity"], 4)
        self.assertAlmostEqual(pos["price"], 40.0, places=2)
        self.assertAlmostEqual(pos["sl_px"], 39.0, places=2)
        self.assertAlmostEqual(pos["tp_px"], 44.0, places=2)

    def test_step7_2_preserved_partial_fill_blocks_additional_buy(self):
        """[STEP 7-2] 위 상태에서 추가 BUY 시도 -> 차단"""
        runner, broker = self.env.build_runner_and_broker()
        self.env.ledger["executions"] = [{
            "ord_dt": self.env.ny_compact, "ord_no": "000030001",
            "stk_cd": "SOXL", "slby_tp": "2",
            "ord_qty": "10", "cntr_qty": "4", "cncl_qty": "6", "ord_remnq": "0", "cntr_uv": "40.00",
        }]
        runner._preserve_verified_partial_buy(
            symbol="SOXL", order_no="000030001", buy_time_str="2026-10-15 10:00:00",
            tp_pct=10.0, sl_pct=2.5, tp_px=44.0, sl_px=39.0, time_stop_min=90,
            strategy_tag="Lumos V4", ref_price=40.0, moe_res={"gating_confidence": 0.78},
            stk_bal_snapshot={"ok": True, "holdings": [], "holdings_count": 0},
        )
        self.assertFalse(runner._verify_pre_buy_ledger_clean("SOXL"))

    def test_step7_3_next_loop_ust21070_reflects_4_shares_clears_mismatch_and_sells_on_sl(self):
        """[STEP 7-3] 다음 루프에서 ust21070이 4주로 반영되면 불일치 플래그 해제 및 정상 보유 관리/SL 청산"""
        runner, broker = self.env.build_runner_and_broker()
        self.env.ledger["executions"] = [{
            "ord_dt": self.env.ny_compact, "ord_no": "000030001",
            "stk_cd": "SOXL", "slby_tp": "2",
            "ord_qty": "10", "cntr_qty": "4", "cncl_qty": "6", "ord_remnq": "0", "cntr_uv": "40.00",
        }]
        runner._preserve_verified_partial_buy(
            symbol="SOXL", order_no="000030001", buy_time_str="2026-10-15 10:00:00",
            tp_pct=10.0, sl_pct=2.5, tp_px=44.0, sl_px=39.0, time_stop_min=90,
            strategy_tag="Lumos V4", ref_price=40.0, moe_res={"gating_confidence": 0.78},
            stk_bal_snapshot={"ok": True, "holdings": [], "holdings_count": 0},
        )
        self.assertIsNotNone(runner._unreconciled_holding_mismatch)

        # Next loop: ust21070 now reflects 4 shares!
        self.env.ledger["holdings"]["SOXL"] = {"qty": 4, "sell_avail": 4, "avg_px": 40.0, "cur_px": 38.5}
        bal_now = broker.get_overseas_stock_balance(force_refresh=True)
        runner._reconcile_active_position_with_ledger(bal_now)
        self.assertIsNone(runner._unreconciled_holding_mismatch)

        def on_send(ledger, rec):
            ledger["holdings"].clear()
            runner._order_fills[rec["ord_no"]] = rec["quantity"]
            return None

        self.env.ledger["on_send_order_hook"] = on_send
        with patch("core.circuit_breaker.CIRCUIT_STATE_FILE", self.env.cb_file), \
             patch("urllib.request.build_opener", side_effect=self.env.make_opener_factory()):
            runner._manage_open_positions(bal_now, realtime_px_override={"SOXL": 38.5})

        self.assertEqual(len(self.env.ledger["sent_orders"]), 1)
        self.assertEqual(self.env.ledger["sent_orders"][0]["quantity"], 4)
        self.assertIsNone(runner._get_active_position())

    def test_step7_4_zero_fill_in_both_ust21070_and_ust21150_creates_no_fake_position(self):
        """[STEP 7-4] ust21070=0주이고 ust21150에도 체결이 없으면 가짜 active_position을 만들지 않음"""
        runner, broker = self.env.build_runner_and_broker()
        saved = runner._preserve_verified_partial_buy(
            symbol="SOXL", order_no="000030001", buy_time_str="2026-10-15 10:00:00",
            tp_pct=10.0, sl_pct=2.5, tp_px=44.0, sl_px=39.0, time_stop_min=90,
            strategy_tag="Lumos V4", ref_price=40.0, moe_res={"gating_confidence": 0.78},
            stk_bal_snapshot={"ok": True, "holdings": [], "holdings_count": 0},
        )
        self.assertFalse(saved)
        self.assertIsNone(runner._get_active_position())

    def test_step7_5_duplicate_rows_in_ust21150_never_overcount_quantity(self):
        """[STEP 7-5] ust21150 중복 행 존재 시 수량 과대계산 금지"""
        runner, broker = self.env.build_runner_and_broker()
        dup_row = {
            "ord_dt": self.env.ny_compact, "ord_no": "000030001", "cntr_no": "E001",
            "stk_cd": "SOXL", "slby_tp": "2",
            "ord_qty": "10", "cntr_qty": "4", "cncl_qty": "6", "ord_remnq": "0", "cntr_uv": "40.00",
        }
        self.env.ledger["executions"] = [dict(dup_row), dict(dup_row)]
        saved = runner._preserve_verified_partial_buy(
            symbol="SOXL", order_no="000030001", buy_time_str="2026-10-15 10:00:00",
            tp_pct=10.0, sl_pct=2.5, tp_px=44.0, sl_px=39.0, time_stop_min=90,
            strategy_tag="Lumos V4", ref_price=40.0, moe_res={"gating_confidence": 0.78},
            stk_bal_snapshot={"ok": True, "holdings": [], "holdings_count": 0},
        )
        self.assertTrue(saved)
        self.assertEqual(runner._get_active_position()["quantity"], 4, "중복 체결 행이 8주로 과대계산되면 안 됩니다.")

    def test_step7_6_unverified_price_in_ust21150_refuses_position_and_records_mismatch(self):
        """[STEP 7-6] ust21150 체결 단가가 0 또는 파싱 불가일 때 임의 가격으로 포지션 생성 금지 및 불일치 경고"""
        runner, broker = self.env.build_runner_and_broker()
        self.env.ledger["executions"] = [{
            "ord_dt": self.env.ny_compact, "ord_no": "000030001",
            "stk_cd": "SOXL", "slby_tp": "2",
            "ord_qty": "10", "cntr_qty": "4", "cncl_qty": "6", "ord_remnq": "0", "cntr_uv": "0.00",
        }]
        saved = runner._preserve_verified_partial_buy(
            symbol="SOXL", order_no="000030001", buy_time_str="2026-10-15 10:00:00",
            tp_pct=10.0, sl_pct=2.5, tp_px=44.0, sl_px=39.0, time_stop_min=90,
            strategy_tag="Lumos V4", ref_price=40.0, moe_res={"gating_confidence": 0.78},
            stk_bal_snapshot={"ok": True, "holdings": [], "holdings_count": 0},
            known_filled_hint=4,
        )
        self.assertFalse(saved)
        self.assertIsNone(runner._get_active_position())
        self.assertIsNotNone(runner._unreconciled_holding_mismatch)

    def test_step7_7_reconcile_updates_active_position_when_4_becomes_2_after_partial_sell(self):
        """[STEP 7-7] ust21150=4주 생성 후 실제로 2주가 이미 매도되어 ust21070=2주인 경우 다음 동기화에서 2주로 정확히 보정"""
        runner, broker = self.env.build_runner_and_broker()
        runner._save_active_position({
            "symbol": "SOXL", "quantity": 4, "qty": 4, "price": 40.0, "buy_price": 40.0,
            "buy_time": "2026-10-15 10:00:00", "order_no": "000030001",
            "tp_pct": 10.0, "sl_pct": 2.5, "tp_px": 44.0, "sl_px": 39.0,
        })
        self.env.ledger["holdings"]["SOXL"] = {"qty": 2, "sell_avail": 2, "avg_px": 40.0, "cur_px": 40.0}
        runner._reconcile_active_position_with_ledger(broker.get_overseas_stock_balance(force_refresh=True))
        self.assertEqual(runner._get_active_position()["quantity"], 2)

    # =========================================================================
    # STEP 8: 통합 E2E 시나리오 A ~ E
    # =========================================================================
    def test_step8_integration_a_buy_chase_to_sell_chase_and_clear(self):
        """[Integration A] 1차 BUY 타임아웃 취소 -> 2차 BUY 체결 -> SL 도달 -> 1차 SELL 타임아웃 취소 -> 2차 SELL 체결 -> 포지션 완전 정리"""
        runner, broker = self.env.build_runner_and_broker()
        runner.ws_streamer.latest_prices["SOXL"] = 40.0

        def on_send(ledger, rec):
            n = len(ledger["sent_orders"])
            if n == 1:
                # 1st BUY: 0 fill, timeout
                pass
            elif n == 2:
                # 2nd BUY: fills 4 shares
                ledger["holdings"]["SOXL"] = {"qty": 4, "sell_avail": 4, "avg_px": rec["price"], "cur_px": rec["price"]}
                runner._order_fills[rec["ord_no"]] = 4
                ledger["executions"].append({
                    "ord_dt": self.env.ny_compact, "ord_no": rec["ord_no"],
                    "stk_cd": "SOXL", "slby_tp": "2",
                    "ord_qty": "4", "cntr_qty": "4", "cncl_qty": "0", "ord_remnq": "0", "cntr_uv": str(rec["price"]),
                })
                ledger["order_history"].append(dict(ledger["executions"][-1]))
            elif n == 3:
                # 1st SELL: 0 fill, timeout
                pass
            elif n == 4:
                # 2nd SELL: fills 4 shares
                ledger["holdings"].clear()
                runner._order_fills[rec["ord_no"]] = 4
                ledger["executions"].append({
                    "ord_dt": self.env.ny_compact, "ord_no": rec["ord_no"],
                    "stk_cd": "SOXL", "slby_tp": "1",
                    "ord_qty": "4", "cntr_qty": "4", "cncl_qty": "0", "ord_remnq": "0", "cntr_uv": str(rec["price"]),
                })
                ledger["order_history"].append(dict(ledger["executions"][-1]))
            return None

        def on_cancel(ledger, rec):
            ledger["open_orders"] = []
            c_no = f"0000{ledger['next_ord_seq'] + 1}"
            ledger["next_ord_seq"] += 1
            # Determine if cancelling 1st BUY or 1st SELL
            slby = "2" if len(ledger["sent_cancels"]) == 1 else "1"
            ledger["order_history"].append({
                "ord_dt": self.env.ny_compact, "ord_no": rec["order_no"],
                "stk_cd": "SOXL", "slby_tp": slby,
                "ord_qty": "4", "cntr_qty": "0", "cncl_qty": "4", "ord_remnq": "0",
                "ord_uv": "40.0000", "ord_stat": "취소완료",
            })
            return {"ok": True, "order_no": rec["order_no"], "cancel_ord_no": c_no, "symbol": "SOXL"}

        self.env.ledger["on_send_order_hook"] = on_send
        self.env.ledger["on_cancel_order_hook"] = on_cancel

        with patch("core.circuit_breaker.CIRCUIT_STATE_FILE", self.env.cb_file), \
             patch("urllib.request.build_opener", side_effect=self.env.make_opener_factory()), \
             patch.object(config, "CANCEL_ORDER_WAIT_TIME", 0.005), \
             patch.object(config, "WS_FILL_POLL_INTERVAL", 0.005):
            buy_ok = runner._execute_buy_with_10s_chase(
                "SOXL", 4, 40.0,
                {"gating_confidence": 0.78, "cross_dir": "LONG", "active_model_id": "B_Normalized_72"},
                {"dynamic_tp_px": 44.0, "dynamic_sl_px": 39.0, "tp_pct": 10.0, "sl_pct": 2.5, "time_stop_minutes": 90},
            )
            self.assertTrue(buy_ok)
            self.assertIsNotNone(runner._get_active_position())

            # Price drops to $38.0 -> SL triggers -> 1st SELL times out -> 2nd SELL fills
            runner.ws_streamer.latest_prices["SOXL"] = 38.0
            self.env.ledger["holdings"]["SOXL"]["cur_px"] = 38.0
            runner._manage_open_positions(broker.get_overseas_stock_balance(force_refresh=True), realtime_px_override={"SOXL": 38.0})

        self.assertIsNone(runner._get_active_position())
        self.assertEqual(len(self.env.ledger["sent_orders"]), 4)

    def test_step8_integration_b_prior_day_sell_reservation_to_today_time_stop_sell(self):
        """[Integration B] 전일 종결된 sell_reservations 존재 -> 다음 거래일 보유 포지션 90분 타임스탑 도달 -> SELL 정상 성공"""
        runner, broker = self.env.build_runner_and_broker()
        self.env.write_cb_state(sell_reservations=[{
            "id": "prev_day_res", "account": "56910000-01", "mode": "REAL (실전투자)",
            "symbol": "SOXL", "trade_date": self.env.ny_yesterday, "quantity": 4,
            "held_before": 4, "phase": "ACCEPTED", "order_no": "000040009",
        }], sell_revision=1)
        self.env.ledger["order_history"] = [{
            "ord_dt": self.env.ny_yesterday_compact, "ord_no": "000040009",
            "stk_cd": "SOXL", "slby_tp": "1",
            "ord_qty": "4", "cntr_qty": "0", "cncl_qty": "4", "ord_remnq": "0",
        }]
        self.env.ledger["holdings"]["SOXL"] = {"qty": 4, "sell_avail": 4, "avg_px": 40.0, "cur_px": 40.1}
        old_bt = (datetime.now() - timedelta(minutes=95)).strftime("%Y-%m-%d %H:%M:%S")
        runner._save_active_position({
            "symbol": "SOXL", "quantity": 4, "price": 40.0, "buy_time": old_bt,
            "order_no": "000010001", "tp_pct": 10.0, "sl_pct": 2.5, "time_stop_minutes": 90,
        })

        def on_send(ledger, rec):
            ledger["holdings"].clear()
            runner._order_fills[rec["ord_no"]] = rec["quantity"]
            return None

        self.env.ledger["on_send_order_hook"] = on_send
        with patch("core.circuit_breaker.CIRCUIT_STATE_FILE", self.env.cb_file), \
             patch("urllib.request.build_opener", side_effect=self.env.make_opener_factory()):
            runner._manage_open_positions(broker.get_overseas_stock_balance(force_refresh=True), realtime_px_override={"SOXL": 40.1})

        self.assertEqual(len(self.env.ledger["sent_orders"]), 1)
        self.assertIsNone(runner._get_active_position())

    def test_step8_integration_c_morning_trade_closed_then_afternoon_buy_succeeds(self):
        """[Integration C] 당일 오전 SOXL 매매 완결(BUY -> SELL) -> 오후 미확정 BUY 취소 확인 후 신규 BUY 신호 -> 정상 진입"""
        runner, broker = self.env.build_runner_and_broker()
        self.env.ledger["executions"] = [
            {"ord_dt": self.env.ny_compact, "ord_no": "000010001", "stk_cd": "SOXL", "slby_tp": "2", "ord_qty": "4", "cntr_qty": "4", "cncl_qty": "0", "ord_remnq": "0"},
            {"ord_dt": self.env.ny_compact, "ord_no": "000010002", "stk_cd": "SOXL", "slby_tp": "1", "ord_qty": "4", "cntr_qty": "4", "cncl_qty": "0", "ord_remnq": "0"},
        ]
        self.env.ledger["order_history"] = [
            {"ord_dt": self.env.ny_compact, "ord_no": "000010001", "stk_cd": "SOXL", "slby_tp": "2", "ord_qty": "4", "cntr_qty": "4", "cncl_qty": "0", "ord_remnq": "0"},
            {"ord_dt": self.env.ny_compact, "ord_no": "000010002", "stk_cd": "SOXL", "slby_tp": "1", "ord_qty": "4", "cntr_qty": "4", "cncl_qty": "0", "ord_remnq": "0"},
            {"ord_dt": self.env.ny_compact, "ord_no": "000020001", "stk_cd": "SOXL", "slby_tp": "2", "ord_qty": "4", "cntr_qty": "0", "cncl_qty": "4", "ord_remnq": "0"},
        ]
        runner._unconfirmed_buy_order = {"symbol": "SOXL", "order_no": "000020001", "target_qty": 4}

        def on_send(ledger, rec):
            ledger["holdings"]["SOXL"] = {"qty": rec["quantity"], "sell_avail": rec["quantity"], "avg_px": rec["price"], "cur_px": rec["price"]}
            runner._order_fills[rec["ord_no"]] = rec["quantity"]
            return None

        self.env.ledger["on_send_order_hook"] = on_send
        with patch("core.circuit_breaker.CIRCUIT_STATE_FILE", self.env.cb_file), \
             patch("urllib.request.build_opener", side_effect=self.env.make_opener_factory()):
            ok = runner._execute_buy_with_10s_chase(
                "SOXL", 5, 40.0,
                {"gating_confidence": 0.78, "cross_dir": "LONG", "active_model_id": "B_Normalized_72"},
                {"dynamic_tp_px": 44.0, "dynamic_sl_px": 39.0, "tp_pct": 10.0, "sl_pct": 2.5, "time_stop_minutes": 90},
            )

        self.assertTrue(ok)
        self.assertEqual(runner._get_active_position()["quantity"], 5)

    def test_step8_integration_d_partial_buy_balance_lag_to_sl_sell(self):
        """[Integration D] 1차 BUY 부분체결 + 잔고 조회 지연 -> active_position 보존 -> 다음 루프 잔고 반영 -> SL 청산 성공"""
        runner, broker = self.env.build_runner_and_broker()
        self.env.ledger["cancel_ok"] = False

        def on_send(ledger, rec):
            if rec["api_id"] == "ust20000":
                # 1st BUY: 4 of 10 filled in ust21150, but ust21070 balance is delayed (0 shares)
                ledger["executions"].append({
                    "ord_dt": self.env.ny_compact, "ord_no": rec["ord_no"],
                    "stk_cd": "SOXL", "slby_tp": "2",
                    "ord_qty": "10", "cntr_qty": "4", "cncl_qty": "6", "ord_remnq": "0", "cntr_uv": "40.00",
                })
            elif rec["api_id"] == "ust20001":
                ledger["holdings"].clear()
                runner._order_fills[rec["ord_no"]] = rec["quantity"]
            return None

        self.env.ledger["on_send_order_hook"] = on_send
        with patch("core.circuit_breaker.CIRCUIT_STATE_FILE", self.env.cb_file), \
             patch("urllib.request.build_opener", side_effect=self.env.make_opener_factory()), \
             patch.object(config, "CANCEL_ORDER_WAIT_TIME", 0.005), \
             patch.object(config, "WS_FILL_POLL_INTERVAL", 0.005):
            runner._execute_buy_with_10s_chase(
                "SOXL", 10, 40.0,
                {"gating_confidence": 0.78, "cross_dir": "LONG", "active_model_id": "B_Normalized_72"},
                {"dynamic_tp_px": 44.0, "dynamic_sl_px": 39.0, "tp_pct": 10.0, "sl_pct": 2.5, "time_stop_minutes": 90},
            )
            pos = runner._get_active_position()
            self.assertIsNotNone(pos, "부분체결 + 잔고 반영 지연 시에도 4주 active_position이 보존되어야 합니다.")
            self.assertEqual(pos["quantity"], 4)

            # Next loop: balance reflects 4 shares, price hits SL ($38.50)
            self.env.ledger["holdings"]["SOXL"] = {"qty": 4, "sell_avail": 4, "avg_px": 40.0, "cur_px": 38.5}
            bal = broker.get_overseas_stock_balance(force_refresh=True)
            runner._reconcile_active_position_with_ledger(bal)
            runner._manage_open_positions(bal, realtime_px_override={"SOXL": 38.5})

        self.assertIsNone(runner._get_active_position())

    def test_step8_integration_e_sell_partial_1_of_4_then_chase_3_and_clear(self):
        """[Integration E] 1차 SELL 부분체결(4주 중 1주 체결 + 3주 취소) -> 2차 SELL 3주 체결 -> 총 4주 정확히 청산, 초과 매도 0건"""
        runner, broker = self.env.build_runner_and_broker()
        self.env.ledger["holdings"]["SOXL"] = {"qty": 4, "sell_avail": 4, "avg_px": 40.0, "cur_px": 38.5}
        runner._save_active_position({
            "symbol": "SOXL", "quantity": 4, "price": 40.0,
            "buy_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "order_no": "000010001", "tp_pct": 10.0, "sl_pct": 2.5,
        })

        def on_send(ledger, rec):
            if len(ledger["sent_orders"]) == 1:
                ledger["holdings"]["SOXL"]["qty"] = 3
                ledger["holdings"]["SOXL"]["sell_avail"] = 3
            else:
                ledger["holdings"].clear()
                runner._order_fills[rec["ord_no"]] = rec["quantity"]
                ledger["executions"].append({
                    "ord_dt": self.env.ny_compact, "ord_no": rec["ord_no"],
                    "stk_cd": "SOXL", "slby_tp": "1",
                    "ord_qty": str(rec["quantity"]), "cntr_qty": str(rec["quantity"]),
                    "cncl_qty": "0", "ord_remnq": "0",
                })
            return None

        def on_cancel(ledger, rec):
            ledger["open_orders"] = []
            row = {
                "ord_dt": self.env.ny_compact, "ord_no": rec["order_no"],
                "stk_cd": "SOXL", "slby_tp": "1",
                "ord_qty": "4", "cntr_qty": "1", "cncl_qty": "3", "ord_remnq": "0",
            }
            ledger["executions"] = [dict(row)]
            ledger["order_history"] = [dict(row)]
            return {"ok": True, "order_no": rec["order_no"], "cancel_ord_no": "000058180", "symbol": "SOXL"}

        self.env.ledger["on_send_order_hook"] = on_send
        self.env.ledger["on_cancel_order_hook"] = on_cancel

        with patch("core.circuit_breaker.CIRCUIT_STATE_FILE", self.env.cb_file), \
             patch("urllib.request.build_opener", side_effect=self.env.make_opener_factory()), \
             patch.object(config, "CANCEL_ORDER_WAIT_TIME", 0.005), \
             patch.object(config, "WS_FILL_POLL_INTERVAL", 0.005):
            runner._manage_open_positions(broker.get_overseas_stock_balance(force_refresh=True), realtime_px_override={"SOXL": 38.5})

        self.assertEqual(len(self.env.ledger["sent_orders"]), 2)
        self.assertEqual(self.env.ledger["sent_orders"][0]["quantity"], 4)
        self.assertEqual(self.env.ledger["sent_orders"][1]["quantity"], 3)
        self.assertIsNone(runner._get_active_position())

    def test_step2_resolve_order_ledger_status_all_6_states(self):
        """[STEP 2] resolve_order_ledger_status가 OPEN, PARTIALLY_FILLED, FILLED, CANCELED_ZERO_FILL, CANCELED_PARTIAL_FILL, UNKNOWN 6종 상태를 정확히 판정"""
        _, broker = self.env.build_runner_and_broker()
        # 1. OPEN
        st_open = broker.resolve_order_ledger_status(
            "0000101", "SOXL", "BUY", expected_qty=10,
            open_orders=[{"ord_no": "0000101", "stk_cd": "SOXL", "slby_tp": "2", "ord_qty": "10", "cntr_qty": "0", "cncl_qty": "0", "ord_remnq": "10"}],
            executions=[], order_history=[],
        )
        self.assertEqual(st_open["status"], "OPEN")

        # 2. PARTIALLY_FILLED (still open)
        st_pf = broker.resolve_order_ledger_status(
            "0000102", "SOXL", "BUY", expected_qty=10,
            open_orders=[{"ord_no": "0000102", "stk_cd": "SOXL", "slby_tp": "2", "ord_qty": "10", "cntr_qty": "4", "cncl_qty": "0", "ord_remnq": "6"}],
            executions=[], order_history=[],
        )
        self.assertEqual(st_pf["status"], "PARTIALLY_FILLED")

        # 3. FILLED
        st_fil = broker.resolve_order_ledger_status(
            "0000103", "SOXL", "SELL", expected_qty=4,
            open_orders=[],
            executions=[{"ord_no": "0000103", "stk_cd": "SOXL", "slby_tp": "1", "ord_qty": "4", "cntr_qty": "4", "cncl_qty": "0", "ord_remnq": "0"}],
            order_history=[],
        )
        self.assertEqual(st_fil["status"], "FILLED")

        # 4. CANCELED_ZERO_FILL (absent from ust21150, present in ust21180)
        st_cz = broker.resolve_order_ledger_status(
            "0000104", "SOXL", "SELL", expected_qty=4,
            open_orders=[],
            executions=[],
            order_history=[
                {"ord_no": "0000104", "orig_ord_no": "000000000", "stk_cd": "SOXL", "slby_tp": "1", "ord_qty": "4", "cntr_qty": "0", "cncl_qty": "4", "ord_remnq": "0"},
                {"ord_no": "0000105", "orig_ord_no": "0000104", "stk_cd": "SOXL", "slby_tp": "1", "ord_qty": "4", "cntr_qty": "0", "cncl_qty": "0", "ord_remnq": "0"},
            ],
        )
        self.assertEqual(st_cz["status"], "CANCELED_ZERO_FILL")

        # 5. CANCELED_PARTIAL_FILL
        st_cp = broker.resolve_order_ledger_status(
            "0000106", "SOXL", "BUY", expected_qty=10,
            open_orders=[],
            executions=[{"ord_no": "0000106", "stk_cd": "SOXL", "slby_tp": "2", "ord_qty": "10", "cntr_qty": "4", "cncl_qty": "6", "ord_remnq": "0"}],
            order_history=[],
        )
        self.assertEqual(st_cp["status"], "CANCELED_PARTIAL_FILL")

        # 6. UNKNOWN (missing from all ledgers)
        st_unk = broker.resolve_order_ledger_status(
            "0000999", "SOXL", "BUY", expected_qty=10,
            open_orders=[], executions=[], order_history=[],
        )
        self.assertEqual(st_unk["status"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main(verbosity=2)

