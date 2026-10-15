import os
import sys
import time
import threading
import json
import logging
import threading
from datetime import datetime, timedelta, time as dtime
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

# Windows  utf-8 ?
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from config import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    INITIAL_CAPITAL_KRW,
    DATA_DIR
)
from core.kiwoom_broker import KiwoomBroker, _mask_sensitive_log
from agents.dispatcher_agent import DispatcherAgent
from core.telegram_controller import TelegramController
from core.state_hub import StateHub
from core.data_lake import MarketDataLake
from core.shadow_sandbox import ShadowSandboxEngine
from core.shadow_tracker import ShadowTracker
from core.circuit_breaker import CircuitBreakerEngine
from core.system_logger import system_logger
from core.moe_orchestrator import MoEMetaOrchestrator
from core.live_experience_logger import LiveExperienceLogger

logger = logging.getLogger("LiveRunner")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s][LiveRunner] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

from core.market_calendar import USMarketCalendar

class KiwoomLiveRunner:

    def _redact_log(self, value: Any) -> Any:
        broker = getattr(self, "broker", None)
        return _mask_sensitive_log(value, (TELEGRAM_BOT_TOKEN,) + tuple(
            getattr(broker, name, "") for name in
            ("_access_token", "app_key", "app_secret", "account_no")
        ))

    def _acquire_order_lock(self) -> bool:
        with self._order_lock:
            if self._is_order_in_progress:
                return False
            self._is_order_in_progress = True
            return True

    def _dispatch_sell(self, **kwargs):
        if not self._acquire_order_lock():
            return
        kwargs['_already_locked'] = True
        import threading
        threading.Thread(target=self._execute_sell_with_10s_chase, kwargs=kwargs, daemon=True).start()
    """
    [??(Kiwoom) ????????  ??? ???]
    1.  ?? ? 100% ?: ? //?/ ?????? ???
    2.   ?AI ???? 100% ? (  ? )
    3. 3????????&   (: 1????/ : 100% ? ? ?)
    4. ??? ???  0.5?/ ? 0.3??  ?HTTP 429 ? ???
    """
    # ??[?????????? (Hard Rule #4: 3.0?????]
    ORDER_TIMEOUT_BUY_SEC: float = config.BUY_TIMEOUT_SEC   #  ?????(3?
    ORDER_TIMEOUT_SELL_SEC: float = config.SELL_TIMEOUT_SEC  #  ?????(3?

    def __init__(self, is_simulation: Optional[bool] = None):
        env_sim = os.getenv("KIWOOM_IS_SIMULATION", "1").strip()
        sim_flag = (env_sim == "1" or env_sim.lower() == "true") if is_simulation is None else bool(is_simulation)
        self.broker = KiwoomBroker(is_simulation=sim_flag)
        self.dispatcher = DispatcherAgent(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        self.controller = TelegramController(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        from core.telegram_notifier import TelegramNotifier
        self.notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        self.state_hub = StateHub()
        self.data_lake = MarketDataLake()
        self.shadow_sandbox = ShadowSandboxEngine()
        self.shadow_tracker = ShadowTracker()

        self.circuit_breaker = CircuitBreakerEngine()
        self.moe_orchestrator = MoEMetaOrchestrator(confidence_threshold=config.GBDT_CONFIDENCE_THRESHOLD, gbdt_threshold=config.GBDT_CONFIDENCE_THRESHOLD)
        self.enable_5m_sniper = False  # ? 15 ? ?? (? 77.55%, PF 3.98  / 5 ?? ? )
#         #self.sniper_engine = PowerHourSniper(
#             tp_pct=0.025,
#             sl_pct=0.0167,
#             time_stop_minutes=30,
#             confidence_threshold=0.55
#         )
        self.experience_logger = LiveExperienceLogger()

        # ??[?????????? (Hard Rule #4: 3.0????]
        self.order_timeout_buy_sec: float = float(os.getenv("LUMOS_ORDER_TIMEOUT_BUY_SEC", str(self.ORDER_TIMEOUT_BUY_SEC)))
        self.order_timeout_sell_sec: float = float(os.getenv("LUMOS_ORDER_TIMEOUT_SELL_SEC", str(self.ORDER_TIMEOUT_SELL_SEC)))
        
        # ??????WebSocket) ???? ? ?
        from core.kiwoom_ws_streamer import KiwoomWebSocketStreamer
        self.ws_streamer = KiwoomWebSocketStreamer(broker=self.broker)
        self.ws_streamer.register_callback(self._on_websocket_tick)
        self.ws_streamer.register_execution_callback(self._on_websocket_execution)

        self.is_running = False
        self._last_market_session = None
        self._active_position: Optional[Dict[str, Any]] = None  # ? ??????
        self._is_order_in_progress = False                      #   ???(AI ? 100% ?)
        self._order_lock = threading.Lock()
        self._order_fills: Dict[str, int] = {}                  # ????  ? {order_no: filled_qty}
        self._eod_liquidation_done = False

        # ??[? ? ?: Daily Circuit Breaker 3-Out Veto]
        self.daily_stoploss_count: int = 0
        self.daily_circuit_breaker_triggered: bool = False
        
        # ??[ ? ?15?? ???  ??
        self._last_veto_time: float = 0.0
        self._last_veto_state: bool = False
        self._last_defense_time: float = 0.0
        self._last_defense_state: bool = False
        self._last_briefing_time: float = 0.0
        self._last_moe_res: Optional[Dict[str, Any]] = None
        self._empty_balance_streak: int = 0
        self._unreconciled_holding_mismatch: Optional[Dict[str, Any]] = None
        self._unconfirmed_buy_order: Optional[Dict[str, Any]] = None

    def get_active_model_info(self, is_daemon_running: Optional[bool] = None) -> Dict[str, Any]:
        """단일 모델 메타데이터 공급원(MLFeatureEngine.get_active_model_info) 위임 조회"""
        daemon_flag = bool(self.is_running) if is_daemon_running is None else bool(is_daemon_running)
        if hasattr(self, "moe_orchestrator") and hasattr(self.moe_orchestrator, "gbdt_engine"):
            eng = self.moe_orchestrator.gbdt_engine
            if hasattr(eng, "get_active_model_info"):
                return eng.get_active_model_info(is_daemon_running=daemon_flag)
        return {
            "active_model_id": "UNKNOWN",
            "feature_schema": "UNKNOWN",
            "num_features": 0,
            "model_sha256": "",
            "sha256": "",
            "model_version": "NONE",
            "load_status": "BLOCKED",
            "fallback_status": "ALL_MODELS_FAILED_ENTRY_BLOCKED",
            "is_fallback": False,
            "is_ready": False,
            "operational_verdict": "BLOCKED",
        }

    def _can_query_broker_method(self, method_name: str) -> bool:
        broker = getattr(self, "broker", None)
        if broker is None or not callable(getattr(broker, method_name, None)):
            return False
        if method_name in getattr(broker, "__dict__", {}) or type(broker).__name__ != "KiwoomBroker":
            return True
        return not hasattr(getattr(broker, "send_order", None), "_mock_name")

    @staticmethod
    def _parse_int_strict(val: Any) -> Optional[int]:
        if val is None or isinstance(val, bool):
            return None
        try:
            s = str(val).replace(",", "").strip()
            if not s:
                return None
            f = float(s)
            if int(f) != f or int(f) < 0:
                return None
            return int(f)
        except Exception:
            return None

    @staticmethod
    def _norm_exec_side(item: Dict[str, Any]) -> str:
        raw_side = str(item.get("side") or "").strip().upper()
        if raw_side in ("BUY", "SELL"):
            return raw_side
        raw_tp = str(item.get("slby_tp") or item.get("sll_buy_tp_cd") or item.get("slby_tp_nm") or "").strip()
        if raw_tp in ("1", "매도") or "매도" in raw_tp:
            return "SELL"
        if raw_tp in ("2", "매수") or "매수" in raw_tp:
            return "BUY"
        return raw_side

    @classmethod
    def _parse_exec_filled_qty(cls, ex: Dict[str, Any]) -> Optional[int]:
        for key in ("tot_che_qty", "cntr_qty", "ccld_qty", "filled_qty", "che_qty"):
            if key in ex:
                return cls._parse_int_strict(ex.get(key))
        return cls._parse_int_strict(0)

    @staticmethod
    def _extract_open_orders_list(open_res: Any) -> Optional[List[Dict[str, Any]]]:
        if not isinstance(open_res, dict) or open_res.get("ok") is not True:
            return None
        orders = open_res.get("orders") if "orders" in open_res else open_res.get("open_orders")
        return orders if isinstance(orders, list) else None

    @staticmethod
    def _normalize_verified_date_str(val: Any) -> Optional[str]:
        if not isinstance(val, str):
            return None
        s = val.strip()
        try:
            if len(s) == 8 and s.isdigit():
                datetime.strptime(s, "%Y%m%d")
                return f"{s[:4]}-{s[4:6]}-{s[6:8]}"
            if len(s) == 10 and s[4] == "-" and s[7] == "-":
                datetime.strptime(s, "%Y-%m-%d")
                return s
        except Exception:
            return None
        return None

    @staticmethod
    def _normalize_verified_time_str(val: Any) -> Optional[str]:
        if not isinstance(val, str):
            return None
        s = val.strip()
        if not s:
            return None
        try:
            if len(s) == 8 and s[2] == ":" and s[5] == ":":
                datetime.strptime(s, "%H:%M:%S")
                return s
            if len(s) == 6 and s.isdigit():
                datetime.strptime(s, "%H%M%S")
                return f"{s[:2]}:{s[2:4]}:{s[4:6]}"
        except Exception:
            return None
        return None

    def _lookup_verified_buy_metadata(
        self,
        symbol: str,
        expected_qty: int,
        hist_executions: List[Dict[str, Any]],
        hist_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], Optional[str]]:
        sym_up = symbol.strip().upper()
        meta_date = None
        if isinstance(hist_metadata, dict):
            sd = self._normalize_verified_date_str(hist_metadata.get("start_date"))
            ed = self._normalize_verified_date_str(hist_metadata.get("end_date"))
            if sd and ed and sd == ed:
                meta_date = sd

        matched_ord_no = None
        matched_time = None
        order_filled_map: Dict[str, int] = {}
        order_time_map: Dict[str, str] = {}

        for item in hist_executions:
            if not isinstance(item, dict):
                continue
            if str(item.get("stk_cd") or item.get("symbol") or "").strip().upper() != sym_up:
                continue
            if self._norm_exec_side(item) != "BUY":
                continue
            fil_q = self._parse_exec_filled_qty(item)
            if fil_q is None or fil_q <= 0:
                continue
            ord_no = str(item.get("ord_no") or item.get("order_no") or "").strip()
            if not ord_no:
                return None, None
            if "tot_che_qty" in item and "cntr_qty" not in item:
                order_filled_map[ord_no] = max(order_filled_map.get(ord_no, 0), fil_q)
            else:
                order_filled_map[ord_no] = order_filled_map.get(ord_no, 0) + fil_q
            matched_ord_no = ord_no

            row_time = None
            bt = item.get("buy_time") or item.get("entry_time") or item.get("timestamp")
            if isinstance(bt, str) and len(bt.strip()) >= 19:
                cand_ts = bt.strip()[:19]
                try:
                    datetime.strptime(cand_ts, "%Y-%m-%d %H:%M:%S")
                    row_time = cand_ts
                except Exception:
                    row_time = None
            if row_time is None:
                row_date = (
                    self._normalize_verified_date_str(item.get("ord_dt"))
                    or self._normalize_verified_date_str(item.get("trade_date"))
                    or meta_date
                )
                if row_date:
                    for t_key in ("cntr_time", "cntr_tmd", "ord_time", "ord_tmd", "ord_resp_time"):
                        hms = self._normalize_verified_time_str(item.get(t_key))
                        if hms:
                            row_time = f"{row_date} {hms}"
                            break
            if row_time:
                order_time_map[ord_no] = row_time
                matched_time = row_time

        if not matched_ord_no:
            return None, None

        ord_filled_qty = order_filled_map.get(matched_ord_no, 0)
        exp_logger = getattr(self, "experience_logger", None)
        if exp_logger is not None and callable(getattr(exp_logger, "_get_connection", None)):
            try:
                conn = exp_logger._get_connection()
                try:
                    c = conn.cursor()
                    c.execute(
                        "SELECT timestamp, order_no, quantity FROM live_order_events WHERE symbol=? AND action='BUY' AND status='FILLED' AND order_no=? ORDER BY event_id DESC LIMIT 1",
                        (sym_up, matched_ord_no),
                    )
                    row = c.fetchone()
                    if row is not None:
                        ts_val = str(row["timestamp"] if isinstance(row, dict) or hasattr(row, "keys") else row[0]).strip()
                        ono_val = str(row["order_no"] if isinstance(row, dict) or hasattr(row, "keys") else row[1]).strip()
                        qty_val = self._parse_int_strict(row["quantity"] if isinstance(row, dict) or hasattr(row, "keys") else row[2])
                        if ono_val == matched_ord_no and qty_val is not None and qty_val in (expected_qty, ord_filled_qty) and len(ts_val) >= 19:
                            cand_db_ts = ts_val[:19]
                            try:
                                datetime.strptime(cand_db_ts, "%Y-%m-%d %H:%M:%S")
                                matched_time = cand_db_ts
                            except Exception:
                                pass
                finally:
                    conn.close()
            except Exception:
                pass
        return matched_time, matched_ord_no

    def _reconcile_active_position_with_ledger(self, stk_bal: Dict[str, Any]):
        if not isinstance(stk_bal, dict) or stk_bal.get("ok") is not True or not isinstance(stk_bal.get("holdings"), list):
            return
        valid_holdings = []
        for h in stk_bal.get("holdings", []):
            if not isinstance(h, dict):
                return
            sym = str(h.get("symbol") or h.get("stk_cd") or "").strip().upper()
            qty_raw = h.get("quantity") if "quantity" in h else h.get("poss_qty", h.get("sell_alowq", 0))
            qty = self._parse_int_strict(qty_raw)
            if qty is None:
                return
            try:
                avg_px = float(
                    str(
                        h.get("purchase_price")
                        or h.get("pchs_avg_pric")
                        or h.get("buy_price")
                        or h.get("avg_price")
                        or h.get("frgn_stk_book_uv")
                        or 0.0
                    ).replace(",", "").strip()
                )
            except Exception:
                avg_px = 0.0
            if sym and qty > 0:
                valid_holdings.append({"symbol": sym, "quantity": qty, "avg_price": avg_px, "raw": h})

        active_pos = self._get_active_position()

        # 1. 실계좌 보유 0주인 경우: 매수 직후 잔고 반영 지연 또는 단일 조회 결과만으로 active_position을 절대 삭제하지 않음
        if not valid_holdings:
            self._unreconciled_holding_mismatch = None
            if not active_pos:
                self._empty_balance_streak = 0
                return
            self._empty_balance_streak = getattr(self, "_empty_balance_streak", 0) + 1
            if not (self._can_query_broker_method("get_open_orders") and self._can_query_broker_method("get_daily_execution_history")):
                return
            open_res = self.broker.get_open_orders()
            open_orders_list = self._extract_open_orders_list(open_res)
            hist_res = self.broker.get_daily_execution_history()
            if (open_orders_list is None
                    or not isinstance(hist_res, dict) or hist_res.get("ok") is not True or not isinstance(hist_res.get("executions"), list)):
                return
            pos_sym = str(active_pos.get("symbol") or "").strip().upper()
            pos_qty = self._parse_int_strict(active_pos.get("quantity", active_pos.get("qty", 0))) or 0
            pos_ord_no = str(active_pos.get("order_no") or "").strip()
            for o_item in open_orders_list:
                if not isinstance(o_item, dict):
                    return
                o_sym = str(o_item.get("stk_cd") or o_item.get("symbol") or o_item.get("symb") or "").strip().upper()
                rem_q = self._parse_int_strict(o_item.get("nccs_qty") if "nccs_qty" in o_item else o_item.get("ord_remnq", o_item.get("remaining_quantity", 0)))
                if rem_q is None or (rem_q > 0 and (not o_sym or o_sym == pos_sym)):
                    return
            total_buy_filled = 0
            total_sell_filled = 0
            closing_sell_confirmed = False
            for ex in hist_res.get("executions", []):
                if not isinstance(ex, dict):
                    return
                ex_sym = str(ex.get("stk_cd") or ex.get("symbol") or "").strip().upper()
                if ex_sym != pos_sym:
                    continue
                fil_q = self._parse_exec_filled_qty(ex)
                rem_q = self._parse_int_strict(ex.get("nccs_qty") if "nccs_qty" in ex else ex.get("ord_remnq", 0))
                can_q = self._parse_int_strict(ex.get("cancelled_qty") if "cancelled_qty" in ex else ex.get("cncl_qty", 0))
                if fil_q is None or rem_q is None or rem_q > 0:
                    return
                side = self._norm_exec_side(ex)
                if side not in ("BUY", "SELL"):
                    if fil_q == 0 and can_q is not None and can_q > 0:
                        continue
                    return
                ex_ord_no = str(ex.get("ord_no") or ex.get("order_no") or "").strip()
                if side == "BUY":
                    total_buy_filled += fil_q
                elif side == "SELL":
                    total_sell_filled += fil_q
                    if fil_q > 0 and (not pos_ord_no or not ex_ord_no or ex_ord_no != pos_ord_no):
                        closing_sell_confirmed = True
            if closing_sell_confirmed and pos_qty > 0 and total_sell_filled >= pos_qty and total_sell_filled >= total_buy_filled:
                recheck_bal = self.broker.get_overseas_stock_balance(force_refresh=True)
                if isinstance(recheck_bal, dict) and recheck_bal.get("ok") is True and not recheck_bal.get("holdings"):
                    self._clear_active_position()
                    self._empty_balance_streak = 0
            return

        # 2. 실계좌 보유 주식이 존재하는 경우
        self._empty_balance_streak = 0
        if active_pos:
            pos_sym = str(active_pos.get("symbol") or "").strip().upper()
            matching = [h for h in valid_holdings if h["symbol"] == pos_sym]
            other = [h for h in valid_holdings if h["symbol"] != pos_sym]
            if other or not matching:
                self._unreconciled_holding_mismatch = {
                    "symbol": valid_holdings[0]["symbol"],
                    "quantity": valid_holdings[0]["quantity"],
                    "price": valid_holdings[0]["avg_price"],
                    "reason": "ACTIVE_POSITION_SYMBOL_MISMATCH",
                }
                logger.error(f"🚨 [원장-로컬 포지션 불일치] 로컬({pos_sym})과 실계좌 보유({valid_holdings[0]['symbol']})가 일치하지 않습니다.")
                return
            self._unreconciled_holding_mismatch = None
            h_info = matching[0]
            h_qty = h_info["quantity"]
            h_px = round(h_info["avg_price"], 2) if h_info["avg_price"] > 0 else float(active_pos.get("price", active_pos.get("buy_price", 0.0)))
            cur_pos_qty = self._parse_int_strict(active_pos.get("quantity", active_pos.get("qty", 0))) or 0
            cur_pos_px = float(active_pos.get("price", active_pos.get("buy_price", 0.0)))
            if h_qty > 0 and h_px > 0 and (cur_pos_qty != h_qty or abs(cur_pos_px - h_px) >= 0.01):
                active_pos["quantity"] = h_qty
                active_pos["qty"] = h_qty
                active_pos["price"] = h_px
                active_pos["buy_price"] = h_px
                tp_pct = float(active_pos.get("tp_pct", config.MAX_TP_PCT * 100.0))
                sl_pct = float(active_pos.get("sl_pct", config.SL_MIN_PCT * 100.0))
                active_pos["tp_px"] = round(h_px * (1.0 + tp_pct / 100.0), 2)
                active_pos["sl_px"] = round(h_px * (1.0 - sl_pct / 100.0), 2)
                self._save_active_position(active_pos)
            return

        # 3. 실계좌에 보유 주식이 있으나 로컬 active_position이 없는 경우: 체결원장 및 로컬 체결기록 대조
        if len(valid_holdings) > 1:
            self._unreconciled_holding_mismatch = {
                "symbol": ",".join(h["symbol"] for h in valid_holdings),
                "quantity": sum(h["quantity"] for h in valid_holdings),
                "reason": "MULTIPLE_SYMBOLS_HELD_WITHOUT_LOCAL_POSITION",
            }
            logger.error("🚨 [원장-로컬 포지션 불일치] 복수 종목이 실계좌에 보유 중이나 로컬 포지션이 없습니다.")
            return

        h_info = valid_holdings[0]
        h_sym = h_info["symbol"]
        h_qty = h_info["quantity"]
        h_px = round(h_info["avg_price"], 2)
        if h_px <= 0 or not self._can_query_broker_method("get_daily_execution_history"):
            self._unreconciled_holding_mismatch = {
                "symbol": h_sym,
                "quantity": h_qty,
                "price": h_px,
                "reason": "UNVERIFIED_PRICE_OR_EXECUTION_HISTORY_UNAVAILABLE",
            }
            logger.error(f"🚨 [원장-로컬 포지션 불일치] {h_sym} {h_qty}주 보유 중이나 체결원장/매수가를 확인할 수 없어 추정 포지션 생성을 거부합니다.")
            return

        hist_res = self.broker.get_daily_execution_history()
        if not isinstance(hist_res, dict) or hist_res.get("ok") is not True or not isinstance(hist_res.get("executions"), list):
            self._unreconciled_holding_mismatch = {
                "symbol": h_sym,
                "quantity": h_qty,
                "price": h_px,
                "reason": "EXECUTION_HISTORY_QUERY_FAILED",
            }
            logger.error(f"🚨 [원장-로컬 포지션 불일치] {h_sym} {h_qty}주 체결원장 조회 실패로 추정 포지션 생성을 거부합니다.")
            return

        net_buy_qty = 0
        for ex in hist_res.get("executions", []):
            if not isinstance(ex, dict):
                continue
            if str(ex.get("stk_cd") or ex.get("symbol") or "").strip().upper() != h_sym:
                continue
            fil_q = self._parse_exec_filled_qty(ex)
            rem_q = self._parse_int_strict(ex.get("nccs_qty") if "nccs_qty" in ex else ex.get("ord_remnq", 0))
            can_q = self._parse_int_strict(ex.get("cancelled_qty") if "cancelled_qty" in ex else ex.get("cncl_qty", 0))
            side = self._norm_exec_side(ex)
            if fil_q is None or rem_q is None or rem_q > 0:
                self._unreconciled_holding_mismatch = {
                    "symbol": h_sym,
                    "quantity": h_qty,
                    "price": h_px,
                    "reason": "UNRESOLVED_OR_AMBIGUOUS_EXECUTION_ROW",
                }
                logger.error(f"🚨 [원장-로컬 포지션 불일치] {h_sym} 미확정 체결 내역이 존재하여 포지션 복구를 중단합니다.")
                return
            if side not in ("BUY", "SELL"):
                if fil_q == 0 and can_q is not None and can_q > 0:
                    continue
                self._unreconciled_holding_mismatch = {
                    "symbol": h_sym,
                    "quantity": h_qty,
                    "price": h_px,
                    "reason": "UNRESOLVED_OR_AMBIGUOUS_EXECUTION_ROW",
                }
                logger.error(f"🚨 [원장-로컬 포지션 불일치] {h_sym} 미확정 체결 내역이 존재하여 포지션 복구를 중단합니다.")
                return
            if side == "BUY":
                net_buy_qty += fil_q
            elif side == "SELL":
                net_buy_qty -= fil_q

        verified_buy_time, verified_ord_no = self._lookup_verified_buy_metadata(
            h_sym, h_qty, hist_res.get("executions", []), hist_res.get("metadata")
        )
        if net_buy_qty != h_qty or not verified_buy_time:
            self._unreconciled_holding_mismatch = {
                "symbol": h_sym,
                "quantity": h_qty,
                "price": h_px,
                "net_buy_qty": net_buy_qty,
                "verified_buy_time": verified_buy_time,
                "reason": "UNVERIFIED_ENTRY_TIME_OR_QUANTITY_MISMATCH",
            }
            logger.error(
                f"🚨 [원장-로컬 포지션 불일치] 실계좌({h_sym} {h_qty}주 @ ${h_px:.2f})와 체결원장 순매수({net_buy_qty}주)/매수시각({verified_buy_time})을 학인할 수 없어 추정 포지션 생성을 거부합니다."
            )
            system_logger.log("ERROR", "LedgerMismatch", f"실계좌 {h_sym} {h_qty}주 보유 불일치 (매수시각/수량 미검증)")
            return

        atr_pct = self._get_5m_atr_pct(h_sym, h_px)
        sl_pct = max(config.SL_MIN_PCT * 100.0, min(config.SL_MAX_PCT * 100.0, atr_pct * config.SL_ATR_MULTIPLIER))
        tp_pct = config.MAX_TP_PCT * 100.0
        tp_px_calc = round(h_px * (1.0 + tp_pct / 100.0), 2)
        sl_px_calc = round(h_px * (1.0 - sl_pct / 100.0), 2)
        recovered_pos = {
            "symbol": h_sym,
            "quantity": h_qty,
            "qty": h_qty,
            "price": h_px,
            "buy_price": h_px,
            "buy_time": verified_buy_time,
            "order_no": verified_ord_no or "",
            "tp_pct": tp_pct,
            "sl_pct": sl_pct,
            "tp_px": tp_px_calc,
            "sl_px": sl_px_calc,
            "dynamic_tp_px": tp_px_calc,
            "dynamic_sl_px": sl_px_calc,
            "time_stop_minutes": config.TIME_STOP_MINUTES,
            "strategy_tag": "Lumos Ledger Recovered",
            "ref_price": h_px,
            "gbdt_confidence": 0.0,
            "cross_dir": "HOLD",
            "features": {},
            "high_price_during_hold": h_px,
            "low_price_during_hold": h_px,
            "buy_attempts": 1,
        }
        self._unreconciled_holding_mismatch = None
        self._save_active_position(recovered_pos)
        logger.info(f"✅ [원장-체결 대조 포지션 복구 완료] {h_sym} {h_qty}주 @ ${h_px:.2f} (매수시각: {verified_buy_time})")

    def _verify_pre_buy_ledger_clean(self, symbol: str) -> bool:
        if self._get_active_position() is not None:
            logger.warning("⚠️ [매수 중단] 이미 활성 포지션(active_position)이 존재합니다.")
            return False

        stk_bal = self.broker.get_overseas_stock_balance(force_refresh=True)
        if not isinstance(stk_bal, dict) or stk_bal.get("ok") is not True or not isinstance(stk_bal.get("holdings"), list):
            logger.error("❌ [매수 중단] 실계좌 잔고 조회 실패 또는 응답 불명확으로 신규 매수를 차단합니다.")
            return False

        for h in stk_bal.get("holdings", []):
            if not isinstance(h, dict):
                logger.error("❌ [매수 중단] 실계좌 잔고 항목 형식이 불명확하여 신규 매수를 차단합니다.")
                return False
            h_sym = str(h.get("symbol") or h.get("stk_cd") or "").strip().upper()
            h_qty = self._parse_int_strict(h.get("quantity") if "quantity" in h else h.get("poss_qty", h.get("sell_alowq", 0)))
            if h_qty is None:
                logger.error("❌ [매수 중단] 실계좌 보유 수량 파싱 실패로 신규 매수를 차단합니다.")
                return False
            if h_qty > 0:
                self._reconcile_active_position_with_ledger(stk_bal)
                logger.warning(f"⚠️ [매수 중단] 실계좌에 이미 보유 종목({h_sym} {h_qty}주)이 존재하여 신규 매수를 차단합니다.")
                return False

        self._unreconciled_holding_mismatch = None

        if self._can_query_broker_method("get_open_orders"):
            open_res = self.broker.get_open_orders()
            open_orders_list = self._extract_open_orders_list(open_res)
            if open_orders_list is None:
                logger.error("❌ [매수 중단] 미체결 주문 원장 조회 실패 또는 응답 불명확으로 신규 매수를 차단합니다.")
                return False
            for o_item in open_orders_list:
                if not isinstance(o_item, dict):
                    logger.error("❌ [매수 중단] 미체결 주문 항목 형식이 불명확하여 신규 매수를 차단합니다.")
                    return False
                o_sym = str(o_item.get("stk_cd") or o_item.get("symbol") or o_item.get("symb") or "").strip().upper()
                rem_q = self._parse_int_strict(
                    o_item.get("nccs_qty") if "nccs_qty" in o_item else o_item.get("ord_remnq", o_item.get("remaining_quantity", 0))
                )
                if rem_q is None:
                    logger.error("❌ [매수 중단] 미체결 주문 잔량 파싱 실패로 신규 매수를 차단합니다.")
                    return False
                if rem_q > 0 and (not o_sym or o_sym in (config.BASE_ASSET_LONG, config.BASE_ASSET_SHORT, symbol.strip().upper())):
                    logger.warning(f"⚠️ [매수 중단] 실계좌 미체결 주문({o_sym} {rem_q}주)이 존재하여 신규 매수를 차단합니다.")
                    return False

        if getattr(self, "_unconfirmed_buy_order", None) is not None:
            if not (self._can_query_broker_method("get_open_orders") and self._can_query_broker_method("get_daily_execution_history")):
                logger.error("❌ [매수 중단] 이전 매수 주문의 접수 여부를 원장으로 확인할 수 없어 신규 매수를 차단합니다.")
                return False
            hist_res = self.broker.get_daily_execution_history()
            if not isinstance(hist_res, dict) or hist_res.get("ok") is not True or not isinstance(hist_res.get("executions"), list):
                logger.error("❌ [매수 중단] 이전 미확정 매수 주문 대조를 위한 체결원장 조회 실패로 신규 매수를 차단합니다.")
                return False
            unconf_sym = str(self._unconfirmed_buy_order.get("symbol") or "").strip().upper()
            for ex in hist_res.get("executions", []):
                if not isinstance(ex, dict):
                    return False
                ex_sym = str(ex.get("stk_cd") or ex.get("symbol") or "").strip().upper()
                if ex_sym and ex_sym != unconf_sym:
                    continue
                side = self._norm_exec_side(ex)
                if side not in ("BUY", "SELL"):
                    return False
                if side == "BUY":
                    req_q = self._parse_int_strict(ex.get("ord_qty"))
                    fil_q = self._parse_exec_filled_qty(ex)
                    can_q = self._parse_int_strict(ex.get("cancelled_qty") if "cancelled_qty" in ex else ex.get("cncl_qty", 0))
                    rem_q = self._parse_int_strict(ex.get("nccs_qty") if "nccs_qty" in ex else ex.get("ord_remnq", 0))
                    if None in (req_q, fil_q, can_q, rem_q) or rem_q > 0 or fil_q + can_q != req_q or fil_q > 0:
                        logger.error(f"❌ [매수 중단] 이전 타임아웃/미확정 매수 주문({unconf_sym})이 원장에서 종결 확인되지 않아 중복 매수를 차단합니다.")
                        return False
            self._unconfirmed_buy_order = None

        return True

    def _resolve_pos_file(self) -> Path:
        custom = getattr(self, "pos_file_path", None)
        return Path(custom) if custom else (DATA_DIR / "active_position.json")

    def _save_active_position(self, pos_data: Dict[str, Any]):
        """???? ? ???? ???? ? ???"""
        self._active_position = pos_data
        self._empty_balance_streak = 0
        try:
            pos_file = self._resolve_pos_file()
            with open(pos_file, "w", encoding="utf-8") as f:
                json.dump(pos_data, f, ensure_ascii=False, indent=2)
            logger.info(f"? [???? ????] {pos_data.get('symbol')} {pos_data.get('quantity')}?@ ${pos_data.get('price')} (?: {pos_data.get('buy_time')})")
        except Exception as e:
            logger.warning(self._redact_log(f"???? ????: {e}"))

    def _get_active_position(self) -> Optional[Dict[str, Any]]:
        if self._active_position:
            return self._active_position
        pos_file = self._resolve_pos_file()
        if pos_file.exists():
            try:
                with open(pos_file, "r", encoding="utf-8") as f:
                    self._active_position = json.load(f)
                    return self._active_position
            except Exception:
                pass
        return None

    def _clear_active_position(self):
        self._active_position = None
        self._empty_balance_streak = 0
        try:
            pos_file = self._resolve_pos_file()
            if pos_file.exists():
                pos_file.unlink()
            logger.info("? [???? ?? 100% ? ??? ?")
        except Exception as e:
            logger.warning(self._redact_log(f"???? ?? ?: {e}"))

    def _get_current_ny_date(self) -> str:
        ny_tz = ZoneInfo("America/New_York")
        return datetime.now().astimezone().astimezone(ny_tz).strftime("%Y-%m-%d")

    def _on_websocket_execution(self, exec_data: Dict[str, Any]):
        """
        """
        logger.info(self._redact_log(f"? [WebSocket ?? ? ?]: {self._redact_log(exec_data)}"))
        if not exec_data:
            return

        #  values  ?   ?
        vals = exec_data.get("values", {}) if isinstance(exec_data.get("values"), dict) else {}
        
        def _get_val(*keys):
            for k in keys:
                if k in exec_data and exec_data[k] is not None and str(exec_data[k]).strip() != "":
                    return exec_data[k]
                if k in vals and vals[k] is not None and str(vals[k]).strip() != "":
                    return vals[k]
            return None

        ord_no_raw = _get_val("order_no", "ord_no", "odno", "orgn_odno", "ord_no1", "1")
        ord_no = str(ord_no_raw).strip() if ord_no_raw is not None else ""
        if not ord_no:
            logger.debug(self._redact_log(f" ?  ? ?: {self._redact_log(exec_data)}"))
            return

        def _to_int(val, default=0):
            try:
                if val is None or str(val).strip() == "":
                    return default
                return int(float(str(val).replace(",", "").strip()))
            except Exception:
                return default

        tot_che_qty = _to_int(_get_val(
            "tot_che_qty", "ft_tot_ccld_qty", "acc_filled_qty", "tot_qty", "tot_ccld_qty", "acc_qty", "14"
        ))
        che_qty = _to_int(_get_val(
            "filled_qty", "che_qty", "ft_ccld_qty", "qty", "ccld_qty", "che_qty1", "11"
        ))
        nccs_raw = _get_val("nccs_qty", "ft_rem_qty", "unfilled_qty", "rem_qty", "ord_rem_qty", "13")

        # ?  ? ???
        if tot_che_qty > 0:
            self._order_fills[ord_no] = tot_che_qty
            logger.info(f"? [ ? ? ]  #{ord_no} ??? : {tot_che_qty}?")
        elif che_qty > 0:
            self._order_fills[ord_no] = self._order_fills.get(ord_no, 0) + che_qty
            logger.info(f"? [ ? ??]  #{ord_no} ???: +{che_qty}?(?{self._order_fills[ord_no]}?")

        # ?0 ? ??100%  ?
        if nccs_raw is not None and _to_int(nccs_raw) == 0:
            current_val = self._order_fills.get(ord_no, che_qty or 1)
            self._order_fills[ord_no] = max(current_val, 1)
            logger.info(f"? [? 100%  ?]  #{ord_no} ???? 0?? (?: {self._order_fills[ord_no]}?")

    def _wait_for_fill(self, order_no: str, symbol: str, is_buy: bool, target_qty: int, timeout_sec: float = None, expected_total_qty: int = None) -> tuple[bool, int, int]:
        """
        """
        if timeout_sec is None:
            timeout_sec = self.order_timeout_buy_sec if is_buy else self.order_timeout_sell_sec
            
        if expected_total_qty is None:
            expected_total_qty = target_qty

        start_t = time.time()
        while time.time() - start_t < timeout_sec:
            ws_filled = self._order_fills.get(order_no, 0)
            if ws_filled >= target_qty:
                return True, ws_filled, 0
            time.sleep(config.WS_FILL_POLL_INTERVAL)

        # 타임아웃 발생시 실제 계좌 잔고를 조회하여 체결 여부 이중 확인
        stk_bal = self.broker.get_overseas_stock_balance(force_refresh=True)
        self._last_wait_fill_stk_bal = stk_bal if isinstance(stk_bal, dict) and stk_bal.get("ok") is True else None
        actual_qty = 0
        if stk_bal.get("ok"):
            for h in stk_bal.get("holdings", []):
                if str(h.get("symbol") or h.get("stk_cd") or "").strip().upper() == symbol:
                    actual_qty = int(float(str(h.get("quantity") or h.get("poss_qty") or 0).replace(",", "")))
                    break

            if is_buy:
                unfilled = max(0, expected_total_qty - actual_qty)
                return (actual_qty >= expected_total_qty), actual_qty, unfilled
            else:
                return (actual_qty <= 0), target_qty - actual_qty, actual_qty
        else:
            logger.error(f"❌ [잔고 조회 실패] {symbol} 체결 확인 중 잔고 조회 실패.")
            ws_filled = self._order_fills.get(order_no, 0)
            unfilled = max(0, target_qty - ws_filled)
            return (ws_filled >= target_qty), ws_filled, unfilled

    def _on_websocket_tick(self, symbol: str, price: float, extra: Dict[str, Any]):
        """
        """
        if self._is_order_in_progress:
            return

        try:
            mkt = USMarketCalendar.get_market_status()
            if not mkt.get("is_open"):
                return

            # 섀도우 트래커 실시간 검사 (독립적으로 실행, 에러 발생 시 실전 로직 보호를 위해 try-except로 완벽 격리)
            try:
                self.shadow_tracker.check_ticks(symbol, price)
            except Exception as e:
                system_logger.log("ERROR", "ShadowTracker", self._redact_log(f"웹소켓 틱 검사 중 에러: {e}"))

            active_pos = self._get_active_position()

            if not active_pos or active_pos.get("symbol") != symbol:
                return

            qty = int(active_pos.get("quantity", 0))
            buy_px = float(active_pos.get("price", 0.0))
            buy_time_str = active_pos.get("buy_time")

            if qty <= 0 or buy_px <= 0:
                return

            #  ?/? ?? (MFE / MAE ? ???
            cur_hi = active_pos.get("high_price_during_hold", price)
            cur_lo = active_pos.get("low_price_during_hold", price)
            if price > cur_hi:
                active_pos["high_price_during_hold"] = price
            if price < cur_lo:
                active_pos["low_price_during_hold"] = price

            pnl_pct = (price - buy_px) / buy_px

            tp_threshold = float(active_pos.get("tp_pct", config.MAX_TP_PCT * 100)) / 100.0 if active_pos.get("tp_pct") else config.MAX_TP_PCT
            sl_threshold = float(active_pos.get("sl_pct", config.SL_MIN_PCT * 100)) / 100.0 if active_pos.get("sl_pct") else config.SL_MIN_PCT
            time_stop_minutes = float(active_pos.get("time_stop_minutes", 90))

            # 1.  ?
            if pnl_pct >= tp_threshold:
                gain_pct = pnl_pct * 100
                logger.info(f"? [??? ????] {symbol} {qty}? ?  (?? +{gain_pct:.2f}%)")
                self._dispatch_sell(
                    symbol=symbol,
                    quantity=qty,
                    reason_desc=f"?  ? (+{gain_pct:.2f}%)",
                    buy_px=buy_px,
                    cur_px=price,
                    is_stoploss=False
                )

            # 2. ??
            elif pnl_pct <= -sl_threshold:
                loss_pct = abs(pnl_pct * 100)
                logger.info(f"? [??? ????] {symbol} {qty}? ?  (?? -{loss_pct:.2f}%)")
                self._dispatch_sell(
                    symbol=symbol,
                    quantity=qty,
                    reason_desc=f"? ?? (-{loss_pct:.2f}%)",
                    buy_px=buy_px,
                    cur_px=price,
                    is_stoploss=True
                )

            # 3. ????
            elif buy_time_str:
                try:
                    buy_dt = datetime.strptime(buy_time_str, "%Y-%m-%d %H:%M:%S")
                    elapsed_min = (datetime.now() - buy_dt).total_seconds() / 60.0
                    if elapsed_min >= time_stop_minutes:
                        cur_pnl = pnl_pct * 100
                        logger.info(f"??[??{time_stop_minutes:.0f}??????] {symbol} {qty}??  (: {elapsed_min:.0f}?| ?? {cur_pnl:+.2f}%)")
                        self._dispatch_sell(
                            symbol=symbol,
                            quantity=qty,
                            reason_desc=f"??{time_stop_minutes:.0f}?????? ({elapsed_min:.0f}?)",
                            buy_px=buy_px,
                            cur_px=price,
                            is_stoploss=False
                        )
                except Exception as te:
                    logger.debug(self._redact_log(f"???? ?: {te}"))

        except Exception as e:
            logger.debug(self._redact_log(f"WS ?? ?: {e}"))

    def _manage_open_positions(self, stk_bal: Dict[str, Any], realtime_px_override: Optional[Dict[str, float]] = None):
        """
        """
        holdings = stk_bal.get("holdings", [])
        if not holdings:
            return

        active_pos = self._get_active_position()
        tp_max_pct = float(active_pos.get("tp_pct", config.MAX_TP_PCT * 100)) / 100.0 if active_pos else config.MAX_TP_PCT
        sl_initial_pct = float(active_pos.get("sl_pct", config.SL_MIN_PCT * 100)) / 100.0 if active_pos else config.SL_MIN_PCT

        for h in holdings:
            sym = str(h.get("symbol") or h.get("stk_cd") or "").strip().upper()
            qty = int(float(str(h.get("quantity") or h.get("poss_qty") or h.get("sell_alowq") or 0).replace(",", "")))
            buy_px = float(str(h.get("purchase_price") or h.get("pchs_avg_pric") or h.get("buy_price") or h.get("avg_price") or h.get("frgn_stk_book_uv") or 0.0).replace(",", ""))

            if not sym or qty <= 0 or buy_px <= 0:
                continue

            if realtime_px_override and sym in realtime_px_override:
                cur_px = float(realtime_px_override[sym])
            else:
                quote = self.broker.get_stock_quote(sym)
                cur_px = float(quote.get("last_price", buy_px))

            if cur_px <= 0:
                cur_px = buy_px

            peak_high = buy_px
            if active_pos:
                peak_high = float(active_pos.get("high_price_during_hold", buy_px))
                cur_lo = float(active_pos.get("low_price_during_hold", cur_px))
                if cur_px > peak_high:
                    peak_high = cur_px
                    active_pos["high_price_during_hold"] = peak_high
                if cur_px < cur_lo:
                    active_pos["low_price_during_hold"] = cur_px

            is_trailing_active = False
            current_sl_px = buy_px * (1.0 - sl_initial_pct)
            trailing_trigger_px = buy_px * (1.0 + config.TRAILING_TRIGGER_PCT)
            
            if peak_high >= trailing_trigger_px:
                is_trailing_active = True
                t_initial = peak_high * (1.0 - config.TRAILING_DROP_PCT)
                current_sl_px = max(buy_px * 1.002, t_initial)
                
            if is_trailing_active:
                t = peak_high * (1.0 - config.TRAILING_DROP_PCT)
                if t > current_sl_px:
                    current_sl_px = t
            
            if active_pos:
                active_pos["dynamic_sl_px"] = current_sl_px
                self._save_active_position(active_pos)
                
            tp_max_px = buy_px * (1.0 + tp_max_pct)

            if cur_px >= tp_max_px:
                gain_pct = (cur_px / buy_px - 1.0) * 100.0
                self._dispatch_sell(
                    symbol=sym,
                    quantity=qty,
                    reason_desc=f"?  ? ? (+{gain_pct:.2f}%)",
                    buy_px=buy_px,
                    cur_px=cur_px,
                    is_stoploss=False
                )
            elif cur_px <= current_sl_px:
                loss_pct = (cur_px / buy_px - 1.0) * 100.0
                reason = f"??  ?? ? ({loss_pct:.2f}%)" if is_trailing_active else f"? ATR ? ? ({loss_pct:.2f}%)"
                self._dispatch_sell(
                    symbol=sym,
                    quantity=qty,
                    reason_desc=reason,
                    buy_px=buy_px,
                    cur_px=cur_px,
                    is_stoploss=True
                )
            elif active_pos and str(active_pos.get("symbol") or "").strip().upper() == sym and active_pos.get("buy_time"):
                try:
                    buy_dt = datetime.strptime(str(active_pos.get("buy_time")).strip()[:19], "%Y-%m-%d %H:%M:%S")
                    time_stop_minutes = float(active_pos.get("time_stop_minutes", config.TIME_STOP_MINUTES))
                    elapsed_min = (datetime.now() - buy_dt).total_seconds() / 60.0
                    if elapsed_min >= time_stop_minutes:
                        cur_pnl = (cur_px / buy_px - 1.0) * 100.0
                        logger.info(f"⏰ [루프 {time_stop_minutes:.0f}분 타임스탑] {sym} {qty}주 청산 발동 (경과: {elapsed_min:.0f}분 | 수익률: {cur_pnl:+.2f}%)")
                        self._dispatch_sell(
                            symbol=sym,
                            quantity=qty,
                            reason_desc=f"보유 {time_stop_minutes:.0f}분 타임스탑 ({elapsed_min:.0f}분)",
                            buy_px=buy_px,
                            cur_px=cur_px,
                            is_stoploss=False
                        )
                except Exception as te:
                    logger.debug(self._redact_log(f"루프 타임스탑 계산 예외: {te}"))

    def _sync_real_ledger_entry(self, symbol: str, default_price: float, default_qty: int) -> Tuple[float, int]:
        """
        """
        try:
            stk_bal = self.broker.get_overseas_stock_balance(force_refresh=True)
            if stk_bal.get("ok"):
                for h in stk_bal.get("holdings", []):
                    sym_h = str(h.get("symbol") or h.get("stk_cd") or "").strip().upper()
                    if sym_h == symbol.upper().strip():
                        p_qty = int(float(str(h.get("quantity") or h.get("poss_qty") or 0).replace(",", "")))
                        b_px = float(str(h.get("purchase_price") or h.get("pchs_avg_pric") or h.get("buy_price") or h.get("avg_price") or h.get("frgn_stk_book_uv") or 0.0).replace(",", ""))
                        if b_px > 0:
                            logger.info(f"? [??? ?? ??? ??: ${default_price:.2f} ??? ? ??: ${b_px:.2f} (? ?: {p_qty}?")
                            return round(b_px, 2), (p_qty if p_qty > 0 else default_qty)
        except Exception as e:
            logger.warning(self._redact_log(f"? ? ?? ???? (?? fallback ??): {e}"))
        return default_price, default_qty

    def _extract_verified_holding_from_balance(self, symbol: str, stk_bal: Any) -> Tuple[Optional[int], Optional[float], bool]:
        if not isinstance(stk_bal, dict) or stk_bal.get("ok") is not True or not isinstance(stk_bal.get("holdings"), list):
            return None, None, False
        sym_up = symbol.strip().upper()
        for h in stk_bal.get("holdings", []):
            if not isinstance(h, dict):
                return None, None, False
            if str(h.get("symbol") or h.get("stk_cd") or "").strip().upper() == sym_up:
                parsed_q = self._parse_int_strict(h.get("quantity") if "quantity" in h else h.get("poss_qty", h.get("sell_alowq", 0)))
                if parsed_q is None:
                    return None, None, False
                try:
                    avg_px = float(
                        str(
                            h.get("purchase_price")
                            or h.get("pchs_avg_pric")
                            or h.get("buy_price")
                            or h.get("avg_price")
                            or h.get("frgn_stk_book_uv")
                            or 0.0
                        ).replace(",", "").strip()
                    )
                except Exception:
                    avg_px = 0.0
                return parsed_q, (round(avg_px, 2) if avg_px > 0 else 0.0), True
        return 0, 0.0, True

    def _preserve_verified_partial_buy(
        self,
        symbol: str,
        order_no: str,
        buy_time_str: str,
        tp_pct: float,
        sl_pct: float,
        tp_px: float,
        sl_px: float,
        time_stop_min: int,
        strategy_tag: str,
        ref_price: float,
        moe_res: Dict[str, Any],
        buy_attempts: int = 1,
        stk_bal_snapshot: Optional[Dict[str, Any]] = None,
        fallback_stk_bal: Optional[Dict[str, Any]] = None,
        known_filled_hint: int = 0,
    ) -> bool:
        if self._get_active_position() is not None:
            return True
        sym_up = symbol.strip().upper()
        bal = stk_bal_snapshot
        if bal is None:
            try:
                bal = self.broker.get_overseas_stock_balance(force_refresh=True)
            except Exception:
                bal = None

        v_qty, v_px, bal_ok = self._extract_verified_holding_from_balance(sym_up, bal)
        used_fallback_bal = False
        if not bal_ok and fallback_stk_bal is not None:
            fb_qty, fb_px, fb_ok = self._extract_verified_holding_from_balance(sym_up, fallback_stk_bal)
            if fb_ok and fb_qty is not None and fb_qty > 0:
                v_qty, v_px, bal_ok = fb_qty, fb_px, True
                used_fallback_bal = True

        if not bal_ok and order_no and self._can_query_broker_method("get_daily_execution_history"):
            try:
                hist_fb = self.broker.get_daily_execution_history()
                if isinstance(hist_fb, dict) and hist_fb.get("ok") is True and isinstance(hist_fb.get("executions"), list):
                    h_sum_q = 0
                    h_cum_q = 0
                    h_use_cum = False
                    h_cost = 0.0
                    h_ambig = False
                    for ex in hist_fb.get("executions", []):
                        if not isinstance(ex, dict):
                            h_ambig = True
                            break
                        ex_no = str(ex.get("ord_no") or ex.get("order_no") or "").strip()
                        if ex_no != str(order_no).strip():
                            continue
                        ex_sym = str(ex.get("stk_cd") or ex.get("symbol") or "").strip().upper()
                        if ex_sym and ex_sym != sym_up:
                            h_ambig = True
                            break
                        if self._norm_exec_side(ex) != "BUY":
                            continue
                        fq = self._parse_exec_filled_qty(ex)
                        rq = self._parse_int_strict(ex.get("nccs_qty") if "nccs_qty" in ex else ex.get("ord_remnq", 0))
                        if fq is None or rq is None or rq > 0:
                            h_ambig = True
                            break
                        if fq <= 0:
                            continue
                        try:
                            epx = float(
                                str(
                                    ex.get("cntr_uv")
                                    or ex.get("che_pric")
                                    or ex.get("ft_ccld_unpr3")
                                    or ex.get("ccld_unpr")
                                    or ex.get("price")
                                    or 0.0
                                ).replace(",", "").strip()
                            )
                        except Exception:
                            epx = 0.0
                        if epx <= 0:
                            h_ambig = True
                            break
                        if "tot_che_qty" in ex and "cntr_qty" not in ex:
                            h_use_cum = True
                            if fq > h_cum_q:
                                h_cum_q = fq
                                h_cost = fq * epx
                        else:
                            h_sum_q += fq
                            h_cost += fq * epx
                    final_hq = h_cum_q if h_use_cum else h_sum_q
                    if not h_ambig and final_hq > 0 and h_cost > 0:
                        v_qty = final_hq
                        v_px = round(h_cost / final_hq, 2)
                        bal_ok = True
                        used_fallback_bal = True
            except Exception:
                pass

        if bal_ok and v_qty is not None and v_qty > 0 and v_px is not None and v_px > 0:
            tp_px_dyn = round(v_px * (1.0 + tp_pct / 100.0), 2)
            sl_px_dyn = round(v_px * (1.0 - sl_pct / 100.0), 2)
            filled_pos = {
                "symbol": symbol,
                "quantity": v_qty,
                "qty": v_qty,
                "price": v_px,
                "buy_price": v_px,
                "buy_time": buy_time_str,
                "order_no": order_no,
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
                "tp_px": tp_px_dyn,
                "sl_px": sl_px_dyn,
                "dynamic_tp_px": tp_px_dyn,
                "dynamic_sl_px": sl_px_dyn,
                "time_stop_minutes": time_stop_min,
                "strategy_tag": strategy_tag,
                "ref_price": ref_price,
                "gbdt_confidence": float(moe_res.get("gating_confidence", 0.60)),
                "cross_dir": moe_res.get("cross_dir", "HOLD"),
                "features": moe_res.get("features", {}),
                "active_model_id": moe_res.get("active_model_id"),
                "active_model_sha256": moe_res.get("active_model_sha256"),
                "active_model_version": moe_res.get("active_model_version"),
                "active_model_schema": moe_res.get("active_model_schema"),
                "active_model_num_features": moe_res.get("active_model_num_features"),
                "active_model_fallback_status": moe_res.get("active_model_fallback_status"),
                "active_model_is_fallback": moe_res.get("active_model_is_fallback"),
                "high_price_during_hold": v_px,
                "low_price_during_hold": v_px,
                "buy_attempts": buy_attempts,
            }
            self._save_active_position(filled_pos)
            try:
                self.experience_logger.record_order_event(
                    symbol=symbol,
                    action="BUY",
                    attempt=buy_attempts,
                    order_no=order_no,
                    price=v_px,
                    quantity=v_qty,
                    status="FILLED",
                    note=f"부분체결 포지션 보존 (원장 확정: {v_qty}주 @ ${v_px:.2f})",
                )
            except Exception:
                pass
            if used_fallback_bal:
                self._unreconciled_holding_mismatch = {
                    "symbol": sym_up,
                    "quantity": v_qty,
                    "price": v_px,
                    "reason": "BALANCE_REQUERY_FAILED_AFTER_PARTIAL_FILL",
                }
            logger.info(f"🛡️ [부분체결 포지션 보호] {symbol} {v_qty}주 @ ${v_px:.2f} (주문 #{order_no}) active_position 저장 완료")
            return True

        unverified_qty = (v_qty if (v_qty is not None and v_qty > 0) else 0) or max(0, int(known_filled_hint or 0))
        if not bal_ok or unverified_qty > 0:
            self._unreconciled_holding_mismatch = {
                "symbol": sym_up,
                "quantity": unverified_qty,
                "price": v_px or 0.0,
                "reason": "UNVERIFIED_PARTIAL_FILL_AFTER_BUY_ABORT",
            }
            logger.error(f"🚨 [부분체결 원장 불일치 보호] {sym_up} 수량/가격 미확정(추정 {unverified_qty}주)으로 임의 포지션 생성을 거부하고 신규 매수를 차단합니다.")
        return False

    def _get_5m_atr_pct(self, symbol: str, cur_px: float) -> float:
        try:
            df_5m = self.data_lake.load_candles(symbol, "5m")
            if not df_5m.empty and len(df_5m) >= 14:
                import pandas as pd
                tr = pd.concat([
                    df_5m['High'] - df_5m['Low'],
                    (df_5m['High'] - df_5m['Close'].shift()).abs(),
                    (df_5m['Low'] - df_5m['Close'].shift()).abs()
                ], axis=1).max(axis=1)
                atr_14 = tr.rolling(14).mean().iloc[-1]
                return (atr_14 / cur_px) * 100.0 if cur_px > 0 else 1.8
        except Exception:
            pass
        return 1.8

    def _execute_buy_with_10s_chase(
        self,
        symbol: str,
        target_qty: int,
        ref_price: float,
        moe_res: Dict[str, Any],
        targets: Dict[str, Any]
    ) -> bool:
        """
        """
        if not self._acquire_order_lock():
            return False

        ord_no_1 = ""
        ord_no_2 = ""
        buy_time_1 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filled_1 = 0
        already_filled_qty = 0
        stk_bal_wait_1 = None
        stk_bal_1 = None
        tp_pct = targets.get("tp_pct", 3.0) if isinstance(targets, dict) else 3.0
        sl_pct = targets.get("sl_pct", 2.0) if isinstance(targets, dict) else 2.0
        tp_px = targets.get("dynamic_tp_px", 0.0) if isinstance(targets, dict) else 0.0
        sl_px = targets.get("dynamic_sl_px", 0.0) if isinstance(targets, dict) else 0.0
        time_stop_min = targets.get("time_stop_minutes", 90) if isinstance(targets, dict) else 90
        strategy_tag = targets.get("strategy_tag", "Lumos V3") if isinstance(targets, dict) else "Lumos V3"

        try:
            if target_qty <= 0:
                logger.warning(f"⚠️ [매수 거부] 비정상 타겟 수량({target_qty}주)으로 진입이 취소되었습니다.")
                return False

            if not self._verify_pre_buy_ledger_clean(symbol):
                return False

            exp_name = moe_res.get("expert_desc", "MoE Gating")
            top_conf = float(moe_res.get("gating_confidence", 0.0)) * 100.0
            tp_px = targets["dynamic_tp_px"]
            sl_px = targets["dynamic_sl_px"]
            tp_pct = targets.get("tp_pct", 3.0)
            sl_pct = targets.get("sl_pct", 2.0)
            time_stop_min = targets.get("time_stop_minutes", 90)
            strategy_tag = targets.get("strategy_tag", "Lumos V3")
            atr_14 = targets.get("atr_14", 0.0)
            all_scores = moe_res.get("all_gating_confidences", {})

            # ----------------------------------------------------
            # 1. 1?  (Ask + $0.03)
            # ----------------------------------------------------
            cur_px = float(self.ws_streamer.get_latest_price(symbol, ref_price))
            if cur_px <= 0:
                cur_px = ref_price
            order_px_1 = round(cur_px + config.BUY_SLIPPAGE_ADJUST, 2)
            buy_time_1 = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._last_wait_fill_stk_bal = None

            logger.info(f"?? [1? ] {symbol} {target_qty}?@ ${order_px_1:.2f} ({self.order_timeout_buy_sec:.0f}? ???)")
            self._unconfirmed_buy_order = {
                "symbol": symbol,
                "target_qty": target_qty,
                "price": order_px_1,
                "timestamp": buy_time_1,
            }
            try:
                ord_res_1 = self.broker.send_order(symbol=symbol, order_type="BUY", quantity=target_qty, price=order_px_1)
            except Exception as send_err_1:
                err_s1 = str(send_err_1)
                if "BUY blocked:" in err_s1 or "Order deferred:" in err_s1:
                    self._unconfirmed_buy_order = None
                raise
            ord_no_1 = str(ord_res_1.get("order_no", "")).strip() if isinstance(ord_res_1, dict) else ""
            if not ord_no_1:
                raise RuntimeError("1차 매수 응답에 주문번호(order_no)가 없어 접수 여부를 확정할 수 없습니다.")
            self._unconfirmed_buy_order = None

            # ?  ?  (1? )
            mode_title = "? [?? ?]" if self.broker.is_simulation else "? [?? ??]"
            score_block = "\n".join([f"??**{k.upper()}:** `{v*100:.1f}??" for k, v in sorted(all_scores.items(), key=lambda x: x[1], reverse=True)])
            # Removed redundant buy_msg to avoid double telegram alerts
            system_logger.log("TRADE", "AutoExecution", f"?? [{self.broker.broker_name} {symbol} 1? ] {target_qty}?@ ${order_px_1:.2f} (???{top_conf:.1f}%)")

            # 1. ???(?  ??0ms   ?)
            is_filled_1, filled_1, unfilled_1 = self._wait_for_fill(
                order_no=ord_no_1,
                symbol=symbol,
                is_buy=True,
                target_qty=target_qty,
                timeout_sec=self.order_timeout_buy_sec
            )
            stk_bal_wait_1 = getattr(self, "_last_wait_fill_stk_bal", None)

            if is_filled_1:
                # 1??100%  ?! (??? ? ?? 100% ???
                real_buy_px, real_qty = self._sync_real_ledger_entry(symbol, order_px_1, target_qty)
                tp_px_dyn = round(real_buy_px * (1.0 + tp_pct / 100.0), 2) if real_buy_px > 0 else tp_px
                sl_px_dyn = round(real_buy_px * (1.0 - sl_pct / 100.0), 2) if real_buy_px > 0 else sl_px

                filled_pos = {
                    "symbol": symbol,
                    "quantity": real_qty,
                    "price": real_buy_px,
                    "buy_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "order_no": ord_no_1,
                    "tp_pct": tp_pct,
                    "sl_pct": sl_pct,
                    "tp_px": tp_px_dyn,
                    "sl_px": sl_px_dyn,
                    "time_stop_minutes": time_stop_min,
                    "strategy_tag": strategy_tag,
                    "ref_price": ref_price,
                    "gbdt_confidence": float(moe_res.get("gating_confidence", 0.60)),
                    "cross_dir": moe_res.get("cross_dir", "HOLD"),
                    "features": moe_res.get("features", {}),
                    "active_model_id": moe_res.get("active_model_id"),
                    "active_model_sha256": moe_res.get("active_model_sha256"),
                    "active_model_version": moe_res.get("active_model_version"),
                    "active_model_schema": moe_res.get("active_model_schema"),
                    "active_model_num_features": moe_res.get("active_model_num_features"),
                    "active_model_fallback_status": moe_res.get("active_model_fallback_status"),
                    "active_model_is_fallback": moe_res.get("active_model_is_fallback"),
                    "high_price_during_hold": real_buy_px,
                    "low_price_during_hold": real_buy_px,
                    "buy_attempts": 1
                }
                self._save_active_position(filled_pos)
                self.experience_logger.record_order_event(
                    symbol=symbol, action="BUY", attempt=1, order_no=ord_no_1,
                    price=real_buy_px, quantity=real_qty, status="FILLED",
                    note=f"1? 100%  ? (? ? : ${real_buy_px:.2f})"
                )
                logger.info(f"??[1? 100%  ?] {symbol} {real_qty}?@ ${real_buy_px:.2f} (? ?: {filled_pos['buy_time']})")
                try:
                    self.notifier.send_entry_alert(
                        ticker=symbol,
                        entry_price=real_buy_px,
                        qty=real_qty,
                        gbdt_prob=float(moe_res.get("gating_confidence", 0.60)),
                        cross_dir=moe_res.get("cross_dir", "HOLD"),
                        tp_price=tp_px_dyn,
                        sl_price=sl_px_dyn,
                        time_stop_minutes=time_stop_min,
                        strategy_tag=strategy_tag,
                        is_simulation=self.broker.is_simulation,
                        model_snapshot_info=moe_res,
                    )
                except Exception as te:
                    logger.warning(self._redact_log(f"텔레그램 매수 체결 알림 발송 에러(무시됨): {te}"))
                return True

            # ----------------------------------------------------
            # 2. 1?? & 2???(??1??)
            # ----------------------------------------------------
            logger.info(f"??[1? {self.order_timeout_buy_sec:.0f}??({unfilled_1}?] 1?  ?? ???1????")
            self.experience_logger.record_order_event(
                symbol=symbol, action="BUY", attempt=1, order_no=ord_no_1,
                price=order_px_1, quantity=unfilled_1, status="UNFILLED_TIMEOUT",
                note=f"1? {self.order_timeout_buy_sec:.0f}?? ??2???"
            )
            cancel_res_1 = None
            if ord_no_1:
                cancel_res_1 = self.broker.cancel_order(order_no=ord_no_1, symbol=symbol, quantity=unfilled_1)
            time.sleep(config.CANCEL_ORDER_WAIT_TIME)

            def _abort_second_buy_and_preserve(bal_snap: Optional[Dict[str, Any]] = None) -> bool:
                self._unconfirmed_buy_order = {"symbol": symbol, "order_no": ord_no_1, "target_qty": target_qty}
                self._preserve_verified_partial_buy(
                    symbol=symbol,
                    order_no=ord_no_1,
                    buy_time_str=buy_time_1,
                    tp_pct=tp_pct,
                    sl_pct=sl_pct,
                    tp_px=tp_px,
                    sl_px=sl_px,
                    time_stop_min=time_stop_min,
                    strategy_tag=strategy_tag,
                    ref_price=ref_price,
                    moe_res=moe_res,
                    buy_attempts=1,
                    stk_bal_snapshot=bal_snap,
                    fallback_stk_bal=stk_bal_wait_1,
                    known_filled_hint=max(filled_1, already_filled_qty),
                )
                return False

            if not ord_no_1 or (isinstance(cancel_res_1, dict) and cancel_res_1.get("ok") is False):
                logger.error(f"❌ [2차 매수 중단] 1차 매수 주문(#{ord_no_1}) 취소 요청이 실패하여 중복 발주를 막기 위해 2차 추격 매수를 중단합니다.")
                return _abort_second_buy_and_preserve()

            sym_up = symbol.strip().upper()
            cancel_ord_no_1 = str(cancel_res_1.get("cancel_ord_no") or "").strip() if isinstance(cancel_res_1, dict) else ""
            open_confirmed_cancel_qty = 0

            if self._can_query_broker_method("get_open_orders"):
                open_res_1 = self.broker.get_open_orders()
                open_orders_1 = self._extract_open_orders_list(open_res_1)
                if open_orders_1 is None:
                    logger.error(f"❌ [2차 매수 중단] 1차 매수 취소 후 미체결 원장 조회 실패로 2차 추격 매수를 중단합니다.")
                    return _abort_second_buy_and_preserve()
                for o_item in open_orders_1:
                    if not isinstance(o_item, dict):
                        return _abort_second_buy_and_preserve()
                    o_no = str(o_item.get("ord_no") or o_item.get("order_no") or "").strip()
                    o_orig = str(o_item.get("orig_ord_no") or o_item.get("original_order_no") or "").strip()
                    o_orig_clean = "" if o_orig in ("", "000000000") else o_orig
                    o_sym = str(o_item.get("stk_cd") or o_item.get("symbol") or o_item.get("symb") or "").strip().upper()
                    rem_q = self._parse_int_strict(
                        o_item.get("nccs_qty") if "nccs_qty" in o_item else o_item.get("ord_remnq", o_item.get("remaining_quantity", 0))
                    )
                    if cancel_ord_no_1 and o_no == cancel_ord_no_1 and o_orig_clean and o_orig_clean != ord_no_1:
                        logger.error(f"❌ [2차 매수 중단] 취소주문(#{cancel_ord_no_1})의 원주문번호({o_orig_clean})가 1차 매수(#{ord_no_1})와 불일치합니다.")
                        return _abort_second_buy_and_preserve()
                    if rem_q is None or (rem_q > 0 and (o_no in (ord_no_1, cancel_ord_no_1) or o_orig_clean == ord_no_1 or o_sym == sym_up)):
                        logger.warning(f"⚠️ [2차 매수 중단] 1차 매수 주문(#{ord_no_1}) 취소 미완료(미체결 잔량 {rem_q}주 잔류)로 2차 추격 매수를 중단합니다.")
                        return _abort_second_buy_and_preserve()
                    if o_no == ord_no_1 or o_orig_clean == ord_no_1 or (cancel_ord_no_1 and o_no == cancel_ord_no_1):
                        if o_sym and o_sym != sym_up:
                            return _abort_second_buy_and_preserve()
                        o_can = self._parse_int_strict(
                            o_item.get("cancelled_qty")
                            if "cancelled_qty" in o_item
                            else (o_item.get("cncl_cfrm_qty") if "cncl_cfrm_qty" in o_item else o_item.get("cncl_qty", 0))
                        )
                        if o_can is not None and o_can > 0 and rem_q == 0:
                            open_confirmed_cancel_qty = max(open_confirmed_cancel_qty, o_can)

            # ? ? ?? ???? 
            stk_bal_1 = self.broker.get_overseas_stock_balance(force_refresh=True)
            if not isinstance(stk_bal_1, dict) or stk_bal_1.get("ok") is not True or not isinstance(stk_bal_1.get("holdings"), list):
                logger.error("❌ [2차 매수 중단] 1차 매수 취소 후 잔고 조회 실패로 체결 수량을 확인할 수 없어 2차 추격 매수를 중단합니다.")
                return _abort_second_buy_and_preserve(stk_bal_1)

            already_filled_qty = 0
            for h in stk_bal_1.get("holdings", []):
                if not isinstance(h, dict):
                    return _abort_second_buy_and_preserve(stk_bal_1)
                if str(h.get("symbol") or h.get("stk_cd") or "").strip().upper() == sym_up:
                    parsed_q = self._parse_int_strict(h.get("quantity") if "quantity" in h else h.get("poss_qty", 0))
                    if parsed_q is None:
                        return _abort_second_buy_and_preserve(stk_bal_1)
                    already_filled_qty = parsed_q
                    break

            if self._can_query_broker_method("get_daily_execution_history"):
                hist_1 = self.broker.get_daily_execution_history()
                if not isinstance(hist_1, dict) or hist_1.get("ok") is not True or not isinstance(hist_1.get("executions"), list):
                    logger.error("❌ [2차 매수 중단] 1차 매수 취소 후 체결원장 조회 실패로 2차 추격 매수를 중단합니다.")
                    return _abort_second_buy_and_preserve(stk_bal_1)

                found_ord_1_evidence = False
                orig_sum_filled = 0
                orig_cum_filled = 0
                used_cum_field = False
                orig_max_cancel = 0
                child_cancel_qty = 0

                for ex in hist_1.get("executions", []):
                    if not isinstance(ex, dict):
                        return _abort_second_buy_and_preserve(stk_bal_1)
                    ex_no = str(ex.get("ord_no") or ex.get("order_no") or "").strip()
                    ex_orig = str(ex.get("orig_ord_no") or ex.get("original_order_no") or "").strip()
                    ex_orig_clean = "" if ex_orig in ("", "000000000") else ex_orig
                    ex_sym = str(ex.get("stk_cd") or ex.get("symbol") or "").strip().upper()
                    ex_cntr_tp = str(ex.get("ord_cntr_tp") or "").strip()

                    if cancel_ord_no_1 and cancel_ord_no_1 != ord_no_1 and ex_no == cancel_ord_no_1 and ex_orig_clean and ex_orig_clean != ord_no_1:
                        logger.error(f"❌ [2차 매수 중단] 체결원장 취소주문(#{cancel_ord_no_1})의 원주문번호({ex_orig_clean})가 1차 매수(#{ord_no_1})와 불일치합니다.")
                        return _abort_second_buy_and_preserve(stk_bal_1)
                    if ex_orig_clean == ord_no_1 and cancel_ord_no_1 and cancel_ord_no_1 != ord_no_1 and ex_no and ex_no not in (cancel_ord_no_1, ord_no_1):
                        logger.error(f"❌ [2차 매수 중단] 원주문(#{ord_no_1})에 연결된 취소주문번호({ex_no})가 응답 취소번호(#{cancel_ord_no_1})와 불일치합니다.")
                        return _abort_second_buy_and_preserve(stk_bal_1)

                    is_cancel_row = (
                        (ex_orig_clean == ord_no_1 and ex_no != ord_no_1)
                        or (bool(cancel_ord_no_1) and cancel_ord_no_1 != ord_no_1 and ex_no == cancel_ord_no_1)
                        or (ex_no == ord_no_1 and ex_cntr_tp == "12")
                    )
                    is_orig_row = (ex_no == ord_no_1 and not is_cancel_row)
                    if not is_cancel_row and not is_orig_row:
                        continue

                    found_ord_1_evidence = True
                    if ex_sym and ex_sym != sym_up:
                        logger.error(f"❌ [2차 매수 중단] 체결원장 주문(#{ex_no})의 종목({ex_sym})이 요청 종목({sym_up})과 불일치합니다.")
                        return _abort_second_buy_and_preserve(stk_bal_1)
                    if self._norm_exec_side(ex) == "SELL":
                        logger.error(f"❌ [2차 매수 중단] 체결원장 주문(#{ex_no})의 매수/매도 구분이 불일치합니다.")
                        return _abort_second_buy_and_preserve(stk_bal_1)

                    req_q = self._parse_int_strict(ex.get("ord_qty"))
                    fil_q = self._parse_exec_filled_qty(ex)
                    can_q = self._parse_int_strict(
                        ex.get("cancelled_qty")
                        if "cancelled_qty" in ex
                        else (ex.get("cncl_cfrm_qty") if "cncl_cfrm_qty" in ex else ex.get("cncl_qty", 0))
                    )
                    rem_q = self._parse_int_strict(ex.get("nccs_qty") if "nccs_qty" in ex else ex.get("ord_remnq", 0))
                    if None in (req_q, fil_q, can_q, rem_q) or rem_q > 0:
                        logger.error(f"❌ [2차 매수 중단] 체결원장에서 1차 매수(#{ord_no_1}) 관련 행의 수량 미확정 또는 잔량({rem_q}) 존재로 2차 추격 매수를 중단합니다.")
                        return _abort_second_buy_and_preserve(stk_bal_1)

                    if is_cancel_row:
                        if fil_q > 0 or can_q <= 0 or can_q != req_q:
                            logger.error(f"❌ [2차 매수 중단] 취소주문 행(#{ex_no})의 취소수량({can_q}/{req_q})이 모순되어 2차 추격 매수를 중단합니다.")
                            return _abort_second_buy_and_preserve(stk_bal_1)
                        child_cancel_qty += can_q
                    else:
                        if (ex_orig_clean and ex_orig_clean != ord_no_1) or req_q != target_qty:
                            logger.error(f"❌ [2차 매수 중단] 원주문(#{ord_no_1})의 주문수량({req_q} != {target_qty}) 또는 원주문번호가 모순되어 2차 추격 매수를 중단합니다.")
                            return _abort_second_buy_and_preserve(stk_bal_1)
                        if "tot_che_qty" in ex and "cntr_qty" not in ex:
                            used_cum_field = True
                            orig_cum_filled = max(orig_cum_filled, fil_q)
                        else:
                            orig_sum_filled += fil_q
                        orig_max_cancel = max(orig_max_cancel, can_q)

                hist_filled_qty = orig_cum_filled if used_cum_field else orig_sum_filled
                if orig_max_cancel > 0 and child_cancel_qty > 0 and orig_max_cancel != child_cancel_qty:
                    logger.error(f"❌ [2차 매수 중단] 원주문 취소수량({orig_max_cancel})과 별도 취소행 수량({child_cancel_qty})이 모순됩니다.")
                    return _abort_second_buy_and_preserve(stk_bal_1)

                confirmed_cancel_qty = max(orig_max_cancel, child_cancel_qty, open_confirmed_cancel_qty)
                if (
                    (not found_ord_1_evidence and open_confirmed_cancel_qty <= 0)
                    or (hist_filled_qty + confirmed_cancel_qty != target_qty)
                    or (hist_filled_qty != already_filled_qty)
                    or (target_qty > already_filled_qty and confirmed_cancel_qty <= 0)
                ):
                    logger.error(
                        f"❌ [2차 매수 중단] 체결원장에서 1차 매수(#{ord_no_1})의 체결({hist_filled_qty}주)/잔고({already_filled_qty}주)/취소({confirmed_cancel_qty}주) 수량이 확정되지 않아 2차 추격 매수를 중단합니다."
                    )
                    return _abort_second_buy_and_preserve(stk_bal_1)
                already_filled_qty = max(already_filled_qty, hist_filled_qty)

            remaining_qty = target_qty - already_filled_qty
            if remaining_qty <= 0:
                real_buy_px, real_qty = self._sync_real_ledger_entry(symbol, order_px_1, already_filled_qty)
                tp_px_dyn = round(real_buy_px * (1.0 + tp_pct / 100.0), 2) if real_buy_px > 0 else tp_px
                sl_px_dyn = round(real_buy_px * (1.0 - sl_pct / 100.0), 2) if real_buy_px > 0 else sl_px

                filled_pos = {
                    "symbol": symbol,
                    "quantity": real_qty,
                    "price": real_buy_px,
                    "buy_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "order_no": ord_no_1,
                    "tp_pct": tp_pct,
                    "sl_pct": sl_pct,
                    "tp_px": tp_px_dyn,
                    "sl_px": sl_px_dyn,
                    "time_stop_minutes": time_stop_min,
                    "strategy_tag": strategy_tag,
                    "ref_price": ref_price,
                    "gbdt_confidence": float(moe_res.get("gating_confidence", 0.60)),
                    "cross_dir": moe_res.get("cross_dir", "HOLD"),
                    "features": moe_res.get("features", {}),
                    "active_model_id": moe_res.get("active_model_id"),
                    "active_model_sha256": moe_res.get("active_model_sha256"),
                    "active_model_version": moe_res.get("active_model_version"),
                    "active_model_schema": moe_res.get("active_model_schema"),
                    "active_model_num_features": moe_res.get("active_model_num_features"),
                    "active_model_fallback_status": moe_res.get("active_model_fallback_status"),
                    "active_model_is_fallback": moe_res.get("active_model_is_fallback"),
                    "high_price_during_hold": real_buy_px,
                    "low_price_during_hold": real_buy_px,
                    "buy_attempts": 1
                }
                self._save_active_position(filled_pos)
                self.experience_logger.record_order_event(
                    symbol=symbol, action="BUY", attempt=1, order_no=ord_no_1,
                    price=real_buy_px, quantity=real_qty, status="FILLED",
                    note=f"1차 전량 체결 확인 (${real_buy_px:.2f})"
                )
                try:
                    self.notifier.send_entry_alert(
                        ticker=symbol,
                        entry_price=real_buy_px,
                        qty=real_qty,
                        gbdt_prob=float(moe_res.get("gating_confidence", 0.60)),
                        cross_dir=moe_res.get("cross_dir", "HOLD"),
                        tp_price=tp_px_dyn,
                        sl_price=sl_px_dyn,
                        time_stop_minutes=time_stop_min,
                        strategy_tag=strategy_tag,
                        is_simulation=self.broker.is_simulation,
                        model_snapshot_info=moe_res,
                    )
                except Exception as te:
                    logger.warning(self._redact_log(f"텔레그램 매수 체결 알림 발송 에러(무시됨): {te}"))
                return True

            # 2??? ???
            latest_px_2 = float(self.ws_streamer.get_latest_price(symbol, cur_px))
            order_px_2 = round(latest_px_2 + 0.03, 2)
            logger.info(f"??[2? ??(1/1)] {symbol} {remaining_qty}?@ ${order_px_2:.2f} ({self.order_timeout_buy_sec:.0f}? ???)")
            self._unconfirmed_buy_order = {
                "symbol": symbol,
                "target_qty": remaining_qty,
                "price": order_px_2,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            try:
                ord_res_2 = self.broker.send_order(symbol=symbol, order_type="BUY", quantity=remaining_qty, price=order_px_2)
            except Exception as send_err_2:
                err_s2 = str(send_err_2)
                if "BUY blocked:" in err_s2 or "Order deferred:" in err_s2:
                    self._unconfirmed_buy_order = None
                if already_filled_qty > 0:
                    self._preserve_verified_partial_buy(
                        symbol=symbol,
                        order_no=ord_no_1,
                        buy_time_str=buy_time_1,
                        tp_pct=tp_pct,
                        sl_pct=sl_pct,
                        tp_px=tp_px,
                        sl_px=sl_px,
                        time_stop_min=time_stop_min,
                        strategy_tag=strategy_tag,
                        ref_price=ref_price,
                        moe_res=moe_res,
                        buy_attempts=1,
                        stk_bal_snapshot=stk_bal_1,
                        fallback_stk_bal=stk_bal_wait_1,
                        known_filled_hint=already_filled_qty,
                    )
                raise
            ord_no_2 = str(ord_res_2.get("order_no", "")).strip() if isinstance(ord_res_2, dict) else ""
            if not ord_no_2:
                raise RuntimeError("2차 매수 응답에 주문번호(order_no)가 없어 접수 여부를 확정할 수 없습니다.")
            self._unconfirmed_buy_order = None
            system_logger.log("TRADE", "OrderChasing", f"??[ 2??? ?? {symbol} {remaining_qty}?@ ${order_px_2:.2f} (???1/1)")

            # 2. ???(?  ??0ms   ?)
            is_filled_2, filled_2, unfilled_2 = self._wait_for_fill(
                order_no=ord_no_2,
                symbol=symbol,
                is_buy=True,
                target_qty=remaining_qty,
                timeout_sec=self.order_timeout_buy_sec,
                expected_total_qty=target_qty
            )

            if is_filled_2:
                # 2??100%  ?! (??? ? ?? 100% ???
                total_qty = already_filled_qty + remaining_qty
                real_buy_px, real_qty = self._sync_real_ledger_entry(symbol, order_px_2, total_qty)
                tp_px_dyn = round(real_buy_px * (1.0 + tp_pct / 100.0), 2) if real_buy_px > 0 else tp_px
                sl_px_dyn = round(real_buy_px * (1.0 - sl_pct / 100.0), 2) if real_buy_px > 0 else sl_px

                filled_pos = {
                    "symbol": symbol,
                    "quantity": real_qty,
                    "price": real_buy_px,
                    "buy_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "order_no": ord_no_2,
                    "tp_pct": tp_pct,
                    "sl_pct": sl_pct,
                    "tp_px": tp_px_dyn,
                    "sl_px": sl_px_dyn,
                    "time_stop_minutes": time_stop_min,
                    "strategy_tag": strategy_tag,
                    "ref_price": ref_price,
                    "gbdt_confidence": float(moe_res.get("gating_confidence", 0.60)),
                    "cross_dir": moe_res.get("cross_dir", "HOLD"),
                    "features": moe_res.get("features", {}),
                    "active_model_id": moe_res.get("active_model_id"),
                    "active_model_sha256": moe_res.get("active_model_sha256"),
                    "active_model_version": moe_res.get("active_model_version"),
                    "active_model_schema": moe_res.get("active_model_schema"),
                    "active_model_num_features": moe_res.get("active_model_num_features"),
                    "active_model_fallback_status": moe_res.get("active_model_fallback_status"),
                    "active_model_is_fallback": moe_res.get("active_model_is_fallback"),
                    "high_price_during_hold": real_buy_px,
                    "low_price_during_hold": real_buy_px,
                    "buy_attempts": 2
                }
                self._save_active_position(filled_pos)
                self.experience_logger.record_order_event(
                    symbol=symbol, action="BUY", attempt=2, order_no=ord_no_2,
                    price=real_buy_px, quantity=remaining_qty, status="FILLED",
                    note=f"2? 100%  ? (? ? : ${real_buy_px:.2f})"
                )
                logger.info(f"??[2? 100%  ?] {symbol} {real_qty}?@ ${real_buy_px:.2f} (? ?: {filled_pos['buy_time']})")
                try:
                    self.notifier.send_entry_alert(
                        ticker=symbol,
                        entry_price=real_buy_px,
                        qty=real_qty,
                        gbdt_prob=float(moe_res.get("gating_confidence", 0.60)),
                        cross_dir=moe_res.get("cross_dir", "HOLD"),
                        tp_price=tp_px_dyn,
                        sl_price=sl_px_dyn,
                        time_stop_minutes=time_stop_min,
                        strategy_tag=strategy_tag,
                        is_simulation=self.broker.is_simulation,
                        model_snapshot_info=moe_res,
                    )
                except Exception as te:
                    logger.warning(self._redact_log(f"텔레그램 매수 체결 알림 발송 에러(무시됨): {te}"))
                return True

            # ----------------------------------------------------
            # 3. 2???????  ????100% 
            # ----------------------------------------------------
            logger.info(f"? [2? 10??({unfilled_2}?]  ?  ????100%  ??AI ???????")
            self.experience_logger.record_order_event(
                symbol=symbol, action="BUY", attempt=2, order_no=ord_no_2,
                price=order_px_2, quantity=unfilled_2, status="CASH_PRESERVED",
                note="2? ??  (??100%  ??AI ???????)"
            )
            cancel_res_2 = None
            if ord_no_2:
                cancel_res_2 = self.broker.cancel_order(order_no=ord_no_2, symbol=symbol, quantity=unfilled_2)
            time.sleep(config.CANCEL_ORDER_WAIT_TIME)

            # ? ???  ?
            stk_bal_final = self.broker.get_overseas_stock_balance(force_refresh=True)
            if not isinstance(stk_bal_final, dict) or stk_bal_final.get("ok") is not True or not isinstance(stk_bal_final.get("holdings"), list):
                logger.error("❌ [2차 매수 취소 후 잔고 조회 실패] 체결 여부를 확정할 수 없어 미확정 매수 상태로 기록합니다.")
                self._unconfirmed_buy_order = {"symbol": symbol, "order_no": ord_no_2, "target_qty": remaining_qty}
                saved_partial = self._preserve_verified_partial_buy(
                    symbol=symbol,
                    order_no=ord_no_1,
                    buy_time_str=buy_time_1,
                    tp_pct=tp_pct,
                    sl_pct=sl_pct,
                    tp_px=tp_px,
                    sl_px=sl_px,
                    time_stop_min=time_stop_min,
                    strategy_tag=strategy_tag,
                    ref_price=ref_price,
                    moe_res=moe_res,
                    buy_attempts=1,
                    stk_bal_snapshot=stk_bal_final,
                    fallback_stk_bal=stk_bal_1 or stk_bal_wait_1,
                    known_filled_hint=max(already_filled_qty, filled_1),
                )
                return saved_partial

            final_filled = already_filled_qty
            for h in stk_bal_final.get("holdings", []):
                if str(h.get("symbol") or h.get("stk_cd") or "").strip().upper() == symbol.strip().upper():
                    parsed_final = self._parse_int_strict(h.get("quantity") if "quantity" in h else h.get("poss_qty", 0))
                    if parsed_final is not None:
                        final_filled = max(final_filled, parsed_final)
                    break

            if isinstance(cancel_res_2, dict) and cancel_res_2.get("ok") is False:
                self._unconfirmed_buy_order = {"symbol": symbol, "order_no": ord_no_2, "target_qty": remaining_qty}

            if final_filled > 0:
                real_buy_px, real_qty = self._sync_real_ledger_entry(symbol, order_px_2, final_filled)
                tp_px_dyn = round(real_buy_px * (1.0 + tp_pct / 100.0), 2) if real_buy_px > 0 else tp_px
                sl_px_dyn = round(real_buy_px * (1.0 - sl_pct / 100.0), 2) if real_buy_px > 0 else sl_px

                filled_pos = {
                    "symbol": symbol,
                    "quantity": real_qty,
                    "price": real_buy_px,
                    "buy_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "order_no": ord_no_2 or ord_no_1,
                    "tp_pct": tp_pct,
                    "sl_pct": sl_pct,
                    "tp_px": tp_px_dyn,
                    "sl_px": sl_px_dyn,
                    "time_stop_minutes": time_stop_min,
                    "strategy_tag": strategy_tag,
                    "ref_price": ref_price,
                    "gbdt_confidence": float(moe_res.get("gating_confidence", 0.60)),
                    "cross_dir": moe_res.get("cross_dir", "HOLD"),
                    "features": moe_res.get("features", {}),
                    "active_model_id": moe_res.get("active_model_id"),
                    "active_model_sha256": moe_res.get("active_model_sha256"),
                    "active_model_version": moe_res.get("active_model_version"),
                    "active_model_schema": moe_res.get("active_model_schema"),
                    "active_model_num_features": moe_res.get("active_model_num_features"),
                    "active_model_fallback_status": moe_res.get("active_model_fallback_status"),
                    "active_model_is_fallback": moe_res.get("active_model_is_fallback"),
                    "high_price_during_hold": real_buy_px,
                    "low_price_during_hold": real_buy_px,
                    "buy_attempts": 2
                }
                self._save_active_position(filled_pos)
                self.experience_logger.record_order_event(
                    symbol=symbol, action="BUY", attempt=2, order_no=ord_no_2 or ord_no_1,
                    price=real_buy_px, quantity=real_qty, status="FILLED",
                    note=f"부분체결 확정 ({real_qty}주 @ ${real_buy_px:.2f})"
                )
                logger.info(f"? [?  ?] {symbol} {real_qty}??? (? ??: ${real_buy_px:.2f})")
                try:
                    self.notifier.send_entry_alert(
                        ticker=symbol,
                        entry_price=real_buy_px,
                        qty=real_qty,
                        gbdt_prob=float(moe_res.get("gating_confidence", 0.60)),
                        cross_dir=moe_res.get("cross_dir", "HOLD"),
                        tp_price=tp_px_dyn,
                        sl_price=sl_px_dyn,
                        time_stop_minutes=time_stop_min,
                        strategy_tag=strategy_tag,
                        is_simulation=self.broker.is_simulation,
                        model_snapshot_info=moe_res,
                    )
                except Exception as te:
                    logger.warning(self._redact_log(f"텔레그램 매수 체결 알림 발송 에러(무시됨): {te}"))
                return True
            else:
                self._clear_active_position()
                system_logger.log("TRADE", "OrderChasing", f"? [  ?] {symbol} ? ? (??100%  ??AI ???????)")
                return False

        except Exception as e:
            if ord_no_1 and self._get_active_position() is None:
                try:
                    self._preserve_verified_partial_buy(
                        symbol=symbol,
                        order_no=ord_no_2 or ord_no_1,
                        buy_time_str=buy_time_1,
                        tp_pct=tp_pct,
                        sl_pct=sl_pct,
                        tp_px=tp_px,
                        sl_px=sl_px,
                        time_stop_min=time_stop_min,
                        strategy_tag=strategy_tag,
                        ref_price=ref_price,
                        moe_res=moe_res if isinstance(moe_res, dict) else {},
                        buy_attempts=2 if ord_no_2 else 1,
                        stk_bal_snapshot=stk_bal_1,
                        fallback_stk_bal=stk_bal_wait_1,
                        known_filled_hint=max(already_filled_qty, filled_1),
                    )
                except Exception:
                    pass
            logger.error(self._redact_log(f"??   ?: {e}"))
            system_logger.log("ERROR", "BuyChase", self._redact_log(f"?   ? : {e}"))
            try:
                self.experience_logger.record_error_event("LiveRunner", "BUY_CHASE_EXCEPTION", "ERROR", self._redact_log(str(e)))
            except Exception:
                pass
            return False
        finally:
            with self._order_lock:
                self._is_order_in_progress = False
                logger.info("? [AI ???? ? ?] ? ? ????")

    def _execute_sell_with_10s_chase(
        self,
        symbol: str,
        quantity: int,
        reason_desc: str,
        buy_px: float = 0.0,
        cur_px: float = 0.0,
        is_stoploss: bool = False,
        is_market_order: bool = False,
        _already_locked: bool = False
    ) -> bool:
        """
        """
        if not _already_locked:
            if not self._acquire_order_lock():
                return False

        try:
            b_name = self.broker.broker_name
            loop_retry = 0

            # ? [??? ? ? (Hard Rule #2)]
            # ? ?? ??? ????(poss_qty) ?
            stk_bal_check = self.broker.get_overseas_stock_balance(force_refresh=True)
            if stk_bal_check.get("ok"):
                actual_qty = 0
                for h in stk_bal_check.get("holdings", []):
                    if str(h.get("symbol") or h.get("stk_cd") or "").strip().upper() == symbol:
                        actual_qty = int(float(str(h.get("quantity") or h.get("poss_qty") or 0).replace(",", "")))
                        break
                if actual_qty <= 0:
                    logger.info(f"? [? ? 0??] {symbol} ?? 100% ? ? ? ?? ? ?AI ???? ?")
                    self._reconcile_active_position_with_ledger(stk_bal_check)
                    return True
                quantity = min(quantity, actual_qty)
            else:
                logger.warning(f"초기 잔고 조회 실패. 전달받은 수량({quantity})으로 매도 진행.")

            while True:
                loop_retry += 1
                latest_px = float(self.ws_streamer.get_latest_price(symbol, cur_px))
                if latest_px <= 0:
                    latest_px = cur_px

                if is_market_order and not self.broker.is_simulation:
                    # {config.PHASE_EOD_CLEAR} EOD ?? 0%  ?: ?? ?? ?(03, price=0.0)  
                    sell_px = 0.0
                    order_label = "? ? ?(Market Order)"
                else:
                    # ? ? ? (? ?? ? ?? ? ): Bid - $0.05 ????
                    sell_px = max(0.01, round(latest_px - 0.05, 2)) if latest_px > 0 else 0.0
                    order_label = f"?  ??(${sell_px:.2f})"

                logger.info(f"? [ ?  ({loop_retry}?)] {symbol} {quantity}?| {order_label} | ?: {reason_desc} ({self.order_timeout_sell_sec:.0f}? ??")
                s_res = self.broker.send_order(symbol=symbol, order_type="SELL", quantity=quantity, price=sell_px)
                s_ord_no = str(s_res.get("order_no", "")).strip()

                # ? 3? ??(?  ??0ms   ?)
                is_sold, sold_qty, rem_qty = self._wait_for_fill(
                    order_no=s_ord_no,
                    symbol=symbol,
                    is_buy=False,
                    target_qty=quantity,
                    timeout_sec=self.order_timeout_sell_sec
                )

                if is_sold:
                    # 100% ? ? ?!
                    active_pos = self._get_active_position() or {}
                    self._clear_active_position()
                    final_pnl_pct = ((latest_px - buy_px) / buy_px * 100) if buy_px > 0 else 0.0
                    
                    try:
                        self.notifier.send_exit_alert(
                            ticker=symbol,
                            entry_price=buy_px,
                            exit_price=latest_px,
                            exit_reason=reason_desc,
                            qty=sold_qty,
                            is_simulation=self.broker.is_simulation,
                            model_snapshot_info=active_pos or self.get_active_model_info(),
                        )
                    except Exception as te:
                        logger.error(self._redact_log(f"텔레그램 매도 알림 1 발송 에러: {te}"))

                    # ? [AI ?? ?? ???Dual CSV+DB) ? ?]
                    high_px = active_pos.get("high_price_during_hold", max(buy_px, latest_px))
                    low_px = active_pos.get("low_price_during_hold", min(buy_px, latest_px))
                    mfe_pct = round(((high_px - buy_px) / buy_px) * 100, 2) if buy_px > 0 else 0.0
                    mae_pct = round(((low_px - buy_px) / buy_px) * 100, 2) if buy_px > 0 else 0.0
                    try:
                        self.experience_logger.record_trade({
                            "mode": self.broker.mode_str,
                            "symbol": symbol,
                            "entry_time": active_pos.get("buy_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                            "exit_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "intended_entry_price": active_pos.get("ref_price", buy_px),
                            "actual_entry_price": buy_px,
                            "intended_exit_price": sell_px if sell_px > 0 else latest_px,
                            "actual_exit_price": latest_px,
                            "quantity": quantity,
                            "pnl_pct": final_pnl_pct,
                            "pnl_usd": round((latest_px - buy_px) * quantity, 2),
                            "exit_reason": reason_desc,
                            "mfe_pct": mfe_pct,
                            "mae_pct": mae_pct,
                            "gbdt_confidence": active_pos.get("gbdt_confidence", 0.0),
                            "cross_dir": active_pos.get("cross_dir", "HOLD"),
                            "buy_attempts": active_pos.get("buy_attempts", 1),
                            "sell_attempts": loop_retry,
                            "features": active_pos.get("features", {})
                        })
                        self.experience_logger.record_order_event(
                            symbol=symbol, action="SELL", attempt=loop_retry, order_no=s_ord_no,
                            price=sell_px, quantity=quantity, status="FILLED", note=f"? ?: {reason_desc}"
                        )
                    except Exception as le:
                        logger.debug(self._redact_log(f"  ? ? (): {le}"))

                    mode_title = "? [?? ?]" if self.broker.is_simulation else "? [?? ??]"
                    system_logger.log("TRADE", "Liquidation", f"? [{mode_title} 100% ? ? ?] {symbol} {quantity}? | ?: {reason_desc} |  ?: {final_pnl_pct:+.2f}%")

                    try:
                        pos_model_id = active_pos.get("active_model_id") or self.get_active_model_info().get("active_model_id", "UNKNOWN")
                        pos_model_ver = active_pos.get("active_model_version") or str(active_pos.get("active_model_sha256") or "")[:12] or "unknown"
                        sell_msg = (
                            f"{mode_title} 100% 매도 청산 완료\n"
                            f"🚨 **브로커:** `{self.broker.broker_name} {self.broker.mode_str}`\n"
                            f"🧠 **진입 모델:** `{pos_model_id}` (`{pos_model_ver}`)\n"
                            f"💡 **사유:** `{reason_desc}`\n"
                            f"📊 **종목/수량:** `{symbol} {quantity:,}주`\n"
                            f"💰 **체결가:** `${latest_px:.2f}` (최종 수익률: {final_pnl_pct:+.2f}%)\n"
                            f"🛡️ **사후 관리:** `100% 현금화 완료 (AI MoE 새 진입 대기)`"
                        )
                        self.dispatcher.send_telegram_message(sell_msg)
                    except Exception as te2:
                        logger.error(self._redact_log(f"텔레그램 매도 알림 2 발송 에러: {te2}"))
                    return True

                if s_ord_no:
                    self.broker.cancel_order(order_no=s_ord_no, symbol=symbol, quantity=rem_qty)
                    time.sleep(config.CANCEL_ORDER_WAIT_TIME)
                
                stk_bal_recheck = self.broker.get_overseas_stock_balance(force_refresh=True)
                if stk_bal_recheck.get("ok"):
                    new_actual_qty = 0
                    for h in stk_bal_recheck.get("holdings", []):
                        if str(h.get("symbol") or h.get("stk_cd") or "").strip().upper() == symbol:
                            new_actual_qty = int(float(str(h.get("quantity") or h.get("poss_qty") or 0).replace(",", "")))
                            break
                    if new_actual_qty <= 0:
                        logger.info(f"🟢 [매도 완결] {symbol} 잔고 0주 확인. 추격 종료.")
                        self._clear_active_position()
                        return True
                    quantity = new_actual_qty
                else:
                    logger.warning(f"잔고 재조회 실패. 기존 수량({quantity})으로 매도 추격 계속.")

        except Exception as e:
            logger.error(self._redact_log(f"매도 추격 주문 에러: {e}"))
            system_logger.log("ERROR", "SellChase", self._redact_log(f"매도 추격 에러: {e}"))
            try:
                self.experience_logger.record_error_event("LiveRunner", "SELL_CHASE_EXCEPTION", "ERROR", self._redact_log(str(e)))
            except Exception:
                pass
            return False
        finally:
            with self._order_lock:
                self._is_order_in_progress = False
                logger.info("🔒 [AI 포지션 청산 완료] 주문 락 해제완료")
    # ??[???????? ]
    _execute_buy_with_chase = _execute_buy_with_10s_chase
    _execute_sell_with_chase = _execute_sell_with_10s_chase

    def _emit_model_state_alerts_from_eval(self, moe_res: Dict[str, Any]):
        """
        Inspects model snapshot metadata from a 15m evaluation result and emits
        deduplicated Telegram alerts for hot-reload, Champion fallback, or BLOCKED state.
        """
        try:
            load_status = str(moe_res.get("active_model_load_status") or "")
            is_fallback = bool(moe_res.get("active_model_is_fallback", False))
            consistency_ok = bool(moe_res.get("model_consistency_ok", True))
            consistency_reason = str(moe_res.get("model_consistency_reason") or "")
            hot_reloaded = bool(moe_res.get("model_hot_reloaded", False))
            prev_sha = str(moe_res.get("previous_model_sha256_before_eval") or "")

            if load_status == "BLOCKED" or not consistency_ok:
                self.notifier.send_model_state_alert(
                    event_type="BLOCKED",
                    model_info=moe_res,
                    reason=consistency_reason or "모든 모델 로드 실패 또는 manifest/메모리 불일치로 신규 매수가 차단되었습니다.",
                )
            elif is_fallback or load_status == "FALLBACK":
                self.notifier.send_model_state_alert(
                    event_type="FALLBACK_TO_CHAMPION",
                    model_info=moe_res,
                    reason="Primary(B_Normalized_72) 모델 비정상 감지로 Champion 폴백 모델이 활성화되었습니다.",
                )
            elif hot_reloaded:
                self.notifier.send_model_state_alert(
                    event_type="MODEL_HOT_RELOADED",
                    model_info=moe_res,
                    previous_model_sha256=prev_sha,
                    reason="자동 재학습/매니페스트 갱신이 감지되어 최신 모델 스냅샷으로 무중단 핫리로드되었습니다.",
                )
        except Exception as e:
            logger.warning(self._redact_log(f"모델 상태 알림 발송 중 에러(무시됨): {e}"))

    def evaluate_15m_cycle_once(
        self,
        allow_order_execution: bool = True,
        sync_intraday: bool = True,
        now_dt: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Executes a single 15-minute AI evaluation cycle (used by _market_execution_loop
        and by orderless live-entrypoint verification tests).
        """
        now_dt = now_dt or datetime.now()
        ny_dt = now_dt.astimezone(ZoneInfo("America/New_York"))
        active_pos = self._get_active_position()

        if sync_intraday:
            try:
                shadow_live_p = {sym: self.ws_streamer.get_latest_price(sym, 0.0) for sym in config.TRADE_SYMBOLS}
                self.shadow_tracker.increment_candle(exit_prices=shadow_live_p)
            except Exception as e:
                system_logger.log("ERROR", "ShadowTracker", self._redact_log(f"캔들 업데이트 중 에러: {e}"))
            self.data_lake.sync_live_intraday_candles(config.ALL_SYMBOLS)

        df_15m = self.data_lake.load_candles(config.BASE_ASSET_LONG, "15m")
        live_prices = {sym: self.ws_streamer.get_latest_price(sym, 0.0) for sym in config.ALL_SYMBOLS}

        moe_res = self.moe_orchestrator.evaluate_dual_filter_signal(
            df_candle_15m=df_15m,
            live_prices=live_prices,
        )
        self._last_moe_res = moe_res

        # Emit any model state transition alert (HOT_RELOADED / FALLBACK / BLOCKED)
        self._emit_model_state_alerts_from_eval(moe_res)

        conf = float(moe_res.get("gating_confidence", 0.0)) * 100.0
        is_appr = bool(moe_res.get("is_approved", False))
        direction = str(moe_res.get("direction", "NONE"))
        gbdt_probs = moe_res.get("gbdt_probs", {"LONG": 0.0, "SHORT": 0.0, "NONE": 0.0})
        p_long = float(gbdt_probs.get("LONG", 0.0)) * 100.0
        p_short = float(gbdt_probs.get("SHORT", 0.0)) * 100.0

        # Send 15m briefing using the exact model snapshot embedded in moe_res
        briefing_msg = self.notifier.compose_signal_briefing_alert(
            moe_res=moe_res,
            ny_time_str=ny_dt.strftime("%H:%M"),
            kst_time_str=now_dt.strftime("%H:%M"),
            has_active_pos=bool(active_pos),
        )
        try:
            self.dispatcher.send_telegram_message(briefing_msg)
        except Exception as te:
            logger.warning(self._redact_log(f"텔레그램 15분 브리핑 발송 에러(무시됨): {te}"))

        # If Cross-Asset Veto blocked an otherwise high-confidence signal, send Veto alert with snapshot
        if bool(moe_res.get("is_cross_veto", False)):
            try:
                raw_dir = "LONG" if p_long >= p_short else "SHORT"
                veto_ticker = config.BASE_ASSET_LONG if raw_dir == "LONG" else config.BASE_ASSET_SHORT
                self.notifier.send_veto_alert(
                    ticker=veto_ticker,
                    gbdt_prob=float(moe_res.get("gating_confidence", 0.0)),
                    veto_reason=str(moe_res.get("veto_reason", "Cross-Asset Veto")),
                    model_snapshot_info=moe_res,
                )
            except Exception as vte:
                logger.warning(self._redact_log(f"텔레그램 Veto 알림 발송 에러(무시됨): {vte}"))

        system_logger.log(
            "INFO",
            "AI",
            f"15분 정기 브리핑 완료 (확신도: {conf:.1f}%, 모델: {moe_res.get('active_model_id')}[{moe_res.get('active_model_version')}])",
        )

        # 1. 쉐도우 트래커(가상 매매): allow_order_execution=True 일 때만 DB에 기록
        if allow_order_execution:
            shadow_dir = "LONG" if p_long >= p_short else "SHORT"
            shadow_sym = config.BASE_ASSET_LONG if shadow_dir == "LONG" else config.BASE_ASSET_SHORT
            shadow_px = live_prices.get(shadow_sym, 0.0)
            try:
                shadow_atr = self._get_5m_atr_pct(shadow_sym, shadow_px) if shadow_px > 0 else 1.8
                self.shadow_tracker.try_enter_shadow_trade(
                    symbol=shadow_sym,
                    price=shadow_px,
                    current_time=ny_dt,
                    atr_pct=(shadow_atr / 100.0),
                )
            except Exception as e:
                system_logger.log("ERROR", "ShadowTracker", self._redact_log(f"가상 진입 추가 중 에러: {e}"))

        # 2. 실전 매매 사전 판단 및 (allow_order_execution=True 시) 주문 실행
        order_attempted = False
        pre_order_decision = "WAIT_HOLD"
        winner_sym = None
        order_qty = 0

        if not bool(moe_res.get("model_consistency_ok", True)) or moe_res.get("active_model_load_status") == "BLOCKED":
            pre_order_decision = "BLOCKED_BY_MODEL_GUARD"
        elif is_appr and direction in ["LONG", "SHORT"]:
            winner_sym = config.BASE_ASSET_LONG if direction == "LONG" else config.BASE_ASSET_SHORT
            cur_px = live_prices.get(winner_sym, 0.0)
            if active_pos:
                pre_order_decision = "SKIP_EXISTING_POSITION"
            elif cur_px <= 0:
                pre_order_decision = "SKIP_INVALID_PRICE"
            elif not allow_order_execution:
                pre_order_decision = f"READY_TO_BUY_{winner_sym}_ORDER_SUPPRESSED"
            else:
                conn_res = self.broker.test_connection()
                if not conn_res.get("ok", True):
                    pre_order_decision = "DEFERRED_BALANCE_QUERY_FAILED"
                    system_logger.log("WARNING", "Trade", f"[B-02 Pre-Buy Guard] 예수금 조회 실패로 {winner_sym} 진입 보류")
                else:
                    usd_avail = float(conn_res.get("usd_order_available", 0.0))
                    order_qty = int((usd_avail * config.MAX_ALLOCATION_RATIO) / (cur_px + config.QTY_CALC_BUFFER))
                    if order_qty > 0:
                        pre_order_decision = f"EXECUTE_BUY_{winner_sym}"
                        system_logger.log("TRADE", "AI", f"V3 AI 신호 발생! {winner_sym} {order_qty}주 매수")
                        atr_pct = self._get_5m_atr_pct(winner_sym, cur_px)
                        sl_pct = max(config.SL_MIN_PCT * 100.0, min(config.SL_MAX_PCT * 100.0, atr_pct * config.SL_ATR_MULTIPLIER))
                        tp_pct = config.MAX_TP_PCT * 100.0
                        targets = {
                            "dynamic_tp_px": round(cur_px * (1.0 + tp_pct / 100.0), 2),
                            "dynamic_sl_px": round(cur_px * (1.0 - sl_pct / 100.0), 2),
                            "tp_pct": tp_pct,
                            "sl_pct": sl_pct,
                            "time_stop_minutes": config.TIME_STOP_MINUTES,
                            "strategy_tag": "Lumos V4 Optimal",
                        }
                        order_attempted = True
                        self._execute_buy_with_10s_chase(
                            symbol=winner_sym,
                            target_qty=order_qty,
                            ref_price=cur_px,
                            moe_res=moe_res,
                            targets=targets,
                        )
                    else:
                        pre_order_decision = "SKIP_INSUFFICIENT_FUNDS"

        return {
            "moe_res": moe_res,
            "briefing_msg": briefing_msg,
            "pre_order_decision": pre_order_decision,
            "order_attempted": order_attempted,
            "winner_symbol": winner_sym,
            "order_qty": order_qty,
        }

    def _market_execution_loop(self):
        logger.info("?? [?  ?????  ? ? (: 15?...")
        system_logger.log("INFO", "LiveRunner", "???  ??????")
        
        while True:
            try:
                mkt = USMarketCalendar.get_market_status()
                current_session = mkt["session_name"]

                if self._last_market_session != current_session:
                    if current_session == "REGULAR_MARKET_OPEN":
                        self.ws_streamer.reset_session_ticks()
                        
                        system_logger.log("TRADE", "MarketSession", f"??? ?? ? ? ({mkt['now_kst_str']})")
                        open_msg = self.notifier.compose_market_open_alert(
                            now_kst_str=mkt["now_kst_str"],
                            mode_str=self.broker.mode_str,
                            model_info=self.get_active_model_info(),
                        )
                        try:
                            self.dispatcher.send_telegram_message(open_msg)
                        except Exception as te:
                            logger.warning(self._redact_log(f"미국장 오픈 텔레그램 발송 에러(무시됨): {te}"))

                    elif current_session in ["AFTER_MARKET_CLOSED", "CLOSED"]:
                        eod_date_file = DATA_DIR / "last_eod_date.json"
                        today_str = datetime.now().strftime("%Y-%m-%d")
                        already_run = False
                        if eod_date_file.exists():
                            try:
                                with open(eod_date_file, "r", encoding="utf-8") as f:
                                    d = json.load(f)
                                if d.get("eod_date") == today_str:
                                    already_run = True
                            except Exception:
                                pass

                        if not already_run:
                            try:
                                # 증권사 통신: 체결 내역 브리핑 발송 (에러 발생 시에도 무시되도록 try-except 가드)
                                try:
                                    exec_hist = self.broker.get_daily_execution_history()
                                    if exec_hist.get("ok") and exec_hist.get("execution_count", 0) > 0:
                                        msg = f"📊 **[오늘의 실제 매매 체결 내역 (증권사 원장)]**\n\n총 {exec_hist['execution_count']}건의 체결 내역이 확인되었습니다.\n"
                                        for ex in exec_hist["executions"]:
                                            sym = ex.get('pdno') or ex.get('stk_cd') or ex.get('symbol') or "알수없음"
                                            od_type = ex.get('sll_buy_dvsn_cd_name') or ex.get('order_type') or ex.get('sll_buy_dvsn_cd') or "체결"
                                            qty = ex.get('ccld_qty') or ex.get('quantity') or ex.get('ord_qty') or 0
                                            px = ex.get('ft_ccld_unpr3') or ex.get('ccld_unpr') or ex.get('price') or 0
                                            msg += f"• {sym} | {od_type} | {qty}주 @ \n"
                                        self.dispatcher.send_telegram_message(msg)
                                        system_logger.log("INFO", "ExecutionHistory", f"당일 체결 내역 브리핑 발송 완료 ({exec_hist['execution_count']}건)")
                                except Exception as eh_err:
                                    system_logger.log("ERROR", "ExecutionHistory", self._redact_log(f"체결 내역 조회/발송 실패 (무시됨): {eh_err}"))
                            except Exception as dummy: pass

                            try:
                                from core.data_lake import DailyAutoPipeline
                                pipeline = DailyAutoPipeline(self.data_lake)
                                eod_res = pipeline.run_full_eod_pipeline(send_telegram=True)
                                
                                # Generate Daily JSON Report
                                try:
                                    from core.eod_reporter import DailyReporter
                                    reporter = DailyReporter(self.broker)
                                    reporter.generate_report()
                                except Exception as rep_err:
                                    logger.warning(self._redact_log(f"[EOD Reporter Error] {rep_err}"))
                                    
                                with open(eod_date_file, "w", encoding="utf-8") as f:
                                    json.dump({"eod_date": today_str, "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
                            except Exception as e:
                                logger.info(self._redact_log(f"[EOD Pipeline Error] {e}"))

                    self._last_market_session = current_session

                # === AI Evaluation & Briefing ===
                if current_session == "REGULAR_MARKET_OPEN" and not self._is_order_in_progress:
                    now_t = time.time()
                    now_dt = datetime.now()
                    force_first_run = (self._last_briefing_time == 0.0)
                    is_target_minute = (now_dt.minute % 15 == 0)
                    
                    if (is_target_minute or force_first_run) and (now_t - self._last_briefing_time) >= 60:
                        self._last_briefing_time = now_t
                        try:
                            self.evaluate_15m_cycle_once(
                                allow_order_execution=True,
                                sync_intraday=True,
                                now_dt=now_dt,
                            )
                        except Exception as e:
                            system_logger.log("ERROR", "AI", self._redact_log(f"AI ? ?? : {e}"))


                if mkt["is_open"] and not self._is_order_in_progress:
                    try:
                        live_p = {
                            config.TRADE_SYMBOLS[0]: self.ws_streamer.get_latest_price(config.TRADE_SYMBOLS[0], 0.0),
                            config.TRADE_SYMBOLS[1]: self.ws_streamer.get_latest_price(config.TRADE_SYMBOLS[1], 0.0)
                        }
                        if current_session == "EOD_LIQUIDATION":
                            try:
                                self.shadow_tracker.flush_eod(exit_prices=live_p)
                            except Exception as e:
                                system_logger.log("ERROR", "ShadowTracker", self._redact_log(f"EOD 초기화 중 에러: {e}"))

                        stk_bal = self.broker.get_overseas_stock_balance(force_refresh=True)
                        if stk_bal.get("ok"):
                            self._reconcile_active_position_with_ledger(stk_bal)
                            if stk_bal.get("holdings"):
                                if current_session == "EOD_LIQUIDATION":
                                    for h in stk_bal.get("holdings"):
                                        sym = str(h.get("symbol") or h.get("stk_cd") or "").strip().upper()
                                        qty = int(float(str(h.get("quantity") or h.get("poss_qty") or 0).replace(",", "")))
                                        try:
                                            buy_px = float(str(h.get("purchase_price") or h.get("pchs_avg_pric") or h.get("buy_price") or 0.0).replace(",", ""))
                                        except Exception:
                                            buy_px = 0.0
                                        if qty > 0:
                                            system_logger.log("TRADE", "EOD", f"Executing {config.PHASE_EOD_CLEAR} EOD Liquidation for {sym}")
                                            self._dispatch_sell(
                                                symbol=sym,
                                                quantity=qty,
                                                reason_desc=f"EOD {config.PHASE_EOD_CLEAR} 100% Liquidation",
                                                is_market_order=True,
                                                buy_px=buy_px
                                            )
                                else:
                                    self._manage_open_positions(stk_bal, realtime_px_override=live_p)
                    except Exception as e:
                        logger.error(self._redact_log(f"❌ [Market Execution Error] {e}"))
                        system_logger.log("ERROR", "MarketLoop", self._redact_log(f"마켓 루프 에러: {e}"))
                sleep_seconds = config.MAIN_LOOP_TICK_OPEN if mkt["is_open"] else config.MAIN_LOOP_TICK_CLOSED
                time.sleep(sleep_seconds)

            except Exception as e:
                system_logger.log("ERROR", "LiveRunner", self._redact_log(f" ? : {e}"))
                time.sleep(config.MAIN_LOOP_ERROR_WAIT)

    def run_once_and_start_listener(self, start_background_threads: bool = True) -> Dict[str, Any]:
        """
        1???? ? ??????  ????? ??? ??
        """
        logger.info("=" * 75)
        # 0. ? [?  ??? ??1?  ? ???&  ??? ?
        logger.info("\n" + "=" * 75)
        logger.info("?  [?  ??? ??  ? ???&  ??? ?  ?")
        logger.info("=" * 75)
        sync_res = USMarketCalendar.verify_time_synchronization()
        for chk_item in sync_res["checklist_items"]:
            logger.info(f"   ??{chk_item}")
        if not sync_res["all_ok"]:
            err_msg = "? [??? ????  ???? ??????? ??? ??!"
            logger.critical(err_msg)
            raise RuntimeError(err_msg)
        logger.info(f"   ??? ? ????100% ??? ({sync_res['dst_text']})")
        logger.info("=" * 75)

        # 1.  ? ? ?
        mkt_status = USMarketCalendar.get_market_status()
        self._last_market_session = mkt_status["session_name"]

        logger.info(f"\n[1/3] ?  ? ? ??:")
        logger.info(f"   ??? ?: {mkt_status['status_desc']}")
        logger.info(f"   ??? ?: {mkt_status['now_kst_str']}")
        logger.info(f"   ??? ?: {mkt_status['now_ny_str']} ({mkt_status['dst_text']})")
        logger.info(f"   ??? : {mkt_status['next_open_kst_str']} (?? ?: {mkt_status['time_until_open_str']})")

        # 2. ? ?? ? ??
        logger.info(f"\n[2/3] {self.broker.broker_name} {self.broker.mode_str}  ?? ? ? ??:")
        conn_res = self.broker.test_connection()
        logger.info(f"   ??OAuth2 ?: {'???  ?' if conn_res['ok'] else '?? ?'}")
        logger.info(self._redact_log(f"   ??? : {self.broker.account_no}-{self.broker.account_type}"))
        logger.info(f"   ?????: ${conn_res['usd_order_available']:,.2f} USD")

        # 2.5. 기존에 걸려있는 모든 미체결 주문 일괄 취소 (수동 주문 잔재로 인한 예수금 잠김 방지)
        logger.info(f"\\n[2.5/3] 기존 미체결 주문 일괄 취소 확인 중...")
        if start_background_threads:
            try:
                cancel_res = self.broker.cancel_all_open_orders()
                if cancel_res:
                    logger.info(f"   => {len(cancel_res)}건의 기존 미체결 주문을 성공적으로 취소했습니다.")
                else:
                    logger.info(f"   => 취소할 기존 미체결 주문이 없습니다.")
            except Exception as e:
                logger.warning(self._redact_log(f"   => 기존 미체결 주문 취소 중 에러 발생 (무시하고 진행): {e}"))

        # 3. 모델 등 정보 설정 (Single Source of Truth: MLFeatureEngine.get_active_model_info())
        active_model_info = self.get_active_model_info()
        start_msg = self.notifier.compose_startup_alert(
            mode_str=self.broker.mode_str,
            broker_name=self.broker.broker_name,
            account_masked=f"{self.broker.account_no}-{self.broker.account_type}",
            mkt_status=mkt_status,
            sync_res=sync_res,
            conn_res=conn_res,
            model_info=active_model_info,
        )
        try:
            if not self.notifier._check_and_record_state_dedup(
                "STARTUP",
                f"{active_model_info.get('active_model_id')}:{active_model_info.get('active_model_sha256')}:{active_model_info.get('load_status')}",
            ):
                self.dispatcher.send_telegram_message(start_msg)
            if active_model_info.get("load_status") == "BLOCKED" or not active_model_info.get("is_manifest_memory_consistent", True):
                self.notifier.send_model_state_alert(
                    event_type="BLOCKED",
                    model_info=active_model_info,
                    reason=str(active_model_info.get("consistency_reason") or "모델 로드 실패로 신규 진입 차단"),
                )
            elif active_model_info.get("is_fallback") or active_model_info.get("load_status") == "FALLBACK":
                self.notifier.send_model_state_alert(
                    event_type="FALLBACK_TO_CHAMPION",
                    model_info=active_model_info,
                    reason="기동 시 Primary 모델 대신 Champion 폴백 모델이 로드되었습니다.",
                )
            system_logger.log(
                "INFO",
                "LiveRunner",
                self._redact_log(
                    f"🚀 {self.broker.mode_str} 시스템 기동 완료 (모델: {active_model_info.get('active_model_id')}[{active_model_info.get('model_version')}], 가용예수금: ${conn_res['usd_order_available']:,.2f})"
                ),
            )
        except Exception as te:
            logger.warning(self._redact_log(f"? ? ? ?: {te}"))

        if not start_background_threads:
            return {
                "ok": True,
                "active_model_info": active_model_info,
                "start_msg": start_msg,
                "mkt_status": mkt_status,
                "sync_res": sync_res,
                "conn_res": conn_res,
            }

        # 4. ????WebSocket) ??? ??
        self.ws_streamer.start()
        logger.info(f"??[{self.broker.broker_name} ??WebSocket ? ???] 10ms ??????? ?")

        # 5. ????   ?????
        trd_thread = threading.Thread(target=self._market_execution_loop, daemon=True)
        trd_thread.start()

        # 6. ??? ????? ? (Foreground Daemon)
        self.controller.listen_loop(poll_interval=2)
        return {"ok": True, "active_model_info": active_model_info}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Lumos Auto Trading Live Runner")
    parser.add_argument("--real", action="store_true", help="Run in real trading mode")
    parser.add_argument("--sim", action="store_true", help="Run in simulation mode")
    args, unknown = parser.parse_known_args()

    is_sim = None
    if args.real:
        is_sim = False
    elif args.sim:
        is_sim = True

    runner = KiwoomLiveRunner(is_simulation=is_sim)
    runner.run_once_and_start_listener()

