import config
import os
# import os
import sys
import time
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
from core.kiwoom_broker import KiwoomBroker
from agents.dispatcher_agent import DispatcherAgent
from core.telegram_controller import TelegramController
from core.state_hub import StateHub
from core.data_lake import MarketDataLake
from core.shadow_sandbox import ShadowSandboxEngine
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

class USMarketCalendar:
    """
    [ ? (NYSE / NASDAQ) ??? ? ??????
    - ????Daylight Saving Time) ?  (3??? ???~ 11?? ???
    - ???? ?:
      * ????(EDT): ?? 22:30 ~ ? 05:00
      * ????(EST): ?? 23:30 ~ ? 06:00
    - (???? ?????
    - ?  ?? ?? ? ??
    """
    @staticmethod
    def get_market_status(now_dt: Optional[datetime] = None) -> Dict[str, Any]:
        if now_dt is None:
            now_dt = datetime.now()

        ny_tz = ZoneInfo("America/New_York")
        kst_tz = ZoneInfo("Asia/Seoul")
        
        if now_dt is None:
            now_local = datetime.now().astimezone()
            now_ny = now_local.astimezone(ny_tz)
            now_kst = now_local.astimezone(kst_tz)
        elif now_dt.tzinfo is None:
            now_kst = now_dt.replace(tzinfo=kst_tz)
            now_ny = now_kst.astimezone(ny_tz)
        else:
            now_ny = now_dt.astimezone(ny_tz)
            now_kst = now_dt.astimezone(kst_tz)
        weekday_ny = now_ny.weekday()  # 0: Mon, ..., 4: Fri, 5: Sat, 6: Sun
        ny_time = now_ny.time()

        is_weekend = (weekday_ny >= 5)
        
        market_open_time = dtime(9, 30)
        trading_cutoff_time = dtime(15, 30)
        eod_liquidation_time = dtime(15, 50)
        market_close_time = dtime(16, 0)

        is_regular_hours = (not is_weekend) and (market_open_time <= ny_time < market_close_time)
        is_phase2_allowed = False
        is_trading_allowed = (not is_weekend) and (market_open_time <= ny_time < trading_cutoff_time)
        is_entry_allowed = is_trading_allowed
        is_eod_liquidation_window = (not is_weekend) and (eod_liquidation_time <= ny_time < market_close_time)

        if is_regular_hours:
            next_open_kst = None
            time_until_open_str = "   "
        else:
            target_ny_date = now_ny.date()
            if is_weekend:
                days_ahead = (7 - weekday_ny)
                target_ny_date += timedelta(days=days_ahead)
            else:
                if ny_time >= market_close_time:
                    if weekday_ny == 4:
                        target_ny_date += timedelta(days=3)
                    else:
                        target_ny_date += timedelta(days=1)
            
            next_open_ny = datetime.combine(target_ny_date, market_open_time, tzinfo=ny_tz)
            next_open_kst = next_open_ny.astimezone(kst_tz)
            delta = next_open_kst - now_kst
            hours, remainder = divmod(int(delta.total_seconds()), 3600)
            mins, secs = divmod(remainder, 60)
            time_until_open_str = f"{hours}hours {mins}mins"

        if is_weekend:
            day_name_kr = "Weekend"
            session_name = "WEEKEND_CLOSED"
            status_desc = f"? ?   ? ({day_name_kr}?)"
        elif not is_regular_hours:
            if ny_time < market_open_time:
                session_name = "PRE_MARKET_WAITING"
                status_desc = "????? ??? ??"
            else:
                session_name = "AFTER_MARKET_CLOSED"
                status_desc = "? ??? (???"
        else:
            if is_eod_liquidation_window:
                session_name = "EOD_LIQUIDATION"
                status_desc = "???? 10??? 0% ?? ? ??????"
            elif is_trading_allowed:
                session_name = "REGULAR_MARKET_OPEN"
                status_desc = "? ? ???Phase 1 ???  ?(15m Model C)"
            elif is_phase2_allowed:
                session_name = "POWER_HOUR_SNIPER"
                status_desc = "??? ???Phase 2 ? ? ??  ?(5m Sniper)"
            else:
                session_name = "REGULAR_MARKET_NO_ENTRY"
                status_desc = "???? ??(?  ,  ?????"

        is_dst = bool(now_ny.dst())

        return {
            "is_open": is_regular_hours,
            "is_entry_allowed": is_entry_allowed,
            "is_trading_allowed": is_trading_allowed,
            "is_phase2_allowed": is_phase2_allowed,
            "is_weekend": is_weekend,
            "is_eod_liquidation_window": is_eod_liquidation_window,
            "session_name": session_name,
            "status_desc": status_desc,
            "now_kst_str": now_kst.strftime("%Y-%m-%d %H:%M:%S KST"),
            "now_ny_str": now_ny.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "is_dst": is_dst,
            "dst_text": "EDT" if is_dst else "EST",
            "next_open_kst_str": next_open_kst.strftime("%Y-%m-%d (%a) %H:%M KST") if next_open_kst else "Open",
            "time_until_open_str": time_until_open_str
        }

    @staticmethod
    def verify_time_synchronization() -> Dict[str, Any]:
        """
        [?  ??? ??1? ? ???? ????? ??
        1. OS  ? ??? ???(ZoneInfo America/New_York) ??? ?
        2. ????EDT: -13h ?) ?????EST: -14h ?) ? ? ??
        3.  ??4?  ?(09:30~14:30 Phase 1 vs 14:30~15:30 Phase 2 vs 15:30 ??vs 15:50 EOD ?)
        """
        ny_tz = ZoneInfo("America/New_York")
        kst_tz = ZoneInfo("Asia/Seoul")
        now_local = datetime.now().astimezone()
        now_ny = now_local.astimezone(ny_tz)
        now_kst = now_local.astimezone(kst_tz)

        # KST? NYT ???
        delta_hours = round((now_kst.utcoffset().total_seconds() - now_ny.utcoffset().total_seconds()) / 3600.0, 1)
        is_dst = bool(now_ny.dst())
        expected_diff = 13.0 if is_dst else 14.0
        time_sync_ok = (delta_hours == expected_diff)

        #  ??4?  ?( ? ?????? ? ? ????
        test_weekday = now_ny.date() - timedelta(days=now_ny.weekday())
        test_p1 = datetime.combine(test_weekday, dtime(11, 0), tzinfo=ny_tz)
        test_p2 = datetime.combine(test_weekday, dtime(15, 0), tzinfo=ny_tz)
        test_cd = datetime.combine(test_weekday, dtime(15, 40), tzinfo=ny_tz)
        test_eod = datetime.combine(test_weekday, dtime(15, 55), tzinfo=ny_tz)

        s_p1 = USMarketCalendar.get_market_status(test_p1)
        s_p2 = USMarketCalendar.get_market_status(test_p2)
        s_cd = USMarketCalendar.get_market_status(test_cd)
        s_eod = USMarketCalendar.get_market_status(test_eod)

        switching_ok = (
            s_p1["is_entry_allowed"] and not s_p1["is_eod_liquidation_window"] and
            s_eod["is_eod_liquidation_window"] and not s_eod["is_entry_allowed"]
        )

        all_ok = bool(time_sync_ok and switching_ok)

        return {
            "all_ok": all_ok,
            "time_sync_ok": time_sync_ok,
            "switching_ok": switching_ok,
            "delta_hours": delta_hours,
            "expected_diff": expected_diff,
            "is_dst": is_dst,
            "dst_text": "????EDT, 13? ?)" if is_dst else "????EST, 14? ?)",
            "now_kst_str": now_kst.strftime("%Y-%m-%d %H:%M:%S KST"),
            "now_ny_str": now_ny.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "checklist_items": [
            ]
        }


class KiwoomLiveRunner:
    """
    [??(Kiwoom) ????????  ??? ???]
    1.  ?? ? 100% ?: ? //?/ ?????? ???
    2.   ?AI ???? 100% ? (  ? )
    3. 3????????&   (: 1????/ : 100% ? ? ?)
    4. ??? ???  0.5?/ ? 0.3??  ?HTTP 429 ? ???
    """
    # ??[?????????? (Hard Rule #4: 3.0?????]
    ORDER_TIMEOUT_BUY_SEC: float = 3.0   #  ?????(3?
    ORDER_TIMEOUT_SELL_SEC: float = 3.0  #  ?????(3?

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
        self._daily_cb_file = DATA_DIR / "daily_circuit_breaker_state.json"
        self._load_daily_cb_state()

        # ??[ ? ?15?? ???  ??
        self._last_veto_time: float = 0.0
        self._last_veto_state: bool = False
        self._last_defense_time: float = 0.0
        self._last_defense_state: bool = False
        self._last_briefing_time: float = 0.0
        self._last_moe_res: Optional[Dict[str, Any]] = None

    def _save_active_position(self, pos_data: Dict[str, Any]):
        """???? ? ???? ???? ? ???"""
        self._active_position = pos_data
        try:
            pos_file = DATA_DIR / "active_position.json"
            with open(pos_file, "w", encoding="utf-8") as f:
                json.dump(pos_data, f, ensure_ascii=False, indent=2)
            logger.info(f"? [???? ????] {pos_data.get('symbol')} {pos_data.get('quantity')}?@ ${pos_data.get('price')} (?: {pos_data.get('buy_time')})")
        except Exception as e:
            logger.warning(f"???? ????: {e}")

    def _get_active_position(self) -> Optional[Dict[str, Any]]:
        if self._active_position:
            return self._active_position
        pos_file = DATA_DIR / "active_position.json"
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
        try:
            pos_file = DATA_DIR / "active_position.json"
            if pos_file.exists():
                pos_file.unlink()
            logger.info("? [???? ?? 100% ? ??? ?")
        except Exception as e:
            logger.warning(f"???? ?? ?: {e}")

    def _get_current_ny_date(self) -> str:
        ny_tz = ZoneInfo("America/New_York")
        return datetime.now().astimezone().astimezone(ny_tz).strftime("%Y-%m-%d")

    def _load_daily_cb_state(self):
        today_ny = self._get_current_ny_date()
        if self._daily_cb_file.exists():
            try:
                with open(self._daily_cb_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("date") == today_ny:
                    self.daily_stoploss_count = int(data.get("stoploss_count", 0))
                    self.daily_circuit_breaker_triggered = bool(data.get("circuit_breaker_triggered", False))
                    logger.info(f"??[? ? ? ] ?: {today_ny} | ? ?: {self.daily_stoploss_count}/3??|  ?: {self.daily_circuit_breaker_triggered}")
                    return
            except Exception as e:
                logger.debug(f"? ? ?  ?: {e}")
        self.daily_stoploss_count = 0
        self.daily_circuit_breaker_triggered = False

    def _save_daily_cb_state(self):
        today_ny = self._get_current_ny_date()
        data = {
            "date": today_ny,
            "stoploss_count": self.daily_stoploss_count,
            "circuit_breaker_triggered": self.daily_circuit_breaker_triggered,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        try:
            with open(self._daily_cb_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"? ? ? ????: {e}")

    def _reset_daily_circuit_breaker(self):
        self.daily_stoploss_count = 0
        self.daily_circuit_breaker_triggered = False
        self._save_daily_cb_state()
        logger.info("? [? ? ? ?? ???(09:30 NYT): daily_stoploss_count = 0, ?   ?")

    def _record_stoploss(self):
        """
#         - '-2.0% ?? ?  ??daily_stoploss_count 1 ? (?/?????)
        """
        self.daily_stoploss_count += 1
        logger.warning(f"? [? ?????] ? ?: {self.daily_stoploss_count}/3??")
        system_logger.log("RISK", "CircuitBreaker", f"? ? -2.0% ?? ({self.daily_stoploss_count}/3??")

        if self.daily_stoploss_count >= 3:
            self.daily_circuit_breaker_triggered = True
            logger.error(f"? [? ? ? ] ? ? 3???(3-Out) ??? ?  ? ")
            system_logger.log("RISK", "CircuitBreaker", "? ? ? ? : ? ????(3-Out Veto)")

            msg = f"""🚨 **일일 3-Out 서킷 브레이커 발동**\n- 금일 손절 횟수: {self.daily_stoploss_count}/3\n- 조치: 신규 매수 전면 차단 (VETO)\n- 현금: 100% 보존"""

            msg = "Circuit Breaker Triggered"
            try:
                self.dispatcher.send_telegram_message(msg)
            except Exception as te:
                logger.warning(f"? ? ?  ?: {te}")

        self._save_daily_cb_state()

    def _on_websocket_execution(self, exec_data: Dict[str, Any]):
        """
        """
        logger.info(f"? [WebSocket ?? ? ?]: {exec_data}")
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
            logger.debug(f" ?  ? ?: {exec_data}")
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

    def _wait_for_fill(self, order_no: str, symbol: str, is_buy: bool, target_qty: int, timeout_sec: Optional[float] = None) -> Tuple[bool, int, int]:
        """
        """
        if timeout_sec is None:
            timeout_sec = self.order_timeout_buy_sec if is_buy else self.order_timeout_sell_sec

        start_t = time.time()
        while time.time() - start_t < timeout_sec:
            ws_filled = self._order_fills.get(order_no, 0)
            if ws_filled >= target_qty:
                return True, ws_filled, 0
            time.sleep(0.05)

        # ????? ????? 1?? ?
        stk_bal = self.broker.get_overseas_stock_balance()
        actual_qty = 0
        if stk_bal.get("ok"):
            for h in stk_bal.get("holdings", []):
                if str(h.get("symbol") or h.get("stk_cd") or "").strip().upper() == symbol:
                    actual_qty = int(float(str(h.get("quantity") or h.get("poss_qty") or 0).replace(",", "")))
                    break

        if is_buy:
            unfilled = max(0, target_qty - actual_qty)
            return (actual_qty >= target_qty), actual_qty, unfilled
        else:
            return (actual_qty <= 0), target_qty - actual_qty, actual_qty

    def _on_websocket_tick(self, symbol: str, price: float, extra: Dict[str, Any]):
        """
        """
        if self._is_order_in_progress:
            return

        try:
            mkt = USMarketCalendar.get_market_status()
            if not mkt.get("is_open"):
                return

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

            tp_threshold = float(active_pos.get("tp_pct", 3.0)) / 100.0 if active_pos.get("tp_pct") else 0.030
            sl_threshold = float(active_pos.get("sl_pct", 2.0)) / 100.0 if active_pos.get("sl_pct") else 0.020
            time_stop_minutes = float(active_pos.get("time_stop_minutes", 90))

            # 1.  ?
            if pnl_pct >= tp_threshold:
                gain_pct = pnl_pct * 100
                logger.info(f"? [??? ????] {symbol} {qty}? ?  (?? +{gain_pct:.2f}%)")
                self._execute_sell_with_10s_chase(
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
                self._execute_sell_with_10s_chase(
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
                        self._execute_sell_with_10s_chase(
                            symbol=symbol,
                            quantity=qty,
                            reason_desc=f"??{time_stop_minutes:.0f}?????? ({elapsed_min:.0f}?)",
                            buy_px=buy_px,
                            cur_px=price,
                            is_stoploss=False
                        )
                except Exception as te:
                    logger.debug(f"???? ?: {te}")

        except Exception as e:
            logger.debug(f"WS ?? ?: {e}")

    def _manage_open_positions(self, stk_bal: Dict[str, Any], realtime_px_override: Optional[Dict[str, float]] = None):
        """
        """
        holdings = stk_bal.get("holdings", [])
        if not holdings:
            return

        active_pos = self._get_active_position()
        tp_max_pct = float(active_pos.get("tp_pct", 3.5)) / 100.0 if active_pos else 0.035
        sl_initial_pct = float(active_pos.get("sl_pct", 2.0)) / 100.0 if active_pos else 0.020

        for h in holdings:
            sym = str(h.get("symbol") or h.get("stk_cd") or "").strip().upper()
            qty = int(float(str(h.get("quantity") or h.get("poss_qty") or h.get("sell_alowq") or 0).replace(",", "")))
            buy_px = float(str(h.get("purchase_price") or h.get("avg_price") or h.get("frgn_stk_book_uv") or 0.0).replace(",", ""))

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
            trailing_trigger_px = buy_px * 1.020
            
            if peak_high >= trailing_trigger_px:
                is_trailing_active = True
                current_sl_px = buy_px * 1.005
                
            if is_trailing_active:
                t = peak_high * 0.992
                if t > current_sl_px:
                    current_sl_px = t
            
            if active_pos:
                active_pos["dynamic_sl_px"] = current_sl_px
                self._save_active_position(active_pos)
                
            tp_max_px = buy_px * (1.0 + tp_max_pct)

            if cur_px >= tp_max_px:
                gain_pct = (cur_px / buy_px - 1.0) * 100.0
                self._execute_sell_with_10s_chase(
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
                self._execute_sell_with_10s_chase(
                    symbol=sym,
                    quantity=qty,
                    reason_desc=reason,
                    buy_px=buy_px,
                    cur_px=cur_px,
                    is_stoploss=True
                )

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
                        b_px = float(str(h.get("purchase_price") or h.get("avg_price") or h.get("frgn_stk_book_uv") or 0.0).replace(",", ""))
                        if b_px > 0:
                            logger.info(f"? [??? ?? ??? ??: ${default_price:.2f} ??? ? ??: ${b_px:.2f} (? ?: {p_qty}?")
                            return round(b_px, 2), (p_qty if p_qty > 0 else default_qty)
        except Exception as e:
            logger.warning(f"? ? ?? ???? (?? fallback ??): {e}")
        return default_price, default_qty

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
        with self._order_lock:
            self._is_order_in_progress = True

        try:
            if target_qty <= 0:
                logger.warning(f"⚠️ [매수 거부] 비정상 타겟 수량({target_qty}주)으로 진입이 취소되었습니다.")
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
            order_px_1 = round(cur_px + 0.03, 2)

            logger.info(f"?? [1? ] {symbol} {target_qty}?@ ${order_px_1:.2f} ({self.order_timeout_buy_sec:.0f}? ???)")
            ord_res_1 = self.broker.send_order(symbol=symbol, order_type="BUY", quantity=target_qty, price=order_px_1)
            ord_no_1 = str(ord_res_1.get("order_no", "")).strip()

            # ?  ?  (1? )
            mode_title = "? [?? ?]" if self.broker.is_simulation else "? [?? ??]"
            score_block = "\n".join([f"??**{k.upper()}:** `{v*100:.1f}??" for k, v in sorted(all_scores.items(), key=lambda x: x[1], reverse=True)])
            buy_msg = f"""{mode_title} 신규 매수 주문 (1차)\n🚨 **브로커:** `{self.broker.broker_name} {self.broker.mode_str}`\n💡 **종목:** `{symbol}`\n📊 **수량:** `{target_qty}`\n💰 **주문가:** `${order_px_1:.2f}`\n\n🔥 **[GBDT 모델 확신도]**\n{score_block}\n\n🎯 **목표가:** `${tp_px:.2f}`\n🛡️ **손절가:** `${sl_px:.2f}`\n⏱️ **시간청산:** {time_stop_min}분"""


            self.dispatcher.send_telegram_message(buy_msg)
            system_logger.log("TRADE", "AutoExecution", f"?? [{self.broker.broker_name} {symbol} 1? ] {target_qty}?@ ${order_px_1:.2f} (???{top_conf:.1f}%)")

            # 1. ???(?  ??0ms   ?)
            is_filled_1, filled_1, unfilled_1 = self._wait_for_fill(
                order_no=ord_no_1,
                symbol=symbol,
                is_buy=True,
                target_qty=target_qty,
                timeout_sec=self.order_timeout_buy_sec
            )

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
                    is_simulation=self.broker.is_simulation
                )
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
            if ord_no_1:
                self.broker.cancel_order(order_no=ord_no_1, symbol=symbol, quantity=unfilled_1)
            time.sleep(0.5)

            # ? ? ?? ???? 
            stk_bal_1 = self.broker.get_overseas_stock_balance(force_refresh=True)
            already_filled_qty = 0
            if stk_bal_1.get("ok"):
                for h in stk_bal_1.get("holdings", []):
                    if str(h.get("symbol") or h.get("stk_cd") or "").strip().upper() == symbol:
                        already_filled_qty = int(float(str(h.get("quantity") or h.get("poss_qty") or 0).replace(",", "")))
                        break

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
                    "high_price_during_hold": real_buy_px,
                    "low_price_during_hold": real_buy_px,
                    "buy_attempts": 1
                }
                self._save_active_position(filled_pos)
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
                    is_simulation=self.broker.is_simulation
                )
                return True

            # 2??? ???
            latest_px_2 = float(self.ws_streamer.get_latest_price(symbol, cur_px))
            order_px_2 = round(latest_px_2 + 0.03, 2)
            logger.info(f"??[2? ??(1/1)] {symbol} {remaining_qty}?@ ${order_px_2:.2f} ({self.order_timeout_buy_sec:.0f}? ???)")
            ord_res_2 = self.broker.send_order(symbol=symbol, order_type="BUY", quantity=remaining_qty, price=order_px_2)
            ord_no_2 = str(ord_res_2.get("order_no", "")).strip()
            system_logger.log("TRADE", "OrderChasing", f"??[ 2??? ?? {symbol} {remaining_qty}?@ ${order_px_2:.2f} (???1/1)")

            # 2. ???(?  ??0ms   ?)
            is_filled_2, filled_2, unfilled_2 = self._wait_for_fill(
                order_no=ord_no_2,
                symbol=symbol,
                is_buy=True,
                target_qty=remaining_qty,
                timeout_sec=self.order_timeout_buy_sec
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
                    is_simulation=self.broker.is_simulation
                )
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
            if ord_no_2:
                self.broker.cancel_order(order_no=ord_no_2, symbol=symbol, quantity=unfilled_2)
            time.sleep(0.5)

            # ? ???  ?
            stk_bal_final = self.broker.get_overseas_stock_balance(force_refresh=True)
            final_filled = 0
            if stk_bal_final.get("ok"):
                for h in stk_bal_final.get("holdings", []):
                    if str(h.get("symbol") or h.get("stk_cd") or "").strip().upper() == symbol:
                        final_filled = int(float(str(h.get("quantity") or h.get("poss_qty") or 0).replace(",", "")))
                        break

            if final_filled > 0:
                real_buy_px, real_qty = self._sync_real_ledger_entry(symbol, order_px_2, final_filled)
                tp_px_dyn = round(real_buy_px * (1.0 + tp_pct / 100.0), 2) if real_buy_px > 0 else tp_px
                sl_px_dyn = round(real_buy_px * (1.0 - sl_pct / 100.0), 2) if real_buy_px > 0 else sl_px

                filled_pos = {
                    "symbol": symbol,
                    "quantity": real_qty,
                    "price": real_buy_px,
                    "buy_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
                    "high_price_during_hold": real_buy_px,
                    "low_price_during_hold": real_buy_px,
                    "buy_attempts": 2
                }
                self._save_active_position(filled_pos)
                logger.info(f"? [?  ?] {symbol} {real_qty}??? (? ??: ${real_buy_px:.2f})")
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
                    is_simulation=self.broker.is_simulation
                )
                return True
            else:
                self._clear_active_position()
                system_logger.log("TRADE", "OrderChasing", f"? [  ?] {symbol} ? ? (??100%  ??AI ???????)")
                return False

        except Exception as e:
            logger.error(f"??   ?: {e}")
            system_logger.log("ERROR", "BuyChase", f"?   ? : {e}")
            try:
                self.experience_logger.record_error_event("LiveRunner", "BUY_CHASE_EXCEPTION", "ERROR", str(e))
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
        is_market_order: bool = False
    ) -> bool:
        """
        """
        with self._order_lock:
            self._is_order_in_progress = True

        try:
            b_name = self.broker.broker_name
            loop_retry = 0

            # ? [??? ? ? (Hard Rule #2)]
            # ? ?? ??? ????(poss_qty) ?
            stk_bal_check = self.broker.get_overseas_stock_balance()
            actual_qty = 0
            if stk_bal_check.get("ok"):
                for h in stk_bal_check.get("holdings", []):
                    if str(h.get("symbol") or h.get("stk_cd") or "").strip().upper() == symbol:
                        actual_qty = int(float(str(h.get("quantity") or h.get("poss_qty") or 0).replace(",", "")))
                        break
            if actual_qty <= 0:
                logger.info(f"? [? ? 0??] {symbol} ?? 100% ? ? ? ?? ? ?AI ???? ?")
                self._clear_active_position()
                return True
            quantity = min(quantity, actual_qty)

            while True:
                loop_retry += 1
                latest_px = float(self.ws_streamer.get_latest_price(symbol, cur_px))
                if latest_px <= 0:
                    latest_px = cur_px

                if is_market_order and not self.broker.is_simulation:
                    # 15:50 EOD ?? 0%  ?: ?? ?? ?(03, price=0.0)  
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
                        logger.debug(f"  ? ? (): {le}")

                    mode_title = "? [?? ?]" if self.broker.is_simulation else "? [?? ??]"
                    system_logger.log("TRADE", "Liquidation", f"? [{mode_title} 100% ? ? ?] {symbol} {quantity}? | ?: {reason_desc} |  ?: {final_pnl_pct:+.2f}%")

                    sell_msg = f"""{mode_title} 100% 매도 청산 완료\n🚨 **브로커:** `{self.broker.broker_name} {self.broker.mode_str}`\n💡 **사유:** `{reason_desc}`\n📊 **종목/수량:** `{symbol} {quantity:,}주`\n💰 **체결가:** `${latest_px:.2f}` (최종 수익률: {final_pnl_pct:+.2f}%)\n🛡️ **사후 관리:** `100% 현금화 완료 (AI MoE 새 진입 대기)`"""
                    self.dispatcher.send_telegram_message(sell_msg)
                    if is_stoploss:
                        self._record_stoploss()
                    return True

                if s_ord_no:
                    self.broker.cancel_order(order_no=s_ord_no, symbol=symbol, quantity=rem_qty)
                    time.sleep(0.5)

        except Exception as e:
            logger.error(f"매도 추격 주문 에러: {e}")
            system_logger.log("ERROR", "SellChase", f"매도 추격 에러: {e}")
            try:
                self.experience_logger.record_error_event("LiveRunner", "SELL_CHASE_EXCEPTION", "ERROR", str(e))
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

    def _market_execution_loop(self):
        logger.info("?? [?  ?????  ? ? (: 15?...")
        system_logger.log("INFO", "LiveRunner", "???  ??????")
        
        while True:
            try:
                mkt = USMarketCalendar.get_market_status()
                current_session = mkt["session_name"]

                if self._last_market_session != current_session:
                    if current_session == "REGULAR_MARKET_OPEN":
                        self._reset_daily_circuit_breaker()
                        self.ws_streamer.reset_session_ticks()
                        
                        system_logger.log("TRADE", "MarketSession", f"??? ?? ? ? ({mkt['now_kst_str']})")
                        open_msg = f"""🔥 **[정규장 매매 개시]**\n\n⏰ 현재 시각: `{mkt['now_kst_str']}`\n🖥️ 실행 모드: `{self.broker.mode_str}`\n🧠 AI 전략: `하이브리드 MoE V3 (09:30~15:30 EDT)`\n🎯 매수 룰: `GBDT 62% 이상`\n🛡️ 청산 룰: `Max TP +3.5% / ATR Trailing Stop`\n🌙 마감청산: `15:50 EDT 0% 오버나잇 전량 시장가`"""
                        self.dispatcher.send_telegram_message(open_msg)

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
                                from core.data_lake import DailyAutoPipeline
                                pipeline = DailyAutoPipeline(self.data_lake)
                                eod_res = pipeline.run_full_eod_pipeline(send_telegram=True)
                                with open(eod_date_file, "w", encoding="utf-8") as f:
                                    json.dump({"eod_date": today_str, "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
                            except Exception as e:
                                logger.info(f"[EOD Pipeline Error] {e}")

                    self._last_market_session = current_session

                # === AI Evaluation & Briefing ===
                if current_session == "REGULAR_MARKET_OPEN" and not self._is_order_in_progress:
                    now_t = time.time()
                    active_pos = self._get_active_position()
                    
                    now_dt = datetime.now()
                    force_first_run = (self._last_briefing_time == 0.0)
                    is_target_minute = (now_dt.minute % 15 == 0)
                    
                    if (is_target_minute or force_first_run) and (now_t - self._last_briefing_time) >= 60:
                        self._last_briefing_time = now_t
                        
                        try:                            # 2) 전 종목 Intraday 캔들 동기화
                            self.data_lake.sync_live_intraday_candles(config.ALL_SYMBOLS)
                            df_15m = self.data_lake.load_candles("TQQQ", "15m")
                            live_prices = {sym: self.ws_streamer.get_latest_price(sym, 0.0) for sym in config.ALL_SYMBOLS}
                            
                            moe_res = self.moe_orchestrator.evaluate_dual_filter_signal(df_candle_15m=df_15m, live_prices=live_prices)
                            self._last_moe_res = moe_res
                            
                            conf = float(moe_res.get('gating_confidence', 0.0)) * 100.0
                            is_appr = moe_res.get('is_approved', False)
                            direction = moe_res.get('direction', 'NONE')
                            
                            dir_str = "매수 대기" if direction == "NONE" else direction
                            briefing_msg = f"""🤖 <b>[Lumos AI 정기 브리핑]</b>
• <b>시간</b>: {now_dt.strftime('%H:%M')} (KST)
• <b>AI 판독 방향</b>: {dir_str}
• <b>GBDT 확신도</b>: {conf:.1f}%
• <b>VIXY</b>: {live_prices.get('VIXY', 0.0):.2f} / <b>IEF</b>: {live_prices.get('IEF', 0.0):.2f}
• <b>상태</b>: {'진입 승인 🚀' if is_appr else '관망 중 👁️'}
"""
                            self.dispatcher.send_telegram_message(briefing_msg)
                            system_logger.log("INFO", "AI", f"15???? ? (?: {conf:.1f}%)")
                            
                            if is_appr and not active_pos and not self.daily_circuit_breaker_triggered:
                                if direction in ["LONG_TQQQ", "SHORT_SQQQ"]:
                                    winner_sym = "TQQQ" if direction == "LONG_TQQQ" else "SQQQ"
                                    cur_px = live_prices.get(winner_sym, 0.0)
                                    
                                    if cur_px > 0:
                                        conn_res = self.broker.test_connection()
                                        usd_avail = float(conn_res.get("usd_order_available", 0.0))
                                        order_qty = int((usd_avail * 0.99) / (cur_px + 0.05))
                                        
                                        if order_qty > 0:
                                            system_logger.log("TRADE", "AI", f"V3 AI  ? ?! {winner_sym} {order_qty}? ?")
                                            atr_14 = float(moe_res.get('features', {}).get('ATR_14', cur_px * 0.018))
                                            atr_pct = (atr_14 / cur_px) * 100.0 if cur_px > 0 else 1.8
                                            sl_pct = max(2.0, min(3.2, atr_pct * 1.5))
                                            tp_pct = 3.5
                                            
                                            targets = {
                                                "dynamic_tp_px": round(cur_px * (1.0 + tp_pct/100.0), 2),
                                                "dynamic_sl_px": round(cur_px * (1.0 - sl_pct/100.0), 2),
                                                "tp_pct": tp_pct,
                                                "sl_pct": sl_pct,
                                                "time_stop_minutes": 90,
                                                "strategy_tag": "Lumos V4 Optimal"
                                            }
                                            self._execute_buy_with_10s_chase(
                                                symbol=winner_sym,
                                                target_qty=order_qty,
                                                ref_price=cur_px,
                                                moe_res=moe_res,
                                                targets=targets
                                            )
                        except Exception as e:
                            system_logger.log("ERROR", "AI", f"AI ? ?? : {e}")


                if mkt["is_open"] and not self._is_order_in_progress:
                    try:
                        stk_bal = self.broker.get_overseas_stock_balance()
                        if stk_bal.get("ok") and stk_bal.get("holdings"):
                            live_p = {
                                "TQQQ": self.ws_streamer.get_latest_price("TQQQ", 0.0),
                                "SQQQ": self.ws_streamer.get_latest_price("SQQQ", 0.0)
                            }
                            if current_session == "EOD_LIQUIDATION":
                                for h in stk_bal.get("holdings"):
                                    sym = str(h.get("symbol") or h.get("stk_cd") or "").strip().upper()
                                    qty = int(float(str(h.get("quantity") or h.get("poss_qty") or 0).replace(",", "")))
                                    if qty > 0:
                                        system_logger.log("TRADE", "EOD", f"Executing 15:50 EOD Liquidation for {sym}")
                                        self._execute_sell_with_10s_chase(
                                            symbol=sym,
                                            quantity=qty,
                                            reason_desc="EOD 15:50 100% Liquidation",
                                            is_market_order=True
                                        )
                            else:
                                self._manage_open_positions(stk_bal, realtime_px_override=live_p)
                    except Exception as e:
                        pass
                sleep_seconds = 3 if mkt["is_open"] else 15
                time.sleep(sleep_seconds)

            except Exception as e:
                system_logger.log("ERROR", "LiveRunner", f" ? : {e}")
                time.sleep(1)

    def run_once_and_start_listener(self):
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
        logger.info(f"   ??? : {self.broker.account_no}-{self.broker.account_type}")
        logger.info(f"   ?????: ${conn_res['usd_order_available']:,.2f} USD")

        # 3. ?  ? ??????
        ai_engine_desc = "`Phase 1: 15 GBDT Model (62% ? ??/ ????Veto ?? / GBDT Feature ?)`"
        start_msg = f"""? **[Lumos ??{self.broker.mode_str} ?  ?????**
??**? ?:** `{mkt_status['now_kst_str']}`
? **? ???** `100% ??? (KST-NYT {sync_res['delta_hours']:.0f}h ? ?,  ???)`
? ** ?:** `{mkt_status['status_desc']}`
??**???:** `{mkt_status['next_open_kst_str']}` ({mkt_status['time_until_open_str']})
? **? ?** `{self.broker.broker_name} ({self.broker.mode_str})`
? ** :** `{self.broker.account_no}-{self.broker.account_type}`
? **???:** `${conn_res['usd_order_available']:,.2f} USD` (??`{conn_res['krw_converted']:,}??)
? ** ????** `{conn_res['holdings_count']}?
? **AI ?:** {ai_engine_desc}
??**? ? ?:** `3-Out Veto ???(? ? {self.daily_stoploss_count}/3??`
??**??:** `10ms WebSocket ??? ????????? ??
??, ???????? ???? ???."""
        try:
            self.dispatcher.send_telegram_message(start_msg)
            system_logger.log("INFO", "LiveRunner", f"? {self.broker.mode_str} ?  ? ??? (: {self.broker.account_no}, ?? ${conn_res['usd_order_available']:,.2f})")
        except Exception as te:
            logger.warning(f"? ? ? ?: {te}")
        
        # 4. ????WebSocket) ??? ??
        self.ws_streamer.start()
        logger.info(f"??[{self.broker.broker_name} ??WebSocket ? ???] 10ms ??????? ?")

        # 5. ????   ?????
        trd_thread = threading.Thread(target=self._market_execution_loop, daemon=True)
        trd_thread.start()

        # 6. ??? ????? ? (Foreground Daemon)
        self.controller.listen_loop(poll_interval=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Lumos Auto Trading Live Runner")
    args, unknown = parser.parse_known_args()

    is_sim = None
    if args.real:
        is_sim = False
    elif args.sim:
        is_sim = True

    runner = KiwoomLiveRunner(is_simulation=is_sim)
    runner.run_once_and_start_listener()

