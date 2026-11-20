"""Live 5m SOXX ORB + VWAP + RVOL + 49%/49% Opposite Relay tests."""
import tempfile
import threading
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo
import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import test_opposite_relay as relay_fixture
from core.pending_order import binding_for
from core.relay_live_runner import RelayLiveRunner, _Broker, _Quotes
from core.relay_policy import RelayBook, exit_reason


def _make_soxx_5m_df(day_str="2026-11-02", signal_type="LONG"):
    """Build synthetic SOXX 5m candles with previous-session history + 09:30..10:00 bars."""
    ny = ZoneInfo("America/New_York")
    rows = []
    # 80 preceding 5m bars from previous session with volume = 10,000 (>= 72 bars for snapshot check)
    prev_base = datetime.fromisoformat(f"2026-10-30T09:30:00").replace(tzinfo=ny)
    for i in range(80):
        ts = prev_base + timedelta(minutes=5 * i)
        rows.append(dict(datetime=ts, Open=220.0, High=220.5, Low=219.5, Close=220.0, Volume=10000.0))

    # 6 ORB bars (09:30..09:55): Open_0930 = 220.0, ORB_High = 221.5, ORB_Low = 219.5 (Range = 2.0 -> 0.909% >= 0.45%)
    day_open = datetime.fromisoformat(f"{day_str}T09:30:00").replace(tzinfo=ny)
    orb_specs = [
        (220.0, 221.0, 219.8, 220.5, 10000.0),
        (220.5, 221.5, 220.0, 221.0, 10000.0),
        (221.0, 221.2, 219.5, 220.2, 10000.0),
        (220.2, 220.8, 219.9, 220.4, 10000.0),
        (220.4, 221.1, 220.1, 220.6, 10000.0),
        (220.6, 221.3, 220.2, 220.8, 10000.0),
    ]
    for i, (o, h, l, c, v) in enumerate(orb_specs):
        ts = day_open + timedelta(minutes=5 * i)
        rows.append(dict(datetime=ts, Open=o, High=h, Low=l, Close=c, Volume=v))

    # 10:00 bar (evaluated at 10:05 close)
    ts_1000 = day_open + timedelta(minutes=30)
    if signal_type == "LONG":
        # Close 222.2 > ORB_High (221.5) & > VWAP, Volume 18000 -> RVOL = 1.8 >= 1.5
        rows.append(dict(datetime=ts_1000, Open=221.0, High=222.5, Low=220.9, Close=222.2, Volume=18000.0))
    elif signal_type == "SHORT":
        # Close 218.5 < ORB_Low (219.5) & < VWAP, Volume 18000 -> RVOL = 1.8 >= 1.5
        rows.append(dict(datetime=ts_1000, Open=219.8, High=219.9, Low=218.2, Close=218.5, Volume=18000.0))
    else:
        # Inside ORB range -> no signal
        rows.append(dict(datetime=ts_1000, Open=220.5, High=220.8, Low=220.2, Close=220.5, Volume=12000.0))

    df = pd.DataFrame(rows).set_index("datetime")
    return df


class TestSoxxOrbLiveRelay(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / "data")
        self.addCleanup(self.tmp.cleanup)
        self.now = datetime(2026, 11, 2, 10, 5, tzinfo=ZoneInfo("America/New_York"))
        p = patch(
            "core.relay_live_runner.USMarketCalendar.get_market_status",
            return_value=dict(is_open=True, session_name="REGULAR_MARKET_OPEN"),
        )
        p.start()
        self.addCleanup(p.stop)

        r = RelayLiveRunner.__new__(RelayLiveRunner)
        self.r = r
        r._scope = threading.local()
        r._relay_lock = threading.RLock()
        r._quote_lock = threading.Lock()
        r._extremes = {}
        r._risk_hits = {}
        r._entry_mutex = threading.Lock()
        r._risk_mutex = threading.Lock()
        r._order_lock = threading.Lock()
        r._wake = threading.Event()
        r._stop = threading.Event()
        r._quotes = {}
        r._last_poll = {}
        r._positions = {}
        r._journal_failed = False
        r._legacy_present = False
        r._mismatch = None
        r._deployment_ok = True
        r._is_order_in_progress = False
        r._order_fills = {}
        r._empty_balance_streak = 0
        r._unreconciled_holding_mismatch = None
        r._pending_buy_write_failed = False
        r.is_running = False
        r._now = lambda: self.now
        r._is_circuit_breaker_halted = lambda: (False, "")

        # Enable 5m SOXX ORB live mode
        r._soxx_orb_enabled = True
        r._eval_interval_minutes = 5
        r._last_5m_trigger_key = None
        r._soxx_orb_daily_trades = {}
        r._soxx_orb_logged_dates = set()
        r._soxx_orb_log_path = Path(self.tmp.name) / "soxx_orb_live_trades.log"
        r._soxx_orb_jsonl_path = Path(self.tmp.name) / "soxx_orb_live_events.jsonl"

        raw = relay_fixture.Broker()
        r._raw_broker = raw
        r._binding = binding_for(raw)
        r._root = Path(self.tmp.name)
        r.book = RelayBook(r._root / "book.json", r._binding)
        r.broker = _Broker(r, raw)
        self.raw = raw

        original = MagicMock()
        original.get_latest_price.return_value = 100.0
        r.ws_streamer = _Quotes(r, original)
        r.shadow_tracker = MagicMock()
        r.notifier = MagicMock()
        r.experience_logger = MagicMock()
        r.dispatcher = MagicMock()
        r.moe_orchestrator = MagicMock()
        r.moe_orchestrator.gbdt_engine.get_active_model_info.return_value = {}
        r.moe_orchestrator.evaluate_dual_filter_signal = r._evaluate_soxx_orb_5m_signal
        r.order_timeout_buy_sec = config.BUY_TIMEOUT_SEC
        r.order_timeout_sell_sec = config.SELL_TIMEOUT_SEC

        def wait(order_no, symbol, is_buy, target_qty, **kwargs):
            r._last_wait_fill_stk_bal = raw.get_overseas_stock_balance()
            held = raw.holdings.get(symbol, (0, 0))[0]
            return (held >= target_qty, held, max(0, target_qty - held)) if is_buy else (held == 0, target_qty - held, held)

        r._wait_for_fill = wait
        r._dispatch_sell = lambda **kwargs: r._execute_sell_with_10s_chase(**kwargs)
        r._emit_model_state_alerts_from_eval = MagicMock()
        r._get_5m_atr_pct = lambda *a: 1.0
        r._collect_cycle_market_snapshot = MagicMock(
            return_value=dict(
                live_prices={config.BASE_ASSET_LONG: 100.0, config.BASE_ASSET_SHORT: 100.0, "SOXX": 222.2},
                quote_time_et="2026-11-02 10:05 ET",
                candle_time_et="2026-11-02 10:00 ET",
                data_stale=False,
                stale_reason="",
            )
        )

        self.quote(config.BASE_ASSET_LONG, 100.0, 100.0)
        self.quote(config.BASE_ASSET_SHORT, 100.0, 100.0)
        r._decision = r._decision_key(self.now)

    def quote(self, symbol, bid, ask=None):
        self.r._on_websocket_tick(symbol, bid, dict(from_ws=True, extra=dict(values={"51": bid, "41": ask or bid})))

    def manage(self):
        self.r._manage_open_positions(self.raw.get_overseas_stock_balance())

    def test_5m_soxx_orb_long_49_chain_49_exits_and_cycle_repeat(self):
        df_5m = _make_soxx_5m_df("2026-11-02", "LONG")
        self.r.data_lake = MagicMock()
        self.r.data_lake.load_candles.return_value = df_5m

        # 1. 10:05 NYT evaluation of 10:00 5m SOXX bar -> LONG -> Buy SOXL 98%
        self.quote(config.BASE_ASSET_LONG, 100.0, 100.0)
        self.quote(config.BASE_ASSET_SHORT, 100.0, 100.0)
        res = self.r.evaluate_15m_cycle_once(
            allow_order_execution=True,
            sync_intraday=True,
            now_dt=self.now,
            market_session="REGULAR_MARKET_OPEN",
        )
        self.assertTrue(res["order_executed"])
        self.assertEqual(res["winner_symbol"], "SOXL")
        self.assertIn("SOXL", self.r._positions)
        self.assertEqual(self.r._positions["SOXL"]["relay_role"], "INITIAL")
        self.assertTrue(self.r._positions["SOXL"]["soxx_orb_strategy"])
        self.assertEqual(self.r._positions["SOXL"]["trailing_trigger_pct"], 0.01)
        self.assertEqual(self.r._positions["SOXL"]["quantity"], 97)

        # 2. SOXL hits -1.5% -> CHAIN is disabled (0% chain allocation), so SOXS is NOT bought
        soxl_entry = self.r._positions["SOXL"]["price"]
        self.quote("SOXL", round(soxl_entry * 0.984, 2))
        self.manage()
        self.assertNotIn("SOXS", self.r._positions)

        # 3. Verify 90m time stop and +10% fixed TP are disabled for soxx_orb_strategy
        pos_soxl = self.r._positions["SOXL"]
        self.assertIsNone(exit_reason(pos_soxl, None, self.now + timedelta(minutes=120), True))
        self.assertIsNone(exit_reason(pos_soxl, pos_soxl["price"] * 1.11, self.now, False))

        # 4. SOXL arms trailing at +1.0% and exits on -0.3% drop from peak -> cycle closes
        peak_px = round(soxl_entry * 1.015, 2)
        self.quote("SOXL", peak_px)
        self.manage()
        self.r._last_poll["SOXL"] = 0.0
        self.quote("SOXL", round(peak_px * (1 - 0.0035), 2))
        self.manage()
        self.assertFalse(self.r._positions)
        self.assertIsNone(self.r.book.data["budget"])
        self.assertIsNone(self.r.book.data["cycle_id"])

        # 6. Next 5m bar (10:10 NYT, evaluating 10:05 bar) -> SHORT -> immediately enters new cycle!
        self.now = datetime(2026, 11, 2, 10, 10, tzinfo=ZoneInfo("America/New_York"))
        self.quote("SOXL", 100.0, 100.0)
        self.quote("SOXS", 100.0, 100.0)
        df_next = df_5m.copy()
        df_next.loc[datetime(2026, 11, 2, 10, 5, tzinfo=ZoneInfo("America/New_York"))] = dict(
            Open=220.0, High=220.1, Low=218.0, Close=218.5, Volume=20000.0
        )
        self.r.data_lake.load_candles.return_value = df_next
        res2 = self.r.evaluate_15m_cycle_once(
            allow_order_execution=True,
            sync_intraday=True,
            now_dt=self.now,
            market_session="REGULAR_MARKET_OPEN",
        )
        self.assertTrue(res2["order_executed"])
        self.assertEqual(res2["winner_symbol"], "SOXS")
        self.assertIn("SOXS", self.r._positions)

        # 7. EOD 15:55 liquidates all remaining positions
        self.now = datetime(2026, 11, 2, 15, 55, tzinfo=ZoneInfo("America/New_York"))
        self.manage()
        self.assertFalse(self.r._positions)
        self.assertFalse(self.raw.holdings)


if __name__ == "__main__":
    unittest.main()
