"""Unit and integration tests for SOXX 5m ORB + VWAP + RVOL shadow strategy."""
import csv
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT / "research/shadow_five_full15m_20261018"
sys.path.insert(0, str(HOME))

from common import protected_changes
from comparison_shadows import build_comparison_shadows
from market_events import MarketEvents
from runtime import Engine, CONFIG
from soxx_orb_shadow import (
    CSV_HEADER,
    SoxxOrbLedger,
    attach_soxx_orb_shadow,
    load_soxx_orb_settings,
)

NY = ZoneInfo("America/New_York")


class TestSoxxOrbShadow(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=str(HOME / "cache"))
        self.addCleanup(self.tmp.cleanup)
        self.tmp_dir = Path(self.tmp.name)
        self.csv_path = self.tmp_dir / "trades.csv"
        self.log_path = self.tmp_dir / "trades.log"
        self.ledger = SoxxOrbLedger(
            csv_path=self.csv_path,
            log_path=self.log_path,
            evaluation_window=lambda day: True,
        )
        self.flow = MarketEvents(["SOXL", "SOXS", "SOXX"], CONFIG["MAIN_LOOP_TICK_OPEN"])

    def _seed_prior_20_volumes(self, vol=1000.0):
        """Seed 20 prior 5m bars from the previous trading day so RVOL SMA_20 is warm."""
        base = datetime(2026, 10, 29, 14, 0, tzinfo=NY)
        for i in range(20):
            bar_dt = base + timedelta(minutes=5 * i)
            self.ledger.on_soxx_5m_bar(
                bar_time=bar_dt,
                open_px=200.0,
                high_px=200.5,
                low_px=199.5,
                close_px=200.0,
                volume=vol,
                allow_trade=False,
            )

    def _seed_orb_0930_to_1000(
        self,
        day_str="2026-10-30",
        open_0930=200.0,
        orb_high=201.5,
        orb_low=199.5,
        vol=1000.0,
    ):
        """Feed the 6 Opening Range 5m candles (09:30, 09:35, 09:40, 09:45, 09:50, 09:55)."""
        hms = ["09:30", "09:35", "09:40", "09:45", "09:50", "09:55"]
        mid = (orb_high + orb_low) / 2.0
        for idx, hm in enumerate(hms):
            dt = datetime.fromisoformat(f"{day_str}T{hm}:00").replace(tzinfo=NY)
            o = open_0930 if idx == 0 else mid
            h = orb_high if idx == 1 else max(o, mid)
            l = orb_low if idx == 2 else min(o, mid)
            c = mid
            self.ledger.on_soxx_5m_bar(
                bar_time=dt,
                open_px=o,
                high_px=h,
                low_px=l,
                close_px=c,
                volume=vol,
                eval_time=dt + timedelta(minutes=5),
            )

    def _quote(self, symbol, bid, ask, at):
        decoded = dict(
            symbol=symbol,
            price=bid,
            bid=bid,
            ask=ask,
            explicit_last=False,
            packet_type="QUOTE",
            protocol_version=2,
        )
        outputs = []
        for ev in self.flow.frame(decoded, at.timestamp(), at.timestamp()):
            if ev["kind"] == "QUOTE":
                outputs.extend(self.ledger.on_quote(ev))
            else:
                outputs.extend(self.ledger.market(ev))
        return outputs

    def test_orb_vwap_rvol_indicator_formulas_and_daily_reset(self):
        self._seed_prior_20_volumes(vol=1000.0)
        # ORB: Open_0930=200.0, High=202.0, Low=199.0 -> ORB_Range=3.0 (1.5% >= 0.45%)
        self._seed_orb_0930_to_1000(
            day_str="2026-10-30",
            open_0930=200.0,
            orb_high=202.0,
            orb_low=199.0,
            vol=1000.0,
        )
        self.assertTrue(self.ledger.orb_ready)
        self.assertAlmostEqual(self.ledger.soxx_open_0930, 200.0)
        self.assertAlmostEqual(self.ledger.orb_high, 202.0)
        self.assertAlmostEqual(self.ledger.orb_low, 199.0)
        self.assertAlmostEqual(self.ledger.orb_range, 3.0)

        # Next bar at 10:00 with Volume=2000 against preceding 20 bars of 1000 -> RVOL = 2.0
        dt_1000 = datetime(2026, 10, 30, 10, 0, tzinfo=NY)
        self.ledger.on_soxx_5m_bar(
            bar_time=dt_1000,
            open_px=201.5,
            high_px=203.0,
            low_px=201.5,
            close_px=202.5,
            volume=2000.0,
            eval_time=dt_1000 + timedelta(minutes=5),
        )
        self.assertAlmostEqual(self.ledger.rvol, 2.0)

        # Verify VWAP resets at 09:30 on the next trading day
        dt_next = datetime(2026, 11, 2, 9, 30, tzinfo=NY)
        self.ledger.on_soxx_5m_bar(
            bar_time=dt_next,
            open_px=210.0,
            high_px=213.0,
            low_px=207.0,
            close_px=210.0,
            volume=500.0,
            eval_time=dt_next + timedelta(minutes=5),
            allow_trade=False,
        )
        # Typical price = (213 + 207 + 210) / 3 = 210.0
        self.assertAlmostEqual(self.ledger.vwap, 210.0)
        self.assertEqual(self.ledger.daily_trade_count, 0)
        self.assertFalse(self.ledger.orb_ready)

    def test_entry_filters_orb_range_time_window_and_daily_limit(self):
        self._seed_prior_20_volumes(vol=1000.0)
        # 1) Narrow ORB range: Open=200, High=200.4, Low=199.8 -> Range=0.6 (0.30% < 0.45%)
        self._seed_orb_0930_to_1000(
            day_str="2026-10-30",
            open_0930=200.0,
            orb_high=200.4,
            orb_low=199.8,
            vol=1000.0,
        )
        dt_1000 = datetime(2026, 10, 30, 10, 0, tzinfo=NY)
        rows = self.ledger.on_soxx_5m_bar(
            bar_time=dt_1000,
            open_px=200.3,
            high_px=201.5,
            low_px=200.2,
            close_px=201.2,
            volume=2000.0,
            eval_time=dt_1000 + timedelta(minutes=5),
        )
        self.assertEqual(rows, [])
        self.assertFalse(self.ledger.pending)

        # 2) Valid ORB range on next day, but bar at 14:35 (outside 10:00 ~ 14:30 window)
        self._seed_orb_0930_to_1000(
            day_str="2026-11-02",
            open_0930=200.0,
            orb_high=201.5,
            orb_low=199.5,
            vol=1000.0,
        )
        dt_1435 = datetime(2026, 11, 2, 14, 35, tzinfo=NY)
        rows_late = self.ledger.on_soxx_5m_bar(
            bar_time=dt_1435,
            open_px=201.0,
            high_px=203.0,
            low_px=201.0,
            close_px=202.5,
            volume=2000.0,
            eval_time=dt_1435 + timedelta(minutes=5),
        )
        self.assertEqual(rows_late, [])
        self.assertFalse(self.ledger.pending)

    def test_long_entry_soxl_95_allocation_and_trailing_stop_exit(self):
        self._seed_prior_20_volumes(vol=1000.0)
        self._seed_orb_0930_to_1000(
            day_str="2026-10-30",
            open_0930=200.0,
            orb_high=201.5,
            orb_low=199.5,
            vol=1000.0,
        )
        # 10:00 5m bar closes at 10:05: SOXX_Close=202.2 > ORB_High(201.5) & > VWAP, RVOL=1.8 >= 1.5
        dt_1000 = datetime(2026, 10, 30, 10, 0, tzinfo=NY)
        eval_dt = dt_1000 + timedelta(minutes=5)
        rows = self.ledger.on_soxx_5m_bar(
            bar_time=dt_1000,
            open_px=201.0,
            high_px=202.5,
            low_px=201.0,
            close_px=202.2,
            volume=1800.0,
            eval_time=eval_dt,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["event"], "VIRTUAL_BUY_INTENT")
        self.assertEqual(rows[0]["symbol"], "SOXL")

        # Next observed WS quote fills SOXL at Ask=50.00
        # 49% of $10,000 = $4,900 -> Qty = int(4900 / 50.0) = 98
        fill_time = eval_dt + timedelta(seconds=1)
        buy_events = self._quote("SOXL", 49.98, 50.00, fill_time)
        buys = [e for e in buy_events if e["event"] == "VIRTUAL_BUY"]
        self.assertEqual(len(buys), 1)
        self.assertEqual(buys[0]["symbol"], "SOXL")
        self.assertEqual(buys[0]["role"], "INITIAL")
        self.assertEqual(buys[0]["quantity"], 98)
        self.assertAlmostEqual(buys[0]["price"], 50.00)
        self.assertEqual(self.ledger.daily_trade_count, 1)

        # Price rises to 50.75 (+1.5% trailing trigger reached)
        t_peak = fill_time + timedelta(seconds=30)
        self._quote("SOXL", 50.75, 50.77, t_peak)
        self.assertAlmostEqual(self.ledger.position["highest_etf"], 50.75)
        self.assertFalse(self.ledger.position["exit_pending"])

        # Price pulls back to 50.59 (-0.315% <= -0.3% from 50.75) -> triggers Trailing exit intent
        t_drop = t_peak + timedelta(seconds=10)
        drop_events = self._quote("SOXL", 50.59, 50.61, t_drop)
        self.assertTrue(any(e["event"] == "VIRTUAL_EXIT_INTENT" and e["reason"] == "Trailing" for e in drop_events))

        # Next observed WS quote fills SELL at Bid=50.58
        t_sell = t_drop + timedelta(seconds=1)
        sell_events = self._quote("SOXL", 50.58, 50.60, t_sell)
        sells = [e for e in sell_events if e["event"] == "VIRTUAL_SELL"]
        self.assertEqual(len(sells), 1)
        sell = sells[0]
        self.assertEqual(sell["reason"], "Trailing")
        self.assertEqual(sell["symbol"], "SOXL")
        self.assertEqual(sell["quantity"], 98)

        # Verify exact Gross_PnL, Total_Fee (0.07% each side), and Net_PnL
        expected_gross = (50.58 - 50.00) * 98
        expected_fee = (50.00 * 98 * 0.0007) + (50.58 * 98 * 0.0007)
        expected_net = expected_gross - expected_fee
        self.assertAlmostEqual(sell["gross_pnl"], expected_gross, places=6)
        self.assertAlmostEqual(sell["total_fee"], expected_fee, places=6)
        self.assertAlmostEqual(sell["net_pnl"], expected_net, places=6)
        self.assertAlmostEqual(self.ledger.cash, 10000.0 + expected_net, places=6)

        # Verify CSV file header and row
        with self.csv_path.open("r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
        self.assertEqual(reader[0], CSV_HEADER)
        self.assertEqual(len(reader), 2)
        self.assertEqual(reader[1][3], "SOXL")
        self.assertEqual(reader[1][9], "Trailing")

    def test_short_entry_soxs_and_all_other_exit_reasons_stop_vwap_eod(self):
        self._seed_prior_20_volumes(vol=1000.0)
        self._seed_orb_0930_to_1000(
            day_str="2026-10-30",
            open_0930=200.0,
            orb_high=201.0,
            orb_low=199.0,
            vol=1000.0,
        )
        # 1) Short Entry (SOXS) at 10:00 (closes at 10:05): SOXX_Close=198.0 < ORB_Low(199.0) & < VWAP, RVOL=2.0
        dt_1000 = datetime(2026, 10, 30, 10, 0, tzinfo=NY)
        eval_1 = dt_1000 + timedelta(minutes=5)
        self.ledger.on_soxx_5m_bar(
            bar_time=dt_1000,
            open_px=199.2,
            high_px=199.2,
            low_px=197.8,
            close_px=198.0,
            volume=2000.0,
            eval_time=eval_1,
        )
        self._quote("SOXS", 24.98, 25.00, eval_1 + timedelta(seconds=1))
        self.assertIsNotNone(self.ledger.position)
        self.assertEqual(self.ledger.position["symbol"], "SOXS")

        # Exit #1: At -1.0% (24.75), CHAIN intent for SOXL is triggered; at -2.0% (24.50), SOXS Stop triggers
        self._quote("SOXS", 24.80, 24.82, eval_1 + timedelta(seconds=5))
        self.assertFalse(self.ledger.position["exit_pending"])
        self.assertFalse(self.ledger.chain_triggered)
        self._quote("SOXS", 24.50, 24.52, eval_1 + timedelta(seconds=10))
        self._quote("SOXS", 24.48, 24.50, eval_1 + timedelta(seconds=11))
        self.assertIsNone(self.ledger.position)
        self.assertEqual(self.ledger.closed[-1]["reason"], "Stop")

        # 2) Second entry on same day (daily_trade_count == 1 < 2): Long SOXL at 10:15
        dt_1015 = datetime(2026, 10, 30, 10, 15, tzinfo=NY)
        eval_2 = dt_1015 + timedelta(minutes=5)
        self.ledger.on_soxx_5m_bar(
            bar_time=dt_1015,
            open_px=200.5,
            high_px=203.0,
            low_px=200.5,
            close_px=202.5,
            volume=2500.0,
            eval_time=eval_2,
        )
        self._quote("SOXL", 39.98, 40.00, eval_2 + timedelta(seconds=1))
        self.assertEqual(self.ledger.daily_trade_count, 2)
        self.assertEqual(self.ledger.position["symbol"], "SOXL")

        # Verify 5m candle close below VWAP does NOT trigger an exit (VWAP exit removed)
        dt_1020 = datetime(2026, 10, 30, 10, 20, tzinfo=NY)
        eval_3 = dt_1020 + timedelta(minutes=5)
        self.ledger.on_soxx_5m_bar(
            bar_time=dt_1020,
            open_px=202.5,
            high_px=212.0,
            low_px=202.5,
            close_px=210.0,
            volume=20000.0,
            eval_time=eval_3,
        )
        self.assertGreater(self.ledger.vwap, 204.0)
        dt_1025 = datetime(2026, 10, 30, 10, 25, tzinfo=NY)
        eval_4 = dt_1025 + timedelta(minutes=5)
        vwap_events = self.ledger.on_soxx_5m_bar(
            bar_time=dt_1025,
            open_px=205.0,
            high_px=205.0,
            low_px=201.7,
            close_px=201.8,
            volume=1000.0,
            eval_time=eval_4,
        )
        self.assertEqual(vwap_events, [])
        self.assertIsNotNone(self.ledger.position)
        self.assertFalse(self.ledger.position["exit_pending"])

        # Exit #2 via Trailing (+1.5% -> 40.60, then -0.3% -> 40.47)
        self._quote("SOXL", 40.60, 40.62, eval_4 + timedelta(seconds=1))
        self._quote("SOXL", 40.47, 40.49, eval_4 + timedelta(seconds=2))
        self._quote("SOXL", 40.46, 40.48, eval_4 + timedelta(seconds=3))
        self.assertIsNone(self.ledger.position)
        self.assertEqual(self.ledger.closed[-1]["reason"], "Trailing")

        # 3) Third entry attempt on the same day succeeds because max_daily_trades == 100
        self.assertEqual(self.ledger.max_daily_trades, 100)
        dt_1030 = datetime(2026, 10, 30, 10, 30, tzinfo=NY)
        third_rows = self.ledger.on_soxx_5m_bar(
            bar_time=dt_1030,
            open_px=205.0,
            high_px=215.0,
            low_px=205.0,
            close_px=214.0,
            volume=5000.0,
            eval_time=dt_1030 + timedelta(minutes=5),
        )
        self.assertEqual(len(third_rows), 1)
        self.assertEqual(third_rows[0]["event"], "VIRTUAL_BUY_INTENT")
        self.ledger.pending.clear()

        # 4) Once daily_trade_count reaches 100, further entries on the same day are blocked
        self.ledger.daily_trade_count = 100
        dt_1035 = datetime(2026, 10, 30, 10, 35, tzinfo=NY)
        blocked_rows = self.ledger.on_soxx_5m_bar(
            bar_time=dt_1035,
            open_px=205.0,
            high_px=215.0,
            low_px=205.0,
            close_px=214.0,
            volume=5000.0,
            eval_time=dt_1035 + timedelta(minutes=5),
        )
        self.assertEqual(blocked_rows, [])

    def test_soxx_orb_stop_removed_only_etf_2pct_stop_and_eod_1555_exit(self):
        self._seed_prior_20_volumes(vol=1000.0)
        self._seed_orb_0930_to_1000(
            day_str="2026-10-30",
            open_0930=200.0,
            orb_high=202.0,
            orb_low=198.0,
            vol=1000.0,
        )
        # ORB_Range = 4.0
        dt_1000 = datetime(2026, 10, 30, 10, 0, tzinfo=NY)
        eval_1 = dt_1000 + timedelta(minutes=5)
        self.ledger.on_soxx_5m_bar(
            bar_time=dt_1000,
            open_px=201.5,
            high_px=203.5,
            low_px=201.5,
            close_px=203.0,
            volume=2000.0,
            eval_time=eval_1,
        )
        self._quote("SOXL", 49.98, 50.00, eval_1 + timedelta(seconds=1))
        # SOXX drops by 0.5 * ORB_Range (from 203.0 to 200.95), but ORB_Range stop is removed so position stays open!
        self._quote("SOXX", 200.95, 201.00, eval_1 + timedelta(seconds=15))
        self.assertIsNotNone(self.ledger.position)
        self.assertFalse(self.ledger.position["exit_pending"])

        # Only ETF -2.0% (50.00 -> 49.00) triggers Stop
        self._quote("SOXL", 49.00, 49.02, eval_1 + timedelta(seconds=16))
        self._quote("SOXL", 48.98, 49.00, eval_1 + timedelta(seconds=17))
        self.assertIsNone(self.ledger.position)
        self.assertEqual(self.ledger.closed[-1]["reason"], "Stop")

        # Re-enter at 11:00 and verify EOD 15:55 forced exit
        dt_1100 = datetime(2026, 10, 30, 11, 0, tzinfo=NY)
        eval_2 = dt_1100 + timedelta(minutes=5)
        self.ledger.on_soxx_5m_bar(
            bar_time=dt_1100,
            open_px=202.0,
            high_px=204.5,
            low_px=202.0,
            close_px=204.0,
            volume=2500.0,
            eval_time=eval_2,
        )
        self._quote("SOXL", 50.00, 50.02, eval_2 + timedelta(seconds=1))
        self.assertIsNotNone(self.ledger.position)

        eod_time = datetime(2026, 10, 30, 15, 55, 0, tzinfo=NY)
        clock_events = self.ledger.market(dict(kind="CLOCK", time=eod_time.isoformat()))
        self.assertTrue(any(e["event"] == "VIRTUAL_EXIT_INTENT" and e["reason"] == "EOD" for e in clock_events))
        self._quote("SOXL", 50.50, 50.52, eod_time + timedelta(seconds=1))
        self.assertIsNone(self.ledger.position)
        self.assertEqual(self.ledger.closed[-1]["reason"], "EOD")

    def test_opposite_chain_buy_49_49_cycle_and_repeat_after_all_flat(self):
        self._seed_prior_20_volumes(vol=1000.0)
        self._seed_orb_0930_to_1000(
            day_str="2026-10-30",
            open_0930=200.0,
            orb_high=201.5,
            orb_low=199.5,
            vol=1000.0,
        )
        # 1) Initial Long (SOXL) signal at 10:00 (closes at 10:05)
        dt_1000 = datetime(2026, 10, 30, 10, 0, tzinfo=NY)
        eval_1 = dt_1000 + timedelta(minutes=5)
        self.ledger.on_soxx_5m_bar(
            bar_time=dt_1000,
            open_px=201.0,
            high_px=202.5,
            low_px=201.0,
            close_px=202.2,
            volume=1800.0,
            eval_time=eval_1,
        )
        # Fill INITIAL SOXL @ $50.00 -> 49% of $10,000 = 98 shares ($4,900)
        self._quote("SOXL", 49.98, 50.00, eval_1 + timedelta(seconds=1))
        self.assertEqual(len(self.ledger.positions), 1)
        self.assertEqual(self.ledger.positions[0]["role"], "INITIAL")
        self.assertEqual(self.ledger.positions[0]["quantity" if "quantity" in self.ledger.positions[0] else "qty"], 98)

        # 2) At -1.0% ($49.50), CHAIN does NOT trigger because chain_trigger_loss_pct is -1.5% (-0.015)
        no_chain_events = self._quote("SOXL", 49.50, 49.52, eval_1 + timedelta(seconds=4))
        self.assertFalse(any(e["event"] == "VIRTUAL_CHAIN_INTENT" for e in no_chain_events))
        self.assertFalse(self.ledger.chain_triggered)

        # SOXL drops -1.5% to $49.25 -> triggers opposite CHAIN buy intent for SOXS (49%)
        chain_events = self._quote("SOXL", 49.25, 49.27, eval_1 + timedelta(seconds=5))
        self.assertTrue(any(e["event"] == "VIRTUAL_CHAIN_INTENT" and e["symbol"] == "SOXS" for e in chain_events))

        # Next SOXS quote fills CHAIN leg @ $25.00 -> 49% of $10,000 = 196 shares ($4,900)
        chain_fill = self._quote("SOXS", 24.98, 25.00, eval_1 + timedelta(seconds=6))
        chain_buys = [e for e in chain_fill if e["event"] == "VIRTUAL_BUY" and e["role"] == "CHAIN"]
        self.assertEqual(len(chain_buys), 1)
        self.assertEqual(chain_buys[0]["symbol"], "SOXS")
        self.assertEqual(chain_buys[0]["quantity"], 196)
        self.assertEqual(len(self.ledger.positions), 2)

        # 3) Initial SOXL hits -2.0% Stop ($49.00) and closes; CHAIN SOXS remains open
        self._quote("SOXL", 49.00, 49.02, eval_1 + timedelta(seconds=10))
        self._quote("SOXL", 48.98, 49.00, eval_1 + timedelta(seconds=11))
        self.assertEqual(len(self.ledger.positions), 1)
        self.assertEqual(self.ledger.positions[0]["symbol"], "SOXS")
        self.assertEqual(self.ledger.positions[0]["role"], "CHAIN")
        # Cycle is NOT closed yet because SOXS is still held
        self.assertEqual(len(self.ledger.cycles), 0)

        # 4) CHAIN SOXS rises +1.6% ($25.40 >= +1.5% trigger) then drops -0.35% ($25.31) -> Trailing exit
        self._quote("SOXS", 25.40, 25.42, eval_1 + timedelta(seconds=20))
        self._quote("SOXS", 25.31, 25.33, eval_1 + timedelta(seconds=25))
        close_cycle_events = self._quote("SOXS", 25.30, 25.32, eval_1 + timedelta(seconds=26))
        self.assertEqual(len(self.ledger.positions), 0)
        self.assertEqual(len(self.ledger.cycles), 1)
        self.assertTrue(any(e["event"] == "VIRTUAL_CYCLE_CLOSED" and e["legs"] == 2 for e in close_cycle_events))

        # 5) All positions are now empty -> next 5m breakout bar immediately starts Cycle #2
        dt_1015 = datetime(2026, 10, 30, 10, 15, tzinfo=NY)
        eval_2 = dt_1015 + timedelta(minutes=5)
        next_cycle_rows = self.ledger.on_soxx_5m_bar(
            bar_time=dt_1015,
            open_px=202.0,
            high_px=204.0,
            low_px=202.0,
            close_px=203.5,
            volume=2200.0,
            eval_time=eval_2,
        )
        self.assertEqual(len(next_cycle_rows), 1)
        self.assertEqual(next_cycle_rows[0]["event"], "VIRTUAL_BUY_INTENT")
        self.assertEqual(next_cycle_rows[0]["role"], "INITIAL")
        self.assertEqual(self.ledger.cycle_number, 2)

        # Verify structured .log and .jsonl files contain full diagnostic events
        log_text = self.log_path.read_text(encoding="utf-8")
        for tag in ("[ORB_READY]", "[ENTRY_SIGNAL]", "[VIRTUAL_BUY]", "[CHAIN_TRIGGER]", "[TRAILING_ARMED]", "[EXIT_INTENT]", "[VIRTUAL_SELL]", "[CYCLE_CLOSED]"):
            self.assertIn(tag, log_text)
        jsonl_lines = self.ledger.jsonl_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertGreaterEqual(len(jsonl_lines), 8)

    def test_engine_attachment_and_readonly_sqlite_sync_without_touching_protected_files(self):
        db_path = self.tmp_dir / "market_test.db"
        conn = sqlite3.connect(db_path)
        conn.execute(
            "CREATE TABLE market_candles (symbol TEXT, timeframe TEXT, datetime TEXT, "
            "open REAL, high REAL, low REAL, close REAL, volume REAL)"
        )
        # Insert 20 prior 5m bars + 6 ORB bars + 1 breakout bar at 10:00
        base = datetime(2026, 10, 29, 14, 0)
        for i in range(20):
            dt_str = (base + timedelta(minutes=5 * i)).strftime("%Y-%m-%d %H:%M:%S")
            conn.execute(
                "INSERT INTO market_candles VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("SOXX", "5m", dt_str, 200.0, 200.5, 199.5, 200.0, 1000.0),
            )
        for idx, hm in enumerate(["09:30", "09:35", "09:40", "09:45", "09:50", "09:55"]):
            conn.execute(
                "INSERT INTO market_candles VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("SOXX", "5m", f"2026-10-30 {hm}:00", 200.0, 202.0 if idx == 1 else 201.0, 199.0 if idx == 2 else 200.0, 200.5, 1000.0),
            )
        conn.execute(
            "INSERT INTO market_candles VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("SOXX", "5m", "2026-10-30 10:00:00", 201.0, 203.5, 201.0, 203.0, 2000.0),
        )
        conn.commit()
        conn.close()

        ro_conn = sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)
        self.addCleanup(ro_conn.close)

        from comparison_control import comparison_runtime_enabled, legacy_tracker_enabled
        self.assertFalse(comparison_runtime_enabled())
        self.assertFalse(legacy_tracker_enabled())

        candidates = build_comparison_shadows() if comparison_runtime_enabled() else {}
        engine = Engine(candidates, evaluation_window=lambda d: True, require_price_monitoring=True)
        soxx_ledger = attach_soxx_orb_shadow(
            engine,
            db=ro_conn,
            evaluation_window=lambda d: True,
            csv_path=self.csv_path,
            log_path=self.log_path,
        )
        self.assertIsNotNone(soxx_ledger)
        self.assertEqual(list(engine.candidates), ["SOXX_ORB_VWAP_RVOL_M15"])
        self.assertEqual(list(engine.ledgers), [("SOXX_ORB_VWAP_RVOL_M15", "primary")])
        self.assertAlmostEqual(soxx_ledger.chain_trigger_loss_pct, -0.015)

        # Clock event at 10:05:01 syncs the 10:00 bar from SQLite and emits VIRTUAL_BUY_INTENT
        t_clock = datetime(2026, 10, 30, 10, 5, 1, tzinfo=NY)
        out = engine.handle(dict(kind="CLOCK", time=t_clock.isoformat()))
        self.assertTrue(any(e.get("event") == "VIRTUAL_BUY_INTENT" and e.get("model_id") == "SOXX_ORB_VWAP_RVOL_M15" for e in out))

        # Verify zero protected files were modified against active_execution_strategy.json
        import json
        from common import sha
        active_strategy = json.loads((ROOT / "models/active_execution_strategy.json").read_text(encoding="utf-8"))
        for rel_path, expected_hash in active_strategy["source_hashes"].items():
            self.assertEqual(sha(ROOT / rel_path), expected_hash, f"Protected file {rel_path} modified")
        active_manifest = json.loads((ROOT / "models/active_model_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(sha(ROOT / active_manifest["active_model_path"]), active_manifest["expected_sha256"])


if __name__ == "__main__":
    unittest.main()
