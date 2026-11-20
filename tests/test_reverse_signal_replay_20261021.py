from pathlib import Path
import importlib.util
from datetime import datetime,timedelta
import unittest
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('reverse_replay',ROOT/'scripts/temp_reverse_signal_replay_20261021.py')
replay=importlib.util.module_from_spec(spec);spec.loader.exec_module(replay)
CFG=dict(SL_MIN_PCT=.02,TRAILING_TRIGGER_PCT=.015,TRAILING_DROP_PCT=.003,
         MAX_TP_PCT=.1,TIME_STOP_MINUTES=90,BACKTEST_FEE_SLIPPAGE=.002,
         PHASE_MAIN_START='09:30',PHASE_MAIN_END='15:30',PHASE_EOD_CLEAR='15:55',
         RELAY_QUOTE_MAX_AGE_SECONDS=3.)


class TestReverseReplay(unittest.TestCase):
    def setUp(self):self.t=datetime(2026,10,20,10,0,tzinfo=ZoneInfo('America/New_York'))
    def row(self,op=100.,high=100.5,low=99.5,close=100.):
        return dict(open=op,high=high,low=low,close=close)
    def bars(self,rows,times=None,start=None):
        times=times or [self.t+timedelta(minutes=5*i) for i in range(len(rows))]
        return replay.replay_bars(start or self.t-timedelta(seconds=1),rows,times,CFG)

    def test_same_bar_stop_and_tp_stop_first(self):
        out,error=self.bars([self.row(high=111,low=97)])
        self.assertIsNone(error);self.assertEqual(out['reason'],'SL');self.assertEqual(out['exit_price'],98.)
        self.assertAlmostEqual(out['net_return_pct'],-2.2)

    def test_current_high_cannot_arm_trailing_before_current_low(self):
        out,error=self.bars([self.row(high=102,low=99.9),self.row(op=101,high=101.5,low=100,close=101)])
        self.assertIsNone(error);self.assertEqual(out['reason'],'TRAIL')
        self.assertEqual(out['exit_time'],(self.t+timedelta(minutes=5)).isoformat())

    def test_stop_gap_fills_at_worse_open(self):
        out,error=self.bars([self.row(),self.row(op=95,high=96,low=94,close=95)])
        self.assertIsNone(error);self.assertEqual(out['exit_price'],95.)

    def test_strictly_future_entry_bar(self):
        out,error=self.bars([self.row(),self.row(op=110,high=111,low=107,close=110)],start=self.t)
        self.assertIsNone(error);self.assertEqual(out['entry_price'],110.)
        self.assertEqual(out['entry_time'],(self.t+timedelta(minutes=5)).isoformat())

    def test_missing_bar_rejected(self):
        out,error=self.bars([self.row(),self.row()],times=[self.t,self.t+timedelta(minutes=10)])
        self.assertIsNone(out);self.assertEqual(error,'MISSING_5M_BAR')

    def test_time_exit_uses_open_before_bar_extremes(self):
        rows=[self.row() for _ in range(18)]+[self.row(op=100.5,high=111,low=90,close=100)]
        out,error=self.bars(rows)
        self.assertIsNone(error);self.assertEqual(out['reason'],'TIME');self.assertEqual(out['exit_price'],100.5)

    def test_eod_uses_1555_open(self):
        times=[self.t.replace(hour=15,minute=50),self.t.replace(hour=15,minute=55)]
        out,error=self.bars([self.row(),self.row(op=100.2,high=111,low=90)],times=times,start=times[0]-timedelta(seconds=1))
        self.assertEqual(error,'ENTRY_AFTER_CUTOFF')
        times=[self.t.replace(hour=15,minute=m) for m in range(25,60,5)]
        rows=[self.row() for _ in times];rows[-1]=self.row(op=100.2,high=111,low=90)
        out,error=self.bars(rows,times=times,start=times[0]-timedelta(seconds=1))
        self.assertIsNone(error);self.assertEqual(out['reason'],'EOD');self.assertEqual(out['exit_price'],100.2)

    def test_quote_exit_waits_for_next_bid(self):
        times=[self.t+timedelta(seconds=i) for i in range(3)]
        out,error=replay.replay_quotes(self.t-timedelta(seconds=1),[(100.,100.1),(97.9,98.),(97.8,97.9)],times,CFG)
        self.assertIsNone(error);self.assertEqual(out['entry_price'],100.1);self.assertEqual(out['exit_price'],97.8)
        self.assertEqual(out['exit_time'],times[-1].isoformat())

    def test_quote_gap_never_fabricates_exit(self):
        out,error=replay.replay_quotes(self.t-timedelta(seconds=1),[(100.,100.1),(97.9,98.)],[self.t,self.t+timedelta(seconds=5)],CFG)
        self.assertIsNone(out);self.assertEqual(error,'QUOTE_GAP_DURING_HOLD')


if __name__=='__main__':unittest.main()
