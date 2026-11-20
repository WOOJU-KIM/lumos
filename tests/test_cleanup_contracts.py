"""Public recovery compatibility and shadow audit correlation, no network."""
import sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"research/shadow_five_full15m_20261018"))
from core.live_runner import KiwoomLiveRunner
from core.pending_order import journal_path
from runtime import Ledger

class TestPendingBuyCleanupContracts(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/"position.json"
    def runner(self):
        obj=KiwoomLiveRunner.__new__(KiwoomLiveRunner)
        obj.pos_file_path=str(self.path)
        obj.broker=SimpleNamespace(account_no="FIXTURE",account_type="TEST",is_simulation=True)
        return obj
    def test_explicit_io_survives_recreation_and_clears_durably(self):
        first=self.runner()
        first._store_pending_buy_order({"symbol":"SOXL","order_no":"FIXTURE","target_qty":4})
        self.assertTrue(journal_path(self.path).is_file())
        second=self.runner()
        self.assertEqual(second._load_pending_buy_order()["order_no"],"FIXTURE")
        second._clear_pending_buy_order()
        self.assertFalse(journal_path(self.path).exists())
        self.assertIsNone(self.runner()._load_pending_buy_order())
    def test_legacy_property_still_reads_and_writes_same_journal(self):
        runner=self.runner()
        runner._unconfirmed_buy_order={"symbol":"SOXL","target_qty":4}
        self.assertEqual(self.runner()._load_pending_buy_order()["target_qty"],4)
        runner._clear_pending_buy_order()
        self.assertIsNone(runner._unconfirmed_buy_order)
    def test_failed_explicit_save_retains_memory_and_failure_latch(self):
        runner=self.runner()
        with patch("core.pending_order.os.replace",side_effect=OSError("TEST_STORAGE_FAILURE")):
            with self.assertRaises(OSError):runner._store_pending_buy_order({"symbol":"SOXL","target_qty":4})
        self.assertTrue(runner._pending_buy_write_failed)
        self.assertEqual(runner._load_pending_buy_order()["target_qty"],4)

class TestShadowAuditCorrelation(unittest.TestCase):
    def time(self,text):return datetime.fromisoformat(text).replace(tzinfo=ZoneInfo("America/New_York"))
    def ledger(self,time="2026-10-19T10:00:00"):
        ledger=Ledger("TEST_MODEL")
        self.assertEqual(ledger.intend(self.time(time),1,1.,"TEST_DECISION","TEST_VERSION",{}),"PENDING_NEXT_QUOTE")
        return ledger
    def quote(self,time,bid=99.9,ask=100.):
        return {"kind":"QUOTE","symbol":"SOXL","time":self.time(time).isoformat(),"quote_time":self.time(time).isoformat(),"bid":bid,"ask":ask}
    def identities(self,row):
        self.assertEqual({k:row[k] for k in ["model_id","cohort","decision_id","model_version"]},
                         {"model_id":"TEST_MODEL","cohort":"primary","decision_id":"TEST_DECISION","model_version":"TEST_VERSION"})
    def test_buy_exit_intent_sell_are_joinable(self):
        ledger=self.ledger()
        buy=ledger.on_quote(self.quote("2026-10-19T10:00:01"))[0]
        self.assertEqual(buy["event"],"VIRTUAL_BUY");self.identities(buy)
        intent=ledger.market({"kind":"TICK","time":self.time("2026-10-19T10:00:02").isoformat(),"symbol":"SOXL","price":97.})[0]
        self.assertEqual(intent["event"],"VIRTUAL_EXIT_INTENT");self.identities(intent)
        sell=ledger.on_quote(self.quote("2026-10-19T10:00:03",bid=97.,ask=97.1))[0]
        self.assertEqual(sell["event"],"VIRTUAL_SELL");self.identities(sell)
    def test_quote_stale_intent_cancel_is_joinable(self):
        row=self.ledger().on_quote(self.quote("2026-10-19T10:00:20"))[0]
        self.assertEqual(row["event"],"BUY_CANCELLED_STALE_INTENT");self.identities(row)
    def test_session_cancel_is_joinable(self):
        row=self.ledger("2026-10-19T15:29:59").on_quote(self.quote("2026-10-19T15:30:00"))[0]
        self.assertEqual(row["event"],"BUY_CANCELLED_SESSION");self.identities(row)
    def test_clock_stale_intent_cancel_is_joinable(self):
        row=self.ledger().market({"kind":"CLOCK","time":self.time("2026-10-19T10:00:20").isoformat()})[0]
        self.assertEqual(row["event"],"BUY_CANCELLED_STALE_INTENT");self.identities(row)

if __name__=="__main__":unittest.main()
