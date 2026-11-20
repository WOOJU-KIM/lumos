"""Persistent admission and failure isolation; synthetic prices, no broker."""
import sys
import datetime
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.shadow_tracker import ShadowTracker

class TestShadowAdmissionPersistence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "shadow.db"
        self.now = datetime.datetime(2026, 10, 14, 9, 45, tzinfo=ZoneInfo("America/New_York"))

    def tracker(self, path=None):
        obj = ShadowTracker(path or self.path)
        def stop():
            obj._db_queue.put(None)
            obj._worker_thread.join(timeout=10)
            self.assertFalse(obj._worker_thread.is_alive())
        self.addCleanup(stop)
        return obj

    def enter(self, tracker, **kwargs):
        return tracker.try_enter_shadow_trade("SOXL", kwargs.pop("price", 30.0),
                                              current_time=kwargs.pop("now", self.now), **kwargs)

    def test_recreated_instance_blocks_same_and_accepts_next_candle(self):
        first = self.tracker()
        self.assertTrue(self.enter(first))
        second = self.tracker()
        self.assertFalse(self.enter(second))
        self.assertTrue(self.enter(second, now=self.now + datetime.timedelta(minutes=15)))

    def test_older_candle_replay_is_blocked_after_next_admission(self):
        tracker = self.tracker()
        self.assertTrue(self.enter(tracker))
        self.assertTrue(self.enter(tracker, now=self.now + datetime.timedelta(minutes=15)))
        self.assertFalse(self.enter(tracker))

    def test_atomic_concurrent_claim_has_one_winner(self):
        trackers = [self.tracker(), self.tracker()]
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(self.enter, trackers))
        self.assertEqual(sum(results), 1)

    def test_full_slots_do_not_consume_admission(self):
        first = self.tracker()
        for slot in first.slots:
            slot["active"] = True
        self.assertFalse(self.enter(first))
        self.assertTrue(self.enter(self.tracker()))

    def test_invalid_prices_do_not_consume_admission(self):
        first = self.tracker()
        for price in (None, 0.0, -1.0, float("nan"), float("inf")):
            self.assertFalse(self.enter(first, price=price))
        self.assertTrue(self.enter(first))

    def test_claim_storage_failure_rejects_without_activating(self):
        tracker = self.tracker()
        with patch("core.shadow_tracker.os.open", side_effect=OSError("test storage unavailable")):
            self.assertFalse(self.enter(tracker))
        self.assertFalse(any(slot["active"] for slot in tracker.slots))
        self.assertIsNone(tracker.last_entered_candle_id)
        self.assertTrue(self.enter(tracker))

    def test_sqlite_logging_failure_does_not_prevent_claim(self):
        tracker = self.tracker()
        with patch("core.shadow_tracker.sqlite3.connect", side_effect=OSError("test DB down")):
            self.assertTrue(self.enter(tracker))
        self.assertFalse(self.enter(self.tracker()))

    def test_distinct_database_namespaces_are_independent(self):
        self.assertTrue(self.enter(self.tracker()))
        self.assertTrue(self.enter(self.tracker(Path(self.tmp.name) / "other.db")))

if __name__ == "__main__":
    unittest.main()
