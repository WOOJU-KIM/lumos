"""Continuous shadow schedule only: never imports/starts the real Runner."""
import datetime,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1]
HOME=ROOT/"research/shadow_five_full15m_20261018"
sys.path.insert(0,str(HOME))
from runtime_control import load_control,date_allowed,should_stop
from runtime import Ledger,Engine
import start_shadow

class TestShadowRuntimeControl(unittest.TestCase):
    def setUp(self):
        self.control={"schema_version":1,"enabled":True,"start_date_ny":"2026-10-19","end_date_ny":None}
    def at(self,text):return datetime.datetime.fromisoformat(text).replace(tzinfo=ZoneInfo("America/New_York"))
    def enter(self,ledger,text):return ledger.intend(self.at(text),1,1.,"test-cycle","test-version",{})
    def test_no_expiry_after_original_week(self):
        for text in ["2026-10-26","2026-11-02","2027-10-18"]:
            with self.subTest(day=text):
                self.assertTrue(date_allowed(text,self.control))
                self.assertFalse(should_stop(text,self.control))
    def test_start_boundary_and_listener_wait(self):
        self.assertFalse(date_allowed("2026-10-18",self.control))
        self.assertFalse(should_stop("2026-10-18",self.control))
        self.assertTrue(date_allowed("2026-10-19",self.control))
    def test_both_cohorts_accept_later_week(self):
        engine=Engine({"test":object()},evaluation_window=lambda day:date_allowed(day,self.control))
        for cohort in ["primary","forced"]:
            self.assertEqual(self.enter(engine.ledgers[("test",cohort)],"2026-10-26T10:00:00"),"PENDING_NEXT_QUOTE")
    def test_offline_replay_stays_frozen(self):
        self.assertEqual(self.enter(Ledger("test"),"2026-10-26T10:00:00"),"ENTRY_SESSION_BLOCK")
    def test_session_cutoff_and_weekend_preserved(self):
        for text in ["2026-10-26T09:29:00","2026-10-26T15:30:00","2026-10-24T10:00:00"]:
            ledger=Ledger("test",evaluation_window=lambda d:date_allowed(d,self.control))
            self.assertEqual(self.enter(ledger,text),"ENTRY_SESSION_BLOCK")
    def test_existing_position_guard_preserved(self):
        ledger=Ledger("test",evaluation_window=lambda d:date_allowed(d,self.control));ledger.positions.append({})
        self.assertEqual(self.enter(ledger,"2026-10-26T10:00:00"),"POSITION_OPEN_OR_PENDING")
    def test_disabled_blocks_entries_and_requests_stop(self):
        self.control["enabled"]=False
        self.assertFalse(date_allowed("2026-10-26",self.control));self.assertTrue(should_stop("2026-10-26",self.control))
    def test_control_update_applies_to_existing_ledger(self):
        ledger=Ledger("test",evaluation_window=lambda d:date_allowed(d,self.control))
        self.control["enabled"]=False
        self.assertEqual(self.enter(ledger,"2026-10-26T10:00:00"),"ENTRY_SESSION_BLOCK")
    def test_disabled_launcher_cannot_respawn(self):
        self.control["enabled"]=False
        with patch.object(start_shadow,"load_control",return_value=self.control),patch.object(start_shadow,"connected") as connected,patch.object(start_shadow.subprocess,"Popen") as spawn:
            self.assertEqual(start_shadow._start_or_reuse()["status"],"SHADOW_DISABLED_OR_EXPIRED")
        spawn.assert_not_called();connected.assert_not_called()
    def test_optional_finite_window_boundaries(self):
        self.control["end_date_ny"]="2026-10-23"
        self.assertTrue(date_allowed("2026-10-23",self.control));self.assertFalse(should_stop("2026-10-23",self.control))
        self.assertFalse(date_allowed("2026-10-26",self.control));self.assertTrue(should_stop("2026-10-26",self.control))
    def test_validate_file_and_missing_or_invalid_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"control.json"
            with self.assertRaises(FileNotFoundError):load_control(path)
            path.write_text(json.dumps(self.control));self.assertEqual(load_control(path),self.control)
            for bad in [dict(self.control,enabled="true"),dict(self.control,end_date_ny="2026-10-18"),dict(self.control,start_date_ny="invalid")]:
                path.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):load_control(path)

if __name__=="__main__":unittest.main()
