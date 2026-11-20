from pathlib import Path
import copy
from contextlib import closing
from datetime import datetime, timedelta
import importlib.util
import json
import sqlite3
import sys
import tempfile
import unittest
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tests'))
from core.shadow_tracker import ShadowTracker, SLOT_KEYS
import test_opposite_relay as fixtures

spec = importlib.util.spec_from_file_location('backfill_shadow_scores', ROOT/'scripts/backfill_shadow_scores_20261021.py')
backfill = importlib.util.module_from_spec(spec); spec.loader.exec_module(backfill)


class TestShadowScores(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name); self.db = self.home/'shadow.db'
        self.tracker = ShadowTracker(self.db)
        def stop():
            self.tracker._db_queue.put(None); self.tracker._worker_thread.join(timeout=10)
            self.assertFalse(self.tracker._worker_thread.is_alive())
        self.addCleanup(stop)
        self.now = datetime(2026,10,20,10,0,tzinfo=ZoneInfo('America/New_York'))
        self.snapshot = dict(event_id='SNAPSHOT',model_sha256='MODEL',inference_complete=True,
            displayed_scores=dict(LONG=.625,SHORT=.55,NONE=.8),
            raw_class_order=['LONG','SHORT','NONE'],raw_class_probabilities=[.4,.3,.3])

    def metadata(self):
        return ShadowTracker.entry_signal_from_decision(dict(learning_snapshot=self.snapshot,
            active_model_sha256='MODEL',gating_confidence=.8),'LONG')

    def enter(self, **kwargs):
        return self.tracker.try_enter_shadow_trade('SOXL',100.,current_time=self.now,**kwargs)

    def close(self):
        self.tracker.check_ticks('SOXL',97.,current_time=self.now+timedelta(seconds=1))
        self.tracker._db_queue.join()

    def test_entry_score_durable_before_close_and_closed_row_linked(self):
        self.assertTrue(self.enter(**self.metadata())); self.tracker._db_queue.join()
        with closing(sqlite3.connect(self.db)) as c, c:
            self.assertEqual(c.execute('select gbdt_confidence,learning_snapshot_id from shadow_entry_signals').fetchone(),(62.5,'SNAPSHOT'))
            self.assertEqual(c.execute('select count(*) from shadow_logs').fetchone()[0],0)
        self.close()
        with closing(sqlite3.connect(self.db)) as c, c:
            row=c.execute('select gbdt_confidence,learning_snapshot_id,model_sha256,signal_id from shadow_logs').fetchone()
            self.assertEqual(row[:3],(62.5,'SNAPSHOT','MODEL')); self.assertTrue(row[3])
        data=json.loads(self.db.with_suffix('.jsonl').read_text(encoding='utf-8').splitlines()[0])
        self.assertEqual(set(data),set(SLOT_KEYS))

    def test_neutral_score_does_not_replace_entry_direction_score(self):
        self.assertEqual(self.metadata()['confidence'],62.5)

    def test_missing_score_is_null_and_forced_entry_still_occurs(self):
        self.assertTrue(self.enter()); self.close()
        with closing(sqlite3.connect(self.db)) as c, c:
            self.assertEqual(c.execute('select gbdt_confidence,score_source from shadow_logs').fetchone(),(None,'UNAVAILABLE'))

    def test_invalid_score_is_not_fabricated(self):
        self.assertTrue(self.enter(confidence=float('nan'))); self.close()
        with closing(sqlite3.connect(self.db)) as c, c:self.assertIsNone(c.execute('select gbdt_confidence from shadow_logs').fetchone()[0])

    def test_legacy_wrapper_without_score_records_missing(self):
        self.assertTrue(self.tracker.add_virtual_trade('SOXL',100.,current_time=self.now))
        self.close()
        with closing(sqlite3.connect(self.db)) as c, c:
            self.assertIsNone(c.execute('select gbdt_confidence from shadow_logs').fetchone()[0])

    def test_model_mismatch_and_incomplete_snapshot_are_unavailable(self):
        self.snapshot['model_sha256']='OTHER'; self.assertIsNone(self.metadata()['confidence'])
        self.snapshot['model_sha256']='MODEL'; self.snapshot['inference_complete']=False
        self.assertIsNone(self.metadata()['confidence'])

    def test_slot_reuse_does_not_reuse_old_score(self):
        self.enter(**self.metadata()); self.close(); self.now+=timedelta(minutes=15)
        self.enter(); self.close()
        with closing(sqlite3.connect(self.db)) as c, c:
            self.assertEqual(c.execute('select gbdt_confidence from shadow_logs order by id').fetchall(),[(62.5,),(None,)])

    def test_real_15m_evaluator_passes_selected_direction_snapshot(self):
        case=fixtures.TestRelay(methodName='test_initial_49_and_chain_49_real_helpers_and_roles')
        case.setUp(); self.addCleanup(case.doCleanups)
        case.r.shadow_tracker=self.tracker
        decision=dict(learning_snapshot=self.snapshot,active_model_sha256='MODEL',
            is_approved=False,direction='NONE',dir_gbdt='NONE',gating_confidence=.8,
            gbdt_probs=dict(LONG=.4,SHORT=.3,NONE=.3),all_gating_confidences={},
            features={},is_cross_veto=False,is_60m_trend_ok=True,dip_ok=True)
        case.r.moe_orchestrator.evaluate_dual_filter_signal.side_effect=lambda **kwargs:decision
        case.evaluate(approved=False,direction='NONE')
        self.tracker._db_queue.join()
        with closing(sqlite3.connect(self.db)) as c, c:
            self.assertEqual(c.execute('select gbdt_confidence,learning_snapshot_id from shadow_entry_signals').fetchone(),(62.5,'SNAPSHOT'))
        self.assertFalse(case.raw.sent)

    def seed_backfill(self, symbol='SOXL', score=0.):
        with closing(sqlite3.connect(self.db)) as c, c:
            c.execute("insert into shadow_logs(symbol,virtual_entry_time,virtual_entry_price,gbdt_confidence,pnl_pct) values(?,?,?,?,?)",(symbol,'2026-10-20 10:00:00',100.,score,-1.))
        path=self.home/'learning.db'
        outcome=dict(snapshot_id='SNAPSHOT',model_sha256='MODEL',decision_time_ny='2026-10-20T10:00:00.100000-04:00')
        with closing(sqlite3.connect(path)) as c, c:
            c.execute('create table learning_events(event_type text,payload_json text)')
            c.executemany('insert into learning_events values(?,?)',[('DECISION',json.dumps(self.snapshot)),('DECISION_OUTCOME',json.dumps(outcome))])
        return path

    def test_backfill_unique_match_backup_and_idempotent(self):
        learning=self.seed_backfill(); plan=backfill.plan(self.db,learning)
        self.assertEqual(len(plan['updates']),1)
        backup=self.home/'backup.db';backfill.apply(self.db,plan,backup)
        with closing(sqlite3.connect(backup)) as c, c:self.assertEqual(c.execute('select gbdt_confidence,pnl_pct from shadow_logs').fetchone(),(0.,-1.))
        with closing(sqlite3.connect(self.db)) as c, c:self.assertEqual(c.execute('select gbdt_confidence,pnl_pct from shadow_logs').fetchone(),(62.5,-1.))
        self.assertEqual(backfill.plan(self.db,learning)['updates'],[])

    def test_backfill_preserves_existing_nonzero(self):
        learning=self.seed_backfill(score=61.)
        self.assertEqual(backfill.plan(self.db,learning)['updates'],[])

    def test_backfill_preserves_explicit_verified_zero(self):
        learning=self.seed_backfill()
        with closing(sqlite3.connect(self.db)) as c, c:
            c.execute("update shadow_logs set score_source='RECORDED_DIRECTION_DISPLAYED_SCORE'")
        self.assertEqual(backfill.plan(self.db,learning)['updates'],[])

    def test_backfill_rejects_wrong_direction(self):
        learning=self.seed_backfill(symbol='SOXS')
        self.assertEqual(backfill.plan(self.db,learning)['updates'],[])

    def test_unmatched_historical_placeholder_becomes_null_with_reason(self):
        learning=self.seed_backfill(symbol='SOXS'); plan=backfill.plan(self.db,learning)
        self.assertEqual(plan['unavailable'][0]['reason'],'DIRECTION_MISMATCH')
        backfill.apply(self.db,plan,self.home/'backup.db')
        with closing(sqlite3.connect(self.db)) as c, c:
            self.assertEqual(c.execute('select gbdt_confidence,score_source,pnl_pct from shadow_logs').fetchone(),
                (None,'UNAVAILABLE_HISTORICAL',-1.))

    def test_backfill_rejects_ambiguous_decisions(self):
        learning=self.seed_backfill()
        with closing(sqlite3.connect(learning)) as c, c:c.execute("insert into learning_events select * from learning_events where event_type='DECISION_OUTCOME'")
        self.assertEqual(backfill.plan(self.db,learning)['updates'],[])

    def test_backfill_changed_source_rolls_back(self):
        learning=self.seed_backfill(); plan=backfill.plan(self.db,learning)
        plan['updates'].append(dict(plan['updates'][0],id=999))
        with self.assertRaises(RuntimeError):backfill.apply(self.db,plan,self.home/'backup.db')
        with closing(sqlite3.connect(self.db)) as c, c:self.assertEqual(c.execute('select gbdt_confidence from shadow_logs').fetchone()[0],0.)


if __name__=='__main__':unittest.main()
