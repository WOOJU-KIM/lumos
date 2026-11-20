"""Learning evidence tests: synthetic prices, isolated storage, no broker or training."""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import sqlite3
import tempfile
import unittest
import threading
import time
import ast
import logging
from contextlib import closing
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
from core.ml_engine import MLFeatureEngine
from core.learning_audit import capture_features, execution_evidence
from core.live_experience_logger import LiveExperienceLogger

if os.environ.get('LUMOS_SHADOW_FEED_ISOLATED') == '1':
    tempfile.tempdir = str(Path(__file__).resolve().parents[1]/'data')


class ClassifierFixture:
    def predict_proba(self, x):
        return np.tile([.15, .25, .60], (len(x), 1))


class TestLearningCapture(unittest.TestCase):
    def frame(self):
        names = MLFeatureEngine.CANONICAL_B_NORMALIZED_72_FEATURES
        f = pd.DataFrame({c: np.full(40, .123456789123) for c in names},
                         index=pd.date_range('2026-10-16 09:30',periods=40,freq='15min'))
        f['panic_signal'] = np.full(len(f), 2, dtype=np.int64)
        f['datetime_dt'] = f.index + pd.Timedelta(minutes=15)
        for c in ['Open','High','Low','Close']: f[c] = 100.
        f['Volume'] = np.full(len(f), 1000, dtype=np.int64)
        return f

    def test_exact_inputs_raw_probabilities_and_stale_reset(self):
        names = MLFeatureEngine.CANONICAL_B_NORMALIZED_72_FEATURES
        engine = MLFeatureEngine.__new__(MLFeatureEngine)
        engine.extract_features = Mock(side_effect=lambda df: df.copy(deep=True))
        snap = SimpleNamespace(model=ClassifierFixture(), feature_names=names,
                               active_model_sha256='TEST_ONLY',feature_schema='B_normalized')
        frame = self.frame()
        output = engine.predict_signal_full(frame,snapshot=snap)
        audit = engine._get_thread_local().last_learning_snapshot
        self.assertEqual(output[0],1)
        self.assertTrue(audit['feature_complete'])
        self.assertEqual(audit['features']['panic_signal'],2.)
        self.assertEqual(audit['features']['RSI_14'],.123456789123)
        self.assertEqual(audit['raw_class_probabilities'],[.15,.25,.60])
        np.testing.assert_array_equal(audit['actual_model_input'],frame[names].iloc[-1].to_numpy())
        engine.predict_signal_full(None,snapshot=snap)
        self.assertIsNone(engine._get_thread_local().last_learning_snapshot)

    def test_missing_raw_feature_is_explicit_and_imputed_input_distinct(self):
        f = self.frame();f.iloc[-1,f.columns.get_loc('RSI_14')] = np.nan
        snap = capture_features(f,MLFeatureEngine.CANONICAL_B_NORMALIZED_72_FEATURES,'TEST','B_normalized')
        self.assertFalse(snap['feature_complete'])
        self.assertIsNone(snap['features']['RSI_14'])
        self.assertIn('RSI_14',snap['missing_or_nonfinite_features'])
        json.dumps(snap,allow_nan=False)

    def test_execution_whitelist_preserves_price_qty_not_credentials(self):
        result = execution_evidence({'order_no':'SYNTHETIC_ORDER','account_no':'DO_NOT_STORE',
            'values':{'filled_qty':'3','tot_che_qty':'5','filled_price':'101.25','fee_usd':'.15'}})
        self.assertEqual(result['incremental_quantity'],3.)
        self.assertEqual(result['cumulative_quantity'],5.)
        self.assertEqual(result['reported_fill_price'],101.25)
        self.assertEqual(result['reported_fee_usd'],.15)
        self.assertNotIn('DO_NOT_STORE',json.dumps(result))
        self.assertNotIn('SYNTHETIC_ORDER',json.dumps(result))

    def test_learning_join_rejects_simulation_and_incomplete_evidence(self):
        tree=ast.parse((Path(__file__).resolve().parents[1]/'core/ml_engine.py').read_text(encoding='utf-8-sig'))
        cl=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MLFeatureEngine')
        method=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name=='train_and_select_top_features')
        start=next(i for i,n in enumerate(method.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='target_live_df' for t in n.targets))
        end=next(i for i,n in enumerate(method.body) if isinstance(n,ast.Try) and any(isinstance(a,ast.Call) and isinstance(a.func,ast.Name) and a.func.id=='LGBMClassifier' for a in ast.walk(n)))
        names=MLFeatureEngine.CANONICAL_B_NORMALIZED_72_FEATURES
        features=capture_features(self.frame(),names,'TEST','B_normalized')['features']
        complete=dict(features,__learning_audit__={'feature_complete':True})
        incomplete=dict(features,__learning_audit__={'feature_complete':False})
        sample=dict(mode='REAL',actual_entry_price=100.,gbdt_confidence=.7,symbol='SOXL',
                    entry_time='2026-10-16 22:30:00',exit_reason='TEST',pnl_pct=1.,features_json=json.dumps(complete))
        rows=pd.DataFrame([sample,dict(sample,mode='VIRTUAL'),dict(sample,features_json=json.dumps(incomplete))])
        base=self.frame()[names].copy();report=Mock()
        import config
        scope=dict(np=np,pd=pd,config=config,target_schema='B_normalized',logger=logging.getLogger('fixture'),self=SimpleNamespace(_report_live_sample_coverage=report,last_training_window_meta={}),
                   live_experience_df=rows,clean_data=base,X=base.copy(),y=pd.Series(0,index=base.index),
                   sample_weights=pd.Series(1.,index=base.index),feature_cols=names)
        exec(compile(ast.fix_missing_locations(ast.Module(body=method.body[start:end],type_ignores=[])),
                     'REAL_PRODUCTION_SAMPLE_JOIN_NO_MODEL_FIT','exec'),scope)
        self.assertEqual(len(scope['X'])-len(base),1)
        self.assertEqual(scope['sample_weights'].iloc[-1],2.5)
        self.assertEqual(report.call_args.args[1],1)

    def test_async_storage_and_db_failure_fallback_are_readable(self):
        with tempfile.TemporaryDirectory(prefix='learning_capture_test_') as tmp:
            home = Path(tmp)
            LiveExperienceLogger._instance = None
            log = LiveExperienceLogger(home/'events.db',home/'trades.csv')
            snap = capture_features(self.frame(),MLFeatureEngine.CANONICAL_B_NORMALIZED_72_FEATURES,'TEST','B_normalized')
            log.record_learning_decision(snap)
            log.record_learning_execution({'order_no':'TEST','filled_qty':1,'filled_price':101.})
            log.record_order_event('SOXL','BUY',1,'TEST',100.,1,'FILLED')
            log.record_learning_cycle({'snapshot_id':snap['event_id'],'order_executed':True})
            log.flush_learning_events()

            with closing(sqlite3.connect(log.learning_db_path)) as db:
                self.assertEqual(db.execute('select count(*) from learning_events').fetchone()[0],4)
            trade=dict(trade_id='TEST_A',mode='TEST_ONLY',symbol='SOXL',
                entry_time='2026-10-16 22:30:00',exit_time='2026-10-16 22:31:00',
                actual_entry_price=100.,actual_exit_price=101.,quantity=1,pnl_pct=1.,pnl_usd=1.,
                exit_reason='TEST',gbdt_confidence=.7,features=snap['features'],
                learning_snapshot_id=snap['event_id'],learning_feature_names=snap['feature_names'])
            log.record_trade(trade)
            with patch.object(log,'_get_connection',side_effect=sqlite3.OperationalError('TEST_FAILURE')), \
                    patch.object(log,'_get_learning_connection',side_effect=sqlite3.OperationalError('TEST_FAILURE')):
                log.record_trade(dict(trade,trade_id='TEST_B'))
                log.record_learning_execution({'order_no':'TEST','filled_qty':1})
                log.flush_learning_events()
            self.assertTrue((home/'trades.csv.learning.jsonl').exists())
            rows = log.get_trades_dataframe()
            self.assertEqual(set(rows.trade_id),{'TEST_A','TEST_B'})
            f = json.loads(rows.iloc[0].features_json)
            self.assertEqual(f['panic_signal'],2.)
            self.assertEqual(f['RSI_14'],.123456789123)
            self.assertEqual(f['__learning_audit__']['fee_status'],'UNKNOWN_NOT_ZERO')
            self.assertEqual(log._learning_failures,[])
            log.flush_learning_events()


    def test_slow_storage_does_not_block_enqueue(self):
        with tempfile.TemporaryDirectory(prefix='learning_enqueue_test_') as tmp:
            home = Path(tmp)
            LiveExperienceLogger._instance = None
            log = LiveExperienceLogger(home/'events.db',home/'trades.csv')
            snap = capture_features(self.frame(),MLFeatureEngine.CANONICAL_B_NORMALIZED_72_FEATURES,'TEST','B_normalized')
            entered, release = threading.Event(), threading.Event()
            original = log._get_learning_connection
            def slow():
                entered.set();release.wait(2)
                return original()
            try:
                with patch.object(log,'_get_learning_connection',side_effect=slow):
                    start = time.perf_counter()
                    log.record_learning_decision(snap)
                    self.assertLess(time.perf_counter()-start,.1)
                    self.assertTrue(entered.wait(1))
                    self.assertFalse(release.is_set())
                    release.set();log.flush_learning_events()
            finally:
                release.set();log.flush_learning_events()


if __name__=='__main__': unittest.main()
