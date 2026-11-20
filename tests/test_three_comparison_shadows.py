"""Real classifier gate + exact WS order/fill sequencing for three portfolios."""
import copy,json,sys,tempfile,unittest,hashlib
from pathlib import Path
from datetime import datetime,timedelta
from unittest.mock import patch,MagicMock
import numpy as np
ROOT=Path(__file__).resolve().parents[1];HOME=ROOT/'research/shadow_five_full15m_20261018'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'));sys.path.insert(0,str(HOME))
from comparison_shadows import build_comparison_shadows,ComparisonLedger,LiveModelView,load_classifier_bundle
from comparison_control import comparison_settings,legacy_tracker_enabled,comparison_runtime_enabled
from runtime import Engine,NY,BUY_WAIT
from market_events import MarketEvents
from test_relay_direction_shadows import packet

class TestThreeComparisonShadows(unittest.TestCase):
    def setUp(self):
        self.now=datetime(2026,10,21,10,0,1,tzinfo=NY)
        self.candidates=build_comparison_shadows()
        self.engine=Engine(self.candidates,evaluation_window=lambda d:True,require_price_monitoring=True)
        self.flow=MarketEvents(['SOXL','SOXS'],3.)
        self.rows=[]
    def ledger(self,key):return self.engine.ledgers[(key,'primary')]
    def emit(self,events):
        for event in events:
            if event['kind']!='MARKET_REJECTED':self.rows.extend(self.engine.handle(event))
    def quote(self,symbol,bid,delta,ask=None):
        at=self.now+timedelta(seconds=delta)
        self.emit(self.flow.frame(dict(symbol=symbol,price=bid,bid=bid,ask=bid if ask is None else ask,
                                      explicit_last=False,packet_type='QUOTE',protocol_version=2),at.timestamp(),at.timestamp()))
    def begin(self,side=1):
        self.quote('SOXL',100.,0);self.quote('SOXS',100.,0)
        view=next(iter(self.candidates.values())).view
        with patch.object(view.model,'predict_proba',return_value=np.array([[.1,.2,.7] if side==1 else [.7,.2,.1]])) as predict:
            self.rows.extend(self.engine.handle(packet(self.now)))
            self.assertEqual(predict.call_count,1)
        self.quote('SOXL' if side==1 else 'SOXS',100.,1)
    def stop_initial(self,first='SOXL'):
        other='SOXS' if first=='SOXL' else 'SOXL'
        self.quote(first,97.9,2)
        self.assertFalse(any(r['event']=='VIRTUAL_REVERSE_INTENT' for r in self.rows))
        self.quote(other,100.,2.1)
        self.assertEqual(len(self.ledger('ABS72_STOP_REVERSE').positions),1)
        self.quote(first,97.8,3)
        self.assertFalse(self.ledger('ABS72_STOP_REVERSE').positions)
        self.assertTrue(any(r['event']=='VIRTUAL_REVERSE_INTENT' for r in self.rows))
        self.quote(other,100.,4)
        return other
    def test_exactly_three_strategies_and_three_primary_ledgers(self):
        self.assertEqual(set(self.candidates),{'ABS72_SINGLE_98','ABS72_RELAY_M15','ABS72_STOP_REVERSE'})
        self.assertEqual(len(self.engine.ledgers),3)
        self.assertTrue(all(c=='primary' for n,c in self.engine.ledgers))
        self.assertFalse(legacy_tracker_enabled())
        self.assertFalse(comparison_runtime_enabled())
        self.assertEqual(len({id(c.view) for c in self.candidates.values()}),1)
        m=json.loads((ROOT/'models/active_model_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(next(iter(self.candidates.values())).view.sha256,m['expected_sha256'])
    def test_all_three_keep_live_threshold_and_neutral(self):
        view=next(iter(self.candidates.values())).view
        for i,p in enumerate([[.33,.33,.34],[.47,.48,.05]]):
            event=packet(self.now+timedelta(minutes=15*i));event['decision_id']+='_'+str(i)
            with patch.object(view.model,'predict_proba',return_value=np.array([p])):
                results=[c.predict(event) for c in self.candidates.values()]
            self.assertTrue(all(r['side']==-1 and r['threshold_applied'] for r in results))
            if i==1:
                from common import load_execution_config
                cfg=load_execution_config();expected=cfg['CALIB_BASE']+(.48-cfg['CALIB_PIVOT'])*cfg['CALIB_SCALE']
                self.assertTrue(all(abs(r['calibrated_confidence']-expected)<1e-12 for r in results))
    def test_first_completed_candle_required(self):
        p=packet(self.now.replace(hour=9,minute=30));view=next(iter(self.candidates.values())).view
        with patch.object(view.model,'predict_proba',return_value=np.array([[.1,.2,.7]])):
            self.assertTrue(all(c.predict(p)['side']==-1 for c in self.candidates.values()))
    def test_single_98_and_relay_49_quantities(self):
        self.begin()
        self.assertEqual(self.ledger('ABS72_SINGLE_98').positions[0]['qty'],97)
        self.assertEqual(self.ledger('ABS72_RELAY_M15').positions[0]['qty'],48)
        self.assertEqual(self.ledger('ABS72_STOP_REVERSE').positions[0]['qty'],48)
    def test_minus_15_only_and_fresh_next_quote_fill(self):
        self.begin();self.quote('SOXL',98.6,2)
        self.assertFalse(any(r['event']=='VIRTUAL_CHAIN_INTENT' for r in self.rows))
        self.quote('SOXL',98.5,3)
        intents=[r for r in self.rows if r['event']=='VIRTUAL_CHAIN_INTENT']
        self.assertEqual([r['model_id'] for r in intents],['ABS72_RELAY_M15'])
        self.quote('SOXS',100.,4)
        self.assertEqual(len(self.ledger('ABS72_RELAY_M15').positions),2)
        self.assertEqual(len(self.ledger('ABS72_SINGLE_98').positions),1)
        self.assertEqual(len(self.ledger('ABS72_STOP_REVERSE').positions),1)
    def test_stop_reversal_sell_confirmed_before_opposite_buy(self):
        self.begin();other=self.stop_initial()
        ledger=self.ledger('ABS72_STOP_REVERSE')
        self.assertEqual([(p['symbol'],p['role']) for p in ledger.positions],[(other,'CHAIN')])
        self.assertEqual(ledger.positions[0]['qty'],48)
        self.assertEqual(len(ledger.closed),1)
        self.assertEqual(ledger.closed[0]['reason'],'SL')
        self.assertEqual(ledger.closed[0]['exit_price'],97.8)
        self.assertEqual(ledger.cycle_number,1)
    def test_short_initial_reverses_into_long(self):
        self.begin(side=0);other=self.stop_initial('SOXS')
        self.assertEqual(other,'SOXL')
        self.assertEqual(self.ledger('ABS72_STOP_REVERSE').positions[0]['symbol'],'SOXL')
    def test_reverse_position_stop_never_creates_third_leg(self):
        self.begin();other=self.stop_initial();self.quote(other,97.9,5);self.quote(other,97.8,6)
        ledger=self.ledger('ABS72_STOP_REVERSE')
        self.assertFalse(ledger.positions);self.assertFalse(ledger.pending)
        self.assertEqual(len(ledger.cycles),1);self.assertEqual(ledger.cycles[0]['legs'],2)
        buys=[r for r in self.rows if r['event']=='VIRTUAL_BUY' and r['model_id']==ledger.model_id]
        self.assertEqual(len(buys),2)
    def test_reverse_profit_uses_chain_one_percent_trailing(self):
        self.begin();other=self.stop_initial();self.quote(other,101.1,5);self.quote(other,100.7,6)
        self.emit(self.flow.timer((self.now+timedelta(seconds=6)).timestamp()))
        self.quote(other,100.6,7)
        ledger=self.ledger('ABS72_STOP_REVERSE')
        self.assertEqual(ledger.closed[-1]['role'],'CHAIN');self.assertEqual(ledger.closed[-1]['reason'],'TRAIL')
        self.assertEqual(len(ledger.cycles),1)
    def test_no_gbdt_reentry_until_cycle_flat_and_next_bar(self):
        self.begin();self.stop_initial();ledger=self.ledger('ABS72_STOP_REVERSE')
        self.assertEqual(ledger.intend(self.now+timedelta(seconds=4),1,2.,'EXTRA','MODEL',{}),'POSITION_OPEN_OR_PENDING')
        self.quote('SOXS',97.9,5);self.quote('SOXS',97.8,6)
        self.assertEqual(ledger.intend(self.now+timedelta(seconds=6),1,2.,'EXTRA','MODEL',{}),'WAIT_NEXT_COMPLETED_15M')
    def test_time_exit_does_not_reverse(self):
        self.begin();at=self.now+timedelta(minutes=91)
        self.emit(self.flow.timer(at.timestamp()));self.quote('SOXL',99.,91*60+1)
        ledger=self.ledger('ABS72_STOP_REVERSE')
        self.assertEqual(ledger.closed[0]['reason'],'TIME');self.assertFalse(ledger.pending)
        self.assertFalse(any(r['event']=='VIRTUAL_REVERSE_INTENT' for r in self.rows))
    def test_eod_closes_both_preemptive_legs(self):
        self.begin();self.quote('SOXL',98.5,2);self.quote('SOXS',100.,3)
        at=self.now.replace(hour=15,minute=55,second=0)
        self.emit(self.flow.timer(at.timestamp()))
        delta=(at-self.now).total_seconds()+1
        self.quote('SOXL',98.,delta);self.quote('SOXS',100.,delta)
        ledger=self.ledger('ABS72_RELAY_M15')
        self.assertFalse(ledger.positions);self.assertTrue(all(r['reason']=='EOD' for r in ledger.closed))
    def test_single_never_buys_opposite_on_loss_or_stop(self):
        self.begin();self.stop_initial();ledger=self.ledger('ABS72_SINGLE_98')
        buys=[r for r in self.rows if r['event']=='VIRTUAL_BUY' and r['model_id']==ledger.model_id]
        self.assertEqual(len(buys),1);self.assertFalse(ledger.positions)
    def test_reverse_buy_timeout_releases_cycle_without_phantom_fill(self):
        self.begin();self.quote('SOXL',97.9,2);self.quote('SOXL',97.8,3)
        ledger=self.ledger('ABS72_STOP_REVERSE')
        self.assertTrue(ledger.pending);self.assertFalse(ledger.positions)
        self.emit(self.flow.timer((self.now+timedelta(seconds=3+BUY_WAIT+1)).timestamp()))
        self.assertFalse(ledger.pending);self.assertFalse(ledger.positions)
        self.assertEqual(ledger.cycles[0]['legs'],1)
    def test_stop_fill_after_entry_cutoff_does_not_reverse(self):
        self.begin()
        at=self.now.replace(hour=15,minute=30,second=0)
        # No intervening CLOCK: the observed stop wins over elapsed TIME on MARK.
        delta=(at-self.now).total_seconds()
        self.quote('SOXL',97.9,delta);self.quote('SOXL',97.8,delta+1)
        ledger=self.ledger('ABS72_STOP_REVERSE')
        self.assertFalse(ledger.positions);self.assertFalse(ledger.pending)
        self.assertTrue(any(r['event']=='VIRTUAL_REVERSE_BLOCKED_SESSION' for r in self.rows))
    def test_initial_profit_trail_does_not_reverse(self):
        self.begin();self.quote('SOXL',101.6,2);self.quote('SOXL',101.1,3)
        self.emit(self.flow.timer((self.now+timedelta(seconds=3)).timestamp()))
        self.quote('SOXL',101.,4)
        ledger=self.ledger('ABS72_STOP_REVERSE')
        self.assertEqual(ledger.closed[0]['reason'],'TRAIL');self.assertFalse(ledger.pending)
        self.assertFalse(any(r['event']=='VIRTUAL_REVERSE_INTENT' for r in self.rows))
    def test_comparison_reload_follows_live_manifest_while_open_legs_keep_version(self):
        import joblib
        import comparison_shadows as module
        self.begin();view=next(iter(self.candidates.values())).view
        old_version=self.ledger('ABS72_SINGLE_98').positions[0]['model_version']
        source=json.loads((ROOT/'models/active_model_manifest.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory(dir=ROOT/'data') as directory:
            p=Path(directory);(p/'models').mkdir();artifact=p/'next.pkl'
            joblib.dump(dict(model=view.model,feature_names=view.names,new_training_id='NEXT_WEEK_FIXTURE'),artifact)
            source['expected_sha256']=hashlib.sha256(artifact.read_bytes()).hexdigest();source['active_model_path']=str(artifact)
            (p/'models/active_model_manifest.json').write_text(json.dumps(source),encoding='utf-8')
            event=packet(self.now+timedelta(minutes=15))
            with patch.object(module,'PROJECT',p):results=[c.predict(event) for c in self.candidates.values()]
            self.assertTrue(all(r['inference_model_sha256']==source['expected_sha256'] for r in results))
            self.assertEqual(self.ledger('ABS72_SINGLE_98').positions[0]['model_version'],old_version)
    def test_disabled_tracker_does_not_open_db_or_start_thread(self):
        from core.shadow_tracker import ShadowTracker
        with patch.object(ShadowTracker,'_init_db') as init:
            tracker=ShadowTracker(enabled=False)
        init.assert_not_called();self.assertIsNone(tracker._worker_thread)
        self.assertFalse(tracker.try_enter_shadow_trade('SOXL',100.))
    def test_all_three_decisions_and_reversal_fills_write_and_read_back(self):
        import storage
        self.begin();other=self.stop_initial();self.quote(other,97.9,5);self.quote(other,97.8,6)
        with tempfile.TemporaryDirectory(dir=ROOT/'data') as directory,patch.object(storage,'HOME',Path(directory)):
            audit=storage.Audit('three_strategy_qa')
            audit.packet(packet(self.now))
            for row in self.rows:audit.append(row)
            path=audit.path;audit.close()
            saved=[json.loads(line) for line in (path/'events.jsonl').read_text(encoding='utf-8').splitlines()]
            decisions=[r for r in saved if r['event']=='DECISION']
            self.assertEqual({r['model_id'] for r in decisions},set(self.candidates))
            sells=[r for r in saved if r['event']=='VIRTUAL_SELL' and r['model_id']=='ABS72_STOP_REVERSE']
            self.assertEqual(len(sells),2);self.assertEqual(sells[0]['role'],'INITIAL')
            self.assertEqual(sells[1]['role'],'CHAIN');self.assertEqual(sells[0]['cycle_id'],sells[1]['cycle_id'])
            stored_input=json.loads((path/'decision_inputs.jsonl').read_text(encoding='utf-8'))
            self.assertIn('MACD',stored_input['relay_features'])
    def test_object_retrain_bundle_loaded_without_live_module_imports(self):
        import joblib,builtins
        from core.moe_orchestrator import MoEMetaOrchestrator
        from core.ml_engine import MLFeatureEngine
        view=next(iter(self.candidates.values())).view
        obj=MoEMetaOrchestrator.__new__(MoEMetaOrchestrator)
        obj.gbdt_engine=MLFeatureEngine.__new__(MLFeatureEngine)
        obj.gbdt_engine.model=view.model;obj.gbdt_engine.feature_names=view.names
        original=builtins.__import__
        def reject(name,*a,**k):
            if name.startswith('core.'):raise AssertionError('Observer attempted a live import')
            return original(name,*a,**k)
        with tempfile.TemporaryDirectory(dir=ROOT/'data') as directory:
            path=Path(directory)/'weekly.pkl';joblib.dump(obj,path)
            with patch('builtins.__import__',side_effect=reject):model,names=load_classifier_bundle(path)
            self.assertEqual(names,view.names);self.assertEqual(list(model.classes_),[0,1,2])

if __name__=='__main__':unittest.main()
