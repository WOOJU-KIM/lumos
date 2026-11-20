"""Entry ablation, cash-funded two-leg cycle and feed integration regressions."""
import copy
import json
import sys
import unittest
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
import numpy as np

HOME=Path(__file__).resolve().parents[1]/'research/shadow_five_full15m_20261018'
sys.path.insert(0,str(HOME))
from relay_directions import DirectionRelay, RelayLedger, build_direction_shadows, settings
from runtime import Engine, CONFIG, NY
from market_events import MarketEvents
from test_shadow_exit_monitoring import packet as base_packet


def packet(now):
    result=base_packet(now)
    names=json.JSONDecoder().decode((HOME/'relay_directions.json').read_text(encoding='utf-8')).get('relay_model', {}).get('feature_names', [])
    result['relay_features']={name:0. for name in names if name not in result['full_features']}
    return result


class TestRelayDirectionSignals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidates=build_direction_shadows()

    def result(self, name, probs, at=None):
        candidate=copy.copy(self.candidates[name]);candidate.model=MagicMock()
        candidate.model.predict_proba.return_value=np.array([probs])
        return candidate.predict(packet(at or datetime(2026,10,20,10,0,1,tzinfo=NY)))

    def test_actual_model_sha_and_two_separate_relay_ledgers(self):
        self.assertEqual(len(self.candidates),2)
        engine=Engine(self.candidates)
        self.assertEqual(len(engine.ledgers),2)
        self.assertTrue(all(isinstance(l,RelayLedger) for l in engine.ledgers.values()))
        active=json.loads((HOME.parents[1]/'models/active_model_manifest.json').read_text())
        self.assertTrue(all(c.settings['artifact_sha256']==active['expected_sha256'] for c in self.candidates.values()))

    def test_score_threshold_removed_in_both_variants(self):
        for name in self.candidates:
            result=self.result(name,[.33,.33,.34])
            self.assertEqual(result['side'],1)
            self.assertFalse(result['threshold_applied'])
            self.assertLess(result['calibrated_confidence'],CONFIG['GBDT_CONFIDENCE_THRESHOLD'])

    def test_neutral_winner_only_blocks_preserved_variant(self):
        self.assertEqual(self.result('RELAY_B72_NO_THRESHOLD',[.08,.8,.12])['side'],-1)
        self.assertEqual(self.result('RELAY_B72_DIRECTION_ONLY',[.08,.8,.12])['side'],1)
        self.assertEqual(self.result('RELAY_B72_DIRECTION_ONLY',[.12,.8,.08])['side'],0)

    def test_ties_are_explicit(self):
        self.assertEqual(self.result('RELAY_B72_NO_THRESHOLD',[.4,.4,.2])['side'],-1)
        self.assertEqual(self.result('RELAY_B72_DIRECTION_ONLY',[.1,.8,.1])['side'],1)
        self.assertEqual(self.result('RELAY_B72_NO_THRESHOLD',[.4,.2,.4])['side'],1)

    def test_model_switch_refuses_silent_old_classifier(self):
        with patch('relay_directions.json.loads',return_value={'expected_sha256':'CHANGED'}):
            with self.assertRaises(RuntimeError):
                self.result('RELAY_B72_DIRECTION_ONLY',[.3,.2,.5])

    def test_bad_probabilities_are_not_forced_into_trade(self):
        for probs in ([0,0,0],[float('nan'),.2,.8],[-.1,.2,.9]):
            with self.subTest(probs=probs),self.assertRaises(ValueError):
                self.result('RELAY_B72_DIRECTION_ONLY',probs)

    def test_open_context_cannot_become_first_entry(self):
        at=datetime(2026,10,20,9,30,1,tzinfo=NY)
        self.assertEqual(self.result('RELAY_B72_DIRECTION_ONLY',[.1,.1,.8],at)['side'],-1)

    def test_absolute_inputs_export_exact_values_without_changing_frozen_schema(self):
        import struct
        import pandas as pd
        import feed_bridge
        from core.ml_engine import MLFeatureEngine
        from runtime import validate_packet, SCHEMA
        names=feed_bridge.FEATURE_NAMES
        self.assertTrue(set(MLFeatureEngine.DOLLAR_SCALE_FEATURES)<=set(names))
        self.assertEqual(len(SCHEMA['net_features']),72)
        at=datetime(2026,10,20,10,0,1,tzinfo=NY)
        times=pd.date_range('2026-10-19 08:00',periods=32,freq='15min')
        frame=pd.DataFrame({n:np.arange(32,dtype=float)+j+.125 for j,n in enumerate(names)})
        frame['datetime_dt']=times
        with patch.object(feed_bridge,'_qa_allow',True),patch.object(feed_bridge,'connected',return_value=True),patch.object(feed_bridge,'publish') as publish:
            feed_bridge.features(frame)
        kind,payload=publish.call_args.args
        self.assertEqual(kind,2);self.assertLess(len(payload),feed_bridge.SLOT-feed_bridge.PREFIX.size)
        rows,cols,size=struct.unpack_from('<HHH',payload)
        exported=payload[6:6+size].decode('ascii').split('\0')
        values=np.frombuffer(payload,dtype='<f8',offset=6+size+rows*8).reshape(rows,cols)
        self.assertEqual(exported,names)
        np.testing.assert_array_equal(values,frame[names].to_numpy())
        p=packet(at)
        p['relay_features']={n:float(frame[n].iloc[-1]) for n in p['relay_features']}
        validate_packet(p)
        original_hash=validate_packet(p)[1]
        p['relay_features']['MACD']+=1
        self.assertNotEqual(validate_packet(p)[1],original_hash)
        p['relay_features']['MACD']=float('nan')
        with self.assertRaises(ValueError):validate_packet(p)


class TestRelayShadowCycles(unittest.TestCase):
    def setUp(self):
        self.now=datetime(2026,10,20,10,0,1,tzinfo=NY)
        self.long,self.short=CONFIG['BASE_ASSET_LONG'],CONFIG['BASE_ASSET_SHORT']
        self.ledger=RelayLedger('FIXTURE')
        self.flow=MarketEvents([self.long,self.short],CONFIG['MAIN_LOOP_TICK_OPEN'])
        self.events=[]
        self.quote(self.long,100.,at=self.now-timedelta(seconds=1))
        self.quote(self.short,100.,at=self.now-timedelta(seconds=.5))

    def quote(self,symbol,bid,ask=None,at=None):
        at=at or self.now
        decoded=dict(symbol=symbol,price=bid,bid=bid,ask=ask or bid,explicit_last=False,
                     packet_type='QUOTE',protocol_version=2)
        for event in self.flow.frame(decoded,at.timestamp(),at.timestamp()):
            rows=self.ledger.on_quote(event) if event['kind']=='QUOTE' else self.ledger.market(event)
            self.events.extend(rows)

    def start(self,side=1):
        self.assertEqual(self.ledger.intend(self.now,side,2.,'D','MODEL',{}),'PENDING_NEXT_QUOTE')
        self.now+=timedelta(seconds=.2)
        self.quote(self.long if side else self.short,100.)
        self.assertEqual(len(self.ledger.positions),1)

    def chain(self,side=1):
        self.start(side)
        source=self.long if side else self.short;target=self.short if side else self.long
        self.now+=timedelta(seconds=.2);self.quote(source,99.)
        self.assertEqual(len(self.ledger.pending),1)
        self.now+=timedelta(seconds=.2);self.quote(target,100.)
        self.assertEqual(len(self.ledger.positions),2)
        return source,target

    def test_symmetric_49_budget_and_next_quote_fill(self):
        for side in (0,1):
            self.setUp()
            source,target=self.chain(side)
            buys=[e for e in self.events if e['event']=='VIRTUAL_BUY']
            self.assertEqual([e['role'] for e in buys],['INITIAL','CHAIN'])
            self.assertTrue(all(e['quantity']*e['price']<=4900 for e in buys))
            self.assertGreaterEqual(self.ledger.cash,200.)
            self.assertEqual({p['symbol'] for p in self.ledger.positions},{source,target})

    def test_no_chain_above_one_percent_boundary(self):
        self.start();self.now+=timedelta(seconds=.2);self.quote(self.long,99.01)
        self.assertFalse(self.ledger.pending)
        self.now+=timedelta(seconds=.2);self.quote(self.long,99.)
        self.assertEqual(self.ledger.pending[0]['role'],'CHAIN')

    def test_rebound_before_chain_fill_cancels_and_cannot_buy_again(self):
        self.start();self.now+=timedelta(seconds=.2);self.quote(self.long,99.)
        self.now+=timedelta(seconds=.2);self.quote(self.long,100.)
        self.now+=timedelta(seconds=.2);self.quote(self.short,100.)
        self.assertEqual(len(self.ledger.positions),1)
        self.assertFalse(self.ledger.pending)
        self.now+=timedelta(seconds=.2);self.quote(self.long,99.)
        self.assertFalse(self.ledger.pending)

    def test_no_rebuy_after_initial_sells_and_gbdt_cannot_enter(self):
        source,target=self.chain()
        self.now+=timedelta(seconds=.2);self.quote(source,98.)
        self.now+=timedelta(seconds=.2);self.quote(source,98.)
        self.assertEqual([p['symbol'] for p in self.ledger.positions],[target])
        self.now+=timedelta(seconds=.2);self.quote(target,99.)
        self.assertFalse(self.ledger.pending)
        self.assertEqual(self.ledger.intend(self.now,1,2.,'NEW','MODEL',{}),'POSITION_OPEN_OR_PENDING')

    def test_no_rebuy_after_chain_sells_first(self):
        source,target=self.chain()
        self.now+=timedelta(seconds=.2);self.quote(source,100.)
        self.now+=timedelta(seconds=.2);self.quote(target,98.)
        self.now+=timedelta(seconds=.2);self.quote(target,98.)
        self.now+=timedelta(seconds=.2);self.quote(source,99.)
        self.assertEqual(len(self.ledger.positions),1)
        self.assertFalse(self.ledger.pending)

    def test_closed_cycle_cash_cost_and_next_bar_only(self):
        source,target=self.chain()
        for symbol in (source,target):
            self.now+=timedelta(seconds=.2);self.quote(symbol,98.)
            self.now+=timedelta(seconds=.2);self.quote(symbol,98.)
        self.assertFalse(self.ledger.positions)
        self.assertFalse(self.ledger.pending)
        cycle=self.ledger.cycles[-1]
        self.assertEqual(cycle['legs'],2)
        self.assertAlmostEqual(cycle['net_pnl'],sum(t['net_pnl'] for t in self.ledger.closed))
        self.assertTrue(all(t['cost_reserve']>=.002 for t in self.ledger.closed))
        self.assertAlmostEqual(self.ledger.cash,10000.+cycle['net_pnl'])
        self.assertEqual(self.ledger.intend(self.now,1,2.,'SAME','MODEL',{}),'WAIT_NEXT_COMPLETED_15M')
        self.now=self.now.replace(minute=15,second=1,microsecond=0)
        self.quote(source,100.);self.quote(target,100.)
        self.assertEqual(self.ledger.intend(self.now,1,2.,'NEXT','MODEL',{}),'PENDING_NEXT_QUOTE')
        self.assertFalse(self.ledger.chain_filled)

    def test_role_trailing_and_polling(self):
        source,target=self.chain()
        self.now+=timedelta(seconds=.2);self.quote(source,101.2)
        self.now+=timedelta(seconds=.2);self.quote(source,100.8)
        self.ledger.market(dict(kind='POLL',time=str(self.now),symbol=source))
        self.assertFalse(self.ledger.pending)
        self.now+=timedelta(seconds=.2);self.quote(target,101.)
        self.now+=timedelta(seconds=.2);self.quote(target,100.69)
        self.assertFalse(self.ledger.pending)
        self.ledger.market(dict(kind='POLL',time=str(self.now),symbol=target))
        self.assertEqual(self.ledger.pending[-1]['reason'],'TRAIL')
        self.assertEqual(self.ledger.pending[-1]['position']['role'],'CHAIN')

    def test_clock_exits_even_without_quotes_and_pending_exit_blocks_entry(self):
        self.start();at=self.now.replace(hour=15,minute=55)
        rows=self.ledger.market(dict(kind='CLOCK',time=str(at)))
        self.assertEqual(rows[0]['reason'],'EOD')
        self.assertEqual(self.ledger.intend(at,1,2.,'EOD','MODEL',{}),'POSITION_OPEN_OR_PENDING')
        self.now=at+timedelta(seconds=1);self.quote(self.long,90.)
        self.assertEqual(self.ledger.closed[-1]['reason'],'EOD')
        self.assertEqual(len(self.ledger.cycles),1)

    def test_time_stop_without_price(self):
        self.start()
        self.now+=timedelta(minutes=CONFIG['TIME_STOP_MINUTES'])
        self.ledger.market(dict(kind='CLOCK',time=str(self.now)))
        self.assertEqual(self.ledger.pending[-1]['reason'],'TIME')

    def test_unproven_quote_or_stale_opposite_cannot_enter(self):
        self.start();self.now+=timedelta(seconds=4)
        self.quote(self.long,99.)
        self.assertFalse(self.ledger.pending)
        before=len(self.ledger.positions)
        self.ledger.on_quote(dict(kind='QUOTE',time=str(self.now),quote_time=str(self.now),symbol=self.short,bid=100.,ask=100.))
        self.assertEqual(len(self.ledger.positions),before)

    def test_stale_intent_no_imaginary_fill(self):
        self.assertEqual(self.ledger.intend(self.now,1,2.,'D','MODEL',{}),'PENDING_NEXT_QUOTE')
        self.now+=timedelta(seconds=7);self.quote(self.long,100.)
        self.assertFalse(self.ledger.positions)
        self.assertEqual(self.ledger.cash,10000.)

    def test_quote_observed_before_intent_cannot_fill_after_arrival_delay(self):
        self.assertEqual(self.ledger.intend(self.now,1,2.,'D','MODEL',{}),'PENDING_NEXT_QUOTE')
        observed=self.now-timedelta(seconds=.1)
        rows=self.ledger.on_quote(dict(kind='QUOTE',time=str(self.now+timedelta(seconds=.2)),
            quote_time=str(observed),observed_at=str(observed),symbol=self.long,bid=100.,ask=100.))
        self.assertFalse(rows)
        self.assertFalse(self.ledger.positions)

    def test_initial_pending_cannot_fill_after_new_15m_boundary(self):
        self.now=self.now.replace(minute=14,second=59,microsecond=0)
        self.quote(self.long,100.);self.quote(self.short,100.)
        self.assertEqual(self.ledger.intend(self.now,1,2.,'D','MODEL',{}),'PENDING_NEXT_QUOTE')
        self.now+=timedelta(seconds=2);self.quote(self.long,100.)
        self.assertFalse(self.ledger.positions)
        self.assertTrue(any(e['event']=='BUY_CANCELLED_NEW_15M_BOUNDARY' for e in self.events))

    def test_shared_feed_both_variants_receive_independent_cycles(self):
        candidates=build_direction_shadows()
        for c in candidates.values():
            c.model=MagicMock();c.model.predict_proba.return_value=np.array([[.08,.8,.12]])
        engine=Engine(candidates,require_price_monitoring=True)
        flow=MarketEvents([self.long,self.short],CONFIG['MAIN_LOOP_TICK_OPEN'])
        for symbol in (self.long,self.short):
            for event in flow.frame(dict(symbol=symbol,price=100.,bid=100.,ask=100.,explicit_last=False,
                                         packet_type='QUOTE',protocol_version=2),self.now.timestamp(),self.now.timestamp()):
                engine.handle(event)
        rows=engine.handle(packet(self.now))
        actions={r['model_id']:r['primary_action'] for r in rows}
        self.assertEqual(actions['RELAY_B72_NO_THRESHOLD'],'MODEL_NONE')
        self.assertEqual(actions['RELAY_B72_DIRECTION_ONLY'],'PENDING_NEXT_QUOTE')


if __name__=='__main__':unittest.main()
