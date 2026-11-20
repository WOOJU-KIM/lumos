"""Virtual-only feed -> five ledgers -> sell-fill regression. No broker imports."""
from pathlib import Path
import sys
import json
import unittest
from datetime import datetime, timedelta

HOME = Path(__file__).resolve().parents[1] / 'research/shadow_five_full15m_20261018'
sys.path.insert(0, str(HOME))
from market_events import MarketEvents, decode_frame, TICK_V1, TICK_V2, MARKET_V2, EXPLICIT_LAST
from runtime import Engine, CONFIG, POLICY, SCHEMA, NY


class Candidate:
    version = 'FIXTURE_ONLY'
    def __init__(self, name, side):
        self.name, self.side = name, side
    def predict(self, packet):
        return dict(model_id=self.name, model_version=self.version,
                    side=self.side, forced_side=self.side,
                    expected_net_return=[.01, .01], threshold=0.)


def packet(now):
    boundary = now.replace(second=0, microsecond=0)
    return dict(kind='DECISION', schema_version=1, time=now.isoformat(),
                decision_time=boundary.isoformat(), decision_id='fixture_' + boundary.isoformat(),
                net_feature_names=SCHEMA['net_features'], sequence=[[0.] * 72 for _ in range(32)],
                full_features={k: 50. if k=='RSI_14' else 0. for k in SCHEMA['full_features']},
                feature_times=[(boundary-timedelta(minutes=15*i)).isoformat() for i in range(31,-1,-1)],
                expected_last_closed_available_at=boundary.isoformat(),
                required_assets_complete=True, cross_asset_available_at={},
                atr5m_pct={CONFIG['BASE_ASSET_LONG']:2., CONFIG['BASE_ASSET_SHORT']:2.})


class TestShadowExitMonitoring(unittest.TestCase):
    def setup_engine(self, side=1):
        self.now=datetime(2026,10,19,10,0,1,tzinfo=NY)
        self.symbol=CONFIG['BASE_ASSET_LONG'] if side else CONFIG['BASE_ASSET_SHORT']
        self.flow=MarketEvents([self.symbol],CONFIG['MAIN_LOOP_TICK_OPEN'])
        self.engine=Engine({n:Candidate(n,side) for n in POLICY['models']}, require_price_monitoring=True)
        self.events=[]
        self.quote(self.now-timedelta(seconds=.5),99.99,100.)
        self.events.extend(self.engine.handle(packet(self.now)))
        self.quote(self.now+timedelta(seconds=1),99.99,100.)
        self.assertEqual(sum(e['event']=='VIRTUAL_BUY' for e in self.events),10)

    def emit(self, events):
        for event in events:
            if event['kind']!='MARKET_REJECTED':
                self.events.extend(self.engine.handle(event))

    def quote(self, at, bid, ask):
        decoded=decode_frame(1,TICK_V1.pack(self.symbol.encode(),(bid+ask)/2,bid,ask,1))
        self.emit(self.flow.frame(decoded,at.timestamp(),at.timestamp()))

    def closed(self, reason):
        sells=[e for e in self.events if e['event']=='VIRTUAL_SELL']
        self.assertEqual(len(sells),10)
        self.assertEqual({e['reason'] for e in sells},{reason})
        self.assertTrue(all(e['cost_reserve']>=.002 for e in sells))
        self.assertTrue(all(not l.positions for l in self.engine.ledgers.values()))
        return sells

    def test_bid_only_stop_loss_fills_next_bid_both_symbols(self):
        for side in (0,1):
            with self.subTest(side=side):
                self.setup_engine(side)
                self.quote(self.now+timedelta(seconds=2),95.,95.02)
                self.assertEqual(sum(e['event']=='VIRTUAL_SELL' for e in self.events),0)
                self.quote(self.now+timedelta(seconds=3),94.9,94.92)
                sells=self.closed('SL')
                self.assertTrue(all(e['exit_price']==94.9 and e['trigger_price_basis']=='WS_BID' for e in sells))

    def test_take_profit(self):
        self.setup_engine();self.quote(self.now+timedelta(seconds=2),111.,111.02)
        self.quote(self.now+timedelta(seconds=3),110.9,110.92)
        self.closed('TP')

    def test_fixed_two_percent_boundary_all_models_and_cohorts(self):
        for side in (0, 1):
            with self.subTest(side=side):
                self.setup_engine(side)
                self.assertTrue(all(p['stop_pct'] == .02 for ledger in self.engine.ledgers.values()
                                    for p in ledger.positions))
                self.quote(self.now+timedelta(seconds=2),98.01,98.03)
                self.assertFalse(any(e['event']=='VIRTUAL_EXIT_INTENT' for e in self.events))
                self.quote(self.now+timedelta(seconds=3),97.99,98.01)
                self.assertEqual(sum(e['event']=='VIRTUAL_EXIT_INTENT' for e in self.events),10)
                self.assertFalse(any(e['event']=='VIRTUAL_SELL' for e in self.events))
                self.quote(self.now+timedelta(seconds=4),97.98,98.)
                self.closed('SL')

    def test_trailing_on_configured_poll_not_each_quote(self):
        self.setup_engine();self.quote(self.now+timedelta(seconds=2),102.,102.02)
        self.quote(self.now+timedelta(seconds=3),101.6,101.62)
        self.assertFalse(any(e['event']=='VIRTUAL_EXIT_INTENT' for e in self.events))
        self.emit(self.flow.timer((self.now+timedelta(seconds=3)).timestamp()))
        self.quote(self.now+timedelta(seconds=4),101.5,101.52)
        self.closed('TRAILING')

    def test_time_without_any_new_market_input_waits_for_next_quote(self):
        self.setup_engine();at=self.now+timedelta(minutes=CONFIG['TIME_STOP_MINUTES'],seconds=2)
        events=self.flow.timer(at.timestamp())
        self.assertEqual([e['kind'] for e in events],['CLOCK'])
        self.emit(events)
        self.assertEqual(sum(e['event']=='VIRTUAL_SELL' for e in self.events),0)
        self.quote(at+timedelta(seconds=1),99.5,99.52)
        self.closed('TIME')

    def test_eod_without_quotes_waits_for_next_quote(self):
        self.setup_engine();hour,minute=map(int,CONFIG['PHASE_EOD_CLEAR'].split(':'))
        at=self.now.replace(hour=hour,minute=minute,second=0)
        self.emit(self.flow.timer(at.timestamp()))
        self.quote(at+timedelta(seconds=1),99.5,99.52)
        self.closed('EOD')

    def test_missing_and_stale_monitor_block_both_cohorts(self):
        self.now=datetime(2026,10,19,10,0,1,tzinfo=NY)
        engine=Engine({n:Candidate(n,1) for n in POLICY['models']},require_price_monitoring=True)
        outputs=engine.handle(packet(self.now))
        self.assertTrue(all(d['primary_action']==d['forced_action']=='PRICE_MONITORING_UNAVAILABLE' for d in outputs))
        self.setup_engine()
        later=self.now+timedelta(minutes=15)
        outputs=self.engine.handle(packet(later))
        self.assertTrue(all(d['primary_action']==d['forced_action']=='PRICE_MONITORING_UNAVAILABLE' for d in outputs))

    def test_packet_with_last_and_quote_does_not_lose_quote_or_fake_tick(self):
        decoded=decode_frame(MARKET_V2,TICK_V2.pack(b'SOXL',105.,99.,100.,1,EXPLICIT_LAST,b'FT'))
        flow=MarketEvents(['SOXL'],3)
        events=flow.frame(decoded,1000.,1000.)
        self.assertEqual([e['kind'] for e in events],['QUOTE','MARK'])
        self.assertEqual(events[1]['price'],99.)
        self.assertEqual(events[1]['price_basis'],'WS_BID')
        self.assertEqual(events[1]['packet_type'],'FT')

    def test_explicit_last_without_quote_is_admitted_and_legacy_price_is_rejected(self):
        flow=MarketEvents(['SOXL'],3)
        for kind,payload,expected in [
            (MARKET_V2,TICK_V2.pack(b'SOXL',101.,0.,0.,1,EXPLICIT_LAST,b'0A'),'MARK'),
            (1,TICK_V1.pack(b'SOXL',101.,0.,0.,1),'MARKET_REJECTED')]:
            events=flow.frame(decode_frame(kind,payload),1000.,1000.)
            self.assertEqual(events[0]['kind'],expected)

    def test_rest_corrupt_crossed_and_stale_never_become_risk_prices(self):
        self.assertIsNone(decode_frame(1,TICK_V1.pack(b'SOXL',101.,100.,102.,0)))
        with self.assertRaises(Exception):decode_frame(MARKET_V2,b'broken')
        with self.assertRaises(ValueError):decode_frame(1,TICK_V1.pack(b'SOXL',101.,102.,100.,1))
        flow=MarketEvents(['SOXL'],3)
        decoded=decode_frame(1,TICK_V1.pack(b'SOXL',101.,100.,102.,1))
        with self.assertRaises(ValueError):flow.frame(decoded,1000.,1004.)


if __name__=='__main__':
    unittest.main()
