"""Delayed virtual feed recovery, with no live ring, broker or network."""
import ast
import argparse
import json
import os
from pathlib import Path
import queue
import sqlite3
import struct
import sys
import threading
import types
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch
from unittest.mock import MagicMock

ROOT=Path(__file__).resolve().parents[1]
HOME=ROOT/'research/shadow_five_full15m_20261018'
sys.path.insert(0,str(HOME));sys.path.insert(0,str(ROOT/'tests'))
from feed_health import FeedHealth
from market_events import MarketEvents,decode_frame,TICK_V2,MARKET_V2,iso
from runtime import Engine,NY
from test_shadow_exit_monitoring import Candidate,packet
import common


class TestFeedRecovery(unittest.TestCase):
    def test_delay_is_skipped_and_freshness_limit_is_not_relaxed(self):
        health=FeedHealth(2.)
        self.assertFalse(health.admit(100.,100.1,102.032,4))
        self.assertTrue(health.gap())
        self.assertTrue(health.admit(103.,103.01,103.02,4))
        self.assertTrue(health.recovered())
        self.assertFalse(health.data_complete)
        self.assertEqual(health.skipped,1)

    def test_transport_and_consumer_queue_are_measured_separately(self):
        h=FeedHealth(2.)
        h.admit(100.,101.6,102.1,2)
        self.assertAlmostEqual(h.max_transport_ms,1600.)
        self.assertAlmostEqual(h.max_queue_ms,500.)
        self.assertEqual(h.skipped_decisions,1)

    def test_backlog_is_one_gap_episode_until_recovery(self):
        h=FeedHealth(2.)
        self.assertTrue(h.gap())
        for _ in range(1000):
            h.admit(100.,100.,104.,4)
            self.assertFalse(h.gap())
        self.assertEqual(h.gaps,1)
        self.assertEqual(h.skipped,1000)
        self.assertTrue(h.recovered())
        self.assertTrue(h.gap())
        self.assertEqual(h.gaps,2)

    def test_future_and_nonfinite_frames_are_not_admitted(self):
        h=FeedHealth(2.)
        self.assertFalse(h.admit(101.,100.,100.,4))
        self.assertFalse(h.admit(float('nan'),100.,100.,4))

    def engine(self):
        return Engine({'FIXTURE':Candidate('FIXTURE',1)},evaluation_window=lambda d:True,
                      require_price_monitoring=True)

    def test_gap_preserves_positions_and_sell_intents_cancels_buys(self):
        engine=self.engine();ledger=engine.ledgers[('FIXTURE','primary')]
        position={'symbol':'SOXL'};sell=dict(action='SELL',position=position)
        ledger.positions.append(position)
        ledger.pending=[sell,dict(action='BUY',decision_id='unfilled')]
        engine.price_observed['SOXL']=datetime(2026,10,21,10,tzinfo=NY)
        rows=engine.feed_gap(datetime(2026,10,21,10,0,3,tzinfo=NY))
        self.assertEqual(ledger.positions,[position]);self.assertEqual(ledger.pending,[sell])
        self.assertEqual(engine.price_observed,{})
        self.assertTrue(any(r['event']=='BUY_CANCELLED_FEED_GAP' for r in rows))

    def test_old_cycle_is_not_reused_and_next_cycle_can_resume(self):
        engine=self.engine();at=datetime(2026,10,21,10,0,3,tzinfo=NY)
        engine.feed_gap(at)
        rows=engine.handle(packet(at))
        self.assertEqual(rows[0]['event'],'DECISION_REJECTED_FEED_GAP')
        later=at+timedelta(minutes=15)
        engine.price_observed['SOXL']=later
        rows=engine.handle(packet(later))
        self.assertTrue(any(r['event']=='DECISION' for r in rows))
        self.assertEqual(rows[0]['feed_gap_count'],1)

    def test_reader_resync_does_not_touch_publisher_or_block(self):
        tree=ast.parse((HOME/'feed_bridge.py').read_text(encoding='utf-8'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Reader')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='resync')
        ns={'struct':struct}
        exec(compile(ast.Module(body=[method],type_ignores=[]),'consumer_resync','exec'),ns)
        memory=bytearray(64);struct.pack_into('<Q',memory,0,25);struct.pack_into('<Q',memory,16,3)
        reader=types.SimpleNamespace(mem=memory,cursor=5,drops=1,beat=lambda:None)
        self.assertEqual(ns['resync'](reader),20)
        self.assertEqual(reader.cursor,25);self.assertEqual(reader.drops,3)
        self.assertEqual(struct.unpack_from('<Q',memory,0)[0],25)

    def test_receive_loop_recovers_ring_gap_and_queue_overflow(self):
        tree=ast.parse((HOME/'live_feed.py').read_text(encoding='utf-8'))
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        receive=next(n for n in main.body if isinstance(n,ast.FunctionDef) and n.name=='receive')
        stop=threading.Event();events=queue.Queue(1);events.put(('old',))
        source=[RuntimeError('FEED_GAP_OR_PRODUCER_DROP'),(1,100.,4,b'a'),(2,100.,4,b'b')]
        def read():
            if not source:stop.set();return None
            item=source.pop(0)
            if isinstance(item,Exception):raise item
            return item
        reader=types.SimpleNamespace(read=read,resync=MagicMock(return_value=2))
        stats=dict(receiver_losses=0,ring_recoveries=0,received=0,max_queue_depth=0)
        failures=[]
        ns=dict(reader=reader,stats=stats,stop=stop,events=events,queue=queue,failures=failures,
                time=types.SimpleNamespace(time=lambda:100.))
        exec(compile(ast.Module(body=[receive],type_ignores=[]),'receive_recovery','exec'),ns)
        ns['receive']()
        self.assertEqual(failures,[]);self.assertEqual(stats['ring_recoveries'],1)
        self.assertEqual(stats['receiver_losses'],4)
        self.assertEqual(events.get_nowait()[:4],(2,100.,4,b'b'))

    def test_three_portfolios_keep_holdings_and_exit_on_fresh_quotes(self):
        import test_three_comparison_shadows as comparison_fixture
        case=comparison_fixture.TestThreeComparisonShadows();case.setUp();case.begin()
        self.assertEqual(sum(len(l.positions) for l in case.engine.ledgers.values()),3)
        case.rows.extend(case.engine.feed_gap(case.now+timedelta(seconds=2)))
        self.assertEqual(sum(len(l.positions) for l in case.engine.ledgers.values()),3)
        case.quote('SOXS',100.,3.)
        case.quote('SOXL',97.,3.1)
        case.quote('SOXL',96.9,4.)
        exits=[r for r in case.rows if r['event']=='VIRTUAL_SELL']
        self.assertEqual(len(exits),3)
        self.assertEqual({r['feed_gap_count'] for r in exits},{1})
        self.assertTrue(all(r['cost_reserve']>=.002 for r in exits))

    def test_controller_continues_from_late_crossed_and_capture_failure(self):
        """Run the real main controller with an in-memory feed and audit sink."""
        base=datetime(2026,10,21,10,0,3,tzinfo=NY).timestamp()
        clock=[base];saved={};records=[];handled=[]
        def frame(bid,ask):return TICK_V2.pack(b'SOXL',0.,bid,ask,1,0,b'QUOTE')
        envelopes=[
            (1,base-2.032,MARKET_V2,frame(100.,101.),base-.01),
            (2,base,MARKET_V2,frame(102.,101.),base),
            (3,base,3,b'FEATURE_CAPTURE_FAILED',base),
            (4,base+.1,MARKET_V2,frame(100.,101.),base+.1),
            (5,base+900.,2,b'fixture_features',base+900.),
            (6,base+900.1,MARKET_V2,frame(100.,101.),base+900.1)]
        class FakeQueue:
            def __init__(self,*a):self.rows=list(envelopes)
            def get(self,**k):
                if not self.rows:raise KeyboardInterrupt()
                item=self.rows.pop(0);clock[0]=max(clock[0],item[1]+.01);return item
        class NoThread:
            def __init__(self,**k):pass
            def start(self):pass
            def join(self,**k):pass
        class Reader:
            def close(self):pass
        class Builder:
            db=types.SimpleNamespace(close=lambda:None)
            def build(self,stamp,payload):return dict(kind='DECISION',time=iso(stamp))
        class TestEngine:
            candidates={};ledgers={}
            def __init__(self,*a,**k):pass
            def handle(self,event):handled.append(event);return []
            def feed_gap(self,now):return []
            def price_monitoring(self,now):return {}
        class Audit:
            max_write_ms=0.
            def __init__(self,*a):pass
            def append(self,event):records.append(event)
            def market(self,event):pass
            def packet(self,event):pass
            def flush(self):pass
            def close(self):pass
        fake_candidates=types.ModuleType('comparison_shadows')
        fake_candidates.build_comparison_shadows=lambda:{}
        ns=dict(argparse=argparse,os=os,datetime=datetime,NY=NY,HOME=HOME,CONFIG=common.CONFIG,
                json=json,load_control=lambda:dict(operation='ISOLATED_QA'),should_stop=lambda *a:False,
                date_allowed=lambda *a:True,save=lambda name,value:saved.update({name:value}),
                Engine=TestEngine,Audit=Audit,PacketBuilder=Builder,Reader=Reader,
                FeedHealth=FeedHealth,MAX_EVENT_AGE=2.,MAX_QUEUE=8192,HEALTH_PERIOD_SECONDS=5,
                MarketEvents=MarketEvents,decode_frame=decode_frame,MARKET_V2=MARKET_V2,
                load_execution_config=common.load_execution_config,sha=lambda p:'FIXTURE',iso=iso,
                queue=types.SimpleNamespace(Queue=FakeQueue,Empty=queue.Empty),sqlite3=sqlite3,
                threading=types.SimpleNamespace(Thread=NoThread,Event=threading.Event),
                time=types.SimpleNamespace(time=lambda:clock[0],monotonic=lambda:0.))
        node=next(n for n in ast.parse((HOME/'live_feed.py').read_text(encoding='utf-8')).body
                  if isinstance(n,ast.FunctionDef) and n.name=='main')
        exec(compile(ast.Module(body=[node],type_ignores=[]),'real_shadow_controller','exec'),ns)
        with patch.object(sys,'argv',['isolated_qa']),patch.dict(sys.modules,comparison_shadows=fake_candidates):
            self.assertEqual(ns['main'](),0)
        self.assertTrue(any(e['kind']=='DECISION' for e in handled))
        self.assertTrue(any(e['event']=='FEED_GAP' for e in records))
        self.assertTrue(any(e['event']=='FEED_RECOVERED' for e in records))
        self.assertFalse(any(e['event']=='RUN_QUARANTINED' for e in records))
        status=saved['qa/feed_latest.json']
        self.assertTrue(status['valid']);self.assertFalse(status['feed_health']['data_complete'])
        self.assertEqual(status['stats']['decisions'],1)


if __name__=='__main__':unittest.main()
