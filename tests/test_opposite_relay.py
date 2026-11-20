"""Virtual ledger QA: production relay calls the legacy execution helpers."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
from core.live_runner import KiwoomLiveRunner, create_live_runner
from core.pending_order import binding_for, journal_path
from core.relay_live_runner import RelayLiveRunner, _Broker, _Quotes
from core.relay_policy import RelayBook, atomic_json, decision_key, exit_reason, trailing_trigger, ws_quote


class Broker:
    is_simulation = True
    account_no = 'SYNTHETIC'; account_type = '01'
    broker_name = 'FIXTURE'; mode_str = 'SIMULATION'
    def __init__(self):
        self.cash = 10000.; self.holdings = {}; self.orders = {}; self.sent = []
        self.fail = False; self.unknown = False
    def test_connection(self):
        return dict(ok=True, deposit_ok=True, stock_ok=True, usd_order_available=self.cash)
    def get_overseas_stock_balance(self, **kwargs):
        return dict(ok=not self.fail, holdings=[dict(symbol=s, quantity=q, purchase_price=p)
                                              for s,(q,p) in self.holdings.items()])
    def get_open_orders(self, *args):
        return dict(ok=not self.fail, orders=[])
    def send_order(self, **order):
        self.sent.append(order)
        if self.unknown:
            raise RuntimeError('UNCERTAIN_TRANSPORT')
        symbol=order['symbol'];qty=order['quantity'];price=order['price']
        number=str(len(self.sent));self.orders[number]=dict(order)
        if order['order_type']=='BUY':
            assert symbol not in self.holdings
            self.holdings[symbol]=(qty,price);self.cash-=qty*price
        else:
            assert qty==self.holdings[symbol][0]
            self.cash+=qty*price;del self.holdings[symbol]
        return dict(ok=True,order_no=number)
    def resolve_order_ledger_status(self, order_no, symbol, side, **kwargs):
        order=self.orders[order_no]
        assert order['symbol']==symbol and order['order_type']==side
        return dict(status='FILLED',filled_qty=order['quantity'])


class TestRelay(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'data')
        self.addCleanup(self.tmp.cleanup)
        self.now=datetime(2026,10,20,10,0,tzinfo=ZoneInfo('America/New_York'))
        p=patch('core.relay_live_runner.USMarketCalendar.get_market_status',
                return_value=dict(is_open=True,session_name='REGULAR_MARKET_OPEN'))
        p.start();self.addCleanup(p.stop)
        alloc_patch=patch.multiple(config, RELAY_INITIAL_ALLOCATION=.49, RELAY_CHAIN_ALLOCATION=.49)
        alloc_patch.start();self.addCleanup(alloc_patch.stop)
        r=RelayLiveRunner.__new__(RelayLiveRunner);self.r=r
        r._scope=threading.local();r._relay_lock=threading.RLock();r._quote_lock=threading.Lock();r._extremes={};r._risk_hits={}
        r._entry_mutex=threading.Lock();r._risk_mutex=threading.Lock();r._order_lock=threading.Lock()
        r._wake=threading.Event();r._stop=threading.Event();r._quotes={};r._last_poll={}
        r._positions={};r._journal_failed=False;r._legacy_present=False;r._mismatch=None
        r._deployment_ok=True
        r._is_order_in_progress=False;r._order_fills={};r._empty_balance_streak=0
        r._unreconciled_holding_mismatch=None;r._pending_buy_write_failed=False
        r.is_running=False;r._now=lambda:self.now
        r._is_circuit_breaker_halted=lambda:(False,'')
        raw=Broker();r._raw_broker=raw;r._binding=binding_for(raw);r._root=Path(self.tmp.name)
        r.book=RelayBook(r._root/'book.json',r._binding)
        r.broker=_Broker(r,raw);self.raw=raw
        original=MagicMock();original.get_latest_price.return_value=100.
        r.ws_streamer=_Quotes(r,original)
        r.shadow_tracker=MagicMock();r.notifier=MagicMock();r.experience_logger=MagicMock();r.dispatcher=MagicMock()
        r.moe_orchestrator=MagicMock();r.moe_orchestrator.gbdt_engine.get_active_model_info.return_value={}
        r.order_timeout_buy_sec=config.BUY_TIMEOUT_SEC;r.order_timeout_sell_sec=config.SELL_TIMEOUT_SEC
        def wait(order_no,symbol,is_buy,target_qty,**kwargs):
            r._last_wait_fill_stk_bal=raw.get_overseas_stock_balance()
            held=raw.holdings.get(symbol,(0,0))[0]
            return (held>=target_qty,held,max(0,target_qty-held)) if is_buy else (held==0,target_qty-held,held)
        r._wait_for_fill=wait
        r._dispatch_sell=lambda **kwargs:r._execute_sell_with_10s_chase(**kwargs)
        self.quote(config.BASE_ASSET_LONG,100.,100.)
        self.quote(config.BASE_ASSET_SHORT,100.,100.)
        r._decision=decision_key(self.now)

    def quote(self,symbol,bid,ask=None):
        self.r._on_websocket_tick(symbol,bid,dict(from_ws=True,extra=dict(values={'51':bid,'41':ask or bid})))

    def buy(self,symbol,role='INITIAL',trigger=None):
        r=self.r
        if role=='INITIAL':
            qty=r._entry_quantity(self.raw.cash,self.r._quote(symbol)['ask'])
        else:
            qty=r.book.quantity(self.raw.cash,self.r._quote(symbol)['ask'],role)
        px=self.r._quote(symbol)['ask']
        return r._execute_buy_with_10s_chase(symbol,qty,px,
            dict(gating_confidence=.7,gbdt_probs={},all_gating_confidences={}),
            dict(tp_pct=config.MAX_TP_PCT*100,dynamic_tp_px=px*(1+config.MAX_TP_PCT),
                 dynamic_sl_px=px*(1-config.SL_MIN_PCT),time_stop_minutes=config.TIME_STOP_MINUTES),
            role=role,trigger_symbol=trigger)

    def manage(self):
        self.r._manage_open_positions(self.raw.get_overseas_stock_balance())

    def test_initial_49_and_chain_49_real_helpers_and_roles(self):
        long,short=config.BASE_ASSET_LONG,config.BASE_ASSET_SHORT
        self.assertTrue(self.buy(long))
        first=self.raw.sent[0];self.assertLessEqual(first['quantity']*first['price'],4900.)
        self.assertEqual(self.r._positions[long]['relay_role'],'INITIAL')
        self.assertFalse(self.r.book.data['operations'])
        entry=self.r._positions[long]['price']
        self.quote(long,entry*(1-config.RELAY_OPPOSITE_TRIGGER_PCT+.0001));self.manage();self.assertEqual(len(self.raw.sent),1)
        self.quote(long,entry*(1-config.RELAY_OPPOSITE_TRIGGER_PCT));self.manage()
        self.assertEqual(set(self.r._positions),set((long,short)))
        self.assertEqual(self.r._positions[short]['relay_role'],'CHAIN')
        self.assertLessEqual(self.raw.sent[-1]['quantity']*self.raw.sent[-1]['price'],4900.)
        self.assertGreaterEqual(self.raw.cash,200.)
        self.manage();self.assertEqual(len(self.raw.sent),2)

    def test_empty_ledger_between_quantity_and_submit_cannot_erase_reservation(self):
        long=config.BASE_ASSET_LONG
        qty=self.r._entry_quantity(self.raw.cash,100.)
        budget=self.r.book.data['budget']
        errors=[]
        def reconcile():
            try:self.r._reconcile_active_position_with_ledger(self.raw.get_overseas_stock_balance())
            except Exception as e:errors.append(e)
        worker=threading.Thread(target=reconcile);worker.start();worker.join(timeout=5)
        self.assertFalse(worker.is_alive());self.assertFalse(errors)
        self.assertEqual(self.r.book.data['budget'],budget)
        self.assertTrue(self.r.book.initial_preparing(self.r._decision))
        self.assertTrue(self.r._execute_buy_with_10s_chase(long,qty,100.,dict(gating_confidence=.7),
            dict(tp_pct=10.,sl_pct=2.,dynamic_tp_px=110.,dynamic_sl_px=98.,time_stop_minutes=90)))
        self.assertEqual(len(self.raw.sent),1)
        self.assertIn(long,self.r._positions)
        self.assertFalse(self.r.book.initial_preparing(self.r._decision))

    def test_pre_submit_failure_releases_preparation_without_faking_position(self):
        long=config.BASE_ASSET_LONG
        qty=self.r._entry_quantity(self.raw.cash,100.)
        self.r._quotes.clear()
        self.assertFalse(self.r._execute_buy_with_10s_chase(long,qty,100.,{},{}))
        self.assertFalse(self.r.book.initial_preparing(self.r._decision))
        self.r._reconcile_active_position_with_ledger(self.raw.get_overseas_stock_balance())
        self.assertIsNone(self.r.book.data['budget'])
        self.assertFalse(self.raw.sent)

    def test_restart_abandons_only_unsubmitted_preparation(self):
        self.assertGreater(self.r._entry_quantity(self.raw.cash,100.),0)
        self.r.book=RelayBook(self.r.book.path,self.r._binding)
        self.assertFalse(self.r.book.initial_preparing(self.r._decision))
        self.r._reconcile_active_position_with_ledger(self.raw.get_overseas_stock_balance())
        self.assertIsNone(self.r.book.data['budget'])
        self.assertEqual(self.r._entry_quantity(self.raw.cash,100.),0)

    def test_preparation_cannot_be_replaced_by_second_begin(self):
        self.assertGreater(self.r._entry_quantity(self.raw.cash,100.),0)
        cycle=self.r.book.data['cycle_id']
        self.assertFalse(self.r.book.begin(5000.,decision_key(self.now+timedelta(minutes=15))))
        self.assertEqual(self.r.book.data['cycle_id'],cycle)

    def test_expired_preparation_does_not_block_future_cycles(self):
        self.assertGreater(self.r._entry_quantity(self.raw.cash,100.),0)
        self.now+=timedelta(minutes=config.RELAY_DECISION_INTERVAL_MINUTES)
        self.r._reconcile_active_position_with_ledger(self.raw.get_overseas_stock_balance())
        self.assertIsNone(self.r.book.data['budget'])

    def test_sell_one_keeps_other_and_blocks_every_additional_buy(self):
        long,short=config.BASE_ASSET_LONG,config.BASE_ASSET_SHORT
        self.assertTrue(self.buy(long));self.quote(long,98.5);self.manage()
        self.quote(long,97.);self.manage()
        self.assertNotIn(long,self.r._positions);self.assertIn(short,self.r._positions)
        # 남은 short가 -1% 빠지면 2번째 체인(long) 정상 발동
        self.quote(short,98.5);self.quote(long,100.);self.manage()
        self.assertEqual(set(self.r._positions),{short,long})

    def test_chain_exits_first_initial_cannot_trigger_second_chain(self):
        long,short=config.BASE_ASSET_LONG,config.BASE_ASSET_SHORT
        self.assertTrue(self.buy(long));self.quote(long,98.5);self.manage()
        self.quote(long,100.);self.quote(short,97.);self.manage()
        self.assertEqual(set(self.r._positions),{long})
        count=len(self.raw.sent)
        # 남은 long이 -1% 빠지면 2번째 체인(short) 정상 발동
        self.quote(long,98.5);self.quote(short,100.);self.manage()
        self.assertEqual(set(self.r._positions),{long,short})
        self.assertGreater(len(self.raw.sent),count)

    def test_chain_reservation_persists_after_sell_and_book_reload(self):
        long,short=config.BASE_ASSET_LONG,config.BASE_ASSET_SHORT
        self.assertTrue(self.buy(long));self.quote(long,98.5);self.manage()
        claim=self.r.book.data['chain_operation_id']
        self.assertTrue(claim)
        self.quote(long,97.);self.manage()
        self.r.book=RelayBook(self.r.book.path,self.r._binding)
        self.assertEqual(self.r.book.data['chain_operation_id'],claim)
        # 최대 한도 미만일 때는 chain_available True
        self.assertTrue(self.r._chain_available())
        # 최대 한도(config.RELAY_MAX_CHAIN_COUNT) 소진 시 chain_available False
        for i in range(len(self.r.book.data.get('chain_operations', [])), config.RELAY_MAX_CHAIN_COUNT):
            self.r.book.operation(long, dict(id=f'C_{i}', action='BUY', role='CHAIN'))
            self.r.book.operation(long, None)
        self.assertFalse(self.r._chain_available())

    def test_new_cycle_resets_chain_only_after_all_positions_close_and_next_bar(self):
        long,short=config.BASE_ASSET_LONG,config.BASE_ASSET_SHORT
        self.assertTrue(self.buy(long));self.quote(long,98.5);self.manage()
        self.quote(long,97.);self.manage()
        self.quote(short,97.);self.manage()
        self.assertFalse(self.r._positions)
        self.assertEqual(self.r._entry_quantity(self.raw.cash,100.),0)
        self.now+=timedelta(minutes=config.RELAY_DECISION_INTERVAL_MINUTES)
        self.quote(long,100.);self.quote(short,100.)
        self.r._decision=decision_key(self.now)
        self.assertTrue(self.buy(long))
        self.assertIsNone(self.r.book.data['chain_operation_id'])
        self.quote(long,98.5);self.manage()
        self.assertEqual(set(self.r._positions),{long,short})

    def test_old_active_cycle_without_reservation_blocks_chain_but_sells(self):
        long=config.BASE_ASSET_LONG
        self.assertTrue(self.buy(long))
        old=dict(self.r.book.data);old.pop('chain_operation_id')
        atomic_json(self.r.book.path,old)
        self.r.book=RelayBook(self.r.book.path,self.r._binding)
        self.quote(long,98.5);self.manage()
        self.assertEqual(len(self.raw.sent),1)
        self.quote(long,97.);self.manage()
        self.assertFalse(self.r._positions)

    def test_second_chain_operation_rejected_by_durable_book(self):
        long,short=config.BASE_ASSET_LONG,config.BASE_ASSET_SHORT
        self.assertTrue(self.buy(long));self.quote(long,98.5);self.manage()
        claim=self.r.book.data['chain_operation_id']
        # 최대 한도까지 등록 허용
        for i in range(len(self.r.book.data.get('chain_operations', [])), config.RELAY_MAX_CHAIN_COUNT):
            self.r.book.operation(short, dict(id=f'CHAIN_{i}', action='BUY', role='CHAIN'))
            self.r.book.operation(short, None)
        # 한도 초과 시 거절
        with self.assertRaises(RuntimeError):
            self.r.book.operation(long, dict(id='EXCEEDED', action='BUY', role='CHAIN'))
        self.assertEqual(len(self.r.book.data['chain_operations']),config.RELAY_MAX_CHAIN_COUNT)
        self.assertFalse(self.r.book.data['operations'])

    def test_symmetric_short_first(self):
        long,short=config.BASE_ASSET_LONG,config.BASE_ASSET_SHORT
        self.assertTrue(self.buy(short));self.quote(short,98.5);self.manage()
        self.assertEqual(self.r._positions[long]['relay_role'],'CHAIN')

    def test_one_percent_trigger_symmetric_boundary_and_trailing_unchanged(self):
        self.assertEqual(config.RELAY_OPPOSITE_TRIGGER_PCT,.015)
        self.assertEqual(config.TRAILING_DROP_PCT,.003)
        self.assertEqual(config.TRAILING_TRIGGER_PCT,.01)
        self.assertEqual(config.RELAY_CHAIN_TRAILING_TRIGGER_PCT,.01)
        short=config.BASE_ASSET_SHORT;long=config.BASE_ASSET_LONG
        self.assertTrue(self.buy(short))
        entry=self.r._positions[short]['price']
        self.quote(short,entry*.9851);self.manage()
        self.assertNotIn(long,self.r._positions)
        self.quote(short,entry*.985);self.manage()
        self.assertIn(long,self.r._positions)
        self.assertEqual(self.r._positions[long]['relay_role'],'CHAIN')

    def test_no_duplicate_target_or_pending_buy(self):
        long=config.BASE_ASSET_LONG
        self.assertTrue(self.buy(long))
        count=len(self.raw.sent);self.assertFalse(self.buy(long));self.assertEqual(count,len(self.raw.sent))
        with self.r._symbol(config.BASE_ASSET_SHORT):
            self.r._store_pending_buy_order(dict(symbol=config.BASE_ASSET_SHORT,target_qty=1))
        self.quote(long,98.5);self.manage();self.assertEqual(count,len(self.raw.sent))

    def test_rest_or_stale_quote_cannot_trigger_chain(self):
        long,short=config.BASE_ASSET_LONG,config.BASE_ASSET_SHORT
        self.assertTrue(self.buy(long))
        self.r._on_websocket_tick(long,97.,dict(from_ws=False,extra=dict(bid=97,ask=97)))
        self.assertEqual(self.r._quote(long)['bid'],100.)
        self.quote(long,98.5);self.r._quotes[short]['time']-=config.RELAY_QUOTE_MAX_AGE_SECONDS+1
        self.manage();self.assertEqual(len(self.raw.sent),1)

    def test_first_role_persists_after_other_leg_closes(self):
        long=config.BASE_ASSET_LONG;self.assertTrue(self.buy(long))
        p=copy.deepcopy(self.r._positions[long]);p['high_price_during_hold']=p['price']*1.012
        with patch.object(config,'TRAILING_TRIGGER_PCT',.015):
            self.assertIsNone(exit_reason(p,p['price']*1.006,self.now,True))
            p['relay_role']='CHAIN'
            self.assertEqual(exit_reason(p,p['price']*1.006,self.now,True),'TRAIL')

    def test_initial_trailing_then_flat_waits_for_new_bar(self):
        long=config.BASE_ASSET_LONG;self.assertTrue(self.buy(long))
        self.quote(long,102.);self.manage()
        self.r._last_poll[long]=0.;self.quote(long,101.5);self.manage()
        self.assertFalse(self.r._positions)
        self.assertEqual(self.r._entry_quantity(self.raw.cash,100.),0)
        self.now+=timedelta(minutes=config.RELAY_DECISION_INTERVAL_MINUTES)
        self.r._decision=decision_key(self.now)
        self.assertGreater(self.r._entry_quantity(self.raw.cash,100.),0)

    def test_unknown_submission_survives_restart_and_blocks_new_buy(self):
        self.raw.unknown=True
        self.assertFalse(self.buy(config.BASE_ASSET_LONG))
        recovered=RelayBook(self.r.book.path,self.r._binding)
        self.assertTrue(recovered.data['operations'][config.BASE_ASSET_LONG]['submission_pending'])
        self.assertFalse(self.r._verify_pre_buy_ledger_clean(config.BASE_ASSET_SHORT))

    def test_failed_ledger_blocks_orders(self):
        self.raw.fail=True
        self.assertFalse(self.buy(config.BASE_ASSET_LONG))
        self.assertFalse(self.raw.sent)

    def test_state_failure_blocks_new_orders(self):
        with patch('core.relay_policy.os.replace',side_effect=OSError('QA_FAILURE')):
            with self.assertRaises(OSError):self.r.book.begin(10000.,decision_key(self.now))
        self.assertTrue(self.r.book.failed)
        self.assertFalse(self.r._verify_pre_buy_ledger_clean(config.BASE_ASSET_LONG))

    def test_account_binding_and_corruption_fail_closed(self):
        self.r.book.begin(10000.,decision_key(self.now))
        with self.assertRaises(RuntimeError):RelayBook(self.r.book.path,'OTHER_ACCOUNT')
        self.r.book.path.write_text('{bad',encoding='utf-8')
        with self.assertRaises(ValueError):RelayBook(self.r.book.path,self.r._binding)

    def test_no_entry_after_cutoff_or_before_first_closed_bar(self):
        self.now=self.now.replace(hour=15,minute=30)
        self.assertEqual(self.r._entry_quantity(10000.,100.),0)
        self.now=self.now.replace(hour=9,minute=40)
        self.assertEqual(self.r._entry_quantity(10000.,100.),0)

    def test_time_and_eod_need_no_risk_quote(self):
        position=dict(price=100.,relay_role='CHAIN',relay_entry_ny=self.now.isoformat())
        self.assertEqual(exit_reason(position,None,self.now+timedelta(minutes=90)),'TIME')
        self.assertEqual(exit_reason(position,None,self.now.replace(hour=15,minute=55)),'EOD')

    def test_quote_parser_rejects_midpoint_and_nonfinite(self):
        self.assertIsNone(ws_quote(dict(from_ws=True,extra={'price':100})))
        self.assertIsNone(ws_quote(dict(from_ws=False,extra={'bid':99,'ask':100})))
        self.assertIsNone(ws_quote(dict(from_ws=True,extra={'bid':float('nan'),'ask':100})))

    def test_budget_caps_after_losses_and_no_double_98_percent(self):
        self.r.book.begin(10000.,decision_key(self.now))
        self.assertEqual(self.r.book.quantity(10000.,99.95,'INITIAL'),49)
        self.assertEqual(self.r.book.quantity(1000.,99.95,'CHAIN'),8)
        self.assertEqual(self.r.book.quantity(100.,99.95,'CHAIN'),0)

    def test_deployment_factory_selects_relay(self):
        with patch.object(RelayLiveRunner,'__init__',return_value=None):
            self.assertIsInstance(create_live_runner(True),RelayLiveRunner)

    def test_halt_and_shutdown_block_chain_but_not_exit_policy(self):
        self.r._is_circuit_breaker_halted=lambda:(True,'HALT')
        self.assertFalse(self.r._can_enter_now())
        self.r._is_circuit_breaker_halted=lambda:(False,'')
        self.r._stop.set();self.assertFalse(self.r._can_enter_now())

    def test_unverified_deployment_cannot_start_order_threads(self):
        self.r._deployment_ok=False
        self.assertFalse(self.r._can_enter_now())
        with self.assertRaises(RuntimeError):self.r.run_once_and_start_listener(True)

    def test_other_account_ready_proof_cannot_authorize_real_entry(self):
        self.raw.is_simulation=False
        with patch('core.circuit_breaker.read_order_state',return_value=dict(
                reconciliation_proof=dict(verified_account='OTHER-01',verified_environment='REAL'))), \
             patch('core.circuit_breaker.order_readiness',return_value={'ready':True}):
            self.assertFalse(self.r._account_ready())
            self.assertFalse(self.r._can_enter_now())

    def test_zero_quantity_does_not_create_order_or_position(self):
        self.raw.cash=1.
        self.assertFalse(self.buy(config.BASE_ASSET_LONG))
        self.assertFalse(self.raw.sent);self.assertFalse(self.r._positions)

    def test_unknown_manual_holding_blocks_new_buy(self):
        self.raw.holdings['UNMANAGED']=(1,100.)
        self.assertFalse(self.buy(config.BASE_ASSET_LONG))
        self.assertFalse(self.raw.sent)

    def test_empty_balance_alone_cannot_erase_position(self):
        long=config.BASE_ASSET_LONG;self.assertTrue(self.buy(long))
        self.raw.holdings.clear()
        self.r._reconcile_active_position_with_ledger(self.raw.get_overseas_stock_balance())
        self.assertIn(long,self.r._positions)
        self.assertIsNotNone(self.r._mismatch)

    def test_recover_partial_canceled_buy_only_from_terminal_evidence(self):
        long=config.BASE_ASSET_LONG
        self.r.book.begin(10000.,decision_key(self.now))
        operation=dict(id='RECOVERY',action='BUY',role='CHAIN',quantity=10,held_before=0,
            created_ny=self.now.isoformat(),orders=[dict(order_no='RECOVERY_ORDER',quantity=10)],
            submission_pending=False)
        self.r.book.operation(long,operation);self.raw.holdings[long]=(4,100.)
        self.raw.resolve_order_ledger_status=lambda *a,**k:dict(status='CANCELED_PARTIAL_FILL',filled_qty=4)
        self.r._reconcile_active_position_with_ledger(self.raw.get_overseas_stock_balance())
        self.assertEqual(self.r._positions[long]['quantity'],4)
        self.assertEqual(self.r._positions[long]['relay_role'],'CHAIN')
        self.assertFalse(self.r.book.data['operations'])

    def test_unknown_order_evidence_preserves_pending_state(self):
        long=config.BASE_ASSET_LONG;self.r.book.begin(10000.,decision_key(self.now))
        operation=dict(id='RECOVERY',action='BUY',role='INITIAL',quantity=10,held_before=0,
            created_ny=self.now.isoformat(),orders=[dict(order_no='UNKNOWN',quantity=10)],submission_pending=False)
        self.r.book.operation(long,operation)
        self.raw.resolve_order_ledger_status=lambda *a,**k:dict(status='UNKNOWN',filled_qty=0)
        self.r._reconcile_active_position_with_ledger(self.raw.get_overseas_stock_balance())
        self.assertIn(long,self.r.book.data['operations']);self.assertFalse(self.raw.sent)

    def test_roles_survive_disk_reload(self):
        long=config.BASE_ASSET_LONG;self.assertTrue(self.buy(long))
        self.r._positions.clear()
        with self.r._symbol(long):
            p=self.r._get_active_position()
        self.assertEqual(p['relay_role'],'INITIAL')
        self.assertEqual(p['trailing_trigger_pct'],config.TRAILING_TRIGGER_PCT)

    def test_two_legs_time_exit_without_new_quote(self):
        long,short=config.BASE_ASSET_LONG,config.BASE_ASSET_SHORT
        self.assertTrue(self.buy(long));self.quote(long,98.5);self.manage()
        self.now+=timedelta(minutes=config.TIME_STOP_MINUTES)
        self.r._quotes.clear();self.manage()
        self.assertFalse(self.r._positions);self.assertFalse(self.raw.holdings)

    def evaluate(self, approved=True, direction='LONG', allow=True):
        import pandas as pd
        r=self.r;r.data_lake=MagicMock()
        r.data_lake.load_candles.return_value=pd.DataFrame({'Close':[100.]*100},
            index=pd.date_range(end=self.now.replace(tzinfo=None)-timedelta(minutes=config.RELAY_DECISION_INTERVAL_MINUTES),
                                periods=100, freq='15min'))
        r._collect_cycle_market_snapshot=MagicMock(return_value=dict(
            live_prices={config.BASE_ASSET_LONG:100.,config.BASE_ASSET_SHORT:100.},
            quote_time_et='2026-10-20 10:00 ET',candle_time_et='2026-10-20 09:45 ET',
            data_stale=False,stale_reason=''))
        r._emit_model_state_alerts_from_eval=MagicMock();r._get_5m_atr_pct=lambda *a:1.
        r.moe_orchestrator.evaluate_dual_filter_signal.return_value=dict(
            is_approved=approved,direction=direction,dir_gbdt=direction,gating_confidence=.75,
            gbdt_probs={'LONG':.7,'SHORT':.1,'NONE':.2},all_gating_confidences={},
            features={},is_cross_veto=False,is_60m_trend_ok=True,dip_ok=True)
        return r.evaluate_15m_cycle_once(allow_order_execution=allow,sync_intraday=False,
            now_dt=self.now,market_session='REGULAR_MARKET_OPEN')

    def test_production_15m_evaluator_executes_49_percent_primary(self):
        result=self.evaluate()
        self.assertTrue(result['order_executed'])
        self.assertEqual(result['order_qty'],48)
        self.assertEqual(self.r._positions[config.BASE_ASSET_LONG]['relay_role'],'INITIAL')
        self.assertEqual(len(self.raw.sent),1)

    def test_production_15m_evaluator_signal_only_and_model_rejection(self):
        self.assertFalse(self.evaluate(allow=False)['order_executed'])
        self.assertFalse(self.evaluate(approved=False,direction='NONE')['order_executed'])
        self.assertFalse(self.raw.sent)

    def test_production_15m_does_not_add_to_an_existing_cycle(self):
        self.assertTrue(self.buy(config.BASE_ASSET_LONG))
        self.assertFalse(self.evaluate()['order_executed'])
        self.assertEqual(len(self.raw.sent),1)

    def test_unconfirmed_execution_is_not_briefed_as_order_allowed(self):
        with patch.object(self.r,'_execute_buy_with_10s_chase',return_value=False):
            result=self.evaluate()
        self.assertFalse(result['order_executed'])
        self.assertFalse(result['order_allowed'])
        self.assertEqual(result['order_block_detail'],'BUY_EXECUTION_NOT_CONFIRMED')
        self.assertFalse(self.r.book.initial_preparing(self.r._decision))

    def test_intra_worker_peak_is_not_lost_before_pullback(self):
        long=config.BASE_ASSET_LONG;self.assertTrue(self.buy(long))
        self.quote(long,102.)
        self.quote(long,101.5)
        self.manage()
        self.assertNotIn(long,self.r._positions)

    def test_transient_stop_touch_is_not_erased_by_recovery(self):
        long=config.BASE_ASSET_LONG;self.assertTrue(self.buy(long))
        self.quote(long,97.);self.quote(long,100.);self.manage()
        self.assertNotIn(long,self.r._positions)
        self.assertEqual(self.raw.sent[-1]['order_type'],'SELL')

    def test_external_change_preserves_entry_and_blocks_chain(self):
        long=config.BASE_ASSET_LONG;self.assertTrue(self.buy(long))
        prior=copy.deepcopy(self.r._positions[long])
        self.raw.holdings[long]=(prior['quantity']+3,prior['price']+1.)
        self.r._reconcile_active_position_with_ledger(self.raw.get_overseas_stock_balance())
        self.assertEqual(self.r._positions[long],prior)
        self.assertEqual(self.r._mismatch,'EXTERNALLY_CHANGED_TRACKED_HOLDING')
        self.quote(long,98.5);self.manage()
        self.assertEqual(len(self.raw.sent),1)
        self.now=self.now.replace(hour=15,minute=56)
        self.manage()
        self.assertFalse(self.raw.holdings)
        self.assertEqual(self.raw.sent[-1]['quantity'],prior['quantity']+3)

    def test_untracked_holdings_eod_uses_broker_quantity_without_fake_position(self):
        self.raw.holdings={s:(5,100.) for s in (config.BASE_ASSET_LONG,config.BASE_ASSET_SHORT)}
        self.r._reconcile_active_position_with_ledger(self.raw.get_overseas_stock_balance())
        self.manage();self.assertFalse(self.raw.sent)
        self.assertFalse(self.r._positions)
        self.now=self.now.replace(hour=15,minute=56)
        self.manage()
        self.assertFalse(self.raw.holdings);self.assertFalse(self.r._positions)
        self.assertFalse(self.r.book.data['operations'])
        self.assertEqual([v['quantity'] for v in self.raw.sent],[5,5])

    def test_sell_balance_failure_releases_order_lock(self):
        self.buy(config.BASE_ASSET_LONG);self.raw.fail=True
        with self.assertRaises(RuntimeError):
            self.r._execute_sell_with_10s_chase(config.BASE_ASSET_LONG,1,'SL')
        self.assertFalse(self.r._is_order_in_progress)

    def test_fresh_completed_bar_required_and_forming_bar_excluded(self):
        import pandas as pd
        expected=self.now-timedelta(minutes=config.RELAY_DECISION_INTERVAL_MINUTES)
        index=pd.date_range(end=self.now,periods=100,freq='15min')
        frame=pd.DataFrame({'Close':[100.]*100},index=index)
        prepared=self.r._prepare_decision_candles(frame,self.now,True)
        self.assertEqual(prepared.index[-1],expected)
        self.assertEqual(self.r._decision,decision_key(self.now))
        self.assertIsNone(self.r._candle_error)
        stale=frame.iloc[:-4]
        self.r._prepare_decision_candles(stale,self.now,True)
        self.assertIsNone(self.r._decision)
        self.assertEqual(self.r._entry_quantity(self.raw.cash,100.),0)

    def test_completed_bar_timezone_and_duplicate_validation(self):
        import pandas as pd
        frame=pd.DataFrame({'Close':[100.]*100},index=pd.date_range(
            end=self.now-timedelta(minutes=config.RELAY_DECISION_INTERVAL_MINUTES),periods=100,freq='15min'))
        frame.index=frame.index.tz_convert('UTC')
        self.r._prepare_decision_candles(frame,self.now,True)
        self.assertEqual(self.r._decision,decision_key(self.now))
        bad=pd.concat([frame,frame.iloc[-1:]])
        self.assertTrue(self.r._prepare_decision_candles(bad,self.now,True).empty)
        self.assertIsNone(self.r._decision)

    def test_stale_bar_blocks_production_evaluator_before_model_and_order(self):
        import pandas as pd
        self.r.data_lake=MagicMock()
        self.r.data_lake.load_candles.return_value=pd.DataFrame({'Close':[100.]*100},
            index=pd.date_range(end=self.now.replace(tzinfo=None)-timedelta(hours=1),periods=100,freq='15min'))
        self.r._emit_model_state_alerts_from_eval=MagicMock()
        result=self.r.evaluate_15m_cycle_once(allow_order_execution=True,sync_intraday=False,
            now_dt=self.now,market_session='REGULAR_MARKET_OPEN')
        self.assertTrue(result['data_stale'])
        self.assertEqual(result['stale_reason'],'LATEST_COMPLETED_15M_CANDLE_MISSING')
        self.r.moe_orchestrator.evaluate_dual_filter_signal.assert_not_called()
        self.assertFalse(self.raw.sent)

    def test_primary_submission_cannot_cross_into_another_bar(self):
        long=config.BASE_ASSET_LONG
        self.r.book.begin(self.raw.cash,decision_key(self.now))
        self.r.book.operation(long,dict(action='BUY',role='INITIAL',
            decision_key=decision_key(self.now-timedelta(minutes=config.RELAY_DECISION_INTERVAL_MINUTES)),
            orders=[],submission_pending=False))
        with self.assertRaises(RuntimeError):
            self.r.broker.send_order(symbol=long,order_type='BUY',quantity=1,price=100.)
        self.assertFalse(self.raw.sent)
        self.assertFalse(self.r.book.data['operations'][long]['submission_pending'])


if __name__=='__main__':
    unittest.main()
