"""Chain-specific live execution cases, synthetic broker/quotes only."""
from pathlib import Path
import sys
import copy
import time
import unittest
from unittest.mock import patch, MagicMock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
import config
import test_opposite_relay as fixtures
from core.kiwoom_broker import KiwoomBroker
from core.relay_policy import RelayBook, trailing_trigger


class TestChainSendPrice(unittest.TestCase):
    def setUp(self):
        self.case=fixtures.TestRelay(methodName='test_initial_49_and_chain_49_real_helpers_and_roles')
        self.case.setUp();self.addCleanup(self.case.doCleanups)
        self.r=self.case.r;self.raw=self.case.raw
        self.source=config.BASE_ASSET_LONG;self.target=config.BASE_ASSET_SHORT

    def initial(self,short=False):
        if short:self.source,self.target=self.target,self.source
        self.assertTrue(self.case.buy(self.source))
        self.case.quote(self.source,98.5)

    def hook_final_preflight(self,callback):
        original=self.raw.get_open_orders;calls=0
        def change(*args):
            nonlocal calls
            calls+=1
            if calls==3:callback()
            return original(*args)
        return patch.object(self.raw,'get_open_orders',side_effect=change)

    def chain(self):
        return self.case.buy(self.target,role='CHAIN',trigger=self.source)

    def assert_repriced_chain(self,short=False):
        self.initial(short)
        with self.hook_final_preflight(lambda:self.case.quote(self.target,101.)):
            self.assertTrue(self.chain())
        order=self.raw.sent[-1]
        self.assertEqual(order['symbol'],self.target)
        self.assertEqual(order['price'],101.03)
        self.assertLessEqual(order['price']*order['quantity'],4900.)
        self.assertGreaterEqual(self.raw.cash,200.)
        self.assertEqual(self.r._positions[self.target]['relay_role'],'CHAIN')
        self.assertEqual(trailing_trigger(self.r._positions[self.target]['relay_role']),.01)
        self.assertEqual(trailing_trigger(self.r._positions[self.source]['relay_role']),.015)
        self.assertFalse(self.chain())
        self.assertEqual(len(self.raw.sent),2)

    def test_long_initial_short_chain_latest_price_and_one_chain(self):
        self.assert_repriced_chain()

    def test_short_initial_long_chain_latest_price_and_one_chain(self):
        self.assert_repriced_chain(True)

    def test_source_rebound_before_send_cancels_chain_without_unknown_submission(self):
        self.initial()
        with self.hook_final_preflight(lambda:self.case.quote(self.source,100.)):
            self.assertFalse(self.chain())
        self.assertEqual(len(self.raw.sent),1)
        self.assertFalse(self.r.book.data['operations'])
        self.assertTrue(self.r.book.data['chain_operation_id'])
        # 3회 연쇄 규칙: 첫 시도가 취소되었어도 남은 기회로 2번째 체인 발동 허용
        self.case.quote(self.source,98.5);self.case.manage()
        self.assertEqual(len(self.raw.sent),2)

    def test_stale_source_before_send_blocks_chain(self):
        self.initial()
        def stale():self.r._quotes[self.source]['time']=time.time()-config.RELAY_QUOTE_MAX_AGE_SECONDS-1
        with self.hook_final_preflight(stale):self.assertFalse(self.chain())
        self.assertEqual(len(self.raw.sent),1)
        self.assertFalse(self.r.book.data['operations'])

    def test_chain_reprice_cannot_exceed_original_cycle_allocation(self):
        self.initial()
        with self.hook_final_preflight(lambda:self.case.quote(self.target,110.)):
            self.assertFalse(self.chain())
        self.assertEqual(len(self.raw.sent),1)
        self.assertFalse(self.r.book.data['operations'])

    def test_chain_reservation_changed_during_preflight_blocks_send(self):
        self.initial()
        with self.hook_final_preflight(lambda:self.r.book.update(chain_operation_id='OTHER')):
            self.assertFalse(self.chain())
        self.assertEqual(len(self.raw.sent),1)

    def test_parent_quantity_changed_during_preflight_blocks_send(self):
        self.initial()
        def changed():self.r._positions[self.source]['quantity']-=1
        with self.hook_final_preflight(changed):self.assertFalse(self.chain())
        self.assertEqual(len(self.raw.sent),1)

    def test_source_rebound_during_connection_preparation_blocks_final_resolver(self):
        self.initial()
        original=KiwoomBroker.__new__(KiwoomBroker)
        original.get_overseas_stock_balance=self.raw.get_overseas_stock_balance
        original.get_open_orders=self.raw.get_open_orders
        original.test_connection=self.raw.test_connection
        def delayed(**kwargs):
            resolver=kwargs['price_resolver']
            resolver()  # Initial request preparation passed.
            self.case.quote(self.source,100.)  # Simulated TLS/connect delay.
            return resolver()  # Must reject before HTTP writes.
        original.send_order=MagicMock(side_effect=delayed)
        self.r.broker.original=original
        self.assertFalse(self.chain())
        self.assertEqual(original.send_order.call_count,1)
        self.assertEqual(len(self.raw.sent),1)
        self.assertFalse(self.r.book.data['operations'])

    def test_partial_chain_reprices_only_remaining_quantity_with_same_reservation(self):
        self.initial();qty=self.r.book.quantity(self.raw.cash,100.,'CHAIN')
        operation=dict(id='PARTIAL_CHAIN',action='BUY',role='CHAIN',trigger_symbol=self.source,
            quantity=qty,held_before=0,created_ny=self.case.now.isoformat(),decision_key=None,
            budget_cap=4900.,orders=[],submission_pending=False)
        self.r.book.operation(self.target,operation)
        attempts=[]
        def send(**order):
            attempts.append(copy.deepcopy(order));number='PARTIAL'+str(len(attempts))
            fill=4 if len(attempts)==1 else order['quantity']
            held,avg=self.raw.holdings.get(self.target,(0,0.))
            self.raw.holdings[self.target]=(held+fill,(held*avg+fill*order['price'])/(held+fill))
            self.raw.cash-=fill*order['price']
            return dict(ok=True,order_no=number)
        with patch.object(self.raw,'send_order',side_effect=send):
            self.r.broker.send_order(symbol=self.target,order_type='BUY',quantity=qty,price=100.03)
            self.case.quote(self.target,101.)
            self.r.broker.send_order(symbol=self.target,order_type='BUY',quantity=qty-4,price=100.03)
        self.assertEqual([a['quantity'] for a in attempts],[qty,qty-4])
        self.assertEqual(attempts[1]['price'],101.03)
        self.assertEqual(self.r.book.data['chain_operation_id'],'PARTIAL_CHAIN')
        self.raw.resolve_order_ledger_status=lambda order_no,*args,**kwargs: (
            dict(status='CANCELED_PARTIAL_FILL',filled_qty=4) if order_no=='PARTIAL1' else
            dict(status='FILLED',filled_qty=qty-4))
        self.r._reconcile_active_position_with_ledger(self.raw.get_overseas_stock_balance())
        self.assertEqual(self.r._positions[self.target]['quantity'],qty)
        self.assertEqual(self.r._positions[self.target]['relay_role'],'CHAIN')
        self.assertFalse(self.r.book.data['operations'])
        self.assertFalse(self.chain())

    def test_unknown_chain_submission_persists_and_blocks_repeat_after_reload(self):
        self.initial();self.raw.unknown=True
        self.assertFalse(self.chain())
        self.r.book=RelayBook(self.r.book.path,self.r._binding)
        self.assertTrue(self.r.book.data['operations'][self.target]['submission_pending'])
        self.assertTrue(self.r.book.data['chain_operation_id'])
        self.assertFalse(self.chain())
        self.assertEqual(len(self.raw.sent),2)
        self.raw.unknown=False
        self.case.quote(self.source,97.);self.case.manage()
        self.assertNotIn(self.source,self.raw.holdings)
        self.assertIn(self.target,self.r.book.data['operations'])


if __name__=='__main__':unittest.main()
