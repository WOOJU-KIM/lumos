"""B-03 adversarial evidence and crash-recovery cases; no real broker calls."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from contextlib import ExitStack
import threading
import config
from core.kiwoom_broker import KiwoomBroker
from core.live_runner import KiwoomLiveRunner


class TestB03EvidenceBoundaries(unittest.TestCase):
    def setUp(self):
        self.broker=KiwoomBroker.__new__(KiwoomBroker)
        self.row=dict(ord_no='fixture-order',stk_cd='SOXL',slby_tp='1',ord_dt='20261016',
                      ord_qty='4',cntr_qty='4',cncl_qty='0',ord_remnq='0')

    def resolve(self,executions=None,orders=None,opened=None,**kwargs):
        return self.broker.resolve_order_ledger_status('fixture-order','SOXL','SELL',expected_qty=4,
            trade_date='20261016',open_orders=[] if opened is None else opened,
            executions=[self.row] if executions is None else executions,
            order_history=[] if orders is None else orders,**kwargs)

    def test_duplicate_terminal_rows_do_not_double_count(self):
        r=self.resolve(executions=[self.row,copy.deepcopy(self.row)])
        self.assertEqual((r['status'],r['filled_qty']),('FILLED',4))

    def test_execution_and_order_history_disagreement_is_unknown(self):
        canceled=dict(self.row,cntr_qty='0',cncl_qty='4')
        self.assertEqual(self.resolve(orders=[canceled])['status'],'UNKNOWN')

    def test_wrong_trade_date_cannot_close_reused_order_number(self):
        self.assertEqual(self.resolve(executions=[dict(self.row,ord_dt='20261015')])['status'],'UNKNOWN')

    def test_wrong_symbol_in_open_order_is_not_accepted(self):
        opened=dict(self.row,stk_cd='SOXS',cntr_qty='0',ord_remnq='4')
        self.assertEqual(self.resolve(opened=[opened])['status'],'UNKNOWN')

    def test_wrong_side_in_open_order_is_not_accepted(self):
        opened=dict(self.row,slby_tp='2',cntr_qty='0',ord_remnq='4')
        self.assertEqual(self.resolve(opened=[opened])['status'],'UNKNOWN')

    def test_missing_remaining_quantity_is_not_zero(self):
        row=dict(self.row);row.pop('ord_remnq')
        self.assertEqual(self.resolve(executions=[row])['status'],'UNKNOWN')

    def test_conflicting_cumulative_and_incremental_fields_use_cumulative(self):
        row=dict(self.row,cntr_qty='1',tot_che_qty='4')
        self.assertEqual(self.resolve(executions=[row])['status'],'FILLED')

    def test_null_cumulative_quantity_is_not_replaced_by_increment(self):
        self.assertEqual(self.resolve(executions=[dict(self.row,tot_che_qty=None)])['status'],'UNKNOWN')

    def test_same_execution_id_with_conflicting_quantity_is_unknown(self):
        a=dict(self.row,cntr_no='fixture-fill',cntr_qty='2',cncl_qty='2')
        b=dict(a,cntr_qty='4',cncl_qty='0')
        self.assertEqual(self.resolve(executions=[a,b])['status'],'UNKNOWN')

    def test_cancel_child_with_wrong_parent_is_unknown(self):
        parent=dict(self.row,cntr_qty='0',cncl_qty='4')
        child=dict(parent,ord_no='fixture-cancel',orig_ord_no='other-parent')
        self.assertEqual(self.resolve(executions=[],orders=[parent,child],cancel_ord_no='fixture-cancel')['status'],'UNKNOWN')

    def test_failed_history_query_cannot_be_hidden_by_execution_evidence(self):
        self.broker.get_open_orders=Mock(return_value={'ok':True,'orders':[]})
        self.broker.get_daily_execution_history=Mock(return_value={'ok':True,'executions':[self.row]})
        self.broker.get_order_history=Mock(return_value={'ok':False,'orders':[]})
        r=self.broker.resolve_order_ledger_status('fixture-order','SOXL','SELL',4,'20261016')
        self.assertEqual(r['status'],'UNKNOWN')

    def test_cancel_rows_are_deduplicated(self):
        p=dict(self.row,cntr_qty='0',cncl_qty='4')
        c=dict(p,ord_no='fixture-cancel',orig_ord_no='fixture-order')
        r=self.resolve(executions=[],orders=[p,c,dict(c)],cancel_ord_no='fixture-cancel')
        self.assertEqual((r['status'],r['cancelled_qty']),('CANCELED_ZERO_FILL',4))

    def test_unidentifiable_duplicate_partial_execution_stays_unknown(self):
        p=dict(self.row,cntr_qty='1',cncl_qty='0')
        self.assertEqual(self.resolve(executions=[p,dict(p)])['status'],'UNKNOWN')

    def test_distinct_execution_ids_are_summed_once(self):
        a=dict(self.row,cntr_qty='2',cntr_no='a');b=dict(a,cntr_no='b')
        self.assertEqual(self.resolve(executions=[a,b,dict(a)])['filled_qty'],4)


class TestB03PendingBuyRecovery(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'position.json'

    def runner(self,simulation=False):
        r=KiwoomLiveRunner.__new__(KiwoomLiveRunner)
        r.pos_file_path=str(self.path)
        r.broker=SimpleNamespace(account_no='FIXTURE_ONLY',account_type='TEST',is_simulation=simulation)
        return r

    def test_pending_buy_survives_new_runner_instance(self):
        a=self.runner();a._unconfirmed_buy_order={'symbol':'SOXL','order_no':'fixture-order','target_qty':4}
        self.assertEqual(getattr(self.runner(),'_unconfirmed_buy_order',None)['order_no'],'fixture-order')

    def test_mode_change_does_not_reuse_real_pending_order(self):
        a=self.runner();a._unconfirmed_buy_order={'symbol':'SOXL','order_no':'fixture-order','target_qty':4}
        r=getattr(self.runner(True),'_unconfirmed_buy_order',None)
        self.assertIsNotNone(r);self.assertNotEqual(r.get('order_no'),'fixture-order')

    def test_clear_is_durable(self):
        a=self.runner();a._unconfirmed_buy_order={'symbol':'SOXL','order_no':'fixture-order','target_qty':4}
        self.assertIsNotNone(getattr(self.runner(),'_unconfirmed_buy_order',None))
        a._unconfirmed_buy_order=None
        self.assertIsNone(getattr(self.runner(),'_unconfirmed_buy_order',None))

    def test_corrupt_journal_blocks_new_entry(self):
        from core.pending_order import journal_path
        journal_path(self.path).write_text('{broken',encoding='utf-8')
        r=self.runner()
        self.assertFalse(r._verify_pre_buy_ledger_clean('SOXL'))

    def test_failed_journal_write_raises_before_order_send(self):
        r=self.runner()
        with patch('core.pending_order.os.replace',side_effect=OSError('TEST_ONLY')):
            with self.assertRaises(OSError):
                r._unconfirmed_buy_order={'symbol':'SOXL','target_qty':4}
        self.assertTrue(r._pending_buy_write_failed)
        self.assertFalse(r._verify_pre_buy_ledger_clean('SOXL'))


class TestB03RunnerBoundaries(unittest.TestCase):
    def setUp(self):
        from test_b03_order_ledger_integrity import RealContractFakeKiwoomSetup
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.env=RealContractFakeKiwoomSetup(Path(self.temp.name));self.env.write_cb_state()
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch('core.circuit_breaker.CIRCUIT_STATE_FILE',self.env.cb_file))
        self.runner,self.broker=self.env.build_runner_and_broker()
        self.stack.enter_context(patch('urllib.request.build_opener',side_effect=self.env.make_opener_factory()))
        self.stack.enter_context(patch.object(config,'CANCEL_ORDER_WAIT_TIME',0.001))
        self.stack.enter_context(patch.object(config,'WS_FILL_POLL_INTERVAL',0.001))

    def sell(self):
        self.env.ledger['holdings']['SOXL']={'qty':4,'sell_avail':4,'avg_px':100.,'cur_px':97.}
        self.runner._save_active_position(dict(symbol='SOXL',quantity=4,price=100.,buy_time='2026-10-16 10:00:00',order_no='fixture-buy'))
        self.runner._wait_for_fill=Mock(return_value=(False,0,4))

    def test_sell_retry_rejects_conflicting_order_history(self):
        self.sell()
        def cancel(ledger,rec):
            p=dict(ord_dt=self.env.ny_compact,ord_no=rec['order_no'],stk_cd='SOXL',slby_tp='1',ord_qty='4',cntr_qty='0',cncl_qty='4',ord_remnq='0')
            ledger['executions']=[p];ledger['order_history']=[dict(p,cntr_qty='1',cncl_qty='3')]
            return dict(ok=True,order_no=rec['order_no'],cancel_ord_no='fixture-cancel')
        self.env.ledger['on_cancel_order_hook']=cancel
        self.assertFalse(self.runner._execute_sell_with_10s_chase('SOXL',4,'TEST',100.,97.))
        self.assertEqual(len(self.env.ledger['sent_orders']),1)
        self.assertIsNotNone(self.runner._get_active_position())

    def test_zero_sell_available_does_not_fall_back_to_possessed(self):
        self.sell()
        def cancel(ledger,rec):
            p=dict(ord_dt=self.env.ny_compact,ord_no=rec['order_no'],stk_cd='SOXL',slby_tp='1',ord_qty='4',cntr_qty='0',cncl_qty='4',ord_remnq='0')
            ledger['executions']=[p];ledger['order_history']=[dict(p)]
            ledger['holdings']['SOXL']['sell_avail']=0
            return dict(ok=True,order_no=rec['order_no'],cancel_ord_no='fixture-cancel')
        self.env.ledger['on_cancel_order_hook']=cancel
        self.assertFalse(self.runner._execute_sell_with_10s_chase('SOXL',4,'TEST',100.,97.))
        self.assertEqual(len(self.env.ledger['sent_orders']),1)

    def test_initial_balance_failure_sends_no_sell(self):
        self.sell();self.env.ledger['balance_ok']=False
        self.assertFalse(self.runner._execute_sell_with_10s_chase('SOXL',4,'TEST',100.,97.))
        self.assertEqual(self.env.ledger['sent_orders'],[])

    def test_ws_fill_count_is_not_completion_if_balance_disagrees(self):
        self.runner._order_fills['fixture-sell']=4
        self.env.ledger['holdings']['SOXL']={'qty':4,'sell_avail':4,'avg_px':100.,'cur_px':97.}
        done,filled,left=self.runner._wait_for_fill('fixture-sell','SOXL',False,4,timeout_sec=.001)
        self.assertFalse(done);self.assertEqual(left,4)

    def test_ws_fill_count_with_balance_failure_stays_unconfirmed(self):
        self.runner._order_fills['fixture-sell']=4;self.env.ledger['balance_ok']=False
        self.assertFalse(self.runner._wait_for_fill('fixture-sell','SOXL',False,4,timeout_sec=.001)[0])

    def test_prior_day_zero_fill_cancel_recovers_after_restart(self):
        day=self.env.ny_yesterday_compact
        self.runner._unconfirmed_buy_order=dict(symbol='SOXL',order_no='fixture-prior',target_qty=4,trade_date=day)
        row=dict(ord_dt=day,ord_no='fixture-prior',stk_cd='SOXL',slby_tp='2',ord_qty='4',cntr_qty='0',cncl_qty='4',ord_remnq='0')
        self.env.ledger['order_history']=[row]
        self.runner._pending_buy_cache=None
        self.assertTrue(self.runner._verify_pre_buy_ledger_clean('SOXL'))
        self.assertIsNone(self.runner._unconfirmed_buy_order)

    def test_accepted_buy_id_is_durable_before_wait_then_released_after_full_fill(self):
        self.runner.ws_streamer.latest_prices['SOXL']=100.
        def send(ledger,rec):
            ledger['holdings']['SOXL']={'qty':4,'sell_avail':4,'avg_px':rec['price'],'cur_px':rec['price']}
        self.env.ledger['on_send_order_hook']=send
        seen=[]
        def wait(**kwargs):
            from core.pending_order import read_pending,journal_path,binding_for
            pending=read_pending(journal_path(self.runner._resolve_pos_file()),binding_for(self.broker))
            self.assertEqual(pending['order_no'],kwargs['order_no']);seen.append(True)
            return True,4,0
        self.runner._wait_for_fill=wait
        targets=dict(dynamic_tp_px=110.,dynamic_sl_px=98.,tp_pct=10.,sl_pct=2.,time_stop_minutes=90)
        self.assertTrue(self.runner._execute_buy_with_10s_chase('SOXL',4,100.,moe_res={'gating_confidence':.8},targets=targets))
        self.assertTrue(seen);self.assertIsNone(self.runner._unconfirmed_buy_order)


if __name__=='__main__':unittest.main()
