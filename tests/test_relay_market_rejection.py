"""Synthetic broker rejection through actual relay/HTTP request code; no network."""
import sys,tempfile,unittest
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
import config
from core.relay_live_runner import RelayLiveRunner
from core.relay_policy import decision_key
from test_b03_order_ledger_integrity import RealContractFakeKiwoomSetup

class TestRelayMarketRejection(unittest.TestCase):
    def exercise(self,chain=False):
        with tempfile.TemporaryDirectory(dir=ROOT/'data') as folder:
            env=RealContractFakeKiwoomSetup(Path(folder));env.write_cb_state()
            runner,broker=env.build_runner_and_broker()
            broker.test_connection=lambda:dict(ok=True,deposit_ok=True,stock_ok=True,usd_order_available=10000.)
            runner.__class__=RelayLiveRunner
            with patch.object(config,'DATA_DIR',Path(folder)):runner._initialize_relay_state()
            runner._deployment_ok=True  # Test fixture only, isolated fake account.
            now=datetime.fromisoformat(env.ny_today+'T10:00:00').replace(tzinfo=ZoneInfo('America/New_York'))
            runner._now=lambda:now;runner._decision=decision_key(now)
            long,short=config.BASE_ASSET_LONG,config.BASE_ASSET_SHORT
            def quote(symbol,price):
                runner._on_websocket_tick(symbol,price,dict(from_ws=True,extra=dict(bid=price,ask=price)))
            def respond(ledger,rec):
                if not chain or rec['symbol']==short:
                    # Deliberately synthetic code/message; not a captured Kiwoom response.
                    return dict(return_code=-999,return_msg='SYNTHETIC: 정규장 주문 가능 시간이 아닙니다')
                ledger['holdings'][long]=dict(qty=rec['quantity'],sell_avail=rec['quantity'],avg_px=rec['price'],cur_px=rec['price'])
                row=dict(ord_dt=env.ny_compact,ord_no=rec['ord_no'],stk_cd=long,slby_tp='2',
                    ord_qty=str(rec['quantity']),cntr_qty=str(rec['quantity']),cntr_no=rec['ord_no']+'F',
                    cncl_qty='0',ord_remnq='0',ord_uv=str(rec['price']))
                ledger['executions'].append(dict(row));ledger['order_history'].append(dict(row))
            env.ledger['on_send_order_hook']=respond
            with patch('core.circuit_breaker.CIRCUIT_STATE_FILE',env.cb_file), \
                 patch('core.relay_live_runner.USMarketCalendar.get_market_status',return_value=dict(is_open=True,session_name='REGULAR_MARKET_OPEN')), \
                 patch('urllib.request.build_opener',side_effect=env.make_opener_factory()):
                quote(long,100.);quote(short,100.)
                qty=runner._entry_quantity(10000.,100.)
                result=runner._execute_buy_with_10s_chase(long,qty,100.,dict(gating_confidence=.7),
                    dict(dynamic_tp_px=110.,dynamic_sl_px=98.,tp_pct=10.,time_stop_minutes=90))
                self.assertEqual(bool(result),chain)
                if chain:
                    quote(long,98.5);runner._manage_open_positions(broker.get_overseas_stock_balance())
                target=short if chain else long
                self.assertNotIn(target,runner._positions)
                self.assertNotIn(target,env.ledger['holdings'])
                self.assertEqual(len(env.ledger['sent_orders']),2 if chain else 1)
                sent=env.ledger['sent_orders'][-1]
                self.assertEqual(sent['api_id'],'ust20000')
                self.assertEqual(sent['symbol'],target)
                self.assertEqual(sent['price'],100.+config.BUY_SLIPPAGE_ADJUST)
                self.assertLessEqual(sent['quantity']*sent['price'],4900.)
                self.assertTrue(runner.book.data['operations'][target]['submission_pending'])
                # Unverified rejection identity stays unresolved: no blind retry/fake fill.
                for _ in range(2):
                    runner._manage_open_positions(broker.get_overseas_stock_balance())
                    self.assertEqual(runner._entry_quantity(10000.,100.),0)
                self.assertEqual(len(env.ledger['sent_orders']),2 if chain else 1)
                self.assertFalse(env.ledger['sent_cancels'])
    def test_initial_market_rejection_no_fill_no_blind_retry(self):self.exercise()
    def test_chain_market_rejection_preserves_parent_no_blind_retry(self):self.exercise(chain=True)

if __name__=='__main__':unittest.main()
