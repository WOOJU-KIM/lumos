"""Real Kiwoom broker request/ledger reducer with intercepted HTTP, no network."""
import sys
import tempfile
import unittest
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).parent))
import config
from core.relay_live_runner import RelayLiveRunner
from core.relay_policy import decision_key
from test_b03_order_ledger_integrity import RealContractFakeKiwoomSetup


class TestRelayBrokerContract(unittest.TestCase):
    def test_real_broker_two_legs_no_reentry_and_eod(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'data') as folder:
            env=RealContractFakeKiwoomSetup(Path(folder));env.write_cb_state()
            runner,broker=env.build_runner_and_broker()
            broker.test_connection=lambda:dict(ok=True,deposit_ok=True,stock_ok=True,usd_order_available=10000.)
            runner.__class__=RelayLiveRunner
            with patch.object(config,'DATA_DIR',Path(folder)):
                runner._initialize_relay_state()
            runner._deployment_ok=True  # Synthetic transport QA, never a production approval.
            now=datetime.fromisoformat(env.ny_today+'T10:00:00').replace(tzinfo=ZoneInfo('America/New_York'))
            runner._now=lambda:now;runner._decision=decision_key(now)
            def quote(symbol,bid):
                runner._on_websocket_tick(symbol,bid,dict(from_ws=True,extra=dict(bid=bid,ask=bid)))
            long,short=config.BASE_ASSET_LONG,config.BASE_ASSET_SHORT
            def fill(ledger,record):
                symbol=record['symbol'];qty=record['quantity'];price=record['price']
                buy=record['api_id']=='ust20000'
                if buy:ledger['holdings'][symbol]=dict(qty=qty,sell_avail=qty,avg_px=price,cur_px=price)
                else:ledger['holdings'].pop(symbol,None)
                row=dict(ord_dt=env.ny_compact,ord_no=record['ord_no'],stk_cd=symbol,
                    slby_tp='2' if buy else '1',ord_qty=str(qty),cntr_qty=str(qty),
                    cntr_no=record['ord_no']+'F',cncl_qty='0',ord_remnq='0',ord_uv=str(price))
                ledger['executions'].append(dict(row));ledger['order_history'].append(dict(row))
            env.ledger['on_send_order_hook']=fill
            with patch('core.circuit_breaker.CIRCUIT_STATE_FILE',env.cb_file), \
                 patch('core.relay_live_runner.USMarketCalendar.get_market_status',return_value=dict(is_open=True,session_name='REGULAR_MARKET_OPEN')), \
                 patch('urllib.request.build_opener',side_effect=env.make_opener_factory()):
                quote(long,100.);quote(short,100.)
                qty=runner._entry_quantity(10000.,100.)
                self.assertTrue(runner._execute_buy_with_10s_chase(long,qty,100.,dict(gating_confidence=.7),
                    dict(dynamic_tp_px=110.,dynamic_sl_px=98.,tp_pct=10.,time_stop_minutes=90)))
                quote(long,98.5);runner._manage_open_positions(broker.get_overseas_stock_balance())
                self.assertEqual(set(runner._positions),{long,short})
                quote(long,97.);runner._manage_open_positions(broker.get_overseas_stock_balance())
                self.assertEqual(set(runner._positions),{short})
                # 남은 short가 -1% 빠지면 2번째 체인(long) 발동 허용 (최대 3회까지)
                quote(short,98.5);runner._manage_open_positions(broker.get_overseas_stock_balance())
                self.assertEqual(set(runner._positions),{short,long})
                self.assertFalse(runner.book.data['operations'])

                self.assertEqual(len(env.ledger['sent_orders']),4)
                self.assertEqual([r['api_id'] for r in env.ledger['sent_orders']],
                                 ['ust20000','ust20000','ust20001','ust20000'])
                now=now.replace(hour=15,minute=55)
                runner._manage_open_positions(broker.get_overseas_stock_balance())
                self.assertFalse(runner._positions)
                self.assertFalse(env.ledger['holdings'])
                self.assertFalse(runner.book.data['operations'])
                # Missing strategy metadata must not prevent actual broker EOD SELL.
                env.ledger['holdings'][long]=dict(qty=5,sell_avail=5,avg_px=100.,cur_px=100.)
                env.ledger['next_ord_seq']+=1
                external=dict(ord_dt=env.ny_compact,ord_no=str(env.ledger['next_ord_seq']).zfill(9),
                    stk_cd=long,slby_tp='2',ord_qty='5',cntr_qty='5',cntr_no='EXTERNAL_FILL',
                    cncl_qty='0',ord_remnq='0',ord_uv='100.')
                env.ledger['executions'].append(dict(external))
                env.ledger['order_history'].append(dict(external))
                runner._manage_open_positions(broker.get_overseas_stock_balance())
                self.assertFalse(env.ledger['holdings'])
                self.assertFalse(runner._positions)
                self.assertEqual(env.ledger['sent_orders'][-1]['api_id'],'ust20001')


if __name__=='__main__':unittest.main()
