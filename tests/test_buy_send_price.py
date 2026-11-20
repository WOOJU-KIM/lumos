"""Fresh limit after broker preflight and TLS, with no real socket/order calls."""
from pathlib import Path
import sys
import json
import unittest
import tempfile
from datetime import datetime
from zoneinfo import ZoneInfo
from unittest.mock import patch, MagicMock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
import config
import test_opposite_relay as fixture_module
from core.kiwoom_broker import _OrderHTTPSConnection
from core.telegram_notifier import TelegramNotifier
from core.relay_live_runner import RelayLiveRunner
from core.relay_policy import decision_key
from test_b03_order_ledger_integrity import RealContractFakeKiwoomSetup


class TestBuySendPrice(unittest.TestCase):
    def setUp(self):
        self.fixture=fixture_module.TestRelay(methodName='test_initial_49_and_chain_49_real_helpers_and_roles')
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.r=self.fixture.r;self.raw=self.fixture.raw;self.long=config.BASE_ASSET_LONG

    def test_quote_change_during_broker_preflight_updates_submitted_limit_and_logs(self):
        original=self.raw.get_open_orders
        def changing(*args):
            self.fixture.quote(self.long,101.)
            return original(*args)
        with patch.object(self.raw,'get_open_orders',side_effect=changing):
            self.assertTrue(self.fixture.buy(self.long))
        self.assertEqual(self.raw.sent[0]['price'],101.03)
        self.assertEqual(self.r._positions[self.long]['price'],101.03)
        self.assertEqual(self.raw.sent[0]['quantity'],48)

    def test_reprice_above_budget_does_not_send_or_leave_uncertain_intent(self):
        original=self.raw.get_open_orders
        calls=0
        def changing(*args):
            nonlocal calls
            calls+=1
            if calls>=3:self.fixture.quote(self.long,110.)
            return original(*args)
        with patch.object(self.raw,'get_open_orders',side_effect=changing):
            self.assertFalse(self.fixture.buy(self.long))
        self.assertFalse(self.raw.sent)
        self.assertFalse(self.r.book.data['operations'])
        self.assertFalse(self.r._is_order_in_progress)

    def test_tls_preparation_precedes_last_quote_and_content_length_is_rebuilt(self):
        events=[];quote={'ask':99.}
        def payload():
            events.append('price')
            return json.dumps({'ord_uv':str(round(quote['ask']+config.BUY_SLIPPAGE_ADJUST,2))}).encode()
        conn=_OrderHTTPSConnection('example.invalid',order_guard=lambda:None,body_resolver=payload)
        def connect():
            events.append('tls');quote['ask']=102.2;conn.sock=MagicMock()
        with patch.object(conn,'connect',side_effect=connect),patch('http.client.HTTPSConnection.request') as request:
            conn.request('POST','/api/us/ordr',body=b'old',headers={'Content-length':'3'})
        self.assertEqual(events,['tls','price'])
        body=request.call_args.kwargs['body'];headers=request.call_args.kwargs['headers']
        self.assertEqual(json.loads(body)['ord_uv'],'102.23')
        self.assertEqual(headers['Content-Length'],str(len(body)))
        self.assertEqual(sum(k.lower()=='content-length' for k in headers),1)

    def test_price_resolver_rejection_before_headers_does_not_call_request(self):
        conn=_OrderHTTPSConnection('example.invalid',order_guard=lambda:None,
                                  body_resolver=MagicMock(side_effect=RuntimeError('STALE_WS')))
        conn.sock=MagicMock()
        with patch('http.client.HTTPSConnection.request') as request,self.assertRaises(RuntimeError):
            conn.request('POST','/api/us/ordr',body=b'old')
        request.assert_not_called()

    def test_existing_transport_without_resolver_is_unchanged(self):
        conn=_OrderHTTPSConnection('example.invalid',order_guard=lambda:None)
        with patch.object(conn,'connect') as connect,patch('http.client.HTTPSConnection.request') as request:
            conn.request('POST','/api/us/ordr',body=b'legacy',headers={'Content-Length':'6'})
        connect.assert_not_called()
        self.assertEqual(request.call_args.kwargs['body'],b'legacy')

    def test_real_broker_http_payload_reprices_after_token_work_and_skips_rest_quote(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'data') as folder:
            env=RealContractFakeKiwoomSetup(Path(folder));env.write_cb_state()
            runner,broker=env.build_runner_and_broker();runner.__class__=RelayLiveRunner
            with patch.object(config,'DATA_DIR',Path(folder)):
                runner._initialize_relay_state()
            runner._deployment_ok=True
            now=datetime.fromisoformat(env.ny_today+'T10:00:00').replace(tzinfo=ZoneInfo('America/New_York'))
            runner._now=lambda:now;runner._decision=decision_key(now)
            def quote(symbol,ask):
                runner._on_websocket_tick(symbol,ask,dict(from_ws=True,extra=dict(bid=ask,ask=ask)))
            quote(config.BASE_ASSET_LONG,100.);quote(config.BASE_ASSET_SHORT,100.)
            runner.book.begin(10000.,runner._decision)
            runner.book.operation(config.BASE_ASSET_LONG,dict(id='PRICE_TEST',action='BUY',role='INITIAL',
                trigger_symbol=None,decision_key=runner._decision,quantity=2,held_before=0,
                created_ny=now.isoformat(),budget_cap=4900.,orders=[],submission_pending=False))
            def token(force_refresh=False):
                quote(config.BASE_ASSET_LONG,101.)
                return 'FAKE_TOKEN'
            broker.get_access_token=token
            broker.get_stock_quote=MagicMock(side_effect=AssertionError('No REST quote for relay BUY'))
            with patch('core.circuit_breaker.CIRCUIT_STATE_FILE',env.cb_file), \
                 patch('core.relay_live_runner.USMarketCalendar.get_market_status',return_value=dict(is_open=True,session_name='REGULAR_MARKET_OPEN')), \
                 patch('urllib.request.build_opener',side_effect=env.make_opener_factory()):
                result=runner.broker.send_order(symbol=config.BASE_ASSET_LONG,order_type='BUY',quantity=2,price=100.03)
            self.assertEqual(env.ledger['sent_orders'][0]['price'],101.03)
            self.assertEqual(result['price'],101.03)
            broker.get_stock_quote.assert_not_called()

    def terminal_orders(self,status):
        self.r.book.begin(10000.,self.r._decision)
        self.r.book.operation(self.long,dict(id='FIXTURE',action='BUY',role='INITIAL',quantity=48,
            held_before=0,created_ny=self.fixture.now.isoformat(),submission_pending=False,
            orders=[{'order_no':'ONE','quantity':48},{'order_no':'TWO','quantity':48}]))
        self.raw.resolve_order_ledger_status=lambda *args,**kwargs:dict(status=status,filled_qty=0)
        self.r._reconcile_active_position_with_ledger(self.raw.get_overseas_stock_balance())

    def test_confirmed_zero_fill_cancellation_has_distinct_outcome(self):
        self.terminal_orders('CANCELED_ZERO_FILL')
        self.assertEqual(self.r._buy_execution_outcome,'BUY_UNFILLED_CANCELLED')
        self.assertEqual(self.r._buy_execution_attempts,2)
        self.assertFalse(self.r.book.data['operations'])

    def test_unknown_evidence_cannot_be_reported_as_cancelled(self):
        self.terminal_orders('UNKNOWN')
        self.assertNotEqual(getattr(self.r,'_buy_execution_outcome',None),'BUY_UNFILLED_CANCELLED')
        self.assertTrue(self.r.book.data['operations'])

    def test_unfilled_evaluator_and_message_do_not_claim_risk_block(self):
        def unfilled(*args,**kwargs):
            self.r._buy_execution_outcome='BUY_UNFILLED_CANCELLED';self.r._buy_execution_attempts=2
            return False
        with patch.object(self.r,'_execute_buy_with_10s_chase',side_effect=unfilled):
            result=self.fixture.evaluate()
        self.assertEqual(result['order_block_reason'],'ORDER_UNFILLED')
        notifier=TelegramNotifier.__new__(TelegramNotifier)
        msg=notifier.compose_signal_briefing_alert(self.r._last_moe_res,'10:30','23:30',
            order_allowed=False,order_executed=False,order_block_reason=result['order_block_reason'],
            order_block_detail=result['order_block_detail'])
        self.assertIn('2회 / 체결 0주 / 취소 확인 완료',msg)
        self.assertIn('주문 처리 상태: 미체결 취소 완료',msg)
        self.assertIn('매수 체결 여부: NO',msg)
        self.assertNotIn('RISK_BLOCK',msg)
        self.assertNotIn('주문 차단',msg)


if __name__=='__main__':unittest.main()
