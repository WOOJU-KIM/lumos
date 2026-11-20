"""Capture real relay notification templates without Telegram or broker network."""
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys
import copy

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
import config
from core.telegram_notifier import TelegramNotifier
import test_opposite_relay as relay_fixture


class TestRelayTelegram(unittest.TestCase):
    def setUp(self):
        self.fixture=relay_fixture.TestRelay(methodName='test_initial_49_and_chain_49_real_helpers_and_roles')
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.r=self.fixture.r
        self.notifier=TelegramNotifier.__new__(TelegramNotifier)
        self.notifier._dispatch_message=MagicMock(return_value={'ok':True})
        self.r.notifier=self.notifier
        self.long,self.short=config.BASE_ASSET_LONG,config.BASE_ASSET_SHORT

    def messages(self,kind):
        return [c.args[0] for c in self.notifier._dispatch_message.call_args_list
                if c.kwargs.get('alert_type')==kind]

    def two_legs(self):
        self.assertTrue(self.fixture.buy(self.long))
        self.fixture.quote(self.long,98.5);self.fixture.manage()
        self.assertEqual(set(self.r._positions),{self.long,self.short})

    def test_initial_and_chain_actual_helper_messages_are_distinct(self):
        self.two_legs();messages=self.messages('ENTRY_BUY')
        self.assertEqual(len(messages),2)
        for expected in ['최초 매수 체결','49%','GBDT 점수: 70.0%','+1.5%','-0.3%','고정 손절','-2%']:
            self.assertIn(expected,messages[0])
        for expected in ['연쇄 매수 체결','49%',self.long,'-1%','+1%','-0.3%','GBDT 신규 판단을 사용하지 않음']:
            self.assertIn(expected,messages[1])
        self.assertNotIn('GBDT 점수:',messages[1])
        self.assertNotIn('GBDT 확신도',messages[1])
        self.assertNotIn('ATR',messages[1])

    def test_one_leg_close_does_not_claim_complete_cycle(self):
        self.two_legs();self.fixture.quote(self.long,97.);self.fixture.manage()
        message=self.messages('EXIT_SELL')[-1]
        self.assertIn('최초 매수분',message)
        self.assertIn(self.short,message)
        self.assertIn('남은 포지션 청산을 기다립니다',message)
        self.assertFalse(self.messages('RELAY_CYCLE_COMPLETE'))
        self.r.dispatcher.send_telegram_message.assert_not_called()

    def test_final_close_verifies_cycle_once_and_remains_closed_on_reconcile(self):
        self.two_legs()
        self.fixture.quote(self.long,97.);self.fixture.manage()
        self.fixture.quote(self.short,97.);self.fixture.manage()
        self.assertEqual(len(self.messages('EXIT_SELL')),2)
        self.assertEqual(len(self.messages('RELAY_CYCLE_COMPLETE')),1)
        self.assertIn('원장 기준',self.messages('RELAY_CYCLE_COMPLETE')[0])
        self.assertIn('다음 새 15분봉',self.messages('RELAY_CYCLE_COMPLETE')[0])
        self.assertIn('연쇄 매수분',self.messages('EXIT_SELL')[-1])
        self.fixture.manage()
        self.assertEqual(len(self.messages('RELAY_CYCLE_COMPLETE')),1)
        self.assertIsNone(self.r.book.data['cycle_id'])

    def test_eod_two_sell_alerts_then_one_cycle_complete(self):
        self.two_legs()
        self.fixture.now=self.fixture.now.replace(hour=15,minute=55)
        self.fixture.manage()
        messages=self.messages('EXIT_SELL')
        self.assertEqual(len(messages),2)
        self.assertTrue(all('Relay EOD' in m for m in messages))
        self.assertEqual(len(self.messages('RELAY_CYCLE_COMPLETE')),1)
        self.assertFalse(self.r._positions)
        self.assertFalse(self.fixture.raw.holdings)

    def test_notification_error_does_not_change_buy_or_reconciled_exit(self):
        self.notifier._dispatch_message.side_effect=RuntimeError('SYNTHETIC_NOTIFICATION_FAILURE')
        self.assertTrue(self.fixture.buy(self.long))
        self.fixture.quote(self.long,97.);self.fixture.manage()
        self.assertFalse(self.r._positions)
        self.assertIsNone(self.r.book.data['cycle_id'])
        self.assertFalse(self.r._is_order_in_progress)

    def test_unresolved_order_does_not_send_cycle_complete(self):
        self.fixture.raw.unknown=True
        self.assertFalse(self.fixture.buy(self.long))
        self.assertTrue(self.r.book.data['operations'])
        self.assertFalse(self.messages('RELAY_CYCLE_COMPLETE'))

    def test_unverified_broker_open_orders_cannot_claim_cycle_complete(self):
        self.assertTrue(self.fixture.buy(self.long))
        with patch.object(self.r,'_orders_empty',return_value=False):
            self.fixture.quote(self.long,97.);self.fixture.manage()
        self.assertFalse(self.r._positions)
        self.assertFalse(self.messages('RELAY_CYCLE_COMPLETE'))

    def test_startup_and_open_show_current_config_without_atr_stop_claim(self):
        startup=self.notifier.compose_startup_alert({})
        opened=self.notifier.compose_market_open_alert({},'SIMULATION','SYNTHETIC')
        for message in [startup,opened]:
            for expected in ['49%','연쇄 1회','-1%','+1.5%','+1%','-0.3%','고정 손절 -2%','15:55','두 종목 모두']:
                self.assertIn(expected,message)
        self.assertNotIn('ATR SL',opened)
        self.assertIn('연결 예정',startup)

    def test_non_relay_notifications_keep_legacy_entry_format(self):
        self.notifier.send_entry_alert(self.long,100.,1,.7,'HOLD',is_simulation=True)
        message=self.messages('ENTRY_BUY')[-1]
        self.assertIn('Lumos 신규 진입',message)
        self.assertIn('GBDT 확신도',message)
        self.assertNotIn('최초 매수 체결',message)

    def test_html_escape_and_cycle_short_id(self):
        self.notifier.send_entry_alert(self.short,100.,1,0.,'HOLD',is_simulation=True,
            model_snapshot_info={'relay_role':'CHAIN','relay_trigger_symbol':'<SOURCE>'})
        self.assertIn('&lt;SOURCE&gt;',self.messages('ENTRY_BUY')[-1])
        self.assertNotIn('<SOURCE>',self.messages('ENTRY_BUY')[-1])
        self.notifier.send_relay_cycle_complete_alert('abcdefghSECRET',True)
        self.assertNotIn('SECRET',self.messages('RELAY_CYCLE_COMPLETE')[-1])

    def test_entry_notification_does_not_mutate_original_model_decision(self):
        source={'gating_confidence':.7,'gbdt_probs':{},'all_gating_confidences':{}}
        before=copy.deepcopy(source)
        self.r.book.begin(self.fixture.raw.cash,self.r._decision)
        self.assertTrue(self.r._execute_buy_with_10s_chase(self.long,48,100.,source,
            dict(tp_pct=config.MAX_TP_PCT*100,sl_pct=config.SL_MIN_PCT*100,
                 dynamic_tp_px=110.,dynamic_sl_px=98.,time_stop_minutes=config.TIME_STOP_MINUTES)))
        self.assertEqual(source,before)


if __name__=='__main__':unittest.main()
