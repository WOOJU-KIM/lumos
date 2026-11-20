"""Real evaluator -> size gate -> Telegram reason, with synthetic broker only."""
from pathlib import Path
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
import config
import test_opposite_relay as fixture_module
from core.telegram_notifier import TelegramNotifier


class TestTelegramEntryReasons(unittest.TestCase):
    def setUp(self):
        self.fixture=fixture_module.TestRelay(methodName='test_initial_49_and_chain_49_real_helpers_and_roles')
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.r=self.fixture.r

    def text(self,result):
        notifier=TelegramNotifier.__new__(TelegramNotifier)
        return notifier.compose_signal_briefing_alert(
            {'direction':'LONG','gating_confidence':.7,'is_approved':True},'10:00','23:00',
            order_allowed=result['order_allowed'],order_executed=result['order_executed'],
            order_block_reason=result['order_block_reason'],order_block_detail=result['order_block_detail'])

    def test_same_bar_real_evaluator_reports_wait_not_insufficient_funds(self):
        self.r.book.update(last_decision=self.r._decision,flat_after=self.r._decision)
        result=self.fixture.evaluate()
        self.assertFalse(result['order_attempted']);self.assertFalse(result['order_executed'])
        self.assertEqual(result['order_block_detail'],'WAIT_NEXT_COMPLETED_15M')
        message=self.text(result)
        self.assertIn('동일 15분봉 재진입 차단',message)
        self.assertNotIn('INSUFFICIENT_FUNDS',message)
        self.assertFalse(self.fixture.raw.sent)

    def test_missing_opposite_quote_reports_ws_wait_not_cash(self):
        self.r._quotes[config.BASE_ASSET_SHORT]['time']=time.time()-config.RELAY_QUOTE_MAX_AGE_SECONDS-1
        result=self.fixture.evaluate()
        self.assertEqual(result['order_block_detail'],'FRESH_WS_QUOTES_REQUIRED')
        self.assertIn('WebSocket 호가 대기',self.text(result))
        self.assertNotIn('INSUFFICIENT_FUNDS',self.text(result))
        self.assertFalse(self.fixture.raw.sent)

    def test_actual_zero_affordable_quantity_retains_funds_reason(self):
        self.fixture.raw.cash=1.
        result=self.fixture.evaluate()
        self.assertEqual(result['order_block_detail'],'INSUFFICIENT_FUNDS')
        self.assertIn('매수 가능한 수량이 0주',self.text(result))
        self.assertFalse(self.fixture.raw.sent)

    def test_success_clears_old_diagnostic_without_changing_quantity(self):
        self.r._entry_block_detail='WAIT_NEXT_COMPLETED_15M'
        result=self.fixture.evaluate()
        self.assertTrue(result['order_executed'])
        self.assertEqual(result['order_qty'],48)
        self.assertIsNone(self.r._entry_block_detail)

    def test_missing_decision_and_safety_guard_have_separate_codes(self):
        decision=self.r._decision;self.r._decision=None
        self.assertEqual(self.r._entry_quantity(10000.,100.),0)
        self.assertEqual(self.r._entry_block_detail,'COMPLETED_15M_DECISION_UNAVAILABLE')
        self.r._decision=decision;self.r._stop.set()
        self.assertEqual(self.r._entry_quantity(10000.,100.),0)
        self.assertEqual(self.r._entry_block_detail,'RELAY_ENTRY_SAFETY_GUARD')

    def test_active_cycle_is_not_reported_as_funds_shortage(self):
        self.assertTrue(self.fixture.buy(config.BASE_ASSET_LONG))
        self.assertEqual(self.r._entry_quantity(self.fixture.raw.cash,100.),0)
        self.assertEqual(self.r._entry_block_detail,'RELAY_CYCLE_ACTIVE')


if __name__=='__main__':unittest.main()
