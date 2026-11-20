"""Opening and delayed bar contracts; broker, clock and collection are mocked."""
import sys
import unittest
import threading
import os
import json
from pathlib import Path
from datetime import timedelta
from unittest.mock import MagicMock, patch
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import test_opposite_relay as relay_fixture
from core.telegram_notifier import TelegramNotifier
from core.data_lake import MarketDataLake


class ClockStop:
    def __init__(self, callback=None):
        self.elapsed = 0.
        self.stopped = False
        self.waits = []
        self.callback = callback

    def is_set(self):
        return self.stopped

    def wait(self, seconds):
        self.waits.append(seconds)
        self.elapsed += seconds
        if self.callback:
            self.callback()
        return self.stopped


class TestCandleReadiness(unittest.TestCase):
    def setUp(self):
        self.fixture = relay_fixture.TestRelay()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.r = self.fixture.r
        self.ready = pd.DataFrame({'Close': [100.] * 100}, index=pd.date_range(
            end=self.fixture.now - timedelta(minutes=config.RELAY_DECISION_INTERVAL_MINUTES),
            periods=100, freq='15min'))
        self.stale = self.ready.iloc[:-1].copy()
        self.r.data_lake = MagicMock()
        self.r.data_lake.load_candles.return_value = self.stale
        self.clock = ClockStop()
        self.r._stop = self.clock
        timer = patch('core.relay_live_runner.time.monotonic', lambda: self.clock.elapsed)
        timer.start()
        self.addCleanup(timer.stop)
        logs = patch('core.relay_live_runner.system_logger.log')
        logs.start()
        self.addCleanup(logs.stop)

    def load(self, sync=True):
        return self.r._load_decision_candles(self.fixture.now, True, sync)

    def test_open_waits_even_if_premarket_bar_is_present(self):
        self.fixture.now = self.fixture.now.replace(hour=9, minute=30)
        self.load()
        self.assertEqual(self.r._candle_error, 'FIRST_15M_CANDLE_NOT_CLOSED')
        self.assertIsNone(self.r._decision)
        self.r.data_lake.sync_live_intraday_candles.assert_not_called()
        self.assertFalse(self.clock.waits)

    def test_first_regular_bar_is_used_at_0945(self):
        self.fixture.now = self.fixture.now.replace(hour=9, minute=45)
        self.r.data_lake.load_candles.return_value = self.ready
        result = self.load()
        self.assertEqual(result.index[-1].strftime('%H:%M'), '09:30')
        self.assertIsNone(self.r._candle_error)

    def test_late_bar_retries_then_succeeds(self):
        def collect(*args, **kwargs):
            if self.r.data_lake.sync_live_intraday_candles.call_count == 3:
                self.r.data_lake.load_candles.return_value = self.ready
        self.r.data_lake.sync_live_intraday_candles.side_effect = collect
        self.load()
        self.assertIsNone(self.r._candle_error)
        self.assertEqual(self.clock.waits, [1., 1.])
        self.assertEqual(self.r.data_lake.sync_live_intraday_candles.call_count, 3)

    def test_missing_bar_stops_at_30_seconds(self):
        self.load()
        self.assertEqual(self.clock.elapsed, 30.)
        self.assertEqual(self.clock.waits, [1.] * 30)
        self.assertIsNone(self.r._decision)
        self.assertEqual(self.r._candle_error, 'LATEST_COMPLETED_15M_CANDLE_MISSING')
        self.assertEqual(self.r.data_lake.sync_live_intraday_candles.call_count, 30)

    def test_collection_time_counts_towards_the_budget(self):
        self.r.data_lake.sync_live_intraday_candles.side_effect = lambda *a, **k: setattr(
            self.clock, 'elapsed', min(30., self.clock.elapsed + 2.))
        self.load()
        self.assertEqual(self.clock.elapsed, 30.)
        self.assertIsNone(self.r._decision)
        self.assertEqual(self.r.data_lake.sync_live_intraday_candles.call_count, 10)

    def test_shutdown_interrupts_wait(self):
        self.clock.callback = lambda: setattr(self.clock, 'stopped', True)
        self.load()
        self.assertEqual(self.clock.elapsed, 1.)
        self.assertEqual(self.r._candle_error, 'CANDLE_WAIT_STOPPED')
        self.assertIsNone(self.r._decision)

    def test_decision_cannot_cross_into_next_cycle(self):
        self.clock.callback = lambda: setattr(self.fixture, 'now', self.fixture.now + timedelta(minutes=15))
        self.load()
        self.assertEqual(self.r._candle_error, 'STALE_15M_DECISION')
        self.assertIsNone(self.r._decision)
        self.assertEqual(self.r.data_lake.sync_live_intraday_candles.call_count, 1)

    def test_offline_evaluation_does_not_fetch_or_wait(self):
        self.load(sync=False)
        self.r.data_lake.sync_live_intraday_candles.assert_not_called()
        self.assertFalse(self.clock.waits)

    def test_invalid_sequence_does_not_retry_until_timeout(self):
        self.r.data_lake.load_candles.return_value = pd.concat([self.ready, self.ready.iloc[-1:]])
        self.load()
        self.assertEqual(self.r._candle_error, 'INVALID_15M_CANDLE_SEQUENCE')
        self.assertFalse(self.clock.waits)

    def test_ws_stop_can_close_position_while_candle_is_waiting(self):
        long = config.BASE_ASSET_LONG
        self.assertTrue(self.fixture.buy(long))
        def risk_worker():
            if long in self.r._positions:
                self.fixture.quote(long, 95.)
                self.fixture.manage()
            self.r.data_lake.load_candles.return_value = self.ready
        def concurrent_risk():
            worker = threading.Thread(target=risk_worker)
            worker.start()
            worker.join(timeout=3.)
            self.assertFalse(worker.is_alive(), 'Candle wait blocked the risk worker')
        self.clock.callback = concurrent_risk
        self.load()
        self.assertNotIn(long, self.r._positions)
        self.assertTrue(any(o['order_type'] == 'SELL' for o in self.fixture.raw.sent))
        self.assertIsNone(self.r._candle_error)

    def test_opening_briefing_has_no_fake_probabilities_or_alarm(self):
        msg = TelegramNotifier().compose_signal_briefing_alert(
            {'direction': 'NONE'}, '09:30', '22:30', data_stale=True,
            stale_reason='FIRST_15M_CANDLE_NOT_CLOSED')
        self.assertIn('첫 봉 완료 대기', msg)
        self.assertIn('산출 전', msg)
        self.assertNotIn('100.0%', msg)
        self.assertNotIn('🚨', msg)

    def test_opening_evaluator_skips_model_and_order(self):
        self.fixture.evaluate(approved=False, direction='NONE')
        self.r.moe_orchestrator.evaluate_dual_filter_signal.reset_mock()
        self.r._collect_cycle_market_snapshot = relay_fixture.RelayLiveRunner._collect_cycle_market_snapshot.__get__(self.r)
        self.fixture.now = self.fixture.now.replace(hour=9, minute=30)
        result = self.r.evaluate_15m_cycle_once(sync_intraday=False, now_dt=self.fixture.now,
                                               market_session='REGULAR_MARKET_OPEN')
        self.r.moe_orchestrator.evaluate_dual_filter_signal.assert_not_called()
        self.assertEqual(result['order_block_reason'], 'FIRST_15M_CANDLE_NOT_CLOSED')
        self.assertFalse(result['order_attempted'])
        self.assertFalse(self.fixture.raw.sent)

    def test_retry_runs_model_and_shadow_candle_increment_only_once(self):
        self.fixture.evaluate(approved=False, direction='NONE')
        self.r.moe_orchestrator.evaluate_dual_filter_signal.reset_mock()
        self.r.shadow_tracker.increment_candle.reset_mock()
        self.r.data_lake.load_candles.return_value = self.stale
        def collect(*args, **kwargs):
            if self.r.data_lake.sync_live_intraday_candles.call_count == 3:
                self.r.data_lake.load_candles.return_value = self.ready
        self.r.data_lake.sync_live_intraday_candles.side_effect = collect
        self.r.evaluate_15m_cycle_once(sync_intraday=True, now_dt=self.fixture.now,
                                      market_session='REGULAR_MARKET_OPEN')
        self.r.moe_orchestrator.evaluate_dual_filter_signal.assert_called_once()
        self.r.shadow_tracker.increment_candle.assert_called_once()
        self.assertFalse(self.fixture.raw.sent)

    def test_expired_collection_deadline_makes_no_request(self):
        lake = MarketDataLake.__new__(MarketDataLake)
        with patch('urllib.request.urlopen') as request:
            self.assertEqual(lake.harvest_symbol(config.BASE_ASSET_LONG, '15m', deadline=0.), 0)
        request.assert_not_called()

    def test_api_timeout_is_capped_by_remaining_budget(self):
        lake = MarketDataLake.__new__(MarketDataLake)
        self.clock.elapsed = 29.5
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"bars": {}}'
        with patch.dict(os.environ, ALPACA_API_KEY='FIXTURE', ALPACA_SECRET_KEY='FIXTURE'), \
                patch('urllib.request.urlopen', return_value=response) as request:
            self.assertEqual(lake.harvest_symbol(config.BASE_ASSET_LONG, '15m', deadline=30.), 0)
        self.assertEqual(request.call_args.kwargs['timeout'], .5)

    def test_data_arriving_after_deadline_is_not_saved(self):
        lake = MarketDataLake.__new__(MarketDataLake)
        lake.insert_candles = MagicMock()
        response = MagicMock()
        def delayed_response():
            self.clock.elapsed = 30.
            return json.dumps({'bars': {config.BASE_ASSET_LONG: [dict(
                t='2026-10-20T13:30:00Z', o=100, h=101, l=99, c=100, v=1)]}}).encode()
        response.__enter__.return_value.read.side_effect = delayed_response
        with patch.dict(os.environ, ALPACA_API_KEY='FIXTURE', ALPACA_SECRET_KEY='FIXTURE'), \
                patch('urllib.request.urlopen', return_value=response):
            self.assertEqual(lake.harvest_symbol(config.BASE_ASSET_LONG, '15m', deadline=30.), 0)
        lake.insert_candles.assert_not_called()

    def test_timeout_evaluator_does_not_call_model_or_order(self):
        self.fixture.evaluate(approved=False, direction='NONE')
        self.r.moe_orchestrator.evaluate_dual_filter_signal.reset_mock()
        self.r._collect_cycle_market_snapshot = relay_fixture.RelayLiveRunner._collect_cycle_market_snapshot.__get__(self.r)
        self.r.data_lake.load_candles.return_value = self.stale
        result = self.r.evaluate_15m_cycle_once(sync_intraday=True, now_dt=self.fixture.now,
                                               market_session='REGULAR_MARKET_OPEN')
        self.assertEqual(self.clock.elapsed, 30.)
        self.r.moe_orchestrator.evaluate_dual_filter_signal.assert_not_called()
        self.assertFalse(result['order_attempted'])
        self.assertEqual(result['order_block_reason'], 'DATA_STALE')

    def test_deployment_verifies_candle_collection_source(self):
        self.assertTrue(self.r._verify_deployment())


if __name__ == '__main__':
    unittest.main()
