"""Fixed stop boundaries, stale live metadata, and shadow policy drift protection."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'research/shadow_five_full15m_20261018'))
import config
import common
from core.live_runner import KiwoomLiveRunner


class TestFixedStopPolicy(unittest.TestCase):
    def runner(self):
        runner = KiwoomLiveRunner.__new__(KiwoomLiveRunner)
        runner._is_order_in_progress = False
        runner._get_active_position = MagicMock(return_value=dict(
            symbol='SOXL', quantity=1, price=100., sl_pct=3.2,
            high_price_during_hold=100., low_price_during_hold=100.))
        runner._save_active_position = MagicMock()
        runner._dispatch_sell = MagicMock()
        runner.shadow_tracker = MagicMock()
        return runner

    def test_live_websocket_ignores_stale_wider_stop(self):
        runner = self.runner()
        with patch('core.live_runner.USMarketCalendar.get_market_status', return_value={'is_open':True}):
            runner._on_websocket_tick('SOXL',98.01,{})
            runner._dispatch_sell.assert_not_called()
            runner._on_websocket_tick('SOXL',97.99,{})
        runner._dispatch_sell.assert_called_once()
        self.assertTrue(runner._dispatch_sell.call_args.kwargs['is_stoploss'])

    def test_live_poll_ignores_stale_wider_stop(self):
        runner = self.runner()
        balance = {'holdings':[dict(symbol='SOXL',quantity=1,purchase_price=100.)]}
        runner._manage_open_positions(balance, {'SOXL':98.01})
        runner._dispatch_sell.assert_not_called()
        runner._manage_open_positions(balance, {'SOXL':97.99})
        runner._dispatch_sell.assert_called_once()

    def test_approved_shadow_config_and_frozen_training_config(self):
        common.verify_source_policy()
        execution = common.load_execution_config()
        self.assertEqual(config.SL_MIN_PCT,config.SL_MAX_PCT)
        self.assertEqual(execution['SL_MIN_PCT'], execution['SL_MAX_PCT'])
        self.assertEqual(common.CONFIG['SL_MAX_PCT'],.032)

    def test_shadow_rejects_unrelated_config_even_with_updated_file_digest(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'data') as tmp:
            folder = Path(tmp)
            source = (ROOT/'config.py').read_bytes().replace(b'SL_MIN_PCT = 0.02',b'SL_MIN_PCT = 0.03')
            (folder/'config.py').write_bytes(source)
            (folder/'policy.json').write_bytes((common.HOME/'policy.json').read_bytes())
            approval=json.loads((common.HOME/'execution_policy.json').read_text())
            approval['approved_config_sha256']=common.sha(folder/'config.py')
            (folder/'execution_policy.json').write_text(json.dumps(approval))
            with patch.object(common,'PROJECT',folder), patch.object(common,'HOME',folder):
                with self.assertRaisesRegex(RuntimeError,'extend beyond'):
                    common.load_execution_config()


if __name__ == '__main__':
    unittest.main()
