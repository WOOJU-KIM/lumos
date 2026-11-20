"""Reproduce Windows position replacement failure with synthetic ledger only."""
from pathlib import Path
import copy
import json
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tests'))
import config
import test_opposite_relay as fixtures
from core.relay_policy import atomic_json


class TestPositionWriteRecovery(unittest.TestCase):
    def setUp(self):
        self.case = fixtures.TestRelay(methodName='test_initial_49_and_chain_49_real_helpers_and_roles')
        self.case.setUp(); self.addCleanup(self.case.doCleanups)
        self.r, self.raw = self.case.r, self.case.raw
        self.long, self.short = config.BASE_ASSET_LONG, config.BASE_ASSET_SHORT
        self.assertTrue(self.case.buy(self.long))

    def fail_write(self):
        position = copy.deepcopy(self.r._positions[self.long])
        with self.r._symbol(self.long), patch('core.relay_live_runner.atomic_json', side_effect=PermissionError('WinError 5')):
            with self.assertRaises(PermissionError): self.r._save_active_position(position)
        self.assertIn(self.long, self.r._position_write_failures)
        self.assertFalse(self.r._journal_failed)

    def reconcile(self):
        self.r._reconcile_active_position_with_ledger(self.raw.get_overseas_stock_balance())

    def test_windows_replace_failure_then_reconcile_allows_exactly_one_chain(self):
        import core.relay_policy as policy
        replace = policy.os.replace
        def fail_position(source, target):
            if Path(target).name == self.long + '.json': raise PermissionError('WinError 5')
            return replace(source, target)
        self.case.quote(self.long, 98.5)
        with patch.object(policy.os, 'replace', side_effect=fail_position): self.case.manage()
        self.assertEqual(len(self.raw.sent), 1)
        self.assertFalse(list(self.r._root.glob('*.tmp')))
        self.reconcile(); self.case.manage(); self.case.manage()
        self.assertFalse(self.r._position_write_failures)
        self.assertEqual(len(self.raw.sent), 2)
        self.assertEqual(self.raw.sent[-1]['symbol'], self.short)
        self.assertEqual(self.r._positions[self.short]['relay_role'], 'CHAIN')

    def test_successful_price_write_alone_does_not_release_failure(self):
        self.fail_write()
        with self.r._symbol(self.long): self.r._save_active_position(self.r._positions[self.long])
        self.case.quote(self.long, 98.5); self.case.manage()
        self.assertTrue(self.r._position_write_failures)
        self.assertEqual(len(self.raw.sent), 1)

    def test_persistent_file_lock_keeps_buy_blocked(self):
        self.fail_write()
        with patch('core.relay_live_runner.atomic_json', side_effect=PermissionError('locked')):
            self.reconcile()
        self.case.quote(self.long, 98.5); self.case.manage()
        self.assertTrue(self.r._position_write_failures)
        self.assertEqual(len(self.raw.sent), 1)

    def test_position_lock_does_not_skip_stoploss_dispatch(self):
        self.case.quote(self.long, 97.)
        with patch('core.relay_live_runner.atomic_json', side_effect=PermissionError('locked')), \
                patch.object(self.r, '_dispatch_sell') as sell:
            self.case.manage()
        self.assertEqual(sell.call_count, 1)
        self.assertEqual(sell.call_args.kwargs['symbol'], self.long)
        self.assertEqual(sell.call_args.kwargs['reason_desc'], 'Relay SL')

    def test_persistent_position_lock_still_allows_real_sell_helper_and_flat_recovery(self):
        self.case.quote(self.long, 97.)
        with patch('core.relay_live_runner.atomic_json', side_effect=PermissionError('locked')):
            self.case.manage()
        self.assertNotIn(self.long, self.raw.holdings)
        self.assertEqual(self.raw.sent[-1]['order_type'], 'SELL')
        self.assertFalse(self.r._position_write_failures)

    def test_invalid_broker_balance_never_releases_failure(self):
        self.fail_write(); self.raw.fail = True
        with self.assertRaises(RuntimeError): self.reconcile()
        self.assertTrue(self.r._position_write_failures)

    def test_recovery_preserves_observed_high_and_low(self):
        self.fail_write()
        self.case.quote(self.long, 101.); self.case.quote(self.long, 98.5)
        self.reconcile()
        self.assertEqual(self.r._positions[self.long]['high_price_during_hold'], 101.)
        self.assertEqual(self.r._positions[self.long]['low_price_during_hold'], 98.5)
        disk = json.loads((self.r._root / (self.long + '.json')).read_text(encoding='utf-8'))
        self.assertEqual(disk['low_price_during_hold'], 98.5)

    def test_external_holding_change_prevents_recovery(self):
        self.fail_write(); qty, avg = self.raw.holdings[self.long]
        self.raw.holdings[self.long] = (qty - 1, avg)
        self.reconcile()
        self.assertTrue(self.r._position_write_failures)
        self.assertEqual(self.r._mismatch, 'EXTERNALLY_CHANGED_TRACKED_HOLDING')

    def test_unconfirmed_open_order_query_prevents_recovery(self):
        self.fail_write()
        with patch.object(self.raw, 'get_open_orders', return_value=dict(ok=False, orders=[])): self.reconcile()
        self.assertTrue(self.r._position_write_failures)

    def test_pending_order_failure_is_never_released_by_position_recovery(self):
        self.fail_write(); self.r._journal_failed = True
        self.reconcile()
        self.assertTrue(self.r._journal_failed)
        self.assertTrue(self.r._position_write_failures)

    def test_book_failure_is_never_released(self):
        self.fail_write(); self.r.book.failed = True
        self.reconcile()
        self.assertTrue(self.r.book.failed)
        self.assertTrue(self.r._position_write_failures)

    def test_changed_book_on_disk_prevents_recovery(self):
        self.fail_write(); changed = copy.deepcopy(self.r.book.data)
        changed['chain_operation_id'] = 'UNVERIFIED'
        atomic_json(self.r.book.path, changed)
        self.reconcile()
        self.assertTrue(self.r._position_write_failures)

    def test_changed_position_identity_on_disk_is_not_overwritten(self):
        self.fail_write(); path = self.r._root / (self.long + '.json')
        changed = json.loads(path.read_text(encoding='utf-8')); changed['relay_leg_id'] = 'OTHER'
        atomic_json(path, changed)
        self.reconcile()
        self.assertTrue(self.r._position_write_failures)
        self.assertEqual(json.loads(path.read_text(encoding='utf-8'))['relay_leg_id'], 'OTHER')

    def test_unknown_submission_prevents_recovery(self):
        self.fail_write()
        self.r.book.operation(self.short, dict(id='UNKNOWN', action='BUY', role='CHAIN',
            trigger_symbol=self.long, orders=[], submission_pending=True))
        self.reconcile()
        self.assertTrue(self.r._position_write_failures)
        self.assertEqual(self.r._mismatch, 'UNRESOLVED_SUBMISSION')


if __name__ == '__main__': unittest.main()
