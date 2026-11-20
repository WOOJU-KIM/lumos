"""Readiness is bound to the broker that is about to send the BUY."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.circuit_breaker import order_readiness, _reconciliation_proof_digest
from core.kiwoom_broker import KiwoomBroker


class TestAccountReadinessBinding(unittest.TestCase):
    def state(self, account='SYNTHETIC-01', environment='REAL'):
        proof=dict(verified_account=account, verified_environment=environment,
                   verified_through='20261019', sell_revision=0, open_orders_count=0,
                   unresolved_sell_count=0, history_order_count=0, today_order_count=0,
                   proof_digest=_reconciliation_proof_digest(account, environment, 0, '20261019', 0, 0))
        return dict(status='NORMAL', migration_state='READY', sell_reservations=[], sell_revision=0,
                    reconciliation_proof=proof)

    def test_current_account_and_environment_required(self):
        state=self.state()
        self.assertTrue(order_readiness(state, 'SYNTHETIC-01', 'REAL')['ready'])
        for account, environment in [('OTHER-01','REAL'),('SYNTHETIC-01','SIMULATION'),('', 'REAL')]:
            self.assertFalse(order_readiness(state,account,environment)['ready'])

    def test_real_broker_buy_rejected_before_quote_or_transport_on_scope_mismatch(self):
        broker=KiwoomBroker.__new__(KiwoomBroker)
        broker.account_no='SYNTHETIC';broker.account_type='01';broker.is_simulation=False
        for state in (self.state('OTHER-01'),self.state(environment='SIMULATION')):
            with patch('core.circuit_breaker.read_order_state',return_value=state), \
                 patch.object(broker,'get_stock_quote') as quote, \
                 patch.object(broker,'_send_tr_request') as transport:
                with self.assertRaises(RuntimeError):
                    broker.send_order('SOXL','BUY',1,100.)
                quote.assert_not_called();transport.assert_not_called()


if __name__ == '__main__':
    unittest.main()
