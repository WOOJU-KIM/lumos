"""Preserved pre-relay model stays shadow-only and honors its original recipe."""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np

HOME=Path(__file__).resolve().parents[1]/'research/shadow_five_full15m_20261018'
sys.path.insert(0,str(HOME))
from common import load_execution_config, verify_source_policy
from legacy_b72 import LegacyB72
from runtime import Engine


class TestLegacyB72(unittest.TestCase):
    def test_approved_config_and_actual_artifact_load(self):
        verify_source_policy();legacy=LegacyB72()
        self.assertEqual(legacy.settings['allocation'],.98)
        self.assertEqual(legacy.settings['trailing_trigger'],.015)
        self.assertEqual(load_execution_config()['RELAY_INITIAL_ALLOCATION'],.98)
        self.assertEqual(len(legacy.features),72)

    def test_neutral_winner_blocks_legacy_shadow_entry(self):
        legacy=LegacyB72();legacy.model=MagicMock()
        legacy.model.predict_proba.return_value=np.array([[.2,.6,.2]])
        result=legacy.predict(dict(full_features={name:0. for name in legacy.features}))
        self.assertEqual(result['side'],-1)

    def test_legacy_has_only_its_single_primary_ledger(self):
        legacy=LegacyB72();engine=Engine({legacy.name:legacy})
        self.assertEqual(set(engine.ledgers),{(legacy.name,'primary')})

    def test_missing_feature_is_not_silently_zero_filled(self):
        legacy=LegacyB72()
        with self.assertRaises(KeyError):legacy.predict(dict(full_features={}))


if __name__=='__main__':unittest.main()
