"""Absolute-feature deployment and subsequent automatic retraining contracts."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import config
from core.ml_engine import MLFeatureEngine
from core.moe_orchestrator import MoEMetaOrchestrator, train_and_save_moe_orchestrator
from core.data_lake import MarketDataLake

ROOT=Path(__file__).resolve().parents[1]

class TestAbsoluteLiveModel(unittest.TestCase):
    def test_active_manifest_and_model_use_exact_legacy_eight(self):
        e=MLFeatureEngine();m=json.loads((ROOT/'models/active_model_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(m['expected_schema'],'v4_champion')
        self.assertEqual(e.active_model_sha256,m['expected_sha256'])
        self.assertEqual(e.feature_names,e.LEGACY_PKL_CHAMPION_72_FEATURES)
        self.assertTrue(set(e.DOLLAR_SCALE_FEATURES)<=set(e.feature_names))
        self.assertFalse(set(e.NORMALIZED_PRICE_FEATURES)&set(e.feature_names))
        self.assertEqual(e.model.get_params()['feature_contri'],[float(config.GBDT_FEATURE_GAIN_WEIGHTS.get(n,1.)) for n in e.feature_names])
        info=e.get_active_model_info(is_daemon_running=True)
        self.assertFalse(info['is_fallback']);self.assertEqual(info['fallback_status'],'PRIMARY_ACTIVE')
        self.assertEqual(info['operational_verdict'],'ACTIVE_V4_CHAMPION')
        self.assertTrue(info['is_manifest_memory_consistent'])

    def test_feature_order_independent_of_input_frame_column_order(self):
        names=MLFeatureEngine.LEGACY_PKL_CHAMPION_72_FEATURES
        f=pd.DataFrame([[1.]*72],columns=list(reversed(names)))
        self.assertEqual(MLFeatureEngine.get_feature_columns(f,'v4_champion'),names)

    def test_execution_config_unchanged(self):
        self.assertEqual(config.RELAY_INITIAL_ALLOCATION,.98)
        self.assertEqual(config.RELAY_CHAIN_ALLOCATION,.0)
        self.assertEqual(config.RELAY_OPPOSITE_TRIGGER_PCT,.015)
        self.assertEqual(config.SL_MIN_PCT,.02);self.assertEqual(config.SL_MAX_PCT,.02)
        self.assertEqual(config.TRAILING_TRIGGER_PCT,.01)
        self.assertEqual(config.RELAY_CHAIN_TRAILING_TRIGGER_PCT,.01)
        self.assertEqual(config.TRAILING_DROP_PCT,.003)

    def test_default_retraining_preserves_absolute_schema_and_prior_model(self):
        raw=MarketDataLake().load_candles(config.BASE_ASSET_LONG,'15m')
        # Full stored history, bounded and purged by the production training method.
        current=json.loads((ROOT/'models/active_model_manifest.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory(dir=ROOT/'data') as folder:
            dest=Path(folder);manifest=dest/'active_model_manifest.json'
            copied=dict(current)
            copied['model_path']=copied['active_model_path']=str(ROOT/current['model_path'])
            manifest.write_text(json.dumps(copied),encoding='utf-8')
            trained=train_and_save_moe_orchestrator(df_15m=raw,models_dir_override=dest,live_experience_df=pd.DataFrame())
            next_manifest=json.loads(manifest.read_text(encoding='utf-8'))
            self.assertEqual(next_manifest['expected_schema'],'v4_champion')
            self.assertEqual(next_manifest['active_model_id'],'Absolute_72')
            self.assertEqual(next_manifest['feature_names'],MLFeatureEngine.LEGACY_PKL_CHAMPION_72_FEATURES)
            self.assertEqual(next_manifest['previous_model_sha256'],current['expected_sha256'])
            e=MLFeatureEngine(manifest_path=str(manifest))
            self.assertEqual(e.active_model_sha256,next_manifest['expected_sha256'])
            self.assertEqual(e.model_fallback_status,'PRIMARY_ACTIVE')
            self.assertEqual(trained.gbdt_engine.feature_schema,'v4_champion')

    def test_invalid_schema_cannot_replace_selected_model(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'data') as folder:
            with self.assertRaises(ValueError):
                train_and_save_moe_orchestrator(schema='E_raw_all',models_dir_override=Path(folder))
            self.assertFalse((Path(folder)/'active_model_manifest.json').exists())

if __name__=='__main__':unittest.main()
