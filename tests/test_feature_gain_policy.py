"""Execute the production gain-policy block without market or broker IO."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import config
from core.ml_engine import MLFeatureEngine

class FeatureGainPolicyTests(unittest.TestCase):
    def execute(self, overrides, schema='B_normalized'):
        source=ast.parse((Path(__file__).resolve().parents[1]/'core/ml_engine.py').read_text(encoding='utf-8-sig'))
        cl=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='MLFeatureEngine')
        method=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name=='train_and_select_top_features')
        start=next(i for i,n in enumerate(method.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='gain_overrides' for t in n.targets))
        end=next(i for i in range(start,len(method.body)) if isinstance(method.body[i],ast.Try))
        scope=dict(np=np,config=SimpleNamespace(GBDT_FEATURE_GAIN_WEIGHTS=overrides),target_schema=schema,feature_cols=MLFeatureEngine.CANONICAL_B_NORMALIZED_72_FEATURES,self=SimpleNamespace(last_training_window_meta={}))
        exec(compile(ast.fix_missing_locations(ast.Module(body=method.body[start:end],type_ignores=[])),'PRODUCTION_GAIN_POLICY','exec'),scope)
        return scope
    def test_two_gain_weights_keep_seventy_columns_at_one(self):
        s=self.execute(config.GBDT_FEATURE_GAIN_WEIGHTS)
        self.assertEqual(s['gain_weights'].count(.3),2)
        self.assertEqual(s['gain_weights'].count(1.),70)
        self.assertEqual(s['self'].last_training_window_meta['feature_gain_weights'],config.GBDT_FEATURE_GAIN_WEIGHTS)
    def test_unknown_feature_fails_before_fit(self):
        with self.assertRaises(ValueError):self.execute({'TYPO':.5})
    def test_invalid_weight_fails_before_fit(self):
        for v in [-1,float('nan'),float('inf')]:
            with self.subTest(weight=v),self.assertRaises(ValueError):self.execute({'EMA_200_Diff':v})
    def test_other_schemas_keep_original_learning_policy(self):
        s=self.execute(config.GBDT_FEATURE_GAIN_WEIGHTS,'v5_challenger')
        self.assertTrue(all(v==1. for v in s['gain_weights']))

if __name__=='__main__':unittest.main()
