import re

with open('tests/test_system_integrity.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix test 6
old_test_6 = '''        with patch.object(moe_v3.gbdt_engine, 'predict_signal', return_value=(1, config.GBDT_CONFIDENCE_THRESHOLD + 0.08, {})):
            with patch.object(moe_v3.cross_asset_model, 'predict_signal', return_value=(1, 0.70, "")):
                with patch.object(moe_v3.gbdt_engine, 'extract_features', return_value=pd.DataFrame({"RSI_14": [50.0]})):
                    with patch.object(moe_v3, '_check_60m_trend_filter', return_value=(True, "OK")):
                        res = moe_v3.evaluate_dual_filter_signal(self.dummy_15m)
                        self.assertEqual(res["direction"], "LONG")'''

new_test_6 = '''        with patch.object(moe_v3.gbdt_engine, 'predict_signal', return_value=(1, config.GBDT_CONFIDENCE_THRESHOLD + 0.08, "reason")):
            with patch.object(moe_v3.cross_asset_model, 'predict_signal', return_value=(1, 0.70, "reason")):
                with patch.object(moe_v3.gbdt_engine, 'extract_features', return_value=pd.DataFrame({"RSI_14": [50.0]})):
                    with patch("config.USE_60M_TREND_FILTER", False):
                        res = moe_v3.evaluate_dual_filter_signal(self.dummy_15m)
                        self.assertEqual(res["direction"], "LONG")'''

content = content.replace(old_test_6, new_test_6)

# Fix test 8
old_test_8 = '''        moe_v3.gbdt_engine.model = MagicMock()
        moe_v3.gbdt_engine.model.predict_proba.return_value = np.array([[0.30, 0.25, 0.45]])
        with patch.object(moe_v3.gbdt_engine, 'extract_features', return_value=pd.DataFrame({"dummy": [0]})):
            res = moe_v3.evaluate_dual_filter_signal(self.dummy_15m)
            self.assertEqual(res["direction"], "NONE")
            self.assertFalse(res["is_approved"])'''

new_test_8 = '''        moe_v3.gbdt_engine.model = MagicMock()
        moe_v3.gbdt_engine.scaler = MagicMock()
        moe_v3.gbdt_engine.scaler.transform.return_value = [[0]]
        moe_v3.gbdt_engine.model.predict_proba.return_value = [[0.30, 0.25, 0.45]]
        with patch.object(moe_v3.gbdt_engine, 'extract_features', return_value=pd.DataFrame({"dummy": [0]})):
            res = moe_v3.evaluate_dual_filter_signal(self.dummy_15m)
            self.assertEqual(res["direction"], "NONE")
            self.assertFalse(res["is_approved"])'''

content = content.replace(old_test_8, new_test_8)

with open('tests/test_system_integrity.py', 'w', encoding='utf-8') as f:
    f.write(content)
