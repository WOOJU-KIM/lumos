import re

with open('tests/test_system_integrity.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Test 5: Add patch for _save_active_position
content = content.replace(
'''            with patch.object(runner.broker, 'send_order', return_value={"order_no": "123"}):
                with patch.object(runner, '_wait_for_fill', return_value=(True, 10, 0)):
                    buy_res = runner._execute_buy_with_10s_chase(''',
'''            with patch.object(runner.broker, 'send_order', return_value={"order_no": "123"}):
                with patch.object(runner, '_wait_for_fill', return_value=(True, 10, 0)):
                    with patch.object(runner, '_save_active_position'):
                        buy_res = runner._execute_buy_with_10s_chase('''
)

# Test 6: Add _check_60m_trend_filter patch and fix assert
old_test_6 = '''        with patch.object(moe_v3.gbdt_engine, 'predict_signal', return_value=(1, config.GBDT_CONFIDENCE_THRESHOLD + 0.08, {})):
            with patch.object(moe_v3.cross_asset_model, 'predict_signal', return_value=(1, 0.70, "")):
                with patch.object(moe_v3.gbdt_engine, 'extract_features', return_value=pd.DataFrame({"RSI_14": [50.0]})):
                    res = moe_v3.evaluate_dual_filter_signal(self.dummy_15m)
                    self.assertEqual(res["direction"], "LONG" if getattr(config, "USE_60M_TREND_FILTER", False) is False else res["direction"])'''

new_test_6 = '''        with patch.object(moe_v3.gbdt_engine, 'predict_signal', return_value=(1, config.GBDT_CONFIDENCE_THRESHOLD + 0.08, {})):
            with patch.object(moe_v3.cross_asset_model, 'predict_signal', return_value=(1, 0.70, "")):
                with patch.object(moe_v3.gbdt_engine, 'extract_features', return_value=pd.DataFrame({"RSI_14": [50.0]})):
                    with patch.object(moe_v3, '_check_60m_trend_filter', return_value=(True, "OK")):
                        res = moe_v3.evaluate_dual_filter_signal(self.dummy_15m)
                        self.assertEqual(res["direction"], "LONG")'''
content = content.replace(old_test_6, new_test_6)

# Test 8: Mock predict_proba instead of predict_signal
old_test_8 = '''        with patch.object(moe_v3.gbdt_engine, 'predict_signal', return_value=(0, 0.45, {"class_probs": [0.30, 0.25, 0.45]})):
            res = moe_v3.evaluate_dual_filter_signal(self.dummy_15m)
            self.assertEqual(res["direction"], "NONE", "Neutral ?률??가???으므?NONE?어???니??")
            self.assertFalse(res["is_approved"], "Neutral???세?면 진입??차단?어???니??")'''

new_test_8 = '''        moe_v3.gbdt_engine.model = MagicMock()
        moe_v3.gbdt_engine.model.predict_proba.return_value = np.array([[0.30, 0.25, 0.45]])
        with patch.object(moe_v3.gbdt_engine, 'extract_features', return_value=pd.DataFrame({"dummy": [0]})):
            res = moe_v3.evaluate_dual_filter_signal(self.dummy_15m)
            self.assertEqual(res["direction"], "NONE")
            self.assertFalse(res["is_approved"])'''

# Due to possible encoding weirdness with ? characters, regex is safer.
pattern_8 = re.compile(
    r"with patch\.object\(moe_v3\.gbdt_engine, 'predict_signal', return_value=\(0, 0\.45, \{\"class_probs\": \[0\.30, 0\.25, 0\.45\]\}\)\):\n"
    r"\s+res = moe_v3\.evaluate_dual_filter_signal\(self\.dummy_15m\)\n"
    r"\s+self\.assertEqual\(res\[\"direction\"\], \"NONE\", [^\n]+\n"
    r"\s+self\.assertFalse\(res\[\"is_approved\"\], [^\n]+"
)
content = pattern_8.sub(new_test_8, content)

with open('tests/test_system_integrity.py', 'w', encoding='utf-8') as f:
    f.write(content)
