import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re

with open('tests/test_system_integrity.py', 'r', encoding='utf-8') as f:
    content = f.read()

if 'sys.path.insert' not in content:
    content = "import sys\nimport os\nsys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))\n" + content

# Fix test_8
old_test_8 = '''        moe_v3.gbdt_engine.model.predict_proba.return_value = [[0.30, 0.25, 0.45]]
        with patch.object(moe_v3.gbdt_engine, 'extract_features', return_value=pd.DataFrame({"dummy": [0]})):
            res = moe_v3.evaluate_dual_filter_signal(self.dummy_15m)
            self.assertEqual(res["direction"], "NONE")
            self.assertFalse(res["is_approved"])'''

new_test_8 = '''        moe_v3.gbdt_engine.model.predict_proba.return_value = np.array([[0.30, 0.45, 0.25]])
        with patch.object(moe_v3.gbdt_engine, 'extract_features', return_value=pd.DataFrame({"dummy": [0]})):
            res = moe_v3.evaluate_dual_filter_signal(self.dummy_15m)
            self.assertEqual(res["direction"], "NONE")
            self.assertFalse(res["is_approved"])
            
            # 반대 경우 (Long이 가장 높을 때)
            moe_v3.gbdt_engine.model.predict_proba.return_value = np.array([[0.25, 0.30, 0.45]])
            res_long = moe_v3.evaluate_dual_filter_signal(self.dummy_15m)
            self.assertEqual(res_long["direction"], "LONG")'''

content = content.replace(old_test_8, new_test_8)

with open('tests/test_system_integrity.py', 'w', encoding='utf-8') as f:
    f.write(content)
