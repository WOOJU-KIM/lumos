# -*- coding: utf-8 -*-
import os

with open('scripts/run_live_test.py', 'r', encoding='utf-8') as f:
    code = f.read()

target = '''                    is_appr = moe_res.get("is_approved", False)
                    direction = moe_res.get("direction", "NONE")'''

replacement = target + '''
                    print(f"[{mock_kst.strftime('%H:%M')}] Eval Result: {is_appr} / {direction}")'''

if 'Eval Result' not in code:
    code = code.replace(target, replacement)
    with open('scripts/run_live_test.py', 'w', encoding='utf-8') as f:
        f.write(code)
    print("Patched print logic!")
