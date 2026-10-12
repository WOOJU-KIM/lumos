# -*- coding: utf-8 -*-
import os

with open('scripts/run_live_test.py', 'r', encoding='utf-8') as f:
    code = f.read()

target = '''safe_end_dt = (ct - datetime.timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")'''
replacement = '''from datetime import timedelta\n                    safe_end_dt = (ct - timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")'''

code = code.replace(target, replacement)
with open('scripts/run_live_test.py', 'w', encoding='utf-8') as f:
    f.write(code)
print("Patched timedelta!")
