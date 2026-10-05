# -*- coding: utf-8 -*-
import os

with open('scripts/run_live_test.py', 'r', encoding='utf-8') as f:
    code = f.read()

target = '''                            from core import config'''
replacement = '''                            import config'''

code = code.replace(target, replacement)
with open('scripts/run_live_test.py', 'w', encoding='utf-8') as f:
    f.write(code)
print("Patched config import!")
