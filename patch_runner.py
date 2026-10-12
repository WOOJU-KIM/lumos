# -*- coding: utf-8 -*-
import os

with open('scripts/run_live_test.py', 'r', encoding='utf-8') as f:
    code = f.read()

target = '''        # 2. 15분 단위 진입 평가 (15분봉)
        if ct.minute % 15 == 0:'''

replacement = '''        # 2. 15분 단위 진입 평가 (15분봉)
        # 09:30 ~ 15:30 정규장 시간(Phase 1, 2)에만 신규 진입 허용. 
        # 그리고 포지션이 없을 때만 진입.
        is_market_open = (ct.hour > 9 or (ct.hour == 9 and ct.minute >= 30)) and ct.hour < 16
        is_entry_time = is_market_open and (ct.hour < 15 or (ct.hour == 15 and ct.minute <= 30))
        
        has_position = len(stk_bal.get("holdings", [])) > 0

        if ct.minute % 15 == 0 and is_entry_time and not has_position:'''

code = code.replace(target, replacement)

target2 = '''            if not global_runner._is_order_in_progress:'''
replacement2 = '''            if not global_runner._is_order_in_progress:'''
code = code.replace(target2, replacement2)

with open('scripts/run_live_test.py', 'w', encoding='utf-8') as f:
    f.write(code)
print("Patched market time and position check!")
