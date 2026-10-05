# -*- coding: utf-8 -*-
import os

with open('scripts/run_live_test.py', 'r', encoding='utf-8') as f:
    code = f.read()

target = '''                        moe_res = global_runner.moe_orchestrator.evaluate_dual_filter_signal(df_candle_15m=df_15m, live_prices=live_prices)'''

replacement = '''                        moe_res = global_runner.moe_orchestrator.evaluate_dual_filter_signal(df_candle_15m=df_15m, current_time_str=ct.strftime("%Y-%m-%d %H:%M:%S"), live_prices=live_prices)'''

code = code.replace(target, replacement)
with open('scripts/run_live_test.py', 'w', encoding='utf-8') as f:
    f.write(code)
print("Patched arguments!")
