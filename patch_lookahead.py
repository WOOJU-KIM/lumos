# -*- coding: utf-8 -*-
import os
import datetime

with open('scripts/run_live_test.py', 'r', encoding='utf-8') as f:
    code = f.read()

target = '''                    df_15m = global_runner.data_lake.load_candles("TQQQ", "15m", end_dt=ct.strftime("%Y-%m-%d %H:%M:%S"))'''

replacement = '''                    
                    # Lookahead bias 방지: ct 시점에서는 ct 이전의 완성된 캔들만 존재함.
                    # 즉 09:30:00 에는 09:15:00 캔들까지만 DB에 존재해야 함.
                    # 따라서 ct에서 1초를 뺀 시간을 end_dt로 사용하여 미래 캔들 조회를 차단함.
                    safe_end_dt = (ct - datetime.timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")
                    df_15m = global_runner.data_lake.load_candles("TQQQ", "15m", end_dt=safe_end_dt)'''

if 'safe_end_dt' not in code:
    code = code.replace(target, replacement)
    with open('scripts/run_live_test.py', 'w', encoding='utf-8') as f:
        f.write(code)
    print("Patched lookahead bias!")
