# -*- coding: utf-8 -*-
import os

with open('scripts/run_live_test.py', 'r', encoding='utf-8') as f:
    code = f.read()

target = '''        mock_kst = ct.replace(tzinfo=ZoneInfo("Asia/Seoul"))
        ct = mock_kst.astimezone(ZoneInfo("America/New_York"))
        feeder.current_mock_time = ct'''

replacement = target + '''
        
        # ?제 최신 가격 업데이트
        for sym in feeder.subscribed_symbols:
            df_sym = feeder.mock_data.get(sym)
            if df_sym is not None and not df_sym.empty:
                row = df_sym[df_sym['dt_obj'] == current_time_ts]
                if not row.empty:
                    feeder._latest_prices[sym] = float(row.iloc[0]['Close'])'''

if '_latest_prices[sym]' not in code:
    code = code.replace(target, replacement)
    with open('scripts/run_live_test.py', 'w', encoding='utf-8') as f:
        f.write(code)
    print("Patched prices logic!")
