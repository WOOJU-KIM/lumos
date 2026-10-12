import re
import pandas as pd

with open('core/weekly_tournament.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace Screen 1: 60분봉 추세
old_60m = """                # [조건 1-3] Screen 1: 60분봉 추세
                past_macro_60 = macro_60m[macro_60m['datetime'] <= cur_time]
                past_long_60 = long_60m[long_60m['datetime'] <= cur_time]"""

new_60m = """                # [조건 1-3] Screen 1: 60분봉 추세 (미래참조 방지: 15m 캔들 종료 시점 이전에 종료된 60m 캔들만 사용)
                import pandas as pd
                cur_end_time = pd.to_datetime(cur_time) + pd.Timedelta(minutes=15)
                past_macro_60 = macro_60m[(macro_60m['datetime_dt'] + pd.Timedelta(minutes=60)) <= cur_end_time]
                past_long_60 = long_60m[(long_60m['datetime_dt'] + pd.Timedelta(minutes=60)) <= cur_end_time]"""

content = content.replace(old_60m, new_60m)

with open('core/weekly_tournament.py', 'w', encoding='utf-8') as f:
    f.write(content)
