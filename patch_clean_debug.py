import os
from pathlib import Path

fpath = Path('scripts/live_trade_test_clean.py')
content = fpath.read_text(encoding='utf-8')

replacement = """                df_15m = data_lake.load_candles("TQQQ", "15m")
                print(f"Data length: {len(df_15m)}")"""

idx1 = content.find('df_15m = data_lake.load_candles("TQQQ", "15m")')
if idx1 != -1:
    content = content[:idx1] + replacement + content[idx1 + len('df_15m = data_lake.load_candles("TQQQ", "15m")'):]
fpath.write_text(content, encoding='utf-8')
