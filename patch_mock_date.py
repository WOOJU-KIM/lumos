import os
from pathlib import Path

fpath = Path('core/mock_kiwoom_feeder.py')
content = fpath.read_text(encoding='utf-8')

target = 'self.target_date = target_date or datetime.now().strftime("%Y-%m-%d")'
replacement = """import sqlite3
        if target_date:
            self.target_date = target_date
        else:
            conn = sqlite3.connect("data/market_data.db")
            import pandas as pd
            df_date = pd.read_sql("SELECT date(datetime) as dt FROM market_candles WHERE symbol='TQQQ' ORDER BY datetime DESC LIMIT 1", conn)
            self.target_date = df_date.iloc[0]['dt']
            conn.close()"""

content = content.replace(target, replacement)
fpath.write_text(content, encoding='utf-8')
