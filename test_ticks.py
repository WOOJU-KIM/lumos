import sqlite3
import pandas as pd
conn = sqlite3.connect('data/market_data.db')
df = pd.read_sql("SELECT count(*) as c FROM market_candles WHERE symbol='TQQQ' AND date(datetime)='2026-09-21'", conn)
print(f"Ticks for TQQQ: {df.iloc[0]['c']}")
