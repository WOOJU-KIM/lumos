import sqlite3, pandas as pd
conn = sqlite3.connect('data/market_data.db')
df = pd.read_sql("SELECT count(*) as cnt, min(datetime), max(datetime) FROM market_candles WHERE date(datetime) = '2026-09-21'", conn)
print(df)
conn.close()
