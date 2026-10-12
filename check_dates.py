import sqlite3, pandas as pd
conn = sqlite3.connect('data/market_data.db')
df = pd.read_sql("SELECT DISTINCT date(datetime) as dt FROM market_candles WHERE symbol='TQQQ' ORDER BY datetime DESC LIMIT 5", conn)
print(df['dt'].tolist())
conn.close()
