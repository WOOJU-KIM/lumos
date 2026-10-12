import sqlite3
import pandas as pd
conn = sqlite3.connect('data/market_data.db')
df = pd.read_sql("SELECT datetime, close FROM market_candles WHERE symbol='SQQQ' AND timeframe='5m' ORDER BY datetime DESC LIMIT 20", conn)
print(df)
conn.close()
