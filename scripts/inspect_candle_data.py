import sqlite3
import config
import pandas as pd

conn = sqlite3.connect('data/market_data.db')
df_15m = pd.read_sql_query("SELECT symbol, timeframe, COUNT(*) as cnt, MIN(datetime) as min_dt, MAX(datetime) as max_dt FROM market_candles GROUP BY symbol, timeframe ORDER BY symbol, timeframe", conn)
print(df_15m.to_string())

df_long = pd.read_sql_query("SELECT datetime, open, high, low, close, volume FROM market_candles WHERE symbol=config.BASE_ASSET_LONG AND timeframe='15m' ORDER BY datetime", conn)
print('TQQQ 15m rows:', len(df_long))
df_long['date'] = df_long['datetime'].str.slice(0, 10)
unique_dates = df_long['date'].unique()
print('Total unique trading days:', len(unique_dates))
print('First 5 days:', unique_dates[:5])
print('Last 5 days:', unique_dates[-5:])
