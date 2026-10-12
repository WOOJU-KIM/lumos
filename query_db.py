import sqlite3
import pandas as pd
conn = sqlite3.connect('data/market_data.db')
query = "SELECT datetime, close FROM market_candles WHERE symbol='TQQQ' AND timeframe='5m' AND datetime LIKE '2026-10-05%' ORDER BY datetime"
print(pd.read_sql(query, conn).to_string())
