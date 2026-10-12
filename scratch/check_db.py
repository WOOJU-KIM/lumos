import sqlite3
import pandas as pd
conn = sqlite3.connect('data/market_data.sqlite')
print(pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn))

try:
    print(pd.read_sql_query("SELECT symbol, timeframe, MIN(datetime) as start, MAX(datetime) as end, COUNT(*) as count FROM market_data GROUP BY symbol, timeframe", conn).to_markdown())
except:
    pass

try:
    print(pd.read_sql_query("SELECT symbol, timeframe, MIN(datetime) as start, MAX(datetime) as end, COUNT(*) as count FROM ohlcv GROUP BY symbol, timeframe", conn).to_markdown())
except:
    pass
