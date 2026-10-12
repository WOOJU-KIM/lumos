import sys
import sqlite3
import config
import pandas as pd
from datetime import datetime

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

conn = sqlite3.connect('data/market_data.db')
df = pd.read_sql_query("SELECT datetime, open, high, low, close, volume FROM market_candles WHERE ticker=config.BASE_ASSET_SHORT AND timeframe='5m' ORDER BY datetime DESC LIMIT 100", conn)
conn.close()

df = df.iloc[::-1].reset_index(drop=True)
print("=== SQQQ 5m Price Trajectory on 2026-08-21 ===")
print(df.to_string())
