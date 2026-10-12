import os
import sys
from pathlib import Path
import pandas as pd
import sqlite3

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config

conn = sqlite3.connect(config.DATA_DIR / "market_data.db")
df_tqqq = pd.read_sql("SELECT datetime, open, high, low, close FROM market_candles WHERE symbol='SQQQ' AND timeframe='5m' AND datetime >= '2026-09-17' ORDER BY datetime", conn)
conn.close()

entry_times = ['2026-09-17 14:00:00', '2026-09-18 14:15:00']

for etime in entry_times:
    filtered = df_tqqq[df_tqqq['datetime'] >= etime]
    entry_row = filtered.iloc[0]
    entry_px = entry_row['close']
    post_candles = df_tqqq[df_tqqq['datetime'] > entry_row['datetime']].head(int(config.TIME_STOP_MINUTES/5))
    
    max_h = post_candles['high'].max()
    min_l = post_candles['low'].min()
    close_px = post_candles.iloc[-1]['close']
    
    print(f"\n[SQQQ at {entry_row['datetime']} (NYT)]")
    print(f"Entry Price: {entry_px}")
    print(f"Max High during {config.TIME_STOP_MINUTES}m: {max_h} (+{((max_h/entry_px)-1)*100:.2f}%)")
    print(f"Min Low during {config.TIME_STOP_MINUTES}m: {min_l} ({((min_l/entry_px)-1)*100:.2f}%)")
    print(f"{config.TIME_STOP_MINUTES}m Close Price: {close_px} ({((close_px/entry_px)-1)*100:.2f}%)")
