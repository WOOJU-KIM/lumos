import os
import sys
from pathlib import Path
import pandas as pd
import sqlite3

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config

conn = sqlite3.connect(config.DATA_DIR / "market_data.db")
df_sqqq = pd.read_sql("SELECT datetime, open, high, low, close FROM market_candles WHERE symbol='SQQQ' AND timeframe='5m' AND datetime >= '2026-09-17' ORDER BY datetime", conn)
conn.close()

entry_times = ['2026-09-17 14:00:00', '2026-09-18 14:15:00']

for etime in entry_times:
    filtered = df_sqqq[df_sqqq['datetime'] >= etime]
    if filtered.empty:
        continue
    entry_row = filtered.iloc[0]
    entry_px = entry_row['close']
    print(f"\n[Trade Simulation] Entry at {entry_row['datetime']} (NYT) - SQQQ Price: {entry_px}")
    
    tp_px = entry_px * (1.0 + config.MAX_TP_PCT)
    sl_px = entry_px * (1.0 - config.SL_MIN_PCT)
    print(f"   -> Target TP: {tp_px:.4f} (+{config.MAX_TP_PCT*100}%) | SL: {sl_px:.4f} (-{config.SL_MIN_PCT*100}%)")
    
    post_candles = df_sqqq[df_sqqq['datetime'] > entry_row['datetime']].head(18)
    
    exit_px = None
    exit_reason = None
    exit_time = None
    
    for _, row in post_candles.iterrows():
        b_h, b_l, b_o, b_c = row['high'], row['low'], row['open'], row['close']
        hit_tp = b_h >= tp_px
        hit_sl = b_l <= sl_px
        
        if hit_tp and hit_sl:
            exit_px = sl_px
            exit_reason = "STOP_LOSS (Simultaneous)"
            exit_time = row['datetime']
            break
        elif hit_tp:
            exit_px = tp_px
            exit_reason = "TAKE_PROFIT"
            exit_time = row['datetime']
            break
        elif hit_sl:
            exit_px = sl_px
            exit_reason = "STOP_LOSS"
            exit_time = row['datetime']
            break
            
    if exit_px is None and not post_candles.empty:
        exit_px = post_candles.iloc[-1]['close']
        exit_reason = "TIME_STOP (90m EOD)"
        exit_time = post_candles.iloc[-1]['datetime']
        
    if exit_px is not None:
        ret = (exit_px / entry_px - 1) * 100
        print(f"   -> EXIT at {exit_time}: {exit_px:.4f} ({ret:+.2f}%) via {exit_reason}")
