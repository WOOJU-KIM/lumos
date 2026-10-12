import sys
import pandas as pd
import sqlite3
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config
from core.ml_engine import MLFeatureEngine

conn = sqlite3.connect(config.DATA_DIR / "market_data.db")
df_tqqq_15m = pd.read_sql("SELECT * FROM market_candles WHERE symbol='TQQQ' AND timeframe='15m' AND datetime >= '2026-07-01' ORDER BY datetime", conn)
df_tqqq_5m = pd.read_sql("SELECT * FROM market_candles WHERE symbol='TQQQ' AND timeframe='5m' AND datetime >= '2026-07-01' ORDER BY datetime", conn)
df_sqqq_5m = pd.read_sql("SELECT * FROM market_candles WHERE symbol='SQQQ' AND timeframe='5m' AND datetime >= '2026-07-01' ORDER BY datetime", conn)
conn.close()

ml = MLFeatureEngine()

# This automatically loads cross-assets from DB and generates all features!
# We just pass the entire df_tqqq_15m
print("Extracting features...")
feat = ml.extract_features(df_tqqq_15m)

feat = feat[feat['datetime'] >= '2026-08-01'].copy()
feat = feat[feat['datetime'] <= '2026-09-30'].copy()

# Add confidence columns (predicts using the model)
print("Predicting signals...")
feat = ml.add_confidence_columns(feat)

signals = []
active_exit = "0000"

df_tqqq_5m.set_index('datetime', inplace=True)
df_sqqq_5m.set_index('datetime', inplace=True)

trades = []

for idx, row in feat.iterrows():
    dt = row['datetime']
    time_only = dt[11:16]
    if time_only >= config.PHASE_MAIN_END: continue
    if dt < active_exit: continue
    
    c_long = row.get("Prob_Long", 0.0)
    c_short = row.get("Prob_Short", 0.0)
    
    direction, conf = 0, 0.0
    if c_long >= c_short and c_long >= config.GBDT_CONFIDENCE_THRESHOLD:
        direction, conf = 1, c_long
    elif c_short > c_long and c_short >= config.GBDT_CONFIDENCE_THRESHOLD:
        direction, conf = -1, c_short
        
    if direction == 1 and config.USE_CROSS_ASSET_VETO:
        if row.get('NVDA_ret_1d', 0) < -0.015: direction = 0
    if direction == -1 and config.USE_CROSS_ASSET_VETO:
        if row.get('VIX_ret_1d', 0) < -0.01: direction = 0
        
    if direction == 1:
        d_type = "LONG_TQQQ"
    elif direction == -1:
        d_type = "SHORT_SQQQ"
    else:
        continue

    sym = "TQQQ" if d_type == "LONG_TQQQ" else "SQQQ"
    target_5m = df_tqqq_5m if sym == "TQQQ" else df_sqqq_5m
    
    post_5m = target_5m.loc[dt:].iloc[1:int(config.TIME_STOP_MINUTES/5)+1]
    if post_5m.empty: continue
    
    entry_px = target_5m.loc[dt, 'close'] if dt in target_5m.index else post_5m.iloc[0]['open']
    tp_px = entry_px * (1.0 + config.MAX_TP_PCT)
    sl_px = entry_px * (1.0 - config.SL_MIN_PCT)
    
    exit_px, exit_reason = None, None
    exit_dt = dt
    
    for edt, r5 in post_5m.iterrows():
        b_h, b_l = r5['high'], r5['low']
        if b_h >= tp_px and b_l <= sl_px:
            exit_px, exit_reason, exit_dt = sl_px, "SL (Simul)", edt
            break
        elif b_h >= tp_px:
            exit_px, exit_reason, exit_dt = tp_px, "TP", edt
            break
        elif b_l <= sl_px:
            exit_px, exit_reason, exit_dt = sl_px, "SL", edt
            break
            
    if exit_px is None:
        exit_px, exit_reason, exit_dt = post_5m.iloc[-1]['close'], "TIME_STOP", post_5m.index[-1]
        
    ret_pct = (exit_px / entry_px - 1.0) * 100
    trades.append({"entry_time": dt, "direction": d_type, "entry_px": entry_px, "exit_px": exit_px, "ret_pct": ret_pct, "reason": exit_reason, "conf": conf, "exit_time": exit_dt})
    active_exit = exit_dt

print("\n--- Summary (August ~ September 2026) ---")
for t in trades:
    print(f"{t['entry_time']} ~ {t['exit_time']} | {t['direction']:>10} | Conf: {t['conf']*100:>5.1f}% | {t['ret_pct']:>6.2f}% | {t['reason']}")

win = sum(1 for t in trades if t['ret_pct'] > 0)
loss = sum(1 for t in trades if t['ret_pct'] <= 0)
total_ret = sum(t['ret_pct'] for t in trades)
print(f"\nTotal Trades: {len(trades)}, Win: {win}, Loss: {loss}")
print(f"Total Return: {total_ret:.2f}%")
