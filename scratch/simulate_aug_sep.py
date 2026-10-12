import os
import sys
import pandas as pd
import sqlite3
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config

conn = sqlite3.connect(config.DATA_DIR / "market_data.db")
df_tqqq_15m = pd.read_sql("SELECT * FROM market_candles WHERE symbol='TQQQ' AND timeframe='15m' AND datetime >= '2026-07-01' ORDER BY datetime", conn)
df_sqqq_15m = pd.read_sql("SELECT * FROM market_candles WHERE symbol='SQQQ' AND timeframe='15m' AND datetime >= '2026-07-01' ORDER BY datetime", conn)
df_tqqq_5m = pd.read_sql("SELECT * FROM market_candles WHERE symbol='TQQQ' AND timeframe='5m' AND datetime >= '2026-08-01' ORDER BY datetime", conn)
df_sqqq_5m = pd.read_sql("SELECT * FROM market_candles WHERE symbol='SQQQ' AND timeframe='5m' AND datetime >= '2026-08-01' ORDER BY datetime", conn)

df_nvda_15m = pd.read_sql("SELECT * FROM market_candles WHERE symbol='NVDA' AND timeframe='15m' AND datetime >= '2026-07-01' ORDER BY datetime", conn)
df_qqq_15m = pd.read_sql("SELECT * FROM market_candles WHERE symbol='QQQ' AND timeframe='15m' AND datetime >= '2026-07-01' ORDER BY datetime", conn)
df_soxx_15m = pd.read_sql("SELECT * FROM market_candles WHERE symbol='SOXX' AND timeframe='15m' AND datetime >= '2026-07-01' ORDER BY datetime", conn)
df_vix_15m = pd.read_sql("SELECT * FROM market_candles WHERE symbol='VIXY' AND timeframe='15m' AND datetime >= '2026-07-01' ORDER BY datetime", conn)
df_ief_15m = pd.read_sql("SELECT * FROM market_candles WHERE symbol='IEF' AND timeframe='15m' AND datetime >= '2026-07-01' ORDER BY datetime", conn)

conn.close()

from core.moe_orchestrator import MoEMetaOrchestrator

moe = MoEMetaOrchestrator()
target_dates = df_tqqq_15m[(df_tqqq_15m['datetime'] >= '2026-08-01') & (df_tqqq_15m['datetime'] <= '2026-09-30')]['datetime'].str.slice(0, 10).unique()

trades = []

for t_date in target_dates:
    day_times = df_tqqq_15m[df_tqqq_15m['datetime'].str.startswith(t_date)]['datetime'].tolist()
    
    for current_time in day_times:
        time_only = current_time[11:16]
        if time_only >= "15:30":
            continue
            
        cur_tqqq = df_tqqq_15m[df_tqqq_15m['datetime'] <= current_time]
        if len(cur_tqqq) < 50: continue
        
        # We need to mock live_prices
        live_prices = {
            "TQQQ": cur_tqqq.iloc[-1]['close'],
            "SQQQ": df_sqqq_15m[df_sqqq_15m['datetime'] <= current_time].iloc[-1]['close'],
            "NVDA": df_nvda_15m[df_nvda_15m['datetime'] <= current_time].iloc[-1]['close'] if not df_nvda_15m.empty else 0,
            "QQQ": df_qqq_15m[df_qqq_15m['datetime'] <= current_time].iloc[-1]['close'] if not df_qqq_15m.empty else 0,
            "SOXX": df_soxx_15m[df_soxx_15m['datetime'] <= current_time].iloc[-1]['close'] if not df_soxx_15m.empty else 0,
            "VIXY": df_vix_15m[df_vix_15m['datetime'] <= current_time].iloc[-1]['close'] if not df_vix_15m.empty else 0,
            "IEF": df_ief_15m[df_ief_15m['datetime'] <= current_time].iloc[-1]['close'] if not df_ief_15m.empty else 0,
        }
        
        moe_res = moe.evaluate_dual_filter_signal(df_candle_15m=cur_tqqq, live_prices=live_prices)
        
        if moe_res.get('is_approved', False):
            direction = moe_res.get('direction', 'NONE')
            if direction == "NONE": continue
            
            sym = "TQQQ" if direction == "LONG_TQQQ" else "SQQQ"
            entry_px = live_prices[sym]
            
            target_5m = df_tqqq_5m if sym == "TQQQ" else df_sqqq_5m
            post_5m = target_5m[target_5m['datetime'] > current_time].head(int(config.TIME_STOP_MINUTES / 5))
            
            tp_px = entry_px * (1.0 + config.MAX_TP_PCT)
            sl_px = entry_px * (1.0 - config.SL_MIN_PCT)
            
            exit_px = None
            exit_reason = None
            exit_dt = None
            
            for _, r5 in post_5m.iterrows():
                b_h, b_l = r5['high'], r5['low']
                if b_h >= tp_px and b_l <= sl_px:
                    exit_px = sl_px
                    exit_reason = "SL (Simul)"
                    exit_dt = r5['datetime']
                    break
                elif b_h >= tp_px:
                    exit_px = tp_px
                    exit_reason = "TP"
                    exit_dt = r5['datetime']
                    break
                elif b_l <= sl_px:
                    exit_px = sl_px
                    exit_reason = "SL"
                    exit_dt = r5['datetime']
                    break
                    
            if exit_px is None and not post_5m.empty:
                exit_px = post_5m.iloc[-1]['close']
                exit_reason = "TIME_STOP"
                exit_dt = post_5m.iloc[-1]['datetime']
                
            if exit_px:
                ret_pct = (exit_px / entry_px - 1.0) * 100
                trades.append({
                    "entry_time": current_time,
                    "direction": direction,
                    "entry_px": entry_px,
                    "exit_time": exit_dt,
                    "exit_px": exit_px,
                    "ret_pct": ret_pct,
                    "reason": exit_reason
                })
                # Skip 120 minutes of checking to prevent overlapping trades
                break 

print("\n--- Summary (August ~ September 2026) ---")
for t in trades:
    print(f"{t['entry_time']} | {t['direction']:>10} | {t['ret_pct']:>6.2f}% | {t['reason']}")
win = sum(1 for t in trades if t['ret_pct'] > 0)
loss = sum(1 for t in trades if t['ret_pct'] <= 0)
total_ret = sum(t['ret_pct'] for t in trades)
print(f"Total Trades: {len(trades)}, Win: {win}, Loss: {loss}")
print(f"Total Return: {total_ret:.2f}%")
