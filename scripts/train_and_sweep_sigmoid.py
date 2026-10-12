import os
import sys
import sqlite3
import config
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List, Any, Optional
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.heterogeneous_models import (
    OrderFlowImbalanceModel,
    TDATopologyModel,
    StateSpaceKalmanModel,
    CrossAssetDislocationModel
)
from core.ml_engine import MLFeatureEngine

# Load historical database
conn = sqlite3.connect('data/market_data.db')
long_df = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol=config.BASE_ASSET_LONG AND timeframe='15m' ORDER BY datetime", conn)
short_df = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol=config.BASE_ASSET_SHORT AND timeframe='15m' ORDER BY datetime", conn)
nvda_df = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol='NVDA' AND timeframe='15m' ORDER BY datetime", conn)
qqq_df = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol='QQQ' AND timeframe='15m' ORDER BY datetime", conn)
macro_df = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol=config.MACRO_TREND_SYMBOL AND timeframe='15m' ORDER BY datetime", conn)
vix_df = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol='^VIX' AND timeframe='15m' ORDER BY datetime", conn)
tnx_df = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol='^TNX' AND timeframe='15m' ORDER BY datetime", conn)
conn.close()

for df in [long_df, short_df, nvda_df, qqq_df, macro_df, vix_df, tnx_df]:
    df['Datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].str.slice(0, 10)
    df.set_index('Datetime', inplace=True)

unique_dates = sorted(long_df['date'].unique())
total_days = len(unique_dates)
total_weeks = total_days / 5.0

print(f"Loaded {len(long_df)} bars across {total_days} days ({total_weeks:.1f} weeks).")

# 1. Build Expert Candidate Signals and Labels
exp_orderflow = OrderFlowImbalanceModel()
exp_tda = TDATopologyModel()
exp_statespace = StateSpaceKalmanModel()
exp_cross = CrossAssetDislocationModel()
ml_engine = MLFeatureEngine()

long_feat = ml_engine.extract_features(long_df)
short_feat = ml_engine.extract_features(short_df)

# Let's extract training samples for each bar
samples = []
n_bars = len(long_df)

for i in range(30, n_bars - 6):
    bar_dt = long_df['datetime'].iloc[i]
    time_str = bar_dt.split(" ")[1][:5]
    
    # Skip pre-market, first bar, and after cutoff (14:30 NYT)
    if time_str < "09:45" or time_str >= "14:30":
        continue
        
    sub_long = long_df.iloc[:i+1].tail(30)
    sub_nvda = nvda_df.iloc[:i+1].tail(30)
    sub_qqq = qqq_df.iloc[:i+1].tail(30)
    sub_macro = macro_df.iloc[:i+1].tail(30)
    sub_vix = vix_df.iloc[:i+1].tail(30)
    sub_tnx = tnx_df.iloc[:i+1].tail(30)
    
    cur_vix = float(sub_vix['Close'].iloc[-1]) if not sub_vix.empty else 16.5
    
    # 5 Regime Sensor Features:
    # 1. elapsed_min
    b_hour = int(time_str[:2])
    b_min = int(time_str[3:5])
    elapsed_min = (b_hour - 9) * 60 + (b_min - 30)
    elapsed_norm = (elapsed_min - 120.0) / 100.0
    
    # 2. vix_norm
    vix_norm = (cur_vix - 17.0) / 4.0
    
    # 3. atr_ratio
    hl = sub_long['High'] - sub_long['Low']
    rec_atr = hl.tail(5).mean()
    avg_atr = hl.tail(20).mean()
    atr_ratio = (rec_atr / avg_atr) if avg_atr > 0 else 1.0
    atr_norm = (atr_ratio - 1.0) / 0.4
    
    # 4. cvd_delta
    c = sub_long['Close']
    o = sub_long['Open']
    h = sub_long['High']
    l = sub_long['Low']
    v = sub_long['Volume']
    hl_diff = (h - l).replace(0, 0.001)
    v_delta = ((c - o) / hl_diff) * v
    avg_v = v.tail(15).mean() + 1e-6
    cvd_delta = float(v_delta.tail(3).mean() / avg_v)
    cvd_norm = np.clip(cvd_delta, -3.0, 3.0)
    
    # 5. dislocation_lag
    s_ret = (sub_long['Close'].iloc[-1] / sub_long['Close'].iloc[-5] - 1.0) * 100.0
    n_ret = (sub_nvda['Close'].iloc[-1] / sub_nvda['Close'].iloc[-5] - 1.0) * 100.0
    q_ret = (sub_qqq['Close'].iloc[-1] / sub_qqq['Close'].iloc[-5] - 1.0) * 100.0
    macro_exp = (n_ret * 0.6 + q_ret * 0.4) * 3.0
    dislocation_lag = float(macro_exp - s_ret)
    disloc_norm = np.clip(dislocation_lag / 1.5, -3.0, 3.0)
    
    R_vec = [elapsed_norm, vix_norm, atr_norm, cvd_norm, disloc_norm]
    
    # Determine future 6-bar ground truth for LONG and SHORT
    entry_px_l = float(long_df['Close'].iloc[i])
    entry_px_s = float(short_df['Close'].iloc[i])
    
    fut_long = long_df.iloc[i+1 : i+7]
    fut_short = short_df.iloc[i+1 : i+7]
    
    # LONG Outcome
    win_long = False
    for _, f_r in fut_long.iterrows():
        if (f_r['High'] - entry_px_l) / entry_px_l >= 0.035:
            win_long = True
            break
        if (f_r['Low'] - entry_px_l) / entry_px_l <= -0.020:
            win_long = False
            break
    else:
        last_ret = (fut_long['Close'].iloc[-1] - entry_px_l) / entry_px_l
        win_long = (last_ret > 0.0030)

    # SHORT Outcome
    win_short = False
    for _, f_r in fut_short.iterrows():
        if (f_r['High'] - entry_px_s) / entry_px_s >= 0.035:
            win_short = True
            break
        if (f_r['Low'] - entry_px_s) / entry_px_s <= -0.020:
            win_short = False
            break
    else:
        last_ret = (fut_short['Close'].iloc[-1] - entry_px_s) / entry_px_s
        win_short = (last_ret > 0.0030)

    # Check each of the 6 tracks' candidates
    # Track 0: GBDT Pattern
    s0 = 1 if (s_ret >= 0.8) else (-1 if s_ret <= -0.8 else 0)
    # Track 1: GBDT Refresh
    s1 = 1 if (s_ret >= 0.5) else (-1 if s_ret <= -0.5 else 0)
    # Track 2: OrderFlow
    cvd_df = exp_orderflow.compute_cvd(sub_long)
    s2, c2, _ = exp_orderflow.predict_signal(cvd_df.iloc[-1])
    # Track 3: TDA
    s3, c3, _ = exp_tda.predict_signal(sub_long)
    # Track 4: StateSpace
    s4, c4, _ = exp_statespace.predict_signal(sub_long['Close'].values)
    # Track 5: CrossAsset
    sx_ret = (sub_macro['Close'].iloc[-1] / sub_macro['Close'].iloc[-5] - 1.0) if len(sub_macro) >= 5 else 0.0
    v_ret = (cur_vix / sub_vix['Close'].iloc[-5] - 1.0) if len(sub_vix) >= 5 else 0.0
    t_ret = (sub_tnx['Close'].iloc[-1] / sub_tnx['Close'].iloc[-5] - 1.0) if len(sub_tnx) >= 5 else 0.0
    s5, c5, _ = exp_cross.predict_signal(s_ret/100, n_ret/100, q_ret/100, sx_ret, v_ret, t_ret)

    track_signals = [s0, s1, s2, s3, s4, s5]
    
    samples.append({
        "bar_idx": i,
        "bar_dt": bar_dt,
        "time_str": time_str,
        "R": R_vec,
        "signals": track_signals,
        "win_long": win_long,
        "win_short": win_short,
        "entry_px_l": entry_px_l,
        "entry_px_s": entry_px_s,
        "fut_long": fut_long,
        "fut_short": fut_short
    })

print(f"Generated {len(samples)} valid intraday evaluation bars.")

# Train Logistic Regressions for each track to predict P(Win | R)
X_all = np.array([s["R"] for s in samples])
models = []

for k in range(6):
    # Filter where track k generated an active signal
    active_indices = [idx for idx, s in enumerate(samples) if s["signals"][k] != 0]
    if len(active_indices) >= 20:
        X_k = X_all[active_indices]
        y_k = np.array([1 if (samples[idx]["signals"][k] == 1 and samples[idx]["win_long"]) or (samples[idx]["signals"][k] == -1 and samples[idx]["win_short"]) else 0 for idx in active_indices])
        clf = LogisticRegression(C=1.0, max_iter=200)
        clf.fit(X_k, y_k)
        models.append(clf)
    else:
        # Default model
        clf = LogisticRegression()
        clf.coef_ = np.zeros((1, 5))
        clf.intercept_ = np.array([0.5])
        clf.classes_ = np.array([0, 1])
        models.append(clf)

print("Trained Sigmoid Gating Weights for 6 Tracks:")
for k, m in enumerate(models):
    print(f"Track {k}: coef={m.coef_[0].round(3)}, intercept={m.intercept_[0]:.3f}")

# Now Run Full Backtest with Sigmoid Gating Layer across 5 Thresholds
INITIAL_CAPITAL_KRW = 10_000_000
FEE_RATE = 0.0030
TAKE_PROFIT_PCT = 0.035
STOP_LOSS_PCT = -0.020
TIME_STOP_BARS = 6
THRESHOLDS = [0.60, 0.70, 0.75, 0.80, 0.85]

results = []

for T in THRESHOLDS:
    capital = float(INITIAL_CAPITAL_KRW)
    trades = []
    equity = [capital]
    
    in_market = False
    current_pos = "NONE"
    entry_px = 0.0
    entry_time = ""
    bars_held = 0
    
    for s in samples:
        bar_dt = s["bar_dt"]
        b_idx = s["bar_idx"]
        
        # If in position, check exits on current bar
        if in_market and current_pos != "NONE":
            bars_held += 1
            curr_row = long_df.iloc[b_idx] if current_pos == config.BASE_ASSET_LONG else short_df.iloc[b_idx]
            curr_h = float(curr_row['High'])
            curr_l = float(curr_row['Low'])
            curr_c = float(curr_row['Close'])
            
            h_ret = (curr_h - entry_px) / entry_px
            l_ret = (curr_l - entry_px) / entry_px
            c_ret = (curr_c - entry_px) / entry_px
            
            exit_hit = False
            actual_ret = 0.0
            exit_rsn = ""
            
            if h_ret >= TAKE_PROFIT_PCT:
                exit_hit = True
                actual_ret = TAKE_PROFIT_PCT - FEE_RATE
                exit_rsn = "TAKE_PROFIT_+3.5%"
            elif l_ret <= STOP_LOSS_PCT:
                exit_hit = True
                actual_ret = STOP_LOSS_PCT - FEE_RATE
                exit_rsn = "STOP_LOSS_-2.0%"
            elif bars_held >= TIME_STOP_BARS:
                exit_hit = True
                actual_ret = c_ret - FEE_RATE
                exit_rsn = "TIME_STOP_90M"
            elif s["time_str"] >= "15:45":
                exit_hit = True
                actual_ret = c_ret - FEE_RATE
                exit_rsn = "EOD_CLOSE"
                
            if exit_hit:
                pnl_krw = int(capital * actual_ret)
                capital += pnl_krw
                equity.append(capital)
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": bar_dt,
                    "ticker": current_pos,
                    "pnl_pct": actual_ret * 100,
                    "pnl_krw": pnl_krw,
                    "is_win": actual_ret > 0,
                    "exit_rsn": exit_rsn
                })
                in_market = False
                current_pos = "NONE"
                bars_held = 0
                continue
                
        # If not in position, check entry conditions
        if not in_market and current_pos == "NONE":
            # Compute Sigmoid scores for all tracks
            R_arr = np.array(s["R"]).reshape(1, -1)
            sig_scores = {}
            for k in range(6):
                if s["signals"][k] != 0:
                    prob = float(models[k].predict_proba(R_arr)[0, 1])
                else:
                    prob = 0.0
                sig_scores[k] = prob
                
            top_k = max(sig_scores, key=sig_scores.get)
            top_score = sig_scores[top_k]
            
            # Sigmoid Absolute Threshold Filter: Top-1 Score >= T
            if top_score >= T and s["signals"][top_k] != 0:
                direction = config.BASE_ASSET_LONG if s["signals"][top_k] == 1 else config.BASE_ASSET_SHORT
                in_market = True
                current_pos = direction
                entry_px = float(long_df['Close'].iloc[b_idx]) if direction == config.BASE_ASSET_LONG else float(short_df['Close'].iloc[b_idx])
                entry_time = bar_dt
                bars_held = 0

    tot_trades = len(trades)
    wins = sum(1 for t in trades if t['is_win'])
    losses = tot_trades - wins
    win_rate = (wins / tot_trades * 100.0) if tot_trades > 0 else 0.0
    avg_per_week = tot_trades / total_weeks
    
    tot_pnl = int(capital - INITIAL_CAPITAL_KRW)
    cum_ret = (tot_pnl / INITIAL_CAPITAL_KRW) * 100.0
    
    gross_win = sum(t['pnl_krw'] for t in trades if t['pnl_krw'] > 0)
    gross_loss = abs(sum(t['pnl_krw'] for t in trades if t['pnl_krw'] < 0))
    pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else 99.0
    
    eq_arr = np.array(equity)
    pk = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - pk) / pk
    mdd = round(abs(float(np.min(dd))) * 100.0, 2) if len(dd) > 0 else 0.0
    
    results.append({
        "threshold": T,
        "threshold_pct": f"{int(T*100)}%",
        "total_trades": tot_trades,
        "avg_trades_per_week": round(avg_per_week, 2),
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 1),
        "profit_factor": pf,
        "mdd": mdd,
        "cum_return": round(cum_ret, 2),
        "total_pnl": tot_pnl
    })

df_res = pd.DataFrame(results)
print("\n" + "=" * 100)
print(f"{'임계치 (T)':<10} | {'총 거래수':<8} | {'주당 평균 거래':<14} | {'전적':<10} | {'승률':<8} | {'Profit Factor':<14} | {'MDD':<8} | {'누적수익률':<10}")
print("-" * 100)
for _, r in df_res.iterrows():
    print(f"{r['threshold_pct']:<10} | {r['total_trades']:<8} | {r['avg_trades_per_week']:<14.2f} | {r['wins']}승 {r['losses']}패    | {r['win_rate']:>5.1f}%  | {r['profit_factor']:>13.2f} | {r['mdd']:>5.2f}% | {r['cum_return']:>+7.2f}%")
print("=" * 100)
