import sys
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config

# Override config for DoE
config.GBDT_CONFIDENCE_THRESHOLD = 0.65
import core.moe_orchestrator # forces load

from core.data_lake import MarketDataLake
from core.ml_engine import MLFeatureEngine
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np

def run_2026(time_stop_mins):
    lake = MarketDataLake()
    long_15m = lake.load_candles(config.BASE_ASSET_LONG, "15m")
    long_15m['datetime'] = pd.to_datetime(long_15m['datetime'])
    
    unique_weeks = sorted(list(set(long_15m['datetime'].dt.strftime('%Y-%W'))))
    weeks_2026 = [w for w in unique_weeks if w.startswith('2026')]
    test_start_idx = unique_weeks.index(weeks_2026[0])
    test_end_idx = unique_weeks.index(weeks_2026[-1]) + 1
    
    ml_engine = MLFeatureEngine(confidence_threshold=0.65)
    long_feat = ml_engine.extract_features(long_15m)
    long_feat['datetime'] = pd.to_datetime(long_feat['datetime'])
    labels = ml_engine.compute_triple_barrier_labels(long_feat)
    target_series = labels.map({1: 2, -1: 0, 0: 1}).fillna(1).astype(int)
    
    ROLLING_WEEKS = 104
    trades = []
    
    for w_idx in range(test_start_idx, test_end_idx):
        cur_test_week = unique_weeks[w_idx]
        train_start_week = unique_weeks[max(0, w_idx - ROLLING_WEEKS)]
        train_end_week = unique_weeks[w_idx - 1]
        
        train_mask = (long_feat['datetime'].dt.strftime('%Y-%W') >= train_start_week) & (long_feat['datetime'].dt.strftime('%Y-%W') <= train_end_week)
        test_mask = (long_feat['datetime'].dt.strftime('%Y-%W') == cur_test_week)
        
        X_train = long_feat[train_mask]
        X_test = long_feat[test_mask]
        if X_train.empty or X_test.empty: continue
        
        y_train = target_series[train_mask]
        
        drop_cols = ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'cum_vp', 'vwap', 'EMA_9', 'EMA_21', 'EMA_50', 'EMA_200']
        feature_cols = [c for c in X_train.columns if c not in drop_cols and not c.startswith('week_') and pd.api.types.is_numeric_dtype(X_train[c])]
        
        model = LGBMClassifier(n_estimators=150, learning_rate=0.03, max_depth=6, random_state=42, n_jobs=1, verbose=-1)
        model.fit(X_train[feature_cols].fillna(0), y_train)
        preds = model.predict_proba(X_test[feature_cols].fillna(0))
        
        active_pos = None
        for i in range(len(X_test)):
            row = X_test.iloc[i]
            
            if active_pos:
                active_pos['bars'] += 1
                bars_held = active_pos['bars']
                
                if bars_held * 15 >= time_stop_mins:
                    trades.append({'ret': -0.005, 'win': 0}) 
                    active_pos = None
                continue
                
            p0 = preds[i][0]
            p2 = preds[i][2]
            conf = max(p0, p2)
            
            if preds[i][1] >= p0 and preds[i][1] >= p2: continue
            if conf >= 0.65:
                active_pos = {'type': 'LONG' if p2 > p0 else 'SHORT', 'bars': 0}
                end_idx = min(i + int(time_stop_mins/15) + 1, len(X_test))
                sim_slice = X_test.iloc[i+1:end_idx]
                p_entry = row['Close']
                hit_sl = False
                hit_tp = False
                for _, s_row in sim_slice.iterrows():
                    h_ret = (s_row['High'] - p_entry) / p_entry
                    l_ret = (s_row['Low'] - p_entry) / p_entry
                    
                    if active_pos['type'] == 'LONG':
                        if l_ret <= -0.02:
                            hit_sl = True; trades.append({'ret': -0.02, 'win': 0}); break
                        elif h_ret >= 0.01:
                            hit_tp = True; trades.append({'ret': 0.01, 'win': 1}); break
                    else:
                        if h_ret >= 0.02:
                            hit_sl = True; trades.append({'ret': -0.02, 'win': 0}); break
                        elif l_ret <= -0.01:
                            hit_tp = True; trades.append({'ret': 0.01, 'win': 1}); break
                
                if hit_sl or hit_tp:
                    active_pos = None
                else:
                    if len(sim_slice) == int(time_stop_mins/15):
                        trades.append({'ret': -0.005, 'win': 0})
                        active_pos = None

    df = pd.DataFrame(trades)
    if len(df) == 0:
        return 0.0, 0, 0.0
    
    win_rate = df['win'].mean() * 100
    total_ret = df['ret'].sum() * 100
    num_trades = len(df)
    return total_ret, num_trades, win_rate

if __name__ == "__main__":
    results = []
    print("="*60)
    print(f"{'Time Stop (m)':<15} | {'PnL (%)':<10} | {'Trades':<10} | {'Win Rate':<10}")
    print("-" * 60)
    for ts in [90, 100, 110, 120]:
        ret, cnt, wr = run_2026(ts)
        results.append((ts, ret, cnt, wr))
        print(f"{ts:<15} | {ret:>+9.2f}% | {cnt:>10} | {wr:>9.1f}%")
    print("="*60)
