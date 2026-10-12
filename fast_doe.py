import os, sys, json, sqlite3, time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import config
from core.data_lake import MarketDataLake
from core.ml_engine import MLFeatureEngine

def run_fast_doe():
    lake = MarketDataLake()
    print("Loading data...")
    long_15m = lake.load_candles(config.BASE_ASSET_LONG, "15m")
    short_15m = lake.load_candles(config.BASE_ASSET_SHORT, "15m")
    long_5m = lake.load_candles(config.BASE_ASSET_LONG, "5m")
    
    unique_weeks = sorted(list(set(long_15m['datetime'].dt.strftime('%Y-%W'))))
    weeks_2026 = [w for w in unique_weeks if w.startswith('2026')]
    if not weeks_2026: return
    test_start_idx = unique_weeks.index(weeks_2026[0])
    test_end_idx = unique_weeks.index(weeks_2026[-1]) + 1
    
    time_stops = [90, 100, 110, 120]
    results = {ts: {'ret_pct': 0.0, 'trades': 0, 'wins': 0} for ts in time_stops}
    
    ml_engine = MLFeatureEngine(confidence_threshold=0.65)
    print("Extracting features...")
    long_feat = ml_engine.extract_features(long_15m)
    labels = ml_engine.compute_triple_barrier_labels(long_feat)
    target_series = labels.map({1: 2, -1: 0, 0: 1}).fillna(1).astype(int)
    
    # 2년 = 104주
    ROLLING_WEEKS = 104
    
    print("Starting WFA Loop...")
    for w_idx in range(test_start_idx, test_end_idx):
        cur_test_week = unique_weeks[w_idx]
        train_start_week = unique_weeks[max(0, w_idx - ROLLING_WEEKS)]
        train_end_week = unique_weeks[w_idx - 1]
        
        train_mask = (long_feat['datetime'].dt.strftime('%Y-%W') >= train_start_week) & (long_feat['datetime'].dt.strftime('%Y-%W') <= train_end_week)
        test_mask = (long_feat['datetime'].dt.strftime('%Y-%W') == cur_test_week)
        
        X_train = long_feat[train_mask].drop(columns=['datetime', 'Open', 'High', 'Low', 'Close', 'Volume'], errors='ignore')
        y_train = target_series[train_mask]
        
        X_test = long_feat[test_mask]
        if X_test.empty or X_train.empty: continue
        
        X_train_num = X_train.select_dtypes(include=[np.number])
        X_test_num = X_test.select_dtypes(include=[np.number])
        
        model = LGBMClassifier(n_estimators=150, learning_rate=0.03, max_depth=6, random_state=42, n_jobs=-1, verbose=-1)
        model.fit(X_train_num, y_train)
        
        preds = model.predict_proba(X_test_num)
        
        # Simulate trades for each time stop
        for ts in time_stops:
            active_pos = None
            ts_trades = 0
            ts_ret = 0.0
            ts_wins = 0
            
            # Simple simulation approximation for speed
            for i in range(len(X_test)):
                if active_pos:
                    active_pos['bars'] += 1
                    if active_pos['bars'] * 15 >= ts:
                        # hit time stop
                        ts_trades += 1
                        ts_ret -= 0.005 # rough penalty
                        active_pos = None
                    continue
                    
                p0 = preds[i][0] # short
                p2 = preds[i][2] # long
                conf = max(p0, p2)
                
                # Check neutral
                if preds[i][1] >= p0 and preds[i][1] >= p2: continue
                
                if conf >= 0.65:
                    active_pos = {'type': 'LONG' if p2 > p0 else 'SHORT', 'bars': 0}
                    # We assume 1/3 win rate with +1%, 2/3 lose with -0.5% roughly based on average if no fine-grained 5m sim
            
            # For a proper sim, we need 5m data. Let's do the proper 5m sim!
            test_start_dt = X_test['datetime'].iloc[0]
            test_end_dt = X_test['datetime'].iloc[-1] + pd.Timedelta(days=1)
            
            # Filter 5m data
            m5 = long_5m[(long_5m['datetime'] >= test_start_dt) & (long_5m['datetime'] <= test_end_dt)]
            s5 = short_15m[(short_15m['datetime'] >= test_start_dt) & (short_15m['datetime'] <= test_end_dt)]
            
            # THIS IS A PLACEHOLDER FOR FAST PNL CALC.
            # I will actually just run it directly.

print("Use simple PnL simulation")
