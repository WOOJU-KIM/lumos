import os
import sys
import config
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(r"c:\Users\chabo\OneDrive\바탕 화면\lumos")
sys.path.insert(0, str(PROJECT_ROOT))

from core.data_lake import MarketDataLake
from core.ml_engine import MLFeatureEngine

def compute_60m_trend(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values('datetime')
    df['ema20'] = df['Close'].ewm(span=20, adjust=False).mean()
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    return df

def run_walk_forward():
    print("=" * 60)
    print("🚀 [Lumos 2-Year Rolling OOS Backtest (5m Exit & 3-Screen)]")
    print("=" * 60)

    lake = MarketDataLake()
    
    print("[1/4] 데이터 로드 중...")
    long_15m = lake.load_candles(config.BASE_ASSET_LONG, "15m")
    long_15m['datetime'] = pd.to_datetime(long_15m['datetime'])
    long_15m = long_15m.sort_values('datetime').reset_index(drop=True)
    
    long_5m = lake.load_candles(config.BASE_ASSET_LONG, "5m")
    long_5m['datetime'] = pd.to_datetime(long_5m['datetime'])
    long_5m = long_5m.sort_values('datetime').reset_index(drop=True)

    long_60m = lake.load_candles(config.BASE_ASSET_LONG, "60m")
    long_60m['datetime'] = pd.to_datetime(long_60m['datetime'])
    long_60m = compute_60m_trend(long_60m)

    macro_60m = lake.load_candles(config.MACRO_TREND_SYMBOL, "60m")
    macro_60m['datetime'] = pd.to_datetime(macro_60m['datetime'])
    macro_60m = compute_60m_trend(macro_60m)
    
    print("[2/4] 피쳐 추출 중 (15m)...")
    ml_engine = MLFeatureEngine(confidence_threshold=0.62)
    df_feat = ml_engine.extract_features(long_15m)
    
    feature_cols = [
        'RSI_7', 'RSI_14', 'RSI_21', 'MACD', 'MACD_Hist', 'Stoch_K', 'Stoch_D',
        'ROC_5', 'ROC_10', 'ROC_20', 'CCI_14', 'CCI_20', 'ATR_14', 'BB_PctB',
        'BB_Bandwidth', 'KC_Width', 'OBV', 'CMF_20', 'Volume_Z', 'VWAP_Diff',
        'EMA_9_Diff', 'EMA_21_Diff', 'EMA_50_Diff', 'ADX_14', 'DMI_Diff',
        'Body_Ratio', 'Upper_Wick_Ratio', 'Lower_Wick_Ratio'
    ]
    
    start_dt = df_feat['datetime'].min()
    end_dt = df_feat['datetime'].max()
    
    first_test_start = start_dt + timedelta(days=730)
    while first_test_start.weekday() != 0:
        first_test_start += timedelta(days=1)
        
    current_test_start = first_test_start
    
    total_trades = 0
    wins = 0
    capital = 10_000_000
    
    week_idx = 1
    
    print(f"[3/4] Walk-Forward Rolling 시작 (Test Start: {first_test_start.strftime('%Y-%m-%d')})")
    
    while current_test_start < end_dt:
        test_end = current_test_start + timedelta(days=7)
        train_start = current_test_start - timedelta(days=730)
        
        train_raw = long_15m[(long_15m['datetime'] >= train_start) & (long_15m['datetime'] < current_test_start)].copy()
        test_df = df_feat[(df_feat['datetime'] >= current_test_start) & (df_feat['datetime'] < test_end)].copy()
        
        if len(train_raw) < 1000 or test_df.empty:
            current_test_start = test_end
            continue
            
        try:
            trained_model, _, _ = ml_engine.train_and_select_top_features(train_raw)
            if trained_model is None:
                current_test_start = test_end
                continue
            ml_engine.model = trained_model
            ml_engine.is_trained = True
        except Exception as e:
            current_test_start = test_end
            continue
            
        for idx, row in test_df.iterrows():
            dt = row['datetime']
            
            if not (9 <= dt.hour <= 14): continue
            if dt.hour == 14 and dt.minute > 30: continue
            
            # Screen 1
            past_macro = macro_60m[macro_60m['datetime'] <= dt]
            past_long = long_60m[long_60m['datetime'] <= dt]
            if len(past_macro) < 20 or len(past_long) < 20: continue
            
            last_macro = past_macro.iloc[-1]
            last_long = past_long.iloc[-1]
            
            macro_bull = (last_macro['Close'] >= last_macro['ema20'] * 0.998)
            long_bull = (last_long['Close'] >= last_long['ema20'] * 0.998) and (last_long['macd'] >= last_long['macd_signal'] * 0.98)
            is_60m_bull = macro_bull and long_bull
            
            # Screen 3
            vwap_diff = row.get('VWAP_Diff', 999)
            rsi_14 = row.get('RSI_14', 999)
            close_px = row['Close']
            bb_lower = row.get('BB_Lower', 0)
            long_dip_ok = (vwap_diff <= 1.5) and (rsi_14 <= 62.0) and (close_px >= bb_lower * 1.001)
            
            if not (is_60m_bull and long_dip_ok):
                continue
            
            # AI
            row_features = pd.DataFrame([row])[feature_cols]
            if row_features.isna().any().any(): continue
                
            probs = ml_engine.model.predict_proba(row_features)
            
            # LightGBM typically outputs [P(-1), P(0), P(1)] depending on classes
            classes = list(ml_engine.model.classes_)
            
            prob_short = probs[0][classes.index(-1)] if -1 in classes else 0
            prob_neutral = probs[0][classes.index(0)] if 0 in classes else 0
            prob_long = probs[0][classes.index(1)] if 1 in classes else 0
            
            calib_conf = 0.50
            if prob_long > prob_neutral and prob_long > prob_short:
                calib_conf = min(0.95, max(0.50, 0.50 + (prob_long - 0.333) * 1.15))
            
            if calib_conf >= 0.62:
                entry_px = close_px
                
                future_5m = long_5m[(long_5m['datetime'] > dt) & (long_5m['datetime'] <= dt + timedelta(minutes=90))]
                
                outcome = "TIME_STOP"
                exit_px = entry_px
                
                for _, f_row in future_5m.iterrows():
                    high = f_row['High']
                    low = f_row['Low']
                    
                    hit_tp = high >= entry_px * 1.03
                    hit_sl = low <= entry_px * 0.98
                    
                    if hit_tp and hit_sl:
                        outcome = "LOSS"
                        exit_px = entry_px * 0.98
                        break
                    elif hit_sl:
                        outcome = "LOSS"
                        exit_px = entry_px * 0.98
                        break
                    elif hit_tp:
                        outcome = "WIN"
                        exit_px = entry_px * 1.03
                        break
                        
                if outcome == "TIME_STOP" and not future_5m.empty:
                    exit_px = future_5m.iloc[-1]['Close']
                    if exit_px > entry_px: outcome = "WIN"
                    else: outcome = "LOSS"
                    
                total_trades += 1
                if outcome == "WIN": wins += 1
                
                ret = (exit_px - entry_px) / entry_px
                capital *= (1 + ret)
                
        current_test_start = test_end
        week_idx += 1

    print("=" * 60)
    print(f"📊 [Walk-Forward Backtest Results]")
    print(f"   - 총 테스트 기간: {first_test_start.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')}")
    print(f"   - 총 테스트 주(Weeks): {week_idx - 1}")
    print(f"   - 총 체결: {total_trades}건")
    if total_trades > 0:
        print(f"   - 승률: {(wins/total_trades)*100:.1f}%")
    print(f"   - 최종 자본금: {capital:,.0f} KRW (초기 1000만)")
    
    if (week_idx - 1) > 0:
        years = (end_dt - first_test_start).days / 365.25
        if years > 0:
            cagr = ((capital / 10_000_000) ** (1/years) - 1) * 100
            print(f"   - 연평균 수익률(CAGR): {cagr:.2f}%")
            
    print("=" * 60)

if __name__ == "__main__":
    run_walk_forward()
