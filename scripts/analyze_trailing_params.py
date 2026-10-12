import os
import sys
from pathlib import Path
import config
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from core.data_lake import MarketDataLake

def analyze_volatility(lake, symbol, tf="5m", lookback_days=504):
    df = lake.load_rolling_candles(symbol, tf, max_trading_days=lookback_days)
    if df.empty:
        return None
    
    # Calculate percentage returns
    df['ret_pct'] = df['Close'].pct_change().abs() * 100.0
    
    # Calculate 5m High-Low spread percentage
    df['hl_spread_pct'] = (df['High'] - df['Low']) / df['Low'] * 100.0
    
    # Calculate True Range percentage
    df['prev_close'] = df['Close'].shift(1)
    df['tr1'] = df['High'] - df['Low']
    df['tr2'] = (df['High'] - df['prev_close']).abs()
    df['tr3'] = (df['Low'] - df['prev_close']).abs()
    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['tr_pct'] = df['tr'] / df['prev_close'] * 100.0
    
    # Calculate ATR (14 period) average
    df['atr_14_pct'] = df['tr_pct'].rolling(14).mean()
    
    # Calculate intraday max swing (open to highest high, or open to lowest low for the day)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date
    daily_swings = []
    for date, group in df.groupby('date'):
        if len(group) > 5:
            day_open = group['Open'].iloc[0]
            day_high = group['High'].max()
            day_low = group['Low'].min()
            up_swing = (day_high - day_open) / day_open * 100.0
            down_swing = (day_open - day_low) / day_open * 100.0
            daily_swings.append({'date': date, 'up_swing': up_swing, 'down_swing': down_swing})
            
    daily_swings_df = pd.DataFrame(daily_swings)
    
    result = {
        'symbol': symbol,
        'count': len(df),
        'avg_5m_ret_pct': df['ret_pct'].mean(),
        'avg_5m_hl_pct': df['hl_spread_pct'].mean(),
        'avg_atr_14_pct': df['atr_14_pct'].mean(),
        'median_atr_14_pct': df['atr_14_pct'].median(),
        'p90_atr_14_pct': df['atr_14_pct'].quantile(0.90),
        'avg_daily_up_swing': daily_swings_df['up_swing'].mean(),
        'avg_daily_down_swing': daily_swings_df['down_swing'].mean()
    }
    return result

def main():
    lake = MarketDataLake()
    symbols = [config.BASE_ASSET_LONG, config.BASE_ASSET_SHORT, "SOXL", "SOXS"]
    results = []
    
    for sym in symbols:
        print(f"Analyzing {sym}...")
        res = analyze_volatility(lake, sym, "5m", 504)
        if res:
            results.append(res)
            
    if not results:
        print("데이터가 충분하지 않습니다. 수집을 기다리거나 먼저 진행하세요.")
        return
        
    res_df = pd.DataFrame(results)
    print("\n[Volatility Analysis (5m candles)]")
    print(res_df.to_string(index=False))
    
    print("\n[Recommendations]")
    for r in results:
        # A simple heuristic:
        # Trailing Drop should be roughly 1.0 ~ 1.5x of average 5m HL spread (noise level)
        # Trigger should be roughly 2.5x to 3x the Trailing Drop to ensure we have actual profit.
        recommended_drop = r['avg_5m_hl_pct'] * 1.1
        recommended_trigger = recommended_drop * 3.0
        print(f"--- {r['symbol']} ---")
        print(f"Avg 5m H/L Spread (Noise): {r['avg_5m_hl_pct']:.3f}%")
        print(f"Avg 14-bar ATR: {r['avg_atr_14_pct']:.3f}%")
        print(f"-> Recommended Trailing Drop: -{recommended_drop/100.0:.4f} (-{recommended_drop:.2f}%)")
        print(f"-> Recommended Trailing Trigger: +{recommended_trigger/100.0:.4f} (+{recommended_trigger:.2f}%)")

if __name__ == '__main__':
    main()
