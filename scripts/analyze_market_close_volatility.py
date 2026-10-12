import sqlite3
import sys
import config
import pandas as pd
import numpy as np

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

def run_analysis():
    conn = sqlite3.connect('data/market_data.db')
    
    # Analyze TQQQ 5m and 15m
    for tf in ['5m', '15m']:
        query = f"""
            SELECT symbol, datetime, open, high, low, close, volume
            FROM market_candles
            WHERE timeframe = '{tf}' AND symbol in (config.BASE_ASSET_LONG, 'QQQ', 'NVDA')
            ORDER BY datetime ASC
        """
        df = pd.read_sql_query(query, conn)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['time'] = df['datetime'].dt.strftime('%H:%M')
        df['date'] = df['datetime'].dt.strftime('%Y-%m-%d')
        
        # Calculate bar range % and abs return %
        df['range_pct'] = (df['high'] - df['low']) / df['open'] * 100.0
        df['body_pct'] = (df['close'] - df['open']).abs() / df['open'] * 100.0
        
        print(f"\n========================================================")
        print(f"📊 [{tf} 타임프레임 시간대별 변동성 및 거래량 정량 분석]")
        print(f"========================================================")
        
        for sym in [config.BASE_ASSET_LONG, 'QQQ', 'NVDA']:
            sub = df[df['symbol'] == sym].copy()
            if sub.empty:
                continue
            
            # Normalize volume by daily total volume
            daily_vol = sub.groupby('date')['volume'].transform('sum')
            sub['vol_share_pct'] = (sub['volume'] / (daily_vol + 1e-9)) * 100.0
            
            print(f"\n--- [{sym} ({tf})] 주요 세션 구간별 변동성 요약 ---")
            
            def get_bucket(t_str):
                if t_str < '10:30':
                    return '1. 개장 러시 (09:30~10:30)'
                elif t_str < '12:00':
                    return '2. 오전 완화 (10:30~12:00)'
                elif t_str < '14:00':
                    return '3. 점심 정체 (12:00~14:00)'
                elif t_str < '15:00':
                    return '4. 오후 재개 (14:00~15:00)'
                elif t_str < '15:30':
                    return '5. 파워아워 전반 (15:00~15:30)'
                else:
                    return '6. 장 마감 직전 (15:30~16:00)'
            
            sub['bucket'] = sub['time'].apply(get_bucket)
            b_summary = sub.groupby('bucket').agg(
                bars=('range_pct', 'count'),
                avg_range_pct=('range_pct', 'mean'),
                max_range_pct=('range_pct', 'max'),
                p90_range_pct=('range_pct', lambda x: np.percentile(x, 90)),
                avg_body_pct=('body_pct', 'mean'),
                extreme_1pct_prob=('range_pct', lambda x: (x >= 1.0).mean() * 100.0),
                extreme_2pct_prob=('range_pct', lambda x: (x >= 2.0).mean() * 100.0),
                avg_vol_share=('vol_share_pct', 'mean')
            ).reset_index()
            
            print(b_summary.to_string(index=False))

    # Detailed Close Breakdown for TQQQ 5m
    print("\n" + "="*80)
    print("🔬 [TQQQ 5분봉: 14:30 ~ 15:55 장 마감 시간대 5분별 초정밀 분석]")
    print("="*80)
    long_5m = pd.read_sql_query("""
        SELECT datetime, open, high, low, close, volume
        FROM market_candles
        WHERE timeframe = '5m' AND symbol = config.BASE_ASSET_LONG
        ORDER BY datetime ASC
    """, conn)
    long_5m['datetime'] = pd.to_datetime(long_5m['datetime'])
    long_5m['time'] = long_5m['datetime'].dt.strftime('%H:%M')
    long_5m['date'] = long_5m['datetime'].dt.strftime('%Y-%m-%d')
    long_5m['range_pct'] = (long_5m['high'] - long_5m['low']) / long_5m['open'] * 100.0
    long_5m['body_pct'] = (long_5m['close'] - long_5m['open']).abs() / long_5m['open'] * 100.0
    daily_vol = long_5m.groupby('date')['volume'].transform('sum')
    long_5m['vol_share_pct'] = (long_5m['volume'] / (daily_vol + 1e-9)) * 100.0
    
    close_long = long_5m[long_5m['time'] >= '14:30'].groupby('time').agg(
        avg_range=('range_pct', 'mean'),
        p90_range=('range_pct', lambda x: np.percentile(x, 90)),
        max_range=('range_pct', 'max'),
        avg_body=('body_pct', 'mean'),
        vol_share=('vol_share_pct', 'mean'),
        prob_over_1pct=('range_pct', lambda x: (x >= 1.0).mean() * 100.0),
        prob_over_1_5pct=('range_pct', lambda x: (x >= 1.5).mean() * 100.0),
        count=('range_pct', 'count')
    ).reset_index()
    print(close_long.to_string(index=False))

    # Also 15m TQQQ detailed breakdown for afternoon
    print("\n" + "="*80)
    print("🔬 [TQQQ 15분봉: 오후 및 장 마감 15분별 분석]")
    print("="*80)
    long_15m = pd.read_sql_query("""
        SELECT datetime, open, high, low, close, volume
        FROM market_candles
        WHERE timeframe = '15m' AND symbol = config.BASE_ASSET_LONG
        ORDER BY datetime ASC
    """, conn)
    long_15m['datetime'] = pd.to_datetime(long_15m['datetime'])
    long_15m['time'] = long_15m['datetime'].dt.strftime('%H:%M')
    long_15m['date'] = long_15m['datetime'].dt.strftime('%Y-%m-%d')
    long_15m['range_pct'] = (long_15m['high'] - long_15m['low']) / long_15m['open'] * 100.0
    long_15m['body_pct'] = (long_15m['close'] - long_15m['open']).abs() / long_15m['open'] * 100.0
    daily_vol_15 = long_15m.groupby('date')['volume'].transform('sum')
    long_15m['vol_share_pct'] = (long_15m['volume'] / (daily_vol_15 + 1e-9)) * 100.0
    
    close_long_15 = long_15m[long_15m['time'] >= '13:00'].groupby('time').agg(
        avg_range=('range_pct', 'mean'),
        p90_range=('range_pct', lambda x: np.percentile(x, 90)),
        max_range=('range_pct', 'max'),
        avg_body=('body_pct', 'mean'),
        vol_share=('vol_share_pct', 'mean'),
        prob_over_1_5pct=('range_pct', lambda x: (x >= 1.5).mean() * 100.0),
        prob_over_2pct=('range_pct', lambda x: (x >= 2.0).mean() * 100.0),
        count=('range_pct', 'count')
    ).reset_index()
    print(close_long_15.to_string(index=False))

if __name__ == '__main__':
    run_analysis()
