import sqlite3
import pandas as pd
import numpy as np

db_path = r'c:\Users\chabo\OneDrive\바탕 화면\lumos\data\market_data.db'
try:
    conn = sqlite3.connect(db_path)
    # Check if we have 1d data
    df_1d = pd.read_sql_query("SELECT * FROM market_candles WHERE symbol='QQQ' AND timeframe='1d'", conn)
    
    if len(df_1d) > 0:
        df = df_1d
        df['date_dt'] = pd.to_datetime(df['datetime'].str[:10])
    else:
        # aggregate from 15m
        df_15m = pd.read_sql_query("SELECT * FROM market_candles WHERE symbol='QQQ' AND timeframe='15m'", conn)
        df_15m['date'] = df_15m['datetime'].str[:10]
        df = df_15m.groupby('date').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        }).reset_index()
        df['date_dt'] = pd.to_datetime(df['date'])
        
    df['Year'] = df['date_dt'].dt.year
    df = df[df['Year'] >= 2023]
    
    df['Total_Range'] = df['high'] - df['low']
    df['Body'] = abs(df['close'] - df['open'])
    # prevent division by zero
    df['Body_Ratio'] = np.where(df['Total_Range'] > 0, df['Body'] / df['Total_Range'], 0)
    df['Ret_Pct'] = abs(df['close'] - df['open']) / df['open'] * 100.0
    
    # 1.0% 이상 움직이면서, 꼬리가 거의 없는(몸통 85% 이상) 원웨이 캔들
    full_bodied = df[(df['Ret_Pct'] >= 1.0) & (df['Body_Ratio'] >= 0.85)]
    
    summary = df.groupby('Year').agg(Total_Days=('Total_Range', 'count'))
    full_summary = full_bodied.groupby('Year').agg(Full_Bodied_Days=('Total_Range', 'count'))
    
    res = summary.join(full_summary).fillna(0)
    res['Percentage'] = (res['Full_Bodied_Days'] / res['Total_Days'] * 100).round(1)
    print("\n--- 일봉 기준 꽉 찬 캔들(원웨이 장세) 통계 ---")
    print(res)
    
except Exception as e:
    print('Error:', e)
