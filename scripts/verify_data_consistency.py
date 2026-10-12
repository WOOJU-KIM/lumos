import os
import sys
import json
import sqlite3
import urllib.request
from datetime import datetime
from pathlib import Path
import config
import pandas as pd
import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(r"c:\Users\chabo\OneDrive\바탕 화면\lumos")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from core.data_lake import MarketDataLake

def fetch_alpaca_bars_full(symbol: str, timeframe: str, start_dt: str, end_dt: str) -> pd.DataFrame:
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    all_bars = []
    page_token = None
    base_url = "https://data.alpaca.markets/v2/stocks/bars"
    
    # Alpaca timeframe formatting
    # 15m -> 15Min, 5m -> 5Min, 60m -> 1Hour
    tf_map = {"15m": "15Min", "5m": "5Min", "60m": "1Hour"}
    alpaca_tf = tf_map.get(timeframe, "15Min")
    
    while True:
        url = f"{base_url}?symbols={symbol}&timeframe={alpaca_tf}&start={start_dt}T00:00:00Z&end={end_dt}T23:59:59Z&feed=iex&adjustment=split&limit=1000"
        if page_token:
            url += f"&page_token={page_token}"
            
        req = urllib.request.Request(url, headers={
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
            "Accept": "application/json"
        })
        
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            bars = data.get("bars", {}).get(symbol, [])
            all_bars.extend(bars)
            page_token = data.get("next_page_token")
            if not page_token:
                break
                
    if not all_bars:
        return pd.DataFrame()
        
    rows = []
    for b in all_bars:
        utc_dt = pd.to_datetime(b["t"])
        ny_dt = utc_dt.tz_convert("America/New_York").tz_localize(None)
        
        rows.append({
            "datetime": ny_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "Open": float(b["o"]),
            "High": float(b["h"]),
            "Low": float(b["l"]),
            "Close": float(b["c"]),
            "Volume": float(b["v"])
        })
        
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df.set_index("datetime", inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    df.sort_index(inplace=True)
    return df

def compare_dataset():
    lake = MarketDataLake()
    symbols = [config.BASE_ASSET_LONG, "NVDA", "QQQ", config.MACRO_TREND_SYMBOL, config.BASE_ASSET_SHORT]
    timeframe = "15m"
    
    print("=" * 85)
    print("🔍 [Lumos DB vs Alpaca API] 동일 기간 데이터 정합성 정밀 검증")
    print(f"⏰ 검증 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S KST')}")
    print("=" * 85)
    
    results = []
    
    for sym in symbols:
        db_df = lake.load_candles(sym, timeframe)
        if db_df.empty:
            print(f"⚠️ {sym} DB 데이터 없음")
            continue
            
        start_date = db_df.index[0].strftime("%Y-%m-%d")
        end_date = db_df.index[-1].strftime("%Y-%m-%d")
        
        print(f"\n📡 [{sym} {timeframe}] Alpaca 데이터 호출 중... ({start_date} ~ {end_date})")
        alpaca_df = fetch_alpaca_bars_full(sym, timeframe, start_date, end_date)
        
        # Merge on exact datetime timestamp
        merged = pd.merge(
            db_df[['Open', 'High', 'Low', 'Close', 'Volume']],
            alpaca_df[['Open', 'High', 'Low', 'Close', 'Volume']],
            left_index=True,
            right_index=True,
            suffixes=('_db', '_alpaca'),
            how='inner'
        )
        
        total_db = len(db_df)
        total_alpaca = len(alpaca_df)
        overlap = len(merged)
        
        if overlap == 0:
            print(f"❌ {sym}: 일치하는 타임스탬프가 0개입니다.")
            continue
            
        # Metrics
        corr_close = merged['Close_db'].corr(merged['Close_alpaca'])
        corr_open = merged['Open_db'].corr(merged['Open_alpaca'])
        corr_ret = (merged['Close_db'].pct_change()).corr(merged['Close_alpaca'].pct_change())
        
        # Close Price Difference
        diff_close = (merged['Close_alpaca'] - merged['Close_db']).abs()
        pct_diff = (diff_close / merged['Close_db']) * 100.0
        
        mae = diff_close.mean()
        max_diff = diff_close.max()
        mape = pct_diff.mean()
        
        # Direction agreement (up/down match)
        db_dir = np.sign(merged['Close_db'] - merged['Open_db'])
        alpaca_dir = np.sign(merged['Close_alpaca'] - merged['Open_alpaca'])
        dir_match_rate = (db_dir == alpaca_dir).mean() * 100.0
        
        res_item = {
            "Symbol": sym,
            "DB 캔들수": total_db,
            "Alpaca 캔들수": total_alpaca,
            "일치 타임스탬프": overlap,
            "종가 상관계수": round(corr_close, 5),
            "수익률 상관계수": round(corr_ret, 5),
            "평균 오차(MAE)": f"${mae:.3f}",
            "평균 오차율(MAPE)": f"{mape:.2f}%",
            "캔들 방향 일치율": f"{dir_match_rate:.1f}%",
            "상태": "정합성 우수 ✅" if corr_close >= 0.999 and mape <= 0.50 else ("양호 ⚠️" if corr_close >= 0.99 else "불일치 ❌")
        }
        results.append(res_item)
        
        print(f"  • DB 캔들: {total_db:,}개 | Alpaca 캔들: {total_alpaca:,}개 | 공통 매칭: {overlap:,}개")
        print(f"  • 종가 상관계수 (Pearson): {corr_close:.5f} (수익률 상관: {corr_ret:.5f})")
        print(f"  • 평균 가격 괴리: ${mae:.3f} (오차율: {mape:.2f}%) | 캔들 음/양봉 일치율: {dir_match_rate:.1f}%")
        
        # Print first 2 and last 2 samples
        print("  [샘플 비교 (DB 종가 vs Alpaca 종가)]")
        sample_head = merged[['Close_db', 'Close_alpaca']].head(2)
        for idx, row in sample_head.iterrows():
            print(f"    {idx}: DB ${row['Close_db']:.2f} vs Alpaca ${row['Close_alpaca']:.2f} (차이: ${abs(row['Close_alpaca']-row['Close_db']):.2f})")
        sample_tail = merged[['Close_db', 'Close_alpaca']].tail(2)
        for idx, row in sample_tail.iterrows():
            print(f"    {idx}: DB ${row['Close_db']:.2f} vs Alpaca ${row['Close_alpaca']:.2f} (차이: ${abs(row['Close_alpaca']-row['Close_db']):.2f})")

    print("\n" + "=" * 85)
    print("📊 [종합 정합성 검증 요약표]")
    print("=" * 85)
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    
    return res_df

if __name__ == "__main__":
    compare_dataset()
