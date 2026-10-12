import os
import sys
import time
import sqlite3
import requests
import config
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
}

SYMBOLS = [config.BASE_ASSET_LONG, config.BASE_ASSET_SHORT, config.MACRO_TREND_SYMBOL, "QQQ", "NVDA", "VIXY", "IEF"]

def fetch_alpaca_5m(symbol: str, start_iso: str, end_iso: str) -> pd.DataFrame:
    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
    all_bars = []
    page_token = None
    page_cnt = 0
    t0 = time.time()
    
    while True:
        params = {
            "timeframe": "5Min",
            "start": start_iso,
            "end": end_iso,
            "limit": 10000,
            "feed": "iex",
            "adjustment": "split"
        }
        if page_token:
            params["page_token"] = page_token
            
        res = requests.get(url, headers=HEADERS, params=params)
        if res.status_code != 200:
            print(f"⚠️ {symbol} 에러: {res.status_code} - {res.text}")
            break
            
        data = res.json()
        bars = data.get("bars", [])
        if not bars:
            break
            
        all_bars.extend(bars)
        page_cnt += 1
        page_token = data.get("next_page_token")
        if not page_token:
            break
            
    elapsed = time.time() - t0
    if not all_bars:
        print(f"⚠️ [{symbol}] 수집된 5분봉 없음")
        return pd.DataFrame()
        
    df = pd.DataFrame(all_bars)
    df['Datetime'] = pd.to_datetime(df['t']).dt.tz_convert('America/New_York').dt.tz_localize(None)
    df.rename(columns={
        'o': 'open',
        'h': 'high',
        'l': 'low',
        'c': 'close',
        'v': 'volume'
    }, inplace=True)
    
    # 정규장 (09:30 ~ 15:55 EDT)
    df['time_str'] = df['Datetime'].dt.strftime('%H:%M')
    df = df[(df['time_str'] >= '09:30') & (df['time_str'] <= '15:55')].copy()
    
    df['datetime'] = df['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df.sort_values('Datetime', inplace=True)
    df.drop_duplicates(subset=['datetime'], inplace=True)
    df['symbol'] = symbol
    df['timeframe'] = '5m'
    df['created_at'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"✅ [{symbol}] 5분봉 {len(df):,}개 수집 완료 ({page_cnt}페이지, {elapsed:.1f}초, 기간: {df['datetime'].iloc[0]} ~ {df['datetime'].iloc[-1]})")
    return df[['symbol', 'timeframe', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'created_at']]

def resample_to_15m(df_5m: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = df_5m.copy()
    df.index = pd.to_datetime(df['datetime'])
    resampled = []
    df['date'] = df.index.strftime('%Y-%m-%d')
    for d, grp in df.groupby('date'):
        for h in range(9, 16):
            for m in (0, 15, 30, 45):
                if h == 9 and m < 30:
                    continue
                if h == 15 and m > 45:
                    continue
                s_dt = pd.Timestamp(f"{d} {h:02d}:{m:02d}:00")
                e_dt = s_dt + pd.Timedelta(minutes=10)
                sub = grp.loc[s_dt:e_dt]
                if not sub.empty:
                    resampled.append({
                        'symbol': symbol,
                        'timeframe': '15m',
                        'datetime': s_dt.strftime('%Y-%m-%d %H:%M:%S'),
                        'open': sub['open'].iloc[0],
                        'high': sub['high'].max(),
                        'low': sub['low'].min(),
                        'close': sub['close'].iloc[-1],
                        'volume': sub['volume'].sum(),
                        'created_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
    return pd.DataFrame(resampled)

def resample_to_60m(df_5m: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = df_5m.copy()
    df.index = pd.to_datetime(df['datetime'])
    resampled = []
    df['date'] = df.index.strftime('%Y-%m-%d')
    for d, grp in df.groupby('date'):
        for h, m in [(9, 30), (10, 30), (11, 30), (12, 30), (13, 30), (14, 30), (15, 30)]:
            s_dt = pd.Timestamp(f"{d} {h:02d}:{m:02d}:00")
            e_dt = pd.Timestamp(f"{d} 15:55:00") if (h, m) == (15, 30) else (s_dt + pd.Timedelta(minutes=55))
            sub = grp.loc[s_dt:e_dt]
            if not sub.empty:
                resampled.append({
                    'symbol': symbol,
                    'timeframe': '60m',
                    'datetime': s_dt.strftime('%Y-%m-%d %H:%M:%S'),
                    'open': sub['open'].iloc[0],
                    'high': sub['high'].max(),
                    'low': sub['low'].min(),
                    'close': sub['close'].iloc[-1],
                    'volume': sub['volume'].sum(),
                    'created_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                })
    return pd.DataFrame(resampled)

def main():
    db_path = PROJECT_ROOT / "data" / "market_data.db"
    conn = sqlite3.connect(db_path)
    
    start_iso = "2020-09-15T09:30:00Z"
    end_iso = "2026-09-14T20:00:00Z"
    
    print("=" * 95)
    print(f"🚀 Alpaca 6개년(2020~2026) 5분봉(Split-Adjusted) 전량 수집 및 15m/60m 통합 동기화 시작")
    print(f"   대상 심볼: {', '.join(SYMBOLS)}")
    print("=" * 95)
    
    total_5m = 0
    total_15m = 0
    total_60m = 0
    t_start = time.time()
    cursor = conn.cursor()
    
    for sym in SYMBOLS:
        df_5m = fetch_alpaca_5m(sym, start_iso, end_iso)
        if not df_5m.empty:
            # 1. 5분봉 적재
            records_5m = df_5m[['symbol', 'timeframe', 'datetime', 'open', 'high', 'low', 'close', 'volume', 'created_at']].to_records(index=False).tolist()
            cursor.executemany("""
                INSERT OR REPLACE INTO market_candles (symbol, timeframe, datetime, open, high, low, close, volume, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records_5m)
            conn.commit()
            total_5m += len(records_5m)
            print(f"   💾 [{sym} 5m]  DB 적재: {len(records_5m):,}개")

            # 2. 15분봉 합성 및 적재
            df_15m = resample_to_15m(df_5m, sym)
            if not df_15m.empty:
                records_15m = df_15m.to_records(index=False).tolist()
                cursor.executemany("""
                    INSERT OR REPLACE INTO market_candles (symbol, timeframe, datetime, open, high, low, close, volume, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, records_15m)
                conn.commit()
                total_15m += len(records_15m)
                print(f"   💾 [{sym} 15m] DB 적재: {len(records_15m):,}개")

            # 3. 60분봉 합성 및 적재
            df_60m = resample_to_60m(df_5m, sym)
            if not df_60m.empty:
                records_60m = df_60m.to_records(index=False).tolist()
                cursor.executemany("""
                    INSERT OR REPLACE INTO market_candles (symbol, timeframe, datetime, open, high, low, close, volume, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, records_60m)
                conn.commit()
                total_60m += len(records_60m)
                print(f"   💾 [{sym} 60m] DB 적재: {len(records_60m):,}개")
            
    total_elapsed = time.time() - t_start
    print("=" * 95)
    print(f"🎉 6개년 데이터 완벽 동기화 완료! (소요시간: {total_elapsed:.1f}초)")
    print(f"   • 5분봉:  총 {total_5m:,}개")
    print(f"   • 15분봉: 총 {total_15m:,}개")
    print(f"   • 60분봉: 총 {total_60m:,}개")
    print("=" * 95)

if __name__ == "__main__":
    main()
