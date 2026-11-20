"""
SOXX 3-for-1 Stock Split Data Synchronization Script
Retroactively updates all historical candles for SOXX in market_data.db
using Alpaca IEX data with adjustment=split.
"""
import os
import sys
import json
import time
import sqlite3
import urllib.request
import pandas as pd
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

from config import DATA_DIR

DB_PATH = DATA_DIR / "market_data.db"
SYMBOL = "SOXX"
TIMEFRAMES = [
    ("60m", "1Hour"),
    ("15m", "15Min"),
    ("5m", "5Min"),
]

def fetch_alpaca_history(symbol: str, alpaca_tf: str, start_dt: str, end_dt: str) -> pd.DataFrame:
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not secret_key:
        raise RuntimeError("ALPACA_API_KEY or ALPACA_SECRET_KEY missing in .env")
        
    base_url = "https://data.alpaca.markets/v2/stocks/bars"
    all_bars = []
    page_token = None
    page_num = 0
    
    t0 = time.time()
    while True:
        page_num += 1
        url = (
            f"{base_url}?symbols={symbol}&timeframe={alpaca_tf}"
            f"&start={start_dt}T00:00:00Z&end={end_dt}T23:59:59Z"
            f"&feed=iex&adjustment=split&limit=10000"
        )
        if page_token:
            url += f"&page_token={page_token}"
            
        req = urllib.request.Request(url, headers={
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key,
            "Accept": "application/json"
        })
        
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                
            bars = data.get("bars", {}).get(symbol, [])
            all_bars.extend(bars)
            print(f"      [페이지 {page_num:2d}] +{len(bars):,}개 수신 (누적: {len(all_bars):,}개) [{time.time()-t0:.1f}s]")
            
            page_token = data.get("next_page_token")
            if not page_token:
                break
            time.sleep(0.1)
        except Exception as e:
            print(f"❌ [Alpaca 에러] {symbol} {alpaca_tf} (page {page_num}): {e}")
            break
            
    if not all_bars:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_bars)
    df = df.rename(columns={
        't': 'datetime',
        'o': 'open',
        'h': 'high',
        'l': 'low',
        'c': 'close',
        'v': 'volume'
    })
    
    # Convert UTC to US/Eastern (America/New_York)
    df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d %H:%M:%S')
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']].drop_duplicates(subset=['datetime']).sort_values('datetime')
    return df

def run_sync():
    print("=" * 80)
    print("🏛 [SOXX 3:1 액면분할 반영 데이터 전면 재동기화 (Self-Healing)]")
    print(f"⏰ 실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S KST')}")
    print(f"📁 타겟 DB: {DB_PATH}")
    print("=" * 80)
    
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    
    # 1. 기존 SOXX 데이터 현황 확인
    cur.execute("SELECT timeframe, count(*), min(datetime), max(datetime) FROM market_candles WHERE symbol = ? GROUP BY timeframe", (SYMBOL,))
    before_stats = cur.fetchall()
    print("\n📊 [현재 DB 내 SOXX 보관 현황 (분할 전 혼합 데이터)]:")
    for tf, count, min_dt, max_dt in before_stats:
        print(f"   • {tf:4s}: {count:7,d}개 ({min_dt} ~ {max_dt})")
        
    # 2. 기존 SOXX 레코드 삭제 (혼합 오염 방지)
    print("\n🧹 기존 SOXX 캔들 정리 (깨끗한 재적재 준비)...")
    cur.execute("DELETE FROM market_candles WHERE symbol = ?", (SYMBOL,))
    conn.commit()
    print("   ✅ 기존 SOXX 캔들 삭제 완료.")
    
    # 3. 각 타임프레임별 Alpaca 수정주가(adjustment=split) 전 기간 수집 및 삽입
    start_dt = "2020-01-01"
    end_dt = datetime.now().strftime("%Y-%m-%d")
    
    total_inserted = 0
    for db_tf, alp_tf in TIMEFRAMES:
        print(f"\n⏳ SOXX [{db_tf}] 전 기간 수정주가 수집 시작...")
        df = fetch_alpaca_history(SYMBOL, alp_tf, start_dt, end_dt)
        if df.empty:
            print(f"   ⚠️ SOXX [{db_tf}] 수집된 데이터 없음!")
            continue
            
        records = [
            (SYMBOL, db_tf, str(row['datetime']), float(row['open']), float(row['high']), float(row['low']), float(row['close']), float(row['volume']))
            for _, row in df.iterrows()
            if float(row['close']) > 0
        ]
        
        cur.executemany("""
            INSERT OR REPLACE INTO market_candles (symbol, timeframe, datetime, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """, records)
        
        start_candle = records[0][2]
        end_candle = records[-1][2]
        cur.execute("""
            INSERT INTO harvest_logs (harvest_date, symbol, timeframe, records_added, start_datetime, end_datetime, status)
            VALUES (?, ?, ?, ?, ?, ?, 'SUCCESS');
        """, (datetime.now().strftime('%Y-%m-%d'), SYMBOL, db_tf, len(records), start_candle, end_candle))
        
        conn.commit()
        print(f"   ✅ SOXX [{db_tf}]: 총 {len(records):,}개 캔들 DB 적재 완료! ({start_candle} ~ {end_candle})")
        total_inserted += len(records)
        
    # 4. 적재 후 검증
    cur.execute("SELECT timeframe, count(*), min(datetime), max(datetime) FROM market_candles WHERE symbol = ? GROUP BY timeframe", (SYMBOL,))
    after_stats = cur.fetchall()
    print("\n" + "=" * 80)
    print(f"🎉 SOXX 액면분할 반영 데이터 동기화 완료! (총 {total_inserted:,}개 캔들 적재)")
    print("📊 [신규 DB 내 SOXX 보관 현황 (수정주가 통일)]:")
    for tf, count, min_dt, max_dt in after_stats:
        print(f"   • {tf:4s}: {count:7,d}개 ({min_dt} ~ {max_dt})")
        
    # 5. 분할 경계일(2026-11-04 ~ 2026-11-06) 연속성 점검
    print("\n🔍 [2026년 11월 분할 전후 가격 연속성 점검 (5분봉)]:")
    cur.execute("""
        SELECT datetime, open, high, low, close, volume 
        FROM market_candles 
        WHERE symbol = 'SOXX' AND timeframe = '5m' AND datetime >= '2026-11-04 15:30:00' AND datetime <= '2026-11-05 10:00:00'
        ORDER BY datetime ASC
    """)
    rows = cur.fetchall()
    for dt, o, h, l, c, v in rows:
        print(f"   {dt} | Open: {o:6.2f} | Close: {c:6.2f} | High: {h:6.2f} | Low: {l:6.2f} | Vol: {v:8,.0f}")
        
    conn.close()
    print("=" * 80)

if __name__ == "__main__":
    run_sync()
