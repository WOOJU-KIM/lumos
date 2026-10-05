import os
import sys
import time
import json
import urllib.request
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

import config
from core.data_lake import MarketDataLake
from core.moe_orchestrator import train_and_save_moe_orchestrator, MoEMetaOrchestrator

def fetch_alpaca_clean_intraday(symbol: str, timeframe: str = "15Min", start_dt: str = "2024-09-15", end_dt: str = "2026-09-14") -> pd.DataFrame:
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    all_bars = []
    page_token = None
    base_url = "https://data.alpaca.markets/v2/stocks/bars"
    
    print(f"📥 [{symbol}] Alpaca {timeframe} 2년치 다운로드 시작 ({start_dt} ~ {end_dt})...")
    t0 = time.time()
    
    while True:
        url = f"{base_url}?symbols={symbol}&timeframe={timeframe}&start={start_dt}T00:00:00Z&end={end_dt}T23:59:59Z&feed=iex&adjustment=split&limit=10000"
        if page_token:
            url += f"&page_token={page_token}"
            
        req = urllib.request.Request(url, headers={
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key
        })
        
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode())
                bars = data.get("bars", {}).get(symbol, [])
                if not bars:
                    break
                all_bars.extend(bars)
                page_token = data.get("next_page_token")
                if not page_token:
                    break
        except Exception as e:
            print(f"❌ 데이터 수집 에러: {e}")
            time.sleep(1)
            
    df = pd.DataFrame(all_bars)
    if df.empty:
        return df
        
    df['t'] = pd.to_datetime(df['t'])
    df.set_index('t', inplace=True)
    df.index = df.index.tz_convert('America/New_York')
    df = df[['o', 'h', 'l', 'c', 'v']]
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    
    t1 = time.time()
    rows = []
    for b in all_bars:
        utc_dt = pd.to_datetime(b["t"])
        ny_dt = utc_dt.tz_convert("America/New_York").tz_localize(None)
        time_str = ny_dt.strftime("%H:%M")
        
        # 정규장 시간 엄격 필터링: 09:30 ~ 15:45 (16:00 전 마감 봉)
        if "09:30" <= time_str <= "15:45":
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
    print(f"  ➔ [{symbol}] 정규장(09:30~16:00) 정제 완료: {len(df):,}개 캔들 (총 {len(df.index.strftime('%Y-%m-%d').unique())}거래일)")
    return df

def resample_to_60m(df_15m: pd.DataFrame) -> pd.DataFrame:
    """15분봉으로부터 정규장 60분봉(09:30, 10:30, 11:30, 12:30, 13:30, 14:30, 15:30) 생성"""
    if df_15m.empty:
        return pd.DataFrame()
        
    # 날짜별 그룹 후 4개씩 또는 60분 단위 집계
    df = df_15m.copy()
    df['date'] = df.index.strftime('%Y-%m-%d')
    
    # 09:30~10:30, 10:30~11:30, 11:30~12:30, 12:30~13:30, 13:30~14:30, 14:30~15:30, 15:30~16:00
    resampled_rows = []
    for d, group in df.groupby('date'):
        # 1시간 단위 청크 분할 (4개 봉씩)
        chunks = [
            group.between_time('09:30', '10:15'),
            group.between_time('10:30', '11:15'),
            group.between_time('11:30', '12:15'),
            group.between_time('12:30', '13:15'),
            group.between_time('13:30', '14:15'),
            group.between_time('14:30', '15:15'),
            group.between_time('15:30', '15:45')
        ]
        start_hours = ['09:30:00', '10:30:00', '11:30:00', '12:30:00', '13:30:00', '14:30:00', '15:30:00']
        
        for chk, s_hr in zip(chunks, start_hours):
            if not chk.empty:
                resampled_rows.append({
                    'datetime': f"{d} {s_hr}",
                    'Open': float(chk['Open'].iloc[0]),
                    'High': float(chk['High'].max()),
                    'Low': float(chk['Low'].min()),
                    'Close': float(chk['Close'].iloc[-1]),
                    'Volume': float(chk['Volume'].sum())
                })
                
    df_60 = pd.DataFrame(resampled_rows)
    if not df_60.empty:
        df_60['datetime'] = pd.to_datetime(df_60['datetime'])
        df_60.set_index('datetime', inplace=True)
        df_60.sort_index(inplace=True)
    return df_60

def main():
    lake = MarketDataLake()
    symbols = config.ALL_SYMBOLS
    
    # 환경 변수에서 가져오거나, 없으면 기본 하드코딩 값 사용
    start_dt = os.getenv("LUMOS_ROLLING_START", "2024-09-15")
    end_dt = os.getenv("LUMOS_ROLLING_END", "2026-09-14")
    
    print("=" * 85)
    print("🚀 [Lumos 2년치 롤링 데이터 적재 및 최신 기준 AI 재학습 파이프라인]")
    print(f"▶ 실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S KST')}")
    print(f"▶ 타겟 기간: {start_dt} ~ {end_dt} (최근 2개년)")
    print(f"📦 대상 심볼: {', '.join(symbols)}")
    print("=" * 85)
    
    # 1. Alpaca 데이터 다운로드 및 DB 적재
    for sym in symbols:
        df_15m = fetch_alpaca_clean_intraday(sym, timeframe="15Min", start_dt=start_dt, end_dt=end_dt)
        if not df_15m.empty:
            ins_15 = lake.insert_candles(sym, "15m", df_15m)
            print(f"  💾 [{sym}] 15분봉 DB 적재 완료: {ins_15:,}개 캔들")
            
            # 60분봉 리샘플링 적재
            df_60m = resample_to_60m(df_15m)
            if not df_60m.empty:
                ins_60 = lake.insert_candles(sym, "60m", df_60m)
                print(f"  💾 [{sym}] 60분봉 DB 적재 완료: {ins_60:,}개 캔들")
                
    # 2. 2년치 TQQQ 15m 로드 및 GBDT 롤링 재학습 실행
    print("\n" + "=" * 85)
    print("🧠 [2단계: 2년치 롤링 데이터 기반 GBDT & MoE 챔피언 재학습 개시]")
    print("=" * 85)
    
    tqqq_15m_all = lake.load_candles("TQQQ", "15m")
    print(f"📊 [DB 누적 총 데이터] TQQQ 15m: 총 {len(tqqq_15m_all):,}개 캔들")
    print(f"   • 시작 시점: {tqqq_15m_all.index[0]}")
    print(f"   • 종료 시점: {tqqq_15m_all.index[-1]}")
    
    # 롤링 재학습 수행 및 모델 파일 저장
    orchestrator = train_and_save_moe_orchestrator(confidence_threshold=0.60, mode="hybrid_v3")
    
    print("\n✅ [학습 완료] 상위 10대 핵심 피처:")
    for rank, (feat, imp) in enumerate(zip(orchestrator.gbdt_engine.top_10_features, getattr(orchestrator.gbdt_engine.model, 'feature_importances_', [])[:10]), 1):
        print(f"   {rank:2d}. {feat}")
        
    print("\n" + "=" * 85)
    print("🎉 2년치 데이터 적재 및 현재 기준 모델 롤링 재학습 완벽 성공!")
    print("=" * 85)

if __name__ == "__main__":
    main()
