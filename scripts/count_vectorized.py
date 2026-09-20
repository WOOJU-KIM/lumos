import os
import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.data_lake import MarketDataLake
from core.ml_engine import MLFeatureEngine

def main():
    print("🚀 전체 데이터 벡터화 분석 시작 (어제 ~ 1주일/1년 전)")
    
    data_lake = MarketDataLake()
    ml_engine = MLFeatureEngine()
    
    # 1. 데이터 로드 (전체)
    df_all = data_lake.load_candles("TQQQ", "15m")
    df_all['dt'] = pd.to_datetime(df_all['datetime'])
    
    # 어제 날짜 구하기 (2026-09-17)
    end_date = pd.Timestamp("2026-09-17 23:59:59")
    start_date_1mo = end_date - pd.DateOffset(weeks=1)
    start_date_1yr = end_date - pd.DateOffset(years=1)
    
    # 2. 전체 데이터에 대해 한 번에 피처 추출 (벡터화)
    # df_all이 크더라도 pandas 연산은 통째로 하는게 훨씬 빠름
    print("⏳ 피처 엔지니어링 수행 중...")
    df_feat = ml_engine.extract_features(df_all)
    df_feat = ml_engine.add_confidence_columns(df_feat)
    
    # df_feat에 dt 복원
    df_feat['dt'] = df_all['dt']
    
    # 3. 1주일치 결과 집계
    df_1mo = df_feat[(df_feat['dt'] >= start_date_1mo) & (df_feat['dt'] <= end_date)]
    df_1mo = df_1mo.set_index('dt').between_time("09:30", "15:45").reset_index()
    
    count_1mo_long = len(df_1mo[(df_1mo['Signal'] == 1) & (df_1mo['Confidence'] >= 0.60)])
    count_1mo_short = len(df_1mo[(df_1mo['Signal'] == -1) & (df_1mo['Confidence'] >= 0.60)])
    
    print("\n==========================================")
    print("📊 1주일치 매매 시그널 분석 결과 (GBDT 확신도 60% 이상 기준)")
    print(f"기간: {start_date_1mo.date()} ~ {end_date.date()}")
    print("==========================================")
    print(f"🔴 LONG (TQQQ 매수) : {count_1mo_long} 회")
    print(f"🔵 SHORT (SQQQ 매수): {count_1mo_short} 회")
    print(f"⚪ NONE (관망/거절) : {len(df_1mo) - count_1mo_long - count_1mo_short} 회")
    print("==========================================")
    
    # 4. 1년치 결과 집계
    df_1yr = df_feat[(df_feat['dt'] >= start_date_1yr) & (df_feat['dt'] <= end_date)]
    df_1yr = df_1yr.set_index('dt').between_time("09:30", "15:45").reset_index()
    
    count_1yr_long = len(df_1yr[(df_1yr['Signal'] == 1) & (df_1yr['Confidence'] >= 0.60)])
    count_1yr_short = len(df_1yr[(df_1yr['Signal'] == -1) & (df_1yr['Confidence'] >= 0.60)])
    
    print("\n==========================================")
    print("📊 1년치 매매 시그널 분석 결과 (GBDT 확신도 60% 이상 기준)")
    print(f"기간: {start_date_1yr.date()} ~ {end_date.date()}")
    print("==========================================")
    print(f"🔴 LONG (TQQQ 매수) : {count_1yr_long} 회")
    print(f"🔵 SHORT (SQQQ 매수): {count_1yr_short} 회")
    print(f"⚪ NONE (관망/거절) : {len(df_1yr) - count_1yr_long - count_1yr_short} 회")
    print("==========================================")

if __name__ == "__main__":
    main()
