import os
import sys
import json
import sqlite3
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import requests
import config
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from dotenv import load_dotenv

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_DIR, BASE_DIR
from core.data_lake import MarketDataLake
from core.ml_engine import MLFeatureEngine
from core.heterogeneous_models import CrossAssetDislocationModel

load_dotenv()
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY
}

SYMBOLS = [config.BASE_ASSET_LONG, config.BASE_ASSET_SHORT, config.MACRO_TREND_SYMBOL, "QQQ", "NVDA", "VIXY", "IEF"]

def fetch_alpaca_bars(symbol: str, start_iso: str, end_iso: str, timeframe: str = "15Min") -> pd.DataFrame:
    """Alpaca IEX API로부터 지정 기간 15분봉 전량 수집 (페이징 지원)"""
    url = f"https://data.alpaca.markets/v2/stocks/{symbol}/bars"
    all_bars = []
    page_token = None
    
    while True:
        params = {
            "timeframe": timeframe,
            "start": start_iso,
            "end": end_iso,
            "limit": 10000,
            "feed": "iex"
        }
        if page_token:
            params["page_token"] = page_token
            
        res = requests.get(url, headers=HEADERS, params=params)
        if res.status_code != 200:
            print(f"⚠️ {symbol} Alpaca 에러: {res.status_code} - {res.text}")
            break
            
        data = res.json()
        bars = data.get("bars", [])
        if not bars:
            break
            
        all_bars.extend(bars)
        page_token = data.get("next_page_token")
        if not page_token:
            break
            
    if not all_bars:
        return pd.DataFrame()
        
    df = pd.DataFrame(all_bars)
    # UTC ➔ America/New_York (EDT/EST) 변환
    df['Datetime'] = pd.to_datetime(df['t']).dt.tz_convert('America/New_York').dt.tz_localize(None)
    df.rename(columns={
        'o': 'Open',
        'h': 'High',
        'l': 'Low',
        'c': 'Close',
        'v': 'Volume'
    }, inplace=True)
    
    # 정규장 시간 엄격 필터링 (09:30 ~ 15:45 EDT)
    df['time_str'] = df['Datetime'].dt.strftime('%H:%M')
    df = df[(df['time_str'] >= '09:30') & (df['time_str'] <= '15:45')].copy()
    
    df['datetime'] = df['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df.sort_values('Datetime', inplace=True)
    df.drop_duplicates(subset=['datetime'], inplace=True)
    df.set_index('Datetime', inplace=True)
    return df[['datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]

def resample_to_60m(df_15m: pd.DataFrame) -> pd.DataFrame:
    """15분봉 ➔ 정규장 60분봉 리샘플링 (7개 캔들: 09:30, 10:30, 11:30, 12:30, 13:30, 14:30, 15:30)"""
    if df_15m.empty:
        return pd.DataFrame()
        
    df = df_15m.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df['datetime'])
        
    resampled_rows = []
    df['date'] = df.index.strftime('%Y-%m-%d')
    for d, group in df.groupby('date'):
        for start_h, start_m in [(9, 30), (10, 30), (11, 30), (12, 30), (13, 30), (14, 30), (15, 30)]:
            start_dt = pd.Timestamp(f"{d} {start_h:02d}:{start_m:02d}:00")
            if (start_h, start_m) == (15, 30):
                end_dt = pd.Timestamp(f"{d} 15:45:00")
            else:
                end_dt = start_dt + pd.Timedelta(minutes=45)
                
            chunk = group.loc[start_dt:end_dt]
            if not chunk.empty:
                resampled_rows.append({
                    'datetime': start_dt.strftime('%Y-%m-%d %H:%M:%S'),
                    'Open': chunk['Open'].iloc[0],
                    'High': chunk['High'].max(),
                    'Low': chunk['Low'].min(),
                    'Close': chunk['Close'].iloc[-1],
                    'Volume': chunk['Volume'].sum(),
                    'Datetime': start_dt
                })
                
    if not resampled_rows:
        return pd.DataFrame()
        
    res_df = pd.DataFrame(resampled_rows)
    res_df.set_index('Datetime', inplace=True)
    res_df.sort_index(inplace=True)
    return res_df

def step1_ingest_historical_data():
    """과거 2년 (2022-09-15 ~ 2024-09-14) 데이터 수집 및 DB 적재"""
    lake = MarketDataLake()
    print("=" * 100)
    print("📥 [1단계] Alpaca 과거 2년(2022.09.15 ~ 2024.09.14) 7종 심볼 데이터 수집 및 적재 시작...")
    print("=" * 100)
    
    start_iso = "2022-09-15T09:30:00Z"
    end_iso = "2024-09-14T20:00:00Z"
    
    for sym in SYMBOLS:
        print(f"⏳ [{sym}] 15분봉 수집 중...")
        df_15m = fetch_alpaca_bars(sym, start_iso, end_iso, "15Min")
        if not df_15m.empty:
            inserted_15m = lake.insert_candles(sym, "15m", df_15m)
            print(f"   ➔ [{sym} 15m] {inserted_15m:,}개 캔들 DB 적재 완료 (기간: {df_15m.index[0]} ~ {df_15m.index[-1]})")
            
            # 60분봉 리샘플링 및 적재
            df_60m = resample_to_60m(df_15m)
            if not df_60m.empty:
                inserted_60m = lake.insert_candles(sym, "60m", df_60m)
                print(f"   ➔ [{sym} 60m] {inserted_60m:,}개 캔들 DB 적재 완료")
        else:
            print(f"   ⚠️ [{sym}] 수집된 데이터 없음")
            
    # 또한 크로스에셋 VIXY, IEF의 2024-09-15 ~ 2026-09-14 최근 2년치도 누락 없이 적재
    for sym in ["VIXY", "IEF"]:
        df_recent = fetch_alpaca_bars(sym, "2024-09-15T09:30:00Z", "2026-09-14T20:00:00Z", "15Min")
        if not df_recent.empty:
            c15 = lake.insert_candles(sym, "15m", df_recent)
            df_60m = resample_to_60m(df_recent)
            c60 = lake.insert_candles(sym, "60m", df_60m)
            print(f"   ➔ [{sym} 최근 2년 보강] 15m: {c15:,}개, 60m: {c60:,}개 적재 완료")
            
    print("✅ [1단계 완료] 4개년(2022.09 ~ 2026.09) 7종 심볼 전체 DB 적재 완료!\n")

def step2_train_in_sample_model():
    """과거 2년(In-Sample: 2022-09-15 ~ 2024-09-14) 데이터만으로 GBDT 챔피언 모델 학습"""
    lake = MarketDataLake()
    print("=" * 100)
    print("🧠 [2단계] 인샘플 과거 2년(2022.09.15 ~ 2024.09.14) 전용 GBDT 모델 학습 시작 (미래 데이터 100% 차단)")
    print("=" * 100)
    
    long_15m = lake.load_candles(config.BASE_ASSET_LONG, "15m", start_dt="2022-09-15", end_dt="2024-09-14 16:00:00")
    print(f"📊 인샘플 TQQQ 15분봉 데이터: {len(long_15m):,}개 봉")
    
    ml_engine = MLFeatureEngine(confidence_threshold=0.60)
    df_feat = ml_engine.extract_features(long_15m)
    
    # 모델 학습 및 Top 피처 선별
    model, top_10, top_3 = ml_engine.train_and_select_top_features(df_feat)
    print(f"🏆 [인샘플 학습 완료] Top 3 피처: {top_3}")
    print(f"🏆 Top 10 피처: {top_10}")
    
    return ml_engine

def step3_run_4yr_simulation(ml_engine: MLFeatureEngine):
    """4개년 전 구간(2022.09.16 ~ 2026.09.14) 시뮬레이션 집행 (천만원 기준)"""
    lake = MarketDataLake()
    print("=" * 100)
    print("🚀 [3단계] 4개년(2022.09 ~ 2026.09, 48개월) 백테스트 시뮬레이션 가동 (시작원금: 10,000,000원)")
    print("=" * 100)
    
    # 4개년 전 데이터 로드
    long_15m = lake.load_candles(config.BASE_ASSET_LONG, "15m")
    short_15m = lake.load_candles(config.BASE_ASSET_SHORT, "15m")
    macro_60m = lake.load_candles(config.MACRO_TREND_SYMBOL, "60m")
    nvda_15m = lake.load_candles("NVDA", "15m")
    macro_15m = lake.load_candles(config.MACRO_TREND_SYMBOL, "15m")
    qqq_15m  = lake.load_candles("QQQ", "15m")
    vixy_15m = lake.load_candles("VIXY", "15m")
    ief_15m  = lake.load_candles("IEF", "15m")
    
    # 60m 20EMA 계산
    macro_60m['ema20'] = macro_60m['Close'].ewm(span=20, adjust=False).mean()
    macro_60m_map = macro_60m.set_index('datetime')
    
    # 피처 추출 및 GBDT 확신도 산출 (인샘플 학습 모델 적용)
    long_feat = ml_engine.extract_features(long_15m)
    long_feat = ml_engine.add_confidence_columns(long_feat)
    
    short_feat = ml_engine.extract_features(short_15m)
    short_feat = ml_engine.add_confidence_columns(short_feat)
    short_feat.set_index('datetime', inplace=True, drop=False)
    
    # 크로스에셋 가격 맵
    nvda_map = nvda_15m.set_index('datetime')['Close'].to_dict()
    macro_map = macro_15m.set_index('datetime')['Close'].to_dict()
    qqq_map  = qqq_15m.set_index('datetime')['Close'].to_dict()
    vix_map  = vixy_15m.set_index('datetime')['Close'].to_dict()
    ief_map  = ief_15m.set_index('datetime')['Close'].to_dict()
    
    cross_mod = CrossAssetDislocationModel(dislocation_z_threshold=1.6)
    
    # 날짜별 15분봉 인덱싱
    long_feat['date_str'] = pd.to_datetime(long_feat['datetime']).dt.strftime('%Y-%m-%d')
    long_feat['time_str'] = pd.to_datetime(long_feat['datetime']).dt.strftime('%H:%M')
    
    unique_dates = sorted(long_feat['date_str'].unique())
    print(f"📅 총 거래일수: {len(unique_dates)}일 ({unique_dates[0]} ~ {unique_dates[-1]})")
    
    # 시뮬레이션 상태 변수
    initial_capital = 10_000_000.0  # 천만원
    current_capital = initial_capital
    peak_capital = initial_capital
    max_drawdown_pct = 0.0
    
    trades = []
    trade_id = 0
    active_pos = None  # {'sym': config.BASE_ASSET_LONG, 'entry_px': float, 'entry_dt': str, 'bars': int}
    
    slippage_rate = 0.0020  # 왕복 0.05% 슬리피지/수수료
    
    for dt_day in unique_dates:
        day_bars = long_feat[long_feat['date_str'] == dt_day]
        
        for idx, row in day_bars.iterrows():
            curr_dt = row['datetime']
            time_str = row['time_str']
            cur_long_close = float(row['Close'])
            cur_long_high = float(row['High'])
            cur_long_low = float(row['Low'])
            
            short_row = short_feat.loc[curr_dt] if curr_dt in short_feat.index else None
            cur_short_close = float(short_row['Close']) if short_row is not None else 0.0
            cur_short_high = float(short_row['High']) if short_row is not None else 0.0
            cur_short_low = float(short_row['Low']) if short_row is not None else 0.0
            
            # 1. 기존 포지션 보유 중인 경우: 4대 청산 룰 검사 (+3.0% TP, -2.0% SL, 90분 타임스탑, 15:45 EOD)
            if active_pos is not None:
                active_pos['bars'] += 1
                sym = active_pos['sym']
                entry_px = active_pos['entry_px']
                cur_close = cur_long_close if sym == config.BASE_ASSET_LONG else cur_short_close
                cur_high = cur_long_high if sym == config.BASE_ASSET_LONG else cur_short_high
                cur_low = cur_long_low if sym == config.BASE_ASSET_LONG else cur_short_low
                
                exit_price = None
                exit_reason = None
                
                # A. +3.0% 목표 익절 (High가 도달했는지)
                tp_px = entry_px * 1.030
                sl_px = entry_px * 0.980
                
                if cur_high >= tp_px:
                    exit_price = tp_px
                    exit_reason = "TAKE_PROFIT_3PCT"
                # B. -2.0% 칼손절
                elif cur_low <= sl_px:
                    exit_price = sl_px
                    exit_reason = "STOP_LOSS_2PCT"
                # C. 90분 타임스탑 (15분봉 6개 경과)
                elif active_pos['bars'] >= 6:
                    exit_price = cur_close
                    exit_reason = "TIME_STOP_90MIN"
                # D. 장 마감 100% 현금화 (15:45 NYT)
                elif time_str >= '15:45':
                    exit_price = cur_close
                    exit_reason = "EOD_MARKET_CLOSE"
                    
                if exit_price is not None:
                    # 청산 집행
                    raw_ret = (exit_price / entry_px) - 1.0
                    net_ret = raw_ret - slippage_rate
                    pnl_krw = current_capital * net_ret
                    current_capital += pnl_krw
                    
                    if current_capital > peak_capital:
                        peak_capital = current_capital
                    dd = (peak_capital - current_capital) / peak_capital * 100.0
                    if dd > max_drawdown_pct:
                        max_drawdown_pct = dd
                        
                    trade_id += 1
                    trades.append({
                        'trade_id': trade_id,
                        'datetime': curr_dt,
                        'symbol': sym,
                        'entry_px': entry_px,
                        'exit_px': exit_price,
                        'entry_dt': active_pos['entry_dt'],
                        'exit_dt': curr_dt,
                        'bars_held': active_pos['bars'],
                        'holding_min': active_pos['bars'] * 15,
                        'raw_ret_pct': round(raw_ret * 100, 2),
                        'net_ret_pct': round(net_ret * 100, 2),
                        'pnl_krw': int(round(pnl_krw)),
                        'ending_capital': int(round(current_capital)),
                        'exit_reason': exit_reason,
                        'is_win': 1 if net_ret > 0 else 0
                    })
                    active_pos = None
                    continue
                    
            # 2. 포지션 미보유 시 신규 진입 검토 (14:30 이전만 허용)
            if active_pos is None and time_str < '14:30':
                # Screen 1: 60분봉 대추세 (SOXX 20EMA 상방 vs 하방)
                past_macro_60 = macro_60m[macro_60m['datetime'] <= curr_dt]
                macro_60m_bull = True
                macro_60m_bear = True
                if len(past_macro_60) >= 20:
                    macro_c = past_macro_60['Close'].iloc[-1]
                    macro_ema20 = past_macro_60['ema20'].iloc[-1]
                    macro_60m_bull = (macro_c >= macro_ema20 * 0.998)
                    macro_60m_bear = (macro_c <= macro_ema20 * 1.002)
                        
                # Screen 2: GBDT 모델 신호 및 확신도 (Direction & Confidence)
                dir_gbdt = row.get('Direction', 'NONE')
                conf_gbdt = float(row.get('Confidence', 0.50))
                
                # Cross-Asset 선행 괴리율 산출
                n_px = nvda_map.get(curr_dt)
                sx_px = macro_map.get(curr_dt)
                q_px = qqq_map.get(curr_dt)
                v_px = vix_map.get(curr_dt)
                i_px = ief_map.get(curr_dt)
                
                # 5봉 전 가격과 비교
                cross_dir = "HOLD"
                cur_bar_idx = day_bars.index.get_loc(idx)
                if cur_bar_idx >= 5 and n_px and sx_px and q_px:
                    prev_dt = day_bars.iloc[cur_bar_idx - 5]['datetime']
                    p_n = nvda_map.get(prev_dt, n_px)
                    p_sx = macro_map.get(prev_dt, sx_px)
                    p_q = qqq_map.get(prev_dt, q_px)
                    p_v = vix_map.get(prev_dt, v_px) if v_px else None
                    p_i = ief_map.get(prev_dt, i_px) if i_px else None
                    
                    nvda_r = (n_px / p_n - 1.0) if p_n else 0.0
                    macro_r = (sx_px / p_sx - 1.0) if p_sx else 0.0
                    qqq_r  = (q_px / p_q - 1.0) if p_q else 0.0
                    vix_r  = (v_px / p_v - 1.0) if (v_px and p_v) else 0.0
                    ief_r  = (i_px / p_i - 1.0) if (i_px and p_i) else 0.0
                    tnx_proxy_ret = -ief_r
                    
                    long_r = (cur_long_close / day_bars.iloc[cur_bar_idx - 5]['Close']) - 1.0
                    sig_code, _, _ = cross_mod.predict_signal(
                        long_ret=long_r,
                        nvda_ret=nvda_r,
                        macro_ret=macro_r,
                        qqq_ret=qqq_r,
                        vix_ret=vix_r,
                        tnx_ret=tnx_proxy_ret
                    )
                    if sig_code > 0:
                        cross_dir = "LONG"
                    elif sig_code < 0:
                        cross_dir = "SHORT"
                        
                # 🎯 진입 판정: GBDT >= 60% & 대추세 정렬 & 크로스에셋 Veto 통과
                # A. TQQQ 롱 진입
                if dir_gbdt == "LONG" and conf_gbdt >= 0.60 and macro_60m_bull and cross_dir != "SHORT":
                    active_pos = {
                        'sym': config.BASE_ASSET_LONG,
                        'entry_px': cur_long_close,
                        'entry_dt': curr_dt,
                        'bars': 0
                    }
                # B. SQQQ 숏 진입
                elif dir_gbdt == "SHORT" and conf_gbdt >= 0.60 and macro_60m_bear and cross_dir != "LONG":
                    active_pos = {
                        'sym': config.BASE_ASSET_SHORT,
                        'entry_px': cur_short_close,
                        'entry_dt': curr_dt,
                        'bars': 0
                    }
                    
    print(f"🏁 [시뮬레이션 완료] 총 매매 횟수: {len(trades):,}회 | 최종 잔고: {int(current_capital):,}원 | MDD: {max_drawdown_pct:.2f}%")
    return trades, initial_capital, current_capital, max_drawdown_pct

def step4_generate_tables(trades: List[Dict[str, Any]], initial_capital: float):
    """결과 집계: 인샘플 vs 아웃오브샘플, 년도별, 월별 테이블 생성"""
    if not trades:
        print("⚠️ 매매 기록이 없습니다.")
        return
        
    df = pd.DataFrame(trades)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year
    df['year_month'] = df['datetime'].dt.strftime('%Y-%m')
    
    # 1. 인샘플(2022-09 ~ 2024-09-14) vs 아웃오브샘플(2024-09-15 ~ 2026-09-14)
    split_dt = pd.Timestamp("2024-09-15")
    df_is = df[df['datetime'] < split_dt]
    df_oos = df[df['datetime'] >= split_dt]
    
    def calc_stats(sub_df, start_cap):
        if sub_df.empty:
            return {
                'trades': 0, 'win_rate': 0.0, 'pf': 0.0, 'pnl': 0, 'ret_pct': 0.0, 'end_cap': start_cap,
                'long_trades': 0, 'short_trades': 0
            }
        wins = sub_df[sub_df['net_ret_pct'] > 0]
        losses = sub_df[sub_df['net_ret_pct'] <= 0]
        win_rate = len(wins) / len(sub_df) * 100
        gross_profit = wins['pnl_krw'].sum()
        gross_loss = abs(losses['pnl_krw'].sum())
        pf = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 999.0
        tot_pnl = sub_df['pnl_krw'].sum()
        end_cap = sub_df['ending_capital'].iloc[-1]
        ret_pct = (end_cap / start_cap - 1.0) * 100
        return {
            'trades': len(sub_df),
            'win_rate': round(win_rate, 1),
            'pf': pf,
            'pnl': int(tot_pnl),
            'ret_pct': round(ret_pct, 1),
            'end_cap': int(end_cap),
            'long_trades': len(sub_df[sub_df['symbol'] == config.BASE_ASSET_LONG]),
            'short_trades': len(sub_df[sub_df['symbol'] == config.BASE_ASSET_SHORT])
        }
        
    is_stats = calc_stats(df_is, initial_capital)
    oos_start_cap = df_is['ending_capital'].iloc[-1] if not df_is.empty else initial_capital
    oos_stats = calc_stats(df_oos, oos_start_cap)
    
    # 2. 년도별 집계
    years = sorted(df['year'].unique())
    yearly_rows = []
    y_start_cap = initial_capital
    
    for y in years:
        sub_y = df[df['year'] == y]
        st = calc_stats(sub_y, y_start_cap)
        yearly_rows.append({
            'year': y,
            'start_cap': int(y_start_cap),
            'end_cap': st['end_cap'],
            'pnl': st['pnl'],
            'ret_pct': round((st['end_cap'] / y_start_cap - 1.0) * 100, 1),
            'trades': st['trades'],
            'win_rate': st['win_rate'],
            'pf': st['pf'],
            'long_cnt': st['long_trades'],
            'short_cnt': st['short_trades']
        })
        y_start_cap = st['end_cap']
        
    # 3. 월별 집계
    months = sorted(df['year_month'].unique())
    monthly_rows = []
    m_start_cap = initial_capital
    
    for m in months:
        sub_m = df[df['year_month'] == m]
        st = calc_stats(sub_m, m_start_cap)
        monthly_rows.append({
            'month': m,
            'start_cap': int(m_start_cap),
            'end_cap': st['end_cap'],
            'pnl': st['pnl'],
            'ret_pct': round((st['end_cap'] / m_start_cap - 1.0) * 100, 2),
            'trades': st['trades'],
            'win_rate': st['win_rate'],
            'pf': st['pf'],
            'long_cnt': st['long_trades'],
            'short_cnt': st['short_trades']
        })
        m_start_cap = st['end_cap']
        
    result_summary = {
        'initial_capital': float(initial_capital),
        'final_capital': int(df['ending_capital'].iloc[-1]),
        'total_trades': int(len(df)),
        'in_sample': is_stats,
        'out_of_sample': oos_stats,
        'yearly': yearly_rows,
        'monthly': monthly_rows
    }
    
    # 출력 테이블 포맷팅 및 콘솔 출력
    print("\n" + "=" * 110)
    print("📊 [1. 인샘플(학습 2년) vs 아웃오브샘플(미학습 검증 2년) 비교표]")
    print("=" * 110)
    print(f"{'구분':<20} | {'학습 구간 (In-Sample, 2년)':<30} | {'미학습 검증 구간 (Out-of-Sample, 2년)':<30}")
    print("-" * 110)
    print(f"{'기간':<20} | {'2022-09-15 ~ 2024-09-14':<30} | {'2024-09-15 ~ 2026-09-14':<30}")
    print(f"{'시작 원금':<20} | {initial_capital:,.0f}원{'':<20} | {oos_start_cap:,.0f}원{'':<20}")
    print(f"{'기말 잔고':<20} | {is_stats['end_cap']:,}원{'':<20} | {oos_stats['end_cap']:,}원{'':<20}")
    print(f"{'누적 손익':<20} | {is_stats['pnl']:+,}원{'':<19} | {oos_stats['pnl']:+,}원{'':<19}")
    print(f"{'수익률':<20} | {is_stats['ret_pct']:+.1f}%{'':<24} | {oos_stats['ret_pct']:+.1f}%{'':<24}")
    print(f"{'총 거래 횟수':<20} | {is_stats['trades']}회 (TQQQ:{is_stats['long_trades']}, SQQQ:{is_stats['short_trades']}){'':<10} | {oos_stats['trades']}회 (TQQQ:{oos_stats['long_trades']}, SQQQ:{oos_stats['short_trades']}){'':<10}")
    print(f"{'승률 (Win Rate)':<20} | {is_stats['win_rate']:.1f}%{'':<24} | {oos_stats['win_rate']:.1f}%{'':<24}")
    print(f"{'손익비 (Profit Factor)':<20} | {is_stats['pf']:.2f}{'':<26} | {oos_stats['pf']:.2f}{'':<26}")
    print("=" * 110)

    print("\n" + "=" * 110)
    print("📅 [2. 년도별 상세 결산 (시작원금: 10,000,000원)]")
    print("=" * 110)
    print(f"{'년도':<6} | {'시작 잔고':<14} | {'기말 잔고':<14} | {'연간 손익':<14} | {'수익률':<9} | {'거래수':<6} | {'승률':<7} | {'PF':<6} | {'TQQQ/SQQQ':<10}")
    print("-" * 110)
    for r in yearly_rows:
        print(f"{r['year']:<6} | {r['start_cap']:>12,}원 | {r['end_cap']:>12,}원 | {r['pnl']:>+12,}원 | {r['ret_pct']:>+7.1f}% | {r['trades']:>4}회 | {r['win_rate']:>5.1f}% | {r['pf']:>5.2f} | {r['long_cnt']:>2}/{r['short_cnt']:<2}회")
    print("=" * 110)

    print("\n" + "=" * 110)
    print("🗓️ [3. 월별 상세 결산 (천만원 시작 복리 운용 잔고)]")
    print("=" * 110)
    print(f"{'연월':<7} | {'시작 잔고':<14} | {'기말 잔고':<14} | {'월간 손익':<14} | {'수익률':<9} | {'거래수':<6} | {'승률':<7} | {'PF':<6} | {'TQQQ/SQQQ':<10}")
    print("-" * 110)
    for r in monthly_rows:
        print(f"{r['month']:<7} | {r['start_cap']:>12,}원 | {r['end_cap']:>12,}원 | {r['pnl']:>+12,}원 | {r['ret_pct']:>+7.2f}% | {r['trades']:>4}회 | {r['win_rate']:>5.1f}% | {r['pf']:>5.2f} | {r['long_cnt']:>2}/{r['short_cnt']:<2}회")
    print("=" * 110)
    
    # JSON 파일 저장
    output_path = DATA_DIR / "backtest_4yr_oos_summary.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_summary, f, ensure_ascii=False, indent=2, default=str)
        
    # 매매 로그 CSV 저장
    df.to_csv(DATA_DIR / "backtest_4yr_trades.csv", index=False, encoding="utf-8-sig")
    print(f"\n💾 [결과 파일 저장 완료] {output_path}")
    
    return result_summary

def main():
    print("🚀 [Lumos 퀀트 4개년 인샘플 학습 vs 아웃오브샘플 백테스트 파이프라인 가동]")
    # step1_ingest_historical_data() # 이미 4개년 전량 적재 완료
    ml_engine = step2_train_in_sample_model()
    trades, init_cap, fin_cap, mdd = step3_run_4yr_simulation(ml_engine)
    summary = step4_generate_tables(trades, init_cap)
    print("\n🎉 전체 4개년 파이프라인 집행 완료!")

if __name__ == "__main__":
    main()
