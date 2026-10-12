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

from config import DATA_DIR, BASE_DIR, MODELS_DIR
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

def step1_ingest_2020_to_2022():
    """과거 4~6년 전 (2020-09-15 ~ 2022-09-14) 7종 심볼 데이터 수집 및 DB 적재"""
    lake = MarketDataLake()
    print("=" * 100)
    print("📥 [1단계] Alpaca 과거 2020.09.15 ~ 2022.09.14 (2개년) 7종 심볼 데이터 수집 및 적재 시작...")
    print("=" * 100)
    
    start_iso = "2020-09-15T09:30:00Z"
    end_iso = "2022-09-14T20:00:00Z"
    
    for sym in SYMBOLS:
        print(f"⏳ [{sym}] 15분봉 수집 중...")
        df_15m = fetch_alpaca_bars(sym, start_iso, end_iso, "15Min")
        if not df_15m.empty:
            inserted_15m = lake.insert_candles(sym, "15m", df_15m)
            print(f"   ➔ [{sym} 15m] {inserted_15m:,}개 캔들 DB 적재 완료 (기간: {df_15m.index[0]} ~ {df_15m.index[-1]})")
            
            df_60m = resample_to_60m(df_15m)
            if not df_60m.empty:
                inserted_60m = lake.insert_candles(sym, "60m", df_60m)
                print(f"   ➔ [{sym} 60m] {inserted_60m:,}개 캔들 DB 적재 완료")
        else:
            print(f"   ⚠️ [{sym}] 수집 데이터 없음")
            
    print("✅ [1단계 완료] 6개년(2020.09 ~ 2026.09) 7종 심볼 전체 DB 적재 완료!\n")

def step2_train_recent_model():
    """[최근 2년치(2024-09-15 ~ 2026-09-14) 데이터로 GBDT 실전 모델 학습]"""
    lake = MarketDataLake()
    print("=" * 100)
    print("🧠 [2단계] 최근 2년치(2024.09.15 ~ 2026.09.14) 실전 모델 학습 시작 (앞으로 실전에 쓸 최신 모델)")
    print("=" * 100)
    
    long_15m = lake.load_candles(config.BASE_ASSET_LONG, "15m", start_dt="2024-09-15", end_dt="2026-09-14 16:00:00")
    print(f"📊 최근 2년 TQQQ 15분봉 데이터: {len(long_15m):,}개 봉")
    
    ml_engine = MLFeatureEngine(confidence_threshold=0.60)
    df_feat = ml_engine.extract_features(long_15m)
    
    # 모델 학습 및 Top 피처 선별
    model, top_10, top_3 = ml_engine.train_and_select_top_features(df_feat)
    print(f"🏆 [최근 2년 모델 학습 완료] Top 3 피처: {top_3}")
    print(f"🏆 Top 10 피처: {top_10}")
    
    # 실전 파일로 동시 저장 (앞으로 쓸 챔피언 모델 갱신)
    joblib.dump(model, MODELS_DIR / "model_champion.pkl")
    print("💾 [실전 모델 갱신 완료] models/model_champion.pkl 저장 완료")
    
    return ml_engine

def step3_run_6yr_simulation(ml_engine: MLFeatureEngine):
    """6개년 전 구간(2020.09.15 ~ 2026.09.14) 시뮬레이션 집행 (천만원 기준)"""
    lake = MarketDataLake()
    print("=" * 100)
    print("🚀 [3단계] 6개년(2020.09 ~ 2026.09, 73개월) 백테스트 시뮬레이션 가동 (시작원금: 10,000,000원)")
    print("=" * 100)
    
    long_15m = lake.load_candles(config.BASE_ASSET_LONG, "15m")
    short_15m = lake.load_candles(config.BASE_ASSET_SHORT, "15m")
    macro_60m = lake.load_candles(config.MACRO_TREND_SYMBOL, "60m")
    nvda_15m = lake.load_candles("NVDA", "15m")
    macro_15m = lake.load_candles(config.MACRO_TREND_SYMBOL, "15m")
    qqq_15m  = lake.load_candles("QQQ", "15m")
    vixy_15m = lake.load_candles("VIXY", "15m")
    ief_15m  = lake.load_candles("IEF", "15m")
    
    macro_60m['ema20'] = macro_60m['Close'].ewm(span=20, adjust=False).mean()
    
    # 피처 추출 및 GBDT 확신도 산출 (최근 2년 학습 모델 적용)
    long_feat = ml_engine.extract_features(long_15m)
    long_feat = ml_engine.add_confidence_columns(long_feat)
    
    nvda_map = nvda_15m.set_index('datetime')['Close'].to_dict()
    macro_map = macro_15m.set_index('datetime')['Close'].to_dict()
    qqq_map  = qqq_15m.set_index('datetime')['Close'].to_dict()
    vix_map  = vixy_15m.set_index('datetime')['Close'].to_dict()
    ief_map  = ief_15m.set_index('datetime')['Close'].to_dict()
    
    short_15m_dict = short_15m.set_index('datetime').to_dict(orient='index')
    
    cross_mod = CrossAssetDislocationModel(dislocation_z_threshold=1.6)
    
    long_feat['date_str'] = pd.to_datetime(long_feat['datetime']).dt.strftime('%Y-%m-%d')
    long_feat['time_str'] = pd.to_datetime(long_feat['datetime']).dt.strftime('%H:%M')
    
    unique_dates = sorted(long_feat['date_str'].unique())
    print(f"📅 총 거래일수: {len(unique_dates)}일 ({unique_dates[0]} ~ {unique_dates[-1]})")
    
    initial_capital = 10_000_000.0
    current_capital = initial_capital
    peak_capital = initial_capital
    max_drawdown_pct = 0.0
    
    trades = []
    trade_id = 0
    active_pos = None
    slippage_rate = 0.0020
    
    for dt_day in unique_dates:
        day_bars = long_feat[long_feat['date_str'] == dt_day]
        
        for idx, row in day_bars.iterrows():
            curr_dt = row['datetime']
            time_str = row['time_str']
            cur_long_close = float(row['Close'])
            cur_long_high = float(row['High'])
            cur_long_low = float(row['Low'])
            
            short_row = short_15m_dict.get(curr_dt)
            cur_short_close = float(short_row['Close']) if short_row else 0.0
            cur_short_high = float(short_row['High']) if short_row else 0.0
            cur_short_low = float(short_row['Low']) if short_row else 0.0
            
            # 1. 포지션 보유 중인 경우 청산 검사
            if active_pos is not None:
                active_pos['bars'] += 1
                sym = active_pos['sym']
                entry_px = active_pos['entry_px']
                cur_close = cur_long_close if sym == config.BASE_ASSET_LONG else cur_short_close
                cur_high = cur_long_high if sym == config.BASE_ASSET_LONG else cur_short_high
                cur_low = cur_long_low if sym == config.BASE_ASSET_LONG else cur_short_low
                
                exit_price = None
                exit_reason = None
                
                tp_px = entry_px * 1.030
                sl_px = entry_px * 0.980
                
                if cur_high >= tp_px:
                    exit_price = tp_px
                    exit_reason = "TAKE_PROFIT_3PCT"
                elif cur_low <= sl_px:
                    exit_price = sl_px
                    exit_reason = "STOP_LOSS_2PCT"
                elif active_pos['bars'] >= 6:
                    exit_price = cur_close
                    exit_reason = "TIME_STOP_90MIN"
                elif time_str >= '15:45':
                    exit_price = cur_close
                    exit_reason = "EOD_MARKET_CLOSE"
                    
                if exit_price is not None:
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
                    
            # 2. 포지션 미보유 시 신규 진입 검토
            if active_pos is None and time_str < '14:30':
                past_macro_60 = macro_60m[macro_60m['datetime'] <= curr_dt]
                macro_60m_bull = True
                macro_60m_bear = True
                if len(past_macro_60) >= 20:
                    macro_c = past_macro_60['Close'].iloc[-1]
                    macro_ema20 = past_macro_60['ema20'].iloc[-1]
                    macro_60m_bull = (macro_c >= macro_ema20 * 0.998)
                    macro_60m_bear = (macro_c <= macro_ema20 * 1.002)
                    
                dir_gbdt = row.get('Direction', 'NONE')
                conf_gbdt = float(row.get('Confidence', 0.50))
                
                n_px = nvda_map.get(curr_dt)
                sx_px = macro_map.get(curr_dt)
                q_px = qqq_map.get(curr_dt)
                v_px = vix_map.get(curr_dt)
                i_px = ief_map.get(curr_dt)
                
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
                        
                if dir_gbdt == "LONG" and conf_gbdt >= 0.60 and macro_60m_bull and cross_dir != "SHORT":
                    active_pos = {
                        'sym': config.BASE_ASSET_LONG,
                        'entry_px': cur_long_close,
                        'entry_dt': curr_dt,
                        'bars': 0
                    }
                elif dir_gbdt == "SHORT" and conf_gbdt >= 0.60 and macro_60m_bear and cross_dir != "LONG":
                    if cur_short_close > 0:
                        active_pos = {
                            'sym': config.BASE_ASSET_SHORT,
                            'entry_px': cur_short_close,
                            'entry_dt': curr_dt,
                            'bars': 0
                        }
                        
    print(f"🏁 [시뮬레이션 완료] 총 매매: {len(trades):,}회 | 최종 잔고: {int(current_capital):,}원 | MDD: {max_drawdown_pct:.2f}%")
    return trades, initial_capital, current_capital, max_drawdown_pct

def step4_generate_3split_tables(trades: List[Dict[str, Any]], initial_capital: float):
    """결과 집계: 2/2/2 3분할 비교표, 년도별, 월별 테이블 생성"""
    if not trades:
        print("⚠️ 매매 기록이 없습니다.")
        return
        
    df = pd.DataFrame(trades)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['year'] = df['datetime'].dt.year
    df['year_month'] = df['datetime'].dt.strftime('%Y-%m')
    
    # 2/2/2 분할 기준일
    dt_split1 = pd.Timestamp("2022-09-15")
    dt_split2 = pd.Timestamp("2024-09-15")
    
    df_p1 = df[df['datetime'] < dt_split1]
    df_p2 = df[(df['datetime'] >= dt_split1) & (df['datetime'] < dt_split2)]
    df_p3 = df[df['datetime'] >= dt_split2]
    
    def calc_stats(sub_df, start_cap):
        if sub_df.empty:
            return {
                'trades': 0, 'win_rate': 0.0, 'pf': 0.0, 'pnl': 0, 'ret_pct': 0.0, 'end_cap': int(start_cap),
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
        
    p1_stats = calc_stats(df_p1, initial_capital)
    p2_start_cap = df_p1['ending_capital'].iloc[-1] if not df_p1.empty else initial_capital
    p2_stats = calc_stats(df_p2, p2_start_cap)
    p3_start_cap = df_p2['ending_capital'].iloc[-1] if not df_p2.empty else p2_start_cap
    p3_stats = calc_stats(df_p3, p3_start_cap)
    
    # 년도별 집계
    years = sorted(df['year'].unique())
    yearly_rows = []
    y_start_cap = initial_capital
    for y in years:
        sub_y = df[df['year'] == y]
        st = calc_stats(sub_y, y_start_cap)
        yearly_rows.append({
            'year': int(y),
            'start_cap': int(y_start_cap),
            'end_cap': int(st['end_cap']),
            'pnl': int(st['pnl']),
            'ret_pct': round((st['end_cap'] / y_start_cap - 1.0) * 100, 1),
            'trades': int(st['trades']),
            'win_rate': float(st['win_rate']),
            'pf': float(st['pf']),
            'long_cnt': int(st['long_trades']),
            'short_cnt': int(st['short_trades'])
        })
        y_start_cap = st['end_cap']
        
    # 월별 집계
    months = sorted(df['year_month'].unique())
    monthly_rows = []
    m_start_cap = initial_capital
    for m in months:
        sub_m = df[df['year_month'] == m]
        st = calc_stats(sub_m, m_start_cap)
        monthly_rows.append({
            'month': str(m),
            'start_cap': int(m_start_cap),
            'end_cap': int(st['end_cap']),
            'pnl': int(st['pnl']),
            'ret_pct': round((st['end_cap'] / m_start_cap - 1.0) * 100, 2),
            'trades': int(st['trades']),
            'win_rate': float(st['win_rate']),
            'pf': float(st['pf']),
            'long_cnt': int(st['long_trades']),
            'short_cnt': int(st['short_trades'])
        })
        m_start_cap = st['end_cap']
        
    result_summary = {
        'initial_capital': float(initial_capital),
        'final_capital': int(df['ending_capital'].iloc[-1]),
        'total_trades': int(len(df)),
        'period_1_oos_2020_2022': p1_stats,
        'period_2_oos_2022_2024': p2_stats,
        'period_3_is_2024_2026': p3_stats,
        'yearly': yearly_rows,
        'monthly': monthly_rows
    }
    
    # 콘솔 출력
    print("\n" + "=" * 115)
    print("📊 [1. 과거 6년부터 2/2/2 (3분할) 비교표: 최근 2년 학습 vs 과거 4년 OOS 검증]")
    print("=" * 115)
    print(f"{'구분':<16} | {'구간 1: 2020-09~2022-09 (OOS 2)':<28} | {'구간 2: 2022-09~2024-09 (OOS 1)':<28} | {'구간 3: 2024-09~2026-09 (학습)':<28}")
    print("-" * 115)
    print(f"{'데이터 성격':<16} | {'완전 미학습 (유동성/폭락장)':<28} | {'완전 미학습 (바닥/반등장)':<28} | {'최근 2년 학습 (실전 운용 모델)':<28}")
    print(f"{'시작 잔고':<16} | {initial_capital:>15,.0f}원{'':<10} | {p2_start_cap:>15,.0f}원{'':<10} | {p3_start_cap:>15,.0f}원{'':<10}")
    print(f"{'기말 잔고':<16} | {p1_stats['end_cap']:>15,}원{'':<10} | {p2_stats['end_cap']:>15,}원{'':<10} | {p3_stats['end_cap']:>15,}원{'':<10}")
    print(f"{'구간 손익':<16} | {p1_stats['pnl']:>+15,}원{'':<10} | {p2_stats['pnl']:>+15,}원{'':<10} | {p3_stats['pnl']:>+15,}원{'':<10}")
    print(f"{'구간 수익률':<16} | {p1_stats['ret_pct']:>+14.1f}%{'':<11} | {p2_stats['ret_pct']:>+14.1f}%{'':<11} | {p3_stats['ret_pct']:>+14.1f}%{'':<11}")
    print(f"{'거래수 (L/S)':<16} | {p1_stats['trades']}회 (L:{p1_stats['long_trades']}/S:{p1_stats['short_trades']}){'':<10} | {p2_stats['trades']}회 (L:{p2_stats['long_trades']}/S:{p2_stats['short_trades']}){'':<10} | {p3_stats['trades']}회 (L:{p3_stats['long_trades']}/S:{p3_stats['short_trades']}){'':<10}")
    print(f"{'승률 (Win Rate)':<16} | {p1_stats['win_rate']:>14.1f}%{'':<11} | {p2_stats['win_rate']:>14.1f}%{'':<11} | {p3_stats['win_rate']:>14.1f}%{'':<11}")
    print(f"{'손익비 (PF)':<16} | {p1_stats['pf']:>14.2f}{'':<12} | {p2_stats['pf']:>14.2f}{'':<12} | {p3_stats['pf']:>14.2f}{'':<12}")
    print("=" * 115)

    print("\n" + "=" * 115)
    print("📅 [2. 년도별 상세 결산 (시작원금: 10,000,000원 복리 운용 잔고)]")
    print("=" * 115)
    print(f"{'년도':<6} | {'시작 잔고':<16} | {'기말 잔고':<16} | {'연간 손익':<16} | {'수익률':<9} | {'거래수':<6} | {'승률':<7} | {'PF':<6} | {'TQQQ/SQQQ':<10}")
    print("-" * 115)
    for r in yearly_rows:
        print(f"{r['year']:<6} | {r['start_cap']:>14,}원 | {r['end_cap']:>14,}원 | {r['pnl']:>+14,}원 | {r['ret_pct']:>+7.1f}% | {r['trades']:>4}회 | {r['win_rate']:>5.1f}% | {r['pf']:>5.2f} | {r['long_cnt']:>2}/{r['short_cnt']:<2}회")
    print("=" * 115)

    print("\n" + "=" * 115)
    print("🗓️ [3. 월별 상세 결산 (천만원 시작 복리 운용 잔고)]")
    print("=" * 115)
    print(f"{'연월':<7} | {'시작 잔고':<16} | {'기말 잔고':<16} | {'월간 손익':<16} | {'수익률':<9} | {'거래수':<6} | {'승률':<7} | {'PF':<6} | {'TQQQ/SQQQ':<10}")
    print("-" * 115)
    for r in monthly_rows:
        print(f"{r['month']:<7} | {r['start_cap']:>14,}원 | {r['end_cap']:>14,}원 | {r['pnl']:>+14,}원 | {r['ret_pct']:>+7.2f}% | {r['trades']:>4}회 | {r['win_rate']:>5.1f}% | {r['pf']:>5.2f} | {r['long_cnt']:>2}/{r['short_cnt']:<2}회")
    print("=" * 115)
    
    # 파일 저장
    output_path = DATA_DIR / "backtest_6yr_3split_summary.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_summary, f, ensure_ascii=False, indent=2, default=str)
        
    df.to_csv(DATA_DIR / "backtest_6yr_3split_trades.csv", index=False, encoding="utf-8-sig")
    print(f"\n💾 [6개년 결과 파일 저장 완료] {output_path}")
    
    return result_summary

def main():
    print("🚀 [Lumos 퀀트 6개년(2020~2026) 2/2/2 3분할 백테스트 파이프라인 가동]")
    step1_ingest_2020_to_2022()
    ml_engine = step2_train_recent_model()
    trades, init_cap, fin_cap, mdd = step3_run_6yr_simulation(ml_engine)
    summary = step4_generate_3split_tables(trades, init_cap)
    print("\n🎉 전체 6개년 2/2/2 파이프라인 집행 완료!")

if __name__ == "__main__":
    main()
