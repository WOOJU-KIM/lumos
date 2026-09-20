import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import random
import time
from pathlib import Path
from datetime import datetime
from scipy.stats import norm, skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.moe_orchestrator import MoEMetaOrchestrator
from config import DATA_DIR

class FastDataLake:
    def __init__(self, data_dict):
        self.data_dict = data_dict
    def load_candles(self, sym, timeframe, end_dt=None):
        df = self.data_dict.get((sym, timeframe))
        if df is None:
            return pd.DataFrame()
        if end_dt:
            df = df[df['datetime'] <= end_dt]
        return df
    def get_candles_with_live_tick(self, sym, timeframe, live_price):
        return self.load_candles(sym, timeframe)

def calculate_dsr(srs, best_sr, returns, n_trials):
    if len(returns) < 3 or n_trials < 2:
        return 0.0, 0.0

    sr_var = np.var(srs)
    sr_std = np.sqrt(sr_var) if sr_var > 0 else 0.001
    
    gamma = 0.5772156649
    z1 = norm.ppf(1 - 1.0 / n_trials)
    z2 = norm.ppf(1 - 1.0 / (n_trials * np.e))
    expected_max_sr = sr_std * ((1 - gamma) * z1 + gamma * z2)

    sk = skew(returns)
    ku = kurtosis(returns, fisher=False)
    n = len(returns)

    psr_std = np.sqrt((1 + 0.5 * best_sr**2 - sk * best_sr + ((ku - 3) / 4) * best_sr**2) / (n - 1))
    
    dsr = norm.cdf((best_sr - expected_max_sr) / psr_std)
    return dsr, expected_max_sr

def run_single_backtest(tqqq_df, data_15m, params, moe):
    INITIAL_CAPITAL = 10000.0
    CONF_TH = params['conf_th']
    TP_PCT = params['tp_pct']
    SL_PCT = params['sl_pct']
    TIME_STOP_BARS = params['time_stop']
    SLIPPAGE = 0.03
    FEE_RATE = 0.0020
    
    capital = INITIAL_CAPITAL
    equity_curve = [capital]
    returns = []
    
    unique_dates = sorted(tqqq_df['date_str'].unique())
    
    for d_str in unique_dates:
        day_tqqq = tqqq_df[tqqq_df['date_str'] == d_str].reset_index(drop=True)
        if len(day_tqqq) < 5:
            continue
            
        active_pos = None
        for b_idx in range(len(day_tqqq)):
            row = day_tqqq.iloc[b_idx]
            bar_time = row['datetime'].strftime('%H:%M')
            cur_c = row['Close']
            
            if active_pos is not None:
                buy_px = active_pos['buy_price']
                sym = active_pos['symbol']
                bars_held = b_idx - active_pos['entry_bar_idx']
                
                if sym == 'TQQQ':
                    cur_h = row['High']
                    cur_l = row['Low']
                else:
                    sq_df = data_15m['SQQQ'][data_15m['SQQQ']['datetime'] == row['datetime']]
                    if not sq_df.empty:
                        cur_h = sq_df['High'].iloc[0]
                        cur_l = sq_df['Low'].iloc[0]
                        cur_c = sq_df['Close'].iloc[0]
                    else:
                        cur_h = cur_l = cur_c = buy_px
                
                max_ret = (cur_h - buy_px) / buy_px
                min_ret = (cur_l - buy_px) / buy_px
                
                exit_price = 0
                if max_ret >= TP_PCT: exit_price = buy_px * (1 + TP_PCT) - SLIPPAGE
                elif min_ret <= SL_PCT: exit_price = buy_px * (1 + SL_PCT) - SLIPPAGE
                elif bars_held >= TIME_STOP_BARS: exit_price = cur_c - SLIPPAGE
                elif b_idx >= len(day_tqqq) - 1 or bar_time >= "15:45": exit_price = cur_c - SLIPPAGE
                
                if exit_price > 0:
                    pnl_amt = (active_pos['shares'] * (exit_price - buy_px)) - (active_pos['invested'] * FEE_RATE)
                    capital += pnl_amt
                    equity_curve.append(capital)
                    returns.append(pnl_amt / active_pos['invested'])
                    active_pos = None
                    
            if active_pos is None and bar_time <= "14:30":
                global_idx = tqqq_df[tqqq_df['datetime'] == row['datetime']].index[0]
                if global_idx >= 60:
                    tqqq_sub = tqqq_df.iloc[global_idx-59:global_idx+1]
                    cur_t_str = row['datetime'].strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Call evaluate
                    moe_res = moe.evaluate_dual_filter_signal(tqqq_sub, current_time_str=cur_t_str, threshold=CONF_TH)
                    
                    if moe_res.get("is_approved", False) and moe_res.get("direction") in ["LONG_TQQQ", "SHORT_SQQQ"]:
                        sym = "TQQQ" if moe_res["direction"] == "LONG_TQQQ" else "SQQQ"
                        base_px = cur_c
                        if sym == "SQQQ":
                            sq_df = data_15m['SQQQ'][data_15m['SQQQ']['datetime'] == row['datetime']]
                            if not sq_df.empty: base_px = sq_df['Close'].iloc[0]
                        
                        entry_px = base_px + SLIPPAGE
                        shares = int(capital / entry_px)
                        if shares > 0:
                            active_pos = {
                                'symbol': sym, 'buy_price': entry_px, 
                                'shares': shares, 'invested': shares * entry_px,
                                'entry_bar_idx': b_idx
                            }
    
    if len(returns) < 5:
        return 0.0, returns, capital
    
    ret_mean = np.mean(returns)
    ret_std = np.std(returns)
    if ret_std == 0:
        return 0.0, returns, capital
        
    sr = (ret_mean / ret_std) * np.sqrt(252 * 26)
    return sr, returns, capital

def main():
    print("🚀 [수축된 샤프지수(DSR) 극한 백테스트 시작 (Fast Memory Mode)]")
    db_path = DATA_DIR / "market_data.db"
    conn = sqlite3.connect(db_path)
    
    data_dict = {}
    data_15m = {}
    
    # Preload all data into memory
    print("메모리에 데이터를 캐싱 중입니다...")
    for s in ["TQQQ", "SQQQ", "NVDA", "QQQ", "^VIX", "SOXX"]:
        df = pd.read_sql_query(f"SELECT datetime, open, high, low, close, volume FROM market_candles WHERE symbol='{s}' AND timeframe='15m'", conn)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        # Sort by datetime to ensure proper slicing
        df.sort_values('datetime', inplace=True)
        data_dict[(s.replace("^VIX", "VIX"), "15m")] = df
        if s in ["TQQQ", "SQQQ"]:
            data_15m[s] = df
            
    for s in ["TQQQ", "NVDA", "QQQ"]:
        df = pd.read_sql_query(f"SELECT datetime, open, high, low, close, volume FROM market_candles WHERE symbol='{s}' AND timeframe='5m'", conn)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        df.sort_values('datetime', inplace=True)
        data_dict[(s, "5m")] = df
        
    df = pd.read_sql_query(f"SELECT datetime, open, high, low, close, volume FROM market_candles WHERE symbol='QQQ' AND timeframe='60m'", conn)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
    df.sort_values('datetime', inplace=True)
    data_dict[("QQQ", "60m")] = df
    conn.close()
    
    tqqq_df = data_15m["TQQQ"]
    tqqq_df = tqqq_df[tqqq_df['datetime'] >= "2026-08-01"].reset_index(drop=True)
    tqqq_df['date_str'] = tqqq_df['datetime'].dt.strftime('%Y-%m-%d')
    
    moe = MoEMetaOrchestrator()
    # OVERRIDE data_lake to use FastDataLake
    moe.data_lake = FastDataLake(data_dict)
    
    N_TRIALS = 15
    print(f"총 {N_TRIALS}개의 무작위 파라미터 조합으로 시뮬레이션 진행...")
    
    results = []
    best_sr = -999
    best_returns = []
    
    for i in range(N_TRIALS):
        start_t = time.time()
        params = {
            'conf_th': random.uniform(0.55, 0.85),
            'tp_pct': random.uniform(0.015, 0.06),
            'sl_pct': random.uniform(-0.04, -0.01),
            'time_stop': random.randint(4, 12)
        }
        
        sr, returns, cap = run_single_backtest(tqqq_df, data_15m, params, moe)
        results.append(sr)
        
        dur = time.time() - start_t
        print(f"[{i+1}/{N_TRIALS}] SR: {sr:.2f} | Params: TH={params['conf_th']:.2f}, TP={params['tp_pct']*100:.1f}% | 소요: {dur:.1f}초")
        
        if sr > best_sr:
            best_sr = sr
            best_returns = returns
            
    # DSR 산출
    dsr, e_max_sr = calculate_dsr(results, best_sr, best_returns, N_TRIALS)
    
    print("\n" + "="*80)
    print("🏆 [Deflated Sharpe Ratio 극한 백테스트 결과]")
    print(f"   • 시뮬레이션 횟수(N): {N_TRIALS}회")
    print(f"   • 최고 Sharpe Ratio: {best_sr:.2f}")
    print(f"   • 기대 최고 패널티(E[max_SR]): {e_max_sr:.2f}")
    print(f"   • 수축된 샤프 지수(DSR): {dsr*100:.2f}%")
    print("="*80)
    if dsr >= 0.95:
        print("DSR 95% 이상: 통계적으로 매우 유의미. 과최적화가 아닙니다!")
    elif dsr >= 0.50:
        print("DSR 50% 이상: 통계적으로 유의미한 편이나 패널티 적용됨.")
    else:
        print("DSR 50% 미만: 다중 검정에 의한 선택 편향 의심.")

if __name__ == "__main__":
    main()
