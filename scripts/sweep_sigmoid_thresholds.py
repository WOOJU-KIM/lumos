import os
import sys
import json
import sqlite3
import config
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List, Any, Optional

# UTF-8 setting
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.data_lake import MarketDataLake
from core.heterogeneous_models import (
    OrderFlowImbalanceModel,
    TDATopologyModel,
    StateSpaceKalmanModel,
    CrossAssetDislocationModel
)
from core.ml_engine import MLFeatureEngine

# Backtest Constants
INITIAL_CAPITAL_KRW = 10_000_000
FEE_RATE = 0.0030  # 0.25% broker fee + 0.05% slippage = 0.30%
TAKE_PROFIT_PCT = 0.035
STOP_LOSS_PCT = -0.020
TIME_STOP_BARS = 6
THRESHOLDS = [0.60, 0.70, 0.75, 0.80, 0.85]

class SigmoidGatingNetwork:
    """
    [Lumos v10.3 MoE Sigmoid 절대평가 게이팅 신경망 레이어]
    - Gating Output = Sigmoid(W · R + b)
    - 각 트랙(0~5)의 적합도를 독립적으로 0.0 ~ 1.0(0% ~ 100%) 사이의 절대 확신도로 출력
    - 6대 트랙:
      * Track 0: GBDT Pattern Sniper (실전 메인 챔피언)
      * Track 1: GBDT Rolling Refreshed (메인 최신화)
      * Track 2: Order Flow CVD Imbalance (수급 불균형)
      * Track 3: TDA Persistent Topology (위상수학 형태붕괴)
      * Track 4: State-Space Kalman Dynamical System (잠재 속도/가속도)
      * Track 5: Cross-Asset Lead-Lag Dislocation (NVDA/QQQ 인과괴리)
    """
    def __init__(self):
        self.tracks = [
            "track0_gbdt",
            "track1_refresh",
            "track2_orderflow",
            "track3_tda",
            "track4_statespace",
            "track5_cross_asset"
        ]
        self.track_names = {
            "track0_gbdt": "Track 0: GBDT 시계열 스나이퍼",
            "track1_refresh": "Track 1: GBDT 일일 롤링 최신화",
            "track2_orderflow": "Track 2: 오더플로우 CVD 수급불균형",
            "track3_tda": "Track 3: TDA 위상수학 형태붕괴",
            "track4_statespace": "Track 4: 상태공간 칼만 동역학",
            "track5_cross_asset": "Track 5: 크로스에셋 인과괴리"
        }
        
        # 가중치 행렬 W (6 x 5) 및 편향 벡터 b (6)
        # R = [elapsed_norm, vix_norm, atr_ratio_norm, cvd_delta_norm, dislocation_lag_norm]
        # Base logits bias to set typical baseline confidence around 0.50 ~ 0.70
        self.b = np.array([0.65, 0.68, 0.55, 0.50, 0.58, 0.62], dtype=np.float64)
        
        # Weights for each sensor: [elapsed, vix, atr_ratio, cvd_delta, dislocation_lag]
        self.W = np.array([
            # Track 0: Mid-day trend & stable ATR
            [0.45, -0.20, -0.30, 0.25, 0.20],
            # Track 1: Mid-day trend & recent momentum
            [0.40, -0.15, -0.25, 0.30, 0.25],
            # Track 2: Early morning (negative elapsed) & high CVD delta & high volume
            [-0.60, 0.20, 0.50, 1.40, 0.20],
            # Track 3: High ATR volatility & structural breakdown
            [-0.30, 0.35, 1.30, 0.30, 0.30],
            # Track 4: High VIX regime & hidden acceleration
            [0.10, 1.35, 0.40, 0.20, 0.30],
            # Track 5: High Dislocation lag (NVDA/QQQ vs TQQQ)
            [0.20, 0.10, 0.30, 0.30, 1.50]
        ], dtype=np.float64)

    def compute_sensor_vector(
        self,
        bar_idx_in_day: int,
        vix_px: float,
        long_sub_15m: pd.DataFrame,
        nvda_sub_15m: pd.DataFrame,
        qqq_sub_15m: pd.DataFrame
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """15분봉 시점 기준 5대 정규화 센서 벡터 R 산출"""
        elapsed_min = bar_idx_in_day * 15.0  # 0 ~ 390
        elapsed_norm = (elapsed_min - 120.0) / 100.0
        
        vix_level = float(vix_px) if vix_px > 0 else 16.5
        vix_norm = (vix_level - 17.0) / 4.0
        
        # ATR ratio
        atr_ratio = 1.0
        if len(long_sub_15m) >= 20:
            h = long_sub_15m['High']
            l = long_sub_15m['Low']
            hl = h - l
            rec_atr = hl.tail(5).mean()
            avg_atr = hl.tail(20).mean()
            if avg_atr > 0:
                atr_ratio = rec_atr / avg_atr
        atr_norm = (atr_ratio - 1.0) / 0.4
        
        # CVD Delta intensity
        cvd_delta = 0.0
        if len(long_sub_15m) >= 5:
            c = long_sub_15m['Close']
            o = long_sub_15m['Open']
            h = long_sub_15m['High']
            l = long_sub_15m['Low']
            v = long_sub_15m['Volume']
            hl_diff = (h - l).replace(0, 0.001)
            v_delta = ((c - o) / hl_diff) * v
            avg_v = v.tail(15).mean() + 1e-6
            cvd_delta = float(v_delta.tail(3).mean() / avg_v)
        cvd_norm = np.clip(cvd_delta, -3.0, 3.0)
        
        # Dislocation lag
        dislocation_lag = 0.0
        if len(long_sub_15m) >= 5 and len(nvda_sub_15m) >= 5 and len(qqq_sub_15m) >= 5:
            s_ret = (long_sub_15m['Close'].iloc[-1] / long_sub_15m['Close'].iloc[-5] - 1.0) * 100.0
            n_ret = (nvda_sub_15m['Close'].iloc[-1] / nvda_sub_15m['Close'].iloc[-5] - 1.0) * 100.0
            q_ret = (qqq_sub_15m['Close'].iloc[-1] / qqq_sub_15m['Close'].iloc[-5] - 1.0) * 100.0
            macro_exp = (n_ret * 0.6 + q_ret * 0.4) * 3.0
            dislocation_lag = float(macro_exp - s_ret)
        disloc_norm = np.clip(dislocation_lag / 1.5, -3.0, 3.0)
        
        R = np.array([elapsed_norm, vix_norm, atr_norm, cvd_norm, disloc_norm], dtype=np.float64)
        raw_regime = {
            "elapsed_min": elapsed_min,
            "vix_level": vix_level,
            "atr_ratio": round(atr_ratio, 2),
            "cvd_delta": round(cvd_delta, 2),
            "dislocation_lag": round(dislocation_lag, 2)
        }
        return R, raw_regime

    def predict_sigmoid_scores(self, R: np.ndarray) -> Dict[str, float]:
        """
        Sigmoid(W · R + b) ➔ 각 트랙의 독립적 절대 확신도 산출 (0.0 ~ 1.0)
        """
        logits = self.W @ R + self.b
        # Sigmoid function
        probs = 1.0 / (1.0 + np.exp(-logits))
        scores = {}
        for i, track_id in enumerate(self.tracks):
            scores[track_id] = round(float(probs[i]), 4)
        return scores


def run_full_threshold_backtest_sweep():
    print("=" * 80)
    print("🚀 [Lumos v10.3 MoE Sigmoid 절대평가 전환 및 임계치 최적화 전수 백테스트]")
    print("=" * 80)
    
    # 1. Load data from SQLite DB
    conn = sqlite3.connect('data/market_data.db')
    long_df = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol=config.BASE_ASSET_LONG AND timeframe='15m' ORDER BY datetime", conn)
    short_df = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol=config.BASE_ASSET_SHORT AND timeframe='15m' ORDER BY datetime", conn)
    nvda_df = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol='NVDA' AND timeframe='15m' ORDER BY datetime", conn)
    qqq_df = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol='QQQ' AND timeframe='15m' ORDER BY datetime", conn)
    vix_df = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol='^VIX' AND timeframe='15m' ORDER BY datetime", conn)
    tnx_df = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol='^TNX' AND timeframe='15m' ORDER BY datetime", conn)
    macro_df = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol=config.MACRO_TREND_SYMBOL AND timeframe='15m' ORDER BY datetime", conn)
    conn.close()

    for df in [long_df, short_df, nvda_df, qqq_df, vix_df, tnx_df, macro_df]:
        df['Datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = df['datetime'].str.slice(0, 10)
        df.set_index('Datetime', inplace=True)

    unique_dates = sorted(long_df['date'].unique())
    total_days = len(unique_dates)
    total_weeks = total_days / 5.0
    print(f"📊 [데이터 레이크] 총 {total_days}개 거래일 (약 {total_weeks:.1f}주), 15분봉 {len(long_df)}개 로드 완료")

    # Initialize models
    gating_net = SigmoidGatingNetwork()
    exp_orderflow = OrderFlowImbalanceModel()
    exp_tda = TDATopologyModel()
    exp_statespace = StateSpaceKalmanModel()
    exp_cross = CrossAssetDislocationModel()

    # Pre-calculate ML features for GBDT models
    ml_engine = MLFeatureEngine()
    long_feat = ml_engine.extract_features(long_df)
    short_feat = ml_engine.extract_features(short_df)
    
    # 2. Iterate through each threshold scenario
    scenario_results = []

    for T in THRESHOLDS:
        capital = float(INITIAL_CAPITAL_KRW)
        trades_list = []
        equity_curve = [capital]
        
        # Position tracking
        in_market = False
        current_pos = "NONE"
        entry_price = 0.0
        entry_bar_time = ""
        entry_bar_idx_in_day = 0
        bars_held = 0
        entry_expert = ""
        entry_confidence = 0.0

        for day_str in unique_dates:
            day_long = long_df[long_df['date'] == day_str].reset_index(drop=True)
            day_short = short_df[short_df['date'] == day_str].reset_index(drop=True)
            
            if len(day_long) < 5 or len(day_short) < 5:
                continue

            num_bars = min(len(day_long), len(day_short))

            for b_idx in range(num_bars):
                row_l = day_long.iloc[b_idx]
                row_s = day_short.iloc[b_idx]
                bar_dt = row_l['datetime']
                time_str = bar_dt.split(" ")[1][:5]
                
                # Check current VIX
                sub_vix = vix_df[vix_df['datetime'] <= bar_dt]
                cur_vix = float(sub_vix.iloc[-1]['Close']) if not sub_vix.empty else 16.5
                
                # Historical sub-windows for feature calculation
                long_sub = long_df[long_df['datetime'] <= bar_dt].tail(50)
                nvda_sub = nvda_df[nvda_df['datetime'] <= bar_dt].tail(50)
                qqq_sub = qqq_df[qqq_df['datetime'] <= bar_dt].tail(50)
                
                # 1. Evaluate Exit Conditions if in position
                if in_market and current_pos != "NONE":
                    bars_held += 1
                    curr_row = row_l if current_pos == config.BASE_ASSET_LONG else row_s
                    curr_high = float(curr_row['High'])
                    curr_low = float(curr_row['Low'])
                    curr_close = float(curr_row['Close'])
                    
                    high_ret = (curr_high - entry_price) / entry_price
                    low_ret = (curr_low - entry_price) / entry_price
                    close_ret = (curr_close - entry_price) / entry_price

                    exit_occurred = False
                    exit_reason = ""
                    actual_ret = 0.0

                    # Rule 1: Take Profit (+3.5%)
                    if high_ret >= TAKE_PROFIT_PCT:
                        exit_occurred = True
                        actual_ret = TAKE_PROFIT_PCT - FEE_RATE
                        exit_reason = "TAKE_PROFIT_+3.5%"
                    # Rule 2: Stop Loss (-2.0%)
                    elif low_ret <= STOP_LOSS_PCT:
                        exit_occurred = True
                        actual_ret = STOP_LOSS_PCT - FEE_RATE
                        exit_reason = "STOP_LOSS_-2.0%"
                    # Rule 3: 90min Time Stop (6 bars)
                    elif bars_held >= TIME_STOP_BARS:
                        exit_occurred = True
                        actual_ret = close_ret - FEE_RATE
                        exit_reason = f"TIME_STOP_{bars_held*15}M"
                    # Rule 4: EOD Forced Close (15:45 NYT)
                    elif b_idx == num_bars - 1:
                        exit_occurred = True
                        actual_ret = close_ret - FEE_RATE
                        exit_reason = "EOD_CLOSE"

                    if exit_occurred:
                        pnl_krw = int(capital * actual_ret)
                        capital += pnl_krw
                        equity_curve.append(capital)
                        trades_list.append({
                            "entry_time": entry_bar_time,
                            "exit_time": bar_dt,
                            "date": day_str,
                            "ticker": current_pos,
                            "entry_price": entry_price,
                            "exit_price": curr_close,
                            "pnl_pct": actual_ret * 100.0,
                            "pnl_krw": pnl_krw,
                            "exit_reason": exit_reason,
                            "bars_held": bars_held,
                            "expert": entry_expert,
                            "confidence": entry_confidence,
                            "is_win": actual_ret > 0
                        })
                        in_market = False
                        current_pos = "NONE"
                        bars_held = 0
                        continue

                # 2. Check Entry Conditions if not in market
                if not in_market and current_pos == "NONE":
                    # Time Guard: Do not enter on first bar (09:30) or after 14:30 NYT (90 min before close)
                    if b_idx == 0 or time_str >= "14:30":
                        continue

                    # Compute 5-sensor vector R
                    R, reg_dict = gating_net.compute_sensor_vector(b_idx, cur_vix, long_sub, nvda_sub, qqq_sub)
                    
                    # Sigmoid Gating scores
                    scores = gating_net.predict_sigmoid_scores(R)
                    
                    # Top-1 Expert Selection
                    top_track = max(scores, key=scores.get)
                    top_conf = scores[top_track]
                    
                    # Check Sigmoid Absolute Threshold: Top-1 Score >= T
                    if top_conf < T:
                        continue  # Signal Rejected by Sigmoid Absolute Threshold!

                    # Run the selected Top-1 Expert to determine direction
                    chosen_direction = "NONE"
                    expert_conf_val = top_conf

                    try:
                        if top_track == "track2_orderflow":
                            cvd_df = exp_orderflow.compute_cvd(long_sub)
                            sig_code, c_val, _ = exp_orderflow.predict_signal(cvd_df.iloc[-1])
                            chosen_direction = "LONG" if sig_code >= 0 else "SHORT"
                        elif top_track == "track3_tda":
                            sig_code, c_val, _ = exp_tda.predict_signal(long_sub)
                            chosen_direction = "LONG" if sig_code >= 0 else "SHORT"
                        elif top_track == "track4_statespace":
                            sig_code, c_val, _ = exp_statespace.predict_signal(long_sub['Close'].values)
                            chosen_direction = "LONG" if sig_code >= 0 else "SHORT"
                        elif top_track == "track5_cross_asset":
                            s_ret = (long_sub['Close'].iloc[-1] / long_sub['Close'].iloc[-5] - 1.0)
                            n_ret = (nvda_sub['Close'].iloc[-1] / nvda_sub['Close'].iloc[-5] - 1.0)
                            q_ret = (qqq_sub['Close'].iloc[-1] / qqq_sub['Close'].iloc[-5] - 1.0)
                            sig_code, c_val, _ = exp_cross.predict_signal(s_ret, n_ret, q_ret, 0.0, -0.01, -0.005)
                            chosen_direction = "LONG" if sig_code >= 0 else "SHORT"
                        else:  # track0_gbdt or track1_refresh
                            c = long_sub['Close']
                            ret_5 = float(c.iloc[-1] / c.iloc[-5] - 1.0) if len(c) >= 5 else 0.0
                            chosen_direction = "SHORT" if ret_5 < -0.004 else "LONG"
                    except Exception:
                        chosen_direction = "LONG"

                    if chosen_direction != "NONE":
                        # Enter Position!
                        in_market = True
                        current_pos = config.BASE_ASSET_LONG if chosen_direction == "LONG" else config.BASE_ASSET_SHORT
                        entry_row = row_l if current_pos == config.BASE_ASSET_LONG else row_s
                        entry_price = float(entry_row['Close'])
                        entry_bar_time = bar_dt
                        entry_bar_idx_in_day = b_idx
                        bars_held = 0
                        entry_expert = gating_net.track_names.get(top_track, top_track)
                        entry_confidence = top_conf

        # Metrics Calculation for threshold T
        total_trades = len(trades_list)
        wins = sum(1 for t in trades_list if t['is_win'])
        losses = total_trades - wins
        win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
        avg_trades_per_week = round(total_trades / total_weeks, 2)
        
        total_pnl = capital - INITIAL_CAPITAL_KRW
        cum_return = (total_pnl / INITIAL_CAPITAL_KRW) * 100.0
        
        # Profit Factor
        gross_profit = sum(t['pnl_krw'] for t in trades_list if t['pnl_krw'] > 0)
        gross_loss = abs(sum(t['pnl_krw'] for t in trades_list if t['pnl_krw'] < 0))
        pf = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
        
        # MDD
        eq_arr = np.array(equity_curve)
        peak = np.maximum.accumulate(eq_arr)
        dd = (eq_arr - peak) / peak
        mdd = round(abs(float(np.min(dd))) * 100.0, 2) if len(dd) > 0 else 0.0

        scenario_results.append({
            "threshold": T,
            "threshold_pct": f"{int(T*100)}%",
            "total_trades": total_trades,
            "avg_trades_per_week": avg_trades_per_week,
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 1),
            "profit_factor": pf,
            "mdd": mdd,
            "cum_return": round(cum_return, 2),
            "final_capital": int(capital),
            "total_pnl": int(total_pnl)
        })

    # Print Table
    df_results = pd.DataFrame(scenario_results)
    print("\n" + "=" * 95)
    print(f"{'임계치 (T)':<10} | {'총 거래수':<8} | {'주당 평균 거래':<14} | {'전적':<10} | {'승률':<8} | {'Profit Factor':<14} | {'MDD':<8} | {'누적수익률':<10}")
    print("-" * 95)
    for _, r in df_results.iterrows():
        print(f"{r['threshold_pct']:<10} | {r['total_trades']:<8} | {r['avg_trades_per_week']:<14.2f} | {r['wins']}승 {r['losses']}패    | {r['win_rate']:>5.1f}%  | {r['profit_factor']:>13.2f} | {r['mdd']:>5.2f}% | {r['cum_return']:>+7.2f}%")
    print("=" * 95)

    return df_results

if __name__ == "__main__":
    run_full_threshold_backtest_sweep()
