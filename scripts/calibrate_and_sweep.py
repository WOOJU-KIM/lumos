import os
import sys
import sqlite3
import config
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.heterogeneous_models import (
    OrderFlowImbalanceModel,
    TDATopologyModel,
    StateSpaceKalmanModel,
    CrossAssetDislocationModel
)
from core.ml_engine import MLFeatureEngine

# Load historical candles from SQLite
conn = sqlite3.connect('data/market_data.db')
long_df = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol=config.BASE_ASSET_LONG AND timeframe='15m' ORDER BY datetime", conn)
short_df = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol=config.BASE_ASSET_SHORT AND timeframe='15m' ORDER BY datetime", conn)
nvda_df = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol='NVDA' AND timeframe='15m' ORDER BY datetime", conn)
qqq_df = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol='QQQ' AND timeframe='15m' ORDER BY datetime", conn)
macro_df = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol=config.MACRO_TREND_SYMBOL AND timeframe='15m' ORDER BY datetime", conn)
vix_df = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol='^VIX' AND timeframe='15m' ORDER BY datetime", conn)
tnx_df = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol='^TNX' AND timeframe='15m' ORDER BY datetime", conn)
conn.close()

for df in [long_df, short_df, nvda_df, qqq_df, macro_df, vix_df, tnx_df]:
    df['Datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].str.slice(0, 10)
    df.set_index('Datetime', inplace=True)

unique_dates = sorted(long_df['date'].unique())
total_weeks = len(unique_dates) / 5.0

print(f"Loaded {len(long_df)} 15m candles over {len(unique_dates)} trading days ({total_weeks:.1f} weeks).")

class LumosV103MoEOrchestrator:
    """
    [Lumos v10.3 MoE AI 메타 오케스트레이터 (Track 6)]
    - Architecture: Gating Output = Sigmoid(W · R + b)
    - 6대 이종 전문가 풀 (Track 0 ~ Track 5)
    - 독립적 0.0 ~ 1.0 절대 확신도 산출 및 Top-1 임계치(T) 필터링
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
        self.track_labels = {
            "track0_gbdt": "Track 0: GBDT 시계열 스나이퍼",
            "track1_refresh": "Track 1: GBDT 일일 롤링 최신화",
            "track2_orderflow": "Track 2: 오더플로우 CVD 수급불균형",
            "track3_tda": "Track 3: TDA 위상수학 형태붕괴",
            "track4_statespace": "Track 4: 상태공간 칼만 동역학",
            "track5_cross_asset": "Track 5: 크로스에셋 인과괴리"
        }
        
        # Expert Sub-models
        self.m2 = OrderFlowImbalanceModel(delta_threshold=1.6, absorption_ratio=2.0)
        self.m3 = TDATopologyModel(entropy_threshold=0.70)
        self.m4 = StateSpaceKalmanModel()
        self.m5 = CrossAssetDislocationModel(dislocation_z_threshold=1.5)
        
        # Sigmoid Gating Parameters: W · R + b
        # Base Logits b: corresponding to prior base confidence around 0.62 ~ 0.72
        # b = ln(p / (1-p)): e.g. p=0.65 -> b=0.619, p=0.70 -> b=0.847, p=0.72 -> b=0.944
        self.b = np.array([0.75, 0.78, 0.70, 0.65, 0.72, 0.82], dtype=np.float64)
        
        # Weights W (6 x 5) for R = [elapsed_norm, vix_norm, atr_norm, cvd_norm, disloc_norm]
        self.W = np.array([
            # Track 0 (GBDT Champion): Prefers mid-day trending & stable ATR
            [0.55, -0.20, -0.30, 0.35, 0.20],
            # Track 1 (GBDT Refresh): Prefers mid-day trend & recent momentum
            [0.50, -0.15, -0.25, 0.40, 0.25],
            # Track 2 (Orderflow CVD): Prefers early open, high volume delta & high absorption
            [-0.70, 0.25, 0.50, 1.20, 0.15],
            # Track 3 (TDA Topology): Prefers high ATR volatility & topological phase disruption
            [-0.35, 0.30, 1.15, 0.25, 0.20],
            # Track 4 (Kalman State-Space): Prefers elevated VIX & noise filtering
            [0.20, 1.10, 0.35, 0.20, 0.25],
            # Track 5 (Cross-Asset): Prefers strong Lead-Lag dislocation (|disloc| >= 0.80)
            [0.30, 0.15, 0.25, 0.30, 1.30]
        ], dtype=np.float64)

    def compute_regime(
        self,
        b_idx: int,
        cur_vix: float,
        long_sub: pd.DataFrame,
        nvda_sub: pd.DataFrame,
        qqq_sub: pd.DataFrame
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        elapsed_min = float(b_idx * 15.0)
        elapsed_norm = (elapsed_min - 120.0) / 100.0
        vix_norm = (cur_vix - 17.0) / 4.0

        atr_ratio = 1.0
        if len(long_sub) >= 20:
            hl = long_sub['High'] - long_sub['Low']
            rec_atr = hl.tail(5).mean()
            avg_atr = hl.tail(20).mean()
            if avg_atr > 0:
                atr_ratio = rec_atr / avg_atr
        atr_norm = (atr_ratio - 1.0) / 0.4

        cvd_delta = 0.0
        if len(long_sub) >= 5:
            c = long_sub['Close']
            o = long_sub['Open']
            h = long_sub['High']
            l = long_sub['Low']
            v = long_sub['Volume']
            hl_diff = (h - l).replace(0, 0.001)
            v_delta = ((c - o) / hl_diff) * v
            avg_v = v.tail(15).mean() + 1e-6
            cvd_delta = float(v_delta.tail(3).mean() / avg_v)
        cvd_norm = np.clip(cvd_delta, -3.0, 3.0)

        dislocation_lag = 0.0
        if len(long_sub) >= 5 and len(nvda_sub) >= 5 and len(qqq_sub) >= 5:
            s_ret = (long_sub['Close'].iloc[-1] / long_sub['Close'].iloc[-5] - 1.0) * 100.0
            n_ret = (nvda_sub['Close'].iloc[-1] / nvda_sub['Close'].iloc[-5] - 1.0) * 100.0
            q_ret = (qqq_sub['Close'].iloc[-1] / qqq_sub['Close'].iloc[-5] - 1.0) * 100.0
            macro_exp = (n_ret * 0.6 + q_ret * 0.4) * 3.0
            dislocation_lag = float(macro_exp - s_ret)
        disloc_norm = np.clip(dislocation_lag / 1.5, -3.0, 3.0)

        R = np.array([elapsed_norm, vix_norm, atr_norm, cvd_norm, disloc_norm], dtype=np.float64)
        reg_dict = {
            "elapsed_min": elapsed_min,
            "vix_level": cur_vix,
            "atr_ratio": round(atr_ratio, 2),
            "cvd_delta": round(cvd_delta, 2),
            "dislocation_lag": round(dislocation_lag, 2)
        }
        return R, reg_dict

    def evaluate_gating(
        self,
        b_idx: int,
        cur_vix: float,
        long_sub: pd.DataFrame,
        nvda_sub: pd.DataFrame,
        qqq_sub: pd.DataFrame,
        threshold: float
    ) -> Dict[str, Any]:
        R, reg_dict = self.compute_regime(b_idx, cur_vix, long_sub, nvda_sub, qqq_sub)
        
        # Gating Output = Sigmoid(W · R + b)
        logits = self.W @ R + self.b
        scores_arr = 1.0 / (1.0 + np.exp(-logits))
        scores = {self.tracks[i]: float(scores_arr[i]) for i in range(len(self.tracks))}

        # Top-1 Selection
        top_track = max(scores, key=scores.get)
        top_gating_score = scores[top_track]

        # Expert Direction & Signal
        direction = "NONE"
        expert_conf = top_gating_score
        sig_code = 0

        try:
            if top_track == "track2_orderflow":
                cvd_df = self.m2.compute_cvd(long_sub)
                sig_code, conf, _ = self.m2.predict_signal(cvd_df.iloc[-1])
                direction = "LONG" if sig_code >= 0 else "SHORT"
            elif top_track == "track3_tda":
                sig_code, conf, _ = self.m3.predict_signal(long_sub)
                direction = "LONG" if sig_code >= 0 else "SHORT"
            elif top_track == "track4_statespace":
                sig_code, conf, _ = self.m4.predict_signal(long_sub['Close'].values)
                direction = "LONG" if sig_code >= 0 else "SHORT"
            elif top_track == "track5_cross_asset":
                s_ret = (long_sub['Close'].iloc[-1] / long_sub['Close'].iloc[-5] - 1.0)
                n_ret = (nvda_sub['Close'].iloc[-1] / nvda_sub['Close'].iloc[-5] - 1.0)
                q_ret = (qqq_sub['Close'].iloc[-1] / qqq_sub['Close'].iloc[-5] - 1.0)
                sig_code, conf, _ = self.m5.predict_signal(s_ret, n_ret, q_ret, 0.0, -0.01, -0.005)
                direction = "LONG" if sig_code >= 0 else "SHORT"
            else: # Track 0 / Track 1 (GBDT)
                c = long_sub['Close']
                ret_5 = float(c.iloc[-1] / c.iloc[-5] - 1.0) if len(c) >= 5 else 0.0
                if ret_5 >= 0.003:
                    direction = "LONG"
                    sig_code = 1
                elif ret_5 <= -0.003:
                    direction = "SHORT"
                    sig_code = -1
                else:
                    sig_code = 1 if ret_5 >= 0 else -1
        except Exception:
            sig_code = 1
            direction = "LONG"

        # Signal Approval: Top-1 Sigmoid Absolute Confidence >= Threshold T
        is_approved = bool(top_gating_score >= threshold and sig_code != 0)

        return {
            "top_track": top_track,
            "track_name": self.track_labels.get(top_track, top_track),
            "top_confidence": top_gating_score,
            "direction": direction,
            "is_approved": is_approved,
            "all_scores": scores,
            "regime": reg_dict
        }

def run_simulation():
    moe = LumosV103MoEOrchestrator()
    INITIAL_CAPITAL_KRW = 10_000_000
    FEE_RATE = 0.0030
    TAKE_PROFIT_PCT = 0.035
    STOP_LOSS_PCT = -0.020
    TIME_STOP_BARS = 6
    THRESHOLDS = [0.60, 0.70, 0.75, 0.80, 0.85]

    results = []

    for T in THRESHOLDS:
        capital = float(INITIAL_CAPITAL_KRW)
        trades = []
        equity = [capital]
        
        in_market = False
        current_pos = "NONE"
        entry_price = 0.0
        entry_time = ""
        bars_held = 0
        entry_track = ""
        entry_conf = 0.0

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

                sub_vix = vix_df[vix_df['datetime'] <= bar_dt]
                cur_vix = float(sub_vix.iloc[-1]['Close']) if not sub_vix.empty else 16.5
                long_sub = long_df[long_df['datetime'] <= bar_dt].tail(30)
                nvda_sub = nvda_df[nvda_df['datetime'] <= bar_dt].tail(30)
                qqq_sub = qqq_df[qqq_df['datetime'] <= bar_dt].tail(30)

                # 1. Exit Evaluation
                if in_market and current_pos != "NONE":
                    bars_held += 1
                    curr_row = row_l if current_pos == config.BASE_ASSET_LONG else row_s
                    curr_h = float(curr_row['High'])
                    curr_l = float(curr_row['Low'])
                    curr_c = float(curr_row['Close'])

                    h_ret = (curr_h - entry_price) / entry_price
                    l_ret = (curr_l - entry_price) / entry_price
                    c_ret = (curr_c - entry_price) / entry_price

                    exit_hit = False
                    actual_ret = 0.0
                    exit_reason = ""

                    if h_ret >= TAKE_PROFIT_PCT:
                        exit_hit = True
                        actual_ret = TAKE_PROFIT_PCT - FEE_RATE
                        exit_reason = "TAKE_PROFIT_+3.5%"
                    elif l_ret <= STOP_LOSS_PCT:
                        exit_hit = True
                        actual_ret = STOP_LOSS_PCT - FEE_RATE
                        exit_reason = "STOP_LOSS_-2.0%"
                    elif bars_held >= TIME_STOP_BARS:
                        exit_hit = True
                        actual_ret = c_ret - FEE_RATE
                        exit_reason = "TIME_STOP_90M"
                    elif b_idx == num_bars - 1:
                        exit_hit = True
                        actual_ret = c_ret - FEE_RATE
                        exit_reason = "EOD_CLOSE"

                    if exit_hit:
                        pnl_krw = int(capital * actual_ret)
                        capital += pnl_krw
                        equity.append(capital)
                        trades.append({
                            "entry_time": entry_time,
                            "exit_time": bar_dt,
                            "date": day_str,
                            "ticker": current_pos,
                            "pnl_pct": actual_ret * 100.0,
                            "pnl_krw": pnl_krw,
                            "exit_reason": exit_reason,
                            "bars_held": bars_held,
                            "track": entry_track,
                            "confidence": entry_conf,
                            "is_win": actual_ret > 0
                        })
                        in_market = False
                        current_pos = "NONE"
                        bars_held = 0
                        continue

                # 2. Entry Evaluation
                if not in_market and current_pos == "NONE":
                    # Time Guard: No entry at first bar (09:30) or after 14:30 NYT
                    if b_idx == 0 or time_str >= "14:30":
                        continue

                    gating_res = moe.evaluate_gating(
                        b_idx, cur_vix, long_sub, nvda_sub, qqq_sub, threshold=T
                    )

                    if gating_res["is_approved"]:
                        in_market = True
                        current_pos = config.BASE_ASSET_LONG if gating_res["direction"] == "LONG" else config.BASE_ASSET_SHORT
                        entry_row = row_l if current_pos == config.BASE_ASSET_LONG else row_s
                        entry_price = float(entry_row['Close'])
                        entry_time = bar_dt
                        bars_held = 0
                        entry_track = gating_res["track_name"]
                        entry_conf = gating_res["top_confidence"]

        # Metric calculations
        total_trades = len(trades)
        wins = sum(1 for t in trades if t['is_win'])
        losses = total_trades - wins
        win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
        avg_trades_per_week = total_trades / total_weeks

        total_pnl = int(capital - INITIAL_CAPITAL_KRW)
        cum_return = (total_pnl / INITIAL_CAPITAL_KRW) * 100.0

        gross_profit = sum(t['pnl_krw'] for t in trades if t['pnl_krw'] > 0)
        gross_loss = abs(sum(t['pnl_krw'] for t in trades if t['pnl_krw'] < 0))
        pf = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

        eq_arr = np.array(equity)
        peak = np.maximum.accumulate(eq_arr)
        dd = (eq_arr - peak) / peak
        mdd = round(abs(float(np.min(dd))) * 100.0, 2) if len(dd) > 0 else 0.0

        results.append({
            "threshold": T,
            "threshold_pct": f"{int(T*100)}%",
            "total_trades": total_trades,
            "avg_trades_per_week": round(avg_trades_per_week, 2),
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 1),
            "profit_factor": pf,
            "mdd": mdd,
            "cum_return": round(cum_return, 2),
            "final_capital": int(capital),
            "total_pnl": total_pnl
        })

    df_res = pd.DataFrame(results)
    print("\n" + "=" * 100)
    print(f"{'임계치 (T)':<10} | {'총 거래수':<8} | {'주당 평균 거래':<14} | {'전적':<10} | {'승률':<8} | {'Profit Factor':<14} | {'MDD':<8} | {'누적수익률':<10}")
    print("-" * 100)
    for _, r in df_res.iterrows():
        print(f"{r['threshold_pct']:<10} | {r['total_trades']:<8} | {r['avg_trades_per_week']:<14.2f} | {r['wins']}승 {r['losses']}패    | {r['win_rate']:>5.1f}%  | {r['profit_factor']:>13.2f} | {r['mdd']:>5.2f}% | {r['cum_return']:>+7.2f}%")
    print("=" * 100)
    return df_res

if __name__ == "__main__":
    run_simulation()
