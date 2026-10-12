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

INITIAL_CAPITAL_KRW = 10_000_000
FEE_RATE = 0.0030  # 0.25% broker fee + 0.05% slippage = 0.30%
TAKE_PROFIT_PCT = 0.035
STOP_LOSS_PCT = -0.020
TIME_STOP_BARS = 6
THRESHOLDS = [0.60, 0.70, 0.75, 0.80, 0.85]

class MoESigmoidOrchestratorV103:
    """
    [Lumos v10.3 MoE AI 메타 오케스트레이터 - Sigmoid 절대평가 게이팅 네트워크]
    - Gating Output = Sigmoid(W · R + b)
    - 6대 이종 전문가 풀:
      * Track 0: GBDT Pattern Sniper (실전 메인 챔피언)
      * Track 1: GBDT Rolling Refreshed (메인 최신화)
      * Track 2: Order Flow CVD Imbalance (수급 불균형)
      * Track 3: TDA Persistent Topology (위상수학 형태붕괴)
      * Track 4: State-Space Kalman Dynamical System (잠재 동역학)
      * Track 5: Cross-Asset Lead-Lag Dislocation (NVDA/QQQ/TNX 인과괴리)
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
        self.experts = {
            "track0_gbdt": MLFeatureEngine(confidence_threshold=0.60),
            "track1_refresh": MLFeatureEngine(confidence_threshold=0.60),
            "track2_orderflow": OrderFlowImbalanceModel(delta_threshold=1.8, absorption_ratio=2.2),
            "track3_tda": TDATopologyModel(entropy_threshold=0.72),
            "track4_statespace": StateSpaceKalmanModel(),
            "track5_cross_asset": CrossAssetDislocationModel(dislocation_z_threshold=1.6)
        }

        # Sigmoid Gating Network Weights W (6 x 5) & bias b (6)
        # R = [elapsed_norm, vix_norm, atr_ratio_norm, cvd_delta_norm, dislocation_lag_norm]
        self.b = np.array([0.45, 0.48, 0.40, 0.35, 0.42, 0.46], dtype=np.float64)
        self.W = np.array([
            [0.55, -0.25, -0.35, 0.20, 0.15],
            [0.50, -0.20, -0.30, 0.25, 0.20],
            [-0.75, 0.25, 0.60, 1.65, 0.15],
            [-0.40, 0.40, 1.45, 0.25, 0.25],
            [0.15, 1.50, 0.45, 0.20, 0.25],
            [0.25, 0.15, 0.35, 0.25, 1.60]
        ], dtype=np.float64)

    def compute_regime_vector(
        self,
        b_idx: int,
        vix_px: float,
        long_sub: pd.DataFrame,
        nvda_sub: pd.DataFrame,
        qqq_sub: pd.DataFrame,
        macro_sub: pd.DataFrame,
        tnx_sub: pd.DataFrame
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        elapsed_min = float(b_idx * 15.0)
        elapsed_norm = (elapsed_min - 120.0) / 100.0
        
        vix_level = float(vix_px) if vix_px > 0 else 16.5
        vix_norm = (vix_level - 17.0) / 4.0

        atr_ratio = 1.0
        if len(long_sub) >= 20:
            h = long_sub['High']
            l = long_sub['Low']
            hl = h - l
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
            "vix_level": vix_level,
            "atr_ratio": round(atr_ratio, 2),
            "cvd_delta": round(cvd_delta, 2),
            "dislocation_lag": round(dislocation_lag, 2)
        }
        return R, reg_dict

    def evaluate_signal(
        self,
        b_idx: int,
        vix_px: float,
        long_sub: pd.DataFrame,
        nvda_sub: pd.DataFrame,
        qqq_sub: pd.DataFrame,
        macro_sub: pd.DataFrame,
        tnx_sub: pd.DataFrame,
        threshold: float
    ) -> Dict[str, Any]:
        R, reg_dict = self.compute_regime_vector(b_idx, vix_px, long_sub, nvda_sub, qqq_sub, macro_sub, tnx_sub)
        
        # Sigmoid Gating Output = Sigmoid(W · R + b)
        logits = self.W @ R + self.b
        probs = 1.0 / (1.0 + np.exp(-logits))
        scores = {self.tracks[i]: float(probs[i]) for i in range(len(self.tracks))}

        # Top-1 Selection
        top_track = max(scores, key=scores.get)
        top_gating_conf = scores[top_track]

        # Expert Signal Evaluation
        direction = "NONE"
        expert_conf = 0.50
        signal_code = 0
        reason = ""

        try:
            if top_track == "track2_orderflow":
                cvd_df = self.experts["track2_orderflow"].compute_cvd(long_sub)
                signal_code, expert_conf, reason = self.experts["track2_orderflow"].predict_signal(cvd_df.iloc[-1])
                if signal_code == 1:
                    direction = "LONG"
                elif signal_code == -1:
                    direction = "SHORT"
            elif top_track == "track3_tda":
                signal_code, expert_conf, reason = self.experts["track3_tda"].predict_signal(long_sub)
                if signal_code == 1:
                    direction = "LONG"
                elif signal_code == -1:
                    direction = "SHORT"
            elif top_track == "track4_statespace":
                signal_code, expert_conf, reason = self.experts["track4_statespace"].predict_signal(long_sub['Close'].values)
                if signal_code == 1:
                    direction = "LONG"
                elif signal_code == -1:
                    direction = "SHORT"
            elif top_track == "track5_cross_asset":
                s_ret = (long_sub['Close'].iloc[-1] / long_sub['Close'].iloc[-5] - 1.0)
                n_ret = (nvda_sub['Close'].iloc[-1] / nvda_sub['Close'].iloc[-5] - 1.0)
                q_ret = (qqq_sub['Close'].iloc[-1] / qqq_sub['Close'].iloc[-5] - 1.0)
                sx_ret = (macro_sub['Close'].iloc[-1] / macro_sub['Close'].iloc[-5] - 1.0) if len(macro_sub) >= 5 else 0.0
                v_ret = (vix_px / vix_sub_prev - 1.0) if 'vix_sub_prev' in locals() else 0.0
                t_ret = (tnx_sub['Close'].iloc[-1] / tnx_sub['Close'].iloc[-5] - 1.0) if len(tnx_sub) >= 5 else 0.0
                signal_code, expert_conf, reason = self.experts["track5_cross_asset"].predict_signal(
                    s_ret, n_ret, q_ret, sx_ret, v_ret, t_ret
                )
                if signal_code == 1:
                    direction = "LONG"
                elif signal_code == -1:
                    direction = "SHORT"
            else: # Track 0 / Track 1 (GBDT Sniper)
                c = long_sub['Close']
                ret_5 = float(c.iloc[-1] / c.iloc[-5] - 1.0) if len(c) >= 5 else 0.0
                if ret_5 >= 0.005:
                    direction = "LONG"
                    expert_conf = min(0.92, 0.65 + ret_5 * 5.0)
                    signal_code = 1
                elif ret_5 <= -0.005:
                    direction = "SHORT"
                    expert_conf = min(0.92, 0.65 + abs(ret_5) * 5.0)
                    signal_code = -1
                else:
                    signal_code = 0
                    expert_conf = 0.50
        except Exception as e:
            signal_code = 0
            direction = "NONE"

        # Signal Approval: Top-1 Sigmoid Confidence >= Threshold T AND Expert confirms signal
        is_approved = bool(top_gating_conf >= threshold and signal_code != 0 and expert_conf >= 0.60)

        return {
            "selected_track": top_track,
            "track_name": self.track_names.get(top_track, top_track),
            "gating_confidence": top_gating_conf,
            "expert_confidence": expert_conf,
            "direction": direction,
            "is_approved": is_approved,
            "reason": reason,
            "scores": scores,
            "regime": reg_dict
        }

def run_moe_sweep():
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
    orchestrator = MoESigmoidOrchestratorV103()

    results_table = []
    
    for T in THRESHOLDS:
        capital = float(INITIAL_CAPITAL_KRW)
        trades = []
        equity = [capital]
        
        in_market = False
        current_pos = "NONE"
        entry_price = 0.0
        entry_bar_time = ""
        bars_held = 0
        entry_track = ""

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
                long_sub = long_df[long_df['datetime'] <= bar_dt].tail(35)
                nvda_sub = nvda_df[nvda_df['datetime'] <= bar_dt].tail(35)
                qqq_sub = qqq_df[qqq_df['datetime'] <= bar_dt].tail(35)
                macro_sub = macro_df[macro_df['datetime'] <= bar_dt].tail(35)
                tnx_sub = tnx_df[tnx_df['datetime'] <= bar_dt].tail(35)

                # Exit check
                if in_market and current_pos != "NONE":
                    bars_held += 1
                    curr_row = row_l if current_pos == config.BASE_ASSET_LONG else row_s
                    curr_high = float(curr_row['High'])
                    curr_low = float(curr_row['Low'])
                    curr_close = float(curr_row['Close'])
                    
                    high_ret = (curr_high - entry_price) / entry_price
                    low_ret = (curr_low - entry_price) / entry_price
                    close_ret = (curr_close - entry_price) / entry_price

                    exit_flag = False
                    exit_reason = ""
                    actual_ret = 0.0

                    if high_ret >= TAKE_PROFIT_PCT:
                        exit_flag = True
                        actual_ret = TAKE_PROFIT_PCT - FEE_RATE
                        exit_reason = "TAKE_PROFIT"
                    elif low_ret <= STOP_LOSS_PCT:
                        exit_flag = True
                        actual_ret = STOP_LOSS_PCT - FEE_RATE
                        exit_reason = "STOP_LOSS"
                    elif bars_held >= TIME_STOP_BARS:
                        exit_flag = True
                        actual_ret = close_ret - FEE_RATE
                        exit_reason = "TIME_STOP"
                    elif b_idx == num_bars - 1:
                        exit_flag = True
                        actual_ret = close_ret - FEE_RATE
                        exit_reason = "EOD_CLOSE"

                    if exit_flag:
                        pnl_krw = int(capital * actual_ret)
                        capital += pnl_krw
                        equity.append(capital)
                        trades.append({
                            "entry_time": entry_bar_time,
                            "exit_time": bar_dt,
                            "date": day_str,
                            "ticker": current_pos,
                            "pnl_pct": actual_ret * 100.0,
                            "pnl_krw": pnl_krw,
                            "exit_reason": exit_reason,
                            "bars_held": bars_held,
                            "track": entry_track,
                            "is_win": actual_ret > 0
                        })
                        in_market = False
                        current_pos = "NONE"
                        bars_held = 0
                        continue

                # Entry check
                if not in_market and current_pos == "NONE":
                    if b_idx == 0 or time_str >= "14:30":
                        continue

                    eval_res = orchestrator.evaluate_signal(
                        b_idx, cur_vix, long_sub, nvda_sub, qqq_sub, macro_sub, tnx_sub, threshold=T
                    )

                    if eval_res["is_approved"]:
                        in_market = True
                        current_pos = config.BASE_ASSET_LONG if eval_res["direction"] == "LONG" else config.BASE_ASSET_SHORT
                        entry_row = row_l if current_pos == config.BASE_ASSET_LONG else row_s
                        entry_price = float(entry_row['Close'])
                        entry_bar_time = bar_dt
                        bars_held = 0
                        entry_track = eval_res["track_name"]

        # Calculate metrics
        total_trades = len(trades)
        wins = sum(1 for t in trades if t['is_win'])
        losses = total_trades - wins
        win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
        avg_trades_week = total_trades / total_weeks
        
        total_pnl = capital - INITIAL_CAPITAL_KRW
        cum_return = (total_pnl / INITIAL_CAPITAL_KRW) * 100.0
        
        gross_profit = sum(t['pnl_krw'] for t in trades if t['pnl_krw'] > 0)
        gross_loss = abs(sum(t['pnl_krw'] for t in trades if t['pnl_krw'] < 0))
        pf = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 99.0

        eq_arr = np.array(equity)
        peak = np.maximum.accumulate(eq_arr)
        dd = (eq_arr - peak) / peak
        mdd = round(abs(float(np.min(dd))) * 100.0, 2) if len(dd) > 0 else 0.0

        results_table.append({
            "threshold": T,
            "threshold_pct": f"{int(T*100)}%",
            "total_trades": total_trades,
            "avg_trades_per_week": round(avg_trades_week, 2),
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 1),
            "profit_factor": pf,
            "mdd": mdd,
            "cum_return": round(cum_return, 2),
            "total_pnl": int(total_pnl)
        })

    df_res = pd.DataFrame(results_table)
    print("\n" + "=" * 100)
    print(f"{'임계치 (T)':<10} | {'총 거래수':<8} | {'주당 평균 거래':<14} | {'전적':<10} | {'승률':<8} | {'Profit Factor':<14} | {'MDD':<8} | {'누적수익률':<10}")
    print("-" * 100)
    for _, r in df_res.iterrows():
        print(f"{r['threshold_pct']:<10} | {r['total_trades']:<8} | {r['avg_trades_per_week']:<14.2f} | {r['wins']}승 {r['losses']}패    | {r['win_rate']:>5.1f}%  | {r['profit_factor']:>13.2f} | {r['mdd']:>5.2f}% | {r['cum_return']:>+7.2f}%")
    print("=" * 100)
    return df_res

if __name__ == "__main__":
    run_moe_sweep()
