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

# Load all 15m and 60m data from SQLite DB
conn = sqlite3.connect('data/market_data.db')
long_15m = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol=config.BASE_ASSET_LONG AND timeframe='15m' ORDER BY datetime", conn)
short_15m = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol=config.BASE_ASSET_SHORT AND timeframe='15m' ORDER BY datetime", conn)
nvda_15m = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol='NVDA' AND timeframe='15m' ORDER BY datetime", conn)
qqq_15m = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol='QQQ' AND timeframe='15m' ORDER BY datetime", conn)
macro_15m = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol=config.MACRO_TREND_SYMBOL AND timeframe='15m' ORDER BY datetime", conn)
vix_15m = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol='^VIX' AND timeframe='15m' ORDER BY datetime", conn)
tnx_15m = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol='^TNX' AND timeframe='15m' ORDER BY datetime", conn)

long_60m = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol=config.BASE_ASSET_LONG AND timeframe='60m' ORDER BY datetime", conn)
macro_60m = pd.read_sql_query("SELECT datetime, open as Open, high as High, low as Low, close as Close, volume as Volume FROM market_candles WHERE symbol=config.MACRO_TREND_SYMBOL AND timeframe='60m' ORDER BY datetime", conn)
conn.close()

for df in [long_15m, short_15m, nvda_15m, qqq_15m, macro_15m, vix_15m, tnx_15m, long_60m, macro_60m]:
    df['Datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].str.slice(0, 10)
    df.set_index('Datetime', inplace=True)

# Compute 60m trend filters
macro_60m['ema20'] = macro_60m['Close'].ewm(span=20, adjust=False).mean()
long_60m['ema20'] = long_60m['Close'].ewm(span=20, adjust=False).mean()
ema12 = long_60m['Close'].ewm(span=12, adjust=False).mean()
ema26 = long_60m['Close'].ewm(span=26, adjust=False).mean()
long_60m['macd'] = ema12 - ema26
long_60m['macd_sig'] = long_60m['macd'].ewm(span=9, adjust=False).mean()

# Compute ML Features
ml_engine = MLFeatureEngine()
long_feat = ml_engine.extract_features(long_15m)
short_feat = ml_engine.extract_features(short_15m)

# Instantiate 4 heterogeneous non-time-series models
exp_orderflow = OrderFlowImbalanceModel(delta_threshold=1.8, absorption_ratio=2.2)
exp_tda = TDATopologyModel(entropy_threshold=0.72)
exp_statespace = StateSpaceKalmanModel()
exp_cross = CrossAssetDislocationModel(dislocation_z_threshold=1.6)

# Test MoE Sigmoid Network across 62 days
unique_dates = sorted(long_15m['date'].unique())
total_weeks = len(unique_dates) / 5.0

print(f"Loaded {len(long_15m)} 15m candles over {len(unique_dates)} trading days ({total_weeks:.1f} weeks).")

class PrecisionMoESigmoidNetwork:
    """
    [Lumos v10.3 MoE Sigmoid 절대평가 게이팅 네트워크]
    - Gating Output = Sigmoid(W · R + b)
    - 6대 전문가:
      0: GBDT Sniper (Champion)
      1: GBDT Rolling Refreshed
      2: OrderFlow CVD 수급
      3: TDA 위상수학
      4: StateSpace 칼만
      5: CrossAsset 인과괴리
    """
    def __init__(self):
        self.tracks = ["gbdt_pattern", "gbdt_refresh", "orderflow", "tda_topology", "statespace_kalman", "cross_asset"]
        self.track_names = {
            "gbdt_pattern": "Track 0: GBDT 시계열 스나이퍼",
            "gbdt_refresh": "Track 1: GBDT 일일 롤링 최신화",
            "orderflow": "Track 2: 오더플로우 CVD 수급불균형",
            "tda_topology": "Track 3: TDA 위상수학 형태붕괴",
            "statespace_kalman": "Track 4: 상태공간 칼만 동역학",
            "cross_asset": "Track 5: 크로스에셋 인과괴리"
        }
        # Prior base logits (corresponding to base conf ~0.60 - 0.70)
        self.b = np.array([0.52, 0.55, 0.48, 0.45, 0.50, 0.58], dtype=np.float64)
        # Weights for [elapsed_norm, vix_norm, atr_norm, cvd_norm, disloc_norm]
        self.W = np.array([
            [0.60, -0.20, -0.30, 0.40, 0.20],
            [0.55, -0.15, -0.25, 0.45, 0.25],
            [-0.75, 0.30, 0.60, 1.40, 0.20],
            [-0.40, 0.35, 1.30, 0.30, 0.25],
            [0.20, 1.25, 0.40, 0.20, 0.30],
            [0.30, 0.20, 0.30, 0.35, 1.50]
        ], dtype=np.float64)

    def evaluate_bar(
        self,
        b_idx: int,
        bar_dt: str,
        time_str: str,
        cur_vix: float,
        row_l: pd.Series,
        row_s: pd.Series,
        long_sub: pd.DataFrame,
        nvda_sub: pd.DataFrame,
        qqq_sub: pd.DataFrame,
        macro_sub: pd.DataFrame,
        tnx_sub: pd.DataFrame,
        macro_60_sub: pd.DataFrame,
        long_60_sub: pd.DataFrame
    ) -> Dict[str, Any]:
        # 1. Compute 5-sensor vector R
        elapsed_min = float(b_idx * 15.0)
        elapsed_norm = (elapsed_min - 120.0) / 100.0
        vix_norm = (cur_vix - 17.0) / 4.0

        hl = long_sub['High'] - long_sub['Low']
        rec_atr = hl.tail(5).mean()
        avg_atr = hl.tail(20).mean()
        atr_ratio = (rec_atr / avg_atr) if avg_atr > 0 else 1.0
        atr_norm = (atr_ratio - 1.0) / 0.4

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

        s_ret = (long_sub['Close'].iloc[-1] / long_sub['Close'].iloc[-5] - 1.0) * 100.0 if len(long_sub) >= 5 else 0.0
        n_ret = (nvda_sub['Close'].iloc[-1] / nvda_sub['Close'].iloc[-5] - 1.0) * 100.0 if len(nvda_sub) >= 5 else 0.0
        q_ret = (qqq_sub['Close'].iloc[-1] / qqq_sub['Close'].iloc[-5] - 1.0) * 100.0 if len(qqq_sub) >= 5 else 0.0
        macro_exp = (n_ret * 0.6 + q_ret * 0.4) * 3.0
        dislocation_lag = float(macro_exp - s_ret)
        disloc_norm = np.clip(dislocation_lag / 1.5, -3.0, 3.0)

        R = np.array([elapsed_norm, vix_norm, atr_norm, cvd_norm, disloc_norm], dtype=np.float64)

        # Gating Sigmoid Output: scores in [0.0, 1.0]
        logits = self.W @ R + self.b
        scores_arr = 1.0 / (1.0 + np.exp(-logits))
        scores = {self.tracks[i]: float(scores_arr[i]) for i in range(len(self.tracks))}

        # Top-1 Expert Selection
        top_expert = max(scores, key=scores.get)
        top_gating_conf = scores[top_expert]

        # 2. Check Expert Model High-Conviction Signal
        expert_sig = 0
        expert_conf = top_gating_conf
        direction = "NONE"

        if top_expert == "orderflow":
            cvd_df = exp_orderflow.compute_cvd(long_sub)
            s_code, c_val, _ = exp_orderflow.predict_signal(cvd_df.iloc[-1])
            if s_code != 0:
                expert_sig = s_code
                direction = "LONG" if s_code == 1 else "SHORT"
                expert_conf = max(top_gating_conf, c_val)

        elif top_expert == "tda_topology":
            s_code, c_val, _ = exp_tda.predict_signal(long_sub)
            if s_code != 0:
                expert_sig = s_code
                direction = "LONG" if s_code == 1 else "SHORT"
                expert_conf = max(top_gating_conf, c_val)

        elif top_expert == "statespace_kalman":
            s_code, c_val, _ = exp_statespace.predict_signal(long_sub['Close'].values)
            if s_code != 0:
                expert_sig = s_code
                direction = "LONG" if s_code == 1 else "SHORT"
                expert_conf = max(top_gating_conf, c_val)

        elif top_expert == "cross_asset":
            sx_ret = (macro_sub['Close'].iloc[-1] / macro_sub['Close'].iloc[-5] - 1.0) if len(macro_sub) >= 5 else 0.0
            v_ret = (cur_vix / vix_15m['Close'].iloc[-5] - 1.0) if len(vix_15m) >= 5 else 0.0
            t_ret = (tnx_sub['Close'].iloc[-1] / tnx_sub['Close'].iloc[-5] - 1.0) if len(tnx_sub) >= 5 else 0.0
            s_code, c_val, _ = exp_cross.predict_signal(s_ret/100, n_ret/100, q_ret/100, sx_ret, v_ret, t_ret)
            if s_code != 0:
                expert_sig = s_code
                direction = "LONG" if s_code == 1 else "SHORT"
                expert_conf = max(top_gating_conf, c_val)

        else: # GBDT Champion / Refresh (Track 0 / 1)
            # 60m Trend Confirmation
            is_60m_bull = False
            is_60m_bear = False
            if len(macro_60_sub) >= 20 and len(long_60_sub) >= 20:
                last_macro = macro_60_sub.iloc[-1]
                last_long = long_60_sub.iloc[-1]
                is_60m_bull = (last_macro['Close'] >= last_macro['ema20'] * 0.998) and (last_long['Close'] >= last_long['ema20'] * 0.998)
                is_60m_bear = (last_macro['Close'] <= last_macro['ema20'] * 1.002) and (last_long['Close'] <= last_long['ema20'] * 1.002)

            long_dip_ok = (row_l.get('VWAP_Diff', 0) <= 1.5) and (row_l.get('RSI_14', 50) <= 62.0)
            short_dip_ok = (row_s.get('VWAP_Diff', 0) <= 1.5) and (row_s.get('RSI_14', 50) <= 62.0)

            if is_60m_bull and long_dip_ok:
                expert_sig = 1
                direction = "LONG"
            elif is_60m_bear and short_dip_ok:
                expert_sig = -1
                direction = "SHORT"

        return {
            "top_expert": top_expert,
            "track_name": self.track_names[top_expert],
            "gating_confidence": top_gating_conf,
            "expert_confidence": expert_conf,
            "expert_sig": expert_sig,
            "direction": direction,
            "all_scores": scores
        }

def run_precision_sweep():
    orchestrator = PrecisionMoESigmoidNetwork()
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
            day_long = long_15m[long_15m['date'] == day_str].reset_index(drop=True)
            day_short = short_15m[short_15m['date'] == day_str].reset_index(drop=True)
            if len(day_long) < 5 or len(day_short) < 5:
                continue

            num_bars = min(len(day_long), len(day_short))

            for b_idx in range(num_bars):
                row_l = long_feat[long_feat['date'] == day_str].iloc[b_idx]
                row_s = short_feat[short_feat['date'] == day_str].iloc[b_idx]
                bar_dt = row_l['datetime']
                time_str = bar_dt.split(" ")[1][:5]

                sub_vix = vix_15m[vix_15m['datetime'] <= bar_dt]
                cur_vix = float(sub_vix.iloc[-1]['Close']) if not sub_vix.empty else 16.5
                long_sub = long_15m[long_15m['datetime'] <= bar_dt].tail(30)
                nvda_sub = nvda_15m[nvda_15m['datetime'] <= bar_dt].tail(30)
                qqq_sub = qqq_15m[qqq_15m['datetime'] <= bar_dt].tail(30)
                macro_sub = macro_15m[macro_15m['datetime'] <= bar_dt].tail(30)
                tnx_sub = tnx_15m[tnx_15m['datetime'] <= bar_dt].tail(30)

                macro_60_sub = macro_60m[macro_60m.index <= row_l.name]
                long_60_sub = long_60m[long_60m.index <= row_l.name]

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
                    if b_idx == 0 or time_str >= "14:30":
                        continue

                    eval_res = orchestrator.evaluate_bar(
                        b_idx, bar_dt, time_str, cur_vix, row_l, row_s,
                        long_sub, nvda_sub, qqq_sub, macro_sub, tnx_sub,
                        macro_60_sub, long_60_sub
                    )

                    # Check Sigmoid Threshold: top_confidence >= T and valid expert signal
                    if eval_res["gating_confidence"] >= T and eval_res["expert_sig"] != 0 and eval_res["direction"] != "NONE":
                        in_market = True
                        current_pos = config.BASE_ASSET_LONG if eval_res["direction"] == "LONG" else config.BASE_ASSET_SHORT
                        entry_row = row_l if current_pos == config.BASE_ASSET_LONG else row_s
                        entry_price = float(entry_row['Close'])
                        entry_time = bar_dt
                        bars_held = 0
                        entry_track = eval_res["track_name"]
                        entry_conf = eval_res["gating_confidence"]

        total_trades = len(trades)
        wins = sum(1 for t in trades if t['is_win'])
        losses = total_trades - wins
        win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
        avg_trades_week = total_trades / total_weeks

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
            "avg_trades_per_week": round(avg_trades_week, 2),
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
    run_precision_sweep()
