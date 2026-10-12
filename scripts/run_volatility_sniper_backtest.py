import os
import sys
import sqlite3
import config
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Set utf-8 stdout for Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.data_lake import MarketDataLake
from core.ml_engine import MLFeatureEngine
from core.heterogeneous_models import CrossAssetDislocationModel

def run_volatility_sniper_backtest():
    print("=" * 115)
    print("🏛️ [Lumos V3: Phase 1 GBDT Model C + Phase 2 Pure Volatility Breakout Sniper Backtest]")
    print(f"⏰ Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} KST")
    print("=" * 115)

    lake = MarketDataLake()
    print("⏳ [1/4] Loading multi-timeframe market candles from DataLake...")

    long_15m = lake.load_candles(config.BASE_ASSET_LONG, "15m")
    short_15m = lake.load_candles(config.BASE_ASSET_SHORT, "15m")
    long_5m  = lake.load_candles(config.BASE_ASSET_LONG, "5m")
    short_5m  = lake.load_candles(config.BASE_ASSET_SHORT, "5m")
    macro_60m = lake.load_candles(config.MACRO_TREND_SYMBOL, "60m")
    long_60m = lake.load_candles(config.BASE_ASSET_LONG, "60m")
    macro_15m = lake.load_candles(config.MACRO_TREND_SYMBOL, "15m")
    nvda_15m = lake.load_candles("NVDA", "15m")
    qqq_15m  = lake.load_candles("QQQ", "15m")
    vix_15m  = lake.load_candles("^VIX", "15m")

    # Screen 1 EMA20 precalculation on 60m
    macro_60m['ema20'] = macro_60m['Close'].ewm(span=20, adjust=False).mean()
    long_60m['ema20'] = long_60m['Close'].ewm(span=20, adjust=False).mean()

    for df in [long_15m, short_15m, long_5m, short_5m, macro_60m, long_60m, macro_15m, nvda_15m, qqq_15m, vix_15m]:
        df['datetime_dt'] = pd.to_datetime(df['datetime'])
        df['date_str'] = df['datetime_dt'].dt.strftime('%Y-%m-%d')
        df['time_str'] = df['datetime_dt'].dt.strftime('%H:%M')

    print("⏳ [2/4] Pre-processing Phase 1 ML GBDT & Cross-Asset Causal Vectors...")
    ml_15m = MLFeatureEngine()
    long_15m_feat = ml_15m.extract_features(long_15m)
    long_15m_feat = ml_15m.add_confidence_columns(long_15m_feat)
    short_15m_feat = ml_15m.extract_features(short_15m)
    short_15m_feat.set_index('datetime', inplace=True, drop=False)

    cross_mod = CrossAssetDislocationModel(dislocation_z_threshold=1.6)
    nvda_map = nvda_15m.set_index('datetime')['Close'].to_dict()
    macro_map = macro_15m.set_index('datetime')['Close'].to_dict()
    qqq_map = qqq_15m.set_index('datetime')['Close'].to_dict()
    vix_map = vix_15m.set_index('datetime')['Close'].to_dict()
    long_close = long_15m['Close'].values
    long_dt = long_15m['datetime'].values

    cross_dirs = []
    cross_confs = []
    for i in range(len(long_15m)):
        if i < 5:
            cross_dirs.append('NONE'); cross_confs.append(0.50); continue
        c_t, p_t = long_dt[i], long_dt[i-5]
        if c_t in nvda_map and p_t in nvda_map and c_t in qqq_map and p_t in qqq_map and c_t in vix_map and p_t in vix_map:
            s_r = float(long_close[i]/long_close[i-5] - 1.0)
            n_r = float(nvda_map[c_t]/nvda_map[p_t] - 1.0)
            sx_r = float(macro_map[c_t]/macro_map[p_t] - 1.0) if c_t in macro_map and p_t in macro_map else n_r
            q_r = float(qqq_map[c_t]/qqq_map[p_t] - 1.0)
            v_r = float(vix_map[c_t]/vix_map[p_t] - 1.0)
            code, conf, _ = cross_mod.predict_signal(long_ret=s_r, nvda_ret=n_r, macro_ret=sx_r, qqq_ret=q_r, vix_ret=v_r, tnx_ret=0.0)
            c_dir = "LONG" if code > 0 else ("SHORT" if code < 0 else 'NONE')
            cross_dirs.append(c_dir); cross_confs.append(conf)
        else:
            cross_dirs.append('NONE'); cross_confs.append(0.50)

    long_15m_feat['cross_dir'] = cross_dirs
    long_15m_feat['cross_conf'] = cross_confs

    print("⏳ [3/4] Engineering Rolling Intraday VWAP, Daily High/Low, and Volume Accelerators...")
    # Real-time Daily Cumulative VWAP, Daily High, Daily Low resetting every day at 09:30
    long_15m_feat['cum_vol'] = long_15m_feat.groupby('date_str')['Volume'].cumsum()
    long_15m_feat['cum_pv'] = long_15m_feat.groupby('date_str').apply(lambda x: (x['Close'] * x['Volume']).cumsum()).reset_index(level=0, drop=True)
    long_15m_feat['intraday_vwap'] = long_15m_feat['cum_pv'] / (long_15m_feat['cum_vol'] + 1e-6)

    # Intraday cumulative High and Low (rolling from 09:30 open up to the current 15m bar)
    long_15m_feat['daily_high'] = long_15m_feat.groupby('date_str')['High'].cummax()
    long_15m_feat['daily_low'] = long_15m_feat.groupby('date_str')['Low'].cummin()

    # Prior 3-candle rolling average volume (momentum accelerator)
    # Using shift(1) to avoid lookahead bias
    long_15m_feat['vol_ma3_prior'] = long_15m_feat.groupby('date_str')['Volume'].shift(1).rolling(3, min_periods=1).mean()

    long_15m_feat.set_index('datetime', inplace=True, drop=False)

    unique_dates = sorted(long_15m['date_str'].unique())

    # Execution Constants
    SLIPPAGE = 0.03
    FEE_RATE = 0.0020
    TIME_STOP_BARS = 18  # 90m for Phase 1

    # Phase 1 Parameters (Fixed Baseline)
    P1_TP = 0.030
    P1_SL = -0.020

    # Phase 2 Parameters (Short Game)
    P2_TP = 0.015
    P2_SL = -0.010

    def run_simulation(enable_phase2: bool = True) -> Dict[str, Any]:
        capital = 10000.0
        trades = []
        equity_curve = [capital]
        daily_stoploss_count = 0

        for d_str in unique_dates:
            day_15 = long_15m_feat[long_15m_feat['date_str'] == d_str]
            day_5_l = long_5m[long_5m['date_str'] == d_str]
            day_5_s = short_5m[short_5m['date_str'] == d_str]
            if len(day_15) < 5:
                continue

            b_idx = 0
            n_bars = len(day_15)
            daily_stoploss_count = 0

            while b_idx < n_bars:
                row_15 = day_15.iloc[b_idx]
                cur_time = row_15['datetime']
                time_str = row_15['time_str']

                # Circuit breaker check (3-Out Veto)
                if daily_stoploss_count >= 3:
                    b_idx += 1; continue

                # Session classification
                is_p1_window = ('09:30' <= time_str <= '14:30')
                is_p2_window = ('14:30' < time_str <= '15:20')

                if not is_p1_window and not (enable_phase2 and is_p2_window):
                    b_idx += 1; continue

                chosen_sym = None
                phase_tag = None
                trigger_desc = None

                # -------------------------------------------------------------
                # [PHASE 1: 정규 세션 09:30 ~ 14:30 (GBDT 60% + Cross-Asset Veto)]
                # -------------------------------------------------------------
                if is_p1_window:
                    dir_gbdt = row_15['Direction']
                    conf_gbdt = float(row_15['Confidence'])
                    dir_cross = row_15['cross_dir']
                    conf_cross = float(row_15['cross_conf'])

                    is_gbdt = (dir_gbdt in ["LONG", "SHORT"]) and (conf_gbdt >= 0.60)
                    is_opposite_veto = (
                        (dir_gbdt == "LONG" and dir_cross == "SHORT" and conf_cross >= 0.60) or
                        (dir_gbdt == "SHORT" and dir_cross == "LONG" and conf_cross >= 0.60)
                    )

                    if not (is_gbdt and not is_opposite_veto):
                        b_idx += 1; continue

                    chosen_sym = config.BASE_ASSET_LONG if dir_gbdt == "LONG" else config.BASE_ASSET_SHORT

                    # Screen 1: 60m trend
                    past_macro = macro_60m[macro_60m['datetime'] <= cur_time]
                    past_long = long_60m[long_60m['datetime'] <= cur_time]
                    if len(past_macro) < 20 or len(past_long) < 20:
                        b_idx += 1; continue
                    macro_c = past_macro['Close'].iloc[-1]
                    long_c = past_long['Close'].iloc[-1]
                    macro_ema = past_macro['ema20'].iloc[-1]
                    long_ema = past_long['ema20'].iloc[-1]
                    if chosen_sym == config.BASE_ASSET_LONG and not (macro_c >= macro_ema * 0.998 and long_c >= long_ema * 0.998):
                        b_idx += 1; continue
                    elif chosen_sym == config.BASE_ASSET_SHORT and not (macro_c <= macro_ema * 1.002):
                        b_idx += 1; continue

                    # Screen 3: Dip filter
                    if chosen_sym == config.BASE_ASSET_LONG:
                        vd = float(row_15.get('VWAP_Diff', 0.0))
                        r14 = float(row_15.get('RSI_14', 50.0))
                        bbl = float(row_15.get('BB_Lower', 0.0))
                        if not (vd <= 1.5 and r14 <= 62.0 and (bbl <= 0 or float(row_15['Close']) >= bbl * 1.001)):
                            b_idx += 1; continue
                    else:
                        if cur_time not in short_15m_feat.index:
                            b_idx += 1; continue
                        rs = short_15m_feat.loc[cur_time]
                        vd = float(rs.get('VWAP_Diff', 0.0))
                        r14 = float(rs.get('RSI_14', 50.0))
                        bbl = float(rs.get('BB_Lower', 0.0))
                        if not (vd <= 1.5 and r14 <= 62.0 and (bbl <= 0 or float(rs['Close']) >= bbl * 1.001)):
                            b_idx += 1; continue

                    phase_tag = "Phase 1 (GBDT Model C)"
                    trigger_desc = f"GBDT {conf_gbdt*100:.1f}%"

                # -------------------------------------------------------------
                # [PHASE 2: 장 막판 순수 변동성 돌파 스나이퍼 14:30 ~ 15:20]
                # (GBDT, Cross-Asset 100% OFF, 순수 VWAP + Daily High/Low 돌파)
                # -------------------------------------------------------------
                elif enable_phase2 and is_p2_window:
                    spot_px = float(row_15['Close'])
                    vwap_px = float(row_15['intraday_vwap'])
                    d_high = float(row_15['daily_high'])
                    d_low = float(row_15['daily_low'])
                    cur_vol = float(row_15['Volume'])
                    vol_ma3 = float(row_15.get('vol_ma3_prior', 0.0))

                    # Momentum acceleration condition: Volume > 3-bar rolling MA
                    vol_surge = (cur_vol > vol_ma3) if vol_ma3 > 0 else True

                    # 1. TQQQ (롱 돌파): Spot > VWAP AND Spot >= Daily_High * 0.99 AND Volume > MA3
                    cond_long = (spot_px > vwap_px) and (spot_px >= d_high * 0.99) and vol_surge

                    # 2. SQQQ (숏 돌파): Spot < VWAP AND Spot <= Daily_Low * 1.01 AND Volume > MA3
                    cond_short = (spot_px < vwap_px) and (spot_px <= d_low * 1.01) and vol_surge

                    if cond_long and not cond_short:
                        chosen_sym = config.BASE_ASSET_LONG
                        phase_tag = "Phase 2 (Vol Sniper)"
                        trigger_desc = f"High Breakout (P={spot_px:.2f} >= {d_high*0.99:.2f}, V={cur_vol/vol_ma3:.1f}x)"
                    elif cond_short and not cond_long:
                        chosen_sym = config.BASE_ASSET_SHORT
                        phase_tag = "Phase 2 (Vol Sniper)"
                        trigger_desc = f"Low Breakdown (P={spot_px:.2f} <= {d_low*1.01:.2f}, V={cur_vol/vol_ma3:.1f}x)"
                    else:
                        b_idx += 1; continue

                if not chosen_sym:
                    b_idx += 1; continue

                # Position Entry
                base_px = float(row_15['Close']) if chosen_sym == config.BASE_ASSET_LONG else float(short_15m_feat.loc[cur_time]['Close'])
                entry_px = round(base_px + SLIPPAGE, 2)
                shares = int(capital / entry_px)
                invested = shares * entry_px
                if shares <= 0:
                    b_idx += 1; continue

                target_5m = day_5_l if chosen_sym == config.BASE_ASSET_LONG else day_5_s
                post_5m = target_5m[target_5m['datetime'] > cur_time]
                if post_5m.empty:
                    b_idx += 1; continue

                # Risk-Reward setting by Phase
                if "Phase 1" in phase_tag:
                    cur_tp = P1_TP; cur_sl = P1_SL
                else:
                    cur_tp = P2_TP; cur_sl = P2_SL

                tp_px = round(entry_px * (1 + cur_tp), 2)
                sl_px = round(entry_px * (1 + cur_sl), 2)
                exit_px = entry_px
                exit_reason = "TimeStop"
                exit_time_str = ""
                bars_held = 0

                eval_5m = post_5m.iloc[:TIME_STOP_BARS]
                for k in range(len(eval_5m)):
                    c5 = eval_5m.iloc[k]
                    t5 = c5['time_str']
                    c5_o, c5_h, c5_l, c5_c = float(c5['Open']), float(c5['High']), float(c5['Low']), float(c5['Close'])
                    bars_held = k + 1
                    hit_tp, hit_sl = (c5_h >= tp_px), (c5_l <= sl_px)

                    if hit_tp and hit_sl:
                        if c5_c >= c5_o:
                            exit_px = round(tp_px - SLIPPAGE, 2)
                            exit_reason = f"TP +{cur_tp*100:.1f}%"
                        else:
                            exit_px = round(sl_px - SLIPPAGE, 2)
                            exit_reason = f"SL {cur_sl*100:.1f}%"
                            daily_stoploss_count += 1
                        exit_time_str = t5
                        break
                    elif hit_tp:
                        exit_px = round(tp_px - SLIPPAGE, 2)
                        exit_reason = f"TP +{cur_tp*100:.1f}%"
                        exit_time_str = t5
                        break
                    elif hit_sl:
                        exit_px = round(sl_px - SLIPPAGE, 2)
                        exit_reason = f"SL {cur_sl*100:.1f}%"
                        daily_stoploss_count += 1
                        exit_time_str = t5
                        break
                    elif t5 >= '15:45' or k == len(eval_5m) - 1:
                        # 15:50 EOD Force MOC Sprint Liquidation
                        exit_px = round(c5_c - SLIPPAGE, 2)
                        exit_reason = "15:50 EOD" if t5 >= '15:45' else "TimeStop 90m"
                        exit_time_str = t5
                        if (exit_px - entry_px) / entry_px <= cur_sl:
                            daily_stoploss_count += 1
                        break

                cost = invested * FEE_RATE
                net_pnl = (shares * (exit_px - entry_px)) - cost
                capital += net_pnl
                equity_curve.append(capital)
                ret = net_pnl / invested

                trades.append({
                    "date": d_str,
                    "symbol": chosen_sym,
                    "phase": phase_tag,
                    "trigger_desc": trigger_desc,
                    "entry_t": time_str,
                    "exit_t": exit_time_str,
                    "entry_px": entry_px,
                    "exit_px": exit_px,
                    "ret_pct": round(ret * 100, 2),
                    "net_pnl": round(net_pnl, 2),
                    "exit_reason": exit_reason,
                    "bars_held": bars_held,
                    "holding_minutes": bars_held * 5
                })

                # Single-position lock: fast-forward past trade holding bars
                consumed = int(np.ceil(bars_held / 3.0))
                b_idx += max(1, consumed)

        tot_tr = len(trades)
        w_tr = [t for t in trades if t['net_pnl'] > 0]
        l_tr = [t for t in trades if t['net_pnl'] <= 0]
        wr = (len(w_tr) / tot_tr * 100) if tot_tr > 0 else 0.0

        p1_trades = [t for t in trades if "Phase 1" in t['phase']]
        p2_trades = [t for t in trades if "Phase 2" in t['phase']]

        p1_wins = [t for t in p1_trades if t['net_pnl'] > 0]
        p1_wr = (len(p1_wins) / len(p1_trades) * 100) if p1_trades else 0.0
        p1_pnl = sum([t['net_pnl'] for t in p1_trades])

        p2_wins = [t for t in p2_trades if t['net_pnl'] > 0]
        p2_wr = (len(p2_wins) / len(p2_trades) * 100) if p2_trades else 0.0
        p2_pnl = sum([t['net_pnl'] for t in p2_trades])

        gross_win = sum([t['ret_pct'] for t in w_tr])
        gross_loss = abs(sum([t['ret_pct'] for t in l_tr])) if l_tr else 0.0001
        pf = (gross_win / gross_loss) if gross_loss > 0 else 99.9
        cum_ret = ((capital - 10000.0) / 10000.0) * 100.0

        eq_arr = np.array(equity_curve)
        pk = np.maximum.accumulate(eq_arr)
        dd = (pk - eq_arr) / pk * 100.0
        mdd = np.max(dd) if len(dd) > 0 else 0.0

        return {
            "enable_phase2": enable_phase2,
            "total_trades": tot_tr,
            "wins": len(w_tr),
            "losses": len(l_tr),
            "win_rate": wr,
            "profit_factor": pf,
            "cum_return": cum_ret,
            "mdd": mdd,
            "final_capital": capital,
            "p1_trades": len(p1_trades),
            "p1_wins": len(p1_wins),
            "p1_losses": len(p1_trades) - len(p1_wins),
            "p1_win_rate": p1_wr,
            "p1_pnl": p1_pnl,
            "p2_trades": len(p2_trades),
            "p2_wins": len(p2_wins),
            "p2_losses": len(p2_trades) - len(p2_wins),
            "p2_win_rate": p2_wr,
            "p2_pnl": p2_pnl,
            "trades": trades
        }

    # 1. Baseline Simulation (Phase 1 Only, 14:30 Shutdown)
    print("⏳ [4/4] Executing Simulations: [1] Baseline vs [2] Phase 1+Phase 2 Hybrid Sniper...")
    base_res = run_simulation(enable_phase2=False)
    hybrid_res = run_simulation(enable_phase2=True)

    delta_cum_ret = hybrid_res['cum_return'] - base_res['cum_return']
    delta_pnl_usd = hybrid_res['final_capital'] - base_res['final_capital']

    # Performance Comparison DataFrame
    df_comparison = pd.DataFrame([
        {
            "운용 전략 모드": "1. Baseline (Phase 1 단독 14:30 셧다운)",
            "총 매매 횟수": f"{base_res['total_trades']}회",
            "전체 승률": f"{base_res['win_rate']:.1f}% ({base_res['wins']}승/{base_res['losses']}패)",
            "손익비 (PF)": f"{base_res['profit_factor']:.2f}",
            "누적 수익률": f"{base_res['cum_return']:+.2f}%",
            "MDD": f"{base_res['mdd']:.2f}%",
            "최종 자본금": f"${base_res['final_capital']:,.2f}",
            "Delta PnL (초과수익)": "기준선 (Base)"
        },
        {
            "운용 전략 모드": "2. Hybrid (Phase 1 + Phase 2 순수 변동성 스나이퍼)",
            "총 매매 횟수": f"{hybrid_res['total_trades']}회",
            "전체 승률": f"{hybrid_res['win_rate']:.1f}% ({hybrid_res['wins']}승/{hybrid_res['losses']}패)",
            "손익비 (PF)": f"{hybrid_res['profit_factor']:.2f}",
            "누적 수익률": f"{hybrid_res['cum_return']:+.2f}%",
            "MDD": f"{hybrid_res['mdd']:.2f}%",
            "최종 자본금": f"${hybrid_res['final_capital']:,.2f}",
            "Delta PnL (초과수익)": f"{delta_cum_ret:+.2f}%p (${delta_pnl_usd:+,.2f})"
        }
    ])

    # Phase 2 Performance Isolation DataFrame
    df_p2_isolated = pd.DataFrame([
        {
            "Phase 2 스나이퍼 매매 횟수": f"{hybrid_res['p2_trades']}회",
            "Phase 2 승 / 패": f"{hybrid_res['p2_wins']}승 {hybrid_res['p2_losses']}패",
            "Phase 2 자체 승률(%)": f"{hybrid_res['p2_win_rate']:.1f}%",
            "Phase 2 순이익 기여분 ($)": f"${hybrid_res['p2_pnl']:+,.2f}",
            "목표 승률 (>60%) 달성 여부": "PASS ✅" if hybrid_res['p2_win_rate'] >= 60.0 else "FAIL ❌"
        }
    ])

    print("\n" + "=" * 135)
    print("🏆 [1. Lumos V3: Baseline 대비 Hybrid 시스템 전체 성과 비교표]")
    print("=" * 135)
    print(df_comparison.to_string(index=False))
    print("=" * 135)

    print("\n" + "=" * 105)
    print("🎯 [2. Phase 2 순수 변동성 돌파 스나이퍼 단독 성과 분리 리포트]")
    print("=" * 105)
    print(df_p2_isolated.to_string(index=False))
    print("=" * 105)

    # Detailed Inspection of Phase 2 Trades
    p2_trades_list = [t for t in hybrid_res['trades'] if "Phase 2" in t['phase']]
    print(f"\n🔍 [3. Phase 2 순수 변동성 스나이퍼 실제 체결 거래 전수 내역 (총 {len(p2_trades_list)}건)]")
    if p2_trades_list:
        pdf = pd.DataFrame(p2_trades_list)
        cols = ['date', 'symbol', 'entry_t', 'exit_t', 'entry_px', 'exit_px', 'ret_pct', 'net_pnl', 'exit_reason', 'trigger_desc']
        print(pdf[cols].to_string(index=False))
    else:
        print("   (Phase 2 발동 조건에 부합한 거래 없음)")

    return base_res, hybrid_res, df_comparison, df_p2_isolated

if __name__ == '__main__':
    run_volatility_sniper_backtest()
