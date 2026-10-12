import os
import sys
import numpy as np
import config
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
import warnings

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
from core.power_hour_quant import PowerHourQuantEngine

# Execution Constants
SLIPPAGE = 0.03       # $0.03 per share
FEE_RATE = 0.0020     # 0.20% roundtrip fee

def simulate_strategy(
    df_long: pd.DataFrame,
    df_short: pd.DataFrame,
    mode: str,                   # 'rule_only', 'rule_gbdt', 'gbdt_only'
    exit_type: str,              # 'fixed_5m', 'fixed_10m', 'fixed_15m', 'tp_sl_1.5_1.0', 'tp_sl_2.0_1.0'
    engine: PowerHourQuantEngine,
    gbdt_threshold: float = 0.52,
    initial_capital: float = 10000.0
) -> Dict[str, Any]:
    """
    파워 아워(14:30 ~ 15:30 EDT) 단기 퀀트 전략 시뮬레이션
    """
    capital = initial_capital
    trades = []
    equity_curve = [capital]

    # Pre-map SQQQ prices by datetime
    short_map_c = df_short.set_index('datetime')['Close'].to_dict()
    short_map_h = df_short.set_index('datetime')['High'].to_dict()
    short_map_l = df_short.set_index('datetime')['Low'].to_dict()
    short_map_o = df_short.set_index('datetime')['Open'].to_dict()

    n_bars = len(df_long)
    b_idx = 0

    while b_idx < n_bars:
        row = df_long.iloc[b_idx]
        cur_time = row['datetime']
        t_str = row['time_str']

        # 파워 아워 시간대(14:30 ~ 15:30 EDT)만 진입 평가
        if not ('14:30' <= t_str <= '15:30'):
            b_idx += 1
            continue

        # 진입 신호 생성 (Mode별 분기)
        action = 'NONE'
        rule_sig = engine.evaluate_rule_signal(row)

        if mode == 'rule_only':
            action = rule_sig
        elif mode == 'rule_gbdt':
            if rule_sig == 'LONG':
                p_win = engine.predict_meta_long(row)
                if p_win >= gbdt_threshold:
                    action = 'LONG'
            elif rule_sig == 'SHORT':
                p_win = engine.predict_meta_short(row)
                if p_win >= gbdt_threshold:
                    action = 'SHORT'
        elif mode == 'gbdt_only':
            action = engine.predict_direct(row, threshold=gbdt_threshold)

        if action == 'NONE':
            b_idx += 1
            continue

        # 매매 대상 심볼 및 진입 가격 결정
        chosen_sym = config.BASE_ASSET_LONG if action == 'LONG' else config.BASE_ASSET_SHORT
        base_px = float(row['Close']) if chosen_sym == config.BASE_ASSET_LONG else float(short_map_c.get(cur_time, 0.0))
        if base_px <= 0:
            b_idx += 1
            continue

        entry_px = round(base_px + SLIPPAGE, 2)
        shares = int(capital / entry_px)
        if shares <= 0:
            b_idx += 1
            continue

        invested = shares * entry_px

        # Exit 평가
        exit_px = entry_px
        exit_reason = ""
        exit_time_str = ""
        bars_held = 0

        # [Exit Mode A, B, C: Fixed Time Exit (5m, 10m, 15m)]
        if exit_type in ['fixed_5m', 'fixed_10m', 'fixed_15m']:
            hold_target = 1 if exit_type == 'fixed_5m' else (2 if exit_type == 'fixed_10m' else 3)
            target_idx = min(b_idx + hold_target, n_bars - 1)
            exit_row = df_long.iloc[target_idx]
            exit_t = exit_row['datetime']
            exit_time_str = exit_row['time_str']

            # 만약 날짜가 바뀌었거나 장마감(15:55)을 넘어선 경우 당일 마지막 봉으로 청산
            if exit_row['date_str'] != row['date_str'] or exit_time_str > '15:55':
                same_day_bars = df_long[(df_long['date_str'] == row['date_str']) & (df_long.index > b_idx)]
                if not same_day_bars.empty:
                    exit_row = same_day_bars.iloc[-1]
                    exit_t = exit_row['datetime']
                    exit_time_str = exit_row['time_str']

            px_out = float(exit_row['Close']) if chosen_sym == config.BASE_ASSET_LONG else float(short_map_c.get(exit_t, base_px))
            exit_px = round(px_out - SLIPPAGE, 2)
            exit_reason = f"Fixed_{hold_target*5}m"
            bars_held = max(1, target_idx - b_idx)

        # [Exit Mode D, E: TP / SL 밴드 + 15:50 EOD 청산]
        elif exit_type.startswith('tp_sl'):
            if exit_type == 'tp_sl_1.5_1.0':
                tp_pct, sl_pct = 0.015, -0.010
            else: # 'tp_sl_2.0_1.0'
                tp_pct, sl_pct = 0.020, -0.010

            tp_px = round(entry_px * (1 + tp_pct), 2)
            sl_px = round(entry_px * (1 + sl_pct), 2)

            max_eval_bars = 12  # 최대 60분
            eval_slice = df_long.iloc[b_idx + 1 : min(b_idx + 1 + max_eval_bars, n_bars)]

            for k in range(len(eval_slice)):
                c_bar = eval_slice.iloc[k]
                t_bar = c_bar['time_str']
                d_bar = c_bar['date_str']
                c_dt = c_bar['datetime']

                # 날짜 변경 방어
                if d_bar != row['date_str']:
                    break

                bars_held = k + 1

                if chosen_sym == config.BASE_ASSET_LONG:
                    c_h, c_l, c_c, c_o = float(c_bar['High']), float(c_bar['Low']), float(c_bar['Close']), float(c_bar['Open'])
                else:
                    c_h = float(short_map_h.get(c_dt, base_px))
                    c_l = float(short_map_l.get(c_dt, base_px))
                    c_c = float(short_map_c.get(c_dt, base_px))
                    c_o = float(short_map_o.get(c_dt, base_px))

                hit_tp = (c_h >= tp_px)
                hit_sl = (c_l <= sl_px)

                if hit_tp and hit_sl:
                    if c_c >= c_o:
                        exit_px = round(tp_px - SLIPPAGE, 2)
                        exit_reason = f"TP +{tp_pct*100:.1f}%"
                    else:
                        exit_px = round(sl_px - SLIPPAGE, 2)
                        exit_reason = f"SL {sl_pct*100:.1f}%"
                    exit_time_str = t_bar
                    break
                elif hit_tp:
                    exit_px = round(tp_px - SLIPPAGE, 2)
                    exit_reason = f"TP +{tp_pct*100:.1f}%"
                    exit_time_str = t_bar
                    break
                elif hit_sl:
                    exit_px = round(sl_px - SLIPPAGE, 2)
                    exit_reason = f"SL {sl_pct*100:.1f}%"
                    exit_time_str = t_bar
                    break
                elif t_bar >= '15:50' or k == len(eval_slice) - 1:
                    exit_px = round(c_c - SLIPPAGE, 2)
                    exit_reason = "15:50 EOD" if t_bar >= '15:50' else "TimeLimit"
                    exit_time_str = t_bar
                    break

            if not exit_reason:
                exit_px = round(base_px - SLIPPAGE, 2)
                exit_reason = "15:50 EOD"
                exit_time_str = '15:50'
                bars_held = 1

        cost = invested * FEE_RATE
        net_pnl = (shares * (exit_px - entry_px)) - cost
        capital += net_pnl
        equity_curve.append(capital)
        ret = net_pnl / invested

        trades.append({
            "date": row['date_str'],
            "datetime": cur_time,
            "time_str": t_str,
            "symbol": chosen_sym,
            "action": action,
            "entry_px": entry_px,
            "exit_px": exit_px,
            "ret_pct": round(ret * 100, 2),
            "net_pnl": round(net_pnl, 2),
            "exit_reason": exit_reason,
            "exit_time_str": exit_time_str,
            "bars_held": bars_held,
            "holding_min": bars_held * 5
        })

        # 포지션 보유 기간 동안 스캐너 잠금 (단일 포지션 릴레이)
        b_idx += max(1, bars_held)

    # Performance Metrics Calculation
    tot_trades = len(trades)
    if tot_trades == 0:
        return {
            "mode": mode, "exit_type": exit_type, "total_trades": 0, "win_rate": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0, "payoff_ratio": 0.0, "expectancy": 0.0,
            "profit_factor": 0.0, "cum_return": 0.0, "mdd": 0.0, "final_capital": capital,
            "hourly_stats": {}, "trades": []
        }

    wins = [t for t in trades if t['net_pnl'] > 0]
    losses = [t for t in trades if t['net_pnl'] <= 0]
    wr = len(wins) / tot_trades * 100.0

    avg_win = np.mean([t['ret_pct'] for t in wins]) if wins else 0.0
    avg_loss = abs(np.mean([t['ret_pct'] for t in losses])) if losses else 0.0
    payoff = (avg_win / avg_loss) if avg_loss > 0 else 99.0

    # Mathematical Expectancy (E = WinRate * AvgWin - LossRate * AvgLoss)
    p_w = len(wins) / tot_trades
    p_l = len(losses) / tot_trades
    expectancy = (p_w * avg_win) - (p_l * avg_loss)

    gross_win = sum([t['ret_pct'] for t in wins])
    gross_loss = abs(sum([t['ret_pct'] for t in losses])) if losses else 0.0001
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else 99.0

    cum_return = ((capital - initial_capital) / initial_capital) * 100.0

    eq_arr = np.array(equity_curve)
    pk = np.maximum.accumulate(eq_arr)
    dd = (pk - eq_arr) / pk * 100.0
    mdd = np.max(dd) if len(dd) > 0 else 0.0

    # Hourly breakdown (14:30 ~ 15:00 vs 15:00 ~ 15:30)
    t_early = [t for t in trades if t['time_str'] < '15:00']
    t_late  = [t for t in trades if t['time_str'] >= '15:00']

    early_wins = [t for t in t_early if t['net_pnl'] > 0]
    late_wins  = [t for t in t_late if t['net_pnl'] > 0]

    early_wr = (len(early_wins) / len(t_early) * 100.0) if t_early else 0.0
    late_wr  = (len(late_wins) / len(t_late) * 100.0) if t_late else 0.0

    early_pnl = sum([t['net_pnl'] for t in t_early])
    late_pnl  = sum([t['net_pnl'] for t in t_late])

    return {
        "mode": mode,
        "exit_type": exit_type,
        "total_trades": tot_trades,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": wr,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "cum_return": cum_return,
        "mdd": mdd,
        "final_capital": capital,
        "hourly_stats": {
            "14:30~15:00": {"trades": len(t_early), "win_rate": early_wr, "net_pnl": early_pnl},
            "15:00~15:30": {"trades": len(t_late),  "win_rate": late_wr,  "net_pnl": late_pnl}
        },
        "trades": trades
    }

def run_comprehensive_power_hour_backtest():
    print("=" * 125)
    print("🏛️ [Lumos Quant: Power Hour (14:30~15:30 EDT) Short-term Strategy & GBDT Multi-Mode Backtest]")
    print(f"⏰ Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} KST")
    print("=" * 125)

    lake = MarketDataLake()
    print("⏳ [1/5] Loading 5m candles for TQQQ & SQQQ from DataLake...")
    df_long = lake.load_candles(config.BASE_ASSET_LONG, "5m")
    df_short = lake.load_candles(config.BASE_ASSET_SHORT, "5m")

    # Align by common timestamps
    common_dt = sorted(list(set(df_long['datetime']).intersection(set(df_short['datetime']))))
    df_long = df_long[df_long['datetime'].isin(common_dt)].sort_values('datetime').reset_index(drop=True)
    df_short = df_short[df_short['datetime'].isin(common_dt)].sort_values('datetime').reset_index(drop=True)

    print(f"   ✓ Aligned 5m Candles: {len(df_long)} bars ({df_long['datetime'].iloc[0]} ~ {df_long['datetime'].iloc[-1]})")

    for df in [df_long, df_short]:
        df['datetime_dt'] = pd.to_datetime(df['datetime'])
        df['date_str'] = df['datetime_dt'].dt.strftime('%Y-%m-%d')
        df['time_str'] = df['datetime_dt'].dt.strftime('%H:%M')

    print("⏳ [2/5] Engineering Volatility Expansion, Breakout, Volume Surge & VWAP Features...")
    engine = PowerHourQuantEngine(n_breakout_bars=6)
    feat_long = engine.compute_features(df_long)

    # Chronological Train/Test Split (70% Train, 30% Test)
    unique_dates = sorted(feat_long['date_str'].unique())
    n_total_days = len(unique_dates)
    n_train_days = int(n_total_days * 0.70)
    train_dates = set(unique_dates[:n_train_days])
    test_dates = set(unique_dates[n_train_days:])

    print(f"⏳ [3/5] Chronological Walk-Forward Train/Test Splitting...")
    print(f"   • Total Trading Days: {n_total_days} days")
    print(f"   • Train Days (70%): {len(train_dates)} days ({unique_dates[0]} ~ {unique_dates[n_train_days-1]})")
    print(f"   • Test Days (30% Out-of-Sample): {len(test_dates)} days ({unique_dates[n_train_days]} ~ {unique_dates[-1]})")

    train_long = feat_long[feat_long['date_str'].isin(train_dates)].reset_index(drop=True)
    test_long  = feat_long[feat_long['date_str'].isin(test_dates)].reset_index(drop=True)
    test_short  = df_short[df_short['date_str'].isin(test_dates)].reset_index(drop=True)

    print("⏳ [4/5] Training Power Hour Dedicated GBDT Models on Train Split...")
    engine.train_models(train_long)
    print("   ✓ LightGBM Meta-Labeling Model (Long/Short) & Direct Model Trained Successfully.")

    # 3 Strategy Modes to Compare
    modes = ['rule_only', 'rule_gbdt', 'gbdt_only']
    
    # 5 Exit Structures to Test
    exit_structures = [
        ('fixed_5m', '5분 고정 청산 (1-Bar Exit)'),
        ('fixed_10m', '10분 고정 청산 (2-Bar Exit)'),
        ('fixed_15m', '15분 고정 청산 (3-Bar Exit)'),
        ('tp_sl_1.5_1.0', 'TP +1.5% / SL -1.0% + EOD'),
        ('tp_sl_2.0_1.0', 'TP +2.0% / SL -1.0% + EOD'),
    ]

    print("⏳ [5/5] Executing Out-of-Sample Backtests across 3 Modes & 5 Exit Structures (Total 15 Runs)...")
    results = []
    # Calibrated GBDT threshold based on train base-rate (29.5% base win rate -> 0.30 cutoff = top 30% highest confidence)
    CALIBRATED_TH = 0.30

    for mode in modes:
        for exit_code, exit_label in exit_structures:
            res = simulate_strategy(
                df_long=test_long,
                df_short=test_short,
                mode=mode,
                exit_type=exit_code,
                engine=engine,
                gbdt_threshold=CALIBRATED_TH
            )
            res['mode_label'] = '1. Rule Only' if mode == 'rule_only' else ('2. Rule + GBDT' if mode == 'rule_gbdt' else '3. GBDT Only')
            res['exit_label'] = exit_label
            results.append(res)

    # Summary Table Construction
    summary_rows = []
    for r in results:
        h1 = r['hourly_stats'].get('14:30~15:00', {})
        h2 = r['hourly_stats'].get('15:00~15:30', {})
        summary_rows.append({
            "전략 모드": r['mode_label'],
            "청산 구조 (Exit)": r['exit_label'],
            "거래수": f"{r['total_trades']}회",
            "승률": f"{r['win_rate']:.1f}%",
            "평균수익": f"{r['avg_win']:+.2f}%",
            "평균손실": f"-{r['avg_loss']:.2f}%",
            "손익비(PF)": f"{r['profit_factor']:.2f}",
            "기대값(E)": f"{r['expectancy']:+.2f}%",
            "누적수익률": f"{r['cum_return']:+.2f}%",
            "MDD": f"{r['mdd']:.2f}%",
            "14:30~15:00 (승률/PnL)": f"{h1.get('win_rate', 0.0):.0f}% ({h1.get('net_pnl', 0.0):+,.0f}$)" if h1.get('trades', 0) > 0 else "-",
            "15:00~15:30 (승률/PnL)": f"{h2.get('win_rate', 0.0):.0f}% ({h2.get('net_pnl', 0.0):+,.0f}$)" if h2.get('trades', 0) > 0 else "-",
        })

    summary_df = pd.DataFrame(summary_rows)

    print("\n" + "=" * 165)
    print("🏆 [Lumos Quant: Power Hour (14:30~15:30 EDT) 3-Way 전략 모드 & 청산 구조 Out-of-Sample 종합 비교 리포트]")
    print("=" * 165)
    print(summary_df.to_string(index=False))
    print("=" * 165)

    # Walk-Forward Cross Validation (3 Folds)
    print("\n" + "=" * 135)
    print("🔄 [과최적화 방지: 3-Fold Walk-Forward 시계열 교차 검증 (TP 1.5% / SL 1.0% 기준)]")
    print("=" * 135)
    wf_folds = [
        ('Fold 1 (July OOS)',   unique_dates[:30], unique_dates[30:55]),
        ('Fold 2 (August OOS)', unique_dates[15:45], unique_dates[45:65]),
        ('Fold 3 (Sept OOS)',   unique_dates[30:60], unique_dates[60:]),
    ]

    wf_rows = []
    for fname, tr_d, te_d in wf_folds:
        tr_df = feat_long[feat_long['date_str'].isin(tr_d)].reset_index(drop=True)
        te_l = feat_long[feat_long['date_str'].isin(te_d)].reset_index(drop=True)
        te_s = df_short[df_short['date_str'].isin(te_d)].reset_index(drop=True)

        eng = PowerHourQuantEngine(n_breakout_bars=6)
        eng.train_models(tr_df)

        for m in ['rule_only', 'rule_gbdt', 'gbdt_only']:
            m_lbl = '1. Rule Only' if m == 'rule_only' else ('2. Rule + GBDT' if m == 'rule_gbdt' else '3. GBDT Only')
            res = simulate_strategy(te_l, te_s, mode=m, exit_type='tp_sl_1.5_1.0', engine=eng, gbdt_threshold=0.30)
            wf_rows.append({
                "검증 폴드 (Fold)": fname,
                "전략 모드": m_lbl,
                "거래 횟수": f"{res['total_trades']}회",
                "승률": f"{res['win_rate']:.1f}%",
                "평균 수익": f"{res['avg_win']:+.2f}%",
                "평균 손실": f"-{res['avg_loss']:.2f}%",
                "기대값(E)": f"{res['expectancy']:+.2f}%",
                "손익비(PF)": f"{res['profit_factor']:.2f}",
                "누적 수익률": f"{res['cum_return']:+.2f}%",
                "MDD": f"{res['mdd']:.2f}%"
            })

    wf_df = pd.DataFrame(wf_rows)
    print(wf_df.to_string(index=False))
    print("=" * 135)

    return results, summary_df, wf_df

if __name__ == '__main__':
    run_comprehensive_power_hour_backtest()
