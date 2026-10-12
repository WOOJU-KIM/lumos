import os
import sys
import json
import sqlite3
import config
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_DIR
from core.data_lake import MarketDataLake
from core.heterogeneous_models import CrossAssetDislocationModel
from core.ml_engine import MLFeatureEngine
from core.moe_orchestrator import MoEMetaOrchestrator
from core.power_hour_sniper import PowerHourSniper

def format_markdown_table(headers: List[str], rows: List[List[Any]], aligns: List[str] = None) -> str:
    if aligns is None:
        aligns = [":---:" for _ in headers]
    col_widths = [len(h) for h in headers]
    for r in rows:
        for i, val in enumerate(r):
            col_widths[i] = max(col_widths[i], len(str(val)))
    header_line = "| " + " | ".join(h.center(col_widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join(
        (":" + "-" * (col_widths[i] - 2) + ":") if aligns[i] == ":---:"
        else (":" + "-" * (col_widths[i] - 1)) if aligns[i] == ":---"
        else ("-" * (col_widths[i] - 1) + ":") for i in range(len(headers))
    ) + " |"
    body_lines = []
    for r in rows:
        row_str = "| " + " | ".join(str(val).center(col_widths[i]) if aligns[i] == ":---:" else str(val).rjust(col_widths[i]) if aligns[i] == "---:" else str(val).ljust(col_widths[i]) for i, val in enumerate(r)) + " |"
        body_lines.append(row_str)
    return "\n".join([header_line, sep_line] + body_lines)

def run_comprehensive_dual_doe():
    lake = MarketDataLake()
    print("=" * 110)
    print("🏛 [Lumos 퀀트 시스템: 15분봉 메인 & 5분봉 스나이퍼 GBDT 7대 확신도 DoE 전구간 백테스트]")
    print(f"⏰ 실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S KST')}")
    print("=" * 110)

    # 1. 데이터 로드
    print("⏳ [1/4] 데이터 레이크에서 전 기간 캔들 로드 중...")
    long_15m = lake.load_candles(config.BASE_ASSET_LONG, "15m")
    short_15m = lake.load_candles(config.BASE_ASSET_SHORT, "15m")
    long_5m  = lake.load_candles(config.BASE_ASSET_LONG, "5m")
    short_5m  = lake.load_candles(config.BASE_ASSET_SHORT, "5m")
    macro_60m = lake.load_candles(config.MACRO_TREND_SYMBOL, "60m")
    long_60m = lake.load_candles(config.BASE_ASSET_LONG, "60m")
    macro_15m = lake.load_candles(config.MACRO_TREND_SYMBOL, "15m")
    macro_5m  = lake.load_candles(config.MACRO_TREND_SYMBOL, "5m")
    nvda_15m = lake.load_candles("NVDA", "15m")
    nvda_5m  = lake.load_candles("NVDA", "5m")
    qqq_15m  = lake.load_candles("QQQ", "15m")
    qqq_5m   = lake.load_candles("QQQ", "5m")
    vix_15m  = lake.load_candles("^VIX", "15m")
    vix_5m   = lake.load_candles("^VIX", "5m")

    for df in [long_15m, short_15m, long_5m, short_5m, macro_60m, long_60m, macro_15m, macro_5m, nvda_15m, nvda_5m, qqq_15m, qqq_5m, vix_15m, vix_5m]:
        if df.empty:
            continue
        if 'datetime' in df.columns:
            df['datetime_dt'] = pd.to_datetime(df['datetime'])
            df['date_str'] = df['datetime_dt'].dt.strftime('%Y-%m-%d')
            df['time_str'] = df['datetime_dt'].dt.strftime('%H:%M')
        else:
            df['datetime_dt'] = pd.to_datetime(df.index)
            df['date_str'] = df['datetime_dt'].dt.strftime('%Y-%m-%d')
            df['time_str'] = df['datetime_dt'].dt.strftime('%H:%M')

    unique_dates = sorted(long_15m['date_str'].unique())
    num_days = len(unique_dates)
    total_weeks = num_days / 5.0
    print(f"   • 분석 기간: {long_15m['datetime'].iloc[0]} ~ {long_15m['datetime'].iloc[-1]}")
    print(f"   • 총 거래일: {num_days}일 (약 {total_weeks:.1f}주)")

    # 60m 지표
    macro_60m['ema20'] = macro_60m['Close'].ewm(span=20, adjust=False).mean()
    long_60m['ema20'] = long_60m['Close'].ewm(span=20, adjust=False).mean()

    # 2. 피처 추출 및 사전 계산
    print("\n⏳ [2/4] 피처 엔지니어링 및 인과 괴리 크로스에셋 Veto 신호 계산 중...")
    moe = MoEMetaOrchestrator(confidence_threshold=0.55, gbdt_threshold=0.55, mode="hybrid_v3")
    
    # 15m 피처
    long_15m_feat = moe.gbdt_engine.extract_features(long_15m)
    long_15m_feat = moe.gbdt_engine.add_confidence_columns(long_15m_feat)
    short_15m_feat = moe.gbdt_engine.extract_features(short_15m)
    short_15m_feat.set_index('datetime', inplace=True, drop=False)

    # 5m 스나이퍼 피처 & 모델
    sniper = PowerHourSniper(confidence_threshold=0.55)
    feat_5m_long = sniper.ml_engine.extract_features(long_5m)
    X_5m_vec = feat_5m_long[sniper.feature_cols].fillna(0.0)
    probs_5m = sniper.model.predict_proba(X_5m_vec) # [p_short, p_flat, p_long]
    feat_5m_long['p_short'] = probs_5m[:, 0]
    feat_5m_long['p_flat'] = probs_5m[:, 1]
    feat_5m_long['p_long'] = probs_5m[:, 2]

    cross_mod = CrossAssetDislocationModel(dislocation_z_threshold=1.6)

    # 15m 크로스에셋 lookup
    nvda_15m_map = nvda_15m.set_index('datetime')['Close'].to_dict()
    macro_15m_map = macro_15m.set_index('datetime')['Close'].to_dict() if not macro_15m.empty else {}
    qqq_15m_map  = qqq_15m.set_index('datetime')['Close'].to_dict()
    vix_15m_map  = vix_15m.set_index('datetime')['Close'].to_dict()
    long_15m_close = long_15m['Close'].values
    long_15m_dt = long_15m['datetime'].values

    cross_dirs_15m = []
    for i in range(len(long_15m)):
        if i < 5:
            cross_dirs_15m.append("NONE")
            continue
        cur_t = long_15m_dt[i]
        past_5_t = long_15m_dt[i-5]
        s_r = float(long_15m_close[i] / long_15m_close[i-5] - 1.0)
        if (cur_t in nvda_15m_map and past_5_t in nvda_15m_map and
            cur_t in qqq_15m_map and past_5_t in qqq_15m_map and
            cur_t in vix_15m_map and past_5_t in vix_15m_map):
            n_r = float(nvda_15m_map[cur_t] / nvda_15m_map[past_5_t] - 1.0)
            sx_r = float(macro_15m_map[cur_t] / macro_15m_map[past_5_t] - 1.0) if (cur_t in macro_15m_map and past_5_t in macro_15m_map) else n_r
            q_r = float(qqq_15m_map[cur_t] / qqq_15m_map[past_5_t] - 1.0)
            v_r = float(vix_15m_map[cur_t] / vix_15m_map[past_5_t] - 1.0)
            sig_code, _, _ = cross_mod.predict_signal(
                long_ret=s_r, nvda_ret=n_r, macro_ret=sx_r, qqq_ret=q_r, vix_ret=v_r, tnx_ret=0.0
            )
            c_dir = "LONG" if sig_code > 0 else ("SHORT" if sig_code < 0 else "NONE")
        else:
            c_dir = "NONE"
        cross_dirs_15m.append(c_dir)
    long_15m_feat['cross_dir'] = cross_dirs_15m
    long_15m_feat.set_index('datetime', inplace=True, drop=False)

    # 5m 크로스에셋 lookup
    nvda_5m_map = nvda_5m.set_index('datetime')['Close'].to_dict()
    macro_5m_map = macro_5m.set_index('datetime')['Close'].to_dict() if not macro_5m.empty else {}
    qqq_5m_map  = qqq_5m.set_index('datetime')['Close'].to_dict()
    vix_5m_map  = vix_5m.set_index('datetime')['Close'].to_dict()
    long_5m_close = long_5m['Close'].values
    long_5m_dt = long_5m['datetime'].values

    cross_dirs_5m = []
    for i in range(len(long_5m)):
        if i < 5:
            cross_dirs_5m.append("HOLD")
            continue
        cur_t = long_5m_dt[i]
        past_5_t = long_5m_dt[i-5]
        s_r = float(long_5m_close[i] / long_5m_close[i-5] - 1.0)
        if (cur_t in nvda_5m_map and past_5_t in nvda_5m_map and
            cur_t in qqq_5m_map and past_5_t in qqq_5m_map and
            cur_t in vix_5m_map and past_5_t in vix_5m_map):
            n_r = float(nvda_5m_map[cur_t] / nvda_5m_map[past_5_t] - 1.0)
            sx_r = float(macro_5m_map[cur_t] / macro_5m_map[past_5_t] - 1.0) if (cur_t in macro_5m_map and past_5_t in macro_5m_map) else n_r
            q_r = float(qqq_5m_map[cur_t] / qqq_5m_map[past_5_t] - 1.0)
            v_r = float(vix_5m_map[cur_t] / vix_5m_map[past_5_t] - 1.0)
            sig_code, _, _ = cross_mod.predict_signal(
                long_ret=s_r, nvda_ret=n_r, macro_ret=sx_r, qqq_ret=q_r, vix_ret=v_r, tnx_ret=0.0
            )
            c_dir = "LONG" if sig_code > 0 else ("SHORT" if sig_code < 0 else "HOLD")
        else:
            c_dir = "HOLD"
        cross_dirs_5m.append(c_dir)
    feat_5m_long['cross_dir'] = cross_dirs_5m
    feat_5m_long.set_index('datetime', inplace=True, drop=False)

    doe_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    INITIAL_CAPITAL = 10_000.0

    # =========================================================================
    # [3/4] 15분봉 메인 모델 (Phase 1: 09:30 ~ 14:30 EDT) DoE 백테스트
    # =========================================================================
    print("\n⏳ [3/4] 15분봉 메인 모델 (09:30~14:30) 7대 확신도 DoE 시뮬레이션 가동 중...")
    results_15m = []

    TP_15M = 0.035
    SL_15M = -0.020
    TIME_STOP_BARS_15M = 18  # 90분 (5m * 18)
    SLIPPAGE = 0.03
    FEE_RATE = 0.0020

    for T in doe_thresholds:
        t_str = f"{int(T*100)}%"
        capital = INITIAL_CAPITAL
        trades = []
        equity_curve = [capital]
        daily_stoploss_count = 0

        for d_str in unique_dates:
            day_15 = long_15m_feat[long_15m_feat['date_str'] == d_str]
            day_5_l = long_5m[long_5m['date_str'] == d_str]
            day_5_s = short_5m[short_5m['date_str'] == d_str]
            if len(day_15) < 5:
                continue

            daily_stoploss_count = 0
            b_idx = 0
            n_bars = len(day_15)

            while b_idx < n_bars:
                row_15 = day_15.iloc[b_idx]
                cur_time = row_15['datetime']
                time_str = row_15['time_str']

                if b_idx < 1 or time_str > "14:30" or daily_stoploss_count >= 3:
                    b_idx += 1
                    continue

                dir_gbdt = row_15['Direction']
                conf_gbdt = float(row_15['Confidence'])
                dir_cross = row_15['cross_dir']

                is_gbdt_trigger = (dir_gbdt in ["LONG", "SHORT"]) and (conf_gbdt >= T)
                is_cross_veto = (
                    (dir_gbdt == "LONG" and dir_cross == "SHORT") or
                    (dir_gbdt == "SHORT" and dir_cross == "LONG")
                )

                if not is_gbdt_trigger or is_cross_veto:
                    b_idx += 1
                    continue

                direction = dir_gbdt

                # Screen 1: 60m trend
                past_macro_60 = macro_60m[macro_60m['datetime'] <= cur_time]
                past_long_60 = long_60m[long_60m['datetime'] <= cur_time]
                is_60m_ok = True
                if len(past_macro_60) >= 20 and len(past_long_60) >= 20:
                    macro_c = past_macro_60['Close'].iloc[-1]
                    long_c = past_long_60['Close'].iloc[-1]
                    macro_ema = past_macro_60['ema20'].iloc[-1]
                    long_ema = past_long_60['ema20'].iloc[-1]
                    if direction == "LONG":
                        is_60m_ok = (macro_c >= macro_ema * 0.998) and (long_c >= long_ema * 0.998)
                    elif direction == "SHORT":
                        is_60m_ok = (macro_c <= macro_ema * 1.002)

                if not is_60m_ok:
                    b_idx += 1
                    continue

                # Screen 3: Dip filter
                if direction == "LONG":
                    vwap_diff = float(row_15.get("VWAP_Diff", 0.0))
                    rsi_14 = float(row_15.get("RSI_14", 50.0))
                    bb_lower = float(row_15.get("BB_Lower", 0.0))
                    cur_close = float(row_15['Close'])
                    dip_ok = (vwap_diff <= 1.5) and (rsi_14 <= 62.0)
                    if bb_lower > 0:
                        dip_ok = dip_ok and (cur_close >= bb_lower * 1.001)
                else:
                    if cur_time in short_15m_feat.index:
                        row_s = short_15m_feat.loc[cur_time]
                        vwap_diff = float(row_s.get("VWAP_Diff", 0.0))
                        rsi_14 = float(row_s.get("RSI_14", 50.0))
                        bb_lower = float(row_s.get("BB_Lower", 0.0))
                        cur_close = float(row_s['Close'])
                        dip_ok = (vwap_diff <= 1.5) and (rsi_14 <= 62.0)
                        if bb_lower > 0:
                            dip_ok = dip_ok and (cur_close >= bb_lower * 1.001)
                    else:
                        dip_ok = False

                if not dip_ok:
                    b_idx += 1
                    continue

                chosen_sym = config.BASE_ASSET_LONG if direction == "LONG" else config.BASE_ASSET_SHORT
                base_px = float(row_15['Close']) if chosen_sym == config.BASE_ASSET_LONG else (float(short_15m_feat.loc[cur_time]['Close']) if cur_time in short_15m_feat.index else 40.0)
                entry_px = round(base_px + SLIPPAGE, 2)
                shares = int(capital / entry_px)
                invested = shares * entry_px
                if shares <= 0:
                    b_idx += 1
                    continue

                target_5m = day_5_l if chosen_sym == config.BASE_ASSET_LONG else day_5_s
                post_5m = target_5m[target_5m['datetime'] > cur_time]
                if post_5m.empty:
                    b_idx += 1
                    continue

                tp_px = round(entry_px * (1 + TP_15M), 2)
                sl_px = round(entry_px * (1 + SL_15M), 2)

                exit_px = entry_px
                exit_reason = "NONE"
                bars_held = 0

                eval_5m = post_5m.iloc[:TIME_STOP_BARS_15M]
                for k in range(len(eval_5m)):
                    c5 = eval_5m.iloc[k]
                    t5 = c5['time_str']
                    c5_o = float(c5['Open'])
                    c5_h = float(c5['High'])
                    c5_l = float(c5['Low'])
                    c5_c = float(c5['Close'])
                    bars_held = k + 1

                    hit_tp = (c5_h >= tp_px)
                    hit_sl = (c5_l <= sl_px)

                    # Intra-bar Illusion 해소 (양봉/음봉 궤적 추적)
                    if hit_tp and hit_sl:
                        if c5_c >= c5_o:
                            exit_px = round(tp_px - SLIPPAGE, 2)
                            exit_reason = "TAKE_PROFIT"
                            break
                        else:
                            exit_px = round(sl_px - SLIPPAGE, 2)
                            exit_reason = "STOP_LOSS"
                            daily_stoploss_count += 1
                            break
                    elif hit_tp:
                        exit_px = round(tp_px - SLIPPAGE, 2)
                        exit_reason = "TAKE_PROFIT"
                        break
                    elif hit_sl:
                        exit_px = round(sl_px - SLIPPAGE, 2)
                        exit_reason = "STOP_LOSS"
                        daily_stoploss_count += 1
                        break
                    elif t5 >= "15:45":
                        exit_px = round(c5_c - SLIPPAGE, 2)
                        exit_reason = "EOD"
                        if (exit_px - entry_px) / entry_px <= -0.020:
                            daily_stoploss_count += 1
                        break
                    elif k == len(eval_5m) - 1:
                        exit_px = round(c5_c - SLIPPAGE, 2)
                        exit_reason = "TIMESTOP"
                        if (exit_px - entry_px) / entry_px <= -0.020:
                            daily_stoploss_count += 1
                        break

                cost_amount = invested * FEE_RATE
                net_pnl = (shares * (exit_px - entry_px)) - cost_amount
                capital += net_pnl
                equity_curve.append(capital)

                ret = net_pnl / invested
                trades.append({
                    "symbol": chosen_sym,
                    "ret": ret,
                    "exit_reason": exit_reason,
                    "bars_held": bars_held
                })

                consumed_15m = int(np.ceil(bars_held / 3.0))
                b_idx += max(1, consumed_15m)

        tot_tr = len(trades)
        w_tr = [t for t in trades if t['ret'] > 0]
        l_tr = [t for t in trades if t['ret'] <= 0]
        wr = (len(w_tr) / tot_tr * 100) if tot_tr > 0 else 0.0
        gross_win = sum([t['ret'] for t in w_tr])
        gross_loss = abs(sum([t['ret'] for t in l_tr])) if l_tr else 0.0001
        pf = (gross_win / gross_loss) if gross_loss > 0 else 99.9
        cum_ret = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100.0

        eq_arr = np.array(equity_curve)
        pk = np.maximum.accumulate(eq_arr)
        dd = (pk - eq_arr) / pk * 100.0
        mdd = np.max(dd) if len(dd) > 0 else 0.0

        long_cnt = len([t for t in trades if t['symbol'] == config.BASE_ASSET_LONG])
        long_w = len([t for t in trades if t['symbol'] == config.BASE_ASSET_LONG and t['ret'] > 0])
        long_wr = (long_w / long_cnt * 100) if long_cnt > 0 else 0.0

        short_cnt = len([t for t in trades if t['symbol'] == config.BASE_ASSET_SHORT])
        short_w = len([t for t in trades if t['symbol'] == config.BASE_ASSET_SHORT and t['ret'] > 0])
        short_wr = (short_w / short_cnt * 100) if short_cnt > 0 else 0.0

        results_15m.append({
            "threshold": t_str,
            "total_trades": tot_tr,
            "trades_per_week": round(tot_tr / total_weeks, 2),
            "wins": len(w_tr),
            "losses": len(l_tr),
            "win_rate": wr,
            "long_trades": long_cnt,
            "long_wr": long_wr,
            "short_trades": short_cnt,
            "short_wr": short_wr,
            "cum_return": cum_ret,
            "profit_factor": pf,
            "mdd": mdd,
            "final_capital": capital
        })

    # =========================================================================
    # [4/4] 5분봉 스나이퍼 모델 (Phase 2: 14:30 ~ 15:30 EDT) DoE 백테스트
    # =========================================================================
    print("⏳ [4/4] 5분봉 스나이퍼 모델 (14:30~15:30) 7대 확신도 DoE 시뮬레이션 가동 중...")
    results_5m = []

    TP_5M = 0.025
    SL_5M = -0.0167
    TIME_STOP_BARS_5M = 6  # 30분 (5m * 6)

    for T in doe_thresholds:
        t_str = f"{int(T*100)}%"
        capital = INITIAL_CAPITAL
        trades = []
        equity_curve = [capital]

        for d_str in unique_dates:
            day_5_l = feat_5m_long[feat_5m_long['date_str'] == d_str]
            day_5_s = short_5m[short_5m['date_str'] == d_str]
            if len(day_5_l) < 20:
                continue

            idx = 0
            n_bars = len(day_5_l)

            while idx < n_bars:
                row_5 = day_5_l.iloc[idx]
                time_str = row_5['time_str']

                # 파워 아워 구간 (14:30 ~ 15:30)
                if not ('14:30' <= time_str < '15:30'):
                    idx += 1
                    continue

                cur_dt = row_5['datetime_dt']
                cur_t_str = row_5['datetime']

                # 5m GBDT 확률
                p_short = float(row_5['p_short'])
                p_flat = float(row_5['p_flat'])
                p_long = float(row_5['p_long'])

                direction = "NONE"
                confidence = p_flat
                if p_long >= p_short and p_long >= p_flat:
                    direction = "LONG"
                    confidence = p_long
                elif p_short >= p_long and p_short >= p_flat:
                    direction = "SHORT"
                    confidence = p_short

                is_gbdt_trigger = (direction in ["LONG", "SHORT"]) and (confidence >= T)
                if not is_gbdt_trigger:
                    idx += 1
                    continue

                # Cross-Asset Veto
                dir_cross = row_5.get('cross_dir', 'HOLD')
                is_cross_veto = (
                    (direction == "LONG" and dir_cross == "SHORT") or
                    (direction == "SHORT" and dir_cross == "LONG")
                )
                if is_cross_veto:
                    idx += 1
                    continue

                # Screen 1: 60m trend
                past_macro = macro_60m[macro_60m['datetime'] <= cur_t_str]
                if len(past_macro) < 20:
                    idx += 1
                    continue
                last_macro = past_macro.iloc[-1]
                if direction == "LONG" and (last_macro['Close'] < last_macro['ema20'] * 0.998):
                    idx += 1
                    continue
                elif direction == "SHORT" and (last_macro['Close'] > last_macro['ema20'] * 1.002):
                    idx += 1
                    continue

                # Screen 3: RSI dip
                rsi_5m = float(row_5.get("RSI_14", 50.0))
                if direction == "LONG" and rsi_5m > 68.0:
                    idx += 1
                    continue
                elif direction == "SHORT" and rsi_5m < 32.0:
                    idx += 1
                    continue

                chosen_sym = config.BASE_ASSET_LONG if direction == "LONG" else config.BASE_ASSET_SHORT
                base_px = float(row_5['Close']) if chosen_sym == config.BASE_ASSET_LONG else float(day_5_s[day_5_s['datetime'] == cur_t_str]['Close'].iloc[0]) if not day_5_s[day_5_s['datetime'] == cur_t_str].empty else 40.0
                entry_px = round(base_px + SLIPPAGE, 2)
                shares = int(capital / entry_px)
                invested = shares * entry_px
                if shares <= 0:
                    idx += 1
                    continue

                target_day = day_5_l if chosen_sym == config.BASE_ASSET_LONG else day_5_s
                sub_5m = target_day.iloc[idx:]
                tp_px = round(entry_px * (1 + TP_5M), 2)
                sl_px = round(entry_px * (1 + SL_5M), 2)

                exit_px = entry_px
                exit_reason = "NONE"
                bars_held = 0

                for s_idx in range(len(sub_5m)):
                    c_bar = sub_5m.iloc[s_idx]
                    bars_held = s_idx + 1
                    t_bar_str = c_bar['time_str']
                    c_h = float(c_bar['High'])
                    c_l = float(c_bar['Low'])
                    c_c = float(c_bar['Close'])

                    hit_tp = (c_h >= tp_px)
                    hit_sl = (c_l <= sl_px)

                    if hit_tp and hit_sl:
                        if c_c >= float(c_bar['Open']):
                            exit_px = round(tp_px - SLIPPAGE, 2)
                            exit_reason = "TAKE_PROFIT"
                            break
                        else:
                            exit_px = round(sl_px - SLIPPAGE, 2)
                            exit_reason = "STOP_LOSS"
                            break
                    elif hit_tp:
                        exit_px = round(tp_px - SLIPPAGE, 2)
                        exit_reason = "TAKE_PROFIT"
                        break
                    elif hit_sl:
                        exit_px = round(sl_px - SLIPPAGE, 2)
                        exit_reason = "STOP_LOSS"
                        break
                    elif bars_held >= TIME_STOP_BARS_5M:
                        exit_px = round(c_c - SLIPPAGE, 2)
                        exit_reason = "TIMESTOP"
                        break
                    elif t_bar_str >= '15:45':
                        exit_px = round(c_c - SLIPPAGE, 2)
                        exit_reason = "EOD"
                        break

                cost_amount = invested * FEE_RATE
                net_pnl = (shares * (exit_px - entry_px)) - cost_amount
                capital += net_pnl
                equity_curve.append(capital)

                ret = net_pnl / invested
                trades.append({
                    "symbol": chosen_sym,
                    "ret": ret,
                    "exit_reason": exit_reason,
                    "bars_held": bars_held
                })

                idx += bars_held

        tot_tr = len(trades)
        w_tr = [t for t in trades if t['ret'] > 0]
        l_tr = [t for t in trades if t['ret'] <= 0]
        wr = (len(w_tr) / tot_tr * 100) if tot_tr > 0 else 0.0
        gross_win = sum([t['ret'] for t in w_tr])
        gross_loss = abs(sum([t['ret'] for t in l_tr])) if l_tr else 0.0001
        pf = (gross_win / gross_loss) if gross_loss > 0 else 99.9
        cum_ret = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100.0

        eq_arr = np.array(equity_curve)
        pk = np.maximum.accumulate(eq_arr)
        dd = (pk - eq_arr) / pk * 100.0
        mdd = np.max(dd) if len(dd) > 0 else 0.0

        long_cnt = len([t for t in trades if t['symbol'] == config.BASE_ASSET_LONG])
        long_w = len([t for t in trades if t['symbol'] == config.BASE_ASSET_LONG and t['ret'] > 0])
        long_wr = (long_w / long_cnt * 100) if long_cnt > 0 else 0.0

        short_cnt = len([t for t in trades if t['symbol'] == config.BASE_ASSET_SHORT])
        short_w = len([t for t in trades if t['symbol'] == config.BASE_ASSET_SHORT and t['ret'] > 0])
        short_wr = (short_w / short_cnt * 100) if short_cnt > 0 else 0.0

        results_5m.append({
            "threshold": t_str,
            "total_trades": tot_tr,
            "trades_per_week": round(tot_tr / total_weeks, 2),
            "wins": len(w_tr),
            "losses": len(l_tr),
            "win_rate": wr,
            "long_trades": long_cnt,
            "long_wr": long_wr,
            "short_trades": short_cnt,
            "short_wr": short_wr,
            "cum_return": cum_ret,
            "profit_factor": pf,
            "mdd": mdd,
            "final_capital": capital
        })

    return results_15m, results_5m

if __name__ == "__main__":
    r15, r5 = run_comprehensive_dual_doe()
    
    headers = ["확신도", "총거래", "주당거래", "승 / 패", "승률(%)", "TQQQ(건/승률)", "SQQQ(건/승률)", "누적수익률", "손익비(PF)", "MDD(%)", "최종자본금($)"]
    aligns = [":---:", ":---:", ":---:", ":---:", ":---:", ":---:", ":---:", "---:", ":---:", ":---:", "---:"]

    print("\n" + "=" * 110)
    print("📊 [1. Phase 1 메인 15분봉 모델 (09:30 ~ 14:30 EDT) DoE 결과표 (크로스에셋 Veto 연계)]")
    print("=" * 110)
    rows_15 = []
    for m in r15:
        rows_15.append([
            m["threshold"], f"{m['total_trades']}회", f"{m['trades_per_week']}회",
            f"{m['wins']}승 {m['losses']}패", f"{m['win_rate']:.1f}%",
            f"{m['long_trades']}건 ({m['long_wr']:.1f}%)", f"{m['short_trades']}건 ({m['short_wr']:.1f}%)",
            f"{m['cum_return']:+.2f}%", f"{m['profit_factor']:.2f}", f"{m['mdd']:.2f}%", f"${m['final_capital']:,.2f}"
        ])
    print(format_markdown_table(headers, rows_15, aligns))

    print("\n" + "=" * 110)
    print("⚡ [2. Phase 2 파워 아워 5분봉 스나이퍼 (14:30 ~ 15:30 EDT) DoE 결과표 (크로스에셋 Veto 연계)]")
    print("=" * 110)
    rows_5 = []
    for m in r5:
        rows_5.append([
            m["threshold"], f"{m['total_trades']}회", f"{m['trades_per_week']}회",
            f"{m['wins']}승 {m['losses']}패", f"{m['win_rate']:.1f}%",
            f"{m['long_trades']}건 ({m['long_wr']:.1f}%)", f"{m['short_trades']}건 ({m['short_wr']:.1f}%)",
            f"{m['cum_return']:+.2f}%", f"{m['profit_factor']:.2f}", f"{m['mdd']:.2f}%", f"${m['final_capital']:,.2f}"
        ])
    print(format_markdown_table(headers, rows_5, aligns))
    print("=" * 110)
