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
from core.moe_orchestrator import MoEMetaOrchestrator
from core.heterogeneous_models import CrossAssetDislocationModel

def run_chronological_5m_backtest():
    print("=" * 125)
    print("🏛️ [Lumos V3: 시간 흐름에 따른 5분봉 정밀 궤적(Chronological 5m) 실전 무결성 백테스트]")
    print(f"⏰ 실행 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} KST")
    print("=" * 125)

    # 1. 시계열 데이터 로드
    lake = MarketDataLake()
    print("⏳ [1/4] 데이터 레이크에서 다중 타임프레임 캔들 로드 중...")

    long_15m = lake.load_candles(config.BASE_ASSET_LONG, "15m")
    short_15m = lake.load_candles(config.BASE_ASSET_SHORT, "15m")
    long_5m  = lake.load_candles(config.BASE_ASSET_LONG, "5m")
    short_5m  = lake.load_candles(config.BASE_ASSET_SHORT, "5m")
    macro_60m = lake.load_candles(config.MACRO_TREND_SYMBOL, "60m")
    long_60m = lake.load_candles(config.BASE_ASSET_LONG, "60m")
    nvda_15m = lake.load_candles("NVDA", "15m")
    qqq_15m  = lake.load_candles("QQQ", "15m")
    vix_15m  = lake.load_candles("^VIX", "15m")
    macro_15m = lake.load_candles(config.MACRO_TREND_SYMBOL, "15m")

    # Datetime 정규화
    for df in [long_15m, short_15m, long_5m, short_5m, macro_60m, long_60m, macro_15m, nvda_15m, qqq_15m, vix_15m]:
        if 'datetime' in df.columns:
            df['datetime_dt'] = pd.to_datetime(df['datetime'])
            df['date_str'] = df['datetime_dt'].dt.strftime('%Y-%m-%d')
            df['time_str'] = df['datetime_dt'].dt.strftime('%H:%M')
        else:
            df['datetime_dt'] = pd.to_datetime(df.index)
            df['date_str'] = df['datetime_dt'].dt.strftime('%Y-%m-%d')
            df['time_str'] = df['datetime_dt'].dt.strftime('%H:%M')

    # Screen 1용 60m EMA20
    macro_60m['ema20'] = macro_60m['Close'].ewm(span=20, adjust=False).mean()
    long_60m['ema20'] = long_60m['Close'].ewm(span=20, adjust=False).mean()

    # 2. MoE 모델 및 피처 엔지니어링
    print("⏳ [2/4] GBDT 3-Class 및 크로스에셋(Cross-Asset) 인과 괴리 사전 피처 추출 중...")
    moe = MoEMetaOrchestrator(confidence_threshold=0.60, gbdt_threshold=0.60, mode="hybrid_v3")

    long_15m_feat = moe.gbdt_engine.extract_features(long_15m)
    long_15m_feat = moe.gbdt_engine.add_confidence_columns(long_15m_feat)

    short_15m_feat = moe.gbdt_engine.extract_features(short_15m)
    short_15m_feat.set_index('datetime', inplace=True, drop=False)

    # 크로스에셋 모델 신호 계산 (패치된 단위 자동정규화 및 파라미터 적용)
    cross_mod = CrossAssetDislocationModel(dislocation_z_threshold=1.6)
    nvda_map = nvda_15m.set_index('datetime')['Close'].to_dict()
    macro_map = macro_15m.set_index('datetime')['Close'].to_dict() if not macro_15m.empty else {}
    qqq_map  = qqq_15m.set_index('datetime')['Close'].to_dict()
    vix_map  = vix_15m.set_index('datetime')['Close'].to_dict()
    long_close_arr = long_15m['Close'].values
    long_dt_arr = long_15m['datetime'].values

    cross_dirs = []
    cross_confs = []

    for i in range(len(long_15m)):
        if i < 5:
            cross_dirs.append("NONE")
            cross_confs.append(0.50)
            continue

        c_t = long_dt_arr[i]
        p_t = long_dt_arr[i-5]

        s_r = float(long_close_arr[i] / long_close_arr[i-5] - 1.0)
        if (c_t in nvda_map and p_t in nvda_map and
            c_t in qqq_map and p_t in qqq_map and
            c_t in vix_map and p_t in vix_map):
            n_r = float(nvda_map[c_t] / nvda_map[p_t] - 1.0)
            sx_r = float(macro_map[c_t] / macro_map[p_t] - 1.0) if (c_t in macro_map and p_t in macro_map) else n_r
            q_r = float(qqq_map[c_t] / qqq_map[p_t] - 1.0)
            v_r = float(vix_map[c_t] / vix_map[p_t] - 1.0)

            sig_code, exp_conf, _ = cross_mod.predict_signal(
                long_ret=s_r,
                nvda_ret=n_r,
                macro_ret=sx_r,
                qqq_ret=q_r,
                vix_ret=v_r,
                tnx_ret=0.0
            )
            c_dir = "LONG" if sig_code > 0 else ("SHORT" if sig_code < 0 else "NONE")
            c_conf = exp_conf
        else:
            c_dir = "NONE"
            c_conf = 0.50

        cross_dirs.append(c_dir)
        cross_confs.append(c_conf)

    long_15m_feat['cross_dir'] = cross_dirs
    long_15m_feat['cross_conf'] = cross_confs
    long_15m_feat.set_index('datetime', inplace=True, drop=False)

    unique_dates = sorted(long_15m_feat['date_str'].unique())

    # 3. 실전 하드룰 파라미터 (Hard Rules)
    INITIAL_CAPITAL = 10_000.0   # $10,000 USD (복리 100% 자본 투입)
    GBDT_THRESHOLD = 0.60        # GBDT >= 60% 공격수 트리거
    TP_PCT = 0.030               # +3.0% 목표 익절
    SL_PCT = -0.020              # -2.0% 칼손절
    TIME_STOP_BARS = 18          # 90분 (5분봉 18개)
    SLIPPAGE = 0.03              # 편도 $0.03 슬리피지/페이업
    FEE_RATE = 0.0020            # 0.20% 증권사 왕복 수수료

    print("⏳ [3/4] 시간 흐름에 따른 일자별 / 5분봉 캔들 단위 정밀 시뮬레이션 실행 중...")

    capital = INITIAL_CAPITAL
    trades = []
    equity_curve = [capital]
    monthly_stats = {}

    for d_str in unique_dates:
        day_long_15 = long_15m_feat[long_15m_feat['date_str'] == d_str]
        if len(day_long_15) < 5:
            continue

        day_long_5 = long_5m[long_5m['date_str'] == d_str]
        day_short_5 = short_5m[short_5m['date_str'] == d_str]

        # [Rule 5] 매 정규장 개장(09:30) 시 일일 서킷 브레이커 카운트 0 리셋
        daily_stoploss_count = 0
        b_idx = 0
        n_bars = len(day_long_15)

        # 시간 흐름 순서대로 15분봉 바 스캔
        while b_idx < n_bars:
            cur_15m_row = day_long_15.iloc[b_idx]
            cur_15m_time = cur_15m_row['datetime']
            time_str = cur_15m_row['time_str']

            # [Rule 1 & 5] 14:30 EDT 이후 신규 진입 칼차단 (No-Entry Cutoff)
            if b_idx < 1 or time_str > "14:30":
                b_idx += 1
                continue

            # [Rule 5] 일일 3-Out 서킷 브레이커 발동 시 당일 추가 진입 전면 차단
            if daily_stoploss_count >= 3:
                b_idx += 1
                continue

            # [Rule 1] 하이브리드 MoE V3 의사결정
            dir_gbdt = cur_15m_row['Direction']
            conf_gbdt = float(cur_15m_row['Confidence'])
            dir_cross = cur_15m_row['cross_dir']

            # 공격수: GBDT >= 60%
            is_gbdt_trigger = (dir_gbdt in ["LONG", "SHORT"]) and (conf_gbdt >= GBDT_THRESHOLD)

            # 방패: 크로스에셋 정반대 방향 시 100% Veto 차단
            is_cross_veto = (
                (dir_gbdt == "LONG" and dir_cross == "SHORT") or
                (dir_gbdt == "SHORT" and dir_cross == "LONG")
            )

            if not is_gbdt_trigger or is_cross_veto:
                b_idx += 1
                continue

            direction = dir_gbdt

            # [Rule 1] 3중 스크린 검증 (Look-ahead bias 방지: t시점 이전 데이터만 사용)
            # Screen 1: 상위 60분봉 추세 정렬
            past_macro_60 = macro_60m[macro_60m['datetime'] <= cur_15m_time]
            past_long_60 = long_60m[long_60m['datetime'] <= cur_15m_time]

            is_60m_trend_ok = True
            if len(past_macro_60) >= 20 and len(past_long_60) >= 20:
                macro_c = past_macro_60['Close'].iloc[-1]
                long_c = past_long_60['Close'].iloc[-1]
                macro_ema20 = past_macro_60['ema20'].iloc[-1]
                long_ema20 = past_long_60['ema20'].iloc[-1]

                if direction == "LONG":
                    is_60m_trend_ok = (macro_c >= macro_ema20 * 0.998) and (long_c >= long_ema20 * 0.998)
                elif direction == "SHORT":
                    is_60m_trend_ok = (macro_c <= macro_ema20 * 1.002)

            if not is_60m_trend_ok:
                b_idx += 1
                continue

            # Screen 3: 단기 눌림목 타점 검증
            if direction == "LONG":
                vwap_diff = float(cur_15m_row.get("VWAP_Diff", 0.0))
                rsi_14 = float(cur_15m_row.get("RSI_14", 50.0))
                bb_lower = float(cur_15m_row.get("BB_Lower", 0.0))
                cur_close = float(cur_15m_row['Close'])
                dip_ok = (vwap_diff <= 1.5) and (rsi_14 <= 62.0)
                if bb_lower > 0:
                    dip_ok = dip_ok and (cur_close >= bb_lower * 1.001)
            else:
                if cur_15m_time in short_15m_feat.index:
                    row_s = short_15m_feat.loc[cur_15m_time]
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

            # -------------------------------------------------------------
            # [진입 집행: 3대 인터락 통과]
            # -------------------------------------------------------------
            chosen_symbol = config.BASE_ASSET_LONG if direction == "LONG" else config.BASE_ASSET_SHORT
            base_px = float(cur_15m_row['Close']) if chosen_symbol == config.BASE_ASSET_LONG else float(short_15m_feat.loc[cur_15m_time]['Close'])
            entry_px = round(base_px + SLIPPAGE, 2)
            shares = int(capital / entry_px)
            invested = shares * entry_px

            if shares <= 0:
                b_idx += 1
                continue

            # -------------------------------------------------------------
            # [5분봉 정밀 궤적(Intraday 5m Path Tracking) 청산 감시]
            # -------------------------------------------------------------
            target_5m_df = day_long_5 if chosen_symbol == config.BASE_ASSET_LONG else day_short_5
            post_5m = target_5m_df[target_5m_df['datetime'] > cur_15m_time]

            if post_5m.empty:
                b_idx += 1
                continue

            tp_px = round(entry_px * (1 + TP_PCT), 2)
            sl_px = round(entry_px * (1 + SL_PCT), 2)

            exit_px = entry_px
            exit_reason = "90분 타임스탑"
            exit_time_str = ""
            bars_held_5m = 0

            eval_5m = post_5m.iloc[:TIME_STOP_BARS]

            for k in range(len(eval_5m)):
                c5 = eval_5m.iloc[k]
                t5_str = c5['time_str']
                c5_o = float(c5['Open'])
                c5_h = float(c5['High'])
                c5_l = float(c5['Low'])
                c5_c = float(c5['Close'])
                bars_held_5m = k + 1

                hit_tp = (c5_h >= tp_px)
                hit_sl = (c5_l <= sl_px)

                # 단일 5분봉 내 동시 터치 시 캔들 진행 방향(시가-종가)으로 순서 분별
                if hit_tp and hit_sl:
                    if c5_c >= c5_o:
                        exit_px = round(tp_px - SLIPPAGE, 2)
                        exit_reason = "🎯 목표익절 (+3.0%)"
                    else:
                        exit_px = round(sl_px - SLIPPAGE, 2)
                        exit_reason = "🛑 칼손절 (-2.0%)"
                        daily_stoploss_count += 1
                    exit_time_str = t5_str
                    break
                elif hit_tp:
                    exit_px = round(tp_px - SLIPPAGE, 2)
                    exit_reason = "🎯 목표익절 (+3.0%)"
                    exit_time_str = t5_str
                    break
                elif hit_sl:
                    exit_px = round(sl_px - SLIPPAGE, 2)
                    exit_reason = "🛑 칼손절 (-2.0%)"
                    daily_stoploss_count += 1
                    exit_time_str = t5_str
                    break
                elif t5_str >= '15:50' or k == len(eval_5m) - 1:
                    exit_px = round(c5_c - SLIPPAGE, 2)
                    exit_reason = "🌙 15:50 EOD 청산" if t5_str >= '15:50' else "⏱️ 90분 타임스탑"
                    exit_time_str = t5_str
                    if (exit_px - entry_px) / entry_px <= SL_PCT:
                        daily_stoploss_count += 1
                    break

            # 정밀 손익 계산 (수수료 및 슬리피지 완전 차감)
            cost = invested * FEE_RATE
            net_pnl = (shares * (exit_px - entry_px)) - cost
            capital += net_pnl
            equity_curve.append(capital)
            ret_pct = net_pnl / invested * 100.0

            trades.append({
                "trade_no": len(trades) + 1,
                "date": d_str,
                "symbol": chosen_symbol,
                "entry_t": time_str,
                "exit_t": exit_time_str,
                "entry_px": entry_px,
                "exit_px": exit_px,
                "shares": shares,
                "ret_pct": round(ret_pct, 2),
                "net_pnl": round(net_pnl, 2),
                "capital": round(capital, 2),
                "exit_reason": exit_reason,
                "holding_min": bars_held_5m * 5,
                "gbdt_conf": round(conf_gbdt * 100, 1),
                "cross_dir": dir_cross
            })

            # 단일 포지션 릴레이: 포지션 청산될 때까지 15분봉 스캐너 동결
            consumed_15m = int(np.ceil(bars_held_5m / 3.0))
            b_idx += max(1, consumed_15m)

    # 4. 성과 분석 및 지표 산출
    print("⏳ [4/4] 퀀트 헤지펀드 핵심 성과 지표 산출 중...")

    tot_trades = len(trades)
    wins = [t for t in trades if t['net_pnl'] > 0]
    losses = [t for t in trades if t['net_pnl'] <= 0]
    wr = (len(wins) / tot_trades * 100.0) if tot_trades > 0 else 0.0

    avg_win = np.mean([t['ret_pct'] for t in wins]) if wins else 0.0
    avg_loss = abs(np.mean([t['ret_pct'] for t in losses])) if losses else 0.0
    payoff = (avg_win / avg_loss) if avg_loss > 0 else 99.0

    p_w = len(wins) / tot_trades if tot_trades > 0 else 0.0
    p_l = len(losses) / tot_trades if tot_trades > 0 else 0.0
    expectancy = (p_w * avg_win) - (p_l * avg_loss)

    gross_win = sum([t['ret_pct'] for t in wins])
    gross_loss = abs(sum([t['ret_pct'] for t in losses])) if losses else 0.0001
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else 99.0

    cum_return = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100.0

    eq_arr = np.array(equity_curve)
    pk = np.maximum.accumulate(eq_arr)
    dd = (pk - eq_arr) / pk * 100.0
    mdd = np.max(dd) if len(dd) > 0 else 0.0

    avg_holding = np.mean([t['holding_min'] for t in trades]) if trades else 0.0

    # 청산 사유별 카운트
    tp_count = len([t for t in trades if "목표익절" in t['exit_reason']])
    sl_count = len([t for t in trades if "칼손절" in t['exit_reason']])
    ts_count = len([t for t in trades if "타임스탑" in t['exit_reason']])
    eod_count = len([t for t in trades if "EOD" in t['exit_reason']])

    # 월별 성과 집계
    df_trades = pd.DataFrame(trades)
    df_trades['month'] = df_trades['date'].str.slice(0, 7)
    month_summary = []
    for m, g in df_trades.groupby('month'):
        m_w = len(g[g['net_pnl'] > 0])
        m_l = len(g[g['net_pnl'] <= 0])
        m_tot = len(g)
        m_wr = m_w / m_tot * 100.0 if m_tot > 0 else 0.0
        m_pnl = g['net_pnl'].sum()
        month_summary.append({
            "월별": m,
            "거래수": f"{m_tot}회",
            "승/패": f"{m_w}승 {m_l}패",
            "승률": f"{m_wr:.1f}%",
            "월간 손익 ($)": f"{m_pnl:+,.2f}$"
        })
    df_month = pd.DataFrame(month_summary)

    # 출력 리포트
    print("\n" + "=" * 125)
    print("🏆 [Lumos V3: 시간 흐름 5분봉 정밀 궤적 백테스트 최종 성적표]")
    print("=" * 125)
    print(f"  • 시작 자본금      : ${INITIAL_CAPITAL:,.2f} USD")
    print(f"  • 최종 자본금      : ${capital:,.2f} USD (순이익: {capital - INITIAL_CAPITAL:+,.2f}$)")
    print(f"  • 누적 수익률      : {cum_return:+.2f}%")
    print(f"  • 최대 낙폭 (MDD)  : {mdd:.2f}%")
    print(f"  • 총 매매 횟수     : {tot_trades}회 ({len(wins)}승 {len(losses)}패)")
    print(f"  • 전체 승률        : {wr:.1f}%")
    print(f"  • 손익비 (PF)      : {profit_factor:.2f}")
    print(f"  • 평균 익절률      : +{avg_win:.2f}%")
    print(f"  • 평균 손실률      : -{avg_loss:.2f}%")
    print(f"  • 페이오프 비율    : {payoff:.2f}")
    print(f"  • 수학적 기대값(E) : {expectancy:+.2f}% / 거래당")
    print(f"  • 평균 보유 시간   : {avg_holding:.1f}분")
    print("=" * 125)

    print("\n📊 [청산 사유별 상세 집계]")
    print(f"  1) 🎯 목표익절 (+3.0%) : {tp_count}회 ({tp_count/tot_trades*100:.1f}%)")
    print(f"  2) 🛑 칼손절   (-2.0%) : {sl_count}회 ({sl_count/tot_trades*100:.1f}%)")
    print(f"  3) ⏱️ 90분 타임스탑    : {ts_count}회 ({ts_count/tot_trades*100:.1f}%)")
    print(f"  4) 🌙 15:50 EOD 청산   : {eod_count}회 ({eod_count/tot_trades*100:.1f}%)")

    print("\n📅 [월별 성과 테이블]")
    print(df_month.to_string(index=False))

    print("\n🔍 [전수 거래 내역 (최근 15건 샘플)]")
    cols = ['trade_no', 'date', 'symbol', 'entry_t', 'exit_t', 'entry_px', 'exit_px', 'ret_pct', 'net_pnl', 'capital', 'exit_reason', 'holding_min']
    print(df_trades[cols].tail(15).to_string(index=False))
    print("=" * 125)

    return trades, df_trades, df_month

if __name__ == '__main__':
    run_chronological_5m_backtest()
