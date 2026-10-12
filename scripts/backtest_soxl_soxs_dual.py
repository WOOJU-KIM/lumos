import os
import sys
import json
import sqlite3
import config
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Windows 콘솔 utf-8 설정
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.data_lake import MarketDataLake
from core.heterogeneous_models import CrossAssetDislocationModel
from core.moe_orchestrator import MoEMetaOrchestrator

def format_markdown_table(df: pd.DataFrame, cols: list) -> str:
    """tabulate 패키지 의존성 없는 고속 마크다운 표 생성기"""
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join([":---:" if "횟수" in c or "승률" in c else "---" for c in cols]) + " |"
    rows = []
    for _, row in df.iterrows():
        row_str = "| " + " | ".join([str(row[c]) for c in cols]) + " |"
        rows.append(row_str)
    return "\n".join([header, separator] + rows)

def run_long_short_dual_backtest():
    print("=" * 100)
    print("🔬 [Lumos 퀀트 시스템: TQQQ(롱) vs SQQQ(숏) 독립 및 통합 듀얼 백테스트]")
    print(f"⏰ 실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S KST')}")
    print("🔒 [보안 확인] 본 검증은 scripts/ 내 격리 스크립트로만 구동되며, core/ 운영 코드는 단 1바이트도 수정되지 않습니다.")
    print("=" * 100)

    # 1. 다중 타임프레임 데이터 로드 (최근 504 거래일 롤링 윈도우)
    lake = MarketDataLake()
    long_15m = lake.load_rolling_candles(config.BASE_ASSET_LONG, '15m', max_trading_days=504)
    short_15m = lake.load_rolling_candles(config.BASE_ASSET_SHORT, '15m', max_trading_days=504)
    long_5m  = lake.load_candles(config.BASE_ASSET_LONG, '5m')
    short_5m  = lake.load_candles(config.BASE_ASSET_SHORT, '5m')
    macro_60m = lake.load_candles(config.MACRO_TREND_SYMBOL, '60m')
    long_60m = lake.load_candles(config.BASE_ASSET_LONG, '60m')
    nvda_15m = lake.load_candles('NVDA', '15m')
    qqq_15m  = lake.load_candles('QQQ', '15m')
    vix_15m  = lake.load_candles('^VIX', '15m')

    for df in [long_15m, short_15m, long_5m, short_5m, macro_60m, long_60m, nvda_15m, qqq_15m, vix_15m]:
        df['date_str'] = df.index.strftime('%Y-%m-%d')
        df['time_str'] = df.index.strftime('%H:%M')

    unique_dates = sorted(long_15m['date_str'].unique())
    num_days = len(unique_dates)
    num_months = num_days / 21.0

    # 2. 피처 사전 계산
    macro_60m['ema20'] = macro_60m['Close'].ewm(span=20, adjust=False).mean()
    long_60m['ema20'] = long_60m['Close'].ewm(span=20, adjust=False).mean()

    moe = MoEMetaOrchestrator()
    long_15m_feat = moe.gbdt_engine.extract_features(long_15m)
    long_15m_feat = moe.gbdt_engine.add_confidence_columns(long_15m_feat)
    short_15m_feat = moe.gbdt_engine.extract_features(short_15m)
    short_15m_feat = moe.gbdt_engine.add_confidence_columns(short_15m_feat)

    cross_mod = CrossAssetDislocationModel(dislocation_z_threshold=1.6)
    cross_sigs, cross_confs, cross_dirs = [], [], []

    for i in range(len(long_15m)):
        t = long_15m.index[i]
        past_nvda = nvda_15m[nvda_15m.index <= t]
        past_qqq  = qqq_15m[qqq_15m.index <= t]
        past_vix  = vix_15m[vix_15m.index <= t]
        past_long = long_15m[long_15m.index <= t]
        
        c_sig, c_conf, c_dir = 0, 0.50, "NONE"
        if len(past_nvda) >= 5 and len(past_qqq) >= 5 and len(past_vix) >= 5 and len(past_long) >= 5:
            n_r = float(past_nvda['Close'].iloc[-1] / past_nvda['Close'].iloc[-5] - 1.0)
            q_r = float(past_qqq['Close'].iloc[-1] / past_qqq['Close'].iloc[-5] - 1.0)
            v_r = float(past_vix['Close'].iloc[-1] / past_vix['Close'].iloc[-5] - 1.0)
            s_r = float(past_long['Close'].iloc[-1] / past_long['Close'].iloc[-5] - 1.0)
            
            sig_code, exp_conf, _ = cross_mod.predict_signal(
                long_ret=s_r, nvda_ret=n_r, qqq_ret=q_r, macro_ret=s_r, vix_ret=v_r, tnx_ret=0.0
            )
            c_sig = sig_code
            c_conf = exp_conf
            if sig_code > 0:
                c_dir = "LONG"
            elif sig_code < 0:
                c_dir = "SHORT"
                
        cross_sigs.append(c_sig)
        cross_confs.append(c_conf)
        cross_dirs.append(c_dir)

    long_15m_feat['cross_sig'] = cross_sigs
    long_15m_feat['cross_conf'] = cross_confs
    long_15m_feat['cross_dir'] = cross_dirs

    # 3. 3개 모드 설정: TQQQ 단독, SQQQ 단독, 통합 듀얼 챔피언
    INITIAL_CAPITAL = 10_000_000.0
    FRICTION_PCT = 0.0034           # 0.34% 마찰비용
    TP_PCT = 0.035                  # +3.5%
    SL_PCT = -0.020                 # -2.0%
    MAX_HOLD_5M_BARS = 18           # 90분
    C_TH = 0.60
    G_TH = 0.60

    modes = [
        {
            "id": "TQQQ_ONLY",
            "name": "TQQQ (3x 롱) 단독 운용",
            "allow_long": True,
            "allow_short": False,
            "desc": "상승장만 스나이핑 (하락장 100% 현금 대기)"
        },
        {
            "id": "SQQQ_ONLY",
            "name": "SQQQ (3x 숏) 단독 운용",
            "allow_long": False,
            "allow_short": True,
            "desc": "하락장만 인버스 스나이핑 (상승장 100% 현금 대기)"
        },
        {
            "id": "COMBINED",
            "name": "통합 듀얼 챔피언 (TQQQ + SQQQ)",
            "allow_long": True,
            "allow_short": True,
            "desc": "양방향 전천후 자율 스나이핑 (시나리오 C 챔피언)"
        }
    ]

    all_results = []
    trade_logs = {}

    for m in modes:
        capital = INITIAL_CAPITAL
        trades = []
        equity_curve = [capital]
        allow_l = m["allow_long"]
        allow_s = m["allow_short"]

        for d_str in unique_dates:
            day_long_15 = long_15m_feat[long_15m_feat['date_str'] == d_str]
            if len(day_long_15) < 5:
                continue

            day_long_5 = long_5m[long_5m['date_str'] == d_str]
            day_short_5 = short_5m[short_5m['date_str'] == d_str]

            b_idx = 0
            n_bars = len(day_long_15)

            while b_idx < n_bars:
                cur_15m_time = day_long_15.index[b_idx]
                time_str = cur_15m_time.strftime('%H:%M')

                if b_idx < 1 or time_str > "14:30":
                    b_idx += 1
                    continue

                row_l = day_long_15.iloc[b_idx]
                row_s = short_15m_feat.loc[cur_15m_time] if cur_15m_time in short_15m_feat.index else None

                # Screen 1
                past_macro = macro_60m[macro_60m.index <= cur_15m_time]
                past_long = long_60m[long_60m.index <= cur_15m_time]
                is_60m_bull = False
                is_60m_bear = False
                if len(past_macro) >= 20 and len(past_long) >= 20:
                    macro_c = past_macro['Close'].iloc[-1]
                    long_c = past_long['Close'].iloc[-1]
                    macro_ema = past_macro['ema20'].iloc[-1]
                    long_ema = past_long['ema20'].iloc[-1]
                    is_60m_bull = (macro_c >= macro_ema * 0.998) and (long_c >= long_ema * 0.998)
                    is_60m_bear = (macro_c <= macro_ema * 1.002)

                # Screen 3
                vwap_diff_l = float(row_l.get("VWAP_Diff", 0.0))
                rsi_14_l = float(row_l.get("RSI_14", 50.0))
                bb_lower_l = float(row_l.get("BB_Lower", 0.0))
                cur_close_l = float(row_l['Close'])
                dip_ok_long = (vwap_diff_l <= 1.5) and (rsi_14_l <= 62.0)
                if bb_lower_l > 0:
                    dip_ok_long = dip_ok_long and (cur_close_l >= bb_lower_l * 1.001)

                dip_ok_short = False
                cur_close_s = 40.0
                if row_s is not None:
                    vwap_diff_s = float(row_s.get("VWAP_Diff", 0.0))
                    rsi_14_s = float(row_s.get("RSI_14", 50.0))
                    bb_lower_s = float(row_s.get("BB_Lower", 0.0))
                    cur_close_s = float(row_s['Close'])
                    dip_ok_short = (vwap_diff_s <= 1.5) and (rsi_14_s <= 62.0)
                    if bb_lower_s > 0:
                        dip_ok_short = dip_ok_short and (cur_close_s >= bb_lower_s * 1.001)

                dir_cross = row_l['cross_dir']
                conf_cross = row_l['cross_conf']
                dir_gbdt = row_l['Direction']
                conf_gbdt = row_l['Confidence']

                chosen_symbol = None
                entry_px = 0.0

                # TQQQ 조건
                if allow_l and (dir_cross == "LONG" and dir_gbdt == "LONG" and
                                conf_cross >= C_TH and conf_gbdt >= G_TH and
                                is_60m_bull and dip_ok_long):
                    chosen_symbol = config.BASE_ASSET_LONG
                    entry_px = cur_close_l

                # SQQQ 조건
                elif allow_s and (dir_cross == "SHORT" and dir_gbdt == "SHORT" and
                                  conf_cross >= C_TH and conf_gbdt >= G_TH and
                                  is_60m_bear and dip_ok_short):
                    chosen_symbol = config.BASE_ASSET_SHORT
                    entry_px = cur_close_s

                if chosen_symbol is None:
                    b_idx += 1
                    continue

                # 5분봉 궤적 추적 청산 시뮬레이션
                df_5m_target = day_long_5 if chosen_symbol == config.BASE_ASSET_LONG else day_short_5
                post_5m = df_5m_target[df_5m_target.index > cur_15m_time]
                if post_5m.empty:
                    b_idx += 1
                    continue

                tp_px = round(entry_px * (1 + TP_PCT), 2)
                sl_px = round(entry_px * (1 + SL_PCT), 2)

                exit_triggered = False
                exit_px = 0.0
                exit_reason = ""
                exit_5m_time = None
                bars_held_5m = 0

                eval_candles = post_5m.iloc[:MAX_HOLD_5M_BARS]

                for k_idx in range(len(eval_candles)):
                    c5 = eval_candles.iloc[k_idx]
                    t5_dt = eval_candles.index[k_idx]
                    t5_str = c5['time_str']
                    c5_o = float(c5['Open'])
                    c5_h = float(c5['High'])
                    c5_l = float(c5['Low'])
                    c5_c = float(c5['Close'])
                    bars_held_5m = k_idx + 1

                    hit_tp = (c5_h >= tp_px)
                    hit_sl = (c5_l <= sl_px)

                    if hit_tp and hit_sl:
                        if c5_c >= c5_o:
                            exit_triggered = True
                            exit_px = tp_px
                            exit_reason = "🎯 TP (+3.5% Take Profit)"
                            exit_5m_time = t5_dt
                            break
                        else:
                            exit_triggered = True
                            exit_px = sl_px
                            exit_reason = "🛑 SL (-2.0% Stop Loss)"
                            exit_5m_time = t5_dt
                            break
                    elif hit_tp:
                        exit_triggered = True
                        exit_px = tp_px
                        exit_reason = "🎯 TP (+3.5% Take Profit)"
                        exit_5m_time = t5_dt
                        break
                    elif hit_sl:
                        exit_triggered = True
                        exit_px = sl_px
                        exit_reason = "🛑 SL (-2.0% Stop Loss)"
                        exit_5m_time = t5_dt
                        break

                    # 15:45 장마감
                    if t5_str >= "15:45":
                        exit_triggered = True
                        exit_px = c5_c
                        exit_reason = "🌙 EOD Liquidation (15:45 NYT)"
                        exit_5m_time = t5_dt
                        break

                    # 90분 타임스탑
                    if bars_held_5m >= MAX_HOLD_5M_BARS:
                        exit_triggered = True
                        exit_px = c5_c
                        exit_reason = "⏰ 90m TimeStop Expired"
                        exit_5m_time = t5_dt
                        break

                if not exit_triggered and len(eval_candles) > 0:
                    last_c = eval_candles.iloc[-1]
                    exit_px = float(last_c['Close'])
                    exit_reason = "🌙 EOD Liquidation"
                    exit_5m_time = eval_candles.index[-1]

                gross_ret = (exit_px - entry_px) / entry_px
                net_ret = gross_ret - FRICTION_PCT  # 0.34% 마찰비용 차감
                pnl_krw = capital * net_ret
                capital += pnl_krw
                equity_curve.append(capital)

                trades.append({
                    "date": d_str,
                    "symbol": chosen_symbol,
                    "entry_time": cur_15m_time.strftime('%Y-%m-%d %H:%M'),
                    "exit_time": exit_5m_time.strftime('%Y-%m-%d %H:%M') if exit_5m_time else "N/A",
                    "entry_price": entry_px,
                    "exit_price": exit_px,
                    "gross_return_pct": round(gross_ret * 100, 2),
                    "net_return_pct": round(net_ret * 100, 2),
                    "pnl_krw": int(pnl_krw),
                    "capital_after": int(capital),
                    "exit_reason": exit_reason,
                    "bars_held_5m": bars_held_5m
                })

                if exit_5m_time:
                    remaining_15m = day_long_15[day_long_15.index > exit_5m_time]
                    if not remaining_15m.empty:
                        b_idx = day_long_15.index.get_loc(remaining_15m.index[0])
                    else:
                        break
                else:
                    b_idx += 1

        total_trades = len(trades)
        df_tr = pd.DataFrame(trades)
        net_wins = df_tr[df_tr['net_return_pct'] > 0]
        real_win_rate = len(net_wins) / total_trades * 100 if total_trades > 0 else 0.0
        total_net_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
        monthly_trades = total_trades / num_months

        pos_sum = df_tr[df_tr['pnl_krw'] > 0]['pnl_krw'].sum() if total_trades > 0 else 0
        neg_sum = abs(df_tr[df_tr['pnl_krw'] < 0]['pnl_krw'].sum()) if total_trades > 0 else 0
        pf = (pos_sum / neg_sum) if neg_sum > 0 else 99.99

        eq = np.array(equity_curve)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        mdd_pct = abs(dd.min()) * 100 if len(eq) > 0 else 0.0

        tp_count = df_tr['exit_reason'].str.contains("TP").sum() if total_trades > 0 else 0
        sl_count = df_tr['exit_reason'].str.contains("SL").sum() if total_trades > 0 else 0
        ts_count = df_tr['exit_reason'].str.contains("TimeStop").sum() if total_trades > 0 else 0
        eod_count = df_tr['exit_reason'].str.contains("EOD").sum() if total_trades > 0 else 0

        trade_logs[m["id"]] = trades
        all_results.append({
            "종목명": m["name"],
            "총 거래 횟수": f"{total_trades}회",
            "월평균 거래 횟수": f"{monthly_trades:.1f}회",
            "실전 승률": f"{real_win_rate:.2f}%",
            "순수익률(마찰비용 차감 후)": f"{total_net_return:+.2f}%",
            "MDD": f"{mdd_pct:.2f}%",
            "손익비(PF)": f"{pf:.2f}",
            "익절/손절": f"{tp_count}승 {sl_count}패",
            "최종 자본금": f"{int(capital):,}원"
        })

    # 4. 결과 출력
    print("\n" + "=" * 100)
    print("🏆 [Lumos 듀얼 챔피언: TQQQ 단독 vs SQQQ 단독 vs 통합 듀얼 최종 성적표]")
    print("=" * 100)
    
    df_cmp = pd.DataFrame(all_results)
    cols = ["종목명", "총 거래 횟수", "월평균 거래 횟수", "실전 승률", "순수익률(마찰비용 차감 후)", "MDD", "손익비(PF)", "최종 자본금"]
    
    print("\n[DataFrame 콘솔 출력]")
    print(df_cmp[cols].to_string(index=False))

    print("\n[마크다운 표 출력]")
    md_table = format_markdown_table(df_cmp, cols)
    print(md_table)

    # 개별 종목별 상세 거래 내역 요약
    print("\n" + "=" * 100)
    print("🔍 [종목별 매매 내역 전수 대조 요약]")
    print("=" * 100)
    for m_id, label in [("TQQQ_ONLY", "TQQQ 롱 단독"), ("SQQQ_ONLY", "SQQQ 숏 단독"), ("COMBINED", "통합 듀얼")]:
        t_list = trade_logs[m_id]
        print(f"\n📌 [{label}] 총 {len(t_list)}건 매매:")
        for idx, t in enumerate(t_list):
            print(f"  #{idx+1:02d} [{t['date']}] {t['symbol']} 진입: {t['entry_time']} -> 청산: {t['exit_time']} | {t['exit_reason']} | 순수익률: {t['net_return_pct']:+.2f}% | 자본금: {t['capital_after']:,}원")

    return df_cmp, trade_logs

if __name__ == "__main__":
    run_long_short_dual_backtest()
