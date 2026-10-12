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

def run_profit_protect_3pct_comparison():
    print("=" * 95)
    print("🔬 [Lumos 퀀트 시스템: Scenario C 기반 '+3.0% 익절 방어선 전진(Profit Protect)' 격리 백테스트]")
    print(f"⏰ 실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S KST')}")
    print("🔒 [보안 확인] 본 검증은 scripts/ 내 격리 스크립트로만 구동되며, core/ 운영 코드는 단 1줄도 수정되지 않습니다.")
    print("=" * 95)

    # 1. 다중 타임프레임 데이터 로드
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

    # 시뮬레이션 파라미터
    INITIAL_CAPITAL = 10_000_000.0
    FRICTION_PCT = 0.0034           # 0.34% 마찰비용
    TP_PCT = 0.035                  # +3.5%
    SL_PCT = -0.020                 # -2.0%
    MAX_HOLD_5M_BARS = 18           # 90분
    C_TH = 0.60
    G_TH = 0.60

    modes = [
        {"id": "Original", "name": "Scenario C (기존 챔피언: 표준 TBM)", "use_pp_3pct": False},
        {"id": "PP_3pct", "name": "Scenario C + 3.0% 방어선 전진(Protect)", "use_pp_3pct": True}
    ]

    all_results = []
    trade_logs = {}

    for m in modes:
        use_pp = m["use_pp_3pct"]
        capital = INITIAL_CAPITAL
        trades = []
        equity_curve = [capital]

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

                # Dual Consensus
                dir_cross = row_l['cross_dir']
                conf_cross = row_l['cross_conf']
                dir_gbdt = row_l['Direction']
                conf_gbdt = row_l['Confidence']

                chosen_symbol = None
                entry_px = 0.0

                if (dir_cross == "LONG" and dir_gbdt == "LONG" and
                    conf_cross >= C_TH and conf_gbdt >= G_TH and
                    is_60m_bull and dip_ok_long):
                    chosen_symbol = config.BASE_ASSET_LONG
                    entry_px = cur_close_l
                elif (dir_cross == "SHORT" and dir_gbdt == "SHORT" and
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

                tp_px = round(entry_px * (1 + TP_PCT), 2)         # +3.5%
                hard_sl_px = round(entry_px * (1 + SL_PCT), 2)    # -2.0%
                pp_3pct_px = round(entry_px * 1.030, 2)           # +3.0%

                exit_triggered = False
                exit_px = 0.0
                exit_reason = ""
                exit_5m_time = None
                bars_held_5m = 0

                pp_3pct_active = False
                max_high_reached = entry_px

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

                    if c5_h > max_high_reached:
                        max_high_reached = c5_h

                    # 1. TP (+3.5%) 검사: 언제나 최우선 목표 달성
                    hit_tp = (c5_h >= tp_px)

                    # 2. +3.0% 방어선 로직
                    if use_pp:
                        # 이미 이전에 +3.0%를 활성화한 상태인 경우
                        if pp_3pct_active:
                            # +3.5% 도달 시 TP
                            if hit_tp:
                                exit_triggered = True
                                exit_px = tp_px
                                exit_reason = "🎯 TP (+3.5% Take Profit)"
                                exit_5m_time = t5_dt
                                break
                            # +3.0% 하향 터치 시 +3.0% 익절 청산
                            elif c5_l <= pp_3pct_px:
                                exit_triggered = True
                                exit_px = pp_3pct_px
                                exit_reason = "🛡️ PROTECT_3.0% (+3.0% 익절 방어)"
                                exit_5m_time = t5_dt
                                break

                        # 이번 봉에서 처음으로 +3.0% 이상에 도달한 경우
                        elif c5_h >= pp_3pct_px:
                            # 만약 같은 봉에서 +3.5%까지 원샷 돌파했다면 TP
                            if hit_tp:
                                exit_triggered = True
                                exit_px = tp_px
                                exit_reason = "🎯 TP (+3.5% Take Profit)"
                                exit_5m_time = t5_dt
                                break
                            else:
                                # +3.0%는 넘었으나 +3.5% 미달
                                # 봉 마감 시점에 +3.0% 아래로 되밀려 마감했거나 저가가 +3.0% 밑으로 꺼진 경우
                                if c5_c < pp_3pct_px:
                                    exit_triggered = True
                                    exit_px = pp_3pct_px
                                    exit_reason = "🛡️ PROTECT_3.0% (+3.0% 도달후 당봉 반락)"
                                    exit_5m_time = t5_dt
                                    break
                                else:
                                    # +3.0% 위에서 안정적으로 마감 -> 다음 봉부터 +3.0% 트레일링 활성화
                                    pp_3pct_active = True

                    # 기본 TBM SL (-2.0%) 검사 (수익보존 미발동 상태일 때만)
                    if not exit_triggered:
                        current_sl_px = pp_3pct_px if (use_pp and pp_3pct_active) else hard_sl_px
                        hit_sl = (c5_l <= current_sl_px)

                        if hit_tp and hit_sl:
                            if c5_c >= c5_o:
                                exit_triggered = True
                                exit_px = tp_px
                                exit_reason = "🎯 TP (+3.5% Take Profit)"
                                exit_5m_time = t5_dt
                                break
                            else:
                                exit_triggered = True
                                exit_px = current_sl_px
                                exit_reason = "🛡️ PROTECT_3.0% (+3.0% 익절 방어)" if (use_pp and pp_3pct_active) else "🛑 SL (-2.0% Stop Loss)"
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
                            exit_px = current_sl_px
                            exit_reason = "🛡️ PROTECT_3.0% (+3.0% 익절 방어)" if (use_pp and pp_3pct_active) else "🛑 SL (-2.0% Stop Loss)"
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
                    "bars_held_5m": bars_held_5m,
                    "max_high_pct": round((max_high_reached - entry_px) / entry_px * 100, 2)
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

        pos_sum = df_tr[df_tr['pnl_krw'] > 0]['pnl_krw'].sum()
        neg_sum = abs(df_tr[df_tr['pnl_krw'] < 0]['pnl_krw'].sum())
        pf = (pos_sum / neg_sum) if neg_sum > 0 else 99.99

        eq = np.array(equity_curve)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / peak
        mdd_pct = abs(dd.min()) * 100

        tp_count = df_tr['exit_reason'].str.contains("TP").sum()
        sl_count = df_tr['exit_reason'].str.contains("SL").sum()
        pp_count = df_tr['exit_reason'].str.contains("PROTECT_3.0%").sum()
        ts_count = df_tr['exit_reason'].str.contains("TimeStop").sum()
        eod_count = df_tr['exit_reason'].str.contains("EOD").sum()

        trade_logs[m["id"]] = trades
        all_results.append({
            "시나리오명": m["name"],
            "총 거래 횟수": f"{total_trades}회",
            "월평균 거래 횟수": f"{monthly_trades:.1f}회",
            "실전 승률": f"{real_win_rate:.2f}%",
            "순수익률(마찰비용 차감 후)": f"{total_net_return:+.2f}%",
            "MDD": f"{mdd_pct:.2f}%",
            "손익비(PF)": f"{pf:.2f}",
            "익절(TP +3.5%)": f"{tp_count}회",
            "방어선익절(+3.0%)": f"{pp_count}회",
            "손절(SL -2.0%)": f"{sl_count}회",
            "타임스탑": f"{ts_count}회",
            "최종 자본금": f"{int(capital):,}원"
        })

    # 4. 결과 출력
    print("\n" + "=" * 105)
    print("🏆 [Scenario C 기존 베이스라인 vs +3.0% 익절 방어선 전진 1:1 비교]")
    print("=" * 105)
    df_cmp = pd.DataFrame(all_results)
    print(df_cmp.to_string(index=False))

    # 개별 거래 내역 대조 출력
    print("\n" + "=" * 105)
    print("🔍 [개별 거래별 청산 사유 및 장중 최고가 상세 대조]")
    print("=" * 105)
    orig_tr = trade_logs["Original"]
    pp_tr = trade_logs["PP_3pct"]

    for i in range(len(orig_tr)):
        o = orig_tr[i]
        p = pp_tr[i] if i < len(pp_tr) else None
        print(f"Trade #{i+1:02d} | 진입: {o['entry_time']} ({o['symbol']})")
        print(f"  • [기존 챔피언] 청산: {o['exit_time']} | 사유: {o['exit_reason']} | 순수익률: {o['net_return_pct']:+.2f}%")
        if p:
            print(f"  • [+3.0% 방어] 청산: {p['exit_time']} | 사유: {p['exit_reason']} | 순수익률: {p['net_return_pct']:+.2f}% (장중최고: +{p['max_high_pct']:.2f}%)")
            if o['exit_reason'] != p['exit_reason']:
                print(f"    ⚠️ >> 결과 변동: {o['exit_reason']} ➔ {p['exit_reason']} (수익률 차이: {p['net_return_pct'] - o['net_return_pct']:+.2f}%)")
        print("-" * 90)

    return df_cmp, trade_logs

if __name__ == "__main__":
    run_profit_protect_3pct_comparison()
