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
    """tabulate 패키지 의존성 없는 자체 고속 마크다운 표 생성기"""
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join([":---:" if "횟수" in c or "승률" in c else "---" for c in cols]) + " |"
    rows = []
    for _, row in df.iterrows():
        row_str = "| " + " | ".join([str(row[c]) for c in cols]) + " |"
        rows.append(row_str)
    return "\n".join([header, separator] + rows)

def run_grid_search_backtest():
    print("=" * 90)
    print("🏛 [Lumos 퀀트 시스템: 듀얼 합의 엔진(Dual Consensus) Grid Search 백테스트]")
    print(f"⏰ 실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S KST')}")
    print("=" * 90)

    # 1. 다중 타임프레임 데이터 로드 (최대 504 거래일 롤링 윈도우)
    print("⏳ [1/4] 데이터 레이크(market_data.db)로부터 다중 타임프레임(5m/15m/60m) 데이터 로드 중...")
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

    # 일자 및 시각 문자열 컬럼 생성
    for df in [long_15m, short_15m, long_5m, short_5m, macro_60m, long_60m, nvda_15m, qqq_15m, vix_15m]:
        df['date_str'] = df.index.strftime('%Y-%m-%d')
        df['time_str'] = df.index.strftime('%H:%M')

    unique_dates = sorted(long_15m['date_str'].unique())
    num_days = len(unique_dates)
    num_months = num_days / 21.0
    print(f"   • 총 거래일: {num_days}일 (약 {num_months:.2f}개월)")
    print(f"   • 데이터 기간: {unique_dates[0]} ~ {unique_dates[-1]}")
    print(f"   • 캔들 볼륨: 15분봉 {len(long_15m):,}개 | 5분봉 {len(long_5m):,}개 | 60분봉 {len(long_60m):,}개")

    # 2. 피처 추출 및 사전 계산 (Precomputations)
    print("\n⏳ [2/4] 기술적 지표, 60분봉 추세 및 머신러닝 피처 사전 계산 중...")
    # 60분봉 EMA20 추세 계산
    macro_60m['ema20'] = macro_60m['Close'].ewm(span=20, adjust=False).mean()
    long_60m['ema20'] = long_60m['Close'].ewm(span=20, adjust=False).mean()

    # MoE 오케스트레이터 및 기학습된 GBDT 챔피언 모델 로드
    moe = MoEMetaOrchestrator()
    long_15m_feat = moe.gbdt_engine.extract_features(long_15m)
    long_15m_feat = moe.gbdt_engine.add_confidence_columns(long_15m_feat)
    
    short_15m_feat = moe.gbdt_engine.extract_features(short_15m)
    short_15m_feat = moe.gbdt_engine.add_confidence_columns(short_15m_feat)

    # 크로스에셋 인과 괴리 모델 신호 계산
    cross_mod = CrossAssetDislocationModel(dislocation_z_threshold=1.6)
    cross_sigs = []
    cross_confs = []
    cross_dirs = []

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

    print("   • 사전 계산 완료: 15분봉 전체 캔들에 대한 GBDT/Cross 신호 매핑 완료")

    # 3. 4대 시나리오 정의
    scenarios = [
        {
            "id": "Scenario A",
            "name": "Scenario A (Baseline)",
            "cross_th": 0.60,
            "gbdt_th": 0.55,
            "desc": "Cross >= 0.60 AND GBDT >= 0.55"
        },
        {
            "id": "Scenario B",
            "name": "Scenario B (Macro-Heavy)",
            "cross_th": 0.65,
            "gbdt_th": 0.52,
            "desc": "Cross >= 0.65 AND GBDT >= 0.52"
        },
        {
            "id": "Scenario C",
            "name": "Scenario C (Micro-Heavy)",
            "cross_th": 0.55,
            "gbdt_th": 0.60,
            "desc": "Cross >= 0.55 AND GBDT >= 0.60"
        },
        {
            "id": "Scenario D",
            "name": "Scenario D (Extreme)",
            "cross_th": 0.65,
            "gbdt_th": 0.60,
            "desc": "Cross >= 0.65 AND GBDT >= 0.60"
        }
    ]

    INITIAL_CAPITAL = 10_000_000.0  # 1천만 원
    FRICTION_PCT = 0.0034           # 0.34% 왕복 마찰비용 강제 차감
    TP_PCT = 0.035                  # +3.5%
    SL_PCT = -0.020                 # -2.0%
    MAX_HOLD_5M_BARS = 18           # 90분 (5분봉 18개)

    print("\n⏳ [3/4] 4대 시나리오 5분봉 궤적 추적(Path Dissection) 시뮬레이션 가동 중...")

    all_scenario_metrics = []
    detailed_trade_logs = {}

    for sc in scenarios:
        sc_name = sc["name"]
        c_th = sc["cross_th"]
        g_th = sc["gbdt_th"]
        
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

                # 개장 직후 첫 봉 노이즈 배제 및 14:30 이후 신규 진입 금지
                if b_idx < 1 or time_str > "14:30":
                    b_idx += 1
                    continue

                row_l = day_long_15.iloc[b_idx]
                row_s = short_15m_feat.loc[cur_15m_time] if cur_15m_time in short_15m_feat.index else None

                # [Screen 1: 60분봉 추세]
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

                # [Screen 3: 5분봉 단기 눌림목]
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

                # [Dual Consensus AND 조건]
                dir_cross = row_l['cross_dir']
                conf_cross = row_l['cross_conf']
                
                dir_gbdt = row_l['Direction']
                conf_gbdt = row_l['Confidence']

                chosen_symbol = None
                entry_px = 0.0

                # TQQQ 롱 승인
                if (dir_cross == "LONG" and dir_gbdt == "LONG" and
                    conf_cross >= c_th and conf_gbdt >= g_th and
                    is_60m_bull and dip_ok_long):
                    chosen_symbol = config.BASE_ASSET_LONG
                    entry_px = cur_close_l

                # SQQQ 숏(하락장) 승인
                elif (dir_cross == "SHORT" and dir_gbdt == "SHORT" and
                      conf_cross >= c_th and conf_gbdt >= g_th and
                      is_60m_bear and dip_ok_short):
                    chosen_symbol = config.BASE_ASSET_SHORT
                    entry_px = cur_close_s

                # 진입 미충족 시 다음 15분봉으로
                if chosen_symbol is None:
                    b_idx += 1
                    continue

                # =========================================================================
                # [5분봉 정밀 궤적 추적(Path Dissection) 트리플 배리어 청산 실행기]
                # =========================================================================
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

                    # [Intra-bar Illusion 해결: 단일 5분봉 내 동시 충돌 시 캔들 파형 추적]
                    if hit_tp and hit_sl:
                        if c5_c >= c5_o:
                            exit_triggered = True
                            exit_px = tp_px
                            exit_reason = "🎯 TP (+3.5% Intra-5m Rescued)"
                            exit_5m_time = t5_dt
                            break
                        else:
                            exit_triggered = True
                            exit_px = sl_px
                            exit_reason = "🛑 SL (-2.0% Intra-5m Confirmed)"
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

                    # 장마감 가드: 15:45 도달 시 당일 전량 청산
                    if t5_str >= "15:45":
                        exit_triggered = True
                        exit_px = c5_c
                        exit_reason = "🌙 EOD Liquidation (15:45 NYT)"
                        exit_5m_time = t5_dt
                        break

                    # 90분 타임스탑: 18개 봉 경과 시 시장가 청산
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

                # =========================================================================
                # [수익률 및 마찰 비용(0.34%) 강제 차감 회계 처리]
                # =========================================================================
                gross_ret = (exit_px - entry_px) / entry_px
                net_ret = gross_ret - FRICTION_PCT  # 0.34% 마찰비용 강제 차감
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

        # =========================================================================
        # [성능 평가 지표 집계]
        # =========================================================================
        total_trades = len(trades)
        if total_trades > 0:
            df_tr = pd.DataFrame(trades)
            net_wins = df_tr[df_tr['net_return_pct'] > 0]
            gross_wins = df_tr[df_tr['gross_return_pct'] > 0]
            
            real_win_rate = len(net_wins) / total_trades * 100
            gross_win_rate = len(gross_wins) / total_trades * 100

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
            ts_count = df_tr['exit_reason'].str.contains("TimeStop").sum()
            eod_count = df_tr['exit_reason'].str.contains("EOD").sum()
        else:
            real_win_rate = 0.0
            gross_win_rate = 0.0
            total_net_return = 0.0
            monthly_trades = 0.0
            pf = 0.0
            mdd_pct = 0.0
            tp_count = sl_count = ts_count = eod_count = 0

        detailed_trade_logs[sc["id"]] = trades
        all_scenario_metrics.append({
            "시나리오명": sc_name,
            "Cross임계치": f"{c_th*100:.0f}%",
            "GBDT임계치": f"{g_th*100:.0f}%",
            "총 거래 횟수": f"{total_trades}회",
            "월평균 거래 횟수": f"{monthly_trades:.1f}회",
            "실전 승률": f"{real_win_rate:.2f}%",
            "순수익률(마찰비용 차감 후)": f"{total_net_return:+.2f}%",
            "MDD": f"{mdd_pct:.2f}%",
            "손익비(PF)": f"{pf:.2f}",
            "최종 자본금": f"{int(capital):,}원",
            "익절/손절/타임/EOD": f"{tp_count}/{sl_count}/{ts_count}/{eod_count}",
            "_raw_trades": total_trades,
            "_raw_ret": total_net_return,
            "_raw_wr": real_win_rate,
            "_raw_pf": pf,
            "_raw_mdd": mdd_pct
        })
        print(f"   ✅ [{sc['id']} 완료] 거래 {total_trades}회 | 순수익률 {total_net_return:+.2f}% | 승률 {real_win_rate:.2f}% | PF {pf:.2f} | MDD {mdd_pct:.2f}%")

    # 4. 결과 출력 및 비교 분석
    print("\n" + "=" * 105)
    print("🏆 [Lumos 듀얼 합의 엔진 임계치 조합 Grid Search 백테스트 최종 비교 분석표]")
    print("=" * 105)
    
    df_res = pd.DataFrame(all_scenario_metrics)
    display_cols = [
        "시나리오명", "총 거래 횟수", "월평균 거래 횟수", "실전 승률",
        "순수익률(마찰비용 차감 후)", "MDD", "손익비(PF)"
    ]
    
    print("\n[DataFrame 콘솔 출력]")
    print(df_res[display_cols].to_string(index=False))

    print("\n[마크다운 표 출력]")
    md_table = format_markdown_table(df_res, display_cols)
    print(md_table)

    output_csv = PROJECT_ROOT / "data" / "grid_search_dual_thresholds.csv"
    df_res.drop(columns=[c for c in df_res.columns if c.startswith("_")]).to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n📁 결과 파일 저장 완료: {output_csv}")

    # 개별 거래 내역 저장
    output_trades_json = PROJECT_ROOT / "data" / "grid_search_trade_logs.json"
    with open(output_trades_json, "w", encoding="utf-8") as f:
        json.dump(detailed_trade_logs, f, indent=2, ensure_ascii=False)
    print(f"📁 상세 거래 로그 저장 완료: {output_trades_json}")

    return all_scenario_metrics, detailed_trade_logs

if __name__ == "__main__":
    run_grid_search_backtest()
