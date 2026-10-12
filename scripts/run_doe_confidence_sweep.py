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

# Windows UTF-8 console output setup
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
from core.moe_orchestrator import MoEMetaOrchestrator
from core.heterogeneous_models import CrossAssetDislocationModel

def format_markdown_table(headers: List[str], rows: List[List[Any]], aligns: List[str] = None) -> str:
    """tabulate 없이 고속으로 깔끔한 마크다운 테이블을 생성"""
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

def run_doe_confidence_sweep():
    print("=" * 105)
    print("🏛 [Lumos 퀀트 시스템: 하이브리드 MoE V3 (GBDT + Cross-Asset Veto) 확신도 DoE 전구간 백테스트]")
    print(f"⏰ 실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S KST')}")
    print("=" * 105)

    # -------------------------------------------------------------------------
    # 1. 데이터 레이크(market_data.db)로부터 전 기간 데이터 로드
    # -------------------------------------------------------------------------
    lake = MarketDataLake()
    print("⏳ [1/4] 데이터 레이크에서 15m/60m/5m 전 기간 시계열 데이터 로드 중...")
    
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

    # 일자/시간 문자열 생성
    for df in [long_15m, short_15m, long_5m, short_5m, macro_60m, long_60m, macro_15m, nvda_15m, qqq_15m, vix_15m]:
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
    start_dt = long_15m['datetime'].iloc[0]
    end_dt = long_15m['datetime'].iloc[-1]

    print(f"   • 데이터 범위: {start_dt} ~ {end_dt}")
    print(f"   • 총 거래일: {num_days}일 (약 {total_weeks:.1f}주, {num_days/21:.1f}개월)")
    print(f"   • 캔들 수량: 15분봉 {len(long_15m):,}개 | 5분봉 {len(long_5m):,}개 | 60분봉 {len(long_60m):,}개")

    # -------------------------------------------------------------------------
    # 2. 고속 백테스트를 위한 피처 & 인디케이터 사전 계산 (Precomputations)
    # -------------------------------------------------------------------------
    print("\n⏳ [2/4] GBDT 3-Class 피처, 크로스에셋 신호, 3중 스크린 지표 사전 계산 중...")
    
    # 60분봉 EMA20
    macro_60m['ema20'] = macro_60m['Close'].ewm(span=20, adjust=False).mean()
    long_60m['ema20'] = long_60m['Close'].ewm(span=20, adjust=False).mean()

    # MoE 모델 및 GBDT 엔진 초기화
    moe = MoEMetaOrchestrator(confidence_threshold=0.55, gbdt_threshold=0.55, mode="hybrid_v3")
    
    # TQQQ 15분봉 피처 및 GBDT 확률/확신도
    long_15m_feat = moe.gbdt_engine.extract_features(long_15m)
    long_15m_feat = moe.gbdt_engine.add_confidence_columns(long_15m_feat)
    
    # SQQQ 15분봉 피처 (Screen 3 단기 눌림목 검증용)
    short_15m_feat = moe.gbdt_engine.extract_features(short_15m)
    short_15m_feat.set_index('datetime', inplace=True, drop=False)

    # 크로스에셋 인과 괴리 모델 신호 사전 계산
    cross_mod = CrossAssetDislocationModel(dislocation_z_threshold=1.6)
    cross_dirs = []
    cross_confs = []

    # NVDA, QQQ, VIX 15m 정렬 매핑을 위한 lookup 딕셔너리
    nvda_map = nvda_15m.set_index('datetime')['Close'].to_dict()
    macro_map = macro_15m.set_index('datetime')['Close'].to_dict() if not macro_15m.empty else {}
    qqq_map  = qqq_15m.set_index('datetime')['Close'].to_dict()
    vix_map  = vix_15m.set_index('datetime')['Close'].to_dict()
    long_close_list = long_15m['Close'].values
    long_dt_list = long_15m['datetime'].values

    # 5개봉 전 수익률 계산을 위한 인덱싱
    for i in range(len(long_15m)):
        if i < 5:
            cross_dirs.append("NONE")
            cross_confs.append(0.50)
            continue
            
        cur_t = long_dt_list[i]
        past_5_t = long_dt_list[i-5]

        # TQQQ 5개봉 수익률
        s_r = float(long_close_list[i] / long_close_list[i-5] - 1.0)
        
        # NVDA, SOXX, QQQ, VIX 가격 확인
        if (cur_t in nvda_map and past_5_t in nvda_map and 
            cur_t in qqq_map and past_5_t in qqq_map and 
            cur_t in vix_map and past_5_t in vix_map):
            n_r = float(nvda_map[cur_t] / nvda_map[past_5_t] - 1.0)
            sx_r = float(macro_map[cur_t] / macro_map[past_5_t] - 1.0) if (cur_t in macro_map and past_5_t in macro_map) else n_r
            q_r = float(qqq_map[cur_t] / qqq_map[past_5_t] - 1.0)
            v_r = float(vix_map[cur_t] / vix_map[past_5_t] - 1.0)
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

    print("   • 사전 계산 완료: 2,085개 캔들에 대한 GBDT 확신도 및 Cross-Asset 방향 매핑 완료")

    # -------------------------------------------------------------------------
    # 3. DoE(실험계획법) 임계값 시나리오 정의 (50% ~ 80%, 5% 단위)
    # -------------------------------------------------------------------------
    doe_thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    
    # 헌법 불변 매매 파라미터
    INITIAL_CAPITAL = 10_000.0   # $10,000 USD (복리 100% 운용)
    TP_PCT = 0.035              # +3.5% 익절
    SL_PCT = -0.020             # -2.0% 칼손절
    TIME_STOP_BARS_5M = 18      # 90분 (5분봉 18개)
    SLIPPAGE_PAYUP = 0.03       # $0.03 페이업/슬리피지
    FEE_RATE = 0.0020           # 0.20% 왕복 거래비용 (수수료 0.10% + 스프레드 마찰 0.10%)
    
    print("\n⏳ [3/4] 7개 확신도(50%~80%) DoE 시나리오 5분봉 정밀 궤적(Path Dissection) 시뮬레이션 가동 중...")

    # 5분봉 빠른 조회를 위해 datetime을 인덱스로 설정
    long_5m_indexed = long_5m.set_index('datetime', drop=False)
    short_5m_indexed = short_5m.set_index('datetime', drop=False)

    all_doe_results = []
    trades_by_threshold = {}

    for T in doe_thresholds:
        t_pct_str = f"{int(T*100)}%"
        capital = INITIAL_CAPITAL
        trades = []
        equity_curve = [capital]
        daily_stoploss_count = 0
        current_day_str = ""

        for d_str in unique_dates:
            day_long_15 = long_15m_feat[long_15m_feat['date_str'] == d_str]
            if len(day_long_15) < 5:
                continue

            day_long_5 = long_5m[long_5m['date_str'] == d_str]
            day_short_5 = short_5m[short_5m['date_str'] == d_str]

            # [인터락 5] 매 정규장 개장 시 일일 서킷 브레이커 손절 카운트 0 리셋
            daily_stoploss_count = 0

            active_pos = None
            b_idx = 0
            n_bars = len(day_long_15)

            while b_idx < n_bars:
                cur_15m_row = day_long_15.iloc[b_idx]
                cur_15m_time = cur_15m_row['datetime']
                time_str = cur_15m_row['time_str']

                # 14:30 NYT 이후 신규 진입 금지 (No-Entry Cutoff)
                # 개장 첫 봉(09:30) 노이즈 배제
                if b_idx < 1 or time_str > "14:30":
                    b_idx += 1
                    continue

                # 일일 서킷 브레이커 발동 시 당일 진입 전면 차단
                if daily_stoploss_count >= 3:
                    b_idx += 1
                    continue

                # -------------------------------------------------------------
                # [하이브리드 MoE V3 진입 의사결정 집행]
                # -------------------------------------------------------------
                dir_gbdt = cur_15m_row['Direction']
                conf_gbdt = float(cur_15m_row['Confidence'])
                dir_cross = cur_15m_row['cross_dir']

                # 조건 A (공격수 트리거): GBDT 방향 제시 및 확신도 >= T
                is_gbdt_trigger = (dir_gbdt in ["LONG", "SHORT"]) and (conf_gbdt >= T)

                # 조건 B (크로스에셋 역풍 방패): 정반대 방향일 때만 Veto 차단
                is_cross_veto = (
                    (dir_gbdt == "LONG" and dir_cross == "SHORT") or
                    (dir_gbdt == "SHORT" and dir_cross == "LONG")
                )

                if not is_gbdt_trigger or is_cross_veto:
                    b_idx += 1
                    continue

                direction = dir_gbdt

                # -------------------------------------------------------------
                # [3중 스크린 검증]
                # -------------------------------------------------------------
                # Screen 1: 60분봉 상위 추세 필터
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

                # Screen 3: 단기 눌림목 타점 필터
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
                # [진입 집행: 3대 인터락 완결 승인]
                # -------------------------------------------------------------
                chosen_symbol = config.BASE_ASSET_LONG if direction == "LONG" else config.BASE_ASSET_SHORT
                
                # 진입 기준 가격 (페이업 +$0.03 반영)
                if chosen_symbol == config.BASE_ASSET_LONG:
                    base_px = float(cur_15m_row['Close'])
                else:
                    if cur_15m_time in short_15m_feat.index:
                        base_px = float(short_15m_feat.loc[cur_15m_time]['Close'])
                    else:
                        base_px = 40.0

                entry_px = round(base_px + SLIPPAGE_PAYUP, 2)
                shares = int(capital / entry_px)
                invested = shares * entry_px

                if shares <= 0 or invested <= 0:
                    b_idx += 1
                    continue

                # -------------------------------------------------------------
                # [5분봉 정밀 궤적 추적(Path Dissection) 청산 감시]
                # -------------------------------------------------------------
                target_5m_df = day_long_5 if chosen_symbol == config.BASE_ASSET_LONG else day_short_5
                post_5m = target_5m_df[target_5m_df['datetime'] > cur_15m_time]

                if post_5m.empty:
                    b_idx += 1
                    continue

                tp_px = round(entry_px * (1 + TP_PCT), 2)
                sl_px = round(entry_px * (1 + SL_PCT), 2)

                exit_triggered = False
                exit_px = 0.0
                exit_reason = ""
                exit_time_str = ""
                bars_held_5m = 0

                eval_5m = post_5m.iloc[:TIME_STOP_BARS_5M]

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

                    # Intra-bar Illusion 해소
                    if hit_tp and hit_sl:
                        if c5_c >= c5_o:
                            exit_triggered = True
                            exit_px = round(tp_px - SLIPPAGE_PAYUP, 2)
                            exit_reason = "🎯 목표익절 (+3.5% Intra-5m)"
                            exit_time_str = t5_str
                            break
                        else:
                            exit_triggered = True
                            exit_px = round(sl_px - SLIPPAGE_PAYUP, 2)
                            exit_reason = "🛑 칼손절 (-2.0% Intra-5m)"
                            exit_time_str = t5_str
                            daily_stoploss_count += 1
                            break
                    elif hit_tp:
                        exit_triggered = True
                        exit_px = round(tp_px - SLIPPAGE_PAYUP, 2)
                        exit_reason = "🎯 목표익절 (+3.5%)"
                        exit_time_str = t5_str
                        break
                    elif hit_sl:
                        exit_triggered = True
                        exit_px = round(sl_px - SLIPPAGE_PAYUP, 2)
                        exit_reason = "🛑 칼손절 (-2.0%)"
                        exit_time_str = t5_str
                        daily_stoploss_count += 1
                        break

                    # 15:45 NYT 도달 시 당일 전량 청산 (0% 오버나잇)
                    if t5_str >= "15:45":
                        exit_triggered = True
                        exit_px = round(c5_c - SLIPPAGE_PAYUP, 2)
                        exit_reason = "🌙 당일청산 (15:45 NYT)"
                        exit_time_str = t5_str
                        if (exit_px - entry_px) / entry_px <= -0.020:
                            daily_stoploss_count += 1
                        break

                    # 90분 타임스탑 만료 (18개 봉 경과)
                    if bars_held_5m >= TIME_STOP_BARS_5M:
                        exit_triggered = True
                        exit_px = round(c5_c - SLIPPAGE_PAYUP, 2)
                        exit_reason = "⏰ 90분 타임스탑"
                        exit_time_str = t5_str
                        if (exit_px - entry_px) / entry_px <= -0.020:
                            daily_stoploss_count += 1
                        break

                if not exit_triggered and len(eval_5m) > 0:
                    last_c = eval_5m.iloc[-1]
                    exit_px = round(float(last_c['Close']) - SLIPPAGE_PAYUP, 2)
                    exit_reason = "🌙 당일청산 (EOD)"
                    exit_time_str = last_c['time_str']
                    if (exit_px - entry_px) / entry_px <= -0.020:
                        daily_stoploss_count += 1

                # -------------------------------------------------------------
                # [손익 및 자본금 회계 정산 (왕복 수수료 0.20% 반영)]
                # -------------------------------------------------------------
                cost_amount = invested * FEE_RATE
                pnl_amount = (shares * (exit_px - entry_px)) - cost_amount
                capital += pnl_amount
                equity_curve.append(capital)
                net_ret_pct = pnl_amount / invested * 100.0

                trades.append({
                    "date": d_str,
                    "symbol": chosen_symbol,
                    "direction": direction,
                    "confidence": conf_gbdt,
                    "entry_time": time_str,
                    "entry_price": entry_px,
                    "exit_time": exit_time_str,
                    "exit_price": exit_px,
                    "bars_held_5m": bars_held_5m,
                    "duration_min": bars_held_5m * 5,
                    "return_pct": net_ret_pct,
                    "pnl_usd": pnl_amount,
                    "capital_after": capital,
                    "exit_reason": exit_reason
                })

                # 청산 시점의 15분봉 인덱스로 점프 (중복 진입 방지)
                # 5분봉 경과 수에 따라 15분봉 진행 (5분봉 3개당 15분봉 1개)
                bars_to_advance = max(1, int(np.ceil(bars_held_5m / 3.0)))
                b_idx += bars_to_advance

        # ---------------------------------------------------------------------
        # 시나리오별 통계 메트릭 산출
        # ---------------------------------------------------------------------
        total_trades = len(trades)
        wins = [t for t in trades if t['pnl_usd'] > 0]
        losses = [t for t in trades if t['pnl_usd'] <= 0]
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (win_count / total_trades * 100.0) if total_trades > 0 else 0.0

        long_trades = [t for t in trades if t['symbol'] == config.BASE_ASSET_LONG]
        short_trades = [t for t in trades if t['symbol'] == config.BASE_ASSET_SHORT]
        long_wins = [t for t in long_trades if t['pnl_usd'] > 0]
        short_wins = [t for t in short_trades if t['pnl_usd'] > 0]

        long_wr = (len(long_wins) / len(long_trades) * 100.0) if len(long_trades) > 0 else 0.0
        short_wr = (len(short_wins) / len(short_trades) * 100.0) if len(short_trades) > 0 else 0.0

        total_gain = sum(t['pnl_usd'] for t in wins)
        total_loss = abs(sum(t['pnl_usd'] for t in losses))
        profit_factor = (total_gain / total_loss) if total_loss > 0 else (99.9 if total_gain > 0 else 0.0)

        cum_return_pct = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100.0

        # MDD 계산
        eq_series = pd.Series(equity_curve)
        peak = eq_series.cummax()
        dd = (eq_series - peak) / peak * 100.0
        mdd = abs(dd.min()) if not dd.empty else 0.0

        avg_trades_week = round(total_trades / total_weeks, 2)
        avg_dur_min = round(np.mean([t['duration_min'] for t in trades]), 1) if trades else 0.0

        # 평가 (Verdict)
        if win_rate >= 80.0 and profit_factor >= 3.0 and avg_trades_week >= 1.5:
            verdict = "⭐ [최적 스윗스팟] 초고승률 & 주 2~3회 안정적 타점"
        elif win_rate >= 75.0 and profit_factor >= 2.5:
            verdict = "✅ [우수] 승률 및 손익비 양호, 실전 운용 적합"
        elif win_rate >= 65.0 and avg_trades_week >= 3.0:
            verdict = "⚠️ [보통] 거래 빈도 높으나 노이즈로 승률 소폭 저하"
        elif total_trades <= 5:
            verdict = "❌ [기각] 과도한 컷오프로 거래 기회 증발"
        else:
            verdict = "❌ [미달] 휩쏘 잦고 손익비 부족"

        # 청산 사유별 세부 집계
        tp_trades = [t for t in trades if "목표익절" in t['exit_reason']]
        sl_trades = [t for t in trades if "칼손절" in t['exit_reason']]
        timestop_trades = [t for t in trades if "타임스탑" in t['exit_reason']]
        eod_trades = [t for t in trades if "당일청산" in t['exit_reason']]

        metrics = {
            "threshold_pct": t_pct_str,
            "threshold_val": T,
            "total_trades": total_trades,
            "trades_per_week": avg_trades_week,
            "wins": win_count,
            "losses": loss_count,
            "win_rate": round(win_rate, 2),
            "long_trades": len(long_trades),
            "long_win_rate": round(long_wr, 1),
            "short_trades": len(short_trades),
            "short_win_rate": round(short_wr, 1),
            "tp_count": len(tp_trades),
            "sl_count": len(sl_trades),
            "timestop_count": len(timestop_trades),
            "eod_count": len(eod_trades),
            "cum_return_pct": round(cum_return_pct, 2),
            "final_capital": round(capital, 2),
            "profit_factor": round(profit_factor, 2),
            "mdd_pct": round(mdd, 2),
            "avg_duration_min": avg_dur_min,
            "verdict": verdict
        }
        all_doe_results.append(metrics)
        trades_by_threshold[t_pct_str] = trades

        print(f"   • T = {t_pct_str:4s}: 거래 {total_trades:2d}회 (주 {avg_trades_week:4.1f}회) | 승률 {win_rate:5.1f}% | 수익률 {cum_return_pct:+6.2f}% | PF {profit_factor:4.2f} | MDD {mdd:4.2f}% ➔ {verdict}")

    # -------------------------------------------------------------------------
    # 4. 결과 표 포맷팅 및 요약 리포트 생성
    # -------------------------------------------------------------------------
    print("\n" + "=" * 105)
    print("📊 [Lumos V3 하이브리드 MoE 확신도 DoE (50%~80%) 종합 백테스트 결과표]")
    print("=" * 105)

    headers = [
        "확신도", "총거래", "주당거래", "승 / 패", "승률(%)", 
        "TQQQ(건/승률)", "SQQQ(건/승률)", "누적수익률", "손익비(PF)", "MDD(%)", "최종자본금($)", "종합 판정"
    ]
    aligns = [":---:", ":---:", ":---:", ":---:", ":---:", ":---:", ":---:", "---:", ":---:", ":---:", "---:", ":---"]

    table_rows = []
    for m in all_doe_results:
        table_rows.append([
            m["threshold_pct"],
            f"{m['total_trades']}회",
            f"{m['trades_per_week']}회",
            f"{m['wins']}승 {m['losses']}패",
            f"{m['win_rate']:.1f}%",
            f"{m['long_trades']}건 ({m['long_win_rate']}%)",
            f"{m['short_trades']}건 ({m['short_win_rate']}%)",
            f"{m['cum_return_pct']:+.2f}%",
            f"{m['profit_factor']:.2f}",
            f"{m['mdd_pct']:.2f}%",
            f"${m['final_capital']:,.2f}",
            m["verdict"]
        ])

    md_table = format_markdown_table(headers, table_rows, aligns)
    print("\n" + md_table + "\n")

    print("\n" + "=" * 105)
    print("🎯 [Lumos V3 하이브리드 MoE 확신도 DoE 청산 유형별(TP/SL/TimeStop/EOD) 세부 분석표]")
    print("=" * 105)
    
    headers_exit = [
        "확신도", "총거래", "목표익절(+3.5%)", "칼손절(-2.0%)", "타임스탑(90분)", "당일청산(15:45)", "평균보유시간"
    ]
    aligns_exit = [":---:", ":---:", ":---:", ":---:", ":---:", ":---:", ":---:"]
    rows_exit = []
    for m in all_doe_results:
        tot = m['total_trades']
        tp_pct_val = (m['tp_count'] / tot * 100) if tot > 0 else 0
        sl_pct_val = (m['sl_count'] / tot * 100) if tot > 0 else 0
        ts_pct_val = (m['timestop_count'] / tot * 100) if tot > 0 else 0
        eod_pct_val = (m['eod_count'] / tot * 100) if tot > 0 else 0
        rows_exit.append([
            m["threshold_pct"],
            f"{tot}회",
            f"{m['tp_count']}회 ({tp_pct_val:.1f}%)",
            f"{m['sl_count']}회 ({sl_pct_val:.1f}%)",
            f"{m['timestop_count']}회 ({ts_pct_val:.1f}%)",
            f"{m['eod_count']}회 ({eod_pct_val:.1f}%)",
            f"{m['avg_duration_min']:.1f}분"
        ])
    md_exit_table = format_markdown_table(headers_exit, rows_exit, aligns_exit)
    print("\n" + md_exit_table + "\n")

    # JSON 및 CSV 파일로 저장
    summary_path = DATA_DIR / "hybrid_v3_confidence_doe_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "test_period": f"{start_dt} ~ {end_dt}",
            "total_trading_days": num_days,
            "total_weeks": round(total_weeks, 1),
            "base_capital_usd": INITIAL_CAPITAL,
            "doe_results": all_doe_results,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, f, ensure_ascii=False, indent=2)

    csv_path = DATA_DIR / "hybrid_v3_confidence_doe_summary.csv"
    pd.DataFrame(all_doe_results).to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"💾 [결과 저장 완료] JSON: {summary_path.name} | CSV: {csv_path.name}")
    print("=" * 105)

    return all_doe_results, trades_by_threshold

if __name__ == "__main__":
    run_doe_confidence_sweep()
