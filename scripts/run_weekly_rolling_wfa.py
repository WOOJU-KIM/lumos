import os
import sys
import json
import sqlite3
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from config import DATA_DIR, BASE_DIR, MODELS_DIR
from core.data_lake import MarketDataLake
from core.ml_engine import MLFeatureEngine
from core.heterogeneous_models import CrossAssetDislocationModel

def sanitize_for_json(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(i) for i in obj]
    return obj

def run_weekly_rolling_walk_forward():
    lake = MarketDataLake()
    print("=" * 115)
    print("🚀 [Lumos 퀀트 시스템: 4개년(2022.09 ~ 2026.09) 주간 롤링 워크포워드 (Weekly WFA) 백테스트]")
    print("   • 롤링 학습 윈도우: 직전 104주 (2년 / 504거래일) 고정 윈도우")
    print("   • 테스트 윈도우: 다음 1주일 (완전 미학습 Out-of-Sample 미래 블라인드)")
    print("   • 주간 재학습 횟수: 총 210회 연속 롤링 재학습 & 시뮬레이션")
    print("   • 시작 원금: 10,000,000원 (천만 원)")
    print("=" * 115)

    # 1. 6개년 시계열 전량 로드
    print("⏳ [1/4] 데이터 레이크에서 6개년(38,000+개 분봉) 데이터 로드 중...")
    tqqq_15m = lake.load_candles("TQQQ", "15m")
    sqqq_15m = lake.load_candles("SQQQ", "15m")
    qqq_60m = lake.load_candles("QQQ", "60m")
    nvda_15m = lake.load_candles("NVDA", "15m")
    soxx_15m = lake.load_candles("SOXX", "15m")
    qqq_15m  = lake.load_candles("QQQ", "15m")
    vixy_15m = lake.load_candles("VIXY", "15m")
    ief_15m  = lake.load_candles("IEF", "15m")

    # 60m QQQ EMA
    qqq_60m['ema_filter'] = qqq_60m['Close'].ewm(span=config.QQQ_EMA_PERIOD, adjust=False).mean()

    # 2. 피처 및 타겟 추출 (전체 시계열 사전 계산)
    print("⏳ [2/4] 머신러닝 피처 및 Triple Barrier 타겟 사전 추출 중...")
    ml_engine = MLFeatureEngine(confidence_threshold=0.60)
    tqqq_feat = ml_engine.extract_features(tqqq_15m)
    labels = ml_engine.compute_triple_barrier_labels(tqqq_feat)

    # 타겟 라벨 매핑: 1 -> 2 (Long TP), -1 -> 0 (Short TP), 0 -> 1 (Neutral)
    target_series = labels.map({1: 2, -1: 0, 0: 1}).fillna(1).astype(int)
    tqqq_feat['target'] = target_series

    # ISO 주차(Year-Week) 태깅
    tqqq_feat['datetime_dt'] = pd.to_datetime(tqqq_feat['datetime'])
    tqqq_feat['date_str'] = tqqq_feat['datetime_dt'].dt.strftime('%Y-%m-%d')
    tqqq_feat['time_str'] = tqqq_feat['datetime_dt'].dt.strftime('%H:%M')
    tqqq_feat['week_id'] = tqqq_feat['datetime_dt'].dt.isocalendar().year.astype(str) + '-' + tqqq_feat['datetime_dt'].dt.isocalendar().week.astype(str).str.zfill(2)

    # 크로스에셋 딕셔너리 고속 캐싱
    nvda_map = nvda_15m.set_index('datetime')['Close'].to_dict()
    soxx_map = soxx_15m.set_index('datetime')['Close'].to_dict()
    qqq_map  = qqq_15m.set_index('datetime')['Close'].to_dict()
    vix_map  = vixy_15m.set_index('datetime')['Close'].to_dict()
    ief_map  = ief_15m.set_index('datetime')['Close'].to_dict()
    sqqq_dict = sqqq_15m.set_index('datetime').to_dict(orient='index')

    cross_mod = CrossAssetDislocationModel(dislocation_z_threshold=1.6)

    # 피처 컬럼 선별
    exclude_cols = {'open', 'high', 'low', 'close', 'volume', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'datetime', 'datetime_dt', 'date_str', 'time_str', 'week_id', 'target', 'date'}
    feature_cols = [c for c in tqqq_feat.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(tqqq_feat[c])]
    print(f"📊 학습 피처 수: {len(feature_cols)}개 피처")

    # 주차 목록 정렬
    unique_weeks = sorted(tqqq_feat['week_id'].unique())
    print(f"📅 총 가용 주차: {len(unique_weeks)}주 ({unique_weeks[0]} ~ {unique_weeks[-1]})")

    ROLLING_WINDOW_WEEKS = 104  # 2년 (104주)
    if len(unique_weeks) <= ROLLING_WINDOW_WEEKS:
        raise RuntimeError("데이터 주차가 104주보다 적습니다.")

    test_start_idx = ROLLING_WINDOW_WEEKS  # 104번째 주부터 첫 테스트 시작
    total_test_weeks = len(unique_weeks) - test_start_idx
    print(f"🎯 주간 롤링 테스트 주차: 총 {total_test_weeks}주 ({unique_weeks[test_start_idx]} ~ {unique_weeks[-1]})\n")

    # 3. 주간 롤링 워크포워드 시뮬레이션
    print("⏳ [3/4] 210회 주간 롤링 재학습 & 블라인드 실전 백테스트 실행 중...")
    t0 = time.time()

    initial_capital = 10_000_000.0  # 천만 원
    current_capital = initial_capital
    peak_capital = initial_capital
    max_drawdown_pct = 0.0

    trades = []
    trade_id = 0
    active_pos = None
    slippage_rate = 0.0020  # 왕복 0.05%

    week_progress = 0

    for w_idx in range(test_start_idx, len(unique_weeks)):
        cur_test_week = unique_weeks[w_idx]
        train_weeks = unique_weeks[w_idx - ROLLING_WINDOW_WEEKS : w_idx]
        
        # A. 직전 104주(2년) 학습 데이터 슬라이스
        train_mask = tqqq_feat['week_id'].isin(train_weeks)
        X_train = tqqq_feat.loc[train_mask, feature_cols].fillna(0.0)
        y_train = tqqq_feat.loc[train_mask, 'target']

        # B. 경량 고속 LightGBM 학습 (1회당 ~0.08초)
        clf = LGBMClassifier(
            objective='multiclass',
            num_class=3,
            class_weight='balanced',
            n_estimators=80,
            max_depth=4,
            learning_rate=0.03,
            random_state=42,
            verbosity=-1,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)

        # C. 이번 1주일(완전 미학습 미래 OOS) 추론
        test_mask = tqqq_feat['week_id'] == cur_test_week
        test_df = tqqq_feat[test_mask].copy()
        if test_df.empty:
            continue

        X_test = test_df[feature_cols].fillna(0.0)
        probs = clf.predict_proba(X_test)

        prob_short = probs[:, 0]
        prob_neutral = probs[:, 1]
        prob_long = probs[:, 2]

        confidences = np.full(len(test_df), 0.50, dtype=float)
        directions = ["NONE"] * len(test_df)

        for i in range(len(test_df)):
            ps, pn, pl = prob_short[i], prob_neutral[i], prob_long[i]
            if pl > pn and pl > ps:
                calib_conf = min(0.95, max(0.50, 0.50 + (pl - 0.333) * 1.15))
                confidences[i] = calib_conf
                directions[i] = "LONG_TQQQ"
            elif ps > pn and ps > pl:
                calib_conf = min(0.95, max(0.50, 0.50 + (ps - 0.333) * 1.15))
                confidences[i] = calib_conf
                directions[i] = "SHORT_SQQQ"

        test_df['Confidence'] = confidences
        test_df['Direction'] = directions

        # D. 이번 1주일 시뮬레이션 집행
        test_dates = sorted(test_df['date_str'].unique())

        for d_str in test_dates:
            day_bars = test_df[test_df['date_str'] == d_str]

            for idx, row in day_bars.iterrows():
                curr_dt = row['datetime']
                time_str = row['time_str']
                cur_tqqq_close = float(row['Close'])
                cur_tqqq_high = float(row['High'])
                cur_tqqq_low = float(row['Low'])

                sqqq_row = sqqq_dict.get(curr_dt)
                cur_sqqq_close = float(sqqq_row['Close']) if sqqq_row else 0.0
                cur_sqqq_high = float(sqqq_row['High']) if sqqq_row else 0.0
                cur_sqqq_low = float(sqqq_row['Low']) if sqqq_row else 0.0

                # 1. 청산 검사 (TP +3%, SL -2%, 90분 타임스탑, 15:45 EOD)
                if active_pos is not None:
                    active_pos['bars'] += 1
                    sym = active_pos['sym']
                    entry_px = active_pos['entry_px']
                    cur_close = cur_tqqq_close if sym == 'TQQQ' else cur_sqqq_close
                    cur_high = cur_tqqq_high if sym == 'TQQQ' else cur_sqqq_high
                    cur_low = cur_tqqq_low if sym == 'TQQQ' else cur_sqqq_low

                    exit_price = None
                    exit_reason = None

                    if 'peak_high' not in active_pos:
                        active_pos['peak_high'] = cur_high
                    else:
                        active_pos['peak_high'] = max(active_pos['peak_high'], cur_high)

                    tp_px = entry_px * (1.0 + config.MAX_TP_PCT)
                    sl_px = entry_px * (1.0 - config.SL_MIN_PCT)

                    current_sl_px = sl_px
                    trailing_trigger_px = entry_px * (1.0 + config.TRAILING_TRIGGER_PCT)
                    if active_pos['peak_high'] >= trailing_trigger_px:
                        safety_sl_px = entry_px * (1.0 + max(0.0, config.TRAILING_TRIGGER_PCT - 0.015))
                        current_sl_px = max(safety_sl_px, active_pos['peak_high'] * (1.0 - config.TRAILING_DROP_PCT))

                    if cur_high >= tp_px:
                        exit_price = tp_px
                        exit_reason = "MAX_TP"
                    elif cur_low <= current_sl_px:
                        exit_price = current_sl_px
                        exit_reason = "TRAILING_SL" if current_sl_px > sl_px else "STOP_LOSS"
                    elif active_pos['bars'] * 15 >= config.TIME_STOP_MINUTES:
                        exit_price = cur_close
                        exit_reason = "TIME_STOP"
                    elif time_str >= config.PHASE_EOD_CLEAR:
                        exit_price = cur_close
                        exit_reason = "EOD_CLEAR"

                    if exit_price is not None:
                        raw_ret = (exit_price / entry_px) - 1.0
                        net_ret = raw_ret - slippage_rate
                        pnl_krw = current_capital * net_ret
                        current_capital += pnl_krw

                        if current_capital > peak_capital:
                            peak_capital = current_capital
                        dd = (peak_capital - current_capital) / peak_capital * 100.0
                        if dd > max_drawdown_pct:
                            max_drawdown_pct = dd

                        trade_id += 1
                        trades.append({
                            'trade_id': trade_id,
                            'datetime': curr_dt,
                            'week_id': cur_test_week,
                            'symbol': sym,
                            'entry_px': entry_px,
                            'exit_px': exit_price,
                            'entry_dt': active_pos['entry_dt'],
                            'exit_dt': curr_dt,
                            'bars_held': active_pos['bars'],
                            'holding_min': active_pos['bars'] * 15,
                            'raw_ret_pct': round(raw_ret * 100, 2),
                            'net_ret_pct': round(net_ret * 100, 2),
                            'pnl_krw': int(round(pnl_krw)),
                            'ending_capital': int(round(current_capital)),
                            'exit_reason': exit_reason,
                            'is_win': 1 if net_ret > 0 else 0
                        })
                        active_pos = None
                        continue

                # 2. 신규 진입 검토
                if active_pos is None and time_str < config.PHASE_MAIN_END:
                    past_qqq_60 = qqq_60m[qqq_60m['datetime'] <= curr_dt]
                    qqq_60m_bull = True
                    qqq_60m_bear = True
                    if len(past_qqq_60) >= 20:
                        q_c = past_qqq_60['Close'].iloc[-1]
                        q_ema = past_qqq_60['ema_filter'].iloc[-1]
                        qqq_60m_bull = (q_c >= q_ema * 0.998)
                        qqq_60m_bear = (q_c <= q_ema * 1.002)

                    dir_gbdt = row.get('Direction', 'NONE')
                    conf_gbdt = float(row.get('Confidence', 0.50))

                    n_px = nvda_map.get(curr_dt)
                    sx_px = soxx_map.get(curr_dt)
                    q_px = qqq_map.get(curr_dt)
                    v_px = vix_map.get(curr_dt)
                    i_px = ief_map.get(curr_dt)

                    cross_dir = "HOLD"
                    cur_bar_idx = day_bars.index.get_loc(idx)
                    if cur_bar_idx >= 5 and n_px and sx_px and q_px:
                        prev_dt = day_bars.iloc[cur_bar_idx - 5]['datetime']
                        p_n = nvda_map.get(prev_dt, n_px)
                        p_sx = soxx_map.get(prev_dt, sx_px)
                        p_q = qqq_map.get(prev_dt, q_px)
                        p_v = vix_map.get(prev_dt, v_px) if v_px else None
                        p_i = ief_map.get(prev_dt, i_px) if i_px else None

                        nvda_r = (n_px / p_n - 1.0) if p_n else 0.0
                        soxx_r = (sx_px / p_sx - 1.0) if p_sx else 0.0
                        qqq_r  = (q_px / p_q - 1.0) if p_q else 0.0
                        vix_r  = (v_px / p_v - 1.0) if (v_px and p_v) else 0.0
                        ief_r  = (i_px / p_i - 1.0) if (i_px and p_i) else 0.0
                        tnx_proxy_ret = -ief_r

                        tqqq_r = (cur_tqqq_close / day_bars.iloc[cur_bar_idx - 5]['Close']) - 1.0
                        sig_code, _, _ = cross_mod.predict_signal(
                            tqqq_ret=tqqq_r,
                            nvda_ret=nvda_r,
                            soxx_ret=soxx_r,
                            qqq_ret=qqq_r,
                            vix_ret=vix_r,
                            tnx_ret=tnx_proxy_ret
                        )
                        if sig_code > 0:
                            cross_dir = "LONG_TQQQ"
                        elif sig_code < 0:
                            cross_dir = "SHORT_SQQQ"

                    if dir_gbdt == "LONG_TQQQ" and conf_gbdt >= config.GBDT_CONFIDENCE_THRESHOLD and qqq_60m_bull and cross_dir != "SHORT_SQQQ":
                        active_pos = {
                            'sym': 'TQQQ',
                            'entry_px': cur_tqqq_close,
                            'entry_dt': curr_dt,
                            'bars': 0,
                            'peak_high': cur_tqqq_close
                        }
                    elif dir_gbdt == "SHORT_SQQQ" and conf_gbdt >= config.GBDT_CONFIDENCE_THRESHOLD and qqq_60m_bear and cross_dir != "LONG_TQQQ":
                        if cur_sqqq_close > 0:
                            active_pos = {
                                'sym': 'SQQQ',
                                'entry_px': cur_sqqq_close,
                                'entry_dt': curr_dt,
                                'bars': 0,
                                'peak_high': cur_sqqq_close
                            }

        week_progress += 1
        if week_progress % 30 == 0 or week_progress == total_test_weeks:
            print(f"   • [{week_progress}/{total_test_weeks}주차 완료] 현재 주: {cur_test_week} | 누적 거래: {len(trades):,}회 | 현재 잔고: {int(current_capital):,}원")

    elapsed = round(time.time() - t0, 1)
    print(f"\n🏁 [주간 롤링 시뮬레이션 완료 ({elapsed}초 소요)] 총 매매: {len(trades):,}회 | 최종 잔고: {int(current_capital):,}원 | MDD: {max_drawdown_pct:.2f}%\n")

    # 4. 결과 집계 및 표 출력
    print("⏳ [4/4] 결과 테이블 생성 및 집계 중...")
    df_trades = pd.DataFrame(trades)
    df_trades['datetime'] = pd.to_datetime(df_trades['datetime'])
    df_trades['year'] = df_trades['datetime'].dt.year
    df_trades['year_month'] = df_trades['datetime'].dt.strftime('%Y-%m')

    def calc_stats(sub_df, start_cap):
        if sub_df.empty:
            return {
                'trades': 0, 'win_rate': 0.0, 'pf': 0.0, 'pnl': 0, 'ret_pct': 0.0, 'end_cap': int(start_cap),
                'tqqq_trades': 0, 'sqqq_trades': 0
            }
        wins = sub_df[sub_df['net_ret_pct'] > 0]
        losses = sub_df[sub_df['net_ret_pct'] <= 0]
        win_rate = len(wins) / len(sub_df) * 100
        gross_profit = wins['pnl_krw'].sum()
        gross_loss = abs(losses['pnl_krw'].sum())
        pf = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 999.0
        tot_pnl = sub_df['pnl_krw'].sum()
        end_cap = sub_df['ending_capital'].iloc[-1]
        ret_pct = (end_cap / start_cap - 1.0) * 100
        return {
            'trades': len(sub_df),
            'win_rate': round(win_rate, 1),
            'pf': pf,
            'pnl': int(tot_pnl),
            'ret_pct': round(ret_pct, 1),
            'end_cap': int(end_cap),
            'tqqq_trades': len(sub_df[sub_df['symbol'] == 'TQQQ']),
            'sqqq_trades': len(sub_df[sub_df['symbol'] == 'SQQQ'])
        }

    # 전체 통산
    total_stats = calc_stats(df_trades, initial_capital)

    # 년도별 집계
    years = sorted(df_trades['year'].unique())
    yearly_rows = []
    y_start_cap = initial_capital
    for y in years:
        sub_y = df_trades[df_trades['year'] == y]
        st = calc_stats(sub_y, y_start_cap)
        yearly_rows.append({
            'year': int(y),
            'start_cap': int(y_start_cap),
            'end_cap': int(st['end_cap']),
            'pnl': int(st['pnl']),
            'ret_pct': round((st['end_cap'] / y_start_cap - 1.0) * 100, 1),
            'trades': int(st['trades']),
            'win_rate': float(st['win_rate']),
            'pf': float(st['pf']),
            'tqqq_cnt': int(st['tqqq_trades']),
            'sqqq_cnt': int(st['sqqq_trades'])
        })
        y_start_cap = st['end_cap']

    # 월별 집계
    months = sorted(df_trades['year_month'].unique())
    monthly_rows = []
    m_start_cap = initial_capital
    for m in months:
        sub_m = df_trades[df_trades['year_month'] == m]
        st = calc_stats(sub_m, m_start_cap)
        monthly_rows.append({
            'month': str(m),
            'start_cap': int(m_start_cap),
            'end_cap': int(st['end_cap']),
            'pnl': int(st['pnl']),
            'ret_pct': round((st['end_cap'] / m_start_cap - 1.0) * 100, 2),
            'trades': int(st['trades']),
            'win_rate': float(st['win_rate']),
            'pf': float(st['pf']),
            'tqqq_cnt': int(st['tqqq_trades']),
            'sqqq_cnt': int(st['sqqq_trades'])
        })
        m_start_cap = st['end_cap']

    result_summary = {
        'initial_capital': float(initial_capital),
        'final_capital': int(df_trades['ending_capital'].iloc[-1]),
        'total_trades': int(len(df_trades)),
        'total_weeks_tested': total_test_weeks,
        'rolling_window_weeks': ROLLING_WINDOW_WEEKS,
        'mdd_pct': round(max_drawdown_pct, 2),
        'total_stats': total_stats,
        'yearly': yearly_rows,
        'monthly': monthly_rows
    }

    # 콘솔 출력
    print("=" * 115)
    print("📊 [1. 주간 롤링 워크포워드 (210주간 매주 재학습) 4개년 종합 성적표]")
    print("=" * 115)
    print(f"• 테스트 기간: {unique_weeks[test_start_idx]} ~ {unique_weeks[-1]} (2022.09 ~ 2026.09, 총 {total_test_weeks}주)")
    print(f"• 롤링 재학습 윈도우: 매주 직전 104주 (2개년, 504거래일)")
    print(f"• 시작 원금: {initial_capital:,.0f}원 ➔ 최종 기말 잔고: {total_stats['end_cap']:,}원")
    print(f"• 총 누적 수익률: {total_stats['ret_pct']:+,.1f}% (총 손익: {total_stats['pnl']:+,}원)")
    print(f"• 총 거래 횟수: {total_stats['trades']}회 (TQQQ: {total_stats['tqqq_trades']}회 / SQQQ: {total_stats['sqqq_trades']}회)")
    print(f"• 주간 롤링 승률 (Win Rate): {total_stats['win_rate']:.1f}%")
    print(f"• 손익비 (Profit Factor): {total_stats['pf']:.2f}")
    print(f"• 4개년 최대 낙폭 (MDD): {max_drawdown_pct:.2f}%")
    print("=" * 115)

    print("\n" + "=" * 115)
    print("📅 [2. 년도별 상세 결산 (천만 원 시작 복리 운용 잔고 추이)]")
    print("=" * 115)
    print(f"{'년도':<6} | {'시작 잔고':<16} | {'기말 잔고':<16} | {'연간 손익':<16} | {'수익률':<9} | {'거래수':<6} | {'승률':<7} | {'PF':<6} | {'TQQQ/SQQQ':<10}")
    print("-" * 115)
    for r in yearly_rows:
        print(f"{r['year']:<6} | {r['start_cap']:>14,}원 | {r['end_cap']:>14,}원 | {r['pnl']:>+14,}원 | {r['ret_pct']:>+7.1f}% | {r['trades']:>4}회 | {r['win_rate']:>5.1f}% | {r['pf']:>5.2f} | {r['tqqq_cnt']:>2}/{r['sqqq_cnt']:<2}회")
    print("=" * 115)

    print("\n" + "=" * 115)
    print("🗓️ [3. 월별 상세 결산 (천만 원 시작 복리 운용 잔고, 49개월)]")
    print("=" * 115)
    print(f"{'연월':<7} | {'시작 잔고':<16} | {'기말 잔고':<16} | {'월간 손익':<16} | {'수익률':<9} | {'거래수':<6} | {'승률':<7} | {'PF':<6} | {'TQQQ/SQQQ':<10}")
    print("-" * 115)
    for r in monthly_rows:
        print(f"{r['month']:<7} | {r['start_cap']:>14,}원 | {r['end_cap']:>14,}원 | {r['pnl']:>+14,}원 | {r['ret_pct']:>+7.2f}% | {r['trades']:>4}회 | {r['win_rate']:>5.1f}% | {r['pf']:>5.2f} | {r['tqqq_cnt']:>2}/{r['sqqq_cnt']:<2}회")
    print("=" * 115)

    # 저장
    clean_summary = sanitize_for_json(result_summary)
    output_path = DATA_DIR / "backtest_weekly_rolling_wfa_summary.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clean_summary, f, ensure_ascii=False, indent=2)

    df_trades.to_csv(DATA_DIR / "backtest_weekly_rolling_wfa_trades.csv", index=False, encoding="utf-8-sig")
    print(f"\n💾 [결과 파일 저장 완료] {output_path}")

    return result_summary

if __name__ == "__main__":
    run_weekly_rolling_walk_forward()
