import os
import sys
import json
import sqlite3
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import config
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMClassifier

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_DIR, BASE_DIR
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

def run_5m_precision_backtest(model_train_mode: str = "22_24"):
    """
    5분봉 초정밀 궤적(5m Chronological Path Tracking) 백테스트
    - model_train_mode:
        - "22_24": 2022.09 ~ 2024.09 2년 학습 모델 (사용자 요청 1번 모델 비교용)
        - "rolling_wfa": 주간 롤링 워크포워드 모델 (매주 직전 104주 재학습)
    """
    lake = MarketDataLake()
    print("=" * 115)
    print(f"🚀 [Lumos 퀀트 시스템: 5분봉 정밀 궤적(5m Chronological Path) 백테스트]")
    print(f"   • 모델 학습 모드: {model_train_mode}")
    print(f"   • 검증 해상도: 5분봉(5m) 캔들 단위 초정밀 청산 추적 (90분 = 5분봉 18개 봉)")
    print(f"   • 시작 원금: 10,000,000원 (천만 원) 복리 운용")
    print("=" * 115)

    # 1. 15m 및 5m 데이터 로드
    print("⏳ [1/4] 데이터 레이크에서 15m 및 5m 캔들 로드 중...")
    long_15m = lake.load_candles(config.BASE_ASSET_LONG, "15m")
    short_15m = lake.load_candles(config.BASE_ASSET_SHORT, "15m")
    macro_60m = lake.load_candles(config.MACRO_TREND_SYMBOL, "60m")
    nvda_15m = lake.load_candles("NVDA", "15m")
    macro_15m = lake.load_candles(config.MACRO_TREND_SYMBOL, "15m")
    qqq_15m  = lake.load_candles("QQQ", "15m")
    vixy_15m = lake.load_candles("VIXY", "15m")
    ief_15m  = lake.load_candles("IEF", "15m")

    # 5분봉 로드
    long_5m = lake.load_candles(config.BASE_ASSET_LONG, "5m")
    short_5m = lake.load_candles(config.BASE_ASSET_SHORT, "5m")

    print(f"   📊 15분봉: TQQQ {len(long_15m):,}개 | SQQQ {len(short_15m):,}개")
    print(f"   📊 5분봉:  TQQQ {len(long_5m):,}개 | SQQQ {len(short_5m):,}개")

    # 60m 20EMA
    macro_60m['ema20'] = macro_60m['Close'].ewm(span=20, adjust=False).mean()

    # 5분봉 인덱스 맵 (고속 슬라이싱용)
    long_5m_by_date = {}
    for d, g in long_5m.groupby(long_5m['datetime'].str.slice(0, 10)):
        long_5m_by_date[d] = g.sort_values('datetime').reset_index(drop=True)

    short_5m_by_date = {}
    for d, g in short_5m.groupby(short_5m['datetime'].str.slice(0, 10)):
        short_5m_by_date[d] = g.sort_values('datetime').reset_index(drop=True)

    # 2. 피처 추출 및 모델 준비
    print("⏳ [2/4] 머신러닝 피처 및 모델 준비 중...")
    ml_engine = MLFeatureEngine(confidence_threshold=0.60)
    long_feat = ml_engine.extract_features(long_15m)
    labels = ml_engine.compute_triple_barrier_labels(long_feat)
    target_series = labels.map({1: 2, -1: 0, 0: 1}).fillna(1).astype(int)
    long_feat['target'] = target_series

    long_feat['datetime_dt'] = pd.to_datetime(long_feat['datetime'])
    long_feat['date_str'] = long_feat['datetime_dt'].dt.strftime('%Y-%m-%d')
    long_feat['time_str'] = long_feat['datetime_dt'].dt.strftime('%H:%M')

    # 무차원화 정규화 피처 추가 (주가 스케일 변화 방어)
    long_feat['ATR_Pct'] = (long_feat['ATR_14'] / (long_feat['Close'] + 1e-9)) * 100.0
    long_feat['MACD_Pct'] = (long_feat['MACD'] / (long_feat['Close'] + 1e-9)) * 100.0
    long_feat['MACD_Signal_Pct'] = (long_feat['MACD_Signal'] / (long_feat['Close'] + 1e-9)) * 100.0
    long_feat['MACD_Hist_Pct'] = (long_feat['MACD_Hist'] / (long_feat['Close'] + 1e-9)) * 100.0

    raw_price_features = {
        'open', 'high', 'low', 'close', 'volume', 'Open', 'High', 'Low', 'Close', 'Volume',
        'datetime', 'datetime_dt', 'date_str', 'time_str', 'week_id', 'target', 'date', 'year',
        'cum_vp', 'vwap', 'EMA_9', 'EMA_21', 'EMA_50', 'EMA_200',
        'BB_Upper', 'BB_Lower', 'KC_Upper', 'KC_Lower', 'ATR_14', 'MACD', 'MACD_Signal', 'MACD_Hist'
    }
    feature_cols = [c for c in long_feat.columns if c not in raw_price_features and pd.api.types.is_numeric_dtype(long_feat[c])]

    # 크로스에셋 딕셔너리
    nvda_map = nvda_15m.set_index('datetime')['Close'].to_dict()
    macro_map = macro_15m.set_index('datetime')['Close'].to_dict()
    qqq_map  = qqq_15m.set_index('datetime')['Close'].to_dict()
    vix_map  = vixy_15m.set_index('datetime')['Close'].to_dict()
    ief_map  = ief_15m.set_index('datetime')['Close'].to_dict()
    short_dict = short_15m.set_index('datetime').to_dict(orient='index')

    cross_mod = CrossAssetDislocationModel(dislocation_z_threshold=1.6)

    long_feat['week_id'] = long_feat['datetime_dt'].dt.isocalendar().year.astype(str) + '-' + long_feat['datetime_dt'].dt.isocalendar().week.astype(str).str.zfill(2)
    unique_weeks = sorted(long_feat['week_id'].unique())
    ROLLING_WINDOW_WEEKS = 104

    if model_train_mode == "22_24":
        # 22~24 모델 사전 학습 (2022-09-15 ~ 2024-09-14)
        train_mask_22_24 = (long_feat['date_str'] >= '2022-09-15') & (long_feat['date_str'] <= '2024-09-14')
        clf_22_24 = LGBMClassifier(
            objective='multiclass',
            num_class=3,
            class_weight='balanced',
            n_estimators=100,
            max_depth=5,
            learning_rate=0.03,
            random_state=42,
            verbosity=-1,
            n_jobs=-1
        )
        clf_22_24.fit(long_feat.loc[train_mask_22_24, feature_cols].fillna(0.0), long_feat.loc[train_mask_22_24, 'target'])
        print("   🧠 [2022~2024 고정 모델 학습 완료]")

        test_mask = long_feat['date_str'] >= '2022-09-15'
        test_df = long_feat[test_mask].copy()

        X_all = test_df[feature_cols].fillna(0.0)
        probs = clf_22_24.predict_proba(X_all)
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
                directions[i] = "LONG"
            elif ps > pn and ps > pl:
                calib_conf = min(0.95, max(0.50, 0.50 + (ps - 0.333) * 1.15))
                confidences[i] = calib_conf
                directions[i] = "SHORT"

        test_df['Confidence'] = confidences
        test_df['Direction'] = directions
    else:
        # 주간 롤링 WFA 추론 (210주간 매주 재학습 & 다음 1주 추론)
        print(f"   🔄 [주간 롤링 WFA 모델 재학습 시작: 총 {len(unique_weeks) - ROLLING_WINDOW_WEEKS}주]")
        test_dfs = []
        for w_idx in range(ROLLING_WINDOW_WEEKS, len(unique_weeks)):
            cur_test_week = unique_weeks[w_idx]
            train_weeks = unique_weeks[w_idx - ROLLING_WINDOW_WEEKS : w_idx]
            train_mask = long_feat['week_id'].isin(train_weeks)
            X_tr = long_feat.loc[train_mask, feature_cols].fillna(0.0)
            y_tr = long_feat.loc[train_mask, 'target']

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
            clf.fit(X_tr, y_tr)

            cur_w_mask = long_feat['week_id'] == cur_test_week
            w_test_df = long_feat[cur_w_mask].copy()
            if w_test_df.empty:
                continue

            X_te = w_test_df[feature_cols].fillna(0.0)
            w_probs = clf.predict_proba(X_te)
            p_s = w_probs[:, 0]
            p_n = w_probs[:, 1]
            p_l = w_probs[:, 2]

            w_confs = np.full(len(w_test_df), 0.50, dtype=float)
            w_dirs = ["NONE"] * len(w_test_df)
            for i in range(len(w_test_df)):
                ps, pn, pl = p_s[i], p_n[i], p_l[i]
                if pl > pn and pl > ps:
                    w_confs[i] = min(0.95, max(0.50, 0.50 + (pl - 0.333) * 1.15))
                    w_dirs[i] = "LONG"
                elif ps > pn and ps > pl:
                    w_confs[i] = min(0.95, max(0.50, 0.50 + (ps - 0.333) * 1.15))
                    w_dirs[i] = "SHORT"

            w_test_df['Confidence'] = w_confs
            w_test_df['Direction'] = w_dirs
            test_dfs.append(w_test_df)

        test_df = pd.concat(test_dfs, ignore_index=True)
        print("   ✅ [주간 롤링 WFA 추론 완료]")

    # 3. 5분봉 단위 정밀 궤적 시뮬레이션
    print("⏳ [3/4] 4개년 5분봉 정밀 궤적 시뮬레이션 실행 중...")
    initial_capital = 10_000_000.0  # 천만 원
    current_capital = initial_capital
    peak_capital = initial_capital
    max_drawdown_pct = 0.0

    trades = []
    trade_id = 0
    active_pos = None
    slippage_rate = 0.0020  # 왕복 0.05%

    unique_dates = sorted(test_df['date_str'].unique())

    for d_str in unique_dates:
        day_bars_15 = test_df[test_df['date_str'] == d_str].reset_index(drop=True)
        day_long_5 = long_5m_by_date.get(d_str, pd.DataFrame())
        day_short_5 = short_5m_by_date.get(d_str, pd.DataFrame())

        daily_stoploss_count = 0
        b_idx = 0
        n_bars = len(day_bars_15)

        while b_idx < n_bars:
            row = day_bars_15.iloc[b_idx]
            curr_dt = row['datetime']
            time_str = row['time_str']
            cur_long_close = float(row['Close'])

            short_row = short_dict.get(curr_dt)
            cur_short_close = float(short_row['Close']) if short_row else 0.0

            # Rule 5: 일일 서킷 브레이커 (3-Out 시 당일 추가 진입 전면 차단)
            if daily_stoploss_count >= 3:
                break

            # 14:30 이후 신규 진입 차단
            if time_str >= '14:30':
                b_idx += 1
                continue

            # Screen 1: 60분봉 20EMA 대추세
            past_macro_60 = macro_60m[macro_60m['datetime'] <= curr_dt]
            macro_60m_bull = True
            macro_60m_bear = True
            if len(past_macro_60) >= 20:
                macro_c = past_macro_60['Close'].iloc[-1]
                macro_ema20 = past_macro_60['ema20'].iloc[-1]
                macro_60m_bull = (macro_c >= macro_ema20 * 0.998)
                macro_60m_bear = (macro_c <= macro_ema20 * 1.002)

            dir_gbdt = row.get('Direction', 'NONE')
            conf_gbdt = float(row.get('Confidence', 0.50))

            # 크로스에셋 괴리율
            n_px = nvda_map.get(curr_dt)
            sx_px = macro_map.get(curr_dt)
            q_px = qqq_map.get(curr_dt)
            v_px = vix_map.get(curr_dt)
            i_px = ief_map.get(curr_dt)

            cross_dir = "HOLD"
            if b_idx >= 5 and n_px and sx_px and q_px:
                prev_dt = day_bars_15.iloc[b_idx - 5]['datetime']
                p_n = nvda_map.get(prev_dt, n_px)
                p_sx = macro_map.get(prev_dt, sx_px)
                p_q = qqq_map.get(prev_dt, q_px)
                p_v = vix_map.get(prev_dt, v_px) if v_px else None
                p_i = ief_map.get(prev_dt, i_px) if i_px else None

                nvda_r = (n_px / p_n - 1.0) if p_n else 0.0
                macro_r = (sx_px / p_sx - 1.0) if p_sx else 0.0
                qqq_r  = (q_px / p_q - 1.0) if p_q else 0.0
                vix_r  = (v_px / p_v - 1.0) if (v_px and p_v) else 0.0
                ief_r  = (i_px / p_i - 1.0) if (i_px and p_i) else 0.0
                tnx_proxy_ret = -ief_r

                long_r = (cur_long_close / day_bars_15.iloc[b_idx - 5]['Close']) - 1.0
                sig_code, _, _ = cross_mod.predict_signal(
                    long_ret=long_r,
                    nvda_ret=nvda_r,
                    macro_ret=macro_r,
                    qqq_ret=qqq_r,
                    vix_ret=vix_r,
                    tnx_ret=tnx_proxy_ret
                )
                if sig_code > 0:
                    cross_dir = "LONG"
                elif sig_code < 0:
                    cross_dir = "SHORT"

            # 진입 조건 검사
            entry_approved = False
            target_sym = None
            entry_price = 0.0

            if dir_gbdt == "LONG" and conf_gbdt >= 0.60 and macro_60m_bull and cross_dir != "SHORT":
                entry_approved = True
                target_sym = config.BASE_ASSET_LONG
                entry_price = cur_long_close
            elif dir_gbdt == "SHORT" and conf_gbdt >= 0.60 and macro_60m_bear and cross_dir != "LONG":
                entry_approved = True
                target_sym = config.BASE_ASSET_SHORT
                entry_price = cur_short_close

            if entry_approved and entry_price > 0:
                # 5분봉 정밀 궤적 추적 시작 (post 5m candles)
                target_5m = day_long_5 if target_sym == config.BASE_ASSET_LONG else day_short_5
                post_5m = target_5m[target_5m['datetime'] > curr_dt].reset_index(drop=True)

                tp_px = entry_price * 1.030
                sl_px = entry_price * 0.980

                exit_price = None
                exit_reason = None
                exit_dt = None
                holding_5m_bars = 0

                # 최대 18개 5분봉(90분) 동안 캔들 단위 정밀 추적
                max_eval_bars = min(len(post_5m), 18)

                for k in range(max_eval_bars):
                    bar_5m = post_5m.iloc[k]
                    b_h = float(bar_5m['High'] if 'High' in bar_5m else bar_5m['high'])
                    b_l = float(bar_5m['Low'] if 'Low' in bar_5m else bar_5m['low'])
                    b_c = float(bar_5m['Close'] if 'Close' in bar_5m else bar_5m['close'])
                    b_o = float(bar_5m['Open'] if 'Open' in bar_5m else bar_5m['open'])
                    b_dt = bar_5m['datetime']
                    b_time = b_dt[11:16]

                    # 5분봉 내부에서 익절과 손절 중 무엇이 먼저 터졌는지 검증
                    hit_tp = (b_h >= tp_px)
                    hit_sl = (b_l <= sl_px)

                    if hit_tp and hit_sl:
                        # 5분봉 단일 봉 내 동시 도달 시: 시가(Open)와의 거리로 순서 판단
                        dist_to_tp = abs(tp_px - b_o)
                        dist_to_sl = abs(sl_px - b_o)
                        if dist_to_tp < dist_to_sl:
                            exit_price = tp_px
                            exit_reason = "TAKE_PROFIT_3PCT (5m High)"
                        else:
                            exit_price = sl_px
                            exit_reason = "STOP_LOSS_2PCT (5m Low)"
                        exit_dt = b_dt
                        holding_5m_bars = k + 1
                        break
                    elif hit_tp:
                        exit_price = tp_px
                        exit_reason = "TAKE_PROFIT_3PCT"
                        exit_dt = b_dt
                        holding_5m_bars = k + 1
                        break
                    elif hit_sl:
                        exit_price = sl_px
                        exit_reason = "STOP_LOSS_2PCT"
                        exit_dt = b_dt
                        holding_5m_bars = k + 1
                        break

                    # 장 마감 청산 (15:45 이후)
                    if b_time >= '15:45':
                        exit_price = b_c
                        exit_reason = "EOD_MARKET_CLOSE (5m)"
                        exit_dt = b_dt
                        holding_5m_bars = k + 1
                        break

                    # 90분 타임스탑 도달 (18번째 5분봉 종가)
                    if k == 17:
                        exit_price = b_c
                        exit_reason = "TIME_STOP_90MIN (5m 18th bar)"
                        exit_dt = b_dt
                        holding_5m_bars = 18
                        break

                # 만약 당일 5분봉이 부족하여 미청산된 경우 (장 마감 도달)
                if exit_price is None:
                    if not post_5m.empty:
                        last_bar = post_5m.iloc[-1]
                        exit_price = float(last_bar['Close'] if 'Close' in last_bar else last_bar['close'])
                        exit_reason = "EOD_MARKET_CLOSE (5m End)"
                        exit_dt = post_5m.iloc[-1]['datetime']
                        holding_5m_bars = len(post_5m)
                    else:
                        exit_price = entry_price
                        exit_reason = "EOD_MARKET_CLOSE"
                        exit_dt = curr_dt
                        holding_5m_bars = 1

                # 청산 집행 및 잔고 업데이트
                raw_ret = (exit_price / entry_price) - 1.0
                net_ret = raw_ret - slippage_rate
                pnl_krw = current_capital * net_ret
                current_capital += pnl_krw

                if net_ret <= -0.015:
                    daily_stoploss_count += 1

                if current_capital > peak_capital:
                    peak_capital = current_capital
                dd = (peak_capital - current_capital) / peak_capital * 100.0
                if dd > max_drawdown_pct:
                    max_drawdown_pct = dd

                trade_id += 1
                trades.append({
                    'trade_id': trade_id,
                    'datetime': curr_dt,
                    'symbol': target_sym,
                    'entry_px': entry_price,
                    'exit_px': exit_price,
                    'entry_dt': curr_dt,
                    'exit_dt': exit_dt,
                    'holding_5m_bars': holding_5m_bars,
                    'holding_min': holding_5m_bars * 5,
                    'raw_ret_pct': round(raw_ret * 100, 2),
                    'net_ret_pct': round(net_ret * 100, 2),
                    'pnl_krw': int(round(pnl_krw)),
                    'ending_capital': int(round(current_capital)),
                    'exit_reason': exit_reason,
                    'is_win': 1 if net_ret > 0 else 0
                })

                # 단일 포지션 릴레이: 포지션 청산될 때까지 15분봉 스캐너 동결
                consumed_15m = int(np.ceil(holding_5m_bars / 3.0))
                b_idx += max(1, consumed_15m)
            else:
                b_idx += 1

    print(f"\n🏁 [5분봉 정밀 궤적 백테스트 완료] 총 매매: {len(trades):,}회 | 최종 잔고: {int(current_capital):,}원 | MDD: {max_drawdown_pct:.2f}%")

    # 4. 결과 테이블 집계 (인샘플 vs 아웃오브샘플, 년도별, 월별)
    df_t = pd.DataFrame(trades)
    df_t['year'] = df_t['entry_dt'].str.slice(0, 4)
    df_t['month'] = df_t['entry_dt'].str.slice(0, 7)

    # In-sample (2022-09 ~ 2024-09) vs Out-of-sample (2024-09 ~ 2026-09)
    in_sample_mask = (df_t['entry_dt'] >= '2022-09-15') & (df_t['entry_dt'] <= '2024-09-14 23:59:59')
    oos_mask = (df_t['entry_dt'] > '2024-09-14 23:59:59')

    in_trades = df_t[in_sample_mask]
    oos_trades = df_t[oos_mask]

    print("\n" + "=" * 115)
    print("📊 [1. 5분봉 정밀 궤적: 인샘플 vs 아웃오브샘플 분할 성적표]")
    print("=" * 115)
    if not in_trades.empty:
        in_w = in_trades[in_trades['net_ret_pct'] > 0]
        in_l = in_trades[in_trades['net_ret_pct'] <= 0]
        in_wr = len(in_w) / len(in_trades) * 100
        in_pf = in_w['net_ret_pct'].sum() / abs(in_l['net_ret_pct'].sum()) if abs(in_l['net_ret_pct'].sum()) > 0 else 99.0
        in_cap = in_trades['ending_capital'].iloc[-1]
        print(f"• 인샘플 구간 (2022.09 ~ 2024.09, 2년치 학습 데이터 안에서 시험):")
        print(f"   거래: {len(in_trades)}회 | 승률: {in_wr:.1f}% | PF: {in_pf:.2f} | 구간 잔고: {in_cap:,}원 (수익률: +{(in_cap/10000000-1)*100:.1f}%)")

    if not oos_trades.empty:
        oos_w = oos_trades[oos_trades['net_ret_pct'] > 0]
        oos_l = oos_trades[oos_trades['net_ret_pct'] <= 0]
        oos_wr = len(oos_w) / len(oos_trades) * 100
        oos_pf = oos_w['net_ret_pct'].sum() / abs(oos_l['net_ret_pct'].sum()) if abs(oos_l['net_ret_pct'].sum()) > 0 else 99.0
        oos_start_cap = in_trades['ending_capital'].iloc[-1] if not in_trades.empty else 10000000
        oos_end_cap = oos_trades['ending_capital'].iloc[-1]
        print(f"• 아웃오브샘플 구간 (2024.09 ~ 2026.09, 미학습 순수 미래 블라인드):")
        print(f"   거래: {len(oos_trades)}회 | 승률: {oos_wr:.1f}% | PF: {oos_pf:.2f} | 구간 잔고: {oos_end_cap:,}원 (미래 2개년 순수익률: +{(oos_end_cap/oos_start_cap-1)*100:.1f}%)")

    # 년도별 테이블
    print("\n" + "=" * 115)
    print("📅 [2. 년도별 상세 결산 (천만 원 시작 5분봉 정밀 복리 잔고 추이)]")
    print("=" * 115)
    print(f"{'년도':<6} | {'시작 잔고':>18} | {'기말 잔고':>18} | {'연간 손익':>18} | {'수익률':>10} | {'거래수':>6} | {'승률':>8} | {'PF':>6} | {'TQQQ/SQQQ'}")
    print("-" * 115)

    yearly_summary = []
    y_start_cap = initial_capital
    for y, g in df_t.groupby('year'):
        y_end_cap = float(g['ending_capital'].iloc[-1])
        y_pnl = y_end_cap - y_start_cap
        y_ret = (y_end_cap / y_start_cap - 1.0) * 100.0
        y_trades = len(g)
        w_cnt = len(g[g['net_ret_pct'] > 0])
        l_cnt = len(g[g['net_ret_pct'] <= 0])
        y_wr = (w_cnt / y_trades * 100.0) if y_trades > 0 else 0.0
        gross_w = g[g['net_ret_pct'] > 0]['net_ret_pct'].sum()
        gross_l = abs(g[g['net_ret_pct'] <= 0]['net_ret_pct'].sum())
        y_pf = (gross_w / gross_l) if gross_l > 0 else 99.0
        long_c = len(g[g['symbol'] == config.BASE_ASSET_LONG])
        short_c = len(g[g['symbol'] == config.BASE_ASSET_SHORT])

        print(f"{y:<6} | {int(y_start_cap):>16,}원 | {int(y_end_cap):>16,}원 | {int(y_pnl):>+16,}원 | {y_ret:>+9.1f}% | {y_trades:>5}회 | {y_wr:>7.1f}% | {y_pf:>6.2f} | {long_c:>3}/{short_c:<3}회")
        yearly_summary.append({
            "year": y, "start_cap": int(y_start_cap), "end_cap": int(y_end_cap),
            "pnl": int(y_pnl), "ret_pct": round(y_ret, 1), "trades": y_trades,
            "win_rate": round(y_wr, 1), "pf": round(y_pf, 2), "long_cnt": long_c, "short_cnt": short_c
        })
        y_start_cap = y_end_cap

    # 월별 테이블
    print("\n" + "=" * 115)
    print("🗓️ [3. 월별 상세 결산 (천만 원 시작 5분봉 정밀 복리 잔고 추이, 49개월)]")
    print("=" * 115)
    print(f"{'연월':<7} | {'시작 잔고':>18} | {'기말 잔고':>18} | {'월간 손익':>18} | {'수익률':>10} | {'거래수':>6} | {'승률':>8} | {'PF':>6} | {'TQQQ/SQQQ'}")
    print("-" * 115)

    monthly_summary = []
    m_start_cap = initial_capital
    for m, g in df_t.groupby('month'):
        m_end_cap = float(g['ending_capital'].iloc[-1])
        m_pnl = m_end_cap - m_start_cap
        m_ret = (m_end_cap / m_start_cap - 1.0) * 100.0
        m_trades = len(g)
        w_cnt = len(g[g['net_ret_pct'] > 0])
        l_cnt = len(g[g['net_ret_pct'] <= 0])
        m_wr = (w_cnt / m_trades * 100.0) if m_trades > 0 else 0.0
        gross_w = g[g['net_ret_pct'] > 0]['net_ret_pct'].sum()
        gross_l = abs(g[g['net_ret_pct'] <= 0]['net_ret_pct'].sum())
        m_pf = (gross_w / gross_l) if gross_l > 0 else 99.0
        long_c = len(g[g['symbol'] == config.BASE_ASSET_LONG])
        short_c = len(g[g['symbol'] == config.BASE_ASSET_SHORT])

        print(f"{m:<7} | {int(m_start_cap):>16,}원 | {int(m_end_cap):>16,}원 | {int(m_pnl):>+16,}원 | {m_ret:>+9.2f}% | {m_trades:>5}회 | {m_wr:>7.1f}% | {m_pf:>6.2f} | {long_c:>3}/{short_c:<3}회")
        monthly_summary.append({
            "month": m, "start_cap": int(m_start_cap), "end_cap": int(m_end_cap),
            "pnl": int(m_pnl), "ret_pct": round(m_ret, 2), "trades": m_trades,
            "win_rate": round(m_wr, 1), "pf": round(m_pf, 2), "long_cnt": long_c, "short_cnt": short_c
        })
        m_start_cap = m_end_cap

    # JSON 저장
    summary_data = {
        "model_train_mode": model_train_mode,
        "initial_capital": initial_capital,
        "final_capital": int(current_capital),
        "total_trades": len(trades),
        "mdd_pct": round(max_drawdown_pct, 2),
        "yearly": yearly_summary,
        "monthly": monthly_summary
    }

    out_file = PROJECT_ROOT / "data" / f"backtest_5m_precision_{model_train_mode}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(summary_data), f, indent=2, ensure_ascii=False)
    print(f"\n💾 [5분봉 정밀 백테스트 결과 저장 완료] {out_file}")

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else "rolling_wfa"
    run_5m_precision_backtest(mode)
