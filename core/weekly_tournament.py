from config import GBDT_CONFIDENCE_THRESHOLD
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import TimeSeriesSplit

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Windows 콘솔 utf-8 설정
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from config import BASE_DIR, MODELS_DIR, DATA_DIR, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from core.data_lake import MarketDataLake
from core.model_registry import ModelRegistry
from core.ml_engine import MLFeatureEngine
from core.moe_orchestrator import MoEMetaOrchestrator, MOE_MODEL_PATH
from core.heterogeneous_models import CrossAssetDislocationModel
from core.system_logger import system_logger
from agents.dispatcher_agent import DispatcherAgent

class WeeklyTournament:
    """
    [주간 단위 3자 토너먼트 거버넌스 엔진 (Weekly 3-Way Tournament)]
    1. 실행 주기: 매주 토요일 06:00 KST (미국 금요장 마감 후 주말 결산)
    2. 3대 후보 맞대결 (실제 거래 모델 동일 하이브리드 MoE 전구간 백테스트):
       - [후보 1: 기존 챔피언] 이번 주 실전을 치른 현재 MoE 챔피언 모델
       - [후보 2: 데이터 최신화] 이번 주 5일치 최신 데이터를 추가하여 504 거래일 롤링 재학습한 MoE 모델
       - [후보 3: 신규 튜닝 모델] 단일 LightGBM 하이퍼파라미터 최적화(TimeSeriesSplit CV) MoE 모델
    3. 종합 스코어 산출 및 1위 챔피언 자동 승격 및 실전 파일(model_champion.pkl) 즉각 교체 적용
    4. 실전 성과 부진(승률 < 50%) 시 골든 베이스라인으로 자동 안전 롤백
    5. 텔레그램으로 성적표 및 공식 결과 카드 즉시 발송
    """
    def __init__(
        self,
        data_lake: Optional[MarketDataLake] = None,
        registry: Optional[ModelRegistry] = None,
        confidence_threshold: float = GBDT_CONFIDENCE_THRESHOLD
    ):
        self.data_lake = data_lake or MarketDataLake()
        self.registry = registry or ModelRegistry()
        self.confidence_threshold = confidence_threshold
        self.dispatcher = DispatcherAgent(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

    def _prepare_market_data_cache(self) -> Dict[str, Any]:
        """고속 전구간 MoE 백테스트를 위한 멀티 타임프레임 및 피처 사전 캐싱"""
        print("⏳ [1/5] 데이터 레이크에서 15m/5m/60m 및 거시 지표 데이터 로드 중...")
        tqqq_15m = self.data_lake.load_candles("TQQQ", "15m")
        sqqq_15m = self.data_lake.load_candles("SQQQ", "15m")
        tqqq_5m  = self.data_lake.load_candles("TQQQ", "5m")
        sqqq_5m  = self.data_lake.load_candles("SQQQ", "5m")
        soxx_60m = self.data_lake.load_candles("SOXX", "60m")
        tqqq_60m = self.data_lake.load_candles("TQQQ", "60m")
        soxx_15m = self.data_lake.load_candles("SOXX", "15m")
        nvda_15m = self.data_lake.load_candles("NVDA", "15m")
        qqq_15m  = self.data_lake.load_candles("QQQ", "15m")
        vix_15m  = self.data_lake.load_candles("^VIX", "15m")

        for df in [tqqq_15m, sqqq_15m, tqqq_5m, sqqq_5m, soxx_60m, tqqq_60m, soxx_15m, nvda_15m, qqq_15m, vix_15m]:
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

        # 60m EMA20 사전 계산
        if not soxx_60m.empty:
            soxx_60m['ema20'] = soxx_60m['Close'].ewm(span=20, adjust=False).mean()
        if not tqqq_60m.empty:
            tqqq_60m['ema20'] = tqqq_60m['Close'].ewm(span=20, adjust=False).mean()

        # 15m 공통 피처 추출
        ml_fe = MLFeatureEngine(confidence_threshold=self.confidence_threshold)
        tqqq_15m_feat = ml_fe.extract_features(tqqq_15m)
        sqqq_15m_feat = ml_fe.extract_features(sqqq_15m)
        sqqq_15m_feat.set_index('datetime', inplace=True, drop=False)

        # 크로스에셋 거시 신호 사전 계산 (NVDA, SOXX, QQQ, VIX 5봉 수익률 기반)
        cross_mod = CrossAssetDislocationModel(dislocation_z_threshold=1.6)
        nvda_map = nvda_15m.set_index('datetime')['Close'].to_dict() if not nvda_15m.empty else {}
        soxx_map = soxx_15m.set_index('datetime')['Close'].to_dict() if not soxx_15m.empty else {}
        qqq_map = qqq_15m.set_index('datetime')['Close'].to_dict() if not qqq_15m.empty else {}
        vix_map = vix_15m.set_index('datetime')['Close'].to_dict() if not vix_15m.empty else {}

        cross_dirs = []
        for idx in range(len(tqqq_15m_feat)):
            row = tqqq_15m_feat.iloc[idx]
            dt = row['datetime']
            tqqq_ret = float(row.get('ROC_5', 0.0)) / 100.0 if 'ROC_5' in row else 0.0

            n_px = nvda_map.get(dt)
            sx_px = soxx_map.get(dt)
            q_px = qqq_map.get(dt)
            v_px = vix_map.get(dt)

            if idx >= 5 and n_px and q_px and v_px:
                prev_dt = tqqq_15m_feat.iloc[idx - 5]['datetime']
                prev_n = nvda_map.get(prev_dt, n_px)
                prev_sx = soxx_map.get(prev_dt, sx_px) if sx_px else n_px
                prev_q = qqq_map.get(prev_dt, q_px)
                prev_v = vix_map.get(prev_dt, v_px)

                nvda_ret = (n_px / prev_n - 1.0) if prev_n > 0 else 0.0
                soxx_ret = (sx_px / prev_sx - 1.0) if (sx_px and prev_sx > 0) else nvda_ret
                qqq_ret = (q_px / prev_q - 1.0) if prev_q > 0 else 0.0
                vix_ret = (v_px / prev_v - 1.0) if prev_v > 0 else 0.0
            else:
                nvda_ret = soxx_ret = qqq_ret = vix_ret = 0.0

            sig_code, _, _ = cross_mod.predict_signal(
                tqqq_ret=tqqq_ret,
                nvda_ret=nvda_ret,
                soxx_ret=soxx_ret,
                qqq_ret=qqq_ret,
                vix_ret=vix_ret,
                tnx_ret=0.0
            )
            if sig_code > 0:
                cross_dirs.append("LONG_TQQQ")
            elif sig_code < 0:
                cross_dirs.append("SHORT_SQQQ")
            else:
                cross_dirs.append("HOLD")

        tqqq_15m_feat['cross_dir'] = cross_dirs
        unique_dates = sorted(tqqq_15m_feat['date_str'].unique())

        # 5분봉 날짜별 그룹핑 (고속 0ms 조회)
        tqqq_5m_by_date = {d: df for d, df in tqqq_5m.groupby('date_str')} if not tqqq_5m.empty else {}
        sqqq_5m_by_date = {d: df for d, df in sqqq_5m.groupby('date_str')} if not sqqq_5m.empty else {}

        return {
            "tqqq_15m": tqqq_15m,
            "tqqq_15m_feat": tqqq_15m_feat,
            "sqqq_15m_feat": sqqq_15m_feat,
            "tqqq_5m_by_date": tqqq_5m_by_date,
            "sqqq_5m_by_date": sqqq_5m_by_date,
            "soxx_60m": soxx_60m,
            "tqqq_60m": tqqq_60m,
            "unique_dates": unique_dates,
            "ml_feature_names": ml_fe.feature_names
        }

    def _load_candidate_1_champion(self) -> Tuple[MoEMetaOrchestrator, str]:
        """[후보 1: 기존 챔피언 (Current Champion)] 현재 실전 운용 중인 MoE 챔피언 모델 로드"""
        champ_path = MODELS_DIR / "model_champion.pkl"
        cand1_moe = None
        if champ_path.exists():
            try:
                cand1_moe = joblib.load(champ_path)
            except Exception:
                pass
        if cand1_moe is None and MOE_MODEL_PATH.exists():
            try:
                cand1_moe = joblib.load(MOE_MODEL_PATH)
            except Exception:
                pass
        if cand1_moe is None:
            cand1_moe = MoEMetaOrchestrator(confidence_threshold=self.confidence_threshold, gbdt_threshold=self.confidence_threshold, mode="hybrid_v3")

        cand1_moe.confidence_threshold = self.confidence_threshold
        cand1_moe.gbdt_threshold = self.confidence_threshold
        cand1_moe.mode = "hybrid_v3"

        model_id = getattr(cand1_moe, "model_id", "M-CURRENT-CHAMPION")
        return cand1_moe, model_id

    def _train_candidate_2_data_refresh(self, df_15m: Optional[pd.DataFrame] = None) -> Tuple[Any, List[str], List[str], List[str]]:
        """
        [후보 2: 데이터 최신화 (Data Refresh 504D)]
        - 최근 2년(정확히 504 거래일) 롤링 윈도우 방식으로 DB(market_data.db) 데이터 추출
        - 1주일 치 데이터 추가 시 가장 오래된 2년 전 과거 데이터(꼬리)를 절삭(Drop)하여 Concept Drift 방지
        - 최신 반도체 시장(TQQQ/SQQQ) 마이크로스트럭처에 가중치 부여된 LightGBM 모델 재학습
        """
        if df_15m is None:
            df_15m = self.data_lake.load_rolling_candles("TQQQ", "15m", max_trading_days=504)

        ml = MLFeatureEngine(confidence_threshold=self.confidence_threshold)
        feat_df = ml.extract_features(df_15m)
        model, top_10, top_3 = ml.train_and_select_top_features(feat_df)

        if model is not None:
            joblib.dump(model, MODELS_DIR / "model_data_refresh.pkl")
            joblib.dump(model, MODELS_DIR / "model_main_data_refresh.pkl")

        return model, ml.feature_names, top_10, top_3

    def _train_candidate_3_hyperparameter_tuned(self, df_features: pd.DataFrame) -> Tuple[Any, List[str], List[str], List[str]]:
        """
        [후보 3: Triple Barrier 3-Class 단일 LightGBM 하이퍼파라미터 튜닝 최적화 모델]
        - 순수 단일 LightGBM 구조 (Random Forest 배깅 완전 배제)
        - 시계열 순서를 보존하는 TimeSeriesSplit 교차검증 기반 핵심 하이퍼파라미터 탐색
        - 최적 파라미터 (learning_rate, max_depth, num_leaves, min_child_samples, colsample_bytree)
        """
        df = df_features.copy().dropna()
        target_series = MLFeatureEngine.compute_triple_barrier_labels(
            df,
            take_profit=0.030,
            stop_loss=0.020,
            horizon=6
        )
        label_map = {-1: 0, 0: 1, 1: 2}
        y = target_series.map(label_map).fillna(1).astype(int)

        feature_cols = [c for c in df.columns if c not in [
            'open', 'high', 'low', 'close', 'volume', 'Open', 'High', 'Low', 'Close', 'Volume',
            'date_str', 'Confidence', 'Signal', 'Direction', 'Prob_Long', 'Prob_Short', 'Prob_Neutral',
            'datetime', 'Datetime', 'Target', 'cross_dir', 'datetime_dt', 'time_str'
        ] and pd.api.types.is_numeric_dtype(df[c])]

        X = df[feature_cols]

        # 단일 LightGBM 하이퍼파라미터 그리드 후보군
        param_candidates = [
            {"learning_rate": 0.03, "max_depth": 4, "num_leaves": 15, "min_child_samples": 20, "colsample_bytree": 0.8},
            {"learning_rate": 0.05, "max_depth": 4, "num_leaves": 15, "min_child_samples": 25, "colsample_bytree": 0.8},
            {"learning_rate": 0.03, "max_depth": 5, "num_leaves": 25, "min_child_samples": 20, "colsample_bytree": 0.85},
            {"learning_rate": 0.04, "max_depth": 3, "num_leaves": 10, "min_child_samples": 30, "colsample_bytree": 0.9},
            {"learning_rate": 0.02, "max_depth": 4, "num_leaves": 15, "min_child_samples": 20, "colsample_bytree": 0.8},
        ]

        # TimeSeriesSplit 교차검증 (미래 데이터 누수 방지)
        tscv = TimeSeriesSplit(n_splits=3)
        best_params = param_candidates[0]
        best_val_score = -999.0

        for params in param_candidates:
            fold_scores = []
            for train_idx, val_idx in tscv.split(X):
                X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
                X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

                clf = LGBMClassifier(
                    objective='multiclass',
                    num_class=3,
                    class_weight='balanced',
                    n_estimators=100,
                    random_state=42,
                    verbose=-1,
                    **params
                )
                clf.fit(X_tr, y_tr)
                score = clf.score(X_val, y_val)
                fold_scores.append(score)

            mean_score = float(np.mean(fold_scores))
            if mean_score > best_val_score:
                best_val_score = mean_score
                best_params = params

        # 최적 하이퍼파라미터로 전체 504D 롤링 데이터 최종 단일 학습
        best_model = LGBMClassifier(
            objective='multiclass',
            num_class=3,
            class_weight='balanced',
            n_estimators=100,
            random_state=42,
            verbose=-1,
            **best_params
        )
        best_model.fit(X, y)

        # 피처 중요도 기준 Top 10 산출
        importances = best_model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        top_10 = [feature_cols[i] for i in sorted_indices[:10]]
        top_3 = top_10[:3]

        return best_model, feature_cols, top_10, top_3

    def _compute_composite_score(self, win_rate: float, pf: float, ret: float, mdd: float) -> float:
        """종합 퀀트 스코어 산출 공식"""
        # Score = (WinRate * 0.4) + (PF * 20) + (Return * 0.3) - (MDD * 0.5)
        score = (win_rate * 0.4) + (pf * 20.0) + (ret * 0.3) - (mdd * 0.5)
        return round(score, 2)

    def _backtest_moe_candidate(
        self,
        moe_obj: MoEMetaOrchestrator,
        cache: Dict[str, Any],
        tp_pct: float = 0.030,
        sl_pct: float = -0.020
    ) -> Dict[str, Any]:
        """
        [실전 매매 동일 하이브리드 MoE 전구간 백테스팅 엔진]
        - GBDT 확신도 60% 공격수 트리거
        - 크로스에셋 정반대 방향 Veto 방패
        - Screen 1 (60분봉 20 EMA) & Screen 3 (5분봉 VWAP/RSI 눌림목)
        - 5분봉 정밀 궤적 추적 (Path Dissection): TP / SL / 90분 타임스탑 / 15:45 NYT EOD 0% 오버나잇
        - 거래비용 0.20% 왕복 및 페이업 $0.03 슬리피지 완전 차감
        - 일일 3회 손절(3-Out) 서킷 브레이커 가동
        """
        INITIAL_CAPITAL = 10_000_000.0
        capital = INITIAL_CAPITAL
        TP_PCT = abs(tp_pct)
        SL_PCT = -abs(sl_pct)
        TIME_STOP_BARS_5M = 18  # 90분 (5분봉 18개)
        SLIPPAGE_PAYUP = 0.03
        FEE_RATE = 0.0020
        T = self.confidence_threshold

        # 1. 15분봉 피처에 대해 후보 모델 확신도 컬럼 생성
        tqqq_15m_scored = moe_obj.gbdt_engine.add_confidence_columns(cache['tqqq_15m_feat'].copy())

        tqqq_5m_by_date = cache['tqqq_5m_by_date']
        sqqq_5m_by_date = cache['sqqq_5m_by_date']
        sqqq_15m_feat = cache['sqqq_15m_feat']
        soxx_60m = cache['soxx_60m']
        tqqq_60m = cache['tqqq_60m']
        unique_dates = cache['unique_dates']

        trades = []
        equity_curve = [capital]

        # 날짜별 15분봉 슬라이스 인덱싱
        scored_by_date = {d: df for d, df in tqqq_15m_scored.groupby('date_str')}

        for d_str in unique_dates:
            day_tqqq_15 = scored_by_date.get(d_str)
            if day_tqqq_15 is None or len(day_tqqq_15) < 5:
                continue

            day_tqqq_5 = tqqq_5m_by_date.get(d_str, pd.DataFrame())
            day_sqqq_5 = sqqq_5m_by_date.get(d_str, pd.DataFrame())

            daily_stoploss_count = 0
            b_idx = 0
            n_bars = len(day_tqqq_15)

            while b_idx < n_bars:
                cur_row = day_tqqq_15.iloc[b_idx]
                time_str = cur_row['time_str']
                cur_time = cur_row['datetime']

                # 09:45 이전 첫 봉 노이즈 배제 및 14:30 이후 신규 진입 금지
                if b_idx < 1 or time_str > "14:30":
                    b_idx += 1
                    continue

                if daily_stoploss_count >= 3:
                    b_idx += 1
                    continue

                dir_gbdt = cur_row['Direction']
                conf_gbdt = float(cur_row['Confidence'])
                dir_cross = cur_row.get('cross_dir', 'HOLD')

                # [인터락 1] GBDT 공격수 트리거 (>= 60%)
                is_gbdt_trigger = (dir_gbdt in ["LONG_TQQQ", "SHORT_SQQQ"]) and (conf_gbdt >= T)

                # [인터락 1-2] 크로스에셋 역풍 방패 (정반대 방향 Veto)
                is_cross_veto = (
                    (dir_gbdt == "LONG_TQQQ" and dir_cross == "SHORT_SQQQ") or
                    (dir_gbdt == "SHORT_SQQQ" and dir_cross == "LONG_TQQQ")
                )
                if not getattr(config, "USE_CROSS_ASSET_VETO", True):
                    is_cross_veto = False

                if not is_gbdt_trigger or is_cross_veto:
                    b_idx += 1
                    continue

                direction = dir_gbdt

                # [인터락 1-3] Screen 1: 60분봉 상위 추세
                past_soxx_60 = soxx_60m[soxx_60m['datetime'] <= cur_time]
                past_tqqq_60 = tqqq_60m[tqqq_60m['datetime'] <= cur_time]
                is_60m_trend_ok = True
                if len(past_soxx_60) >= 20 and len(past_tqqq_60) >= 20:
                    soxx_c = past_soxx_60['Close'].iloc[-1]
                    tqqq_c = past_tqqq_60['Close'].iloc[-1]
                    soxx_ema20 = past_soxx_60['ema20'].iloc[-1]
                    tqqq_ema20 = past_tqqq_60['ema20'].iloc[-1]
                    if direction == "LONG_TQQQ":
                        is_60m_trend_ok = (soxx_c >= soxx_ema20 * 0.998) and (tqqq_c >= tqqq_ema20 * 0.998)
                    else:
                        is_60m_trend_ok = (soxx_c <= soxx_ema20 * 1.002)

                if not getattr(config, "USE_60M_TREND_FILTER", True):
                    is_60m_trend_ok = True

                if not is_60m_trend_ok:
                    b_idx += 1
                    continue

                # [인터락 1-4] Screen 3: 단기 눌림목 타점 필터
                if direction == "LONG_TQQQ":
                    vwap_diff = float(cur_row.get("VWAP_Diff", 0.0))
                    rsi_14 = float(cur_row.get("RSI_14", 50.0))
                    bb_lower = float(cur_row.get("BB_Lower", 0.0))
                    cur_close = float(cur_row['Close'])
                    dip_ok = (vwap_diff <= 1.5) and (rsi_14 <= 62.0)
                    if bb_lower > 0:
                        dip_ok = dip_ok and (cur_close >= bb_lower * 1.001)
                else:
                    if cur_time in sqqq_15m_feat.index:
                        row_s = sqqq_15m_feat.loc[cur_time]
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

                # 진입 확정!
                chosen_symbol = "TQQQ" if direction == "LONG_TQQQ" else "SQQQ"
                if chosen_symbol == "TQQQ":
                    base_px = float(cur_row['Close'])
                else:
                    if cur_time in sqqq_15m_feat.index:
                        base_px = float(sqqq_15m_feat.loc[cur_time]['Close'])
                    else:
                        base_px = 40.0

                entry_px = round(base_px + SLIPPAGE_PAYUP, 2)
                shares = int(capital / entry_px)
                invested = shares * entry_px
                if shares <= 0 or invested <= 0:
                    b_idx += 1
                    continue

                # 5분봉 정밀 궤적 추적
                target_5m_df = day_tqqq_5 if chosen_symbol == "TQQQ" else day_sqqq_5
                post_5m = target_5m_df[target_5m_df['datetime'] > cur_time]
                if post_5m.empty:
                    b_idx += 1
                    continue

                tp_px = round(entry_px * (1 + TP_PCT), 2)
                sl_px = round(entry_px * (1 + SL_PCT), 2)

                exit_triggered = False
                exit_px = 0.0
                exit_reason = ""
                bars_held_5m = 0
                eval_5m = post_5m.iloc[:TIME_STOP_BARS_5M]

                for k in range(len(eval_5m)):
                    c5 = eval_5m.iloc[k]
                    c5_o = float(c5['Open'])
                    c5_h = float(c5['High'])
                    c5_l = float(c5['Low'])
                    c5_c = float(c5['Close'])
                    t5_str = c5['time_str']
                    bars_held_5m = k + 1

                    hit_tp = (c5_h >= tp_px)
                    hit_sl = (c5_l <= sl_px)

                    if hit_tp and hit_sl:
                        if c5_c >= c5_o:
                            exit_triggered = True
                            exit_px = round(tp_px - SLIPPAGE_PAYUP, 2)
                            exit_reason = f"TP(+{TP_PCT*100:.1f}%)"
                            break
                        else:
                            exit_triggered = True
                            exit_px = round(sl_px - SLIPPAGE_PAYUP, 2)
                            exit_reason = f"SL({SL_PCT*100:.1f}%)"
                            daily_stoploss_count += 1
                            break
                    elif hit_tp:
                        exit_triggered = True
                        exit_px = round(tp_px - SLIPPAGE_PAYUP, 2)
                        exit_reason = f"TP(+{TP_PCT*100:.1f}%)"
                        break
                    elif hit_sl:
                        exit_triggered = True
                        exit_px = round(sl_px - SLIPPAGE_PAYUP, 2)
                        exit_reason = f"SL({SL_PCT*100:.1f}%)"
                        daily_stoploss_count += 1
                        break

                    if t5_str >= "15:45":
                        exit_triggered = True
                        exit_px = round(c5_c - SLIPPAGE_PAYUP, 2)
                        exit_reason = "EOD(15:45)"
                        if exit_px < entry_px:
                            daily_stoploss_count += 1
                        break

                if not exit_triggered:
                    last_c5 = eval_5m.iloc[-1]
                    exit_px = round(float(last_c5['Close']) - SLIPPAGE_PAYUP, 2)
                    exit_reason = "TimeStop(90m)"
                    if exit_px < entry_px:
                        daily_stoploss_count += 1

                # 손익 계산 (비용 0.20% 반영)
                cost_amt = invested * FEE_RATE
                pnl_amt = (shares * (exit_px - entry_px)) - cost_amt
                capital += pnl_amt
                equity_curve.append(capital)

                trades.append({
                    "symbol": chosen_symbol,
                    "direction": direction,
                    "entry_time": cur_row['datetime'],
                    "entry_px": entry_px,
                    "exit_px": exit_px,
                    "pnl": pnl_amt,
                    "reason": exit_reason
                })

                # 당일 포지션 종료 후 다음 봉으로 전진
                jump_bars = max(1, int(np.ceil(bars_held_5m / 3.0)))
                b_idx += jump_bars

        # 성적표 집계
        total_trades = len(trades)
        wins = sum(1 for t in trades if t['pnl'] > 0)
        losses = total_trades - wins
        win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0

        gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        pf = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (99.9 if gross_profit > 0 else 0.0)

        total_return = round(((capital / INITIAL_CAPITAL) - 1.0) * 100.0, 2)

        eq_series = pd.Series(equity_curve)
        peak = eq_series.cummax()
        dd = (eq_series - peak) / peak
        mdd = round(abs(float(dd.min())) * 100.0, 2)

        score = self._compute_composite_score(win_rate, pf, total_return, mdd)

        return {
            "win_rate_pct": round(win_rate, 2),
            "profit_factor": pf,
            "total_return_pct": total_return,
            "mdd_pct": mdd,
            "total_trades_count": total_trades,
            "total_wins": wins,
            "total_losses": losses,
            "score": score,
            "final_capital_krw": int(capital)
        }

    def run_tournament(self) -> Dict[str, Any]:
        """
        [주간 3자 토너먼트 거버넌스 감사 및 실전 챔피언 자동 적용]
        1. 데이터 캐시 준비 (15m/5m/60m 및 거시 지표)
        2. 3대 후보 모델 MoE 하이브리드 V3 빌드
        3. 실전 매매 동일 백테스트 (GBDT 60% + 크로스에셋 Veto + 3중 스크린 + 5분봉 정밀 청산)
        4. 퀀트 종합 스코어 기반 1위 선정 및 실전 챔피언 파일(model_champion.pkl) 즉각 교체/적용
        5. SQLite 레지스트리 기록 및 텔레그램 공식 결과 보고서 발송
        """
        today_str = datetime.now().strftime("%Y-%m-%d")
        print("=" * 80)
        print(f"🏆  [Lumos 주간 3자 토너먼트 거버넌스 감사: {today_str}]  🏆")
        print(f"🧠  평가 엔진: Lumos V3 실전 하이브리드 MoE (GBDT 60% + 크로스에셋 Veto + 3중 스크린 + 5분봉 궤적 청산)")
        print("=" * 80)
        system_logger.log("INFO", "WeeklyTournament", f"🏆 주간 3자 토너먼트 거버넌스 감사 실행 ({today_str})")

        # 1. 시세 데이터 및 인디케이터 캐시
        cache = self._prepare_market_data_cache()

        # 2. 3대 후보 모델 구성
        # [후보 1: 기존 챔피언]
        cand1_moe, cand1_id = self._load_candidate_1_champion()

        # [후보 2: 데이터 최신화 (504D 롤링)]
        cand2_model, cand2_feats, cand2_top10, cand2_top3 = self._train_candidate_2_data_refresh()
        cand2_moe = MoEMetaOrchestrator(confidence_threshold=self.confidence_threshold, gbdt_threshold=self.confidence_threshold, mode="hybrid_v3")
        cand2_moe.gbdt_engine.model = cand2_model
        cand2_moe.gbdt_engine.feature_names = cand2_feats
        cand2_moe.gbdt_engine.top_10_features = cand2_top10
        cand2_moe.gbdt_engine.top_3_features = cand2_top3
        cand2_id = f"M-{datetime.now().strftime('%Y%m%d')}-REFRESH-504D"

        # [후보 3: 신규 하이퍼 튜닝 (단일 LightGBM 하이퍼파라미터 최적화)]
        df_tqqq_504 = self.data_lake.load_rolling_candles("TQQQ", "15m", max_trading_days=504)
        ml_temp = MLFeatureEngine(confidence_threshold=self.confidence_threshold)
        feat_df_504 = ml_temp.extract_features(df_tqqq_504)
        cand3_model, cand3_feats, cand3_top10, cand3_top3 = self._train_candidate_3_hyperparameter_tuned(feat_df_504)
        cand3_moe = MoEMetaOrchestrator(confidence_threshold=self.confidence_threshold, gbdt_threshold=self.confidence_threshold, mode="hybrid_v3")
        cand3_moe.gbdt_engine.model = cand3_model
        cand3_moe.gbdt_engine.feature_names = cand3_feats
        cand3_moe.gbdt_engine.top_10_features = cand3_top10
        cand3_moe.gbdt_engine.top_3_features = cand3_top3
        cand3_id = f"M-{datetime.now().strftime('%Y%m%d')}-TUNED-GBDT"

        # 3. 실전 동일 MoE 백테스트 실행 (Apples to Apples 동일 구간 검증)
        print("\n⏳ [2/5] [후보 1: 기존 챔피언] 실전 MoE 백테스트 평가 중...")
        res_cand1 = self._backtest_moe_candidate(cand1_moe, cache)
        print(f"   • [후보 1] 완료: 승률 {res_cand1['win_rate_pct']:.1f}% ({res_cand1['total_wins']}승 {res_cand1['total_losses']}패) | PF {res_cand1['profit_factor']:.2f} | 수익률 +{res_cand1['total_return_pct']:.2f}% | MDD -{res_cand1['mdd_pct']:.2f}% (종합점수: {res_cand1['score']:.1f}점)")

        print("\n⏳ [3/5] [후보 2: 데이터 최신화] 실전 MoE 백테스트 평가 중...")
        res_cand2 = self._backtest_moe_candidate(cand2_moe, cache)
        print(f"   • [후보 2] 완료: 승률 {res_cand2['win_rate_pct']:.1f}% ({res_cand2['total_wins']}승 {res_cand2['total_losses']}패) | PF {res_cand2['profit_factor']:.2f} | 수익률 +{res_cand2['total_return_pct']:.2f}% | MDD -{res_cand2['mdd_pct']:.2f}% (종합점수: {res_cand2['score']:.1f}점)")

        print("\n⏳ [4/5] [후보 3: 신규 하이퍼 튜닝] 실전 MoE 백테스트 평가 중...")
        res_cand3 = self._backtest_moe_candidate(cand3_moe, cache)
        print(f"   • [후보 3] 완료: 승률 {res_cand3['win_rate_pct']:.1f}% ({res_cand3['total_wins']}승 {res_cand3['total_losses']}패) | PF {res_cand3['profit_factor']:.2f} | 수익률 +{res_cand3['total_return_pct']:.2f}% | MDD -{res_cand3['mdd_pct']:.2f}% (종합점수: {res_cand3['score']:.1f}점)")

        # 4. 순위 정렬 및 1위 챔피언 확정
        candidates = [
            {
                "candidate_type": "기존 챔피언 (Champion)",
                "model_id": cand1_id,
                "moe_obj": cand1_moe,
                "win_rate": res_cand1["win_rate_pct"],
                "profit_factor": res_cand1["profit_factor"],
                "total_return": res_cand1["total_return_pct"],
                "mdd": res_cand1["mdd_pct"],
                "trades": res_cand1["total_trades_count"],
                "wins": res_cand1["total_wins"],
                "losses": res_cand1["total_losses"],
                "score": res_cand1["score"]
            },
            {
                "candidate_type": "데이터 최신화 (Data Refresh)",
                "model_id": cand2_id,
                "moe_obj": cand2_moe,
                "win_rate": res_cand2["win_rate_pct"],
                "profit_factor": res_cand2["profit_factor"],
                "total_return": res_cand2["total_return_pct"],
                "mdd": res_cand2["mdd_pct"],
                "trades": res_cand2["total_trades_count"],
                "wins": res_cand2["total_wins"],
                "losses": res_cand2["total_losses"],
                "score": res_cand2["score"]
            },
            {
                "candidate_type": "신규 하이퍼 튜닝 (Hyperparameter Tuned)",
                "model_id": cand3_id,
                "moe_obj": cand3_moe,
                "win_rate": res_cand3["win_rate_pct"],
                "profit_factor": res_cand3["profit_factor"],
                "total_return": res_cand3["total_return_pct"],
                "mdd": res_cand3["mdd_pct"],
                "trades": res_cand3["total_trades_count"],
                "wins": res_cand3["total_wins"],
                "losses": res_cand3["total_losses"],
                "score": res_cand3["score"]
            }
        ]

        candidates.sort(key=lambda x: x["score"], reverse=True)
        winner = candidates[0]
        is_champion_defended = (winner["candidate_type"] == "기존 챔피언 (Champion)")

        # 5. [핵심] 실제 거래 모델 즉시 디스크 적용 (Apply to Production)
        print("\n⏳ [5/5] 선발 챔피언 모델 실전 적용 및 텔레그램 리포트 발송...")
        winner_moe = winner["moe_obj"]
        winner_moe.confidence_threshold = self.confidence_threshold
        winner_moe.gbdt_threshold = self.confidence_threshold
        winner_moe.mode = "hybrid_v3"

        if is_champion_defended:
            print(f"   🛡️ [챔피언 방어 성공] 기존 챔피언({winner['model_id']})이 최고 점수({winner['score']}점)로 실전 운용을 지속합니다.")
            system_logger.log("INFO", "WeeklyTournament", f"🛡️ 챔피언 방어 완료: {winner['model_id']} (점수: {winner['score']}점)")
        else:
            print(f"   🔄 [신규 챔피언 공식 승격 및 교체] {winner['candidate_type']} ({winner['model_id']}) ➔ 실전 모델 파일(model_champion.pkl) 교체 완료!")
            system_logger.log("INFO", "WeeklyTournament", f"🔄 신규 챔피언 승격 및 적용: {winner['candidate_type']} ({winner['model_id']})")

        joblib.dump(winner_moe, MOE_MODEL_PATH)
        joblib.dump(winner_moe, MODELS_DIR / "model_champion.pkl")
        # 주말 백업 아카이브
        archive_path = MODELS_DIR / f"model_champion_{today_str.replace('-', '')}.pkl"
        joblib.dump(winner_moe, archive_path)

        # DB 기록
        for c in candidates:
            is_win = (c["model_id"] == winner["model_id"])
            self.registry.record_tournament_evaluation(
                eval_date=today_str,
                model_id=c["model_id"],
                candidate_type=c["candidate_type"],
                win_rate=c["win_rate"],
                profit_factor=c["profit_factor"],
                total_return=c["total_return"],
                mdd=c["mdd"],
                composite_score=c["score"],
                is_selected=is_win
            )

        # 텔레그램 리포트 작성 및 발송
        report = self._compose_tournament_report(today_str, candidates, winner)
        send_res = self.dispatcher.send_telegram_message(report)
        if send_res.get("ok"):
            print("   🚀 >>> 주간 토너먼트 결산 성적표 텔레그램 발송 완료! <<< 🚀")

        return {
            "eval_date": today_str,
            "winner": winner,
            "rankings": candidates
        }

    def _compose_tournament_report(self, eval_date: str, candidates: List[Dict[str, Any]], winner: Dict[str, Any]) -> str:
        """텔레그램용 주간 토너먼트 성적표 카드 작성 (방어 vs 승격 명시)"""
        is_champion_defended = (winner["candidate_type"] == "기존 챔피언 (Champion)")
        if is_champion_defended:
            status_banner = "✅ **[챔피언 유지 방어 완료]**\n기존 실전 챔피언 모델의 우수성이 입증되어 차주 실전 운용을 지속합니다."
        else:
            status_banner = f"🔄 **[신규 챌린저 실전 승격 완료]**\n`{winner['candidate_type']}` 모델이 최고 점수를 획득하여 차주 실전 챔피언(`model_champion.pkl`)으로 공식 교체/적용되었습니다."

        ranks_md = []
        medals = ["🥇 1위 (차주 실전 챔피언)", "🥈 2위", "🥉 3위"]
        for idx, c in enumerate(candidates):
            badge = "👑 [WINNER]" if idx == 0 else f"#{idx+1}"
            ranks_md.append(
                f"{medals[idx]} {badge} **[{c['candidate_type']}]**\n"
                f"  • 모델 ID: `{c['model_id']}`\n"
                f"  • 승률: `{c['win_rate']:.1f}%` ({c.get('wins', 0)}승 {c.get('losses', 0)}패) | 손익비(PF): `{c['profit_factor']:.2f}`\n"
                f"  • 누적수익률: `+{c['total_return']:.2f}%` | MDD: `-{c['mdd']:.2f}%`\n"
                f"  • 퀀트 종합 스코어: **`{c['score']:.1f}점`**"
            )

        ranks_str = "\n\n".join(ranks_md)

        return f"""🏆 **[Lumos 주간 3자 토너먼트 거버넌스 공식 결과 보고]**
━━━━━━━━━━━━━━━━━━━━
📅 **결산 일자:** `{eval_date} (주말 정기 결산)`
🧠 **평가 엔진:** `Lumos V3 실전 하이브리드 MoE (GBDT 60% + 크로스에셋 Veto + 3중 스크린 + 5분봉 정밀 청산)`
🏛 **검증 데이터:** 로컬 영구 데이터 레이크(`market_data.db`) 전수 검증

{status_banner}

━━━━━━━━━━━━━━━━━━━━
📊 **[3대 후보 맞대결 성적표]**
{ranks_str}

━━━━━━━━━━━━━━━━━━━━
🎯 **[차주 실전 가동 확정 및 파일 적용]**
• **선발 모델:** `{winner['model_id']}` (`{winner['candidate_type']}`)
• **실전 성적:** 승률 `{winner['win_rate']:.1f}%` ({winner.get('wins', 0)}승 {winner.get('losses', 0)}패) / PF `{winner['profit_factor']:.2f}` / 수익률 `+{winner['total_return']:.2f}%` / MDD `-{winner['mdd']:.2f}%`
• **종합 점수:** **`{winner['score']:.1f}점`**
• **적용 파일:** `models/model_champion.pkl` (실전 가동 즉시 반영)

🛡 **[안전 거버넌스 롤백 룰]**
실전 운용 중 주간 승률이 50% 미만으로 저하될 경우, 안전 기준점인 `model_golden_baseline.pkl (+24.38%)`로 자동 롤백됩니다."""
