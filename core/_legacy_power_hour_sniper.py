"""
Lumos V3: Power Hour Sniper Engine (5-Minute Intraday Sprint)
- Operating Window: 14:30 ~ 15:30 EDT (미국 장 마감 90분 전 ~ 30분 전)
- Model: 5-minute LightGBM Multiclass Classifier
- Rules: TP +2.5% / SL -1.67% (손익비 1.50:1) / 30분 타임스탑 / GBDT 확신도 >= 55%
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import MODELS_DIR
from core.data_lake import MarketDataLake
from core.ml_engine import MLFeatureEngine
from core.heterogeneous_models import CrossAssetDislocationModel

SNIPER_MODEL_PATH = MODELS_DIR / "model_sniper_5m.pkl"

FEATURE_COLS = [
    'RSI_7', 'RSI_14', 'RSI_21', 'MACD', 'MACD_Hist', 'Stoch_K', 'Stoch_D',
    'ROC_5', 'ROC_10', 'ROC_20', 'CCI_14', 'CCI_20', 'ATR_14', 'BB_PctB',
    'BB_Bandwidth', 'KC_Width', 'OBV', 'CMF_20', 'Volume_Z', 'VWAP_Diff',
    'EMA_9_Diff', 'EMA_21_Diff', 'EMA_50_Diff', 'ADX_14', 'DMI_Diff',
    'Body_Ratio', 'Upper_Wick_Ratio', 'Lower_Wick_Ratio'
]

class PowerHourSniper:
    """
    [Phase 2: 장 마감 파워 아워 전용 5분봉 초고속 스나이퍼]
    - 익절: +2.5% | 손절: -1.67% (손익비 1.50:1)
    - 타임스탑: 30분 (5분봉 6개 캔들)
    - GBDT 확신도: >= 55% (DOE 최적 알파 세팅: 14승 7패 승률 66.7% / +207만원 기여)
    """
    def __init__(
        self,
        tp_pct: float = 0.025,
        sl_pct: float = 0.0167,
        time_stop_minutes: int = 30,
        confidence_threshold: float = 0.55
    ):
        self.tp_pct = tp_pct
        self.sl_pct = sl_pct
        self.time_stop_minutes = time_stop_minutes
        self.confidence_threshold = confidence_threshold
        self.data_lake = MarketDataLake()
        self.ml_engine = MLFeatureEngine()
        self.cross_asset_model = CrossAssetDislocationModel(dislocation_z_threshold=1.6)
        self.model: Optional[LGBMClassifier] = None
        self.feature_cols = FEATURE_COLS
        self._load_or_train()

    def _load_or_train(self):
        """디스크에서 5m 스나이퍼 모델 로드 또는 즉시 학습"""
        if SNIPER_MODEL_PATH.exists():
            try:
                saved_dict = joblib.load(SNIPER_MODEL_PATH)
                if isinstance(saved_dict, dict) and "model" in saved_dict:
                    self.model = saved_dict["model"]
                    self.feature_cols = saved_dict.get("feature_cols", FEATURE_COLS)
                    self.confidence_threshold = saved_dict.get("confidence_threshold", self.confidence_threshold)
                    return
                elif isinstance(saved_dict, LGBMClassifier):
                    self.model = saved_dict
                    return
            except Exception:
                pass

        # 파일이 없으면 자동 학습
        self.train_and_save()

    def train_and_save(self, df_5m: Optional[pd.DataFrame] = None) -> LGBMClassifier:
        """5분봉 데이터로 Triple Barrier (+2.5% / -1.67% / 30m) 라벨링 후 모델 학습 및 저장"""
        if df_5m is None:
            df_5m = self.data_lake.load_candles("TQQQ", "5m")

        df_5m = df_5m.copy()
        if 'datetime' in df_5m.columns:
            df_5m['datetime_dt'] = pd.to_datetime(df_5m['datetime'])
            df_5m['date_str'] = df_5m['datetime_dt'].dt.strftime('%Y-%m-%d')
            df_5m['time_str'] = df_5m['datetime_dt'].dt.strftime('%H:%M')
        elif isinstance(df_5m.index, pd.DatetimeIndex):
            df_5m['datetime_dt'] = df_5m.index
            df_5m['date_str'] = df_5m['datetime_dt'].strftime('%Y-%m-%d')
            df_5m['time_str'] = df_5m['datetime_dt'].strftime('%H:%M')

        feat_5m = self.ml_engine.extract_features(df_5m)
        clean_5m = feat_5m.dropna(subset=self.feature_cols).copy()

        labels_5m = MLFeatureEngine.compute_triple_barrier_labels(
            clean_5m,
            take_profit=self.tp_pct,
            stop_loss=self.sl_pct,
            horizon=int(self.time_stop_minutes / 5)
        )

        y_5m = labels_5m.map({-1: 0, 0: 1, 1: 2}).fillna(1).astype(int)
        X_5m = clean_5m[self.feature_cols]

        clf = LGBMClassifier(
            objective='multiclass',
            num_class=3,
            class_weight='balanced',
            n_estimators=100,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=31,
            min_child_samples=20,
            random_state=42,
            verbose=-1
        )
        clf.fit(X_5m, y_5m)
        self.model = clf

        save_dict = {
            "model": clf,
            "feature_cols": self.feature_cols,
            "tp_pct": self.tp_pct,
            "sl_pct": self.sl_pct,
            "time_stop_minutes": self.time_stop_minutes,
            "confidence_threshold": self.confidence_threshold,
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(save_dict, SNIPER_MODEL_PATH)
        return clf

    def evaluate_sniper_signal(
        self,
        df_candle_5m: pd.DataFrame,
        current_time_str: Optional[str] = None,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        [5분봉 스나이퍼 실시간 타점 평가]
        1. 5분봉 피처 풀 생성
        2. GBDT 확신도 평가 (>= 55%)
        3. 60분봉 대추세 정렬 (Screen 1)
        4. 5분봉 단기 과열 방지 (RSI 필터)
        """
        if self.model is None:
            self._load_or_train()

        T = threshold if threshold is not None else self.confidence_threshold

        # 방어 가드: 빈 데이터프레임 또는 데이터 부족 시 안전 반환
        if df_candle_5m is None or df_candle_5m.empty or len(df_candle_5m) < 15:
            return {
                "expert_desc": "5m Power Hour Sniper",
                "direction": "NONE",
                "confidence": 0.0,
                "threshold_applied": float(T),
                "is_approved": False,
                "is_cross_veto": False,
                "dir_cross": "HOLD",
                "conf_cross": 0.50,
                "is_60m_trend_ok": False,
                "is_dip_ok": False,
                "dynamic_tp_px": 0.0,
                "dynamic_sl_px": 0.0,
                "tp_pct": round(self.tp_pct * 100.0, 2),
                "sl_pct": round(self.sl_pct * 100.0, 2),
                "time_stop_minutes": self.time_stop_minutes,
                "timestamp": current_time_str or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

        # 피처 생성
        df_feat = self.ml_engine.extract_features(df_candle_5m)
        if df_feat.empty:
            return {
                "expert_desc": "5m Power Hour Sniper",
                "direction": "NONE",
                "confidence": 0.0,
                "threshold_applied": float(T),
                "is_approved": False,
                "is_cross_veto": False,
                "dir_cross": "HOLD",
                "conf_cross": 0.50,
                "is_60m_trend_ok": False,
                "is_dip_ok": False,
                "dynamic_tp_px": 0.0,
                "dynamic_sl_px": 0.0,
                "tp_pct": round(self.tp_pct * 100.0, 2),
                "sl_pct": round(self.sl_pct * 100.0, 2),
                "time_stop_minutes": self.time_stop_minutes,
                "timestamp": current_time_str or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

        last_row = df_feat.iloc[-1]

        # 피처 벡터 추출
        X_vec = df_feat[self.feature_cols].iloc[[-1]].copy()
        for col in self.feature_cols:
            if col not in X_vec.columns or pd.isna(X_vec[col].iloc[0]):
                X_vec[col] = 0.0

        probs = self.model.predict_proba(X_vec)[0]  # [p_short, p_flat, p_long]
        p_short, p_flat, p_long = probs[0], probs[1], probs[2]

        direction = "NONE"
        confidence = p_flat
        if p_long >= p_short and p_long >= p_flat:
            direction = "LONG_TQQQ"
            confidence = p_long
        elif p_short >= p_long and p_short >= p_flat:
            direction = "SHORT_SQQQ"
            confidence = p_short

        is_gbdt_trigger = (direction in ["LONG_TQQQ", "SHORT_SQQQ"]) and (confidence >= T)

        # 🛡️ Cross-Asset Veto 방패 검증 (선행 매크로 상충 역풍 차단)
        is_cross_veto = False
        dir_cross = "HOLD"
        conf_cross = 0.50
        try:
            if current_time_str:
                nvda_5m = self.data_lake.load_candles("NVDA", "5m", end_dt=current_time_str)
                qqq_5m = self.data_lake.load_candles("QQQ", "5m", end_dt=current_time_str)
                soxx_5m = self.data_lake.load_candles("SOXX", "5m", end_dt=current_time_str)
                vix_5m = self.data_lake.load_candles("^VIX", "5m", end_dt=current_time_str)
            else:
                nvda_5m = self.data_lake.load_candles("NVDA", "5m")
                qqq_5m = self.data_lake.load_candles("QQQ", "5m")
                soxx_5m = self.data_lake.load_candles("SOXX", "5m")
                vix_5m = self.data_lake.load_candles("^VIX", "5m")

            if not nvda_5m.empty and not qqq_5m.empty and not vix_5m.empty:
                n_c = nvda_5m['Close'] if 'Close' in nvda_5m else nvda_5m['close']
                q_c = qqq_5m['Close'] if 'Close' in qqq_5m else qqq_5m['close']
                v_c = vix_5m['Close'] if 'Close' in vix_5m else vix_5m['close']
                soxx_c = soxx_5m['Close'] if 'Close' in soxx_5m else soxx_5m['close'] if not soxx_5m.empty else n_c
                s_c = df_candle_5m['Close'] if 'Close' in df_candle_5m else df_candle_5m['close']

                nvda_r = float(n_c.iloc[-1] / n_c.iloc[-5] - 1.0)
                qqq_r = float(q_c.iloc[-1] / q_c.iloc[-5] - 1.0)
                vix_r = float(v_c.iloc[-1] / v_c.iloc[-5] - 1.0)
                soxx_r = float(soxx_c.iloc[-1] / soxx_c.iloc[-5] - 1.0) if not soxx_5m.empty else nvda_r
                tqqq_r = float(s_c.iloc[-1] / s_c.iloc[-5] - 1.0)

                sig_code, exp_conf, _ = self.cross_asset_model.predict_signal(
                    tqqq_ret=tqqq_r,
                    nvda_ret=nvda_r,
                    soxx_ret=soxx_r,
                    qqq_ret=qqq_r,
                    vix_ret=vix_r,
                    tnx_ret=0.0
                )
                conf_cross = exp_conf
                if sig_code > 0:
                    dir_cross = "LONG_TQQQ"
                elif sig_code < 0:
                    dir_cross = "SHORT_SQQQ"

                dir_gbdt = direction
                is_cross_veto = (
                    (dir_gbdt == "LONG_TQQQ" and dir_cross == "SHORT_SQQQ") or
                    (dir_gbdt == "SHORT_SQQQ" and dir_cross == "LONG_TQQQ")
                )
                if not getattr(config, "USE_CROSS_ASSET_VETO", True):
                    is_cross_veto = False
        except Exception:
            pass

        # Screen 1: 상위 60분봉 추세 정렬
        is_60m_trend_ok = True
        try:
            soxx_60m = self.data_lake.load_candles("SOXX", "60m")
            if not soxx_60m.empty and len(soxx_60m) >= 20:
                s_c = soxx_60m['Close'] if 'Close' in soxx_60m else soxx_60m['close']
                soxx_c = s_c.iloc[-1]
                soxx_ema = s_c.ewm(span=20, adjust=False).mean().iloc[-1]
                if direction == "LONG_TQQQ":
                    is_60m_trend_ok = (soxx_c >= soxx_ema * 0.998)
                elif direction == "SHORT_SQQQ":
                    is_60m_trend_ok = (soxx_c <= soxx_ema * 1.002)
        except Exception:
            pass

        if not getattr(config, "USE_60M_TREND_FILTER", True):
            is_60m_trend_ok = True

        # Screen 3: 단기 5분봉 과열 필터
        rsi_5m = float(last_row.get("RSI_14", 50.0))
        is_dip_ok = True
        if direction == "LONG_TQQQ" and rsi_5m > 68.0:
            is_dip_ok = False
        elif direction == "SHORT_SQQQ" and rsi_5m < 32.0:
            is_dip_ok = False

        is_approved = bool(is_gbdt_trigger and (not is_cross_veto) and is_60m_trend_ok and is_dip_ok)

        cur_px = float(last_row.get("Close", last_row.get("close", 0.0)))
        dynamic_tp_px = round(cur_px * (1 + self.tp_pct), 2)
        dynamic_sl_px = round(cur_px * (1 - self.sl_pct), 2)

        return {
            "expert_desc": "5m Power Hour Sniper",
            "direction": direction,
            "confidence": float(confidence),
            "threshold_applied": float(T),
            "is_approved": is_approved,
            "is_cross_veto": is_cross_veto,
            "dir_cross": dir_cross,
            "conf_cross": float(conf_cross),
            "is_60m_trend_ok": is_60m_trend_ok,
            "is_dip_ok": is_dip_ok,
            "dynamic_tp_px": dynamic_tp_px,
            "dynamic_sl_px": dynamic_sl_px,
            "tp_pct": round(self.tp_pct * 100.0, 2),
            "sl_pct": round(self.sl_pct * 100.0, 2),
            "time_stop_minutes": self.time_stop_minutes,
            "timestamp": current_time_str or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
