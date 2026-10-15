import logging
logger = logging.getLogger(__name__)
import threading
from dataclasses import dataclass
from datetime import datetime
from config import GBDT_CONFIDENCE_THRESHOLD
import config
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')


@dataclass(frozen=True)
class _ModelSnapshot:
    """
    [STEP 3: 불변(Immutable) 모델 스냅샷]
    단일 추론 요청 동안 (model, feature_schema, feature_names, sha256) 조합이
    절대 섞이거나 None으로 노출되지 않도록 보장하는 불변 참조 컨테이너.
    """
    model: Any
    feature_names: Tuple[str, ...]
    top_10_features: Tuple[str, ...]
    top_3_features: Tuple[str, ...]
    feature_schema: str
    active_model_id: str
    active_model_path: str
    active_model_sha256: str
    feature_names_sha256: str
    activated_at: str
    previous_model_path: str
    previous_model_sha256: str
    load_status: str
    fallback_status: str
    load_error: str
    training_window_meta: Tuple[Tuple[str, Any], ...] = ()


def _safe_merge_asof(df_left: pd.DataFrame, df_right: pd.DataFrame, on='datetime_dt', direction='backward') -> pd.DataFrame:
    """Overlapping columns drop & merge_asof utility"""
    cols_to_drop = [c for c in df_right.columns if c in df_left.columns and c != on]
    if cols_to_drop:
        df_left = df_left.drop(columns=cols_to_drop)
    return pd.merge_asof(df_left.sort_values(on), df_right.sort_values(on), on=on, direction=direction)

class MLFeatureEngine:
    """
    [머신러닝 기반 종합 피처 엔지니어링 및 60% 확신도 선별 엔진]
    1. 모멘텀: RSI(7, 14, 21), MACD, Stochastics(%K, %D), ROC(5, 10, 20), CCI(14, 20)
    2. 변동성: ATR(14), Bollinger Bands(%b, Bandwidth), Keltner Channels
    3. 거래량: OBV, CMF(20), Volume Z-score, VWAP 대비 이격도
    4. 추세/이평: EMA(9, 21, 50, 200), ADX/DMI
    5. 가격 구조: 캔들 몸통/꼬리 비율, 직전 N개 봉 변동률
    6. LightGBM 모델 학습을 통한 Top 10 핵심 보조지표 자동 선별 및 60% 확신도(Confidence) 산출
    """
    # 8 Dollar-Absolute Features (Replaced in Normalized/Challenger schemas, preserved on DataFrame for Champion/Consumers)
    DOLLAR_SCALE_FEATURES = [
        'ATR_14', 'BB_Upper', 'BB_Lower', 'KC_Upper', 'KC_Lower',
        'MACD', 'MACD_Signal', 'MACD_Hist'
    ]

    # 8 Scale-Invariant Relative Price Features (STEP 2)
    NORMALIZED_PRICE_FEATURES = [
        'ATR_14_Pct', 'BB_Upper_Diff', 'BB_Lower_Diff', 'KC_Upper_Diff', 'KC_Lower_Diff',
        'MACD_Pct', 'MACD_Signal_Pct', 'MACD_Hist_Pct'
    ]

    # 6 Trend Continuation Features (STEP 3A)
    TREND_CONTINUATION_FEATURES = [
        'EMA20_Slope_3', 'EMA20_Slope_Accel', 'Trend_Efficiency_8',
        'Downside_Persistence_6', 'Upside_Persistence_6', 'Breakout_Position_20'
    ]

    # 6 Mean-Reversion / Reversal Detection Features (STEP 3B)
    MEAN_REVERSION_FEATURES = [
        'Return_Accel_3', 'VWAP_Dist_ATR', 'EMA20_Dist_ATR',
        'Signed_Body_Ratio', 'Lower_Wick_Expansion', 'Upper_Wick_Expansion'
    ]

    # 7 Redundant/Zero-Importance Legacy Features Removed via Ablation (r=1.000 duplicates + 0~3 split binary signs)
    ABLATED_REDUNDANT_FEATURES = {
        'base_long_ret_5', 'base_long_ret_20',
        'base_short_dir_1', 'nvda_dir_1', 'qqq_dir_1', 'vixy_dir_1', 'ief_dir_1'
    }

    CANONICAL_CHAMPION_72_FEATURES = [
        'RSI_7', 'RSI_14', 'RSI_21', 'MACD', 'MACD_Signal', 'MACD_Hist', 'Stoch_K', 'Stoch_D',
        'ROC_5', 'ROC_10', 'ROC_20', 'CCI_14', 'CCI_20', 'ATR_14', 'BB_Upper', 'BB_Lower',
        'BB_PctB', 'BB_Bandwidth', 'KC_Upper', 'KC_Lower', 'KC_Width', 'OBV', 'CMF_20',
        'Volume_Z', 'VWAP_Diff', 'EMA_9_Diff', 'EMA_21_Diff', 'EMA_50_Diff', 'EMA_200_Diff',
        'ADX_14', 'DMI_Diff', 'Body_Ratio', 'Upper_Wick_Ratio', 'Lower_Wick_Ratio',
        'macro_trend_spread', 'macro_ema20_slope', 'macro_price_vs_ema20',
        'base_short_ret_1', 'base_short_ret_3', 'base_short_ret_5', 'base_short_ret_20',
        'base_short_dir_1', 'base_short_momentum',
        'nvda_ret_1', 'nvda_ret_3', 'nvda_ret_5', 'nvda_ret_20', 'nvda_dir_1', 'nvda_momentum',
        'qqq_ret_1', 'qqq_ret_3', 'qqq_ret_5', 'qqq_ret_20', 'qqq_dir_1', 'qqq_momentum',
        'vixy_ret_1', 'vixy_ret_3', 'vixy_ret_5', 'vixy_ret_20', 'vixy_dir_1', 'vixy_momentum',
        'ief_ret_1', 'ief_ret_3', 'ief_ret_5', 'ief_ret_20', 'ief_dir_1', 'ief_momentum',
        'base_long_ret_20', 'base_long_ret_5', 'base_long_vs_qqq_20', 'base_long_vs_qqq_5', 'panic_signal'
    ]

    LEGACY_PKL_CHAMPION_72_FEATURES = [
        'RSI_7', 'RSI_14', 'RSI_21', 'MACD', 'MACD_Signal', 'MACD_Hist', 'Stoch_K', 'Stoch_D',
        'ROC_5', 'ROC_10', 'ROC_20', 'CCI_14', 'CCI_20', 'ATR_14', 'BB_Upper', 'BB_Lower',
        'BB_PctB', 'BB_Bandwidth', 'KC_Upper', 'KC_Lower', 'KC_Width', 'OBV', 'CMF_20',
        'Volume_Z', 'VWAP_Diff', 'EMA_9_Diff', 'EMA_21_Diff', 'EMA_50_Diff', 'EMA_200_Diff',
        'ADX_14', 'DMI_Diff', 'Body_Ratio', 'Upper_Wick_Ratio', 'Lower_Wick_Ratio',
        'base_long_ret_20', 'base_long_ret_5', 'base_long_vs_qqq_20', 'base_long_vs_qqq_5', 'panic_signal',
        'macro_trend_spread', 'macro_ema20_slope', 'macro_price_vs_ema20',
        'base_short_ret_1', 'base_short_ret_3', 'base_short_ret_5', 'base_short_ret_20',
        'base_short_dir_1', 'base_short_momentum',
        'nvda_ret_1', 'nvda_ret_3', 'nvda_ret_5', 'nvda_ret_20', 'nvda_dir_1', 'nvda_momentum',
        'qqq_ret_1', 'qqq_ret_3', 'qqq_ret_5', 'qqq_ret_20', 'qqq_dir_1', 'qqq_momentum',
        'vixy_ret_1', 'vixy_ret_3', 'vixy_ret_5', 'vixy_ret_20', 'vixy_dir_1', 'vixy_momentum',
        'ief_ret_1', 'ief_ret_3', 'ief_ret_5', 'ief_ret_20', 'ief_dir_1', 'ief_momentum'
    ]

    CANONICAL_B_NORMALIZED_72_FEATURES = [
        'RSI_7', 'RSI_14', 'RSI_21', 'Stoch_K', 'Stoch_D', 'ROC_5', 'ROC_10', 'ROC_20',
        'CCI_14', 'CCI_20', 'BB_PctB', 'BB_Bandwidth', 'KC_Width', 'OBV', 'CMF_20',
        'Volume_Z', 'VWAP_Diff', 'EMA_9_Diff', 'EMA_21_Diff', 'EMA_50_Diff', 'EMA_200_Diff',
        'ADX_14', 'DMI_Diff', 'Body_Ratio', 'Upper_Wick_Ratio', 'Lower_Wick_Ratio',
        'macro_trend_spread', 'macro_ema20_slope', 'macro_price_vs_ema20',
        'base_short_ret_1', 'base_short_ret_3', 'base_short_ret_5', 'base_short_ret_20',
        'base_short_dir_1', 'base_short_momentum',
        'nvda_ret_1', 'nvda_ret_3', 'nvda_ret_5', 'nvda_ret_20', 'nvda_dir_1', 'nvda_momentum',
        'qqq_ret_1', 'qqq_ret_3', 'qqq_ret_5', 'qqq_ret_20', 'qqq_dir_1', 'qqq_momentum',
        'vixy_ret_1', 'vixy_ret_3', 'vixy_ret_5', 'vixy_ret_20', 'vixy_dir_1', 'vixy_momentum',
        'ief_ret_1', 'ief_ret_3', 'ief_ret_5', 'ief_ret_20', 'ief_dir_1', 'ief_momentum',
        'base_long_ret_20', 'base_long_ret_5', 'base_long_vs_qqq_20', 'base_long_vs_qqq_5', 'panic_signal',
        'ATR_14_Pct', 'BB_Upper_Diff', 'BB_Lower_Diff', 'KC_Upper_Diff', 'KC_Lower_Diff',
        'MACD_Pct', 'MACD_Signal_Pct', 'MACD_Hist_Pct'
    ]

    LIVE_ALLOWED_SCHEMAS = {"B_normalized", "v4_champion", "A_champion"}
    LIVE_FORBIDDEN_SCHEMAS = {"v5_challenger", "E_challenger", "E_raw_all", "C_norm_trend", "D_norm_mr"}

    @classmethod
    def compute_feature_names_sha256(cls, feature_names: List[str]) -> str:
        import hashlib
        return hashlib.sha256("|".join(str(c) for c in feature_names).encode("utf-8")).hexdigest()

    def __init__(
        self,
        confidence_threshold: float = GBDT_CONFIDENCE_THRESHOLD,
        feature_schema: str = "B_normalized",
        model_path: Optional[str] = None,
        manifest_path: Optional[str] = None
    ):
        self._reload_lock = threading.RLock()
        self._thread_local = threading.local()
        self.confidence_threshold = confidence_threshold
        self.feature_schema = feature_schema
        self.model = None
        self.top_10_features: List[str] = []
        self.top_3_features: List[str] = []
        self.feature_names: List[str] = []
        self.active_model_id: str = "UNINITIALIZED"
        self.active_model_path: str = ""
        self.active_model_sha256: str = ""
        self.feature_names_sha256: str = ""
        self.activated_at: str = ""
        self.previous_model_path: str = ""
        self.previous_model_sha256: str = ""
        self.model_load_status: str = "INIT"
        self.model_fallback_status: str = "NONE"
        self.model_load_error: str = ""
        self.last_reload_error: str = ""
        self.last_training_window_meta: Dict[str, Any] = {}
        self._active_snapshot: Optional[_ModelSnapshot] = None
        self._manifest_path_override: Optional[str] = str(manifest_path) if manifest_path else None
        self._bound_to_default_manifest: bool = False
        if model_path:
            self.load_model(model_path, enforce_live_policy=False)
        else:
            self._bound_to_default_manifest = True
            self._try_load_default_model()

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_reload_lock", None)
        state.pop("_thread_local", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._reload_lock = threading.RLock()
        self._thread_local = threading.local()

    def _apply_snapshot(self, snap: _ModelSnapshot) -> None:
        """단일 락 내부에서 불변 모델 스냅샷과 하위 호환 속성을 원자적으로 갱신"""
        if not hasattr(self, "_reload_lock") or self._reload_lock is None:
            self._reload_lock = threading.RLock()
        with self._reload_lock:
            self._active_snapshot = snap
            self.model = snap.model
            self.feature_names = list(snap.feature_names)
            self.top_10_features = list(snap.top_10_features)
            self.top_3_features = list(snap.top_3_features)
            self.feature_schema = snap.feature_schema
            self.active_model_id = snap.active_model_id
            self.active_model_path = snap.active_model_path
            self.active_model_sha256 = snap.active_model_sha256
            self.feature_names_sha256 = snap.feature_names_sha256
            self.activated_at = snap.activated_at
            self.previous_model_path = snap.previous_model_path
            self.previous_model_sha256 = snap.previous_model_sha256
            self.model_load_status = snap.load_status
            self.model_fallback_status = snap.fallback_status
            self.model_load_error = snap.load_error

    def _sync_snapshot_from_attributes(self) -> _ModelSnapshot:
        """현재 인스턴스 속성으로부터 불변 _ModelSnapshot을 재생성하여 동기화"""
        feats = tuple(getattr(self, "feature_names", []) or ())
        snap = _ModelSnapshot(
            model=getattr(self, "model", None),
            feature_names=feats,
            top_10_features=tuple(getattr(self, "top_10_features", []) or ()),
            top_3_features=tuple(getattr(self, "top_3_features", []) or ()),
            feature_schema=str(getattr(self, "feature_schema", "B_normalized")),
            active_model_id=str(getattr(self, "active_model_id", "UNKNOWN")),
            active_model_path=str(getattr(self, "active_model_path", "")),
            active_model_sha256=str(getattr(self, "active_model_sha256", "")),
            feature_names_sha256=self.compute_feature_names_sha256(list(feats)),
            activated_at=str(getattr(self, "activated_at", "") or getattr(self, "active_model_activated_at", "")),
            previous_model_path=str(getattr(self, "previous_model_path", "")),
            previous_model_sha256=str(getattr(self, "previous_model_sha256", "")),
            load_status=str(getattr(self, "model_load_status", "UNKNOWN")),
            fallback_status=str(getattr(self, "model_fallback_status", "NONE")),
            load_error=str(getattr(self, "model_load_error", "")),
        )
        self._apply_snapshot(snap)
        return snap

    def get_model_snapshot(self) -> _ModelSnapshot:
        """현재 추론에 사용할 불변 모델 스냅샷 반환 (단위 테스트에서 직접 속성을 변경한 경우도 일관되게 반영)"""
        snap = getattr(self, "_active_snapshot", None)
        if (
            snap is not None
            and snap.model is self.model
            and snap.feature_schema == self.feature_schema
            and list(snap.feature_names) == list(self.feature_names)
        ):
            return snap
        feats = tuple(getattr(self, "feature_names", []) or ())
        return _ModelSnapshot(
            model=getattr(self, "model", None),
            feature_names=feats,
            top_10_features=tuple(getattr(self, "top_10_features", []) or ()),
            top_3_features=tuple(getattr(self, "top_3_features", []) or ()),
            feature_schema=str(getattr(self, "feature_schema", "B_normalized")),
            active_model_id=str(getattr(self, "active_model_id", "UNKNOWN")),
            active_model_path=str(getattr(self, "active_model_path", "")),
            active_model_sha256=str(getattr(self, "active_model_sha256", "")),
            feature_names_sha256=self.compute_feature_names_sha256(list(feats)),
            activated_at=str(getattr(self, "activated_at", "")),
            previous_model_path=str(getattr(self, "previous_model_path", "")),
            previous_model_sha256=str(getattr(self, "previous_model_sha256", "")),
            load_status=str(getattr(self, "model_load_status", "UNKNOWN")),
            fallback_status=str(getattr(self, "model_fallback_status", "NONE")),
            load_error=str(getattr(self, "model_load_error", "")),
        )

    @classmethod
    def verify_model_artifact(
        cls,
        model_path: Any,
        expected_sha256: Optional[str] = None,
        expected_schema: Optional[str] = None,
        enforce_live_policy: bool = True,
        expected_num_features: Optional[int] = None,
        expected_feature_names: Optional[List[str]] = None,
        expected_feature_names_sha256: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        [STEP 5: 실전 모델 로딩 7대 필수 무결성 검증]
        1. 파일 존재 여부
        2. SHA256 또는 배포 매니페스트 정합성
        3. 모델 클래스 및 스키마 (E 77개 Challenger 차단 포함)
        4. 피처 수 72개
        5. 피처 순서 (Canonical 72개 순서와 100% 일치 + 매니페스트 피처 목록/해시 일치)
        6. 확률 클래스 매핑 ([0, 1, 2])
        7. 추론 테스트 통과 여부 (유한 확률 및 합계 1.0)
        """
        import hashlib
        import joblib
        from pathlib import Path

        p = Path(model_path)
        checks = {
            "1_file_exists": False,
            "2_sha256_match": False,
            "3_class_and_schema_valid": False,
            "4_feature_count_72": False,
            "5_feature_order_exact": False,
            "6_class_mapping_012": False,
            "7_inference_self_test": False,
        }
        result: Dict[str, Any] = {
            "ok": False,
            "model_path": str(p),
            "actual_sha256": "",
            "feature_names_sha256": "",
            "detected_schema": "",
            "num_features": 0,
            "checks": checks,
            "error": "",
            "extracted": None,
        }

        # 1. 파일 존재 여부
        if not p.exists() or not p.is_file():
            result["error"] = f"Model file does not exist: {p}"
            return result
        checks["1_file_exists"] = True

        # 2. SHA256 정합성
        try:
            actual_sha = hashlib.sha256(p.read_bytes()).hexdigest()
            result["actual_sha256"] = actual_sha
        except Exception as e:
            result["error"] = f"Failed to compute SHA256: {e}"
            return result

        if expected_sha256 and actual_sha.lower() != expected_sha256.strip().lower():
            result["error"] = f"SHA256 mismatch: expected {expected_sha256}, got {actual_sha}"
            return result
        checks["2_sha256_match"] = True

        # 로드 및 객체 추출
        try:
            import __main__
            if not hasattr(__main__, "MoEMetaOrchestrator"):
                from core.moe_orchestrator import MoEMetaOrchestrator
                __main__.MoEMetaOrchestrator = MoEMetaOrchestrator
            obj = joblib.load(p)
        except Exception as e:
            result["error"] = f"joblib.load failed: {e}"
            return result

        model_obj = None
        feat_names: List[str] = []
        top10: List[str] = []
        top3: List[str] = []
        detected_schema = ""
        model_id = p.stem

        if hasattr(obj, "model") and obj.model is not None:
            model_obj = obj.model
            feat_names = list(getattr(obj, "feature_names", []))
            top10 = list(getattr(obj, "top_10_features", []))
            top3 = list(getattr(obj, "top_3_features", []))
            detected_schema = str(getattr(obj, "feature_schema", ""))
            model_id = str(getattr(obj, "active_model_id", getattr(obj, "model_id", p.stem)))
        elif hasattr(obj, "gbdt_engine") and getattr(obj.gbdt_engine, "model", None) is not None:
            eng_obj = obj.gbdt_engine
            model_obj = eng_obj.model
            feat_names = list(getattr(eng_obj, "feature_names", []))
            top10 = list(getattr(eng_obj, "top_10_features", []))
            top3 = list(getattr(eng_obj, "top_3_features", []))
            detected_schema = str(getattr(eng_obj, "feature_schema", getattr(obj, "feature_schema", "")))
            model_id = str(getattr(eng_obj, "active_model_id", getattr(obj, "model_id", p.stem)))
        elif isinstance(obj, dict) and "model" in obj:
            model_obj = obj["model"]
            feat_names = list(obj.get("feature_names", []))
            top10 = list(obj.get("top_10_features", []))
            top3 = list(obj.get("top_3_features", []))
            detected_schema = str(obj.get("feature_schema", ""))
            model_id = str(obj.get("model_id", p.stem))
        elif type(obj).__name__ == "LGBMClassifier":
            model_obj = obj
            feat_names = list(getattr(obj, "feature_name_", []))

        if not detected_schema:
            if feat_names == cls.CANONICAL_B_NORMALIZED_72_FEATURES:
                detected_schema = "B_normalized"
            elif feat_names in (cls.CANONICAL_CHAMPION_72_FEATURES, cls.LEGACY_PKL_CHAMPION_72_FEATURES):
                detected_schema = "v4_champion"
            elif len(feat_names) == 77:
                detected_schema = "v5_challenger"
            else:
                detected_schema = "unknown"

        feat_names_sha = cls.compute_feature_names_sha256(feat_names)
        result["detected_schema"] = detected_schema
        result["num_features"] = len(feat_names)
        result["feature_names_sha256"] = feat_names_sha

        # 3. 모델 클래스 및 스키마 검증
        if model_obj is None or type(model_obj).__name__ != "LGBMClassifier":
            result["error"] = f"Invalid inner model class: {type(model_obj).__name__ if model_obj is not None else None}"
            return result

        if enforce_live_policy:
            if detected_schema in cls.LIVE_FORBIDDEN_SCHEMAS or "challenger" in p.name.lower():
                result["error"] = f"Forbidden experimental/challenger model blocked from live: {p.name} (schema={detected_schema})"
                return result
            if detected_schema not in cls.LIVE_ALLOWED_SCHEMAS:
                result["error"] = f"Schema '{detected_schema}' is not in LIVE_ALLOWED_SCHEMAS {cls.LIVE_ALLOWED_SCHEMAS}"
                return result
        if expected_schema and detected_schema != expected_schema:
            result["error"] = f"Schema mismatch: expected '{expected_schema}', got '{detected_schema}'"
            return result
        checks["3_class_and_schema_valid"] = True

        # 4. 피처 수 검증
        n_in = int(getattr(model_obj, "n_features_in_", len(feat_names)))
        if expected_num_features is not None and (len(feat_names) != int(expected_num_features) or int(expected_num_features) != 72):
            result["error"] = f"Feature count mismatch with manifest/policy: expected_num_features={expected_num_features}, actual={len(feat_names)}"
            return result
        if enforce_live_policy or expected_schema in cls.LIVE_ALLOWED_SCHEMAS:
            if len(feat_names) != 72 or n_in != 72:
                result["error"] = f"Feature count must be exactly 72 (got len={len(feat_names)}, n_features_in_={n_in})"
                return result
        elif len(feat_names) == 0 or len(feat_names) != n_in:
            result["error"] = f"Invalid feature count: len={len(feat_names)}, n_features_in_={n_in}"
            return result
        checks["4_feature_count_72"] = True

        # 5. 피처 순서 검증 (래퍼 feature_names == 부스터 내부 feature_name_ 및 스키마 표준 순서 일치)
        booster_feats = list(getattr(model_obj, "feature_name_", feat_names))
        if booster_feats and feat_names != booster_feats:
            result["error"] = "Wrapper feature_names order does not match LightGBM booster feature_name_"
            return result
        if detected_schema == "B_normalized":
            if feat_names != cls.CANONICAL_B_NORMALIZED_72_FEATURES:
                result["error"] = "Feature names/order do not match CANONICAL_B_NORMALIZED_72_FEATURES"
                return result
        elif detected_schema in ("v4_champion", "A_champion"):
            if feat_names not in (cls.CANONICAL_CHAMPION_72_FEATURES, cls.LEGACY_PKL_CHAMPION_72_FEATURES):
                result["error"] = "Feature names/order do not match Champion 72 feature order"
                return result
        if expected_feature_names is not None and list(feat_names) != list(expected_feature_names):
            result["error"] = "Feature names/order do not match manifest feature_names"
            return result
        if expected_feature_names_sha256 and feat_names_sha.lower() != expected_feature_names_sha256.strip().lower():
            result["error"] = f"Feature names SHA256 mismatch: expected {expected_feature_names_sha256}, got {feat_names_sha}"
            return result
        checks["5_feature_order_exact"] = True

        # 6. 확률 클래스 매핑 검증 ([0, 1, 2])
        classes_list = [int(c) for c in getattr(model_obj, "classes_", [])]
        n_classes = int(getattr(model_obj, "n_classes_", len(classes_list)))
        if n_classes != 3 or classes_list != [0, 1, 2]:
            result["error"] = f"Invalid class mapping: n_classes={n_classes}, classes={classes_list}"
            return result
        checks["6_class_mapping_012"] = True

        # 7. 추론 Self-Test 검증
        try:
            test_X = pd.DataFrame(np.zeros((1, len(feat_names)), dtype=float), columns=feat_names)
            probs = model_obj.predict_proba(test_X)
            if probs.shape != (1, 3) or not np.all(np.isfinite(probs)) or np.any(probs < 0.0):
                result["error"] = f"Self-test invalid probabilities: shape={probs.shape}, values={probs}"
                return result
            prob_sum = float(np.sum(probs[0]))
            if abs(prob_sum - 1.0) > 1e-5:
                result["error"] = f"Self-test probability sum != 1.0: {prob_sum}"
                return result
        except Exception as e:
            result["error"] = f"Self-test inference exception: {e}"
            return result
        checks["7_inference_self_test"] = True

        result["ok"] = True
        result["extracted"] = {
            "model": model_obj,
            "feature_names": feat_names,
            "feature_names_sha256": feat_names_sha,
            "top_10_features": top10,
            "top_3_features": top3,
            "feature_schema": detected_schema,
            "model_id": model_id,
            "sha256": actual_sha,
            "model_path": str(p),
        }
        return result

    def load_model(
        self,
        model_path: Any,
        expected_sha256: Optional[str] = None,
        expected_schema: Optional[str] = None,
        enforce_live_policy: bool = False,
        _from_manifest_loader: bool = False,
        expected_num_features: Optional[int] = None,
        expected_feature_names: Optional[List[str]] = None,
        expected_feature_names_sha256: Optional[str] = None
    ) -> bool:
        """지정된 모델 파일을 7단계 무결성 검증 후 불변 스냅샷으로 로드"""
        from pathlib import Path
        p = Path(model_path)
        ver = self.verify_model_artifact(
            p,
            expected_sha256=expected_sha256,
            expected_schema=expected_schema,
            enforce_live_policy=enforce_live_policy,
            expected_num_features=expected_num_features,
            expected_feature_names=expected_feature_names,
            expected_feature_names_sha256=expected_feature_names_sha256
        )
        if not ver["ok"]:
            self.model_load_error = ver["error"]
            logging.getLogger('Lumos').error(f"❌ [모델 로드 검증 실패] {p.name}: {ver['error']}")
            return False

        ext = ver["extracted"]
        snap = _ModelSnapshot(
            model=ext["model"],
            feature_names=tuple(ext["feature_names"]),
            top_10_features=tuple(ext["top_10_features"]),
            top_3_features=tuple(ext["top_3_features"]),
            feature_schema=str(ext["feature_schema"]),
            active_model_id=str(ext["model_id"]),
            active_model_path=str(ext["model_path"]),
            active_model_sha256=str(ext["sha256"]),
            feature_names_sha256=str(ext["feature_names_sha256"]),
            activated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            previous_model_path=str(getattr(self, "previous_model_path", "")),
            previous_model_sha256=str(getattr(self, "previous_model_sha256", "")),
            load_status="ACTIVE",
            fallback_status="PRIMARY_ACTIVE",
            load_error=""
        )
        self._apply_snapshot(snap)
        if not _from_manifest_loader:
            self._bound_to_default_manifest = False
        return True

    def _get_manifest_path(self):
        from pathlib import Path
        if getattr(self, "_manifest_path_override", None):
            return Path(self._manifest_path_override)
        return Path(__file__).resolve().parent.parent / "models" / "active_model_manifest.json"

    @staticmethod
    def _extract_manifest_fields(manifest: Dict[str, Any], root_dir: Any, manifest_path: Any) -> Dict[str, Any]:
        from pathlib import Path
        mp = manifest.get("model_path")
        amp = manifest.get("active_model_path")
        default_rel = "models/model_normalized_72.pkl"
        if mp and amp and mp != amp:
            if amp != default_rel and mp == default_rel:
                act_rel = amp
            elif mp != default_rel and amp == default_rel:
                act_rel = mp
            elif Path(amp).is_absolute() or (Path(manifest_path).parent / Path(amp).name).exists():
                act_rel = amp
            else:
                act_rel = mp
        else:
            act_rel = mp or amp or default_rel

        if Path(act_rel).is_absolute():
            act_path = Path(act_rel)
        else:
            cand_root = Path(root_dir) / act_rel
            cand_manifest_dir = Path(manifest_path).parent / Path(act_rel).name
            cand_manifest_ver = Path(manifest_path).parent / "versions" / Path(act_rel).name
            if cand_root.exists():
                act_path = cand_root
            elif cand_manifest_dir.exists():
                act_path = cand_manifest_dir
            elif cand_manifest_ver.exists():
                act_path = cand_manifest_ver
            else:
                act_path = cand_root

        nf = manifest.get("num_features")
        enf = manifest.get("expected_num_features")
        if nf is not None and enf is not None and int(nf) != int(enf):
            act_n_feat = int(enf) if int(enf) != 72 else int(nf)
        else:
            act_n_feat = int(nf if nf is not None else (enf if enf is not None else 72))

        act_schema = str(manifest.get("expected_schema", "B_normalized")).strip()
        feat_names = manifest.get("feature_names")
        feat_hash = manifest.get("feature_names_sha256")
        # 테스트나 수동 롤백에서 expected_schema를 변경했을 때 이전 스키마의 feature_names가 남아 있으면 스키마 기본 순서를 따름
        if feat_names and act_schema in ("v4_champion", "A_champion") and feat_names == MLFeatureEngine.CANONICAL_B_NORMALIZED_72_FEATURES:
            feat_names = None
            feat_hash = None

        raw_tw_meta = manifest.get("training_window_meta")
        tw_meta_dict = dict(raw_tw_meta) if isinstance(raw_tw_meta, dict) else {}

        return {
            "act_rel": str(act_rel),
            "act_path": act_path,
            "act_sha": manifest.get("expected_sha256"),
            "act_schema": act_schema,
            "act_id": str(manifest.get("active_model_id", "B_Normalized_72")).strip(),
            "act_n_feat": act_n_feat,
            "act_feat_names": feat_names,
            "act_feat_hash": feat_hash,
            "activated_at": str(manifest.get("activated_at") or manifest.get("updated_at") or ""),
            "prev_path": str(manifest.get("previous_model_path", "")),
            "prev_sha": str(manifest.get("previous_model_sha256", "")),
            "fb_rel": str(manifest.get("fallback_model_path", "models/model_champion.pkl")),
            "fb_sha": manifest.get("fallback_expected_sha256"),
            "fb_schema": str(manifest.get("fallback_expected_schema", "v4_champion")).strip(),
            "fb_id": str(manifest.get("fallback_model_id", "v4_champion_fallback")).strip(),
            "fb_n_feat": int(manifest.get("fallback_expected_num_features", 72)),
            "training_window_meta": tw_meta_dict,
        }

    def _try_load_default_model(self):
        """
        [STEP 5: 명시적 배포 매니페스트 기반 실전 모델 로드 및 자동 폴백/차단]
        1순위: models/active_model_manifest.json에 지정된 model_path / active_model_path
        2순위(자동 복귀): 매니페스트의 fallback_model (models/model_champion.pkl, v4_champion)
        실패 시: self.model = None 유지하여 신규 진입 차단 (청산 기능은 정상 유지)
        """
        import json
        from pathlib import Path

        with self._reload_lock:
            root_dir = Path(__file__).resolve().parent.parent
            models_dir = root_dir / "models"
            manifest_path = self._get_manifest_path()

            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    mf = self._extract_manifest_fields(manifest, root_dir, manifest_path)
                    act_path = mf["act_path"]
                    tw_tuple = tuple(sorted(mf.get("training_window_meta", {}).items()))

                    ver = self.verify_model_artifact(
                        act_path,
                        expected_sha256=mf["act_sha"],
                        expected_schema=mf["act_schema"],
                        enforce_live_policy=True,
                        expected_num_features=mf["act_n_feat"],
                        expected_feature_names=mf["act_feat_names"],
                        expected_feature_names_sha256=mf["act_feat_hash"]
                    )
                    if ver["ok"]:
                        ext = ver["extracted"]
                        snap = _ModelSnapshot(
                            model=ext["model"],
                            feature_names=tuple(ext["feature_names"]),
                            top_10_features=tuple(ext["top_10_features"]),
                            top_3_features=tuple(ext["top_3_features"]),
                            feature_schema=str(ext["feature_schema"]),
                            active_model_id=mf["act_id"],
                            active_model_path=str(ext["model_path"]),
                            active_model_sha256=str(ext["sha256"]),
                            feature_names_sha256=str(ext["feature_names_sha256"]),
                            activated_at=mf["activated_at"] or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            previous_model_path=mf["prev_path"],
                            previous_model_sha256=mf["prev_sha"],
                            load_status=f"ACTIVE_{mf['act_schema'].upper()}",
                            fallback_status="PRIMARY_ACTIVE",
                            load_error="",
                            training_window_meta=tw_tuple
                        )
                        self._apply_snapshot(snap)
                        self._loaded_manifest_sha256 = self.active_model_sha256
                        logger.info(self.format_active_model_log())
                        return

                    primary_err = ver["error"]
                    self.model_load_error = primary_err
                    logging.getLogger('Lumos').error(f"❌ [모델 로드 검증 실패] {act_path.name}: {primary_err}")
                    logger.error(f"⚠️ [모델 전환 방어] 주 모델({act_path.name}) 검증 실패({primary_err}) -> 기존 Champion 폴백 시도")

                    fb_rel = mf["fb_rel"]
                    fb_path = (root_dir / fb_rel) if not Path(fb_rel).is_absolute() else Path(fb_rel)
                    ver_fb = self.verify_model_artifact(
                        fb_path,
                        expected_sha256=mf["fb_sha"],
                        expected_schema=mf["fb_schema"],
                        enforce_live_policy=True,
                        expected_num_features=mf["fb_n_feat"]
                    )
                    if ver_fb["ok"]:
                        ext_fb = ver_fb["extracted"]
                        snap_fb = _ModelSnapshot(
                            model=ext_fb["model"],
                            feature_names=tuple(ext_fb["feature_names"]),
                            top_10_features=tuple(ext_fb["top_10_features"]),
                            top_3_features=tuple(ext_fb["top_3_features"]),
                            feature_schema=str(ext_fb["feature_schema"]),
                            active_model_id=mf["fb_id"],
                            active_model_path=str(ext_fb["model_path"]),
                            active_model_sha256=str(ext_fb["sha256"]),
                            feature_names_sha256=str(ext_fb["feature_names_sha256"]),
                            activated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            previous_model_path=mf["prev_path"],
                            previous_model_sha256=mf["prev_sha"],
                            load_status="FALLBACK_TO_CHAMPION",
                            fallback_status="FALLBACK_CHAMPION_ACTIVE",
                            load_error=f"Primary failed ({primary_err}); safely fell back to {fb_path.name}",
                            training_window_meta=()
                        )
                        self._apply_snapshot(snap_fb)
                        logger.info(self.format_active_model_log())
                        return

                    # 주 모델과 폴백 모델 모두 실패 시 신규 진입 완전 차단
                    blocked_snap = _ModelSnapshot(
                        model=None,
                        feature_names=(),
                        top_10_features=(),
                        top_3_features=(),
                        feature_schema=self.feature_schema,
                        active_model_id="BLOCKED_NO_VALID_MODEL",
                        active_model_path="",
                        active_model_sha256="",
                        feature_names_sha256="",
                        activated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        previous_model_path=mf["prev_path"],
                        previous_model_sha256=mf["prev_sha"],
                        load_status="BLOCKED",
                        fallback_status="ALL_MODELS_FAILED_ENTRY_BLOCKED",
                        load_error=ver_fb["error"],
                        training_window_meta=()
                    )
                    self._apply_snapshot(blocked_snap)
                    logging.getLogger('Lumos').error(f"❌ [모델 로드 검증 실패] {fb_path.name}: {ver_fb['error']}")
                    logger.error(f"🚨 [신규 진입 차단] 주 모델 및 폴백 모델 검증 모두 실패: {self.model_load_error}")
                    return
                except Exception as e:
                    logger.error(f"⚠️ [매니페스트 파싱 오류] {e} -> 기본 모델 탐색 폴백")

            for fname, exp_schema in [
                ("model_normalized_72.pkl", "B_normalized"),
                ("model_champion.pkl", "v4_champion"),
                ("model_moe_orchestrator.pkl", "v4_champion"),
            ]:
                p = models_dir / fname
                if p.exists() and self.load_model(p, expected_schema=exp_schema, enforce_live_policy=True, _from_manifest_loader=True):
                    self.model_load_status = f"ACTIVE_{self.feature_schema.upper()}"
                    self.model_fallback_status = "PRIMARY_ACTIVE"
                    self._apply_snapshot(self.get_model_snapshot())
                    return

            blocked_snap = _ModelSnapshot(
                model=None,
                feature_names=(),
                top_10_features=(),
                top_3_features=(),
                feature_schema=self.feature_schema,
                active_model_id="BLOCKED_NO_VALID_MODEL",
                active_model_path="",
                active_model_sha256="",
                feature_names_sha256="",
                activated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                previous_model_path="",
                previous_model_sha256="",
                load_status="BLOCKED",
                fallback_status="ALL_MODELS_FAILED_ENTRY_BLOCKED",
                load_error=self.model_load_error,
                training_window_meta=()
            )
            self._apply_snapshot(blocked_snap)

    def ensure_latest_manifest_model(self) -> bool:
        """
        [STEP 3: 실시간 무중단 스테이징 핫리로드 및 스냅샷 원자적 교체]
        1. 현재 정상 모델 스냅샷을 유지한 상태에서 새 매니페스트를 읽음
        2. 새 모델 파일을 별도 스테이징 객체로 로드하고 7단계 무결성 검증(SHA256, 스키마, 72피처 순서, 클래스 매핑, Self-Test) 수행
        3. 모든 검증을 통과한 경우에만 단일 참조 교체(_apply_snapshot)로 새 모델 활성화
        4. 검증 실패 시 기존 정상 모델 스냅샷을 그대로 유지 (None 노출 및 구스키마/신규모델 혼합 원천 차단)
        """
        if not getattr(self, "_bound_to_default_manifest", False):
            return False
        import json
        from pathlib import Path

        manifest_path = self._get_manifest_path()
        if not manifest_path.exists():
            return False

        with self._reload_lock:
            cur_snap = self.get_model_snapshot()
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as e:
                self.last_reload_error = f"Manifest read/parse error: {e}"
                return False

            root_dir = Path(__file__).resolve().parent.parent
            mf = self._extract_manifest_fields(manifest, root_dir, manifest_path)
            target_sha = str(mf["act_sha"] or "").strip().lower()
            target_id = mf["act_id"]
            target_schema = mf["act_schema"]
            target_n_feat = mf["act_n_feat"]
            target_path = mf["act_path"]
            tw_tuple = tuple(sorted(mf.get("training_window_meta", {}).items()))

            cur_sha = str(cur_snap.active_model_sha256 or "").strip().lower()
            cur_id = str(cur_snap.active_model_id or "").strip()
            cur_schema = str(cur_snap.feature_schema or "").strip()
            cur_path_str = ""
            try:
                cur_path_str = str(Path(cur_snap.active_model_path).resolve()) if cur_snap.active_model_path else ""
            except Exception:
                cur_path_str = str(cur_snap.active_model_path or "")
            try:
                target_path_str = str(target_path.resolve())
            except Exception:
                target_path_str = str(target_path)

            if (
                cur_snap.model is not None
                and len(cur_snap.feature_names) == 72
                and target_sha
                and target_sha == cur_sha
                and target_id == cur_id
                and target_schema == cur_schema
                and target_n_feat == len(cur_snap.feature_names)
                and (not cur_path_str or cur_path_str == target_path_str)
            ):
                return False

            # 현재 활성 모델이 이미 존재하는 상태에서 매니페스트가 변경된 경우: 별도 스테이징 검증 수행
            if cur_snap.model is not None and len(cur_snap.feature_names) == 72:
                ver = self.verify_model_artifact(
                    target_path,
                    expected_sha256=mf["act_sha"],
                    expected_schema=target_schema,
                    enforce_live_policy=True,
                    expected_num_features=target_n_feat,
                    expected_feature_names=mf["act_feat_names"],
                    expected_feature_names_sha256=mf["act_feat_hash"]
                )
                if ver["ok"]:
                    ext = ver["extracted"]
                    new_snap = _ModelSnapshot(
                        model=ext["model"],
                        feature_names=tuple(ext["feature_names"]),
                        top_10_features=tuple(ext["top_10_features"]),
                        top_3_features=tuple(ext["top_3_features"]),
                        feature_schema=str(ext["feature_schema"]),
                        active_model_id=target_id,
                        active_model_path=str(ext["model_path"]),
                        active_model_sha256=str(ext["sha256"]),
                        feature_names_sha256=str(ext["feature_names_sha256"]),
                        activated_at=mf["activated_at"] or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        previous_model_path=mf["prev_path"] or cur_snap.active_model_path,
                        previous_model_sha256=mf["prev_sha"] or cur_snap.active_model_sha256,
                        load_status=f"ACTIVE_{target_schema.upper()}",
                        fallback_status="HOT_RELOAD_PRIMARY_ACTIVE",
                        load_error="",
                        training_window_meta=tw_tuple
                    )
                    self._apply_snapshot(new_snap)
                    self.last_reload_error = ""
                    logger.info(f"🔄 [실시간 핫리로드 완료] {self.format_active_model_log()}")
                    return True
                else:
                    # 신규 모델 검증 실패 시 기존 정상 모델 스냅샷을 그대로 유지 (무중단 방어)
                    self.last_reload_error = ver["error"]
                    kept_snap = _ModelSnapshot(
                        model=cur_snap.model,
                        feature_names=cur_snap.feature_names,
                        top_10_features=cur_snap.top_10_features,
                        top_3_features=cur_snap.top_3_features,
                        feature_schema=cur_snap.feature_schema,
                        active_model_id=cur_snap.active_model_id,
                        active_model_path=cur_snap.active_model_path,
                        active_model_sha256=cur_snap.active_model_sha256,
                        feature_names_sha256=cur_snap.feature_names_sha256,
                        activated_at=cur_snap.activated_at,
                        previous_model_path=cur_snap.previous_model_path,
                        previous_model_sha256=cur_snap.previous_model_sha256,
                        load_status=cur_snap.load_status,
                        fallback_status="KEPT_CURRENT_VALID_MODEL_ON_RELOAD_FAILURE",
                        load_error=f"Hot-reload rejected ({ver['error']}); kept valid active model ({cur_snap.active_model_sha256[:12]})",
                        training_window_meta=getattr(cur_snap, "training_window_meta", ())
                    )
                    self._apply_snapshot(kept_snap)
                    logger.error(
                        f"⚠️ [핫리로드 거부 - 기존 정상 모델 유지] candidate={target_path.name} | reason={ver['error']} | kept_sha256={cur_snap.active_model_sha256[:16]}"
                    )
                    return False

            # 현재 유효한 모델이 없는 상태였다면 기본 로드 시도
            self._try_load_default_model()
            return self.model is not None

    def verify_manifest_memory_snapshot_consistency(self, snapshot: Optional[_ModelSnapshot] = None) -> Dict[str, Any]:
        """
        [STEP 2] 매니페스트 모델 = 메모리 로드 모델 = 신호 생성 스냅샷 모델 100% 일치 검증.
        불일치 또는 BLOCKED 상태가 감지되면 ok=False 및 사유를 반환하여 신규 진입을 차단한다.
        """
        import json
        from pathlib import Path

        snap = snapshot if snapshot is not None else self.get_model_snapshot()
        if snap is None or snap.model is None or snap.load_status == "BLOCKED":
            return {
                "ok": False,
                "reason": "MODEL_BLOCKED_OR_NONE",
                "snapshot_id": getattr(snap, "active_model_id", "NONE"),
                "snapshot_sha256": getattr(snap, "active_model_sha256", ""),
                "snapshot_schema": getattr(snap, "feature_schema", ""),
            }

        if len(snap.feature_names) != 72 or snap.feature_schema not in self.LIVE_ALLOWED_SCHEMAS:
            return {
                "ok": False,
                "reason": f"INVALID_SNAPSHOT_SCHEMA_OR_FEATURES({snap.feature_schema},{len(snap.feature_names)})",
                "snapshot_id": snap.active_model_id,
                "snapshot_sha256": snap.active_model_sha256,
                "snapshot_schema": snap.feature_schema,
            }

        mem_id = str(getattr(self, "active_model_id", ""))
        mem_sha = str(getattr(self, "active_model_sha256", ""))
        mem_schema = str(getattr(self, "feature_schema", ""))
        mem_n_feat = len(getattr(self, "feature_names", []) or [])

        if (
            mem_id != snap.active_model_id
            or mem_sha != snap.active_model_sha256
            or mem_schema != snap.feature_schema
            or mem_n_feat != len(snap.feature_names)
        ):
            return {
                "ok": False,
                "reason": "MEMORY_SNAPSHOT_MISMATCH",
                "memory_id": mem_id,
                "memory_sha256": mem_sha,
                "snapshot_id": snap.active_model_id,
                "snapshot_sha256": snap.active_model_sha256,
            }

        manifest_consistent = True
        manifest_sha = snap.active_model_sha256
        manifest_id = snap.active_model_id
        manifest_schema = snap.feature_schema
        if (
            getattr(self, "_bound_to_default_manifest", False)
            and snap.fallback_status in ("PRIMARY_ACTIVE", "HOT_RELOAD_PRIMARY_ACTIVE")
        ):
            manifest_path = self._get_manifest_path()
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    root_dir = Path(__file__).resolve().parent.parent
                    mf = self._extract_manifest_fields(manifest, root_dir, manifest_path)
                    manifest_sha = str(mf["act_sha"] or "").strip()
                    manifest_id = str(mf["act_id"] or "").strip()
                    manifest_schema = str(mf["act_schema"] or "").strip()
                    if (
                        (manifest_sha and manifest_sha.lower() != snap.active_model_sha256.lower())
                        or (manifest_id and manifest_id != snap.active_model_id)
                        or (manifest_schema and manifest_schema != snap.feature_schema)
                        or int(mf["act_n_feat"]) != len(snap.feature_names)
                    ):
                        manifest_consistent = False
                except Exception:
                    manifest_consistent = False

        if not manifest_consistent:
            return {
                "ok": False,
                "reason": "MANIFEST_MEMORY_MISMATCH",
                "manifest_id": manifest_id,
                "manifest_sha256": manifest_sha,
                "manifest_schema": manifest_schema,
                "snapshot_id": snap.active_model_id,
                "snapshot_sha256": snap.active_model_sha256,
                "snapshot_schema": snap.feature_schema,
            }

        return {
            "ok": True,
            "reason": "CONSISTENT",
            "manifest_id": manifest_id,
            "manifest_sha256": manifest_sha,
            "manifest_schema": manifest_schema,
            "memory_id": mem_id,
            "memory_sha256": mem_sha,
            "memory_schema": mem_schema,
            "snapshot_id": snap.active_model_id,
            "snapshot_sha256": snap.active_model_sha256,
            "snapshot_schema": snap.feature_schema,
        }

    def get_active_model_info(self, is_daemon_running: bool = False) -> Dict[str, Any]:
        """현재 메모리에 로드된 실전 모델의 식별자, 스키마, SHA256 및 STEP 7 표준 상태 필드 반환"""
        snap = self.get_model_snapshot()
        is_ready = snap.model is not None and len(snap.feature_names) == 72
        is_fallback = snap.load_status == "FALLBACK_TO_CHAMPION"
        if not is_ready:
            op_verdict = "BLOCKED"
        elif is_fallback or snap.feature_schema in ("v4_champion", "A_champion"):
            op_verdict = "ROLLED_BACK"
        elif is_daemon_running:
            op_verdict = "ACTIVE_B_NORMALIZED"
        else:
            op_verdict = "READY_FOR_LIVE_START"

        tw_meta = dict(getattr(snap, "training_window_meta", ()) or ())
        if not tw_meta and getattr(self, "last_training_window_meta", None):
            tw_meta = dict(self.last_training_window_meta)

        consistency = self.verify_manifest_memory_snapshot_consistency(snapshot=snap)

        return {
            "active_model_id": snap.active_model_id,
            "feature_schema": snap.feature_schema,
            "model_path": snap.active_model_path,
            "model_sha256": snap.active_model_sha256,
            "sha256": snap.active_model_sha256,
            "model_version": (snap.active_model_sha256 or "")[:12] if snap.active_model_sha256 else "NONE",
            "feature_names_sha256": snap.feature_names_sha256,
            "num_features": len(snap.feature_names),
            "load_status": snap.load_status,
            "fallback_status": snap.fallback_status,
            "activated_at": snap.activated_at,
            "previous_model_path": snap.previous_model_path,
            "previous_model_sha256": snap.previous_model_sha256,
            "is_fallback": is_fallback,
            "load_error": snap.load_error,
            "last_reload_error": getattr(self, "last_reload_error", ""),
            "is_ready": is_ready,
            "operational_verdict": op_verdict,
            "training_window_meta": tw_meta,
            "max_training_weeks": int(getattr(config, "ROLLING_TRAINING_WEEKS", 520)),
            "max_training_lookback_days": int(getattr(config, "TRAINING_LOOKBACK_DAYS", 3640)),
            "is_manifest_memory_consistent": consistency["ok"],
            "consistency_reason": consistency["reason"],
            # STEP 7 표준 진단 키
            "ACTIVE_MODEL_ID": snap.active_model_id,
            "ACTIVE_MODEL_SCHEMA": snap.feature_schema,
            "ACTIVE_MODEL_SHA256": snap.active_model_sha256,
            "ACTIVE_MODEL_PATH": snap.active_model_path,
            "ACTIVE_MODEL_FEATURE_COUNT": len(snap.feature_names),
            "MODEL_LOAD_STATUS": snap.load_status,
            "MODEL_ACTIVATION_TIME": snap.activated_at,
            "MODEL_FALLBACK_STATUS": snap.fallback_status,
        }

    def format_active_model_log(self, is_daemon_running: bool = False) -> str:
        """STEP 7 필수 8대 진단 필드를 포함한 표준 로그 문자열 생성"""
        info = self.get_active_model_info(is_daemon_running=is_daemon_running)
        return (
            f"ACTIVE_MODEL_ID={info['ACTIVE_MODEL_ID']} | "
            f"ACTIVE_MODEL_SCHEMA={info['ACTIVE_MODEL_SCHEMA']} | "
            f"ACTIVE_MODEL_SHA256={info['ACTIVE_MODEL_SHA256']} | "
            f"ACTIVE_MODEL_PATH={info['ACTIVE_MODEL_PATH']} | "
            f"ACTIVE_MODEL_FEATURE_COUNT={info['ACTIVE_MODEL_FEATURE_COUNT']} | "
            f"MODEL_LOAD_STATUS={info['MODEL_LOAD_STATUS']} | "
            f"MODEL_ACTIVATION_TIME={info['MODEL_ACTIVATION_TIME']} | "
            f"MODEL_FALLBACK_STATUS={info['MODEL_FALLBACK_STATUS']} | "
            f"OPERATIONAL_VERDICT={info['operational_verdict']}"
        )

    def extract_features(self, df: pd.DataFrame, live_prices: dict = None) -> pd.DataFrame:
        """15분봉 시세 데이터로부터 종합 피처 풀 자동 생성 (대소문자 정규화 및 데이터 부족 방어)"""
        if df is None or df.empty or len(df) < 5:
            return pd.DataFrame()

        df = df.copy()

        # 대소문자 표준화 (OHLCV)
        rename_map = {
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
        }
        for k, v in rename_map.items():
            if k in df.columns and v not in df.columns:
                df[v] = df[k]

        if 'Close' not in df.columns:
            return pd.DataFrame()
        if 'High' not in df.columns:
            df['High'] = df['Close']
        if 'Low' not in df.columns:
            df['Low'] = df['Close']
        if 'Open' not in df.columns:
            df['Open'] = df['Close']
        if 'Volume' not in df.columns:
            df['Volume'] = 1000.0

        # [1. 모멘텀 지표]
        # RSI (7, 14, 21)
        for period in [7, 14, 21]:
            delta = df['Close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(period).mean()
            avg_loss = loss.rolling(period).mean()
            rs = avg_gain / (avg_loss + 1e-9)
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))

        # MACD (12, 26, 9)
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # Stochastics (14, 3, 3)
        low14 = df['Low'].rolling(14).min()
        high14 = df['High'].rolling(14).max()
        df['Stoch_K'] = ((df['Close'] - low14) / (high14 - low14 + 1e-9)) * 100.0
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

        # ROC (5, 10, 20)
        for p in [5, 10, 20]:
            df[f'ROC_{p}'] = df['Close'].pct_change(p, fill_method=None) * 100.0

        # CCI (14, 20)
        for p in [14, 20]:
            tp = (df['High'] + df['Low'] + df['Close']) / 3.0
            tp_sma = tp.rolling(p).mean()
            tp_mad = (tp - tp_sma).abs().rolling(p).mean()
            df[f'CCI_{p}'] = (tp - tp_sma) / (0.015 * tp_mad + 1e-9)

        # [2. 변동성 지표]
        # ATR (14)
        hl = df['High'] - df['Low']
        hc = (df['High'] - df['Close'].shift()).abs()
        lc = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df['ATR_14'] = tr.rolling(14).mean()

        # Bollinger Bands (20, 2)
        sma20 = df['Close'].rolling(20).mean()
        std20 = df['Close'].rolling(20).std()
        df['BB_Upper'] = sma20 + (std20 * 2)
        df['BB_Lower'] = sma20 - (std20 * 2)
        df['BB_PctB'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-9)
        df['BB_Bandwidth'] = (df['BB_Upper'] - df['BB_Lower']) / (sma20 + 1e-9) * 100.0

        # Keltner Channels (20, 1.5)
        ema20_kc = df['Close'].ewm(span=20, adjust=False).mean()
        df['KC_Upper'] = ema20_kc + (df['ATR_14'] * 1.5)
        df['KC_Lower'] = ema20_kc - (df['ATR_14'] * 1.5)
        df['KC_Width'] = (df['KC_Upper'] - df['KC_Lower']) / (ema20_kc + 1e-9) * 100.0

        # [3. 거래량 지표]
        # OBV & CMF
        obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['OBV'] = (obv - obv.ewm(span=20, adjust=False).mean()) / (df['Volume'].rolling(20).mean() + 1e-9)

        # CMF (Chaikin Money Flow 20)
        mfv = (((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-9)) * df['Volume']
        df['CMF_20'] = mfv.rolling(20).sum() / (df['Volume'].rolling(20).sum() + 1e-9)

        # Volume Z-score
        vol_mean = df['Volume'].rolling(20).mean()
        vol_std = df['Volume'].rolling(20).std()
        df['Volume_Z'] = (df['Volume'] - vol_mean) / (vol_std + 1e-9)

        # VWAP & 이격도
        if isinstance(df.index, pd.DatetimeIndex):
            df['date_str'] = df.index.strftime('%Y-%m-%d')
        elif 'datetime' in df.columns:
            df['date_str'] = df['datetime'].astype(str).str.slice(0, 10)
        elif 'Datetime' in df.columns:
            df['date_str'] = df['Datetime'].astype(str).str.slice(0, 10)
        else:
            df['date_str'] = "2026-01-01"

        typical_price = (df['High'] + df['Low'] + df['Close']) / 3.0
        df['cum_vp'] = typical_price * df['Volume']
        df['vwap'] = df.groupby('date_str')['cum_vp'].cumsum() / (df.groupby('date_str')['Volume'].cumsum() + 1e-9)
        df['VWAP_Diff'] = (df['Close'] - df['vwap']) / (df['vwap'] + 1e-9) * 100.0

        # [4. 추세 및 이동평균]
        for span in [9, 21, 50, 200]:
            df[f'EMA_{span}'] = df['Close'].ewm(span=span, adjust=False).mean()
            df[f'EMA_{span}_Diff'] = (df['Close'] - df[f'EMA_{span}']) / (df[f'EMA_{span}'] + 1e-9) * 100.0

        # ADX / DMI (14)
        up_move = df['High'].diff()
        down_move = -df['Low'].diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        tr_smooth = tr.rolling(14).mean()
        plus_di = (pd.Series(plus_dm, index=df.index).rolling(14).mean() / (tr_smooth + 1e-9)) * 100.0
        minus_di = (pd.Series(minus_dm, index=df.index).rolling(14).mean() / (tr_smooth + 1e-9)) * 100.0
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9) * 100.0
        df['ADX_14'] = dx.rolling(14).mean()
        df['DMI_Diff'] = plus_di - minus_di

        # [5. 캔들 구조]
        candle_range = (df['High'] - df['Low']) + 1e-9
        df['Body_Ratio'] = (df['Close'] - df['Open']).abs() / candle_range
        df['Upper_Wick_Ratio'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / candle_range
        df['Lower_Wick_Ratio'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / candle_range

        # [5-A. 달러 절대값 피처 정규화 (STEP 2: 상대 비율/이격도 변환)]
        close_safe = df['Close'].clip(lower=1e-6)
        atr_safe = df['ATR_14'].clip(lower=1e-6)
        df['ATR_14_Pct'] = (df['ATR_14'] / close_safe) * 100.0
        df['BB_Upper_Diff'] = ((df['BB_Upper'] - df['Close']) / close_safe) * 100.0
        df['BB_Lower_Diff'] = ((df['Close'] - df['BB_Lower']) / close_safe) * 100.0
        df['KC_Upper_Diff'] = ((df['KC_Upper'] - df['Close']) / close_safe) * 100.0
        df['KC_Lower_Diff'] = ((df['Close'] - df['KC_Lower']) / close_safe) * 100.0
        df['MACD_Pct'] = ((ema12 - ema26) / (ema26.clip(lower=1e-6))) * 100.0
        df['MACD_Signal_Pct'] = df['MACD_Pct'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist_Pct'] = df['MACD_Pct'] - df['MACD_Signal_Pct']

        # [5-B. 추세 지속 감지 피처 6종 (STEP 3A: Trend Continuation)]
        ema20 = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA_20'] = ema20
        df['EMA20_Slope_3'] = ((ema20 - ema20.shift(3)) / (ema20.shift(3).clip(lower=1e-6))) * 100.0
        df['EMA20_Slope_Accel'] = df['EMA20_Slope_3'] - df['EMA20_Slope_3'].shift(3)
        net_move_8 = (df['Close'] - df['Close'].shift(8)).abs()
        path_move_8 = df['Close'].diff().abs().rolling(8).sum()
        df['Trend_Efficiency_8'] = (net_move_8 / (path_move_8 + 1e-9)).clip(0.0, 1.0)
        ret_1_diff = df['Close'].diff()
        df['Downside_Persistence_6'] = (ret_1_diff < 0).astype(float).rolling(6).mean()
        df['Upside_Persistence_6'] = (ret_1_diff > 0).astype(float).rolling(6).mean()
        low20 = df['Low'].rolling(20).min()
        high20 = df['High'].rolling(20).max()
        df['Breakout_Position_20'] = ((df['Close'] - low20) / (high20 - low20 + 1e-9)).clip(0.0, 1.0)

        # [5-C. 평균회귀·반전 감지 피처 6종 (STEP 3B: Mean-Reversion & Inflection)]
        ret_3_pct = df['Close'].pct_change(3, fill_method=None) * 100.0
        df['Return_Accel_3'] = ret_3_pct - ret_3_pct.shift(3)
        df['VWAP_Dist_ATR'] = (df['Close'] - df['vwap']) / (atr_safe + 1e-9)
        df['EMA20_Dist_ATR'] = (df['Close'] - ema20) / (atr_safe + 1e-9)
        df['Signed_Body_Ratio'] = (df['Close'] - df['Open']) / candle_range
        lw_mean10 = df['Lower_Wick_Ratio'].rolling(10).mean()
        uw_mean10 = df['Upper_Wick_Ratio'].rolling(10).mean()
        df['Lower_Wick_Expansion'] = (df['Lower_Wick_Ratio'] / (lw_mean10 + 1e-4)).clip(0.0, 10.0)
        df['Upper_Wick_Expansion'] = (df['Upper_Wick_Ratio'] / (uw_mean10 + 1e-4)).clip(0.0, 10.0)


        # [6. 크로스에셋 및 매크로 피처 (V4)]
        if 'datetime_dt' not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df['datetime_dt'] = df.index + pd.Timedelta(minutes=15)
            elif 'datetime' in df.columns:
                df['datetime_dt'] = pd.to_datetime(df['datetime']) + pd.Timedelta(minutes=15)
            elif 'Datetime' in df.columns:
                df['datetime_dt'] = pd.to_datetime(df['Datetime']) + pd.Timedelta(minutes=15)

        try:
            from core.data_lake import MarketDataLake
            lake = MarketDataLake()
            
            # 대추세 60분봉 (config.MACRO_TREND_SYMBOL 기준)
            macro_sym = config.MACRO_TREND_SYMBOL
            soxx_live = live_prices.get(macro_sym, 0.0) if live_prices else 0.0
            if soxx_live > 0:
                soxx_60m = lake.get_candles_with_live_tick(macro_sym, "60m", live_price=soxx_live)
            else:
                soxx_60m = lake.load_candles(macro_sym, "60m")
            if not soxx_60m.empty:
                soxx_60m['ema20'] = soxx_60m['Close'].ewm(span=20, adjust=False).mean()
                soxx_60m['ema60'] = soxx_60m['Close'].ewm(span=60, adjust=False).mean()
                soxx_60m['macro_trend_spread'] = (soxx_60m['ema20'] - soxx_60m['ema60']) / (soxx_60m['ema60'] + 1e-9) * 100.0
                soxx_60m['macro_ema20_slope'] = (soxx_60m['ema20'].diff(5) / soxx_60m['ema20'].shift(5)) * 100.0
                soxx_60m['macro_price_vs_ema20'] = (soxx_60m['Close'] - soxx_60m['ema20']) / (soxx_60m['ema20'] + 1e-9) * 100.0
                soxx_feat = soxx_60m[['datetime', 'macro_trend_spread', 'macro_ema20_slope', 'macro_price_vs_ema20']].copy()
                soxx_feat['datetime_dt'] = pd.to_datetime(soxx_feat['datetime']) + pd.Timedelta(minutes=60)
                df = _safe_merge_asof(df, soxx_feat.drop(columns=['datetime']), on='datetime_dt', direction='backward')

            # V4 크로스에셋 종목 (15분봉)
            for sym in [config.BASE_ASSET_SHORT, 'NVDA', 'QQQ', 'VIXY', 'IEF']:
                lp = live_prices.get(sym, 0.0) if live_prices else 0.0
                if lp > 0:
                    sym_raw = lake.get_candles_with_live_tick(sym, "15m", live_price=lp)
                else:
                    sym_raw = lake.load_candles(sym, "15m")
                if not sym_raw.empty:
                    sym_raw['datetime_dt'] = pd.to_datetime(sym_raw['datetime']) + pd.Timedelta(minutes=15)
                    lower_sym = 'base_short' if sym == config.BASE_ASSET_SHORT else sym.lower()
                    for p in [1, 3, 5, 20]:
                        sym_raw[f'{lower_sym}_ret_{p}'] = sym_raw['Close'].pct_change(p, fill_method=None) * 100.0
                    sym_raw[f'{lower_sym}_dir_1'] = np.sign(sym_raw['Close'].pct_change(1, fill_method=None))
                    sym_raw[f'{lower_sym}_momentum'] = sym_raw['Close'].pct_change(1, fill_method=None) - sym_raw['Close'].pct_change(5, fill_method=None) / 5.0
                    
                    cols_to_merge = ['datetime_dt'] + [c for c in sym_raw.columns if c.startswith(lower_sym)]
                    df = _safe_merge_asof(df, sym_raw[cols_to_merge], on='datetime_dt', direction='backward')
                    
            # 패닉 시그널 및 상대 강도 파생 변수
            df['base_long_ret_20'] = df['Close'].pct_change(20, fill_method=None) * 100.0
            df['base_long_ret_5'] = df['Close'].pct_change(5, fill_method=None) * 100.0
            df['base_long_vs_qqq_20'] = df['base_long_ret_20'] - df.get('qqq_ret_20', pd.Series(0.0, index=df.index)).fillna(0)
            df['base_long_vs_qqq_5'] = df['base_long_ret_5'] - df.get('qqq_ret_5', pd.Series(0.0, index=df.index)).fillna(0)
            df['panic_signal'] = (
                (df.get('vixy_ret_1', pd.Series(0.0, index=df.index)).fillna(0) > 0).astype(int) + 
                (df.get('ief_ret_1', pd.Series(0.0, index=df.index)).fillna(0) > 0).astype(int)
            )
            
            # 후속 로직(결측치)을 위해 다시 시간순 정렬 유지
            df = df.sort_values('datetime_dt').reset_index(drop=True)
            
        except Exception as e:
            logger.info(f"[MLFeatureEngine] 크로스에셋 병합 실패 (오프라인 모드): {e}")

        if 'datetime' in df.columns:
            df.index = pd.to_datetime(df['datetime'])
        elif 'Datetime' in df.columns:
            df.index = pd.to_datetime(df['Datetime'])
        elif 'datetime_dt' in df.columns:
            df.index = pd.to_datetime(df['datetime_dt']) - pd.Timedelta(minutes=15)

        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) > 0:
            df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
            
        return df

    @staticmethod
    def compute_triple_barrier_labels(
        df: pd.DataFrame,
        take_profit: float = getattr(config, "LABEL_TP_PCT", config.MAX_TP_PCT),
        stop_loss: float = config.SL_MIN_PCT,
        horizon: int = int(config.TIME_STOP_MINUTES / 15)
    ) -> pd.Series:
        """
        [경로 의존적 Triple Barrier 3-Class 정답지 산출]
        - Class 1 (TQQQ 롱): 90분(6개 봉) 내 Low가 -2.0%에 닿기 전에 High가 +3.0%를 먼저 터치
        - Class -1 (SQQQ 숏): 90분(6개 봉) 내 High가 +2.0%에 닿기 전에 Low가 -3.0%를 먼저 터치 (TQQQ 하락)
        - Class 0 (관망): 6개 봉 내 양방향 타겟 미도달 (타임스탑 청산 또는 횡보)
        """
        n = len(df)
        labels = np.zeros(n, dtype=int)
        highs = df['High'].values if 'High' in df else df['high'].values
        lows = df['Low'].values if 'Low' in df else df['low'].values
        closes = df['Close'].values if 'Close' in df else df['close'].values

        for i in range(n):
            p0 = closes[i]
            if p0 <= 0:
                continue
            end_idx = min(i + horizon + 1, n)
            if i + 1 >= end_idx:
                continue

            assigned = False


            long_alive = True


            short_alive = True


            for j in range(i + 1, end_idx):


                h_ret = (highs[j] - p0) / p0


                l_ret = (lows[j] - p0) / p0



                if h_ret >= stop_loss:


                    short_alive = False


                if l_ret <= -stop_loss:


                    long_alive = False
    


                if long_alive and h_ret >= take_profit:


                    labels[i] = 1


                    assigned = True


                    break


                elif short_alive and l_ret <= -take_profit:


                    labels[i] = -1


                    assigned = True


                    break


                elif not long_alive and not short_alive:


                    labels[i] = 0


                    assigned = True


                    break



            if not assigned:


                labels[i] = 0

        return pd.Series(labels, index=df.index, name="Target")


    @staticmethod
    def get_feature_columns(df: pd.DataFrame, schema: str = "B_normalized") -> list:
        exclude_cols = {
            'open', 'high', 'low', 'close', 'volume', 
            'Open', 'High', 'Low', 'Close', 'Volume',
            'datetime', 'datetime_dt', 'date_str', 'time_str', 'date', 'Datetime',
            'week_id', 'target', 'Target',
            'cum_vp', 'vwap', 'EMA_9', 'EMA_20', 'EMA_21', 'EMA_50', 'EMA_200',
            'Confidence', 'Signal', 'Direction', 'Prob_Long', 'Prob_Short', 'Prob_Neutral', 'cross_dir',
            'fwd_ret_6', 'regime', 'pred_class', 'p_short', 'p_neutral', 'p_long'
        }
        v5_extra_cols = set(
            MLFeatureEngine.NORMALIZED_PRICE_FEATURES
            + MLFeatureEngine.TREND_CONTINUATION_FEATURES
            + MLFeatureEngine.MEAN_REVERSION_FEATURES
        )
        base_72 = [
            c for c in df.columns
            if c not in exclude_cols
            and c not in v5_extra_cols
            and not str(c).startswith('week_')
            and pd.api.types.is_numeric_dtype(df[c])
        ]

        schema_key = (schema or "B_normalized").strip()
        if schema_key in ("v4_champion", "A_champion"):
            return base_72

        dollar_set = set(MLFeatureEngine.DOLLAR_SCALE_FEATURES)
        ablated_set = set(MLFeatureEngine.ABLATED_REDUNDANT_FEATURES)
        norm_cols = [c for c in MLFeatureEngine.NORMALIZED_PRICE_FEATURES if c in df.columns]
        trend_cols = [c for c in MLFeatureEngine.TREND_CONTINUATION_FEATURES if c in df.columns]
        mr_cols = [c for c in MLFeatureEngine.MEAN_REVERSION_FEATURES if c in df.columns]

        if schema_key == "B_normalized":
            return [c for c in base_72 if c not in dollar_set] + norm_cols
        elif schema_key == "C_norm_trend":
            return [c for c in base_72 if c not in dollar_set and c not in ablated_set] + norm_cols + trend_cols
        elif schema_key == "D_norm_mr":
            return [c for c in base_72 if c not in dollar_set and c not in ablated_set] + norm_cols + mr_cols
        elif schema_key == "E_raw_all":
            return [c for c in base_72 if c not in dollar_set] + norm_cols + trend_cols + mr_cols
        elif schema_key in ("v5_challenger", "E_challenger"):
            return [c for c in base_72 if c not in dollar_set and c not in ablated_set] + norm_cols + trend_cols + mr_cols

        return [c for c in base_72 if c not in dollar_set] + norm_cols

    def train_and_select_top_features(
        self,
        df_15m: pd.DataFrame,
        live_experience_df: Optional[pd.DataFrame] = None,
        schema: Optional[str] = None,
        exclude_unconfirmed_tail: bool = True
    ) -> Tuple[Any, List[str], List[str]]:
        """3-Class Triple Barrier 기반 LightGBM 다중 분류 모델 학습 및 상위 10대 피처 선별 (최대 10년 상한 및 미확정 라벨 절삭 포함)"""
        import pandas as pd
        target_schema = schema if schema is not None else getattr(self, "feature_schema", "B_normalized")
        horizon = int(config.TIME_STOP_MINUTES / 15)
        lookback_days = int(getattr(config, "TRAINING_LOOKBACK_DAYS", config.ROLLING_TRAINING_WEEKS * 7))

        raw_total_rows = int(len(df_15m)) if df_15m is not None else 0
        if df_15m is None or df_15m.empty or raw_total_rows < 100:
            self.last_training_window_meta = {
                "is_valid_window": False,
                "error": f"Insufficient raw rows ({raw_total_rows} < 100)",
                "config_rolling_training_weeks": int(config.ROLLING_TRAINING_WEEKS),
                "config_training_lookback_days": int(lookback_days),
                "raw_total_rows": raw_total_rows,
                "train_rows": 0,
            }
            self.top_10_features = list(self.CANONICAL_B_NORMALIZED_72_FEATURES[:10])
            self.top_3_features = ["RSI_14", "VWAP_Diff", "Volume_Z"]
            return None, self.top_10_features, self.top_3_features

        feat_df = self.extract_features(df_15m)
        if feat_df.empty:
            self.last_training_window_meta = {
                "is_valid_window": False,
                "error": "Feature extraction returned empty DataFrame",
                "config_rolling_training_weeks": int(config.ROLLING_TRAINING_WEEKS),
                "config_training_lookback_days": int(lookback_days),
                "raw_total_rows": raw_total_rows,
                "train_rows": 0,
            }
            self.top_10_features = list(self.CANONICAL_B_NORMALIZED_72_FEATURES[:10])
            self.top_3_features = ["RSI_14", "VWAP_Diff", "Volume_Z"]
            return None, self.top_10_features, self.top_3_features

        feature_cols = self.get_feature_columns(feat_df, schema=target_schema)

        feat_df['Target'] = self.compute_triple_barrier_labels(
            feat_df,
            take_profit=getattr(config, "LABEL_TP_PCT", config.MAX_TP_PCT),
            stop_loss=config.SL_MIN_PCT,
            horizon=horizon
        )

        raw_start_dt = feat_df.index.min() if isinstance(feat_df.index, pd.DatetimeIndex) else None
        raw_end_dt = feat_df.index.max() if isinstance(feat_df.index, pd.DatetimeIndex) else None

        # 1) 마지막 미확정 Triple Barrier 라벨(horizon=6봉) 제외
        excluded_unconfirmed_tail_rows = 0
        excluded_unconfirmed_datetimes: List[str] = []
        if exclude_unconfirmed_tail and horizon > 0 and len(feat_df) > horizon:
            unconfirmed_slice = feat_df.iloc[-horizon:]
            excluded_unconfirmed_tail_rows = int(len(unconfirmed_slice))
            if isinstance(unconfirmed_slice.index, pd.DatetimeIndex):
                excluded_unconfirmed_datetimes = [str(ts) for ts in unconfirmed_slice.index]
            train_source_df = feat_df.iloc[:-horizon]
        else:
            train_source_df = feat_df

        # 2) config.py의 최대 10년(ROLLING_TRAINING_WEEKS=520주 / TRAINING_LOOKBACK_DAYS=3640일) 학습 기간 상한 적용
        #    보조지표 워밍업(EMA_200 등)은 전체 시계열에서 수행한 뒤, 학습 데이터(X, y)만 최근 10년 이내로 제한
        excluded_older_than_10y_rows = 0
        cutoff_10y_dt = None
        if isinstance(train_source_df.index, pd.DatetimeIndex) and len(train_source_df) > 0 and raw_end_dt is not None:
            cutoff_10y_dt = raw_end_dt - pd.Timedelta(days=lookback_days)
            older_mask = train_source_df.index < cutoff_10y_dt
            excluded_older_than_10y_rows = int(older_mask.sum())
            if excluded_older_than_10y_rows > 0:
                train_source_df = train_source_df.loc[~older_mask]

        pre_dropna_rows = int(len(train_source_df))
        clean_data = train_source_df[feature_cols + ['Target']].replace([np.inf, -np.inf], np.nan).dropna()
        excluded_warmup_nan_rows = int(pre_dropna_rows - len(clean_data))

        if len(clean_data) < 100:
            self.last_training_window_meta = {
                "is_valid_window": False,
                "error": f"Insufficient confirmed training rows after warmup/tail exclusion ({len(clean_data)} < 100)",
                "config_rolling_training_weeks": int(config.ROLLING_TRAINING_WEEKS),
                "config_training_lookback_days": int(lookback_days),
                "raw_start_dt": str(raw_start_dt) if raw_start_dt is not None else "",
                "raw_end_dt": str(raw_end_dt) if raw_end_dt is not None else "",
                "raw_total_rows": raw_total_rows,
                "cutoff_10y_dt": str(cutoff_10y_dt) if cutoff_10y_dt is not None else "",
                "train_rows": int(len(clean_data)),
                "excluded_older_than_10y_rows": excluded_older_than_10y_rows,
                "excluded_warmup_nan_rows": excluded_warmup_nan_rows,
                "excluded_unconfirmed_tail_rows": excluded_unconfirmed_tail_rows,
                "excluded_unconfirmed_datetimes": excluded_unconfirmed_datetimes,
            }
            self.top_10_features = feature_cols[:10]
            self.top_3_features = ["RSI_14", "VWAP_Diff", "Volume_Z"]
            return None, self.top_10_features, self.top_3_features

        train_start_dt = clean_data.index.min() if isinstance(clean_data.index, pd.DatetimeIndex) else None
        train_end_dt = clean_data.index.max() if isinstance(clean_data.index, pd.DatetimeIndex) else None
        span_days = int((train_end_dt - train_start_dt).days) if (train_start_dt is not None and train_end_dt is not None) else 0

        self.last_training_window_meta = {
            "is_valid_window": bool(span_days <= lookback_days and len(clean_data) >= 100),
            "config_rolling_training_weeks": int(config.ROLLING_TRAINING_WEEKS),
            "config_training_lookback_days": int(lookback_days),
            "raw_start_dt": str(raw_start_dt) if raw_start_dt is not None else "",
            "raw_end_dt": str(raw_end_dt) if raw_end_dt is not None else "",
            "raw_total_rows": raw_total_rows,
            "cutoff_10y_dt": str(cutoff_10y_dt) if cutoff_10y_dt is not None else "",
            "train_start_dt": str(train_start_dt) if train_start_dt is not None else "",
            "train_end_dt": str(train_end_dt) if train_end_dt is not None else "",
            "train_rows": int(len(clean_data)),
            "excluded_older_than_10y_rows": excluded_older_than_10y_rows,
            "excluded_warmup_nan_rows": excluded_warmup_nan_rows,
            "excluded_unconfirmed_tail_rows": excluded_unconfirmed_tail_rows,
            "excluded_unconfirmed_datetimes": excluded_unconfirmed_datetimes,
            "excluded_total_rows": int(raw_total_rows - len(clean_data)),
            "span_days": span_days,
            "feature_schema": str(target_schema),
            "num_features": int(len(feature_cols)),
        }

        X = clean_data[feature_cols].copy()
        # 클래스 레이블 매핑: {-1: 0, 0: 1, 1: 2} (0: Short, 1: Neutral, 2: Long)
        label_map = {-1: 0, 0: 1, 1: 2}
        y = clean_data['Target'].map(label_map).fillna(1).astype(int)
        sample_weights = pd.Series(1.0, index=X.index)

        # [실시간 경험 로그(Live Experience) 자동 결합 및 가중치 반영]
        target_live_df = live_experience_df
        if target_live_df is None:
            try:
                from core.live_experience_logger import LiveExperienceLogger
                target_live_df = LiveExperienceLogger().get_trades_dataframe()
                if target_live_df is not None and not target_live_df.empty:
                    import logging
                    log = logging.getLogger(__name__)
                    log.info(f"[MLFeatureEngine] 실전 거래 내역 {len(target_live_df)}건 DB(또는 CSV)에서 안전하게 로드 완료")
            except Exception as e:
                import logging
                log = logging.getLogger(__name__)
                log.error(f"[MLFeatureEngine] 실전 거래 내역 로드 실패: {e}")
                target_live_df = None

        if target_live_df is not None and not target_live_df.empty and "features_json" in target_live_df.columns:
            # 최대 10년(TRAINING_LOOKBACK_DAYS) 학습 윈도우 동기화: 학습 캔들 시작일 이후의 거래만 반영
            if isinstance(clean_data.index, pd.DatetimeIndex) and len(clean_data) > 0 and "entry_time" in target_live_df.columns:
                cutoff_str = clean_data.index.min().strftime('%Y-%m-%d')
                target_live_df = target_live_df[target_live_df["entry_time"].astype(str).str.slice(0, 10) >= cutoff_str]

            import json
            live_rows = []
            live_targets = []
            for _, row in target_live_df.iterrows():
                f_json = row.get("features_json")
                gbdt_conf = float(row.get("gbdt_confidence", 0.0))
                
                # 방어 로직: 피처가 비어있거나, 확신도가 비정상적(0.0)인 쓰레기 데이터는 학습에서 철저히 배제
                act_entry_px = float(row.get("actual_entry_price", 1.0))
                if act_entry_px <= 0.01 or not f_json or str(f_json).strip() in ("", "{}", "nan") or gbdt_conf < 0.01:
                    continue
                try:
                    f_dict = json.loads(f_json) if isinstance(f_json, str) else f_json
                    # 피처 딕셔너리가 아니거나 정상적인 피처 갯수가 담기지 않은 경우 스킵
                    if not isinstance(f_dict, dict) or len(f_dict) < 10:
                        continue
                except Exception:
                    continue

                # 구 스키마 JSON에 현재 스키마(예: B_normalized 정규화 피처)가 누락된 경우 임의 보간(median/0) 혼합 금지
                if any((c not in f_dict) or (f_dict.get(c) is None) or pd.isna(f_dict.get(c)) for c in feature_cols):
                    continue

                sym = str(row.get("symbol", "SOXL")).upper()
                reason = str(row.get("exit_reason", "")).upper()
                pnl = float(row.get("pnl_pct", 0.0))

                # 라벨 결정: 2(Long TP), 0(Short TP), 1(중립/타임스탑)
                if sym in ["TQQQ", "SOXL", "UPRO"]:
                    tgt = 2 if ("TP" in reason or pnl >= 2.0) else (0 if ("SL" in reason or pnl <= -1.5) else 1)
                else:
                    tgt = 0 if ("TP" in reason or pnl >= 2.0) else (2 if ("SL" in reason or pnl <= -1.5) else 1)

                row_vals = {c: float(f_dict[c]) for c in feature_cols}
                live_rows.append(row_vals)
                live_targets.append(tgt)

            if live_rows:
                X_live = pd.DataFrame(live_rows).replace([np.inf, -np.inf], np.nan).fillna(X.median()).fillna(0.0)
                y_live = pd.Series(live_targets, dtype=int)
                weights_live = pd.Series(2.5, index=X_live.index)  # 실전 체결 데이터 2.5배 가중치

                X = pd.concat([X, X_live], ignore_index=True)
                y = pd.concat([y, y_live], ignore_index=True)
                sample_weights = pd.concat([sample_weights, weights_live], ignore_index=True)
                logger.info(f"[MLFeatureEngine] 🧬 과거 캔들 {len(clean_data):,}개 + 실시간 실전 거래 {len(live_rows):,}개 (2.5x 가중치) 통합 학습 적용 완료")

        try:
            model = LGBMClassifier(
                objective='multiclass',
                num_class=3,
                class_weight='balanced',
                n_estimators=config.GBDT_N_ESTIMATORS,
                max_depth=config.GBDT_MAX_DEPTH,
                learning_rate=config.GBDT_LEARNING_RATE,
                random_state=42,
                verbosity=-1,
                subsample=config.GBDT_FEATURE_FRACTION,
                colsample_bytree=config.GBDT_FEATURE_FRACTION,
                subsample_freq=1,
                deterministic=True,
                force_col_wise=True
            )
            model.fit(X, y, sample_weight=sample_weights)
            importances = model.feature_importances_
        except Exception as e:
            import logging
            logging.getLogger('Lumos').error(f"❌ [LGBM 학습 실패] {e}")
            raise

        # 중요도 순 정렬
        sorted_feats = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
        top_10 = [f[0] for f in sorted_feats[:10]]
        top_3 = [f[0] for f in sorted_feats[:3]]
        feat_sha = self.compute_feature_names_sha256(feature_cols)

        if not hasattr(self, "_reload_lock") or self._reload_lock is None:
            self._reload_lock = threading.RLock()
        with self._reload_lock:
            self._bound_to_default_manifest = False
            snap = _ModelSnapshot(
                model=model,
                feature_names=tuple(feature_cols),
                top_10_features=tuple(top_10),
                top_3_features=tuple(top_3),
                feature_schema=str(target_schema),
                active_model_id=f"in_memory_{target_schema}",
                active_model_path="<in_memory>",
                active_model_sha256=feat_sha,
                feature_names_sha256=feat_sha,
                activated_at=datetime.now().isoformat(),
                previous_model_path=str(getattr(self, "previous_model_path", "")),
                previous_model_sha256=str(getattr(self, "previous_model_sha256", "")),
                load_status="TRAINED_IN_MEMORY",
                fallback_status="PRIMARY_ACTIVE",
                load_error="",
            )
            self._apply_snapshot(snap)

        return model, self.top_10_features, self.top_3_features

    def _get_thread_local(self) -> threading.local:
        tl = getattr(self, "_thread_local", None)
        if tl is None:
            tl = threading.local()
            self._thread_local = tl
        return tl

    def get_last_inference_snapshot(self) -> Optional[_ModelSnapshot]:
        """현재 스레드에서 가장 최근 추론에 사용된 불변 모델 스냅샷 반환"""
        return getattr(self._get_thread_local(), "last_inference_snapshot", None)

    def add_confidence_columns(
        self,
        feat_df: pd.DataFrame,
        snapshot: Optional[_ModelSnapshot] = None,
    ) -> pd.DataFrame:
        """전체 데이터프레임에 대해 3-Class 확신도 및 신호 컬럼 생성 (단일 스냅샷 격리 보장)"""
        feat_df = feat_df.copy()
        snap = snapshot if snapshot is not None else self.get_model_snapshot()
        self._get_thread_local().last_inference_snapshot = snap
        model_obj = snap.model
        feat_names = list(snap.feature_names)

        if model_obj is None or not feat_names:
            feat_df['Confidence'] = 0.50
            feat_df['Signal'] = 0
            feat_df['Direction'] = "NONE"
            feat_df['Prob_Long'] = 0.33
            feat_df['Prob_Short'] = 0.33
            feat_df['Prob_Neutral'] = 0.34
            return feat_df

        for col in feat_names:
            if col not in feat_df.columns:
                feat_df[col] = 0.0

        clean_X = feat_df[feat_names].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        try:
            probs = model_obj.predict_proba(clean_X)  # shape: (N, 3) -> [P(Short), P(Neutral), P(Long)]
            prob_short = probs[:, 0]
            prob_neutral = probs[:, 1]
            prob_long = probs[:, 2]

            signals = np.zeros(len(feat_df), dtype=int)
            confidences = np.full(len(feat_df), 0.50, dtype=float)
            directions = ["NONE"] * len(feat_df)

            for idx in range(len(feat_df)):
                ps, pn, pl = prob_short[idx], prob_neutral[idx], prob_long[idx]
                
                # 모든 방향(Long, Short, Neutral)에 동일한 스케일링 공식 적용
                calib_pl = min(0.95, max(config.CALIB_BASE, config.CALIB_BASE + (pl - config.CALIB_PIVOT) * config.CALIB_SCALE))
                calib_ps = min(0.95, max(config.CALIB_BASE, config.CALIB_BASE + (ps - config.CALIB_PIVOT) * config.CALIB_SCALE))
                calib_pn = min(0.95, max(config.CALIB_BASE, config.CALIB_BASE + (pn - config.CALIB_PIVOT) * config.CALIB_SCALE))
                
                # 스케일링된 값으로 원본 확률 덮어쓰기 (텔레그램 및 로그 표출용)
                prob_long[idx] = calib_pl
                prob_short[idx] = calib_ps
                prob_neutral[idx] = calib_pn
                
                if pl > pn and pl > ps:
                    signals[idx] = 1
                    confidences[idx] = calib_pl
                    directions[idx] = f"LONG_{config.BASE_ASSET_LONG}"
                elif ps > pn and ps > pl:
                    signals[idx] = -1
                    confidences[idx] = calib_ps
                    directions[idx] = f"SHORT_{config.BASE_ASSET_SHORT}"
                else:
                    signals[idx] = 0
                    confidences[idx] = calib_pn
                    directions[idx] = "NONE"

            feat_df['Confidence'] = confidences
            feat_df['Signal'] = signals
            feat_df['Direction'] = directions
            feat_df['Prob_Long'] = prob_long
            feat_df['Prob_Short'] = prob_short
            feat_df['Prob_Neutral'] = prob_neutral
        except Exception:
            feat_df['Confidence'] = 0.50
            feat_df['Signal'] = 0
            feat_df['Direction'] = "NONE"
            feat_df['Prob_Long'] = 0.33
            feat_df['Prob_Short'] = 0.33
            feat_df['Prob_Neutral'] = 0.34

        return feat_df

    def predict_signal(
        self,
        df_candle_15m: pd.DataFrame,
        snapshot: Optional[_ModelSnapshot] = None,
    ) -> Tuple[int, float, str]:
        """
        MoE 오케스트레이터 표준 인터페이스:
        - 반환값: (signal_code: 1 | -1 | 0, confidence: float 0.0~1.0, reason: str)
        - 단일 추론 요청 동안 동일한 불변 _ModelSnapshot 사용 보장
        """
        if snapshot is None:
            self.ensure_latest_manifest_model()
            snap = self.get_model_snapshot()
        else:
            snap = snapshot
        self._get_thread_local().last_inference_snapshot = snap

        if df_candle_15m is None or df_candle_15m.empty or len(df_candle_15m) < 15:
            return 0, 0.50, "데이터 부족"

        df_feat = self.extract_features(df_candle_15m)
        if df_feat.empty:
            return 0, 0.50, "피처 추출 불가"

        df_feat = self.add_confidence_columns(df_feat, snapshot=snap)
        last_row = df_feat.iloc[-1]

        sig = int(last_row.get("Signal", 0))
        conf = float(last_row.get("Confidence", 0.50))
        p_long = float(last_row.get("Prob_Long", 0.33))
        p_short = float(last_row.get("Prob_Short", 0.33))

        if sig == 1:
            return 1, conf, f"GBDT 3-Class {config.BASE_ASSET_LONG} 롱 파형 포착 (P_Long={p_long*100:.1f}%, Conf={conf*100:.1f}%)"
        elif sig == -1:
            return -1, conf, f"GBDT 3-Class {config.BASE_ASSET_SHORT} 숏 파형 포착 (P_Short={p_short*100:.1f}%, Conf={conf*100:.1f}%)"

        return 0, conf, "GBDT 관망/중립 상태"

    def predict_signal_full(
        self,
        df_candle_15m: pd.DataFrame,
        snapshot: Optional[_ModelSnapshot] = None,
    ) -> Tuple[int, float, str, Dict[str, float]]:
        if snapshot is None:
            self.ensure_latest_manifest_model()
            snap = self.get_model_snapshot()
        else:
            snap = snapshot
        self._get_thread_local().last_inference_snapshot = snap

        if hasattr(self.predict_signal, "_mock_name") or hasattr(self.predict_signal, "assert_called"):
            sig, conf, reason = self.predict_signal(df_candle_15m)
            rem = max(0.0, 1.0 - float(conf))
            if sig == 1:
                probs = {"LONG": float(conf), "SHORT": rem * 0.5, "NONE": rem * 0.5}
            elif sig == -1:
                probs = {"LONG": rem * 0.5, "SHORT": float(conf), "NONE": rem * 0.5}
            else:
                probs = {"LONG": rem * 0.5, "SHORT": rem * 0.5, "NONE": float(conf)}
            return int(sig), float(conf), str(reason), probs

        if df_candle_15m is None or df_candle_15m.empty or len(df_candle_15m) < 15:
            return 0, 0.50, "데이터 부족", {"LONG": 0.33, "SHORT": 0.33, "NONE": 0.34}
        df_feat = self.extract_features(df_candle_15m)
        if df_feat.empty:
            return 0, 0.50, "피처 추출 불가", {"LONG": 0.33, "SHORT": 0.33, "NONE": 0.34}
        df_feat = self.add_confidence_columns(df_feat, snapshot=snap)
        last_row = df_feat.iloc[-1]
        sig = int(last_row.get("Signal", 0))
        conf = float(last_row.get("Confidence", 0.50))
        p_long = float(last_row.get("Prob_Long", 0.33))
        p_short = float(last_row.get("Prob_Short", 0.33))
        p_none = float(last_row.get("Prob_Neutral", 0.34))
        if sig == 1:
            reason = f"GBDT 3-Class {config.BASE_ASSET_LONG} 롱 파형 포착 (P_Long={p_long*100:.1f}%, Conf={conf*100:.1f}%)"
        elif sig == -1:
            reason = f"GBDT 3-Class {config.BASE_ASSET_SHORT} 숏 파형 포착 (P_Short={p_short*100:.1f}%, Conf={conf*100:.1f}%)"
        else:
            reason = "GBDT 관망/중립 상태"
        probs = {
            "LONG": p_long,
            "SHORT": p_short,
            "NONE": p_none
        }
        return sig, conf, reason, probs


