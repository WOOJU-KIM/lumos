import os
import sys
import hashlib
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from core.ml_engine import MLFeatureEngine
from core.moe_orchestrator import MoEMetaOrchestrator


class TestLGBMTrendMeanReversion(unittest.TestCase):
    """
    STEP 6 & STEP 7: LightGBM Trend + Mean-Reversion Challenger 독립 검증 테스트
    - 미래 데이터 누출(Look-ahead bias) 원천 차단 검증
    - 피처 스키마 및 피처 순서 결정론적 일치 검증
    - NaN, ±Inf, 0 변동성 캔들 방어 검증
    - 학습·추론 피처 계산 일치 검증
    - 기존 출력 인터페이스(predict_signal, predict_signal_full, MoEMetaOrchestrator) 100% 호환성 검증
    - 추세 지속 vs 평균회귀 반전 패턴 식별 검증
    """

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        n = 260
        dates = pd.date_range("2026-09-01 09:30", periods=n, freq="15min")
        returns = np.random.normal(0.0002, 0.008, size=n)
        close = 40.0 * np.cumprod(1.0 + returns)
        open_px = np.roll(close, 1)
        open_px[0] = 40.0
        high = np.maximum(open_px, close) * (1.0 + np.abs(np.random.normal(0.003, 0.002, size=n)))
        low = np.minimum(open_px, close) * (1.0 - np.abs(np.random.normal(0.003, 0.002, size=n)))
        vol = np.random.randint(80_000, 500_000, size=n).astype(float)

        cls.sample_df = pd.DataFrame(
            {
                "datetime": dates.strftime("%Y-%m-%d %H:%M:%S"),
                "Open": open_px,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": vol,
            }
        )

    def test_1_no_future_data_leakage_causal_invariance(self):
        """[STEP 7.1] 미래 캔들 유무에 따른 과거 시점 피처 값 100% 일치 (미래 데이터 누출 0건)"""
        eng = MLFeatureEngine(feature_schema="v5_challenger")
        # Isolate from external DB I/O to test pure indicator causality first, then with full DB
        with patch("core.data_lake.MarketDataLake.load_candles", return_value=pd.DataFrame()):
            full_feat = eng.extract_features(self.sample_df)
            cutoff = 180
            partial_feat = eng.extract_features(self.sample_df.iloc[:cutoff].copy())

        cols = MLFeatureEngine.get_feature_columns(full_feat, schema="v5_challenger")
        self.assertGreaterEqual(len(cols), 40)

        for col in cols:
            v_full = full_feat[col].iloc[cutoff - 1]
            v_part = partial_feat[col].iloc[cutoff - 1]
            if pd.isna(v_full) and pd.isna(v_part):
                continue
            self.assertAlmostEqual(
                float(v_full),
                float(v_part),
                places=10,
                msg=f"미래 데이터 누출 의심: 피처 {col} 값이 미래 캔들 절삭 전({v_full})과 후({v_part})에 다릅니다.",
            )

    def test_2_feature_schema_separation_and_order_determinism(self):
        """[STEP 2 & 7.2] Champion(72)과 Challenger(77) 스키마 분리, 순서 일관성 및 달러 절대값 배제 검증"""
        eng = MLFeatureEngine()
        feat_df = eng.extract_features(self.sample_df)

        cols_champ = MLFeatureEngine.get_feature_columns(feat_df, schema="v4_champion")
        cols_norm = MLFeatureEngine.get_feature_columns(feat_df, schema="B_normalized")
        cols_chall_1 = MLFeatureEngine.get_feature_columns(feat_df, schema="v5_challenger")
        cols_chall_2 = MLFeatureEngine.get_feature_columns(feat_df, schema="v5_challenger")

        self.assertEqual(len(cols_champ), 72, "Champion 스키마는 정확히 72개 피처여야 합니다.")
        self.assertEqual(len(cols_norm), 72, "Normalized B 스키마는 정확히 72개 피처여야 합니다.")
        self.assertEqual(len(cols_chall_1), 77, "Challenger v5 스키마는 정확히 77개 피처여야 합니다.")
        self.assertEqual(cols_chall_1, cols_chall_2, "피처 스키마 순서가 결정론적으로 동일해야 합니다.")

        # Challenger 스키마에는 달러 절대값 피처 8종과 중복/0중요도 피처 7종이 없어야 함
        for dollar_col in MLFeatureEngine.DOLLAR_SCALE_FEATURES:
            self.assertNotIn(dollar_col, cols_chall_1, f"달러 절대값 피처 {dollar_col}가 Challenger 스키마에 남아 있습니다.")
            self.assertIn(dollar_col, feat_df.columns, f"하위 호환용 원본 컬럼 {dollar_col}는 DataFrame에 유지되어야 합니다.")

        for norm_col in MLFeatureEngine.NORMALIZED_PRICE_FEATURES:
            self.assertIn(norm_col, cols_chall_1)

        for trend_col in MLFeatureEngine.TREND_CONTINUATION_FEATURES:
            self.assertIn(trend_col, cols_chall_1)

        for mr_col in MLFeatureEngine.MEAN_REVERSION_FEATURES:
            self.assertIn(mr_col, cols_chall_1)

    def test_3_nan_and_inf_resilience(self):
        """[STEP 7.3] NaN, ±Inf, 고저가 동일(0 범위) 이상치 입력 시 크래시 방지 및 유한 확률 출력 검증"""
        dirty_df = self.sample_df.copy()
        dirty_df.loc[50, "Close"] = np.nan
        dirty_df.loc[60, "Volume"] = np.inf
        dirty_df.loc[70, ["Open", "High", "Low", "Close"]] = 42.0  # candle_range == 0

        chall_path = PROJECT_ROOT / "models" / "model_challenger_trend_mr.pkl"
        eng = MLFeatureEngine(feature_schema="v5_challenger", model_path=str(chall_path))
        feat_df = eng.extract_features(dirty_df)

        cols = MLFeatureEngine.get_feature_columns(feat_df, schema="v5_challenger")
        has_inf = np.isinf(feat_df[cols].to_numpy(dtype=float, na_value=0.0)).any()
        self.assertFalse(has_inf, "extract_features 결과에 ±Inf가 남아 있어서는 안 됩니다.")

        sig, conf, reason, probs = eng.predict_signal_full(dirty_df)
        self.assertIn(sig, [-1, 0, 1])
        self.assertTrue(np.isfinite(conf) and 0.50 <= conf <= 0.95)
        for k in ("LONG", "SHORT", "NONE"):
            self.assertIn(k, probs)
            self.assertTrue(np.isfinite(probs[k]) and 0.50 <= probs[k] <= 0.95)

    def test_4_train_and_inference_consistency_and_determinism(self):
        """[STEP 6 & 7.4] 학습·추론 피처 일치 및 동일 입력에 대한 100% 결정론적 재현성 검증"""
        eng = MLFeatureEngine(feature_schema="v5_challenger")
        with patch("core.data_lake.MarketDataLake.load_candles", return_value=pd.DataFrame()):
            model, top10, top3 = eng.train_and_select_top_features(
                self.sample_df,
                live_experience_df=pd.DataFrame(),
                schema="v5_challenger",
            )
            self.assertIsNotNone(model)
            self.assertEqual(len(top10), 10)
            self.assertEqual(len(top3), 3)

            res1 = eng.predict_signal_full(self.sample_df)
            res2 = eng.predict_signal_full(self.sample_df)

        self.assertEqual(res1[0], res2[0])
        self.assertEqual(res1[1], res2[1])
        self.assertEqual(res1[2], res2[2])
        self.assertEqual(res1[3], res2[3])

    def test_5_output_interface_compatibility_with_moe_orchestrator(self):
        """[STEP 6] 기존 소비자(MoEMetaOrchestrator)가 코드 수정 없이 Challenger 모델을 즉시 사용 가능한지 검증"""
        chall_path = PROJECT_ROOT / "models" / "model_challenger_trend_mr.pkl"
        self.assertTrue(chall_path.exists(), "Challenger 모델 파일이 존재해야 합니다.")

        moe_champ = MoEMetaOrchestrator(confidence_threshold=config.GBDT_CONFIDENCE_THRESHOLD, mode="hybrid_v3")
        moe_chall = MoEMetaOrchestrator(confidence_threshold=config.GBDT_CONFIDENCE_THRESHOLD, mode="hybrid_v3")
        loaded = moe_chall.gbdt_engine.load_model(chall_path)
        self.assertTrue(loaded)
        self.assertEqual(len(moe_chall.gbdt_engine.feature_names), 77)

        out_champ = moe_champ.evaluate_dual_filter_signal(self.sample_df)
        out_chall = moe_chall.evaluate_dual_filter_signal(self.sample_df)

        # 반환 키 집합 및 자료형 100% 일치 확인
        self.assertEqual(set(out_champ.keys()), set(out_chall.keys()))
        for k, v in out_champ.items():
            if k != "features":
                self.assertEqual(type(v), type(out_chall[k]), f"키 {k}의 반환 타입이 다릅니다.")
        self.assertIn(out_chall["direction"], ["LONG", "SHORT", "NONE"])
        self.assertEqual(set(out_chall["gbdt_probs"].keys()), {"LONG", "SHORT", "NONE"})

    def test_6_trend_continuation_vs_mean_reversion_feature_semantics(self):
        """[STEP 3 & 7.5] 급락 가속(추세 지속)과 급락 감속+아래꼬리(평균회귀 반등)에서 신규 피처 부호·방향 검증"""
        n = 40
        dates = pd.date_range("2026-09-10 09:30", periods=n, freq="15min")

        # 시나리오 A: 하락 가속 + 음봉 지속 + 저점 돌파 (Falling Knife)
        px_a = np.linspace(50.0, 44.0, n)
        # 마지막 6봉에서 하락폭 가속
        px_a[-6:] = [45.0, 44.6, 44.1, 43.4, 42.5, 41.3]
        df_falling_knife = pd.DataFrame({
            "datetime": dates.strftime("%Y-%m-%d %H:%M:%S"),
            "Open": px_a + 0.3,
            "High": px_a + 0.35,
            "Low": px_a - 0.05,
            "Close": px_a,
            "Volume": [200_000.0] * n,
        })

        # 시나리오 B: 하락 감속 + 긴 아래꼬리 매수세 유입 + 양봉 전환 (Exhaustion Rebound)
        px_b = px_a.copy()
        # 직전 3봉(-6..-4)은 급락했으나 최근 3봉(-3..-1)은 하락 멈추고 저점 회복
        px_b[-6:] = [45.0, 43.8, 42.5, 42.2, 42.1, 42.6]
        open_b = px_b + 0.2
        open_b[-1] = 42.0  # 마지막 봉 양봉 (Close 42.6 > Open 42.0)
        low_b = px_b - 0.1
        low_b[-1] = 41.0   # 마지막 봉 긴 아래꼬리 (Low 41.0 -> Close 42.6)
        high_b = np.maximum(open_b, px_b) + 0.1
        df_rebound = pd.DataFrame({
            "datetime": dates.strftime("%Y-%m-%d %H:%M:%S"),
            "Open": open_b,
            "High": high_b,
            "Low": low_b,
            "Close": px_b,
            "Volume": [200_000.0] * n,
        })

        eng = MLFeatureEngine(feature_schema="v5_challenger")
        with patch("core.data_lake.MarketDataLake.load_candles", return_value=pd.DataFrame()):
            fa = eng.extract_features(df_falling_knife).iloc[-1]
            fb = eng.extract_features(df_rebound).iloc[-1]

        # 시나리오 A (하락 가속): Downside_Persistence_6 높음, Return_Accel_3 < 0, Signed_Body_Ratio < 0, Breakout_Position_20 낮음
        self.assertGreaterEqual(float(fa["Downside_Persistence_6"]), 0.80)
        self.assertLess(float(fa["Return_Accel_3"]), 0.0)
        self.assertLess(float(fa["Signed_Body_Ratio"]), 0.0)
        self.assertLess(float(fa["Breakout_Position_20"]), 0.10)

        # 시나리오 B (하락 감속 + 아래꼬리 반등): Return_Accel_3 > 0, Signed_Body_Ratio > 0, Lower_Wick_Expansion 확대
        self.assertGreater(float(fb["Return_Accel_3"]), 0.0)
        self.assertGreater(float(fb["Signed_Body_Ratio"]), 0.0)
        self.assertGreater(float(fb["Lower_Wick_Expansion"]), float(fa["Lower_Wick_Expansion"]))
        self.assertGreater(float(fb["EMA20_Slope_Accel"]), float(fa["EMA20_Slope_Accel"]))


if __name__ == "__main__":
    unittest.main()
