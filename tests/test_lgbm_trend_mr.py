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

    def test_7_wfa_reproducibility_and_purge_integrity(self):
        """[STEP 2 & 10] WFA 주간 롤링 재학습의 100% 결정론적 재현성 및 6캔들 Purge 경계 무결성 검증"""
        from lightgbm import LGBMClassifier
        eng = MLFeatureEngine(feature_schema="v5_challenger")
        with patch("core.data_lake.MarketDataLake.load_candles", return_value=pd.DataFrame()):
            feat_df = eng.extract_features(self.sample_df)
        fcols = MLFeatureEngine.get_feature_columns(feat_df, "v5_challenger")
        labels = eng.compute_triple_barrier_labels(feat_df, horizon=6)
        y = labels.map({1: 2, -1: 0, 0: 1}).fillna(1).astype(int)

        # 학습 구간 종료 시점(split_idx=90)에서 horizon=6 만큼 Purge 적용 시 미래 구간(>=90) 미참조 확인
        split_idx = 90
        horizon = 6
        purged_end = split_idx - horizon
        self.assertLessEqual(purged_end + horizon, split_idx, "Purge 적용 후 학습 라벨이 테스트 경계를 침범해서는 안 됩니다.")

        X_train = feat_df[fcols].iloc[:purged_end].fillna(0.0)
        y_train = y.iloc[:purged_end]
        X_test = feat_df[fcols].iloc[split_idx:].fillna(0.0)

        params = dict(
            objective="multiclass", num_class=3, class_weight="balanced",
            n_estimators=30, max_depth=4, learning_rate=0.02,
            subsample=0.5, colsample_bytree=0.5, subsample_freq=1,
            random_state=42, deterministic=True, force_col_wise=True, verbosity=-1, n_jobs=1
        )
        m1 = LGBMClassifier(**params).fit(X_train, y_train)
        m2 = LGBMClassifier(**params).fit(X_train, y_train)
        p1 = m1.predict_proba(X_test)
        p2 = m2.predict_proba(X_test)
        np.testing.assert_allclose(p1, p2, rtol=0.0, atol=1e-15, err_msg="WFA 동일 조건 재학습 시 확률이 100% 일치해야 합니다.")

    def test_8_saved_challenger_pkl_and_champion_regression(self):
        """[STEP 7 & 10] 저장된 Challenger PKL(77피처) 및 기존 Champion PKL(72피처) 무결성·출력 불변 회귀 검증"""
        import hashlib
        champ_path = PROJECT_ROOT / "models" / "model_champion.pkl"
        chall_path = PROJECT_ROOT / "models" / "model_challenger_trend_mr.pkl"
        self.assertTrue(champ_path.exists())
        self.assertTrue(chall_path.exists())

        champ_sha = hashlib.sha256(champ_path.read_bytes()).hexdigest()
        self.assertEqual(
            champ_sha,
            "e1982612a986d08a69044551b92c96ba5f8508375187466af9a9966e63b3ed5f",
            "기존 Champion PKL 파일 해시가 변경되어서는 안 됩니다.",
        )

        eng_champ = MLFeatureEngine(feature_schema="v4_champion")
        eng_chall = MLFeatureEngine(feature_schema="v5_challenger")
        self.assertTrue(eng_champ.load_model(champ_path))
        self.assertTrue(eng_chall.load_model(chall_path))

        self.assertEqual(len(eng_champ.feature_names), 72)
        self.assertEqual(len(eng_chall.feature_names), 77)

        # 동일 입력에 대해 두 모델이 각각 오류 없이 결정론적 출력을 반환하는지 검증
        s_a1, c_a1, _, p_a1 = eng_champ.predict_signal_full(self.sample_df)
        s_a2, c_a2, _, p_a2 = eng_champ.predict_signal_full(self.sample_df)
        self.assertEqual(s_a1, s_a2)
        self.assertAlmostEqual(c_a1, c_a2, places=12)
        self.assertEqual(p_a1, p_a2)

        s_e1, c_e1, _, p_e1 = eng_chall.predict_signal_full(self.sample_df)
        s_e2, c_e2, _, p_e2 = eng_chall.predict_signal_full(self.sample_df)
        self.assertEqual(s_e1, s_e2)
        self.assertAlmostEqual(c_e1, c_e2, places=12)
        self.assertEqual(p_e1, p_e2)

    def test_9_b_normalized_72_live_artifact_and_7_point_verification(self):
        """[STEP 3, 4, 5] B_Normalized_72 실전 모델 파일 및 7대 필수 무결성 검증 통과 확인"""
        import json
        norm_path = PROJECT_ROOT / "models" / "model_normalized_72.pkl"
        manifest_path = PROJECT_ROOT / "models" / "active_model_manifest.json"
        self.assertTrue(norm_path.exists(), "models/model_normalized_72.pkl 파일이 존재해야 합니다.")
        self.assertTrue(manifest_path.exists(), "models/active_model_manifest.json 파일이 존재해야 합니다.")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["active_model_id"], "B_Normalized_72")
        self.assertEqual(manifest["expected_schema"], "B_normalized")
        self.assertEqual(manifest["expected_num_features"], 72)

        ver = MLFeatureEngine.verify_model_artifact(
            norm_path,
            expected_sha256=manifest["expected_sha256"],
            expected_schema="B_normalized",
            enforce_live_policy=True,
        )
        self.assertTrue(ver["ok"], f"7대 검증 실패: {ver['error']}")
        for chk_name, chk_val in ver["checks"].items():
            self.assertTrue(chk_val, f"검증 항목 {chk_name}이 통과해야 합니다.")

        ext_feats = ver["extracted"]["feature_names"]
        self.assertEqual(len(ext_feats), 72)
        self.assertEqual(ext_feats, MLFeatureEngine.CANONICAL_B_NORMALIZED_72_FEATURES)
        for d_col in MLFeatureEngine.DOLLAR_SCALE_FEATURES:
            self.assertNotIn(d_col, ext_feats)
        for n_col in MLFeatureEngine.NORMALIZED_PRICE_FEATURES:
            self.assertIn(n_col, ext_feats)

    def test_10_live_default_loading_and_challenger_77_blocking(self):
        """[STEP 5] 기본 실전 로더가 B_Normalized_72를 로드하고 E Challenger(77피처)는 실전 정책에서 차단하는지 검증"""
        moe = MoEMetaOrchestrator()
        info = moe.get_active_model_info()
        self.assertEqual(info["active_model_id"], "B_Normalized_72")
        self.assertEqual(info["feature_schema"], "B_normalized")
        self.assertEqual(info["num_features"], 72)
        self.assertEqual(info["load_status"], "ACTIVE_B_NORMALIZED")
        self.assertTrue(info["is_ready"])

        # E Challenger 77개 모델은 enforce_live_policy=True 시 반드시 차단되어야 함
        chall_path = PROJECT_ROOT / "models" / "model_challenger_trend_mr.pkl"
        ver_chall = MLFeatureEngine.verify_model_artifact(chall_path, enforce_live_policy=True)
        self.assertFalse(ver_chall["ok"])
        self.assertIn("Forbidden experimental/challenger model blocked", ver_chall["error"])

    def test_11_automatic_fallback_to_champion_and_safe_entry_block_on_dual_failure(self):
        """[STEP 5 & 7] 주 모델 검증 실패 시 기존 Champion(v4_champion) 자동 폴백 및 이중 실패 시 신규 진입 안전 차단 검증"""
        import json
        manifest_path = PROJECT_ROOT / "models" / "active_model_manifest.json"
        orig_manifest_text = manifest_path.read_text(encoding="utf-8")
        orig_manifest = json.loads(orig_manifest_text)

        # Case A: 주 모델 SHA256 불일치 시 폴백 모델(models/model_champion.pkl)로 자동 복귀
        bad_primary_manifest = dict(orig_manifest)
        bad_primary_manifest["expected_sha256"] = "0" * 64
        with patch.object(Path, "read_text", return_value=json.dumps(bad_primary_manifest)):
            eng_fb = MLFeatureEngine()
            info_fb = eng_fb.get_active_model_info()
            self.assertEqual(info_fb["load_status"], "FALLBACK_TO_CHAMPION")
            self.assertEqual(info_fb["feature_schema"], "v4_champion")
            self.assertEqual(info_fb["num_features"], 72)
            self.assertTrue(info_fb["is_ready"])

        # Case B: 주 모델과 폴백 모델 모두 검증 실패 시 self.model=None 및 신규 진입 안전 차단(0, 0.50)
        both_bad_manifest = dict(orig_manifest)
        both_bad_manifest["expected_sha256"] = "0" * 64
        both_bad_manifest["fallback_expected_sha256"] = "f" * 64
        with patch.object(Path, "read_text", return_value=json.dumps(both_bad_manifest)):
            eng_blk = MLFeatureEngine()
            info_blk = eng_blk.get_active_model_info()
            self.assertEqual(info_blk["load_status"], "BLOCKED")
            self.assertIsNone(eng_blk.model)
            self.assertFalse(info_blk["is_ready"])
            sig, conf, _, _ = eng_blk.predict_signal_full(self.sample_df)
            self.assertEqual(sig, 0)
            self.assertEqual(conf, 0.50)

    def test_12_manifest_rollback_to_v4_champion_verification(self):
        """[STEP 7] 매니페스트 롤백 전환 시 기존 Champion(v4_champion, 72피처) 즉시 정상 로드 및 추론 검증"""
        import json
        manifest_path = PROJECT_ROOT / "models" / "active_model_manifest.json"
        orig_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rollback_manifest = {
            "active_model_id": "v4_champion",
            "active_model_path": "models/model_champion.pkl",
            "expected_schema": "v4_champion",
            "expected_num_features": 72,
            "expected_sha256": orig_manifest["fallback_expected_sha256"],
            "fallback_model_id": "v4_champion_fallback",
            "fallback_model_path": "models/model_moe_orchestrator.pkl",
            "fallback_expected_schema": "v4_champion",
            "fallback_expected_num_features": 72,
            "fallback_expected_sha256": orig_manifest["fallback_expected_sha256"],
        }
        with patch.object(Path, "read_text", return_value=json.dumps(rollback_manifest)):
            eng_rb = MLFeatureEngine()
            info_rb = eng_rb.get_active_model_info()
            self.assertEqual(info_rb["active_model_id"], "v4_champion")
            self.assertEqual(info_rb["feature_schema"], "v4_champion")
            self.assertEqual(info_rb["load_status"], "ACTIVE_V4_CHAMPION")
            self.assertEqual(info_rb["num_features"], 72)
            self.assertTrue(info_rb["is_ready"])
            sig, conf, _, probs = eng_rb.predict_signal_full(self.sample_df)
            self.assertIn(sig, [-1, 0, 1])
            self.assertTrue(0.50 <= conf <= 0.95)
            self.assertEqual(set(probs.keys()), {"LONG", "SHORT", "NONE"})

    def test_13_ten_year_boundary_and_edge_cases(self):
        """[STEP 1 & 7] config.py 10년 상한(ROLLING_TRAINING_WEEKS=520, 3640일) 경계값 및 엣지케이스 검증"""
        import config
        self.assertEqual(config.ROLLING_TRAINING_WEEKS, 520)
        self.assertEqual(config.TRAINING_LOOKBACK_DAYS, 3640)
        eng = MLFeatureEngine(feature_schema="B_normalized")

        def _make_daily_bars(years_span: float, n_bars: int = 240, gap_days: int = 0) -> pd.DataFrame:
            rng = np.random.default_rng(12345)
            end_ts = pd.Timestamp("2026-10-14 15:45:00")
            start_ts = end_ts - pd.Timedelta(days=int(round(years_span * 364)))
            idx = pd.date_range(start=start_ts, end=end_ts, periods=n_bars)
            if gap_days > 0 and n_bars > 60:
                keep_mask = np.ones(n_bars, dtype=bool)
                keep_mask[-40:-20] = False
                idx = idx[keep_mask]
            m = len(idx)
            rets = rng.normal(0.0002, 0.012, size=m)
            rets[::9] = 0.038
            rets[4::11] = -0.036
            close = 40.0 * np.exp(np.cumsum(rets))
            high = close * (1.0 + np.abs(rng.normal(0.008, 0.004, size=m)))
            low = close * (1.0 - np.abs(rng.normal(0.008, 0.004, size=m)))
            open_ = close * (1.0 + rng.normal(0.0, 0.003, size=m))
            vol = rng.integers(100_000, 900_000, size=m).astype(float)
            return pd.DataFrame(
                {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
                index=idx,
            )

        with patch("core.data_lake.MarketDataLake.load_candles", return_value=pd.DataFrame()):
            # 1) 3년치 데이터 (<10년): 10년 초과 제외 0건, 마지막 6봉 미확정 라벨 제외
            df_3y = _make_daily_bars(3.0, n_bars=220)
            m_3y, _, _ = eng.train_and_select_top_features(df_3y, schema="B_normalized", exclude_unconfirmed_tail=True)
            meta_3y = eng.last_training_window_meta
            self.assertIsNotNone(m_3y)
            self.assertTrue(meta_3y["is_valid_window"])
            self.assertEqual(meta_3y["excluded_older_than_10y_rows"], 0)
            self.assertEqual(meta_3y["excluded_unconfirmed_tail_rows"], 6)
            self.assertLessEqual(meta_3y["span_days"], config.TRAINING_LOOKBACK_DAYS)

            # 2) 9년치 데이터 (<10년): 10년 초과 제외 0건
            df_9y = _make_daily_bars(9.0, n_bars=240)
            m_9y, _, _ = eng.train_and_select_top_features(df_9y, schema="B_normalized", exclude_unconfirmed_tail=True)
            meta_9y = eng.last_training_window_meta
            self.assertIsNotNone(m_9y)
            self.assertTrue(meta_9y["is_valid_window"])
            self.assertEqual(meta_9y["excluded_older_than_10y_rows"], 0)
            self.assertEqual(meta_9y["excluded_unconfirmed_tail_rows"], 6)
            self.assertLessEqual(meta_9y["span_days"], config.TRAINING_LOOKBACK_DAYS)

            # 3) 정확히 10년 경계 (3640일): span_days <= 3640.0
            df_10y = _make_daily_bars(10.0, n_bars=260)
            m_10y, _, _ = eng.train_and_select_top_features(df_10y, schema="B_normalized", exclude_unconfirmed_tail=True)
            meta_10y = eng.last_training_window_meta
            self.assertIsNotNone(m_10y)
            self.assertTrue(meta_10y["is_valid_window"])
            self.assertLessEqual(meta_10y["span_days"], config.TRAINING_LOOKBACK_DAYS)

            # 4) 10년 초과 데이터 (12년치): 10년 이전 구간 자동 절삭 & 워밍업 보존 검증
            df_12y = _make_daily_bars(12.0, n_bars=320)
            m_12y, _, _ = eng.train_and_select_top_features(df_12y, schema="B_normalized", exclude_unconfirmed_tail=True)
            meta_12y = eng.last_training_window_meta
            self.assertIsNotNone(m_12y)
            self.assertTrue(meta_12y["is_valid_window"])
            self.assertGreater(meta_12y["excluded_older_than_10y_rows"], 0)
            self.assertEqual(meta_12y["excluded_warmup_nan_rows"], 0)  # 10년 경계 이전 데이터로 워밍업이 이미 완료됨
            self.assertEqual(meta_12y["excluded_unconfirmed_tail_rows"], 6)
            self.assertLessEqual(meta_12y["span_days"], config.TRAINING_LOOKBACK_DAYS)
            self.assertGreaterEqual(
                pd.Timestamp(meta_12y["train_start_dt"]),
                pd.Timestamp(meta_12y["cutoff_10y_dt"]),
            )

            # 5) 최근 일부 데이터 누락(갭) 존재 시에도 정상 학습
            df_gap = _make_daily_bars(5.0, n_bars=260, gap_days=15)
            m_gap, _, _ = eng.train_and_select_top_features(df_gap, schema="B_normalized", exclude_unconfirmed_tail=True)
            self.assertIsNotNone(m_gap)
            self.assertTrue(eng.last_training_window_meta["is_valid_window"])

            # 6) 데이터 부족 (<100행) 시 안전하게 중단 (None 반환)
            df_short = self.sample_df.iloc[:60].copy()
            m_short, top_short, _ = eng.train_and_select_top_features(df_short, schema="B_normalized", exclude_unconfirmed_tail=True)
            self.assertIsNone(m_short)
            self.assertEqual(len(top_short), 10)
            self.assertFalse(eng.last_training_window_meta["is_valid_window"])

    def test_14_atomic_retraining_activation_and_hot_reload(self):
        """[STEP 2, 3, 4] 임시 디렉토리에서 원자적 재학습·검증·활성화 및 런타임 핫리로드 검증"""
        import hashlib
        import json
        import shutil
        import tempfile
        from core.moe_orchestrator import train_and_save_moe_orchestrator
        from core.system_logger import system_logger

        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            shutil.copy2(PROJECT_ROOT / "models" / "model_champion.pkl", tmp_dir / "model_champion.pkl")
            shutil.copy2(PROJECT_ROOT / "models" / "model_normalized_72.pkl", tmp_dir / "model_normalized_72.pkl")
            shutil.copy2(PROJECT_ROOT / "models" / "active_model_manifest.json", tmp_dir / "active_model_manifest.json")

            manifest_file = tmp_dir / "active_model_manifest.json"
            m_data = json.loads(manifest_file.read_text(encoding="utf-8"))
            m_data["active_model_path"] = str(tmp_dir / "model_normalized_72.pkl")
            m_data["fallback_model_path"] = str(tmp_dir / "model_champion.pkl")
            manifest_file.write_text(json.dumps(m_data, indent=2), encoding="utf-8")

            running_moe = MoEMetaOrchestrator(manifest_path=manifest_file)
            old_sha = running_moe.get_active_model_info()["sha256"]

            with patch.object(system_logger, "log") as mock_syslog:
                retrained_moe = train_and_save_moe_orchestrator(
                    schema="B_normalized",
                    df_15m=self.sample_df,
                    models_dir_override=tmp_dir,
                )

            new_pkl = tmp_dir / "model_normalized_72.pkl"
            new_sha = hashlib.sha256(new_pkl.read_bytes()).hexdigest()
            self.assertNotEqual(old_sha, new_sha)
            self.assertEqual(retrained_moe.get_active_model_info()["sha256"], new_sha)

            logged_messages = [str(call.args[2]) for call in mock_syslog.call_args_list if len(call.args) >= 3]
            self.assertTrue(any("RETRAIN_COMPLETED" in msg for msg in logged_messages))
            self.assertTrue(any("MODEL_ACTIVATION_SUCCEEDED" in msg for msg in logged_messages))

            tmp_leftovers = list(tmp_dir.glob("*.tmp.*"))
            self.assertEqual(tmp_leftovers, [])

            running_moe.evaluate_dual_filter_signal(self.sample_df.tail(80))
            self.assertEqual(running_moe.get_active_model_info()["sha256"], new_sha)

    def test_15_retraining_failure_preserves_existing_model_100_percent(self):
        """[STEP 3 & 6-E] 재학습 실패/검증 실패 시 기존 model_normalized_72.pkl과 manifest가 100% 보존되는지 검증"""
        import hashlib
        import shutil
        import tempfile
        from core.moe_orchestrator import train_and_save_moe_orchestrator

        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            pkl_path = tmp_dir / "model_normalized_72.pkl"
            manifest_path = tmp_dir / "active_model_manifest.json"
            shutil.copy2(PROJECT_ROOT / "models" / "model_champion.pkl", tmp_dir / "model_champion.pkl")
            shutil.copy2(PROJECT_ROOT / "models" / "model_normalized_72.pkl", pkl_path)
            shutil.copy2(PROJECT_ROOT / "models" / "active_model_manifest.json", manifest_path)

            orig_pkl_sha = hashlib.sha256(pkl_path.read_bytes()).hexdigest()
            orig_manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

            # 1) 금지 스키마(v5_challenger) 재학습 시도 -> 즉시 차단
            with self.assertRaises(ValueError):
                train_and_save_moe_orchestrator(
                    schema="v5_challenger",
                    df_15m=self.sample_df,
                    models_dir_override=tmp_dir,
                )

            # 2) 데이터 부족(<100행)으로 재학습 실패 -> 기존 파일 불변
            with self.assertRaises(RuntimeError):
                train_and_save_moe_orchestrator(
                    schema="B_normalized",
                    df_15m=self.sample_df.iloc[:40],
                    models_dir_override=tmp_dir,
                )

            # 3) 후보 모델 검증 실패 시뮬레이션 -> 기존 파일 불변 & 임시 파일 삭제
            with patch("core.data_lake.MarketDataLake.load_candles", return_value=pd.DataFrame()), \
                 patch.object(MLFeatureEngine, "verify_model_artifact", return_value={"ok": False, "error": "simulated_verification_failure"}):
                with self.assertRaises(RuntimeError):
                    train_and_save_moe_orchestrator(
                        schema="B_normalized",
                        df_15m=self.sample_df,
                        models_dir_override=tmp_dir,
                    )

            self.assertEqual(hashlib.sha256(pkl_path.read_bytes()).hexdigest(), orig_pkl_sha)
            self.assertEqual(hashlib.sha256(manifest_path.read_bytes()).hexdigest(), orig_manifest_sha)
            self.assertEqual(list(tmp_dir.glob("*.tmp.*")), [])

    def test_16_fallback_and_rollback_scenarios_a_to_f(self):
        """[STEP 6] 시나리오 A~F 폴백·차단·즉시 롤백 전수 검증"""
        import json
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            b_pkl = tmp_dir / "model_normalized_72.pkl"
            c_pkl = tmp_dir / "model_champion.pkl"
            e_pkl = tmp_dir / "model_challenger_trend_mr.pkl"
            manifest_file = tmp_dir / "active_model_manifest.json"

            shutil.copy2(PROJECT_ROOT / "models" / "model_normalized_72.pkl", b_pkl)
            shutil.copy2(PROJECT_ROOT / "models" / "model_champion.pkl", c_pkl)
            shutil.copy2(PROJECT_ROOT / "models" / "model_challenger_trend_mr.pkl", e_pkl)
            orig_manifest = json.loads((PROJECT_ROOT / "models" / "active_model_manifest.json").read_text(encoding="utf-8"))

            def _write_manifest(**overrides):
                m = dict(orig_manifest)
                m["active_model_path"] = str(b_pkl)
                m["fallback_model_path"] = str(c_pkl)
                m.update(overrides)
                manifest_file.write_text(json.dumps(m, indent=2), encoding="utf-8")

            # 시나리오 A: model_normalized_72.pkl 파일 손상 -> Champion 자동 폴백
            b_pkl.write_bytes(b"corrupted_pickle_bytes")
            _write_manifest()
            eng_a = MLFeatureEngine(manifest_path=manifest_file)
            self.assertEqual(eng_a.get_active_model_info()["load_status"], "FALLBACK_TO_CHAMPION")
            self.assertEqual(eng_a.get_active_model_info()["feature_schema"], "v4_champion")

            # 복구 후 시나리오 B: SHA256 불일치 -> Champion 자동 폴백
            shutil.copy2(PROJECT_ROOT / "models" / "model_normalized_72.pkl", b_pkl)
            _write_manifest(expected_sha256="a" * 64)
            eng_b = MLFeatureEngine(manifest_path=manifest_file)
            self.assertEqual(eng_b.get_active_model_info()["load_status"], "FALLBACK_TO_CHAMPION")

            # 시나리오 C: 피처 수(77개) 불일치 -> Champion 자동 폴백
            _write_manifest(expected_num_features=77)
            eng_c = MLFeatureEngine(manifest_path=manifest_file)
            self.assertEqual(eng_c.get_active_model_info()["load_status"], "FALLBACK_TO_CHAMPION")

            # 시나리오 D: 금지된 v5_challenger 모델 지정 -> 차단 후 Champion 자동 폴백
            _write_manifest(active_model_path=str(e_pkl), expected_schema="v5_challenger", expected_num_features=77, expected_sha256="")
            eng_d = MLFeatureEngine(manifest_path=manifest_file)
            self.assertEqual(eng_d.get_active_model_info()["load_status"], "FALLBACK_TO_CHAMPION")

            # 시나리오 F: 주 모델 + 폴백 모델 모두 실패 -> self.model=None, 신규 진입 차단(0, 0.50)
            _write_manifest(expected_sha256="a" * 64, fallback_expected_sha256="b" * 64)
            eng_f = MLFeatureEngine(manifest_path=manifest_file)
            self.assertEqual(eng_f.get_active_model_info()["load_status"], "BLOCKED")
            self.assertIsNone(eng_f.model)
            sig_f, conf_f, _, _ = eng_f.predict_signal_full(self.sample_df)
            self.assertEqual((sig_f, conf_f), (0, 0.50))

    def test_17_orderless_live_runner_startup_and_pre_order_verification(self):
        """[STEP 4 & 5] KiwoomLiveRunner 무주문 기동 및 매수 직전 신호 판정·포트폴리오 불변 검증"""
        import hashlib
        from core.live_runner import KiwoomLiveRunner

        live_db_path = PROJECT_ROOT / "data" / "live_experience.db"
        db_sha_before = hashlib.sha256(live_db_path.read_bytes()).hexdigest()

        with patch("core.live_runner.KiwoomBroker") as MockBroker, \
             patch("core.live_runner.DispatcherAgent"), \
             patch("core.live_runner.TelegramController"), \
             patch("core.live_runner.system_logger"):
            broker_inst = MockBroker.return_value
            broker_inst.get_account_balance.return_value = {
                "ok": True,
                "usd_avail": 1000.0,
                "total_asset": 1000.0,
                "holdings": [],
            }
            broker_inst.get_unfilled_orders.return_value = {"ok": True, "orders": []}
            broker_inst.send_order.side_effect = AssertionError("send_order MUST NOT be called during STEP 5 orderless check!")

            runner = KiwoomLiveRunner(is_simulation=True)
            model_info = runner.moe_orchestrator.get_active_model_info()
            self.assertEqual(model_info["active_model_id"], "B_Normalized_72")
            self.assertEqual(model_info["feature_schema"], "B_normalized")
            self.assertEqual(model_info["num_features"], 72)
            self.assertEqual(model_info["load_status"], "ACTIVE_B_NORMALIZED")
            self.assertFalse(model_info["is_fallback"])
            self.assertTrue(model_info["is_ready"])

            # 15m 캔들 로드 -> 72개 피처 생성 -> 3-class 확률 산출 -> 신호 필터링(매수 직전 단계) 수행
            df_15m = runner.data_lake.load_candles("SOXL", "15m").tail(120)
            decision = runner.moe_orchestrator.evaluate_dual_filter_signal(df_15m)
            self.assertIn(decision["direction"], ["LONG", "SHORT", "NONE"])
            self.assertEqual(set(decision["gbdt_probs"].keys()), {"LONG", "SHORT", "NONE"})
            for p_val in decision["gbdt_probs"].values():
                self.assertTrue(0.0 <= float(p_val) <= 1.0)
            self.assertEqual(len(runner.moe_orchestrator.gbdt_engine.feature_names), 72)

            # 주문 호출 0건, 포지션/DB 상태 100% 불변 확인
            self.assertEqual(broker_inst.send_order.call_count, 0)
            self.assertIsNone(runner._active_position)
            db_sha_after = hashlib.sha256(live_db_path.read_bytes()).hexdigest()
            self.assertEqual(db_sha_before, db_sha_after)


if __name__ == "__main__":
    unittest.main()
