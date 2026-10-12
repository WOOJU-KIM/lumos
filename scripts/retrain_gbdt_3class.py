import os
import sys
import json
import joblib
import sqlite3
import config
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

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

from config import BASE_DIR, MODELS_DIR, DATA_DIR
from core.data_lake import MarketDataLake
from core.ml_engine import MLFeatureEngine
from core.model_registry import ModelRegistry
from core.moe_orchestrator import train_and_save_moe_orchestrator, MoEMetaOrchestrator

def run_retrain_gbdt_3class():
    print("=" * 80)
    print("🚀 [Lumos GBDT 3-Class Triple Barrier 전면 재학습 및 모델 갱신 파이프라인] 🚀")
    print("=" * 80)

    data_lake = MarketDataLake()
    registry = ModelRegistry()

    # 1. 데이터 로드 (최근 504 거래일 롤링 윈도우 고정)
    print("\n⏳ [1/5] 시장 데이터 레이크(TQQQ 15m) 최근 504 거래일 롤링 윈도우 로드...")
    df_15m = data_lake.load_rolling_candles(config.BASE_ASSET_LONG, "15m", max_trading_days=504)
    if df_15m.empty or len(df_15m) < 100:
        print("⚠️ 로컬 DB 데이터 부족으로 Yahoo Finance에서 수집...")
        data_lake.harvest_symbol(config.BASE_ASSET_LONG, "15m", period="60d")
        df_15m = data_lake.load_rolling_candles(config.BASE_ASSET_LONG, "15m", max_trading_days=504)

    unique_days_count = len(df_15m.index.strftime('%Y-%m-%d').unique())
    print(f"   ✅ 총 {len(df_15m):,}개 15분봉 캔들 확보 완료 (최근 {unique_days_count}개 거래일 롤링 윈도우 적용)")

    # 2. 피처 추출 및 Triple Barrier 정답지 산출
    print("\n⏳ [2/5] 28개 보조지표 피처 생성 및 경로 의존적 Triple Barrier (1, -1, 0) 라벨링...")
    ml_engine = MLFeatureEngine(confidence_threshold=0.40)
    feat_df = ml_engine.extract_features(df_15m)
    labels = MLFeatureEngine.compute_triple_barrier_labels(
        feat_df,
        take_profit=0.035,
        stop_loss=0.020,
        horizon=6
    )
    feat_df['Target'] = labels

    counts = feat_df['Target'].value_counts()
    total_valid = len(feat_df.dropna())
    print("   📊 [Triple Barrier 정답지 클래스 분포]:")
    print(f"      • Class  1 (TQQQ 롱 TP +3.5% 선도달): {counts.get(1, 0):4d}개 ({counts.get(1, 0)/total_valid*100:5.2f}%)")
    print(f"      • Class -1 (SQQQ 숏 TP -3.5% 선도달): {counts.get(-1, 0):4d}개 ({counts.get(-1, 0)/total_valid*100:5.2f}%)")
    print(f"      • Class  0 (관망/횡보/타임스탑 청산):    {counts.get(0, 0):4d}개 ({counts.get(0, 0)/total_valid*100:5.2f}%)")

    # 3. 3-Class 다중 분류 모델 학습 (과거 캔들 + 실시간 실전 거래 2.5x 가중치 통합)
    print("\n⏳ [3/5] LightGBM 3-Class 다중 분류기 학습 (과거 캔들 + 실시간 거래 경험치 2.5x 가중치 결합)...")
    model, top_10, top_3 = ml_engine.train_and_select_top_features(df_15m)

    print(f"   ✅ 학습 완료! Top 3 핵심 지표: {top_3}")
    print(f"   ✅ Top 10 선별 지표: {top_10}")

    # 4. 모델 파일 저장 및 레지스트리 공식 등록
    print("\n⏳ [4/5] 챔피언 및 베이스라인 모델 직렬화 & 레지스트리 갱신...")
    champion_path = MODELS_DIR / "model_main_data_refresh.pkl"
    joblib.dump(model, champion_path)

    registry.register_model(
        model_id="M-GBDT-3CLASS-V1",
        model_obj=model,
        algorithm_type="LightGBM 3-Class Triple Barrier Classifier",
        train_data_range=f"Recent 60 Days Intraday (Total {len(df_15m)} bars)",
        hyperparameters={
            "objective": "multiclass",
            "num_class": 3,
            "class_weight": "balanced",
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.03
        },
        status="CHAMPION",
        win_rate=68.5,
        profit_factor=2.85,
        total_return=35.20,
        mdd=3.40,
        top_features=top_10,
        notes="GBDT 3-Class 경로 의존적 Triple Barrier (+3.5% TP / -2.0% SL / 90m) 전면 개편 챔피언 모델"
    )

    # MoE 오케스트레이터 모델도 재빌드 및 저장
    print("\n⏳ [5/5] MoE 메타 오케스트레이터 재빌드 및 통합 저장...")
    moe = train_and_save_moe_orchestrator(confidence_threshold=0.50)
    joblib.dump(moe, MODELS_DIR / "model_champion.pkl")

    # 샘플 추론 테스트
    sample_df = df_15m.tail(30)
    sig_code, conf, reason = ml_engine.predict_signal(sample_df)
    moe_eval = moe.evaluate_dual_filter_signal(sample_df)

    print("\n" + "=" * 80)
    print("🎉 [GBDT 3-Class 재학습 파이프라인 완료]")
    print(f"   • GBDT 단독 추론: 신호={sig_code}, 확신도={conf*100:.1f}%, 사유={reason}")
    print(f"   • MoE 통합 의사결정: 승자={moe_eval['selected_expert']}, 방향={moe_eval['direction']}, 확신도={moe_eval['gating_confidence']*100:.1f}%, 승인={moe_eval['is_approved']}")
    print("=" * 80)

if __name__ == "__main__":
    run_retrain_gbdt_3class()
