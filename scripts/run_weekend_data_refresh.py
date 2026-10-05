import os
import sys
import json
import joblib
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Windows 콘솔 UTF-8 설정
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from config import BASE_DIR, MODELS_DIR, DATA_DIR
from core.data_lake import MarketDataLake, DailyAutoPipeline
from core.ml_engine import MLFeatureEngine
from core.model_registry import ModelRegistry
from core.system_logger import system_logger
from core.backtest_engine import GranularBacktestEngine

def run_weekend_data_refresh():
    print("=" * 80)
    print("🏛 [Lumos 주말 정기 데이터 최신화 (Data Refresh) 504 거래일 롤링 윈도우 파이프라인]")
    print(f"⏰ 실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S KST')}")
    print("=" * 80)
    system_logger.log("INFO", "DataRefresh", "🏛 주말 정기 데이터 최신화 504 거래일 롤링 윈도우 파이프라인 가동")

    data_lake = MarketDataLake()
    registry = ModelRegistry()

    # 1. 최근 504 거래일(Trading Days) 하드코딩 롤링 윈도우 쿼리 로드
    print("\n⏳ [1/5] market_data.db에서 최근 504 거래일(2년) 롤링 윈도우 데이터 추출...")
    df_15m = data_lake.load_rolling_candles("TQQQ", "15m", max_trading_days=config.ROLLING_TRAINING_WEEKS * 5)

    if df_15m.empty or len(df_15m) < 100:
        print("⚠️ 로컬 DB 데이터 부족으로 Yahoo Finance에서 60일치 수집 후 재시도...")
        data_lake.harvest_symbol("TQQQ", "15m", period="60d")
        df_15m = data_lake.load_rolling_candles("TQQQ", "15m", max_trading_days=config.ROLLING_TRAINING_WEEKS * 5)

    trading_days = sorted(df_15m.index.strftime('%Y-%m-%d').unique())
    num_days = len(trading_days)
    start_day = trading_days[0] if trading_days else "N/A"
    end_day = trading_days[-1] if trading_days else "N/A"

    print(f"   • 데이터 추출 완료: 총 {len(df_15m):,}개 15분봉 캔들")
    print(f"   • 거래일 윈도우: {num_days}일 (최대 504일 고정 롤링)")
    print(f"   • 데이터 기간: {start_day} ~ {end_day}")
    print(f"   • 꼬리 절삭(Drop) 메커니즘: 신규 1주일치 추가 시 2년 전 과거 데이터 자동 절삭 완료")

    # 2. 피처 추출 및 Triple Barrier 정답지 라벨링
    print("\n⏳ [2/5] 28개 종합 피처 추출 및 Triple Barrier (+3.0% TP / -2.0% SL / 90m) 라벨링...")
    ml_engine = MLFeatureEngine(confidence_threshold=config.GBDT_CONFIDENCE_THRESHOLD)
    feat_df = ml_engine.extract_features(df_15m)
    labels = MLFeatureEngine.compute_triple_barrier_labels(
        feat_df,
        take_profit=config.MAX_TP_PCT,
        stop_loss=config.SL_MIN_PCT,
        horizon=6
    )
    feat_df['Target'] = labels

    counts = feat_df['Target'].value_counts()
    valid_n = len(feat_df.dropna())
    print("   📊 [정답지 클래스 분포]:")
    print(f"      • Class  1 (TQQQ 롱 TP +3.0% 선도달): {counts.get(1, 0):4d}개 ({counts.get(1, 0)/valid_n*100:5.2f}%)")
    print(f"      • Class -1 (SQQQ 숏 TP -3.0% 선도달): {counts.get(-1, 0):4d}개 ({counts.get(-1, 0)/valid_n*100:5.2f}%)")
    print(f"      • Class  0 (관망 / 횡보 / 타임스탑):    {counts.get(0, 0):4d}개 ({counts.get(0, 0)/valid_n*100:5.2f}%)")

    # 3. LightGBM 3-Class 다중 분류기 훈련 (Concept Drift 방지 최신 Regime 가중)
    print("\n⏳ [3/5] LightGBM 3-Class 모델 훈련 (class_weight='balanced', Concept Drift 방지)...")
    model, top_10, top_3 = ml_engine.train_and_select_top_features(feat_df)
    print(f"   ✅ 훈련 완료! Top 3 핵심 지표: {top_3}")
    print(f"   ✅ Top 10 선별 지표: {top_10}")

    # 4. 모델 파일 저장 및 레지스트리 공식 등록
    print("\n⏳ [4/5] 최신화 모델 직렬화 & 레지스트리 갱신...")
    cand2_path = MODELS_DIR / "model_data_refresh.pkl"
    main_refresh_path = MODELS_DIR / "model_main_data_refresh.pkl"
    joblib.dump(model, cand2_path)
    joblib.dump(model, main_refresh_path)

    # 4-1. Lumos V3 하이브리드 MoE 실전 메인 모델 동시 최신화 (GBDT 65% + Cross-Asset Veto)
    from core.moe_orchestrator import MoEMetaOrchestrator, MOE_MODEL_PATH
    hybrid_moe = MoEMetaOrchestrator(confidence_threshold=config.GBDT_CONFIDENCE_THRESHOLD, gbdt_threshold=config.GBDT_CONFIDENCE_THRESHOLD, mode="hybrid_v3")
    hybrid_moe.gbdt_engine.model = model
    hybrid_moe.gbdt_engine.feature_names = ml_engine.feature_names
    hybrid_moe.gbdt_engine.top_10_features = top_10
    hybrid_moe.gbdt_engine.top_3_features = top_3
    joblib.dump(hybrid_moe, MOE_MODEL_PATH)
    joblib.dump(hybrid_moe, MODELS_DIR / "model_champion.pkl")
    print(f"   ✅ [Lumos V3 하이브리드 MoE 최신화 완료] ➔ {MOE_MODEL_PATH.name}, model_champion.pkl")

    # 4-2. Lumos 레거시 안전 모델(100% 동시합의) 동시 최신화 및 영구 보존
    safe_moe = MoEMetaOrchestrator(confidence_threshold=config.GBDT_CONFIDENCE_THRESHOLD, gbdt_threshold=config.GBDT_CONFIDENCE_THRESHOLD, mode="legacy_dual_consensus")
    safe_moe.gbdt_engine.model = model
    safe_moe.gbdt_engine.feature_names = ml_engine.feature_names
    safe_moe.gbdt_engine.top_10_features = top_10
    safe_moe.gbdt_engine.top_3_features = top_3
    safe_path = MODELS_DIR / "model_moe_legacy_safe.pkl"
    safe_champ_path = MODELS_DIR / "model_champion_legacy_safe.pkl"
    joblib.dump(safe_moe, safe_path)
    joblib.dump(safe_moe, safe_champ_path)
    print(f"   ✅ [Lumos 레거시 안전 모델(100% 합의) 동시 최신화 완료] ➔ {safe_path.name}")

    today_str = datetime.now().strftime('%Y-%m-%d')
    cand2_model_id = f"M-{datetime.now().strftime('%Y%m%d')}-REFRESH-504D"
    print(f"   ✅ 단독 GBDT 모델 저장 완료: {cand2_path.name}, {main_refresh_path.name}")

    # [5] (비활성화됨) 기존에는 여기서 구모델과 경쟁(토너먼트)을 시켰으나, 사용자 요청으로 무조건 최신 롤링 모델을 사용합니다.
    print('\n' + '=' * 80)
    print('✅ [Lumos 주말 504D 롤링 학습 및 실전 모델 즉시 교체 완료]')
    print('   -> 토너먼트 경쟁 없이 최신 롤링 모델(model_champion.pkl) 적용 완료')
    print('=' * 80)
if __name__ == "__main__":
    run_weekend_data_refresh()
