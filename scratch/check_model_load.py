import os
import sys
from pathlib import Path
from datetime import datetime
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.moe_orchestrator import MoEMetaOrchestrator

try:
    print("1. MoE 오케스트레이터 초기화 테스트 (모델 로딩)...")
    moe = MoEMetaOrchestrator()
    print("✅ MoEMetaOrchestrator 인스턴스 생성 성공!")
    
    gbdt_engine = moe.gbdt_engine
    if getattr(gbdt_engine, "model", None) is not None:
        print("✅ GBDT 엔진 모델 로딩 성공!")
        print(f"   - 보유 특징량 수: {len(getattr(gbdt_engine, 'feature_names', []))}")
        print(f"   - Top 3 지표: {getattr(gbdt_engine, 'top_3_features', [])}")
    else:
        print("❌ GBDT 엔진 모델 로딩 실패!")
        
    print("\n2. 최신 파일 갱신 시점 확인...")
    model_path = PROJECT_ROOT / "models" / "model_moe_orchestrator.pkl"
    if model_path.exists():
        mtime = os.path.getmtime(model_path)
        dt = datetime.fromtimestamp(mtime)
        print(f"✅ model_moe_orchestrator.pkl 존재함 (갱신시간: {dt.strftime('%Y-%m-%d %H:%M:%S')})")
    else:
        print("❌ model_moe_orchestrator.pkl 파일을 찾을 수 없습니다!")
except Exception as e:
    print(f"❌ 오류 발생: {e}")
