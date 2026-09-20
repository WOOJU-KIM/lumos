import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.data_lake import DailyAutoPipeline

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

print("🚀 강제 모델 재학습 스크립트 (TQQQ 기준)")
pipeline = DailyAutoPipeline()

# 1. 메인 챔피언 포함 전체 모델 강제 재학습 (TQQQ/SQQQ)
print("1. 전체 모델 롤링 학습 시작...")
res1 = pipeline.run_step2_retrain_all_models()
print("전체 모델 재학습 결과:", res1)

# 2. Track 1 Refresh 모델 강제 교체
print("2. Track 1 롤링 모델 갱신 시작...")
res2 = pipeline.run_step2_retrain_refresh_model()
print("Track 1 모델 갱신 결과:", res2)

print("완료!")
