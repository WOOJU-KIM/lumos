import os
import sys
import json
import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.data_lake import MarketDataLake, DailyAutoPipeline
from core.model_registry import ModelRegistry

print("=" * 75)
print("🚀 [Lumos EOD 자동 데이터 아카이빙 및 롤링 모델 재학습 실행] 🚀")
print("=" * 75)

# Step 1. 9종 심볼 데이터 하베스팅
print("\n[Step 1] 9종 심볼 최신 분봉 데이터 수집 및 DB 적재 (UPSERT)...")
pipeline = DailyAutoPipeline()
import subprocess
subprocess.run([sys.executable, str(PROJECT_ROOT / 'scripts' / 'run_alpaca_full_sync.py')])
harvest_res = {'harvested_at': 'Now', 'total_records_updated': 'ALL'}
print(f"• 수집 완료 시각: {harvest_res.get('harvested_at')}")
print(f"• 신규/갱신 적재 레코드: {harvest_res.get('total_records_updated')}개")
for k, v in harvest_res.get('details', {}).items():
    print(f"   - {k}: {v}개")

# Step 2. Track 1 메인 최신화 롤링 모델 재학습
print("\n[Step 2] 최신화 모델 (M-DATA-REFRESH / Track 1) 전자동 롤링 재학습...")
from zoneinfo import ZoneInfo
now_kst = datetime.now().astimezone(ZoneInfo('Asia/Seoul'))
if now_kst.weekday() == 5:  # Saturday KST
    print('\n[Step 2] 토요일 아침입니다. 모델 롤링 재학습 및 적용을 시작합니다...')
    retrain_res = pipeline.run_step2_retrain_refresh_model()
    print(f'✅ 학습 완료 여부: {retrain_res.get("ok")}')
    print(f'📊 모델 ID: {retrain_res.get("model_id")}')
    print(f'📁 저장 경로: {retrain_res.get("file_path")}')
    print(f'🕒 갱신 시간: {retrain_res.get("updated_at")}')
else:
    print('\n[Step 2] 주중(화~금 아침)이므로 데이터 백업만 수행하고, 모델 롤링 학습 및 교체는 스킵합니다. (토요일에만 적용)')
    retrain_res = {}
# print(f"• 재학습 완료 여부: {retrain_res.get('ok')}")
# print(f"• 모델 ID: {retrain_res.get('model_id')}")
# print(f"• 저장 파일: {retrain_res.get('file_path')}")
# print(f"• 갱신 시각: {retrain_res.get('updated_at')}")

# Step 3. 데이터 레이크 상태 조회
print("\n[Step 3] 데이터 레이크(market_data.db) 최종 상태 검증...")
lake = MarketDataLake()
summary = lake.get_data_lake_summary()
print(f"• 총 적재 캔들 수: {summary['total_candles']:,}개")
print(f"• 고유 심볼 수: {summary['unique_symbols']}개")
for row in summary['breakdown']:
    print(f"   • {row['symbol']:<5} ({row['timeframe']:<3}): {row['earliest']} ~ {row['latest']} ({row['candle_count']:,}개)")

# Step 4. 모델 레지스트리 상태
print("\n[Step 4] 모델 레지스트리 최신 현황...")
reg = ModelRegistry()
with reg._get_connection() as conn:
    df_reg = pd.read_sql("SELECT model_id, algorithm_type, train_data_range, created_at, status, win_rate, total_return FROM model_registry;", conn)
    print(df_reg.to_string(index=False))

print("\n✅ EOD 일일 데이터 백업 및 모델 롤링 재학습 완료!")
