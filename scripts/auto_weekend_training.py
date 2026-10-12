import os
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from core.market_calendar import USMarketCalendar
from core.telegram_notifier import TelegramNotifier

def main():
    ny_tz = ZoneInfo("America/New_York")
    notifier = TelegramNotifier()
    
    # 1. 어제(뉴욕 시간 기준)가 이번 주의 마지막 거래일이었는지 확인
    # (한국 시간 금/토 새벽에 이 스크립트가 돌기 때문)
    now_ny = datetime.now(ny_tz)
    yesterday_ny = now_ny - timedelta(days=1)
    
    is_last = USMarketCalendar.is_last_trading_day_of_week(yesterday_ny)
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 미국 날짜 기준 어제({yesterday_ny.date()})가 주 마지막 거래일인가? {is_last}")
    
    if not is_last:
        print("-> 아직 마지막 거래일이 아니므로 데이터 백업 및 학습을 건너뜁니다.")
        return
        
    print("-> 🎯 마지막 거래일 마감이 확인되었습니다! 데이터 롤링 및 재학습 파이프라인을 가동합니다.")
    notifier._dispatch_message(f"🤖 [Lumos Auto-Train] 이번 주 거래 마감 확인. 롤링 백업 및 재학습을 시작합니다.")
    
    # config에서 지정된 일수만큼 동적으로 시작일 계산 (기본 730일 = 2년)
    end_date = yesterday_ny.date()
    start_date = end_date - timedelta(days=config.TRAINING_LOOKBACK_DAYS)
    
    end_dt_str = end_date.strftime("%Y-%m-%d")
    start_dt_str = start_date.strftime("%Y-%m-%d")
    
    print(f"-> 롤링 타겟 기간: {start_dt_str} ~ {end_dt_str}")
    
    # 스크립트 실행 (환경 변수로 기간을 넘겨주어 ingest 스크립트가 반영하게 할 수 있음)
    env = os.environ.copy()
    env["LUMOS_ROLLING_START"] = start_dt_str
    env["LUMOS_ROLLING_END"] = end_dt_str
    
    script_path = str(PROJECT_ROOT / "scripts" / "ingest_alpaca_2yr_and_retrain.py")
    
    try:
        subprocess.run([sys.executable, script_path], env=env, check=True)
        print("✅ 롤링 백업 및 재학습 파이프라인 무사 완료!")
        notifier._dispatch_message(f"✅ [Lumos Auto-Train] 최근 {config.TRAINING_LOOKBACK_DAYS}일({start_dt_str}~{end_dt_str}) 데이터 적재 및 GBDT 재학습 완료! 다음 주 매매 준비 끝.")
    except subprocess.CalledProcessError as e:
        print(f"❌ 파이프라인 실행 중 에러 발생: {e}")
        notifier._dispatch_message(f"🚨 [Lumos Auto-Train] 재학습 중 에러가 발생했습니다! 로그를 확인하세요.")
        
    # 알림 전송 대기 및 셧다운
    notifier.shutdown()

if __name__ == "__main__":
    main()
