import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import sqlite3

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.mock_kiwoom_feeder import MockKiwoomFeeder
from core.data_lake import MarketDataLake
from core.moe_orchestrator import MoEMetaOrchestrator
import config
from config import DATA_DIR, GBDT_CONFIDENCE_THRESHOLD

def main():
    print("🚀 시작: 매매 테스트 (최근 1주일 / 5일 롤링)")
    
    conn = sqlite3.connect(DATA_DIR / "market_data.db")
    df_dates = pd.read_sql("SELECT DISTINCT date(datetime) as dt FROM market_candles WHERE symbol='TQQQ' ORDER BY datetime DESC LIMIT 5", conn)
    target_dates = sorted(df_dates['dt'].tolist())
    conn.close()
    
    print(f"✅ 시뮬레이션 대상 5일: {target_dates}")
    log_file = DATA_DIR / 'trade_test_logs.txt'
    
    for t_date in target_dates:
        print(f"\n=====================================")
        print(f"🗓️ 진행 중인 매매 테스트 날짜: {t_date}")
        print(f"=====================================")
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n=====================================\n")
            f.write(f"🚀 매매 테스트 시작: {t_date}\n")
            f.write(f"=====================================\n")

        # 1. 컴포넌트 초기화
        feeder = MockKiwoomFeeder(target_date=t_date)
        data_lake = feeder.data_lake # Feeder가 덮어친 DataLake 사용 (미래 참조 방지)
        moe_orchestrator = MoEMetaOrchestrator()
        
        # 2. 피더 백그라운드 시작 (가상 시간 재생)
        feeder.start()
        
        # 3. 매매 테스트 루프 (15분마다 모델 판단 릴레이)
        last_eval_time = None
        
        while True:
            if getattr(feeder, "is_finished", False):
                print(f"✅ {t_date} 캔들 데이터 재생 종료")
                break

            current_time = feeder.current_mock_time
            
            if current_time is None:
                time.sleep(0.01)
                continue
                
            if current_time.minute % 15 == 0:
                if last_eval_time != current_time:
                    last_eval_time = current_time
                    
                    df_15m = data_lake.load_candles("TQQQ", "15m")
                    if df_15m.empty:
                        continue
                        
                    live_prices = {sym: feeder.get_latest_price(sym, 0.0) for sym in config.ALL_SYMBOLS}
                    
                    moe_res = moe_orchestrator.evaluate_dual_filter_signal(df_candle_15m=df_15m, live_prices=live_prices)
                    
                    conf = float(moe_res.get('gating_confidence', 0.0)) * 100.0
                    is_appr = moe_res.get('is_approved', False)
                    direction = moe_res.get('direction', 'NONE')
                    reason = moe_res.get('details', {}).get('ml_reason', 'N/A')
                    
                    mock_time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')
                    test_log = f"[{mock_time_str}] 🧪 매매테스트 | 방향: {direction:<10} | GBDT 확신도: {conf:>5.2f}% | 🎯 진입승인: {is_appr} | 이유: {reason}"
                    print(test_log)
                    
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(test_log + '\n')

            time.sleep(0.005)

if __name__ == "__main__":
    main()
