import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.shadow_tracker import ShadowTracker
from core.system_logger import system_logger

def test_shadow_tracker_async():
    print("=== Shadow Tracker Async DB Test ===")
    tracker = ShadowTracker()
    
    # 1. 초기 셋팅
    print(f"[초기화] 현재 섀도우 포지션 개수: {len(tracker.shadow_positions)}")
    
    # 2. 가상 진입 테스트 (75% 확신도로 SOXL 35.0 매수 패스, SL -2.0% 설정)
    tracker.add_virtual_trade("SOXL", 35.0, 75.0, 2.0)
    print(f"[진입 후] 현재 섀도우 포지션 개수: {len(tracker.shadow_positions)}")
    
    # 3. 틱 데이터 수신 시뮬레이션
    prices = [35.1, 35.3, 35.5, 35.8, 35.7, 35.6] # 35.8(+2.28%) 찍고 -> 35.6(-0.55% drawdown)으로 트레일링 스탑 발동
    for px in prices:
        print(f"틱 수신: SOXL @ {px:.2f}")
        tracker.check_ticks("SOXL", px)
        time.sleep(0.05)
    
    print(f"[틱 후] 현재 리스트 개수 (청산되었으면 0): {len(tracker.shadow_positions)}")

    # 4. 비동기 DB 기록을 위해 메인 스레드 잠시 대기 (매우 중요)
    print("비동기 DB 저장을 위해 0.5초 대기...")
    time.sleep(0.5)

    # 5. DB 검증
    import sqlite3
    from config import DATA_DIR
    db_path = DATA_DIR / "shadow_trades_live.db"
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT * FROM shadow_logs ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    print("\n[DB 로깅 확인]")
    print(row)
    conn.close()

if __name__ == "__main__":
    test_shadow_tracker_async()
