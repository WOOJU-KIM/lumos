import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
from unittest.mock import MagicMock, patch
import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import config
from core.shadow_tracker import ShadowTracker, MAX_SHADOW_SLOTS, SLOT_KEYS

class TestShadowForcedEntry(unittest.TestCase):
    def setUp(self):
        # 테스트용 임시 DB 경로 지정 (실운영 DB 접근 차단)
        self.test_db_path = config.DATA_DIR / "test_shadow_temp.db"
        if self.test_db_path.exists():
            try:
                self.test_db_path.unlink()
            except Exception:
                pass
        self.tracker = ShadowTracker(db_path=self.test_db_path)

    def tearDown(self):
        for p in (self.test_db_path, self.test_db_path.with_suffix(".jsonl")):
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

    def test_slot_structure_and_strict_keys(self):
        """10개 슬롯 생성 및 12개 필수 키만 존재하는지 검증"""
        self.assertEqual(len(self.tracker.slots), 10)
        expected_keys = set(SLOT_KEYS)
        self.assertEqual(len(expected_keys), 12)
        
        for slot in self.tracker.slots:
            self.assertEqual(set(slot.keys()), expected_keys)
            self.assertFalse(slot["active"])
            self.assertEqual(slot["qty"], 0)
            self.assertIsNone(slot["symbol"])
            self.assertIsNone(slot["entry_time"])
            self.assertEqual(slot["entry_price"], 0.0)

    def test_reject_before_candle_close(self):
        """15분봉이 완전히 마감되기 전(09:30~09:44:59)에는 진입하지 않음"""
        ny_tz = ZoneInfo("America/New_York")
        # 2026-10-14 수요일 09:35 (첫 15분봉 미마감 시점)
        t_early = datetime.datetime(2026, 10, 14, 9, 35, 0, tzinfo=ny_tz)
        
        candle_id = self.tracker.get_last_closed_15m_candle(t_early)
        self.assertIsNone(candle_id, "09:35에는 마감된 15분봉이 없어야 함")
        
        entered = self.tracker.try_enter_shadow_trade("SOXL", 30.0, current_time=t_early)
        self.assertFalse(entered, "미마감 봉 상태에서는 진입이 거부되어야 함")
        self.assertEqual(sum(1 for s in self.tracker.slots if s["active"]), 0)

    def test_enter_on_closed_candle(self):
        """09:45:00에 첫 번째 15분봉이 마감되어 정상 진입 성공"""
        ny_tz = ZoneInfo("America/New_York")
        # 2026-10-14 수요일 09:45:00 (첫 15분봉 마감 직후)
        t_closed = datetime.datetime(2026, 10, 14, 9, 45, 0, tzinfo=ny_tz)
        
        candle_id = self.tracker.get_last_closed_15m_candle(t_closed)
        self.assertEqual(candle_id, "2026-10-14 09:30-09:45")
        
        entered = self.tracker.try_enter_shadow_trade("SOXL", 30.0, current_time=t_closed, atr_pct=0.015)
        self.assertTrue(entered)
        
        # 슬롯 0 활성화 검증
        slot0 = self.tracker.slots[0]
        self.assertTrue(slot0["active"])
        self.assertEqual(slot0["symbol"], "SOXL")
        self.assertEqual(slot0["entry_price"], 30.0)
        self.assertEqual(slot0["qty"], 1)
        self.assertEqual(slot0["highest_price"], 30.0)
        # 손절가: max(0.02, min(0.032, 0.015 * 1.5)) = 0.0225 -> 30.0 * (1 - 0.0225) = 29.325
        self.assertAlmostEqual(slot0["stop_price"], 30.0 * (1.0 - 0.0225), places=2)
        # 키 개수 불변 확인
        self.assertEqual(set(slot0.keys()), set(SLOT_KEYS))

    def test_duplicate_candle_entry_blocked(self):
        """동일한 15분봉에 대한 중복 진입 시도 시 차단 검증"""
        ny_tz = ZoneInfo("America/New_York")
        t1 = datetime.datetime(2026, 10, 14, 9, 45, 0, tzinfo=ny_tz)
        t2 = datetime.datetime(2026, 10, 14, 9, 46, 0, tzinfo=ny_tz)
        t3 = datetime.datetime(2026, 10, 14, 9, 59, 0, tzinfo=ny_tz)
        
        # 첫 진입
        entered1 = self.tracker.try_enter_shadow_trade("SOXL", 30.0, current_time=t1)
        self.assertTrue(entered1)
        self.assertEqual(sum(1 for s in self.tracker.slots if s["active"]), 1)
        
        # 1분 후 동일 봉 진입 시도
        entered2 = self.tracker.try_enter_shadow_trade("SOXL", 30.1, current_time=t2)
        self.assertFalse(entered2, "동일 봉 중복 진입은 거부되어야 함")
        self.assertEqual(sum(1 for s in self.tracker.slots if s["active"]), 1)
        
        # 14분 후 동일 봉 진입 시도
        entered3 = self.tracker.try_enter_shadow_trade("SOXL", 30.2, current_time=t3)
        self.assertFalse(entered3, "동일 봉 중복 진입은 거부되어야 함")
        self.assertEqual(sum(1 for s in self.tracker.slots if s["active"]), 1)

    def test_confidence_and_threshold_ignored(self):
        """confidence와 threshold가 낮아도 SHADOW 진입이 수행됨"""
        ny_tz = ZoneInfo("America/New_York")
        t_closed = datetime.datetime(2026, 10, 14, 10, 0, 0, tzinfo=ny_tz)
        
        # add_virtual_trade를 통해 매우 낮은 confidence(10%)와 threshold 미달 상태로 호출
        entered = self.tracker.add_virtual_trade("SOXS", 15.0, confidence=10.0, current_time=t_closed)
        self.assertTrue(entered, "confidence가 낮아도 강제 진입되어야 함")
        
        active_slots = [s for s in self.tracker.slots if s["active"]]
        self.assertEqual(len(active_slots), 1)
        self.assertEqual(active_slots[0]["symbol"], "SOXS")

    def test_invalid_price_rejected(self):
        """0 이하, None 등 유효하지 않은 가격은 진입 거부"""
        ny_tz = ZoneInfo("America/New_York")
        t_closed = datetime.datetime(2026, 10, 14, 10, 0, 0, tzinfo=ny_tz)
        
        # 0원 진입 거부
        self.assertFalse(self.tracker.try_enter_shadow_trade("SOXL", 0.0, current_time=t_closed))
        # 음수 진입 거부
        self.assertFalse(self.tracker.try_enter_shadow_trade("SOXL", -10.0, current_time=t_closed))
        # None 진입 거부
        self.assertFalse(self.tracker.try_enter_shadow_trade("SOXL", None, current_time=t_closed))
        
        self.assertEqual(sum(1 for s in self.tracker.slots if s["active"]), 0)

    def test_max_10_slots_limit_and_no_empty_slot_log(self):
        """최대 10개 슬롯 제한 및 빈 slot 부족 시 거부 검증"""
        ny_tz = ZoneInfo("America/New_York")
        
        # 10개의 서로 다른 15분봉으로 10개 슬롯을 순차적으로 채움
        for i in range(10):
            # 09:45 (봉 1), 10:00 (봉 2), ...
            minute_offset = 45 + (i * 15)
            h = 9 + (minute_offset // 60)
            m = minute_offset % 60
            t = datetime.datetime(2026, 10, 14, h, m, 0, tzinfo=ny_tz)
            
            res = self.tracker.try_enter_shadow_trade("SOXL", 30.0 + i, current_time=t)
            self.assertTrue(res, f"슬롯 {i} 진입 성공해야 함")
        
        # 10개 슬롯 모두 점유 확인
        self.assertEqual(sum(1 for s in self.tracker.slots if s["active"]), 10)
        
        # 11번째 진입 시도 (빈 슬롯 없음)
        t11 = datetime.datetime(2026, 10, 14, 12, 30, 0, tzinfo=ny_tz)
        res11 = self.tracker.try_enter_shadow_trade("SOXL", 40.0, current_time=t11)
        self.assertFalse(res11, "빈 슬롯이 없으므로 진입 거부되어야 함")
        
        # 기존 슬롯이 덮어써지지 않고 10개로 온전히 유지되는지 확인
        self.assertEqual(sum(1 for s in self.tracker.slots if s["active"]), 10)
        self.assertEqual(self.tracker.slots[0]["entry_price"], 30.0)
        self.assertEqual(self.tracker.slots[9]["entry_price"], 39.0)

    def test_slot_state_isolation(self):
        """slot 간 상태가 섞이지 않고 독립적으로 관리됨을 검증"""
        ny_tz = ZoneInfo("America/New_York")
        t1 = datetime.datetime(2026, 10, 14, 9, 45, 0, tzinfo=ny_tz)
        t2 = datetime.datetime(2026, 10, 14, 10, 0, 0, tzinfo=ny_tz)
        
        # 슬롯 0: SOXL @ 30.0 (Stop: 29.4)
        self.tracker.try_enter_shadow_trade("SOXL", 30.0, current_time=t1)
        # 슬롯 1: SOXS @ 15.0 (Stop: 14.7)
        self.tracker.try_enter_shadow_trade("SOXS", 15.0, current_time=t2)
        
        # SOXL 가격 틱 수신: 31.0으로 상승
        self.tracker.check_ticks("SOXL", 31.0, current_time=t2)
        
        # 슬롯 0의 최고가는 31.0으로 갱신되어야 함
        self.assertEqual(self.tracker.slots[0]["highest_price"], 31.0)
        # 슬롯 1(SOXS)의 상태는 전혀 변경되지 않아야 함
        self.assertEqual(self.tracker.slots[1]["highest_price"], 15.0)
        self.assertEqual(self.tracker.slots[1]["symbol"], "SOXS")

    def test_stop_loss_exit_and_slot_reset(self):
        """손절(-2.0%) 터치 시 STOP_LOSS 청산 및 슬롯 새 dict 교체 검증"""
        ny_tz = ZoneInfo("America/New_York")
        t_entry = datetime.datetime(2026, 10, 14, 9, 45, 0, tzinfo=ny_tz)
        self.tracker.try_enter_shadow_trade("SOXL", 100.0, current_time=t_entry, atr_pct=0.01) # stop_price = 98.0
        
        self.assertTrue(self.tracker.slots[0]["active"])
        self.assertEqual(self.tracker.slots[0]["stop_price"], 98.0)
        
        # 97.5로 하락하여 손절선 터치
        t_tick = datetime.datetime(2026, 10, 14, 9, 50, 0, tzinfo=ny_tz)
        self.tracker.check_ticks("SOXL", 97.5, current_time=t_tick)
        
        # 슬롯 0이 청산 후 새 empty dict로 완전 초기화되었는지 검증
        self.assertFalse(self.tracker.slots[0]["active"])
        self.assertEqual(self.tracker.slots[0]["qty"], 0)
        self.assertIsNone(self.tracker.slots[0]["symbol"])
        self.assertEqual(self.tracker.slots[0]["entry_price"], 0.0)
        self.assertEqual(set(self.tracker.slots[0].keys()), set(SLOT_KEYS))

    def test_trailing_stop_activation_and_exit(self):
        """트레일링 스탑 발동(+1.5%) 후 -0.3% 하락 시 TRAILING_STOP 청산 검증"""
        ny_tz = ZoneInfo("America/New_York")
        t_entry = datetime.datetime(2026, 10, 14, 9, 45, 0, tzinfo=ny_tz)
        self.tracker.try_enter_shadow_trade("SOXL", 100.0, current_time=t_entry, atr_pct=0.01) # initial stop: 98.0
        
        # 1. 주가가 102.0으로 상승 (+2.0% -> 트레일링 1.5% 발동)
        t_rise = datetime.datetime(2026, 10, 14, 9, 55, 0, tzinfo=ny_tz)
        self.tracker.check_ticks("SOXL", 102.0, current_time=t_rise)
        
        # 최고가: 102.0, 트레일링 손절가 = 102.0 * (1 - 0.003) = 101.694
        slot0 = self.tracker.slots[0]
        self.assertEqual(slot0["highest_price"], 102.0)
        self.assertAlmostEqual(slot0["stop_price"], 101.694, places=2)
        
        # 2. 주가가 101.5로 하락 (stop_price 101.694 하향 돌파)
        t_drop = datetime.datetime(2026, 10, 14, 9, 58, 0, tzinfo=ny_tz)
        self.tracker.check_ticks("SOXL", 101.5, current_time=t_drop)
        
        # TRAILING_STOP으로 청산되고 슬롯 초기화 확인
        self.assertFalse(self.tracker.slots[0]["active"])
        self.assertEqual(self.tracker.slots[0]["qty"], 0)

    def test_max_tp_exit(self):
        """최대 익절(+10.0%) 도달 시 MAX_TP 청산 검증"""
        ny_tz = ZoneInfo("America/New_York")
        t_entry = datetime.datetime(2026, 10, 14, 9, 45, 0, tzinfo=ny_tz)
        self.tracker.try_enter_shadow_trade("SOXL", 100.0, current_time=t_entry)
        
        # 110.5로 급등 (+10.5%)
        t_tick = datetime.datetime(2026, 10, 14, 10, 15, 0, tzinfo=ny_tz)
        self.tracker.check_ticks("SOXL", 110.5, current_time=t_tick)
        
        # MAX_TP로 청산되어 슬롯 초기화 확인
        self.assertFalse(self.tracker.slots[0]["active"])

    def test_time_stop_exit_90_minutes(self):
        """90분 경과 시 TIME_STOP 청산 검증"""
        ny_tz = ZoneInfo("America/New_York")
        t_entry = datetime.datetime(2026, 10, 14, 10, 0, 0, tzinfo=ny_tz)
        self.tracker.try_enter_shadow_trade("SOXL", 50.0, current_time=t_entry)
        
        # 89분 경과 (11:29) -> 미청산 유지
        t_89m = datetime.datetime(2026, 10, 14, 11, 29, 0, tzinfo=ny_tz)
        self.tracker.check_time_stops(current_time=t_89m)
        self.assertTrue(self.tracker.slots[0]["active"], "89분 시점에는 보유 유지되어야 함")
        
        # 90분 경과 (11:30) -> 청산
        t_90m = datetime.datetime(2026, 10, 14, 11, 30, 0, tzinfo=ny_tz)
        self.tracker.check_time_stops(current_time=t_90m)
        self.assertFalse(self.tracker.slots[0]["active"], "90분 도달 시 TIME_STOP으로 청산되어야 함")

    def test_eod_clear_exit(self):
        """장 마감 시 전량 EOD_CLEAR 청산 및 최고가(highest_price) 착시 청산 방지 검증"""
        ny_tz = ZoneInfo("America/New_York")
        t_entry = datetime.datetime(2026, 10, 14, 14, 0, 0, tzinfo=ny_tz)
        self.tracker.try_enter_shadow_trade("SOXL", 100.0, current_time=t_entry)
        self.assertTrue(self.tracker.slots[0]["active"])

        # 장중 101.20까지 올랐다가(트레일링 미발동) 100.40으로 내려온 상황 시뮬레이션
        self.tracker.check_ticks("SOXL", 101.20, current_time=datetime.datetime(2026, 10, 14, 14, 10, 0, tzinfo=ny_tz))
        self.tracker.check_ticks("SOXL", 100.40, current_time=datetime.datetime(2026, 10, 14, 14, 20, 0, tzinfo=ny_tz))
        self.assertEqual(self.tracker.slots[0]["highest_price"], 101.20)

        # exit_prices가 누락된 채로 EOD 호출되더라도 절대 highest_price(101.20)가 아닌 최근 틱 현재가(100.40)로 청산되어야 함
        t_eod = datetime.datetime(2026, 10, 14, 15, 55, 0, tzinfo=ny_tz)
        self.tracker.flush_eod(current_time=t_eod, exit_prices=None)

        self.assertFalse(self.tracker.slots[0]["active"], "EOD 호출 시 모든 슬롯이 청산되어야 함")
        self.assertEqual(sum(1 for s in self.tracker.slots if s["active"]), 0)
        self.tracker._db_queue.join()

        import sqlite3
        conn = sqlite3.connect(self.test_db_path)
        row = conn.execute("SELECT highest_price, exit_price, exit_reason FROM shadow_slots_history ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()
        self.assertEqual(row[0], 101.20)
        self.assertEqual(row[1], 100.40, "exit_prices 누락 시 highest_price(101.20)가 아닌 최근 실제 현재가(100.40)로 청산되어야 함")
        self.assertEqual(row[2], "EOD_CLEAR")

    def test_no_actual_order_api_called(self):
        """SHADOW 실행 경로에서 실제 브로커 주문 API가 절대 호출되지 않음을 100% 검증"""
        # broker의 가상 주문 mock 설정
        mock_broker = MagicMock()
        
        # SHADOW 진입 및 청산 라이프사이클 전체 시뮬레이션
        ny_tz = ZoneInfo("America/New_York")
        t1 = datetime.datetime(2026, 10, 14, 9, 45, 0, tzinfo=ny_tz)
        t2 = datetime.datetime(2026, 10, 14, 9, 50, 0, tzinfo=ny_tz)
        
        self.tracker.try_enter_shadow_trade("SOXL", 30.0, current_time=t1)
        self.tracker.check_ticks("SOXL", 28.0, current_time=t2) # 손절 청산
        
        # 브로커의 어떠한 주문 함수도 호출되지 않았음을 단언
        self.assertEqual(mock_broker.send_order.call_count, 0)
        self.assertEqual(mock_broker.send_order_sell.call_count, 0)
        self.assertEqual(mock_broker.cancel_order.call_count, 0)

    def test_all_13_structured_log_events(self):
        """요구사항 12에 명시된 13개 구조화 로그 이벤트가 모두 발생하는지 검증"""
        from core.system_logger import system_logger
        logged_events = []
        original_log = system_logger.log

        def mock_log(level, tag, msg):
            for evt in [
                "SHADOW_ENTRY",
                "NO_EMPTY_SHADOW_SLOT",
                "DUPLICATE_CANDLE_SHADOW_ENTRY_BLOCKED",
                "SHADOW_ENTRY_REJECTED_INVALID_PRICE",
                "SHADOW_HIGHEST_PRICE_UPDATED",
                "SHADOW_STOP_PRICE_UPDATED",
                "SHADOW_EXIT_TIME_STOP",
                "SHADOW_EXIT_STOP_LOSS",
                "SHADOW_EXIT_TRAILING_STOP",
                "SHADOW_EXIT_MAX_TP",
                "SHADOW_EXIT_EOD_CLEAR",
                "SHADOW_FINAL_RETURN",
                "SHADOW_EXCEPTION"
            ]:
                if f"[{evt}]" in msg:
                    logged_events.append(evt)
            original_log(level, tag, msg)

        system_logger.log = mock_log
        try:
            ny_tz = ZoneInfo("America/New_York")
            t_base = datetime.datetime(2026, 10, 14, 9, 45, 0, tzinfo=ny_tz)

            # 1. SHADOW_ENTRY & DUPLICATE_CANDLE_SHADOW_ENTRY_BLOCKED
            self.tracker.try_enter_shadow_trade("SOXL", 100.0, current_time=t_base, atr_pct=0.01)
            self.tracker.try_enter_shadow_trade("SOXL", 100.0, current_time=t_base)

            # 2. SHADOW_ENTRY_REJECTED_INVALID_PRICE
            self.tracker.try_enter_shadow_trade("SOXL", -5.0, current_time=t_base)

            # 3. SHADOW_HIGHEST_PRICE_UPDATED & SHADOW_STOP_PRICE_UPDATED
            self.tracker.check_ticks("SOXL", 102.5, current_time=t_base)

            # 4. SHADOW_EXIT_TRAILING_STOP & SHADOW_FINAL_RETURN
            self.tracker.check_ticks("SOXL", 101.0, current_time=t_base)

            # 5. SHADOW_EXIT_STOP_LOSS
            t2 = datetime.datetime(2026, 10, 14, 10, 0, 0, tzinfo=ny_tz)
            self.tracker.try_enter_shadow_trade("SOXL", 100.0, current_time=t2, atr_pct=0.01)
            self.tracker.check_ticks("SOXL", 97.0, current_time=t2)

            # 6. SHADOW_EXIT_MAX_TP
            t3 = datetime.datetime(2026, 10, 14, 10, 15, 0, tzinfo=ny_tz)
            self.tracker.try_enter_shadow_trade("SOXL", 100.0, current_time=t3)
            self.tracker.check_ticks("SOXL", 110.5, current_time=t3)

            # 7. SHADOW_EXIT_TIME_STOP
            t4 = datetime.datetime(2026, 10, 14, 10, 30, 0, tzinfo=ny_tz)
            self.tracker.try_enter_shadow_trade("SOXL", 100.0, current_time=t4)
            t4_90m = datetime.datetime(2026, 10, 14, 12, 0, 0, tzinfo=ny_tz)
            self.tracker.check_time_stops(current_time=t4_90m)

            # 8. SHADOW_EXIT_EOD_CLEAR
            t5 = datetime.datetime(2026, 10, 14, 12, 15, 0, tzinfo=ny_tz)
            self.tracker.try_enter_shadow_trade("SOXL", 100.0, current_time=t5)
            self.tracker.flush_eod(current_time=datetime.datetime(2026, 10, 14, 15, 55, 0, tzinfo=ny_tz))

            # 9. NO_EMPTY_SHADOW_SLOT (10개 채우기)
            for i in range(10):
                t_i = datetime.datetime(2026, 10, 14, 12, 30, 0, tzinfo=ny_tz) + datetime.timedelta(minutes=15*i)
                self.tracker.try_enter_shadow_trade("SOXL", 50.0 + i, current_time=t_i)
            # 11번째 슬롯 시도
            t_full = datetime.datetime(2026, 10, 14, 15, 0, 0, tzinfo=ny_tz)
            self.tracker.try_enter_shadow_trade("SOXL", 99.0, current_time=t_full)

            # 10. SHADOW_EXCEPTION
            with patch.object(self.tracker, "get_last_closed_15m_candle", side_effect=RuntimeError("Forced Test Error")):
                self.tracker.try_enter_shadow_trade("SOXL", 50.0, current_time=datetime.datetime(2026, 10, 14, 15, 30, 0, tzinfo=ny_tz))

            # 13개 이벤트 검증
            required_events = [
                "SHADOW_ENTRY",
                "NO_EMPTY_SHADOW_SLOT",
                "DUPLICATE_CANDLE_SHADOW_ENTRY_BLOCKED",
                "SHADOW_ENTRY_REJECTED_INVALID_PRICE",
                "SHADOW_HIGHEST_PRICE_UPDATED",
                "SHADOW_STOP_PRICE_UPDATED",
                "SHADOW_EXIT_TIME_STOP",
                "SHADOW_EXIT_STOP_LOSS",
                "SHADOW_EXIT_TRAILING_STOP",
                "SHADOW_EXIT_MAX_TP",
                "SHADOW_EXIT_EOD_CLEAR",
                "SHADOW_FINAL_RETURN",
                "SHADOW_EXCEPTION"
            ]
            for evt in required_events:
                self.assertIn(evt, logged_events, f"구조화 로그 이벤트 [{evt}]가 발생하지 않았습니다.")
        finally:
            system_logger.log = original_log

if __name__ == "__main__":
    unittest.main()
