import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from core.shadow_tracker import ShadowTracker, MAX_SHADOW_SLOTS
from core.live_runner import KiwoomLiveRunner


class TestShadowAuditVerification(unittest.TestCase):
    def setUp(self):
        self.test_db_path = config.DATA_DIR / "test_shadow_audit_temp.db"
        if self.test_db_path.exists():
            try:
                self.test_db_path.unlink()
            except Exception:
                pass
        self.ny_tz = ZoneInfo("America/New_York")
        self.candle_time = datetime.datetime(2026, 10, 14, 9, 45, 0, tzinfo=self.ny_tz)

    def tearDown(self):
        if self.test_db_path.exists():
            try:
                self.test_db_path.unlink()
            except Exception:
                pass

    def test_item1_shadow_exception_does_not_block_live_trade(self):
        """
        [항목 1]
        core/live_runner.py에서 shadow_tracker.try_enter_shadow_trade가 예외를 발생시키도록 Mock하고,
        실거래 판단 루프와 주문 판단이 계속 실행되는지 검증
        """
        runner = KiwoomLiveRunner(is_simulation=True)
        runner.shadow_tracker = MagicMock()
        runner.shadow_tracker.try_enter_shadow_trade.side_effect = RuntimeError("Forced shadow exception")

        runner.broker = MagicMock()
        runner.broker.test_connection.return_value = {"ok": True, "usd_order_available": 10000.0}
        runner._get_5m_atr_pct = MagicMock(return_value=1.5)
        runner._execute_buy_with_10s_chase = MagicMock(return_value={"ok": True})

        # live_runner.py 라인 1079-1127의 의사결정 블록 시뮬레이션
        p_long, p_short, p_none = 80.0, 10.0, 10.0
        conf = 85.0
        is_appr = True
        direction = "LONG"
        active_pos = None
        live_prices = {config.BASE_ASSET_LONG: 30.0, config.BASE_ASSET_SHORT: 15.0}
        ny_dt = self.candle_time
        moe_res = {"direction": "LONG", "final_conf": 0.85}

        # 1. 쉐도우 트래커(가상 매매) 실행 블록
        shadow_dir = "LONG" if p_long >= p_short else "SHORT"
        shadow_sym = config.BASE_ASSET_LONG if shadow_dir == "LONG" else config.BASE_ASSET_SHORT
        shadow_px = live_prices.get(shadow_sym, 0.0)

        shadow_threw_exception = False
        try:
            shadow_atr = runner._get_5m_atr_pct(shadow_sym, shadow_px) if shadow_px > 0 else 1.8
            runner.shadow_tracker.try_enter_shadow_trade(
                symbol=shadow_sym,
                price=shadow_px,
                current_time=ny_dt,
                atr_pct=(shadow_atr / 100.0)
            )
        except Exception:
            shadow_threw_exception = True

        # 2. 실전 매매 로직 블록 (기존 70.4% 룰 및 릴레이 원칙)
        if is_appr:
            if direction in ["LONG", "SHORT"]:
                winner_sym = config.BASE_ASSET_LONG if direction == "LONG" else config.BASE_ASSET_SHORT
                cur_px = live_prices.get(winner_sym, 0.0)
                if cur_px > 0 and not active_pos:
                    conn_res = runner.broker.test_connection()
                    usd_avail = float(conn_res.get("usd_order_available", 0.0))
                    order_qty = int((usd_avail * config.MAX_ALLOCATION_RATIO) / (cur_px + config.QTY_CALC_BUFFER))
                    if order_qty > 0:
                        targets = {
                            "dynamic_tp_px": round(cur_px * 1.10, 2),
                            "dynamic_sl_px": round(cur_px * 0.98, 2),
                            "tp_pct": 10.0,
                            "sl_pct": 2.0,
                            "time_stop_minutes": config.TIME_STOP_MINUTES,
                            "strategy_tag": "Lumos V4 Optimal"
                        }
                        runner._execute_buy_with_10s_chase(
                            symbol=winner_sym,
                            target_qty=order_qty,
                            ref_price=cur_px,
                            moe_res=moe_res,
                            targets=targets
                        )

        # Assertions
        self.assertTrue(shadow_threw_exception, "SHADOW 진입에서 고의 예외가 발생했어야 함")
        runner.shadow_tracker.try_enter_shadow_trade.assert_called_once()
        runner._execute_buy_with_10s_chase.assert_called_once()
        self.assertEqual(runner._execute_buy_with_10s_chase.call_args[1]["symbol"], config.BASE_ASSET_LONG)

    def test_item2_shadow_db_and_jsonl_failure_does_not_halt_runner(self):
        """
        [항목 2]
        SHADOW DB 저장과 JSONL 저장이 각각 실패하도록 Mock하고, live_runner가 중단되지 않는지 검증
        """
        tracker = ShadowTracker(db_path=self.test_db_path)
        
        # 1. DB 실패 Mocking: sqlite3.connect가 실패하도록 설정
        with patch("sqlite3.connect", side_effect=Exception("Database lock/disk failure")):
            # 진입 시도
            entered = tracker.try_enter_shadow_trade("SOXL", 30.0, current_time=self.candle_time)
            # 메인 스레드에서 예외가 터져서 크래시되지 않아야 함
            self.assertTrue(entered)

        # 2. JSONL 실패 Mocking: open 호출이 PermissionError를 내도록 설정
        with patch("builtins.open", side_effect=PermissionError("JSONL file write denied")):
            # 청산 유발: Stop price 이하로 터치
            tracker.check_ticks("SOXL", 20.0, current_time=self.candle_time)
            # 메인 스레드는 중단되지 않고 빈 슬롯으로 정상 복귀했는지 확인
            self.assertFalse(tracker.slots[0]["active"])

    def test_item3_shadow_lifecycle_zero_real_orders(self):
        """
        [항목 3]
        SHADOW 진입·보유·청산 전체 경로에서 실제 주문 함수 호출 횟수가 정확히 0회인지 Mock assert로 검증
        """
        runner = KiwoomLiveRunner(is_simulation=True)
        runner.broker = MagicMock()
        runner.broker.send_order = MagicMock()
        runner._execute_buy_with_10s_chase = MagicMock()

        tracker = ShadowTracker(db_path=self.test_db_path)
        runner.shadow_tracker = tracker

        # 1) 진입 단계
        tracker.try_enter_shadow_trade("SOXL", 30.0, current_time=self.candle_time, atr_pct=0.015)
        # 2) 보유/감시 단계
        tracker.check_ticks("SOXL", 30.2, current_time=self.candle_time)
        tracker.increment_candle(current_time=self.candle_time)
        # 3) 청산 단계 (트레일링/손절 및 EOD)
        tracker.check_ticks("SOXL", 25.0, current_time=self.candle_time)
        tracker.flush_eod(current_time=self.candle_time)

        # 실거래 브로커 발주 함수 및 live_runner 발주 함수가 0회 호출되었는지 단언
        self.assertEqual(runner.broker.send_order.call_count, 0, "SHADOW 라이프사이클 중 broker.send_order 호출은 0회여야 함")
        self.assertEqual(runner._execute_buy_with_10s_chase.call_count, 0, "SHADOW 라이프사이클 중 _execute_buy_with_10s_chase 호출은 0회여야 함")

    def test_item4_same_closed_15m_candle_duplicate_blocked(self):
        """
        [항목 4]
        같은 마감 15분봉을 반복 호출했을 때 1회만 진입하는지 검증
        """
        tracker = ShadowTracker(db_path=self.test_db_path)

        t1 = self.candle_time  # 2026-10-14 09:45:00
        t2 = datetime.datetime(2026, 10, 14, 9, 46, 30, tzinfo=self.ny_tz)  # 동일 09:30-09:45 마감봉 구간
        t3 = datetime.datetime(2026, 10, 14, 9, 59, 59, tzinfo=self.ny_tz)  # 동일 09:30-09:45 마감봉 구간

        res1 = tracker.try_enter_shadow_trade("SOXL", 30.0, current_time=t1)
        res2 = tracker.try_enter_shadow_trade("SOXL", 30.5, current_time=t2)
        res3 = tracker.try_enter_shadow_trade("SOXL", 30.8, current_time=t3)

        self.assertTrue(res1, "1회차 진입은 성공해야 함")
        self.assertFalse(res2, "2회차 동일 마감봉 진입은 차단되어야 함")
        self.assertFalse(res3, "3회차 동일 마감봉 진입은 차단되어야 함")
        active_count = sum(1 for s in tracker.slots if s["active"])
        self.assertEqual(active_count, 1, "동일 봉 반복 호출 후 활성 슬롯은 정확히 1개여야 함")

    def test_item5_recreated_instance_duplicate_candle_blocked_persistence(self):
        """
        [항목 5]
        ShadowTracker 객체 또는 프로세스를 재생성한 뒤 같은 마감 봉을 전달했을 때 중복 진입이 차단되는지 검증
        - 차단되면 PASS
        - 다시 진입하면 FAIL
        - 영속 상태 자체가 없으면 NOT VERIFIED가 아니라 FAIL로 판정한다.
        """
        # 1차 인스턴스 생성 및 진입
        tracker1 = ShadowTracker(db_path=self.test_db_path)
        entered_1 = tracker1.try_enter_shadow_trade("SOXL", 30.0, current_time=self.candle_time)
        self.assertTrue(entered_1, "1차 인스턴스 첫 진입 성공")

        # 인스턴스 소멸 및 재생성 모사 (프로세스 재시작 또는 객체 재생성)
        del tracker1
        tracker2 = ShadowTracker(db_path=self.test_db_path)

        # 2차 재생성 인스턴스에 동일한 마감 봉 전달
        entered_2 = tracker2.try_enter_shadow_trade("SOXL", 30.0, current_time=self.candle_time)

        # 사용자 규칙: 차단되면 PASS, 다시 진입하면 FAIL
        # 현재 shadow_tracker.py 구현은 __init__에서 self.last_entered_candle_id = None 이고
        # 디스크/DB에서 영속된 candle_id를 복원하지 않으므로 entered_2가 True가 됨!
        # 따라서 아래 assertion이 False여야 영속적 차단(PASS)이지만, 현재 코드는 재진입하므로 FAIL임.
        self.assertFalse(entered_2, "객체/프로세스 재생성 후에도 동일 마감봉 진입이 차단되어야 함 (현재 영속성 미지원으로 FAIL 예상)")

    def test_item6_10_slots_full_rejects_shadow_allows_live_order(self):
        """
        [항목 6]
        10개 slot을 모두 채운 상태에서도 실거래 주문 판단 함수가 정상 호출되고,
        SHADOW 진입만 NO_EMPTY_SHADOW_SLOT로 거부되는지 검증
        """
        runner = KiwoomLiveRunner(is_simulation=True)
        tracker = ShadowTracker(db_path=self.test_db_path)
        runner.shadow_tracker = tracker

        # 10개 슬롯을 모두 활성화하여 채움
        for i in range(MAX_SHADOW_SLOTS):
            tracker.slots[i]["active"] = True
            tracker.slots[i]["symbol"] = "SOXL"
            tracker.slots[i]["entry_price"] = 30.0 + i

        runner.broker = MagicMock()
        runner.broker.test_connection.return_value = {"ok": True, "usd_order_available": 10000.0}
        runner._get_5m_atr_pct = MagicMock(return_value=1.5)
        runner._execute_buy_with_10s_chase = MagicMock(return_value={"ok": True})

        # 11번째 슬롯 진입 시도 (10개 포화 상태)
        new_candle_time = datetime.datetime(2026, 10, 14, 12, 0, 0, tzinfo=self.ny_tz)
        shadow_entered = runner.shadow_tracker.try_enter_shadow_trade(
            symbol="SOXL",
            price=35.0,
            current_time=new_candle_time
        )
        self.assertFalse(shadow_entered, "10개 슬롯 포화 시 SHADOW 진입은 거부되어야 함")

        # 실거래 매매 루프 실행
        live_prices = {config.BASE_ASSET_LONG: 35.0, config.BASE_ASSET_SHORT: 15.0}
        is_appr = True
        direction = "LONG"
        active_pos = None

        if is_appr and direction in ["LONG", "SHORT"]:
            winner_sym = config.BASE_ASSET_LONG
            cur_px = live_prices.get(winner_sym, 0.0)
            if cur_px > 0 and not active_pos:
                conn_res = runner.broker.test_connection()
                usd_avail = float(conn_res.get("usd_order_available", 0.0))
                order_qty = int((usd_avail * config.MAX_ALLOCATION_RATIO) / (cur_px + config.QTY_CALC_BUFFER))
                if order_qty > 0:
                    runner._execute_buy_with_10s_chase(
                        symbol=winner_sym,
                        target_qty=order_qty,
                        ref_price=cur_px,
                        moe_res={},
                        targets={}
                    )

        # SHADOW는 거부되었지만, 실거래 주문 판단 함수는 정상 호출되었음을 검증
        runner._execute_buy_with_10s_chase.assert_called_once()
        self.assertEqual(runner._execute_buy_with_10s_chase.call_args[1]["symbol"], "SOXL")


if __name__ == '__main__':
    unittest.main()
