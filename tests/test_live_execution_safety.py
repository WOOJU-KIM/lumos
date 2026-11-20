import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.kiwoom_ws_streamer import KiwoomWebSocketStreamer
from core.live_runner import KiwoomLiveRunner
from core.telegram_notifier import TelegramNotifier

class TestLiveExecutionSafety(unittest.TestCase):
    def test_1_websocket_live_tick_tracking(self):
        """[검증 1] WebSocket 실시간 틱 수신 추적 및 게이팅 검증"""
        streamer = KiwoomWebSocketStreamer()
        streamer.is_connected = True
        
        # 초기 상태: 틱 수신 전이므로 False
        self.assertFalse(streamer.has_received_live_tick("TQQQ"))
        self.assertFalse(streamer.is_live_stream_active("TQQQ"))

        # REST 백업 틱은 ws_tick_received를 True로 만들지 않아야 함
        streamer._update_price("TQQQ", 100.5, {"provider": "REST_FALLBACK"}, from_ws=False)
        self.assertFalse(streamer.has_received_live_tick("TQQQ"))

        # 실제 WebSocket 틱 유입 시 True로 전환
        streamer._update_price("TQQQ", 100.8, {"provider": "WS_0A"}, from_ws=True)
        self.assertTrue(streamer.has_received_live_tick("TQQQ"))
        self.assertTrue(streamer.is_live_stream_active("TQQQ"))
        self.assertEqual(streamer.ws_tick_count["TQQQ"], 1)
        print("✅ [Test 1 통과] WebSocket 실시간 틱 게이팅 및 REST 오인 방지 검증 완료")

    def test_2_ledger_entry_price_synchronization(self):
        """[검증 2] 증권사 원장 실사 매입단가 동기화 검증"""
        runner = KiwoomLiveRunner(is_simulation=True)
        # Mock broker balance with actual execution price $101.50
        mock_bal = {
            "ok": True,
            "holdings": [
                {
                    "symbol": "TQQQ",
                    "quantity": 467,
                    "purchase_price": 101.50,
                    "avg_price": 101.50,
                    "frgn_stk_book_uv": 101.50
                }
            ]
        }
        runner.broker.get_overseas_stock_balance = MagicMock(return_value=mock_bal)
        
        # Order was sent at $121.85, but broker ledger filled at $101.50
        synced_px, synced_qty = runner._sync_real_ledger_entry(
            symbol="TQQQ",
            default_price=121.85,
            default_qty=577
        )
        self.assertEqual(synced_px, 101.50)
        self.assertEqual(synced_qty, 467)
        print("✅ [Test 2 통과] 주문단가($121.85) ➔ 원장 실제 매입단가($101.50) 동기화 검증 완료")

    def test_3_alert_mode_title_transparency(self):
        """[검증 3] 텔레그램 청산 및 매수 알림 헤더에 모의/실전 명확 표기 검증"""
        notifier = TelegramNotifier()
        dispatched_messages = []
        notifier._dispatch_message = lambda text, alert_type: dispatched_messages.append((text, alert_type))

        # 모의투자 청산 알림
        notifier.send_exit_alert(
            ticker="TQQQ",
            entry_price=101.50,
            exit_price=101.65,
            exit_reason="🛑 칼손절 방어 (-16.58%)",
            is_simulation=True
        )
        exit_msg, _ = dispatched_messages[-1]
        self.assertIn("[키움 모의투자]", exit_msg)
        self.assertNotIn("[키움 실전투자]", exit_msg)

        # 실전투자 청산 알림
        notifier.send_exit_alert(
            ticker="TQQQ",
            entry_price=101.50,
            exit_price=101.65,
            exit_reason="🎯 목표 익절 (+3.0%)",
            is_simulation=False
        )
        exit_real_msg, _ = dispatched_messages[-1]
        self.assertIn("[키움 실전투자]", exit_real_msg)
        self.assertNotIn("[키움 모의투자]", exit_real_msg)

        print("✅ [Test 3 통과] 텔레그램 청산 알림 헤더 [모의투자]/[실전투자] 투명 표기 검증 완료")

    def test_4_ft_packet_quote_parsing(self):
        """[검증 4] Kiwoom WebSocket FT(해외주식 10호가) 패킷 정상 시세 인식 및 틱 활성화 검증"""
        import json
        streamer = KiwoomWebSocketStreamer()
        streamer.is_connected = True

        raw_ft = json.dumps({
            "trnm": "REAL",
            "data": [
                {
                    "type": "FT",
                    "item": "TQQQ",
                    "values": {
                        "21": "095700",
                        "41": "-101.5000",
                        "51": "-101.4000",
                        "61": "100",
                        "71": "200"
                    }
                }
            ]
        })

        # FT 패킷 처리 전
        self.assertFalse(streamer.has_received_live_tick("TQQQ"))

        # FT 패킷 처리 후: 체결 통보로 잘못 스킵되지 않고 즉시 틱 수신 True 및 호가 반영
        streamer._process_message(raw_ft)
        self.assertTrue(streamer.has_received_live_tick("TQQQ"))
        self.assertEqual(streamer.get_latest_price("TQQQ"), 101.45)
        print("✅ [Test 4 통과] FT 호가 패킷 시세 정상 파싱 및 WebSocket 틱 활성화 검증 완료")

if __name__ == "__main__":
    unittest.main()

