import datetime
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd

import config
from core.data_lake import MarketDataLake
from core.live_runner import KiwoomLiveRunner
from core.moe_orchestrator import MoEMetaOrchestrator

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestB05PreMarketSignalAndTransition(unittest.TestCase):
    """
    B-05 변경: 실거래 Live Runner 가동 및 프리마켓 AI 신호 활성화 전수 검증 스위트
    - 시나리오 1 ~ 10 전수 검증
    - 프리마켓 시세/캔들 데이터 무결성 및 DB 비오염 검증
    - 실계좌 주문 0건 보장
    """

    _cached_moe = None
    _cached_df_15m = None

    @classmethod
    def setUpClass(cls):
        cls._cached_moe = MoEMetaOrchestrator(
            confidence_threshold=config.GBDT_CONFIDENCE_THRESHOLD,
            gbdt_threshold=config.GBDT_CONFIDENCE_THRESHOLD,
        )
        lake = MarketDataLake()
        cls._cached_df_15m = lake.load_candles(config.BASE_ASSET_LONG, "15m").tail(120)

    def setUp(self):
        self.tmp_dir_obj = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.tmp_dir = Path(self.tmp_dir_obj.name)
        self.sent_notifier_msgs = []
        self.sent_dispatcher_msgs = []

    def tearDown(self):
        self.tmp_dir_obj.cleanup()

    def _build_isolated_runner(self):
        """실계좌 주문이 100% 차단된 격리 KiwoomLiveRunner 생성"""
        broker_patcher = patch("core.live_runner.KiwoomBroker")
        disp_patcher = patch("core.live_runner.DispatcherAgent")
        ctrl_patcher = patch("core.live_runner.TelegramController")
        http_patcher = patch("core.telegram_notifier.TelegramNotifier._send_http_request")
        moe_patcher = patch("core.live_runner.MoEMetaOrchestrator", return_value=self._cached_moe)
        cb_patcher = patch("core.circuit_breaker.CIRCUIT_STATE_FILE", self.tmp_dir / "isolated_cb.json")

        self.addCleanup(broker_patcher.stop)
        self.addCleanup(disp_patcher.stop)
        self.addCleanup(ctrl_patcher.stop)
        self.addCleanup(http_patcher.stop)
        self.addCleanup(moe_patcher.stop)
        self.addCleanup(cb_patcher.stop)

        MockBroker = broker_patcher.start()
        MockDisp = disp_patcher.start()
        ctrl_patcher.start()
        mock_http = http_patcher.start()
        moe_patcher.start()
        cb_patcher.start()

        broker_inst = MockBroker.return_value
        broker_inst.is_simulation = False
        broker_inst.broker_name = "키움증권"
        broker_inst.mode_str = "실전투자"
        broker_inst.account_no = "81234567"
        broker_inst.account_type = "01"
        broker_inst.test_connection.return_value = {
            "ok": True,
            "deposit_ok": True,
            "stock_ok": True,
            "usd_order_available": 5000.0,
            "holdings_count": 0,
        }
        broker_inst.get_account_deposit.return_value = {
            "ok": True,
            "usd_order_available": 5000.0,
        }
        broker_inst.get_overseas_stock_balance.return_value = {
            "ok": True,
            "holdings": [],
            "usd_order_available": 5000.0,
        }
        broker_inst.get_open_orders.return_value = {"ok": True, "open_orders": []}
        broker_inst.get_daily_execution_history.return_value = {"ok": True, "executions": []}
        broker_inst.get_stock_quote.side_effect = lambda sym: {
            "ok": True,
            "symbol": sym,
            "last_price": 100.01 if sym == "SOXL" else 46.56,
            "quote_time_et": "2026-10-15 07:15 ET",
            "quote_session": "PRE_MARKET",
            "quote_age_sec": 12.0,
            "is_previous_session_quote": False,
            "is_fresh": True,
            "stale_reason": "",
            "source": "yfinance_1m_prepost",
        }

        disp_inst = MockDisp.return_value
        disp_inst.send_telegram_message.side_effect = lambda msg: (
            self.sent_dispatcher_msgs.append(msg) or {"ok": True}
        )
        mock_http.side_effect = lambda msg: (
            self.sent_notifier_msgs.append(msg) or {"ok": True}
        )

        runner = KiwoomLiveRunner(is_simulation=False)
        runner.pos_file_path = str(self.tmp_dir / "isolated_active_position.json")
        runner._active_position = None
        runner.data_lake.load_candles = MagicMock(return_value=self._cached_df_15m)
        runner.notifier.bot_token = "DUMMY_TOKEN"
        runner.notifier.chat_id = "DUMMY_CHAT"
        runner.notifier.async_mode = False
        return runner, broker_inst

    def _mock_moe_decision(
        self,
        direction="NONE",
        dir_gbdt=None,
        conf=0.512,
        is_approved=False,
        is_cross_veto=False,
        is_60m_trend_ok=True,
        dip_ok=True,
        probs=None,
    ):
        if dir_gbdt is None:
            dir_gbdt = direction
        if probs is None:
            if direction == "LONG" or dir_gbdt == "LONG":
                probs = {"LONG": 0.62, "SHORT": 0.18, "NONE": 0.20}
            elif direction == "SHORT" or dir_gbdt == "SHORT":
                probs = {"LONG": 0.15, "SHORT": 0.65, "NONE": 0.20}
            else:
                probs = {"LONG": 0.25, "SHORT": 0.25, "NONE": 0.50}
        return {
            "selected_expert": "gbdt_pattern",
            "expert_desc": "GBDT_72_B_Normalized",
            "direction": direction,
            "dir_gbdt": dir_gbdt,
            "cross_dir": direction,
            "is_approved": is_approved,
            "is_cross_veto": is_cross_veto,
            "is_60m_trend_ok": is_60m_trend_ok,
            "dip_ok": dip_ok,
            "gating_confidence": conf,
            "expert_confidence": conf,
            "threshold_applied": float(config.GBDT_CONFIDENCE_THRESHOLD),
            "gbdt_probs": probs,
            "all_gating_confidences": {"cross_asset": conf, "gbdt_pattern": conf},
            "regime_snapshot": {},
            "features": {},
            "active_model_id": "B_Normalized_72",
            "active_model_name": "B_Normalized_72",
            "active_model_schema": "B_normalized",
            "active_model_sha256": "ef954c3be286d4fd4e429c5ed21b7e3da4842927c53e594ada86460029a0dcae",
            "active_model_version": "ef954c3be286",
            "active_model_num_features": 72,
            "active_model_load_status": "ACTIVE_B_NORMALIZED",
            "active_model_fallback_status": "PRIMARY_ACTIVE",
            "active_model_is_fallback": False,
            "model_hot_reloaded": False,
            "model_consistency_ok": True,
            "model_consistency_reason": "CONSISTENT",
            "timestamp": "2026-10-15 07:15:00",
        }

    def test_01_scenario_1_premarket_none_signal(self):
        """시나리오 1: PRE_MARKET_WAITING + NONE 신호 -> 텔레그램 전송, 주문 0건"""
        runner, broker = self._build_isolated_runner()
        pre_dt = datetime.datetime(2026, 10, 15, 16, 15, 0, tzinfo=ZoneInfo("America/New_York"))

        with patch.object(
            runner.moe_orchestrator,
            "evaluate_dual_filter_signal",
            return_value=self._mock_moe_decision(direction="NONE", conf=0.521, is_approved=False),
        ):
            res = runner.evaluate_15m_cycle_once(
                allow_order_execution=True,  # True를 넘겨도 프리마켓 하드 인터락이 강제 차단해야 함
                sync_intraday=False,
                now_dt=pre_dt,
                market_session="PRE_MARKET_WAITING",
            )

        self.assertEqual(res["market_state"], "PRE_MARKET")
        self.assertEqual(res["session_state"], "PRE_MARKET_WAITING")
        self.assertFalse(res["order_allowed"])
        self.assertEqual(res["order_block_reason"], "PRE_MARKET")
        self.assertFalse(res["order_executed"])
        self.assertEqual(res["premarket_policy"], "NONE")
        self.assertEqual(broker.send_order.call_count, 0)

        self.assertEqual(len(self.sent_dispatcher_msgs), 1)
        msg = self.sent_dispatcher_msgs[0]
        self.assertIn("[ LUMOS AI SIGNAL ]", msg)
        self.assertIn("시장 상태: PRE_MARKET (PRE_MARKET_WAITING)", msg)
        self.assertIn("운영 모드: REAL", msg)
        self.assertIn("활성 모델: B_Normalized_72", msg)
        self.assertIn("AI 판단: NONE", msg)
        self.assertIn("주문 가능 여부: BLOCKED", msg)
        self.assertIn("주문 차단 사유: PRE_MARKET", msg)
        self.assertIn("실제 주문 실행 여부: NO", msg)
        self.assertIn("프리마켓 정책: NONE", msg)

    def test_02_scenario_2_premarket_long_signal_above_threshold_blocked(self):
        """시나리오 2: PRE_MARKET_WAITING + LONG 신호 (확신도 >= 70.4%) -> SIGNAL_ONLY — NO ORDER, 주문 0건"""
        runner, broker = self._build_isolated_runner()
        pre_dt = datetime.datetime(2026, 10, 15, 16, 15, 0, tzinfo=ZoneInfo("America/New_York"))

        with patch.object(
            runner.moe_orchestrator,
            "evaluate_dual_filter_signal",
            return_value=self._mock_moe_decision(direction="LONG", conf=0.752, is_approved=True),
        ), patch.object(runner, "_execute_buy_with_10s_chase") as mock_buy:
            res = runner.evaluate_15m_cycle_once(
                allow_order_execution=True,
                sync_intraday=False,
                now_dt=pre_dt,
                market_session="PRE_MARKET_WAITING",
            )

        self.assertFalse(res["order_allowed"])
        self.assertEqual(res["order_block_reason"], "PRE_MARKET")
        self.assertFalse(res["order_executed"])
        self.assertEqual(res["premarket_policy"], "SIGNAL_ONLY — NO ORDER")
        self.assertEqual(mock_buy.call_count, 0)
        self.assertEqual(broker.send_order.call_count, 0)

        msg = self.sent_dispatcher_msgs[-1]
        self.assertIn("[ LUMOS AI SIGNAL ]", msg)
        self.assertIn("시장 상태: PRE_MARKET (PRE_MARKET_WAITING)", msg)
        self.assertIn("AI 판단: LONG", msg)
        self.assertIn("확신도: 75.2%", msg)
        self.assertIn(f"진입 임계값: {config.GBDT_CONFIDENCE_THRESHOLD*100:.1f}%", msg)
        self.assertIn("주문 가능 여부: BLOCKED", msg)
        self.assertIn("주문 차단 사유: PRE_MARKET", msg)
        self.assertIn("실제 주문 실행 여부: NO", msg)
        self.assertIn("SIGNAL_ONLY — NO ORDER", msg)

    def test_03_scenario_3_premarket_short_signal_above_threshold_blocked(self):
        """시나리오 3: PRE_MARKET_WAITING + SHORT 신호 (확신도 >= 70.4%) -> SIGNAL_ONLY — NO ORDER, 주문 0건"""
        runner, broker = self._build_isolated_runner()
        pre_dt = datetime.datetime(2026, 10, 15, 16, 15, 0, tzinfo=ZoneInfo("America/New_York"))

        with patch.object(
            runner.moe_orchestrator,
            "evaluate_dual_filter_signal",
            return_value=self._mock_moe_decision(direction="SHORT", conf=0.781, is_approved=True),
        ), patch.object(runner, "_execute_buy_with_10s_chase") as mock_buy:
            res = runner.evaluate_15m_cycle_once(
                allow_order_execution=True,
                sync_intraday=False,
                now_dt=pre_dt,
                market_session="PRE_MARKET_WAITING",
            )

        self.assertFalse(res["order_allowed"])
        self.assertEqual(res["order_block_reason"], "PRE_MARKET")
        self.assertFalse(res["order_executed"])
        self.assertEqual(res["premarket_policy"], "SIGNAL_ONLY — NO ORDER")
        self.assertEqual(mock_buy.call_count, 0)
        self.assertEqual(broker.send_order.call_count, 0)

        msg = self.sent_dispatcher_msgs[-1]
        self.assertIn("AI 판단: SHORT", msg)
        self.assertIn("확신도: 78.1%", msg)
        self.assertIn("주문 가능 여부: BLOCKED", msg)
        self.assertIn("주문 차단 사유: PRE_MARKET", msg)
        self.assertIn("SIGNAL_ONLY — NO ORDER", msg)

    def test_04_scenario_4_premarket_stale_data_suppresses_signal(self):
        """시나리오 4: PRE_MARKET_WAITING + Stale Data (전일 종가 또는 지연 시세) -> SIGNAL_UNAVAILABLE_STALE_DATA 및 추론 억제"""
        runner, broker = self._build_isolated_runner()
        pre_dt = datetime.datetime(2026, 10, 15, 16, 15, 0, tzinfo=ZoneInfo("America/New_York"))

        broker.get_stock_quote.side_effect = lambda sym: {
            "ok": False,
            "symbol": sym,
            "last_price": 0.0,
            "quote_time_et": "2026-10-14 16:00 ET",
            "quote_session": "REGULAR_MARKET",
            "quote_age_sec": 57600.0,
            "is_previous_session_quote": True,
            "is_fresh": False,
            "stale_reason": "PREVIOUS_SESSION_QUOTE_REJECTED(2026-10-14 16:00 ET)",
        }

        with patch.object(
            runner.moe_orchestrator, "evaluate_dual_filter_signal"
        ) as mock_eval, patch.object(runner, "_execute_buy_with_10s_chase") as mock_buy:
            res = runner.evaluate_15m_cycle_once(
                allow_order_execution=True,
                sync_intraday=False,
                now_dt=pre_dt,
                market_session="PRE_MARKET_WAITING",
            )

        # 신뢰할 수 없는 상태에서 모델 추론이 호출되지 않고 억제되었는지 확인
        self.assertEqual(mock_eval.call_count, 0)
        self.assertEqual(mock_buy.call_count, 0)
        self.assertTrue(res["data_stale"])
        self.assertFalse(res["order_allowed"])
        self.assertEqual(res["order_block_reason"], "DATA_STALE")
        self.assertEqual(res["premarket_policy"], "SIGNAL_UNAVAILABLE_STALE_DATA")

        msg = self.sent_dispatcher_msgs[-1]
        self.assertIn("[ LUMOS AI SIGNAL ]", msg)
        self.assertIn("주문 가능 여부: BLOCKED", msg)
        self.assertIn("주문 차단 사유: DATA_STALE", msg)
        self.assertIn("SIGNAL_UNAVAILABLE_STALE_DATA", msg)

    def test_05_scenario_5_auto_transition_premarket_to_regular_open(self):
        """시나리오 5: PRE_MARKET_WAITING -> REGULAR_MARKET_OPEN 무재기동 자동 전환"""
        runner, broker = self._build_isolated_runner()

        statuses = [
            {
                "is_open": False,
                "session_name": "PRE_MARKET_WAITING",
                "status_desc": "프리마켓 대기 중",
                "now_kst_str": "2026-10-15 22:29:50 KST",
                "now_ny_str": "2026-10-15 09:29:50 EDT",
                "dst_text": "서머타임 적용",
            },
            {
                "is_open": True,
                "session_name": "REGULAR_MARKET_OPEN",
                "status_desc": "미국 정규장 오픈",
                "now_kst_str": "2026-10-15 22:30:01 KST",
                "now_ny_str": "2026-10-15 09:30:01 EDT",
                "dst_text": "서머타임 적용",
            },
        ]
        state = {"idx": 0}
        observed_calls = []

        def _fake_mkt_status(*args, **kwargs):
            idx = min(state["idx"], len(statuses) - 1)
            return statuses[idx]

        def _fake_eval_15m(allow_order_execution=True, sync_intraday=True, now_dt=None, market_session=None):
            observed_calls.append(
                {
                    "market_session": market_session,
                    "allow_order_execution": allow_order_execution,
                    "sync_intraday": sync_intraday,
                }
            )
            return {"ok": True}

        def _fake_sleep(_sec):
            state["idx"] += 1
            if state["idx"] >= 2:
                raise StopIteration("Loop test complete")

        with patch("core.live_runner.USMarketCalendar.get_market_status", side_effect=_fake_mkt_status), \
             patch.object(runner, "evaluate_15m_cycle_once", side_effect=_fake_eval_15m), \
             patch("core.live_runner.time.sleep", side_effect=_fake_sleep):
            with self.assertRaises(StopIteration):
                runner._market_execution_loop()

        self.assertEqual(len(observed_calls), 2)
        # 1틱: PRE_MARKET_WAITING (allow_order_execution=False, sync_intraday=False)
        self.assertEqual(observed_calls[0]["market_session"], "PRE_MARKET_WAITING")
        self.assertFalse(observed_calls[0]["allow_order_execution"])
        self.assertFalse(observed_calls[0]["sync_intraday"])
        # 2틱: 09:30 EDT 진입 시 프로세스 재기동 없이 REGULAR_MARKET_OPEN 전환 및 즉시 평가
        self.assertEqual(observed_calls[1]["market_session"], "REGULAR_MARKET_OPEN")
        self.assertTrue(observed_calls[1]["allow_order_execution"])
        self.assertTrue(observed_calls[1]["sync_intraday"])
        self.assertTrue(any("미국장 오픈 알림" in m for m in self.sent_dispatcher_msgs))

    def test_06_scenario_6_regular_market_valid_long_signal_executes_order(self):
        """시나리오 6: REGULAR_MARKET_OPEN + 유효 LONG 신호 -> 모든 안전 조건 통과 시 주문 실행 허용"""
        runner, broker = self._build_isolated_runner()
        reg_dt = datetime.datetime(2026, 10, 15, 16, 15, 0, tzinfo=ZoneInfo("America/New_York"))
        broker.get_stock_quote.side_effect = lambda sym: {
            "ok": True,
            "symbol": sym,
            "last_price": 101.25 if sym == "SOXL" else 45.80,
            "quote_time_et": "2026-10-15 10:00 ET",
            "quote_session": "REGULAR_MARKET",
            "quote_age_sec": 3.0,
            "is_previous_session_quote": False,
            "is_fresh": True,
        }

        with patch.object(
            runner.moe_orchestrator,
            "evaluate_dual_filter_signal",
            return_value=self._mock_moe_decision(direction="LONG", conf=0.745, is_approved=True),
        ), patch.object(runner, "_execute_buy_with_10s_chase", return_value=True) as mock_buy:
            res = runner.evaluate_15m_cycle_once(
                allow_order_execution=True,
                sync_intraday=False,
                now_dt=reg_dt,
                market_session="REGULAR_MARKET_OPEN",
            )

        self.assertTrue(res["order_allowed"])
        self.assertEqual(res["order_block_reason"], "NONE")
        self.assertTrue(res["order_executed"])
        self.assertEqual(mock_buy.call_count, 1)
        self.assertEqual(mock_buy.call_args.kwargs["symbol"], "SOXL")

        msg = self.sent_dispatcher_msgs[-1]
        self.assertIn("시장 상태: REGULAR_MARKET (REGULAR_MARKET_OPEN)", msg)
        self.assertIn("주문 가능 여부: ALLOWED", msg)
        self.assertIn("주문 차단 사유: NONE", msg)
        self.assertIn("실제 주문 실행 여부: YES", msg)

    def test_07_scenario_7_regular_market_long_signal_plus_filter_veto_blocks_order(self):
        """시나리오 7: REGULAR_MARKET_OPEN + LONG 신호 + Filter Veto -> 주문 차단 및 사유 기록"""
        runner, broker = self._build_isolated_runner()
        reg_dt = datetime.datetime(2026, 10, 15, 16, 15, 0, tzinfo=ZoneInfo("America/New_York"))

        with patch.object(
            runner.moe_orchestrator,
            "evaluate_dual_filter_signal",
            return_value=self._mock_moe_decision(
                direction="NONE",
                dir_gbdt="LONG",
                conf=0.760,
                is_approved=False,
                is_cross_veto=True,
            ),
        ), patch.object(runner, "_execute_buy_with_10s_chase") as mock_buy:
            res = runner.evaluate_15m_cycle_once(
                allow_order_execution=True,
                sync_intraday=False,
                now_dt=reg_dt,
                market_session="REGULAR_MARKET_OPEN",
            )

        self.assertFalse(res["order_allowed"])
        self.assertEqual(res["order_block_reason"], "RISK_BLOCK")
        self.assertEqual(res["order_block_detail"], "CROSS_ASSET_VETO")
        self.assertFalse(res["order_executed"])
        self.assertEqual(mock_buy.call_count, 0)

        msg = self.sent_dispatcher_msgs[-1]
        self.assertIn("주문 가능 여부: BLOCKED", msg)
        self.assertIn("주문 차단 사유: RISK_BLOCK", msg)
        self.assertIn("차단 상세: CROSS_ASSET_VETO", msg)
        self.assertIn("실제 주문 실행 여부: NO", msg)

    def test_08_scenario_8_regular_market_long_signal_plus_existing_position_blocks_order(self):
        """시나리오 8: REGULAR_MARKET_OPEN + LONG 신호 + 기존 포지션 보유 -> 중복 진입 차단"""
        runner, broker = self._build_isolated_runner()
        reg_dt = datetime.datetime(2026, 10, 15, 16, 15, 0, tzinfo=ZoneInfo("America/New_York"))
        runner._active_position = {"symbol": "SOXL", "quantity": 10, "price": 98.50}
        broker.get_overseas_stock_balance.return_value = {
            "ok": True,
            "holdings": [{"symbol": "SOXL", "poss_qty": 10, "buy_price": 98.50}],
            "usd_order_available": 4000.0,
        }

        with patch.object(
            runner.moe_orchestrator,
            "evaluate_dual_filter_signal",
            return_value=self._mock_moe_decision(direction="LONG", conf=0.760, is_approved=True),
        ), patch.object(runner, "_execute_buy_with_10s_chase") as mock_buy:
            res = runner.evaluate_15m_cycle_once(
                allow_order_execution=True,
                sync_intraday=False,
                now_dt=reg_dt,
                market_session="REGULAR_MARKET_OPEN",
            )

        self.assertFalse(res["order_allowed"])
        self.assertEqual(res["order_block_reason"], "RISK_BLOCK")
        self.assertEqual(res["order_block_detail"], "EXISTING_POSITION")
        self.assertFalse(res["order_executed"])
        self.assertEqual(mock_buy.call_count, 0)

        msg = self.sent_dispatcher_msgs[-1]
        self.assertIn("주문 가능 여부: BLOCKED", msg)
        self.assertIn("차단 상세: EXISTING_POSITION", msg)

    def test_09_scenario_9_regular_market_long_signal_plus_open_order_blocks_order(self):
        """시나리오 9: REGULAR_MARKET_OPEN + LONG 신호 + 미체결 주문 존재 -> 중복 주문 차단"""
        runner, broker = self._build_isolated_runner()
        reg_dt = datetime.datetime(2026, 10, 15, 16, 15, 0, tzinfo=ZoneInfo("America/New_York"))
        broker.get_open_orders = MagicMock(
            return_value={
                "ok": True,
                "open_orders": [
                    {
                        "order_no": "0099112",
                        "symbol": "SOXL",
                        "side": "BUY",
                        "nccs_qty": 5,
                        "unfilled_qty": 5,
                    }
                ],
            }
        )

        with patch.object(
            runner.moe_orchestrator,
            "evaluate_dual_filter_signal",
            return_value=self._mock_moe_decision(direction="LONG", conf=0.760, is_approved=True),
        ), patch.object(runner, "_execute_buy_with_10s_chase") as mock_buy:
            res = runner.evaluate_15m_cycle_once(
                allow_order_execution=True,
                sync_intraday=False,
                now_dt=reg_dt,
                market_session="REGULAR_MARKET_OPEN",
            )

        self.assertFalse(res["order_allowed"])
        self.assertEqual(res["order_block_reason"], "RISK_BLOCK")
        self.assertEqual(res["order_block_detail"], "UNCONFIRMED_ORDER_OR_LEDGER_GUARD")
        self.assertFalse(res["order_executed"])
        self.assertEqual(mock_buy.call_count, 0)

        msg = self.sent_dispatcher_msgs[-1]
        self.assertIn("주문 가능 여부: BLOCKED", msg)
        self.assertIn("차단 상세: UNCONFIRMED_ORDER_OR_LEDGER_GUARD", msg)

    def test_10_scenario_10_regular_market_long_signal_plus_circuit_breaker_halt_blocks_order(self):
        """시나리오 10: REGULAR_MARKET_OPEN + LONG 신호 + 서킷브레이커 HALT -> 진입 차단"""
        runner, broker = self._build_isolated_runner()
        reg_dt = datetime.datetime(2026, 10, 15, 16, 15, 0, tzinfo=ZoneInfo("America/New_York"))
        runner.daily_circuit_breaker_triggered = True

        with patch.object(
            runner.moe_orchestrator,
            "evaluate_dual_filter_signal",
            return_value=self._mock_moe_decision(direction="LONG", conf=0.790, is_approved=True),
        ), patch.object(runner, "_execute_buy_with_10s_chase") as mock_buy:
            res = runner.evaluate_15m_cycle_once(
                allow_order_execution=True,
                sync_intraday=False,
                now_dt=reg_dt,
                market_session="REGULAR_MARKET_OPEN",
            )

        self.assertFalse(res["order_allowed"])
        self.assertEqual(res["order_block_reason"], "RISK_BLOCK")
        self.assertIn("CIRCUIT_BREAKER", res["order_block_detail"])
        self.assertFalse(res["order_executed"])
        self.assertEqual(mock_buy.call_count, 0)

        msg = self.sent_dispatcher_msgs[-1]
        self.assertIn("주문 가능 여부: BLOCKED", msg)
        self.assertIn("CIRCUIT_BREAKER", msg)

    def test_11_data_lake_never_contaminates_db_with_premarket_bars(self):
        """프리마켓/애프터마켓 캔들이 market_data.db에 절대 저장되지 않는지 검증"""
        import json

        tmp_db = self.tmp_dir / "test_market_data.db"
        lake = MarketDataLake(db_path=tmp_db)

        # UTC 기준:
        # 11:00Z = 07:00 EDT (프리마켓 -> 제외)
        # 13:15Z = 09:15 EDT (프리마켓 -> 제외)
        # 13:30Z = 09:30 EDT (정규장 -> 보존)
        # 13:45Z = 09:45 EDT (정규장 -> 보존)
        # 20:00Z = 16:00 EDT (애프터마켓 -> 제외)
        fake_alpaca_resp = {
            "bars": {
                "SOXL": [
                    {"t": "2026-10-15T11:00:00Z", "o": 99.0, "h": 99.5, "l": 98.5, "c": 99.2, "v": 0},
                    {"t": "2026-10-15T13:15:00Z", "o": 99.5, "h": 100.0, "l": 99.0, "c": 99.8, "v": 0},
                    {"t": "2026-10-15T13:30:00Z", "o": 100.0, "h": 101.0, "l": 99.5, "c": 100.5, "v": 150000},
                    {"t": "2026-10-15T13:45:00Z", "o": 101.0, "h": 102.0, "l": 100.5, "c": 101.5, "v": 180000},
                    {"t": "2026-10-15T20:00:00Z", "o": 102.0, "h": 102.5, "l": 101.5, "c": 102.1, "v": 0},
                ]
            },
            "next_page_token": None,
        }

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(fake_alpaca_resp).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch.dict("os.environ", {"ALPACA_API_KEY": "DUMMY", "ALPACA_SECRET_KEY": "DUMMY"}), \
             patch("urllib.request.urlopen", return_value=mock_resp):
            inserted = lake.harvest_symbol("SOXL", "15m", period="5d")

        self.assertEqual(inserted, 2)
        loaded = lake.load_candles("SOXL", "15m")
        self.assertEqual(len(loaded), 2)
        for ts in loaded["datetime"].tolist():
            t_part = str(ts)[11:19]
            self.assertGreaterEqual(t_part, "09:30:00")
            self.assertLess(t_part, "16:00:00")


if __name__ == "__main__":
    unittest.main()
