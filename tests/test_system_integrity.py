import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

import config
from core.moe_orchestrator import MoEMetaOrchestrator
from core.live_runner import KiwoomLiveRunner
from core.kiwoom_broker import KiwoomBroker

class TestSystemIntegrity(unittest.TestCase):
    
    def setUp(self):
        self.moe = MoEMetaOrchestrator(
            confidence_threshold=config.GBDT_CONFIDENCE_THRESHOLD, 
            gbdt_threshold=config.GBDT_CONFIDENCE_THRESHOLD, 
            mode="hybrid_v3"
        )
        dates = pd.date_range("2026-08-24 09:30", periods=50, freq="15min")
        self.dummy_15m = pd.DataFrame({
            "Open": np.linspace(100, 105, 50),
            "High": np.linspace(101, 106, 50),
            "Low": np.linspace(99, 104, 50),
            "Close": np.linspace(100.5, 105.5, 50),
            "Volume": [100000] * 50
        }, index=dates)

    def test_1_confidence_under_threshold_must_be_rejected(self):
        """[인터락 1] Lumos V3 GBDT 확신도 임계값 미달 시 무조건 매수 거부"""
        threshold = config.GBDT_CONFIDENCE_THRESHOLD
        res = self.moe.evaluate_dual_filter_signal(self.dummy_15m, threshold=threshold)
        if res.get("gating_confidence", 0.0) < threshold:
            self.assertFalse(res["is_approved"], "확신도 미달 시 is_approved는 False여야 합니다.")

    def test_2_no_holdings_no_sell_orders(self):
        """[인터락 2] 실제 원장 잔고(holdings == 0)가 없으면 어떤 매도(청산) 발주 불가"""
        runner = KiwoomLiveRunner(is_simulation=True)
        empty_stk_bal = {"holdings_count": 0, "holdings": []}
        
        with patch.object(runner, '_dispatch_sell') as mock_sell:
            runner._manage_open_positions(empty_stk_bal)
            mock_sell.assert_not_called()

    def test_3_cancel_tr_format(self):
        """[인터락 3] 미체결 주문 취소 TR 규격 무결성 확인"""
        broker = KiwoomBroker(is_simulation=True)
        import inspect
        sig = inspect.signature(broker.cancel_order)
        self.assertIn("order_no", sig.parameters)
        self.assertIn("symbol", sig.parameters)

    def test_4_no_arbitrary_market_open_clear(self):
        """[인터락 4] 장개장 강제 청산(CarryOverClear) 완전 영구 제거 및 ATR 트레일링 스탑 준수"""
        runner = KiwoomLiveRunner(is_simulation=True)
        import inspect
        source = inspect.getsource(runner._market_execution_loop)
        self.assertNotIn("CarryOverClear", source, "CarryOverClear(장개장 강제 청산) 로직이 존재해서는 안됩니다.")

    def test_5_daily_circuit_breaker_abolished(self):
        """[인터락 5] 일일 3아웃 손절(3-Out) 룰 영구 폐지 검증"""
        runner = KiwoomLiveRunner(is_simulation=True)
        runner.dispatcher.send_telegram_message = MagicMock()
        runner.notifier._send_http_request = MagicMock()
        
        runner.daily_stoploss_count = 3
        runner.daily_circuit_breaker_triggered = True

        with patch.object(runner.broker, 'get_overseas_stock_balance', return_value={"ok": True, "holdings": []}):
            with patch.object(runner.broker, 'send_order', return_value={"order_no": "123"}):
                with patch.object(runner, '_wait_for_fill', return_value=(True, 10, 0)):
                    with patch.object(runner, '_save_active_position'):
                        buy_res = runner._execute_buy_with_10s_chase(
                        symbol=config.BASE_ASSET_LONG,
                        target_qty=10,
                        ref_price=30.0,
                        moe_res={"gating_confidence": 0.8, "expert_desc": "Test"},
                        targets={"dynamic_tp_px": 35.0, "dynamic_sl_px": 28.0}
                    )
                    self.assertTrue(buy_res, "3아웃 제한 없이 구매가 진행되어야 합니다.")

    def test_6_hybrid_moe_single_trigger_logic(self):
        """[인터락 6] Lumos V3 하이브리드 MoE 결합 무결성 검증"""
        moe_v3 = MoEMetaOrchestrator(
            confidence_threshold=config.GBDT_CONFIDENCE_THRESHOLD, 
            gbdt_threshold=config.GBDT_CONFIDENCE_THRESHOLD, 
            mode="hybrid_v3"
        )
        moe_v3.cross_asset_model = MagicMock()
        moe_v3.data_lake = MagicMock()
        moe_v3.data_lake.load_candles.return_value = self.dummy_15m
        
        with patch.object(moe_v3.gbdt_engine, 'predict_signal', return_value=(1, config.GBDT_CONFIDENCE_THRESHOLD + 0.08, "reason")):
            with patch.object(moe_v3.cross_asset_model, 'predict_signal', return_value=(1, 0.70, "reason")):
                with patch.object(moe_v3.gbdt_engine, 'extract_features', return_value=pd.DataFrame({"RSI_14": [50.0]})):
                    with patch("config.USE_60M_TREND_FILTER", False):
                        res = moe_v3.evaluate_dual_filter_signal(self.dummy_15m)
                        self.assertEqual(res["direction"], "LONG")

    def test_7_time_synchronization_and_model_switching_integrity(self):
        """[인터락 7] 글로벌 시간 동기화(KST-NYT) 무결성 검증"""
        from core.market_calendar import USMarketCalendar
        res = USMarketCalendar.verify_time_synchronization()
        self.assertTrue(res["all_ok"], f"시간 동기화 검증 실패: {res}")

    def test_8_neutral_prob_highest_must_be_rejected(self):
        """[인터락 8] GBDT 3-Class 예측 중 Neutral(관망) 확률이 가장 높다면 무조건 관망"""
        moe_v3 = MoEMetaOrchestrator(
            confidence_threshold=0.0, # 낮춰서 로직만 테스트
            gbdt_threshold=0.0, 
            mode="hybrid_v3"
        )
        moe_v3.cross_asset_model = MagicMock()
        moe_v3.data_lake = MagicMock()
        moe_v3.data_lake.load_candles.return_value = self.dummy_15m
        
        moe_v3.gbdt_engine.model = MagicMock()
        moe_v3.gbdt_engine.scaler = MagicMock()
        moe_v3.gbdt_engine.scaler.transform.return_value = [[0]]
        moe_v3.gbdt_engine.model.predict_proba.return_value = np.array([[0.30, 0.45, 0.25]])
        with patch.object(moe_v3.gbdt_engine, 'extract_features', return_value=pd.DataFrame({"dummy": [0]})):
            res = moe_v3.evaluate_dual_filter_signal(self.dummy_15m)
            self.assertEqual(res["direction"], "NONE")
            self.assertFalse(res["is_approved"])
            
            # 반대 경우 (Long이 가장 높을 때)
            moe_v3.gbdt_engine.model.predict_proba.return_value = np.array([[0.25, 0.30, 0.45]])
            res_long = moe_v3.evaluate_dual_filter_signal(self.dummy_15m)
            self.assertEqual(res_long["direction"], "LONG")

    def test_9_allocation_rule_and_slippage(self):
        """[인터락 9] 98% 비중 룰 및 백테스트 수수료율 룰 준수"""
        self.assertAlmostEqual(config.MAX_ALLOCATION_RATIO, 0.98, places=2, msg="98% 진입 비중 룰 훼손!")
        self.assertAlmostEqual(config.QTY_CALC_BUFFER, 0.05, places=2, msg="수량 계산 버퍼($0.05) 훼손!")
        self.assertGreaterEqual(config.BACKTEST_FEE_SLIPPAGE, 0.0020, msg="백테스트 수수료율이 0.20% 미만으로 설정됨!")

if __name__ == "__main__":
    unittest.main()
