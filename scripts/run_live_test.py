# -*- coding: utf-8 -*-
import os, sys, time
from pathlib import Path
from unittest.mock import patch
from datetime import datetime as real_datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["KIWOOM_IS_SIMULATION"] = "1"

from core.live_runner import KiwoomLiveRunner
from core.kiwoom_broker import KiwoomBroker
from core.mock_kiwoom_feeder import MockKiwoomFeeder
import core.market_calendar
from core.system_logger import system_logger

# -------------------------------------------------------------
# 1. Market Calendar Mock
# -------------------------------------------------------------
original_get_market_status = core.market_calendar.USMarketCalendar.get_market_status
def mock_get_market_status(*args, **kwargs):
    res = original_get_market_status(*args, **kwargs)
    res["is_open"] = True  # 강제로 열림 처리 (시뮬레이션)
    res["is_trading_allowed"] = True
    return res
core.market_calendar.USMarketCalendar.get_market_status = mock_get_market_status

def mock_verify_time_synchronization():
    return {
        "all_ok": True, "time_sync_ok": True, "switching_ok": True,
        "delta_hours": 13.0, "expected_diff": 13.0, "is_dst": True,
        "dst_text": "EDT", "now_kst_str": "MOCK KST",
        "now_ny_str": "MOCK NYT", "checklist_items": ["Mock item"]
    }

# -------------------------------------------------------------
# 2. Fake Broker (Overrides all Kiwoom Network Calls)
# -------------------------------------------------------------
class FakeBroker:
    def __init__(self, real_broker):
        self.real_broker = real_broker
        self.fake_holdings = []
        self.usd_avail = 100000.0
        self.broker_name = "Kiwoom"
        self.account_no = "61112456"
        self.account_type = "01"
    
    def __getattr__(self, name):
        return getattr(self.real_broker, name)
        
    def send_order(self, symbol, order_type, quantity, price, **kwargs):
        import uuid
        ord_no = str(uuid.uuid4())
        if order_type == 'BUY':
            self.fake_holdings.append({'symbol': symbol, 'quantity': quantity, 'poss_qty': quantity, 'purchase_price': price, 'buy_time': '2026-09-21'})
            self.usd_avail -= quantity * price
            print(f"\n✅ [FakeBroker] {symbol} {quantity}주 매수 체결 완료 (가용달러: )")
        elif order_type == 'SELL':
            for h in self.fake_holdings[:]:
                if h['symbol'] == symbol:
                    h['quantity'] -= quantity
                    h['poss_qty'] -= quantity
                    if h['quantity'] <= 0:
                        self.fake_holdings.remove(h)
                    print(f"\n✅ [FakeBroker] {symbol} {quantity}주 매도 체결 완료")
                    break
        return {'ok': True, 'order_no': ord_no}
        
    def get_overseas_stock_balance(self, force_refresh=False):
        return {'ok': True, 'holdings': self.fake_holdings}
        
    def get_overseas_deposit(self):
        return {'ok': True, 'usd_order_available': self.usd_avail}
        
    def get_overseas_unfilled_orders(self):
        return {'ok': True, 'unfilled_orders': []}
        
    def cancel_order(self, *args, **kwargs): pass
    def cancel_all_open_orders(self): return []
    def get_official_broker_report(self):
        return {'ok': True, 'holdings': self.fake_holdings, 'avail_usd': self.usd_avail, 'total_eval_usd': self.usd_avail}
    def test_connection(self):
        return {'ok': True, 'usd_order_available': self.usd_avail, 'holdings_count': len(self.fake_holdings)}

# -------------------------------------------------------------
# 3. Main Test Execution (Synchronous loop)
# -------------------------------------------------------------
def main():
    print("🚀 시작: 9월 21일 End-to-End 매매 테스트 (LiveRunner & Telegram)")
    
    # 1. 초기화 및 Broker 주입
    global_runner = KiwoomLiveRunner(is_simulation=True)
    global_runner.broker = FakeBroker(KiwoomBroker(is_simulation=True))
    print("Created FakeBroker")
    
    # 2. WebSocket Streamer (Fake)
    feeder = MockKiwoomFeeder(global_runner.broker, target_date="2026-09-21")
    print("Created MockKiwoomFeeder")
    global_runner.ws_streamer = feeder
    print("Assigned ws_streamer")
    
    # 3. 매수체결 대기 우회 (즉시 체결)
    global_runner._wait_for_fill = lambda order_no, symbol, is_buy, target_qty, timeout_sec=None: (True, target_qty, 0)
    print("Assigned _wait_for_fill")
    
    # 4. Telegram 필터링 (진입/청산만 출력)
    def filtered_send_tg(message, event_time_str=None):
        if "15분 시그널" in message or "15m Signal" in message or "Briefing" in message or "진입 대기" in message or "정기 브리핑" in message:
            return
        print(f"\n🔔 [텔레그램 발송 됨]\n{message}")

    global_runner.dispatcher.send_telegram_message = filtered_send_tg
    print("Assigned send_telegram_message")
    
    # 5. 수동으로 타임라인 반복
    global_runner._last_briefing_time = 0.0
    global_runner._is_order_in_progress = False
    
    print("⏳ 매매 루프 실행 중... (15분봉 진입 평가 / 5분봉 트레일링 스탑 적용)")
    
    for current_time_ts in feeder.timeline:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        import pandas as pd
        
        if isinstance(current_time_ts, str):
            ct = datetime.strptime(current_time_ts, "%Y-%m-%d %H:%M:%S")
        elif isinstance(current_time_ts, pd.Timestamp):
            ct = current_time_ts.to_pydatetime()
        else:
            ct = current_time_ts
            
        mock_kst = ct.replace(tzinfo=ZoneInfo("Asia/Seoul"))
        ct = mock_kst.astimezone(ZoneInfo("America/New_York"))
        feeder.current_mock_time = ct
        
        # ?제 최신 가격 업데이트
        for sym in feeder.subscribed_symbols:
            df_sym = feeder.mock_data.get(sym)
            if df_sym is not None and not df_sym.empty:
                row = df_sym[df_sym['dt_obj'] == current_time_ts]
                if not row.empty:
                    feeder._latest_prices[sym] = float(row.iloc[0]['Close'])
        
        # 1. 포지션 관리 (5분봉)
        stk_bal = global_runner.broker.get_overseas_stock_balance()
        if stk_bal.get("holdings"):
            live_p = {
                config.BASE_ASSET_LONG: feeder.get_latest_price(config.BASE_ASSET_LONG, 0.0),
                config.BASE_ASSET_SHORT: feeder.get_latest_price(config.BASE_ASSET_SHORT, 0.0)
            }
            # Trailing Stop을 위해 _manage_open_positions 내부에서 현재가 평가
            with patch('core.live_runner.datetime') as mock_dt, patch('core.live_runner.time.time') as mock_t:
                mock_dt.now.return_value = mock_kst
                mock_t.return_value = ct.timestamp()
                global_runner._manage_open_positions(stk_bal, realtime_px_override=live_p)
                
        # 2. 15분 단위 진입 평가 (15분봉)
        # 09:30 ~ 15:30 정규장 시간(Phase 1, 2)에만 신규 진입 허용. 
        # 그리고 포지션이 없을 때만 진입.
        is_market_open = (ct.hour > 9 or (ct.hour == 9 and ct.minute >= 30)) and ct.hour < 16
        is_entry_time = is_market_open and (ct.hour < 15 or (ct.hour == 15 and ct.minute <= 30))
        
        has_position = len(stk_bal.get("holdings", [])) > 0

        if ct.minute % 15 == 0 and is_entry_time and not has_position:
            if not global_runner._is_order_in_progress:
                try:
                    global_runner._is_order_in_progress = True
                    
                    # Lookahead bias 방지: ct 시점에서는 ct 이전의 완성된 캔들만 존재함.
                    # 즉 09:30:00 에는 09:15:00 캔들까지만 DB에 존재해야 함.
                    # 따라서 ct에서 1초를 뺀 시간을 end_dt로 사용하여 미래 캔들 조회를 차단함.
                    from datetime import timedelta
                    safe_end_dt = (ct - timedelta(seconds=1)).strftime("%Y-%m-%d %H:%M:%S")
                    df_15m = global_runner.data_lake.load_candles(config.BASE_ASSET_LONG, "15m", end_dt=safe_end_dt)
                    
                    if df_15m.empty:
                        continue
                        
                    live_prices = {
                        config.BASE_ASSET_LONG: feeder.get_latest_price(config.BASE_ASSET_LONG, 0.0),
                        config.BASE_ASSET_SHORT: feeder.get_latest_price(config.BASE_ASSET_SHORT, 0.0)
                    }
                    
                    # AI 엔진 호출 시 datetime을 KST로 속여야 함
                    with patch('core.live_runner.datetime') as mock_dt:
                        mock_dt.now.return_value = mock_kst
                        moe_res = global_runner.moe_orchestrator.evaluate_dual_filter_signal(df_candle_15m=df_15m, current_time_str=ct.strftime("%Y-%m-%d %H:%M:%S"), live_prices=live_prices)
                    
                    is_appr = moe_res.get("is_approved", False)
                    direction = moe_res.get("direction", "NONE")
                    print(f"[{mock_kst.strftime('%H:%M')}] Eval Result: {is_appr} / {direction}")
                    
                    if is_appr and ("LONG" in direction or "SHORT" in direction):
                        winner_sym = config.BASE_ASSET_LONG if direction == "LONG" else config.BASE_ASSET_SHORT
                        cur_px = live_prices.get(winner_sym, 0.0)
                        
                        if cur_px > 0:
                            conn_res = global_runner.broker.test_connection()
                            usd_avail = float(conn_res.get("usd_order_available", 0.0))
                            
                            import config
                            order_qty = int((usd_avail * config.MAX_ALLOCATION_RATIO) / (cur_px + config.QTY_CALC_BUFFER))
                            
                            if order_qty > 0:
                                print(f"\n[{mock_kst.strftime('%Y-%m-%d %H:%M:%S')} (NYT: {ct.strftime('%H:%M:%S')})] 💡 매수 시그널 발생! ({direction})")
                                # 시간 mock 적용된 상태로 즉시 체결
                                with patch('core.live_runner.datetime') as m_dt_buy, patch('core.live_runner.time.time') as m_time_buy:
                                    m_dt_buy.now.return_value = mock_kst
                                    m_time_buy.return_value = ct.timestamp()
                                    
                                    atr_14 = float(moe_res.get("atr_14", 0.0))
                                    atr_pct = (atr_14 / cur_px) * 100.0 if cur_px > 0 else 2.0
                                    sl_pct = max(config.SL_MIN_PCT * 100.0, min(config.SL_MAX_PCT * 100.0, atr_pct * config.SL_ATR_MULTIPLIER))
                                    tp_pct = config.MAX_TP_PCT * 100.0
                                    targets = {
                                        "dynamic_tp_px": round(cur_px * (1.0 + tp_pct/100.0), 2),
                                        "dynamic_sl_px": round(cur_px * (1.0 - sl_pct/100.0), 2),
                                        "tp_pct": tp_pct,
                                        "sl_pct": sl_pct
                                    }
                                    global_runner._execute_buy_with_10s_chase(winner_sym, order_qty, cur_px, moe_res, targets)
                
                finally:
                    global_runner._is_order_in_progress = False

    print("\n✅ 시뮬레이션 완료. 장 종료.")

if __name__ == '__main__':
    main()
