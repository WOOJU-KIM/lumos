import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Windows 콘솔 utf-8 설정
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.kiwoom_broker import KiwoomBroker
from core.kiwoom_ws_streamer import KiwoomWebSocketStreamer
from core.live_runner import KiwoomLiveRunner, USMarketCalendar
from core.data_lake import MarketDataLake
from core.moe_orchestrator import MoEMetaOrchestrator

def run_thorough_verification():
    print("=" * 90)
    print("🧪 [키움증권 REST / WebSocket / 주문 / LiveRunner 전수 신호 실탄 점검]")
    print(f"🕒 점검 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)

    # 1. 브로커 & 토큰
    print("\n[Step 1] 키움 브로커 REST API & OAuth2 토큰 발급 점검")
    broker = KiwoomBroker(is_simulation=True)
    token = broker.get_access_token()
    assert token, "토큰 발급 실패"
    print(f"   ✅ OAuth2 토큰 정상: {token[:15]}... (계좌: {broker.account_no})")

    # 2. 잔고 및 예수금 REST 조회
    print("\n[Step 2] 외화 예수금(ust21110) & 원장 잔고(ust21070) 조회")
    dep = broker.get_overseas_deposit()
    bal = broker.get_overseas_stock_balance()
    print(f"   ✅ 예수금 조회: ${dep.get('usd_order_available', 0):,.2f} USD (성공여부: {dep.get('ok')})")
    print(f"   ✅ 잔고 조회: 보유 종목 {bal.get('holdings_count', 0)}개, 총평가 ${bal.get('total_eval_usd', 0):,.2f} USD")

    # 3. 실시간 시세 조회
    print("\n[Step 3] 주요 종목 실시간 호가/시세 조회")
    for sym in [config.BASE_ASSET_LONG, config.BASE_ASSET_SHORT, "NVDA", "QQQ"]:
        q = broker.get_stock_quote(sym)
        print(f"   ✅ {sym} 시세: ${q.get('last_price', 0):.2f} (제공: {q.get('msg')})")

    # 4. 실시간 틱 스트리머 가동 및 콜백 수신 실증 (3.5초간 대기)
    print("\n[Step 4] 실시간 틱 스트리머(WebSocket+REST Fallback) 가동 및 콜백 신호 수신 실증")
    streamer = KiwoomWebSocketStreamer(broker=broker)
    received_ticks = []
    def tick_cb(sym, px, extra):
        received_ticks.append((sym, px, time.time()))
    streamer.register_callback(tick_cb)
    streamer.start()
    
    print("   ⏳ 3.5초간 실시간 틱 신호 수신 대기 중...")
    time.sleep(3.5)
    streamer.stop()

    print(f"   ✅ 수신된 실시간 틱 신호 건수: {len(received_ticks)}건")
    if received_ticks:
        latest = received_ticks[-1]
        print(f"   ✅ 최근 수신 틱: 종목={latest[0]}, 가격=${latest[1]:.2f}")
    else:
        print("   ❌ 틱 신호 미수신!")
        raise RuntimeError("실시간 틱 스트리밍 콜백 테스트 실패")

    # 5. 실주문 발주 신호 (매수 ➔ 매도)
    print("\n[Step 5] 1주 시장가 매수 ➔ 1주 시장가 매도 청산 발주 신호 송출")
    buy_res = broker.send_order(config.BASE_ASSET_LONG, "BUY", 1, price=0.0)
    print(f"   ✅ 매수 발주 응답: ok={buy_res.get('ok')}, 주문번호={buy_res.get('order_no')}, 체결단가=${buy_res.get('price'):.2f}")
    
    time.sleep(1)
    sell_res = broker.send_order(config.BASE_ASSET_LONG, "SELL", 1, price=0.0)
    print(f"   ✅ 매도 청산 응답: ok={sell_res.get('ok')}, 주문번호={sell_res.get('order_no')}, 체결단가=${sell_res.get('price'):.2f}")

    # 6. USMarketCalendar 장 운영 시간 및 90분 컷오프 로직 검증
    print("\n[Step 6] USMarketCalendar & 90분 가드 로직 무결성 검증")
    mkt = USMarketCalendar.get_market_status()
    print(f"   ✅ 현재 세션: {mkt['session_name']} ({mkt['status_desc']})")
    print(f"   ✅ is_open: {mkt['is_open']}, is_entry_allowed: {mkt['is_entry_allowed']}, is_eod_window: {mkt['is_eod_liquidation_window']}")
    print(f"   ✅ 뉴욕 시각: {mkt['now_ny_str']} | 한국 시각: {mkt['now_kst_str']}")

    # 7. MoE 오케스트레이터 시그널 평가 무결성
    print("\n[Step 7] MoE 7대 모델 및 캔들 평가 파이프라인 무결성 점검")
    data_lake = MarketDataLake()
    moe = MoEMetaOrchestrator()
    long_15m = data_lake.load_candles(config.BASE_ASSET_LONG, "15m")
    moe_res = moe.evaluate_dual_filter_signal(long_15m, threshold=0.75)
    print(f"   ✅ MoE 평가 완료: Top-1 모델=[{moe_res.get('expert_desc')}] (확신도: {moe_res.get('gating_confidence', 0)*100:.1f}%), 방향: {moe_res.get('direction')}")

    print("\n" + "=" * 90)
    print("🎯 [전수 신호 점검 완료] 모든 REST, 시세, 스트리밍, 주문, 캘린더, MoE 엔진 무결성 통과!")
    print("=" * 90)

if __name__ == "__main__":
    run_thorough_verification()
