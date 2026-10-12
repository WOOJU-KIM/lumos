import os
import config
import sys
import json
import time
import urllib.request
import urllib.error
import asyncio
import websockets
from datetime import datetime
from pathlib import Path

# Windows console encoding
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.kiwoom_broker import KiwoomBroker
from dotenv import load_dotenv

load_dotenv()

def run_kiwoom_mock_comms_check():
    print("=" * 80)
    print("🚀 [키움증권 모의거래 통신 전수 점검 및 실시간 신호 송수신 테스트 시작]")
    print(f"⏰ 실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # 0. 브로커 인스턴스 초기화 (모의투자 모드 강제 적용)
    broker = KiwoomBroker(is_simulation=True)
    print(f"\n[기본 정보]")
    print(f"• 환경 모드: {broker.mode_str}")
    print(f"• Base URL: {broker.base_url}")
    print(f"• 계좌번호: {broker.account_no}")
    print(f"• 계좌상품코드: {broker.account_type}")
    print(f"• App Key: {broker.app_key[:6]}******" if broker.app_key else "• App Key: 없음")

    results = {}

    # 1. OAuth2 인증 및 토큰 발급 통신 점검 (/oauth2/token)
    print("\n" + "-" * 80)
    print("1️⃣ [통신 체크 1] OAuth2 인증 토큰 발급 (/oauth2/token)")
    print("-" * 80)
    token_start = time.time()
    try:
        token = broker.get_access_token(force_refresh=True)
        token_elapsed = round(time.time() - token_start, 3)
        print(f"✅ 토큰 발급 성공! (소요시간: {token_elapsed}초)")
        print(f"   • Access Token: {token[:20]}...{token[-10:]}")
        print(f"   • 만료 일시: {broker._token_expires_at}")
        results["1_oauth2_token"] = {
            "status": "SUCCESS",
            "elapsed_sec": token_elapsed,
            "token_preview": f"{token[:20]}...",
            "expires_at": str(broker._token_expires_at)
        }
    except Exception as e:
        token_elapsed = round(time.time() - token_start, 3)
        print(f"❌ 토큰 발급 실패: {e}")
        results["1_oauth2_token"] = {
            "status": "FAIL",
            "elapsed_sec": token_elapsed,
            "error": str(e)
        }
        print("토큰 발급 실패로 인해 이후 TR 통신을 중단합니다.")
        return results

    # 2. 외화 예수금 조회 통신 점검 (TR: ust21110, Endpoint: /api/us/acnt)
    print("\n" + "-" * 80)
    print("2️⃣ [통신 체크 2] 외화 예수금 조회 (TR: ust21110 | /api/us/acnt)")
    print("-" * 80)
    dep_start = time.time()
    try:
        # 캐시 없이 직접 TR 호출
        dep_body = {
            "cano": broker.account_no,
            "acnt_prdt_cd": broker.account_type
        }
        raw_dep_res = broker._send_tr_request(endpoint="/api/us/acnt", api_id="ust21110", body_dict=dep_body)
        dep_elapsed = round(time.time() - dep_start, 3)
        
        parsed_dep = broker.get_overseas_deposit()
        print(f"📊 [외화예수금 리턴 결과 요약] (소요시간: {dep_elapsed}초)")
        print(f"   • Return Code: {raw_dep_res.get('return_code')}")
        print(f"   • Return Msg: {raw_dep_res.get('return_msg')}")
        print(f"   • USD 외화 예수금: ${parsed_dep.get('usd_deposit', 0.0):,.2f}")
        print(f"   • USD 주문 가능금액: ${parsed_dep.get('usd_order_available', 0.0):,.2f}")
        print(f"   • 원화 환산금액: {parsed_dep.get('krw_converted', 0):,}원")
        
        results["2_foreign_deposit"] = {
            "status": "SUCCESS" if raw_dep_res.get("return_code") == 0 else "FAIL",
            "elapsed_sec": dep_elapsed,
            "return_code": raw_dep_res.get("return_code"),
            "return_msg": raw_dep_res.get("return_msg"),
            "usd_deposit": parsed_dep.get("usd_deposit", 0.0),
            "usd_order_available": parsed_dep.get("usd_order_available", 0.0),
            "krw_converted": parsed_dep.get("krw_converted", 0),
            "raw_response": raw_dep_res
        }
    except Exception as e:
        dep_elapsed = round(time.time() - dep_start, 3)
        print(f"❌ 외화 예수금 조회 실패: {e}")
        results["2_foreign_deposit"] = {"status": "FAIL", "elapsed_sec": dep_elapsed, "error": str(e)}

    # 3. 미국주식 원장 잔고/보유종목 조회 통신 점검 (TR: ust21070, Endpoint: /api/us/acnt)
    print("\n" + "-" * 80)
    print("3️⃣ [통신 체크 3] 미국주식 원장 잔고/보유종목 조회 (TR: ust21070 | /api/us/acnt)")
    print("-" * 80)
    bal_start = time.time()
    try:
        bal_body = {
            "cano": broker.account_no,
            "acnt_prdt_cd": broker.account_type,
            "qry_tp": "1"
        }
        raw_bal_res = broker._send_tr_request(endpoint="/api/us/acnt", api_id="ust21070", body_dict=bal_body)
        bal_elapsed = round(time.time() - bal_start, 3)
        parsed_bal = broker.get_overseas_stock_balance()
        
        print(f"📊 [원장잔고 리턴 결과 요약] (소요시간: {bal_elapsed}초)")
        print(f"   • Return Code: {raw_bal_res.get('return_code')}")
        print(f"   • Return Msg: {raw_bal_res.get('return_msg')}")
        print(f"   • 총 평가금액: ${parsed_bal.get('total_eval_usd', 0.0):,.2f}")
        print(f"   • 총 매입금액: ${parsed_bal.get('total_purchase_usd', 0.0):,.2f}")
        print(f"   • 총 평가손익: ${parsed_bal.get('total_pnl_usd', 0.0):,.2f}")
        print(f"   • 보유 종목 수: {parsed_bal.get('holdings_count', 0)}개")
        for h in parsed_bal.get("holdings", []):
            print(f"     - [{h.get('symbol')}] {h.get('quantity')}주 | 평단: ${h.get('purchase_price')} | 현재가: ${h.get('eval_price')} | 손익: ${h.get('pnl_amount_usd')} ({h.get('pnl_rate')}%)")

        results["3_stock_balance"] = {
            "status": "SUCCESS" if raw_bal_res.get("return_code") == 0 else "FAIL",
            "elapsed_sec": bal_elapsed,
            "return_code": raw_bal_res.get("return_code"),
            "return_msg": raw_bal_res.get("return_msg"),
            "total_eval_usd": parsed_bal.get("total_eval_usd", 0.0),
            "total_purchase_usd": parsed_bal.get("total_purchase_usd", 0.0),
            "total_pnl_usd": parsed_bal.get("total_pnl_usd", 0.0),
            "holdings": parsed_bal.get("holdings", []),
            "raw_response": raw_bal_res
        }
    except Exception as e:
        bal_elapsed = round(time.time() - bal_start, 3)
        print(f"❌ 원장 잔고 조회 실패: {e}")
        results["3_stock_balance"] = {"status": "FAIL", "elapsed_sec": bal_elapsed, "error": str(e)}

    # 4. 체결/미체결 내역 조회 통신 점검 (TR: ust21050, Endpoint: /api/us/acnt)
    print("\n" + "-" * 80)
    print("4️⃣ [통신 체크 4] 체결/미체결 내역 조회 (TR: ust21050 | /api/us/acnt)")
    print("-" * 80)
    ord_start = time.time()
    try:
        ord_body = {
            "cano": broker.account_no,
            "acnt_prdt_cd": broker.account_type,
            "qry_tp": "0"
        }
        raw_open_res = broker._send_tr_request(endpoint="/api/us/acnt", api_id="ust21050", body_dict=ord_body)
        ord_elapsed = round(time.time() - ord_start, 3)
        parsed_open = broker.get_open_orders()
        
        print(f"📊 [체결/미체결 리턴 결과 요약] (소요시간: {ord_elapsed}초)")
        print(f"   • Return Code: {raw_open_res.get('return_code')}")
        print(f"   • Return Msg: {raw_open_res.get('return_msg')}")
        print(f"   • 주문 내역 건수: {parsed_open.get('open_orders_count', 0)}건")
        for o in parsed_open.get("orders", [])[:5]:
            print(f"     - 주문번호: {o.get('ord_no')} | 종목: {o.get('stk_cd') or o.get('symb')} | 구분: {o.get('trde_tp_nm') or o.get('sll_buy_tp_cd')} | 수량: {o.get('ord_qty')} | 단가: {o.get('ord_uv')}")

        results["4_order_history"] = {
            "status": "SUCCESS" if raw_open_res.get("return_code") == 0 else "FAIL",
            "elapsed_sec": ord_elapsed,
            "return_code": raw_open_res.get("return_code"),
            "return_msg": raw_open_res.get("return_msg"),
            "orders_count": parsed_open.get("open_orders_count", 0),
            "orders": parsed_open.get("orders", []),
            "raw_response": raw_open_res
        }
    except Exception as e:
        ord_elapsed = round(time.time() - ord_start, 3)
        print(f"❌ 체결/미체결 내역 조회 실패: {e}")
        results["4_order_history"] = {"status": "FAIL", "elapsed_sec": ord_elapsed, "error": str(e)}

    # 5. 미국주식 시세 조회 통신 점검 (TQQQ, SQQQ, NVDA)
    print("\n" + "-" * 80)
    print("5️⃣ [통신 체크 5] 미국주식 실시간 시세 조회 (TQQQ, SQQQ, NVDA)")
    print("-" * 80)
    quotes = {}
    for sym in [config.BASE_ASSET_LONG, config.BASE_ASSET_SHORT, "NVDA"]:
        q_start = time.time()
        q = broker.get_stock_quote(sym)
        q_elapsed = round(time.time() - q_start, 3)
        print(f"   • [{sym}] 현재가: ${q.get('last_price', 0.0):.2f} (제공자: {q.get('provider')}, 소요시간: {q_elapsed}초)")
        quotes[sym] = {
            "price": q.get("last_price", 0.0),
            "provider": q.get("provider"),
            "ok": q.get("ok"),
            "elapsed_sec": q_elapsed
        }
    results["5_market_quotes"] = quotes

    # 6. 미국주식 모의 주문 직접 신호 송출 및 리턴 수신 테스트 (TR: ust20000 매수 / ust20004 취소)
    print("\n" + "-" * 80)
    print("6️⃣ [통신 체크 6] 키움 모의투자 실시간 매수 신호 송출 및 리턴 수신 (TR: ust20000 | /api/us/ordr)")
    print("-" * 80)
    test_sym = config.BASE_ASSET_LONG
    quote_px = quotes.get(test_sym, {}).get("price", 28.50)
    # 현재가 대비 충분히 낮게 지정가 설정하여 바로 체결되지 않고 미체결로 생성되게 테스트 (취소 테스트 연계)
    test_limit_px = round(quote_px * 0.5, 2)  # 50% 낮은 가격으로 지정가 주문
    
    order_start = time.time()
    try:
        print(f"📡 [신호 송출] {test_sym} 1주 지정가(${test_limit_px}) 매수 주문 전송 중...")
        buy_res = broker.send_order(
            symbol=test_sym,
            order_type="BUY",
            quantity=1,
            price=test_limit_px,
            exchange="NY"
        )
        order_elapsed = round(time.time() - order_start, 3)
        ord_no = buy_res.get("order_no", "")
        
        print(f"✅ [주문 신호 리턴 수신 성공] (소요시간: {order_elapsed}초)")
        print(f"   • 주문번호(ord_no): {ord_no}")
        print(f"   • Return Code: {buy_res.get('return_code')}")
        print(f"   • Return Msg: {buy_res.get('msg')}")
        print(f"   • 원시 응답: {buy_res.get('raw_response')}")

        results["6_buy_order"] = {
            "status": "SUCCESS",
            "elapsed_sec": order_elapsed,
            "order_no": ord_no,
            "return_code": buy_res.get("return_code"),
            "return_msg": buy_res.get("msg"),
            "raw_response": buy_res.get("raw_response")
        }

        # 7. 방금 발생시킨 미체결 주문 취소 통신 점검 (TR: ust20004)
        if ord_no:
            print("\n" + "-" * 80)
            print(f"7️⃣ [통신 체크 7] 방금 발주한 주문({ord_no}) 취소 신호 송출 (TR: ust20004 | /api/us/ordr)")
            print("-" * 80)
            cancel_start = time.time()
            time.sleep(0.5) # 잠시 텀
            cncl_res = broker.cancel_order(order_no=ord_no, symbol=test_sym, exchange="NY")
            cancel_elapsed = round(time.time() - cancel_start, 3)
            print(f"✅ [주문 취소 신호 리턴 수신] (소요시간: {cancel_elapsed}초)")
            print(f"   • 성공 여부: {cncl_res.get('ok')}")
            print(f"   • Return Msg: {cncl_res.get('msg')}")
            print(f"   • 원시 응답: {cncl_res.get('raw_response')}")
            results["7_cancel_order"] = {
                "status": "SUCCESS" if cncl_res.get("ok") else "FAIL",
                "elapsed_sec": cancel_elapsed,
                "order_no": ord_no,
                "return_msg": cncl_res.get("msg"),
                "raw_response": cncl_res.get("raw_response")
            }
        else:
            results["7_cancel_order"] = {"status": "SKIPPED", "reason": "No order_no"}

    except Exception as e:
        order_elapsed = round(time.time() - order_start, 3)
        print(f"❌ 매수 주문 신호 송수신 실패: {e}")
        results["6_buy_order"] = {"status": "FAIL", "elapsed_sec": order_elapsed, "error": str(e)}
        results["7_cancel_order"] = {"status": "SKIPPED", "reason": "Buy order failed"}

    # 8. 키움 실시간 WebSocket 통신 점검 (wss://mockapi.kiwoom.com:10000/api/us/websocket)
    print("\n" + "-" * 80)
    print("8️⃣ [통신 체크 8] 실시간 웹소켓(WebSocket) 통신 연결 및 핸드셰이크 점검")
    print("-" * 80)
    
    async def check_ws_communication():
        ws_url = "wss://mockapi.kiwoom.com:10000/api/us/websocket"
        print(f"🌐 WebSocket 연결 시도: {ws_url}")
        headers = {"authorization": f"Bearer {token}"}
        try:
            async with websockets.connect(ws_url, additional_headers=headers, open_timeout=5, ping_timeout=5) as ws:
                print("   • WebSocket TCP Handshake 성공!")
                
                # LOGIN 패킷 전송
                login_packet = {
                    "trnm": "LOGIN",
                    "token": token
                }
                print(f"   • [WS 송출] LOGIN 패킷: {json.dumps(login_packet)}")
                await ws.send(json.dumps(login_packet))
                login_raw = await asyncio.wait_for(ws.recv(), timeout=5)
                print(f"   • [WS 수신] LOGIN 리턴: {login_raw}")
                login_data = json.loads(login_raw)

                # REG 구독 패킷 전송
                reg_packet = {
                    "trnm": "REG",
                    "grp_no": "1",
                    "refresh": "1",
                    "data": [
                        {
                            "item": [{"jmcode": config.BASE_ASSET_LONG, "stex_tp": "ND"}, {"jmcode": config.BASE_ASSET_SHORT, "stex_tp": "ND"}],
                            "type": ["0A", "0B", "FT"]
                        }
                    ]
                }
                print(f"   • [WS 송출] REG 패킷: {json.dumps(reg_packet)}")
                await ws.send(json.dumps(reg_packet))
                reg_raw = await asyncio.wait_for(ws.recv(), timeout=5)
                print(f"   • [WS 수신] REG 리턴: {reg_raw}")
                reg_data = json.loads(reg_raw)

                return {
                    "status": "SUCCESS",
                    "login_return": login_data,
                    "reg_return": reg_data
                }
        except Exception as e:
            print(f"   ⚠️ WebSocket 연결/응답 결과: {e}")
            return {
                "status": "ERROR_OR_UNAVAILABLE",
                "error": str(e),
                "note": "모의투자 웹소켓 서버 포트 또는 장외시간 세션 상태"
            }

    try:
        ws_res = asyncio.run(check_ws_communication())
        results["8_websocket"] = ws_res
    except Exception as e:
        results["8_websocket"] = {"status": "FAIL", "error": str(e)}

    # 종합 결과 요약 출력
    print("\n" + "=" * 80)
    print("📋 [키움증권 모의거래 통신 전수 점검 종합 결과]")
    print("=" * 80)
    for k, v in results.items():
        st = v.get("status", "UNKNOWN")
        sec = v.get("elapsed_sec", 0)
        print(f"• {k:<25}: [{st}] (소요: {sec}s)")
    print("=" * 80)

    # JSON 저장
    output_file = PROJECT_ROOT / "data" / "kiwoom_mock_comm_check_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"📁 상세 리턴 데이터 JSON 저장 완료: {output_file}")

    return results

if __name__ == "__main__":
    run_kiwoom_mock_comms_check()
