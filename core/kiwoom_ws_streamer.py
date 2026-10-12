import asyncio
import json
import logging
import threading
import time
from datetime import datetime
from typing import Dict, Any, Callable, Optional, List
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import config

import websockets
from core.kiwoom_broker import KiwoomBroker
from core.system_logger import system_logger

logger = logging.getLogger("KiwoomWebSocket")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s][KiwoomWS] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class KiwoomWebSocketStreamer:
    """
    [Lumos 고성능 실시간 웹소켓(WebSocket) 스트리밍 엔진]
    - 키움증권 해외주식 실시간 틱/호가(ust01000) 및 체결 통보를 밀리초(ms) 단위 이벤트 수신
    - 콜백 함수(on_tick)를 통해 +3.0% 익절 / -2.0% 손절 / ATR 조건 도달 즉시 0.01초 즉시 발주
    - 연결 단절 시 자동 재접속(Auto-Reconnect) 및 REST API 하이브리드 자동 백업 지원
    """
    def __init__(self, broker: Optional[KiwoomBroker] = None):
        self.broker = broker or KiwoomBroker()
        self.is_simulation = self.broker.is_simulation
        
        # WebSocket Base URL (공식 포트 10000 / 미국주식 엔드포인트)
        if self.is_simulation:
            self.ws_url = "wss://mockapi.kiwoom.com:10000/api/us/websocket"
        else:
            self.ws_url = "wss://api.kiwoom.com:10000/api/us/websocket"

        self.subscribed_symbols: List[str] = ["SOXL", "SOXS", config.BASE_ASSET_LONG, config.BASE_ASSET_SHORT, config.MACRO_TREND_SYMBOL, "NVDA", "QQQ", "VIXY", "IEF"]
        self.latest_prices: Dict[str, float] = {}
        self.latest_ticks: Dict[str, Dict[str, Any]] = {}
        self.ws_tick_received: Dict[str, bool] = {}     # 실제 WebSocket 틱 수신 여부 {symbol: bool}
        self.ws_tick_count: Dict[str, int] = {}         # 실제 WebSocket 틱 수신 횟수 {symbol: count}
        self.last_ws_tick_time: Dict[str, float] = {}   # 가장 최근 WebSocket 틱 수신 시각 {symbol: timestamp}
        self.is_running = False
        self.is_connected = False
        self.ws_auth_success = False
        
        self._callbacks: List[Callable[[str, float, Dict[str, Any]], None]] = []
        self._execution_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def has_received_live_tick(self, symbol: str) -> bool:
        """해당 종목에 대해 실제 키움 실시간 WebSocket 틱을 1회 이상 수신했는지 검증"""
        sym_up = symbol.upper().strip()
        return self.is_connected and self.ws_tick_received.get(sym_up, False)

    def reset_session_ticks(self):
        """매일 정규장 개장(09:30 NYT) 시점에 실시간 틱 수신 상태를 초기화하여 당일 틱 수신 검증"""
        self.ws_tick_received.clear()
        self.ws_tick_count.clear()
        self.last_ws_tick_time.clear()
        logger.info("🔄 [WebSocket 틱 세션 초기화] 정규장 개장: 당일 실시간 틱 감시 시작")

    def is_live_stream_active(self, symbol: str, max_staleness_sec: float = 60.0) -> bool:
        """해당 종목의 실시간 WebSocket 스트림이 현재 활성화(최근 n초 이내 틱 유입) 상태인지 검증"""
        sym_up = symbol.upper().strip()
        if not (self.is_connected and self.ws_tick_received.get(sym_up, False)):
            return False
        last_t = self.last_ws_tick_time.get(sym_up, 0.0)
        return (time.time() - last_t) <= max_staleness_sec

    def register_callback(self, cb: Callable[[str, float, Dict[str, Any]], None]):
        """실시간 틱 수신 시 실행할 콜백 함수 등록"""
        if cb not in self._callbacks:
            self._callbacks.append(cb)

    def register_execution_callback(self, cb: Callable[[Dict[str, Any]], None]):
        """실시간 체결 통보 수신 시 실행할 콜백 함수 등록"""
        if cb not in self._execution_callbacks:
            self._execution_callbacks.append(cb)

    def get_latest_price(self, symbol: str, default: Optional[float] = None) -> float:
        """메모리에 캐시된 가장 최신 실시간 틱 가격 반환 (0ms 지연, 없으면 REST 현재가 자동 폴백)"""
        if symbol in self.latest_prices and self.latest_prices[symbol] > 0:
            return self.latest_prices[symbol]
        if default is not None and default > 0:
            return default
        try:
            quote = self.broker.get_stock_quote(symbol)
            px = float(quote.get("last_price", 0.0))
            if px > 0:
                self.latest_prices[symbol] = px
                return px
        except Exception:
            pass
        return default if default is not None else 0.0

    async def _perform_handshake_and_subscribe(self, websocket, token: str):
        """키움 WebSocket LOGIN 인증 및 미국주식 REG 종목 구독 등록"""
        # 1. LOGIN 인증 패킷 송출
        login_packet = {
            "trnm": "LOGIN",
            "token": token
        }
        await websocket.send(json.dumps(login_packet))
        login_res_raw = await asyncio.wait_for(websocket.recv(), timeout=5)
        login_res = json.loads(login_res_raw)
        
        if login_res.get("return_code") != 0:
            raise RuntimeError(f"WebSocket LOGIN 실패: {login_res.get('return_msg')}")
        
        self.ws_auth_success = True
        logger.info(f"🔑 [WebSocket LOGIN 인증 성공] {login_res.get('return_msg', '정상 세션 연결')}")

        # 2. 미국주식 REG 종목 실시간 구독 등록
        reg_packet = {
            "trnm": "REG",
            "grp_no": "1",
            "refresh": "1",
            "data": [
                {
                    "item": [{"jmcode": sym, "stex_tp": "NY" if sym in ["SOXL", "SOXS", "SPY", "DIA", "VIXY"] else "ND"} for sym in self.subscribed_symbols],
                    "type": ["0A", "0B", "FT"]
                }
            ]
        }
        await websocket.send(json.dumps(reg_packet))
        reg_res_raw = await asyncio.wait_for(websocket.recv(), timeout=5)
        reg_res = json.loads(reg_res_raw)
        logger.info(f"📡 [WebSocket REG 구독 성공] 대상 종목: {self.subscribed_symbols} (Code: {reg_res.get('return_code')})")

    async def _ws_loop(self):
        """비동기 웹소켓 메인 수신 루프 (연결 실패 시 지능형 REST 폴링 백업 가동)"""
        last_log_time = 0.0
        while self.is_running:
            try:
                token = self.broker.get_access_token()
                headers = {"authorization": f"Bearer {token}"}
                logger.info(f"🌐 [WebSocket 연결 시도] {self.ws_url} (모드: {self.broker.mode_str})...")
                
                async with websockets.connect(self.ws_url, additional_headers=headers, ping_interval=30, ping_timeout=10, open_timeout=5) as ws:
                    self.is_connected = True
                    logger.info("✅ [WebSocket 연결 성공] 실시간 초고속 틱 스트리밍 세션 오픈")
                    system_logger.log("INFO", "WebSocket", f"⚡ 키움 실시간 WebSocket 세션 연결 완료 ({self.broker.mode_str})")

                    # LOGIN 및 REG 구독 핸드셰이크 집행
                    await self._perform_handshake_and_subscribe(ws, token)

                    while self.is_running:
                        msg = await ws.recv()
                        self._process_message(msg)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.is_connected = False
                self.ws_auth_success = False
                now_t = time.time()
                if now_t - last_log_time > 30:
                    logger.warning(f"⚠️ [WebSocket 연결 예외]: {e} ➔ 초고속 REST 폴링 백업 실시간 가동 중")
                    last_log_time = now_t
                
                try:
                    await asyncio.sleep(5)
                except asyncio.CancelledError:
                    break

    def _poll_rest_fallback_loop(self):
        """WebSocket 단절/미지원 시 실시간 가격 피드를 공급하는 REST 백업 스레드"""
        while self.is_running:
            try:
                if not self.is_connected:
                    for sym in self.subscribed_symbols:
                        quote = self.broker.get_stock_quote(sym)
                        px = float(quote.get("last_price", 0.0))
                        if px > 0:
                            self._update_price(sym, px, {"provider": "REST_FALLBACK", "quote": quote})
                time.sleep(1.0)
            except Exception as e:
                time.sleep(2.0)

    def _process_message(self, raw_msg: str):
        """수신된 실시간 틱 데이터 및 주문/체결 통보 파싱 및 콜백 전파 (Kiwoom 0A/0B/0C/FT/CHEG 지원)"""
        try:
            if not raw_msg:
                return

            # [1] JSON 메시지인 경우
            if raw_msg.startswith("{"):
                data = json.loads(raw_msg)
                
                # A. 실시간 체결 통보 검출 (단일/루트 레벨)
                if any(k in data for k in ["odno", "ord_no", "order_no", "tot_che_qty", "acc_filled_qty", "ft_tot_ccld_qty", "che_qty", "ft_ccld_qty"]):
                    logger.info(f"🔔 [WebSocket 실시간 체결 통보 포착 (JSON 루트)]: {self.broker._redact_log(data)}")
                    self._dispatch_execution_event(data)
                    return

                # B. REAL 데이터 배열 포맷 (trnm: REAL / CHEG / ORDER)
                if data.get("trnm") in ["REAL", "CHEG", "ORDER", "EXEC"] and "data" in data:
                    for d_item in data["data"]:
                        real_type = str(d_item.get("type", "") or data.get("type", "")).upper()
                        vals = d_item.get("values", d_item)

                        # 체결 통보 (0C, CHEG, ORDER 또는 주문번호/체결수량 필드 보유 시)
                        # ※ 주의: FT는 해외주식 10단계 호가(Ask/Bid) 시세 패킷이므로 체결 통보에서 반드시 제외
                        if (real_type in ["0C", "CHEG", "ORDER"] and real_type != "FT") or any(k in vals for k in ["odno", "ord_no", "order_no", "ft_tot_ccld_qty", "acc_filled_qty"]):
                            logger.info(f"🔔 [WebSocket 실시간 체결 통보 포착 (배열)]: {self.broker._redact_log(vals)}")
                            self._dispatch_execution_event(vals)
                            continue

                        # 시세/틱 데이터 (0A: 체결가, 0B: 호가, FT: 해외주식 실시간 호가/시세 등)
                        s = str(d_item.get("item", "") or d_item.get("jmcode", "") or vals.get("symbol", "")).strip().upper()
                        p_float = 0.0

                        # 1) 체결가 / 현재가 필드 우선 탐색 (10: 현재가, last_price, cur_price)
                        p_val = vals.get("last_price") or vals.get("cur_price") or vals.get("price") or vals.get("10")
                        if p_val is not None:
                            try:
                                p_float = abs(float(str(p_val).replace(",", "").strip()))
                            except Exception:
                                p_float = 0.0

                        # 2) FT 호가 필드 탐색 (41: 매도1호가, 51: 매수1호가)
                        if p_float <= 0.0:
                            ask_val = vals.get("41")
                            bid_val = vals.get("51")
                            try:
                                ask_px = abs(float(str(ask_val).replace(",", "").strip())) if ask_val else 0.0
                                bid_px = abs(float(str(bid_val).replace(",", "").strip())) if bid_val else 0.0
                                if ask_px > 0 and bid_px > 0:
                                    p_float = round((ask_px + bid_px) / 2.0, 2)
                                elif ask_px > 0:
                                    p_float = ask_px
                                elif bid_px > 0:
                                    p_float = bid_px
                            except Exception:
                                pass

                        if s and p_float > 0:
                            self._update_price(s, p_float, d_item, from_ws=True)
                    return

                # C. 단일 종목 시세 JSON 포맷
                sym = data.get("symbol", "") or data.get("symb", "") or data.get("item", "") or data.get("jmcode", "")
                px = float(data.get("price", 0.0) or data.get("last_price", 0.0) or data.get("cur_px", 0.0) or data.get("cur_price", 0.0) or data.get("now_pric", 0.0))
                if sym and px > 0:
                    self._update_price(sym, px, data, from_ws=True)
                    return

            # [2] '|' 구분자 스트림인 경우
            elif "|" in raw_msg:
                tokens = raw_msg.split("|")
                # 체결 통보 스트림 (예: 0|0C|계좌|주문번호|종목|... 또는 |0C|...)
                if len(tokens) >= 5 and any(t in ["0C", "CHEG", "ORDER"] for t in tokens[:3]):
                    exec_dict = {
                        "type": tokens[1] if len(tokens) > 1 else "0C",
                        "account_no": tokens[2] if len(tokens) > 2 else "",
                        "order_no": tokens[3] if len(tokens) > 3 else "",
                        "symbol": tokens[4] if len(tokens) > 4 else "",
                        "che_qty": tokens[7] if len(tokens) > 7 else "",
                        "che_price": tokens[8] if len(tokens) > 8 else "",
                        "nccs_qty": tokens[9] if len(tokens) > 9 else "",
                        "tot_che_qty": tokens[10] if len(tokens) > 10 else "",
                        "raw": raw_msg
                    }
                    logger.info(f"🔔 [WebSocket 실시간 체결 통보 포착 (Pipe)]: {exec_dict}")
                    self._dispatch_execution_event(exec_dict)
                    return

                # 일반 틱 시세 스트림
                if len(tokens) >= 4:
                    sym = tokens[1]
                    try:
                        px = abs(float(tokens[3]))
                        self._update_price(sym, px, {"raw": raw_msg}, from_ws=True)
                    except ValueError:
                        pass
        except Exception as e:
            logger.debug(f"WS 메시지 파싱 오류: {e}")

    def _update_price(self, symbol: str, price: float, extra: Dict[str, Any], from_ws: bool = False):
        """가격 캐시 갱신 및 즉시 콜백 호출 (1ms 이내 처리)"""
        sym_clean = str(symbol).strip().upper()
        self.latest_prices[sym_clean] = price
        self.latest_ticks[sym_clean] = {
            "symbol": sym_clean,
            "price": price,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "extra": extra
        }
        if from_ws:
            self.ws_tick_received[sym_clean] = True
            self.last_ws_tick_time[sym_clean] = time.time()
            self.ws_tick_count[sym_clean] = self.ws_tick_count.get(sym_clean, 0) + 1

        # 등록된 콜백 실행 (익절/손절 즉시 감시)
        for cb in self._callbacks:
            try:
                cb(sym_clean, price, self.latest_ticks[sym_clean])
            except Exception as e:
                logger.error(f"콜백 실행 오류: {e}")

    def _dispatch_execution_event(self, exec_data: Dict[str, Any]):
        """실시간 체결 통보 콜백 전파"""
        for cb in self._execution_callbacks:
            try:
                cb(exec_data)
            except Exception as e:
                logger.error(f"체결 콜백 실행 오류: {e}")

    def notify_simulated_execution(self, exec_data: Dict[str, Any]):
        """모의/실전 즉시 체결 이벤트 전파 (Rest 체결 확인 시 등)"""
        self._dispatch_execution_event(exec_data)

    def start(self):
        """백그라운드 스레드에서 WebSocket 리스너 및 REST 백업 폴러 비동기 가동"""
        if self.is_running:
            return
        self.is_running = True

        def run_ws():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            try:
                self._loop.run_until_complete(self._ws_loop())
            except asyncio.CancelledError:
                pass
            finally:
                try:
                    self._loop.close()
                except Exception:
                    pass

        self._thread = threading.Thread(target=run_ws, daemon=True, name="KiwoomWebSocketThread")
        self._thread.start()

        # REST 폴링 백업 스레드 가동
        self._fallback_thread = threading.Thread(target=self._poll_rest_fallback_loop, daemon=True, name="KiwoomRestFallbackThread")
        self._fallback_thread.start()

        logger.info("🚀 [KiwoomWebSocketStreamer + REST 실시간 백업 스트리머 가동 완료]")

    def stop(self):
        """웹소켓 리스너 및 백업 스트리머 정상 종료"""
        self.is_running = False
        self.is_connected = False
        if self._loop and self._loop.is_running():
            try:
                def _cancel_and_stop():
                    for task in asyncio.all_tasks(self._loop):
                        task.cancel()
                self._loop.call_soon_threadsafe(_cancel_and_stop)
            except Exception:
                pass
        logger.info("🛑 [KiwoomWebSocketStreamer 중지 완료]")
