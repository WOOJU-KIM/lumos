import sqlite3
import datetime
import threading
import queue
from pathlib import Path
from zoneinfo import ZoneInfo
import config
from core.system_logger import system_logger

MAX_SHADOW_SLOTS = 10
SLOT_KEYS = (
    "slot_id",
    "active",
    "symbol",
    "entry_time",
    "entry_price",
    "qty",
    "highest_price",
    "stop_price",
    "exit_time",
    "exit_price",
    "exit_reason",
    "net_return_pct"
)

class ShadowTracker:
    def __init__(self, db_path: Path = None):
        self._lock = threading.RLock()
        self.slots = [self._create_empty_slot(i) for i in range(MAX_SHADOW_SLOTS)]
        self.last_entered_candle_id = None
        
        self.db_path = db_path or (config.DATA_DIR / "shadow_trades_live.db")
        self._db_queue = queue.Queue()
        self._init_db()
        
        self._worker_thread = threading.Thread(target=self._db_worker, daemon=True)
        self._worker_thread.start()

    @staticmethod
    def _create_empty_slot(slot_id: int) -> dict:
        """정확히 요구된 12개 키만 가지는 빈 슬롯 dict 생성"""
        return {
            "slot_id": slot_id,
            "active": False,
            "symbol": None,
            "entry_time": None,
            "entry_price": 0.0,
            "qty": 0,
            "highest_price": 0.0,
            "stop_price": 0.0,
            "exit_time": None,
            "exit_price": 0.0,
            "exit_reason": None,
            "net_return_pct": 0.0
        }

    def _init_db(self):
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path, timeout=1.0)
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS shadow_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    virtual_entry_time TEXT,
                    virtual_entry_price REAL,
                    gbdt_confidence REAL,
                    virtual_exit_time TEXT,
                    virtual_exit_price REAL,
                    pnl_pct REAL,
                    exit_reason TEXT,
                    highest_price_seen REAL,
                    candles_passed INTEGER,
                    sl_pct REAL
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS shadow_slots_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    slot_id INTEGER,
                    active INTEGER,
                    symbol TEXT,
                    entry_time TEXT,
                    entry_price REAL,
                    qty INTEGER,
                    highest_price REAL,
                    stop_price REAL,
                    exit_time TEXT,
                    exit_price REAL,
                    exit_reason TEXT,
                    net_return_pct REAL
                )
            """)
            try:
                c.execute("SELECT sl_pct FROM shadow_logs LIMIT 1")
            except sqlite3.OperationalError as e:
                if "no such column" in str(e):
                    c.execute("ALTER TABLE shadow_logs ADD COLUMN sl_pct REAL")
            conn.commit()
            conn.close()
        except Exception as e:
            system_logger.log("ERROR", "ShadowTracker", f"[SHADOW_EXCEPTION] DB 초기화 실패: {e}")

    def _db_worker(self):
        """비동기 큐에 쌓인 DB 저장 요청을 처리하는 워커 스레드 (지시된 12개 키 dict 그대로 영구 저장)"""
        while True:
            try:
                task = self._db_queue.get()
                if task is None:
                    break
                
                slot_data = task
                conn = sqlite3.connect(self.db_path, timeout=5.0)
                c = conn.cursor()
                
                # 1. 지시된 12개 키 dict 구조 그대로 shadow_slots_history에 저장
                c.execute("""
                    INSERT INTO shadow_slots_history 
                    (slot_id, active, symbol, entry_time, entry_price, qty, highest_price, stop_price, exit_time, exit_price, exit_reason, net_return_pct)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    slot_data.get("slot_id"),
                    0,
                    slot_data.get("symbol"),
                    slot_data.get("entry_time"),
                    slot_data.get("entry_price"),
                    slot_data.get("qty", 1),
                    slot_data.get("highest_price"),
                    slot_data.get("stop_price"),
                    slot_data.get("exit_time"),
                    slot_data.get("exit_price"),
                    slot_data.get("exit_reason"),
                    slot_data.get("net_return_pct")
                ))

                # 2. 레거시 shadow_logs 호환 저장
                c.execute("""
                    INSERT INTO shadow_logs 
                    (symbol, virtual_entry_time, virtual_entry_price, gbdt_confidence, virtual_exit_time, virtual_exit_price, pnl_pct, exit_reason, highest_price_seen, candles_passed, sl_pct)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    slot_data.get("symbol"),
                    slot_data.get("entry_time"),
                    slot_data.get("entry_price"),
                    0.0,
                    slot_data.get("exit_time"),
                    slot_data.get("exit_price"),
                    slot_data.get("net_return_pct"),
                    slot_data.get("exit_reason"),
                    slot_data.get("highest_price"),
                    0,
                    0.0
                ))
                conn.commit()
                conn.close()

                # 3. JSONL 파일에도 지시된 12개 키 dict 그대로 한 줄씩 append 저장
                try:
                    import json
                    jsonl_path = self.db_path.parent / "shadow_trades_history.jsonl"
                    with open(jsonl_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(slot_data, ensure_ascii=False) + "\n")
                except Exception as je:
                    system_logger.log("WARNING", "ShadowTracker", f"JSONL 기록 실패: {je}")
            except Exception as e:
                system_logger.log("ERROR", "ShadowTracker", f"[SHADOW_EXCEPTION] DB 비동기 저장 중 에러: {e}")
            finally:
                self._db_queue.task_done()

    def get_last_closed_15m_candle(self, dt: datetime.datetime = None):
        """
        미 동부시간(NYT) 기준으로 완전히 마감된 최근 15분봉의 식별자(candle_id)를 반환.
        정규장(09:30~16:00 NYT) 기준:
        - 09:45:00 이전에는 마감된 15분봉이 없으므로 None 반환.
        - 09:45:00~09:59:59 -> 09:30~09:45 봉 마감 식별자 반환.
        - 주말 또는 16:00 이후 장 마감 시각 이후에는 None 반환.
        """
        try:
            ny_tz = ZoneInfo("America/New_York")
            if dt is None:
                now_ny = datetime.datetime.now(ny_tz)
            else:
                if dt.tzinfo is None:
                    now_ny = dt.replace(tzinfo=ny_tz)
                else:
                    now_ny = dt.astimezone(ny_tz)

            # 월(0) ~ 금(4) 평일만 정규장
            if now_ny.weekday() >= 5:
                return None

            now_m = now_ny.hour * 60 + now_ny.minute
            market_open_m = 9 * 60 + 30   # 570 (09:30)
            market_close_m = 16 * 60       # 960 (16:00)

            # 09:45 이전에는 아직 완전히 마감된 15분봉이 존재하지 않음
            if now_m < market_open_m + 15:
                return None

            # 정규장 종료 이후
            if now_m > market_close_m:
                return None

            completed_bars = (now_m - market_open_m) // 15
            bar_start_m = market_open_m + (completed_bars - 1) * 15
            bar_end_m = market_open_m + completed_bars * 15

            sh, sm = divmod(bar_start_m, 60)
            eh, em = divmod(bar_end_m, 60)
            date_str = now_ny.strftime("%Y-%m-%d")
            return f"{date_str} {sh:02d}:{sm:02d}-{eh:02d}:{em:02d}"
        except Exception as e:
            system_logger.log("ERROR", "ShadowTracker", f"[SHADOW_EXCEPTION] 캔들 마감 계산 실패: {e}")
            return None

    def try_enter_shadow_trade(
        self,
        symbol: str,
        price: float,
        current_time = None,
        atr_pct: float = None,
        force_candle_id: str = None
    ) -> bool:
        """
        15분봉 마감 시 호출되는 SHADOW 강제 진입 엔진.
        - 실제 주문 API 호출 절대 금지.
        - 마감된 15분봉 1회 한정 진입 (동일 봉 중복 차단).
        - 최대 10개 독립 슬롯 관리 (빈 슬롯 부재 시 NO_EMPTY_SHADOW_SLOT).
        - 유효하지 않은 가격/시각 거부 (SHADOW_ENTRY_REJECTED_INVALID_PRICE).
        """
        try:
            with self._lock:
                # 1. 가격 유효성 검사
                if price is None or not isinstance(price, (int, float)) or price <= 0.0:
                    system_logger.log("WARNING", "ShadowTracker", f"[SHADOW_ENTRY_REJECTED_INVALID_PRICE] 가격 유효하지 않음: {price}")
                    return False

                # 2. 시각 유효성 및 datetime 변환
                ny_tz = ZoneInfo("America/New_York")
                if current_time is None:
                    curr_dt = datetime.datetime.now(ny_tz)
                elif isinstance(current_time, str):
                    try:
                        curr_dt = datetime.datetime.fromisoformat(current_time)
                        if curr_dt.tzinfo is None:
                            curr_dt = curr_dt.replace(tzinfo=ny_tz)
                    except Exception:
                        try:
                            curr_dt = datetime.datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S").replace(tzinfo=ny_tz)
                        except Exception:
                            system_logger.log("WARNING", "ShadowTracker", f"[SHADOW_ENTRY_REJECTED_INVALID_PRICE] 시각 파싱 실패: {current_time}")
                            return False
                elif isinstance(current_time, datetime.datetime):
                    if current_time.tzinfo is None:
                        curr_dt = current_time.replace(tzinfo=ny_tz)
                    else:
                        curr_dt = current_time.astimezone(ny_tz)
                else:
                    system_logger.log("WARNING", "ShadowTracker", f"[SHADOW_ENTRY_REJECTED_INVALID_PRICE] 지원하지 않는 시각 타입: {type(current_time)}")
                    return False

                # 3. 15분봉 마감 여부 검증
                candle_id = force_candle_id or self.get_last_closed_15m_candle(curr_dt)
                if not candle_id:
                    system_logger.log("INFO", "ShadowTracker", "[SHADOW_ENTRY_REJECTED_INVALID_PRICE] 마감된 15분봉 부재 (정규장 미마감 봉)")
                    return False

                # 4. 동일 봉 중복 진입 방지
                if self.last_entered_candle_id == candle_id:
                    system_logger.log("INFO", "ShadowTracker", f"[DUPLICATE_CANDLE_SHADOW_ENTRY_BLOCKED] 동일 15분봉 중복 진입 차단: {candle_id}")
                    return False

                # 5. 빈 슬롯 탐색 (최대 10개)
                empty_slot = None
                for s in self.slots:
                    if not s["active"]:
                        empty_slot = s
                        break

                if empty_slot is None:
                    system_logger.log("WARNING", "ShadowTracker", "[NO_EMPTY_SHADOW_SLOT] 신규 SHADOW 진입 거부: 10개 슬롯 모두 점유 중")
                    return False

                # 6. 실전 동일 초기 손절가(stop_price) 계산
                sl_atr_mult = getattr(config, "SL_ATR_MULTIPLIER", 1.5)
                sl_min = getattr(config, "SL_MIN_PCT", 0.02)
                sl_max = getattr(config, "SL_MAX_PCT", 0.032)

                if atr_pct is not None and atr_pct > 0:
                    effective_sl_pct = max(sl_min, min(sl_max, atr_pct * sl_atr_mult))
                else:
                    effective_sl_pct = sl_min

                initial_stop = price * (1.0 - effective_sl_pct)
                entry_time_str = curr_dt.strftime("%Y-%m-%d %H:%M:%S")

                # 7. 슬롯 할당 (정확히 12개 키 유지)
                slot_id = empty_slot["slot_id"]
                empty_slot["active"] = True
                empty_slot["symbol"] = str(symbol).upper()
                empty_slot["entry_time"] = entry_time_str
                empty_slot["entry_price"] = float(price)
                empty_slot["qty"] = 1
                empty_slot["highest_price"] = float(price)
                empty_slot["stop_price"] = round(float(initial_stop), 4)
                empty_slot["exit_time"] = None
                empty_slot["exit_price"] = 0.0
                empty_slot["exit_reason"] = None
                empty_slot["net_return_pct"] = 0.0

                self.last_entered_candle_id = candle_id

                system_logger.log(
                    "INFO",
                    "ShadowTracker",
                    f"[SHADOW_ENTRY] Slot {slot_id} 진입 성공: {empty_slot['symbol']} @ {price:.2f} | Stop: {empty_slot['stop_price']:.2f} | 봉: {candle_id}"
                )
                return True
        except Exception as e:
            system_logger.log("ERROR", "ShadowTracker", f"[SHADOW_EXCEPTION] SHADOW 진입 실패: {e}")
            return False

    def add_virtual_trade(self, symbol: str, price: float, confidence: float = 0.0, sl_pct: float = None, current_time = None):
        """기존 live_runner.py 호출과의 하위 호환성을 위한 래퍼 메서드"""
        atr_pct = None
        if sl_pct is not None and sl_pct > 0:
            atr_pct = (sl_pct / 100.0) / getattr(config, "SL_ATR_MULTIPLIER", 1.5)
        return self.try_enter_shadow_trade(symbol, price, current_time=current_time, atr_pct=atr_pct)

    def check_ticks(self, symbol: str, current_price: float, current_time = None):
        """
        실시간 틱 수신 시 보유 슬롯 독립 갱신 및 청산 규칙 적용.
        - 최고가 갱신: SHADOW_HIGHEST_PRICE_UPDATED
        - 트레일링 스탑가 갱신: SHADOW_STOP_PRICE_UPDATED
        - 청산 우선순위: 손절/트레일링 -> 최대 익절 -> 90분 타임스탑
        """
        try:
            if current_price is None or current_price <= 0.0:
                return

            sym = str(symbol).upper()
            curr_dt = self._parse_dt(current_time)
            time_str = curr_dt.strftime("%Y-%m-%d %H:%M:%S")

            trigger_pct = getattr(config, "TRAILING_TRIGGER_PCT", 0.015)
            drop_pct = getattr(config, "TRAILING_DROP_PCT", 0.003)
            max_tp_pct = getattr(config, "MAX_TP_PCT", 0.10)
            time_stop_min = getattr(config, "TIME_STOP_MINUTES", 90)

            with self._lock:
                for slot_id, slot in enumerate(self.slots):
                    if not slot["active"] or slot["symbol"] != sym:
                        continue

                    entry_px = slot["entry_price"]

                    # 1. 최고가 갱신
                    if current_price > slot["highest_price"]:
                        slot["highest_price"] = float(current_price)
                        system_logger.log(
                            "DEBUG",
                            "ShadowTracker",
                            f"[SHADOW_HIGHEST_PRICE_UPDATED] Slot {slot_id} ({sym}) 최고가 갱신: {current_price:.2f}"
                        )

                    # 2. 트레일링 발동 여부 및 손절가 상향
                    is_trailing_active = slot["highest_price"] >= entry_px * (1.0 + trigger_pct)
                    if is_trailing_active:
                        candidate_stop = round(slot["highest_price"] * (1.0 - drop_pct), 4)
                        if candidate_stop > slot["stop_price"]:
                            slot["stop_price"] = candidate_stop
                            system_logger.log(
                                "INFO",
                                "ShadowTracker",
                                f"[SHADOW_STOP_PRICE_UPDATED] Slot {slot_id} ({sym}) 트레일링 스탑가 갱신: {candidate_stop:.2f}"
                            )

                    # 3. 청산 판정 (보수적/비관적 우선순위 준수)
                    # A. 손절 또는 트레일링 스탑 청산
                    if current_price <= slot["stop_price"]:
                        reason = "TRAILING_STOP" if is_trailing_active else "STOP_LOSS"
                        log_evt = "SHADOW_EXIT_TRAILING_STOP" if is_trailing_active else "SHADOW_EXIT_STOP_LOSS"
                        self._close_slot(slot_id, current_price, time_str, reason, log_evt)
                        continue

                    # B. 최대 익절 청산 (MAX_TP)
                    if current_price >= entry_px * (1.0 + max_tp_pct):
                        self._close_slot(slot_id, current_price, time_str, "MAX_TP", "SHADOW_EXIT_MAX_TP")
                        continue

                    # C. 보유시간 90분 타임스탑 체크
                    if self._is_time_stop_elapsed(slot["entry_time"], curr_dt, time_stop_min):
                        self._close_slot(slot_id, current_price, time_str, "TIME_STOP", "SHADOW_EXIT_TIME_STOP")
                        continue
        except Exception as e:
            system_logger.log("ERROR", "ShadowTracker", f"[SHADOW_EXCEPTION] check_ticks 실행 에러: {e}")

    def increment_candle(self, current_time = None):
        """15분 주기 브리핑 시 호출되어 90분 타임스탑을 일괄 검사"""
        self.check_time_stops(current_time)

    def check_time_stops(self, current_time = None):
        """90분 경과한 활성 슬롯 강제 청산"""
        try:
            curr_dt = self._parse_dt(current_time)
            time_str = curr_dt.strftime("%Y-%m-%d %H:%M:%S")
            time_stop_min = getattr(config, "TIME_STOP_MINUTES", 90)

            with self._lock:
                for slot_id, slot in enumerate(self.slots):
                    if not slot["active"]:
                        continue

                    if self._is_time_stop_elapsed(slot["entry_time"], curr_dt, time_stop_min):
                        exit_px = slot["highest_price"] if slot["highest_price"] > 0 else slot["entry_price"]
                        self._close_slot(slot_id, exit_px, time_str, "TIME_STOP", "SHADOW_EXIT_TIME_STOP")
        except Exception as e:
            system_logger.log("ERROR", "ShadowTracker", f"[SHADOW_EXCEPTION] check_time_stops 실행 에러: {e}")

    def flush_eod(self, current_time = None, exit_prices: dict = None):
        """장 마감 시(15:55 NYT) 전량 시장가 청산 및 슬롯 초기화"""
        try:
            curr_dt = self._parse_dt(current_time)
            time_str = curr_dt.strftime("%Y-%m-%d %H:%M:%S")

            with self._lock:
                for slot_id, slot in enumerate(self.slots):
                    if not slot["active"]:
                        continue

                    sym = slot["symbol"]
                    px = 0.0
                    if exit_prices and sym in exit_prices and exit_prices[sym] > 0:
                        px = exit_prices[sym]
                    elif slot["highest_price"] > 0:
                        px = slot["highest_price"]
                    else:
                        px = slot["entry_price"]

                    self._close_slot(slot_id, px, time_str, "EOD_CLEAR", "SHADOW_EXIT_EOD_CLEAR")
        except Exception as e:
            system_logger.log("ERROR", "ShadowTracker", f"[SHADOW_EXCEPTION] flush_eod 실행 에러: {e}")

    def _close_slot(self, slot_id: int, exit_price: float, exit_time_str: str, exit_reason: str, log_event: str):
        """
        슬롯 청산 처리:
        1. 순수익률 계산 (수수료/슬리피지 0.20% 선차감)
        2. 구조화 로그 기록 (청산 이벤트 및 SHADOW_FINAL_RETURN)
        3. 비동기 DB 저장 큐에 전달
        4. 슬롯을 새로운 empty dict로 100% 교체하여 상태 격리
        """
        slot = self.slots[slot_id]
        if not slot["active"]:
            return

        entry_px = slot["entry_price"]
        raw_return = (exit_price / entry_px) - 1.0 if entry_px > 0 else 0.0
        fee_slippage = getattr(config, "BACKTEST_FEE_SLIPPAGE", 0.0020)
        net_return_pct = round((raw_return - fee_slippage) * 100.0, 4)

        slot["exit_time"] = exit_time_str
        slot["exit_price"] = float(exit_price)
        slot["exit_reason"] = exit_reason
        slot["net_return_pct"] = net_return_pct

        system_logger.log(
            "INFO",
            "ShadowTracker",
            f"[{log_event}] Slot {slot_id} ({slot['symbol']}) 가상 청산 완료 | 사유: {exit_reason} | 진입가: {entry_px:.2f} | 청산가: {exit_price:.2f}"
        )
        system_logger.log(
            "INFO",
            "ShadowTracker",
            f"[SHADOW_FINAL_RETURN] Slot {slot_id} ({slot['symbol']}) 최종 순수익률: {net_return_pct:.2f}% (수수료/슬리피지 {fee_slippage*100:.2f}% 반영)"
        )

        # DB 비동기 저장 요청
        self._db_queue.put(dict(slot))

        # 완전히 새로운 empty dict 객체로 교체하여 이전 거래 잔여 데이터 100% 격리
        self.slots[slot_id] = self._create_empty_slot(slot_id)

    @staticmethod
    def _parse_dt(t):
        ny_tz = ZoneInfo("America/New_York")
        if t is None:
            return datetime.datetime.now(ny_tz)
        if isinstance(t, str):
            try:
                dt = datetime.datetime.fromisoformat(t)
                return dt if dt.tzinfo else dt.replace(tzinfo=ny_tz)
            except Exception:
                try:
                    return datetime.datetime.strptime(t, "%Y-%m-%d %H:%M:%S").replace(tzinfo=ny_tz)
                except Exception:
                    return datetime.datetime.now(ny_tz)
        if isinstance(t, datetime.datetime):
            return t if t.tzinfo else t.replace(tzinfo=ny_tz)
        return datetime.datetime.now(ny_tz)

    @staticmethod
    def _is_time_stop_elapsed(entry_time_str: str, curr_dt: datetime.datetime, limit_minutes: int) -> bool:
        if not entry_time_str:
            return False
        try:
            entry_dt = datetime.datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=curr_dt.tzinfo)
            elapsed_sec = (curr_dt - entry_dt).total_seconds()
            return elapsed_sec >= (limit_minutes * 60)
        except Exception:
            return False
