import sqlite3
import datetime
import threading
import queue
from pathlib import Path
from config import DATA_DIR, GBDT_CONFIDENCE_THRESHOLD, TIME_STOP_MINUTES, MAX_TP_PCT, TRAILING_TRIGGER_PCT, TRAILING_DROP_PCT, SL_MIN_PCT, SL_MAX_PCT
from core.system_logger import system_logger

class ShadowTracker:
    def __init__(self):
        self.shadow_positions = []
        self.db_path = DATA_DIR / "shadow_trades_live.db"
        
        # 1. 비동기 DB 저장을 위한 큐와 워커 스레드 초기화
        self._db_queue = queue.Queue()
        self._init_db()
        
        self._worker_thread = threading.Thread(target=self._db_worker, daemon=True)
        self._worker_thread.start()

    def _init_db(self):
        # 타임아웃 1.0초 설정으로 Lock 시 즉각 반환되도록 (초기화 단계)
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
        # 컬럼 마이그레이션이 필요하다면 여기서만 처리
        try:
            c.execute("SELECT sl_pct FROM shadow_logs LIMIT 1")
        except sqlite3.OperationalError as e:
            if "no such column" in str(e):
                c.execute("ALTER TABLE shadow_logs ADD COLUMN sl_pct REAL")
        conn.commit()
        conn.close()

    def _db_worker(self):
        """비동기 큐에 쌓인 DB 저장 요청을 처리하는 워커 스레드"""
        while True:
            try:
                task = self._db_queue.get()
                if task is None: # 종료 시그널
                    break
                
                # task는 (pos, exit_time, exit_price, reason, pnl_pct) 튜플
                pos, exit_time, exit_price, reason, pnl_pct = task
                
                # 타임아웃을 주어 메인 스레드에 영향이 없도록 하며, 워커 스레드도 무한 대기 방지
                conn = sqlite3.connect(self.db_path, timeout=5.0)
                c = conn.cursor()
                c.execute("""
                    INSERT INTO shadow_logs 
                    (symbol, virtual_entry_time, virtual_entry_price, gbdt_confidence, virtual_exit_time, virtual_exit_price, pnl_pct, exit_reason, highest_price_seen, candles_passed, sl_pct)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pos['symbol'], pos['virtual_entry_time'], pos['virtual_entry_price'], pos['gbdt_confidence'],
                    exit_time, exit_price, round(pnl_pct, 2), reason, pos['highest_price_seen'], pos['candles_passed'], pos.get('sl_pct', 2.5)
                ))
                conn.commit()
                conn.close()
                system_logger.log("INFO", "ShadowTracker", f"가상 청산 완료: {pos['symbol']} | 사유: {reason} | 수익률: {pnl_pct:.2f}%")
                
            except Exception as e:
                system_logger.log("ERROR", "ShadowTracker", f"DB 비동기 저장 중 에러 발생: {e}")
            finally:
                self._db_queue.task_done()

    def add_virtual_trade(self, symbol: str, price: float, confidence: float, sl_pct: float = 2.5):
        if len(self.shadow_positions) >= 10:
            self.shadow_positions.pop(0)

        self.shadow_positions.append({
            'symbol': symbol,
            'virtual_entry_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'virtual_entry_price': price,
            'gbdt_confidence': confidence,
            'highest_price_seen': price,
            'candles_passed': 0,
            'sl_pct': sl_pct
        })
        system_logger.log("INFO", "ShadowTracker", f"가상 진입 기록: {symbol} @ {price} (Conf: {confidence}%, SL: -{sl_pct}%)")

    def increment_candle(self):
        for pos in self.shadow_positions:
            pos['candles_passed'] += 1
        self._check_time_stop()

    def _check_time_stop(self):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        to_remove = []
        max_candles = int(TIME_STOP_MINUTES / 15)
        for pos in self.shadow_positions:
            if pos['candles_passed'] >= max_candles:
                self._save_to_db(pos, now, pos['highest_price_seen'], "TIME_STOP")
                to_remove.append(pos)
        
        for pos in to_remove:
            self.shadow_positions.remove(pos)

    def flush_eod(self):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for pos in self.shadow_positions:
            self._save_to_db(pos, now, pos['highest_price_seen'], "EOD_CLEAR")
        self.shadow_positions.clear()

    def check_ticks(self, symbol: str, current_price: float):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        to_remove = []

        max_tp = MAX_TP_PCT * 100.0
        trigger_pct = TRAILING_TRIGGER_PCT * 100.0
        drop_pct = TRAILING_DROP_PCT * 100.0

        for pos in self.shadow_positions:
            if pos['symbol'] != symbol:
                continue

            entry_px = pos['virtual_entry_price']
            
            if current_price > pos['highest_price_seen']:
                pos['highest_price_seen'] = current_price

            highest_px = pos['highest_price_seen']
            pnl_pct = (current_price / entry_px - 1.0) * 100.0

            if pnl_pct >= max_tp:
                self._save_to_db(pos, now, current_price, "MAX_TP", pnl_pct)
                to_remove.append(pos)
                continue

            highest_pnl = (highest_px / entry_px - 1.0) * 100.0
            if highest_pnl >= trigger_pct:
                drawdown = (current_price / highest_px - 1.0) * 100.0
                if drawdown <= -drop_pct:
                    self._save_to_db(pos, now, current_price, "TRAILING_STOP", pnl_pct)
                    to_remove.append(pos)
                    continue

            if pnl_pct <= -pos['sl_pct']:
                self._save_to_db(pos, now, current_price, "STOP_LOSS", pnl_pct)
                to_remove.append(pos)
                continue

        for pos in to_remove:
            self.shadow_positions.remove(pos)

    def _save_to_db(self, pos, exit_time, exit_price, reason, pnl_pct=0.0):
        if pnl_pct == 0.0:
            pnl_pct = (exit_price / pos['virtual_entry_price'] - 1.0) * 100.0
        # 동기식 접근으로 인한 블로킹 방지를 위해 백그라운드 큐로 전달
        self._db_queue.put((pos, exit_time, exit_price, reason, pnl_pct))
