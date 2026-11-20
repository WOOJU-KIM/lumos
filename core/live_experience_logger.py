import os
import json
import time
import sqlite3
import threading
import logging
import queue
import atexit
from contextlib import closing
import config
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
from config import DATA_DIR
from core.learning_audit import execution_evidence, json_value

logger = logging.getLogger(__name__)

DEFAULT_CSV_PATH = DATA_DIR / "live_trades.csv"
DEFAULT_DB_PATH = DATA_DIR / "live_experience.db"

CSV_COLUMNS = [
    "trade_id",
    "mode",
    "symbol",
    "entry_time",
    "exit_time",
    "hold_minutes",
    "intended_entry_price",
    "actual_entry_price",
    "entry_slippage_usd",
    "intended_exit_price",
    "actual_exit_price",
    "exit_slippage_usd",
    "quantity",
    "pnl_pct",
    "pnl_usd",
    "pnl_krw",
    "exit_reason",
    "mfe_pct",
    "mae_pct",
    "gbdt_confidence",
    "cross_dir",
    "buy_attempts",
    "sell_attempts",
    "features_json"
]

class LiveExperienceLogger:
    """
    [Lumos V3 실시간 트레이딩 경험 데이터 로거 (AI 재학습용 Dual Tier Hub)]
    1. CSV (data/live_trades.csv):
       - 판다스(pd.read_csv) 및 엑셀 친화적 단일 평면 장부
       - 진입 시점 48개 피처 스냅샷, 실측 슬리피지, MFE/MAE, 보유시간 영구 기록
    2. SQLite (data/live_experience.db):
       - ACID 트랜잭션 무결성 보장
       - 거래 내역(live_trades), 3초 스마트 타임아웃/주문 이벤트(live_order_events), 에러 전문(live_system_errors) 3개 테이블 분할 적재
    3. Thread-Safe:
       - threading.Lock을 통한 동시성 제어로 매매 루프 간섭 0% 보장
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LiveExperienceLogger, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, db_path: Optional[Path] = None, csv_path: Optional[Path] = None):
        if getattr(self, "_initialized", False):
            return
        self.db_path = db_path or DEFAULT_DB_PATH
        self.csv_path = csv_path or DEFAULT_CSV_PATH
        self.learning_db_path = self.db_path.with_suffix(self.db_path.suffix + ".learning.db")
        self._write_lock = threading.Lock()
        self._init_db()
        self._init_csv()
        try:
            with closing(self._get_learning_connection()) as learning:
                learning.execute("""CREATE TABLE IF NOT EXISTS learning_events (
                    event_id TEXT PRIMARY KEY, event_type TEXT NOT NULL,
                    observed_at TEXT NOT NULL, payload_json TEXT NOT NULL)""")
                learning.commit()
        except Exception:
            logger.error("Learning journal unavailable; append-only fallback will be used")
        self._learning_events = queue.Queue()
        self._learning_failures = []
        self._learning_writer = threading.Thread(target=self._learning_writer_loop, daemon=True)
        self._learning_writer.start()
        atexit.register(self.flush_learning_events)
        self._initialized = True

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _get_learning_connection(self):
        # Separate file and no lock wait: detailed evidence cannot lock trade storage.
        return sqlite3.connect(self.learning_db_path, timeout=0)

    def _init_db(self):
        """SQLite 데이터베이스 테이블 초기화"""
        with self._write_lock:
            conn = self._get_connection()
            try:
                c = conn.cursor()
                # 1. 완료된 실전/모의 거래 전용 테이블
                c.execute("""
                CREATE TABLE IF NOT EXISTS live_trades (
                    trade_id TEXT PRIMARY KEY,
                    mode TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT NOT NULL,
                    hold_minutes REAL DEFAULT 0.0,
                    intended_entry_price REAL DEFAULT 0.0,
                    actual_entry_price REAL NOT NULL,
                    entry_slippage_usd REAL DEFAULT 0.0,
                    intended_exit_price REAL DEFAULT 0.0,
                    actual_exit_price REAL NOT NULL,
                    exit_slippage_usd REAL DEFAULT 0.0,
                    quantity INTEGER NOT NULL,
                    pnl_pct REAL NOT NULL,
                    pnl_usd REAL NOT NULL,
                    pnl_krw INTEGER DEFAULT 0,
                    exit_reason TEXT NOT NULL,
                    mfe_pct REAL DEFAULT 0.0,
                    mae_pct REAL DEFAULT 0.0,
                    gbdt_confidence REAL DEFAULT 0.0,
                    cross_dir TEXT DEFAULT 'HOLD',
                    buy_attempts INTEGER DEFAULT 1,
                    sell_attempts INTEGER DEFAULT 1,
                    features_json TEXT DEFAULT '{}',
                    created_at TEXT NOT NULL
                )
                """)

                # 2. 주문 체이싱 및 3초 타임아웃 이벤트 테이블
                c.execute("""
                CREATE TABLE IF NOT EXISTS live_order_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    attempt INTEGER DEFAULT 1,
                    order_no TEXT DEFAULT '',
                    price REAL DEFAULT 0.0,
                    quantity INTEGER DEFAULT 0,
                    status TEXT NOT NULL,
                    note TEXT DEFAULT ''
                )
                """)

                # 3. 증권사 API 에러 및 시스템 예외 전문 테이블
                c.execute("""
                CREATE TABLE IF NOT EXISTS live_system_errors (
                    error_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    source TEXT NOT NULL,
                    error_type TEXT NOT NULL,
                    error_code TEXT DEFAULT '',
                    message TEXT NOT NULL,
                    payload_json TEXT DEFAULT '{}'
                )
                """)
                conn.commit()
            finally:
                conn.close()

    def _init_csv(self):
        """CSV 파일 헤더 생성 (미존재 시)"""
        with self._write_lock:
            if not self.csv_path.exists():
                try:
                    df = pd.DataFrame(columns=CSV_COLUMNS)
                    df.to_csv(self.csv_path, index=False, encoding="utf-8-sig")
                except Exception:
                    pass

    def record_trade(self, trade_data: Dict[str, Any]) -> str:
        """
        완료된 1회 거래를 CSV 및 SQLite DB에 안전하게 동시 저장
        """
        now_dt = datetime.now()
        now_str = now_dt.strftime("%Y-%m-%d %H:%M:%S")

        # 1. 고유 Trade ID 생성
        trade_id = trade_data.get("trade_id")
        if not trade_id:
            trade_id = f"LIVE_{now_dt.strftime('%Y%m%d_%H%M%S')}_{trade_data.get('symbol', 'STK')}"

        # 2. 파생 메트릭 계산
        entry_time_str = trade_data.get("entry_time", now_str)
        exit_time_str = trade_data.get("exit_time", now_str)
        hold_min = float(trade_data.get("hold_minutes", 0.0))
        if hold_min <= 0.0:
            try:
                t_in = datetime.strptime(entry_time_str, "%Y-%m-%d %H:%M:%S")
                t_out = datetime.strptime(exit_time_str, "%Y-%m-%d %H:%M:%S")
                hold_min = round(max((t_out - t_in).total_seconds() / 60.0, 0.0), 2)
            except Exception:
                hold_min = 0.0

        act_entry_px = float(trade_data.get("actual_entry_price", 0.0))
        int_entry_px = float(trade_data.get("intended_entry_price", act_entry_px))
        entry_slip = round(act_entry_px - int_entry_px, 4)

        act_exit_px = float(trade_data.get("actual_exit_price", 0.0))
        int_exit_px = float(trade_data.get("intended_exit_price", act_exit_px))
        exit_slip = round(int_exit_px - act_exit_px, 4)

        qty = int(trade_data.get("quantity", 0))
        pnl_pct = float(trade_data.get("pnl_pct", 0.0))
        pnl_usd = float(trade_data.get("pnl_usd", round((act_exit_px - act_entry_px) * qty, 2)))
        pnl_krw = int(trade_data.get("pnl_krw", int(pnl_usd * 1350)))

        features = trade_data.get("features", {})
        if isinstance(features, str):
            features = json.loads(features)
        features = json_value(dict(features)) if isinstance(features, dict) else {}
        required = list(trade_data.get("learning_feature_names", []))
        missing = [name for name in required if features.get(name) is None]
        features["__learning_audit__"] = {
            "snapshot_id": trade_data.get("learning_snapshot_id"),
            "model_sha256": trade_data.get("active_model_sha256"),
            "model_id": trade_data.get("active_model_id"),
            "feature_schema": trade_data.get("active_model_schema"),
            "feature_names": trade_data.get("learning_feature_names", []),
            "exit_price_source": trade_data.get("exit_price_source", "UNVERIFIED_LEGACY_VALUE"),
            "fee_usd": trade_data.get("fee_usd"),
            "fee_status": "OBSERVED" if trade_data.get("fee_usd") is not None else "UNKNOWN_NOT_ZERO",
            "execution_order_references": trade_data.get("execution_order_references", {}),
            "risk_settings": trade_data.get("risk_settings", {}),
            "missing_features": missing,
            "feature_complete": bool(required) and not missing,
            "net_pnl_status": "UNVERIFIED_UNTIL_EXECUTION_AND_FEES_RECONCILED",
            "raw_trade_values": {
                "entry_time": entry_time_str, "exit_time": exit_time_str,
                "actual_entry_price": act_entry_px, "actual_exit_price": act_exit_px,
                "intended_entry_price": int_entry_px, "intended_exit_price": int_exit_px,
                "quantity": qty, "pnl_pct": pnl_pct, "pnl_usd": pnl_usd,
                "gross_pnl_from_recorded_prices_usd": (act_exit_px - act_entry_px) * qty,
                "utc_offset_at_recording": now_dt.astimezone().strftime("%z"),
            },
        }
        if not required or missing:
            logger.warning("Trade learning snapshot incomplete: schema_available=%s missing_count=%s",
                           bool(required), len(missing))
        features_json_str = json.dumps(features, ensure_ascii=False, allow_nan=False)

        row_dict = {
            "trade_id": trade_id,
            "mode": trade_data.get("mode", "VIRTUAL"),
            "symbol": trade_data.get("symbol", config.BASE_ASSET_LONG),
            "entry_time": entry_time_str,
            "exit_time": exit_time_str,
            "hold_minutes": hold_min,
            "intended_entry_price": round(int_entry_px, 2),
            "actual_entry_price": round(act_entry_px, 2),
            "entry_slippage_usd": entry_slip,
            "intended_exit_price": round(int_exit_px, 2),
            "actual_exit_price": round(act_exit_px, 2),
            "exit_slippage_usd": exit_slip,
            "quantity": qty,
            "pnl_pct": round(pnl_pct, 2),
            "pnl_usd": round(pnl_usd, 2),
            "pnl_krw": pnl_krw,
            "exit_reason": trade_data.get("exit_reason", "MANUAL"),
            "mfe_pct": round(float(trade_data.get("mfe_pct", 0.0)), 2),
            "mae_pct": round(float(trade_data.get("mae_pct", 0.0)), 2),
            "gbdt_confidence": round(float(trade_data.get("gbdt_confidence", 0.0)), 4),
            "cross_dir": trade_data.get("cross_dir", "HOLD"),
            "buy_attempts": int(trade_data.get("buy_attempts", 1)),
            "sell_attempts": int(trade_data.get("sell_attempts", 1)),
            "features_json": features_json_str
        }

        db_saved = csv_saved = False
        with self._write_lock:
            # 1. SQLite 적재
            conn = None
            try:
                conn = self._get_connection()
                c = conn.cursor()
                c.execute("""
                INSERT OR REPLACE INTO live_trades (
                    trade_id, mode, symbol, entry_time, exit_time, hold_minutes,
                    intended_entry_price, actual_entry_price, entry_slippage_usd,
                    intended_exit_price, actual_exit_price, exit_slippage_usd,
                    quantity, pnl_pct, pnl_usd, pnl_krw, exit_reason,
                    mfe_pct, mae_pct, gbdt_confidence, cross_dir,
                    buy_attempts, sell_attempts, features_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row_dict["trade_id"], row_dict["mode"], row_dict["symbol"],
                    row_dict["entry_time"], row_dict["exit_time"], row_dict["hold_minutes"],
                    row_dict["intended_entry_price"], row_dict["actual_entry_price"], row_dict["entry_slippage_usd"],
                    row_dict["intended_exit_price"], row_dict["actual_exit_price"], row_dict["exit_slippage_usd"],
                    row_dict["quantity"], row_dict["pnl_pct"], row_dict["pnl_usd"], row_dict["pnl_krw"],
                    row_dict["exit_reason"], row_dict["mfe_pct"], row_dict["mae_pct"],
                    row_dict["gbdt_confidence"], row_dict["cross_dir"], row_dict["buy_attempts"],
                    row_dict["sell_attempts"], row_dict["features_json"], now_str
                ))
                conn.commit()
                db_saved = True
            except Exception:
                logger.error("Trade database write failed; attempting CSV fallback", exc_info=True)
            finally:
                if conn is not None: conn.close()

            # 2. CSV 파일 Append
            try:
                df_row = pd.DataFrame([row_dict])
                df_row = df_row[CSV_COLUMNS]
                if not self.csv_path.exists():
                    df_row.to_csv(self.csv_path, index=False, encoding="utf-8-sig")
                else:
                    df_row.to_csv(self.csv_path, mode="a", header=False, index=False, encoding="utf-8-sig")
                csv_saved = True
            except Exception:
                logger.error("Trade CSV write failed; database_saved=%s", db_saved, exc_info=True)

        if not (db_saved or csv_saved):
            raise IOError("Both trade storage destinations failed")

        return trade_id

    def record_learning_decision(self, snapshot):
        if not isinstance(snapshot, dict):
            logger.error("Learning decision snapshot unavailable")
            return None
        self._learning_events.put_nowait((snapshot["event_id"], "DECISION", snapshot["observed_at"], snapshot))
        return snapshot["event_id"]

    def record_learning_execution(self, data):
        from uuid import uuid4
        evidence = execution_evidence(data)
        self._learning_events.put_nowait((uuid4().hex, "EXECUTION", datetime.now().astimezone().isoformat(), evidence))

    def record_learning_cycle(self, summary):
        from uuid import uuid4
        self._learning_events.put_nowait((uuid4().hex, "DECISION_OUTCOME", datetime.now().astimezone().isoformat(), json_value(summary)))

    def _learning_writer_loop(self):
        while True:
            event = self._learning_events.get()
            try:
                encoded = json.dumps(event[3], ensure_ascii=False, allow_nan=False)
                conn = self._get_learning_connection()
                try:
                    conn.execute("INSERT INTO learning_events VALUES (?, ?, ?, ?)", (*event[:3], encoded))
                    conn.commit()
                finally:
                    conn.close()
            except Exception:
                logger.error("Learning database event write failed; using append-only fallback", exc_info=True)
                try:
                    fallback = self.csv_path.with_suffix(self.csv_path.suffix + ".learning.jsonl")
                    with fallback.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(dict(event_id=event[0],event_type=event[1],observed_at=event[2],payload=event[3]),allow_nan=False)+"\n")
                except Exception:
                    self._learning_failures.append(event)
                    logger.critical("Learning event could not be persisted; retained in memory", exc_info=True)
            finally:
                self._learning_events.task_done()

    def flush_learning_events(self):
        self._learning_events.join()

    def record_order_event(
        self,
        symbol: str,
        action: str,
        attempt: int,
        order_no: str,
        price: float,
        quantity: int,
        status: str,
        note: str = ""
    ):
        """주문 이벤트(체결, 3초 타임아웃, 호가 재발주, 100% 현금 보존 취소) 기록"""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        from uuid import uuid4
        evidence = execution_evidence({"order_no": order_no, "symbol": symbol})
        evidence.update(source="OBSERVED_APP_ORDER_EVENT_NOT_EXECUTION_PRICE_PROOF",
                        action=action, status=status, attempt=int(attempt),
                        recorded_price=float(price), recorded_quantity=int(quantity))
        self._learning_events.put_nowait((uuid4().hex, "ORDER", datetime.now().astimezone().isoformat(), evidence))
        with self._write_lock:
            conn = self._get_connection()
            try:
                c = conn.cursor()
                c.execute("""
                INSERT INTO live_order_events (
                    timestamp, symbol, action, attempt, order_no, price, quantity, status, note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (now_str, symbol, action, attempt, str(order_no), float(price), int(quantity), status, note))
                conn.commit()
            except Exception:
                pass
            finally:
                conn.close()

    def record_error_event(
        self,
        source: str,
        error_type: str,
        error_code: str,
        message: str,
        payload: Any = None
    ):
        """증권사 통신 에러 및 내부 예외 전문 기록"""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        payload_str = json.dumps(payload, ensure_ascii=False) if payload else "{}"
        with self._write_lock:
            conn = self._get_connection()
            try:
                c = conn.cursor()
                c.execute("""
                INSERT INTO live_system_errors (
                    timestamp, source, error_type, error_code, message, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """, (now_str, source, error_type, str(error_code), str(message), payload_str))
                conn.commit()
            except Exception:
                pass
            finally:
                conn.close()

    def get_trades_dataframe(self) -> pd.DataFrame:
        """저장된 전체 실전 거래 내역을 판다스 DataFrame으로 반환 (AI 학습용 - DB 최우선)"""
        frames = []
        with self._write_lock:
            # 1. DB 최우선 로드 (Single Source of Truth)
            conn = None
            try:
                conn = self._get_connection()
                df = pd.read_sql_query("SELECT * FROM live_trades ORDER BY entry_time ASC", conn)
                if not df.empty:
                    frames.append(df)
            except Exception:
                logger.error("Trade learning database read failed; checking CSV fallback")
            finally:
                if conn is not None: conn.close()

            # 2. DB 실패 시 보조 수단으로 CSV 읽기
            if self.csv_path.exists():
                try:
                    frames.append(pd.read_csv(self.csv_path))
                except Exception:
                    logger.error("Trade learning CSV read failed")
            
            if not frames:
                return pd.DataFrame(columns=CSV_COLUMNS)
            # Database wins on duplicates; CSV-only successfully persisted trades survive.
            return pd.concat(frames, ignore_index=True).drop_duplicates("trade_id", keep="first").sort_values("entry_time")

    def get_summary_stats(self) -> Dict[str, Any]:
        """모의/실전 누적 트레이딩 성적 및 슬리피지 통계 요약"""
        df = self.get_trades_dataframe()
        if df.empty:
            return {
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate_pct": 0.0,
                "total_pnl_usd": 0.0,
                "total_pnl_krw": 0,
                "avg_slippage_usd": 0.0,
                "avg_hold_minutes": 0.0
            }

        total_trades = len(df)
        wins = len(df[df["pnl_pct"] > 0])
        losses = len(df[df["pnl_pct"] <= 0])
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
        total_usd = float(df["pnl_usd"].sum())
        total_krw = int(df["pnl_krw"].sum())
        avg_entry_slip = float(df["entry_slippage_usd"].mean()) if "entry_slippage_usd" in df.columns else 0.0
        avg_hold = float(df["hold_minutes"].mean()) if "hold_minutes" in df.columns else 0.0

        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": round(win_rate, 2),
            "total_pnl_usd": round(total_usd, 2),
            "total_pnl_krw": total_krw,
            "avg_slippage_usd": round(avg_entry_slip, 4),
            "avg_hold_minutes": round(avg_hold, 1)
        }
