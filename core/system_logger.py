import logging
logger = logging.getLogger(__name__)
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import deque

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "data"
LOGS_DIR.mkdir(exist_ok=True)
SYSTEM_LOGS_FILE = LOGS_DIR / "system_logs.jsonl"

class SystemLogger:
    """
    [Lumos 중앙 집중형 실시간 시스템 로거 (Quant System Log Hub)]
    - 모든 서브모듈(라이브러너, 키움브로커, MoE 오케스트레이터, 서킷브레이커, 웹서버)의
      실시간 동작 로그를 일괄 수집하고 JSONL 파일 및 메모리 링 버퍼에 영구 보존
    - 웹 콘솔(/api/logs)에 실시간 고속 스트리밍 데이터 제공
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(SystemLogger, cls).__new__(cls)
            cls._instance._init_logger()
        return cls._instance

    def _init_logger(self):
        self.log_file = SYSTEM_LOGS_FILE
        self.ring_buffer = deque(maxlen=500)
        self._load_initial_logs()

    def _load_initial_logs(self):
        """기존 로그 파일에서 최근 100건 로드"""
        if self.log_file.exists():
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    for line in lines[-100:]:
                        if line.strip():
                            try:
                                self.ring_buffer.append(json.loads(line.strip()))
                            except Exception:
                                pass
            except Exception:
                pass

        if len(self.ring_buffer) == 0:
            # Initial log notices do not verify external service or runtime status.
            self.log("INFO", "SystemCore", "Initial log notices created; system readiness is not verified.")
            self.log("KIWOOM", "KiwoomBroker", "Broker authentication and account balance synchronization are not verified.")
            self.log("MOE", "MoEOrchestrator", "AI model loading and activation are not verified.")
            self.log("INFO", "MarketGuard", "Market guard configuration and activation are not verified.")
            self.log("INFO", "LiveRunner", "Trading runner readiness and execution status are not verified.")

    def log(self, level: str, source: str, message: str, meta: Optional[Dict[str, Any]] = None):
        """
        새로운 로그 이벤트 기록
        - level: 'INFO', 'TRADE', 'MOE', 'KIWOOM', 'WARN', 'ERROR'
        """
        now = datetime.now()
        entry = {
            "id": int(time.time() * 1000),
            "timestamp": now.strftime("%H:%M:%S"),
            "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "level": level.upper(),
            "source": source,
            "message": message,
            "meta": meta or {}
        }

        self.ring_buffer.append(entry)

        # 콘솔 표준 출력
        prefix = f"[{entry['timestamp']}][{entry['level']}][{entry['source']}]"
        try:
            logger.info(f"{prefix} {message}")
        except Exception:
            pass

        # JSONL 파일에 영구 기록
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def get_logs(self, limit: int = 100, level: str = "ALL") -> List[Dict[str, Any]]:
        """웹 콘솔 조회를 위한 최근 로그 반환"""
        logs = list(self.ring_buffer)
        if level != "ALL":
            target_level = level.upper()
            logs = [l for l in logs if l.get("level") == target_level or (target_level == "ERROR" and l.get("level") in ["WARN", "ERROR"])]
        return logs[-limit:]

# Global Singleton
system_logger = SystemLogger()
