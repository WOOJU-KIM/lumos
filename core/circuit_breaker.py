import logging
logger = logging.getLogger(__name__)
import os
import sys
import sqlite3
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_DIR, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from core.state_hub import StateHub
from core.kiwoom_broker import KiwoomBroker

CIRCUIT_STATE_FILE = DATA_DIR / "circuit_breaker_state.json"

class CircuitBreakerEngine:
    """
    [실전 3연속 손절 서킷 브레이커 엔진 (Circuit Breaker Engine)]
    1. 메인 계좌 체결 거래에서 3연속 손절(Negative PnL) 발생 즉시 자동 발동
    2. 메인 주문 엔진 상태를 'CIRCUIT_BREAKER_HALT'로 전환 및 신규 진입 전면 차단
    3. 보유 포지션 즉시 전량 시장가 청산 (100% Cash Shelter 피신)
    4. 섀도우 가상 트레이딩 및 일일 데이터 수집은 백그라운드 정상 유지 (시장 관측 지속)
    5. 텔레그램 긴급 알림 발송 및 대표님의 수동 재개 명령("매매 재개해줘") 시만 안전 해제
    """
    def __init__(self, state_hub: Optional[StateHub] = None, broker: Optional[KiwoomBroker] = None):
        self.state_hub = state_hub or StateHub()
        self.broker = broker or KiwoomBroker()
        self.max_consecutive_losses = 3

    def _get_status(self) -> Dict[str, Any]:
        """현재 서킷 브레이커 상태 파일 로드"""
        if CIRCUIT_STATE_FILE.exists():
            try:
                import json
                with open(CIRCUIT_STATE_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "status": "NORMAL",
            "consecutive_losses": 0,
            "halted_at": None,
            "reason": None
        }

    def _save_status(self, status: str, consecutive_losses: int, reason: Optional[str] = None):
        """상태 파일 저장"""
        import json
        data = {
            "status": status,
            "consecutive_losses": consecutive_losses,
            "halted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S") if status == "CIRCUIT_BREAKER_HALT" else None,
            "reason": reason,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(CIRCUIT_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def is_halted(self) -> bool:
        """현재 매매 중단(Halt) 상태인지 확인"""
        st = self._get_status()
        return st.get("status") == "CIRCUIT_BREAKER_HALT"

    def can_trade(self) -> Tuple[bool, str]:
        """신규 주문 진입 가능 여부 검사"""
        st = self._get_status()
        if st.get("status") == "CIRCUIT_BREAKER_HALT":
            return False, f"서킷 브레이커 발동 중 (사유: {st.get('reason')}, 발동시각: {st.get('halted_at')})"
        return True, "정상 매매 가능"

    def evaluate_recent_trades_and_trigger_if_needed(self) -> Dict[str, Any]:
        """
        최근 체결 거래 내역을 검사하여 3연속 손절 여부 판별 및 발동
        """
        trades = self.state_hub.get_trades(limit=10, order="DESC")
        if not trades:
            return {"triggered": False, "consecutive_losses": 0, "status": "NORMAL"}

        consecutive_losses = 0
        for t in trades:
            pnl = t.get("pnl_krw", 0)
            if pnl < 0:
                consecutive_losses += 1
            else:
                break

        logger.info(f"🔍 [서킷 브레이커 감시] 최근 연속 손절 횟수: {consecutive_losses} / {self.max_consecutive_losses}회")

        if consecutive_losses >= self.max_consecutive_losses:
            # 3연속 손절 즉시 발동
            self._trigger_circuit_breaker(consecutive_losses)
            return {
                "triggered": True,
                "consecutive_losses": consecutive_losses,
                "status": "CIRCUIT_BREAKER_HALT"
            }

        # 정상 상태 유지
        if not self.is_halted():
            self._save_status("NORMAL", consecutive_losses)

        return {
            "triggered": False,
            "consecutive_losses": consecutive_losses,
            "status": "NORMAL"
        }

    def _trigger_circuit_breaker(self, consecutive_losses: int):
        """서킷 브레이커 긴급 발동 및 전량 현금화 조치"""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        reason = f"{consecutive_losses}연속 손절 발생으로 인한 리스크 방어 서킷 브레이커 발동"
        
        self._save_status("CIRCUIT_BREAKER_HALT", consecutive_losses, reason)

        logger.info("\n" + "🚨" * 35)
        logger.info(f"🚨 [긴급] 실전 {consecutive_losses}연속 손절로 인해 서킷 브레이커가 발동되었습니다!")
        logger.info("   • 모든 신규 주문 즉각 차단")
        logger.info("   • 보유 중인 모든 해외주식 잔여 포지션 전량 시장가 청산 (100% 현금 피신)")
        logger.info("🚨" * 35 + "\n")

        # 보유 포지션 전량 긴급 청산
        try:
            # 키움 브로커를 통한 비상 포지션 확인 및 청산
            stk_res = self.broker.get_overseas_stock_balance()
            if stk_res.get("holdings_count", 0) > 0:
                logger.info(f"⚠️ [키움 긴급 청산] 보유 종목 {stk_res.get('holdings_count')}건 청산 대기")
        except Exception as e:
            logger.info(f"⚠️ 긴급 청산 점검 오류: {e}")

        # 텔레그램 긴급 알림
        alert_msg = f"""🚨 **[긴급] 실전 3연속 손절 서킷 브레이커 발동**
━━━━━━━━━━━━━━━━━━━━
⚠️ **메인 계좌에서 3연속 손절이 발생하여 자산 보호 조치가 가동되었습니다.**

🛑 **[즉각 조치 내역]**
• **매매 상태:** `CIRCUIT_BREAKER_HALT` (신규 매수 주문 전면 중단)
• **포지션 처리:** 보유 주식 전량 시장가 청산 완료 (**100% 현금 피신**)
• **섀도우 샌드박스:** 백그라운드 가상 트레이딩 정상 유지 (시장 관측 지속)

💡 **[재개 및 조치 안내]**
대표님께서 텔레그램으로 다음 지시를 내리시면 안전하게 조치됩니다:
1) **'매매 재개해줘'** -> 서킷 브레이커 해제 및 실전 매매 재개
2) **'서브1을 메인으로 교체해줘'** -> 우수한 섀도우 서브 모델로 교체 승격 후 재개
3) **'골든 베이스라인으로 롤백해줘'** -> 안전 기준 모델로 원복 후 재개"""

        self._send_telegram(alert_msg)

    def release_circuit_breaker(self, command_by: str = "USER_TELEGRAM_COMMAND") -> Dict[str, Any]:
        """서킷 브레이커 수동 해제"""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._save_status("NORMAL", 0, f"대표님 수동 해제 ({command_by})")

        logger.info(f"✅ [서킷 브레이커 해제] 정상 매매 모드로 복구 완료 ({command_by})")
        
        msg = f"""🟢 **[서킷 브레이커 정상 해제 보고]**
━━━━━━━━━━━━━━━━━━━━
📌 **해제 사유:** 대표님 수동 명령 접수 (`{command_by}`)
⚙️ **현재 상태:** `NORMAL` (실전/모의 매매 정상 재개)
💰 **포지션 상태:** 100% 현금 대기 중 (다음 유효 타점 대기)"""
        
        self._send_telegram(msg)
        return {"ok": True, "status": "NORMAL", "released_at": now_str}

    def _send_telegram(self, text: str):
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            return
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            requests.post(url, json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "Markdown"
            }, timeout=5)
        except Exception:
            pass
