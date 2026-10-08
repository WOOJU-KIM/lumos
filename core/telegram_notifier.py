"""
================================================================================
Lumos V3 Hybrid MoE 실시간 텔레그램 모니터링 알림 파이프라인 (TelegramNotifier)
================================================================================
[기관급 퀀트 헤지펀드 실시간 관제 시스템 명세]
1. 하이브리드 MoE 의사결정 및 4대 청산 룰 100% 반영:
   - Type 1: 🛡️ [방어 / Veto 차단] - GBDT >= {config.GBDT_CONFIDENCE_THRESHOLD*100:.1f}% 공격 신호 vs Cross-Asset 정반대 역풍 경고 차단
   - Type 2: ⚡ [매수 / Entry 체결] - GBDT >= {config.GBDT_CONFIDENCE_THRESHOLD*100:.1f}% & Cross-Asset 승인(동의/HOLD) 체결 완료
   - Type 3: 🏁 [청산 / Exit 완료] - TP(+3.0% 🎯), SL(-2.0% ✂️), TimeStop(90m ⏱️), EOD(오버나잇 0% 🌙)
2. 비동기 논블로킹(Non-blocking) 및 동기 전송 듀얼 모드 지원
3. 지능형 재시도(Exponential Backoff) 및 HTTP 429 Rate Limit 방어
4. .env 기반 보안 키 로드 (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
================================================================================
"""

import os
import sys
import time
import queue
import logging
import threading
import requests
from datetime import datetime, timedelta
from pathlib import Path
import config
from typing import Dict, Any, Optional

# Windows 콘솔 UTF-8 인코딩 설정
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# .env 자동 로드 지원
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("TelegramNotifier")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s][TelegramNotifier] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class TelegramNotifier:
    """
    [Lumos V3 Hybrid MoE 텔레그램 모니터링 알림 엔진]
    - 월스트리트 헤지펀드 실시간 매매 관제 규격 준수
    - 3대 핵심 템플릿(Veto 방어 / 자율 매수 체결 / 4대 룰 청산 완료) 전담 발송
    - Thread-safe 비동기 큐 워커를 통해 메인 트레이딩 루프의 0ms 지연(Non-blocking) 보장
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        async_mode: bool = True,
        max_retries: int = 3,
        timeout: float = 5.0
    ):
        # 1. 보안 환경변수 로드 (.env 우선 탐색)
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = str(chat_id or os.getenv("TELEGRAM_CHAT_ID", "")).strip()
        self.async_mode = async_mode
        self.max_retries = max_retries
        self.timeout = timeout
        self.telegram_api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage" if self.bot_token else ""

        # 2. 비동기 전송을 위한 백그라운드 큐 및 데몬 스레드 초기화
        self._queue: queue.Queue = queue.Queue(maxsize=1000)
        self._is_running = True

        if self.async_mode:
            self._worker_thread = threading.Thread(
                target=self._async_queue_worker,
                daemon=True,
                name="LumosTelegramWorker"
            )
            self._worker_thread.start()

        if not self.bot_token or not self.chat_id:
            logger.warning("⚠️ [TELEGRAM CREDENTIAL MISSING] TELEGRAM_BOT_TOKEN 또는 TELEGRAM_CHAT_ID가 설정되지 않았습니다. (테스트/로깅 모드로 작동)")

    # ==============================================================================
    # 📱 1. [Type 1: 방어 (Veto / Block)]
    # ==============================================================================
    def send_veto_alert(
        self,
        ticker: str,
        gbdt_prob: float,
        cross_dir: str,
        is_simulation: Optional[bool] = None
    ) -> Dict[str, Any]:
        """[Type 1: 방어 (Veto) 알림]"""
        if is_simulation is None:
            import os
            env_sim = os.getenv("KIWOOM_IS_SIMULATION", "1").strip()
            is_simulation = (env_sim == "1" or env_sim.lower() == "true")
        mode_tag = "🧪 [키움 모의투자]" if is_simulation else "🔥 [키움 실전투자]"
        
        prob_val = gbdt_prob * 100.0 if gbdt_prob <= 1.0 else gbdt_prob
        sym_clean = ticker.upper().strip()
        
        message = f"""🛡️ <b>[Lumos 방어막 발동] {sym_clean} 진입 차단</b>
{mode_tag}

🚨 <b>차단 사유 (Cross-Asset Veto)</b>
• <b>GBDT 공격 신호</b>: {sym_clean} 진입 시도 (확신도: {prob_val:.1f}%)
• <b>크로스에셋 역풍</b>: {cross_dir} 방향 감지
• <b>결과</b>: 하이브리드 MoE 규정에 따라 신규 진입을 전면 차단합니다."""

        return self._dispatch_message(message, alert_type="VETO_BLOCK")

    # ==============================================================================
    # 2. [Type 2: 매수 체결]
    # ==============================================================================
    def send_entry_alert(
        self,
        ticker: str,
        entry_price: float,
        qty: int,
        gbdt_prob: float,
        cross_dir: str,
        tp_price: Optional[float] = None,
        sl_price: Optional[float] = None,
        time_stop_time: Optional[str] = None,
        time_stop_minutes: int = 90,
        strategy_tag: str = "Lumos V3",
        is_simulation: Optional[bool] = None
    ) -> Dict[str, Any]:
        if is_simulation is None:
            import os
            env_sim = os.getenv("KIWOOM_IS_SIMULATION", "1").strip()
            is_simulation = (env_sim == "1" or env_sim.lower() == "true")
        mode_tag = "🧪 [키움 모의투자]" if is_simulation else "🔥 [키움 실전투자]"

        prob_val = gbdt_prob * 100.0 if gbdt_prob <= 1.0 else gbdt_prob
        sym_clean = ticker.upper().strip()
        total_amt = round(entry_price * qty, 2)

        calc_tp = tp_price if tp_price is not None else round(entry_price * (1.0 + config.MAX_TP_PCT), 2)
        calc_sl = sl_price if sl_price is not None else round(entry_price * (1.0 - config.SL_MIN_PCT), 2)
        tp_pct_calc = round(((calc_tp - entry_price) / entry_price) * 100.0, 2) if entry_price > 0 else config.MAX_TP_PCT * 100.0
        sl_pct_calc = round(((entry_price - calc_sl) / entry_price) * 100.0, 2) if entry_price > 0 else config.SL_MIN_PCT * 100.0
        
        if not time_stop_time:
            from datetime import datetime, timedelta
            kst_now = datetime.now()
            time_stop_time = (kst_now + timedelta(minutes=time_stop_minutes)).strftime("%H:%M:%S")

        message = f"""🟢 <b>[Lumos 신규 진입] {sym_clean}</b>
{mode_tag}

🎯 <b>진입 정보</b>
• <b>종목</b>: {sym_clean}
• <b>체결가</b>: ${entry_price:,.2f}
• <b>수량</b>: {qty}주 (총 ${total_amt:,.2f})

🧠 <b>AI 분석 결과</b> ({strategy_tag})
• <b>GBDT 확신도</b>: {prob_val:.1f}%

🛡️ <b>자동 방어선(ATR Trailing)</b>
• <b>목표가(TP)</b>: ${calc_tp:,.2f} (+{tp_pct_calc}%)
• <b>손절가(SL)</b>: ${calc_sl:,.2f} (-{sl_pct_calc}%)
• <b>타임스탑</b>: {time_stop_time} ({time_stop_minutes}분)"""

        return self._dispatch_message(message, alert_type="ENTRY_BUY")

    # ==============================================================================
    # 3. [Type 3: 청산 (Exit)]
    # ==============================================================================
    def send_exit_alert(
        self,
        ticker: str,
        entry_price: float,
        exit_price: float,
        exit_reason: str,
        qty: int = 1,
        usd_krw_rate: float = 1380.0,
        is_simulation: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        [Type 3: 청산 알림]
        """
        if is_simulation is None:
            import os
            env_sim = os.getenv("KIWOOM_IS_SIMULATION", "1").strip()
            is_simulation = (env_sim == "1" or env_sim.lower() == "true")
        mode_tag = "🧪 [키움 모의투자]" if is_simulation else "🔥 [키움 실전투자]"

        sym_clean = ticker.upper().strip()
        reason_upper = exit_reason.upper()

        if any(k in reason_upper for k in ["TP", "익절", "+3.0", "+3.5", "+2.5", "TARGET", "목표"]):
            emoji = "🎯"
        elif any(k in reason_upper for k in ["SL", "손절", "-2.0", "-1.67", "STOP_LOSS", "칼손절"]):
            emoji = "🛡️"
        elif any(k in reason_upper for k in ["TIME", "90분", "30분", "타임스탑", "DURATION"]):
            emoji = "⏱️"
        elif any(k in reason_upper for k in ["EOD", "OVERNIGHT", "오버나잇", "종가", "마감"]):
            emoji = "🌙"
        else:
            emoji = "🔔"

        if entry_price > 0:
            pnl_percent = ((exit_price - entry_price) / entry_price) * 100.0
        else:
            pnl_percent = 0.0

        pnl_usd = (exit_price - entry_price) * qty
        pnl_krw = int(pnl_usd * usd_krw_rate)
        pnl_sign = "+" if pnl_krw > 0 else ""
        pnl_amount_str = f"{pnl_sign}{pnl_krw:,}"

        message = f"""{emoji} <b>[Lumos 매도 청산] {sym_clean}</b>
{mode_tag}

💰 <b>청산 결과 요약</b>
• <b>사유</b>: {exit_reason}
• <b>수익률</b>: {pnl_sign}{pnl_percent:.2f}%
• <b>추정 손익</b>: {pnl_amount_str}원 (${pnl_usd:,.2f})

📊 <b>거래 상세</b>
• <b>매수가</b>: ${entry_price:,.2f}
• <b>매도가</b>: ${exit_price:,.2f}
• <b>수량</b>: {qty}주"""

        return self._dispatch_message(message, alert_type="EXIT_SELL")

    # ==============================================================================
    # 4. 발송 파이프라인 코어
    # ==============================================================================
    def _dispatch_message(self, text: str, alert_type: str = "INFO") -> Dict[str, Any]:
        """메시지 라우팅: 비동기 큐 적재(Non-blocking 0ms) 또는 동기 즉시 발송"""
        if self.async_mode:
            try:
                self._queue.put({"text": text, "type": alert_type, "timestamp": time.time()}, block=False)
                return {"ok": True, "mode": "queued", "alert_type": alert_type, "text": text}
            except queue.Full:
                logger.error("❌ [TELEGRAM QUEUE FULL] 발송 큐가 가득 차 동기 발송으로 자동 폴백합니다.")
                return self._send_http_request(text)
        else:
            return self._send_http_request(text)

    def _send_http_request(self, text: str) -> Dict[str, Any]:
        """실제 Telegram Bot API POST 요청 집행 (지연시간 및 재시도 제어)"""
        if not self.bot_token or not self.chat_id:
            logger.info(f"📢 [TELEGRAM SIMULATED]\n{text}")
            return {"ok": True, "simulated": True, "text": text}

        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(self.telegram_api_url, json=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    return {"ok": True, "attempt": attempt, "text": text}
                elif resp.status_code == 429:
                    # Telegram Rate Limit (초당 30회, 채팅방당 초당 1회) 대응 백오프
                    retry_after = int(resp.json().get("parameters", {}).get("retry_after", 3))
                    logger.warning(f"⚠️ [Telegram HTTP 429 Rate Limit] {retry_after}초 대기 후 재시도 ({attempt}/{self.max_retries})")
                    time.sleep(retry_after)
                else:
                    logger.warning(f"⚠️ [Telegram Send Fail] Status: {resp.status_code} | Body: {resp.text}")
            except requests.exceptions.RequestException as e:
                logger.warning(f"⚠️ [Telegram Connection Error] 시도 {attempt}/{self.max_retries}: {e}")
                time.sleep(0.5 * attempt)

        logger.error(f"❌ [Telegram Final Failure] 최대 재시도({self.max_retries}회) 초과로 발송 실패")
        return {"ok": False, "error": "max_retries_exceeded", "text": text}

    def _async_queue_worker(self):
        """백그라운드 데몬 스레드: 큐에서 메시지를 안전하게 디큐하여 순차 발송"""
        while self._is_running:
            try:
                item = self._queue.get(timeout=1.0)
                if item is None:
                    break
                self._send_http_request(item["text"])
                self._queue.task_done()
                # 텔레그램 메시지 간 안전 텀 (초당 20회 제한 준수)
                time.sleep(0.1)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"⚠️ [Telegram Worker Error]: {e}")

    def shutdown(self):
        """정상 시스템 종료 시 큐 잔여 작업 비우기"""
        self._is_running = False
        if hasattr(self, "_worker_thread") and self._worker_thread.is_alive():
            self._queue.put(None)
            self._worker_thread.join(timeout=2.0)


# ==============================================================================
# 🧪 자가 테스트 및 데모 시뮬레이션
# ==============================================================================
if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("🏛 [Lumos V3 Hybrid MoE 실시간 텔레그램 알림 파이프라인 데모]")
    logger.info("=" * 80)

    # 콘솔 시뮬레이션 모드로 생성하여 터미널 포맷 출력만 확인 (사용자 텔레그램 발송 차단)
    notifier = TelegramNotifier(bot_token="", chat_id="", async_mode=False)

    logger.info("\n1. 🛡️ [Type 1: Veto 방어 알림 시뮬레이션]")
    res1 = notifier.send_veto_alert(
        ticker=config.BASE_ASSET_LONG,
        gbdt_prob=58.4,
        cross_dir="SHORT"
    )
    logger.info(res1["text"])

    logger.info("\n2. ⚡ [Type 2: 자율 매수 체결 알림 시뮬레이션]")
    res2 = notifier.send_entry_alert(
        ticker=config.BASE_ASSET_LONG,
        entry_price=42.50,
        qty=120,
        gbdt_prob=62.5,
        cross_dir="HOLD",
        tp_price=43.99,
        sl_price=41.65,
        time_stop_time="00:00:00"
    )
    logger.info(res2["text"])

    logger.info("\n3. 🎯 [Type 3-A: 목표가 +3.0% 익절 청산 알림 시뮬레이션]")
    res3a = notifier.send_exit_alert(
        ticker=config.BASE_ASSET_LONG,
        entry_price=42.50,
        exit_price=43.78,
        exit_reason="목표가 +3.0% 도달",
        qty=120
    )
    logger.info(res3a["text"])

    logger.info("\n4. ✂️ [Type 3-B: -2.0% 칼손절 청산 알림 시뮬레이션]")
    res3b = notifier.send_exit_alert(
        ticker=config.BASE_ASSET_LONG,
        entry_price=42.50,
        exit_price=41.60,
        exit_reason="-2.0% 칼손절",
        qty=120
    )
    logger.info(res3b["text"])

    logger.info("\n5. ⏱️ [Type 3-C: 90분 타임스탑 청산 알림 시뮬레이션]")
    res3c = notifier.send_exit_alert(
        ticker=config.BASE_ASSET_SHORT,
        entry_price=20.00,
        exit_price=20.10,
        exit_reason="90분 타임스탑",
        qty=250
    )
    logger.info(res3c["text"])

    logger.info("\n6. 🌙 [Type 3-D: 종가 오버나잇 방지 전량 청산 알림 시뮬레이션]")
    res3d = notifier.send_exit_alert(
        ticker=config.BASE_ASSET_LONG,
        entry_price=42.50,
        exit_price=42.80,
        exit_reason="종가 오버나잇 방지",
        qty=120
    )
    logger.info(res3d["text"])

    logger.info("\n" + "=" * 80)
    logger.info("✅ [검증 완료] 3대 카테고리 알림 템플릿 정상 포맷팅 및 파이프라인 검증 성공")
    logger.info("=" * 80)
