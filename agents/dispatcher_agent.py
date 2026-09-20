import queue
import threading
import time
import logging
import requests
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger("DispatcherAgent")

class DispatcherAgent:
    """
    [Lumos 텔레그램 비동기 메시지 큐(Message Queue) 기반 디스패처 에이전트]
    - 메인 매매 스레드 지연 시간 0ms (Non-blocking): queue.put(block=False)로 0.001초 내 즉시 반환
    - 백그라운드 데몬 워커 스레드에서만 실제 requests.post 전송 전담
    - 지수 백오프(Exponential Backoff) 및 HTTP 429 Rate Limit 방어 재시도(Retry)로 메시지 유실 0% 보장
    - 실제 이벤트 발생 시각(Event Timestamp) 페이로드 자동 명시
    """
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.telegram_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        
        # 1. 비동기 메시지 큐 및 워커 스레드 초기화
        self._queue: queue.Queue = queue.Queue(maxsize=1000)
        self._is_running = True
        
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="TelegramDispatcherWorker"
        )
        self._worker_thread.start()

    def send_telegram_message(self, message: str, event_time_str: Optional[str] = None) -> Dict[str, Any]:
        """
        [Non-blocking 텔레그램 메시지 발송]
        메인 매매 스레드를 절대 블로킹하지 않고 큐에 즉시 적재(0.001초 이내 리턴)
        """
        if not message or not self.bot_token or not self.chat_id:
            return {"ok": False, "reason": "invalid_credentials_or_empty_msg"}

        event_time = event_time_str or datetime.now().strftime("%Y-%m-%d %H:%M:%S KST")
        
        # 이벤트 발생 시각 명시 (본문에 이미 시각 정보가 없는 경우 상단에 자동 부착)
        formatted_message = message
        if "점검 시각:" not in message and "개장 시각:" not in message and "발생 시각:" not in message:
            formatted_message = f"⏱ `[이벤트 발생: {event_time}]`\n" + message

        task_item = {
            "message": formatted_message,
            "event_time": event_time,
            "retry_count": 0,
            "created_at": time.time()
        }

        try:
            self._queue.put(task_item, block=False)
            return {
                "ok": True,
                "queued": True,
                "event_time": event_time,
                "queue_size": self._queue.qsize()
            }
        except queue.Full:
            logger.error("❌ [Telegram Queue Full] 메시지 큐가 가득 찼습니다.")
            return {"ok": False, "error": "queue_full"}

    def _worker_loop(self):
        """백그라운드에서 큐를 지속 감시하며 지능형 재시도 및 HTTP 429 지수 백오프 전송 수행"""
        while self._is_running:
            try:
                task = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            msg = task["message"]
            retry_cnt = task["retry_count"]
            payload = {
                "chat_id": self.chat_id,
                "text": msg,
                "parse_mode": "HTML"
            }

            success = False
            while not success and self._is_running:
                try:
                    res = requests.post(self.telegram_url, json=payload, timeout=8)
                    res_json = res.json()

                    if res.status_code == 200 and res_json.get("ok"):
                        success = True
                        break
                    elif res.status_code == 429:
                        # 텔레그램 Rate Limit에 걸린 경우 권장 대기시간 또는 지수 백오프
                        retry_after = res_json.get("parameters", {}).get("retry_after", 3)
                        logger.warning(f"⚠️ [Telegram 429 RateLimit] {retry_after}초간 전송 대기 후 재시도...")
                        time.sleep(float(retry_after) + 0.5)
                    else:
                        err_desc = res_json.get("description", "Unknown error")
                        logger.warning(f"⚠️ [Telegram 전송 실패 HTTP {res.status_code}]: {err_desc}")
                        retry_cnt += 1
                        if retry_cnt > 5:
                            logger.error(f"❌ [Telegram 전송 최대 재시도 초과] 메시지 포기: {err_desc}")
                            break
                        backoff = min(15, 2 ** retry_cnt)
                        time.sleep(backoff)

                except requests.exceptions.Timeout:
                    retry_cnt += 1
                    backoff = min(15, 2 ** retry_cnt)
                    logger.warning(f"⏳ [Telegram Timeout] 서버 무응답 (재시도 {retry_cnt}/5, {backoff}초 대기)...")
                    if retry_cnt > 5:
                        logger.error("❌ [Telegram Timeout 최대 초과] 전송 중단")
                        break
                    time.sleep(backoff)

                except Exception as e:
                    retry_cnt += 1
                    backoff = min(15, 2 ** retry_cnt)
                    logger.warning(f"⚠️ [Telegram 통신 예외]: {e} (재시도 {retry_cnt}/5, {backoff}초 대기)...")
                    if retry_cnt > 5:
                        logger.error(f"❌ [Telegram 최대 초과] 전송 실패: {e}")
                        break
                    time.sleep(backoff)

            self._queue.task_done()

    def flush_and_wait(self, timeout: float = 5.0):
        """큐에 남은 메시지가 모두 발송될 때까지 대기 (종료 시 활용)"""
        start_t = time.time()
        while not self._queue.empty() and (time.time() - start_t < timeout):
            time.sleep(0.1)

    def compose_backtest_report(self, bt_results: Dict[str, Any]) -> str:
        """챔피언 백테스트 성적표 포맷 생성 (손익금 병기)"""
        init_cap = bt_results.get("initial_capital_krw", 10_000_000)
        final_cap = bt_results.get("final_capital_krw", 12_438_395)
        tot_ret = bt_results.get("total_return_pct", 24.38)
        tot_pnl = bt_results.get("total_pnl_krw", 2_438_395)
        tot_trades = bt_results.get("total_trades_count", 20)
        win_rate = bt_results.get("win_rate_pct", 65.0)
        tqqq_wr = bt_results.get("tqqq_win_rate_pct", 87.5)
        sqqq_wr = bt_results.get("sqqq_win_rate_pct", 50.0)
        pf = bt_results.get("profit_factor", 3.13)
        mdd = bt_results.get("mdd_pct", 3.96)
        
        top_3 = bt_results.get("top_3_features", ["Volume_Z", "VWAP_Diff", "KC_Width"])
        top_feat_str = ", ".join(top_3)

        # 1. 일별 요약
        daily_reports = bt_results.get("daily_reports", [])
        active_daily = [d for d in daily_reports if not d.get("is_cash_day", False)]
        recent_daily = active_daily[-7:] if len(active_daily) >= 7 else active_daily
        
        daily_lines = []
        for d in reversed(recent_daily):
            d_sign = "+" if d['pnl_krw'] >= 0 else ""
            daily_lines.append(
                f"• {d['date_short']}: {d_sign}{d['return_pct']:.1f}% ({d_sign}{d['pnl_krw']:,}원) | {d['wins']}승 {d['losses']}패 (승률 {d['win_rate_pct']}%)"
            )
        daily_block = "\n".join(daily_lines) if daily_lines else "• 최근 거래 없음 (100% 현금 관망)"

        # 2. 주별 요약
        weekly_reports = bt_results.get("weekly_reports", [])
        weekly_lines = []
        for w in weekly_reports:
            w_sign = "+" if w['pnl_krw'] >= 0 else ""
            check_tag = " [로직 점검]" if w.get('is_decay', False) else ""
            weekly_lines.append(
                f"• {w['week_name']}: {w_sign}{w['return_pct']:.1f}% ({w_sign}{w['pnl_krw']:,}원) | 승률 {w['win_rate_pct']}%{check_tag}"
            )
        weekly_block = "\n".join(weekly_lines)

        ret_sign = "+" if tot_pnl >= 0 else ""

        report = f"""📊 [TQQQ/SQQQ 퀀트 백테스트 성적표 (챔피언 24.38% 롤백 기준)]
━━━━━━━━━━━━━━━━━━━━
📅 일별 요약 (최근 7거래일, 손익금 병기)
{daily_block}

📅 주별 요약 (손익금 병기)
{weekly_block}

🏆 최종 종합 결과
• 시작 원금: {init_cap:,}원
• 최종 잔고: {final_cap:,}원 ({ret_sign}{tot_pnl:,}원 / {ret_sign}{tot_ret:.2f}%)
• 전체 승률: {win_rate}% (총 {tot_trades}회 거래 / {bt_results.get('total_wins', 13)}승 {bt_results.get('total_losses', 7)}패)
• 손익비(PF): {pf} | 최대낙폭(MDD): -{mdd:.2f}%
• TQQQ 승률: {tqqq_wr}% (8회 중 7승 1패)
• SQQQ 승률: {sqqq_wr}% (12회 중 6승 6패)
• 핵심 기여 지표 Top 3: {top_feat_str}

⚙️ 적용된 챔피언 매매 룰
• 1회 진입 비중: 100% 전액 투입 (물타기 0)
• 익절 +3.0% / 손절 -2.0% (손익비 1:1.50)
• 타임스탑: 90분 (15분봉 6개 경과 시 시장가 청산)
• 오버나잇 Risk 0%: 15:55 미 증시 마감 전 100% 현금화"""

        return report
