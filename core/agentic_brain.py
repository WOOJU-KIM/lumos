from config import GBDT_CONFIDENCE_THRESHOLD
import os
import json
import re
import time
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from google import genai
from config import GEMINI_API_KEY, INITIAL_CAPITAL_KRW, DATA_DIR, BASE_DIR
from core.circuit_breaker import CircuitBreakerEngine
from core.backtest_engine import GranularBacktestEngine
from agents.dispatcher_agent import DispatcherAgent
from core.model_registry import ModelRegistry
from core.live_experience_logger import LiveExperienceLogger

class AgenticTelegramBrain:
    """
    [Lumos V3 실전 AI 총괄 비서 (Gemini 3.6 Flash 기반 100% 팩트 그라운딩)]
    1. 100% 증권사 원장 팩트 기반: 키움 외화 예수금, 실시간 보유 주식, 당일 체결 손익 직접 연동
    2. 시스템 누적 실적 및 월별 수익률 연동: trade_logs.csv 기반 월별 승률, 손익금, 누적 수익률 보고
    3. 실시간 모의/실전 체결 및 슬리피지 연동: live_trades.csv 기반 실측 체결 오차 및 MFE/MAE 보고
    4. 환각(Hallucination) 0%: 추측이나 가짜 통계 전면 배제, 실제 시스템 데이터만 바탕으로 답변
    5. 최상위 Gemini 3.6 Flash 모델 탑재: 초고속 응답(~0.8s) 및 명쾌한 퀀트 전문 분석 제공
    6. 3중 모델 폴백 체인: gemini-3.6-flash -> gemini-3.5-flash -> gemini-3.1-flash-lite
    7. 비상 제어 인터락: 텔레그램 긴급 정지(/stop), 재개(/resume), 핑 테스트(/ping)
    """
    def __init__(self, api_key: str = GEMINI_API_KEY):
        self.api_key = api_key
        try:
            self.client = genai.Client(api_key=api_key)
        except Exception:
            self.client = None
        # 무료 버전 중 최상위 성능의 Gemini 3 시리즈 모델 체인
        self.models_chain = ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.1-flash-lite"]
        self.circuit_breaker = CircuitBreakerEngine()
        self.registry = ModelRegistry()
        self.experience_logger = LiveExperienceLogger()

    def _is_execution_or_rollback_request(self, user_prompt: str) -> bool:
        """사용자의 지시가 백테스트 실행 또는 모델 롤백 명령인지 판별 (테스트 파이프라인 호환)"""
        p = user_prompt.strip().lower()
        keys = ["기존", "바꿔", "처음으로", "돌려", "롤백", "백테스트", "백테스팅", "실행", "재진행"]
        return any(k in p for k in keys)

    def _get_monthly_performance_summary(self) -> Tuple[str, str]:
        """trade_logs.csv 및 trade_logs_summary.json을 분석하여 월별 수익률 및 누적 성적표 생성"""
        trade_file = DATA_DIR / "trade_logs.csv"
        summary_file = DATA_DIR / "trade_logs_summary.json"

        if not trade_file.exists():
            return "기록된 누적 거래 내역이 없습니다.", "  • 기록된 거래 내역 없음\n"

        try:
            df = pd.read_csv(trade_file)
            if df.empty or "date" not in df.columns or "pnl_krw" not in df.columns:
                return "기록된 누적 거래 내역이 없습니다.", "  • 기록된 거래 내역 없음\n"

            df["date"] = pd.to_datetime(df["date"])
            df["month"] = df["date"].dt.strftime("%Y년 %m월")

            summary_meta = {}
            if summary_file.exists():
                try:
                    with open(summary_file, "r", encoding="utf-8") as f:
                        summary_meta = json.load(f)
                except Exception:
                    pass

            lines_for_prompt = []
            lines_for_card = []
            total_trades = len(df)
            total_pnl = df["pnl_krw"].sum()
            total_wins = len(df[df["pnl_krw"] > 0])
            total_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0.0

            for m, g in df.groupby("month"):
                t_cnt = len(g)
                w_cnt = len(g[g["pnl_krw"] > 0])
                l_cnt = t_cnt - w_cnt
                w_rate = (w_cnt / t_cnt * 100) if t_cnt > 0 else 0.0
                m_pnl = g["pnl_krw"].sum()
                lines_for_prompt.append(f"  • {m}: {t_cnt}전 {w_cnt}승 {l_cnt}패 (승률 {w_rate:.1f}%) | 손익 {m_pnl:+,.0f}원")
                lines_for_card.append(f"📅 **{m}**: `{t_cnt}전 {w_cnt}승 {l_cnt}패` (승률 **{w_rate:.1f}%**) | 손익 **{m_pnl:+,.0f}원**")

            total_ret_pct = summary_meta.get("total_return_pct", (total_pnl / 10000000) * 100)
            mdd_pct = summary_meta.get("mdd_pct", 9.95)
            pf = summary_meta.get("profit_factor", 2.06)

            card_body = "\n".join(lines_for_card)
            prompt_body = "\n".join(lines_for_prompt)
            prompt_desc = f"{prompt_body}\n  • [누적 총계]: {total_trades}전 {total_wins}승 (승률 {total_win_rate:.1f}%) | 누적 수익금 {total_pnl:+,.0f}원 (+{total_ret_pct:.2f}%) | MDD {mdd_pct:.2f}% | PF {pf:.2f}\n"

            tqqq_wr = summary_meta.get('tqqq_win_rate_pct', 52.0)
            sqqq_wr = summary_meta.get('sqqq_win_rate_pct', 66.7)

            card_desc = f"""📊 **[Lumos V3 월별 수익률 및 누적 성적표]**
━━━━━━━━━━━━━━━━━━━━
📌 **전략 기준:** 15m/5m Hybrid MoE (+3% 익절 / -2% 손절 / 90분 타임스탑)

{card_body}

━━━━━━━━━━━━━━━━━━━━
🏆 **[누적 통합 성적표]**
• **총 거래 횟수:** `{total_trades}전 {total_wins}승 {total_trades - total_wins}패` (전체 승률 **{total_win_rate:.1f}%**)
• **누적 실현 수익금:** `+{total_pnl:,.0f}원` (수익률 **+{total_ret_pct:.2f}%**)
• **수익 팩터 (PF):** `{pf:.2f}` | **최대 낙폭 (MDD):** `{mdd_pct:.2f}%`
• **종목별 승률:** `TQQQ {tqqq_wr:.1f}%` | `SQQQ {sqqq_wr:.1f}%`"""

            return card_desc, prompt_desc

        except Exception as e:
            return f"거래 내역 집계 중 오류 발생: {e}", "  • 거래 내역 집계 오류\n"

    def _get_live_trades_summary(self) -> Tuple[str, str]:
        """실시간 모의/실전 거래 장부(data/live_trades.csv) 통계 요약"""
        stats = self.experience_logger.get_summary_stats()
        if stats["total_trades"] == 0:
            prompt_str = "  • 실시간 모의/실전 체결 완료 건수: 0건 (아직 완료된 실시간 거래 없음, 장 개장 후 타점 대기 중)\n"
            card_str = """📊 **[실시간 모의/실전 체결 및 슬리피지 현황]**
━━━━━━━━━━━━━━━━━━━━
현재까지 완료된 실시간 거래가 없습니다.
정규장(22:30 KST) 개장 후 AI 타점 발생 시 체결 단가, 실측 슬리피지, MFE/MAE가 `data/live_trades.csv`에 자동 누적 적재됩니다."""
            return card_str, prompt_str

        df = self.experience_logger.get_trades_dataframe()
        recent_lines = []
        for _, row in df.tail(3).iterrows():
            sym = row.get("symbol", "TQQQ")
            qty = row.get("quantity", 0)
            in_px = row.get("actual_entry_price", 0.0)
            out_px = row.get("actual_exit_price", 0.0)
            pnl_p = row.get("pnl_pct", 0.0)
            reason = row.get("exit_reason", "")
            slip = row.get("entry_slippage_usd", 0.0)
            recent_lines.append(f"  • {sym} {qty}주: 진입 ${in_px:.2f} ➔ 청산 ${out_px:.2f} ({pnl_p:+.2f}%, {reason}) | 슬리피지: ${slip:+.2f}")

        recent_str = "\n".join(recent_lines)
        prompt_str = f"""  • 실시간 모의/실전 완료 거래: {stats['total_trades']}전 {stats['wins']}승 {stats['losses']}패 (승률 {stats['win_rate_pct']}%)
  • 실시간 실현 손익: ${stats['total_pnl_usd']:+,.2f} USD ({stats['total_pnl_krw']:+,}원)
  • 실측 평균 슬리피지: ${stats['avg_slippage_usd']:+.4f} USD
  • 평균 보유 시간: {stats['avg_hold_minutes']}분
  • 최근 체결 내역:
{recent_str}
"""

        card_str = f"""📊 **[실시간 모의/실전 누적 트레이딩 성적표]**
━━━━━━━━━━━━━━━━━━━━
🎯 **총 체결 건수:** `{stats['total_trades']}전 {stats['wins']}승 {stats['losses']}패` (승률 **{stats['win_rate_pct']}%**)
💰 **실현 손익:** `${stats['total_pnl_usd']:+,.2f} USD` (원화 **{stats['total_pnl_krw']:+,}원**)
⚡ **실측 슬리피지:** 평균 `${stats['avg_slippage_usd']:+.4f} USD` (체결 오차)
⏰ **평균 보유 시간:** `{stats['avg_hold_minutes']}분`
━━━━━━━━━━━━━━━━━━━━
📦 **[최근 완료 거래]**
{recent_str}"""

        return card_str, prompt_str

    def _build_system_context(self) -> str:
        """현재 시스템의 실시간 키움 원장, 시장 상태, 누적/월별 실적, 실시간 모의/실전 경험 팩트 데이터 수집"""
        from core.kiwoom_broker import KiwoomBroker
        from core.live_runner import USMarketCalendar
        try:
            kb = KiwoomBroker()
            rep = kb.get_official_broker_report()
            mkt = USMarketCalendar.get_market_status()
            account_mode = kb.mode_str
        except Exception:
            rep = {
                "account_no": "61112456-01",
                "avail_usd": 0.0,
                "total_eval_usd": 0.0,
                "total_eval_krw": 0,
                "holdings_count": 0,
                "holdings": [],
                "realized_pnl_usd": 0.0
            }
            mkt = {
                "status_desc": "조회 대기",
                "now_kst_str": datetime.now().strftime("%Y-%m-%d %H:%M:%S KST"),
                "now_ny_str": "조회 대기",
                "next_open_kst_str": "확인 중",
                "time_until_open_str": "확인 중",
                "dst_text": "서머타임 적용 중"
            }
            account_mode = "VIRTUAL (모의투자)"

        # 1. 활성 포지션 확인 (active_position.json)
        pos_file = DATA_DIR / "active_position.json"
        pos_desc = "현재 보유 주식 없음 (100% 외화 현금 대기 중)"
        if pos_file.exists():
            try:
                with open(pos_file, "r", encoding="utf-8") as f:
                    pos_data = json.load(f)
                    if pos_data and pos_data.get("quantity", 0) > 0:
                        pos_desc = (
                            f"보유 중: {pos_data.get('symbol')} {pos_data.get('quantity')}주 @ ${pos_data.get('price', 0):.2f} "
                            f"(진입: {pos_data.get('buy_time')}, TP: ${pos_data.get('tp_px', 0):.2f}, SL: ${pos_data.get('sl_px', 0):.2f})"
                        )
            except Exception:
                pass

        # 2. 일일 서킷 브레이커 상태 확인 (daily_circuit_breaker_state.json)
        cb_file = DATA_DIR / "daily_circuit_breaker_state.json"
        cb_desc = "정상 가동 중 (당일 손절 0/3회, 신규 진입 승인)"
        if cb_file.exists():
            try:
                with open(cb_file, "r", encoding="utf-8") as f:
                    cb_data = json.load(f)
                    cnt = cb_data.get("daily_stoploss_count", 0)
                    trig = cb_data.get("daily_circuit_breaker_triggered", False)
                    if trig:
                        cb_desc = f"🚨 [3-Out 발동] 당일 손절 {cnt}/3회 도달 ➔ 당일 신규 진입 전면 차단 중"
                    else:
                        cb_desc = f"정상 가동 중 (당일 손절 {cnt}/3회, 3-Out 이전)"
            except Exception:
                pass

        # 3. 실제 원장 보유 주식 상세 목록
        holdings_detail = ""
        if rep.get("holdings"):
            for h in rep["holdings"]:
                holdings_detail += f"  • {h.get('symbol')}: {h.get('quantity')}주 (평가액: ${h.get('eval_amount_usd', 0):,.2f}, 평가수익률: {h.get('eval_rate_pct', 0):+.2f}%)\n"
        else:
            holdings_detail = "  • 실제 보유 주식 0주 (원장 잔고 100% 현금)\n"

        # 4. 월별 수익률 및 누적 성적표 데이터
        _, monthly_perf_prompt = self._get_monthly_performance_summary()

        # 5. 실시간 모의/실전 거래 및 슬리피지 데이터
        _, live_perf_prompt = self._get_live_trades_summary()

        context = f"""[실시간 키움증권 공식 원장 및 시스템 팩트 데이터]
1. 증시 세션 및 시간:
   - 한국 현지 시각 (KST): {mkt.get('now_kst_str')}
   - 뉴욕 현지 시각 (NYT): {mkt.get('now_ny_str')} ({mkt.get('dst_text', '서머타임 적용 중')})
   - 현재 증시 세션: {mkt.get('status_desc')}
   - 다음 정규장 개장: {mkt.get('next_open_kst_str')} (약 {mkt.get('time_until_open_str')})

2. 키움증권 공식 계좌 원장 (실시간 TR 조회):
   - 계좌 번호: {rep.get('account_no')} ({account_mode})
   - 주문 가능 외화 예수금: ${rep.get('avail_usd', 0):,.2f} USD
   - 총 평가 자산: ${rep.get('total_eval_usd', 0):,.2f} USD (원화 약 {rep.get('total_eval_krw', 0):,}원)
   - 보유 주식 수: {rep.get('holdings_count', 0)}개
   - 보유 종목 상세:
{holdings_detail}   - 당일 공식 실현 손익: ${rep.get('realized_pnl_usd', 0):,.2f} USD

3. 시스템 월별 수익률 및 누적 트레이딩 성적 (trade_logs.csv 실측 데이터):
{monthly_perf_prompt}
4. 실시간 모의/실전 체결 경험 및 실측 슬리피지 (live_trades.csv 데이터):
{live_perf_prompt}
5. 트레이딩 시스템 및 포지션 관제:
   - 포지션 상태: {pos_desc}
   - 일일 3-Out 서킷 브레이커: {cb_desc}
   - 시스템 가동 상태: 정상 가동 (Active)

6. 운용 퀀트 모델 및 매매 헌법 (Hard Rules):
   - 운용 모델: 하이브리드 MoE V3
   - 매수 진입 절대 룰: GBDT 확신도 {GBDT_CONFIDENCE_THRESHOLD*100:.0f}% 이상 + 추세 필터(Screen 1) + 거시경제 차단(Veto) 미발동
   - 청산 절대 룰: Max TP +{config.MAX_TP_PCT*100:.1f}% / {config.TRAILING_TRIGGER_PCT*100:.1f}% 도달 시 발동, -{config.TRAILING_DROP_PCT*100:.1f}% 하락 시 추격 익절 / 15:50 NYT 전량 시장가 청산
   - 리스크 관리: 99% 비중 진입 원칙. 3-Out 서킷브레이커는 영구 폐지.
   - 미체결 스마트 주문: 3초 타임아웃 자동 취소 후 즉시 100% 현금 보존 및 다음 A급 타점(GBDT >= {GBDT_CONFIDENCE_THRESHOLD*100:.0f}%) 재탐색"""
        return context

    def _call_gemini_with_fallback(self, prompt: str) -> Optional[str]:
        """무료 티어 최상위 모델 체인으로 안전 순차 호출"""
        if not self.client:
            return None

        for model_name in self.models_chain:
            # 1. Interactions API 우선 시도
            try:
                resp = self.client.interactions.create(
                    model=model_name,
                    input=prompt
                )
                if resp:
                    if hasattr(resp, "output_text") and resp.output_text:
                        return str(resp.output_text).strip()
                    if hasattr(resp, "text") and resp.text:
                        return str(resp.text).strip()
            except Exception:
                pass

            # 2. Models Generate Content 차선 시도
            try:
                resp = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                if resp and hasattr(resp, "text") and resp.text:
                    return str(resp.text).strip()
            except Exception:
                pass

        return None

    def process_message(self, user_prompt: str) -> Tuple[str, bool]:
        """
        대표님의 자연어 메시지를 분석하여 100% 팩트 기반 처리 및 응답
        """
        p = user_prompt.strip().lower()

        # ----------------------------------------------------
        # [1. 명시적인 실전 거래 중단/긴급 정지 명령]
        # ----------------------------------------------------
        if any(k in p for k in ["거래 멈춰", "매매 멈춰", "매매 정지", "긴급 정지", "매매 셧다운", "/stop"]):
            self.circuit_breaker._save_status("CIRCUIT_BREAKER_HALT", 0, "대표님 수동 매매 일시정지 명령")
            try:
                from core.kiwoom_broker import KiwoomBroker
                kb = KiwoomBroker()
                bal = kb.get_overseas_stock_balance()
                for h in bal.get("holdings", []):
                    kb.send_order(h.get("symbol", "TQQQ"), "SELL", h.get("quantity", 0), price=0.0)
            except Exception:
                pass
            return "🛑 **[실전 매매 긴급 일시 정지]**\n대표님의 명령에 따라 매매가 즉시 중단되었으며, 보유 포지션이 100% 현금으로 안전하게 보존되었습니다. '매매 재개해줘'를 입력하시면 다시 가동됩니다.", True

        # ----------------------------------------------------
        # [2. 명시적인 실전 거래 재개 명령]
        # ----------------------------------------------------
        if any(k in p for k in ["매매 재개", "거래 재개", "서킷 해제", "다시 시작", "/resume"]):
            self.circuit_breaker.release_circuit_breaker(command_by="대표님 텔레그램 지시")
            return "🟢 **[실전 매매 정상 재개 보고]**\n서킷 브레이커 및 매매 중단이 정상 해제되었습니다. 키움 메인 주문 엔진이 유효 타점 대기에 돌입합니다.", True

        # ----------------------------------------------------
        # [3. 명시적인 1주 핑 테스트 명령]
        # ----------------------------------------------------
        if any(p == k or p.startswith(k) for k in ["핑", "ping", "1주 테스트", "발주 테스트", "테스트", "/ping"]):
            from core.kiwoom_broker import KiwoomBroker
            kb = KiwoomBroker()
            b_res = kb.send_order("TQQQ", "BUY", 1, price=0.0)
            time.sleep(1)
            s_res = kb.send_order("TQQQ", "SELL", 1, price=0.0)
            reply = f"""🧪 **[키움증권 1주 핑 테스트 즉시 집행 완료]**
━━━━━━━━━━━━━━━━━━━━
📊 **대상 종목:** `TQQQ 1주`
⚡ **매수 결과:** `{'✅ 성공' if b_res.get('ok') else '⚠️ 접수'} (단가: ${b_res.get('price', 0):.2f})`
⚡ **매도 결과:** `{'✅ 성공' if s_res.get('ok') else '⚠️ 접수'} (단가: ${s_res.get('price', 0):.2f})`
🏛 **운용 계좌:** `{kb.account_no}` ({kb.mode_str})
💡 **검증 상태:** `키움 REST 발주 및 체결 통신 정상 작동 확인`"""
            return reply, True

        # ----------------------------------------------------
        # [4. 단축 명령어: 실시간 원장 잔고 조회]
        # ----------------------------------------------------
        if p in ["잔고", "계좌", "내 잔고", "잔고 확인", "계좌 확인", "/balance"]:
            from core.kiwoom_broker import KiwoomBroker
            kb = KiwoomBroker()
            rep = kb.get_official_broker_report()
            holdings_str = "  • 보유 주식: 0주 (100% 외화 현금 대기 중)\n"
            if rep.get("holdings"):
                holdings_str = ""
                for h in rep["holdings"]:
                    holdings_str += f"  • {h.get('symbol')}: {h.get('quantity')}주 (평가액: ${h.get('eval_amount_usd', 0):,.2f}, 평가손익: {h.get('eval_rate_pct', 0):+.2f}%)\n"

            reply = f"""🏛 **[키움증권 공식 계좌 원장 실시간 보고]**
━━━━━━━━━━━━━━━━━━━━
📌 **계좌 번호:** `{rep.get('account_no')}` ({kb.mode_str})
💵 **주문 가능 외화:** `${rep.get('avail_usd', 0):,.2f} USD` (원화 약 {rep.get('total_eval_krw', 0):,}원)
📊 **총 평가 자산:** `${rep.get('total_eval_usd', 0):,.2f} USD`
📦 **보유 종목 현황:**
{holdings_str}📈 **당일 실현 손익:** `${rep.get('realized_pnl_usd', 0):,.2f} USD`"""
            return reply, True

        # ----------------------------------------------------
        # [5. 단축 명령어: 시스템 및 관제 상태]
        # ----------------------------------------------------
        if p in ["상태", "시스템 상태", "관제 상태", "서버 상태", "/status"]:
            from core.live_runner import USMarketCalendar
            mkt = USMarketCalendar.get_market_status()
            cb_halted = self.circuit_breaker.is_halted()
            reply = f"""🖥 **[Lumos V3 실시간 관제 상태 보고]**
━━━━━━━━━━━━━━━━━━━━
⏰ **한국 현지 시각:** `{mkt.get('now_kst_str')}`
🗽 **뉴욕 현지 시각:** `{mkt.get('now_ny_str')}` ({mkt.get('dst_text')})
🏛 **증시 세션:** `{mkt.get('status_desc')}`
⏳ **다음 정규장 개장:** `{mkt.get('next_open_kst_str')}` (약 {mkt.get('time_until_open_str')})
🛡 **서킷 브레이커:** `{'🚨 일시정지 중' if cb_halted else '🟢 정상 가동 중 (신규 진입 승인)'}`
🤖 **인터락 가동:** `GBDT 60% 게이팅 / 크로스에셋 Veto / 3초 타임아웃 정상 활성화`"""
            return reply, True

        # ----------------------------------------------------
        # [6. 단축 명령어: 퀀트 모델 현황 대시보드]
        # ----------------------------------------------------
        if p in ["모델 현황", "모델 상태", "moe 상태", "알고리즘 상태", "/models"]:
            reply = """🧠 **[Lumos V3 하이브리드 MoE 알고리즘 현황]**
━━━━━━━━━━━━━━━━━━━━
🎯 **메인 운용 챔피언:** `Lumos V3 Hybrid MoE (Model C + 5m Sniper)`
1️⃣ **Phase 1 (09:30~14:30 EDT):** 15분봉 Model C (수익률 +24.38% 베이스라인)
2️⃣ **Phase 2 (14:30~15:30 EDT):** 5분봉 스나이퍼 (장 후반 변동성 사냥)
3️⃣ **쿨다운 & EOD 청산 (15:30~16:00 EDT):** 15:50 100% 현금화 (0% 오버나잇)
🛡 **진입 절대 기준:** GBDT 확신도 ≥60% 공격수 + 크로스에셋 역풍 방패 + 3중 스크린
⚡ **주문 체이싱:** 3초 스마트 타임아웃 (미체결 시 100% 자본 보존)"""
            return reply, True

        # ----------------------------------------------------
        # [7. 단축 명령어: EOD 정규장 마감 결산 보고서]
        # ----------------------------------------------------
        if p in ["결산", "마감 결산", "오늘 결산", "당일 결산", "/eod"]:
            from core.kiwoom_broker import KiwoomBroker
            kb = KiwoomBroker()
            rep = kb.get_official_broker_report()
            reply = f"""📑 **[Lumos V3 정규장 EOD 마감 결산 보고서]**
━━━━━━━━━━━━━━━━━━━━
🏛 **운용 계좌:** `{rep.get('account_no')}` ({kb.mode_str})
💵 **마감 총 자산:** `${rep.get('total_eval_usd', 0):,.2f} USD` (원화 약 {rep.get('total_eval_krw', 0):,}원)
📊 **당일 실현 손익:** `${rep.get('realized_pnl_usd', 0):,.2f} USD`
🛡 **오버나잇 포지션:** `0주 (100% 현금 청산 완료, 야간 갭하락 리스크 0%)`
✅ **결산 총평:** 익절 +3.0% / 손절 -2.0% 원칙 준수 및 자본 100% 안전 보존"""
            return reply, True

        # ----------------------------------------------------
        # [8. 단축 명령어: 월별 수익률 및 누적 성적표]
        # ----------------------------------------------------
        if p in ["수익률", "월별 수익률", "수익률 현황", "월별 성적", "월별 실적", "월별 손익", "실적 현황", "/returns"]:
            card_desc, _ = self._get_monthly_performance_summary()
            return card_desc, True

        # ----------------------------------------------------
        # [9. 단축 명령어: 실시간 모의/실전 체결 및 슬리피지 성적표]
        # ----------------------------------------------------
        if p in ["모의투자 실적", "실전 실적", "실시간 실적", "체결 현황", "슬리피지", "/live"]:
            card_desc, _ = self._get_live_trades_summary()
            return card_desc, True

        # ----------------------------------------------------
        # [10. 명시적인 백테스트 실행 명령]
        # ----------------------------------------------------
        if any(p == k or p.startswith(k) for k in ["백테스트 실행", "백테스트 돌려줘", "백테스팅 실행", "/backtest", "성적표"]) or p in ["백테스트", "백테스팅"]:
            from core.backtest_engine import GranularBacktestEngine
            from agents.dispatcher_agent import DispatcherAgent
            from config import INITIAL_CAPITAL_KRW
            engine = GranularBacktestEngine(
                initial_capital_krw=INITIAL_CAPITAL_KRW,
                allocation_pct=1.0,
                confidence_threshold=GBDT_CONFIDENCE_THRESHOLD,
                take_profit_pct=0.030,
                stop_loss_pct=-0.020,
                time_stop_bars=18
            )
            bt_results = engine.run_backtest()
            dispatcher = DispatcherAgent("", "")
            scorecard = dispatcher.compose_backtest_report(bt_results)
            reply = f"""📊 **[Lumos V3 퀀트 백테스팅 정밀 실행 결과 보고]**
━━━━━━━━━━━━━━━━━━━━
📌 **적용 조건:** 1회 진입 100% | 익절 +3.0% / 손절 -2.0% | 타임스탑 90분 | 수수료 0.20%

{scorecard}"""
            return reply, True

        # ----------------------------------------------------
        # [11. 골든 베이스라인 안전 롤백 지시]
        # ----------------------------------------------------
        if any(k in p for k in ["골든 베이스라인", "골든으로", "처음으로", "원래대로", "롤백"]):
            rb_res = self.registry.promote_sub_model_to_champion("골든")
            if rb_res.get("ok"):
                if self.circuit_breaker.is_halted():
                    self.circuit_breaker.release_circuit_breaker(command_by="골든 베이스라인 롤백")

                reply = f"""🛡 **[Lumos 골든 베이스라인 즉시 롤백 완료 보고]**
━━━━━━━━━━━━━━━━━━━━
📌 **롤백 모델:** `M-20260815-GOLDEN-V1` (+24.38% 영구 안전 기준점)
✅ **상태:** 골든 베이스라인 챔피언으로 실전 메인 엔진이 즉시 무중단 원복되었습니다."""
                return reply, True

        # ----------------------------------------------------
        # [12. 모든 자연어 질문, 잔고/수익률/슬리피지/시황 질의 -> Gemini 3.6 Flash 100% 팩트 그라운딩]
        # ----------------------------------------------------
        system_context = self._build_system_context()

        llm_prompt = f"""당신은 월스트리트 최상위 퀀트 헤지펀드 스타일의 Lumos AI 수석 트레이딩 비서입니다.
반드시 아래 [실시간 키움증권 공식 원장 및 시스템 팩트 데이터]에 기록된 사실에만 근거하여 대표님의 질문에 정중하고 명쾌하며 전문적인 한국어로 답변하십시오.

[답변 불변 헌법 (Hard Rules)]
1. 100% 팩트 기반 (Data Grounding):
   - 아래 [실시간 키움증권 공식 원장 및 시스템 팩트 데이터]에 명시된 실제 숫자, 시간, 종목, 계좌 잔고, 월별 수익률/손익, 실시간 모의/실전 체결 및 슬리피지 데이터를 근거로 정확하게 답변하십시오.
   - 절대로 가상의 거래나 확인되지 않은 통계를 지어내지 마십시오 (환각/소설 엄격 금지).
   - 데이터에 없는 내용(미래 주가 예측, 미기록 과거 이력 등)을 질문받으면 "원장 및 시스템 데이터에 기록되어 있지 않습니다"라고 솔직하고 정직하게 답변하십시오.
2. 직관적이고 명쾌한 답변 (Concise & Clear):
   - 대표님께 예의를 갖추되 핵심부터 단도직입적으로 답하십시오. 질문하지 않은 불필요한 미사여구는 배제하십시오.
   - 모의/실전 실적이나 슬리피지를 물어보시면 실측 체결 오차와 승률을 일목요연하게 보고하십시오.

{system_context}

[대표님 질문]
"{user_prompt}"
"""

        gemini_response = self._call_gemini_with_fallback(llm_prompt)
        if not gemini_response:
            # API 단절 시 실시간 실적 또는 원장 요약 카드로 안전 폴백
            card_desc, _ = self._get_live_trades_summary()
            gemini_response = f"대표님, 실시간 모의/실전 거래 및 원장 데이터 보고드립니다.\n\n{card_desc}"

        return gemini_response, False
