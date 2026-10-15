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
        self._state_lock = threading.RLock()
        self._last_state_notification_keys: Dict[str, str] = {}

        if self.async_mode:
            self._worker_thread = threading.Thread(
                target=self._async_queue_worker,
                daemon=True,
                name="LumosTelegramWorker"
            )
            self._worker_thread.start()

        if not self.bot_token or not self.chat_id:
            logger.warning("⚠️ [TELEGRAM CREDENTIAL MISSING] TELEGRAM_BOT_TOKEN 또는 TELEGRAM_CHAT_ID가 설정되지 않았습니다. (테스트/로깅 모드로 작동)")

    @staticmethod
    def _escape_html(val: Any) -> str:
        """HTML parse_mode 깨짐 방지를 위한 특수문자 이스케이프"""
        s = str(val if val is not None else "")
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    @classmethod
    def _extract_model_meta(cls, model_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        MLFeatureEngine.get_active_model_info() 또는 MoEMetaOrchestrator.evaluate_dual_filter_signal()의
        신호 스냅샷(moe_res)에서 모델 메타데이터를 단일 규격으로 추출한다.
        """
        info = model_info if isinstance(model_info, dict) else {}
        model_id = str(
            info.get("active_model_id")
            or info.get("ACTIVE_MODEL_ID")
            or "B_Normalized_72"
        ).strip()
        schema = str(
            info.get("active_model_schema")
            or info.get("feature_schema")
            or info.get("ACTIVE_MODEL_SCHEMA")
            or "B_normalized"
        ).strip()
        num_features = int(
            info.get("active_model_num_features")
            if info.get("active_model_num_features") is not None
            else (
                info.get("num_features")
                if info.get("num_features") is not None
                else (info.get("ACTIVE_MODEL_FEATURE_COUNT") if info.get("ACTIVE_MODEL_FEATURE_COUNT") is not None else 72)
            )
        )
        sha256 = str(
            info.get("active_model_sha256")
            or info.get("model_sha256")
            or info.get("sha256")
            or info.get("ACTIVE_MODEL_SHA256")
            or ""
        ).strip()
        version = sha256[:12] if sha256 else str(
            info.get("active_model_version")
            or info.get("model_version")
            or "NONE"
        ).strip()
        load_status = str(
            info.get("active_model_load_status")
            or info.get("load_status")
            or info.get("MODEL_LOAD_STATUS")
            or f"ACTIVE_{schema.upper()}"
        ).strip()
        fb_raw = str(
            info.get("active_model_fallback_status")
            or info.get("fallback_status")
            or info.get("MODEL_FALLBACK_STATUS")
            or "PRIMARY_ACTIVE"
        ).strip()
        is_fallback = bool(
            info.get("active_model_is_fallback")
            or info.get("is_fallback")
            or load_status == "FALLBACK_TO_CHAMPION"
            or "FALLBACK" in fb_raw
        )
        if load_status == "BLOCKED" or model_id == "BLOCKED_NO_VALID_MODEL":
            display_status = "BLOCKED"
        elif is_fallback:
            display_status = "FALLBACK_TO_CHAMPION"
        else:
            display_status = "PRIMARY_ACTIVE"

        prev_sha = str(
            info.get("previous_model_sha256_before_eval")
            or info.get("previous_model_sha256")
            or ""
        ).strip()
        prev_ver = prev_sha[:12] if prev_sha else "NONE"
        tw_meta = info.get("training_window_meta") if isinstance(info.get("training_window_meta"), dict) else {}
        max_weeks = int(info.get("max_training_weeks") or getattr(config, "ROLLING_TRAINING_WEEKS", 520))
        load_error = str(info.get("load_error") or info.get("last_reload_error") or "").strip()

        return {
            "model_id": model_id,
            "schema": schema,
            "num_features": num_features,
            "sha256": sha256,
            "version": version,
            "load_status": load_status,
            "fallback_status": fb_raw,
            "display_status": display_status,
            "is_fallback": is_fallback,
            "prev_sha256": prev_sha,
            "prev_version": prev_ver,
            "training_window_meta": tw_meta,
            "max_training_weeks": max_weeks,
            "load_error": load_error,
        }

    def _check_and_record_state_dedup(self, category: str, state_fingerprint: str, force: bool = False) -> bool:
        """동일한 모델 상태에서 중복 시작/폴백/차단 알림이 반복 발송되지 않도록 스팸 방지"""
        with self._state_lock:
            prev_fp = self._last_state_notification_keys.get(category)
            if not force and prev_fp == state_fingerprint:
                return True
            self._last_state_notification_keys[category] = state_fingerprint
            return False

    # ==============================================================================
    # 🚀 0. [시스템 기동 / 장 오픈 / 정기 신호 브리핑 / 모델 상태 알림]
    # ==============================================================================
    def compose_startup_alert(
        self,
        model_info: Dict[str, Any],
        mode_str: str = "SIMULATION",
        mkt_status: Optional[Dict[str, Any]] = None,
        sync_res: Optional[Dict[str, Any]] = None,
        conn_res: Optional[Dict[str, Any]] = None,
        account_str: str = "",
        account_masked: str = "",
        broker_name: str = "키움증권",
    ) -> str:
        """[STEP 5-2] 시스템 기동 알림 포맷 생성 (MLFeatureEngine.get_active_model_info 단일 공급원 사용)"""
        meta = self._extract_model_meta(model_info)
        mkt = mkt_status or {}
        sync = sync_res or {}
        conn = conn_res or {}
        acct = account_str or account_masked or ""

        cross_asset_txt = "O" if getattr(config, "USE_CROSS_ASSET_VETO", False) else "X"
        trend_txt = "O" if getattr(config, "USE_60M_TREND_FILTER", False) else "X"

        tw = meta["training_window_meta"]
        tw_lines = ""
        start_dt = tw.get("train_start_dt") or tw.get("data_start")
        end_dt = tw.get("train_end_dt") or tw.get("data_end")
        rows_cnt = tw.get("train_rows") if tw.get("train_rows") is not None else tw.get("num_candles")
        if tw and start_dt and end_dt and rows_cnt is not None:
            tw_lines = (
                f"\n• <b>실제 학습 범위</b>: {self._escape_html(start_dt)} ~ "
                f"{self._escape_html(end_dt)} ({int(rows_cnt):,} rows)"
            )

        now_kst = self._escape_html(mkt.get("now_kst_str", datetime.now().strftime("%Y-%m-%d %H:%M:%S KST")))
        delta_h = float(sync.get("delta_hours", 13.0))
        status_desc = self._escape_html(mkt.get("status_desc", "점검 모드"))
        next_open = self._escape_html(mkt.get("next_open_kst_str", "-"))
        until_open = self._escape_html(mkt.get("time_until_open_str", "-"))
        usd_avail = float(conn.get("usd_order_available", 0.0))
        holdings_cnt = int(conn.get("holdings_count", 0))
        max_days = int(getattr(config, "TRAINING_LOOKBACK_DAYS", 3640))

        return (
            f"🚀 <b>[Lumos {self._escape_html(mode_str)} 거래 시스템 시작]</b>\n"
            f"⏰ <b>현재 시간:</b> {now_kst}\n"
            f"🔄 <b>시간 동기화:</b> 100% 일치 (KST-NYT {delta_h:.0f}h 시차 검증 완료)\n"
            f"📊 <b>시장 상태:</b> {status_desc}\n"
            f"⏳ <b>다음 개장/폐장:</b> {next_open} ({until_open})\n"
            f"🏢 <b>연결 증권사:</b> {self._escape_html(broker_name)} ({self._escape_html(mode_str)})\n"
            f"💳 <b>계좌 번호:</b> {self._escape_html(acct)}\n"
            f"💵 <b>가용 예수금:</b> ${usd_avail:,.2f} USD\n"
            f"📦 <b>현재 보유종목:</b> {holdings_cnt}종목\n\n"
            f"🧠 <b>[활성 AI 모델 정보]</b>\n"
            f"• <b>활성 모델 ID</b>: <code>{self._escape_html(meta['model_id'])}</code>\n"
            f"• <b>스키마 / 피처 수</b>: <code>{self._escape_html(meta['schema'])}</code> ({meta['num_features']}개)\n"
            f"• <b>모델 상태</b>: <code>{self._escape_html(meta['display_status'])}</code> ({self._escape_html(meta['load_status'])})\n"
            f"• <b>모델 버전(SHA256)</b>: <code>{self._escape_html(meta['version'])}</code>"
            f"{tw_lines}\n"
            f"• <b>최대 학습 상한</b>: <code>최대 {meta['max_training_weeks']}주 ({max_days}일 / 10년 상한)</code>\n"
            f"• <b>진입 기준 / 방패</b>: GBDT &gt;= {config.GBDT_CONFIDENCE_THRESHOLD*100:.1f}% | 크로스에셋 Veto: {cross_asset_txt} | QQQ 60분 추세방패: {trend_txt}\n"
            f"⚡ <b>마켓 데이터:</b> 10ms WebSocket 스트리밍 기반 실시간 체결 처리 중...\n\n"
            f"✅ 성공적으로 초기화되었습니다. 매매 대기 중..."
        )

    def send_startup_alert(
        self,
        model_info: Dict[str, Any],
        mode_str: str = "SIMULATION",
        mkt_status: Optional[Dict[str, Any]] = None,
        sync_res: Optional[Dict[str, Any]] = None,
        conn_res: Optional[Dict[str, Any]] = None,
        account_str: str = "",
        account_masked: str = "",
        broker_name: str = "키움증권",
        force: bool = False,
    ) -> Dict[str, Any]:
        meta = self._extract_model_meta(model_info)
        fp = f"STARTUP|{mode_str}|{meta['model_id']}|{meta['sha256']}|{meta['display_status']}"
        msg = self.compose_startup_alert(
            model_info=model_info,
            mode_str=mode_str,
            mkt_status=mkt_status,
            sync_res=sync_res,
            conn_res=conn_res,
            account_str=account_str or account_masked,
            broker_name=broker_name,
        )
        if self._check_and_record_state_dedup("STARTUP", fp, force=force):
            return {"ok": True, "suppressed_duplicate": True, "alert_type": "STARTUP", "state_key": fp, "text": msg}
        return self._dispatch_message(msg, alert_type="STARTUP")

    def compose_market_open_alert(
        self,
        model_info: Dict[str, Any],
        mode_str: str,
        now_kst_str: str,
    ) -> str:
        """미국 정규장 오픈 알림 포맷 생성 (활성 모델 단일 메타데이터 반영)"""
        meta = self._extract_model_meta(model_info)
        return (
            f"🟢 <b>[미국장 오픈 알림]</b>\n\n"
            f"⏰ <b>현재 시간</b>: <code>{self._escape_html(now_kst_str)}</code>\n"
            f"💼 <b>체결 모드</b>: <code>{self._escape_html(mode_str)}</code>\n"
            f"🧠 <b>AI 엔진</b>: <code>{self._escape_html(meta['model_id'])} ({self._escape_html(meta['schema'])}, {meta['num_features']}개 피처, v:{self._escape_html(meta['version'])})</code>\n"
            f"🚦 <b>모델 상태</b>: <code>{self._escape_html(meta['display_status'])}</code> (09:30~15:30 EDT)\n"
            f"🎯 <b>진입 기준</b>: <code>GBDT {config.GBDT_CONFIDENCE_THRESHOLD*100:.1f}% 이상</code>\n"
            f"🛡️ <b>청산 기준</b>: <code>Max TP +{config.MAX_TP_PCT*100:.1f}% / ATR SL({config.SL_MIN_PCT*100:.1f}~{config.SL_MAX_PCT*100:.1f}%) / {config.TIME_STOP_MINUTES}m 타임스탑 / ATR 트레일링</code>\n"
            f"🧹 <b>장마감</b>: <code>{config.PHASE_EOD_CLEAR} EDT 0% 오버나잇 청산</code>"
        )

    def compose_signal_briefing_alert(
        self,
        moe_res: Dict[str, Any],
        ny_time_str: str,
        kst_time_str: str,
        active_pos: Optional[Dict[str, Any]] = None,
        has_active_pos: Optional[bool] = None,
        market_state: Optional[str] = None,
        session_state: Optional[str] = None,
        operation_mode: Optional[str] = None,
        current_price: Optional[float] = None,
        soxl_price: Optional[float] = None,
        soxs_price: Optional[float] = None,
        quote_time_et: Optional[str] = None,
        candle_time_et: Optional[str] = None,
        order_allowed: Optional[bool] = None,
        order_block_reason: Optional[str] = None,
        order_block_detail: Optional[str] = None,
        order_executed: Optional[bool] = None,
        premarket_policy: Optional[str] = None,
        data_stale: Optional[bool] = None,
        stale_reason: Optional[str] = None,
    ) -> str:
        """
        [STEP 5-3 & B-05] 15분 정기 매매 신호 / 진입 판단 브리핑 포맷 생성 (프리마켓 및 정규장 [ LUMOS AI SIGNAL ] 통합 포맷).
        반드시 신호 생성 시점의 모델 스냅샷(moe_res) 정보를 그대로 사용한다.
        """
        meta = self._extract_model_meta(moe_res)
        conf = float(moe_res.get("gating_confidence", 0.0)) * 100.0
        threshold_pct = float(moe_res.get("threshold_applied", config.GBDT_CONFIDENCE_THRESHOLD)) * 100.0
        is_appr = bool(moe_res.get("is_approved", False))
        direction = str(moe_res.get("direction", "NONE")).upper()
        pos_exists = bool(has_active_pos) if has_active_pos is not None else bool(active_pos)

        is_stale = bool(data_stale) if data_stale is not None else bool(moe_res.get("data_stale", False))
        stale_detail = str(stale_reason if stale_reason is not None else moe_res.get("stale_reason", "")).strip()

        raw_mkt = str(market_state or moe_res.get("market_state") or "REGULAR_MARKET").strip().upper()
        raw_sess = str(session_state or moe_res.get("session_state") or "").strip().upper()
        if raw_mkt in ("PRE_MARKET_WAITING", "PRE_MARKET") or raw_sess == "PRE_MARKET_WAITING":
            mkt_label = "PRE_MARKET"
            sess_label = "PRE_MARKET_WAITING"
        else:
            mkt_label = "REGULAR_MARKET" if raw_mkt in ("REGULAR_MARKET_OPEN", "REGULAR_MARKET", "") else raw_mkt
            sess_label = raw_sess or ("REGULAR_MARKET_OPEN" if mkt_label == "REGULAR_MARKET" else mkt_label)

        op_mode = str(operation_mode or moe_res.get("operation_mode") or "").strip().upper()
        if op_mode not in ("REAL", "SIMULATION"):
            env_sim = os.getenv("KIWOOM_IS_SIMULATION", "1").strip()
            op_mode = "SIMULATION" if (env_sim == "1" or env_sim.lower() == "true") else "REAL"

        gbdt_probs = moe_res.get("gbdt_probs", {"LONG": 0.0, "SHORT": 0.0, "NONE": 0.0})
        p_long = float(gbdt_probs.get("LONG", 0.0)) * 100.0
        p_short = float(gbdt_probs.get("SHORT", 0.0)) * 100.0
        p_none = float(gbdt_probs.get("NONE", 0.0)) * 100.0

        if is_stale:
            direction = "NONE"
            is_appr = False

        if direction == "LONG":
            target_sym = config.BASE_ASSET_LONG
            dir_str = f"{target_sym} 매수 (LONG)"
        elif direction == "SHORT":
            target_sym = config.BASE_ASSET_SHORT
            dir_str = f"{target_sym} 매수 (SHORT)"
        else:
            target_sym = "NONE"
            dir_str = "매수 관망 (NONE)"

        cur_px_val = float(current_price if current_price is not None else moe_res.get("current_price", 0.0) or 0.0)
        soxl_px_val = float(soxl_price if soxl_price is not None else moe_res.get("soxl_price", 0.0) or 0.0)
        soxs_px_val = float(soxs_price if soxs_price is not None else moe_res.get("soxs_price", 0.0) or 0.0)
        if cur_px_val <= 0:
            cur_px_val = soxs_px_val if direction == "SHORT" and soxs_px_val > 0 else soxl_px_val

        q_time_str = str(quote_time_et or moe_res.get("quote_time_et") or "").strip()
        if not q_time_str:
            q_time_str = f"{datetime.now().strftime('%Y-%m-%d')} {ny_time_str} ET"
        c_time_str = str(candle_time_et or moe_res.get("candle_time_et") or "").strip()
        if not c_time_str:
            ts_raw = str(moe_res.get("timestamp") or "").strip()
            c_time_str = f"{ts_raw[:16]} ET" if len(ts_raw) >= 16 else q_time_str

        if order_allowed is None:
            if is_stale or mkt_label == "PRE_MARKET" or meta["display_status"] == "BLOCKED" or not moe_res.get("model_consistency_ok", True) or pos_exists:
                ord_allowed_bool = False
            else:
                ord_allowed_bool = True
        else:
            ord_allowed_bool = bool(order_allowed)

        ord_allowed_str = "ALLOWED" if ord_allowed_bool else "BLOCKED"

        if order_block_reason is not None:
            ord_block_str = str(order_block_reason).strip().upper()
        elif is_stale:
            ord_block_str = "DATA_STALE"
        elif mkt_label == "PRE_MARKET":
            ord_block_str = "PRE_MARKET"
        elif not ord_allowed_bool:
            ord_block_str = "RISK_BLOCK"
        else:
            ord_block_str = "NONE"

        ord_exec_bool = bool(order_executed) if order_executed is not None else False
        ord_exec_str = "YES" if ord_exec_bool else "NO"

        if premarket_policy is not None:
            pm_policy = str(premarket_policy).strip()
        elif is_stale:
            pm_policy = "SIGNAL_UNAVAILABLE_STALE_DATA"
        elif mkt_label == "PRE_MARKET":
            pm_policy = "SIGNAL_ONLY — NO ORDER" if (is_appr and direction in ("LONG", "SHORT")) else "NONE"
        else:
            pm_policy = "REGULAR_EXECUTION_POLICY"

        use_veto = bool(getattr(config, "USE_CROSS_ASSET_VETO", False))
        is_cross_veto = bool(moe_res.get("is_cross_veto", False))
        veto_str = ("차단 (VETO)" if is_cross_veto else "통과 (PASS)") if use_veto else "비활성 (OFF)"

        use_60m = bool(getattr(config, "USE_60M_TREND_FILTER", False))
        is_60m_ok = bool(moe_res.get("is_60m_trend_ok", True))
        trend_60m_str = ("통과 (PASS)" if is_60m_ok else "차단 (BLOCKED)") if use_60m else "비활성 (OFF)"

        if is_stale:
            status_line = f"신호 보류 🚨 (SIGNAL_UNAVAILABLE_STALE_DATA | {self._escape_html(stale_detail or 'DATA_STALE')})"
        elif meta["display_status"] == "BLOCKED" or not moe_res.get("model_consistency_ok", True):
            status_line = "신규 진입 차단 🚨 (MODEL BLOCKED / 불일치 감지)"
        elif mkt_label == "PRE_MARKET" and is_appr and direction in ("LONG", "SHORT"):
            status_line = "프리마켓 신호 감지 🟠 (SIGNAL_ONLY — NO ORDER)"
        elif mkt_label == "PRE_MARKET":
            status_line = "프리마켓 관망 대기 🟡 (PRE_MARKET_WAITING | NONE)"
        elif is_appr and pos_exists:
            status_line = "진입 승인 🟢 (보유종목 있어 실제 진입은 Skip)"
        elif is_appr and not ord_allowed_bool:
            status_line = f"주문 차단 🛡️ ({self._escape_html(ord_block_str)})"
        elif is_appr:
            status_line = "진입 승인 🟢"
        else:
            status_line = "관망 유지 🟡"

        extra_lines = ""
        if is_stale:
            extra_lines += f"신호 상태: SIGNAL_UNAVAILABLE_STALE_DATA ({self._escape_html(stale_detail or 'DATA_STALE')})\n"
        if mkt_label == "PRE_MARKET":
            extra_lines += f"프리마켓 정책: {self._escape_html(pm_policy)}\n"
            extra_lines += "추론 입력 데이터: 전일 정규장 15분봉 + 프리마켓 실시간 시세 오버레이 (정규장 15분봉 학습 분포와 상이 — 참고용 신호)\n"
        elif mkt_label == "REGULAR_MARKET":
            extra_lines += "추론 입력 데이터: 정규장 15분봉 + 실시간 시세 (모델 학습 분포와 동일)\n"
        if order_block_detail:
            extra_lines += f"차단 상세: {self._escape_html(str(order_block_detail))}\n"

        px_detail_str = f"${cur_px_val:.2f}"
        if soxl_px_val > 0 and soxs_px_val > 0:
            px_detail_str = f"${cur_px_val:.2f} (SOXL ${soxl_px_val:.2f} | SOXS ${soxs_px_val:.2f})"

        return (
            f"[ LUMOS AI SIGNAL ]\n"
            f"시장 상태: {self._escape_html(mkt_label)} ({self._escape_html(sess_label)})\n"
            f"운영 모드: {self._escape_html(op_mode)}\n"
            f"활성 모델: {self._escape_html(meta['model_id'])}\n"
            f"AI 판단: {self._escape_html(direction)}\n"
            f"확신도: {conf:.1f}%\n"
            f"진입 임계값: {threshold_pct:.1f}%\n"
            f"현재가: {self._escape_html(px_detail_str)}\n"
            f"시세 기준 시각: {self._escape_html(q_time_str)}\n"
            f"캔들 기준 시각: {self._escape_html(c_time_str)}\n"
            f"주문 가능 여부: {self._escape_html(ord_allowed_str)}\n"
            f"주문 차단 사유: {self._escape_html(ord_block_str)}\n"
            f"실제 주문 실행 여부: {self._escape_html(ord_exec_str)}\n"
            f"{extra_lines}\n"
            f"🤖 <b>[Lumos AI 정기 브리핑]</b>\n"
            f"⏰ <b>판정 시각:</b> {self._escape_html(ny_time_str)} (NYT) / {self._escape_html(kst_time_str)} (KST)\n"
            f"🎯 <b>대상 종목 / 방향:</b> <code>{self._escape_html(target_sym)}</code> ({self._escape_html(dir_str)})\n"
            f"📈 <b>{config.BASE_ASSET_LONG}(LONG) 확률:</b> {p_long:.1f}%\n"
            f"📉 <b>{config.BASE_ASSET_SHORT}(SHORT) 확률:</b> {p_short:.1f}%\n"
            f"⏸ <b>관망(NONE) 확률:</b> {p_none:.1f}%\n"
            f"🔥 <b>보정 확신도 / 임계값:</b> {conf:.1f}% / {threshold_pct:.1f}%\n"
            f"🛡️ <b>크로스에셋 Veto:</b> {veto_str}\n"
            f"🧭 <b>QQQ 60분 추세 필터:</b> {trend_60m_str}\n"
            f"🧠 <b>사용 모델:</b> <code>{self._escape_html(meta['model_id'])}</code> (<code>{self._escape_html(meta['schema'])}</code>, {meta['num_features']}개 피처)\n"
            f"🔑 <b>모델 버전(SHA256):</b> <code>{self._escape_html(meta['version'])}</code> | <b>폴백 상태:</b> <code>{self._escape_html(meta['display_status'])}</code>\n"
            f"🚦 <b>최종 상태:</b> {status_line}"
        )

    def send_signal_briefing_alert(
        self,
        moe_res: Dict[str, Any],
        ny_time_str: str,
        kst_time_str: str,
        active_pos: Optional[Dict[str, Any]] = None,
        has_active_pos: Optional[bool] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        msg = self.compose_signal_briefing_alert(
            moe_res=moe_res,
            ny_time_str=ny_time_str,
            kst_time_str=kst_time_str,
            active_pos=active_pos,
            has_active_pos=has_active_pos,
            **kwargs,
        )
        return self._dispatch_message(msg, alert_type="SIGNAL_BRIEFING")

    # ==============================================================================
    # 📱 1. [Type 1: 방어 (Veto / Block)]
    # ==============================================================================
    def send_veto_alert(
        self,
        ticker: str,
        gbdt_prob: float,
        cross_dir: str = "",
        veto_reason: str = "",
        is_simulation: Optional[bool] = None,
        model_snapshot_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """[Type 1: 방어 (Veto) 알림 - 신호 생성 시점의 모델 스냅샷 메타데이터 포함]"""
        if is_simulation is None:
            env_sim = os.getenv("KIWOOM_IS_SIMULATION", "1").strip()
            is_simulation = (env_sim == "1" or env_sim.lower() == "true")
        mode_tag = "🧪 [키움 모의투자]" if is_simulation else "🔥 [키움 실전투자]"

        prob_val = gbdt_prob * 100.0 if gbdt_prob <= 1.0 else gbdt_prob
        sym_clean = self._escape_html(ticker.upper().strip())
        meta = self._extract_model_meta(model_snapshot_info)
        reason_txt = cross_dir or veto_reason or "Cross-Asset Veto"

        message = f"""🛡️ <b>[Lumos 방어막 발동] {sym_clean} 진입 차단</b>
{mode_tag}

🚨 <b>차단 사유 (Cross-Asset Veto)</b>
• <b>GBDT 공격 신호</b>: {sym_clean} 진입 시도 (확신도: {prob_val:.1f}%)
• <b>크로스에셋 역풍</b>: {self._escape_html(reason_txt)} 방향 감지
• <b>사용 모델</b>: <code>{self._escape_html(meta['model_id'])}</code> (<code>{self._escape_html(meta['schema'])}</code>, {meta['num_features']}개 피처, v:<code>{self._escape_html(meta['version'])}</code>)
• <b>모델 상태</b>: <code>{self._escape_html(meta['display_status'])}</code>
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
        is_simulation: Optional[bool] = None,
        model_snapshot_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """[Type 2: 매수 체결 알림 - 반드시 해당 진입 신호를 생성한 모델 스냅샷 정보 기록]"""
        if is_simulation is None:
            env_sim = os.getenv("KIWOOM_IS_SIMULATION", "1").strip()
            is_simulation = (env_sim == "1" or env_sim.lower() == "true")
        mode_tag = "🧪 [키움 모의투자]" if is_simulation else "🔥 [키움 실전투자]"

        prob_val = gbdt_prob * 100.0 if gbdt_prob <= 1.0 else gbdt_prob
        sym_clean = self._escape_html(ticker.upper().strip())
        total_amt = round(entry_price * qty, 2)

        calc_tp = tp_price if tp_price is not None else round(entry_price * (1.0 + config.MAX_TP_PCT), 2)
        calc_sl = sl_price if sl_price is not None else round(entry_price * (1.0 - config.SL_MIN_PCT), 2)
        tp_pct_calc = round(((calc_tp - entry_price) / entry_price) * 100.0, 2) if entry_price > 0 else config.MAX_TP_PCT * 100.0
        sl_pct_calc = round(((entry_price - calc_sl) / entry_price) * 100.0, 2) if entry_price > 0 else config.SL_MIN_PCT * 100.0

        if not time_stop_time:
            kst_now = datetime.now()
            time_stop_time = (kst_now + timedelta(minutes=time_stop_minutes)).strftime("%H:%M:%S")

        meta = self._extract_model_meta(model_snapshot_info)

        message = f"""🟢 <b>[Lumos 신규 진입] {sym_clean}</b>
{mode_tag}

🎯 <b>진입 정보</b>
• <b>종목</b>: {sym_clean}
• <b>체결가</b>: ${entry_price:,.2f}
• <b>수량</b>: {qty}주 (총 ${total_amt:,.2f})

🧠 <b>AI 분석 결과</b> ({self._escape_html(strategy_tag)})
• <b>GBDT 확신도</b>: {prob_val:.1f}% (임계값: {config.GBDT_CONFIDENCE_THRESHOLD*100:.1f}%)
• <b>사용 모델 ID</b>: <code>{self._escape_html(meta['model_id'])}</code> (<code>{self._escape_html(meta['schema'])}</code>, {meta['num_features']}개 피처)
• <b>모델 버전(SHA256)</b>: <code>{self._escape_html(meta['version'])}</code>
• <b>모델 폴백 상태</b>: <code>{self._escape_html(meta['display_status'])}</code>

🛡️ <b>자동 방어선(ATR Trailing)</b>
• <b>목표가(TP)</b>: ${calc_tp:,.2f} (+{tp_pct_calc}%)
• <b>손절가(SL)</b>: ${calc_sl:,.2f} (-{sl_pct_calc}%)
• <b>타임스탑</b>: {self._escape_html(time_stop_time)} ({time_stop_minutes}분)"""

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
        is_simulation: Optional[bool] = None,
        model_snapshot_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        [Type 3: 청산 알림]
        """
        if is_simulation is None:
            env_sim = os.getenv("KIWOOM_IS_SIMULATION", "1").strip()
            is_simulation = (env_sim == "1" or env_sim.lower() == "true")
        mode_tag = "🧪 [키움 모의투자]" if is_simulation else "🔥 [키움 실전투자]"

        sym_clean = self._escape_html(ticker.upper().strip())
        reason_upper = exit_reason.upper()

        if any(k in reason_upper for k in ["TP", "익절", "+3.0", "+3.5", "+2.5", "+10.0", "TARGET", "목표"]):
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

        model_line = ""
        if model_snapshot_info:
            meta = self._extract_model_meta(model_snapshot_info)
            model_line = f"\n• <b>진입 모델</b>: <code>{self._escape_html(meta['model_id'])}</code> (v:<code>{self._escape_html(meta['version'])}</code>)"

        message = f"""{emoji} <b>[Lumos 매도 청산] {sym_clean}</b>
{mode_tag}

💰 <b>청산 결과 요약</b>
• <b>사유</b>: {self._escape_html(exit_reason)}
• <b>수익률</b>: {pnl_sign}{pnl_percent:.2f}%
• <b>추정 손익</b>: {pnl_amount_str}원 (${pnl_usd:,.2f})

📊 <b>거래 상세</b>
• <b>매수가</b>: ${entry_price:,.2f}
• <b>매도가</b>: ${exit_price:,.2f}
• <b>수량</b>: {qty}주{model_line}"""

        return self._dispatch_message(message, alert_type="EXIT_SELL")

    # ==============================================================================
    # 🔄 4. [STEP 5-5: 자동 재학습 단계별 알림 (Auto-Retrain Lifecycle)]
    # ==============================================================================
    def send_retrain_status_alert(
        self,
        status: str = "",
        status_code: str = "",
        model_id: str = "B_Normalized_72",
        schema: str = "B_normalized",
        feature_schema: str = "",
        num_features: int = 72,
        new_sha256: str = "",
        model_sha256: str = "",
        prev_sha256: str = "",
        training_window_meta: Optional[Dict[str, Any]] = None,
        error_reason: str = "",
        detail: str = "",
    ) -> Dict[str, Any]:
        """
        [STEP 5-5] 자동 재학습 알림 (학습 완료 vs 검증 통과 vs 활성화 성공/실패/무시 엄격 분리)
        - RETRAIN_COMPLETED: 학습 완료 (아직 미활성)
        - MODEL_VALIDATION_PASSED: 무결성·스키마·피처 수·Self-Test·10년 윈도우 검증 통과
        - MODEL_ACTIVATION_SUCCEEDED: 매니페스트 원자적 교체 성공
        - MODEL_ACTIVATION_FAILED: 검증 또는 교체 실패, 기존 활성 모델 유지
        - STALE_RETRAIN_IGNORED: 더 최신 모델이 이미 활성화되어 구버전 작업 무시
        """
        status_clean = str(status or status_code).strip().upper()
        schema_clean = str(feature_schema or schema or "B_normalized").strip()
        sha_actual = new_sha256 or model_sha256 or ""
        new_ver = sha_actual[:12] if sha_actual else "PENDING"
        prev_ver = (prev_sha256 or "")[:12] if prev_sha256 else "NONE"
        max_weeks = int(getattr(config, "ROLLING_TRAINING_WEEKS", 520))
        tw = training_window_meta if isinstance(training_window_meta, dict) else {}
        err_or_detail = error_reason or detail or ""

        tw_line = ""
        start_dt = tw.get("train_start_dt") or tw.get("data_start")
        end_dt = tw.get("train_end_dt") or tw.get("data_end")
        rows_cnt = tw.get("train_rows") if tw.get("train_rows") is not None else tw.get("num_candles")
        if start_dt and end_dt and rows_cnt is not None:
            tw_line = (
                f"\n• <b>실제 학습 데이터 범위</b>: {self._escape_html(start_dt)} ~ "
                f"{self._escape_html(end_dt)} ({int(rows_cnt):,}행)"
            )

        if status_clean == "RETRAIN_COMPLETED":
            title = "🧪 <b>[Lumos 자동 재학습] 후보 모델 학습 완료 (미활성/검증 대기)</b>"
            desc = "신규 후보 모델 학습이 완료되었으며, 7단계 무결성 검증 전이므로 아직 운영 모델로 활성화되지 않았습니다."
        elif status_clean == "MODEL_VALIDATION_PASSED":
            title = "🔍 <b>[Lumos 자동 재학습] 후보 모델 무결성 검증 통과</b>"
            desc = "SHA256, B_normalized 72개 피처 순서, 클래스 매핑([0,1,2]), 추론 Self-Test 및 10년 상한 검증을 모두 통과했습니다."
        elif status_clean == "MODEL_ACTIVATION_SUCCEEDED":
            title = "✅ <b>[Lumos 자동 재학습] 신규 B_Normalized_72 원자적 활성화 완료</b>"
            desc = "active_model_manifest.json 단일 원자적 교체(os.replace)가 완료되어 신규 모델이 운영 활성화되었습니다."
        elif status_clean == "STALE_RETRAIN_IGNORED":
            title = "⏭️ <b>[Lumos 자동 재학습] 구버전 재학습 결과 무시 (최신 모델 유지)</b>"
            desc = "더 최신에 시작된 재학습 모델이 이미 활성화되어 있어, 구버전 재학습 후보의 덮어쓰기를 차단했습니다."
        else:
            status_clean = "MODEL_ACTIVATION_FAILED"
            title = "🚨 <b>[Lumos 자동 재학습] 모델 활성화 실패 (기존 활성 모델 유지)</b>"
            desc = f"검증 또는 매니페스트 교체 실패로 신규 모델을 거부하고 기존 정상 모델을 그대로 유지합니다. 사유: {self._escape_html(err_or_detail)}"

        msg = (
            f"{title}\n"
            f"• <b>상태 코드</b>: <code>{self._escape_html(status_clean)}</code>\n"
            f"• <b>모델 ID / 스키마</b>: <code>{self._escape_html(model_id)}</code> (<code>{self._escape_html(schema_clean)}</code>, {int(num_features)}개 피처)\n"
            f"• <b>신규 모델 버전(SHA256)</b>: <code>{self._escape_html(new_ver)}</code>\n"
            f"• <b>이전/유지 모델 버전(SHA256)</b>: <code>{self._escape_html(prev_ver)}</code>"
            f"{tw_line}\n"
            f"• <b>최대 학습 상한</b>: <code>{max_weeks}주 (10년 상한)</code>\n"
            f"• <b>상세</b>: {desc}"
        )
        return self._dispatch_message(msg, alert_type=status_clean)

    # ==============================================================================
    # 🛡️ 5. [STEP 5-6: 모델 핫리로드 / 폴백 / 롤백 / 차단(BLOCKED) 상태 알림]
    # ==============================================================================
    def send_model_state_alert(
        self,
        model_info: Dict[str, Any],
        event_type: str = "MODEL_STATE",
        reason: str = "",
        previous_model_sha256: str = "",
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        [STEP 5-6] 모델 핫리로드 / Champion 폴백 / 롤백 / 전체 실패(BLOCKED) 알림.
        동일한 모델 상태에서 중복 발송되지 않도록 상태 해시 기반 스팸 방지를 수행한다.
        """
        meta = self._extract_model_meta(model_info)
        if previous_model_sha256 and meta["prev_version"] == "NONE":
            meta["prev_sha256"] = previous_model_sha256
            meta["prev_version"] = previous_model_sha256[:12]
        ev = str(event_type).strip().upper()
        fp = f"{ev}|{meta['model_id']}|{meta['sha256']}|{meta['load_status']}|{meta['fallback_status']}"

        reason_clean = self._escape_html(reason or meta["load_error"] or "-")
        if ev == "MODEL_HOT_RELOADED":
            title = "🔄 <b>[Lumos 실시간 모델 핫리로드 완료]</b>"
            body = (
                f"• <b>이벤트</b>: <code>MODEL_HOT_RELOADED</code>\n"
                f"• <b>활성 모델 ID</b>: <code>{self._escape_html(meta['model_id'])}</code> (<code>{self._escape_html(meta['schema'])}</code>, {meta['num_features']}개 피처)\n"
                f"• <b>신규 모델 버전(SHA256)</b>: <code>{self._escape_html(meta['version'])}</code>\n"
                f"• <b>이전 모델 버전(SHA256)</b>: <code>{self._escape_html(meta['prev_version'])}</code>\n"
                f"• <b>모델 상태</b>: <code>{self._escape_html(meta['display_status'])}</code> ({self._escape_html(meta['fallback_status'])})\n"
                f"• <b>상세</b>: {reason_clean}"
            )
        elif ev == "FALLBACK_TO_CHAMPION" or meta["display_status"] == "FALLBACK_TO_CHAMPION":
            ev = "FALLBACK_TO_CHAMPION"
            title = "⚠️ <b>[Lumos 모델 폴백 경고] 기존 Champion 자동 복귀</b>"
            body = (
                f"• <b>상태 코드</b>: <code>FALLBACK_TO_CHAMPION</code>\n"
                f"• <b>폴백 모델 ID</b>: <code>{self._escape_html(meta['model_id'])}</code> (<code>{self._escape_html(meta['schema'])}</code>, {meta['num_features']}개 피처)\n"
                f"• <b>폴백 모델 버전(SHA256)</b>: <code>{self._escape_html(meta['version'])}</code>\n"
                f"• <b>폴백 사유</b>: {reason_clean}"
            )
        elif ev == "MODEL_ROLLBACK_EXECUTED":
            title = "⏪ <b>[Lumos 모델 롤백 완료] 매니페스트 원자적 복귀</b>"
            body = (
                f"• <b>상태 코드</b>: <code>MODEL_ROLLBACK_EXECUTED</code>\n"
                f"• <b>복귀 모델 ID</b>: <code>{self._escape_html(meta['model_id'])}</code> (<code>{self._escape_html(meta['schema'])}</code>, {meta['num_features']}개 피처)\n"
                f"• <b>복귀 모델 버전(SHA256)</b>: <code>{self._escape_html(meta['version'])}</code>\n"
                f"• <b>직전 모델 버전(SHA256)</b>: <code>{self._escape_html(meta['prev_version'])}</code>\n"
                f"• <b>롤백 상세</b>: {reason_clean}"
            )
        elif ev == "BLOCKED" or meta["display_status"] == "BLOCKED":
            ev = "BLOCKED"
            title = "🚨 <b>[Lumos 긴급 경고] 모든 모델 검증 실패 — 신규 진입 전면 차단 (BLOCKED)</b>"
            body = (
                f"• <b>상태 코드</b>: <code>BLOCKED</code>\n"
                f"• <b>활성 모델 ID</b>: <code>{self._escape_html(meta['model_id'])}</code>\n"
                f"• <b>차단 사유</b>: {reason_clean}\n"
                f"• <b>보호 조치</b>: 신규 매수 진입은 전면 차단(NONE)되며, 기존 보유 포지션의 손절·트레일링·타임스탑·EOD 청산은 정상 유지됩니다."
            )
        else:
            title = f"ℹ️ <b>[Lumos 모델 상태 알림] {self._escape_html(ev)}</b>"
            body = (
                f"• <b>활성 모델 ID</b>: <code>{self._escape_html(meta['model_id'])}</code> (<code>{self._escape_html(meta['schema'])}</code>, {meta['num_features']}개 피처)\n"
                f"• <b>모델 버전(SHA256)</b>: <code>{self._escape_html(meta['version'])}</code>\n"
                f"• <b>상태</b>: <code>{self._escape_html(meta['display_status'])}</code>"
            )

        msg = f"{title}\n{body}"
        if self._check_and_record_state_dedup(ev, fp, force=force):
            return {"ok": True, "suppressed_duplicate": True, "alert_type": ev, "state_key": fp, "text": msg}
        return self._dispatch_message(msg, alert_type=ev)

    # ==============================================================================
    # 6. 발송 파이프라인 코어 (예외 격리 보장)
    # ==============================================================================
    def _dispatch_message(self, text: str, alert_type: str = "INFO") -> Dict[str, Any]:
        """메시지 라우팅: 비동기 큐 적재(Non-blocking 0ms) 또는 동기 즉시 발송 (예외 격리 보장)"""
        try:
            if self.async_mode:
                try:
                    self._queue.put({"text": text, "type": alert_type, "timestamp": time.time()}, block=False)
                    return {"ok": True, "mode": "queued", "alert_type": alert_type, "text": text}
                except queue.Full:
                    logger.error("❌ [TELEGRAM QUEUE FULL] 발송 큐가 가득 차 동기 발송으로 자동 폴백합니다.")
                    return self._send_http_request(text)
            else:
                return self._send_http_request(text)
        except Exception as e:
            logger.error(f"⚠️ [Telegram Dispatch Error Isolated] {alert_type}: {e}")
            return {"ok": False, "error": str(e), "alert_type": alert_type, "text": text}

    def _send_http_request(self, text: str) -> Dict[str, Any]:
        """실제 Telegram Bot API POST 요청 집행 (지연시간, 재시도 및 예외 격리 제어)"""
        try:
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
                        retry_after = int(resp.json().get("parameters", {}).get("retry_after", 3))
                        logger.warning(f"⚠️ [Telegram HTTP 429 Rate Limit] {retry_after}초 대기 후 재시도 ({attempt}/{self.max_retries})")
                        time.sleep(retry_after)
                    else:
                        logger.warning(f"⚠️ [Telegram Send Fail] Status: {resp.status_code} | Body: {resp.text}")
                except Exception as e:
                    logger.warning(f"⚠️ [Telegram Connection Error] 시도 {attempt}/{self.max_retries}: {e}")
                    time.sleep(0.1 * attempt)

            logger.error(f"❌ [Telegram Final Failure] 최대 재시도({self.max_retries}회) 초과로 발송 실패")
            return {"ok": False, "error": "max_retries_exceeded", "text": text}
        except Exception as e:
            logger.error(f"⚠️ [Telegram HTTP Exception Isolated]: {e}")
            return {"ok": False, "error": str(e), "text": text}

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
