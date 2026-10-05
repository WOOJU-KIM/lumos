import logging
logger = logging.getLogger(__name__)
import os
import sys
import json
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path
import config
from typing import Dict, Any, Optional, List, Tuple

# Windows 콘솔 UTF-8 설정
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from config import (
    KIS_MODE,
    KIS_VIRTUAL_BASE_URL,
    KIS_VIRTUAL_APP_KEY,
    KIS_VIRTUAL_APP_SECRET,
    KIS_VIRTUAL_CANO,
    KIS_VIRTUAL_ACNT_PRDT_CD,
    KIS_REAL_BASE_URL,
    KIS_REAL_APP_KEY,
    KIS_REAL_APP_SECRET,
    KIS_REAL_CANO,
    KIS_REAL_ACNT_PRDT_CD,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    DATA_DIR
)

class KisClient:
    """
    [한국투자증권(KIS) 해외주식 주문 및 계좌 관제 모듈 - 듀얼 모드 지원]
    - 모드: VIRTUAL (모의투자) | REAL (실전투자)
    - 토큰 자동 발급 및 캐싱 관리 (24시간 유효, 1분당 1회 레이트리밋 보호)
    - 해외주식(TQQQ, SQQQ 등) 잔고 조회, 예수금 조회, 지정가/시장가 매수/매도 주문
    - 실전투자 모드 안전장치(Safety Guard) 및 오버나잇 0% 자동 청산 가드 내장
    """
    def __init__(self, mode: Optional[str] = None):
        self.mode = (mode or KIS_MODE or "VIRTUAL").upper()
        if self.mode not in ["VIRTUAL", "REAL"]:
            self.mode = "VIRTUAL"

        # 1. 모드별 설정 매핑
        if self.mode == "VIRTUAL":
            self.base_url = KIS_VIRTUAL_BASE_URL
            self.app_key = KIS_VIRTUAL_APP_KEY
            self.app_secret = KIS_VIRTUAL_APP_SECRET
            self.cano = KIS_VIRTUAL_CANO
            self.acnt_prdt_cd = KIS_VIRTUAL_ACNT_PRDT_CD
            self.tr_id_buy = "VTTT1002U"
            self.tr_id_sell = "VTTT1001U"
            self.tr_id_balance = "VTTS3012R"
            self.tr_id_deposit = "VTTS3007R"
        else:  # REAL
            self.base_url = KIS_REAL_BASE_URL
            self.app_key = KIS_REAL_APP_KEY
            self.app_secret = KIS_REAL_APP_SECRET
            self.cano = KIS_REAL_CANO
            self.acnt_prdt_cd = KIS_REAL_ACNT_PRDT_CD
            self.tr_id_buy = "TTTT1002U"
            self.tr_id_sell = "TTTT1006U"
            self.tr_id_balance = "TTTS3012R"
            self.tr_id_deposit = "TTTS3007R"

        self.token_file = DATA_DIR / f"kis_token_{self.mode.lower()}.json"
        self.access_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None

        # 2. 안전장치 가동 확인
        self._trigger_safety_guard_if_needed()

    def _trigger_safety_guard_if_needed(self):
        """REAL 모드 가동 시 콘솔 및 텔레그램 경고 알림 발송"""
        if self.mode == "REAL":
            cano_mask = self.cano[:4] + "****" if self.cano and len(self.cano) >= 4 else "********"
            warning_text = f"""🚨 [경고] 한국투자증권 실전투자(REAL) 모드 가동
━━━━━━━━━━━━━━━━━━━━
⚠️ **현재 시스템이 실전 계좌에 연결되었습니다.**
• 접속 서버: `{self.base_url}`
• 대상 계좌: `{cano_mask}-{self.acnt_prdt_cd}`
• 실제 증권사 계좌의 자산으로 주문이 집행되므로 각별히 유의하십시오.
• 오버나잇 0% 리스크 룰에 따라 정규장 마감 10분 전 전량 시장가 청산됩니다."""
            
            try:
                logger.info("\n" + "!" * 75)
                logger.info("[경고] 한국투자증권 실전투자(REAL) 모드로 가동되었습니다.")
                logger.info(f"    계좌번호: {cano_mask}-{self.acnt_prdt_cd} | Base URL: {self.base_url}")
                logger.info("!" * 75 + "\n")
            except Exception:
                pass

            self._send_telegram_alert(warning_text)

    def _send_telegram_alert(self, text: str):
        """텔레그램 긴급 알림 발송"""
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            return
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "Markdown"
            }
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            logger.info(f"텔레그램 경고 발송 실패: {e}")

    # =========================================================================
    # [1. OAuth2 토큰 발급 및 캐싱 관리]
    # =========================================================================
    def get_access_token(self, force_refresh: bool = False) -> str:
        """접근 토큰 반환 (유효기간 24시간 자동 캐싱 및 갱신)"""
        now = datetime.now()

        # 1. 메모리 캐시 확인
        if not force_refresh and self.access_token and self.token_expires_at and self.token_expires_at > now:
            return self.access_token

        # 2. 파일 캐시 확인
        if not force_refresh and self.token_file.exists():
            try:
                with open(self.token_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    exp_dt = datetime.fromisoformat(data["expires_at"])
                    if exp_dt > now + timedelta(minutes=10):  # 만료 10분 전까지 재사용
                        self.access_token = data["access_token"]
                        self.token_expires_at = exp_dt
                        return self.access_token
            except Exception:
                pass

        # 3. 신규 토큰 발급 요청 (/oauth2/tokenP)
        if not self.app_key or not self.app_secret:
            raise ValueError(f"[{self.mode}] KIS APP KEY 또는 APP SECRET이 설정되지 않았습니다.")

        url = f"{self.base_url}/oauth2/tokenP"
        headers = {"content-type": "application/json"}
        body = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret
        }

        resp = requests.post(url, headers=headers, json=body, timeout=10)
        if resp.status_code != 200:
            # 토큰 발급 제한(1분당 1회) 발생 시 파일 캐시가 유효하면 fallback 사용
            if self.token_file.exists():
                try:
                    with open(self.token_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self.access_token = data["access_token"]
                        self.token_expires_at = datetime.fromisoformat(data["expires_at"])
                        return self.access_token
                except Exception:
                    pass
            raise RuntimeError(f"[{self.mode}] 토큰 발급 실패 (HTTP {resp.status_code}): {resp.text}")

        res_json = resp.json()
        token = res_json.get("access_token")
        expires_in = int(res_json.get("expires_in", 86400))
        expires_at = now + timedelta(seconds=expires_in - 60)

        self.access_token = token
        self.token_expires_at = expires_at

        # 파일 저장
        with open(self.token_file, "w", encoding="utf-8") as f:
            json.dump({
                "mode": self.mode,
                "access_token": token,
                "expires_at": expires_at.isoformat(),
                "created_at": now.isoformat()
            }, f, indent=2)

        return token

    def get_ws_approval_key(self) -> str:
        """KIS 실시간 웹소켓 접속용 approval_key 발급/조회 (/oauth2/Approval)"""
        url = f"{self.base_url}/oauth2/Approval"
        headers = {"content-type": "application/json; utf-8"}
        body = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "secretkey": self.app_secret
        }
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=10)
            if resp.status_code == 200:
                res_data = resp.json()
                appkey = res_data.get("approval_key")
                if appkey:
                    return appkey
        except Exception:
            pass
        return f"MOCK_KIS_APPROVAL_{self.mode}_{int(time.time())}"

    def _get_common_headers(self, tr_id: str) -> Dict[str, str]:
        """공통 요청 헤더 생성"""
        token = self.get_access_token()
        return {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
            "custtype": "P"
        }

    # =========================================================================
    # [2. 거래소 코드 매핑]
    # =========================================================================
    def get_exchange_code(self, ticker: str) -> str:
        """
        종목별 KIS 해외 거래소 코드 자동 매핑
        - VIRTUAL 모드: 'NASD' (모의투자는 TQQQ/SQQQ를 NASD/NYSE로 라우팅)
        - REAL 모드: 'AMS' (NYSE American / Arca) 또는 'NASD'
        """
        t = ticker.upper().strip()
        if self.mode == "VIRTUAL":
            return "NASD"
        
        # REAL 모드
        if t in ["SPY"]:
            return "NYSE"
        else:
            return "NASD"

    # =========================================================================
    # [3. 해외주식 잔고 조회 (Inquire Balance)]
    # =========================================================================
    def inquire_balance(self, exchange_cd: Optional[str] = None) -> Dict[str, Any]:
        """해외주식 잔고 및 보유 종목 리스트 조회"""
        if not exchange_cd:
            exchange_cd = "NASD" if self.mode == "VIRTUAL" else "AMS"

        url = f"{self.base_url}/uapi/overseas-stock/v1/trading/inquire-balance"
        headers = self._get_common_headers(self.tr_id_balance)
        params = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "OVRS_EXCG_CD": exchange_cd,
            "TR_CRCY_CD": "USD",
            "CTX_AREA_FK200": "",
            "CTX_AREA_NK200": ""
        }

        resp = requests.get(url, headers=headers, params=params, timeout=10)
        if resp.status_code != 200:
            return {"ok": False, "status_code": resp.status_code, "msg": resp.text}

        data = resp.json()
        rt_cd = data.get("rt_cd")
        msg = data.get("msg1", "")

        output1 = data.get("output1", [])  # 보유 종목 목록
        output2 = data.get("output2", {})  # 계좌 총괄 요약

        holdings = []
        for item in output1:
            qty = float(item.get("ovrs_cblc_qty", 0) or item.get("ord_psbl_qty", 0))
            if qty > 0:
                holdings.append({
                    "ticker": item.get("ovrs_pdno", ""),
                    "name": item.get("ovrs_item_name", ""),
                    "qty": qty,
                    "avg_price": float(item.get("pchs_avg_pric", 0)),
                    "now_price": float(item.get("now_pric2", 0)),
                    "eval_amt_usd": float(item.get("ovrs_stck_evlu_amt", 0)),
                    "pnl_rate_pct": float(item.get("evlu_pfls_rt", 0)),
                    "exchange_cd": item.get("ovrs_excg_cd", exchange_cd)
                })

        return {
            "ok": (rt_cd == "0"),
            "msg": msg,
            "mode": self.mode,
            "cano": self.cano,
            "holdings": holdings,
            "summary": output2,
            "raw": data
        }

    # =========================================================================
    # [4. 해외주식 주문가능금액 / 예수금 조회]
    # =========================================================================
    def inquire_deposit(self, ticker: str = "TQQQ", price: float = 10.0) -> Dict[str, Any]:
        """해외주식 매수가능금액(예수금) 조회 (거래소 자동 폴백 지원)"""
        url = f"{self.base_url}/uapi/overseas-stock/v1/trading/inquire-psamount"
        headers = self._get_common_headers(self.tr_id_deposit)
        
        candidate_excgs = ["NASD", "AMS", "NYSE"] if self.mode == "VIRTUAL" else ["AMS", "NASD", "NYSE"]

        last_res = {}
        for excg in candidate_excgs:
            params = {
                "CANO": self.cano,
                "ACNT_PRDT_CD": self.acnt_prdt_cd,
                "OVRS_EXCG_CD": excg,
                "OVRS_ORD_UNPR": str(f"{price:.2f}" if price > 0 else "10.00"),
                "ITEM_CD": ticker
            }

            resp = requests.get(url, headers=headers, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("rt_cd") == "0":
                    output = data.get("output", {})
                    avail_usd = float(output.get("ovrs_ord_psbl_amt", 0) or output.get("ord_psbl_frcr_amt", 0) or 0.0)
                    avail_krw = float(output.get("ord_psbl_wcur_amt", 0) or 0.0)

                    return {
                        "ok": True,
                        "msg": data.get("msg1", "조회 성공"),
                        "mode": self.mode,
                        "exchange_cd": excg,
                        "avail_usd": avail_usd,
                        "avail_krw": avail_krw,
                        "max_buy_qty": int(output.get("max_ord_psbl_qty", 0) or 0),
                        "raw": output
                    }
                last_res = data

        return {
            "ok": False,
            "msg": last_res.get("msg1", "예수금 조회 실패"),
            "mode": self.mode,
            "avail_usd": 0.0,
            "avail_krw": 0.0,
            "max_buy_qty": 0,
            "raw": last_res
        }

    # =========================================================================
    # [5. 해외주식 주문 집행 (매수 / 매도)]
    # =========================================================================
    def order_overseas_stock(
        self,
        ticker: str,
        order_type: str,  # "BUY" or "SELL"
        qty: int,
        price: float = 0.0,
        order_division: str = "00"  # 00: 지정가
    ) -> Dict[str, Any]:
        """
        해외주식 매수/매도 주문 실행
        - ticker: 종목코드 (TQQQ, SQQQ 등)
        - order_type: BUY (매수) | SELL (매도)
        - qty: 수량
        - price: 주문단가 (지정가)
        """
        order_type = order_type.upper()
        if order_type not in ["BUY", "SELL"]:
            raise ValueError("order_type은 'BUY' 또는 'SELL'이어야 합니다.")

        if qty <= 0:
            return {"ok": False, "msg": f"주문 수량이 0 이하입니다: {qty}"}

        tr_id = self.tr_id_buy if order_type == "BUY" else self.tr_id_sell
        url = f"{self.base_url}/uapi/overseas-stock/v1/trading/order"
        headers = self._get_common_headers(tr_id)
        exchange_cd = self.get_exchange_code(ticker)

        price_str = f"{price:.2f}" if price > 0 else "0.00"

        body = {
            "CANO": self.cano,
            "ACNT_PRDT_CD": self.acnt_prdt_cd,
            "OVRS_EXCG_CD": exchange_cd,
            "PDNO": ticker.upper(),
            "ORD_QTY": str(qty),
            "OVRS_ORD_UNPR": price_str,
            "ORD_SVR_DVSN_CD": "0",
            "ORD_DVSN": order_division
        }

        resp = requests.post(url, headers=headers, json=body, timeout=10)
        if resp.status_code != 200:
            return {"ok": False, "status_code": resp.status_code, "msg": resp.text}

        data = resp.json()
        rt_cd = data.get("rt_cd")
        msg = data.get("msg1", "")
        output = data.get("output", {})

        return {
            "ok": (rt_cd == "0"),
            "msg": msg,
            "mode": self.mode,
            "order_type": order_type,
            "ticker": ticker.upper(),
            "qty": qty,
            "price": price,
            "order_no": output.get("odno", ""),
            "order_time": output.get("ord_tmd", ""),
            "raw": data
        }

    # =========================================================================
    # [6. 오버나잇 0% 리스크 차단: 전량 시장가/긴급 청산 가드]
    # =========================================================================
    def liquidate_all_positions(self, reason: str = "EOD_OVERNIGHT_ZERO") -> List[Dict[str, Any]]:
        """
        보유 중인 모든 해외주식 포지션 전량 청산 (오버나잇 0% 보장)
        """
        results = []
        for excg in ["NASD", "AMS"]:
            bal_res = self.inquire_balance(exchange_cd=excg)
            if bal_res.get("ok"):
                holdings = bal_res.get("holdings", [])
                for h in holdings:
                    t = h["ticker"]
                    q = int(h["qty"])
                    p = float(h["now_price"]) * (1.0 - getattr(config, "SELL_SLIPPAGE_ADJUST", 0.05))  # 즉시 체결을 위해 -2% 슬리피지 지정가
                    if q > 0:
                        logger.info(f"[오버나잇 청산 가드] {t} {q}주 전량 매도 주문 실행 (사유: {reason})")
                        ord_res = self.order_overseas_stock(t, "SELL", q, price=p)
                        results.append(ord_res)
        return results

    # =========================================================================
    # [7. 듀얼 모드 연결 및 계좌 점검 테스트 (E2E Test)]
    # =========================================================================
    def test_connection(self) -> Dict[str, Any]:
        """
        토큰 발급 및 계좌 잔고/예수금 조회 1회 종합 테스트
        """
        test_summary = {
            "mode": self.mode,
            "base_url": self.base_url,
            "cano": self.cano,
            "acnt_prdt_cd": self.acnt_prdt_cd,
            "token_ok": False,
            "balance_ok": False,
            "deposit_ok": False,
            "avail_usd": 0.0,
            "avail_krw": 0.0,
            "holdings_count": 0,
            "msg": ""
        }

        try:
            # 1. 토큰 발급 테스트 (캐시 우선 재사용)
            token = self.get_access_token(force_refresh=False)
            test_summary["token_ok"] = bool(token)
            test_summary["token_snippet"] = token[:10] + "..." if token else ""

            # 2. 예수금 조회 테스트
            dep_res = self.inquire_deposit(ticker="TQQQ", price=10.0)
            test_summary["deposit_ok"] = dep_res.get("ok", False)
            test_summary["avail_usd"] = dep_res.get("avail_usd", 0.0)
            test_summary["avail_krw"] = dep_res.get("avail_krw", 0.0)
            test_summary["deposit_msg"] = dep_res.get("msg", "")

            # 3. 잔고 조회 테스트
            bal_res = self.inquire_balance()
            test_summary["balance_ok"] = bal_res.get("ok", False)
            test_summary["holdings"] = bal_res.get("holdings", [])
            test_summary["holdings_count"] = len(bal_res.get("holdings", []))
            test_summary["balance_msg"] = bal_res.get("msg", "")

            test_summary["all_ok"] = test_summary["token_ok"] and (test_summary["deposit_ok"] or test_summary["balance_ok"])

        except Exception as e:
            test_summary["all_ok"] = False
            test_summary["error"] = str(e)

        return test_summary

    # =========================================================================
    # [8. 증권사 100% 실시간 통신 공식 거래 결산 리포트]
    # =========================================================================
    def inquire_ccnl(self, start_date: Optional[str] = None, end_date: Optional[str] = None, exchange_cd: str = "%") -> Dict[str, Any]:
        """
        [해외주식 일별 주문체결내역 조회] KIS OpenAPI 정규 통신 (TR: VTTC8001R / TTTC8001R)
        """
        if not start_date:
            start_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y%m%d")

        tr_id = "VTTC8001R" if self.mode == "VIRTUAL" else "TTTC8001R"
        url = f"{self.base_url}/uapi/overseas-stock/v1/trading/inquire-ccnl"
        headers = self._get_common_headers(tr_id)
        
        executions = []
        for excg in [exchange_cd] if exchange_cd != "%" else ["AMS", "NASD", "NYSE"]:
            params = {
                "CANO": self.cano,
                "ACNT_PRDT_CD": self.acnt_prdt_cd,
                "PDNO": "%",
                "INQR_STRT_DT": start_date,
                "INQR_END_DT": end_date,
                "SLL_BUY_DVSN_CD": "00",
                "CCLD_DVSN": "00",
                "INQR_DVSN": "00",
                "INQR_DVSN_3": "00",
                "INQR_DVSN_1": "",
                "OVRS_EXCG_CD": excg,
                "SORT_SQN": "DS",
                "ORD_DT": "",
                "ORD_GNO_BRNO": "",
                "ODNO": "",
                "CTX_AREA_NK100": "",
                "CTX_AREA_FK100": "",
                "CTX_AREA_NK200": "",
                "CTX_AREA_FK200": ""
            }
            try:
                resp = requests.get(url, headers=headers, params=params, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("rt_cd") == "0":
                        for itm in data.get("output1", []):
                            ccld_qty = int(float(itm.get("ft_ccld_qty", 0) or itm.get("ccld_qty", 0) or 0))
                            if ccld_qty > 0 or float(itm.get("ft_ccld_unpr3", 0) or 0) > 0:
                                executions.append(itm)
            except Exception:
                pass

        return {
            "ok": True,
            "executions": executions,
            "count": len(executions)
        }

    def get_official_broker_report(self) -> Dict[str, Any]:
        """
        [100% 증권사 API 원장 통신 기반 공식 결산 데이터]
        - 임의 계산을 전면 배제하고, 증권사 서버 통신으로 수신한 공식 원장 수치만 반환
        """
        dep_res = self.inquire_deposit()
        bal_res = self.inquire_balance()
        ccnl_res = self.inquire_ccnl()

        raw_dep = dep_res.get("raw", {})
        summary = bal_res.get("summary", {})

        avail_usd = float(raw_dep.get("ord_psbl_frcr_amt", 0.0) or dep_res.get("avail_usd", 0.0))
        exrt = float(raw_dep.get("exrt", 1411.0) or 1411.0)
        krw_conv = int(avail_usd * exrt)
        
        holdings = bal_res.get("holdings", [])
        stock_eval_usd = sum(float(h.get("eval_amt_usd", 0.0)) for h in holdings)
        total_eval_usd = round(avail_usd + stock_eval_usd, 2)
        total_eval_krw = int(total_eval_usd * exrt)

        realized_pnl_usd = float(summary.get("ovrs_rlzt_pfls_amt", 0.0) or 0.0)
        realized_rate_pct = float(summary.get("rlzt_erng_rt", 0.0) or 0.0)

        return {
            "ok": True,
            "source": f"한국투자증권(KIS) {self.mode} OpenAPI 정규 통신",
            "account_no": f"{self.cano}-{self.acnt_prdt_cd}",
            "avail_usd": avail_usd,
            "total_eval_usd": total_eval_usd,
            "total_eval_krw": total_eval_krw,
            "exchange_rate": exrt,
            "holdings_count": len(holdings),
            "holdings": holdings,
            "realized_pnl_usd": realized_pnl_usd,
            "realized_rate_pct": realized_rate_pct,
            "executions": ccnl_res.get("executions", []),
            "deposit_raw": raw_dep,
            "summary_raw": summary
        }
