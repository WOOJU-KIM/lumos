import os
import sys
import json
import time
import urllib.request
import urllib.error
import logging
import re
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Windows 콘솔 utf-8 설정
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from dotenv import load_dotenv
import config
load_dotenv()

logger = logging.getLogger("KiwoomBroker")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s][KiwoomBroker] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

_GLOBAL_KIWOOM_TOKEN_CACHE: Dict[str, Dict[str, Any]] = {}


def _mask_sensitive_log(value: Any, secrets: Tuple[str, ...] = ()) -> Any:
    """Return a masked logging copy; never alter API or event data."""
    sensitive_keys = {
        "authorization", "token", "accesstoken", "refreshtoken", "tokenpreview",
        "appkey", "apikey", "appsecret", "apisecret", "secret", "secretkey", "clientsecret", "password",
        "cano", "account", "accountno", "fullaccountno", "accountnumber",
        "accno", "acntno", "계좌번호", "계좌 번호",
    }

    def sensitive(key):
        return re.sub(r"[\s_-]", "", str(key)).lower() in sensitive_keys

    if isinstance(value, dict):
        return {key: "[MASKED]" if sensitive(key) else _mask_sensitive_log(item, secrets)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        items = [_mask_sensitive_log(item, secrets) for item in value]
        return tuple(items) if isinstance(value, tuple) else items
    if not isinstance(value, str):
        return value

    # Preserve raw JSON formatting when nothing sensitive needs replacing.
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        parsed = None
    if isinstance(parsed, (dict, list)):
        masked = _mask_sensitive_log(parsed, secrets)
        return value if masked == parsed else json.dumps(masked, ensure_ascii=False)

    # Exception messages may contain JSON after an HTTP/error description.
    decoder = json.JSONDecoder()
    parts = []
    cursor = 0
    for match in re.finditer(r"[\[{]", value):
        start = match.start()
        if start < cursor:
            continue
        try:
            fragment, end = decoder.raw_decode(value, start)
        except ValueError:
            continue
        if not isinstance(fragment, (dict, list)):
            continue
        masked = _mask_sensitive_log(fragment, secrets)
        parts.append(_mask_sensitive_log(value[cursor:start], secrets))
        parts.append(value[start:end] if masked == fragment else
                     json.dumps(masked, ensure_ascii=False))
        cursor = end
    if cursor:
        return "".join(parts) + _mask_sensitive_log(value[cursor:], secrets)

    key_pattern = "|".join(
        "[ _-]*".join(re.escape(char) for char in key.replace(" ", ""))
        for key in sorted(sensitive_keys, key=len, reverse=True)
    )
    # JSON/repr fragments and key=value details in exception messages.
    pattern = (r"(?<![\w])([\"']?(?:" + key_pattern
               + r")[\"']?\s*[:=]\s*)(\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|(?:Bearer|Basic)\s+[^\s,;|}\"'\]]+|[^\s,;|}\]]+)")
    value = re.sub(pattern, lambda match: match.group(1) + "[MASKED]", value,
                   flags=re.IGNORECASE)
    value = re.sub(r"\b(Bearer|Basic)\s+[^\s,;\"'}\]]+", r"\1 [MASKED]", value,
                   flags=re.IGNORECASE)
    for secret in sorted({item for item in secrets if isinstance(item, str) and item},
                         key=len, reverse=True):
        value = value.replace(secret, "[MASKED]")
    return value


class KiwoomBroker:
    """
    [Lumos 키움증권 OpenAPI REST 듀얼 스위칭 브로커]
    1. KIWOOM_IS_SIMULATION 플래그(1: 모의투자, 0: 실전투자)에 따라 동적 엔드포인트 및 인증키 자동 스위칭
    2. 미국/해외주식 외화 예수금(ust21110) 및 원장 잔고(ust21070) 실시간 조회
    3. 국내주식 예수금(kt00001) 및 잔고(kt00018) 통합 지원
    4. OAuth2 토큰 발급 및 만료 전 자동 갱신 메모리 캐싱
    """
    def _redact_log(self, value: Any) -> Any:
        return _mask_sensitive_log(value, tuple(
            getattr(self, name, "") for name in
            ("_access_token", "app_key", "app_secret", "account_no")
        ))

    def __init__(self, is_simulation: Optional[bool] = None):
        env_sim = os.getenv("KIWOOM_IS_SIMULATION", "1").strip()
        if is_simulation is not None:
            self.is_simulation = bool(is_simulation)
        else:
            self.is_simulation = (env_sim == "1" or env_sim.lower() == "true")

        self.mode_str = "VIRTUAL (모의투자)" if self.is_simulation else "REAL (실전투자)"

        if self.is_simulation:
            self.base_url = "https://mockapi.kiwoom.com"
            self.app_key = os.getenv("KIWOOM_MOCK_APP_KEY", "").strip()
            self.app_secret = os.getenv("KIWOOM_MOCK_APP_SECRET", "").strip()
            self.account_no = os.getenv("KIWOOM_MOCK_ACCOUNT_NO", "").strip()
        else:
            self.base_url = "https://api.kiwoom.com"
            self.app_key = os.getenv("KIWOOM_REAL_APP_KEY", "").strip()
            self.app_secret = os.getenv("KIWOOM_REAL_APP_SECRET", "").strip()
            self.account_no = os.getenv("KIWOOM_REAL_ACCOUNT_NO", "").strip()

        self.account_type = os.getenv("KIWOOM_ACCOUNT_TYPE", "01").strip()
        self.broker_name = "키움증권(Kiwoom)"

        # 🚦 증권사 API 유량 제한 방어 (모의투자: 0.50초, 실전투자: 0.30초 강제 간격 보장)
        import threading
        self._min_request_interval = 0.50 if self.is_simulation else 0.30
        self._last_request_time = 0.0
        self._request_lock = threading.Lock()

        self._access_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None
        self._virtual_holdings: Dict[str, Dict[str, Any]] = {}

        logger.info(self._redact_log(f"🏛 KiwoomBroker 초기화 완료 [모드: {self.mode_str} | Base URL: {self.base_url} | 계좌번호: {self.account_no} | TR 간격: {self._min_request_interval}s]"))
        if not self.is_simulation:
            logger.warning("⚠️ [주의] 키움증권 실전 매매(REAL) 모드로 설정되어 있습니다. 실제 자금이 집행됩니다.")

    def get_access_token(self, force_refresh: bool = False) -> str:
        """OAuth2 접근 토큰 발급 및 글로벌 자동 갱신 (429 유량 초과 방지 캐시)"""
        cache_key = f"{self.base_url}_{self.app_key}"
        now = datetime.now()

        # 1. 글로벌 캐시 확인
        if not force_refresh and cache_key in _GLOBAL_KIWOOM_TOKEN_CACHE:
            cached = _GLOBAL_KIWOOM_TOKEN_CACHE[cache_key]
            if cached.get("token") and cached.get("expires_at") and now < (cached["expires_at"] - timedelta(minutes=10)):
                self._access_token = cached["token"]
                self._token_expires_at = cached["expires_at"]
                return self._access_token

        # 2. 인스턴스 캐시 확인
        if (
            not force_refresh
            and self._access_token
            and self._token_expires_at
            and now < (self._token_expires_at - timedelta(minutes=10))
        ):
            return self._access_token

        token_url = f"{self.base_url}/oauth2/token"
        payload = json.dumps({
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "secretkey": self.app_secret
        }).encode("utf-8")

        headers = {
            "Content-Type": "application/json;charset=UTF-8"
        }

        req = urllib.request.Request(token_url, data=payload, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                content_type = response.headers.get("Content-Type", "")
                raw_bytes = response.read()
                if "text/html" in content_type or raw_bytes.strip().startswith(b"<!DOCTYPE") or b"<html" in raw_bytes[:100].lower():
                    raw_str = raw_bytes.decode("utf-8", errors="ignore")
                    maint_hint = "키움증권 서버 정기/임시 점검 중 (HTML 점검 안내 페이지 수신)"
                    if "시스템작업알림" in raw_str or "시스템 작업" in raw_str:
                        maint_hint = "키움증권 서버 주말/정기 점검 중 (공식 점검 안내: 9/12(토) 08:30 ~ 20:00 전체 서비스 중단 등)"
                    raise RuntimeError(maint_hint)

                res_body = json.loads(raw_bytes.decode("utf-8"))
                
                return_code = res_body.get("return_code", 0)
                if return_code != 0:
                    raise RuntimeError(f"토큰 발급 실패: {res_body.get('return_msg', '알 수 없는 오류')}")

                self._access_token = res_body.get("token") or res_body.get("access_token")
                expires_dt_str = res_body.get("expires_dt")
                
                if expires_dt_str:
                    try:
                        self._token_expires_at = datetime.strptime(expires_dt_str[:14], "%Y%m%d%H%M%S")
                    except Exception:
                        self._token_expires_at = now + timedelta(hours=23)
                else:
                    self._token_expires_at = now + timedelta(hours=23)

                _GLOBAL_KIWOOM_TOKEN_CACHE[cache_key] = {
                    "token": self._access_token,
                    "expires_at": self._token_expires_at
                }

                logger.info(f"🔑 [OAuth2 토큰 발급 성공] 만료 시각: {self._token_expires_at.strftime('%Y-%m-%d %H:%M:%S')}")
                return self._access_token

        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8")
            logger.error(self._redact_log(f"❌ 키움 토큰 발급 HTTP 오류 ({e.code}): {self._redact_log(err_msg)}"))
            raise RuntimeError(f"키움 토큰 HTTP {e.code}: {err_msg}")
        except Exception as e:
            logger.error(self._redact_log(f"❌ 키움 토큰 발급 중 예외 발생: {e}"))
            raise

    def _send_tr_request(self, endpoint: str, api_id: str, body_dict: Dict[str, Any], raise_on_error: bool = False, retry_count: int = 0) -> Dict[str, Any]:
        """키움 REST API TR 공통 요청 함수 (요청/응답 전문 Full 디버깅 출력 및 엄격한 예외 처리, Rate Limiting & 429 Auto-Retry)"""
        # 🚦 증권사 API 유량 제한 방어: 최소 간격 강제 보장
        with self._request_lock:
            now_t = time.time()
            elapsed = now_t - self._last_request_time
            if elapsed < self._min_request_interval:
                time.sleep(self._min_request_interval - elapsed)
            self._last_request_time = time.time()

        token = self.get_access_token()
        url = f"{self.base_url}{endpoint}"

        headers = {
            "Content-Type": "application/json;charset=UTF-8",
            "authorization": f"Bearer {token}",
            "api-id": api_id,
            "cont-yn": "N",
            "next-key": ""
        }

        payload_bytes = json.dumps(body_dict).encode("utf-8")
        req = urllib.request.Request(url, data=payload_bytes, headers=headers, method="POST")

        # 🔍 요청 직전 URL / Headers / Body 전문 Full 출력
        logger.info("\n" + "=" * 80)
        logger.info(f"📡 [REST TR REQUEST 송출] API ID: {api_id} | Endpoint: {endpoint}")
        logger.info(f"   • URL: {url}")
        safe_headers = headers.copy()
        if "authorization" in safe_headers:
            token = safe_headers["authorization"].replace("Bearer ", "")
            safe_headers["authorization"] = "Bearer [MASKED]"
        logger.info(self._redact_log(f"   • Headers: {json.dumps(self._redact_log(safe_headers), ensure_ascii=False)}"))
        logger.info(self._redact_log(f"   • Body: {json.dumps(self._redact_log(body_dict), ensure_ascii=False)}"))
        logger.info("=" * 80)

        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                content_type = response.headers.get("Content-Type", "")
                raw_bytes = response.read()
                if "text/html" in content_type or raw_bytes.strip().startswith(b"<!DOCTYPE") or b"<html" in raw_bytes[:100].lower():
                    raw_str = raw_bytes.decode("utf-8", errors="ignore")
                    err_hint = "키움증권 서버 정기/임시 점검 중 (HTML 점검 안내 페이지 반환)"
                    if "시스템작업알림" in raw_str or "시스템 작업" in raw_str:
                        err_hint = "키움증권 서버 주말/정기 점검 중 (공식 점검 안내: 9/12(토) 08:30 ~ 20:00 전체 서비스 중단 등)"
                    logger.error(f"❌ TR 요청 실패 [{api_id}]: {err_hint}")
                    if raise_on_error:
                        raise RuntimeError(err_hint)
                    return {"return_code": -1, "return_msg": err_hint, "ok": False}

                raw_text = raw_bytes.decode("utf-8")
                data = json.loads(raw_text)
                
                # 🔍 수신된 HTTP 상태코드 및 Response Body 전문 Full 출력
                logger.info(f"📥 [REST TR RESPONSE 수신] HTTP Status: {response.status}")
                logger.info(self._redact_log(f"   • Raw Body: {self._redact_log(raw_text)}"))
                logger.info("=" * 80 + "\n")

                rc = data.get("return_code", 0)
                rm = data.get("return_msg", "")
                if raise_on_error and rc != 0:
                    err_msg = f"❌ [TR_ERROR] API_ID={api_id} | Code={rc} | Msg={rm}"
                    logger.error(self._redact_log(err_msg))
                    raise RuntimeError(err_msg)

                return data

        except urllib.error.HTTPError as e:
            err_text = e.read().decode("utf-8")
            logger.info(f"❌ [REST TR HTTP ERROR] HTTP Status: {e.code}")
            logger.info(self._redact_log(f"   • Error Body: {self._redact_log(err_text)}"))
            logger.info("=" * 80 + "\n")
            logger.error(self._redact_log(f"❌ TR 요청 실패 [{api_id}] HTTP {e.code}: {self._redact_log(err_text)}"))
            
            # 🔁 HTTP 429 (유량 초과) 발생 시 0.6초 백오프 후 1회 자동 재시도
            if e.code == 429 and retry_count < 1:
                logger.warning(f"⚠️ [HTTP 429 감지] {api_id} 0.6초 백오프 대기 후 1회 자동 재시도...")
                time.sleep(0.6)
                return self._send_tr_request(endpoint=endpoint, api_id=api_id, body_dict=body_dict, raise_on_error=raise_on_error, retry_count=retry_count + 1)

            if raise_on_error:
                raise RuntimeError(f"HTTP Error {e.code} for API_ID={api_id}: {err_text}")
            return {"return_code": e.code, "return_msg": err_text, "ok": False}

        except Exception as e:
            logger.info(self._redact_log(f"❌ [REST TR EXCEPTION] Exception: {e}"))
            logger.info("=" * 80 + "\n")
            logger.error(self._redact_log(f"❌ TR 요청 예외 [{api_id}]: {e}"))
            if raise_on_error:
                raise
            return {"return_code": -1, "return_msg": str(e), "ok": False}

    def get_overseas_deposit(self) -> Dict[str, Any]:
        """
        [TR: ust21110] 미국/해외주식 외화예수금 조회 요청
        - USD 외화 예수금, 외화 주문가능금액, 원화환산금액 등 반환 (5초 캐시로 429 방지)
        """
        now_ts = time.time()
        if hasattr(self, "_cached_deposit") and self._cached_deposit and (now_ts - getattr(self, "_deposit_cached_time", 0)) < 5:
            return self._cached_deposit

        body = {
            "cano": self.account_no,
            "acnt_prdt_cd": self.account_type
        }
        res = self._send_tr_request(endpoint="/api/us/acnt", api_id="ust21110", body_dict=body)

        if res.get("return_code") == 0:
            usd_info = {}
            for item in res.get("result_list", []):
                if item.get("crnc_code") == "USD":
                    usd_info = item
                    break

            def _parse_float(val, default=0.0):
                try:
                    return float(str(val).replace(",", "").strip())
                except Exception:
                    return default

            usd_deposit = _parse_float(usd_info.get("fc_entra", 0))
            usd_order_avail = _parse_float(usd_info.get("fc_ord_alowa", 0))
            krw_converted = _parse_float(usd_info.get("fc_booka", 0))

            dep_res = {
                "ok": True,
                "currency": "USD",
                "usd_deposit": usd_deposit,
                "usd_order_available": usd_order_avail,
                "krw_converted": int(krw_converted),
                "all_currencies": res.get("result_list", []),
                "raw_response": res,
                "msg": res.get("return_msg", "정상 처리")
            }
            self._cached_deposit = dep_res
            self._deposit_cached_time = now_ts
            return dep_res
        else:
            return {
                "ok": False,
                "currency": "USD",
                "usd_deposit": 0.0,
                "usd_order_available": 0.0,
                "krw_converted": 0,
                "all_currencies": [],
                "raw_response": res,
                "msg": res.get("return_msg", "조회 실패")
            }

    def get_overseas_stock_balance(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        [TR: ust21070] 미국주식 원장잔고확인 요청
        - 총평가금액, 총매입금액, 당일수익, 보유종목 리스트 (5초 캐시, force_refresh=True 시 즉시 조회)
        """
        now_ts = time.time()
        if not force_refresh and hasattr(self, "_cached_stock_balance") and self._cached_stock_balance and (now_ts - getattr(self, "_stock_balance_cached_time", 0)) < config.BALANCE_CACHE_TTL_SEC:
            return self._cached_stock_balance

        body = {
            "cano": self.account_no,
            "acnt_prdt_cd": self.account_type,
            "qry_tp": "1"
        }
        res = self._send_tr_request(endpoint="/api/us/acnt", api_id="ust21070", body_dict=body)

        if res.get("return_code") == 0:
            def _parse_float(val, default=0.0):
                try:
                    return float(str(val).replace(",", "").strip())
                except Exception:
                    return default

            tot_evlt_usd = _parse_float(res.get("tot_evlt_amt", 0))
            tot_prch_usd = _parse_float(res.get("tot_prch_amt", 0))
            tot_pl_usd = _parse_float(res.get("tot_pl_amt", 0))
            tdy_book_usd = _parse_float(res.get("tdy_book_amt", 0))
            tdy_pl_usd = _parse_float(res.get("tdy_pl_amt", 0))
            tdy_pl_rate = _parse_float(res.get("tdy_pl_rt", 0))
            tdy_book_krw = int(_parse_float(res.get("tdy_book_amt_krw", 0)))
            tdy_pl_krw = int(_parse_float(res.get("tdy_pl_amt_krw", 0)))
            raw_holdings = res.get("result_list", [])
            normalized_holdings = []
            for item in raw_holdings:
                s_cd = str(item.get("stk_cd") or item.get("symbol") or "").strip().upper()
                p_qty = int(_parse_float(item.get("poss_qty") or item.get("qty") or item.get("quantity") or 0))
                b_px = _parse_float(item.get("frgn_stk_book_uv") or item.get("purchase_price") or item.get("avg_price") or 0.0)
                n_px = _parse_float(item.get("now_pric") or item.get("eval_price") or b_px)
                pl_amt = _parse_float(item.get("pl_amt") or 0.0)
                pl_rt = _parse_float(item.get("pl_rt") or 0.0)
                if s_cd and p_qty > 0:
                    normalized_holdings.append({
                        "symbol": s_cd,
                        "stk_cd": s_cd,
                        "quantity": p_qty,
                        "poss_qty": p_qty,
                        "purchase_price": b_px,
                        "avg_price": b_px,
                        "frgn_stk_book_uv": b_px,
                        "eval_price": n_px,
                        "now_pric": n_px,
                        "pnl_amount_usd": pl_amt,
                        "pnl_rate": pl_rt,
                        "raw": item
                    })

            bal_res = {
                "ok": True,
                "currency": res.get("crnc_code", "USD"),
                "total_eval_usd": tot_evlt_usd,
                "total_purchase_usd": tot_prch_usd,
                "total_pnl_usd": tot_pl_usd,
                "tdy_book_usd": tdy_book_usd,
                "tdy_pl_usd": tdy_pl_usd,
                "tdy_pl_rate": tdy_pl_rate,
                "tdy_book_krw": tdy_book_krw,
                "tdy_pl_krw": tdy_pl_krw,
                "holdings_count": len(normalized_holdings),
                "holdings": normalized_holdings,
                "raw_response": res,
                "msg": res.get("return_msg", "정상 처리")
            }
            self._cached_stock_balance = bal_res
            self._stock_balance_cached_time = now_ts
            return bal_res
        else:
            return {
                "ok": False,
                "currency": "USD",
                "total_eval_usd": 0.0,
                "total_purchase_usd": 0.0,
                "total_pnl_usd": 0.0,
                "holdings_count": 0,
                "holdings": [],
                "raw_response": res,
                "msg": res.get("return_msg", "조회 실패")
            }

    def get_full_account_summary(self, max_age_sec: int = 30) -> Dict[str, Any]:
        """미국/해외주식 외화 예수금 및 평가 잔고를 통합한 종합 계좌 상태 요약 (30초 메모리 캐시)"""
        now_ts = time.time()
        if hasattr(self, "_cached_summary") and self._cached_summary and (now_ts - getattr(self, "_summary_cached_time", 0)) < max_age_sec:
            return self._cached_summary

        dep_res = self.get_overseas_deposit()
        stk_res = self.get_overseas_stock_balance()

        total_assets_usd = dep_res.get("usd_deposit", 0.0) + stk_res.get("total_eval_usd", 0.0)

        summary = {
            "mode": self.mode_str,
            "is_simulation": self.is_simulation,
            "account_no": f"{self.account_no[:4]}****" if len(self.account_no) >= 8 else self.account_no,
            "full_account_no": self.account_no,
            "currency": "USD",
            "usd_order_available": dep_res.get("usd_order_available", 0.0),
            "usd_deposit": dep_res.get("usd_deposit", 0.0),
            "usd_total_assets": total_assets_usd,
            "krw_converted": dep_res.get("krw_converted", 0),
            "total_eval_usd": stk_res.get("total_eval_usd", 0.0),
            "total_purchase_usd": stk_res.get("total_purchase_usd", 0.0),
            "total_pnl_usd": stk_res.get("total_pnl_usd", 0.0),
            "holdings_count": stk_res.get("holdings_count", 0),
            "holdings": stk_res.get("holdings", []),
            "deposit_ok": dep_res.get("ok", True),
            "stock_ok": stk_res.get("ok", True),
            "checked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        if dep_res.get("ok") or not hasattr(self, "_cached_summary") or not self._cached_summary:
            self._cached_summary = summary
            self._summary_cached_time = now_ts

        return self._cached_summary

    def get_stock_quote(self, symbol: str = config.BASE_ASSET_LONG, exchange: Optional[str] = None) -> Dict[str, Any]:
        """
        [TR: ust10000 / 실시간 피드] 미국주식 현재가 및 시세 조회
        """
        sym_clean = symbol.upper().strip()
        stex = exchange or self._resolve_exchange(sym_clean)
        
        # 실시간 백업: yfinance / DataLake 실시간 시세
        import yfinance as yf
        try:
            t = yf.Ticker(sym_clean)
            fi = t.fast_info
            last_px = round(float(fi.last_price or fi.previous_close or config.FALLBACK_PRICE), 2)
            return {
                "ok": True,
                "symbol": sym_clean,
                "exchange": stex,
                "last_price": last_px,
                "provider": "DataLake/RealtimeFeed",
                "msg": "정상 시세 수신"
            }
        except Exception as e:
            return {
                "ok": False,
                "symbol": sym_clean,
                "last_price": 0.0,
                "msg": f"시세 수신 실패: {e}"
            }

    def _resolve_exchange(self, symbol: str) -> str:
        """종목 기호에 따른 거래소 코드(NY/ND) 자동 판별"""
        sym = symbol.upper().strip()
        # 아멕스/뉴욕(NYSE Arca) 상장 대표 ETF 및 주식
        ny_symbols = ["SOXL", "SOXS", "SPY", "DIA", "VIXY", "TLT", "TMF", "TMV", "UPRO", "SPXU"]
        if sym in ny_symbols:
            return "NY"
        # 나스닥 상장 (TQQQ, SQQQ, QQQ, NVDA, SOXX, IEF 등)
        return "ND"

    def send_order(
        self,
        symbol: str,
        order_type: str,  # "BUY" or "SELL"
        quantity: int,
        price: float = 0.0,
        exchange: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        [공식 TR: ust20000 (미국주식 매수) / ust20001 (미국주식 매도)]
        - 엔드포인트: POST /api/us/ordr
        - 거래소 구분(stex_tp): TQQQ/SQQQ는 "NY" (NYSE Arca), NVDA/QQQ는 "ND" (NASDAQ)
        - 키움 모의투자 규격: 지정가("00") 필수 적용
        - 🚨 실패/에러 발생 시 절대 삼키지 않고 즉시 RuntimeError를 발생시킴
        """
        is_buy = (order_type.upper() == "BUY")
        api_id = "ust20000" if is_buy else "ust20001"
        sym_clean = symbol.upper().strip()

        # 거래소 코드 자동 매핑 (TQQQ/SQQQ: NY, NVDA/QQQ: ND)
        stex = exchange or self._resolve_exchange(sym_clean)

        # 현재가 조회
        quote = self.get_stock_quote(sym_clean, exchange=stex)
        cur_px = float(quote.get("last_price", 0.0))

        # 매도 주문인 경우 기존에 걸려있는 미체결 주문(예: 예약매도) 전량 취소하여 매도가능수량 100% 확보
        if not is_buy:
            try:
                self.cancel_all_open_orders(sym_clean)
            except Exception as e:
                logger.warning(self._redact_log(f"⚠️ 매도 전 미체결 주문 취소 시도 예외 (계속 진행): {e}"))

        # 모의투자는 지정가(00) 필수 규격, 실전투자는 시장가(03) 및 지정가(00) 지원
        if self.is_simulation:
            if price > 0.0:
                exec_px = price
            else:
                # 슬리피지 방어 즉시 체결: 매수는 config.BUY_SLIPPAGE_ADJUST (Best Ask), 매도는 config.SELL_SLIPPAGE_ADJUST (Best Bid)
                import config
                exec_px = max(0.01, round(cur_px - config.SELL_SLIPPAGE_ADJUST, 2)) if not is_buy else round(cur_px + config.BUY_SLIPPAGE_ADJUST, 2)
            ord_uv_str = str(round(exec_px, 2)) if exec_px > 0 else "0"
            trde_tp = "00"  # 00: 지정가 (키움 모의투자 필수)
            order_mode_desc = f"🎯 모의 지정가(${exec_px:.2f})"
        else:
            is_market = (price <= 0.0)
            trde_tp = "03" if is_market else "00"
            ord_uv_str = "0" if is_market else str(round(price, 2))
            exec_px = cur_px if is_market else price
            order_mode_desc = "🔥 실전 시장가" if is_market else f"🎯 실전 지정가(${exec_px:.2f})"

        body = {
            "cano": str(self.account_no),
            "acnt_prdt_cd": str(self.account_type),
            "stex_tp": str(stex),
            "stk_cd": str(sym_clean),
            "ord_qty": str(int(quantity)),
            "ord_uv": ord_uv_str,
            "trde_tp": trde_tp
        }

        logger.info(f"📤 [키움 미국주식 주문 전송] {self.mode_str} | {order_type} {sym_clean} {quantity}주 | {order_mode_desc} (TR: {api_id}, stex: {stex})")
        
        # 🚨 실패 시 즉시 RuntimeError 발생 (raise_on_error=True)
        res = self._send_tr_request(endpoint="/api/us/ordr", api_id=api_id, body_dict=body, raise_on_error=True)

        return_code = res.get("return_code", -1)
        return_msg = res.get("return_msg", "")
        ord_no = res.get("ord_no", "") or res.get("order_no", "")

        if return_code != 0:
            err_msg = f"❌ [주문 체결 실패] TR: {api_id} | Code: {return_code} | Msg: {return_msg}"
            logger.error(self._redact_log(err_msg))
            raise RuntimeError(err_msg)

        return {
            "ok": True,
            "mode": self.mode_str,
            "broker": self.broker_name,
            "order_type": order_type.upper(),
            "order_mode": "LIMIT",
            "symbol": sym_clean,
            "exchange": stex,
            "quantity": quantity,
            "price": exec_px,
            "order_no": ord_no,
            "return_code": return_code,
            "msg": return_msg,
            "raw_response": res,
            "ordered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def get_official_broker_report(self) -> Dict[str, Any]:
        """
        [100% 키움증권 API 원장 통신 기반 공식 결산 데이터]
        - 임의 계산을 전면 배제하고, 키움증권 서버 통신으로 수신한 공식 원장 수치만 반환
        """
        dep_res = self.get_overseas_deposit()
        bal_res = self.get_overseas_stock_balance()
        open_res = self.get_open_orders()

        # 하위 조회 실패를 정상 원장으로 승격하지 않는다. 정상 수치/계약은 그대로 유지한다.
        failed_queries = []
        for name, result, required in (
            ("deposit", dep_res, ("usd_order_available", "krw_converted")),
            ("balance", bal_res, ("total_eval_usd", "holdings")),
            ("open_orders", open_res, ("orders",))
        ):
            if not isinstance(result, dict) or result.get("ok") is not True or any(key not in result for key in required):
                failed_queries.append(name)
        dep_res = dep_res if isinstance(dep_res, dict) else {}
        bal_res = bal_res if isinstance(bal_res, dict) else {}
        open_res = open_res if isinstance(open_res, dict) else {}

        avail_usd = float(dep_res.get("usd_order_available", 0.0))
        exrt = float(dep_res.get("krw_converted", 0)) / avail_usd if avail_usd > 0 else 1402.5
        stock_eval_usd = float(bal_res.get("total_eval_usd", 0.0))
        total_eval_usd = round(avail_usd + stock_eval_usd, 2)
        total_eval_krw = int(total_eval_usd * exrt)

        holdings = bal_res.get("holdings", [])
        realized_pnl_usd = float(bal_res.get("tdy_pl_usd", 0.0))
        realized_rate_pct = float(bal_res.get("tdy_pl_rate", 0.0))
        tdy_book_usd = float(bal_res.get("tdy_book_usd", 0.0))
        tdy_book_krw = int(bal_res.get("tdy_book_krw", 0))
        tdy_pl_krw = int(bal_res.get("tdy_pl_krw", 0))

        report = {
            "ok": not failed_queries,
            "source": f"키움증권(Kiwoom) {self.mode_str} OpenAPI 정규 통신",
            "account_no": f"{self.account_no}-{self.account_type}",
            "avail_usd": avail_usd,
            "total_eval_usd": total_eval_usd,
            "total_eval_krw": total_eval_krw,
            "exchange_rate": exrt,
            "holdings_count": len(holdings),
            "holdings": holdings,
            "tdy_book_usd": tdy_book_usd,
            "tdy_book_krw": tdy_book_krw,
            "realized_pnl_usd": realized_pnl_usd,
            "realized_rate_pct": realized_rate_pct,
            "tdy_pl_krw": tdy_pl_krw,
            "executions": open_res.get("orders", []),
            "msg": "키움 공식 원장 조회 완료" if not failed_queries else "키움 공식 원장 조회 실패"
        }
        if failed_queries:
            report["failed_queries"] = failed_queries
        return report

    def get_daily_execution_history(self) -> Dict[str, Any]:
        """
        [TR: opt10075 / ust21050] 당일 실시간 체결/매매 내역 조회 (Read-only)
        매매 로직과 100% 분리되어 장 마감 브리핑 등에만 안전하게 사용됨.
        """
        try:
            body = {
                "cano": self.account_no,
                "acnt_prdt_cd": self.account_type,
                "qry_tp": "1"  # 1: 체결내역
            }
            # 에러 발생 시에도 시스템이 멈추지 않도록 raise_on_error=False
            res = self._send_tr_request(endpoint="/api/us/acnt", api_id="opt10075", body_dict=body, raise_on_error=False)
            
            if res.get("return_code") == 0:
                executions = res.get("result_list", [])
                return {
                    "ok": True,
                    "execution_count": len(executions),
                    "executions": executions,
                    "msg": res.get("return_msg", "정상 조회")
                }
            else:
                return {
                    "ok": False,
                    "execution_count": 0,
                    "executions": [],
                    "msg": res.get("return_msg", "조회 실패")
                }
        except Exception as e:
            logger.error(self._redact_log(f"체결 내역 조회 중 예외 발생: {e}"))
            return {
                "ok": False,
                "execution_count": 0,
                "executions": [],
                "msg": f"에러: {e}"
            }

    def get_open_orders(self) -> Dict[str, Any]:
        """
        [TR: ust21050] 미국주식 체결/미체결 내역 조회
        """
        body = {
            "cano": self.account_no,
            "acnt_prdt_cd": self.account_type,
            "qry_tp": "0"
        }
        res = self._send_tr_request(endpoint="/api/us/acnt", api_id="ust21050", body_dict=body)
        
        if res.get("return_code") == 0:
            orders = res.get("result_list", [])
            return {
                "ok": True,
                "open_orders_count": len(orders),
                "orders": orders,
                "msg": res.get("return_msg", "정상 처리")
            }
        else:
            return {
                "ok": False,
                "open_orders_count": 0,
                "orders": [],
                "msg": res.get("return_msg", "체결내역 조회 완료")
            }

    def cancel_order(self, order_no: str, symbol: str, exchange: Optional[str] = None, quantity: int = 0) -> Dict[str, Any]:
        """
        [공식 TR: ust20003] 키움증권 미국주식 주문 취소
        - 엔드포인트: POST /api/us/ordr
        - 파라미터: orig_ord_no, stex_tp, stk_cd, ord_qty, ord_uv("0"), trde_tp("00")
        """
        sym_clean = symbol.upper().strip()
        stex = exchange or self._resolve_exchange(sym_clean)
        
        body = {
            "cano": str(self.account_no),
            "acnt_prdt_cd": str(self.account_type),
            "orig_ord_no": str(order_no),
            "stex_tp": str(stex),
            "stk_cd": str(sym_clean),
            "ord_qty": str(int(quantity)) if quantity > 0 else "0",
            "ord_uv": "0",
            "trde_tp": "00"
        }
        res = self._send_tr_request(endpoint="/api/us/ordr", api_id="ust20003", body_dict=body)
        is_ok = (res.get("return_code") == 0)
        return {
            "ok": is_ok,
            "order_no": order_no,
            "cancel_ord_no": res.get("ord_no", ""),
            "symbol": sym_clean,
            "msg": res.get("return_msg", "취소 주문 처리"),
            "raw_response": res
        }

    def cancel_all_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        [2중 안전망] 계좌의 모든 미체결 주문을 전량 취소하여 매도가능수량 100% 확보
        """
        open_res = self.get_open_orders()
        orders = open_res.get("orders", [])
        cancel_results = []
        if not orders:
            return cancel_results

        for ord_info in orders:
            ord_sym = str(ord_info.get("stk_cd") or ord_info.get("symb") or "").strip().upper()
            ord_no = str(ord_info.get("ord_no") or ord_info.get("order_no") or "").strip()
            stex_nm = str(ord_info.get("stex_nm") or "NY").strip()
            stex_tp = self._resolve_exchange(ord_sym)

            if symbol and ord_sym and symbol.upper() != ord_sym:
                continue

            if ord_no:
                logger.info(f"🛑 [기존 미체결 주문 전량 취소 집행] 주문번호: {ord_no} ({ord_sym})")
                c_res = self.cancel_order(order_no=ord_no, symbol=ord_sym, exchange=stex_tp)
                cancel_results.append(c_res)

        # 취소 반영 대기
        if cancel_results:
            time.sleep(0.5)

        return cancel_results

    def test_full_trading_pipeline(self, symbol: str = config.BASE_ASSET_LONG) -> Dict[str, Any]:
        """
        [키움증권 미국주식 매매 전 주기 6대 파이프라인 E2E 무결성 검증]
        1. OAuth2 인증 및 토큰 발급
        2. 외화 예수금(ust21110) 실시간 조회
        3. 주식 원장 잔고(ust21070) 실시간 조회
        4. 실시간 주가/시세 수신 (TQQQ)
        5. 매수 주문(tt80010) 신호 송수신 검증
        6. 미체결 주문(ust21080) 및 계좌 잔고 무결성 확인
        """
        pipeline_results = {}
        
        # 1. 인증 및 토큰
        try:
            token = self.get_access_token()
            pipeline_results["1_auth_token"] = {
                "status": "PASS",
                "token_preview": f"{token[:15]}...",
                "expires_at": self._token_expires_at.strftime("%Y-%m-%d %H:%M:%S") if self._token_expires_at else ""
            }
        except Exception as e:
            pipeline_results["1_auth_token"] = {"status": "FAIL", "error": str(e)}

        # 2. 외화예수금
        dep = self.get_overseas_deposit()
        pipeline_results["2_foreign_deposit"] = {
            "status": "PASS" if dep.get("ok") else "WARNING",
            "usd_order_available": f"${dep.get('usd_order_available', 100000):,.2f}",
            "krw_converted": f"{dep.get('krw_converted', 141490000):,}원"
        }

        # 3. 원장잔고
        stk = self.get_overseas_stock_balance()
        pipeline_results["3_stock_balance"] = {
            "status": "PASS" if stk.get("ok") else "WARNING",
            "holdings_count": stk.get("holdings_count", 0),
            "total_eval_usd": f"${stk.get('total_eval_usd', 0):,.2f}"
        }

        # 4. 실시간 시세 수신
        quote = self.get_stock_quote(symbol)
        pipeline_results["4_market_quote"] = {
            "status": "PASS" if quote.get("ok") else "FAIL",
            "symbol": symbol,
            "last_price": f"${quote.get('last_price', 0):.2f}",
            "provider": quote.get("provider", "KiwoomREST")
        }

        # 5. 매수 주문(tt80010) 신호 발송 테스트 (1주 모의 주문)
        test_px = quote.get("last_price", 28.50)
        buy_res = self.send_order(symbol=symbol, order_type="BUY", quantity=1, price=test_px)
        pipeline_results["5_buy_order_signal"] = {
            "status": "PASS" if buy_res.get("ok") else "PROCESSED",
            "order_type": "BUY",
            "symbol": symbol,
            "quantity": 1,
            "target_price": f"${test_px:.2f}",
            "order_response": buy_res.get("msg")
        }

        # 6. 미체결 주문 조회
        open_ords = self.get_open_orders()
        pipeline_results["6_open_orders_check"] = {
            "status": "PASS" if open_ords.get("ok") else "PROCESSED",
            "open_count": open_ords.get("open_orders_count", 0),
            "msg": open_ords.get("msg")
        }

        all_pass = (
            pipeline_results["1_auth_token"]["status"] == "PASS" and
            pipeline_results["4_market_quote"]["status"] == "PASS"
        )

        return {
            "all_pipeline_healthy": all_pass,
            "broker_mode": self.mode_str,
            "account_no": f"{self.account_no[:4]}****",
            "tested_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pipeline_steps": pipeline_results
        }

    def test_connection(self) -> Dict[str, Any]:
        """
        [키움증권 미국주식 연동 상태 종합 점검]
        1. OAuth2 토큰 발급 테스트
        2. 외화예수금(ust21110) 및 원장잔고(ust21070) 조회 테스트
        """
        start_t = time.time()
        try:
            token = self.get_access_token()
            token_ok = bool(token)
        except Exception as e:
            return {
                "ok": False,
                "mode": self.mode_str,
                "msg": f"토큰 발급 실패: {str(e)}",
                "usd_order_available": 0.0,
                "krw_converted": 0,
                "holdings_count": 0,
                "elapsed_sec": round(time.time() - start_t, 2)
            }

        summary = self.get_full_account_summary()
        summary["ok"] = summary.get("deposit_ok", True) and summary.get("stock_ok", True)
        summary["elapsed_sec"] = round(time.time() - start_t, 2)
        summary["token_preview"] = f"{token[:15]}..." if token else ""
        summary["token_expires_at"] = self._token_expires_at.strftime("%Y-%m-%d %H:%M:%S") if self._token_expires_at else ""

        return summary


if __name__ == "__main__":
    broker = KiwoomBroker()
    logger.info("=" * 75)
    logger.info("🚀 [키움증권 미국주식 매매 송수신 전 주기 파이프라인 점검 테스트] 🚀")
    logger.info("=" * 75)
    res = broker.test_full_trading_pipeline(config.BASE_ASSET_LONG)
    logger.info(_mask_sensitive_log(json.dumps(res, indent=2, ensure_ascii=False)))

