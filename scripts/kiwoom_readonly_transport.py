"""Serialization boundary for four US query TRs; real execution stays gated.

Sealed fixture replay or the dormant HTTP executor with a sealed Fake session
are supported. No runtime flag enables real networking or credential loading.
"""

import json
import math
from urllib.parse import urlsplit

ENDPOINT = "/api/us/acnt"
HOSTS = {"REAL": "api.kiwoom.com", "SIMULATION": "mockapi.kiwoom.com"}
FIELDS = {
    "ust21050": frozenset(("ord_dt", "slby_tp", "stex_tp", "stk_cd")),
    "ust21150": frozenset(("ord_dt", "query_tp", "slby_tp", "stex_tp", "stk_cd", "oppo_trde_tp", "fr_ord_no")),
    "ust21180": frozenset(("strt_dt", "end_dt", "slby_tp", "stex_tp", "stk_cd", "oppo_trde_tp")),
    "ust21510": frozenset(("slby_tp", "stex_tp", "stk_cd")),
}
MAX_PAGES = 100
MAX_RETRIES = 2


class ReadOnlyTransportError(ValueError):
    """Messages are fixed codes and never echo input or server exceptions."""


def _plain(value):
    if type(value) in (str, int, bool, type(None)):
        return
    if type(value) is list:
        for item in value:
            _plain(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _plain(item)
        return
    raise ReadOnlyTransportError("INVALID_MOCK_VALUE")


class InjectedCredentials:
    """In-memory credential interface only; no authentication or loading.

    Values are used only inside sealed FakeHTTPExchange and are not logged.
    The constructor does not establish account binding or credential validity.
    """
    __slots__ = ("_token",)

    def __init__(self, token="SYNTHETIC_MOCK_TOKEN"):
        if type(token) is not str or not token or any(ord(c) < 33 or ord(c) > 126 for c in token):
            raise ReadOnlyTransportError("INVALID_INJECTED_CREDENTIAL")
        self._token = token

    def __repr__(self):
        return "InjectedCredentials([REDACTED])"


def _boundary(url, method, api_id, environment, body, continuation, timeout):
    if type(environment) is not str or environment not in HOSTS:
        raise ReadOnlyTransportError("INVALID_ENVIRONMENT")
    if type(url) is not str or url != "https://" + HOSTS[environment] + ENDPOINT:
        raise ReadOnlyTransportError("URL_NOT_ALLOWED")
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.netloc != HOSTS[environment] or parsed.path != ENDPOINT or parsed.query or parsed.fragment:
        raise ReadOnlyTransportError("URL_NOT_ALLOWED")
    if method != "POST":
        raise ReadOnlyTransportError("METHOD_NOT_ALLOWED")
    if type(api_id) is not str or api_id not in FIELDS:
        raise ReadOnlyTransportError("TR_NOT_ALLOWED")
    if type(body) is not dict or any(type(k) is not str or k not in FIELDS[api_id] for k in body) or any(type(v) is not str for v in body.values()):
        raise ReadOnlyTransportError("INVALID_QUERY_BODY")
    if type(continuation) is not tuple or len(continuation) != 2 or continuation[0] not in ("N", "Y") or type(continuation[1]) is not str or len(continuation[1]) > 50 or any(ord(c) < 32 or ord(c) > 126 for c in continuation[1]):
        raise ReadOnlyTransportError("INVALID_CONTINUATION")
    if continuation[0] == "Y" and not continuation[1] or continuation[0] == "N" and continuation[1]:
        raise ReadOnlyTransportError("INVALID_CONTINUATION")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 60:
        raise ReadOnlyTransportError("INVALID_TIMEOUT")


class FakeHTTPExchange:
    """Sealed, JSON-only HTTP fixture replay. Requests are sanitized metadata.

    Each fixture has http_status/body/headers; timeout=True simulates expiry.
    Only counters, approved TR and continuation flag are retained in call logs.
    """
    __slots__ = ("_responses", "_wire_requests", "requests")

    def __init__(self, responses):
        _plain(responses)
        if type(responses) is not list:
            raise ReadOnlyTransportError("INVALID_MOCK_RESPONSES")
        self._responses = json.loads(json.dumps(responses))
        self._wire_requests = []
        self.requests = []

    def _send(self, *, url, method, api_id, environment, body, continuation, timeout, credentials, wire):
        # Checked again immediately before every simulated wire send, including retries.
        _boundary(url, method, api_id, environment, body, continuation, timeout)
        if type(credentials) is not InjectedCredentials:
            raise ReadOnlyTransportError("INJECTED_CREDENTIALS_REQUIRED")
        if type(credentials._token) is not str or not credentials._token or any(ord(c) < 33 or ord(c) > 126 for c in credentials._token):
            raise ReadOnlyTransportError("INVALID_INJECTED_CREDENTIAL")
        expected_headers = {"api-id": api_id, "cont-yn": continuation[0], "next-key": continuation[1],
                            "authorization": "Bearer " + credentials._token, "Content-Type": "application/json;charset=UTF-8"}
        if type(wire) is not dict or wire.get("url") != url or wire.get("method") != method or wire.get("headers") != expected_headers or wire.get("timeout") != timeout or wire.get("allow_redirects") is not False or type(wire.get("body")) is not str:
            raise ReadOnlyTransportError("INVALID_SERIALIZED_REQUEST")
        try:
            decoded = json.loads(wire["body"])
        except (ValueError, RecursionError):
            raise ReadOnlyTransportError("INVALID_SERIALIZED_REQUEST") from None
        if decoded != body:
            raise ReadOnlyTransportError("INVALID_SERIALIZED_REQUEST")
        index = len(self.requests)
        self.requests.append({"tr": api_id, "environment": environment, "method": "POST", "endpoint": ENDPOINT,
                              "cont-yn": continuation[0], "next-key": "[REDACTED]" if continuation[1] else "",
                              "authorization": "[REDACTED]", "timeout": timeout, "redirects": False})
        # Serialize actual approved request material only in transient memory.
        encoded_body, headers = wire["body"], wire["headers"]
        # Private memory capture allows cursor/serialization assertions. No token retained.
        self._wire_requests.append({"url": url, "body_json": encoded_body,
                                   "headers": {**headers, "authorization": "[REDACTED]"}})
        if index >= len(self._responses):
            return {"http_status": 599, "body": {}, "headers": {}}
        return json.loads(json.dumps(self._responses[index]))


class ReadOnlyTransport:
    """Bounded Mock transport; arbitrary callables and subclasses are rejected."""
    __slots__ = ("_exchange", "_credentials", "environment", "url", "timeout", "retries", "_page_count", "_pages", "requests")

    def __init__(self, exchange, *, environment="SIMULATION", credentials=None, timeout=10, retries=2, mode="mock"):
        if mode != "mock":
            raise ReadOnlyTransportError("REAL_API_NOT_IMPLEMENTED_OR_APPROVED")
        self._validate_exchange(exchange)
        if type(environment) is not str or environment not in HOSTS:
            raise ReadOnlyTransportError("INVALID_ENVIRONMENT")
        if type(retries) is not int or not 0 <= retries <= MAX_RETRIES:
            raise ReadOnlyTransportError("INVALID_RETRY_LIMIT")
        self._exchange = exchange
        scoped = self._scoped_credentials(exchange)
        self._credentials = (scoped if scoped is not None else InjectedCredentials()) if credentials is None else credentials
        if type(self._credentials) is not InjectedCredentials:
            raise ReadOnlyTransportError("INJECTED_CREDENTIALS_REQUIRED")
        self.environment = environment
        self.url = "https://" + HOSTS[environment] + ENDPOINT
        self.timeout, self.retries = timeout, retries
        _boundary(self.url, "POST", "ust21050", environment, {"slby_tp": "0"}, ("N", ""), timeout)
        self._page_count, self._pages, self.requests = 0, [], []

    @staticmethod
    def _validate_exchange(exchange):
        if type(exchange) is FakeHTTPExchange:
            return
        try:
            from .kiwoom_readonly_http import ReadOnlyHTTPExchange
        except ImportError:
            from kiwoom_readonly_http import ReadOnlyHTTPExchange
        if type(exchange) is not ReadOnlyHTTPExchange:
            raise ReadOnlyTransportError("FAKE_HTTP_EXCHANGE_REQUIRED")

    @staticmethod
    def _scoped_credentials(exchange):
        if type(exchange) is FakeHTTPExchange:
            return None
        ReadOnlyTransport._validate_exchange(exchange)
        return type(exchange).credentials(exchange)

    def request(self, api_id, body, continuation=("N", ""), endpoint=ENDPOINT, *, method="POST", url=None):
        if endpoint != ENDPOINT:
            raise ReadOnlyTransportError("PATH_NOT_ALLOWED")
        target = self.url if url is None else url
        _boundary(target, method, api_id, self.environment, body, continuation, self.timeout)
        self._validate_exchange(self._exchange)
        if type(self._credentials) is not InjectedCredentials or type(self.retries) is not int or not 0 <= self.retries <= MAX_RETRIES:
            raise ReadOnlyTransportError("INVALID_TRANSPORT_CONFIGURATION")
        if self._page_count >= MAX_PAGES:
            return {"http_status": 599, "body": {"return_code": -1}, "headers": {}, "transport_error": "PAGE_LIMIT_EXCEEDED"}
        self._page_count += 1
        self.requests.append({"tr": api_id, "page": self._page_count})
        response = None
        for attempt in range(self.retries + 1):
            if type(self._credentials._token) is not str or not self._credentials._token or any(ord(c) < 33 or ord(c) > 126 for c in self._credentials._token):
                raise ReadOnlyTransportError("INVALID_INJECTED_CREDENTIAL")
            wire = {"url": target, "method": method, "body": json.dumps(body), "timeout": self.timeout,
                    "allow_redirects": False, "headers": {"api-id": api_id, "cont-yn": continuation[0],
                    "next-key": continuation[1], "authorization": "Bearer " + self._credentials._token,
                    "Content-Type": "application/json;charset=UTF-8"}}
            response = type(self._exchange)._send(self._exchange, url=target, method=method, api_id=api_id,
                environment=self.environment, body=body, continuation=continuation, timeout=self.timeout, credentials=self._credentials, wire=wire)
            # All attempts are private redaction evidence, including transient 429.
            self._pages.append(response)
            if type(response) is dict and type(response.get("body")) is str:
                try:
                    decoded_evidence = json.loads(response["body"])
                except (ValueError, RecursionError):
                    pass  # Fixed parse failure below; never echo malformed server text.
                else:
                    self._pages.append(decoded_evidence)
            if type(response) is not dict:
                response = {"http_status": 599, "body": {}, "headers": {}, "transport_error": "INVALID_HTTP_RESPONSE"}
                break
            if response.get("timeout") is True:
                response = {"http_status": 599, "body": {}, "headers": {}, "transport_error": "HTTP_TIMEOUT"}
                break
            status = response.get("http_status")
            if type(status) is not int or not 100 <= status <= 599:
                response = {"http_status": 599, "body": {}, "headers": {}, "transport_error": "INVALID_HTTP_STATUS"}
                break
            response["http_status"] = status
            if 300 <= status < 400:
                response = {"http_status": status, "body": {}, "headers": {}, "transport_error": "REDIRECT_REJECTED"}
                break
            if status != 429 or attempt == self.retries:
                break
        if response.get("http_status") == 200:
            payload = response.get("body")
            if type(payload) is str:
                try:
                    payload = json.loads(payload)
                except (ValueError, RecursionError):
                    response = {"http_status": 599, "body": {}, "headers": {}, "transport_error": "INVALID_RESPONSE_JSON"}
                else:
                    response["body"] = payload
            if type(response.get("body")) is not dict:
                response = {"http_status": 599, "body": {}, "headers": {}, "transport_error": "INVALID_RESPONSE_BODY"}
            elif type(response.get("headers")) is not dict or any(type(k) is not str or type(v) is not str for k, v in response["headers"].items()):
                response = {"http_status": 599, "body": {}, "headers": {}, "transport_error": "INVALID_RESPONSE_HEADERS"}
            else:
                headers = response["headers"]
                if len({key.lower() for key in headers}) != len(headers):
                    response = {"http_status": 599, "body": {}, "headers": {}, "transport_error": "DUPLICATE_RESPONSE_HEADERS"}
                else:
                    response["headers"] = {key.lower(): value for key, value in headers.items()}
        # Scrub known credentials even when reflected into unlabelled fields.
        token = self._credentials._token
        # Authentication keys/account may be reflected into an unlabelled query
        # field. Never depend solely on response labels to protect known values.
        secrets = ((token,) if type(self._exchange) is FakeHTTPExchange else
                   type(self._exchange).secret_values(self._exchange))
        def scrub(value):
            if type(value) is dict:
                return {("[REDACTED]" if any(secret in key for secret in secrets) else key): scrub(item) for key, item in value.items()}
            if type(value) is list:
                return [scrub(item) for item in value]
            if type(value) is str:
                for secret in secrets:
                    value = value.replace(secret, "[REDACTED]")
                return value
            if type(value) is int and any(secret in str(value) for secret in secrets):
                return None
            return value
        response = scrub(response)
        return json.loads(json.dumps(response))
