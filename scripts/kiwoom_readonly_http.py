"""Dormant stdlib HTTPS executor and sealed memory-only authentication fixtures.

The real HTTPSConnection factory is present but its stage gate cannot be opened
by a runtime flag. Only FakeHTTPSession can execute in this approval stage.
No environment, dotenv file, token cache, broker or operational file is read.
"""

import http.client
import json
import socket
import ssl
import os
from datetime import datetime, timezone

try:
    from .kiwoom_readonly_transport import (ENDPOINT, HOSTS, InjectedCredentials,
        ReadOnlyTransportError, _boundary, _plain)
except ImportError:
    from kiwoom_readonly_transport import (ENDPOINT, HOSTS, InjectedCredentials,
        ReadOnlyTransportError, _boundary, _plain)

MAX_RESPONSE_BYTES = 1048576


def _credential_access_gate():
    raise ReadOnlyTransportError("OPERATIONAL_CREDENTIAL_ACCESS_NOT_APPROVED")


def _credential_values(values, environment, account):
    if type(environment) is not str or environment not in HOSTS:
        raise ReadOnlyTransportError("INVALID_ENVIRONMENT")
    if type(account) is not str or not account or any(ord(c) < 33 for c in account):
        raise ReadOnlyTransportError("INVALID_CREDENTIAL_ACCOUNT")
    names = ExistingCredentialSource.fields(environment)
    if type(values) is not dict:
        raise ReadOnlyTransportError("INVALID_CREDENTIAL_SOURCE")
    parts = tuple(values.get(name) for name in names)
    if any(type(part) is not str or not part or any(ord(c) < 33 or ord(c) > 126 for c in part) for part in parts):
        raise ReadOnlyTransportError("INVALID_CREDENTIAL_SOURCE")
    if parts[2] != account:
        raise ReadOnlyTransportError("CREDENTIAL_ACCOUNT_MISMATCH")
    return parts


class FakeEnvironmentSource:
    """Supplied memory snapshot only. It does not read process environment."""
    __slots__ = ("_values",)

    def __init__(self, values):
        _plain(values)
        if type(values) is not dict:
            raise ReadOnlyTransportError("INVALID_CREDENTIAL_SOURCE")
        self._values = dict(values)

    def __repr__(self):
        return "FakeEnvironmentSource([REDACTED])"

    def load(self, environment, account):
        return _credential_values(self._values, environment, account)


def _oauth_boundary(environment, method, path, api_id, body, timeout):
    # Independent authentication boundary. Query TRs can never reach this path.
    if type(environment) is not str or environment not in HOSTS:
        raise ReadOnlyTransportError("INVALID_ENVIRONMENT")
    if method != "POST" or path != "/oauth2/token" or api_id != "au10001":
        raise ReadOnlyTransportError("AUTH_ENDPOINT_NOT_ALLOWED")
    if type(body) is not dict or set(body) != {"grant_type", "appkey", "secretkey"} or body.get("grant_type") != "client_credentials":
        raise ReadOnlyTransportError("INVALID_AUTH_BODY")
    if any(type(item) is not str or not item or any(ord(c) < 33 or ord(c) > 126 for c in item) for item in body.values()):
        raise ReadOnlyTransportError("INVALID_AUTH_BODY")
    # Reuse query timeout validation, not query execution or endpoint access.
    _boundary("https://" + HOSTS[environment] + ENDPOINT, "POST", "ust21050",
              environment, {"slby_tp": "0"}, ("N", ""), timeout)


class OAuthCredentialProvider:
    """Separate au10001 issuer; no account binding or history proof is implied.

    This approval stage accepts only a sealed memory source/session. The real
    credential and network branches exist but are blocked before side effects.
    Official expires_dt has no specified timezone. Without explicit UTC expiry
    supplied by the Fake fixture, a token is not eligible for query execution.
    No persistent cache, token refresh retry or authentication CLI exists.
    """
    __slots__ = ("_source", "_session", "_credentials", "_environment", "_account",
                 "_expiry_utc", "_fixture_expiry", "_raw_expiry", "_secrets", "timeout", "_scope", "_issued_material")

    def __init__(self, source, session=None, *, environment="SIMULATION", account="MOCK_SCOPE",
                 timeout=10, fixture_expiry_utc=None):
        if type(source) not in (FakeEnvironmentSource, ExistingCredentialSource):
            raise ReadOnlyTransportError("INVALID_CREDENTIAL_SOURCE")
        if session is not None and type(session) is not FakeHTTPSession:
            raise ReadOnlyTransportError("FAKE_HTTPS_SESSION_REQUIRED")
        if fixture_expiry_utc is not None and (type(source) is not FakeEnvironmentSource or type(session) is not FakeHTTPSession
                or type(fixture_expiry_utc) is not datetime or fixture_expiry_utc.tzinfo is not timezone.utc):
            raise ReadOnlyTransportError("INVALID_FIXTURE_EXPIRY")
        self._source, self._session = source, session
        self._environment, self._account, self.timeout = environment, account, timeout
        self._scope = (environment, account, timeout, source, session, fixture_expiry_utc)
        self._issued_material = None
        self._credentials, self._expiry_utc, self._raw_expiry = None, None, None
        self._fixture_expiry, self._secrets = fixture_expiry_utc, ()
        # Validate scope/timeout now without loading any credentials.
        if type(environment) is not str or environment not in HOSTS:
            raise ReadOnlyTransportError("INVALID_ENVIRONMENT")
        if type(account) is not str or not account or any(ord(c) < 33 for c in account):
            raise ReadOnlyTransportError("INVALID_CREDENTIAL_ACCOUNT")
        _boundary("https://" + HOSTS[environment] + ENDPOINT, "POST", "ust21050", environment,
                  {"slby_tp": "0"}, ("N", ""), timeout)

    def __repr__(self):
        return "OAuthCredentialProvider([REDACTED])"

    def _issue(self):
        snapshot = (self._environment, self._account, self.timeout, self._source, self._session, self._fixture_expiry)
        if snapshot != self._scope:
            raise ReadOnlyTransportError("AUTH_CONFIGURATION_CHANGED")
        if type(self._source) not in (FakeEnvironmentSource, ExistingCredentialSource):
            raise ReadOnlyTransportError("INVALID_CREDENTIAL_SOURCE")
        appkey, secretkey, account = type(self._source).load(self._source, self._environment, self._account)
        self._secrets = (appkey, secretkey, account)
        body = {"grant_type": "client_credentials", "appkey": appkey, "secretkey": secretkey}
        _oauth_boundary(self._environment, "POST", "/oauth2/token", "au10001", body, self.timeout)
        if self._session is not None and type(self._session) is not FakeHTTPSession:
            raise ReadOnlyTransportError("FAKE_HTTPS_SESSION_REQUIRED")
        context = ssl.create_default_context()
        if context.verify_mode != ssl.CERT_REQUIRED or not context.check_hostname:
            raise ReadOnlyTransportError("TLS_VERIFICATION_REQUIRED")
        conn = None
        try:
            conn = (_network_connection(HOSTS[self._environment], self.timeout, context) if self._session is None else
                    FakeHTTPSession.connect(self._session, HOSTS[self._environment], self.timeout, context))
            # Connecting must not change the token's intended account/environment.
            # Compare to the pre-connect snapshot, not to a freshly mutated host.
            if snapshot != (self._environment, self._account, self.timeout, self._source, self._session, self._fixture_expiry):
                raise ReadOnlyTransportError("AUTH_CONFIGURATION_CHANGED")
            if type(self._source).load(self._source, snapshot[0], snapshot[1]) != (appkey, secretkey, account):
                raise ReadOnlyTransportError("AUTH_CONFIGURATION_CHANGED")
            _oauth_boundary(snapshot[0], "POST", "/oauth2/token", "au10001", body, snapshot[2])
            # No existing bearer is invented. The spec's generic auth header is
            # inconsistent with initial-issuance examples; live contract is NV.
            conn.request("POST", "/oauth2/token", json.dumps(body).encode("utf-8"),
                         {"api-id": "au10001", "Content-Type": "application/json;charset=UTF-8"})
            response = conn.getresponse()
            status = response.status
            if type(status) is not int or status != 200:
                code = "AUTH_REDIRECT_REJECTED" if type(status) is int and 300 <= status < 400 else "AUTHENTICATION_FAILED"
                raise ReadOnlyTransportError(code)
            pairs = response.getheaders()
            if type(pairs) is not list or any(type(p) not in (tuple, list) or len(p) != 2 or any(type(v) is not str for v in p) for p in pairs):
                raise ReadOnlyTransportError("INVALID_AUTH_HEADERS")
            if len({key.lower() for key, _ in pairs}) != len(pairs):
                raise ReadOnlyTransportError("INVALID_AUTH_HEADERS")
            headers = {key.lower(): value for key, value in pairs}
            if headers.get("content-encoding", "identity").lower() != "identity":
                raise ReadOnlyTransportError("INVALID_AUTH_ENCODING")
            declared = headers.get("content-length")
            if declared is not None and (not declared.isascii() or not declared.isdecimal() or int(declared) > MAX_RESPONSE_BYTES):
                raise ReadOnlyTransportError("INVALID_AUTH_LENGTH")
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if type(raw) is not bytes or len(raw) > MAX_RESPONSE_BYTES:
                raise ReadOnlyTransportError("AUTH_RESPONSE_SIZE_EXCEEDED")
            if declared is not None and len(raw) != int(declared):
                raise ReadOnlyTransportError("PARTIAL_AUTH_RESPONSE")
            try:
                result = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeError, RecursionError):
                raise ReadOnlyTransportError("INVALID_AUTH_RESPONSE") from None
            if type(result) is not dict or type(result.get("return_code")) not in (str, int) or str(result["return_code"]) != "0":
                raise ReadOnlyTransportError("AUTHENTICATION_FAILED")
            if type(result.get("token_type")) is not str or result["token_type"].lower() != "bearer":
                raise ReadOnlyTransportError("INVALID_AUTH_TOKEN_TYPE")
            token = InjectedCredentials(result.get("token"))
            raw_expiry = result.get("expires_dt")
            if type(raw_expiry) is not str or len(raw_expiry) != 14 or not raw_expiry.isascii() or not raw_expiry.isdecimal():
                raise ReadOnlyTransportError("INVALID_AUTH_EXPIRY")
            try:
                datetime.strptime(raw_expiry, "%Y%m%d%H%M%S")
            except ValueError:
                raise ReadOnlyTransportError("INVALID_AUTH_EXPIRY") from None
            if snapshot != (self._environment, self._account, self.timeout, self._source, self._session, self._fixture_expiry):
                raise ReadOnlyTransportError("AUTH_CONFIGURATION_CHANGED")
            if type(self._source).load(self._source, snapshot[0], snapshot[1]) != (appkey, secretkey, account):
                raise ReadOnlyTransportError("AUTH_CONFIGURATION_CHANGED")
            self._raw_expiry, self._credentials = raw_expiry, token
            self._issued_material = (appkey, secretkey, account)
            self._secrets += (token._token,)
            self._expiry_utc = self._fixture_expiry
        except ReadOnlyTransportError:
            raise
        except (socket.timeout, TimeoutError):
            raise ReadOnlyTransportError("AUTHENTICATION_TIMEOUT") from None
        except Exception:
            raise ReadOnlyTransportError("AUTHENTICATION_EXECUTION_FAILED") from None
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def load(self, environment, account):
        if (self._environment, self._account, self.timeout, self._source, self._session, self._fixture_expiry) != self._scope:
            raise ReadOnlyTransportError("AUTH_CONFIGURATION_CHANGED")
        if environment != self._environment:
            raise ReadOnlyTransportError("CREDENTIAL_ENVIRONMENT_MISMATCH")
        if account != self._account:
            raise ReadOnlyTransportError("CREDENTIAL_ACCOUNT_MISMATCH")
        if self._credentials is None:
            self._issue()
        if (self._environment, self._account, self.timeout, self._source, self._session, self._fixture_expiry) != self._scope:
            raise ReadOnlyTransportError("AUTH_CONFIGURATION_CHANGED")
        if type(self._source).load(self._source, environment, account) != self._issued_material:
            raise ReadOnlyTransportError("AUTH_CONFIGURATION_CHANGED")
        if self._expiry_utc is None:
            raise ReadOnlyTransportError("AUTH_EXPIRY_TIMEZONE_NOT_VERIFIED")
        if datetime.now(timezone.utc) >= self._expiry_utc:
            raise ReadOnlyTransportError("AUTHENTICATION_EXPIRED")
        return self._credentials

    def secret_values(self):
        return self._secrets


class ExistingCredentialSource:
    """Static description of the project's existing environment/RAM source.

    OAuth au10001 is POST /oauth2/token with grant_type=client_credentials,
    appkey, secretkey; response token/token_type/expires_dt. Authentication is
    deliberately not sent by this query-only tool. expires_dt has no documented
    timezone; a future approved provider must supply an unambiguous UTC expiry.
    """

    @staticmethod
    def fields(environment):
        if environment not in HOSTS:
            raise ReadOnlyTransportError("INVALID_ENVIRONMENT")
        prefix = "KIWOOM_REAL_" if environment == "REAL" else "KIWOOM_MOCK_"
        return (prefix + "APP_KEY", prefix + "APP_SECRET", prefix + "ACCOUNT_NO")

    def load(self, environment, account):
        # This separate stage gate runs BEFORE even consulting os.environ.
        _credential_access_gate()
        names = self.fields(environment)
        values = {name: os.environ.get(name, "").strip() for name in names}
        return _credential_values(values, environment, account)


class FakeCredentialProvider:
    """Sealed in-memory token/scope/expiry snapshot; scope is not broker proof."""
    __slots__ = ("_credentials", "_account", "_environment", "_expires_at", "_failure")

    def __init__(self, *, token="SYNTHETIC_MOCK_TOKEN", account="MOCK_SCOPE",
                 environment="SIMULATION", expires_at=None, failure=False):
        self._credentials = InjectedCredentials(token)
        if type(account) is not str or not account or any(ord(c) < 32 for c in account):
            raise ReadOnlyTransportError("INVALID_CREDENTIAL_ACCOUNT")
        if type(environment) is not str or environment not in HOSTS:
            raise ReadOnlyTransportError("INVALID_CREDENTIAL_ENVIRONMENT")
        if expires_at is None:
            expires_at = datetime.max.replace(tzinfo=timezone.utc)
        if type(expires_at) is not datetime or expires_at.tzinfo is not timezone.utc:
            raise ReadOnlyTransportError("INVALID_CREDENTIAL_EXPIRY")
        if type(failure) is not bool:
            raise ReadOnlyTransportError("INVALID_AUTH_FAILURE_FLAG")
        self._account, self._environment = account, environment
        self._expires_at, self._failure = expires_at, failure

    def __repr__(self):
        return "FakeCredentialProvider([REDACTED])"

    def load(self, environment, account):
        if self._failure:
            raise ReadOnlyTransportError("AUTHENTICATION_FAILED")
        if environment != self._environment:
            raise ReadOnlyTransportError("CREDENTIAL_ENVIRONMENT_MISMATCH")
        if account != self._account:
            raise ReadOnlyTransportError("CREDENTIAL_ACCOUNT_MISMATCH")
        if datetime.now(timezone.utc) >= self._expires_at:
            raise ReadOnlyTransportError("AUTHENTICATION_EXPIRED")
        return self._credentials


class _FakeResponse:
    def __init__(self, fixture):
        self.status = fixture.get("http_status", 200)
        self._headers = fixture.get("headers", {})
        payload = fixture.get("body", {})
        self._data = (payload if type(payload) is str else json.dumps(payload)).encode("utf-8")

    def getheaders(self):
        if type(self._headers) is dict:
            return list(self._headers.items())
        return self._headers

    def read(self, count):
        return self._data[:count]


class _FakeConnection:
    def __init__(self, session, fixture, timeout):
        self._session, self._fixture, self.timeout = session, fixture, timeout

    def request(self, method, path, body, headers):
        # OAuth input material is never retained, even by fixture request logs.
        self._session.requests.append({"method": method, "path": path,
            "body": "[REDACTED]" if path == "/oauth2/token" else body.decode("utf-8"),
            "headers": {**headers, "authorization": "[REDACTED]"}})
        if self._fixture.get("timeout") is True:
            raise socket.timeout()

    def getresponse(self):
        return _FakeResponse(self._fixture)

    def close(self):
        self._session.closed += 1


class FakeHTTPSession:
    """JSON-only fixture connections, no caller-provided executable callbacks."""
    __slots__ = ("_responses", "connections", "requests", "closed")

    def __init__(self, responses):
        _plain(responses)
        if type(responses) is not list or any(type(x) is not dict for x in responses):
            raise ReadOnlyTransportError("INVALID_MOCK_RESPONSES")
        self._responses = json.loads(json.dumps(responses))
        self.connections, self.requests, self.closed = [], [], 0

    def connect(self, host, timeout, context):
        if context.verify_mode != ssl.CERT_REQUIRED or not context.check_hostname:
            raise ReadOnlyTransportError("TLS_VERIFICATION_REQUIRED")
        index = len(self.connections)
        self.connections.append({"host": host, "timeout": timeout, "tls_verified": True})
        fixture = self._responses[index] if index < len(self._responses) else {"http_status": 599}
        return _FakeConnection(self, fixture, timeout)


def _network_connection(host, timeout, context):
    # No token, flag, constructor argument or CLI option opens this gate.
    raise ReadOnlyTransportError("REAL_NETWORK_NOT_APPROVED")
    # Future enablement needs a separate approved code change. HTTPSConnection
    # ignores HTTP(S)_PROXY/ALL_PROXY and does not follow redirects.
    return http.client.HTTPSConnection(host, timeout=timeout, context=context)


class ReadOnlyHTTPExchange:
    """Real HTTP protocol execution over a sealed Fake session at this stage."""
    __slots__ = ("_session", "_provider", "account", "environment", "max_response_bytes", "requests")

    def __init__(self, session=None, *, provider, account="MOCK_SCOPE",
                 environment="SIMULATION", max_response_bytes=MAX_RESPONSE_BYTES):
        if session is not None and type(session) is not FakeHTTPSession:
            raise ReadOnlyTransportError("FAKE_HTTPS_SESSION_REQUIRED")
        if type(provider) not in (FakeCredentialProvider, OAuthCredentialProvider):
            raise ReadOnlyTransportError("FAKE_CREDENTIAL_PROVIDER_REQUIRED")
        if type(max_response_bytes) is not int or not 1 <= max_response_bytes <= MAX_RESPONSE_BYTES:
            raise ReadOnlyTransportError("INVALID_RESPONSE_SIZE_LIMIT")
        self._session, self._provider = session, provider
        self.account, self.environment = account, environment
        self.max_response_bytes, self.requests = max_response_bytes, []
        type(provider).load(provider, environment, account)

    def credentials(self):
        if type(self._provider) not in (FakeCredentialProvider, OAuthCredentialProvider):
            raise ReadOnlyTransportError("FAKE_CREDENTIAL_PROVIDER_REQUIRED")
        return type(self._provider).load(self._provider, self.environment, self.account)

    def secret_values(self):
        credentials = self.credentials()
        if type(self._provider) is OAuthCredentialProvider:
            secrets = OAuthCredentialProvider.secret_values(self._provider)
            return (secrets[0], secrets[1], credentials._token)
        # Account is internal scope evidence, redacted at the final report boundary.
        return (credentials._token,)

    def _send(self, *, url, method, api_id, environment, body, continuation,
              timeout, credentials, wire):
        _boundary(url, method, api_id, environment, body, continuation, timeout)
        if environment != self.environment:
            raise ReadOnlyTransportError("CREDENTIAL_ENVIRONMENT_MISMATCH")
        scoped_credentials = self.credentials()
        if credentials is not scoped_credentials:
            raise ReadOnlyTransportError("CREDENTIAL_SCOPE_MISMATCH")
        expected_headers = {"api-id": api_id, "cont-yn": continuation[0], "next-key": continuation[1],
            "authorization": "Bearer " + credentials._token, "Content-Type": "application/json;charset=UTF-8"}
        if type(wire) is not dict or wire.get("url") != url or wire.get("method") != method or wire.get("headers") != expected_headers or wire.get("timeout") != timeout or wire.get("allow_redirects") is not False:
            raise ReadOnlyTransportError("INVALID_SERIALIZED_REQUEST")
        try:
            if type(wire.get("body")) is not str or json.loads(wire["body"]) != body:
                raise ReadOnlyTransportError("INVALID_SERIALIZED_REQUEST")
        except (ValueError, RecursionError):
            raise ReadOnlyTransportError("INVALID_SERIALIZED_REQUEST") from None
        if type(self.max_response_bytes) is not int or not 1 <= self.max_response_bytes <= MAX_RESPONSE_BYTES:
            raise ReadOnlyTransportError("INVALID_RESPONSE_SIZE_LIMIT")
        if self._session is not None and type(self._session) is not FakeHTTPSession:
            raise ReadOnlyTransportError("FAKE_HTTPS_SESSION_REQUIRED")
        context = ssl.create_default_context()
        if context.verify_mode != ssl.CERT_REQUIRED or not context.check_hostname:
            raise ReadOnlyTransportError("TLS_VERIFICATION_REQUIRED")
        conn = None
        try:
            # Direct stdlib factory never consults environment proxy settings.
            conn = (_network_connection(HOSTS[environment], timeout, context) if self._session is None
                else FakeHTTPSession.connect(self._session, HOSTS[environment], timeout, context))
            _boundary(url, method, api_id, environment, body, continuation, timeout)
            self.credentials()  # Re-check expiry/scope immediately before wire send.
            conn.request("POST", ENDPOINT, wire["body"].encode("utf-8"), expected_headers)
            self.requests.append({"tr": api_id, "environment": environment, "authorization": "[REDACTED]"})
            response = conn.getresponse()
            status = response.status
            if type(status) is not int or not 100 <= status <= 599:
                return self._error("INVALID_HTTP_STATUS")
            if 300 <= status < 400:
                return self._error("REDIRECT_REJECTED", status)
            if status in (401, 403):
                return self._error("AUTHENTICATION_REJECTED", status)
            error_status = status if status >= 400 else 599
            pairs = response.getheaders()
            if type(pairs) is not list or any(type(pair) not in (tuple, list) or len(pair) != 2 or type(pair[0]) is not str or type(pair[1]) is not str for pair in pairs):
                return self._error("INVALID_RESPONSE_HEADERS", error_status)
            if len({key.lower() for key, value in pairs}) != len(pairs):
                return self._error("DUPLICATE_RESPONSE_HEADERS", error_status)
            headers = {key.lower(): value for key, value in pairs}
            declared = headers.get("content-length")
            if declared is not None:
                if not declared.isascii() or not declared.isdecimal():
                    return self._error("INVALID_CONTENT_LENGTH", error_status)
                declared = int(declared)
                if declared > self.max_response_bytes:
                    return self._error("RESPONSE_SIZE_EXCEEDED", error_status)
            if headers.get("content-encoding", "identity").lower() != "identity":
                return self._error("UNSUPPORTED_CONTENT_ENCODING", error_status)
            raw = response.read(self.max_response_bytes + 1)
            if type(raw) is not bytes or len(raw) > self.max_response_bytes:
                return self._error("RESPONSE_SIZE_EXCEEDED", error_status)
            if declared is not None and len(raw) != declared:
                return self._error("PARTIAL_HTTP_RESPONSE", error_status)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeError, ValueError, RecursionError):
                return self._error("INVALID_RESPONSE_JSON", error_status)
            if type(payload) is not dict:
                return self._error("INVALID_RESPONSE_BODY", error_status)
            return {"http_status": status, "body": payload, "headers": headers}
        except (socket.timeout, TimeoutError):
            return self._error("HTTP_TIMEOUT")
        except ReadOnlyTransportError:
            raise
        except Exception:
            return self._error("HTTP_EXECUTION_FAILED")
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass  # Do not replace the fixed original error with secret-bearing text.

    @staticmethod
    def _error(code, status=599):
        return {"http_status": status, "body": {}, "headers": {}, "transport_error": code}
