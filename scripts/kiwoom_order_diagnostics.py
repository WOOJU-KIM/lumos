"""Isolated, memory-only Kiwoom order-contract diagnostics.

No broker/config imports, operational authentication, state writer or real mode
exists. Transport accepts already-in-memory fixtures or a gated HTTP executor. The
official reference is Kiwoom-REST-API commit 953e5dbff123f437ab4d11a78a95191a685eb51f.
Collection completion is never an assertion of ledger completeness or READY.
"""

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone

try:
    from .kiwoom_readonly_transport import ReadOnlyTransport, ReadOnlyTransportError
except ImportError:
    from kiwoom_readonly_transport import ReadOnlyTransport, ReadOnlyTransportError


SPEC_COMMIT = "953e5dbff123f437ab4d11a78a95191a685eb51f"
ENDPOINT = "/api/us/acnt"
MAX_PAGES = 100
QUERY_FIELDS = {
    "ust21050": frozenset(("ord_dt", "slby_tp", "stex_tp", "stk_cd")),
    "ust21150": frozenset(("ord_dt", "query_tp", "slby_tp", "stex_tp", "stk_cd", "oppo_trde_tp", "fr_ord_no")),
    "ust21180": frozenset(("strt_dt", "end_dt", "slby_tp", "stex_tp", "stk_cd", "oppo_trde_tp")),
    "ust21510": frozenset(("slby_tp", "stex_tp", "stk_cd")),
}


class DiagnosticInputError(ValueError):
    """Fixed-code errors never include raw input, credentials or server text."""


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        # argparse normally echoes invalid choices/unknown credential arguments.
        raise DiagnosticInputError("INVALID_CLI_ARGUMENTS")


def _json_value(value):
    # Reject objects with hooks before copying or formatting fixture data.
    if type(value) in (str, int, bool, type(None)):
        return
    if type(value) is list:
        for item in value:
            _json_value(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _json_value(item)
        return
    raise DiagnosticInputError("INVALID_FIXTURE_VALUE")


def _date(value):
    if type(value) is not str or not re.fullmatch(r"[0-9]{8}", value):
        raise DiagnosticInputError("INVALID_QUERY_DATE")
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError:
        raise DiagnosticInputError("INVALID_QUERY_DATE") from None
    return value


def _request(api_id, body, endpoint=ENDPOINT):
    if type(api_id) is not str or api_id not in QUERY_FIELDS or endpoint != ENDPOINT:
        raise DiagnosticInputError("READ_ONLY_TR_REQUIRED")
    if type(body) is not dict or any(type(k) is not str or k not in QUERY_FIELDS[api_id] for k in body):
        raise DiagnosticInputError("UNDOCUMENTED_REQUEST_FIELD")
    if any(type(v) is not str for v in body.values()):
        raise DiagnosticInputError("INVALID_REQUEST_FIELD_TYPE")
    if body.get("slby_tp") not in ("0", "1", "2"):
        raise DiagnosticInputError("INVALID_SIDE_FILTER")
    if api_id == "ust21150" and body.get("query_tp") not in ("1", "2", "3", "4", "5", "6"):
        raise DiagnosticInputError("INVALID_DAILY_QUERY_FILTER")
    if api_id == "ust21180":
        if _date(body.get("strt_dt")) > _date(body.get("end_dt")):
            raise DiagnosticInputError("INVALID_QUERY_RANGE")
    elif "ord_dt" in body:
        _date(body["ord_dt"])
    if "stex_tp" in body and body["stex_tp"] not in ("ND", "NY", "NA"):
        raise DiagnosticInputError("INVALID_EXCHANGE_FILTER")


class FakeTransport:
    """Sealed fixture replay, not an interface to arbitrary HTTP callables.

    Envelope keys: body, headers, optional observed_account/environment. Scope
    observations are fixture assertions, never verified token/account binding.
    No production transport or auth parameter is accepted.
    """

    __slots__ = ("_pages", "requests")

    def __init__(self, pages):
        _json_value(pages)
        if type(pages) is not list:
            raise DiagnosticInputError("INVALID_FIXTURE_PAGES")
        self._pages = json.loads(json.dumps(pages))
        self.requests = []

    def request(self, api_id, body, continuation=("N", ""), endpoint=ENDPOINT):
        _request(api_id, body, endpoint)
        if type(continuation) is not tuple or len(continuation) != 2 or continuation[0] not in ("N", "Y") or type(continuation[1]) is not str:
            raise DiagnosticInputError("INVALID_CONTINUATION")
        index = len(self.requests)
        self.requests.append({"tr": api_id, "body": dict(body), "continuation": continuation, "endpoint": ENDPOINT})
        if index >= len(self._pages):
            return {"body": {"return_code": -1}, "headers": {"cont-yn": "N"}}
        return json.loads(json.dumps(self._pages[index]))


def _reference(value):
    # Neither account, order number nor arbitrary server text is exported.
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16] if type(value) is str and value else None


def _sensitive_values(value):
    """Collect explicit sensitive labels from memory fixtures, never files.

    Unknown, unlabelled numbers are not classified as credentials or PII.
    """
    labels = frozenset(("token", "accesstoken", "refreshtoken", "authorization",
                        "appsecret", "apikey", "appkey", "secretkey", "secret",
                        "password", "account", "accountno", "accountnumber",
                        "observedaccount", "requestedaccount", "cano", "acntno", "accno",
                        "phone", "phonenumber", "email", "ssn", "personalid",
                        "residentregistrationnumber", "name", "address",
                        "계좌", "계좌번호", "토큰", "비밀번호", "전화번호",
                        "이메일", "주민등록번호", "성명", "주소"))
    found = set()

    def collect(item, sensitive=False):
        if type(item) is dict:
            for key, child in item.items():
                label = re.sub(r"[^a-z0-9가-힣]", "", key.lower())
                collect(child, sensitive or label in labels)
        elif type(item) is list:
            for child in item:
                collect(child, sensitive)
        elif sensitive and type(item) in (str, int):
            secret = str(item)
            if secret:
                found.add(secret)
                bearer = re.fullmatch(r"(?i)Bearer\s+(.+)", secret.strip())
                if bearer:
                    found.add(bearer.group(1))

    collect(value)
    return found


def _redact_known_values(value, sensitive_values):
    """Defense in depth for labelled secrets reflected into quantity/date.

    Unlabelled arbitrary PII cannot be inferred from all possible numbers. This
    guard covers explicitly labelled memory data and known account identifiers.
    """
    known = set()
    for secret in sensitive_values:
        if type(secret) is str and secret:
            known.add(secret)
            digits = re.sub(r"[-\s]", "", secret)
            if re.fullmatch(r"[0-9]+", digits):
                known.add(digits)
                known.add(digits.lstrip("0") or "0")

    def redact(item):
        if type(item) is dict:
            return {key: redact(child) for key, child in item.items()}
        if type(item) is list:
            return [redact(child) for child in item]
        if type(item) is str and any(secret in item for secret in known):
            return "[REDACTED]"
        if type(item) is int and any(secret in str(item) for secret in known):
            return None
        return item

    return redact(value)


def _quantity(value):
    if type(value) is int and value >= 0:
        return value
    if type(value) is str and re.fullmatch(r"[0-9]+", value):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _normalize(row, api_id, body):
    issues = []
    quantities = {}
    for source, target in (("ord_qty", "ordered_quantity"), ("cntr_qty", "reported_execution_quantity"), ("ord_remnq", "remaining_quantity"), ("cncl_qty", "reported_cancel_quantity"), ("mdfy_qty", "reported_modify_quantity")):
        quantities[target] = _quantity(row.get(source))
        if source in row and (type(row[source]) is not str or quantities[target] is None):
            issues.append("INVALID_QUANTITY_FORMAT")
    for field in ("ord_no", "orig_ord_no", "stk_cd", "slby_tp", "ord_stat", "ord_stat_nm", "ord_cntr_tp"):
        if field in row and type(row[field]) is not str:
            issues.append("INVALID_TEXT_FIELD_TYPE")
    order_no = row.get("ord_no") if type(row.get("ord_no")) is str and row["ord_no"].strip() else None
    original = row.get("orig_ord_no") if api_id in ("ust21050", "ust21510") and type(row.get("orig_ord_no")) is str else None
    if api_id == "ust21050" and original and re.fullmatch(r"0+", original):
        original = None
    if order_no is not None and not re.fullmatch(r"[0-9]+", order_no):
        issues.append("INVALID_ORDER_NUMBER_FORMAT")
    if original and not re.fullmatch(r"[0-9]+", original):
        issues.append("INVALID_ORIGINAL_ORDER_NUMBER_FORMAT")
    day = row.get("ord_dt") if api_id == "ust21180" else body.get("ord_dt")
    if api_id == "ust21180" and "ord_dt" in row:
        try:
            day = _date(day)
            if not body["strt_dt"] <= day <= body["end_dt"]:
                issues.append("ROW_DATE_OUTSIDE_QUERY_RANGE")
        except DiagnosticInputError:
            day = None
            issues.append("INVALID_ORDER_DATE")
    raw_side = row.get("slby_tp")
    side = {"1": "SELL", "2": "BUY"}.get(raw_side) if api_id in ("ust21050", "ust21510") and type(raw_side) is str else None
    if api_id in ("ust21050", "ust21510") and "slby_tp" in row and side is None:
        issues.append("INVALID_SIDE_FORMAT")
    raw_kind = row.get("ord_cntr_tp")
    kind = {"10": "ORIGINAL", "11": "MODIFY_REQUEST", "12": "CANCEL_REQUEST"}.get(raw_kind, "UNKNOWN") if api_id == "ust21050" and type(raw_kind) is str else "UNKNOWN"
    ordered = quantities["ordered_quantity"]
    # cntr_qty is not documented as cumulative: do not infer outstanding by subtraction.
    if ordered is not None and any(quantities[name] is not None and quantities[name] > ordered for name in ("reported_execution_quantity", "remaining_quantity")):
        issues.append("QUANTITY_EXCEEDS_ORDERED")
    unknown = ["cumulative_execution_semantics", "acceptance_finality", "cancel_finality", "account_binding"]
    if order_no is None:
        unknown.append("order_number")
    if side is None:
        unknown.append("side")
    if day is None:
        unknown.append("order_date")
    for key, value in quantities.items():
        if value is None:
            unknown.append(key)
    return ({"order_ref": _reference(order_no), "original_order_ref": _reference(original), "symbol_ref": _reference(row.get("stk_cd")),
             "side": side or "UNKNOWN", "order_kind": kind, "order_date": day,
             **quantities, "cumulative_filled_quantity": None, "cancel_confirmed": None,
             "modify_confirmed": None, "source_tr": api_id, "unverified_fields": sorted(set(unknown))}, issues)


def diagnose(transport, api_id, body, *, account="MOCK_SCOPE", environment="SIMULATION", mode="mock", max_pages=MAX_PAGES, reservations=None):
    """Return sanitized evidence; never update an operational object or file.

    reservations is an in-memory list of unresolved SELLs with optional order_no,
    symbol, account and environment. Matching observations do not clear them.
    """
    if mode != "mock":
        raise DiagnosticInputError("REAL_API_NOT_IMPLEMENTED_OR_APPROVED")
    if type(transport) not in (FakeTransport, ReadOnlyTransport):
        raise DiagnosticInputError("FAKE_TRANSPORT_ONLY")
    if type(max_pages) is not int or not 1 <= max_pages <= MAX_PAGES:
        raise DiagnosticInputError("INVALID_QUERY_PAGE_BUDGET")
    if type(account) is not str or not account.strip() or environment not in ("SIMULATION", "REAL"):
        raise DiagnosticInputError("INVALID_REQUEST_SCOPE")
    if type(transport) is ReadOnlyTransport and transport.environment != environment:
        raise DiagnosticInputError("INVALID_REQUEST_SCOPE")
    if type(transport) is ReadOnlyTransport:
        exchange_account = getattr(transport._exchange, "account", None)
        if exchange_account is not None and exchange_account != account:
            raise DiagnosticInputError("CREDENTIAL_ACCOUNT_MISMATCH")
    _request(api_id, body)
    if reservations is None:
        reservations = []
    _json_value(reservations)
    if type(reservations) is not list or any(type(item) is not dict for item in reservations):
        raise DiagnosticInputError("INVALID_RESERVATIONS")
    started = datetime.now(timezone.utc).isoformat()
    pages, rows, seen_keys, continuation = [], [], set(), ("N", "")
    failure, terminated, list_missing = None, False, False
    while len(pages) < max_pages:
        try:
            if type(transport) is FakeTransport:
                response = FakeTransport.request(transport, api_id, body, continuation)
            else:
                response = ReadOnlyTransport.request(transport, api_id, body, continuation)
        except ReadOnlyTransportError as exc:
            code = exc.args[0] if exc.args and exc.args[0] in (
                "AUTHENTICATION_FAILED", "AUTHENTICATION_EXPIRED", "CREDENTIAL_ENVIRONMENT_MISMATCH",
                "CREDENTIAL_ACCOUNT_MISMATCH", "CREDENTIAL_SCOPE_MISMATCH", "REAL_NETWORK_NOT_APPROVED") else "TRANSPORT_REJECTED"
            response = {"body": {"return_code": -1}, "headers": {}, "http_status": 599, "transport_error": code}
        page = {"page": len(pages) + 1, "ok": False, "row_count": 0}
        pages.append(page)
        if type(response) is not dict or type(response.get("body")) is not dict:
            failure = "INVALID_RESPONSE_TYPE"
            break
        if "http_status" in response and (type(response["http_status"]) is not int or response["http_status"] != 200):
            code = response.get("transport_error")
            failure = code if code in ("AUTHENTICATION_FAILED", "AUTHENTICATION_EXPIRED",
                "AUTHENTICATION_REJECTED", "CREDENTIAL_ENVIRONMENT_MISMATCH", "CREDENTIAL_ACCOUNT_MISMATCH",
                "CREDENTIAL_SCOPE_MISMATCH", "REAL_NETWORK_NOT_APPROVED") else "HTTP_QUERY_FAILED"
            break
        for key, requested in (("observed_account", account), ("observed_environment", environment)):
            if key in response and response[key] != requested:
                failure = "SCOPE_MISMATCH"
        payload = response["body"]
        # Supplemental scope echoes are not specified proof of token binding.
        for key in ("account", "account_no", "cano"):
            if key in payload and payload[key] != account:
                failure = "SCOPE_MISMATCH"
        if "environment" in payload and payload["environment"] != environment:
            failure = "SCOPE_MISMATCH"
        if failure:
            break
        if type(payload.get("return_code")) is not int or payload["return_code"] != 0:
            failure = "QUERY_FAILED"
            break
        batch = payload.get("result_list", [])
        if type(batch) is not list or any(type(row) is not dict for row in batch):
            failure = "INVALID_RESULT_LIST"
            break
        list_missing = list_missing or "result_list" not in payload
        rows.extend(batch)
        page.update(ok=True, row_count=len(batch))
        headers = response.get("headers")
        if type(headers) is not dict:
            failure = "CONTINUATION_HEADERS_MISSING"
            break
        flag, key = headers.get("cont-yn"), headers.get("next-key")
        if flag == "N":
            terminated = True
            break
        if flag != "Y":
            failure = "CONTINUATION_FLAG_UNKNOWN"
            break
        if type(key) is not str or not key.strip() or len(key) > 50:
            failure = "NEXT_KEY_MISSING_OR_INVALID"
            break
        if key in seen_keys:
            failure = "NEXT_KEY_REPEATED"
            break
        seen_keys.add(key)
        continuation = ("Y", key)
    if failure is None and not terminated:
        failure = "QUERY_PAGE_BUDGET_EXCEEDED"
    if failure:
        pages[-1]["error"] = failure
    normalized, issues, observed, identities = [], [], {}, {}
    identity_ambiguities = 0
    for row in rows:
        for field in ("account", "account_no", "cano"):
            if field in row and row[field] != account:
                issues.append("ROW_SCOPE_MISMATCH")
        if "environment" in row and row["environment"] != environment:
            issues.append("ROW_SCOPE_MISMATCH")
        order, row_issues = _normalize(row, api_id, body)
        normalized.append(order)
        issues.extend(row_issues)
        ref = order["order_ref"]
        if ref is not None:
            identity = (ref, order["order_date"])
            if identity in identities:
                if order["order_date"] is None:
                    identity_ambiguities += 1
                else:
                    issues.append("PAGE_SNAPSHOT_INCONSISTENT" if identities[identity] != order else "DUPLICATE_ORDER_OBSERVATION")
            identities[identity] = order
            observed.setdefault(ref, []).append(order)
    unresolved_links = 0
    for child in normalized:
        original = child["original_order_ref"]
        if original is None:
            if child["order_kind"] in ("MODIFY_REQUEST", "CANCEL_REQUEST"):
                unresolved_links += 1
            continue
        if original == child["order_ref"]:
            issues.append("SELF_REFERENTIAL_ORIGINAL_ORDER")
            unresolved_links += 1
            continue
        parents = observed.get(original, [])
        # Original-order dates are not returned. A cross-date ID reuse or a
        # missing date must never be resolved by taking an arbitrary first row.
        candidates = [parent for parent in parents if child["order_date"] is not None and parent["order_date"] == child["order_date"]]
        if len(candidates) != 1:
            unresolved_links += 1
            if parents:
                identity_ambiguities += 1
            continue
        parent = candidates[0]
        if parent["symbol_ref"] is not None and child["symbol_ref"] is not None and parent["symbol_ref"] != child["symbol_ref"]:
            issues.append("ORIGINAL_LINK_SYMBOL_MISMATCH")
        if parent["side"] != "UNKNOWN" and child["side"] != "UNKNOWN" and parent["side"] != child["side"]:
            issues.append("ORIGINAL_LINK_SIDE_MISMATCH")
        if parent["symbol_ref"] is None or child["symbol_ref"] is None or parent["side"] == "UNKNOWN" or child["side"] == "UNKNOWN":
            unresolved_links += 1
    unknown_orders = sum(1 for row in normalized if row["order_ref"] is None or row["side"] == "UNKNOWN")
    unconfirmed_sell_orders = [{"order_ref": row["order_ref"], "remaining_quantity": row["remaining_quantity"],
                               "reason": "ACCEPTANCE_CUMULATIVE_QUANTITY_AND_TERMINAL_STATE_UNVERIFIED"}
                              for row in normalized if row["side"] == "SELL"]
    unresolved_sells = []
    for item in reservations:
        ref = _reference(item.get("order_no"))
        candidates = observed.get(ref, []) if ref else []
        if item.get("trade_date") is not None:
            candidates = [row for row in candidates if row["order_date"] == item["trade_date"]]
        matching = candidates[0] if len(candidates) == 1 and candidates[0]["order_date"] is not None else None
        scope_match = item.get("account", account) == account and item.get("environment", environment) == environment
        symbol_match = item.get("symbol") is None or matching is not None and _reference(item["symbol"]) == matching["symbol_ref"]
        if not scope_match:
            issues.append("RESERVATION_SCOPE_MISMATCH")
        phase = item.get("phase")
        unresolved_sells.append({"reservation_ref": _reference(item.get("id", item.get("reservation_id"))), "order_ref": ref,
                                 "phase": phase if phase in ("PREPARED", "SENDING", "ACCEPTED", "UNKNOWN") else "UNKNOWN",
                                 "observation_matched": bool(matching and scope_match and symbol_match and matching["side"] == "SELL"),
                                 "resolved": False, "reason": "AUTHORITATIVE_CUMULATIVE_AND_FINALITY_EVIDENCE_REQUIRED"})
    api_collection_complete = failure is None and terminated and not list_missing
    complete = api_collection_complete and not issues
    statuses = ["INCOMPLETE_HISTORY", "NOT_VERIFIED"]
    statuses.append("CONTRACT_MISMATCH" if issues or failure in ("INVALID_RESPONSE_TYPE", "INVALID_RESULT_LIST", "SCOPE_MISMATCH", "CONTINUATION_FLAG_UNKNOWN") else "CONTRACT_MATCH" if complete else "NOT_VERIFIED")
    if not complete:
        statuses.append("PARTIAL_RESPONSE")
    if unknown_orders or unresolved_links or unresolved_sells or unconfirmed_sell_orders or identity_ambiguities:
        statuses.append("UNKNOWN_ORDER")
    report = {"mode": "MOCK", "source_spec_commit": SPEC_COMMIT, "tr": api_id, "endpoint": ENDPOINT,
            "requested_account": "[MASKED]", "requested_environment": environment,
            "scope_binding_verified": False, "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
            "requested_start_date": body.get("strt_dt", body.get("ord_dt")), "requested_end_date": body.get("end_dt", body.get("ord_dt")),
            "ok": complete, "complete": complete, "collection_complete": complete, "api_collection_complete": api_collection_complete,
            "link_verification_complete": False, "history_complete": False, "ready": False,
            "statuses": sorted(set(statuses)), "failure": failure, "issues": sorted(set(issues)),
            "pages_requested": len(pages), "page_limit": max_pages, "continuation_terminated": terminated,
            "optional_list_absent": list_missing, "pages": pages, "orders": normalized if complete else [],
            "partial_orders": normalized if not complete else [], "unknown_order_count": unknown_orders,
            "unresolved_original_links": unresolved_links, "unresolved_sells": unresolved_sells,
            "identity_ambiguity_count": identity_ambiguities,
            "unconfirmed_sell_orders": unconfirmed_sell_orders,
            "unverified": ["REAL_RESPONSE_COMPATIBILITY", "TOKEN_ACCOUNT_BINDING", "LEDGER_COVERAGE", "PREVIOUS_DAY_UNRESOLVED_ORDERS",
                           "CUMULATIVE_EXECUTION_SEMANTICS", "ORDER_ACCEPTANCE_AND_CANCEL_FINALITY", "PUBLICATION_DELAY",
                           "RESPONSE_FRESHNESS", "PAGE_SNAPSHOT_CONSISTENCY", "ORIGINAL_ORDER_DATE_AND_LEDGER_LINKAGE", "ACK_LOSS_ORDER_IDENTIFICATION", "READY_TRANSITION"],
            "reservation_journal_supplied": bool(reservations), "reservations_mutated": False, "operational_state_mutated": False}
    sensitive = _sensitive_values(transport._pages) | _sensitive_values(reservations) | {account}
    if type(transport) is ReadOnlyTransport and type(transport._credentials._token) is str:
        sensitive.add(transport._credentials._token)
    return _redact_known_values(report, sensitive)


def main(argv=None):
    parser = _SafeArgumentParser(description="Memory-only, Mock-only order query diagnostics; no real API transport exists.")
    parser.add_argument("--mode", choices=("mock", "real"), default="mock")
    parser.add_argument("--tr", choices=tuple(QUERY_FIELDS), default="ust21050")
    parser.add_argument("--date", default="20261013", help="Fixture date YYYYMMDD, not an operational market clock.")
    try:
        args = parser.parse_args(argv)
    except DiagnosticInputError:
        print(json.dumps({"ok": False, "ready": False, "error": "INVALID_CLI_ARGUMENTS"}))
        return 2
    body = {"slby_tp": "0"}
    if args.tr == "ust21180":
        body.update(strt_dt=args.date, end_dt=args.date)
    elif args.tr != "ust21510":
        body["ord_dt"] = args.date
    if args.tr == "ust21150":
        body["query_tp"] = "5"
    try:
        result = diagnose(FakeTransport([{"body": {"return_code": 0, "result_list": []}, "headers": {"cont-yn": "N"}}]), args.tr, body, mode=args.mode)
    except DiagnosticInputError as error:
        print(json.dumps({"ok": False, "ready": False, "error": str(error)}))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
