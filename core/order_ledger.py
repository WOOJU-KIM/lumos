"""Pure order evidence reduction shared by broker and runner; no I/O or policy settings."""
import re


def _text(row, *keys):
    for key in keys:
        if key in row and row[key] is not None:
            return str(row[key]).strip()
    return ""


def _quantity(row, *keys):
    text = _text(row, *keys)
    if not re.fullmatch(r"[0-9]+", text):
        raise ValueError("MISSING_OR_INVALID_QUANTITY")
    return int(text)


def scan_order_rows(rows, order_no, symbol, side, expected_qty=None,
                    trade_date=None, cancel_order_no=None, expected_account=None):
    """Keep absence, malformed evidence and zero distinct. Never sum duplicate fills.

    cntr_qty is an execution quantity; tot_che_qty is explicitly cumulative. An
    unidentifiable duplicate partial execution is ambiguous, not two fills.
    """
    result = dict(found=False, error=None, requested=None, filled=0,
                  parent_cancel=0, child_cancel=0, remaining=0)
    if not isinstance(rows, list):
        result['error'] = 'INVALID_LEDGER_LIST'
        return result
    expected_date = str(trade_date or '').replace('-', '')
    originals, children, seen = [], [], {}
    for row in rows:
        if not isinstance(row, dict):
            result['error'] = 'INVALID_LEDGER_ROW'
            return result
        number = _text(row, 'ord_no', 'order_no')
        parent = _text(row, 'orig_ord_no', 'original_order_no')
        if parent.isdecimal() and int(parent) == 0:
            parent = ''
        linked = number == order_no or (parent and parent == order_no) or (cancel_order_no and number == cancel_order_no)
        if not linked:
            continue
        result['found'] = True
        is_child = (parent == order_no and number != order_no) or (cancel_order_no and number == cancel_order_no and number != order_no) or (number == order_no and _text(row, 'ord_cntr_tp') == '12')
        actual_symbol = _text(row, 'stk_cd', 'symbol').upper()
        raw_side = _text(row, 'side', 'slby_tp', 'slby_tp_nm')
        actual_side = {'1': 'SELL', '2': 'BUY', '매수': 'BUY', '매도': 'SELL'}.get(raw_side, raw_side.upper())
        actual_date = _text(row, 'ord_dt', 'trade_date', 'order_date').replace('-', '')
        if (actual_symbol != symbol.upper() or (not is_child and actual_side != side.upper())
                or (is_child and actual_side and actual_side != side.upper())
                or (actual_date and expected_date and actual_date != expected_date)
                or (expected_account and 'account' in row and row['account'] != expected_account)
                or (parent and parent != order_no)
                or (is_child and cancel_order_no and number not in (order_no, cancel_order_no))):
            result['error'] = 'ORDER_IDENTITY_MISMATCH'
            return result
        try:
            requested = _quantity(row, 'ord_qty', 'ordered_quantity')
            filled = _quantity(row, 'tot_che_qty') if 'tot_che_qty' in row else _quantity(row, 'cntr_qty')
            canceled = _quantity(row, 'cancelled_qty', 'cncl_cfrm_qty', 'cncl_qty')
            remaining = _quantity(row, 'nccs_qty', 'ord_remnq', 'remaining_quantity')
        except ValueError:
            result['error'] = 'MISSING_OR_INVALID_QUANTITY'
            return result
        if requested <= 0 or max(filled, canceled, remaining) > requested:
            result['error'] = 'QUANTITY_OUT_OF_RANGE'
            return result
        if not is_child and expected_qty is not None and requested != expected_qty:
            result['error'] = 'ORDER_QUANTITY_MISMATCH'
            return result
        signature = tuple(sorted((str(k), str(v)) for k, v in row.items()))
        execution_id = _text(row, 'cntr_no', 'che_no', 'exec_no')
        identity = (is_child, number, execution_id) if execution_id else signature
        if identity in seen:
            if seen[identity] != signature:
                result['error'] = 'EXECUTION_ID_CONFLICT'
                return result
            if execution_id or is_child or filled + canceled + remaining == requested:
                continue
            result['error'] = 'AMBIGUOUS_DUPLICATE_EXECUTION'
            return result
        seen[identity] = signature
        value = (requested, filled, canceled, remaining, 'tot_che_qty' in row)
        (children if is_child else originals).append(value)
    if not originals:
        if children:
            result['error'] = 'MISSING_ORIGINAL_ORDER'
        return result
    quantities = {v[0] for v in originals}
    if len(quantities) != 1:
        result['error'] = 'ORDER_QUANTITY_CONFLICT'
        return result
    result['requested'] = originals[0][0]
    cumulative = [v[1] for v in originals if v[4]]
    increments = sum(v[1] for v in originals if not v[4])
    result['filled'] = max(cumulative) if cumulative else increments
    if cumulative and increments > result['filled']:
        result['error'] = 'CUMULATIVE_EXECUTION_CONFLICT'
        return result
    result['parent_cancel'] = max(v[2] for v in originals)
    result['remaining'] = max(v[3] for v in originals)
    if any(v[1] > 0 or v[3] > 0 for v in children):
        result['error'] = 'INVALID_CANCEL_CHILD'
        return result
    result['child_cancel'] = sum(v[2] for v in children)
    if result['parent_cancel'] and result['child_cancel'] and result['parent_cancel'] != result['child_cancel']:
        result['error'] = 'CANCEL_QUANTITY_CONFLICT'
    if result['filled'] + max(result['parent_cancel'], result['child_cancel']) + result['remaining'] > result['requested']:
        result['error'] = 'QUANTITY_CONSERVATION_FAILED'
    return result


def resolve_order_evidence(open_orders, executions, order_history, order_no, symbol,
                           side, expected_qty=None, trade_date=None, cancel_order_no=None):
    result = dict(status='UNKNOWN', ord_qty=0, filled_qty=0, cancelled_qty=0,
                  remaining_qty=0, found_in_ledger=False, has_open_order=False,
                  reason='NO_ORDER_EVIDENCE')
    facts = [scan_order_rows(rows, order_no, symbol, side, expected_qty, trade_date, cancel_order_no)
             for rows in (open_orders, executions, order_history)]
    result['found_in_ledger'] = any(f['found'] for f in facts)
    if any(f['error'] for f in facts):
        result['reason'] = next(f['error'] for f in facts if f['error'])
        return result
    opened = facts[0]
    if opened['remaining'] > 0:
        result.update(status='PARTIALLY_FILLED' if opened['filled'] else 'OPEN',
                      ord_qty=opened['requested'], filled_qty=opened['filled'],
                      cancelled_qty=max(opened['parent_cancel'], opened['child_cancel']),
                      remaining_qty=opened['remaining'], has_open_order=True, reason='LIVE_OPEN_ORDER')
        return result
    observed = [f for f in facts if f['requested'] is not None]
    if not observed:
        return result
    if len({f['requested'] for f in observed}) != 1 or len({f['filled'] for f in observed}) != 1:
        result['reason'] = 'CROSS_LEDGER_QUANTITY_CONFLICT'
        return result
    requested = observed[0]['requested']
    filled = observed[0]['filled']
    canceled = max(max(f['parent_cancel'], f['child_cancel']) for f in observed)
    remaining = max(f['remaining'] for f in observed)
    result.update(ord_qty=requested, filled_qty=filled, cancelled_qty=canceled, remaining_qty=remaining)
    if remaining or filled + canceled != requested:
        result['reason'] = 'ORDER_NOT_TERMINAL'
        return result
    if filled == requested:
        result['status'] = 'FILLED'
    elif filled:
        result['status'] = 'CANCELED_PARTIAL_FILL'
    else:
        result['status'] = 'CANCELED_ZERO_FILL'
    result['reason'] = 'TERMINAL_ORDER_VERIFIED'
    return result
