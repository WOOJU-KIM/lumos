"""Atomic crash-recovery evidence for a BUY whose terminal state is not verified."""
import hashlib
import json
import os
import uuid
from pathlib import Path


def binding_for(broker):
    identity = (str(getattr(broker, 'account_no', '')), str(getattr(broker, 'account_type', '')),
                bool(getattr(broker, 'is_simulation', True)))
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


def journal_path(position_path):
    path = Path(position_path)
    return path.with_suffix('.pending_buy.json')


def read_pending(path, binding):
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        order = value['order']
        if (value['version'] != 1 or value['binding_sha256'] != binding
                or not isinstance(order, dict) or not order.get('symbol')
                or type(order.get('target_qty')) is not int or order['target_qty'] <= 0):
            raise ValueError('Unverified pending order identity')
        return order
    except (OSError, ValueError, KeyError, TypeError):
        # A corrupt/mismatched journal must block new BUYs, not look empty.
        return {'recovery_error': 'UNVERIFIED_PENDING_BUY_JOURNAL'}


def write_pending(path, binding, order):
    if order is None:
        if path.exists():
            path.unlink()
        return
    if not isinstance(order, dict) or not order.get('symbol') or type(order.get('target_qty')) is not int or order['target_qty'] <= 0:
        raise ValueError('Invalid pending BUY evidence')
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(dict(version=1, binding_sha256=binding, order=order), stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
