"""Opposite relay policy and atomic account-bound journal. No broker calls."""
import copy
import json
import math
import os
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import config


def symbols():
    return (config.BASE_ASSET_LONG, config.BASE_ASSET_SHORT)


def opposite(symbol):
    long, short = symbols()
    if symbol not in (long, short):
        raise ValueError('Unknown relay symbol')
    return short if symbol == long else long


def entry_window(now):
    now = now.astimezone(ZoneInfo('America/New_York'))
    return now.weekday() < 5 and config.PHASE_MAIN_START <= now.strftime('%H:%M') < config.PHASE_MAIN_END


def decision_key(now):
    now = now.astimezone(ZoneInfo('America/New_York'))
    return now.replace(minute=now.minute // config.RELAY_DECISION_INTERVAL_MINUTES *
                       config.RELAY_DECISION_INTERVAL_MINUTES, second=0, microsecond=0).isoformat()


def trailing_trigger(role):
    if role not in ('INITIAL', 'CHAIN'):
        raise ValueError('Unverified relay entry role')
    return config.TRAILING_TRIGGER_PCT if role == 'INITIAL' else config.RELAY_CHAIN_TRAILING_TRIGGER_PCT


def ws_quote(packet):
    """Accept explicit WebSocket bid/ask only, never midpoint/REST prices."""
    if not isinstance(packet, dict) or packet.get('from_ws') is not True:
        return None
    raw = packet.get('extra', {})
    values = raw.get('values', raw) if isinstance(raw, dict) else {}
    def number(keys):
        for key in keys:
            if key in values:
                try:
                    value = abs(float(str(values[key]).replace(',', '')))
                except (ValueError, TypeError):
                    return None
                return value if math.isfinite(value) and value > 0 else None
        return None
    bid = number(('51', 'bid_price', 'bid'))
    ask = number(('41', 'ask_price', 'ask'))
    return (bid, ask) if bid is not None and ask is not None and bid <= ask else None


def exit_reason(position, bid, now, polling=True):
    """Bid-based SL/TP, cadence-based trailing, and independent time/EOD exits."""
    entry = float(position['price'])
    if now.strftime('%H:%M') >= config.PHASE_EOD_CLEAR:
        return 'EOD'
    if position.get('soxx_orb_strategy'):
        if bid is not None:
            if bid <= entry * (1 - config.SL_MIN_PCT):
                return 'SL'
            peak = float(position.get('high_price_during_hold', entry))
            if polling and peak >= entry * (1 + config.TRAILING_TRIGGER_PCT):
                if bid <= peak * (1 - config.TRAILING_DROP_PCT):
                    return 'TRAIL'
        return None
    started = datetime.fromisoformat(position['relay_entry_ny'])
    if bid is not None:
        if bid <= entry * (1 - config.SL_MIN_PCT):
            return 'SL'
        if bid >= entry * (1 + config.MAX_TP_PCT):
            return 'TP'
        peak = float(position.get('high_price_during_hold', entry))
        if polling and peak >= entry * (1 + trailing_trigger(position['relay_role'])):
            if bid <= peak * (1 - config.TRAILING_DROP_PCT):
                return 'TRAIL'
    if now - started >= timedelta(minutes=config.TIME_STOP_MINUTES):
        return 'TIME'
    return None


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class RelayBook:
    def __init__(self, path, binding):
        self.path, self.binding = Path(path), binding
        self.instance_id = uuid.uuid4().hex
        self.lock = threading.RLock()
        self.failed = False
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding='utf-8'))
            if self.data.get('binding') != binding or self.data.get('version') != 1:
                raise RuntimeError('Relay journal identity mismatch')
            if not isinstance(self.data.get('operations'), dict):
                raise RuntimeError('Invalid relay operations journal')
        else:
            self.data = dict(version=1, binding=binding, budget=None, cycle_id=None,
                             last_decision=None, flat_after=None, operations={})

    def update(self, **values):
        with self.lock:
            if self.failed:
                raise RuntimeError('Relay persistence is unavailable')
            candidate = dict(self.data, **values)
            try:
                atomic_json(self.path, candidate)
            except Exception:
                self.failed = True
                raise
            self.data = candidate

    def begin(self, available, key):
        with self.lock:
            if not math.isfinite(available) or available <= 0:
                raise ValueError('Invalid available USD')
            if key in (self.data['last_decision'], self.data['flat_after']):
                return False
            if self.data['operations'] or self.data['budget'] is not None:
                return False
            # chain_operations: 이번 사이클에서 예약된 연쇄 매수 ID 목록 (최대 RELAY_MAX_CHAIN_COUNT)
            # chain_operation_id: 하위 호환성을 위해 최근 체인 op ID 보존 (시작 시 None)
            self.update(budget=float(available), cycle_id=uuid.uuid4().hex, last_decision=key,
                        chain_operations=[], chain_operation_id=None, initial_reservation_owner=self.instance_id)
            return True

    def initial_preparing(self, key):
        with self.lock:
            return (self.data.get('initial_reservation_owner') == self.instance_id and
                    self.data.get('last_decision') == key)

    def release_initial(self):
        with self.lock:
            if self.data.get('initial_reservation_owner') == self.instance_id:
                self.update(initial_reservation_owner=None)

    def quantity(self, available, price, role):
        budget = self.data['budget']
        if budget is None or not math.isfinite(price) or price <= 0:
            return 0
        ratio = config.RELAY_INITIAL_ALLOCATION if role == 'INITIAL' else config.RELAY_CHAIN_ALLOCATION
        cap = min(budget * ratio, max(0., available - budget * config.RELAY_CASH_BUFFER))
        return int(cap / (price + config.QTY_CALC_BUFFER))

    def operation(self, symbol, value):
        if symbol not in symbols():
            raise ValueError('Unknown relay symbol')
        with self.lock:
            operations = copy.deepcopy(self.data['operations'])
            previous = copy.deepcopy(operations.get(symbol))
            if value is None:
                operations.pop(symbol, None)
            else:
                operations[symbol] = copy.deepcopy(value)
            updates = dict(operations=operations)
            if value is not None and value.get('action') == 'BUY' and value.get('role') == 'INITIAL':
                updates['initial_reservation_owner'] = None
            if value is not None and value.get('action') == 'BUY' and value.get('role') == 'CHAIN':
                # 연쇄 매수 예약: 같은 사이클 내 RELAY_MAX_CHAIN_COUNT 회까지 허용.
                # 동일 operation ID는 재발주 등 기존 오퍼레이션 재시도로 허용한다.
                existing = list(self.data.get('chain_operations', []))
                op_id = value['id']
                if op_id not in existing:
                    if len(existing) >= config.RELAY_MAX_CHAIN_COUNT:
                        raise RuntimeError('Relay chain limit reached for this cycle')
                    existing.append(op_id)
                    updates['chain_operations'] = existing
                    updates['chain_operation_id'] = op_id
                else:
                    if self.data.get('chain_operation_id') != op_id:
                        raise RuntimeError('Relay chain reservation mismatch')
            self.update(**updates)
            self.event('OPERATION_TERMINAL' if value is None else 'OPERATION_UPDATED',
                       symbol=symbol, operation=value if value is not None else previous)

    def event(self, event, **values):
        with self.lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                row = dict(event=event, time=datetime.now(ZoneInfo('America/New_York')).isoformat(),
                           cycle_id=self.data['cycle_id'], **values)
                with (self.path.parent / 'events.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(row, allow_nan=False, ensure_ascii=False) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
            except Exception:
                self.failed = True
                raise
