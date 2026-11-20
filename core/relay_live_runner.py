"""Two-symbol live relay. Reuses verified broker order/cancel/fill machinery.

The legacy runner remains the 98% single-position reference implementation.
Every real order here is serialized and scoped to its own durable symbol journal.
"""
import copy
import hashlib
import json
import math
import os
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import config
from core.live_runner import KiwoomLiveRunner, USMarketCalendar, logger
from core.pending_order import binding_for, journal_path, read_pending, write_pending
from core.relay_policy import (RelayBook, atomic_json, decision_key, entry_window,
                               exit_reason, opposite, symbols, trailing_trigger, ws_quote)
from core.system_logger import system_logger
from core.kiwoom_broker import KiwoomBroker


class _Quotes:
    def __init__(self, runner, original):
        self.runner, self.original = runner, original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def get_latest_price(self, symbol, default=None):
        action = getattr(self.runner._scope, 'action', None)
        quote = self.runner._quote(symbol)
        if action == 'BUY':
            if quote is None:
                raise RuntimeError('Order deferred: fresh WebSocket Ask unavailable')
            return quote['ask']
        if action == 'SELL' and quote is not None:
            return quote['bid']
        return self.original.get_latest_price(symbol, default)


class _Broker:
    def __init__(self, runner, original):
        self.runner, self.original = runner, original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def send_order(self, **kwargs):
        r = self.runner
        symbol, side = kwargs['symbol'], kwargs['order_type']
        operation = copy.deepcopy(r.book.data['operations'].get(symbol))
        if not operation or operation['action'] != side:
            raise RuntimeError('Order deferred: durable relay intent missing')
        if side == 'BUY':
            if not r._can_enter_now() or r._quote(symbol) is None:
                raise RuntimeError('Order deferred: entry session or WebSocket quote invalid')
            if operation['role'] == 'INITIAL' and operation.get('decision_key') != r._decision_key(r._now()):
                raise RuntimeError('Order deferred: completed candle decision expired')
            if operation['role'] == 'CHAIN' and not r._chain_condition(operation['trigger_symbol']):
                raise RuntimeError('Order deferred: opposite trigger no longer valid')
            if operation['role'] == 'CHAIN':
                chain_ops = r.book.data.get('chain_operations')
                if chain_ops is None:
                    chain_ops = [r.book.data['chain_operation_id']] if r.book.data.get('chain_operation_id') else []
                if r.book.data.get('chain_operation_id') != operation['id'] or operation['id'] not in chain_ops:
                    raise RuntimeError('Order deferred: cycle chain reservation mismatch')
            balance = self.original.get_overseas_stock_balance(force_refresh=True)
            holdings = r._holdings(balance)
            held, average = holdings.get(symbol, (0, 0.))
            if held and not operation['orders']:
                raise RuntimeError('Order deferred: target became occupied before submission')
            if operation['role'] == 'CHAIN':
                trigger = operation['trigger_symbol']
                parent = r._positions.get(trigger)
                verified = holdings.get(trigger)
                if (not parent or not verified or verified[0] != int(parent['quantity']) or
                        not math.isclose(verified[1], float(parent['price']))):
                    raise RuntimeError('Order deferred: triggering holding no longer verified')
            if not r._orders_empty():
                raise RuntimeError('Order deferred: broker open orders changed before submission')
            if held * average + kwargs['quantity'] * kwargs['price'] > operation['budget_cap']:
                raise RuntimeError('Order deferred: relay allocation exceeded')
            if r._quote(symbol) is None or not r._can_enter_now():
                raise RuntimeError('Order deferred: quote/session expired during preflight')
        operation['submission_pending'] = True
        r.book.operation(symbol, operation)
        pricing_rejected = False
        def resolve_price():
            nonlocal pricing_rejected
            try:
                quote = r._quote(symbol)
                if quote is None or not r._can_enter_now():
                    raise RuntimeError('Order deferred: fresh WebSocket Ask/session unavailable at send')
                if operation['role'] == 'INITIAL' and operation.get('decision_key') != r._decision_key(r._now()):
                    raise RuntimeError('Order deferred: completed candle expired at send')
                if operation['role'] == 'CHAIN' and not r._chain_condition(operation['trigger_symbol']):
                    raise RuntimeError('Order deferred: chain trigger expired at send')
                if operation['role'] == 'CHAIN':
                    current_intent = r.book.data['operations'].get(symbol, {})
                    current_parent = r._positions.get(operation['trigger_symbol'])
                    chain_ops = r.book.data.get('chain_operations')
                    if chain_ops is None:
                        chain_ops = [r.book.data['chain_operation_id']] if r.book.data.get('chain_operation_id') else []
                    if (r.book.data.get('chain_operation_id') != operation['id'] or
                            operation['id'] not in chain_ops or
                            current_intent.get('id') != operation['id'] or
                            not current_parent or int(current_parent['quantity']) != verified[0] or
                            not math.isclose(float(current_parent['price']), verified[1])):
                        raise RuntimeError('Order deferred: chain reservation/parent changed at send')
                price = round(quote['ask'] + config.BUY_SLIPPAGE_ADJUST, 2)
                if held * average + kwargs['quantity'] * price > operation['budget_cap']:
                    raise RuntimeError('Order deferred: latest Ask exceeds relay budget cap')
                current = copy.deepcopy(r.book.data['operations'][symbol])
                current['limit_pricing'] = dict(ask=quote['ask'], bid=quote['bid'],
                    quote_observed_epoch=quote['time'], quote_age_seconds=time.time()-quote['time'],
                    buffer=config.BUY_SLIPPAGE_ADJUST, submitted_limit=price)
                r.book.operation(symbol, current)
                return price
            except Exception:
                pricing_rejected = True
                raise
        try:
            if side == 'BUY' and isinstance(self.original, KiwoomBroker):
                result = self.original.send_order(**kwargs, price_resolver=resolve_price)
            else:
                if side == 'BUY':
                    kwargs['price'] = resolve_price()
                result = self.original.send_order(**kwargs)
                if side == 'BUY' and isinstance(result, dict):
                    result = dict(result, price=kwargs['price'])
        except Exception:
            # Resolver rejection is before HTTP construction, so no BUY was sent.
            # Transport failures after a successful resolver stay unresolved.
            if pricing_rejected:
                current = copy.deepcopy(r.book.data['operations'][symbol])
                current['submission_pending'] = False
                r.book.operation(symbol, current)
            raise
        operation = copy.deepcopy(r.book.data['operations'][symbol])
        order_no = str(result.get('order_no') or '') if isinstance(result, dict) else ''
        if order_no:
            operation['orders'].append(dict(order_no=order_no, quantity=kwargs['quantity']))
            operation['submission_pending'] = False
        # Missing identity is deliberately unresolved, even when the local call returned.
        r.book.operation(symbol, operation)
        return result


class RelayLiveRunner(KiwoomLiveRunner):
    _uses_relay_notifications = True

    def __init__(self, is_simulation=None):
        super().__init__(is_simulation=is_simulation)
        self._initialize_relay_state()

    def _initialize_relay_state(self):
        """Bind journals only after the concrete broker/account is available."""
        self._scope = threading.local()
        self._relay_lock = threading.RLock()
        self._quote_lock = threading.Lock()
        self._entry_mutex = threading.Lock()
        self._risk_mutex = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._quotes = {}
        self._extremes = {}
        self._risk_hits = {}
        self._last_poll = {}
        self._balance_cache = None
        self._balance_at = 0.
        self._decision = None
        self._mismatch = None
        self._journal_failed = False
        self._position_write_failures = set()
        self._raw_broker = self.broker
        self._binding = binding_for(self.broker)
        self._root = Path(config.DATA_DIR) / config.RELAY_STATE_DIRNAME / self._binding
        self.book = RelayBook(self._root / 'book.json', self._binding)
        self._positions = {}
        self._legacy_present = (Path(config.DATA_DIR) / 'active_position.json').exists() or (
            Path(config.DATA_DIR) / 'active_position.pending_buy.json').exists()
        self.broker = _Broker(self, self.broker)
        self.ws_streamer = _Quotes(self, self.ws_streamer)
        self._soxx_orb_enabled = os.getenv('LUMOS_DISABLE_SOXX_ORB_LIVE', '0').strip() != '1'
        self._eval_interval_minutes = 5 if self._soxx_orb_enabled else config.RELAY_DECISION_INTERVAL_MINUTES
        self._last_5m_trigger_key = None
        self._soxx_orb_daily_trades = {}
        self._soxx_orb_logged_dates = set()
        self._soxx_orb_log_path = Path(config.DATA_DIR) / 'soxx_orb_live_trades.log'
        self._soxx_orb_jsonl_path = Path(config.DATA_DIR) / 'soxx_orb_live_events.jsonl'
        if self._soxx_orb_enabled and hasattr(self, 'moe_orchestrator') and self.moe_orchestrator is not None:
            self._orig_evaluate_dual_filter_signal = self.moe_orchestrator.evaluate_dual_filter_signal
            self.moe_orchestrator.evaluate_dual_filter_signal = self._evaluate_soxx_orb_5m_signal
        self._deployment_ok = self._verify_deployment()
        for symbol in symbols():
            with self._symbol(symbol):
                self._get_active_position()
        if not math.isclose(config.RELAY_INITIAL_ALLOCATION + config.RELAY_CHAIN_ALLOCATION +
                            config.RELAY_CASH_BUFFER, 1.):
            raise RuntimeError('Relay allocation does not sum to total capital')
        if (not (0 < config.RELAY_INITIAL_ALLOCATION <= config.MAX_ALLOCATION_RATIO and
                 0 <= config.RELAY_CHAIN_ALLOCATION < 1 and
                 0 < config.RELAY_CASH_BUFFER < 1) or
                config.RELAY_INITIAL_ALLOCATION + config.RELAY_CHAIN_ALLOCATION > config.MAX_ALLOCATION_RATIO or
                not 0 < config.RELAY_OPPOSITE_TRIGGER_PCT < config.SL_MIN_PCT or
                not 0 < config.RELAY_CHAIN_TRAILING_TRIGGER_PCT <= config.TRAILING_TRIGGER_PCT):
            raise RuntimeError('Invalid relay risk/allocation settings')

    def _verify_deployment(self):
        try:
            value = json.loads((Path(config.MODELS_DIR) / 'active_execution_strategy.json').read_text(encoding='utf-8'))
            required = {'config.py', 'core/live_runner.py', 'core/relay_live_runner.py', 'core/relay_policy.py',
                        'core/kiwoom_ws_streamer.py', 'core/circuit_breaker.py', 'core/kiwoom_broker.py',
                        'core/telegram_notifier.py', 'core/data_lake.py'}
            if (value.get('strategy_id') != config.LIVE_EXECUTION_STRATEGY or
                    value.get('qa_passed') is not True or set(value['source_hashes']) != required):
                return False
            root = Path(config.BASE_DIR)
            for name, digest in value['source_hashes'].items():
                raw = (root / name).read_bytes()
                if digest not in (
                    hashlib.sha256(raw).hexdigest(),
                    hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest(),
                    hashlib.sha256(raw.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')).hexdigest(),
                ):
                    return False
            return True
        except (OSError, ValueError, TypeError, KeyError):
            return False

    @contextmanager
    def _symbol(self, symbol, action=None):
        old = (getattr(self._scope, 'symbol', None), getattr(self._scope, 'action', None))
        self._scope.symbol, self._scope.action = symbol, action
        try:
            yield
        finally:
            self._scope.symbol, self._scope.action = old

    def _now(self):
        return datetime.now(ZoneInfo('America/New_York'))

    def _decision_interval(self):
        return 5 if getattr(self, '_soxx_orb_enabled', False) else config.RELAY_DECISION_INTERVAL_MINUTES

    def _decision_key(self, now):
        if not getattr(self, '_soxx_orb_enabled', False):
            return decision_key(now)
        now = now.astimezone(ZoneInfo('America/New_York'))
        interval = self._decision_interval()
        return now.replace(minute=(now.minute // interval) * interval, second=0, microsecond=0).isoformat()

    def _candle_symbol(self):
        return 'SOXX' if getattr(self, '_soxx_orb_enabled', False) else config.BASE_ASSET_LONG

    def _candle_timeframe(self):
        return '5m' if getattr(self, '_soxx_orb_enabled', False) else '15m'

    def _write_soxx_orb_log(self, line):
        path = getattr(self, '_soxx_orb_log_path', None)
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('a', encoding='utf-8') as stream:
                    stream.write(line + '\n')
            except OSError:
                pass

    def _append_soxx_orb_event(self, event, **fields):
        path = getattr(self, '_soxx_orb_jsonl_path', None)
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                record = dict(
                    event=event,
                    time=self._now().isoformat(),
                    strategy_id='SOXX_ORB_VWAP_RVOL',
                    cycle_id=self.book.data.get('cycle_id') if hasattr(self, 'book') else None,
                    **fields,
                )
                with path.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
            except OSError:
                pass

    def _can_enter_now(self):
        halted, _ = self._is_circuit_breaker_halted()
        return (self._deployment_ok and self._account_ready() and not halted and not self._stop.is_set() and entry_window(self._now()) and
                USMarketCalendar.get_market_status().get('session_name') == 'REGULAR_MARKET_OPEN')

    def _account_ready(self):
        if self._raw_broker.is_simulation:
            return True
        from core.circuit_breaker import read_order_state, order_readiness
        state = read_order_state()
        proof = state.get('reconciliation_proof', {}) if isinstance(state, dict) else {}
        identity = str(self._raw_broker.account_no) + '-' + str(self._raw_broker.account_type)
        return (order_readiness(state).get('ready') is True and
                proof.get('verified_account') == identity and proof.get('verified_environment') == 'REAL')

    def _quote(self, symbol):
        with self._quote_lock:
            q = self._quotes.get(symbol)
            if q is None or not 0 <= time.time() - q['time'] <= config.RELAY_QUOTE_MAX_AGE_SECONDS:
                return None
            return dict(q)

    def _resolve_pos_file(self):
        symbol = getattr(self._scope, 'symbol', None)
        if symbol not in symbols():
            raise RuntimeError('Symbol-scoped position required')
        return self._root / (symbol + '.json')

    def _get_active_position(self):
        if not hasattr(self, '_scope'):
            return None
        symbol = getattr(self._scope, 'symbol', None)
        if symbol is None:
            with self._relay_lock:
                if self._positions:
                    return copy.deepcopy(next(iter(self._positions.values())))
                if self.book.data['operations'] or self._pending_any():
                    return {'symbol': 'RELAY_PENDING', 'quantity': 0}
            return None
        if symbol not in self._positions and self._resolve_pos_file().exists():
            value = json.loads(self._resolve_pos_file().read_text(encoding='utf-8'))
            if value.get('relay_binding') != self._binding or value.get('relay_role') not in ('INITIAL', 'CHAIN'):
                raise RuntimeError('Unverified relay position metadata')
            if value.get('symbol') != symbol or int(value['quantity']) <= 0:
                raise RuntimeError('Invalid relay position')
            self._positions[symbol] = value
        result = copy.deepcopy(self._positions.get(symbol))
        if result is not None and getattr(self._scope, 'action', None) == 'SELL':
            result['relay_remaining_symbols'] = sorted(s for s in self._positions if s != symbol)
        return result

    def _save_active_position(self, position):
        symbol = getattr(self._scope, 'symbol', None)
        if symbol != position.get('symbol'):
            raise RuntimeError('Cross-symbol position write rejected')
        with self._relay_lock:
            prior = self._positions.get(symbol, {})
            operation = self.book.data['operations'].get(symbol, {})
            role = prior.get('relay_role') or operation.get('role')
            if role not in ('INITIAL', 'CHAIN'):
                raise RuntimeError('Missing relay role')
            soxx_orb = bool(getattr(self, '_soxx_orb_enabled', False) or prior.get('soxx_orb_strategy'))
            trig_pct = config.TRAILING_TRIGGER_PCT if soxx_orb else trailing_trigger(role)
            value = dict(position, relay_binding=self._binding, relay_role=role,
                         relay_cycle_id=prior.get('relay_cycle_id', self.book.data['cycle_id']),
                         relay_leg_id=prior.get('relay_leg_id', operation.get('id')),
                         relay_entry_ny=prior.get('relay_entry_ny', operation.get('created_ny')),
                         trailing_trigger_pct=trig_pct)
            if soxx_orb:
                value['soxx_orb_strategy'] = True
            try:
                atomic_json(self._resolve_pos_file(), value)
            except Exception:
                failures = getattr(self, '_position_write_failures', set())
                if symbol not in failures:
                    system_logger.log('ERROR', 'Relay',
                        f'Position persistence unavailable for {symbol}; BUY paused until ledger-verified recovery')
                failures.add(symbol)
                self._position_write_failures = failures
                raise
            self._positions[symbol] = value

    def _recover_position_writes(self, holdings):
        """Release only position-write failures, after fresh ledger and disk proof.

        Pending-order/book failures remain sticky. Called with _relay_lock held
        by the normal ledger refresh; no additional retry interval or broker loop.
        """
        failures = getattr(self, '_position_write_failures', set())
        if (not failures or self._mismatch or self._journal_failed or self.book.failed or
                self._is_order_in_progress or self.book.data['operations'] or self._pending_any()):
            return
        if not self._orders_empty():
            return
        try:
            with self.book.lock:
                if json.loads(self.book.path.read_text(encoding='utf-8')) != self.book.data:
                    return
                candidates = {}
                for symbol in symbols():
                    with self._symbol(symbol):
                        path = self._resolve_pos_file()
                        position = self._positions.get(symbol)
                        if position is None:
                            if symbol in holdings or path.exists():
                                return
                            continue
                        qty, average = holdings.get(symbol, (0, 0.))
                        disk = json.loads(path.read_text(encoding='utf-8'))
                        identity = ('symbol', 'quantity', 'price', 'relay_binding',
                                    'relay_cycle_id', 'relay_leg_id', 'relay_role')
                        if (qty != int(position['quantity']) or
                                not math.isclose(average, float(position['price'])) or
                                any(disk.get(key) != position.get(key) for key in identity)):
                            return
                        candidate = copy.deepcopy(position)
                        with self._quote_lock:
                            observed = self._extremes.get(symbol)
                        if observed and observed[0] == position['relay_leg_id']:
                            candidate['high_price_during_hold'] = max(
                                float(candidate.get('high_price_during_hold', average)), observed[1])
                            candidate['low_price_during_hold'] = min(
                                float(candidate.get('low_price_during_hold', average)), observed[2])
                        candidates[symbol] = candidate
                for symbol, candidate in candidates.items():
                    with self._symbol(symbol):
                        atomic_json(self._resolve_pos_file(), candidate)
                self._positions.update(candidates)
                self._position_write_failures.clear()
            system_logger.log('INFO', 'Relay',
                'Position persistence recovered after broker holdings, open orders and durable state verification')
        except (OSError, ValueError, TypeError, KeyError):
            # Existing refresh cadence retries. Keep BUY blocked and exits active.
            return

    def _clear_active_position(self):
        # The legacy sell helper asks to clear before reporting/logging. Defer the
        # removal until the fresh account AND order ledgers prove final closure.
        if getattr(self._scope, 'action', None) == 'SELL':
            return
        if self._get_active_position() is not None:
            raise RuntimeError('Filled relay position cannot be erased by BUY cleanup')

    def _load_pending_buy_order(self):
        return read_pending(journal_path(self._resolve_pos_file()), self._binding)

    def _store_pending_buy_order(self, value):
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            write_pending(journal_path(self._resolve_pos_file()), self._binding, value)
        except Exception:
            self._journal_failed = True
            raise

    def _pending_any(self):
        for symbol in symbols():
            if read_pending(journal_path(self._root / (symbol + '.json')), self._binding) is not None:
                return True
        return False

    def _holdings(self, balance):
        if not isinstance(balance, dict) or balance.get('ok') is not True or not isinstance(balance.get('holdings'), list):
            raise RuntimeError('Unverified broker holdings')
        result = {}
        for row in balance['holdings']:
            symbol = str(row.get('symbol') or row.get('stk_cd') or '').upper().strip()
            qty = self._parse_int_strict(row.get('quantity', row.get('poss_qty')))
            if qty is None or qty < 0 or (qty and (not symbol or symbol in result)):
                raise RuntimeError('Ambiguous broker holdings')
            if qty:
                average = float(str(row.get('purchase_price') or row.get('avg_price') or
                                    row.get('pchs_avg_pric') or row.get('frgn_stk_book_uv') or 0).replace(',', ''))
                if not math.isfinite(average) or average <= 0:
                    raise RuntimeError('Missing broker entry price')
                result[symbol] = (qty, average)
        return result

    def _orders_empty(self):
        value = self.broker.get_open_orders()
        rows = self._extract_open_orders_list(value)
        if not isinstance(value, dict) or value.get('ok') is not True or value.get('complete') is False or rows is None:
            return False
        for row in rows:
            qty = self._parse_int_strict(row.get('nccs_qty', row.get('remaining_quantity', row.get('ord_remnq'))))
            if qty is None or qty > 0:
                return False
        return True

    def _verify_pre_buy_ledger_clean(self, symbol):
        if (self._legacy_present or self._journal_failed or getattr(self, '_position_write_failures', None) or self.book.failed or self._mismatch or
                symbol in self._positions or self._pending_any() or not self._orders_empty()):
            return False
        holdings = self._holdings(self.broker.get_overseas_stock_balance(force_refresh=True))
        if symbol in holdings or any(s not in self._positions for s in holdings):
            return False
        if any(holdings.get(s, (0, 0))[0] != int(p['quantity']) for s, p in self._positions.items()):
            return False
        if any(not math.isclose(holdings[s][1], float(p['price'])) for s,p in self._positions.items()):
            return False
        # Only the current durable BUY operation may coexist with another leg.
        return all(s == symbol and op['action'] == 'BUY' for s, op in self.book.data['operations'].items())

    def _entry_size_blocked(self, reason):
        self._entry_block_detail = reason
        return 0

    def _entry_quantity(self, usd_available, price):
        self._entry_block_detail = None
        if self._decision is None:
            return self._entry_size_blocked('COMPLETED_15M_DECISION_UNAVAILABLE')
        if not self._can_enter_now():
            return self._entry_size_blocked('RELAY_ENTRY_SAFETY_GUARD')
        if self._get_active_position() is not None:
            return self._entry_size_blocked('RELAY_CYCLE_ACTIVE')
        if any(self._quote(s) is None for s in symbols()):
            return self._entry_size_blocked('FRESH_WS_QUOTES_REQUIRED')
        if self._decision != self._decision_key(self._now()):
            return self._entry_size_blocked('STALE_15M_DECISION')
        first_closed = self._now().replace(hour=int(config.PHASE_MAIN_START.split(':')[0]),
            minute=int(config.PHASE_MAIN_START.split(':')[1]), second=0, microsecond=0)
        if self._now() < first_closed + timedelta(minutes=self._decision_interval()):
            return self._entry_size_blocked('FIRST_15M_CANDLE_NOT_CLOSED')
        with self.book.lock:
            if not self.book.begin(float(usd_available), self._decision):
                reason = ('WAIT_NEXT_COMPLETED_15M' if self._decision in
                          (self.book.data['last_decision'], self.book.data['flat_after']) else
                          'RELAY_CYCLE_RESERVED_OR_PENDING')
                return self._entry_size_blocked(reason)
            quantity = self.book.quantity(float(usd_available), price, 'INITIAL')
        if quantity <= 0:
            return self._entry_size_blocked('INSUFFICIENT_FUNDS')
        return quantity

    def _prepare_decision_candles(self, candles, ny_dt, is_regular_market):
        """DB timestamps are NY candle opens; require the just-completed bar."""
        if not is_regular_market:
            return candles
        import pandas as pd
        interval = self._decision_interval()
        self._decision = None
        self._candle_error = 'LATEST_COMPLETED_15M_CANDLE_MISSING'
        self._candle_last_time = None
        open_hour, open_minute = map(int, config.PHASE_MAIN_START.split(':'))
        first_close = ny_dt.replace(hour=open_hour, minute=open_minute, second=0, microsecond=0)
        first_close += timedelta(minutes=interval)
        if ny_dt < first_close:
            self._candle_error = 'FIRST_15M_CANDLE_NOT_CLOSED'
            return candles.iloc[:0] if candles is not None else candles
        if candles is None or candles.empty:
            return candles
        try:
            if isinstance(candles.index, pd.DatetimeIndex):
                stamps = candles.index
            elif 'datetime' in candles.columns:
                stamps = pd.DatetimeIndex(pd.to_datetime(candles['datetime'], errors='raise'))
            else:
                self._candle_error = 'INVALID_15M_CANDLE_SEQUENCE'
                return candles.iloc[:0]
            if stamps.hasnans or stamps.has_duplicates or not stamps.is_monotonic_increasing:
                self._candle_error = 'INVALID_15M_CANDLE_SEQUENCE'
                return candles.iloc[:0]
            stamps = stamps.tz_localize('America/New_York') if stamps.tz is None else stamps.tz_convert('America/New_York')
            current = datetime.fromisoformat(self._decision_key(ny_dt))
            expected = current - timedelta(minutes=interval)
            completed = candles.loc[stamps <= expected].copy()
            closed_stamps = stamps[stamps <= expected]
            if len(closed_stamps):
                self._candle_last_time = closed_stamps[-1].isoformat()
            if len(closed_stamps) and closed_stamps[-1] == expected:
                self._decision = current.isoformat()
                self._candle_error = None
            return completed
        except (TypeError, ValueError):
            self._candle_error = 'INVALID_15M_CANDLE_SEQUENCE'
            return candles.iloc[:0]

    def _load_decision_candles(self, ny_dt, is_regular_market, sync_intraday):
        if not is_regular_market or not sync_intraday:
            if getattr(self, '_soxx_orb_enabled', False):
                if sync_intraday:
                    self.data_lake.sync_live_intraday_candles(config.ALL_SYMBOLS)
                candles = self.data_lake.load_candles(self._candle_symbol(), self._candle_timeframe())
                return self._prepare_decision_candles(candles, ny_dt, is_regular_market)
            return super()._load_decision_candles(ny_dt, is_regular_market, sync_intraday)
        c_sym, c_tf = self._candle_symbol(), self._candle_timeframe()
        interval = self._decision_interval()
        candles = self.data_lake.load_candles(c_sym, c_tf)
        prepared = self._prepare_decision_candles(candles, ny_dt, True)
        if self._candle_error == 'FIRST_15M_CANDLE_NOT_CLOSED':
            return prepared
        # Only the evaluation thread waits. RelayRisk and the WS callback keep
        # monitoring real-time prices; no quote/order/ledger lock is held here.
        started = time.monotonic()
        deadline = started + config.CANDLE_READY_TIMEOUT_SECONDS
        attempts = 0
        cycle = self._decision_key(ny_dt)
        while not self._stop.is_set() and time.monotonic() < deadline:
            if self._decision_key(self._now()) != cycle:
                self._decision = None
                self._candle_error = 'STALE_15M_DECISION'
                break
            attempts += 1
            self.data_lake.sync_live_intraday_candles(config.ALL_SYMBOLS, deadline=deadline)
            candles = self.data_lake.load_candles(c_sym, c_tf)
            prepared = self._prepare_decision_candles(candles, ny_dt, True)
            if self._candle_error != 'LATEST_COMPLETED_15M_CANDLE_MISSING':
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0 or self._stop.wait(min(config.CANDLE_READY_RETRY_SECONDS, remaining)):
                break
        if self._stop.is_set() or time.monotonic() >= deadline or self._decision_key(self._now()) != cycle:
            self._decision = None
            self._candle_error = ('CANDLE_WAIT_STOPPED' if self._stop.is_set() else
                                  'STALE_15M_DECISION' if self._decision_key(self._now()) != cycle else
                                  'LATEST_COMPLETED_15M_CANDLE_MISSING')
        system_logger.log('WARNING' if self._candle_error else 'INFO', 'CandleReadiness',
            f"cycle={cycle} symbol={c_sym} tf={c_tf} attempts={attempts} elapsed={time.monotonic()-started:.3f}s "
            f"expected={(datetime.fromisoformat(cycle) - timedelta(minutes=interval)).isoformat()} "
            f"last={self._candle_last_time} "
            f"result={self._candle_error or 'READY'}")
        return prepared

    def _evaluate_soxx_orb_5m_signal(self, df_candle_15m, live_prices=None, **kwargs):
        """Evaluate SOXX 5-minute ORB (09:30~10:00, >=0.45%) + Intraday VWAP + 20-bar RVOL (>=1.5)."""
        import pandas as pd
        now_ny = self._now()
        df = df_candle_15m
        if df is None or df.empty:
            df = self.data_lake.load_candles('SOXX', '5m')
        if df is None or df.empty:
            return dict(
                selected_expert='soxx_orb_5m',
                expert_desc='SOXX_5M_NO_CANDLES',
                gating_confidence=0.0,
                expert_confidence=0.0,
                threshold_applied=float(config.GBDT_CONFIDENCE_THRESHOLD),
                direction='NONE',
                dir_gbdt='NONE',
                cross_dir='NONE',
                is_approved=False,
                is_cross_veto=False,
                is_60m_trend_ok=True,
                dip_ok=True,
                all_gating_confidences={'soxx_orb_5m': 0.0},
                gbdt_probs={'LONG': 0.0, 'SHORT': 0.0, 'NONE': 1.0},
                features={},
                active_model_id='SOXX_ORB_VWAP_RVOL',
                active_model_sha256='SOXX_5M_ORB_VWAP_RVOL_RELAY_V1',
                active_model_version='SOXX_5M_ORB_VWAP_RVOL_RELAY_V1',
                active_model_schema='SOXX_5M_ORB_VWAP_RVOL',
                active_model_num_features=3,
                active_model_load_status='OK',
                active_model_fallback_status='PRIMARY',
                active_model_is_fallback=False,
                model_hot_reloaded=False,
                model_consistency_ok=True,
                model_consistency_reason='CONSISTENT',
                timestamp=now_ny.strftime('%Y-%m-%d %H:%M:%S'),
            )

        work = df.copy()
        if isinstance(work.index, pd.DatetimeIndex):
            idx = work.index
        elif 'datetime' in work.columns:
            idx = pd.DatetimeIndex(pd.to_datetime(work['datetime'], errors='coerce'))
            work.index = idx
        else:
            idx = pd.DatetimeIndex(pd.to_datetime(work.index, errors='coerce'))
            work.index = idx
        if idx.tz is None:
            work.index = work.index.tz_localize('America/New_York')
        else:
            work.index = work.index.tz_convert('America/New_York')
        work = work.sort_index()

        col_map = {str(c).lower(): c for c in work.columns}
        o_col = col_map.get('open', 'Open')
        h_col = col_map.get('high', 'High')
        l_col = col_map.get('low', 'Low')
        c_col = col_map.get('close', 'Close')
        v_col = col_map.get('volume', 'Volume')

        hms = work.index.strftime('%H:%M')
        reg_mask = (hms >= '09:30') & (hms < '16:00') & (work.index.weekday < 5)
        reg = work.loc[reg_mask]
        if reg.empty:
            reg = work

        last_ts = reg.index[-1]
        last_row = reg.iloc[-1]
        bar_date = last_ts.date()
        bar_hm = last_ts.strftime('%H:%M')
        open_px = float(last_row[o_col])
        high_px = float(last_row[h_col])
        low_px = float(last_row[l_col])
        close_px = float(last_row[c_col])
        vol_val = float(last_row[v_col])

        # 1. RVOL: current 5m SOXX volume / preceding 20 regular-session 5m SOXX volumes SMA_20
        prev_vols = reg[v_col].iloc[:-1].tail(20).astype(float).tolist()
        if len(prev_vols) >= 20:
            sma_20_vol = sum(prev_vols) / 20.0
            rvol = (vol_val / sma_20_vol) if sma_20_vol > 0 else 0.0
        else:
            sma_20_vol = 0.0
            rvol = 0.0

        # 2. Intraday cumulative VWAP from 09:30 on bar_date
        day_bars = reg.loc[reg.index.date == bar_date]
        tp_series = (day_bars[h_col].astype(float) + day_bars[l_col].astype(float) + day_bars[c_col].astype(float)) / 3.0
        vol_series = day_bars[v_col].astype(float)
        cum_vol = float(vol_series.sum())
        vwap = float((tp_series * vol_series).sum() / cum_vol) if cum_vol > 0 else float((high_px + low_px + close_px) / 3.0)

        # 3. Opening Range (09:30 ~ 10:00, first 6 five-minute bars: 09:30..09:55)
        orb_hms = ('09:30', '09:35', '09:40', '09:45', '09:50', '09:55')
        day_hms = day_bars.index.strftime('%H:%M')
        orb_df = day_bars.loc[day_hms.isin(orb_hms)]
        bar_0930 = day_bars.loc[day_hms == '09:30']
        soxx_open_0930 = float(bar_0930[o_col].iloc[0]) if not bar_0930.empty else (
            float(orb_df[o_col].iloc[0]) if not orb_df.empty else None
        )
        orb_high = float(orb_df[h_col].max()) if not orb_df.empty else None
        orb_low = float(orb_df[l_col].min()) if not orb_df.empty else None
        orb_range = (orb_high - orb_low) if (orb_high is not None and orb_low is not None) else None
        unique_orb_hms = set(orb_df.index.strftime('%H:%M'))
        orb_ready = (
            len(unique_orb_hms) == 6
            and soxx_open_0930 is not None
            and soxx_open_0930 > 0
            and orb_range is not None
        )
        orb_range_ratio = (orb_range / soxx_open_0930) if (orb_ready and soxx_open_0930) else 0.0
        day_trade_enabled = bool(orb_ready and orb_range_ratio >= (0.0045 - 1e-12))

        date_key = str(bar_date)
        logged_dates = getattr(self, '_soxx_orb_logged_dates', set())
        if orb_ready and date_key not in logged_dates:
            logged_dates.add(date_key)
            self._soxx_orb_logged_dates = logged_dates
            self._write_soxx_orb_log(
                f"[{last_ts.strftime('%Y-%m-%d %H:%M:%S')}][ORB_READY] "
                f"open_0930={soxx_open_0930:.4f} orb_high={orb_high:.4f} "
                f"orb_low={orb_low:.4f} orb_range={orb_range:.4f} ({orb_range_ratio*100.0:.3f}%) "
                f"enabled={day_trade_enabled}"
            )
            self._append_soxx_orb_event(
                'SOXX_ORB_READY',
                date=date_key,
                bar_time=last_ts.isoformat(),
                soxx_open_0930=soxx_open_0930,
                orb_high=orb_high,
                orb_low=orb_low,
                orb_range=orb_range,
                orb_range_pct=orb_range_ratio * 100.0,
                day_trade_enabled=day_trade_enabled,
            )

        daily_trades = int(getattr(self, '_soxx_orb_daily_trades', {}).get(date_key, 0))
        in_entry_window = ('10:00' <= bar_hm <= '14:30') and (now_ny.strftime('%H:%M') < config.PHASE_EOD_CLEAR)
        long_signal = False
        short_signal = False
        signal_reason = 'WAIT_HOLD'

        if not orb_ready:
            signal_reason = f'ORB_FORMING({len(unique_orb_hms)}/6)'
        elif not day_trade_enabled:
            signal_reason = f'ORB_RANGE_FILTER_BLOCKED({orb_range_ratio*100.0:.3f}%<0.45%)'
        elif not in_entry_window:
            signal_reason = f'OUTSIDE_ENTRY_WINDOW(bar={bar_hm})'
        elif daily_trades >= 100:
            signal_reason = 'DAILY_TRADE_LIMIT_REACHED(100)'
        else:
            long_signal = bool((close_px > orb_high) and (close_px > vwap) and (rvol >= 1.5 - 1e-12))
            short_signal = bool((close_px < orb_low) and (close_px < vwap) and (rvol >= 1.5 - 1e-12))
            if long_signal:
                signal_reason = f'SOXX_LONG_BREAKOUT(close={close_px:.2f}>orb_high={orb_high:.2f},vwap={vwap:.2f},rvol={rvol:.2f})'
            elif short_signal:
                signal_reason = f'SOXX_SHORT_BREAKDOWN(close={close_px:.2f}<orb_low={orb_low:.2f},vwap={vwap:.2f},rvol={rvol:.2f})'
            else:
                signal_reason = f'NO_BREAKOUT_OR_RVOL(close={close_px:.2f},vwap={vwap:.2f},rvol={rvol:.2f})'

        if long_signal:
            direction = 'LONG'
            is_approved = True
            conf = 1.0
            probs = {'LONG': 1.0, 'SHORT': 0.0, 'NONE': 0.0}
        elif short_signal:
            direction = 'SHORT'
            is_approved = True
            conf = 1.0
            probs = {'LONG': 0.0, 'SHORT': 1.0, 'NONE': 0.0}
        else:
            direction = 'NONE'
            is_approved = False
            conf = 0.0
            probs = {'LONG': 0.0, 'SHORT': 0.0, 'NONE': 1.0}

        if is_approved:
            target_sym = config.BASE_ASSET_LONG if direction == 'LONG' else config.BASE_ASSET_SHORT
            self._write_soxx_orb_log(
                f"[{now_ny.strftime('%Y-%m-%d %H:%M:%S')}][ENTRY_SIGNAL] "
                f"bar={bar_hm} dir={direction} exec_ticker={target_sym} "
                f"soxx_close={close_px:.4f} orb_high={orb_high:.4f} orb_low={orb_low:.4f} "
                f"vwap={vwap:.4f} rvol={rvol:.2f}"
            )
            self._append_soxx_orb_event(
                'SOXX_5M_ENTRY_SIGNAL',
                bar_time=last_ts.isoformat(),
                direction=direction,
                symbol=target_sym,
                soxx_close=close_px,
                orb_high=orb_high,
                orb_low=orb_low,
                orb_range=orb_range,
                vwap=vwap,
                rvol=rvol,
                signal_reason=signal_reason,
            )
        else:
            self._append_soxx_orb_event(
                'SOXX_5M_NO_SIGNAL',
                bar_time=last_ts.isoformat(),
                soxx_close=close_px,
                orb_high=orb_high,
                orb_low=orb_low,
                orb_range=orb_range,
                vwap=vwap,
                rvol=rvol,
                signal_reason=signal_reason,
            )

        system_logger.log(
            'TRADE' if is_approved else 'INFO',
            'SOXX_5M_ORB',
            f"[SOXX 5m] bar={bar_hm} close={close_px:.2f} orb={orb_low or 0.0:.2f}~{orb_high or 0.0:.2f} "
            f"vwap={vwap:.2f} rvol={rvol:.2f} -> dir={direction} approved={is_approved} ({signal_reason})"
        )

        indicators = dict(
            bar_time=last_ts.isoformat(),
            bar_hm=bar_hm,
            soxx_open=open_px,
            soxx_high=high_px,
            soxx_low=low_px,
            soxx_close=close_px,
            soxx_volume=vol_val,
            soxx_open_0930=soxx_open_0930,
            orb_high=orb_high,
            orb_low=orb_low,
            orb_range=orb_range,
            orb_range_pct=orb_range_ratio * 100.0,
            orb_ready=orb_ready,
            day_trade_enabled=day_trade_enabled,
            vwap=vwap,
            rvol=rvol,
            sma_20_vol=sma_20_vol,
            signal_reason=signal_reason,
        )
        return dict(
            selected_expert='soxx_orb_5m',
            expert_desc=f"SOXX 5m ORB+VWAP+RVOL ({signal_reason})",
            gating_confidence=conf,
            expert_confidence=conf,
            threshold_applied=float(config.GBDT_CONFIDENCE_THRESHOLD),
            regime_snapshot=indicators,
            soxx_5m_indicators=indicators,
            direction=direction,
            cross_dir=direction,
            dir_gbdt=direction,
            is_approved=is_approved,
            is_cross_veto=False,
            is_60m_trend_ok=True,
            dip_ok=True,
            all_gating_confidences={'soxx_orb_5m': conf},
            gbdt_probs=probs,
            features={'ATR_14': close_px * 0.018, 'SOXX_Close': close_px, 'SOXX_VWAP': vwap, 'SOXX_RVOL': rvol},
            details={'ml_reason': signal_reason},
            active_model_id='SOXX_ORB_VWAP_RVOL',
            active_model_sha256='SOXX_5M_ORB_VWAP_RVOL_RELAY_V1',
            active_model_version='SOXX_5M_ORB_VWAP_RVOL_RELAY_V1',
            active_model_schema='SOXX_5M_ORB_VWAP_RVOL',
            active_model_num_features=3,
            active_model_load_status='OK',
            active_model_fallback_status='PRIMARY',
            active_model_is_fallback=False,
            model_hot_reloaded=False,
            model_consistency_ok=True,
            model_consistency_reason='CONSISTENT',
            timestamp=now_ny.strftime('%Y-%m-%d %H:%M:%S'),
        )

    def _collect_cycle_market_snapshot(self, *args, **kwargs):
        snapshot = super()._collect_cycle_market_snapshot(*args, **kwargs)
        if getattr(self, '_candle_error', None):
            snapshot.update(data_stale=True, stale_reason=self._candle_error)
        return snapshot

    def evaluate_15m_cycle_once(self, *args, **kwargs):
        self._decision = None
        self._candle_error = None
        # The base evaluator retains all model/Neutral/feature/shield interlocks.
        try:
            return super().evaluate_15m_cycle_once(*args, **kwargs)
        finally:
            self.book.release_initial()
            self._wake.set()

    def _execute_buy_with_10s_chase(self, symbol, target_qty, ref_price, moe_res, targets,
                                    role='INITIAL', trigger_symbol=None):
        self._buy_execution_outcome = None
        self._buy_execution_attempts = None
        if not self._entry_mutex.acquire(blocking=False):
            return False
        try:
            with self._symbol(symbol, 'BUY'):
                if not self._can_enter_now() or not self._verify_pre_buy_ledger_clean(symbol):
                    return False
                if role == 'CHAIN':
                    if (not self._chain_available() or not self._chain_condition(trigger_symbol)):
                        return False
                elif role != 'INITIAL' or self._positions or self.book.data['operations']:
                    return False
                cash = self.broker.test_connection()
                if cash.get('ok') is not True or cash.get('deposit_ok') is False:
                    return False
                quote = self._quote(symbol)
                if quote is None:
                    return False
                available = float(cash.get('usd_order_available', 0))
                qty = min(target_qty, self.book.quantity(available, quote['ask'], role))
                if qty <= 0:
                    return False
                operation = dict(id=uuid.uuid4().hex, action='BUY', role=role, trigger_symbol=trigger_symbol,
                                 decision_key=self._decision if role == 'INITIAL' else None,
                                 quantity=qty, held_before=0, created_ny=self._now().isoformat(),
                                 budget_cap=min(self.book.data['budget'] * (
                                     config.RELAY_INITIAL_ALLOCATION if role == 'INITIAL' else config.RELAY_CHAIN_ALLOCATION),
                                     max(0., available - self.book.data['budget'] * config.RELAY_CASH_BUFFER)),
                                 orders=[], submission_pending=False)
                self.book.operation(symbol, operation)
                try:
                    targets = dict(targets, strategy_tag=config.LIVE_EXECUTION_STRATEGY)
                    source = self._positions.get(trigger_symbol, {})
                    trigger_quote = self._quote(trigger_symbol) if trigger_symbol else None
                    # Notification context only; preserve the original model's
                    # decision and learning snapshot without mutating its dict.
                    moe_res = dict(moe_res, relay_role=role, relay_cycle_id=self.book.data['cycle_id'],
                                   relay_trigger_symbol=trigger_symbol,
                                   relay_source_entry=source.get('price'),
                                   relay_trigger_bid=trigger_quote['bid'] if trigger_quote else None)
                    if getattr(self, '_soxx_orb_enabled', False) and role == 'CHAIN':
                        src_px = float(source.get('price') or 0.0)
                        trig_bid = float(trigger_quote['bid']) if trigger_quote else 0.0
                        ret_pct = ((trig_bid - src_px) / src_px * 100.0) if src_px > 0 else 0.0
                        self._write_soxx_orb_log(
                            f"[{self._now().strftime('%Y-%m-%d %H:%M:%S')}][CHAIN_TRIGGER] "
                            f"source={trigger_symbol} entry={src_px:.4f} cur={trig_bid:.4f} ({ret_pct:.2f}%) -> "
                            f"chain_buy={symbol} (49% allocation)"
                        )
                        self._append_soxx_orb_event(
                            'LIVE_CHAIN_TRIGGER',
                            source_symbol=trigger_symbol,
                            symbol=symbol,
                            source_entry=src_px,
                            trigger_bid=trig_bid,
                            source_return_pct=ret_pct,
                        )
                    result = super()._execute_buy_with_10s_chase(symbol, qty, quote['ask'], moe_res, targets)
                    if result and getattr(self, '_soxx_orb_enabled', False):
                        d_key = str(self._now().date())
                        trades_map = getattr(self, '_soxx_orb_daily_trades', {})
                        trades_map[d_key] = int(trades_map.get(d_key, 0)) + 1
                        self._soxx_orb_daily_trades = trades_map
                        pos_now = self._positions.get(symbol, {})
                        self._write_soxx_orb_log(
                            f"[{self._now().strftime('%Y-%m-%d %H:%M:%S')}][LIVE_BUY] "
                            f"role={role} symbol={symbol} price={float(pos_now.get('price', quote['ask'])):.4f} "
                            f"qty={int(pos_now.get('quantity', qty))} daily_trades={trades_map[d_key]}"
                        )
                        self._append_soxx_orb_event(
                            'LIVE_BUY_FILLED',
                            role=role,
                            symbol=symbol,
                            price=float(pos_now.get('price', quote['ask'])),
                            quantity=int(pos_now.get('quantity', qty)),
                            daily_trade_count=trades_map[d_key],
                        )
                finally:
                    self._reconcile_active_position_with_ledger(self.broker.get_overseas_stock_balance(force_refresh=True))
                return result
        finally:
            try:
                if role == 'INITIAL':
                    self.book.release_initial()
                    self._wake.set()
            finally:
                self._entry_mutex.release()

    def _execute_sell_with_10s_chase(self, symbol, quantity, reason_desc, **kwargs):
        if not kwargs.get('_already_locked'):
            if not self._acquire_order_lock():
                return False
            kwargs['_already_locked'] = True
        entered_helper = False
        try:
            with self._symbol(symbol, 'SELL'):
                if symbol in self.book.data['operations']:
                    return False
                position = self._get_active_position()
                # A missing local position must not defeat broker-based EOD liquidation.
                holdings = self._holdings(self.broker.get_overseas_stock_balance(force_refresh=True))
                held, average = holdings.get(symbol, (0, 0.))
                recovery = position is None and self._now().strftime('%H:%M') >= config.PHASE_EOD_CLEAR
                if not held or (position is None and not recovery):
                    return False
                operation = dict(id=uuid.uuid4().hex, action='SELL',
                                 role='RECOVERY' if recovery else position['relay_role'],
                                 quantity=held, held_before=held,
                                 created_ny=self._now().isoformat(), orders=[], submission_pending=False,
                                 reason=reason_desc)
                with self._quote_lock:
                    hit = self._risk_hits.get(symbol)
                if position and hit and hit[0] == position['relay_leg_id']:
                    operation['risk_observation'] = hit
                self.book.operation(symbol, operation)
                if recovery:
                    kwargs.update(is_market_order=True, buy_px=average)
                entered_helper = True
                try:
                    sold_res = super()._execute_sell_with_10s_chase(symbol, held, reason_desc, **kwargs)
                    if sold_res and getattr(self, '_soxx_orb_enabled', False):
                        self._write_soxx_orb_log(
                            f"[{self._now().strftime('%Y-%m-%d %H:%M:%S')}][LIVE_SELL] "
                            f"role={operation['role']} symbol={symbol} qty={held} entry={average:.4f} "
                            f"reason={reason_desc}"
                        )
                        self._append_soxx_orb_event(
                            'LIVE_SELL_FILLED',
                            role=operation['role'],
                            symbol=symbol,
                            quantity=held,
                            entry_price=average,
                            reason=reason_desc,
                        )
                    return sold_res
                finally:
                    self._reconcile_active_position_with_ledger(self.broker.get_overseas_stock_balance(force_refresh=True))
        finally:
            if not entered_helper:
                with self._order_lock:
                    self._is_order_in_progress = False
            self._wake.set()

    _execute_buy_with_chase = _execute_buy_with_10s_chase
    _execute_sell_with_chase = _execute_sell_with_10s_chase

    def _reconcile_active_position_with_ledger(self, balance):
        with self._relay_lock:
            holdings = self._holdings(balance)
            self._mismatch = None
            for symbol in symbols():
                with self._symbol(symbol):
                    position = self._get_active_position()
                    operation = copy.deepcopy(self.book.data['operations'].get(symbol))
                    qty, average = holdings.get(symbol, (0, 0.))
                    if operation and not self._is_order_in_progress:
                        if operation.get('submission_pending') or not operation['orders']:
                            if not operation.get('submission_pending') and not self._load_pending_buy_order():
                                self.book.operation(symbol, None)
                                operation = None
                            else:
                                self._mismatch = 'UNRESOLVED_SUBMISSION'
                                continue
                        if operation:
                            fills = 0
                            zero_cancelled = True
                            for order in operation['orders']:
                                evidence = self.broker.resolve_order_ledger_status(
                                    order['order_no'], symbol, operation['action'], expected_qty=order['quantity'],
                                    trade_date=operation['created_ny'][:10].replace('-', ''))
                                if evidence.get('status') not in ('FILLED', 'CANCELED_PARTIAL_FILL', 'CANCELED_ZERO_FILL'):
                                    self._mismatch = 'UNRESOLVED_ORDER_LEDGER'
                                    break
                                fills += int(evidence['filled_qty'])
                                zero_cancelled = zero_cancelled and evidence['status'] == 'CANCELED_ZERO_FILL'
                            else:
                                expected = fills if operation['action'] == 'BUY' else operation['held_before'] - fills
                                if expected != qty or expected < 0:
                                    self._mismatch = 'RELAY_FILL_HOLDING_MISMATCH'
                                    continue
                                if operation['action'] == 'BUY' and qty:
                                    if position is None:
                                        position = dict(symbol=symbol, quantity=qty, qty=qty, price=average,
                                                        buy_price=average, buy_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                                        order_no=operation['orders'][0]['order_no'],
                                                        high_price_during_hold=average, low_price_during_hold=average,
                                                        relay_recovered=True)
                                    position.update(quantity=qty, qty=qty, price=average, buy_price=average)
                                    self._save_active_position(position)
                                if operation['action'] == 'BUY':
                                    if not qty and not fills and operation['orders'] and zero_cancelled:
                                        self._buy_execution_outcome = 'BUY_UNFILLED_CANCELLED'
                                        self._buy_execution_attempts = len(operation['orders'])
                                    self._clear_pending_buy_order()
                                elif qty and position:
                                    # Only ledger-proven fills may change tracked quantity.
                                    position.update(quantity=qty, qty=qty)
                                    self._save_active_position(position)
                                if not qty and position:
                                    self._resolve_pos_file().unlink()
                                    self._positions.pop(symbol, None)
                                    system_logger.log('TRADE', 'Relay', f'Confirmed closed relay leg {symbol}')
                                self.book.event('BROKER_LEDGERS_RECONCILED', symbol=symbol,
                                    leg_id=operation['id'], role=operation['role'], action=operation['action'],
                                    filled_quantity=fills, remaining_holding=qty, broker_average_entry=average,
                                    entry_price=(position or {}).get('price'), orders=operation['orders'],
                                    exit_price=None, exit_price_source='NOT_AVAILABLE_FROM_QUANTITY_REDUCER')
                                self.book.operation(symbol, None)
                                operation = None
                    position = self._get_active_position()
                    if qty and position:
                        if not operation and (qty != int(position['quantity']) or
                                not math.isclose(average, float(position['price']))):
                            self._mismatch = 'EXTERNALLY_CHANGED_TRACKED_HOLDING'
                    elif qty or (position and not operation):
                        self._mismatch = 'UNKNOWN_OR_EXTERNALLY_CHANGED_HOLDING'
            if any(s not in symbols() for s in holdings):
                self._mismatch = 'UNMANAGED_BROKER_HOLDING'
            # Atomic with initial quantity reservation: empty holdings alone
            # cannot complete a cycle while its initial BUY is preparing.
            with self.book.lock:
                if (not holdings and not self._positions and not self.book.data['operations'] and
                        not self._pending_any() and not self.book.initial_preparing(self._decision_key(self._now())) and
                        self.book.data['budget'] is not None):
                    completed_cycle = self.book.data['cycle_id']
                    self.book.update(budget=None, cycle_id=None, flat_after=self._decision_key(self._now()),
                                     initial_reservation_owner=None)
                    if getattr(self, '_soxx_orb_enabled', False):
                        self._write_soxx_orb_log(
                            f"[{self._now().strftime('%Y-%m-%d %H:%M:%S')}][CYCLE_CLOSED] "
                            f"cycle_id={completed_cycle} status=ALL_POSITIONS_EMPTY_READY_FOR_NEXT_INITIAL"
                        )
                        self._append_soxx_orb_event('LIVE_CYCLE_CLOSED', completed_cycle_id=completed_cycle)
                    try:
                        if self._orders_empty():
                            self.notifier.send_relay_cycle_complete_alert(
                                cycle_id=completed_cycle, is_simulation=self.broker.is_simulation)
                        else:
                            logger.warning('Relay cycle notification withheld: broker open orders not confirmed empty')
                    except Exception:
                        logger.warning('Relay cycle notification failed; reconciled trading state preserved')
            self._unreconciled_holding_mismatch = {'reason': self._mismatch} if self._mismatch else None
            self._recover_position_writes(holdings)

    def _chain_condition(self, symbol):
        with self._relay_lock:
            position = self._positions.get(symbol)
            q = self._quote(symbol)
            return bool(position and q and q['bid'] <= float(position['price']) *
                        (1 - config.RELAY_OPPOSITE_TRIGGER_PCT))

    def _chain_available(self):
        # An older active journal without the cycle reservation is ambiguous:
        # preserve its SELL management, but refuse another BUY until flat.
        with self._relay_lock:
            if config.RELAY_CHAIN_ALLOCATION <= 0:
                return False
            if self.book.data.get('cycle_id') is None:
                return False
            if 'chain_operation_id' not in self.book.data:
                return False
            chain_ops = self.book.data.get('chain_operations')
            if chain_ops is None:
                chain_ops = [self.book.data['chain_operation_id']] if self.book.data.get('chain_operation_id') else []
            max_chains = 0 if getattr(self, '_soxx_orb_enabled', False) else config.RELAY_MAX_CHAIN_COUNT
            if len(chain_ops) >= max_chains:
                return False
            if len(self._positions) != 1:
                return False
            if getattr(self, '_soxx_orb_enabled', False):
                sole_pos = next(iter(self._positions.values()))
                if sole_pos.get('relay_role') != 'INITIAL':
                    return False
            return True

    def _manage_open_positions(self, balance, realtime_px_override=None):
        if not self._risk_mutex.acquire(blocking=False):
            return
        try:
            holdings = self._holdings(balance)
            now = self._now()
            for symbol, (qty, average) in holdings.items():
                if symbol not in symbols():
                    continue
                with self._symbol(symbol):
                    position = self._get_active_position()
                    if position is None:
                        if now.strftime('%H:%M') >= config.PHASE_EOD_CLEAR and symbol not in self.book.data['operations']:
                            self._dispatch_sell(symbol=symbol, quantity=qty, buy_px=average,
                                                reason_desc='Relay EOD untracked holding', is_market_order=True)
                        continue
                    quote = self._quote(symbol)
                    bid = quote['bid'] if quote else None
                    if bid is not None:
                        with self._quote_lock:
                            observed = self._extremes.get(symbol)
                        high, low = (observed[1], observed[2]) if observed and observed[0] == position['relay_leg_id'] else (bid, bid)
                        position['high_price_during_hold'] = max(float(position.get('high_price_during_hold', average)), high)
                        position['low_price_during_hold'] = min(float(position.get('low_price_during_hold', average)), low)
                        if not getattr(self, '_position_write_failures', None):
                            try:
                                self._save_active_position(position)
                            except OSError:
                                # A price metadata write failure must not skip SL/TP/EOD.
                                # The failed symbol stays blocked for BUY until reconciliation.
                                pass
                    poll = time.monotonic() - self._last_poll.get(symbol, 0.) >= config.MAIN_LOOP_TICK_OPEN
                    if poll:
                        self._last_poll[symbol] = time.monotonic()
                    reason = exit_reason(position, bid, now, poll)
                    with self._quote_lock:
                        hit = self._risk_hits.get(symbol)
                    if reason != 'EOD' and hit and hit[0] == position['relay_leg_id']:
                        reason = hit[1]
                    if reason and not self._is_order_in_progress and symbol not in self.book.data['operations']:
                        self._dispatch_sell(symbol=symbol, quantity=qty, buy_px=average, cur_px=bid or 0.,
                                            reason_desc='Relay ' + reason, is_stoploss=reason in ('SL', 'TRAIL'),
                                            is_market_order=reason == 'EOD' or bid is None)
            if (self._is_order_in_progress or self._mismatch or self._journal_failed or
                    getattr(self, '_position_write_failures', None) or self.book.failed):
                return
            if (not self._chain_available() or self.book.data['operations'] or
                    self._pending_any() or not self._can_enter_now()):
                return
            symbol = next(iter(self._positions))
            other = opposite(symbol)
            if other in holdings or not self._chain_condition(symbol) or self._quote(other) is None:
                return
            quote = self._quote(other)
            targets = dict(tp_pct=config.MAX_TP_PCT * 100, dynamic_tp_px=quote['ask'] * (1 + config.MAX_TP_PCT),
                           dynamic_sl_px=quote['ask'] * (1 - config.SL_MIN_PCT), sl_pct=config.SL_MIN_PCT * 100,
                           time_stop_minutes=config.TIME_STOP_MINUTES, strategy_tag=config.LIVE_EXECUTION_STRATEGY)
            self._execute_buy_with_10s_chase(other, self.book.quantity(
                float(self.broker.test_connection().get('usd_order_available', 0)), quote['ask'], 'CHAIN'),
                quote['ask'], dict(gating_confidence=0., expert_desc='PRICE_TRIGGERED_CHAIN'), targets,
                role='CHAIN', trigger_symbol=symbol)
        finally:
            self._risk_mutex.release()

    def _on_websocket_tick(self, symbol, price, extra):
        if not hasattr(self, '_scope'):
            return
        # Decode the same provenance-checked packet format used by the shadow feed.
        quote = ws_quote(extra)
        if symbol in symbols() and quote is not None:
            # Never wait behind broker reconciliation from the WebSocket reader.
            # Preserve every observed high/low even when the order worker is busy.
            with self._quote_lock:
                self._quotes[symbol] = dict(bid=quote[0], ask=quote[1], time=time.time())
                position = self._positions.get(symbol)
                if position:
                    leg_id = position['relay_leg_id']
                    prior = self._extremes.get(symbol)
                    high, low = (prior[1], prior[2]) if prior and prior[0] == leg_id else (quote[0], quote[0])
                    new_high = max(high, quote[0])
                    new_low = min(low, quote[0])
                    self._extremes[symbol] = (leg_id, new_high, new_low)
                    entry_px = float(position['price'])
                    if getattr(self, '_soxx_orb_enabled', False) or position.get('soxx_orb_strategy'):
                        reason = ('SL' if quote[0] <= entry_px * (1 - config.SL_MIN_PCT)
                                  else 'TRAIL' if (new_high >= entry_px * (1 + config.TRAILING_TRIGGER_PCT) and
                                                   quote[0] <= new_high * (1 - config.TRAILING_DROP_PCT))
                                  else None)
                    else:
                        reason = ('SL' if quote[0] <= entry_px * (1 - config.SL_MIN_PCT)
                                  else 'TP' if quote[0] >= entry_px * (1 + config.MAX_TP_PCT) else None)
                    previous = self._risk_hits.get(symbol)
                    if reason and (not previous or previous[0] != leg_id or reason == 'SL'):
                        self._risk_hits[symbol] = (leg_id, reason, quote[0], time.time())
            self._wake.set()
        try:
            self.shadow_tracker.check_ticks(symbol, price)
        except Exception:
            logger.exception('Legacy tracker callback failed')

    def _risk_loop(self):
        while not self._stop.is_set():
            self._wake.wait(config.MAIN_LOOP_TICK_OPEN)
            self._wake.clear()
            try:
                if not USMarketCalendar.get_market_status().get('is_open') or self._is_order_in_progress:
                    continue
                if time.monotonic() - self._balance_at >= config.RELAY_LEDGER_REFRESH_SECONDS:
                    self._balance_cache = self.broker.get_overseas_stock_balance(force_refresh=True)
                    self._reconcile_active_position_with_ledger(self._balance_cache)
                    self._balance_at = time.monotonic()
                self._manage_open_positions(self._balance_cache)
            except Exception as error:
                self._mismatch = 'RELAY_RISK_LOOP_ERROR'
                system_logger.log('ERROR', 'Relay', self._redact_log(str(error)))

    def _market_execution_loop(self):
        worker = threading.Thread(target=self._risk_loop, daemon=True, name='RelayRisk')
        self._risk_thread = worker
        worker.start()
        super()._market_execution_loop()

    def _execution_should_stop(self):
        stopped = self._stop.is_set()
        if not stopped and getattr(self, '_soxx_orb_enabled', False):
            now_ny = self._now()
            if now_ny.minute % 5 == 0:
                key_5m = self._decision_key(now_ny)
                if getattr(self, '_last_5m_trigger_key', None) != key_5m:
                    self._last_5m_trigger_key = key_5m
                    self._last_briefing_time = 0.0
        return stopped

    def run_once_and_start_listener(self, start_background_threads=True):
        if start_background_threads and not self._deployment_ok:
            raise RuntimeError('Relay deployment QA/hash verification failed; entries remain disabled')
        try:
            return super().run_once_and_start_listener(start_background_threads=start_background_threads)
        finally:
            if start_background_threads:
                self._stop.set()
                self._wake.set()
                self.ws_streamer.stop()
                worker = getattr(self, '_risk_thread', None)
                if worker is not None and worker is not threading.current_thread():
                    worker.join(timeout=config.MAIN_LOOP_TICK_OPEN)

    def get_active_model_info(self, is_daemon_running=None):
        info = super().get_active_model_info(is_daemon_running)
        info['execution_strategy'] = config.LIVE_EXECUTION_STRATEGY
        info['execution_strategy_verified'] = self._deployment_ok
        info['execution_account_ready'] = self._account_ready()
        info['legacy_shadow_strategy'] = config.RELAY_LEGACY_SHADOW_ID
        return info
