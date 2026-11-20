"""B72 entry ablations with independent, cash-funded virtual relay portfolios.

No live modules, broker, credentials, sockets or timers. Consumes the existing
validated decision and WebSocket event stream. Fills require a subsequent quote.
"""
from datetime import timedelta
import json
import math

from common import HOME, PROJECT, CONFIG, load_execution_config, sha
from legacy_b72 import LegacyB72
from runtime import Ledger, stamp, BUY_WAIT


def settings():
    value = json.loads((HOME/'relay_directions.json').read_text(encoding='utf-8'))
    if (value['schema_version'] != 1 or value['direction_tie'] != 'LONG' or
            value['neutral_tie'] != 'BLOCK' or value['repeat_chain'] is not False):
        raise ValueError('Unsupported relay shadow recipe')
    return value


def boundary(now):
    minutes = load_execution_config()['RELAY_DECISION_INTERVAL_MINUTES']
    return now.replace(minute=now.minute//minutes*minutes, second=0, microsecond=0)


class DirectionRelay(LegacyB72):
    cohorts = ('primary',)

    def __init__(self, name):
        self.recipe = settings()
        legacy = json.loads((HOME/'legacy_strategy.json').read_text())
        if 'relay_model' in self.recipe:
            legacy.update(self.recipe['relay_model'])
        super().__init__(settings_override=legacy)
        if not self.recipe['enabled']:
            raise ValueError('Relay direction shadows disabled')
        self.mode = self.recipe['strategies'][name]
        if self.mode not in ('KEEP_NEUTRAL', 'IGNORE_NEUTRAL'):
            raise ValueError('Unknown relay direction mode')
        self.name = name
        self.ledger_factory = RelayLedger
        self.version = self.settings['artifact_sha256'] + ':' + sha(HOME/'relay_directions.json')

    def predict(self, packet):
        # Keep using the exact live model. Refuse comparisons after an unobserved
        # model switch rather than silently comparing different classifiers.
        active = json.loads((PROJECT/'models/active_model_manifest.json').read_text(encoding='utf-8'))
        if active['expected_sha256'] != self.settings['artifact_sha256']:
            raise RuntimeError('Relay shadow requires current live B72 artifact')
        result = super().predict(packet)
        ps, pn, pl = result['raw_probabilities']
        if (not all(math.isfinite(p) and 0 <= p <= 1 for p in (ps, pn, pl)) or
                not math.isclose(ps+pn+pl, 1., abs_tol=1e-6)):
            raise ValueError('Invalid B72 probabilities')
        side = 1 if pl >= ps else 0
        reason = 'DIRECTION_ONLY'
        if self.mode == 'KEEP_NEUTRAL' and pn >= max(ps, pl):
            side, reason = -1, 'NEUTRAL_WINNER'
        if stamp(packet['feature_times'][-1]) != stamp(packet['decision_time']):
            side, reason = -1, 'LATEST_CLOSED_CANDLE_MISSING'
        scheduled = stamp(packet['decision_time'])
        if scheduled < scheduled.replace(hour=9, minute=30)+timedelta(minutes=load_execution_config()['RELAY_DECISION_INTERVAL_MINUTES']):
            side, reason = -1, 'FIRST_REGULAR_CANDLE_NOT_CLOSED'
        result.update(model_id=self.name, model_version=self.version, side=side,
                      forced_side=side, execution_strategy='OPPOSITE_RELAY_V1',
                      entry_mode=self.mode, entry_reason=reason, threshold_applied=False,
                      execution_policy_sha256=sha(HOME/'execution_policy.json'))
        return result


class RelayLedger(Ledger):
    """Virtual analogue of the live relay, with explicit cash and cycle P&L.

    QUOTE is observed before MARK by MarketEvents. An intent cannot fill on its
    triggering quote; a later quote rechecks session, source loss and freshness.
    """
    def __init__(self, model_id, cohort='primary', evaluation_window=None):
        super().__init__(model_id, cohort, evaluation_window)
        self.recipe = settings()
        self.cash = float(self.recipe['initial_cash'])
        self.quotes = {}
        self.budget = None
        self.cycle_id = None
        self.cycle_number = 0
        self.cycle_start = None
        self.cycle_closed_start = 0
        self.flat_after = None
        self.chain_filled = False
        self.cycles = []

    def _row(self, event, now, **values):
        return dict(event=event, time=str(now), model_id=self.model_id,
                    cohort=self.cohort, cycle_id=self.cycle_id, **values)

    def _session(self, now):
        return ('09:30' <= now.strftime('%H:%M') < CONFIG['PHASE_MAIN_END']
                and now.weekday() < 5 and self._evaluation_window(now.date()))

    def _quote(self, symbol, now):
        q = self.quotes.get(symbol)
        if q and 0 <= (now-q['observed']).total_seconds() <= self.execution_config['RELAY_QUOTE_MAX_AGE_SECONDS']:
            return q
        return None

    def _finish(self, now):
        if self.budget is None or self.positions or self.pending:
            return []
        row = self._row('VIRTUAL_CYCLE_CLOSED', now, start_time=str(self.cycle_start),
                        starting_equity=self.budget, ending_equity=self.cash,
                        net_pnl=self.cash-self.budget, net_return=self.cash/self.budget-1,
                        legs=len(self.closed)-self.cycle_closed_start)
        self.cycles.append(row)
        self.budget = None
        self.cycle_id = None
        self.flat_after = boundary(now)
        return [row]

    def intend(self, time, side, atr_pct, decision_id, model_version, prediction):
        if side < 0:
            return 'MODEL_NONE'
        if self.positions or self.pending or self.budget is not None:
            return 'POSITION_OPEN_OR_PENDING'
        if self.flat_after is not None and boundary(time) <= self.flat_after:
            return 'WAIT_NEXT_COMPLETED_15M'
        if not self._session(time):
            return 'ENTRY_SESSION_BLOCK'
        if any(self._quote(symbol, time) is None for symbol in
               (CONFIG['BASE_ASSET_LONG'], CONFIG['BASE_ASSET_SHORT'])):
            return 'FRESH_BOTH_WS_QUOTES_REQUIRED'
        self.budget = self.cash
        self.cycle_number += 1
        self.cycle_id = self.model_id+':'+decision_id+':'+str(self.cycle_number)
        self.cycle_start = time
        self.cycle_closed_start = len(self.closed)
        self.chain_filled = False
        self.pending.append(dict(action='BUY', role='INITIAL', time=time, side=side,
                                 decision_id=decision_id, model_version=model_version,
                                 prediction=prediction, decision_boundary=boundary(time)))
        return 'PENDING_NEXT_QUOTE'

    def _source_valid(self, pending, now):
        source = pending.get('source')
        q = self._quote(source['symbol'], now) if source else None
        return bool(source in self.positions and not source['exit_pending'] and q and
                    q['bid'] <= source['entry']*(1-self.execution_config['RELAY_OPPOSITE_TRIGGER_PCT']))

    def _cancel_buys(self, now):
        rows = []
        for pending in list(self.pending):
            if pending['action'] != 'BUY':
                continue
            reason = None
            if (now-pending['time']).total_seconds() > BUY_WAIT:
                reason = 'STALE_INTENT'
            elif not self._session(now):
                reason = 'SESSION'
            elif pending['role'] == 'INITIAL' and boundary(now) != pending['decision_boundary']:
                reason = 'NEW_15M_BOUNDARY'
            if reason:
                self.pending.remove(pending)
                rows.append(self._row('BUY_CANCELLED_'+reason, now, role=pending['role'],
                                      decision_id=pending['decision_id']))
        return rows

    def on_quote(self, event):
        now, symbol = stamp(event['time']), event['symbol']
        bid, ask = float(event['bid']), float(event['ask'])
        if not (math.isfinite(bid) and math.isfinite(ask) and 0 < bid <= ask):
            raise ValueError('Invalid relay bid/ask')
        # Existing recorded-event QA may use quote_time alone. New relay ledgers
        # require the observed WebSocket provenance exported by MarketEvents.
        if not event.get('observed_at') or stamp(event['observed_at']) != stamp(event['quote_time']):
            return []
        observed = stamp(event['quote_time'])
        if not 0 <= (now-observed).total_seconds() <= self.execution_config['RELAY_QUOTE_MAX_AGE_SECONDS']:
            return []
        self.quotes[symbol] = dict(bid=bid, ask=ask, observed=observed)
        rows = self._cancel_buys(now)
        for pending in list(self.pending):
            target = CONFIG['BASE_ASSET_LONG'] if pending['side'] == 1 else CONFIG['BASE_ASSET_SHORT']
            if target != symbol or now <= pending['time'] or observed <= pending['time']:
                continue
            if pending['action'] == 'SELL':
                p = pending['position']
                self.pending.remove(pending)
                if p not in self.positions:
                    raise RuntimeError('Sell without virtual holding')
                gross = bid/p['entry']-1
                cost = max(CONFIG['BACKTEST_FEE_SLIPPAGE'], self.execution_config['RELAY_EXECUTION_COST_RESERVE'])
                net = gross-cost
                self.cash += p['qty']*p['entry']*(1+net)
                self.positions.remove(p)
                row = self._row('VIRTUAL_SELL', now, decision_id=p['decision_id'],
                    model_version=p['model_version'], symbol=symbol, role=p['role'], leg_id=p['leg_id'],
                    entry_time=str(p['entry_time']), exit_time=str(now), entry_price=p['entry'],
                    exit_price=bid, quantity=p['qty'], gross_return=gross, cost_reserve=cost,
                    net_return=net, net_pnl=p['qty']*p['entry']*net, reason=pending['reason'],
                    trigger_price=pending['trigger_price'], trigger_price_basis=pending['price_basis'],
                    fill_assumption='NEXT_OBSERVED_WS_BID_NO_QUEUE_OR_MARKET_IMPACT_MODEL')
                self.closed.append(row); rows.append(row)
                continue
            self.pending.remove(pending)
            if pending['role'] == 'CHAIN' and not self._source_valid(pending, now):
                rows.append(self._row('BUY_CANCELLED_SOURCE_CHANGED', now, role='CHAIN'))
                continue
            if any(p['symbol'] == symbol for p in self.positions):
                raise RuntimeError('Duplicate relay symbol')
            ratio = self.execution_config['RELAY_INITIAL_ALLOCATION' if pending['role'] == 'INITIAL' else 'RELAY_CHAIN_ALLOCATION']
            cap = min(self.budget*ratio, max(0., self.cash-self.budget*self.execution_config['RELAY_CASH_BUFFER']))
            qty = int(cap/(ask+CONFIG['QTY_CALC_BUFFER']))
            if qty <= 0:
                rows.append(self._row('BUY_CANCELLED_CASH', now, role=pending['role']))
                continue
            self.cash -= qty*ask
            p = dict(symbol=symbol, side=pending['side'], qty=qty, entry=ask, entry_time=now,
                     peak=ask, role=pending['role'], exit_pending=False,
                     leg_id=self.cycle_id+':'+str(len(self.closed)-self.cycle_closed_start+len(self.positions)+1),
                     decision_id=pending['decision_id'], model_version=pending['model_version'],
                     prediction=pending['prediction'])
            self.positions.append(p)
            if p['role'] == 'CHAIN':
                self.chain_filled = True
            rows.append(self._row('VIRTUAL_BUY', now, symbol=symbol, role=p['role'], leg_id=p['leg_id'],
                price=ask, quantity=qty, decision_id=p['decision_id'], model_version=p['model_version'],
                cycle_budget=self.budget, allocation=ratio, remaining_cash=self.cash,
                fill_assumption='NEXT_OBSERVED_WS_ASK_NO_QUEUE_OR_MARKET_IMPACT_MODEL'))
        return rows+self._finish(now)

    def _chain(self, now):
        if (len(self.positions) != 1 or self.pending or not self._session(now) or
                (self.chain_filled and not self.recipe['repeat_chain'])):
            return []
        p = self.positions[0]
        source = self._quote(p['symbol'], now)
        side = 1-p['side']
        target = CONFIG['BASE_ASSET_LONG'] if side == 1 else CONFIG['BASE_ASSET_SHORT']
        if (p['exit_pending'] or not source or not self._quote(target, now) or
                source['bid'] > p['entry']*(1-self.execution_config['RELAY_OPPOSITE_TRIGGER_PCT'])):
            return []
        self.chain_filled = True  # One reserved chain operation per cycle, as in live.
        self.pending.append(dict(action='BUY', role='CHAIN', time=now, side=side, source=p,
                                 decision_id=p['decision_id'], model_version=p['model_version'],
                                 prediction=p['prediction']))
        return [self._row('VIRTUAL_CHAIN_INTENT', now, source_symbol=p['symbol'],
                         symbol=target, source_entry=p['entry'], trigger_bid=source['bid'])]

    def market(self, event):
        now, kind = stamp(event['time']), event['kind']
        rows = self._cancel_buys(now)
        symbol = event.get('symbol')
        q = self._quote(symbol, now) if symbol else None
        bid = q['bid'] if q else None
        for p in self.positions:
            if p['exit_pending']:
                continue
            relevant = symbol == p['symbol']
            price = bid if relevant else None
            if relevant and kind in ('MARK', 'POLL') and price is not None:
                p['peak'] = max(p['peak'], price)
            reason = None
            if now.strftime('%H:%M') >= CONFIG['PHASE_EOD_CLEAR']:
                reason = 'EOD'
            elif price is not None and price <= p['entry']*(1-self.execution_config['SL_MIN_PCT']):
                reason = 'SL'
            elif price is not None and price >= p['entry']*(1+CONFIG['MAX_TP_PCT']):
                reason = 'TP'
            elif kind == 'POLL' and price is not None:
                trigger = CONFIG['TRAILING_TRIGGER_PCT'] if p['role'] == 'INITIAL' else self.execution_config['RELAY_CHAIN_TRAILING_TRIGGER_PCT']
                if p['peak'] >= p['entry']*(1+trigger) and price <= p['peak']*(1-CONFIG['TRAILING_DROP_PCT']):
                    reason = 'TRAIL'
            if reason is None and now-p['entry_time'] >= timedelta(minutes=CONFIG['TIME_STOP_MINUTES']):
                reason = 'TIME'
            if reason:
                p['exit_pending'] = True
                self.pending.append(dict(action='SELL', time=now, side=p['side'], position=p,
                                         reason=reason, trigger_price=price,
                                         price_basis='WS_BID' if price is not None else 'CLOCK'))
                rows.append(self._row('VIRTUAL_EXIT_INTENT', now, symbol=p['symbol'], role=p['role'],
                                      leg_id=p['leg_id'], reason=reason, trigger_price=price))
        return rows+self._chain(now)+self._finish(now)


def build_direction_shadows():
    recipe = settings()
    return {name: DirectionRelay(name) for name in recipe['strategies']} if recipe['enabled'] else {}
