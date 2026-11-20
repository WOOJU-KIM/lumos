"""Recorded-event virtual portfolios. No broker, network, live module or scheduler imports."""
from common import HOME, CONFIG, POLICY, literal_setting, load_execution_config
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import hashlib
import json
import math

NY = ZoneInfo('America/New_York')
BUY_WAIT = 2*literal_setting('BUY_TIMEOUT_SEC')
SCHEMA = json.loads((HOME/'inputs/schema.json').read_text())


def stamp(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError('Timezone-aware event time required')
    return result.astimezone(NY)


def validate_packet(packet):
    now = stamp(packet['time'])
    scheduled = stamp(packet['decision_time'])
    if packet.get('kind')!='DECISION' or packet.get('schema_version')!=1:
        raise ValueError('Wrong packet schema')
    if packet['net_feature_names']!=SCHEMA['net_features'] or set(packet['full_features'])!=set(SCHEMA['full_features']):
        raise ValueError('Unexpected feature keys/order; no credentials or extra fields accepted')
    if 'relay_features' in packet:
        recipe = json.loads((HOME/'relay_directions.json').read_text(encoding='utf-8'))
        extra_names = set(recipe.get('relay_model', {}).get('feature_names', [])) - set(SCHEMA['full_features'])
        if set(packet['relay_features']) != extra_names:
            raise ValueError('Unexpected relay feature keys')
    if len(packet['sequence'])!=32 or any(len(row)!=72 for row in packet['sequence']):
        raise ValueError('Expected32x72 sequence')
    history = [stamp(t) for t in packet['feature_times']]
    if len(history)!=32 or any(a>=b for a,b in zip(history,history[1:])) or history[-1]>scheduled:
        raise ValueError('Future/unordered feature history')
    if scheduled.minute%15 or scheduled.second or scheduled.microsecond:
        raise ValueError('Decision must use a15minute boundary')
    if not scheduled<=now<scheduled+timedelta(minutes=15):
        raise ValueError('Decision completion time outside its cycle')
    if history[-1] != stamp(packet['expected_last_closed_available_at']):
        raise ValueError('Stale input: latest expected completed candle missing')
    if not packet.get('required_assets_complete',False):
        raise ValueError('Cross-asset snapshot incomplete')
    if any(stamp(t)>scheduled for t in packet['cross_asset_available_at'].values()):
        raise ValueError('Future cross-asset data')
    # No inference about an overnight gap is made here; the producer must supply calendar-derived expectation.
    for symbol in [CONFIG['BASE_ASSET_SHORT'],CONFIG['BASE_ASSET_LONG']]:
        atr = float(packet['atr5m_pct'][symbol])
        if not math.isfinite(atr) or atr<=0:
            raise ValueError('Missing/invalid completed5m ATR')
    values = [v for row in packet['sequence'] for v in row]+list(packet['full_features'].values())+list(packet.get('relay_features', {}).values())
    if not all(isinstance(v,(float,int)) and math.isfinite(v) for v in values):
        raise ValueError('Missing/non-finite features; no silent zero fill')
    payload = {k:packet[k] for k in ['sequence','net_feature_names','full_features','feature_times','cross_asset_available_at']}
    if 'relay_features' in packet:
        payload['relay_features'] = packet['relay_features']
    return now, hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()


class Ledger:
    def __init__(self, model_id, cohort='primary', evaluation_window=None):
        self.model_id, self.cohort = model_id, cohort
        self.execution_config = load_execution_config()
        # Offline replay retains its frozen evaluation week. Only the live
        # shadow consumer injects the separately approved execution window.
        self._evaluation_window = evaluation_window or (
            lambda day: POLICY['evaluation_week_ny'][0] <= str(day) <= POLICY['evaluation_week_ny'][1])
        self.cash = 10000.
        self.positions = []
        self.pending = []
        self.closed = []

    def intend(self, time, side, atr_pct, decision_id, model_version, prediction):
        if side<0 or (self.cohort=='primary' and (self.positions or self.pending)):
            return 'POSITION_OPEN_OR_PENDING' if side>=0 else 'MODEL_NONE'
        if not '09:30'<=time.strftime('%H:%M')<CONFIG['PHASE_MAIN_END'] or time.weekday()>4 or not self._evaluation_window(time.date()):
            return 'ENTRY_SESSION_BLOCK'
        self.pending.append(dict(action='BUY',time=time,side=side,atr_pct=atr_pct,
                                 decision_id=decision_id,model_version=model_version,prediction=prediction))
        return 'PENDING_NEXT_QUOTE'

    def on_quote(self, event):
        now = stamp(event['time']); symbol=event['symbol']
        bid, ask = float(event['bid']),float(event['ask'])
        if not (0<bid<=ask) or not math.isfinite(ask):
            raise ValueError('Invalid bid/ask')
        if not 0 <= (now-stamp(event['quote_time'])).total_seconds() <= CONFIG['MAIN_LOOP_TICK_OPEN']:
            return []
        events=[]
        for pending in list(self.pending):
            s=CONFIG['BASE_ASSET_LONG'] if pending['side']==1 else CONFIG['BASE_ASSET_SHORT']
            if s!=symbol or now<=pending['time']:
                continue
            if pending['action']=='BUY':
                if (now-pending['time']).total_seconds()>BUY_WAIT:
                    self.pending.remove(pending)
                    events.append(dict(event='BUY_CANCELLED_STALE_INTENT',time=str(now),
                                       model_id=self.model_id,cohort=self.cohort,
                                       decision_id=pending['decision_id'],model_version=pending['model_version']))
                    continue
                if not '09:30'<=now.strftime('%H:%M')<CONFIG['PHASE_MAIN_END']:
                    self.pending.remove(pending)
                    events.append(dict(event='BUY_CANCELLED_SESSION',time=str(now),
                                       model_id=self.model_id,cohort=self.cohort,
                                       decision_id=pending['decision_id'],model_version=pending['model_version']))
                    continue
                # Cohort positions are independent unit-notional observations, not a combined equity account.
                qty=1 if self.cohort=='forced' else int(self.cash*CONFIG['MAX_ALLOCATION_RATIO']/(ask+CONFIG['QTY_CALC_BUFFER']))
                self.pending.remove(pending)
                if qty<=0:
                    continue
                stop_pct=self.execution_config['SL_MIN_PCT']
                position=dict(symbol=s,side=pending['side'],qty=qty,entry=ask,entry_time=now,peak=ask,
                              stop_pct=stop_pct,decision_id=pending['decision_id'],model_version=pending['model_version'],
                              prediction=pending['prediction'],exit_pending=False)
                self.positions.append(position)
                events.append(dict(event='VIRTUAL_BUY',time=str(now),symbol=s,price=ask,quantity=qty,
                                   model_id=self.model_id,cohort=self.cohort,
                                   decision_id=pending['decision_id'],model_version=pending['model_version'],
                                   fill_assumption='NEXT_AVAILABLE_ASK_NO_QUEUE_OR_MARKET_IMPACT_MODEL',
                                   price_basis='WS_ASK' if event.get('observed_at') else 'RECORDED_ASK'))
            else:
                p=pending['position'];self.pending.remove(pending)
                if p not in self.positions:
                    continue
                gross=bid/p['entry']-1;reserve=max(.002,CONFIG['BACKTEST_FEE_SLIPPAGE']);net=gross-reserve
                row=dict(model_id=self.model_id,cohort=self.cohort,decision_id=p['decision_id'],model_version=p['model_version'],
                         symbol=s,entry_time=str(p['entry_time']),exit_time=str(now),entry_price=p['entry'],exit_price=bid,
                         quantity=p['qty'],gross_return=gross,cost_reserve=reserve,net_return=net,reason=pending['reason'],
                         trigger_price_basis=pending.get('price_basis'),
                         trigger_price=pending.get('trigger_price'),
                         fill_assumption='NEXT_AVAILABLE_BID_NO_QUEUE_OR_MARKET_IMPACT_MODEL')
                self.closed.append(row);self.positions.remove(p)
                if self.cohort=='primary':
                    self.cash+=p['qty']*p['entry']*net
                events.append(dict(event='VIRTUAL_SELL',**row))
        return events

    def market(self, event):
        now=stamp(event['time']);kind=event['kind'];price=float(event.get('price',0));symbol=event.get('symbol')
        if kind=='MARK' or (kind=='POLL' and event.get('price_basis')):
            if event.get('price_basis') not in ('WS_BID','WS_EXPLICIT_LAST'):
                raise ValueError('Unverified risk price basis')
            if not 0 <= (now-stamp(event['observed_at'])).total_seconds() <= CONFIG['MAIN_LOOP_TICK_OPEN']:
                raise ValueError('Stale risk mark')
        records=[dict(event='BUY_CANCELLED_STALE_INTENT',model_id=self.model_id,cohort=self.cohort,
                      time=str(now),decision_id=p['decision_id'],model_version=p['model_version'])
                 for p in self.pending if p['action']=='BUY' and (now-p['time']).total_seconds()>BUY_WAIT]
        self.pending[:]=[p for p in self.pending if p['action']!='BUY' or (now-p['time']).total_seconds()<=BUY_WAIT]
        for p in self.positions:
            if p['exit_pending'] or (symbol is not None and symbol!=p['symbol']):
                continue
            if kind in ['TICK','MARK','POLL']:
                if not math.isfinite(price) or price<=0:
                    raise ValueError('Invalid tick price')
                p['peak']=max(p['peak'],price)
            pnl=price/p['entry']-1 if price>0 else None
            trailing=p['peak']>=p['entry']*(1+CONFIG['TRAILING_TRIGGER_PCT'])
            stop=max(p['entry']*1.002,p['peak']*(1-CONFIG['TRAILING_DROP_PCT'])) if trailing else p['entry']*(1-p['stop_pct'])
            reason=None
            # WebSocket bid/explicit-last marks check initial TP/SL and update
            # the peak. Trailing remains on the configured polling cadence.
            if now.strftime('%H:%M')>=CONFIG['PHASE_EOD_CLEAR']:
                reason='EOD'
            elif pnl is not None and pnl>=CONFIG['MAX_TP_PCT']:
                reason='TP'
            elif kind in ('TICK','MARK') and pnl is not None and pnl<=-p['stop_pct']:
                reason='SL'
            elif kind=='POLL' and price>0 and price<=stop:
                reason='TRAILING' if trailing else 'SL'
            elif (now-p['entry_time']).total_seconds()>=CONFIG['TIME_STOP_MINUTES']*60:
                reason='TIME'
            if reason:
                p['exit_pending']=True
                self.pending.append(dict(action='SELL',time=now,side=p['side'],position=p,reason=reason,
                                         price_basis=event.get('price_basis'),trigger_price=price if price>0 else None))
                records.append(dict(event='VIRTUAL_EXIT_INTENT',model_id=self.model_id,cohort=self.cohort,
                                    time=str(now),decision_id=p['decision_id'],model_version=p['model_version'],reason=reason,
                                    price_basis=event.get('price_basis'),trigger_price=price if price>0 else None))
        return records


class Engine:
    def __init__(self,candidates,evaluation_window=None,require_price_monitoring=False):
        self.candidates=candidates
        self.ledgers={(name,cohort):getattr(candidate,'ledger_factory',Ledger)(name,cohort,evaluation_window) for name,candidate in candidates.items()
                      for cohort in getattr(candidate,'cohorts',('primary','forced'))}
        self.seen=set();self.last_event=None;self.errors=[];self.decisions=[]
        self.require_price_monitoring=require_price_monitoring
        self.price_observed={}
        self.feed_gap_count=0
        self.entry_resume_after=None

    def feed_gap(self, now):
        """Keep holdings/SELL intents; discard unfilled BUY intents and old quotes."""
        self.feed_gap_count+=1
        self.entry_resume_after=now.replace(minute=now.minute//15*15,second=0,microsecond=0)
        self.price_observed.clear()
        output=[]
        for (name,cohort),ledger in self.ledgers.items():
            for pending in list(ledger.pending):
                if pending['action']=='BUY':
                    ledger.pending.remove(pending)
                    output.append(dict(event='BUY_CANCELLED_FEED_GAP',time=now.isoformat(),
                        model_id=name,cohort=cohort,decision_id=pending.get('decision_id')))
            if hasattr(ledger,'quotes'):ledger.quotes.clear()
            if hasattr(ledger,'_reverse_wait'):ledger._reverse_wait=None
            if hasattr(ledger,'_finish'):output.extend(ledger._finish(now))
        return output

    def price_monitoring(self,now):
        return {symbol: symbol in self.price_observed and
                0 <= (now-self.price_observed[symbol]).total_seconds() <= CONFIG['MAIN_LOOP_TICK_OPEN']
                for symbol in (CONFIG['BASE_ASSET_LONG'],CONFIG['BASE_ASSET_SHORT'])}

    def handle(self,event):
        now=stamp(event['time'])
        if self.last_event is not None and now<self.last_event:
            raise ValueError('Out-of-order event; do not sort or silently reinterpret ticks')
        self.last_event=now
        output=[]
        if event['kind']=='DECISION':
            now,digest=validate_packet(event)
            if self.entry_resume_after is not None and stamp(event['decision_time'])<=self.entry_resume_after:
                return [dict(event='DECISION_REJECTED_FEED_GAP',time=now.isoformat(),
                             decision_id=event['decision_id'],reason='WAIT_NEXT_COMPLETED_15M')]
            decision_id=event['decision_id']
            cycle=stamp(event['decision_time']).isoformat()
            if decision_id in self.seen or cycle in self.seen:
                raise ValueError('Duplicate decision ID or15minute cycle')
            self.seen.update([decision_id,cycle])
            for name,candidate in self.candidates.items():
                try:
                    prediction=candidate.predict(event)
                    row=dict(event='DECISION',time=str(now),decision_id=decision_id,input_sha256=digest,**prediction)
                    for cohort,side_key in [('primary','side'),('forced','forced_side')]:
                        if (name,cohort) not in self.ledgers:
                            continue
                        side=prediction[side_key]
                        symbol=CONFIG['BASE_ASSET_LONG'] if side==1 else CONFIG['BASE_ASSET_SHORT']
                        # Shared current configuration filters; callers must provide observed directional flags when enabled.
                        blocked=(CONFIG['USE_CROSS_ASSET_VETO'] and event.get('cross_veto',{}).get(str(side),True)) or (
                            CONFIG['USE_60M_TREND_FILTER'] and not event.get('trend_ok',{}).get(str(side),False))
                        rsi=float(event['full_features']['RSI_14'])
                        blocked=blocked or (side==1 and rsi>CONFIG['RSI_OVERBOUGHT_THRESHOLD']) or (side==0 and rsi<CONFIG['RSI_OVERSOLD_THRESHOLD'])
                        if side>=0 and self.require_price_monitoring and not self.price_monitoring(now)[symbol]:
                            action='PRICE_MONITORING_UNAVAILABLE'
                        elif cohort=='primary' and blocked:
                            action='CONFIG_FILTER_BLOCK'
                        else:
                            action=self.ledgers[(name,cohort)].intend(now,side,float(event['atr5m_pct'][symbol]),decision_id,candidate.version,prediction)
                        row[cohort+'_action']=action
                    self.decisions.append(row);output.append(row)
                except Exception as error:
                    # Candidate failure is explicit and never prevents remaining candidates/exit processing.
                    row=dict(event='CANDIDATE_ERROR',time=str(now),model_id=name,error_type=type(error).__name__)
                    self.errors.append(row);output.append(row)
        elif event['kind'] in ['TICK','MARK','POLL','CLOCK']:
            for ledger in self.ledgers.values():
                output.extend(ledger.market(event))
            if event['kind']=='MARK':
                self.price_observed[event['symbol']]=stamp(event['observed_at'])
        elif event['kind']=='QUOTE':
            for ledger in self.ledgers.values():
                output.extend(ledger.on_quote(event))
        else:
            raise ValueError('Unknown event kind')
        for row in output:row['feed_gap_count']=self.feed_gap_count
        return output
