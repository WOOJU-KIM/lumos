"""Read-only shadow consumer. Explicit startup; never launches/restarts the live runner.
No broker imports/API/network. Named shared memory is the only live input.
"""
from common import *
guard()
verify_source_policy()
import argparse
import queue
import sqlite3
import struct
import threading
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from feed_bridge import Reader, TICK, FEATURE_NAMES
from runtime import Engine, validate_packet
from storage import Audit
from runtime_control import load_control, date_allowed, should_stop
from market_events import MarketEvents, decode_frame, MARKET_V2
from lifecycle import validate_owner, ProcessWatch, owner_stop_reason
from feed_health import FeedHealth

NY=ZoneInfo('America/New_York')
SCHEMA=json.loads((HOME/'inputs/schema.json').read_text())
MAX_EVENT_AGE=2.0
MAX_QUEUE=8192
HEALTH_PERIOD_SECONDS=5


def iso(epoch):
    return datetime.fromtimestamp(epoch,NY).isoformat()


def decode_market(payload):
    symbol,price,bid,ask,from_ws=TICK.unpack(payload)
    if not from_ws:
        # A running producer may still use the older publisher until its next
        # normal restart. Exclude REST without stopping WebSocket observation.
        return None
    return symbol.rstrip(b'\0').decode('ascii'),price,bid,ask


class PacketBuilder:
    """Use captured exact live features. Only ATR/session metadata is read separately.
    Queries are SQLite URI mode=ro, query_only, busy_timeout=0; never live APIs.
    Cross-asset provenance is checked against DB as-of; not claimed atomic provenance.
    """
    def __init__(self):
        self.db=sqlite3.connect((PROJECT/'data/market_data.db').as_uri()+'?mode=ro',uri=True,timeout=0)
        self.db.execute('pragma query_only=ON')
        self.calendar=mcal.get_calendar('NYSE')
        self.seen=set()

    def bars(self,symbol,tf,t,n):
        minutes={'5m':5,'15m':15,'60m':60}[tf]
        cutoff=(t.replace(tzinfo=None)-timedelta(minutes=minutes)).strftime('%Y-%m-%d %H:%M:%S')
        return self.db.execute('select datetime,high,low,close from market_candles where symbol=? and timeframe=? and datetime<=? order by datetime desc limit ?', (symbol,tf,cutoff,n)).fetchall()[::-1]

    def build(self,stamp,payload):
        now=datetime.fromtimestamp(stamp,NY)
        scheduled=now.replace(minute=now.minute//15*15,second=0,microsecond=0)
        if scheduled.isoformat() in self.seen:return None
        schedule=self.calendar.schedule(start_date=str(scheduled.date()),end_date=str(scheduled.date()))
        if schedule.empty:raise ValueError('NON_TRADING_DAY')
        op=schedule.iloc[0].market_open.tz_convert(NY).to_pydatetime()
        close=schedule.iloc[0].market_close.tz_convert(NY).to_pydatetime()
        if not op<=scheduled<min(close-timedelta(minutes=30),scheduled.replace(hour=15,minute=30)):
            return None
        rows,cols,names_size=struct.unpack_from('<HHH',payload)
        names=payload[6:6+names_size].decode('ascii').split('\0')
        if rows!=32 or names!=FEATURE_NAMES or cols!=len(names):raise ValueError('FEATURE_CAPTURE_SCHEMA')
        offset=6+names_size
        times=np.frombuffer(payload,dtype='<i8',count=rows,offset=offset).copy()
        values=np.frombuffer(payload,dtype='<f8',count=rows*cols,offset=offset+rows*8).reshape(rows,cols)
        # Production datetime_dt is naive New York candle availability, not UTC.
        available=[pd.Timestamp(int(t)).tz_localize(NY).to_pydatetime() for t in times]
        expected=scheduled
        if scheduled==op:
            previous=self.bars(CONFIG['BASE_ASSET_LONG'],'15m',scheduled,1)
            if not previous:raise ValueError('MISSING_OPEN_CONTEXT')
            expected=pd.Timestamp(previous[-1][0]).tz_localize(NY).to_pydatetime()+timedelta(minutes=15)
        cross={}
        for symbol in ['SOXS','NVDA','QQQ','VIXY','IEF']:
            b=self.bars(symbol,'15m',scheduled,1)
            if not b:raise ValueError('MISSING_CROSS_CONTEXT')
            at=pd.Timestamp(b[-1][0]).tz_localize(NY).to_pydatetime()+timedelta(minutes=15)
            # At open accept prior completed context; otherwise demand current bar.
            if scheduled>op and at!=scheduled:raise ValueError('STALE_CROSS_CONTEXT')
            cross[symbol]=at.isoformat()
        # SOXX 60m participates in the production feature engine.
        b=self.bars('SOXX','60m',scheduled,1)
        if not b:raise ValueError('MISSING_SOXX60_CONTEXT')
        cross['SOXX_60m']=(pd.Timestamp(b[-1][0]).tz_localize(NY).to_pydatetime()+timedelta(minutes=60)).isoformat()
        atr={}
        for symbol in [CONFIG['BASE_ASSET_SHORT'],CONFIG['BASE_ASSET_LONG']]:
            b=self.bars(symbol,'5m',scheduled,15)
            if len(b)!=15:raise ValueError('MISSING_ATR_CONTEXT')
            last=pd.Timestamp(b[-1][0]).tz_localize(NY).to_pydatetime()+timedelta(minutes=5)
            if scheduled>op and last!=scheduled:raise ValueError('STALE_ATR_CONTEXT')
            a=np.array([r[1:] for r in b],dtype=float)
            tr=np.maximum(a[1:,0]-a[1:,1],np.maximum(abs(a[1:,0]-a[:-1,2]),abs(a[1:,1]-a[:-1,2])))
            atr[symbol]=float(tr.mean()/a[-1,2]*100)
        full={k:float(values[-1,j]) for j,k in enumerate(names) if k in SCHEMA['full_features']}
        relay_features={k:float(values[-1,j]) for j,k in enumerate(names) if k not in SCHEMA['full_features']}
        full['minutes_to_eod']=((close-timedelta(minutes=5))-scheduled).total_seconds()/60
        packet=dict(kind='DECISION',schema_version=1,time=now.isoformat(),decision_time=scheduled.isoformat(),
            decision_id='live_'+scheduled.isoformat(),net_feature_names=SCHEMA['net_features'],
            sequence=values[:,[names.index(k) for k in SCHEMA['net_features']]].tolist(),full_features=full,
            relay_features=relay_features,
            feature_times=[t.isoformat() for t in available],cross_asset_available_at=cross,
            expected_last_closed_available_at=expected.isoformat(),required_assets_complete=True,atr5m_pct=atr)
        # Enabled shields require observed live flags, which this version does not export.
        if CONFIG['USE_CROSS_ASSET_VETO'] or CONFIG['USE_60M_TREND_FILTER']:
            raise ValueError('ENABLED_LIVE_FILTER_FLAGS_NOT_EXPORTED')
        validate_packet(packet);self.seen.add(scheduled.isoformat())
        return packet


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seconds',type=float,default=0)
    parser.add_argument('--feed-tag',default=None)
    parser.add_argument('--owner-pid',type=int,default=None)
    parser.add_argument('--owner-created',type=int,default=None)
    parser.add_argument('--owner-token',default=None)
    args=parser.parse_args()
    if args.feed_tag is not None and os.environ.get('LUMOS_SHADOW_FEED_ISOLATED')!='1':
        raise ValueError('Custom feed tags require isolated QA mode')
    owner=None;owner_watch=None
    if any(v is not None for v in (args.owner_pid,args.owner_created,args.owner_token)):
        owner=validate_owner(dict(pid=args.owner_pid,created_filetime=args.owner_created,token=args.owner_token))
    control=load_control()
    if should_stop(datetime.now(NY).date(),control):
        save('qa/feed_listener.json',dict(status='DISABLED_OR_EXPIRED',feed_received=False,
             real_orders_possible=False,live_runner_started=False,runtime_control=control))
        return 0
    if control.get('operation')=='FOLLOW_LIVE_RUNNER' and owner is None and os.environ.get('LUMOS_SHADOW_FEED_ISOLATED')!='1':
        save('qa/feed_listener.json',dict(status='WAITING_FOR_LIVE_RUNNER',feed_received=False,real_orders_possible=False))
        return 0
    if owner is not None:
        owner_watch=ProcessWatch(owner['pid'],owner['created_filetime'])
        reason=owner_stop_reason(HOME,owner,owner_watch)
        if reason:
            owner_watch.close()
            save('qa/feed_listener.json',dict(status='STOPPED',owner=owner,stop_reason=reason,feed_received=False,real_orders_possible=False))
            return 0
    save('qa/feed_listener.json',dict(pid=os.getpid(),status='LOADING_MODELS',feed_received=False,
         real_orders_possible=False,live_runner_started=False))
    # Preload before background I/O mode; that mode can heavily throttle DLL startup.
    # This separate process already starts with below-normal CPU priority.
    from comparison_shadows import build_comparison_shadows
    comp_recipe=json.loads((HOME/'comparison_shadows.json').read_text(encoding='utf-8'))
    candidates=build_comparison_shadows() if comp_recipe.get('runtime_enabled',True) else {}
    engine=Engine(candidates,
                  evaluation_window=lambda day: date_allowed(day,control),require_price_monitoring=True)
    if owner_watch is not None:
        reason=owner_stop_reason(HOME,owner,owner_watch)
        if reason:
            owner_watch.close()
            save('qa/feed_listener.json',dict(status='STOPPED',owner=owner,stop_reason=reason,feed_received=False,real_orders_possible=False))
            return 0
    background_io=False
    # Keep the launcher's below-normal CPU priority, but not Windows background
    # I/O/memory throttling: the audit writer must keep up with the WS stream.
    # Two-second admission remains strict; late data marks a gap, not a shutdown.
    # Fully preload models before advertising a consumer heartbeat.
    audit=Audit('live_shadow_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    builder=PacketBuilder();reader=Reader(args.feed_tag) if args.feed_tag else Reader()
    if hasattr(builder.db,'execute'):
        from soxx_orb_shadow import attach_soxx_orb_shadow
        attach_soxx_orb_shadow(engine,db=builder.db,evaluation_window=lambda day: date_allowed(day,control))
    events=queue.Queue(MAX_QUEUE);stop=threading.Event()
    failures=[];stats=dict(received=0,decisions=0,quotes=0,ticks=0,marks=0,polls=0,
                          ignored_rest=0,rejected_market=0,max_arrival_age_ms=0.,api_calls=0,
                          receiver_losses=0,ring_recoveries=0,max_queue_depth=0,max_build_ms=0.,
                          max_engine_ms=0.,max_audit_write_ms=0.)
    health=FeedHealth(MAX_EVENT_AGE)
    market_flow=MarketEvents([CONFIG['BASE_ASSET_LONG'],CONFIG['BASE_ASSET_SHORT'],'SOXX'],CONFIG['MAIN_LOOP_TICK_OPEN'])
    def receive():
        try:
            while not stop.is_set():
                try:item=reader.read()
                except RuntimeError as error:
                    if str(error) not in ('FEED_GAP_OR_PRODUCER_DROP','FEED_OVERWRITE'):raise
                    stats['receiver_losses']+=max(1,reader.resync())
                    stats['ring_recoveries']+=1
                    continue
                if item is None:stop.wait(.001);continue
                envelope=(*item,time.time())
                try:events.put_nowait(envelope)
                except queue.Full:
                    try:events.get_nowait()
                    except queue.Empty:pass
                    stats['receiver_losses']+=1
                    events.put_nowait(envelope)
                stats['received']+=1
                stats['max_queue_depth']=max(stats['max_queue_depth'],events.qsize())
        except Exception as e:
            failures.append(type(e).__name__+':'+str(e));stop.set()
    thread=threading.Thread(target=receive,daemon=True);thread.start()
    save('qa/feed_listener.json',dict(pid=os.getpid(),status='LISTENING',started_at=datetime.now(NY).isoformat(),
        feed_received=False,real_orders_possible=False,live_runner_started=False))
    start=time.monotonic();valid=True;last_health=0.;monitor_state=None;stop_reason='LOOP_ENDED'
    audit.append(dict(event='RUN_START',feed='NAMED_PAGEFILE_RING',real_orders=False,
        cross_provenance='DB_ASOF_CHECK_NOT_ATOMIC_SOURCE_SNAPSHOT',fill='OBSERVED_NEXT_BID_ASK_ONLY',api_calls=0,
        background_io_mode=background_io,runtime_control=dict(control),
        risk_price_policy='OBSERVED_WS_BID_OR_EXPLICIT_LAST_NOT_SYNTHETIC_TICKS',
        feed_recovery_policy='SKIP_STALE_RESYNC_RING_PRESERVE_HOLDINGS_WAIT_NEXT_DECISION',
        source_hashes={name:sha(HOME/name) for name in
                       ('live_feed.py','feed_health.py','runtime.py','storage.py','feed_bridge.py')},
        price_monitoring_required_for_entry=True,owner=owner))
    audit.append(dict(event='EXECUTION_EXIT_POLICY',initial_stop_pct=load_execution_config()['SL_MIN_PCT'],
        stop_mode='FIXED',execution_policy_sha256=sha(HOME/'execution_policy.json')))
    audit.append(dict(event='THREE_COMPARISON_SHADOW_POLICY',
        recipe=json.loads((HOME/'comparison_shadows.json').read_text(encoding='utf-8')),
        recipe_sha256=sha(HOME/'comparison_shadows.json'),
        source_sha256=sha(HOME/'comparison_shadows.py'),
        model_ids=list(candidates),real_orders=False))
    def dispatch_market(event):
        if event['kind']=='MARKET_REJECTED':
            stats['rejected_market']+=1
            audit.append(dict(event='MARKET_REJECTED',**{k:v for k,v in event.items() if k!='kind'}))
            return
        engine_start=time.monotonic()
        output=engine.handle(event)
        stats['max_engine_ms']=max(stats['max_engine_ms'],(time.monotonic()-engine_start)*1000)
        audit.market(event)
        if event['kind']=='QUOTE':stats['quotes']+=1
        elif event['kind']=='MARK':stats['marks']+=1
        elif event['kind']=='POLL':stats['polls']+=1
        for row in output:audit.append(row)
    def begin_gap(reason, now):
        if health.gap():
            market_flow.prices.clear()
            for row in engine.feed_gap(datetime.fromtimestamp(now,NY)):audit.append(row)
            audit.append(dict(event='FEED_GAP',time=iso(now),reason=reason,
                positions_preserved=True,new_entries='WAIT_NEXT_COMPLETED_15M',**health.snapshot()))
    observed_losses=0
    try:
        while not stop.is_set() and (not args.seconds or time.monotonic()-start<args.seconds):
            if stats['receiver_losses']!=observed_losses:
                observed_losses=stats['receiver_losses']
                begin_gap('RING_OR_CONSUMER_QUEUE_LOSS',time.time())
            # Run even if all incoming frames are REST/rejected/duplicate
            # features. No feed category can starve time-based liquidation.
            for event in market_flow.timer(time.time()):dispatch_market(event)
            if time.monotonic()-last_health>=HEALTH_PERIOD_SECONDS:
                if owner_watch is not None:
                    reason=owner_stop_reason(HOME,owner,owner_watch)
                    if reason:
                        stop_reason=reason
                        break
                updated_control=load_control()
                if updated_control!=control:
                    control.clear();control.update(updated_control)
                    audit.append(dict(event='RUNTIME_CONTROL_CHANGED',time=datetime.now(NY).isoformat(),
                                      runtime_control=dict(control)))
                if should_stop(datetime.now(NY).date(),control):
                    stop_reason='RUNTIME_DISABLED_OR_EXPIRED'
                    break
                observed=engine.price_monitoring(datetime.now(NY))
                if observed!=monitor_state:
                    audit.append(dict(event='PRICE_MONITORING_STATE',time=datetime.now(NY).isoformat(),
                                      symbols=observed,new_entries_require_fresh_price=True))
                    monitor_state=observed
                audit.flush()
                stats['max_audit_write_ms']=audit.max_write_ms
                save('qa/feed_listener.json',dict(pid=os.getpid(),status='LISTENING',stats=stats,
                    feed_health=dict(health.snapshot(),audit_max_write_ms=audit.max_write_ms),
                    updated_at=datetime.now(NY).isoformat(),feed_received=bool(stats['received']),
                    real_orders_possible=False,live_runner_started=False,background_io_mode=background_io,
                    runtime_control=dict(control),price_monitoring=observed,
                    open_positions={n:len(engine.ledgers[(n,'primary')].positions) for n in engine.candidates},
                    pending_exits=sum(p['exit_pending'] for ledger in engine.ledgers.values() for p in ledger.positions),owner=owner))
                request=HOME/'qa/feed_stop_request.json'
                if request.exists() and json.loads(request.read_text()).get('pid')==os.getpid():
                    stop_reason='WORKER_STOP_REQUESTED'
                    break
                last_health=time.monotonic()
            try:item=events.get(timeout=.1)
            except queue.Empty:
                continue
            seq,stamp,kind,payload,received=item
            if stats['receiver_losses']!=observed_losses:
                observed_losses=stats['receiver_losses']
                begin_gap('RING_OR_CONSUMER_QUEUE_LOSS',time.time())
            now=time.time()
            age=now-stamp;stats['max_arrival_age_ms']=max(stats['max_arrival_age_ms'],age*1000)
            if not health.admit(stamp,received,now,kind):
                begin_gap('LATE_OR_FUTURE_FRAME_SKIPPED',now)
                continue
            if kind==3:
                begin_gap('LIVE_FEATURE_CAPTURE_FAILED',now)
                audit.append(dict(event='DECISION_REJECTED',time=iso(now),reason='LIVE_FEATURE_CAPTURE_FAILED'))
                continue
            if kind==2:
                build_start=time.monotonic()
                try:p=builder.build(stamp,payload)
                except (ValueError,sqlite3.Error) as e:
                    audit.append(dict(event='DECISION_REJECTED',time=iso(stamp),reason=str(e)));continue
                finally:stats['max_build_ms']=max(stats['max_build_ms'],(time.monotonic()-build_start)*1000)
                if p is None:continue
                if not health.admit(stamp,received,time.time(),kind):
                    begin_gap('FEATURE_BUILD_EXCEEDED_FRESHNESS',time.time())
                    continue
                p['time']=iso(time.time())
                audit.packet(p)
                if not health.admit(stamp,received,time.time(),kind):
                    begin_gap('AUDIT_WRITE_EXCEEDED_FRESHNESS',time.time())
                    continue
                if stats['receiver_losses']!=observed_losses:
                    observed_losses=stats['receiver_losses']
                    begin_gap('RING_OR_CONSUMER_QUEUE_LOSS',time.time())
                engine_start=time.monotonic()
                output=engine.handle(p);stats['decisions']+=1
                stats['max_engine_ms']=max(stats['max_engine_ms'],(time.monotonic()-engine_start)*1000)
                for row in output:audit.append(row)
            elif kind in (1,MARKET_V2):
                try:market=decode_frame(kind,payload)
                except ValueError as error:
                    if str(error) not in ('CROSSED_QUOTE','INVALID_MARKET_NUMBER'):raise
                    stats['rejected_market']+=1
                    begin_gap(str(error),time.time())
                    audit.append(dict(event='MARKET_REJECTED',time=iso(time.time()),reason=str(error)))
                    continue
                if market is None:
                    stats['ignored_rest']+=1
                    continue
                try:market_events=market_flow.frame(market,stamp,time.time())
                except ValueError as error:
                    if str(error)!='STALE_OR_FUTURE_MARKET_FRAME':raise
                    begin_gap(str(error),time.time())
                    continue
                for event in market_events:dispatch_market(event)
            else:raise RuntimeError('UNKNOWN_FEED_KIND')
            if health.recovered():
                audit.append(dict(event='FEED_RECOVERED',time=iso(time.time()),**health.snapshot()))
        if failures:raise RuntimeError(failures[0])
    except KeyboardInterrupt:
        stop_reason='KEYBOARD_INTERRUPT'
    except Exception as e:
        valid=False;stop_reason='QUARANTINED';audit.append(dict(event='RUN_QUARANTINED',reason=str(e)))
    finally:
        stop.set();thread.join(timeout=1);reader.close();builder.db.close()
        stats['max_audit_write_ms']=audit.max_write_ms
        if owner_watch is not None:owner_watch.close()
        audit.append(dict(event='RUN_STOP',valid=valid,stats=stats,feed_health=health.snapshot(),owner=owner,stop_reason=stop_reason,unresolved_positions={
            n:len(engine.ledgers[(n,'primary')].positions) for n in engine.candidates},
            unresolved_all_cohorts={n:{cohort:len(engine.ledgers[(n,cohort)].positions)
                for cohort in ('primary','forced') if (n,cohort) in engine.ledgers} for n in engine.candidates}));audit.close()
        save('qa/feed_latest.json',dict(valid=valid,stats=stats,feed_health=health.snapshot(),live_ready=False,orders_possible=False))
        save('qa/feed_listener.json',dict(pid=os.getpid(),status='STOPPED',feed_received=bool(stats['received']),
            stats=stats,valid=valid,feed_health=health.snapshot(),real_orders_possible=False,owner=owner,stop_reason=stop_reason))
    return 0 if valid else 2


if __name__=='__main__':raise SystemExit(main())
