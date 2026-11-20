"""Transport failure/latency and real historical feature capture QA. No live runner."""
import json
import struct
import time
import uuid
import statistics
from pathlib import Path
import feed_bridge as bridge

HOME=Path(__file__).resolve().parent
checks={}
def check(name,value):
    checks[name]=bool(value)
    assert value,name

reader=bridge.Reader('Local\\LumosShadowQA_'+uuid.uuid4().hex)
original=bridge._memory;bridge._memory=reader.mem
bridge._qa_allow=True

# Stall consumer completely. Publishing cannot wait for it, even after overwrite.
samples=[]
for i in range(10000):
    start=time.perf_counter_ns();bridge.market('SOXL',100.,{'values':{'41':'100.01','51':'99.99','secret':'MUST_NOT_LEAVE'}},True)
    samples.append((time.perf_counter_ns()-start)/1000)
check('stalled_consumer_publish_completes',len(samples)==10000)
try:reader.read()
except RuntimeError:check('overflow_quarantined',True)
else:check('overflow_quarantined',False)
reader.cursor=struct.unpack_from('<Q',reader.mem,0)[0]
reader.drops=struct.unpack_from('<Q',reader.mem,16)[0]

bridge.market('SOXL',100.,{'values':{'41':100.01,'51':99.99,'account':'NEVER_FORWARD'}},True)
item=reader.read();check('numeric_market_roundtrip',item[2]==bridge.MARKET_V2 and bridge.TICK_V2.unpack(item[3])[1:4]==(0.,99.99,100.01))
check('secrets_never_serialized',b'NEVER_FORWARD' not in item[3])
bridge._lock.acquire()
start=time.perf_counter_ns();bridge.market('SOXL',100.,from_ws=True)
blocked_us=(time.perf_counter_ns()-start)/1000
bridge._lock.release();check('busy_producer_lock_drops_without_wait',blocked_us<10000)
try:reader.read()
except RuntimeError:check('producer_lock_drop_detected',True)
else:check('producer_lock_drop_detected',False)
reader.drops=struct.unpack_from('<Q',reader.mem,16)[0]

struct.pack_into('<d',reader.mem,8,0.)
seq=struct.unpack_from('<Q',reader.mem,0)[0]
bridge.market('SOXL',100.);check('dead_worker_fast_noop',seq==struct.unpack_from('<Q',reader.mem,0)[0])
reader.beat()
duplicate=bridge.Reader('Local\\LumosShadowDuplicateQA')
try:
    try:bridge.Reader('Local\\LumosShadowDuplicateQA')
    except RuntimeError:check('duplicate_consumer_blocked',True)
    else:check('duplicate_consumer_blocked',False)
finally:duplicate.close()

seq=struct.unpack_from('<Q',reader.mem,0)[0];bridge._qa_allow=False
bridge.market('SOXL',100.);check('unbound_research_source_ignored',struct.unpack_from('<Q',reader.mem,0)[0]==seq)
fake_engine=object();fake_streamer=object()
bridge.bind(fake_engine,fake_streamer,True)
check('simulation_cannot_bind',bridge._live_engine is None and bridge._live_streamer is None)
bridge.bind(fake_engine,fake_streamer,False)
bridge.market('SOXL',100.,source=object());check('wrong_streamer_ignored',struct.unpack_from('<Q',reader.mem,0)[0]==seq)
bridge.market('SOXL',100.,from_ws=True,source=fake_streamer);check('bound_streamer_publishes',reader.read() is not None)
bridge._qa_allow=True

from live_feed import PacketBuilder, decode_market
check('observed_ws_frame_decodes',decode_market(bridge.TICK.pack(b'SOXL',100.,99.,101.,True))[0]=='SOXL')
check('rest_poll_never_pretends_to_be_tick',decode_market(bridge.TICK.pack(b'SOXL',100.,0.,0.,False)) is None)
import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
native=pd.read_pickle(HOME/'inputs/native_features.pkl')
labels=pd.read_pickle(HOME/'inputs/labels.pkl')
builder=PacketBuilder();accepted=None;rejections={};feature_us=[]
for _,row in labels.tail(120).iterrows():
    pos=int(row.feature_row);frame=native.iloc[pos-31:pos+1]
    reader.beat()
    start=time.perf_counter_ns();bridge.features(frame);feature_us.append((time.perf_counter_ns()-start)/1000)
    item=reader.read()
    check('feature_transport_kind',item is not None and item[2]==2)
    try:p=builder.build(pd.Timestamp(row.decision_time).tz_localize('America/New_York').timestamp()+.1,item[3])
    except ValueError as e:rejections[str(e)]=rejections.get(str(e),0)+1;continue
    if p is not None:
        check('exact_unrounded_last72',np.array_equal(np.asarray(p['sequence'][-1]),frame[json.loads((HOME/'inputs/schema.json').read_text())['net_features']].iloc[-1].to_numpy()))
        accepted=p;break
check('actual_db_packet_accepted',accepted is not None)

from adapters import Candidate
outputs=[Candidate(n).predict(accepted) for n in json.loads((HOME/'policy.json').read_text())['models']]
check('all_five_models_consume_captured_input',len(outputs)==5)

# Optional observer cannot mutate caller input or require a working receiver.
before=native.tail(32).copy(deep=True);frame=before.copy(deep=True)
reader.beat()
bridge.features(frame);check('input_frame_not_mutated',frame.equals(before))
reader.close();bridge._memory=original;builder.db.close()
latency=dict(samples=len(samples),tick_p50_us=statistics.median(samples),tick_p99_us=sorted(samples)[9899],
             tick_max_us=max(samples),feature_capture_max_us=max(feature_us),busy_lock_us=blocked_us)
result=dict(status='PASS',checks=checks,latency=latency,packet_rejections=rejections,
            real_orders=0,broker_api_calls=0,fixture_market_events_not_performance=True,
            actual_live_session_validation='PENDING_NEXT_SESSION',live_ready=False)
(HOME/'qa/feed_transport.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
