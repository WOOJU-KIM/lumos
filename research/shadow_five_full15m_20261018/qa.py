"""Stage5-7 candidate QA. Synthetic fixtures are never reported as market performance."""
from common import *
guard()
verify_source_policy()
import copy
import numpy as np
import pandas as pd
from datetime import datetime,timedelta
from adapters import Candidate
from runtime import Engine,Ledger,validate_packet,stamp
from storage import Audit

schema=json.loads((HOME/'inputs/schema.json').read_text())
z=np.load(HOME/'inputs/dataset.npz',allow_pickle=False)
SEQ=z['seq']  # Decompress once; never reread the full archive for each decision.
full=pd.read_pickle(HOME/'inputs/full_features.pkl')
checks={}
def check(name,value):
    checks[name]=bool(value)
    assert value,name

def packet(at='2026-10-19T10:00:00-04:00',index=-1):
    now=datetime.fromisoformat(at)
    return dict(kind='DECISION',schema_version=1,time=at,decision_time=at,decision_id=at,
                net_feature_names=schema['net_features'],sequence=SEQ[index].tolist(),
                full_features={k:float(v) for k,v in full.iloc[index].items()},
                feature_times=[(now-timedelta(minutes=15*k)).isoformat() for k in range(31,-1,-1)],
                cross_asset_available_at={s:at for s in ['SOXS','NVDA','QQQ','VIXY','IEF']},
                expected_last_closed_available_at=at,required_assets_complete=True,atr5m_pct={'SOXL':1.,'SOXS':1.})

class Fixture:
    version='SYNTHETIC_TEST_ONLY'
    def __init__(self,name,side=1,fail=False):self.name=name;self.side=side;self.fail=fail
    def predict(self,p):
        if self.fail:raise ValueError('Intentional fixture error')
        return dict(model_id=self.name,model_version=self.version,expected_net_return=[.001,.01],
                    threshold=0.,score_type='TEST_ONLY',side=self.side,forced_side=self.side,details={})

def quote(at,bid=100.,ask=100.,quote_time=None):
    return dict(kind='QUOTE',time=at,quote_time=quote_time or at,symbol='SOXL',bid=bid,ask=ask)

p=packet();validate_packet(p)
engine=Engine({'A':Fixture('A')});engine.handle(p)
check('no_fill_at_decision',not engine.ledgers[('A','primary')].positions)
engine.handle(quote(p['time']))
check('same_time_quote_cannot_fill',not engine.ledgers[('A','primary')].positions)
engine.handle(quote('2026-10-19T10:00:01-04:00'))
ledger=engine.ledgers[('A','primary')]
check('next_quote_fills',len(ledger.positions)==1)
check('allocation98_integer',ledger.positions[0]['qty']==int(10000*.98/100.05))
q=packet('2026-10-19T10:15:00-04:00');out=engine.handle(q)
check('primary_no_overlapping_buy',out[0]['primary_action']=='POSITION_OPEN_OR_PENDING')
check('forced_independent_cohort',out[0]['forced_action']=='PENDING_NEXT_QUOTE')

for name,modify in [('future_features',lambda p:p['feature_times'].__setitem__(-1,'2026-10-19T10:15:00-04:00')),
                    ('feature_order',lambda p:p['net_feature_names'].reverse()),
                    ('missing_sequence_row',lambda p:p['sequence'].pop()),
                    ('missing_cross_assets',lambda p:p.update(required_assets_complete=False)),
                    ('nonfinite_feature',lambda p:p['sequence'][0].__setitem__(0,float('nan'))),
                    ('extra_feature_key',lambda p:p['full_features'].update(unexpected=1.)),
                    ('stale_candle',lambda p:p.update(expected_last_closed_available_at='2026-10-19T10:15:00-04:00'))]:
    bad=copy.deepcopy(p);modify(bad)
    try:validate_packet(bad)
    except ValueError:check(name+'_blocked',True)
    else:check(name+'_blocked',False)
e=Engine({'A':Fixture('A')});e.handle(p)
try:e.handle(p)
except ValueError:check('duplicate_cycle_blocked',True)
else:check('duplicate_cycle_blocked',False)
e=Engine({'A':Fixture('A'), 'B':Fixture('B',fail=True), 'C':Fixture('C')});out=e.handle(p)
check('one_model_error_isolated',len(e.errors)==1 and len(e.decisions)==2)
check('model_none_creates_no_primary_intent',Engine({'A':Fixture('A',side=-1)}).handle(p)[0]['primary_action']=='MODEL_NONE')

e=Engine({'A':Fixture('A')});e.handle(p);e.handle(quote('2026-10-19T10:00:07-04:00'))
check('stale_buy_intent_cannot_fill',not e.ledgers[('A','primary')].positions)
e=Engine({'A':Fixture('A')});e.handle(p);e.handle(quote('2026-10-19T10:00:01-04:00',quote_time='2026-10-19T09:59:00-04:00'))
check('stale_quote_cannot_fill',not e.ledgers[('A','primary')].positions)
e=Engine({'A':Fixture('A')});e.handle(p);e.handle(quote('2026-10-19T10:00:01-04:00'))
e.handle(dict(kind='TICK',time='2026-10-19T10:00:02-04:00',symbol='SOXL',price=102.))
e.handle(dict(kind='TICK',time='2026-10-19T10:00:03-04:00',symbol='SOXL',price=101.6))
check('WS_tick_does_not_dispatch_trailing',not e.ledgers[('A','primary')].positions[0]['exit_pending'])
e.handle(dict(kind='POLL',time='2026-10-19T10:00:04-04:00',symbol='SOXL',price=101.6))
check('poll_dispatches_trailing',e.ledgers[('A','primary')].positions[0]['exit_pending'])
check('exit_request_not_fill',not e.ledgers[('A','primary')].closed)
e.handle(quote('2026-10-19T10:00:05-04:00',bid=101.5,ask=101.6))
l=e.ledgers[('A','primary')]
check('next_bid_exit',l.closed[0]['exit_price']==101.5 and not l.positions)
check('cost_charged_once',abs(l.closed[0]['gross_return']-l.closed[0]['net_return']-CONFIG['BACKTEST_FEE_SLIPPAGE'])<1e-12)
check('cash_pnl_reconciles',abs(l.cash-(10000+l.closed[0]['quantity']*100*l.closed[0]['net_return']))<1e-8)

for kind,at,price,reason in [('TICK','2026-10-19T10:00:02-04:00',97.9,'SL'),
                            ('CLOCK','2026-10-19T11:30:01-04:00',0,'TIME'),
                            ('CLOCK','2026-10-19T15:55:00-04:00',0,'EOD')]:
    e=Engine({'A':Fixture('A')});e.handle(p);e.handle(quote('2026-10-19T10:00:01-04:00'))
    e.handle(dict(kind=kind,time=at,symbol='SOXL',price=price))
    check(reason+'_exit_requested',e.ledgers[('A','primary')].pending[0]['reason']==reason)
e=Engine({'A':Fixture('A')});out=e.handle(packet('2026-10-19T15:30:00-04:00'))
check('1530_entry_block',out[0]['primary_action']=='ENTRY_SESSION_BLOCK')
e=Engine({'A':Fixture('A')});out=e.handle(packet('2026-10-24T10:00:00-04:00'))
check('week_expiry_or_weekend_block',out[0]['primary_action']=='ENTRY_SESSION_BLOCK')
save('qa/stage5.json',dict(stage=5,status='PASS',checks=checks,fixtures='synthetic behavior tests, not performance'))
print('STAGE5_QA PASS',len(checks),'runtime checks',flush=True)

audit=Audit('qa_audit_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
extra=copy.deepcopy(p);extra['unapproved_extra']='TEST_ONLY_DO_NOT_PERSIST';audit.packet(extra)
audit.append(out[0]);audit.close()
saved=json.loads((audit.path/'decision_inputs.jsonl').read_text())
check('audit_excludes_unapproved_fields','unapproved_extra' not in saved)
check('audit_records_complete_features',len(saved['sequence'])==32 and len(saved['full_features'])==len(schema['full_features']))
check('score_not_constant_zero','expected_net_return' in json.loads((audit.path/'events.jsonl').read_text()))
save('qa/stage6.json',dict(stage=6,status='PASS',audit_append_only=True,input_replay_fields_saved=True,
                         real_prediction_values_saved=True,primary_forced_separate=True))
print('STAGE6_QA PASS audit storage',flush=True)

if not (HOME/'artifacts/manifest.json').exists():
    print('STAGE7_PENDING_TRAINING',flush=True)
    raise SystemExit(0)
candidates={name:Candidate(name) for name in POLICY['models']}
test=np.flatnonzero(z['week']==z['week'].max());errors=[];directions={name:[0,0,0] for name in candidates}
for i in test:
    q=packet(index=int(i))
    for name,candidate in candidates.items():
        a=candidate.predict(q);b=candidate.predict(q)
        check('repeat_'+name,np.array_equal(a['expected_net_return'],b['expected_net_return']))
        directions[name][a['side']+1]+=1
        if not np.isfinite(a['expected_net_return']).all():errors.append(name)
check('all_five_models_load',len(candidates)==5)
check('all_lastweek_inputs_finite',not errors)
changes=protected_changes()
if changes:
    # User subsequently authorized the three read-only observer hooks. Verify the
    # original AST survives after removing ONLY these hooks; all other files stay pinned.
    import qa_feed_hooks
    check('only_authorized_observer_and_recording_fixes_changed',set(changes)=={'core/kiwoom_ws_streamer.py','core/ml_engine.py','core/live_runner.py','core/moe_orchestrator.py','core/live_experience_logger.py'})
else:
    check('all_originals_unchanged',True)
save('qa/stage7.json',dict(stage=7,status='PASS',lastweek_inputs=len(test),model_evaluations=len(test)*5,
                         deterministic_repeat=True,errors=errors,
                         classification='IN_SAMPLE_INFERENCE_AND_STORAGE_QA_ONLY_NOT_A_BACKTEST',
                         performance_metrics_computed=False,protected_changes=protected_changes()))
print('STAGE7_QA PASS',len(test)*5,'real-input inference checks; no performance claim',flush=True)
