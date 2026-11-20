"""Offline shadow-only refit. Real chosen-side outcomes, never inverted class labels.

Historical 32-row inputs are REPLAY context plus the archived live scalar snapshot.
They are deliberately not claimed to be exact observed inference tensors. The
production model, its manifest, broker and operational databases are untouched.
"""
from common import *
guard(); verify_source_policy()
import ast
import copy
import pickle
import sqlite3
from collections import Counter
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from lightgbm import LGBMClassifier, LGBMRegressor
from network import torch, Net

VERSION='live_weighted_v1'
OUT=HOME/'artifacts'/VERSION
OUT.mkdir(exist_ok=False)
LIVE_WEIGHT=2.5  # Same existing production live-experience weight; research-only setting.
NY=ZoneInfo('America/New_York'); KST=ZoneInfo('Asia/Seoul')
CUT=pd.Timestamp(POLICY['training_cutoff_ny']).value+1
protected=[PROJECT/'config.py', *[p for root in ['models','backups'] for p in (PROJECT/root).rglob('*') if p.is_file()]]
protected += [p for p in (PROJECT/'data').glob('*') if p.is_file() and p.suffix in ['.db','.csv','.json']]
protected += [p for p in (HOME/'artifacts').glob('*') if p.is_file()]
before={str(p):sha(p) for p in protected}
SC=json.loads((HOME/'inputs/schema.json').read_text())
Z=np.load(HOME/'inputs/dataset.npz'); S=Z['seq']; Y=Z['y'].copy(); T=Z['times']; E=Z['end']; SV=Z['seq_valid']
F=pd.read_pickle(HOME/'inputs/full_features.pkl'); D=pd.read_pickle(HOME/'inputs/labels.pkl')
base_n=len(S); WEIGHTS=np.ones_like(Y)
db=sqlite3.connect((PROJECT/'data/live_experience.db').as_uri()+'?mode=ro',uri=True,timeout=0)
db.execute('pragma query_only=ON')
live=pd.read_sql_query('select * from live_trades order by entry_time',db); db.close()
calendar=mcal.get_calendar('NYSE').schedule(start_date='2026-01-01',end_date=str(pd.Timestamp(CUT).date()))
closes={str(day.date()):row.market_close.tz_convert(NY).tz_localize(None) for day,row in calendar.iterrows()}
accepted=[]; excluded=Counter(); extras=[]; newf=[]; seen=set(); replaced=0
for _,r in live.iterrows():
    try:
        entry=pd.Timestamp(r.entry_time).tz_localize(KST).tz_convert(NY).tz_localize(None)
        end=pd.Timestamp(r.exit_time).tz_localize(KST).tz_convert(NY).tz_localize(None)
        if not str(r['mode']).startswith('REAL'): excluded['NOT_REAL']+=1; continue
        if r.symbol not in [CONFIG['BASE_ASSET_SHORT'],CONFIG['BASE_ASSET_LONG']]: excluded['OTHER_ASSET']+=1; continue
        if not entry<end or end.value>=CUT or entry.value<CUT-pd.Timedelta(weeks=CONFIG['ROLLING_TRAINING_WEEKS']).value:
            excluded['OUTSIDE_CONFIRMED_WINDOW']+=1; continue
        f=json.loads(r.features_json)
        if 'panic_signal' not in f and all(k in f and float(f[k]) in [-1.,0.,1.] for k in ['vixy_dir_1','ief_dir_1']):
            f['panic_signal']=int(f['vixy_dir_1']>0)+int(f['ief_dir_1']>0)
        if any(k not in f or not np.isfinite(float(f[k])) for k in SC['full_features'] if k!='minutes_to_eod'):
            excluded['INCOMPLETE_FEATURES']+=1; continue
        # Match the actual evaluation cycle, never a later bar. All history was
        # available by this cycle; the scalar snapshot itself is legacy/rounded.
        cycle=entry.floor('15min'); matches=np.flatnonzero(T==cycle.value)
        if len(matches)!=1 or not SV[matches[0]]: excluded['NO_REPLAY_CONTEXT']+=1; continue
        i=int(matches[0]); side=int(r.symbol==CONFIG['BASE_ASSET_LONG'])
        key=(cycle.value,side)
        if key in seen: excluded['DUPLICATE_CYCLE_SIDE']+=1; continue
        if str(cycle.date()) not in closes: excluded['NO_SESSION']+=1; continue
        gross=float(r.pnl_pct)/100
        if not np.isfinite(gross): excluded['INVALID_OUTCOME']+=1; continue
        assert pd.Timestamp(D.iloc[i].feature_available_at)<=cycle<=entry<end
        seq=S[i].copy(); seq[-1]=[float(f[k]) for k in SC['net_features']]
        f['minutes_to_eod']=(closes[str(cycle.date())]-pd.Timedelta(minutes=5)-cycle).total_seconds()/60
        full=[float(f[k]) for k in SC['full_features']]
        assert np.isfinite(seq).all() and np.isfinite(full).all()
        # Replace the matching historical chosen-side label, avoiding double
        # counting one cycle as both proxy and real experience. Opposite labels
        # remain untouched; opposite labels on the live row stay missing.
        replaced+=int(np.isfinite(Y[i,side])); Y[i,side]=np.nan
        yy=np.full(2,np.nan); yy[side]=gross-max(.002,CONFIG['BACKTEST_FEE_SLIPPAGE'])
        ee=np.full(2,-1,dtype=np.int64); ee[side]=end.value
        ww=np.zeros(2); ww[side]=LIVE_WEIGHT
        extras.append((seq,yy,cycle.value,ee,ww)); newf.append(full); seen.add(key)
        accepted.append(dict(cycle=str(cycle),exit=str(end),side=side,base_row=i,
            sequence_provenance='REPLAY_ARCHIVED_HISTORY_PLUS_ROUNDED_LIVE_SNAPSHOT',
            target_provenance='APP_GROSS_QUOTE_PROXY_MINUS_ESTIMATED_ROUNDTRIP_COST',
            cost=max(.002,CONFIG['BACKTEST_FEE_SLIPPAGE']),weight=LIVE_WEIGHT))
    except (ValueError,TypeError,KeyError): excluded['MALFORMED_RECORD']+=1
assert extras,'No complete real samples; do not pretend weighting was applied'
S=np.concatenate([S,np.stack([x[0] for x in extras])]); Y=np.vstack([Y,*[x[1] for x in extras]])
T=np.r_[T,[x[2] for x in extras]]; E=np.vstack([E,*[x[3] for x in extras]])
WEIGHTS=np.vstack([WEIGHTS,*[x[4] for x in extras]]); SV=np.r_[SV,np.ones(len(extras),bool)]
F=pd.concat([F,pd.DataFrame(newf,columns=F.columns)],ignore_index=True)
mask=np.isfinite(Y)&(E<CUT)&(T[:,None]<CUT)&(T[:,None]>=CUT-pd.Timedelta(weeks=CONFIG['ROLLING_TRAINING_WEEKS']).value)
assert (WEIGHTS[base_n:][mask[base_n:]]==LIVE_WEIGHT).all()
assert np.isfinite(Y[base_n:]).sum()==len(extras)  # No counterfactual real labels.
oof_folds=json.loads((HOME/'qa/fresh_oof_folds.json').read_text())
# Every accepted real trade postdates every OOF training cutoff. Reusing the
# existing purged OOF predictions/calibration is exact for those unchanged fits.
assert all(pd.Timestamp(a['cycle'])>=pd.Timestamp(f['cut']) for a in accepted for f in oof_folds)
oldmanifest=json.loads((HOME/'artifacts/manifest.json').read_text()); manifest=copy.deepcopy(oldmanifest)
P=model_params(); bundles={}; fit_audit={}
for name,x in [('NET_GBDT',S[:,-1]),('NET_LAG_GBDT',S[:,::-1].reshape(len(S),-1))]:
    valid=mask&(True if name=='NET_GBDT' else SV[:,None]); models=[]
    for side in [0,1]:
        ii=np.flatnonzero(valid[:,side]); model=LGBMRegressor(**P,objective='regression')
        model.fit(x[ii],Y[ii,side]*100,sample_weight=WEIGHTS[ii,side]); models.append(model)
    with (HOME/'artifacts'/oldmanifest['models'][name]['path']).open('rb') as stream: b=pickle.load(stream)
    b.update(models=models,training_rows_per_side=valid.sum(axis=0).tolist())
    bundles[name]=b
    fit_audit[name]=dict(live_rows_per_side=valid[base_n:].sum(axis=0).tolist(),weight=LIVE_WEIGHT)
    print('FIT_DONE',name,fit_audit[name],flush=True)

side=np.tile([0,1],len(F)); rid=np.repeat(np.arange(len(F)),2)
trend=np.where((F.EMA20_Slope_3>0)&(F.EMA20_Dist_ATR>0),1,np.where((F.EMA20_Slope_3<0)&(F.EMA20_Dist_ATR<0),0,-1))
mr=np.where((F.VWAP_Dist_ATR<=-.5)&(F.Signed_Body_Ratio>0),1,np.where((F.VWAP_Dist_ATR>=.5)&(F.Signed_Body_Ratio<0),0,-1))
tf=np.repeat(trend,2)==side; mf=np.repeat(mr,2)==side; groups=side*4+tf*2+mf
X=F.iloc[rid].reset_index(drop=True).copy(); X['action_side']=side; X['trend_available']=tf.astype(int); X['mr_available']=mf.astype(int)
target=Y.reshape(-1); weights=WEIGHTS.reshape(-1); v=mask.reshape(-1); ids=np.flatnonzero(v); models=[]
heads=[]
for head in range(3):
    use=ids if head==0 else ids[target[ids]>0] if head==1 else ids[target[ids]<=0]
    m=LGBMClassifier(**P,objective='binary') if head==0 else LGBMRegressor(**P,objective='regression')
    yy=(target[use]>0).astype(int) if head==0 else target[use] if head==1 else -target[use]
    m.fit(X.iloc[use],yy,sample_weight=weights[use]); models.append(m)
    heads.append(int((use>=base_n*2).sum()))
def avg(sel): return float(np.average(target[sel],weights=weights[sel]))
gw=avg(v&(target>0)); gl=-avg(v&(target<=0)); means={}
for group in range(8):
    win=v&(groups==group)&(target>0); loss=v&(groups==group)&(target<=0)
    means[group]=(avg(win) if win.sum()>=40 else gw,-avg(loss) if loss.sum()>=40 else gl)
for name in ['FULL_COND_PAYOFF_F1.0','FULL_HIST_PAYOFF_F1.0']:
    with (HOME/'artifacts'/oldmanifest['models'][name]['path']).open('rb') as stream: b=pickle.load(stream)
    b.update(models=models,means=means,training_rows_per_side=mask.sum(axis=0).tolist())
    bundles[name]=b; fit_audit[name]=dict(live_rows_per_side=mask[base_n:].sum(axis=0).tolist(),live_rows_per_head=heads,weight=LIVE_WEIGHT)
    print('FIT_DONE',name,fit_audit[name],flush=True)

# Keep the original architecture, batching and previously selected epoch count.
# Change only its missing-target MSE to per-head sample-weighted masked MSE.
source=ast.parse((HOME/'train.py').read_text())
fn=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='train_net')
code=ast.unparse(fn).replace('mask[ii].astype(np.float32)','(mask[ii] * WEIGHTS[ii]).astype(np.float32)')
assert 'mask[ii] * WEIGHTS[ii]' in code
exec(compile(code,'ISOLATED_WEIGHTED_TCN','exec'),globals())
torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
with (HOME/'artifacts'/oldmanifest['models']['NET_TCN']['path']).open('rb') as stream: b=pickle.load(stream)
valid=mask&SV[:,None]; net,mean,std,_,_=train_net(valid,b['epochs']); net.eval()
# Full repeat verifies determinism, including the weighted loss and shuffling.
again,mean2,std2,_,_=train_net(valid,b['epochs'])
assert all(torch.equal(value,again.state_dict()[key]) for key,value in net.state_dict().items())
assert np.array_equal(mean,mean2) and np.array_equal(std,std2)
weights_path=OUT/'NET_TCN.pt'; torch.save(net.state_dict(),weights_path)
b.update(weights=str(weights_path.relative_to(HOME/'artifacts')),weights_sha256=sha(weights_path),mean=mean,std=std,training_rows_per_side=valid.sum(axis=0).tolist())
bundles['NET_TCN']=b; fit_audit['NET_TCN']=dict(live_rows_per_side=valid[base_n:].sum(axis=0).tolist(),weight=LIVE_WEIGHT,weighted_loss=True,deterministic_repeat=True)
print('FIT_DONE NET_TCN',fit_audit['NET_TCN'],flush=True)

for name,b in bundles.items():
    b.update(live_experience_weight=LIVE_WEIGHT,live_experience_rows=len(extras),
        live_experience_provenance='LEGACY_ROUNDED_SCALAR_AND_REPLAY_CONTEXT_ESTIMATED_NET',
        calibration_status='FROZEN_PURGED_OOF_BEFORE_LIVE_SAMPLES',training_version=VERSION)
    path=OUT/(name+'.pkl')
    with path.open('xb') as stream: pickle.dump(b,stream)
    manifest['models'][name].update(path=str(path.relative_to(HOME/'artifacts')),artifact_sha256=sha(path),
        training_rows_per_side=b['training_rows_per_side'],live_rows_per_side=fit_audit[name]['live_rows_per_side'],live_experience_weight=LIVE_WEIGHT)
manifest['live_weighted_version']=VERSION
save('artifacts/'+VERSION+'/manifest.json',manifest)

from adapters import Candidate
checks={}; probe=np.flatnonzero(Z['seq_valid'])[-120:]
for name in bundles:
    c=Candidate(name,OUT/'manifest.json'); old=Candidate(name); deltas=[]
    for i in probe:
        packet=dict(net_feature_names=SC['net_features'],sequence=S[i],full_features=F.iloc[i].to_dict())
        a=c.predict(packet); repeat=c.predict(packet); ref=old.predict(packet)
        assert a==repeat and np.isfinite(a['expected_net_return']).all()
        deltas.append(np.abs(np.array(a['expected_net_return'])-ref['expected_net_return']))
    checks[name]=dict(real_input_probes=len(probe),deterministic_predictions=True,
        mean_absolute_ev_change=float(np.mean(deltas)),performance_claim=False)
assert all(sha(Path(p))==digest for p,digest in before.items())
result=dict(status='PASS_NOT_YET_PROMOTED',live_weight=LIVE_WEIGHT,live_rows=len(extras),
    side_counts=dict(SOXL=sum(a['side']==1 for a in accepted),SOXS=sum(a['side']==0 for a in accepted)),
    excluded=dict(excluded),replaced_historical_chosen_side_labels=replaced,base_rows=base_n,
    fit_audit=fit_audit,prediction_qa=checks,source_sha256=sha(Path(__file__)),
    protected_files=len(before),protected_changes=[],api_calls=0,
    warnings=['Legacy snapshots rounded and not guaranteed exact inference tensors',
        'Sequence context REPLAY, not observed live 32-row tensors',
        'Gross quote-proxy outcomes minus estimated cost, not broker-reconciled net',
        'All usable real outcomes SOXL; no invented SOXS outcomes',
        'Calibrators, thresholds, TCN epochs frozen from prior purged OOF; require new OOS validation'],
    calibration_reuse_proof=dict(fold_count=len(oof_folds),latest_training_cut=max(f['cut'] for f in oof_folds),earliest_live_cycle=min(a['cycle'] for a in accepted)))
save('artifacts/'+VERSION+'/training_qa.json',result)
print('WEIGHTED_SHADOW_QA',json.dumps(result),flush=True)
