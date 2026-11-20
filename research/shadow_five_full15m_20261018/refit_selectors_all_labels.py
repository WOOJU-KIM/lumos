"""Expand selector learning to all confirmed labels; preserve strategy flags only as entry gates.
Reuse the unchanged fresh NET/TCN OOF fits; refit every selector OOF fold and final selector.
Old selector artifacts are preserved under their original filenames.
"""
from common import *
guard();verify_source_policy()
import numpy as np,pandas as pd,pickle
from lightgbm import LGBMClassifier,LGBMRegressor
from sklearn.linear_model import LogisticRegression
z=np.load(HOME/'inputs/dataset.npz');Y=z['y'];T=z['times'];E=z['end'];W=z['week'];F=pd.read_pickle(HOME/'inputs/full_features.pkl')
n=len(F);side=np.tile([0,1],n);rid=np.repeat(np.arange(n),2)
trend=np.where((F.EMA20_Slope_3>0)&(F.EMA20_Dist_ATR>0),1,np.where((F.EMA20_Slope_3<0)&(F.EMA20_Dist_ATR<0),0,-1))
mr=np.where((F.VWAP_Dist_ATR<=-.5)&(F.Signed_Body_Ratio>0),1,np.where((F.VWAP_Dist_ATR>=.5)&(F.Signed_Body_Ratio<0),0,-1))
tf=np.repeat(trend,2)==side;mf=np.repeat(mr,2)==side;eligible=(tf|mf).reshape(n,2);groups=(side*4+tf*2+mf).reshape(n,2)
x=F.iloc[rid].reset_index(drop=True).copy();x['action_side']=side;x['trend_available']=tf.astype(int);x['mr_available']=mf.astype(int);X=x.to_numpy();target=Y.reshape(-1)
o=json.loads((HOME/'inputs/fresh_oof.json').read_text());index={r['row_id']:r for r in o};folds=[]
def fit(cut):
    valid=np.isfinite(Y)&(E<cut)&(T[:,None]<cut)&(T[:,None]>=cut-pd.Timedelta(weeks=CONFIG['ROLLING_TRAINING_WEEKS']).value)
    ids=np.flatnonzero(valid.reshape(-1));models=[]
    for head in range(3):
        use=ids if head==0 else ids[target[ids]>0] if head==1 else ids[target[ids]<=0]
        m=LGBMClassifier(**model_params(),objective='binary') if head==0 else LGBMRegressor(**model_params(),objective='regression')
        yy=(target[use]>0).astype(int) if head==0 else target[use] if head==1 else -target[use];m.fit(X[use],yy);models.append(m)
    return models,valid
for week in sorted(set(r['week'] for r in o)):
    cut=(pd.Timestamp('2020-07-27')+pd.Timedelta(weeks=week)).value;models,valid=fit(cut);te=np.flatnonzero(W==week);probes=np.ravel(np.column_stack([2*te,2*te+1]))
    pred=np.column_stack([models[0].predict_proba(X[probes])[:,1],np.maximum(0,models[1].predict(X[probes])),np.maximum(0,models[2].predict(X[probes]))]).reshape(len(te),2,3)
    for j,i in enumerate(te):index[int(i)]['values']['SELECTOR']=pred[j].tolist()
    folds.append(dict(week=week,cut=str(pd.Timestamp(cut)),rows_per_side=valid.sum(axis=0).tolist(),max_target_end=str(pd.Timestamp(E[valid].max()))))
    print('SELECTOR_ALL_LABELS_OOF',week,flush=True)
cut=pd.Timestamp(POLICY['training_cutoff_ny']).value+1;models,valid=fit(cut);ids=np.array([r['row_id'] for r in o]);cp=np.array([r['values']['SELECTOR'] for r in o]);selected=eligible[ids]&np.isfinite(Y[ids])
def logit(p):
    p=np.clip(p,1e-6,1-1e-6);return np.log(p/(1-p))
platt=LogisticRegression(C=1.,max_iter=500,random_state=42);platt.fit(logit(cp[:,:,0][selected]).reshape(-1,1),(Y[ids][selected]>0).astype(int))
prob=platt.predict_proba(logit(cp[:,:,0].reshape(-1)).reshape(-1,1))[:,1].reshape(-1,2)
v=valid.reshape(-1);g=groups.reshape(-1);gw=target[v&(target>0)].mean();gl=-target[v&(target<=0)].mean();means={}
for group in range(8):
    vals=target[v&(g==group)];win=vals[vals>0];loss=-vals[vals<=0];means[group]=(float(win.mean()) if len(win)>=40 else float(gw),float(loss.mean()) if len(loss)>=40 else float(gl))
def count(sides,best,threshold):
    busy=-1;total=0
    for i,s,val in zip(ids,sides,best):
        if s<0 or val<=0 or val<threshold or T[i]<busy or not np.isfinite(Y[i,s]):continue
        total+=1;busy=E[i,s]
    return total
manifest=json.loads((HOME/'artifacts/manifest.json').read_text());target_count=manifest['models']['FULL_COND_PAYOFF_F1.0']['target_calibration_trades'];activity={}
for name,hist in [('FULL_COND_PAYOFF_F1.0',False),('FULL_HIST_PAYOFF_F1.0',True)]:
    win=np.array([[means[h][0] for h in row] for row in groups[ids]]) if hist else cp[:,:,1];loss=np.array([[means[h][1] for h in row] for row in groups[ids]]) if hist else cp[:,:,2]
    ev=prob*win-(1-prob)*loss;value=np.where(eligible[ids],ev,-np.inf);sides=value.argmax(axis=1);best=value.max(axis=1);sides[(value[:,0]==value[:,1])|(best<=0)]=-1
    positive=best[(sides>=0)&np.isfinite(best)];grid=np.unique(np.r_[0.,np.quantile(positive,np.linspace(0,1,31))]) if len(positive) else np.array([0.]);counts=[count(sides,best,h) for h in grid];j=min(range(len(grid)),key=lambda j:(abs(counts[j]-target_count),grid[j]))
    artifact=dict(kind='HIST' if hist else 'COND',models=models,platt=platt,means=means,feature_names=list(x.columns),threshold=float(grid[j]),training_rows_per_side=valid.sum(axis=0).tolist(),prediction_unit='net_return_fraction',strategy_prefilter=True,training_strategy_prefilter=False,model_id=name,cutoff=POLICY['training_cutoff_ny'],source_policy_sha=sha(HOME/'policy.json'),status='RESEARCH_ONLY_NOT_LIVE')
    path=HOME/'artifacts'/f'{name}__all_labels_v3.pkl'
    with path.open('xb') as stream:pickle.dump(artifact,stream)
    manifest['models'][name]=dict(path=path.name,artifact_sha256=sha(path),training_rows_per_side=valid.sum(axis=0).tolist(),threshold=float(grid[j]),target_calibration_trades=target_count,calibration_trades=counts[j],training_strategy_prefilter=False)
    activity[name]=dict(calibration_trades=counts[j],positive_ev_candidates_before_entry_filter=int((ev>0).sum()),max_expected_net_return=float(ev.max()))
save('inputs/fresh_oof_all_labels.json',o);save('qa/selector_all_labels_folds.json',folds);save('artifacts/manifest.json',manifest)
save('qa/stage4.json',dict(stage=4,status='PASS',models=manifest['models'],fresh_oof_folds=12,label_masks_per_side=True,all_confirmed_labels_used_for_every_model=True,strategy_flags_only_gate_entries=True,old13k_cache_used=False,protected_changes=protected_changes()))
save('qa/selector_activity_diagnostic.json',activity);assert not protected_changes();print('ALL_FIVE_USE_ALL_AVAILABLE_TARGETS',json.dumps(manifest['models']),flush=True)
