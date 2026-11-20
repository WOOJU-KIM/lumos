"""Fresh12week OOF calibration and final five-model refit on the full-DB, per-side dataset."""
from common import *
guard();verify_source_policy()
import numpy as np,pandas as pd,pickle,time
from lightgbm import LGBMClassifier,LGBMRegressor
from sklearn.linear_model import LogisticRegression
from feature_source import Engine
from network import torch,Net
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
Z=np.load(HOME/'inputs/dataset.npz');S=Z['seq'];Y=Z['y'];T=Z['times'];E=Z['end'];W=Z['week'];SV=Z['seq_valid']
F=pd.read_pickle(HOME/'inputs/full_features.pkl');NF=pd.read_pickle(HOME/'inputs/native_features.pkl')
SC=json.loads((HOME/'inputs/schema.json').read_text());P=model_params();START=time.time();MAXW=int(W.max());OOF=[];FOLDS=[]
side=np.tile([0,1],len(F));rid=np.repeat(np.arange(len(F)),2)
slope=F.EMA20_Slope_3.to_numpy();dist=F.EMA20_Dist_ATR.to_numpy();dev=F.VWAP_Dist_ATR.to_numpy();body=F.Signed_Body_Ratio.to_numpy()
trend=np.where((slope>0)&(dist>0),1,np.where((slope<0)&(dist<0),0,-1));mr=np.where((dev<=-.5)&(body>0),1,np.where((dev>=.5)&(body<0),0,-1))
TF=np.repeat(trend,2)==side;MF=np.repeat(mr,2)==side;EL=TF|MF;GROUP=side*4+TF.astype(int)*2+MF.astype(int)
XF=F.iloc[rid].reset_index(drop=True).copy();XF['action_side']=side;XF['trend_available']=TF.astype(int);XF['mr_available']=MF.astype(int);XA=XF.to_numpy();YY=Y.reshape(-1)
static=S[:,-1,:];lag=S[:,::-1,:].reshape(len(S),-1)

def masks(cut):
    low=cut-pd.Timedelta(weeks=CONFIG['ROLLING_TRAINING_WEEKS']).value
    return np.isfinite(Y)&(E<cut)&(T[:,None]<cut)&(T[:,None]>=low)

def fit_gbdt(cut,te):
    mask=masks(cut);out={};art={}
    for name,x in [('NET_GBDT',static),('NET_LAG_GBDT',lag)]:
        mm=mask&(True if name=='NET_GBDT' else SV[:,None]);models=[];pr=[]
        for s in [0,1]:
            ids=np.flatnonzero(mm[:,s]);assert len(ids)>100 and (T[ids].max()-T[ids].min())/86400e9>=728
            m=LGBMRegressor(**P,objective='regression');m.fit(x[ids],Y[ids,s]*100);models.append(m);pr.append(m.predict(x[te])/100)
        out[name]=np.column_stack(pr);art[name]=dict(kind=name,models=models,feature_names=SC['net_features'],threshold=0.,prediction_unit='net_return_fraction',training_rows_per_side=mm.sum(axis=0).tolist())
    # Learn from every confirmed directional label; strategy flags gate entries, not training data.
    ids=np.flatnonzero(mask.reshape(-1));models=[]
    for h in range(3):
        use=ids if h==0 else ids[YY[ids]>0] if h==1 else ids[YY[ids]<=0]
        m=LGBMClassifier(**P,objective='binary') if h==0 else LGBMRegressor(**P,objective='regression')
        target=(YY[use]>0).astype(int) if h==0 else YY[use] if h==1 else -YY[use]
        m.fit(XA[use],target);models.append(m)
    probe=np.ravel(np.column_stack([2*te,2*te+1]));pr=np.column_stack([models[0].predict_proba(XA[probe])[:,1],np.maximum(0,models[1].predict(XA[probe])),np.maximum(0,models[2].predict(XA[probe]))])
    out['SELECTOR']=pr.reshape(len(te),2,3);art['SELECTOR']=dict(models=models,feature_names=list(XF.columns),mask=mask,training_rows=int(len(ids)))
    return out,art

def train_net(mask,epochs,validation=None):
    ids=np.flatnonzero(mask.any(axis=1)&SV);assert len(ids)>100
    mean=S[ids].mean(axis=(0,1));std=np.maximum(S[ids].std(axis=(0,1)),1e-5);scaled=np.clip((S-mean)/std,-8,8).astype(np.float32)
    torch.manual_seed(42);rng=np.random.default_rng(42);bias=np.array([Y[mask[:,s],s].mean()*100 for s in [0,1]])
    net=Net(bias);opt=torch.optim.Adam(net.parameters(),lr=.001);target=np.nan_to_num(Y*100).astype(np.float32);history=[];best=float('inf');bestep=6;stale=0
    for epoch in range(epochs):
        net.train();order=rng.permutation(ids)
        for start in range(0,len(ids),256):
            ii=order[start:start+256];pred=net(torch.from_numpy(scaled[ii].transpose(0,2,1)));ma=torch.from_numpy(mask[ii].astype(np.float32))
            loss=(((pred-torch.from_numpy(target[ii]))**2)*ma).sum()/ma.sum()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
        if validation is not None:
            vi=np.flatnonzero(validation.any(axis=1)&SV);net.eval()
            with torch.no_grad():pp=np.concatenate([net(torch.from_numpy(scaled[vi[j:j+512]].transpose(0,2,1))).numpy() for j in range(0,len(vi),512)])
            mse=float((((pp-target[vi])**2)*validation[vi]).sum()/validation[vi].sum());history.append(mse)
            if mse<best-1e-5:best=mse;bestep=epoch+1;stale=0
            else:stale+=1
            if epoch+1>=6 and stale>=4:break
    return net,mean,std,max(6,bestep),history

def fit_tcn(cut,te):
    mask=masks(cut)&SV[:,None];vc=cut-pd.Timedelta(weeks=8).value;inner=masks(vc)&SV[:,None];valid=mask&(T[:,None]>=vc)
    _,_,_,epochs,hist=train_net(inner,24,valid)
    net,mean,std,_,_=train_net(mask,epochs);net.eval();scaled=np.clip((S[te]-mean)/std,-8,8).astype(np.float32)
    with torch.no_grad():pred=np.concatenate([net(torch.from_numpy(scaled[j:j+512].transpose(0,2,1))).numpy() for j in range(0,len(te),512)])/100
    return pred,dict(net=net,mean=mean,std=std,epochs=epochs,history=hist,training_rows_per_side=mask.sum(axis=0).tolist())

def baseline_predictions(cut,te):
    # Production recipe on all native15m history; this is only a frequency reference, never activated.
    past=NF[NF.datetime_dt<pd.Timestamp(cut)].copy();engine=Engine()
    labels=engine.compute_triple_barrier_labels(past,take_profit=getattr(__import__('feature_source').C,'LABEL_TP_PCT',CONFIG['MAX_TP_PCT']),stop_loss=CONFIG['SL_MIN_PCT'],horizon=int(CONFIG['TIME_STOP_MINUTES']/15))
    past['target']=labels.map({-1:0,0:1,1:2});past=past.iloc[:-int(CONFIG['TIME_STOP_MINUTES']/15)]
    past=past.loc[past.index>=past.index.max()-pd.Timedelta(weeks=CONFIG['ROLLING_TRAINING_WEEKS'])]
    good=np.isfinite(past[SC['net_features']].to_numpy()).all(axis=1);past=past.loc[good]
    m=LGBMClassifier(**P,objective='multiclass',num_class=3,class_weight='balanced');m.fit(past[SC['net_features']],past.target.astype(int))
    return m.predict_proba(pd.DataFrame(static[te],columns=SC['net_features']))

for week in range(MAXW-11,MAXW+1):
    cut=(pd.Timestamp('2020-07-27')+pd.Timedelta(weeks=week)).value;te=np.flatnonzero(W==week)
    out,_=fit_gbdt(cut,te);out['NET_TCN'],_=fit_tcn(cut,te);out['BASELINE']=baseline_predictions(cut,te)
    for j,i in enumerate(te):
        OOF.append(dict(row_id=int(i),week=week,values={name:v[j].tolist() for name,v in out.items()}))
    mask=masks(cut);assert (E[mask]<cut).all()
    FOLDS.append(dict(week=week,cut=str(pd.Timestamp(cut)),max_train_label_end=str(pd.Timestamp(E[mask].max())),train_rows_per_side=mask.sum(axis=0).tolist(),test_rows=len(te)))
    save('qa/fresh_oof_progress.json',FOLDS);print('FRESH_OOF',week,'/12folds',round(time.time()-START,1),'sec',flush=True)
cut=pd.Timestamp(POLICY['training_cutoff_ny']).value+1;te=np.array([len(S)-1]);_,ART=fit_gbdt(cut,te);_,TCN=fit_tcn(cut,te)
ids=np.array([r['row_id'] for r in OOF]);assert np.all(T[ids]<cut)
META={}
def artifact(name,bundle):
    path=HOME/'artifacts'/f'{name}.pkl';assert not path.exists()
    bundle.update(model_id=name,cutoff=POLICY['training_cutoff_ny'],source_policy_sha=sha(HOME/'policy.json'),status='RESEARCH_ONLY_NOT_LIVE')
    with path.open('xb') as stream:pickle.dump(bundle,stream)
    META[name]=dict(path=path.name,artifact_sha256=sha(path),training_rows_per_side=bundle.get('training_rows_per_side'),threshold=bundle['threshold'])

for name in ['NET_GBDT','NET_LAG_GBDT','NET_TCN']:
    pred=np.array([r['values'][name] for r in OOF]);bias=np.array([np.mean(Y[ids[np.isfinite(Y[ids,s])],s]-pred[np.isfinite(Y[ids,s]),s]) for s in [0,1]])
    if name=='NET_TCN':
        weights=HOME/'artifacts/NET_TCN.pt';torch.save(TCN['net'].state_dict(),weights)
        b=dict(kind=name,weights=weights.name,weights_sha256=sha(weights),feature_names=SC['net_features'],mean=TCN['mean'],std=TCN['std'],epochs=TCN['epochs'],training_rows_per_side=TCN['training_rows_per_side'],threshold=0.,prediction_unit='net_return_fraction')
    else:b=ART[name]
    b['bias']=bias;artifact(name,b)

cp=np.array([r['values']['SELECTOR'] for r in OOF]);raw_p=cp[:,:,0];selector_mask=EL.reshape(-1,2)[ids]&np.isfinite(Y[ids]);flat=raw_p[selector_mask]
def logit(p):
    p=np.clip(p,1e-6,1-1e-6);return np.log(p/(1-p))
platt=LogisticRegression(C=1.,max_iter=500,random_state=42);platt.fit(logit(flat).reshape(-1,1),(Y[ids][selector_mask]>0).astype(int))
prob=platt.predict_proba(logit(raw_p.reshape(-1)).reshape(-1,1))[:,1].reshape(-1,2)
mask=ART['SELECTOR']['mask'].reshape(-1);means={};globalwin=YY[mask&(YY>0)].mean();globalloss=-YY[mask&(YY<=0)].mean()
for g in range(8):
    vals=YY[mask&(GROUP==g)];win=vals[vals>0];loss=-vals[vals<=0]
    means[g]=(float(win.mean()) if len(win)>=40 else float(globalwin),float(loss.mean()) if len(loss)>=40 else float(globalloss))
def count(sides,scores,threshold):
    busy=-1;n=0
    for i,s,v in zip(ids,sides,scores):
        if s<0 or v<=0 or v<threshold or T[i]<busy or not np.isfinite(Y[i,s]):continue
        n+=1;busy=E[i,s]
    return n
raw=np.array([r['values']['BASELINE'] for r in OOF]);scores=np.clip(CONFIG['CALIB_BASE']+(raw.max(axis=1)-CONFIG['CALIB_PIVOT'])*CONFIG['CALIB_SCALE'],CONFIG['CALIB_BASE'],.95)
sides=np.where(raw[:,2]>raw[:,0],1,0);sides[(raw[:,1]>=np.maximum(raw[:,0],raw[:,2]))|(raw[:,0]==raw[:,2])|(scores<CONFIG['GBDT_CONFIDENCE_THRESHOLD'])]=-1
target=max(24,count(sides,np.ones(len(ids)),0))
for name,hist in [('FULL_COND_PAYOFF_F1.0',False),('FULL_HIST_PAYOFF_F1.0',True)]:
    groups=GROUP.reshape(-1,2)[ids]
    win=np.array([[means[g][0] for g in row] for row in groups]) if hist else cp[:,:,1]
    loss=np.array([[means[g][1] for g in row] for row in groups]) if hist else cp[:,:,2]
    ev=prob*win-(1-prob)*loss;ev=np.where(EL.reshape(-1,2)[ids],ev,-np.inf);sides=np.argmax(ev,axis=1);best=ev.max(axis=1);sides[(ev[:,0]==ev[:,1])|(best<=0)]=-1
    pos=best[(sides>=0)&np.isfinite(best)];grid=np.unique(np.r_[0.,np.quantile(pos,np.linspace(0,1,31))]) if len(pos) else np.array([0.]);counts=[count(sides,best,v) for v in grid];j=min(range(len(grid)),key=lambda j:(abs(counts[j]-target),grid[j]))
    b=ART['SELECTOR'];artifact(name,dict(kind='HIST' if hist else 'COND',models=b['models'],platt=platt,means=means,feature_names=b['feature_names'],threshold=float(grid[j]),training_rows_per_side=[int(mask[side==s].sum()) for s in [0,1]],prediction_unit='net_return_fraction',strategy_prefilter=True))
    META[name].update(target_calibration_trades=target,calibration_trades=counts[j])
save('inputs/fresh_oof.json',OOF);save('qa/fresh_oof_folds.json',FOLDS)
assert set(META)==set(POLICY['models']) and not protected_changes()
save('artifacts/manifest.json',dict(status='TRAINED_NOT_CONNECTED',runtime_enabled=False,cutoff=POLICY['training_cutoff_ny'],policy_sha256=sha(HOME/'policy.json'),models=META))
save('qa/stage4.json',dict(stage=4,status='PASS',models=META,fresh_oof_folds=12,label_masks_per_side=True,TCN_masked_missing_target_loss=True,old13k_cache_used=False,protected_changes=[]))
print('FULL15M_TRAIN_QA PASS',json.dumps(META),flush=True)
