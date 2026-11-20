"""Frozen 12-week paired-threshold B72 WFA; no live constructors or deployment.

All training methods are compiled unchanged from production. Candidate thresholds
are explicitly authorized sandbox overrides, applied to calibrated confidence.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *
guard();verify_source_policy()
import ast
import types
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Any,Optional,Dict,List,Tuple
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
import feature_source
from feature_source import Engine as PureFeatures, Lake,C
os.environ['MPLCONFIGDIR']=str(HOME/'cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=HOME/'threshold_64_704_20261018';OUT.mkdir(exist_ok=False)
START=pd.Timestamp('2026-07-27');END=pd.Timestamp('2026-10-16 23:59:59')
THRESHOLDS=(.64,.704)
before={str(p):sha(p) for root in ['models','backups','core'] for p in (PROJECT/root).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
before[str(PROJECT/'config.py')]=sha(PROJECT/'config.py')
for p in (PROJECT/'data').glob('*'):
    if p.is_file() and p.suffix in ['.db','.csv','.json']:before[str(p)]=sha(p)
before[str(PROJECT/'scripts/run_weekly_rolling_wfa.py')]=sha(PROJECT/'scripts/run_weekly_rolling_wfa.py')
source=ast.parse((PROJECT/'core/ml_engine.py').read_text(encoding='utf-8-sig'))
cl=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='MLFeatureEngine')
snap=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='_ModelSnapshot')
methods={'train_and_select_top_features','_report_live_sample_coverage','compute_feature_names_sha256','_apply_snapshot'}
derived=ast.ClassDef(name='OfflineEngine',bases=[ast.Name(id='PureFeatures',ctx=ast.Load())],keywords=[],body=[n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name in methods],decorator_list=[])
module=types.ModuleType('lumos_threshold_offline_source');sys.modules[module.__name__]=module
def classifier(**kwargs):return LGBMClassifier(**dict(kwargs,n_jobs=1))
module.__dict__.update(np=np,pd=pd,config=C,logging=logging,logger=logging.getLogger('threshold_wfa'),threading=threading,datetime=datetime,
    dataclass=dataclass,Any=Any,Optional=Optional,Dict=Dict,List=List,Tuple=Tuple,PureFeatures=PureFeatures,LGBMClassifier=classifier)
exec(compile(ast.fix_missing_locations(ast.Module(body=[snap,derived],type_ignores=[])),'UNCHANGED_PRODUCTION_TRAINING','exec'),module.__dict__)
Engine=module.OfflineEngine;feature_source.ns['MLFeatureEngine']=Engine
Z=np.load(HOME/'inputs/dataset.npz');D=pd.read_pickle(HOME/'inputs/labels.pkl');F=pd.read_pickle(HOME/'inputs/full_features.pkl')
NAMES=list(Engine.CANONICAL_B_NORMALIZED_72_FEATURES);T=pd.to_datetime(Z['times']);Y=Z['y']
selected=(T>=START)&(T<=END);ids=np.flatnonzero(selected)
bad_days=sorted(set(str(T[i].date()) for i in ids if not np.isfinite(Y[i]).all()))
assert len(bad_days)==4,'Frozen data changed; review coverage rather than silently changing the sample'
good=np.array([str(t.date()) not in bad_days for t in T])
assert not CONFIG['USE_CROSS_ASSET_VETO'] and not CONFIG['USE_60M_TREND_FILTER'],'Enabled shields need their actual historical inputs'
assert CONFIG['RSI_OVERBOUGHT_THRESHOLD']==100 and CONFIG['RSI_OVERSOLD_THRESHOLD']==0
assert CONFIG['BACKTEST_FEE_SLIPPAGE']>=.002
raw=Lake().load_candles(CONFIG['BASE_ASSET_LONG'],'15m')
prob=np.full((len(T),3),np.nan);folds=[]
for week in sorted(set(Z['week'][ids])):
    cut=pd.Timestamp('2020-07-27')+pd.Timedelta(weeks=int(week));test=ids[Z['week'][ids]==week]
    past=raw.loc[raw.index+pd.Timedelta(minutes=15)<=cut].copy()
    engine=Engine();engine.feature_schema='B_normalized'
    # This is the pre-live-weight B72 recipe. The empty frame also prevents any
    # automatic access to application experience storage or future real labels.
    model,_,_=engine.train_and_select_top_features(past,live_experience_df=pd.DataFrame(),schema='B_normalized',exclude_unconfirmed_tail=True)
    assert model is not None and engine.feature_names==NAMES
    meta=engine.last_training_window_meta
    assert meta['span_days']>=728 and pd.Timestamp(meta['raw_end_dt'])+pd.Timedelta(minutes=15)<=cut
    assert pd.Timestamp(meta['train_end_dt'])<cut and meta['excluded_unconfirmed_tail_rows']==6
    assert np.all(D.iloc[test].feature_available_at.to_numpy()<=T[test].to_numpy())
    prob[test]=model.predict_proba(F.iloc[test][NAMES])
    assert np.isfinite(prob[test]).all() and np.allclose(prob[test].sum(axis=1),1)
    folds.append(dict(week=int(week),cut=str(cut),training_rows=meta['train_rows'],training_start=meta['train_start_dt'],
        training_last_label_origin=meta['train_end_dt'],last_raw_bar=meta['raw_end_dt'],test_rows=len(test),live_rows=0))
    print('WFA_FIT',str(cut.date()),meta['train_rows'],flush=True)

score=np.clip(CONFIG['CALIB_BASE']+(prob.max(axis=1)-CONFIG['CALIB_PIVOT'])*CONFIG['CALIB_SCALE'],CONFIG['CALIB_BASE'],.95)
direction=np.full(len(T),-1,int)
direction[(prob[:,0]>prob[:,1])&(prob[:,0]>prob[:,2])]=0
direction[(prob[:,2]>prob[:,1])&(prob[:,2]>prob[:,0])]=1
assert (direction[(prob[:,1]>=prob[:,0])&(prob[:,1]>=prob[:,2])]==-1).all()
five={};atr={}
for symbol in [CONFIG['BASE_ASSET_SHORT'],CONFIG['BASE_ASSET_LONG']]:
    d=Lake().load_candles(symbol,'5m');tr=pd.concat([d.High-d.Low,(d.High-d.Close.shift()).abs(),(d.Low-d.Close.shift()).abs()],axis=1).max(axis=1)
    atr[symbol]=tr.rolling(14).mean();five[symbol]=d

def path(symbol,t):
    d=five[symbol];loc=d.index.get_loc(t);assert isinstance(loc,(int,np.integer)) and loc>=14
    row=d.iloc[loc];entry=float(row.Open);assert entry>0 and row.Volume>0
    stop_pct=float(np.clip(atr[symbol].iloc[loc-1]/entry*CONFIG['SL_ATR_MULTIPLIER'],CONFIG['SL_MIN_PCT'],CONFIG['SL_MAX_PCT']))
    initial=entry*(1-stop_pct);stop=initial;tp=entry*(1+CONFIG['MAX_TP_PCT']);peak=entry
    # All evaluation sessions in this fixed period have the regular NYSE close.
    eod_hour,eod_minute=map(int,CONFIG['PHASE_EOD_CLEAR'].split(':'))
    eod=t.normalize()+pd.Timedelta(hours=eod_hour,minutes=eod_minute)
    deadline=min(t+pd.Timedelta(minutes=CONFIG['TIME_STOP_MINUTES']),eod);points=[];at=t
    while at<deadline:
        assert at in d.index,'Unresolved missing path must fail, never skip a losing trade'
        r=d.loc[at];o,h,l,c=map(float,[r.Open,r.High,r.Low,r.Close]);assert r.Volume>0 and np.isfinite([o,h,l,c]).all()
        # SL/TRAIL already fixed from prior candles; never update with current
        # high before checking this candle's low. SL wins simultaneous TP touch.
        reason='TRAIL' if stop>initial else 'SL'
        if o<=stop:exit_price=o;exit_time=at;break
        if l<=stop:exit_price=stop;exit_time=at+pd.Timedelta(minutes=5);break
        if h>=tp:exit_price=tp;exit_time=at+pd.Timedelta(minutes=5);reason='TP';break
        points.append((at+pd.Timedelta(minutes=5),c));peak=max(peak,h)
        if peak>=entry*(1+CONFIG['TRAILING_TRIGGER_PCT']):
            stop=max(stop,entry*1.002,peak*(1-CONFIG['TRAILING_DROP_PCT']))
        at+=pd.Timedelta(minutes=5)
    else:
        assert deadline in d.index and d.loc[deadline].Volume>0
        exit_price=float(d.loc[deadline].Open);exit_time=deadline;reason='EOD' if deadline==eod else 'TIME'
    return dict(entry_price=entry,exit_price=exit_price,exit_time=exit_time,reason=reason,
        net_return=exit_price/entry-1-CONFIG['BACKTEST_FEE_SLIPPAGE'],sl_pct=stop_pct,points=points)

trades={};equities={};audit=[]
for threshold in THRESHOLDS:
    cash=10000.;busy=pd.Timestamp.min;rows=[];eq=[(START,cash)];total_signals=0;blocked=0
    for i in ids:
        status='NONE'
        if not good[i]:status='DATA_DAY_EXCLUDED'
        elif direction[i]>=0 and score[i]>=threshold:
            total_signals+=1
            if T[i]<busy:blocked+=1;status='POSITION_BLOCKED'
            else:
                s=direction[i];symbol=CONFIG['BASE_ASSET_LONG'] if s==1 else CONFIG['BASE_ASSET_SHORT']
                assert np.isfinite(Y[i,s])
                result=path(symbol,T[i]);entry=result['entry_price'];qty=int(cash*CONFIG['MAX_ALLOCATION_RATIO']/(entry+CONFIG['QTY_CALC_BUFFER']))
                assert qty>0
                before_cash=cash
                for at,close in result.pop('points'):
                    if at<result['exit_time']:
                        eq.append((at,before_cash+qty*(close-entry)-qty*entry*CONFIG['BACKTEST_FEE_SLIPPAGE']))
                pnl=qty*entry*result['net_return'];cash+=pnl;busy=result['exit_time'];eq.append((busy,cash))
                rows.append(dict(threshold=threshold,entry_time=T[i],symbol=symbol,confidence=float(score[i]),quantity=qty,
                    pnl_usd=pnl,equity=cash,**result));status='TRADE'
        audit.append(dict(time=str(T[i]),threshold=threshold,status=status,confidence=float(score[i]),side=int(direction[i]),
            raw_short=float(prob[i,0]),raw_neutral=float(prob[i,1]),raw_long=float(prob[i,2])))
    trades[threshold]=pd.DataFrame(rows);equities[threshold]=pd.DataFrame(eq,columns=['time','equity'])
    trades[threshold].to_csv(OUT/f'trades_{threshold}.csv',index=False)

def metrics(threshold,start,end):
    rows=trades[threshold];rows=rows[(rows.entry_time>=start)&(rows.entry_time<=end)]
    earlier=trades[threshold].loc[trades[threshold].entry_time<start];initial=10000. if earlier.empty else float(earlier.iloc[-1].equity)
    final=initial+rows.pnl_usd.sum();days=(end.normalize()-start.normalize()).days+1
    curve=equities[threshold];curve=curve.loc[(curve.time>=start)&(curve.time<=end)].copy()
    values=np.r_[initial,curve.equity.to_numpy(),final];mdd=float((values/np.maximum.accumulate(values)-1).min())
    wins=rows.loc[rows.pnl_usd>0,'pnl_usd'].sum();loss=-rows.loc[rows.pnl_usd<0,'pnl_usd'].sum()
    weeks=12 if start==START else 1
    return dict(period=f'{start.date()} ~ {end.date()}',threshold=threshold,trades=len(rows),weekly_trades=len(rows)/weeks,
        return_pct=(final/initial-1)*100,CAGR_pct=((final/initial)**(365.25/days)-1)*100,MDD_pct=mdd*100,
        win_rate_pct=float((rows.pnl_usd>0).mean()*100),profit_factor=float(wins/loss) if loss else None,
        avg_trade_net_pct=float(rows.net_return.mean()*100),SOXL=int((rows.symbol==CONFIG['BASE_ASSET_LONG']).sum()),SOXS=int((rows.symbol==CONFIG['BASE_ASSET_SHORT']).sum()),
        average_hold_minutes=float(((rows.exit_time-rows.entry_time).dt.total_seconds()/60).mean()),
        exits=rows.reason.value_counts().to_dict())
summary=[metrics(th,start,END) for start in [START,pd.Timestamp('2026-10-12')] for th in THRESHOLDS]
assert all(sha(Path(p))==digest for p,digest in before.items()),'Original data/model/config changed'
comparison=dict(status='PASS',method='WEEKLY_EXPANDING_WFA_UNCHANGED_PRE_LIVE_B72_TRAINING_RECIPE',
    distinction='Historical weekly models, not the Oct16-trained frozen artifact replayed on its training data',
    start=str(START.date()),end=str(END.date()),source_model_family='B_Normalized_72',
    thresholds=list(THRESHOLDS),raw_probability_equivalents=[CONFIG['CALIB_PIVOT']+(th-CONFIG['CALIB_BASE'])/CONFIG['CALIB_SCALE'] for th in THRESHOLDS],
    fee_roundtrip=CONFIG['BACKTEST_FEE_SLIPPAGE'],allocation=CONFIG['MAX_ALLOCATION_RATIO'],folds=folds,
    total_sessions=len(set(T[ids].date)),eligible_sessions=len(set(T[ids[good[ids]]].date)),excluded_sessions=bad_days,
    evaluation_rows=len(ids),eligible_rows=int(good[ids].sum()),results=summary,broker_api_calls=0,real_orders=0,
    protected_files=len(before),protected_changes=[],MDD_definition='5m close mark-to-market plus realized exit points',
    limitations=['5m OHLC execution proxy differs from live tick/quotes; no claim of live return parity',
        'Four complete-session exclusions applied identically before threshold comparison',
        'Annualized return from 12 weeks or one week is descriptive, not a forecast',
        'Prices at first 5m open after completed15m decision; actual inference latency and spread approximated by cost reserve'])
(OUT/'results.json').write_text(json.dumps(comparison,indent=2,default=str,allow_nan=False),encoding='utf-8')
pd.DataFrame(audit).to_csv(OUT/'signal_audit.csv',index=False)
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False})
fig,axes=plt.subplots(2,1,figsize=(10,7),gridspec_kw={'height_ratios':[2,1]},constrained_layout=True)
for th in THRESHOLDS:
    e=equities[th];axes[0].step(e.time,(e.equity/10000-1)*100,where='post',label=f'확신도 {th*100:g}%')
    vals=e.equity.to_numpy();axes[1].step(e.time,(vals/np.maximum.accumulate(vals)-1)*100,where='post',label=f'{th*100:g}%')
axes[0].set(title='기존 B72 학습 방식: 64% vs 70.4% | 주간 워크포워드',ylabel='누적 순수익률 (%)')
axes[1].set(ylabel='낙폭 (%)',xlabel='뉴욕 거래일')
for ax in axes:ax.axhline(0,color='gray',lw=.7);ax.grid(alpha=.2);ax.legend()
fig.suptitle('2026-07-27 ~ 10-16 · 동일 55거래일 · 왕복 비용 0.20% · 15분 판단 / 5분 청산',fontsize=11)
fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
print('THRESHOLD_COMPARISON',json.dumps(comparison,default=str),flush=True)
