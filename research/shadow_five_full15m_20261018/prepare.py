"""Rebuild from every current DB15m row, with independent directional labels and exclusion audit."""
from common import *
guard();verify_source_policy()
from feature_source import Lake,Engine,RAW,C
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from collections import Counter

lake=Lake();engine=Engine();base=lake.load_candles(C.BASE_ASSET_LONG,'15m')
f=engine.extract_features(base).sort_index()
names=list(Engine.CANONICAL_B_NORMALIZED_72_FEATURES)
full_names=names+list(Engine.TREND_CONTINUATION_FEATURES)+list(Engine.MEAN_REVERSION_FEATURES)
calendar=mcal.get_calendar('NYSE').schedule(start_date=str(base.index.min().date()),end_date='2026-10-16')
sessions={str(day.date()):(r.market_open.tz_convert('America/New_York').tz_localize(None),r.market_close.tz_convert('America/New_York').tz_localize(None)) for day,r in calendar.iterrows()}
f['minutes_to_eod']=[((sessions[str(t.date())][1]-pd.Timedelta(minutes=5))-(t+pd.Timedelta(minutes=15))).total_seconds()/60 if str(t.date()) in sessions else 0. for t in f.index]
full_names+=['minutes_to_eod']
native=f[names].replace([np.inf,-np.inf],np.nan).to_numpy(dtype=np.float32)
five={};arrays={}
for s in [C.BASE_ASSET_SHORT,C.BASE_ASSET_LONG]:
    d=lake.load_candles(s,'5m')
    tr=pd.concat([d.High-d.Low,(d.High-d.Close.shift()).abs(),(d.Low-d.Close.shift()).abs()],axis=1).max(axis=1)
    d['atr']=tr.rolling(14).mean();five[s]=d
    arrays[s]=(d.index.asi8,d[['Open','High','Low','Close','Volume','atr']].to_numpy(dtype=float))

def outcome(symbol,t,eod):
    ix,a=arrays[symbol];loc=int(np.searchsorted(ix,t.value))
    if loc>=len(ix) or ix[loc]!=t.value:return None,'MISSING_ENTRY_5M'
    if loc<14 or not np.isfinite(a[loc-1,5]):return None,'MISSING_PAST_ATR'
    entry=float(a[loc,0]);atr=float(a[loc-1,5])
    if not np.isfinite(entry) or entry<=0 or atr<=0:return None,'INVALID_ENTRY_OR_ATR'
    sl=np.clip(atr/entry*C.SL_ATR_MULTIPLIER,C.SL_MIN_PCT,C.SL_MAX_PCT)
    stop=entry*(1-sl);tp=entry*(1+C.MAX_TP_PCT);peak=entry
    deadline=min(t+pd.Timedelta(minutes=C.TIME_STOP_MINUTES),eod);at=t
    while at<deadline:
        j=int(np.searchsorted(ix,at.value))
        if j>=len(ix) or ix[j]!=at.value:return None,'MISSING_PATH_BEFORE_EXIT'
        o,h,l,c,v,_=a[j]
        if not np.isfinite([o,h,l,c]).all() or min(o,h,l,c)<=0:return None,'INVALID_OHLC'
        if v<=0:return None,'ZERO_VOLUME_UNVERIFIED_FILL'
        if o<=stop:price=o;end=at;break
        if l<=stop:price=stop;end=at+pd.Timedelta(minutes=5);break
        if h>=tp:price=tp;end=at+pd.Timedelta(minutes=5);break
        peak=max(peak,h)
        if peak>=entry*(1+C.TRAILING_TRIGGER_PCT):stop=max(entry*(1-sl),entry*1.002,peak*(1-C.TRAILING_DROP_PCT))
        at+=pd.Timedelta(minutes=5)
    else:
        j=int(np.searchsorted(ix,deadline.value))
        if j>=len(ix) or ix[j]!=deadline.value:return None,'MISSING_DEADLINE_PRICE'
        if a[j,4]<=0:return None,'ZERO_VOLUME_UNVERIFIED_FILL'
        price=a[j,0];end=deadline
    return (float(price/entry-1-max(.002,C.BACKTEST_FEE_SLIPPAGE)),end.value),None

rows=[];seqs=[];fulls=[];targets=[];endpoints=[];seq_valid=[];audit=[]
available=f.index+pd.Timedelta(minutes=15)
for day,(op,close) in sessions.items():
    cutoff=min(pd.Timestamp(day+' '+C.PHASE_MAIN_END),close-pd.Timedelta(minutes=30))
    for t in pd.date_range(op,cutoff-pd.Timedelta(minutes=15),freq='15min'):
        r=dict(decision_time=str(t),status='ELIGIBLE',SOXS='UNAVAILABLE',SOXL='UNAVAILABLE')
        pos=int(available.searchsorted(t,side='right')-1)
        if pos<0:r['status']='NO_PAST15M';audit.append(r);continue
        # Market opening may legitimately use previous completed session/pre-market data.
        if t>op and available[pos]!=t:r['status']='MISSING_LATEST15M';audit.append(r);continue
        if not np.isfinite(native[pos]).all() or not np.isfinite(f.iloc[pos][full_names].to_numpy(dtype=float)).all():
            r['status']='FEATURE_WARMUP_OR_MISSING';audit.append(r);continue
        values=[];ends=[]
        for symbol in [C.BASE_ASSET_SHORT,C.BASE_ASSET_LONG]:
            result,error=outcome(symbol,t,close-pd.Timedelta(minutes=5))
            r[symbol]=error or 'LABEL_CONFIRMED'
            values.append(result[0] if result else np.nan);ends.append(result[1] if result else -1)
        # Even unlabeled decisions are kept; eligibility masks are applied per target/model at fit time.
        valid=pos>=31 and bool(np.isfinite(native[pos-31:pos+1]).all())
        seq=np.zeros((32,72),np.float32)
        if valid:seq[:]=native[pos-31:pos+1]
        else:seq[-1]=native[pos]
        seqs.append(seq);seq_valid.append(valid);targets.append(values);endpoints.append(ends)
        rows.append(dict(decision_time=t,feature_available_at=available[pos],bar_start=f.index[pos],
                         feature_row=pos,week=int((t.normalize()-pd.Timestamp('2020-07-27')).days//7)))
        fulls.append(f.iloc[pos][full_names].to_numpy(dtype=float));audit.append(r)
D=pd.DataFrame(rows);X=pd.DataFrame(fulls,columns=full_names);Y=np.asarray(targets);END=np.asarray(endpoints,dtype=np.int64)
cut=pd.Timestamp(POLICY['training_cutoff_ny']).value
assert (END[np.isfinite(Y)]<=cut).all() and (D.feature_available_at<=D.decision_time).all()
np.savez_compressed(HOME/'inputs/dataset.npz',seq=np.stack(seqs),seq_valid=np.array(seq_valid),y=Y,
                    times=D.decision_time.to_numpy().astype('datetime64[ns]').astype('int64'),end=END,
                    week=D.week.to_numpy(),exits=END)
X.to_pickle(HOME/'inputs/full_features.pkl');D.to_pickle(HOME/'inputs/labels.pkl');f.to_pickle(HOME/'inputs/native_features.pkl')
pd.DataFrame(audit).to_csv(HOME/'inputs/decision_coverage.csv',index=False)
save('inputs/schema.json',dict(net_features=names,full_features=full_names,sequence_order='oldest_to_newest',outcome_side_order=['SOXS','SOXL']))
# Raw row accounting is separate from calendar decision/label accounting.
raw_status=Counter()
for t in f.index:
    av=t+pd.Timedelta(minutes=15);day=str(av.date())
    if day not in sessions or not sessions[day][0]<=av<min(pd.Timestamp(day+' '+C.PHASE_MAIN_END),sessions[day][1]-pd.Timedelta(minutes=30)):
        raw_status['OUTSIDE_ENTRY_WINDOW_CONTEXT_RETAINED']+=1
    elif not np.isfinite(f.loc[t,names].to_numpy(dtype=float)).all():raw_status['FEATURE_WARMUP_OR_MISSING']+=1
    else:raw_status['ENTRY_WINDOW_NATIVE_ROW']+=1
assert sum(raw_status.values())==len(base)==len(f)
summary=dict(stage=3,status='PASS_DATA_ACCOUNTED_NOT_FEED_READY',raw_native15m_rows=len(base),feature_rows=len(f),raw_row_accounting=dict(raw_status),
             calendar_decisions=len(audit),retained_feature_decisions=len(D),sequence_valid_decisions=int(np.sum(seq_valid)),
             SOXS_labels=int(np.isfinite(Y[:,0]).sum()),SOXL_labels=int(np.isfinite(Y[:,1]).sum()),
             both_labels=int(np.isfinite(Y).all(axis=1).sum()),at_least_one_label=int(np.isfinite(Y).any(axis=1).sum()),
             decision_status_counts=dict(Counter(a['status'] for a in audit)),
             label_status_counts={s:dict(Counter(a[s] for a in audit)) for s in ['SOXS','SOXL']},
             all_native_rows_used_for_features=True,paired_intersection_not_required=True,full90m_path_not_required=True,
             original_active_training_already_uses_full_native15m=True,protected_changes=protected_changes())
assert not summary['protected_changes'];save('qa/stage3.json',summary)
save('inputs/source_hashes.json',{'market_data.db':sha(PROJECT/'data/market_data.db'),'ml_engine.py':sha(PROJECT/'core/ml_engine.py'),'config.py':sha(PROJECT/'config.py')})
print('FULL_DB_COVERAGE',json.dumps(summary),flush=True)
