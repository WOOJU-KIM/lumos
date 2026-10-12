"""
[Lumos] TP/SL 전략 3종 비교 WFA 백테스트
전략별 Triple Barrier 라벨을 다르게 학습한 뒤 동일 OOS 구간 비교:
  A: Fixed +3.0% TP / -2.0% SL
  B: Fixed +3.5% TP / -2.0% SL
  C: ATR dynamic (TP=1.5xATR 2~5% cap, SL=1.0xATR 1~3% cap)
주간 롤링 WFA: 104주 학습 → 1주 OOS, GBDT>=62%, 14:30 컷오프
"""
import os, sys, json, time
from pathlib import Path
import numpy as np
import config
import pandas as pd
from lightgbm import LGBMClassifier

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.data_lake import MarketDataLake
from core.ml_engine import MLFeatureEngine

CONF = 0.62
INIT_CAP = 10_000_000.0
ROLL_WKS = 104
SLIP = 0.0020
HORIZON = 6
CUT_ENTRY = "14:30"
CUT_EOD   = "15:45"

def mk_ca(df, name, periods=(1,3,5,20)):
    d = df.copy()
    d["dt2"] = pd.to_datetime(d["datetime"]) + pd.Timedelta(minutes=15)
    for p in periods:
        d[f"{name}_ret_{p}"] = d["Close"].pct_change(p)*100
    d[f"{name}_dir_1"]    = np.sign(d["Close"].pct_change(1))
    d[f"{name}_momentum"] = d["Close"].pct_change(1) - d["Close"].pct_change(5)/5.0
    cols = ["dt2"] + [c for c in d.columns if c.startswith(name)]
    return d[cols].rename(columns={"dt2":"datetime_dt"})

def labels_fixed(df, tp, sl, h=HORIZON):
    n=len(df); L=np.zeros(n,int)
    H=df["High"].values; Lo=df["Low"].values; C=df["Close"].values
    for i in range(n):
        p0=C[i]
        if p0<=0: continue
        ei=min(i+h+1,n)
        if i+1>=ei: continue
        for j in range(i+1,ei):
            hr=(H[j]-p0)/p0; lr=(Lo[j]-p0)/p0
            if hr>=tp and lr>-sl: L[i]=1; break
            elif lr<=-tp and hr<sl: L[i]=-1; break
            elif (hr>=tp and lr<=-sl) or (lr<=-tp and hr>=sl): L[i]=0; break
    return pd.Series(L, index=df.index)

def labels_atr(df, h=HORIZON):
    n=len(df); L=np.zeros(n,int)
    H=df["High"].values; Lo=df["Low"].values; C=df["Close"].values
    A=df["ATR_14"].values if "ATR_14" in df.columns else np.full(n,np.nan)
    for i in range(n):
        p0=C[i]
        if p0<=0: continue
        av=A[i] if not np.isnan(A[i]) else p0*0.018
        ap=av/p0
        tp=max(0.02,min(0.05,1.5*ap)); sl=max(0.01,min(0.03,1.0*ap))
        ei=min(i+h+1,n)
        if i+1>=ei: continue
        for j in range(i+1,ei):
            hr=(H[j]-p0)/p0; lr=(Lo[j]-p0)/p0
            if hr>=tp and lr>-sl: L[i]=1; break
            elif lr<=-tp and hr<sl: L[i]=-1; break
            elif (hr>=tp and lr<=-sl) or (lr<=-tp and hr>=sl): L[i]=0; break
    return pd.Series(L, index=df.index)

def run_wfa(df, fcols, weeks, short_d, name, tp=0.030, sl=0.020, atr=False):
    print(f"\n{'='*90}\n[{name}] WFA 시작\n{'='*90}")
    t0=time.time(); cap=INIT_CAP; peak=INIT_CAP; mdd=0.0
    trd=[]; tid=0; n_wks=len(weeks)-ROLL_WKS

    for wi in range(ROLL_WKS, len(weeks)):
        tw=weeks[wi]; trw=weeks[wi-ROLL_WKS:wi]
        mask=df["week_id"].isin(trw)
        dtr=df[mask].copy()
        lbl = labels_atr(dtr) if atr else labels_fixed(dtr,tp,sl)
        dtr["target"]=lbl.values
        y=dtr["target"].map({1:2,-1:0,0:1}).fillna(1).astype(int)
        X=dtr[fcols].fillna(0.0)
        clf=LGBMClassifier(objective="multiclass",num_class=3,class_weight="balanced",
                           n_estimators=85,max_depth=4,learning_rate=0.03,
                           random_state=42,verbosity=-1,n_jobs=-1)
        clf.fit(X,y)

        dts=df[df["week_id"]==tw].copy()
        if dts.empty: continue
        pr=clf.predict_proba(dts[fcols].fillna(0.0))
        ps,pn,pl=pr[:,0],pr[:,1],pr[:,2]
        cf=np.full(len(dts),0.50); dr=["NONE"]*len(dts)
        for i in range(len(dts)):
            if pl[i]>pn[i] and pl[i]>ps[i]:
                cf[i]=min(0.95,max(0.50,0.50+(pl[i]-0.333)*1.15)); dr[i]="LONG"
            elif ps[i]>pn[i] and ps[i]>pl[i]:
                cf[i]=min(0.95,max(0.50,0.50+(ps[i]-0.333)*1.15)); dr[i]="SHORT"
        dts=dts.copy(); dts["Conf"]=cf; dts["Dir"]=dr
        ap=None
        for ds in sorted(dts["date_str"].unique()):
            db=dts[dts["date_str"]==ds].reset_index(drop=True); slc=0
            for _,row in db.iterrows():
                cdt=row["datetime"]; ts=row["time_str"]
                sc=float(row["Close"]); sh=float(row["High"]); sl2=float(row["Low"])
                av=float(row.get("ATR_14",sc*0.018))
                sr=short_d.get(cdt)
                xc=float(sr["Close"]) if sr else 0.0
                xh=float(sr["High"])  if sr else 0.0
                xl=float(sr["Low"])   if sr else 0.0
                if ap is not None:
                    ap["bars"]+=1
                    sym=ap["sym"]; ep=ap["entry_px"]
                    cc=sc if sym==config.BASE_ASSET_LONG else xc
                    ch=sh if sym==config.BASE_ASSET_LONG else xh
                    cl=sl2 if sym==config.BASE_ASSET_LONG else xl
                    if atr:
                        ea=ap.get("eAtr",ep*0.018); ap2=ea/ep
                        tpu=max(0.02,min(0.05,1.5*ap2)); slu=max(0.01,min(0.03,1.0*ap2))
                    else:
                        tpu=tp; slu=sl
                    tpx=ep*(1+tpu); slx=ep*(1-slu)
                    ex=None; er=None
                    if ch>=tpx: ex=tpx; er="TP"
                    elif cl<=slx: ex=slx; er="SL"
                    elif ap["bars"]>=HORIZON: ex=cc; er="TIMESTOP"
                    elif ts>=CUT_EOD: ex=cc; er="EOD"
                    if ex is not None:
                        r=(ex/ep)-1; nr=r-SLIP; pk=cap*nr; cap+=pk
                        if cap>peak: peak=cap
                        dd=(peak-cap)/peak*100
                        if dd>mdd: mdd=dd
                        if nr<=-0.015: slc+=1
                        tid+=1
                        trd.append({"tid":tid,"entry_dt":ap["entry_dt"],"exit_dt":cdt,
                                    "sym":sym,"epx":round(ep,4),"expx":round(ex,4),
                                    "bars":ap["bars"],"reason":er,
                                    "ret_pct":round(nr*100,3),"pnl":int(round(pk)),
                                    "ecap":int(round(cap)),"win":1 if nr>0 else 0,
                                    "conf":round(ap["conf"],3),
                                    "tpu":round(tpu*100,2),"slu":round(slu*100,2)})
                        ap=None; continue
                if ap is None and slc<3 and ts<CUT_ENTRY:
                    gd=row.get("Dir","NONE"); gc=float(row.get("Conf",0.50))
                    if gc>=CONF:
                        if gd=="LONG" and sc>0:
                            ap={"sym":config.BASE_ASSET_LONG,"entry_px":sc,"entry_dt":cdt,"bars":0,"conf":gc,"eAtr":av}
                        elif gd=="SHORT" and xc>0:
                            ap={"sym":config.BASE_ASSET_SHORT,"entry_px":xc,"entry_dt":cdt,"bars":0,"conf":gc,"eAtr":av}

        wn=wi-ROLL_WKS+1
        if wn%30==0 or wn==n_wks:
            print(f"  [{wn}/{n_wks}주] {tw} | 거래:{len(trd):,} | 잔고:{int(cap):,}원")

    el=round(time.time()-t0,1); tr=(cap/INIT_CAP-1)*100
    print(f"\n✅ [{name}] {el}s | 거래:{len(trd):,} | {tr:+.1f}% | MDD:{mdd:.1f}%")
    return {"name":name,"final":int(cap),"ret":round(tr,2),"mdd":round(mdd,2),"n":len(trd),"trd":trd}

def summarize(res):
    if not res["trd"]: return res
    df=pd.DataFrame(res["trd"])
    df["year"]=df["entry_dt"].str[:4]
    wins=df[df["ret_pct"]>0]; loss=df[df["ret_pct"]<=0]
    gp=wins["ret_pct"].sum(); gl=abs(loss["ret_pct"].sum())
    res["wr"]=round(len(wins)/len(df)*100,2)
    res["pf"]=round(gp/gl,3) if gl>0 else 99.0
    res["exit_dist"]=df["reason"].value_counts().to_dict()
    res["avg_tp"]=round(df["tpu"].mean(),2); res["avg_sl"]=round(df["slu"].mean(),2)
    yrs=[]
    ys=INIT_CAP
    for y,g in df.groupby("year"):
        ye=float(g["ecap"].iloc[-1])
        gw=g[g["ret_pct"]>0]["ret_pct"].sum(); gl2=abs(g[g["ret_pct"]<=0]["ret_pct"].sum())
        yrs.append({"year":y,"ret":round((ye/ys-1)*100,1),"n":len(g),
                    "wr":round(len(g[g["ret_pct"]>0])/len(g)*100,1),
                    "pf":round(gw/gl2,3) if gl2>0 else 99.0,
                    "L":int((g["sym"]==config.BASE_ASSET_LONG).sum()),"S":int((g["sym"]==config.BASE_ASSET_SHORT).sum())})
        ys=ye
    res["yearly"]=yrs
    return res

def san(o):
    if isinstance(o,(np.integer,np.int64,np.int32)): return int(o)
    if isinstance(o,(np.floating,np.float64,np.float32)): return float(o)
    if isinstance(o,dict): return {k:san(v) for k,v in o.items()}
    if isinstance(o,list): return [san(i) for i in o]
    return o

def main():
    lake=MarketDataLake(); ml=MLFeatureEngine(confidence_threshold=CONF)
    print("="*90+"\n📦 데이터 로드 중...\n"+"="*90)
    s15=lake.load_candles(config.BASE_ASSET_LONG,"15m").sort_values("datetime").reset_index(drop=True)
    x15=lake.load_candles(config.BASE_ASSET_SHORT,"15m").sort_values("datetime").reset_index(drop=True)
    o60=lake.load_candles(config.MACRO_TREND_SYMBOL,"60m").sort_values("datetime").reset_index(drop=True)
    n15=lake.load_candles("NVDA","15m").sort_values("datetime").reset_index(drop=True)
    q15=lake.load_candles("QQQ","15m").sort_values("datetime").reset_index(drop=True)
    v15=lake.load_candles("VIXY","15m").sort_values("datetime").reset_index(drop=True)
    i15=lake.load_candles("IEF","15m").sort_values("datetime").reset_index(drop=True)

    o60["ema20"]=o60["Close"].ewm(span=20,adjust=False).mean()
    o60["ema60"]=o60["Close"].ewm(span=60,adjust=False).mean()
    o60["mts"]=(o60["ema20"]-o60["ema60"])/(o60["ema60"]+1e-9)*100
    o60["mes"]=(o60["ema20"].diff(5)/o60["ema20"].shift(5))*100
    o60["mpv"]=(o60["Close"]-o60["ema20"])/(o60["ema20"]+1e-9)*100
    mc=["mts","mes","mpv"]
    sf=o60[["datetime"]+mc].copy()
    sf["datetime_dt"]=pd.to_datetime(sf["datetime"])+pd.Timedelta(minutes=60)

    print("⏳ 피처 추출...")
    df=ml.extract_features(s15)
    df["ATR_Pct"]=(df["ATR_14"]/(df["Close"]+1e-9))*100
    df["MACD_Pct"]=(df["MACD"]/(df["Close"]+1e-9))*100
    df["MACD_Signal_Pct"]=(df["MACD_Signal"]/(df["Close"]+1e-9))*100
    df["MACD_Hist_Pct"]=(df["MACD_Hist"]/(df["Close"]+1e-9))*100
    df["datetime_dt"]=pd.to_datetime(df["datetime"])+pd.Timedelta(minutes=15)

    dm=pd.merge_asof(df.sort_values("datetime_dt"),
                     sf.sort_values("datetime_dt")[["datetime_dt"]+mc],
                     on="datetime_dt",direction="backward")
    for nm,rd in [("nvda",n15),("qqq",q15),("vixy",v15),("ief",i15)]:
        dm=pd.merge_asof(dm.sort_values("datetime_dt"),
                         mk_ca(rd,nm).sort_values("datetime_dt"),
                         on="datetime_dt",direction="backward")

    dm["long_ret_20"]=dm["Close"].pct_change(20)*100
    dm["long_ret_5"]=dm["Close"].pct_change(5)*100
    dm["long_vs_qqq_20"]=dm["long_ret_20"]-dm.get("qqq_ret_20",pd.Series(0.0,index=dm.index)).fillna(0)
    dm["long_vs_qqq_5"]=dm["long_ret_5"]-dm.get("qqq_ret_5",pd.Series(0.0,index=dm.index)).fillna(0)
    dm["panic"]=(
        (dm.get("vixy_ret_1",pd.Series(0.0,index=dm.index)).fillna(0)>0).astype(int)+
        (dm.get("ief_ret_1",pd.Series(0.0,index=dm.index)).fillna(0)>0).astype(int)
    )
    dm["datetime_dt"]=pd.to_datetime(dm["datetime"])
    dm["date_str"]=dm["datetime_dt"].dt.strftime("%Y-%m-%d")
    dm["time_str"]=dm["datetime_dt"].dt.strftime("%H:%M")
    dm["week_id"]=(dm["datetime_dt"].dt.isocalendar().year.astype(str)+"-"+
                   dm["datetime_dt"].dt.isocalendar().week.astype(str).str.zfill(2))

    EX={"open","high","low","close","volume","Open","High","Low","Close","Volume",
        "datetime","datetime_dt","date_str","time_str","week_id","target","date",
        "cum_vp","vwap","EMA_9","EMA_21","EMA_50","EMA_200",
        "BB_Upper","BB_Lower","KC_Upper","KC_Lower","ATR_14","MACD","MACD_Signal","MACD_Hist","panic"}
    fc=[c for c in dm.columns if c not in EX and pd.api.types.is_numeric_dtype(dm[c])]
    print(f"📊 피처: {len(fc)}개")
    wks=sorted(dm["week_id"].unique())
    print(f"📅 주차: {len(wks)}주 | OOS: {len(wks)-ROLL_WKS}주 ({wks[ROLL_WKS]}~{wks[-1]})")
    sd=x15.set_index("datetime").to_dict(orient="index")

    strats=[
        dict(name="A: Fixed +3.0%/-2.0%",tp=0.030,sl=0.020,atr=False),
        dict(name="B: Fixed +3.5%/-2.0%",tp=0.035,sl=0.020,atr=False),
        dict(name="C: ATR Dynamic",       tp=0.030,sl=0.020,atr=True),
    ]
    results=[]
    for s in strats:
        r=run_wfa(dm,fc,wks,sd,s["name"],tp=s["tp"],sl=s["sl"],atr=s["atr"])
        r=summarize(r); results.append(r)

    print("\n\n"+"="*90)
    print("📊 [3-way TP/SL 비교 결과]")
    print("="*90)
    print(f"{'전략':<25}|{'수익률':>10}|{'MDD':>7}|{'거래':>6}|{'승률':>7}|{'PF':>6}|{'avgTP':>7}|{'avgSL':>7}")
    print("-"*90)
    for r in results:
        print(f"{r['name']:<25}|{r['ret']:>+9.1f}%|{r['mdd']:>6.1f}%|"
              f"{r['n']:>5,}|{r['wr']:>6.1f}%|{r['pf']:>6.3f}|"
              f"{r['avg_tp']:>6.2f}%|{r['avg_sl']:>6.2f}%")

    ays=sorted(set(y["year"] for r in results for y in r.get("yearly",[])))
    print(f"\n{'연도':<6}|"+" | ".join(f"{r['name'][:20]:<20}" for r in results))
    print("-"*80)
    for y in ays:
        row=f"{y:<6}|"
        for r in results:
            yd=next((x for x in r.get("yearly",[]) if x["year"]==y),None)
            row+=(f"{yd['ret']:>+7.1f}% W{yd['wr']:.0f}% PF{yd['pf']:.2f}" if yd else "N/A".ljust(22))+" | "
        print(row)

    print("\n청산 사유:")
    for r in results:
        print(f"  {r['name']}: "+" | ".join(f"{k}={v}" for k,v in sorted(r.get("exit_dist",{}).items())))

    op=PROJECT_ROOT/"data"/"backtest_tpsl_comparison_wfa.json"
    with open(op,"w",encoding="utf-8") as f:
        json.dump(san({"ts":pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                       "conf":CONF,"roll":ROLL_WKS,
                       "results":[{k:v for k,v in r.items() if k!="trd"} for r in results]}),
                  f,indent=2,ensure_ascii=False)
    for r in results:
        tag=r["name"].split(":")[0].strip()
        pd.DataFrame(r["trd"]).to_csv(PROJECT_ROOT/"data"/f"backtest_tpsl_{tag}_trades.csv",
                                      index=False,encoding="utf-8-sig")
    print(f"\n💾 {op}\n✅ 완료")

if __name__=="__main__":
    main()
