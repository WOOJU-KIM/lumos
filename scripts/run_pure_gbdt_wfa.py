"""
[Lumos] 순수 GBDT (크로스에셋 피처 내장) 단일 트리거 주간 롤링 WFA 백테스트
=====================================================================
사용자 요청 사항 완벽 반영:
1. 멀티스크린 (60분봉, 5분봉 눌림목) 필터 완전 폐지
2. CrossAsset 방향성 Veto 폐지 (이미 GBDT 피처로 내장되어 있음)
3. 주간 롤링 WFA: 과거 2년(104주) 학습 -> 미래 1주 블라인드 OOS 테스트
4. Confidence Sweep: 60%, 62%, 64%
5. 손익절: +3.0% / -2.0% (학습 라벨링 및 시뮬레이션 양쪽 모두 동일 적용)
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

ROLL_WKS  = 104
INIT_CAP  = 10_000_000.0
TP_PCT    = 0.030
SL_PCT    = 0.020
SLIP      = 0.03        # 달러 슬리피지
FEE_RATE  = 0.0020
TS_BARS   = 18          # 5분봉 90분
CUT_ENTRY = "14:30"
CUT_EOD   = "15:45"
CONFS     = [0.60, 0.62, 0.64]

def run_conf_wfa(df_feat, long_5m_d, short_5m_d, unique_weeks, fcols, conf_thr, label):
    """단일 conf 임계값으로 WFA 시뮬레이션 (GBDT 단일 트리거)"""
    print(f"\n{'='*90}\n[{label}] conf>={conf_thr*100:.0f}% WFA 시작\n{'='*90}")
    t0   = time.time()
    cap  = INIT_CAP; peak = INIT_CAP; mdd = 0.0
    trd  = []; tid = 0; n_wks = len(unique_weeks) - ROLL_WKS

    for wi in range(ROLL_WKS, len(unique_weeks)):
        tw  = unique_weeks[wi]
        trw = unique_weeks[wi-ROLL_WKS:wi]

        # ── 1. 과거 2년 데이터로 학습 ──
        mask  = df_feat["week_id"].isin(trw)
        dtr   = df_feat[mask].copy()
        n = len(dtr)
        lbl = np.zeros(n, int)
        H  = dtr["High"].values; Lo = dtr["Low"].values; C = dtr["Close"].values
        # TP 3.0%, SL -2.0% 라벨링
        for i in range(n):
            p0 = C[i]
            if p0 <= 0: continue
            ei = min(i+7, n)
            for j in range(i+1, ei):
                hr = (H[j]-p0)/p0; lr = (Lo[j]-p0)/p0
                if hr >= TP_PCT and lr > -SL_PCT: lbl[i]=1; break
                elif lr <= -TP_PCT and hr < SL_PCT: lbl[i]=-1; break
                elif (hr>=TP_PCT and lr<=-SL_PCT) or (lr<=-TP_PCT and hr>=SL_PCT): break

        y = pd.Series(lbl).map({1:2,-1:0,0:1}).fillna(1).astype(int)
        X = dtr[fcols].fillna(0.0)
        
        clf = LGBMClassifier(objective="multiclass", num_class=3, class_weight="balanced",
                             n_estimators=85, max_depth=4, learning_rate=0.03,
                             random_state=42, verbosity=-1, n_jobs=-1)
        clf.fit(X, y)

        # ── 2. OOS 1주 블라인드 테스트 ──
        dts_wk = df_feat[df_feat["week_id"] == tw].copy()
        if dts_wk.empty: continue
        
        pr = clf.predict_proba(dts_wk[fcols].fillna(0.0))
        p_s, p_n, p_l = pr[:,0], pr[:,1], pr[:,2]
        
        gbdt_dirs = []; gbdt_confs = []
        for i in range(len(dts_wk)):
            ps, pn, pl = p_s[i], p_n[i], p_l[i]
            if pl > pn and pl > ps:
                gbdt_dirs.append("LONG")
                gbdt_confs.append(min(0.95, max(0.50, 0.50+(pl-0.333)*1.15)))
            elif ps > pn and ps > pl:
                gbdt_dirs.append("SHORT")
                gbdt_confs.append(min(0.95, max(0.50, 0.50+(ps-0.333)*1.15)))
            else:
                gbdt_dirs.append("NONE"); gbdt_confs.append(0.50)
                
        dts_wk["gbdt_dir"]  = gbdt_dirs
        dts_wk["gbdt_conf"] = gbdt_confs

        # 일별 시뮬레이션
        for d_str in sorted(dts_wk["date_str"].unique()):
            day15 = dts_wk[dts_wk["date_str"] == d_str]
            slc   = 0
            b_idx = 0; rows = list(day15.iterrows()); n_bars = len(rows)

            while b_idx < n_bars:
                dt_idx, row = rows[b_idx]
                cur_dt  = row["datetime"]
                ts      = row["time_str"]

                if b_idx < 1 or ts > CUT_ENTRY:
                    b_idx += 1; continue
                if slc >= 3:
                    b_idx += 1; continue

                gbdt_dir  = row["gbdt_dir"]
                gbdt_conf = float(row["gbdt_conf"])

                # 오직 GBDT 방향과 신뢰도로만 진입 결정 (필터 일체 없음)
                if gbdt_dir not in ("LONG","SHORT") or gbdt_conf < conf_thr:
                    b_idx += 1; continue

                sym = config.BASE_ASSET_LONG if gbdt_dir=="LONG" else config.BASE_ASSET_SHORT
                
                # SHORT_SQQQ인 경우 SQQQ 가격 확인 로직을 위해 원래는 SQQQ 데이터를 조회해야 하나
                # 여기서는 TQQQ 데이터셋을 루프 돌고 있으므로, 진입가를 찾기 위해 short_5m_d 사용
                if sym == config.BASE_ASSET_LONG:
                    base_px = float(row["Close"])
                else:
                    d5_short = short_5m_d.get(d_str, pd.DataFrame())
                    short_at_time = d5_short[d5_short["datetime"] <= cur_dt]
                    if short_at_time.empty:
                        b_idx += 1; continue
                    base_px = float(short_at_time.iloc[-1]["Close"])

                entry_px = round(base_px + SLIP, 2)
                shares   = int(cap / entry_px)
                invested = shares * entry_px
                if shares <= 0 or invested <= 0:
                    b_idx += 1; continue

                # 5분봉 Path Dissection
                day5 = long_5m_d.get(d_str, pd.DataFrame()) if sym==config.BASE_ASSET_LONG else short_5m_d.get(d_str, pd.DataFrame())
                if day5.empty:
                    b_idx += 1; continue
                post5 = day5[day5["datetime"] > cur_dt]
                if post5.empty:
                    b_idx += 1; continue

                tp_px  = round(entry_px*(1+TP_PCT), 2)
                sl_px  = round(entry_px*(1-SL_PCT), 2)
                eval5  = post5.iloc[:TS_BARS]
                ex_px  = None; ex_reason = "TIMESTOP"; ex_ts = ""

                for _, c5 in eval5.iterrows():
                    t5  = c5["time_str"]
                    c5h = float(c5["High"]); c5l = float(c5["Low"]); c5c = float(c5["Close"]); c5o = float(c5["Open"])
                    htp = c5h >= tp_px; hsl = c5l <= sl_px
                    
                    if htp and hsl:
                        ex_px = round((tp_px if c5c>=c5o else sl_px) - SLIP, 2)
                        ex_reason = "TP" if c5c>=c5o else "SL"; ex_ts=t5; break
                    elif htp:
                        ex_px = round(tp_px - SLIP, 2); ex_reason = "TP"; ex_ts=t5; break
                    elif hsl:
                        ex_px = round(sl_px - SLIP, 2); ex_reason = "SL"; ex_ts=t5; break
                    elif t5 >= CUT_EOD:
                        ex_px = round(c5c - SLIP, 2); ex_reason = "EOD"; ex_ts=t5; break
                        
                if ex_px is None:
                    last = eval5.iloc[-1]
                    ex_px = round(float(last["Close"]) - SLIP, 2)
                    ex_ts = last["time_str"]

                raw_ret  = (ex_px / entry_px) - 1.0
                fee      = FEE_RATE
                net_ret  = raw_ret - fee
                pnl      = cap * net_ret
                cap     += pnl
                if cap > peak: peak = cap
                dd = (peak-cap)/peak*100
                if dd > mdd: mdd = dd
                if net_ret <= -0.015: slc += 1
                tid += 1
                trd.append({"tid":tid,"dt":cur_dt,"sym":sym,"ep":round(entry_px,4),
                             "xp":round(ex_px,4),"reason":ex_reason,"ret":round(net_ret*100,3),
                             "pnl":int(round(pnl)),"cap":int(round(cap)),
                             "win":1 if net_ret>0 else 0,"conf":round(gbdt_conf,3)})
                b_idx += 1

        wn = wi-ROLL_WKS+1
        if wn % 30 == 0 or wn == n_wks:
            print(f"   [{wn}/{n_wks}주] {tw} | 거래:{len(trd):,} | 잔고:{int(cap):,}원")

    el = round(time.time()-t0,1)
    tr = (cap/INIT_CAP-1)*100
    print(f"\n✅ [{label}] {el}s | 거래:{len(trd):,} | {tr:+.1f}% | MDD:{mdd:.1f}%")
    return {"label":label,"conf":conf_thr,"final":int(cap),"ret":round(tr,2),
            "mdd":round(mdd,2),"n":len(trd),"trd":trd}

def summarize(res):
    if not res["trd"]: return res
    df = pd.DataFrame(res["trd"])
    df["year"] = df["dt"].str[:4]
    wins = df[df["ret"]>0]; loss = df[df["ret"]<=0]
    gp = wins["ret"].sum(); gl = abs(loss["ret"].sum())
    res["wr"] = round(len(wins)/len(df)*100,2)
    res["pf"] = round(gp/gl,3) if gl>0 else 99.0
    res["exit_dist"] = df["reason"].value_counts().to_dict()
    yrs=[]; ys=INIT_CAP
    for y,g in df.groupby("year"):
        ye=float(g["cap"].iloc[-1])
        gw=g[g["ret"]>0]["ret"].sum(); gl2=abs(g[g["ret"]<=0]["ret"].sum())
        yrs.append({"year":y,"ret":round((ye/ys-1)*100,1),"n":len(g),
                    "wr":round(len(g[g["ret"]>0])/len(g)*100,1),
                    "pf":round(gw/gl2,3) if gl2>0 else 99.0})
        ys=ye
    res["yearly"]=yrs
    return res

def san(o):
    if isinstance(o,(np.integer,)): return int(o)
    if isinstance(o,(np.floating,)): return float(o)
    if isinstance(o,dict): return {k:san(v) for k,v in o.items()}
    if isinstance(o,list): return [san(i) for i in o]
    return o

def main():
    lake = MarketDataLake()
    ml   = MLFeatureEngine() # 여기에 CrossAsset 피처 포함됨

    print("="*90+"\n📦 데이터 로드 및 피처 추출 (순수 GBDT)\n"+"="*90)
    long_15m = lake.load_candles(config.BASE_ASSET_LONG,"15m").sort_values("datetime").reset_index(drop=True)
    long_5m  = lake.load_candles(config.BASE_ASSET_LONG,"5m").sort_values("datetime").reset_index(drop=True)
    short_5m  = lake.load_candles(config.BASE_ASSET_SHORT,"5m").sort_values("datetime").reset_index(drop=True)

    for df in [long_15m, long_5m, short_5m]:
        df["datetime_dt"] = pd.to_datetime(df["datetime"])
        df["date_str"]    = df["datetime_dt"].dt.strftime("%Y-%m-%d")
        df["time_str"]    = df["datetime_dt"].dt.strftime("%H:%M")

    print("⏳ MLFeatureEngine 피처 추출 중 (CrossAsset 내부 계산 포함)...")
    df_feat = ml.extract_features(long_15m)
    df_feat["datetime_dt"] = pd.to_datetime(df_feat["datetime"])
    df_feat["date_str"] = df_feat["datetime_dt"].dt.strftime("%Y-%m-%d")
    df_feat["time_str"] = df_feat["datetime_dt"].dt.strftime("%H:%M")
    df_feat["week_id"]  = (df_feat["datetime_dt"].dt.isocalendar().year.astype(str)+"-"+
                           df_feat["datetime_dt"].dt.isocalendar().week.astype(str).str.zfill(2))

    EX = {"open","high","low","close","volume","Open","High","Low","Close","Volume",
          "datetime","datetime_dt","date_str","time_str","week_id","date","cross_dir","cross_conf"}
    fcols = [c for c in df_feat.columns if c not in EX and pd.api.types.is_numeric_dtype(df_feat[c])]
    print(f"📊 사용된 피처 수: {len(fcols)}개 (CrossAsset 반환값 등 모두 GBDT가 학습)")

    long_5m_d = {d:g for d,g in long_5m.groupby("date_str")}
    short_5m_d = {d:g for d,g in short_5m.groupby("date_str")}

    wks = sorted(df_feat["week_id"].unique())
    n_oos = len(wks) - ROLL_WKS
    print(f"📅 총 {len(wks)}주 | OOS {n_oos}주 ({wks[ROLL_WKS]} ~ {wks[-1]})")

    all_res = []
    for conf in CONFS:
        lbl = f"Conf≥{int(conf*100)}%"
        res = run_conf_wfa(df_feat, long_5m_d, short_5m_d, wks, fcols, conf, lbl)
        res = summarize(res); all_res.append(res)

    print("\n\n"+"="*90)
    print("📊 [순수 GBDT WFA 결과] (Veto/스크린 필터 없음, 과거 2년 슬라이딩 학습)")
    print("="*90)
    print(f"{'전략':<15}|{'수익률':>10}|{'MDD':>7}|{'거래':>6}|{'승률':>7}|{'PF':>6}")
    print("-"*60)
    for r in all_res:
        print(f"{r['label']:<15}|{r['ret']:>+9.1f}%|{r['mdd']:>6.1f}%|{r['n']:>5,}|{r['wr']:>6.1f}%|{r['pf']:>6.3f}")

    ays = sorted(set(y["year"] for r in all_res for y in r.get("yearly",[])))
    print(f"\n{'연도':<6}|"+" | ".join(f"{r['label']:<20}" for r in all_res))
    print("-"*80)
    for y in ays:
        row = f"{y:<6}|"
        for r in all_res:
            yd = next((x for x in r.get("yearly",[]) if x["year"]==y), None)
            row += (f"{yd['ret']:>+7.1f}% W{yd['wr']:.0f}% PF{yd['pf']:.2f}" if yd else "N/A".ljust(25))+" | "
        print(row)

    print("\n청산 사유:")
    for r in all_res:
        print(f"  {r['label']}: "+" | ".join(f"{k}={v}" for k,v in sorted(r.get("exit_dist",{}).items())))

    op = PROJECT_ROOT/"data"/"backtest_pure_gbdt_wfa.json"
    with open(op,"w",encoding="utf-8") as f:
        json.dump(san({"ts":pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                       "results":[{k:v for k,v in r.items() if k!="trd"} for r in all_res]}),
                  f,indent=2,ensure_ascii=False)
    print(f"\n💾 {op}\n✅ 완료")

if __name__=="__main__":
    main()
