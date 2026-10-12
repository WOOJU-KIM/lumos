"""
[Lumos] 학습 라벨링 (Fixed vs ATR) x 실전 매매 (ATR Trailing) 비교 DoE
=====================================================================
어제 57%를 기록한 완벽한 셋업(15분봉 학습 + 5분봉 ATR 트레일링 청산 + Veto 없음)을 기반으로,
학습할 때의 정답지(Labeling)를 어떻게 주는 것이 더 좋은지 비교하는 DoE입니다.

- Scenario A (어제 방식): 학습은 Fixed(+3.0%/-2.0%)로 하고, 실전 매매는 ATR 트레일링 적용
- Scenario B (일치 방식): 학습도 ATR(+3.5%/동적-SL)로 하고, 실전 매매도 ATR 트레일링 적용
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
SLIP      = 0.03
FEE_RATE  = 0.0020
TS_BARS   = 6           # 15분봉 6개 (학습 라벨링용 90분)
CUT_ENTRY = "14:30"
CONFS     = [0.60, 0.62, 0.64]

def run_wfa_scenario(df_feat, long_5m_d, short_5m_d, unique_weeks, fcols, conf_thr, label, mode):
    print(f"\n{'='*90}\n[{label}] conf>={conf_thr*100:.0f}% 시작 (라벨링: {mode})\n{'='*90}")
    t0   = time.time()
    cap  = INIT_CAP; peak = INIT_CAP; mdd = 0.0
    trd  = []; tid = 0; n_wks = len(unique_weeks) - ROLL_WKS
    
    for wi in range(ROLL_WKS, len(unique_weeks)):
        tw  = unique_weeks[wi]
        trw = unique_weeks[wi-ROLL_WKS:wi]

        # ── 1. 과거 2년 학습 ──
        mask  = df_feat["week_id"].isin(trw)
        dtr   = df_feat[mask].copy()
        n = len(dtr)
        lbl = np.zeros(n, int)
        H = dtr["High"].values; Lo = dtr["Low"].values; C = dtr["Close"].values
        atr_pcts = dtr["ATR_Pct"].fillna(1.8).values
        
        for i in range(n):
            p0 = C[i]
            if p0 <= 0: continue
            ei = min(i + TS_BARS + 1, n)
            
            # 라벨링 모드에 따른 TP/SL 설정
            if mode == "FIXED":
                tp = 0.030
                sl = 0.020
            else:
                tp = 0.035
                sl = max(0.020, min(0.032, (atr_pcts[i] / 100.0) * 1.5))
                
            for j in range(i+1, ei):
                hr = (H[j]-p0)/p0; lr = (Lo[j]-p0)/p0
                hit_l_tp = (hr >= tp); hit_l_sl = (lr <= -sl)
                hit_s_tp = (lr <= -tp); hit_s_sl = (hr >= sl)
                
                if hit_l_tp and not hit_l_sl:
                    lbl[i] = 1; break
                elif hit_s_tp and not hit_s_sl:
                    lbl[i] = -1; break
                elif (hit_l_sl and hit_s_sl) or (hit_l_tp and hit_l_sl) or (hit_s_tp and hit_s_sl):
                    lbl[i] = 0; break

        y = pd.Series(lbl).map({1:2,-1:0,0:1}).fillna(1).astype(int)
        X = dtr[fcols].fillna(0.0)
        
        clf = LGBMClassifier(objective="multiclass", num_class=3, class_weight="balanced",
                             n_estimators=85, max_depth=4, learning_rate=0.03,
                             random_state=42, verbosity=-1, n_jobs=-1)
        clf.fit(X, y)

        # ── 2. OOS 테스트 ──
        dts_wk = df_feat[df_feat["week_id"] == tw].copy()
        if dts_wk.empty: continue
        
        pr = clf.predict_proba(dts_wk[fcols].fillna(0.0))
        p_s, p_n, p_l = pr[:,0], pr[:,1], pr[:,2]
        
        gbdt_dirs = []; gbdt_confs = []
        for i in range(len(dts_wk)):
            ps, pn, pl = p_s[i], p_n[i], p_l[i]
            # 어제 모델의 핵심: 확률 보정 (Stretching) 로직 재현
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

                if gbdt_dir not in ("LONG","SHORT") or gbdt_conf < conf_thr:
                    b_idx += 1; continue

                sym = config.BASE_ASSET_LONG if gbdt_dir=="LONG" else config.BASE_ASSET_SHORT
                
                if sym == config.BASE_ASSET_LONG:
                    entry_px = float(row["Close"])
                else:
                    d5_short = short_5m_d.get(d_str, pd.DataFrame())
                    short_at_time = d5_short[d5_short["datetime"] <= cur_dt]
                    if short_at_time.empty:
                        b_idx += 1; continue
                    entry_px = float(short_at_time.iloc[-1]["Close"])

                entry_px = round(entry_px + SLIP, 2)
                shares   = int(cap / entry_px)
                if shares <= 0:
                    b_idx += 1; continue

                # ── 3. 실전 5분봉 ATR Trailing 매매 ──
                day5 = long_5m_d.get(d_str, pd.DataFrame()) if sym==config.BASE_ASSET_LONG else short_5m_d.get(d_str, pd.DataFrame())
                if day5.empty:
                    b_idx += 1; continue
                post5 = day5[day5["datetime"] > cur_dt]
                if post5.empty:
                    b_idx += 1; continue

                atr_pct = float(row.get("ATR_Pct", 1.8))
                sl_pct  = max(0.020, min(0.032, (atr_pct / 100.0) * 1.5))
                
                tp_max_px = entry_px * 1.035
                trail_trig_px = entry_px * 1.020
                curr_sl_px = entry_px * (1.0 - sl_pct)
                is_trail = False
                peak_h = entry_px
                
                ex_px = None; ex_reason = "TIMESTOP"; ex_ts = ""; holding_5m_bars = 0

                # 최대 18봉(90분) 평가
                eval5 = post5.iloc[:18]
                for k, c5 in eval5.iterrows():
                    t5 = c5["time_str"]
                    bh = float(c5["High"]); bl = float(c5["Low"]); bc = float(c5["Close"])
                    
                    if bh > peak_h: peak_h = bh
                    
                    # 트레일링 스탑 발동
                    if not is_trail and peak_h >= trail_trig_px:
                        is_trail = True
                        curr_sl_px = entry_px * 1.002 # 본절+약수익 보존
                        
                    if is_trail:
                        t = peak_h * 0.992 # 고점 대비 0.8% 하락시 청산
                        if t > curr_sl_px: curr_sl_px = t
                        
                    if bh >= tp_max_px:
                        ex_px = tp_max_px; ex_reason = "TP_MAX"; ex_ts = t5; holding_5m_bars = (k - eval5.index[0] + 1); break
                    if bl <= curr_sl_px:
                        ex_px = curr_sl_px; ex_reason = "TRAIL_SL" if is_trail else "ATR_SL"; ex_ts = t5; holding_5m_bars = (k - eval5.index[0] + 1); break
                    if t5 >= "15:45":
                        ex_px = bc; ex_reason = "EOD"; ex_ts = t5; holding_5m_bars = (k - eval5.index[0] + 1); break
                        
                if ex_px is None:
                    last = eval5.iloc[-1]
                    ex_px = float(last["Close"]); ex_ts = last["time_str"]
                    holding_5m_bars = len(eval5)

                ex_px = round(ex_px - SLIP, 2)
                raw_ret  = (ex_px / entry_px) - 1.0
                net_ret  = raw_ret - FEE_RATE
                pnl      = cap * net_ret
                cap     += pnl
                
                if cap > peak: peak = cap
                dd = (peak-cap)/peak*100
                if dd > mdd: mdd = dd
                if net_ret <= -0.015: slc += 1
                tid += 1
                trd.append({"tid":tid,"dt":cur_dt,"sym":sym,
                             "reason":ex_reason,"ret":round(net_ret*100,3),
                             "pnl":int(round(pnl)),"cap":int(round(cap)),
                             "win":1 if net_ret>0 else 0,"conf":round(gbdt_conf,3)})
                
                # 중복 매수 방지: 보유했던 5분봉 갯수만큼 15분봉 인덱스를 건너뜁니다.
                b_idx += max(1, int(np.ceil(holding_5m_bars / 3.0)))

        wn = wi-ROLL_WKS+1
        if wn % 30 == 0 or wn == n_wks:
            print(f"   [{wn}/{n_wks}주] {tw} | 거래:{len(trd):,} | 잔고:{int(cap):,}원")

    el = round(time.time()-t0,1)
    tr = (cap/INIT_CAP-1)*100
    print(f"\n✅ [{label}] {el}s | {tr:+.1f}% | MDD:{mdd:.1f}%")
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

def mk_ca(df, name, periods=[1, 3, 5, 20]):
    d = df.copy()
    d['datetime_dt'] = pd.to_datetime(d['datetime']) + pd.Timedelta(minutes=15)
    for p in periods:
        d[f'{name}_ret_{p}'] = d['Close'].pct_change(p) * 100.0
    d[f'{name}_dir_1']    = np.sign(d['Close'].pct_change(1))
    d[f'{name}_momentum'] = d['Close'].pct_change(1) - d['Close'].pct_change(5) / 5.0
    return d[['datetime_dt'] + [c for c in d.columns if c.startswith(name)]]

def main():
    lake = MarketDataLake()
    ml   = MLFeatureEngine()

    print("="*90+"\n📦 데이터 로드 (어제 환경 100% 재현)\n"+"="*90)
    long_15m = lake.load_candles(config.BASE_ASSET_LONG,"15m").sort_values("datetime").reset_index(drop=True)
    macro_60m = lake.load_candles(config.MACRO_TREND_SYMBOL,"60m").sort_values("datetime").reset_index(drop=True)
    nvda_15m = lake.load_candles("NVDA","15m").sort_values("datetime").reset_index(drop=True)
    qqq_15m  = lake.load_candles("QQQ","15m").sort_values("datetime").reset_index(drop=True)
    vixy_15m = lake.load_candles("VIXY","15m").sort_values("datetime").reset_index(drop=True)
    ief_15m  = lake.load_candles("IEF","15m").sort_values("datetime").reset_index(drop=True)
    
    long_5m  = lake.load_candles(config.BASE_ASSET_LONG,"5m").sort_values("datetime").reset_index(drop=True)
    short_5m  = lake.load_candles(config.BASE_ASSET_SHORT,"5m").sort_values("datetime").reset_index(drop=True)
    
    long_5m['time_str'] = pd.to_datetime(long_5m['datetime']).dt.strftime('%H:%M')
    short_5m['time_str'] = pd.to_datetime(short_5m['datetime']).dt.strftime('%H:%M')

    # CrossAsset 피처 병합 (어제와 완전 동일한 로직)
    macro_60m['ema20'] = macro_60m['Close'].ewm(span=20, adjust=False).mean()
    macro_60m['ema60'] = macro_60m['Close'].ewm(span=60, adjust=False).mean()
    macro_60m['macro_trend_spread'] = (macro_60m['ema20'] - macro_60m['ema60']) / (macro_60m['ema60'] + 1e-9) * 100.0
    macro_60m['macro_ema20_slope']  = (macro_60m['ema20'].diff(5) / macro_60m['ema20'].shift(5)) * 100.0
    macro_60m['macro_price_vs_ema20'] = (macro_60m['Close'] - macro_60m['ema20']) / (macro_60m['ema20'] + 1e-9) * 100.0
    macro_cols = ['macro_trend_spread', 'macro_ema20_slope', 'macro_price_vs_ema20']
    macro_feat = macro_60m[['datetime'] + macro_cols].copy()
    macro_feat['datetime_dt'] = pd.to_datetime(macro_feat['datetime']) + pd.Timedelta(minutes=60)

    df_feat_base = ml.extract_features(long_15m)
    df_feat_base['datetime_dt'] = pd.to_datetime(df_feat_base['datetime'])

    df_merged = pd.merge_asof(df_feat_base.sort_values('datetime_dt'),
                              macro_feat.sort_values('datetime_dt')[['datetime_dt']+macro_cols],
                              on='datetime_dt', direction='backward')
    
    for sym, raw in [('nvda', nvda_15m), ('qqq', qqq_15m), ('vixy', vixy_15m), ('ief', ief_15m)]:
        df_merged = pd.merge_asof(df_merged.sort_values('datetime_dt'),
                                  mk_ca(raw, sym).sort_values('datetime_dt'),
                                  on='datetime_dt', direction='backward')

    df_merged['long_ret_20'] = df_merged['Close'].pct_change(20) * 100.0
    df_merged['long_ret_5']  = df_merged['Close'].pct_change(5)  * 100.0
    df_merged['long_vs_qqq_20'] = df_merged['long_ret_20'] - df_merged.get('qqq_ret_20', pd.Series(0.0, index=df_merged.index)).fillna(0)
    df_merged['long_vs_qqq_5']  = df_merged['long_ret_5']  - df_merged.get('qqq_ret_5',  pd.Series(0.0, index=df_merged.index)).fillna(0)
    df_merged['panic_signal'] = (
        (df_merged.get('vixy_ret_1', pd.Series(0.0, index=df_merged.index)).fillna(0) > 0).astype(int) +
        (df_merged.get('ief_ret_1',  pd.Series(0.0, index=df_merged.index)).fillna(0) > 0).astype(int)
    )

    df_merged['date_str'] = df_merged['datetime_dt'].dt.strftime('%Y-%m-%d')
    df_merged['time_str'] = df_merged['datetime_dt'].dt.strftime('%H:%M')
    df_merged['week_id']  = (df_merged['datetime_dt'].dt.isocalendar().year.astype(str) + '-' +
                              df_merged['datetime_dt'].dt.isocalendar().week.astype(str).str.zfill(2))
    df_feat = df_merged.copy()

    EX = {'open','high','low','close','volume','Open','High','Low','Close','Volume',
          'datetime','datetime_dt','date_str','time_str','week_id','target','date','year',
          'cum_vp','vwap','EMA_9','EMA_21','EMA_50','EMA_200',
          'BB_Upper','BB_Lower','KC_Upper','KC_Lower','ATR_14','MACD','MACD_Signal','MACD_Hist'}
    fcols = [c for c in df_feat.columns if c not in EX and pd.api.types.is_numeric_dtype(df_feat[c])]
    
    # ATR_Pct 생성
    df_feat["ATR_Pct"] = (df_feat["ATR_14"] / (df_feat["Close"] + 1e-9)) * 100.0

    print(f"📊 피처 개수: {len(fcols)}개 (CrossAsset 및 Panic Signal 완벽 복원)")

    # 5분봉 딕셔너리
    long_5m_d = {d:g for d,g in long_5m.groupby(long_5m["datetime"].str[:10])}
    short_5m_d = {d:g for d,g in short_5m.groupby(short_5m["datetime"].str[:10])}

    wks = sorted(df_feat["week_id"].unique())
    print(f"📅 총 {len(wks)}주 | OOS {len(wks)-ROLL_WKS}주")

    all_res = []
    # 시나리오 A: FIXED 라벨링 (어제 방식)
    for conf in CONFS:
        lbl = f"A(FIXED)_Conf≥{int(conf*100)}%"
        res = run_wfa_scenario(df_feat, long_5m_d, short_5m_d, wks, fcols, conf, lbl, mode="FIXED")
        all_res.append(summarize(res))

    # 시나리오 B: ATR 라벨링 (일치 방식)
    for conf in CONFS:
        lbl = f"B(ATR)_Conf≥{int(conf*100)}%"
        res = run_wfa_scenario(df_feat, long_5m_d, short_5m_d, wks, fcols, conf, lbl, mode="ATR")
        all_res.append(summarize(res))

    print("\n\n"+"="*90)
    print("📊 [라벨링 방식 (Fixed vs ATR) 비교 DoE 결과]")
    print("="*90)
    print(f"{'전략':<20}|{'수익률':>10}|{'MDD':>7}|{'거래':>6}|{'승률':>7}|{'PF':>6}")
    print("-"*65)
    for r in all_res:
        print(f"{r['label']:<20}|{r['ret']:>+9.1f}%|{r['mdd']:>6.1f}%|{r['n']:>5,}|{r['wr']:>6.1f}%|{r['pf']:>6.3f}")

    op = PROJECT_ROOT/"data"/"backtest_atr_alignment_doe.json"
    with open(op,"w",encoding="utf-8") as f:
        json.dump(san({"ts":pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                       "results":[{k:v for k,v in r.items() if k!="trd"} for r in all_res]}),
                  f,indent=2,ensure_ascii=False)
    print(f"\n💾 {op}\n✅ 완료")

if __name__=="__main__":
    main()
