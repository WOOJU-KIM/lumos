import os
import sys
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_DIR
from core.data_lake import MarketDataLake
from core.ml_engine import MLFeatureEngine
import config
from core.heterogeneous_models import CrossAssetDislocationModel

def run_identical_wfa():
    lake = MarketDataLake()
    print("=" * 100)
    print("🚀 [Lumos 퀀트 시스템: 실전 모델 100% 동일 주간 롤링 WFA 백테스트]")
    print("   • 실전 매매 로직 100% 동일 적용 (ATR 트레일링 스탑, Veto, 60m/5m 멀티스크린)")
    print("   • 롤링 윈도우: 104주 (2년) 학습 -> 1주 테스트")
    print("   • 페널티: 왕복 수수료 및 슬리피지 최소 0.20% (0.0020) 일괄 차감")
    print("=" * 100)

    # 1. Load Data
    print("⏳ 데이터 로드 중...")
    tqqq_15m = lake.load_candles("TQQQ", "15m")
    soxx_60m = lake.load_candles("SOXX", "60m")
    tqqq_5m = lake.load_candles("TQQQ", "5m")
    sqqq_5m = lake.load_candles("SQQQ", "5m")
    
    nvda_15m = lake.load_candles("NVDA", "15m")
    soxx_15m = lake.load_candles("SOXX", "15m")
    qqq_15m  = lake.load_candles("QQQ", "15m")

    # 2. Precompute Features
    print("⏳ 머신러닝 피처 생성 중...")
    ml_engine = MLFeatureEngine(confidence_threshold=0.62)
    tqqq_feat = ml_engine.extract_features(tqqq_15m)
    labels = ml_engine.compute_triple_barrier_labels(tqqq_feat)
    tqqq_feat['target'] = labels.map({1: 2, -1: 0, 0: 1}).fillna(1).astype(int)
    
    tqqq_feat['datetime_dt'] = pd.to_datetime(tqqq_feat['datetime'])
    tqqq_feat['week_id'] = tqqq_feat['datetime_dt'].dt.isocalendar().year.astype(str) + '-' + tqqq_feat['datetime_dt'].dt.isocalendar().week.astype(str).str.zfill(2)

    # 60m 20EMA
    soxx_60m['ema20'] = soxx_60m['Close'].ewm(span=20, adjust=False).mean()
    
    # 3. Mappings for speed
    nvda_map = nvda_15m.set_index('datetime')['Close'].to_dict()
    soxx_map = soxx_15m.set_index('datetime')['Close'].to_dict()
    qqq_map  = qqq_15m.set_index('datetime')['Close'].to_dict()
    soxx_60_dt = soxx_60m.set_index('datetime')
    
    tqqq_5_dt = tqqq_5m.set_index('datetime')
    sqqq_5_dt = sqqq_5m.set_index('datetime')
    
    cross_mod = CrossAssetDislocationModel(dislocation_z_threshold=1.6)
    
    exclude_cols = {'open', 'high', 'low', 'close', 'volume', 'Open', 'High', 'Low', 'Close', 'Volume', 'datetime', 'datetime_dt', 'week_id', 'target', 'date', 'date_str', 'time_str'}
    feature_cols = [c for c in tqqq_feat.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(tqqq_feat[c])]
    
    unique_weeks = sorted(tqqq_feat['week_id'].unique())
    ROLLING_WINDOW = 104
    test_start_idx = ROLLING_WINDOW
    
    if len(unique_weeks) <= ROLLING_WINDOW:
        print("데이터 부족")
        return

    capital = config.INITIAL_CAPITAL_KRW if 'INITIAL_CAPITAL_KRW' in dir(config) else 10_000_000.0
    initial_capital = capital
    slippage_fee = config.BACKTEST_FEE_SLIPPAGE
    trades = []
    trade_id = 0
    
    print(f"📊 학습 피처: {len(feature_cols)}개, 테스트 주차: {len(unique_weeks) - test_start_idx}주")
    t0 = time.time()

    for w_idx in range(test_start_idx, len(unique_weeks)):
        test_week = unique_weeks[w_idx]
        train_weeks = unique_weeks[w_idx - ROLLING_WINDOW : w_idx]
        
        train_mask = tqqq_feat['week_id'].isin(train_weeks)
        X_train = tqqq_feat.loc[train_mask, feature_cols].fillna(0.0)
        y_train = tqqq_feat.loc[train_mask, 'target']
        
        if len(X_train) < 100: continue
            
        import lightgbm as lgb
        clf = lgb.LGBMClassifier(objective='multiclass', num_class=3, class_weight='balanced', n_estimators=80, max_depth=4, learning_rate=0.03, random_state=42, verbosity=-1, n_jobs=-1)
        clf.fit(X_train, y_train)
        
        test_mask = tqqq_feat['week_id'] == test_week
        test_df = tqqq_feat[test_mask].copy()
        if test_df.empty: continue
            
        X_test = test_df[feature_cols].fillna(0.0)
        probs = clf.predict_proba(X_test)
        
        # Test loop
        for i in range(len(test_df)):
            row = test_df.iloc[i]
            curr_dt = row['datetime']
            time_str = row['datetime_dt'].strftime('%H:%M')
            if time_str >= '15:30': continue # 쿨다운
                
            pl, pn, ps = probs[i, 2], probs[i, 1], probs[i, 0]
            dir_gbdt = "NONE"
            conf = 0.50
            if pl > pn and pl > ps:
                conf = min(0.95, max(0.50, 0.50 + (pl - 0.333)*1.15))
                dir_gbdt = "LONG_TQQQ"
            elif ps > pn and ps > pl:
                conf = min(0.95, max(0.50, 0.50 + (ps - 0.333)*1.15))
                dir_gbdt = "SHORT_SQQQ"
                
            if conf < 0.62 or dir_gbdt == "NONE":
                continue
                
            # 60m trend
            past_soxx = soxx_60_dt[soxx_60_dt.index <= curr_dt]
            if len(past_soxx) < 20: continue
            soxx_c = past_soxx['Close'].iloc[-1]
            soxx_ema = past_soxx['ema20'].iloc[-1]
            
            is_60m_ok = False
            if dir_gbdt == "LONG_TQQQ":
                is_60m_ok = (soxx_c >= soxx_ema * 0.998)
            else:
                is_60m_ok = (soxx_c <= soxx_ema * 1.002)
            if not is_60m_ok: continue
                
            # CrossAsset Veto
            cross_dir = "HOLD"
            cur_idx_raw = test_df.index[i]
            feat_loc = tqqq_feat.index.get_loc(cur_idx_raw)
            if feat_loc >= 5:
                prev_dt = tqqq_feat.iloc[feat_loc - 5]['datetime']
                
                n_px = nvda_map.get(curr_dt); p_n = nvda_map.get(prev_dt, n_px)
                sx_px = soxx_map.get(curr_dt); p_sx = soxx_map.get(prev_dt, sx_px)
                q_px = qqq_map.get(curr_dt); p_q = qqq_map.get(prev_dt, q_px)
                
                if n_px and p_n and sx_px and p_sx and q_px and p_q:
                    nvda_r = (n_px / p_n) - 1.0
                    soxx_r = (sx_px / p_sx) - 1.0
                    qqq_r = (q_px / p_q) - 1.0
                    tqqq_r = (row['Close'] / tqqq_feat.iloc[feat_loc - 5]['Close']) - 1.0
                    
                    sig_code, _, _ = cross_mod.predict_signal(tqqq_r, nvda_r, soxx_r, qqq_r, 0.0, 0.0)
                    if sig_code > 0: cross_dir = "LONG_TQQQ"
                    elif sig_code < 0: cross_dir = "SHORT_SQQQ"
            
            is_veto = (dir_gbdt == "LONG_TQQQ" and cross_dir == "SHORT_SQQQ") or (dir_gbdt == "SHORT_SQQQ" and cross_dir == "LONG_TQQQ")
            if not getattr(config, "USE_CROSS_ASSET_VETO", True):
                is_veto = False
            if is_veto: continue
                
            # 5m RSI dip
            rsi_14 = float(row.get('RSI_14', 50))
            if dir_gbdt == "LONG_TQQQ" and rsi_14 > 62.0: continue
            if dir_gbdt == "SHORT_SQQQ" and rsi_14 < 38.0: continue
                
            # PASS! We have an entry. Manage position using 5m bars.
            target_sym = "TQQQ" if dir_gbdt == "LONG_TQQQ" else "SQQQ"
            target_dt_df = tqqq_5_dt if target_sym == "TQQQ" else sqqq_5_dt
            
            # 90m -> 120m means 24 bars of 5m
            future_bars = target_dt_df[target_dt_df.index >= curr_dt].head(25) # 120 min = 24 bars + 1 entry
            if len(future_bars) < 2: continue
                
            entry_px = float(row.get('close', row.get('Close', 0.0)))
            if target_sym == "SQQQ":
                if curr_dt in sqqq_5m.index:
                    row_5m = sqqq_5m.loc[curr_dt]
                    entry_px = float(row_5m.get('close', row_5m.get('Close', 0.0)))
                else:
                    entry_px = 0.0
            
            if entry_px <= 0: continue
            
            sl_px = entry_px * 0.968
            tp_px = entry_px * (1.0 + config.MAX_TP_PCT)
            
            atr_14 = float(row.get('ATR_14', entry_px * 0.018))
            atr_pct = (atr_14 / entry_px) * 100.0 if entry_px > 0 else 1.8
            sl_pct = max(2.0, min(3.2, atr_pct * 1.5)) / 100.0
            
            sl_px_initial = entry_px * (1.0 - sl_pct)
            
            current_sl_px = sl_px_initial
            trailing_trigger_px = entry_px * (1.0 + 0.02)
            
            peak_high = entry_px
            exit_px = 0.0
            exit_reason = ""
            bars_held = 0
            is_trailing = False
            
            for j in range(1, len(future_bars)):
                f_row = future_bars.iloc[j]
                f_dt = future_bars.index[j]
                f_time = pd.to_datetime(f_dt).strftime('%H:%M')
                
                f_high = float(f_row['High'])
                f_low = float(f_row['Low'])
                f_close = float(f_row['Close'])
                bars_held += 1
                
                if f_high > peak_high:
                    peak_high = f_high
                    if peak_high >= trailing_trigger_px:
                        is_trailing = True
                        current_sl_px = max(current_sl_px, entry_px * 1.005)
                    if is_trailing:
                        current_sl_px = max(current_sl_px, peak_high * (1.0 - 0.008))
                
                if f_time >= '15:50':
                    exit_px = f_close
                    exit_reason = "EOD_1550"
                    break
                    
                if f_high >= tp_px:
                    exit_px = tp_px
                    exit_reason = "MAX_TP_3.5%"
                    break
                    
                if f_low <= current_sl_px:
                    exit_px = current_sl_px
                    exit_reason = "TRAILING_SL" if is_trailing else "ATR_SL"
                    break
                    
                if bars_held >= 18:
                    exit_px = f_close
                    exit_reason = "TIME_STOP_120m"
                    break
                    
            if exit_px > 0:
                raw_ret = (exit_px / entry_px) - 1.0
                net_ret = raw_ret - slippage_fee
                
                # 99% allocation rule check (simulated)
                order_qty = int((capital * config.MAX_ALLOCATION_RATIO) / (entry_px + config.QTY_CALC_BUFFER))
                pos_capital = order_qty * entry_px
                pnl = pos_capital * net_ret
                
                capital += pnl
                trade_id += 1
                trades.append({
                    "trade_id": trade_id,
                    "date": curr_dt[:10],
                    "entry_dt": curr_dt,
                    "symbol": target_sym,
                    "entry_px": round(entry_px, 2),
                    "exit_px": round(exit_px, 2),
                    "net_ret_pct": round(net_ret * 100, 2),
                    "exit_reason": exit_reason,
                    "capital": int(capital)
                })

        if w_idx % 20 == 0 or w_idx == len(unique_weeks) - 1:
            print(f"   • [{w_idx}/{len(unique_weeks)}] {test_week} 완료 | 누적매매: {len(trades)}회 | 잔고: {capital:,.0f}원")

    elapsed = time.time() - t0
    df = pd.DataFrame(trades)
    print(f"\n✅ 백테스트 완료! ({elapsed:.1f}초 소요)")
    print(f"총 매매: {len(df)}회, 최종 잔고: {capital:,.0f}원")
    
    output_path = DATA_DIR / "wfa_live_identical_trades.csv"
    if not df.empty:
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"저장 완료: {output_path}")
        
        wins = df[df['net_ret_pct'] > 0]
        win_rate = len(wins) / len(df) * 100
        total_ret = (capital / initial_capital - 1.0) * 100
        
        print("-" * 50)
        print(f"📈 최종 승률: {win_rate:.1f}%")
        print(f"💰 누적 수익률: {total_ret:+.1f}%")
        print("-" * 50)

if __name__ == "__main__":
    run_identical_wfa()
