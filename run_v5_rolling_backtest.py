#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, time, pandas as pd, numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
import config
from core.data_lake import MarketDataLake
from core.ml_engine import MLFeatureEngine

if sys.platform.startswith('win'):
    try: sys.stdout.reconfigure(encoding='utf-8')
    except: pass

import config

def run_v5_rolling_backtest():
    print("=" * 90)
    print("🚀 [Lumos V5 Walk-Forward Backtest (Ultra-Fast Vectorized)]")
    print("=" * 90)

    t_start = time.time()
    print("⏳ [1/4] 전체 데이터 메모리 캐싱 및 패치 중...")
    
    dl = MarketDataLake()
    symbols = list(set(["TQQQ", "SQQQ", "NVDA", "QQQ", "VIXY", "IEF", config.MACRO_TREND_SYMBOL]))
    all_data = {"15m": {}, "60m": {}, "5m": {}}

    for s in symbols:
        for tf in ["15m", "60m", "5m"]:
            df = dl.load_candles(s, tf).reset_index()
            if 'datetime' in df.columns and 'Datetime' not in df.columns:
                df.rename(columns={'datetime': 'Datetime'}, inplace=True)
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            df.set_index('Datetime', inplace=True, drop=False)
            all_data[tf][s] = df

    # 1차 몽키패치: MarketDataLake가 메모리 캐시에서 즉시 반환하도록 함 (SQLite 접근 원천 차단)
    def fast_load_candles(self, symbol, timeframe, start_dt=None, end_dt=None):
        sym = symbol.upper().strip()
        tf = timeframe.lower().strip()
        print(f"      [Patched] load_candles called for {sym} {tf}")
        df = all_data.get(tf, {}).get(sym, pd.DataFrame())
        return df.copy()
    MarketDataLake.load_candles = fast_load_candles
    
    # SQLite Lock 방지
    MarketDataLake.__init__ = lambda self: None
    
    print("⏳ [2/4] 피처 일괄 계산 중...", flush=True)
    engine = MLFeatureEngine()
    tqqq_df = all_data["15m"]["TQQQ"].copy()
    
    # 이 호출은 패치된 load_candles를 사용하므로 0.1초만에 완료됨
    feat_df = engine.extract_features(tqqq_df)
    
    feat_df_cache = feat_df.copy()
    if 'Datetime' not in feat_df_cache.columns and 'datetime' in feat_df_cache.columns:
        feat_df_cache.rename(columns={'datetime': 'Datetime'}, inplace=True)
    if 'Datetime' not in feat_df_cache.columns:
        feat_df_cache = feat_df_cache.reset_index()
        feat_df_cache.rename(columns={'index': 'Datetime', 'datetime': 'Datetime'}, inplace=True, errors='ignore')
    feat_df_cache['Datetime'] = pd.to_datetime(feat_df_cache['Datetime'])
    feat_df_cache.set_index('Datetime', inplace=True, drop=False)

    # 2차 몽키패치: ML 학습 시 피처를 재계산하지 않고 캐시에서 슬라이싱만 하도록 함
    def fast_extract_features(self, df, live_prices=None):
        if df is None or df.empty: return pd.DataFrame()
        dt_col = 'Datetime' if 'Datetime' in df.columns else 'datetime'
        
        # DataFrame/Series handling
        col_data = df[dt_col]
        if isinstance(col_data, pd.DataFrame):
            col_data = col_data.iloc[:, 0]
            
        start = col_data.iloc[0]
        end = col_data.iloc[-1]
        
        res = feat_df_cache.loc[start:end].copy()
        if dt_col == 'datetime' and 'Datetime' in res.columns:
            res.rename(columns={'Datetime': 'datetime'}, inplace=True)
        return res
    MLFeatureEngine.extract_features = fast_extract_features

    # Master DF 준비 (Vectorized Trading 용)
    master = tqqq_df[['Datetime', 'Open', 'High', 'Low', 'Close']].copy().reset_index(drop=True)
    master['tqqq_ret5'] = master['Close'] / master['Close'].shift(5) - 1.0
    
    cols_to_merge = [c for c in feat_df_cache.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume', 'open', 'high', 'low', 'close', 'volume']]
    if 'Datetime' not in cols_to_merge: cols_to_merge.append('Datetime')
    master = master.merge(feat_df_cache[cols_to_merge].reset_index(drop=True), on='Datetime', how='left')

    sqqq = all_data["15m"]["SQQQ"][['Datetime', 'High', 'Low', 'Close']].copy().reset_index(drop=True)
    sqqq.rename(columns={'High': 'SQQQ_High', 'Low': 'SQQQ_Low', 'Close': 'SQQQ_Close'}, inplace=True)
    master = master.merge(sqqq, on='Datetime', how='left')

    soxx = all_data["15m"]["SOXX"][['Datetime', 'Close']].copy().reset_index(drop=True)
    soxx.rename(columns={'Close': 'SOXX_Close'}, inplace=True)
    master = master.merge(soxx, on='Datetime', how='left')

    soxx60 = all_data["60m"]["SOXX"][['Datetime', 'Close']].copy().reset_index(drop=True)
    soxx60['soxx_ema20_60m'] = soxx60['Close'].ewm(span=20, adjust=False).mean()
    soxx60 = soxx60[['Datetime', 'soxx_ema20_60m']].dropna()
    master = pd.merge_asof(master.sort_values('Datetime'), soxx60.sort_values('Datetime'), on='Datetime', direction='backward')

    master.ffill(inplace=True)
    master.fillna(0, inplace=True)

    print(f"   ✅ 데이터 준비 완료: {len(master):,} rows ({time.time()-t_start:.1f}s)", flush=True)

    # ────────────────────────────────────────────────────────────────
    # 백테스트 파라미터 
    # ────────────────────────────────────────────────────────────────
    INITIAL_CAPITAL = 10000.0
    capital = INITIAL_CAPITAL
    peak_capital = INITIAL_CAPITAL
    max_drawdown = 0.0
    trades = []
    weekly_results = []
    
    CONFIDENCE_THRESHOLD = config.GBDT_CONFIDENCE_THRESHOLD
    MAX_TP_PCT = config.MAX_TP_PCT
    SL_MIN = config.SL_MIN_PCT
    SL_MAX = config.SL_MAX_PCT
    SL_ATR_MULT = config.SL_ATR_MULTIPLIER
    TRAILING_TRIGGER = config.TRAILING_TRIGGER_PCT
    TRAILING_DROP = config.TRAILING_DROP_PCT
    TIME_STOP_BARS = int(config.TIME_STOP_MINUTES / 15)
    SLIPPAGE_PAYUP = config.BUY_SLIPPAGE_ADJUST
    SELL_SLIPPAGE = config.SELL_SLIPPAGE_ADJUST
    FEE = config.BACKTEST_FEE_SLIPPAGE
    ALLOC_RATIO = config.MAX_ALLOCATION_RATIO
    RSI_OB = config.RSI_OVERBOUGHT_THRESHOLD
    RSI_OS = config.RSI_OVERSOLD_THRESHOLD

    min_date = master['Datetime'].min()
    max_date = master['Datetime'].max()
    test_start = min_date + timedelta(days=730)
    while test_start.weekday() != 0: test_start += timedelta(days=1)
    test_start = test_start.replace(hour=0, minute=0, second=0)

    print("\n⏳ [3/4] Walk-Forward 롤링 백테스트 초고속 실행 중...", flush=True)
    
    dates = master['Datetime'].values
    times = master['Datetime'].dt.strftime('%H:%M').values
    c_tqqq = master['Close'].values
    h_tqqq = master['High'].values
    l_tqqq = master['Low'].values
    c_sqqq = master['SQQQ_Close'].values
    h_sqqq = master['SQQQ_High'].values
    l_sqqq = master['SQQQ_Low'].values

    # 크로스에셋 수익률 계산 후 master에 머지
    df_nvda = all_data["15m"]["NVDA"][['Datetime', 'Close']].copy().reset_index(drop=True)
    df_nvda['nvda_ret5'] = df_nvda['Close'] / df_nvda['Close'].shift(5) - 1.0
    master = master.merge(df_nvda[['Datetime', 'nvda_ret5']], on='Datetime', how='left')

    df_qqq = all_data["15m"]["QQQ"][['Datetime', 'Close']].copy().reset_index(drop=True)
    df_qqq['qqq_ret5'] = df_qqq['Close'] / df_qqq['Close'].shift(5) - 1.0
    master = master.merge(df_qqq[['Datetime', 'qqq_ret5']], on='Datetime', how='left')

    df_vixy = all_data["15m"]["VIXY"][['Datetime', 'Close']].copy().reset_index(drop=True)
    df_vixy['vixy_ret5'] = df_vixy['Close'] / df_vixy['Close'].shift(5) - 1.0
    master = master.merge(df_vixy[['Datetime', 'vixy_ret5']], on='Datetime', how='left')

    master['nvda_ret5'] = master['nvda_ret5'].ffill().fillna(0)
    master['qqq_ret5'] = master['qqq_ret5'].ffill().fillna(0)
    master['vixy_ret5'] = master['vixy_ret5'].ffill().fillna(0)
    master['soxx_ret5'] = master['SOXX_Close'] / master['SOXX_Close'].shift(5) - 1.0

    macro_score = (master['nvda_ret5'] * 0.40) + (master['soxx_ret5'] * 0.30) + (master['qqq_ret5'] * 0.20) - (master['vixy_ret5'] * 0.10)
    dislocation = macro_score * 3.0 - master['tqqq_ret5']
    
    is_60m_trend_long = master['SOXX_Close'] >= master['soxx_ema20_60m'] * 0.998
    is_60m_trend_short = master['SOXX_Close'] <= master['soxx_ema20_60m'] * 1.002
    rsi_14 = master['RSI_14'].values

    week_count = 0
    while test_start < max_date:
        test_end = test_start + timedelta(days=7)
        train_start = test_start - timedelta(days=730)

        train_mask = (master['Datetime'] >= train_start) & (master['Datetime'] < test_start)
        test_mask = (master['Datetime'] >= test_start) & (master['Datetime'] < test_end)
        
        train_df = master[train_mask]
        if len(train_df) < 1000:
            test_start = test_end
            continue
            
        test_indices = np.where(test_mask)[0]
        if len(test_indices) == 0:
            test_start = test_end
            continue
            
        engine.train_and_select_top_features(train_df.rename(columns={'Datetime': 'datetime'}))
        
        X_test = master.loc[test_indices, engine.feature_names].fillna(0.0)
        probs = engine.model.predict_proba(X_test)
        classes = list(engine.model.classes_)
        p_short = probs[:, classes.index(0)] if 0 in classes else np.zeros(len(X_test))
        p_neutral = probs[:, classes.index(1)] if 1 in classes else np.ones(len(X_test))
        p_long = probs[:, classes.index(2)] if 2 in classes else np.zeros(len(X_test))

        gbdt_dirs = np.full(len(X_test), "NONE", dtype=object)
        for i in range(len(X_test)):
            ps, pn, pl = p_short[i], p_neutral[i], p_long[i]
            if pl > pn and pl > ps:
                calib_conf = min(0.95, max(0.50, 0.50 + (pl - 0.333) * 1.15))
                if calib_conf >= CONFIDENCE_THRESHOLD: gbdt_dirs[i] = "LONG_TQQQ"
            elif ps > pn and ps > pl:
                calib_conf = min(0.95, max(0.50, 0.50 + (ps - 0.333) * 1.15))
                if calib_conf >= CONFIDENCE_THRESHOLD: gbdt_dirs[i] = "SHORT_SQQQ"

        week_start_capital = capital
        week_trades = 0
        week_wins = 0
        active_pos = None
        
        for local_i, global_i in enumerate(test_indices):
            bar_time = times[global_i]
            
            if active_pos is not None:
                sym = active_pos['symbol']
                buy_px = active_pos['buy_price']
                bars_held = local_i - active_pos['entry_local_idx']
                
                if sym == 'TQQQ':
                    cur_h, cur_l, cur_c = h_tqqq[global_i], l_tqqq[global_i], c_tqqq[global_i]
                else:
                    cur_h, cur_l, cur_c = h_sqqq[global_i], l_sqqq[global_i], c_sqqq[global_i]
                if cur_c == 0: cur_h = cur_l = cur_c = buy_px
                    
                max_ret = (cur_h - buy_px) / buy_px
                min_ret = (cur_l - buy_px) / buy_px
                cur_ret = (cur_c - buy_px) / buy_px
                
                if max_ret > active_pos['peak_ret']: active_pos['peak_ret'] = max_ret
                if active_pos['peak_ret'] >= TRAILING_TRIGGER: active_pos['trailing_active'] = True
                    
                exit_triggered, exit_px, reason = False, 0.0, ""
                sl_pct = active_pos['initial_sl']
                
                if max_ret >= MAX_TP_PCT:
                    exit_triggered = True; exit_px = buy_px * (1 + MAX_TP_PCT) - SELL_SLIPPAGE; reason = f"🎯 목표익절 (+{MAX_TP_PCT*100:.1f}%)"
                elif active_pos['trailing_active'] and (active_pos['peak_ret'] - cur_ret >= TRAILING_DROP):
                    exit_triggered = True; exit_px = cur_c - SELL_SLIPPAGE; reason = f"🛡️ 트레일링 스탑 (고점-{TRAILING_DROP*100:.1f}%)"
                elif min_ret <= sl_pct:
                    exit_triggered = True; exit_px = buy_px * (1 + sl_pct) - SELL_SLIPPAGE; reason = f"🛑 손절 ({sl_pct*100:.1f}%)"
                elif bars_held >= TIME_STOP_BARS:
                    exit_triggered = True; exit_px = cur_c - SELL_SLIPPAGE; reason = f"⏰ {config.TIME_STOP_MINUTES}분 타임스탑"
                elif bar_time >= "15:50":
                    exit_triggered = True; exit_px = cur_c - SELL_SLIPPAGE; reason = "🌙 장마감 청산 (15:50)"
                    
                if exit_triggered:
                    cost_amount = active_pos['invested'] * FEE
                    pnl_amount = (active_pos['shares'] * (exit_px - buy_px)) - cost_amount
                    capital += pnl_amount
                    net_ret_pct = pnl_amount / active_pos['invested']
                    
                    trades.append({
                        "date": pd.to_datetime(dates[global_i]).strftime('%Y-%m-%d'), "symbol": sym, "direction": active_pos['direction'],
                        "entry_time": active_pos['entry_time'], "entry_price": buy_px, "exit_time": bar_time, "exit_price": exit_px,
                        "return_pct": round(net_ret_pct * 100, 2), "pnl_usd": round(pnl_amount, 2), "capital_after": round(capital, 2), "exit_reason": reason
                    })
                    week_trades += 1
                    if pnl_amount > 0: week_wins += 1
                    active_pos = None

            if active_pos is None and "09:30" <= bar_time <= "15:30":
                g_dir = gbdt_dirs[local_i]
                if g_dir in ["LONG_TQQQ", "SHORT_SQQQ"]:
                    disloc = dislocation.values[global_i]
                    is_veto = (g_dir == "LONG_TQQQ" and disloc <= -0.012) or (g_dir == "SHORT_SQQQ" and disloc >= 0.012)
                    if not getattr(config, "USE_CROSS_ASSET_VETO", True):
                        is_veto = False
                    
                    trend_ok = is_60m_trend_long.values[global_i] if g_dir == "LONG_TQQQ" else is_60m_trend_short.values[global_i]
                    if not getattr(config, "USE_60M_TREND_FILTER", True):
                        trend_ok = True
                        
                    cur_rsi = rsi_14[global_i]
                    dip_ok = (cur_rsi <= RSI_OB) if g_dir == "LONG_TQQQ" else (cur_rsi >= RSI_OS)
                    
                    if not is_veto and trend_ok and dip_ok:
                        sym = "TQQQ" if g_dir == "LONG_TQQQ" else "SQQQ"
                        base_px = c_tqqq[global_i] if sym == "TQQQ" else c_sqqq[global_i]
                        if base_px > 0:
                            entry_px = base_px + SLIPPAGE_PAYUP
                            shares = int((capital * ALLOC_RATIO) / (entry_px + config.QTY_CALC_BUFFER))
                            invested = shares * entry_px
                            
                            if shares > 0 and invested > 0:
                                atr_14 = np.mean(h_tqqq[global_i-13:global_i+1] - l_tqqq[global_i-13:global_i+1]) if global_i >= 13 else (h_tqqq[global_i]-l_tqqq[global_i])
                                atr_sl_pct = (atr_14 * SL_ATR_MULT) / base_px
                                initial_sl = max(SL_MIN, min(SL_MAX, atr_sl_pct)) * -1.0
                                
                                active_pos = {
                                    "symbol": sym, "direction": g_dir, "entry_time": bar_time, "entry_local_idx": local_i,
                                    "buy_price": entry_px, "shares": shares, "invested": invested, "initial_sl": initial_sl,
                                    "peak_ret": 0.0, "trailing_active": False
                                }

        week_pnl = capital - week_start_capital
        week_ret_pct = (week_pnl / week_start_capital) * 100 if week_start_capital > 0 else 0

        if capital > peak_capital: peak_capital = capital
        current_dd = (peak_capital - capital) / peak_capital * 100 if peak_capital > 0 else 0
        if current_dd > max_drawdown: max_drawdown = current_dd

        week_str = test_start.strftime("%Y-%m-%d")
        weekly_results.append({
            "Week": week_str, "Trades": week_trades, "Wins": week_wins, "Losses": week_trades - week_wins,
            "Weekly_PnL": round(week_pnl, 2), "Weekly_Return_Pct": round(week_ret_pct, 2),
            "Cumulative_Capital": round(capital, 2), "Drawdown_Pct": round(current_dd, 2)
        })
        week_count += 1
        
        test_start = test_end

    total_elapsed = time.time() - t_start

    print("\n" + "=" * 90)
    print("🏆 [Lumos V5 Walk-Forward Rolling Backtest 초고속 성적표]")
    print("=" * 90)

    total_trades = sum(w["Trades"] for w in weekly_results)
    total_wins = sum(w["Wins"] for w in weekly_results)
    total_losses = total_trades - total_wins
    total_return_pct = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    win_rate = (total_wins / total_trades) * 100 if total_trades > 0 else 0
    years_elapsed = max((pd.to_datetime(weekly_results[-1]["Week"]) - pd.to_datetime(weekly_results[0]["Week"])).days / 365.25, 0.01) if weekly_results else 1.0
    cagr = ((capital / INITIAL_CAPITAL) ** (1.0 / years_elapsed) - 1.0) * 100 if capital > 0 else 0.0

    winning_trades = [t for t in trades if t["pnl_usd"] > 0]
    losing_trades = [t for t in trades if t["pnl_usd"] <= 0]
    avg_win = np.mean([t["return_pct"] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t["return_pct"] for t in losing_trades]) if losing_trades else 0
    
    sum_win_pnl = sum(t["pnl_usd"] for t in winning_trades)
    sum_loss_pnl = abs(sum(t["pnl_usd"] for t in losing_trades))
    profit_factor = sum_win_pnl / sum_loss_pnl if sum_loss_pnl != 0 else float('inf')

    print(f"\n📊 소요 시간: {total_elapsed:.1f}초")
    print(f"💰 초기 자본금:    ${INITIAL_CAPITAL:>12,.2f}")
    print(f"💵 최종 자본금:    ${capital:>12,.2f}")
    print(f"📈 총 누적 수익률: {total_return_pct:>+11.2f}%")
    print(f"📈 연평균 수익률(CAGR): {cagr:>+8.2f}%")
    print(f"📉 최대 낙폭(MDD):     {max_drawdown:>8.2f}%")
    print(f"{'─'*50}")
    print(f"🎯 총 거래 횟수:   {total_trades:>6}건")
    print(f"✅ 승률:           {win_rate:>7.1f}%  ({total_wins}승 / {total_losses}패)")
    print(f"📊 Profit Factor:  {profit_factor:>7.2f}")
    print(f"📊 평균 수익/손실: {avg_win:>+7.2f}% / {avg_loss:>+7.2f}%")

    report_dir = PROJECT_ROOT / "backtest_history"
    report_dir.mkdir(exist_ok=True)
    report_filename = datetime.now().strftime("%Y%m%d_%H%M%S") + "_fast_backtest_report.md"
    report_path = report_dir / report_filename
    with open(report_path, 'w', encoding='utf-8') as f:
         f.write(f"# Lumos V5 Fast Vectorized Backtest\n**기간**: {years_elapsed:.1f}년\n**수익률**: {total_return_pct:.2f}%\n**MDD**: {max_drawdown:.2f}%\n**거래횟수**: {total_trades}건\n")
    print(f"   ✅ 리포트 저장 완료: {report_path}")

if __name__ == "__main__":
    run_v5_rolling_backtest()
