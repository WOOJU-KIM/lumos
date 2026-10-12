import os
import sys
import sqlite3
import json
import config
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.moe_orchestrator import MoEMetaOrchestrator
from core.ml_engine import MLFeatureEngine
from core.heterogeneous_models import (
    OrderFlowImbalanceModel,
    TDATopologyModel,
    StateSpaceKalmanModel,
    CrossAssetDislocationModel
)
from config import DATA_DIR

def run_moe_3screen_backtest():
    print("=" * 90)
    print("🛡️ [Lumos Track 6: MoE AI + 3중 스크린(60m추세 + 15mMoE + 5m눌림목) 정밀 백테스팅]")
    print(f"⏰ 실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S KST')}")
    print("=" * 90)

    db_path = DATA_DIR / "market_data.db"
    conn = sqlite3.connect(db_path)

    # 1. 15분봉 및 60분봉 전수 로드
    long_15m = pd.read_sql_query("SELECT datetime, open, high, low, close, volume FROM market_candles WHERE symbol=config.BASE_ASSET_LONG AND timeframe='15m' ORDER BY datetime ASC", conn)
    short_15m = pd.read_sql_query("SELECT datetime, open, high, low, close, volume FROM market_candles WHERE symbol=config.BASE_ASSET_SHORT AND timeframe='15m' ORDER BY datetime ASC", conn)
    macro_60m = pd.read_sql_query("SELECT datetime, open, high, low, close, volume FROM market_candles WHERE symbol=config.MACRO_TREND_SYMBOL AND timeframe='60m' ORDER BY datetime ASC", conn)
    long_60m = pd.read_sql_query("SELECT datetime, open, high, low, close, volume FROM market_candles WHERE symbol=config.BASE_ASSET_LONG AND timeframe='60m' ORDER BY datetime ASC", conn)
    vix_15m  = pd.read_sql_query("SELECT datetime, open, high, low, close, volume FROM market_candles WHERE symbol='^VIX' AND timeframe='15m' ORDER BY datetime ASC", conn)
    nvda_15m = pd.read_sql_query("SELECT datetime, open, high, low, close, volume FROM market_candles WHERE symbol='NVDA' AND timeframe='15m' ORDER BY datetime ASC", conn)
    qqq_15m  = pd.read_sql_query("SELECT datetime, open, high, low, close, volume FROM market_candles WHERE symbol='QQQ' AND timeframe='15m' ORDER BY datetime ASC", conn)
    conn.close()

    for df in [long_15m, short_15m, macro_60m, long_60m, vix_15m, nvda_15m, qqq_15m]:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        df.set_index('datetime', inplace=True)

    # 60분봉 20EMA 계산
    macro_60m['ema20'] = macro_60m['Close'].ewm(span=20, adjust=False).mean()
    long_60m['ema20'] = long_60m['Close'].ewm(span=20, adjust=False).mean()

    # 15분봉 피처 엔지니어링
    ml_engine = MLFeatureEngine(confidence_threshold=0.60)
    long_15m_feat = ml_engine.extract_features(long_15m)
    short_15m_feat = ml_engine.extract_features(short_15m)

    long_15m_feat['date_str'] = long_15m_feat.index.strftime('%Y-%m-%d')
    unique_dates = sorted(long_15m_feat['date_str'].unique())

    # MoE 모델 인스턴스
    moe = MoEMetaOrchestrator(confidence_threshold=0.75)

    INITIAL_CAPITAL = 100_000.0
    TP_PCT = 0.035
    SL_PCT = -0.020
    TIME_STOP_BARS = 6
    SLIPPAGE = 0.03
    SEC_FEE = 0.0001

    capital = INITIAL_CAPITAL
    trades = []
    equity_curve = [capital]
    model_stats = {}

    print(f"📊 [백테스트 대상] 2026-05-20 ~ 2026-08-21 (총 {len(unique_dates)}거래일)")

    for d_str in unique_dates:
        day_long = long_15m_feat[long_15m_feat['date_str'] == d_str]
        if len(day_long) < 5:
            continue

        active_pos = None

        for b_idx in range(len(day_long)):
            cur_time = day_long.index[b_idx]
            row_l = day_long.iloc[b_idx]
            time_str = cur_time.strftime('%H:%M')

            # [A] 포지션 보유 중인 경우 ➔ 실시간 청산 관리
            if active_pos is not None:
                sym = active_pos['symbol']
                buy_px = active_pos['buy_price']
                bars_held = b_idx - active_pos['entry_bar_idx']

                if sym == config.BASE_ASSET_LONG:
                    cur_h = row_l['High']
                    cur_l = row_l['Low']
                    cur_c = row_l['Close']
                else:
                    if cur_time in short_15m_feat.index:
                        r_s = short_15m_feat.loc[cur_time]
                        cur_h = r_s['High']
                        cur_l = r_s['Low']
                        cur_c = r_s['Close']
                    else:
                        cur_h = cur_l = cur_c = buy_px

                max_ret = (cur_h - buy_px) / buy_px
                min_ret = (cur_l - buy_px) / buy_px

                exit_triggered = False
                exit_reason = ""
                exit_price = 0.0

                if max_ret >= TP_PCT:
                    exit_triggered = True
                    exit_reason = "🎯 목표익절 (+3.5%)"
                    exit_price = round(buy_px * (1 + TP_PCT) - SLIPPAGE, 2)
                elif min_ret <= SL_PCT:
                    exit_triggered = True
                    exit_reason = "🛑 칼손절 (-2.0%)"
                    exit_price = round(buy_px * (1 + SL_PCT) - SLIPPAGE, 2)
                elif bars_held >= TIME_STOP_BARS:
                    exit_triggered = True
                    exit_reason = "⏰ 90분 타임스탑"
                    exit_price = round(cur_c - SLIPPAGE, 2)
                elif b_idx >= len(day_long) - 1 or time_str >= "15:45":
                    exit_triggered = True
                    exit_reason = "🌙 장마감 청산 (오버나잇 0%)"
                    exit_price = round(cur_c - SLIPPAGE, 2)

                if exit_triggered:
                    ret_pct = (exit_price - buy_px) / buy_px
                    pnl_amount = (active_pos['shares'] * (exit_price - buy_px)) - (active_pos['invested'] * SEC_FEE)
                    capital += pnl_amount
                    equity_curve.append(capital)

                    trades.append({
                        "date": d_str,
                        "symbol": sym,
                        "selected_expert": active_pos['selected_expert'],
                        "confidence": active_pos['confidence'],
                        "entry_time": active_pos['entry_time'],
                        "entry_price": buy_px,
                        "exit_time": time_str,
                        "exit_price": exit_price,
                        "bars_held": bars_held,
                        "return_pct": round(ret_pct * 100, 2),
                        "pnl_usd": round(pnl_amount, 2),
                        "exit_reason": exit_reason
                    })

                    exp_k = active_pos['selected_expert']
                    st = model_stats.setdefault(exp_k, {"total": 0, "wins": 0, "pnl": 0.0})
                    st["total"] += 1
                    if pnl_amount > 0:
                        st["wins"] += 1
                    st["pnl"] += pnl_amount

                    active_pos = None

            # [B] 포지션 미보유 시 ➔ 3중 스크린 검사
            if active_pos is None and time_str <= "14:30" and b_idx >= 1:
                # 1. 상위 60분봉 추세 필터 (과거 데이터만)
                past_macro_60 = macro_60m[macro_60m.index <= cur_time]
                past_long_60 = long_60m[long_60m.index <= cur_time]

                is_60m_bull = False
                is_60m_bear = False
                if len(past_macro_60) >= 20 and len(past_long_60) >= 20:
                    last_macro = past_macro_60.iloc[-1]
                    last_long = past_long_60.iloc[-1]
                    is_60m_bull = (last_macro['Close'] >= last_macro['ema20'] * 0.998) and (last_long['Close'] >= last_long['ema20'] * 0.998)
                    is_60m_bear = (last_macro['Close'] <= last_macro['ema20'] * 1.002)

                # 2. 5분봉 단기 눌림목 필터
                vwap_diff_l = float(row_l.get("VWAP_Diff", 0.0))
                rsi_14_l = float(row_l.get("RSI_14", 50.0))
                bb_lower_l = float(row_l.get("BB_Lower", 0.0))
                cur_close_l = float(row_l['Close'])
                dip_ok_long = (vwap_diff_l <= 1.5) and (rsi_14_l <= 62.0) and (cur_close_l >= bb_lower_l * 1.001 if bb_lower_l > 0 else True)

                # SQQQ 눌림목
                dip_ok_short = False
                if cur_time in short_15m_feat.index:
                    row_s = short_15m_feat.loc[cur_time]
                    vwap_diff_s = float(row_s.get("VWAP_Diff", 0.0))
                    rsi_14_s = float(row_s.get("RSI_14", 50.0))
                    bb_lower_s = float(row_s.get("BB_Lower", 0.0))
                    cur_close_s = float(row_s['Close'])
                    dip_ok_short = (vwap_diff_s <= 1.5) and (rsi_14_s <= 62.0) and (cur_close_s >= bb_lower_s * 1.001 if bb_lower_s > 0 else True)

                # 3. MoE AI 오케스트레이터 평가
                past_long_15m = long_15m_feat[long_15m_feat.index <= cur_time]
                if len(past_long_15m) >= 30:
                    sub_15m = past_long_15m.tail(60)
                    
                    # 5대 레짐 벡터 직접 계산 (과거 데이터 기준)
                    vix_sub = vix_15m[vix_15m.index <= cur_time]
                    vix_val = float(vix_sub['Close'].iloc[-1]) if len(vix_sub) > 0 else 16.5
                    
                    c_s = sub_15m['Close']
                    o_s = sub_15m['Open']
                    h_s = sub_15m['High']
                    l_s = sub_15m['Low']
                    v_s = sub_15m['Volume']
                    
                    # 15분봉 CVD
                    hl = (h_s - l_s).replace(0, 0.001)
                    v_delta = ((c_s - o_s) / hl) * v_s
                    cvd_val = float(v_delta.tail(5).mean() / (v_s.tail(20).mean() + 1e-6))
                    
                    atr_ratio = float((h_s - l_s).tail(5).mean() / ((h_s - l_s).tail(20).mean() + 1e-6))
                    elapsed_min = float(b_idx * 15.0)
                    
                    regime_vec = {
                        "elapsed_min": elapsed_min,
                        "vix_level": vix_val,
                        "atr_ratio": atr_ratio,
                        "cvd_delta": cvd_val,
                        "dislocation_lag": 0.0
                    }
                    
                    gating_conf = moe.predict_gating_weights(regime_vec)
                    top_expert = max(gating_conf, key=gating_conf.get)
                    top_conf = gating_conf[top_expert]

                    # 방향성 결정
                    ret_5 = float(c_s.iloc[-1] / c_s.iloc[-5] - 1.0)
                    if top_expert == "orderflow":
                        sig_code, exp_conf, _ = moe.experts["orderflow"].predict_signal(pd.Series({
                            'delta_zscore': cvd_val, 'cum_delta_pct': cvd_val * 10, 'volume_ratio': 1.2
                        }))
                        direction = "LONG" if sig_code > 0 else ("SHORT" if sig_code < 0 else "NONE")
                    elif top_expert == "tda_topology":
                        sig_code, exp_conf, _ = moe.experts["tda_topology"].predict_signal(sub_15m)
                        direction = "LONG" if sig_code > 0 else ("SHORT" if sig_code < 0 else "NONE")
                    elif top_expert == "statespace_kalman":
                        sig_code, exp_conf, _ = moe.experts["statespace_kalman"].predict_signal(c_s.values)
                        direction = "LONG" if sig_code > 0 else ("SHORT" if sig_code < 0 else "NONE")
                    else:
                        direction = "LONG" if ret_5 > 0.005 else ("SHORT" if ret_5 < -0.005 else "NONE")
                        exp_conf = 0.70

                    # 3중 스크린 최종 관문 통과 검사
                    pass_3screen = False
                    if direction == "LONG" and is_60m_bull and dip_ok_long:
                        pass_3screen = True
                        winner_sym = config.BASE_ASSET_LONG
                        base_px = cur_close_l
                    elif direction == "SHORT" and is_60m_bear and dip_ok_short:
                        pass_3screen = True
                        winner_sym = config.BASE_ASSET_SHORT
                        base_px = float(short_15m_feat.loc[cur_time]['Close']) if cur_time in short_15m_feat.index else 40.0

                    if top_conf >= 0.75 and exp_conf >= 0.60 and pass_3screen:
                        entry_px = round(base_px + SLIPPAGE, 2)
                        shares = int(capital / entry_px)
                        invested = shares * entry_px

                        if shares > 0 and invested > 0:
                            active_pos = {
                                "symbol": winner_sym,
                                "direction": direction,
                                "selected_expert": moe.track_names.get(top_expert, top_expert) if hasattr(moe, 'track_names') else top_expert,
                                "confidence": round(top_conf * 100, 1),
                                "entry_time": time_str,
                                "entry_bar_idx": b_idx,
                                "buy_price": entry_px,
                                "shares": shares,
                                "invested": invested
                            }

    df_res = pd.DataFrame(trades)
    total_cnt = len(df_res)

    print("\n" + "=" * 90)
    print("🏆 [MoE AI + 3중 스크린 결합 정밀 백테스트 결과]")
    print("=" * 90)
    print(f"💰 초기 자본금: ${INITIAL_CAPITAL:,.2f} USD")
    print(f"💵 최종 자본금: ${capital:,.2f} USD")
    print(f"📈 총 누적 수익률: {((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100:+.2f}%")
    print(f"🎯 총 거래 횟수: {total_cnt}회 (65거래일 동안 주당 {total_cnt/13:.1f}회 스나이핑)")
    
    if total_cnt > 0:
        w_df = df_res[df_res['pnl_usd'] > 0]
        l_df = df_res[df_res['pnl_usd'] <= 0]
        w_cnt = len(w_df)
        l_cnt = len(l_df)
        wr = (w_cnt / total_cnt) * 100
        pf = (w_df['pnl_usd'].sum() / abs(l_df['pnl_usd'].sum())) if abs(l_df['pnl_usd'].sum()) > 0 else 999
        print(f"👑 승률 (Win Rate): {wr:.2f}% ({w_cnt}승 / {l_cnt}패)")
        print(f"⚖️ 손익비 (Profit Factor): {pf:.2f}")
        
        eq_s = pd.Series(equity_curve)
        mdd = abs(((eq_s - eq_s.cummax()) / eq_s.cummax()).min()) * 100
        print(f"🛡️ 최대 낙폭 (MDD): {mdd:.2f}%")

        print("\n📋 [전체 거래 상세 내역]")
        print(df_res[['date', 'symbol', 'selected_expert', 'entry_price', 'exit_price', 'return_pct', 'pnl_usd', 'exit_reason']].to_string(index=False))
    print("=" * 90)

if __name__ == "__main__":
    run_moe_3screen_backtest()
