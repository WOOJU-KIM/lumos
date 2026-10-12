import os
import sys
import sqlite3
import json
import config
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.heterogeneous_models import CrossAssetDislocationModel
from core.ml_engine import MLFeatureEngine
from config import DATA_DIR

def run_comprehensive_dual_backtest():
    print("=" * 105)
    print("🏆 [Lumos 정예 듀얼 챔피언: 개별 2종 및 융합 모델 전 기간 백테스팅]")
    print(f"⏰ 실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S KST')}")
    print("=" * 105)

    db_path = DATA_DIR / "market_data.db"
    conn = sqlite3.connect(db_path)

    long_15m = pd.read_sql_query("SELECT datetime, open, high, low, close, volume FROM market_candles WHERE symbol=config.BASE_ASSET_LONG AND timeframe='15m' ORDER BY datetime ASC", conn)
    short_15m = pd.read_sql_query("SELECT datetime, open, high, low, close, volume FROM market_candles WHERE symbol=config.BASE_ASSET_SHORT AND timeframe='15m' ORDER BY datetime ASC", conn)
    macro_15m = pd.read_sql_query("SELECT datetime, open, high, low, close, volume FROM market_candles WHERE symbol=config.MACRO_TREND_SYMBOL AND timeframe='15m' ORDER BY datetime ASC", conn)
    macro_60m = pd.read_sql_query("SELECT datetime, open, high, low, close, volume FROM market_candles WHERE symbol=config.MACRO_TREND_SYMBOL AND timeframe='60m' ORDER BY datetime ASC", conn)
    long_60m = pd.read_sql_query("SELECT datetime, open, high, low, close, volume FROM market_candles WHERE symbol=config.BASE_ASSET_LONG AND timeframe='60m' ORDER BY datetime ASC", conn)
    vix_15m  = pd.read_sql_query("SELECT datetime, open, high, low, close, volume FROM market_candles WHERE symbol='^VIX' AND timeframe='15m' ORDER BY datetime ASC", conn)
    nvda_15m = pd.read_sql_query("SELECT datetime, open, high, low, close, volume FROM market_candles WHERE symbol='NVDA' AND timeframe='15m' ORDER BY datetime ASC", conn)
    qqq_15m  = pd.read_sql_query("SELECT datetime, open, high, low, close, volume FROM market_candles WHERE symbol='QQQ' AND timeframe='15m' ORDER BY datetime ASC", conn)
    conn.close()

    for df in [long_15m, short_15m, macro_15m, macro_60m, long_60m, vix_15m, nvda_15m, qqq_15m]:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        df.set_index('datetime', inplace=True)

    macro_60m['ema20'] = macro_60m['Close'].ewm(span=20, adjust=False).mean()
    long_60m['ema20'] = long_60m['Close'].ewm(span=20, adjust=False).mean()

    ml_engine = MLFeatureEngine(confidence_threshold=0.40)
    long_15m_feat = ml_engine.extract_features(long_15m)
    short_15m_feat = ml_engine.extract_features(short_15m)
    long_15m_feat['date_str'] = long_15m_feat.index.strftime('%Y-%m-%d')
    unique_dates = sorted(long_15m_feat['date_str'].unique())

    cross_mod = CrossAssetDislocationModel(dislocation_z_threshold=1.6)

    INITIAL_CAPITAL = 100_000.0
    TP_PCT = 0.035
    SL_PCT = -0.020
    TIME_STOP_BARS = 6
    SLIPPAGE = 0.03
    SEC_FEE = 0.0001

    models = [
        ("Model A: 크로스에셋 인과 괴리 단독", "cross"),
        ("Model B: 3중 타임프레임 GBDT 파형 단독", "gbdt"),
        ("Model C: [정예 융합] 크로스에셋 + GBDT 듀얼 챔피언", "dual")
    ]

    all_results = {}
    trade_logs_dict = {}

    for label, mode in models:
        capital = INITIAL_CAPITAL
        trades = []
        equity_curve = [capital]
        exit_reasons = {}

        for d_str in unique_dates:
            day_long = long_15m_feat[long_15m_feat['date_str'] == d_str]
            if len(day_long) < 5:
                continue

            active_pos = None

            for b_idx in range(len(day_long)):
                cur_time = day_long.index[b_idx]
                row_l = day_long.iloc[b_idx]
                time_str = cur_time.strftime('%H:%M')

                # [A] 보유 포지션 청산 감시
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
                    exit_price = 0.0
                    reason_str = ""

                    if max_ret >= TP_PCT:
                        exit_triggered = True
                        exit_price = round(buy_px * (1 + TP_PCT) - SLIPPAGE, 2)
                        reason_str = "🎯 목표익절 (+3.5%)"
                    elif min_ret <= SL_PCT:
                        exit_triggered = True
                        exit_price = round(buy_px * (1 + SL_PCT) - SLIPPAGE, 2)
                        reason_str = "🛑 칼손절 (-2.0%)"
                    elif bars_held >= TIME_STOP_BARS:
                        exit_triggered = True
                        exit_price = round(cur_c - SLIPPAGE, 2)
                        reason_str = "⏰ 90분 타임스탑"
                    elif b_idx >= len(day_long) - 1 or time_str >= "15:45":
                        exit_triggered = True
                        exit_price = round(cur_c - SLIPPAGE, 2)
                        reason_str = "🌙 장마감 청산 (오버나잇 0%)"

                    if exit_triggered:
                        ret_pct = (exit_price - buy_px) / buy_px
                        pnl_amount = (active_pos['shares'] * (exit_price - buy_px)) - (active_pos['invested'] * SEC_FEE)
                        capital += pnl_amount
                        equity_curve.append(capital)

                        trades.append({
                            "date": d_str,
                            "time": time_str,
                            "symbol": sym,
                            "direction": active_pos['direction'],
                            "trigger_model": active_pos['trigger_model'],
                            "entry_price": buy_px,
                            "exit_price": exit_price,
                            "return_pct": round(ret_pct * 100, 2),
                            "pnl_usd": round(pnl_amount, 2),
                            "capital_after": round(capital, 2),
                            "exit_reason": reason_str
                        })
                        exit_reasons[reason_str] = exit_reasons.get(reason_str, 0) + 1
                        active_pos = None

                # [B] 포지션 미보유 시 ➔ 3중 스크린 검사
                if active_pos is None and time_str <= "14:30" and b_idx >= 1:
                    past_macro_60 = macro_60m[macro_60m.index <= cur_time]
                    past_long_60 = long_60m[long_60m.index <= cur_time]

                    is_60m_bull = False
                    is_60m_bear = False
                    if len(past_macro_60) >= 20 and len(past_long_60) >= 20:
                        last_macro = past_macro_60.iloc[-1]
                        last_long = past_long_60.iloc[-1]
                        is_60m_bull = (last_macro['Close'] >= last_macro['ema20'] * 0.998) and (last_long['Close'] >= last_long['ema20'] * 0.998)
                        is_60m_bear = (last_macro['Close'] <= last_macro['ema20'] * 1.002)

                    vwap_diff_l = float(row_l.get("VWAP_Diff", 0.0))
                    rsi_14_l = float(row_l.get("RSI_14", 50.0))
                    bb_lower_l = float(row_l.get("BB_Lower", 0.0))
                    cur_close_l = float(row_l['Close'])
                    dip_ok_long = (vwap_diff_l <= 1.5) and (rsi_14_l <= 62.0) and (cur_close_l >= bb_lower_l * 1.001 if bb_lower_l > 0 else True)

                    dip_ok_short = False
                    if cur_time in short_15m_feat.index:
                        row_s = short_15m_feat.loc[cur_time]
                        vwap_diff_s = float(row_s.get("VWAP_Diff", 0.0))
                        rsi_14_s = float(row_s.get("RSI_14", 50.0))
                        bb_lower_s = float(row_s.get("BB_Lower", 0.0))
                        cur_close_s = float(row_s['Close'])
                        dip_ok_short = (vwap_diff_s <= 1.5) and (rsi_14_s <= 62.0) and (cur_close_s >= bb_lower_s * 1.001 if bb_lower_s > 0 else True)

                    past_long_15m = long_15m_feat[long_15m_feat.index <= cur_time]
                    if len(past_long_15m) >= 30:
                        sub_15m = past_long_15m.tail(60)
                        ret_5 = float(sub_15m['Close'].iloc[-1] / sub_15m['Close'].iloc[-5] - 1.0)

                        dir_cross = "NONE"
                        past_nvda = nvda_15m[nvda_15m.index <= cur_time]
                        past_macro = macro_15m[macro_15m.index <= cur_time]
                        past_qqq = qqq_15m[qqq_15m.index <= cur_time]
                        past_vix = vix_15m[vix_15m.index <= cur_time]
                        if len(past_nvda) >= 5 and len(past_qqq) >= 5 and len(past_vix) >= 5:
                            nvda_r = float(past_nvda['Close'].iloc[-1] / past_nvda['Close'].iloc[-5] - 1.0)
                            macro_r = float(past_macro['Close'].iloc[-1] / past_macro['Close'].iloc[-5] - 1.0) if len(past_macro) >= 5 else nvda_r
                            qqq_r = float(past_qqq['Close'].iloc[-1] / past_qqq['Close'].iloc[-5] - 1.0)
                            vix_r = float(past_vix['Close'].iloc[-1] / past_vix['Close'].iloc[-5] - 1.0)
                            long_r = ret_5
                            sig_code, _, _ = cross_mod.predict_signal(
                                long_ret=long_r,
                                nvda_ret=nvda_r,
                                macro_ret=macro_r,
                                qqq_ret=qqq_r,
                                vix_ret=vix_r,
                                tnx_ret=0.0
                            )
                            dir_cross = "LONG" if sig_code > 0 else ("SHORT" if sig_code < 0 else "NONE")

                        dir_gbdt = "NONE"
                        conf_l = float(row_l.get('Confidence', 0.50))
                        conf_s = float(short_15m_feat.loc[cur_time].get('Confidence', 0.50)) if cur_time in short_15m_feat.index else 0.50
                        if conf_l >= 0.40:
                            dir_gbdt = "LONG"
                        elif conf_s >= 0.40:
                            dir_gbdt = "SHORT"

                        final_dir = "NONE"
                        trig_mod = ""
                        if mode == "cross":
                            final_dir = dir_cross
                            trig_mod = "크로스에셋"
                        elif mode == "gbdt":
                            final_dir = dir_gbdt
                            trig_mod = "GBDT파형"
                        elif mode == "dual":
                            if dir_cross != "NONE":
                                final_dir = dir_cross
                                trig_mod = "크로스에셋"
                            elif dir_gbdt != "NONE":
                                final_dir = dir_gbdt
                                trig_mod = "GBDT파형"

                        pass_3screen = False
                        if final_dir == "LONG" and is_60m_bull and dip_ok_long:
                            pass_3screen = True
                            winner_sym = config.BASE_ASSET_LONG
                            base_px = cur_close_l
                        elif final_dir == "SHORT" and is_60m_bear and dip_ok_short:
                            pass_3screen = True
                            winner_sym = config.BASE_ASSET_SHORT
                            base_px = float(short_15m_feat.loc[cur_time]['Close']) if cur_time in short_15m_feat.index else 40.0

                        if pass_3screen:
                            entry_px = round(base_px + SLIPPAGE, 2)
                            shares = int(capital / entry_px)
                            invested = shares * entry_px

                            if shares > 0 and invested > 0:
                                active_pos = {
                                    "symbol": winner_sym,
                                    "direction": final_dir,
                                    "trigger_model": trig_mod,
                                    "entry_bar_idx": b_idx,
                                    "buy_price": entry_px,
                                    "shares": shares,
                                    "invested": invested
                                }

        df_t = pd.DataFrame(trades)
        t_cnt = len(df_t)
        if t_cnt > 0:
            w_df = df_t[df_t['pnl_usd'] > 0]
            l_df = df_t[df_t['pnl_usd'] <= 0]
            w_cnt = len(w_df)
            l_cnt = len(l_df)
            wr = (w_cnt / t_cnt) * 100
            ret = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
            tot_w = w_df['pnl_usd'].sum()
            tot_l = abs(l_df['pnl_usd'].sum())
            pf = (tot_w / tot_l) if tot_l > 0 else 999.0
            eq_s = pd.Series(equity_curve)
            mdd = abs(((eq_s - eq_s.cummax()) / eq_s.cummax()).min()) * 100
            long_trades = df_t[df_t['symbol'] == config.BASE_ASSET_LONG]
            short_trades = df_t[df_t['symbol'] == config.BASE_ASSET_SHORT]
            long_wr = (len(long_trades[long_trades['pnl_usd'] > 0]) / len(long_trades) * 100) if len(long_trades) > 0 else 0
            short_wr = (len(short_trades[short_trades['pnl_usd'] > 0]) / len(short_trades) * 100) if len(short_trades) > 0 else 0
        else:
            w_cnt = l_cnt = long_wr = short_wr = 0
            wr = ret = pf = mdd = 0.0

        all_results[label] = {
            "trades_count": t_cnt,
            "wins": w_cnt,
            "losses": l_cnt,
            "win_rate_pct": round(wr, 2),
            "long_win_rate_pct": round(long_wr, 2),
            "short_win_rate_pct": round(short_wr, 2),
            "profit_factor": round(pf, 2),
            "total_return_pct": round(ret, 2),
            "net_pnl_usd": round(capital - INITIAL_CAPITAL, 2),
            "final_capital_usd": round(capital, 2),
            "mdd_pct": round(mdd, 2),
            "exit_reasons": exit_reasons
        }
        trade_logs_dict[label] = df_t

    # =========================================================================
    # 최종 리포트 출력
    # =========================================================================
    print("=" * 105)
    print("📊 [3대 전략 전 기간 백테스트 비교 요약표 (2026-05-20 ~ 2026-08-21 / 65거래일)]")
    print("=" * 105)
    
    summary_rows = []
    for k, v in all_results.items():
        summary_rows.append({
            "전략 명칭": k,
            "총 거래수": f"{v['trades_count']}회",
            "승률 (전체)": f"{v['win_rate_pct']:.2f}% ({v['wins']}승/{v['losses']}패)",
            "TQQQ 승률": f"{v['long_win_rate_pct']:.1f}%",
            "SQQQ 승률": f"{v['short_win_rate_pct']:.1f}%",
            "손익비 (PF)": f"{v['profit_factor']:.2f}",
            "총 수익률": f"{v['total_return_pct']:+.2f}%",
            "순익 (USD)": f"${v['net_pnl_usd']:+,.2f}",
            "MDD": f"{v['mdd_pct']:.2f}%"
        })
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print("=" * 105)

    print("\n🎯 [청산 사유별 상세 통계]")
    for k, v in all_results.items():
        print(f"\n▶ [{k}]")
        for r_name, count in v['exit_reasons'].items():
            pct = (count / v['trades_count']) * 100 if v['trades_count'] > 0 else 0
            print(f"   • {r_name}: {count}회 ({pct:.1f}%)")

    # CSV 파일 저장
    for k, df_log in trade_logs_dict.items():
        clean_name = k.split(":")[0].replace(" ", "_").lower()
        df_log.to_csv(DATA_DIR / f"{clean_name}_trades.csv", index=False, encoding="utf-8-sig")

    with open(DATA_DIR / "dual_champion_all_periods_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n💾 상세 거래 로그 파일들이 '{DATA_DIR}' 디렉토리에 저장되었습니다.")

if __name__ == "__main__":
    run_comprehensive_dual_backtest()
