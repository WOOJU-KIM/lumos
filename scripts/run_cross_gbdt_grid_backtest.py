import os
import sys
import json
import sqlite3
import config
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.data_lake import MarketDataLake
from core.heterogeneous_models import CrossAssetDislocationModel
from core.ml_engine import MLFeatureEngine

def format_markdown_table(headers: List[str], rows: List[List[Any]], aligns: List[str] = None) -> str:
    if aligns is None:
        aligns = [":---:" for _ in headers]
    col_widths = [len(h) for h in headers]
    for r in rows:
        for i, val in enumerate(r):
            col_widths[i] = max(col_widths[i], len(str(val)))
    header_line = "| " + " | ".join(h.center(col_widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join(
        (":" + "-" * (col_widths[i] - 2) + ":") if aligns[i] == ":---:"
        else (":" + "-" * (col_widths[i] - 1)) if aligns[i] == ":---"
        else ("-" * (col_widths[i] - 1) + ":") for i in range(len(headers))
    ) + " |"
    body_lines = []
    for r in rows:
        row_str = "| " + " | ".join(str(val).center(col_widths[i]) if aligns[i] == ":---:" else str(val).rjust(col_widths[i]) if aligns[i] == "---:" else str(val).ljust(col_widths[i]) for i, val in enumerate(r)) + " |"
        body_lines.append(row_str)
    return "\n".join([header_line, sep_line] + body_lines)

def run_grid_search():
    lake = MarketDataLake()
    print("=" * 110)
    print("🏛 [Lumos 퀀트 시스템: GBDT x 크로스에셋 확신도 전구간(50%~80%) 2D 그리드 백테스트]")
    print(f"⏰ 실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S KST')}")
    print("=" * 110)

    # 1. 데이터 로드
    long_15m = lake.load_candles(config.BASE_ASSET_LONG, "15m")
    short_15m = lake.load_candles(config.BASE_ASSET_SHORT, "15m")
    long_5m  = lake.load_candles(config.BASE_ASSET_LONG, "5m")
    short_5m  = lake.load_candles(config.BASE_ASSET_SHORT, "5m")
    macro_60m = lake.load_candles(config.MACRO_TREND_SYMBOL, "60m")
    long_60m = lake.load_candles(config.BASE_ASSET_LONG, "60m")
    macro_15m = lake.load_candles(config.MACRO_TREND_SYMBOL, "15m")
    nvda_15m = lake.load_candles("NVDA", "15m")
    qqq_15m  = lake.load_candles("QQQ", "15m")
    vix_15m  = lake.load_candles("^VIX", "15m")

    macro_60m['ema20'] = macro_60m['Close'].ewm(span=20, adjust=False).mean()
    long_60m['ema20'] = long_60m['Close'].ewm(span=20, adjust=False).mean()

    for df in [long_15m, short_15m, long_5m, short_5m, macro_60m, long_60m, macro_15m, nvda_15m, qqq_15m, vix_15m]:
        df['datetime_dt'] = pd.to_datetime(df['datetime'])
        df['date_str'] = df['datetime_dt'].dt.strftime('%Y-%m-%d')
        df['time_str'] = df['datetime_dt'].dt.strftime('%H:%M')

    ml_15m = MLFeatureEngine()
    long_15m_feat = ml_15m.extract_features(long_15m)
    long_15m_feat = ml_15m.add_confidence_columns(long_15m_feat)
    short_15m_feat = ml_15m.extract_features(short_15m)
    short_15m_feat.set_index('datetime', inplace=True, drop=False)

    cross_mod = CrossAssetDislocationModel(dislocation_z_threshold=1.6)
    nvda_map = nvda_15m.set_index('datetime')['Close'].to_dict()
    macro_map = macro_15m.set_index('datetime')['Close'].to_dict()
    qqq_map = qqq_15m.set_index('datetime')['Close'].to_dict()
    vix_map = vix_15m.set_index('datetime')['Close'].to_dict()
    long_close = long_15m['Close'].values
    long_dt = long_15m['datetime'].values

    cross_dirs = []
    cross_confs = []
    for i in range(len(long_15m)):
        if i < 5:
            cross_dirs.append('NONE'); cross_confs.append(0.50); continue
        c_t, p_t = long_dt[i], long_dt[i-5]
        if c_t in nvda_map and p_t in nvda_map and c_t in qqq_map and p_t in qqq_map and c_t in vix_map and p_t in vix_map:
            s_r = float(long_close[i]/long_close[i-5] - 1.0)
            n_r = float(nvda_map[c_t]/nvda_map[p_t] - 1.0)
            sx_r = float(macro_map[c_t]/macro_map[p_t] - 1.0) if c_t in macro_map and p_t in macro_map else n_r
            q_r = float(qqq_map[c_t]/qqq_map[p_t] - 1.0)
            v_r = float(vix_map[c_t]/vix_map[p_t] - 1.0)
            code, conf, desc = cross_mod.predict_signal(long_ret=s_r, nvda_ret=n_r, macro_ret=sx_r, qqq_ret=q_r, vix_ret=v_r, tnx_ret=0.0)
            c_dir = "LONG" if code > 0 else ("SHORT" if code < 0 else 'NONE')
            cross_dirs.append(c_dir)
            cross_confs.append(conf)
        else:
            cross_dirs.append('NONE'); cross_confs.append(0.50)

    long_15m_feat['cross_dir'] = cross_dirs
    long_15m_feat['cross_conf'] = cross_confs
    long_15m_feat.set_index('datetime', inplace=True, drop=False)

    unique_dates = sorted(long_15m['date_str'].unique())
    num_days = len(unique_dates)
    total_weeks = num_days / 5.0

    TP_15M = 0.030
    SL_15M = -0.020
    SLIPPAGE = 0.03
    FEE_RATE = 0.0020
    TIME_STOP_BARS = 18

    levels = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    def run_sim(g_th: float, c_th: float, mode: str = "consensus"):
        capital = 10000.0
        trades = []
        equity_curve = [capital]
        daily_stoploss_count = 0

        for d_str in unique_dates:
            day_15 = long_15m_feat[long_15m_feat['date_str'] == d_str]
            day_5_l = long_5m[long_5m['date_str'] == d_str]
            day_5_s = short_5m[short_5m['date_str'] == d_str]
            if len(day_15) < 5: continue

            daily_stoploss_count = 0
            b_idx = 0
            n_bars = len(day_15)

            while b_idx < n_bars:
                row_15 = day_15.iloc[b_idx]
                cur_time = row_15['datetime']
                time_str = row_15['time_str']

                if b_idx < 1 or time_str > '14:30' or daily_stoploss_count >= 3:
                    b_idx += 1; continue

                dir_gbdt = row_15['Direction']
                conf_gbdt = float(row_15['Confidence'])
                dir_cross = row_15['cross_dir']
                conf_cross = float(row_15['cross_conf'])

                chosen_sym = None
                if mode == "consensus":
                    # [듀얼 합의 AND 로직] 두 모델이 같은 방향 제시 + GBDT>=G AND 크로스에셋>=C
                    if dir_gbdt == "LONG" and dir_cross == "LONG" and conf_gbdt >= g_th and conf_cross >= c_th:
                        chosen_sym = config.BASE_ASSET_LONG
                    elif dir_gbdt == "SHORT" and dir_cross == "SHORT" and conf_gbdt >= g_th and conf_cross >= c_th:
                        chosen_sym = config.BASE_ASSET_SHORT
                elif mode == "hybrid_gated":
                    # [하이브리드 MoE 점수 게이팅] GBDT>=G, 크로스에셋 반대 Veto 차단 (반대 확신도>=C일 때 Veto), 동방향 시 확인
                    is_gbdt = (dir_gbdt in ["LONG", "SHORT"]) and (conf_gbdt >= g_th)
                    is_opposite_veto = (
                        (dir_gbdt == "LONG" and dir_cross == "SHORT" and conf_cross >= c_th) or
                        (dir_gbdt == "SHORT" and dir_cross == "LONG" and conf_cross >= c_th)
                    )
                    if is_gbdt and not is_opposite_veto:
                        chosen_sym = config.BASE_ASSET_LONG if dir_gbdt == "LONG" else config.BASE_ASSET_SHORT

                if not chosen_sym:
                    b_idx += 1; continue

                # Screen 1: 60m trend
                past_macro = macro_60m[macro_60m['datetime'] <= cur_time]
                past_long = long_60m[long_60m['datetime'] <= cur_time]
                if len(past_macro) < 20 or len(past_long) < 20:
                    b_idx += 1; continue
                macro_c = past_macro['Close'].iloc[-1]
                long_c = past_long['Close'].iloc[-1]
                macro_ema = past_macro['ema20'].iloc[-1]
                long_ema = past_long['ema20'].iloc[-1]
                if chosen_sym == config.BASE_ASSET_LONG and not (macro_c >= macro_ema * 0.998 and long_c >= long_ema * 0.998):
                    b_idx += 1; continue
                elif chosen_sym == config.BASE_ASSET_SHORT and not (macro_c <= macro_ema * 1.002):
                    b_idx += 1; continue

                # Screen 3: Dip filter
                if chosen_sym == config.BASE_ASSET_LONG:
                    vd = float(row_15.get('VWAP_Diff', 0.0))
                    r14 = float(row_15.get('RSI_14', 50.0))
                    bbl = float(row_15.get('BB_Lower', 0.0))
                    if not (vd <= 1.5 and r14 <= 62.0 and (bbl <= 0 or float(row_15['Close']) >= bbl * 1.001)):
                        b_idx += 1; continue
                else:
                    if cur_time not in short_15m_feat.index:
                        b_idx += 1; continue
                    rs = short_15m_feat.loc[cur_time]
                    vd = float(rs.get('VWAP_Diff', 0.0))
                    r14 = float(rs.get('RSI_14', 50.0))
                    bbl = float(rs.get('BB_Lower', 0.0))
                    if not (vd <= 1.5 and r14 <= 62.0 and (bbl <= 0 or float(rs['Close']) >= bbl * 1.001)):
                        b_idx += 1; continue

                base_px = float(row_15['Close']) if chosen_sym == config.BASE_ASSET_LONG else float(short_15m_feat.loc[cur_time]['Close'])
                entry_px = round(base_px + SLIPPAGE, 2)
                shares = int(capital / entry_px)
                invested = shares * entry_px
                if shares <= 0: b_idx += 1; continue

                target_5m = day_5_l if chosen_sym == config.BASE_ASSET_LONG else day_5_s
                post_5m = target_5m[target_5m['datetime'] > cur_time]
                if post_5m.empty: b_idx += 1; continue

                tp_px = round(entry_px * (1 + TP_15M), 2)
                sl_px = round(entry_px * (1 + SL_15M), 2)
                exit_px = entry_px
                bars_held = 0

                eval_5m = post_5m.iloc[:TIME_STOP_BARS]
                for k in range(len(eval_5m)):
                    c5 = eval_5m.iloc[k]
                    t5 = c5['time_str']
                    c5_o, c5_h, c5_l, c5_c = float(c5['Open']), float(c5['High']), float(c5['Low']), float(c5['Close'])
                    bars_held = k + 1
                    hit_tp, hit_sl = (c5_h >= tp_px), (c5_l <= sl_px)

                    if hit_tp and hit_sl:
                        if c5_c >= c5_o:
                            exit_px = round(tp_px - SLIPPAGE, 2)
                            break
                        else:
                            exit_px = round(sl_px - SLIPPAGE, 2)
                            daily_stoploss_count += 1
                            break
                    elif hit_tp:
                        exit_px = round(tp_px - SLIPPAGE, 2)
                        break
                    elif hit_sl:
                        exit_px = round(sl_px - SLIPPAGE, 2)
                        daily_stoploss_count += 1
                        break
                    elif t5 >= '15:45' or k == len(eval_5m) - 1:
                        exit_px = round(c5_c - SLIPPAGE, 2)
                        if (exit_px - entry_px) / entry_px <= -0.020:
                            daily_stoploss_count += 1
                        break

                cost = invested * FEE_RATE
                net_pnl = (shares * (exit_px - entry_px)) - cost
                capital += net_pnl
                equity_curve.append(capital)

                ret = net_pnl / invested
                trades.append({
                    "symbol": chosen_sym,
                    "ret": ret,
                    "pnl": net_pnl
                })

                consumed = int(np.ceil(bars_held / 3.0))
                b_idx += max(1, consumed)

        tot_tr = len(trades)
        w_tr = [t for t in trades if t['ret'] > 0]
        l_tr = [t for t in trades if t['ret'] <= 0]
        wr = (len(w_tr) / tot_tr * 100) if tot_tr > 0 else 0.0
        gross_win = sum([t['ret'] for t in w_tr])
        gross_loss = abs(sum([t['ret'] for t in l_tr])) if l_tr else 0.0001
        pf = (gross_win / gross_loss) if gross_loss > 0 else (99.9 if gross_win > 0 else 0.0)
        cum_ret = ((capital - 10000.0) / 10000.0) * 100.0

        eq_arr = np.array(equity_curve)
        pk = np.maximum.accumulate(eq_arr)
        dd = (pk - eq_arr) / pk * 100.0
        mdd = np.max(dd) if len(dd) > 0 else 0.0

        long_tr = [t for t in trades if t['symbol'] == config.BASE_ASSET_LONG]
        short_tr = [t for t in trades if t['symbol'] == config.BASE_ASSET_SHORT]
        long_w = [t for t in long_tr if t['ret'] > 0]
        short_w = [t for t in short_tr if t['ret'] > 0]
        long_wr = (len(long_w) / len(long_tr) * 100) if long_tr else 0.0
        short_wr = (len(short_w) / len(short_tr) * 100) if short_tr else 0.0

        return {
            "g_th": f"{int(g_th*100)}%",
            "c_th": f"{int(c_th*100)}%",
            "total_trades": tot_tr,
            "trades_per_week": round(tot_tr / total_weeks, 2),
            "wins": len(w_tr),
            "losses": len(l_tr),
            "win_rate": wr,
            "long_trades": len(long_tr),
            "long_wr": long_wr,
            "short_trades": len(short_tr),
            "short_wr": short_wr,
            "cum_return": cum_ret,
            "profit_factor": pf,
            "mdd": mdd,
            "final_capital": capital
        }

    # =========================================================================
    # [1] 1:1 대각선 매칭 (G_th == C_th: 50%-50%, 55%-55%, ..., 80%-80%)
    # =========================================================================
    print("\n⏳ [1/3] 1:1 대칭 매칭 (GBDT = Cross-Asset) 7대 구간 백테스트...")
    diag_consensus = [run_sim(th, th, mode="consensus") for th in levels]
    diag_hybrid = [run_sim(th, th, mode="hybrid_gated") for th in levels]

    # =========================================================================
    # [2] 7x7 전구간 2D 그리드 (49개 조합) 전수 백테스트
    # =========================================================================
    print("⏳ [2/3] 7x7 전구간 49개 조합 2D 그리드 백테스트 가동 중...")
    grid_consensus = []
    for g in levels:
        row = []
        for c in levels:
            res = run_sim(g, c, mode="consensus")
            row.append(res)
        grid_consensus.append(row)

    grid_hybrid = []
    for g in levels:
        row = []
        for c in levels:
            res = run_sim(g, c, mode="hybrid_gated")
            row.append(res)
        grid_hybrid.append(row)

    print("⏳ [3/3] 결과 표 포맷팅 및 출력 준비 완료!")
    return diag_consensus, diag_hybrid, grid_consensus, grid_hybrid, levels

if __name__ == "__main__":
    diag_c, diag_h, grid_c, grid_h, levels = run_grid_search()

    headers_diag = ["GBDT", "크로스에셋", "총거래", "주당거래", "승 / 패", "승률(%)", "TQQQ(승률)", "SQQQ(승률)", "누적수익률", "손익비(PF)", "MDD(%)", "최종자본금($)"]
    aligns_diag = [":---:", ":---:", ":---:", ":---:", ":---:", ":---:", ":---:", ":---:", "---:", ":---:", ":---:", "---:"]

    print("\n" + "=" * 110)
    print("📊 [1. 듀얼 합의(Dual Consensus AND 로직) 1:1 대칭 매칭 결과표 (GBDT = Cross-Asset)]")
    print("   * 규칙: GBDT와 크로스에셋이 같은 방향으로 동시에 해당 확신도 이상일 때만 승인")
    print("=" * 110)
    rows_dc = []
    for m in diag_c:
        rows_dc.append([
            m["g_th"], m["c_th"], f"{m['total_trades']}회", f"{m['trades_per_week']}회",
            f"{m['wins']}승 {m['losses']}패", f"{m['win_rate']:.1f}%",
            f"{m['long_trades']}건 ({m['long_wr']:.1f}%)", f"{m['short_trades']}건 ({m['short_wr']:.1f}%)",
            f"{m['cum_return']:+.2f}%", f"{m['profit_factor']:.2f}", f"{m['mdd']:.2f}%", f"${m['final_capital']:,.2f}"
        ])
    print(format_markdown_table(headers_diag, rows_dc, aligns_diag))

    print("\n" + "=" * 110)
    print("📊 [2. 하이브리드 MoE (공격수 GBDT + 방패 Cross-Asset Veto) 1:1 대칭 매칭 결과표]")
    print("   * 규칙: GBDT 확신도 >= G_th 진입 시, 크로스에셋 역풍이 C_th 이상일 때만 VETO 차단")
    print("=" * 110)
    rows_dh = []
    for m in diag_h:
        rows_dh.append([
            m["g_th"], m["c_th"], f"{m['total_trades']}회", f"{m['trades_per_week']}회",
            f"{m['wins']}승 {m['losses']}패", f"{m['win_rate']:.1f}%",
            f"{m['long_trades']}건 ({m['long_wr']:.1f}%)", f"{m['short_trades']}건 ({m['short_wr']:.1f}%)",
            f"{m['cum_return']:+.2f}%", f"{m['profit_factor']:.2f}", f"{m['mdd']:.2f}%", f"${m['final_capital']:,.2f}"
        ])
    print(format_markdown_table(headers_diag, rows_dh, aligns_diag))

    # 7x7 그리드 승률 매트릭스 출력
    lvl_names = [f"{int(x*100)}%" for x in levels]
    matrix_headers = ["GBDT \\ Cross"] + [f"C={x}" for x in lvl_names]
    matrix_aligns = [":---:"] + [":---:" for _ in lvl_names]

    print("\n" + "=" * 110)
    print("🗺️ [3. 듀얼 합의(Dual Consensus) 7x7 전구간 승률(%) 매트릭스 (행: GBDT, 열: Cross-Asset)]")
    print("=" * 110)
    grid_c_rows = []
    for i, g_str in enumerate(lvl_names):
        r = [f"G={g_str}"]
        for j in range(len(levels)):
            res = grid_c[i][j]
            if res['total_trades'] > 0:
                r.append(f"{res['win_rate']:.1f}% ({res['total_trades']}회)")
            else:
                r.append("- (0회)")
        grid_c_rows.append(r)
    print(format_markdown_table(matrix_headers, grid_c_rows, matrix_aligns))

    print("\n" + "=" * 110)
    print("🗺️ [4. 듀얼 합의(Dual Consensus) 7x7 전구간 누적 수익률(%) 매트릭스 (행: GBDT, 열: Cross-Asset)]")
    print("=" * 110)
    grid_ret_rows = []
    for i, g_str in enumerate(lvl_names):
        r = [f"G={g_str}"]
        for j in range(len(levels)):
            res = grid_c[i][j]
            if res['total_trades'] > 0:
                r.append(f"{res['cum_return']:+.1f}%")
            else:
                r.append("0.0%")
        grid_ret_rows.append(r)
    print(format_markdown_table(matrix_headers, grid_ret_rows, matrix_aligns))

    print("\n" + "=" * 110)
    print("🗺️ [5. 하이브리드 MoE(공격수 GBDT + Veto 방패) 7x7 전구간 승률(%) 매트릭스 (행: GBDT 진입허들, 열: Veto 확신도허들)]")
    print("=" * 110)
    grid_h_wr_rows = []
    for i, g_str in enumerate(lvl_names):
        r = [f"G={g_str}"]
        for j in range(len(levels)):
            res = grid_h[i][j]
            if res['total_trades'] > 0:
                r.append(f"{res['win_rate']:.1f}% ({res['total_trades']}회)")
            else:
                r.append("- (0회)")
        grid_h_wr_rows.append(r)
    print(format_markdown_table(matrix_headers, grid_h_wr_rows, matrix_aligns))

    print("\n" + "=" * 110)
    print("🗺️ [6. 하이브리드 MoE(공격수 GBDT + Veto 방패) 7x7 전구간 누적 수익률(%) 매트릭스 (행: GBDT 진입허들, 열: Veto 확신도허들)]")
    print("=" * 110)
    grid_h_ret_rows = []
    for i, g_str in enumerate(lvl_names):
        r = [f"G={g_str}"]
        for j in range(len(levels)):
            res = grid_h[i][j]
            if res['total_trades'] > 0:
                r.append(f"{res['cum_return']:+.1f}%")
            else:
                r.append("0.0%")
        grid_h_ret_rows.append(r)
    print(format_markdown_table(matrix_headers, grid_h_ret_rows, matrix_aligns))
    print("=" * 110)

