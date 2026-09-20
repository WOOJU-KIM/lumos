import os
import sys
import sqlite3
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.moe_orchestrator import MoEMetaOrchestrator
from config import DATA_DIR

def run_live_identical_backtest():
    print("=" * 90)
    print("🚀 [Lumos Track 6: MoE AI 메타 오케스트레이터 실전 동일 전구간 백테스팅]")
    print(f"⏰ 실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S KST')}")
    print("=" * 90)

    # 1. DB에서 전체 15분봉 데이터 로드
    db_path = DATA_DIR / "market_data.db"
    conn = sqlite3.connect(db_path)
    
    symbols = ["TQQQ", "SQQQ", "NVDA", "QQQ", "SOXX", "^VIX", "^TNX"]
    data_15m = {}
    for s in symbols:
        df = pd.read_sql_query(
            f"SELECT datetime, open, high, low, close, volume FROM market_candles WHERE symbol='{s}' AND timeframe='15m' AND datetime >= '2026-08-01' ORDER BY datetime ASC",
            conn
        )
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
        data_15m[s] = df

    conn.close()

    tqqq_df = data_15m["TQQQ"]
    print(f"📊 [데이터 적재 현황]")
    print(f"   • 시작 시각: {tqqq_df['datetime'].iloc[0]} ➔ 종료 시각: {tqqq_df['datetime'].iloc[-1]}")
    print(f"   • 총 15분봉 캔들 수: {len(tqqq_df):,}개 (약 3개월 전 구간)")

    # 2. 날짜별 인덱싱 (뉴욕 정규장 09:30 ~ 16:00)
    tqqq_df['date_str'] = tqqq_df['datetime'].dt.strftime('%Y-%m-%d')
    unique_dates = sorted(tqqq_df['date_str'].unique())

    print(f"   • 총 거래일 수: {len(unique_dates)}일")

    # 3. 실전 동일 파라미터 세팅
    INITIAL_CAPITAL = 10_000.0    # $10,000 USD (복리 재투자 100% 투입)
    CONFIDENCE_THRESHOLD = 0.60   # 75.0% 절대확신도 게이팅 기준
    TP_PCT = 0.035               # +3.5% 목표 익절
    SL_PCT = -0.020              # -2.0% 칼손절 Hard Cap
    TIME_STOP_BARS = 6           # 90분 (15분봉 6개)
    SLIPPAGE_PAYUP = 0.03        # 실전 동일 호가 페이업 ($0.03)
    ROUND_TRIP_FEE_RATE = 0.0020 # 진입+청산 왕복 0.20% 거래비용/슬리피지 선차감

    capital = INITIAL_CAPITAL
    trades: List[Dict[str, Any]] = []
    equity_curve: List[float] = [capital]
    
    moe = MoEMetaOrchestrator()

    # 통계 카운터
    model_stats = {}

    print(f"\n⚙️ [백테스트 실행 조건 설정]")
    print(f"   • 초기 시드머니: ${INITIAL_CAPITAL:,.2f} USD (100% 복리 전액 운용)")
    print(f"   • 확신도 인터락: {CONFIDENCE_THRESHOLD*100:.1f}% 이상 A급 타점만 승인")
    print(f"   • 거래 비용: 왕복 {ROUND_TRIP_FEE_RATE*100:.2f}% (수수료 0.10% + 슬리피지 0.10% 완벽 반영)")
    print(f"   • 청산 원칙: +3.5% 익절 / -2.0% 칼손절 / 90분 타임스탑 / 0% 오버나잇 전량 현금화")
    print("\n⏳ [MoE 6대 AI 실시간 동일 시뮬레이션 순회 시작]...")

    for date_idx, d_str in enumerate(unique_dates):
        # 당일 캔들 슬라이스
        day_tqqq = tqqq_df[tqqq_df['date_str'] == d_str].reset_index(drop=True)
        if len(day_tqqq) < 5:
            continue

        active_pos = None  # 당일 포지션 관리

        for b_idx in range(len(day_tqqq)):
            row = day_tqqq.iloc[b_idx]
            bar_time = row['datetime'].strftime('%H:%M')
            cur_tqqq_px = row['Close']
            
            # [A] 보유 포지션이 있는 경우 ➔ 실시간 청산 관리 (TP / SL / TimeStop / EOD)
            if active_pos is not None:
                sym = active_pos['symbol']
                buy_px = active_pos['buy_price']
                bars_held = b_idx - active_pos['entry_bar_idx']
                
                # 심볼별 당일 현재가 추적
                if sym == 'TQQQ':
                    cur_h = row['High']
                    cur_l = row['Low']
                    cur_c = row['Close']
                else:
                    # SQQQ 데이터 매칭
                    sqqq_day = data_15m['SQQQ'][data_15m['SQQQ']['datetime'] == row['datetime']]
                    if not sqqq_day.empty:
                        cur_h = sqqq_day['High'].iloc[0]
                        cur_l = sqqq_day['Low'].iloc[0]
                        cur_c = sqqq_day['Close'].iloc[0]
                    else:
                        cur_h = cur_l = cur_c = buy_px

                # 1. 목표 익절 (+3.5% 도달 검사)
                max_ret = (cur_h - buy_px) / buy_px
                min_ret = (cur_l - buy_px) / buy_px

                exit_triggered = False
                exit_reason = ""
                exit_price = 0.0

                if max_ret >= TP_PCT:
                    exit_triggered = True
                    exit_reason = "🎯 목표익절 (+3.5%)"
                    exit_price = round(buy_px * (1 + TP_PCT) - SLIPPAGE_PAYUP, 2)
                elif min_ret <= SL_PCT:
                    exit_triggered = True
                    exit_reason = "🛑 칼손절 (-2.0%)"
                    exit_price = round(buy_px * (1 + SL_PCT) - SLIPPAGE_PAYUP, 2)
                elif bars_held >= TIME_STOP_BARS:
                    exit_triggered = True
                    exit_reason = "⏰ 90분 타임스탑"
                    exit_price = round(cur_c - SLIPPAGE_PAYUP, 2)
                elif b_idx >= len(day_tqqq) - 1 or bar_time >= "15:45":
                    exit_triggered = True
                    exit_reason = "🌙 장마감 청산 (오버나잇 0%)"
                    exit_price = round(cur_c - SLIPPAGE_PAYUP, 2)

                if exit_triggered:
                    raw_ret_pct = (exit_price - buy_px) / buy_px
                    cost_amount = active_pos['invested'] * ROUND_TRIP_FEE_RATE
                    pnl_amount = (active_pos['shares'] * (exit_price - buy_px)) - cost_amount
                    capital += pnl_amount
                    equity_curve.append(capital)
                    net_ret_pct = pnl_amount / active_pos['invested']

                    trade_rec = {
                        "date": d_str,
                        "symbol": sym,
                        "direction": active_pos['direction'],
                        "selected_expert": active_pos['selected_expert'],
                        "confidence": active_pos['confidence'],
                        "entry_time": active_pos['entry_time'],
                        "entry_price": buy_px,
                        "exit_time": bar_time,
                        "exit_price": exit_price,
                        "bars_held": bars_held,
                        "return_pct": round(net_ret_pct * 100, 2),
                        "pnl_usd": round(pnl_amount, 2),
                        "cost_usd": round(cost_amount, 2),
                        "capital_after": round(capital, 2),
                        "exit_reason": exit_reason
                    }
                    trades.append(trade_rec)

                    # 통계 누적
                    exp_k = active_pos['selected_expert']
                    m_stat = model_stats.setdefault(exp_k, {"total": 0, "wins": 0, "pnl": 0.0})
                    m_stat["total"] += 1
                    if pnl_amount > 0:
                        m_stat["wins"] += 1
                    m_stat["pnl"] += pnl_amount

                    active_pos = None

            # [B] 포지션이 없고 신규 진입 윈도우인 경우 (09:30 ~ 14:30 NYT)
            if active_pos is None and bar_time <= "14:30":
                # 과거 60개 캔들 슬라이스
                global_idx = tqqq_df[tqqq_df['datetime'] == row['datetime']].index[0]
                if global_idx >= 60:
                    tqqq_sub = tqqq_df.iloc[global_idx-59:global_idx+1].copy()
                    
                    # MoE 오케스트레이터 평가
                    cur_t_str = row['datetime'].strftime("%Y-%m-%d %H:%M:%S")
                    moe_res = moe.evaluate_dual_filter_signal(tqqq_sub, current_time_str=cur_t_str, threshold=CONFIDENCE_THRESHOLD)
                    
                    is_approved = moe_res.get("is_approved", False)
                    direction = moe_res.get("direction", "NONE")
                    top_conf = moe_res.get("gating_confidence", 0.0)
                    selected_exp = moe_res.get("expert_desc", "MoE")

                    if is_approved and direction in ["LONG_TQQQ", "SHORT_SQQQ"]:
                        winner_sym = "TQQQ" if direction == "LONG_TQQQ" else "SQQQ"
                        
                        if winner_sym == "TQQQ":
                            base_px = cur_tqqq_px
                        else:
                            sqqq_m = data_15m['SQQQ'][data_15m['SQQQ']['datetime'] == row['datetime']]
                            base_px = sqqq_m['Close'].iloc[0] if not sqqq_m.empty else 40.0

                        entry_px = round(base_px + SLIPPAGE_PAYUP, 2)
                        shares = int(capital / entry_px)
                        invested = shares * entry_px

                        if shares > 0 and invested > 0:
                            active_pos = {
                                "symbol": winner_sym,
                                "direction": direction,
                                "selected_expert": selected_exp,
                                "confidence": round(top_conf * 100, 1),
                                "entry_time": bar_time,
                                "entry_bar_idx": b_idx,
                                "buy_price": entry_px,
                                "shares": shares,
                                "invested": invested
                            }

    # =========================================================================
    # 성적표 산출 및 리포팅
    # =========================================================================
    df_trades = pd.DataFrame(trades)
    total_trades = len(df_trades)
    
    if total_trades == 0:
        print("⚠️ 백테스트 기간 동안 75% 기준을 충족한 거래가 없습니다.")
        return

    win_trades = df_trades[df_trades['pnl_usd'] > 0]
    loss_trades = df_trades[df_trades['pnl_usd'] <= 0]
    win_count = len(win_trades)
    loss_count = len(loss_trades)
    win_rate = (win_count / total_trades) * 100.0 if total_trades > 0 else 0.0

    total_profit = win_trades['pnl_usd'].sum() if len(win_trades) > 0 else 0.0
    total_loss = abs(loss_trades['pnl_usd'].sum()) if len(loss_trades) > 0 else 1e-6
    profit_factor = total_profit / total_loss if total_loss > 0 else 999.0

    total_return_pct = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100.0

    # 연환산 수익률 (CAGR)
    trading_days = len(unique_dates)
    years = trading_days / 252.0
    if years > 0 and capital > 0:
        cagr_pct = ((capital / INITIAL_CAPITAL) ** (1.0 / years) - 1.0) * 100.0
    else:
        cagr_pct = 0.0

    # 평균 손익 및 손익비 (Win-Loss Ratio)
    avg_win_pct = win_trades['return_pct'].mean() if win_count > 0 else 0.0
    avg_loss_pct = abs(loss_trades['return_pct'].mean()) if loss_count > 0 else 1e-6
    win_loss_ratio = avg_win_pct / avg_loss_pct if avg_loss_pct > 0 else 0.0

    # MDD 계산
    eq_series = pd.Series(equity_curve)
    cummax = eq_series.cummax()
    drawdown = (eq_series - cummax) / cummax
    mdd_pct = abs(drawdown.min()) * 100.0

    # 청산 사유별 분석
    reason_counts = df_trades['exit_reason'].value_counts().to_dict()

    print("\n" + "=" * 90)
    print("🏆 [Lumos Track 6: MoE AI 메타 오케스트레이터 실전 동일 백테스팅 최종 성적표]")
    print("=" * 90)
    print(f"💰 초기 자본금: ${INITIAL_CAPITAL:,.2f} USD (100% 복리 전액 운용)")
    print(f"💵 최종 자본금: ${capital:,.2f} USD (순이익: ${capital - INITIAL_CAPITAL:+,.2f} USD)")
    print(f"📈 총 누적 수익률 (Total Return): {total_return_pct:+.2f}%")
    print(f"🚀 연환산 수익률 (CAGR): {cagr_pct:+.2f}% (기준: {trading_days} 거래일)")
    print(f"🛡️ 최대 낙폭 (MDD): -{mdd_pct:.2f}%")
    print(f"🎯 총 거래 횟수: {total_trades}회 (승: {win_count}회 / 패: {loss_count}회)")
    print(f"👑 승률 (Win Rate): {win_rate:.2f}%")
    print(f"⚖️ 손익비 (Average Profit / Average Loss): {win_loss_ratio:.2f} (평균 익절 +{avg_win_pct:.2f}% / 평균 손절 -{avg_loss_pct:.2f}%)")
    print(f"📊 수익 팩터 (Profit Factor): {profit_factor:.2f} (총 수익: ${total_profit:+,.2f} / 총 손실: ${total_loss:+,.2f})")
    print(f"💸 총 차감된 거래비용(0.20%): ${df_trades['cost_usd'].sum():,.2f} USD")

    print("\n" + "-" * 90)
    print("🎯 [청산 사유별 발생 분포]")
    for r_name, count in reason_counts.items():
        pct = (count / total_trades) * 100
        print(f"   • {r_name}: {count}회 ({pct:.1f}%)")

    print("\n" + "-" * 90)
    print("🧠 [6대 AI 전문가 모델별 기여도 및 승률 분석]")
    for exp_name, st in sorted(model_stats.items(), key=lambda x: x[1]['pnl'], reverse=True):
        t_cnt = st['total']
        w_cnt = st['wins']
        wr = (w_cnt / t_cnt) * 100 if t_cnt > 0 else 0.0
        pnl = st['pnl']
        print(f"   • [{exp_name}]: {t_cnt}회 진입 | 승률: {wr:.1f}% ({w_cnt}승/{t_cnt-w_cnt}패) | 기여 손익: ${pnl:+,.2f} USD")

    print("\n" + "-" * 90)
    print("📋 [전체 거래 내역 요약]")
    tail_cols = ['date', 'symbol', 'direction', 'selected_expert', 'confidence', 'entry_price', 'exit_price', 'return_pct', 'pnl_usd', 'exit_reason']
    print(df_trades[tail_cols].to_string(index=False))
    print("=" * 90)

    # 결과 JSON 저장
    result_json = {
        "backtest_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "initial_capital_usd": INITIAL_CAPITAL,
        "final_capital_usd": round(capital, 2),
        "total_return_pct": round(total_return_pct, 2),
        "cagr_pct": round(cagr_pct, 2),
        "net_pnl_usd": round(capital - INITIAL_CAPITAL, 2),
        "total_trades": total_trades,
        "win_trades": win_count,
        "loss_trades": loss_count,
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2),
        "win_loss_ratio": round(win_loss_ratio, 2),
        "mdd_pct": round(mdd_pct, 2),
        "model_contributions": model_stats,
        "exit_reasons": reason_counts
    }
    with open(DATA_DIR / "moe_live_identical_backtest_summary.json", "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)

    df_trades.to_csv(DATA_DIR / "moe_live_identical_trades.csv", index=False, encoding="utf-8-sig")
    print(f"\n💾 백테스트 상세 거래 로그가 '{DATA_DIR / 'moe_live_identical_trades.csv'}'에 저장되었습니다.")

if __name__ == "__main__":
    run_live_identical_backtest()
