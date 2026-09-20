import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from config import DATA_DIR

def run_monte_carlo_analysis(n_simulations: int = 20000, initial_capital: float = 10_000_000.0):
    trade_logs_path = DATA_DIR / "wfa_tqqq_trades.csv"
    if not trade_logs_path.exists():
        raise FileNotFoundError(f"{trade_logs_path}가 존재하지 않습니다.")

    df = pd.read_csv(trade_logs_path)
    df['pnl_pct_num'] = df['return_pct'] / 100.0

    wins = df[df['pnl_pct_num'] > 0]['pnl_pct_num'].values
    losses = df[df['pnl_pct_num'] <= 0]['pnl_pct_num'].values

    current_win_rate = len(wins) / len(df)
    avg_win = np.mean(wins)
    avg_loss = np.mean(losses)
    payoff_ratio = abs(avg_win / avg_loss)

    print("=" * 105)
    print("🎲 [Lumos 퀀트 트레이딩 모델 몬테카를로 리스크 & 파산 확률 정밀 분석 보고서]")
    print("=" * 105)
    print(f"📊 [입력 데이터 기초 통계 (data/wfa_tqqq_trades.csv 기준)]")
    print(f"- 총 백테스트 표본: {len(df)}회 거래 (TQQQ/SQQQ)")
    print(f"- 현재 모델 승률: {current_win_rate * 100:.2f}% ({len(wins)}승 / {len(losses)}패)")
    print(f"- 승리 시 평균 수익률: {avg_win * 100:+.2f}% (최대 +3.00%)")
    print(f"- 패배 시 평균 손실률: {avg_loss * 100:+.2f}% (최대 -2.00%)")
    print(f"- 손익비 (Payoff Ratio): {payoff_ratio:.2f}")
    print(f"- 청산 사유별 구성: 타임스탑(90m) {np.sum(df['exit_reason']=='TIME_STOP_90M')}회 | "
          f"익절(+3.0%) {np.sum(df['exit_reason']=='TAKE_PROFIT_3.0%')}회 | "
          f"손절(-2.0%) {np.sum(df['exit_reason']=='STOP_LOSS_2.0%')}회 | "
          f"장마감청산 {np.sum(df['exit_reason']=='END_OF_DAY_CLOSE')}회")
    print(f"- 몬테카를로 시뮬레이션 경로: 각 조건별 {n_simulations:,}회 무작위 패스 추출")
    print("=" * 105)

    scenarios = [
        {"name": "1. 현재 모델 실전 승률", "win_rate": current_win_rate, "tag": "Current (64.3%)"},
        {"name": "2. 승률 60% 보수적 시나리오", "win_rate": 0.60, "tag": "Conservative (60.0%)"},
        {"name": "3. 승률 40% 역풍/침체 시나리오", "win_rate": 0.40, "tag": "Adverse (40.0%)"},
    ]

    horizons = [
        {"label": "56회 거래 (현재 백테스트 동일 기간)", "n_trades": 56},
        {"label": "100회 거래 (약 6개월~1년 중기 운용)", "n_trades": 100},
        {"label": "250회 거래 (약 1.5년~2년 장기 누적)", "n_trades": 250},
    ]

    all_horizon_results = {}

    np.random.seed(42)

    for hor in horizons:
        n_trades = hor["n_trades"]
        horizon_label = hor["label"]
        results_summary = []

        for sc in scenarios:
            wr = sc["win_rate"]
            name = sc["name"]

            is_win = np.random.rand(n_simulations, n_trades) < wr
            win_returns = np.random.choice(wins, size=(n_simulations, n_trades), replace=True)
            loss_returns = np.random.choice(losses, size=(n_simulations, n_trades), replace=True)

            returns = np.where(is_win, win_returns, loss_returns)

            # 복리 자산 곡선
            multipliers = 1.0 + returns
            cum_multipliers = np.cumprod(multipliers, axis=1)
            capital_curves = np.hstack([np.ones((n_simulations, 1)), cum_multipliers]) * initial_capital

            final_capitals = capital_curves[:, -1]
            total_returns = (final_capitals - initial_capital) / initial_capital * 100.0

            # Drawdown
            running_max = np.maximum.accumulate(capital_curves, axis=1)
            drawdowns = (running_max - capital_curves) / running_max * 100.0
            max_drawdowns = np.max(drawdowns, axis=1)

            # 파산 확률 (MDD 기준)
            prob_dd_15 = np.mean(max_drawdowns >= 15.0) * 100.0
            prob_dd_20 = np.mean(max_drawdowns >= 20.0) * 100.0
            prob_dd_30 = np.mean(max_drawdowns >= 30.0) * 100.0
            prob_dd_50 = np.mean(max_drawdowns >= 50.0) * 100.0
            prob_dd_70 = np.mean(max_drawdowns >= 70.0) * 100.0
            prob_loss_finish = np.mean(final_capitals < initial_capital) * 100.0

            # 최대 연속 손실 계산
            is_loss = ~is_win
            max_consec_losses = []
            for i in range(n_simulations):
                row = is_loss[i]
                max_c = 0
                cur_c = 0
                for val in row:
                    if val:
                        cur_c += 1
                        if cur_c > max_c:
                            max_c = cur_c
                    else:
                        cur_c = 0
                max_consec_losses.append(max_c)
            max_consec_losses = np.array(max_consec_losses)

            stat = {
                "horizon": horizon_label,
                "n_trades": n_trades,
                "scenario": name,
                "tag": sc["tag"],
                "win_rate_pct": wr * 100.0,
                "expected_return_pct": np.mean(total_returns),
                "median_return_pct": np.median(total_returns),
                "worst_return_pct": np.min(total_returns),
                "p01_return_pct": np.percentile(total_returns, 1),
                "p05_return_pct": np.percentile(total_returns, 5),
                "p25_return_pct": np.percentile(total_returns, 25),
                "p75_return_pct": np.percentile(total_returns, 75),
                "p95_return_pct": np.percentile(total_returns, 95),
                "p99_return_pct": np.percentile(total_returns, 99),
                "best_return_pct": np.max(total_returns),
                "mean_mdd_pct": np.mean(max_drawdowns),
                "median_mdd_pct": np.median(max_drawdowns),
                "p95_mdd_pct": np.percentile(max_drawdowns, 95),
                "p99_mdd_pct": np.percentile(max_drawdowns, 99),
                "worst_mdd_pct": np.max(max_drawdowns),
                "prob_dd_15_pct": prob_dd_15,
                "prob_dd_20_pct": prob_dd_20,
                "prob_dd_30_pct": prob_dd_30,
                "prob_dd_50_pct": prob_dd_50,
                "prob_dd_70_pct": prob_dd_70,
                "prob_loss_finish_pct": prob_loss_finish,
                "avg_max_consec_loss": np.mean(max_consec_losses),
                "p95_consec_loss": np.percentile(max_consec_losses, 95),
                "p99_consec_loss": np.percentile(max_consec_losses, 99),
                "worst_consec_loss": np.max(max_consec_losses),
                "median_final_capital": np.median(final_capitals),
                "worst_final_capital": np.min(final_capitals),
                "p01_final_capital": np.percentile(final_capitals, 1),
                "p05_final_capital": np.percentile(final_capitals, 5),
                "p95_final_capital": np.percentile(final_capitals, 95),
            }
            results_summary.append(stat)

        all_horizon_results[horizon_label] = pd.DataFrame(results_summary)

    return all_horizon_results

if __name__ == "__main__":
    results = run_monte_carlo_analysis(n_simulations=20000)
    for h_label, df_res in results.items():
        print("\n" + "=" * 105)
        print(f"📌 [기간 설정: {h_label}]")
        print("=" * 105)
        for idx, r in df_res.iterrows():
            print(f"\n▶ [{r['scenario']} (승률 {r['win_rate_pct']:.1f}%)]")
            print(f"  • [수익률 분포]")
            print(f"    - 기대(평균) 수익률: {r['expected_return_pct']:+.2f}% | 중앙값: {r['median_return_pct']:+.2f}%")
            print(f"    - 20,000회 중 최악 경로 (Absolute Worst 1-path): {r['worst_return_pct']:+.2f}% (최저 잔고: {int(r['worst_final_capital']):,}원)")
            print(f"    - 하위 1% 극단 구간 (99% VaR): {r['p01_return_pct']:+.2f}% (잔고: {int(r['p01_final_capital']):,}원)")
            print(f"    - 하위 5% 비관적 구간 (95% VaR): {r['p05_return_pct']:+.2f}% (잔고: {int(r['p05_final_capital']):,}원)")
            print(f"    - 상위 5% 낙관적 구간: {r['p95_return_pct']:+.2f}% (잔고: {int(r['p95_final_capital']):,}원)")
            print(f"  • [최대 낙폭 (MDD)]")
            print(f"    - 평균 MDD: {r['mean_mdd_pct']:.2f}% | 중앙값 MDD: {r['median_mdd_pct']:.2f}%")
            print(f"    - 95% 신뢰수준 MDD: {r['p95_mdd_pct']:.2f}% | 99% 신뢰수준 MDD: {r['p99_mdd_pct']:.2f}% | 최악 MDD: {r['worst_mdd_pct']:.2f}%")
            print(f"  • [연속 손실 (Consecutive Losses)]")
            print(f"    - 평균 최대 연패: {r['avg_max_consec_loss']:.1f}연패 | 95% 한계: {int(r['p95_consec_loss'])}연패 | 최악: {int(r['worst_consec_loss'])}연패")
            print(f"  • [파산 확률 (Risk of Ruin)]")
            print(f"    - 경고선 [MDD ≥ 15% 도달 확률]: {r['prob_dd_15_pct']:.2f}%")
            print(f"    - 주의선 [MDD ≥ 20% 도달 확률]: {r['prob_dd_20_pct']:.2f}%")
            print(f"    - 위험선 [MDD ≥ 30% 도달 확률]: {r['prob_dd_30_pct']:.2f}%")
            print(f"    - 전략 폐기선 [MDD ≥ 50% 반토막 확률]: {r['prob_dd_50_pct']:.2f}%")
            print(f"    - 계좌 깡통 [MDD ≥ 70% 소실 확률]: {r['prob_dd_70_pct']:.2f}%")
            print(f"    - 기간 종료 시점 원금 손실 마감 확률: {r['prob_loss_finish_pct']:.2f}%")
        print("=" * 105)
