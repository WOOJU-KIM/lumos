import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.backtest_engine import GranularBacktestEngine

engine = GranularBacktestEngine(initial_capital_krw=10_000_000, confidence_threshold=0.40)
res = engine.run_backtest()

print("=" * 80)
print("🏆 [원조 챔피언 전략: 3중 스크린 GBDT 스나이퍼 백테스트 결과]")
print("=" * 80)
print(f"💰 초기 자본금: 10,000,000원")
print(f"💵 최종 자본금: {int(res['final_capital_krw']):,}원")
print(f"📈 총 누적 수익률: {res['total_return_pct']:+.2f}%")
total_trades = res.get("total_trades_count", res.get("total_trades", 0))
weeks_count = len(res.get("weekly_reports", [])) or 13
print(f"🎯 총 거래 횟수: {total_trades}회 ({weeks_count}주 동안 주당 {total_trades/weeks_count:.1f}회 거래)")
print(f"👑 승률: {res['win_rate_pct']:.2f}% ({res['total_wins']}승 / {res['total_losses']}패)")
print(f"⚖️ 손익비: {res['profit_factor']:.2f}")
print(f"🛡️ MDD: {res['mdd_pct']:.2f}%")
print("=" * 80)
