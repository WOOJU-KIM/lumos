import sys
from pathlib import Path
import re

with open('scripts/temp_run_wfa_refactored.py', 'r', encoding='utf-8') as f:
    c = f.read()

# 1. Modify function signature
c = c.replace('def run_weekly_rolling_walk_forward():', 'def run_weekly_rolling_walk_forward(custom_time_stop):')

# 2. Modify test week loop to run only 2026 weeks
target_loop = "for w_idx in range(len(unique_weeks) - 1, len(unique_weeks)):"
replacement_loop = """
    weeks_2026 = [w for w in unique_weeks if w.startswith('2026')]
    test_start_idx = unique_weeks.index(weeks_2026[0]) if weeks_2026 else len(unique_weeks)-1
    test_end_idx = unique_weeks.index(weeks_2026[-1]) + 1 if weeks_2026 else len(unique_weeks)
    for w_idx in range(test_start_idx, test_end_idx):
"""
c = c.replace(target_loop, replacement_loop)

# 3. Replace config.TIME_STOP_MINUTES with custom_time_stop
c = c.replace('config.TIME_STOP_MINUTES', 'custom_time_stop')

# 4. Modify the print statements to be quiet
c = re.sub(r'print\(.*?\[[1-4]/4\].*?\)', 'pass', c)
c = re.sub(r'print\(f"   • \[\{week_progress\}/.*?\]"\)', 'pass', c)

# 5. Modify the end to return stats instead of printing massive tables
end_block_start = c.find('print("=" * 115)\n    print("📊 [1. 주간 롤링')
if end_block_start != -1:
    c = c[:end_block_start] + """
    return total_stats, max_drawdown_pct
"""

# 6. Add the main loop
c += """
if __name__ == "__main__":
    results = []
    for ts in [90, 100, 110, 120]:
        print(f"\\n▶ Running WFA for 2026 with TIME_STOP = {ts} minutes...")
        stats, mdd = run_weekly_rolling_walk_forward(ts)
        results.append({
            'TimeStop': ts,
            'Ret%': stats['ret_pct'],
            'MDD%': round(mdd, 2),
            'WinRate%': stats['win_rate'],
            'Trades': stats['trades'],
            'PF': stats['pf'],
            'SOXL': stats['long_trades'],
            'SOXS': stats['short_trades']
        })
        print(f"  Result: Ret {stats['ret_pct']}% | MDD {mdd:.2f}% | WinRate {stats['win_rate']}% | Trades {stats['trades']}")

    print("\\n" + "="*80)
    print("🏆 [2026년 타임스탑 파라미터 최적화 (DoE) 결과]")
    print("="*80)
    print(f"{'TimeStop':<10} | {'Ret %':<8} | {'MDD %':<8} | {'WinRate %':<10} | {'Trades':<8} | {'PF':<6} | {'SOXL/SOXS':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['TimeStop']:<10} | {r['Ret%']:>+7.1f}% | {r['MDD%']:>7.2f}% | {r['WinRate%']:>8.1f}% | {r['Trades']:>6} | {r['PF']:>5.2f} | {r['SOXL']}/{r['SOXS']}")
    print("="*80)
"""

with open('scripts/run_doe_time_stop.py', 'w', encoding='utf-8') as f:
    f.write(c)

print("Created run_doe_time_stop.py")
