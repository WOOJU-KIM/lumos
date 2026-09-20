import pandas as pd
import matplotlib.pyplot as plt

df_orig = pd.read_csv('data/wfa_trades_03.csv')
df_v1 = pd.read_csv('data/wfa_trades_rsi_100_0.csv')
df_v2 = pd.read_csv('data/wfa_trades_v2_features.csv')
df_v3 = pd.read_csv('data/wfa_trades_v3_stacked.csv')

for df in [df_orig, df_v1, df_v2, df_v3]:
    df['entry_dt'] = pd.to_datetime(df['entry_dt'])

def get_stats(df, start=None, end=None):
    if start and end:
        sub_df = df[(df['entry_dt'] >= start) & (df['entry_dt'] <= end)]
    else:
        sub_df = df
    if len(sub_df) == 0: return 0, 0, 0, 0, 0
    wins = len(sub_df[sub_df['net_ret_pct'] > 0])
    win_rate = wins / len(sub_df) * 100
    cum_ret = (sub_df['net_ret_pct']/100 + 1).prod() - 1
    
    # Calculate CAGR
    days = (sub_df['entry_dt'].iloc[-1] - sub_df['entry_dt'].iloc[0]).days
    years = days / 365.25 if days > 0 else 1
    cagr = (cum_ret + 1) ** (1 / years) - 1
    
    return len(sub_df), win_rate, cum_ret*100, cagr*100

t_o, w_o, r_o, c_o = get_stats(df_orig)
t_1, w_1, r_1, c_1 = get_stats(df_v1)
t_2, w_2, r_2, c_2 = get_stats(df_v2)
t_3, w_3, r_3, c_3 = get_stats(df_v3)

# MDD calc
def get_mdd(df):
    cap = pd.concat([pd.Series([10_000_000]), df['capital']]).reset_index(drop=True)
    drawdown = (cap - cap.cummax()) / cap.cummax()
    return drawdown.min() * 100

mdd_o = get_mdd(df_orig)
mdd_1 = get_mdd(df_v1)
mdd_2 = get_mdd(df_v2)
mdd_3 = get_mdd(df_v3)

with open(r'C:\Users\chabo\.gemini\antigravity\brain\63fbbc63-cf1f-4855-b29f-0c4f0dbdb994\v3_stacked_report.md', 'w', encoding='utf-8') as f:
    f.write(f'''# 피처 스태킹 (V3) 4중 비교 백테스트 리포트

## 📊 종합 성과 비교
| 항목 | 1. 원본 (RSI 족쇄) | 2. V1 (파생) | 3. V2 (맹점 보완) | **4. V3 (스태킹 & 튜닝)** |
| :--- | :--- | :--- | :--- | :--- |
| **누적 수익률** | {r_o:.1f}% | {r_1:.1f}% | {r_2:.1f}% | **{r_3:.1f}%** |
| **연평균(CAGR)**| {c_o:.1f}% | {c_1:.1f}% | {c_2:.1f}% | **{c_3:.1f}%** |
| **최대 낙폭(MDD)**| {mdd_o:.1f}% | {mdd_1:.1f}% | {mdd_2:.1f}% | **{mdd_3:.1f}%** |
| **매매 횟수**   | {t_o}회 | {t_1}회 | {t_2}회 | **{t_3}회** |
| **승률**        | {w_o:.1f}% | {w_1:.1f}% | {w_2:.1f}% | **{w_3:.1f}%** |

---
![4개 모델 수익률 비교](C:/Users/chabo/.gemini/antigravity/brain/63fbbc63-cf1f-4855-b29f-0c4f0dbdb994/equity_curve_v3.png)
''')

# Plot
plt.figure(figsize=(13, 7))
df_orig_ts = df_orig.set_index('entry_dt')['capital']
df_v1_ts = df_v1.set_index('entry_dt')['capital']
df_v2_ts = df_v2.set_index('entry_dt')['capital']
df_v3_ts = df_v3.set_index('entry_dt')['capital']

plt.plot(df_orig_ts.index, df_orig_ts.values, label='1. Original (Base + RSI Filter)', color='dodgerblue', linewidth=1.5, alpha=0.5)
plt.plot(df_v1_ts.index, df_v1_ts.values, label='2. V1 (Features + Time Decay)', color='magenta', linewidth=1.5, alpha=0.5)
plt.plot(df_v2_ts.index, df_v2_ts.values, label='3. V2 (Math Fixes)', color='red', linewidth=1.5, alpha=0.5)
plt.plot(df_v3_ts.index, df_v3_ts.values, label='4. V3 (Feature Stacking & Tuning)', color='gold', linewidth=3)

plt.title('Equity Curve Comparison: Evolution of the AI Model', fontsize=15, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Account Balance (KRW)', fontsize=12)
plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plt.savefig(r'C:\Users\chabo\.gemini\antigravity\brain\63fbbc63-cf1f-4855-b29f-0c4f0dbdb994\equity_curve_v3.png', dpi=300)
print('V3 Plot created successfully.')
