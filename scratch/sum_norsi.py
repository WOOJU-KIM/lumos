import pandas as pd
import matplotlib.pyplot as plt

df_orig = pd.read_csv('data/wfa_trades_03.csv')
df_norsi = pd.read_csv('data/wfa_no_rsi_trades.csv')

df_orig['entry_dt'] = pd.to_datetime(df_orig['entry_dt'])
df_norsi['entry_dt'] = pd.to_datetime(df_norsi['entry_dt'])

def get_stats(df, start=None, end=None):
    if start and end:
        sub_df = df[(df['entry_dt'] >= start) & (df['entry_dt'] <= end)]
    else:
        sub_df = df
    if len(sub_df) == 0: return 0, 0, 0, 0, 0
    wins = len(sub_df[sub_df['net_ret_pct'] > 0])
    win_rate = wins / len(sub_df) * 100
    cum_ret = (sub_df['net_ret_pct']/100 + 1).prod() - 1
    tqqq_cnt = len(sub_df[sub_df['symbol'] == 'TQQQ'])
    sqqq_cnt = len(sub_df[sub_df['symbol'] == 'SQQQ'])
    return len(sub_df), win_rate, cum_ret*100, tqqq_cnt, sqqq_cnt

t_o, w_o, r_o, l_o, s_o = get_stats(df_orig)
t_n, w_n, r_n, l_n, s_n = get_stats(df_norsi)

t_o_ai, w_o_ai, r_o_ai, l_o_ai, s_o_ai = get_stats(df_orig, '2023-07-01', '2024-08-31')
t_n_ai, w_n_ai, r_n_ai, l_n_ai, s_n_ai = get_stats(df_norsi, '2023-07-01', '2024-08-31')

t_o_bm, w_o_bm, r_o_bm, l_o_bm, s_o_bm = get_stats(df_orig, '2022-01-01', '2022-12-31')
t_n_bm, w_n_bm, r_n_bm, l_n_bm, s_n_bm = get_stats(df_norsi, '2022-01-01', '2022-12-31')

# Write markdown report
with open(r'C:\Users\chabo\.gemini\antigravity\brain\63fbbc63-cf1f-4855-b29f-0c4f0dbdb994\norsi_report.md', 'w', encoding='utf-8') as f:
    f.write(f'''# RSI 제한 해제 (순수 AI 모델) 백테스트 리포트

RSI 62/38 차단 필터를 완전히 해제(100/0으로 설정)하고, 오직 GBDT 모델의 신호와 크로스에셋 Veto만으로 진입한 결과입니다.

## 📊 종합 성능 비교 (4년 누적)
| 항목 | 원본 (RSI 62/38 필터) | 제한 해제 (순수 AI 모델) | 차이 |
| :--- | :--- | :--- | :--- |
| **누적 수익률** | **{r_o:.1f}%** | **{r_n:.1f}%** | {r_n - r_o:+.1f}%p |
| **총 매매 횟수** | {t_o}회 (L: {l_o}, S: {s_o}) | {t_n}회 (L: {l_n}, S: {s_n}) | {t_n - t_o:+}회 |
| **승률** | {w_o:.1f}% | {w_n:.1f}% | {w_n - w_o:+.1f}%p |

## 🔍 주요 국면별 비교 분석

### 1. AI 랠리 구간 (23.07 ~ 24.08)
| 항목 | 원본 (RSI 62/38 필터) | 제한 해제 (순수 AI 모델) |
| :--- | :--- | :--- |
| **수익률** | {r_o_ai:.1f}% | **{r_n_ai:.1f}%** |
| **매매 횟수** | {t_o_ai}회 (L: {l_o_ai}, S: {s_o_ai}) | {t_n_ai}회 (L: {l_n_ai}, S: {s_n_ai}) |
> 💡 RSI 제한이 풀리면서 대세 상승장에서 롱(TQQQ) 진입이 얼마나 폭증했는지, 그로 인해 랠리의 과실을 챙겼는지 확인해 보세요.

### 2. 폭락장 (2022년)
| 항목 | 원본 (RSI 62/38 필터) | 제한 해제 (순수 AI 모델) |
| :--- | :--- | :--- |
| **수익률** | {r_o_bm:.1f}% | **{r_n_bm:.1f}%** |
| **매매 횟수** | {t_o_bm}회 (L: {l_o_bm}, S: {s_o_bm}) | {t_n_bm}회 (L: {l_n_bm}, S: {s_n_bm}) |
> 💡 오히려 RSI 족쇄가 풀리면서 숏(SQQQ) 진입이 늘어나 방어력이 더 강해졌는지, 아니면 쓸데없는 매매로 수익을 갉아먹었는지 확인해 보세요.

---
![RSI 비교 차트](C:/Users/chabo/.gemini/antigravity/brain/63fbbc63-cf1f-4855-b29f-0c4f0dbdb994/equity_curve_norsi.png)
''')

# Create Plot
plt.figure(figsize=(12, 6))
df_orig_ts = df_orig.set_index('entry_dt')['capital']
df_norsi_ts = df_norsi.set_index('entry_dt')['capital']

plt.plot(df_orig_ts.index, df_orig_ts.values, label='Original (With RSI Filter)', color='dodgerblue', linewidth=2)
plt.plot(df_norsi_ts.index, df_norsi_ts.values, label='No RSI (Pure AI & Veto)', color='mediumseagreen', linewidth=2, alpha=0.9)

plt.title('Equity Curve Comparison (Original vs No RSI Filter)', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Account Balance (KRW)', fontsize=12)
plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plt.savefig(r'C:\Users\chabo\.gemini\antigravity\brain\63fbbc63-cf1f-4855-b29f-0c4f0dbdb994\equity_curve_norsi.png', dpi=300)
print('Summary and Plot created successfully.')
