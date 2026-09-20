import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

df_tqqq = pd.read_csv("data/wfa_live_identical_trades.csv")
df_tqqq = pd.read_csv("data/wfa_tqqq_trades.csv")
df_upro = pd.read_csv("data/wfa_upro_trades.csv")

df_tqqq['entry_dt'] = pd.to_datetime(df_tqqq['entry_dt'])
df_tqqq['entry_dt'] = pd.to_datetime(df_tqqq['entry_dt'])
df_upro['entry_dt'] = pd.to_datetime(df_upro['entry_dt'])

start_date = df_tqqq['entry_dt'].iloc[0]

df_tqqq_aligned = df_tqqq[df_tqqq['entry_dt'] >= start_date].copy()
df_upro_aligned = df_upro[df_upro['entry_dt'] >= start_date].copy()

if not df_tqqq_aligned.empty:
    initial_tqqq = df_tqqq_aligned['capital'].iloc[0]
    df_tqqq_aligned['capital'] = (df_tqqq_aligned['capital'] / initial_tqqq) * 10000000

if not df_upro_aligned.empty:
    initial_upro = df_upro_aligned['capital'].iloc[0]
    df_upro_aligned['capital'] = (df_upro_aligned['capital'] / initial_upro) * 10000000

plt.figure(figsize=(12, 6))
plt.plot(df_tqqq['entry_dt'], df_tqqq['capital'], label='TQQQ (Semi) Equity', color='blue', alpha=0.7)
plt.plot(df_tqqq_aligned['entry_dt'], df_tqqq_aligned['capital'], label='TQQQ (NDX) Equity', color='red', alpha=0.9)
plt.plot(df_upro_aligned['entry_dt'], df_upro_aligned['capital'], label='UPRO (SP500) Equity', color='green', alpha=0.9)

plt.title('Lumos V5: TQQQ vs TQQQ vs UPRO Backtest Comparison (Aligned)')
plt.xlabel('Date')
plt.ylabel('Equity (KRW) - Log Scale')
plt.yscale('log')
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend(loc='upper left')
plt.tight_layout()

plt.savefig("backtest_history/combined_chart_all.png")

# Calculate aligned CAGR/Return
def calc_metrics(df):
    ret = (df['capital'].iloc[-1] / df['capital'].iloc[0] - 1) * 100
    days = (df['entry_dt'].iloc[-1] - df['entry_dt'].iloc[0]).days
    cagr = ((df['capital'].iloc[-1] / df['capital'].iloc[0]) ** (365.25/days) - 1) * 100
    return ret, cagr

print("TQQQ:", calc_metrics(df_tqqq))
print("TQQQ:", calc_metrics(df_tqqq_aligned))
print("UPRO:", calc_metrics(df_upro_aligned))

