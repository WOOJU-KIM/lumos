"""Render recorded trades and real 5m closes; flat cash periods stay flat."""
from common import *
guard()
import numpy as np
import pandas as pd
from feature_source import Lake
os.environ['MPLCONFIGDIR']=str(HOME/'cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

folder=HOME/'threshold_64_704_20261018'
summary=json.loads((folder/'results.json').read_text())
plt.rcParams.update({'font.family':'Malgun Gothic','axes.unicode_minus':False})
fig,axes=plt.subplots(2,1,figsize=(10,7),gridspec_kw={'height_ratios':[2,1]},constrained_layout=True)
for threshold in summary['thresholds']:
    rows=pd.read_csv(folder/f'trades_{threshold}.csv',parse_dates=['entry_time','exit_time'])
    equity=[(pd.Timestamp(summary['start']),10000.)]
    for _,r in rows.iterrows():
        prior=float(r.equity-r.pnl_usd)
        d=Lake().load_candles(r.symbol,'5m')
        d=d.loc[(d.index>=r.entry_time)&(d.index+pd.Timedelta(minutes=5)<r.exit_time)]
        for at,bar in d.iterrows():
            value=prior+r.quantity*(bar.Close-r.entry_price)-r.quantity*r.entry_price*CONFIG['BACKTEST_FEE_SLIPPAGE']
            equity.append((at+pd.Timedelta(minutes=5),value))
        equity.append((r.exit_time,r.equity))
    equity.append((pd.Timestamp(summary['end'])+pd.Timedelta(hours=16),float(rows.iloc[-1].equity)))
    e=pd.DataFrame(equity,columns=['time','equity']);v=e.equity.to_numpy();draw=(v/np.maximum.accumulate(v)-1)*100
    expected=next(r for r in summary['results'] if r['threshold']==threshold and r['period'].startswith(summary['start']))
    assert abs(draw.min()-expected['MDD_pct'])<1e-8
    assert abs((v[-1]/10000-1)*100-expected['return_pct'])<1e-8
    e.to_csv(folder/f'equity_{threshold}.csv',index=False)
    axes[0].step(e.time,(v/10000-1)*100,where='post',label=f'확신도 {threshold*100:g}%')
    axes[1].step(e.time,draw,where='post',label=f'{threshold*100:g}%')
axes[0].set(title='기존 B72 학습 방식: 64% vs 70.4% | 주간 워크포워드',ylabel='누적 순수익률 (%)')
axes[1].set(ylabel='낙폭 (%)',xlabel='뉴욕 거래일')
for ax in axes:ax.axhline(0,color='gray',lw=.7);ax.grid(alpha=.2);ax.legend()
fig.suptitle('2026-07-27 ~ 10-16 · 동일 55거래일 · 왕복 비용 0.20% · 15분 판단 / 5분 청산',fontsize=11)
fig.savefig(folder/'comparison.png',dpi=160);plt.close(fig)
print('CHART_VALUES_MATCH_RECORDED_METRICS')
