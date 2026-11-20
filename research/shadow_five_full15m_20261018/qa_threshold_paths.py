"""Synthetic adversarial path checks. These fixtures are not market performance."""
from common import *
guard()
import ast
import numpy as np
import pandas as pd

tree=ast.parse((HOME/'threshold_64_704.py').read_text(encoding='utf-8'))
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='path')
settings=dict(CONFIG);settings['TIME_STOP_MINUTES']=10  # Synthetic fixture deadline only.
scope=dict(np=np,pd=pd,CONFIG=settings,five={},atr={})
exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),'PATH_UNDER_TEST','exec'),scope)
t=pd.Timestamp('2026-10-12 10:00:00');index=pd.date_range(t-pd.Timedelta(minutes=70),t+pd.Timedelta(minutes=10),freq='5min')
def fixture():
    d=pd.DataFrame(dict(Open=100.,High=100.,Low=100.,Close=100.,Volume=100.),index=index)
    scope['five']['TEST_ONLY']=d;scope['atr']['TEST_ONLY']=pd.Series(1.,index=index)
    return d
d=fixture();d.loc[t,['High','Low']]=[111.,97.]
r=scope['path']('TEST_ONLY',t)
assert r['reason']=='SL' and r['exit_price']==98.,'SL must precede same-candle TP'
d=fixture();d.loc[t,['High','Low']]=[105.,99.];d.loc[t+pd.Timedelta(minutes=5),'Open']=104.
r=scope['path']('TEST_ONLY',t)
assert r['reason']=='TRAIL' and r['exit_price']==104. and r['exit_time']==t+pd.Timedelta(minutes=5),'Current high cannot tighten current low stop'
assert abs(r['net_return']-(.04-CONFIG['BACKTEST_FEE_SLIPPAGE']))<1e-12,'Charge costs exactly once'
d=fixture();d.loc[t+pd.Timedelta(minutes=5),'Open']=97.
r=scope['path']('TEST_ONLY',t)
assert r['exit_price']==97.,'Gap through SL must use the worse open'
d=fixture();r=scope['path']('TEST_ONLY',t)
assert r['reason']=='TIME' and r['exit_time']==t+pd.Timedelta(minutes=10)
result=dict(status='PASS',checks=5,fixtures='SYNTHETIC_NOT_PERFORMANCE',cases=['SL_before_TP','prior_peak_only','cost_once','gap_stop','time_stop'])
save('threshold_64_704_20261018/path_qa.json',result)
print(json.dumps(result))
