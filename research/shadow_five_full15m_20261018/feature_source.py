"""Compile unchanged pure production feature methods; read price DB only, never live constructors."""
from common import PROJECT, POLICY
import ast, types, sys, sqlite3, logging, threading
from datetime import datetime
from typing import Any,Optional,Dict,List,Tuple
import numpy as np
import pandas as pd

C=types.ModuleType('config')
tree=ast.parse((PROJECT/'config.py').read_text(encoding='utf-8-sig'))
for node in tree.body:
    if isinstance(node,ast.Assign):
        try:value=ast.literal_eval(node.value)
        except (ValueError,TypeError):continue
        for target in node.targets:
            if isinstance(target,ast.Name) and target.id.isupper() and not any(s in target.id for s in ['TOKEN','KEY','SECRET','ACCOUNT','PASSWORD']):
                setattr(C,target.id,value)
C.TRADE_SYMBOLS=[C.BASE_ASSET_LONG,C.BASE_ASSET_SHORT]
C.TRAINING_LOOKBACK_DAYS=C.ROLLING_TRAINING_WEEKS*7
sys.modules['config']=C
RAW={}
class Lake:
    def load_candles(self,symbol,timeframe,start_dt=None,end_dt=None):
        key=(symbol,timeframe)
        if key not in RAW:
            conn=sqlite3.connect((PROJECT/'data/market_data.db').as_uri()+'?mode=ro',uri=True)
            conn.execute('pragma query_only=ON')
            d=pd.read_sql_query('select datetime,open as Open,high as High,low as Low,close as Close,volume as Volume from market_candles where symbol=? and timeframe=? order by datetime',conn,params=key)
            conn.close();d.index=pd.to_datetime(d.datetime);d.index.name='Datetime'
            minutes={'5m':5,'15m':15,'60m':60}[timeframe]
            d=d.loc[d.index+pd.Timedelta(minutes=minutes)<=pd.Timestamp(POLICY['training_cutoff_ny'])]
            if d.index.has_duplicates:raise ValueError('Duplicate market timestamp')
            RAW[key]=d
        d=RAW[key]
        if start_dt:d=d.loc[d.index>=pd.Timestamp(start_dt)]
        if end_dt:d=d.loc[d.index<=pd.Timestamp(end_dt)]
        return d.copy()
    def get_candles_with_live_tick(self,symbol,timeframe,live_price=None):
        return self.load_candles(symbol,timeframe)
stub=types.ModuleType('core.data_lake');stub.MarketDataLake=Lake
sys.modules['core']=types.ModuleType('core');sys.modules['core.data_lake']=stub
source=ast.parse((PROJECT/'core/ml_engine.py').read_text(encoding='utf-8-sig'))
cl=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='MLFeatureEngine')
selected={'extract_features','compute_triple_barrier_labels','get_feature_columns','add_confidence_columns','_get_thread_local'}
body=[n for n in cl.body if isinstance(n,ast.Assign) or isinstance(n,ast.FunctionDef) and n.name in selected]
pure=ast.ClassDef(name='PureFeatures',bases=[],keywords=[],body=body,decorator_list=[])
module=ast.Module(body=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name=='_safe_merge_asof']+[pure],type_ignores=[])
ns=dict(np=np,pd=pd,config=C,logging=logging,logger=logging.getLogger('candidate_feature'),threading=threading,
        Any=Any,Optional=Optional,Dict=Dict,List=List,Tuple=Tuple,_ModelSnapshot=Any)
exec(compile(ast.fix_missing_locations(module),'UNCHANGED_PRODUCTION_FEATURE_METHODS','exec'),ns)
Engine=ns['PureFeatures']
