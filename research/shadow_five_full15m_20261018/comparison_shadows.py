"""Exactly three independent WS portfolios sharing the selected live classifier.

No broker imports, API calls, credentials or live-order methods. Paper sell/buy
fills require a subsequent, fresh observed WebSocket quote, with explicit costs.
"""
from datetime import timedelta
from functools import partial
import hashlib
import json
import math
import joblib
import numpy as np
import pandas as pd
from joblib.numpy_pickle import NumpyUnpickler
from joblib.numpy_pickle_utils import _validate_fileobject_and_memmap
from common import HOME,PROJECT,load_execution_config,sha
from comparison_control import comparison_settings,main_session_start
from relay_directions import RelayLedger
from runtime import stamp


class _StoredState:
    """Read serialized application state without importing/constructing live modules."""
    def __setstate__(self,state):self.__dict__.update(state)


class _ObserverUnpickler(NumpyUnpickler):
    def find_class(self,module,name):
        if module.startswith('core.'):
            return _StoredState
        return super().find_class(module,name)


def load_classifier_bundle(path):
    with path.open('rb') as stream:
        with _validate_fileobject_and_memmap(stream,str(path),None) as (fileobj,mmap_mode):
            obj=_ObserverUnpickler(str(path),fileobj,ensure_native_byte_order=True,mmap_mode=mmap_mode).load()
    if isinstance(obj,dict):return obj['model'],list(obj['feature_names'])
    engine=getattr(obj,'gbdt_engine',obj)
    return engine.model,list(engine.feature_names)


class LiveModelView:
    def __init__(self):
        self.sha256=None;self.last_decision=None;self.last_prediction=None
        self.reload()

    def reload(self):
        manifest=json.loads((PROJECT/'models/active_model_manifest.json').read_text(encoding='utf-8'))
        if manifest['expected_schema']!='v4_champion' or manifest['expected_num_features']!=72:
            raise ValueError('Comparison requires the user-selected absolute live model')
        digest=manifest['expected_sha256']
        if digest==self.sha256:return
        path=PathFromManifest(manifest)
        if sha(path)!=digest:raise ValueError('Live comparison artifact SHA mismatch')
        model,names=load_classifier_bundle(path)
        if (names!=manifest['feature_names'] or len(names)!=72 or
                list(model.feature_name_)!=names or list(model.classes_)!=[0,1,2] or
                hashlib.sha256('|'.join(names).encode()).hexdigest()!=manifest['feature_names_sha256']):
            raise ValueError('Live comparison feature order/classes mismatch')
        self.model=model;self.names=names;self.sha256=digest;self.model_id=manifest['active_model_id']

    def predict(self,packet):
        decision=packet['decision_id']
        # The complete event uses one immutable model version across all three.
        if decision==self.last_decision:return dict(self.last_prediction)
        self.reload()
        values=dict(packet['full_features'],**packet.get('relay_features',{}))
        x=pd.DataFrame([[float(values[n]) for n in self.names]],columns=self.names)
        if not np.isfinite(x.to_numpy()).all():raise ValueError('Incomplete live comparison features')
        ps,pn,pl=map(float,self.model.predict_proba(x)[0])
        if not all(math.isfinite(p) and 0<=p<=1 for p in (ps,pn,pl)) or not math.isclose(ps+pn+pl,1.,abs_tol=1e-6):
            raise ValueError('Invalid live comparison probabilities')
        cfg=load_execution_config()
        score=min(.95,max(cfg['CALIB_BASE'],cfg['CALIB_BASE']+(max(ps,pn,pl)-cfg['CALIB_PIVOT'])*cfg['CALIB_SCALE']))
        side=1 if pl>ps and pl>pn else 0 if ps>pl and ps>pn else -1
        reason='LIVE_GATE_PASSED'
        if side<0:reason='NEUTRAL_OR_TIE'
        elif score<cfg['GBDT_CONFIDENCE_THRESHOLD']:side,reason=-1,'BELOW_LIVE_THRESHOLD'
        scheduled=stamp(packet['decision_time']);hour,minute=main_session_start()
        if (stamp(packet['feature_times'][-1])!=scheduled or
                scheduled<scheduled.replace(hour=hour,minute=minute)+timedelta(minutes=cfg['RELAY_DECISION_INTERVAL_MINUTES'])):
            side,reason=-1,'LATEST_CLOSED_CANDLE_MISSING'
        result=dict(side=side,forced_side=side,calibrated_confidence=score,raw_probabilities=[ps,pn,pl],
                    inference_model_id=self.model_id,inference_model_sha256=self.sha256,
                    entry_reason=reason,threshold_applied=True)
        self.last_decision=decision;self.last_prediction=dict(result)
        return result


def PathFromManifest(manifest):
    from pathlib import Path
    p=Path(manifest['active_model_path'])
    return p if p.is_absolute() else PROJECT/p


class ComparisonCandidate:
    cohorts=('primary',)
    def __init__(self,name,definition,view):
        self.name=name;self.definition=dict(definition);self.view=view
        self.version=view.sha256+':'+sha(HOME/'comparison_shadows.json')+':'+name
        self.ledger_factory=partial(ComparisonLedger,definition=self.definition)

    def predict(self,packet):
        result=self.view.predict(packet)
        self.version=result['inference_model_sha256']+':'+sha(HOME/'comparison_shadows.json')+':'+self.name
        result.update(model_id=self.name,model_version=self.version,execution_strategy=self.definition['mode'],
                      execution_policy_sha256=sha(HOME/'comparison_shadows.json'))
        return result


class ComparisonLedger(RelayLedger):
    def __init__(self,model_id,cohort='primary',evaluation_window=None,definition=None):
        super().__init__(model_id,cohort,evaluation_window)
        if cohort!='primary':raise ValueError('Comparison has one ledger per strategy')
        self.definition=dict(definition);self.mode=self.definition['mode']
        self.recipe=comparison_settings();self.cash=float(self.recipe['initial_cash'])
        self.execution_config=dict(self.execution_config)
        for setting,key in [('RELAY_INITIAL_ALLOCATION','initial_allocation'),('RELAY_CHAIN_ALLOCATION','chain_allocation')]:
            self.execution_config[setting]=float(self.definition[key])
        initial=self.execution_config['RELAY_INITIAL_ALLOCATION'];chain=self.execution_config['RELAY_CHAIN_ALLOCATION']
        if not (0<initial<=self.execution_config['MAX_ALLOCATION_RATIO'] and 0<=chain and initial+chain<=self.execution_config['MAX_ALLOCATION_RATIO']):
            raise ValueError('Comparison allocation exceeds approved capital')
        if self.mode=='SINGLE' and chain!=0:raise ValueError('Single strategy cannot allocate chain capital')
        if self.mode=='PRELOSS':
            trigger=float(self.definition['chain_trigger_loss'])
            if not 0<trigger<self.execution_config['SL_MIN_PCT']:raise ValueError('Invalid preloss trigger')
            self.execution_config['RELAY_OPPOSITE_TRIGGER_PCT']=trigger
        self._reverse_wait=None

    def _chain(self,now):
        return super()._chain(now) if self.mode=='PRELOSS' else []

    def _finish(self,now):
        if self._reverse_wait is not None:return []
        return super()._finish(now)

    def _source_valid(self,pending,now):
        if self.mode!='AFTER_SL':return super()._source_valid(pending,now)
        fill=pending.get('sl_fill')
        return bool(fill and not self.positions and self.chain_filled and
                    fill['cycle_id']==self.cycle_id and self.closed and
                    self.closed[-1]['leg_id']==fill['leg_id'] and
                    self.closed[-1]['reason']=='SL' and self.closed[-1]['role']=='INITIAL')

    def market(self,event):
        rows=super().market(event)
        if self.mode=='AFTER_SL' and not self.chain_filled:
            for row in rows:
                if row['event']=='VIRTUAL_EXIT_INTENT' and row['role']=='INITIAL' and row['reason']=='SL':
                    self._reverse_wait=dict(leg_id=row['leg_id'],cycle_id=self.cycle_id)
        return rows

    def on_quote(self,event):
        rows=super().on_quote(event)
        if self.mode!='AFTER_SL' or self._reverse_wait is None:return rows
        now=stamp(event['time'])
        for row in list(rows):
            if row['event']!='VIRTUAL_SELL' or row['leg_id']!=self._reverse_wait['leg_id']:continue
            proof=dict(self._reverse_wait);self._reverse_wait=None
            if row['role']!='INITIAL' or row['reason']!='SL' or self.positions:
                raise RuntimeError('Unverified stop reversal source')
            if not self._session(now):
                rows.append(self._row('VIRTUAL_REVERSE_BLOCKED_SESSION',now,source_leg_id=row['leg_id']))
                return rows+self._finish(now)
            self.chain_filled=True # Reserve once; a timeout never starts a third leg.
            side=0 if row['symbol']==self.execution_config['BASE_ASSET_LONG'] else 1
            prediction=dict(model_id=self.model_id,inference_model_sha256=row['model_version'].split(':')[0],entry_reason='CONFIRMED_INITIAL_SL_FILL')
            self.pending.append(dict(action='BUY',role='CHAIN',time=now,side=side,sl_fill=proof,
                                     decision_id=row['decision_id'],model_version=row['model_version'],prediction=prediction))
            rows.append(self._row('VIRTUAL_REVERSE_INTENT',now,source_leg_id=row['leg_id'],
                                  source_exit_price=row['exit_price'],source_exit_time=row['exit_time'],
                                  symbol=self.execution_config['BASE_ASSET_LONG'] if side==1 else self.execution_config['BASE_ASSET_SHORT']))
        return rows+self._finish(now)


def build_comparison_shadows():
    registry=comparison_settings();view=LiveModelView()
    return {name:ComparisonCandidate(name,definition,view) for name,definition in registry['strategies'].items()}
