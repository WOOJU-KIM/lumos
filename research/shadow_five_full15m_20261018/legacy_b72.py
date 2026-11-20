"""Frozen pre-relay B72 strategy, virtual orders only; no live application imports."""
import json
import joblib
import numpy as np
import pandas as pd
from joblib.numpy_pickle import NumpyUnpickler
from joblib.numpy_pickle_utils import _validate_fileobject_and_memmap
from common import HOME, CONFIG, sha


class _StoredState:
    def __setstate__(self, state):
        self.__dict__.update(state)


class _ObserverUnpickler(NumpyUnpickler):
    def find_class(self, module, name):
        if module.startswith('core.'):
            return _StoredState
        return super().find_class(module, name)


def _load_bundle(path):
    with path.open('rb') as stream:
        with _validate_fileobject_and_memmap(stream, str(path), None) as (fileobj, mmap_mode):
            obj = _ObserverUnpickler(str(path), fileobj, ensure_native_byte_order=True, mmap_mode=mmap_mode).load()
    if isinstance(obj, dict):
        return obj['model'], list(obj['feature_names'])
    engine = getattr(obj, 'gbdt_engine', obj)
    return engine.model, list(engine.feature_names)


class LegacyB72:
    cohorts = ('primary',)

    def __init__(self, settings_override=None):
        self.settings = (dict(settings_override) if settings_override is not None else
                         json.loads((HOME/'legacy_strategy.json').read_text()))
        if not self.settings['enabled']:
            raise RuntimeError('Legacy strategy disabled')
        self.name = self.settings['strategy_id']
        path = HOME/self.settings['artifact_path']
        if sha(path) != self.settings['artifact_sha256']:
            raise RuntimeError('Legacy B72 artifact hash mismatch')
        self.model, self.features = _load_bundle(path)
        if self.features != self.settings['feature_names'] or len(self.features) != 72:
            raise RuntimeError('Legacy B72 feature schema mismatch')
        if list(self.model.classes_) != [0, 1, 2]:
            raise RuntimeError('Legacy B72 class mapping mismatch')
        if (self.settings['allocation'] != CONFIG['MAX_ALLOCATION_RATIO'] or
                self.settings['trailing_trigger'] != CONFIG['TRAILING_TRIGGER_PCT'] or
                self.settings['trailing_drop'] != CONFIG['TRAILING_DROP_PCT'] or
                self.settings['initial_stop'] != CONFIG['SL_MIN_PCT'] or
                self.settings['threshold'] != CONFIG['GBDT_CONFIDENCE_THRESHOLD'] or
                self.settings['time_stop_minutes'] != CONFIG['TIME_STOP_MINUTES'] or
                self.settings['max_tp'] != CONFIG['MAX_TP_PCT']):
            raise RuntimeError('Legacy execution recipe mismatch')
        self.version = self.settings['artifact_sha256']

    def predict(self, packet):
        values = dict(packet['full_features'], **packet.get('relay_features', {}))
        x = pd.DataFrame([[float(values[name]) for name in self.features]], columns=self.features)
        if not np.isfinite(x.to_numpy()).all():
            raise ValueError('Missing legacy B72 input')
        ps, pn, pl = map(float, self.model.predict_proba(x)[0])
        score = min(.95, max(CONFIG['CALIB_BASE'], CONFIG['CALIB_BASE'] +
                            (max(ps, pl) - CONFIG['CALIB_PIVOT']) * CONFIG['CALIB_SCALE']))
        side = 1 if pl > ps and pl > pn else 0 if ps > pl and ps > pn else -1
        if score < self.settings['threshold']:
            side = -1
        return dict(model_id=self.name, model_version=self.version, side=side, forced_side=side,
                    calibrated_confidence=score, raw_probabilities=[ps, pn, pl],
                    execution_strategy=self.name)
