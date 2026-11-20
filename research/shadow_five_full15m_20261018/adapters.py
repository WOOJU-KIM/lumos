"""One output contract for five independently versioned candidate policies."""
from common import HOME, sha
import json
import pickle
import numpy as np


class Candidate:
    def __init__(self, name, manifest_path=None):
        # Explicit candidate manifests allow QA before changing the shadow deployment.
        manifest = json.loads((manifest_path or HOME/'artifacts/manifest.json').read_text())
        if manifest['status']!='TRAINED_NOT_CONNECTED':
            raise RuntimeError('Candidates quarantined pending dataset coverage QA')
        if manifest['policy_sha256']!=sha(HOME/'policy.json'):
            raise ValueError('Policy changed after training; create a new version')
        entry = manifest['models'][name]
        path = HOME/'artifacts'/entry['path']
        if sha(path) != entry['artifact_sha256']:
            raise ValueError('Candidate artifact integrity mismatch')
        with path.open('rb') as stream:
            self.bundle = pickle.load(stream)
        if self.bundle['source_policy_sha']!=manifest['policy_sha256']:
            raise ValueError('Model/policy version mismatch')
        self.name = name
        self.version = entry['artifact_sha256']
        self.bundle['feature_names'] = list(self.bundle['feature_names'])
        self.net = None
        if self.bundle['kind'] == 'NET_TCN':
            from network import torch, Net
            torch.set_num_threads(1)
            weights = HOME/'artifacts'/self.bundle['weights']
            if sha(weights) != self.bundle['weights_sha256']:
                raise ValueError('TCN weights integrity mismatch')
            self.net = Net([0., 0.])
            self.net.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
            self.net.eval()

    def predict(self, packet):
        b = self.bundle
        if b['kind'].startswith('NET_'):
            if packet['net_feature_names'] != b['feature_names']:
                raise ValueError('72-feature name/order mismatch')
            seq = np.asarray(packet['sequence'], dtype=np.float32)
            if seq.shape != (32, 72) or not np.isfinite(seq).all():
                raise ValueError('Invalid32x72 input')
            if b['kind'] == 'NET_TCN':
                from network import torch
                scaled = np.clip((seq-b['mean'])/b['std'], -8, 8).astype(np.float32)
                with torch.no_grad():
                    raw = self.net(torch.from_numpy(scaled.T[None, ...])).numpy()[0]/100
            else:
                X = seq[-1:,:] if b['kind']=='NET_GBDT' else seq[::-1,:].reshape(1,-1)
                raw = np.array([m.predict(X)[0]/100 for m in b['models']])
            ev = raw + b['bias']
            eligible = np.array([True, True])
            details = dict(raw_net_return=raw.tolist(), residual_adjustment=b['bias'].tolist())
        else:
            f = packet['full_features']
            slope, dist, dev, body = [float(f[k]) for k in ['EMA20_Slope_3','EMA20_Dist_ATR','VWAP_Dist_ATR','Signed_Body_Ratio']]
            trend = 1 if slope>0 and dist>0 else 0 if slope<0 and dist<0 else -1
            mr = 1 if dev<=-.5 and body>0 else 0 if dev>=.5 and body<0 else -1
            rows, groups, eligible = [], [], []
            for side in [0, 1]:
                tf, mf = int(trend==side), int(mr==side)
                row = dict(f, action_side=side, trend_available=tf, mr_available=mf)
                rows.append([row[k] for k in b['feature_names']])
                groups.append(side*4+tf*2+mf)
                eligible.append(bool(tf or mf))
            X = np.asarray(rows, dtype=float)
            if not np.isfinite(X).all():
                raise ValueError('Non-finite selector features')
            if list(b['models'][0].feature_name_) == b['feature_names']:
                import pandas as pd
                X = pd.DataFrame(X, columns=b['feature_names'])
            raw_prob = b['models'][0].predict_proba(X)[:,1]
            p = np.clip(raw_prob, 1e-6, 1-1e-6)
            p = b['platt'].predict_proba(np.log(p/(1-p)).reshape(-1,1))[:,1]
            if b['kind']=='HIST':
                win = np.array([b['means'][g][0] for g in groups])
                loss = np.array([b['means'][g][1] for g in groups])
            else:
                win = np.maximum(0,b['models'][1].predict(X))
                loss = np.maximum(0,b['models'][2].predict(X))
            ev = p*win-(1-p)*loss
            eligible = np.asarray(eligible)
            details = dict(raw_win_probability=raw_prob.tolist(), calibrated_win_probability=p.tolist(),
                           expected_win=win.tolist(), expected_loss=loss.tolist(),
                           entry_strategy_eligible=eligible.tolist(), calibration_domain=eligible.tolist())
        if not np.isfinite(ev).all():
            raise ValueError('Non-finite prediction')
        primary = np.where(eligible, ev, -np.inf)
        side = int(np.argmax(primary))
        accepted = bool(primary[side]>0 and primary[side]>=b['threshold'] and primary[0]!=primary[1])
        forced = int(np.argmax(ev)) if ev[0]!=ev[1] else -1
        return dict(model_id=self.name, model_version=self.version, expected_net_return=ev.tolist(),
                    threshold=float(b['threshold']), score_type='expected_net_return_fraction',
                    side=side if accepted else -1, forced_side=forced, details=details)
