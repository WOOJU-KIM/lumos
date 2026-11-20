"""Paired threshold diagnostic using the exact activated post-retraining artifact.

Reuse the tested execution simulator, not its weekly training loop. This is
explicitly an in-sample historical replay, never represented as OOS WFA.
"""
import ast
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))

replacement=ast.parse('''
import pickle
active_manifest_path=PROJECT/'models/active_model_manifest.json'
active_manifest=json.loads(active_manifest_path.read_text(encoding='utf-8'))
active_artifact=PROJECT/active_manifest['model_path']
assert active_manifest['model_path']==active_manifest['active_model_path']
assert sha(active_artifact)==active_manifest['expected_sha256']=='0deb9282eab7e1bf1a5b2d694878f13534ea0cc2632f9b57676d46d906b24932'
with active_artifact.open('rb') as stream:active_bundle=pickle.load(stream)
assert active_bundle['feature_schema']=='B_normalized'
assert active_bundle['feature_names']==NAMES
model=active_bundle['model']
assert list(model.classes_)==[0,1,2] and model.n_features_in_==72
assert list(model.feature_name_)==NAMES
assert active_manifest['live_experience_rows']==17 and active_manifest['live_experience_weight']==2.5
assert np.all(D.iloc[ids].feature_available_at.to_numpy()<=T[ids].to_numpy())
prob[ids]=model.predict_proba(F.iloc[ids][NAMES])
assert np.isfinite(prob[ids]).all() and np.allclose(prob[ids].sum(axis=1),1.)
assert np.array_equal(prob[ids],model.predict_proba(F.iloc[ids][NAMES]))
folds=[]
print('EXACT_ACTIVATED_ARTIFACT_LOADED; NO_TRAINING; IN_SAMPLE_REPLAY',flush=True)
''').body

metadata=ast.parse('''
comparison.update(method='FIXED_ACTIVATED_POST_LIVE17_MODEL_IN_SAMPLE_REPLAY',
    distinction='Exact current production artifact evaluated on dates used in its training; NOT OOS and NOT WFA',
    active_artifact_sha256=sha(active_artifact),active_manifest_sha256=sha(active_manifest_path),
    active_artifact_path=active_manifest['model_path'],active_candidate_id=active_bundle['model_id'],
    live_training_rows=17,live_training_weight=2.5,training_cutoff=active_bundle['cutoff'],
    training_rows=active_bundle['training_window_meta']['train_rows'],retraining_performed=False,
    training_overlap=True,approved_as_performance_evidence=False,
    simulator_sha256=sha(HOME/'threshold_64_704.py'))
comparison['limitations'].insert(0,'IN_SAMPLE: model trained through Oct16, including these evaluation dates and 17 actual outcomes; cannot infer future performance')
''').body

tree=ast.parse((HERE/'threshold_64_704.py').read_text(encoding='utf-8'))
class ActiveReplay(ast.NodeTransformer):
    replaced=0
    def visit_For(self,node):
        if isinstance(node.target,ast.Name) and node.target.id=='week':
            self.replaced+=1
            return replacement
        return self.generic_visit(node)
    def visit_Constant(self,node):
        if node.value=='threshold_64_704_20261018':return ast.copy_location(ast.Constant('threshold_64_704_live17_20261018'),node)
        if node.value=='기존 B72 학습 방식: 64% vs 70.4% | 주간 워크포워드':
            return ast.copy_location(ast.Constant('교체 후 B72 실전 가중 모델 | 64% vs 70.4% | 학습기간 내 재현'),node)
        return node

transform=ActiveReplay();tree=transform.visit(tree)
assert transform.replaced==1,'Exactly one training loop must be removed'
body=[]
for node in tree.body:
    body.append(node)
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='comparison' for t in node.targets):body.extend(metadata)
tree.body=body
scope={'__file__':str(HERE/'threshold_64_704.py'),'__name__':'__main__'}
exec(compile(ast.fix_missing_locations(tree),'EXACT_ACTIVE_MODEL_THRESHOLD_REPLAY','exec'),scope)

# Reconstruct the chart from recorded trades and verify its values match metrics.
plot=(HERE/'plot_threshold_64_704.py').read_text(encoding='utf-8')
plot=plot.replace("guard()", "# Isolation guard is already installed by the comparison")
plot=plot.replace('threshold_64_704_20261018','threshold_64_704_live17_20261018')
plot=plot.replace('기존 B72 학습 방식: 64% vs 70.4% | 주간 워크포워드','교체 후 B72 실전 가중 모델 | 64% vs 70.4% | 학습기간 내 재현')
exec(compile(plot,'EXACT_ACTIVE_REPLAY_CHART','exec'),scope)
