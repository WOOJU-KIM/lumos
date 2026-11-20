"""Promote only the five isolated shadow artifacts after offline QA and own-worker stop."""
from common import *
guard(); verify_source_policy()
from adapters import Candidate

version='live_weighted_v1'
folder=HOME/'artifacts'/version
qa=json.loads((folder/'training_qa.json').read_text())
assert qa['status']=='PASS_NOT_YET_PROMOTED' and not qa['protected_changes']
assert qa['live_rows']>0 and qa['api_calls']==0
assert qa['source_sha256']==sha(HOME/'refit_live_weighted.py')
health=json.loads((HOME/'qa/feed_listener.json').read_text())
assert health['status']=='STOPPED','Stop only the shadow worker before replacing its models'
assert health.get('stats',{}).get('received',0)==0,'Do not reset an active shadow ledger'
for name in POLICY['models']:
    c=Candidate(name,folder/'manifest.json')
    assert c.bundle['live_experience_weight']==2.5
    assert sum(qa['fit_audit'][name]['live_rows_per_side'])==qa['live_rows']
current=HOME/'artifacts/manifest.json'
previous=folder/'previous_manifest.json'
with previous.open('xb') as stream:stream.write(current.read_bytes())
temp=folder/'promotion_manifest.tmp'
with temp.open('xb') as stream:stream.write((folder/'manifest.json').read_bytes())
os.replace(temp,current)
save('artifacts/'+version+'/deployment.json',dict(status='SHADOW_MANIFEST_PROMOTED',
    manifest_sha256=sha(current),previous_manifest_sha256=sha(previous),
    real_production_model_changed=False,real_runner_started=False,orders=0))
print('SHADOW_MANIFEST_PROMOTED; production model untouched',flush=True)
