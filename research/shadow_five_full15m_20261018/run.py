"""Explicit recorded-file replay only. No daemon, broker subscription or automatic startup."""
from common import *
guard()
verify_source_policy()
import argparse
from datetime import datetime
from adapters import Candidate
from runtime import Engine, validate_packet
from storage import Audit

parser=argparse.ArgumentParser()
parser.add_argument('--check',action='store_true')
parser.add_argument('--replay',type=Path)
args=parser.parse_args()
if not (HOME/'artifacts/manifest.json').exists():
    print(json.dumps(dict(status='TRAINING_OR_QA_PENDING',runtime_enabled=False,live_ready=False)))
    raise SystemExit(0 if args.check or not args.replay else 2)
manifest=json.loads((HOME/'artifacts/manifest.json').read_text())
if args.check or not args.replay:
    activation=json.loads((HOME/'activation.json').read_text())
    from feed_bridge import connected
    print(json.dumps(dict(models=list(manifest['models']),status=manifest['status'],
                          artifact_training_status_frozen=True,listener_available=connected(),
                          production_hook=activation.get('production_hook_installed',False),
                          observed_feed_status=json.loads((HOME/'qa/feed_listener.json').read_text()) if (HOME/'qa/feed_listener.json').exists() else None,
                          live_ready=False)))
    raise SystemExit(0)
engine=Engine({name:Candidate(name) for name in POLICY['models']})
audit=Audit('recorded_replay_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
count=0
try:
    with args.replay.open(encoding='utf-8-sig') as stream:
        for line in stream:
            if len(line)>2_000_000:
                raise ValueError('Event too large')
            event=json.loads(line)
            if event['kind']=='DECISION':
                validate_packet(event)
                audit.packet(event)
            else:
                audit.market(event)
            for output in engine.handle(event):
                audit.append(output)
            count+=1
finally:
    audit.close()
print(json.dumps(dict(recorded_events=count,model_errors=len(engine.errors),
                      closed_primary={name:len(engine.ledgers[(name,'primary')].closed) for name in POLICY['models']},
                      unresolved_primary={name:dict(open=len(engine.ledgers[(name,'primary')].positions),pending=len(engine.ledgers[(name,'primary')].pending)) for name in POLICY['models']},
                      execution='assumed next bid/ask; not certified real fills',network_calls=0)))
