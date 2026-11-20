"""Finite cross-process memory transport test; no models, broker or live runner."""
import json
import time
from pathlib import Path
from feed_bridge import Reader
home=Path(__file__).parent
reader=Reader('Local\\LumosShadowCrossProcessQA')
(home/'qa/feed_receiver_ready.json').write_text(json.dumps({'ready':True}))
start=time.monotonic();count=0;seq=0
while time.monotonic()-start<8:
    event=reader.read()
    if event is not None:
        count+=1;seq=event[0]
    else:time.sleep(.001)
reader.close()
(home/'qa/feed_cross_process.json').write_text(json.dumps(dict(status='PASS' if count else 'FAIL',events=count,last_sequence=seq)))
print(json.dumps(dict(events=count)))
