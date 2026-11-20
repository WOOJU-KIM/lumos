"""Append-only candidate audit. No production database connections."""
from common import HOME
from datetime import datetime
import json
import time

PACKET_KEYS = {'kind','schema_version','time','decision_time','decision_id','net_feature_names','sequence',
               'full_features','feature_times','cross_asset_available_at','expected_last_closed_available_at',
               'required_assets_complete','atr5m_pct','cross_veto','trend_ok','relay_features'}


class Audit:
    def __init__(self, name):
        if not name.replace('_','').replace('-','').isalnum():
            raise ValueError('Unsafe run name')
        self.path=HOME/'runs'/name
        self.path.mkdir(parents=True,exist_ok=False)
        self.events=(self.path/'events.jsonl').open('x',encoding='utf-8')
        self.inputs=(self.path/'decision_inputs.jsonl').open('x',encoding='utf-8')
        self.max_write_ms=0.

    def packet(self, packet):
        start=time.monotonic()
        allowed={k:v for k,v in packet.items() if k in PACKET_KEYS}
        self.inputs.write(json.dumps(allowed,allow_nan=False)+'\n');self.inputs.flush()
        self.max_write_ms=max(self.max_write_ms,(time.monotonic()-start)*1000)

    def append(self, event, flush=True):
        start=time.monotonic()
        self.events.write(json.dumps(event,allow_nan=False)+'\n')
        if flush:self.events.flush()
        self.max_write_ms=max(self.max_write_ms,(time.monotonic()-start)*1000)

    def market(self,event):
        allowed={k:v for k,v in event.items() if k in {'kind','time','symbol','price','bid','ask','quote_time',
            'price_basis','observed_at','packet_type','protocol_version','reason'}}
        self.append(dict(event='MARKET_INPUT',**allowed),flush=False)

    def flush(self):
        start=time.monotonic()
        self.events.flush();self.inputs.flush()
        self.max_write_ms=max(self.max_write_ms,(time.monotonic()-start)*1000)

    def close(self):
        self.events.close();self.inputs.close()
