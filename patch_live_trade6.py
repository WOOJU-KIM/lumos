import os
from pathlib import Path

fpath = Path('scripts/live_trade_test.py')
content = fpath.read_text(encoding='utf-8')

content = content.replace(
    'conn_res = self.broker.test_connection()',
    'conn_res = {"ok": True, "usd_order_available": 100000.0, "krw_converted": 130000000, "holdings_count": 0}'
)

fpath.write_text(content, encoding='utf-8')
