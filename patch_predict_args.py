import os
from pathlib import Path

fpath = Path('core/moe_orchestrator.py')
content = fpath.read_text(encoding='utf-8')

target = "gbdt_sig, gbdt_conf, _ = self.gbdt_engine.predict_signal(live_tqqq_15m, live_prices=live_prices)"
replacement = "gbdt_sig, gbdt_conf, _ = self.gbdt_engine.predict_signal(live_tqqq_15m)"

content = content.replace(target, replacement)
fpath.write_text(content, encoding='utf-8')
