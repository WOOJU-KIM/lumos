import re

with open('scripts/run_weekly_rolling_wfa.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Update WFA tp_pct
content = content.replace('labels = ml_engine.compute_triple_barrier_labels(long_feat)', 'labels = ml_engine.compute_triple_barrier_labels(long_feat, take_profit=getattr(config, "LABEL_TP_PCT", config.MAX_TP_PCT))')
with open('scripts/run_weekly_rolling_wfa.py', 'w', encoding='utf-8') as f:
    f.write(content)

with open('core/weekly_tournament.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Update tournament tp_pct
content = content.replace('take_profit=config.MAX_TP_PCT', 'take_profit=getattr(config, "LABEL_TP_PCT", config.MAX_TP_PCT)')
with open('core/weekly_tournament.py', 'w', encoding='utf-8') as f:
    f.write(content)
