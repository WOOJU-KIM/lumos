import re

with open('scripts/run_weekend_data_refresh.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_call = """    labels = MLFeatureEngine.compute_triple_barrier_labels(
        feat_df,
        take_profit=config.MAX_TP_PCT,
        stop_loss=config.SL_MIN_PCT,
        horizon=6
    )"""

new_call = """    labels = MLFeatureEngine.compute_triple_barrier_labels(
        feat_df,
        take_profit=config.LABEL_TP_PCT,
        stop_loss=config.SL_MIN_PCT,
        horizon=int(config.TIME_STOP_MINUTES / 15)
    )"""

content = content.replace(old_call, new_call)

with open('scripts/run_weekend_data_refresh.py', 'w', encoding='utf-8') as f:
    f.write(content)
