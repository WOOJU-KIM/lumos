import re

# Update run_weekly_rolling_wfa.py
with open('scripts/run_weekly_rolling_wfa.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_clf = '''        clf = LGBMClassifier(
            objective='multiclass',
            num_class=3,
            class_weight='balanced',
            n_estimators=config.GBDT_N_ESTIMATORS,
            max_depth=4,
            learning_rate=config.GBDT_LEARNING_RATE,
            random_state=42,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
            n_jobs=-1
        )'''

new_clf = '''        clf = LGBMClassifier(
            objective='multiclass',
            num_class=3,
            class_weight='balanced',
            n_estimators=config.GBDT_N_ESTIMATORS,
            max_depth=config.GBDT_MAX_DEPTH,
            learning_rate=config.GBDT_LEARNING_RATE,
            subsample=config.GBDT_FEATURE_FRACTION,
            colsample_bytree=config.GBDT_FEATURE_FRACTION,
            subsample_freq=1,
            random_state=42,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
            n_jobs=-1
        )'''

content = content.replace(old_clf, new_clf)

# Update calibration in WFA
old_calib = '''        final_p_long = 0.50 + (prob_long - 0.333) * 1.15'''
new_calib = '''        final_p_long = config.CALIB_BASE_CONF + (prob_long - config.CALIB_OFFSET) * config.CALIB_MULTIPLIER'''
content = content.replace(old_calib, new_calib)

with open('scripts/run_weekly_rolling_wfa.py', 'w', encoding='utf-8') as f:
    f.write(content)


# Update ml_engine.py
with open('core/ml_engine.py', 'r', encoding='utf-8') as f:
    content2 = f.read()

old_ml_clf = '''                subsample=config.GBDT_FEATURE_FRACTION,
                colsample_bytree=config.GBDT_FEATURE_FRACTION,
                deterministic=True,
                force_col_wise=True'''

new_ml_clf = '''                subsample=config.GBDT_FEATURE_FRACTION,
                colsample_bytree=config.GBDT_FEATURE_FRACTION,
                subsample_freq=1,
                deterministic=True,
                force_col_wise=True'''

content2 = content2.replace(old_ml_clf, new_ml_clf)

# Change MAX_TP_PCT to LABEL_TP_PCT in compute_triple_barrier_labels
content2 = content2.replace('take_profit: float = config.MAX_TP_PCT,', 'take_profit: float = getattr(config, "LABEL_TP_PCT", config.MAX_TP_PCT),')
content2 = content2.replace('take_profit=config.MAX_TP_PCT,', 'take_profit=getattr(config, "LABEL_TP_PCT", config.MAX_TP_PCT),')

with open('core/ml_engine.py', 'w', encoding='utf-8') as f:
    f.write(content2)
