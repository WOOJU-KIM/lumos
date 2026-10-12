import re
with open('core/ml_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_lgbm = """            model = LGBMClassifier(
                objective='multiclass',
                num_class=3,
                class_weight='balanced',
                n_estimators=config.GBDT_N_ESTIMATORS,
                max_depth=4,
                learning_rate=config.GBDT_LEARNING_RATE,
                random_state=42,
                verbosity=-1
            )"""

new_lgbm = """            model = LGBMClassifier(
                objective='multiclass',
                num_class=3,
                class_weight='balanced',
                n_estimators=config.GBDT_N_ESTIMATORS,
                max_depth=config.GBDT_MAX_DEPTH,
                learning_rate=config.GBDT_LEARNING_RATE,
                random_state=42,
                verbosity=-1,
                deterministic=True,
                force_col_wise=True
            )"""

content = content.replace(old_lgbm, new_lgbm)

with open('core/ml_engine.py', 'w', encoding='utf-8') as f:
    f.write(content)
