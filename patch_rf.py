import re
with open('core/ml_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_fallback = """        except Exception:
            model = RandomForestClassifier(
                n_estimators=config.GBDT_N_ESTIMATORS,
                max_depth=5,
                class_weight='balanced',
                random_state=42
            )
            model.fit(X, y, sample_weight=sample_weights)
            importances = model.feature_importances_"""

new_fallback = """        except Exception as e:
            import logging
            logging.getLogger('Lumos').error(f"❌ [LGBM 학습 실패] {e}")
            raise"""

content = content.replace(old_fallback, new_fallback)

with open('core/ml_engine.py', 'w', encoding='utf-8') as f:
    f.write(content)
