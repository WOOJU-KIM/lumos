import config
import pandas as pd

with open('scripts/run_weekly_rolling_wfa.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Feature Exclusion Fix
old_exclude = """    exclude_cols = {'open', 'high', 'low', 'close', 'volume', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'datetime', 'datetime_dt', 'date_str', 'time_str', 'week_id', 'target', 'date'}"""
new_exclude = """    exclude_cols = {'open', 'high', 'low', 'close', 'volume', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'datetime', 'datetime_dt', 'date_str', 'time_str', 'week_id', 'target', 'date', 'Target',
                    'cum_vp', 'vwap', 'EMA_9', 'EMA_21', 'EMA_50', 'EMA_200'}  # Aligned with ml_engine.py"""
content = content.replace(old_exclude, new_exclude)

# 2. Premarket Bug Fix
old_premarket = "if active_pos is None and time_str < config.PHASE_MAIN_END:"
new_premarket = "if active_pos is None and config.PHASE_MAIN_START <= time_str < config.PHASE_MAIN_END:"
content = content.replace(old_premarket, new_premarket)

# 3. Live Experience Merge & LGBM Params
old_train = """        clf = LGBMClassifier(
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
        )
        clf.fit(X_train, y_train)"""

new_train = """        import json
        import numpy as np
        
        # Base training weights (1.0)
        sample_weights = pd.Series(1.0, index=X_train.index)
        
        # Merge live_trades.csv dynamically (just like ml_engine.py)
        live_csv_path = config.DATA_DIR / 'live_trades.csv'
        if live_csv_path.exists():
            try:
                live_df = pd.read_csv(live_csv_path)
                # Prevent look-ahead bias: only use trades before the current test week
                max_train_dt_str = long_feat.loc[train_mask, 'datetime_dt'].max().strftime('%Y-%m-%d %H:%M:%S')
                live_df = live_df[pd.to_datetime(live_df['entry_time']) <= max_train_dt_str]
                
                live_rows = []
                live_targets = []
                for _, row in live_df.iterrows():
                    f_json = row.get('features_json')
                    gbdt_conf = float(row.get('gbdt_confidence', 0.0))
                    act_entry_px = float(row.get('actual_entry_price', 1.0))
                    
                    if act_entry_px <= 0.01 or not f_json or str(f_json).strip() in ('', '{}', 'nan') or gbdt_conf < 0.01:
                        continue
                    try:
                        f_dict = json.loads(f_json) if isinstance(f_json, str) else f_json
                        if not isinstance(f_dict, dict) or len(f_dict) < 10:
                            continue
                    except Exception:
                        continue
                        
                    sym = str(row.get('symbol', config.BASE_ASSET_LONG)).upper()
                    reason = str(row.get('exit_reason', '')).upper()
                    pnl = float(row.get('pnl_pct', 0.0))
                    
                    if sym == config.BASE_ASSET_LONG:
                        tgt = 2 if ('TP' in reason or pnl >= 2.0) else (0 if ('SL' in reason or pnl <= -1.5) else 1)
                    else:
                        tgt = 0 if ('TP' in reason or pnl >= 2.0) else (2 if ('SL' in reason or pnl <= -1.5) else 1)
                        
                    row_vals = {c: f_dict.get(c, np.nan) for c in feature_cols}
                    live_rows.append(row_vals)
                    live_targets.append(tgt)
                    
                if live_rows:
                    X_live = pd.DataFrame(live_rows).fillna(X_train.median())
                    y_live = pd.Series(live_targets, dtype=int)
                    weights_live = pd.Series(2.5, index=X_live.index)
                    
                    X_train = pd.concat([X_train, X_live], ignore_index=True)
                    y_train = pd.concat([y_train, y_live], ignore_index=True)
                    sample_weights = pd.concat([sample_weights, weights_live], ignore_index=True)
            except Exception as e:
                pass

        clf = LGBMClassifier(
            objective='multiclass',
            num_class=3,
            class_weight='balanced',
            n_estimators=config.GBDT_N_ESTIMATORS,
            max_depth=4,
            learning_rate=config.GBDT_LEARNING_RATE,
            random_state=42,
            verbosity=-1
        )
        clf.fit(X_train, y_train, sample_weight=sample_weights)"""
content = content.replace(old_train, new_train)

with open('scripts/run_weekly_rolling_wfa_refactored.py', 'w', encoding='utf-8') as f:
    f.write(content)
