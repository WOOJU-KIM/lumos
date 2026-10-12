with open('scripts/run_weekly_rolling_wfa.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

with open('scripts/temp_run_wfa_oct_1_2.py', 'w', encoding='utf-8') as f:
    skip_lgb = False
    for line in lines:
        if 'for w_idx in range(test_start_idx, len(unique_weeks)):' in line:
            f.write('    for w_idx in range(len(unique_weeks) - 1, len(unique_weeks)):\n')
        elif 'clf = LGBMClassifier(' in line:
            skip_lgb = True
            f.write('        # Use MLFeatureEngine exactly like live trading, including live_trades.csv up to cutoff\n')
            f.write('        max_train_dt = long_feat.loc[train_mask, "datetime_dt"].max()\n')
            f.write('        train_df = long_15m[pd.to_datetime(long_15m["datetime"]) <= max_train_dt]\n')
            f.write('        ml_live = MLFeatureEngine(confidence_threshold=config.GBDT_CONFIDENCE_THRESHOLD)\n')
            f.write('        # Load live_trades.csv but strictly cutoff before test week\n')
            f.write('        live_df = pd.read_csv("data/live_trades.csv")\n')
            f.write('        live_df = live_df[pd.to_datetime(live_df["entry_time"]) <= max_train_dt]\n')
            f.write('        clf, _, _ = ml_live.train_and_select_top_features(train_df, live_experience_df=live_df)\n')
        elif skip_lgb and 'clf.fit(X_train, y_train)' in line:
            skip_lgb = False
        elif 'X_test = test_df[feature_cols].fillna(0.0)' in line:
            f.write('        X_test = test_df[clf.feature_name_].fillna(0.0)\n')
        elif 'if active_pos is None and time_str < config.PHASE_MAIN_END:' in line:
            f.write('                if active_pos is None and config.PHASE_MAIN_START <= time_str < config.PHASE_MAIN_END:\n')
        elif 'pass_cross_short = (cross_dir != "LONG") if config.USE_CROSS_ASSET_VETO else True' in line:
            f.write(line)
            f.write('                    # Debug Print added\n')
            f.write('                    print(f"[{curr_dt_str}] Dir: {dir_gbdt} | Conf: {conf_gbdt:.3f} | 60m_bull={qqq_60m_bull}, 60m_bear={qqq_60m_bear} | Cross: {cross_dir}")\n')
        elif 'df_trades = pd.DataFrame(trades)' in line:
            f.write('    df_trades = pd.DataFrame(trades)\n')
            f.write('    print("\\n=== Trades for the target week ===")\n')
            f.write('    for t in trades:\n')
            f.write('        print(f"Entry: {t[\'entry_dt\']} | Exit: {t[\'exit_dt\']} | Sym: {t[\'symbol\']} | Entry Px: {t[\'entry_px\']:.2f} | Ret: {t[\'net_ret_pct\']}% | Reason: {t[\'exit_reason\']}")\n')
            f.write('    print("====================================\\n")\n')
        else:
            if not skip_lgb:
                f.write(line)
