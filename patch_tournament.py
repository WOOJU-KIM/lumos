import re

with open('core/weekly_tournament.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Fix default params in _backtest_moe_candidate
old_def = """    def _backtest_moe_candidate(
        self,
        moe_obj: MoEMetaOrchestrator,
        cache: Dict[str, Any],
        tp_pct: float = 0.030,
        sl_pct: float = -0.020
    ) -> Dict[str, Any]:"""
new_def = """    def _backtest_moe_candidate(
        self,
        moe_obj: MoEMetaOrchestrator,
        cache: Dict[str, Any],
        tp_pct: float = config.MAX_TP_PCT,
        sl_pct: float = -config.SL_MIN_PCT
    ) -> Dict[str, Any]:"""
content = content.replace(old_def, new_def)

# 2. Fix constants in _backtest_moe_candidate
old_consts = """        TIME_STOP_BARS_5M = 18  # 90분 (5분봉 18개)
        SLIPPAGE_PAYUP = 0.03
        FEE_RATE = 0.0020"""
new_consts = """        TIME_STOP_BARS_5M = int(config.TIME_STOP_MINUTES / 5)
        SLIPPAGE_PAYUP = config.SLIPPAGE_PAYUP if hasattr(config, "SLIPPAGE_PAYUP") else 0.03
        FEE_RATE = config.BACKTEST_FEE_SLIPPAGE"""
content = content.replace(old_consts, new_consts)

# 3. Remove 3-Out circuit breaker
old_3out_1 = """                if daily_stoploss_count >= 3:
                    b_idx += 1
                    continue"""
new_3out_1 = ""
content = content.replace(old_3out_1, new_3out_1)

# 4. Fix TP/SL hit logic
old_hit = """                    if hit_tp and hit_sl:
                        if c5_c >= c5_o:
                            exit_triggered = True
                            exit_px = round(tp_px - SLIPPAGE_PAYUP, 2)
                            exit_reason = f"TP(+{TP_PCT*100:.1f}%)"
                            break
                        else:
                            exit_triggered = True
                            exit_px = round(sl_px - SLIPPAGE_PAYUP, 2)
                            exit_reason = f"SL({SL_PCT*100:.1f}%)"
                            daily_stoploss_count += 1
                            break"""

new_hit = """                    if hit_tp and hit_sl:
                        # 손절 우선주의 (가장 비관적 처리)
                        exit_triggered = True
                        exit_px = round(sl_px - SLIPPAGE_PAYUP, 2)
                        exit_reason = f"SL({SL_PCT*100:.1f}%)"
                        daily_stoploss_count += 1
                        break"""
content = content.replace(old_hit, new_hit)

# 5. Remove deterministic violation in LightGBM and fix TimeSeriesSplit evaluation on same data
# In _train_candidate_3_hyperparameter_tuned:
old_lgb = """                clf = LGBMClassifier(
                    objective='multiclass',
                    num_class=3,
                    class_weight='balanced',
                    n_estimators=100,
                    random_state=42,
                    verbose=-1,
                    **params
                )"""
new_lgb = """                clf = LGBMClassifier(
                    objective='multiclass',
                    num_class=3,
                    class_weight='balanced',
                    n_estimators=config.GBDT_N_ESTIMATORS,
                    random_state=42,
                    verbose=-1,
                    deterministic=True,
                    force_col_wise=True,
                    **params
                )"""
content = content.replace(old_lgb, new_lgb)

old_lgb_final = """        best_model = LGBMClassifier(
            objective='multiclass',
            num_class=3,
            class_weight='balanced',
            n_estimators=100,
            random_state=42,
            verbose=-1,
            **best_params
        )"""
new_lgb_final = """        best_model = LGBMClassifier(
            objective='multiclass',
            num_class=3,
            class_weight='balanced',
            n_estimators=config.GBDT_N_ESTIMATORS,
            random_state=42,
            verbose=-1,
            deterministic=True,
            force_col_wise=True,
            **best_params
        )"""
content = content.replace(old_lgb_final, new_lgb_final)

# Target labels were hardcoded to 0.030 / 0.020 / 6
old_target = """        target_series = MLFeatureEngine.compute_triple_barrier_labels(
            df,
            take_profit=0.030,
            stop_loss=0.020,
            horizon=6
        )"""
new_target = """        target_series = MLFeatureEngine.compute_triple_barrier_labels(
            df,
            take_profit=config.MAX_TP_PCT,
            stop_loss=config.SL_MIN_PCT,
            horizon=int(config.TIME_STOP_MINUTES / 15)
        )"""
content = content.replace(old_target, new_target)

with open('core/weekly_tournament.py', 'w', encoding='utf-8') as f:
    f.write(content)
