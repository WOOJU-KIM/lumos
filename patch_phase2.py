import re

# 1. ml_engine.py fixes
with open('core/ml_engine.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix model loading order
load_old = """        for fname in ["model_main_data_refresh.pkl", "model_champion.pkl", "model_moe_orchestrator.pkl"]:"""
load_new = """        for fname in ["model_champion.pkl", "model_moe_orchestrator.pkl", "model_main_data_refresh.pkl"]:"""
content = content.replace(load_old, load_new)

# Fix except Exception: pass in model loading
except_old = """                except Exception:
                    pass"""
except_new = """                except Exception as e:
                    import logging
                    logging.getLogger('Lumos').error(f"❌ [모델 로드 에러] {fname} 로드 실패: {e}")"""
content = content.replace(except_old, except_new)

# Fix label computation (Look-ahead bias & concurrent touch priority)
label_old = """        for i in range(n):
            p0 = closes[i]
            if p0 <= 0:
                continue
            end_idx = min(i + horizon + 1, n)
            if i + 1 >= end_idx:
                continue

            assigned = False
            for j in range(i + 1, end_idx):
                h_ret = (highs[j] - p0) / p0
                l_ret = (lows[j] - p0) / p0

                # 롱 배리어 (+3.0% TP vs -2.0% SL)
                hit_long_tp = (h_ret >= take_profit)
                hit_long_sl = (l_ret <= -stop_loss)

                # 숏 배리어 (-3.0% TP for short vs +2.0% SL for short)
                hit_short_tp = (l_ret <= -take_profit)
                hit_short_sl = (h_ret >= stop_loss)

                if hit_long_tp and not hit_long_sl:
                    labels[i] = 1   # TQQQ 승리
                    assigned = True
                    break
                elif hit_short_tp and not hit_short_sl:
                    labels[i] = -1  # SQQQ 승리
                    assigned = True
                    break
                elif (hit_long_sl and hit_short_sl) or (hit_long_tp and hit_long_sl) or (hit_short_tp and hit_short_sl):
                    labels[i] = 0   # 동시 급변동은 회피
                    assigned = True
                    break

            if not assigned:
                labels[i] = 0"""

label_new = """        for i in range(n):
            p0 = closes[i]
            if p0 <= 0: continue
            end_idx = min(i + horizon + 1, n)
            
            long_label = 0
            short_label = 0
            
            for j in range(i + 1, end_idx):
                h_ret = (highs[j] - p0) / p0
                l_ret = (lows[j] - p0) / p0
                
                if long_label == 0:
                    hit_long_tp = (h_ret >= take_profit)
                    hit_long_sl = (l_ret <= -stop_loss)
                    if hit_long_sl:
                        long_label = -999 
                    elif hit_long_tp:
                        long_label = 1
                        
                if short_label == 0:
                    hit_short_tp = (l_ret <= -take_profit)
                    hit_short_sl = (h_ret >= stop_loss)
                    if hit_short_sl:
                        short_label = -999 
                    elif hit_short_tp:
                        short_label = -1
                        
                if long_label != 0 and short_label != 0:
                    break
                    
            if long_label == 1 and short_label == -1:
                labels[i] = 0 
            elif long_label == 1:
                labels[i] = 1
            elif short_label == -1:
                labels[i] = -1
            else:
                labels[i] = 0"""

content = content.replace(label_old, label_new)

# Fix hardcoded values in extract_features calling compute_triple_barrier_labels
feat_target_old = """        feat_df['Target'] = self.compute_triple_barrier_labels(
            feat_df,
            take_profit=0.030,
            stop_loss=config.SL_MIN_PCT,
            horizon=6
        )"""

feat_target_new = """        feat_df['Target'] = self.compute_triple_barrier_labels(
            feat_df,
            take_profit=config.MAX_TP_PCT,
            stop_loss=config.SL_MIN_PCT,
            horizon=int(config.TIME_STOP_MINUTES / 15)
        )"""
content = content.replace(feat_target_old, feat_target_new)

# Fix deterministic training
train_old = """            model = lgb.LGBMClassifier(
                n_estimators=config.GBDT_N_ESTIMATORS,
                learning_rate=config.GBDT_LEARNING_RATE,
                max_depth=config.GBDT_MAX_DEPTH,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                class_weight='balanced'
            )"""

train_new = """            model = lgb.LGBMClassifier(
                n_estimators=config.GBDT_N_ESTIMATORS,
                learning_rate=config.GBDT_LEARNING_RATE,
                max_depth=config.GBDT_MAX_DEPTH,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                class_weight='balanced',
                deterministic=True,
                force_col_wise=True
            )"""
content = content.replace(train_old, train_new)

with open('core/ml_engine.py', 'w', encoding='utf-8') as f:
    f.write(content)


# 2. data_lake.py fixes (Fake candle)
with open('core/data_lake.py', 'r', encoding='utf-8') as f:
    dl = f.read()

fake_candle_old = """        if last_dt == cur_candle_dt:
            # 기존 마지막 캔들 업데이트
            df_copy.loc[last_dt, 'Close'] = live_price
            df_copy.loc[last_dt, 'High'] = max(df_copy.loc[last_dt, 'High'], live_price)
            df_copy.loc[last_dt, 'Low'] = min(df_copy.loc[last_dt, 'Low'], live_price)
        else:
            # 새로운 형성 중인 캔들 추가
            new_row = pd.DataFrame([{
                'datetime': cur_candle_dt.strftime('%Y-%m-%d %H:%M:%S'),
                'Open': live_price,
                'High': live_price,
                'Low': live_price,
                'Close': live_price,
                'Volume': 100.0
            }], index=[cur_candle_dt])
            df_copy = pd.concat([df_copy, new_row])"""

fake_candle_new = """        if last_dt == cur_candle_dt:
            # 기존 마지막 캔들 업데이트
            df_copy.loc[last_dt, 'Close'] = live_price
            df_copy.loc[last_dt, 'High'] = max(df_copy.loc[last_dt, 'High'], live_price)
            df_copy.loc[last_dt, 'Low'] = min(df_copy.loc[last_dt, 'Low'], live_price)
        else:
            # 형성 중인 캔들이 아직 수집되지 않은 경우 가짜 캔들을 만들지 않고 이전 종가로 대체하지도 않음.
            # 머신러닝 피처 훼손 방지를 위해 그대로 반환.
            pass"""
dl = dl.replace(fake_candle_old, fake_candle_new)

with open('core/data_lake.py', 'w', encoding='utf-8') as f:
    f.write(dl)

