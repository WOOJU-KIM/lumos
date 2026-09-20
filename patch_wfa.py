import re

with open('scripts/run_weekly_rolling_wfa.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Imports
content = content.replace("from config import DATA_DIR, BASE_DIR, MODELS_DIR", "import config\nfrom config import DATA_DIR, BASE_DIR, MODELS_DIR")

# 2. Data load
old_load = """    tqqq_15m = lake.load_candles("TQQQ", "15m")
    sqqq_15m = lake.load_candles("SQQQ", "15m")
    soxx_60m = lake.load_candles("SOXX", "60m")
    nvda_15m = lake.load_candles("NVDA", "15m")
    soxx_15m = lake.load_candles("SOXX", "15m")
    qqq_15m  = lake.load_candles("QQQ", "15m")
    vixy_15m = lake.load_candles("VIXY", "15m")
    ief_15m  = lake.load_candles("IEF", "15m")

    # 60m 20EMA
    soxx_60m['ema20'] = soxx_60m['Close'].ewm(span=20, adjust=False).mean()"""

new_load = """    tqqq_15m = lake.load_candles("TQQQ", "15m")
    sqqq_15m = lake.load_candles("SQQQ", "15m")
    qqq_60m = lake.load_candles("QQQ", "60m")
    nvda_15m = lake.load_candles("NVDA", "15m")
    soxx_15m = lake.load_candles("SOXX", "15m")
    qqq_15m  = lake.load_candles("QQQ", "15m")
    vixy_15m = lake.load_candles("VIXY", "15m")
    ief_15m  = lake.load_candles("IEF", "15m")

    # 60m QQQ EMA
    qqq_60m['ema_filter'] = qqq_60m['Close'].ewm(span=config.QQQ_EMA_PERIOD, adjust=False).mean()"""
content = content.replace(old_load, new_load)

# 3. Exit logic
old_exit = """                    tp_px = entry_px * 1.030
                    sl_px = entry_px * 0.980

                    if cur_high >= tp_px:
                        exit_price = tp_px
                        exit_reason = "TAKE_PROFIT_3PCT"
                    elif cur_low <= sl_px:
                        exit_price = sl_px
                        exit_reason = "STOP_LOSS_2PCT"
                    elif active_pos['bars'] >= 6:
                        exit_price = cur_close
                        exit_reason = "TIME_STOP_90MIN"
                    elif time_str >= '15:45':
                        exit_price = cur_close
                        exit_reason = "EOD_MARKET_CLOSE\""""

new_exit = """                    if 'peak_high' not in active_pos:
                        active_pos['peak_high'] = cur_high
                    else:
                        active_pos['peak_high'] = max(active_pos['peak_high'], cur_high)

                    tp_px = entry_px * (1.0 + config.MAX_TP_PCT)
                    sl_px = entry_px * (1.0 - config.SL_MIN_PCT)

                    current_sl_px = sl_px
                    trailing_trigger_px = entry_px * (1.0 + config.TRAILING_TRIGGER_PCT)
                    if active_pos['peak_high'] >= trailing_trigger_px:
                        safety_sl_px = entry_px * (1.0 + max(0.0, config.TRAILING_TRIGGER_PCT - 0.015))
                        current_sl_px = max(safety_sl_px, active_pos['peak_high'] * (1.0 - config.TRAILING_DROP_PCT))

                    if cur_high >= tp_px:
                        exit_price = tp_px
                        exit_reason = "MAX_TP"
                    elif cur_low <= current_sl_px:
                        exit_price = current_sl_px
                        exit_reason = "TRAILING_SL" if current_sl_px > sl_px else "STOP_LOSS"
                    elif active_pos['bars'] * 15 >= config.TIME_STOP_MINUTES:
                        exit_price = cur_close
                        exit_reason = "TIME_STOP"
                    elif time_str >= config.PHASE_EOD_CLEAR:
                        exit_price = cur_close
                        exit_reason = "EOD_CLEAR\""""
content = content.replace(old_exit, new_exit)

# 4. Entry logic
old_entry = """                # 2. 신규 진입 검토
                if active_pos is None and time_str < '14:30':
                    past_soxx_60 = soxx_60m[soxx_60m['datetime'] <= curr_dt]
                    soxx_60m_bull = True
                    soxx_60m_bear = True
                    if len(past_soxx_60) >= 20:
                        soxx_c = past_soxx_60['Close'].iloc[-1]
                        soxx_ema20 = past_soxx_60['ema20'].iloc[-1]
                        soxx_60m_bull = (soxx_c >= soxx_ema20 * 0.998)
                        soxx_60m_bear = (soxx_c <= soxx_ema20 * 1.002)"""

new_entry = """                # 2. 신규 진입 검토
                if active_pos is None and time_str < config.PHASE_MAIN_END:
                    past_qqq_60 = qqq_60m[qqq_60m['datetime'] <= curr_dt]
                    qqq_60m_bull = True
                    qqq_60m_bear = True
                    if len(past_qqq_60) >= 20:
                        q_c = past_qqq_60['Close'].iloc[-1]
                        q_ema = past_qqq_60['ema_filter'].iloc[-1]
                        qqq_60m_bull = (q_c >= q_ema * 0.998)
                        qqq_60m_bear = (q_c <= q_ema * 1.002)"""
content = content.replace(old_entry, new_entry)

# 5. Signal trigger
old_sig = """                    if dir_gbdt == "LONG_TQQQ" and conf_gbdt >= 0.60 and soxx_60m_bull and cross_dir != "SHORT_SQQQ":
                        active_pos = {
                            'sym': 'TQQQ',
                            'entry_px': cur_tqqq_close,
                            'entry_dt': curr_dt,
                            'bars': 0
                        }
                    elif dir_gbdt == "SHORT_SQQQ" and conf_gbdt >= 0.60 and soxx_60m_bear and cross_dir != "LONG_TQQQ":"""

new_sig = """                    if dir_gbdt == "LONG_TQQQ" and conf_gbdt >= config.GBDT_CONFIDENCE_THRESHOLD and qqq_60m_bull and cross_dir != "SHORT_SQQQ":
                        active_pos = {
                            'sym': 'TQQQ',
                            'entry_px': cur_tqqq_close,
                            'entry_dt': curr_dt,
                            'bars': 0,
                            'peak_high': cur_tqqq_close
                        }
                    elif dir_gbdt == "SHORT_SQQQ" and conf_gbdt >= config.GBDT_CONFIDENCE_THRESHOLD and qqq_60m_bear and cross_dir != "LONG_TQQQ":"""
content = content.replace(old_sig, new_sig)

# For short sqqq active pos tracking
old_sq = """                            active_pos = {
                                'sym': 'SQQQ',
                                'entry_px': cur_sqqq_close,
                                'entry_dt': curr_dt,
                                'bars': 0
                            }"""
new_sq = """                            active_pos = {
                                'sym': 'SQQQ',
                                'entry_px': cur_sqqq_close,
                                'entry_dt': curr_dt,
                                'bars': 0,
                                'peak_high': cur_sqqq_close
                            }"""
content = content.replace(old_sq, new_sq)

with open('scripts/run_weekly_rolling_wfa.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("PATCHED!")
