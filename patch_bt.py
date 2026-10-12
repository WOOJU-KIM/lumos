import pathlib

txt = pathlib.Path('scripts/temp_run_5m_precision_backtest.py').read_text(encoding='utf-8')

replacements = {
    'soxx_60m = lake.load_candles("SOXX", "60m")': 'qqq_60m = lake.load_candles("QQQ", "60m")',
    "soxx_60m['ema20'] = soxx_60m['Close'].ewm(span=20, adjust=False).mean()": "qqq_60m['ema_filter'] = qqq_60m['Close'].ewm(span=config.QQQ_EMA_PERIOD, adjust=False).mean()",
    "past_soxx_60 = soxx_60m[soxx_60m['datetime'] <= curr_dt]": "past_qqq_60 = qqq_60m[qqq_60m['datetime'] <= curr_dt]",
    "if len(past_soxx_60) >= 20:": "if len(past_qqq_60) >= config.QQQ_EMA_PERIOD:",
    "soxx_c = past_soxx_60['Close'].iloc[-1]": "qqq_c = past_qqq_60['Close'].iloc[-1]",
    "soxx_ema20 = past_soxx_60['ema20'].iloc[-1]": "qqq_ema = past_qqq_60['ema_filter'].iloc[-1]",
    "soxx_60m_bull = (soxx_c >= soxx_ema20 * 0.998)": "qqq_60m_bull = (qqq_c >= qqq_ema * 0.998)",
    "soxx_60m_bear = (soxx_c <= soxx_ema20 * 1.002)": "qqq_60m_bear = (qqq_c <= qqq_ema * 1.002)",
    "pass_60m_bull = soxx_60m_bull if config.USE_60M_TREND_FILTER else True": "pass_60m_bull = qqq_60m_bull if config.USE_60M_TREND_FILTER else True",
    "pass_60m_bear = soxx_60m_bear if config.USE_60M_TREND_FILTER else True": "pass_60m_bear = qqq_60m_bear if config.USE_60M_TREND_FILTER else True",
    'dir_gbdt == "LONG_TQQQ" and conf_gbdt >= 0.60': 'dir_gbdt == "LONG_TQQQ" and conf_gbdt >= config.GBDT_CONFIDENCE_THRESHOLD',
    'dir_gbdt == "SHORT_SQQQ" and conf_gbdt >= 0.60': 'dir_gbdt == "SHORT_SQQQ" and conf_gbdt >= config.GBDT_CONFIDENCE_THRESHOLD',
    "slippage_rate = 0.0020": "slippage_rate = config.BACKTEST_FEE_SLIPPAGE"
}

for k, v in replacements.items():
    txt = txt.replace(k, v)

pathlib.Path('scripts/temp_run_all_5m_strict.py').write_text(txt, encoding='utf-8')
print("Patched.")
