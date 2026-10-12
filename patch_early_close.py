import re
with open('core/market_calendar.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_times = """        market_open_time = _parse_time(config.PHASE_MAIN_START)
        trading_cutoff_time = _parse_time(config.PHASE_MAIN_END)
        eod_liquidation_time = _parse_time(config.PHASE_EOD_CLEAR)
        market_close_time = _parse_time(config.MARKET_CLOSE_TIME)"""

new_times = """        market_open_time = _parse_time(config.PHASE_MAIN_START)
        trading_cutoff_time = _parse_time(config.PHASE_MAIN_END)
        eod_liquidation_time = _parse_time(config.PHASE_EOD_CLEAR)
        market_close_time = _parse_time(config.MARKET_CLOSE_TIME)

        # Early close handling
        try:
            import pandas_market_calendars as mcal
            nyse = mcal.get_calendar('NYSE')
            today_str = now_ny.strftime('%Y-%m-%d')
            schedule = nyse.schedule(start_date=today_str, end_date=today_str)
            if not schedule.empty:
                actual_close = schedule.iloc[0]['market_close'].astimezone(ny_tz)
                if actual_close.time() != dtime(16, 0):
                    market_close_time = actual_close.time()
                    eod_liquidation_time = (actual_close - timedelta(minutes=5)).time()
                    trading_cutoff_time = (actual_close - timedelta(minutes=30)).time()
        except Exception:
            pass"""

content = content.replace(old_times, new_times)

with open('core/market_calendar.py', 'w', encoding='utf-8') as f:
    f.write(content)
