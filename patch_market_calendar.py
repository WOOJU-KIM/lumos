import re

with open('core/market_calendar.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_block = """        # Early close handling
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
            pass

        is_regular_hours = (not is_weekend) and (market_open_time <= ny_time < market_close_time)"""

new_block = """        # Holiday and Early close handling
        is_holiday = False
        try:
            import pandas_market_calendars as mcal
            nyse = mcal.get_calendar('NYSE')
            today_str = now_ny.strftime('%Y-%m-%d')
            schedule = nyse.schedule(start_date=today_str, end_date=today_str)
            if schedule.empty:
                is_holiday = True
            else:
                actual_close = schedule.iloc[0]['market_close'].astimezone(ny_tz)
                if actual_close.time() != dtime(16, 0):
                    market_close_time = actual_close.time()
                    eod_liquidation_time = (actual_close - timedelta(minutes=5)).time()
                    trading_cutoff_time = (actual_close - timedelta(minutes=30)).time()
        except Exception:
            pass

        is_regular_hours = (not is_weekend) and (not is_holiday) and (market_open_time <= ny_time < market_close_time)"""

content = content.replace(old_block, new_block)

# Also update the flags
old_flags = """        is_phase2_allowed = False
        is_trading_allowed = (not is_weekend) and (market_open_time <= ny_time < trading_cutoff_time)
        is_entry_allowed = is_trading_allowed
        is_eod_liquidation_window = (not is_weekend) and (eod_liquidation_time <= ny_time < market_close_time)"""

new_flags = """        is_phase2_allowed = False
        is_trading_allowed = (not is_weekend) and (not is_holiday) and (market_open_time <= ny_time < trading_cutoff_time)
        is_entry_allowed = is_trading_allowed
        is_eod_liquidation_window = (not is_weekend) and (not is_holiday) and (eod_liquidation_time <= ny_time < market_close_time)"""

content = content.replace(old_flags, new_flags)

# Also update the description for holiday
old_desc = """        if is_weekend:
            day_name_kr = "Weekend"
            session_name = "WEEKEND_CLOSED"
            status_desc = f"주말 ?장 ({day_name_kr})"
        elif not is_regular_hours:"""

new_desc = """        if is_weekend:
            day_name_kr = "Weekend"
            session_name = "WEEKEND_CLOSED"
            status_desc = "주말 휴장 (Weekend)"
        elif is_holiday:
            session_name = "HOLIDAY_CLOSED"
            status_desc = "미국 증시 휴장일 (Holiday)"
        elif not is_regular_hours:"""

content = content.replace(old_desc, new_desc)

with open('core/market_calendar.py', 'w', encoding='utf-8') as f:
    f.write(content)
