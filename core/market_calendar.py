import time
from datetime import datetime, time as datetime_time
from zoneinfo import ZoneInfo
from typing import Dict, Any, Optional
import requests
from utils.logger import system_logger

class USMarketCalendar:
    """
    [ ? (NYSE / NASDAQ) ??? ? ??????
    - ????Daylight Saving Time) ?  (3??? ???~ 11?? ???
    - ???? ?:
      * ????(EDT): ?? 22:30 ~ ? 05:00
      * ????(EST): ?? 23:30 ~ ? 06:00
    - (???? ?????
    - ?  ?? ?? ? ??
    """
    @staticmethod
    def get_market_status(now_dt: Optional[datetime] = None) -> Dict[str, Any]:
        if now_dt is None:
            now_dt = datetime.now()

        ny_tz = ZoneInfo("America/New_York")
        kst_tz = ZoneInfo("Asia/Seoul")
        
        if now_dt is None:
            now_local = datetime.now().astimezone()
            now_ny = now_local.astimezone(ny_tz)
            now_kst = now_local.astimezone(kst_tz)
        elif now_dt.tzinfo is None:
            now_kst = now_dt.replace(tzinfo=kst_tz)
            now_ny = now_kst.astimezone(ny_tz)
        else:
            now_ny = now_dt.astimezone(ny_tz)
            now_kst = now_dt.astimezone(kst_tz)
        weekday_ny = now_ny.weekday()  # 0: Mon, ..., 4: Fri, 5: Sat, 6: Sun
        ny_time = now_ny.time()

        is_weekend = (weekday_ny >= 5)
        
        market_open_time = dtime(9, 30)
        trading_cutoff_time = dtime(15, 30)
        eod_liquidation_time = dtime(15, 50)
        market_close_time = dtime(16, 0)

        is_regular_hours = (not is_weekend) and (market_open_time <= ny_time < market_close_time)
        is_phase2_allowed = False
        is_trading_allowed = (not is_weekend) and (market_open_time <= ny_time < trading_cutoff_time)
        is_entry_allowed = is_trading_allowed
        is_eod_liquidation_window = (not is_weekend) and (eod_liquidation_time <= ny_time < market_close_time)

        if is_regular_hours:
            next_open_kst = None
            time_until_open_str = "   "
        else:
            target_ny_date = now_ny.date()
            if is_weekend:
                days_ahead = (7 - weekday_ny)
                target_ny_date += timedelta(days=days_ahead)
            else:
                if ny_time >= market_close_time:
                    if weekday_ny == 4:
                        target_ny_date += timedelta(days=3)
                    else:
                        target_ny_date += timedelta(days=1)
            
            next_open_ny = datetime.combine(target_ny_date, market_open_time, tzinfo=ny_tz)
            next_open_kst = next_open_ny.astimezone(kst_tz)
            delta = next_open_kst - now_kst
            hours, remainder = divmod(int(delta.total_seconds()), 3600)
            mins, secs = divmod(remainder, 60)
            time_until_open_str = f"{hours}hours {mins}mins"

        if is_weekend:
            day_name_kr = "Weekend"
            session_name = "WEEKEND_CLOSED"
            status_desc = f"? ?   ? ({day_name_kr}?)"
        elif not is_regular_hours:
            if ny_time < market_open_time:
                session_name = "PRE_MARKET_WAITING"
                status_desc = "????? ??? ??"
            else:
                session_name = "AFTER_MARKET_CLOSED"
                status_desc = "? ??? (???"
        else:
            if is_eod_liquidation_window:
                session_name = "EOD_LIQUIDATION"
                status_desc = "???? 10??? 0% ?? ? ??????"
            elif is_trading_allowed:
                session_name = "REGULAR_MARKET_OPEN"
                status_desc = "? ? ???Phase 1 ???  ?(15m Model C)"
            elif is_phase2_allowed:
                session_name = "POWER_HOUR_SNIPER"
                status_desc = "??? ???Phase 2 ? ? ??  ?(5m Sniper)"
            else:
                session_name = "REGULAR_MARKET_NO_ENTRY"
                status_desc = "???? ??(?  ,  ?????"

        is_dst = bool(now_ny.dst())

        return {
            "is_open": is_regular_hours,
            "is_entry_allowed": is_entry_allowed,
            "is_trading_allowed": is_trading_allowed,
            "is_phase2_allowed": is_phase2_allowed,
            "is_weekend": is_weekend,
            "is_eod_liquidation_window": is_eod_liquidation_window,
            "session_name": session_name,
            "status_desc": status_desc,
            "now_kst_str": now_kst.strftime("%Y-%m-%d %H:%M:%S KST"),
            "now_ny_str": now_ny.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "is_dst": is_dst,
            "dst_text": "EDT" if is_dst else "EST",
            "next_open_kst_str": next_open_kst.strftime("%Y-%m-%d (%a) %H:%M KST") if next_open_kst else "Open",
            "time_until_open_str": time_until_open_str
        }

    @staticmethod
    def verify_time_synchronization() -> Dict[str, Any]:
        """
        [?  ??? ??1? ? ???? ????? ??
        1. OS  ? ??? ???(ZoneInfo America/New_York) ??? ?
        2. ????EDT: -13h ?) ?????EST: -14h ?) ? ? ??
        3.  ??4?  ?(09:30~14:30 Phase 1 vs 14:30~15:30 Phase 2 vs 15:30 ??vs 15:50 EOD ?)
        """
        ny_tz = ZoneInfo("America/New_York")
        kst_tz = ZoneInfo("Asia/Seoul")
        now_local = datetime.now().astimezone()
        now_ny = now_local.astimezone(ny_tz)
        now_kst = now_local.astimezone(kst_tz)

        # KST? NYT ???
        delta_hours = round((now_kst.utcoffset().total_seconds() - now_ny.utcoffset().total_seconds()) / 3600.0, 1)
        is_dst = bool(now_ny.dst())
        expected_diff = 13.0 if is_dst else 14.0
        time_sync_ok = (delta_hours == expected_diff)

        #  ??4?  ?( ? ?????? ? ? ????
        test_weekday = now_ny.date() - timedelta(days=now_ny.weekday())
        test_p1 = datetime.combine(test_weekday, dtime(11, 0), tzinfo=ny_tz)
        test_p2 = datetime.combine(test_weekday, dtime(15, 0), tzinfo=ny_tz)
        test_cd = datetime.combine(test_weekday, dtime(15, 40), tzinfo=ny_tz)
        test_eod = datetime.combine(test_weekday, dtime(15, 55), tzinfo=ny_tz)

        s_p1 = USMarketCalendar.get_market_status(test_p1)
        s_p2 = USMarketCalendar.get_market_status(test_p2)
        s_cd = USMarketCalendar.get_market_status(test_cd)
        s_eod = USMarketCalendar.get_market_status(test_eod)

        switching_ok = (
            s_p1["is_entry_allowed"] and not s_p1["is_eod_liquidation_window"] and
            s_eod["is_eod_liquidation_window"] and not s_eod["is_entry_allowed"]
        )

        all_ok = bool(time_sync_ok and switching_ok)

        return {
            "all_ok": all_ok,
            "time_sync_ok": time_sync_ok,
            "switching_ok": switching_ok,
            "delta_hours": delta_hours,
            "expected_diff": expected_diff,
            "is_dst": is_dst,
            "dst_text": "????EDT, 13? ?)" if is_dst else "????EST, 14? ?)",
            "now_kst_str": now_kst.strftime("%Y-%m-%d %H:%M:%S KST"),
            "now_ny_str": now_ny.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "checklist_items": [
            ]
        }


