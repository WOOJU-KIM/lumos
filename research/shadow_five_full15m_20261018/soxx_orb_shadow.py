"""SOXX 5-minute ORB + Intraday VWAP + RVOL shadow strategy with 49%+49% Opposite Relay.

Isolated shadow execution module:
- Never imports live runner, broker, or order APIs.
- All signal indicators (ORB 09:30~10:00, daily 09:30 VWAP, 20-bar RVOL) are
  computed strictly on SOXX 5-minute bars in America/New_York time.
- Initial entry allocates 49% of cycle cash to SOXL (Long) or SOXS (Short).
- If the initial leg reaches -1.0%, a 49% opposite-ETF chain buy is triggered.
- Each held leg is monitored independently for 3 exit conditions only:
  Stop (-2.0% ETF), Trailing (+1.5% / -0.3%), and EOD (15:55 liquidate all).
- Once all legs are closed (all positions empty), the cycle ends and the strategy
  immediately resumes scanning for the next initial entry signal.
"""
import csv
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

from common import HOME, CONFIG, load_execution_config, sha
from runtime import BUY_WAIT, stamp

NY = ZoneInfo("America/New_York")
CSV_HEADER = [
    "날짜",
    "진입시간",
    "청산시간",
    "종목",
    "진입가",
    "청산가",
    "수량",
    "순손익($)",
    "순수익률(%)",
    "청산사유",
]
ORB_BAR_HMS = ("09:30", "09:35", "09:40", "09:45", "09:50", "09:55")
EPS = 1e-12


def load_soxx_orb_settings():
    path = HOME / "soxx_orb_shadow.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if (
        cfg.get("schema_version") != 1
        or cfg.get("signal_ticker") != "SOXX"
        or cfg.get("long_exec_ticker") != "SOXL"
        or cfg.get("short_exec_ticker") != "SOXS"
    ):
        raise ValueError("Invalid SOXX ORB shadow configuration")
    return cfg


def _parse_ny_time(value):
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=NY)
        return value.astimezone(NY)
    text = str(value).strip()
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=NY)
    return dt.astimezone(NY)


class SoxxOrbCandidate:
    """Adapter for Engine registration so live_feed health/audit tracks this shadow."""

    cohorts = ("primary",)

    def __init__(self, settings=None, ledger_kwargs=None):
        self.settings = dict(settings or load_soxx_orb_settings())
        self.name = self.settings["strategy_id"]
        self.version = "SOXX_5M_ORB_VWAP_RVOL_RELAY_V1:" + sha(HOME / "soxx_orb_shadow.json")
        self._ledger_kwargs = dict(ledger_kwargs or {})

    def ledger_factory(self, model_id, cohort="primary", evaluation_window=None):
        return SoxxOrbLedger(
            model_id=model_id,
            cohort=cohort,
            evaluation_window=evaluation_window,
            settings=self.settings,
            **self._ledger_kwargs,
        )

    def predict(self, packet):
        return dict(
            model_id=self.name,
            model_version=self.version,
            side=-1,
            forced_side=-1,
            entry_reason="SOXX_5M_SIGNAL_DRIVEN",
            execution_strategy="SOXX_ORB_VWAP_RVOL_RELAY_V1",
            threshold_applied=False,
        )


class SoxxOrbLedger:
    """SOXX 5m signal + SOXL/SOXS 49%/49% opposite-relay virtual execution ledger."""

    def __init__(
        self,
        model_id=None,
        cohort="primary",
        evaluation_window=None,
        settings=None,
        db=None,
        csv_path=None,
        log_path=None,
        immediate_fill=False,
    ):
        self.settings = dict(settings or load_soxx_orb_settings())
        self.model_id = model_id or str(self.settings.get("strategy_id", "SOXX_ORB_VWAP_RVOL_M15"))
        self.cohort = cohort
        self.execution_config = load_execution_config()
        self._evaluation_window = evaluation_window or (lambda day: True)
        self.db = db
        self.immediate_fill = bool(immediate_fill)

        default_dir = HOME / "runs"
        csv_filename = str(self.settings.get("csv_filename", "soxx_orb_m15_shadow_trades.csv"))
        log_filename = str(self.settings.get("log_filename", "soxx_orb_m15_shadow_trades.log"))
        jsonl_filename = str(self.settings.get("jsonl_filename", "soxx_orb_m15_shadow_trades.jsonl"))
        self.csv_path = Path(csv_path) if csv_path is not None else (default_dir / csv_filename)
        self.log_path = Path(log_path) if log_path is not None else (default_dir / log_filename)
        self.jsonl_path = self.log_path.with_suffix(".jsonl") if log_path is not None else (default_dir / jsonl_filename)

        self.cash = float(self.settings["initial_cash"])
        self.allocation_ratio = float(self.settings.get("initial_allocation", self.settings["allocation_ratio"]))
        self.initial_allocation = float(self.settings.get("initial_allocation", self.allocation_ratio))
        self.chain_allocation = float(self.settings.get("chain_allocation", 0.49))
        self.chain_trigger_loss_pct = float(self.settings.get("chain_trigger_loss_pct", -0.015))
        self.fee_rate = float(self.settings["fee_rate"])
        self.orb_min_range_ratio = float(self.settings["orb_min_range_ratio"])
        self.rvol_lookback = int(self.settings["rvol_lookback_bars"])
        self.rvol_min = float(self.settings["rvol_min_threshold"])
        self.entry_start_hm = str(self.settings["entry_start_hm"])
        self.entry_end_hm = str(self.settings["entry_end_hm"])
        self.max_daily_trades = int(self.settings["max_daily_trades"])
        self.hard_stop_etf_pct = float(self.settings["hard_stop_etf_pct"])
        self.trailing_trigger_pct = float(self.settings["trailing_trigger_pct"])
        self.trailing_drop_pct = float(self.settings["trailing_drop_pct"])
        self.eod_exit_hm = str(self.settings["eod_exit_hm"])

        self.positions = []
        self.pending = []
        self.closed = []
        self.cycles = []
        self.quotes = {}
        self.trade_logs = []

        # Relay cycle state
        self.cycle_budget = None
        self.cycle_id = None
        self.cycle_number = 0
        self.cycle_start_time = None
        self.cycle_closed_start = 0
        self.chain_triggered = False

        # Intraday SOXX 5m indicator state
        self.current_date = None
        self.daily_trade_count = 0
        self.orb_bars = {}
        self.soxx_open_0930 = None
        self.orb_high = None
        self.orb_low = None
        self.orb_range = None
        self.orb_ready = False
        self.cum_tp_vol = 0.0
        self.cum_vol = 0.0
        self.vwap = None
        self.rvol = 0.0
        self.sma_20_vol = None
        self.volume_history = []
        self.seen_bar_times = set()
        self.last_soxx_price = None
        self.last_soxx_bar = None
        self._last_db_query_ts = None
        self._db_bootstrapped = False

    @property
    def position(self):
        return self.positions[0] if self.positions else None

    def _reset_day(self, day):
        self.current_date = day
        self.daily_trade_count = 0
        self.orb_bars = {}
        self.soxx_open_0930 = None
        self.orb_high = None
        self.orb_low = None
        self.orb_range = None
        self.orb_ready = False
        self.cum_tp_vol = 0.0
        self.cum_vol = 0.0
        self.vwap = None

    def _row(self, event, now, **values):
        record = dict(
            event=event,
            time=str(now),
            model_id=self.model_id,
            cohort=self.cohort,
            cycle_id=self.cycle_id,
            **values,
        )
        self._append_jsonl(record)
        return record

    def _write_log(self, line):
        self.trade_logs.append(line)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")

    def _append_jsonl(self, record):
        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self.jsonl_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _finish_cycle(self, now):
        """When all positions and pending orders are empty, close the cycle and allow next initial entry."""
        if self.cycle_budget is None or self.positions or self.pending:
            return []
        legs = len(self.closed) - self.cycle_closed_start
        rows = []
        if legs > 0:
            net_pnl = self.cash - self.cycle_budget
            net_ret_pct = ((self.cash / self.cycle_budget) - 1.0) * 100.0 if self.cycle_budget > 0 else 0.0
            cycle_row = self._row(
                "VIRTUAL_CYCLE_CLOSED",
                now,
                start_time=str(self.cycle_start_time),
                starting_equity=self.cycle_budget,
                ending_equity=self.cash,
                net_pnl=net_pnl,
                net_return=(self.cash / self.cycle_budget) - 1.0 if self.cycle_budget > 0 else 0.0,
                net_return_pct=net_ret_pct,
                legs=legs,
            )
            self.cycles.append(cycle_row)
            rows.append(cycle_row)
            self._write_log(
                f"[{_parse_ny_time(now).strftime('%Y-%m-%d %H:%M:%S')}][CYCLE_CLOSED] "
                f"cycle_id={self.cycle_id} legs={legs} start_cash=${self.cycle_budget:.2f} "
                f"end_cash=${self.cash:.2f} cycle_net_pnl=${net_pnl:.2f} ({net_ret_pct:+.2f}%)"
            )
        self.cycle_budget = None
        self.cycle_id = None
        self.cycle_start_time = None
        self.chain_triggered = False
        return rows

    def intend(self, time, side, atr_pct, decision_id, model_version, prediction):
        """15m GBDT decision hook is ignored; this strategy trades exclusively on SOXX 5m signals."""
        return "SOXX_5M_SIGNAL_DRIVEN"

    def _sync_db_bars(self, now):
        """Synchronize completed SOXX 5m bars from read-only SQLite DB up to `now`."""
        if self.db is None or not hasattr(self.db, "execute"):
            return []
        now_ny = _parse_ny_time(now)
        expected_open = now_ny.replace(minute=(now_ny.minute // 5) * 5, second=0, microsecond=0) - timedelta(minutes=5)
        if self._db_bootstrapped and expected_open.isoformat() in self.seen_bar_times:
            return []
        if self._last_db_query_ts is not None and 0 <= (now_ny - self._last_db_query_ts).total_seconds() < 1.0:
            return []
        self._last_db_query_ts = now_ny
        cutoff_open = (now_ny.replace(tzinfo=None) - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
        limit = 80 if not self._db_bootstrapped else 12
        rows = self.db.execute(
            "SELECT datetime, open, high, low, close, volume "
            "FROM market_candles WHERE symbol=? AND timeframe=? AND datetime<=? "
            "ORDER BY datetime DESC LIMIT ?",
            ("SOXX", "5m", cutoff_open, limit),
        ).fetchall()[::-1]
        if not rows:
            return []
        was_bootstrapped = self._db_bootstrapped
        self._db_bootstrapped = True
        latest_bar_date = _parse_ny_time(rows[-1][0]).date()
        events = []
        for idx, (dt_str, o, h, l, c, v) in enumerate(rows):
            bar_dt = _parse_ny_time(dt_str)
            if bar_dt.isoformat() in self.seen_bar_times:
                continue
            is_last_live_bar = idx == len(rows) - 1 and bar_dt.date() == now_ny.date() == latest_bar_date
            bar_close_dt = bar_dt + timedelta(minutes=5)
            if was_bootstrapped and is_last_live_bar and 0 <= (now_ny - bar_close_dt).total_seconds() <= 300:
                eval_dt = now_ny
            else:
                eval_dt = bar_close_dt if bar_close_dt <= now_ny else now_ny
            ev = self.on_soxx_5m_bar(
                bar_time=bar_dt,
                open_px=o,
                high_px=h,
                low_px=l,
                close_px=c,
                volume=v,
                eval_time=eval_dt,
                allow_trade=is_last_live_bar,
            )
            events.extend(ev)
        return events

    def on_soxx_5m_bar(
        self,
        bar_time,
        open_px,
        high_px,
        low_px,
        close_px,
        volume,
        eval_time=None,
        allow_trade=True,
        exec_ask=None,
        exec_bid=None,
    ):
        """Process a completed SOXX 5-minute candle and evaluate exit/entry rules."""
        bar_dt = _parse_ny_time(bar_time)
        now = _parse_ny_time(eval_time) if eval_time is not None else bar_dt
        open_px = float(open_px)
        high_px = float(high_px)
        low_px = float(low_px)
        close_px = float(close_px)
        volume = float(volume)
        if not all(math.isfinite(x) for x in (open_px, high_px, low_px, close_px, volume)):
            raise ValueError("Non-finite SOXX 5m candle values")
        if low_px <= 0 or high_px < low_px or volume < 0:
            raise ValueError("Invalid SOXX 5m candle price/volume")

        bar_key = bar_dt.isoformat()
        if bar_key in self.seen_bar_times:
            return []
        self.seen_bar_times.add(bar_key)

        hm = bar_dt.strftime("%H:%M")
        if not ("09:30" <= hm < "16:00") or bar_dt.weekday() > 4:
            return []

        if self.current_date != bar_dt.date() or hm == "09:30":
            if self.current_date != bar_dt.date():
                self._reset_day(bar_dt.date())
            elif hm == "09:30" and not self.orb_bars:
                self._reset_day(bar_dt.date())

        # 1. RVOL: current 5m SOXX_Volume / preceding 20 5m SOXX_Volume SMA_20
        if len(self.volume_history) >= self.rvol_lookback:
            window = self.volume_history[-self.rvol_lookback :]
            self.sma_20_vol = sum(window) / float(self.rvol_lookback)
            self.rvol = (volume / self.sma_20_vol) if self.sma_20_vol > 0 else 0.0
        else:
            self.sma_20_vol = None
            self.rvol = 0.0
        self.volume_history.append(volume)
        if len(self.volume_history) > max(100, self.rvol_lookback * 3):
            self.volume_history = self.volume_history[-max(100, self.rvol_lookback * 3) :]

        # 2. Intraday VWAP: cumulative from 09:30, reset daily at 09:30
        typical_price = (high_px + low_px + close_px) / 3.0
        self.cum_tp_vol += typical_price * volume
        self.cum_vol += volume
        self.vwap = (self.cum_tp_vol / self.cum_vol) if self.cum_vol > 0 else typical_price

        # 3. Opening Range (09:30 ~ 10:00, first 6 five-minute candles: 09:30..09:55)
        was_orb_ready = self.orb_ready
        if hm in ORB_BAR_HMS:
            if hm == "09:30" or self.soxx_open_0930 is None:
                if hm == "09:30":
                    self.soxx_open_0930 = open_px
            self.orb_bars[hm] = dict(open=open_px, high=high_px, low=low_px, close=close_px, volume=volume)
            self.orb_high = max(b["high"] for b in self.orb_bars.values())
            self.orb_low = min(b["low"] for b in self.orb_bars.values())
            self.orb_range = self.orb_high - self.orb_low
            self.orb_ready = (
                len(self.orb_bars) == int(self.settings["orb_bar_count"])
                and self.soxx_open_0930 is not None
                and self.soxx_open_0930 > 0
            )
            if self.orb_ready and not was_orb_ready and allow_trade:
                orb_pct = (self.orb_range / self.soxx_open_0930) * 100.0
                active_flag = (self.orb_range / self.soxx_open_0930) >= (self.orb_min_range_ratio - EPS)
                self._row(
                    "SOXX_ORB_READY",
                    now,
                    date=str(bar_dt.date()),
                    soxx_open_0930=self.soxx_open_0930,
                    orb_high=self.orb_high,
                    orb_low=self.orb_low,
                    orb_range=self.orb_range,
                    orb_range_pct=orb_pct,
                    day_trade_enabled=active_flag,
                )
                self._write_log(
                    f"[{bar_dt.strftime('%Y-%m-%d %H:%M:%S')}][ORB_READY] "
                    f"open_0930={self.soxx_open_0930:.4f} orb_high={self.orb_high:.4f} "
                    f"orb_low={self.orb_low:.4f} orb_range={self.orb_range:.4f} ({orb_pct:.3f}%) "
                    f"enabled={active_flag}"
                )

        self.last_soxx_price = close_px
        self.last_soxx_bar = dict(
            bar_time=str(bar_dt),
            open=open_px,
            high=high_px,
            low=low_px,
            close=close_px,
            volume=volume,
            vwap=self.vwap,
            rvol=self.rvol,
            orb_high=self.orb_high,
            orb_low=self.orb_low,
            orb_range=self.orb_range,
            orb_ready=self.orb_ready,
        )

        if not allow_trade or not self._evaluation_window(bar_dt.date()):
            return []

        rows = self._cancel_stale_buys(now)

        # If holding any positions (INITIAL and/or CHAIN), only EOD (15:55) is checked on 5m bar timestamps;
        # VWAP exit is disabled so exits occur strictly on Stop (-2.0%), Trailing (+1.5%/-0.3%), or EOD (15:55).
        if self.positions:
            for p in list(self.positions):
                if p["exit_pending"]:
                    continue
                if hm >= self.eod_exit_hm or now.strftime("%H:%M") >= self.eod_exit_hm:
                    rows.extend(
                        self._trigger_exit(
                            p=p,
                            now=now,
                            reason="EOD",
                            trigger_price=close_px,
                            price_basis="SOXX_5M_CLOSE",
                            immediate_bid=exec_bid,
                        )
                    )
            rows.extend(self._finish_cycle(now))
            return rows

        # 4. Initial Entry Logic (10:00 ~ 14:30 NYT, only when all positions and pending orders are empty)
        if self.positions or self.pending:
            return rows
        if self.daily_trade_count >= self.max_daily_trades:
            return rows
        if not self.orb_ready or self.soxx_open_0930 is None or self.soxx_open_0930 <= 0:
            return rows
        if (self.orb_range / self.soxx_open_0930) < (self.orb_min_range_ratio - EPS):
            return rows
        if not (self.entry_start_hm <= hm <= self.entry_end_hm):
            return rows
        if now.strftime("%H:%M") >= self.eod_exit_hm:
            return rows

        long_signal = (
            (close_px > self.orb_high)
            and (close_px > self.vwap)
            and (self.rvol >= self.rvol_min - EPS)
        )
        short_signal = (
            (close_px < self.orb_low)
            and (close_px < self.vwap)
            and (self.rvol >= self.rvol_min - EPS)
        )

        if not long_signal and not short_signal:
            self._append_jsonl(
                dict(
                    event="SOXX_5M_NO_SIGNAL",
                    time=str(now),
                    bar_time=str(bar_dt),
                    close=close_px,
                    vwap=self.vwap,
                    rvol=self.rvol,
                    orb_high=self.orb_high,
                    orb_low=self.orb_low,
                )
            )
            return rows

        side = 1 if long_signal else 0
        target_symbol = self.settings["long_exec_ticker"] if side == 1 else self.settings["short_exec_ticker"]
        decision_id = f"soxx_5m_{bar_dt.strftime('%Y%m%d_%H%M')}"
        self._write_log(
            f"[{_parse_ny_time(now).strftime('%Y-%m-%d %H:%M:%S')}][ENTRY_SIGNAL] "
            f"bar={bar_dt.strftime('%H:%M')} dir={'LONG' if side == 1 else 'SHORT'} "
            f"exec_ticker={target_symbol} soxx_close={close_px:.4f} orb_high={self.orb_high:.4f} "
            f"orb_low={self.orb_low:.4f} vwap={self.vwap:.4f} rvol={self.rvol:.2f}"
        )

        # Start a new cycle if not already active
        if self.cycle_budget is None:
            self.cycle_budget = self.cash
            self.cycle_number += 1
            self.cycle_id = f"{self.model_id}:{decision_id}:{self.cycle_number}"
            self.cycle_start_time = now
            self.cycle_closed_start = len(self.closed)
            self.chain_triggered = False

        intent = dict(
            action="BUY",
            role="INITIAL",
            time=now,
            bar_time=bar_dt,
            side=side,
            symbol=target_symbol,
            entry_soxx=close_px,
            orb_high=self.orb_high,
            orb_low=self.orb_low,
            orb_range=self.orb_range,
            vwap=self.vwap,
            rvol=self.rvol,
            decision_id=decision_id,
            model_version="SOXX_5M_ORB_VWAP_RVOL_RELAY_V1",
        )
        rows.append(
            self._row(
                "VIRTUAL_BUY_INTENT",
                now,
                role="INITIAL",
                symbol=target_symbol,
                side=side,
                decision_id=decision_id,
                soxx_close=close_px,
                orb_high=self.orb_high,
                orb_low=self.orb_low,
                orb_range=self.orb_range,
                vwap=self.vwap,
                rvol=self.rvol,
            )
        )
        if exec_ask is not None or self.immediate_fill:
            fill_px = float(exec_ask) if exec_ask is not None else float(self.quotes.get(target_symbol, {}).get("ask", 0.0))
            if fill_px > 0:
                rows.extend(self._execute_buy(intent, now, fill_px, price_basis="IMMEDIATE_ASK"))
            else:
                self.pending.append(intent)
        else:
            self.pending.append(intent)
        return rows

    def _cancel_stale_buys(self, now):
        rows = []
        for pending in list(self.pending):
            if pending["action"] != "BUY":
                continue
            reason = None
            if (now - pending["time"]).total_seconds() > BUY_WAIT:
                reason = "STALE_INTENT"
            elif now.strftime("%H:%M") >= self.eod_exit_hm:
                reason = "EOD_CUTOFF"
            if reason is not None:
                self.pending.remove(pending)
                rows.append(
                    self._row(
                        "BUY_CANCELLED_" + reason,
                        now,
                        role=pending.get("role", "INITIAL"),
                        symbol=pending["symbol"],
                        decision_id=pending["decision_id"],
                    )
                )
        if rows:
            rows.extend(self._finish_cycle(now))
        return rows

    def _execute_buy(self, pending, now, ask_price, price_basis="WS_ASK"):
        ask_price = float(ask_price)
        if not math.isfinite(ask_price) or ask_price <= 0:
            raise ValueError("Invalid virtual buy price")
        role = pending.get("role", "INITIAL")
        if any(p["symbol"] == pending["symbol"] for p in self.positions):
            return []
        if self.daily_trade_count >= self.max_daily_trades:
            return [self._row("BUY_CANCELLED_DAILY_LIMIT", now, role=role, symbol=pending["symbol"])]

        if self.cycle_budget is None:
            self.cycle_budget = self.cash
            self.cycle_number += 1
            self.cycle_id = f"{self.model_id}:{pending['decision_id']}:{self.cycle_number}"
            self.cycle_start_time = now
            self.cycle_closed_start = len(self.closed)
            self.chain_triggered = False

        ratio = self.initial_allocation if role == "INITIAL" else self.chain_allocation
        alloc_cash = min(self.cycle_budget * ratio, self.cash)
        qty = int(alloc_cash / ask_price)
        while qty > 0 and (ask_price * qty * (1.0 + self.fee_rate)) > self.cash:
            qty -= 1
        if qty <= 0:
            return [self._row("BUY_CANCELLED_CASH", now, role=role, symbol=pending["symbol"])]

        buy_notional = ask_price * qty
        buy_fee = buy_notional * self.fee_rate
        self.cash -= (buy_notional + buy_fee)
        self.daily_trade_count += 1

        leg_index = (len(self.closed) - self.cycle_closed_start) + len(self.positions) + 1
        leg_id = f"{self.cycle_id}:{leg_index}"
        pos = dict(
            symbol=pending["symbol"],
            side=pending["side"],
            role=role,
            leg_id=leg_id,
            qty=qty,
            entry=ask_price,
            entry_etf=ask_price,
            highest_etf=ask_price,
            peak=ask_price,
            entry_soxx=float(pending["entry_soxx"]),
            orb_high=float(pending["orb_high"]),
            orb_low=float(pending["orb_low"]),
            orb_range=float(pending["orb_range"]),
            entry_vwap=float(pending["vwap"]),
            entry_rvol=float(pending["rvol"]),
            buy_fee=buy_fee,
            entry_time=now,
            decision_id=pending["decision_id"],
            model_version=pending["model_version"],
            trailing_armed=False,
            exit_pending=False,
        )
        self.positions.append(pos)
        if role == "CHAIN":
            self.chain_triggered = True
        self._write_log(
            f"[{_parse_ny_time(now).strftime('%Y-%m-%d %H:%M:%S')}][VIRTUAL_BUY] "
            f"role={role} leg_id={leg_id} symbol={pos['symbol']} price={ask_price:.4f} "
            f"qty={qty} notional=${buy_notional:.2f} fee=${buy_fee:.2f} "
            f"soxx_ref={pos['entry_soxx']:.4f} vwap={pos['entry_vwap']:.4f} rvol={pos['entry_rvol']:.2f} "
            f"remaining_cash=${self.cash:.2f} daily_trades={self.daily_trade_count}"
        )
        return [
            self._row(
                "VIRTUAL_BUY",
                now,
                role=role,
                leg_id=leg_id,
                symbol=pos["symbol"],
                side=pos["side"],
                price=ask_price,
                quantity=qty,
                allocation=ratio,
                cycle_budget=self.cycle_budget,
                buy_fee=buy_fee,
                entry_soxx=pos["entry_soxx"],
                orb_range=pos["orb_range"],
                daily_trade_count=self.daily_trade_count,
                remaining_cash=self.cash,
                decision_id=pos["decision_id"],
                model_version=pos["model_version"],
                price_basis=price_basis,
                fill_assumption="NEXT_OBSERVED_WS_ASK_NO_QUEUE_OR_MARKET_IMPACT_MODEL",
            )
        ]

    def _check_chain_trigger(self, initial_pos, now, current_etf_price):
        """If INITIAL leg hits -1.0% (chain_trigger_loss_pct), queue 49% opposite ETF buy."""
        if (
            self.chain_triggered
            or initial_pos.get("role") != "INITIAL"
            or initial_pos.get("exit_pending")
            or current_etf_price is None
            or current_etf_price <= 0
        ):
            return []
        if now.strftime("%H:%M") >= self.eod_exit_hm:
            return []
        if self.daily_trade_count >= self.max_daily_trades:
            return []
        if any(p.get("role") == "CHAIN" for p in self.positions) or any(
            p.get("action") == "BUY" and p.get("role") == "CHAIN" for p in self.pending
        ):
            return []

        etf_ret = (current_etf_price - initial_pos["entry_etf"]) / initial_pos["entry_etf"]
        if etf_ret > self.chain_trigger_loss_pct + EPS:
            return []

        self.chain_triggered = True
        opposite_side = 0 if initial_pos["side"] == 1 else 1
        opposite_symbol = (
            self.settings["long_exec_ticker"] if opposite_side == 1 else self.settings["short_exec_ticker"]
        )
        soxx_px = float(self.last_soxx_price) if self.last_soxx_price is not None else float(initial_pos["entry_soxx"])
        chain_intent = dict(
            action="BUY",
            role="CHAIN",
            time=now,
            side=opposite_side,
            symbol=opposite_symbol,
            source_symbol=initial_pos["symbol"],
            source_entry_etf=initial_pos["entry_etf"],
            source_trigger_price=current_etf_price,
            entry_soxx=soxx_px,
            orb_high=initial_pos["orb_high"],
            orb_low=initial_pos["orb_low"],
            orb_range=initial_pos["orb_range"],
            vwap=self.vwap if self.vwap is not None else initial_pos["entry_vwap"],
            rvol=self.rvol if self.rvol is not None else initial_pos["entry_rvol"],
            decision_id=initial_pos["decision_id"],
            model_version=initial_pos["model_version"],
        )
        rows = [
            self._row(
                "VIRTUAL_CHAIN_INTENT",
                now,
                role="CHAIN",
                source_symbol=initial_pos["symbol"],
                symbol=opposite_symbol,
                side=opposite_side,
                source_entry=initial_pos["entry_etf"],
                trigger_price=current_etf_price,
                source_return_pct=etf_ret * 100.0,
                decision_id=initial_pos["decision_id"],
            )
        ]
        self._write_log(
            f"[{_parse_ny_time(now).strftime('%Y-%m-%d %H:%M:%S')}][CHAIN_TRIGGER] "
            f"source={initial_pos['symbol']} entry={initial_pos['entry_etf']:.4f} "
            f"cur={current_etf_price:.4f} ({etf_ret * 100.0:.2f}%) -> "
            f"chain_buy={opposite_symbol} (49% allocation)"
        )
        if self.immediate_fill:
            opp_ask = float(self.quotes.get(opposite_symbol, {}).get("ask", 0.0))
            if opp_ask > 0:
                rows.extend(self._execute_buy(chain_intent, now, opp_ask, price_basis="IMMEDIATE_ASK"))
            else:
                self.pending.append(chain_intent)
        else:
            self.pending.append(chain_intent)
        return rows

    def _trigger_exit(self, p, now, reason, trigger_price, price_basis, immediate_bid=None):
        if p["exit_pending"]:
            return []
        p["exit_pending"] = True
        sell_intent = dict(
            action="SELL",
            role=p.get("role", "INITIAL"),
            time=now,
            side=p["side"],
            symbol=p["symbol"],
            position=p,
            reason=reason,
            trigger_price=trigger_price,
            price_basis=price_basis,
        )
        rows = [
            self._row(
                "VIRTUAL_EXIT_INTENT",
                now,
                role=p.get("role", "INITIAL"),
                leg_id=p.get("leg_id"),
                symbol=p["symbol"],
                reason=reason,
                trigger_price=trigger_price,
                price_basis=price_basis,
                decision_id=p["decision_id"],
                model_version=p["model_version"],
            )
        ]
        self._write_log(
            f"[{_parse_ny_time(now).strftime('%Y-%m-%d %H:%M:%S')}][EXIT_INTENT] "
            f"role={p.get('role', 'INITIAL')} leg_id={p.get('leg_id')} symbol={p['symbol']} "
            f"reason={reason} trigger_price={trigger_price} basis={price_basis}"
        )
        if immediate_bid is not None or self.immediate_fill:
            bid_px = float(immediate_bid) if immediate_bid is not None else float(self.quotes.get(p["symbol"], {}).get("bid", 0.0))
            if bid_px <= 0 and trigger_price is not None and p["symbol"] in ("SOXL", "SOXS") and price_basis != "SOXX_5M_CLOSE":
                bid_px = float(trigger_price)
            if bid_px > 0:
                rows.extend(self._execute_sell(sell_intent, now, bid_px))
            else:
                self.pending.append(sell_intent)
        else:
            self.pending.append(sell_intent)
        return rows

    def _execute_sell(self, pending, now, bid_price):
        p = pending["position"]
        if p not in self.positions:
            return []
        bid_price = float(bid_price)
        if not math.isfinite(bid_price) or bid_price <= 0:
            raise ValueError("Invalid virtual sell price")

        virtual_buy_price = float(p["entry_etf"])
        virtual_sell_price = bid_price
        qty = int(p["qty"])
        reason = str(pending["reason"])

        gross_pnl = (virtual_sell_price - virtual_buy_price) * qty
        buy_fee = virtual_buy_price * qty * self.fee_rate
        sell_fee = virtual_sell_price * qty * self.fee_rate
        total_fee = buy_fee + sell_fee
        net_pnl = gross_pnl - total_fee
        buy_notional = virtual_buy_price * qty
        net_return = (net_pnl / buy_notional) if buy_notional > 0 else 0.0
        net_return_pct = net_return * 100.0

        self.cash += (virtual_sell_price * qty - sell_fee)
        self.positions.remove(p)

        entry_ny = _parse_ny_time(p["entry_time"])
        exit_ny = _parse_ny_time(now)
        trade_date = exit_ny.strftime("%Y-%m-%d")
        entry_time_str = entry_ny.strftime("%H:%M:%S")
        exit_time_str = exit_ny.strftime("%H:%M:%S")

        csv_row = [
            trade_date,
            entry_time_str,
            exit_time_str,
            p["symbol"],
            f"{virtual_buy_price:.4f}",
            f"{virtual_sell_price:.4f}",
            str(qty),
            f"{net_pnl:.2f}",
            f"{net_return_pct:.2f}",
            reason,
        ]
        log_line = (
            f"[{trade_date} {exit_time_str}][VIRTUAL_SELL] "
            f"entry={entry_time_str} exit={exit_time_str} "
            f"role={p.get('role', 'INITIAL')} leg_id={p.get('leg_id')} symbol={p['symbol']} "
            f"buy={virtual_buy_price:.4f} sell={virtual_sell_price:.4f} peak={p['highest_etf']:.4f} "
            f"qty={qty} gross_pnl=${gross_pnl:.2f} fee=${total_fee:.2f} "
            f"net_pnl=${net_pnl:.2f} net_ret={net_return_pct:.2f}% reason={reason} "
            f"cash=${self.cash:.2f}"
        )
        self._append_csv_and_log(csv_row, log_line)

        row = self._row(
            "VIRTUAL_SELL",
            now,
            role=p.get("role", "INITIAL"),
            leg_id=p.get("leg_id"),
            date=trade_date,
            symbol=p["symbol"],
            entry_time=str(p["entry_time"]),
            exit_time=str(now),
            entry_time_hms=entry_time_str,
            exit_time_hms=exit_time_str,
            entry_price=virtual_buy_price,
            exit_price=virtual_sell_price,
            quantity=qty,
            gross_pnl=gross_pnl,
            total_fee=total_fee,
            net_pnl=net_pnl,
            gross_return=(virtual_sell_price / virtual_buy_price) - 1.0,
            cost_reserve=total_fee / buy_notional if buy_notional > 0 else 0.0,
            net_return=net_return,
            net_return_pct=net_return_pct,
            reason=reason,
            entry_soxx=p["entry_soxx"],
            highest_etf=p["highest_etf"],
            orb_range=p["orb_range"],
            decision_id=p["decision_id"],
            model_version=p["model_version"],
            trigger_price=pending.get("trigger_price"),
            trigger_price_basis=pending.get("price_basis"),
            csv_row=csv_row,
            log_line=log_line,
            fill_assumption="NEXT_OBSERVED_WS_BID_NO_QUEUE_OR_MARKET_IMPACT_MODEL",
        )
        self.closed.append(row)
        return [row]

    def _append_csv_and_log(self, csv_row, log_line):
        if self.csv_path is not None:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
            with self.csv_path.open("a", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                if write_header:
                    writer.writerow(CSV_HEADER)
                writer.writerow(csv_row)
        self._write_log(log_line)

    def _evaluate_continuous_exit(self, p, now, symbol, price, price_basis):
        """Evaluate continuous exit conditions (Stop, Trailing, EOD) and -1% Chain Trigger."""
        if p["exit_pending"]:
            return []
        rows = []
        reason = None
        trigger_px = price

        # 1. Hard Stop Loss (ETF -2.0% only)
        if symbol == p["symbol"] and price is not None and price > 0:
            etf_ret = (price - p["entry_etf"]) / p["entry_etf"]
            if etf_ret <= self.hard_stop_etf_pct + EPS:
                reason = "Stop"

        # 2. Update Highest_ETF & Check Trailing Stop (+1.5% peak reached, then -0.3% drop from peak)
        if reason is None and symbol == p["symbol"] and price is not None and price > 0:
            p["highest_etf"] = max(p["highest_etf"], price)
            p["peak"] = p["highest_etf"]
            peak_ret = (p["highest_etf"] - p["entry_etf"]) / p["entry_etf"]
            if peak_ret >= self.trailing_trigger_pct - EPS:
                if not p.get("trailing_armed"):
                    p["trailing_armed"] = True
                    stop_line = p["highest_etf"] * (1.0 + self.trailing_drop_pct)
                    rows.append(
                        self._row(
                            "TRAILING_ARMED",
                            now,
                            role=p.get("role", "INITIAL"),
                            leg_id=p.get("leg_id"),
                            symbol=p["symbol"],
                            entry_etf=p["entry_etf"],
                            highest_etf=p["highest_etf"],
                            peak_return_pct=peak_ret * 100.0,
                            trailing_stop_price=stop_line,
                        )
                    )
                    self._write_log(
                        f"[{_parse_ny_time(now).strftime('%Y-%m-%d %H:%M:%S')}][TRAILING_ARMED] "
                        f"role={p.get('role', 'INITIAL')} symbol={p['symbol']} "
                        f"entry={p['entry_etf']:.4f} peak={p['highest_etf']:.4f} (+{peak_ret * 100.0:.2f}%) "
                        f"trail_stop={stop_line:.4f}"
                    )
                drawdown = (price - p["highest_etf"]) / p["highest_etf"]
                if drawdown <= self.trailing_drop_pct + EPS:
                    reason = "Trailing"

        # 3. EOD Time Exit (15:55 America/New_York)
        if reason is None and now.strftime("%H:%M") >= self.eod_exit_hm:
            reason = "EOD"

        # 4. Check -1.0% Opposite Chain Buy trigger on INITIAL leg (if not exiting right now)
        if reason is None and symbol == p["symbol"] and price is not None and price > 0:
            rows.extend(self._check_chain_trigger(p, now, price))

        if reason is not None:
            rows.extend(self._trigger_exit(p, now, reason, trigger_px, price_basis))
        return rows

    def on_quote(self, event):
        now = stamp(event["time"])
        symbol = event["symbol"]
        bid, ask = float(event["bid"]), float(event["ask"])
        if not (math.isfinite(bid) and math.isfinite(ask) and 0 < bid <= ask):
            raise ValueError("Invalid quote bid/ask")
        quote_time = stamp(event.get("quote_time", event["time"]))
        if event.get("observed_at") and stamp(event["observed_at"]) != quote_time:
            return []
        if not 0 <= (now - quote_time).total_seconds() <= CONFIG["MAIN_LOOP_TICK_OPEN"]:
            return []

        self.quotes[symbol] = dict(bid=bid, ask=ask, observed=quote_time)
        if symbol == self.settings["signal_ticker"]:
            self.last_soxx_price = (bid + ask) / 2.0

        rows = self._sync_db_bars(now)
        rows.extend(self._cancel_stale_buys(now))

        # 1. Fill any prior pending BUY or SELL orders for this symbol
        for pending in list(self.pending):
            if pending["symbol"] != symbol or now <= pending["time"] or quote_time <= pending["time"]:
                continue
            self.pending.remove(pending)
            if pending["action"] == "BUY":
                rows.extend(self._execute_buy(pending, now, ask, price_basis="WS_ASK"))
            elif pending["action"] == "SELL":
                rows.extend(self._execute_sell(pending, now, bid))

        # 2. Continuously monitor all open positions against this quote
        for p in list(self.positions):
            if not p["exit_pending"] and now > _parse_ny_time(p["entry_time"]):
                if symbol == p["symbol"]:
                    rows.extend(self._evaluate_continuous_exit(p, now, symbol, bid, "WS_BID"))
                elif symbol == self.settings["signal_ticker"]:
                    rows.extend(self._evaluate_continuous_exit(p, now, symbol, bid, "SOXX_WS_BID"))

        rows.extend(self._finish_cycle(now))
        return rows

    def market(self, event):
        now = stamp(event["time"])
        kind = event["kind"]
        if kind == "SOXX_5M_BAR":
            return self.on_soxx_5m_bar(
                bar_time=event.get("bar_time", event["time"]),
                open_px=event["open"],
                high_px=event["high"],
                low_px=event["low"],
                close_px=event["close"],
                volume=event["volume"],
                eval_time=now,
                exec_ask=event.get("exec_ask"),
                exec_bid=event.get("exec_bid"),
            )

        rows = self._sync_db_bars(now)
        rows.extend(self._cancel_stale_buys(now))

        symbol = event.get("symbol")
        price = float(event["price"]) if event.get("price") is not None else None
        if kind in ("TICK", "MARK", "POLL") and price is not None and price > 0:
            if not math.isfinite(price):
                raise ValueError("Invalid market price")
            if symbol == self.settings["signal_ticker"]:
                self.last_soxx_price = price

        for p in list(self.positions):
            if not p["exit_pending"]:
                basis = event.get("price_basis") or kind
                if kind == "CLOCK":
                    rows.extend(self._evaluate_continuous_exit(p, now, None, None, "CLOCK"))
                elif symbol in (p["symbol"], self.settings["signal_ticker"]) and price is not None and price > 0:
                    rows.extend(self._evaluate_continuous_exit(p, now, symbol, price, basis))
                elif now.strftime("%H:%M") >= self.eod_exit_hm:
                    rows.extend(self._evaluate_continuous_exit(p, now, None, None, "CLOCK"))

        rows.extend(self._finish_cycle(now))
        return rows


def attach_soxx_orb_shadow(engine, db=None, evaluation_window=None, csv_path=None, log_path=None):
    """Attach the SOXX ORB/VWAP/RVOL shadow strategy to an active shadow Engine."""
    settings = load_soxx_orb_settings()
    if not settings.get("enabled", True):
        return None
    candidate = SoxxOrbCandidate(
        settings=settings,
        ledger_kwargs=dict(db=db, csv_path=csv_path, log_path=log_path),
    )
    ledger = candidate.ledger_factory(candidate.name, "primary", evaluation_window)
    engine.candidates[candidate.name] = candidate
    engine.ledgers[(candidate.name, "primary")] = ledger
    return ledger
