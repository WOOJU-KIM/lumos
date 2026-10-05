import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import Dict, Any, List
from config import DATA_DIR
import config
from core.ml_engine import MLFeatureEngine

TRADE_LOGS_CSV = DATA_DIR / "trade_logs.csv"
TRADE_LOGS_SUMMARY_JSON = DATA_DIR / "trade_logs_summary.json"

class GranularBacktestEngine:
    """
    [머신러닝 기반 3중 타임프레임 TQQQ/SQQQ 자율 퀀트 백테스팅 엔진 (100% 전액 투입 챔피언 베이스라인)]
    (수정됨: 15분 진입 후 5분봉 궤적 추적, 0.20% 수수료 차감, ATR 트레일링 스탑 적용, 손절 비관적 편향)
    """
    def __init__(
        self,
        initial_capital_krw: int = 10_000_000,
        allocation_pct: float = 0.98,
        confidence_threshold: float = 0.40,
        take_profit_pct: float = 0.030,
        stop_loss_pct: float = -0.020,
        time_stop_bars: int = 6
    ):
        # 파라미터를 유지하되 실전 config 값을 우선적으로 사용합니다.
        self.initial_capital_krw = initial_capital_krw
        self.allocation_pct = config.MAX_ALLOCATION_RATIO
        self.confidence_threshold = config.GBDT_CONFIDENCE_THRESHOLD
        self.take_profit_pct = config.MAX_TP_PCT
        self.time_stop_bars = time_stop_bars
        self.ml_engine = MLFeatureEngine(confidence_threshold=config.GBDT_CONFIDENCE_THRESHOLD)

    def _compute_60m_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """60분봉 상위 추세 지표 계산"""
        df = df.copy()
        df['ema20'] = df['Close'].ewm(span=20, adjust=False).mean()
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        return df

    def _add_atr(self, df: pd.DataFrame, period=14) -> pd.DataFrame:
        """5분봉 데이터에 ATR(14) 계산 추가"""
        df = df.copy()
        df['Prev_Close'] = df['Close'].shift(1)
        df['TR1'] = df['High'] - df['Low']
        df['TR2'] = (df['High'] - df['Prev_Close']).abs()
        df['TR3'] = (df['Low'] - df['Prev_Close']).abs()
        df['TR'] = df[['TR1', 'TR2', 'TR3']].max(axis=1)
        df['ATR_14'] = df['TR'].rolling(window=period).mean()
        df.drop(['Prev_Close', 'TR1', 'TR2', 'TR3', 'TR'], axis=1, inplace=True)
        return df

    def run_backtest(self) -> Dict[str, Any]:
        """머신러닝 3중 타임프레임 백테스트 수행 및 성적표 산출"""
        from core.data_lake import MarketDataLake
        data_lake = MarketDataLake()
        
        tqqq_15m_raw = data_lake.load_candles("TQQQ", "15m")
        sqqq_15m_raw = data_lake.load_candles("SQQQ", "15m")
        tqqq_60m_raw = data_lake.load_candles("TQQQ", "60m")
        sqqq_60m_raw = data_lake.load_candles("SQQQ", "60m")
        soxx_60m_raw = data_lake.load_candles("SOXX", "60m")

        # 5분봉 추가 로드 (청산 궤적 추적용)
        tqqq_5m_raw = data_lake.load_candles("TQQQ", "5m")
        sqqq_5m_raw = data_lake.load_candles("SQQQ", "5m")

        if len(tqqq_15m_raw) < 100 or len(sqqq_15m_raw) < 100 or len(tqqq_5m_raw) < 100:
            # 야후 파이낸스 백업 로드 로직 (5m 포함)
            tqqq_15m_raw = yf.Ticker("TQQQ").history(period="60d", interval="15m")
            sqqq_15m_raw = yf.Ticker("SQQQ").history(period="60d", interval="15m")
            tqqq_60m_raw = yf.Ticker("TQQQ").history(period="60d", interval="60m")
            sqqq_60m_raw = yf.Ticker("SQQQ").history(period="60d", interval="60m")
            soxx_60m_raw = yf.Ticker("SOXX").history(period="60d", interval="60m")
            tqqq_5m_raw = yf.Ticker("TQQQ").history(period="60d", interval="5m")
            sqqq_5m_raw = yf.Ticker("SQQQ").history(period="60d", interval="5m")

            if tqqq_15m_raw.empty or sqqq_15m_raw.empty:
                raise RuntimeError("시세 데이터 수집 실패")
            
            data_lake.insert_candles("TQQQ", "15m", tqqq_15m_raw)
            data_lake.insert_candles("SQQQ", "15m", sqqq_15m_raw)
            data_lake.insert_candles("TQQQ", "60m", tqqq_60m_raw)
            data_lake.insert_candles("SQQQ", "60m", sqqq_60m_raw)
            data_lake.insert_candles("SOXX", "60m", soxx_60m_raw)
            data_lake.insert_candles("TQQQ", "5m", tqqq_5m_raw)
            data_lake.insert_candles("SQQQ", "5m", sqqq_5m_raw)

        # 2. 머신러닝 피처 추출 및 중요도 학습 (15분봉)
        tqqq_15m_feat = self.ml_engine.extract_features(tqqq_15m_raw)
        sqqq_15m_feat = self.ml_engine.extract_features(sqqq_15m_raw)

        if self.ml_engine.model is None or not self.ml_engine.feature_names:
            _, top_10_features, top_3_features = self.ml_engine.train_and_select_top_features(tqqq_15m_feat)
        else:
            top_10_features = self.ml_engine.top_10_features or self.ml_engine.feature_names[:10]
            top_3_features = self.ml_engine.top_3_features or top_10_features[:3]

        tqqq_15m_feat = self.ml_engine.add_confidence_columns(tqqq_15m_feat)
        sqqq_15m_feat = self.ml_engine.add_confidence_columns(sqqq_15m_feat)

        tqqq_60m = self._compute_60m_trend(tqqq_60m_raw)
        sqqq_60m = self._compute_60m_trend(sqqq_60m_raw)
        soxx_60m = self._compute_60m_trend(soxx_60m_raw)

        # 5분봉 ATR 계산 추가
        tqqq_5m = self._add_atr(tqqq_5m_raw)
        sqqq_5m = self._add_atr(sqqq_5m_raw)

        unique_dates = sorted(list(set(tqqq_15m_feat['date_str']).intersection(set(sqqq_15m_feat['date_str']))))

        # 3. 3중 타임프레임 스나이퍼 시뮬레이션
        sim_res = self._execute_triple_screen_simulation(
            tqqq_15m_feat, sqqq_15m_feat, tqqq_60m, sqqq_60m, soxx_60m, 
            tqqq_5m, sqqq_5m, unique_dates
        )

        sim_res["top_10_features"] = top_10_features
        sim_res["top_3_features"] = top_3_features
        sim_res["allocation_pct"] = self.allocation_pct
        sim_res["confidence_threshold"] = self.confidence_threshold
        sim_res["take_profit_pct"] = self.take_profit_pct
        sim_res["time_stop_minutes"] = self.time_stop_bars * 15

        all_trade_records = sim_res.get("all_trade_records", [])
        if all_trade_records:
            df_logs = pd.DataFrame(all_trade_records)
            df_logs.to_csv(TRADE_LOGS_CSV, index=False, encoding="utf-8-sig")
            
            with open(TRADE_LOGS_SUMMARY_JSON, "w", encoding="utf-8") as f:
                json.dump({
                    "total_trades_logged": len(all_trade_records),
                    "initial_capital_krw": self.initial_capital_krw,
                    "final_capital_krw": sim_res["final_capital_krw"],
                    "total_return_pct": sim_res["total_return_pct"],
                    "total_pnl_krw": sim_res["total_pnl_krw"],
                    "win_rate_pct": sim_res["win_rate_pct"],
                    "tqqq_win_rate_pct": sim_res["tqqq_win_rate_pct"],
                    "sqqq_win_rate_pct": sim_res["sqqq_win_rate_pct"],
                    "mdd_pct": sim_res["mdd_pct"],
                    "profit_factor": sim_res["profit_factor"],
                    "allocation_pct": self.allocation_pct,
                    "top_3_features": top_3_features,
                    "top_10_features": top_10_features,
                    "last_updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }, f, ensure_ascii=False, indent=2)

        return sim_res

    def _execute_triple_screen_simulation(
        self,
        tqqq_15m: pd.DataFrame,
        sqqq_15m: pd.DataFrame,
        tqqq_60m: pd.DataFrame,
        sqqq_60m: pd.DataFrame,
        soxx_60m: pd.DataFrame,
        tqqq_5m: pd.DataFrame,
        sqqq_5m: pd.DataFrame,
        unique_dates: List[str]
    ) -> Dict[str, Any]:
        capital = float(self.initial_capital_krw)
        equity_curve = [capital]
        daily_reports = []

        total_wins, total_losses, total_trades = 0, 0, 0
        tqqq_wins, tqqq_losses = 0, 0
        sqqq_wins, sqqq_losses = 0, 0

        all_closed_trades = []
        all_trade_records = []
        trade_id_seq = 1

        for date_str in unique_dates:
            day_tqqq_15m = tqqq_15m[tqqq_15m['date_str'] == date_str]
            day_sqqq_15m = sqqq_15m[sqqq_15m['date_str'] == date_str]
            
            if len(day_tqqq_15m) < 5 or len(day_sqqq_15m) < 5:
                continue

            day_start_capital = capital
            day_trades_count = 0
            day_wins = 0
            day_losses = 0

            current_pos = "NONE"
            entry_price = 0.0
            entry_time_str = ""
            pos_capital = 0.0
            
            # 동적 청산 파라미터
            dynamic_sl_pct = 0.0
            trailing_trigger_px = 0.0
            trailing_activated = False
            highest_since_entry = 0.0
            bars_held_5m = 0
            max_bars_held_5m = self.time_stop_bars * (15 // 5) # 90 minutes = 18 5m bars

            num_bars_15m = min(len(day_tqqq_15m), len(day_sqqq_15m))

            last_exit_time = pd.Timestamp.min

            for b_idx in range(num_bars_15m):
                row_l_15m = day_tqqq_15m.iloc[b_idx]
                row_s_15m = day_sqqq_15m.iloc[b_idx]
                current_15m_time = row_l_15m.name

                if current_15m_time <= last_exit_time:
                    continue

                if current_pos == "NONE":
                    if b_idx < 1: 
                        continue

                    # 60m Trend Check
                    past_soxx = soxx_60m[soxx_60m.index <= current_15m_time]
                    past_tqqq_60 = tqqq_60m[tqqq_60m.index <= current_15m_time]
                    past_sqqq_60 = sqqq_60m[sqqq_60m.index <= current_15m_time]

                    is_60m_bull, is_60m_bear = False, False
                    if len(past_soxx) >= 20 and len(past_tqqq_60) >= 20:
                        last_soxx = past_soxx.iloc[-1]
                        last_l60 = past_tqqq_60.iloc[-1]
                        last_s60 = past_sqqq_60.iloc[-1] if len(past_sqqq_60) >= 20 else last_l60

                        soxx_bull = (last_soxx['Close'] >= last_soxx['ema20'] * 0.998)
                        tqqq_bull = (last_l60['Close'] >= last_l60['ema20'] * 0.998) and (last_l60['macd'] >= last_l60['macd_signal'] * 0.98)
                        is_60m_bull = soxx_bull and tqqq_bull

                        soxx_bear = (last_soxx['Close'] <= last_soxx['ema20'] * 1.002)
                        sqqq_bull = (last_s60['Close'] >= last_s60['ema20'] * 0.998) and (last_s60['macd'] >= last_s60['macd_signal'] * 0.98)
                        is_60m_bear = soxx_bear and sqqq_bull

                    conf_tqqq = float(row_l_15m.get('Confidence', 0.50))
                    conf_sqqq = float(row_s_15m.get('Confidence', 0.50))

                    tqqq_dip_ok = (row_l_15m['VWAP_Diff'] <= 1.5) and (row_l_15m['RSI_14'] <= 62.0) and (row_l_15m['Close'] >= row_l_15m['BB_Lower'] * 1.001)
                    sqqq_dip_ok = (row_s_15m['VWAP_Diff'] <= 1.5) and (row_s_15m['RSI_14'] <= 62.0) and (row_s_15m['Close'] >= row_s_15m['BB_Lower'] * 1.001)

                    if is_60m_bull and (conf_tqqq >= self.confidence_threshold) and tqqq_dip_ok:
                        current_pos = "TQQQ"
                        entry_price = float(row_l_15m['Close'])
                    elif is_60m_bear and (conf_sqqq >= self.confidence_threshold) and sqqq_dip_ok:
                        current_pos = "SQQQ"
                        entry_price = float(row_s_15m['Close'])
                    
                    if current_pos != "NONE":
                        entry_time_str = str(current_15m_time)
                        pos_capital = capital * self.allocation_pct
                        
                        # 5m ATR 계산 및 동적 SL 설정 (Rule 3)
                        df_5m_ref = tqqq_5m if current_pos == "TQQQ" else sqqq_5m
                        past_5m = df_5m_ref[df_5m_ref.index <= current_15m_time]
                        if not past_5m.empty and not pd.isna(past_5m.iloc[-1].get('ATR_14')):
                            atr_val = past_5m.iloc[-1]['ATR_14']
                            atr_pct = (atr_val / entry_price) * config.SL_ATR_MULTIPLIER
                            dynamic_sl_pct = max(config.SL_MIN_PCT, min(config.SL_MAX_PCT, atr_pct))
                        else:
                            dynamic_sl_pct = 0.025 # 기본값

                        trailing_trigger_px = entry_price * (1.0 + config.TRAILING_TRIGGER_PCT)
                        trailing_activated = False
                        highest_since_entry = entry_price
                        bars_held_5m = 0
                        
                        # 5분봉 기반 청산 궤적 추적 (Rule 8)
                        future_5m = df_5m_ref[(df_5m_ref.index > current_15m_time) & (df_5m_ref.index.strftime('%Y-%m-%d') == date_str)]
                        
                        exit_price = 0.0
                        exit_reason = ""
                        exit_time_str = ""
                        
                        for i, (_, row_5m) in enumerate(future_5m.iterrows()):
                            bars_held_5m += 1
                            curr_high = float(row_5m['High'])
                            curr_low = float(row_5m['Low'])
                            curr_close = float(row_5m['Close'])
                            exit_time_str = str(row_5m.name)
                            
                            highest_since_entry = max(highest_since_entry, curr_high)
                            
                            # 비관적 편향 우선: 손절선(SL) 터치 검사
                            sl_px = entry_price * (1.0 - dynamic_sl_pct)
                            if curr_low <= sl_px:
                                exit_price = sl_px
                                exit_reason = f"STOP_LOSS_{dynamic_sl_pct*100:.1f}%"
                                break
                                
                            # 트레일링 스탑 청산 터치 검사
                            if trailing_activated:
                                trail_exit_px = highest_since_entry * (1.0 - config.TRAILING_DROP_PCT)
                                if curr_low <= trail_exit_px:
                                    exit_price = trail_exit_px
                                    exit_reason = "TRAILING_STOP_EXIT"
                                    break
                            
                            # 트레일링 스탑 활성화 검사
                            if not trailing_activated and curr_high >= trailing_trigger_px:
                                trailing_activated = True
                                
                            # Max TP 검사
                            max_tp_px = entry_price * (1.0 + config.MAX_TP_PCT)
                            if curr_high >= max_tp_px:
                                exit_price = max_tp_px
                                exit_reason = f"MAX_TP_{config.MAX_TP_PCT*100:.1f}%"
                                break
                                
                            # 타임스탑 검사
                            if bars_held_5m >= max_bars_held_5m:
                                exit_price = curr_close
                                exit_reason = f"TIME_STOP_{self.time_stop_bars*15}M"
                                break
                                
                            # 장 마감 종가 청산
                            if i == len(future_5m) - 1:
                                exit_price = curr_close
                                exit_reason = "END_OF_DAY_CLOSE"
                                break
                                
                        # 루프 종료 후 정산
                        if exit_price > 0:
                            raw_ret = (exit_price - entry_price) / entry_price
                            # 수수료/슬리피지 강제 차감 (Rule 4)
                            actual_ret = raw_ret - config.BACKTEST_FEE_SLIPPAGE
                            pnl_krw = int(pos_capital * actual_ret)
                            capital += pnl_krw
                            all_closed_trades.append(actual_ret)
                            
                            day_trades_count += 1
                            total_trades += 1
                            if actual_ret >= 0:
                                day_wins += 1
                                total_wins += 1
                                if current_pos == "TQQQ": tqqq_wins += 1
                                else: sqqq_wins += 1
                            else:
                                day_losses += 1
                                total_losses += 1
                                if current_pos == "TQQQ": tqqq_losses += 1
                                else: sqqq_losses += 1

                            all_trade_records.append({
                                "trade_id": f"TRD_{trade_id_seq:04d}",
                                "date": date_str,
                                "entry_time": entry_time_str,
                                "exit_time": exit_time_str,
                                "ticker": current_pos,
                                "entry_price": round(entry_price, 2),
                                "exit_price": round(exit_price, 2),
                                "pnl_pct": f"{actual_ret * 100:+.2f}%",
                                "pnl_krw": pnl_krw,
                                "capital_after": int(capital),
                                "exit_reason": exit_reason,
                                "bars_held": bars_held_5m
                            })
                            trade_id_seq += 1
                            current_pos = "NONE"
                            last_exit_time = pd.to_datetime(exit_time_str)

            day_pnl_krw = int(capital - day_start_capital)
            day_return_pct = round((day_pnl_krw / day_start_capital) * 100, 2) if day_start_capital > 0 else 0.0
            day_win_rate = round((day_wins / (day_wins + day_losses) * 100), 1) if (day_wins + day_losses) > 0 else 0.0

            daily_reports.append({
                "date": date_str,
                "date_short": date_str[5:].replace("-", "/"),
                "pnl_krw": day_pnl_krw,
                "return_pct": day_return_pct,
                "total_trades": day_trades_count,
                "wins": day_wins,
                "losses": day_losses,
                "win_rate_pct": day_win_rate,
                "is_cash_day": (day_trades_count == 0)
            })
            equity_curve.append(capital)

        weekly_reports = self._aggregate_weekly(daily_reports)

        total_pnl_krw = int(capital - self.initial_capital_krw)
        total_return_pct = round((total_pnl_krw / self.initial_capital_krw) * 100, 2)

        eq_arr = np.array(equity_curve)
        peak = np.maximum.accumulate(eq_arr)
        drawdown = (eq_arr - peak) / peak
        mdd_pct = round(float(abs(np.min(drawdown)) * 100), 2) if len(drawdown) > 0 else 0.0

        closed_arr = np.array(all_closed_trades) if len(all_closed_trades) > 0 else np.array([0.0])
        wins_arr = closed_arr[closed_arr > 0]
        losses_arr = closed_arr[closed_arr < 0]
        
        win_rate_pct = round((total_wins / (total_wins + total_losses) * 100), 1) if (total_wins + total_losses) > 0 else 0.0

        tqqq_total = tqqq_wins + tqqq_losses
        sqqq_total = sqqq_wins + sqqq_losses
        tqqq_win_rate_pct = round((tqqq_wins / tqqq_total * 100), 1) if tqqq_total > 0 else 0.0
        sqqq_win_rate_pct = round((sqqq_wins / sqqq_total * 100), 1) if sqqq_total > 0 else 0.0

        gross_profit = float(np.sum(wins_arr)) if len(wins_arr) > 0 else 0.001
        gross_loss = float(abs(np.sum(losses_arr))) if len(losses_arr) > 0 else 0.001
        profit_factor = round(gross_profit / gross_loss, 2)

        return {
            "initial_capital_krw": self.initial_capital_krw,
            "final_capital_krw": int(capital),
            "total_pnl_krw": total_pnl_krw,
            "total_return_pct": total_return_pct,
            "mdd_pct": mdd_pct,
            "total_trades_count": total_trades,
            "total_wins": total_wins,
            "total_losses": total_losses,
            "win_rate_pct": win_rate_pct,
            "tqqq_wins": tqqq_wins,
            "tqqq_losses": tqqq_losses,
            "tqqq_win_rate_pct": tqqq_win_rate_pct,
            "sqqq_wins": sqqq_wins,
            "sqqq_losses": sqqq_losses,
            "sqqq_win_rate_pct": sqqq_win_rate_pct,
            "profit_factor": profit_factor,
            "daily_reports": daily_reports,
            "weekly_reports": weekly_reports,
            "all_trade_records": all_trade_records
        }

    def _aggregate_weekly(self, daily_reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        weeks_map: Dict[str, List[Dict[str, Any]]] = {}

        for rep in daily_reports:
            dt = datetime.strptime(rep['date'], '%Y-%m-%d')
            year, week_num, _ = dt.isocalendar()
            week_key = f"{year}-W{week_num:02d}"
            
            if week_key not in weeks_map:
                weeks_map[week_key] = []
            weeks_map[week_key].append(rep)

        weekly_list = []
        w_idx = 1
        for w_key, days in weeks_map.items():
            w_pnl_krw = sum(d['pnl_krw'] for d in days)
            w_return_pct = round((w_pnl_krw / self.initial_capital_krw) * 100, 2)
            w_wins = sum(d['wins'] for d in days)
            w_losses = sum(d['losses'] for d in days)
            w_trades = sum(d['total_trades'] for d in days)
            w_win_rate = round((w_wins / (w_wins + w_losses) * 100), 1) if (w_wins + w_losses) > 0 else 0.0
            
            is_decay = w_return_pct < 0.0
            
            weekly_list.append({
                "week_name": f"{w_idx}주차",
                "week_code": w_key,
                "pnl_krw": w_pnl_krw,
                "return_pct": w_return_pct,
                "total_trades": w_trades,
                "wins": w_wins,
                "losses": w_losses,
                "win_rate_pct": w_win_rate,
                "is_decay": is_decay
            })
            w_idx += 1

        return weekly_list
