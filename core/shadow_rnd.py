import json
import config
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from typing import Dict, Any, List
from config import SHADOW_RND_CACHE_FILE

class ShadowRndEngine:
    """
    [3. Shadow R&D Engine (백그라운드 차세대 로직 연구)]
    메인 로직이 가동 중인 동안 백그라운드에서 다양한 진입/손절/스위칭 파라미터를
    지속적으로 백테스트하여 더 나은 대체 로직(Candidate Logic)을 상시 발굴하고 랭킹을 산출합니다.
    """
    
    # 상시 연구 대상 후보 전략 파라미터 풀
    CANDIDATE_POOL = [
        {
            "id": "logic_v1_0_baseline",
            "name": "Base Alpha (원칙 기준)",
            "dip_buy_pct": -1.0,
            "stop_loss_pct": -2.0,
            "hedge_switch_pct": -1.5,
            "take_profit_pct": 4.5,
            "description": "대표님 원안: -1% 선제 매수, -2% 칼손절, -1.5% 스위칭"
        },
        {
            "id": "logic_v1_1_agile_scalp",
            "name": "Agile Scalper (고기동 스캘핑)",
            "dip_buy_pct": -0.8,
            "stop_loss_pct": -1.5,
            "hedge_switch_pct": -1.0,
            "take_profit_pct": 3.5,
            "description": "민감한 -0.8% 선제진입, -1.5% 타이트 손절, 빠른 SQQQ 헤지 전환"
        },
        {
            "id": "logic_v1_2_deep_swing",
            "name": "Deep Swing (보수적 눌림 스윙)",
            "dip_buy_pct": -1.3,
            "stop_loss_pct": -2.5,
            "hedge_switch_pct": -2.0,
            "take_profit_pct": 6.0,
            "description": "깊은 눌림목(-1.3%) 대기, 넉넉한 익절(+6.0%) 타겟"
        },
        {
            "id": "logic_v1_3_vix_adaptive",
            "name": "Volatility Sniper (변동성 스나이퍼)",
            "dip_buy_pct": -0.9,
            "stop_loss_pct": -1.8,
            "hedge_switch_pct": -1.2,
            "take_profit_pct": 4.0,
            "description": "중간 변동성 최적화: -0.9% 진입, -1.8% 손절, -1.2% 스위칭"
        },
        {
            "id": "logic_v1_4_bear_shield",
            "name": "Bear Shield (하방 방어형)",
            "dip_buy_pct": -1.5,
            "stop_loss_pct": -1.5,
            "hedge_switch_pct": -0.8,
            "take_profit_pct": 3.0,
            "description": "극단적 하방 방어: SQQQ 스위칭을 -0.8%로 최우선 가동"
        }
    ]

    def __init__(self, lookback_period: str = "1y"):
        self.lookback_period = lookback_period

    def run_backtest_simulation(self, historical_data: Dict[str, pd.DataFrame] = None) -> Dict[str, Any]:
        """과거 데이터에 대해 모든 후보 로직을 동시 백테스팅하여 랭킹 도출"""
        
        if historical_data is None:
            long_df = yf.Ticker(config.BASE_ASSET_LONG).history(period=self.lookback_period)
            short_df = yf.Ticker(config.BASE_ASSET_SHORT).history(period=self.lookback_period)
            vix_df = yf.Ticker("^VIX").history(period=self.lookback_period)
        else:
            long_df = historical_data.get(config.BASE_ASSET_LONG)
            short_df = historical_data.get(config.BASE_ASSET_SHORT)
            vix_df = historical_data.get("VIX")

        # 공통 일자 인덱스 정렬
        common_index = long_df.index.intersection(short_df.index).intersection(vix_df.index)
        long_close = long_df.loc[common_index, 'Close']
        short_close = short_df.loc[common_index, 'Close']
        vix_close = vix_df.loc[common_index, 'Close']
        
        candidates_results = []

        for candidate in self.CANDIDATE_POOL:
            res = self._simulate_single_logic(candidate, long_close, short_close, vix_close)
            candidates_results.append(res)

        # 복합 점수(Sharpe * 40 + WinRate * 30 + ProfitFactor * 20 + Return * 10) 기준 정렬
        candidates_results.sort(key=lambda x: x["composite_score"], reverse=True)
        
        champion = candidates_results[0]
        
        output = {
            "tested_period": f"Recent {self.lookback_period} ({len(common_index)} bars)",
            "total_candidates": len(candidates_results),
            "champion_logic": champion,
            "rankings": candidates_results
        }

        # 캐시 저장
        try:
            with open(SHADOW_RND_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        return output

    def _simulate_single_logic(self, logic_cfg: Dict[str, Any], tqqq: pd.Series, sqqq: pd.Series, vix: pd.Series) -> Dict[str, Any]:
        """단일 파라미터 셋에 대한 시뮬레이션 계산"""
        
        dip_buy = logic_cfg["dip_buy_pct"] / 100.0
        stop_loss = logic_cfg["stop_loss_pct"] / 100.0
        hedge_switch = logic_cfg["hedge_switch_pct"] / 100.0
        take_profit = logic_cfg["take_profit_pct"] / 100.0

        long_ret = tqqq.pct_change().fillna(0).values
        short_ret = sqqq.pct_change().fillna(0).values
        vix_vals = vix.values

        trades = []
        equity_curve = [1.0]
        
        in_pos = "NONE" # config.BASE_ASSET_LONG, config.BASE_ASSET_SHORT, "NONE"
        entry_price_long = 0.0
        entry_price_short = 0.0
        current_equity = 1.0

        for i in range(1, len(long_ret)):
            day_ret_long = long_ret[i]
            day_ret_short = short_ret[i]
            
            # 포지션이 없는 경우
            if in_pos == "NONE":
                # TQQQ 눌림목 진입 조건
                if day_ret_long <= dip_buy and vix_vals[i] < 28.0:
                    in_pos = config.BASE_ASSET_LONG
                    entry_price_long = tqqq.iloc[i]
                elif day_ret_long <= hedge_switch or vix_vals[i] >= 28.0:
                    in_pos = config.BASE_ASSET_SHORT
                    entry_price_short = sqqq.iloc[i]
            
            # TQQQ 보유 중
            elif in_pos == config.BASE_ASSET_LONG:
                current_pnl = day_ret_long
                # 칼손절 및 헤지 스위칭 검사
                if current_pnl <= stop_loss or current_pnl <= hedge_switch:
                    # 손절 후 즉시 SQQQ 헤지 스위칭
                    current_equity *= (1.0 + current_pnl)
                    trades.append(current_pnl)
                    in_pos = config.BASE_ASSET_SHORT if current_pnl <= hedge_switch else "NONE"
                elif current_pnl >= take_profit:
                    # 익절 청산
                    current_equity *= (1.0 + current_pnl)
                    trades.append(current_pnl)
                    in_pos = "NONE"
                else:
                    current_equity *= (1.0 + current_pnl * 0.8) # 일부 보유
            
            # SQQQ 보유 중
            elif in_pos == config.BASE_ASSET_SHORT:
                current_pnl = day_ret_short
                if current_pnl <= stop_loss:
                    current_equity *= (1.0 + current_pnl)
                    trades.append(current_pnl)
                    in_pos = "NONE"
                elif current_pnl >= take_profit:
                    current_equity *= (1.0 + current_pnl)
                    trades.append(current_pnl)
                    in_pos = "NONE"
                else:
                    current_equity *= (1.0 + current_pnl * 0.8)
                    
            equity_curve.append(current_equity)

        # 지표 산출
        trades_arr = np.array(trades) if len(trades) > 0 else np.array([0.01])
        wins = trades_arr[trades_arr > 0]
        losses = trades_arr[trades_arr < 0]
        
        win_rate = float((len(wins) / len(trades_arr)) * 100) if len(trades_arr) > 0 else 50.0
        total_return = float((equity_curve[-1] - 1.0) * 100)
        
        # MDD 계산
        eq_arr = np.array(equity_curve)
        peak = np.maximum.accumulate(eq_arr)
        drawdown = (eq_arr - peak) / peak
        mdd = float(abs(np.min(drawdown)) * 100) if len(drawdown) > 0 else 0.0
        
        # Sharpe 계산 (일별 수익률 기준 연율화)
        daily_diffs = np.diff(equity_curve) / np.array(equity_curve[:-1])
        sharpe = float((np.mean(daily_diffs) / (np.std(daily_diffs) + 1e-6)) * np.sqrt(252)) if len(daily_diffs) > 1 else 1.0
        
        gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.01
        gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 0.01
        profit_factor = round(gross_profit / gross_loss, 2)

        composite_score = round(sharpe * 35.0 + (win_rate * 0.3) + (profit_factor * 15.0) + (max(0, 40 - mdd) * 0.5), 2)

        return {
            "id": logic_cfg["id"],
            "name": logic_cfg["name"],
            "parameters": {
                "dip_buy_pct": logic_cfg["dip_buy_pct"],
                "stop_loss_pct": logic_cfg["stop_loss_pct"],
                "hedge_switch_pct": logic_cfg["hedge_switch_pct"],
                "take_profit_pct": logic_cfg["take_profit_pct"]
            },
            "description": logic_cfg["description"],
            "metrics": {
                "total_return_pct": round(total_return, 2),
                "win_rate_pct": round(win_rate, 1),
                "sharpe_ratio": round(sharpe, 2),
                "mdd_pct": round(mdd, 2),
                "profit_factor": profit_factor,
                "trades_count": len(trades_arr)
            },
            "composite_score": composite_score
        }
