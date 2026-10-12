import config
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Any

class MarketRegimeClassifier:
    """
    [1. Market Regime Classifier (시장 국면 분석 엔진)]
    1~3년 장기 데이터와 VIX/SOX 지수를 분석하여 시장 국면(Bull Trend / Bear Trend / High Volatility Sideway / Consolidation)을 정의하고
    국면에 따라 전략 모드(TQQQ 롱 가중치, SQQQ 숏 가중치, 스위칭 임계값)를 동적으로 스위칭합니다.
    """
    def __init__(self, lookback_period: str = "2y"):
        self.lookback_period = lookback_period

    def classify(self, historical_data: Dict[str, pd.DataFrame] = None) -> Dict[str, Any]:
        """시장 국면 분류 및 적응형 퀀트 파라미터 산출"""
        if historical_data is None:
            # yfinance로 장기 데이터 로드 (TQQQ, SOXX, ^VIX)
            long_df = yf.Ticker(config.BASE_ASSET_LONG).history(period=self.lookback_period)
            macro_df = yf.Ticker(config.MACRO_TREND_SYMBOL).history(period=self.lookback_period)
            vix_df = yf.Ticker("^VIX").history(period=self.lookback_period)
        else:
            long_df = historical_data.get(config.BASE_ASSET_LONG)
            macro_df = historical_data.get(config.MACRO_TREND_SYMBOL, long_df)
            vix_df = historical_data.get("VIX")

        # 1. 테크니컬 지표 계산
        close_long = long_df['Close']
        current_long = float(close_long.iloc[-1])
        ma20 = float(close_long.tail(20).mean())
        ma50 = float(close_long.tail(50).mean())
        ma200 = float(close_long.tail(200).mean()) if len(close_long) >= 200 else float(close_long.mean())

        # VIX 지표
        current_vix = float(vix_df['Close'].iloc[-1])
        vix_ma20 = float(vix_df['Close'].tail(20).mean())
        vix_percentile = float((vix_df['Close'] < current_vix).mean() * 100)

        # 20일 실현 변동성 (연율화)
        returns = close_long.pct_change().dropna()
        realized_vol_20d = float(returns.tail(20).std() * np.sqrt(252) * 100)

        # SOXX(반도체 지수) 모멘텀 (20일 등락률)
        macro_close = macro_df['Close']
        macro_20d_perf = float((macro_close.iloc[-1] - macro_close.iloc[-20]) / macro_close.iloc[-20] * 100) if len(macro_close) >= 20 else 0.0

        # 2. 국면 판정 로직
        is_bull_ma = (current_long > ma50) and (ma50 > ma200)
        is_bear_ma = (current_long < ma50) and (ma50 < ma200)
        
        if current_vix >= 25.0 or realized_vol_20d >= 85.0:
            regime = "HIGH_VOLATILITY_SIDEWAY"
            regime_name_kr = "고변동성 횡보 / 경계 국면 (High Volatility)"
            description = "VIX 및 실현 변동성이 극도로 높은 구간입니다. 포지션 규모를 대폭 축소하고 타이트한 손절 및 빠른 차익실현이 필요합니다."
            params = {
                "long_weight_mult": 0.5,
                "short_weight_mult": 0.8,
                "dip_buy_threshold_pct": -1.2,
                "stop_loss_pct": -1.5,
                "hedge_switch_threshold_pct": -1.2,
                "target_regime_mode": "CAPITAL_PRESERVATION"
            }
        elif is_bull_ma and current_vix < 20.0:
            regime = "BULL_TREND"
            regime_name_kr = "강력한 상승 추세 국면 (Bull Trend)"
            description = "반도체 지수가 중장기 이평선 위에서 정배열을 형성하며 VIX가 20pt 미만으로 안정적입니다. TQQQ 선제 눌림목 매수를 적극 추천합니다."
            params = {
                "long_weight_mult": 1.2,
                "short_weight_mult": 0.3,
                "dip_buy_threshold_pct": -0.8,
                "stop_loss_pct": -2.0,
                "hedge_switch_threshold_pct": -2.2,
                "target_regime_mode": "AGGRESSIVE_LONG"
            }
        elif is_bear_ma or (current_long < ma200 and macro_20d_perf < -5.0):
            regime = "BEAR_TREND"
            regime_name_kr = "하락 추세 / 약세장 국면 (Bear Trend)"
            description = "중장기 이평선이 역배열이거나 지수 하방 압력이 큽니다. SQQQ 헤지 스위칭을 우선시하고 TQQQ 진입은 보수적으로 제한합니다."
            params = {
                "long_weight_mult": 0.4,
                "short_weight_mult": 1.5,
                "dip_buy_threshold_pct": -1.5,
                "stop_loss_pct": -1.5,
                "hedge_switch_threshold_pct": -1.0,
                "target_regime_mode": "HEDGE_PRIORITY"
            }
        else:
            regime = "CONSOLIDATION"
            regime_name_kr = "수렴 및 저변동 횡보 국면 (Consolidation)"
            description = "뚜렷한 추세 없이 박스권 내에서 수렴하는 국면입니다. 기본 원칙에 입각한 정량 분할 매매를 수행합니다."
            params = {
                "long_weight_mult": 1.0,
                "short_weight_mult": 0.6,
                "dip_buy_threshold_pct": -1.0,
                "stop_loss_pct": -2.0,
                "hedge_switch_threshold_pct": -1.8,
                "target_regime_mode": "STANDARD_BALANCED"
            }

        return {
            "regime": regime,
            "regime_name_kr": regime_name_kr,
            "description": description,
            "metrics": {
                "current_long": round(current_long, 2),
                "ma20": round(ma20, 2),
                "ma50": round(ma50, 2),
                "ma200": round(ma200, 2),
                "current_vix": round(current_vix, 2),
                "vix_percentile": round(vix_percentile, 1),
                "realized_vol_20d_pct": round(realized_vol_20d, 1),
                "macro_20d_perf_pct": round(macro_20d_perf, 2)
            },
            "strategy_params": params
        }
