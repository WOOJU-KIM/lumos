import logging
logger = logging.getLogger(__name__)
import config
from config import GBDT_CONFIDENCE_THRESHOLD
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BASE_DIR, MODELS_DIR
from core.data_lake import MarketDataLake
from core.heterogeneous_models import CrossAssetDislocationModel
from core.ml_engine import MLFeatureEngine

MOE_MODEL_PATH = MODELS_DIR / "model_moe_orchestrator.pkl"

EXPERT_NAMES = [
    "cross_asset",   # 크로스에셋 인과 괴리 모델 (NVDA/QQQ 선행 시차 기반)
    "gbdt_pattern"   # 3중 타임프레임 파형 GBDT 스나이퍼 모델
]

EXPERT_DESCRIPTIONS = {
    "cross_asset": "크로스에셋 인과 괴리 스나이퍼 (NVDA/QQQ 리드)",
    "gbdt_pattern": "GBDT 3-Class 파형 스나이퍼 (보조지표 피처 패턴)",
    "ensemble_consensus": "듀얼 AI 합의 스나이퍼 (크로스에셋+GBDT 동시 승인)",
    "hybrid_moe_v3": f"하이브리드 MoE v3 (GBDT {config.GBDT_CONFIDENCE_THRESHOLD*100:.1f}% 공격수 + 크로스에셋 Veto 방패 통과)",
    "conflict_rejected": "AI 신호 충돌 / 거시 역풍 차단 (크로스에셋 Veto 발동으로 매매 취소)"
}

class MoEMetaOrchestrator:
    """
    [Lumos 듀얼 챔피언 MoE 오케스트레이터: 크로스에셋 + GBDT 파형 듀얼 합의 (Dual Consensus Veto)]
    - 듀얼 합의 (AND 조건) 다중 에이전트 의사결정 아키텍처
    - 크로스에셋(>=60%)과 GBDT 파형(>=60%)의 방향이 100% 일치할 때만 최종 승인
    - 한쪽이라도 미달하거나 방향 상충 시 100% 매매 취소(Veto) ➔ 휩쏘 및 불필요한 칼손절 원천 차단
    """
    def __init__(
        self,
        confidence_threshold: float = GBDT_CONFIDENCE_THRESHOLD,
        cross_asset_threshold: float = GBDT_CONFIDENCE_THRESHOLD,
        gbdt_threshold: float = GBDT_CONFIDENCE_THRESHOLD,
        mode: str = "hybrid_v3"
    ):
        self.confidence_threshold = confidence_threshold
        self.cross_asset_threshold = cross_asset_threshold
        self.gbdt_threshold = gbdt_threshold
        self.mode = mode
        self.data_lake = MarketDataLake()
        self.cross_asset_model = CrossAssetDislocationModel(dislocation_z_threshold=1.6)
        self.gbdt_engine = MLFeatureEngine(confidence_threshold=0.40)
        self._try_load_default_models()
        self.experts = {
            "cross_asset": self.cross_asset_model,
            "gbdt_pattern": self.gbdt_engine
        }

    def _try_load_default_models(self):
        """디스크에 직렬화된 최신 챔피언 모델 자동 로드"""
        if getattr(self.gbdt_engine, "model", None) is not None:
            return

        import __main__
        if not hasattr(__main__, "MoEMetaOrchestrator"):
            __main__.MoEMetaOrchestrator = MoEMetaOrchestrator

        for p in [MOE_MODEL_PATH, Path(__file__).resolve().parent.parent / "models" / "model_champion.pkl"]:
            if p.exists():
                try:
                    obj = joblib.load(p)
                    if hasattr(obj, "gbdt_engine") and getattr(obj.gbdt_engine, "model", None) is not None:
                        self.gbdt_engine.model = obj.gbdt_engine.model
                        self.gbdt_engine.feature_names = getattr(obj.gbdt_engine, "feature_names", [])
                        self.gbdt_engine.top_10_features = getattr(obj.gbdt_engine, "top_10_features", [])
                        self.gbdt_engine.top_3_features = getattr(obj.gbdt_engine, "top_3_features", [])
                        break
                except Exception:
                    pass

    def compute_regime_vector(self, current_time_str: Optional[str] = None) -> Dict[str, float]:
        """현재 시장의 5대 상태 벡터 산출"""
        long_5m = self.data_lake.load_candles(config.TRADE_SYMBOLS[0], "5m")
        vix_15m = self.data_lake.load_candles(config.CROSS_ASSET_SYMBOLS[3] if len(config.CROSS_ASSET_SYMBOLS)>3 else "^VIX", "15m")
        nvda_5m = self.data_lake.load_candles(config.CROSS_ASSET_SYMBOLS[0], "5m")
        qqq_5m = self.data_lake.load_candles(config.CROSS_ASSET_SYMBOLS[1], "5m")

        elapsed_min = 120.0
        if current_time_str:
            try:
                t = datetime.strptime(current_time_str[-8:], "%H:%M:%S").time()
                elapsed_min = float(max(0, min(390, (t.hour - 9) * 60 + (t.minute - 30))))
            except Exception:
                pass

        vix_level = 16.5
        if not vix_15m.empty:
            vix_c = vix_15m['Close'] if 'Close' in vix_15m else vix_15m['close']
            vix_level = float(vix_c.iloc[-1])

        atr_ratio = 1.05
        if not long_5m.empty and len(long_5m) >= 25:
            s_h = long_5m['High'] if 'High' in long_5m else long_5m['high']
            s_l = long_5m['Low'] if 'Low' in long_5m else long_5m['low']
            recent_atr = (s_h - s_l).tail(5).mean()
            avg_atr = (s_h - s_l).tail(25).mean()
            if avg_atr > 0:
                atr_ratio = round(float(recent_atr / avg_atr), 3)

        cvd_delta = 1.28
        if not long_5m.empty and len(long_5m) >= 10:
            s_c = long_5m['Close'] if 'Close' in long_5m else long_5m['close']
            s_o = long_5m['Open'] if 'Open' in long_5m else long_5m['open']
            s_h = long_5m['High'] if 'High' in long_5m else long_5m['high']
            s_l = long_5m['Low'] if 'Low' in long_5m else long_5m['low']
            s_v = long_5m['Volume'] if 'Volume' in long_5m else long_5m['volume']
            vol_delta = (s_c - s_o) / (s_h - s_l + 1e-6) * s_v
            cvd_delta = round(float(vol_delta.tail(5).mean() / (s_v.tail(20).mean() + 1e-6)), 3)

        dislocation_lag = 0.0
        if (
            not nvda_5m.empty and not qqq_5m.empty and not long_5m.empty and
            len(nvda_5m) >= 5 and len(qqq_5m) >= 5 and len(long_5m) >= 5
        ):
            n_c = nvda_5m['Close'] if 'Close' in nvda_5m else nvda_5m['close']
            q_c = qqq_5m['Close'] if 'Close' in qqq_5m else qqq_5m['close']
            s_c = long_5m['Close'] if 'Close' in long_5m else long_5m['close']
            nvda_ret = (n_c.iloc[-1] / n_c.iloc[-5] - 1.0) * 100
            qqq_ret = (q_c.iloc[-1] / q_c.iloc[-5] - 1.0) * 100
            long_ret = (s_c.iloc[-1] / s_c.iloc[-5] - 1.0) * 100
            macro_expected = (nvda_ret * 0.6 + qqq_ret * 0.4) * 3.0
            dislocation_lag = round(float(macro_expected - long_ret), 3)

        return {
            "elapsed_min": elapsed_min,
            "vix_level": vix_level,
            "atr_ratio": atr_ratio,
            "cvd_delta": cvd_delta,
            "dislocation_lag": dislocation_lag
        }

    def calculate_dynamic_targets(self, df_15m: pd.DataFrame, cur_px: float) -> Dict[str, float]:
        """목표 익절가(+3.0%) 및 칼손절가(-2.0%) 산출"""
        if cur_px <= 0:
            if df_15m is not None and not df_15m.empty:
                c = df_15m['Close'] if 'Close' in df_15m else df_15m.get('close')
                if c is not None and not c.empty:
                    cur_px = float(c.iloc[-1])
            if cur_px <= 0:
                cur_px = 100.0

        tp_px = round(cur_px * (1.0 + config.MAX_TP_PCT), 2)
        sl_px = round(cur_px * (1.0 - config.SL_MIN_PCT), 2)
        atr_14 = 0.0
        try:
            if df_15m is not None and not df_15m.empty and len(df_15m) >= 14:
                h = df_15m['High'] if 'High' in df_15m else df_15m['high']
                l = df_15m['Low'] if 'Low' in df_15m else df_15m['low']
                atr_14 = round(float((h - l).tail(14).mean()), 2)
        except Exception:
            pass
        return {
            "cur_px": cur_px,
            "dynamic_tp_px": tp_px,
            "dynamic_sl_px": sl_px,
            "hard_cap_sl_px": sl_px,
            "tp_pct": config.MAX_TP_PCT * 100.0,
            "sl_pct": config.SL_MIN_PCT * 100.0,
            "atr_14": atr_14
        }

    def evaluate_dual_filter_signal(
        self,
        df_candle_15m: pd.DataFrame,
        current_time_str: Optional[str] = None,
        threshold: Optional[float] = None,
        live_prices: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        [듀얼 챔피언: 크로스에셋 + GBDT 실시간 의사결정 집행]
        1단계: 시장 레짐 벡터 및 최근 수익률 산출
        2단계: 크로스에셋 및 GBDT 모델 각각의 신호 및 확신도(Confidence) 동시 산출
        3단계: MoE 확신도 점수 직접 비교를 통한 Top-1 승자 모델 선정 (동등 경쟁 점수제)
        4단계: 3중 스크린(60m 추세 & 5m 눌림목) 및 듀얼 합의(Cross>=60% & GBDT>=60%) 최종 승인
        """
        # 방어 가드: 빈 데이터프레임 또는 데이터 부족 시 안전 반환
        if df_candle_15m is None or df_candle_15m.empty or len(df_candle_15m) < 15:
            return {
                "selected_expert": "none",
                "expert_desc": "데이터 부족 (신호 산출 불가)",
                "gating_confidence": 0.0,
                "expert_confidence": 0.0,
                "threshold_applied": float(threshold if threshold is not None else self.gbdt_threshold),
                "regime_snapshot": {},
                "direction": "NONE",
                "is_approved": False,
                "is_60m_trend_ok": False,
                "dip_ok": False,
                "all_gating_confidences": {"cross_asset": 0.5, "gbdt_pattern": 0.5},
                "timestamp": current_time_str or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

        # 0. 타임스탬프 동기화
        if current_time_str is None:
            if isinstance(df_candle_15m.index, pd.DatetimeIndex):
                current_time_str = df_candle_15m.index[-1].strftime("%Y-%m-%d %H:%M:%S")
            elif "datetime" in df_candle_15m.columns:
                current_time_str = str(df_candle_15m["datetime"].iloc[-1])

        regime_vec = self.compute_regime_vector(current_time_str)

        # 1. 시세 데이터 및 수익률 산출
        c = df_candle_15m['close'] if 'close' in df_candle_15m else df_candle_15m['Close']
        ret_5 = float((c.iloc[-1] / c.iloc[-5] - 1.0)) if len(c) >= 5 else 0.0

        # 2. [크로스에셋: 방향성 검토 (역풍 방패 Veto)]
        dir_cross = "NONE"
        conf_cross = 0.50
        is_cross_veto = False
        def _get_live_df(sym, default_dt):
            lp = live_prices.get(sym, 0.0) if live_prices else 0.0
            if lp > 0:
                return self.data_lake.get_candles_with_live_tick(sym, "15m", live_price=lp)
            if current_time_str:
                return self.data_lake.load_candles(sym, "15m", end_dt=current_time_str)
            return self.data_lake.load_candles(sym, "15m")

        try:
            def _get_c(sym):
                df = _get_live_df(sym, current_time_str)
                if df is None or df.empty: return pd.Series()
                return df['Close'] if 'Close' in df else df['close']
            
            n_c = _get_c("NVDA")
            q_c = _get_c("QQQ")
            v_c = _get_c("VIX")
            s_c = _get_c(config.TRADE_SYMBOLS[0])
            macro_c = _get_c(config.MACRO_TREND_SYMBOL)
            
            if len(n_c) >= 5 and len(q_c) >= 5 and len(v_c) >= 5 and len(s_c) >= 5:
                nvda_r = float(n_c.iloc[-1] / n_c.iloc[-5] - 1.0)
                qqq_r = float(q_c.iloc[-1] / q_c.iloc[-5] - 1.0)
                vix_r = float(v_c.iloc[-1] / v_c.iloc[-5] - 1.0)
                macro_r = float(macro_c.iloc[-1] / macro_c.iloc[-5] - 1.0) if not macro_c.empty else nvda_r
                long_r = float(s_c.iloc[-1] / s_c.iloc[-5] - 1.0)

                sig_code, exp_conf, _ = getattr(self, "cross_asset_model", self).predict_signal(
                    long_ret=long_r, nvda_ret=nvda_r, macro_ret=macro_r, qqq_ret=qqq_r, vix_ret=vix_r, tnx_ret=0.0
                ) if hasattr(self, "cross_asset_model") else (0, 0.5, "")
                
                conf_cross = exp_conf
                if sig_code > 0: dir_cross = "LONG"
                elif sig_code < 0: dir_cross = "SHORT"
        except Exception as e:
            logger.info(f"CROSS_ASSET EXCEPTION: {e}")
            pass

        # 3. [GBDT 3-Class 파형 스나이퍼 모델 신호 및 확신도 계산]
        dir_gbdt = "NONE"
        conf_gbdt = 0.50
        gbdt_probs = {"LONG": 0.33, "SHORT": 0.33, "NONE": 0.34}
        try:
            live_long_15m = _get_live_df(config.TRADE_SYMBOLS[0], current_time_str) if "live_prices" in locals() else df_candle_15m
            gbdt_sig, gbdt_conf, _, gbdt_probs = self.gbdt_engine.predict_signal_full(live_long_15m)
            conf_gbdt = gbdt_conf
            if gbdt_sig == 1:
                dir_gbdt = "LONG"
            elif gbdt_sig == -1:
                dir_gbdt = "SHORT"
        except Exception:
            pass

        is_cross_veto = (
            (dir_gbdt == "LONG" and dir_cross == "SHORT") or
            (dir_gbdt == "SHORT" and dir_cross == "LONG")
        )

        if not getattr(config, "USE_CROSS_ASSET_VETO", True):
            is_cross_veto = False



        # 4. [MoE 의사결정 집행]
        cross_hurdle = getattr(self, "cross_asset_threshold", GBDT_CONFIDENCE_THRESHOLD)
        gbdt_hurdle = threshold if threshold is not None else getattr(self, "gbdt_threshold", GBDT_CONFIDENCE_THRESHOLD)

        if getattr(self, "mode", "hybrid_v3") == "hybrid_v3":
            # -----------------------------------------------------------------
            # [Lumos V3 하이브리드 MoE 의사결정]
            # - 조건 A (공격수): GBDT가 TQQQ 또는 SQQQ 방향 제시 및 확신도 >= gbdt_hurdle (GBDT_CONFIDENCE_THRESHOLD)
            # - 방패(Veto): 크로스에셋 역방향 검출 시 진입 차단 (Veto)
            # - 멀티스크린 필터: SOXX 60분봉 추세 및 5분봉 RSI 눌림목 확인
            # -----------------------------------------------------------------
            is_gbdt_trigger = (dir_gbdt in ["LONG", "SHORT"]) and (conf_gbdt >= gbdt_hurdle)

            if is_gbdt_trigger:
                selected_expert = "hybrid_moe_v3"
                direction = dir_gbdt
                final_conf = conf_gbdt
            else:
                selected_expert = "gbdt_pattern" if conf_gbdt >= 0.50 else "cross_asset"
                direction = "NONE"
                final_conf = conf_gbdt
        else:
            # [레거시 안전 모델: 듀얼 합의 (Dual Consensus AND 로직)]
            is_consensus = (
                dir_cross in ["LONG", "SHORT"] and
                dir_cross == dir_gbdt and
                conf_cross >= cross_hurdle and
                conf_gbdt >= gbdt_hurdle
            )
            if is_consensus:
                selected_expert = "ensemble_consensus"
                direction = dir_cross
                final_conf = max(conf_cross, conf_gbdt)
            else:
                if dir_cross in ["LONG", "SHORT"] and dir_gbdt in ["LONG", "SHORT"] and dir_cross != dir_gbdt:
                    selected_expert = "conflict_rejected"
                else:
                    selected_expert = "cross_asset" if conf_cross >= conf_gbdt else "gbdt_pattern"
                direction = "NONE"
                final_conf = max(conf_cross, conf_gbdt)

        # 5. [3중 스크린 검증] - QQQ 60분봉 추세
        is_60m_trend_ok = True
        try:
            qqq_live = live_prices.get("QQQ", 0.0) if live_prices else 0.0
            if qqq_live > 0:
                qqq_60m = self.data_lake.get_candles_with_live_tick("QQQ", "60m", live_price=qqq_live)
            else:
                qqq_60m = self.data_lake.load_candles("QQQ", "60m")
                
            if not qqq_60m.empty and len(qqq_60m) >= 20:
                s_c = qqq_60m['Close'] if 'Close' in qqq_60m else qqq_60m['close']
                qqq_c = s_c.iloc[-1]
                qqq_ema = s_c.ewm(span=config.QQQ_EMA_PERIOD, adjust=False).mean().iloc[-1]
                if direction == "LONG":
                    is_60m_trend_ok = (qqq_c >= qqq_ema * 0.998)
                elif direction == "SHORT":
                    is_60m_trend_ok = (qqq_c <= qqq_ema * 1.002)
        except Exception:
            pass

        if not getattr(config, "USE_60M_TREND_FILTER", True):
            is_60m_trend_ok = True

        df_feat = getattr(self, "gbdt_engine", self).extract_features(df_candle_15m, live_prices=live_prices) if hasattr(self, "gbdt_engine") else df_candle_15m
        last_row = df_feat.iloc[-1] if not df_feat.empty else {}

        dip_ok = True
        try:
            rsi_5m = float(last_row.get("RSI_14", 50.0))
            if direction == "LONG" and rsi_5m > config.RSI_OVERBOUGHT_THRESHOLD:
                dip_ok = False
            elif direction == "SHORT" and rsi_5m < config.RSI_OVERSOLD_THRESHOLD:
                dip_ok = False
        except Exception:
            pass

        # 6. [최종 매수 승인]
        is_approved = bool(direction in ["LONG", "SHORT"] and not is_cross_veto and is_60m_trend_ok and dip_ok)
        all_confidences = {
            "cross_asset": round(conf_cross, 4),
            "gbdt_pattern": round(conf_gbdt, 4)
        }

        # AI 재학습을 위한 피처 스냅샷 추출
        feat_dict = {}
        try:
            for col in df_feat.columns:
                val = last_row[col]
                if isinstance(val, (int, float)) and not pd.isna(val):
                    feat_dict[col] = round(float(val), 4)
        except Exception:
            pass

        decision_meta = {
            "selected_expert": str(selected_expert),
            "expert_desc": str(EXPERT_DESCRIPTIONS.get(selected_expert, selected_expert)),
            "gating_confidence": float(final_conf),
            "expert_confidence": float(final_conf),
            "threshold_applied": float(gbdt_hurdle),
            "regime_snapshot": regime_vec,
            "direction": str(direction),
            "cross_dir": str(dir_cross),
            "dir_gbdt": str(dir_gbdt),
            "is_approved": bool(is_approved),
            "is_60m_trend_ok": bool(is_60m_trend_ok),
            "dip_ok": bool(dip_ok),
            "all_gating_confidences": all_confidences,
            "gbdt_probs": gbdt_probs,
            "features": feat_dict,
            "timestamp": current_time_str or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return decision_meta

def train_and_save_moe_orchestrator(confidence_threshold: float = GBDT_CONFIDENCE_THRESHOLD, mode: str = "hybrid_v3") -> MoEMetaOrchestrator:
    logger.info(f"🚀 [훈련 개시] Lumos 듀얼 챔피언 MoE 스나이퍼 학습 중... (임계값 {confidence_threshold*100:.0f}%)")
    orchestrator = MoEMetaOrchestrator(confidence_threshold=confidence_threshold, gbdt_threshold=confidence_threshold, mode=mode)
    data_lake = MarketDataLake()
    long_15m = data_lake.load_candles(config.TRADE_SYMBOLS[0], "15m")
    if not long_15m.empty and len(long_15m) >= 100:
        orchestrator.gbdt_engine.train_and_select_top_features(long_15m)

    # 1. 신규 메인 실전 하이브리드 V3 모델 저장
    joblib.dump(orchestrator, MOE_MODEL_PATH)
    joblib.dump(orchestrator, Path(__file__).resolve().parent.parent / "models" / "model_champion.pkl")
    logger.info(f"✅ [Lumos V3 하이브리드 MoE (GBDT {config.GBDT_CONFIDENCE_THRESHOLD*100:.1f}% + Cross-Asset Veto) 실전 모델 저장 완료] ➔ {MOE_MODEL_PATH}")

    # 2. 레거시 안전 모델(100% 동시합의)도 독립 파일로 영구 동시 보존
    safe_orchestrator = MoEMetaOrchestrator(confidence_threshold=GBDT_CONFIDENCE_THRESHOLD, gbdt_threshold=GBDT_CONFIDENCE_THRESHOLD, mode="legacy_dual_consensus")
    safe_orchestrator.gbdt_engine = orchestrator.gbdt_engine
    safe_path = Path(__file__).resolve().parent.parent / "models" / "model_moe_legacy_safe.pkl"
    safe_champ_path = Path(__file__).resolve().parent.parent / "models" / "model_champion_legacy_safe.pkl"
    joblib.dump(safe_orchestrator, safe_path)
    joblib.dump(safe_orchestrator, safe_champ_path)
    logger.info(f"✅ [Lumos 레거시 안전 모델(100% 동시합의) 동시 보존 완료] ➔ {safe_path.name}")

    return orchestrator

if __name__ == "__main__":
    moe = train_and_save_moe_orchestrator()
    lake = MarketDataLake()
    tqqq = lake.load_candles(config.TRADE_SYMBOLS[0], "15m")
    sample_eval = moe.evaluate_dual_filter_signal(tqqq)
    logger.info("Lumos V3 Hybrid MoE Sample Decision:")
    logger.info(json.dumps(sample_eval, indent=2, ensure_ascii=False))

