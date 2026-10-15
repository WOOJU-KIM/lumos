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
        mode: str = "hybrid_v3",
        manifest_path: Optional[Path] = None
    ):
        self.confidence_threshold = confidence_threshold
        self.cross_asset_threshold = cross_asset_threshold
        self.gbdt_threshold = gbdt_threshold
        self.mode = mode
        self.data_lake = MarketDataLake()
        self.cross_asset_model = CrossAssetDislocationModel(dislocation_z_threshold=1.6)
        self.gbdt_engine = MLFeatureEngine(
            confidence_threshold=0.40,
            feature_schema="B_normalized",
            manifest_path=manifest_path
        )
        self._try_load_default_models()
        self.experts = {
            "cross_asset": self.cross_asset_model,
            "gbdt_pattern": self.gbdt_engine
        }

    def _try_load_default_models(self):
        """디스크에 직렬화된 실전 운영 모델을 매니페스트 및 7대 무결성 검증을 거쳐 로드"""
        if getattr(self.gbdt_engine, "model", None) is not None:
            return
        self.gbdt_engine._try_load_default_model()

    def get_active_model_info(self) -> Dict[str, Any]:
        """현재 MoE 오케스트레이터가 사용 중인 GBDT 실전 모델 상태 반환"""
        return self.gbdt_engine.get_active_model_info()

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
            early_snap = self.gbdt_engine.get_model_snapshot() if hasattr(self, "gbdt_engine") and hasattr(self.gbdt_engine, "get_model_snapshot") else None
            e_sha = early_snap.active_model_sha256 if early_snap is not None else getattr(getattr(self, "gbdt_engine", None), "active_model_sha256", "")
            e_load = early_snap.load_status if early_snap is not None else getattr(getattr(self, "gbdt_engine", None), "model_load_status", "")
            e_fb = early_snap.fallback_status if early_snap is not None else getattr(getattr(self, "gbdt_engine", None), "model_fallback_status", "")
            return {
                "selected_expert": "none",
                "expert_desc": "데이터 부족 (신호 산출 불가)",
                "gating_confidence": 0.0,
                "expert_confidence": 0.0,
                "threshold_applied": float(threshold if threshold is not None else self.gbdt_threshold),
                "regime_snapshot": {},
                "direction": "NONE",
                "cross_dir": "NONE",
                "dir_gbdt": "NONE",
                "is_approved": False,
                "is_cross_veto": False,
                "is_60m_trend_ok": False,
                "dip_ok": False,
                "all_gating_confidences": {"cross_asset": 0.5, "gbdt_pattern": 0.5},
                "gbdt_probs": {"LONG": 0.33, "SHORT": 0.33, "NONE": 0.34},
                "active_model_id": early_snap.active_model_id if early_snap is not None else getattr(getattr(self, "gbdt_engine", None), "active_model_id", ""),
                "active_model_sha256": e_sha,
                "active_model_version": (e_sha or "")[:12] if e_sha else "NONE",
                "active_model_schema": early_snap.feature_schema if early_snap is not None else getattr(getattr(self, "gbdt_engine", None), "feature_schema", ""),
                "active_model_num_features": len(early_snap.feature_names) if early_snap is not None else len(getattr(getattr(self, "gbdt_engine", None), "feature_names", []) or []),
                "active_model_load_status": e_load,
                "active_model_fallback_status": e_fb,
                "active_model_is_fallback": bool(e_load == "FALLBACK_TO_CHAMPION" or "FALLBACK" in str(e_fb)),
                "model_hot_reloaded": False,
                "model_consistency_ok": bool(early_snap is not None and early_snap.model is not None and e_load != "BLOCKED"),
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
            v_c = _get_c(getattr(config, "VIX_SYMBOL", "VIXY"))
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

        # 3. [GBDT 3-Class 파형 스나이퍼 모델 신호 및 확신도 계산 - 단일 불변 스냅샷 격리]
        dir_gbdt = "NONE"
        conf_gbdt = 0.50
        gbdt_probs = {"LONG": 0.33, "SHORT": 0.33, "NONE": 0.34}
        gbdt_snap = None
        prev_snap = None
        model_hot_reloaded = False
        consistency_res = {"ok": True, "reason": "CONSISTENT"}
        try:
            if hasattr(self, "gbdt_engine"):
                if hasattr(self.gbdt_engine, "get_model_snapshot"):
                    prev_snap = self.gbdt_engine.get_model_snapshot()
                if hasattr(self.gbdt_engine, "ensure_latest_manifest_model"):
                    model_hot_reloaded = bool(self.gbdt_engine.ensure_latest_manifest_model())
                if hasattr(self.gbdt_engine, "get_model_snapshot"):
                    gbdt_snap = self.gbdt_engine.get_model_snapshot()
                if hasattr(self.gbdt_engine, "verify_manifest_memory_snapshot_consistency"):
                    consistency_res = self.gbdt_engine.verify_manifest_memory_snapshot_consistency(snapshot=gbdt_snap)
            live_long_15m = _get_live_df(config.TRADE_SYMBOLS[0], current_time_str) if "live_prices" in locals() else df_candle_15m
            if consistency_res.get("ok", True):
                if gbdt_snap is not None:
                    gbdt_sig, gbdt_conf, _, gbdt_probs = self.gbdt_engine.predict_signal_full(live_long_15m, snapshot=gbdt_snap)
                else:
                    gbdt_sig, gbdt_conf, _, gbdt_probs = self.gbdt_engine.predict_signal_full(live_long_15m)
                conf_gbdt = gbdt_conf
                if gbdt_sig == 1:
                    dir_gbdt = "LONG"
                elif gbdt_sig == -1:
                    dir_gbdt = "SHORT"
            else:
                logger.error(f"🚨 [신규 진입 차단] 모델 스냅샷/메모리/매니페스트 불일치 또는 BLOCKED 상태 감지: {consistency_res}")
                dir_gbdt = "NONE"
                conf_gbdt = 0.34
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

        if not consistency_res.get("ok", True):
            direction = "NONE"

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

        # 6. [최종 매수 승인] (모델 무결성/일치성 실패 시 무조건 False 강제)
        is_approved = bool(
            consistency_res.get("ok", True)
            and direction in ["LONG", "SHORT"]
            and not is_cross_veto
            and is_60m_trend_ok
            and dip_ok
        )
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

        snap_id = gbdt_snap.active_model_id if gbdt_snap is not None else getattr(self.gbdt_engine, "active_model_id", "")
        snap_sha = gbdt_snap.active_model_sha256 if gbdt_snap is not None else getattr(self.gbdt_engine, "active_model_sha256", "")
        snap_schema = gbdt_snap.feature_schema if gbdt_snap is not None else getattr(self.gbdt_engine, "feature_schema", "")
        snap_n_feat = len(gbdt_snap.feature_names) if gbdt_snap is not None else len(getattr(self.gbdt_engine, "feature_names", []) or [])
        snap_load_status = gbdt_snap.load_status if gbdt_snap is not None else getattr(self.gbdt_engine, "model_load_status", "")
        snap_fb_status = gbdt_snap.fallback_status if gbdt_snap is not None else getattr(self.gbdt_engine, "model_fallback_status", "")
        snap_path = gbdt_snap.active_model_path if gbdt_snap is not None else getattr(self.gbdt_engine, "active_model_path", "")
        snap_act_at = gbdt_snap.activated_at if gbdt_snap is not None else getattr(self.gbdt_engine, "activated_at", "")

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
            "is_cross_veto": bool(is_cross_veto),
            "is_60m_trend_ok": bool(is_60m_trend_ok),
            "dip_ok": bool(dip_ok),
            "all_gating_confidences": all_confidences,
            "gbdt_probs": gbdt_probs,
            "features": feat_dict,
            "active_model_id": snap_id,
            "active_model_sha256": snap_sha,
            "active_model_version": (snap_sha or "")[:12] if snap_sha else "NONE",
            "active_model_schema": snap_schema,
            "active_model_num_features": snap_n_feat,
            "active_model_load_status": snap_load_status,
            "active_model_fallback_status": snap_fb_status,
            "active_model_is_fallback": bool(snap_load_status == "FALLBACK_TO_CHAMPION" or "FALLBACK" in str(snap_fb_status)),
            "active_model_path": snap_path,
            "active_model_activated_at": snap_act_at,
            "previous_model_sha256_before_eval": prev_snap.active_model_sha256 if prev_snap is not None else "",
            "model_hot_reloaded": bool(model_hot_reloaded),
            "model_consistency_ok": bool(consistency_res.get("ok", True)),
            "model_consistency_reason": str(consistency_res.get("reason", "CONSISTENT")),
            "timestamp": current_time_str or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return decision_meta


import threading
import time
from contextlib import contextmanager

_MANIFEST_ACTIVATION_LOCK = threading.RLock()


@contextmanager
def _acquire_manifest_activation_lock(target_dir: Path, timeout_sec: float = 30.0):
    """프로세스 내 RLock + 파일 시스템 원자적 락 파일(.manifest_activation.lock) 이중 보호"""
    lock_file = target_dir / ".manifest_activation.lock"
    with _MANIFEST_ACTIVATION_LOCK:
        deadline = time.monotonic() + timeout_sec
        fd = None
        while True:
            try:
                fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()}:{threading.get_ident()}:{time.time_ns()}".encode("utf-8"))
                break
            except FileExistsError:
                # 오래된 죽은 프로세스 락 파일 정리 (60초 초과)
                try:
                    if lock_file.exists() and (time.time() - lock_file.stat().st_mtime) > 60.0:
                        lock_file.unlink()
                        continue
                except Exception:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Timed out waiting for manifest activation lock: {lock_file}")
                time.sleep(0.01)
        try:
            yield
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except Exception:
                    pass
            try:
                if lock_file.exists():
                    lock_file.unlink()
            except Exception:
                pass


def _write_json_atomic_temp(tmp_path: Path, payload: Dict[str, Any]) -> None:
    """임시 JSON 파일을 디스크에 기록하고 flush + fsync 수행"""
    raw_bytes = json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")
    with open(tmp_path, "wb") as f_out:
        f_out.write(raw_bytes)
        f_out.flush()
        os.fsync(f_out.fileno())


def train_and_save_moe_orchestrator(
    confidence_threshold: float = GBDT_CONFIDENCE_THRESHOLD,
    mode: str = "hybrid_v3",
    schema: str = "B_normalized",
    df_15m: Optional[pd.DataFrame] = None,
    models_dir_override: Optional[Path] = None,
    live_experience_df: Optional[pd.DataFrame] = None,
    _job_started_at_ns: Optional[int] = None,
    notifier: Optional[Any] = None,
) -> MoEMetaOrchestrator:
    """
    [STEP 2 원자적 모델 활성화 아키텍처]
    1. 신규 학습 모델을 고유 임시 파일(.tmp)에 저장 후 flush + fsync
    2. 7대 무결성 검증 + 10년 학습 윈도우 규정 준수 검증
    3. 불변 버전 파일(models/versions/model_normalized_72_<timestamp>_<sha256prefix>.pkl)로 확정 (기존 PKL 덮어쓰기 0건)
    4. 활성화 락 획득 후 동시 재학습 구버전 덮어쓰기(stale job) 차단 검증
    5. 오직 active_model_manifest.json 단 1개만 os.replace(tmp_manifest_path, manifest_path)로 원자적 전환
    """
    import hashlib
    from core.system_logger import system_logger

    job_started_at_ns = int(_job_started_at_ns if _job_started_at_ns is not None else time.time_ns())

    if schema in MLFeatureEngine.LIVE_FORBIDDEN_SCHEMAS or schema != "B_normalized":
        raise ValueError(f"Live retraining strictly requires schema='B_normalized' (got '{schema}')")

    target_dir = Path(models_dir_override) if models_dir_override is not None else MODELS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    versions_dir = target_dir / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = target_dir / "active_model_manifest.json"
    unique_tag = f"{os.getpid()}.{threading.get_ident()}.{job_started_at_ns}"
    tmp_model_path = versions_dir / f".model_normalized_72.pkl.tmp.{unique_tag}"
    tmp_manifest_path = target_dir / f".active_model_manifest.json.tmp.{unique_tag}"

    def _safe_notify_retrain(status_code: str, **kwargs):
        if notifier is not None and hasattr(notifier, "send_retrain_status_alert"):
            try:
                notifier.send_retrain_status_alert(status=status_code, **kwargs)
            except Exception as ne:
                logger.warning(f"Telegram retrain notification failed (ignored): {ne}")

    logger.info(f"🚀 [훈련 개시] Lumos 듀얼 챔피언 MoE 스나이퍼 학습 중... (스키마: {schema}, 임계값 {confidence_threshold*100:.0f}%)")
    orchestrator = MoEMetaOrchestrator(
        confidence_threshold=confidence_threshold,
        gbdt_threshold=confidence_threshold,
        mode=mode,
        manifest_path=manifest_path if manifest_path.exists() else None
    )
    orchestrator.gbdt_engine.feature_schema = schema

    if df_15m is not None:
        long_15m = df_15m.copy()
    else:
        data_lake = MarketDataLake()
        long_15m = data_lake.load_candles(config.TRADE_SYMBOLS[0], "15m")

    if long_15m is None or long_15m.empty or len(long_15m) < 100:
        err_msg = "Insufficient training data (<100 bars); aborting retraining to preserve existing active model."
        system_logger.log("ERROR", "ModelActivation", f"MODEL_ACTIVATION_FAILED | reason={err_msg}")
        _safe_notify_retrain(
            "MODEL_ACTIVATION_FAILED",
            model_id="B_Normalized_72",
            schema=schema,
            num_features=72,
            error_reason=err_msg,
        )
        raise RuntimeError(err_msg)

    trained_model, _, _ = orchestrator.gbdt_engine.train_and_select_top_features(
        long_15m,
        schema=schema,
        exclude_unconfirmed_tail=True,
        live_experience_df=live_experience_df
    )
    if trained_model is None or orchestrator.gbdt_engine.model is None:
        err_msg = "Retraining failed to produce a valid LightGBM model; preserving existing active model."
        system_logger.log("ERROR", "ModelActivation", f"MODEL_ACTIVATION_FAILED | reason={err_msg}")
        _safe_notify_retrain(
            "MODEL_ACTIVATION_FAILED",
            model_id="B_Normalized_72",
            schema=schema,
            num_features=72,
            error_reason=err_msg,
        )
        raise RuntimeError(err_msg)

    canonical_features = list(MLFeatureEngine.CANONICAL_B_NORMALIZED_72_FEATURES)
    canonical_features_sha256 = MLFeatureEngine.compute_feature_names_sha256(canonical_features)

    try:
        # 1. 임시 파일로 신규 모델 저장 + 디스크 flush/fsync
        orchestrator.gbdt_engine.active_model_id = "B_Normalized_72"
        with open(tmp_model_path, "wb") as f_out:
            joblib.dump(orchestrator, f_out)
            f_out.flush()
            os.fsync(f_out.fileno())

        cand_sha256 = hashlib.sha256(tmp_model_path.read_bytes()).hexdigest()
        window_meta = getattr(orchestrator.gbdt_engine, "last_training_window_meta", {})
        system_logger.log(
            "INFO",
            "AutoRetrain",
            f"RETRAIN_COMPLETED | schema={schema} | tmp_path={tmp_model_path} | sha256={cand_sha256} | train_rows={window_meta.get('train_rows', 0)} | span_days={window_meta.get('span_days', 0)}"
        )
        _safe_notify_retrain(
            "RETRAIN_COMPLETED",
            model_id="B_Normalized_72",
            schema=schema,
            num_features=72,
            new_sha256=cand_sha256,
            training_window_meta=window_meta,
        )

        # 2~5. SHA256, 스키마·피처 수·순서, 클래스 매핑([0,1,2]), 추론 self-test 검증
        ver = MLFeatureEngine.verify_model_artifact(
            tmp_model_path,
            expected_schema="B_normalized",
            expected_num_features=72,
            expected_feature_names=canonical_features,
            expected_feature_names_sha256=canonical_features_sha256,
            enforce_live_policy=True
        )
        if not ver["ok"]:
            raise RuntimeError(f"Trained B_normalized candidate failed 7-point verification: {ver['error']}")

        # 6. 10년 학습 윈도우 규정 준수 검증
        max_lookback_days = float(getattr(config, "TRAINING_LOOKBACK_DAYS", config.ROLLING_TRAINING_WEEKS * 7))
        if not window_meta.get("is_valid_window", False) or float(window_meta.get("span_days", 0.0)) > max_lookback_days + 1e-6:
            raise RuntimeError(f"Trained B_normalized candidate failed 10-year window compliance check: {window_meta}")

        actual_sha256 = ver["actual_sha256"]
        sha_prefix = actual_sha256[:12]
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        versioned_filename = f"model_normalized_72_{ts_str}_{sha_prefix}.pkl"
        versioned_model_path = versions_dir / versioned_filename
        if versioned_model_path.exists():
            # 기존 불변 버전 파일 절대 덮어쓰기 금지: 동일 초 내 생성 시 고유 타임스탬프 접미사 부여
            versioned_filename = f"model_normalized_72_{ts_str}_{job_started_at_ns % 1000000:06d}_{sha_prefix}.pkl"
            versioned_model_path = versions_dir / versioned_filename

        # 7. 불변 버전 파일로 확정 (기존 활성 모델 PKL 덮어쓰기 없음)
        os.replace(tmp_model_path, versioned_model_path)

        # 버전 파일 확정 후 재검증
        ver_final = MLFeatureEngine.verify_model_artifact(
            versioned_model_path,
            expected_sha256=actual_sha256,
            expected_schema="B_normalized",
            expected_num_features=72,
            expected_feature_names=canonical_features,
            expected_feature_names_sha256=canonical_features_sha256,
            enforce_live_policy=True
        )
        if not ver_final["ok"]:
            raise RuntimeError(f"Versioned B_normalized artifact failed post-move verification: {ver_final['error']}")

        system_logger.log(
            "INFO",
            "ModelValidation",
            f"MODEL_VALIDATION_PASSED | model_id=B_Normalized_72 | schema=B_normalized | num_features=72 | sha256={actual_sha256} | path={versioned_model_path}"
        )
        _safe_notify_retrain(
            "MODEL_VALIDATION_PASSED",
            model_id="B_Normalized_72",
            schema="B_normalized",
            num_features=72,
            new_sha256=actual_sha256,
            training_window_meta=window_meta,
        )

        # 8. 활성화 락 획득 후 manifest 단일 원자적 교체 수행
        with _acquire_manifest_activation_lock(target_dir):
            prev_model_path = "models/model_normalized_72.pkl"
            prev_model_sha256 = ""
            if manifest_path.exists():
                try:
                    existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    existing_job_ns = int(existing_manifest.get("job_started_at_ns", 0) or 0)
                    if existing_job_ns > job_started_at_ns:
                        active_sha = str(existing_manifest.get("expected_sha256") or "")
                        system_logger.log(
                            "WARNING",
                            "ModelActivation",
                            f"STALE_RETRAIN_IGNORED | candidate_job_ns={job_started_at_ns} | active_job_ns={existing_job_ns} | candidate_path={versioned_model_path}"
                        )
                        _safe_notify_retrain(
                            "STALE_RETRAIN_IGNORED",
                            model_id="B_Normalized_72",
                            schema="B_normalized",
                            num_features=72,
                            new_sha256=actual_sha256,
                            prev_sha256=active_sha,
                            training_window_meta=window_meta,
                        )
                        raise RuntimeError(
                            f"STALE_RETRAIN_IGNORED: Newer model already activated (active_job_ns={existing_job_ns} > candidate_job_ns={job_started_at_ns})"
                        )
                    prev_model_path = str(
                        existing_manifest.get("model_path")
                        or existing_manifest.get("active_model_path")
                        or prev_model_path
                    )
                    prev_model_sha256 = str(existing_manifest.get("expected_sha256") or "")
                except RuntimeError:
                    raise
                except Exception:
                    pass

            champ_path = target_dir / "model_champion.pkl"
            if not champ_path.exists():
                champ_path = MODELS_DIR / "model_champion.pkl"
            champ_sha = hashlib.sha256(champ_path.read_bytes()).hexdigest() if champ_path.exists() else ""

            try:
                rel_active_path = versioned_model_path.relative_to(BASE_DIR).as_posix()
            except ValueError:
                rel_active_path = str(versioned_model_path)

            try:
                rel_fallback_path = champ_path.relative_to(BASE_DIR).as_posix()
            except ValueError:
                rel_fallback_path = str(champ_path)

            activated_at_iso = datetime.now().astimezone().isoformat()
            manifest_data = {
                "active_model_id": "B_Normalized_72",
                "model_path": rel_active_path,
                "active_model_path": rel_active_path,
                "expected_sha256": actual_sha256,
                "expected_schema": "B_normalized",
                "num_features": 72,
                "expected_num_features": 72,
                "feature_names": canonical_features,
                "feature_names_sha256": canonical_features_sha256,
                "activated_at": activated_at_iso,
                "previous_model_path": prev_model_path,
                "previous_model_sha256": prev_model_sha256,
                "fallback_model_id": "v4_champion_fallback",
                "fallback_model_path": rel_fallback_path,
                "fallback_expected_schema": "v4_champion",
                "fallback_expected_num_features": 72,
                "fallback_expected_sha256": champ_sha,
                "forbidden_schemas": sorted(list(MLFeatureEngine.LIVE_FORBIDDEN_SCHEMAS)),
                "job_started_at_ns": job_started_at_ns,
                "training_window_meta": {
                    "config_rolling_training_weeks": window_meta.get("config_rolling_training_weeks"),
                    "config_training_lookback_days": window_meta.get("config_training_lookback_days"),
                    "train_start_dt": window_meta.get("train_start_dt"),
                    "train_end_dt": window_meta.get("train_end_dt"),
                    "train_rows": window_meta.get("train_rows"),
                    "span_days": window_meta.get("span_days"),
                    "excluded_older_than_10y_rows": window_meta.get("excluded_older_than_10y_rows"),
                    "excluded_warmup_nan_rows": window_meta.get("excluded_warmup_nan_rows"),
                    "excluded_unconfirmed_tail_rows": window_meta.get("excluded_unconfirmed_tail_rows")
                },
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            _write_json_atomic_temp(tmp_manifest_path, manifest_data)
            parsed_manifest = json.loads(tmp_manifest_path.read_text(encoding="utf-8"))
            if (
                parsed_manifest.get("active_model_id") != "B_Normalized_72"
                or parsed_manifest.get("model_path") != rel_active_path
                or parsed_manifest.get("expected_schema") != "B_normalized"
                or int(parsed_manifest.get("num_features", 0)) != 72
                or parsed_manifest.get("expected_sha256") != actual_sha256
                or parsed_manifest.get("feature_names_sha256") != canonical_features_sha256
            ):
                raise RuntimeError("Candidate manifest failed validation before atomic swap.")

            # 단일 원자적 교체: 오직 manifest_path 1개만 os.replace 수행
            os.replace(tmp_manifest_path, manifest_path)

        orchestrator.gbdt_engine.active_model_id = "B_Normalized_72"
        orchestrator.gbdt_engine.active_model_path = str(versioned_model_path)
        orchestrator.gbdt_engine.active_model_sha256 = actual_sha256
        orchestrator.gbdt_engine.active_model_activated_at = activated_at_iso
        orchestrator.gbdt_engine.active_model_load_status = "LOADED_PRIMARY"
        orchestrator.gbdt_engine._loaded_manifest_sha256 = actual_sha256
        orchestrator.gbdt_engine._manifest_path = manifest_path
        orchestrator.gbdt_engine._bound_to_default_manifest = True
        orchestrator.gbdt_engine._sync_snapshot_from_attributes()

        system_logger.log(
            "INFO",
            "ModelActivation",
            f"MODEL_ACTIVATION_SUCCEEDED | model_id=B_Normalized_72 | path={versioned_model_path} | sha256={actual_sha256} | previous_path={prev_model_path}"
        )
        _safe_notify_retrain(
            "MODEL_ACTIVATION_SUCCEEDED",
            model_id="B_Normalized_72",
            schema="B_normalized",
            num_features=72,
            new_sha256=actual_sha256,
            prev_sha256=prev_model_sha256,
            training_window_meta=window_meta,
        )
        logger.info(f"✅ [B_Normalized_72 불변 버전 저장 및 단일 매니페스트 원자적 활성화 완료] ➔ {versioned_model_path} (SHA256={actual_sha256[:16]}...)")
    except Exception as act_err:
        if "STALE_RETRAIN_IGNORED" not in str(act_err):
            existing_sha = ""
            if manifest_path.exists():
                try:
                    existing_sha = str(json.loads(manifest_path.read_text(encoding="utf-8")).get("expected_sha256") or "")
                except Exception:
                    pass
            system_logger.log(
                "ERROR",
                "ModelActivation",
                f"MODEL_ACTIVATION_FAILED | reason={act_err} | preserved_sha256={existing_sha}"
            )
            _safe_notify_retrain(
                "MODEL_ACTIVATION_FAILED",
                model_id="B_Normalized_72",
                schema="B_normalized",
                num_features=72,
                prev_sha256=existing_sha,
                error_reason=str(act_err),
            )
        raise
    finally:
        # 9. 실패 또는 완료 후 임시 파일 정리 (기존 활성 모델 및 버전 PKL 100% 보존 보장)
        for p in (tmp_model_path, tmp_manifest_path):
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

    return orchestrator


def rollback_active_model_manifest(
    target: str = "previous",
    models_dir_override: Optional[Path] = None,
    notifier: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    [Scenario H 롤백 헬퍼]
    - target="previous": active_model_manifest.json의 previous_model_path / previous_model_sha256로 단일 os.replace 롤백
    - target="champion": 기존 Champion(models/model_champion.pkl, v4_champion, 72피처)으로 단일 os.replace 롤백
    """
    from core.system_logger import system_logger

    target_dir = Path(models_dir_override) if models_dir_override is not None else MODELS_DIR
    manifest_path = target_dir / "active_model_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with _acquire_manifest_activation_lock(target_dir):
        current_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cur_path_str = str(current_manifest.get("model_path") or current_manifest.get("active_model_path") or "")
        cur_sha = str(current_manifest.get("expected_sha256") or "")

        if target == "previous":
            target_path_str = str(current_manifest.get("previous_model_path") or "")
            target_sha = str(current_manifest.get("previous_model_sha256") or "")
            if not target_path_str:
                raise RuntimeError("No previous_model_path recorded in active_model_manifest.json")
            resolved_path = Path(target_path_str)
            if not resolved_path.is_absolute():
                cand_in_target = target_dir / resolved_path.name
                cand_versions_in_target = target_dir / "versions" / resolved_path.name
                resolved_path = (BASE_DIR / resolved_path) if (BASE_DIR / resolved_path).exists() else (
                    cand_in_target if cand_in_target.exists() else (
                        cand_versions_in_target if cand_versions_in_target.exists() else (target_dir.parent / resolved_path)
                    )
                )
            ver = MLFeatureEngine.verify_model_artifact(
                resolved_path,
                expected_sha256=target_sha if target_sha else None,
                enforce_live_policy=True,
            )
            if not ver["ok"]:
                raise RuntimeError(f"Rollback target ('previous') failed verification: {ver['error']}")
            target_schema = ver["detected_schema"]
            target_id = "B_Normalized_72" if target_schema == "B_normalized" else "v4_champion"
            feat_names = list(ver["extracted"]["feature_names"])
        elif target == "champion":
            champ_path = target_dir / "model_champion.pkl"
            if not champ_path.exists():
                champ_path = MODELS_DIR / "model_champion.pkl"
            ver = MLFeatureEngine.verify_model_artifact(
                champ_path,
                expected_schema="v4_champion",
                expected_num_features=72,
                enforce_live_policy=True,
            )
            if not ver["ok"]:
                raise RuntimeError(f"Rollback target ('champion') failed verification: {ver['error']}")
            try:
                target_path_str = champ_path.relative_to(BASE_DIR).as_posix()
            except ValueError:
                target_path_str = str(champ_path)
            target_sha = ver["actual_sha256"]
            target_schema = "v4_champion"
            target_id = "v4_champion"
            feat_names = list(ver["extracted"]["feature_names"])
        else:
            raise ValueError(f"Unsupported rollback target: {target}")

        activated_at_iso = datetime.now().astimezone().isoformat()
        updated_manifest = dict(current_manifest)
        updated_manifest.update({
            "active_model_id": target_id,
            "model_path": target_path_str,
            "active_model_path": target_path_str,
            "expected_sha256": ver["actual_sha256"],
            "expected_schema": target_schema,
            "num_features": len(feat_names),
            "expected_num_features": len(feat_names),
            "feature_names": feat_names,
            "feature_names_sha256": ver["feature_names_sha256"],
            "activated_at": activated_at_iso,
            "previous_model_path": cur_path_str,
            "previous_model_sha256": cur_sha,
            "job_started_at_ns": time.time_ns(),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

        tmp_manifest = target_dir / f".active_model_manifest.rollback.tmp.{os.getpid()}.{time.time_ns()}"
        try:
            _write_json_atomic_temp(tmp_manifest, updated_manifest)
            os.replace(tmp_manifest, manifest_path)
        finally:
            if tmp_manifest.exists():
                try:
                    tmp_manifest.unlink()
                except Exception:
                    pass

        system_logger.log(
            "WARNING",
            "ModelRollback",
            f"MODEL_ROLLBACK_EXECUTED | target={target} | active_model_id={target_id} | path={target_path_str} | sha256={ver['actual_sha256']}"
        )
        if notifier is not None and hasattr(notifier, "send_model_state_alert"):
            try:
                notifier.send_model_state_alert(
                    model_info={
                        "active_model_id": target_id,
                        "feature_schema": target_schema,
                        "num_features": len(feat_names),
                        "model_sha256": ver["actual_sha256"],
                        "model_path": target_path_str,
                        "previous_model_sha256": cur_sha,
                        "previous_model_path": cur_path_str,
                        "load_status": f"ROLLED_BACK_TO_{target.upper()}",
                        "fallback_status": "ROLLED_BACK",
                    },
                    event_type="MODEL_ROLLBACK_EXECUTED",
                    reason=f"Target={target}",
                )
            except Exception as ne:
                logger.warning(f"Telegram rollback notification failed (ignored): {ne}")
        return updated_manifest


if __name__ == "__main__":
    moe = train_and_save_moe_orchestrator()
    lake = MarketDataLake()
    tqqq = lake.load_candles(config.TRADE_SYMBOLS[0], "15m")
    sample_eval = moe.evaluate_dual_filter_signal(tqqq)
    logger.info("Lumos V3 Hybrid MoE Sample Decision:")
    logger.info(json.dumps(sample_eval, indent=2, ensure_ascii=False))


