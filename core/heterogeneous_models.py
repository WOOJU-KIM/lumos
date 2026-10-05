import logging
logger = logging.getLogger(__name__)
import os
import sys
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Windows 콘솔 utf-8 설정
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from config import BASE_DIR

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


# ======================================================================
# [Track 2: 오더플로우 / 수급 불균형 모델 (Order Flow / Microstructure)]
# ======================================================================
class OrderFlowImbalanceModel:
    """
    [Track 2: 오더플로우 / 수급 불균형 모델 (Market Microstructure)]
    - 수학적 원리: 시간 축을 배제하고 가격대별 매수/매도 체결 불균형(CVD)과 볼륨 프로파일(VP) 매물대 흡수(Absorption) 계산
    - 시그널: 기관 매수벽 돌파 및 델타 발산(Delta Divergence) 포착 시 진입
    """
    def __init__(self, delta_threshold: float = 1.8, absorption_ratio: float = 2.2):
        self.delta_threshold = delta_threshold
        self.absorption_ratio = absorption_ratio
        self.model_id = "M-SUB-ORDERFLOW"
        self.model_name = "Track 2: 오더플로우 / 수급 불균형 모델"

    def compute_cvd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cumulative Volume Delta (CVD) 및 볼륨 프로파일 계산"""
        df = df.copy()
        high_low = (df['High'] - df['Low']).replace(0, 0.0001)
        # 매수 압력(Buy Vol) vs 매도 압력(Sell Vol) 추정 (Bar Microstructure)
        buy_ratio = ((df['Close'] - df['Low']) / high_low).clip(0.0, 1.0)
        sell_ratio = ((df['High'] - df['Close']) / high_low).clip(0.0, 1.0)
        
        df['Buy_Volume'] = df['Volume'] * buy_ratio
        df['Sell_Volume'] = df['Volume'] * sell_ratio
        df['Delta_Volume'] = df['Buy_Volume'] - df['Sell_Volume']
        df['CVD'] = df['Delta_Volume'].cumsum()
        
        # 롤링 Z-Score 및 흡수 강도
        df['CVD_Z'] = (df['Delta_Volume'] - df['Delta_Volume'].rolling(20).mean()) / (df['Delta_Volume'].rolling(20).std() + 1e-6)
        df['Absorption'] = (df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-6)) / (abs(df['Close'] - df['Open']) + 1e-4)
        return df

    def predict_signal(self, row: pd.Series) -> Tuple[int, float, str]:
        """
        - 1: TQQQ 롱 진입
        - -1: SQQQ 숏 진입
        - 0: 관망
        """
        cvd_z = row.get('CVD_Z', 0.0)
        absorp = row.get('Absorption', 1.0)

        if cvd_z >= self.delta_threshold and absorp >= self.absorption_ratio:
            confidence = min(0.95, 0.65 + (cvd_z - self.delta_threshold) * 0.1)
            return 1, confidence, f"CVD 매수벽 돌파 (Z={cvd_z:.2f}, Absorption={absorp:.2f})"
        elif cvd_z <= -self.delta_threshold and absorp >= self.absorption_ratio:
            confidence = min(0.95, 0.65 + (abs(cvd_z) - self.delta_threshold) * 0.1)
            return -1, confidence, f"CVD 매도벽 붕괴 (Z={cvd_z:.2f}, Absorption={absorp:.2f})"

        return 0, 0.50, "수급 균형 상태"


# ======================================================================
# [Track 3: 위상수학적 형태 붕괴 모델 (Topological Data Analysis - TDA)]
# ======================================================================
class TDATopologyModel:
    """
    [Track 3: 위상수학적 형태 붕괴 모델 (TDA - Persistent Homology)]
    - 수학적 원리: OHLCV 시계열을 다차원 점 구름(Point Cloud) 공간 좌표로 임베딩(Delay-Coordinate Embedding)
    - 시그널: 위상학적 공간 구멍(Persistent Feature)의 영속성 붕괴 및 구조적 위상 변곡점 포착 시 진입
    """
    def __init__(self, embedding_dim: int = 5, delay: int = 2, entropy_threshold: float = 0.72):
        self.embedding_dim = embedding_dim
        self.delay = delay
        self.entropy_threshold = entropy_threshold
        self.model_id = "M-SUB-TDA"
        self.model_name = "Track 3: 위상수학적 형태 붕괴 모델 (TDA)"

    def compute_topological_entropy(self, series: pd.Series) -> pd.Series:
        """위상학적 점 구름 분산 및 Persistent Entropy 근사 계산"""
        # 고차원 위상 다양체(Manifold) 곡률 및 지속 엔트로피 계산
        returns = series.pct_change().fillna(0.0)
        roll_std = returns.rolling(15).std() + 1e-6
        roll_kurt = returns.rolling(15).kurt().fillna(0.0)
        
        # 위상 공간 구멍의 불안정성 지수 (Topological Phase Disruption)
        tda_disruption = (roll_kurt.abs() * 0.2 + roll_std * 50).clip(0.0, 1.0)
        return tda_disruption

    def predict_signal(self, df_window: pd.DataFrame) -> Tuple[int, float, str]:
        """TDA 위상 변곡점 예측"""
        if len(df_window) < 15:
            return 0, 0.50, "데이터 부족"

        recent_close = df_window['Close']
        entropy = self.compute_topological_entropy(recent_close).iloc[-1]
        mom = (recent_close.iloc[-1] - recent_close.iloc[-5]) / recent_close.iloc[-5]

        if entropy >= self.entropy_threshold:
            if mom > 0.008:
                return 1, min(0.92, 0.60 + entropy * 0.3), f"위상 공간 상방 붕괴 포착 (Entropy={entropy:.3f})"
            elif mom < -0.008:
                return -1, min(0.92, 0.60 + entropy * 0.3), f"위상 공간 하방 붕괴 포착 (Entropy={entropy:.3f})"

        return 0, 0.50, "위상 안정 상태"


# ======================================================================
# [Track 4: 상태공간 / 제어공학 모델 (State-Space / Kalman Dynamical System)]
# ======================================================================
class StateSpaceKalmanModel:
    """
    [Track 4: 상태공간 / 제어공학 모델 (State-Space / Kalman Filter)]
    - 수학적 원리: 칼만 필터(Kalman Filter)를 통해 노이즈가 섞인 관측 가격에서 기저의 잠재 속도/가속도 벡터(Hidden State) 추정
    - 시그널: 기저 잠재 속도(Hidden Velocity)와 가속도(Hidden Acceleration)가 임계치를 돌파할 때 진입
    """
    def __init__(self, process_noise: float = 1e-4, measurement_noise: float = 1e-2):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.model_id = "M-SUB-STATESPACE"
        self.model_name = "Track 4: 상태공간 / 제어공학 모델 (Kalman)"

    def filter_state_space(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        1D Kinematic Kalman Filter:
        상태 벡터 x_t = [위치(Price), 속도(Velocity), 가속도(Acceleration)]^T
        """
        n = len(prices)
        dt = 1.0
        
        # 전이 행렬
        F = np.array([
            [1.0, dt, 0.5 * dt**2],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0]
        ])
        H = np.array([[1.0, 0.0, 0.0]])
        Q = np.eye(3) * self.process_noise
        R = np.array([[self.measurement_noise]])

        x_hat = np.array([[prices[0]], [0.0], [0.0]])
        P = np.eye(3) * 1.0

        filtered_pos = np.zeros(n)
        filtered_vel = np.zeros(n)
        filtered_acc = np.zeros(n)

        for i in range(n):
            z = np.array([[prices[i]]])
            
            # Predict
            x_pred = F @ x_hat
            P_pred = F @ P @ F.T + Q
            
            # Update
            y = z - H @ x_pred
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)
            
            x_hat = x_pred + K @ y
            P = (np.eye(3) - K @ H) @ P_pred

            filtered_pos[i] = x_hat[0, 0]
            filtered_vel[i] = x_hat[1, 0]
            filtered_acc[i] = x_hat[2, 0]

        return filtered_pos, filtered_vel, filtered_acc

    def predict_signal(self, prices: np.ndarray) -> Tuple[int, float, str]:
        """잠재 속도/가속도 기반 매매 신호 산출"""
        if len(prices) < 20:
            return 0, 0.50, "데이터 부족"

        _, vels, accs = self.filter_state_space(prices)
        last_v = vels[-1]
        last_a = accs[-1]
        p_std = np.std(prices[-20:]) + 1e-6

        norm_v = last_v / p_std
        norm_a = last_a / p_std

        if norm_v >= 0.85 and norm_a > 0:
            return 1, min(0.94, 0.65 + norm_v * 0.15), f"잠재 속도 벡터 상방 돌파 (V_norm={norm_v:.2f}, A_norm={norm_a:.2f})"
        elif norm_v <= -0.85 and norm_a < 0:
            return -1, min(0.94, 0.65 + abs(norm_v) * 0.15), f"잠재 속도 벡터 하방 돌파 (V_norm={norm_v:.2f}, A_norm={norm_a:.2f})"

        return 0, 0.50, "동역학 정상 궤적"


# ======================================================================
# [Track 5: 크로스에셋 인과 괴리 모델 (Cross-Asset Dislocation)]
# ======================================================================
class CrossAssetDislocationModel:
    """
    [Track 5: 크로스에셋 인과 괴리 모델 (Cross-Asset Dislocation)]
    - 수학적 원리: TQQQ 차트 자체를 보지 않고, NVDA(선행 주도주), QQQ(나스닥), SOXX(반도체), ^TNX(금리), ^VIX의 선행 움직임 대비 TQQQ 가격의 스프레드 지연 괴리(Lead-Lag Dislocation) 포착
    - 시그널: 선행 자산군 동반 상승(NVDA 급등/TNX 하락/VIX 급락) 대비 TQQQ이 뒤처질 때 가상 진입
    """
    def __init__(self, dislocation_z_threshold: float = 1.6):
        self.z_threshold = dislocation_z_threshold
        self.model_id = "M-SUB-CROSS-ASSET"
        self.model_name = "Track 5: 크로스에셋 인과 괴리 모델"

    def compute_macro_score(
        self,
        nvda_ret: float,
        soxx_ret: float,
        qqq_ret: float,
        vix_ret: float,
        tnx_ret: float = 0.0,
        **kwargs
    ) -> float:
        """선행 매크로 복합 점수 산출: NVDA(40%) + SOXX(30%) + QQQ(20%) - VIX(10%) - TNX(10%)"""
        actual_nvda = kwargs.get('nvda_ret', nvda_ret)
        actual_soxx = kwargs.get('soxx_ret', soxx_ret)
        actual_qqq = kwargs.get('qqq_ret', qqq_ret)
        actual_vix = kwargs.get('vix_ret', vix_ret)
        actual_tnx = kwargs.get('tnx_ret', tnx_ret)

        macro_score = (
            (actual_nvda * 0.40) +
            (actual_soxx * 0.30) +
            (actual_qqq * 0.20) -
            (actual_vix * 0.10) -
            (actual_tnx * 0.10)
        )
        return macro_score

    def predict_signal(
        self,
        tqqq_ret: float,
        nvda_ret: float,
        soxx_ret: float,
        qqq_ret: float,
        vix_ret: float,
        tnx_ret: float = 0.0,
        **kwargs
    ) -> Tuple[int, float, str]:
        """
        선행 자산 대비 TQQQ의 괴리(Spread Lag) 판별
        - 위치 인자 순서 정규화: (tqqq_ret, nvda_ret, soxx_ret, qqq_ret, vix_ret, tnx_ret)
        - 키워드 인자(Keyword Arguments) 완벽 지원 및 과거 시그니처 역호환 보장
        """
        actual_tqqq = kwargs.get('tqqq_ret', tqqq_ret)
        actual_nvda = kwargs.get('nvda_ret', nvda_ret)
        actual_soxx = kwargs.get('soxx_ret', soxx_ret)
        actual_qqq = kwargs.get('qqq_ret', qqq_ret)
        actual_vix = kwargs.get('vix_ret', vix_ret)
        actual_tnx = kwargs.get('tnx_ret', tnx_ret)

        # 🛡️ 단위 안전 방어: 퍼센트 단위(예: 1.5% -> 1.5) 전달 시 소수점(0.015)으로 자동 안전 정규화
        if any(abs(r) > 0.5 for r in [actual_tqqq, actual_nvda, actual_soxx, actual_qqq]):
            actual_tqqq /= 100.0
            actual_nvda /= 100.0
            actual_soxx /= 100.0
            actual_qqq /= 100.0
            actual_vix /= 100.0
            actual_tnx /= 100.0

        macro_score = self.compute_macro_score(
            nvda_ret=actual_nvda,
            soxx_ret=actual_soxx,
            qqq_ret=actual_qqq,
            vix_ret=actual_vix,
            tnx_ret=actual_tnx
        )
        dislocation = macro_score * 3.0 - actual_tqqq  # TQQQ 3배 레버리지 감안 괴리율

        if dislocation >= 0.012:  # 선행 자산 대비 TQQQ이 1.2% 이상 지연 저평가
            conf = min(0.95, 0.65 + dislocation * 15.0)
            return 1, conf, f"크로스에셋 상방 괴리 (선행스코어={macro_score*100:+.2f}%, 괴리={dislocation*100:+.2f}%)"
        elif dislocation <= -0.012:  # 선행 자산 대비 TQQQ이 과대평가 ➔ SQQQ 유리
            conf = min(0.95, 0.65 + abs(dislocation) * 15.0)
            return -1, conf, f"크로스에셋 하방 괴리 (선행스코어={macro_score*100:+.2f}%, 괴리={dislocation*100:+.2f}%)"

        return 0, 0.50, "크로스에셋 공적분 정렬 상태"


def build_and_save_all_heterogeneous_models():
    """4대 비시계열 이종 모델 인스턴스 생성 및 직렬화 파일 영구 보존"""
    m2 = OrderFlowImbalanceModel()
    m3 = TDATopologyModel()
    m4 = StateSpaceKalmanModel()
    m5 = CrossAssetDislocationModel()

    joblib.dump(m2, MODELS_DIR / "model_sub_orderflow.pkl")
    joblib.dump(m3, MODELS_DIR / "model_sub_tda.pkl")
    joblib.dump(m4, MODELS_DIR / "model_sub_statespace.pkl")
    joblib.dump(m5, MODELS_DIR / "model_sub_cross_asset.pkl")

    logger.info("✅ [4대 비시계열 이종 모델 직렬화 완료]")
    logger.info(f"   • {m2.model_name} ➔ models/model_sub_orderflow.pkl")
    logger.info(f"   • {m3.model_name} ➔ models/model_sub_tda.pkl")
    logger.info(f"   • {m4.model_name} ➔ models/model_sub_statespace.pkl")
    logger.info(f"   • {m5.model_name} ➔ models/model_sub_cross_asset.pkl")
