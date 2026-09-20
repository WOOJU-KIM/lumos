from config import GBDT_CONFIDENCE_THRESHOLD
import config
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')

class MLFeatureEngine:
    """
    [머신러닝 기반 종합 피처 엔지니어링 및 60% 확신도 선별 엔진]
    1. 모멘텀: RSI(7, 14, 21), MACD, Stochastics(%K, %D), ROC(5, 10, 20), CCI(14, 20)
    2. 변동성: ATR(14), Bollinger Bands(%b, Bandwidth), Keltner Channels
    3. 거래량: OBV, CMF(20), Volume Z-score, VWAP 대비 이격도
    4. 추세/이평: EMA(9, 21, 50, 200), ADX/DMI
    5. 가격 구조: 캔들 몸통/꼬리 비율, 직전 N개 봉 변동률
    6. LightGBM 모델 학습을 통한 Top 10 핵심 보조지표 자동 선별 및 60% 확신도(Confidence) 산출
    """
    def __init__(self, confidence_threshold: float = GBDT_CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.top_10_features: List[str] = []
        self.top_3_features: List[str] = []
        self.feature_names: List[str] = []
        self._try_load_default_model()

    def _try_load_default_model(self):
        """디스크에 저장된 최신 GBDT 3-Class 모델 자동 로드"""
        import joblib
        from pathlib import Path
        for fname in ["model_main_data_refresh.pkl", "model_champion.pkl", "model_moe_orchestrator.pkl"]:
            p = Path(__file__).resolve().parent.parent / "models" / fname
            if p.exists():
                try:
                    obj = joblib.load(p)
                    if hasattr(obj, "model") and obj.model is not None:
                        self.model = obj.model
                        self.feature_names = getattr(obj, "feature_names", [])
                        self.top_10_features = getattr(obj, "top_10_features", [])
                        self.top_3_features = getattr(obj, "top_3_features", [])
                        break
                    elif hasattr(obj, "gbdt_engine") and getattr(obj.gbdt_engine, "model", None) is not None:
                        self.model = obj.gbdt_engine.model
                        self.feature_names = getattr(obj.gbdt_engine, "feature_names", [])
                        self.top_10_features = getattr(obj.gbdt_engine, "top_10_features", [])
                        self.top_3_features = getattr(obj.gbdt_engine, "top_3_features", [])
                        break
                    elif type(obj).__name__ == "LGBMClassifier":
                        self.model = obj
                        self.feature_names = getattr(obj, "feature_name_", [])
                        self.top_10_features = []
                        self.top_3_features = []
                        break
                except Exception:
                    pass

    def extract_features(self, df: pd.DataFrame, live_prices: dict = None) -> pd.DataFrame:
        """15분봉 시세 데이터로부터 종합 피처 풀 자동 생성 (대소문자 정규화 및 데이터 부족 방어)"""
        if df is None or df.empty or len(df) < 5:
            return pd.DataFrame()

        df = df.copy()

        # 대소문자 표준화 (OHLCV)
        rename_map = {
            'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
        }
        for k, v in rename_map.items():
            if k in df.columns and v not in df.columns:
                df[v] = df[k]

        if 'Close' not in df.columns:
            return pd.DataFrame()
        if 'High' not in df.columns:
            df['High'] = df['Close']
        if 'Low' not in df.columns:
            df['Low'] = df['Close']
        if 'Open' not in df.columns:
            df['Open'] = df['Close']
        if 'Volume' not in df.columns:
            df['Volume'] = 1000.0

        # [1. 모멘텀 지표]
        # RSI (7, 14, 21)
        for period in [7, 14, 21]:
            delta = df['Close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(period).mean()
            avg_loss = loss.rolling(period).mean()
            rs = avg_gain / (avg_loss + 1e-9)
            df[f'RSI_{period}'] = 100 - (100 / (1 + rs))

        # MACD (12, 26, 9)
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        # Stochastics (14, 3, 3)
        low14 = df['Low'].rolling(14).min()
        high14 = df['High'].rolling(14).max()
        df['Stoch_K'] = ((df['Close'] - low14) / (high14 - low14 + 1e-9)) * 100.0
        df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

        # ROC (5, 10, 20)
        for p in [5, 10, 20]:
            df[f'ROC_{p}'] = df['Close'].pct_change(p) * 100.0

        # CCI (14, 20)
        for p in [14, 20]:
            tp = (df['High'] + df['Low'] + df['Close']) / 3.0
            tp_sma = tp.rolling(p).mean()
            tp_mad = (tp - tp_sma).abs().rolling(p).mean()
            df[f'CCI_{p}'] = (tp - tp_sma) / (0.015 * tp_mad + 1e-9)

        # [2. 변동성 지표]
        # ATR (14)
        hl = df['High'] - df['Low']
        hc = (df['High'] - df['Close'].shift()).abs()
        lc = (df['Low'] - df['Close'].shift()).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        df['ATR_14'] = tr.rolling(14).mean()

        # Bollinger Bands (20, 2)
        sma20 = df['Close'].rolling(20).mean()
        std20 = df['Close'].rolling(20).std()
        df['BB_Upper'] = sma20 + (std20 * 2)
        df['BB_Lower'] = sma20 - (std20 * 2)
        df['BB_PctB'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-9)
        df['BB_Bandwidth'] = (df['BB_Upper'] - df['BB_Lower']) / (sma20 + 1e-9) * 100.0

        # Keltner Channels (20, 1.5)
        ema20_kc = df['Close'].ewm(span=20, adjust=False).mean()
        df['KC_Upper'] = ema20_kc + (df['ATR_14'] * 1.5)
        df['KC_Lower'] = ema20_kc - (df['ATR_14'] * 1.5)
        df['KC_Width'] = (df['KC_Upper'] - df['KC_Lower']) / (ema20_kc + 1e-9) * 100.0

        # [3. 거래량 지표]
        # OBV & CMF
        obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['OBV'] = (obv - obv.ewm(span=20, adjust=False).mean()) / (df['Volume'].rolling(20).mean() + 1e-9)

        # CMF (Chaikin Money Flow 20)
        mfv = (((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'] + 1e-9)) * df['Volume']
        df['CMF_20'] = mfv.rolling(20).sum() / (df['Volume'].rolling(20).sum() + 1e-9)

        # Volume Z-score
        vol_mean = df['Volume'].rolling(20).mean()
        vol_std = df['Volume'].rolling(20).std()
        df['Volume_Z'] = (df['Volume'] - vol_mean) / (vol_std + 1e-9)

        # VWAP & 이격도
        if isinstance(df.index, pd.DatetimeIndex):
            df['date_str'] = df.index.strftime('%Y-%m-%d')
        elif 'datetime' in df.columns:
            df['date_str'] = df['datetime'].astype(str).str.slice(0, 10)
        elif 'Datetime' in df.columns:
            df['date_str'] = df['Datetime'].astype(str).str.slice(0, 10)
        else:
            df['date_str'] = "2026-01-01"

        typical_price = (df['High'] + df['Low'] + df['Close']) / 3.0
        df['cum_vp'] = typical_price * df['Volume']
        df['vwap'] = df.groupby('date_str')['cum_vp'].cumsum() / (df.groupby('date_str')['Volume'].cumsum() + 1e-9)
        df['VWAP_Diff'] = (df['Close'] - df['vwap']) / (df['vwap'] + 1e-9) * 100.0

        # [4. 추세 및 이동평균]
        for span in [9, 21, 50, 200]:
            df[f'EMA_{span}'] = df['Close'].ewm(span=span, adjust=False).mean()
            df[f'EMA_{span}_Diff'] = (df['Close'] - df[f'EMA_{span}']) / (df[f'EMA_{span}'] + 1e-9) * 100.0

        # ADX / DMI (14)
        up_move = df['High'].diff()
        down_move = -df['Low'].diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        tr_smooth = tr.rolling(14).mean()
        plus_di = (pd.Series(plus_dm, index=df.index).rolling(14).mean() / (tr_smooth + 1e-9)) * 100.0
        minus_di = (pd.Series(minus_dm, index=df.index).rolling(14).mean() / (tr_smooth + 1e-9)) * 100.0
        dx = (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9) * 100.0
        df['ADX_14'] = dx.rolling(14).mean()
        df['DMI_Diff'] = plus_di - minus_di

        # [5. 캔들 구조]
        candle_range = (df['High'] - df['Low']) + 1e-9
        df['Body_Ratio'] = (df['Close'] - df['Open']).abs() / candle_range
        df['Upper_Wick_Ratio'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / candle_range
        df['Lower_Wick_Ratio'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / candle_range


        # [6. 크로스에셋 및 매크로 피처 (V4)]
        if 'datetime_dt' not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df['datetime_dt'] = df.index + pd.Timedelta(minutes=15)
            elif 'datetime' in df.columns:
                df['datetime_dt'] = pd.to_datetime(df['datetime']) + pd.Timedelta(minutes=15)
            elif 'Datetime' in df.columns:
                df['datetime_dt'] = pd.to_datetime(df['Datetime']) + pd.Timedelta(minutes=15)

        try:
            from core.data_lake import MarketDataLake
            lake = MarketDataLake()
            
            # 대추세 60분봉 (config.MACRO_TREND_SYMBOL 기준)
            macro_sym = config.MACRO_TREND_SYMBOL
            soxx_live = live_prices.get(macro_sym, 0.0) if live_prices else 0.0
            if soxx_live > 0:
                soxx_60m = lake.get_candles_with_live_tick(macro_sym, "60m", live_price=soxx_live)
            else:
                soxx_60m = lake.load_candles(macro_sym, "60m")
            if not soxx_60m.empty:
                soxx_60m['ema20'] = soxx_60m['Close'].ewm(span=20, adjust=False).mean()
                soxx_60m['ema60'] = soxx_60m['Close'].ewm(span=60, adjust=False).mean()
                soxx_60m['macro_trend_spread'] = (soxx_60m['ema20'] - soxx_60m['ema60']) / (soxx_60m['ema60'] + 1e-9) * 100.0
                soxx_60m['macro_ema20_slope'] = (soxx_60m['ema20'].diff(5) / soxx_60m['ema20'].shift(5)) * 100.0
                soxx_60m['macro_price_vs_ema20'] = (soxx_60m['Close'] - soxx_60m['ema20']) / (soxx_60m['ema20'] + 1e-9) * 100.0
                soxx_feat = soxx_60m[['datetime', 'macro_trend_spread', 'macro_ema20_slope', 'macro_price_vs_ema20']].copy()
                soxx_feat['datetime_dt'] = pd.to_datetime(soxx_feat['datetime']) + pd.Timedelta(minutes=60)
                
                df = pd.merge_asof(df.sort_values('datetime_dt'), 
                                   soxx_feat.sort_values('datetime_dt').drop(columns=['datetime']), 
                                   on='datetime_dt', direction='backward')

            # V4 크로스에셋 종목 (15분봉)
            for sym in ['NVDA', 'QQQ', 'VIXY', 'IEF']:
                lp = live_prices.get(sym, 0.0) if live_prices else 0.0
                if lp > 0:
                    sym_raw = lake.get_candles_with_live_tick(sym, "15m", live_price=lp)
                else:
                    sym_raw = lake.load_candles(sym, "15m")
                if not sym_raw.empty:
                    sym_raw['datetime_dt'] = pd.to_datetime(sym_raw['datetime']) + pd.Timedelta(minutes=15)
                    lower_sym = sym.lower()
                    for p in [1, 3, 5, 20]:
                        sym_raw[f'{lower_sym}_ret_{p}'] = sym_raw['Close'].pct_change(p) * 100.0
                    sym_raw[f'{lower_sym}_dir_1'] = np.sign(sym_raw['Close'].pct_change(1))
                    sym_raw[f'{lower_sym}_momentum'] = sym_raw['Close'].pct_change(1) - sym_raw['Close'].pct_change(5) / 5.0
                    
                    cols_to_merge = ['datetime_dt'] + [c for c in sym_raw.columns if c.startswith(lower_sym)]
                    df = pd.merge_asof(df.sort_values('datetime_dt'), 
                                       sym_raw[cols_to_merge].sort_values('datetime_dt'), 
                                       on='datetime_dt', direction='backward')
                    
            # 패닉 시그널 및 상대 강도 파생 변수
            df['tqqq_ret_20'] = df['Close'].pct_change(20) * 100.0
            df['tqqq_ret_5'] = df['Close'].pct_change(5) * 100.0
            df['tqqq_vs_qqq_20'] = df['tqqq_ret_20'] - df.get('qqq_ret_20', pd.Series(0.0, index=df.index)).fillna(0)
            df['tqqq_vs_qqq_5'] = df['tqqq_ret_5'] - df.get('qqq_ret_5', pd.Series(0.0, index=df.index)).fillna(0)
            df['panic_signal'] = (
                (df.get('vixy_ret_1', pd.Series(0.0, index=df.index)).fillna(0) > 0).astype(int) + 
                (df.get('ief_ret_1', pd.Series(0.0, index=df.index)).fillna(0) > 0).astype(int)
            )
            
            # 후속 로직(결측치)을 위해 다시 시간순 정렬 유지
            df = df.sort_values('datetime_dt').reset_index(drop=True)
            
        except Exception as e:
            print(f"[MLFeatureEngine] 크로스에셋 병합 실패 (오프라인 모드): {e}")

        if 'datetime' in df.columns:
            df.index = pd.to_datetime(df['datetime'])
        elif 'Datetime' in df.columns:
            df.index = pd.to_datetime(df['Datetime'])
            
        return df

    @staticmethod
    def compute_triple_barrier_labels(
        df: pd.DataFrame,
        take_profit: float = 0.030,
        stop_loss: float = 0.020,
        horizon: int = 6
    ) -> pd.Series:
        """
        [경로 의존적 Triple Barrier 3-Class 정답지 산출]
        - Class 1 (TQQQ 롱): 90분(6개 봉) 내 Low가 -2.0%에 닿기 전에 High가 +3.0%를 먼저 터치
        - Class -1 (SQQQ 숏): 90분(6개 봉) 내 High가 +2.0%에 닿기 전에 Low가 -3.0%를 먼저 터치 (TQQQ 하락)
        - Class 0 (관망): 6개 봉 내 양방향 타겟 미도달 (타임스탑 청산 또는 횡보)
        """
        n = len(df)
        labels = np.zeros(n, dtype=int)
        highs = df['High'].values if 'High' in df else df['high'].values
        lows = df['Low'].values if 'Low' in df else df['low'].values
        closes = df['Close'].values if 'Close' in df else df['close'].values

        for i in range(n):
            p0 = closes[i]
            if p0 <= 0:
                continue
            end_idx = min(i + horizon + 1, n)
            if i + 1 >= end_idx:
                continue

            assigned = False
            for j in range(i + 1, end_idx):
                h_ret = (highs[j] - p0) / p0
                l_ret = (lows[j] - p0) / p0

                # 롱 배리어 (+3.0% TP vs -2.0% SL)
                hit_long_tp = (h_ret >= take_profit)
                hit_long_sl = (l_ret <= -stop_loss)

                # 숏 배리어 (-3.0% TP for short vs +2.0% SL for short)
                hit_short_tp = (l_ret <= -take_profit)
                hit_short_sl = (h_ret >= stop_loss)

                if hit_long_tp and not hit_long_sl:
                    labels[i] = 1   # TQQQ 롱 승리
                    assigned = True
                    break
                elif hit_short_tp and not hit_short_sl:
                    labels[i] = -1  # SQQQ 숏 승리
                    assigned = True
                    break
                elif (hit_long_sl and hit_short_sl) or (hit_long_tp and hit_long_sl) or (hit_short_tp and hit_short_sl):
                    labels[i] = 0   # 동시 급변동 노이즈
                    assigned = True
                    break

            if not assigned:
                labels[i] = 0

        return pd.Series(labels, index=df.index, name="Target")

    def train_and_select_top_features(self, df_15m: pd.DataFrame, live_experience_df: Optional[pd.DataFrame] = None) -> Tuple[Any, List[str], List[str]]:
        """3-Class Triple Barrier 기반 LightGBM 다중 분류 모델 학습 및 상위 10대 피처 선별 (실시간 실전 로그 자동 병합 및 가중치 반영)"""
        feat_df = self.extract_features(df_15m)

        import pandas as pd
        feature_cols = [c for c in feat_df.columns if c not in ['datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Target', 'cum_vp', 'vwap', 'EMA_9', 'EMA_21', 'EMA_50', 'EMA_200'] and not c.startswith('week_') and pd.api.types.is_numeric_dtype(feat_df[c])]

        feat_df['Target'] = self.compute_triple_barrier_labels(
            feat_df,
            take_profit=0.030,
            stop_loss=0.020,
            horizon=6
        )

        clean_data = feat_df[feature_cols + ['Target']].dropna()
        if len(clean_data) < 100:
            self.top_10_features = feature_cols[:10]
            self.top_3_features = ["RSI_14", "VWAP_Diff", "Volume_Z"]
            return None, self.top_10_features, self.top_3_features

        X = clean_data[feature_cols].copy()
        # 클래스 레이블 매핑: {-1: 0, 0: 1, 1: 2} (0: Short, 1: Neutral, 2: Long)
        label_map = {-1: 0, 0: 1, 1: 2}
        y = clean_data['Target'].map(label_map).fillna(1).astype(int)
        sample_weights = pd.Series(1.0, index=X.index)

        # [실시간 경험 로그(Live Experience) 자동 결합 및 가중치 반영]
        target_live_df = live_experience_df
        if target_live_df is None:
            try:
                from config import DATA_DIR
                live_csv = DATA_DIR / "live_trades.csv"
                if live_csv.exists():
                    target_live_df = pd.read_csv(live_csv)
            except Exception:
                target_live_df = None

        if target_live_df is not None and not target_live_df.empty and "features_json" in target_live_df.columns:
            # 2년(504 거래일) 롤링 윈도우 동기화: 과거 캔들 시작일 이후의 거래만 학습에 반영 (과거 꼬리 자동 절삭)
            if isinstance(clean_data.index, pd.DatetimeIndex) and len(clean_data) > 0 and "entry_time" in target_live_df.columns:
                cutoff_str = clean_data.index.min().strftime('%Y-%m-%d')
                target_live_df = target_live_df[target_live_df["entry_time"].astype(str).str.slice(0, 10) >= cutoff_str]

            import json
            live_rows = []
            live_targets = []
            for _, row in target_live_df.iterrows():
                f_json = row.get("features_json")
                if not f_json or str(f_json).strip() in ("", "{}", "nan"):
                    continue
                try:
                    f_dict = json.loads(f_json) if isinstance(f_json, str) else f_json
                except Exception:
                    continue

                sym = str(row.get("symbol", "TQQQ")).upper()
                reason = str(row.get("exit_reason", "")).upper()
                pnl = float(row.get("pnl_pct", 0.0))

                # 라벨 결정: 2(Long TP), 0(Short TP), 1(중립/타임스탑)
                if sym == "TQQQ":
                    tgt = 2 if ("TP" in reason or pnl >= 2.0) else (0 if ("SL" in reason or pnl <= -1.5) else 1)
                else:
                    tgt = 0 if ("TP" in reason or pnl >= 2.0) else (2 if ("SL" in reason or pnl <= -1.5) else 1)

                row_vals = {c: f_dict.get(c, np.nan) for c in feature_cols}
                live_rows.append(row_vals)
                live_targets.append(tgt)

            if live_rows:
                X_live = pd.DataFrame(live_rows).fillna(X.median())
                y_live = pd.Series(live_targets, dtype=int)
                weights_live = pd.Series(2.5, index=X_live.index)  # 실전 체결 데이터 2.5배 가중치

                X = pd.concat([X, X_live], ignore_index=True)
                y = pd.concat([y, y_live], ignore_index=True)
                sample_weights = pd.concat([sample_weights, weights_live], ignore_index=True)
                print(f"[MLFeatureEngine] 🧬 과거 캔들 {len(clean_data):,}개 + 실시간 실전 거래 {len(live_rows):,}개 (2.5x 가중치) 통합 학습 적용 완료")

        try:
            model = LGBMClassifier(
                objective='multiclass',
                num_class=3,
                class_weight='balanced',
                n_estimators=85,
                max_depth=4,
                learning_rate=0.03,
                random_state=42,
                verbosity=-1
            )
            model.fit(X, y, sample_weight=sample_weights)
            importances = model.feature_importances_
        except Exception:
            model = RandomForestClassifier(
                n_estimators=85,
                max_depth=5,
                class_weight='balanced',
                random_state=42
            )
            model.fit(X, y, sample_weight=sample_weights)
            importances = model.feature_importances_

        self.model = model
        self.feature_names = feature_cols

        # 중요도 순 정렬
        sorted_feats = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
        self.top_10_features = [f[0] for f in sorted_feats[:10]]
        self.top_3_features = [f[0] for f in sorted_feats[:3]]

        return model, self.top_10_features, self.top_3_features

    def add_confidence_columns(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """전체 데이터프레임에 대해 3-Class 확신도 및 신호 컬럼 생성"""
        feat_df = feat_df.copy()
        if self.model is None or not self.feature_names:
            feat_df['Confidence'] = 0.50
            feat_df['Signal'] = 0
            feat_df['Direction'] = "NONE"
            feat_df['Prob_Long'] = 0.33
            feat_df['Prob_Short'] = 0.33
            feat_df['Prob_Neutral'] = 0.34
            return feat_df

        for col in self.feature_names:
            if col not in feat_df.columns:
                feat_df[col] = 0.0

        clean_X = feat_df[self.feature_names].fillna(0.0)
        try:
            probs = self.model.predict_proba(clean_X)  # shape: (N, 3) -> [P(Short), P(Neutral), P(Long)]
            prob_short = probs[:, 0]
            prob_neutral = probs[:, 1]
            prob_long = probs[:, 2]

            signals = np.zeros(len(feat_df), dtype=int)
            confidences = np.full(len(feat_df), 0.50, dtype=float)
            directions = ["NONE"] * len(feat_df)

            for idx in range(len(feat_df)):
                ps, pn, pl = prob_short[idx], prob_neutral[idx], prob_long[idx]
                if pl > pn and pl > ps:
                    signals[idx] = 1
                    # 3-Class Calibration: 33.3% 기준선 -> 50%~95% 스케일링
                    calib_conf = min(0.95, max(0.50, 0.50 + (pl - 0.333) * 1.15))
                    confidences[idx] = calib_conf
                    directions[idx] = "LONG_TQQQ"
                elif ps > pn and ps > pl:
                    signals[idx] = -1
                    calib_conf = min(0.95, max(0.50, 0.50 + (ps - 0.333) * 1.15))
                    confidences[idx] = calib_conf
                    directions[idx] = "SHORT_SQQQ"
                else:
                    signals[idx] = 0
                    confidences[idx] = pn
                    directions[idx] = "NONE"

            feat_df['Confidence'] = confidences
            feat_df['Signal'] = signals
            feat_df['Direction'] = directions
            feat_df['Prob_Long'] = prob_long
            feat_df['Prob_Short'] = prob_short
            feat_df['Prob_Neutral'] = prob_neutral
        except Exception:
            feat_df['Confidence'] = 0.50
            feat_df['Signal'] = 0
            feat_df['Direction'] = "NONE"
            feat_df['Prob_Long'] = 0.33
            feat_df['Prob_Short'] = 0.33
            feat_df['Prob_Neutral'] = 0.34

        return feat_df

    def predict_signal(self, df_candle_15m: pd.DataFrame) -> Tuple[int, float, str]:
        """
        MoE 오케스트레이터 표준 인터페이스:
        - 반환값: (signal_code: 1 | -1 | 0, confidence: float 0.0~1.0, reason: str)
        """
        if df_candle_15m is None or df_candle_15m.empty or len(df_candle_15m) < 15:
            return 0, 0.50, "데이터 부족"

        df_feat = self.extract_features(df_candle_15m)
        if df_feat.empty:
            return 0, 0.50, "피처 추출 불가"

        df_feat = self.add_confidence_columns(df_feat)
        last_row = df_feat.iloc[-1]

        sig = int(last_row.get("Signal", 0))
        conf = float(last_row.get("Confidence", 0.50))
        p_long = float(last_row.get("Prob_Long", 0.33))
        p_short = float(last_row.get("Prob_Short", 0.33))

        if sig == 1:
            return 1, conf, f"GBDT 3-Class TQQQ 롱 파형 포착 (P_Long={p_long*100:.1f}%, Conf={conf*100:.1f}%)"
        elif sig == -1:
            return -1, conf, f"GBDT 3-Class SQQQ 숏 파형 포착 (P_Short={p_short*100:.1f}%, Conf={conf*100:.1f}%)"

        return 0, conf, "GBDT 관망/중립 상태"

    def predict_signal_full(self, df_candle_15m: pd.DataFrame) -> Tuple[int, float, str, Dict[str, float]]:
        sig, conf, reason = self.predict_signal(df_candle_15m)
        df_feat = self.extract_features(df_candle_15m)
        if df_feat.empty:
            return sig, conf, reason, {"LONG": 0.33, "SHORT": 0.33, "NONE": 0.34}
        df_feat = self.add_confidence_columns(df_feat)
        last_row = df_feat.iloc[-1]
        probs = {
            "LONG": float(last_row.get("Prob_Long", 0.33)),
            "SHORT": float(last_row.get("Prob_Short", 0.33)),
            "NONE": float(last_row.get("Prob_Neutral", 0.34))
        }
        return sig, conf, reason, probs
