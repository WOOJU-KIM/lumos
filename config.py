import os
from pathlib import Path
from dotenv import load_dotenv

# .env 로드
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 퀀트 헤지펀드 핵심 설정 파라미터
INITIAL_CAPITAL_KRW = 10_000_000  # 초기 운용 자본금 10,000,000원
GBDT_CONFIDENCE_THRESHOLD = 0.60  # GBDT 파형 모델 단일 트리거 임계값 (60%)

# ==============================================================================
# 한국투자증권(KIS) OpenAPI 환경 설정 (VIRTUAL vs REAL)
# ==============================================================================
KIS_MODE = os.getenv("KIS_MODE", "VIRTUAL").upper()

# 1. 모의투자 (VIRTUAL)
KIS_VIRTUAL_BASE_URL = os.getenv("KIS_VIRTUAL_BASE_URL", "https://openapivts.koreainvestment.com:29443")
KIS_VIRTUAL_APP_KEY = os.getenv("KIS_VIRTUAL_APP_KEY") or os.getenv("KIS_APP_KEY")
KIS_VIRTUAL_APP_SECRET = os.getenv("KIS_VIRTUAL_APP_SECRET") or os.getenv("KIS_APP_SECRET")
KIS_VIRTUAL_CANO = os.getenv("KIS_VIRTUAL_CANO", "50201878")
KIS_VIRTUAL_ACNT_PRDT_CD = os.getenv("KIS_VIRTUAL_ACNT_PRDT_CD", "01")

# 2. 실전투자 (REAL)
KIS_REAL_BASE_URL = os.getenv("KIS_REAL_BASE_URL", "https://openapi.koreainvestment.com:9443")
KIS_REAL_APP_KEY = os.getenv("KIS_REAL_APP_KEY")
KIS_REAL_APP_SECRET = os.getenv("KIS_REAL_APP_SECRET")
KIS_REAL_CANO = os.getenv("KIS_REAL_CANO", "10031896")
KIS_REAL_ACNT_PRDT_CD = os.getenv("KIS_REAL_ACNT_PRDT_CD", "01")

PORTFOLIO_STATE_FILE = DATA_DIR / "portfolio_state.json"
EVOLUTION_LOG_FILE = DATA_DIR / "evolution_log.json"
SHADOW_RND_CACHE_FILE = DATA_DIR / "shadow_rnd_cache.json"
TRADE_LOGS_CSV = DATA_DIR / "trade_logs.csv"
TRADE_LOGS_SUMMARY_JSON = DATA_DIR / "trade_logs_summary.json"

def validate_env():
    missing = []
    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN")
    if not TELEGRAM_CHAT_ID:
        missing.append("TELEGRAM_CHAT_ID")
    if not GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")
    
    if missing:
        raise ValueError(f"필수 환경 변수가 누락되었습니다: {', '.join(missing)}")
    return True

# ==============================================================================
# Lumos V5 핵심 매매/운영 불변 파라미터 (AGENTS.md 기준)
# ==============================================================================

# 1. 진입 판단 데이터 (Entry Logic)
GBDT_CONFIDENCE_THRESHOLD = 0.60  # 하이브리드 MoE 진입 최소 확신도 (60%)
USE_CROSS_ASSET_VETO = False      # 크로스에셋 Veto 방패 사용 여부
USE_60M_TREND_FILTER = False      # 60분봉 추세 필터 사용 여부
MACRO_TREND_SYMBOL = "QQQ"        # GBDT 대추세 및 60분봉 추세 방패 기준 종목
QQQ_EMA_PERIOD = 20              # QQQ 추세 필터 이평선 기간
RSI_PERIOD = 14                   # RSI 기본 계산 기간
RSI_OVERBOUGHT_THRESHOLD = 100    # RSI 과매수 진입 금지 기준 (V3 적용으로 족쇄 해제)
RSI_OVERSOLD_THRESHOLD = 0        # RSI 과매도 진입 금지 기준 (V3 적용으로 족쇄 해제)
ROLLING_TRAINING_WEEKS = 104      # 모델 롤링 학습 데이터 기간 (104주 = 2년)

# [V3 상태공간 모델 적용 머신러닝 파라미터]
GBDT_N_ESTIMATORS = 120           # 트리 개수 깊고 많이 생성 (기존 80)
GBDT_LEARNING_RATE = 0.02         # 학습 속도 하향 (기존 0.03)
GBDT_FEATURE_FRACTION = 0.5       # 과적합 방지 (전체 피처 중 50%만 사용)

# 2. 리스크 관리/청산 (Exit & Risk Management)
SL_MIN_PCT = 0.02                 # 최소 손절 라인 (2.0%)
SL_MAX_PCT = 0.032                # 최대 손절 라인 (3.2%)
SL_ATR_MULTIPLIER = 1.5           # ATR 기반 손절 배수
MAX_TP_PCT = 0.10                 # 목표 익절 라인 (10.0%)
TRAILING_TRIGGER_PCT = 0.015      # 트레일링 스탑 발동 조건 (+1.5%)
TRAILING_DROP_PCT = 0.003         # 트레일링 스탑 청산 조건 (최고점 대비 -0.3%)
TIME_STOP_MINUTES = 120           # 강제 타임 스탑 (120분)

# 3. 운영 타임라인 (Operational Timeline - NYT 기준)
PHASE_MAIN_START = "09:30"        # Main Phase 시작
PHASE_MAIN_END = "15:30"          # 신규 진입 마감
PHASE_COOLDOWN_END = "15:50"      # 쿨다운 종료 및 EOD 청산 시작
PHASE_EOD_CLEAR = "15:50"         # 100% 현금화 오버나잇 청산 단독 수행 시간
MARKET_CLOSE_TIME = "16:00"       # 장 마감 시간

# 4. 주문 미체결 방어 로직 (Order & Slippage)
MAX_ALLOCATION_RATIO = 0.98       # 매수비중 98% 고정 (2% 버퍼)
BUY_TIMEOUT_SEC = 3               # 매수 미체결 타임아웃
SELL_TIMEOUT_SEC = 3              # 매도 미체결 타임아웃
BUY_SLIPPAGE_ADJUST = 0.03        # 매수 호가 조정 (Ask + 0.03)
SELL_SLIPPAGE_ADJUST = 0.05       # 매도 호가 조정 (Bid - 0.05)
QTY_CALC_BUFFER = 0.05            # 수량 계산 시 가격 버퍼 (cur_px + 0.05)

# 5. 백테스트 (Backtest)
BACKTEST_FEE_SLIPPAGE = 0.0020    # 백테스트 가짜수익 방어 최소 수수료 (0.20%)

# 6. 종목 심볼 설정 (Symbols Configuration)
TRADE_SYMBOLS = ["TQQQ", "SQQQ"]
CROSS_ASSET_SYMBOLS = ["NVDA", "QQQ", "SOXX", "VIXY", "IEF"]
ALL_SYMBOLS = TRADE_SYMBOLS + CROSS_ASSET_SYMBOLS
