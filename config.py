import os
from pathlib import Path
from dotenv import load_dotenv

# .env 濡쒕뱶
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ?????ㅼ??????듭떖 ?ㅼ젙 ?뚮씪誘명꽣
INITIAL_CAPITAL_KRW = 10_000_000  # 珥덇린 ?댁슜 ?먮낯湲?10,000,000??
GBDT_CONFIDENCE_THRESHOLD = 0.704  # GBDT ?뚰삎 紐⑤뜽 ?⑥씪 ?몃━嫄??꾧퀎媛?(65%)

# ==============================================================================
# ?쒓뎅?ъ옄利앷텒(KIS) OpenAPI ?섍꼍 ?ㅼ젙 (VIRTUAL vs REAL)
# ==============================================================================
KIS_MODE = os.getenv("KIS_MODE", "VIRTUAL").upper()

# 1. 紐⑥쓽?ъ옄 (VIRTUAL)
KIS_VIRTUAL_BASE_URL = os.getenv("KIS_VIRTUAL_BASE_URL", "https://openapivts.koreainvestment.com:29443")
KIS_VIRTUAL_APP_KEY = os.getenv("KIS_VIRTUAL_APP_KEY") or os.getenv("KIS_APP_KEY")
KIS_VIRTUAL_APP_SECRET = os.getenv("KIS_VIRTUAL_APP_SECRET") or os.getenv("KIS_APP_SECRET")
KIS_VIRTUAL_CANO = os.getenv("KIS_VIRTUAL_CANO", "50201878")
KIS_VIRTUAL_ACNT_PRDT_CD = os.getenv("KIS_VIRTUAL_ACNT_PRDT_CD", "01")

# 2. ?ㅼ쟾?ъ옄 (REAL)
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
        raise ValueError(f"?꾩닔 ?섍꼍 蹂??섍? ?꾨씫?섏뿀?듬땲?? {', '.join(missing)}")
    return True

# ==============================================================================
# Lumos V5 ?듭떖 留ㅻℓ/?댁쁺 遺덈? ?뚮씪誘명꽣 (AGENTS.md 湲곗?)
# ==============================================================================

# 1. 吏꾩엯 ?먮떒 ?곗씠??(Entry Logic)
GBDT_CONFIDENCE_THRESHOLD = 0.704  # ?섏씠釉뚮━??MoE 吏꾩엯 理쒖냼 ?뺤떊??(65%)
USE_CROSS_ASSET_VETO = False      # ?щ줈?ㅼ뿉??Veto 諛⑺뙣 ?곸슜 ?щ?
USE_60M_TREND_FILTER = False      # 60遺꾨큺 異붿꽭 ?꾪꽣 ?곸슜 ?щ?
MACRO_TREND_SYMBOL="QQQ"        # GBDT 대추세 (QQQ 60분봉 추세 방패 기준 종목)
QQQ_EMA_PERIOD = 20              # QQQ 異붿꽭 ?꾪꽣 ?댄룊??湲곌컙
RSI_PERIOD = 14                   # RSI 湲곕낯 怨꾩궛 湲곌컙
RSI_OVERBOUGHT_THRESHOLD = 100    # RSI 怨쇰ℓ??吏꾩엯 湲덉? 湲곗? (V3 ?곸슜?쇰줈 議깆뇙 ?댁젣)
RSI_OVERSOLD_THRESHOLD = 0        # RSI 怨쇰ℓ??吏꾩엯 湲덉? 湲곗? (V3 ?곸슜?쇰줈 議깆뇙 ?댁젣)
ROLLING_TRAINING_WEEKS = 520      # 모델 롤링 누적 확장 학습 최대 10년 한도 (Expanding Window)      # 紐⑤뜽 濡ㅻ쭅 ?숈뒿 ?곗씠??湲곌컙 (104二?= 2??

# [V3 ?곹깭怨듦컙 紐⑤뜽 ?곸슜 癒몄떊?щ떇 ?뚮씪誘명꽣]
GBDT_N_ESTIMATORS = 120           # ?몃━ 媛쒖닔 源딄퀬 留롮씠 ?앹꽦 (湲곗〈 80)
GBDT_LEARNING_RATE = 0.02         # ?숈뒿 ?띾룄 ?섑뼢 (湲곗〈 0.03)
GBDT_FEATURE_FRACTION = 0.5       # 怨쇱쟻??諛⑹? (?꾩껜 ?쇱쿂 以?50%留??ъ슜)

# 2. 由ъ뒪??愿?由?泥?궛 (Exit & Risk Management)
SL_MIN_PCT = 0.02                 # 理쒖냼 ?먯젅 ?쇱씤 (2.0%)
SL_MAX_PCT = 0.032                # 理쒕? ?먯젅 ?쇱씤 (3.2%)
SL_ATR_MULTIPLIER = 1.5           # ATR 湲곕컲 ?먯젅 諛곗닔
MAX_TP_PCT = 0.10                 # 紐⑺몴 ?듭젅 ?쇱씤 (10.0%)
TRAILING_TRIGGER_PCT = 0.015      # 트레일링 스탑 발동 조건 (+1.5%)
TRAILING_DROP_PCT = 0.003       # 트레일링 스탑 청산 조건 (최고점 대비 -0.3%)
TIME_STOP_MINUTES = 90           # 강제 타임 스탑 (90분)

# 3. ?댁쁺 ???꾨씪??(Operational Timeline - NYT 湲곗?)
PHASE_MAIN_START = "09:30"        # Main Phase ?쒖옉
PHASE_MAIN_END = "15:30"          # ?좉퇋 吏꾩엯 留덇컧
PHASE_COOLDOWN_END = "15:55"      # 荑⑤떎??醫낅즺 諛?EOD 泥?궛 ?쒖옉
PHASE_EOD_CLEAR = "15:55"         # 100% ?꾧툑???ㅻ쾭?섏엲 泥?궛 ?⑤룆 ?섑뻾 ?쒓컙
MARKET_CLOSE_TIME = "16:00"       # ??留덇컧 ?쒓컙

# 4. 二쇰Ц 誘몄껜寃?諛⑹뼱 濡쒖쭅 (Order & Slippage)
MAX_ALLOCATION_RATIO = 0.98       # 留ㅼ닔鍮꾩쨷 98% 怨좎젙 (2% 踰꾪띁)
BUY_TIMEOUT_SEC = 3               # 留ㅼ닔 誘몄껜寃????꾩븘??
SELL_TIMEOUT_SEC = 3              # 留ㅻ룄 誘몄껜寃????꾩븘??
BUY_SLIPPAGE_ADJUST = 0.03        # 留ㅼ닔 ?멸? 議곗젙 (Ask + 0.03)
SELL_SLIPPAGE_ADJUST = 0.05       # 留ㅻ룄 ?멸? 議곗젙 (Bid - 0.05)
QTY_CALC_BUFFER = 0.05            # ?섎웾 怨꾩궛 ??媛?寃?踰꾪띁 (cur_px + 0.05)

# 5. 諛깊뀒?ㅽ듃 (Backtest)
BACKTEST_FEE_SLIPPAGE = 0.0020    # 諛깊뀒?ㅽ듃 媛?吏쒖닔??諛⑹뼱 理쒖냼 ?섏닔猷?(0.20%)

# 6. 醫낅ぉ ?щ낵 ?ㅼ젙 (Symbols Configuration)
BASE_ASSET_LONG="SOXL"
BASE_ASSET_SHORT="SOXS"
TRADE_SYMBOLS = [BASE_ASSET_LONG, BASE_ASSET_SHORT]
CROSS_ASSET_SYMBOLS = ["NVDA", "QQQ", "SOXX", "VIXY", "IEF"]
ALL_SYMBOLS = list(set(TRADE_SYMBOLS + CROSS_ASSET_SYMBOLS + ["SOXL", "SOXS"]))

# --- POLLING & TIMEOUT SETTINGS ---
WS_FILL_POLL_INTERVAL = 0.05
BALANCE_CACHE_TTL_SEC = 5.0
CANCEL_ORDER_WAIT_TIME = 0.5
MAIN_LOOP_TICK_OPEN = 3
MAIN_LOOP_TICK_CLOSED = 15
MAIN_LOOP_ERROR_WAIT = 1.0


# 7. ?먮룞 ?숈뒿 ?ㅼ젙 (Auto Training Configuration)
TRAINING_LOOKBACK_DAYS = 730
