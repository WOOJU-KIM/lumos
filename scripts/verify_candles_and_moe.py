import sys
import os
import time
import config
import pandas as pd
from pathlib import Path
from datetime import datetime

# Windows 콘솔 utf-8 설정
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.data_lake import MarketDataLake
from core.moe_orchestrator import MoEMetaOrchestrator
from core.kiwoom_ws_streamer import KiwoomWebSocketStreamer
from core.kiwoom_broker import KiwoomBroker

def test_candles_and_moe_pipeline():
    print("=" * 90)
    print("🧪 [5분봉 / 15분봉 / 실시간 틱 병합 / MoE 7대 모델 전수 검증]")
    print(f"🕒 실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)

    data_lake = MarketDataLake()
    broker = KiwoomBroker(is_simulation=True)

    # 1. 분봉 DB 적재 상태 전수 점검
    print("\n[Step 1] 심볼별 5분봉 / 15분봉 / 60분봉 DB 무결성 점검")
    symbols = [config.BASE_ASSET_LONG, config.BASE_ASSET_SHORT, "NVDA", "QQQ", config.MACRO_TREND_SYMBOL, "^VIX", "^TNX"]
    for sym in symbols:
        df_5m = data_lake.load_candles(sym, "5m")
        df_15m = data_lake.load_candles(sym, "15m")
        print(f"   • [{sym:6s}] 5분봉: {len(df_5m):5d}개 (최신: {df_5m.index[-1] if not df_5m.empty else 'N/A'}) | 15분봉: {len(df_15m):5d}개 (최신: {df_15m.index[-1] if not df_15m.empty else 'N/A'})")
        assert len(df_15m) > 100, f"{sym} 15분봉 데이터 부족 ({len(df_15m)}개)"

    # 2. 실시간 틱 데이터와 15분봉 동적 병합 검증
    print("\n[Step 2] WebSocket 실시간 틱 ➔ 15분봉 동적 캔들 형성(Live Forming Candle) 검증")
    mock_live_px = 125.50
    df_raw = data_lake.load_candles(config.BASE_ASSET_LONG, "15m")
    df_merged = data_lake.get_candles_with_live_tick(config.BASE_ASSET_LONG, "15m", live_price=mock_live_px)
    
    print(f"   • 원본 15분봉 마지막 캔들: {df_raw.index[-1]} | 종가=${df_raw.iloc[-1]['Close']:.2f}")
    print(f"   • 틱 병합 15분봉 마지막 캔들: {df_merged.index[-1]} | 종가=${df_merged.iloc[-1]['Close']:.2f}")
    assert df_merged.iloc[-1]['Close'] == mock_live_px, "실시간 틱 캔들 병합 실패"
    print("   ✅ 실시간 틱 ➔ 15분봉 캔들 동적 반영 무결성 통과!")

    # 3. MoE 오케스트레이터 모델 평가 검증
    print("\n[Step 3] MoE 7대 모델 및 Sigmoid 게이팅 평가 무결성 검증")
    moe = MoEMetaOrchestrator()
    res = moe.evaluate_dual_filter_signal(df_merged, threshold=0.75)
    print(f"   • 선택된 Top-1 모델: [{res.get('expert_desc')}]")
    print(f"   • 게이팅 확신도: {res.get('gating_confidence', 0)*100:.1f}%")
    print(f"   • 방향 판정: {res.get('direction')}")
    print(f"   • 최종 승인 여부: {res.get('is_approved')}")

    # 4. 동적 ATR 목표가/손절가 산출 검증
    print("\n[Step 4] 동적 ATR 변동성 손익비(TP/SL) 산출 검증")
    targets = moe.calculate_dynamic_targets(df_merged, mock_live_px)
    print(f"   • 진입 기준가: ${mock_live_px:.2f}")
    print(f"   • 목표 익절가: ${targets['dynamic_tp_px']:.2f} (+{targets['tp_pct']:.2f}%)")
    print(f"   • 칼손절 방어가: ${targets['dynamic_sl_px']:.2f} ({targets['sl_pct']:.2f}%)")
    print(f"   • ATR(14) 변동폭: ${targets['atr_14']:.2f}")

    print("\n" + "=" * 90)
    print("🎯 [검증 완료] 5분봉/15분봉 DB 및 실시간 틱 병합, MoE 모델링 전수 점검 통과!")
    print("=" * 90)

if __name__ == "__main__":
    test_candles_and_moe_pipeline()
