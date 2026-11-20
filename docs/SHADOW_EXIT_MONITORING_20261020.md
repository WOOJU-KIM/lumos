# 쉐도우 가격 청산 감시 수정 — 2026-10-20

사용자 승인 범위: 쉐도우에 매수뿐 아니라 손절·익절·트레일링·시간/EOD 청산까지 반영한다. 실전 Runner, broker, config, 모델/매니페스트는 변경하지 않았다.

## 원인과 수정

기존 수신기는 bid/ask가 있는 패킷을 QUOTE로만 처리했다. QUOTE는 대기 주문 체결에만 사용되고, TICK이 없으면 POLL 가격도 생기지 않았다. 10/19 run은 QUOTE 43,499건, TICK/POLL 0건으로 TIME/EOD 청산만 발생했다.

- `market_events.py`: 순수 입력 변환 및 가상 시계. 실제 WebSocket Bid를 `MARK(price_basis=WS_BID)`로 감시한다. Bid/Ask 중간값을 거래 틱으로 만들지 않는다. 호가가 없을 때는 명시적인 last-price 필드를 전달한 V2에 한해 `WS_EXPLICIT_LAST`를 사용할 수 있다.
- `feed_bridge.py`: V2 프레임(kind 4)에 명시적인 가격 필드 유무와 원본 packet type을 제한된 크기로 전달한다. 계좌/주문번호/raw payload는 전달하지 않는다. REST는 게시하지 않는다.
- 구 V1 publisher의 유효 bid/ask도 지원한다. V1의 호가 없는 callback price는 출처를 확인할 수 없어 거절한다. 모든 frame은 WebSocket 출처·유한 값·호가 순서를 검사한다.
- `runtime.py`: MARK에서 고점과 초기 SL/TP를 검사하고, config cadence의 POLL에서 트레일링을 검사한다. Bid 기준 peak는 체결가 peak가 아니다. 두 방식의 가격 궤적 차이를 숨기지 않는다.
- `live_feed.py`: 실시간 경로에서 새 진입은 해당 종목의 신선한 감시 가격이 있을 때만 허용한다. 모든 feed가 끊겨도 CLOCK이 동작해 TIME/EOD 매도 의도를 생성한다. 매도 체결은 다음 유효 Bid를 기다리고, 없는 체결을 생성하지 않는다. health에 종목별 가격 감시 상태·열린 primary 포지션·전체 청산 대기 수를 기록한다.
- 가격 기반 매도 의도/체결에 trigger price와 basis를 저장한다. MARK/POLL 입력에도 관측 시각·packet type·protocol version을 저장한다. 실제 브로커 주문/거절/시장 충격을 재현하는 체결 엔진은 아니다.

## 검증

- `tests/test_shadow_exit_monitoring.py`: 9개 통과. 5개 모델×primary/forced 원장, 양방향 자산, SL/TP/trailing/TIME/EOD, 다음 Bid 체결, 무호가 대기, stale 입력·무감시 진입 차단, REST/손상 frame 거절, 출처 보존.
- `tests/test_shadow_websocket_only.py`: 3개 통과. REST 제외, publisher 출처와 V2 호가 보존, 손상 frame 처리.
- `tests/test_shadow_runtime_control.py`: 11개 통과.
- 최신 core/config/models/tests를 SHA 대조한 격리 복제에서 `python tests/test_system_integrity.py`: 9개 통과.
- 격리 named ring에서 실제 V2 publisher→별도 worker→5개 모델 선로딩 검증: 입력 41, QUOTE 40, MARK 41, POLL 2, valid 종료, API/실주문 0건.
- 실제 10/19 저장 입력 22 cycle과 호가 43,499건으로 5개 모델 110판단 재생: SL/TRAILING 청산 발생, 오류/미청산 0. **청산 경로 회귀이며 공식 백테스트·OOS 성과가 아니다.** 이전 실제 기록은 덮어쓰지 않았다.
- 격리 테스트의 최초 임시 파일 경로가 쓰기 가드에 막힌 실행은 임시 폴더를 복제 내부로 지정한 후 재실행했다. 가드를 해제하지 않았다.

증거: `research/shadow_exit_fix_20261020/`의 테스트 출력, replay_result.json, worker_smoke.json, protected_before.json, result.json. 해당 result의 변경 SHA를 기준으로 한다.

## 실제 적용과 남은 확인

기존 PID 33216과 생성 시각·명령행을 확인하고, primary/forced 모두 매수·매도 이벤트가 일치해 미청산이 0임을 확인한 후 정상 종료를 요청했다. 새 수신기를 시작했다. 기존 run은 독립된 기록으로 보존한다.

적용 당시 실전 producer가 종료된 장외 시간이므로 실제 시장의 새 MARK/POLL/청산 발생까지 확인했다고 주장하지 않는다. 새 수신기는 입력 대기 상태이며, 가격이 없으면 진입을 막는다. 다음 실전 producer 정상 시작 시 V2 publisher를 로딩한다. 실행 중인 구 publisher에도 V1 호가 호환이 적용된다.

10/19 기존 run의 손익은 가격 기반 청산 미작동 상태의 기록이다. 수정 후 실제 거래 성과 또는 완전한 실전 복제로 해석하지 않는다.
