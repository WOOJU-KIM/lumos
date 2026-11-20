# 진입 차단 사유 텔레그램 표시 수정

2026-10-20 사용자 지시. 22:58에 동일 09:45 봉의 재진입 차단으로 수량이 0이 됐으나, 공통 브리핑이 모든 0주 결과를 `INSUFFICIENT_FUNDS`라고 표시하던 문제를 수정했다.

`RelayLiveRunner._entry_quantity()`의 각 기존 차단 분기에서 표시용 사유를 남긴다. 실제 검사 순서와 진입 조건은 유지한다. `KiwoomLiveRunner`는 그 사유를 브리핑에 전달하며, 사유를 제공하지 않는 기존 단일 포지션 수량 계산은 기존 동작을 유지한다. `TelegramNotifier`는 알려진 코드를 한글로 설명하고 원래 코드를 함께 표시한다.

| 상황 | 코드 / 한글 안내 |
|---|---|
| 같은 봉이 이미 처리됨 | WAIT_NEXT_COMPLETED_15M / 동일 15분봉 재진입 차단. 다음 새 15분봉 대기 |
| 양 종목 최신 호가 부족 | FRESH_WS_QUOTES_REQUIRED / 양 종목의 신선한 WebSocket 호가 대기 |
| 최신 완료 봉 판단 없음 | COMPLETED_15M_DECISION_UNAVAILABLE / 최신 완료 15분봉 판단이 없어 대기 |
| 판단 봉 만료 | STALE_15M_DECISION / 이전 15분봉 판단이 만료되어 새 판단 대기 |
| 첫 정규장 봉 미완료 | FIRST_15M_CANDLE_NOT_CLOSED / 개장 후 첫 15분봉 완료 대기 |
| 진입 안전 검사 | RELAY_ENTRY_SAFETY_GUARD / 거래 시간·계좌·실행 상태 안전 검사 미통과 |
| 포지션·주문 존재 | RELAY_CYCLE_ACTIVE / 기존 포지션 또는 미해결 주문 정리 대기 |
| 사이클 예산·주문 예약 존재 | RELAY_CYCLE_RESERVED_OR_PENDING / 진행 중인 사이클 또는 예약 주문 정리 대기 |
| 배정 한도와 현금 버퍼 적용 후 0주 | INSUFFICIENT_FUNDS / 배정 한도·현금 버퍼 적용 후 매수 가능한 수량이 0주 |
| 매수 호출 결과 미확인 | BUY_EXECUTION_NOT_CONFIRMED / 매수 실행 또는 체결 확인이 완료되지 않음 |

로그에도 `Entry quantity blocked: <code>`를 남긴다. 다음 수량 계산 시작 시 이전 표시 사유를 초기화하므로 과거 차단 이유를 재사용하지 않는다.

격리 복제에서 신규 6개 검증은 실제 15분 평가→수량 차단→텔레그램 문구 전달을 확인한다. 동일 봉과 호가 부족을 잔액 부족으로 표시하지 않고, 진짜 배정 가능 수량 0주는 별도 유지하며, 정상 진입 수량·기존 사이클 보호도 유지한다. 핵심 무결성 9개 및 관련 회귀 263개, 총 272개 통과했다. 네트워크·외부 쓰기 차단 시도 0건. config·모델·브로커·원장·READY 변경 없음.

실행 매니페스트의 소스 SHA를 업데이트했다. 현재 실전 Runner와 쉐도우를 재시작하지 않았으므로 실행 중 메모리에는 기존 알림 코드가 남아 있다. 다음 정상 Runner 재시작 시 적용된다. 실주문·실계좌 API·테스트 텔레그램 발송은 수행하지 않았다.

증거: `research/telegram_entry_reason_20261020/result.json`, `targeted.txt`, `integrity.txt`, `regression.txt`, `task_changes.diff`.
