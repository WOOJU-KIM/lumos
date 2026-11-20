# 쉐도우 기간 제한 해제

2026-10-18 KST. 사용자 지시: 실전 운영과 별도로 쉐도우 5개를 명시적인 중지 지시까지 계속 평가한다.

## 현재 실행 설정

`research/shadow_five_full15m_20261018/runtime_control.json`이 실제 쉐도우 실행 기간을 제어한다.

- `enabled: true`
- `start_date_ny: 2026-10-19`
- `end_date_ny: null` — 고정 종료일 없음.
- `operation: UNTIL_EXPLICIT_USER_STOP`
- 적용 범위는 쉐도우뿐이며 실제 주문 기능이 없다.

기존 `policy.json`의 10/19~23은 **학습 아티팩트 provenance 및 기존 오프라인 재현 기간**으로 보존했다. 그 파일을 바꾸면 모델/policy SHA 결합이 깨지므로 덮어쓰거나 manifest SHA를 바꾸지 않았다. 학습 cutoff, 모델, 임계값과 가상 체결 정책도 그대로다. 기간 연장을 재학습이나 실전 READY 승인으로 해석하지 않는다.

## 구현과 중지 방법

- `runtime_control.py`: 실행 설정 검증, 날짜 허용 및 중지 판정. 파일 누락·손상 시 무기한 허용으로 대체하지 않는다.
- `runtime.py`: Ledger/Engine에 선택적인 날짜 판정 함수를 전달한다. 기본값은 기존 고정 평가 주간이며, 실제 수신기만 별도 실행 기간을 전달한다. primary와 forced 모두 같은 실행 기간을 적용한다.
- `live_feed.py`: 기존 고정 주간 종료 조건을 실행 설정으로 교체. 시작 시 검사하며, 기존 5초 health 주기에 설정을 다시 읽는다. 실행 시작·변경과 health에 실행 설정을 기록한다.
- `start_shadow.py`: 실행 설정이 중지 또는 만료 상태라면 자동 시작도 하지 않는다. 실제 Runner 시작 경로를 추가하지 않았다.

사용자가 중지를 지시하면 **실전 설정을 건드리지 않고** 실행 설정의 `enabled`를 false로 원자적으로 저장한다. 수신기는 health 검사 때 중지하고, 다음 실전 Runner 시작의 쉐도우 자동 시작도 이 값을 존중한다. 열린 가상 포지션이 있다면 RUN_STOP의 미해결 수량을 보존해 평가에서 구분해야 한다. 중지 지시를 임의의 재학습·실전 청산 요청으로 해석하지 않는다.

이번 반영에서는 수신 이벤트 0개인 기존 쉐도우 worker만 PID와 생성 시각으로 식별해 graceful stop 후 다시 시작했다. 기존 stop request를 분석 디렉터리에 보존했다. 새 worker는 LISTENING이며 health의 `runtime_control.end_date_ny`가 null임을 확인했다. 실제 Runner를 가동하거나 종료하지 않았다.

호스트와 실시간 데이터 제공 프로세스가 실행돼 있어야 입력을 받을 수 있다. PC가 꺼진 시간의 틱을 복원하거나 가짜 데이터로 채우지 않는다. 기존 자동 시작은 실전 Runner 가동 시 그대로 동작한다. 데이터 누락·지연 같은 기존 무결성 실패는 계속 격리 처리하며, 무기한 운영을 이유로 오류를 무시하거나 자동으로 정상 판정하지 않는다. 이 변경이 매주 모델을 자동 재학습시키는 것은 아니다.

## 검증

- 신규 일정 테스트 11/11 통과: 종료 주 이후 허용, 시작 경계, 양쪽 cohort 전달, 기존 오프라인 기간 보존, 장중 시간·주말·보유 중 진입 차단 보존, 중지 후 진입/자동 시작 차단, 유한 기간 및 손상/누락 설정 거부.
- 10/26으로 시간을 고정한 격리 수신기에서 모델 5개를 로딩하고 별도 이름의 ring으로 정상 대기 후 제한 시간 종료. 고정 10/23 만료가 제거됐음을 확인. disabled 상태에서는 ring조차 열지 않음을 확인.
- 필수 `python tests/test_system_integrity.py` 9/9 통과.
- 프로덕션 core/config, 운영 data, models/backups, 기존 쉐도우 policy/모델 아티팩트 등 보호 파일 299개 SHA256 동일.
- 실제 주문, 브로커 API, Telegram 전송, 실제 Runner 가동 0건. 현재 실제 시장 입력은 아직 0개이며 장중 관측을 대체하지 않는다.

근거: `research/shadow_continuous_20261018/`의 수정 전 소스, `protected_before.json`, `unit_tests.json`, `system_integrity.txt`, `future_worker_qa.json`, `old_worker_stopped.json`, `result.json`.
