# 동작 보존 중심의 코드 정리

2026-10-19 KST. 사용자 범위: 과도한 클래스·패턴 도입 없이 필요한 역할 분리와 가독성 정리.

## 변경

- `core/live_runner.py`: 미확인 BUY의 읽기·저장·해제를 명시적인 `_load_pending_buy_order`, `_store_pending_buy_order`, `_clear_pending_buy_order` 호출로 표현했다. 기존 속성 인터페이스는 호환용으로 남겼다. 읽기·쓰기의 순서, 메모리 fallback, 원자적 파일 저장, 실패 latch, journal 해제 조건은 보존했다.
- 같은 파일의 매수·매도 취소 후 원장 검사 두 중첩 함수를 `_scan_closed_order_rows`로 묶었다. 동일 4개 호출부에서 날짜·방향·수량·취소 주문 인자를 명시한다. 오류와 잔여 수량의 거절 조건, 반환 tuple은 동일하다.
- `core/order_ledger.py`: 세미콜론과 중첩 삼항식을 풀어 주문 종결 상태 판정을 읽기 쉽게 정리했다. 판정 규칙은 바꾸지 않았다.
- 쉐도우 `runtime.py`: VIRTUAL_BUY, 취소 및 청산 의도 이벤트에 누락된 model_id/cohort/decision_id/model_version을 보완했다. 기존 필드 값과 가상 체결·청산·비용 계산은 유지했다.
- 신규 클래스·상속·범용 프레임워크를 도입하지 않았다. 긴 매매 함수 전체 재작성, 기존 READY 검사 결함 수정, 설정·임계값·학습 변경은 범위에 포함하지 않았다.

## 검증

- 관련 서로 다른 테스트 총 **151개 통과**. 이 집계에 필수 `python tests/test_system_integrity.py` 직접 실행 9개가 포함된다.
- 신규 검증 7개: 명시적 주문 저장·재생성 복구·해제, 기존 속성 호환, 저장 실패 시 메모리 및 latch 보존, 쉐도우 매수→청산과 취소 이벤트 ID 연결.
- 주문 상태 I/O 본문 및 공통 scanner 본문이 추출 전과 동일함을 AST로 대조했다. 내부 호출을 기존 표현으로 대응시킨 뒤 매수·매도·진입 전 원장 확인·체결 후 journal 해제의 AST가 일치했다. 기존 Runner 메서드 34개는 원본 그대로다.
- 원장 판정 입력 1,000개를 수정 전후 함수에 넣어 결과 동일함을 확인했다.
- 쉐도우 2개 cohort × 손절·익절·트레일링·TIME·EOD·stale quote·stale clock·session 취소의 16개 경우에서 현금·보유·pending·완료 기록과 기존 이벤트 필드가 동일했다. 추가 연결 필드는 신규 단위 테스트에서 확인했다.
- 별도 이름의 ring에서 feed QA 19개 조건 통과. 실제 과거 피처의 5개 후보 추론과 10,000개 합성 이벤트 전송 검증이며 실거래 성능 증거는 아니다.
- config, 모델·manifest·backup, 운영 data, 쉐도우 학습 policy·아티팩트·실행 기간 등 보호 파일 **264개 SHA256 동일**.

검증은 `research/code_cleanup_20261019/isolated_env/`에서 비밀 환경변수와 dotenv 로딩을 차단하고 네트워크·외부 쓰기를 금지하여 수행했다. 새 격리 폴더의 DB/CSV 초기 생성 때문에 최초 B-03 종료 시 보호 검사가 실패했으며 초기화 후 통과했다. 검증 도중 격리 DB를 교체한 시점 문제도 있었고, 분석 경로 안의 해당 파일·sidecar를 보존한 뒤 읽기 전용 원본에서 SQLite backup을 생성하고 재실행했다. 프로덕션 파일을 고치거나 assertion을 약화해 통과시키지 않았다. feed QA에서는 실제 Runner 바인딩을 쓰지 않고 테스트용 source와 전용 ring을 사용했다. WinAPI handle 인자 형식은 QA 프로세스에서만 명시했다.

기존 쉐도우 PID는 이미 실행 중이 아니었다. 계속 운영하라는 사용자 지시에 따라 검증된 새 소스로 쉐도우 전용 수신기만 시작해 LISTENING과 종료일 없음 설정을 확인했다. 실제 Runner·주문·브로커 API·Telegram 전송은 실행하지 않았다. 실행 중인 실전 프로세스에 대한 hot reload는 하지 않았으며 실전 코드는 다음 프로세스 시작 때 적용된다. 전체 READY나 수익성을 이번 정리로 재판정하지 않는다.

## 근거

`research/code_cleanup_20261019/`의 수정 전 소스, `protected_before.json`, `behavior_equivalence.json`, `shadow_cleanup_comparison.json`, 테스트별 결과, `system_integrity.txt`, `feed_transport.json`, `result.json`을 참조한다. 격리 폴더는 테스트 결과와 함께 보존했다.
