# Live Runner와 쉐도우 실행 수명 연결

2026-10-20 사용자 지시: Live Runner 실행 시 쉐도우 자동 실행, Runner 종료 시 쉐도우 함께 종료.

## 적용

- `runtime_control.json`: `enabled=true`, `operation=FOLLOW_LIVE_RUNNER`. 날짜 제한은 기존 start 및 end=null을 유지한다. Runner가 없는 상태에서는 standalone 자동 기동하지 않는다. 이번 작업에서 실전 Runner나 운영 쉐도우를 가동하지 않았다.
- `feed_bridge.autostart`: 실전 background startup에서 별도 helper를 요청한다. 호출자 PID·정확한 Windows 생성 시각·새 세션 토큰을 전달한다. 모델 로드를 기다리지 않으며 simulation/격리 환경에서는 시작하지 않는다. 중복 세션 요청은 무시한다.
- `core/live_runner.py`: 기존 startup/매매/수신 호출은 유지하고 background 실행 부분을 `try/finally`로 감싸 `autostop`을 호출한다. 정상 listener 반환, KeyboardInterrupt, startup/실행 예외에도 해당 세션의 종료를 요청한다. dry-run 검사는 기존 위치에서 먼저 반환한다.
- `start_shadow.py`: 기존 launcher mutex·consumer singleton과 worker PID/생성 시각 검증을 유지한다. owner identity와 취소 상태를 검사해 시작 도중 Runner가 종료되는 경합을 처리한다. 기존 worker가 같은 owner면 재사용하고, 다른 살아 있는 owner의 worker는 건드리지 않는다. 이전 세션이 종료 중이면 helper에서 정상 종료를 기다린 후 새 worker를 시작한다.
- `lifecycle.py`: 실제 owner process handle을 유지해 종료를 감지한다. PID가 재사용돼도 생성 시각이 다르면 owner로 인정하지 않는다. PID로 다른 프로세스를 강제 종료하지 않는다. 정상 종료 요청 파일은 세션 토큰별로 분리해 다음 Runner를 잘못 중지하지 않는다.
- `live_feed.py`: 모델 로드 전후와 기존 5초 health 주기에 owner 종료/세션 stop 요청을 확인한다. Runner 강제 종료(`os._exit`/프로세스 종료)도 worker가 감지해 자기 자신의 로그/파일/핸들을 정상 종료한다. 정상 stop은 전역 enabled를 끄지 않으므로 다음 Runner가 다시 시작할 수 있다.
- Runner 종료로 가상 포지션이 남으면 실제 Bid 체결을 만들어내지 않고 RUN_STOP의 primary/forced 미해결 수량에 기록한다. 다음 run에서 자동 복구한다고 주장하지 않는다.

## 검증

- 최신 core/config/models/tests를 SHA 대조한 격리 복제에서 `python tests/test_system_integrity.py` 9개 통과.
- 신규 lifecycle 12개, 실행 기간 11개, 가격 청산 경로 9개, WebSocket 전용 입력 3개: 합계 **44개 통과**.
- Runner AST는 새 종료 wrapper를 제거하면 수정 직전 전체 AST와 동일하다. 매매·원장·모델 판단 조건은 변경하지 않았다. config 및 모델/매니페스트 해시는 동일하다.
- 실제 owner fixture와 실제 5개 모델 worker를 별도 isolated named ring으로 실행했다. 정상 stop 요청에서는 owner가 살아 있는 동안 worker가 `LIVE_RUNNER_STOP_REQUESTED`로 정상 종료했다. 다음 기동을 위한 enabled=true를 보존했다.
- 두 번째 owner를 `os._exit`로 종료했을 때 worker가 process handle로 감지해 `LIVE_RUNNER_EXITED`로 정상 종료했다. 테스트는 실제 Runner/브로커/네트워크/실주문을 사용하지 않았다.
- 이 두 프로세스 smoke의 worker RUN_STOP 및 상태 저장은 성공했다. 복제 환경 sitecustomize의 종료 진단 쓰기만 후보 workspace 밖 경로라 추가 가드에 차단됐다. 이는 worker lifecycle 실패와 구분하며 운영 경로의 가드를 완화하지 않았다.
- `qa_autostart.py`도 새 lifecycle 계약 테스트를 실행하도록 갱신했다. 과거 AST 위치 가정과 오래된 전체 baseline에 의존하지 않는다.

증거 및 변경 SHA: `research/shadow_runner_lifecycle_20261020/result.json`, 테스트 출력, `normal_worker_result.json`, `abrupt_worker_result.json`, 수정 전 소스 백업.

현재 운영 프로세스는 종료 상태이며 다음 실제 Live Runner 기동부터 적용된다. 무손실 종료나 가상 미청산 포지션 복구, 시스템 전체 실거래 READY 승인을 의미하지 않는다.
