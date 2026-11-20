# B-03 주문 원장 정합성 개선 및 회귀 결과

기준일: **2026-10-18 KST**. 코드 상태: **CODE_FIXED_OFFLINE_VERIFIED**. 전체 실전 READY: **미승인**.

이 문서는 같은 날짜의 실전/쉐도우 구축 문서 이후 변경분이다. 과거 SHA 스냅샷은 이전 상태의 증거로 보존한다. 현재 소스 해시는 `B03_FIX_STATE_20261018.json`을 참조한다. 모델 활성화 기록의 과거 `B03=NOT_CLEARED`를 이번 작업에서 임의로 덮어쓰지 않았다.

## 1. 재현과 수정 범위

수정 전 B-03 기존 39개 테스트는 모두 통과했다. 추가한 최초 13개 경계 테스트 중 11개가 실패/에러를 재현했다. 단순히 기존 테스트 통과를 근거로 B-03가 해결됐다고 판정하지 않았다.

확인된 문제:

- 중복 체결 행을 반복 합산해 이미 완료된 주문을 미확인으로 판정.
- 체결원장만 먼저 신뢰하여 주문내역의 모순을 놓침.
- 주문 번호가 같아도 날짜·방향·종목·취소 원주문 연결이 다른 증거를 수용할 수 있음.
- 누락된 잔여 수량을 0으로 가정.
- 미확인 매수 상태가 메모리에만 있어 재시작 시 유실되고, 접수 응답 직후 너무 일찍 제거됨.
- WS 체결 카운터만으로 완료를 반환해 잔고 반영 지연/중복 통지와의 대조가 생략됨.
- 매도 가능 수량이 명시적으로 0이어도 보유 수량으로 되돌리는 재주문 fallback.

## 2. 실제 변경

| 파일 | 변경 내용 |
|---|---|
| `core/order_ledger.py` | I/O 없는 공통 증거 reducer. 명시적 수량, identity/date 검증, 체결 ID/행 중복 처리, cumulative와 incremental 구분, 취소 child 검증, 원장 간 모순/UNKNOWN |
| `core/kiwoom_broker.py` | `resolve_order_ledger_status`가 모든 조회 결과를 공통 reducer로 판정. 조회 실패·incomplete를 빈 원장으로 바꾸지 않음. 기존 sell reservation의 요약에도 공통 scanner 사용 |
| `core/pending_order.py` | 계좌·환경 hash에 결합된 미확인 BUY 기록. 임시 파일 + flush/fsync + atomic replace. 손상/다른 환경의 파일은 recovery error로 신규 매수 차단 |
| `core/live_runner.py` | 매수·매도 재주문 scanner 통합, 가능한 주문내역을 항상 추가 대조. 미확인 BUY 저장/복원 및 접수 ID 보존. 확인된 전량 체결과 포지션 파일 저장 이후 해제. 전일 0체결 취소 복구. WS 수량 hint는 fresh balance 재조회로 확인. 잔고 실패 시 SELL 중단, sell_available=0 fallback 제거 |
| `tests/test_b03_hardening.py` | 새 경계·재시작·주문 경로 테스트 26개 |
| `tests/test_system_integrity.py` | 각 테스트의 포지션/주문 journal 경로를 임시 디렉터리로 분리. 기존 인터락 assertion은 유지 |

Runner 기존 메서드 중 수정한 것은 `__init__`, `_verify_pre_buy_ledger_clean`, `_wait_for_fill`, `_execute_buy_with_10s_chase`, `_execute_sell_with_10s_chase`다. 새 recovery 관련 property/helper 외에 모델 평가, WS 실행 메시지 callback, 포지션 관리, feed hook, 필터 메서드 AST는 이전과 동일함을 확인했다.

`config.py`, 활성 모델 및 manifest, Fallback, `m_moe_orchestrator.pkl`, 기존 backups, 운영 DB/CSV/상태 JSON은 변경하지 않았다. 모델 재학습·활성화나 쉐도우 정책 수정도 없었다.

## 3. 미확인 BUY 기록의 수명

기존 포지션 파일에서 파생한 별도 경로를 사용한다. 기본 포지션이 `data/active_position.json`이면 **`data/active_position.pending_buy.json`**이다. 이 작업은 운영 journal을 생성/수정하지 않았으며, 이후 정상 주문 실행 시 코드가 저장한다.

1. 주문을 보내기 전에 PREPARE 증거(종목·목표 수량·NY 거래일 등)를 저장한다. 저장 실패 시 주문을 보내지 않는다.
2. 접수되면 주문 번호를 기록하고, 체결 대기 동안 유지한다.
3. 전량 체결이 확인되고 포지션이 파일에도 저장됐을 때 해제한다.
4. 부분체결/취소 미확인/접수 여부 불명은 보존한다. 재시작하면 같은 계좌·환경에서 복원한다.
5. 같은 날 종결·평탄화 확인은 기존 진입 전 원장 대조 경로를 따른다. 전일 0체결 취소는 해당 날짜로 재조회해 해제할 수 있다.
6. 전일 체결/부분체결인데 폐쇄 증거가 충분하지 않은 상태, 주문 번호 없는 PREPARE, 손상 파일은 자동 초기화하지 않는다. 명시적인 원장 재조정이 필요하며 새 진입은 차단한다.

계좌 식별자는 원문 대신 hash만 저장한다. journal 복원 자체가 매도 권한은 아니다. 실제 보유 수량이 확인돼야 매도하며, 불명확한 매수 상태를 지우기 위해 circuit breaker 상태를 임의 초기화하지 않는다.

## 4. 최종 회귀

격리 복제 환경, 비밀값/dotenv 제거, network/외부 write 차단 아래 실행했다. 실제 주문·브로커 API·Telegram 전송·실제 Live Runner 가동은 0건이다.

| 스위트 | 결과 |
|---|---:|
| 기존 B-03 | 39/39 통과 |
| B-03 추가 경계·통합 | 26/26 통과 |
| 시스템 핵심 인터락 | 9/9 통과, 직접 `python tests/test_system_integrity.py` 및 반복 실행 확인 |
| 학습 증거 | 6/6 통과 |
| 실전 거래 로그 | 4/4 통과 |
| 기존 forced shadow | 15/15 통과 |
| 새 쉐도우 5개 추론 | 실제 과거 입력 120개×5 = 600개, 반복 예측 동일; 성능 백테스트가 아님 |
| 실전 실행 안전성 | 3/4 통과, 기존 실패 1 |
| 방어/브리핑 알림 | 기존 오류 2 |
| B-05 프리마켓·장 전환 | 3/11 통과, 기존 실패 8 |
| 구 쉐도우 audit | 5/6 통과, 기존 실패 1 |

확장 회귀의 실패/오류 12개는 **정확한 수정 전 Runner/Broker 원본을 복제 환경에 다시 넣어 동일 테스트 ID와 결과를 재현**했다. 이번 변경으로 생긴 신규 실패로 분류하지 않는다. 총 122개 중 110개 통과, 기존 12개 미해결이다. 코드가 최신인지 검증하기 위해 각 결과에 소스 SHA와 격리 환경을 연결한다.

기존 12개 내용:

- 알림 모드 표식 emoji 기대값 차이 1개.
- 제거/변경된 `send_defense_alert`, `send_periodic_briefing` 인터페이스를 기대하는 알림 테스트 2개.
- 오래된 입력을 사용해 `DATA_STALE`에서 차단되는데 PRE_MARKET/RISK_BLOCK/진입 성공을 기대하는 프리마켓 테스트 8개.
- 구 `ShadowTracker`의 프로세스 재생성 후 동일 봉 진입 방지 영속성 미지원 1개. 새 5개 쉐도우 JSONL 원장과 혼동하지 않는다.

테스트를 반복할 때 예전 테스트의 미확인 BUY journal이 다음 테스트에 승계돼 핵심 테스트 하나가 차단되는 문제도 발견했다. 영속 주문 차단을 우회하지 않고 **테스트별 경로 격리**로 해결했다. production의 복구 차단 동작은 유지한다.

## 5. 해석과 남은 조건

이번 결과는 B-03의 재현된 코드 결함 및 모의 원장 회귀 통과를 의미한다. 모든 실전 위험이 제거됐다는 인증이 아니다. 실제 서버 원장 필드/계좌·환경 결합/최신성·취소 완료 증거의 확인, 운영 상태의 공식 reconciliation, 장중 feed 및 latency 관측은 별도로 필요하다.

전체 READY를 true로 바꾸거나 운영 `circuit_breaker_state.json`/모델 manifest를 수정하지 않았다. 실제 운영의 미확인 주문은 새 코드 적용 후 보수적으로 차단될 수 있다. 이를 실패로 보고 journal을 삭제해서는 안 된다.

이 작업에서는 Git checkout/reset/merge/commit/push를 하지 않았다. 기존 사용자 변경을 보존하며 변경 파일 검토와 Git 작업은 다음 단계로 분리한다. 상세 검증 증거는 `research/b03_fix_20261018/` 및 해당 실행의 격리 QA 결과에 있다. 복제 환경의 fixture/log에는 합성 계좌·주문 데이터가 있으므로 외부 배포하지 않는다.
