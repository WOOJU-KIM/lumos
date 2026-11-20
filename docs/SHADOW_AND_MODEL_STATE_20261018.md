# 실전 B72·쉐도우 5개: 구현 및 분석 기준

> **2026-10-22 쉐도우 피드 지연 복구:** 2초 초과 입력 한 건으로 run을 종료하던 처리를 변경했다. 2초 신선도는 유지하며 오래된/미래 입력을 버리고, 큐 넘침·공유 메모리 누락/덮어쓰기는 소비자 커서만 재동기화한다. 보유·매도 대기는 유지하고 미체결 가상 BUY·기존 가격 캐시를 폐기하며 신규 GBDT 진입은 다음 완료 15분 주기부터 재개한다. 일시적 crossed/비정상 호가와 피처 전달 실패도 명시적으로 거절하고 계속 수신한다. `FEED_GAP/FEED_RECOVERED`, 전송/큐/평가/파일 저장 지연과 `feed_health.data_complete=false`, 거래의 `feed_gap_count`를 기록한다. `valid=true`는 처리 무결성이며 공백 없는 시장 관측을 뜻하지 않는다. 공백 중 SL/MFE/MAE·체결을 복원하거나 가짜로 생성하지 않는다. 시장 입력은 버퍼링 후 health 주기/청산 때 flush하고 판단·거래 이벤트는 즉시 flush한다. 수명 제어·명시적 중지·손상 프로토콜/감사 저장 실패의 격리는 유지한다. 검증: 신규 10개 포함 관련 334개 및 무결성 9개 통과, `research/shadow_feed_recovery_20261022/final_result.json`. 실제 Runner/실주문/Telegram 전송 없이 다음 정상 Runner 시작부터 적용한다. 실전 주문 경로·모델·config는 변경하지 않았다.

> **2026-10-22 완료 봉 대기:** 사용자 승인으로 `CANDLE_READY_RETRY_SECONDS=1.0`, `CANDLE_READY_TIMEOUT_SECONDS=30.0`을 추가했다. Relay의 정규장 첫 봉 완료 전은 정상 대기로 표시하고 모델 추론·진입을 보류한다. 이후 최신 완료 봉 누락은 수집 시간을 포함한 30초 예산 안에서 재수집하며, 시간초과·주기 변경·종료 시 이전 봉으로 진입하지 않는다. 재시도는 평가 스레드에서만 수행하고 기존 WS 콜백/RelayRisk 청산 스레드는 유지한다. 쉐도우 설정 승인에는 새 운영 설정만 추가하며 고정 학습 정책과 모델은 보존한다. QA와 변경 전 사본은 `research/candle_readiness_20261022/` 참조. 이 변경은 비교 쉐도우의 `FEED_TOO_LATE_QUARANTINE` 자동 복구를 해결한 것이 아니며 운영 Runner를 실행하거나 READY를 새로 승인하지 않는다.

> **2026-10-21 텔레그램 재점검:** 현재 절대값 모델과 연쇄 전략 알림을 검증하고, 예외 경로의 이전 정규화 모델 기본 이름·보정 점수의 확률 표기·학습/실전 분포 동일 단정을 정정했다. 실전 매매 조건은 보존한다. [재점검 기록](TELEGRAM_RECHECK_20261021.md) 및 `research/telegram_recheck_20261021/result.json` 참조. 실제 전송·Runner 기동은 수행하지 않는다.

> **2026-10-21 후속: 15분 강제 진입 트래커 유지:** 기존 트래커를 현재 실전 절대값 모델의 방향·점수로 다시 활성화한다. 비교 쉐도우 3개는 유지하여 총 관측 전략 4개다. 아래 트래커 제외 기록은 이전 지시 시점의 상태다. 회귀 230개·무결성 9개와 실제 모델 점수의 진입/틱 청산 DB 저장을 검증했다. [유지 및 검증 기록](SHADOW_FORCED_RESTORE_20261021.md).

> **2026-10-21 최신 쉐도우 3개 개편:** 사용자 지시로 과거 실시간 쉐도우 전부와 15분 강제 진입 트래커를 실행 대상에서 제외하고, 실전과 동일한 절대값 모델의 `98% 단일`, `49/49·−1.5% 선제 연쇄`, `49% SL 체결 후 반대 49%`의 3개 원장만 선택한다. 실전 49/49·−1% 발동·손절 −2%는 유지한다. 과거 로그는 보존한다. [개편 기록](SHADOW_THREE_REVISION_20261021.md), `research/shadow_three_revision_20261021/activation_result.json` 및 실제 선택 파일을 참조한다. 아래 기존 모델 수·구성은 각 과거 시점의 기록이다.

> **2026-10-21 절대값 피처 실전 전환:** 사용자 지시로 기존 절대값 8개를 포함한 72개와 이전 순서로 보유 전체 기간을 재학습해 실전 `Absolute_72`로 전환한다. 49%/49%·진입·청산 조건과 기존 고정 쉐도우는 보존한다. 주간 자동 재학습도 선택 스키마를 유지하며 방향 비교 쉐도우 2개는 같은 새 모델을 사용한다. 검증·최종 활성화 상태는 [절대값 전환 기록](ABSOLUTE_72_ACTIVATION_20261021.md), `research/live_absolute_20261021/activation_result.json` 및 실제 활성 매니페스트를 함께 확인한다. 운영 러너는 종료 상태를 유지한다.

> **2026-10-21 B72 가중치 재학습·활성화:** 사용자 지시로 EMA200 이격도·QQQ 장기 추세 간격의 학습 gain 가중치를 각각 0.5로 반영했다. 최대 10년 상한 내 보유 전체 기간(2020~2026-10-20)으로 재학습한 SHA `1e580ffe9378...`를 활성화했다. 실전 49/49·64% 임계값은 유지한다. 방향 비교 쉐도우 2개는 새 모델에 연결하고 기존 모델 보존 쉐도우는 유지했다. 모의 틱 2,730건·통제 시나리오 6개·관련 회귀 216개·무결성 9개 검증, Runner 종료 유지. [변경 및 검증 기록](B72_GAIN_HALF_ACTIVATION_20261021.md).

> **2026-10-21 강제진입 트래커 점수 저장 수정:** 진입 방향 보정점수·판단 ID·모델 SHA를 진입부터 저장하고 청산에 연결한다. 과거 점수 45건(금일 22건 포함) 복원, 원본 없는 89건은 미확인으로 표시했다. 거래 손익은 보존했다. 무결성 9개·관련 311개 통과, 러너 종료 유지. [수정 기록](SHADOW_SCORE_FIX_20261021.md).

> **2026-10-21 저장 오류 후 연쇄 차단 복구:** 포지션 파일의 일시적 저장 오류를 주문 기록 실패와 분리했다. 청산 감시는 계속하고, 원장·미체결·파일 정합성과 재저장 성공 후에만 BUY 차단을 해제한다. 무결성 9개·관련 296개 통과. [변경 기록](POSITION_WRITE_RECOVERY_20261021.md).

> **2026-10-20 연쇄 매수 추가 검증:** 최종 전송가 계산 시 연쇄 예약·주문 의도와 최초 포지션 수량/평균가를 다시 대조한다. 양방향 연쇄, 최신 호가·예산·부분 체결·접수 불명 중복 차단을 검증했다. 무결성 9개·관련 282개 통과. 다음 정상 Runner 시작부터 적용. [검증 기록](CHAIN_SEND_QA_20261020.md).

> **2026-10-20 매수 전송가 수정:** 필수 원장 검사·유량/인증·TLS 준비 후 최신 WS Ask + config 버퍼로 매수 본문을 갱신한다. 예산·시간·호가·READY 가드는 유지하며, 0주 취소 확인은 위험 차단과 구분해 알린다. 무결성 9개·관련 272개 통과. 실행 중 Runner는 자동 갱신되지 않아 다음 정상 재시작부터 적용. [변경 기록](BUY_SEND_PRICE_20261020.md).

> **2026-10-20 진입 차단 알림 사유 수정:** 같은 15분봉 대기·호가 부족 등으로 수량이 0이 된 경우를 일괄 잔액 부족으로 표시하던 오류를 수정했다. 검사 조건은 유지하고 실제 분기 코드와 한글 설명을 표시한다. 무결성 9개·관련 263개 통과. 실행 중 프로세스는 재시작하지 않아 다음 정상 Runner 재시작부터 적용된다. [변경 기록](TELEGRAM_ENTRY_REASON_20261020.md).

> **2026-10-20 장중 진입 경쟁 조건 수정:** 최초 수량 계산 후 주문 의도 저장 전 빈 원장 대사가 예산을 지우는 경로를 재현·수정했다. 준비 예약의 인스턴스 소유권과 book 잠금으로 보호하고 실패·만료 시 해제한다. 미확정 매수 브리핑도 정정했다. 쉐도우 background I/O 제한을 제거하되 freshness 검사는 유지한다. 무결성 9개·관련 257개 통과, 다음 정상 Runner 실행부터 적용. [수정 기록](ENTRY_RESERVATION_FIX_20261020.md).

> **2026-10-20 텔레그램 최신화:** 최초·연쇄 진입 근거와 개별 추적 조건, 남은 포지션, 원장 확인 후 사이클 종료를 구분한다. 한쪽 청산에도 ‘100% 현금화’라던 중복 구 알림은 연쇄 Runner에서 억제했다. 무결성 9개·관련 251개 통과, 실제 전송·Runner 실행 없음. [알림 변경 기록](RELAY_TELEGRAM_20261020.md).

> **2026-10-20 사이클 제한·쉐도우 추가:** 후속 지시로 실전은 **최초 1회 + 연쇄 1회 후 모두 청산될 때까지 추가 매수 금지**로 변경했다. 반복 연쇄는 차단하고, 재시작에도 예약을 보존한다. 점수 기준만 제거/점수와 중립 모두 제거한 진입 비교 쉐도우 2개를 같은 49%·49% 방식으로 추가했다. 현재 sidecar는 8개 전략·13개 원장. 무결성 9개 및 관련 240개 통과, 운영 프로세스 미실행. [최신 변경 기록](RELAY_CYCLE_AND_DIRECTION_SHADOWS_20261020.md).

> **2026-10-20 반대 매수 발동선 변경:** 후속 지시로 실전 연쇄 매수는 자기 매수가 대비 **-1.0%**에서 발동한다. 고점 대비 추적 -0.3%, 최초/연쇄 트레일링 발동 +1.5%/+1%, 손절 -2%는 유지한다. 격리 223개 검증 통과, 다음 Runner 실행부터 적용. [변경 기록](RELAY_TRIGGER_1PCT_20261020.md). 아래 -1.5% 발동은 이전 시점의 기록이다.

> **2026-10-20 후속 수정:** 연쇄 전략의 미추적 보유 EOD·외부 수량 변경·완료 15분봉·계좌 결합 검사를 수정하고 격리 222개 검증을 통과했다. 현재 실전계좌의 실제 원장 대사로 READY 증명 불일치를 해소했다. 운영 Runner·실주문은 미실행. 현재 상태는 [후속 수정 기록](RELAY_FIXES_20261020.md)을 참조한다. 아래 READY 차단 표시는 각 과거 점검 시점의 기록이다.

> **2026-10-20 실전 연쇄 전략 전환:** 후속 사용자 지시로 기본 실행 경로를 `OPPOSITE_RELAY_V1`로 선택했다. 최초 49%/+1.5% 트레일링, −1.5%에서 빈 반대 종목 49% 연쇄 진입/+1% 트레일링, 공통 추적 −0.3%·손절 −2%. B72는 최초 신호에 계속 사용하고 이전 98% 단일 포지션 전략을 별도 쉐도우로 보존해 6개 전략을 연결했다. 격리 213개 검증 통과, 운영 Runner 미실행. 현재 계좌와 READY 증명이 달라 실전 신규 진입은 차단된다. [실전 전환 기록](OPPOSITE_RELAY_ACTIVATION_20261020.md)을 참조한다.

> **2026-10-20 초기 손절 변경:** 사용자 지시에 따라 실전·기존 15분 강제진입 트래커·추가 쉐도우 5개(primary/forced)의 초기 손절을 −2.0%로 고정했다. ATR에 따른 3.2% 확대를 중단하고 트레일링 등 다른 조건은 유지한다. 모델/학습 고정 정책은 보존하며 별도 승인 실행 정책으로 config 변경을 검증한다. 다음 정상 Runner 시작부터 적용. [변경·검증 기록](FIXED_STOP_20261020.md), `research/fixed_stop_20261020/result.json`을 참조한다.

> **2026-10-20 Runner 수명 연결:** 사용자 후속 지시에 따라 쉐도우는 `FOLLOW_LIVE_RUNNER` 모드로 동작한다. 실전 Runner의 background startup에서 자동 시작하며, 정상 종료 요청 또는 소유 Runner 프로세스 종료를 확인하면 함께 종료한다. 다음 Runner 시작을 위해 enabled는 true로 유지한다. 현재는 운영 프로세스를 실행하지 않았다. [Runner 수명 연결 기록](SHADOW_RUNNER_LIFECYCLE_20261020.md)의 적용 범위·검증과 새 SHA를 참조한다.

> **2026-10-20 쉐도우 청산 감시 후속 수정:** 호가만 수신될 때 가격 기반 청산이 실행되지 않은 문제를 수정했다. WebSocket Bid/명시적 last-price의 출처를 보존한 MARK와 config cadence의 POLL을 연결하고, 독립 CLOCK 및 무감시 신규 진입 차단을 추가했다. 실제 거래 틱으로 위장하지 않으며, 실전 코드·config·모델은 변경하지 않았다. 원인·검증·실행 적용 범위는 [쉐도우 청산 감시 수정](SHADOW_EXIT_MONITORING_20261020.md)을 참조한다. 아래 초기 구현의 tick-only 청산 설명과 구분한다.

> **2026-10-19 WebSocket 전용 쉐도우 입력:** `feed_bridge.market`은 REST 백업 가격을 게시하지 않는다. 이미 실행 중인 구 publisher와의 호환을 위해 `live_feed`도 REST 프레임을 평가에서 제외하고 `stats.ignored_rest`만 증가시킨다. REST 한 건으로 전체 run을 격리 종료하던 처리를 제거했으며, 손상 프레임·지연·overflow 검증은 유지한다. 실전 Runner/가격 캐시/콜백·config·모델은 변경하지 않았다. 쉐도우만 재시작했으며 WebSocket 호가 수신을 확인했다. 이는 거래 틱 수신이나 5개 모델의 판단 완료 증거와는 구분한다. 신규 회귀 3개 및 최신 소스 격리 복제에서 핵심 무결성 9개 통과. 변경 SHA와 관측 상태: `research/shadow_ws_only_20261019/result.json`.

> **2026-10-19 코드 정리:** 주문 상태 I/O 명시화, 중복 원장 검사 추출, 쉐도우 거래 이벤트 연결 ID 보완은 [동작 보존 코드 정리 기록](CODE_CLEANUP_20261019.md)을 참조한다. 관련 151개 테스트와 전후 비교를 통과했으며 모델·설정·매매 판정 규칙은 유지했다. 최신 변경 소스 SHA는 `research/code_cleanup_20261019/result.json`에 있다.

> **실행 기간 변경:** 사용자 지시에 따라 쉐도우는 2026-10-19부터 명시적 중지까지 계속 평가한다. 실제 실행 기간은 별도 `runtime_control.json`으로 제어하며, 기존 policy의 고정 주간은 학습/오프라인 provenance다. [무기한 실행 변경 기록](SHADOW_CONTINUOUS_RUNTIME_20261018.md)을 함께 읽는다. 실전 매매 경로와 모델은 변경하지 않았다.

> **B-03 호환성 후속 확인:** [수정 전후 정상 주문 경로 비교](B03_NORMAL_ORDER_COMPATIBILITY_20261018.md). 실제 조회 주문 식별자 22개 판정·수량이 일치했고 평탄 계좌의 진입 검사와 모의 정상 매수→매도 사이클이 양쪽 모두 통과했다. 앞서 발견한 READY 검사 문제는 B-03 이전부터 동일하게 존재하며, B-03가 새로 거래를 차단한다는 근거로 해석하지 않는다.

> **운영 준비 후속 점검:** [읽기 전용 READY 점검](READINESS_CHECK_20261018.md)에서 실제 조회·WebSocket 인증 결과와 함께, 현재 계좌와 일치하지 않는 기존 검증 증명을 readiness 함수가 수용하는 결함을 확인했다. 전체 READY는 보류하며 이전 회귀 통과와 구분한다.

> **추가 후속 검증:** 기존 회귀 실패 12건의 수정과 관련 133개 통과 결과는 [회귀 후속 기록](REGRESSION12_FIX_20261018.md) 및 `REGRESSION12_FIX_STATE_20261018.json`을 참조한다. 아래 과거 상태·SHA와 구분하며 전체 실전 READY는 여전히 미승인이다.

작성 기준: **2026-10-18 KST**. 대상 평가 주간: **2026-10-19~23, America/New_York**.

> **같은 날짜의 후속 코드 변경:** B-03 원장 정합성·미확인 매수 복구 개선과 회귀 결과는 [B-03 변경 기록](B03_ORDER_SAFETY_FIX_20261018.md)을 함께 읽는다. 아래 내용과 기존 SHA 스냅샷은 B-03 수정 전 구축 기준을 포함한다. 최신 소스 해시는 `B03_FIX_STATE_20261018.json`으로 보완하며, 전체 실전 READY는 여전히 미승인이다.

이 문서는 대화에서 승인받아 구축한 기능, 실제 활성화 이력, 저장 형식과 한계를 정리한다. 미래의 상태를 보장하지 않는다. 분석·코드 리뷰 전에 실제 소스와 매니페스트를 다시 읽고, 동반 스냅샷 `SHADOW_MODEL_STATE_20261018.json`의 SHA256과 비교한다. 매니페스트/로그를 읽지 않고 이 문서만으로 현재 모델이나 READY를 판정하지 않는다.

## 1. 현재 상태와 읽는 순서

1. 루트 `AGENTS.md`의 불변 규칙과 현재 사용자 지시를 먼저 확인한다.
2. `models/active_model_manifest.json`: 실제 실전 주 모델·이전 모델·Fallback의 경로와 SHA256.
3. `research/shadow_five_full15m_20261018/artifacts/manifest.json`: 쉐도우 5개 실제 아티팩트.
4. 같은 디렉터리 `policy.json`, `qa/feed_listener.json`, `qa/feed_autostart.json`, 최신 `runs/live_shadow_*`: 고정 정책과 관측 상태.
5. 본 문서의 소스 지도, 학습 provenance, 검증 결과를 따라 원문을 재확인한다.

현재는 실전 B72 교체와 쉐도우 가중 학습 모델 연결이 완료됐다. 작성 시 수신기 상태는 `LISTENING`, 실제 시장 입력 수신은 0건이다. 이 상태는 연결 대기이며 장중 정상 동작을 입증하지 않는다. PID나 heartbeat는 고정 사실로 취급하지 않는다.

**B-03 주문 안전성 문제는 `NOT_CLEARED`, 전체 시스템 `live_ready=False`다.** 모델 교체/오프라인 QA 통과를 실계좌 운영 READY와 동일시하지 않는다. 이번 구축·분석 과정에서 실제 Live Runner 실행, 실주문, 브로커 API 호출, Telegram 전송은 수행하지 않았다. 테스트의 모의 객체 생성은 실거래 가동이 아니다.

## 2. 실전 모델 교체

| 구분 | 실제 내용 |
|---|---|
| 실전 모델 계열 | `B_Normalized_72`, 스키마 `B_normalized`, 3-Class LightGBM |
| 활성 버전 | `models/versions/model_normalized_72_live17_20261016_0deb9282eab7.pkl` |
| 활성 SHA256 | `0deb9282eab7e1bf1a5b2d694878f13534ea0cc2632f9b57676d46d906b24932` |
| 교체 전 버전 | `models/versions/model_normalized_72_20261017_050941_4bb24c5217cd.pkl` |
| 교체 전 SHA256 | `4bb24c5217cd810a95dc467fcfdbfa227c912bfde7fe48e27f35507a726dc3f7` |
| Fallback | `models/model_champion.pkl`, `v4_champion_fallback`; 교체하지 않음 |
| 원시/학습 표본 | native 15m 44,300개 → 과거 학습 44,256개 + 실전 17개 |
| 제외 | warmup/NaN 38개, 미확정 horizon 말단 6개 |
| 학습 범위 | 과거 보유 데이터부터 최대 520주, 실제 약 6년; 학습 기점 2020-07-29, 마지막 라벨 원점 2026-10-16 14:15 NY |
| 데이터 cutoff | 2026-10-16 16:00 NY; horizon의 미래 가격까지 cutoff 이전 데이터만 사용 |
| 실전 가중치 | 일반 표본 1.0, 수용 실전 표본 2.5; 기존 balanced class weight 유지 |

활성 포인터인 `active_model_manifest.json` 한 파일을 원자적으로 교체했다. 모델은 새 불변 버전 파일로 복사했으며 기존 주 모델, Champion, `m_moe_orchestrator.pkl`, 운영 DB, 설정값은 보존했다. 새 프로세스는 활성 매니페스트를 따라 로딩한다. 실행 중 프로세스에 대한 적용 여부는 별도 메모리 모델 SHA/로딩 상태로 검증해야 한다.

증거: `research/live_retrain_20261018/training_result_v2.json`, `final_validation_v2.json`, `activation_live17_v2_20261018/activation_result.json`, `isolated_loader_qa.json`. 백업은 같은 activation 디렉터리의 `previous_manifest.json`, `previous_active_model.pkl`; 매니페스트에도 previous 경로·SHA가 있다. 롤백은 별도 변경 작업이며 사용자 지시 없이 실행하지 않는다.

후보 PKL 안의 `OFFLINE_CANDIDATE_NOT_ACTIVATED` 및 `training_quality.approved_for_live=False`는 제작 당시의 메타데이터다. 승인된 활성화는 매니페스트와 activation 결과로 확인한다. 활성화를 이유로 기존 데이터 품질 제약이나 B-03 미해결 표시를 삭제하지 않는다.

## 3. 실전 경험 표본의 의미와 한계

- 원본 실전 완료 기록 35개 중 SOXL/SOXS 27개, 다른 자산 계열 8개다.
- 현재 피처 스키마로 복원 가능한 것은 **17개, 모두 SOXL**이다. 나머지 SOXL/SOXS 10개는 피처가 불완전해 제외했다. SOXS 실전 학습 표본이 있다고 주장하지 않는다.
- 구 로그의 NumPy 정수 직렬화 누락 때문에 없던 `panic_signal`을 기록된 VIXY/IEF 방향 부호에서 메모리 내 유도했다. `INFERRED`이며 원본 DB/JSON은 고치지 않았다.
- 과거 피처는 소수 4자리 스냅샷이며 정확한 모델 입력과 완전히 같다고 보증할 수 없다.
- 과거 청산 가격은 앱의 최신 호가/가격 proxy다. 증권사 체결 평균가·수수료와 정산된 순손익이 아니다. 수수료 unknown을 0으로 처리하지 않는다.
- 실전 B72는 **기존 production 클래스 라벨 규칙**을 그대로 적용했다. 17개 라벨은 SHORT 8, NEUTRAL 7, LONG 2다. 이는 관측된 반대 종목의 수익 정답이 아니다.
- 쉐도우 수익 모델은 B72 클래스 라벨을 복사하지 않는다. 실제 매매한 방향의 앱 기록 gross return에서 최소 왕복 0.20% 추정 비용을 차감한 **추정 net target**을 학습한다. 반대 방향 실전 target은 생성하지 않는다.
- 따라서 B72와 쉐도우에 같은 17건과 2.5 가중치를 사용해도 목적함수·라벨 정의·중복 처리 방식이 다르다.

## 4. 쉐도우 모델 5개

홈 디렉터리: `research/shadow_five_full15m_20261018/`. 활성 학습 버전: `artifacts/live_weighted_v1/`. 각 모델의 정확한 경로·SHA는 쉐도우 매니페스트를 따른다.

| 모델 ID | 입력 | 학습/출력 |
|---|---|---|
| `NET_GBDT` | 최신 72개 | 방향별 LGBMRegressor 2개, 비용 차감 수익률 예측 + OOF 잔차 보정 |
| `NET_LAG_GBDT` | 32×72, 최신→과거 순서로 펼친 2,304개 | 방향별 회귀 2개, 같은 net EV 단위 |
| `NET_TCN` | 32×72, 과거→최신 | CPU causal convolution, 훈련 평균/표준편차 정규화 후 ±8 clipping, 2-head masked MSE |
| `FULL_COND_PAYOFF_F1.0` | 시장 피처 85 + 방향/추세/평균회귀 flag 3 = 88개 | 승리 확률 classifier + 승리 이익/패배 손실 regressor; `p×gain−(1−p)×loss` |
| `FULL_HIST_PAYOFF_F1.0` | 동일 88개 | 승리 확률 + 방향×추세×평균회귀 8개 그룹의 가중 이익/손실 평균 |

공통:

- 방향 배열 순서는 **[SOXS, SOXL]**이다. 실전 B72 클래스 순서 **[SHORT, NEUTRAL, LONG] = [0,1,2]**와 혼동하지 않는다.
- 72개 canonical 피처와 추가 12개 추세/MR 피처, `minutes_to_eod`가 85개를 구성한다. 정확한 순서의 원장은 `inputs/schema.json`이다. 과거 문서의 135개/다른 스키마와 동일하다고 가정하지 않는다.
- 가격 스케일 제거가 필요한 피처는 production `B_normalized` 정의를 따른다. GBDT는 별도의 전체 피처 표준화를 하지 않으며 TCN만 별도 training scaler를 사용한다.
- 현재 5개 net EV 임계값은 모두 **0.0**, 양수여야 진입하며 동률은 NONE이다. B72 최종 확신도 64%/70.4%를 이 모델들에 대입하지 않는다.
- Selector는 trend 또는 MR eligibility를 진입 단계에서 검사한다. 학습 데이터 자체는 eligibility로 자르지 않는다.
- `F1.0`은 과거 비교 거래 빈도의 목표 배율 명칭이다. 확신도 100%나 거래 빈도 보장을 뜻하지 않는다. 과거 calibration 진입 수는 COND 0, HIST 2였으며 현재 가중 모델의 미래 실전 거래 횟수가 아니다. 거래를 늘리려고 음의 EV 진입을 허용하지 않는다.

### 학습 데이터와 가중치

- native 피처 원시 44,300개에서 calendar-aligned 평가 입력 34,429개 구성. 한 방향 라벨이 없다고 다른 방향의 유효 라벨까지 버리는 paired intersection을 사용하지 않는다.
- 최종 방향별 수용 target 수 [SOXS, SOXL]: static/selector [13,206,24,405], LAG/TCN [13,206,24,402]. 3개 차이는 sequence context 가용성 차이다. 모든 raw bar가 학습 정답을 갖는다는 뜻은 아니다.
- 실전 17개 각각 2.5 가중치 적용. GBDT/selector의 `sample_weight`, TCN의 방향별 weighted masked loss, HIST 그룹 평균에 반영했다. 단일 목표 17개만 존재하며 selector의 분기별 실전 수는 classifier 17, win regressor 9, loss regressor 8이다.
- 같은 cycle/선택 방향의 역사적 proxy 라벨 17개를 실전 표본으로 대체해 중복 학습을 막았다. 반대 방향의 유효 역사 라벨은 보존했다. B72 재학습은 기존 production 방식대로 과거 표본에 실전 17개를 추가했다.
- 구 로그에는 exact 32-row tensor가 없다. LAG/TCN은 **과거 DB에서 재현한 context + 반올림된 당시 scalar snapshot**으로 구성했다. `OBSERVED` tensor라고 표시하지 않는다.
- OOF 학습 cutoff의 최댓값은 2026-10-12, 수용 실전 최초 cycle은 10-15다. 따라서 과거 12개 fold 학습에는 이 17개가 들어갈 수 없다. 기존 purged OOF 보정값·임계값·TCN epoch 수를 유지했다. 가중 최종 모델에 대해 새 OOS calibration이 검증된 것은 아니다.
- 가중 학습 entry point는 `refit_live_weighted.py`, 연결은 `promote_live_weighted.py`다. 일회성 버전 파일을 덮어쓰지 못하게 설계돼 있다. 기존 `train.py`/`refit_selectors_all_labels.py`만 다시 실행하면 이 가중 학습 경로와 같다고 주장할 수 없다. 쉐도우 주간 자동 재학습 스케줄러까지 구축됐다는 근거는 없다.

## 5. 실전→쉐도우 데이터 전달과 자동 실행

```text
main.py → KiwoomLiveRunner
  ├─ MoEMetaOrchestrator → MLFeatureEngine → 활성 B72 → 기존 실주문 경로
  ├─ KiwoomWebSocketStreamer → tick/quote 관측 publisher
  └─ MLFeatureEngine.predict_signal_full → pre-inference 피처 publisher
          ↓ Windows named pagefile ring
       live_feed.py (독립 쉐도우 프로세스)
          → PacketBuilder → Candidate 5개 → 가상 Ledger → JSONL Audit
```

- Runner가 실전 engine/streamer를 `bind`하고, 정상 background startup에서 `autostart`를 별도 helper thread로 요청한다. simulation 및 격리 QA에서는 연결/자동 실행하지 않는다.
- `start_shadow.py`가 launcher mutex와 consumer singleton, PID+생성 시각을 확인해 기존 프로세스를 재사용한다. 새 프로세스는 hidden/below-normal priority이며 모델 로드 후 background I/O mode를 적용한다.
- publisher는 모델 계산, 디스크 로그, broker API 또는 consumer 응답 대기를 수행하지 않는다. lock은 nonblocking이다. 약 64MiB의 2,048×32,768-byte ring을 사용한다.
- heartbeat가 없거나 2초 이상 오래되면 publisher는 best-effort no-op한다. 슬롯 덮어쓰기·누락·지연·overflow는 데이터 손실/격리 사유이며 전달 보장이 아니다. producer 예외로 실전 매매를 막지 않는다. 전체 시스템 지연 0을 입증한 것은 아니다.
- 실제 WS에서 온 tick/quote만 수용한다. REST fallback은 실시간 관측 tick으로 바꾸지 않는다.
- 피처는 32×84와 시각을 전달하고, consumer가 calendar `minutes_to_eod`를 붙인다. 날짜·정규장·15분 cycle·as-of cross asset·ATR context를 검증한다. SQLite 시장 DB는 URI `mode=ro`, `query_only`로 읽는다. 브로커/외부 API 호출은 없다.
- cross asset provenance DB 확인은 atomic한 원본 snapshot 보증이 아니다. enabled Veto/60m filter의 실제 flag export가 없는 경우 조용히 통과시키지 않고 거절한다.
- 현재 정책 적용은 **10/19~23 NY 주간 한정**이다. 다음 주까지 무기한 자동 거래한다고 가정하지 말고 새로운 정책/QA를 검토한다.
- 정상 실전 기동이 쉐도우의 실제 장중 수신·모든 모델의 진입을 보장하지 않는다. `received`, `decisions`, reject/error, heartbeat 및 실제 로그를 관측해야 한다.

## 6. 가상 거래와 실전의 차이

- 모델별 **primary**: 독립 가상 자금 USD 10,000, 단일 포지션, 98% 배분, 기존 포지션/미체결이면 중복 진입 차단.
- 모델별 **forced**: 방향별 예측 비교용 독립 unit trade, 복수 중첩 가능. 합쳐서 하나의 투자 계좌 수익률/MDD로 계산하면 안 된다.
- 실전과 같은 데이터에서 출발하되 가상 매수는 다음 유효 ask, 매도는 다음 bid로 처리한다. 의도/주문 요청은 체결이 아니다. quote가 없으면 체결을 꾸며내지 않는다. 취소·미해결 포지션을 실패 표본에서 삭제하지 않는다.
- 비용은 최소 왕복 0.20%를 한 번 차감한다. 실제 queue position, 시장 충격, 브로커 거절·수수료를 완전히 재현한 체결은 아니다.
- tick은 peak/초기 SL·TP·time을 처리하며 trailing은 config cadence의 가상 POLL로 검사한다. 실전과 polling phase 및 판단 지연까지 동일하다고 보장하지 않는다.
- EOD·time 이벤트 역시 feed 처리 흐름을 확인해야 한다. 데이터 중단 상황에서 가상 CLOCK/청산이 자동 완결됐다고 가정하지 않는다.
- 가상 원장은 메모리 상태다. 프로세스 재시작 시 기존 미청산 원장을 로그에서 자동 복구하는 구현이 확인되지 않았다. 장중 임의 재시작은 별도 run으로 구분하고 미해결 포지션을 추적한다.

## 7. 저장 경로와 이벤트 조인

### 새 쉐도우 5개: JSONL

`research/shadow_five_full15m_20261018/runs/live_shadow_<실행시각>/`

| 파일 | 내용 |
|---|---|
| `decision_inputs.jsonl` | 허용 목록으로 제한된 판단 입력: sequence 32×72, full_features 85개, 피처/cross 시각, cycle, decision_id, ATR metadata |
| `events.jsonl` | `MARKET_INPUT`, `DECISION`, `VIRTUAL_BUY`, `VIRTUAL_EXIT_INTENT`, `VIRTUAL_SELL`, 취소/거절/오류, `RUN_START/STOP/QUARANTINED` |

JSONL 한 줄이 JSON 객체 하나다. append 후 `flush()`하며 run마다 새 디렉터리다. **fsync·정전 시 무손실·자동 DB 색인을 보장하는 저장소는 아니다.** 이 5개 전용 거래 DB는 만들지 않았다. 시장 DB의 readonly 조회와 쉐도우 기록 저장을 혼동하지 않는다.

`DECISION`에는 model_id/version, decision_id, input_sha256, expected_net_return, threshold, side/forced_side, prediction details, primary/forced action이 있다. `VIRTUAL_SELL`에는 model_id/cohort/decision_id/version, 진입·청산 시각과 가격, 수량, gross/net/cost, reason이 있다. 수익률은 primary와 forced를 분리해 집계한다.

**현재 추적성 제약:** `VIRTUAL_BUY` 일부 진입 이벤트는 model_id/cohort/decision_id가 직접 붙지 않는다(`runtime.py:Ledger.on_quote`). 시간·종목만으로 동시 모델 진입을 단정 조인하지 않는다. 명시적 ID가 있는 판단·청산과 비교하고, 불명확한 join은 `UNVERIFIED`로 남긴다. 이 문서화 작업에서는 실행 로직을 수정하지 않았다.

### 실전 거래 및 학습 증거: SQLite + CSV + 별도 journal

| 경로 | 역할 |
|---|---|
| `data/live_experience.db` | 완료 거래 `live_trades`, 주문 이벤트 `live_order_events`, 시스템 에러 등 기존 DB |
| `data/live_trades.csv` | 완료 거래의 평면 CSV; DB와 중복 제거해 읽음 |
| `data/live_experience.db.learning.db` | 새 비동기 학습 증거 journal, `learning_events(event_id,event_type,observed_at,payload_json)` |
| `data/live_trades.csv.learning.jsonl` | learning DB 기록 실패 시 append fallback |

새 journal은 다음 정상 Runner/logger 초기화 때 생성된다. 존재 여부는 실제 실행 이후 확인한다. `DECISION/EXECUTION/ORDER/DECISION_OUTCOME`을 구분한다. queue worker가 별도 DB에 기록하며 decision enqueue에서 DB 대기를 하지 않는다. 양쪽 저장이 실패하면 메모리 보관+critical 로그이며 강제 종료 후 복구가 보장되지 않는다. 완료 거래의 기존 DB/CSV 쓰기 전체가 비동기로 바뀐 것은 아니다.

### 기존 쉐도우와 혼동 금지

`core/shadow_tracker.py` 및 기존 `shadow_trades.db` 등은 별도의 구 점수 연구 경로다. 새 5개 JSONL 원장과 같은 데이터셋으로 가정하지 않는다. 기존 DB의 0점/score 해석 등 알려진 제약은 원 코드·로그를 재확인한다. 존재하는 파일 이름만으로 현재 사용 중이라고 판정하지 않는다.

## 8. 학습 기록 보강 변경점과 소스 지도

| 소스 | 승인된 변경/책임 |
|---|---|
| `core/learning_audit.py` | NumPy scalar 포함 유한 수치 직렬화, explicit null, pre-inference history/input/raw probability/model SHA 증거, 실행 메시지 whitelist·주문 참조 hash |
| `core/ml_engine.py` | thread-local 정확한 추론 snapshot, 확률 보정 전 raw 값과 실제 정제 입력 보존, stale reset; 학습 표본 coverage, REAL mode/불완전 audit 필터; 쉐도우 피처 관측 hook |
| `core/moe_orchestrator.py` | 실제 추론 snapshot 우선 사용, NumPy 정수 누락/불필요한 반올림 개선, 모델 일치 조건 아래 learning_snapshot 전달; 기존 방패·판단 유지 |
| `core/live_runner.py` | snapshot/model 참조를 포지션·완료 거래에 전달; 판단·판단 결과·실행 메시지 비동기 관측, 자동 쉐도우 시작 요청; 기존 주문/포지션 인터락 유지 |
| `core/kiwoom_ws_streamer.py` | 기존 캐시/콜백을 보존하는 WS tick/quote 관측 publisher |
| `core/live_experience_logger.py` | full-precision feature JSON/audit metadata, separate learning journal/fallback, DB·CSV 누락 병합, 저장 실패 명시 |
| `feed_bridge.py`, `start_shadow.py`, `live_feed.py` | nonblocking feed, singleton 자동 시작, 수신/검증·격리 |
| `adapters.py`, `runtime.py`, `storage.py` | 모델 인터페이스/SHA, 가상 원장, JSONL audit |
| `feature_source.py`, `common.py` | application constructor 없이 순수 코드 추출, readonly 데이터, network/외부 write/child-process 차단 |
| `prepare.py`, `train.py`, `refit_selectors_all_labels.py` | 전체 15m 피처 및 방향별 라벨/OOF의 초기 구축 단계 |
| `refit_live_weighted.py`, `promote_live_weighted.py` | 17건×2.5 가중 최종 재학습과 쉐도우 매니페스트 전환 |
| `research/live_retrain_20261018/train_candidate.py`, `activate_candidate_v2.py` | 실전 B72 오프라인 후보 재학습 및 승인된 2단계 활성화 |

새 학습 snapshot은 피처 순서/해시, 모델 SHA, raw class probabilities, actual_model_input, 최대 스키마 길이(B72는 최대 72행)의 history를 보존한다. null/불완전은 명시하며 missing 값이 없는 것처럼 저장하지 않는다. 실행 메시지는 앱 주문 의도/접수와 증권사 체결 증거를 분리하고, 없는 fill price·fees는 추정값으로 채우지 않는다. future 학습에서 이 journal의 정확한 tensor/체결 증거를 자동으로 모두 활용하도록 학습 join까지 완성됐다고 가정하지 않는다. 현재 가중 스크립트의 실제 데이터 선택을 재확인한다.

## 9. 설정과 과거 상태 문구의 주의점

- 현재 `config.py`의 `GBDT_CONFIDENCE_THRESHOLD`는 **0.64**다. `CALIB_BASE=0.50`, `CALIB_PIVOT=0.333`, `CALIB_SCALE=1.15`; 최종 점수는 clamp한 affine 값이며 합이 1인 확률이나 실증 승률이 아니다. Neutral 최고/동률이면 NONE이다.
- `AGENTS.md`의 70.4% 문구와 현재 config 64% 사이에 불일치가 있다. 이 문서는 그 사실을 기록할 뿐이며 정책 변경 권한이 아니다. 기존 요청대로 config를 임의 수정하지 않았다. 분석에서 어느 임계값을 사용했는지 반드시 명시한다.
- 현재 Veto=False, 60m 방패=False. 방패를 꺼도 cross/macro 피처는 모델에 남는다.
- 배분 0.98, 초기 SL 2.0~3.2%/ATR×1.5, max TP 10%, trailing +1.5%에서 고점−0.3%, time-stop 90분, 15:30 NY 신규 진입 종료·15:55 EOD. 실제 실행 시 config와 로그를 재검증한다.
- NY 10/19 주간은 EDT: 09:30=22:30 KST, 15:30=다음 날 04:30 KST, 15:55=다음 날 04:55 KST. DST 이후에는 고정 KST 시각을 재사용하지 않는다.
- 쉐도우 `manifest.status=TRAINED_NOT_CONNECTED`, `runtime_enabled=false` 및 policy의 과거 feed 설명은 초기 작성 문자열이 유지돼 있다. 현재 adapter도 이 legacy status를 검사한다. 실제 연결은 별도 launcher/heartbeat/run으로 확인하며 단순히 true/CONNECTED로 바꾸면 호환성 오류가 날 수 있다. 이 불일치는 문서로 기록했으며 이번 작업에서 바꾸지 않았다.
- `docs/MAINTENANCE_GUIDE.md`의 2026.09 기준 104주/62%/TP 3%는 과거 설명이다. 현재 운영 설정으로 가져오지 않는다. 현재 config, 활성 artifact, label TP와 청산 TP를 구분한다.

## 10. 검증 증거와 재실행 원칙

| 검증 | 관측 결과 및 근거 |
|---|---|
| 핵심 무결성 | `tests/test_system_integrity.py` 9개 통과, credential/network 차단 복제 환경에서 실행 |
| 학습 증거 | `tests/test_learning_capture.py` 6개 통과: 정확한 snapshot/누락 표시, whitelist, fallback, 느린 저장의 enqueue 분리, REAL 2.5 가중 수용 |
| 쉐도우 기능 | `qa.py`: runtime 29개, JSONL audit, 최근 1주 입력 600회 및 반복 예측 일치; 수익률 백테스트가 아님 |
| 코드 보존 | `qa_feed_hooks.py` 및 `capture_ast.py`: 승인된 관측 hook/기록 변경을 분리해 핵심 AST 보존 확인 |
| 가중 학습 | `artifacts/live_weighted_v1/training_qa.json`: 5개 모두 17개×2.5; TCN 전체 반복 state 일치; 120입력/모델 반복 추론 |
| 활성화 | activation QA: production loader 7항목, 실제 입력 120개 확률 유한/합계1/클래스순서 검증; 단일 manifest 변경과 다른 원본 194개 무변경 확인 |
| 체결 경로 simulator | `qa_threshold_paths.py`: SL/TP 동시 SL 우선, 현재 high로 현재 low의 trail 갱신 금지, 비용1회, gap SL, time stop 5개 통과 |

테스트를 실프로젝트에서 무조건 실행하면 import/constructor가 운영 경로에 접근할 수 있다. 기존 격리 복제 환경은 `%TEMP%/lumos_shadow5_integrity_20261018`이며 sitecustomize가 환경 비밀값/dotenv/network/외부 write/child-process를 차단한다. 재사용 전 복제 core/config/test와 현재 파일 SHA를 비교한다. `PYTHONPATH`를 복제 root로 설정하고 `LUMOS_SHADOW_FEED_ISOLATED=1`로 실행해야 한다. 단순히 cwd만 바꿔 격리가 됐다고 가정하지 않는다. Temp가 없거나 오래되면 같은 격리를 새로 구성한다.

초기 `qa/protected_before.json` 등의 baseline은 승인된 실전 모델 교체 전 상태다. 현재 매니페스트·새 버전으로 인한 차이를 무조건 새 회귀로 판정하거나 baseline을 조용히 갱신하지 않는다. activation 결과 및 원본 보존 증거로 승인 변경을 설명하고 나머지 차이를 검토한다. 이후 QA가 이 오래된 baseline assertion 때문에 실패할 수 있으므로 결과를 숨기지 않는다.

## 11. 64% vs 70.4% 비교 이력: 섞으면 안 되는 두 실험

공통: 2026-07-27~10-16 NY, 가용 59거래일 중 SOXL 경로 불완전한 8/14·8/24·9/10·10/6을 공통 제외한 55일, 15m 판단/5m 청산, 왕복 0.20%, 98% 배분. CAGR은 짧은 기간의 연율화일 뿐 예측이 아니다.

| 실험 | 임계값 | 12주 수익률 | 횟수 | 승률 | MDD |
|---|---:|---:|---:|---:|---:|
| 교체 전 B72 **학습 방식**, 주간 과거-only WFA | 64% | −12.58% | 56 | 53.57% | −23.69% |
| 동일 WFA | 70.4% | +1.67% | 7 | 57.14% | −3.76% |
| **교체 후 활성 PKL 그대로**, 학습기간 내 재현 | 64% | +57.92% | 51 | 76.47% | −4.35% |
| 동일 활성 PKL 재현 | 70.4% | +17.96% | 16 | 81.25% | −3.40% |

두 실험은 평가 방식이 다르므로 수익률 차이를 실전 17건 가중 학습의 개선 효과로 해석하면 안 된다. 후자는 미래 데이터까지 학습한 고정 모델을 과거에 적용한 **IN_SAMPLE 진단**이며 공식 OOS 백테스트나 예측력 증거가 아니다. 실전 틱 청산과 5m OHLC의 차이도 남아 있다.

증거 디렉터리: 쉐도우 홈의 `threshold_64_704_20261018/`와 `threshold_64_704_live17_20261018/`. 각각 `results.json`, `signal_audit.csv`, `trades_*.csv`, `equity_*.csv`, `comparison.png`. 실행 소스는 `threshold_64_704.py`, `replay_active_live17_thresholds.py`, `plot_threshold_64_704.py`; 공식 WFA 원본은 수정하지 않았다. 사용자 교정 이후 비교 대상은 교체 후 실전 가중 B72다.

## 12. 이후 분석·코드 평가 시 필수 구분

1. 현 활성·이전·Fallback·5개 쉐도우를 모델 SHA로 구분한다. 같은 B72 이름만으로 동일 모델이라 판단하지 않는다.
2. OBSERVED / REPLAY / INFERRED / UNVERIFIED를 표시한다. raw 확률·표시 점수·net EV를 섞지 않는다.
3. 판단 횟수, 방향 신호, entry intent, 접수, 체결, 완료 거래를 따로 집계한다. primary와 forced도 분리한다.
4. 로그 누락·입력 지연·reject·unresolved·강제 종료를 손익 계산에서 조용히 제거하지 않는다. sample coverage/제외 사유를 함께 보고한다.
5. 라벨 방향, 학습 cutoff와 target 확정 시각, 비용, 동일 cycle 중복 여부, 정확한 feature 순서/해시를 확인한다.
6. 실전 vs 가상 체결 차이, 재학습 가중 목표의 차이, OOF vs IN_SAMPLE을 구분한다. 표본 17개가 전부 SOXL이라는 제약을 유지한다.
7. 설정/주문/활성 모델을 분석 도중 자동 변경하지 않는다. 기존 사용자 승인 범위를 확인한다. 새 모델 활성화는 기존 전환·롤백 체계를 유지한다.
8. B-03 해결 증거와 장중 feed 검증 없이는 READY를 선언하지 않는다. 성능 표가 좋아도 이 판단은 바뀌지 않는다.
9. 구현이 바뀌면 본 문서와 SHA 스냅샷을 함께 갱신하고 이전 평가의 provenance를 보존한다. 사용자에게는 기본적으로 대화창에 결과를 전달한다.
