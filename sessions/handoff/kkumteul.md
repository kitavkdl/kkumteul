# 꿈틀 인수인계
마지막 갱신: 2026-11-26 03:0x ET · 밤 작업 재개(사용자 11/26 02:39 ET)
요약: 세계 시뮬레이션 꿈틀 — 아무도 가르치지 않았는데 생물이 스스로 행동·전략·신호를 만드는지 본다(공개 레포, 목적·규칙은 CLAUDE.md). **11/26 새벽 실험 재개** — 사용자 "오전 8시까지 할 만한 작업 쭉, 묻지 말고 끝까지, ultracode·/develop 창의성 허용, 메모리·CPU 뻑나지 않게", 이어서 "시간 남으면 develop이나 우리의 목표(철학, 규율과 법, 시뮬레이션 인식)까지". `/goal`: 목요일(11/26) 오전 8시 전까지 계속.

## ★ 11/26 밤 작업 (가지 `night/2026-11-26-kkumteul`, 작업 폴더 `.claude/worktrees/night-2026-11-26`, main에 안 합침)
- 가지는 `night/2026-11-07b-kkumteul-learn`(맛 기억·개인 학습 포함) 위에 main을 합친 것. 원격에 push함.
- 고침 `53f1000`: 겨루기 마당이 독 순서를 새 씨앗으로 다시 뽑던 버그 → `kind_seed`·`kind_step0`·`measure.kind_aligned`. **다시 재기 끝**(`3d62b5d`·`a41aa98`, results.md 끝 두 절, status 반영): 고정 a·b 배움 0 벌 가를 수 없음(전멸 0) → 볼드윈 부분 지지, 15,000은 +0.30(옛 +2.28은 버그로 부풀음), 500은 +1.03 그대로.
- **사회 학습 구현 끝**(과제 1~7 `874d7e3`…`6c244c0`, 전체 시험 472 통과, 예측은 results.md "사회 학습 — 예측"에 실행 전 기록). **실험 체인 돌고 있음**: 유닛 `kkumteul-soc`(04:5x ET 시작, 코드 사본 `runs/code-faf9852`, 기록 `runs/soc-exp/`, 로그 `runs/soc-exp/chain.log`). 순서: 판정 0 핵심 넷(outcome-d0.05·d1.0, blind-d1.0, off-d1.0)을 바탕 a·b 먼저 → 나머지 판정 0 → 관문(`gate.json`, 0 통과·3 검출 불능·4 도구 깨짐) → 통과면 진화 갈래·판정. 한 판 약 10~15분(서버 부하에 따라). **할 일: 핵심 여덟 판이 끝나면 바로 읽어(`taste_gene_judge.py gate`를 있는 판으로 돌리거나 j-dose-*.json 직접) results.md·status.md에 중간 판정**, 8시 전에 끝난 만큼 적는다.
- 부모의 가르침(3단계): 설계 둘째 판·계획 `2b45163`(`docs/superpowers/specs/2026-11-26-parent-teaching-design.md`, `plans/2026-11-26-parent-teaching.md`, 과제 0~9) — **다음: 구현 워크플로**(사회 학습 구현과 같은 꼴: 과제마다 TDD 구현 → sonnet 검토 → 고치기, 시험은 memcap 2G·2스레드, 한 번에 하나). 실험은 사회 학습 체인이 끝난 뒤.
- 규범의 씨앗: 설계 둘째 판 `25893f8`(`docs/superpowers/specs/2026-11-26-norms-seed-design.md`, 계획은 아직) — 가르침 구현 뒤 계획.
- 그다음 후보: `/develop 창의성`, 5단계(말하는 뇌, `docs/stage5-proposal.md`) 작은 시험판 — 유료 API는 쓰지 않음.
- 워크플로 쓰는 법(이번 밤 잘 된 꼴): 설계 = 초안 → 규칙·명세 축 sonnet 검토 둘 → 둘째 판+계획 커밋, 구현 = 과제마다 구현(opus)·검토(sonnet)·고치기. 에이전트에게 시뮬레이션은 시키지 않는다. 같은 폴더에서 동시에 두 구현 워크플로를 돌리지 않는다.
- 사용자 `/goal`: 목요일(11/26) 오전 8시 전까지 계속 작업. 8시가 되면 실험 체인은 그대로 두고(멈추지 않음) 아침 브리핑용으로 인수인계·status를 정리한다.
- 서버가 다른 일로 매우 바쁨(부하 25~30, 남은 메모리 5GB 안팎) — 시뮬레이션 한 번에 하나(1.5GB·2스레드), 시험 2GB, 에이전트 동시 2~3.

## 다음 세션이 이어서 할 것
0. 사용자 최근 말(원문, 자동)
- (11/26 02:39) "오전 8시까지 … 쭉 돌려도 되긴 해 ultracode도 한시 허용 … /develop 창의성 해도 되고 … 물어볼거나 막히는거 없이 완전 쭉 … 메모리나 cpu 뻑나거나 그런것만 없도록" → 진행 중(위 절)
1. (11/09~11/26 새벽까지) 실험 중지였음 — 11/26 새벽 사용자 지시로 재개. 상주 세션(announcement·insight·research)은 여전히 띄우지 않음.
   - 끈 것: 앱의 꿈틀 세션 셋 보관, RC 대기 창 `rc-kkumteul` 종료, 남아 있던 로그 보기 프로세스 종료. 서버 세계 둘(`kkumteul-main`·`kkumteul-world4`)은 11/09 이전에 이미 멈춤. 밤샘 체인 2026-11-07·07b는 둘 다 `stopped`.
   - 그대로 둔 것(지우지 않음): 밤샘 가지와 작업 폴더(`.claude/worktrees/night-*`), `runs/`, 아티팩트.
2. 마지막 결과(밤샘 가지 `night/2026-11-07b-kkumteul-learn`, **main에 안 합침**): 신호 판정 `e9ba9cb`(world4 경계, atk-a 쓰지 않음), 개인 학습 `a118582`(주기 500 쓴다 — 씨앗 둘은 500만, 15,000은 5/6 전멸; 독 비율 0.43~0.46으로 이긴 까닭 미확인; 고정 갈래는 겨루기가 깨져 볼드윈 보류). 정리 아티팩트 https://claude.ai/artifact/E1vteozVx5B4Tvwf5JUweg
3. 재개하라고 하시면 먼저: ① 위 가지를 합칠지 사용자에게 ② announcement가 11/07 11:17 ET에 꺼진 원인 — wrap-spawn이 띄운 백그라운드 데몬 세션은 8시간 입력이 없으면 Claude Code가 자동 정리한다(`~/.claude/daemon.log`의 `bg retire … idle-prompt, idle 8h`). 상주 세션은 tmux로 띄우거나 세션 안 예약 작업을 둬야 한다 — 사용자 결정 대기.
4. 재개 때 11/06 판 "다음 할 일"(맛 기억 다음·사냥·2부-다·암수 큰 순서)과 0번 스크립트 위치: `grep -n '2026-11-09 대청소' archive/kkumteul-history.md`. 11/06 판의 현재 상태·일하는 방식(서버 세계 재개 명령, 시험 명령, 메모리 상한 1.5GB·2스레드, 검토 둘 규칙, SDD 모델)은 `grep -n '^## 일하는 방식\|^## 현재 상태' archive/kkumteul-history.md`(2026-11-12 새 양식 전환 때 옮김 블록 안).
5. 11/06 판의 "먼저 할 것"(다시 보기 아티팩트 새 파일 올리기, 맛 기억 가지 `night/2026-11-06-kkumteul-taste` main 합치기)은 지금 상태 미확인 — "시작 때 확인"의 가지 목록으로 본다.

## 결정 대기 (사용자)
- 실험을 언제 다시 시작할지(사용자 말씀 전엔 아무것도 안 띄움)
- 밤샘 가지 `night/2026-11-07b-kkumteul-learn`(와 맛 기억 가지)를 main에 합칠지
- 상주 세션을 tmux로 띄울지, 세션 안 예약 작업을 둘지(8시간 자동 정리 — 메모리 `bg-sessions-retire-after-8h`)

## 시작 때 확인
- `git log --oneline -5` — 이 파일 뒤 새 커밋이 있는지
- `git branch --no-merged main` — main에 안 합친 가지(밤샘 `night/…`·맛 기억·`research`)
- `tmux ls 2>/dev/null | grep -i kkumteul || echo '(없음)'` — 꿈틀 세션이 떠 있지 않은 것이 정상(중지 중)
- 끝난 것: 다시 보기 둘째 판 화면 `f02619a` · 맛 기억 판정 가~라 결과 문서 `6cb6085`(맛 기억 가지)
- 끝난 것: 밤샘 가지 신호 판정 `e9ba9cb`·개인 학습 `a118582`(main 미합침) · 상주 세션 구조(에이전트 파일) `fb33047`·`2db2aa4`
- 끝난 것: 실험 중지 기록 `1542aa4` · 인수인계 대청소 `a2d98db`

## 교훈 (상황 → 찾을 곳)  — 교훈 파일 `archive/kkumteul-history.md`
- 공개 레포 문서·인수인계에 경로·다른 프로젝트 이야기를 적을 때 → grep -n '공개 레포 문서' archive/kkumteul-history.md
- 설계·계획에 목표 비율·"몇 %"·"지금 형식에 없으면"을 적을 때 → grep -n '목표 비율\|미루지 않는다' archive/kkumteul-history.md
- 장부 시험·측정 코드를 쓰거나 고칠 때, 비율 하나로 행동을 읽을 때 → grep -n '장부 시험\|뻔히 지는\|물질 수지' archive/kkumteul-history.md
- 협동·신호·감각 장치를 겨룰 때(문턱, 입력 0 막기, 섞인 겨루기, 겨루기 승리) → grep -n '협동이 필요\|입력을 0으로\|섞인 겨루기\|겨루기 승리만' archive/kkumteul-history.md
- 움직임을 "반응"으로 읽기 전 → grep -n '0번 대조' archive/kkumteul-history.md
- 오래 도는 실험을 시작할 때(코드 사본·/tmp·승인 창) → grep -n 'runs/code-\|메모리 디스크\|승인 창이 뜰 수' archive/kkumteul-history.md
- 문서에 시각을 적을 때, 사용자에게 git 합치기 명령을 드릴 때 → grep -n 'date +%H:%M\|merge-into-main' archive/kkumteul-history.md
- 알림 턴에 교대할 때 → grep -n '알림 턴에 할 일' archive/kkumteul-history.md
- 메모리·CLAUDE.md로 올린 것: numba prange·rng 용도 번호·에너지 장부·커밋하면 push(CLAUDE.md), `bg-sessions-retire-after-8h`·`phone-notify-script`·`background-sessions-work-on-a-branch`(메모리)
