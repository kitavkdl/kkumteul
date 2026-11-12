# 꿈틀 인수인계
마지막 갱신: 2026-11-12 12:21 ET · 새 양식 전환(내용은 11/09 02:5x ET 실험 중지 판 그대로)
요약: 세계 시뮬레이션 꿈틀 — 아무도 가르치지 않았는데 생물이 스스로 행동·전략·신호를 만드는지 본다(공개 레포, 목적·규칙은 CLAUDE.md). **지금은 실험 중지** — 사용자 11/09 02:4x ET "일단 실험 중지다". 11/06 판 인수인계·교훈 전문은 `archive/kkumteul-history.md`.

## 다음 세션이 이어서 할 것
0. 사용자 최근 말(원문, 자동)
- (11/12 12:16) "(개인 내용 — 생략)" → 처리됨(꿈틀 인수인계 새 양식 전환)
1. ⛔ **실험 중지 유지** — 사용자가 다시 시작하라고 하실 때까지 실험·밤샘 체인·상주 세션(announcement·insight·research)을 띄우지 않는다.
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
