# 꿈틀 인수인계
마지막 갱신: 2026-11-26 07:3x ET · 밤 3차 — 사회 학습 판정 나 두 씨앗 지지·가르침 관문 None 고침(시험 597)
요약: 세계 시뮬레이션 꿈틀 — 아무도 가르치지 않았는데 생물이 스스로 행동·전략·신호를 만드는지 본다(공개 레포, 목적·규칙은 CLAUDE.md). 11/26 새벽 실험 재개. **작업 가지 `night/2026-11-26-kkumteul`, 작업 폴더 `.claude/worktrees/night-2026-11-26`(main에 안 합침, 원격에 push함).** 사용자 `/goal`: 목요일(11/26) 오전 8시 전까지 계속, 묻지 말고 끝까지, ultracode 한시 허용, 메모리·CPU 뻑나지 않게.

## 다음 세션이 이어서 할 것
0. 사용자 최근 말(원문, 자동)
- (11/26 02:38) "꿈틀 지금 작업할거 많아?" → 처리됨(남은 일 답함)
- (11/26 02:39) "지금 구독량 완전 쭉 쓸 수 있어서(알다시피 오전 8시까지) 할만한 작업들 지금 쭉 돌려도 되긴 해 ultracode도 한시 허용할게. overnight 느낌으로다가. 곧 내가 잘거라서. 그리고 /develop 창의성 << 이거 해도 되고. 대신 나한테 물어볼거나 뭐 막히는거 없이 완전 쭉 진행해줘야해. 다만 메모리나 뭐 cpu 뻑나거나 그런것만 없도록 반드시 지속해서 진행해주던가" → 거의 처리(1·3·4번 남음)
- (11/26 02:42) "[끼어든 말] 다 했는데 오전 8시까지(지금 5시간 20분정도 남았네) 만약 시간이 더 남았으면 계속 develop이나 우리의 목표(철학, 규율과 법, 시뮬레이션 인식)정도까지 계속 나아가줘 develop도 마찬가지고" → 아직(3·4번)
1. **사회 학습 체인 계속 읽기** — 유닛 `kkumteul-soc`(코드 사본 `runs/code-faf9852`, 기록 `runs/soc-exp/`). **판정 나 두 씨앗 끝, 지지**(`results.md` "판정 나 결과, 씨앗 a·b", status 갱신): `outcome` +0.21·+0.24(6/6), `blind` 음, 음성 대조 통과, 보고 배우기 속도 켬 0.55·0.69 / 불활성 0.19·0.16. 개인 배움 속도 하락은 불활성에도 있어 로저스 역설로 못 읽음(07:1x 중간 해석 거둠). 체인은 07:31 ET부터 씨앗 c 갈래 → 판정 → 시작 속도 0.2 갈래(몇 시간 더). 끝나면 씨앗 c·0.2 결과를 같은 절 아래에 덧붙인다.
2. **가르침 체인 준비** — 구현 끝, 예측 절 커밋돼 있음, `teach_gate` None 고침 끝(`_ratio`가 None·nan이면 nan → 그 판 닫힘, 시험 5건 추가, 전체 597 통과). 남은 것: 사회 학습 체인이 끝난 뒤 `wayfinder/stage2-4/teach_chain.sh`를 **새 코드 사본**(지금 가지 머리)으로 돌림(한 번에 하나, 1.5GB·2스레드). 체인 띄우기는 서버 실행 시작이니 사용자 깨어 있으면 한 줄 보고 후.
3. **규범의 씨앗 구현** — 계획 둘째 판 `f0a1cca`(`docs/superpowers/plans/2026-11-26-norms-seed.md`, 과제 0~19, sonnet 검토 22건 반영). 가르침과 같은 꼴 워크플로(과제마다 TDD 구현 → sonnet 검토 → 고치기, 같은 폴더에 구현 워크플로 둘 동시 금지). 열린 물음은 결정 대기 둘째 줄.
4. **develop 창의성 다음 실측** — 2바퀴 기록 `sessions/develop/kkumteul-communication.md` "2바퀴"(`6403420`): 여섯 관점 방안·겹친 곳·실측 순서 2~8, 실측 1(용량 반응 추세: a 단조 +, b 뒤집힌 U, blind 강한 음)은 함. 다음: 실측 2(`saweat` 막기 겨루기)·3(방언 지도, 읽기 전용)을 사회 학습 체인 뒤에, 4(쌍둥이·심은 효과 눈금자, `measure.py` TDD). 5단계(말하는 뇌, `docs/stage5-proposal.md`) 시험판 검토는 아직.
4b. 정렬 고침(`152357b`) 2축 검토 남은 것: 묶음 → `measure.compete` 전체를 지나는 시험이 없음(지금 시험은 cfg 값·가짜 assay만), `hand_learner.curve`·`solo`가 `cfg_good_kind`를 쓰는지 시험 없음 → 작은 세계로 하나씩 추가. `taste_judge.py`는 아레나가 아니라 원래 세계에 씨앗을 덮어쓴 세계라 일부러 안 고침(지시와 다르게 한 것).
5. 8시 무렵: 아침 브리핑용으로 status·인수인계 정리(무엇을 재었고 무엇이 돌고 있는지).
5b. 주의: `soc_chain.sh`를 도는 체인과 같은 기록 폴더로 DRY 돌리지 말 것 — 레포 판은 `5dda27e`로 고쳤지만 도는 코드 사본(`runs/code-faf9852`)은 옛 판이라 DRY가 갈래 로그를 자른다.
6. 이전 계획(11/09 이전, 이제 위 순서 뒤): 11/06 판 "다음 할 일"(맛 기억 다음·사냥·2부-다·암수) → `grep -n '2026-11-09 대청소' archive/kkumteul-history.md`. 일하는 방식·서버 세계 재개 명령 → `grep -n '^## 일하는 방식\|^## 현재 상태' archive/kkumteul-history.md`.
7. 11/06 판 "먼저 할 것"(다시 보기 새 파일 올리기 — `runs/replay-two.html` 14MB라 통째 읽기 불가로 이번 밤엔 안 올림) — 상태 미확인.

## 결정 대기 (사용자)
- 밤샘 가지들(`night/2026-11-26-kkumteul`이 07b·맛 기억을 모두 포함)을 main에 합칠지 — 합치기 한 줄: `cd ~/git/kkumteul && bash scripts/merge-into-main.sh night/2026-11-26-kkumteul && echo 저장됨`
- 공격이 죄수의 딜레마 구조(친 쪽 +0.08, 맞은 쪽 −0.2)라 벌 없는 관습이 설 틈이 없다 — 매-비둘기로 바꾸는 `fight_extra`를 규범 판정 전에 넣을지(develop 2바퀴 기록 "사람 결정 대기")
- 상주 세션(announcement·insight·research)을 tmux로 띄울지, 세션 안 예약 작업을 둘지(앱 백그라운드 세션은 8시간 입력 없으면 자동 정리 — 메모리 `bg-sessions-retire-after-8h`)

## 막다른 길
- 고정 갈래 겨루기 "양쪽 전멸"을 볼드윈 증거로 읽기 → 마당이 독 순서를 새 씨앗으로 뽑던 측정 버그였음(`53f1000`로 고침, 옛 판정 15,000 +2.28도 부풀었던 값).
- 사회 학습 체인 계획 그대로(판정 0 22판+갈래 12) → 서버 부하로 8시 안에 못 끝나 핵심 넷 먼저로 순서 바꿈(`faf9852`).

## 시작 때 확인
- `systemctl --user is-active kkumteul-soc; tail -5 ~/git/kkumteul/runs/soc-exp/progress.txt` — 실험 체인 진행
- `git -C ~/git/kkumteul/.claude/worktrees/night-2026-11-26 log --oneline -5` — 이 파일 뒤 새 커밋
- `free -m | sed -n 2p; uptime` — 서버가 바쁨: 시뮬레이션 한 번에 하나(1.5GB·2스레드), 시험 2GB, 에이전트 동시 2~3
- 끝난 것: 가르침 관문 None 고침·판정 나 중간 결과(밤 3차 커밋)
- 끝난 것: 부모의 가르침 구현 과제 1~9 `40aa090`…`2030141`(전체 시험 592) · 정렬 지적 반영 `152357b`(1b) · 사회 학습 판정 0·관문 `5949cd7`
- 끝난 것: 규범 계획 `f0a1cca` · develop 2바퀴 `6403420` · 체인 DRY 로그 `5dda27e`
- 끝난 것: 사회 학습 구현 1~7 `874d7e3`…`6c244c0` · 가르침 설계 `2b45163` · 규범 설계 `25893f8` · 독 순서 고침 `53f1000`

## 교훈 (상황 → 찾을 곳)  — 교훈 파일 `archive/kkumteul-history.md`
- 공개 레포 문서·인수인계에 경로·다른 프로젝트 이야기를 적을 때 → grep -n '공개 레포 문서' archive/kkumteul-history.md
- 설계·계획에 목표 비율·"몇 %"·"지금 형식에 없으면"을 적을 때 → grep -n '목표 비율\|미루지 않는다' archive/kkumteul-history.md
- 장부 시험·측정 코드를 쓰거나 고칠 때, 비율 하나로 행동을 읽을 때 → grep -n '장부 시험\|뻔히 지는\|물질 수지' archive/kkumteul-history.md
- 협동·신호·감각 장치를 겨룰 때(섞인 마당은 무리 이득을 0으로 만든다 — develop 2바퀴) → grep -n '협동이 필요\|입력을 0으로\|섞인 겨루기\|겨루기 승리만' archive/kkumteul-history.md
- 겨루기 결과가 "양쪽 전멸"로 깨질 때 → 마당 설정이 원래 세계와 어긋났는지 먼저 — `grep -n '고정 갈래 다시 재기' wayfinder/stage2-4/results.md`
- 오래 도는 실험을 시작할 때(코드 사본·/tmp·승인 창) → grep -n 'runs/code-\|메모리 디스크\|승인 창이 뜰 수' archive/kkumteul-history.md
- 워크플로로 설계·구현할 때(잘 된 꼴: 초안 → sonnet 검토 → 둘째 판 / 과제마다 구현·검토·고치기, 한 과제 약 15~20분, 에이전트에게 시뮬레이션 안 시킴) → 이 줄
- 문서에 시각을 적을 때, 합치기 명령을 드릴 때 → grep -n 'date +%H:%M\|merge-into-main' archive/kkumteul-history.md
- 메모리·CLAUDE.md로 올린 것: numba prange·rng 용도 번호·에너지 장부·커밋하면 push(CLAUDE.md), `bg-sessions-retire-after-8h`·`phone-notify-script`·`background-sessions-work-on-a-branch`(메모리)
