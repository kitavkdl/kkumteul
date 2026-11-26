# 꿈틀 인수인계
마지막 갱신: 2026-11-26 04:5x ET · 밤 작업 재개 — 독 순서 버그 고침·사회 학습 구현 끝·실험 체인 돌림·가르침·규범 설계
요약: 세계 시뮬레이션 꿈틀 — 아무도 가르치지 않았는데 생물이 스스로 행동·전략·신호를 만드는지 본다(공개 레포, 목적·규칙은 CLAUDE.md). 11/26 새벽 실험 재개. **작업 가지 `night/2026-11-26-kkumteul`, 작업 폴더 `.claude/worktrees/night-2026-11-26`(main에 안 합침, 원격에 push함).** 사용자 `/goal`: 목요일(11/26) 오전 8시 전까지 계속, 묻지 말고 끝까지, ultracode 한시 허용, 메모리·CPU 뻑나지 않게.

## 다음 세션이 이어서 할 것
0. 사용자 최근 말(원문, 자동)
- (11/26 02:38) "꿈틀 지금 작업할거 많아?" → 처리됨(남은 일 답함)
- (11/26 02:39) "지금 구독량 완전 쭉 쓸 수 있어서(알다시피 오전 8시까지) 할만한 작업들 지금 쭉 돌려도 되긴 해 ultracode도 한시 허용할게. overnight 느낌으로다가. 곧 내가 잘거라서. 그리고 /develop 창의성 << 이거 해도 되고. 대신 나한테 물어볼거나 뭐 막히는거 없이 완전 쭉 진행해줘야해. 다만 메모리나 뭐 cpu 뻑나거나 그런것만 없도록 반드시 지속해서 진행해주던가" → 아직(1~4번)
- (11/26 02:42) "[끼어든 말] 다 했는데 오전 8시까지(지금 5시간 20분정도 남았네) 만약 시간이 더 남았으면 계속 develop이나 우리의 목표(철학, 규율과 법, 시뮬레이션 인식)정도까지 계속 나아가줘 develop도 마찬가지고" → 아직(3·4번)
1. **사회 학습 실험 체인 읽기** — 유닛 `kkumteul-soc`(04:47 ET 시작, 코드 사본 `runs/code-faf9852`, 기록 `runs/soc-exp/`, 로그 `runs/soc-exp/chain.log`). 순서: 판정 0 핵심 넷(outcome-d0.05·d1.0, blind-d1.0, off-d1.0) 바탕 a·b → 나머지 판정 0 → 관문(`gate.json`: 0 통과·3 검출 불능·4 도구 깨짐) → 통과면 진화 갈래·판정. 한 판 10~15분. 핵심 여덟 판이 끝나면 바로 읽어 `wayfinder/stage2-4/results.md`(예측 절 "사회 학습 — 예측"이 실행 전에 있음)·`docs/status.md`에 중간 판정. 완료 기준: 8시 전에 끝난 만큼 판정이 문서에 있음. 체인은 8시에도 멈추지 않는다.
1b. **2축 검토 지적(규칙 축, 반영 대기)**: `wayfinder/stage2-4/hand_learner.py`(166·239·304줄 근처)·`taste_judge.py`·`hunt_hand.py`가 `k.good_kind(acfg.seed, …)`로 독을 직접 센다 — `measure.kind_aligned`를 거친 마당에선 실제 독과 어긋남 → `k.cfg_good_kind(cfg, step)`으로 바꾸고 시험(지금 도는 체인은 이 도구를 안 씀). **명세 축(치명, 반영 대기)**: 정렬은 `measure.side_from_world`에만 있다 — `pack.load_side`(묶음 겨루기, 스텝 정보 없음)와 `experiment._assay`(`a["cfg"] = b["cfg"] = base`로 덮어씀)는 옛 마당 그대로 → 둘에도 정렬을 넣고, `load_side`·`compete` 전체를 지나는 시험 추가. 사회 학습 체인은 `taste_gene_judge rate-zero` → `measure.load_side` 경로라 정렬됨(영향 없음).
2. **부모의 가르침 구현** — 설계 둘째 판·계획 `2b45163`(`docs/superpowers/specs/2026-11-26-parent-teaching-design.md`, `docs/superpowers/plans/2026-11-26-parent-teaching.md`, 과제 0~9). 사회 학습 구현과 같은 꼴의 워크플로(과제마다 TDD 구현 → sonnet 검토 → 고치기). 실험은 사회 학습 체인 뒤.
3. **규범의 씨앗** — 설계 둘째 판 `25893f8`(`docs/superpowers/specs/2026-11-26-norms-seed-design.md`). 다음: 구현 계획(가르침 구현 뒤).
4. `/develop 창의성` 한 바퀴, 5단계(말하는 뇌, `docs/stage5-proposal.md`) 작은 시험판 검토 — 유료 API는 쓰지 않음.
5. 8시 무렵: 아침 브리핑용으로 status·인수인계 정리(무엇을 재었고 무엇이 돌고 있는지).
6. 이전 계획(11/09 이전, 이제 위 순서 뒤): 11/06 판 "다음 할 일"(맛 기억 다음·사냥·2부-다·암수) → `grep -n '2026-11-09 대청소' archive/kkumteul-history.md`. 일하는 방식·서버 세계 재개 명령 → `grep -n '^## 일하는 방식\|^## 현재 상태' archive/kkumteul-history.md`.
7. 11/06 판 "먼저 할 것"(다시 보기 새 파일 올리기 — `runs/replay-two.html` 14MB라 통째 읽기 불가로 이번 밤엔 안 올림) — 상태 미확인.

## 결정 대기 (사용자)
- 밤샘 가지들(`night/2026-11-26-kkumteul`이 07b·맛 기억을 모두 포함)을 main에 합칠지 — 합치기 한 줄: `cd ~/git/kkumteul && bash scripts/merge-into-main.sh night/2026-11-26-kkumteul && echo 저장됨`
- 상주 세션(announcement·insight·research)을 tmux로 띄울지, 세션 안 예약 작업을 둘지(앱 백그라운드 세션은 8시간 입력 없으면 자동 정리 — 메모리 `bg-sessions-retire-after-8h`)

## 막다른 길
- 고정 갈래 겨루기 "양쪽 전멸"을 볼드윈 증거로 읽기 → 마당이 독 순서를 새 씨앗으로 뽑던 측정 버그였음(`53f1000`로 고침, 옛 판정 15,000 +2.28도 부풀었던 값).
- 사회 학습 체인 계획 그대로(판정 0 22판+갈래 12) → 서버 부하로 8시 안에 못 끝나 핵심 넷 먼저로 순서 바꿈(`faf9852`).

## 시작 때 확인
- `systemctl --user is-active kkumteul-soc; tail -5 ~/git/kkumteul/runs/soc-exp/chain.log` — 실험 체인 진행
- `git -C ~/git/kkumteul/.claude/worktrees/night-2026-11-26 log --oneline -5` — 이 파일 뒤 새 커밋
- `free -m | sed -n 2p; uptime` — 서버가 다른 일로 매우 바쁨(부하 25~38, 남은 메모리 4~6GB): 시뮬레이션 한 번에 하나(1.5GB·2스레드), 시험 2GB, 에이전트 동시 2~3
- 끝난 것: 독 순서 고침·다시 재기 `53f1000`·`3d62b5d`·`a41aa98`(고정 배움 0 가를 수 없음 씨앗 둘, 15,000 +0.30, 500 +1.03)
- 끝난 것: 사회 학습 설계·계획 `4fa2462`, 구현 과제 1~7 `874d7e3`…`6c244c0`(전체 시험 472 통과)
- 끝난 것: 실험 재개(11/09 중지 → 11/26 사용자 지시) · 가르침 설계 `2b45163` · 규범 설계 `25893f8`

## 교훈 (상황 → 찾을 곳)  — 교훈 파일 `archive/kkumteul-history.md`
- 공개 레포 문서·인수인계에 경로·다른 프로젝트 이야기를 적을 때 → grep -n '공개 레포 문서' archive/kkumteul-history.md
- 설계·계획에 목표 비율·"몇 %"·"지금 형식에 없으면"을 적을 때 → grep -n '목표 비율\|미루지 않는다' archive/kkumteul-history.md
- 장부 시험·측정 코드를 쓰거나 고칠 때, 비율 하나로 행동을 읽을 때 → grep -n '장부 시험\|뻔히 지는\|물질 수지' archive/kkumteul-history.md
- 협동·신호·감각 장치를 겨룰 때 → grep -n '협동이 필요\|입력을 0으로\|섞인 겨루기\|겨루기 승리만' archive/kkumteul-history.md
- 겨루기 결과가 "양쪽 전멸"로 깨질 때 → 마당 설정이 원래 세계와 어긋났는지 먼저(이번 밤 독 순서) — `grep -n '고정 갈래 다시 재기' wayfinder/stage2-4/results.md`
- 오래 도는 실험을 시작할 때(코드 사본·/tmp·승인 창) → grep -n 'runs/code-\|메모리 디스크\|승인 창이 뜰 수' archive/kkumteul-history.md
- 워크플로로 설계·구현할 때(이번 밤 잘 된 꼴: 초안 → 규칙·명세 sonnet 검토 → 둘째 판+계획 / 과제마다 구현·검토·고치기, 같은 폴더에 구현 워크플로 둘 동시 금지, 에이전트에게 시뮬레이션 안 시킴) → 이 줄
- 문서에 시각을 적을 때, 합치기 명령을 드릴 때 → grep -n 'date +%H:%M\|merge-into-main' archive/kkumteul-history.md
- 메모리·CLAUDE.md로 올린 것: numba prange·rng 용도 번호·에너지 장부·커밋하면 push(CLAUDE.md), `bg-sessions-retire-after-8h`·`phone-notify-script`·`background-sessions-work-on-a-branch`(메모리)
