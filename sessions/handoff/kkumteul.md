# 꿈틀 인수인계
마지막 갱신: 2026-11-26 17:4x ET · 낮 — 사회 학습 끝(다섯 갈래 지지)·가르침 관문 −1 곁 부족으로 닫힘·속도 상한 0.1 점검 도는 중·규범 구현 과제 0~6 끝(검토 지적 4건 남음)
요약: 세계 시뮬레이션 꿈틀 — 아무도 가르치지 않았는데 생물이 스스로 행동·전략·신호를 만드는지 본다(공개 레포, 목적·규칙은 CLAUDE.md). 11/26 새벽 실험 재개. **작업 가지 `night/2026-11-26-kkumteul`, 작업 폴더 `.claude/worktrees/night-2026-11-26`(main에 안 합침, 원격에 push함).** 사용자 16:26 "싹다 진행해 쭉" — 묻지 말고 끝까지, 서버 메모리·CPU 안 터지게.

## 다음 세션이 이어서 할 것
0. 사용자 최근 말(원문, 자동)
- (11/26 16:21) "아니 너 얼마나 뭐 한거야 어느정도 한거고 남아있는게 어디고 어느 단계까지 왔어?" → 처리됨(단계 표로 답함)
- (11/26 16:26) "ㅇㅋ 싹다 진행해 쭉." → 진행 중(사회 학습 정리 끝, 가르침 2번, 규범 3번)
- (11/26 16:27) "[끼어든 말] 그리고 이건 그냥 나의 한 아이디어인데, 사람이 다른 동물들이랑 다른 이유가 2족 보행을 한 것이주요 원인 중 하나라는 것을 보았는데 이것처럼 지금 이거를 바로 구현할 수는 없더라도 다른 포(피)식자나 기타 동물들(지금 세계에 있는) 애들은 전부 모든 다리를 보행하는 데에만 사용하고 우리가 focus하는 그 애들은 두다리는 보행, 두팔은 뭔가를 사용할 수 있도록 하는 것도 괜찮은 아이디어 아닐까" → 처리됨(`sessions/tasks.md` "아이디어 후보", 가르침·규범·섬 뒤)
- (11/26 16:33) "뭘 얻고 뭘 잃는건데? 매 비룯기 구조는 뭐고? 비교하는건 필요할 것 같기는 한데" → 답함(친 쪽 +0.08·맞은 쪽 −0.2·맞공격 −0.12 = 죄수의 딜레마), 3번
- (11/26 16:34) "[끼어든 말] ㅇㅇ 체크포인트에서 비교해서 진행하면 될듯" → 확정: 규범 구현 뒤 `fight_extra` 추가, 규범 판정을 지금 구조·매-비둘기 둘로(3번)
1. **사회 학습 — 끝.** 판정 나 다섯 갈래(씨앗 a·b·c, 시작 속도 0.2 a·b) 모두 지지(`results.md` 두 절, status 갱신). 개인 배움 속도 하락은 불활성에도 있어 로저스 역설로 못 읽음.
2. **가르침 — 관문 −1에서 닫힘(16:46 ET, `results.md` "관문 −1 결과")**: 새끼당 기대 부모 독 끼니 관찰 0.353·0.442 < 0.5(곁 부족), −1c도 무리 미달. 속도 상한은 `par_near_frac`를 +0.004·+0.018밖에 못 올림. **도는 것**: 설계 6절 다음 판 `juv_move_floor=0.1` 세계 점검(유닛 `kkumteul-teach-floor`; 바탕 260만에서 3만 스텝 branch `--set preset=taste taste_inherit=0 taste_gene_enabled=true kind_flip_period=500 juv_enabled=true juv_move_floor=0.1` → `taste_gene_judge.py observe` 5,000스텝, 결과 `runs/teach-exp/w-on01-{a,b}.json`, 끝나면 `progress.txt`에 "on01 끝"). 다음: 기대 관찰 = par_near_frac × 200 × 끼니율 × 독 비율을 셈해 결과 절에 덧붙임. 두 바탕 ≥ 0.5면 설계·예측을 고쳐 쓰고 `teach_chain.sh`를 그 판으로 다시(새 코드 사본), 못 넘으면 "부모 방향 입력 + 따라가기 출력" 판을 설계 셋째 판으로 → 구현. 곁에 안 머무는 채로 "가르침이 안 쓰인다"고 쓰지 않는다.
3. **규범의 씨앗 구현 — 과제 0~6 끝**(`1c65d07`…`4292336`, 전체 시험 648 통과). sonnet 검토(치명 0) 지적을 **다음 묶음 구현자가 먼저 고친다**(전부 반영): ① [중요] 만료 스텝 재표시 때 앞 에피소드가 안 세어짐(`stages.py` `_attach_mark` ~669) → 과제 9 `norm_account`에서 닫기 + "만료 스텝 재표시" 시험 ② [중요] `NT_TAKE_KIN`이 "계보 또는 닮음 ≥0.92"(`stages.py` ~880; 계획은 계보만, 설계 176행은 닮음 문턱을 사후에) → 계보만, `tests/test_norm.py` ~450도(과제 8 `pun_role` 1은 계획대로 닮음 포함) ③ [사소] 커널에서 벌금 `got = min(got, dmg*attack_gain)` ④ [사소] `test_scan_not_called_when_both_off`에 njit이라 호출을 못 센다는 주석. 그다음 과제 7~12 → sonnet 검토 → 13~18 → 검토 → **추가 과제 `fight_extra`**(맞공격 때 추가 손실, 기본 꺼짐) → 규범 체인 두 판. 구현은 서브에이전트(opus) 묶음 하나씩(서버 메모리), 브리핑: 계획 절·공통 규칙·worktree 한 줄 명령·`memcap --max 2G` 시험·과제마다 커밋 push.
4. **develop 창의성 다음 실측** — `sessions/develop/kkumteul-communication.md` "2바퀴"(`6403420`) 실측 순서 2~8: 다음은 2(`saweat` 막기 겨루기)·3(방언 지도, 읽기 전용)·4(쌍둥이·심은 효과 눈금자, `measure.py` TDD). 5단계(`docs/stage5-proposal.md`) 시험판 검토는 아직.
5b. 주의: 코드 사본 `runs/code-faf9852`의 `soc_chain.sh`는 옛 판이라 같은 기록 폴더로 DRY 돌리면 갈래 로그를 자른다(레포 판은 `5dda27e`로 고침).
6. 이전 계획(11/09 이전, 위 순서 뒤): 11/06 판 "다음 할 일"(맛 기억 다음·사냥·2부-다·암수) → `grep -n '2026-11-09 대청소' archive/kkumteul-history.md`. 일하는 방식·서버 세계 재개 명령 → `grep -n '^## 일하는 방식\|^## 현재 상태' archive/kkumteul-history.md`.
7. 11/06 판 "먼저 할 것"(다시 보기 새 파일 올리기 — `runs/replay-two.html` 14MB라 통째 읽기 불가) — 상태 미확인.

## 결정 대기 (사용자)
- 밤샘 가지(`night/2026-11-26-kkumteul`, 07b·맛 기억 포함)를 main에 합칠지 — `cd ~/git/kkumteul && bash scripts/merge-into-main.sh night/2026-11-26-kkumteul && echo 저장됨`
- 상주 세션(announcement·insight·research)을 tmux로 띄울지, 세션 안 예약 작업을 둘지(앱 백그라운드 세션 8시간 자동 정리 — 메모리 `bg-sessions-retire-after-8h`)

## 막다른 길
- 고정 갈래 겨루기 "양쪽 전멸"을 볼드윈 증거로 읽기 → 마당이 독 순서를 새 씨앗으로 뽑던 측정 버그(`53f1000`).
- 가르침 곁 머무름을 새끼 속도 상한(0.3)으로 → `par_near_frac` +0.004·+0.018뿐(16:46 ET 관문 −1).

## 시작 때 확인
- `systemctl --user is-active kkumteul-teach-floor; tail -3 ~/git/kkumteul/runs/teach-exp/progress.txt` — 속도 상한 0.1 점검
- `git -C ~/git/kkumteul/.claude/worktrees/night-2026-11-26 log --oneline -5` — 이 파일 뒤 새 커밋(`규범의 씨앗 N:`로 구현 진도)
- `free -m | sed -n 2p; uptime` — 시뮬레이션 한 번에 하나(1.5GB·2스레드), 시험 2GB
- 끝난 것: 아침 브리핑용 정리(5번) · 정렬 시험 셋 `a7f61b7`·`db96d54`(4b) · 가르침 관문 None 고침 `57c1ce8`
- 끝난 것: 사회 학습 판정 나 다섯 갈래 `c3c7e50` 뒤 · 가르침 구현 1~9 `2030141` · 규범 계획 `f0a1cca`

## 교훈 (상황 → 찾을 곳)  — 교훈 파일 `archive/kkumteul-history.md`
- 공개 레포 문서·인수인계에 경로·다른 프로젝트 이야기를 적을 때 → grep -n '공개 레포 문서' archive/kkumteul-history.md
- 설계·계획에 목표 비율·"몇 %"를 적을 때 → grep -n '목표 비율\|미루지 않는다' archive/kkumteul-history.md
- 장부 시험·측정 코드를 쓰거나 고칠 때 → grep -n '장부 시험\|뻔히 지는\|물질 수지' archive/kkumteul-history.md
- 협동·신호·감각 장치를 겨룰 때 → grep -n '협동이 필요\|입력을 0으로\|섞인 겨루기\|겨루기 승리만' archive/kkumteul-history.md
- 겨루기 결과가 "양쪽 전멸"로 깨질 때 → `grep -n '고정 갈래 다시 재기' wayfinder/stage2-4/results.md`
- 오래 도는 실험을 시작할 때(코드 사본, memcap `--bg`는 `$PWD`가 작업 폴더라 `env -C <사본>`으로) → grep -n 'runs/code-\|메모리 디스크\|승인 창이 뜰 수' archive/kkumteul-history.md
- 단서를 씨앗 하나로 해석할 때 → 불활성 갈래와도 견준다(로저스 해석을 씨앗 a만 보고 적었다가 거둠, `results.md` "판정 나 결과, 씨앗 a·b")
- 문서에 시각을 적을 때, 합치기 명령을 드릴 때 → grep -n 'date +%H:%M\|merge-into-main' archive/kkumteul-history.md
- 메모리·CLAUDE.md로 올린 것: numba prange·rng 용도 번호·에너지 장부·커밋하면 push(CLAUDE.md), `bg-sessions-retire-after-8h`·`phone-notify-script`·`background-sessions-work-on-a-branch`(메모리)
