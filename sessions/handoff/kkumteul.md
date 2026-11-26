# 꿈틀 인수인계
마지막 갱신: 2026-11-26 17:4x ET · 낮 — 사회 학습 끝(다섯 갈래 지지)·가르침 관문 −1 곁 부족으로 닫힘·속도 상한 0.1 점검 도는 중·규범 구현 과제 0~6 끝(검토 지적 4건 남음)
요약: 세계 시뮬레이션 꿈틀 — 아무도 가르치지 않았는데 생물이 스스로 행동·전략·신호를 만드는지 본다(공개 레포, 목적·규칙은 CLAUDE.md). 11/26 새벽 실험 재개. **작업 가지 `night/2026-11-26-kkumteul`, 작업 폴더 `.claude/worktrees/night-2026-11-26`(main에 안 합침, 원격에 push함).** 사용자 `/goal`: 목요일(11/26) 오전 8시 전까지 계속, 묻지 말고 끝까지, ultracode 한시 허용, 메모리·CPU 뻑나지 않게.

## 다음 세션이 이어서 할 것
0. 사용자 최근 말(원문, 자동)
- (11/26 02:38) "꿈틀 지금 작업할거 많아?" → 처리됨(남은 일 답함)
- (11/26 02:39) "지금 구독량 완전 쭉 쓸 수 있어서(알다시피 오전 8시까지) 할만한 작업들 지금 쭉 돌려도 되긴 해 ultracode도 한시 허용할게. overnight 느낌으로다가. 곧 내가 잘거라서. 그리고 /develop 창의성 << 이거 해도 되고. 대신 나한테 물어볼거나 뭐 막히는거 없이 완전 쭉 진행해줘야해. 다만 메모리나 뭐 cpu 뻑나거나 그런것만 없도록 반드시 지속해서 진행해주던가" → 거의 처리(1·3·4번 남음)
- (11/26 16:26) "ㅇㅋ 싹다 진행해 쭉." → 진행 중(가르침 체인 시작, 사회 학습 결과 정리 끝, 규범 구현 다음)
- (11/26 16:34) "ㅇㅇ 체크포인트에서 비교해서 진행하면 될듯" → 확정(3번)
- (11/26 16:33) "뭘 얻고 뭘 잃는건데? 매 비둘기 구조는 뭐고? 비교하는건 필요할 것 같기는 한데" → 답함. 결정: 규범 구현(과제 0~18) 뒤 `fight_extra`(맞공격 때 추가 손실, 기본 꺼짐) 과제를 덧붙이고, 규범 판정을 지금 구조·매-비둘기 둘 다 같은 체크포인트에서 갈라 돌린다
- (11/26 16:27) 두 발 서기·빈 손 아이디어 → 처리됨(`sessions/tasks.md` "아이디어 후보", 가르침·규범·섬 뒤)
- (11/26 02:42) "[끼어든 말] 다 했는데 오전 8시까지(지금 5시간 20분정도 남았네) 만약 시간이 더 남았으면 계속 develop이나 우리의 목표(철학, 규율과 법, 시뮬레이션 인식)정도까지 계속 나아가줘 develop도 마찬가지고" → 아직(3·4번)
1. **사회 학습 — 끝.** 판정 나 다섯 갈래(씨앗 a·b·c, 시작 속도 0.2 a·b) 모두 지지(`results.md` 두 절, status 갱신). 체인 08:44 ET 끝.
2. **가르침 — 관문 −1에서 닫힘(16:46 ET, `results.md` "관문 −1 결과")**: 새끼당 기대 부모 독 끼니 관찰 0.353·0.442 < 0.5(곁 부족), −1c도 무리 미달. 속도 상한은 `par_near_frac`를 +0.004·+0.018밖에 못 올림. **지금 도는 것**: 설계 6절 다음 판 `juv_move_floor=0.1` 세계 점검(유닛 `kkumteul-teach-floor`, 스크립트는 스크래치라 사라질 수 있음 — 내용: 바탕 260만에서 3만 스텝 branch `--set preset=taste taste_inherit=0 taste_gene_enabled=true kind_flip_period=500 juv_enabled=true juv_move_floor=0.1` → `taste_gene_judge.py observe` 5,000스텝, 결과 `runs/teach-exp/w-on01-{a,b}.json`, 진행 `runs/teach-exp/progress.txt` "on01 끝"). 끝나면 기대 관찰 = par_near_frac × 200 × 끼니율 × 독 비율을 셈해 결과 절에 덧붙임. ≥ 0.5(두 바탕)면 `teach_chain.sh`를 `juv_move_floor=0.1` 판으로 다시(설계·예측을 먼저 고쳐 씀), 못 넘으면 "부모 방향 입력 + 따라가기 출력" 판을 설계 셋째 판으로 쓰고 구현. 곁에 안 머무는 채로 "가르침이 안 쓰인다"고 쓰지 않는다.
3. **규범의 씨앗 구현 — 과제 0~6 끝**(`1c65d07`·`8fd5689`·`3a547b8`·`32506e9`·`ff5def6`·`4292336`, 전체 시험 648 통과, 서브에이전트 opus). sonnet 검토(치명 0) 지적을 **다음 묶음 구현자가 먼저 고친다**(전부 반영으로 판단함): ① [중요] 만료 스텝 재표시 때 앞 에피소드가 안 세어짐(`stages.py` `_attach_mark` ~669) → 과제 9 `norm_account`에서 닫기 처리 + "만료 스텝 재표시" 시험 ② [중요] `NT_TAKE_KIN`이 "계보 또는 닮음 ≥0.92"로 구현됨(`stages.py` ~880, 계획은 계보 친족만, 설계 176행은 닮음 문턱을 사후에) → 계보만으로, `tests/test_norm.py` ~450 시험도 고침(과제 8 `pun_role` 1은 계획대로 닮음 포함 — 그대로) ③ [사소] 커널에서 벌금 `got = min(got, dmg*attack_gain)` 막기(장부 시험은 got 오류를 못 잡음) ④ [사소] `test_scan_not_called_when_both_off`에 njit이라 호출을 못 센다는 한계 주석. 그다음 과제 7~12 → sonnet 검토 → 13~18 → 검토 → **추가 과제 `fight_extra`**(맞공격 때 추가 손실, 기본 꺼짐; 사용자 16:33·16:34 "체크포인트에서 비교해서 진행") → 규범 체인을 지금 구조·매-비둘기 둘로. 묶음 구현은 한 번에 하나(서버 메모리), 브리핑 꼴은 이 세션처럼: 계획 절·공통 규칙·worktree 한 줄 명령·memcap 2G 시험·과제마다 커밋 push.
4. **develop 창의성 다음 실측** — 2바퀴 기록 `sessions/develop/kkumteul-communication.md` "2바퀴"(`6403420`): 여섯 관점 방안·겹친 곳·실측 순서 2~8, 실측 1(용량 반응 추세: a 단조 +, b 뒤집힌 U, blind 강한 음)은 함. 다음: 실측 2(`saweat` 막기 겨루기)·3(방언 지도, 읽기 전용)을 사회 학습 체인 뒤에, 4(쌍둥이·심은 효과 눈금자, `measure.py` TDD). 5단계(말하는 뇌, `docs/stage5-proposal.md`) 시험판 검토는 아직.
4b. 정렬 고침(`152357b`) 2축 검토 남은 것 — **끝남(밤 3차)**: 묶음 → `load_side` → 진짜 `compete` 마당 세계의 커널 독 순서 시험, `hand_learner.curve`·`solo` 정렬 시험(셋 다 정렬을 빼면 깨지는 것 확인). `taste_judge.py`는 일부러 안 고침(원래 세계에 씨앗을 덮어쓴 세계).
5. 8시 무렵: 아침 브리핑용으로 status·인수인계 정리(무엇을 재었고 무엇이 돌고 있는지).
5b. 주의: `soc_chain.sh`를 도는 체인과 같은 기록 폴더로 DRY 돌리지 말 것 — 레포 판은 `5dda27e`로 고쳤지만 도는 코드 사본(`runs/code-faf9852`)은 옛 판이라 DRY가 갈래 로그를 자른다.
6. 이전 계획(11/09 이전, 이제 위 순서 뒤): 11/06 판 "다음 할 일"(맛 기억 다음·사냥·2부-다·암수) → `grep -n '2026-11-09 대청소' archive/kkumteul-history.md`. 일하는 방식·서버 세계 재개 명령 → `grep -n '^## 일하는 방식\|^## 현재 상태' archive/kkumteul-history.md`.
7. 11/06 판 "먼저 할 것"(다시 보기 새 파일 올리기 — `runs/replay-two.html` 14MB라 통째 읽기 불가로 이번 밤엔 안 올림) — 상태 미확인.

## 결정 대기 (사용자)
- 밤샘 가지들(`night/2026-11-26-kkumteul`이 07b·맛 기억을 모두 포함)을 main에 합칠지 — 합치기 한 줄: `cd ~/git/kkumteul && bash scripts/merge-into-main.sh night/2026-11-26-kkumteul && echo 저장됨`
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
