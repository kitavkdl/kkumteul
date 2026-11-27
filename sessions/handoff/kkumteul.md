# 꿈틀 인수인계
마지막 갱신: 2026-11-27 00:50 ET · 밤 — 규범 두 판 체인이 둘 다 관문 −1에서 닫힘(−1b 손실 몫 부족 + −1e 읽기 버그), 밤샘 가지를 main에 합침, 쌍둥이·눈금자 측정 도는 중
요약: 세계 시뮬레이션 꿈틀 — 아무도 가르치지 않았는데 생물이 스스로 행동·전략·신호를 만드는지 본다(공개 레포, 목적·규칙은 CLAUDE.md). **작업 가지 `night/2026-11-26-kkumteul`, 작업 폴더 `.claude/worktrees/night-2026-11-26`. 11/27 00:4x에 main에 통째로 합침(`946b871..e903493`) — 이 가지에 커밋하면 `git push` 뒤 `git push origin HEAD:main`, 원래 폴더(main)는 `git merge --ff-only origin/main`.** 사용자 16:26 "싹다 진행해 쭉" — 묻지 말고 끝까지, 서버 메모리·CPU 안 터지게. 사용자 11/27 00:41: 규범 체인과 짧은 측정은 같이 돌려도 됨.

## 다음 세션이 이어서 할 것
0. 사용자 최근 말(원문, 자동)
- (11/27 00:41) "ㅇㅇ 같이 돌려도 됨. 밤샘 가지도 쓸만한것들 main에 합쳐주고" → 처리됨(main 빨리감기 합침, `twin-measure` 동시 실행 — 4번)
- (11/26 16:26) "ㅇㅋ 싹다 진행해 쭉." → 진행 중(1번)
- (11/26 16:27) 두 발 걷기·두 팔 쓰기 아이디어 → 처리됨(`sessions/tasks.md` "아이디어 후보")
- (11/26 16:34) "ㅇㅇ 체크포인트에서 비교해서 진행하면 될듯" → 확정: 규범을 지금 구조·매-비둘기 두 판으로(1번)
- (11/26 20:41) 디스크 → 처리됨(`runs/_pytest_tmp`만 지움)
1. **규범 두 판 — 관문 −1에서 닫힘, 고쳐서 다시 띄우기가 첫 일.** `norm-two`(코드 사본 `runs/code-81e6fa4`)가 지금 구조 판 23:23, 매-비둘기 판 00:22에 둘 다 종료 코드 5(미판정)로 끝남. 판정 0·진화 갈래 안 돎. 원자료 `runs/norm-exp-t50-i0.015{,-fx0.08-hawk}/gate-world.json`.
   - ① **−1e 읽기 버그**: `arena-check.json` 첫 줄에 체크포인트 경고("주의: …다른 계산 코드로 저장됐다")가 stdout으로 섞여 `norm_judge.py` 531줄 `_load_json`이 None → 미판정. 실제 값은 take 0·pun 0(통과). 고칠 곳: 경고를 stderr로 보내거나 체인이 json만 남기게(TDD, 시험 추가). 다른 `> *.json` 명령(knock-norm·knock-vis 등)도 같은 위험인지 확인.
   - ② **−1b 진짜 못 넘음**: 지금 구조 판 `take_loss_frac` 0.048(문턱 0.08~0.15), 관문 권고 "norm_impulse를 올린다"(조정 최대 3번). 매-비둘기 판 값은 같은 json에서 읽기. 규칙대로 **결과 문서에 판정·조정을 먼저 적고** `IMP=<새 값>`으로 두 판 다시(꼬리표가 바뀌어 새 폴더). 새 값은 손실 몫이 IMP에 대략 비례한다고 보면 0.015×(0.11/0.048)≈0.034 — 추정, 적을 때 근거 같이. −1a·−1c(유병률 2.7%, 원천 맞음)·−1d는 통과.
   - ③ 고친 커밋으로 새 코드 사본 → `runs/norm-two.sh`에 `IMP` 넘겨 `memcap --bg norm-two` 다시(앞 유닛은 끝남). `results.md` "두 판 체인 띄움" 절과 status의 "도는 중·12.5일"을 이 결과로 고쳐 쓴다.
   - 두 판 견주기 `wayfinder/stage2-4/norm_twin_compare.py`. 남은 걱정: 매-비둘기 판의 판정 0은 집행자 싸움 비용 포함이라 같은 잣대로 안 견줌, 손 집행자가 표시 없는 이웃도 침, 기울기 시작판 켜짐(문턱 위 0.0007).
2. **사회 학습 — 끝.** 판정 나 다섯 갈래 모두 지지(`results.md`, status).
3. **가르침 — 끝(접음).** 끼니 비용 반복 18도 가를 수 없음(`0071eae`): `costonly`도 같은 양이라 마당 틀의 몫. 직접 지표는 작동.
4. **develop 실측**(`sessions/develop/kkumteul-communication.md` "2바퀴"): 2 `saweat` 막기·3 방언 지도 끝, 4 쌍둥이·눈금자 구현 끝(`compete(twin, plant)`, `measure ruler`, `knockout split --twin`). 측정 **도는 중**(11/27 00:43~, 유닛 `twin-measure`, `runs/twin-measure.py`, 사본 `runs/code-e903493`, 결과 `runs/twin-measure.json`·`.log`, 약 3시간). 끝나면 develop 문서 "실측 4"에: 쌍둥이 중립 sd가 옛 중립보다 작고 0.05 아래면 막기 겨루기 기본값 후보, 눈금자 기울기로 지난 "가를 수 없음"들을 "섭취의 몇 %"로. 5단계는 아직.
4b. **시험 돌리는 법**: `NUMBA_NUM_THREADS=2 <메모리 상한 도구> --max 2G -E NUMBA_NUM_THREADS -- env -C <작업 폴더> nice <레포>/.venv/bin/python -m pytest -q tests/`(약 6분, 784개). `--bg` 유닛에는 변수를 `env -C <사본> ROOT=… CODE=… bash …`로.
5b. 주의: 코드 사본 `runs/code-faf9852`의 `soc_chain.sh`는 옛 판(같은 기록 폴더로 DRY하면 로그를 자름).
6. 이전 계획(11/09 이전): 11/06 판 "다음 할 일" → `grep -n '2026-11-09 대청소' sessions/handoff/archive/kkumteul-history.md`.
7. 11/06 판 "먼저 할 것"(다시 보기 새 파일 올리기, `runs/replay-two.html` 14MB) — 상태 미확인.

## 결정 대기 (사용자)
- 상주 세션(announcement·insight·research)을 tmux로 띄울지, 세션 안 예약 작업을 둘지(앱 백그라운드 세션 8시간 자동 정리)
- (알림, 막지 않음) 매-비둘기 판은 "맞은 쪽이 매면 둘 다 손실" 정의로 진행(같은 스텝 맞공격은 0.5%만 걸림).
- (디스크) 끝난 갈래 중간 체크포인트 약 19G — 11/26 안 지우기로 하심. 빠듯하면 다시 여쭘.

## 막다른 길
- 고정 갈래 겨루기 "양쪽 전멸"을 볼드윈 증거로 → 마당 독 순서 측정 버그(`53f1000`).
- 가르침 곁 머무름을 새끼 속도 상한으로 → +0.03 남짓. 따름 반응(F1)으로 해결.
- world4 `saweat`가 신호 자리를 차지한다 → 막아도 가를 수 없음.
- 매-비둘기를 "같은 스텝 맞공격 추가 손실"로 → 공격의 0.5%라 거의 안 걸림(`a915ea9`).
- 방언을 world4에서 찾기 → 배움 0이라 문화적 방언이 생길 길 없음.
- 가르침 끼니 비용 판 순이득을 가르침 효과로 → `costonly`도 같은 크기(`0071eae`).

## 시작 때 확인
- `systemctl --user is-active twin-measure; tail -3 runs/twin-measure.log` — 측정 진행(레포 원래 폴더 기준)
- `git -C <레포>/.claude/worktrees/night-2026-11-26 log --oneline -3` — 이 파일 뒤 새 커밋
- `free -m | sed -n 2p; df -h ~ | tail -1` — 체인 다시 띄우기 전 메모리·디스크
- 끝난 것: 두 판 체인 띄움(관문 −1에서 닫힘, `runs/norm-two.sh` 종료 코드 찍기 고침) · 쌍둥이·눈금자 구현(784 통과) · main 합침 `e903493`
- 끝난 것(이전): 가르침 접음 `0071eae` · hawk 규칙·체인 `34e53b3` · gate-hand 미판정 `e6038cb`

## 교훈 (상황 → 찾을 곳) — 교훈 파일 `sessions/handoff/archive/kkumteul-history.md`
- 공개 레포 문서에 경로·다른 프로젝트 이야기를 적을 때 → grep -n '공개 레포 문서' sessions/handoff/archive/kkumteul-history.md
- 설계에 목표 비율·"몇 %"를 적을 때 → grep -n '목표 비율\|미루지 않는다' sessions/handoff/archive/kkumteul-history.md
- 장부 시험·측정 코드를 쓸 때 → grep -n '장부 시험\|뻔히 지는\|물질 수지' sessions/handoff/archive/kkumteul-history.md
- 협동·신호·감각 장치를 겨룰 때 → grep -n '협동이 필요\|입력을 0으로\|섞인 겨루기' sessions/handoff/archive/kkumteul-history.md
- 겨루기가 "양쪽 전멸"로 깨질 때 → grep -n '고정 갈래 다시 재기' wayfinder/stage2-4/results.md
- 오래 도는 실험을 시작할 때(코드 사본, memcap `--bg`) → grep -n 'runs/code-\|메모리 디스크\|승인 창이 뜰 수' sessions/handoff/archive/kkumteul-history.md
- 오래 도는 체인을 띄운 뒤 → 첫 관문이 닫힐 때까지(약 1시간) progress를 직접 보고 "도는 중"이라 적는다(11/27: 띄우고 "12.5일" 보고, 실제는 1시간 만에 닫힘)
- 판정 도구가 json을 stdout 리다이렉트로 받을 때 → 경고가 섞이면 관문이 미판정(11/27 −1e)
- 새 상호작용 장치를 넣을 때 → 사건 빈도를 계수로 먼저 잰다(`results.md` "매-비둘기 판 다시 설계")
- 마당 판정에서 "대조 칸도 같은 양"이면 → 처리 효과로 읽지 않는다(`results.md` "끼니 비용 판 반복 18 결과")
- 유전체 함수인 값을 유전체 거리로 통제하면 거꾸로 치우친다 → `dialect_map.py` 주석
- 메모리·CLAUDE.md로 올린 것: numba prange·rng 용도 번호·에너지 장부·커밋하면 push(CLAUDE.md), `bg-sessions-retire-after-8h`·`phone-notify-script`·`background-sessions-work-on-a-branch`(메모리)
