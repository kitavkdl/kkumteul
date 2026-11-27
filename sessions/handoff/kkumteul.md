# 꿈틀 인수인계
마지막 갱신: 2026-11-27 02:0x ET · 밤 — 규범 지금 구조 판 관문 −1 통과(판정 0 도는 중). 앞: 규범 두 판을 고쳐 다시 띄움(−1e 읽기 버그 `74ef54e`, norm_impulse 0.035, 첫 점검 창 손실 몫 0.090~0.099로 문턱 안), 쌍둥이·눈금자 측정 끝, "꿈틀 여정" 아티팩트에 8~10장 추가
요약: 세계 시뮬레이션 꿈틀 — 아무도 가르치지 않았는데 생물이 스스로 행동·전략·신호를 만드는지 본다(공개 레포, 목적·규칙은 CLAUDE.md). **작업 가지 `night/2026-11-26-kkumteul`, 작업 폴더 `.claude/worktrees/night-2026-11-26`. 이 가지에 커밋하면 `git push` 뒤 `git push origin HEAD:main`, 원래 폴더(main)는 `git merge --ff-only origin/main`.** 사용자 16:26 "싹다 진행해 쭉" — 묻지 말고 끝까지, 서버 메모리·CPU 안 터지게. 사용자 11/27 00:41: 규범 체인과 짧은 측정은 같이 돌려도 됨.

## 다음 세션이 이어서 할 것
0. 사용자 최근 말(원문, 자동)
- (11/27 01:49) "[끼어든 말] 그리고 지금 하던것까지만 하고 기술적 내용이랑 멋있게 포장하고 멋있는 중요한 것들 쓴 아티팩트 원래 있던거에서 수정해주고 작업 마무리해줘" → 처리됨("꿈틀 여정" https://claude.ai/artifact/EZUf6Mkz429VDmrkWddk3H 에 8 배우는 생물·9 규범의 씨앗·10 11월 27일 상태 추가, 버전 7)
- (11/27 00:41) "ㅇㅇ 같이 돌려도 됨. 밤샘 가지도 쓸만한것들 main에 합쳐주고" → 처리됨(main 합침, 측정 끝 — 4번)
- (11/26 16:26) "ㅇㅋ 싹다 진행해 쭉." → 진행 중(1번)
- (11/26 16:27) 두 발 걷기·두 팔 쓰기 아이디어 → 처리됨(`sessions/tasks.md` "아이디어 후보")
- (11/26 16:34) "ㅇㅇ 체크포인트에서 비교해서 진행하면 될듯" → 확정: 규범을 지금 구조·매-비둘기 두 판으로(1번)
1. **규범 두 판 — 다시 띄움(11/27 00:53 ET, 유닛 `norm-two`, 사본 `runs/code-74ef54e`, 로그 `runs/norm-two.log`).** 앞 실행은 관문 −1에서 닫힘(−1b 0.048, −1e 읽기 버그 → 경고 stderr로 고침 `74ef54e`). **조정 1/3: `IMP=0.035`**(`results.md` "두 판 관문 −1 결과와 조정"). 기록 폴더 `runs/norm-exp-t50-i0.035{,-fx0.08-hawk}`. 01:43~01:52 점검 창 손실 몫 0.090·0.099·0.093(sig-a·none-a·sig-b), 유병률 9.1%·5.8%·4.2%(상한 10% 가까움 — 볼 것).
   - **관문 −1 통과(11/27 01:58 ET, 지금 구조 판)**: 손실 몫 0.091, 유병률 5.6%, 나머지 모두 통과 — `results.md` "지금 구조 판 관문 −1 다시 잼 — 통과", status 고침. 지금 판정 0(용량 반응) 도는 중, 다음 관문(α·β·γ·δ·−2)은 어림 약 15시간 뒤(11/27 17시 ET 무렵) — 그때 `gate.json` 읽고 판정 적기. **매-비둘기 판은 지금 구조 판 전체(어림 약 7일) 뒤에 시작**(앞 인수인계의 03:15 어림은 앞 판이 닫힐 때 얘기).
   - **`memcap --bg`에는 변수를 `--` 뒤 `env -C <사본> ROOT=… CODE=… IMP=… bash <레포>/runs/norm-two.sh`로**(앞에 두면 유닛에 안 넘어가 즉시 실패).
   - 두 판 견주기 `wayfinder/stage2-4/norm_twin_compare.py`. 남은 걱정: 매-비둘기 판의 판정 0은 집행자 싸움 비용 포함이라 같은 잣대로 안 견줌, 손 집행자가 표시 없는 이웃도 침, 기울기 시작판 켜짐(문턱 위 0.0007).
   - 아티팩트 "꿈틀 여정" 관문 통과 반영함(버전 8, 11/27 02:1x). 다음 관문 판정 나면 같은 식으로(Artifact read로 받아 고침).
4. **develop 실측** — 4 쌍둥이·눈금자 **측정 끝**(`sessions/develop/kkumteul-communication.md` "실측 4 결과"): 쌍둥이 중립 sd 0.071 > 옛 0.041 → 기본값 안 바꿈. 눈금자 섭취 1%당 r ≈ 0.033(검출 한계 = 섭취 2~4%). 다음은 5단계(설정만 격자) — 규범 체인 뒤.
4b. **시험 돌리는 법**: `NUMBA_NUM_THREADS=2 <메모리 상한 도구> --max 2G -E NUMBA_NUM_THREADS -- env -C <작업 폴더> nice <레포>/.venv/bin/python -m pytest -q -p no:cacheprovider tests/`(약 4분, 785개).
5b. 주의: 코드 사본 `runs/code-faf9852`의 `soc_chain.sh`는 옛 판(같은 기록 폴더로 DRY하면 로그를 자름).
6. 이전 계획(11/09 이전): 11/06 판 "다음 할 일" → `grep -n '2026-11-09 대청소' sessions/handoff/archive/kkumteul-history.md`.
7. 11/06 판 "먼저 할 것"(다시 보기 새 파일 올리기, `runs/replay-two.html` 14MB) — 상태 미확인.

## 결정 대기 (사용자)
- 매-비둘기 판을 지금 구조 판과 **나란히** 띄울지(지금은 차례로라 약 7일 뒤 시작). 나란히면 CPU 2스레드·메모리 약 0.5GB 더, 서버 부하 평균 약 8/12. 차례 그대로가 기본.
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
- 쌍둥이 겨루기를 막기 겨루기 기본값으로 → 중립 sd가 오히려 큼(0.071 대 0.041).

## 시작 때 확인
- `systemctl --user is-active norm-two; tail -4 runs/norm-two.log` — 규범 체인 관문 −1 결과(레포 원래 폴더 기준)
- `git -C <레포>/.claude/worktrees/night-2026-11-26 log --oneline -3` — 이 파일 뒤 새 커밋
- `free -m | sed -n 2p; df -h ~ | tail -1` — 메모리·디스크
- 끝난 것: 2. **사회 학습 — 끝.** 판정 나 다섯 갈래 모두 지지 · 3. **가르침 — 끝(접음).** `0071eae`
- 끝난 것: (11/26 20:41) 디스크 → `runs/_pytest_tmp`만 지움 · −1e 읽기 버그 고침·IMP 0.035 다시 띄움 `74ef54e`
- 끝난 것: 쌍둥이·눈금자 측정·develop 실측 4 기록 · 아티팩트 8~10장

## 교훈 (상황 → 찾을 곳) — 교훈 파일 `sessions/handoff/archive/kkumteul-history.md`
- 공개 레포 문서에 경로·다른 프로젝트 이야기를 적을 때 → grep -n '공개 레포 문서' sessions/handoff/archive/kkumteul-history.md
- 설계에 목표 비율·"몇 %"를 적을 때 → grep -n '목표 비율\|미루지 않는다' sessions/handoff/archive/kkumteul-history.md
- 장부 시험·측정 코드를 쓸 때 → grep -n '장부 시험\|뻔히 지는\|물질 수지' sessions/handoff/archive/kkumteul-history.md
- 협동·신호·감각 장치를 겨룰 때 → grep -n '협동이 필요\|입력을 0으로\|섞인 겨루기' sessions/handoff/archive/kkumteul-history.md
- 겨루기가 "양쪽 전멸"로 깨질 때 → grep -n '고정 갈래 다시 재기' wayfinder/stage2-4/results.md
- 오래 도는 실험을 시작할 때(코드 사본, memcap `--bg`) → grep -n 'runs/code-\|메모리 디스크\|승인 창이 뜰 수' sessions/handoff/archive/kkumteul-history.md
- 오래 도는 체인을 띄운 뒤 → 첫 관문이 닫힐 때까지(약 1시간) progress를 직접 보고 "도는 중"이라 적는다(11/27: 띄우고 "12.5일" 보고, 실제는 1시간 만에 닫힘)
- 판정 도구가 json을 stdout 리다이렉트로 받을 때 → 경고는 stderr로(11/27 −1e, `kkumteul/checkpoint.py`)
- 새 상호작용 장치를 넣을 때 → 사건 빈도를 계수로 먼저 잰다(`results.md` "매-비둘기 판 다시 설계")
- 설계 때 어림한 효과 크기 → 3배 빗나갈 수 있음(norm_impulse 14% 어림 → 4.8%), 관문 조정 횟수를 남겨 둔다(`results.md` "두 판 관문 −1 결과와 조정")
- 마당 판정에서 "대조 칸도 같은 양"이면 → 처리 효과로 읽지 않는다(`results.md` "끼니 비용 판 반복 18 결과")
- 유전체 함수인 값을 유전체 거리로 통제하면 거꾸로 치우친다 → `dialect_map.py` 주석
- 메모리·CLAUDE.md로 올린 것: numba prange·rng 용도 번호·에너지 장부·커밋하면 push(CLAUDE.md), `bg-sessions-retire-after-8h`·`phone-notify-script`·`background-sessions-work-on-a-branch`(메모리)
