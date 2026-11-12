# 꿈틀 announcement 인수인계
마지막 갱신: 2026-11-12 12:21 ET · 새 양식 전환(내용은 11/06 20:0x ET 판 그대로)
요약: 꿈틀 상주 세션(메인 구현·insight·research)의 현황을 사용자에게 알리고 중계하는 역할 — 구현은 하지 않는다(사용자 11/06 17:47 ET). 11/09 "일단 실험 중지"(`kkumteul.md` ⛔)로 지금은 상주 세션·체인을 띄우지 않는다. 공개 레포라 이 파일에 로컬 절대 경로·서버 구성·개인 정보를 적지 않는다. 지난 원문(당시 현황·구조·교대 절차 전문)·교훈 전문은 `archive/kkumteul-announcement-history.md`.

## 다음 세션이 이어서 할 것
0. 사용자 최근 말(원문, 자동)
- (11/12 12:16) "(개인 내용 — 생략)" → 처리됨(announcement 인수인계 새 양식 전환)
1. **실험 중지 중** — 사용자가 재개하라고 하실 때까지 이 역할 세션도 띄우지 않는다. 밤샘 체인 2026-11-07b는 `stopped`(`kkumteul.md`).
2. 재개하면 역할·구조를 이렇게 복원한다(전문은 archive "2026-11-12 새 양식 전환 때 옮김"):
   - 현황은 `ListAgents`(busy/idle) + 각 세션 기록 jsonl의 마지막 말·도구(읽기만)로 본다. 다른 세션에 말 넣기는 `SendMessage`만. 실험 실행·설정 변경·합치기 같은 규칙 A 일은 사용자에게.
   - 에이전트 파일 셋 `.claude/agents/`: `insight.md`(opus), `researcher.md`(sonnet, Bash·브라우저 없음), `idea-lens.md`(`tools: []`). 상주 세션은 `claude --agent insight` / `claude --agent researcher`로 띄워야 제한이 걸린다. `docs/status.md`는 메인 세션이 갱신한다.
   - 모델·노력도(사용자 11/06 18:0x): insight·idea-lens·announcement는 claude-opus-5-5·medium, research는 claude-sonnet-5-5·high, 메인 구현은 Opus·high.
   - 상주 세션 교대: insight·research가 맥락이 차면 announcement에 "교대 요청"(할 일 3~5줄)을 보낸다 → `~/.claude/bin/wrap-spawn kkumteul-insight <레포 경로> claude-opus-5-5 medium insight` / `wrap-spawn kkumteul-research <레포 경로> claude-sonnet-5-5 high researcher` → `띄움:` 확인 → 새 세션에 SendMessage(`ListAgents`의 `[ref]`) → 옛 세션은 `claude agents --json`의 id로 `claude stop <id>`.
   - 커밋·push(사용자 11/06 18:0x): main에 자동 커밋·push, 검사는 개인정보·API 키·env만. 리서처는 Bash가 없어 announcement가 커밋한다 — 전용 작업 폴더 `.claude/worktrees/research`(가지 `research`)에서 add·commit·fetch·rebase origin/main·push HEAD:main, 그 뒤 main 폴더 `git pull --ff-only`.

## 결정 대기 (사용자)
- 실험 재개 시점, 그리고 상주 세션을 tmux로 띄울지(백그라운드 데몬 세션은 8시간 입력이 없으면 자동 정리됨 — `kkumteul.md` 결정 대기와 같은 건)

## 시작 때 확인
- `python3 ~/.claude/skills/overnight/night.py status 2026-11-07b` — 체인이 `stopped`인지
- `git worktree list` — 밤샘·research 작업 폴더가 남아 있는지
- `ls .claude/agents/` — insight·researcher·idea-lens 파일이 그대로인지
- 끝난 것: 상주 세션 구조(에이전트 파일 셋·status.md) `fb33047`·`2db2aa4` (11/06)
- 끝난 것: 밤샘 체인 2026-11-07b 시작·정지 기록은 `kkumteul.md`의 "마지막 결과"

## 교훈 (상황 → 찾을 곳)  — 교훈 파일 `archive/kkumteul-announcement-history.md`
- 밤샘 체인을 새로 걸 때, 체인이 멈췄는지 확인할 때, 재시작을 기다릴 때 → grep -n '체인을 새로 걸\|체인이 멈췄는지\|/overnight' archive/kkumteul-announcement-history.md
- 다른 세션에 SendMessage "받는 곳"을 알릴 때, 세션 이름이 바뀌었을 때 → grep -n '세션 이름은 첫 프롬프트' archive/kkumteul-announcement-history.md
- 서브에이전트 `tools` 제한을 정할 때 → grep -n 'tools`를 빼면' archive/kkumteul-announcement-history.md
