# 꿈틀 announcement 인수인계
마지막 갱신: 2026-11-06 18:0x ET (announcement 2131b3d6)

## 역할 (대표님 11/06 17:47 ET)
꿈틀 세션들의 현황을 대표님께 알리고 중계한다. 구현은 하지 않는다.
- 지켜볼 세션: 메인 구현(지금은 밤샘 체인 `night-kkumteul-learn-2`, 작업 폴더 `.claude/worktrees/night-2026-11-07-kkumteul-learn`, 가지 `night/2026-11-07-kkumteul-learn`), `kkumteul-insight`, `kkumteul-research`.
- 현황은 `ListAgents`(busy/idle) + 각 세션 기록 jsonl의 마지막 말·도구(읽기만)로 본다. 메인 가지는 `git -C <작업 폴더> log --oneline`.
- 다른 세션에 말 넣기는 `SendMessage`만. 실험 실행·설정 변경·합치기 같은 규칙 A 일은 대표님께.

## 구조 (11/06 이 세션에서 만듦, `fb33047`, `2db2aa4`)
- `.claude/agents/insight.md` — opus, tools `Read, Glob, Grep, Agent(idea-lens), ListAgents, SendMessage`. 0단계: status.md 갱신일이 결과 문서보다 오래면 메인에게 갱신 요청. 결론은 메인에게.
- `.claude/agents/researcher.md` — sonnet, tools `ListAgents, SendMessage, WebSearch, WebFetch, Read, Write, Edit, Glob, Grep`(Bash·브라우저 없음). `docs/research/raw/`에 쌓고 인사이트에게만 알림.
- `.claude/agents/idea-lens.md` — `tools: []`(도구 0개). 인사이트가 status.md 전문·리서치 요약을 붙여 준다.
- 상주 세션은 `claude --agent insight` / `claude --agent researcher`로 띄워야 프롬프트·도구 제한이 걸린다. 떠 있는 세션은 파일을 고쳐도 안 바뀜.
- `docs/status.md` — 세계 규칙·관찰(수치만)·해석(추정만)·막힌 점·기각. 갱신은 메인 몫(레포 CLAUDE.md 「세션 운영」).

## 모델·노력도와 상주 세션 교대 (대표님 11/06 17:5x ET, 18:0x 정정)
- **insight·idea-lens·announcement는 claude-opus-5-5 · medium, research는 claude-sonnet-5-5 · high** (대표님 17:39 "리서처는 model: sonnet", 18:0x "리서치는 원래 소넷 높음"). 앞 세션이 17:48 "모든 애들 이 모델·노력도"를 리서처까지로 읽어 opus로 바꿨다가 정정함. 메인 구현은 Opus·high(대표님 18:0x 확정: "넌 오퍼스 중간, 리서치는 소넷 높음, 인사이트는 오퍼스 중간, 메인구현은 오퍼스 높음").
- **insight·research는 계속 돈다.** 맥락이 차면 에이전트 파일 「교대」 절대로 announcement에 "교대 요청"(할 일 3~5줄)을 보낸다. 인수인계 파일·유언은 남기지 않는다.
- 교대 요청을 받으면: ① `~/.claude/bin/wrap-spawn kkumteul-insight <레포 경로> claude-opus-5-5 medium insight` / `wrap-spawn kkumteul-research <레포 경로> claude-sonnet-5-5 high researcher` (5번째 인자가 `--agent`) ② `띄움:` 출력 확인 ③ 새 세션에 SendMessage로 할 일 전달 — 이름이 겹치니 `ListAgents`의 `[ref]`(bg 쪽이 새 세션)로 보낸다 ④ 옛 세션 끄기: `~/.local/bin/claude agents --json`에서 id → `claude stop <id>`. 손으로 띄운 세션(id 없음)은 안 꺼지니 대표님께 창을 닫아 달라고 한 줄.
- **커밋·push (대표님 11/06 18:0x):** 이 레포는 main에 자동 커밋·push. 검사는 개인정보·API 키·env만. 리서처는 Bash가 없어 raw 파일을 못 올리므로 announcement가 커밋한다(리서처가 파일 이름을 알려 줌). 리서처는 백그라운드라 main 폴더에 못 쓰므로 전용 작업 폴더 `.claude/worktrees/research`(가지 `research`)에 쓴다 → `git -C <그 폴더>`로 add·commit·fetch·rebase origin/main·push HEAD:main, 그 뒤 main 폴더 `git pull --ff-only`. 리서처를 새로 띄우면 이 경로를 할 일에 넣는다.

## 지금 걸린 것 (위에서부터)
1. **옛 창 둘 닫기 대기**: `claude --agent insight`(pid 1248540)·`claude --agent researcher`(pid 1249338) — 대표님께 요청함. 그 전엔 같은 이름이 둘씩이라 메시지가 옛 창으로 갈 수 있음(옛 insight가 정정을 보내와 새 insight에 전달함).
2. **인사이트 첫 바퀴 결론(정정판)**은 메인에 전달됨: 1순위 대조 b 기억 입력 뒤섞기(mode=shuffle)로 판정 가 지표·막기 겨루기 재실행 + 1인당 섭취량·먹은 것의 독 여부 기록. 막기 −0.33은 중립 −0.007±0.05라 진짜 효과. 실행은 대표님 확인 뒤, 새 방향 다음.
3. **status.md 낡음**: 새 insight가 메인에게 갱신 요청함(가지 results.md에 판정 셋·11/07 정정). 메인은 자기 가지에서 고치므로 main 합칠 때 들어왔는지 확인.

## 현황 (18:0x ET)
- 메인: 신호 판정 측정 구현 중. insight `[111f6c]`(opus·medium, bg): 0단계 갱신 대기 → 개인 학습 방향 발산 예정. research `[a53060]`(sonnet·high, bg): 할 일 받음.

## 교훈·주의
- ★ (11/06) 세션 이름은 첫 프롬프트 훅으로만 붙는다. 떠 있는 세션은 스스로 이름을 못 바꾼다(대표님이 `/rename`). 앱이 제목을 바꾸면 SendMessage 주소도 바뀌니, 다른 세션에 "받는 곳"을 알릴 때 `ListAgents`로 실제 이름을 확인한다(이 세션이 `kkumteul-2e`로 알렸다가 이름이 바뀌어 정정함).
- ★ (11/06) 서브에이전트 `tools`를 빼면 전부 물려받는다. `[]`가 0개. 문서에 없어 `claude -p`로 시험해 확인했다. "list-agents 스킬"은 없다 — `ListAgents`·`SendMessage` 도구.
- 공개 레포: 이 파일에 로컬 절대 경로·서버 구성·개인 정보 적지 않는다.
