# 꿈틀 announcement 인수인계
마지막 갱신: 2026-11-06 17:4x ET (첫 판, 설정 세션 `79b8905e`에서 넘김)

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

## 지금 걸린 것 (위에서부터)
1. **인사이트 첫 바퀴**: 주제 "맛 기억 음성 대조(대조 b +0.08, 막기 −0.33, 방향 반대)가 깨진 원인 후보와 가르는 최소 실험". 결론은 `night-kkumteul-learn-2`로 가게 정정해 보냄(처음엔 이 설정 세션으로 적었음). 결론이 나오면 대표님께 요약.
2. **status.md 낡음 확인**: 메인 가지의 새 결과 — 맛 기억 물려받기 `taste_rate` 0.2도 못 넘어 접음(`519490f`), 사냥 손 개체 결과(`811c584`), 신호 판정 측정 구현 중 — 가 main의 status.md에 없다. 그 가지는 status 갱신 규칙이 생기기 전에 갈라져 메인이 규칙을 모를 수 있다. 가지를 main에 합칠 때 status.md가 갱신됐는지 보고, 아니면 메인에게 SendMessage로 요청.
3. 리서처는 요청 없는 동안 "진화하지 않은 입력 칸이 켜졌을 때 행동이 바뀌는 현상(떠도는 가중치·숨은 선호·굴절적응)"을 조사 중(Arak & Enquist 숨은 선호, 재현 반론 포함).

## 현황 (17:47 ET)
- 메인: 신호 판정 측정 구현 중, 정상. 대표님 새 순서: 측정 → 개인 학습 → 사회 학습 → 가르침 → 섬.
- insight: 0단계 통과, 판정 가~라 읽는 중. research: 웹 조사 중. 옛 리서처(`--agent` 없이 Bash 쓰던 것)는 꺼짐.

## 교훈·주의
- ★ (11/06) 세션 이름은 첫 프롬프트 훅으로만 붙는다. 떠 있는 세션은 스스로 이름을 못 바꾼다(대표님이 `/rename`). 앱이 제목을 바꾸면 SendMessage 주소도 바뀌니, 다른 세션에 "받는 곳"을 알릴 때 `ListAgents`로 실제 이름을 확인한다(이 세션이 `kkumteul-2e`로 알렸다가 이름이 바뀌어 정정함).
- ★ (11/06) 서브에이전트 `tools`를 빼면 전부 물려받는다. `[]`가 0개. 문서에 없어 `claude -p`로 시험해 확인했다. "list-agents 스킬"은 없다 — `ListAgents`·`SendMessage` 도구.
- 공개 레포: 이 파일에 로컬 절대 경로·서버 구성·개인 정보 적지 않는다.
