# 신호 판정 측정 — 계획 (2026-11-07 밤)

설계: `docs/superpowers/specs/2026-11-07-signal-measure-design.md` (검토 두 갈래 판단표대로). 고치는 파일은 `kkumteul/knockout.py`, `tests/test_signal_info.py`, `tests/test_knockout.py`(describe 칸 번호 시험) 셋. 세계 코드·`config.py`·`rng.py`는 고치지 않는다.

## 과제
1. **describe 칸 번호 바로잡기**(구현 중 발견): 내는 쪽 상황의 에너지·방금 먹음·냄새가 `_base`(시야 + 내부 8칸 뒤 = 단계 입력 시작)에서 세어져 실제로는 스트레스 호르몬·들은 신호 두 칸이었다. 내부 감각은 `2 * vision_sectors`부터다(`kernels.sense`). 시험: 손으로 넣은 obs에서 상황 열이 그 칸을 집는지.
2. **섞기 막기** `knockout(cfg, channel, tag, mode="zero"|"shuffle", seed, frac)`
   - `zero`는 지금 그대로(세 감각 함수 뒤 모두 지움 — 옛 결과와 같은 동작).
   - `shuffle`은 그 칸을 쓰는 함수 뒤에서만 한 번: 시야·맛 → `kernels.sense`, 듣기·닮음·먹는 것·사회 → `stages.social_sense`, 흔적 → `stages.trail_sense`.
   - A(tag) 산 개체 가운데 `frac` 몫을 그 스텝에 무작위로 골라(1이면 전부) 그 칸 묶음을 행째로 순열. 지역 생성기 `numpy.random.default_rng(seed)`.
   - 블록이 내주는 값: 칸 번호 배열과 같은 꼴을 유지하되 `.stats`(섞은 호출 수, 고른 행 수, 실제로 값이 바뀐 행 수)를 붙인다 — `KnockoutCols`(ndarray 하위 클래스).
3. **`split --mode shuffle --frac`**: 출력에 `mode`·`frac`·`changed_row_frac`, 반복마다 A·B 출생·사망 수(`compete`의 사건을 셈 — `measure`를 고치지 않고 tag별 끝 머릿수와 함께), 중립 흔들림으로 검출 가능한 최소 r(2 × 중립 sd × t95/√reps 근사 — 임시).
   - 출생·사망 수는 `measure.compete`가 사건을 버리므로 이번에는 넣지 않고 끝 머릿수만(판단표 명세 11 — 부분 반영으로 문서에 적는다).
4. **상호정보량 바탕**: `mi_binned(x, y, bins)`(분위 칸, 동점은 같은 칸, 2차원이면 열마다 칸을 나눠 결합 부호), `mi_diff(x, y, y_null_list, ids, folds, seed)`(개체 id로 겹 나눔, 겹마다 관측 − 영가설 평균, t 구간, 부호 뒤집기 정확 검정 p), `mi_gain(x, y, ids, steps, ...)`(영가설 = 같은 스텝에서 "고정된 무작위 순서의 옆 개체"의 y), `holm(ps, alpha)`.
5. **`info` 하위 명령**: 체크포인트에서 `steps` 동안 돌며 `stride`마다 표본. 스텝 전에 h·Hin·Hrec 사본.
   - 내는 쪽: sig0·sig1·합·차 × 에너지·방금 먹음·냄새·호르몬 넷(마음 꺼지면 "해당 없음"). 관측 값과 개체 안 값(개체 평균을 뺀 것). Holm. 판정용 = 신호 2 × (에너지, 방금 먹음) 4짝 가운데 Holm 통과 ≥ 1 이면서 그 짝의 개체 안 값도 통과.
   - 듣는 쪽: 들은 4칸 결합(칸마다 3분위) ↔ 행동 결합(추력·회전, 칸마다 4분위). 반응 = I(들은 것; F(obs)) − I(들은 것; F(obs, 들은 칸만 고정 순서 옆 개체 것)). 층 갈래: 섞을 짝을 시야 세기 3분위 층 안에서만. 기준선 = 같은 표본에서 Win 듣기 행을 개체끼리 섞은 뇌로 같은 값. 켜진 세계면 공격·주기 출력도 따로.
   - `--seed`를 주면 세계 씨앗을 바꿔 돈다(반복 사이 구간용). 출력 JSON에 체크포인트·스텝·씨앗·표본 수·값·구간.
6. 전체 시험 → 커밋 → 예측 먼저 → 판정 실험(world4 900만, atk-a 20만) → 결과 문서.
