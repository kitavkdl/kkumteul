# 부모의 학습 보조 투자("가르침" 후보) 구현 계획 (2026-11-26 밤)

설계: `docs/superpowers/specs/2026-11-26-parent-teaching-design.md` 둘째 판(검토 반영). 이 계획은 그 설계를 **과제 아홉 개(+ 전제 확인 0)**로 쪼개 한 명이 순서대로 구현할 수 있게 한다.
**사회 학습 구현(`docs/superpowers/plans/2026-11-26-social-learning.md` 과제 1~7)이 끝난 코드 위에 얹는다.** 그 계획이 추가하는 필드·칸·함수 이름을 그대로 참조한다:
`soc_g`, `meal_k`, `meal_o`, `n_seen`, `n_seen_poison`, `n_seen_kin`, `seen_age_sum`, `watch_meals(step, s, fp, ip)`, `Q_SOC`, `Q_SOC_MODE`, `P_SOC_RATE`·`P_SOC_COST`·`P_SOC_GENE_SIGMA`·`P_SOC_RADIUS`,
`S_SOC_GENE`(32)·`S_SOC_BLIND`(33), `SOC_ID_STRIDE`, `measure.OPTIONAL_GENOME == ("taste_g", "soc_g")`, `knockout.TASTE_OFF`, `taste_gene_judge.py rate-zero --gene soc --soc-d --set --no-individual --no-neutral`·`gate`, `wayfinder/stage2-4/soc_chain.sh`, 판정 원자료 `runs/soc-exp/j-dose-<이름>-<태그>.json`.
과제마다 TDD: **시험 먼저 → 빨강 확인 → 구현 → 초록 확인 → 커밋**. 설정 값은 전부 임시 결정(`config.py`에서만 바꾼다).

공통 규칙(과제마다 지킨다)
- 시험 명령: `python3 -m pytest -q tests/<파일>`. 서버에서는 메모리 상한 도구로 감싸고(`--max 2G`), `NUMBA_NUM_THREADS=2`, `nice`로 낮춘다. `kernels.py`·`config.py`를 고치면 numba 캐시가 지문(`_cache._SOURCES`)으로 새로 지어지므로 첫 실행이 느리다.
- 박은 해시를 쓰는 기존 시험이 그대로 통과해야 한다: `tests/test_taste.py`, `tests/test_stages.py`, `tests/test_harsh.py`, `tests/test_prey.py`, `tests/test_soc_taste.py`(소수 시험은 과제 끝에, 전체는 과제 9에서).
- 에너지가 생기는 길은 먹기와 명시적 주입뿐이다 — 새 비용은 에너지를 빼고 `L_BRAIN`에 더하기만, 새 낭비·손실은 `L_LOST`로만. 설정 검사가 새는 값(floor > 1, 음수 비용·rate)을 막는다. 장부 시험이 통과해야 한다.
- `rng.py`의 용도 번호는 뒤에 덧붙인다(34, 35). 기존 호출 순서·번호를 바꾸지 않는다.
- numba: 새 커널(`juv_pass`·`teach_pass`·`feed_pass`)은 **순차 `@njit(cache=True)`**(`prange` 금지, 읽기·쓰기는 상태 묶음 `s`를 거친다). 새 커널은 `kernels.py`에만 둔다.
- **나이 규칙**: `world.step`에서 `move`가 끝에서 `age += 1`하므로 `move`는 증가 전 나이, 그 뒤(`eat`·패스·`watch_meals`·출생 직전)는 증가 뒤 나이를 본다. 모든 곳이 `a = age − 1`을 "move 시점 나이"로 쓴다(`move`만 `a = age`). 미숙기(관찰자·졸업·계수) = `age ≤ juv_steps`, 어른(시연자) = `age > juv_steps`.
- 공개 레포: 문서·주석에 로컬 절대 경로·서버 구성·다른 프로젝트 이야기를 쓰지 않는다(레포 상대 경로만, 체인 스크립트 안의 경로는 변수).
- 고친 파일만 `git add`, 커밋하면 바로 push(프로젝트 규칙).

## 과제 0. 전제 확인 (코드 안 고침)
이 계획은 **사회 학습 과제 1~5(최소)가 커밋되고 시험이 초록**이어야 시작한다(`watch_meals`·`eat`·`copy_genome`·`P_*`/`Q_*`·`GENOME`·`world.step`·`measure.compete`가 같은 줄이다).
- `git log --oneline | grep "사회 학습"`로 과제 1~5 커밋을 확인. `grep -n "def watch_meals" kkumteul/kernels.py`, `grep -n "TASTE_OFF" kkumteul/knockout.py`, `grep -n "def kind_aligned" kkumteul/measure.py`가 모두 있어야 한다. 없으면 **멈추고** 사회 학습을 먼저 끝낸다(`git status`가 깨끗해야 한다 — 다른 에이전트의 미커밋 변경과 섞지 않는다).
- 시험 한 번: `python3 -m pytest -q tests/test_soc_taste.py tests/test_taste_gene.py tests/test_taste.py tests/test_knockout.py` 통과.
- 번호 읽기: `grep -n "N_FP\|N_IP" kkumteul/kernels.py`(현재 `range(64)`·`range(29)`는 사회 학습 반영 뒤 값, 이 계획은 그 **맨 뒤**에 덧붙임), `grep -n "^S_" kkumteul/rng.py`(마지막이 33이면 34·35, 다르면 그 뒤 번호로 맞추고 설계·시험의 번호를 고친다), `grep -n "P_SOC_RADIUS" kkumteul/kernels.py`(`pack_params`가 켜짐과 무관하게 채우는지 — 이 계획의 곁 계수기가 `soc_radius`를 쓴다).
- 바탕 체크포인트: `ls runs/g-p500-a/checkpoints runs/g-p500-b/checkpoints`에 `ckpt_000002600000.npz`·`.json`이 둘 다 있어야 체인을 돌린다(없으면 체인 금지).
- 사회 학습 체인이 아직 도는 중이면 이 계획의 **체인은 돌리지 않는다**(한 번에 하나). 구현·시험은 상관없다.

---

## 과제 1. 설정·난수·필드·옛 호환 (동작 변화 없음)
**바꾸는 파일**: `kkumteul/config.py`, `kkumteul/rng.py`, `kkumteul/kernels.py`, `kkumteul/measure.py`, `kkumteul/knockout.py`, 새 `tests/test_teach.py`, `tests/test_taste_gene.py`, `tests/test_knockout.py`.

**시험 먼저**
- `tests/test_teach.py`
  - `test_teach_config_defaults_and_checks`: 기본 꺼짐(`juv_enabled`·`juv_measure`·`teach_enabled`·`feed_enabled`=False), `juv_steps=200`, `juv_eat_floor=0.5`, `juv_move_floor=0.3`, `juv_poison_mult=1.0`, `juv_know_theta=0.2`, `teach_mode="kin"`, `teach_gain=0.5`, `teach_cost=0.0005`, `teach_cost_when="near"`, `teach_gene_sigma=0.02`, `feed_rate=0.02`, `feed_gene_sigma=0.02`.
    검사: `juv_steps < 1`, floor ∉ (0, 1], `juv_poison_mult < 1`, `juv_know_theta`가 (0, 1] 밖, `teach_gain` ∉ [0, 1], `teach_cost`·`teach_gene_sigma`·`feed_rate`·`feed_gene_sigma` < 0, 모르는 `teach_mode`·`teach_cost_when`이면 `ValueError`.
  - `test_teach_requires_soc_outcome_and_kinds`: `teach_enabled`는 `juv_enabled`·`soc_taste_enabled`·`food_kinds_enabled`를 요구하고 `soc_taste_mode == "outcome"`이어야 한다(`what`·`blind`·`inert`이면 `ValueError`). `feed_enabled`는 `juv_enabled`만 요구. `juv_measure`는 아무것도 요구 안 함.
  - `test_teach_mode_string_override`: `cfg.with_overrides(["teach_enabled=true", "teach_mode=knower", "teach_cost_when=meal"])`가 문자열을 받는다(사회 학습이 만든 문자열 지원 재사용).
  - `test_teach_fields_registered`: `AGENT_FIELDS`에서 `show_g`·`feed_g`가 `soc_g` **바로 뒤**, float32, 저장. 저장 안 함: `know_age`(int32)·`know_src`(int8)·`n_seen_par`·`n_seen_par_poison`·`n_taught`·`n_taught_knower`·`par_near_steps`·`par_alive_steps`·`juv_alive_steps`(int32)·`adult_near_sum`(int64)·`par_slot`·`jscr_i`(int32)·`jscr_d`(float64)·`tflag`(int8). `HASH_DEFAULTS`: `show_g`·`feed_g` 0, **`know_age` −1**. `alloc_state`의 새 상태에서 `know_age`가 전부 −1, 나머지 새 필드는 0, `grad_tab.shape == (TAG_ROWS, N_GT)`(`TAG_ROWS == 8`), `"grad_tab" in OTHER_SCRATCH`이고 `SAVED_FIELDS`에는 없음.
  - `test_know_age_minus_one_after_checkpoint_roundtrip`: 세계를 저장·복원한 뒤에도 산 개체의 `know_age`가 전부 −1(복원 대상이 아니라 `alloc_state` 기본값). `measure.inject_agent`로 넣은 개체도 −1과 0 초기화.
  - `test_teach_in_genome`: `"show_g" in measure.GENOME`, `"feed_g" in measure.GENOME`, `measure.OPTIONAL_GENOME == ("taste_g", "soc_g", "show_g", "feed_g")`. 만든 묶음에서 `show_g`·`feed_g`만 빼고 `np.savez`로 다시 저장해도 `pack.read`가 0으로 읽는다(한쪽만 없는 경우도).
  - `test_teach_off_world_hash_unchanged`: 처음부터 꺼진 세계(`taste_world("taste")` 등)의 해시가 기존 박은 해시 시험과 같다.
  - `test_old_checkpoint_loads_show_feed_zero`: 0 아닌 `show_g`·`feed_g`가 저장·복원되고, 없는 옛 `.npz`는 0으로 읽힌다.
- `tests/test_taste_gene.py::test_new_rng_stream_appended`: 사회 학습이 만든 일반화 꼴(모든 `S_*` 값이 서로 다르고 31·32·33 고정)에 `S_TEACH_GENE == 34`, `S_FEED_GENE == 35`를 더한다(`max == 33` 같은 단정이 남아 있으면 지금 고친다).
- `tests/test_knockout.py`: `"teach_enabled=false" in ko.TASTE_OFF`. `test_knockout_taste_split_with_teach_on_passes_config_check`: `juv_enabled`·`soc_taste_enabled`·`teach_enabled`를 켠 체크포인트 설정에 `TASTE_OFF`를 `with_overrides`로 한꺼번에 적용해도 `ValueError`가 안 남.
- **빨강 확인**: `python3 -m pytest -q tests/test_teach.py tests/test_taste_gene.py tests/test_knockout.py`.

**구현**
1. `config.py`: 설정(`juv_enabled`, `juv_measure`, `juv_steps`, `juv_eat_floor`, `juv_move_floor`, `juv_poison_mult`, `juv_know_theta`, `teach_enabled`, `teach_mode`, `teach_gain`, `teach_cost`, `teach_cost_when`, `teach_gene_sigma`, `feed_enabled`, `feed_rate`, `feed_gene_sigma`)과 모드 상수·사전: `TEACH_KIN=0, TEACH_COSTONLY=1, TEACH_ANY=2, TEACH_KNOWER=3, TEACH_DRIFT=4`(`TEACH_MODES = {"kin": ..}`), `WHEN_NEAR=0, WHEN_MEAL=1`(`TEACH_WHENS`). `validate()`: 위 검사. 주석에 "사회 학습의 `soc_taste_mode`와 이름이 겹치지 않게 `costonly`·`drift`" 한 줄.
2. `rng.py`: `S_TEACH_GENE = 34`("보여 주기 강도 유전자의 변이"), `S_FEED_GENE = 35`("먹이 대주기 유전자의 변이").
3. `kernels.py`: `AGENT_FIELDS`에 위 필드(저장 둘은 `soc_g` 바로 뒤, 저장 안 함은 `n_seen*` 옆). `HASH_DEFAULTS`에 `"show_g": 0, "feed_g": 0, "know_age": -1`. 상수 `TAG_ROWS = 8`, `GT_*`(졸업표 칸 인덱스: `GT_GRAD, GT_DEAD, GT_N1, GT_M1, GT_K1, GT_N2, GT_M2, GT_K2, GT_M3, GT_K3, GT_SRC1..GT_SRC4, GT_PAR_NEAR, GT_PAR_ALIVE, GT_ADULT_NEAR, GT_ALIVE_STEPS, GT_SEEN, GT_SEEN_PAR, GT_SEEN_PAR_POISON, GT_TAUGHT, GT_TAUGHT_KNOWER, GT_MEALS, GT_POISON_MEALS, GT_E_GRAD, GT_TEACH_COST, GT_FEED_GIVEN, N_GT`)와 파이썬 쪽 이름표 `GT_NAMES`(소문자 이름 튜플, 판정 도구가 씀). `OTHER_SCRATCH`에 `"grad_tab"`, `alloc_state`에 `d["grad_tab"] = np.zeros((TAG_ROWS, N_GT), dtype=np.float64)`.
   `P_*`: `P_JUV_EAT_FLOOR, P_JUV_MOVE_FLOOR, P_JUV_POISON_MULT, P_JUV_KNOW_THETA, P_TEACH_GAIN, P_TEACH_COST, P_TEACH_GENE_SIGMA, P_FEED_RATE, P_FEED_GENE_SIGMA, P_FEED_LOSS, P_FEED_MIN`를 `N_FP` 앞(맨 뒤)에 덧붙이고 `range(...)` 개수 +11. `Q_*`: `Q_JUV, Q_JUV_COUNT, Q_JUV_STEPS, Q_TEACH, Q_TEACH_MODE, Q_TEACH_WHEN, Q_FEED`를 `N_IP` 앞에 덧붙이고 +7.
   `pack_params`: `juv_on = cfg.juv_enabled`, `ip[Q_JUV] = int(juv_on)`, `ip[Q_JUV_COUNT] = int(juv_on or cfg.juv_measure)`, `ip[Q_JUV_STEPS] = cfg.juv_steps`, `ip[Q_TEACH] = int(cfg.teach_enabled and juv_on and soc_on)`(`soc_on`은 사회 학습이 만든 값), `ip[Q_TEACH_MODE]`, `ip[Q_TEACH_WHEN]`, `ip[Q_FEED] = int(cfg.feed_enabled and juv_on)`, `fp[P_*]`(`P_FEED_LOSS = cfg.give_loss`, `P_FEED_MIN = cfg.give_min_energy`).
4. `measure.py`: `GENOME`에 `"show_g"`, `"feed_g"` 덧붙임, `OPTIONAL_GENOME = ("taste_g", "soc_g", "show_g", "feed_g")`(`pack.py`는 이미 일반화돼 있음). `inject_agent`의 초기화에 `know_age = −1`, 나머지 새 저장 안 함 필드 0(과제 3에서 필드가 쓰이기 전에 미리).
5. `knockout.py`: `TASTE_OFF`에 `"teach_enabled=false"` 추가(`feed_enabled`는 맛과 무관해 안 넣는다).

**초록 확인**: `python3 -m pytest -q tests/test_teach.py tests/test_taste_gene.py tests/test_taste.py tests/test_knockout.py tests/test_soc_taste.py`. 커밋: "부모의 가르침 1: 설정·난수·필드·옛 호환(동작 변화 없음)".

---

## 과제 2. 미숙기: 먹이 효율·속도 상한·독 배수
**바꾸는 파일**: `kkumteul/kernels.py`(`juv_ramp`, `eat`, `move`), `tests/test_teach.py`.

**시험 먼저**(손 시험은 `tests/test_taste.py`의 `feed`·`_bad`·`lone` 패턴; 비용·낭비는 `eat`·`move`를 직접 불러 잰다)
- `test_juv_ramp_hand_values`: `juv_ramp(a, floor, steps)` = `floor + (1 − floor) × min(1, a/steps)`; `a=0`→0.5, `a=100`→0.75, `a=200`·`a=500`→1(floor 0.5, steps 200).
- `test_eat_efficiency_by_age`: 나이 `age = a + 1`(`eat`는 증가 뒤 나이)인 개체가 들판 알 하나를 먹으면 에너지 증가가 `food_energy × ramp(a)`이고 `s.eaten`도 같고 `L_LOST`가 `food_energy × (1 − ramp)` 늘어남. `a = 0, 100, 200, 300`. 어른은 옛 동작(차액 0).
- `test_eat_corpse_full_efficiency`: 같은 새끼가 **사체·고기**(`KIND_CORPSE`·`KIND_MEAT`)를 먹으면 효율 1(`L_LOST` 안 늘음).
- `test_eat_poison_mult_juvenile_only`: `juv_poison_mult=3`이면 미숙기 새끼가 독을 먹을 때 에너지가 `poison × 3` 줄고 `L_LOST`도 `f_energy + poison × 3` 늘음, 어른은 `poison`.
- `test_move_speed_cap_by_age`: 갓 태어난 새끼(`age=0`)의 속도가 `v_max × floor`를 안 넘고, 나이가 차면 풀리고, 어른은 `v_max`. `move`가 증가 전 `age`를 `a`로 씀(손셈: `age=100`→ 상한 0.65 × v_max).
- `test_juv_knobs_neutral_hash_equals_off`: `juv_enabled=true`지만 `juv_eat_floor=1`·`juv_move_floor=1`·`juv_poison_mult=1`이면 상태 해시가 꺼진 세계와 같다 — **soc·conflict가 꺼진 세계와, `soc_taste_enabled`·`conflict_enabled`가 켜진 세계 둘 다**(두 세계마다 꺼진 짝과 비교).
- `test_juv_off_never_branches`: 꺼진 세계는 박은 해시 시험이 그대로(과제 끝에 기존 시험으로 확인).
- `test_ledger_closes_with_juv`: `juv_enabled`·`juv_poison_mult=2`로 세계를 1,500스텝 돌려 `rel_residual(w) < 1e-9`. 낭비 줄을 `L_LOST`에 안 더하면 실패한다(구현 뒤 한 번 지워 실패하는 것을 보고 되돌린다).
- 빨강 확인.

**구현**
1. `juv_ramp(a, floor, steps)`: `@njit(cache=True, inline="always")` 작은 함수(`good_kind` 근처). `a >= steps`면 1.0.
2. `eat`: `juv = ip[Q_JUV] != 0`. 산 개체 루프에서 `a = s.age[i] − 1`, `m = 1.0`, `pm = 1.0`; `if juv and a < steps: m = juv_ramp(a, fp[P_JUV_EAT_FLOOR], steps); pm = fp[P_JUV_POISON_MULT]`. 독 분기: `s.energy[i] -= fp[P_POISON] × pm`, `s.ledger[L_LOST] += s.f_energy[j] + fp[P_POISON] × pm`. 좋은 알 분기: **`always_edible(fk)`이면 `gain = s.f_energy[j]`**, 아니면 `gain = s.f_energy[j] × m`; `s.energy[i] += gain`, `s.eaten[i] += gain`, `s.ledger[L_LOST] += s.f_energy[j] − gain`(0이면 더하지 않는 분기는 두지 않는다 — `x + 0.0`은 비트가 같다).
3. `move`: `vm = vmax`; `if ip[Q_JUV] != 0 and s.age[i] < ip[Q_JUV_STEPS]: vm = vmax × juv_ramp(s.age[i], fp[P_JUV_MOVE_FLOOR], ip[Q_JUV_STEPS])`(`move`의 `age`는 증가 전 = `a`); 속도 상한 비교에 `vm`을 쓴다. 이동 비용·`age += 1`은 그대로.
4. `world.set_config`: 주석 한 줄("`juv_*`를 런타임에 바꿔도 유전체는 그대로").

**초록 확인**: `python3 -m pytest -q tests/test_teach.py tests/test_taste.py tests/test_stages.py tests/test_harsh.py tests/test_prey.py`. 커밋: "부모의 가르침 2: 미숙기(효율·속도 상한·독 배수)".

---

## 과제 3. `juv_pass`: 곁 머무름 계수기·졸업표·미숙기 사망
**바꾸는 파일**: `kkumteul/kernels.py`(`juv_pass`, `_reset_body`, `process_deaths`), `kkumteul/world.py`(`step`, `grad_table`, `reset_grad_table`, summary), `kkumteul/bench.py`, `tests/test_teach.py`.

**시험 먼저**(손 시험: 개체를 직접 놓고 `juv_pass`를 부른다. **격자를 안 짓는다** — 선형 탐색이 이웃을 본다는 것이 설계의 요점이다)
- `test_juv_pass_counts_parent_near`: 부모·새끼를 반지름 안에 놓으면 `par_near_steps`·`par_alive_steps`·`juv_alive_steps`가 오르고, 반지름 밖이면 `par_alive_steps`·`juv_alive_steps`만 오름. 부모 id가 −1이거나 부모가 죽었으면 `par_alive_steps`가 안 오름(`juv_alive_steps`는 오름). 양성 확인: 가까운 쌍에서 `par_near_steps == 1`부터 확인.
- `test_juv_pass_cell_boundary_and_wrap`: 칸 경계(`x = cell × k`) 양쪽 0.5 안쪽 쌍(예: x = 7.8과 8.2, 거리 0.4)·거리 `soc_radius` 바로 안(7.99)과 바로 밖(8.01)·토러스 반대편(0.5와 511.5)을 정확히 센다(격자 없이도 놓치지 않음).
- `test_adult_near_sum_excludes_parent`: 곁에 부모 아닌 어른 둘 + 부모 하나면 `adult_near_sum`이 2씩 오름(부모는 안 셈), 미숙기 이웃은 안 셈(어른 = `age > juv_steps`).
- `test_juv_pass_in_juv_only_world`: `juv_enabled`만 켜고 `soc`·`conflict`·`teach`·`feed` 모두 꺼진 작은 세계를 몇백 스텝 돌려 `juv_alive_steps`가 양수이고 `par_alive_steps ≤ juv_alive_steps`. **`juv_measure`만 켠 세계**(`juv_enabled=false`)도 센다. 두 경우 모두 `soc_taste_enabled`가 꺼져 있어도 돈다(첫째 판의 "`teach_on`에 묶여 못 잼" 구멍 고정).
- `test_juv_measure_hash_equals_off`: `juv_measure`만 켠 세계의 상태 해시가 꺼진 세계와 같다(계수기·졸업표는 저장 안 함).
- `test_juv_pass_off_not_called`: 꺼진 세계에서 `juv_pass` 호출 0(감싸서 셈).
- `test_grad_record_conditions`: 태어난 스텝부터 센 새끼가 나이 `juv_steps`가 되는 스텝에 `GT_GRAD` +1(태그 행), 졸업 때 `E_GRAD`·`MEALS`·`POISON_MEALS`·`SEEN*`·곁 계수 합산, 스냅샷 `S1`(`juv_steps // 4`)·`S2`(`// 2`)에 `GT_N`·`GT_M`(= `−taste_m[i, 독 칸]`)·`GT_K`(앎 누적) 합산(독 칸 = `1 − good_kind(...)`). **`parent == −1`인 개체, `inject_agent`로 중간 나이에 들어온 개체, 계수기를 중간부터 센(`juv_alive_steps != age`) 개체는 기록 안 됨.**
- `test_juvenile_death_recorded`: 미숙기에 굶어 죽은 새끼(`parent ≥ 0`, `juv_alive_steps == age`)는 `process_deaths`에서 `GT_DEAD` +1(태그 행), 어른·부분 계수 개체는 아님. 졸업률 = `GRAD ÷ (GRAD + DEAD)`.
- `test_grad_tag_fold_no_overflow`: 태그 5·99·−3 개체가 졸업·사망해도 `grad_tab` 밖이 안 쓰임(`row = min(max(tag, 0), 7)`): 다른 배열의 해시·`grad_tab.shape` 불변, 행 7·행 0에 접혀 쌓임.
- `test_new_counters_reset_on_birth`: `_reset_body`로 슬롯을 재사용하면 `juv_alive_steps`·`par_*`·`adult_near_sum`·`n_seen_par*`·`n_taught*`·`know_src`가 0, `know_age`가 −1.
- `test_world_grad_table_api`: `world.grad_table()`이 `{태그: {GT_NAMES: 값}}` 사전을 돌려주고 `reset_grad_table()`이 0으로. 체크포인트로 이어 돌린 직후 이미 자라던 새끼는 기록에 안 들어감(재개 시점의 `juv_alive_steps`가 0에서 시작하므로 `!= age`).
- `test_juv_pass_cost_scales`(선택): 산 개체 600, 새끼 120일 때 `juv_pass` 한 번이 현실적 시간 안(예: < 5 ms)에 끝남(어림 확인용, 느리면 설계 약점 4의 사설 격자 대안을 정한다).
- 빨강 확인.

**구현**
1. `juv_pass(step, s, fp, ip)` — `@njit(cache=True)` 순차. `n = s.x.shape[0]`, `steps = ip[Q_JUV_STEPS]`, `r2 = fp[P_SOC_RADIUS]²`.
   - `for i: s.par_slot[i] = -1`.
   - `for i in range(n)`: 죽었거나 `s.age[i] > steps`(어른)이면 건너뜀. 미숙기: `s.juv_alive_steps[i] += 1`. `p = s.parent[i]`; `q = -1`; `near = False`; `k = 0`. `for j in range(n)`: `j == i`거나 죽었으면 건너뜀; `if p >= 0 and s.ids[j] == p: q = j; (거리 ≤ r이면 near = True)`; 부모가 아니고 어른(`s.age[j] > steps`)이고 거리² ≤ r2이면 `k += 1`(거리는 `_wdiff`). `s.par_slot[i] = q`; `if q >= 0: s.par_alive_steps[i] += 1; if near: s.par_near_steps[i] += 1`; `s.adult_near_sum[i] += k`.
   - **기록 조건**: `born_here = s.parent[i] >= 0 and s.juv_alive_steps[i] == s.age[i]`. `born_here`이고 `s.age[i] == steps // 4`이면 스냅샷 1(`GT_N1 += 1`, `GT_M1 += −taste_m[i, 독 칸]`, 앎이면 `GT_K1 += 1`), `== steps // 2`이면 스냅샷 2, `== steps`이면 **졸업**: `GT_GRAD += 1`, `GT_M3`·`GT_K3`, `know_src` 칸(`GT_SRC1..4`), 곁·본 끼니·가르침 계수 합, `GT_MEALS += n_meal`, `GT_POISON_MEALS += n_poison`, `GT_E_GRAD += energy`. 행은 `row = min(max(tag, 0), TAG_ROWS − 1)`. 독 칸·앎 판정은 도우미 `_poison_slot(step, ip)`(= `1 − good_kind(ip[Q_KIND_SEED], step + ip[Q_KIND_STEP0], ip[Q_KIND_PERIOD])`, 먹이 종류가 꺼진 세계는 −1을 돌려 스냅샷·앎을 건너뜀)와 `_knows(s, i, kp, fp)`(= `kp >= 0 and taste_m[i, kp] <= −fp[P_JUV_KNOW_THETA]`).
   - docstring: "순차 전용, prange 금지. 격자를 안 쓰고 산 개체를 선형으로 훑는다(칸 경계 0.5 안쪽 이웃을 놓치지 않고 `s.a_start`를 안 건드린다). O(J×N)."
2. `process_deaths`: `_remove` 호출 앞(굶음·공격 사망 가지)에 `if ip[Q_JUV_COUNT] != 0 and s.parent[i] >= 0 and s.age[i] <= ip[Q_JUV_STEPS] and s.juv_alive_steps[i] == s.age[i]: s.grad_tab[row, GT_DEAD] += 1`. 노쇠 사망은 미숙기일 수 없어 안 넣는다. 주석: "`strike`(운석) 사망은 안 센다".
3. `_reset_body`: 새 필드 초기화(`know_age = −1`, 나머지 0, `par_slot = −1`).
4. `world.step`: `k.eat(...)`(사회 학습이 켜졌으면 그 뒤의 격자 재구성·`watch_meals` 앞)에 `if self.ip[k.Q_JUV_COUNT]: k.juv_pass(t, s, fp, ip)`. 즉 순서는 `eat` → [soc: `build_grid`] → `juv_pass` → [teach_pass: 과제 5] → [soc: `watch_meals`] → [feed_pass: 과제 7]. `world.grad_table()`: `s.grad_tab`을 `{태그: {이름: 값}}`로, `reset_grad_table()`: 0으로. `summary`: `juv_enabled`·`juv_measure`일 때 `juv_births_recorded`(졸업+사망)·`grad_rate`·`par_near_frac`(= `GT_PAR_NEAR ÷ GT_PAR_ALIVE`, 분모 0이면 없음).
5. `bench.py`: `k.eat(...)` 뒤 `if ip[k.Q_JUV_COUNT]:`로 `juv_pass`(장치를 켠 세계 성능을 잴 때만).

**초록 확인**: `python3 -m pytest -q tests/test_teach.py tests/test_taste.py tests/test_soc_taste.py tests/test_stages.py`(soc 계획의 `build_grid` 호출 수 시험이 juv가 꺼진 세계라 안 깨지는 것도 이 시험으로 확인). 커밋: "부모의 가르침 3: juv_pass 곁 계수·졸업표·미숙기 사망".

---

## 과제 4. 유전자 `show_g`·`feed_g`: 변이
**바꾸는 파일**: `kkumteul/kernels.py`(`copy_genome`, `_reset_body`), `kkumteul/world.py`(summary), `tests/test_teach.py`.

**시험 먼저**(`tests/test_taste_gene.py`의 `lone`·`_natural_child` 패턴)
- `test_show_feed_gene_copied_when_off`: 장치가 꺼졌으면 자식이 부모 `show_g`·`feed_g`를 그대로 베낌(변이 없음), 재주입·대조 새끼도 베낌.
- `test_show_gene_mutates_and_clips_when_on`: `teach_enabled`(필요 설정 모두 켬)이면 평균이 부모 값 근처에서 `teach_gene_sigma`만큼 흩어지고 [0, 1] 밖으로 안 나감(많은 자식). 난수는 `S_TEACH_GENE`(자식 id, 0). `feed_enabled`면 `feed_g`가 `S_FEED_GENE`(자식 id, 0)로 같은 꼴.
- `test_gene_zero_for_initial_and_random_agent`: 처음 개체·`add_random_agent`는 0.
- `test_gene_checkpoint_roundtrip`: 0 아닌 값의 저장·복원(과제 1에서 확인한 것 + 자식 대에서).
- `test_summary_show_feed_mean`: `teach_enabled`일 때만 `world.summary()["show_g_mean"]`, `feed_enabled`일 때만 `feed_g_mean`.
- 빨강 확인.

**구현**
1. `copy_genome` 맨 끝(`soc_g` 블록 뒤): `g3 = float(s.show_g[i])`; `if mut and ip[Q_TEACH] != 0: g3 += fp[P_TEACH_GENE_SIGMA] × normal(seed, S_TEACH_GENE, cid, 0)`; [0, 1]로 자르고 `s.show_g[j] = g3`. `feed_g`도 `ip[Q_FEED]`·`S_FEED_GENE`·`P_FEED_GENE_SIGMA`로 같은 꼴. **앞의 난수 호출은 안 건드린다.**
2. `_reset_body`: `s.show_g[j] = 0.0`, `s.feed_g[j] = 0.0`(재주입 몸 초기화; 베끼는 경로는 `copy_genome`이 뒤에서 처리).
3. `world.summary`: `show_g_mean`·`feed_g_mean`.
4. 암수 설계 ⑥ 메모를 `copy_genome` 주석 한 줄로("`cross_genome`이 생기면 `show_g`·`feed_g`는 낱값 유전자로 변이 목록과 섞기 목록 둘 다에").

**초록 확인**: `python3 -m pytest -q tests/test_teach.py tests/test_taste_gene.py tests/test_taste.py`. 커밋: "부모의 가르침 4: show_g·feed_g 변이".

---

## 과제 5. 보여 주기 비용 `near` (`teach_pass`)
**바꾸는 파일**: `kkumteul/kernels.py`(`teach_pass`), `kkumteul/world.py`(`step`), `kkumteul/bench.py`, `tests/test_teach.py`.

**시험 먼저**(손으로 놓은 부모·새끼, `teach_pass`를 직접 부름)
- `test_teach_cost_kin_only_when_own_juvenile_near`: `kin`에서 어른 p의 **자기** 미숙기 새끼가 반지름 안일 때만 `teach_cost × clamp(show_g[p], 0, 1)`이 빠짐. 새끼가 없음·반지름 밖·죽음이면 0. 비친족 미숙기가 곁이면 `kin`에서 0, `any`에서 비용. 새끼가 둘이어도 어른 한 번(스텝당 한 번).
- `test_teach_cost_juvenile_not_payer`: 미숙기 개체(`age ≤ juv_steps`)는 시연자(비용 내는 쪽)가 아님.
- `test_teach_cost_modes`: `costonly`는 비용 있음, `drift`는 비용 없음, `knower`는 `taste_m[p, 독 칸] ≤ −juv_know_theta`인 부모만 냄.
- `test_teach_cost_ledger_and_tab`: 에너지 감소 = `L_BRAIN` 증가 = 태그별(`row = min(max(tag, 0), 7)`) `GT_TEACH_COST`. 비용 줄을 장부에서 빼면 실패.
- `test_teach_when_meal_skips_teach_pass`: `teach_cost_when="meal"`이면 `teach_pass`가 안 불림(비용은 과제 6의 `watch_meals`에서).
- `test_teach_pass_off_not_called`: 꺼진 세계·`teach_enabled=false`인 세계에서 호출 0.
- `test_teach_pass_not_requiring_grid`: `build_grid`를 안 부르고 손으로 놓은 쌍에서도 동작(격자 안 씀).
- 빨강 확인.

**구현**
1. `teach_pass(step, s, fp, ip)` 순차: `for q: s.tflag[q] = 0`. `mode = ip[Q_TEACH_MODE]`. 미숙기 새끼 i(`age ≤ steps`, 산)마다:
   - `any`: 모든 산 어른 j(`age > steps`)가 `soc_radius` 안이면 `tflag[j] = 1`.
   - 그 밖의 모드: `q = s.par_slot[i]`(`juv_pass`가 채움), `q ≥ 0`이고 어른(`age > steps`)이고 거리 ≤ r이고, 모드가 `knower`면 `_knows(s, q, kp, fp)`일 때 `tflag[q] = 1`.
   `for q in range(n)`(오름차순): `tflag[q] == 1`이고 `mode != TEACH_DRIFT`이고 `show_g[q] > 0`이면 `c = fp[P_TEACH_COST] × clamp(show_g[q], 0, 1)`, `s.energy[q] -= c`, `s.ledger[L_BRAIN] += c`, `s.grad_tab[row(tag[q]), GT_TEACH_COST] += c`.
2. `world.step`: `juv_pass` 바로 뒤에 `if self.ip[k.Q_TEACH] and self.ip[k.Q_TEACH_WHEN] == WHEN_NEAR: k.teach_pass(t, s, fp, ip)`. `bench.py`도 같은 꼴.

**초록 확인**: `python3 -m pytest -q tests/test_teach.py tests/test_taste.py tests/test_soc_taste.py`. 커밋: "부모의 가르침 5: 보여 주기 비용(near)".

---

## 과제 6. `watch_meals`의 boost·앎 도우미·`meal` 비용·`_kin_class`
**바꾸는 파일**: `kkumteul/kernels.py`(`watch_meals`, `eat`, `_kin_class`, `_note_know`), `tests/test_teach.py`, `tests/test_soc_taste.py`(리팩터링 회귀만).

**시험 먼저**(손 시험: `build_grid` 먼저, 이웃이 실제로 보이는 **양성 확인을 먼저**)
- `test_kin_class_matches_soc_inline`: `_kin_class(s, i, j)`가 0(남)·1(j가 i의 부모)·2(그 밖의 친족: i가 j의 부모, 같은 부모)를 돌려주고 사회 학습의 `n_seen_kin`(부모·자식·형제)이 리팩터링 뒤에도 같다(`tests/test_soc_taste.py`의 기존 시험 그대로 통과).
- `test_boost_kin_parent_only`: `kin`에서 자기 새끼 관찰자의 실제 속도가 `soc_rate + soc_g + teach_gain × show_g[j]`(클램프), 다른 미숙기에게는 `soc_rate + soc_g`만, 어른 관찰자에게도 그만(`any`에서는 곁의 미숙기 누구나 boost). 합이 1에서 잘림(`soc_g`가 이미 커도).
- `test_boost_zero_for_costonly_drift_and_nonknower`: `costonly`·`drift`는 boost 0, `knower`는 시연자가 독 모양을 모를 때 0·알 때 boost.
- `test_taught_counters`: boost가 붙은 관찰마다 관찰자 `n_taught`+1, 시연자가 그 스텝 독 모양을 안다면 `n_taught_knower`+1(**모드가 `knower`가 아니어도**), 부모 끼니를 보면 `n_seen_par`+1(독이면 `n_seen_par_poison`도).
- `test_meal_cost_only_when_boost_applies`: `teach_cost_when="meal"`에서 시연자 j가 먹고 자격 있는 새끼가 그것을 본 스텝에만 `teach_cost × show_g[j]`가 빠지고(관찰자 수와 무관, 한 스텝 한 번) 새끼가 없으면 0. `costonly`·`knower`는 boost 0이어도 "자격 있는 상황"이면 걷음(`drift`는 안 걷음). 에너지 감소 = `L_BRAIN` 증가 = `GT_TEACH_COST`.
- `test_know_helper_sources`: 자기 끼니로 기억이 `−juv_know_theta`에 닿으면 `know_src == 1`·`know_age == age`, 부모 끼니를 보고 닿으면 2, 형제 끼니 3, 비친족 4. 한 번만 기록(`know_age`가 이미 ≥ 0이면 안 덮음). 독 모양은 그 스텝 `good_kind`로(주기 경계 전후 한 번씩).
- `test_watch_meals_teach_off_unchanged`: `teach_enabled=false`면 `watch_meals`의 갱신이 사회 학습 시험과 같다(회귀).
- `test_boost_order_deterministic`: 여러 어른·여러 새끼가 겹친 같은 시나리오를 두 번 돌려 해시가 같음.
- 빨강 확인.

**구현**
1. `_kin_class(s, i, j)`: 사회 학습 `watch_meals`의 인라인 친족 판정(`parent[i] == ids[j]`, `parent[j] == ids[i]`, 같은 `parent ≥ 0`)을 함수로 뽑는다(`@njit(cache=True, inline="always")`). 사회 학습의 `n_seen_kin`은 `_kin_class != 0`으로 바꾼다(동작 불변).
2. `_note_know(s, i, kp, src, fp)`: `kp ≥ 0 and s.know_age[i] < 0 and _knows(...)`이면 `s.know_age[i] = s.age[i]`, `s.know_src[i] = src`. 호출은 `eat`의 맛 갱신 직후(`src = 1`, `ip[Q_JUV_COUNT] != 0`이고 미숙기일 때만)와 `watch_meals`의 갱신 직후(`src = 1 + _kin_class`: 부모 2, 그 밖의 친족 3, 남 4).
3. `watch_meals`: 시연자 j(`meal_o[j] != 0`)가 어른(`age[j] > steps`)이고 `ip[Q_TEACH]`일 때, 관찰자 i가 미숙기(`age[i] ≤ steps`)인 쌍마다: 자격 = `kin`·`costonly`·`drift`: `_kin_class == 1`, `knower`: `_kin_class == 1` and `_knows(s, j, kp, fp)`, `any`: 모두. `boost = 0`; `mode ∉ {costonly, drift}`이고 자격이면 `boost = fp[P_TEACH_GAIN] × clamp(show_g[j], 0, 1)`. `rate_i = clamp(soc_rate + soc_g[i] + boost, 0, 1)`로 갱신(사회 학습 갱신식 그대로 `rate_i`만 바꿈), `boost > 0`이면 `n_taught[i] += 1`, `_knows(j)`이면 `n_taught_knower[i] += 1`; `_kin_class == 1`이면 `n_seen_par[i] += 1`(독이면 `n_seen_par_poison`도). `teach_cost_when == meal`이고 자격 있는 관찰이 하나라도 있었던 시연자 j는 그 스텝 한 번 `c = teach_cost × clamp(show_g[j])`를 `energy`·`L_BRAIN`·`GT_TEACH_COST`에(`drift`는 제외). `kp`(독 모양 칸)는 `_poison_slot`.
   docstring에 boost·`meal` 비용 설명과 "순차 전용"을 보강한다. 적용 순서 = 시연자 칸 번호 오름차순(그대로).
4. `eat`는 사회 학습 `meal_*` 기록과 `watch`와 무관하게 위 앎 도우미만 더한다.

**초록 확인**: `python3 -m pytest -q tests/test_teach.py tests/test_soc_taste.py tests/test_taste.py`. 커밋: "부모의 가르침 6: boost·앎 도우미·meal 비용·친족 분류 함수".

---

## 과제 7. 먹이 대주기 (`feed_pass`)
**바꾸는 파일**: `kkumteul/kernels.py`(`feed_pass`), `kkumteul/world.py`(`step`), `kkumteul/bench.py`, `tests/test_teach.py`.

**시험 먼저**
- `test_feed_only_own_juvenile_within_touch`: 어른 p가 닿는 거리(`2 × agent_radius + 0.5`) 안 **자기** 미숙기 새끼 가운데 **가장 가까운 하나**에게만 `feed_rate × clamp(feed_g[p], 0, 1)`을 건넴. 비친족·어른·거리 밖은 못 받음. 새끼가 둘이면 가까운 쪽.
- `test_feed_min_energy_and_loss`: 주는 쪽 에너지 ≤ `give_min_energy`이면 못 줌(그 위에서는 주고 나서 `amt`만큼 아래로 내려갈 수 있음), 받는 양 = `(1 − give_loss) ×` 건넨 양, 손실 `L_LOST`, 태그별 `GT_FEED_GIVEN`.
- `test_feed_ledger_closes`: `feed_enabled`로 세계를 1,500스텝 돌려 `rel_residual < 1e-9`; 손실을 `L_LOST`에 안 더하면 실패.
- `test_feed_independent_of_teach`: `teach_enabled=false`·`soc_taste_enabled=false`여도 동작(`feed_enabled`는 `juv_enabled`만 요구).
- `test_feed_off_not_called`.
- 빨강 확인.

**구현**: `feed_pass(step, s, fp, ip)` 순차. `for q: s.jscr_i[q] = −1; s.jscr_d[q] = 1e30`. 미숙기 새끼 i마다 `q = s.par_slot[i]`; `q ≥ 0`, 어른, `feed_g[q] > 0`, 거리² ≤ 닿는 거리²이면 `d2 < jscr_d[q]`일 때 `jscr_d[q] = d2`, `jscr_i[q] = i`. 그 뒤 `for q`(오름차순): `jscr_i[q] ≥ 0`이고 `energy[q] > fp[P_FEED_MIN]`이면 `amt = fp[P_FEED_RATE] × clamp(feed_g[q], 0, 1)`, `got = amt × (1 − fp[P_FEED_LOSS])`, `energy[q] −= amt`, `energy[i] += got`, `L_LOST += amt − got`, `GT_FEED_GIVEN += amt`. `world.step`은 `watch_meals` 뒤 `if self.ip[k.Q_FEED]: k.feed_pass(...)`(소셜 `give` 앞). `stages.give`는 안 건드린다.

**초록 확인**: `python3 -m pytest -q tests/test_teach.py tests/test_stages.py`. 커밋: "부모의 가르침 7: 먹이 대주기".

---

## 과제 8. 끄는 규칙 정리·겨루기 출력·옛 도구
**바꾸는 파일**: `kkumteul/measure.py`, `wayfinder/stage2-4/taste_judge.py`, `wayfinder/stage2-4/hand_learner.py`(`TASTE_OFF`를 쓰도록 — 사회 학습 과제 5가 이미 바꿨으면 확인만), `tests/test_teach.py`, `tests/test_knockout.py`.

**시험 먼저**
- `test_flip_diff_and_hand_learner_taste_off_with_teach`: `flip_diff`·`hand_learner`의 끔 쪽 덮어쓰기가 `juv`·`soc`·`teach`를 켠 설정에서 설정 검사에 안 걸림(`knockout.TASTE_OFF`를 씀).
- `test_compete_reports_grad_by_tag`: `measure.compete`의 결과에 태그별(`0`=A, `1`=B) `grad_by_tag`(`{태그: {GT_NAMES: 값}}`, 시작 때 `world.reset_grad_table()`)가 있음(장치가 꺼진 세계는 전부 0). `measure.inject_agent`로 들어온 개체는 기록에 안 들어가 `GRAD + DEAD`가 태어난 새끼 수와 맞음(시험용 작은 세계에서 출생 로그와 대조).
- `test_assay_sides_reports_grad_by_tag_per_rep`: `assay_sides` 출력에 반복별 `grad_by_tag_list`(반복 순서대로 태그별 사전)가 있고 기존 키·값이 안 바뀜(기존 `assay` 시험 통과).
- 빨강 확인.

**구현**: `measure.compete`: 끝에 `world.grad_table()`을 `grad_by_tag`로 돌려줌, 시작에 `reset_grad_table()`. `assay_sides`: 반복별 목록 `grad_by_tag_list`를 더한다. 옛 도구(`taste_judge.py`·`hand_learner.py`)가 `TASTE_OFF`를 쓰는지 확인하고 아니면 고친다.

**초록 확인**: `python3 -m pytest -q tests/test_teach.py tests/test_knockout.py tests/test_taste_judge.py`. 커밋: "부모의 가르침 8: 겨루기 졸업표 출력·끄는 규칙 정리".

---

## 과제 9. 판정 도구와 체인 스크립트 초안
**바꾸는 파일**: `wayfinder/stage2-4/taste_gene_judge.py`, 새 `wayfinder/stage2-4/teach_gate.py`(순수 계산), 새 `tests/test_teach_judge.py`, 새 `wayfinder/stage2-4/teach_chain.sh`, `wayfinder/stage2-4/results.md`(예측 절 — 실행 **전에**), `docs/status.md`(막힌 점·다음 후보 칸 한 줄).

**시험 먼저**(`tests/test_teach_judge.py`; 시뮬레이션 없이 도는 것을 최대로)
- `test_zeroed_side_show_feed`: `zeroed_side(side, "show")`가 `show_g[:] = 0`(원본 배열 안 건드림), `"feed"`는 `feed_g`. `juv_enabled`·`teach_enabled`가 꺼진 설정이면 `ValueError`("부모의 가르침 장치가 꺼진 체크포인트"), `feed`는 `feed_enabled`. 오류 문구의 허용 이름에 show·feed 추가.
- `test_apply_options_teach_d`: `--teach-d D`가 B쪽 `show_g[:] = D`(`feed`는 `feed_g`), `--set`(여러 번)이 옆 설정에 적용(사회 학습의 `--set`·`--no-individual`·`--no-neutral` 재사용, 뒤의 `--set`이 앞을 덮어씀).
- `test_net_per_birth_hand_values`(`teach_gate`): 합성 `grad_by_tag`(A·B의 `grad`·`dead`·`e_grad`·`teach_cost`)에서 `G = e_grad_B/(grad_B + dead_B) − e_grad_A/(grad_A + dead_A)`, `C`, `N = G − C`가 손셈과 같고 반복별 구간(`measure.t95`, df = n − 1)이 맞다. `X = C ÷ (teach_cost × d)`, 손익분기 `c* = G ÷ (X × d)`.
- `test_gate_world`: 관문 −1a(켠 무리 ≥ 끈의 0.9·장부 통과·재주입 급증 없음), −1b(`par_near_frac × juv_steps × 어른 끼니율 × 독 비율 ≥ 0.5`), −1c(mult 3: 무리 ≥ 0.9·졸업률 ≥ 끈의 0.5)가 합성 값에서 각각 통과·실패로 갈림. 종료 코드: 0 통과, 3 세계 못 섬(−1a·−1c), 4 곁 부족(−1b).
- `test_detectability_ratio`: 관문 −2: 예상 순이득(`N`)과 검출 한계(음성 대조 칸 반복별 순이득 차의 sd × t95 ÷ √n)의 비가 1 아래인 d는 β를 쓰지 않는다. 관문 −2b: `N × 순이득 ÷ 17 ≥ 1`(N=무리 크기 입력)이 합성 값에서 갈림.
- `test_gate_alpha_beta`: 합성 결과로 "작동 + 이득"·"작동 + 이득을 가를 수 없음"·"작동 아님"·"도구 깨짐"(음성 대조 실패·양쪽 전멸 ≥ 3판) 네 경우. α: `kin`의 (가) 미숙기 독 비율 차(A−B)가 `costonly`보다 Welch 양이고 ≥ 0.01, (나) 독 모양 기억 깊이 차 양(6판 중 5판). β: `N` 구간 아래끝 > 0, `kin`−`costonly`의 `G` 차 Welch 양, 음성 대조 통과. 정보 비싼 세계 칸(`-exp-`)은 −1c 통과일 때만 따로 판정.
- `test_gate_combines_tags_exit_codes`: 한 바탕이라도 통과면 0, 모두 "가를 수 없음"이면 3, 통과 없이 깨진 바탕이 있으면 4.
- `test_prereq_soc`: 사회 학습 판정 0의 `j-dose-outcome-d0.25-<태그>.json`·`d1.0`의 `first10_by_tag`(A−B)가 Welch로 양이면 0, 아니면 3(진행/건너뜀).
- `test_observe_window_reports_grad`: 작은 세계 `observe --steps N`이 창 시작에 `reset_grad_table()`하고 졸업률·`par_near_frac`·`n_seen_par`·`show_g` 평균·분포를 냄(장치를 켠 설정일 때만).
- 빨강 확인.

**구현**
1. `teach_gate.py`(순수): 위 계산(`net_per_birth`, `breakeven`, `gate_world`, `detectability`, `gate_alpha_beta`, `gate_tags`).
2. `taste_gene_judge.py`: `zeroed_side`에 `show`·`feed` 분기, `rate-zero`에 `--gene {taste,eta,soc,show,feed}`, `--teach-d FLOAT`(B쪽 `show_g`/`feed_g`를 d로); 출력에 적용한 옵션을 적는다. `observe`에 `--steps N` 창(시작 `reset_grad_table`)과 졸업표·곁·`n_seen_par`·`show_g` 평균·분위수. 하위 명령 `gate-world --dir <폴더> --tags a b`(−1), `prereq-soc --dir <soc 결과 폴더> --tags a b`(선결), `gate-teach --dir <폴더> --tags a b`(−2·α·β·손익분기 비용). 원자료 이름 규칙은 아래 체인.
3. **문서의 실제 사용 예를 시간 제한을 걸고 직접 돌린다**(시험 통과만으로 "됐다"고 하지 않는다):
   `PYTHONPATH=. timeout 600 python3 wayfinder/stage2-4/taste_gene_judge.py rate-zero <작은 세계로 만든 juv·soc·teach 켠 체크포인트> --gene show --teach-d 1.0 --n 20 --steps 400 --reps 2 --no-neutral --json <임시 폴더>/x.json`
   (체크포인트는 시험 보조 함수로 작은 세계를 저장해 만든다. 서버의 큰 실행은 건드리지 않는다.) 결과 JSON에 `grad_by_tag_list`가 있고 `gate-teach`가 합성이 아닌 이 파일을 읽어 오류 없이 끝나는지까지 본다.
4. 체인 스크립트(아래 초안) — `bash -n`과 `DRY=1`로 명령이 맞는지 본다.
5. `results.md`에 "부모의 가르침 — 예측" 절을 설계 6절 예측 표 그대로(실행 전 날짜·시각 표시)로 적고, 판정 규칙(관문 −2·−2b·−1a/b/c·α·β·손익분기·진화 판정 문장·단서 문턱 `max(불활성 sd, 0.10)` — 재확인 결과 0.09가 0.10으로 바뀜)을 한 번 더 옮긴다. 문턱은 실행 중에 바꾸지 않는다. 중립 모의(N=500·400세대·σ=0.02·0 자름)의 0.30 ± 0.09는 한 번짜리 스크립트로 재확인해 값을 적는다.
6. 전체 시험을 한 번: `python3 -m pytest -q`(메모리 상한 도구 `--max 2G`, `NUMBA_NUM_THREADS=2`, `nice`). 실패는 이 과제에서 고친다.
7. 체인을 돌릴지는 서버 상황(사회 학습 체인과 동시에 안 돎)을 보고 정한다. **한 번에 하나, 2스레드, 메모리 상한 1,500M, `nice -n 15`.**

### 체인 스크립트 초안 (`wayfinder/stage2-4/teach_chain.sh`)
```bash
#!/bin/bash
# 부모의 학습 보조 투자("가르침" 후보) 실험 체인(사용자 새 방향 3번). 설계: docs/superpowers/specs/2026-11-26-parent-teaching-design.md 둘째 판.
# 예측·문턱: wayfinder/stage2-4/results.md "부모의 가르침 — 예측"(실행 전에 적어 둔다).
# 앞 실험(사회 학습 체인 포함)이 끝난 뒤 한 번에 하나(2스레드, nice).
# 순서: 선결(사회 학습) → 세계 점검(−1) → 판정 0(용량 반응) → 관문(−2·α·β) → 진화 갈래 → 갈래별 판정.
#
#   ROOT=<레포 원래 폴더> CODE=<커밋한 코드 사본 폴더> [WAIT_FOR=<pgrep -f 패턴>] [DRY=1] \
#     <메모리 상한 도구> --max 1500M --bg teach-chain --log "$ROOT/runs/teach-exp/chain.log" -- bash wayfinder/stage2-4/teach_chain.sh
# 상한은 묶음 전체에 걸린다. 갈래·판정마다 done/json 표시가 있어 다시 불러도 이어 간다.
# 시간 어림(서버 부하에 따라 달라짐, 추정): 세계 점검 약 20분, 판정 0 약 3시간, 진화 갈래 열두 개 약 3.5시간, 갈래별 판정 약 1.5시간.
# 관문에서 닫히면 판정 0까지 약 3.5시간으로 끝난다.
set -u
R=${ROOT:?ROOT=레포 원래 폴더}
C=${CODE:?CODE=코드 사본 폴더}
E=$R/runs/teach-exp
SOCE=$R/runs/soc-exp            # 사회 학습 판정 0 원자료(선결 읽기)
PY=$R/.venv/bin/python
J=wayfinder/stage2-4/taste_gene_judge.py
START=ckpt_000002600000.npz     # 바탕: 개인 학습 갈래 g-p500-a/b 260만
W_END=ckpt_000002630000.npz     # 세계 점검 갈래(3만 스텝) 끝
END=ckpt_000002900000.npz       # 진화 갈래(30만 스텝) 끝
export NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-2} PYTHONPATH=.
mkdir -p "$E"
note() { echo "$(TZ=America/New_York date '+%m-%d %H:%M ET') $*" | tee -a "$E/progress.txt"; }
runit() {  # DRY=1이면 명령만 찍는다
  if [ -n "${DRY:-}" ]; then echo "DRY: $*"; return 0; fi
  "$@"
}

if [ -n "${WAIT_FOR:-}" ]; then
  while pgrep -f "$WAIT_FOR" >/dev/null; do sleep 30; done
  note "앞 실험 끝남 확인 — 체인 시작"
fi
cd "$C" || exit 1

base_ckpt() { echo "$R/runs/g-p500-$1/checkpoints/$START"; }   # 바탕 태그 a|b

pre_check() {  # 체크포인트 설정이 설계가 기대는 세계인지(다르면 멈춘다)
  local tag=$1 ck
  ck=$(base_ckpt "$tag")
  [ -f "$ck" ] || { note "바탕 체크포인트 없음: $ck"; return 1; }
  "$PY" - "${ck%.npz}.json" <<'PYEOF' || return 1
import json, sys
c = json.load(open(sys.argv[1]))["config"]
want = {"food_kinds_enabled": True, "kind_flip_period": 500, "taste_enabled": True,
        "taste_gene_enabled": True, "poison": 0.3, "taste_inherit": 0.0, "food_cap_replace": True,
        "taste_rate": 0.5, "social_enabled": False}
bad = {k: (c.get(k), v) for k, v in want.items() if c.get(k) != v}
if bad:
    print("체크포인트 설정이 기대와 다름:", bad)
    sys.exit(1)
PYEOF
}

# ---- 세계 점검(판정 −1): 같은 체크포인트에서 갈라 3만 스텝, 끝에서 5,000스텝 창을 읽는다
wbranch() {  # 태그 이름 설정...   (이름: off=juv_measure만, on=미숙기 기본, exp=독 배수 3)
  local tag=$1 name=$2; shift 2
  local dir=$R/runs/t-w-$name-$tag ck out=$E/w-$name-$tag
  local sets=(); for s in "$@"; do sets+=(--set "$s"); done
  if [ ! -f "$dir/done" ]; then
    note "세계 $name-$tag 시작 ($*)"
    runit nice -n 15 "$PY" -m kkumteul.run branch --from "$(base_ckpt "$tag")" --run "$dir" --steps 30000 --quiet \
      "${sets[@]}" > "$dir.log" 2>&1
    runit touch "$dir/done"
  fi
  ck=$dir/checkpoints/$W_END
  [ -s "$out.json" ] && return 0
  [ -f "$ck" ] || { note "세계 $name-$tag: 체크포인트 없음 $ck"; return 0; }
  runit nice -n 15 "$PY" $J observe "$ck" --steps 5000 --json "$out.json" > "$out.log" 2>&1
  note "세계 $name-$tag 읽음: $(tail -1 "$out.log" 2>/dev/null | cut -c1-160)"
}

# ---- 용량 반응(판정 0)
dose() {  # 바탕태그 이름 d [판정 옵션...]  (이름 예: kin-d0.5). 기본 옵션 뒤의 --set이 앞을 덮어쓴다
  local tag=$1 name=$2 d=$3; shift 3
  local ck out
  ck=$(base_ckpt "$tag"); out=$E/j-dose-$name-$tag
  [ -s "$out.json" ] && { note "용량 반응 $name-$tag 이미 있음"; return 0; }
  note "용량 반응 $name-$tag 시작"
  runit nice -n 15 "$PY" $J rate-zero "$ck" --gene "${GENE:-show}" --teach-d "$d" \
    --set juv_enabled=true --set soc_taste_enabled=true --set soc_taste_mode=outcome --set soc_rate=0.05 \
    --set teach_enabled=true --set teach_mode=kin "$@" --json "$out.json" > "$out.log" 2>&1
  local rc=$?
  note "용량 반응 $name-$tag 끝 ($rc): $(tail -1 "$out.log" 2>/dev/null | cut -c1-160)"
}

dose_all() {  # 바탕태그 하나의 판정 0 열아홉 칸. 중립은 첫 칸에서만(min_detectable)
  local t=$1
  dose "$t" kin-d0.1 0.1
  for d in 0.5 1.0; do dose "$t" kin-d$d $d --no-neutral; done
  for d in 0.1 0.5 1.0; do dose "$t" costonly-d$d $d --set teach_mode=costonly --no-neutral; done
  for d in 0.5 1.0; do dose "$t" any-d$d $d --set teach_mode=any --no-neutral; done
  dose "$t" knower-d1.0 1.0 --set teach_mode=knower --no-neutral
  dose "$t" off-d1.0 1.0 --set teach_enabled=false --no-neutral                       # 음성 대조
  dose "$t" nosoc-kin-d1.0 1.0 --soc-none --no-neutral                                 # 보고 배우기 없음(양쪽 soc_g = −soc_rate)
  for d in 0.5 1.0; do dose "$t" kin-nocost-d$d $d --set teach_cost=0 --no-neutral; done
  for d in 0.5 1.0; do dose "$t" kin-meal-d$d $d --set teach_cost_when=meal --no-neutral; done
  dose "$t" costonly-meal-d1.0 1.0 --set teach_mode=costonly --set teach_cost_when=meal --no-neutral
  GENE=feed dose "$t" feed-d1.0 1.0 --set teach_enabled=false --set feed_enabled=true --no-neutral   # 돌봄(feed_g를 d로)
  if [ "${EXP_OK:-0}" = 1 ]; then   # 관문 −1c 통과 때만(정보 비싼 세계)
    dose "$t" kin-exp-d1.0 1.0 --set juv_poison_mult=3 --no-neutral
    dose "$t" costonly-exp-d1.0 1.0 --set teach_mode=costonly --set juv_poison_mult=3 --no-neutral
  fi
}

run_branch() {  # 이름 씨앗(a=바탕 a·같은 흐름, b=바탕 b, c=바탕 a·--new-stream) 설정...
  local name=$1 seed=$2; shift 2
  local base=$seed; [ "$seed" = c ] && base=a
  local dir=$R/runs/t-$name-$seed
  [ -f "$dir/done" ] && { note "$name-$seed 이미 끝남, 건너뜀"; return 0; }
  [ -f "$dir/run.json" ] && { note "$name-$seed 중간에 끊긴 흔적 — 건너뜀"; return 0; }
  local extra=(); [ "$seed" = c ] && extra=(--new-stream)
  local sets=(); for s in "$@"; do sets+=(--set "$s"); done
  note "$name-$seed 시작 ($*)"
  runit nice -n 15 "$PY" -m kkumteul.run branch --from "$(base_ckpt "$base")" --run "$dir" --steps 300000 --quiet \
    "${extra[@]}" "${sets[@]}" > "$dir.log" 2>&1
  local rc=$?
  runit touch "$dir/done"
  note "$name-$seed 끝 (종료 코드 $rc): $(tail -1 "$dir.log" 2>/dev/null | cut -c1-160)"
}

judge() {  # 갈래-씨앗 무엇(kin|costonly|any|ctrl|nocost|nosoc|obs)
  local tag=$1 what=$2
  local ck=$R/runs/t-$tag/checkpoints/$END out=$E/j-$what-$tag
  [ -s "$out.json" ] && { note "판정 $what-$tag 이미 있음"; return 0; }
  [ -f "$ck" ] || { note "판정 $what-$tag: 체크포인트 없음 $ck"; return 0; }
  note "판정 $what-$tag 시작"
  local opts=()
  case $what in
    kin)      ;;
    costonly) opts=(--set teach_mode=costonly) ;;
    any)      opts=(--set teach_mode=any) ;;
    ctrl)     opts=(--set teach_enabled=false) ;;            # 음성 대조
    nocost)   opts=(--set teach_cost=0) ;;
    nosoc)    opts=(--soc-none) ;;
  esac
  if [ "$what" = obs ]; then
    runit nice -n 15 "$PY" $J observe "$ck" --steps 5000 --json "$out.json" > "$out.log" 2>&1
  else
    runit nice -n 15 "$PY" $J rate-zero "$ck" --gene show "${opts[@]}" --json "$out.json" > "$out.log" 2>&1
  fi
  local rc=$?
  note "판정 $what-$tag 끝 ($rc): $(tail -1 "$out.log" 2>/dev/null | cut -c1-160)"
}

pre_check a && pre_check b || { note "바탕 확인 실패 — 체인 중단"; exit 1; }

# 0. 선결: 사회 학습 판정 0의 직접 지표가 작동이어야 kin·knower·any를 돈다(아니면 feed·세계 점검만)
SOC_OK=1
if [ -n "${DRY:-}" ]; then
  echo "DRY: $PY $J prereq-soc --dir $SOCE --tags a b"
else
  "$PY" $J prereq-soc --dir "$SOCE" --tags a b > "$E/prereq.json" 2> "$E/prereq.log" || SOC_OK=0
fi
note "선결(사회 학습 직접 지표) SOC_OK=$SOC_OK"

# 1. 세계 점검(−1): 끔(측정만)·켬·정보 비싼 세계, 씨앗 a·b
G="preset=taste taste_inherit=0 taste_gene_enabled=true kind_flip_period=500"
for t in a b; do
  wbranch $t off  $G juv_measure=true
  wbranch $t on   $G juv_enabled=true
  wbranch $t exp  $G juv_enabled=true juv_poison_mult=3
done
runit "$PY" $J gate-world --dir "$E" --tags a b > "$E/gate-world.json" 2> "$E/gate-world.log"
rc=$?
note "관문 −1 종료 코드 $rc (0 통과, 3 세계 못 섬, 4 곁 부족): $(tr -d '\n ' < "$E/gate-world.json" 2>/dev/null | cut -c1-200)"
EXP_OK=$("$PY" -c "import json,sys; print(int(json.load(open('$E/gate-world.json')).get('exp_ok', False)))" 2>/dev/null || echo 0)
export EXP_OK
if [ "$rc" != 0 ] && [ -z "${DRY:-}" ]; then
  note "관문 −1 못 넘음 — 판정 0을 돌리지 않고 닫는다(설계 6절: 판 바꿈은 문서를 고쳐 쓰고 정한다)"
  exit 0
fi

# 2. 판정 0: 용량 반응(진화 전). 선결이 안 서면 feed 칸 외에는 건너뜀
for t in a b; do
  if [ "$SOC_OK" = 1 ]; then dose_all $t
  else dose $t off-d1.0 1.0 --set teach_enabled=false --no-neutral
       GENE=feed dose $t feed-d1.0 1.0 --set teach_enabled=false --set feed_enabled=true --no-neutral; fi
done
runit "$PY" $J gate-teach --dir "$E" --tags a b > "$E/gate.json" 2> "$E/gate.log"
rc=$?
note "관문(−2·α·β) 종료 코드 $rc (0 통과, 3 가를 수 없음, 4 도구 깨짐): $(tr -d '\n ' < "$E/gate.json" 2>/dev/null | cut -c1-200)"
if [ "$rc" != 0 ] && [ -z "${DRY:-}" ]; then
  note "관문 못 넘음 — 진화 갈래를 돌리지 않고 닫는다(가를 수 없음·도구 깨짐은 결과 문서에 그대로 적는다)"
  exit 0
fi

# 3. 진화 갈래: 같은 체크포인트에서 갈라 30만 스텝. kin·drift·off는 a·b·c, any·feed는 a·b
S="juv_enabled=true soc_taste_enabled=true soc_taste_mode=outcome soc_rate=0.05"
for s in a b c; do
  run_branch kin   $s $G $S teach_enabled=true teach_mode=kin
  run_branch drift $s $G $S teach_enabled=true teach_mode=drift
  run_branch off   $s $G $S teach_enabled=false
done
for s in a b; do
  run_branch any  $s $G $S teach_enabled=true teach_mode=any
  run_branch feed $s $G $S teach_enabled=false feed_enabled=true
done
# 4. 판정: kin 갈래는 판정 나(kin·costonly·any·음성 대조·비용 없음·보고 배우기 없음)+단서, 나머지는 단서(observe)만
for s in a b; do
  for w in kin costonly any ctrl nocost nosoc obs; do judge kin-$s $w; done
done
for s in a b c; do
  judge drift-$s obs
  judge off-$s obs
done
judge kin-c obs
for s in a b; do judge any-$s obs; judge feed-$s obs; done
# 5. 시간이 남으면: 정보 비싼 세계(−1c 통과 때)에서 kin·drift 한 번 더
if [ "${EXP_OK:-0}" = 1 ]; then
  for s in a b; do
    run_branch kin-exp   $s $G $S teach_enabled=true teach_mode=kin juv_poison_mult=3
    run_branch drift-exp $s $G $S teach_enabled=true teach_mode=drift juv_poison_mult=3
    judge kin-exp-$s kin
    judge kin-exp-$s obs
    judge drift-exp-$s obs
  done
fi
note "체인 끝"
```
(체인은 갈래 폴더 `runs/t-<갈래>-<씨앗>`, 판정 원자료 `runs/teach-exp/j-*.json`·`w-*.json`을 쓴다 — 모두 `runs/` 아래라 git에 안 올라간다. 갈래마다 `done` 표시가 있어 끊겨도 다시 불러 이어 간다. `--soc-none`(양쪽 `soc_g = −soc_rate`)·`--gene feed`는 B쪽 `feed_g`를 `--teach-d` 값으로 둔다. `--soc-none`은 과제 9 구현에서 `rate-zero`에 더하는 옵션이다.)

8. 체인 결과를 읽을 때: 판정 문장은 설계 6절 그대로 — "같은 끝 무리에서 `kin`의 **순이득 `N` 구간 아래끝이 0 위, `costonly`와 `G` 차가 Welch로 양, 직접 지표 (가) 차가 `costonly`보다 0.01 이상, 음성 대조 통과, 두 씨앗 같은 쪽**"일 때만 지지(뜻: 부모의 학습 보조 투자가 이득이 된다 — ①은 부여).
   결과 문서(`wayfinder/stage2-4/results.md`)에 판정을 적을 때 `docs/status.md`의 관찰·기각 칸과 마지막 갱신일도 같이 고친다(관찰에는 잰 값만, 해석에는 추정만). `observe`의 `n_seen_par`로 설계의 셈(새끼당 본 부모 끼니 0.5~2.3)을 실측으로 확인한다.
   커밋: "부모의 가르침 9: 판정 도구와 체인 스크립트 초안".

---

## 과제 요약
0. 전제 확인(사회 학습 과제 1~5 커밋·초록, 번호 읽기, 바탕 체크포인트)
1. 설정·난수·필드·옛 호환(동작 변화 없음, `know_age` −1)
2. 미숙기: 먹이 효율(들판 알만)·속도 상한·독 배수, 손잡이 분리 시험
3. `juv_pass`: 곁 머무름 계수기(격자 없이)·졸업표(태그 접힘)·미숙기 사망
4. 유전자 `show_g`·`feed_g` 변이
5. 보여 주기 비용 `near`(`teach_pass`)
6. `watch_meals`의 boost·앎 도우미·`meal` 비용·`_kin_class`
7. 먹이 대주기(`feed_pass`)
8. 겨루기 졸업표 출력·끄는 규칙 정리
9. 판정 도구(`teach_gate.py`·관문)와 체인 스크립트 초안, 예측 기록
