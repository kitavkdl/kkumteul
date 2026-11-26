# 사회 학습 구현 계획 (2026-11-26 밤)

설계: `docs/superpowers/specs/2026-11-07-social-learning-design.md` 둘째 판(검토 두 갈래 반영). 이 계획은 그 설계를 **과제 일곱 개**로 쪼개 한 명이 순서대로 구현할 수 있게 한다.
과제마다 TDD: **시험 먼저 → 빨강 확인 → 구현 → 초록 확인 → 커밋**. 설정 값은 전부 임시 결정(`config.py`에서만 바꾼다).

공통 규칙(과제마다 지킨다)
- 시험 명령: `python3 -m pytest -q tests/<파일>`. 서버에서는 메모리 상한 도구로 감싸고(`--max 2G`), `NUMBA_NUM_THREADS=2`, `nice`로 낮춘다.
  새 커널·`kernels.py`·`config.py`를 고치면 numba 캐시가 지문(`_cache._SOURCES`)으로 새로 지어지므로 첫 실행이 느리다.
- 박은 해시를 쓰는 기존 시험이 그대로 통과해야 한다: `tests/test_taste.py`, `tests/test_stages.py`, `tests/test_harsh.py`, `tests/test_prey.py`(소수 시험은 과제 끝에 돌리고, 전체는 과제 7에서).
- 에너지가 생기는 길은 먹기와 명시적 주입뿐이다 — 새 비용은 에너지를 빼고 `L_BRAIN`에 더하기만 한다. 에너지 장부 시험이 통과해야 한다.
- `rng.py`의 용도 번호는 뒤에 덧붙인다(32, 33). 기존 호출 순서를 바꾸지 않는다.
- 공개 레포: 문서·주석에 로컬 절대 경로·서버 구성·다른 프로젝트 이야기를 쓰지 않는다(레포 상대 경로만, 체인 스크립트 안의 경로는 변수).
- 고친 파일만 `git add`, 커밋하면 바로 push(프로젝트 규칙).

전제 확인(과제 0, 코드 안 고침): 이 계획이 쓰는 겨루기 마당의 독 순서 잇기(`measure.kind_aligned`, `Config.kind_seed`·`kind_step0`, `kernels.cfg_good_kind`)가 커밋돼 있어야 한다
(`grep -n "def kind_aligned" kkumteul/measure.py`; 없으면 그 변경을 먼저 합친다 — 독이 오래 안 바뀌는 바탕에서 겨루기가 깨지는 것을 막는 고침). `P_*`·`Q_*` 칸은 그 시점의 **맨 뒤**에 덧붙인다(줄 번호는 이 문서의 어림일 뿐).
바탕 체크포인트가 있는지도 본다: `ls runs/g-p500-a/checkpoints runs/g-p500-b/checkpoints`에 `ckpt_000002600000.npz`가 둘 다 있어야 한다(없으면 체인을 돌리지 않는다).

---

## 과제 1. 설정·난수·필드·옛 호환 (동작 변화 없음)
**바꾸는 파일**: `kkumteul/config.py`, `kkumteul/rng.py`, `kkumteul/kernels.py`, `kkumteul/measure.py`, `kkumteul/pack.py`, `tests/test_taste_gene.py`, 새 `tests/test_soc_taste.py`.

**시험 먼저**
- `tests/test_soc_taste.py`
  - `test_soc_config_defaults_and_checks`: 기본 꺼짐(`soc_taste_enabled=False`), `soc_taste_mode="outcome"`, `soc_rate=0.05`, `soc_gene_sigma=0.02`, `soc_cost=0.0005`, `soc_radius=8.0`.
    맛 기억이 꺼진 설정(`small("taste_size", soc_taste_enabled=True)`)에서 켜면 `ValueError`, `soc_radius > cell`·`soc_radius <= 0`·`soc_rate`가 [0, 1] 밖·`soc_cost < 0`·`soc_gene_sigma < 0`·모르는 모드 이름은 `ValueError`.
  - `test_soc_mode_string_override`: `cfg.with_overrides(["soc_taste_enabled=true", "soc_taste_mode=blind"])`가 문자열 값을 받는다(지금 `with_overrides`는 bool·int 아니면 `float(v)`라 문자열이 터진다).
  - `test_soc_fields_registered`: `AGENT_FIELDS`에서 `soc_g`가 `taste_g` **바로 뒤**, 저장 필드이고 `HASH_DEFAULTS["soc_g"] == 0`. `meal_k`·`meal_o`·`n_seen`·`n_seen_poison`·`n_seen_kin`·`seen_age_sum`은 저장 안 함. 새 `alloc_state`가 모두 0.
  - `test_soc_in_genome`: `"soc_g" in measure.GENOME`, `measure.OPTIONAL_GENOME == ("taste_g", "soc_g")`.
  - `test_pack_reads_bundle_without_soc_g`, `test_pack_reads_bundle_without_taste_and_soc_g`: 만든 묶음에서 해당 배열만 빼고(`np.savez`로 다시 저장) `pack.read`가 0으로 채워 읽는다.
  - `test_soc_off_world_hash_unchanged`: 처음부터 꺼진 세계(`taste_world("taste")` 등)를 돌린 해시가 기존 박은 해시 시험과 같다 — 새 필드가 전부 기본값이라 건너뛰어진다는 것을 기존 시험(`test_taste.py:29-30`, `:282`)의 통과로 확인.
  - `test_old_checkpoint_loads_soc_g_zero`: 체크포인트 저장·복원이 `soc_g` 값을 보존하고(0 아닌 값을 넣고), `soc_g`가 없는 옛 `.npz`는 0으로 읽힌다(`HASH_DEFAULTS` 경로 — `load_arrays`가 건너뛰고 `alloc_state`가 채운다).
- `tests/test_taste_gene.py`의 `test_new_rng_stream_appended`를 일반화한다: 모든 `S_*` 값이 서로 다르고(`len(set(vals)) == len(vals)`), `S_TASTE_GENE == 31`, `S_SOC_GENE == 32`, `S_SOC_BLIND == 33`.
- `test_soc_blind_key_unique`(`tests/test_soc_taste.py`): 묶음 함수(과제 4가 쓰는 `oid × 2**32 + eid`)가 서로 다른 (관찰자, 시연자) 쌍에서 안 겹친다(작은 id 범위 전수 + 경계값 `2**32 - 1`).
- **빨강 확인**: `python3 -m pytest -q tests/test_soc_taste.py tests/test_taste_gene.py` — 위 시험이 속성·키 없음으로 실패하는 것을 본다.

**구현**
1. `config.py`: 설정 여섯(`soc_taste_enabled`, `soc_taste_mode`(문자열), `soc_rate`, `soc_gene_sigma`, `soc_cost`, `soc_radius`)과 모드 상수(`SOC_OUTCOME=0, SOC_BLIND=1, SOC_INERT=2, SOC_WHAT=3`, 이름↔번호 사전).
   `validate()`: 위 검사. `with_overrides`: `types[k] == "str"`이면 문자열 그대로(타입 문자열은 `from __future__ import annotations` 때문에 `'str'`).
2. `rng.py`: `S_SOC_GENE = 32`, `S_SOC_BLIND = 33`(주석: "보고 배우기 속도 유전자의 변이", "눈가림 대조의 모양 동전").
3. `kernels.py`: `AGENT_FIELDS`에 `soc_g`(float32, 저장, `taste_g` 바로 뒤), `meal_k`(int8), `meal_o`(float32), `n_seen`·`n_seen_poison`·`n_seen_kin`(int32), `seen_age_sum`(int64)(전부 저장 안 함, `n_poison10` 뒤).
   `HASH_DEFAULTS`에 `"soc_g": 0`. `P_*`: `P_SOC_RATE, P_SOC_COST, P_SOC_GENE_SIGMA, P_SOC_RADIUS`를 `N_FP` 앞(맨 뒤)에 덧붙이고 `range(...)` 개수를 +4. `Q_*`: `Q_SOC, Q_SOC_MODE`를 `N_IP` 앞에 덧붙이고 `range(...)` 개수를 +2.
   `pack_params`: `soc_on = cfg.soc_taste_enabled and taste_on`(`taste_on`은 이미 있는 맛 블록 조건), `ip[Q_SOC] = int(soc_on)`, `ip[Q_SOC_MODE] = 모드 번호`, `fp[P_SOC_*]`.
4. `measure.py`: `GENOME`에 `"soc_g"` 덧붙임, `OPTIONAL_GENOME = ("taste_g", "soc_g")`. `pack.py` `read`: `taste_g` 한 줄을 `for name in OPTIONAL_GENOME: if name not in genomes and "gen" in genomes: 0으로 채움`으로 일반화.
   `measure.inject_agent`의 0 초기화 줄에 `meal_*`·`n_seen*`·`seen_age_sum`을 더한다(과제 3에서 필드가 쓰이기 전에 미리).

**초록 확인**: `python3 -m pytest -q tests/test_soc_taste.py tests/test_taste_gene.py tests/test_taste.py tests/test_knockout.py` 통과(`test_knockout.py`는 이 과제에서 아직 안 고침 — 과제 5에서). 커밋: "사회 학습 1: 설정·난수·필드·옛 호환(동작 변화 없음)".

---

## 과제 2. 보고 배우는 속도 유전자: 변이·비용
**바꾸는 파일**: `kkumteul/kernels.py`(`copy_genome`, `_reset_body`, `eat`), `kkumteul/world.py`(summary), `tests/test_soc_taste.py`.

**시험 먼저**(`tests/test_taste_gene.py`의 `lone`·`_eat_n`·`_natural_child` 패턴을 본뜬다; 비용은 `eat`만 직접 불러 잰다 — 세계를 돌리면 에너지 감각 입력이 달라져 이동 비용이 갈라진다)
- `test_soc_gene_copied_when_off`: 장치 꺼짐이면 자식이 부모 `soc_g`를 그대로 베낌(변이 없음), 재주입·대조 새끼도 베낌.
- `test_soc_gene_mutates_and_clips_when_on`: 켜짐이면 평균이 부모 값 근처에서 `soc_gene_sigma`만큼 흩어지고, 범위가 `[−soc_rate, 1 − soc_rate]` 밖으로 안 나간다(많은 자식으로). 난수는 `S_SOC_GENE`(자식 id, 0).
- `test_soc_gene_zero_for_initial_and_random_agent`: 처음 개체·`add_random_agent`는 0.
- `test_soc_cost_per_step_and_none_when_off`: `soc_cost=0.01`, `soc_g=0.45`(실제 속도 0.05 + 0.45 = 0.5)면 스텝당 0.005 더 듦, 장치 꺼짐이면 0. 에너지 감소 = 장부 `L_BRAIN` 증가.
- `test_soc_cost_zero_rate_is_free`: 실제 속도 0이면 비용 0.
- `test_soc_cost_paid_by_full_agent`: 배불러(`energy >= e_max`) `continue`하는 개체도 비용을 냄(비용이 `eat` 맨 위 산 개체 루프에 있음을 고정).
- `test_soc_cost_independent_of_taste_gene`: `taste_gene_enabled`가 꺼져 있어도 `soc_taste_enabled`면 `soc_cost`를 냄(두 장치가 따로 비용을 냄).
- `test_ledger_closes_with_soc_cost`: `soc_cost=0.01`로 세계를 1,500스텝 돌려 `rel_residual(w) < 1e-9`. 비용 줄을 장부에 안 더하면 실패한다(구현 뒤 한 번 줄을 지워 실패하는 것을 보고 되돌린다).
- `test_soc_gene_checkpoint_roundtrip`: 0 아닌 `soc_g`의 저장·복원.
- `test_summary_soc_rate_mean`: `soc_taste_enabled`일 때만 `world.summary()["soc_rate_mean"]`가 있고 `clip(soc_rate + soc_g, 0, 1)`의 평균과 같음.
- 빨강 확인: 위 시험이 실패하는 것을 본다.

**구현**
1. `copy_genome` 맨 끝(`taste_g` 블록 뒤): `g2 = float(s.soc_g[i])`; `if mut and ip[Q_SOC] != 0: g2 += fp[P_SOC_GENE_SIGMA] * normal(seed, S_SOC_GENE, cid, 0)`, `lo = −fp[P_SOC_RATE]`, `hi = 1 − fp[P_SOC_RATE]`로 자르고 `s.soc_g[j] = g2`.
2. `_reset_body`: `s.soc_g[j] = 0.0`(재주입 몸 초기화, `copy_genome`이 뒤에서 베끼는 경로는 그대로 — 기존 `taste_g`와 같은 위치), `meal_o[j] = 0`, `n_seen`·`n_seen_poison`·`n_seen_kin`·`seen_age_sum` 0.
3. `eat`: `taste_g` 비용 줄 바로 뒤, 산 개체 루프 안에서(`if taste:` 안, `if s.energy[i] >= e_max: continue` 앞) `if soc:` 로 `rate_s = clamp(fp[P_SOC_RATE] + s.soc_g[i], 0, 1)`, `c = fp[P_SOC_COST] * rate_s`, `s.energy[i] -= c`, `s.ledger[L_BRAIN] += c`.
   `soc = taste and ip[Q_SOC] != 0`. (`inert` 모드도 비용을 낸다. 비용은 `watch_meals`에 두지 않는다.)
4. `world.summary`: `soc_taste_enabled`일 때 `soc_rate_mean`(맛 블록의 `taste_rate_mean` 옆, 같은 꼴).
5. `world.set_config`: **`soc_g`를 지우지 않는다**(설계 5절, `taste_g`와 같은 선례). 주석 한 줄.

**초록 확인**: `python3 -m pytest -q tests/test_soc_taste.py tests/test_taste_gene.py`. 커밋: "사회 학습 2: soc_g 변이·비용".

---

## 과제 3. 끼니 기록·`watch_meals`(outcome·inert·what)·세계 연결·관찰 계수기
**바꾸는 파일**: `kkumteul/kernels.py`(`eat` 기록, 새 `watch_meals`), `kkumteul/world.py`(`step`), `kkumteul/bench.py`, `tests/test_soc_taste.py`.

**시험 먼저**(손으로 놓은 개체. 손 시험은 `build_grid`로 개체 격자를 먼저 짓고, 이웃이 실제로 보이는 **양성 확인을 먼저** 한다 — 안 그러면 `a_start`가 0이라 "반지름 밖이면 그대로" 시험이 헛통과한다. 먹이 놓기는 `tests/test_taste.py`의 `feed`·`_bad`를 쓴다)
- `test_meal_record_poison_first`: 한 스텝에 독 알과 좋은 알을 모두 먹이면(독 먼저·좋은 알 먼저 두 순서) 기록은 독(`meal_o == −1`, `meal_k`는 독의 모양 칸).
- `test_meal_record_cleared_when_no_meal`: 한 스텝 먹고 다음 스텝 못 먹으면 `meal_o == 0`, 배불러 `continue`한 개체·죽은 칸도 0.
- `test_watch_meals_moves_observer_toward_outcome`: j가 반지름 안에서 독을 먹으면 i의 그 모양 칸이 `soc_rate`(실제 속도)만큼 움직임: `taste_m[i, mc] == 속도 × (−1 − 이전값)`, 다른 칸은 그대로. 먹은 j 자신의 기억은 이 커널로 안 바뀜(자기 끼니는 개인 학습).
- `test_watch_meals_radius_edges`: 거리가 `soc_radius` 바로 안이면 보이고 바로 밖이면 안 보임, 격자 칸 경계 건너편(예: x = 7.9와 8.1)도 보임, 토러스 반대편(예: x = 0.5와 511.5)도 보임.
- `test_watch_meals_skips_dead_and_nonmeal`: 죽은 관찰자·죽은 시연자·`meal_o == 0`인 개체는 건너뜀, `n_seen`도 안 오름.
- `test_watch_meals_modes`: `what`은 독을 먹어도 +1 쪽, `inert`은 기억 안 바뀜(`n_seen`은 오름), `outcome`은 위와 같음.
- `test_watch_meals_zero_rate_observer`: 실제 속도가 0인 관찰자는 기억이 안 바뀌고 `n_seen`은 오름(보기는 보았다).
- `test_watch_meals_order_deterministic`: 여러 관찰자·여러 시연자가 겹친 같은 시나리오를 두 번 만들어 돌려 `state_hash`가 같음. 한 관찰자 한 칸을 시연자 j1 < j2가 차례로 갱신하는 값이 j 칸 번호순 식과 같음(손셈).
- `test_seen_counters`: `n_seen`, `n_seen_poison`, `n_seen_kin`(부모·자식·형제 시연자만), `seen_age_sum`(시연자 나이 합)이 손 시나리오에서 맞음.
- `test_world_step_rebuilds_grid_before_watch`: `soc_taste_enabled` 세계에서 `k.watch_meals`를 감싸(spy) 호출 순간의 `a_start`·`a_items`가 현재 위치로 **새로 지은 것**과 같음(충돌 뒤 낡은 격자가 아님). 장치 꺼짐 세계에서는 `watch_meals`도 추가 `build_grid`도 안 불림.
- `test_soc_off_world_never_calls_watch`: 장치가 꺼진 세계를 몇 스텝 돌려 `watch_meals` 호출 0(감싸서 셈), 상태 해시가 장치 넣기 전과 같음(처음부터 꺼진 세계).
- 빨강 확인.

**구현**
1. `eat`: 루프 맨 위 `s.ate[i] = 0.0` 옆에 `s.meal_o[i] = 0.0`(`if not s.alive[i]: continue` 앞). taste 갱신 블록(`taste_m` 갱신) 바로 옆, **독 분기 `continue`보다 앞**에서 기록:
   `if soc: if poison: s.meal_k[i] = mc; s.meal_o[i] = −1.0 elif s.meal_o[i] != −1.0: s.meal_k[i] = mc; s.meal_o[i] = 1.0`(독 우선, 아니면 마지막 좋은 알).
2. 새 `watch_meals(step, s, fp, ip)`(`kernels.py`, `@njit(cache=True)` **순차**): docstring에 "순차 전용: prange 금지, 상태 묶음 `s`를 거쳐 쓴다. 적용 순서 = 시연자 j의 칸 번호 오름차순(같은 관찰자의 한 칸을 여러 시연자가 차례로 갱신하는 지수평활은 비가환)".
   `for j in range(n)`: `meal_o[j] == 0`이거나 죽었으면 건너뜀. j의 격자 칸(`P_INV_CELL`, 가장자리 클램프)에서 3×3 칸을 `a_start`/`a_items`로 훑어, 관찰자 i(`i != j`, 산, `_wdiff`로 거리² ≤ `soc_radius²`)마다:
   실제 속도 `rate_i = clamp(soc_rate + soc_g[i], 0, 1)`; `n_seen[i] += 1`, 독이면 `n_seen_poison[i] += 1`, 친족이면 `n_seen_kin[i] += 1`(`parent[i] == ids[j]` 또는 `parent[j] == ids[i]` 또는 `parent[i] >= 0`이고 `parent[i] == parent[j]`), `seen_age_sum[i] += age[j]`;
   모드별 갱신: `outcome`: `mc = meal_k[j]`, `o = meal_o[j]`, `taste_m[i, mc] += rate_i × (o − taste_m[i, mc])`; `what`: `o = +1`; `inert`: 갱신 없음; `blind`는 과제 4.
3. `world.step`: `k.eat(...)` 바로 뒤, `st.give` 앞에 `if self.ip[k.Q_SOC]: k.build_grid(s.x, s.y, s.alive, inv_cell, gw, gh, s.a_start, s.a_items, s.cursor); k.watch_meals(t, s, fp, ip)`.
   (장치가 꺼져 있으면 추가 호출 없음. 부수 효과: 켜면 `give`·`attack`도 새 격자를 본다.)
4. `bench.py`: `k.eat(...)` 뒤에 `if ip[k.Q_SOC]:`로 같은 두 호출(장치를 켠 세계 성능을 잴 때만).

**초록 확인**: `python3 -m pytest -q tests/test_soc_taste.py tests/test_taste_gene.py tests/test_taste.py`. 커밋: "사회 학습 3: 끼니 기록·watch_meals(outcome/inert/what)·계수기".

---

## 과제 4. 눈가림 모드(`blind`)
**바꾸는 파일**: `kkumteul/kernels.py`(`watch_meals`), `tests/test_soc_taste.py`.

**시험 먼저**
- `test_blind_reproducible`: 같은 씨앗·스텝·관찰자·시연자면 같은 모양 칸이 뽑힘(두 번 돌려 `state_hash`).
- `test_blind_balanced`: 많은 (스텝, 쌍)으로 두 칸이 반반쯤(이항 5시그마 안).
- `test_blind_same_observations_as_outcome`: 같은 시나리오에서 `blind`와 `outcome`의 `n_seen`·`n_seen_poison`(보기 자체)이 같고, 바뀌는 것은 갱신할 모양 칸뿐(정보량만 0).
- `test_blind_key_unique`(과제 1에서 만든 것)가 `watch_meals`가 쓰는 같은 묶음 함수를 쓰도록 연결(상수를 한 곳에 둔다: `SOC_ID_STRIDE = 1 << 32`).
- 빨강 확인.

**구현**: `blind`: `mc = int(u01(ip[Q_SEED], S_SOC_BLIND, step, ids[i] * SOC_ID_STRIDE + ids[j]) * 2.0)`(모양 칸 동전), 결과 `o = meal_o[j]`는 그대로, 속도·시점 같음. 커널 docstring에 "id < 2^32 가정"을 적는다.
**초록 확인**: `python3 -m pytest -q tests/test_soc_taste.py`. 커밋: "사회 학습 4: 눈가림 모드".

---

## 과제 5. 끄는 규칙·겨루기 출력·옛 도구 정리
**바꾸는 파일**: `kkumteul/knockout.py`, `kkumteul/measure.py`, `wayfinder/stage2-4/taste_judge.py`, `wayfinder/stage2-4/hand_learner.py`, `tests/test_knockout.py`, `tests/test_soc_taste.py`.

**시험 먼저**
- `tests/test_knockout.py:117` 단정을 `TASTE_OFF`로 바꾼다: `ko.OFF_RULES[ch] == ko.TASTE_OFF` 그리고 `"soc_taste_enabled=false" in ko.OFF_RULES[ch]`(`taste`·`taste_vis`·`taste_val` 세 채널).
- `test_knockout_taste_split_with_soc_on_passes_config_check`(`tests/test_soc_taste.py`): `soc_taste_enabled`와 `taste_gene_enabled`를 켠 체크포인트 설정에 `TASTE_OFF`를 `with_overrides`로 한꺼번에 적용해도 `ValueError`가 안 남(대조 규칙 설정이 만들어짐).
- `test_flip_diff_taste_off_with_soc` / `test_hand_learner_taste_off_with_soc`: `taste_judge.flip_diff`가 쓰는 끔 목록과 `hand_learner`의 `with_set`이 장치를 켠 설정에서 검사에 안 걸림(두 곳이 `knockout.TASTE_OFF`를 씀을 확인).
- `test_compete_reports_r_list_and_first10`: `measure.compete`의 결과에 태그별 첫 10끼(`first10`: `{태그: [독 합, 먹은 끼니 합]}`, `n_poison10`과 `min(n_meal, FIRST_MEALS)`를 끝에 산 개체로 합) 키가 있고, `assay_sides` 출력에 `r_list`(반복별 r × 1000)와 `first10_by_tag`(반복 합)가 있음. 기존 키·값은 안 바뀜(기존 `assay` 시험 통과).
- 빨강 확인.

**구현**
1. `knockout.py`: `TASTE_OFF = ["taste_enabled=false", "taste_gene_enabled=false", "soc_taste_enabled=false"]`를 정의하고 `OFF_RULES`의 taste·taste_vis·taste_val 항목을 그것으로(리스트 복사).
2. `taste_judge.py`의 `flip_diff`(약 233줄): 끔 쪽 덮어쓰기를 `knockout.TASTE_OFF`로. `hand_learner.py`(약 355줄): `with_set(cfg, ",".join(knockout.TASTE_OFF))`(그 줄의 `with_set`은 "이름=값,이름=값" 문자열을 한 번에 적용하고 검사하므로 순서가 무관).
3. `measure.compete`: 끝 표본 뒤에 태그별 첫 10끼를 계산해 `first10`으로 돌려준다. `assay_sides`: 반복별 `r_list`, 태그별 `first10_by_tag`를 출력에 더한다(기존 키 유지).

**초록 확인**: `python3 -m pytest -q tests/test_knockout.py tests/test_soc_taste.py tests/test_taste_judge.py`. 커밋: "사회 학습 5: 끄는 규칙·겨루기 출력".

---

## 과제 6. 판정 도구: `taste_gene_judge.py --gene soc`와 관문
**바꾸는 파일**: `wayfinder/stage2-4/taste_gene_judge.py`, 새 `tests/test_soc_judge.py`.

**시험 먼저**(`tests/test_soc_judge.py`; 시뮬레이션 없이 도는 것을 최대로)
- `test_zeroed_side_soc`: `zeroed_side(side, "soc")`가 `soc_g = −soc_rate`(실제 0), 원본 배열은 안 건드림; `soc_taste_enabled`가 꺼진 설정이면 `ValueError`("보고 배우기가 꺼진 체크포인트").
- `test_zeroed_side_unknown_gene_message`: `taste`·`eta`·`soc` 이름만 허용.
- `test_apply_options`: `--set key=val`(여러 번)이 옆 설정에 `with_overrides`로 적용, `--soc-d D`가 B쪽 `soc_g[:] = D − soc_rate`, `--no-individual`이 두 쪽 모두 `taste_g[:] = −taste_rate`. 작은 세계의 `measure.side_from_world`로 만든 옆으로.
- `test_gate_tag_pass`, `test_gate_tag_undetectable`, `test_gate_tag_broken`: 합성 결과 사전(`r_per_1000`, `r_list`, `min_detectable_r_per_1000`, 전멸 횟수)으로 `gate_tag`를 시험 — 통과(어떤 d에서 ①outcome r>0·6판 중 5판 이상 B가 큼 ②outcome−blind Welch 양·`min_detectable` 위 ③음성 대조 0 ± `min_detectable`·양쪽 전멸 ≤2), 검출 불능(①② 실패·대조 멀쩡), 도구 깨짐(대조 실패 또는 양쪽 전멸 3판 이상).
- `test_gate_combines_tags_exit_codes`: 한 바탕이라도 통과면 0, 모두 검출 불능이면 3, 통과 없이 깨진 바탕이 있으면 4, 개인 학습 없는 판만 통과면 0이되 `only_noind: true`.
- `test_observe_reports_soc_fields`: 작은 세계 `observe`가 `soc_rate`(평균·사분위)와 `n_seen_*`를 냄(장치를 켠 설정일 때만).
- 빨강 확인.

**구현**(`taste_gene_judge.py`)
1. `zeroed_side(side, gene)`: `gene == "soc"` 분기(위 시험의 오류·`g["soc_g"][:] = −cfg.soc_rate`), 오류 문구의 허용 이름에 soc 추가.
2. `rate_zero(path, gene, ..., sets=None, soc_d=None, no_individual=False, neutral=True)`: 읽은 옆의 `cfg`에 `sets`를 `with_overrides`로 적용(`dict(side, cfg=...)`), `soc_d`가 있으면 B쪽 `soc_g[:] = soc_d − cfg.soc_rate`, `no_individual`이면 두 쪽 `taste_g[:] = −cfg.taste_rate`. 출력에 적용한 옵션(`sets`, `soc_d`, `no_individual`)을 적는다. `neutral=False`면 `knockout._neutral`을 건너뛰고 `min_detectable`을 안 낸다(`winner`는 중립 없이도 돈다).
3. CLI: `rate-zero`에 `--gene {taste,eta,soc}`, `--set KEY=VAL`(append), `--soc-d FLOAT`, `--no-individual`, `--no-neutral`.
4. `observe`: 장치를 켠 설정이면 `soc_rate`(평균·사분위, 돌리기 전)와, 돌린 뒤 산 개체의 `n_seen`·`n_seen_poison`·`n_seen_kin`·`seen_age_sum` 합과 비율(`seen_poison_frac`, `seen_kin_frac`, 산 개체 1,000스텝당 본 끼니 근사). 개인 속도(`taste_rate`)는 이미 있다.
5. `gate`(하위 명령, 순수 계산 `gate_tag(results)`와 파일 읽는 겉껍질): `gate --dir <결과 폴더> --tags a b`가 `j-dose-<이름>-<태그>.json`들을 읽어 판정(이름 규칙: `outcome-d0.05`, `outcome-d0.25`, `outcome-d1.0`, `blind-d0.05`, `blind-d0.25`, `blind-d1.0`, `off-d1.0`, `outcome-nocost-d0.25`, `outcome-nocost-d1.0`, `outcome-noind-d1.0`, `blind-noind-d1.0`). `min_detectable`은 `outcome-d0.05`에서 읽는다.
   JSON 한 장(태그별 상태·d별 통과·값)을 출력하고 종료 코드: 0 통과, 3 검출 불능, 4 도구 깨짐. 비용 있는 판(`outcome-d*`)이 관문이고 비용 없는 판·개인 학습 없는 판은 읽기용.
   Welch 비교는 `measure.t95(df)`를 쓴다(df는 `n − 1`로 어림, 반복 같은 수라 보수적).

**초록 확인**: `python3 -m pytest -q tests/test_soc_judge.py`. 문서의 실제 사용 예를 시간 제한을 걸고 직접 돌린다(시험 통과만으로 "됐다"고 하지 않는다):
`PYTHONPATH=. timeout 600 python3 wayfinder/stage2-4/taste_gene_judge.py rate-zero <작은 세계로 만든 soc 켠 체크포인트> --gene soc --soc-d 1.0 --n 20 --steps 300 --reps 2 --no-neutral --json <임시 폴더>/x.json`
(체크포인트는 시험 보조 함수로 작은 세계를 저장해 만든다. 서버의 큰 실행은 건드리지 않는다.) 커밋: "사회 학습 6: 판정 도구(--gene soc)와 관문".

---

## 과제 7. 실험 체인 스크립트와 예측 기록
**바꾸는 파일**: 새 `wayfinder/stage2-4/soc_chain.sh`, `wayfinder/stage2-4/results.md`(예측 절 — 실행 **전에** 적는다), `docs/status.md`(막힌 점·다음 후보 칸 한 줄).

1. `results.md`에 "사회 학습 — 예측" 절을 설계 4절의 예측 표 그대로(실행 전 날짜·시각 표시)로 적고, 판정 규칙(관문·"검출 불능"·"도구 깨짐"·진화 갈래 판정 문장·단서 문턱)을 한 번 더 옮긴다. 문턱은 실행 중에 바꾸지 않는다(바꿔야 하면 사유와 함께 별도로 적는다).
2. `soc_chain.sh`(아래 초안). 시작 전 `bash -n`과 `DRY=1`로 명령이 맞는지 본다.
3. 전체 시험을 한 번 돌린다: `python3 -m pytest -q`(메모리 상한 도구로 `--max 2G`, `NUMBA_NUM_THREADS=2`, `nice`). 실패가 있으면 이 과제에서 고친다.
4. 체인을 돌릴지는 서버 상황(다른 실행과 동시에 돌지 않는다)을 보고 정한다. 실험은 **한 번에 하나**, 2스레드, 메모리 상한 1,500M, `nice -n 15`.

### 체인 스크립트 초안 (`wayfinder/stage2-4/soc_chain.sh`)
```bash
#!/bin/bash
# 사회 학습 실험 체인(사용자 새 방향 2번). 설계: docs/superpowers/specs/2026-11-07-social-learning-design.md 둘째 판.
# 예측·문턱: wayfinder/stage2-4/results.md "사회 학습 — 예측"(실행 전에 적어 둔다).
# 앞 실험이 끝난 뒤 한 번에 하나(2스레드, nice). 순서: 판정 0(용량 반응) → 관문 → 진화 갈래 → 갈래별 판정.
#
#   ROOT=<레포 원래 폴더> CODE=<커밋한 코드 사본 폴더> [WAIT_FOR=<pgrep -f 패턴>] [DRY=1] \
#     <메모리 상한 도구> --max 1500M --bg soc-chain --log "$ROOT/runs/soc-exp/chain.log" -- bash wayfinder/stage2-4/soc_chain.sh
# 상한은 묶음 전체에 걸린다. 갈래·판정마다 done/json 표시가 있어 다시 불러도 이어 간다.
# 시간 어림(서버 부하에 따라 달라짐, 추정): 판정 0 약 1.5시간, 갈래 여섯 약 1.5시간, 갈래별 판정 약 1시간.
set -u
R=${ROOT:?ROOT=레포 원래 폴더}
C=${CODE:?CODE=코드 사본 폴더}
E=$R/runs/soc-exp
PY=$R/.venv/bin/python
J=wayfinder/stage2-4/taste_gene_judge.py
START=ckpt_000002600000.npz     # 바탕: 개인 학습 갈래 g-p500-a/b 260만
END=ckpt_000002900000.npz       # 30만 스텝 뒤
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
        "taste_gene_enabled": True, "poison": 0.3, "taste_inherit": 0.0}
bad = {k: (c.get(k), v) for k, v in want.items() if c.get(k) != v}
if bad:
    print("체크포인트 설정이 기대와 다름:", bad)
    sys.exit(1)
PYEOF
}

dose() {  # 바탕태그 이름 d [판정 옵션...]  (이름 예: outcome-d0.25)
  local tag=$1 name=$2 d=$3; shift 3
  local ck out
  ck=$(base_ckpt "$tag"); out=$E/j-dose-$name-$tag
  [ -s "$out.json" ] && { note "용량 반응 $name-$tag 이미 있음"; return 0; }
  note "용량 반응 $name-$tag 시작"
  runit nice -n 15 "$PY" $J rate-zero "$ck" --gene soc --soc-d "$d" --set soc_taste_enabled=true --set soc_rate=0.05 \
    "$@" --json "$out.json" > "$out.log" 2>&1
  local rc=$?
  note "용량 반응 $name-$tag 끝 ($rc): $(tail -1 "$out.log" 2>/dev/null | cut -c1-160)"
}

dose_all() {  # 바탕태그 하나의 판정 0 열한 판. 중립은 첫 판에서만(min_detectable)
  local t=$1
  dose "$t" outcome-d0.05 0.05
  for d in 0.25 1.0; do dose "$t" outcome-d$d $d --no-neutral; done
  for d in 0.05 0.25 1.0; do dose "$t" blind-d$d $d --set soc_taste_mode=blind --no-neutral; done
  dose "$t" off-d1.0 1.0 --set soc_taste_enabled=false --no-neutral                 # 음성 대조
  for d in 0.25 1.0; do dose "$t" outcome-nocost-d$d $d --set soc_cost=0 --no-neutral; done   # 비용 치우침 분리
  dose "$t" outcome-noind-d1.0 1.0 --no-individual --no-neutral                    # 개인 학습 없는 판(2×2 한 칸)
  dose "$t" blind-noind-d1.0 1.0 --no-individual --set soc_taste_mode=blind --no-neutral
}

run_branch() {  # 이름 씨앗(a=바탕 a·같은 흐름, b=바탕 b, c=바탕 a·--new-stream) 설정...
  local name=$1 seed=$2; shift 2
  local base=$seed; [ "$seed" = c ] && base=a
  local dir=$R/runs/s-$name-$seed
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

judge() {  # 갈래-씨앗 무엇(out|blind|ctrl|nocost|noind|noindb|obs)
  local tag=$1 what=$2
  local ck=$R/runs/s-$tag/checkpoints/$END out=$E/j-$what-$tag
  [ -s "$out.json" ] && { note "판정 $what-$tag 이미 있음"; return 0; }
  [ -f "$ck" ] || { note "판정 $what-$tag: 체크포인트 없음 $ck"; return 0; }
  note "판정 $what-$tag 시작"
  local opts=()
  case $what in
    out)     ;;
    blind)   opts=(--set soc_taste_mode=blind) ;;
    ctrl)    opts=(--set soc_taste_enabled=false) ;;          # 음성 대조(soc 꺼짐)
    nocost)  opts=(--set soc_cost=0) ;;
    noind)   opts=(--no-individual) ;;
    noindb)  opts=(--no-individual --set soc_taste_mode=blind) ;;
  esac
  if [ "$what" = obs ]; then
    runit nice -n 15 "$PY" $J observe "$ck" --json "$out.json" > "$out.log" 2>&1
  else
    runit nice -n 15 "$PY" $J rate-zero "$ck" --gene soc "${opts[@]}" --json "$out.json" > "$out.log" 2>&1
  fi
  local rc=$?
  note "판정 $what-$tag 끝 ($rc): $(tail -1 "$out.log" 2>/dev/null | cut -c1-160)"
}

pre_check a && pre_check b || { note "바탕 확인 실패 — 체인 중단"; exit 1; }

# 1. 판정 0: 용량 반응(진화 전)
for t in a b; do dose_all $t; done
runit "$PY" $J gate --dir "$E" --tags a b > "$E/gate.json" 2> "$E/gate.log"
rc=$?
note "관문 종료 코드 $rc (0 통과, 3 검출 불능, 4 도구 깨짐): $(tr -d '\n ' < "$E/gate.json" 2>/dev/null | cut -c1-200)"
if [ "$rc" != 0 ] && [ -z "${DRY:-}" ]; then
  note "관문 못 넘음 — 진화 갈래를 돌리지 않고 닫는다(검출 불능·도구 깨짐은 결과 문서에 그대로 적는다)"
  exit 0
fi

# 2. 진화 갈래: 같은 체크포인트에서 갈라 30만 스텝(씨앗 a·b). 설정은 체크포인트에서 오되 감사용으로 박는다
G="preset=taste taste_inherit=0 taste_gene_enabled=true kind_flip_period=500"
for s in a b; do
  run_branch on    $s $G soc_taste_enabled=true soc_taste_mode=outcome soc_rate=0.05
  run_branch inert $s $G soc_taste_enabled=true soc_taste_mode=inert   soc_rate=0.05
  run_branch off   $s $G soc_taste_enabled=false
done
# 3. 판정: 켠 갈래는 판정 나(outcome·blind·음성 대조·비용 없음·개인 학습 없음)+단서, 불활성·끔은 단서(observe)만
for s in a b; do
  for w in out blind ctrl nocost noind noindb obs; do judge on-$s $w; done
  judge inert-$s obs
  judge off-$s obs
done
# 4. 시간이 남으면: 씨앗 c, 첫걸음 위치 0.2 갈래
run_branch on    c $G soc_taste_enabled=true soc_taste_mode=outcome soc_rate=0.05
run_branch inert c $G soc_taste_enabled=true soc_taste_mode=inert   soc_rate=0.05
run_branch off   c $G soc_taste_enabled=false
for w in out blind ctrl obs; do judge on-c $w; done
judge inert-c obs
judge off-c obs
for s in a b; do
  run_branch on02 $s $G soc_taste_enabled=true soc_taste_mode=outcome soc_rate=0.2
  for w in out blind ctrl obs; do judge on02-$s $w; done
done
note "체인 끝"
```
(체인은 갈래 폴더 `runs/s-<갈래>-<씨앗>`, 판정 원자료 `runs/soc-exp/j-*.json`을 쓴다 — 모두 `runs/` 아래라 git에 안 올라간다. 갈래마다 `done` 표시가 있어 중간에 끊겨도 다시 불러 이어 간다.)

5. 체인 결과를 읽을 때: 판정 문장은 설계 4절 그대로 — "같은 끝 무리에서 outcome r(B−A)이 양이고 `min_detectable` 위, blind와 Welch 차이가 양, 음성 대조 통과, 두 씨앗 같은 쪽"일 때만 지지.
   결과 문서(`wayfinder/stage2-4/results.md`)에 판정을 적을 때 `docs/status.md`의 관찰·기각 칸과 마지막 갱신일도 같이 고친다(관찰에는 잰 값만, 해석에는 추정만). `observe`의 `n_seen*`로 설계의 셈(본 끼니/한 생 6~20)을 실측으로 확인하고 `n_seen_kin`은 3번(부모의 가르침) 설계 앞에서 쓴다.
   커밋: "사회 학습 7: 실험 체인과 예측 기록".

---

## 과제 요약
1. 설정·난수·필드·옛 호환(동작 변화 없음)
2. 보고 배우는 속도 유전자: 변이·비용
3. 끼니 기록·`watch_meals`(outcome·inert·what)·세계 연결·관찰 계수기
4. 눈가림 모드(`blind`)
5. 끄는 규칙·겨루기 출력·옛 도구 정리
6. 판정 도구 `--gene soc`·관문
7. 실험 체인 스크립트와 예측 기록
