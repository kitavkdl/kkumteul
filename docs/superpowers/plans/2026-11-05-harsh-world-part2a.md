# 혹독한 세계 2부-가 — 공유 첫 과제·가을 열매철·들기 (구현 계획)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 2부와 암수가 같이 쓰는 저장·해시 틀을 먼저 만들고, 저장이 생길 수 있는 잉여 창(가을 열매철)과 물건을 들고 내려놓는 장치(③)를 만든다.

**Architecture:** 해시는 "기본값과 같으면 넣지 않는다"는 한 규칙으로 새 필드·늘린 배열 꼬리를 건너뛰어 장치를 끈 세계의 해시를 지킨다. 계수기·`budget`은 확장 크기 세계에서만 늘린다. 가을은 `World.season`의 풀 배수 계단에 한 칸을 더한다. 들기는 먹이 칸을 물건으로 쓰고(손 `a_held` ↔ 물건 `f_holder`), 들판 알만 상한에 세는 계수기 `C_NFIELD`를 들기가 켜졌을 때만 쓴다.

**Tech Stack:** Python 3, numpy, numba(`njit`), pytest.

**Spec:** `docs/superpowers/specs/2026-11-05-harsh-world-design.md`(여섯째 판) — "입출력 크기", ① 끝의 "가을 열매철", ③, "장부·계수기·해시·다시 보기", "구현 메모". 검토 기록 `2026-11-05-harsh-world-review.md` 끝의 "2부-가 계획 검토 두 갈래"와 그 판단 표(이 계획은 그 표를 반영한 둘째 판이다). 암수 설계 `2026-11-05-sex-design.md`의 "저장·해시·기록" 1·2는 과제 1과 같은 일이다. ④~⑥과 섞기 대조는 2부-나 계획.

## Global Constraints
- 장치는 기본 꺼짐. 모두 끈 세계는 `tests/test_stages.py`의 `PINNED_HASH_300` 세 값이 그대로여야 한다. 확장 크기 세계는 과제 1이 박는 `PINNED_EXT_HASH_300` 세 값이 그대로여야 한다(1부 코드 `0945b10`이 낸 값).
- `preset=harsh`(1부 = `ext` + 겨울 + 부패)의 뜻은 바꾸지 않는다. 새 묶음 `preset=harsh2`는 2부 장치를 차례로 덧붙여 가며 2부-나 끝에 완성된다(실험은 실행 폴더의 `config.json`에 설정 전체가 남으므로 재현은 그것으로 한다).
- 에너지가 생기는 길은 먹기와 명시적 주입뿐. 장치마다 `|residual|/(spawn+inject) < 1e-9`.
- `prange` 안에서 상태 묶음(`s.x`)을 거쳐 쓰지 않는다. 새 커널은 순차(`@njit(cache=True)`)로.
- 난수 용도 번호는 바꾸지 않고, 이 계획이 처음 쓰는 번호는 없다(과제 1이 표에 이름만 적는다).
- **바깥에서 부르는 커널**(`spawn_food`, `niche_world`, `sense`, `move` 등 `world.py`·`measure`·`bench.kernel_breakdown`·`knockout.wrap`·시험이 부르는 것)에 인자를 더하면 기본값으로 옛 호출을 살린다. **안에서만 부르는 도우미**(`_place_food`, `_drop_food_slot`, `_evict_oldest`, `make_room`, `_remove`)의 새 인자는 기본값 없이 둔다 — 빠뜨린 호출을 numba가 컴파일 오류로 잡게(바깥 호출이 없음을 확인했다).
- 공개 레포: 코드·주석에 로컬 경로, "과제 N"·"2부" 같은 작업 과정 표현을 쓰지 않는다(상태를 말하는 표현으로).
- 시험 실행: 작업 폴더에서 `NUMBA_NUM_THREADS=2`, 메모리 상한 2GB, 낮은 우선순위로 `python -m pytest -q -p no:cacheprovider <대상>`. 전체 약 2.5분. 서브에이전트는 시험만 돌리고 시뮬레이션 실험은 돌리지 않는다. 시험 임시 폴더는 pytest `tmp_path`만. 시험 개수는 적어 두지 않고 실행 출력으로 확인한다.
- 커밋 메시지는 한국어, 고친 파일만 `git add`(`.crew`, `.superpowers` 제외), 커밋 뒤 작업 가지로 push.

## Review Focus
1. **1부 확장 세계 체크포인트(계수기 24칸)를 새 코드로 이어 돌리면** 계수기가 40칸으로 채워 읽히고, 저장 해시 검증이 건너뛰어지지 않으며(채운 꼬리가 0이므로) 그 세계의 해시가 1부와 같다 — 과제 1 `test_old_ext_checkpoint_loads_exact`, `PINNED_EXT_HASH_300`.
2. **들고 있는 개체가 죽으면** 물건이 그 자리에 떨어지고, 죽은 칸이 새 개체로 다시 쓰여도 그 개체가 물건을 든 꼴이 되지 않는다 — 과제 3 `test_death_drops_item_and_slot_reuse_is_empty_handed`.
3. **각본이 실행 중에 `carry_enabled`를 켜면** 들판 계수기가 그 순간의 들판 알 수로 맞춰지고, 끄면 든 물건이 모두 내려진다 — 과제 3 `test_carry_toggle_recounts_and_releases`.
4. **놓는 곳·없애는 곳 어디서든** `C_NFIELD`가 살아 있는 들판 알 수와 매 스텝 같다(유입·거름·사체·부패·운석·상한 교체) — 과제 3 `test_field_counter_matches_every_step`.
5. **들기 + 상한 교체 + 부패**를 같이 켜면 교체가 든 알·내려놓은 알을 고르지 않고 장부가 맞다 — 과제 3 `test_cap_replace_skips_stored_items`.
6. **들기를 켠 실행을 다시 보기로 적으면** 든 알이 "먹힘(모름) + 새 먹이"로 적히지 않고, 들림·내려놓음이 한 물건의 이어진 기록으로 남는다 — 과제 4 `test_replay_tracks_carried_item`.

---

### Task 1: 공유 첫 과제 — 해시 규칙·번호표·배열 길이

**Files:**
- Modify: `kkumteul/kernels.py` (계수기 번호 영역 `C_SPOILED` 아래, `N_COUNTERS`, `alloc_state`, `PERTURB_FLEX` 근처에 기본값 표)
- Modify: `kkumteul/world.py` (`state_hash`, `load_arrays`)
- Modify: `kkumteul/config.py` (`Config.ext_size` 속성)
- Modify: `kkumteul/rng.py` (용도 번호 표 끝)
- Modify: `kkumteul/_cache.py` (`_SOURCES`)
- Test: `tests/test_harsh.py`

**Interfaces:**
- Produces (kernels.py): 계수기 번호 `C_NFIELD = 22`, `C_CARRIES = 23`, `C_STORE_EVICTED = 24`, `C_DEATH_PREDATED = 25`, `C_SCAVENGED = 26`, `C_PREY_KILLED = 27`, `C_ANIMAL_REPELLED = 28`, `C_CHASES = 29`, `C_MATINGS = 30`, `C_DEATH_CONTEST = 31`, `C_INFECTIONS = 32`, `C_DEATH_INFECTED = 33`(34·35 암수 예비), `C_RESCUES = 36`(37~39 예비). `N_COUNTERS_BASE = 24`, `N_COUNTERS_EXT = 40`. `N_COUNTERS`는 지우지 말고 `N_COUNTERS_BASE`와 같은 값으로 둔다(바깥 코드 호환). `N_BUDGET_BASE = 2`, `N_BUDGET_EXT = 3`.
- Produces (kernels.py): `HASH_DEFAULTS: dict[str, int | float]` — 저장 필드 이름 → 기본값. 지금은 `{"elig_in": 0, "elig_rec": 0}`. 과제 3이 `a_held: -1`, `f_holder: -1`, `f_dropped: 0`을 더한다. `PERTURB_FLEX`는 `set_config`가 계속 쓰므로 남긴다.
- Produces (config.py): `Config.ext_size -> bool` = `n_spare_in >= N_STAGE_IN + N_EXT_IN and n_spare_out >= N_STAGE_OUT + N_EXT_OUT`.
- Produces (rng.py): `S_ANIMAL_POS = 19`, `S_WANDER = 20`, `S_CHASE = 21`, `S_CONFUSE = 22`, `S_MEAT = 23`, `S_SHUFFLE = 24`, `S_SEX = 25`, `S_ORNAMENT = 26`, `S_CROSS = 27`, `S_PATHOGEN = 28`, `S_RESERVOIR = 29`, `S_SEX_SPARE = 30` — 이름과 한 줄 뜻만(아직 아무도 부르지 않음).

- [ ] **Step 1: 실패하는 시험을 쓴다** (`tests/test_harsh.py`)

```python
# 1부 코드(0945b10)가 낸 300스텝 해시. 확장 크기 세계에 새 필드·늘린 계수기가 들어와도 장치를 끈 동안은 같아야 한다.
PINNED_EXT_HASH_300 = {
    ("ext",): "f99625e844a559b3d3518c2dc18ca85fb83642f4bc71046d3c74afdea5ec3fdf",
    ("harsh",): "edfcfe6ec79c8676c656d64cf4a01145c9a9b245e3cad99c9ffb10d9ec53a3ef",
    ("harsh", "stage4", "conflict", "kinds"): "09def17063df81639d606807ce530c6de55df7e87e8f00f6ac95c0f6dc990ed7",
}

@pytest.mark.parametrize("presets", list(PINNED_EXT_HASH_300))
def test_ext_worlds_keep_pinned_hash(presets):
    w = World(small(*presets)); w.run(300)
    assert w.state_hash() == PINNED_EXT_HASH_300[presets]

def test_counter_and_budget_length_follow_ext_size():
    assert len(World(small()).s.counters) == k.N_COUNTERS_BASE   # 기본 크기
    w = World(small("ext"))
    assert len(w.s.counters) == k.N_COUNTERS_EXT and len(w.s.budget) == k.N_BUDGET_EXT

def test_ext_hash_ignores_zero_tail():
    w = World(small("harsh")); w.run(50)
    h = w.state_hash()
    w.s = w.s._replace(counters=w.s.counters[:24].copy(), budget=w.s.budget[:2].copy())
    assert w.state_hash() == h          # 꼬리가 0이면 옛 길이 해시와 같다
    w2 = World(small("harsh")); w2.run(50)
    w2.s.counters[k.C_RESCUES] = 1
    assert w2.state_hash() != h         # 꼬리에 값이 있으면 들어간다

def test_old_ext_checkpoint_loads_exact(tmp_path):
    # 확장 세계를 저장한 뒤 체크포인트의 counters를 24칸, budget을 2칸으로 잘라 1부 체크포인트처럼 만든다.
    # 새 코드로 읽으면 채워 읽기가 되고, load_arrays가 True(exact)를 돌려주며, checkpoint.load의 저장 해시 검증을 통과한다.
    ...
```
(`small()`의 묶음 이름은 위치 인자다 — `tests/test_stages.py`의 정의를 따른다. 저장·읽기는 `tests/test_harsh.py`의 기존 resume 시험 꼴을 따른다.)

- [ ] **Step 2: 실패를 확인한다** — `... tests/test_harsh.py -k "pinned_hash or length_follow or zero_tail or loads_exact"` → `N_COUNTERS_BASE` 없음 등으로 FAIL(`test_ext_worlds_keep_pinned_hash`는 지금 코드에서 PASS — 이 시험은 지킴이다).

- [ ] **Step 3: 구현한다**
  - `alloc_state(cfg)`: `counters` 길이 = `N_COUNTERS_EXT if cfg.ext_size else N_COUNTERS_BASE`, `budget` 길이도 같은 조건으로 3 / 2.
  - `state_hash`: 개체·먹이 필드 반복 모두에서 `name in HASH_DEFAULTS`이고 살아 있는 칸 값이 모두 그 기본값이면(`(v == d).all()` — `.any()`가 아니다, 기본값 −1이 있다) 건너뛴다. `counters`는 `N_COUNTERS_BASE` 뒤 꼬리가 모두 0이면 앞 24칸만, `budget`은 `N_BUDGET_BASE` 뒤가 모두 0이면 앞 2칸만 넣는다.
  - `load_arrays`: 빠진 필드가 `HASH_DEFAULTS`에 있으면 `alloc_state`의 기본값을 그대로 두고 `missing`에 넣지 않는다(지금의 `PERTURB_FLEX` 예외를 이 표로 바꾼다). `counters`·`budget`을 더 긴 쪽으로 채워 읽은 경우는 `exact`를 끄지 않는다(채운 꼬리가 0이라 해시 규칙상 저장 때와 같다). 그 밖의 채워 읽기는 지금처럼 `exact = False`.
  - `_SOURCES`에 `"stages.py", "world.py"`를 더한다.

- [ ] **Step 4: 시험 통과 + 박아 둔 해시 그대로** — 전체 시험. 기대: 모두 PASS. 특히 `PINNED_HASH_300`, `PINNED_EXT_HASH_300`, `test_harsh.py`의 갈라 켜기·resume 해시 시험.

- [ ] **Step 5: 커밋** — `git add kkumteul/kernels.py kkumteul/world.py kkumteul/config.py kkumteul/rng.py kkumteul/_cache.py tests/test_harsh.py` → "공유 저장 틀: 해시 기본값 규칙, 계수기·난수 번호표, 확장 세계에서만 계수기 40칸, 1부 확장 해시 박기".

---

### Task 2: 가을 열매철

**Files:**
- Modify: `kkumteul/config.py` (겨울 설정 아래 `autumn_len`, `autumn_mult`, `validate`, `preset()`)
- Modify: `kkumteul/world.py:102-122` (`in_winter` 옆 `in_autumn`, `season`)
- Test: `tests/test_harsh.py`

**Interfaces:**
- Produces (config.py): `autumn_len: int = 0`, `autumn_mult: float = 1.0`(주석: 겨울 바로 앞 `autumn_len` 스텝의 풀 배수, 지구의 가을 열매 몰림, 임시 결정). 묶음 `preset("harsh2")` = `harsh` + `autumn_len=800, autumn_mult=2.0`(과제 4가 `carry_enabled=True`를 더한다). `preset()`의 모르는 이름 오류문에 `harsh2`를 더한다.
- Produces (world.py): `World.in_autumn(t) -> bool` = 겨울이 켜져 있고 `P - W - A <= t % P < P - W`. `World.season(t)`의 첫 값(풀 배수): 겨울 `winter_mult`, 가을 `autumn_mult`, 나머지 `(P - W*winter_mult - A*autumn_mult) / (P - W - A)`. 기초 대사 배수와 sin·cos는 그대로.

- [ ] **Step 1: 실패하는 시험을 쓴다**

```python
def test_autumn_off_keeps_season_values():
    w = World(small("harsh"))
    assert w.season(100)[0] == pytest.approx((4000 - 1200 * 0.3) / 2800)   # 1.3, 1부와 같음
    assert not w.in_autumn(2500)

def test_autumn_mast_keeps_year_mean():
    w = World(small("harsh2"))
    P = w.cfg.season_period
    g = [w.season(t)[0] for t in range(P)]
    assert sum(g) / P == pytest.approx(1.0, rel=1e-12)
    assert w.season(2000)[0] == pytest.approx(2.0) and w.in_autumn(2000)    # 2000..2799 가을
    assert w.season(100)[0] == pytest.approx(1.02)
    assert w.season(2000)[1] == 1.0                                          # 가을 기초 대사는 그대로

def test_autumn_validate():
    with pytest.raises(ValueError):
        small("harsh", autumn_len=2800)                    # 가을 + 겨울이 해 전체
    with pytest.raises(ValueError):
        small("harsh", autumn_len=800, autumn_mult=0.5)    # 가을 배수는 1 이상
    with pytest.raises(ValueError):
        small("harsh", autumn_len=800, autumn_mult=5.0)    # 나머지 철 배수가 음수
    with pytest.raises(ValueError):
        small("ext", autumn_len=800, autumn_mult=2.0)      # 겨울이 꺼진 세계에 가을만

def test_autumn_ledger_closes():
    # harsh2(들기 전) 세계를 season_period=600, winter_len=150, autumn_len=150으로 줄여 1,800스텝(세 해) → 잔차 < 1e-9,
    # 가을 스텝의 유입 몫이 다른 철보다 크다(budget·L_SPAWN 증가분으로)
    ...
```

- [ ] **Step 2: 실패를 확인한다** — `-k autumn` → `in_autumn` 없음으로 FAIL.

- [ ] **Step 3: 구현한다** — `validate`: `autumn_len >= 0`, `autumn_len > 0`이면 `season_enabled`, `autumn_len + winter_len < season_period`, `autumn_mult >= 1`, 나머지 철 배수 `>= 0`. `season`이 가을이면 배수 `autumn_mult`. `spoil_mult`는 겨울만 본다(바꾸지 않는다).

- [ ] **Step 4: 시험 통과** — `tests/test_harsh.py` 전체와 `tests/test_stages.py`. 기대: PASS, 1부 겨울 시험(`test_winter_*`, `test_niche_share_follows_season_mult`)과 `PINNED_EXT_HASH_300` 그대로.

- [ ] **Step 5: 커밋** — "가을 열매철: 겨울 앞 풀 배수 계단(기본 꺼짐), harsh2 묶음".

---

### Task 3: ③ 들기 — 물건 칸과 장부

**Files:**
- Modify: `kkumteul/kernels.py` (`AGENT_FIELDS`, `FOOD_FIELDS`, `HASH_DEFAULTS`, `alloc_state`, `pack_params`의 `ip` 칸, `_place_food`, `_drop_food_slot`, `_evict_oldest`, `make_room`, `spawn_food`, `spoil_food`, `eat`, `_reset_body`, `_remove`과 그 호출 다섯 곳(`process_deaths` 셋, `passive_events`, `strike`), `strike`의 먹이 태우기)
- Modify: `kkumteul/stages.py` (`niche_world`의 상한 비교 661행·놓기 686~687행, `isp` 칸)
- Modify: `kkumteul/world.py` (`NEW_SINCE_STAGE1`, `set_config`, `load_arrays` 뒤 들판 수 다시 세기, `recount_field`, `release_all`)
- Modify: `kkumteul/measure.py` (`inject_agent`가 `a_held=-1`)
- Modify: `kkumteul/config.py` (`carry_enabled`, `carry_cost_mult`, `needs_ext`, `field_cap`)
- Test: `tests/test_harsh.py`

**Interfaces:**
- Consumes: 과제 1의 `C_NFIELD`, `C_CARRIES`, `C_STORE_EVICTED`, `HASH_DEFAULTS`.
- Produces (config.py): `carry_enabled: bool = False`, `carry_cost_mult: float = 1.3`(지구 참고 없음, 임시). `needs_ext`에 `carry_enabled`를 더한다. `Config.field_cap -> bool` = `carry_enabled`(2부-나가 `or prey_enabled`를 더한다).
- Produces (kernels.py): 필드 `a_held`(int32, 개체, 저장), `f_holder`(int32, 먹이, 저장), `f_dropped`(int8, 먹이, 저장 — 1이면 "들판을 떠난 적 있는 물건". 들 때 켜고, 내려놓아도 1로 남는다). 기본값 −1, −1, 0. `ip` 칸 `Q_FIELD_CAP`(= `int(cfg.field_cap)`), `Q_CARRY`(= `int(cfg.carry_enabled)`), `fp` 칸 `P_CARRY_COST`. `stages.py`의 `isp`에 `I_FIELD_CAP`.
- Produces (kernels.py), **새 인자는 모두 기본값 없음**:
  - `is_field(s, j) -> bool` = `f_dropped[j] == 0 and f_kind[j] != KIND_CORPSE`(2부-나가 고기도 뺀다).
  - `_place_food(s, px, py, e, kind, step, field_cap)` — 놓으면 `f_holder=-1`, `f_dropped=0`, `field_cap`이고 `is_field`면 `C_NFIELD += 1`.
  - `_drop_food_slot(s, j, field_cap)` — 손을 풀고(`f_holder[j] >= 0`이면 `a_held[f_holder[j]] = -1`, `f_holder[j] = -1`), `field_cap`이고 들판 알이면 `C_NFIELD -= 1`, `f_alive=False`, `C_NFOOD -= 1`.
  - `_evict_oldest(s, field_cap)` — `field_cap`이면 들판 알만 후보, 칸 비우기는 `_drop_food_slot`.
  - `make_room(s, cap, replace, field_cap)` — `field_cap`이면 `C_NFIELD`로 비교.
  - `evict_stored(s, field_cap) -> bool` — **들지 않은 들판 아닌 물건(내려놓은 알·사체) 가운데 가장 먼저 놓인 것**(같으면 낮은 칸)을 `L_LOST`로 치우고 `C_STORE_EVICTED += 1`. 없으면 False.
  - `store_room(s, field_cap) -> bool` — 들판 아닌 물건 수(`C_NFOOD − C_NFIELD`)가 개체 칸 수(`s.x.shape[0]` = `max_agents`)보다 작으면 True, 아니면 `evict_stored`의 결과.
  - `_remove(s, i, cause, step, recycle, nd, field_cap)` — 들고 있던 물건을 그 자리에 내려놓는다(`f_holder=-1`, `a_held=-1`, `f_dropped`는 이미 1). 사체는 `field_cap`이면 먼저 `store_room`, 못 얻으면 지금처럼 `L_LOST`. 호출 다섯 곳은 모두 `ip[Q_FIELD_CAP]`을 넘긴다.
- Produces (world.py): `World.recount_field()` — `C_NFIELD`를 살아 있는 들판 알 수로 다시 센다(numpy). `World.release_all()` — 모든 손을 푼다(`f_dropped`는 그대로).

- [ ] **Step 1: 실패하는 시험을 쓴다** (모두 `small("ext", carry_enabled=True)` 꼴, 행동 커널은 과제 4 — 여기서는 `a_held`/`f_holder`/`f_dropped`를 시험이 직접 쓴다. 손을 맞출 때는 들 때처럼 `f_dropped=1`과 `C_NFIELD -= 1`도 같이 맞추는 시험 도우미 `hold(w, i, j)`/`unhold(w, i)`를 둔다)

```python
def test_carry_fields_default_and_hash_unchanged():
    w = World(small("ext"))
    assert (w.s.a_held == -1).all() and (w.s.f_holder == -1).all() and (w.s.f_dropped == 0).all()
    w.run(20); h = w.state_hash()
    i = int(np.flatnonzero(w.s.alive)[0])
    w.s.a_held[i] = 5
    assert w.state_hash() != h          # 기본값이 아니면 해시에 들어간다
    w.s.a_held[i] = -1
    assert w.state_hash() == h          # 모두 기본값이면 빠진다

def test_old_checkpoint_without_carry_fields_loads_exact(tmp_path):
    # 확장 세계를 저장한 뒤 체크포인트에서 a_held, f_holder, f_dropped를 빼고 읽는다 →
    # 기본값으로 채워지고 load_arrays가 True, 저장 해시 검증 통과
    ...

def test_field_counter_matches_every_step():
    # carry 켠 ext 세계 + stage4(거름 niche_world) + corpse 켬 + spoil_enabled + food_cap_replace + 작은 food_cap,
    # 각본 없이 매 스텝 strike를 가끔 손으로 부른다(burn_food). 400스텝 동안 매 스텝
    # C_NFIELD == (f_alive & f_dropped==0 & f_kind != KIND_CORPSE).sum(), C_NFOOD == f_alive.sum(). 잔차 < 1e-9
    ...

def test_eat_releases_and_counts_field():
    # 들판 알을 먹으면 C_NFIELD가 하나 줄고, 내려놓은 알(f_dropped=1)을 먹으면 C_NFIELD는 그대로
    ...

def test_held_item_is_not_eaten():
    # 든 알 곁에 배고픈 다른 개체를 두고 한 스텝 → 알이 그대로 f_alive, f_holder 그대로
    ...

def test_death_drops_item_and_slot_reuse_is_empty_handed():
    # 물건을 든 개체를 굶겨 죽인다 → 물건 f_holder == -1, f_dropped == 1, 자리 = 죽은 자리, f_alive
    # 이어 그 칸에 새 개체가 나면(process_births/reseed) a_held == -1. strike로 죽인 경우도 같다
    ...

def test_cap_counts_field_only():
    # food_cap=50, 들판 알 50 + 내려놓은 알 30(C_NFOOD 80): 다음 유입은 놓이지 않고 L_DISCARD로 간다(C_NFIELD == cap).
    # 들판 알 하나를 먹이면 다음 유입 한 알이 놓인다. 내려놓은 알 30은 상한에 세지 않는다
    # (반대로 들판 알 20 + 내려놓은 알 30이면 C_NFOOD가 cap을 넘어도 유입이 놓인다). 장부 잔차 < 1e-9
    ...

def test_cap_replace_skips_stored_items():
    # food_cap_replace=True + spoil_enabled: 가장 오래된 것이 내려놓은 알·든 알이어도 교체는 들판 알을 고른다
    ...

def test_store_room_evicts_oldest_unheld():
    # 들판 아닌 물건이 max_agents개면 store_room이 가장 오래된 "들지 않은" 것(내려놓은 알이나 사체)을 L_LOST로 치우고
    # C_STORE_EVICTED += 1. 들판 아닌 물건이 모두 든 것이면 False. 사체가 칸을 채운 경우도 사체를 치워 자리를 낸다
    ...

def test_carry_toggle_recounts_and_releases():
    # carry 꺼진 ext 세계를 돌린 뒤 set_config로 켬 → C_NFIELD == 들판 알 수(사체 제외)
    # 손을 몇 개 맞춘 뒤 set_config로 끔 → a_held 모두 -1, f_holder 모두 -1, f_dropped는 그대로
    ...

def test_carry_ledger_closes():
    # carry 켠 ext 세계 300스텝(매 50스텝 hold/unhold 도우미로 손을 무작위로 맞추고 푼다) → |residual|/(spawn+inject) < 1e-9
    ...
```
(`...`는 시험을 쓰는 사람이 채운다. 단언은 위 주석에 적힌 값이다.)

- [ ] **Step 2: 실패를 확인한다** — `-k "carry or field or store or death_drops or held_item"` → `a_held` 없음으로 FAIL.

- [ ] **Step 3: 구현한다**
  - 먹이 칸을 없애는 곳을 모두 `_drop_food_slot(s, j, ip[Q_FIELD_CAP])`로: `eat`(지금 807행 근처에서 직접 `f_alive=False`·`C_NFOOD -= 1`), `_evict_oldest`(358행), `spoil_food`(427행), `strike`의 먹이 태우기(1332행). 고친 뒤 `grep -n "f_alive\[.*\] = False\|C_NFOOD\] -="`가 `_drop_food_slot` 안 한 곳만 남는지 본다.
  - 놓는 곳을 모두 `field_cap`과 함께: `spawn_food`의 `make_room`·`_place_food`(385·402행), `niche_world`의 상한 비교(661행: `I_FIELD_CAP`이면 `C_NFIELD`)·`make_room`·`_place_food`(686~687행), `_remove`의 사체.
  - `_remove`: 시그니처를 위 Interfaces대로 바꾸고 호출 다섯 곳을 모두 고친다. 들고 있던 물건 내려놓기는 `recycle`과 상관없이 늘 한다.
  - `eat`은 `f_holder[j] >= 0`인 물건을 건너뛴다(먹으려면 내려놓는다).
  - `_reset_body`와 `measure.inject_agent`: `a_held[j] = -1`.
  - `alloc_state`: `a_held`, `f_holder`를 −1로 채운다. 세 필드를 `NEW_SINCE_STAGE1`과 `HASH_DEFAULTS`에 넣는다(과제 1의 `load_arrays` 규칙으로 빠진 옛 체크포인트도 exact로 읽힌다). `World` 생성·`load_arrays` 뒤 `field_cap`이면 `recount_field()`.
  - `set_config`: `field_cap`이 새로 켜지면 `recount_field()`, `carry_enabled`가 꺼지면 `release_all()`.

- [ ] **Step 4: 시험 통과 + 해시** — 전체 시험. 기대: 모두 PASS, `PINNED_HASH_300`·`PINNED_EXT_HASH_300` 그대로, 1부 겨울·부패·상한 교체 시험 그대로.

- [ ] **Step 5: 커밋** — "들기 물건 칸: 손과 물건 필드, 들판 계수기(들기 켤 때만), 저장 칸 상한, 먹이 칸 없애기를 한 함수로".

---

### Task 4: ③ 들기 — 행동·감각·비용·다시 보기

**Files:**
- Modify: `kkumteul/kernels.py` (새 `carry` 커널, `move`의 무게 비용, `_sense_par`·`sense`의 자기 물건 빼기, 새 `write_held`)
- Modify: `kkumteul/world.py` (`step`: `collide` 뒤 `carry`, 감각 뒤 `write_held`)
- Modify: `kkumteul/replay.py` (`_food_snapshot`에 `f_holder`·`f_dropped`, `_read_food`의 같은 물건 알아보기와 들림·내려놓음·저장 칸 치움, `F_CARRY` 깃발, WHY 코드, `seen` 키)
- Modify: `kkumteul/replay.html` (깃발 `F_CARRY`이면 개체 앞에 작은 알, 범례 "든 알", WHY 표에 새 코드)
- Modify: `kkumteul/config.py` (`preset("harsh2")`에 `carry_enabled=True`)
- Test: `tests/test_harsh.py`, `tests/test_replay.py`(기존 `seen` 사전 비교 152행 근처를 새 키에 맞게), `tests/replay_screen_probe.js`(깃발 128 개체의 알 그리기 세기)

**Interfaces:**
- Consumes: 과제 3의 필드, `is_field`, `store_room`, `_drop_food_slot`, `Q_CARRY`, `P_CARRY_COST`, `C_CARRIES`. `config.ACT_CARRY = 9`, `EXT_CARRY = 0`, `Config.ext_in0`.
- Produces (kernels.py): `carry(step, s, fp, ip)` 순차 커널. `write_held(alive, a_held, obs, col)` 순차 커널(들고 있으면 1, 아니면 0).
- Produces (replay.py): 깃발 `F_CARRY = 128`(개체 장면, 비어 있는 마지막 비트). WHY 코드 `WHY_PICKED = 7`(들려서 들판에서 빠짐), `WHY_STORED_OUT = 8`(저장 칸이 차서 치워짐). `seen`에 `"picked"`, `"dropped"`, `"stored_out"`을 늘 넣는다. 기록 한 줄 형식(`FOOD_DTYPE`)은 바꾸지 않는다.

- [ ] **Step 1: 실패하는 시험을 쓴다**

```python
def test_carry_pick_and_drop_by_output():
    # 마음 꺼진 carry 세계, 개체 하나 곁에 들판 알 둘(거리 0.5, 0.8). bout[ACT_CARRY]=+5 → 한 스텝 뒤
    # a_held == 가까운 알, f_holder == 그 개체, f_dropped == 1, C_NFIELD 1 줄고 C_CARRIES == 1
    # bout[ACT_CARRY]=-5 → 한 스텝 뒤 a_held == -1, 알은 그 자리에 f_holder == -1, f_dropped == 1
    ...

def test_sleeping_agent_keeps_item():
    # 마음 켠 carry 세계에서 든 개체를 잠재운다(asleep=1) → after_think가 출력을 0으로 만들어도 a_held 그대로
    ...

def test_held_item_follows_after_collide():
    # 든 개체 둘을 겹쳐 놓아 collide가 밀어내게 → 스텝 뒤 (fx, fy) == (x, y) (매 스텝)
    ...

def test_hand_and_item_point_to_each_other():
    # carry 켠 세계를 마음 켜고 300스텝, 매 스텝 a_held[i] = j ⇔ f_holder[j] = i, 든 물건은 f_alive,
    # C_NFIELD == 살아 있는 들판 알 수
    ...

def test_held_item_not_seen_by_holder():
    # 든 개체의 시야 칸에 자기 알이 안 잡히고 곁 개체 시야에는 잡힌다
    ...

def test_carry_costs_more_to_move():
    # 같은 추력으로 한 스텝: 든 개체의 ΔL_MOVE == 빈손의 1.3배
    ...

def test_held_input_written():
    # 든 개체의 obs[ext_in0 + EXT_CARRY] == 1, 빈손 0. 기본 크기 세계에는 쓰지 않는다(넘침 없음)
    ...

def test_carry_on_default_size_is_rejected():
    with pytest.raises(ValueError):
        small(carry_enabled=True)       # needs_ext

def test_carry_behaviour_ledger_closes():
    # 마음 켠 carry 세계에서 산 개체 절반의 bout[ACT_CARRY]를 +3으로 맞추고 600스텝(실제 carry 커널과 무게 비용이 돈다)
    # → 잔차 < 1e-9, C_CARRIES > 0, L_MOVE가 같은 세계의 carry 끈 갈래와 다르다(장치를 빼면 이 단언이 실패한다)
    ...

def test_carry_world_does_not_depend_on_thread_count():
    # tests/test_core.py의 같은 이름 시험 꼴: carry 켠 세계(bout[ACT_CARRY] 양수)를 스레드 1개와 2개 이상으로 300스텝 → 해시 같음
    ...

def test_branch_into_harsh2(tmp_path):
    # 기본 크기 세계를 400스텝 돌려 저장 → Runner.branch(..., ["preset=harsh2"]) → 이어 300스텝 → 잔차 < 1e-9,
    # 갈라 켠 세계의 a_held 모두 -1에서 시작, 다시 resume하면 해시가 같다(기존 test_branch_into_ext_from_default_size 꼴)
    ...
```
`tests/test_replay.py`:
```python
def test_replay_tracks_carried_item(tmp_path):
    # 마음 꺼진 carry 세계, 개체 하나 곁에 들판 알 하나. 몇 스텝 들고 움직인 뒤 내려놓게 출력을 맞추고 적는다 →
    # 그 알의 첫 행이 WHY_PICKED로 닫히고, 내려놓은 자리에서 행 하나가 열린다(합계 두 행), 든 동안 그 개체 장면의 깃발에 F_CARRY,
    # seen["unknown"] == 0, seen["picked"] == 1, seen["dropped"] == 1
    ...

def test_replay_store_eviction_is_not_unknown(tmp_path):
    # max_agents를 작게 해 저장 칸이 차게 만든 carry 세계를 적는다 → C_STORE_EVICTED 증가분만큼 WHY_STORED_OUT, unknown 늘지 않음
    ...
```

- [ ] **Step 2: 실패를 확인한다** — `-k "carry or held or hand_and_item or sleeping or harsh2 or tracks_carried or store_eviction"` → `carry` 없음으로 FAIL.

- [ ] **Step 3: 구현한다**
  - `carry`: ① 든 물건의 자리를 든 개체 자리로 맞춘다. ② 시작 칸 `step % n`부터 산 개체를 돈다(낮은 칸만 늘 먼저 들지 않게). **자는 개체(`asleep > 0`)는 건너뛴다**(든 채로 잔다). 출력 `act[i, ACT_CARRY] > 0`이고 빈손이면 닿는 거리(`agent_radius + food_radius`) 안의 들지 않은 물건(들판 알·내려놓은 알·사체) 가운데 가장 가까운 것을 든다. 들판 알이었으면 먼저 `store_room`이 True여야 들고, 들면 `C_NFIELD -= 1`, `f_dropped = 1`. 들면 `C_CARRIES += 1`. `≤ 0`이고 들고 있으면 내려놓는다. `n_out <= ACT_CARRY`(기본 크기)면 아무것도 하지 않는다. 먹이 격자는 스텝 처음에 만든 것을 쓴다(든 물건 말고는 먹이가 움직이지 않는다).
  - `move`: `a_held[i] >= 0`이면 이동·회전 비용 × `P_CARRY_COST`(`L_MOVE`). 물건 자리는 `move`가 아니라 `carry` ①이 맞춘다(설계의 "move 안에서 따라 옮기고"를 한 곳으로 줄임 — 감각은 스텝 처음이라 결과가 같다).
  - `_sense_par`: 인자에 `f_holder`를 배열로 직접 더하고 `f_holder[j] == i`인 물건은 그 개체의 시야에서 뺀다.
  - `World.step` 순서: `collide` → `carry`(들기 켜졌을 때) → `eat`. 감각 뒤 `write_held(s.alive, s.a_held, s.obs, cfg.ext_in0 + EXT_CARRY)`(들기 켜졌을 때).
  - `replay`:
    - `_food_snapshot`에 `f_holder`, `f_dropped`를 더한다.
    - 같은 물건 = 칸 + `f_alive` 둘 다 + `f_born` 같음 + `f_kind` 같음, 그리고 들기가 꺼진 세계에서는 지금처럼 자리도 같음(꺼진 세계의 기록이 지금과 비트까지 같게). 들기가 켜진 세계에서는 자리를 보지 않는다.
    - 같은 물건의 `f_holder`가 −1 → 0 이상: 그 행을 `WHY_PICKED`로 닫는다(`seen["picked"]`). 0 이상 → −1: 그 자리에서 새 행을 연다(`seen["dropped"]`, 새로 난 먹이로 세지 않음). 든 동안은 그 개체 장면 깃발에 `F_CARRY`.
    - 사라진 것 가르기 순서: 상한 교체(들기가 켜졌으면 후보를 들판 알로 거름) → 저장 칸 치움(`C_STORE_EVICTED` 증가분만큼, 스텝 앞 사진에서 들지 않은 들판 아닌 물건 가운데 가장 오래된 것, `WHY_STORED_OUT`) → 부패 → 먹힘.
    - 각본이 들기를 끈 스텝의 풀림도 내려놓음으로 적힌다(스텝 앞 사진을 각본 적용 뒤에 다시 뜬다).
  - `replay.html`: 깃발 128이면 개체 머리 앞에 작은 알을 그린다. 범례에 "든 알". 먹이 WHY 표가 있으면 7·8을 더한다.

- [ ] **Step 4: 시험 통과** — 전체 시험과 `node tests/replay_screen_check.js`, `node tests/replay_screen_probe.js`. 기대: 모두 PASS, `PINNED_HASH_300`·`PINNED_EXT_HASH_300` 그대로, `test_replay.py`의 기존 `unknown ≤ 2%` 시험 그대로.

- [ ] **Step 5: 커밋** — "들기 행동: 출력으로 들고 내려놓기, 든 알 따라가기·무게 비용·자기 시야에서 빼기, 들기 입력, 다시 보기가 든 알을 한 물건으로 적음".

---

### Task 5: 0번 숫자 재기 — 가을과 들기 (세션이 직접, 서브에이전트 아님)

- [ ] **Step 1**: 예측을 `wayfinder/stage2-4/results.md`에 먼저 적고 커밋한다(가을을 켜면 가을 끝 들판 알이 상한(2,000)에 닿아 넘침이 `L_DISCARD`로 간다 / 들기를 켜도 처음 1만 스텝 안에는 들고 있는 비율이 거의 0이다(`bout[ACT_CARRY] = -1`에서 시작) / 겨울 감소폭은 0번과 같은 범위 / 부패는 들기 전처럼 0에 가깝다).
- [ ] **Step 2**: 커밋한 코드의 사본(`runs/code-<해시>`)으로, 1부 0번과 같은 체크포인트에서 갈래 넷을 하나씩(2스레드, `memcap --bg --max 1500M`, 약 1만 스텝): `ext` 대조, `harsh`, `harsh` + 가을, `harsh2`(가을 + 들기).
- [ ] **Step 3**: 개체 수·`reseeds`·죽음 원인·겨울 감소폭·가을 `L_DISCARD`·들고 있는 비율·내려놓은 알 수·`C_CARRIES`·`C_STORE_EVICTED`·`C_SPOILED`와 부패 `L_LOST`를 결과 문서에 적는다. 목표를 벗어난 숫자는 설계의 근거에 따라 고치고 Ruling으로 남긴다.
