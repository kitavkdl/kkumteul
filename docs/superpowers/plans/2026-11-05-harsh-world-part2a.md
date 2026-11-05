# 혹독한 세계 2부-가 — 공유 첫 과제·가을 열매철·들기 (구현 계획)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 2부와 암수가 같이 쓰는 저장·해시 틀을 먼저 만들고, 저장이 생길 수 있는 잉여 창(가을 열매철)과 물건을 들고 내려놓는 장치(③)를 만든다.

**Architecture:** 해시는 "기본값과 같으면 넣지 않는다"는 한 규칙으로 새 필드·늘린 배열 꼬리를 건너뛰어 장치를 끈 세계의 해시를 지킨다. 계수기·`budget`은 확장 크기 세계에서만 늘린다. 가을은 `World.season`의 풀 배수 계단에 한 칸을 더한다. 들기는 먹이 칸을 물건으로 쓰고(손 `a_held` ↔ 물건 `f_holder`), 들판 알만 상한에 세는 계수기 `C_NFIELD`를 들기가 켜졌을 때만 쓴다.

**Tech Stack:** Python 3, numpy, numba(`njit`), pytest.

**Spec:** `docs/superpowers/specs/2026-11-05-harsh-world-design.md`(여섯째 판) — "입출력 크기", ① 끝의 "가을 열매철", ③, "장부·계수기·해시·다시 보기", "구현 메모". 검토 기록 `2026-11-05-harsh-world-review.md` 끝 표. 암수 설계 `2026-11-05-sex-design.md`의 "저장·해시·기록" 1·2는 과제 1과 같은 일이다. ④~⑥과 섞기 대조는 2부-나 계획.

## Global Constraints
- 장치는 기본 꺼짐. 모두 끈 세계는 `tests/test_stages.py`의 `PINNED_HASH_300` 세 값이 그대로여야 한다.
- `preset=harsh`(1부 = `ext` + 겨울 + 부패)의 뜻은 바꾸지 않는다. 새 묶음 `preset=harsh2`는 2부 장치를 차례로 덧붙여 가며 2부-나 끝에 완성된다(실험은 실행 폴더의 `config.json`에 설정 전체가 남으므로 재현은 그것으로 한다).
- 에너지가 생기는 길은 먹기와 명시적 주입뿐. 장치마다 `|residual|/(spawn+inject) < 1e-9`.
- `prange` 안에서 상태 묶음(`s.x`)을 거쳐 쓰지 않는다. 새 커널은 순차(`@njit(cache=True)`)로.
- 난수 용도 번호는 바꾸지 않고, 이 계획이 처음 쓰는 번호는 없다(과제 1이 표에 이름만 적는다).
- 커널에 인자를 더하면 기본값으로 옛 호출(`measure`, `bench.kernel_breakdown`, `knockout.wrap`, 시험)을 살린다.
- 공개 레포: 코드·주석에 로컬 경로, "과제 N"·"2부" 같은 작업 과정 표현을 쓰지 않는다(상태를 말하는 표현으로).
- 시험 실행: `NUMBA_NUM_THREADS=2 ~/.local/bin/memcap --max 2G -E NUMBA_NUM_THREADS -- nice -n 10 ../../../.venv/bin/python -m pytest -q -p no:cacheprovider <대상>`(작업 폴더에서). 전체 약 2.5분. 서브에이전트는 시험만 돌리고 시뮬레이션 실험은 돌리지 않는다. 시험 임시 폴더는 pytest `tmp_path`만(서버 `/tmp`는 메모리 디스크).
- 커밋 메시지는 한국어, 고친 파일만 `git add`(`.crew`, `.superpowers` 제외), 커밋 뒤 `git push origin worktree-night-1105c`.

## Review Focus
1. **1부 확장 세계 체크포인트(계수기 24칸)를 새 코드로 이어 돌리면** 계수기가 40칸으로 채워 읽히고, 그 세계의 해시가 꼬리 0 규칙 덕에 옛 코드와 같다 — 과제 1 `test_ext_hash_ignores_zero_tail`.
2. **들고 있는 개체가 죽으면** 물건이 그 자리에 떨어지고, 죽은 칸이 새 개체로 다시 쓰여도 그 개체가 물건을 든 꼴이 되지 않는다 — 과제 3 `test_death_drops_item_and_slot_reuse_is_empty_handed`.
3. **각본이 실행 중에 `carry_enabled`를 켜면** 들판 계수기가 그 순간의 들판 알 수로 맞춰지고, 끄면 든 물건이 모두 내려진다 — 과제 3 `test_carry_toggle_recounts_and_releases`.
4. **들기 + 상한 교체(`food_cap_replace`) + 부패**를 같이 켜면 교체가 든 알·내려놓은 알을 고르지 않고 장부가 맞다 — 과제 3 `test_cap_replace_skips_stored_items`.
5. **들기를 켠 실행을 다시 보기로 적으면** 든 알이 "사라짐 + 새 먹이"로 적히지 않고 `unknown ≤ 2%`를 지킨다 — 과제 4 `test_replay_tracks_carried_item`.

---

### Task 1: 공유 첫 과제 — 해시 규칙·번호표·배열 길이

**Files:**
- Modify: `kkumteul/kernels.py` (계수기 번호 영역 `C_SPOILED` 아래, `N_COUNTERS`, `alloc_state`, `PERTURB_FLEX` 근처에 기본값 표)
- Modify: `kkumteul/world.py` (`state_hash`)
- Modify: `kkumteul/config.py` (`Config.ext_size` 속성)
- Modify: `kkumteul/rng.py` (용도 번호 표 끝)
- Modify: `kkumteul/_cache.py` (`_SOURCES`)
- Test: `tests/test_harsh.py`

**Interfaces:**
- Produces (kernels.py): 계수기 번호 `C_NFIELD = 22`, `C_CARRIES = 23`, `C_STORE_EVICTED = 24`, `C_DEATH_PREDATED = 25`, `C_SCAVENGED = 26`, `C_PREY_KILLED = 27`, `C_ANIMAL_REPELLED = 28`, `C_CHASES = 29`, `C_MATINGS = 30`, `C_DEATH_CONTEST = 31`, `C_INFECTIONS = 32`, `C_DEATH_INFECTED = 33`(34·35 암수 예비), `C_RESCUES = 36`(37~39 예비). `N_COUNTERS_BASE = 24`, `N_COUNTERS_EXT = 40`. `N_COUNTERS`는 지우지 말고 `N_COUNTERS_BASE`와 같은 값으로 둔다(바깥 코드 호환). `N_BUDGET_BASE = 2`, `N_BUDGET_EXT = 3`.
- Produces (kernels.py): `HASH_DEFAULTS: dict[str, int | float]` — 저장 필드 이름 → 기본값. 지금은 `{"elig_in": 0, "elig_rec": 0}`(지금의 `PERTURB_FLEX` 건너뛰기를 이 표로 옮김). 과제 3이 `a_held: -1`, `f_holder: -1`, `f_dropped: 0`을 더한다.
- Produces (config.py): `Config.ext_size -> bool` = `n_spare_in >= N_STAGE_IN + N_EXT_IN and n_spare_out >= N_STAGE_OUT + N_EXT_OUT`.
- Produces (rng.py): `S_ANIMAL_POS = 19`, `S_WANDER = 20`, `S_CHASE = 21`, `S_CONFUSE = 22`, `S_MEAT = 23`, `S_SHUFFLE = 24`, `S_SEX = 25`, `S_ORNAMENT = 26`, `S_CROSS = 27`, `S_PATHOGEN = 28`, `S_RESERVOIR = 29`, `S_SEX_SPARE = 30` — 이름과 한 줄 뜻만(아직 아무도 부르지 않음).

- [ ] **Step 1: 실패하는 시험을 쓴다** (`tests/test_harsh.py`)

```python
def test_counter_and_budget_length_follow_ext_size():
    assert len(World(small()).s.counters) == k.N_COUNTERS_BASE   # 기본 크기
    w = World(small(**preset("ext")))
    assert len(w.s.counters) == k.N_COUNTERS_EXT and len(w.s.budget) == k.N_BUDGET_EXT

def test_ext_hash_ignores_zero_tail():
    w = World(small(**preset("harsh")))
    w.run(50)
    h = w.state_hash()
    w.s = w.s._replace(counters=w.s.counters[:24].copy(), budget=w.s.budget[:2].copy())
    assert w.state_hash() == h          # 꼬리가 0이면 옛 길이 해시와 같다
    w2 = World(small(**preset("harsh"))); w2.run(50)
    w2.s.counters[k.C_RESCUES] = 1
    assert w2.state_hash() != h         # 꼬리에 값이 있으면 들어간다
```
(`small()`·`preset()`의 실제 쓰임은 `tests/test_harsh.py`의 기존 시험을 따른다. 기본값 건너뛰기 자체는 `PINNED_HASH_300`이 지키고, 기본값이 아닐 때 들어가는지는 과제 3의 `test_carry_fields_default_and_hash_unchanged`가 지킨다.)

- [ ] **Step 2: 실패를 확인한다** — `... tests/test_harsh.py -k "length_follow or zero_tail"` → `N_COUNTERS_BASE` 없음으로 FAIL.

- [ ] **Step 3: 구현한다**
  - `alloc_state(cfg)`: `counters` 길이 = `N_COUNTERS_EXT if cfg.ext_size else N_COUNTERS_BASE`, `budget` 길이도 같은 조건으로 3 / 2. `load_arrays`의 채워 읽기는 이미 `counters`·`budget`의 커지는 쪽을 받는다(바꾸지 않는다).
  - `state_hash`: 개체·먹이 필드 반복에서 `name in HASH_DEFAULTS`이고 살아 있는 칸 값이 모두 그 기본값이면 건너뛴다(`(v == d).all()` — `.any()`가 아니다, 기본값 −1이 있다). 먹이 필드 반복에도 같은 건너뛰기. `counters`는 `N_COUNTERS_BASE` 뒤 꼬리가 모두 0이면 앞 24칸만, `budget`은 `N_BUDGET_BASE` 뒤가 모두 0이면 앞 2칸만 넣는다.
  - `_SOURCES`에 `"stages.py", "world.py"`를 더한다.

- [ ] **Step 4: 시험 통과 + `PINNED_HASH_300` 그대로** — 전체 시험을 돌린다. 기대: 모두 PASS(1부 194건 + 새 2건). 특히 `tests/test_stages.py`의 박아 둔 해시와 `test_harsh.py`의 갈라 켜기·resume 해시 시험.

- [ ] **Step 5: 커밋** — `git add kkumteul/kernels.py kkumteul/world.py kkumteul/config.py kkumteul/rng.py kkumteul/_cache.py tests/test_harsh.py` → "공유 저장 틀: 해시 기본값 규칙, 계수기·난수 번호표, 확장 세계에서만 계수기 40칸".

---

### Task 2: 가을 열매철

**Files:**
- Modify: `kkumteul/config.py` (겨울 설정 아래 `autumn_len`, `autumn_mult`, `validate`, `preset()`)
- Modify: `kkumteul/world.py:102-122` (`in_winter` 옆 `in_autumn`, `season`)
- Test: `tests/test_harsh.py`

**Interfaces:**
- Produces (config.py): `autumn_len: int = 0`, `autumn_mult: float = 1.0`(주석: 겨울 바로 앞 `autumn_len` 스텝의 풀 배수, 지구의 가을 열매 몰림, 임시 결정). 묶음 `preset("harsh2")` = `harsh` + `autumn_len=800, autumn_mult=2.0`(과제 3이 `carry_enabled=True`를 더한다). `preset()`의 모르는 이름 오류문에 `harsh2`를 더한다.
- Produces (world.py): `World.in_autumn(t) -> bool` = 겨울이 켜져 있고 `P - W - A <= t % P < P - W`. `World.season(t)`의 첫 값(풀 배수): 겨울 `winter_mult`, 가을 `autumn_mult`, 나머지 `(P - W*winter_mult - A*autumn_mult) / (P - W - A)`. 기초 대사 배수와 sin·cos는 그대로.

- [ ] **Step 1: 실패하는 시험을 쓴다**

```python
def test_autumn_off_keeps_season_values():
    w = World(small(**preset("harsh")))
    assert w.season(100)[0] == pytest.approx((4000 - 1200 * 0.3) / 2800)   # 1.3, 1부와 같음
    assert not w.in_autumn(2500)

def test_autumn_mast_keeps_year_mean():
    w = World(small(**preset("harsh2")))
    P = w.cfg.season_period
    g = [w.season(t)[0] for t in range(P)]
    assert sum(g) / P == pytest.approx(1.0, rel=1e-12)
    assert w.season(2000)[0] == pytest.approx(2.0) and w.in_autumn(2000)    # 2000..2799 가을
    assert w.season(100)[0] == pytest.approx(1.02)
    assert w.season(2000)[1] == 1.0                                          # 가을 기초 대사는 그대로

def test_autumn_validate():
    with pytest.raises(ValueError):
        small(**preset("harsh"), autumn_len=2800)          # 가을 + 겨울이 해 전체
    with pytest.raises(ValueError):
        small(**preset("harsh"), autumn_len=800, autumn_mult=0.5)   # 가을 배수는 1 이상
    with pytest.raises(ValueError):
        small(**preset("harsh"), autumn_len=800, autumn_mult=5.0)   # 나머지 철 배수가 음수
```

- [ ] **Step 2: 실패를 확인한다** — `-k autumn` → `in_autumn` 없음으로 FAIL.

- [ ] **Step 3: 구현한다** — `validate`: `autumn_len >= 0`, `autumn_len + winter_len < season_period`, `autumn_mult >= 1`, 나머지 철 배수 `>= 0`. `season`이 가을이면 배수 `autumn_mult`. `spoil_mult`는 겨울만 본다(바꾸지 않는다).

- [ ] **Step 4: 시험 통과** — `tests/test_harsh.py` 전체와 `tests/test_stages.py`. 기대: PASS, 1부 겨울 시험(`test_winter_*`, `test_niche_share_follows_season_mult`) 그대로.

- [ ] **Step 5: 커밋** — "가을 열매철: 겨울 앞 풀 배수 계단(기본 꺼짐), harsh2 묶음".

---

### Task 3: ③ 들기 — 물건 칸과 장부

**Files:**
- Modify: `kkumteul/kernels.py` (`AGENT_FIELDS`, `FOOD_FIELDS`, `HASH_DEFAULTS`, `alloc_state`, `pack_params`의 `ip` 칸, `_place_food`, `_drop_food_slot`, `_evict_oldest`, `make_room`, `spawn_food`, `spoil_food`, `eat`, `_reset_body`, `_remove`, `strike`)
- Modify: `kkumteul/stages.py` (`niche_world`의 상한 비교, `isp` 칸)
- Modify: `kkumteul/world.py` (`NEW_SINCE_STAGE1`, `load_arrays`의 채워 읽기, `set_config`, 들판 수 다시 세기)
- Modify: `kkumteul/measure.py` (`inject_agent`가 `a_held=-1`)
- Modify: `kkumteul/config.py` (`carry_enabled`, `carry_cost_mult`, `needs_ext`, `field_cap`)
- Test: `tests/test_harsh.py`

**Interfaces:**
- Consumes: 과제 1의 `C_NFIELD`, `C_CARRIES`, `C_STORE_EVICTED`, `HASH_DEFAULTS`.
- Produces (config.py): `carry_enabled: bool = False`, `carry_cost_mult: float = 1.3`(지구 참고 없음, 임시). `needs_ext`에 `carry_enabled`를 더한다. `Config.field_cap -> bool` = `carry_enabled`(2부-나가 `or prey_enabled`를 더한다).
- Produces (kernels.py): 필드 `a_held`(int32, 개체, 저장), `f_holder`(int32, 먹이, 저장), `f_dropped`(int8, 먹이, 저장 — 1이면 "옮겨진 적 있는 물건", 들판 알이 아님). 기본값 −1, −1, 0. `ip` 칸 `Q_FIELD_CAP`(= `int(cfg.field_cap)`), `Q_CARRY`(= `int(cfg.carry_enabled)`), `fp` 칸 `P_CARRY_COST`. `stages.py`의 `isp`에 `I_FIELD_CAP`.
- Produces (kernels.py): `is_field(s, j) -> bool` = `f_dropped[j] == 0 and f_kind[j] != KIND_CORPSE`(2부-나가 고기도 뺀다). `_place_food(s, px, py, e, kind, step, field_cap=0)` — 놓으면 `f_holder=-1`, `f_dropped=0`, `field_cap`이고 `is_field`면 `C_NFIELD += 1`. `_drop_food_slot(s, j, field_cap=0)` — 손을 풀고(`a_held[f_holder[j]] = -1`), `field_cap`이고 들판 알이면 `C_NFIELD -= 1`. `_evict_oldest(s, field_cap=0)` — `field_cap`이면 들판 알만 후보. `make_room(s, cap, replace, field_cap=0)` — `field_cap`이면 `C_NFIELD`로 비교. `evict_stored(s) -> bool` — 들고 있지 않은 `f_dropped==1` 물건 가운데 가장 먼저 놓인 것을 `L_LOST`로 치우고 `C_STORE_EVICTED += 1`. `store_room(s, n_agents_max) -> bool` — 들판 아닌 물건 수(`C_NFOOD − C_NFIELD`)가 `max_agents`보다 작으면 True, 아니면 `evict_stored`의 결과.
- Produces (world.py): `World.recount_field()` — `C_NFIELD`를 살아 있는 들판 알 수로 다시 센다(numpy). `World.release_all()` — 모든 손을 푼다(`f_dropped`는 그대로).

- [ ] **Step 1: 실패하는 시험을 쓴다** (모두 `carry_enabled=True` + `preset("ext")` 세계, 손으로 물건·손을 맞춘다. 행동 커널은 과제 4 — 여기서는 `a_held`/`f_holder`를 시험이 직접 쓴다)

```python
def test_carry_fields_default_and_hash_unchanged():
    w = World(small(**preset("ext")))
    assert (w.s.a_held == -1).all() and (w.s.f_holder == -1).all()
    w.run(20); h = w.state_hash()
    i = int(np.flatnonzero(w.s.alive)[0])
    w.s.a_held[i] = 5
    assert w.state_hash() != h          # 기본값이 아니면 해시에 들어간다
    w.s.a_held[i] = -1
    assert w.state_hash() == h          # 모두 기본값이면 빠진다(장치를 끈 세계는 PINNED_HASH_300이 지킨다)

def test_eat_releases_and_counts_field():
    # 들판 알을 먹으면 C_NFIELD가 줄고, 내려놓은 알(f_dropped=1)을 먹으면 C_NFIELD는 그대로
    ...
    assert w.s.counters[k.C_NFIELD] == field_before - 1

def test_death_drops_item_and_slot_reuse_is_empty_handed():
    # 물건을 든 개체를 굶겨 죽인다 → 물건 f_holder == -1, f_dropped == 1, 자리 = 죽은 자리, f_alive
    # 이어 그 칸에 새 개체가 나면(process_births/reseed) a_held == -1
    ...

def test_cap_counts_field_only():
    # food_cap=50, 들판 알 50 + 내려놓은 알 30: 새 유입이 놓인다(C_NFIELD < cap 이 될 때까지가 아니라
    # C_NFIELD == cap이면 막힘), 내려놓은 알 30은 상한에 세지 않는다. 장부 잔차 < 1e-9
    ...

def test_cap_replace_skips_stored_items():
    # food_cap_replace=True + spoil_enabled: 가장 오래된 것이 내려놓은 알이어도 교체는 들판 알을 고른다
    ...

def test_store_room_evicts_oldest_dropped():
    # 들판 아닌 물건이 max_agents개면 store_room이 가장 오래된 내려놓은 알을 L_LOST로 치우고 C_STORE_EVICTED += 1,
    # 든 알만 남았으면 False
    ...

def test_carry_toggle_recounts_and_releases():
    # carry 꺼진 ext 세계를 돌린 뒤 set_config로 켬 → C_NFIELD == 들판 알 수(사체 제외)
    # 손을 몇 개 맞춘 뒤 set_config로 끔 → a_held 모두 -1, f_holder 모두 -1
    ...

def test_carry_ledger_closes():
    # carry 켠 ext 세계 300스텝(손을 매 50스텝 무작위로 맞추고 푸는 도우미) → |residual|/(spawn+inject) < 1e-9
    ...
```
(`...`는 시험을 쓰는 사람이 채운다. 단언은 위 주석에 적힌 값이다.)

- [ ] **Step 2: 실패를 확인한다** — `-k "carry or field or store or death_drops"` → `a_held` 없음으로 FAIL.

- [ ] **Step 3: 구현한다**
  - 먹이 칸을 없애는 곳을 모두 `_drop_food_slot(s, j, ip[Q_FIELD_CAP])`로: `eat`(지금 직접 `f_alive=False`·`C_NFOOD -= 1`), `_evict_oldest`, `spoil_food`, `strike`의 먹이 태우기. `niche_world`도 `I_FIELD_CAP`이면 `C_NFIELD`로 상한 비교.
  - 사체(`_remove`의 `_place_food`)는 `field_cap`이면 먼저 `store_room`; 못 얻으면 지금처럼 `L_LOST`. 들고 있던 개체가 죽으면 그 물건을 내려놓는다(`f_holder=-1`, `a_held=-1`, `f_dropped`는 이미 1).
  - `_reset_body`와 `measure.inject_agent`: `a_held[j] = -1`.
  - `alloc_state`: `a_held`, `f_holder`를 −1로 채운다. `load_arrays`: 세 필드를 `NEW_SINCE_STAGE1`에 넣고, 빠졌으면 기본값으로 채운다. 읽은 뒤 `field_cap`이면 `recount_field()`.
  - `set_config`: `field_cap`이 새로 켜지면 `recount_field()`, `carry_enabled`가 꺼지면 `release_all()`.
  - `HASH_DEFAULTS`에 세 필드.
  - `eat`은 `f_holder[j] >= 0`인 물건을 건너뛴다(먹으려면 내려놓는다).

- [ ] **Step 4: 시험 통과 + 해시** — 전체 시험. 기대: 모두 PASS, `PINNED_HASH_300` 그대로, 1부 겨울·부패·상한 교체 시험 그대로.

- [ ] **Step 5: 커밋** — "들기 물건 칸: 손과 물건 필드, 들판 계수기(들기 켤 때만), 저장 칸 상한, 먹이 칸 없애기를 한 함수로".

---

### Task 4: ③ 들기 — 행동·감각·비용·다시 보기

**Files:**
- Modify: `kkumteul/kernels.py` (새 `carry` 커널, `move`의 무게 비용, `_sense_par`·`sense`의 자기 물건 빼기, 새 `write_held`)
- Modify: `kkumteul/world.py` (`step`: `collide` 뒤 `carry`, 감각 뒤 `write_held`)
- Modify: `kkumteul/replay.py` (`_read_food`의 같은 물건 알아보기, 교체 재구성 거르기, WHY 코드)
- Modify: `kkumteul/config.py` (`preset("harsh2")`에 `carry_enabled=True`)
- Test: `tests/test_harsh.py`, `tests/test_replay.py`

**Interfaces:**
- Consumes: 과제 3의 필드, `is_field`, `store_room`, `_drop_food_slot`, `Q_CARRY`, `P_CARRY_COST`, `C_CARRIES`. `config.ACT_CARRY = 9`, `EXT_CARRY = 0`, `Config.ext_in0`.
- Produces (kernels.py): `carry(step, s, fp, ip)` 순차 커널. `write_held(alive, a_held, obs, col)` 순차 커널(들고 있으면 1, 아니면 0).
- Produces (replay.py): WHY 코드 `WHY_PICKED`, `WHY_DROPPED`(기존 번호 뒤에 덧붙임 — 화면 쪽 표가 있으면 같이).

- [ ] **Step 1: 실패하는 시험을 쓴다**

```python
def test_carry_pick_and_drop_by_output():
    # 마음 꺼진 carry 세계, 개체 하나 곁에 들판 알 둘(거리 0.5, 0.8). bout[ACT_CARRY]=+5 → 한 스텝 뒤
    # a_held == 가까운 알, f_holder == 그 개체, f_dropped == 1, C_NFIELD 1 줄고 C_CARRIES == 1
    # bout[ACT_CARRY]=-5 → 한 스텝 뒤 a_held == -1, 알은 그 자리에 f_holder == -1
    ...

def test_held_item_follows_after_collide():
    # 든 개체 둘을 겹쳐 놓아 collide가 밀어내게 → 스텝 뒤 (fx, fy) == (x, y) (매 스텝)
    ...

def test_hand_and_item_point_to_each_other():
    # carry 켠 세계를 마음 켜고 300스텝, 매 스텝 a_held[i] = j ⇔ f_holder[j] = i, 든 물건은 f_alive
    ...

def test_held_item_not_eaten_and_not_seen_by_holder():
    # 든 알은 곁의 배고픈 개체도 먹지 못한다 / 든 개체의 시야 칸에 자기 알이 안 잡히고 곁 개체 시야에는 잡힌다
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
```
`tests/test_replay.py`:
```python
def test_replay_tracks_carried_item(tmp_path):
    # carry를 켜고 bout[ACT_CARRY]를 양수로 맞춘 짧은 실행을 적는다 → 기록에서 든 알 수만큼 "사라짐+새 먹이"가 생기지 않음,
    # seen["unknown"] / 사라진 수 ≤ 0.02, WHY_PICKED가 한 번 이상
    ...
```

- [ ] **Step 2: 실패를 확인한다** — `-k "carry or held or hand_and_item or tracks_carried"` → `carry` 없음으로 FAIL.

- [ ] **Step 3: 구현한다**
  - `carry`: ① 든 물건의 자리를 든 개체 자리로 맞춘다. ② 시작 칸 `step % n`부터 산 개체를 돈다(낮은 칸만 늘 먼저 들지 않게). 출력 `act[i, ACT_CARRY] > 0`이고 빈손이면 닿는 거리(`agent_radius + food_radius`) 안의 들지 않은 물건 가운데 가장 가까운 것을 든다. 들판 알이었으면 먼저 `store_room`이 True여야 들고, 들면 `C_NFIELD -= 1`, `f_dropped = 1`. `≤ 0`이고 들고 있으면 내려놓는다. `n_out <= ACT_CARRY`(기본 크기)면 아무것도 하지 않는다. 먹이 격자는 스텝 처음에 만든 것을 쓴다(든 물건 말고는 먹이가 움직이지 않는다).
  - `move`: `a_held[i] >= 0`이면 이동·회전 비용 × `P_CARRY_COST`(`L_MOVE`). 물건 자리는 `move`가 아니라 `carry` ①이 맞춘다.
  - `_sense_par`: 인자에 `f_holder`를 더하고 `f_holder[j] == i`인 물건은 그 개체의 시야에서 뺀다.
  - `World.step` 순서: `collide` → `carry`(들기 켜졌을 때) → `eat`. 감각 뒤 `write_held(s.alive, s.a_held, s.obs, cfg.ext_in0 + EXT_CARRY)`(들기 켜졌을 때).
  - `replay._read_food`: 같은 물건 = 칸 + `f_alive` 둘 다 + `f_born` 같음 + `f_kind` 같음(자리는 보지 않는다). 같은 물건인데 `f_holder`가 −1 → 0 이상이면 `WHY_PICKED`, 반대면 `WHY_DROPPED` 사건으로 적고, 그려지는 자리는 그 물건을 따라가게 한다(지금 기록 형식에 자리 갱신이 없으면 이 사건으로 갱신). 들기가 켜졌으면 교체 재구성 후보를 들판 알(`f_dropped == 0`, 사체 아님)로 거른다.

- [ ] **Step 4: 시험 통과** — 전체 시험. 기대: 모두 PASS, `PINNED_HASH_300` 그대로, `test_replay.py`의 기존 `unknown ≤ 2%` 시험 그대로.

- [ ] **Step 5: 커밋** — "들기 행동: 출력으로 들고 내려놓기, 든 알 따라가기·무게 비용·자기 시야에서 빼기, 들기 입력, 다시 보기가 든 알을 같은 물건으로 적음".

---

### Task 5: 0번 숫자 재기 — 가을과 들기 (세션이 직접, 서브에이전트 아님)

- [ ] **Step 1**: 예측을 `wayfinder/stage2-4/results.md`에 먼저 적고 커밋한다(가을을 켜면 가을 끝 들판 알이 상한(2,000)에 닿아 넘침이 `L_DISCARD`로 간다 / 들기를 켜도 처음 1만 스텝 안에는 들고 있는 비율이 거의 0이다(`bout[ACT_CARRY] = -1`에서 시작) / 겨울 감소폭은 0번과 같은 범위).
- [ ] **Step 2**: 커밋한 코드의 사본(`runs/code-<해시>`)으로, 1부 0번과 같은 체크포인트에서 갈래 넷을 하나씩(2스레드, `memcap --bg --max 1500M`, 약 1만 스텝): `ext` 대조, `harsh`, `harsh` + 가을, `harsh2`(가을 + 들기).
- [ ] **Step 3**: 개체 수·`reseeds`·죽음 원인·겨울 감소폭·가을 `L_DISCARD`·들고 있는 비율·내려놓은 알 수·`C_STORE_EVICTED`를 결과 문서에 적는다. 목표를 벗어난 숫자는 설계의 근거에 따라 고치고 Ruling으로 남긴다.
