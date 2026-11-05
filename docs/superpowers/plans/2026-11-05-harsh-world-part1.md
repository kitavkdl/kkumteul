# 혹독한 세계 1부 — 확장 칸·갈라 켜기·겨울·부패 (구현 계획)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 혹독한 세계 장치 ①(겨울)·②(부패)와, 모든 새 장치가 쓸 확장 입출력 칸과 "옛 체크포인트에서 확장 크기로 갈라 켜기" 경로를 만든다.

**Architecture:** 기본 입출력 뒤에 늘 같은 자리의 확장 블록(입력 13, 출력 1)을 둔다. 크기는 `preset=ext`가 올리고, 체크포인트를 읽을 때 `max(저장값, 요청값)`으로 올려 배열을 0으로 채워 읽는다. 겨울은 스텝만으로 정해지는 계단 함수로 풀 유입과 기초 대사에 배수를 건다(인자로 넘김). 부패는 먹이의 `f_born`을 보고 순차 커널에서 줄인다.

**Tech Stack:** Python 3, numpy, numba(`njit`), pytest.

**Spec:** `docs/superpowers/specs/2026-11-05-harsh-world-design.md`(셋째 판). ③~⑤(들기·사냥감·포식자/약탈자)는 지구 참고 조사(`docs/research/earth-references.md`) 뒤 2부 계획으로 쓴다.

## Global Constraints
- 장치는 기본 꺼짐. 모두 끈 세계는 `tests/test_stages.py`의 `PINNED_HASH_300` 세 값이 그대로여야 한다.
- 에너지가 생기는 길은 먹기와 명시적 주입뿐. 장치마다 `|residual|/(spawn+inject) < 1e-9`.
- `prange` 안에서 상태 묶음(`s.x`)을 거쳐 쓰지 않는다. 새 커널은 순차(`@njit(cache=True)`)로.
- 난수 용도 번호(`rng.py`)는 바꾸지 않는다. 이 계획은 새 난수를 쓰지 않는다.
- 커널에 인자를 더하면 기본값(`=1.0`)으로 옛 호출(`measure`, `bench.kernel_breakdown`, `knockout.wrap`, 시험)을 살린다.
- 시험 실행: `NUMBA_NUM_THREADS=2 memcap --max 2G -E NUMBA_NUM_THREADS -- nice -n 10 <원래 폴더>/.venv/bin/python -m pytest -q -p no:cacheprovider <대상>`. 전체 약 2분. 서브에이전트는 시험만 돌리고 시뮬레이션 실험은 돌리지 않는다.
- 커밋 메시지는 한국어, 고친 파일만 `git add`, 커밋 뒤 `git push`(가지 `worktree-night-1105c`).

## Review Focus
1. **확장 크기 세계를 다시 `preset=ext`로 갈라도** 크기가 두 번 늘지 않는다(26/7 그대로) — 과제 2 `test_branch_ext_is_idempotent`.
2. **갈라 켠 확장 세계를 `resume`하면** 끊지 않은 것과 해시가 같다 — 과제 2 `test_ext_branch_resumes_exactly`.
3. **각본이 실행 중에 `season_enabled=1`을 켰는데 크기가 기본이면** 넘쳐 쓰지 않고 `validate`가 분명한 오류를 낸다 — 과제 1 `test_harsh_device_needs_ext_size`.
4. **겨울과 거름(`stage4`)을 같이 켜면** 거름 몫도 겨울 배수를 따라 장부가 맞다 — 과제 3 `test_winter_with_niche_conserves_energy`.
5. **부패와 상한 교체(`food_cap_replace`)를 같이 켜면** 둘 다 `L_LOST`로 가고 장부가 맞다 — 과제 4 `test_spoil_with_cap_replace_conserves_energy`.

---

### Task 1: 확장 칸 상수·묶음·검사

**Files:**
- Modify: `kkumteul/config.py` (상수 영역 `N_STAGE_IN` 아래, `preset()`, `validate()`, 새 설정 필드는 과제 3·4에서)
- Modify: `kkumteul/kernels.py:842-868` (`random_genome`)
- Test: `tests/test_harsh.py` (새 파일, `tests/test_stages.py`의 `small()`·`rel_residual()`을 가져와 쓴다)

**Interfaces:**
- Produces (config.py): `N_EXT_IN = 13`, `N_EXT_OUT = 1`, `ACT_CARRY = 9`(= `N_ACTIONS + N_STAGE_OUT`), 확장 입력 안 자리 `EXT_CARRY = 0`, `EXT_SEASON = 1`(sin, cos 두 칸), `EXT_PREY = 3`, `EXT_PRED = 5`, `EXT_SCAV = 7`, `EXT_SEX = 9`(네 칸). 확장 입력의 절대 번호 = `2*vision_sectors + N_INTERNAL + N_STAGE_IN + EXT_*`.
- Produces: `Config.needs_ext -> bool` (지금은 `season_enabled or spoil_enabled`; 2부·암수가 여기에 덧붙인다). `Config.ext_in0 -> int`(확장 블록 첫 입력 번호).
- Produces: `preset("ext") == dict(n_spare_in=26, n_spare_out=7)`, `preset("harsh") == {**preset("ext"), season_enabled=True, spoil_enabled=True}`(2부가 장치를 덧붙인다). 모르는 이름 오류 문구에 `ext, harsh`를 더한다.

- [ ] **Step 1: 실패하는 시험 쓰기** — `tests/test_harsh.py`

```python
def test_ext_preset_sizes():
    c = small("ext")
    assert (c.n_spare_in, c.n_spare_out) == (N_STAGE_IN + N_EXT_IN, N_STAGE_OUT + N_EXT_OUT) == (26, 7)
    assert c.n_out > ACT_CARRY and c.ext_in0 == 2 * c.vision_sectors + N_INTERNAL + N_STAGE_IN

def test_harsh_device_needs_ext_size():
    with pytest.raises(ValueError, match="ext"):
        small(season_enabled=True)          # 기본 크기에 장치만 켬
    small("ext", season_enabled=True)       # 확장 크기면 된다

def test_ext_world_gives_new_agents_carry_bias_off():
    w = World(small("ext"))
    assert (w.s.bout[w.s.alive, ACT_CARRY] == -1.0).all()
```
(`season_enabled`는 과제 3에서 생기므로, 이 과제에서는 필드 두 개 `season_enabled: bool = False`, `spoil_enabled: bool = False`만 먼저 넣는다.)

- [ ] **Step 2: 실패 확인** — `... pytest -q tests/test_harsh.py` → ImportError(`N_EXT_IN`).
- [ ] **Step 3: 구현** — 상수·`needs_ext`·`ext_in0`·묶음 두 개. `validate`: `needs_ext`이면 `n_spare_in >= N_STAGE_IN + N_EXT_IN and n_spare_out >= N_STAGE_OUT + N_EXT_OUT`, 아니면 `ValueError("혹독한 세계·암수 장치는 확장 크기가 필요하다: preset=ext")`. `random_genome`: `if n_out > ACT_CARRY: s.bout[j, ACT_CARRY] = -1.0`(기본 크기 9칸에서는 건드리지 않는다 — numba는 범위를 검사하지 않아 다음 개체를 덮는다).
- [ ] **Step 4: 통과 확인** — `tests/test_harsh.py`와 `tests/test_stages.py::test_worlds_without_harsh_keep_pinned_hash` 통과.
- [ ] **Step 5: 커밋** — `혹독한 세계 1부-1: 확장 칸 상수와 preset=ext·harsh, 장치는 확장 크기를 요구`

### Task 2: 갈라 켜기 경로 — 설정을 먼저 만들고 크기를 올려 읽기

**Files:**
- Modify: `kkumteul/checkpoint.py:71-97` (`load`, 새 `read_config`)
- Modify: `kkumteul/run.py:220-280` (`Runner.branch`)
- Modify: `kkumteul/replay.py:615-630` (`record`)
- Test: `tests/test_harsh.py`

**Interfaces:**
- Consumes: 과제 1의 `ACT_CARRY`, `preset("ext")`.
- Produces: `ckpt.read_config(path) -> Config` — 체크포인트 json의 설정을 지금 최소 크기(`N_STAGE_*`)로 올린 것. 세계는 만들지 않는다.
- Produces: `ckpt.load(path, cfg=None, verify=True)` — `cfg`가 있으면 `n_spare_in/out`을 `max(저장값, cfg값, N_STAGE_*)`로 올려 세계를 만들고(`upgraded`면 저장 해시 비교 건너뜀), 같은 크기로 맞춘 `cfg`를 `set_config`한다. 돌려주는 meta에 `"enlarged_out": (옛 n_out, 새 n_out)`을 넣는다(크기가 안 바뀌면 같은 두 값).
- Produces: `World.calm_new_outputs(old_n_out: int) -> list[str]` — 살아 있는 개체의 `bout[:, o]`를 새 개체 기본값으로 맞춘다(`old_n_out <= ACT_CARRY < n_out`이면 `ACT_CARRY`를 -1). 손댄 것을 `["bout[ACT_CARRY]=-1"]`처럼 돌려준다.
- `Runner.branch`: `old = ckpt.read_config(path)`(각본 부모면 지금처럼 `config.json`), `cfg = old.with_overrides(...)`(+ seed·passive), `world, cmeta = ckpt.load(path, cfg)`, `edits += world.calm_new_outputs(cmeta["enlarged_out"][0])`. 이어서 `calm_attack` 처리는 그대로. `diff`는 크기 필드를 포함해 그대로 계산.
- `replay.record`: `overrides`가 있으면 `cfg = ckpt.read_config(path).with_overrides(overrides)`로 `ckpt.load(path, cfg)`, 이어 `calm_new_outputs`. 없으면 지금처럼.

- [ ] **Step 1: 실패하는 시험 쓰기**

```python
def _main(tmp_path, *presets):
    r = Runner.new(tmp_path / "main", small(*presets)); r.loop(400, quiet=True); r.close()

def test_branch_into_ext_from_default_size(tmp_path):
    _main(tmp_path, "stage4")
    src, _ = ckpt.load(next((tmp_path / "main" / "checkpoints").glob("*400*.npz")))
    r = Runner.branch(tmp_path / "main", tmp_path / "ext", 400, ["preset=ext"], False, False)
    a, n_in0, n_out0 = r.world.s.alive, src.cfg.n_in, src.cfg.n_out
    assert r.world.cfg.n_spare_in == 26 and r.world.s.Win.shape[1] == r.world.cfg.n_in
    assert np.array_equal(r.world.s.Win[a][:, :n_in0], src.s.Win[a][:, :n_in0])
    assert not r.world.s.Win[a][:, n_in0:].any() and not r.world.s.Wout[a][:, n_out0:].any()
    assert (r.world.s.bout[a, ACT_CARRY] == -1.0).all() and "bout[ACT_CARRY]=-1" in r.meta["genome_edits"]
    r.loop(400, quiet=True); r.close()
    assert rel_residual(r.world) < 1e-9

def test_branch_ext_is_idempotent(tmp_path):
    _main(tmp_path, "ext")
    r = Runner.branch(tmp_path / "main", tmp_path / "again", 400, ["preset=ext"], False, False)
    assert (r.world.cfg.n_spare_in, r.world.cfg.n_spare_out) == (26, 7) and r.meta["genome_edits"] == []
    r.close()

def test_ext_branch_resumes_exactly(tmp_path):
    _main(tmp_path)
    r = Runner.branch(tmp_path / "main", tmp_path / "ext", 400, ["preset=ext"], False, False)
    r.loop(400, quiet=True); r.close()
    again = Runner.resume(tmp_path / "ext")
    assert again.world.state_hash() == r.world.state_hash(); again.close()

def test_replay_record_with_ext_override(tmp_path):
    _main(tmp_path)
    p = next((tmp_path / "main" / "checkpoints").glob("*400*.npz"))
    rep = replay.record(p, steps=50, overrides=["preset=ext"])
    assert rep is not None
```
(체크포인트 파일 이름 꼴은 `ckpt.list_checkpoints`를 따라 고쳐 쓴다.)

- [ ] **Step 2: 실패 확인** — `set_config`의 "n_spare_in은 실행 중에 바꿀 수 없다" 오류.
- [ ] **Step 3: 구현** — 위 Interfaces대로. `set_config`의 크기 금지 목록은 그대로 둔다(크기는 `load`만 올린다).
- [ ] **Step 4: 통과 확인** — `tests/test_harsh.py`, `tests/test_stages.py`, `tests/test_replay.py` 통과(옛 갈라 켜기 시험 `test_branch_from_stage1_into_later_stages`, `test_branch_turning_conflict_on_calms_attack_bias` 포함).
- [ ] **Step 5: 커밋** — `혹독한 세계 1부-2: 갈라 켜기·다시 보기가 설정을 먼저 만들어 확장 크기로 읽는다`

### Task 3: ① 겨울

**Files:**
- Modify: `kkumteul/config.py` (필드·검사), `kkumteul/world.py:91-136` (`step`, 새 `season()`), `kkumteul/kernels.py` (`spawn_food`, `move`), `kkumteul/stages.py` (`niche_world`, `after_move`)
- Test: `tests/test_harsh.py`

**Interfaces:**
- 설정(임시 결정, 설계 ①): `season_period: int = 4000`, `winter_len: int = 1200`, `winter_mult: float = 0.3`, `winter_basal_mult: float = 2.0`. 검사: `1 <= winter_len < season_period`, `0 <= winter_mult <= 1`, `winter_basal_mult >= 1`.
- Produces: `World.season(t: int) -> tuple[float, float, float, float]` = `(grass_mult, basal_mult, sin, cos)`. 해 위상 `ph = (t % P) / P`, `sin/cos = sin/cos(2π·ph)`. **겨울은 해의 끝** `t % P >= P - winter_len`(갈라 켜는 스텝이 보통 P의 배수라 봄에서 시작하게). 겨울: `(winter_mult, winter_basal_mult)`, 나머지: `((P - winter_mult·W) / (P - W), 1.0)` — 해 평균 풀 배수 1. 꺼져 있으면 `(1.0, 1.0, 0.0, 0.0)`.
- 커널 인자(기본 1.0): `spawn_food(step, s, fp, ip, mult=1.0)` — `budget[0] += P_INFLOW * mult`. `niche_world(step, s, fs, isp, mult=1.0)` — `share = F_INFLOW * mult * F_FERT_FRAC * frac`. `move(s, fp, ip, basal_mult=1.0)` — 기초 대사와 `L_BASAL`에 곱함. `after_move(s, fs, isp, basal_mult=1.0)` — 잠 환급 `back`에 곱함.
- 감각: `season_enabled`이면 `World.step`이 감각 커널들 뒤·생각 앞에 살아 있는 개체의 `obs[:, ext_in0 + EXT_SEASON]`, `+1`에 sin, cos를 쓴다(작은 순차 커널 `kernels.write_ext2(alive, obs, col, a, b)`).

- [ ] **Step 1: 실패하는 시험 쓰기**

```python
def test_season_schedule_keeps_yearly_mean():
    w = World(small("ext", season_enabled=True))
    P, W = w.cfg.season_period, w.cfg.winter_len
    g = [w.season(t)[0] for t in range(P)]
    assert abs(np.mean(g) - 1.0) < 1e-12
    assert w.season(0)[0] > 1.0 and w.season(P - 1) == (pytest.approx(0.3), 2.0, *w.season(P - 1)[2:])

def test_winter_cuts_grass_and_raises_basal():
    on = World(small("ext", season_enabled=True, season_period=400, winter_len=200))
    off = World(small("ext"))
    for w in (on, off): w.run(400)
    assert on.s.ledger[k.L_BASAL] > off.s.ledger[k.L_BASAL]
    assert rel_residual(on) < 1e-9

def test_season_input_is_written():
    w = World(small("ext", season_enabled=True)); w.run(5)
    a = w.s.alive; c = w.cfg.ext_in0 + EXT_SEASON
    _, _, sn, cs = w.season(w.step_no - 1)
    assert np.allclose(w.s.obs[a, c], sn) and np.allclose(w.s.obs[a, c + 1], cs)

def test_winter_with_niche_conserves_energy():
    w = World(small("stage4", "ext", season_enabled=True, season_period=400, winter_len=200))
    w.run(1200); w.check_sane()
    assert rel_residual(w) < 1e-9
```

- [ ] **Step 2: 실패 확인.**
- [ ] **Step 3: 구현** — `World.step`에서 `season(t)`를 한 번 부르고 인자로 넘긴다(`fp`에 쓰지 않는다: 각본이 `set_config`를 자주 부른다).
- [ ] **Step 4: 통과 확인** — `tests/test_harsh.py`, 박아 둔 해시 셋, 전체 시험.
- [ ] **Step 5: 커밋** — `혹독한 세계 1부-3: 겨울(풀 0.3배·기초 대사 2배, 해 평균 1), 해 위상 감각`

### Task 4: ② 부패

**Files:**
- Modify: `kkumteul/config.py`, `kkumteul/kernels.py` (새 `spoil_food`, 새 `_drop_food_slot`, `C_SPOILED = 21`), `kkumteul/world.py` (`step`: `spawn_food` 바로 뒤)
- Modify: `kkumteul/replay.py` (사라짐 까닭에 "상함" — 지금 까닭 표를 따라)
- Test: `tests/test_harsh.py`

**Interfaces:**
- 설정(임시 결정, 설계 ②): `spoil_after: int = 400`, `spoil_rate: float = 0.00115`, `spoil_min_frac: float = 0.05`. 검사: `spoil_after >= 0`, `0 <= spoil_rate <= 1`, `0 <= spoil_min_frac < 1`.
- Produces: `kernels._drop_food_slot(s, j) -> None` — `f_alive[j]=False`, `C_NFOOD -= 1`. 에너지 장부는 부르는 쪽이 쓴다. (2부 ③에서 먹힘·상한 교체·운석·약탈이 이 함수를 거치게 바꾸고 손을 푼다. 이 과제에서는 부패만 쓴다.)
- Produces: `kernels.spoil_food(step, s, fp, ip) -> None` (순차) — 살아 있는 먹이 가운데 `step - f_born >= spoil_after`인 것은 `lost = f_energy * spoil_rate`를 빼 `L_LOST`에. 남은 것이 `spoil_min_frac * food_energy` 미만이면 남은 것도 `L_LOST`에 넣고 `_drop_food_slot`, `C_SPOILED += 1`. 사체(`KIND_CORPSE`)도 상한다. `N_COUNTERS`는 24 그대로(21번이 비어 있다).

- [ ] **Step 1: 실패하는 시험 쓰기**

```python
def test_food_spoils_after_delay_with_half_life():
    w = World(small("ext", spoil_enabled=True, food_inflow=0.0, init_agents=20))
    j = int(np.flatnonzero(~w.s.f_alive)[0]); k._place_food(w.s, 10.0, 10.0, 1.0, k.KIND_A, 0)
    w.s.alive[:] = False                         # 먹는 개체 없이 먹이만 본다
    w.run(400); assert w.s.f_energy[j] == pytest.approx(1.0)
    w.run(600); assert w.s.f_energy[j] == pytest.approx((1 - 0.00115) ** 600, rel=1e-9)
    w.run(2500); assert not w.s.f_alive[j] and w.s.counters[k.C_SPOILED] >= 1
    assert rel_residual(w) < 1e-9 or w.energy_balance()["spawn"] == 0
```
(개체를 모두 끄는 방식이 `min_agents` 재주입과 부딪히면 `min_agents=0`을 준다. 장부 검사는 `energy_balance()`의 `residual`이 0에 가까운지로 바꿔도 된다.)

```python
def test_spoil_world_runs_and_conserves_energy():
    w = World(small("stage4", "ext", spoil_enabled=True)); w.run(1500); w.check_sane()
    assert w.summary()["pop"] > 20 and rel_residual(w) < 1e-9

def test_spoil_with_cap_replace_conserves_energy():
    w = World(small("ext", spoil_enabled=True, food_cap_replace=True)); w.run(1500)
    assert rel_residual(w) < 1e-9
```

- [ ] **Step 2: 실패 확인.**
- [ ] **Step 3: 구현.** `replay`의 먹이 사라짐 까닭에 `C_SPOILED` 증가분을 "상함"으로(지금 까닭 표 구조를 따른다). `tests/test_replay.py`의 `unknown ≤ 2%`를 지킨다.
- [ ] **Step 4: 통과 확인** — 전체 시험.
- [ ] **Step 5: 커밋** — `혹독한 세계 1부-4: 부패(400스텝 뒤 반감기 약 600), 먹이 칸 없애기 함수`

### Task 5: 0번 숫자 재기 (실험 — 세션이 직접, 서브에이전트 아님)

**Files:**
- Modify: `wayfinder/stage2-4/results.md` (끝에 "혹독한 세계 0번: 겨울·부패" 절)

- [ ] **Step 1: 예측을 결과 문서에 먼저 적는다** — 겨울 감소폭(설계 목표: 겨울 개체 수 감소 절반 안쪽), 부패로 사라지는 먹이 비율.
- [ ] **Step 2: 갈래 셋을 world4 최신 체크포인트에서** `preset=ext`(대조), `preset=ext season_enabled=1`, `preset=ext spoil_enabled=1`, 각 약 1만 스텝(겨울 둘 이상), 2스레드·1.5GB·한 번에 하나, `memcap`. 경로는 원래 폴더의 `runs/`, 코드는 커밋한 코드의 사본(`runs/` 아래).
- [ ] **Step 3: 읽기** — 개체 수 시계열(겨울 저점/여름 고점), `reseeds`, 죽음 원인, `C_SPOILED`. 목표에서 벗어나면 `winter_mult`·`winter_basal_mult`를 고쳐 다시(지구 참고 조사의 겨울 숫자와 견줘 정한다).
- [ ] **Step 4: 커밋** — 결과 문서와 바꾼 기본값.
