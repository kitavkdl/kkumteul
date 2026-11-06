# 맛 기억과 물려주기 (구현 계획)

둘째 판(2026-11-06 02:1x ET, 서버 28). 첫 판(`76911c5`)을 맥락 없는 검토 두 갈래(규칙 축·명세 축, sonnet)가 읽었고, 지적을 코드로 확인해 반영했다 — 판단 표는 이 문서 끝 "검토 반영".

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 개체마다 "보이는 모양 둘 × 좋았나 나빴나" 맛 기억을 두고, 뇌가 그것을 입력 10칸(기억으로 물든 시야 8 + 기억 값 2)으로 읽고, 자연 출생 때 자손에게 물려주는 장치를 기본 꺼짐으로 넣은 뒤, 판정에 쓸 손 개체와 측정 도구를 만든다.

**Architecture:** 기억 `taste_m`(개체당 float32 둘)은 저장 필드이고 기본값 0이라 장치를 끈 세계의 해시는 그대로다. 갱신과 잊기는 순차 커널 `eat` 안에서, 맛 블록 쓰기는 병렬 감각 커널 `_sense_par` 끝(시야 잡음 앞)에서 배열을 직접 받아 한다. 맛 블록은 확장 블록 바로 뒤(`taste_in0 = ext_in0 + N_EXT_IN`)이고, 커널은 `ip[Q_TASTE_IN0]`(꺼지면 −1)로 켜짐과 자리를 함께 안다. 물려주기는 `_child`에서 `kind == BIRTH_NATURAL`일 때만.

**Tech Stack:** Python 3, numpy, numba(`njit`), pytest.

**Spec:** `docs/superpowers/specs/2026-11-06-taste-memory-design.md`(둘째 판). 검토 기록 `docs/superpowers/specs/2026-11-06-taste-memory-review.md`.

## 설계가 맡기거나 적지 않아 이 계획이 정한 것
- **측정용 계수기 셋**(저장하지 않음, `_reset_body`가 0으로): `n_meal`(땅에서 먹은 알 수, 사체·고기 포함 — 끼니는 끼니다), `n_poison`(그 가운데 독), `n_poison10`(처음 `FIRST_MEALS = 10`끼 가운데 독. `FIRST_MEALS`는 `config.py` 상수). 설계는 `n_meal`, `n_poison` 둘이지만, 한 스텝에 여러 알을 먹을 수 있어(`eat`은 배가 찰 때까지 먹는다) 바깥에서 "10끼째"를 정확히 끊을 수 없다. 그래서 커널이 처음 10끼 안의 독을 따로 센다. 계수기는 장치가 꺼져 있어도 센다(저장·해시 밖이라 세계를 바꾸지 않는다).
- 설정 칸: 실수 `P_TASTE_RATE`, `P_TASTE_DECAY`, `P_TASTE_INHERIT`를 `N_FP` 앞 맨 끝에, 정수 `Q_TASTE_IN0`을 `N_IP` 앞 맨 끝에 덧붙인다(`range(58)`, `range(24)`). `Q_TASTE_IN0 = cfg.taste_in0 if cfg.taste_enabled else -1`.
- 맛 블록 안 자리 상수(`config.py`): `N_TASTE_IN = 10`, `TASTE_VIS = 0`(물든 시야 8칸), `TASTE_VAL = 8`(기억 값 2칸).
- `validate`: `taste_enabled`이면 `n_spare_in ≥ N_STAGE_IN + N_EXT_IN + N_TASTE_IN`(36)과 `vision_sectors == 8`을 요구한다. 크기 검사는 이 장치만의 것이고 `needs_ext`에 넣지 않는다(맛 장치만 켠 세계가 혹독한 세계 장치 검사에 걸리지 않게). `taste_rate`, `taste_decay`, `taste_inherit`는 장치가 꺼져 있어도 0~1을 검사한다.
- 묶음: `taste_size = dict(n_spare_in=N_STAGE_IN + N_EXT_IN + N_TASTE_IN, n_spare_out=N_STAGE_OUT + N_EXT_OUT)`, `taste = {**taste_size, "taste_enabled": True}`. 둘 다 `ext_size`가 참이라 계수기·예산 배열이 확장 길이다(대조와 장치가 같다).
- 감각 막기: 채널 셋 `taste`(맛 블록 10칸), `taste_vis`(8칸), `taste_val`(2칸), 대조 규칙은 셋 다 `["taste_enabled=false"]`. `vision` 채널은 세계가 맛 블록 크기(`n_in ≥ taste_in0 + N_TASTE_IN`)면 물든 시야 8칸도 지운다(장치가 꺼져 있어 그 칸이 0이어도 같은 칸을 지운다 — 칸 목록이 설정 켬/끔에 따라 바뀌지 않게). 크기가 모자란 세계에서 맛 채널을 고르면 `ValueError`.
- 측정 도구는 새 스크립트 `wayfinder/stage2-4/taste_judge.py` 하나에 하위 명령 둘(`flip-diff`: 판정 가, `first-meals`: 판정 나). 순수 계산(구간 값, 짝지은 차이, 부트스트랩, 새끼 가르기)은 함수로 떼어 시험한다. 손 개체는 `wayfinder/stage2-4/hand_learner.py`를 넓힌다.
- 판정 나에서 재주입 새끼를 가르는 법: 그 스텝에 `C_RESEED`가 늘었으면 그 스텝에 생긴 새끼는 모두 뺀다(재주입은 무리가 `min_agents` 아래일 때만 일어나 드물다). 뺀 수를 출력에 적는다.
- 판정 나에서 죽은 새끼의 끼니를 잃지 않는 법: 한 스텝은 먹기 → 죽음 → 출생 순이고, 죽은 칸은 같은 스텝의 출생이 다시 써서 계수기가 0이 된다. 그래서 도구는 `kernels.eat`를 감싸(`knockout`이 `sense`를 감싸는 것과 같은 꼴, 블록을 나가면 되돌림) **먹기 직후** 산 새끼의 `n_meal`·`n_poison10`을 읽는다. 먹기 뒤에는 그 스텝에 더 먹는 일이 없으므로 이 값이 그 스텝의 끝값이다.
- 판정 나의 "마지막 실제 바뀜"은 실행 시작 전까지 거슬러 센다: `last_flip_at_or_before(seed, t, period)`가 `good_kind`(순수 함수)를 구간 경계마다 거꾸로 견주어 찾는다.
- 부트스트랩: 10,000번 다시 뽑기, `np.random.default_rng(0)`, 2.5·97.5 백분위. 판정 가는 바뀜 단위로, 판정 나는 **(씨앗, 태어난 바뀜 구간) 묶음 단위**로 다시 뽑는다(같은 씨앗·같은 구간의 새끼는 혈통과 바뀜을 함께 겪어 서로 독립이 아니다). 뽑기·백분위는 도우미 하나(`_resample`)를 둘이 쓴다.
- 판정 가의 두 갈래는 **같은 읽기 길**로 읽는다: 켬도 `overrides=["taste_enabled=true"]`를 준다(빈 덮어쓰기는 `ckpt.load(path)`로 가서 끔 갈래와 길이 다르다). 이 덕에 ① 대조 갈래(장치 끈 채 진화)에도 같은 명령으로 켬/끔을 견줄 수 있다(설계 "①의 끝 체크포인트에 같은 켬/끔 → 차이 0 ± 0.02").
- 감각 막기의 음성 대조(① 끝에서 `taste` 막기)는 ①을 맛 장치 켠 규칙으로 돌려야 뜻이 있다(꺼진 규칙에서는 맛 블록이 이미 0). 그래서 `knockout split`에 체크포인트 규칙 덮어쓰기 `--set key=value`(여러 번)를 더한다.

## Global Constraints
- 장치는 기본 꺼짐. `tests/test_stages.py`의 `PINNED_HASH_300` 세 값과 `tests/test_harsh.py`의 `PINNED_EXT_HASH_300` 세 값이 그대로여야 한다. 과제 1이 `preset=taste_size` 세계의 해시 `PINNED_TASTE_SIZE_HASH_300`을 박고, 그 뒤 과제는 이 값도 지킨다. **과제마다 마지막 시험 단계에서 세 묶음을 따로 돌린다.**
- 에너지가 생기는 길은 먹기와 명시적 주입뿐. 맛 장치는 에너지를 바꾸지 않는다. 장부 시험은 `|residual|/(spawn+inject) < 1e-9`.
- `prange` 안에서 상태 묶음(`s.taste_m`)을 거쳐 쓰지 않는다. `_sense_par`에는 `taste_m`을 배열 인자로 넘긴다. 바깥에서 부르는 `sense(step, s, fp, ip)`의 시그니처는 그대로다.
- 난수를 새로 쓰지 않는다. `rng.py`와 난수 호출 순서를 바꾸지 않는다.
- numba는 범위를 검사하지 않는다. 맛 블록 쓰기는 `Q_TASTE_IN0 ≥ 0`이고 `Q_TASTE_IN0 + N_TASTE_IN ≤ n_in`일 때만.
- 설정값은 모두 `config.py`에 둔다. 새 숫자 주석에 "임시 결정"과 근거를 말로 적는다(`taste_rate = 0.5`: 한 번이면 반, 두 번이면 3/4 / `taste_decay = 0.002`: 반감기 약 350스텝, 독이 실제로 바뀌는 평균 간격 약 3,000의 8분의 1 / `taste_inherit = 0.0`: 실험이 0과 1을 견준다).
- 공개 레포: 코드·주석·시험에 로컬 경로, 다른 프로젝트, "과제 N" 같은 작업 과정 표현을 쓰지 않는다.
- 시험 실행(이하 `<시험>`): 작업 폴더에서 `NUMBA_NUM_THREADS=2 memcap --max 2G -E NUMBA_NUM_THREADS -- nice -n 10 <원래 폴더>/.venv/bin/python -m pytest -q -p no:cacheprovider <대상>`. 전체 약 3분. 서브에이전트는 시험만 돌리고 시뮬레이션 실험은 돌리지 않는다. 시험 임시 폴더는 pytest `tmp_path`만.
- 커밋 메시지는 한국어, 고친 파일만 `git add`(`.crew`, `.superpowers` 제외). 커밋 뒤 작업 가지로 push.

## Review Focus
1. **옛 체크포인트(입력 13칸, `taste_m` 없음)를 `preset=taste`로 갈라 읽으면** `taste_m`이 0, 늘어난 입력 가중치가 0이고, 첫 스텝부터 맛 블록에 값이 써지며 넘치지 않는다 — 과제 3 `test_branch_old_checkpoint_into_taste`.
2. **각본이 실행 중에 `taste_enabled`를 끄면** 기억이 0으로 지워지고, 그 뒤 맛 블록 10칸이 0이며, 다시 켜면 0에서 다시 배운다 — 과제 1 `test_set_config_off_clears_taste` + 과제 3 `test_taste_block_zero_when_off`.
3. **배가 찬 개체(`energy ≥ e_max`)도 잊는다**(먹지 않는 스텝에 잊기를 건너뛰면 배부른 개체의 기억이 굳는다) — 과제 2 `test_full_agent_still_forgets`.
4. **시야 잡음이 커서 잡음 뒤 시야 칸 부호가 뒤집혀도** 물든 시야는 잡음 전 부호(모양)의 기억을 읽는다 — 과제 3 `test_taste_vis_uses_sign_before_noise`.
5. **한 스텝에 여러 알을 먹어 10끼를 넘기는 개체**의 `n_poison10`은 처음 10알 안의 독만 센다 — 과제 2 `test_first_ten_meals_counted_exactly`.
6. **독을 먹은 스텝에 죽은 새끼**도 판정 나의 끼니에 들어간다 — 과제 7 `test_eat_tap_sees_meals_before_death`.

---

### Task 1: 설정·묶음·저장 필드·끄면 지우기

**Files:**
- Modify: `kkumteul/config.py` (설정 필드, 상수, `taste_in0`, `validate`, `preset`)
- Modify: `kkumteul/kernels.py` (`AGENT_FIELDS`, `HASH_DEFAULTS`, `P_*`/`Q_*`, `pack_params`)
- Modify: `kkumteul/world.py` (`set_config`)
- Modify: `kkumteul/measure.py` (`inject_agent`)
- Create: `tests/test_taste.py`

**Interfaces:**
- Produces: `Config.taste_enabled: bool = False`, `taste_rate: float = 0.5`, `taste_decay: float = 0.002`, `taste_inherit: float = 0.0`; 상수 `N_TASTE_IN = 10`, `TASTE_VIS = 0`, `TASTE_VAL = 8`; 속성 `Config.taste_in0 -> int`(= `ext_in0 + N_EXT_IN`); 묶음 `taste`, `taste_size`. 상태 필드 `taste_m`(float32, 뒤 차원 `"2"`, 저장), `n_meal`, `n_poison`, `n_poison10`(int32, 저장 안 함). `kernels.FIRST_MEALS = 10`. 설정 칸 `P_TASTE_RATE`, `P_TASTE_DECAY`, `P_TASTE_INHERIT`, `Q_TASTE_IN0`. 시험 도우미 `tests/test_taste.py`의 `taste_world(*presets, **kw) -> World`(= `World(small("stage2", "kinds", *presets, **kw))`, `small`은 `test_stages`에서 가져온다).

- [ ] **Step 0: 기준 해시를 지금 코드에서 뽑는다.** 아무것도 고치기 전 HEAD에서 `World(small(n_spare_in=36, n_spare_out=7))`와 `World(small("stage2", "kinds", n_spare_in=36, n_spare_out=7))`를 300스텝 돌린 `state_hash()` 둘을 출력해 둔다. 이것이 `PINNED_TASTE_SIZE_HASH_300`의 두 값(키 `()`, `("stage2", "kinds")`)이다(`test_stages.py`의 박은 해시와 같은 관례: 장치를 넣기 전 코드가 낸 값).

- [ ] **Step 1: 실패하는 시험을 쓴다** (`tests/test_taste.py`)

가져오기: `from test_stages import rel_residual, small`, `from kkumteul import kernels as k, measure, checkpoint as ckpt`, `from kkumteul.config import ACT_TURN, FIRST_MEALS, N_EXT_IN, N_STAGE_IN, N_TASTE_IN, TASTE_VAL, TASTE_VIS, preset`, `from kkumteul.world import World`(앞에 `tests/test_harsh.py`처럼 `sys.path` 두 줄).

```python
def test_taste_config_and_presets():
    c = small("taste")
    assert c.taste_enabled and c.n_spare_in == N_STAGE_IN + N_EXT_IN + N_TASTE_IN
    assert c.taste_in0 == c.ext_in0 + N_EXT_IN and c.n_in >= c.taste_in0 + N_TASTE_IN
    z = small("taste_size")
    assert not z.taste_enabled and z.n_spare_in == c.n_spare_in and z.n_spare_out == c.n_spare_out
    with pytest.raises(ValueError, match="taste"):
        preset("nope")                      # 오류 메시지가 taste, taste_size를 알려 준다
    with pytest.raises(ValueError):
        small("ext", taste_enabled=True)    # 크기가 모자람
    with pytest.raises(ValueError):
        small("taste", vision_sectors=6)
    for name in ("taste_rate", "taste_decay", "taste_inherit"):
        with pytest.raises(ValueError):
            small(**{name: 1.5})
        with pytest.raises(ValueError):
            small(**{name: -0.1})

@pytest.mark.parametrize("presets", list(PINNED_TASTE_SIZE_HASH_300))
def test_taste_size_world_hash_pinned(presets):
    # taste_size 묶음 = 장치 끈 같은 크기 세계(Step 0에서 장치 넣기 전 코드로 뽑은 값)
    w = World(small(*presets, "taste_size"))
    w.run(300)
    assert w.state_hash() == PINNED_TASTE_SIZE_HASH_300[presets]

def test_taste_m_saved_and_old_checkpoint_reads_zero(tmp_path):
    # 저장 → 읽기 → 같은 해시. 0이 아닌 기억은 해시에 들어간다. taste_m 없는 배열 묶음을 읽으면 0
    # (장치 끈 크기 세계에서 한다: 켠 세계는 먹으면서 기억이 0이 아니게 되어 h0가 기억 0 상태가 아니다)
    w = taste_world("taste_size"); w.run(50)
    h0 = w.state_hash()
    w.s.taste_m[w.s.alive] = 0.25
    assert w.state_hash() != h0
    arrays = {kk: v.copy() for kk, v in w.state_arrays().items()}
    w2 = World(w.cfg, populate=False); w2.step_no = w.step_no
    assert w2.load_arrays(arrays) and w2.state_hash() == w.state_hash()
    del arrays["taste_m"]
    w3 = World(w.cfg, populate=False); w3.step_no = w.step_no
    assert w3.load_arrays(arrays)           # 빠진 것으로 치지 않는다(HASH_DEFAULTS)
    assert (w3.s.taste_m == 0).all() and w3.state_hash() == h0

def test_set_config_off_clears_taste():
    w = taste_world("taste"); w.s.taste_m[:] = 0.5
    w.set_config(w.cfg.with_overrides(["taste_enabled=false"]))
    assert (w.s.taste_m == 0).all() and w.ip[k.Q_TASTE_IN0] == -1

def test_inject_agent_starts_with_blank_taste():
    w = taste_world("taste"); w.s.taste_m[:] = 0.5; w.s.n_meal[:] = 7
    side = measure.side_from_world(w)
    j_id = measure.inject_agent(w, side["genomes"], 0, 1, 10.0, 10.0, 0.0, 0)
    j = int(np.flatnonzero(w.s.ids == j_id)[0])
    assert (w.s.taste_m[j] == 0).all() and w.s.n_meal[j] == 0 and w.s.n_poison10[j] == 0
```

`test_taste_m_saved_and_old_checkpoint_reads_zero`에서 `load_arrays` 앞에 필요한 것이 다르면 `tests/test_stages.py`의 체크포인트 왕복 시험을 따른다. `PINNED_TASTE_SIZE_HASH_300`은 Step 0에서 뽑은 두 값이다.

- [ ] **Step 2: 시험이 실패하는지 본다**

Run: `<시험> tests/test_taste.py`
Expected: FAIL (`preset` 이름 없음, `taste_enabled` 없음 등)

- [ ] **Step 3: 구현한다**
  - `config.py`: 위 필드 넷(주석은 Global Constraints의 근거), 상수 셋, `taste_in0` 속성, `validate` 검사(설계가 맡긴 것 절), `preset` 표에 `taste_size`·`taste` 더하고 오류 메시지 이름 목록에 `taste, taste_size`.
  - `config.py`에 `FIRST_MEALS = 10`(측정용, 판정 나의 "첫 10끼").
  - `kernels.py`: `AGENT_FIELDS`의 저장 필드 끝(`a_held` 뒤)에 `("taste_m", np.float32, ("2",), True)`, 저장 안 하는 필드 끝에 `n_meal`·`n_poison`·`n_poison10`(int32). `HASH_DEFAULTS`에 `"taste_m": 0`. 설정 칸 넷과 `pack_params` 채우기.
  - `world.py` `set_config`: `old.taste_enabled and not cfg.taste_enabled`이면 `self.s.taste_m[:] = 0.0`.
  - `measure.inject_agent`: 지우는 이름 목록에 `"taste_m"`, 그리고 `n_meal`·`n_poison`·`n_poison10`을 0으로.
  - `PINNED_TASTE_SIZE_HASH_300`에 Step 0 값을 박는다(주석: 맛 장치를 넣기 전 코드가 낸 값, 장치를 끈 세계의 계산을 일부러 바꿀 때만 고친다).

- [ ] **Step 4: 시험이 통과하는지 본다**

Run: `<시험> tests/test_taste.py`, `<시험> tests/test_stages.py -k pinned`, `<시험> tests/test_harsh.py -k pinned`
Expected: 모두 PASS

- [ ] **Step 5: 커밋**

```bash
git add kkumteul/config.py kkumteul/kernels.py kkumteul/world.py kkumteul/measure.py tests/test_taste.py
git commit -m "맛 기억: 설정·묶음·저장 필드(기본 꺼짐, 해시 그대로)"
```

---

### Task 2: 먹은 순간 갱신·잊기·계수기

**Files:**
- Modify: `kkumteul/kernels.py` (`eat`, `_reset_body`)
- Test: `tests/test_taste.py`

**Interfaces:**
- Consumes: Task 1의 필드·설정 칸.
- Produces: `eat`가 `Q_TASTE_IN0 ≥ 0`일 때 기억을 갱신하고 잊는다. 계수기 셋은 늘 센다. 시험 도우미 `feed(w, i, kind, count=1) -> None`: 개체 i의 에너지를 2.0으로 다시 맞추고(`e_max` 8에 닿아 먹지 않는 일이 없게. `feed`를 쓰는 시험은 장부를 단언하지 않는다), 개체 i의 자리에 그 종류 먹이 `count`알을 `_place`(`test_harsh`의 것과 같은 꼴, 장부에 `L_SPAWN` 적음)로 놓고, 먹이 격자를 다시 짓고(`k.build_grid(s.fx, s.fy, s.f_alive, ...)`), `k.eat(w.step_no, w.s, w.fp, w.ip)`를 한 번 부른다. 독인 종류는 `1 - k.good_kind(cfg.seed, w.step_no, cfg.kind_flip_period)`.

시험 세계: `taste_world("taste", init_agents=1, min_agents=0, food_inflow=0.0)`, 개체 하나의 에너지를 `e_max`보다 충분히 낮게(예: 2.0) 맞춘다.

- [ ] **Step 1: 실패하는 시험을 쓴다**

```python
def test_eat_updates_memory_by_seen_shape():
    # 잊기는 eat 첫머리라 eat를 부를 때마다 먼저 (1 - 0.002)배가 된다. 칸 = 1 if 종류 B else 0
    w = ...; good = k.good_kind(w.cfg.seed, w.step_no, w.cfg.kind_flip_period); bad = 1 - good
    slot = lambda kind: 1 if kind == k.KIND_B else 0
    feed(w, 0, good)
    assert w.s.taste_m[0, slot(good)] == pytest.approx(0.5)
    feed(w, 0, bad)
    assert w.s.taste_m[0, slot(bad)] == pytest.approx(-0.5)
    assert w.s.taste_m[0, slot(good)] == pytest.approx(0.5 * 0.998)          # 다른 칸은 잊기만
    feed(w, 0, bad)
    assert w.s.taste_m[0, slot(bad)] == pytest.approx(-0.7495, abs=1e-6)     # −0.499 + 0.5 × (−1 + 0.499)
    w.s.taste_m[0] = 0.0
    feed(w, 0, k.KIND_CORPSE)
    assert w.s.taste_m[0, 0] == pytest.approx(0.5) and w.s.taste_m[0, 1] == 0.0   # 사체는 + 모양 칸

def test_full_agent_still_forgets():
    w = ...; w.s.taste_m[0] = (0.8, -0.6); w.s.energy[0] = w.cfg.e_max
    k.eat(...)                              # 먹지는 않는다
    assert w.s.taste_m[0] == pytest.approx((0.8 * 0.998, -0.6 * 0.998))

def test_forgetting_per_step():
    w = ...; w.s.taste_m[0] = (0.8, -0.6); w.run(100)          # 먹이 없는 세계(food_inflow=0)
    assert w.s.taste_m[0] == pytest.approx((0.8 * 0.998 ** 100, -0.6 * 0.998 ** 100), rel=1e-5)

def test_taste_off_leaves_memory_and_counts_meals():
    w = taste_world("taste_size", init_agents=1, ...)
    feed(w, 0, poison_kind)
    assert (w.s.taste_m == 0).all() and w.s.n_meal[0] == 1 and w.s.n_poison[0] == 1 and w.s.n_poison10[0] == 1

def test_first_ten_meals_counted_exactly():
    for _ in range(9):
        feed(w, 0, good)
    feed(w, 0, bad, count=3)            # 한 번의 eat에 독 셋(에너지는 2 − 3으로 음수가 되지만 eat만 부르므로 죽지 않는다)
    assert (w.s.n_meal[0], w.s.n_poison[0], w.s.n_poison10[0]) == (12, 3, 1)

def test_taste_eat_ledger_closes():
    w = taste_world("taste"); w.run(1500)
    assert rel_residual(w) < 1e-9
```

`_place`가 테스트 파일에 없으면 `test_harsh._place`를 가져다 쓴다(같은 꼴).

- [ ] **Step 2: 시험이 실패하는지 본다** — Run: `<시험> tests/test_taste.py`, Expected: 새 시험 FAIL

- [ ] **Step 3: `eat(step, s, fp, ip)`를 고친다**
  - 개체 반복 첫머리: `s.ate[i] = 0.0` 뒤, 산 개체이고 `ip[Q_TASTE_IN0] >= 0`이면 두 칸 모두 `× (1 - fp[P_TASTE_DECAY])` — **배부름 검사보다 앞**. 그다음 `energy >= e_max`면 건너뛴다(지금 한 줄인 조건을 둘로 나눈다).
  - 한 알마다(`_drop_food_slot` 뒤, 독 분기 전): `poison = kinds and not always_edible(f_kind) and f_kind != good`. `if s.n_meal[i] < FIRST_MEALS and poison: s.n_poison10[i] += 1`, `s.n_meal[i] += 1`, `if poison: s.n_poison[i] += 1`. 장치가 켜져 있으면 `c = 1 if f_kind == KIND_B else 0`, `o = -1.0 if poison else 1.0`, `m[c] += rate * (o - m[c])`. 그 뒤 기존 독 분기의 조건을 `if poison:`으로 바꿔 같은 판정을 두 번 쓰지 않는다(에너지 계산은 그대로).
  - `_reset_body`: `taste_m[j, 0..1] = 0`, 계수기 셋 0.

- [ ] **Step 4: 시험이 통과하는지 본다**

Run: `<시험> tests/test_taste.py`, `<시험> tests/test_stages.py -k pinned`, `<시험> tests/test_harsh.py -k pinned`
Expected: 모두 PASS(`PINNED_TASTE_SIZE_HASH_300` 포함)

- [ ] **Step 5: 커밋** — `git add kkumteul/kernels.py tests/test_taste.py`, 메시지 "맛 기억: 먹은 순간 갱신과 잊기, 개체별 끼니 계수기"

---

### Task 3: 감각 맛 블록

**Files:**
- Modify: `kkumteul/kernels.py` (`_sense_par`, `sense`)
- Test: `tests/test_taste.py`

**Interfaces:**
- Consumes: `Q_TASTE_IN0`, `taste_m`, `N_TASTE_IN`, `TASTE_VIS`, `TASTE_VAL`.
- Produces: 감각 뒤 `obs[i, t0 + TASTE_VIS + sec]`(sec 0..7)와 `obs[i, t0 + TASTE_VAL + 0..1]`. `_sense_par`의 새 마지막 인자 `taste_m`(배열).

- [ ] **Step 1: 실패하는 시험을 쓴다**

```python
def test_taste_vis_reads_memory_by_shape():
    # 먹이 하나(종류 A, + 모양)를 개체 왼쪽 앞에 놓고 m = (-0.5, 0.3): 그 시야 칸 v > 0 → 맛 칸 = v * -0.5.
    # 종류 B로 바꾸면 시야 칸 v < 0 → 맛 칸 = |v| * 0.3. 기억 값 두 칸 = m 그대로. 먹이 없는 칸은 0
    # 장치를 빼면 실패하는 꼴: m을 (0, 0)으로 바꾸면 그 칸이 0이 된다

def test_taste_vis_uses_sign_before_noise():
    # vision_noise를 크게(예: 5.0) 켜고 같은 배치: 맛 칸은 잡음 없는 세계에서 계산한 값과 같다
    # (잡음 없는 같은 세계를 따로 만들어 견준다. 시야 칸 자체는 다르다는 것도 단언)

def test_taste_block_zero_when_off():
    # taste_size(장치 끔)에서 m을 손으로 0.5로 넣어도 맛 블록 10칸이 0

def test_branch_old_checkpoint_into_taste(tmp_path):
    # small("stage2", "kinds") 세계를 Runner로 400스텝 돌려 체크포인트를 만들고(test_stages의 체크포인트 시험 꼴),
    # ckpt.load(path, saved.with_overrides(["preset=taste"]))로 읽는다:
    # taste_m 0, Win[:, taste_in0:, :] 0, 한 스텝 뒤 기억 값 칸이 m과 같고 check_sane 통과
```

먹이를 개체 왼쪽 앞에 놓는 자리: 개체 `heading = 0`, 먹이를 `(x + 10, y + 3)` 근처에 두고 시야 칸 값이 0이 아닌 칸 번호를 시험이 먼저 찾아 그 칸으로 단언한다(칸 번호를 박지 않는다). 감각만 부를 때는 `k.build_grid`(개체·먹이) 뒤 `k.sense(w.step_no, w.s, w.fp, w.ip)`.

- [ ] **Step 2: 시험이 실패하는지 본다** — Expected: 새 시험 FAIL

- [ ] **Step 3: 구현한다**
  - `_sense_par(..., f_start, f_items, taste_m)`: 개체마다 `if noise > 0.0` 블록 **앞**에서, `t0 = ip[Q_TASTE_IN0]`이 0 이상이고 `t0 + N_TASTE_IN <= n_in`이면 `sec in range(nsec)`마다 `v = obs[i, sec]`; `v > 0`이면 `v * taste_m[i, 0]`, `v < 0`이면 `-v * taste_m[i, 1]`, 아니면 0을 `obs[i, t0 + TASTE_VIS + sec]`에, 그리고 `obs[i, t0 + TASTE_VAL + c] = taste_m[i, c]`.
  - `sense(step, s, fp, ip)`가 `s.taste_m`을 넘긴다.

- [ ] **Step 4: 시험이 통과하는지 본다** — Run: `<시험> tests/test_taste.py` + 두 pinned 묶음. Expected: PASS

- [ ] **Step 5: 커밋** — "맛 기억: 감각 맛 블록(물든 시야 8칸, 기억 값 2칸)"

---

### Task 4: 물려주기·기록·장부·켠 세계 해시

**Files:**
- Modify: `kkumteul/kernels.py` (`_child`)
- Modify: `kkumteul/world.py` (`summary`)
- Test: `tests/test_taste.py`

**Interfaces:**
- Consumes: Task 1~3.
- Produces: 자연 출생 새끼 `taste_m = taste_inherit × 부모 값`. `summary()` 키 `taste_plus`, `taste_minus`(장치 켬), `taste_on_poison`(장치 켬 + 먹이 종류 켬). 상수 `PINNED_TASTE_HASH_300`(시험 파일).

- [ ] **Step 1: 실패하는 시험을 쓴다**

```python
@pytest.mark.parametrize("inherit", [0.0, 1.0, 0.5])
def test_natural_child_inherits(inherit):
    # 부모 하나만 산 세계, 부모 m = (0.6, -0.4), 번식 조건을 채워(에너지 e_repro 위, 나이 repro_min_age 위)
    # k.process_births 한 번: 새끼 m == inherit * (0.6, -0.4)

def test_reseed_and_passive_children_start_blank():
    # k.reseed(무리를 min_agents 아래로 줄인 세계)와 passive_events(passive=True 세계)로 생긴 새끼는 m == 0

def test_taste_world_ledger_closes_and_memory_moves():
    on = taste_world("taste", taste_inherit=1.0); on.run(1500)
    assert rel_residual(on) < 1e-9
    assert np.abs(on.s.taste_m[on.s.alive]).max() > 0.1
    off = taste_world("taste_size"); off.run(1500)
    assert on.state_hash() != off.state_hash()

def test_taste_summary_keys():
    # on.summary()에 taste_plus == m[:,0] 평균, taste_minus == m[:,1] 평균,
    # taste_on_poison == (독이 A면 m[:,0], B면 m[:,1]) 평균 — 독 = 1 - good_kind(seed, step_no - 1, period)
    # taste_size 세계의 summary에는 세 키가 없다

def test_taste_world_hash_pinned():
    w = taste_world("taste", taste_inherit=1.0); w.run(300)
    assert w.state_hash() == PINNED_TASTE_HASH_300
```

번식·재주입·대조 실행 새끼를 만드는 세계 꼴은 `tests/test_core.py`나 `tests/test_stages.py`에 이미 있는 번식·재주입 시험을 따른다. `PINNED_TASTE_HASH_300`은 Step 3 뒤 출력으로 박는다(주석: 맛 장치를 켠 세계, 장치 계산을 일부러 바꿀 때만 고친다).

- [ ] **Step 2: 시험이 실패하는지 본다**
- [ ] **Step 3: 구현한다**
  - `_child`: `copy_genome` 뒤 `if kind == BIRTH_NATURAL and ip[Q_TASTE_IN0] >= 0:` 두 칸 `= fp[P_TASTE_INHERIT] * s.taste_m[i, c]`(`_reset_body`가 이미 0으로 지웠다).
  - `summary()`: 개체가 있을 때의 묶음 안, `food_kinds_enabled` 키 옆에 세 키. `taste_on_poison`의 독 쪽 칸은 `good_kind(cfg.seed, max(self.step_no - 1, 0), cfg.kind_flip_period)`의 반대(설계 "직전 스텝 기준": `step` 끝에 `step_no = t + 1`이 되므로 방금 돈 스텝은 `step_no − 1`이다. 옆의 기존 키 `good_kind`는 다음 스텝 값이라 다를 수 있다 — 키 주석에 적는다).
- [ ] **Step 4: 시험이 통과하는지 본다** — `<시험> tests/test_taste.py` + 두 pinned 묶음. Expected: PASS
- [ ] **Step 5: 커밋** — "맛 기억: 자연 출생 물려주기, 기록, 켠 세계 해시"

---

### Task 5: 감각 막기 채널

**Files:**
- Modify: `kkumteul/knockout.py` (`CHANNELS`, `OFF_RULES`, `columns`, `split`의 `base_overrides`, 명령줄 `--set`, 모듈 설명)
- Test: `tests/test_knockout.py`(이 파일은 `from kkumteul import knockout as ko`로 가져온다 — 시험의 `columns`, `OFF_RULES`, `knockout`은 `ko.` 접두어로 쓴다)

`split(path, channel, ..., base_overrides: list[str] | None = None)`: 주면 `side["cfg"].with_overrides(base_overrides)`를 체크포인트 규칙으로 쓴다. 명령줄 `split --set key=value`(여러 번)가 이것을 채운다. 쓰임: ① 대조 갈래 끝에서 `--set taste_enabled=true --channel taste`(음성 대조).

**Interfaces:**
- Consumes: `Config.taste_in0`, `N_TASTE_IN`, `TASTE_VIS`, `TASTE_VAL`.
- Produces: `columns(cfg, "taste") == list(range(t0, t0 + 10))`, `"taste_vis"` = `t0..t0+7`, `"taste_val"` = `t0+8, t0+9`; `columns(cfg, "vision")`이 맛 블록 크기 세계에서 `range(2*nsec)` + `taste_vis` 칸.

- [ ] **Step 1: 실패하는 시험을 쓴다**

```python
def test_taste_channels_columns_and_rules():
    c = small("stage2", "kinds", "taste"); t0 = c.taste_in0
    assert columns(c, "taste") == list(range(t0, t0 + 10))
    assert columns(c, "taste_vis") == list(range(t0, t0 + 8))
    assert columns(c, "taste_val") == [t0 + 8, t0 + 9]
    assert columns(c, "vision") == list(range(2 * c.vision_sectors)) + list(range(t0, t0 + 8))
    assert columns(small("stage2"), "vision") == list(range(2 * 8))     # 맛 블록 없는 세계는 그대로
    for ch in ("taste", "taste_vis", "taste_val"):
        assert OFF_RULES[ch] == ["taste_enabled=false"]
        with pytest.raises(ValueError):
            columns(small("stage2"), ch)

def test_taste_knockout_wipes_block():
    # 맛 장치 켠 세계에서 m을 0이 아닌 값으로 두고 knockout(cfg, "taste", tag=0) 안에서 한 스텝:
    # tag 0 개체의 맛 블록 10칸은 0, tag 1 개체는 0이 아닌 칸이 있다. "vision" 막기는 물든 시야 칸도 0

def test_split_base_overrides_reach_rules(monkeypatch):
    # measure.assay_sides를 가짜로 바꿔 받은 a["cfg"]를 붙잡는다: base_overrides=["taste_enabled=true"]를 주면
    # 장치 끈 체크포인트(taste_size)에서도 a["cfg"].taste_enabled가 참
```

- [ ] **Step 2: 실패 확인** — `<시험> tests/test_knockout.py`
- [ ] **Step 3: 구현한다** (설계가 맡긴 것 절의 규칙대로. 명령줄 `--channel` 선택지는 `CHANNELS`를 따르므로 따로 고치지 않는다)
- [ ] **Step 4: 통과 확인** — `<시험> tests/test_knockout.py tests/test_taste.py` + 두 pinned 묶음
- [ ] **Step 5: 커밋** — "감각 막기: 맛 블록 채널 셋, 시야 막기가 물든 시야도 지움"

---

### Task 6: 손 개체(양성 대조)

**Files:**
- Modify: `wayfinder/stage2-4/hand_learner.py`
- Test: `tests/test_taste.py`(손 배선만 단언. 실험은 돌리지 않는다)

**Interfaces:**
- Consumes: `preset=taste`, `Config.taste_in0`, `measure.side_from_world`, `checkpoint.load(path, cfg)`.
- Produces: `make_hand(..., taste_g: float = 0.0, taste_in0: int = -1)` — `taste_g ≠ 0`이면 `Win[:, taste_in0 + TASTE_VIS + s, 0] = taste_g if s >= nsec // 2 else -taste_g`. `taste_g ≠ 0`인데 `taste_in0 < 0`이면 `ValueError`(음수 번호로 엉뚱한 칸에 쓰지 않게 — `grid`·`curve` 하위 명령이 `HAND_VARIANTS=taste`를 받아도 조용히 틀리지 않는다). 바뀜 뒤 값 `after_flips(series, period, seed, win=300) -> list[dict]`: 실제 바뀜(`good`이 바뀐 줄)마다 `{"flip", "first": 바뀜 뒤 win스텝의 독 비율, "rest": 그 뒤 다음 바뀜까지의 독 비율}`. 변형 `VARIANTS["taste"] = dict(eta=0.0, alpha=0.0, g=0.0, turn=1.0, taste_g=0.4)`. `curve(cfg, genomes, n, steps, seed, every=250, lab=True)` — `lab=False`면 LAB 설정도 번식 끄기도 하지 않는다. `curve`의 `series` 줄마다 `placed`(산 들판 먹이 수: `f_alive`, `f_holder < 0`, 종류 0·1)와 `placed_poison`(그 가운데 지금 독인 종류의 비율). 하위 명령 `taste`.

- [ ] **Step 1: 실패하는 시험을 쓴다**

```python
def test_hand_taste_wiring():
    sys.path.insert(0, str(ROOT / "wayfinder" / "stage2-4")); import hand_learner as hl
    w = taste_world("taste"); side = measure.side_from_world(w)
    t0 = w.cfg.taste_in0
    g = hl.make_hand(side["genomes"], 4, taste_in0=t0, **hl.VARIANTS["taste"])
    assert (g["Win"][:, :16, 0] == 0).all()                       # 원래 시야는 0
    assert (g["Win"][:, t0 + 4:t0 + 8, 0] == 0.4).all() and (g["Win"][:, t0:t0 + 4, 0] == -0.4).all()
    assert (g["Wout"][:, ACT_TURN, 0] == 1.0).all() and (g["eta"] == 0).all()
    with pytest.raises(ValueError):
        hl.make_hand(side["genomes"], 4, **hl.VARIANTS["taste"])     # taste_in0 없이

def test_after_flips_uses_real_flips():
    # series: 250스텝 간격 줄, good이 0,0,...,1(스텝 1750에서 바뀜),1,...  — 다시 정해졌지만 안 바뀐 경계(1500)는 바뀜이 아니다
    # 결과는 바뀜 하나, first = 1750~2050 줄들의 Σpoisoned/Σeaten, rest = 그 뒤 다음 바뀜(없으면 끝)까지
```

- [ ] **Step 2: 실패 확인**
- [ ] **Step 3: 구현한다.** 하위 명령 `taste <체크포인트> <스텝> <출력.json>`: 저장 설정에 `preset=taste`를 덧붙인 cfg로 `ckpt.load(path, cfg)` → `side_from_world`로 맛 크기 유전체를 얻는다. 세계 둘(`lab=True`, `lab=False`) × 맛 켬/끔(`taste_enabled=false`, 같은 크기) × 씨앗 `HAND_SEEDS`(기본 `102,100`)로 `curve`를 `HAND_N`(기본 400)마리, `every=100`으로 돌려 `after_flips`의 바뀜별 `first`·`rest`와 그 평균, `placed`·`placed_poison`을 적는다. 판정 0 통과 여부를 한 줄로 낸다: 맛 켬의 `rest` 평균 < 0.35이고 맛 끔의 `rest` 평균이 0.5 ± 0.05. 출력 줄 꼴은 기존 `curve` 하위 명령을 따른다. 모듈 설명에 하위 명령과 예측을 적는다.
- [ ] **Step 4: 통과 확인** — `<시험> tests/test_taste.py`
- [ ] **Step 5: 커밋** — "손 개체: 맛 기억을 읽는 배선과 켬/끔 비교"

---

### Task 7: 판정 도구(가: 바뀜별 켬/끔, 나: 새끼의 첫 10끼)

**Files:**
- Create: `wayfinder/stage2-4/taste_judge.py`
- Test: `tests/test_taste_judge.py`

**Interfaces:**
- Consumes: `replay.record(ckpt, steps, every, from_flip=True, overrides)`(돌려주는 것의 `steps`, `good_kind`, `blind`), `checkpoint.load`, `checkpoint.read_config`, `World.run`, `kernels.good_kind`, `kernels.C_RESEED`, 필드 `n_meal`, `n_poison10`, `birth_step`, `ids`.
- Produces (순수 함수, 시험 대상):
  - `flip_deltas(steps: np.ndarray, good: np.ndarray, blind: np.ndarray, win: int = 300) -> list[dict]` — 실제 바뀜(`good`이 바뀐 장면)마다 `{"flip": 스텝, "early": [flip, flip+win)의 blind 평균, "late": [다음 바뀜−win, 다음 바뀜)의 평균, "delta": late − early}`. 다음 바뀜이 없거나 두 구간이 겹치거나(바뀜 간격 < 2·win) 비면 넣지 않는다.
  - `paired_diff(on: list[dict], off: list[dict]) -> np.ndarray` — 같은 `flip` 스텝끼리 짝지어 `delta_on − delta_off`.
  - `bootstrap_ci(x: np.ndarray, reps: int = 10000, seed: int = 0) -> tuple[float, float, float]` — (평균, 2.5%, 97.5%).
  - `last_flip_at_or_before(seed: int, t: int, period: int, lookback: int = 200) -> int | None` — `t`를 넘지 않는 마지막 실제 바뀜 스텝(구간 경계 `b·period`에서 `good_kind(b) != good_kind(b−1)`인 가장 큰 것). `lookback` 구간 안에 없으면 `None`.
  - `split_children(rows: list[dict], win: int = 300) -> dict` — `rows`는 새끼마다 `{"since_flip": 태어난 스텝 − last_flip_at_or_before(태어난 스텝), "cluster": (씨앗, 그 바뀜 스텝), "meals": min(n_meal, 10), "poison": n_poison10}`. `{"old": [...], "fresh": [...]}`(`since_flip > win`이 old, `0 ≤ since_flip ≤ win`이 fresh, `None`은 old).
  - `pooled_ratio(rows) -> float` — Σpoison / Σmeals(끼니 0인 새끼는 분모·분자 모두 0으로 들어간다).
  - `ratio_diff_ci(a: list[dict], b: list[dict], reps=10000, seed=0) -> tuple[float, float, float]` — 두 무리를 각자 **`cluster` 묶음째** 다시 뽑아 `pooled_ratio(a) − pooled_ratio(b)`의 (값, 2.5%, 97.5%).
  - `eat_tap(world, on_eat: Callable[[State], None])` — 컨텍스트 관리자. 블록 안에서 `kernels.eat`를 감싸 원래 `eat` 직후 `on_eat(s)`를 부르고, 나가면 되돌린다(`knockout.knockout`과 같은 꼴, `World.step`이 `k.eat`를 모듈에서 찾으므로 감싸기가 먹힌다).
- 명령줄:
  - `flip-diff <체크포인트> [--steps 30000] [--every 10] [--win 300] [--expect zero|gain]`: 같은 체크포인트를 `overrides=["taste_enabled=true"]`(켬)와 `["taste_enabled=false"]`(끔)로 — 같은 읽기 길 — `record`해 `flip_deltas` → `paired_diff` → `bootstrap_ci`. 출력: 쓸 수 있는 바뀜 수, 바뀜별 표, 평균과 구간. `--expect gain`(기본, ②)은 문턱(평균 > 0.05, 아래 끝 > 0, 바뀜 8번 이상), `--expect zero`(①)는 |평균| ≤ 0.02 통과 여부를 낸다. 바뀜이 8번 아래면 "바뀜 모자람 — `--steps`를 늘려 다시"를 내고 판정하지 않는다(30,000스텝에서 쓸 수 있는 바뀜은 평균 약 9번이라 모자랄 수 있다). `--json <파일>`.
  - `first-meals <체크포인트> [--steps 30000] [--seeds 1,2,3] [--win 300]`: 씨앗마다 `taste_inherit=1`과 `=0`으로(그리고 `seed=<씨앗>`) 읽어 `eat_tap` 안에서 한 스텝씩 돌린다. 먹기 직후 콜백이 산 새끼의 `n_meal`·`n_poison10`을 `ids`로 갱신한다. 스텝이 끝나면 새로 생긴 `ids`(그 스텝 `birth_step`)를 새끼로 등록하되 그 스텝에 `C_RESEED`가 늘었으면 빼고 뺀 수를 센다. 끝에 `split_children` → 켬/끔의 old끼리 `ratio_diff_ci`(끔 − 켬, 양수면 물려받은 쪽이 독을 덜 먹음), fresh도 같이 적는다. 무리 크기·평균 에너지 평균도 적는다. 문턱(old 차이 > 0.05, 아래 끝 > 0) 통과 여부. `--json <파일>`.

- [ ] **Step 1: 실패하는 시험을 쓴다** (`tests/test_taste_judge.py`, `sys.path`로 `wayfinder/stage2-4`를 넣어 가져온다)

```python
def test_flip_deltas_windows():
    steps = np.arange(0, 6000, 10); good = np.where(steps < 1500, 0, np.where(steps < 4500, 1, 0))
    blind = np.where((steps >= 1500) & (steps < 1800), 0.3, 0.6)
    d = flip_deltas(steps, good, blind)
    assert [x["flip"] for x in d] == [1500]           # 4500의 바뀜은 다음 바뀜이 없어 빠진다
    assert d[0]["early"] == pytest.approx(0.3) and d[0]["late"] == pytest.approx(0.6) and d[0]["delta"] == pytest.approx(0.3)

def test_flip_deltas_skips_short_gaps():
    # 바뀜 간격 500(< 2 × 300)인 바뀜은 넣지 않는다

def test_paired_diff_matches_by_flip():
    on = [{"flip": 1500, "delta": 0.2}, {"flip": 4500, "delta": 0.1}]
    off = [{"flip": 4500, "delta": 0.0}, {"flip": 1500, "delta": 0.05}]
    assert paired_diff(on, off).tolist() == pytest.approx([0.15, 0.1])

def test_bootstrap_ci_brackets_mean_and_is_repeatable():
    x = np.array([0.1, 0.12, 0.08, 0.11]); a = bootstrap_ci(x); assert a == bootstrap_ci(x)
    assert a[1] <= a[0] <= a[2] and a[0] == pytest.approx(x.mean())

def test_split_children_and_pooled_ratio():
    rows = [{"since_flip": 500, "meals": 10, "poison": 2}, {"since_flip": 100, "meals": 10, "poison": 6},
            {"since_flip": 900, "meals": 3, "poison": 1}, {"since_flip": None, "meals": 0, "poison": 0}]
    sp = split_children(rows)
    assert len(sp["old"]) == 3 and len(sp["fresh"]) == 1
    assert pooled_ratio(sp["old"]) == pytest.approx(3 / 13)    # 10끼 못 채운 새끼도 먹은 만큼 들어간다

def test_last_flip_looks_back_before_run_start():
    # good_kind로 실제 바뀜 경계 하나(b0·period)를 찾아 두고, t = b0·period + 100에서 그 스텝이 나온다.
    # 그 경계와 다음 실제 바뀜 사이의, 다시 정했지만 안 바뀐 경계는 건너뛴다

def test_ratio_diff_ci_resamples_clusters():
    # 한 묶음 안 새끼 100마리가 모두 독 10/10, 다른 묶음 100마리가 0/10인 무리 a와, 묶음 둘이 모두 5/10인 b:
    # 값은 0이지만 구간이 넓다(묶음 둘을 다시 뽑으므로 끝이 ±0.5 근처). 새끼 단위로 뽑았다면 구간이 좁았을 것

def test_eat_tap_sees_meals_before_death():
    # 개체 하나 세계(taste_world("taste", init_agents=1, min_agents=0, food_inflow=0)), 에너지 0.5,
    # 그 자리에 독 한 알: eat_tap 안에서 w.run(1) → 콜백이 n_meal 1, n_poison10 1을 보고, 스텝 뒤 개체는 죽어 있다
```

- [ ] **Step 2: 실패 확인** — `<시험> tests/test_taste_judge.py`
- [ ] **Step 3: 구현한다.** 명령줄 부분의 실행 확인(작은 체크포인트로 짧게)은 진행자가 한다 — 서브에이전트는 시험만 돌린다. 모듈 설명에 두 하위 명령, 판정 문턱, "독 0 대조는 쓰지 않는다(이름표 분기)"를 적는다.
- [ ] **Step 4: 통과 확인** — `<시험> tests/test_taste_judge.py`, 그리고 전체 `<시험>`(약 3분)
- [ ] **Step 5: 커밋** — "판정 도구: 바뀜별 켬/끔 차이와 새끼의 첫 10끼"

---

## 구현 뒤 실행 순서 (진행자 몫, 과제 아님)
1. 결과 문서(`wayfinder/stage2-4/results.md` 끝 새 절)에 판정 0·가·나·다·라의 예측과 문턱을 **실행 전에** 적고 커밋한다.
2. 측정 도구를 작은 입력으로 먼저 돌려 최대 메모리를 잰다(`/usr/bin/time -v`). `record`의 장면 수는 `steps / every`라 `--every 10`에서 30,000스텝이 3,000장면이다.
3. 판정 0(손 개체) → 안 되면 장치를 고친다.
4. 진화 갈래 셋 × 씨앗 둘(`k4-ctrl` 230만에서 `kkumteul.run branch`, `preset=taste_size` / `preset=taste` / `preset=taste`+`taste_inherit=1`, 각 30만 스텝), 커밋한 코드 사본에서, 2스레드·1.5GB·한 번에 하나.
5. 가(`flip-diff`, ② 끝은 `--expect gain`, ① 끝은 `--expect zero`) → 나(`first-meals`, ③ 끝) → 다(`measure` 겨루기 ② 대 ①. `assay_sides(a, b, ...)`는 `b["cfg"]`로 세계를 만들므로 ②를 b로 준다) → 라(`knockout split`, 채널 `taste`·`taste_vis`·`taste_val`, 음성 대조 ① 끝에 `--set taste_enabled=true --channel taste`, 양성 대조 `vision`).

## 검토 반영 (첫 판 `76911c5` → 둘째 판)
맥락 없는 서브에이전트 둘(sonnet)이 따로 읽었다(02:0x ET). 지적은 Claude가 코드로 확인했다.

| 축 | 지적 | 확인 | 판단 |
|---|---|---|---|
| 명세 | 독 먹은 스텝에 죽은 새끼의 끼니가 사라진다(먹기 → 죽음 → 출생, 죽은 칸 재사용) | 맞음(`world.py` step 순서) | 반영: `eat_tap`으로 먹기 직후 읽음, 시험 더함 |
| 명세 | ① 음성 대조가 뜻이 없다(장치 끈 규칙이라 켬=끔, 막기도 0을 막음) | 맞음 | 반영: 켬도 덮어쓰기 길, `--expect zero`, `knockout split --set` |
| 명세 | 나의 구간이 새끼 단위 뽑기라 너무 좁다 | 맞음 | 반영: (씨앗, 바뀜) 묶음 뽑기 |
| 명세 | 마지막 바뀜을 실행 안에서만 센다 | 맞음 | 반영: `last_flip_at_or_before` |
| 명세 | 바뀜 8번이 모자랄 수 있음, 두 갈래 읽기 길이 다름 | 맞음 | 반영: 모자라면 판정 않고 알림, 같은 길 |
| 명세 | 판정 0이 "바뀜 뒤"가 아니라 다시 정하는 주기로 잼, 0.35 판정 없음, `taste_in0` −1 쓰기 | 맞음 | 반영: `after_flips`, 통과 줄, `ValueError` |
| 명세 | `taste_size` = 같은 크기 끈 세계 해시 시험 없음 | 맞음 | 반영: Step 0에서 장치 전 코드로 뽑아 박음 |
| 명세 | 과제 3 시험이 주석뿐 | 부분 | 그대로: 단언할 값과 자리를 적었고 칸 번호는 시험이 찾게 함 |
| 규칙 | 저장 시험의 h0가 과제 2 뒤 깨짐 | 맞음 | 반영: `taste_size` 세계로 |
| 규칙 | 10끼 시험이 `e_max`에 막힘 | 맞음 | 반영: `feed`가 에너지를 2로 다시 맞춤 |
| 규칙 | 시험 가져오기 줄·`ko.` 접두어 | 맞음 | 반영 |
| 규칙 | `n_meal`이 사체·고기도 셈 | 맞음 | 기각: 끼니는 끼니다(설계 "먹은 알 수"). 정의에 적음 |
| 규칙 | 계획에 로컬 경로 | 맞음 | 반영: `memcap`, `<원래 폴더>` |
| 규칙 | 박은 해시를 구현 뒤 출력으로 뽑음, 세계가 다름 | 맞음 | 반영: Step 0, 두 세계 |
| 규칙 | `taste_on_poison`의 스텝 기준이 기존 `good_kind` 키와 다름, 스텝 0 | 맞음 | 부분: 설계대로 방금 돈 스텝(`step_no − 1`)을 쓰고 주석에 차이를 적음, `max(…, 0)` |
| 규칙 | `FIRST_MEALS`가 kernels.py | 맞음 | 반영: config.py |
| 규칙 | 켠 세계 장부 시험이 과제 4에야 | 맞음 | 반영: 과제 2에 |
| 규칙 | 냄새: 독 판정 중복 | 맞음 | 반영: `if poison:` |
| 규칙 | 냄새: `_reset_body`·`inject_agent` 초기화 목록 중복, −1 센티넬, 상수 묶음 반복 | 판단 사항 | 그대로: 기존 구조(두 초기화 자리는 이미 따로 있음), 이번 범위 밖 |
