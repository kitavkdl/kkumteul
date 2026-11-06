# 혹독한 세계 2부-나 — 큰 사냥감·뭉쳐 자기 (구현 계획)

둘째 판(2026-11-05). 맥락 없는 검토 두 갈래(규칙 축·명세 축)를 코드로 확인해 반영했다. 판단 표는 `docs/superpowers/specs/2026-11-05-harsh-world-review.md` 끝 절 "2부-나 계획 검토".

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 여럿이 함께 해야 이득인 큰 사냥감(④)과 겨울에 붙어 있으면 덜 드는 기초 대사(⑥)를 만들고, 앞 묶음이 남긴 사소한 지적 셋을 고친 뒤 진짜 체크포인트에서 숫자를 잰다.

**Architecture:** 동물은 개체·먹이와 따로 고정 크기 배열(`an_*`, `max_animals`칸)에 두고 새 파일 `kkumteul/animals.py`의 순차 커널이 다룬다. 사냥감은 유입의 한 몫(`budget[2]`)에서 태어나고, 다 차 있으면 그 몫이 풀 배수를 곱해 풀로 돌아간다. 쓰러지면 고기(`KIND_MEAT`, 들판 아닌 물건) 24알이 된다. 뭉쳐 자기는 겨울에 닿은 이웃 수로 개체마다 기초 대사 절감 비율을 적어 두고 `move`가 그 비율을 쓴다. 장치를 끈 세계는 계산 결과가 비트까지 같아(절감 0이면 `× 1.0`) 해시가 그대로다.

**Tech Stack:** Python 3, numpy, numba(`njit`), pytest, node(다시 보기 화면 점검).

**Spec:** `docs/superpowers/specs/2026-11-05-harsh-world-design.md`(여섯째 판) — ④, ⑥, "장부·계수기·해시·다시 보기", "구현 메모", "판정" 0. 검토 기록 `docs/superpowers/specs/2026-11-05-harsh-world-review.md` 끝 절 "2부-가 구현 뒤 남은 사소한 지적". 앞 묶음 계획 `docs/superpowers/plans/2026-11-05-harsh-world-part2a.md`. ⑤(포식자·약탈자·헷갈림)와 섞기 대조는 2부-다 계획이다.

## 설계와 다른 점
`config.py` 기본값으로 셈해 본 결과다. 설계 숫자는 바꾸지 않았고, 시험과 결과 문서가 아래대로 다룬다.

| # | 설계 | 셈 | 이 계획 |
|---|---|---|---|
| 1 | 혼자 사냥은 비용 14 > 얻는 것 약 7이라 손해 | 개체는 에너지가 `e_max`(8) 아래일 때만 한 알(1)씩 먹으므로 몸에 담을 수 있는 것은 9 아래다. 혼자 이어 치면 스텝당 0.14(반격 0.12 + 공격 비용 0.02) + 기초 대사를 잃어 약 62스텝 만에 바닥나고, 사냥감은 100스텝째에 쓰러진다. 둘이 사냥하면 각 4를 잃는데 world4 평균 에너지가 1.9라 배부른 개체만 버틴다(넷이면 1.25) | 숫자 그대로 둔다. "혼자는 손해"보다 강하게 **혼자는 이어 쳐서는 못 잡는다**(치고 빠지기만 가능, 상처는 100스텝 쉬어야 아문다). 손 시험이 `혼자 비용 > e_max + food_energy`를 단언하고 0번 결과에 적는다 |
| 2 | 함께할수록 이득이 커지다 넷 근처에서 포화, "그 뒤 평평" | 쓰러뜨리는 시간(100·50·34·25·25·25스텝)과 머릿수당 비용(14·4·2.04·1.25·1.1·1.0)은 넷 뒤로 평평하다. 고기 24알을 n이 나누므로 배 채움(6) 기준 순이득은 −8·2·3.96·4.75·3.7·3.0으로 넷 뒤에 줄어든다 | 시험은 "넷까지 늘고 그 뒤로는 늘지 않음", "넷부터 쓰러뜨리는 시간이 같음"으로 단언한다 |
| 3 | 혼자 잡아 고기 곁에 머물며 여러 번 먹는 경우까지 셈해야 한다 | 여름에는 고기가 400스텝 뒤 3배 빨리 상해 약 1,270스텝에 사라지고, 머무는 개체는 약 8을 먹는다(< 14). 겨울 머리에 잡으면 부패가 0.2배라 약 1,900스텝 남고 약 12.6을 먹는다(14에 가까움) | 손 시험은 여름 `먹은 양 < 14`와 겨울 먹은 양 > 여름만 단언한다. 겨울 값은 0번 결과 문서에 적는다 |
| 4 | 뭉쳐 자기: 닿은(반지름 2 안) 개체 수 | `collide`가 겹친 개체를 중심 거리 2.0(지름)까지 밀어내므로 "2 안"은 부동소수 경계에 걸려 거의 세지지 않는다 | 건넴·공격과 같은 닿음 거리(중심 거리 `2 × agent_radius + 0.5` = 2.5)로 센다 |
| 5 | "늘 먹을 수 있는 종류"를 한 함수 `always_edible(kind)`로 | 다시 보기는 배열로 가려야 한다 | 상수 묶음 `ALWAYS_EDIBLE` 하나에서 numba 스칼라 `always_edible`과 numpy `always_edible_mask` 두 판을 만든다. 시험이 0~3에서 둘이 같음을 지킨다 |
| 6 | 판정 0: 0번에서 겨울 저점이 약 250에서 얼마나 오르는지 본다 | 갈래에서 `bout[ACT_ATTACK] = -1`로 맞추면 1만 스텝 안에 공격이 거의 없어 잴 거리가 없을 수 있다(들기가 0번 둘째에서 한 번도 쓰이지 않은 것과 같은 까닭). 사냥이 다 되면 겨울 먹이는 0.3 → 0.51배, 저점의 위쪽 한계는 약 1.7배(약 400)다 | 설계대로 낸 갈래에 **진단 갈래 하나**(공격 편향 0에서 시작)를 더한다. 판정 규칙은 바꾸지 않는다 |

## 설계가 맡기거나 적지 않아 이 계획이 정한 것
- 사냥감 반지름 `prey_radius = 2.0`. 공격이 닿는 중심 거리 = `prey_radius + agent_radius + 0.5` = 3.5. 사냥감은 개체와 부딪치지 않는다(겹쳐도 된다). 운석은 동물을 다치게 하지 않는다.
- 떠돌기: 목적지는 먹이 자리(목초지) 중심 가운데 하나다. 목적지에서 `prey_graze_r = 18`(먹이 자리 퍼짐과 같은 값이지만 뜻이 달라 따로 둔다) 안을 `prey_stay`스텝 동안 `0.1 v_max`로 방향을 조금씩 흔들며 맴돈다. **가두기**: 맴도는 동안 목적지까지 감긴 거리가 `prey_graze_r`보다 크면 그 스텝의 방향은 목적지 쪽(+ 흔들기)이다. 끝나면 다른 먹이 자리를 골라 `0.3 v_max`로 간다. 고르기는 재시도 없이 정한다: 난수로 번호 k 하나를 뽑고, 그 중심이 지금 자리에서 `2 × prey_graze_r` 안이면 k+1, k+2, …(번호를 돌며) 처음으로 먼 것을 쓴다. 모두 가까우면 k를 쓴다. 먹이 자리가 없으면 세계 안 균등한 점으로 간다.
- **난수 주소(바꾸지 않는다)**: 목적지 고르기 도우미 `_pick_target(seed, stream, a, b0, ...)`는 `b0`(먹이 자리 번호), `b0+1`·`b0+2`(먹이 자리가 없을 때 x·y)를 쓴다. `spawn_prey`: `S_ANIMAL_POS`, `a = step`, 자리 `b = 5*slot`·`5*slot+1`, 첫 목적지 `b0 = 5*slot+2`. `move_animals`: `S_WANDER`, `a = slot`, 흔들기 `b = 4*step`, 새 목적지 `b0 = 4*step+1`. 같은 스텝에 태어나고 움직여도 흐름이 달라 겹치지 않는다.
- 동물 필드에 `an_born`(수명 셈)을 더한다. 저장하지 않는 스크래치는 `an_natt`, `an_dmg`(동물), `a_hit`, `huddle_cut`(개체)이다. `an_timer > 0`이면 맴도는 중(남은 스텝), `0`이면 가는 중이다. `an_rest`는 마지막으로 맞은 뒤 지난 스텝 수(`prey_rest`에서 멈춤)다 — 필드 주석에 그대로 적는다.
- 쓰러뜨린 사냥감마다 그 스텝의 공격자 수(`an_natt`)를 새 계수기 `C_HUNTERS = 37`(예비 37~39의 첫 칸)에 더한다. `C_HUNTERS / C_PREY_KILLED`가 한 마리에 붙은 평균 머릿수다(판정 2 "사냥에 붙은 개체 수"의 가장 싼 꼴). 누가 쳤는지(무임승차, 판정 6)는 남기지 않는다 — ⑤와 함께 사냥 기록 장치로 만든다.
- 겨루기 세계(`measure.arena_config`)는 설정을 그대로 넘기므로 사냥감·뭉쳐 자기도 켜진 채 겨룬다. 설계 "측정 경기장에서도 위 규칙으로 생긴다"대로이고 고치지 않는다.
- 포화는 한 스텝 피해의 상한 `prey_sat_n × attack_damage`(출력 1 기준 넷 몫)로 둔다.
- 반격으로 에너지가 바닥난 개체는 닿음 감각을 `TOUCH_HIT`로 받아 `DEATH_KILLED`로 센다(`harsh2`는 공격이 꺼져 있어 이 원인은 반격 죽음뿐이다). 반격은 쓰러지는 스텝에도 든다.
- 공격과 사냥감이 함께 켜지면 공격 출력의 비용은 `st.attack`에서 한 번만 든다. `attack_animals`는 공격이 꺼진 세계에서만 비용을 매긴다. 한 출력이 같은 스텝에 이웃과 동물을 둘 다 칠 수 있다.
- 사냥감 감각 두 칸 = 시야 거리·시야각 안의 가장 가까운 사냥감에 대해 `가까움 × (cos, sin)`(머리 방향 기준). 가까움 = `1 − (거리 − prey_radius) / vision_range`. 없으면 0, 0.
- 수명으로 사라진 사냥감은 고기를 남기지 않는다(설계대로 `L_LOST`).

## Global Constraints
- 장치는 기본 꺼짐. 모두 끈 세계는 `tests/test_stages.py`의 `PINNED_HASH_300` 세 값, 확장 크기 세계는 `tests/test_harsh.py`의 `PINNED_EXT_HASH_300` 세 값이 그대로여야 한다. **과제마다 Step 4에서 두 시험을 따로 돌린다.** 과제 5가 사냥감·뭉쳐 자기 세계의 해시 `PINNED_PREY_HASH_300`을 박는다(2부-다가 지킨다).
- `preset=harsh`의 뜻은 바꾸지 않는다. `preset=harsh2`에는 과제 5가 `prey_enabled`, `huddle_enabled`를 더한다(⑤는 2부-다가 더한다).
- 에너지가 생기는 길은 먹기와 명시적 주입뿐. 사냥감은 유입 몫에서 태어나며 `L_SPAWN`에 적는다. 장치마다 `|residual|/(spawn+inject) < 1e-9`. 장부 시험은 장치를 빼면 실패하는 꼴로 쓴다(빠지면 틀어지는 양을 단언한다).
- `prange` 안에서 상태 묶음(`s.x`)을 거쳐 쓰지 않는다. 새 커널은 모두 순차(`@njit(cache=True)`)다. 감각 커널은 배열을 직접 받는다.
- 난수 용도 번호는 바꾸지 않는다. 이 계획이 처음 쓰는 번호는 `S_ANIMAL_POS = 19`(사냥감 나는 자리), `S_WANDER = 20`(떠돌기), `S_MEAT = 23`(고기 뿌리기) 셋뿐이다. 21·22·24는 2부-다 몫이라 쓰지 않는다.
- 동물은 ⑤와 같은 배열을 쓴다. 종류는 `animals.ANIMAL_PREY = 0`, `ANIMAL_PRED = 1`, `ANIMAL_SCAV = 2`(1·2는 이름만)이다. 상처·회복은 종류와 무관한 도우미 `wound`, `heal`로 두고, `attack_animals`는 종류를 가리지 않고 가장 가까운 산 동물을 친다. 결과(반격·고기)만 종류로 가른다. 동물 필드도 개체·먹이 필드와 같은 해시 규칙을 따른다: `state_hash`의 동물 반복문은 `name in HASH_DEFAULTS`이고 산 동물 칸의 값이 모두 그 기본값이면 건너뛴다. 그래서 다음에 더할 동물 필드는 `HASH_DEFAULTS`에 기본값을 적으면 `PINNED_PREY_HASH_300`을 바꾸지 않는다.
- **바깥에서 부르는 커널**(`spawn_food`, `spoil_food`, `move`, `sense` 등)에 인자를 더하면 기본값으로 옛 호출을 살린다. 안에서만 부르는 도우미와 새 `animals.py` 커널의 인자는 기본값 없이 둔다.
- 설정값은 모두 `config.py`에 둔다. 새 숫자 주석에 "임시 결정"과 근거(지구값과 출처, 또는 셈을 말로)를 적는다. "계획", "표 N", "다른 점 N" 같은 문서 안 번호는 쓰지 않는다.
- 공개 레포: 코드·주석·시험에 로컬 경로, 다른 프로젝트, "과제 N"·"2부" 같은 작업 과정 표현을 쓰지 않는다(장치·상태를 말하는 표현으로 쓴다).
- 시험 실행: 작업 폴더에서 `NUMBA_NUM_THREADS=2`, 메모리 상한 2GB, 낮은 우선순위로 `python -m pytest -q -p no:cacheprovider <대상>`(이하 `<시험>`). 전체 약 3분. 서브에이전트는 시험만 돌리고 시뮬레이션 실험은 돌리지 않는다. 시험 임시 폴더는 pytest `tmp_path`만 쓴다. 시험 개수는 적지 않고 실행 출력으로 확인한다.
- 커밋 메시지는 한국어로 쓰고, 고친 파일만 `git add`한다(`.crew`, `.superpowers` 제외). 커밋 뒤 작업 가지로 push한다.

## Review Focus
1. **각본이 실행 중에 `prey_enabled`를 끄고 다시 켜면** 산 사냥감이 없어지고(에너지 → `L_LOST`), `budget[2]`가 `budget[0]`으로 돌아가며, 다시 켜면 몫이 차는 대로 새로 태어나고, 장부가 닫힌다 — 과제 2 `test_prey_toggle_off_clears_and_on_again`.
2. **반격으로 에너지가 음수가 된 개체**는 그 스텝에 `DEATH_KILLED`로 죽고, 음수 잔액까지 장부에 들어간다 — 과제 3 `test_counter_strike_can_kill_as_killed`.
3. **저장 칸이 모두 든 물건으로 찬 세계에서 사냥감이 쓰러지면** 고기가 `L_LOST`로 가고, 들판 알 칸을 침범하지 않으며, 장부가 닫힌다 — 과제 3 `test_kill_with_full_store_loses_meat_to_ledger`.
4. **세계의 이어진 가장자리 너머에서 친 공격**도 닿는다(감긴 거리) — 과제 3 `test_attack_reaches_across_wrap`.
5. **겨울에 붙어 자는 개체가 잠까지 자면** 잠 환급이 깎인 기초 대사 기준이라 에너지가 생기지 않는다 — 과제 4 `test_huddle_sleep_refund_uses_cut_basal`.
6. **동물 필드가 없는 옛 체크포인트**(1부·들기까지의 확장 체크포인트)는 exact로 읽히고 저장 해시 검증을 통과하며, `harsh2`로 갈라 켜면 공격 편향이 −1로 맞춰진다 — 과제 2 `test_old_checkpoint_without_animal_fields_loads_exact`, 과제 5 `test_branch_into_harsh2_calms_attack`.

---

### Task 1: 고기 종류·늘 먹을 수 있는 종류, 앞 묶음 남은 지적 (numba 커널 몇 곳)

**Files:**
- Modify: `kkumteul/kernels.py` (`KIND_CORPSE` 아래에 `KIND_MEAT`, `ALWAYS_EDIBLE`, `always_edible`, `always_edible_mask` / `is_field` / `eat`의 독 조건 / `spoil_food` / 실수 칸 표에 `P_MEAT_SPOIL` / 정수 칸 표에서 `Q_CARRY`와 그 주석을 지움 / `carry`의 `if field_cap != 0` 지움)
- Modify: `kkumteul/world.py` (`recount_field`가 `always_edible_mask`를 씀, `set_config`에서 `field_cap`이 꺼지면 `C_NFIELD = 0`)
- Modify: `kkumteul/config.py` (`meat_spoil_mult`, `validate`에 "`field_cap`이면 `ext_size`")
- Test: `tests/test_harsh.py` (도우미 `hold`, `_place_stored`, `_n_field`가 사체만이 아니라 `always_edible`을 보게 고침)

**Interfaces:**
- Produces (kernels.py): `KIND_MEAT = 3`. `ALWAYS_EDIBLE = (KIND_CORPSE, KIND_MEAT)`. `always_edible(kind) -> bool`(`@njit(cache=True, inline="always")`, 스칼라). `always_edible_mask(kinds: np.ndarray) -> np.ndarray[bool]` = `np.isin(kinds, ALWAYS_EDIBLE)`(파이썬). `is_field(s, j)` = `f_dropped[j] == 0 and not always_edible(f_kind[j])`. 실수 칸 `P_MEAT_SPOIL`(= `cfg.meat_spoil_mult`).
- Produces (config.py): `meat_spoil_mult: float = 3.0`(주석: 고기는 부패 속도에 이 배수를 곱한다. 지구: 씨앗·견과 ≪ 과일 ≪ 고기, DeVault et al. 2004 [2차]. 임시 결정). `validate`: `meat_spoil_mult >= 1`, `spoil_rate * meat_spoil_mult <= 1`(에너지가 음수가 되지 않게). `self.field_cap and not self.ext_size`면 `ValueError`("들판 계수기는 확장 크기 세계에서만 쓴다(계수기 칸이 모자란다)") — `store_room`·`evict_stored`는 스스로 경계를 보지 않으므로, `field_cap`을 넓히는 장치가 `needs_ext`를 빠뜨려도 칸 밖에 쓰기 전에 여기서 막는다(지금은 `needs_ext`와 겹치는 막이다).
- Produces: `spoil_food`는 고기 칸에서만 `rate * fp[P_MEAT_SPOIL]`을 쓴다. 고기가 아닌 칸의 계산식은 지금과 비트까지 같다(곱하지 않는 분기).

- [ ] **Step 1: 실패하는 시험을 쓴다** (`tests/test_harsh.py`)

```python
def test_always_edible_scalar_and_mask_agree():
    kinds = np.arange(4, dtype=np.int8)
    assert [bool(k.always_edible(int(v))) for v in kinds] == k.always_edible_mask(kinds).tolist() == [False, False, True, True]

def test_meat_is_never_poison():
    # kinds 세계(빈 세계 + 개체 하나), 지금 독인 종류가 아닌 쪽과 상관없이 고기 한 알을 닿게 놓고 _eat_once →
    # 개체 에너지 += 1.0, C_POISONED 그대로, 장부 잔차 < 1e-9
    ...

def test_meat_is_not_field():
    # carry 켠 ext 빈 세계에서 _place(..., kind=k.KIND_MEAT) → C_NFIELD 그대로, C_NFOOD += 1, recount_field 뒤에도 그대로
    ...

def test_meat_spoils_faster():
    # spoil 켠 빈 세계, 같은 스텝에 난 A 한 알과 고기 한 알, spoil_after + 100스텝 spoil_food →
    # 고기 에너지 == (1 - 3r)**100, A == (1 - r)**100 (r = spoil_rate), 둘 다 rel 1e-12
    ...

def test_corpse_takes_store_room_on_death():
    # small("ext", carry_enabled=True, max_agents=40, init_agents=10, min_agents=0, food_inflow=0.0):
    # 내려놓은 알 40(_place_stored, born 0..39)로 들판 아닌 물건 = 40 = max_agents. 개체 하나를 age = max_age, energy 3.0으로 →
    # process_deaths → C_STORE_EVICTED == 1, born 0인 알 칸은 비고, 죽은 자리에 KIND_CORPSE(에너지 3.0)가 있고,
    # C_NFOOD - C_NFIELD == 40, 잔차 < 1e-9. (_remove의 store_room 조건을 늘 참으로 바꾸면 41칸·C_STORE_EVICTED 0이라 실패)
    ...

def test_field_counter_zeroed_when_field_cap_off():
    # carry 켠 ext 세계를 20스텝 → set_config로 carry 끔 → C_NFIELD == 0. 다시 켜면 == 살아 있는 들판 알 수
    ...

def test_release_all_puts_items_at_holder():
    # 든 개체 둘을 서로 다른 자리로 옮긴 뒤 release_all → 물건 자리 == 든 개체 자리, f_holder·a_held 모두 -1, f_dropped 1
    ...

def test_carry_cost_mult_below_one_rejected():
    with pytest.raises(ValueError):
        small("ext", carry_enabled=True, carry_cost_mult=0.9)
```

- [ ] **Step 2: 실패를 확인한다** — `<시험> tests/test_harsh.py -k "always_edible or meat or corpse_takes or zeroed or release_all_puts or below_one"` → `KIND_MEAT` 없음 등으로 FAIL(`corpse_takes`·`release_all_puts`·`below_one`은 지금 코드에서 PASS해도 된다 — 지킴 시험이다).

- [ ] **Step 3: 구현한다**
  - `eat`: `kinds and not always_edible(s.f_kind[j]) and s.f_kind[j] != good`이면 독.
  - `spoil_food`: `r = rate`, 고기면 `r = rate * fp[P_MEAT_SPOIL]`, `lost = f_energy * r`.
  - `Q_CARRY`: 정수 칸 표·`pack_params`에서 지운다(마지막 칸이라 다른 번호는 그대로다). `grep -rn Q_CARRY kkumteul tests` 결과가 비어야 한다.
  - `carry`: 들판 알을 들 때 `if field_cap != 0:` 분기를 없애고 안쪽을 늘 한다(들기가 켜져 있으면 늘 `field_cap`이다).
  - `set_config`: `old.field_cap and not cfg.field_cap`이면 `s.counters[k.C_NFIELD] = 0`.
  - 시험 도우미 `hold`, `_place_stored`, `_n_field`의 `!= KIND_CORPSE`를 `always_edible` / `always_edible_mask`로 바꾼다.

- [ ] **Step 4: 시험 통과 + 해시** — `<시험> tests/test_stages.py::test_worlds_without_harsh_keep_pinned_hash tests/test_harsh.py::test_ext_worlds_keep_pinned_hash`, 이어 전체 시험. 기대: 모두 PASS.

- [ ] **Step 5: 커밋** — `git add kkumteul/kernels.py kkumteul/world.py kkumteul/config.py tests/test_harsh.py` → "고기 종류와 늘 먹을 수 있는 종류 한 규칙, 고기 부패 배수, 들판 계수기 정리(꺼지면 0, 확장 크기 요구), 쓰이지 않던 Q_CARRY 지움".

남은 지적 4~10 가운데 이 과제에 넣은 것은 4(꺼지면 0), 6(죽은 분기), 7의 일부(`release_all` 자리, `carry_cost_mult` 검사)다. 5(들기를 중간에 켰을 때 사체가 많으면 새 먹이가 버려짐), 7의 나머지, 8(다시 보기 드문 경우), 9(감각 사거리 경계), 10(작은 것)은 넣지 않는다 — 장부는 닫히고 고치는 값이 이번 장치와 무관하다.

---

### Task 2: ④ 동물 배열·사냥감 출생·떠돌기·유입 나누기·켜짐→꺼짐 (여러 파일, numba, 장부)

**Files:**
- Create: `kkumteul/animals.py`
- Modify: `kkumteul/kernels.py` (`ANIMAL_FIELDS`, `ANIMAL_SAVED`, `State`, `SAVED_FIELDS`, `alloc_state`)
- Modify: `kkumteul/world.py` (`__init__`·`set_config`의 `fa, ia` 묶기와 금지 목록 `max_animals`, `step`, `energy_balance`, `load_arrays`, `state_hash`, `clear_animals`)
- Modify: `kkumteul/config.py` (사냥감 설정, `needs_ext`, `field_cap`, `animals_on`, `validate`)
- Modify: `kkumteul/scenario.py` (`FIXED_SIZE`에 `max_animals`)
- Modify: `kkumteul/_cache.py` (`_SOURCES`에 `animals.py`)
- Create: `tests/test_prey.py`

**Interfaces:**
- Consumes: 과제 1의 `KIND_MEAT`, `is_field`. 앞 묶음의 `_place_food`, `store_room`, `C_PREY_KILLED`, `N_BUDGET_EXT`, `HASH_DEFAULTS`.
- Produces (config.py), 모두 임시 결정이고 근거는 설계 ④다: `prey_enabled: bool = False`, `max_animals: int = 64`(배열 크기, 실행 중 못 바꿈), `n_prey: int = 4`, `prey_hp: float = 20.0`, `prey_regen: float = 0.05`(스텝당, 쉰 뒤), `prey_rest: int = 100`, `prey_strike: float = 0.12`, `prey_sat_n: int = 4`, `prey_meat: int = 24`, `prey_inflow_frac: float = 0.3`, `prey_life: int = 12000`, `prey_stay: int = 1500`, `prey_graze_speed: float = 0.1`, `prey_travel_speed: float = 0.3`(둘 다 `v_max` 배수), `prey_radius: float = 2.0`, `prey_graze_r: float = 18.0`(뒤의 둘은 설계가 맡긴 값이라 주석에 "임시 결정, 먹이 자리 퍼짐과 같은 크기" 꼴로 적는다). `needs_ext`에 `prey_enabled`. `field_cap` = `carry_enabled or prey_enabled`. `animals_on -> bool` = `prey_enabled`(⑤가 넓힌다). `validate`: `1 <= max_animals`, `0 <= n_prey <= max_animals`, `prey_hp > 0`, `prey_regen >= 0`, `prey_rest >= 0`, `prey_strike >= 0`, `prey_sat_n >= 1`, `prey_meat >= 1`, `0 <= prey_inflow_frac < 1`, `prey_life >= 1`, `prey_stay >= 1`, `0 < prey_graze_speed <= 1`, `0 < prey_travel_speed <= 1`, `prey_radius > 0`, `prey_graze_r > 0`.
- Produces (kernels.py): `ANIMAL_FIELDS` = (이름, dtype, 저장) — `an_alive` bool, `an_kind` int8, `an_x`, `an_y`, `an_heading`, `an_hp`, `an_energy`, `an_tx`, `an_ty` float64, `an_timer`, `an_rest` int32, `an_born` int64(이상 저장), `an_natt` int32, `an_dmg` float64(저장 안 함). 길이는 늘 `cfg.max_animals`. `ANIMAL_SAVED` = 저장하는 이름들의 튜플. `State`·`SAVED_FIELDS`에 넣는다.
- Produces (animals.py):
  - `ANIMAL_PREY = 0`, `ANIMAL_PRED = 1`, `ANIMAL_SCAV = 2`.
  - 실수 칸 `(AF_W, AF_H, AF_INFLOW, AF_PREY_FRAC, AF_MEAT_E, AF_HP, AF_REGEN, AF_STRIKE, AF_RADIUS, AF_REACH, AF_GRAZE_V, AF_TRAVEL_V, AF_GRAZE_R, AF_ATK_DMG, AF_ATK_COST, N_FA)`(쓰는 곳이 없는 칸은 두지 않는다), 정수 칸 `(AI_SEED, AI_N_PREY, AI_REST, AI_SAT_N, AI_MEAT, AI_LIFE, AI_STAY, AI_CONFLICT, AI_N_PATCH, AI_PATCH_LIFE, N_IA)`. `AF_MEAT_E = prey_meat * food_energy`, `AF_REACH = prey_radius + agent_radius + 0.5`, `AF_GRAZE_V = prey_graze_speed * v_max`, `AF_TRAVEL_V = prey_travel_speed * v_max`.
  - `pack_animal_params(cfg) -> (fa: np.ndarray[float64], ia: np.ndarray[int64])`.
  - `prey_inflow(s, fa, ia, mult) -> None` — `budget[2] += inflow * frac`. `budget[2] > meat_e`면 넘친 만큼에 `mult`를 곱해 `budget[0]`으로 보내고 `budget[2] = meat_e`로 둔다.
  - `spawn_prey(step, s, fa, ia) -> int` — 산 사냥감이 `n_prey`보다 적고 `budget[2] >= meat_e`이고 빈 칸이 있으면 하나 낳는다. 자리는 `S_ANIMAL_POS`, `(step, 5*slot)`·`(step, 5*slot+1)`, 세계 안 균등. `hp = prey_hp`, `energy = meat_e`, `an_rest = prey_rest`, `an_timer = 0`, `an_born = step`, 첫 목적지를 고른다(난수 주소는 위 "이 계획이 정한 것"). `budget[2] -= meat_e`, `L_SPAWN += meat_e`. 낳은 수를 돌려준다.
  - `move_animals(step, s, fa, ia) -> None` — 사냥감마다: `step - an_born >= prey_life`면 없앤다(`L_LOST += an_energy`). 아니면 위 "이 계획이 정한 것"의 떠돌기(맴돌기 `an_timer > 0`·가두기 / 가기 `an_timer == 0`, 목적지에서 `prey_graze_r` 안에 들면 `an_timer = prey_stay`). 난수 주소는 위 "이 계획이 정한 것". 목적지는 `kernels.patch_center(seed, k, step, ...)`. 자리는 감는다(`_wrap`).
  - `animal_energy(s) -> float` — 산 동물의 `an_energy` 합.
- Produces (world.py): `self.fa, self.ia`. `World.clear_animals(kind: int) -> float` — 그 종류의 산 동물을 모두 없애고 에너지를 `L_LOST`에 넣은 뒤 그 합을 돌려준다. `energy_balance`는 `animal_energy`를 더하고 `"animals"` 키를 낸다.

- [ ] **Step 1: 실패하는 시험을 쓴다** (`tests/test_prey.py`, 머리는 `tests/test_harsh.py`처럼 `from test_stages import rel_residual, small`)

```python
def _prey_world(*presets, **kw):
    return World(small(*presets, "ext", prey_enabled=True, **kw))

def test_prey_needs_ext_and_turns_field_cap_on():
    with pytest.raises(ValueError):
        small(prey_enabled=True)
    cfg = small("ext", prey_enabled=True)
    assert cfg.field_cap and cfg.needs_ext and cfg.animals_on

def test_max_animals_is_locked():
    w = _prey_world()
    with pytest.raises(ValueError, match="배열 크기"):
        w.set_config(dataclasses.replace(w.cfg, max_animals=65))
    assert "max_animals" in scenario.LOCKED

def test_prey_born_from_prey_budget_and_ledger_closes():
    # 빈 세계(init_agents=0, min_agents=0, food_inflow 2.2): 300스텝 뒤 산 사냥감 == n_prey(4), 모두 hp == prey_hp,
    # an_energy == prey_meat * food_energy. 잔차 < 1e-9 (energy_balance가 animal_energy를 빼면 잔차가 96만큼 어긋나 실패)
    ...

def test_inflow_split_keeps_year_intake_when_no_hunting():
    # 빈 세계, small("harsh", "ext", prey_enabled=True, season_period=600, winter_len=150, autumn_len=150, autumn_mult=2.0).
    # 600스텝 돌려 사냥감 넷과 budget[2] == meat_e를 확인한 뒤, 600..1199의 들어온 양
    # IN = L_SPAWN + L_DISCARD + budget.sum()의 증가분 == Σ_t food_inflow * season(t)[0] (rel 1e-9),
    # 그 가운데 겨울 스텝(1050..1199)만의 증가분 == 150 * food_inflow * winter_mult (rel 1e-9).
    # (넘친 몫을 배수 없이 풀로 돌리면 겨울 몫이 0.51배가 되어 실패한다)
    ...

def test_world_step_passes_grass_share_to_niche_world():
    # test_world_step_passes_season_mult_to_niche_world 꼴, prey 켬: budget[1] 이동량이
    # food_inflow * season(t)[0] * (1 - prey_inflow_frac) * fert_frac * frac
    ...

def test_prey_world_counts_field_only():
    # 공격 없는 사냥감 세계(사체 켬) 400스텝, 매 스텝 C_NFIELD == (f_alive & f_dropped==0 & ~always_edible_mask(f_kind)).sum()
    ...

def test_prey_grazes_then_travels():
    # 사냥감 하나(prey_stay=400), move_animals를 1,200번: 맴도는 동안 한 스텝 이동 <= graze_speed*v_max + 1e-12이고
    # 맴도는 동안 늘 목적지에서 prey_graze_r + graze_speed*v_max 안(400스텝이면 가두기 없이는 40까지 벗어날 수 있어 가두기를 가린다),
    # 가는 스텝의 이동 == travel_speed*v_max(rel 1e-9), 목적지가 한 번 이상 바뀐다.
    # 같은 seed·같은 칸으로 두 번 돌리면 자취가 비트까지 같다(난수 주소가 상태에 기대지 않음)
    ...

def test_prey_dies_of_age_into_lost():
    # prey_life=30: 30스텝 뒤 그 사냥감이 없고 L_LOST가 meat_e만큼 늘고 잔차 < 1e-9
    ...

def test_prey_toggle_off_clears_and_on_again():
    # 사냥감 넷 + budget[2] > 0인 세계에서 set_config(prey 끔, carry도 끔) → an_alive 모두 False, L_LOST += 끄기 전 an_energy 합,
    # budget[2] == 0, budget[0] += 끄기 전 budget[2], C_NFIELD == 0, 이어 100스텝 잔차 < 1e-9.
    # 다시 켜면 몫이 차는 대로 새로 태어난다(300스텝 뒤 넷)
    ...

def test_animal_fields_enter_hash_only_when_alive():
    w = _prey_world(init_agents=0, min_agents=0); w.run(200)
    h = w.state_hash()
    a = int(np.flatnonzero(w.s.an_alive)[0])
    w.s.an_hp[a] -= 1.0
    assert w.state_hash() != h
    w.s.an_alive[:] = False
    h2 = w.state_hash()
    w.s.an_hp[:] = 123.0
    assert w.state_hash() == h2

def test_old_checkpoint_without_animal_fields_loads_exact(tmp_path):
    # 확장 세계(harsh)를 저장한 뒤 체크포인트에서 ANIMAL_SAVED를 모두 빼고 읽는다 → load_arrays가 True,
    # ckpt.load가 저장 해시 검증을 통과, an_alive 모두 False (test_old_checkpoint_without_carry_fields_loads_exact 꼴)
    ...

def test_prey_world_ledger_closes_long():
    # small("harsh", "stage4", "ext", prey_enabled=True) 1,500스텝(공격 출력은 처음 개체의 -1 그대로) → check_sane, 잔차 < 1e-9
    ...
```

- [ ] **Step 2: 실패를 확인한다** — `<시험> tests/test_prey.py` → `prey_enabled` 없음으로 FAIL.

- [ ] **Step 3: 구현한다**
  - `World.step` 순서(사냥감이 켜졌을 때만 부른다): 풀 배수 `grass` → `an.prey_inflow(s, fa, ia, grass)` → 풀 몫 `g = grass * (1.0 - cfg.prey_inflow_frac)`(꺼졌으면 `g = grass` 그대로, 곱하지 않는다) → `niche_world(..., g)` → `spawn_food(..., g)` → 부패 → `an.spawn_prey` → 격자·감각·생각 → `move`·`after_move` → `an.move_animals` → `collide` 이하 그대로.
  - `state_hash`: 먹이 필드 다음에, 산 동물이 하나라도 있으면 `ANIMAL_SAVED`마다 `v = arr[an_alive]`를 넣되, 개체·먹이 반복문과 같은 `HASH_DEFAULTS` 건너뛰기를 둔다. 산 동물이 없으면 아무것도 넣지 않는다. 시험 `test_animal_fields_enter_hash_only_when_alive`에 "`HASH_DEFAULTS`에 동물 필드 기본값을 넣은 꼴(monkeypatch로 `an_timer`: 그 세계의 값)이면 그 필드가 빠진다"를 더한다.
  - `load_arrays`: 빠진 필드가 `HASH_DEFAULTS`이거나 `ANIMAL_SAVED`면 기본값 그대로 두고 빠진 것으로 치지 않는다.
  - `set_config`: 금지 목록에 `max_animals`. `fa, ia`를 다시 묶는다. `old.prey_enabled and not cfg.prey_enabled`면 `clear_animals(ANIMAL_PREY)`, `budget[0] += budget[2]`, `budget[2] = 0`.
  - `alloc_state`: 동물 배열은 모두 0(`an_alive` False).

- [ ] **Step 4: 시험 통과 + 해시** — `<시험> tests/test_stages.py::test_worlds_without_harsh_keep_pinned_hash tests/test_harsh.py::test_ext_worlds_keep_pinned_hash`, 이어 전체(`tests/test_scenario.py::test_locked_list_covers_everything_world_refuses` 포함). 기대: 모두 PASS.

- [ ] **Step 5: 커밋** — `git add kkumteul/animals.py kkumteul/kernels.py kkumteul/world.py kkumteul/config.py kkumteul/scenario.py kkumteul/_cache.py tests/test_prey.py` → "큰 사냥감 바탕: 동물 배열, 유입 몫에서 태어나기, 넘친 몫은 풀 배수로 풀에, 떠돌기·수명, 끄면 정리, 동물 에너지 장부".

---

### Task 3: ④ 사냥 — 상처·회복·반격 나눠 맞기·포화·고기 (numba, 장부)

**Files:**
- Modify: `kkumteul/animals.py` (`wound`, `heal`, `attack_animals`, `_scatter_meat`)
- Modify: `kkumteul/kernels.py` (`AGENT_FIELDS`에 스크래치 `a_hit`, 계수기 `C_HUNTERS = 37`과 예비 주석 `# 38~39: 예비`)
- Modify: `kkumteul/world.py` (`step`: `st.attack` 자리 다음에 `an.attack_animals`)
- Test: `tests/test_prey.py`

**Interfaces:**
- Consumes: 과제 2의 동물 배열, `fa`/`ia` 칸, `AF_REACH`, `AF_MEAT_E`. `kernels.store_room`, `_place_food`, `TOUCH_HIT`, `C_PREY_KILLED`, `L_LOST`, `L_MOVE`. `config.ACT_ATTACK`. `rng.S_MEAT`.
- Produces (kernels.py): 개체 필드 `("a_hit", np.int32, (), False)` — 이번 스텝에 친 동물 칸(−1이면 없음).
- Produces (animals.py):
  - `wound(s, a, dmg) -> None` — `an_hp[a] -= dmg`, `an_rest[a] = 0`.
  - `heal(s, a, hp_max, regen, rest) -> None` — `an_rest[a] += 1`, `an_rest[a] >= rest`면 `an_hp[a] = min(hp_max, an_hp[a] + regen)`. (`an_rest`는 int32 넘침이 없게 `rest`에서 멈춘다)
  - `attack_animals(step, s, fa, ia) -> int` — 쓰러뜨린 수를 돌려준다. 차례:
    1. 모든 동물 칸의 `an_natt`·`an_dmg`를 0으로 둔다. 산 개체마다 `a_hit = -1`. 공격 출력 `a > 0`이고 에너지 > 0이면, 공격이 꺼진 세계(`AI_CONFLICT == 0`)에서만 비용 `attack_cost * a`(`L_MOVE`, 빗나가도 든다)를 매긴다. 감긴 중심 거리 `AF_REACH` 안의 가장 가까운 산 동물(종류 무관)을 `a_hit`에 적고 `an_natt += 1`, `an_dmg += attack_damage * a`.
    2. 맞은 사냥감마다 `wound(s, a, min(an_dmg, prey_sat_n * attack_damage))`.
    3. 사냥감을 친 개체마다 `prey_strike / an_natt[a]`를 잃는다(`L_LOST`), `touch = TOUCH_HIT`.
    4. `an_hp <= 0`인 사냥감은 쓰러진다: `C_PREY_KILLED += 1`, `C_HUNTERS += an_natt`, `_scatter_meat`, `an_alive = False`, `an_energy = 0`.
    5. 이번 스텝에 맞지 않은 산 사냥감마다 `heal(s, a, prey_hp, prey_regen, prey_rest)`.
  - `_scatter_meat(step, s, fa, ia, a) -> None` — `e = an_energy / prey_meat`를 `prey_meat`번 놓는다. 자리 = 사냥감 자리 + `prey_radius * normal(seed, S_MEAT, serial, 2q)`·`(…, 2q+1)`(감음), `serial` = 늘리기 전 `C_PREY_KILLED`. `store_room(s, 1)`이 참이고 `_place_food(s, px, py, e, KIND_MEAT, step, 1)`이 참이면 놓이고, 아니면 `L_LOST += e`.

- [ ] **Step 1: 실패하는 시험을 쓴다** (`tests/test_prey.py`. 커널을 직접 부르는 손 경기장: `food_inflow=0.0`, `min_agents=0`, `n_prey=0`(저절로 안 태어남), 사냥감은 `_put_prey`로 놓고 `L_SPAWN += meat_e`. 개체는 `s.act`를 직접 쓰고 에너지는 30(장부 `L_INJECT`에 적음)으로 둬 먹지도 죽지도 않게 한다)

`tests/test_harsh.py`의 도우미 `hold`, `_place`, `_fix_out`을 불러 쓴다.

```python
def _hunt_arena(n, **kw) -> World: ...                # small("ext", prey_enabled=True, n_prey=0, init_agents=n, min_agents=0, food_inflow=0.0, **kw), 개체 에너지 30
def _put_prey(w, x, y, hp=None) -> int: ...          # 빈 동물 칸에 사냥감 하나, 칸 번호
def _ring(w, n, cx, cy, r=2.5) -> np.ndarray: ...     # 산 개체 n마리를 (cx, cy) 둘레 반지름 r에 놓고 act[ACT_ATTACK]=1.0, 칸 번호들
def _hunt(w, a, idx, max_steps=200) -> tuple[int, np.ndarray]: ...  # attack_animals를 쓰러질 때까지, (스텝 수, 개체별 잃은 에너지)

def test_hunt_cost_curve():
    want_steps = {1: 100, 2: 50, 3: 34, 4: 25, 5: 25, 6: 25}
    steps, loss = {}, {}
    for n in range(1, 7):
        w = _hunt_arena(n)
        steps[n], lost = _hunt(w, _put_prey(w, 100.0, 100.0), _ring(w, n, 100.0, 100.0))
        assert abs(steps[n] - want_steps[n]) <= 1
        loss[n] = float(lost.mean())
        assert lost == pytest.approx(steps[n] * (0.12 / n + 0.02), rel=1e-9)
        assert (w.s.f_kind[w.s.f_alive] == k.KIND_MEAT).sum() == 24
        assert w.s.f_energy[w.s.f_alive].sum() == pytest.approx(24 * w.cfg.food_energy, rel=1e-12)
        assert rel_residual(w) < 1e-9
    cfg = small("ext", prey_enabled=True)
    assert loss[1] > cfg.e_max + cfg.food_energy           # 혼자는 이어 쳐서는 못 잡는다: 몸에 담을 수 있는 에너지보다 비용이 크다
    net = {n: min(24.0 / n, 6.0) - loss[n] for n in loss}  # 배 채움 6 기준
    assert net[1] < 0 < net[2] < net[3] < net[4]
    assert net[5] <= net[4] and net[6] <= net[4]           # 넷 뒤로 늘지 않는다
    assert steps[4] == steps[5] == steps[6]                # 포화
    # 각 n의 세계에서 C_HUNTERS == n, C_PREY_KILLED == 1 (쓰러뜨린 스텝의 공격자 수가 쌓인다)

def test_saturation_caps_damage_per_step():
    # 여섯이 한 스텝 → hp가 정확히 20 - 4*0.2 == 19.2
    ...

def test_wounds_heal_only_after_rest():
    # 혼자 10스텝 침(hp 18) → 치지 않고 99스텝 뒤 hp == 18, 100스텝째 18.05, 오래 두면 20에서 멈춘다
    ...

def test_counter_strike_split_and_touch_hit():
    # 셋이 한 스텝: 각 에너지 -0.04(L_LOST) -0.02(L_MOVE), touch == TOUCH_HIT.
    # conflict 켠 같은 세계에서는 attack_animals가 비용을 매기지 않는다: 각 -0.04만
    ...

def test_counter_strike_can_kill_as_killed():
    # 에너지 0.01인 개체가 혼자 치면 → world.step 뒤 그 개체가 없고 사망 기록 cause == DEATH_KILLED, C_DEATH_KILLED == 1, 잔차 < 1e-9
    ...

def test_kill_with_full_store_loses_meat_to_ledger():
    # max_agents=8, 개체 8 모두가 들판 알을 하나씩 든 꼴(hold 도우미, 들판 아닌 물건 = 8 = max_agents), 넷이 hp 0.5 사냥감을 침 →
    # 고기 0알, L_LOST가 meat_e + 반격만큼 늘고, C_NFOOD - C_NFIELD == 8, 잔차 < 1e-9
    ...

def test_attack_reaches_across_wrap():
    # 사냥감 (1, 1), 개체 (world_w - 1, 1) (감긴 거리 2) → 한 스텝 뒤 hp == 19.8
    ...

def _camp(winter: bool) -> float:
    # _hunt_arena(1, spoil_enabled=True[, season_enabled=True]), winter면 step_no = season_period - winter_len(겨울 첫 스텝).
    # 개체 하나, 나이 0, 에너지 6(혼자 14를 치른 뒤), 멈춰 세움(_fix_out(w, i, carry=-5.0): 추력 0, 번식 -5, 들기 -5,
    # 공격은 편향 0이라 출력 0 → 치지 않음),
    # 고기 24알을 born = step_no로 개체 둘레 1.0 안에 놓는다(_place, kind=KIND_MEAT, L_SPAWN에 적음).
    # 고기가 다 없어질 때까지(최대 3,000스텝) world.step, 그 개체의 s.eaten을 돌려준다
    ...

def test_lone_hunter_camping_at_meat():
    summer, winter = _camp(False), _camp(True)
    assert summer < 14.0                                   # 여름에 혼자 머물러 먹어도 비용을 못 메운다
    assert winter > summer                                 # 겨울엔 고기가 늦게 상해 더 먹는다
    assert winter < 14.0                                   # 겨울에도 혼자는 손해다(설계 ④). 셈으로는 약 12.6이라 여유가 작다
    # 이 단언이 실패하면 시험을 느슨하게 고치지 않는다. 멈추고 값을 보고한다(설계 숫자를 다시 정할 일이다)

def test_hunting_world_ledger_closes():
    # small("ext", prey_enabled=True, prey_hp=2.0)(부패 없음), 200스텝 뒤 산 개체 모두 bout[ACT_ATTACK]=+3, 산 사냥감마다 산 개체 하나의
    # 자리로 옮겨 놓고 600스텝 → C_PREY_KILLED > 0, 매 스텝 센 고기 알 수의 최대 > 끝 값(부패가 없으니 줄어든 것은 먹힌 것),
    # 잔차 < 1e-9. 같은 세계를 prey 끄고 돌리면 L_LOST가 다르다
    ...
```

- [ ] **Step 2: 실패를 확인한다** — `<시험> tests/test_prey.py -k "hunt or saturation or heal or counter or full_store or wrap or camping"` → `attack_animals` 없음으로 FAIL.

- [ ] **Step 3: 구현한다** — Interfaces의 차례대로 짠다. `World.step`: `if cfg.conflict_enabled: st.attack(...)` 다음에 `if cfg.prey_enabled: an.attack_animals(t, s, self.fa, self.ia)`. `bench.kernel_breakdown`, `measure`, `knockout`은 고치지 않는다(동물 커널을 부르지 않는다).

- [ ] **Step 4: 시험 통과 + 해시** — 두 박은 해시 시험, 이어 전체. 기대: 모두 PASS.

- [ ] **Step 5: 커밋** — `git add kkumteul/animals.py kkumteul/kernels.py kkumteul/world.py tests/test_prey.py` → "사냥: 상처가 쌓이고 쉬면 아묾, 반격을 공격자 수로 나눠 맞음, 넷 몫에서 포화, 쓰러지면 고기 24알".

---

### Task 4: ⑥ 뭉쳐 자기 (numba, 장부)

**Files:**
- Modify: `kkumteul/config.py` (`huddle_enabled`, `huddle_cap`, `huddle_exp`, `needs_ext`, `validate`)
- Modify: `kkumteul/kernels.py` (`AGENT_FIELDS`에 스크래치 `huddle_cut`, 실수 칸 `P_HUDDLE_CAP`·`P_HUDDLE_EXP`, 새 `huddle` 커널, `move`의 기초 대사)
- Modify: `kkumteul/stages.py` (`after_move`의 잠 환급)
- Modify: `kkumteul/world.py` (`step`, `set_config`)
- Test: `tests/test_harsh.py`

**Interfaces:**
- Produces (config.py): `huddle_enabled: bool = False`(겨울에만 닿은 이웃 수로 기초 대사를 줄인다. 붙어 있으면 생기는 물리 효과), `huddle_cap: float = 0.4`, `huddle_exp: float = 0.33`(지구: 겨울 대사 6~53% 절약, 대략 N^−0.33, 상한 0.3~0.5 — Gilbert et al. 2010 [2차]. 임시 결정). `needs_ext`에 `huddle_enabled`. `validate`: 켜면 `season_enabled`가 있어야 하고, `0 <= huddle_cap < 1`, `huddle_exp >= 0`.
- Produces (kernels.py): 개체 필드 `("huddle_cut", np.float64, (), False)`. `huddle(s, fp, ip, on) -> None` 순차 — `on`이면 산 개체마다 스텝 처음에 세운 개체 격자로 중심 거리 `2 * agent_radius + 0.5` 안의 다른 산 개체 수 n을 세어 `huddle_cut = min(cap, 1 − (1 + n) ** −exp)`. `on`이 아니면 모두 0.
- Produces: `move(s, fp, ip, basal_mult=1.0)`의 기초 대사 = `fp[P_BASAL] * basal_mult * (1.0 - s.huddle_cut[i])`. `after_move`의 잠 환급 = `fs[F_BASAL] * basal_mult * (1.0 - s.huddle_cut[i]) * (1.0 - fs[F_SLEEP_BASAL])`. 곱하는 차례를 이대로 두면 `huddle_cut == 0`일 때 지금과 비트까지 같다.

- [ ] **Step 1: 실패하는 시험을 쓴다**

```python
def _huddle_world(**kw):
    return World(small("harsh", huddle_enabled=True, season_period=400, winter_len=200, **kw))

@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5])
def test_huddle_cut_follows_neighbours(n):
    # init_agents = n+2, min_agents=0. n+1마리를 한 점 둘레 반지름 1.2 원에 놓고(모두 서로 2.5 안), 한 마리는 멀리 →
    # 개체 격자를 세우고 k.huddle(s, fp, ip, True)을 직접 부른다 →
    # 그 무리의 huddle_cut == min(0.4, 1 - (1 + n) ** -0.33) (n=1 → 0.2045, 2 → 0.3041, 3 → 0.3671, 4·5 → 0.4), 멀리 있는 개체 0
    ...

def test_huddle_only_in_winter():
    # 같은 무리, 봄 스텝(100)에 한 스텝: _one_step_basal == n * basal_cost (줄지 않음), huddle_cut 모두 0
    ...

def test_huddle_cuts_basal_and_ledger_closes():
    # 무리를 지어 놓은 같은 세계의 겨울 스텝(399) 한 스텝을 huddle 켬·끔으로: 켬의 ΔL_BASAL < 끔의 ΔL_BASAL이고 그 차이 ==
    # Σ basal_cost * winter_basal_mult * huddle_cut (스텝 앞에 살아 있던 칸의 값, rel 1e-12). 두 세계 모두 잔차 < 1e-9.
    # 이어 1,200스텝 잔차 < 1e-9 (huddle이 아무것도 안 하면 첫 단언이 실패한다)
    ...

def test_huddle_sleep_refund_uses_cut_basal():
    # test_winter_basal_multiplier_reaches_sleep_refund 꼴에 huddle 켬: 자는 개체의 한 스텝 순 기초 대사 ==
    # m * (1 - cut) * sleep_basal (m = basal_cost * winter_basal_mult), 모두 >= 0
    ...

def test_huddle_needs_season_and_ext():
    with pytest.raises(ValueError):
        small("ext", huddle_enabled=True)                  # 겨울 없음
    with pytest.raises(ValueError):
        small(season_enabled=True, huddle_enabled=True)    # 기본 크기

def test_huddle_off_clears_cut():
    # 겨울 스텝에 huddle_cut이 0보다 큰 세계에서 set_config로 끔 → huddle_cut 모두 0
    ...
```

- [ ] **Step 2: 실패를 확인한다** — `<시험> tests/test_harsh.py -k huddle` → `huddle_enabled` 없음으로 FAIL.

- [ ] **Step 3: 구현한다** — `World.step`: 생각(`think`/`think_plastic`, `after_think`) 다음, `move` 앞에서 `if cfg.huddle_enabled: k.huddle(s, fp, ip, self.in_winter(t))`. `set_config`: `old.huddle_enabled and not cfg.huddle_enabled`면 `s.huddle_cut[:] = 0.0`. `summary`: 켜져 있으면 `"huddle_cut_mean"` = 산 개체 `huddle_cut` 평균(산 개체가 없으면 0.0). `if n:` 블록 밖에 둔다.

- [ ] **Step 4: 시험 통과 + 해시** — 두 박은 해시 시험과 `tests/test_harsh.py -k "winter_basal"`(1부 겨울 기초 대사 시험이 비트까지 그대로), 이어 전체. 기대: 모두 PASS.

- [ ] **Step 5: 커밋** — `git add kkumteul/config.py kkumteul/kernels.py kkumteul/stages.py kkumteul/world.py tests/test_harsh.py` → "뭉쳐 자기: 겨울에 닿은 이웃 수로 기초 대사를 줄임(상한 0.4), 잠 환급도 줄인 대사 기준".

---

### Task 5: 사냥감 감각·갈라 켜기의 공격 편향·harsh2 묶음·기록·해시 박기 (여러 파일, numba)

**Files:**
- Modify: `kkumteul/animals.py` (`sense_animals`)
- Modify: `kkumteul/world.py` (`step`의 감각, `attack_newly_on`, `World.calm_attack`, `summary`)
- Modify: `kkumteul/run.py` (`Runner.branch` 조건, `write_stats`의 `w_killed`)
- Modify: `kkumteul/replay.py` (`record`의 덧붙임 경로)
- Modify: `kkumteul/config.py` (`preset("harsh2")`)
- Test: `tests/test_prey.py`, `tests/test_harsh.py`(`harsh2`를 쓰는 기존 시험이 깨지면 고침)

**Interfaces:**
- Consumes: 과제 2·3·4의 모든 것. `config.EXT_PREY = 3`, `Config.ext_in0`, `Config.animals_on`.
- Produces (animals.py): `sense_animals(alive, x, y, heading, obs, col, an_alive, an_kind, an_x, an_y, kind, radius, vrange, fov, w, h) -> None` 순차 — 산 개체마다 `kind` 종류의 산 동물 가운데 `거리 − radius < vrange`이고 `|각도| − asin(min(1, radius / 거리)) <= fov / 2`(거리 ≤ radius면 늘 보임)인 가장 가까운 것 하나로 `obs[i, col] = c * cos(각도)`, `obs[i, col+1] = c * sin(각도)`를 쓴다(`c = 1 − (거리 − radius) / vrange`, 각도는 머리 기준 −π~π). 없으면 0, 0. ⑤가 `EXT_PRED`·`EXT_SCAV`에 같은 커널을 쓴다.
- Produces (world.py): `attack_newly_on(old: Config, new: Config) -> bool` = `(new.conflict_enabled and not old.conflict_enabled) or (new.animals_on and not old.animals_on)`. `World.calm_attack() -> list[str]` — 산 개체 `bout[:, ACT_ATTACK] = -1`, `["bout[ACT_ATTACK]=-1"]`을 돌려준다. `summary`: 사냥감이 켜져 있으면 `"prey_alive"`, `"prey_killed"`(`C_PREY_KILLED`), `"prey_hunters"`(`C_HUNTERS`), `"deaths_killed"`. 이 키들과 과제 4의 `"huddle_cut_mean"`은 개체 수와 상관없이 내야 하므로 `summary`의 `if n:` 블록 **밖**(`self.cfg`로)에 둔다.
- Produces (config.py): `preset("harsh2")` = `harsh` + 가을 + `carry_enabled` + `prey_enabled` + `huddle_enabled`.

- [ ] **Step 1: 실패하는 시험을 쓴다**

```python
def test_prey_direction_input():
    # 개체 (100,100) 머리 0, 사냥감 (108,100) → obs[ext_in0+EXT_PREY : +2] == (0.625, 0.0) (c = 1 - (8-2)/16)
    # 30도 왼쪽 거리 8 → (0.625 cos30°, 0.625 sin30°). (100,108)(90도, 시야 밖) → (0,0). (92,100) 뒤 → (0,0).
    # (120,100)(틈 18 > 16) → (0,0). 앞에 8·12 둘 → 8짜리
    ...

def test_attack_newly_on():
    base, ext = small(), small("ext")
    assert attack_newly_on(ext, small("ext", prey_enabled=True))
    assert attack_newly_on(base, small("conflict"))
    assert not attack_newly_on(small("ext", prey_enabled=True), small("ext", prey_enabled=True))
    assert not attack_newly_on(base, ext)

def test_branch_into_harsh2_calms_attack(tmp_path):
    # 기본 크기 main을 400스텝 → Runner.branch(..., ["preset=harsh2"]) → prey·huddle·carry·season 켜짐,
    # 산 개체 bout[:, ACT_ATTACK] 모두 -1, run.json의 genome_edits에 "bout[ACT_ATTACK]=-1", 이어 300스텝 잔차 < 1e-9,
    # resume하면 해시가 같다. 대조 갈래 ["preset=ext"]는 calm_attack=True일 때만 -1이다
    ...

def test_replay_record_calms_attack_for_animal_override(tmp_path, monkeypatch):
    # World.calm_attack을 부른 횟수를 센다: replay.record(ckpt, steps=10, overrides=["preset=harsh2"]) → 1번,
    # overrides 없이 → 0번
    ...

def test_prey_world_does_not_depend_on_thread_count():
    # 사냥이 일어나는 세계(test_hunting_world_ledger_closes의 꼴, huddle·season 짧게)를 스레드 1개와 2개 이상으로 300스텝 → 해시 같음,
    # C_PREY_KILLED > 0
    ...

def test_summary_reports_prey():
    w = World(small("harsh2")); w.run(200)
    out = w.summary()
    assert out["prey_alive"] == int(w.s.an_alive.sum()) and "prey_killed" in out and "huddle_cut_mean" in out
    w.s.alive[:] = False                                   # 개체가 모두 없어도 키는 남는다
    assert "prey_alive" in w.summary() and "huddle_cut_mean" in w.summary()

# 이 커밋의 코드가 낸 300스텝 해시. 포식자·약탈자를 더한 뒤에도 그것들을 끈 이 설정은 같아야 한다.
PREY_HASH_CFG = dict(season_period=200, winter_len=60, autumn_len=40, autumn_mult=2.0,
                     carry_enabled=True, prey_enabled=True, huddle_enabled=True)
PINNED_PREY_HASH_300 = "<Step 3 끝에서 계산해 넣는다>"

def test_prey_world_keeps_pinned_hash():
    w = World(small("harsh", **PREY_HASH_CFG)); w.run(300)
    assert int(w.s.an_alive.sum()) > 0                     # 사냥감이 해시에 들어간다. 300스텝 안에 겨울(140~199)도 있다
    assert w.state_hash() == PINNED_PREY_HASH_300
```

- [ ] **Step 2: 실패를 확인한다** — `<시험> tests/test_prey.py -k "direction or newly_on or calms or thread or summary or pinned"` → `sense_animals`·`attack_newly_on` 없음으로 FAIL.

- [ ] **Step 3: 구현한다**
  - `World.step`: 감각 블록 끝(들기 입력 다음)에 `if cfg.prey_enabled: an.sense_animals(s.alive, s.x, s.y, s.heading, s.obs, cfg.ext_in0 + EXT_PREY, s.an_alive, s.an_kind, s.an_x, s.an_y, an.ANIMAL_PREY, cfg.prey_radius, cfg.vision_range, cfg.vision_fov, cfg.world_w, cfg.world_h)`.
  - `Runner.branch`: 조건을 `calm_attack or attack_newly_on(old, cfg)`로 바꾸고 `edits += world.calm_attack()`. `write_stats`: `w_killed`를 `conflict_enabled or prey_enabled`일 때 낸다.
  - `replay.record`: 덧붙임 경로에서 `calm_new_outputs` 다음에 `if attack_newly_on(ckpt.read_config(path), world.cfg): world.calm_attack()`. 그렇게 했으면 `notes`에 "공격 출력 편향을 −1로 맞췄습니다"를 넣는다.
  - `preset("harsh2")`에 둘을 더한다. 전체 시험에서 `harsh2`를 쓰는 기존 시험(`test_autumn_*`, `test_branch_into_harsh2`)이 깨지면, 그 시험의 뜻(가을·들기)대로 묶음 대신 `harsh` + 가을 + `carry_enabled`로 고쳐 쓴다.
  - 해시 박기: 다른 시험이 모두 통과한 뒤 `World(small("harsh", **PREY_HASH_CFG))`를 300스텝 돌려 `state_hash()`를 시험의 `PINNED_PREY_HASH_300`에 넣는다. 스레드 1개와 2개에서 같은 값인지 확인한 뒤 넣는다.

- [ ] **Step 4: 시험 통과 + 해시** — 두 박은 해시 시험과 `tests/test_prey.py::test_prey_world_keeps_pinned_hash`, 이어 전체. 기대: 모두 PASS.

- [ ] **Step 5: 커밋** — `git add kkumteul/animals.py kkumteul/world.py kkumteul/run.py kkumteul/replay.py kkumteul/config.py tests/test_prey.py tests/test_harsh.py` → "사냥감 방향 감각, 동물 장치를 새로 켜는 갈래·다시 보기에서 공격 편향 −1, harsh2에 사냥감·뭉쳐 자기, 사냥감 세계 해시 박기".

---

### Task 6: 다시 보기 — 고기·동물 (replay.py, replay.html, node 점검)

**Files:**
- Modify: `kkumteul/replay.py` (`Recorder.__init__`·`_capture`·`finish`, `Replay.__init__`, `pack`, `_read_food`, meta `on`·`cfg`)
- Modify: `kkumteul/replay.html` (먹이 종류 3 그리기, 동물 그리기, 범례)
- Modify: `tests/replay_screen_probe.js` (머리 주석의 빛깔 목록에 사냥감·고기)
- Test: `tests/test_replay.py`

**Interfaces:**
- Consumes: `kernels.always_edible_mask`, `KIND_MEAT`, `animals.ANIMAL_PREY`, 동물 배열.
- Produces (replay.py): 장면마다 산 동물을 칸 차례로 적는다 — `an_off`(장면별 시작, `<u4`), `an_x`, `an_y`(`_q16`), `an_kind`(`u1`), `an_hp`(`u1` = `clip(hp / prey_hp * 255)`). `Replay.__init__`에 마지막 인자 `animals: dict | None = None`. `pack`은 이 다섯 배열을 늘 넣는다(동물이 없으면 길이 0). meta `on`에 `"prey"`, `"huddle"`, `cfg`에 `"prey_radius"`. `seen` 키는 바꾸지 않는다(`tests/test_replay.py`의 사전 비교 그대로).
- Produces: `_read_food`의 규칙 —
  - 독 판정: `cfg.food_kinds_enabled and not always_edible(kind) and kind != good`. `_capture`의 `bad`도 `~always_edible_mask(kinds)`.
  - 상한 교체 후보 거르기: 조건을 `carry`에서 `cfg.field_cap`으로, 들판 알 = `(p_dropped == 0) & ~always_edible_mask(p_kind)`.
  - 저장 칸 치움 후보: `(p_dropped == 1) | always_edible_mask(p_kind)`.
  - 썩음 재구성: 커널과 같은 차례로 `r = spoil_rate * spoil_mult(t)`, 고기면 `r * meat_spoil_mult`, `left = e − e * r`.
  - 위치 비교(`same`)는 지금처럼 `carry`일 때만 뺀다(고기는 움직이지 않는다).

- [ ] **Step 1: 실패하는 시험을 쓴다** (`tests/test_replay.py`)

```python
def test_replay_records_animals_and_meat():
    # 사냥감 세계(kinds 켬, prey_hp=2.0, 개체 무리 곁에 사냥감, 산 개체 bout[ACT_ATTACK]=+3)를 Recorder로 120스텝 →
    # 장면마다 동물 수 == 그 장면의 산 동물 수(an_off 차이), 쓰러진 뒤 종류 3인 먹이 행이 24개 열리고,
    # 고기를 먹은 행의 why는 WHY_GOOD(독 아님), seen["unknown"] == 0, pack 결과에 an_x 등 다섯 배열
    ...

def test_replay_tells_spoiled_meat_from_eaten():
    # test_replay_tells_spoiled_food_from_eaten 꼴(tests/test_harsh.py)에서 고기 한 알: 하한 아래로 떨어지는 스텝을
    # 고기 배수(3)로 셈해 WHY_SPOILED로 적는다(배수를 빼면 먹힘으로 적혀 실패)
    ...

def test_replay_without_animals_has_empty_animal_arrays(kinds_run):
    # 동물 없는 세계의 기록: an_off는 모두 0이고 an_x 길이 0, pack 디렉터리에 다섯 배열이 길이 0으로 있다
    ...
```

- [ ] **Step 2: 실패를 확인한다** — `<시험> tests/test_replay.py -k "animals or spoiled_meat"` → `an_off` 없음으로 FAIL.

- [ ] **Step 3: 구현한다** — Interfaces의 규칙대로 고친다. `replay.html`: `an_*` 배열이 있으면 지도에 사냥감을 반지름 `prey_radius`의 원으로 그린다(채움 한 빛깔, 체력은 테두리 호). 먹이 종류 3은 고기 빛깔로 그린다. 범례에 "사냥감", "고기". 새 빛깔 이름을 `tests/replay_screen_probe.js` 머리 주석의 목록에 더한다.

- [ ] **Step 4: 시험 통과** — 전체 시험, `node tests/replay_screen_check.js`(기존 쓰임대로 `test_screen_logic_agrees_with_the_recording`가 부른다). 기대: 모두 PASS, `test_replay.py`의 `unknown ≤ 2%` 시험 그대로. 두 박은 해시 시험도 돌린다.

- [ ] **Step 5: 커밋** — `git add kkumteul/replay.py kkumteul/replay.html tests/replay_screen_probe.js tests/test_replay.py` → "다시 보기: 사냥감을 장면마다 그리고, 고기는 독이 아니며 3배 빨리 상하는 것으로 가림".

---

### Task 7: 0번 숫자 재기 — 사냥감과 뭉쳐 자기 (세션이 직접, 서브에이전트 아님)

앞 묶음 계획의 과제 5와 같은 꼴이다. 2스레드, `memcap --bg --max 1500M`, 한 번에 하나만 돌린다. 갈래는 커밋한 코드의 사본(`runs/code-<해시>`)에서, world4 1,200만 체크포인트(봄에서 시작)부터 약 1만 스텝(겨울 둘: 2,800~4,000, 6,800~8,000) 돌린다. 기록은 200스텝, 체크포인트는 2,000스텝마다. 읽기는 앞 0번의 읽기 스크립트(git 밖, 인수인계에 자리)를 늘려 쓴다.

- [ ] **Step 1: 예측을 `wayfinder/stage2-4/results.md`에 먼저 적고 커밋한다.** 새 절 "혹독한 세계 0번 셋째: 사냥감과 뭉쳐 자기"에 적는다:
  - 이름: 이 코드의 `preset=harsh2`는 가을·들기·사냥감·뭉쳐 자기다(⑤는 아직 없음). 0번 둘째의 `h1-harsh2`는 가을·들기뿐이었다 — 결과 문서에 두 뜻을 한 줄로 적는다.
  - 기준은 0번 둘째의 `h1-harsh2`(가을 + 들기)다. 사냥감·뭉쳐 자기를 끈 세계는 이 코드에서도 같은 결과가 나와야 한다. 의심되면 2,000스텝만 다시 돌려 첫 체크포인트의 해시를 견준다.
  - 사냥감(편향 −1): 1만 스텝에 쓰러뜨린 수 0~5. 사냥감 넷이 늘 차 있어 몫이 풀로 돌아가므로(겨울엔 0.3배) 겨울 저점은 기준(230~244)과 가를 수 없다. 공격 출력이 0보다 큰 개체가 새로 비용을 내므로 그 비율은 처음 1% 아래이고 1만 스텝 뒤 더 줄어든다. 반격 죽음(`deaths_killed`)은 한 자리 수다.
  - 사냥감(진단, 편향 0): 공격하는 개체가 절반 가까이로 늘어 공격 비용(스텝당 최대 0.02, 기초 대사의 약 7배) 때문에 처음 몇백 스텝 굶어 죽는 수가 크게 는다. 쓰러뜨림은 수십이지만 겨울 저점을 올리지 못한다.
  - 비용 대조(편향 0, 못 잡는 사냥감): 진단 갈래와 같은 손댐에 `prey_hp`만 크게 해 쓰러지지 않게 한다. 공격 비용·반격만 있고 고기는 없다. 진단 갈래와의 겨울 저점 차이가 사냥 이득이다. 예측: 진단 갈래보다 저점이 같거나 낮다(사냥이 이득이면 진단이 높다). 위쪽 한계는 사냥이 몫(스텝당 2.7)을 다 쓸 때 겨울 먹이 0.51배, 저점 약 1.7배(약 400)이고, 그러려면 겨울에 8.9스텝마다 한 마리(겨울 하나에 약 135마리)를 잡아야 한다.
  - 뭉쳐 자기: 겨울 저점이 기준보다 0~15% 높다. 겨울 `huddle_cut_mean`은 0.05~0.2다. 여름은 같다.
  - 둘 다: 두 효과의 합이다. 재주입은 모든 갈래에서 0이다.
  - 사냥이 일어나는 갈래의 한 마리에 붙은 평균 머릿수(`prey_hunters / prey_killed`)는 3~5다.
- [ ] **Step 2: 갈래 다섯을 하나씩 돌린다.** `python -m kkumteul.run branch --from <world4 실행> --at 12000000 --run runs/<갈래> --steps 10000 --quiet --set preset=harsh2 …`
  - `h2-prey`: `--set huddle_enabled=0`.
  - `h2-huddle`: `--set prey_enabled=0`.
  - `h2-full`: 덧붙임 없음.
  - `h2-prey-b0`(진단): `h2-prey`와 같은 설정이다. 실행 폴더 밖의 임시 스크립트가 `Runner.branch` 뒤 산 개체의 `bout[:, ACT_ATTACK] = 0`으로 바꾸고, **손댐을 실행 폴더에 남긴다**: `run.json`의 `genome_edits`에 `"bout[ACT_ATTACK]=0 (진단, 스크립트 <파일 이름>)"`을 덧붙여 다시 쓰고, 첫 체크포인트를 다시 저장하고, 스크립트 사본을 실행 폴더에 둔다. 그 뒤 첫 체크포인트를 읽어 산 개체의 공격 편향이 모두 0인지 확인하고 `loop`한다(도중에 `resume`해도 0이 이어진다). 결과 문서에도 적는다.
  - `h2-prey-b0-hp`(비용 대조): `h2-prey-b0`와 같은 스크립트·손댐에 `--set prey_hp=1e9`.
  - 판정 단계의 대조 갈래(`preset=ext`)는 설계대로 공격 편향도 −1로 맞춰야 하는데 `attack_newly_on`은 동물 장치가 없는 대조 갈래를 잡지 않는다. 그 갈래는 `--calm-attack`을 붙인다 — 결과 문서에 미리 적는다.
- [ ] **Step 3: 숫자를 결과 문서에 적는다.** 해 위상별 개체 수, 여름 고점·가을 고점·겨울 저점·감소폭, 재주입, 죽음 원인(굶음·늙음·반격), `prey_killed`·산 사냥감 수, 공격 출력 > 0 비율, 고기 알 수와 썩은 고기, 겨울 `huddle_cut_mean`, 한 마리에 붙은 평균 머릿수, 개체 에너지 분포(에너지가 4·2.04·1.25 이상인 개체 비율 — 둘·셋·넷 사냥을 버틸 수 있는 몫), 고기 곁에서 공격 없이 먹는 무임승차는 이 장치로 잴 수 없다는 것(사냥 기록 장치는 ⑤와 함께), 갈래별 실행 속도(sps — 판정 3의 빼기 갈래 순서를 정하는 데 쓴다)를 적는다. 손 시험 `test_lone_hunter_camping_at_meat`의 여름·겨울 먹은 양(14와 견줌)도 적는다. 운석은 동물을 다치게 하지 않는다는 것도 한 줄 적는다.
- [ ] **Step 4: 판정한다.** 세계가 무너지면(재주입 > 0, 또는 겨울 저점이 기준의 절반 아래) 원인을 갈래로 가른다. 설계 근거에 맞는 값 하나만 고쳐(사냥감이면 `prey_strike`나 사냥감 몫, 뭉쳐 자기는 지구값이라 고치지 않음) `config.py`와 설계 문서를 함께 고치고 판정(Ruling)으로 남긴다. 목표 숫자가 없는 것(겨울 저점 상승)은 "잴 수 있었는가 / 위쪽 한계 대비 얼마인가"로 적는다. 진단 갈래에서도 사냥이 저점을 못 올리면, 그 까닭(공격 비용, 만나기 어려움, 반격 죽음)을 숫자로 가른다.
