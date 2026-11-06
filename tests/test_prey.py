"""큰 사냥감 시험: 동물 배열, 유입 몫에서 태어나기, 떠돌기·수명, 끄면 정리, 동물 에너지 장부,
사냥(상처·회복·반격 나눠 맞기·포화·고기), 사냥감 감각, 공격 편향 맞추기, 요약, 박은 해시.

    python3 -m pytest -q tests/test_prey.py
"""
import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_harsh import _fix_out, _place, hold  # noqa: E402
from test_stages import rel_residual, small  # noqa: E402
from kkumteul import animals as an  # noqa: E402
from kkumteul import checkpoint as ckpt  # noqa: E402
from kkumteul import kernels as k  # noqa: E402
from kkumteul import replay  # noqa: E402
from kkumteul import scenario  # noqa: E402
from kkumteul.config import ACT_ATTACK, EXT_PREY  # noqa: E402
from kkumteul.run import Runner  # noqa: E402
from kkumteul.world import World, attack_newly_on  # noqa: E402


def _prey_world(*presets, **kw):
    return World(small(*presets, "ext", prey_enabled=True, **kw))


def _n_prey(s):
    return int((s.an_alive & (s.an_kind == an.ANIMAL_PREY)).sum())


def _meat_e(cfg):
    return cfg.prey_meat * cfg.food_energy


def _wdist(ax, ay, bx, by, w, h):
    return math.hypot(k._wdiff(ax - bx, w), k._wdiff(ay - by, h))


def test_prey_needs_ext_and_turns_field_cap_on():
    with pytest.raises(ValueError):
        small(prey_enabled=True)
    cfg = small("ext", prey_enabled=True)
    assert cfg.field_cap and cfg.needs_ext and cfg.animals_on
    off = small("ext")
    assert not off.field_cap and not off.animals_on


def test_prey_config_validated():
    for kw in (dict(max_animals=0), dict(n_prey=65), dict(n_prey=-1), dict(prey_hp=0.0), dict(prey_regen=-0.1),
               dict(prey_rest=-1), dict(prey_strike=-0.1), dict(prey_sat_n=0), dict(prey_meat=0),
               dict(prey_inflow_frac=1.0), dict(prey_inflow_frac=-0.1), dict(prey_life=0), dict(prey_stay=0),
               dict(prey_graze_speed=0.0), dict(prey_graze_speed=1.5), dict(prey_travel_speed=0.0),
               dict(prey_travel_speed=1.5), dict(prey_radius=0.0), dict(prey_graze_r=0.0),
               dict(prey_wander_turn=-0.1), dict(prey_wander_turn=4.0)):
        with pytest.raises(ValueError):
            small("ext", prey_enabled=True, **kw)
    small("ext", prey_enabled=True, n_prey=64, prey_inflow_frac=0.0)


def test_max_animals_is_locked():
    w = _prey_world()
    with pytest.raises(ValueError, match="배열 크기"):
        w.set_config(dataclasses.replace(w.cfg, max_animals=65))
    assert "max_animals" in scenario.LOCKED


def test_animal_arrays_have_max_animals_slots_and_start_empty():
    w = World(small("ext", max_animals=10))
    for name, dt, _ in k.ANIMAL_FIELDS:
        v = getattr(w.s, name)
        assert v.shape == (10,) and v.dtype == dt, name
    assert not w.s.an_alive.any()
    assert set(k.ANIMAL_SAVED) <= set(k.SAVED_FIELDS)
    assert "an_natt" not in k.ANIMAL_SAVED and "an_dmg" not in k.ANIMAL_SAVED


def test_prey_born_from_prey_budget_and_ledger_closes():
    # 빈 세계: 사냥감 몫(유입의 prey_inflow_frac)이 한 마리 몫만큼 찰 때마다 하나씩 태어나 n_prey에서 멈춘다.
    # energy_balance가 동물 에너지를 빼면 잔차가 n_prey × meat_e(96)만큼 어긋난다
    w = _prey_world(init_agents=0, min_agents=0)
    w.run(300)
    s, cfg = w.s, w.cfg
    assert _n_prey(s) == cfg.n_prey == 4
    a = s.an_alive
    assert (s.an_hp[a] == cfg.prey_hp).all()
    assert (s.an_energy[a] == _meat_e(cfg)).all()
    assert (s.an_born[a] < 300).all() and (s.an_rest[a] == cfg.prey_rest).all()
    b = w.energy_balance()
    assert b["animals"] == pytest.approx(cfg.n_prey * _meat_e(cfg), rel=1e-12)
    assert rel_residual(w) < 1e-9


def test_inflow_split_keeps_year_intake_when_no_hunting():
    # 사냥감이 다 차 있고 아무도 사냥하지 않으면 들어온 양(먹이로 난 것 + 버린 것 + 예산 변화)이
    # 사냥감이 없는 세계와 같다: 스텝마다 food_inflow × 풀 배수. 넘친 사냥감 몫도 풀 배수를 곱해 풀로 간다
    w = World(small("harsh", "ext", prey_enabled=True, season_period=600, winter_len=150, autumn_len=150,
                    autumn_mult=2.0, init_agents=0, min_agents=0))
    w.run(600)
    s, cfg = w.s, w.cfg
    assert _n_prey(s) == cfg.n_prey
    assert s.budget[2] == _meat_e(cfg)

    def intake():
        L = s.ledger
        return float(L[k.L_SPAWN] + L[k.L_DISCARD] + s.budget.sum())

    in0 = intake()
    w.run(450)                       # 600..1049
    in_w = intake()
    w.run(150)                       # 1050..1199: 겨울
    in1 = intake()
    assert w.in_winter(1050) and w.in_winter(1199) and not w.in_winter(1049)
    want = sum(cfg.food_inflow * w.season(t)[0] for t in range(600, 1200))
    assert in1 - in0 == pytest.approx(want, rel=1e-9)
    assert in1 - in_w == pytest.approx(150 * cfg.food_inflow * cfg.winter_mult, rel=1e-9)
    assert _n_prey(s) == cfg.n_prey
    assert rel_residual(w) < 1e-9


@pytest.mark.parametrize("t", [100, 399])
def test_world_step_passes_grass_share_to_niche_world(t):
    # 사냥감이 켜지면 거름 몫은 풀 몫의 배수(풀 배수 × (1 - prey_inflow_frac))를 따른다
    w = World(small("stage4", "ext", season_enabled=True, season_period=400, winter_len=200, prey_enabled=True))
    w.step_no = t
    w.s.soil[:] = 0.0
    w.s.soil[0] = 0.1 * w.cfg.soil_ref
    w.s.budget[:] = 0.0
    mult = w.season(t)[0]
    frac = 0.1 * (1.0 - w.cfg.soil_decay)
    share = w.cfg.food_inflow * mult * (1.0 - w.cfg.prey_inflow_frac) * w.cfg.fert_frac * frac
    assert share < w.cfg.food_energy and mult != pytest.approx(1.0)
    w.step()
    assert w.s.budget[1] == pytest.approx(share, rel=1e-9)
    assert w.s.budget[2] == pytest.approx(w.cfg.food_inflow * w.cfg.prey_inflow_frac, rel=1e-12)


def test_prey_world_counts_field_only():
    # 사냥감 세계는 들판 알만 센다(공격이 없으니 고기는 없고, 늙어 죽은 개체의 사체는 세지 않는다)
    w = _prey_world(max_age=300)
    for t in range(400):
        w.step()
        s = w.s
        n = int((s.f_alive & (s.f_dropped == 0) & ~k.always_edible_mask(s.f_kind)).sum())
        assert s.counters[k.C_NFIELD] == n, t
    assert w.s.counters[k.C_DEATH_AGE] > 0 and _n_prey(w.s) == w.cfg.n_prey
    assert rel_residual(w) < 1e-9


def _wander_trail(second_prey):
    """사냥감 하나(0번 칸)를 move_animals만으로 1,200스텝 움직이고 자취를 돌려준다.
    second_prey면 1번 칸에 다른 사냥감을 하나 더 둔다(0번 칸의 자취가 다른 칸의 상태에 기대지 않는지 본다)."""
    w = World(small("ext", prey_enabled=True, init_agents=0, min_agents=0, food_inflow=0.0, n_prey=2,
                    prey_stay=400))
    s, fa, ia, cfg = w.s, w.fa, w.ia, w.cfg
    W, H = cfg.world_w, cfg.world_h
    for _ in range(2 if second_prey else 1):
        s.budget[2] = fa[an.AF_MEAT_E]
        assert an.spawn_prey(0, s, fa, ia) == 1
    assert s.an_alive[0] and s.an_alive[1] == second_prey
    gv = cfg.prey_graze_speed * cfg.v_max
    tv = cfg.prey_travel_speed * cfg.v_max
    trail, targets = [], set()
    grazed = traveled = 0
    for t in range(1, 1201):
        x0, y0, tx, ty, timer = s.an_x[0], s.an_y[0], s.an_tx[0], s.an_ty[0], s.an_timer[0]
        an.move_animals(t, s, fa, ia)
        step_len = _wdist(s.an_x[0], s.an_y[0], x0, y0, W, H)
        if timer > 0:
            grazed += 1
            assert step_len <= gv + 1e-12, t
            # 가두기가 없으면 400스텝 맴도는 동안 40까지 벗어날 수 있다
            assert _wdist(s.an_x[0], s.an_y[0], tx, ty, W, H) <= cfg.prey_graze_r + gv, t
        else:
            traveled += 1
            assert step_len == pytest.approx(tv, rel=1e-9), t
        assert 0.0 <= s.an_x[0] < W and 0.0 <= s.an_y[0] < H
        targets.add((tx, ty))
        trail.append((s.an_x[0], s.an_y[0], s.an_heading[0]))
    assert grazed >= cfg.prey_stay and traveled > 0 and len(targets) >= 2
    return np.array(trail)


def test_prey_grazes_then_travels():
    a = _wander_trail(second_prey=False)
    b = _wander_trail(second_prey=True)
    assert np.array_equal(a, b)          # 난수 주소가 (칸, 스텝)뿐이라 다른 칸이 있어도 비트까지 같다


def test_pick_target_prefers_far_patch_and_falls_back_to_uniform():
    cfg = small("ext", prey_enabled=True)
    fa, ia = an.pack_animal_params(cfg)
    W, H = cfg.world_w, cfg.world_h
    for b0 in range(0, 60, 3):
        cx, cy = k.patch_center(cfg.seed, 0, 7, cfg.n_patches, cfg.patch_life, W, H)
        tx, ty = an._pick_target(cfg.seed, 20, 3, b0, cx, cy, 7, fa, ia)
        assert _wdist(tx, ty, cx, cy, W, H) > 2 * cfg.prey_graze_r   # 지금 자리에 붙은 먹이 자리는 고르지 않는다
    ia0 = ia.copy()
    ia0[an.AI_N_PATCH] = 0
    tx, ty = an._pick_target(cfg.seed, 20, 3, 0, 1.0, 1.0, 7, fa, ia0)
    assert 0.0 <= tx < W and 0.0 <= ty < H and (tx, ty) != (1.0, 1.0)


def test_prey_dies_of_age_into_lost():
    w = _prey_world(init_agents=0, min_agents=0, food_inflow=0.0, prey_life=30)
    s = w.s
    s.budget[2] = _meat_e(w.cfg)         # 첫 스텝에 한 마리가 태어나게 한다(장부에는 태어날 때 L_SPAWN으로 들어간다)
    w.run(30)                            # 스텝 0에 태어나 0..29를 산다
    assert _n_prey(s) == 1 and s.an_born[s.an_alive][0] == 0
    lost = s.ledger[k.L_LOST]
    w.step()                             # 스텝 30: 수명이 다해 사라진다
    assert _n_prey(s) == 0
    assert s.ledger[k.L_LOST] - lost == pytest.approx(_meat_e(w.cfg), rel=1e-12)
    assert w.energy_balance()["animals"] == 0.0
    assert rel_residual(w) < 1e-9


def test_prey_toggle_off_clears_and_on_again():
    base = dict(init_agents=0, min_agents=0)
    w = World(small("ext", prey_enabled=True, carry_enabled=True, **base))
    w.run(300)
    s = w.s
    assert _n_prey(s) == 4 and s.budget[2] > 0 and s.counters[k.C_NFIELD] > 0
    e_an = float(s.an_energy[s.an_alive].sum())
    b0, b2, lost = float(s.budget[0]), float(s.budget[2]), float(s.ledger[k.L_LOST])
    w.set_config(small("ext", **base))
    s = w.s
    assert not s.an_alive.any()
    assert s.ledger[k.L_LOST] == pytest.approx(lost + e_an, rel=1e-12)
    assert s.budget[2] == 0.0 and s.budget[0] == b0 + b2
    assert s.counters[k.C_NFIELD] == 0
    w.run(100)
    assert not s.an_alive.any() and s.budget[2] == 0.0
    assert rel_residual(w) < 1e-9
    w.set_config(small("ext", prey_enabled=True, **base))
    w.run(300)
    assert _n_prey(w.s) == 4
    assert rel_residual(w) < 1e-9


def test_clear_animals_returns_energy_into_lost():
    w = _prey_world(init_agents=0, min_agents=0)
    w.run(300)
    s = w.s
    e_an = float(s.an_energy[s.an_alive].sum())
    lost = float(s.ledger[k.L_LOST])
    assert w.clear_animals(an.ANIMAL_PRED) == 0.0 and _n_prey(s) == 4   # 다른 종류는 건드리지 않는다
    assert w.clear_animals(an.ANIMAL_PREY) == pytest.approx(e_an, rel=1e-12)
    assert _n_prey(s) == 0 and s.ledger[k.L_LOST] == pytest.approx(lost + e_an, rel=1e-12)
    assert rel_residual(w) < 1e-9


def test_animal_fields_enter_hash_only_when_alive(monkeypatch):
    w = _prey_world(init_agents=0, min_agents=0); w.run(200)
    h = w.state_hash()
    a = int(np.flatnonzero(w.s.an_alive)[0])
    w.s.an_hp[a] -= 1.0
    assert w.state_hash() != h
    # 기본값 규칙: HASH_DEFAULTS에 동물 필드 기본값이 있고 산 동물의 값이 모두 그 값이면 그 필드는 빠진다
    alive = w.s.an_alive
    w.s.an_timer[alive] = 7
    monkeypatch.setattr(k, "ANIMAL_SAVED", tuple(n for n in k.ANIMAL_SAVED if n != "an_timer"))
    h_without = w.state_hash()
    monkeypatch.undo()
    assert w.state_hash() != h_without
    monkeypatch.setitem(k.HASH_DEFAULTS, "an_timer", 7)
    assert w.state_hash() == h_without
    w.s.an_timer[a] = 8
    assert w.state_hash() != h_without   # 기본값이 아닌 칸이 하나라도 있으면 들어간다
    monkeypatch.undo()
    w.s.an_alive[:] = False
    h2 = w.state_hash()
    w.s.an_hp[:] = 123.0
    assert w.state_hash() == h2


def test_old_checkpoint_without_animal_fields_loads_exact(tmp_path):
    # 확장 세계를 저장한 뒤 체크포인트에서 동물 필드를 모두 빼고 읽는다 →
    # 빈 동물 배열로 채워지고 load_arrays가 True, 저장 해시 검증 통과
    w = World(small("harsh")); w.run(60)
    path = ckpt.save(w, tmp_path)
    with np.load(path) as z:
        arrays = {n: z[n] for n in z.files}
    for name in k.ANIMAL_SAVED:
        del arrays[name]
    np.savez(path, **arrays)
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text())
    meta["npz_sha256"] = ckpt.file_sha256(path)
    meta_path.write_text(json.dumps(meta))
    w2, _ = ckpt.load(path)                       # 저장 해시 검증을 켠 채로 읽는다
    assert not w2.s.an_alive.any()
    assert w2.state_hash() == w.state_hash()
    w3 = World(small("harsh"), populate=False)
    assert w3.load_arrays(arrays) is True
    assert not w3.s.an_alive.any()


def test_prey_world_ledger_closes_long():
    w = World(small("harsh", "stage4", "ext", prey_enabled=True))
    w.run(1500)
    w.check_sane()
    assert _n_prey(w.s) > 0
    assert rel_residual(w) < 1e-9


# ---- 사냥: 커널을 직접 부르는 손 경기장 ----
def _hunt_arena(n, **kw) -> World:
    """먹이가 나지 않고 사냥감이 저절로 태어나지 않는 세계. 개체 n마리는 에너지 30(먹지도 죽지도 않게)이고
    더한 에너지는 장부의 L_INJECT에 적는다."""
    w = World(small("ext", prey_enabled=True, n_prey=0, init_agents=n, min_agents=0, food_inflow=0.0, **kw))
    s = w.s
    a = s.alive
    s.ledger[k.L_INJECT] += float((30.0 - s.energy[a]).sum())
    s.energy[a] = 30.0
    return w


def _put_prey(w, x, y, hp=None) -> int:
    """빈 동물 칸에 사냥감 하나를 놓고 칸 번호를 돌려준다. 몸의 에너지는 장부의 L_SPAWN에 적는다."""
    s, cfg = w.s, w.cfg
    j = int(np.flatnonzero(~s.an_alive)[0])
    meat_e = _meat_e(cfg)
    s.an_alive[j] = True
    s.an_kind[j] = an.ANIMAL_PREY
    s.an_x[j], s.an_y[j], s.an_heading[j] = x, y, 0.0
    s.an_hp[j] = cfg.prey_hp if hp is None else hp
    s.an_energy[j] = meat_e
    s.an_tx[j], s.an_ty[j] = x, y
    s.an_timer[j] = 0
    s.an_rest[j] = cfg.prey_rest
    s.an_born[j] = w.step_no
    s.ledger[k.L_SPAWN] += meat_e
    return j


def _ring(w, n, cx, cy, r=2.5) -> np.ndarray:
    """산 개체 n마리를 (cx, cy) 둘레 반지름 r에 고르게 놓고 공격 출력만 1로 둔다. 칸 번호들을 돌려준다."""
    s, cfg = w.s, w.cfg
    idx = np.flatnonzero(s.alive)[:n]
    assert len(idx) == n
    for q, i in enumerate(idx):
        ang = 2.0 * math.pi * q / n
        s.x[i] = (cx + r * math.cos(ang)) % cfg.world_w
        s.y[i] = (cy + r * math.sin(ang)) % cfg.world_h
        s.act[i] = 0.0
        s.act[i, ACT_ATTACK] = 1.0
    return idx


def _hunt(w, a, idx, max_steps=200) -> tuple[int, np.ndarray]:
    """사냥감 a가 쓰러질 때까지 attack_animals만 부른다. (스텝 수, 개체별 잃은 에너지)를 돌려준다."""
    s = w.s
    e0 = s.energy[idx].copy()
    for t in range(max_steps):
        an.attack_animals(w.step_no + t, s, w.fa, w.ia)
        if not s.an_alive[a]:
            return t + 1, e0 - s.energy[idx]
    raise AssertionError("사냥감이 쓰러지지 않았다")


def test_hunt_cost_curve():
    want_steps = {1: 100, 2: 50, 3: 34, 4: 25, 5: 25, 6: 25}
    steps, loss = {}, {}
    for n in range(1, 7):
        w = _hunt_arena(n)
        steps[n], lost = _hunt(w, _put_prey(w, 100.0, 100.0), _ring(w, n, 100.0, 100.0))
        assert abs(steps[n] - want_steps[n]) <= 1, n
        loss[n] = float(lost.mean())
        assert lost == pytest.approx(steps[n] * (0.12 / n + 0.02), rel=1e-9)   # 반격은 쓰러지는 스텝에도 든다
        s = w.s
        assert (s.f_kind[s.f_alive] == k.KIND_MEAT).sum() == 24
        assert s.f_energy[s.f_alive].sum() == pytest.approx(24 * w.cfg.food_energy, rel=1e-12)
        assert s.counters[k.C_PREY_KILLED] == 1 and s.counters[k.C_HUNTERS] == n   # 쓰러뜨린 스텝의 공격자 수
        assert not s.an_alive.any() and w.energy_balance()["animals"] == 0.0
        assert rel_residual(w) < 1e-9
    cfg = small("ext", prey_enabled=True)
    assert loss[1] > cfg.e_max + cfg.food_energy           # 혼자는 이어 쳐서는 못 잡는다: 몸에 담을 수 있는 에너지보다 비용이 크다
    net = {n: min(24.0 / n, 6.0) - loss[n] for n in loss}  # 배 채움 6 기준
    assert net[1] < 0 < net[2] < net[3] < net[4]
    assert net[5] <= net[4] and net[6] <= net[4]           # 넷 뒤로 늘지 않는다
    assert steps[4] == steps[5] == steps[6]                # 포화


def test_meat_scatters_around_prey_and_wraps():
    w = _hunt_arena(4)
    s, cfg = w.s, w.cfg
    a = _put_prey(w, 1.0, 100.0, hp=0.5)                   # 세계 가장자리 곁: 고기 자리도 감긴다
    an.attack_animals(w.step_no, s, w.fa, w.ia)
    assert s.an_alive[a] and s.an_natt[a] == 0             # 공격 출력이 없으면 아무도 치지 않는다(쉰 사냥감은 아문다)
    assert s.an_hp[a] == pytest.approx(0.5 + cfg.prey_regen, rel=1e-12)
    _ring(w, 4, 1.0, 100.0)
    an.attack_animals(w.step_no, s, w.fa, w.ia)
    meat = s.f_alive & (s.f_kind == k.KIND_MEAT)
    assert meat.sum() == 24
    assert (s.f_born[meat] == w.step_no).all() and (s.f_holder[meat] == -1).all() and (s.f_dropped[meat] == 0).all()
    d = np.array([_wdist(x, y, 1.0, 100.0, cfg.world_w, cfg.world_h) for x, y in zip(s.fx[meat], s.fy[meat])])
    assert d.max() < 4 * cfg.prey_radius and d.mean() > 0.5 * cfg.prey_radius
    assert ((0.0 <= s.fx[meat]) & (s.fx[meat] < cfg.world_w)).all()
    assert (s.fx[meat] > cfg.world_w / 2).any()            # 가장자리 너머로 감긴 고기가 있다
    assert s.counters[k.C_NFIELD] == 0                     # 고기는 들판 알이 아니다


def test_saturation_caps_damage_per_step():
    w = _hunt_arena(6)
    s = w.s
    a = _put_prey(w, 100.0, 100.0)
    _ring(w, 6, 100.0, 100.0)
    an.attack_animals(w.step_no, s, w.fa, w.ia)
    assert s.an_natt[a] == 6 and s.an_dmg[a] == pytest.approx(6 * 0.2, rel=1e-12)
    assert s.an_hp[a] == pytest.approx(20.0 - 4 * 0.2, rel=1e-12)   # 넷 몫까지만 쌓인다


def test_wounds_heal_only_after_rest():
    w = _hunt_arena(1)
    s, fa, ia = w.s, w.fa, w.ia
    a = _put_prey(w, 100.0, 100.0)
    i = _ring(w, 1, 100.0, 100.0)[0]
    for t in range(10):
        an.attack_animals(t, s, fa, ia)
    hp10 = float(s.an_hp[a])
    assert hp10 == pytest.approx(18.0, rel=1e-12) and s.an_rest[a] == 0
    s.act[i, ACT_ATTACK] = 0.0
    for t in range(10, 109):
        an.attack_animals(t, s, fa, ia)
    assert s.an_hp[a] == hp10 and s.an_rest[a] == 99        # 99스텝 쉬어도 아물지 않는다
    an.attack_animals(109, s, fa, ia)
    assert s.an_hp[a] == pytest.approx(hp10 + 0.05, rel=1e-12) and s.an_rest[a] == 100
    for t in range(110, 2000):
        an.attack_animals(t, s, fa, ia)
    assert s.an_hp[a] == 20.0 and s.an_rest[a] == 100       # 체력은 처음 값에서, 쉰 스텝 수는 prey_rest에서 멈춘다


def test_counter_strike_split_and_touch_hit():
    for conflict in (False, True):
        w = _hunt_arena(4, conflict_enabled=conflict)
        s = w.s
        a = _put_prey(w, 100.0, 100.0)
        idx = _ring(w, 3, 100.0, 100.0)
        far = np.flatnonzero(s.alive)[3]                    # 닿는 사냥감이 없는데 공격 출력을 낸 개체
        s.x[far], s.y[far] = 200.0, 200.0
        s.act[far] = 0.0
        s.act[far, ACT_ATTACK] = 1.0
        s.touch[:] = 0.0
        lost0, move0 = float(s.ledger[k.L_LOST]), float(s.ledger[k.L_MOVE])
        an.attack_animals(w.step_no, s, w.fa, w.ia)
        cost = 0.0 if conflict else 0.02                    # 공격이 켜진 세계에서는 st.attack이 이미 매겼다
        assert s.energy[idx] == pytest.approx(30.0 - 0.04 - cost, rel=1e-12)
        assert (s.touch[idx] == k.TOUCH_HIT).all() and (s.a_hit[idx] == a).all()
        assert s.energy[far] == pytest.approx(30.0 - cost, rel=1e-12)   # 빗나가도 비용은 든다
        assert s.touch[far] == 0.0 and s.a_hit[far] == -1
        assert s.ledger[k.L_LOST] - lost0 == pytest.approx(0.12, rel=1e-9)
        assert s.ledger[k.L_MOVE] - move0 == pytest.approx(4 * cost, abs=1e-12)
        assert rel_residual(w) < 1e-9


def test_counter_strike_can_kill_as_killed():
    w = _hunt_arena(1)
    s = w.s
    a = _put_prey(w, 100.0, 100.0)
    i = _ring(w, 1, 100.0, 100.0)[0]
    _fix_out(w, i, carry=-5.0)
    s.bout[i, ACT_ATTACK] = 3.0                             # 마음이 꺼져 있으면 출력 = tanh(편향)
    s.ledger[k.L_INJECT] += 0.01 - s.energy[i]
    s.energy[i] = 0.01
    w.step()
    assert not s.alive[i]
    assert s.an_hp[a] < w.cfg.prey_hp                       # 친 뒤 반격을 맞고 죽었다
    deaths = np.concatenate(w.deaths)
    assert len(deaths) == 1 and deaths["cause"][0] == k.DEATH_KILLED
    assert s.counters[k.C_DEATH_KILLED] == 1 and s.counters[k.C_DEATH_STARVE] == 0
    assert rel_residual(w) < 1e-9                           # 음수 잔액까지 장부에 들어간다


def test_kill_with_full_store_loses_meat_to_ledger():
    w = _hunt_arena(8, max_agents=8)
    s, cfg = w.s, w.cfg
    alive = np.flatnonzero(s.alive)
    for q, i in enumerate(alive):
        hold(w, i, _place(w, x=10.0 + 3.0 * q, y=10.0))
    assert s.counters[k.C_NFOOD] - s.counters[k.C_NFIELD] == 8 == cfg.max_agents
    a = _put_prey(w, 100.0, 100.0, hp=0.5)
    _ring(w, 4, 100.0, 100.0)
    lost0 = float(s.ledger[k.L_LOST])
    an.attack_animals(w.step_no, s, w.fa, w.ia)
    assert not s.an_alive[a] and s.counters[k.C_PREY_KILLED] == 1
    assert (s.f_alive & (s.f_kind == k.KIND_MEAT)).sum() == 0
    assert s.ledger[k.L_LOST] - lost0 == pytest.approx(_meat_e(cfg) + 0.12, rel=1e-12)
    assert s.counters[k.C_NFOOD] - s.counters[k.C_NFIELD] == 8
    assert (s.a_held[alive] >= 0).all()                     # 든 물건은 밀려나지 않는다
    assert rel_residual(w) < 1e-9


def test_attack_reaches_across_wrap():
    w = _hunt_arena(1)
    s, cfg = w.s, w.cfg
    a = _put_prey(w, 1.0, 1.0)
    i = _ring(w, 1, 1.0, 1.0)[0]
    s.x[i], s.y[i] = cfg.world_w - 1.0, 1.0                 # 감긴 거리 2
    an.attack_animals(w.step_no, s, w.fa, w.ia)
    assert s.a_hit[i] == a
    assert s.an_hp[a] == pytest.approx(19.8, rel=1e-12)


def _camp(winter: bool) -> float:
    """혼자 사냥감을 쓰러뜨린 뒤 고기 곁에 머물며 먹는 개체가 먹은 양. winter면 겨울 첫 스텝에 잡은 꼴이다."""
    kw = dict(spoil_enabled=True, season_enabled=True) if winter else dict(spoil_enabled=True)
    w = _hunt_arena(1, **kw)
    s, cfg = w.s, w.cfg
    if winter:
        w.step_no = cfg.season_period - cfg.winter_len
        assert w.in_winter(w.step_no) and not w.in_winter(w.step_no - 1)
    i = int(np.flatnonzero(s.alive)[0])
    s.age[i] = 0
    s.ledger[k.L_INJECT] += 6.0 - s.energy[i]
    s.energy[i] = 6.0                                       # 혼자 14를 치른 뒤
    _fix_out(w, i, carry=-5.0)                              # 멈춰 세움. 공격 편향 0이라 출력 0 → 치지 않는다
    x0, y0 = float(s.x[i]), float(s.y[i])
    for q in range(24):
        ang = 2.0 * math.pi * q / 24
        r = (q + 1) / 24.0
        _place(w, e=cfg.food_energy, born=w.step_no, kind=k.KIND_MEAT,
               x=(x0 + r * math.cos(ang)) % cfg.world_w, y=(y0 + r * math.sin(ang)) % cfg.world_h)
    for _ in range(3000):
        if not (s.f_alive & (s.f_kind == k.KIND_MEAT)).any():
            break
        w.step()
    assert not (s.f_alive & (s.f_kind == k.KIND_MEAT)).any()
    assert rel_residual(w) < 1e-9
    return float(s.eaten[i])


def test_lone_hunter_camping_at_meat():
    summer, winter = _camp(False), _camp(True)
    assert summer < 14.0                                   # 여름에 혼자 머물러 먹어도 비용을 못 메운다
    assert winter > summer                                 # 겨울엔 고기가 늦게 상해 더 먹는다
    assert winter < 14.0                                   # 겨울에도 혼자는 손해다. 셈으로는 약 12.6이라 여유가 작다


def _hunting_world(prey: bool):
    """200스텝 뒤 산 개체 모두의 공격 편향을 +3으로 두고, 산 사냥감마다 산 개체 하나의 자리로 옮겨 600스텝 돌린다.
    매 스텝 센 고기 알 수를 함께 돌려준다."""
    kw = dict(prey_enabled=True) if prey else {}
    w = World(small("ext", prey_hp=2.0, **kw))
    w.run(200)
    s = w.s
    s.bout[s.alive, ACT_ATTACK] = 3.0
    ag = np.flatnonzero(s.alive)
    for q, j in enumerate(np.flatnonzero(s.an_alive)):
        s.an_x[j], s.an_y[j] = s.x[ag[q]], s.y[ag[q]]
    meat = []
    for _ in range(600):
        w.step()
        meat.append(int((s.f_alive & (s.f_kind == k.KIND_MEAT)).sum()))
    return w, meat


def test_hunting_world_ledger_closes():
    w, meat = _hunting_world(True)
    s = w.s
    w.check_sane()
    assert s.counters[k.C_PREY_KILLED] > 0
    assert s.counters[k.C_HUNTERS] >= s.counters[k.C_PREY_KILLED]
    assert max(meat) > meat[-1]                            # 부패가 없으니 줄어든 것은 먹힌 것
    assert rel_residual(w) < 1e-9
    off, meat_off = _hunting_world(False)                  # 같은 세계에서 사냥감만 끈 갈래
    assert max(meat_off) == 0 and rel_residual(off) < 1e-9
    assert s.ledger[k.L_LOST] != off.s.ledger[k.L_LOST]


# ---- 사냥감 감각, 동물 장치를 새로 켜는 갈래·다시 보기, 요약, 해시 ----
def _sense_one(prey_xy, heading=0.0, at=(100.0, 100.0)):
    """개체 하나를 at에 머리 heading으로 두고, 사냥감들을 prey_xy에 둔 세계에서 사냥감 감각 두 칸을 돌려준다."""
    w = _hunt_arena(1)
    s, cfg = w.s, w.cfg
    i = int(np.flatnonzero(s.alive)[0])
    s.x[i], s.y[i], s.heading[i] = at[0], at[1], heading
    for x, y in prey_xy:
        _put_prey(w, x, y)
    col = cfg.ext_in0 + EXT_PREY
    s.obs[i, col:col + 2] = 9.0                            # 없으면 0으로 덮어쓰는지도 본다
    an.sense_animals(s.alive, s.x, s.y, s.heading, s.obs, col, s.an_alive, s.an_kind, s.an_x, s.an_y,
                     an.ANIMAL_PREY, cfg.prey_radius, cfg.vision_range, cfg.vision_fov, cfg.world_w, cfg.world_h)
    return tuple(float(v) for v in s.obs[i, col:col + 2])


def test_prey_direction_input():
    c = 1.0 - (8.0 - 2.0) / 16.0                           # 0.625
    assert _sense_one([(108.0, 100.0)]) == pytest.approx((c, 0.0), abs=1e-7)
    a30 = math.radians(30.0)
    got = _sense_one([(100.0 + 8.0 * math.cos(a30), 100.0 + 8.0 * math.sin(a30))])
    assert got == pytest.approx((c * math.cos(a30), c * math.sin(a30)), abs=1e-6)
    assert _sense_one([(100.0, 108.0)]) == (0.0, 0.0)      # 90도, 시야(120도) 밖
    assert _sense_one([(92.0, 100.0)]) == (0.0, 0.0)       # 뒤
    assert _sense_one([(120.0, 100.0)]) == (0.0, 0.0)      # 틈 18 > 16
    assert _sense_one([(112.0, 100.0), (108.0, 100.0)]) == pytest.approx((c, 0.0), abs=1e-7)   # 가까운 것
    # 시야 끝에 걸친 몸은 보인다: 중심은 61도(시야 밖)지만 반지름 2가 시야 안으로 들어온다
    a61 = math.radians(61.0)
    edge = _sense_one([(100.0 + 8.0 * math.cos(a61), 100.0 + 8.0 * math.sin(a61))])
    assert edge[0] == pytest.approx(c * math.cos(a61), abs=1e-6) and edge[1] > 0.0
    # 머리 방향 기준: 머리가 90도면 (100, 108)이 바로 앞
    assert _sense_one([(100.0, 108.0)], heading=math.pi / 2) == pytest.approx((c, 0.0), abs=1e-6)
    # 감긴 거리: (252, 100)에서 보면 (4, 100)은 가장자리 너머 앞 8이다
    assert _sense_one([(4.0, 100.0)], at=(252.0, 100.0)) == pytest.approx((c, 0.0), abs=1e-6)


def test_world_step_writes_prey_sense():
    w = _hunt_arena(1)
    s, cfg = w.s, w.cfg
    i = int(np.flatnonzero(s.alive)[0])
    s.x[i], s.y[i], s.heading[i] = 100.0, 100.0, 0.0
    _put_prey(w, 108.0, 100.0)
    w.step()
    col = cfg.ext_in0 + EXT_PREY
    assert float(s.obs[i, col]) == pytest.approx(0.625) and float(s.obs[i, col + 1]) == pytest.approx(0.0, abs=1e-7)
    off = World(small("ext", n_prey=0, init_agents=1, min_agents=0, food_inflow=0.0))
    off.step()
    assert not off.s.obs[:, col:col + 2].any()             # 사냥감이 꺼지면 두 칸은 0


def test_attack_newly_on():
    base, ext = small(), small("ext")
    assert attack_newly_on(ext, small("ext", prey_enabled=True))
    assert attack_newly_on(base, small("conflict"))
    assert not attack_newly_on(small("ext", prey_enabled=True), small("ext", prey_enabled=True))
    assert not attack_newly_on(base, ext)
    assert not attack_newly_on(small("ext", prey_enabled=True), ext)    # 끄는 쪽은 아니다


def _drifted_main(tmp_path):
    """기본 크기 세계. 공격이 꺼진 채 떠돈 공격 편향을 흉내 내고 400스텝 돌려 저장한다."""
    r = Runner.new(tmp_path / "main", small())
    r.world.s.bout[:, ACT_ATTACK] = 0.7
    r.loop(400, quiet=True)
    r.close()


def test_branch_into_harsh2_calms_attack(tmp_path):
    _drifted_main(tmp_path)
    r = Runner.branch(tmp_path / "main", tmp_path / "h2", 400, ["preset=harsh2"], False, False)
    c, s = r.world.cfg, r.world.s
    assert c.prey_enabled and c.huddle_enabled and c.carry_enabled and c.season_enabled
    assert (s.bout[s.alive, ACT_ATTACK] == -1.0).all() and "bout[ACT_ATTACK]=-1" in r.meta["genome_edits"]
    meta = json.loads((tmp_path / "h2" / "run.json").read_text())
    assert "bout[ACT_ATTACK]=-1" in meta["genome_edits"]
    r.loop(300, quiet=True); r.close()
    assert rel_residual(r.world) < 1e-9
    again = Runner.resume(tmp_path / "h2")
    assert again.world.state_hash() == r.world.state_hash(); again.close()
    plain = Runner.branch(tmp_path / "main", tmp_path / "ext", 400, ["preset=ext"], False, False)
    assert (plain.world.s.bout[plain.world.s.alive, ACT_ATTACK] > 0).any()
    assert "bout[ACT_ATTACK]=-1" not in plain.meta["genome_edits"]
    plain.close()
    ctrl = Runner.branch(tmp_path / "main", tmp_path / "ctrl", 400, ["preset=ext"], False, False, calm_attack=True)
    assert (ctrl.world.s.bout[ctrl.world.s.alive, ACT_ATTACK] == -1.0).all()
    ctrl.close()


def test_replay_record_calms_attack_for_animal_override(tmp_path, monkeypatch):
    _drifted_main(tmp_path)
    calls = []
    orig = World.calm_attack

    def counted(self):
        calls.append(1)
        return orig(self)

    monkeypatch.setattr(World, "calm_attack", counted)
    path = dict(ckpt.list_checkpoints(tmp_path / "main" / "checkpoints"))[400]
    rep = replay.record(path, steps=10, overrides=["preset=harsh2"])
    assert len(calls) == 1 and any("공격 출력 편향" in n for n in rep.meta["notes"])
    rep = replay.record(path, steps=10)
    assert len(calls) == 1 and not any("공격 출력 편향" in n for n in rep.meta["notes"])
    replay.record(path, steps=10, overrides=["preset=ext"])
    assert len(calls) == 1


def _thread_world():
    """사냥이 일어나는 세계(뭉쳐 자기·겨울을 짧게 켬). 100스텝 뒤 공격 편향을 +3으로 두고 사냥감을 개체 자리로 옮겨
    200스텝 더 돌린다. 개체가 곧 흩어지므로 사냥감 체력을 세 번 치면 쓰러지게 낮춘다."""
    w = World(small("ext", prey_hp=0.5, prey_enabled=True, season_enabled=True, season_period=200, winter_len=60,
                    huddle_enabled=True))
    w.run(100)
    s = w.s
    s.bout[s.alive, ACT_ATTACK] = 3.0
    ag = np.flatnonzero(s.alive)
    for q, j in enumerate(np.flatnonzero(s.an_alive)):
        s.an_x[j], s.an_y[j] = s.x[ag[q]], s.y[ag[q]]
    w.run(200)
    return w


def test_prey_world_does_not_depend_on_thread_count():
    import numba
    n0 = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        a = _thread_world()
        numba.set_num_threads(max(2, min(4, n0)))
        b = _thread_world()
    finally:
        numba.set_num_threads(n0)
    assert a.s.counters[k.C_PREY_KILLED] > 0
    assert a.state_hash() == b.state_hash()


def test_summary_reports_prey():
    w = World(small("harsh2"))
    w.run(200)
    out = w.summary()
    assert out["prey_alive"] == int(w.s.an_alive.sum()) and out["prey_alive"] > 0
    assert out["prey_killed"] == int(w.s.counters[k.C_PREY_KILLED])
    assert out["prey_hunters"] == int(w.s.counters[k.C_HUNTERS])
    assert out["deaths_killed"] == int(w.s.counters[k.C_DEATH_KILLED]) and "huddle_cut_mean" in out
    w.s.alive[:] = False                                   # 개체가 모두 없어도 키는 남는다
    assert "prey_alive" in w.summary() and "huddle_cut_mean" in w.summary()
    assert "prey_alive" not in World(small("harsh")).summary()


# 이 커밋의 코드가 낸 300스텝 해시. 포식자·약탈자를 더한 뒤에도 그것들을 끈 이 설정은 같아야 한다.
PREY_HASH_CFG = dict(season_period=200, winter_len=60, autumn_len=40, autumn_mult=2.0,
                     carry_enabled=True, prey_enabled=True, huddle_enabled=True)
PINNED_PREY_HASH_300 = "7547f8c3959541e0061604453f66f452bb1b018b464cc480cdd45072f2a1f6d3"


def test_prey_world_keeps_pinned_hash():
    w = World(small("harsh", **PREY_HASH_CFG))
    w.run(300)
    assert int(w.s.an_alive.sum()) > 0                     # 사냥감이 해시에 들어간다. 300스텝 안에 겨울(140~199)도 있다
    assert w.state_hash() == PINNED_PREY_HASH_300
