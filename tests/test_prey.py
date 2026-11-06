"""큰 사냥감 시험: 동물 배열, 유입 몫에서 태어나기, 떠돌기·수명, 끄면 정리, 동물 에너지 장부.

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

from test_stages import rel_residual, small  # noqa: E402
from kkumteul import animals as an  # noqa: E402
from kkumteul import checkpoint as ckpt  # noqa: E402
from kkumteul import kernels as k  # noqa: E402
from kkumteul import scenario  # noqa: E402
from kkumteul.world import World  # noqa: E402


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
