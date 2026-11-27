"""겨루기 쌍둥이(rb = ra)와 심은 효과 눈금자(B 태그에 먹은 양의 몇 %를 명시적 주입).
develop 2바퀴 실측 4: 중립 흔들림을 줄이고, "가를 수 없음"을 "섭취의 몇 % 미만"으로 읽을 자."""
import numpy as np
import pytest

from kkumteul import kernels as k
from kkumteul import measure
from kkumteul.config import Config
from kkumteul.world import World


def _cfg(**kw):
    base = dict(seed=3, world_w=256.0, world_h=256.0, init_agents=60, max_agents=200)
    base.update(kw)
    return Config(**base)


def _side():
    w = World(_cfg())
    w.run(50)
    return measure.side_from_world(w)


def test_twin_pairs_same_row(monkeypatch):
    side = _side()
    seen = []
    orig = measure.inject_agent

    def spy(world, genomes, row, tag, *a, **kw):
        seen.append((row, tag))
        return orig(world, genomes, row, tag, *a, **kw)
    monkeypatch.setattr(measure, "inject_agent", spy)
    measure.compete(side["cfg"], side["genomes"], side["genomes"], 12, 100, 7, sample_every=50, twin=True)
    assert len(seen) == 24
    for i in range(0, 24, 2):
        (r1, t1), (r2, t2) = seen[i], seen[i + 1]
        assert r1 == r2 and {t1, t2} == {0, 1}


def test_twin_needs_same_rows():
    side = _side()
    g = side["genomes"]
    half = {key: v[: len(v) // 2] for key, v in g.items()}
    with pytest.raises(ValueError):
        measure.compete(side["cfg"], g, half, 5, 50, 1, twin=True)


def test_plant_income_adds_fraction_of_intake_and_keeps_ledger():
    w = World(_cfg(init_agents=80))
    s = w.s
    alive = np.flatnonzero(s.alive)
    s.tag[alive[::2]] = 1
    prev_id, prev_eaten = s.ids.copy(), s.eaten.copy()
    w.run(200)
    bal0 = w.energy_balance()["residual"]
    m = s.alive & (s.tag == 1)
    same = s.ids == prev_id
    d = np.where(same, s.eaten - prev_eaten, s.eaten)[m]
    room = np.maximum(w.cfg.e_max - s.energy[m], 0.0)
    want = float(np.minimum(0.1 * np.maximum(d, 0.0), room).sum())
    e0, inj0 = float(s.energy[m].sum()), float(s.ledger[k.L_INJECT])
    added = measure.plant_income(w, 0.1, 1, prev_id, prev_eaten)
    assert want > 0
    assert added == pytest.approx(want, rel=1e-4)
    assert float(s.energy[m].sum()) - e0 == pytest.approx(added, rel=1e-4)
    assert float(s.ledger[k.L_INJECT]) - inj0 == pytest.approx(added, rel=1e-6)
    assert w.energy_balance()["residual"] == pytest.approx(bal0, abs=1e-2)
    assert (prev_id == s.ids).all() and (prev_eaten == s.eaten).all()     # 다음 호출의 기준을 지금으로
    assert measure.plant_income(w, 0.1, 1, prev_id, prev_eaten) == 0.0    # 그새 먹은 것이 없으면 0
    # e_max 넘게는 주지 않는다(원래 넘어 있던 개체는 0)
    full = np.flatnonzero(s.alive & (s.tag == 1))[:3]
    s.energy[full] = w.cfg.e_max + 0.5
    s.eaten[full] += 1.0
    e_full = s.energy[full].copy()
    measure.plant_income(w, 0.5, 1, prev_id, prev_eaten)
    assert (s.energy[full] == e_full).all()


def test_plant_zero_is_unchanged_and_big_plant_wins():
    side = _side()
    cfg, g = side["cfg"], side["genomes"]
    base = measure.compete(cfg, g, g, 30, 1500, 5, sample_every=250, twin=True)
    zero = measure.compete(cfg, g, g, 30, 1500, 5, sample_every=250, twin=True, plant=0.0)
    assert base["r"] == zero["r"] and base["series"] == zero["series"]
    big = measure.compete(cfg, g, g, 30, 1500, 5, sample_every=250, twin=True, plant=1.0)
    assert big["r"] > base["r"]
    assert big["plant_injected"] > 0 and base["plant_injected"] == 0.0


def test_ruler_cells_and_equiv():
    side = _side()
    res = measure.ruler(side, fracs=(0.0, 1.0), n_each=20, steps=600, reps=2, seed=9, sample_every=200)
    assert [c["plant"] for c in res["cells"]] == [0.0, 1.0]
    assert all(len(c["r_list"]) == 2 for c in res["cells"])
    assert "r_per_unit_plant" in res
    fake = {"r_per_unit_plant": 2.0}
    assert measure.plant_equiv(0.1, fake) == pytest.approx(0.05)
    assert measure.plant_equiv(0.1, {"r_per_unit_plant": 0.0}) is None


def test_assay_twin_neutral_uses_same_genomes(monkeypatch):
    side = _side()
    calls = []
    orig = measure.compete

    def spy(cfg, ga, gb, *a, **kw):
        calls.append((ga is gb, kw.get("twin", False)))
        return orig(cfg, ga, gb, *a, **kw)
    monkeypatch.setattr(measure, "compete", spy)
    measure.assay_sides(side, side, n_each=8, steps=100, reps=2, seed=3, twin=True)
    assert calls and all(tw for _, tw in calls)
    assert all(same for same, _ in calls[2:])     # 중립(빈 막기 대조) = 같은 유전체 두 벌
