"""맛 배움 속도 유전자 시험(설계 docs/superpowers/specs/2026-11-07-individual-learning-design.md).

실제 배움 속도 = clamp(taste_rate + taste_g, 0, 1). taste_g는 개체마다 물려받는 유전자(설정값에서 벗어난 몫).
장치(taste_gene_enabled)가 꺼져 있으면 taste_g는 쓰이지 않고 비용도 없다.

    python3 -m pytest -q tests/test_taste_gene.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_stages import rel_residual, small  # noqa: E402
from test_taste import _bad, _natural_child, feed, taste_world  # noqa: E402
from kkumteul import kernels as k  # noqa: E402
from kkumteul import measure  # noqa: E402
from kkumteul import rng  # noqa: E402
from kkumteul.world import World  # noqa: E402

GENE = ("taste_gene_enabled=true",)


def gene_world(**kw) -> World:
    return taste_world("taste", **{"taste_gene_enabled": True, **kw})


def lone(gene: bool, **kw) -> World:
    w = taste_world("taste", taste_gene_enabled=gene, init_agents=1, min_agents=0, food_inflow=0.0, **kw)
    w.s.energy[0] = 2.0
    return w


def slot(kind) -> int:
    return 1 if kind == k.KIND_B else 0


# ---- 설정 ----
def test_config_defaults_and_checks():
    c = small("taste")
    assert not c.taste_gene_enabled and c.taste_gene_sigma > 0 and c.taste_cost > 0
    assert small("taste", taste_gene_enabled=True).taste_gene_enabled
    with pytest.raises(ValueError):
        small("taste_size", taste_gene_enabled=True)        # 맛 기억이 꺼져 있으면 뜻이 없다
    with pytest.raises(ValueError):
        small("taste", taste_gene_sigma=-0.1)
    with pytest.raises(ValueError):
        small("taste", taste_cost=-0.1)


def test_new_rng_stream_appended():
    assert rng.S_TASTE_GENE == 31
    others = [v for kk, v in vars(rng).items() if kk.startswith("S_") and kk != "S_TASTE_GENE"]
    assert max(others) < 31


# ---- 먹기: 실제 배움 속도 ----
def test_eat_uses_gene_rate():
    w = lone(True)
    good = 1 - _bad(w)
    w.s.taste_g[0] = -0.3                   # 0.5 - 0.3 = 0.2
    feed(w, 0, good)
    assert w.s.taste_m[0, slot(good)] == pytest.approx(0.2)


def test_eat_rate_clamped():
    w = lone(True)
    good = 1 - _bad(w)
    w.s.taste_g[0] = 0.9                    # 1.4 → 1
    feed(w, 0, good)
    assert w.s.taste_m[0, slot(good)] == pytest.approx(1.0)
    w = lone(True)
    w.s.taste_g[0] = -0.9                   # −0.4 → 0: 배우지 않는다
    feed(w, 0, good)
    assert w.s.taste_m[0, slot(good)] == 0.0


def test_gene_ignored_when_off():
    w = lone(False)
    good = 1 - _bad(w)
    w.s.taste_g[0] = -0.3
    feed(w, 0, good)
    assert w.s.taste_m[0, slot(good)] == pytest.approx(0.5)


# ---- 비용 ----
def test_cost_per_step_and_none_when_off():
    on = lone(True, taste_cost=0.01)
    off = lone(False, taste_cost=0.01)
    on.s.taste_g[0] = -0.1                  # 실제 속도 0.4 → 스텝당 0.004 더 든다
    off.s.taste_g[0] = -0.1
    for w in (on, off):
        w.s.energy[0] = 5.0
        w.run(10)
        assert w.s.alive[0]
    assert off.s.energy[0] - on.s.energy[0] == pytest.approx(10 * 0.004, rel=1e-6)


def test_cost_zero_rate_is_free():
    on = lone(True, taste_cost=0.01)
    off = lone(False)
    on.s.taste_g[0] = -0.5                  # 실제 속도 0
    for w in (on, off):
        w.s.energy[0] = 5.0
        w.run(10)
    assert on.s.energy[0] == pytest.approx(off.s.energy[0])


def test_ledger_closes_with_cost():
    # 비용이 장부에 안 들어가면 잔차가 남는다(비용을 크게 잡아 가릴 수 있게)
    w = gene_world(taste_cost=0.01)
    w.run(1500)
    assert rel_residual(w) < 1e-9


# ---- 물려주기·변이 ----
def _parent(gene: bool, g: float, **kw) -> World:
    w = taste_world("taste", taste_gene_enabled=gene, init_agents=1, food_inflow=0.0, min_agents=0, **kw)
    w.s.taste_g[0] = g
    return w


def test_child_mutates_gene():
    w = _parent(True, 0.1)
    j = _natural_child(w)
    assert w.s.taste_g[j] != pytest.approx(0.1)
    assert abs(w.s.taste_g[j] - 0.1) < 0.2  # σ 0.02의 열 배 안
    assert w.s.taste_g[0] == pytest.approx(0.1)


def test_child_copies_without_mutation():
    w = _parent(True, 0.1, mutation_enabled=False)
    assert w.s.taste_g[_natural_child(w)] == pytest.approx(0.1)
    w = _parent(False, 0.1)                 # 장치 끔: 변이 없이 베낀다(쓰이지는 않는다)
    assert w.s.taste_g[_natural_child(w)] == pytest.approx(0.1)


def test_child_gene_clamped():
    # 실제 속도가 [0, 1] 안: taste_g ∈ [−taste_rate, 1 − taste_rate]
    vals = []
    for g in (0.5, -0.5):
        for sd in range(6):
            w = _parent(True, g, taste_gene_sigma=1.0, seed=sd)
            vals.append(float(w.s.taste_g[_natural_child(w)]))
    assert min(vals) >= -0.5 - 1e-6 and max(vals) <= 0.5 + 1e-6
    assert len(set(np.round(vals, 6))) > 2


def test_reset_body_clears_gene():
    w = lone(True)
    j = int(np.flatnonzero(~w.s.alive)[0])
    w.s.taste_g[j] = 0.3
    k._reset_body(w.s, j, 999, -1, 0, 0, w.step_no, 10.0, 10.0, 0.0, 1.0)
    assert w.s.taste_g[j] == 0.0


def test_inject_agent_carries_gene():
    w = gene_world()
    w.s.taste_g[w.s.alive] = -0.2
    side = measure.side_from_world(w)
    assert "taste_g" in side["genomes"]
    j_id = measure.inject_agent(w, side["genomes"], 0, 1, 10.0, 10.0, 0.0, 0)
    j = int(np.flatnonzero(w.s.ids == j_id)[0])
    assert w.s.taste_g[j] == pytest.approx(-0.2)


# ---- 저장·해시·기록 ----
def test_off_world_gene_stays_zero_and_hash_same_as_before():
    # 장치를 끈 맛 세계: taste_g는 0으로 남는다(해시는 기존 박은 값 시험이 지킨다)
    w = taste_world("taste")
    w.run(300)
    assert (w.s.taste_g == 0).all()


def test_gene_on_with_zero_sigma_and_cost_matches_off():
    # 변이 폭 0·비용 0이면 장치를 켜도 상태가 똑같다: 새 난수 용도가 다른 흐름을 건드리지 않는다
    on = gene_world(taste_gene_sigma=0.0, taste_cost=0.0)
    off = taste_world("taste")
    on.run(400)
    off.run(400)
    assert on.state_hash() == off.state_hash()


def test_gene_world_moves_and_repeats():
    a = gene_world()
    b = gene_world()
    a.run(1500)
    b.run(1500)
    assert a.state_hash() == b.state_hash()
    assert np.abs(a.s.taste_g[a.s.alive]).max() > 0


def test_old_checkpoint_reads_zero_gene():
    w = gene_world()
    w.run(50)
    w.s.taste_g[w.s.alive] = 0.1
    arrays = {kk: v.copy() for kk, v in w.state_arrays().items()}
    del arrays["taste_g"]
    w2 = World(w.cfg, populate=False)
    w2.step_no = w.step_no
    assert w2.load_arrays(arrays)
    assert (w2.s.taste_g == 0).all()


def test_summary_rate_mean_only_when_on():
    w = gene_world()
    w.s.taste_g[w.s.alive] = -0.1
    assert w.summary()["taste_rate_mean"] == pytest.approx(0.4)
    assert "taste_rate_mean" not in taste_world("taste").summary()
