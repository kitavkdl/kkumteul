"""한쪽 감각 막기(knockout): 겨루기에서 한 무리만 어떤 입력을 못 받게 한다."""
import numpy as np
import pytest

from kkumteul import kernels as k
from kkumteul import knockout as ko
from kkumteul import stages as st
from kkumteul.config import IN_HEAR, IN_KIN, IN_SAWEAT, IN_TRAIL, N_INTERNAL, Config
from kkumteul.world import World


def small(*presets, **kw) -> Config:
    base = dict(world_w=256.0, world_h=256.0, max_agents=300, init_agents=20, food_cap=300,
                food_inflow=2.2, n_patches=4, min_agents=0, seed=5)
    base.update(kw)
    cfg = Config.from_dict({**Config().to_dict(), **base})
    return cfg.with_overrides([f"preset={p}" for p in presets])


def test_channel_columns():
    cfg = small("stage4")
    base = 2 * cfg.vision_sectors + N_INTERNAL
    assert ko.columns(cfg, "hear") == list(range(base + IN_HEAR, base + IN_HEAR + 4))
    assert ko.columns(cfg, "kin") == [base + IN_KIN]
    assert ko.columns(cfg, "saweat") == [base + IN_SAWEAT]
    assert ko.columns(cfg, "trail") == list(range(base + IN_TRAIL, base + IN_TRAIL + 3))
    assert ko.columns(cfg, "social") == ko.columns(cfg, "hear") + [base + IN_KIN, base + IN_SAWEAT]
    assert ko.columns(cfg, "vision") == list(range(2 * cfg.vision_sectors))
    with pytest.raises(ValueError):
        ko.columns(cfg, "smellovision")


def _facing_pair(cfg):
    w = World(cfg)
    s = w.s
    i, j = np.flatnonzero(s.alive)[:2]
    s.alive[np.flatnonzero(s.alive)[2:]] = False
    s.x[i], s.y[i], s.heading[i] = 100.0, 100.0, 0.0
    s.x[j], s.y[j], s.heading[j] = 110.0, 100.0, np.pi
    s.sig[i] = s.sig[j] = (1.0, -1.0)
    gw, gh = w.cfg.grid_w, w.cfg.grid_h
    k.build_grid(s.x, s.y, s.alive, 1.0 / w.cfg.cell, gw, gh, s.a_start, s.a_items, s.cursor)
    return w, i, j


def test_knocked_out_side_hears_nothing_other_side_hears():
    cfg = small("stage3")
    w, i, j = _facing_pair(cfg)
    s = w.s
    s.tag[i], s.tag[j] = 0, 1
    hear = ko.columns(cfg, "hear")
    with ko.knockout(cfg, "hear", tag=0):
        st.social_sense(s, w.fs, w.isp)
    assert np.all(s.obs[i, hear] == 0.0), "막은 쪽(tag 0)은 아무것도 못 듣는다"
    assert np.abs(s.obs[j, hear]).max() > 0.3, "다른 쪽은 그대로 듣는다"
    kin = ko.columns(cfg, "kin")[0]
    assert s.obs[i, kin] != 0.0, "듣기만 막았으니 닮음 감각은 남는다"


def test_patch_is_undone_and_nothing_changes_outside():
    cfg = small("stage3")
    orig = (k.sense, st.social_sense, st.trail_sense)
    with ko.knockout(cfg, "hear", tag=0):
        assert st.social_sense is not orig[1]
    assert (k.sense, st.social_sense, st.trail_sense) == orig
    w, i, j = _facing_pair(cfg)
    w.s.tag[i] = 0
    st.social_sense(w.s, w.fs, w.isp)
    assert np.abs(w.s.obs[i, ko.columns(cfg, "hear")]).max() > 0.3


def test_vision_knockout_through_a_whole_step():
    """세계 한 스텝을 통째로 돌려도(감각 → 생각) 막은 쪽의 시야 입력은 0이다."""
    cfg = small("stage4", init_agents=40)
    w = World(cfg)
    s = w.s
    live = np.flatnonzero(s.alive)
    s.tag[live[::2]] = 0
    s.tag[live[1::2]] = 1
    vis = ko.columns(cfg, "vision")
    with ko.knockout(cfg, "vision", tag=0):
        w.run(3)
    live = np.flatnonzero(s.alive)
    a = live[s.tag[live] == 0]
    b = live[s.tag[live] == 1]
    assert np.all(s.obs[np.ix_(a, vis)] == 0.0)
    assert np.abs(s.obs[np.ix_(b, vis)]).sum() > 0.0


def test_knockout_of_a_sense_that_is_off_changes_nothing():
    """규칙에 없는 감각(사회 감각이 꺼진 세계의 듣기)을 막아도 세계가 똑같이 흘러간다."""
    cfg = small("stage2")
    assert not cfg.social_enabled
    w1, w2 = World(cfg), World(cfg)
    w1.s.tag[:] = 0
    w2.s.tag[:] = 0
    w1.run(20)
    with ko.knockout(cfg, "hear", tag=0):
        w2.run(20)
    assert w1.state_hash() == w2.state_hash()
