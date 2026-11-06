"""신호 판정 측정(설계 docs/superpowers/specs/2026-11-07-signal-measure-design.md):
섞기 막기(knockout mode=shuffle)와 상호정보량(knockout info)."""
import numpy as np
import pytest

from kkumteul import kernels as k
from kkumteul import knockout as ko
from kkumteul import stages as st
from kkumteul.world import World

from test_knockout import _facing_pair, small


# ---- 상호정보량 추정 ----
def test_mi_binned_independent_near_zero_dependent_large():
    rng = np.random.default_rng(0)
    x = rng.normal(size=20000)
    y = rng.normal(size=20000)
    assert ko.mi_binned(x, y, bins=8) < 0.01
    assert ko.mi_binned(x, x + 0.1 * rng.normal(size=20000), bins=8) > 1.0
    # 굽은 관계(상관은 0)도 잡는다
    assert abs(np.corrcoef(x, x * x)[0, 1]) < 0.05
    assert ko.mi_binned(x, x * x, bins=8) > 0.5


def test_mi_binned_constant_is_zero():
    x = np.zeros(1000)
    y = np.random.default_rng(1).normal(size=1000)
    assert ko.mi_binned(x, y, bins=8) == 0.0


def _groups(n_ind, per, rng):
    """개체 n_ind마리가 각각 표본 per개(이어진 스텝)를 낸 꼴: 개체 id, 스텝 번호."""
    ids = np.repeat(np.arange(n_ind), per)
    steps = np.tile(np.arange(per), n_ind)
    return ids, steps


def test_mi_gain_null_interval_contains_zero():
    rng = np.random.default_rng(2)
    ids, steps = _groups(200, 30, rng)
    x = rng.normal(size=len(ids))
    y = rng.normal(size=len(ids))
    r = ko.mi_gain(x, y, ids, steps, bins=8, folds=10, n_null=5, seed=3)
    assert r["lo"] <= 0.0 <= r["hi"] or abs(r["mean"]) < 0.005
    assert not r["above_zero"]


def test_mi_gain_dependent_above_zero():
    rng = np.random.default_rng(4)
    ids, steps = _groups(200, 30, rng)
    x = rng.normal(size=len(ids))
    y = np.tanh(x) + 0.5 * rng.normal(size=len(ids))
    r = ko.mi_gain(x, y, ids, steps, bins=8, folds=10, n_null=5, seed=5)
    assert r["above_zero"] and r["lo"] > 0.05


def test_mi_gain_fixed_tags_on_persistent_state_not_inflated():
    """신호가 개체마다 고정된 꼬리표이고 상황도 개체마다 거의 고정이지만 둘이 무관하면,
    같은 스텝 안에서 섞는 영가설은 반복된 같은 짝 때문에 부풀 수 있다 — 겹(개체 단위) 구간이 그걸 막아야 한다."""
    rng = np.random.default_rng(6)
    n_ind, per = 120, 40
    ids, steps = _groups(n_ind, per, rng)
    tag = rng.normal(size=n_ind)[ids]
    state = rng.normal(size=n_ind)[ids] + 0.05 * rng.normal(size=len(ids))
    r = ko.mi_gain(tag, state, ids, steps, bins=8, folds=10, n_null=5, seed=7)
    assert not r["above_zero"], r


# ---- 섞기 막기 ----
def test_shuffle_permutes_rows_within_tagged_side_only():
    cfg = small("stage3", init_agents=40)
    w = World(cfg)
    s = w.s
    live = np.flatnonzero(s.alive)
    s.tag[live[::2]] = 0
    s.tag[live[1::2]] = 1
    hear = ko.columns(cfg, "hear")
    rng = np.random.default_rng(0)
    s.obs[np.ix_(live, hear)] = rng.normal(size=(len(live), len(hear)))
    before = s.obs.copy()
    a = live[s.tag[live] == 0]
    b = live[s.tag[live] == 1]
    ko._shuffle_rows(s, np.asarray(hear), 0, np.random.default_rng(1))
    assert np.array_equal(s.obs[b], before[b]), "다른 쪽은 그대로"
    # A의 행들은 A의 원래 행들의 재배열(행째로, 칸끼리 짝 유지)
    got = {tuple(r) for r in s.obs[np.ix_(a, hear)]}
    want = {tuple(r) for r in before[np.ix_(a, hear)]}
    assert got == want
    assert not np.array_equal(s.obs[np.ix_(a, hear)], before[np.ix_(a, hear)]), "실제로 섞였다"
    other = [c for c in range(cfg.n_in) if c not in hear]
    assert np.array_equal(s.obs[np.ix_(a, other)], before[np.ix_(a, other)]), "다른 칸은 그대로"


def test_shuffle_mode_through_social_sense_keeps_heard_values():
    """섞기 블록 안에서 사회 감각을 돌리면 A가 들은 값들의 모음은 그대로이고 짝만 바뀐다."""
    cfg = small("stage3")
    w, i, j = _facing_pair(cfg)
    s = w.s
    s.tag[i], s.tag[j] = 0, 0
    hear = ko.columns(cfg, "hear")
    st.social_sense(s, w.fs, w.isp)
    plain = {tuple(s.obs[r, hear]) for r in (i, j)}
    with ko.knockout(cfg, "hear", tag=0, mode="shuffle", seed=0):
        st.social_sense(s, w.fs, w.isp)
    assert {tuple(s.obs[r, hear]) for r in (i, j)} == plain


def test_shuffle_mode_of_a_sense_that_is_off_changes_nothing():
    cfg = small("stage2")
    w1, w2 = World(cfg), World(cfg)
    w1.s.tag[:] = 0
    w2.s.tag[:] = 0
    w1.run(20)
    with ko.knockout(cfg, "hear", tag=0, mode="shuffle", seed=0):
        w2.run(20)
    assert w1.state_hash() == w2.state_hash()


def test_shuffle_mode_does_not_touch_world_rng():
    """섞기는 지역 난수만 쓴다: 막은 칸이 행동에 안 닿는 세계에서 해시가 같아야 한다(듣기 가중치 0)."""
    cfg = small("stage3", init_agents=40)
    w1, w2 = World(cfg), World(cfg)
    for w in (w1, w2):
        hear = ko.columns(cfg, "hear")
        w.s.Win[:, hear, :] = 0.0
        w.s.tag[:] = 0
    w1.run(15)
    with ko.knockout(cfg, "hear", tag=0, mode="shuffle", seed=0):
        w2.run(15)
    assert w1.state_hash() == w2.state_hash()


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        with ko.knockout(small("stage3"), "hear", tag=0, mode="blur"):
            pass
