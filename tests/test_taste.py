"""맛 기억 장치 시험: 설정·묶음, 저장 필드, 끄면 지우기.

    python3 -m pytest -q tests/test_taste.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_stages import rel_residual, small  # noqa: E402,F401
from kkumteul import checkpoint as ckpt  # noqa: E402,F401
from kkumteul import kernels as k  # noqa: E402
from kkumteul import measure  # noqa: E402
from kkumteul.config import (ACT_TURN, FIRST_MEALS, N_EXT_IN, N_STAGE_IN, N_TASTE_IN, TASTE_VAL,  # noqa: E402,F401
                             TASTE_VIS, preset)
from kkumteul.world import World  # noqa: E402

# 맛 장치를 넣기 전 코드가 같은 크기(n_spare_in=36, n_spare_out=7) 세계를 300스텝 돌려 낸 해시.
# taste_size 묶음(장치 끈 같은 크기 세계)은 이 값을 그대로 내야 한다. 장치를 끈 세계의 계산을 일부러 바꿀 때만 고친다.
PINNED_TASTE_SIZE_HASH_300 = {
    (): "209278e89cb28d47c96dff4d06f9d6115de5949170ef7842c6ceb9e65486ae18",
    ("stage2", "kinds"): "cbd3d3d4d9e35ea02424d1fe18c453ea8f3b56f1aadc76c4ee39013d96bc1af2",
}


def taste_world(*presets, **kw) -> World:
    return World(small("stage2", "kinds", *presets, **kw))


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
    # taste_size 묶음 = 장치 끈 같은 크기 세계(장치를 넣기 전 코드로 뽑은 값)
    w = World(small(*presets, "taste_size"))
    w.run(300)
    assert w.state_hash() == PINNED_TASTE_SIZE_HASH_300[presets]


def test_taste_m_saved_and_old_checkpoint_reads_zero(tmp_path):
    # 저장 → 읽기 → 같은 해시. 0이 아닌 기억은 해시에 들어간다. taste_m 없는 배열 묶음을 읽으면 0
    # (장치 끈 크기 세계에서 한다: 켠 세계는 먹으면서 기억이 0이 아니게 되어 h0가 기억 0 상태가 아니다)
    w = taste_world("taste_size")
    w.run(50)
    h0 = w.state_hash()
    w.s.taste_m[w.s.alive] = 0.25
    assert w.state_hash() != h0
    arrays = {kk: v.copy() for kk, v in w.state_arrays().items()}
    w2 = World(w.cfg, populate=False)
    w2.step_no = w.step_no
    assert w2.load_arrays(arrays) and w2.state_hash() == w.state_hash()
    del arrays["taste_m"]
    w3 = World(w.cfg, populate=False)
    w3.step_no = w.step_no
    assert w3.load_arrays(arrays)           # 빠진 것으로 치지 않는다(HASH_DEFAULTS)
    assert (w3.s.taste_m == 0).all() and w3.state_hash() == h0


def test_set_config_off_clears_taste():
    w = taste_world("taste")
    w.s.taste_m[:] = 0.5
    w.set_config(w.cfg.with_overrides(["taste_enabled=false"]))
    assert (w.s.taste_m == 0).all() and w.ip[k.Q_TASTE_IN0] == -1


def test_inject_agent_starts_with_blank_taste():
    w = taste_world("taste")
    w.s.taste_m[:] = 0.5
    w.s.n_meal[:] = 7
    w.s.n_poison[:] = 3
    w.s.n_poison10[:] = 3
    side = measure.side_from_world(w)
    j_id = measure.inject_agent(w, side["genomes"], 0, 1, 10.0, 10.0, 0.0, 0)
    j = int(np.flatnonzero(w.s.ids == j_id)[0])
    assert (w.s.taste_m[j] == 0).all() and w.s.n_meal[j] == 0
    assert w.s.n_poison[j] == 0 and w.s.n_poison10[j] == 0


# ---- 먹은 순간 갱신·잊기·계수기 ----
from test_harsh import _place  # noqa: E402


def _bad(w) -> int:
    return 1 - k.good_kind(w.cfg.seed, w.step_no, w.cfg.kind_flip_period)


def feed(w, i, kind, count=1) -> None:
    """개체 i 자리에 kind 먹이 count알을 놓고 eat를 한 번 부른다(에너지는 2.0으로 다시 맞춘다)."""
    s, cfg = w.s, w.cfg
    s.energy[i] = 2.0                       # e_max에 닿아 먹지 않는 일이 없게(장부는 단언하지 않는다)
    for _ in range(count):
        _place(w, kind=kind, x=float(s.x[i]), y=float(s.y[i]))
    k.build_grid(s.fx, s.fy, s.f_alive, 1.0 / cfg.cell, cfg.grid_w, cfg.grid_h, s.f_start, s.f_items, s.cursor)
    k.eat(w.step_no, s, w.fp, w.ip)


def lone_world(*presets) -> World:
    w = taste_world(*presets, init_agents=1, min_agents=0, food_inflow=0.0)
    w.s.energy[0] = 2.0
    return w


def test_eat_updates_memory_by_seen_shape():
    # 잊기는 eat 첫머리라 eat를 부를 때마다 먼저 (1 - 0.002)배가 된다. 칸 = 1 if 종류 B else 0
    w = lone_world("taste")
    bad = _bad(w)
    good = 1 - bad
    slot = lambda kind: 1 if kind == k.KIND_B else 0  # noqa: E731
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
    w = lone_world("taste")
    w.s.taste_m[0] = (0.8, -0.6)
    w.s.energy[0] = w.cfg.e_max
    _place(w, kind=1 - _bad(w), x=float(w.s.x[0]), y=float(w.s.y[0]))
    s, cfg = w.s, w.cfg
    k.build_grid(s.fx, s.fy, s.f_alive, 1.0 / cfg.cell, cfg.grid_w, cfg.grid_h, s.f_start, s.f_items, s.cursor)
    k.eat(w.step_no, s, w.fp, w.ip)        # 배불러 먹지는 않는다
    assert s.n_meal[0] == 0 and s.ate[0] == 0.0
    assert s.taste_m[0] == pytest.approx((0.8 * 0.998, -0.6 * 0.998))


def test_forgetting_per_step():
    w = lone_world("taste")
    w.s.taste_m[0] = (0.8, -0.6)
    w.run(100)                              # 먹이 없는 세계(food_inflow=0)
    assert w.s.alive[0] and w.s.n_meal[0] == 0
    assert w.s.taste_m[0] == pytest.approx((0.8 * 0.998 ** 100, -0.6 * 0.998 ** 100), rel=1e-5)


def test_taste_off_leaves_memory_and_counts_meals():
    w = lone_world("taste_size")
    feed(w, 0, _bad(w))
    assert (w.s.taste_m == 0).all() and w.s.n_meal[0] == 1 and w.s.n_poison[0] == 1 and w.s.n_poison10[0] == 1


def test_first_ten_meals_counted_exactly():
    w = lone_world("taste")
    bad = _bad(w)
    for _ in range(9):
        feed(w, 0, 1 - bad)
    feed(w, 0, bad, count=3)               # 한 번의 eat에 독 셋(에너지는 음수가 되지만 eat만 부르므로 죽지 않는다)
    assert (w.s.n_meal[0], w.s.n_poison[0], w.s.n_poison10[0]) == (12, 3, 1)


def test_taste_eat_ledger_closes():
    w = taste_world("taste")
    w.run(1500)
    assert rel_residual(w) < 1e-9


def test_reset_body_clears_taste_and_meals():
    # 죽은 칸을 새 개체가 다시 쓸 때 앞 주인의 기억과 끼니 수가 넘어가지 않는다
    w = lone_world("taste")
    j = int(np.flatnonzero(~w.s.alive)[0])
    w.s.taste_m[j] = (0.5, -0.5)
    w.s.n_meal[j], w.s.n_poison[j], w.s.n_poison10[j] = 7, 3, 2
    k._reset_body(w.s, j, 999, -1, 0, 0, w.step_no, 10.0, 10.0, 0.0, 1.0)
    assert (w.s.taste_m[j] == 0).all() and w.s.n_meal[j] == 0
    assert w.s.n_poison[j] == 0 and w.s.n_poison10[j] == 0


# ---- 감각 맛 블록 ----
from test_harsh import _sense_once  # noqa: E402
from kkumteul.run import Runner  # noqa: E402


def _sight(kind, m, *presets, **kw):
    """혼자 있는 개체(heading 0) 왼쪽 앞에 kind 먹이 하나를 두고 감각만 부른다. (세계, obs 한 줄)."""
    w = taste_world(*(presets or ("taste",)), init_agents=1, min_agents=0, food_inflow=0.0, **kw)
    s = w.s
    assert s.alive[0]
    s.f_alive[:] = False                    # 처음 뿌린 먹이를 치운다(감각만 보므로 장부는 보지 않는다)
    s.x[0], s.y[0], s.heading[0] = 100.0, 100.0, 0.0
    _place(w, kind=kind, x=110.0, y=103.0)
    s.taste_m[0] = m
    _sense_once(w)
    return w, s.obs[0].copy()


def _seen(obs) -> int:
    """먹이 시야 칸 중 값이 가장 큰 칸의 번호(먹이가 두 칸에 걸칠 수 있다). 그 칸은 0이 아니어야 한다."""
    sec = int(np.argmax(np.abs(obs[:8])))
    assert obs[sec] != 0.0
    return sec


def test_taste_vis_reads_memory_by_shape():
    m = (-0.5, 0.3)
    w, o = _sight(k.KIND_A, m)
    t0 = w.cfg.taste_in0
    vis, tv = o[:8], o[t0 + TASTE_VIS:t0 + TASTE_VIS + 8]
    assert (vis >= 0).all() and vis[_seen(o)] > 0
    assert tv == pytest.approx(vis * -0.5)                         # 먹이 없는 칸은 0
    assert (tv[vis == 0] == 0).all() and (tv[vis > 0] < 0).all()
    assert o[t0 + TASTE_VAL:t0 + TASTE_VAL + 2] == pytest.approx(m)
    w, o = _sight(k.KIND_B, m)
    vis, tv = o[:8], o[t0 + TASTE_VIS:t0 + TASTE_VIS + 8]
    assert (vis <= 0).all() and vis[_seen(o)] < 0
    assert tv == pytest.approx(-vis * 0.3)
    assert (tv[vis == 0] == 0).all() and (tv[vis < 0] > 0).all()
    assert o[t0 + TASTE_VAL:t0 + TASTE_VAL + 2] == pytest.approx(m)
    w, o = _sight(k.KIND_A, (0.0, 0.0))     # 기억이 없으면 그 칸도 0
    assert not o[t0:t0 + N_TASTE_IN].any() and o[_seen(o)] > 0


def test_taste_vis_uses_sign_before_noise():
    m = (0.7, -0.4)
    for kind in (k.KIND_A, k.KIND_B):
        w, clean = _sight(kind, m)
        _, noisy = _sight(kind, m, "taste", vision_noise=5.0)
        t0 = w.cfg.taste_in0
        assert not np.array_equal(clean[:8], noisy[:8])            # 시야 칸은 잡음으로 달라진다
        assert np.array_equal(clean[t0:t0 + N_TASTE_IN], noisy[t0:t0 + N_TASTE_IN])
        assert clean[t0 + TASTE_VIS + _seen(clean)] != 0.0


def test_taste_block_zero_when_off():
    w, o = _sight(k.KIND_A, (0.5, 0.5), "taste_size")
    t0 = w.cfg.taste_in0
    assert o[_seen(o)] > 0 and not o[t0:t0 + N_TASTE_IN].any()
    # 켠 채 한 번 감각해 칸을 채운 뒤, 기억은 두고 장치만 끄면 다음 감각에서 칸이 0이 된다
    w, o = _sight(k.KIND_A, (0.5, 0.5))
    assert o[t0 + TASTE_VIS + _seen(o)] != 0.0 and o[t0 + TASTE_VAL] == 0.5
    w.ip[k.Q_TASTE_IN0] = -1
    _sense_once(w)
    assert (w.s.taste_m[0] == 0.5).all() and not w.s.obs[0, t0:t0 + N_TASTE_IN].any()


def test_branch_old_checkpoint_into_taste(tmp_path):
    r = Runner.new(tmp_path / "r", small("stage2", "kinds"))
    r.loop(400, quiet=True)
    r.close()
    p = ckpt.latest(tmp_path / "r" / "checkpoints")
    saved, _ = ckpt.load(p)
    assert not saved.cfg.taste_enabled
    w, _ = ckpt.load(p, saved.cfg.with_overrides(["preset=taste"]))
    t0 = w.cfg.taste_in0
    assert w.cfg.taste_enabled and w.ip[k.Q_TASTE_IN0] == t0
    assert (w.s.taste_m == 0).all() and not w.s.Win[:, t0:, :].any()
    w.run(20)                               # 먹어서 기억이 생기게 한다
    s = w.s
    m0, ids0 = s.taste_m.copy(), s.ids.copy()
    w.step()                                # 감각은 먹기 전이라 감각 칸 = 그 스텝 들어올 때의 기억
    same = s.alive & (s.ids == ids0)
    assert same.sum() >= 10 and np.abs(m0[same]).max() > 0
    assert np.array_equal(s.obs[same, t0 + TASTE_VAL:t0 + TASTE_VAL + 2], m0[same])
    w.check_sane()


# ---- 물려주기·기록 ----
# 맛 장치를 켠 세계(taste_inherit=1)를 300스텝 돌린 해시. 맛 장치의 계산을 일부러 바꿀 때만 고친다.
PINNED_TASTE_HASH_300 = "2ae2c2b913c2fbb0f11899b3f25edbaccd260d4e6cf9eb5c54efc1aded4d0b1c"

PARENT_M = (0.6, -0.4)


def _parent_world(inherit, **kw) -> World:
    """부모 하나만 산 세계(먹이 없음). 부모 기억 = PARENT_M."""
    w = taste_world("taste", taste_inherit=inherit, init_agents=1, food_inflow=0.0, **kw)
    assert w.s.alive[0] and w.s.alive.sum() == 1
    w.s.taste_m[0] = PARENT_M
    return w


@pytest.mark.parametrize("inherit", [0.0, 1.0, 0.5])
def test_natural_child_inherits(inherit):
    w = _parent_world(inherit, min_agents=0)
    s = w.s
    s.energy[0] = w.cfg.e_repro + 1.0
    s.age[0] = w.cfg.repro_min_age + 1
    s.act[0, k.ACT_REPRO] = 1.0             # 뇌가 번식을 고른 것으로 친다
    s.birth_step[0] = w.step_no - 1         # 태어난 그 스텝에는 번식하지 않는다
    assert k.process_births(w.step_no, s, w.fp, w.ip) == 1
    j = int(np.flatnonzero(s.alive & (s.parent == s.ids[0]))[0])
    assert s.taste_m[j] == pytest.approx((inherit * 0.6, inherit * -0.4))
    assert s.taste_m[0] == pytest.approx(PARENT_M)          # 부모 기억은 그대로


def test_reseed_and_passive_children_start_blank():
    # 둘 다 산 부모의 복제본으로 새끼를 넣지만(_child), 물려받기를 켜도 기억은 0에서 시작한다.
    # 새끼를 하나만 넣는다(둘째는 첫 새끼를 부모로 고를 수 있어 부모 기억이 0일 수 있다)
    w = _parent_world(1.0, min_agents=2)
    s = w.s
    assert k.reseed(w.step_no, s, w.fp, w.ip, 0) == 1
    kids = np.flatnonzero(s.alive & (s.parent == s.ids[0]))
    assert len(kids) == 1 and (s.taste_m[kids] == 0).all()
    w = _parent_world(1.0, min_agents=0, passive=True)
    s = w.s
    assert k.passive_events(w.step_no, s, w.fp, w.ip, 1, 0) == (1, 0)
    kids = np.flatnonzero(s.alive & (s.parent == s.ids[0]))
    assert len(kids) == 1 and (s.taste_m[kids] == 0).all()


def test_taste_world_ledger_closes_and_memory_moves():
    on = taste_world("taste", taste_inherit=1.0)
    on.run(1500)
    assert rel_residual(on) < 1e-9
    assert np.abs(on.s.taste_m[on.s.alive]).max() > 0.1
    off = taste_world("taste_size")
    off.run(1500)
    assert on.state_hash() != off.state_hash()


def test_taste_summary_keys():
    on = taste_world("taste", taste_inherit=1.0)
    on.run(300)
    s, cfg = on.s, on.cfg
    m = s.taste_m[s.alive]
    assert np.abs(m).max() > 0
    out = on.summary()
    assert out["taste_plus"] == pytest.approx(float(m[:, 0].mean()))
    assert out["taste_minus"] == pytest.approx(float(m[:, 1].mean()))
    bad = 1 - k.good_kind(cfg.seed, on.step_no - 1, cfg.kind_flip_period)    # 방금 돈 스텝의 독
    col = 1 if bad == k.KIND_B else 0
    assert out["taste_on_poison"] == pytest.approx(float(m[:, col].mean()))
    # 독이 바뀌는 스텝 t(t - 1과 t의 good_kind가 다르다)에서도 방금 돈 스텝(t - 1)의 독 쪽을 적는다.
    # summary는 step_no만 읽으므로 그 자리로 옮겨 본다
    on.s.taste_m[s.alive] = np.where(np.arange(2) == 0, 0.25, -0.75).astype(np.float32)
    t = next(t for t in range(1, 10 ** 6)
             if k.good_kind(cfg.seed, t - 1, cfg.kind_flip_period) != k.good_kind(cfg.seed, t, cfg.kind_flip_period))
    on.step_no = t
    out = on.summary()
    was_bad = 1 - k.good_kind(cfg.seed, t - 1, cfg.kind_flip_period)
    assert out["good_kind"] == was_bad                                       # 다음 스텝은 독이 좋은 먹이가 된다
    assert out["taste_on_poison"] == pytest.approx(-0.75 if was_bad == k.KIND_B else 0.25)
    off = taste_world("taste_size")
    off.run(10)
    assert not {"taste_plus", "taste_minus", "taste_on_poison"} & set(off.summary())
    alone = World(small("taste"))           # 먹이 종류가 꺼진 세계: 독 쪽 키만 없다
    out = alone.summary()
    assert {"taste_plus", "taste_minus"} <= set(out) and "taste_on_poison" not in out


def test_taste_world_hash_pinned():
    w = taste_world("taste", taste_inherit=1.0)
    w.run(300)
    assert w.state_hash() == PINNED_TASTE_HASH_300
