"""맛 기억 장치 시험: 설정·묶음·저장, 먹기 갱신·잊기, 감각 맛 블록, 물려주기·기록, 손 개체(양성 대조).

    python3 -m pytest -q tests/test_taste.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_harsh import _place, _sense_once  # noqa: E402
from test_stages import rel_residual, small  # noqa: E402
from kkumteul import checkpoint as ckpt  # noqa: E402
from kkumteul import kernels as k  # noqa: E402
from kkumteul import measure  # noqa: E402
from kkumteul.config import ACT_TURN, N_EXT_IN, N_STAGE_IN, N_TASTE_IN, TASTE_VAL, TASTE_VIS, preset  # noqa: E402
from kkumteul.run import Runner  # noqa: E402
from kkumteul.world import World  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

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


def test_taste_m_saved_and_old_checkpoint_reads_zero():
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
# (11/06 다시 박음: 자는 개체의 물든 시야를 지우기로 고침 — Ruling 6, 사용자 승인)
PINNED_TASTE_HASH_300 = "0ae38c53b1e07a1ca8f1746f47492c002d76a69903a04da004252712eb60311b"

PARENT_M = (0.6, -0.4)


def _parent_world(inherit, size="taste", **kw) -> World:
    """부모 하나만 산 세계(먹이 없음). 부모 기억 = PARENT_M. size는 맛 블록 크기 묶음(taste 또는 장치 끈 taste_size)."""
    w = taste_world(size, taste_inherit=inherit, init_agents=1, food_inflow=0.0, **kw)
    assert w.s.alive[0] and w.s.alive.sum() == 1
    w.s.taste_m[0] = PARENT_M
    return w


def _natural_child(w) -> int:
    """부모(칸 0)가 자연 출생으로 새끼 하나를 낳게 하고 새끼 칸을 돌려준다."""
    s = w.s
    s.energy[0] = w.cfg.e_repro + 1.0
    s.age[0] = w.cfg.repro_min_age + 1
    s.act[0, k.ACT_REPRO] = 1.0             # 뇌가 번식을 고른 것으로 친다
    s.birth_step[0] = w.step_no - 1         # 태어난 그 스텝에는 번식하지 않는다
    assert k.process_births(w.step_no, s, w.fp, w.ip) == 1
    return int(np.flatnonzero(s.alive & (s.parent == s.ids[0]))[0])


@pytest.mark.parametrize("inherit", [0.0, 1.0, 0.5])
def test_natural_child_inherits(inherit):
    w = _parent_world(inherit, min_agents=0)
    j = _natural_child(w)
    assert w.s.taste_m[j] == pytest.approx((inherit * 0.6, inherit * -0.4))
    assert w.s.taste_m[0] == pytest.approx(PARENT_M)        # 부모 기억은 그대로


def test_natural_child_does_not_inherit_when_taste_off():
    # 맛 장치를 끈 같은 크기 세계에서는 taste_inherit=1이어도 새끼가 부모 기억을 베끼지 않는다
    w = _parent_world(1.0, size="taste_size", min_agents=0)
    assert not w.cfg.taste_enabled and w.ip[k.Q_TASTE_IN0] == -1
    j = _natural_child(w)
    assert (w.s.taste_m[j] == 0).all()
    assert w.s.taste_m[0] == pytest.approx(PARENT_M)


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


# ---- 손 개체(양성 대조) ----
def _hand_learner():
    sys.path.insert(0, str(ROOT / "wayfinder" / "stage2-4"))
    import hand_learner
    return hand_learner


def test_hand_taste_wiring():
    hl = _hand_learner()
    w = taste_world("taste")
    side = measure.side_from_world(w)
    t0 = w.cfg.taste_in0
    g = hl.make_hand(side["genomes"], 4, taste_in0=t0, **hl.VARIANTS["taste"])
    assert (g["Win"][:, :16, 0] == 0).all()                       # 원래 시야는 0
    assert (g["Win"][:, t0 + 4:t0 + 8, 0] == 0.4).all() and (g["Win"][:, t0:t0 + 4, 0] == -0.4).all()
    assert (g["Wout"][:, ACT_TURN, 0] == 1.0).all() and (g["eta"] == 0).all()
    with pytest.raises(ValueError):
        hl.make_hand(side["genomes"], 4, **hl.VARIANTS["taste"])     # taste_in0 없이


def test_after_flips_uses_real_flips():
    hl = _hand_learner()
    period, every, end = 250, 250, 3000

    def good(sd, t):
        return int(k.good_kind(sd, t, period))

    # 0~1749는 한 종류(1500 경계에서 다시 정해졌지만 그대로), 1750 경계에서 바뀌고 끝까지 그대로인 씨앗
    seed = next(sd for sd in range(1, 10 ** 6)
                if len({good(sd, t) for t in range(0, 1750, period)}) == 1
                and len({good(sd, t) for t in range(1750, end, period)}) == 1 and good(sd, 1749) != good(sd, 1750))
    # 줄 = curve의 꼴: 스텝 S 줄은 앞 줄 스텝부터 S 직전까지 먹은 것, good은 방금 돈 스텝(S − 1)의 좋은 종류
    series = [{"step": s, "good": good(seed, s - 1), "eaten": 10 + i, "poisoned": i % 4}
              for i, s in enumerate(range(every, end + 1, every))]
    by = {r["step"]: r for r in series}

    def ratio(steps):
        return sum(by[s]["poisoned"] for s in steps) / sum(by[s]["eaten"] for s in steps)

    out = hl.after_flips(series, period, seed, win=300)
    assert len(out) == 1 and out[0]["flip"] == 1750
    # 1750~2050에 시작한 줄(스텝 2000, 2250), 그 뒤 끝까지(2500, 2750, 3000)
    assert out[0]["first"] == pytest.approx(ratio([2000, 2250]))
    assert out[0]["rest"] == pytest.approx(ratio([2500, 2750, 3000]))
    # 씨앗으로 보면 바뀌지 않은 경계(750)에서 good이 바뀐 줄은 받지 않는다(씨앗이나 주기를 잘못 넘긴 것)
    bad = [dict(r) for r in series]
    bad[3]["good"] = 1 - bad[3]["good"]
    with pytest.raises(ValueError):
        hl.after_flips(bad, period, seed)


@pytest.mark.parametrize("lab", [True, False])
def test_hand_curve_counts_placed_food(lab, monkeypatch):
    hl = _hand_learner()
    cfg = small("stage2", "kinds", "taste")
    side = measure.side_from_world(World(cfg))
    g = hl.make_hand(side["genomes"], 4, taste_in0=cfg.taste_in0, **hl.VARIANTS["taste"])
    seen = {}
    real = hl.measure.inject_agent

    def spy(world, genomes, *a, **kw):     # curve가 실제로 만든 세계의 설정과 넣는 유전체를 본다
        seen["cfg"], seen["repro"] = world.cfg, genomes["bout"][:, k.ACT_REPRO].copy()
        return real(world, genomes, *a, **kw)
    monkeypatch.setattr(hl.measure, "inject_agent", spy)
    x = hl.curve(cfg, g, 4, 200, 1, every=100, lab=lab)
    assert len(x["series"]) == 2
    for r in x["series"]:
        assert r["placed"] > 0 and 0.0 <= r["placed_poison"] <= 1.0
    lab_cfg = measure.arena_config(hl.with_set(cfg, hl.LAB), 1, 4)
    plain_cfg = measure.arena_config(cfg, 1, 4)
    assert lab_cfg.to_dict() != plain_cfg.to_dict()       # LAB을 걸면 설정이 달라진다(아래 단언이 뜻이 있게)
    assert seen["cfg"].to_dict() == (lab_cfg if lab else plain_cfg).to_dict()
    if lab:
        assert (seen["repro"] == -5.0).all()
    else:                                                # 번식 편향을 건드리지 않는다
        assert (seen["repro"] == g["bout"][:, k.ACT_REPRO]).all() and not (seen["repro"] == -5.0).all()


def test_taste_verdict_per_world():
    # 판정 0은 세계마다 낸다: LAB만 통과하면 두 세계를 합친 평균이 문턱을 넘어도 전체는 실패다
    hl = _hand_learner()
    rest = {(True, True): [0.1, 0.1], (True, False): [0.5, 0.5],
            (False, True): [0.5, 0.5], (False, False): [0.5, 0.5]}
    v = hl.taste_verdict(rest)
    assert [(w["lab"], w["pass"]) for w in v["worlds"]] == [(True, True), (False, False)]
    assert v["pass"] is False
    assert v["worlds"][0]["on_rest"] == pytest.approx(0.1) and v["worlds"][1]["off_rest"] == pytest.approx(0.5)
    rest[(False, True)] = [0.2, 0.3]
    assert hl.taste_verdict(rest)["pass"] is True


def test_hand_taste_command_runs(tmp_path, monkeypatch, capsys):
    # 맛 블록이 없는 옛 체크포인트를 맛 크기로 읽어 세계 둘 × 켬/끔을 짧게 돌린다(배선·인자·판정 길을 본다).
    # 독을 50스텝마다 다시 정해 400스텝 안에 실제 바뀜이 생기게 한다(판정 값은 보지 않는다)
    hl = _hand_learner()
    r = Runner.new(tmp_path / "r", small("stage2", "kinds"))
    r.loop(400, quiet=True)
    r.close()
    p = ckpt.latest(tmp_path / "r" / "checkpoints")
    out = tmp_path / "taste.json"
    monkeypatch.setenv("HAND_N", "4")
    monkeypatch.setenv("HAND_SEEDS", "1")
    monkeypatch.setenv("HAND_SET", "kind_flip_period=50")
    monkeypatch.setattr(sys, "argv", ["hand_learner.py", "taste", str(p), "400", str(out)])
    hl.main()
    res = json.loads(out.read_text())
    runs = [(x["lab"], x["taste"]) for x in res["taste"]]
    assert runs == [(True, True), (True, False), (False, True), (False, False)]
    assert all(len(x["series"]) == 4 and "placed_poison" in x["series"][0] for x in res["taste"])
    assert any(x["flips"] for x in res["taste"])             # 바뀜이 있어 판정이 rest 값을 받는다
    text = capsys.readouterr().out
    assert "판정 0 LAB" in text and "판정 0 그대로" in text
    assert set(res["verdict0"]) == {"pass", "worlds"}
    assert [w["lab"] for w in res["verdict0"]["worlds"]] == [True, False]
    assert all(set(w) == {"lab", "pass", "on_rest", "off_rest"} for w in res["verdict0"]["worlds"])


def test_sleeping_agent_taste_vision_cleared():
    """자는 개체는 시야를 지우므로 시야에서 나온 물든 시야 8칸도 지운다. 기억 값 2칸은 시야가 아니라 남는다."""
    w = taste_world("taste")
    s = w.s
    t0 = w.cfg.taste_in0
    vis = slice(t0 + TASTE_VIS, t0 + TASTE_VAL)
    val = slice(t0 + TASTE_VAL, t0 + TASTE_VAL + 2)
    w.run(5)
    live = np.flatnonzero(s.alive)
    s.taste_m[live] = (0.6, -0.4)
    sleep, awake = live[::2], live[1::2]
    s.asleep[sleep], s.fatigue[sleep] = 1.0, 50.0       # 한 스텝에 깨지 않게
    s.asleep[awake], s.fatigue[awake] = 0.0, 0.0
    ids0 = s.ids.copy()
    w.step()
    sleep = sleep[s.alive[sleep] & (s.ids[sleep] == ids0[sleep]) & (s.asleep[sleep] > 0.0)]
    awake = awake[s.alive[awake] & (s.ids[awake] == ids0[awake]) & (s.asleep[awake] == 0.0)]
    assert len(sleep) >= 5 and len(awake) >= 5
    assert np.all(s.obs[sleep, :2 * w.cfg.vision_sectors] == 0.0)   # 시야는 원래 지워진다
    assert np.all(s.obs[sleep, vis] == 0.0)
    assert np.abs(s.obs[awake, vis]).max() > 0.0                    # 깨어 있는 쪽은 물든 시야가 그대로
    assert np.abs(s.obs[sleep, val]).max() > 0.0                    # 기억 값은 남는다
