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
