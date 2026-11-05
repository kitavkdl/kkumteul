"""혹독한 세계 장치 시험(1부): 확장 칸, 갈라 켜기, 장치별 장부.

    python3 -m pytest -q tests/test_harsh.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_stages import rel_residual, small  # noqa: E402
from kkumteul import checkpoint as ckpt  # noqa: E402
from kkumteul import replay  # noqa: E402
from kkumteul import kernels as k  # noqa: E402
from kkumteul import stages as st  # noqa: E402
from kkumteul.config import ACT_CARRY, EXT_SEASON, N_EXT_IN, N_EXT_OUT, N_INTERNAL, N_STAGE_IN, N_STAGE_OUT  # noqa: E402
from kkumteul.run import Runner  # noqa: E402
from kkumteul.world import World  # noqa: E402


def test_ext_preset_sizes():
    c = small("ext")
    assert (c.n_spare_in, c.n_spare_out) == (N_STAGE_IN + N_EXT_IN, N_STAGE_OUT + N_EXT_OUT) == (26, 7)
    assert c.n_out > ACT_CARRY and c.ext_in0 == 2 * c.vision_sectors + N_INTERNAL + N_STAGE_IN


def test_harsh_device_needs_ext_size():
    with pytest.raises(ValueError, match="ext"):
        small(season_enabled=True)          # 기본 크기에 장치만 켬
    small("ext", season_enabled=True)       # 확장 크기면 된다


def test_ext_world_gives_new_agents_carry_bias_off():
    w = World(small("ext"))
    assert (w.s.bout[w.s.alive, ACT_CARRY] == -1.0).all()


def test_default_size_has_no_carry_slot():
    c = small()
    assert c.n_out == ACT_CARRY and not c.needs_ext   # 기본 9칸이라 ACT_CARRY 칸이 없다


def test_harsh_preset_turns_devices_on():
    c = small("harsh")
    assert c.season_enabled and c.spoil_enabled and c.needs_ext


# ---- 갈라 켜기: 기본 크기 체크포인트에서 확장 크기로 ----
def _main(tmp_path, *presets):
    r = Runner.new(tmp_path / "main", small(*presets)); r.loop(400, quiet=True); r.close()


def _ckpt_at(run_dir, step):
    return dict(ckpt.list_checkpoints(run_dir / "checkpoints"))[step]


def test_branch_into_ext_from_default_size(tmp_path):
    _main(tmp_path, "stage4")
    src, _ = ckpt.load(_ckpt_at(tmp_path / "main", 400))
    r = Runner.branch(tmp_path / "main", tmp_path / "ext", 400, ["preset=ext"], False, False)
    a, n_in0, n_out0 = r.world.s.alive, src.cfg.n_in, src.cfg.n_out
    assert r.world.cfg.n_spare_in == 26 and r.world.s.Win.shape[1] == r.world.cfg.n_in
    assert np.array_equal(r.world.s.Win[a][:, :n_in0], src.s.Win[a][:, :n_in0])
    assert not r.world.s.Win[a][:, n_in0:].any() and not r.world.s.Wout[a][:, n_out0:].any()
    assert (r.world.s.bout[a, ACT_CARRY] == -1.0).all() and "bout[ACT_CARRY]=-1" in r.meta["genome_edits"]
    r.loop(400, quiet=True); r.close()
    assert rel_residual(r.world) < 1e-9


def test_branch_ext_is_idempotent(tmp_path):
    _main(tmp_path, "ext")
    r = Runner.branch(tmp_path / "main", tmp_path / "again", 400, ["preset=ext"], False, False)
    assert (r.world.cfg.n_spare_in, r.world.cfg.n_spare_out) == (26, 7) and r.meta["genome_edits"] == []
    r.close()


def test_ext_branch_resumes_exactly(tmp_path):
    _main(tmp_path)
    r = Runner.branch(tmp_path / "main", tmp_path / "ext", 400, ["preset=ext"], False, False)
    r.loop(400, quiet=True); r.close()
    again = Runner.resume(tmp_path / "ext")
    assert again.world.state_hash() == r.world.state_hash(); again.close()


def test_replay_record_with_ext_override(tmp_path):
    _main(tmp_path)
    rep = replay.record(_ckpt_at(tmp_path / "main", 400), steps=50, overrides=["preset=ext"])
    assert rep is not None


# ---- 과제 3: 겨울 ----
def test_season_schedule_keeps_yearly_mean():
    w = World(small("ext", season_enabled=True))
    P, W = w.cfg.season_period, w.cfg.winter_len
    g = [w.season(t)[0] for t in range(P)]
    assert abs(np.mean(g) - 1.0) < 1e-12
    assert w.season(0)[0] > 1.0
    assert w.season(P - 1)[:2] == (pytest.approx(0.3), 1.5)      # 해의 마지막 스텝은 겨울
    assert w.season(P - W - 1)[1] == 1.0 and w.season(P - W)[:2] == (pytest.approx(0.3), 1.5)
    assert w.season(P)[0] == w.season(0)[0]                        # 해가 되풀이된다


def test_season_off_is_neutral():
    assert World(small("ext")).season(3999) == (1.0, 1.0, 0.0, 0.0)


def test_season_config_validated():
    for bad in (dict(winter_len=0), dict(winter_len=4000), dict(winter_mult=1.5), dict(winter_basal_mult=0.9)):
        with pytest.raises(ValueError):
            small("ext", season_enabled=True, **bad)


def test_winter_cuts_grass_and_raises_basal():
    on = World(small("ext", season_enabled=True, season_period=400, winter_len=200))
    off = World(small("ext"))
    for w in (on, off):
        w.run(400)
    assert on.s.ledger[k.L_BASAL] > off.s.ledger[k.L_BASAL]
    assert rel_residual(on) < 1e-9
    # 겨울만 도는 구간(200~400)에서는 들어오는 풀도 준다
    cold = World(small("ext", season_enabled=True, season_period=400, winter_len=200))
    base = World(small("ext"))
    for w in (cold, base):
        w.step_no = 200
        w.run(200)
    assert cold.s.ledger[k.L_SPAWN] < 0.5 * base.s.ledger[k.L_SPAWN]


def test_season_input_is_written():
    w = World(small("ext", season_enabled=True))
    w.run(5)
    a = w.s.alive
    c = w.cfg.ext_in0 + EXT_SEASON
    _, _, sn, cs = w.season(w.step_no - 1)
    assert np.allclose(w.s.obs[a, c], sn) and np.allclose(w.s.obs[a, c + 1], cs)


def test_winter_with_niche_conserves_energy():
    w = World(small("stage4", "ext", season_enabled=True, season_period=400, winter_len=200))
    w.run(1200)
    w.check_sane()
    assert rel_residual(w) < 1e-9


@pytest.mark.parametrize("mult", [0.3, 1.0])
def test_niche_share_follows_season_mult(mult):
    # 거름 몫이 겨울 배수를 따르는지: niche_world를 한 번 불러 budget 이동량을 센다.
    # (장부 잔차에는 budget이 들어가지 않아 위 보존 시험만으로는 이것을 가리지 못한다)
    w = World(small("stage4", "ext", season_enabled=True))
    w.s.soil[:] = 0.0
    w.s.soil[0] = 0.1 * w.cfg.soil_ref          # 몫이 먹이 한 알(1.0)보다 작게 잡아 먹이가 놓이지 않게 한다
    w.s.budget[:] = 0.0
    st.niche_world(0, w.s, w.fs, w.isp, mult)
    frac = float(w.s.soil.sum()) / w.cfg.soil_ref
    share = w.cfg.food_inflow * mult * w.cfg.fert_frac * frac
    assert share < w.cfg.food_energy
    assert w.s.budget[1] == pytest.approx(share, rel=1e-6)
    assert w.s.budget[0] == pytest.approx(-share, rel=1e-6)


# ---- 부패 ----
def _empty_world(*presets, **kw):
    """먹는 개체가 없는 세계(init_agents=0, min_agents=0). 먹이는 시험이 직접 놓는다."""
    return World(small(*presets, init_agents=0, min_agents=0, food_inflow=0.0, **kw))


def _place(w, e=1.0, born=0, kind=k.KIND_A):
    """먹이 하나를 놓고 칸 번호를 돌려준다. 에너지 장부에도 들어온 것으로 적는다."""
    before = w.s.f_alive.copy()
    assert k._place_food(w.s, 10.0, 10.0, e, kind, born)
    w.s.ledger[k.L_SPAWN] += e
    return int(np.flatnonzero(w.s.f_alive & ~before)[0])


def test_food_spoils_after_delay_with_half_life():
    w = _empty_world("ext", spoil_enabled=True)
    j = _place(w)
    w.run(400)
    assert w.s.f_energy[j] == pytest.approx(1.0)                       # 난 지 400스텝까지는 그대로
    w.run(600)
    assert w.s.f_energy[j] == pytest.approx((1 - 0.00115) ** 600, rel=1e-9)
    assert 0.49 < w.s.f_energy[j] < 0.51                                # 반감기 약 600
    w.run(2500)
    assert not w.s.f_alive[j] and w.s.counters[k.C_SPOILED] == 1 and w.s.counters[k.C_NFOOD] == 0
    assert rel_residual(w) < 1e-9
    assert w.s.ledger[k.L_LOST] == pytest.approx(1.0, rel=1e-9)         # 줄어든 양과 남은 것이 모두 L_LOST로


def test_spoil_off_leaves_food_alone():
    w = _empty_world("ext")
    j = _place(w)
    w.run(1500)
    assert w.s.f_energy[j] == 1.0 and w.s.counters[k.C_SPOILED] == 0


def test_corpse_spoils_too():
    w = _empty_world("ext", spoil_enabled=True)
    j = _place(w, born=-1000, kind=k.KIND_CORPSE)
    w.run(1)
    assert w.s.f_energy[j] == pytest.approx(1 - 0.00115, rel=1e-12)


def test_drop_food_slot_clears_slot_and_count():
    w = _empty_world("ext")
    j = _place(w)
    k._drop_food_slot(w.s, j)
    assert not w.s.f_alive[j] and w.s.counters[k.C_NFOOD] == 0


@pytest.mark.parametrize("t0,mult", [(0, 1.0), (199, 1.0), (200, 0.2), (399, 0.2)])
def test_spoil_slows_in_winter(t0, mult):
    w = _empty_world("ext", spoil_enabled=True, season_enabled=True, season_period=400, winter_len=200)
    j = _place(w, born=-1000)
    w.step_no = t0
    w.run(1)
    assert w.s.f_energy[j] == pytest.approx(1.0 - 0.00115 * mult, rel=1e-12)   # 겨울(200~399)에는 0.2배
    assert rel_residual(w) < 1e-9


def test_spoil_without_season_ignores_winter_mult():
    w = _empty_world("ext", spoil_enabled=True, season_period=400, winter_len=200)
    j = _place(w, born=-1000)
    w.step_no = 300
    w.run(1)
    assert w.s.f_energy[j] == pytest.approx(1.0 - 0.00115, rel=1e-12)           # 겨울이 꺼져 있으면 늘 1배


def test_spoil_config_validated():
    for bad in (dict(spoil_after=-1), dict(spoil_rate=1.5), dict(spoil_rate=-0.1),
                dict(spoil_min_frac=1.0), dict(spoil_winter_mult=1.5)):
        with pytest.raises(ValueError):
            small("ext", spoil_enabled=True, **bad)


def test_spoil_world_runs_and_conserves_energy():
    w = World(small("stage4", "ext", spoil_enabled=True))
    w.run(1500)
    w.check_sane()
    assert w.summary()["pop"] > 20 and rel_residual(w) < 1e-9


def test_fast_spoil_drops_food_and_conserves_energy():
    # 기본값으로는 1500스텝 안에 사라지는 먹이가 없다. 빠르게 썩혀 칸이 비워지는 길까지 장부로 본다
    w = World(small("stage4", "ext", spoil_enabled=True, spoil_after=100, spoil_rate=0.01))
    w.run(1500)
    w.check_sane()
    assert w.s.counters[k.C_SPOILED] > 20 and rel_residual(w) < 1e-9
    assert w.s.counters[k.C_NFOOD] == int(w.s.f_alive.sum())


def test_spoil_with_cap_replace_conserves_energy():
    w = World(small("ext", spoil_enabled=True, food_cap_replace=True))
    w.run(1500)
    assert rel_residual(w) < 1e-9
    # 상한에 닿은 세계에서도 마찬가지다
    w = World(small("ext", spoil_enabled=True, food_cap_replace=True, food_cap=150))
    w.run(1500)
    assert w.s.counters[k.C_FOOD_REPLACED] > 0 and rel_residual(w) < 1e-9


def test_spoil_and_cap_replace_in_one_step_both_go_to_lost():
    # 상한(4)이 찬 세계에 새 먹이가 하나 나는 스텝: 가장 오래된 하나는 밀려나고, 하한 아래로 내려간 둘은 썩어 사라진다
    w = World(small("ext", spoil_enabled=True, food_cap_replace=True, food_cap=4, init_agents=0, min_agents=0,
                    food_inflow=1.0))
    old = [_place(w, born=-1000), _place(w)]
    rotten = [_place(w, e=0.05001, born=-400), _place(w, e=0.05001, born=-400)]
    w.run(1)
    c = w.s.counters
    assert c[k.C_FOOD_REPLACED] == 1 and c[k.C_SPOILED] == 2 and c[k.C_NFOOD] == int(w.s.f_alive.sum()) == 2
    assert not w.s.f_alive[old[0]] and not w.s.f_alive[rotten[0]] and not w.s.f_alive[rotten[1]]
    assert w.s.f_alive[old[1]]
    assert w.s.ledger[k.L_LOST] == pytest.approx(1.0 + 2 * 0.05001, rel=1e-9)   # 밀려난 하나 + 썩은 둘의 전부
    assert rel_residual(w) < 1e-9


def test_replay_tells_spoiled_food_from_eaten():
    w = World(small("ext", spoil_enabled=True, spoil_after=60, spoil_rate=0.02))
    w.run(300)
    c0 = w.s.counters.copy()
    rec = replay.Recorder(w, every=5)
    rec.advance(300)
    rep = rec.finish()
    dc = w.s.counters - c0
    eaten, spoiled = int(dc[k.C_EATEN]), int(dc[k.C_SPOILED])
    assert eaten > 50 and spoiled > 5
    assert rep.seen["spoiled"] == spoiled
    assert int((rep.food["why"] == replay.WHY_SPOILED).sum()) == spoiled
    assert rep.seen["unknown"] <= 0.02 * eaten
