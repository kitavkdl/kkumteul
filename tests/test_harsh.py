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
