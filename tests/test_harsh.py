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
from kkumteul.config import ACT_CARRY, N_EXT_IN, N_EXT_OUT, N_INTERNAL, N_STAGE_IN, N_STAGE_OUT  # noqa: E402
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
