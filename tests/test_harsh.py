"""혹독한 세계 장치 시험(1부): 확장 칸, 갈라 켜기, 장치별 장부.

    python3 -m pytest -q tests/test_harsh.py
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_stages import small as _small  # noqa: E402
from kkumteul.config import ACT_CARRY, N_EXT_IN, N_EXT_OUT, N_INTERNAL, N_STAGE_IN, N_STAGE_OUT, Config  # noqa: E402
from kkumteul.world import World  # noqa: E402


def small(*presets, **kw) -> Config:
    """test_stages.small과 같되, 묶음을 먼저 적용한 뒤 kw를 덧붙인다(장치는 확장 크기 위에서 검사)."""
    cfg = _small(*presets)
    return Config.from_dict({**cfg.to_dict(), **kw}) if kw else cfg


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
