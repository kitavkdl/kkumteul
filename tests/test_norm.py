"""규범의 씨앗(가로채기·벌주기) 시험.

설계 docs/superpowers/specs/2026-11-26-norms-seed-design.md, 계획 docs/superpowers/plans/2026-11-26-norms-seed.md.

    python3 -m pytest -q tests/test_norm.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_stages import PINNED_HASH_300, rel_residual, small  # noqa: E402
from kkumteul import config as cfgmod  # noqa: E402
from kkumteul import kernels as k  # noqa: E402
from kkumteul import measure  # noqa: E402
from kkumteul import stages as st  # noqa: E402
from kkumteul.config import Config  # noqa: E402
from kkumteul.world import World  # noqa: E402

NORM_DEFAULTS = {
    "norm_measure": False, "norm_enabled": False, "norm_mode": "full", "norm_mark_src": "take",
    "norm_mark_tau": 50, "norm_mark_noise": 0.0, "norm_sig_theta": 0.5, "norm_pun_return": 0.1,
    "norm_aim": "marked", "norm_inhibit": 0.8, "norm_mem_tau": 50, "norm_mem_rate": 1.0, "norm_mem_obs": 0.2,
    "norm_mem_obs_mode": "outcome", "norm_mem_inherit": 0.0, "norm_impulse": 0.0, "norm_impulse_len": 5,
    "norm_mix_period": 0, "norm_mix_frac": 0.2, "norm_block_tag": -1, "norm_block3_tag": -1,
}


def norm_cfg(*presets, **kw) -> Config:
    """규범 장치를 켠 작은 설정(4단계 + 다툼 + norm 크기)."""
    return small("stage4", "conflict", "norm", *presets, **kw)


# ======================= 과제 1: 설정·검사·상수·preset =======================
def test_norm_config_defaults():
    keys = {f for f in Config().to_dict() if f.startswith("norm_")}
    assert keys == set(NORM_DEFAULTS) and len(keys) == 21
    c = Config()
    for name, v in NORM_DEFAULTS.items():
        got = getattr(c, name)
        assert got == v and type(got) is type(v), name


def test_norm_config_range_checks():
    base = dict(world_w=256.0, world_h=256.0)
    for bad in (dict(norm_pun_return=-0.01), dict(norm_pun_return=0.51), dict(norm_inhibit=-0.1),
                dict(norm_inhibit=1.1), dict(norm_mem_rate=-0.1), dict(norm_mem_rate=1.1), dict(norm_mem_obs=-0.1),
                dict(norm_mem_obs=1.1), dict(norm_impulse=-0.1), dict(norm_impulse=1.1),
                dict(norm_mark_noise=-0.1), dict(norm_mark_noise=1.1), dict(norm_mix_frac=-0.1),
                dict(norm_mix_frac=1.1), dict(norm_mark_tau=0), dict(norm_mem_tau=0), dict(norm_impulse_len=0),
                dict(norm_mix_period=-1), dict(norm_mode="nope"), dict(norm_mark_src="nope"), dict(norm_aim="nope"),
                dict(norm_mem_obs_mode="nope"), dict(norm_mem_inherit=-0.1), dict(norm_mem_inherit=1.1)):
        with pytest.raises(ValueError):
            small("stage4", "conflict", **base, **bad)
    # attack_gain을 바꾸면 위 끝도 같이 움직인다
    assert small("stage4", "conflict", attack_gain=0.3, norm_pun_return=0.3).norm_pun_return == 0.3
    with pytest.raises(ValueError):
        small("stage4", "conflict", attack_gain=0.3, norm_pun_return=0.31)
    # 끝값은 된다
    c = small("stage4", "conflict", norm_pun_return=0.0, norm_inhibit=1.0, norm_mark_tau=1, norm_mem_tau=1,
              norm_impulse_len=1, norm_impulse=1.0)
    assert c.norm_pun_return == 0.0 and c.norm_mark_tau == 1


def test_norm_config_dependencies():
    # 기본 설정은 통과한다(의존 키는 기본값이 아닐 때만 요구한다)
    Config().validate()
    small().validate()
    # norm_measure는 conflict 요구
    with pytest.raises(ValueError):
        small("stage4", norm_measure=True)
    assert small("stage4", "conflict", norm_measure=True).norm_measure
    # norm_enabled는 conflict·social·입력 크기·vision_sectors 8 요구
    assert norm_cfg().norm_enabled
    with pytest.raises(ValueError):
        small("stage4", "norm")                                  # conflict 없음
    with pytest.raises(ValueError):
        small("stage2", "conflict", "norm")                      # social 없음
    with pytest.raises(ValueError):
        small("stage4", "conflict", "taste_size", norm_enabled=True)   # 입력 크기 모자람
    with pytest.raises(ValueError):
        small("stage4", "conflict", "norm", vision_sectors=6)
    # 기본값이 아니면 norm_enabled 요구
    for kw in (dict(norm_mark_src="sig"), dict(norm_mark_src="none"), dict(norm_mark_noise=0.01),
               dict(norm_mem_obs_mode="blind"), dict(norm_mem_obs_mode="off"), dict(norm_block_tag=0),
               dict(norm_block3_tag=1), dict(norm_mode="inert")):
        with pytest.raises(ValueError):
            small("stage4", "conflict", "norm_size", **kw)
        assert norm_cfg(**kw).norm_enabled, kw
    # norm_impulse > 0은 conflict만 요구
    assert small("stage4", "conflict", norm_impulse=0.015).norm_impulse == 0.015
    assert small("stage1", "conflict", norm_impulse=0.015).norm_impulse == 0.015
    with pytest.raises(ValueError):
        small("stage4", norm_impulse=0.015)
    # norm_mix_period > 0은 carry·teach 꺼짐 요구
    assert small("stage4", "conflict", norm_mix_period=200).norm_mix_period == 200
    with pytest.raises(ValueError):
        small("stage4", "conflict", "harsh2", norm_mix_period=200)
    with pytest.raises(ValueError):
        small("stage2", "kinds", "taste", soc_taste_enabled=True, juv_enabled=True, teach_enabled=True,
              conflict_enabled=True, norm_mix_period=200)


def test_norm_mem_inherit_guard():
    with pytest.raises(ValueError, match="과제 19"):
        norm_cfg(norm_mem_inherit=0.1)


def test_norm_string_overrides():
    c = small("stage4", "conflict").with_overrides(
        ["preset=norm", "norm_mode=inert", "norm_mark_src=sig", "norm_aim=nearest", "norm_mem_obs_mode=blind"])
    assert c.norm_enabled and c.norm_mode == "inert" and c.norm_mark_src == "sig"
    assert c.norm_aim == "nearest" and c.norm_mem_obs_mode == "blind"
    with pytest.raises(ValueError):
        c.with_overrides(["norm_mode=nope"])


def test_norm_presets_and_sizes():
    assert cfgmod.N_NORM_IN == 12 and cfgmod.N_NORM_VIS == 8
    assert (cfgmod.NORM_VIS, cfgmod.NORM_SELF, cfgmod.NORM_LOCAL, cfgmod.NORM_SAWPUN, cfgmod.NORM_MEM) == (0, 8, 9, 10, 11)
    assert cfgmod.NORM_MODES == {"full": 0, "inert": 1}
    assert cfgmod.NORM_SRCS == {"take": 0, "sig": 1, "none": 2}
    assert cfgmod.NORM_AIMS == {"marked": 0, "nearest": 1}
    assert cfgmod.NORM_OBS_MODES == {"outcome": 0, "blind": 1, "off": 2}
    ns = cfgmod.preset("norm_size")
    ts = cfgmod.preset("taste_size")
    assert ns == {**ts, "n_spare_in": ts["n_spare_in"] + cfgmod.N_NORM_IN}
    assert ns["n_spare_in"] == 13 + 13 + 10 + 12 == 48
    assert cfgmod.preset("norm") == {**ns, "norm_enabled": True}
    c = small("norm_size")
    assert c.n_in == 72 and c.vision_sectors == 8
    assert c.norm_in0 == c.taste_in0 + cfgmod.N_TASTE_IN
    assert c.n_in == c.norm_in0 + cfgmod.N_NORM_IN
    with pytest.raises(ValueError) as e:
        cfgmod.preset("모르는 이름")
    assert "norm_size" in str(e.value) and "norm" in str(e.value)


def test_old_configs_validate():
    Config().validate()
    for name in ("stage1", "stage2", "stage3", "stage4", "conflict", "kinds", "ext", "harsh", "harsh2", "taste",
                 "taste_size"):
        small(name).validate()
    small("stage4", "conflict", "kinds", "taste").validate()
    old = {kk: v for kk, v in Config().to_dict().items() if not kk.startswith("norm_")}
    Config.from_dict(old).validate()

