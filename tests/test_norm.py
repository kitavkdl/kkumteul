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


# ======================= 과제 2: 난수·저장 필드·계수 배열·설정 칸·옛 호환 =======================
NORM_SAVED = {"mark_t": (np.int32, -1), "pun_t": (np.int32, -1), "mem": (np.float32, 0)}
NORM_UNSAVED = {"mark_vslot": (np.int32, -2), "mark_vid": (np.int64, -2), "mark_pun": (np.int8, 0),
                "pun_role": (np.int8, -1)}


def _assert_norm_fresh(s, j):
    assert s.mark_t[j] == -1 and s.pun_t[j] == -1 and s.mem[j] == 0
    assert s.mark_vslot[j] == -2 and s.mark_vid[j] == -2 and s.mark_pun[j] == 0 and s.pun_role[j] == -1


def test_norm_fields_registered():
    names = [f[0] for f in k.AGENT_FIELDS]
    info = {f[0]: f for f in k.AGENT_FIELDS}
    saved = [f[0] for f in k.AGENT_FIELDS if f[3]]
    assert saved[-3:] == ["mark_t", "pun_t", "mem"]
    assert names.index("mark_t") == names.index("feed_g") + 1
    for name, (dt, dflt) in NORM_SAVED.items():
        assert info[name][1] is dt and info[name][2] == () and info[name][3] is True, name
        assert name in k.SAVED_FIELDS and k.HASH_DEFAULTS[name] == dflt, name
    for name, (dt, _) in NORM_UNSAVED.items():
        assert info[name][1] is dt and info[name][2] == () and info[name][3] is False, name
        assert name not in k.SAVED_FIELDS, name
    w = World(small("stage4", "conflict", "norm_size"))
    for name, (dt, dflt) in {**NORM_SAVED, **NORM_UNSAVED}.items():
        v = getattr(w.s, name)
        assert v.dtype == np.dtype(dt) and (v == dflt).all(), name


def test_norm_scratch_arrays():
    assert k.TAG_ROWS_NORM == 5
    for name in ("norm_tab", "norm_tab_role", "norm_hist"):
        assert name in k.OTHER_SCRATCH and name not in k.SAVED_FIELDS
    assert "grad_tab" in k.OTHER_SCRATCH and k.TAG_ROWS == 8
    assert len(k.NT_NAMES) == k.N_NT and len(set(k.NT_NAMES)) == k.N_NT
    order = ("TAKE", "TAKE_KIN", "TAKE_SIM0", "TAKE_SIM1", "TAKE_SIM2", "TAKE_SIM3", "TAKE_LINE", "TAKE_IMP",
             "TAKE_ON_MARKED", "PUN", "TAKE_KILL", "PUN_KILL", "PUN_2ND", "PUN_3_KIN", "PUN_3_STR",
             "PUN_3_SIM0", "PUN_3_SIM1", "PUN_3_SIM2", "PUN_3_SIM3", "PUN_3_LINE", "PUN_NOVICT", "PUN_UNKNOWN",
             "PUN_VGONE", "EXPO", "OPP_SELF", "OPP_OTHER", "OPP_OTHER_SIM0", "OPP_OTHER_SIM1", "OPP_OTHER_SIM2",
             "OPP_OTHER_SIM3", "OPP_OTHER_LINE", "OPP_MARKED", "OPP_PLAIN", "ATK_VOL_MARKED", "ATK_VOL_PLAIN",
             "MARK_EPIS0", "MARK_EPIS1", "MARK_EPIS2", "MARK_EPIS_PUN0", "MARK_EPIS_PUN1", "MARK_EPIS_PUN2",
             "ALIVE_STEPS", "MARKED_STEPS", "SAW_PUN", "TAKE_DMG_RECV", "PUN_DMG_RECV", "VIOL_DMG_RECV",
             "TAKE_GAIN", "PUN_FINE", "PUN_COST", "ATK_COST", "EATEN_DEAD", "LIVES_DONE")
    assert k.N_NT == len(order)
    for idx, name in enumerate(order):
        assert getattr(k, "NT_" + name) == idx, name
        assert k.NT_NAMES[idx] == name.lower()
    assert (k.RT_PUN_FINE, k.RT_PUN_COST, k.RT_N_PUN, k.N_RT) == (0, 1, 2, 3)
    assert k.RT_ROLES == ("self", "kin", "stranger")
    w = World(small("stage4", "conflict", "norm_size"))
    assert w.s.norm_tab.shape == (5, k.N_NT) and w.s.norm_tab.dtype == np.float64
    assert w.s.norm_tab_role.shape == (3, k.N_RT) and w.s.norm_tab_role.dtype == np.float64
    assert w.s.norm_hist.shape == (4, 4, 2) and w.s.norm_hist.dtype == np.float64
    assert not w.s.norm_tab.any() and not w.s.norm_tab_role.any() and not w.s.norm_hist.any()


@pytest.mark.parametrize("presets", list(PINNED_HASH_300))
def test_norm_off_world_hash_unchanged(presets):
    w = World(small(*presets))
    w.run(300)
    assert w.state_hash() == PINNED_HASH_300[presets]


def test_state_hash_skips_default_norm_fields():
    w = World(small("stage4", "conflict", "norm_size"))
    w.run(20)
    h0 = w.state_hash()
    a = np.flatnonzero(w.s.alive)
    for name, v in (("mark_t", 7), ("pun_t", 3), ("mem", 0.5)):
        getattr(w.s, name)[a[0]] = v
        assert w.state_hash() != h0, name
        getattr(w.s, name)[:] = NORM_SAVED[name][1]
        assert w.state_hash() == h0, name
    # 저장 안 하는 필드는 해시에 안 들어간다
    w.s.mark_vslot[:] = 3
    w.s.mark_pun[:] = 1
    assert w.state_hash() == h0


def test_old_checkpoint_loads_norm_defaults():
    w = World(small("stage4", "conflict", "norm_size"))
    w.run(20)
    a = np.flatnonzero(w.s.alive)
    w.s.mark_t[a[:3]] = 15
    w.s.pun_t[a[:2]] = 12
    w.s.mem[a[:4]] = 0.25
    arrays = {kk: v.copy() for kk, v in w.state_arrays().items()}
    assert {"mark_t", "pun_t", "mem"} <= set(arrays)
    h = w.state_hash()
    w1 = World(w.cfg, populate=False)
    assert w1.load_arrays({kk: v.copy() for kk, v in arrays.items()})
    assert (w1.s.mark_t == w.s.mark_t).all() and (w1.s.pun_t == w.s.pun_t).all() and (w1.s.mem == w.s.mem).all()
    assert w1.state_hash() == h
    # 옛 묶음(필드 없음)은 기본값으로 읽고, 기본값뿐이면 해시가 그 필드가 생기기 전과 같다
    w.s.mark_t[:] = -1
    w.s.pun_t[:] = -1
    w.s.mem[:] = 0
    h_def = w.state_hash()
    for name in ("mark_t", "pun_t", "mem"):
        del arrays[name]
    w2 = World(w.cfg, populate=False)
    assert w2.load_arrays(arrays)       # exact: 빠진 것으로 치지 않는다
    assert (w2.s.mark_t == -1).all() and (w2.s.pun_t == -1).all() and (w2.s.mem == 0).all()
    assert (w2.s.mark_vslot == -2).all() and (w2.s.mark_vid == -2).all() and (w2.s.pun_role == -1).all()
    assert w2.state_hash() == h_def
    # 유전체 배열은 안 늘었다(묶음 읽기 그대로)
    assert measure.OPTIONAL_GENOME == ("taste_g", "soc_g", "show_g", "feed_g")
    assert not ({"mark_t", "pun_t", "mem"} & set(measure.GENOME))


def test_inject_agent_and_reset_body_init_norm_state():
    w = World(small("stage4", "conflict", "norm_size"))
    w.run(5)
    s = w.s
    for name, v in (("mark_t", 9), ("pun_t", 9), ("mem", 0.7), ("mark_vslot", 4), ("mark_vid", 99),
                    ("mark_pun", 1), ("pun_role", 2)):
        getattr(s, name)[:] = v
    side = measure.side_from_world(w)
    j_id = measure.inject_agent(w, side["genomes"], 0, 1, 10.0, 10.0, 0.0, 17)
    j = int(np.flatnonzero(s.ids == j_id)[0])
    _assert_norm_fresh(s, j)
    # _reset_body로 재사용한 칸
    dead = int(np.flatnonzero(~s.alive)[0])
    k._reset_body(s, dead, 12345, -1, 0, 0, 5, 50.0, 50.0, 0.0, 1.0)
    _assert_norm_fresh(s, dead)


def test_set_config_off_clears_norm_state():
    cfg_on = small("stage4", "conflict", "norm")
    w = World(cfg_on)
    w.run(5)
    s = w.s
    a = np.flatnonzero(s.alive)
    for name, v in (("mark_t", 3), ("pun_t", 4), ("mem", 0.6), ("mark_vslot", 2), ("mark_vid", 77),
                    ("mark_pun", 1), ("pun_role", 0)):
        getattr(s, name)[a] = v
    w.set_config(Config.from_dict({**cfg_on.to_dict(), "norm_enabled": False}))
    for j in a:
        _assert_norm_fresh(s, j)
    w.set_config(cfg_on)                       # 다시 켜도 낡은 표시가 없다
    assert (s.mark_t[a] == -1).all() and (s.mem[a] == 0).all()
    # norm_measure만 끄는 것·충동만 바꾸는 것은 지울 것이 없다
    cfg_m = small("stage4", "conflict", "norm_size", norm_measure=True, norm_impulse=0.01)
    w2 = World(cfg_m)
    w2.s.mark_t[:] = 5
    w2.set_config(Config.from_dict({**cfg_m.to_dict(), "norm_measure": False, "norm_impulse": 0.0}))
    assert (w2.s.mark_t == 5).all()


def test_norm_stage_params_packed():
    c = norm_cfg(norm_pun_return=0.2, norm_inhibit=0.6, norm_mem_rate=0.9, norm_mem_obs=0.3, norm_mem_tau=40,
                 norm_impulse=0.02, norm_mark_noise=0.001, norm_sig_theta=0.4, norm_mark_tau=25, norm_mix_frac=0.3,
                 norm_measure=True, norm_mode="inert", norm_mark_src="sig", norm_aim="nearest",
                 norm_mem_obs_mode="blind", norm_impulse_len=7, norm_block_tag=1, norm_block3_tag=2,
                 norm_mix_period=100)
    fs, isp = st.pack_stage_params(c)
    assert fs[st.F_NORM_PUN_RET] == 0.2 and fs[st.F_NORM_INHIBIT] == 0.6 and fs[st.F_NORM_MEM_RATE] == 0.9
    assert fs[st.F_NORM_MEM_OBS] == 0.3 and fs[st.F_NORM_MEM_TAU] == 40 and fs[st.F_NORM_IMPULSE] == 0.02
    assert fs[st.F_NORM_NOISE] == 0.001 and fs[st.F_NORM_SIG_THETA] == 0.4 and fs[st.F_NORM_TAU] == 25
    assert fs[st.F_NORM_MIX_FRAC] == 0.3 and fs[st.F_NORM_RADIUS] == c.soc_radius
    assert fs[st.F_NORM_VRANGE] == c.vision_range and fs[st.F_NORM_FOV] == c.vision_fov
    assert isp[st.I_NORM_MEASURE] == 1 and isp[st.I_NORM_ON] == 1 and isp[st.I_NORM_MODE] == cfgmod.NORM_INERT
    assert isp[st.I_NORM_SRC] == cfgmod.SRC_SIG and isp[st.I_NORM_AIM] == cfgmod.AIM_NEAREST
    assert isp[st.I_NORM_OBS_MODE] == cfgmod.OBS_BLIND and isp[st.I_NORM_IMP_LEN] == 7
    assert isp[st.I_NORM_BLOCK] == 1 and isp[st.I_NORM_BLOCK3] == 2 and isp[st.I_NORM_MIX_PERIOD] == 100
    assert isp[st.I_NORM_IN0] == c.norm_in0 and isp[st.I_NORM_VIS0] == c.norm_in0 + cfgmod.NORM_VIS
    # 꺼진 세계
    fs0, isp0 = st.pack_stage_params(small("stage4", "conflict", "norm_size"))
    assert isp0[st.I_NORM_ON] == 0 and isp0[st.I_NORM_MEASURE] == 0
    assert isp0[st.I_NORM_IN0] == -1 and isp0[st.I_NORM_VIS0] == -1
    assert isp0[st.I_NORM_BLOCK] == -1 and isp0[st.I_NORM_BLOCK3] == -1
    # 기존 칸은 그대로, 새 칸은 뒤에만
    assert st.F_PERTURB == 46 and st.I_TASTE_VIS0 == 13
    new_f = (st.F_NORM_PUN_RET, st.F_NORM_INHIBIT, st.F_NORM_MEM_RATE, st.F_NORM_MEM_OBS, st.F_NORM_MEM_TAU,
             st.F_NORM_IMPULSE, st.F_NORM_NOISE, st.F_NORM_SIG_THETA, st.F_NORM_TAU, st.F_NORM_MIX_FRAC,
             st.F_NORM_RADIUS, st.F_NORM_VRANGE, st.F_NORM_FOV)
    assert new_f == tuple(range(47, 60)) and st.N_FS == 60 == len(fs0)
    new_i = (st.I_NORM_MEASURE, st.I_NORM_ON, st.I_NORM_MODE, st.I_NORM_SRC, st.I_NORM_AIM, st.I_NORM_OBS_MODE,
             st.I_NORM_IMP_LEN, st.I_NORM_BLOCK, st.I_NORM_BLOCK3, st.I_NORM_MIX_PERIOD, st.I_NORM_IN0,
             st.I_NORM_VIS0)
    assert new_i == tuple(range(14, 26)) and st.N_IS == 26 == len(isp0)
