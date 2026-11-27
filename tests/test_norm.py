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
from kkumteul import rng as rngmod  # noqa: E402
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
                dict(norm_mem_obs_mode="nope"), dict(norm_mem_inherit=-0.1), dict(norm_mem_inherit=1.1),
                dict(norm_inhibit=1.0)):
        with pytest.raises(ValueError):
            small("stage4", "conflict", **base, **bad)
    # attack_gain을 바꾸면 위 끝도 같이 움직인다
    assert small("stage4", "conflict", attack_gain=0.3, norm_pun_return=0.3).norm_pun_return == 0.3
    with pytest.raises(ValueError):
        small("stage4", "conflict", attack_gain=0.3, norm_pun_return=0.31)
    # 끝값은 된다
    # norm_inhibit = 1은 막는다: mem = 1(맞은 직후)이면 가로채기 피해가 0이 되어 친 개체가 비용만 내고 표시도 0 피해로 붙는다
    c = small("stage4", "conflict", norm_pun_return=0.0, norm_inhibit=0.99, norm_mark_tau=1, norm_mem_tau=1,
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


# ======================= 과제 3: 훑기와 norm_measure 계수기 =======================
def hand_world(*presets, n=8, **kw):
    """모든 칸이 산 작은 세계(먹이 없음). 개체는 서로 멀리(25 간격) 두고, 손 시험이 몇을 옮겨 붙인다."""
    cfg = small("stage4", "conflict", *presets, max_agents=n, init_agents=n, min_agents=0, food_inflow=0.0, **kw)
    w = World(cfg)
    s = w.s
    assert int(s.alive.sum()) == n
    for q in range(n):
        s.x[q], s.y[q] = 20.0 + 25.0 * (q % 9), 30.0 + 40.0 * (q // 9)
    s.act[:] = 0
    s.touch[:] = 0
    s.energy[:] = 1.0
    s.parent[:] = -1
    s.color[:] = 0.5
    s.tag[:] = 0
    s.asleep[:] = 0
    return w


def regrid(w):
    s, c = w.s, w.cfg
    k.build_grid(s.x, s.y, s.alive, 1.0 / c.cell, c.grid_w, c.grid_h, s.a_start, s.a_items, s.cursor)


def put(w, i, x, y):
    w.s.x[i], w.s.y[i] = x, y


def nt(w, row, col):
    return float(w.s.norm_tab[row, col])


def _brute(s, cfg, i, t, tau):
    """i에게 닿는(거리 < 2r + 0.5) 가장 가까운 이웃과 표시된 이웃(에너지 > 0, 자기 제외, 토러스)."""
    r2 = (2 * cfg.agent_radius + 0.5) ** 2
    nb, nm, bd, bm = -1, -1, r2, r2
    for j in np.flatnonzero(s.alive):
        if j == i or s.energy[j] <= 0:
            continue
        dx = (s.x[j] - s.x[i] + cfg.world_w / 2) % cfg.world_w - cfg.world_w / 2
        dy = (s.y[j] - s.y[i] + cfg.world_h / 2) % cfg.world_h - cfg.world_h / 2
        d2 = dx * dx + dy * dy
        if d2 < bd:
            bd, nb = d2, j
        if s.mark_t[j] >= 0 and t - s.mark_t[j] <= tau and d2 < bm:
            bm, nm = d2, j
    return nb, nm


def test_scan_matches_nearest_touching():
    w = World(small("stage4", "conflict", "norm_size", norm_measure=True, init_agents=300, max_agents=400))
    s, cfg = w.s, w.cfg
    gen = np.random.default_rng(3)
    a = np.flatnonzero(s.alive)
    # 빽빽하게: 절반은 가운데 상자, 절반은 토러스 가장자리를 걸친 상자
    s.x[a] = np.where(np.arange(len(a)) % 2 == 0, 100.0 + gen.uniform(0, 25, len(a)), gen.uniform(-12, 12, len(a)) % 256)
    s.y[a] = np.where(np.arange(len(a)) % 2 == 0, 100.0 + gen.uniform(0, 25, len(a)), gen.uniform(-12, 12, len(a)) % 256)
    s.energy[a[::7]] = 0.0                     # 에너지 없는 이웃은 건너뛴다
    t = 100
    s.mark_t[a[::3]] = t - gen.integers(0, 60, len(a[::3]))     # 일부는 표시 창(τ = 50) 밖
    regrid(w)
    n_nb = n_nm = 0
    for i in a:
        nb, nm = st._scan_touching(s, w.fs, w.isp, i, t)
        assert nb == st._nearest_touching(s, w.fs, w.isp, i, True)
        bnb, bnm = _brute(s, cfg, i, t, cfg.norm_mark_tau)
        assert (nb, nm) == (bnb, bnm), i
        n_nb += nb >= 0
        n_nm += nm >= 0
    assert n_nb > 50 and n_nm > 10          # 양성 확인: 실제로 닿는 이웃이 있다


def test_expo_counts_gap_neighbors_and_ignores_touch():
    w = hand_world("norm_size", norm_measure=True)
    s = w.s
    put(w, 0, 100.0, 100.0)
    put(w, 4, 102.2, 100.0)                  # 겹치지 않는 틈(2.0 < d < 2.5): collide의 touch는 안 켜진다
    regrid(w)
    st.attack(0, s, w.fs, w.isp)
    assert nt(w, 0, k.NT_EXPO) == 2 and nt(w, 0, k.NT_ALIVE_STEPS) == 8
    s.touch[4] = k.TOUCH_HIT
    s.act[0, cfgmod.ACT_ATTACK] = 1.0
    st.attack(1, s, w.fs, w.isp)
    assert nt(w, 0, k.NT_EXPO) == 4 and nt(w, 0, k.NT_ALIVE_STEPS) == 16
    # 에너지가 0 이하인 개체는 차례가 없다(ALIVE_STEPS에는 세고 EXPO에는 안 센다)
    s.act[:] = 0
    s.energy[0] = 0.0
    st.attack(2, s, w.fs, w.isp)
    assert nt(w, 0, k.NT_EXPO) == 4 and nt(w, 0, k.NT_ALIVE_STEPS) == 24


def test_measure_counts_take_by_actor_tag():
    w = hand_world("norm_size", norm_measure=True)
    s, cfg = w.s, w.cfg
    put(w, 0, 100.0, 100.0)
    put(w, 4, 101.5, 100.0)
    regrid(w)
    s.tag[0], s.tag[4] = 1, 2
    s.parent[4] = s.ids[0]                    # 0이 4의 부모(계보 친족)
    s.act[0, cfgmod.ACT_ATTACK] = 1.0
    st.attack(0, s, w.fs, w.isp)
    dmg = cfg.attack_damage
    assert nt(w, 1, k.NT_TAKE) == 1 and nt(w, 2, k.NT_TAKE) == 0
    assert nt(w, 2, k.NT_TAKE_DMG_RECV) == pytest.approx(dmg) and nt(w, 1, k.NT_TAKE_DMG_RECV) == 0
    assert nt(w, 1, k.NT_TAKE_GAIN) == pytest.approx(dmg * cfg.attack_gain)
    assert nt(w, 2, k.NT_VIOL_DMG_RECV) == pytest.approx(dmg)
    assert nt(w, 1, k.NT_TAKE_KIN) == 1 and nt(w, 1, k.NT_TAKE_LINE) == 1
    assert nt(w, 1, k.NT_TAKE_SIM3) == 1                      # 같은 색 = 닮음 1
    assert nt(w, 1, k.NT_ATK_COST) == pytest.approx(cfg.attack_cost)
    # 남이고 색이 먼 쌍: 닮음 1 − 0.3/√3 ≈ 0.827 → 구간 0, 친족 아님
    s.parent[4] = -1
    s.color[4] = (0.8, 0.5, 0.5)
    st.attack(1, s, w.fs, w.isp)
    assert nt(w, 1, k.NT_TAKE) == 2 and nt(w, 1, k.NT_TAKE_KIN) == 1 and nt(w, 1, k.NT_TAKE_LINE) == 1
    assert nt(w, 1, k.NT_TAKE_SIM0) == 1
    # TAKE_KIN은 계보만(부모·자식·형제): 닮음 0.93(구간 2)이어도 계보가 없으면 친족 칸에 안 든다
    s.color[4] = (0.5 + 0.07 * 1.7320508075688772, 0.5, 0.5)
    st.attack(2, s, w.fs, w.isp)
    assert nt(w, 1, k.NT_TAKE_SIM2) == 1 and nt(w, 1, k.NT_TAKE_KIN) == 1 and nt(w, 1, k.NT_TAKE_LINE) == 1
    assert st._sim_bin(0.8799) == 0 and st._sim_bin(0.88) == 1 and st._sim_bin(0.92) == 2 and st._sim_bin(0.96) == 3


def test_viol_dmg_equals_total_in_measure_only():
    w = World(small("stage4", "conflict", "norm_size", norm_measure=True))
    w.s.bout[:, cfgmod.ACT_ATTACK] = 0.5
    w.run(300)
    tab = w.s.norm_tab
    assert tab[:, k.NT_TAKE].sum() > 0
    assert tab[:, k.NT_VIOL_DMG_RECV].sum() == pytest.approx(tab[:, k.NT_TAKE_DMG_RECV].sum())
    assert tab[:, k.NT_TAKE].sum() == w.s.counters[k.C_ATTACKS]


def test_norm_tag_fold_no_overflow():
    w = hand_world("norm_size", norm_measure=True)
    s = w.s
    put(w, 0, 100.0, 100.0)
    put(w, 4, 101.5, 100.0)
    regrid(w)
    s.act[0, cfgmod.ACT_ATTACK] = 1.0
    for step, (ta, tb) in enumerate(((-3, 4), (99, -1), (4, 99))):
        s.tag[0], s.tag[4] = ta, tb
        s.energy[4] = 1.0
        st.attack(step, s, w.fs, w.isp)
    assert nt(w, 4, k.NT_TAKE) == 3 and nt(w, 4, k.NT_TAKE_DMG_RECV) > 0
    assert s.norm_tab[:4, k.NT_TAKE].sum() == 0
    assert not s.norm_tab_role.any() and not s.norm_hist.any()


def test_measure_only_hash_equals_off():
    hashes = []
    for kw in (dict(), dict(norm_measure=True)):
        w = World(small("stage4", "conflict", "norm_size", **kw))
        w.s.bout[:, cfgmod.ACT_ATTACK] = 0.5
        w.run(300)
        hashes.append(w.state_hash())
        if kw:
            assert w.s.norm_tab[:, k.NT_TAKE].sum() > 0 and w.s.norm_tab[:, k.NT_EXPO].sum() > 0
    assert hashes[0] == hashes[1]


def test_scan_not_called_when_both_off():
    # 공통 규칙은 "꺼짐 = 호출 0"을 함수를 감싸 세어 확인하라지만, 훑기(_scan_touching)는 njit인 attack 안에서 불려
    # 파이썬에서 감싸도 호출이 안 잡힌다(호출 수를 못 센다) → 훑기에서만 세는 계수가 0인지로 대신 확인한다
    w = World(small("stage4", "conflict", "norm_size", norm_impulse=0.05))
    w.s.bout[:, cfgmod.ACT_ATTACK] = 0.5
    w.run(100)
    assert w.s.counters[k.C_ATTACKS] > 0
    assert not w.s.norm_tab.any()


def test_summary_norm_measure_keys():
    w = World(small("stage4", "conflict", "norm_size", norm_measure=True))
    w.run(30)
    out = w.summary()
    tab = w.s.norm_tab
    assert out["norm_expo_frac"] == pytest.approx(tab[:, k.NT_EXPO].sum() / tab[:, k.NT_ALIVE_STEPS].sum())
    assert out["norm_take_per_step"] == pytest.approx(tab[:, k.NT_TAKE].sum() / tab[:, k.NT_ALIVE_STEPS].sum())
    w0 = World(small("stage4", "conflict", "norm_size"))
    w0.run(5)
    assert "norm_expo_frac" not in w0.summary()


# ======================= 과제 4: 떨리는 손(norm_impulse) =======================
def test_impulse_interval_is_stateless_and_length_L():
    seed, p = 7, 0.3
    for L in (1, 5, 10):
        for aid in range(40):
            for blk in range(6):
                want = rngmod.u01(seed, rngmod.S_NORM_IMPULSE, blk, aid) < p
                for t in range(blk * L, blk * L + L):          # 구간 안 L스텝 내내 같다(저장할 것이 없다)
                    assert st._impulse_on(seed, t, aid, L, p) == want
                assert st._impulse_on(seed, blk * L, aid, L, p) == st._impulse_on(seed, blk * L, aid, L, p)


def test_impulse_probability_close_to_p():
    seed, p = 3, 0.015
    starts = sum(st._impulse_on(seed, blk * 5, aid, 5, p) for aid in range(20000) for blk in range(5))
    assert starts / 100000 == pytest.approx(p, abs=0.002)
    for L in (1, 10):
        on = sum(st._impulse_on(seed, t, aid, L, p) for aid in range(2000) for t in range(50))
        assert on / 100000 == pytest.approx(p, abs=0.003)          # 개체 스텝 중 충동 비율은 L과 무관하게 p


def _imp_scene(*presets, **kw):
    """0번 하나만 깨어 있는 손 세계(나머지는 자서 충동이 강제되지 않는다). 0은 뇌 출력 −1."""
    w = hand_world(*presets, **kw)
    s = w.s
    s.asleep[:] = 1.0
    s.asleep[0] = 0.0
    s.act[:, cfgmod.ACT_ATTACK] = 0.0
    s.act[0, cfgmod.ACT_ATTACK] = -1.0
    put(w, 0, 100.0, 100.0)
    put(w, 4, 101.5, 100.0)            # 가장 가까운 닿는 이웃
    put(w, 5, 100.0, 102.3)            # 더 먼 닿는 이웃
    regrid(w)
    return w


def test_impulse_forces_attack_and_hits_nearest():
    w = _imp_scene("norm", norm_impulse=1.0)
    s, cfg = w.s, w.cfg
    s.mark_t[5] = 0                    # 먼 쪽이 표시돼 있어도 충동은 가장 가까운 이웃을 친다(조준을 안 쓴다)
    st.attack(0, s, w.fs, w.isp)
    assert s.energy[4] == pytest.approx(1.0 - cfg.attack_damage)
    assert s.energy[0] == pytest.approx(1.0 - cfg.attack_cost + cfg.attack_damage * cfg.attack_gain)
    assert s.energy[5] == 1.0
    # 가장 가까운 이웃이 표시돼 있어도 벌이 아니라 가로채기다(이득 attack_gain)
    s.energy[:] = 1.0
    s.mark_t[:] = -1
    s.mark_t[4] = 1
    st.attack(1, s, w.fs, w.isp)
    assert s.energy[0] == pytest.approx(1.0 - cfg.attack_cost + cfg.attack_damage * cfg.attack_gain)
    assert s.pun_t[4] == -1
    # 닿는 이웃이 없으면 빗나가 비용만 든다
    s.energy[:] = 1.0
    put(w, 4, 150.0, 150.0)
    put(w, 5, 150.0, 180.0)
    regrid(w)
    st.attack(2, s, w.fs, w.isp)
    assert s.energy[0] == pytest.approx(1.0 - cfg.attack_cost)


def test_impulse_does_not_touch_brain_output():
    w = _imp_scene("norm_size", norm_impulse=1.0)
    st.attack(0, w.s, w.fs, w.isp)
    assert w.s.act[0, cfgmod.ACT_ATTACK] == -1.0
    assert w.s.energy[4] < 1.0                  # 그래도 쳤다


def test_impulse_skips_sleeping_agents():
    w = _imp_scene("norm_size", norm_impulse=1.0)
    w.s.asleep[0] = 1.0
    st.attack(0, w.s, w.fs, w.isp)
    assert (w.s.energy == 1.0).all()


def test_impulse_counts_all_attacks_in_interval():
    w = _imp_scene("norm_size", norm_measure=True, norm_impulse=1.0)
    s = w.s
    s.act[0, cfgmod.ACT_ATTACK] = 0.5          # 뇌 출력이 이미 0 위여도 충동 구간 안이면 충동으로 센다
    st.attack(0, s, w.fs, w.isp)
    assert nt(w, 0, k.NT_TAKE_IMP) == 1 and nt(w, 0, k.NT_TAKE) == 1
    assert nt(w, 0, k.NT_ATK_VOL_PLAIN) == 0 and nt(w, 0, k.NT_ATK_VOL_MARKED) == 0
    w.fs[st.F_NORM_IMPULSE] = 0.0              # 충동 끔: 이제 자발 공격
    s.energy[:] = 1.0
    st.attack(1, s, w.fs, w.isp)
    assert nt(w, 0, k.NT_TAKE_IMP) == 1 and nt(w, 0, k.NT_TAKE) == 2
    assert nt(w, 0, k.NT_ATK_VOL_PLAIN) == 1 and nt(w, 0, k.NT_ATK_VOL_MARKED) == 0
    # 기회 분모는 자는 개체도 센다(0·4·5가 서로 닿는다: 스텝마다 셋)
    assert nt(w, 0, k.NT_OPP_PLAIN) == 6 and nt(w, 0, k.NT_OPP_MARKED) == 0


def test_impulse_world_needs_conflict_only_and_changes_hash():
    hashes = {}
    for p in (0.0, 0.05):
        w = World(small("stage4", "conflict", "norm_size", norm_impulse=p))
        w.run(300)
        hashes[p] = w.state_hash()
    w0 = World(small("stage4", "conflict", "norm_size"))
    w0.run(300)
    assert hashes[0.0] == w0.state_hash() and hashes[0.05] != w0.state_hash()
    w1 = World(small("stage1", "conflict", norm_impulse=0.015))     # 마음·사회 없이도 돈다
    w1.run(100)
    assert w1.s.counters[k.C_ATTACKS] > 0


# ======================= 과제 5: 표시(mark_t)와 원천 =======================
def test_mark_visibility_window_and_brightness():
    t0 = 10
    for tau in (1, 5, 49, 50):
        for e in range(0, tau + 3):
            marked = st._is_marked(t0 + e, t0, tau)
            assert marked == (e <= tau), (tau, e)
            b = st._bright(t0 + e, t0, tau)
            if e <= tau:
                assert b == pytest.approx(1.0 - max(0, e - 1) / tau)
            else:
                assert b == 0.0          # 경과 τ + 1에 정확히 없음(정수 시각이라 float 잔여가 없다)
        assert st._bright(t0, t0, tau) == 1.0 and st._bright(t0 + 1, t0, tau) == 1.0
        assert st._bright(t0 + tau, t0, tau) == pytest.approx(1.0 / tau)
    assert not st._is_marked(5, -1, 50) and st._bright(5, -1, 50) == 0.0


def _pair(*presets, **kw):
    w = hand_world(*presets, **kw)
    put(w, 0, 100.0, 100.0)
    put(w, 4, 101.5, 100.0)
    regrid(w)
    return w


def test_take_marks_only_the_taker():
    w = _pair("norm")
    s = w.s
    s.act[0, cfgmod.ACT_ATTACK] = 1.0
    st.attack(3, s, w.fs, w.isp)
    assert s.mark_t[0] == 3 and s.mark_vslot[0] == 4 and s.mark_vid[0] == s.ids[4] and s.mark_pun[0] == 0
    assert s.mark_t[4] == -1 and s.mark_vslot[4] == -2
    # 빗나간 공격(닿는 이웃 없음)은 표시가 안 붙는다
    s.mark_t[:] = -1
    put(w, 4, 150.0, 150.0)
    regrid(w)
    st.attack(4, s, w.fs, w.isp)
    assert (s.mark_t == -1).all()
    # 측정만 켠 세계·원천 none 세계는 가로채도 표시가 없다
    for kw in (dict(), ):
        wm = _pair("norm_size", norm_measure=True, **kw)
        wm.s.act[0, cfgmod.ACT_ATTACK] = 1.0
        st.attack(3, wm.s, wm.fs, wm.isp)
        assert wm.s.energy[4] < 1.0 and (wm.s.mark_t == -1).all()
    wn = _pair("norm", norm_mark_src="none")
    wn.s.act[0, cfgmod.ACT_ATTACK] = 1.0
    st.attack(3, wn.s, wn.fs, wn.isp)
    assert wn.s.energy[4] < 1.0 and (wn.s.mark_t == -1).all()


def test_remark_extends_same_episode():
    w = hand_world("norm", norm_mark_tau=5)
    s = w.s
    st._attach_mark(s, 0, 10, 4, 99, 5)                 # 표시가 없던 개체: 새 에피소드
    assert s.mark_t[0] == 10 and s.mark_pun[0] == 0 and s.mark_vid[0] == 99
    s.mark_pun[0] = 1                                    # 이 에피소드 안에 벌 받음
    st._attach_mark(s, 0, 15, 5, 77, 5)                 # 아직 표시된 동안(경과 5 ≤ τ): 같은 에피소드의 연장
    assert s.mark_t[0] == 15 and s.mark_pun[0] == 1 and s.mark_vslot[0] == 5 and s.mark_vid[0] == 77
    st._attach_mark(s, 0, 21, -1, -1, 5)                # 만료 뒤(경과 6 > τ): 새 에피소드
    assert s.mark_t[0] == 21 and s.mark_pun[0] == 0 and s.mark_vslot[0] == -1


def _sig_world(**kw):
    w = hand_world("norm", norm_mark_src="sig", **kw)
    w.s.act[:] = 0
    return w


def test_mark_source_sig():
    w = _sig_world()
    s = w.s
    s.act[0, cfgmod.ACT_SIG1] = 0.51           # 출력은 이미 tanh를 지난 값: tanh를 또 씌우면(0.47) 문턱을 못 넘는다
    s.act[1, cfgmod.ACT_SIG1] = 0.49
    s.act[2, cfgmod.ACT_SIG1] = 0.9
    s.asleep[2] = 1.0                          # 자는 개체는 안 붙는다
    s.act[3, cfgmod.ACT_SIG0] = 0.9            # 신호 1번은 원천이 아니다
    st.norm_mark_events(7, s, w.fs, w.isp)
    assert s.mark_t[0] == 7 and s.mark_vslot[0] == -1 and s.mark_vid[0] == -1
    assert s.mark_t[1] == -1 and s.mark_t[2] == -1 and s.mark_t[3] == -1
    # full 모드는 트리거 값에 위축을 곱한다: 0.6 × (1 − 0.8 × 1) = 0.12 < 0.5
    s.mark_t[:] = -1
    s.act[0, cfgmod.ACT_SIG1] = 0.6
    s.mem[0] = 1.0
    st.norm_mark_events(8, s, w.fs, w.isp)
    assert s.mark_t[0] == -1
    wi = _sig_world(norm_mode="inert")         # inert는 곱이 없다
    wi.s.act[0, cfgmod.ACT_SIG1] = 0.6
    wi.s.mem[0] = 1.0
    st.norm_mark_events(8, wi.s, wi.fs, wi.isp)
    assert wi.s.mark_t[0] == 8
    # take 원천에서는 신호가 표시를 안 붙인다
    wt = hand_world("norm")
    wt.s.act[0, cfgmod.ACT_SIG1] = 0.9
    st.norm_mark_events(8, wt.s, wt.fs, wt.isp)
    assert (wt.s.mark_t == -1).all()


def test_mark_noise_stateless():
    q = 0.3
    w = hand_world("norm", norm_mark_src="none", norm_mark_noise=q)
    s = w.s
    seed = w.cfg.seed
    hits = 0
    for t in range(20):
        st.norm_mark_events(t, s, w.fs, w.isp)
        for i in range(8):
            want = rngmod.u01(seed, rngmod.S_NORM_NOISE, t, s.ids[i]) < q
            assert (s.mark_t[i] == t) == want
            if want:
                assert s.mark_vslot[i] == -1 and s.mark_vid[i] == -1
            hits += want
    assert 20 < hits < 80
    # 같은 시나리오 두 번이 같다
    w2 = hand_world("norm", norm_mark_src="none", norm_mark_noise=q)
    for t in range(20):
        st.norm_mark_events(t, w2.s, w2.fs, w2.isp)
    assert (w2.s.mark_t == s.mark_t).all()


def test_mark_events_run_after_think_before_attack(monkeypatch):
    calls = []
    for name in ("after_think", "norm_mark_events", "attack"):
        orig = getattr(st, name)

        def wrap(*a, _o=orig, _n=name):
            calls.append(_n)
            return _o(*a)
        monkeypatch.setattr(st, name, wrap)
    w = World(small("stage4", "conflict", "norm"))
    w.run(3)
    assert calls == ["after_think", "norm_mark_events", "attack"] * 3
    calls.clear()
    for kw in (dict(), dict(norm_measure=True)):
        w0 = World(small("stage4", "conflict", "norm_size", **kw))
        w0.run(3)
    assert "norm_mark_events" not in calls


def test_marks_not_set_by_measure_only():
    w = World(small("stage4", "conflict", "norm_size", norm_measure=True))
    w.s.bout[:, cfgmod.ACT_ATTACK] = 0.5
    w.run(200)
    assert w.s.norm_tab[:, k.NT_TAKE].sum() > 0
    assert (w.s.mark_t == -1).all() and (w.s.pun_t == -1).all()


# ======================= 과제 6: 조준·벌 분기·벌금·inert =======================
def _tri(*presets, **kw):
    """0(공격자), 4(가장 가까운 닿는 이웃), 5(더 먼 닿는 이웃). 4·5는 서로 안 닿는다."""
    w = hand_world(*presets, **kw)
    put(w, 0, 100.0, 100.0)
    put(w, 4, 101.5, 100.0)
    put(w, 5, 100.0, 102.3)
    regrid(w)
    w.s.act[0, cfgmod.ACT_ATTACK] = 1.0
    return w


def test_pun_branch_pays_fine_not_take_gain():
    w = _pair("norm")
    s, cfg = w.s, w.cfg
    s.tag[0], s.tag[4] = 1, 2
    s.act[0, cfgmod.ACT_ATTACK] = 1.0
    s.mark_t[4] = 9                              # 4는 표시돼 있다(경과 1)
    lost, moved = s.ledger[k.L_LOST], s.ledger[k.L_MOVE]
    st.attack(10, s, w.fs, w.isp)
    dmg, fine = cfg.attack_damage, cfg.attack_damage * cfg.norm_pun_return
    assert s.energy[4] == pytest.approx(1.0 - dmg)
    assert s.energy[0] == pytest.approx(1.0 - cfg.attack_cost + fine)
    assert s.ledger[k.L_LOST] == pytest.approx(lost + dmg - fine)
    assert s.ledger[k.L_MOVE] == pytest.approx(moved + cfg.attack_cost)
    assert s.mark_t[0] == -1, "집행자는 표시되지 않는다"
    assert s.pun_t[4] == 10 and s.touch[4] == k.TOUCH_HIT and s.mark_pun[4] == 1
    assert nt(w, 1, k.NT_PUN) == 1 and nt(w, 1, k.NT_TAKE) == 0
    assert nt(w, 2, k.NT_PUN_DMG_RECV) == pytest.approx(dmg) and nt(w, 2, k.NT_VIOL_DMG_RECV) == 0
    assert nt(w, 1, k.NT_PUN_FINE) == pytest.approx(fine) and nt(w, 1, k.NT_PUN_COST) == pytest.approx(cfg.attack_cost)
    assert nt(w, 1, k.NT_ATK_COST) == pytest.approx(cfg.attack_cost)
    # 표시 안 된 대상이면 평소 가로채기이고 공격자가 표시된다
    s.energy[:] = 1.0
    s.mark_t[:] = -1
    st.attack(11, s, w.fs, w.isp)
    assert s.energy[0] == pytest.approx(1.0 - cfg.attack_cost + dmg * cfg.attack_gain)
    assert s.mark_t[0] == 11 and s.pun_t[4] == 10
    assert nt(w, 1, k.NT_TAKE) == 1 and nt(w, 2, k.NT_VIOL_DMG_RECV) == pytest.approx(dmg)


def test_pun_return_equal_to_attack_gain_is_plain_attack():
    w = _pair("norm", norm_pun_return=0.5)
    s, cfg = w.s, w.cfg
    s.act[0, cfgmod.ACT_ATTACK] = 1.0
    s.mark_t[4] = 10
    lost = s.ledger[k.L_LOST]
    st.attack(10, s, w.fs, w.isp)
    assert s.pun_t[4] == 10
    assert s.energy[0] == pytest.approx(1.0 - cfg.attack_cost + cfg.attack_damage * cfg.attack_gain)
    assert s.ledger[k.L_LOST] == pytest.approx(lost + cfg.attack_damage * (1 - cfg.attack_gain))


def test_pun_fine_capped_at_attack_gain_in_kernel():
    # 설정 검사가 norm_pun_return > attack_gain을 막지만 커널도 벌금을 dmg × attack_gain으로 자른다(이중 안전)
    w = _pair("norm")
    s, cfg = w.s, w.cfg
    w.fs[st.F_NORM_PUN_RET] = 0.9               # 검사를 우회한 값
    s.act[0, cfgmod.ACT_ATTACK] = 1.0
    s.mark_t[4] = 10
    lost = s.ledger[k.L_LOST]
    st.attack(10, s, w.fs, w.isp)
    assert s.pun_t[4] == 10
    assert s.energy[0] == pytest.approx(1.0 - cfg.attack_cost + cfg.attack_damage * cfg.attack_gain)
    assert s.ledger[k.L_LOST] == pytest.approx(lost + cfg.attack_damage * (1 - cfg.attack_gain))


def test_inert_mode_all_attacks_are_takes():
    w = _tri("norm", norm_mode="inert")
    s, cfg = w.s, w.cfg
    s.mark_t[5] = 20                             # 먼 쪽이 표시: 조준은 full과 같이 표시된 쪽
    s.mem[0] = 0.5
    st.attack(20, s, w.fs, w.isp)
    assert s.energy[5] == pytest.approx(1.0 - cfg.attack_damage) and s.energy[4] == 1.0
    assert s.energy[0] == pytest.approx(1.0 - cfg.attack_cost + cfg.attack_damage * cfg.attack_gain)
    assert s.pun_t[5] == -1 and s.mark_t[0] == 20 and s.mark_vslot[0] == 5
    assert nt(w, 0, k.NT_PUN) == 0 and nt(w, 0, k.NT_TAKE) == 1 and nt(w, 0, k.NT_TAKE_ON_MARKED) == 1
    assert nt(w, 0, k.NT_VIOL_DMG_RECV) == 0     # 표시 있던 대상을 친 피해는 위반 피해가 아니다
    assert s.mem[0] == 0.5 and s.mem[5] == 0     # 기억 갱신 없음


def test_aim_marked_prefers_marked_neighbor():
    w = _tri("norm")
    s = w.s
    s.mark_t[5] = 30
    st.attack(30, s, w.fs, w.isp)
    assert s.energy[5] < 1.0 and s.energy[4] == 1.0 and s.pun_t[5] == 30
    # 표시가 없으면 가장 가까운 쪽(기존과 같음)
    w2 = _tri("norm")
    st.attack(30, w2.s, w2.fs, w2.isp)
    assert w2.s.energy[4] < 1.0 and w2.s.energy[5] == 1.0
    # nearest는 기존 동작
    w3 = _tri("norm", norm_aim="nearest")
    w3.s.mark_t[5] = 30
    st.attack(30, w3.s, w3.fs, w3.isp)
    assert w3.s.energy[4] < 1.0 and w3.s.energy[5] == 1.0


def test_wrong_target_becomes_take_and_marks_me():
    w = _tri("norm", norm_aim="nearest")
    s, cfg = w.s, w.cfg
    s.mark_t[5] = 40
    st.attack(40, s, w.fs, w.isp)
    assert s.pun_t[4] == -1 and s.mark_t[0] == 40 and s.mark_vslot[0] == 4 and s.mark_vid[0] == s.ids[4]
    assert s.energy[0] == pytest.approx(1.0 - cfg.attack_cost + cfg.attack_damage * cfg.attack_gain)


def test_kill_counters_pun_vs_take():
    w = _pair("norm")
    s = w.s
    s.act[0, cfgmod.ACT_ATTACK] = 1.0
    s.mark_t[4] = 5
    s.energy[4] = 0.05
    st.attack(5, s, w.fs, w.isp)
    assert s.energy[4] == 0.0 and nt(w, 0, k.NT_PUN_KILL) == 1 and nt(w, 0, k.NT_TAKE_KILL) == 0
    w2 = _pair("norm")
    w2.s.act[0, cfgmod.ACT_ATTACK] = 1.0
    w2.s.energy[4] = 0.05
    st.attack(5, w2.s, w2.fs, w2.isp)
    assert nt(w2, 0, k.NT_TAKE_KILL) == 1 and nt(w2, 0, k.NT_PUN_KILL) == 0


def test_norm_world_lives_and_ledger_closes():
    w = World(small("stage4", "conflict", "norm", norm_impulse=0.015))
    w.s.bout[:, cfgmod.ACT_ATTACK] = 0.5          # 처음 개체가 자발로도 치게 해서 벌 분기를 실제로 돌린다
    w.run(1500)
    w.check_sane()
    assert rel_residual(w) < 1e-9
    tab = w.s.norm_tab
    assert tab[:, k.NT_PUN].sum() > 0 and tab[:, k.NT_TAKE].sum() > 0 and tab[:, k.NT_TAKE_IMP].sum() > 0
    assert w.summary()["pop"] > 10


def test_inert_isolated_hash_equals_off():
    hashes = []
    for kw in (dict(), dict(norm_enabled=True, norm_mode="inert", norm_impulse=0.0, norm_mark_src="none",
                            norm_mark_noise=0.0)):
        w = World(small("stage4", "conflict", "norm_size", **kw))
        w.s.bout[:, cfgmod.ACT_ATTACK] = 0.5
        w.run(300)
        hashes.append(w.state_hash())
        if kw:
            assert w.s.norm_tab[:, k.NT_TAKE].sum() > 0
    assert hashes[0] == hashes[1]


def test_cost_and_target_order_equivalent():
    # 대상 고르기를 비용 앞으로 옮겨도(막기 없는 세계) 설계 순서(비용 → 대상)의 손셈과 같다: 비용으로 내 에너지가
    # 0 아래로 가도 이미 차례를 받은 공격은 그대로 맞는다
    w = _pair("norm")
    s, cfg = w.s, w.cfg
    s.act[0, cfgmod.ACT_ATTACK] = 0.5
    s.energy[0] = 0.005
    s.mark_t[4] = 3
    lost, moved = s.ledger[k.L_LOST], s.ledger[k.L_MOVE]
    st.attack(3, s, w.fs, w.isp)
    cost, dmg = cfg.attack_cost * 0.5, cfg.attack_damage * 0.5
    fine = dmg * cfg.norm_pun_return
    assert s.energy[0] == pytest.approx(0.005 - cost + fine)
    assert s.energy[4] == pytest.approx(1.0 - dmg)
    assert s.ledger[k.L_MOVE] == pytest.approx(moved + cost) and s.ledger[k.L_LOST] == pytest.approx(lost + dmg - fine)
    assert nt(w, 0, k.NT_PUN_COST) == pytest.approx(cost) and nt(w, 0, k.NT_PUN_FINE) == pytest.approx(fine)


# ======================= 과제 7: 삼가기 기억·위축·norm_tick =======================
def test_mem_rises_on_punishment_and_decays():
    w = _pair("norm", norm_mem_rate=0.5)
    s = w.s
    s.act[0, cfgmod.ACT_ATTACK] = 1.0
    s.mark_t[4] = 10
    s.mem[4] = 0.2
    st.attack(10, s, w.fs, w.isp)
    assert s.pun_t[4] == 10 and s.mem[4] == pytest.approx(0.2 + 0.5 * 0.8)
    assert s.mem[0] == 0                          # 집행자는 안 오른다
    w1 = _pair("norm")                            # 기본 1.0: 한 번에 최대
    w1.s.act[0, cfgmod.ACT_ATTACK] = 1.0
    w1.s.mark_t[4] = 10
    st.attack(10, w1.s, w1.fs, w1.isp)
    assert w1.s.mem[4] == 1.0
    # norm_tick: 스텝마다 1/τ씩 줄고 τ번째에 정확히 0(float32 잔여를 자른다)
    for tau in (5, 49, 50):
        wt = hand_world("norm", norm_mem_tau=tau)
        st_ = wt.s
        st_.mem[:] = 0.0
        st_.mem[2] = 1.0
        st_.mem[3] = 0.3
        for q in range(tau - 1):
            st.norm_tick(st_, wt.fs, wt.isp)
            assert st_.mem[2] > 0.0, (tau, q)
            assert st_.mem[2] == pytest.approx(1.0 - (q + 1) / tau, abs=1e-5)
        st.norm_tick(st_, wt.fs, wt.isp)
        assert st_.mem[2] == 0.0 and st_.mem[3] == 0.0 and (st_.mem >= 0).all()


def test_inhibition_multiplies_takes_only():
    w = _pair("norm")
    s, cfg = w.s, w.cfg
    s.act[0, cfgmod.ACT_ATTACK] = 1.0
    s.mem[0] = 0.5
    st.attack(10, s, w.fs, w.isp)
    a_eff = 1.0 * (1 - cfg.norm_inhibit * 0.5)
    assert s.energy[4] == pytest.approx(1.0 - cfg.attack_damage * a_eff)
    # 충동 가로채기에도 곱해진다
    wi = _imp_scene("norm", norm_impulse=1.0)
    si = wi.s
    si.mem[0] = 1.0
    st.attack(0, si, wi.fs, wi.isp)
    assert si.energy[4] == pytest.approx(1.0 - cfg.attack_damage * (1 - cfg.norm_inhibit))
    # 벌에는 안 곱해진다(벌주는 쪽은 삼가지 않는다)
    wp = _pair("norm")
    wp.s.act[0, cfgmod.ACT_ATTACK] = 1.0
    wp.s.mem[0] = 1.0
    wp.s.mark_t[4] = 10
    st.attack(10, wp.s, wp.fs, wp.isp)
    assert wp.s.pun_t[4] == 10 and wp.s.energy[4] == pytest.approx(1.0 - cfg.attack_damage)
    # inert는 위축·기억 갱신이 없다
    wn = _pair("norm", norm_mode="inert")
    wn.s.act[0, cfgmod.ACT_ATTACK] = 1.0
    wn.s.mem[0] = 1.0
    wn.s.mark_t[4] = 10
    st.attack(10, wn.s, wn.fs, wn.isp)
    assert wn.s.energy[4] == pytest.approx(1.0 - cfg.attack_damage) and wn.s.mem[4] == 0.0


def test_inhibit_scales_damage_not_cost():
    w = _pair("norm")
    s, cfg = w.s, w.cfg
    s.act[0, cfgmod.ACT_ATTACK] = 1.0
    s.mem[0] = 1.0
    lost, moved = s.ledger[k.L_LOST], s.ledger[k.L_MOVE]
    st.attack(10, s, w.fs, w.isp)
    dmg = cfg.attack_damage * 0.2
    assert s.energy[4] == pytest.approx(1.0 - dmg)
    assert s.energy[0] == pytest.approx(1.0 - cfg.attack_cost + dmg * cfg.attack_gain)
    assert s.ledger[k.L_MOVE] == pytest.approx(moved + cfg.attack_cost)
    assert s.ledger[k.L_LOST] == pytest.approx(lost + dmg * (1 - cfg.attack_gain))
    assert nt(w, 0, k.NT_TAKE_GAIN) == pytest.approx(dmg * cfg.attack_gain)


def test_inhibit_applies_to_sig_mark_trigger():
    w = _sig_world()
    s = w.s
    s.act[0, cfgmod.ACT_SIG1] = 0.9
    s.mem[0] = 1.0                                # 0.9 × 0.2 = 0.18 < 0.5
    s.act[1, cfgmod.ACT_SIG1] = 0.9
    sig_before = s.act[:, cfgmod.ACT_SIG1].copy()
    st.norm_mark_events(5, s, w.fs, w.isp)
    assert s.mark_t[0] == -1 and s.mark_t[1] == 5
    assert (s.act[:, cfgmod.ACT_SIG1] == sig_before).all()    # 실제 신호 출력은 그대로
    # inert는 곱이 없다
    wi = _sig_world(norm_mode="inert")
    wi.s.act[0, cfgmod.ACT_SIG1] = 0.9
    wi.s.mem[0] = 1.0
    st.norm_mark_events(5, wi.s, wi.fs, wi.isp)
    assert wi.s.mark_t[0] == 5


def test_inhibit_zero_leaves_output_unchanged_memory_still_rises():
    w = _pair("norm", norm_inhibit=0.0)
    s, cfg = w.s, w.cfg
    s.act[0, cfgmod.ACT_ATTACK] = 1.0
    s.mem[0] = 1.0
    st.attack(10, s, w.fs, w.isp)
    assert s.energy[4] == pytest.approx(1.0 - cfg.attack_damage)
    # 벌 받으면 기억은 오른다
    s.energy[:] = 1.0
    s.act[0, cfgmod.ACT_ATTACK] = 0.0
    s.act[4, cfgmod.ACT_ATTACK] = 1.0             # 4가 표시된 0을 친다(0은 방금 가로채 표시됨)
    st.attack(11, s, w.fs, w.isp)
    assert s.pun_t[0] == 11 and s.mem[0] == 1.0
    s.mem[0] = 0.0
    s.mark_t[0] = 12
    st.attack(12, s, w.fs, w.isp)
    assert s.mem[0] == 1.0


def test_norm_tick_runs_before_norm_sense_slot(monkeypatch):
    calls = []
    for name in ("social_sense", "trail_sense", "norm_tick", "think_plastic"):
        orig = getattr(st, name)

        def wrap(*a, _o=orig, _n=name):
            calls.append(_n)
            return _o(*a)
        monkeypatch.setattr(st, name, wrap)
    w = World(small("stage4", "conflict", "norm"))
    w.run(2)
    assert calls == ["social_sense", "trail_sense", "norm_tick", "think_plastic"] * 2
    calls.clear()
    for kw in (dict(), dict(norm_measure=True)):
        World(small("stage4", "conflict", "norm_size", **kw)).run(2)
    assert "norm_tick" not in calls


def test_mem_not_inherited_by_default():
    w = World(small("stage4", "conflict", "norm"))
    s = w.s
    s.mem[s.alive] = 1.0
    born = 0
    for _ in range(300):
        before = s.ids.copy()
        alive_before = s.alive.copy()
        w.step()
        new = s.alive & (~alive_before | (s.ids != before))
        if new.any():
            # 새로 난 개체는 mem 0에서 시작해 이 스텝 norm_tick 뒤에도 0(감쇠만 있다)
            assert (s.mem[new] == 0).all()
            born += int(new.sum())
    assert born > 0


# ======================= 과제 8: 집행 역할·친족·에너지 표 =======================
def _pun_scene(**kw):
    """0(집행자)이 표시된 4를 친다. 6은 멀리 있는 피해자 후보."""
    w = _pair("norm", **kw)
    s = w.s
    s.act[0, cfgmod.ACT_ATTACK] = 1.0
    return w


def _punish(w, step, vslot, vid):
    s = w.s
    s.energy[:] = 1.0
    s.mark_t[4] = step
    s.mark_vslot[4] = vslot
    s.mark_vid[4] = vid
    st.attack(step, s, w.fs, w.isp)
    assert s.pun_t[4] == step


def test_no_kin_theta_key():
    assert "norm_kin_theta" not in Config().to_dict()


def test_pun_role_counts():
    w = _pun_scene()
    s, cfg = w.s, w.cfg
    fine, cost = cfg.attack_damage * cfg.norm_pun_return, cfg.attack_cost
    rt = s.norm_tab_role
    # 2자: 피해자가 집행자 자신
    _punish(w, 10, 0, s.ids[0])
    assert nt(w, 0, k.NT_PUN_2ND) == 1 and s.pun_role[4] == 0
    assert rt[0, k.RT_N_PUN] == 1 and rt[0, k.RT_PUN_FINE] == pytest.approx(fine)
    assert rt[0, k.RT_PUN_COST] == pytest.approx(cost)
    # 3자 친족(계보): 6이 0의 자식, 같은 색(닮음 구간 3)
    s.parent[6] = s.ids[0]
    _punish(w, 20, 6, s.ids[6])
    assert nt(w, 0, k.NT_PUN_3_KIN) == 1 and nt(w, 0, k.NT_PUN_3_LINE) == 1 and nt(w, 0, k.NT_PUN_3_SIM3) == 1
    assert s.pun_role[4] == 1 and rt[1, k.RT_N_PUN] == 1
    # 3자 남: 계보 없음, 색이 멀다(구간 0)
    s.parent[6] = -1
    s.color[6] = (0.8, 0.5, 0.5)
    _punish(w, 30, 6, s.ids[6])
    assert nt(w, 0, k.NT_PUN_3_STR) == 1 and nt(w, 0, k.NT_PUN_3_SIM0) == 1 and nt(w, 0, k.NT_PUN_3_LINE) == 1
    assert s.pun_role[4] == 2 and rt[2, k.RT_N_PUN] == 1
    # 닮음만 높은 남(0.93, 구간 2): pun_role·PUN_3_KIN은 닮음 상위 두 구간도 친족으로 친다(계보 칸은 안 오름)
    s.color[6] = (0.5 + 0.07 * 1.7320508075688772, 0.5, 0.5)
    _punish(w, 40, 6, s.ids[6])
    assert nt(w, 0, k.NT_PUN_3_KIN) == 2 and nt(w, 0, k.NT_PUN_3_SIM2) == 1 and nt(w, 0, k.NT_PUN_3_LINE) == 1
    assert s.pun_role[4] == 1 and rt[1, k.RT_N_PUN] == 2
    assert nt(w, 0, k.NT_PUN) == 4 and rt[:, k.RT_N_PUN].sum() == 4


def test_pun_unknown_novict_vgone():
    w = _pun_scene()
    s = w.s
    _punish(w, 10, -1, -1)                       # sig·잡음 표시: 피해자 없음
    assert nt(w, 0, k.NT_PUN_NOVICT) == 1 and s.pun_role[4] == -1
    _punish(w, 20, -2, -2)                       # 이어 읽은 직후 등: 피해자 미지
    assert nt(w, 0, k.NT_PUN_UNKNOWN) == 1 and s.pun_role[4] == -1
    vid6 = s.ids[6]
    s.alive[6] = False                           # 피해자가 죽었다
    _punish(w, 30, 6, vid6)
    assert nt(w, 0, k.NT_PUN_VGONE) == 1 and s.pun_role[4] == -1
    s.alive[6] = True
    s.ids[6] = vid6 + 1000                       # 그 칸이 다른 개체로 바뀌었다(칸 재사용)
    _punish(w, 40, 6, vid6)
    assert nt(w, 0, k.NT_PUN_VGONE) == 2 and s.pun_role[4] == -1
    assert nt(w, 0, k.NT_PUN_2ND) + nt(w, 0, k.NT_PUN_3_KIN) + nt(w, 0, k.NT_PUN_3_STR) == 0
    assert not s.norm_tab_role[:, k.RT_N_PUN].any()


def test_opportunity_counts_self_other_sim_line():
    w = _pair("norm")
    s = w.s
    s.mark_t[4] = 5
    s.mark_vslot[4], s.mark_vid[4] = 0, s.ids[0]          # 4가 0에게서 가로챘다
    st.attack(5, s, w.fs, w.isp)                          # 아무도 안 친다: 훑기만
    assert nt(w, 0, k.NT_OPP_SELF) == 1 and nt(w, 0, k.NT_OPP_OTHER) == 0
    assert nt(w, 0, k.NT_OPP_MARKED) == 1 and nt(w, 0, k.NT_OPP_PLAIN) == 1    # 4의 차례: 0은 표시 안 됨
    s.parent[6] = s.ids[0]                                # 피해자 6은 0의 자식(같은 색)
    s.mark_vslot[4], s.mark_vid[4] = 6, s.ids[6]
    st.attack(6, s, w.fs, w.isp)
    assert nt(w, 0, k.NT_OPP_OTHER) == 1 and nt(w, 0, k.NT_OPP_OTHER_LINE) == 1
    assert nt(w, 0, k.NT_OPP_OTHER_SIM3) == 1 and nt(w, 0, k.NT_OPP_SELF) == 1
    # 피해자 없음·미지·죽음은 어느 분모에도 안 든다
    for vs, vid in ((-1, -1), (-2, -2)):
        s.mark_vslot[4], s.mark_vid[4] = vs, vid
        st.attack(7, s, w.fs, w.isp)
    assert nt(w, 0, k.NT_OPP_OTHER) == 1 and nt(w, 0, k.NT_OPP_SELF) == 1 and nt(w, 0, k.NT_OPP_MARKED) == 4
    # 실제 세계에서 p2·p3 ≤ 1(분모가 분자의 기회 전체)
    wr = World(small("stage4", "conflict", "norm", norm_impulse=0.02))
    wr.s.bout[:, cfgmod.ACT_ATTACK] = 0.5
    wr.run(500)
    tot = wr.s.norm_tab.sum(axis=0)
    assert tot[k.NT_PUN] > 0 and tot[k.NT_OPP_SELF] + tot[k.NT_OPP_OTHER] > 0
    assert tot[k.NT_PUN_2ND] <= tot[k.NT_OPP_SELF]
    assert tot[k.NT_PUN_3_KIN] + tot[k.NT_PUN_3_STR] <= tot[k.NT_OPP_OTHER]
    sims = sum(tot[k.NT_OPP_OTHER_SIM0 + b] for b in range(4))
    assert sims == tot[k.NT_OPP_OTHER] and tot[k.NT_OPP_OTHER_LINE] <= tot[k.NT_OPP_OTHER]


def test_selectivity_denominators():
    w = hand_world("norm")
    s = w.s
    put(w, 0, 100.0, 100.0)
    put(w, 1, 101.5, 100.0)                  # 0의 이웃(표시됨)
    put(w, 2, 150.0, 150.0)
    put(w, 3, 151.5, 150.0)                  # 2의 이웃(표시 안 됨)
    regrid(w)
    s.mark_t[1] = 3
    s.mark_vslot[1], s.mark_vid[1] = 7, s.ids[7]
    s.act[0, cfgmod.ACT_ATTACK] = 1.0
    s.act[2, cfgmod.ACT_ATTACK] = 1.0
    st.attack(3, s, w.fs, w.isp)
    # 0: 표시된 기회 1·자발 공격 1. 1: 닿는 0이 표시 안 됨(plain). 2: plain 기회 1·자발 공격 1. 3: 닿는 2가 방금 표시됐다면
    # 순서에 따라 marked가 될 수 있어 둘의 합만 본다
    om, op = nt(w, 0, k.NT_OPP_MARKED), nt(w, 0, k.NT_OPP_PLAIN)
    am, ap = nt(w, 0, k.NT_ATK_VOL_MARKED), nt(w, 0, k.NT_ATK_VOL_PLAIN)
    assert om + op == 4 and am == 1 and ap == 1 and om >= 1 and op >= 2
    sel = (am / om) / (ap / op)
    assert sel == pytest.approx((1 / om) / (1 / op))


def test_energy_tables_by_tag_and_role():
    w = World(small("stage4", "conflict", "norm", norm_impulse=0.02))
    s = w.s
    s.bout[:, cfgmod.ACT_ATTACK] = 0.5
    s.tag[s.alive] = np.arange(int(s.alive.sum())) % 6 - 1      # 태그 −1~4: 접기(행 4)가 안 넘친다
    w.run(200)
    checked = 0
    for _ in range(40):
        w.step()
        t = w.step_no
        led, tab = s.ledger.copy(), s.norm_tab.copy()
        st.attack(t, s, w.fs, w.isp)
        d = s.norm_tab - tab
        tot = d.sum(axis=0)
        assert s.ledger[k.L_MOVE] - led[k.L_MOVE] == pytest.approx(tot[k.NT_ATK_COST], abs=1e-9)
        lost = tot[k.NT_TAKE_DMG_RECV] + tot[k.NT_PUN_DMG_RECV] - tot[k.NT_TAKE_GAIN] - tot[k.NT_PUN_FINE]
        assert s.ledger[k.L_LOST] - led[k.L_LOST] == pytest.approx(lost, abs=1e-9)
        checked += tot[k.NT_PUN] > 0
    tot = s.norm_tab.sum(axis=0)
    rt = s.norm_tab_role
    assert tot[k.NT_PUN] > 0 and checked > 0
    roles = tot[k.NT_PUN_2ND] + tot[k.NT_PUN_3_KIN] + tot[k.NT_PUN_3_STR]
    assert rt[:, k.RT_N_PUN].sum() == roles
    assert rt[:, k.RT_PUN_FINE].sum() <= tot[k.NT_PUN_FINE] + 1e-12
    assert rt[:, k.RT_PUN_COST].sum() <= tot[k.NT_PUN_COST] + 1e-12
    assert s.norm_tab.shape[0] == k.TAG_ROWS_NORM
    assert set(np.unique(s.pun_role[s.alive])) <= {-1, 0, 1, 2}


def test_summary_norm_fields():
    w = World(small("stage4", "conflict", "norm", norm_impulse=0.02))
    w.s.bout[:, cfgmod.ACT_ATTACK] = 0.5
    w.run(400)
    out = w.summary()
    tot = w.s.norm_tab.sum(axis=0)
    assert tot[k.NT_OPP_SELF] > 0 and tot[k.NT_OPP_OTHER] > 0
    assert out["norm_pun_rate_2nd"] == pytest.approx(tot[k.NT_PUN_2ND] / tot[k.NT_OPP_SELF])
    assert out["norm_pun_rate_3rd"] == pytest.approx((tot[k.NT_PUN_3_KIN] + tot[k.NT_PUN_3_STR]) / tot[k.NT_OPP_OTHER])
    wm = World(small("stage4", "conflict", "norm_size", norm_measure=True))
    wm.run(20)
    assert "norm_pun_rate_2nd" not in wm.summary() and "norm_pun_rate_3rd" not in wm.summary()


# ======================= 과제 9: 구간표·표시 에피소드·유병률·죽을 때 닫기 =======================
def test_interval_table_boundaries_and_exclusions():
    w = _pair("norm")
    s = w.s
    h = s.norm_hist
    s.pun_t[0], s.pun_role[0] = 100, 2           # 0은 3자 남에게 벌 받았다
    st.attack(100, s, w.fs, w.isp)               # 경과 0: 자기 차례가 와도 표에서 뺀다
    assert not h.any()
    want = {101: 0, 109: 0, 110: 1, 149: 1, 150: 2, 299: 2, 300: 3, 1000: 3}     # 경과 = t − 100
    expo = np.zeros(4)
    for t, b in want.items():
        st.attack(t, s, w.fs, w.isp)
        expo[b] += 1
        assert (h[2, :, 0] == expo).all(), t
    assert not h[[0, 1, 3]].any() and not h[2, :, 1].any()
    # 가로채기는 차례를 받을 때의 칸에 센다(친 뒤 내가 표시돼도 행은 벌의 역할 그대로)
    s.act[0, cfgmod.ACT_ATTACK] = 1.0
    st.attack(120, s, w.fs, w.isp)
    assert h[2, 1, 1] == 1 and h[2, 1, 0] == expo[1] + 1 and s.mark_t[0] == 120
    # 표시됐으나 벌 안 받은 개체(행 3): 경과는 표시가 붙은 뒤로
    w3 = _pair("norm")
    s3 = w3.s
    s3.mark_t[4], s3.mark_vslot[4], s3.mark_vid[4] = 50, 6, s3.ids[6]
    st.attack(50, s3, w3.fs, w3.isp)             # 경과 0은 뺀다
    st.attack(55, s3, w3.fs, w3.isp)
    st.attack(70, s3, w3.fs, w3.isp)
    assert s3.norm_hist[3, 0, 0] == 1 and s3.norm_hist[3, 1, 0] == 1 and s3.norm_hist[3].sum() == 2
    assert not s3.norm_hist[:3].any()
    # 벌 받았으나 역할을 가를 수 없는 개체(pun_role −1)는 표에서 뺀다. 측정만 켠 세계는 표가 늘 0
    w4 = _pair("norm")
    w4.s.pun_t[0], w4.s.pun_role[0] = 10, -1
    st.attack(20, w4.s, w4.fs, w4.isp)
    assert not w4.s.norm_hist.any()
    wm = World(small("stage4", "conflict", "norm_size", norm_measure=True))
    wm.s.bout[:, cfgmod.ACT_ATTACK] = 0.5
    wm.run(100)
    assert not wm.s.norm_hist.any()


def _epis(w, kind):
    return nt(w, 0, k.NT_MARK_EPIS0 + kind), nt(w, 0, k.NT_MARK_EPIS_PUN0 + kind)


def test_prevalence_and_episode_counts():
    w = hand_world("norm", norm_mark_tau=5)
    s = w.s
    st._attach_mark(s, 0, 10, 4, s.ids[4], 5)              # 종류 0(가로채기)
    st._attach_mark(s, 1, 10, -1, -1, 5)                   # 종류 1(피해자 없음)
    s.mark_pun[1] = 1
    s.mark_t[2], s.mark_vid[2] = 10, -2                    # 종류 2(미지)
    for t in range(10, 16):
        st.norm_account(t, s, w.fs, w.isp)
        assert _epis(w, 0) == (0, 0)
    assert nt(w, 0, k.NT_MARKED_STEPS) == 3 * 6            # 붙은 스텝과 그 뒤 τ스텝
    st.norm_account(16, s, w.fs, w.isp)                    # 만료 스텝(경과 τ + 1)에 닫는다
    assert _epis(w, 0) == (1, 0) and _epis(w, 1) == (1, 1) and _epis(w, 2) == (1, 0)
    for t in range(17, 40):
        st.norm_account(t, s, w.fs, w.isp)
    assert _epis(w, 0) == (1, 0) and nt(w, 0, k.NT_MARKED_STEPS) == 18      # 한 번만 센다


def test_punished_then_remarked_counts_as_punished():
    w = hand_world("norm", norm_mark_tau=5)
    s = w.s
    st._attach_mark(s, 0, 10, 4, s.ids[4], 5)
    s.mark_pun[0] = 1                                      # 벌 받음
    s.pun_t[0] = 11
    st._attach_mark(s, 0, 13, 4, s.ids[4], 5)              # 다시 가로챔: 같은 에피소드의 연장(pun_t < mark_t)
    for t in range(10, 30):
        st.norm_account(t, s, w.fs, w.isp)
    assert _epis(w, 0) == (1, 1)


def test_remark_at_expiry_step_closes_previous_episode():
    # 만료 스텝(경과 τ + 1)에 다시 붙으면 norm_account가 그 스텝 끝에 못 보므로 붙이는 쪽이 앞 에피소드를 닫는다
    w = hand_world("norm", norm_mark_tau=5)
    s = w.s
    st._attach_mark(s, 0, 10, 4, s.ids[4], 5)
    s.mark_pun[0] = 1
    for t in range(10, 16):
        st.norm_account(t, s, w.fs, w.isp)
    st._attach_mark(s, 0, 16, -1, -1, 5)                   # 만료 스텝에 재표시(잡음): 새 에피소드
    assert _epis(w, 0) == (1, 1) and s.mark_pun[0] == 0
    st.norm_account(16, s, w.fs, w.isp)
    assert _epis(w, 0) == (1, 1) and _epis(w, 1) == (0, 0)  # 두 번 안 센다
    for t in range(17, 30):
        st.norm_account(t, s, w.fs, w.isp)
    assert _epis(w, 1) == (1, 0)
    # 같은 스텝에 두 번 붙어도(잡음 뒤 가로채기) 앞 에피소드는 한 번만
    w2 = hand_world("norm", norm_mark_tau=5)
    s2 = w2.s
    st._attach_mark(s2, 0, 10, 4, s2.ids[4], 5)
    st._attach_mark(s2, 0, 16, -1, -1, 5)
    st._attach_mark(s2, 0, 16, 4, s2.ids[4], 5)
    assert _epis(w2, 0) == (1, 0) and s2.mark_vid[0] == s2.ids[4]
    # 이미 만료돼 닫힌 뒤(경과 > τ + 1)에 붙으면 앞 것을 또 세지 않는다
    w3 = hand_world("norm", norm_mark_tau=5)
    s3 = w3.s
    st._attach_mark(s3, 0, 10, 4, s3.ids[4], 5)
    for t in range(10, 20):
        st.norm_account(t, s3, w3.fs, w3.isp)
    st._attach_mark(s3, 0, 20, 4, s3.ids[4], 5)
    assert _epis(w3, 0) == (1, 0)


def test_episode_closed_at_death_no_double_count():
    w = hand_world("norm", norm_mark_tau=5)
    s, fp, ip = w.s, w.fp, w.ip
    t = 40
    s.mark_t[0], s.mark_vid[0], s.mark_pun[0] = 38, s.ids[4], 1     # 표시된 채 맞아 죽음
    s.energy[0], s.touch[0] = 0.0, k.TOUCH_HIT
    s.mark_t[1], s.mark_vid[1] = t - 6, -1                           # 이 스텝 만료이고 굶어 죽음
    s.energy[1] = 0.0
    s.mark_t[2], s.mark_vid[2] = t - 1, -2                           # 표시된 채 노쇠
    s.age[2] = w.cfg.max_age
    s.energy[3] = 0.0                                                # 표시 없이 굶음
    s.eaten[:4] = (1.5, 2.0, 3.0, 0.25)
    st.norm_account(t, s, w.fs, w.isp)
    assert _epis(w, 1) == (1, 0)
    nd = k.process_deaths(t, s, fp, ip)
    assert nd == 4 and not s.alive[:4].any()
    assert _epis(w, 0) == (1, 1) and _epis(w, 1) == (1, 0) and _epis(w, 2) == (1, 0)
    assert nt(w, 0, k.NT_LIVES_DONE) == 4 and nt(w, 0, k.NT_EATEN_DEAD) == pytest.approx(6.75)


def test_death_hook_records_eaten_for_net_income():
    w = hand_world("norm")
    s = w.s
    s.tag[0], s.tag[1] = 2, 7                     # 태그 행(7은 행 4에 접힌다)
    s.energy[0] = s.energy[1] = 0.0
    s.eaten[0], s.eaten[1] = 4.0, 1.0
    k.process_deaths(5, s, w.fp, w.ip)
    assert nt(w, 2, k.NT_EATEN_DEAD) == 4.0 and nt(w, 2, k.NT_LIVES_DONE) == 1
    assert nt(w, 4, k.NT_EATEN_DEAD) == 1.0 and nt(w, 4, k.NT_LIVES_DONE) == 1


def test_account_off_not_called(monkeypatch):
    calls = []
    for mod, name in ((st, "attack"), (st, "norm_account"), (k, "process_deaths")):
        orig = getattr(mod, name)

        def wrap(*a, _o=orig, _n=name):
            calls.append(_n)
            return _o(*a)
        monkeypatch.setattr(mod, name, wrap)
    World(small("stage4", "conflict", "norm")).run(3)
    assert calls == ["attack", "norm_account", "process_deaths"] * 3
    calls.clear()
    for kw in (dict(), dict(norm_measure=True)):
        World(small("stage4", "conflict", "norm_size", **kw)).run(3)
    assert "norm_account" not in calls
    # process_deaths 훅(njit)은 감쌀 수 없다 → 측정만 켠 세계에서 죽어도 훅의 계수가 0인지로 본다
    wm = hand_world("norm_size", norm_measure=True)
    wm.s.energy[:3] = 0.0
    wm.s.mark_t[0] = 0
    assert k.process_deaths(1, wm.s, wm.fp, wm.ip) == 3
    assert not wm.s.norm_tab.any()


def test_summary_prevalence():
    w = World(small("stage4", "conflict", "norm", norm_impulse=0.02))
    w.s.bout[:, cfgmod.ACT_ATTACK] = 0.5
    w.run(300)
    out = w.summary()
    tot = w.s.norm_tab.sum(axis=0)
    # 미지(종류 2: 이어 읽기 전에 붙은 표시)는 벌 여부를 모르므로 표시당 집행 확률에서 뺀다
    epis = sum(tot[k.NT_MARK_EPIS0 + q] for q in range(2))
    epun = sum(tot[k.NT_MARK_EPIS_PUN0 + q] for q in range(2))
    assert tot[k.NT_MARKED_STEPS] > 0 and epis > 0
    assert out["norm_mark_prevalence"] == pytest.approx(tot[k.NT_MARKED_STEPS] / tot[k.NT_ALIVE_STEPS])
    assert out["norm_punished_per_episode"] == pytest.approx(epun / epis)
    wm = World(small("stage4", "conflict", "norm_size", norm_measure=True))
    wm.run(20)
    assert "norm_mark_prevalence" not in wm.summary() and "norm_punished_per_episode" not in wm.summary()


# ======================= 과제 10: 감각 블록(_see_norm) =======================
def _sense(w, t):
    """세계의 감각 순서대로: 격자 → 본 감각 → norm_sense. 본 감각만 지난 obs 사본도 돌려준다."""
    s = w.s
    regrid(w)
    k.sense(t, s, w.fp, w.ip)
    before = s.obs.copy()
    st.norm_sense(t, s, w.fs, w.isp)
    return before


def _ncols(w):
    c0 = w.cfg.norm_in0
    return c0, w.cfg.vision_sectors


def _sense_scene(**kw):
    w = hand_world("norm", **kw)
    s = w.s
    s.heading[:] = 0.0
    put(w, 0, 100.0, 100.0)
    return w


def test_norm_vis_direction_and_brightness():
    w = _sense_scene()
    s = w.s
    c0, ns = _ncols(w)
    put(w, 4, 110.0, 102.0)
    s.mark_t[4] = 97                                         # t = 100: 경과 3, 밝기 1 − 2/50
    before = _sense(w, 100)
    vis = s.obs[0, c0 + cfgmod.NORM_VIS:c0 + cfgmod.NORM_VIS + 8]
    ch1 = before[0, ns:2 * ns]
    assert (ch1 > 0).any(), "양성 확인: 본 시야가 그 이웃을 본다"
    assert ((vis != 0) == (ch1 != 0)).all()
    assert vis == pytest.approx(ch1 * (1 - 2 / 50), abs=1e-6)
    s.mark_t[4] = -1
    _sense(w, 100)
    assert not s.obs[0, c0:c0 + 8].any()


def test_norm_vis_equals_closeness_times_brightness():
    for (dx, dy, e) in ((6.0, 0.0, 0), (12.0, 4.0, 1), (9.0, -5.0, 30), (14.5, 0.5, 50)):
        w = _sense_scene()
        s = w.s
        c0, ns = _ncols(w)
        put(w, 4, 100.0 + dx, 100.0 + dy)
        s.mark_t[4] = 200 - e
        before = _sense(w, 200)
        b = st._bright(200, 200 - e, 50)
        ch1 = before[0, ns:2 * ns]
        assert (ch1 > 0).any()
        assert s.obs[0, c0:c0 + 8] == pytest.approx(ch1 * b, abs=1e-6), (dx, dy, e)


def test_norm_vis_unmarked_near_does_not_occlude_marked_far():
    w = _sense_scene()
    s = w.s
    c0, ns = _ncols(w)
    put(w, 4, 106.0, 100.0)                                   # 가깝고 표시 없음
    put(w, 5, 113.0, 100.0)                                   # 멀고 표시됨, 같은 방향
    s.mark_t[5] = 100
    before = _sense(w, 100)
    vis = s.obs[0, c0:c0 + 8]
    assert vis.max() > 0
    # 같은 칸의 본 시야는 가까운 4의 가까움, 표시 시야는 먼 5의 가까움(× 밝기 1)
    sec = int(np.argmax(vis))
    assert before[0, ns + sec] > vis[sec]
    assert vis[sec] == pytest.approx(1.0 - (13.0 - w.cfg.agent_radius) / w.cfg.vision_range, abs=1e-6)
    # 순서 무관: 칸마다 최댓값(먼 표시 + 가까운 표시면 가까운 쪽)
    s.mark_t[4] = 100
    _sense(w, 100)
    assert s.obs[0, c0 + sec] == pytest.approx(1.0 - (6.0 - w.cfg.agent_radius) / w.cfg.vision_range, abs=1e-6)


def test_norm_vis_leaves_ordinary_vision_untouched():
    w = World(small("stage4", "conflict", "norm"))
    s = w.s
    a = np.flatnonzero(s.alive)
    s.mark_t[a[::2]] = 0
    s.pun_t[a[::3]] = 0
    s.mem[a[::5]] = 0.7
    before = _sense(w, 1)
    c0 = w.cfg.norm_in0
    cols = np.r_[0:c0, c0 + cfgmod.N_NORM_IN:s.obs.shape[1]]
    assert (s.obs[:, cols] == before[:, cols]).all()
    assert s.obs[a, c0:c0 + 8].any() and s.obs[a, c0 + cfgmod.NORM_MEM].any()


def test_see_norm_sector_span_matches_see():
    gen = np.random.default_rng(5)
    hits = 0
    for _ in range(40):
        w = _sense_scene()
        s = w.s
        c0, ns = _ncols(w)
        ang = gen.uniform(-np.pi, np.pi)
        d = gen.uniform(0.5, 15.5)
        put(w, 4, 100.0 + d * np.cos(ang), 100.0 + d * np.sin(ang))
        s.mark_t[4] = 10
        before = _sense(w, 10)
        assert ((s.obs[0, c0:c0 + 8] != 0) == (before[0, ns:2 * ns] != 0)).all()
        hits += (before[0, ns:2 * ns] != 0).any()
    assert hits > 5


def test_norm_vis_torus_and_self_excluded():
    w = _sense_scene()
    s = w.s
    c0, ns = _ncols(w)
    put(w, 0, 2.0, 100.0)
    s.heading[0] = np.pi                                       # −x쪽을 본다
    put(w, 4, 250.0, 100.0)                                    # 토러스 반대편(간격 8)
    s.mark_t[4] = 50
    s.mark_t[0] = 50                                           # 자기 표시는 NORM_SELF로만
    _sense(w, 50)
    assert s.obs[0, c0:c0 + 8].max() == pytest.approx(1.0 - (8.0 - w.cfg.agent_radius) / w.cfg.vision_range, abs=1e-6)
    assert s.obs[0, c0 + cfgmod.NORM_SELF] == 1.0
    s.mark_t[4] = -1
    _sense(w, 50)
    assert not s.obs[0, c0:c0 + 8].any()


def test_norm_scalars():
    w = _sense_scene()
    s, cfg = w.s, w.cfg
    c0, _ = _ncols(w)
    hr = cfg.hear_range
    put(w, 4, 110.0, 100.0)            # 거리 10
    put(w, 5, 100.0, 80.0)             # 거리 20(시야 밖, 들리는 거리 안)
    put(w, 6, 100.0, 130.0)            # 거리 30(들리는 거리 밖)
    t = 300
    s.mark_t[0] = t - 11               # 내 밝기 1 − 10/50
    s.mark_t[4] = t - 1                # 밝기 1
    s.mark_t[5] = t - 26               # 밝기 0.5
    s.mark_t[6] = t                    # 밖이라 무시
    s.pun_t[4] = t - 2                 # 방금(1스텝 전)이 아니다
    s.pun_t[5] = t - 1                 # 방금 벌 받음
    s.pun_t[6] = t - 1
    s.mem[0] = 0.375
    _sense(w, t)
    w4, w5 = 1 - 10 / hr, 1 - 20 / hr
    o = s.obs[0, c0:c0 + cfgmod.N_NORM_IN]
    assert o[cfgmod.NORM_SELF] == pytest.approx(0.8)
    assert o[cfgmod.NORM_LOCAL] == pytest.approx((w4 * 1.0 + w5 * 0.5) / (w4 + w5), abs=1e-6)
    assert o[cfgmod.NORM_SAWPUN] == pytest.approx(w5, abs=1e-6)
    assert o[cfgmod.NORM_MEM] == 0.375
    # 이웃이 없으면 곁 값은 0
    assert s.obs[1, c0 + cfgmod.NORM_LOCAL] == 0 and s.obs[1, c0 + cfgmod.NORM_SAWPUN] == 0


def test_sleeping_norm_vis_zero_scalars_kept():
    w = _sense_scene()
    s = w.s
    assert w.cfg.mind_enabled
    c0, _ = _ncols(w)
    put(w, 4, 108.0, 100.0)
    s.mark_t[4] = 100
    s.mark_t[0] = 100
    s.mem[0] = 0.5
    _sense(w, 100)
    assert s.obs[0, c0:c0 + 8].any()
    s.asleep[0] = 5.0
    st.mind_update(s, w.fs, w.isp)
    assert not s.obs[0, c0:c0 + 8].any()
    assert s.obs[0, c0 + cfgmod.NORM_SELF] == 1.0 and s.obs[0, c0 + cfgmod.NORM_MEM] == 0.5


def test_norm_block_all_zero_when_off(monkeypatch):
    calls = []
    orig = st.norm_sense

    def wrap(*a):
        calls.append(1)
        return orig(*a)
    monkeypatch.setattr(st, "norm_sense", wrap)
    for kw in (dict(), dict(norm_measure=True)):
        w = World(small("stage4", "conflict", "norm_size", **kw))
        w.run(3)
        c0 = w.cfg.norm_in0
        assert not w.s.obs[:, c0:c0 + cfgmod.N_NORM_IN].any()
    assert not calls
    World(small("stage4", "conflict", "norm")).run(2)
    assert len(calls) == 2


def test_norm_sense_thread_determinism():
    import numba
    w = World(small("stage4", "conflict", "norm", norm_impulse=0.05))
    w.s.bout[:, cfgmod.ACT_ATTACK] = 0.5
    w.run(80)
    s = w.s
    assert (s.mark_t[s.alive] >= 0).any()
    outs = []
    old = numba.get_num_threads()
    try:
        for nthr in sorted({1, numba.config.NUMBA_NUM_THREADS}):
            numba.set_num_threads(nthr)
            s.obs[:] = 0
            st.norm_sense(w.step_no, s, w.fs, w.isp)
            outs.append(s.obs.copy())
    finally:
        numba.set_num_threads(old)
    assert len(outs) > 1, "스레드 1과 여럿을 견줘야 한다(NUMBA_NUM_THREADS ≥ 2로 돌린다)"
    for o in outs[1:]:
        assert (o == outs[0]).all()


def test_norm_sense_only_reads_state():
    w = World(small("stage4", "conflict", "norm", norm_impulse=0.05))
    w.s.bout[:, cfgmod.ACT_ATTACK] = 0.5
    w.run(80)
    s = w.s
    snap = (s.mark_t.copy(), s.pun_t.copy(), s.mem.copy())
    st.norm_sense(w.step_no, s, w.fs, w.isp)
    assert (s.mark_t == snap[0]).all() and (s.pun_t == snap[1]).all() and (s.mem == snap[2]).all()


# ======================= 과제 12: watch_pun·눈가림·attack 뒤 격자 재구성 =======================
def _watch_scene(**kw):
    w = hand_world("norm", **kw)
    s = w.s
    s.mem[:] = 0.0
    put(w, 0, 7.8, 100.0)                    # 벌 받은 개체(칸 경계 8 바로 안쪽)
    s.pun_t[0] = 50
    return w


def test_watch_pun_raises_neighbor_mem_within_radius():
    w = _watch_scene()
    s, cfg = w.s, w.cfg
    r = cfg.soc_radius
    put(w, 1, 8.2, 100.0)                    # 칸 경계 건너 0.4
    put(w, 2, 7.8 + r - 0.01, 100.0)         # 반지름 바로 안
    put(w, 3, 7.8 + r + 0.01, 100.0)         # 반지름 바로 밖
    put(w, 4, 255.9, 100.0)                  # 토러스 반대편(거리 7.9)
    put(w, 5, 7.8, 100.0 + r - 0.01)         # 아래 칸 경계 건너
    put(w, 6, 9.0, 101.0)                    # 죽은 이웃
    s.alive[6] = False
    regrid(w)
    st.watch_pun(50, s, w.fs, w.isp)
    obs = cfg.norm_mem_obs
    assert s.mem[1] == pytest.approx(obs) and s.mem[2] == pytest.approx(obs) and s.mem[4] == pytest.approx(obs)
    assert s.mem[5] == pytest.approx(obs)
    assert s.mem[3] == 0.0 and s.mem[6] == 0.0
    assert s.mem[0] == 0.0, "벌 받은 본인은 여기서 안 오른다(norm_mem_rate로 이미 올랐다)"
    assert nt(w, 0, k.NT_SAW_PUN) == 4
    # 이번 스텝의 벌이 아니면(pun_t != t) 아무 일도 없다
    st.watch_pun(51, s, w.fs, w.isp)
    assert s.mem[1] == pytest.approx(obs) and nt(w, 0, k.NT_SAW_PUN) == 4


def test_watch_pun_order_is_deterministic():
    hashes = []
    for _ in range(2):
        w = _watch_scene()
        s = w.s
        put(w, 1, 10.0, 100.0)
        put(w, 2, 12.0, 100.0)               # 1은 0과 2 둘의 벌을 본다
        s.pun_t[2] = 50
        s.mem[1] = 0.5
        regrid(w)
        st.watch_pun(50, s, w.fs, w.isp)
        assert s.mem[1] == pytest.approx(1 - 0.5 * 0.8 * 0.8)
        assert s.mem[0] == pytest.approx(0.2) and s.mem[2] == pytest.approx(0.2)    # 서로의 벌을 본다
        hashes.append(w.state_hash())
    assert hashes[0] == hashes[1]


def _blind_world(punish, p=0.25):
    w = _watch_scene(norm_mem_obs_mode="blind")
    s = w.s
    put(w, 1, 10.0, 100.0)
    if not punish:
        s.pun_t[0] = -1
    regrid(w)
    # 이전 스텝까지의 눈가림 비율(지수평활 [합, 무게]): p
    s.norm_blind[0], s.norm_blind[1] = p, 1.0
    return w


def test_mem_obs_mode_blind_and_off():
    # blind: 같은 스텝에 벌이 났을 때와 안 났을 때 동전 집합이 같다(시간 정보 없음)
    for t in (50, 51, 52, 77):
        wa, wb = _blind_world(True), _blind_world(False)
        wa.s.pun_t[0] = t                    # wa는 이 스텝에 벌이 났고 wb는 안 났다
        st.watch_pun(t, wa.s, wa.fs, wa.isp)
        st.watch_pun(t, wb.s, wb.fs, wb.isp)
        assert ((wa.s.mem > 0) == (wb.s.mem > 0)).all(), t
        want = np.array([rngmod.u01(wa.cfg.seed, rngmod.S_NORM_BLIND, t, wa.s.ids[i]) < 0.25 for i in range(8)])
        assert ((wa.s.mem > 0) == want).all()
        assert nt(wa, 0, k.NT_SAW_PUN) == 1 and nt(wb, 0, k.NT_SAW_PUN) == 0     # 실제 관찰은 센다
        # 비율은 이 스텝의 관찰을 동전 뒤에 섞는다(이 스텝의 벌은 다음 스텝부터 비율에 든다)
        a = 1.0 / wa.cfg.norm_mem_tau
        assert wa.s.norm_blind[0] == pytest.approx((1 - a) * 0.25 + a * 1 / 8)
        assert wb.s.norm_blind[0] == pytest.approx((1 - a) * 0.25)
        assert wa.s.norm_blind[1] == pytest.approx(1.0)
    # 같은 시나리오 두 번이 같다
    w1, w2 = _blind_world(True), _blind_world(True)
    st.watch_pun(50, w1.s, w1.fs, w1.isp)
    st.watch_pun(50, w2.s, w2.fs, w2.isp)
    assert (w1.s.mem == w2.s.mem).all()
    # 처음(관찰한 스텝이 없다)에는 비율 0, 한 스텝 본 뒤에는 그 스텝의 비율(무게로 나눠 시작 편향이 없다)
    w0 = _blind_world(True, p=0.0)
    w0.s.norm_blind[:] = 0.0
    st.watch_pun(50, w0.s, w0.fs, w0.isp)
    assert not w0.s.mem.any()
    assert w0.s.norm_blind[0] / w0.s.norm_blind[1] == pytest.approx(1 / 8)
    # off: 기억은 안 바뀌고 관찰 수는 센다
    wo = _watch_scene(norm_mem_obs_mode="off")
    put(wo, 1, 10.0, 100.0)
    regrid(wo)
    st.watch_pun(50, wo.s, wo.fs, wo.isp)
    assert not wo.s.mem.any() and nt(wo, 0, k.NT_SAW_PUN) == 1


def _blind_run(w, steps, pun_every, t0=0):
    """벌 받은 0번 곁에 1번만 있는 장면을 steps 스텝: pun_every마다 0번이 벌 받는다. (동전 상승 수, 실제 관찰 수)."""
    ups = 0
    saw0 = w.s.norm_tab[:, k.NT_SAW_PUN].sum()
    for t in range(t0, t0 + steps):
        w.s.pun_t[0] = t if pun_every and t % pun_every == 0 else -1
        w.s.mem[:] = 0.0
        st.watch_pun(t, w.s, w.fs, w.isp)
        ups += int((w.s.mem > 0).sum())
    return ups, w.s.norm_tab[:, k.NT_SAW_PUN].sum() - saw0


def test_blind_long_run_matches_outcome_and_tracks_recent_rate():
    # 장기 평균 상승 횟수 = 실제 본 벌 수(= outcome의 상승 횟수)
    w = _blind_world(True, p=0.0)
    w.s.norm_blind[:] = 0.0
    ups, saw = _blind_run(w, 6000, 4)
    assert saw == 1500
    assert ups == pytest.approx(saw, rel=0.08)
    # 비율은 최근 ~norm_mem_tau 스텝을 본다(누적 평균이면 벌이 그친 뒤에도 오래 남는다): 벌이 그치고 4τ 뒤 e^-4 배
    tau = w.cfg.norm_mem_tau
    p_before = w.s.norm_blind[0] / w.s.norm_blind[1]
    assert p_before == pytest.approx(1 / 32, rel=0.3)
    _blind_run(w, 4 * tau, 0, t0=6000)
    p_after = w.s.norm_blind[0] / w.s.norm_blind[1]
    assert p_after == pytest.approx(p_before * (1 - 1 / tau) ** (4 * tau), rel=1e-6)
    assert p_after < 0.03 * p_before


def test_blind_rate_not_reset_with_tables():
    w = _blind_world(True)
    w.reset_norm_tables()
    assert w.s.norm_blind[1] == 1.0 and "norm_blind" not in k.SAVED_FIELDS and "norm_blind" in k.OTHER_SCRATCH


def test_saw_pun_counter():
    got = []
    for mode in ("outcome", "blind", "off"):
        w = _watch_scene(norm_mem_obs_mode=mode)
        s = w.s
        put(w, 1, 10.0, 100.0)
        put(w, 2, 12.0, 100.0)
        put(w, 3, 30.0, 100.0)
        s.pun_t[2] = 50
        s.tag[1] = 3
        regrid(w)
        st.watch_pun(50, s, w.fs, w.isp)
        got.append(s.norm_tab[:, k.NT_SAW_PUN].copy())
    assert (got[0] == got[1]).all() and (got[0] == got[2]).all()
    assert got[0][3] == 2 and got[0].sum() == 4         # 1(태그 3)은 둘을, 0과 2는 서로를 본다


def test_grid_rebuild_after_attack_only_when_norm_on(monkeypatch):
    calls = []
    for mod, name in ((k, "build_grid"), (st, "attack"), (st, "watch_pun"), (k, "process_deaths")):
        orig = getattr(mod, name)

        def wrap(*a, _o=orig, _n=name):
            calls.append(_n)
            return _o(*a)
        monkeypatch.setattr(mod, name, wrap)

    def between():
        out, cur = [], None
        for c in calls:
            if c == "attack":
                cur = []
            elif c == "process_deaths":
                out.append(cur)
                cur = None
            elif cur is not None:
                cur.append(c)
        return out
    World(small("stage4", "conflict", "norm")).run(3)
    assert between() == [["build_grid", "watch_pun"]] * 3
    for kw in (dict(), dict(norm_measure=True)):
        calls.clear()
        World(small("stage4", "conflict", "norm_size", **kw)).run(3)
        assert between() == [[]] * 3 and "watch_pun" not in calls


def test_attack_neighbor_set_unchanged_by_rebuild():
    # 격자 재구성은 attack 뒤라 attack이 보는 이웃은 꺼진 세계와 같다: 불활성·고립 세계와 꺼진 세계를 나란히 돌려
    # 스텝마다 공격 수·에너지가 같은지 본다(격자를 attack 앞에 지으면 이웃이 바뀌어 여기서 어긋난다)
    ws = []
    for kw in (dict(), dict(norm_enabled=True, norm_mode="inert", norm_impulse=0.0, norm_mark_src="none")):
        w = World(small("stage4", "conflict", "norm_size", **kw))
        w.s.bout[:, cfgmod.ACT_ATTACK] = 0.5
        ws.append(w)
    for _ in range(200):
        for w in ws:
            w.step()
        a, b = ws
        assert a.s.counters[k.C_ATTACKS] == b.s.counters[k.C_ATTACKS]
        assert (a.s.energy == b.s.energy).all()
    assert ws[0].s.counters[k.C_ATTACKS] > 0


def test_inert_isolated_hash_equals_off_still_holds():
    test_inert_isolated_hash_equals_off()


def test_watch_pun_off_not_called(monkeypatch):
    calls = []
    orig = st.watch_pun
    monkeypatch.setattr(st, "watch_pun", lambda *a: (calls.append(1), orig(*a))[1])
    for kw in (dict(), dict(norm_measure=True), dict(norm_impulse=0.05)):
        World(small("stage4", "conflict", "norm_size", **kw)).run(3)
    assert not calls
    World(small("stage4", "conflict", "norm")).run(2)
    assert len(calls) == 2


def test_resume_gives_same_result_for_saved_fields():
    cfg = small("stage4", "conflict", "norm", norm_impulse=0.03)
    w = World(cfg)
    w.s.bout[:, cfgmod.ACT_ATTACK] = 0.5
    w.run(250)
    assert (w.s.mark_t[w.s.alive] >= 0).any() and (w.s.pun_t[w.s.alive] >= 0).any() and w.s.mem.any()
    arrays = {kk: v.copy() for kk, v in w.state_arrays().items()}
    step0 = w.step_no
    w.run(200)
    want = w.state_hash()
    w2 = World(cfg, populate=False)
    w2.step_no = step0
    assert w2.load_arrays(arrays)
    assert (w2.s.mark_vid[w2.s.alive] == -2).all()           # 저장 안 하는 피해자는 미지(−2)로 돌아온다
    w2.run(200)
    assert w2.state_hash() == want
    # 계수기는 다르다: 이어 읽기 전에 붙은 표시는 끊김 없는 쪽에는 없는 미지(종류 2) 에피소드로 닫히고,
    # 표시당 집행 확률에서는 빠진다
    assert w.s.norm_tab[:, k.NT_MARK_EPIS2].sum() == 0 and w.s.norm_tab[:, k.NT_PUN_UNKNOWN].sum() == 0
    assert w2.s.norm_tab[:, k.NT_MARK_EPIS2].sum() > 0
    tot = w2.s.norm_tab.sum(axis=0)
    epis = tot[k.NT_MARK_EPIS0] + tot[k.NT_MARK_EPIS1]
    if epis > 0:
        want_rate = (tot[k.NT_MARK_EPIS_PUN0] + tot[k.NT_MARK_EPIS_PUN1]) / epis
        assert w2.summary()["norm_punished_per_episode"] == pytest.approx(want_rate)


def test_resume_marks_from_before_load_count_as_unknown():
    """이어 읽으면 저장 안 하는 mark_pun(이 에피소드에 벌 받았나)을 모른다: 읽을 때 표시가 있던 개체는 mark_pun = 2(미지)로
    두고, 그 에피소드는 연장되거나 다시 벌 받아도 종류 2로 닫혀 표시당 집행 확률에 안 든다. 동작(해시)은 안 바뀐다."""
    w = hand_world("norm", norm_mark_tau=5)
    s = w.s
    st._attach_mark(s, 0, 10, 4, s.ids[4], 5)
    s.mark_pun[0] = 1                         # 저장 전에 벌 받았다(이어 읽으면 잃는다)
    st._attach_mark(s, 1, 10, 4, s.ids[4], 5)
    w.step_no = 12
    arrays = {kk: v.copy() for kk, v in w.state_arrays().items()}
    w2 = World(w.cfg, populate=False)
    w2.step_no = 12
    w2.load_arrays(arrays)
    s2 = w2.s
    assert s2.mark_pun[0] == 2 and s2.mark_pun[1] == 2 and (s2.mark_pun[2:] == 0).all()
    st._attach_mark(s2, 1, 13, 4, s2.ids[4], 5)       # 연장: 피해자는 알게 되지만 앞부분의 벌 여부는 모른다
    for t in range(12, 25):
        st.norm_account(t, s2, w2.fs, w2.isp)
    assert _epis(w2, 0) == (0, 0) and _epis(w2, 1) == (0, 0) and _epis(w2, 2) == (2, 0)
    # 만료 뒤 새로 붙으면 다시 아는 에피소드다
    st._attach_mark(s2, 0, 30, 4, s2.ids[4], 5)
    assert s2.mark_pun[0] == 0


def test_punish_keeps_unknown_episode_flag():
    """미지 에피소드(mark_pun 2) 개체를 벌해도 2로 남는다(1로 덮으면 '벌 받은 에피소드'만 골라 세는 편향)."""
    w = hand_world("norm")
    s = w.s
    s.mark_t[1], s.mark_vid[1], s.mark_pun[1] = 9, -2, 2
    put(w, 0, 100.0, 100.0)
    put(w, 1, 101.0, 100.0)
    s.act[0, cfgmod.ACT_ATTACK] = 1.0
    regrid(w)
    st.attack(10, s, w.fs, w.isp)
    assert s.pun_t[1] == 10 and s.mark_pun[1] == 2


# ======================= 과제 13: 벌주기 막기(norm_block_tag·norm_block3_tag) =======================
def _block_scene(**kw):
    """0번(공격자)이 닿는 1번(표시됨, 피해자 4번)과 조금 더 먼 2번(표시 없음)을 둔 장면. 0번은 자발 공격 1."""
    w = hand_world("norm", **kw)
    s = w.s
    put(w, 0, 100.0, 100.0)
    put(w, 1, 101.0, 100.0)
    put(w, 2, 98.6, 100.0)
    st._attach_mark(s, 1, 9, 4, s.ids[4], w.cfg.norm_mark_tau)
    s.act[0, cfgmod.ACT_ATTACK] = 1.0
    regrid(w)
    return w


def _snap(w):
    s = w.s
    return (s.energy.copy(), s.ledger.copy(), s.counters.copy(), s.pun_t.copy(), s.mark_t.copy(), s.mem.copy())


def test_block_tag_cannot_punish_marked_and_pays_nothing():
    # 양성 확인: 막기 없는 태그(0)는 표시된 1번을 벌한다
    w = _block_scene(norm_block_tag=2)
    st.attack(10, w.s, w.fs, w.isp)
    assert w.s.pun_t[1] == 10 and nt(w, 0, k.NT_PUN) == 1
    # 태그 2는 취소: 에너지·비용·장부·계수(공격 쪽) 모두 그대로, 대체 대상(2번)으로 넘어가지도 않는다
    w = _block_scene(norm_block_tag=2)
    s = w.s
    s.tag[0] = 2
    before = _snap(w)
    st.attack(10, s, w.fs, w.isp)
    for a, b in zip(before, _snap(w)):
        assert (a == b).all()
    for col in (k.NT_PUN, k.NT_TAKE, k.NT_ATK_COST, k.NT_ATK_VOL_MARKED, k.NT_ATK_VOL_PLAIN, k.NT_PUN_COST):
        assert nt(w, 2, col) == 0, k.NT_NAMES[col]
    assert s.touch[1] == 0 and s.touch[2] == 0
    # 표시 없는 이웃은 평소대로 친다(가로채기)
    s.mark_t[1] = -1
    st.attack(11, s, w.fs, w.isp)
    assert nt(w, 2, k.NT_TAKE) == 1 and s.mark_t[0] == 11


def test_block3_allows_self_defense_only():
    for vid_of, want_pun in (("me", True), ("other", False), ("none", False), ("unknown", False)):
        w = _block_scene(norm_block3_tag=3)
        s = w.s
        s.tag[0] = 3
        if vid_of == "me":
            s.mark_vslot[1], s.mark_vid[1] = 0, s.ids[0]
        elif vid_of == "none":
            s.mark_vslot[1], s.mark_vid[1] = -1, -1
        elif vid_of == "unknown":
            s.mark_vslot[1], s.mark_vid[1] = -2, -2
        e0 = s.energy[0]
        st.attack(10, s, w.fs, w.isp)
        if want_pun:
            assert s.pun_t[1] == 10 and nt(w, 3, k.NT_PUN_2ND) == 1, vid_of
        else:
            assert s.pun_t[1] == -1 and s.energy[0] == e0 and nt(w, 3, k.NT_ATK_COST) == 0, vid_of
            assert s.touch[2] == 0, vid_of


def test_block_minus_one_has_no_effect():
    hashes = []
    for kw in (dict(), dict(norm_block_tag=3, norm_block3_tag=2)):    # 아무도 태그 2·3이 아니다
        w = World(norm_cfg(norm_impulse=0.03, **kw))
        w.s.bout[:, cfgmod.ACT_ATTACK] = 0.5
        w.run(150)
        assert w.s.norm_tab[:, k.NT_PUN].sum() > 0
        hashes.append(w.state_hash())
    assert hashes[0] == hashes[1]


def test_block_does_not_cancel_impulse():
    # 충동(떨리는 손)은 표시를 모른다: 막힌 태그라도 가장 가까운 이웃(여기서는 표시된 1번)을 가로챈다
    w = _block_scene(norm_block_tag=2, norm_block3_tag=3, norm_impulse=1.0)
    s = w.s
    s.tag[0] = 2
    s.act[0, cfgmod.ACT_ATTACK] = 0.0
    st.attack(10, s, w.fs, w.isp)
    assert nt(w, 2, k.NT_TAKE_IMP) == 1 and nt(w, 2, k.NT_TAKE_ON_MARKED) == 1 and s.pun_t[1] == -1
    assert s.touch[1] == k.TOUCH_HIT


def test_blocked_opportunity_still_counted():
    w = _block_scene(norm_block_tag=2)
    s = w.s
    s.tag[0] = 2
    st.attack(10, s, w.fs, w.isp)
    assert nt(w, 2, k.NT_EXPO) == 1 and nt(w, 2, k.NT_OPP_MARKED) == 1     # 기회는 있었다
    assert nt(w, 2, k.NT_ATK_VOL_MARKED) == 0 and nt(w, 2, k.NT_ATK_VOL_PLAIN) == 0     # 공격은 안 나갔다


def test_block_requires_norm_enabled():
    for kw in (dict(norm_block_tag=2), dict(norm_block3_tag=3)):
        with pytest.raises(ValueError):
            small("stage4", "conflict", "norm_size", **kw)
        assert norm_cfg(**kw).norm_enabled


# ======================= 과제 14: 겨루기 마당 준비·다태그 겨루기·공격 행 다시 뽑기 =======================
def _old_ckpt(tmp_path, *presets, steps=30):
    """norm_size 없이 자란 작은 세계(옛 크기)의 체크포인트."""
    from kkumteul import checkpoint as ckpt
    w = World(small("stage4", *presets))
    w.run(steps)
    return ckpt.save(w, tmp_path / "ck")


def test_arena_side_resized_to_norm_size(tmp_path):
    path = _old_ckpt(tmp_path)
    ref = measure.side_from_world(World(norm_cfg()))
    side = measure.norm_arena_side(path)
    assert side["cfg"].n_in == ref["cfg"].n_in > small("stage4").n_in
    measure.check_same_shape(ref, side)
    measure.check_same_shape(side, ref)
    c0 = side["cfg"].norm_in0
    assert not side["genomes"]["Win"][:, c0:, :].any()       # 늘어난 입력 열의 가중치는 0
    # 같은 체크포인트를 cfg 없이 읽으면 크기가 달라 못 겨룬다(첫째 판의 가장 큰 구멍)
    with pytest.raises(ValueError):
        measure.check_same_shape(measure.load_side(path), ref)
    # 덧붙일 설정도 받는다
    assert measure.norm_arena_side(path, ["conflict_enabled=true"])["cfg"].conflict_enabled


def _arena_cfg(**kw):
    return norm_cfg(max_agents=400, **kw)


def test_resident_attack_row_fixed_and_arena_has_no_violence():
    side = measure.side_from_world(World(norm_cfg()))
    g = side["genomes"]
    A = cfgmod.ACT_ATTACK
    # 일부러 틀린 거주민: 공격 편향 +7, 출력 행이 큼
    gen = np.random.default_rng(3)
    g["bout"][:, A] = 7.0
    g["Wout"][:, A, :] = (3.0 * gen.standard_normal(g["Wout"][:, A, :].shape)).astype(g["Wout"].dtype)
    fixed = measure.fix_resident_attack(g)
    assert (fixed["bout"][:, A] == -1).all() and not fixed["Wout"][:, A, :].any()
    assert (g["bout"][:, A] == 7.0).all(), "사본을 고친다"
    other = [o for o in range(g["bout"].shape[1]) if o != A]
    assert (fixed["bout"][:, other] == g["bout"][:, other]).all() and (fixed["Wout"][:, other] == g["Wout"][:, other]).all()
    cfg = _arena_cfg()
    # 편향만 고치면(calm_attack 꼴) Wout·h로 가로채기가 남는다
    bias_only = {kk: v.copy() for kk, v in g.items()}
    bias_only["bout"][:, A] = -1.0
    res = measure.compete_multi(cfg, [(bias_only, 0, 300)], 300, seed=11)
    assert res["norm_by_tag"][0]["take"] > 0
    # 출력 행까지 0이면 마당(충동 0)에 폭력이 없다(관문 −1e)
    res = measure.compete_multi(cfg, [(fixed, 0, 300)], 300, seed=11)
    assert res["norm_by_tag"][0]["take"] == 0 and res["norm_by_tag"][0]["pun"] == 0
    assert res["norm_by_tag"][0]["alive_steps"] > 0


def test_compete_multi_groups_and_same_seed_pairs():
    side = measure.side_from_world(World(norm_cfg()))
    g = measure.fix_resident_attack(side["genomes"])
    cfg = _arena_cfg(norm_impulse=0.02)
    with pytest.raises(ValueError):
        measure.compete_multi(cfg, [(g, 0, 300), (g, 1, 101)], 10, seed=1)
    groups = [(g, 0, 60), (g, 1, 60), (g, 2, 30)]
    r1, w1 = measure.compete_multi(cfg, groups, 60, seed=5, return_world=True)
    r2, w2 = measure.compete_multi(cfg, groups, 60, seed=5, return_world=True)
    assert w1.state_hash() == w2.state_hash() and r1["alive_by_tag"] == r2["alive_by_tag"]
    assert not w1.cfg.mutation_enabled
    # 같은 씨앗이면 태그 구성이 달라도 칸(넣는 순서)마다 같은 자리·방향·나이로 시작한다
    _, a = measure.compete_multi(cfg, groups, 0, seed=5, return_world=True)
    _, b = measure.compete_multi(cfg, [(g, 3, 150)], 0, seed=5, return_world=True)
    for name in ("x", "y", "heading", "age"):
        assert (getattr(a.s, name)[:150] == getattr(b.s, name)[:150]).all(), name
    tags = a.s.tag[:150]
    assert [(tags == t).sum() for t in (0, 1, 2)] == [60, 60, 30]
    # 넣는 순서를 섞어 낮은 칸이 한 태그에 몰리지 않는다
    assert len(set(tags[:40].tolist())) == 3 and not (np.diff(tags) >= 0).all()
    # 기존 compete는 그대로 돈다
    assert "r" in measure.compete(cfg, g, g, 20, 20, seed=1)


def test_compete_multi_reports_norm_tables():
    side = measure.side_from_world(World(norm_cfg()))
    g = measure.fix_resident_attack(side["genomes"])
    ge = {kk: v.copy() for kk, v in g.items()}
    ge["bout"][:, cfgmod.ACT_ATTACK] = 0.5                  # 공격하는 쪽(태그 1)
    res = measure.compete_multi(_arena_cfg(norm_impulse=0.02), [(g, 0, 150), (ge, 1, 150)], 200, seed=2)
    for key in ("norm_by_tag", "role_table", "alive_by_tag", "eaten_by_tag", "mark_prevalence",
                "mark_prevalence_by_tag", "net_income_by_tag", "series"):
        assert key in res, key
    assert set(res["norm_by_tag"][1]) == set(k.NT_NAMES)
    assert set(res["role_table"]) == set(k.RT_ROLES)
    assert res["norm_by_tag"][1]["take"] + res["norm_by_tag"][1]["pun"] > 0
    assert sum(res["alive_by_tag"].values()) > 0 and res["eaten_by_tag"][0] > 0
    assert 0.0 <= res["mark_prevalence"] <= 1.0


def test_net_income_per_step_by_tag_hand_values():
    w = hand_world("norm")
    s = w.s
    s.tag[:4] = 1
    s.eaten[:] = 0.0
    s.eaten[0], s.eaten[5] = 2.0, 3.0
    tab = s.norm_tab
    vals = {k.NT_EATEN_DEAD: 10.0, k.NT_TAKE_GAIN: 1.5, k.NT_PUN_FINE: 0.5, k.NT_TAKE_DMG_RECV: 2.0,
            k.NT_PUN_DMG_RECV: 1.0, k.NT_ATK_COST: 0.4, k.NT_ALIVE_STEPS: 100.0, k.NT_LIVES_DONE: 4.0}
    for col, v in vals.items():
        tab[1, col] = v
    tab[0, k.NT_ALIVE_STEPS] = 50.0
    out = measure.net_income_by_tag(w)
    assert out[1]["net_per_step"] == pytest.approx((10.0 + 2.0 + 1.5 + 0.5 - 2.0 - 1.0 - 0.4) / 100.0)
    assert out[1]["eaten_per_life_done"] == pytest.approx(10.0 / 4.0)
    assert out[1]["life_mean"] == pytest.approx(100.0 / (4.0 + 4))        # 끝난 생 4 + 산 개체 4
    assert out[0]["net_per_step"] == pytest.approx(3.0 / 50.0)
    assert out[0]["eaten_per_life_done"] is None                           # 끝난 생이 없다
    assert out[2]["net_per_step"] is None                                  # 개체 스텝이 없다


def test_reseed_attack_row_matches_new_agent_distribution():
    A = cfgmod.ACT_ATTACK
    w = World(norm_cfg(init_agents=500, max_agents=600))
    s = w.s
    a = s.alive
    s.bout[a, A] = 3.0                                       # 떠돈 공격 행
    s.Wout[a, A, :] = 9.0
    before = {name: getattr(s, name).copy() for name in ("Win", "Wrec", "bh", "Wout", "bout", "n_hidden")}
    edits = w.reseed_attack_row(b0=-1.0, seed=5)
    assert edits == ["reseed_attack_row(b0=-1.0)"]
    assert (s.bout[a, A] == -1.0).all()
    row = s.Wout[a, A, :].astype(np.float64)
    sd = w.cfg.init_w_out
    assert abs(row.mean()) < 0.05 * sd and row.std() == pytest.approx(sd, rel=0.05)
    # n_hidden 밖 열도 새 개체와 같은 분포(random_genome이 h_max 전체를 채운다)
    hmax_used = int(s.n_hidden[a].max())
    assert hmax_used < w.cfg.h_max
    assert s.Wout[a, A, hmax_used:].std() == pytest.approx(sd, rel=0.1)
    # 다른 배열·다른 출력 행·죽은 칸은 그대로
    other = [o for o in range(s.bout.shape[1]) if o != A]
    assert (s.bout[:, other] == before["bout"][:, other]).all() and (s.Wout[:, other] == before["Wout"][:, other]).all()
    assert (s.bout[~a] == before["bout"][~a]).all() and (s.Wout[~a] == before["Wout"][~a]).all()
    for name in ("Win", "Wrec", "bh", "n_hidden"):
        assert (getattr(s, name) == before[name]).all(), name
    # 같은 씨앗이면 같고, 기울기 시작판 b0 = −0.3도 받는다
    w2 = World(norm_cfg(init_agents=500, max_agents=600))
    w2.reseed_attack_row(b0=-0.3, seed=5)
    assert (w2.s.bout[a, A] == np.float32(-0.3)).all() and (w2.s.Wout[a, A, :] == s.Wout[a, A, :]).all()


def test_branch_reseed_option_recorded_and_applies_to_all(tmp_path):
    import json
    from kkumteul.run import Runner, main as run_main
    cfg = small("stage4", "norm_size")
    m = Runner.new(tmp_path / "main", cfg)
    m.loop(400, quiet=True)
    m.close()
    on = ["conflict_enabled=true", "norm_enabled=true"]
    A = cfgmod.ACT_ATTACK
    r = Runner.branch(tmp_path / "main", tmp_path / "re", 400, on, False, False, reseed_attack=True, attack_b0=-0.3)
    a = r.world.s.alive
    assert (r.world.s.bout[a, A] == np.float32(-0.3)).all()
    r.close()
    edits = json.loads((tmp_path / "re" / "run.json").read_text())["genome_edits"]
    assert edits == ["bout[ACT_ATTACK]=-1", "reseed_attack_row(b0=-0.3)"]     # calm_attack 뒤에 겹쳐 건다
    plain = Runner.branch(tmp_path / "main", tmp_path / "plain", 400, on, False, False)
    plain.close()
    assert json.loads((tmp_path / "plain" / "run.json").read_text())["genome_edits"] == ["bout[ACT_ATTACK]=-1"]
    # CLI
    run_main(["branch", "--from", str(tmp_path / "main"), "--run", str(tmp_path / "cli"), "--at", "400",
              "--steps", "1", "--quiet", "--set", on[0], "--set", on[1], "--reseed-attack-row", "--attack-b0", "-0.3"])
    assert json.loads((tmp_path / "cli" / "run.json").read_text())["genome_edits"][-1] == "reseed_attack_row(b0=-0.3)"
    run_main(["branch", "--from", str(tmp_path / "main"), "--run", str(tmp_path / "cli2"), "--at", "400",
              "--steps", "1", "--quiet", "--reseed-attack-row"])
    assert json.loads((tmp_path / "cli2" / "run.json").read_text())["genome_edits"] == ["reseed_attack_row(b0=-1.0)"]
    with pytest.raises(SystemExit):
        run_main(["branch", "--from", str(tmp_path / "main"), "--run", str(tmp_path / "cli3"), "--at", "400",
                  "--steps", "1", "--quiet", "--attack-b0", "-0.3"])


def test_branch_resizes_to_norm_size(tmp_path):
    from kkumteul.run import Runner
    m = Runner.new(tmp_path / "main", small("stage4", "conflict"))
    m.loop(400, quiet=True)
    m.close()
    r = Runner.branch(tmp_path / "main", tmp_path / "n", 400, ["preset=norm_size", "norm_enabled=true"], False, False)
    w = r.world
    assert w.cfg.n_in == norm_cfg().n_in and w.cfg.norm_enabled
    assert not w.s.Win[w.s.alive, w.cfg.norm_in0:, :].any()
    r.loop(50, quiet=True)
    r.close()
    assert w.step_no == 450

