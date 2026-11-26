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

