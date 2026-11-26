"""부모의 학습 보조 투자("가르침" 후보) 시험. 과제 1: 설정·난수·필드·옛 호환(동작 변화 없음).

설계 docs/superpowers/specs/2026-11-26-parent-teaching-design.md, 계획 docs/superpowers/plans/2026-11-26-parent-teaching.md.

    python3 -m pytest -q tests/test_teach.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_stages import small  # noqa: E402
from test_taste import PINNED_TASTE_SIZE_HASH_300, taste_world  # noqa: E402
from kkumteul import config as cfgmod  # noqa: E402
from kkumteul import kernels as k  # noqa: E402
from kkumteul import measure  # noqa: E402
from kkumteul import pack  # noqa: E402
from kkumteul.world import World  # noqa: E402

NEW_UNSAVED = {
    "know_age": np.int32, "know_src": np.int8, "n_seen_par": np.int32, "n_seen_par_poison": np.int32,
    "n_taught": np.int32, "n_taught_knower": np.int32, "par_near_steps": np.int32, "par_alive_steps": np.int32,
    "juv_alive_steps": np.int32, "adult_near_sum": np.int64, "par_slot": np.int32, "jscr_i": np.int32,
    "jscr_d": np.float64, "tflag": np.int8,
}


def teach_cfg(**kw):
    """보고 배우기(outcome)·먹이 종류·미숙기·보여 주기를 모두 켠 작은 설정."""
    return small("stage2", "kinds", "taste", soc_taste_enabled=True, juv_enabled=True, teach_enabled=True, **kw)


# ---- 설정 ----
def test_teach_config_defaults_and_checks():
    c = small("taste")
    assert c.juv_enabled is False and c.juv_measure is False
    assert c.teach_enabled is False and c.feed_enabled is False
    assert c.juv_steps == 200 and c.juv_eat_floor == 0.5 and c.juv_move_floor == 0.3
    assert c.juv_poison_mult == 1.0 and c.juv_know_theta == 0.2
    assert c.teach_mode == "kin" and c.teach_gain == 0.5 and c.teach_cost == 0.0005
    assert c.teach_cost_when == "near" and c.teach_gene_sigma == 0.02
    assert c.feed_rate == 0.02 and c.feed_gene_sigma == 0.02
    assert small(juv_steps=1).juv_steps == 1
    assert small(juv_eat_floor=1.0, juv_move_floor=1.0).juv_eat_floor == 1.0    # 1까지는 된다(효과 없는 손잡이)
    assert small(juv_poison_mult=1.0, juv_know_theta=1.0).juv_know_theta == 1.0
    assert small("taste", soc_taste_enabled=True, juv_enabled=True, teach_enabled=True,
                 food_kinds_enabled=True, teach_gain=1.0, teach_cost=0.0).teach_gain == 1.0
    for bad in (dict(juv_steps=0), dict(juv_steps=-5),
                dict(juv_eat_floor=0.0), dict(juv_eat_floor=-0.1), dict(juv_eat_floor=1.1),
                dict(juv_move_floor=0.0), dict(juv_move_floor=-0.1), dict(juv_move_floor=1.1),
                dict(juv_poison_mult=0.9), dict(juv_know_theta=0.0), dict(juv_know_theta=-0.1),
                dict(juv_know_theta=1.1),
                dict(teach_gain=-0.1), dict(teach_gain=1.1), dict(teach_cost=-0.1),
                dict(teach_gene_sigma=-0.1), dict(feed_rate=-0.1), dict(feed_gene_sigma=-0.1),
                dict(teach_mode="nope"), dict(teach_cost_when="nope")):
        with pytest.raises(ValueError):
            small("taste", **bad)


def test_teach_mode_constants():
    assert (cfgmod.TEACH_KIN, cfgmod.TEACH_COSTONLY, cfgmod.TEACH_ANY, cfgmod.TEACH_KNOWER,
            cfgmod.TEACH_DRIFT) == (0, 1, 2, 3, 4)
    assert cfgmod.TEACH_MODES == {"kin": 0, "costonly": 1, "any": 2, "knower": 3, "drift": 4}
    assert (cfgmod.WHEN_NEAR, cfgmod.WHEN_MEAL) == (0, 1)
    assert cfgmod.TEACH_WHENS == {"near": 0, "meal": 1}
    for mode in cfgmod.TEACH_MODES:
        assert small("taste", teach_mode=mode).teach_mode == mode
    for when in cfgmod.TEACH_WHENS:
        assert small("taste", teach_cost_when=when).teach_cost_when == when


def test_teach_requires_soc_outcome_and_kinds():
    c = teach_cfg()
    assert c.teach_enabled and c.juv_enabled and c.soc_taste_enabled and c.food_kinds_enabled
    base = dict(soc_taste_enabled=True, juv_enabled=True, teach_enabled=True)
    with pytest.raises(ValueError):
        small("stage2", "kinds", "taste", soc_taste_enabled=True, teach_enabled=True)          # 미숙기가 꺼짐
    with pytest.raises(ValueError):
        small("stage2", "kinds", "taste", juv_enabled=True, teach_enabled=True)                # 보고 배우기가 꺼짐
    with pytest.raises(ValueError):
        small("taste", **base)                                                                  # 먹이 종류가 꺼짐
    for mode in ("what", "blind", "inert"):
        with pytest.raises(ValueError):
            teach_cfg(soc_taste_mode=mode)
    assert teach_cfg(soc_taste_mode="outcome").teach_enabled
    # 먹이 대주기는 미숙기만 요구한다
    with pytest.raises(ValueError):
        small(feed_enabled=True)
    assert small(juv_enabled=True, feed_enabled=True).feed_enabled
    # 미숙기 측정은 아무것도 요구하지 않는다
    assert small(juv_measure=True).juv_measure
    assert small("stage2", juv_measure=True).juv_measure


def test_teach_mode_string_override():
    c = teach_cfg().with_overrides(["teach_mode=knower", "teach_cost_when=meal"])
    assert c.teach_mode == "knower" and c.teach_cost_when == "meal"
    c2 = small("stage2", "kinds", "taste").with_overrides(
        ["soc_taste_enabled=true", "juv_enabled=true", "teach_enabled=true", "teach_mode=costonly"])
    assert c2.teach_enabled and c2.teach_mode == "costonly"
    with pytest.raises(ValueError):
        teach_cfg().with_overrides(["teach_mode=nope"])
    with pytest.raises(ValueError):
        teach_cfg().with_overrides(["teach_cost_when=nope"])


def test_teach_params_packed():
    c = teach_cfg(juv_steps=120, juv_eat_floor=0.6, juv_move_floor=0.4, juv_poison_mult=2.0, juv_know_theta=0.3,
                  teach_mode="knower", teach_cost_when="meal", teach_gain=0.7, teach_cost=0.002,
                  teach_gene_sigma=0.03, feed_enabled=True, feed_rate=0.05, feed_gene_sigma=0.04,
                  give_loss=0.2, give_min_energy=1.5)
    fp, ip = k.pack_params(c)
    assert ip[k.Q_JUV] == 1 and ip[k.Q_JUV_COUNT] == 1 and ip[k.Q_JUV_STEPS] == 120
    assert ip[k.Q_TEACH] == 1 and ip[k.Q_TEACH_MODE] == cfgmod.TEACH_KNOWER
    assert ip[k.Q_TEACH_WHEN] == cfgmod.WHEN_MEAL and ip[k.Q_FEED] == 1
    assert fp[k.P_JUV_EAT_FLOOR] == 0.6 and fp[k.P_JUV_MOVE_FLOOR] == 0.4
    assert fp[k.P_JUV_POISON_MULT] == 2.0 and fp[k.P_JUV_KNOW_THETA] == 0.3
    assert fp[k.P_TEACH_GAIN] == 0.7 and fp[k.P_TEACH_COST] == 0.002 and fp[k.P_TEACH_GENE_SIGMA] == 0.03
    assert fp[k.P_FEED_RATE] == 0.05 and fp[k.P_FEED_GENE_SIGMA] == 0.04
    assert fp[k.P_FEED_LOSS] == 0.2 and fp[k.P_FEED_MIN] == 1.5
    # 꺼진 세계
    fp0, ip0 = k.pack_params(small("taste"))
    assert ip0[k.Q_JUV] == 0 and ip0[k.Q_JUV_COUNT] == 0 and ip0[k.Q_TEACH] == 0 and ip0[k.Q_FEED] == 0
    assert k.N_FP == len(fp0) and k.N_IP == len(ip0)
    # 측정만 켜면 계수는 세지만 동작 장치는 꺼진 채
    _, ipm = k.pack_params(small(juv_measure=True))
    assert ipm[k.Q_JUV] == 0 and ipm[k.Q_JUV_COUNT] == 1 and ipm[k.Q_TEACH] == 0
    # 미숙기·보고 배우기 없이 feed만(feed는 juv가 필요하다)·teach만(검사가 막는다) — 켜짐은 juv와 soc에 묶임
    _, ipf = k.pack_params(small(juv_enabled=True, feed_enabled=True))
    assert ipf[k.Q_FEED] == 1 and ipf[k.Q_TEACH] == 0
    # 곁 거리는 사회 학습 설정이 꺼져 있어도 채워져 있다(곁 계수기가 쓴다)
    assert fp0[k.P_SOC_RADIUS] == small("taste").soc_radius


def test_new_p_q_slots_appended_at_end():
    # 새 칸은 기존 칸 뒤에만 붙는다(기존 번호 불변)
    assert k.P_SOC_RADIUS == 62 and k.Q_SOC_MODE == 27
    new_p = (k.P_JUV_EAT_FLOOR, k.P_JUV_MOVE_FLOOR, k.P_JUV_POISON_MULT, k.P_JUV_KNOW_THETA, k.P_TEACH_GAIN,
             k.P_TEACH_COST, k.P_TEACH_GENE_SIGMA, k.P_FEED_RATE, k.P_FEED_GENE_SIGMA, k.P_FEED_LOSS, k.P_FEED_MIN)
    assert new_p == tuple(range(63, 74)) and k.N_FP == 74
    new_q = (k.Q_JUV, k.Q_JUV_COUNT, k.Q_JUV_STEPS, k.Q_TEACH, k.Q_TEACH_MODE, k.Q_TEACH_WHEN, k.Q_FEED)
    assert new_q == tuple(range(28, 35)) and k.N_IP == 35


# ---- 필드 ----
def test_teach_fields_registered():
    names = [f[0] for f in k.AGENT_FIELDS]
    assert names.index("show_g") == names.index("soc_g") + 1
    assert names.index("feed_g") == names.index("show_g") + 1
    info = {f[0]: f for f in k.AGENT_FIELDS}
    for name in ("show_g", "feed_g"):
        assert info[name][1] is np.float32 and info[name][2] == () and info[name][3] is True
        assert name in k.SAVED_FIELDS and k.HASH_DEFAULTS[name] == 0
    for name, dt in NEW_UNSAVED.items():
        assert info[name][1] is dt and info[name][2] == () and info[name][3] is False, name
        assert name not in k.SAVED_FIELDS, name
    assert k.HASH_DEFAULTS["know_age"] == -1
    assert "know_age" not in k.SAVED_FIELDS
    assert k.TAG_ROWS == 8
    assert "grad_tab" in k.OTHER_SCRATCH and "grad_tab" not in k.SAVED_FIELDS
    w = World(small("stage2", "kinds", "taste"))
    assert (w.s.know_age == -1).all()
    for name in ("show_g", "feed_g") + tuple(n for n in NEW_UNSAVED if n != "know_age"):
        assert (getattr(w.s, name) == 0).all(), name
    for name, dt in NEW_UNSAVED.items():
        assert getattr(w.s, name).dtype == np.dtype(dt), name
    assert w.s.grad_tab.shape == (k.TAG_ROWS, k.N_GT) and w.s.grad_tab.dtype == np.float64
    assert (w.s.grad_tab == 0).all()


def test_grad_table_columns():
    assert len(k.GT_NAMES) == k.N_GT and len(set(k.GT_NAMES)) == k.N_GT
    order = ("GRAD", "DEAD", "N1", "M1", "K1", "N2", "M2", "K2", "M3", "K3", "SRC1", "SRC2", "SRC3", "SRC4",
             "PAR_NEAR", "PAR_ALIVE", "ADULT_NEAR", "ALIVE_STEPS", "SEEN", "SEEN_PAR", "SEEN_PAR_POISON",
             "TAUGHT", "TAUGHT_KNOWER", "MEALS", "POISON_MEALS", "E_GRAD", "TEACH_COST", "FEED_GIVEN")
    assert k.N_GT == len(order)
    for idx, name in enumerate(order):
        assert getattr(k, "GT_" + name) == idx, name
        assert k.GT_NAMES[idx] == name.lower()


def test_know_age_minus_one_after_checkpoint_roundtrip(tmp_path):
    from kkumteul import checkpoint as ckpt
    w = World(small("stage2", "kinds", "taste"))
    w.run(30)
    assert (w.s.know_age[w.s.alive] == -1).all()
    w.s.know_age[:] = 5          # 저장 안 함 필드: 복원 대상이 아니므로 이 값은 사라져야 한다
    path = ckpt.save(w, tmp_path)
    w2, _ = ckpt.load(path)
    a = np.flatnonzero(w2.s.alive)
    assert len(a) > 0 and (w2.s.know_age[a] == -1).all()
    assert (w2.s.know_age == -1).all()
    # 새 필드는 해시에 안 들어가 저장 때의 해시와 같다
    assert w2.state_hash() == w.state_hash()
    # inject_agent로 넣은 개체도 −1과 0 초기화
    side = measure.side_from_world(w2)
    w2.s.n_seen_par[:] = 3
    w2.s.know_src[:] = 2
    w2.s.juv_alive_steps[:] = 9
    w2.s.par_slot[:] = 4
    w2.s.tflag[:] = 1
    w2.s.jscr_d[:] = 0.5
    j_id = measure.inject_agent(w2, side["genomes"], 0, 1, 10.0, 10.0, 0.0, 17)
    j = int(np.flatnonzero(w2.s.ids == j_id)[0])
    assert w2.s.know_age[j] == -1
    for name in NEW_UNSAVED:
        if name != "know_age":
            assert getattr(w2.s, name)[j] == 0, name


# ---- 묶음 호환 ----
def test_teach_in_genome():
    assert "show_g" in measure.GENOME and "feed_g" in measure.GENOME
    assert measure.OPTIONAL_GENOME == ("taste_g", "soc_g", "show_g", "feed_g")


def _bundle(tmp_path, drop):
    """살아 있는 세계의 유전체로 묶음을 만들고, drop에 든 배열을 뺀 묶음을 다시 쓴다."""
    w = taste_world("taste")
    w.run(20)
    g = measure.extract_genomes(w)
    arrays = dict(g)
    arrays[pack.MARK] = np.array(pack.PACK_VERSION, dtype=np.int64)
    arrays["config_json"] = np.array(json.dumps(w.cfg.to_dict(), sort_keys=True))
    arrays["info_json"] = np.array(json.dumps({"label": "t", "n": len(g["n_hidden"])}, sort_keys=True))
    n = len(g["n_hidden"])
    cut = tmp_path / "cut.npz"
    np.savez(cut, **{kk: v for kk, v in arrays.items() if kk not in drop})
    return cut, n


@pytest.mark.parametrize("drop", [("show_g", "feed_g"), ("show_g",), ("feed_g",),
                                  ("taste_g", "soc_g", "show_g", "feed_g")])
def test_pack_reads_bundle_without_show_feed(tmp_path, drop):
    cut, n = _bundle(tmp_path, drop)
    g, _, _ = pack.read(cut)
    for name in drop:
        assert (g[name] == 0).all() and len(g[name]) == n and g[name].dtype == np.float32, name
    assert set(g) == set(measure.GENOME) | {"gen"}
    # 다른 유전체 배열이 빠지면 여전히 오류다
    cut2, _ = _bundle(tmp_path, ("show_g", "feed_g", "Wout"))
    with pytest.raises(ValueError, match="빠졌다"):
        pack.read(cut2)


# ---- 해시·체크포인트 ----
@pytest.mark.parametrize("presets", list(PINNED_TASTE_SIZE_HASH_300))
def test_teach_off_world_hash_unchanged(presets):
    # 처음부터 꺼진 세계의 해시는 새 필드(전부 기본값)가 생기기 전 값과 같다
    w = World(small(*presets, "taste_size"))
    w.run(300)
    assert w.state_hash() == PINNED_TASTE_SIZE_HASH_300[presets]
    assert (w.s.show_g == 0).all() and (w.s.feed_g == 0).all()


def test_old_checkpoint_loads_show_feed_zero():
    w = taste_world("taste")
    w.run(30)
    a = np.flatnonzero(w.s.alive)
    w.s.show_g[a] = np.linspace(0.0, 0.3, len(a)).astype(np.float32)
    w.s.feed_g[a] = np.linspace(0.1, 0.4, len(a)).astype(np.float32)
    arrays = {kk: v.copy() for kk, v in w.state_arrays().items()}
    assert "show_g" in arrays and "feed_g" in arrays
    w1 = World(w.cfg, populate=False)
    w1.step_no = w.step_no
    assert w1.load_arrays({kk: v.copy() for kk, v in arrays.items()})
    assert (w1.s.show_g == w.s.show_g).all() and (w1.s.feed_g == w.s.feed_g).all()
    assert np.abs(w1.s.show_g).max() > 0 and np.abs(w1.s.feed_g).max() > 0
    # 없는 옛 체크포인트는 0으로 읽힌다
    del arrays["show_g"]
    del arrays["feed_g"]
    w2 = World(w.cfg, populate=False)
    w2.step_no = w.step_no
    assert w2.load_arrays(arrays)
    assert (w2.s.show_g == 0).all() and (w2.s.feed_g == 0).all()
    assert (w2.s.know_age == -1).all()


def test_nonzero_show_feed_enters_state_hash():
    w = taste_world("taste")
    w.run(20)
    h0 = w.state_hash()
    w.s.show_g[w.s.alive] = 0.25
    assert w.state_hash() != h0
    w.s.show_g[:] = 0.0
    w.s.feed_g[w.s.alive] = 0.25
    assert w.state_hash() != h0
    w.s.feed_g[:] = 0.0
    assert w.state_hash() == h0
