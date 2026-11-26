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

from test_harsh import _place  # noqa: E402
from test_stages import rel_residual, small  # noqa: E402
from test_taste import PINNED_TASTE_SIZE_HASH_300, _natural_child, taste_world  # noqa: E402
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
    assert (w.s.par_slot[w.s.alive] == -1).all()       # 산 개체는 _reset_body가 -1(부모 칸 없음)로 시작시킨다
    for name in ("show_g", "feed_g") + tuple(n for n in NEW_UNSAVED if n not in ("know_age", "par_slot")):
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
        if name == "par_slot":
            assert w2.s.par_slot[j] == -1               # -1 = 부모 칸 없음(0은 0번 칸을 가리키는 값이라 안 된다)
        elif name != "know_age":
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


# ---- 과제 2: 미숙기(먹이 효율·속도 상한·독 배수) ----
def _lone_juv(**kw) -> World:
    """개체 하나·먹이 유입 없음. 미숙기 시험은 eat·move를 직접 부른다."""
    w = taste_world("taste", init_agents=1, min_agents=0, food_inflow=0.0, **kw)
    assert w.s.alive[0]
    return w


def _eat_one(w, kind, age, e=1.0):
    """개체 0(나이 age = eat가 보는 증가 뒤 나이)의 자리에 kind 알 하나를 놓고 eat를 한 번 부른다.
    (에너지 변화, eaten 변화, L_LOST 변화)를 돌려준다."""
    s, cfg = w.s, w.cfg
    s.age[0] = age
    s.energy[0] = 2.0
    eaten0 = float(s.eaten[0])
    lost0 = float(s.ledger[k.L_LOST])
    _place(w, e=e, kind=kind, x=float(s.x[0]), y=float(s.y[0]))
    k.build_grid(s.fx, s.fy, s.f_alive, 1.0 / cfg.cell, cfg.grid_w, cfg.grid_h, s.f_start, s.f_items, s.cursor)
    k.eat(w.step_no, s, w.fp, w.ip)
    return float(s.energy[0]) - 2.0, float(s.eaten[0]) - eaten0, float(s.ledger[k.L_LOST]) - lost0


def _good(w) -> int:
    return k.good_kind(w.cfg.seed, w.step_no, w.cfg.kind_flip_period)


def test_juv_ramp_hand_values():
    assert k.juv_ramp(0, 0.5, 200) == pytest.approx(0.5)
    assert k.juv_ramp(100, 0.5, 200) == pytest.approx(0.75)
    assert k.juv_ramp(199, 0.5, 200) == pytest.approx(0.5 + 0.5 * 199 / 200)
    assert k.juv_ramp(200, 0.5, 200) == 1.0
    assert k.juv_ramp(500, 0.5, 200) == 1.0
    assert k.juv_ramp(0, 0.3, 200) == pytest.approx(0.3)
    assert k.juv_ramp(100, 0.3, 200) == pytest.approx(0.65)
    assert k.juv_ramp(0, 1.0, 200) == 1.0          # 손잡이를 1로 두면 효과 없음


@pytest.mark.parametrize("a", [0, 100, 200, 300])
def test_eat_efficiency_by_age(a):
    w = _lone_juv(juv_enabled=True)
    dE, dEaten, dLost = _eat_one(w, _good(w), age=a + 1)       # eat는 증가 뒤 나이(a + 1)를 본다
    ramp = 0.5 + 0.5 * min(1.0, a / 200.0)
    assert dE == pytest.approx(1.0 * ramp)
    assert dEaten == pytest.approx(1.0 * ramp)
    assert dLost == pytest.approx(1.0 * (1.0 - ramp), abs=1e-12)
    if a >= 200:
        assert dE == 1.0 and dLost == 0.0                      # 어른은 옛 동작


def test_eat_juv_off_is_old_behavior():
    w = _lone_juv()
    dE, dEaten, dLost = _eat_one(w, _good(w), age=1)
    assert dE == 1.0 and dEaten == 1.0 and dLost == 0.0


@pytest.mark.parametrize("kind", [k.KIND_CORPSE, k.KIND_MEAT])
def test_eat_corpse_full_efficiency(kind):
    w = _lone_juv(juv_enabled=True)
    dE, dEaten, dLost = _eat_one(w, kind, age=1, e=3.0)        # 갓 태어난 새끼가 사체·고기를 먹는다
    assert dE == 3.0 and dEaten == 3.0 and dLost == 0.0
    # 같은 새끼가 들판 알을 먹으면 효율이 곱해진다(양성 대조)
    dE, _, dLost = _eat_one(w, _good(w), age=1, e=3.0)
    assert dE == pytest.approx(1.5) and dLost == pytest.approx(1.5)


def test_eat_poison_mult_juvenile_only():
    w = _lone_juv(juv_enabled=True, juv_poison_mult=3.0)
    bad = 1 - _good(w)
    p = float(w.cfg.poison)
    dE, dEaten, dLost = _eat_one(w, bad, age=51)               # 미숙기
    assert dE == pytest.approx(-p * 3.0) and dEaten == 0.0
    assert dLost == pytest.approx(1.0 + p * 3.0)
    dE, dEaten, dLost = _eat_one(w, bad, age=201)              # a = 200: 어른
    assert dE == pytest.approx(-p) and dLost == pytest.approx(1.0 + p)
    w2 = _lone_juv()                                           # 꺼짐이면 배수 없음
    dE, _, dLost = _eat_one(w2, 1 - _good(w2), age=51)
    assert dE == pytest.approx(-p) and dLost == pytest.approx(1.0 + p)


def _speed_after_move(w, age, v0=10.0) -> float:
    """개체 0이 age(증가 전 나이)에서 명령 없이 큰 속도로 한 스텝 움직인 뒤의 속력."""
    s = w.s
    s.age[0] = age
    s.vx[0], s.vy[0] = v0, 0.0
    s.act[0, :] = 0.0
    k.move(s, w.fp, w.ip)
    assert s.age[0] == age + 1                                 # age += 1은 그대로
    return float(np.hypot(s.vx[0], s.vy[0]))


def test_move_speed_cap_by_age():
    w = _lone_juv(juv_enabled=True)
    vmax = float(w.cfg.v_max)
    assert _speed_after_move(w, 0) == pytest.approx(vmax * 0.3)
    assert _speed_after_move(w, 100) == pytest.approx(vmax * 0.65)     # move는 증가 전 나이를 a로 쓴다
    assert _speed_after_move(w, 199) == pytest.approx(vmax * (0.3 + 0.7 * 199 / 200))
    assert _speed_after_move(w, 200) == pytest.approx(vmax)            # 어른
    assert _speed_after_move(w, 900) == pytest.approx(vmax)
    w2 = _lone_juv()
    assert _speed_after_move(w2, 0) == pytest.approx(vmax)             # 꺼짐: 옛 동작


def _neutral_pair(*presets, **kw):
    off = taste_world(*presets, **kw)
    on = taste_world(*presets, juv_enabled=True, juv_eat_floor=1.0, juv_move_floor=1.0, juv_poison_mult=1.0, **kw)
    return off, on


@pytest.mark.parametrize("presets,kw", [
    (("taste",), {}),
    (("taste", "conflict"), dict(soc_taste_enabled=True)),
])
def test_juv_knobs_neutral_hash_equals_off(presets, kw):
    off, on = _neutral_pair(*presets, **kw)
    off.run(300)
    on.run(300)
    assert on.cfg.juv_enabled and not off.cfg.juv_enabled
    assert on.cfg.conflict_enabled == off.cfg.conflict_enabled == ("conflict" in presets)
    assert on.state_hash() == off.state_hash()


def test_juv_defaults_change_world():
    # 양성 대조: 손잡이를 기본값으로 두면 세계가 달라진다(중립 시험이 아무것도 안 보는 시험이 아님을 확인)
    off = taste_world("taste")
    on = taste_world("taste", juv_enabled=True)
    off.run(300)
    on.run(300)
    assert on.state_hash() != off.state_hash()


def test_ledger_closes_with_juv():
    w = taste_world("taste", juv_enabled=True, juv_poison_mult=2.0)
    w.run(1500)
    assert w.s.ledger[k.L_LOST] > 0
    assert rel_residual(w) < 1e-9


# ---- 과제 3: juv_pass(곁 머무름 계수기·졸업표·미숙기 사망) ----
def _juv_world(n, **kw) -> World:
    """개체 n마리(전부 어른, 부모 없음)를 서로 멀리 놓은 세계. 격자는 짓지 않는다(juv_pass는 격자를 안 쓴다)."""
    kw.setdefault("juv_enabled", True)
    w = taste_world("taste", init_agents=n, min_agents=0, food_inflow=0.0, **kw)
    s = w.s
    assert int(s.alive.sum()) == n
    for i in range(n):
        s.x[i] = 20.0 + 40.0 * (i % 6)
        s.y[i] = 20.0 + 40.0 * (i // 6)
        s.ids[i] = 1000 + i
        s.parent[i] = -1
        s.age[i] = 250
        s.energy[i] = 2.0
        s.tag[i] = 0
        s.taste_m[i] = 0.0
        s.know_age[i] = -1
        s.know_src[i] = 0
        for name in ("n_seen_par", "n_seen_par_poison", "n_taught", "n_taught_knower", "par_near_steps",
                     "par_alive_steps", "juv_alive_steps", "adult_near_sum", "n_seen", "n_meal", "n_poison"):
            getattr(s, name)[i] = 0
    return w


def _at_xy(w, i, x, y):
    w.s.x[i] = x
    w.s.y[i] = y


def _child(w, i, parent_idx, age, tag=0, born_here=True):
    """i를 미숙기 새끼로 만든다. born_here면 juv_alive_steps가 다음 juv_pass 뒤 age와 같아진다(태어난 스텝부터 센 개체)."""
    s = w.s
    s.parent[i] = s.ids[parent_idx] if parent_idx is not None else -1
    s.age[i] = age
    s.tag[i] = tag
    s.juv_alive_steps[i] = age - 1 if born_here else max(age - 31, 0)


def _jpass(w):
    k.juv_pass(w.step_no, w.s, w.fp, w.ip)


def test_juv_pass_counts_parent_near():
    w = _juv_world(6)
    s = w.s
    _at_xy(w, 1, 23.0, 20.0)                       # 부모(0번)에게서 3: 곁
    _child(w, 1, 0, 50)
    _child(w, 2, 0, 50)                            # 부모에게서 멀다(2번은 (100, 20)쯤)
    _child(w, 3, None, 50)                         # 부모 id가 -1
    _child(w, 4, 5, 50)
    s.alive[5] = False                             # 부모가 죽었다
    _jpass(w)
    assert s.par_near_steps[1] == 1 and s.par_alive_steps[1] == 1 and s.juv_alive_steps[1] == 50
    assert s.par_near_steps[2] == 0 and s.par_alive_steps[2] == 1 and s.juv_alive_steps[2] == 50
    assert s.par_alive_steps[3] == 0 and s.juv_alive_steps[3] == 50
    assert s.par_alive_steps[4] == 0 and s.par_near_steps[4] == 0 and s.juv_alive_steps[4] == 50
    assert s.par_slot[1] == 0 and s.par_slot[2] == 0 and s.par_slot[3] == -1 and s.par_slot[4] == -1
    # 어른(0번)은 안 센다
    assert s.juv_alive_steps[0] == 0 and s.par_alive_steps[0] == 0 and s.par_slot[0] == -1
    _jpass(w)                                      # 두 번째 스텝: 누적
    assert s.par_near_steps[1] == 2 and s.par_alive_steps[1] == 2


def test_juv_pass_cell_boundary_and_wrap():
    w = _juv_world(8)
    s = w.s
    cell, size, r = w.cfg.cell, w.cfg.world_w, w.cfg.soc_radius
    assert r == cell == 8.0
    # (부모, 새끼, 기대): 칸 경계 양쪽, 거리 r 바로 안·밖, 토러스 반대편
    cases = [(7.8, 8.2, True), (100.0, 100.0 + 7.99, True), (140.0, 140.0 + 8.01, False), (0.5, size - 0.5, True)]
    for p, (xp, xc, _) in enumerate(cases):
        i_par, i_kid = 2 * p, 2 * p + 1
        _at_xy(w, i_par, xp, 20.0 + 40.0 * p)
        _at_xy(w, i_kid, xc, 20.0 + 40.0 * p)
        _child(w, i_kid, i_par, 50)
    _jpass(w)
    for p, (_, _, near) in enumerate(cases):
        assert s.par_near_steps[2 * p + 1] == int(near), p
        assert s.par_alive_steps[2 * p + 1] == 1


def test_adult_near_sum_excludes_parent():
    w = _juv_world(8)
    s = w.s
    _at_xy(w, 0, 100.0, 100.0)                     # 부모
    _at_xy(w, 1, 102.0, 100.0)                     # 새끼
    _at_xy(w, 2, 98.0, 100.0)                      # 부모 아닌 어른(곁)
    _at_xy(w, 3, 100.0, 105.0)                     # 부모 아닌 어른(곁)
    _at_xy(w, 4, 100.0, 140.0)                     # 어른, 멀다
    _at_xy(w, 5, 101.0, 98.0)                      # 미숙기 이웃: 안 센다
    _child(w, 1, 0, 50)
    _child(w, 5, 0, 60)
    _at_xy(w, 6, 103.0, 101.0)                     # 부모 없는 새끼도 곁의 어른 셋(0·2·3)을 센다
    _child(w, 6, None, 40)
    _jpass(w)
    _jpass(w)
    assert s.adult_near_sum[1] == 2 * 2            # 어른 둘 × 두 스텝(부모는 빼고, 미숙기 이웃도 뺀다)
    assert s.par_near_steps[1] == 2
    assert s.adult_near_sum[6] == 3 * 2


@pytest.mark.parametrize("kw", [dict(juv_enabled=True), dict(juv_measure=True)])
def test_juv_pass_in_juv_only_world(kw):
    w = taste_world("taste", food_inflow=12.0, food_cap=1500, **kw)       # 출생이 이어지게 먹이를 넉넉히
    assert not w.cfg.soc_taste_enabled and not w.cfg.conflict_enabled
    assert not w.cfg.teach_enabled and not w.cfg.feed_enabled
    assert w.cfg.juv_enabled == kw.get("juv_enabled", False)
    w.run(300)
    s, a = w.s, w.s.alive
    assert w.s.counters[k.C_BIRTHS] > 0
    assert int(s.juv_alive_steps[a].sum()) > 0
    assert int(s.par_alive_steps[a].sum()) > 0
    assert (s.par_alive_steps[a] <= s.juv_alive_steps[a]).all()
    assert (s.par_near_steps[a] <= s.par_alive_steps[a]).all()


def test_juv_measure_hash_equals_off():
    off = taste_world("taste")
    on = taste_world("taste", juv_measure=True)
    off.run(300)
    on.run(300)
    assert on.cfg.juv_measure and not on.cfg.juv_enabled
    assert on.state_hash() == off.state_hash()
    assert int(on.s.juv_alive_steps[on.s.alive].sum()) > 0       # 양성 대조: 계수기는 실제로 돈다


def test_juv_pass_off_not_called(monkeypatch):
    calls = []
    real = k.juv_pass
    monkeypatch.setattr(k, "juv_pass", lambda *a: calls.append(1) or real(*a))
    off = taste_world("taste")
    off.run(5)
    assert calls == []
    on = taste_world("taste", juv_measure=True)
    on.run(5)
    assert len(calls) == 5


def test_grad_record_conditions():
    w = _juv_world(12)
    s = w.s
    bad = 1 - k.cfg_good_kind(w.cfg, w.step_no)
    steps = w.cfg.juv_steps
    assert (steps // 4, steps // 2, steps) == (50, 100, 200)
    c, nr1, nr2 = 6, 7, 8                          # 기록 대상(태그 2), 기록 안 되는 둘(태그 3)
    for age, taste, know_age in ((50, -0.5, -1), (100, -0.1, 40), (200, -0.4, 40)):
        _child(w, c, 0, age, tag=2)
        s.taste_m[c, bad] = taste
        s.know_age[c] = know_age
        _child(w, nr1, None, age, tag=3)           # parent == -1(inject_agent로 들어온 개체)
        _child(w, nr2, 1, age, tag=3, born_here=False)    # 계수기를 중간부터 센 개체(juv_alive_steps != age)
        s.taste_m[nr1, bad] = s.taste_m[nr2, bad] = -0.9
        if age == steps:
            s.par_near_steps[c], s.par_alive_steps[c], s.adult_near_sum[c] = 30, 100, 250
            s.n_seen[c], s.n_seen_par[c], s.n_seen_par_poison[c] = 12, 5, 2
            s.n_taught[c], s.n_taught_knower[c] = 4, 3
            s.n_meal[c], s.n_poison[c] = 6, 2
            s.energy[c] = 3.5
            s.know_src[c] = 2
        _jpass(w)
        row = s.grad_tab[2]
        if age == 50:
            assert (row[k.GT_N1], row[k.GT_K1]) == (1, 1) and row[k.GT_M1] == pytest.approx(0.5)
            assert row[k.GT_N2] == 0 and row[k.GT_GRAD] == 0
        if age == 100:
            assert (row[k.GT_N2], row[k.GT_K2]) == (1, 1)      # 모양 기억은 얕아도 앎은 누적(한 번 알면 유지)
            assert row[k.GT_M2] == pytest.approx(0.1) and row[k.GT_GRAD] == 0
    assert row[k.GT_GRAD] == 1 and row[k.GT_DEAD] == 0
    assert row[k.GT_M3] == pytest.approx(0.4) and row[k.GT_K3] == 1
    assert row[k.GT_SRC2] == 1 and row[k.GT_SRC1] == row[k.GT_SRC3] == row[k.GT_SRC4] == 0
    assert row[k.GT_PAR_NEAR] == 30 and row[k.GT_PAR_ALIVE] == 101      # 그 스텝의 계수까지 합산
    assert row[k.GT_ADULT_NEAR] == 250 and row[k.GT_ALIVE_STEPS] == 200
    assert (row[k.GT_SEEN], row[k.GT_SEEN_PAR], row[k.GT_SEEN_PAR_POISON]) == (12, 5, 2)
    assert (row[k.GT_TAUGHT], row[k.GT_TAUGHT_KNOWER]) == (4, 3)
    assert (row[k.GT_MEALS], row[k.GT_POISON_MEALS]) == (6, 2)
    assert row[k.GT_E_GRAD] == pytest.approx(3.5)
    assert not s.grad_tab[3].any()                 # 기록 조건을 못 채운 개체는 어느 스냅샷·졸업에도 안 들어간다
    assert not np.delete(s.grad_tab, 2, axis=0).any()


def test_grad_snapshot_skipped_when_kinds_off():
    # 먹이 종류가 꺼진 세계: 독 칸이 없으니 스냅샷·앎은 건너뛰고 졸업 수·곁 계수만 쌓인다
    w = World(small("stage2", juv_enabled=True, init_agents=3, min_agents=0, food_inflow=0.0))
    assert not w.cfg.food_kinds_enabled
    s = w.s
    for i in range(3):
        s.ids[i] = 1000 + i
        s.parent[i] = -1
        s.age[i] = 250
        s.x[i], s.y[i] = 20.0 + 40.0 * i, 20.0
    s.parent[1] = 1000
    s.age[1] = 200
    s.juv_alive_steps[1] = 199
    _jpass(w)
    row = w.s.grad_tab[0]
    assert row[k.GT_GRAD] == 1 and row[k.GT_N1] == row[k.GT_N2] == row[k.GT_M3] == row[k.GT_K3] == 0


def test_juvenile_death_recorded():
    w = _juv_world(8)
    s = w.s
    steps = w.cfg.juv_steps
    _child(w, 6, 0, 50, tag=4)                     # 미숙기에 굶어 죽는다
    _child(w, 5, 0, 50, tag=4, born_here=False)    # 계수기를 중간부터 센 개체
    _child(w, 4, 0, steps, tag=4)                  # 나이 == juv_steps: 이 스텝 juv_pass가 졸업으로 센 뒤라 사망으로 또 안 센다
    s.juv_alive_steps[6] = 50
    s.juv_alive_steps[5] = 20
    s.juv_alive_steps[4] = steps
    s.age[7] = 250                                 # 어른이 굶어 죽는다(parent도 있다)
    s.parent[7] = s.ids[0]
    s.juv_alive_steps[7] = 250
    for i in (4, 5, 6, 7):
        s.energy[i] = 0.0
    nd = k.process_deaths(w.step_no, s, w.fp, w.ip)
    assert nd == 4 and not s.alive[[4, 5, 6, 7]].any()
    assert s.grad_tab[4, k.GT_DEAD] == 1
    assert s.grad_tab.sum() == 1


def test_juvenile_death_not_recorded_when_off():
    w = _juv_world(3, juv_enabled=False)
    _child(w, 1, 0, 50)
    w.s.juv_alive_steps[1] = 50
    w.s.energy[1] = 0.0
    k.process_deaths(w.step_no, w.s, w.fp, w.ip)
    assert not w.s.alive[1] and not w.s.grad_tab.any()


def test_grad_tag_fold_no_overflow():
    w = _juv_world(9)
    s = w.s
    steps = w.cfg.juv_steps
    guard = (s.ledger.copy(), s.counters.copy(), s.energy.copy(), s.f_energy.copy())
    shape = s.grad_tab.shape
    for i, tag in ((3, 5), (4, 99), (5, -3)):
        _child(w, i, 0, steps, tag=tag)
    _jpass(w)
    assert s.grad_tab.shape == shape == (k.TAG_ROWS, k.N_GT)
    for a, b in zip(guard, (s.ledger, s.counters, s.energy, s.f_energy)):
        assert np.array_equal(a, b)
    assert s.grad_tab[5, k.GT_GRAD] == 1 and s.grad_tab[7, k.GT_GRAD] == 1 and s.grad_tab[0, k.GT_GRAD] == 1
    assert s.grad_tab[:, k.GT_GRAD].sum() == 3
    # 사망도 같은 접기
    for i, tag in ((6, 99), (7, -3)):
        _child(w, i, 0, 50, tag=tag)
        s.juv_alive_steps[i] = 50
        s.energy[i] = 0.0
    k.process_deaths(w.step_no, s, w.fp, w.ip)
    assert s.grad_tab[7, k.GT_DEAD] == 1 and s.grad_tab[0, k.GT_DEAD] == 1 and s.grad_tab[:, k.GT_DEAD].sum() == 2


def test_new_counters_reset_on_birth():
    w = _juv_world(3)
    s = w.s
    j = int(np.flatnonzero(~s.alive)[0])
    for name in NEW_UNSAVED:
        getattr(s, name)[j] = 7
    s.n_seen_par[j] = s.n_seen_par_poison[j] = 3
    k._reset_body(s, j, 999, -1, 0, 0, w.step_no, 10.0, 10.0, 0.0, 1.0)
    for name in NEW_UNSAVED:
        want = {"know_age": -1, "par_slot": -1}.get(name, 0)
        assert getattr(s, name)[j] == want, name


def test_world_grad_table_api(tmp_path):
    from kkumteul import checkpoint as ckpt
    w = _juv_world(8)
    s = w.s
    _child(w, 3, 0, w.cfg.juv_steps, tag=2)
    _jpass(w)
    tab = w.grad_table()
    assert sorted(tab) == list(range(k.TAG_ROWS))
    assert tuple(tab[2]) == k.GT_NAMES
    assert tab[2]["grad"] == 1.0 and tab[2]["alive_steps"] == float(w.cfg.juv_steps) and tab[0]["grad"] == 0.0
    w.reset_grad_table()
    assert not s.grad_tab.any() and all(v == 0.0 for row in w.grad_table().values() for v in row.values())
    # 체크포인트로 이어 돌린 직후 이미 자라던 새끼는 기록에 안 들어간다
    w = taste_world("taste", juv_enabled=True, food_inflow=12.0, food_cap=1500)
    w.run(260)
    a = w.s.alive
    assert (a & (w.s.age <= w.cfg.juv_steps) & (w.s.parent >= 0)).any()    # 자라던 새끼가 있다
    path = ckpt.save(w, tmp_path)
    w2, _ = ckpt.load(path)
    assert (w2.s.juv_alive_steps == 0).all()
    nid0 = int(w2.s.counters[k.C_NEXT_ID])                       # 재개 뒤 새로 매겨진 id = 재개 뒤 태어난 개체(재주입 새끼 포함)
    n_age0 = int((w2.s.alive & (w2.s.age == 0)).sum())          # 저장 직전 스텝에 태어난 새끼는 재개 첫 스텝부터 센다
    w2.run(1)
    grown = w2.s.alive & (w2.s.age <= w2.cfg.juv_steps) & (w2.s.age > 1)
    assert grown.any() and (w2.s.juv_alive_steps[grown] == 1).all()     # 이미 자라던 새끼는 age와 어긋난다
    w2.run(250)
    # 졸업표에 든 새끼는 재개 뒤에 태어난 개체뿐이다: 졸업 + 사망 <= 재개 뒤 새 id 수(+ 재개 때 갓 난 새끼)
    t = w2.grad_table()
    n_rec = sum(row["grad"] + row["dead"] for row in t.values())
    assert n_rec > 0
    assert n_rec <= int(w2.s.counters[k.C_NEXT_ID]) - nid0 + n_age0


def test_juv_pass_cost_scales():
    # 어림 확인: 산 개체 600·새끼 120이면 juv_pass 한 번이 현실적 시간 안에 끝난다
    import time
    w = _juv_world(6)
    s = w.s
    rng = np.random.default_rng(0)
    idx = np.arange(600)
    s.alive[:] = False
    s.alive[idx] = True
    s.x[idx] = rng.uniform(0.0, w.cfg.world_w, 600)
    s.y[idx] = rng.uniform(0.0, w.cfg.world_h, 600)
    s.ids[idx] = 1000 + idx
    s.age[idx] = 250
    s.parent[idx] = -1
    s.parent[:120] = s.ids[300:420]
    s.age[:120] = 50
    s.juv_alive_steps[:120] = 49
    _jpass(w)
    t0 = time.perf_counter()
    for _ in range(20):
        _jpass(w)
    per = (time.perf_counter() - t0) / 20
    assert per < 0.005, per


def test_summary_juv_keys():
    off = taste_world("taste")
    off.run(3)
    assert not {"juv_births_recorded", "grad_rate", "par_near_frac"} & set(off.summary())
    w = _juv_world(8)
    assert w.summary()["juv_births_recorded"] == 0 and "grad_rate" not in w.summary() and "par_near_frac" not in w.summary()
    _child(w, 3, 0, w.cfg.juv_steps)
    _at_xy(w, 3, 21.0, 20.0)
    _child(w, 4, 0, 50, tag=1)
    w.s.energy[4] = 0.0
    _jpass(w)
    k.process_deaths(w.step_no, w.s, w.fp, w.ip)
    sm = w.summary()
    assert sm["juv_births_recorded"] == 2 and sm["grad_rate"] == pytest.approx(0.5) and sm["par_near_frac"] == 1.0


# ---- 과제 4: show_g·feed_g 변이 ----
def _gparent(on_teach: bool, on_feed: bool, g_show: float, g_feed: float, **kw) -> World:
    """부모 한 마리(칸 0)만 있는 세계. 켜짐 여부로 teach/feed 장치를 고른다."""
    extra = dict(soc_taste_enabled=True, juv_enabled=True, teach_enabled=on_teach, feed_enabled=on_feed)
    w = taste_world("taste", init_agents=1, food_inflow=0.0, min_agents=0, **{**extra, **kw})
    w.s.show_g[0] = g_show
    w.s.feed_g[0] = g_feed
    return w


def test_show_feed_gene_copied_when_off():
    w = _gparent(False, False, 0.3, 0.7, teach_gene_sigma=0.5, feed_gene_sigma=0.5)
    assert w.ip[k.Q_TEACH] == 0 and w.ip[k.Q_FEED] == 0
    j = _natural_child(w)
    assert w.s.show_g[j] == pytest.approx(0.3) and w.s.feed_g[j] == pytest.approx(0.7)
    # 켜져 있어도 변이가 꺼지면(mutation_enabled=False) 그대로 베낀다
    w = _gparent(True, True, 0.3, 0.7, mutation_enabled=False)
    j = _natural_child(w)
    assert w.s.show_g[j] == pytest.approx(0.3) and w.s.feed_g[j] == pytest.approx(0.7)
    # 한쪽만 켜면 그쪽만 변이
    w = _gparent(True, False, 0.3, 0.7, teach_gene_sigma=0.1, feed_gene_sigma=0.1)
    j = _natural_child(w)
    assert w.s.show_g[j] != pytest.approx(0.3) and w.s.feed_g[j] == pytest.approx(0.7)
    w = _gparent(False, True, 0.3, 0.7, teach_gene_sigma=0.1, feed_gene_sigma=0.1)
    j = _natural_child(w)
    assert w.s.show_g[j] == pytest.approx(0.3) and w.s.feed_g[j] != pytest.approx(0.7)


def test_reinjected_child_copies_gene_when_off():
    w = _gparent(False, False, 0.3, 0.7)
    side = measure.side_from_world(w)
    assert "show_g" in side["genomes"] and "feed_g" in side["genomes"]
    j_id = measure.inject_agent(w, side["genomes"], 0, 1, 10.0, 10.0, 0.0, 0)
    j = int(np.flatnonzero(w.s.ids == j_id)[0])
    assert w.s.show_g[j] == pytest.approx(0.3) and w.s.feed_g[j] == pytest.approx(0.7)


def test_show_gene_mutates_and_clips_when_on():
    vals_s, vals_f = [], []
    for sd in range(24):
        w = _gparent(True, True, 0.5, 0.5, teach_gene_sigma=0.1, feed_gene_sigma=0.1, seed=sd)
        assert w.ip[k.Q_TEACH] == 1 and w.ip[k.Q_FEED] == 1
        j = _natural_child(w)
        vals_s.append(float(w.s.show_g[j]))
        vals_f.append(float(w.s.feed_g[j]))
        assert w.s.show_g[0] == pytest.approx(0.5) and w.s.feed_g[0] == pytest.approx(0.5)
    for vals in (vals_s, vals_f):
        assert len(set(np.round(vals, 6))) > 10
        assert 0.35 < np.mean(vals) < 0.65 and max(abs(v - 0.5) for v in vals) < 0.5
    # [0, 1] 밖으로 안 나간다
    cs, cf = [], []
    for sd in range(12):
        for g in (0.0, 1.0):
            w = _gparent(True, True, g, g, teach_gene_sigma=1.0, feed_gene_sigma=1.0, seed=sd)
            j = _natural_child(w)
            cs.append(float(w.s.show_g[j]))
            cf.append(float(w.s.feed_g[j]))
    for vals in (cs, cf):
        assert min(vals) >= 0.0 and max(vals) <= 1.0
        assert min(vals) == 0.0 or max(vals) == 1.0          # 실제로 잘렸다
        assert len(set(np.round(vals, 6))) > 4


def test_gene_uses_own_rng_streams():
    # 같은 자식 id·씨앗이면 show_g의 변이는 S_TEACH_GENE 난수 하나로 정해진다
    from kkumteul import rng as rngmod
    w = _gparent(True, True, 0.5, 0.5, teach_gene_sigma=0.1, feed_gene_sigma=0.1, seed=7)
    j = _natural_child(w)
    cid = int(w.s.ids[j])
    seed = int(w.ip[k.Q_SEED])
    exp_s = min(max(0.5 + 0.1 * float(rngmod.normal(seed, rngmod.S_TEACH_GENE, cid, 0)), 0.0), 1.0)
    exp_f = min(max(0.5 + 0.1 * float(rngmod.normal(seed, rngmod.S_FEED_GENE, cid, 0)), 0.0), 1.0)
    assert w.s.show_g[j] == pytest.approx(exp_s, abs=1e-5)
    assert w.s.feed_g[j] == pytest.approx(exp_f, abs=1e-5)


def test_gene_zero_for_initial_and_random_agent():
    w = _gparent(True, True, 0.0, 0.0)
    assert (w.s.show_g == 0).all() and (w.s.feed_g == 0).all()
    w2 = taste_world("taste", soc_taste_enabled=True, juv_enabled=True, teach_enabled=True, feed_enabled=True)
    assert (w2.s.show_g[w2.s.alive] == 0).all() and (w2.s.feed_g[w2.s.alive] == 0).all()
    w2.s.show_g[:] = 0.4
    w2.s.feed_g[:] = 0.4
    j = int(np.flatnonzero(~w2.s.alive)[0])
    k._reset_body(w2.s, j, 999, -1, 0, 0, w2.step_no, 10.0, 10.0, 0.0, 1.0)
    assert w2.s.show_g[j] == 0.0 and w2.s.feed_g[j] == 0.0


def test_gene_checkpoint_roundtrip():
    w = _gparent(True, True, 0.5, 0.5, teach_gene_sigma=0.1, feed_gene_sigma=0.1, seed=3)
    j = _natural_child(w)
    gs, gf = float(w.s.show_g[j]), float(w.s.feed_g[j])
    assert gs != 0.5 and gf != 0.5
    arrays = {kk: v.copy() for kk, v in w.state_arrays().items()}
    w2 = World(w.cfg, populate=False)
    w2.step_no = w.step_no
    assert w2.load_arrays(arrays)
    j2 = int(np.flatnonzero(w2.s.ids == w.s.ids[j])[0])
    assert w2.s.show_g[j2] == pytest.approx(gs) and w2.s.feed_g[j2] == pytest.approx(gf)


def test_summary_show_feed_mean():
    w = _gparent(True, True, 0.3, 0.6)
    sm = w.summary()
    assert sm["show_g_mean"] == pytest.approx(0.3) and sm["feed_g_mean"] == pytest.approx(0.6)
    w = _gparent(True, False, 0.3, 0.6)
    sm = w.summary()
    assert "show_g_mean" in sm and "feed_g_mean" not in sm
    w = _gparent(False, True, 0.3, 0.6)
    sm = w.summary()
    assert "feed_g_mean" in sm and "show_g_mean" not in sm
    w = _gparent(False, False, 0.3, 0.6)
    assert not {"show_g_mean", "feed_g_mean"} & set(w.summary())


# ---- 과제 5: teach_pass(보여 주기 비용, near) ----
def _teach_world(n, **kw) -> World:
    """_juv_world에 보고 배우기(outcome)·보여 주기를 켠 세계. 개체는 서로 멀리 놓인다(0번 (20,20), 1번 (60,20) ...)."""
    kw.setdefault("soc_taste_enabled", True)
    kw.setdefault("teach_enabled", True)
    return _juv_world(n, **kw)


def _tpass(w):
    """juv_pass(par_slot을 채움) 뒤 teach_pass. 격자는 안 짓는다. 스텝 전 에너지를 돌려준다."""
    before = w.s.energy.copy()
    _jpass(w)
    k.teach_pass(w.step_no, w.s, w.fp, w.ip)
    return before


def _paid(w, before, i) -> float:
    return float(before[i] - w.s.energy[i])


def test_teach_cost_kin_only_when_own_juvenile_near():
    cost = 0.0005
    w = _teach_world(8)
    assert w.ip[k.Q_TEACH] == 1 and w.ip[k.Q_TEACH_WHEN] == 0
    s = w.s
    s.show_g[0] = 0.8
    _at_xy(w, 1, 23.0, 20.0)
    _child(w, 1, 0, 50)                            # 자기 새끼, 곁
    before = _tpass(w)
    assert _paid(w, before, 0) == pytest.approx(cost * 0.8)            # 양성 확인
    assert all(_paid(w, before, i) == 0.0 for i in range(1, 8))
    # 반지름 밖
    w = _teach_world(8)
    w.s.show_g[0] = 0.8
    _at_xy(w, 1, 20.0 + 9.0, 20.0)
    _child(w, 1, 0, 50)
    before = _tpass(w)
    assert _paid(w, before, 0) == 0.0
    # 새끼가 없다
    w = _teach_world(8)
    w.s.show_g[0] = 0.8
    before = _tpass(w)
    assert _paid(w, before, 0) == 0.0
    # 새끼가 죽었다
    w = _teach_world(8)
    w.s.show_g[0] = 0.8
    _at_xy(w, 1, 23.0, 20.0)
    _child(w, 1, 0, 50)
    w.s.alive[1] = False
    before = _tpass(w)
    assert _paid(w, before, 0) == 0.0
    # 강도 0이면 0, 강도는 [0, 1]로 자른다
    for g, want in ((0.0, 0.0), (1.7, cost), (-0.5, 0.0)):
        w = _teach_world(8)
        w.s.show_g[0] = g
        _at_xy(w, 1, 23.0, 20.0)
        _child(w, 1, 0, 50)
        before = _tpass(w)
        assert _paid(w, before, 0) == pytest.approx(want)
    # 새끼가 둘이어도 스텝당 한 번
    w = _teach_world(8)
    w.s.show_g[0] = 0.8
    _at_xy(w, 1, 23.0, 20.0)
    _at_xy(w, 2, 17.0, 22.0)
    _child(w, 1, 0, 50)
    _child(w, 2, 0, 60)
    before = _tpass(w)
    assert _paid(w, before, 0) == pytest.approx(cost * 0.8)


def test_teach_cost_kin_vs_any_for_nonkin_juvenile():
    cost = 0.0005
    for mode, want in (("kin", 0.0), ("any", cost * 0.8)):
        w = _teach_world(8, teach_mode=mode)
        w.s.show_g[0] = 0.8
        _at_xy(w, 1, 23.0, 20.0)
        _child(w, 1, 3, 50)                        # 부모는 멀리 있는 3번
        before = _tpass(w)
        assert _paid(w, before, 0) == pytest.approx(want), mode
        assert _paid(w, before, 3) == 0.0          # 3번은 새끼 곁이 아니다(kin에서도 any에서도 0)
    # any: 부모 없는 새끼 곁의 어른 모두가 낸다(각자 강도만큼)
    w = _teach_world(8, teach_mode="any")
    w.s.show_g[0] = 0.8
    w.s.show_g[2] = 0.4
    _at_xy(w, 1, 23.0, 20.0)
    _at_xy(w, 2, 20.0, 25.0)
    _child(w, 1, None, 50)
    before = _tpass(w)
    assert _paid(w, before, 0) == pytest.approx(cost * 0.8)
    assert _paid(w, before, 2) == pytest.approx(cost * 0.4)


def test_teach_cost_juvenile_not_payer():
    cost = 0.0005
    w = _teach_world(8)
    s = w.s
    s.show_g[0] = 1.0
    s.show_g[1] = 1.0
    _at_xy(w, 1, 23.0, 20.0)
    _at_xy(w, 2, 25.0, 20.0)
    _child(w, 1, 0, 50)                            # 1번은 미숙기이면서 2번의 부모
    _child(w, 2, 1, 30)
    before = _tpass(w)
    assert _paid(w, before, 0) == pytest.approx(cost)       # 어른 0번은 낸다
    assert _paid(w, before, 1) == 0.0                       # 미숙기 1번은 시연자가 아니다
    # any에서도 미숙기는 안 낸다
    w = _teach_world(8, teach_mode="any")
    w.s.show_g[1] = 1.0
    _at_xy(w, 1, 23.0, 20.0)
    _at_xy(w, 2, 25.0, 20.0)
    _child(w, 1, None, 50)
    _child(w, 2, None, 30)
    before = _tpass(w)
    assert all(_paid(w, before, i) == 0.0 for i in range(8))


def test_teach_cost_modes():
    cost = 0.0005
    w = _teach_world(8)
    kp = 1 - k.cfg_good_kind(w.cfg, w.step_no)

    def run(mode, knows):
        w = _teach_world(8, teach_mode=mode)
        w.s.show_g[0] = 0.8
        _at_xy(w, 1, 23.0, 20.0)
        _child(w, 1, 0, 50)
        w.s.taste_m[0, kp] = -0.5 if knows else 0.0
        before = _tpass(w)
        return _paid(w, before, 0)

    assert run("costonly", False) == pytest.approx(cost * 0.8)
    assert run("costonly", True) == pytest.approx(cost * 0.8)
    assert run("drift", False) == 0.0 and run("drift", True) == 0.0
    assert run("knower", False) == 0.0
    assert run("knower", True) == pytest.approx(cost * 0.8)
    assert run("kin", False) == pytest.approx(cost * 0.8)
    # 시연자의 앎은 그 스텝 기억만 본다: 미숙기에 한 번 안 부모(know_age >= 0)도 지금 기억이 0이면 모른다
    w = _teach_world(8, teach_mode="knower")
    w.s.show_g[0] = 0.8
    _at_xy(w, 1, 23.0, 20.0)
    _child(w, 1, 0, 50)
    w.s.know_age[0] = 10
    w.s.taste_m[0, kp] = 0.0
    before = _tpass(w)
    assert _paid(w, before, 0) == 0.0


def test_teach_cost_ledger_and_tab():
    cost = 0.0005
    w = _teach_world(12)
    s = w.s
    for p, tag, g in ((0, 5, 0.8), (2, 99, 0.5), (4, -3, 1.0)):
        s.show_g[p] = g
        s.tag[p] = tag
        _at_xy(w, p + 1, s.x[p] + 3.0, s.y[p])
        _child(w, p + 1, p, 50)
    led0 = float(s.ledger[k.L_BRAIN])
    before = _tpass(w)
    drop = sum(_paid(w, before, i) for i in range(12))
    assert drop == pytest.approx(cost * (0.8 + 0.5 + 1.0))
    assert float(s.ledger[k.L_BRAIN]) - led0 == pytest.approx(drop)        # 에너지 감소 = L_BRAIN 증가
    assert s.grad_tab[5, k.GT_TEACH_COST] == pytest.approx(cost * 0.8)
    assert s.grad_tab[7, k.GT_TEACH_COST] == pytest.approx(cost * 0.5)     # 99는 7행에 접힌다
    assert s.grad_tab[0, k.GT_TEACH_COST] == pytest.approx(cost * 1.0)     # -3은 0행에 접힌다
    assert s.grad_tab[:, k.GT_TEACH_COST].sum() == pytest.approx(drop)


def test_teach_ledger_closes_with_teach_pass():
    w = taste_world("taste", soc_taste_enabled=True, juv_enabled=True, teach_enabled=True, teach_cost=0.02)
    w.s.show_g[w.s.alive] = 0.8                    # 돌아가는 동안 비용이 실제로 걷히게
    w.run(1500)
    assert rel_residual(w) < 1e-9
    assert float(w.s.grad_tab[:, k.GT_TEACH_COST].sum()) > 0.0              # 양성 확인


def test_teach_when_meal_skips_teach_pass(monkeypatch):
    calls = []
    real = k.teach_pass
    monkeypatch.setattr(k, "teach_pass", lambda *a: calls.append(1) or real(*a))
    w = taste_world("taste", soc_taste_enabled=True, juv_enabled=True, teach_enabled=True, teach_cost_when="meal")
    w.run(5)
    assert calls == []
    w = taste_world("taste", soc_taste_enabled=True, juv_enabled=True, teach_enabled=True, teach_cost_when="near")
    w.run(5)
    assert len(calls) == 5                         # 양성 대조


def test_teach_pass_off_not_called(monkeypatch):
    calls = []
    real = k.teach_pass
    monkeypatch.setattr(k, "teach_pass", lambda *a: calls.append(1) or real(*a))
    taste_world("taste").run(5)
    taste_world("taste", soc_taste_enabled=True, juv_enabled=True).run(5)       # teach_enabled=false
    taste_world("taste", juv_measure=True).run(5)
    assert calls == []


def test_teach_pass_not_requiring_grid():
    """손으로 놓은 쌍: build_grid를 안 부르고(a_start가 비어 있어도) 동작한다."""
    w = _teach_world(8)
    w.s.show_g[0] = 0.8
    _at_xy(w, 1, 23.0, 20.0)
    _child(w, 1, 0, 50)
    s = w.s
    s.a_start[:] = 0
    s.a_items[:] = 0
    before = _tpass(w)
    assert _paid(w, before, 0) == pytest.approx(0.0005 * 0.8)


# ---- 과제 6: watch_meals의 boost·앎 도우미·meal 비용·_kin_class ----
def _wm_world(n=8, **kw) -> World:
    """_teach_world에 보고 배우는 바탕 속도 0.5. 끼니 기록·soc_g는 0."""
    kw.setdefault("soc_rate", 0.5)
    w = _teach_world(n, **kw)
    w.s.soc_g[:] = 0.0
    w.s.meal_o[:] = 0.0
    w.s.meal_k[:] = 0
    return w


def _wm(w):
    s, cfg = w.s, w.cfg
    k.build_grid(s.x, s.y, s.alive, 1.0 / cfg.cell, cfg.grid_w, cfg.grid_h, s.a_start, s.a_items, s.cursor)
    k.watch_meals(w.step_no, s, w.fp, w.ip)


def _wm_meal(w, j, mc, o):
    w.s.meal_k[j] = mc
    w.s.meal_o[j] = o


def _kp(w) -> int:
    return 1 - k.cfg_good_kind(w.cfg, w.step_no)


def _wm_scene(w, show=0.6, mc=1, o=-1.0):
    """0번 어른(부모, 강도 show)이 모양 칸 mc로 끼니 o를 먹는다. 1번은 0번의 새끼, 2번은 남의 새끼(부모는 멀리 있는 3번),
    4번은 곁의 어른. 모두 0번에게서 반지름 8 안."""
    s = w.s
    s.show_g[0] = show
    _at_xy(w, 1, 23.0, 20.0)
    _child(w, 1, 0, 50)
    _at_xy(w, 2, 20.0, 25.0)
    _child(w, 2, 3, 50)
    _at_xy(w, 4, 24.0, 24.0)
    _wm_meal(w, 0, mc, o)


def test_kin_class_matches_soc_inline():
    w = _wm_world(6)
    s = w.s
    s.ids[:6] = (10, 11, 12, 13, 14, 15)
    s.parent[:6] = (-1, 10, 10, -1, 99, 99)
    kc = lambda i, j: int(k._kin_class(s, i, j))
    assert kc(1, 0) == 1                            # j(0)가 i(1)의 부모
    assert kc(0, 1) == 2                            # i가 j의 부모(자식)
    assert kc(1, 2) == 2 and kc(2, 1) == 2          # 형제(같은 부모)
    assert kc(4, 5) == 2                            # 부모가 죽었어도 같은 id면 형제
    assert kc(1, 3) == 0 and kc(3, 0) == 0          # 부모 없음끼리는 형제가 아니다
    assert kc(1, 4) == 0                            # 남
    # 부모와 형제가 겹치면 부모가 먼저
    s.parent[0] = 11
    assert kc(1, 0) == 1 and kc(0, 1) == 1


def test_boost_kin_parent_only():
    w = _wm_world()
    _wm_scene(w)
    s = w.s
    s.soc_g[1] = 0.1
    _wm(w)
    assert s.n_seen[1] == 1 and s.n_seen[2] == 1 and s.n_seen[4] == 1      # 양성 확인: 모두 보았다
    assert s.taste_m[1, 1] == pytest.approx(-(0.5 + 0.1 + 0.5 * 0.6))      # 자기 새끼: 바탕 + soc_g + boost
    assert s.taste_m[2, 1] == pytest.approx(-0.5)                          # 남의 새끼: 바탕만
    assert s.taste_m[4, 1] == pytest.approx(-0.5)                          # 어른 관찰자: 바탕만
    assert s.taste_m[3, 1] == 0.0                                          # 멀리 있어 못 봄
    # 합이 1에서 잘린다
    w = _wm_world()
    _wm_scene(w)
    w.s.soc_g[1] = 0.9
    _wm(w)
    assert w.s.taste_m[1, 1] == pytest.approx(-1.0)
    # 시연자가 미숙기이면 boost 없음
    w = _wm_world()
    _wm_scene(w)
    w.s.age[0] = 150
    w.s.soc_g[1] = 0.1
    _wm(w)
    assert w.s.taste_m[1, 1] == pytest.approx(-0.6)
    # 강도 0이면 boost 0
    w = _wm_world()
    _wm_scene(w, show=0.0)
    _wm(w)
    assert w.s.taste_m[1, 1] == pytest.approx(-0.5)
    # any: 곁의 미숙기 누구나 boost, 어른 관찰자는 그대로
    w = _wm_world(teach_mode="any")
    _wm_scene(w)
    w.s.soc_g[1] = 0.1
    _wm(w)
    assert w.s.taste_m[1, 1] == pytest.approx(-0.9)
    assert w.s.taste_m[2, 1] == pytest.approx(-0.8)
    assert w.s.taste_m[4, 1] == pytest.approx(-0.5)


def test_boost_zero_for_costonly_drift_and_nonknower():
    for mode in ("costonly", "drift"):
        w = _wm_world(teach_mode=mode)
        _wm_scene(w)
        _wm(w)
        assert w.s.n_seen[1] == 1
        assert w.s.taste_m[1, 1] == pytest.approx(-0.5), mode
        assert w.s.n_taught[1] == 0
    for knows, want in ((False, -0.5), (True, -0.8)):
        w = _wm_world(teach_mode="knower")
        kp = _kp(w)
        _wm_scene(w, mc=kp)
        w.s.taste_m[0, kp] = -0.5 if knows else 0.0
        _wm(w)
        assert w.s.taste_m[1, kp] == pytest.approx(want), knows
        assert w.s.taste_m[2, kp] == pytest.approx(-0.5)                   # 남의 새끼는 knower에서도 boost 없음
    # 양성 대조: kin은 시연자가 몰라도 boost
    w = _wm_world(teach_mode="kin")
    _wm_scene(w, mc=_kp(w))
    _wm(w)
    assert w.s.taste_m[1, _kp(w)] == pytest.approx(-0.8)
    # 한 번 안 시연자(know_age >= 0)도 그 스텝 기억이 0이면 knower는 boost 없음, n_taught_knower도 안 오른다
    w = _wm_world(teach_mode="knower")
    kp = _kp(w)
    _wm_scene(w, mc=kp)
    w.s.know_age[0] = 10
    w.s.taste_m[0, kp] = 0.0
    _wm(w)
    assert w.s.taste_m[1, kp] == pytest.approx(-0.5)
    assert w.s.n_taught[1] == 0 and w.s.n_taught_knower[1] == 0
    w = _wm_world(teach_mode="kin")
    kp = _kp(w)
    _wm_scene(w, mc=kp)
    w.s.know_age[0] = 10
    w.s.taste_m[0, kp] = 0.0
    _wm(w)
    assert w.s.n_taught[1] == 1 and w.s.n_taught_knower[1] == 0


def test_taught_counters():
    for mode in ("kin", "knower"):
        w = _wm_world(teach_mode=mode)
        kp = _kp(w)
        _wm_scene(w, mc=kp)
        s = w.s
        s.taste_m[0, kp] = -0.5                    # 시연자는 독 모양을 안다
        _wm(w)
        assert s.n_taught[1] == 1 and s.n_taught_knower[1] == 1, mode
        assert s.n_seen_par[1] == 1 and s.n_seen_par_poison[1] == 1
        assert s.n_taught[2] == 0 and s.n_seen_par[2] == 0                 # 남의 새끼
        assert s.n_taught[4] == 0 and s.n_seen_par[4] == 0                 # 어른 관찰자
    # kin에서 시연자가 모르면 boost는 붙고 knower 수는 안 오른다
    w = _wm_world()
    kp = _kp(w)
    _wm_scene(w, mc=kp)
    _wm(w)
    assert w.s.n_taught[1] == 1 and w.s.n_taught_knower[1] == 0 and w.s.n_seen_par[1] == 1
    # 좋은 끼니: 독 수는 안 오른다
    w = _wm_world()
    _wm_scene(w, mc=1 - kp, o=1.0)
    _wm(w)
    assert w.s.n_seen_par[1] == 1 and w.s.n_seen_par_poison[1] == 0 and w.s.n_taught[1] == 1
    # costonly: boost 없으니 n_taught 0, 부모 끼니는 센다
    w = _wm_world(teach_mode="costonly")
    _wm_scene(w, mc=kp)
    _wm(w)
    assert w.s.n_taught[1] == 0 and w.s.n_seen_par[1] == 1
    # 한 번 더 보면 누적
    _wm(w)
    assert w.s.n_seen_par[1] == 2


def _meal_drop(w, i=0):
    s = w.s
    e0, l0 = float(s.energy[i]), float(s.ledger[k.L_BRAIN])
    t0 = float(s.grad_tab[:, k.GT_TEACH_COST].sum())
    _wm(w)
    drop = e0 - float(s.energy[i])
    assert float(s.ledger[k.L_BRAIN]) - l0 == pytest.approx(drop, abs=1e-15)        # 에너지 감소 = L_BRAIN 증가
    assert float(s.grad_tab[:, k.GT_TEACH_COST].sum()) - t0 == pytest.approx(drop, abs=1e-15)
    return drop


def test_meal_cost_only_when_boost_applies():
    cost = 0.0005
    w = _wm_world(teach_cost_when="meal")
    _wm_scene(w, show=0.8)
    _at_xy(w, 5, 17.0, 22.0)
    _child(w, 5, 0, 60)                            # 새끼가 둘이어도 한 번
    s = w.s
    s.tag[0] = 99
    assert _meal_drop(w) == pytest.approx(cost * 0.8)
    assert s.grad_tab[7, k.GT_TEACH_COST] == pytest.approx(cost * 0.8)       # 태그 99는 7행
    assert s.n_taught[1] == 1 and s.n_taught[5] == 1                         # 양성 확인: 효과는 붙었다
    # near는 watch_meals가 안 건는다
    w = _wm_world(teach_cost_when="near")
    _wm_scene(w, show=0.8)
    assert _meal_drop(w) == 0.0 and w.s.n_taught[1] == 1
    # 새끼가 없다 / 반지름 밖 / 죽음 / 미숙기 아닌 곁
    w = _wm_world(teach_cost_when="meal")
    _wm_scene(w, show=0.8)
    w.s.alive[1] = False
    assert _meal_drop(w) == 0.0
    w = _wm_world(teach_cost_when="meal")
    _wm_scene(w, show=0.8)
    _at_xy(w, 1, 20.0 + 9.0, 20.0)
    assert _meal_drop(w) == 0.0
    w = _wm_world(teach_cost_when="meal")
    _wm_scene(w, show=0.8)
    w.s.age[1] = 250                               # 자란 자식은 미숙기가 아니다
    assert _meal_drop(w) == 0.0
    # 끼니가 없으면(meal_o == 0) 0
    w = _wm_world(teach_cost_when="meal")
    _wm_scene(w, show=0.8)
    w.s.meal_o[0] = 0.0
    assert _meal_drop(w) == 0.0
    # 모드별: kin·costonly·any 낸다, drift 안 낸다, knower는 아는 부모만
    for mode, knows, want in (("kin", False, 0.8), ("costonly", False, 0.8), ("any", False, 0.8), ("drift", False, 0.0),
                              ("knower", False, 0.0), ("knower", True, 0.8)):
        w = _wm_world(teach_mode=mode, teach_cost_when="meal")
        kp = _kp(w)
        _wm_scene(w, show=0.8, mc=kp)
        w.s.taste_m[0, kp] = -0.5 if knows else 0.0
        assert _meal_drop(w) == pytest.approx(cost * want), (mode, knows)
    # know_age >= 0이어도 그 스텝 기억이 0이면 knower의 meal 비용은 0
    w = _wm_world(teach_mode="knower", teach_cost_when="meal")
    kp = _kp(w)
    _wm_scene(w, show=0.8, mc=kp)
    w.s.know_age[0] = 10
    w.s.taste_m[0, kp] = 0.0
    assert _meal_drop(w) == 0.0
    # any: 부모 아닌 새끼만 곁에 있어도 낸다(kin은 0)
    for mode, want in (("kin", 0.0), ("any", 0.8)):
        w = _wm_world(teach_mode=mode, teach_cost_when="meal")
        _wm_scene(w, show=0.8)
        w.s.parent[1] = -1
        assert _meal_drop(w) == pytest.approx(cost * want), mode


def test_meal_cost_ledger_tab_two_demonstrators():
    cost = 0.0005
    w = _wm_world(teach_cost_when="meal")
    s = w.s
    # 시연자 0(태그 2, 강도 0.8)과 시연자 6(태그 5, 강도 0.4)이 각자 새끼 곁에서 먹는다
    s.show_g[0], s.show_g[6] = 0.8, 0.4
    s.tag[0], s.tag[6] = 2, 5
    _at_xy(w, 1, 23.0, 20.0)
    _child(w, 1, 0, 50)
    _at_xy(w, 6, 100.0, 60.0)
    _at_xy(w, 7, 103.0, 60.0)
    _child(w, 7, 6, 50)
    _wm_meal(w, 0, 1, -1.0)
    _wm_meal(w, 6, 0, 1.0)
    e0 = s.energy.copy()
    led0 = float(s.ledger[k.L_BRAIN])
    _wm(w)
    assert float(e0[0] - s.energy[0]) == pytest.approx(cost * 0.8)
    assert float(e0[6] - s.energy[6]) == pytest.approx(cost * 0.4)
    assert float(s.ledger[k.L_BRAIN]) - led0 == pytest.approx(cost * 1.2)
    assert s.grad_tab[2, k.GT_TEACH_COST] == pytest.approx(cost * 0.8)
    assert s.grad_tab[5, k.GT_TEACH_COST] == pytest.approx(cost * 0.4)


def _poison_step_pair():
    """독 모양이 바뀌는 스텝 경계(t0 - 1, t0)를 찾는다."""
    w = _lone_juv(juv_enabled=True, kind_flip_period=100)
    per = w.cfg.kind_flip_period
    for n in range(1, 60):
        if k.cfg_good_kind(w.cfg, n * per - 1) != k.cfg_good_kind(w.cfg, n * per):
            return n * per - 1, n * per
    raise AssertionError("독 모양이 안 바뀌는 씨앗")


def test_know_helper_sources():
    # 1: 자기 끼니
    w = _lone_juv(juv_enabled=True)
    s = w.s
    s.taste_m[0] = 0.0
    s.know_age[0] = -1
    s.know_src[0] = 0
    _eat_one(w, _good(w), age=51)                  # 좋은 알은 앎이 아니다
    assert s.know_age[0] == -1 and s.know_src[0] == 0
    _eat_one(w, 1 - _good(w), age=51)
    assert s.taste_m[0, _kp(w)] <= -0.2            # 양성 확인
    assert s.know_age[0] == 51 and s.know_src[0] == 1
    _eat_one(w, 1 - _good(w), age=61)              # 한 번만 기록
    assert s.know_age[0] == 51 and s.know_src[0] == 1
    # 어른은 기록하지 않는다
    w = _lone_juv(juv_enabled=True)
    w.s.taste_m[0] = 0.0
    _eat_one(w, 1 - _good(w), age=201)
    assert w.s.know_age[0] == -1
    # 장치가 꺼진 세계는 기록하지 않는다. juv_measure만 켠 세계는 기록한다
    w = _lone_juv()
    w.s.taste_m[0] = 0.0
    _eat_one(w, 1 - _good(w), age=51)
    assert w.s.know_age[0] == -1
    w = _lone_juv(juv_measure=True)
    w.s.taste_m[0] = 0.0
    _eat_one(w, 1 - _good(w), age=51)
    assert w.s.know_age[0] == 51 and w.s.know_src[0] == 1
    # 독 모양은 그 스텝 good_kind로: 경계 전(t0 - 1)·후(t0) 각각
    t_a, t_b = _poison_step_pair()
    w = _lone_juv(juv_enabled=True, kind_flip_period=100)
    w.s.taste_m[0] = 0.0
    w.step_no = t_a
    _eat_one(w, 1 - _good_at(w), age=51)
    assert w.s.know_age[0] == 51
    old_poison = 1 - k.cfg_good_kind(w.cfg, t_a)
    w.s.know_age[0] = -1
    w.step_no = t_b
    assert 1 - k.cfg_good_kind(w.cfg, t_b) != old_poison                   # 독 모양이 바뀌었다
    _eat_one(w, old_poison, age=71)                # 이제는 좋은 알: 예전 독 칸의 기억이 깊어도 새 독 칸이 아니라 앎이 아니다
    assert w.s.know_age[0] == -1
    _eat_one(w, 1 - old_poison, age=72)            # 새 독을 먹는다
    assert w.s.know_age[0] == 72 and w.s.know_src[0] == 1


def _good_at(w) -> int:
    return k.cfg_good_kind(w.cfg, w.step_no)


def _watch_know(setup):
    w = _wm_world()
    kp = _kp(w)
    s = w.s
    _at_xy(w, 1, 23.0, 20.0)
    _child(w, 1, None, 50)
    s.parent[1] = -1
    setup(w)
    _wm_meal(w, 0, kp, -1.0)
    _wm(w)
    return w


def test_know_helper_watch_sources():
    def parent(w):
        w.s.parent[1] = w.s.ids[0]
        w.s.age[0] = 250

    def sibling(w):
        w.s.parent[1] = 777
        w.s.parent[0] = 777
        w.s.age[0] = 250

    def child_of_observer(w):
        w.s.parent[0] = w.s.ids[1]

    def stranger(w):
        w.s.parent[0] = -1

    for setup, src in ((parent, 2), (sibling, 3), (child_of_observer, 3), (stranger, 4)):
        w = _watch_know(setup)
        assert w.s.taste_m[1, _kp(w)] <= -0.2, setup.__name__                # 양성 확인: 기억이 움직였다
        assert w.s.know_age[1] == 50 and w.s.know_src[1] == src, setup.__name__
    # 한 번만 기록: 이미 앎이면 안 덮는다
    w = _wm_world()
    _at_xy(w, 1, 23.0, 20.0)
    _child(w, 1, 0, 50)
    w.s.know_age[1] = 30
    w.s.know_src[1] = 1
    _wm_meal(w, 0, _kp(w), -1.0)
    _wm(w)
    assert w.s.know_age[1] == 30 and w.s.know_src[1] == 1
    # 어른 관찰자는 기록하지 않는다
    w = _wm_world()
    _at_xy(w, 1, 23.0, 20.0)
    _child(w, 1, 0, 250)
    _wm_meal(w, 0, _kp(w), -1.0)
    _wm(w)
    assert w.s.taste_m[1, _kp(w)] <= -0.2 and w.s.know_age[1] == -1
    # 독 모양이 아닌 칸이 움직인 것은 앎이 아니다
    w = _wm_world()
    _at_xy(w, 1, 23.0, 20.0)
    _child(w, 1, 0, 50)
    _wm_meal(w, 0, 1 - _kp(w), -1.0)
    _wm(w)
    assert w.s.know_age[1] == -1


def test_watch_meals_teach_off_unchanged():
    """teach_enabled=false: watch_meals의 갱신이 사회 학습 단독 세계와 같다. 앎·부모 끼니 계수기는 읽기 전용으로 쌓인다."""
    off = _wm_world(teach_enabled=False)
    _wm_scene(off)
    off.s.soc_g[1] = 0.1
    _wm(off)
    base = World(small("stage2", "kinds", "taste", soc_taste_enabled=True, init_agents=8, min_agents=0,
                       food_inflow=0.0, soc_rate=0.5))
    for i in range(8):
        for name in ("x", "y", "energy", "age", "ids", "parent", "tag", "soc_g", "show_g"):
            getattr(base.s, name)[i] = getattr(off.s, name)[i]
    base.s.taste_m[:] = 0.0
    base.s.meal_o[:] = off.s.meal_o
    base.s.meal_k[:] = off.s.meal_k
    _wm(base)
    assert np.array_equal(off.s.taste_m, base.s.taste_m)
    assert off.s.taste_m[1, 1] == pytest.approx(-0.6)
    assert off.s.n_taught.sum() == 0 and off.s.n_taught_knower.sum() == 0
    assert off.s.n_seen_par[1] == 1                # 읽기 전용 계수기
    assert float(off.s.grad_tab[:, k.GT_TEACH_COST].sum()) == 0.0


def _crowd_scene(w):
    s = w.s
    s.show_g[0], s.show_g[6] = 0.7, 0.3
    s.age[6] = 250
    _at_xy(w, 1, 23.0, 20.0)
    _child(w, 1, 0, 50)
    _at_xy(w, 6, 20.0, 26.0)
    _at_xy(w, 2, 24.0, 25.0)
    _child(w, 2, 6, 40)
    _at_xy(w, 4, 18.0, 17.0)
    _child(w, 4, 0, 30)
    s.parent[6] = s.ids[0]
    _wm_meal(w, 0, 1, -1.0)
    _wm_meal(w, 6, 1, 1.0)
    _wm_meal(w, 4, 1, -1.0)


def test_boost_order_deterministic():
    hashes = []
    for _ in range(2):
        w = _wm_world(teach_cost_when="meal")
        _crowd_scene(w)
        _wm(w)
        hashes.append((w.state_hash(), w.s.taste_m.copy(), float(w.s.energy[0]), float(w.s.energy[6])))
    assert hashes[0][0] == hashes[1][0]
    assert np.array_equal(hashes[0][1], hashes[1][1]) and hashes[0][2:] == hashes[1][2:]
    # 값 확인: 1번 관찰자는 칸 번호 오름차순(0 < 4 < 6)으로 세 시연자를 본다. 지수평활은 비가환이라 순서가 값을 가른다
    w = _wm_world(teach_cost_when="meal")
    _crowd_scene(w)
    _wm(w)
    assert w.s.n_seen[1] == 3
    m = 0.0
    m += (0.5 + 0.5 * 0.7) * (-1.0 - m)            # 0
    m += 0.5 * (-1.0 - m)                          # 4(1과 같은 부모의 미숙기, 독)
    m += 0.5 * (1.0 - m)                           # 6(좋은 알)
    assert w.s.taste_m[1, 1] == pytest.approx(m)
