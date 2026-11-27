"""규범의 씨앗 판정 도구 시험(norm_gate 순수 계산 + norm_judge 명령). 시뮬레이션 없이 도는 것을 최대로.

계획 docs/superpowers/plans/2026-11-26-norms-seed.md 과제 17.

    python3 -m pytest -q tests/test_norm_judge.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "wayfinder" / "stage2-4"))

import norm_gate as ng  # noqa: E402
import norm_judge as nj  # noqa: E402
from test_stages import small  # noqa: E402
from kkumteul import checkpoint as ckpt  # noqa: E402
from kkumteul import kernels as k  # noqa: E402
from kkumteul import measure  # noqa: E402
from kkumteul.world import World  # noqa: E402


# ---------------- 합성 자료 ----------------
def row(**kw):
    r = {name: 0.0 for name in k.NT_NAMES}
    r.update({key: float(v) for key, v in kw.items()})
    return r


def run(dr=0.01, net=None, alive=None, epp=None, prev=0.05, energy=None, steps=10000.0, dr_all=None):
    """합성 한 판: 태그별 계수표·순수입·산 개체 수. net = {태그: 개체 스텝당 순수입}.
    dr_all = 가로채기 피해율(TAKE_DMG_RECV ÷ 개체 스텝, 없으면 dr과 같게)."""
    net = net or {0: 0.006}
    alive = alive or {t: 100 for t in net}
    nbt, nibt = {}, {}
    for t, v in net.items():
        ep = 100.0
        nbt[t] = row(viol_dmg_recv=dr * steps, take_dmg_recv=(dr if dr_all is None else dr_all) * steps,
                     alive_steps=steps, mark_epis0=ep,
                     mark_epis_pun0=(epp if epp is not None else 0.3) * ep)
        nibt[t] = {"net_per_step": v, "eaten_per_life_done": 9.0 + t, "life_mean": 1500.0 + t}
    return {"norm_by_tag": nbt, "net_income_by_tag": nibt, "alive_by_tag": alive, "mark_prevalence": prev,
            "energy_mean_by_tag": energy or {t: 1.0 for t in net}}


def cell(drs, **kw):
    return ng.cell_summary({"runs": [run(dr=d, **kw) for d in drs]})


JIT = [0.0, 0.0002, -0.0002, 0.0001, -0.0001, 0.00005]


def cells_working():
    base = [0.010 + j for j in JIT]
    return {
        "base": cell(base),
        "hand-phi0.1": cell([0.0097 + j for j in JIT]),
        "hand-phi0.3": cell([0.0088 + j for j in JIT]),
        "hand-phi1.0": cell([0.0070 + j for j in JIT]),
        "hand-phi0.3-inert": cell([0.0100 + j for j in JIT]),
        "off": cell([0.0100 - j for j in JIT]),
    }


# ======================= 칸 정의 =======================
def test_cells_table_complete():
    want = {"hand-phi0.1", "hand-phi0.3", "hand-phi1.0", "hand3-phi0.3", "hand-phi0.3-inert", "hand-phi0.3-nearest",
            "hand-phi0.3-ret0", "hand-phi0.3-ret0.5", "hand-phi0.3-inh0", "hand-phi0.3-inh0.4", "hand-phi0.3-L1",
            "hand-phi0.3-L10", "hand-phi0.3-tau1", "hand-phi0.3-sig-ret0", "hand-phi0.3-sig-ret0.1",
            "hand-phi0.3-none-q-ret0", "hand-phi0.3-none-q-ret0.1", "hand-phi0.3-noise0.0005",
            "hand-phi0.3-noise0.002", "off", "noimp", "base", "grp-1", "grp-2", "grp-3"}
    assert set(nj.CELLS) == want
    assert len(nj.CORE) == 14 and len(nj.DELTA) == 4 and len(nj.REST) == 7
    assert set(nj.CORE) | set(nj.DELTA) | set(nj.REST) == want
    assert not (set(nj.CORE) & set(nj.DELTA)) and not (set(nj.CORE) & set(nj.REST))
    C = nj.CELLS
    assert "norm_enabled=true" in C["hand-phi0.3"]["sets"]
    assert C["off"]["sets"] == ["norm_enabled=false", "norm_measure=true"]
    assert "norm_impulse=0" in C["noimp"]["sets"] and "norm_enabled=true" in C["noimp"]["sets"]
    assert C["base"]["phi"] == 0.0 and C["base"]["arena"] == "base"
    assert [C[f"grp-{t}"]["arena"] for t in (1, 2, 3)] == ["all1", "all2", "all3"]
    for name in nj.DELTA:
        assert C[name]["calib"] in ("sig", "noise")
        assert not any(s.startswith(("norm_sig_theta", "norm_mark_noise")) for s in C[name]["sets"])
    with pytest.raises(ValueError):
        nj.cell_sets("hand-phi0.3-sig-ret0", None)
    assert "norm_sig_theta=0.37" in nj.cell_sets("hand-phi0.3-sig-ret0", {"sig_theta": 0.37, "noise_q": 0.001})
    assert "norm_mark_noise=0.001" in nj.cell_sets("hand-phi0.3-none-q-ret0.1", {"sig_theta": 0.37, "noise_q": 0.001})


def test_arena_composition_and_settings():
    cfg = small("stage4", "conflict", "norm_size", max_agents=900, conflict_enabled=True, norm_impulse=0.015)
    g = measure.extract_genomes(World(cfg))

    def comp(name, arena="std", calib=None):
        groups, acfg, _ = nj.arena_groups(name, g, cfg, 800, arena, calib)
        return {t: c for _, t, c in groups}, acfg
    c, a = comp("hand-phi0.1")
    assert c == {0: 640, 1: 80, 2: 80} and a.norm_block_tag == 2 and a.norm_block3_tag == -1
    c, a = comp("hand-phi0.3")
    assert c == {0: 320, 1: 240, 2: 240} and a.norm_enabled
    c, a = comp("hand3-phi0.3")
    assert c == {0: 320, 1: 240, 3: 240} and a.norm_block3_tag == 3 and a.norm_block_tag == -1
    for name, tag in (("grp-1", 1), ("grp-2", 2), ("grp-3", 3), ("hand-phi1.0", 1)):
        c, _ = comp(name)
        assert c == {tag: 800}
    c, a = comp("base")
    assert c == {0: 800} and a.norm_enabled
    c, a = comp("off")
    assert not a.norm_enabled and a.norm_measure and a.norm_block_tag == -1 and a.norm_impulse == 0.015
    c, a = comp("hand-phi0.3", "mix")
    assert a.norm_mix_period == 200 and a.norm_mix_frac == 0.2
    c, a = comp("hand-phi0.3", "std")
    assert a.norm_mix_period == 0
    assert measure.arena_config(a, 1, 800).mutation_enabled is False


# ======================= 위반 피해율·순수입 =======================
def test_damage_rate_definition_mode_independent():
    full = row(viol_dmg_recv=6.0, alive_steps=1000.0, take_dmg_recv=6.0)
    inert = row(viol_dmg_recv=6.0, alive_steps=1000.0, take_dmg_recv=9.0, take_on_marked=15.0)
    off = row(viol_dmg_recv=6.0, alive_steps=1000.0, take_dmg_recv=6.0)
    assert ng.damage_rate(full) == ng.damage_rate(inert) == ng.damage_rate(off) == pytest.approx(0.006)
    tagged = {0: row(viol_dmg_recv=2.0, alive_steps=500.0), 1: row(viol_dmg_recv=4.0, alive_steps=500.0)}
    assert ng.damage_rate(tagged) == pytest.approx(0.006)
    assert ng.damage_rate(row()) is None
    # off·noimp 칸의 결과에도 같은 열이 있다(hand-dose가 off 칸을 norm_measure로 돌린다)
    s = ng.cell_summary({"runs": [run(dr=0.004), run(dr=0.006)]})
    assert s["dr"] == [pytest.approx(0.004), pytest.approx(0.006)]


def test_alpha_off_control_uses_take_damage_rate():
    """off(표시 없음)의 위반 피해율은 표시된 대상을 친 충동 피해까지 세어 base보다 높게 나온다(정의가 다름).
    α의 off 대조는 두 칸에서 정의가 같은 가로채기 피해율(TAKE_DMG_RECV ÷ 개체 스텝)로 견준다."""
    off = row(viol_dmg_recv=10.0, take_dmg_recv=10.0, alive_steps=1000.0)          # 표시가 없어 모든 피해가 위반
    base = row(viol_dmg_recv=9.2, take_dmg_recv=10.0, take_on_marked=4.0, alive_steps=1000.0)
    assert ng.damage_rate(off) > ng.damage_rate(base)
    assert ng.damage_rate_all(off) == ng.damage_rate_all(base) == pytest.approx(0.01)
    c = cells_working()
    # off의 위반 피해율만 8% 높고(표시된 대상 몫) 가로채기 피해율은 base와 같다 → 대조 통과
    c["off"] = ng.cell_summary({"runs": [run(dr=0.0108 - j, dr_all=0.0100 - j) for j in JIT]})
    a = ng.gate_alpha(c)
    assert a["verdict"] == "작동" and a["controls"]["off"]["rate"] == "dr_all"
    assert ng.welch(c["off"]["dr"], c["base"]["dr"])["sign"] == 1      # 옛 정의였다면 도구 깨짐
    # 가로채기 피해율이 정말 다르면 깨짐
    c["off"] = ng.cell_summary({"runs": [run(dr=0.0100 - j, dr_all=0.0110 - j) for j in JIT]})
    assert ng.gate_alpha(c)["verdict"] == "도구 깨짐"
    assert "오조준" in ng.LABEL_MISAIM


def test_net_income_levels_hand_values():
    nets = [0.0010, 0.0012, 0.0008, 0.0011, 0.0009, 0.0010]
    runs = [run(net={0: 0.006, 1: 0.006 + d, 2: 0.006}) for d in nets]
    out = ng.n_ind(runs, 1, 2)
    v = np.array(nets)
    half = measure.t95(5) * v.std(ddof=1) / math.sqrt(6)
    assert out["mean"] == pytest.approx(v.mean())
    assert out["lo"] == pytest.approx(v.mean() - half) and out["hi"] == pytest.approx(v.mean() + half)
    assert out["life"]["mean"] == pytest.approx(v.mean() * 1560)
    runs3 = [run(net={1: 0.006, 3: 0.006 + d}) for d in nets]
    assert ng.n_ind(runs3, 1, 3)["mean"] == pytest.approx(-v.mean())
    # N_grp·N_grp3: 같은 씨앗 짝(전원 1 마당 대 전원 3 마당)
    xs = [run(net={1: 0.0070}, alive={1: 300}, energy={1: 1.2}) for _ in range(6)]
    ys = [run(net={3: 0.0060 + d / 10}, alive={3: 200}, energy={3: 1.0}) for d in nets]
    g3 = ng.n_grp(xs, ys, 1, 3)
    assert g3["mean"] == pytest.approx(0.0070 - (0.0060 + v.mean() / 10))
    assert g3["log_ratio"]["mean"] == pytest.approx(math.log(300.5 / 200.5))
    assert g3["energy_diff"]["mean"] == pytest.approx(0.2)
    # 2차 딜레마: N_grp3 > 0 ≥ N_ind3 → 크기 = N_grp3
    dil = ng.second_order_dilemma({"lo": -0.0001, "mean": 0.0}, {"lo": 0.0005, "mean": 0.0009})
    assert dil == {"open": True, "size": 0.0009}
    assert ng.second_order_dilemma({"lo": 0.0001, "mean": 0.0002}, {"lo": 0.0005, "mean": 0.0009})["open"] is False


def test_net_income_reports_lifetime_companions():
    runs = [run(net={1: 0.007, 2: 0.006}) for _ in range(3)]
    out = ng.n_ind(runs, 1, 2)
    assert out["companions"]["1"]["eaten_per_life_done"] == pytest.approx(10.0)
    assert out["companions"]["2"]["life_mean"] == pytest.approx(1502.0)
    assert "보수 편향" in out["bias_note"] and "집행자에게 불리한" in out["bias_note"]
    # json을 거친 문자열 키도 읽는다
    js = json.loads(json.dumps(runs))
    assert ng.n_ind(js, 1, 2)["mean"] == pytest.approx(0.001)


# ======================= 관문 α·β·γ·δ =======================
def test_gate_alpha_beta_gamma_delta():
    c = cells_working()
    a = ng.gate_alpha(c)
    assert a["verdict"] == "작동" and a["monotone"] and a["drop"] >= 0.07
    # 음성 대조 깨짐: inert가 base보다 뚜렷이 낮다
    c2 = dict(c, **{"hand-phi0.3-inert": cell([0.0080 + j for j in JIT])})
    assert ng.gate_alpha(c2)["verdict"] == "도구 깨짐"
    # 감소가 7% 아래면 작동 아님
    c3 = dict(c, **{"hand-phi0.3": cell([0.0096 + j for j in JIT])})
    assert ng.gate_alpha(c3)["verdict"] == "작동 아님"
    # 단조 깨짐
    c4 = dict(c, **{"hand-phi1.0": cell([0.0095 + j for j in JIT])})
    assert not ng.gate_alpha(c4)["monotone"]
    assert ng.gate_alpha({"base": c["base"]})["verdict"] == "미판정"
    # β
    pos, zero = {"lo": 0.0001, "mean": 0.0003}, {"lo": -0.0002, "mean": 0.0}
    b = ng.gate_beta(pos, pos, pos)
    assert b["all"] and b["verdict"] == "3자 집행 벌주기 씨앗이 집행자 본인에게 이득이 된다"
    b = ng.gate_beta(pos, zero, pos)
    assert b["ga"] and not b["na"] and b["dilemma"]["open"] and "2차 딜레마" in b["verdict"]
    assert ng.gate_beta(pos, pos, pos, {"detectable": False})["verdict"] == "검출 한계 아래"
    # γ
    g = ng.gate_gamma(c)
    assert g["verdict"] == "만족"
    cg = dict(c, **{"hand-phi0.1": cell([0.0100 + j for j in JIT])})
    assert "만족하지 않는다" in ng.gate_gamma(cg)["verdict"]
    cl = dict(c, **{"hand-phi0.3-L1": cell([0.0100 - j for j in JIT])})
    assert "상습 위반자" in ng.gate_gamma(cl)["L1"]
    cl = dict(c, **{"hand-phi0.3-L1": cell([0.0088 - j for j in JIT])})
    assert "억지가 아닌 곳" in ng.gate_gamma(cl)["L1"]
    # δ (관문 −1의 50% 규칙을 world에서 읽는다)
    wok = {"-1c": {"delta_readable": True, "src_match": True}}
    wbad = {"-1c": {"delta_readable": False, "src_match": False}}
    assert ng.gate_delta(c, wok)["verdict"] == "δ 미판정"
    take = cell(JIT, epp=0.40, net={1: 0.0070, 2: 0.006})
    take["epp"] = [0.40 + j * 10 for j in JIT]
    sig = cell(JIT, epp=0.20, net={1: 0.0062, 2: 0.006})
    sig["epp"] = [0.20 + j * 10 for j in JIT]
    nonq = cell(JIT, epp=0.20, net={1: 0.0062, 2: 0.006})
    nonq["epp"] = [0.21 + j * 10 for j in JIT]
    cd = dict(c, **{"hand-phi0.3-ret0": take, "hand-phi0.3-sig-ret0": sig, "hand-phi0.3-none-q-ret0": nonq})
    assert ng.gate_delta(cd, wok)["verdict"] == "기능에서 값을 얻는다"
    assert ng.gate_delta(cd)["verdict"] == "δ 미판정"                         # 세계 점검 결과 없음
    assert ng.gate_delta(cd, wbad)["verdict"] == "유병률 맞춤 밖이라 δ를 읽지 않음"   # 세 원천 50% 밖
    same = cell(JIT, epp=0.40, net={1: 0.0070, 2: 0.006})
    same["epp"] = [0.40 - j * 10 for j in JIT]
    cs = dict(cd, **{"hand-phi0.3-sig-ret0": same, "hand-phi0.3-none-q-ret0": same})
    assert ng.gate_delta(cs, wok)["verdict"] == "표시 자체가 약탈 대상"
    far = cell(JIT, prev=0.2)
    assert ng.gate_delta(dict(cd, **{"hand-phi0.3-sig-ret0": far}), wok)["verdict"] == "유병률이 달라 가를 수 없음"
    # 합친 네 경우: 작동 + 이득 / 작동 + 가를 수 없음 / 작동 아님 / 도구 깨짐
    neutral = [0.00001, -0.00001, 0.00002, -0.00002, 0.0, 0.00001]
    full = dict(cd)
    full["base"] = dict(full["base"], neutral=neutral)
    gain = [run(dr=0.0088 + j, net={0: 0.006, 1: 0.0062 + j, 2: 0.006}) for j in JIT]
    full["hand-phi0.3"] = ng.cell_summary({"runs": gain})
    # 나머지 필요한 칸(CORE·DELTA): N_ind3·N_grp3가 읽는 태그 자료까지
    flat_net = {0: 0.006, 1: 0.006, 3: 0.006}
    full["hand3-phi0.3"] = ng.cell_summary({"runs": [run(dr=0.0088 + j, net={0: 0.006, 1: 0.006 + j, 3: 0.006})
                                                     for j in JIT]})
    for name, tg in (("grp-1", 1), ("grp-2", 2), ("grp-3", 3)):
        full[name] = ng.cell_summary({"runs": [run(dr=0.008 + j, net={tg: 0.006 + j}) for j in JIT]})
    for name in ng.HAND_REQUIRED:
        full.setdefault(name, cell([0.0090 + j for j in JIT], net=flat_net))
    gh = ng.gate_hand(full, world=wok)
    assert gh["case"] == "작동 + 이득" and gh["code"] == 0 and gh["missing"] == []
    flat = dict(full)
    flat["hand-phi0.3"] = ng.cell_summary({"runs": [run(dr=0.0088 + j, net={0: 0.006, 1: 0.006 + j, 2: 0.006})
                                                    for j in JIT]})
    assert ng.gate_hand(flat, world=wok)["code"] == 3
    assert ng.gate_hand(flat, world=wok)["case"] == "작동 + 이득을 가를 수 없음"
    assert ng.gate_hand(dict(full, **{"hand-phi0.3": c3["hand-phi0.3"]}), world=wok)["case"] == "작동 아님"
    assert ng.gate_hand(dict(full, **{"hand-phi0.3-inert": c2["hand-phi0.3-inert"]}), world=wok)["code"] == 4
    nodelta = {key: v for key, v in full.items() if key != "hand-phi0.3-sig-ret0"}
    assert ng.gate_hand(nodelta, world=wok)["code"] == 5      # δ 칸이 없으면 0으로 두지 않는다
    assert ng.gate_hand(full)["code"] == 5                    # 관문 −1 결과(δ의 50% 규칙)가 없으면 미판정
    # α 칸 밖의 필요한 칸(grp-*·hand3·noimp·ret0.5 …)이 하나라도 빠지면 미판정 — 이득이 있어 보여도 0이 아니다
    for gone in ("grp-3", "hand3-phi0.3", "noimp", "hand-phi0.3-ret0.5", "hand-phi0.3-none-q-ret0.1"):
        g5 = ng.gate_hand({key: v for key, v in full.items() if key != gone}, world=wok)
        assert g5["code"] == 5 and gone in g5["missing"], gone
    # 칸은 있는데 N_ind3가 읽는 태그(1·3) 자료가 없으면 미판정
    g5 = ng.gate_hand(dict(full, **{"hand3-phi0.3": cell(JIT, net={0: 0.006})}), world=wok)
    assert g5["code"] == 5 and "N_ind3" in g5["missing"]
    # base의 중립 반복(관문 −2)이 없으면 미판정
    nb = dict(full, base=cells_working()["base"])
    assert ng.gate_hand(nb, world=wok)["code"] == 5


def test_detectability_minus2_and_threshold():
    ctrl = [0.0001, -0.0001, 0.00005, -0.00005, 0.0, 0.0]
    lim = np.std(ctrl, ddof=1) * measure.t95(5) / math.sqrt(6)
    d = ng.detectability(0.0002, ctrl, pop=500)
    assert d["limit"] == pytest.approx(lim) and d["ratio"] == pytest.approx(0.0002 / lim) and d["detectable"]
    assert d["selection"] == pytest.approx(500 * 0.0002 * 1560 / 9.8) and d["selection_ok"]
    assert not ng.detectability(0.00001, ctrl)["detectable"]
    t = ng.threshold_above([-1.0, -0.5, 0.2, -3.0], w_atk=np.zeros((4, 8)), w_turn=np.ones((4, 8)))
    assert t["frac_above"] == pytest.approx(0.25) and t["slope_start"] == 0
    assert t["w_vis_turn"]["mean"] == 1.0
    t = ng.threshold_above([-1.0] * 200 + [0.1])
    assert t["frac_above"] < 0.01 and t["slope_start"] == 1


def _wobs(prev=0.05, tl=0.1, pop=500, nz=0.3, pr=0.3, **kw):
    o = {"pop_mean": pop, "pop_min": pop - 50, "reseeds": 0, "ledger_rel": 1e-12, "theta_t": 0.3,
         "take_loss_frac": tl, "prevalence": prev, "vis_nonzero_frac": nz, "vis_present_frac": pr}
    o.update(kw)
    return o


def test_gate_world_minus1():
    def obs(**over):
        base = {n: [_wobs(), _wobs()] for n in nj.W_NAMES}
        base.update(over)
        return base
    ac = {"take": 0, "pun": 0}
    g = ng.gate_world(obs(), ac)
    assert g["code"] == 0 and all(g[key]["ok"] for key in ("-1a", "-1b", "-1c", "-1d", "-1e"))
    assert ng.gate_world(obs(full=[_wobs(pop=400)] * 2), ac)["code"] == 3                  # −1a 무리
    assert ng.gate_world(obs(full=[_wobs(pop_min=0)] * 2), ac)["code"] == 3                # −1a 멸종
    assert ng.gate_world(obs(full=[_wobs(reseeds=50)] * 2), ac)["code"] == 3              # −1a 재주입 급증
    assert ng.gate_world(obs(full=[_wobs(ledger_rel=1e-3)] * 2), ac)["code"] == 3        # −1a 장부
    g = ng.gate_world(obs(full=[_wobs(tl=0.03)] * 2), ac)
    assert g["code"] == 4 and "올린다" in g["-1b"]["advice"]                                # −1b
    assert "내린다" in ng.gate_world(obs(full=[_wobs(tl=0.3)] * 2), ac)["-1b"]["advice"]
    g = ng.gate_world(obs(full=[_wobs(prev=0.13)] * 2), ac)
    assert g["code"] == 3 and "25" in g["-1c"]["advice"]                                    # −1c 상한
    g = ng.gate_world(obs(sig=[_wobs(prev=0.09)] * 2), ac)
    assert g["-1c"]["ok"] and not g["-1c"]["src_match"] and not g["-1c"]["delta_readable"]  # 원천 50% 밖
    g = ng.gate_world(obs(sig=[_wobs(prev=0.07)] * 2), ac)
    assert g["-1c"]["delta_readable"]                                                       # 1.4배는 50% 안
    assert ng.gate_world(obs(full=[_wobs(nz=0.1, pr=0.3)] * 2), ac)["code"] == 3          # −1d
    assert ng.gate_world(obs(), {"take": 3, "pun": 0})["code"] == 3                         # −1e
    assert ng.gate_world(obs(), None)["code"] == 5                                          # arena-check 없음
    # 한 씨앗뿐인 세계나 빠진 원천은 통과시키지 않는다(need = 씨앗 수)
    assert ng.gate_world(obs(), ac, need=2)["code"] == 0
    g = ng.gate_world(obs(full=[_wobs()]), ac, need=2)
    assert g["code"] == 5 and g["missing"] == ["full"]
    g = ng.gate_world({key: v for key, v in obs().items() if key != "none"}, ac, need=2)
    assert g["code"] == 5 and "none" in g["missing"]


def test_bottom_judgment():
    std = row(pun_2nd=20, opp_self=100, pun_3_kin=30, pun_3_str=10, opp_other=400, opp_other_sim2=60,
              opp_other_sim3=40, pun_3_line=15, opp_other_line=50)
    r = ng.bottom_rates(std)
    assert r["p2"] == pytest.approx(0.2) and r["p3"] == pytest.approx(0.1)
    assert r["kin_infl_sim"] == pytest.approx((30 / 40) / (100 / 400))
    assert r["kin_infl_line"] == pytest.approx((15 / 40) / (50 / 400))
    assert r["share3"] == pytest.approx(40 / 60)
    mix = row(pun_2nd=18, opp_self=100, pun_3_kin=8, pun_3_str=8, opp_other=400, opp_other_sim2=60,
              opp_other_sim3=40, pun_3_line=2, opp_other_line=50)
    runs_s = [{"norm_by_tag": {1: std}, "late_frac": 0.5} for _ in range(3)]
    runs_m = [{"norm_by_tag": {1: mix}} for _ in range(3)]
    b = ng.bottom(runs_s, runs_m, tau1_runs=[{"late_frac": 0.0}] * 3)
    assert b["p3"]["std_minus_mix"]["mean"] == pytest.approx(0.1 - 16 / 400)
    assert b["kin_infl_usable"] is True and b["late_frac"]["mean"] == 0.5 and b["late_frac_tau1"]["mean"] == 0.0
    assert "단서" in b["note"]
    # 닮음과 계보가 반대쪽이면 친족 부풀림을 안 쓴다
    odd = dict(std, pun_3_line=1.0)
    assert ng.bottom([{"norm_by_tag": {1: odd}}] * 2, runs_m[:2])["kin_infl_usable"] is False


def _bsum(n_ind=(0.00015, 0.0002, 0.00025), n_ind3=(0.00015, 0.0002, 0.00025), sel=(1.1, 1.3, 1.5), grp3=None):
    def lvl(t):
        return {"lo": t[0], "mean": t[1], "hi": t[2]}
    return {"n_ind": lvl(n_ind), "n_ind3": lvl(n_ind3), "selectivity": lvl(sel),
            "n_grp3": lvl(grp3) if grp3 else None, "r_CA": {"mean": 0.1}, "min_detectable": 0.2}


def test_main_judgment_sentence_and_codes():
    null = _bsum((-0.0001, 0.0, 0.0001), (-0.0001, 0.0, 0.0001), (0.8, 1.0, 1.2))
    noimp = _bsum((-0.0001, 0.0, 0.0001))
    # knockout r = ln(원래 ÷ 막은 쪽)의 기울기: 양이면 막기가 해롭다 → 켠 무리가 inert보다 커야 "가를 수 있는 방향"
    knock = {"on": 2.0, "inert": 0.0, "limit": 0.5}
    m = ng.main_judgment(_bsum(), {"inert": null, "off": null}, noimp, knock, _bsum())
    assert m["code"] == 0 and m["verdict"] == "그 수준(3자 집행)의 벌주기 씨앗이 개인에게 이득"
    assert "크거나 같다" in m["note_win"]
    m = ng.main_judgment(_bsum(n_ind3=(-0.0001, 0.0, 0.0001)), {"inert": null, "off": null}, noimp, knock,
                         _bsum(n_ind3=(-0.0001, 0.0, 0.0001)))
    assert m["code"] == 0 and "2자 맞대응" in m["verdict"]
    # 선택성 아래끝이 1 아래 → 가를 수 없음
    assert ng.main_judgment(_bsum(sel=(0.9, 1.2, 1.5)), {"inert": null}, noimp, knock, _bsum())["code"] == 3
    # 씨앗이 갈림
    other = _bsum((-0.0001, 0.0, 0.0001), (-0.0001, 0.0, 0.0001))
    assert ng.main_judgment(_bsum(), {"inert": null}, noimp, knock, other)["code"] == 3
    # 음성 대조 깨짐
    bad = _bsum((0.0001, 0.0002, 0.0003))
    assert ng.main_judgment(_bsum(), {"inert": null}, bad, knock, _bsum())["code"] == 4
    # 2차 딜레마 문장은 따로(OR 아님)
    dil = _bsum(n_ind=(-0.0001, 0.0, 0.0001), n_ind3=(-0.0001, 0.0, 0.0001), grp3=(0.0001, 0.0003, 0.0005))
    m = ng.main_judgment(dil, {"inert": null}, noimp, knock, dil)
    assert m["code"] == 3 and "2차 딜레마" in m["dilemma"]
    assert ng.main_judgment({}, {}, None)["code"] == 5
    # 감각 막기 방향: 켠 무리의 r가 inert보다 작으면(막기가 덜 해로우면) 크기가 커도 통과가 아니다
    rev = {"on": -2.0, "inert": 0.0, "limit": 0.5}
    m = ng.main_judgment(_bsum(), {"inert": null}, noimp, rev, _bsum())
    assert m["code"] == 3 and m["conditions"]["knock_vs_inert"] is False
    # null과 구간이 겹치면(null이 0 위가 아니어도) 구분 안 됨
    wide = _bsum((-0.0001, 0.00012, 0.0003), (-0.0001, 0.00012, 0.0003))
    m = ng.main_judgment(_bsum(), {"inert": wide}, noimp, knock, _bsum())
    assert m["code"] == 3 and m["conditions"]["null_distinct"] is False
    # sd·n이 있으면 Welch로
    lv_on = {"lo": 0.0001, "mean": 0.0002, "hi": 0.0003, "sd": 0.0001, "n": 6}
    lv_null = {"lo": -0.0001, "mean": 0.0, "hi": 0.0001, "sd": 0.0001, "n": 6}
    assert ng._distinct_above(lv_on, lv_null) is True       # 구간은 겹치지만(0.0001 = 0.0001) Welch로는 가른다
    assert ng._distinct_above(lv_null, lv_on) is False
    # 원자료가 없으면 3이 아니라 미판정(5): 다른 씨앗·noimp·막기·null
    assert ng.main_judgment(_bsum(), {"inert": null}, noimp, knock, None)["code"] == 5
    assert ng.main_judgment(_bsum(), {"inert": null}, None, knock, _bsum())["code"] == 5
    assert ng.main_judgment(_bsum(), {"inert": null}, noimp, None, _bsum())["code"] == 5
    assert ng.main_judgment(_bsum(), {"inert": null, "off": None}, noimp, knock, _bsum())["code"] == 5
    assert ng.main_judgment(_bsum(), {}, noimp, knock, _bsum())["code"] == 5


def test_calibrate_prevalence_bisect():
    calls = []

    def fn(theta):           # θ가 크면 유병률이 내려가는 단조 함수
        calls.append(theta)
        return 0.16 * (1.0 - theta) / 2.0
    out = ng.calibrate_sig_theta(fn, 0.05, -1.0, 1.0, iters=6)
    assert len(calls) <= 6 and out["ok"] and abs(out["measured"] - 0.05) <= 0.2 * 0.05
    q0 = ng.noise_q0(0.08, 50)
    assert q0 == pytest.approx(0.08 / 51) and 0.0015 < q0 < 0.0016
    assert ng.noise_q_fix(q0, 0.08, 0.04) == pytest.approx(2 * q0)
    assert ng.noise_q_fix(q0, 0.08, 0.0) == q0


def test_saved_est_hand_values():
    h = np.zeros((4, 4, 2))
    h[3, :, 0] = 100.0
    h[3, :, 1] = 20.0          # 표시됐으나 안 벌받음: 가로채기율 0.2
    h[0, 0] = (50.0, 5.0)      # 2자 벌 받음, 구간 0: 0.1
    h[2, 1] = (10.0, 0.0)      # 3자 남, 구간 1: 0
    out = ng.saved_est(h, take_dmg_recv=30.0, take=100.0)
    assert out["mean_dmg"] == pytest.approx(0.3)
    assert out["self"] == pytest.approx((0.2 - 0.1) * 50 * 0.3)
    assert out["stranger"] == pytest.approx(0.2 * 10 * 0.3)
    assert out["kin"] == 0.0 and out["label"] == ng.LABEL_HIST


def test_output_labels_fixed():
    assert "조작 확인용" in ng.LABEL_HIST and "엔진 위축" in ng.LABEL_HIST
    assert ng.LABEL_IMPOSED == "부여된 위축 아래서"
    assert "이득 구조 확인이지 관습이 아니다" == ng.LABEL_ARB
    assert ng.RESULTS_HEAD.startswith("①②와 벌의 물리·위축은 부여") and "'규범'이라는 말을 쓰지 않는다" in ng.RESULTS_HEAD
    assert set(ng.THRESHOLDS) >= {"alpha_min_drop", "take_loss", "prevalence", "src_match", "src_ratio_max",
                                  "selection_min", "slope_frac", "life_steps", "life_income"}
    assert ng.THRESHOLDS["take_loss"] == (0.08, 0.15) and ng.THRESHOLDS["prevalence"] == (0.005, 0.10)


# ======================= 작은 세계(시뮬레이션) =======================
def test_observe_window_resets_and_reports():
    cfg = small("stage4", "conflict", "norm", norm_impulse=0.03)
    w = World(cfg)
    w.run(200)
    w.s.norm_tab[:] = 99.0         # 창 시작에 지워져야 한다
    out = nj.observe_world(w, 300, sample_every=100, late_tau=cfg.norm_mark_tau)
    assert out["norm_tab"]["alive_steps"] == pytest.approx(float(w.s.norm_tab[:, k.NT_ALIVE_STEPS].sum()))
    for key in ("prevalence", "theta_t", "take_loss_frac", "punished_per_episode", "p2", "p3", "hist",
                "sim_neighbor_hist", "sim_random_hist", "vis_nonzero_frac", "vis_present_frac", "late_frac"):
        assert key in out, key
    assert 0.0 <= out["theta_t"] <= 1.0 and out["take_loss_frac"] is not None
    assert out["vis_awake_samples"] > 0
    if out["vis_present_frac"]:
        assert abs(out["vis_nonzero_frac"] - out["vis_present_frac"]) <= 0.2 * out["vis_present_frac"] + 0.02
    assert out["ledger_rel"] < 1e-9
    # 장치를 끈 세계(측정만): 표시·벌 값은 안 낸다
    wm = World(small("stage4", "conflict", "norm_size", norm_measure=True, norm_impulse=0.03))
    om = nj.observe_world(wm, 200, sample_every=100)
    assert "theta_t" in om and "prevalence" not in om and "p2" not in om


def test_hand_dose_and_gate_hand_end_to_end(tmp_path):
    cfg = small("stage4", "conflict", "norm_size", max_agents=200, init_agents=150)
    w = World(cfg)
    w.run(50)
    path = ckpt.save(w, tmp_path / "checkpoints")
    out = nj.hand_dose(path, "hand-phi0.3", "std", n=60, steps=200, reps=2)
    assert len(out["runs"]) == 2 and out["groups"] == [[0, 24], [1, 18], [2, 18]]
    r0 = out["runs"][0]
    assert set(r0["norm_by_tag"]) == {0, 1, 2} and r0["damage_rate"] is not None
    assert "n_ind" in out["summary"]
    (tmp_path / "h-hand-phi0.3-std.json").write_text(json.dumps(out, default=float))
    g = nj.gate_hand_dir(tmp_path)
    assert g["code"] == 5 and "base" in g["missing"]
    base = nj.hand_dose(path, "base", "std", n=60, steps=100, reps=2)
    assert len(base["neutral"]) == 2
    off = nj.hand_dose(path, "off", "std", n=60, steps=100, reps=1)
    assert off["runs"][0]["damage_rate"] is not None
