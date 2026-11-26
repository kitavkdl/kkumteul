"""부모의 학습 보조 투자 판정 도구 시험(과제 9): `taste_gene_judge.py --gene show|feed --teach-d`, 순수 계산(`teach_gate.py`),
관문 −1·−2·α·β, 선결, observe 창.

설계 docs/superpowers/specs/2026-11-26-parent-teaching-design.md 6절, 계획 docs/superpowers/plans/2026-11-26-parent-teaching.md 과제 9.
시뮬레이션 없이 도는 것(순수 계산·옵션 적용·파일 읽기)이 대부분이고, observe 창과 사용 예 한 번만 작은 세계를 잠깐 돌린다.

    python3 -m pytest -q tests/test_teach_judge.py
"""
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_stages import small  # noqa: E402
from kkumteul import checkpoint as ckpt  # noqa: E402
from kkumteul import kernels as k  # noqa: E402
from kkumteul import measure  # noqa: E402
from kkumteul.world import World  # noqa: E402

STAGE = Path(__file__).resolve().parents[1] / "wayfinder" / "stage2-4"


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, STAGE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _judge():
    return _load("taste_gene_judge_teach", "taste_gene_judge.py")


def _gate():
    return _load("teach_gate_test", "teach_gate.py")


def teach_world(**kw) -> World:
    return World(small("stage2", "kinds", "taste", soc_taste_enabled=True, juv_enabled=True, teach_enabled=True,
                       taste_gene_enabled=True, **kw))


# ---- zeroed_side ----
def test_zeroed_side_show_feed():
    tj = _judge()
    w = teach_world(feed_enabled=True)
    w.s.show_g[w.s.alive] = 0.4
    w.s.feed_g[w.s.alive] = 0.6
    side = measure.side_from_world(w)
    z = tj.zeroed_side(side, "show")
    assert np.all(z["genomes"]["show_g"] == 0.0)
    assert np.allclose(side["genomes"]["show_g"], 0.4)                       # 원본 배열을 안 건드림
    assert np.allclose(z["genomes"]["feed_g"], 0.6)                          # 다른 유전자는 그대로
    f = tj.zeroed_side(side, "feed")
    assert np.all(f["genomes"]["feed_g"] == 0.0) and np.allclose(f["genomes"]["show_g"], 0.4)
    assert np.allclose(side["genomes"]["feed_g"], 0.6)


def test_zeroed_side_teach_off_raises():
    tj = _judge()
    juv_only = measure.side_from_world(World(small("stage2", "kinds", "taste", soc_taste_enabled=True, juv_enabled=True)))
    with pytest.raises(ValueError, match="부모의 가르침 장치가 꺼진 체크포인트"):
        tj.zeroed_side(juv_only, "show")                                     # teach 꺼짐
    plain = measure.side_from_world(World(small("stage2", "kinds", "taste", soc_taste_enabled=True)))
    with pytest.raises(ValueError, match="부모의 가르침 장치가 꺼진 체크포인트"):
        tj.zeroed_side(plain, "show")                                        # juv 꺼짐
    with pytest.raises(ValueError, match="부모의 가르침 장치가 꺼진 체크포인트"):
        tj.zeroed_side(juv_only, "feed")                                     # feed 꺼짐
    tj.zeroed_side(juv_only, "show", check=False)                            # 음성 대조는 검사를 건너뛴다
    tj.zeroed_side(juv_only, "feed", check=False)


def test_zeroed_side_unknown_gene_message_lists_show_feed():
    tj = _judge()
    side = measure.side_from_world(teach_world())
    with pytest.raises(ValueError) as e:
        tj.zeroed_side(side, "nope")
    for name in ("taste", "eta", "soc", "show", "feed"):
        assert name in str(e.value)


# ---- 옵션 ----
def test_apply_options_teach_d():
    tj = _judge()
    w = teach_world(feed_enabled=True)
    side = measure.side_from_world(w)
    a = tj.zeroed_side(side, "show")
    b = side
    a2, b2 = tj.apply_options(a, b, teach_d=1.0, teach_gene="show")
    assert np.all(b2["genomes"]["show_g"] == 1.0) and np.all(a2["genomes"]["show_g"] == 0.0)
    assert np.all(b["genomes"]["show_g"] == 0.0)                              # 원본 배열을 안 건드림
    af = tj.zeroed_side(side, "feed")
    a3, b3 = tj.apply_options(af, b, teach_d=0.5, teach_gene="feed")
    assert np.all(b3["genomes"]["feed_g"] == 0.5) and np.all(b3["genomes"]["show_g"] == 0.0)
    # --set(여러 번)이 두 쪽 설정에 적용되고, 뒤의 --set이 앞을 덮어쓴다
    a4, b4 = tj.apply_options(a, b, sets=["teach_mode=costonly", "teach_cost=0", "teach_mode=any"])
    for s in (a4, b4):
        assert s["cfg"].teach_mode == "any" and s["cfg"].teach_cost == 0.0
    assert b["cfg"].teach_mode == "kin"                                       # 원래 옆은 그대로
    # --soc-none: 양쪽 soc_g = −soc_rate(보고 배우기 실제 속도 0)
    a5, b5 = tj.apply_options(a, b, soc_none=True)
    for s in (a5, b5):
        assert np.allclose(s["genomes"]["soc_g"], -s["cfg"].soc_rate)
    assert np.allclose(b["genomes"]["soc_g"], 0.0)


def test_build_pair_show_with_sets_and_negative_control():
    tj = _judge()
    side = measure.side_from_world(World(small("stage2", "kinds", "taste", soc_taste_enabled=True,
                                               taste_gene_enabled=True)))      # 장치가 꺼진 체크포인트
    sets = ["juv_enabled=true", "teach_enabled=true", "teach_mode=kin"]
    a, b = tj.build_pair(side, "show", sets=sets, teach_d=1.0)               # 옵션으로 켜면 된다
    assert b["cfg"].teach_enabled and np.all(b["genomes"]["show_g"] == 1.0) and np.all(a["genomes"]["show_g"] == 0.0)
    with pytest.raises(ValueError, match="부모의 가르침 장치가 꺼진 체크포인트"):
        tj.build_pair(side, "show", teach_d=1.0)                             # 옵션도 없으면 오류
    # 음성 대조: teach_enabled=false를 --set으로 주면 켜짐 검사를 건너뛴다
    a, b = tj.build_pair(side, "show", sets=["juv_enabled=true", "teach_enabled=false"], teach_d=1.0)
    assert not b["cfg"].teach_enabled and np.all(b["genomes"]["show_g"] == 1.0)
    # feed: feed_enabled를 직접 만지면 건너뜀, 아니면 검사
    a, b = tj.build_pair(side, "feed", sets=["juv_enabled=true", "feed_enabled=true"], teach_d=1.0)
    assert b["cfg"].feed_enabled and np.all(b["genomes"]["feed_g"] == 1.0) and np.all(a["genomes"]["feed_g"] == 0.0)
    with pytest.raises(ValueError, match="부모의 가르침 장치가 꺼진 체크포인트"):
        tj.build_pair(side, "feed", teach_d=1.0)


# ---- 합성 졸업표 ----
def _tab(grad, dead, e_grad, teach_cost=0.0, **kw):
    d = {n: 0.0 for n in k.GT_NAMES}
    d.update(grad=float(grad), dead=float(dead), e_grad=float(e_grad), teach_cost=float(teach_cost))
    d.update({kk: float(v) for kk, v in kw.items()})
    return d


def test_net_per_birth_hand_values():
    tg = _gate()
    gl = [
        {0: _tab(90, 10, 180), 1: _tab(95, 5, 210, teach_cost=5.0)},                               # 1.8 → 2.1, 비용 0.05
        {0: _tab(80, 20, 160, teach_cost=1.0), 1: _tab(90, 10, 190, teach_cost=10.0)},             # 1.6 → 1.9, 비용 0.09
        {0: _tab(100, 0, 200), 1: _tab(100, 0, 224, teach_cost=5.0)},                              # 2.0 → 2.24, 비용 0.05
    ]
    out = tg.net_per_birth(gl)
    assert out["per_rep"]["G"] == pytest.approx([0.3, 0.3, 0.24])
    assert out["per_rep"]["C"] == pytest.approx([0.05, 0.09, 0.05])
    assert out["per_rep"]["N"] == pytest.approx([0.25, 0.21, 0.19])
    n = np.array([0.25, 0.21, 0.19])
    sd = float(n.std(ddof=1))
    assert out["N"]["mean"] == pytest.approx(n.mean())
    assert out["N"]["sd"] == pytest.approx(sd)
    assert out["N"]["n"] == 3
    edge = measure.t95(2) * sd / math.sqrt(3)                                                        # df = n − 1 = 2
    assert out["N"]["edge"] == pytest.approx(edge)
    assert out["N"]["lo"] == pytest.approx(n.mean() - edge) and out["N"]["hi"] == pytest.approx(n.mean() + edge)
    assert out["G"]["mean"] == pytest.approx(np.mean([0.3, 0.3, 0.24]))
    # json을 한 번 거쳐 태그 키가 문자열이 돼도 같다
    again = tg.net_per_birth(json.loads(json.dumps(gl)))
    assert again["per_rep"]["N"] == pytest.approx(out["per_rep"]["N"])


def test_net_per_birth_counts_feed_given_and_skips_empty_reps():
    tg = _gate()
    gl = [{0: _tab(100, 0, 200), 1: _tab(100, 0, 210, feed_given=4.0)},          # 건넨 양 0.04도 비용
          {0: _tab(0, 0, 0), 1: _tab(0, 0, 0)}]                                  # 태어난 새끼가 없는 반복은 건너뜀(nan)
    out = tg.net_per_birth(gl)
    assert out["per_rep"]["G"][0] == pytest.approx(0.1) and out["per_rep"]["C"][0] == pytest.approx(0.04)
    assert out["per_rep"]["N"][0] == pytest.approx(0.06)
    assert math.isnan(out["per_rep"]["N"][1]) and out["N"]["n"] == 1


def test_breakeven_hand_values():
    tg = _gate()
    # C = 0.05, teach_cost 0.0005, d = 1 → X = C ÷ (teach_cost × d) = 100, c* = G ÷ (X × d) = 0.2 ÷ 100 = 0.002
    be = tg.breakeven(G=0.2, C=0.05, teach_cost=0.0005, d=1.0)
    assert be["X"] == pytest.approx(100.0) and be["c_star"] == pytest.approx(0.002)
    assert be["profitable_at_cost"] is True                                          # c* > 0.0005
    be = tg.breakeven(G=0.2, C=0.025, teach_cost=0.0005, d=0.5)
    assert be["X"] == pytest.approx(100.0) and be["c_star"] == pytest.approx(0.2 / (100.0 * 0.5))
    assert tg.breakeven(G=0.02, C=0.05, teach_cost=0.0005, d=1.0)["profitable_at_cost"] is False
    none = tg.breakeven(G=0.2, C=0.0, teach_cost=0.0, d=1.0)                         # 비용 0 칸: 정의 안 됨
    assert none["X"] is None and none["c_star"] is None and none["profitable_at_cost"] is None
    assert tg.breakeven(G=0.2, C=0.05, teach_cost=0.0005, d=0.0)["c_star"] is None


def test_detectability_ratio():
    tg = _gate()
    ctl = [0.02, -0.02, 0.01, -0.01, 0.03, -0.03]
    sd = float(np.std(ctl, ddof=1))
    limit = measure.t95(5) * sd / math.sqrt(6)
    det = tg.detectability(0.08, ctl)
    assert det["limit"] == pytest.approx(limit) and det["ratio"] == pytest.approx(0.08 / limit)
    assert det["ok"] is True
    det = tg.detectability(0.5 * limit, ctl)                    # 비가 1 아래 = 검출 한계 아래(β를 쓰지 않는다)
    assert det["ratio"] == pytest.approx(0.5) and det["ok"] is False
    assert tg.detectability(-0.08, ctl)["ok"] is False          # 이득이 음이면 해당 없음
    assert tg.detectability(0.08, [0.01])["ok"] is False         # 표본 하나로는 한계를 못 잰다


def test_selection_strength_gate_2b():
    tg = _gate()
    assert tg.selection_strength(500, 0.034) == pytest.approx(500 * 0.034 / 17.0)
    assert tg.selection_ok(500, 0.034) is True                  # 설계: N=500에서 순이득 0.034 위
    assert tg.selection_ok(500, 0.03) is False
    assert tg.selection_ok(50, 0.1) is False
    assert tg.selection_ok(500, -0.1) is False


# ---- 합성 겨루기 결과(rate_zero 꼴) ----
U = [1.0, -1.0, 2.0, -2.0, 0.5, -0.5]


def _cell(*, pf=(0.40, 0.40), depth=(0.10, 0.10), e=(2.0, 2.0), cost_b=0.0, noise_e=0.01, noise_pf=0.002,
          d=1.0, teach_cost=0.0005, both=0, reps=6):
    """반복 6판의 합성 겨루기 결과. 태어난 새끼는 태그마다 100(졸업 90 + 사망 10).
    pf: 졸업 때 독 끼니 비율 (A, B), depth: 독 모양 기억 깊이 (A, B), e: 태어난 새끼당 졸업 에너지 (A, B)."""
    gl = []
    for j in range(reps):
        u = U[j]
        tabs = {}
        for tag, (p, dp, ee) in enumerate(zip(pf, depth, e)):
            pj = p + (noise_pf * u if tag == 1 else 0.0)
            ej = ee + (noise_e * u if tag == 1 else 0.0)
            tabs[tag] = _tab(90, 10, ej * 100, teach_cost=(cost_b * 100 if tag == 1 else 0.0),
                             meals=900, poison_meals=pj * 900, m3=dp * 90, n2=90, m2=dp * 90,
                             n1=90, m1=dp * 90, k3=40)
        gl.append(tabs)
    return {"grad_by_tag_list": gl, "reps": reps, "both_extinct_runs": both,
            "r_list": [0.0] * reps, "teach": {"teach_d": d, "teach_cost": teach_cost}}


def _good_results(pop_noise=0.01):
    """작동 + 이득: kin의 독 끼니 비율 차 0.04(costonly 0), 깊이 +0.05, 순이득 +0.15, 음성 대조는 잡음뿐."""
    r = {"off-d1.0": _cell(noise_e=pop_noise)}
    for d in (0.1, 0.5, 1.0):
        r[f"kin-d{d}"] = _cell(pf=(0.40, 0.36), depth=(0.10, 0.15), e=(2.0, 2.2), cost_b=0.05, d=d, noise_e=pop_noise)
        r[f"costonly-d{d}"] = _cell(pf=(0.40, 0.40), depth=(0.10, 0.10), e=(2.0, 2.0), cost_b=0.05, d=d,
                                    noise_e=pop_noise)
    return r


def test_gate_alpha_beta_works_and_gains():
    tg = _gate()
    g = tg.gate_alpha_beta(_good_results(), pop=500)
    assert g["status"] == "pass"
    assert g["alpha"]["1.0"]["works"] is True and g["alpha"]["1.0"]["pf_welch"]["pos"] is True
    assert g["alpha"]["1.0"]["pf_welch"]["diff"] == pytest.approx(0.04, abs=0.01)
    b = g["beta"]["1.0"]
    assert b["N"]["lo"] > 0 and b["G_welch"]["pos"] is True and b["det"]["ok"] is True and b["gain"] is True
    assert b["N"]["mean"] == pytest.approx(0.15, abs=0.02)
    assert g["control"]["ok"] is True
    assert g["selection"]["ok"] is True and g["evolve_ok"] is True
    # 손익분기: 순이득 0.15, G 0.2 → c* = 0.2 ÷ (X × d), X = 0.05 ÷ (0.0005 × 1) = 100
    assert g["breakeven"]["kin-d1.0"]["c_star"] == pytest.approx(0.002, rel=0.1)


def test_gate_alpha_beta_works_but_gain_undetermined():
    tg = _gate()
    r = _good_results(pop_noise=0.05)                           # 음성 대조의 잡음이 커서 한계가 큼
    for d in (0.1, 0.5, 1.0):
        r[f"kin-d{d}"] = _cell(pf=(0.40, 0.36), depth=(0.10, 0.15), e=(2.0, 2.06), cost_b=0.05, d=d, noise_e=0.05)
        r[f"costonly-d{d}"] = _cell(pf=(0.40, 0.40), depth=(0.10, 0.10), e=(2.0, 2.0), cost_b=0.05, d=d, noise_e=0.05)
    g = tg.gate_alpha_beta(r, pop=500)
    assert g["status"] == "works_undetermined"                  # 작동하나 이득이 비용을 넘는지는 가를 수 없음
    assert g["alpha"]["1.0"]["works"] is True
    assert g["beta"]["1.0"]["gain"] is False
    assert g["beta"]["1.0"]["det"]["ok"] is False               # 관문 −2: 검출 한계 아래라 β를 쓰지 않는다
    assert g["evolve_ok"] is False                              # 관문 −2b: N × 순이득 ÷ 17 < 1


def test_gate_alpha_beta_not_working():
    tg = _gate()
    r = _good_results()
    for d in (0.1, 0.5, 1.0):                                   # kin의 독 끼니 비율 차가 costonly와 같다
        r[f"kin-d{d}"] = _cell(pf=(0.40, 0.40), depth=(0.10, 0.10), e=(2.0, 2.2), cost_b=0.05, d=d)
    g = tg.gate_alpha_beta(r, pop=500)
    assert g["status"] == "not_working" and g["evolve_ok"] is False
    # 차가 0.01 아래면(Welch는 양이어도) 작동이 아니다
    r = _good_results()
    for d in (0.1, 0.5, 1.0):
        r[f"kin-d{d}"] = _cell(pf=(0.40, 0.395), depth=(0.10, 0.15), e=(2.0, 2.2), cost_b=0.05, d=d, noise_pf=0.0005)
    assert tg.gate_alpha_beta(r, pop=500)["status"] == "not_working"
    # (나) 깊이 차가 6판 중 5판 양이 아니면 작동이 아니다
    r = _good_results()
    for d in (0.1, 0.5, 1.0):
        r[f"kin-d{d}"] = _cell(pf=(0.40, 0.36), depth=(0.10, 0.10), e=(2.0, 2.2), cost_b=0.05, d=d)
    assert tg.gate_alpha_beta(r, pop=500)["status"] == "not_working"


def test_gate_alpha_beta_broken_tool():
    tg = _gate()
    r = _good_results()
    r["off-d1.0"] = _cell(both=3)                               # 음성 대조의 양쪽 전멸 3판
    g = tg.gate_alpha_beta(r)
    assert g["status"] == "broken" and "전멸" in g["reason"]
    r = _good_results()
    r["off-d1.0"] = _cell(e=(2.0, 2.4), noise_e=0.01)           # 음성 대조의 순이득이 한계 밖
    assert tg.gate_alpha_beta(r)["status"] == "broken"
    r = _good_results()
    del r["off-d1.0"]                                           # 음성 대조가 없으면 판정을 쓰지 않는다
    g = tg.gate_alpha_beta(r)
    assert g["status"] == "broken" and "음성 대조" in g["reason"]
    assert tg.gate_alpha_beta({"off-d1.0": _cell()})["status"] == "broken"     # 읽을 kin 칸이 없음


def test_gate_alpha_beta_exp_world_only_when_exp_ok():
    tg = _gate()
    r = _good_results()
    r["kin-exp-d1.0"] = _cell(pf=(0.40, 0.30), depth=(0.10, 0.20), e=(2.0, 2.5), cost_b=0.05)
    r["costonly-exp-d1.0"] = _cell(pf=(0.40, 0.40), depth=(0.10, 0.10), e=(2.0, 2.0), cost_b=0.05)
    assert tg.gate_alpha_beta(r, exp_ok=False)["exp"] is None
    g = tg.gate_alpha_beta(r, exp_ok=True)
    assert g["exp"]["status"] == "pass" and g["exp"]["beta"]["1.0"]["gain"] is True
    # 읽기용 칸(meal 비용·any·knower·feed)은 값을 낸다
    r["kin-meal-d1.0"] = _cell(pf=(0.40, 0.36), depth=(0.10, 0.15), e=(2.0, 2.2), cost_b=0.01)
    g = tg.gate_alpha_beta(r)
    assert "kin-meal-d1.0" in g["breakeven"] and g["readings"]["kin-meal-d1.0"]["N"]["mean"] > 0


def test_first_step_gain_needs_continuous_index():
    tg = _gate()
    r = _good_results()
    g = tg.gate_alpha_beta(r)
    assert g["first_step"]["satisfied"] is True                 # d=0.1에서 깊이가 costonly보다 Welch로 양
    r["kin-d0.1"] = _cell(pf=(0.40, 0.40), depth=(0.10, 0.10), e=(2.0, 2.0), cost_b=0.05, d=0.1)
    assert tg.gate_alpha_beta(r)["first_step"]["satisfied"] is False


def test_gate_combines_tags_exit_codes():
    tg = _gate()
    good = _good_results()
    undet = _good_results(pop_noise=0.05)
    for d in (0.1, 0.5, 1.0):
        undet[f"kin-d{d}"] = _cell(pf=(0.40, 0.36), depth=(0.10, 0.15), e=(2.0, 2.06), cost_b=0.05, d=d, noise_e=0.05)
        undet[f"costonly-d{d}"] = _cell(pf=(0.40, 0.40), depth=(0.10, 0.10), e=(2.0, 2.0), cost_b=0.05, d=d,
                                        noise_e=0.05)
    notw = _good_results()
    for d in (0.1, 0.5, 1.0):
        notw[f"kin-d{d}"] = _cell(pf=(0.40, 0.40), depth=(0.10, 0.10), e=(2.0, 2.2), cost_b=0.05, d=d)
    bad = _good_results()
    bad["off-d1.0"] = _cell(both=3)
    c = tg.gate_tags({"a": good, "b": undet})
    assert c["exit_code"] == 0 and c["status"] == "pass" and set(c["tags"]) == {"a", "b"}
    assert tg.gate_tags({"a": undet, "b": notw})["exit_code"] == 3             # 모두 가를 수 없음
    assert tg.gate_tags({"a": undet, "b": undet})["status"] == "undetermined"
    c = tg.gate_tags({"a": bad, "b": undet})
    assert c["exit_code"] == 4 and c["status"] == "broken"                      # 통과 없이 깨진 바탕이 있음
    assert tg.gate_tags({"a": bad, "b": good})["exit_code"] == 0                # 한 바탕이라도 통과면 0
    assert tg.gate_tags({})["exit_code"] == 4


# ---- 관문 −1(세계 점검) ----
def _obs(pop=500.0, ledger=1e-12, reseeds=0, near=0.4, meal=0.0158, poison=0.43, grad_rate=0.9, juv_steps=200):
    return {"pop_mean": pop, "ledger_rel_residual": ledger, "reseeds": reseeds, "par_near_frac": near,
            "meal_rate_per_agent_step": meal, "poison_frac": poison, "grad_rate": grad_rate, "juv_steps": juv_steps}


def _world(on=None, exp=None):
    off = _obs(near=0.2, grad_rate=0.95)
    on_kw = dict(pop=480.0, **(on or {}))
    exp_kw = dict(pop=470.0, grad_rate=0.6)
    exp_kw.update(exp or {})
    return {"off": off, "on": _obs(**on_kw), "exp": _obs(**exp_kw)}


def test_gate_world_passes_and_expected_par_poison_formula():
    tg = _gate()
    w = tg.gate_world({"a": _world(), "b": _world()})
    assert w["exit_code"] == 0 and w["status"] == "pass" and w["exp_ok"] is True
    t = w["tags"]["a"]
    assert t["a"]["pass"] and t["b"]["pass"] and t["c"]["pass"]
    # −1b: par_near_frac × juv_steps × 끼니율 × 독 비율
    assert t["b"]["expected_par_poison_seen"] == pytest.approx(0.4 * 200 * 0.0158 * 0.43)


@pytest.mark.parametrize("over,key", [
    ({"on": {"ledger": 1e-3}}, "a"),                       # 장부가 안 닫힘
    ({"on": {"reseeds": 500}}, "a"),                       # 재주입 급증
])
def test_gate_world_1a_fails_exit_3(over, key):
    tg = _gate()
    w = tg.gate_world({"a": _world(**over), "b": _world()})
    assert w["exit_code"] == 3 and w["tags"]["a"]["a"]["pass"] is False


def test_gate_world_1a_pop_ratio():
    tg = _gate()
    bad = _world()
    bad["on"]["pop_mean"] = 400.0                           # 끈 무리 500의 0.8
    w = tg.gate_world({"a": bad, "b": _world()})
    assert w["exit_code"] == 3 and w["tags"]["a"]["a"]["pop_ratio"] == pytest.approx(0.8)
    ok = _world()
    ok["on"]["pop_mean"] = 450.0                            # 정확히 0.9는 통과
    assert tg.gate_world({"a": ok, "b": _world()})["exit_code"] == 0


def test_gate_world_1b_not_enough_company_exit_4():
    tg = _gate()
    bad = _world(on={"near": 0.2})                          # 0.2 × 200 × 0.0158 × 0.43 = 0.27 < 0.5
    w = tg.gate_world({"a": bad, "b": _world()})
    assert w["exit_code"] == 4 and w["tags"]["a"]["b"]["pass"] is False
    # −1a도 못 넘으면 세계가 못 선 쪽(3)이 먼저
    bad["on"]["pop_mean"] = 100.0
    assert tg.gate_world({"a": bad, "b": _world()})["exit_code"] == 3


def test_gate_world_1c_only_closes_exp_world():
    tg = _gate()
    for over in ({"exp": {"grad_rate": 0.3}}, ):             # 졸업률 0.3 < 끈 0.95의 0.5
        w = tg.gate_world({"a": _world(**over), "b": _world()})
        assert w["tags"]["a"]["c"]["pass"] is False and w["exp_ok"] is False
        assert w["exit_code"] == 0                          # 정보 비싼 세계 판만 닫는다(주 판은 간다)
    bad = _world()
    bad["exp"]["pop_mean"] = 400.0                          # 0.8 < 0.9
    assert tg.gate_world({"a": bad, "b": _world()})["exp_ok"] is False
    miss = _world()
    del miss["exp"]
    w = tg.gate_world({"a": miss, "b": _world()})
    assert w["exp_ok"] is False and w["exit_code"] == 0


def test_gate_world_missing_main_files_exit_3():
    tg = _gate()
    w = tg.gate_world({"a": {"off": _obs()}, "b": _world()})        # 켠 세계 판독이 없음
    assert w["exit_code"] == 3
    assert tg.gate_world({})["exit_code"] == 3


# ---- 선결: 사회 학습 판정 0 ----
def _soc_cell(a, b):
    return {"first10_by_tag": {"0": list(a), "1": list(b)}}


def test_prereq_soc_welch_on_first10():
    tg = _gate()
    good = {"outcome-d0.25": _soc_cell([400, 1000], [300, 1000]), "outcome-d1.0": _soc_cell([400, 1000], [398, 1000])}
    flat = {"outcome-d0.25": _soc_cell([400, 1000], [395, 1000]), "outcome-d1.0": _soc_cell([400, 1000], [402, 1000])}
    p = tg.prereq_soc({"a": good, "b": flat})
    assert p["exit_code"] == 0 and p["tags"]["a"]["ok"] is True and p["tags"]["b"]["ok"] is False
    assert p["tags"]["a"]["d"]["0.25"]["diff"] == pytest.approx(0.1)
    assert tg.prereq_soc({"a": flat, "b": flat})["exit_code"] == 3
    assert tg.prereq_soc({"a": {}, "b": {}})["exit_code"] == 3                  # 파일이 없으면 진행 안 함
    rev = {"outcome-d0.25": _soc_cell([300, 1000], [400, 1000])}                # A−B가 음이면 아님
    assert tg.prereq_soc({"a": rev})["exit_code"] == 3
    zero = {"outcome-d0.25": _soc_cell([0, 0], [0, 0])}
    assert tg.prereq_soc({"a": zero})["exit_code"] == 3                         # 끼니 0은 읽을 수 없음


# ---- 파일 읽기(CLI) ----
def test_cli_gate_commands_read_files(tmp_path, capsys):
    tj = _judge()
    for tag in ("a", "b"):
        for name, res in _good_results().items():
            (tmp_path / f"j-dose-{name}-{tag}.json").write_text(json.dumps(res))
        for name, obs in _world().items():
            (tmp_path / f"w-{name}-{tag}.json").write_text(json.dumps(obs))
    out = tmp_path / "gw.json"
    assert tj.main(["gate-world", "--dir", str(tmp_path), "--tags", "a", "b", "--json", str(out)]) == 0
    got = json.loads(out.read_text())
    assert got["exp_ok"] is True and set(got["tags"]) == {"a", "b"}
    out = tmp_path / "gate.json"
    assert tj.main(["gate-teach", "--dir", str(tmp_path), "--tags", "a", "b", "--json", str(out)]) == 0
    got = json.loads(out.read_text())
    assert got["status"] == "pass" and set(got["tags"]) == {"a", "b"}
    assert got["tags"]["a"]["exp"] is None                                   # 정보 비싼 세계 칸이 없으면 안 읽음
    # 파일이 하나도 없는 바탕은 도구 깨짐
    assert tj.main(["gate-teach", "--dir", str(tmp_path), "--tags", "zz"]) == 4
    assert tj.main(["gate-world", "--dir", str(tmp_path), "--tags", "zz"]) == 3
    # 선결
    for tag in ("a", "b"):
        for d in ("0.25", "1.0"):
            (tmp_path / f"j-dose-outcome-d{d}-{tag}.json").write_text(json.dumps(_soc_cell([400, 1000], [300, 1000])))
    assert tj.main(["prereq-soc", "--dir", str(tmp_path), "--tags", "a", "b"]) == 0
    assert tj.main(["prereq-soc", "--dir", str(tmp_path / "none"), "--tags", "a"]) == 3


def test_gate_teach_cli_reads_exp_ok_from_gate_world_file(tmp_path):
    tj = _judge()
    r = _good_results()
    r["kin-exp-d1.0"] = _cell(pf=(0.40, 0.30), depth=(0.10, 0.20), e=(2.0, 2.5), cost_b=0.05)
    r["costonly-exp-d1.0"] = _cell(pf=(0.40, 0.40), depth=(0.10, 0.10), e=(2.0, 2.0), cost_b=0.05)
    for name, res in r.items():
        (tmp_path / f"j-dose-{name}-a.json").write_text(json.dumps(res))
    out = tmp_path / "g.json"
    tj.main(["gate-teach", "--dir", str(tmp_path), "--tags", "a", "--json", str(out)])
    assert json.loads(out.read_text())["tags"]["a"]["exp"] is None            # gate-world.json이 없으면 읽지 않는다
    (tmp_path / "gate-world.json").write_text(json.dumps({"exp_ok": True}))
    tj.main(["gate-teach", "--dir", str(tmp_path), "--tags", "a", "--json", str(out)])
    assert json.loads(out.read_text())["tags"]["a"]["exp"]["status"] == "pass"


# ---- observe 창 ----
def test_observe_window_reports_grad(tmp_path, monkeypatch):
    tj = _judge()
    w = teach_world(feed_enabled=True)
    w.run(300)
    w.s.show_g[w.s.alive] = 0.25
    path = ckpt.save(w, tmp_path)
    calls = []
    real = World.reset_grad_table
    monkeypatch.setattr(World, "reset_grad_table", lambda self: calls.append(self.step_no) or real(self))
    o = tj.observe(path, steps=500)
    assert calls == [300]                                                      # 창 시작에서 한 번
    assert o["juv_steps"] == w.cfg.juv_steps and o["pop_mean"] > 0
    assert o["show_g"]["mean"] == pytest.approx(0.25) and "q25" in o["show_g"] and "q75" in o["show_g"]
    assert "feed_g" in o
    # 같은 창을 손으로 다시 돌린 졸업표와 같다
    w2, _ = ckpt.load(path)
    real(w2)
    w2.run(500)
    tab = {name: sum(row[name] for row in w2.grad_table().values()) for name in k.GT_NAMES}
    assert o["grad_table"] == pytest.approx(tab)
    born = tab["grad"] + tab["dead"]
    assert o["grad_rate"] == pytest.approx(tab["grad"] / born if born else float("nan"), nan_ok=True)
    assert o["n_seen_par_alive"] == int(w2.s.n_seen_par[w2.s.alive].sum())
    w3, _ = ckpt.load(path)
    assert o["reseeds"] == int(w2.s.counters[k.C_RESEED]) - int(w3.s.counters[k.C_RESEED])
    assert o["ledger_rel_residual"] is not None and o["ledger_rel_residual"] < 1e-6
    if tab["par_alive"]:
        assert o["par_near_frac"] == pytest.approx(tab["par_near"] / tab["par_alive"])
    assert o["meal_rate_per_agent_step"] > 0
    json.dumps(o, default=float)


def test_observe_window_off_world_has_no_juv_keys(tmp_path):
    tj = _judge()
    w = World(small("stage2", "kinds", "taste"))
    w.run(100)
    path = ckpt.save(w, tmp_path)
    o = tj.observe(path, steps=100)
    assert "grad_table" not in o and "show_g" not in o and "pop_mean" not in o


def test_observe_juv_measure_world_reads_grad(tmp_path):
    tj = _judge()
    w = World(small("stage2", "kinds", "taste", juv_measure=True))
    w.run(300)
    path = ckpt.save(w, tmp_path)
    o = tj.observe(path, steps=400)
    assert "grad_table" in o and "show_g" not in o and o["pop_mean"] > 0        # 장치(teach)는 꺼진 측정 세계


# ---- 사용 예 직접 실행(작은 세계) ----
def test_rate_zero_show_runs_end_to_end(tmp_path):
    tj = _judge()
    w = teach_world(feed_enabled=True)
    w.run(300)
    path = ckpt.save(w, tmp_path)
    out = tmp_path / "x.json"
    code = tj.main(["rate-zero", str(path), "--gene", "show", "--teach-d", "1.0", "--n", "20", "--steps", "400",
                    "--reps", "2", "--no-neutral", "--json", str(out)])
    assert code == 0
    res = json.loads(out.read_text())
    assert res["gene"] == "show" and len(res["grad_by_tag_list"]) == 2
    assert res["teach_d"] == 1.0 and res["teach"]["teach_cost"] == w.cfg.teach_cost
    tg = _gate()
    n = tg.net_per_birth(res["grad_by_tag_list"])                              # json을 거친 파일 값으로 계산이 끝난다
    assert n["N"]["n"] <= 2
    # feed도 같은 꼴
    out2 = tmp_path / "f.json"
    code = tj.main(["rate-zero", str(path), "--gene", "feed", "--teach-d", "1.0", "--n", "20", "--steps", "300",
                    "--reps", "2", "--no-neutral", "--set", "teach_enabled=false", "--set", "feed_enabled=true",
                    "--json", str(out2)])
    assert code == 0 and json.loads(out2.read_text())["gene"] == "feed"


@pytest.mark.parametrize("where,key", [
    ("exp", "grad_rate"),    # 정보 비싼 세계에 새끼 0 → 졸업률 None
    ("off", "grad_rate"),    # 끈 세계에 새끼 0
    ("exp", "pop_mean"),
    ("on", "pop_mean"),
    ("off", "pop_mean"),
])
def test_gate_world_none_readings_close_not_crash(where, key):
    """observe가 None(새끼 0 등)을 돌려줘도 TypeError로 죽지 않고 해당 판을 닫는다."""
    tg = _gate()
    bad = _world()
    bad[where][key] = None
    w = tg.gate_world({"a": bad, "b": _world()})
    if key == "grad_rate":
        assert w["exp_ok"] is False and w["exit_code"] == 0
    elif where == "exp":
        assert w["exp_ok"] is False
    else:
        assert w["exit_code"] == 3
