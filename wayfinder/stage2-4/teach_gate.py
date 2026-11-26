"""부모의 학습 보조 투자("가르침" 후보) 판정의 순수 계산: 새끼당 순이득, 관문 −1·−2·−2b·α·β, 손익분기 비용, 선결.

설계 docs/superpowers/specs/2026-11-26-parent-teaching-design.md 6절, 계획 docs/superpowers/plans/2026-11-26-parent-teaching.md 과제 9.
시뮬레이션을 돌리지 않는다. 입력은 `taste_gene_judge.py rate-zero`가 남긴 결과 사전(`grad_by_tag_list` 등)과
`observe` 창 결과 사전이고, 파일 읽기는 `taste_gene_judge.py`가 한다(여기서는 사전만 받는다).
모든 문턱은 임시 결정이며 실행 중에 바꾸지 않는다.

졸업표 한 줄(`grad_by_tag_list[반복][태그]`)은 `kernels.GT_NAMES` 이름의 사전이다. 태그 0 = A(보여 주기 0 벌), 1 = B(강도 d).
json을 한 번 거치면 태그 키가 "0"·"1" 문자열이 되므로 둘 다 읽는다.
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kkumteul import measure  # noqa: E402

# ---- 문턱(임시 결정, 실행 중에 바꾸지 않는다) ----
FRAC_POSITIVE = 5 / 6              # (나): 6판 중 5판 이상 양
CONTROL_MAX_BOTH = 2               # 음성 대조의 양쪽 전멸이 이 판 수를 넘으면 도구 깨짐
ALPHA_MIN_DIFF = 0.01              # (가) kin − costonly 독 끼니 비율 차의 하한
POP_RATIO_MIN = 0.9                # −1a·−1c: 켠(또는 mult 3) 무리 ≥ 끈 무리의 이 비율
GRAD_RATIO_MIN_EXP = 0.5           # −1c: 졸업률 ≥ 끈 세계의 이 비율
EXPECTED_PAR_POISON_MIN = 0.5      # −1b: 새끼당 기대 부모 독 끼니 관찰 수
INCOME = 17.0                      # −2b: 한 생 수입(약 1,000스텝 × 0.017)
SELECTION_MIN = 1.0                # −2b: N × 순이득 ÷ 17
LEDGER_TOL = 1e-6                  # 장부 상대 잔차(수백만 스텝 누적 부동소수를 감안해 시험의 1e-9보다 느슨하게)
RESEED_FACTOR = 2.0                # −1a: 재주입이 끈 세계의 이 배 + RESEED_SLACK을 넘으면 급증
RESEED_SLACK = 5
ALPHA_DS = ("0.1", "0.5", "1.0")
DEFAULT_TEACH_COST = 0.0005
DEFAULT_POP = 500


# ---- 통계 ----
def summarize(vals) -> dict:
    """평균·표준편차·표준오차·반복 수와 t 구간(`measure.t95`, df = n − 1). nan은 건너뛴다."""
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    n = int(len(v))
    if n == 0:
        return {"mean": float("nan"), "sd": float("nan"), "sem": float("nan"), "n": 0,
                "edge": float("nan"), "lo": float("nan"), "hi": float("nan")}
    mean = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else float("nan")
    sem = sd / math.sqrt(n) if n > 1 else float("nan")
    edge = float(measure.t95(n - 1) * sem) if n > 1 else float("nan")
    return {"mean": mean, "sd": sd, "sem": sem, "n": n, "edge": edge, "lo": mean - edge, "hi": mean + edge}


def welch(x, y) -> dict:
    """두 반복 목록의 평균 차(x − y)와 Welch 한계(`measure.t95`, df = min(n) − 1, 기존 관문과 같다). pos = 차가 양이고 한계 위."""
    sx, sy = summarize(x), summarize(y)
    diff = sx["mean"] - sy["mean"]
    if sx["n"] < 2 or sy["n"] < 2 or not np.isfinite(diff):
        return {"diff": diff, "edge": float("inf"), "pos": False, "n": [sx["n"], sy["n"]]}
    se = math.sqrt(sx["sem"] ** 2 + sy["sem"] ** 2)
    df = max(1, min(sx["n"], sy["n"]) - 1)
    edge = float(measure.t95(df) * se)
    return {"diff": float(diff), "edge": edge, "pos": bool(diff > 0 and diff > edge), "n": [sx["n"], sy["n"]]}


def _n_positive(vals) -> tuple[int, int]:
    """(양인 반복 수, 6판 중 5판 규칙의 필요 수). nan은 건너뛴다."""
    v = [x for x in vals if x is not None and np.isfinite(x)]
    need = int(math.ceil(FRAC_POSITIVE * len(v) - 1e-9)) if v else 0
    return int(sum(x > 0 for x in v)), need


# ---- 새끼당 순이득 ----
def _row(g: dict, tag: int):
    for key in (tag, str(tag)):
        if key in g:
            return g[key]
    return None


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def per_rep(grad_by_tag_list) -> dict:
    """반복마다 태그별 졸업표에서 지표를 뽑는다(태어난 새끼 = grad + dead, 죽은 새끼의 졸업 에너지는 0).
    G = e_grad_B/태어난_B − e_grad_A/태어난_A, C = (teach_cost + feed_given)_B/태어난_B − A쪽 같은 값, N = G − C(건넨 양도 비용).
    pf_diff = (A − B) 졸업 때 독 끼니 비율, depth_diff / depth1_diff / depth2_diff = (B − A) 독 모양 기억 깊이(졸업·S1·S2),
    know_diff = (B − A) 졸업 때 안 새끼 비율. 태어난 새끼가 없는 반복은 nan."""
    keys = ("G", "C", "N", "pf_diff", "depth_diff", "depth1_diff", "depth2_diff", "know_diff", "grad_rate_a",
            "grad_rate_b", "born_a", "born_b")
    out = {kk: [] for kk in keys}
    nan = float("nan")
    for g in grad_by_tag_list:
        a, b = _row(g, 0), _row(g, 1)
        if a is None or b is None:
            for kk in keys:
                out[kk].append(nan)
            continue
        ba, bb = a["grad"] + a["dead"], b["grad"] + b["dead"]
        if ba <= 0 or bb <= 0:
            for kk in keys:
                out[kk].append(nan)
            continue
        ga = a["e_grad"] / ba
        gb = b["e_grad"] / bb
        ca = (a.get("teach_cost", 0.0) + a.get("feed_given", 0.0)) / ba
        cb = (b.get("teach_cost", 0.0) + b.get("feed_given", 0.0)) / bb
        out["G"].append(gb - ga)
        out["C"].append(cb - ca)
        out["N"].append((gb - ga) - (cb - ca))
        out["pf_diff"].append(_ratio(a["poison_meals"], a["meals"]) - _ratio(b["poison_meals"], b["meals"]))
        out["depth_diff"].append(_ratio(b["m3"], b["grad"]) - _ratio(a["m3"], a["grad"]))
        out["depth1_diff"].append(_ratio(b["m1"], b["n1"]) - _ratio(a["m1"], a["n1"]))
        out["depth2_diff"].append(_ratio(b["m2"], b["n2"]) - _ratio(a["m2"], a["n2"]))
        out["know_diff"].append(_ratio(b["k3"], b["grad"]) - _ratio(a["k3"], a["grad"]))
        out["grad_rate_a"].append(a["grad"] / ba)
        out["grad_rate_b"].append(b["grad"] / bb)
        out["born_a"].append(ba)
        out["born_b"].append(bb)
    return out


def net_per_birth(grad_by_tag_list) -> dict:
    """새끼당 순이득: {"per_rep": per_rep(...), "G": 통계, "C": 통계, "N": 통계}."""
    pr = per_rep(grad_by_tag_list)
    return {"per_rep": pr, "G": summarize(pr["G"]), "C": summarize(pr["C"]), "N": summarize(pr["N"])}


def breakeven(G: float, C: float, teach_cost: float, d: float) -> dict:
    """손익분기 비용. 순이득이 d × (G₁ − c × X)로 선형이므로 X = C ÷ (teach_cost × d)(새끼당 비용 노출),
    c* = G ÷ (X × d). c* > teach_cost면 그 비용에서도 이득이다. 정의가 안 되면(비용 0 칸·d 0·C ≤ 0) None."""
    if not (np.isfinite(G) and np.isfinite(C)) or teach_cost <= 0 or d <= 0 or C <= 0:
        return {"X": None, "c_star": None, "profitable_at_cost": None, "teach_cost": teach_cost, "d": d}
    X = float(C / (teach_cost * d))
    c_star = float(G / (X * d))
    return {"X": X, "c_star": c_star, "profitable_at_cost": bool(c_star > teach_cost), "teach_cost": teach_cost, "d": d}


# ---- 관문 −2(검출 가능성), −2b(선택이 드리프트를 이기나) ----
def detectability(net_mean: float, control_n_list) -> dict:
    """관문 −2: 검출 한계 = 음성 대조 칸 반복별 순이득의 sd × t95(n−1) ÷ √n. 비 = 예상 순이득 ÷ 한계.
    비가 1 아래(또는 이득이 음, 표본 부족)이면 ok=False — 그 d는 β를 쓰지 않는다."""
    c = summarize(control_n_list)
    if c["n"] < 2 or not np.isfinite(c["sd"]):
        return {"limit": float("nan"), "ratio": float("nan"), "ok": False}
    limit = float(measure.t95(c["n"] - 1) * c["sd"] / math.sqrt(c["n"]))
    if limit <= 0:
        ratio = float("inf") if net_mean > 0 else float("nan")
    else:
        ratio = float(net_mean / limit)
    return {"limit": limit, "ratio": ratio, "ok": bool(net_mean > 0 and np.isfinite(ratio) and ratio >= 1.0)}


def selection_strength(pop: float, net: float, income: float = INCOME) -> float:
    """관문 −2b: 선택 강도 = 무리 크기 × (순이득 ÷ 한 생 수입)."""
    return float(pop * net / income)


def selection_ok(pop: float, net: float, income: float = INCOME) -> bool:
    return bool(np.isfinite(net) and net > 0 and selection_strength(pop, net, income) >= SELECTION_MIN)


# ---- 관문 α·β ----
def _control(ctl: dict) -> dict:
    """음성 대조(off-d1.0): 순이득 평균이 한계 안(0 ± 한계)이고 양쪽 전멸이 CONTROL_MAX_BOTH판 이하."""
    n = net_per_birth(ctl["grad_by_tag_list"])
    nlist = n["per_rep"]["N"]
    det = detectability(1.0, nlist)       # 한계만 쓴다(예상 이득 값은 의미 없음)
    limit = det["limit"]
    both = int(ctl.get("both_extinct_runs", 0))
    mean = n["N"]["mean"]
    ok = bool(np.isfinite(limit) and np.isfinite(mean) and abs(mean) <= limit and both <= CONTROL_MAX_BOTH)
    return {"ok": ok, "N_mean": mean, "limit": limit, "both_extinct_runs": both, "n_list": nlist}


def _d_of(name: str, res: dict | None = None) -> float | None:
    if res is not None and res.get("teach", {}).get("teach_d") is not None:
        return float(res["teach"]["teach_d"])
    m = re.search(r"-d([0-9.]+)$", name)
    return float(m.group(1)) if m else None


def _teach_cost_of(res: dict) -> float:
    return float((res.get("teach") or {}).get("teach_cost", DEFAULT_TEACH_COST))


def _evaluate(results: dict, kin_fmt: str, cost_fmt: str, ds, ctl: dict) -> dict:
    """한 세계(기본·정보 비싼)의 α·β. kin_fmt·cost_fmt는 칸 이름 틀("kin-d{d}")."""
    alpha, beta, gl = {}, {}, {}
    for d in ds:
        kin, cost = results.get(kin_fmt.format(d=d)), results.get(cost_fmt.format(d=d))
        if kin is None or cost is None:
            continue
        pk, pc = per_rep(kin["grad_by_tag_list"]), per_rep(cost["grad_by_tag_list"])
        pf = welch(pk["pf_diff"], pc["pf_diff"])
        a1 = bool(pf["pos"] and pf["diff"] >= ALPHA_MIN_DIFF)
        npos, need = _n_positive(pk["depth_diff"])
        a2 = bool(need > 0 and npos >= need)
        alpha[d] = {"pf_welch": pf, "a_pf": a1, "depth_n_pos": npos, "depth_need": need, "a_depth": a2,
                    "works": bool(a1 and a2 and ctl["ok"])}
        nstat = summarize(pk["N"])
        det = detectability(nstat["mean"], ctl["n_list"])
        gw = welch(pk["G"], pc["G"])
        beta[d] = {"N": nstat, "G": summarize(pk["G"]), "C": summarize(pk["C"]), "det": det, "G_welch": gw,
                   "lo_pos": bool(np.isfinite(nstat["lo"]) and nstat["lo"] > 0),
                   "gain": bool(np.isfinite(nstat["lo"]) and nstat["lo"] > 0 and gw["pos"] and det["ok"] and ctl["ok"])}
    works = any(a["works"] for a in alpha.values())
    gains = any(b["gain"] for b in beta.values())
    if not alpha:
        status = "broken"
    elif not works:
        status = "not_working"
    elif gains:
        status = "pass"
    else:
        status = "works_undetermined"
    return {"status": status, "alpha": alpha, "beta": beta, "works": works, "gains": gains}


def gate_alpha_beta(results: dict, pop: float = DEFAULT_POP, exp_ok: bool = False) -> dict:
    """한 바탕(태그)의 관문. results: {칸 이름: rate_zero 결과}. 칸 이름은 kin-d0.1/0.5/1.0, costonly-d*, any-d*, knower-d1.0,
    off-d1.0(음성 대조), nosoc-kin-d1.0, kin-nocost-d*, kin-meal-d*, costonly-meal-d1.0, feed-d1.0, kin-exp-d1.0, costonly-exp-d1.0.
    상태: "pass"(작동 + 이득이 비용을 넘음), "works_undetermined"(작동하나 이득을 가를 수 없음), "not_working"(작동 아님),
    "broken"(도구 깨짐: 음성 대조가 없거나 깨짐, 읽을 kin 칸이 없음).
    α(작동): 어떤 d에서 kin의 (가) 독 끼니 비율 차(A−B)가 costonly보다 Welch로 양이고 ≥ 0.01, (나) 깊이 차(B−A)가 6판 중 5판 양, 음성 대조 통과.
    β(이득): 어떤 d(near 비용 칸)에서 ① kin의 N 구간 아래끝 > 0 ② kin − costonly의 G 차가 Welch로 양 ③ 음성 대조 통과 ④ 관문 −2 비가 1 이상.
    evolve_ok: α 작동 그리고 관문 −2b(무리 크기 pop × 최대 N ÷ 17 ≥ 1). 정보 비싼 세계(exp)는 exp_ok(−1c 통과)일 때만 따로 판정한다."""
    out = {"status": "", "reason": "", "alpha": {}, "beta": {}, "control": None, "selection": None, "evolve_ok": False,
           "breakeven": {}, "readings": {}, "first_step": None, "exp": None}
    ctl_res = results.get("off-d1.0")
    if ctl_res is None:
        out.update(status="broken", reason="음성 대조(off-d1.0)가 없어 판정을 쓰지 않는다")
        return out
    ctl = _control(ctl_res)
    out["control"] = {k_: v for k_, v in ctl.items() if k_ != "n_list"}
    if not ctl["ok"]:
        out.update(status="broken", reason=f"음성 대조가 깨짐: 순이득 평균 {ctl['N_mean']:.3f} (한계 ±{ctl['limit']:.3f}), "
                                          f"양쪽 전멸 {ctl['both_extinct_runs']}판")
        return out
    main = _evaluate(results, "kin-d{d}", "costonly-d{d}", ALPHA_DS, ctl)
    out.update(alpha=main["alpha"], beta=main["beta"], status=main["status"])
    if main["status"] == "broken":
        out["reason"] = "읽을 kin·costonly 칸이 없음"
        return out
    nmeans = [b["N"]["mean"] for b in main["beta"].values() if np.isfinite(b["N"]["mean"])]
    best = max(nmeans) if nmeans else float("nan")
    out["selection"] = {"pop": pop, "best_N": best, "strength": selection_strength(pop, best) if nmeans else float("nan"),
                        "ok": selection_ok(pop, best) if nmeans else False}
    out["evolve_ok"] = bool(main["works"] and out["selection"]["ok"])
    # 읽기용 칸: 새끼당 지표와 손익분기 비용
    for name, res in results.items():
        if name == "off-d1.0" or "grad_by_tag_list" not in res:
            continue
        nb = net_per_birth(res["grad_by_tag_list"])
        pr = nb["per_rep"]
        out["readings"][name] = {"N": nb["N"], "G": nb["G"], "C": nb["C"], "pf_diff": summarize(pr["pf_diff"]),
                                 "depth_diff": summarize(pr["depth_diff"]), "know_diff": summarize(pr["know_diff"]),
                                 "grad_rate_a": summarize(pr["grad_rate_a"]), "grad_rate_b": summarize(pr["grad_rate_b"])}
        if name.startswith("kin-") and "nocost" not in name:
            d = _d_of(name, res)
            if d is not None:
                out["breakeven"][name] = breakeven(nb["G"]["mean"], nb["C"]["mean"], _teach_cost_of(res), d)
    # 첫걸음 이득: d=0.1 kin의 연속 지표가 costonly보다 0 위로 가를 수 있을 때만 만족
    k01, c01 = results.get("kin-d0.1"), results.get("costonly-d0.1")
    if k01 is not None and c01 is not None:
        pk, pc = per_rep(k01["grad_by_tag_list"]), per_rep(c01["grad_by_tag_list"])
        w3, w1 = welch(pk["depth_diff"], pc["depth_diff"]), welch(pk["depth1_diff"], pc["depth1_diff"])
        out["first_step"] = {"depth_welch": w3, "depth_s1_welch": w1, "satisfied": bool(w3["pos"] or w1["pos"])}
    if exp_ok and "kin-exp-d1.0" in results and "costonly-exp-d1.0" in results:
        out["exp"] = _evaluate(results, "kin-exp-d{d}", "costonly-exp-d{d}", ("1.0",), ctl)
    # 상태 문구
    reason = {"pass": "작동하고 이득이 비용을 넘음", "works_undetermined": "작동하나 이득이 비용을 넘는지는 가를 수 없음",
              "not_working": "작동 아님(α 못 넘음)"}
    out["reason"] = reason.get(out["status"], "")
    return out


def gate_tags(per_tag: dict, pop: float = DEFAULT_POP, exp_oks: dict | None = None) -> dict:
    """바탕마다 gate_alpha_beta를 돌려 합친다. 종료 코드: 한 바탕이라도 진화 갈래로 갈 만하면(α 작동 + −2b) 0,
    없고 도구 깨진 바탕이 있으면(또는 바탕이 없으면) 4, 그 밖에(모두 가를 수 없음·작동 아님) 3."""
    exp_oks = exp_oks or {}
    tags = {t: gate_alpha_beta(res, pop, bool(exp_oks.get(t, False))) for t, res in per_tag.items()}
    if any(g["evolve_ok"] for g in tags.values()):
        status, code = "pass", 0
    elif any(g["status"] == "broken" for g in tags.values()) or not tags:
        status, code = "broken", 4
    else:
        status, code = "undetermined", 3
    return {"status": status, "exit_code": code, "pop": pop, "tags": tags}


# ---- 관문 −1(세계 점검) ----
def _expected_par_poison(o: dict) -> float:
    """새끼당 기대 부모 독 끼니 관찰 수 = par_near_frac × juv_steps × 어른 끼니율 × 독 비율."""
    vals = (o.get("par_near_frac"), o.get("juv_steps"), o.get("meal_rate_per_agent_step"), o.get("poison_frac"))
    if any(v is None or not np.isfinite(v) for v in vals):
        return float("nan")
    return float(vals[0] * vals[1] * vals[2] * vals[3])


def _stands(o: dict, off: dict) -> dict:
    """한 세계가 선다(−1a·−1c 공통 조각): 무리 ≥ 끈의 0.9, 장부 통과."""
    ratio = _ratio(o.get("pop_mean", float("nan")), off.get("pop_mean", float("nan")))
    led = o.get("ledger_rel_residual")
    ledger_ok = bool(led is not None and np.isfinite(led) and led < LEDGER_TOL)
    return {"pop_ratio": ratio, "pop_ok": bool(np.isfinite(ratio) and ratio >= POP_RATIO_MIN - 1e-12), "ledger_ok": ledger_ok}


def _gate_world_tag(w: dict) -> dict:
    off, on, exp = w.get("off"), w.get("on"), w.get("exp")
    out = {"a": {"pass": False}, "b": {"pass": False}, "c": {"pass": False}}
    if off is None or on is None:
        out["a"]["reason"] = "끈·켠 세계 판독이 없음"
        out["b"]["reason"] = "끈·켠 세계 판독이 없음"
        return out
    s = _stands(on, off)
    r_on, r_off = on.get("reseeds") or 0, off.get("reseeds") or 0
    spike = bool(r_on > RESEED_FACTOR * r_off + RESEED_SLACK)
    out["a"] = dict(s, reseeds_on=r_on, reseeds_off=r_off, reseed_spike=spike,
                    **{"pass": bool(s["pop_ok"] and s["ledger_ok"] and not spike)})
    exp_seen = _expected_par_poison(on)
    out["b"] = {"expected_par_poison_seen": exp_seen, "par_near_frac": on.get("par_near_frac"),
                "par_near_frac_off": off.get("par_near_frac"),
                "pass": bool(np.isfinite(exp_seen) and exp_seen >= EXPECTED_PAR_POISON_MIN)}
    if exp is not None:
        e = _stands(exp, off)
        grad_ratio = _ratio(exp.get("grad_rate", float("nan")), off.get("grad_rate", float("nan")))
        out["c"] = dict(e, grad_ratio=grad_ratio,
                        **{"pass": bool(e["pop_ok"] and e["ledger_ok"] and np.isfinite(grad_ratio)
                                        and grad_ratio >= GRAD_RATIO_MIN_EXP)})
    else:
        out["c"]["reason"] = "정보 비싼 세계 판독이 없음"
    return out


def gate_world(per_tag: dict) -> dict:
    """관문 −1. per_tag: {태그: {"off": observe 결과(끔 + juv_measure), "on": 켬(juv_enabled), "exp": 정보 비싼 세계(mult 3)}}.
    −1a(세계가 선다): 켠 무리 ≥ 끈의 0.9, 장부 통과, 재주입 급증 없음. −1b(곁 머무름): par_near_frac × juv_steps × 끼니율 × 독 비율 ≥ 0.5.
    −1c(정보 비싼 세계): mult 3 무리 ≥ 끈의 0.9, 졸업률 ≥ 끈의 0.5, 장부 통과 — 못 넘으면 정보 비싼 세계 판만 닫는다(exp_ok=False).
    종료 코드: 0 통과(주 판), 3 세계 못 섬(−1a), 4 곁 부족(−1b). 모든 바탕이 넘어야 통과다(보수적)."""
    tags = {t: _gate_world_tag(w) for t, w in per_tag.items()}
    a_ok = bool(tags) and all(g["a"]["pass"] for g in tags.values())
    b_ok = bool(tags) and all(g["b"]["pass"] for g in tags.values())
    exp_ok = bool(tags) and all(g["c"]["pass"] for g in tags.values())
    if not a_ok:
        code, status = 3, "world_fails"
    elif not b_ok:
        code, status = 4, "no_company"
    else:
        code, status = 0, "pass"
    return {"status": status, "exit_code": code, "a_ok": a_ok, "b_ok": b_ok, "exp_ok": exp_ok, "tags": tags}


# ---- 선결: 사회 학습 판정 0 ----
PREREQ_DS = ("0.25", "1.0")


def _first10_diff(res: dict) -> dict:
    """사회 학습 겨루기 결과의 first10_by_tag(태그별 [독 합, 끼니 합], 반복을 합침)에서 A − B 독 비율과 비율 차 z(Welch 꼴)."""
    f = res.get("first10_by_tag") or {}
    a, b = f.get("0") or f.get(0), f.get("1") or f.get(1)
    if not a or not b or a[1] <= 0 or b[1] <= 0:
        return {"diff": float("nan"), "z": float("nan"), "pos": False}
    pa, pb = a[0] / a[1], b[0] / b[1]
    se = math.sqrt(pa * (1 - pa) / a[1] + pb * (1 - pb) / b[1])
    diff = pa - pb
    z = float(diff / se) if se > 0 else (float("inf") if diff > 0 else float("nan"))
    return {"diff": float(diff), "z": z, "pos": bool(diff > 0 and z > measure.t95(10 ** 6))}


def prereq_soc(per_tag: dict) -> dict:
    """선결. per_tag: {태그: {"outcome-d0.25": 사회 학습 겨루기 결과, "outcome-d1.0": …}}. 태그별로 d 둘 가운데 하나라도
    첫 10끼 독 비율 A − B가 Welch(비율 차 z, 끼니가 수천이라 정규)로 양이면 그 바탕은 ok. 한 바탕이라도 ok면 0(진행), 아니면 3(건너뜀)."""
    tags = {}
    for t, res in per_tag.items():
        d = {dd: _first10_diff(res[f"outcome-d{dd}"]) for dd in PREREQ_DS if f"outcome-d{dd}" in res}
        tags[t] = {"d": d, "ok": any(v["pos"] for v in d.values())}
    ok = any(v["ok"] for v in tags.values())
    return {"status": "pass" if ok else "skip", "exit_code": 0 if ok else 3, "tags": tags}
