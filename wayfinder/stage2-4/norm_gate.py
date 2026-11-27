"""규범의 씨앗(가로채기·벌주기) 판정의 순수 계산: 위반 피해율, 순수입 차와 구간, 관문 −1·−2·−2b·α·β·γ·δ, 판정 나·다,
표지 원천 유병률 맞추기, 지킨 피해의 추정.

설계 docs/superpowers/specs/2026-11-26-norms-seed-design.md "판정", 계획 docs/superpowers/plans/2026-11-26-norms-seed.md 과제 17.
시뮬레이션을 돌리지 않는다. 입력은 `norm_judge.py`가 남긴 결과 사전이고(파일 읽기는 norm_judge가 한다), 여기서는 사전만 받는다.
모든 문턱은 THRESHOLDS 한 곳에 있고 결과 문서 "규범의 씨앗 — 예측"의 문턱과 같은 값이다(실행 중에 바꾸지 않는다).

결과 사전의 꼴:
- 칸 요약(`cell_summary`): hand-dose json 하나 → 반복별 목록 {"dr": 위반 피해율, "epp": 표시당 집행 확률, "prev": 표시 유병률,
  "net": {태그: 개체 스텝당 순수입}, "alive": {태그: 끝 개체 수}, "energy": {태그: 끝 평균 에너지}} + 양쪽 전멸 판 수.
- json을 한 번 거치면 태그 키가 "0"·"1" 문자열이 되므로 둘 다 읽는다(`tagget`).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kkumteul import measure  # noqa: E402

# ---- 문턱(임시 결정, 실행 중에 바꾸지 않는다. 결과 문서 예측 절과 같은 값) ----
THRESHOLDS = {
    "alpha_min_drop": 0.07,        # α: hand-φ0.3의 위반 피해율이 base보다 이 비율 이상 낮아야(관문 −2의 검출 한계 가정)
    "monotone_frac": 5 / 6,        # α: φ 단조(0.1 ≥ 0.3 ≥ 1.0)가 반복의 이 비율 이상
    "control_max_both": 2,         # α: 음성 대조 칸의 양쪽(전부) 전멸 판이 이것을 넘으면 도구 깨짐
    "life_steps": 1560.0,          # 한 생 환산(평균 수명, 설계 "판정 −2")
    "life_income": 9.8,            # 한 생 수입(1,560스텝 × 0.0063)
    "selection_min": 1.0,          # −2: N × 순이득 ÷ 한 생 수입
    "detect_min": 1.0,             # −2: 예상 순이득 ÷ 검출 한계가 이 값 아래면 β를 안 쓴다
    "slope_frac": 0.01,            # −2b: 문턱 위 개체 비율이 이 아래면 기울기 시작판(b0 = −0.3)
    "pop_ratio_min": 0.9,          # −1a: 켠 무리 ≥ 끈(측정만) 무리의 이 비율
    "reseed_factor": 2.0,          # −1a: 재주입이 끈 세계의 이 배 + slack을 넘으면 급증
    "reseed_slack": 5,
    "ledger_tol": 1e-6,            # −1a: 장부 상대 잔차
    "take_loss": (0.08, 0.15),     # −1b: 맞은 피해 ÷ 먹이 수입
    "imp_adjust_max": 3,           # −1b: norm_impulse 조정 최대 횟수
    "prevalence": (0.005, 0.10),   # −1c: 표시 유병률
    "tau_steps": (50, 25, 12),     # −1c: 유병률이 넘으면 τ를 이 순서로
    "src_match": 0.5,              # −1c: 세 표지 원천 유병률이 서로 50% 이내(맞춤 관문: 최대 ÷ 최소 ≤ 1.5)
    "src_ratio_max": 2.0,          # δ·라: 세 원천의 끝 유병률이 2배를 넘게 갈리면 가를 수 없음
    "vis_match": 0.2,              # −1d: NORM_VIS가 0이 아닌 비율과 시야 안 표시 존재 비율의 상대 차
    "calib_tol": 0.2,              # 유병률 맞추기: 목표의 20% 안
    "calib_iters": 6,
}

# ---- 출력 한계 문구(결과 문서 머리에 과제 18이 그대로 옮긴다) ----
LABEL_HIST = "조작 확인용(엔진 위축이 있으면 규범 없이도 낮게 나온다)"
LABEL_IMPOSED = "부여된 위축 아래서"
LABEL_ARB = "이득 구조 확인이지 관습이 아니다"
LABEL_BIAS = ("개체 스텝으로 정규화해 맞아 죽어 수명이 줄어드는 몫이 빠진다 — 집행자에게 불리한 보수 편향"
              "(N_ind3 > 0은 더 믿을 만하고 ≤ 0은 수명 몫을 같이 읽는다)")
RESULTS_HEAD = ("①②와 벌의 물리·위축은 부여, ⑤⑥과 벌주기 성향은 측정, ④의 뇌 삼감과 소통·전달은 이 판 밖. "
                "위반은 엔진이 주는 떨리는 손(뇌가 못 말림)이고 표시는 정직·즉시·전원 공개의 엔진 방송이다. "
                "이 판에서는 '규범'이라는 말을 쓰지 않는다(쓰는 말: 벌주기 씨앗, 집행, 맞대응 방어).")
LABEL_CLUE = "서로 다른 두 진화 갈래의 계수기 차는 단서다(판정 문장에 못 쓴다)"


# ======================= 작은 도구 =======================
def tagget(d: dict, tag):
    """태그 키 사전에서 int·str 어느 쪽이든 읽는다. 없으면 None."""
    if d is None:
        return None
    if tag in d:
        return d[tag]
    return d.get(str(tag), d.get(int(tag)) if str(tag).lstrip("-").isdigit() else None)


def _div(a, b):
    return float(a) / float(b) if b else None


def _clean(vals) -> list[float]:
    return [float(v) for v in vals if v is not None and np.isfinite(v)]


def level_intervals(vals, scale: float = 1.0) -> dict:
    """반복별 값의 평균과 t 구간(measure.t95, df = n − 1). scale을 곱한 값(한 생 환산 등)도 같이."""
    v = np.asarray(_clean(vals), dtype=float)
    n = len(v)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "half": None, "lo": None, "hi": None}
    m = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else float("nan")
    half = measure.t95(n - 1) * sd / math.sqrt(n) if n > 1 else float("inf")
    out = {"n": n, "mean": m, "sd": sd, "half": half, "lo": m - half, "hi": m + half}
    if scale != 1.0:
        out["life"] = {"mean": m * scale, "lo": (m - half) * scale, "hi": (m + half) * scale}
    return out


def welch(a, b) -> dict:
    """평균 차 a − b와 Welch 가장자리(t95(min(n) − 1) × √(va/na + vb/nb)). diff > edge면 a가 크다, diff < −edge면 작다."""
    a, b = np.asarray(_clean(a), float), np.asarray(_clean(b), float)
    if len(a) < 2 or len(b) < 2:
        return {"diff": None, "edge": None, "sign": 0, "ok": False}
    diff = float(a.mean() - b.mean())
    edge = measure.t95(min(len(a), len(b)) - 1) * math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return {"diff": diff, "edge": edge, "sign": 1 if diff > edge else (-1 if diff < -edge else 0), "ok": True}


# ======================= 위반 피해율·순수입 =======================
def damage_rate(nt) -> float | None:
    """위반 피해율 = 때린 순간 표시가 없던 대상이 받은 피해(VIOL_DMG_RECV) ÷ 개체 스텝(ALIVE_STEPS).
    nt는 계수표 한 행(NT_NAMES 사전)이거나 태그별 사전({태그: 행}, 합쳐서 계산). full·inert·off를 같은 정의로 견준다
    (inert에서 표시 있던 대상을 친 피해 TAKE_ON_MARKED는 분자에 안 든다). 표시가 없는 세계(off, 계수기만)는 모든 피해."""
    rows = [nt] if "viol_dmg_recv" in nt else list(nt.values())
    num = sum(float(r.get("viol_dmg_recv", 0.0)) for r in rows)
    den = sum(float(r.get("alive_steps", 0.0)) for r in rows)
    return num / den if den > 0 else None


def punished_per_episode(nt) -> float | None:
    """표시당 집행 확률 = 벌 받은 닫힌 에피소드 ÷ 닫힌 에피소드(종류 0·1; 미지 2는 뺀다)."""
    rows = [nt] if "mark_epis0" in nt else list(nt.values())
    ep = sum(float(r.get("mark_epis0", 0)) + float(r.get("mark_epis1", 0)) for r in rows)
    pun = sum(float(r.get("mark_epis_pun0", 0)) + float(r.get("mark_epis_pun1", 0)) for r in rows)
    return pun / ep if ep > 0 else None


def selectivity(nt) -> float | None:
    """선택성 = (ATK_VOL_MARKED ÷ OPP_MARKED) ÷ (ATK_VOL_PLAIN ÷ OPP_PLAIN)."""
    a = _div(nt.get("atk_vol_marked", 0), nt.get("opp_marked", 0))
    b = _div(nt.get("atk_vol_plain", 0), nt.get("opp_plain", 0))
    if a is None or not b:
        return None
    return a / b


def net_income_diff(run: dict, a, b) -> float | None:
    """한 판의 개체 스텝당 순수입 차(태그 a − 태그 b). run["net_income_by_tag"]는 measure.net_income_by_tag의 태그별 값."""
    na, nb = tagget(run["net_income_by_tag"], a), tagget(run["net_income_by_tag"], b)
    if not na or not nb or na.get("net_per_step") is None or nb.get("net_per_step") is None:
        return None
    return float(na["net_per_step"]) - float(nb["net_per_step"])


def n_ind(runs: list[dict], a=1, b=2) -> dict:
    """N_ind(태그 a − b, 마당 갑이면 1 − 2; 마당 을이면 1 − 3 = N_ind3): 반복별 순수입 차의 구간과 한 생 환산(× 1,560),
    그리고 짝(태그별 끝난 생의 평균 먹은 양·평균 수명)과 보수 편향 문구."""
    diffs = [net_income_diff(r, a, b) for r in runs]
    out = level_intervals(diffs, THRESHOLDS["life_steps"])
    out["per_rep"] = diffs
    comp = {}
    for t in (a, b):
        rows = [tagget(r["net_income_by_tag"], t) or {} for r in runs]
        comp[str(t)] = {"eaten_per_life_done": _mean_or_none([x.get("eaten_per_life_done") for x in rows]),
                        "life_mean": _mean_or_none([x.get("life_mean") for x in rows])}
    out["companions"] = comp
    out["bias_note"] = LABEL_BIAS
    return out


def _mean_or_none(vals):
    v = _clean(vals)
    return float(np.mean(v)) if v else None


def n_grp(runs_x: list[dict], runs_y: list[dict], tag_x=1, tag_y=2) -> dict:
    """N_grp(전원 tag_x 마당 대 전원 tag_y 마당, 같은 씨앗 짝): 주 값은 개체 스텝당 순수입 차(N_ind와 같은 단위라
    `N_grp3 − N_ind3`(공공 몫)를 바로 읽는다), 짝으로 끝 개체 수 로그 비 ln((n_x + 0.5) ÷ (n_y + 0.5))와 끝 평균 에너지 차."""
    net, logr, en = [], [], []
    for x, y in zip(runs_x, runs_y):
        nx, ny = tagget(x["net_income_by_tag"], tag_x), tagget(y["net_income_by_tag"], tag_y)
        if nx and ny and nx.get("net_per_step") is not None and ny.get("net_per_step") is not None:
            net.append(float(nx["net_per_step"]) - float(ny["net_per_step"]))
        ax, ay = tagget(x["alive_by_tag"], tag_x) or 0, tagget(y["alive_by_tag"], tag_y) or 0
        logr.append(math.log((ax + 0.5) / (ay + 0.5)))
        ex, ey = tagget(x.get("energy_mean_by_tag", {}), tag_x), tagget(y.get("energy_mean_by_tag", {}), tag_y)
        if ex is not None and ey is not None:
            en.append(float(ex) - float(ey))
    out = level_intervals(net, THRESHOLDS["life_steps"])
    out["log_ratio"] = level_intervals(logr)
    out["energy_diff"] = level_intervals(en)
    return out


def second_order_dilemma(nind3: dict, ngrp3: dict) -> dict:
    """2차 딜레마: N_grp3 아래끝 > 0 ≥ N_ind3 아래끝이면 열려 있고 크기는 N_grp3(평균)."""
    if nind3.get("lo") is None or ngrp3.get("lo") is None:
        return {"open": None, "size": None}
    open_ = ngrp3["lo"] > 0 and nind3["lo"] <= 0
    return {"open": bool(open_), "size": ngrp3["mean"] if open_ else None}


# ======================= 칸 요약 =======================
def cell_summary(res: dict) -> dict:
    """hand-dose 결과 json(runs 목록) → 반복별 목록. 끝에 전부 죽은 판(산 개체 0)을 both_extinct로 센다."""
    reps = res.get("runs", [])
    out = {"cell": res.get("cell"), "arena": res.get("arena"), "dr": [], "epp": [], "prev": [], "runs": reps,
           "both_extinct": 0}
    for r in reps:
        out["dr"].append(damage_rate(r["norm_by_tag"]))
        out["epp"].append(punished_per_episode(r["norm_by_tag"]))
        out["prev"].append(r.get("mark_prevalence"))
        if sum(int(v) for v in r["alive_by_tag"].values()) == 0:
            out["both_extinct"] += 1
    if "neutral" in res:
        out["neutral"] = res["neutral"]
    return out


# ======================= 관문 −2·−2b =======================
def detection_limit(ctrl_diffs) -> float | None:
    """검출 한계 = 음성 대조(같은 무리 둘로 나눈) 반복별 순이득 차의 sd × t95(n − 1) ÷ √n."""
    v = _clean(ctrl_diffs)
    if len(v) < 2:
        return None
    return float(np.std(v, ddof=1)) * measure.t95(len(v) - 1) / math.sqrt(len(v))


def detectability(expected: float, ctrl_diffs, pop: float = 500.0) -> dict:
    """관문 −2: 예상 순이득(N, 개체 스텝당)과 검출 한계의 비, 진화 선택 강도 N_pop × (순이득 × 한 생) ÷ 한 생 수입."""
    lim = detection_limit(ctrl_diffs)
    ratio = abs(expected) / lim if lim else None
    sel = pop * abs(expected) * THRESHOLDS["life_steps"] / THRESHOLDS["life_income"]
    return {"expected": expected, "limit": lim, "ratio": ratio,
            "detectable": bool(ratio is not None and ratio >= THRESHOLDS["detect_min"]),
            "selection": sel, "selection_ok": bool(sel >= THRESHOLDS["selection_min"])}


def threshold_above(pre_act, w_atk=None, w_turn=None) -> dict:
    """관문 −2b: 저장된 h로 근사한 pre-act(= bout[공격] + Wout[공격]·h) > 0인 개체 비율과,
    NORM_VIS → 공격·회전 출력의 실효 가중치(Σ_k Win[c, k] × Wout[o, k]) 분포. 1% 아래면 기울기 시작판 플래그."""
    p = np.asarray(pre_act, dtype=float)
    frac = float((p > 0).mean()) if len(p) else 0.0
    out = {"n": int(len(p)), "frac_above": frac, "pre_act_median": float(np.median(p)) if len(p) else None,
           "slope_start": int(frac < THRESHOLDS["slope_frac"])}
    for name, w in (("w_vis_attack", w_atk), ("w_vis_turn", w_turn)):
        if w is not None and len(np.asarray(w)):
            a = np.asarray(w, dtype=float).ravel()
            out[name] = {"mean": float(a.mean()), "sd": float(a.std()), "p05": float(np.percentile(a, 5)),
                         "p95": float(np.percentile(a, 95)), "abs_mean": float(np.abs(a).mean())}
    return out


# ======================= 관문 α·β·γ·δ =======================
def _need(cells: dict, names) -> list[str]:
    return [n for n in names if n not in cells or not _clean(cells[n]["dr"])]


def gate_alpha(cells: dict) -> dict:
    """관문 α(작동): hand-phi0.3의 위반 피해율이 base보다 Welch로 낮고 7% 이상 낮으며, φ 단조(반복의 5/6 이상
    0.1 ≥ 0.3 ≥ 1.0), inert·off가 base ± 검출 한계(Welch로 못 가름)이고 양쪽 전멸 2판 이하 → "작동".
    음성 대조가 깨지면 "도구 깨짐", 그 밖은 "작동 아님", 칸이 없으면 "미판정"."""
    miss = _need(cells, ("base", "hand-phi0.1", "hand-phi0.3", "hand-phi1.0", "hand-phi0.3-inert", "off"))
    if miss:
        return {"verdict": "미판정", "missing": miss}
    base, h03 = cells["base"]["dr"], cells["hand-phi0.3"]["dr"]
    w = welch(base, h03)
    bm = float(np.mean(_clean(base)))
    drop = (bm - float(np.mean(_clean(h03)))) / bm if bm > 0 else 0.0
    lower = w["sign"] == 1 and drop >= THRESHOLDS["alpha_min_drop"]
    seq = list(zip(cells["hand-phi0.1"]["dr"], h03, cells["hand-phi1.0"]["dr"]))
    mono_n = sum(1 for a, b, c in seq if None not in (a, b, c) and a >= b >= c)
    mono = mono_n >= math.ceil(THRESHOLDS["monotone_frac"] * len(seq) - 1e-9)
    ctrl = {}
    broken = False
    for name in ("hand-phi0.3-inert", "off"):
        wc = welch(cells[name]["dr"], base)
        ext = cells[name]["both_extinct"]
        ok = wc["ok"] and wc["sign"] == 0 and ext <= THRESHOLDS["control_max_both"]
        ctrl[name] = {"welch": wc, "both_extinct": ext, "ok": ok}
        broken |= not ok
    if broken:
        verdict = "도구 깨짐"
    elif lower and mono:
        verdict = "작동"
    else:
        verdict = "작동 아님"
    return {"verdict": verdict, "welch_base_minus_h03": w, "drop": drop, "lower": lower,
            "monotone_reps": mono_n, "monotone": mono, "controls": ctrl}


def gate_beta(nind: dict, nind3: dict, ngrp3: dict, detect: dict | None = None) -> dict:
    """관문 β: (가) N_ind 아래끝 > 0(집행자 본인 이득 — 맞대응 방어만으로도 통과), (나) N_ind3 아래끝 > 0(3자 집행의
    본인 이득 — "벌주기 씨앗이 이득이 된다"는 (나)일 때만), (다) N_grp3 아래끝 > 0(집단 이득), 2차 딜레마 = (나) ≤ 0 이고 (다) > 0.
    관문 −2의 검출 가능성이 1 아래면 "검출 한계 아래"."""
    if detect is not None and not detect.get("detectable"):
        return {"verdict": "검출 한계 아래", "detect": detect}

    def above(d):
        return None if d.get("lo") is None else bool(d["lo"] > 0)
    ga, na, da = above(nind), above(nind3), above(ngrp3)
    dil = second_order_dilemma(nind3, ngrp3)
    if na is None:
        sentence = "미판정(N_ind3 없음)"
    elif na:
        sentence = "3자 집행 벌주기 씨앗이 집행자 본인에게 이득이 된다"
    elif dil["open"]:
        sentence = "모두가 하면 집단이 이득인데 개인이 이득이 아닌 칸(2차 딜레마가 열려 있음)"
    elif ga:
        sentence = "집행자 본인 이득은 있으나 2자 맞대응 방어까지만(3자 집행 이득은 가를 수 없음)"
    else:
        sentence = "이득을 가를 수 없음"
    return {"verdict": sentence, "ga": ga, "na": na, "da": da, "all": bool(ga and na and da),
            "dilemma": dil, "detect": detect}


def gate_gamma(cells: dict) -> dict:
    """관문 γ(첫걸음): hand-phi0.1이 base보다 Welch로 낮고 inert 칸과 가를 수 있을 때만 "만족".
    L1 칸이 L5(hand-phi0.3)와 가를 수 없으면 "기울기는 억지가 아닌 곳에서", L1이 base와 가를 수 없으면(감소 0)
    "기울기는 상습 위반자 가정에서"."""
    miss = _need(cells, ("base", "hand-phi0.1", "hand-phi0.3-inert"))
    if miss:
        return {"verdict": "미판정", "missing": miss}
    w_base = welch(cells["base"]["dr"], cells["hand-phi0.1"]["dr"])
    w_inert = welch(cells["hand-phi0.3-inert"]["dr"], cells["hand-phi0.1"]["dr"])
    ok = w_base["sign"] == 1 and w_inert["sign"] == 1
    out = {"verdict": "만족" if ok else "첫걸음 이득을 이 장치는 만족하지 않는다(검출 한계 아래)",
           "welch_base": w_base, "welch_inert": w_inert}
    if "hand-phi0.3-L1" in cells and "hand-phi0.3" in cells:
        l1 = cells["hand-phi0.3-L1"]["dr"]
        w15 = welch(l1, cells["hand-phi0.3"]["dr"])
        w10 = welch(cells["base"]["dr"], l1)
        if w10["ok"] and w10["sign"] == 0:
            out["L1"] = "기울기는 상습 위반자 가정에서 온다(L1에서 감소 0)"
        elif w15["ok"] and w15["sign"] == 0:
            out["L1"] = "기울기는 억지가 아닌 곳에서 온다(L1이 L5와 같음)"
        else:
            out["L1"] = "L1은 0과 L5 사이(가를 수 없음)"
        out["welch_L1_L5"], out["welch_base_L1"] = w15, w10
    return out


def gate_delta(cells: dict) -> dict:
    """관문 δ(임의 대 기능, ret0에서): sig·none-q의 표시당 집행 확률이 take(hand-phi0.3-ret0)보다 Welch로 낮고 N_ind도
    작으면 "기능에서 값을 얻는다", 같으면 "표시 자체가 약탈 대상". 세 원천의 끝 유병률이 2배를 넘게 갈리면
    "유병률이 달라 가를 수 없음". 칸이 없으면 "δ 미판정"(종료 코드를 0으로 두지 않는다)."""
    names = {"take": "hand-phi0.3-ret0", "sig": "hand-phi0.3-sig-ret0", "none": "hand-phi0.3-none-q-ret0"}
    miss = [v for v in names.values() if v not in cells]
    if miss:
        return {"verdict": "δ 미판정", "missing": miss}
    prev = {k_: _mean_or_none(cells[v]["prev"]) for k_, v in names.items()}
    pv = [p for p in prev.values() if p]
    if len(pv) < 3 or max(pv) / min(pv) > THRESHOLDS["src_ratio_max"]:
        return {"verdict": "유병률이 달라 가를 수 없음", "prevalence": prev, "note": LABEL_ARB}
    take = cells[names["take"]]
    res = {}
    lower_all = True
    for src in ("sig", "none"):
        c = cells[names[src]]
        we = welch(take["epp"], c["epp"])
        nt, nc = n_ind(take["runs"]), n_ind(c["runs"])
        smaller = nt["mean"] is not None and nc["mean"] is not None and nc["mean"] < nt["mean"]
        res[src] = {"welch_epp_take_minus": we, "n_ind_take": nt["mean"], "n_ind_src": nc["mean"],
                    "n_ind_smaller": smaller}
        lower_all &= we["sign"] == 1 and smaller
    if lower_all:
        verdict = "기능에서 값을 얻는다"
    elif all(res[s]["welch_epp_take_minus"]["sign"] == 0 for s in res):
        verdict = "표시 자체가 약탈 대상"
    else:
        verdict = "가를 수 없음"
    return {"verdict": verdict, "prevalence": prev, "by_src": res, "note": LABEL_ARB}


def gate_hand(cells: dict, mix_cells: dict | None = None, expected: float = 0.05 / 1560, pop: float = 500.0) -> dict:
    """판정 0 전체(표준 마당 칸 요약 cells, 섞은 마당 mix_cells): α·β·γ·δ·−2와 종료 코드.
    0 = 작동 + 이득, 3 = 작동 + 이득을 가를 수 없음 또는 작동 아님, 4 = 도구 깨짐, 5 = 미판정(원자료 없음, δ 포함)."""
    a = gate_alpha(cells)
    ctrl = cells.get("base", {}).get("neutral")
    det = detectability(expected, ctrl, pop) if ctrl else None
    out = {"alpha": a, "detect": det}
    if a["verdict"] in ("미판정",):
        out.update(case="미판정", code=5)
        return out
    nind = n_ind(cells["hand-phi0.1"]["runs"]) if "hand-phi0.1" in cells else {}
    nind = n_ind(cells["hand-phi0.3"]["runs"]) if "hand-phi0.3" in cells else nind
    nind3 = n_ind(cells["hand3-phi0.3"]["runs"], 1, 3) if "hand3-phi0.3" in cells else {}
    ngrp = n_grp(cells["grp-1"]["runs"], cells["grp-2"]["runs"], 1, 2) if {"grp-1", "grp-2"} <= set(cells) else {}
    ngrp3 = n_grp(cells["grp-1"]["runs"], cells["grp-3"]["runs"], 1, 3) if {"grp-1", "grp-3"} <= set(cells) else {}
    out.update(n_ind=nind, n_ind3=nind3, n_grp=ngrp, n_grp3=ngrp3)
    out["beta"] = gate_beta(nind, nind3, ngrp3, det)
    out["gamma"] = gate_gamma(cells)
    out["delta"] = gate_delta(cells)
    if mix_cells:
        out["mix"] = {"n_ind": n_ind(mix_cells["hand-phi0.3"]["runs"]) if "hand-phi0.3" in mix_cells else None,
                      "n_ind3": n_ind(mix_cells["hand3-phi0.3"]["runs"], 1, 3) if "hand3-phi0.3" in mix_cells else None}
    if a["verdict"] == "도구 깨짐":
        out.update(case="도구 깨짐", code=4)
    elif a["verdict"] != "작동":
        out.update(case="작동 아님", code=3)
    elif out["delta"]["verdict"] == "δ 미판정":
        out.update(case="δ 미판정", code=5)
    elif out["beta"].get("ga") or out["beta"].get("na"):
        out.update(case="작동 + 이득", code=0)
    else:
        out.update(case="작동 + 이득을 가를 수 없음", code=3)
    return out


# ======================= 판정 −1(세계 점검) =======================
def gate_world(obs: dict, arena_check: dict | None = None) -> dict:
    """관문 −1. obs = {이름: [씨앗별 observe 결과]}, 이름 ∈ meas·inert·full·noimp·sig·none. 결과 사전 키는
    pop_mean·pop_min·reseeds·ledger_rel·theta_t·take_loss_frac·prevalence·vis_nonzero_frac·vis_present_frac.
    −1a(켠 무리 ≥ 끈 무리 × 0.9, 재주입 급증·멸종 없음, 장부), −1b(take_loss_frac 0.08~0.15), −1c(유병률 0.5~10%,
    세 원천이 서로 50% 이내), −1d(NORM_VIS 0 아닌 비율 ≈ 시야 안 표시 존재 비율), −1e(거주민 공격 0).
    종료 코드: 0 통과, 3 세계 못 섬(−1a·−1c·−1d·−1e), 4 딜레마 크기 밖(−1b)."""
    T = THRESHOLDS

    def mean(name, key):
        return _mean_or_none([o.get(key) for o in obs.get(name, [])])
    out = {}
    # −1a
    pm, pf = mean("meas", "pop_mean"), mean("full", "pop_mean")
    rs_m, rs_f = mean("meas", "reseeds"), mean("full", "reseeds")
    led = [o.get("ledger_rel") for v in obs.values() for o in v]
    extinct = any((o.get("pop_min") or 0) == 0 for v in obs.values() for o in v)
    a_ok = (pm is not None and pf is not None and pf >= T["pop_ratio_min"] * pm and not extinct
            and (rs_f or 0) <= T["reseed_factor"] * (rs_m or 0) + T["reseed_slack"]
            and all(x is not None and abs(x) < T["ledger_tol"] for x in led))
    out["-1a"] = {"ok": bool(a_ok), "pop_meas": pm, "pop_full": pf, "reseeds_meas": rs_m, "reseeds_full": rs_f,
                  "extinct": extinct, "ledger_max": max((abs(x) for x in led if x is not None), default=None)}
    # −1b
    tl, th = mean("full", "take_loss_frac"), mean("full", "theta_t")
    lo, hi = T["take_loss"]
    b_ok = tl is not None and lo <= tl <= hi
    rec_b = None if b_ok else ("norm_impulse를 올린다" if tl is not None and tl < lo else "norm_impulse를 내린다")
    out["-1b"] = {"ok": bool(b_ok), "theta_t": th, "take_loss_frac": tl, "advice": rec_b,
                  "max_adjust": T["imp_adjust_max"]}
    # −1c
    prev = mean("full", "prevalence")
    plo, phi = T["prevalence"]
    src = {s: mean(s, "prevalence") for s in ("full", "sig", "none")}
    sv = [v for v in src.values() if v]
    src_ok = len(sv) == 3 and max(sv) / min(sv) <= 1.0 + T["src_match"]
    c_ok = prev is not None and plo <= prev <= phi
    out["-1c"] = {"ok": bool(c_ok), "prevalence": prev, "src_prevalence": src, "src_match": bool(src_ok),
                  "advice": None if c_ok else "τ를 50 → 25 → 12 순으로 줄인다(그래도 넘으면 norm_impulse를 내린다)",
                  "delta_readable": bool(src_ok)}
    # −1d
    nz, pr = mean("full", "vis_nonzero_frac"), mean("full", "vis_present_frac")
    d_ok = nz is not None and pr is not None and pr > 0 and abs(nz - pr) <= T["vis_match"] * pr
    out["-1d"] = {"ok": bool(d_ok), "vis_nonzero_frac": nz, "vis_present_frac": pr}
    # −1e
    e_ok = bool(arena_check) and arena_check.get("take", 1) == 0 and arena_check.get("pun", 1) == 0
    out["-1e"] = {"ok": e_ok, "arena_check": arena_check}
    if not (a_ok and c_ok and d_ok and e_ok):
        code = 3
    elif not b_ok:
        code = 4
    else:
        code = 0
    out["code"] = code
    return out


# ======================= 판정 다(바닥)·라 =======================
def bottom_rates(nt: dict, kin_opp_frac: float | None = None) -> dict:
    """계수표 한 행(합친 행)에서 p2 = PUN_2ND ÷ OPP_SELF, p3 = (PUN_3_KIN + PUN_3_STR) ÷ OPP_OTHER,
    친족 부풀림 ⒜(닮음 구간) = (PUN_3_KIN ÷ (PUN_3_KIN + PUN_3_STR)) ÷ (3자 기회의 친족 비율: OPP_OTHER_SIM2·3 ÷ OPP_OTHER),
    ⒝(계보) = (PUN_3_LINE ÷ 3자 벌) ÷ (OPP_OTHER_LINE ÷ OPP_OTHER), 2자 대 3자 몫 = 3자 벌 ÷ (2자 + 3자 벌)."""
    p3n = float(nt.get("pun_3_kin", 0)) + float(nt.get("pun_3_str", 0))
    opp_o = float(nt.get("opp_other", 0))
    kin_frac = kin_opp_frac if kin_opp_frac is not None else _div(
        float(nt.get("opp_other_sim2", 0)) + float(nt.get("opp_other_sim3", 0)), opp_o)
    line_frac = _div(nt.get("opp_other_line", 0), opp_o)
    pk = _div(nt.get("pun_3_kin", 0), p3n)
    pl = _div(nt.get("pun_3_line", 0), p3n)
    tot = float(nt.get("pun_2nd", 0)) + p3n
    return {"p2": _div(nt.get("pun_2nd", 0), nt.get("opp_self", 0)), "p3": _div(p3n, opp_o),
            "kin_infl_sim": pk / kin_frac if pk is not None and kin_frac else None,
            "kin_infl_line": pl / line_frac if pl is not None and line_frac else None,
            "share3": _div(p3n, tot)}


def sum_rows(norm_by_tag: dict, tags=None) -> dict:
    """태그별 계수표를 한 행으로 더한다(tags를 주면 그 태그만)."""
    out: dict = {}
    for t, row in norm_by_tag.items():
        if tags is not None and int(t) not in tags:
            continue
        for key, v in row.items():
            out[key] = out.get(key, 0.0) + float(v)
    return out


def bottom(std_runs: list[dict], mix_runs: list[dict], tau1_runs: list[dict] | None = None) -> dict:
    """판정 다: 같은 끝 무리의 표준 마당·섞은 마당(같은 씨앗 짝) 반복별 p2·p3·친족 부풀림·3자 몫과 그 차(표준 − 섞음).
    친족 부풀림은 닮음 ⒜와 계보 ⒝가 같은 쪽(둘 다 > 1 또는 둘 다 ≤ 1)일 때만 쓴다. 평판(늦은 집행 비율:
    경과 10~τ ÷ 0~10)은 반복별 late_frac(bottom 명령이 잰다)으로, tau1 칸이 있으면 그것과 견준다."""
    def per(runs):
        return [bottom_rates(sum_rows(r["norm_by_tag"])) for r in runs]
    s, m = per(std_runs), per(mix_runs)
    out = {}
    for key in ("p2", "p3", "kin_infl_sim", "kin_infl_line", "share3"):
        sv, mv = [x[key] for x in s], [x[key] for x in m]
        diff = [a - b for a, b in zip(sv, mv) if a is not None and b is not None]
        out[key] = {"std": level_intervals(sv), "mix": level_intervals(mv), "std_minus_mix": level_intervals(diff)}
    ks, kl = out["kin_infl_sim"]["std"]["mean"], out["kin_infl_line"]["std"]["mean"]
    agree = ks is not None and kl is not None and ((ks > 1) == (kl > 1))
    out["kin_infl_usable"] = bool(agree)
    late = [r.get("late_frac") for r in std_runs]
    out["late_frac"] = level_intervals(late)
    if tau1_runs:
        out["late_frac_tau1"] = level_intervals([r.get("late_frac") for r in tau1_runs])
    out["note"] = LABEL_CLUE
    return out


def arbitrary(take_runs: list[dict], sig_runs: list[dict]) -> dict:
    """판정 라: 임의-관습 끝 무리(sig) 대 규범 켬(take)의 표시당 집행 확률. 끝 유병률이 2배 넘게 갈리면 가를 수 없음."""
    e_t = [punished_per_episode(r["norm_by_tag"]) for r in take_runs]
    e_s = [punished_per_episode(r["norm_by_tag"]) for r in sig_runs]
    p_t, p_s = _mean_or_none([r.get("mark_prevalence") for r in take_runs]), _mean_or_none(
        [r.get("mark_prevalence") for r in sig_runs])
    if not p_t or not p_s or max(p_t, p_s) / min(p_t, p_s) > THRESHOLDS["src_ratio_max"]:
        return {"verdict": "유병률이 달라 가를 수 없음", "prev": [p_t, p_s], "note": LABEL_ARB}
    w = welch(e_t, e_s)
    return {"verdict": {1: "임의 표지의 집행 확률이 더 낮다", 0: "가를 수 없음", -1: "임의 표지의 집행 확률이 더 높다"}[w["sign"]],
            "welch_take_minus_sig": w, "prev": [p_t, p_s], "note": LABEL_ARB}


# ======================= 판정 나(주) =======================
def main_judgment(on: dict, nulls: dict, noimp: dict | None, knock: dict | None = None,
                  other_seed: dict | None = None) -> dict:
    """주 판정 나. on/nulls[이름]/noimp = block-compete 결과 사전 하나(같은 씨앗)의 요약:
    {"n_ind": level, "n_ind3": level, "n_grp3": level, "selectivity": level, "r_CA": level, "r_CB": level,
     "min_detectable": float}. knock = {"on": 감각 막기 r 평균, "inert": 같은 값, "limit": 한계}.
    other_seed = 다른 씨앗의 같은 요약(두 씨앗이 같은 쪽인지).
    "그 수준의 벌주기 씨앗이 개인에게 이득"은 모든 조건이 설 때만. N_grp3만 0 위면 2차 딜레마 문장을 따로 쓴다(OR 아님).
    종료 코드: 0 = 한 바탕이라도 통과, 3 = 모두 가를 수 없음, 4 = 깨짐(음성 대조가 0 ± 한계를 벗어남), 5 = 미판정."""
    if not on:
        return {"verdict": "미판정(켠 끝 무리 결과 없음)", "code": 5}

    def lo_pos(d, key, thr=0.0):
        v = (d or {}).get(key) or {}
        return v.get("lo") is not None and v["lo"] > thr

    lvl3 = lo_pos(on, "n_ind3")
    lvl2 = lo_pos(on, "n_ind")
    sel = lo_pos(on, "selectivity", 1.0)
    null_ok = True
    for name, d in (nulls or {}).items():
        for key in ("n_ind", "n_ind3"):
            if lo_pos(d, key):         # null 끝 무리도 0 위면 켠 무리와 구분 안 됨
                if not ((on.get(key) or {}).get("lo", -1) > ((d.get(key) or {}).get("hi") or 0)):
                    null_ok = False
    neg_ok = None
    if noimp:
        v = noimp.get("n_ind") or {}
        neg_ok = v.get("lo") is not None and v["lo"] <= 0 <= v["hi"]
    knock_ok = None
    if knock and knock.get("on") is not None and knock.get("inert") is not None:
        knock_ok = abs(knock["on"] - knock["inert"]) > (knock.get("limit") or 0)
    seeds_ok = True
    if other_seed:
        seeds_ok = (lo_pos(other_seed, "n_ind3") == lvl3) and (lo_pos(other_seed, "n_ind") == lvl2)
    conds = {"n_ind3_or_n_ind": bool(lvl3 or lvl2), "selectivity": sel, "knock_vs_inert": knock_ok,
             "null_distinct": null_ok, "negative_control": neg_ok, "seeds_agree": seeds_ok}
    if neg_ok is False:
        return {"verdict": "깨짐(음성 대조가 0 ± 한계 밖)", "conditions": conds, "code": 4}
    all_ok = all(v is True for v in conds.values())
    out = {"conditions": conds}
    if all_ok:
        level = "3자 집행" if lvl3 else "2자 맞대응"
        out["verdict"] = f"그 수준({level})의 벌주기 씨앗이 개인에게 이득"
        out["code"] = 0
    else:
        out["verdict"] = "가를 수 없음"
        out["code"] = 3
    if lo_pos(on, "n_grp3") and not lvl3:
        out["dilemma"] = "모두가 하면 집단이 이득인데 개인이 이득이 아닌 칸(2차 딜레마가 열려 있음)"
    r = on.get("r_CA") or {}
    if r.get("mean") is not None and r["mean"] >= -(on.get("min_detectable") or 0):
        out["note_win"] = "A쪽이 비용을 안 내 유리하므로 이기거나 비기면 '벌주기 값이 이득보다 크거나 같다'까지만"
    return out


def combine_tags(runs: list[dict], key: str, tag) -> list:
    """반복별 결과에서 태그 하나의 값 목록(run[key][tag])."""
    return [tagget(r.get(key, {}), tag) for r in runs]


# ======================= 표지 원천 유병률 맞추기 =======================
def calibrate_sig_theta(measure_fn, target: float, lo: float, hi: float, iters: int | None = None) -> dict:
    """sig 원천 문턱 θ를 이분 탐색한다(θ가 크면 유병률이 내려가는 단조를 가정). 목표의 20% 안에 들면 멈춘다.
    measure_fn(θ) → 측정 유병률. 돌려준다: {"sig_theta", "measured", "tries": [(θ, 유병률)...], "ok"}."""
    iters = THRESHOLDS["calib_iters"] if iters is None else iters
    tries = []
    best = None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        p = float(measure_fn(mid))
        tries.append((mid, p))
        if best is None or abs(p - target) < abs(best[1] - target):
            best = (mid, p)
        if abs(p - target) <= THRESHOLDS["calib_tol"] * target:
            break
        if p > target:
            lo = mid
        else:
            hi = mid
    return {"sig_theta": best[0], "measured": best[1], "tries": tries,
            "ok": bool(abs(best[1] - target) <= THRESHOLDS["calib_tol"] * target)}


def noise_q0(prevalence: float, tau: int) -> float:
    """none 원천의 잡음 확률 시작값: 표시 하나가 τ + 1스텝 지속(겹침 무시)이라 q0 = P ÷ (τ + 1)."""
    return float(prevalence) / (tau + 1)


def noise_q_fix(q0: float, target: float, measured: float) -> float:
    """한 번 확인 측정 뒤 비례로 고친다(q1 = q0 × 목표 ÷ 측정). 측정이 0이면 q0 그대로."""
    return q0 * target / measured if measured > 0 else q0


# ======================= 지킨 피해의 추정 =======================
def saved_est(hist, take_dmg_recv: float, take: float) -> dict:
    """norm_hist[행, 구간, (닿은 개체 스텝, 가로챔)]: 행 0~2 = 집행자 역할별 벌 받은 개체, 행 3 = 표시됐으나 안 벌받음.
    역할 r의 지킨 피해 추정 = Σ_구간 (가로채기율_3 − 가로채기율_r) × 닿은 스텝_r × 평균 피해(TAKE_DMG_RECV ÷ TAKE).
    조작 확인용(엔진 위축이 있으면 규범 없이도 낮게 나온다 — LABEL_HIST)."""
    h = np.asarray(hist, dtype=float)
    dmg = take_dmg_recv / take if take > 0 else 0.0
    out = {"mean_dmg": dmg, "label": LABEL_HIST}
    for r, role in enumerate(("self", "kin", "stranger")):
        tot = 0.0
        for b in range(h.shape[1]):
            if h[3, b, 0] > 0 and h[r, b, 0] > 0:
                tot += (h[3, b, 1] / h[3, b, 0] - h[r, b, 1] / h[r, b, 0]) * h[r, b, 0] * dmg
        out[role] = tot
    return out
