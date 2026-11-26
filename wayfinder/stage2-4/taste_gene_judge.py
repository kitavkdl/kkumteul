"""맛 배움 속도 유전자 판정 나: 진화가 남긴 배움이 값을 하나(설계 2026-11-07-individual-learning-design.md).

rate-zero (판정 나):
    갈래 끝 체크포인트의 산 생물로 두 벌을 만든다. A(tag 0)는 배움을 없앤 벌 — 맛 배움 속도 유전자를
    taste_g = −taste_rate로 두어 실제 배움 속도가 0(기억이 바뀌지 않음, 비용도 0). B는 원래 벌.
    체크포인트 설정 그대로의 겨루기 세계(measure.assay_sides, 변이 꺼짐이라 taste_g는 그대로 물려진다)에서 겨루고,
    같은 무리를 둘로 나눈 중립 겨루기로 잡음을 잰다(knockout.split과 같은 꼴: r, 중립, min_detectable, verdict).
    A가 지면(r < 0이고 판정이 B 쪽) 진화가 남긴 배움이 값을 한다.
    치우침: A는 배움 값(taste_cost × 속도)을 치르지 않는다 — 비용만큼 A에 유리하다. 그래서 A가 지면 보수적인 결론이고,
    A가 이기거나 비기면 "배움 값이 이득보다 크거나 같다"까지만 말할 수 있다.
    --gene eta: 같은 꼴로 헤브 학습률 유전자 eta를 0으로 둔 벌(기존 평생 학습 규칙이 살아남았나, 계획 1순위).
    --gene soc: 같은 꼴로 보고 배우기 속도 유전자 soc_g를 −soc_rate로 둔 벌(실제 속도 0, 비용도 0).
      겨루기 설정을 바꾸는 옵션(설계 4절 판정 0, 용량 반응): --set KEY=VAL(여러 번, 체크포인트 설정 위에 덮어쓴다 — 예: soc_taste_enabled=true
      soc_taste_mode=blind soc_cost=0), --soc-d D(B쪽을 모두 실제 속도 D로), --no-individual(두 쪽 모두 개인 학습 속도 0: 사회 몫만),
      --no-neutral(중립 겨루기를 건너뜀 — 그러면 min_detectable도 안 낸다).

observe (단서, 판정 아님):
    끼니 계수기는 체크포인트에 저장되지 않으므로 체크포인트 설정 그대로 --steps(기본 3,000) 더 돌려 센다.
    무리 크기, 실제 배움 속도 평균·사분위(돌리기 전), 그동안 먹힌 알 가운데 독 비율(세계 계수기 C_POISONED / C_EATEN 차이),
    끝에 산 개체의 첫 10끼 독 비율(n_poison10 / min(n_meal, 10) 합 — 읽은 뒤 태어난 개체는 태어난 뒤부터, 그 전부터 산
    개체는 읽은 뒤부터 센 끼니). 독 비율 0.5 근처면 가려 먹지 못하는 것이다.
    보고 배우기를 켠 설정이면: soc_rate(실제 속도 평균·사분위, 돌리기 전)와, 돌린 뒤 산 개체의 본 끼니 수(n_seen·n_seen_poison·
    n_seen_kin·seen_age_sum 합)와 비율(seen_poison_frac, seen_kin_frac, 산 개체 1,000스텝당 본 끼니 seen_per_1000_steps — 근사).

gate (판정 0의 관문, 순수 계산 + 파일 읽기):
    rate-zero가 남긴 j-dose-<이름>-<태그>.json들을 읽어 바탕(태그)마다 통과 / 검출 불능 / 도구 깨짐을 가른다(규칙은 gate_tag 문서).
    종료 코드: 0 통과(한 바탕이라도), 3 검출 불능(모든 바탕), 4 통과 없이 도구 깨진 바탕이 있음.

부모의 가르침(설계 2026-11-26-parent-teaching-design.md 6절, 순수 계산은 teach_gate.py):
    rate-zero --gene show|feed: A(tag 0)는 보여 주기 유전자 show_g(feed는 feed_g)를 0으로 둔 벌, B는 원래 벌.
      --teach-d D: B쪽 show_g(feed는 feed_g)를 모두 D로(용량 반응, D ∈ {0.1, 0.5, 1.0}). --soc-none: 양쪽 soc_g = −soc_rate(보고 배우기 없음).
      결과에 태그별 졸업표(grad_by_tag_list)·적용한 옵션(teach_d, soc_none, teach{모드·비용 …})이 들어 있다.
      --set teach_enabled=false 같은 음성 대조·--set teach_cost_when=meal 같은 변형은 --set으로 준다(뒤의 --set이 앞을 덮어쓴다).
    observe --steps N: juv_enabled나 juv_measure인 설정이면 창 시작에 졸업표를 0으로 하고, 창 동안 졸업률·곁 머무름(par_near_frac)·
      본 부모 끼니·무리 평균·재주입·장부·끼니율을 내고, 창 시작의 show_g·feed_g 평균·분위수를 낸다.
    gate-world --dir D --tags a b: 관문 −1(w-<off|on|exp>-<태그>.json). 종료 코드 0 통과, 3 세계 못 섬(−1a), 4 곁 부족(−1b). exp_ok = −1c.
    gate-teach --dir D --tags a b [--pop 500]: 관문 −2·−2b·α·β·손익분기 비용(j-dose-<칸>-<태그>.json). 0 진화 갈래로 갈 만함, 3 가를 수 없음, 4 도구 깨짐.
      D에 gate-world.json이 있고 exp_ok가 참이면 정보 비싼 세계 칸(kin-exp-d1.0 …)도 따로 판정한다.
    prereq-soc --dir D --tags a b: 사회 학습 판정 0(j-dose-outcome-d0.25·d1.0-<태그>.json)의 직접 지표 확인. 0 진행, 3 건너뜀.

    PYTHONPATH=. python wayfinder/stage2-4/taste_gene_judge.py rate-zero <체크포인트.npz> [--gene taste|eta|soc|show|feed]
        [--n 400] [--steps 8000] [--reps 6] [--seed 6000] [--set KEY=VAL ...] [--soc-d D] [--no-individual] [--no-neutral]
        [--teach-d D] [--soc-none] [--json <출력.json>]
    PYTHONPATH=. python wayfinder/stage2-4/taste_gene_judge.py observe <체크포인트.npz> [--steps 3000] [--json <출력.json>]
    PYTHONPATH=. python wayfinder/stage2-4/taste_gene_judge.py gate --dir <결과 폴더> --tags a b [--json <출력.json>]
    PYTHONPATH=. python wayfinder/stage2-4/taste_gene_judge.py gate-world|gate-teach|prereq-soc --dir <결과 폴더> --tags a b [--json <출력.json>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))      # teach_gate(같은 폴더의 순수 계산)

from kkumteul import checkpoint as ckpt  # noqa: E402
from kkumteul import kernels as k  # noqa: E402
from kkumteul import knockout, measure  # noqa: E402
from kkumteul.config import FIRST_MEALS  # noqa: E402
import teach_gate  # noqa: E402

GENES = ("taste", "eta", "soc", "show", "feed")
# --set이 이 키를 직접 건드리면(음성 대조로 장치를 끄는 판 포함) 그 유전자의 "장치 켜짐" 검사를 건너뛴다
TOUCH_KEYS = {"soc": "soc_taste_enabled", "show": "teach_enabled", "feed": "feed_enabled"}
TEACH_OFF_MSG = "부모의 가르침 장치가 꺼진 체크포인트다: {} 유전자가 쓰이지 않는다(--set {}=true로 켠다)"


def zeroed_side(side: dict, gene: str = "taste", check: bool = True) -> dict:
    """배움을 없앤 한 벌(유전체 사본). taste: taste_g = −taste_rate(실제 속도 0), eta: eta = 0,
    soc: soc_g = −soc_rate(보고 배우기 실제 속도 0), show: show_g = 0(보여 주기 0), feed: feed_g = 0(먹이 대주기 0).
    check=False면 장치 켜짐 검사를 건너뛴다(음성 대조에서 장치를 끄는 설정을 줄 때)."""
    cfg = side["cfg"]
    g = {kk: v.copy() for kk, v in side["genomes"].items()}
    if gene == "taste":
        if not cfg.taste_enabled:
            raise ValueError("맛 기억이 꺼진 체크포인트다: 배움 속도를 0으로 해도 달라지는 것이 없다")
        if not cfg.taste_gene_enabled:
            raise ValueError("맛 배움 속도 유전자가 꺼진 체크포인트다: taste_g가 쓰이지 않는다")
        g["taste_g"][:] = -cfg.taste_rate
    elif gene == "eta":
        g["eta"][:] = 0.0
    elif gene == "soc":
        if check and not cfg.soc_taste_enabled:
            raise ValueError("보고 배우기가 꺼진 체크포인트다: soc_g가 쓰이지 않는다(--set soc_taste_enabled=true로 켠다)")
        g["soc_g"][:] = -cfg.soc_rate
    elif gene == "show":
        if check and not (cfg.juv_enabled and cfg.teach_enabled):
            raise ValueError(TEACH_OFF_MSG.format("show_g", "juv_enabled=true --set teach_enabled"))
        g["show_g"][:] = 0.0
    elif gene == "feed":
        if check and not (cfg.juv_enabled and cfg.feed_enabled):
            raise ValueError(TEACH_OFF_MSG.format("feed_g", "juv_enabled=true --set feed_enabled"))
        g["feed_g"][:] = 0.0
    else:
        raise ValueError(f"모르는 유전자: {gene} ({', '.join(GENES)} 중 하나)")
    return dict(side, genomes=g, path=f"{side['path']} ({gene} 배움 0)")


def _with_cfg(side: dict, cfg) -> dict:
    return dict(side, cfg=cfg, genomes={kk: v.copy() for kk, v in side["genomes"].items()})


def apply_options(a: dict, b: dict, sets: list[str] | None = None, soc_d: float | None = None,
                  no_individual: bool = False, teach_d: float | None = None, teach_gene: str = "show",
                  soc_none: bool = False) -> tuple[dict, dict]:
    """두 쪽에 겨루기 옵션을 적용한 사본 한 쌍을 돌려준다(원본은 그대로).
    sets: "이름=값" 문자열들을 두 쪽 설정에 덮어쓴다(Config.with_overrides). soc_d: B쪽 soc_g = D − soc_rate(실제 속도 D;
    soc_rate는 sets를 적용한 뒤의 값). no_individual: 두 쪽 모두 taste_g = −taste_rate(개인 학습 0, 사회 몫만 본다).
    teach_d: B쪽 show_g(teach_gene="feed"면 feed_g)를 모두 D로. soc_none: 두 쪽 모두 soc_g = −soc_rate(보고 배우기 없음, 가르침만 본다).
    A쪽을 0으로 만드는 일(zeroed_side)은 이 함수 밖에서, sets를 적용한 뒤에 해야 soc_rate가 어긋나지 않는다(build_pair)."""
    a2 = _with_cfg(a, a["cfg"].with_overrides(sets) if sets else a["cfg"])
    b2 = _with_cfg(b, b["cfg"].with_overrides(sets) if sets else b["cfg"])
    if soc_d is not None:
        b2["genomes"]["soc_g"][:] = soc_d - b2["cfg"].soc_rate
    if no_individual:
        for s in (a2, b2):
            if not s["cfg"].taste_gene_enabled:
                raise ValueError("맛 배움 속도 유전자가 꺼진 설정이라 --no-individual이 뜻이 없다(taste_g가 쓰이지 않는다)")
            s["genomes"]["taste_g"][:] = -s["cfg"].taste_rate
    if soc_none:
        for s in (a2, b2):
            s["genomes"]["soc_g"][:] = -s["cfg"].soc_rate
    if teach_d is not None:
        if teach_gene not in ("show", "feed"):
            raise ValueError(f"--teach-d는 --gene show|feed에서만 쓴다: {teach_gene}")
        b2["genomes"][f"{teach_gene}_g"][:] = teach_d
    return a2, b2


def build_pair(side: dict, gene: str, sets: list[str] | None = None, soc_d: float | None = None,
               no_individual: bool = False, teach_d: float | None = None, soc_none: bool = False) -> tuple[dict, dict]:
    """겨루기 한 쌍 (A: 배움 0 벌, B: 원래 벌). sets를 먼저 설정에 적용한 뒤 A를 0으로 만들고 나머지 옵션을 적용한다.
    soc·show·feed는 --set이 그 장치 켜짐 키(TOUCH_KEYS)를 직접 건드릴 때(음성 대조로 끄는 판 포함) 켜짐 검사를 건너뛴다."""
    cfg = side["cfg"].with_overrides(sets) if sets else side["cfg"]
    base = _with_cfg(side, cfg)
    key = TOUCH_KEYS.get(gene)
    touched = bool(key) and any(x.split("=", 1)[0].strip() == key for x in (sets or []))
    a = zeroed_side(base, gene, check=not touched)
    return apply_options(a, base, None, soc_d, no_individual, teach_d, gene, soc_none)


def _teach_info(cfg, d: float | None) -> dict:
    """결과에 적는 가르침 옵션(관문의 비용 노출·손익분기 셈이 읽는다)."""
    return {"teach_d": d, "teach_enabled": bool(cfg.teach_enabled), "teach_mode": cfg.teach_mode,
            "teach_cost": float(cfg.teach_cost), "teach_cost_when": cfg.teach_cost_when, "teach_gain": float(cfg.teach_gain),
            "feed_enabled": bool(cfg.feed_enabled), "feed_rate": float(cfg.feed_rate), "juv_enabled": bool(cfg.juv_enabled),
            "juv_steps": int(cfg.juv_steps), "juv_poison_mult": float(cfg.juv_poison_mult)}


def rate_zero(path: Path, gene: str = "taste", n: int = 400, steps: int = 8000, reps: int = 6,
              seed: int = 6000, sets: list[str] | None = None, soc_d: float | None = None,
              no_individual: bool = False, neutral: bool = True, teach_d: float | None = None,
              soc_none: bool = False) -> dict:
    side = measure.load_side(path)
    a, b = build_pair(side, gene, sets, soc_d, no_individual, teach_d, soc_none)
    res = measure.assay_sides(a, b, n, steps, reps, seed, neutral=False)
    res["gene"] = gene
    res["sets"] = list(sets or [])
    res["soc_d"] = soc_d
    res["no_individual"] = bool(no_individual)
    res["teach_d"] = teach_d
    res["soc_none"] = bool(soc_none)
    res["teach"] = _teach_info(b["cfg"], teach_d)
    if neutral:
        # 중립 겨루기는 B쪽(옵션을 적용한 무리·설정)을 둘로 갈라 잰다: 옵션 아래에서의 측정 잡음
        res["neutral_r_per_1000"] = knockout._neutral(b, b["cfg"], n, steps, reps, seed)
        nsd = res["neutral_r_per_1000"].get("sd", float("nan"))
        res["min_detectable_r_per_1000"] = float(measure.t95(reps - 1) * nsd / np.sqrt(reps)) if reps > 1 else None
    res["winner"] = measure.winner(res)
    res["verdict"] = measure.verdict(res, f"배움 0({gene}) 쪽이", "원래 쪽이")
    return res


def observe(path: Path, steps: int = 3000) -> dict:
    world, _ = ckpt.load(Path(path))
    s, cfg = world.s, world.cfg
    a = s.alive
    out = {"ckpt": str(path), "step": world.step_no, "pop": int(a.sum()), "steps": steps,
           "kind_flip_period": cfg.kind_flip_period, "taste_gene_enabled": bool(cfg.taste_gene_enabled)}
    if not a.any():
        return out
    if cfg.taste_enabled:
        rate = np.clip(cfg.taste_rate + (s.taste_g[a] if cfg.taste_gene_enabled else 0.0), 0.0, 1.0)
        out["taste_rate"] = {"mean": float(np.mean(rate)), "q25": float(np.quantile(rate, 0.25)),
                             "median": float(np.median(rate)), "q75": float(np.quantile(rate, 0.75))}
    out["eta_mean"] = float(s.eta[a].mean())
    if cfg.soc_taste_enabled:
        srate = np.clip(cfg.soc_rate + s.soc_g[a], 0.0, 1.0)
        out["soc_rate"] = {"mean": float(np.mean(srate)), "q25": float(np.quantile(srate, 0.25)),
                           "median": float(np.median(srate)), "q75": float(np.quantile(srate, 0.75))}
    juv_on = bool(cfg.juv_enabled or cfg.juv_measure)
    if cfg.teach_enabled:       # 유전자 분포는 돌리기 전(창 동안에도 변이가 돌 수 있어서, taste_rate·soc_rate와 같은 규칙)
        out["show_g"] = _dist(s.show_g[a])
    if cfg.feed_enabled:
        out["feed_g"] = _dist(s.feed_g[a])
    e0, p0 = int(s.counters[k.C_EATEN]), int(s.counters[k.C_POISONED])
    if juv_on:
        # 창 시작에 졸업표를 0으로: 이어 돌린 프로세스는 재개 시점부터만 쌓이므로 5,000스텝쯤 새로 센다(설계 5절)
        world.reset_grad_table()
        r0 = int(s.counters[k.C_RESEED])
        pops, agent_steps, done = [], 0.0, 0
        while done < steps:
            n = min(WINDOW_CHUNK, steps - done)
            before = int(s.alive.sum())
            world.run(n)
            agent_steps += 0.5 * (before + int(s.alive.sum())) * n
            pops.append(int(s.alive.sum()))
            done += n
    else:
        world.run(steps)
    a = s.alive
    eaten = int(s.counters[k.C_EATEN]) - e0
    out["eaten"] = eaten
    out["poison_frac"] = float((int(s.counters[k.C_POISONED]) - p0) / eaten) if eaten else None
    first = int(np.minimum(s.n_meal[a], FIRST_MEALS).sum())
    out["poison_frac_first10"] = float(s.n_poison10[a].sum() / first) if first else None
    out["pop_end"] = int(a.sum())
    if cfg.soc_taste_enabled:
        seen = int(s.n_seen[a].sum())
        out["n_seen"] = seen
        out["n_seen_poison"] = int(s.n_seen_poison[a].sum())
        out["n_seen_kin"] = int(s.n_seen_kin[a].sum())
        out["seen_age_sum"] = int(s.seen_age_sum[a].sum())
        out["seen_poison_frac"] = float(out["n_seen_poison"] / seen) if seen else None
        out["seen_kin_frac"] = float(out["n_seen_kin"] / seen) if seen else None
        out["seen_per_1000_steps"] = float(seen / (int(a.sum()) * steps) * 1000) if a.any() and steps else None
    if juv_on:
        out.update(_juv_window(world, steps, pops, agent_steps, eaten, int(s.counters[k.C_RESEED]) - r0))
    return out


WINDOW_CHUNK = 250     # observe 창에서 무리 크기를 재는 간격(스텝)


def _dist(v: np.ndarray) -> dict:
    return {"mean": float(np.mean(v)), "q25": float(np.quantile(v, 0.25)), "median": float(np.median(v)),
            "q75": float(np.quantile(v, 0.75)), "max": float(np.max(v)), "frac_zero": float(np.mean(v <= 0.0))}


def _juv_window(world, steps: int, pops: list, agent_steps: float, eaten: int, reseeds: int) -> dict:
    """관찰 창의 미숙기 판독(졸업표·곁 머무름·본 부모 끼니·무리 평균·재주입·장부·끼니율). 값은 이 창 안에서만 쌓인 것이다."""
    s, cfg = world.s, world.cfg
    tab = {name: sum(row[name] for row in world.grad_table().values()) for name in k.GT_NAMES}
    born = tab["grad"] + tab["dead"]
    bal = world.energy_balance()
    inflow = bal["spawn"] + bal["inject"]
    grad = tab["grad"]
    a = s.alive
    out = {
        "juv_steps": int(cfg.juv_steps), "pop_mean": float(np.mean(pops)) if pops else None, "reseeds": reseeds,
        "ledger_rel_residual": float(abs(bal["residual"]) / inflow) if inflow > 0 else None,
        "meal_rate_per_agent_step": float(eaten / agent_steps) if agent_steps > 0 else None,
        "grad_table": tab,
        "grad_rate": float(grad / born) if born else None,
        "par_near_frac": float(tab["par_near"] / tab["par_alive"]) if tab["par_alive"] else None,
        "adult_near_per_juv_step": float(tab["adult_near"] / tab["alive_steps"]) if tab["alive_steps"] else None,
        "seen_par_per_grad": float(tab["seen_par"] / grad) if grad else None,
        "seen_par_poison_per_grad": float(tab["seen_par_poison"] / grad) if grad else None,
        "taught_per_grad": float(tab["taught"] / grad) if grad else None,
        "taught_knower_frac": float(tab["taught_knower"] / tab["taught"]) if tab["taught"] else None,
        "grad_poison_meal_frac": float(tab["poison_meals"] / tab["meals"]) if tab["meals"] else None,
        "n_seen_par_alive": int(s.n_seen_par[a].sum()),
    }
    return out


# ---- 관문(판정 0, 설계 4절) ----
GATE_DS = ("0.05", "0.25", "1.0")
CONTROL_MAX_BOTH = 2          # 음성 대조의 양쪽 전멸이 이 판 수를 넘으면 도구 깨짐
FRAC_POSITIVE = 5 / 6         # ①: 반복 중 B가 큰 판의 몫(6판 중 5판 = 이 비율 이상)


def _mean(res: dict) -> float:
    return float(res["r_per_1000"]["mean"])


def _pair_pass(o: dict, bl: dict | None, md: float) -> dict:
    """한 쌍(outcome, blind)의 ①②. ① outcome r > 0이고 반복의 5/6 이상(6판 중 5판)에서 B가 큼.
    ② outcome − blind 차이가 양이고 Welch 검정(measure.t95, df = n − 1)과 md(min_detectable) 둘 다 넘음."""
    out = {"r_outcome": _mean(o)}
    rl = o.get("r_list") or []
    n_pos = int(sum(x > 0 for x in rl))
    need = int(np.ceil(FRAC_POSITIVE * len(rl) - 1e-9)) if rl else 0
    out["n_pos"] = n_pos
    out["c1"] = bool(out["r_outcome"] > 0 and rl and n_pos >= need)
    if bl is None:
        out.update(r_blind=None, welch_diff=None, welch_edge=None, c2=False)
    else:
        diff = out["r_outcome"] - _mean(bl)
        so, sb = o["r_per_1000"], bl["r_per_1000"]
        se = float(np.sqrt(so["sem"] ** 2 + sb["sem"] ** 2)) if np.isfinite(so["sem"]) and np.isfinite(sb["sem"]) else float("inf")
        df = max(1, min(so["n"], sb["n"]) - 1)
        edge = float(measure.t95(df) * se)
        out.update(r_blind=_mean(bl), welch_diff=diff, welch_edge=edge, c2=bool(diff > 0 and diff > edge and diff > md))
    out["pass"] = bool(out["c1"] and out["c2"])
    return out


def gate_tag(results: dict) -> dict:
    """한 바탕(태그)의 관문. results: {판 이름: rate_zero 결과}, 이름은 outcome-d0.05 … blind-d1.0, off-d1.0, outcome-nocost-d0.25·d1.0,
    outcome-noind-d1.0, blind-noind-d1.0. 없는 판은 건너뛴다.
    규칙: min_detectable은 outcome-d0.05에서 읽는다(없으면 도구 깨짐). 음성 대조(off-d1.0)는 |r 평균| ≤ min_detectable이고
    양쪽 전멸이 2판 이하여야 한다 — 아니면(또는 대조가 없으면) "broken". 대조가 멀쩡하면, 비용 있는 판(outcome-d*와 blind-d*의 쌍,
    개인 학습 없는 outcome-noind-d1.0·blind-noind-d1.0 쌍) 가운데 하나라도 ①②를 통과하면 "pass"(비용 있는 d가 통과 못하고 개인 학습 없는
    판만 통과하면 only_noind), 아무것도 통과 못 하면 "undetectable". 비용 없는 판(nocost)은 읽기용이라 값만 적는다."""
    out = {"status": "", "reason": "", "only_noind": False, "min_detectable": None, "d": {}, "noind": None, "nocost": {},
           "control": None}
    ref = results.get("outcome-d0.05") or {}
    md = ref.get("min_detectable_r_per_1000")
    out["min_detectable"] = md
    ctl = results.get("off-d1.0")
    if ctl is not None:
        ok = bool(md is not None and abs(_mean(ctl)) <= md and ctl.get("both_extinct_runs", 0) <= CONTROL_MAX_BOTH)
        out["control"] = {"r_mean": _mean(ctl), "both_extinct_runs": ctl.get("both_extinct_runs", 0), "ok": ok}
    if md is None:
        out.update(status="broken", reason="min_detectable을 읽을 수 없음(outcome-d0.05가 없거나 중립 겨루기를 건너뜀)")
        return out
    if ctl is None:
        out.update(status="broken", reason="음성 대조(off-d1.0)가 없어 판정을 쓰지 않는다")
        return out
    if not out["control"]["ok"]:
        out.update(status="broken", reason=f"음성 대조가 깨짐: r 평균 {out['control']['r_mean']:.2f} "
                   f"(한계 ±{md:.2f}), 양쪽 전멸 {out['control']['both_extinct_runs']}판")
        return out
    for d in GATE_DS:
        o = results.get(f"outcome-d{d}")
        if o is not None:
            out["d"][d] = _pair_pass(o, results.get(f"blind-d{d}"), md)
    o, bl = results.get("outcome-noind-d1.0"), results.get("blind-noind-d1.0")
    if o is not None:
        out["noind"] = _pair_pass(o, bl, md)
    for d in ("0.25", "1.0"):
        o = results.get(f"outcome-nocost-d{d}")
        if o is not None:
            bl = results.get(f"blind-d{d}")
            out["nocost"][d] = {"r_outcome": _mean(o), "r_blind_ref": _mean(bl) if bl is not None else None}
    if not out["d"] and out["noind"] is None:
        out.update(status="broken", reason="읽을 outcome 판이 없음")
        return out
    cost_pass = any(v["pass"] for v in out["d"].values())
    noind_pass = bool(out["noind"] and out["noind"]["pass"])
    if cost_pass or noind_pass:
        out.update(status="pass", only_noind=bool(not cost_pass and noind_pass))
    else:
        out.update(status="undetectable", reason="모든 d에서 ①②가 안 됨(음성 대조는 멀쩡)")
    return out


def gate_tags(per_tag: dict) -> dict:
    """바탕마다 gate_tag를 돌려 합친다. 한 바탕이라도 통과면 0(통과한 바탕이 모두 개인 학습 없는 판만 통과면 only_noind),
    통과 없이 깨진 바탕이 있으면 4, 그 밖에(모두 검출 불능) 3."""
    tags = {tag: gate_tag(res) for tag, res in per_tag.items()}
    passed = [g for g in tags.values() if g["status"] == "pass"]
    if passed:
        status, code = "pass", 0
    elif any(g["status"] == "broken" for g in tags.values()) or not tags:
        status, code = "broken", 4
    else:
        status, code = "undetectable", 3
    return {"status": status, "exit_code": code, "only_noind": bool(passed and all(g["only_noind"] for g in passed)),
            "tags": tags}


def gate(directory: Path, tags: list[str]) -> dict:
    per_tag = {}
    for tag in tags:
        res = {}
        for f in sorted(Path(directory).glob(f"j-dose-*-{tag}.json")):
            name = f.name[len("j-dose-"):-len(f"-{tag}.json")]
            res[name] = json.loads(f.read_text())
        per_tag[tag] = res
    return gate_tags(per_tag)


# ---- 부모의 가르침 관문(순수 계산은 teach_gate.py, 여기는 파일 읽기) ----
def _read_cells(directory: Path, tag: str, prefix: str = "j-dose-") -> dict:
    res = {}
    tail = f"-{tag}.json"
    for f in sorted(Path(directory).glob(f"{prefix}*{tail}")):
        if f.stat().st_size:
            res[f.name[len(prefix):-len(tail)]] = json.loads(f.read_text())
    return res


def gate_world(directory: Path, tags: list[str]) -> dict:
    per_tag = {}
    for tag in tags:
        w = {}
        for name in ("off", "on", "exp"):
            f = Path(directory) / f"w-{name}-{tag}.json"
            if f.is_file() and f.stat().st_size:
                w[name] = json.loads(f.read_text())
        per_tag[tag] = w
    return teach_gate.gate_world(per_tag)


def gate_teach(directory: Path, tags: list[str], pop: float = teach_gate.DEFAULT_POP) -> dict:
    per_tag = {tag: _read_cells(directory, tag) for tag in tags}
    exp_ok = False
    f = Path(directory) / "gate-world.json"
    if f.is_file() and f.stat().st_size:
        exp_ok = bool(json.loads(f.read_text()).get("exp_ok", False))
    return teach_gate.gate_tags(per_tag, pop, {t: exp_ok for t in tags})


def prereq_soc(directory: Path, tags: list[str]) -> dict:
    per_tag = {}
    for tag in tags:
        cells = {}
        for dd in teach_gate.PREREQ_DS:
            f = Path(directory) / f"j-dose-outcome-d{dd}-{tag}.json"
            if f.is_file() and f.stat().st_size:
                cells[f"outcome-d{dd}"] = json.loads(f.read_text())
        per_tag[tag] = cells
    return teach_gate.prereq_soc(per_tag)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rate-zero")
    r.add_argument("ckpt", type=Path)
    r.add_argument("--gene", choices=GENES, default="taste")
    r.add_argument("--teach-d", type=float, default=None,
                   help="--gene show|feed: B쪽 show_g(feed는 feed_g)를 모두 D로")
    r.add_argument("--soc-none", action="store_true", help="양쪽 soc_g = −soc_rate(보고 배우기 없음, 가르침만 본다)")
    r.add_argument("--n", type=int, default=400)
    r.add_argument("--steps", type=int, default=8000)
    r.add_argument("--reps", type=int, default=6)
    r.add_argument("--seed", type=int, default=6000)
    r.add_argument("--set", dest="sets", action="append", metavar="KEY=VAL",
                   help="겨루기 설정 덮어쓰기(여러 번 줄 수 있다)")
    r.add_argument("--soc-d", type=float, default=None, help="B쪽을 모두 보고 배우기 실제 속도 D로(soc_g = D − soc_rate)")
    r.add_argument("--no-individual", action="store_true", help="두 쪽 모두 개인 학습 속도 0(taste_g = −taste_rate)")
    r.add_argument("--no-neutral", action="store_true", help="중립 겨루기를 건너뜀(min_detectable도 안 낸다)")
    r.add_argument("--json", type=Path)
    o = sub.add_parser("observe")
    o.add_argument("ckpt", type=Path)
    o.add_argument("--steps", type=int, default=3000)
    o.add_argument("--json", type=Path)
    g = sub.add_parser("gate")
    g.add_argument("--dir", type=Path, required=True)
    g.add_argument("--tags", nargs="+", required=True)
    g.add_argument("--json", type=Path)
    for name in ("gate-world", "gate-teach", "prereq-soc"):
        q = sub.add_parser(name)
        q.add_argument("--dir", type=Path, required=True)
        q.add_argument("--tags", nargs="+", required=True)
        q.add_argument("--json", type=Path)
        if name == "gate-teach":
            q.add_argument("--pop", type=float, default=teach_gate.DEFAULT_POP, help="관문 −2b의 무리 크기")
    args = p.parse_args(argv)
    code = 0
    if args.cmd == "rate-zero":
        out = rate_zero(args.ckpt, args.gene, args.n, args.steps, args.reps, args.seed, args.sets, args.soc_d,
                        args.no_individual, not args.no_neutral, args.teach_d, args.soc_none)
    elif args.cmd == "gate":
        out = gate(args.dir, args.tags)
        code = out["exit_code"]
    elif args.cmd in ("gate-world", "gate-teach", "prereq-soc"):
        if args.cmd == "gate-world":
            out = gate_world(args.dir, args.tags)
        elif args.cmd == "gate-teach":
            out = gate_teach(args.dir, args.tags, args.pop)
        else:
            out = prereq_soc(args.dir, args.tags)
        code = out["exit_code"]
    else:
        out = observe(args.ckpt, args.steps)
    text = json.dumps(out, ensure_ascii=False, indent=1, default=float)
    if args.json:
        args.json.write_text(text)
    print(text)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
