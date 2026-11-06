"""손으로 만든 "배워서 가려 먹는 개체" 시험 (2026-11-05, 결과는 results.md "네 번째 시도").

회로: 은닉 뉴런 0 하나. 먹이 시야 칸(값의 부호 = 먹이 종류) → 회전. 왼쪽/오른쪽 칸의 가중치 부호가 반대라
종류 A(+)에는 다가가고 B(-)는 피한다. 평생 학습(Hin)이 이 가중치의 부호를 뒤집으면 반대 종류를 쫓는다.
보상 가중치는 에너지 변화만 쓴다. 행동 편향: 앞으로 가기, 번식, 졸리면 자기.

    PYTHONPATH=. python wayfinder/stage2-4/hand_learner.py solo <체크포인트.npz> <스텝> <출력.json>
    PYTHONPATH=. python wayfinder/stage2-4/hand_learner.py assay <체크포인트.npz> <스텝> <출력.json>
    PYTHONPATH=. python wayfinder/stage2-4/hand_learner.py grid <체크포인트.npz> <스텝> <출력.json> <설정 묶음>...

solo: 한 무리만 풀고 독 비율(독을 먹은 수 / 먹은 수, 처음 1/4은 버림)을 잰다.
assay: 학습하는 손 설계 개체(B)를 체크포인트의 진화한 무리(A)와 겨루게 한다(measure.assay_sides).
       HAND_ASSAY=<변형>으로 손 개체를 고르고(기본 learn), HAND_SET은 두 쪽이 겨루는 세계의 설정에 들어간다.
grid:  설정 묶음("perturb_sigma=0.1,trace_decay=0.95" 꼴)마다 변형(HAND_VARIANTS)과 씨앗(HAND_SEEDS)을 돌려
       좋은 종류가 A인 구간과 B인 구간의 독 비율을 따로 적는다(결과는 results.md "다섯 번째 시도").
curve: 학습 규칙만 재는 세계(LAB: 대사 비용 0, 번식 끔, 늙어 죽지 않음 → 같은 개체가 끝까지 산다)에서
       구간마다 독 비율이 처음 1/3에서 마지막 1/3로 어떻게 바뀌는지와 배선이 옳은 쪽으로 기운 정도를 적는다.
       HAND_N으로 마릿수를 바꾼다(적을수록 한 마리가 많이 먹는다). 결과는 results.md "여섯 번째 시도".
taste: 맛 기억을 읽는 손 개체(변형 taste: 물든 시야 왼쪽 네 칸 +0.4, 오른쪽 네 칸 −0.4, 회전 1, 원래 시야 0,
       학습 없음)가 독이 바뀐 뒤 독을 피하는지 본다. 체크포인트를 preset=taste 크기로 읽고, 세계 둘(LAB과
       체크포인트 설정 그대로) × 맛 켬/끔(taste_enabled=false, 같은 크기) × 씨앗(HAND_SEEDS)을 돌린다.
       실제 바뀜마다 바뀐 뒤 300스텝(first)과 그 뒤 다음 바뀜까지(rest)의 독 비율, 놓인 먹이 수와 그 가운데 독을 적는다.
       예측: 맛 켬은 바뀐 뒤 몇 백 스텝 안에 독 비율 0.35 아래로, 맛 끔(물든 시야 0 → 아무 데로나)은 0.5.
       판정 0은 세계마다 낸다: 맛 켬의 rest 평균 < 0.35이고 맛 끔의 rest 평균이 0.5 ± 0.05면 그 세계 통과.
       전체 통과 = 두 세계 모두 통과(그대로 세계에서도 되는지가 예측의 절반이라 합친 평균으로 내지 않는다).

    PYTHONPATH=. python wayfinder/stage2-4/hand_learner.py taste <체크포인트.npz> <스텝> <출력.json>
"""
import json
import os
import math
import sys
from pathlib import Path

import numpy as np

from kkumteul import checkpoint as ckpt
from kkumteul import kernels as k
from kkumteul import measure
from kkumteul.config import (ACT_ATTACK, ACT_DEPOSIT, ACT_GIVE, ACT_REPRO, ACT_SIG0, ACT_SIG1,
                             ACT_SLEEP, ACT_THRUST, ACT_TURN, TASTE_VIS)
from kkumteul.world import World

VARIANTS = {
    "learn": dict(eta=0.1, alpha=4.0, g=2.0),     # 배운다
    "fixed": dict(eta=0.0, alpha=0.0, g=2.0),     # 같은 회로, 학습 없음(늘 A를 쫓는다)
    "fast": dict(eta=0.2, alpha=10.0, g=2.0),     # 빨리 배운다(Hin 0.2면 뒤집힌다)
    "fast_slow": dict(eta=0.2, alpha=10.0, g=2.0, thrust=0.4),   # 빨리 배우고 천천히 다닌다
    "fixed_slow": dict(eta=0.0, alpha=0.0, g=2.0, thrust=0.4),
    "fast_b": dict(eta=0.2, alpha=10.0, g=2.0, rw_bias=0.016),   # + reward_baseline_rate=0 으로 쓴다
    "blank": dict(eta=0.2, alpha=10.0, g=0.0),    # 타고난 취향 없이 배운 것만으로 쫓는다
    "blank_slow": dict(eta=0.05, alpha=10.0, g=0.0),
    "fast_slow_eta": dict(eta=0.05, alpha=10.0, g=2.0),
    # 포화하지 않는 배선(시야 가중치 0.4, 회전 출력 가중치 1): 조금 흔들면 행동이 바뀐다
    "soft": dict(eta=0.2, alpha=10.0, g=0.4, turn=1.0),         # 타고난 취향(A) + 학습
    "soft_fixed": dict(eta=0.0, alpha=0.0, g=0.4, turn=1.0),    # 같은 회로, 학습 없음
    "blank1": dict(eta=0.2, alpha=10.0, g=0.0, turn=1.0),       # 타고난 취향 없이 학습만
    "blank1_slow": dict(eta=0.05, alpha=10.0, g=0.0, turn=1.0),
    "blank1_fixed": dict(eta=0.0, alpha=0.0, g=0.0, turn=1.0),  # 아무것도 안 쫓고 안 배운다(바닥)
    # 맛 기억을 읽는다: 물든 시야(기억이 좋다고 한 모양은 +, 독이라고 한 모양은 −) → 회전. 원래 시야는 0.
    # 맛 블록 번호(taste_in0)를 넘겨야 만들 수 있다(make_hand)
    "taste": dict(eta=0.0, alpha=0.0, g=0.0, turn=1.0, taste_g=0.4),
}
TASTE_PASS = 0.35       # 판정 0: 맛 켬의 바뀐 뒤 독 비율이 이것보다 낮아야 한다
TASTE_CTRL = (0.5, 0.05)    # 판정 0: 맛 끔은 0.5 ± 0.05(아무 데로나 먹는다)

# 학습 규칙만 재는 세계: 굶어 죽지도 늙어 죽지도 않는다. 보상은 먹은 순간의 에너지 변화뿐이다.
LAB = ("basal_cost=0,thrust_cost=0,turn_cost=0,brain_cost=0,synapse_cost=0,plast_cost=0,"
       "max_age=100000000,e_start=6,e_max=1000000000")     # e_max: 배가 차서 안 먹는 일이 없게


def make_hand(tmpl: dict, n: int, eta: float, alpha: float, g: float, rw_energy: float = 1.0,
              rw_ate: float = 0.0, nsec: int = 8, thrust: float = 1.0, rw_bias: float = 0.0,
              turn: float = 3.0, taste_g: float = 0.0, taste_in0: int = -1) -> dict:
    if taste_g != 0 and taste_in0 < 0:
        raise ValueError("맛 블록을 읽는 배선은 맛 블록 번호가 필요하다: make_hand(..., taste_in0=cfg.taste_in0)")
    out = {name: np.repeat(v[:1], n, axis=0).copy() for name, v in tmpl.items()}
    for name in ("Win", "Wrec", "bh", "Wout", "bout", "alpha", "rw"):
        out[name][...] = 0
    for s in range(nsec):
        out["Win"][:, s, 0] = g if s >= nsec // 2 else -g     # 높은 번호 칸 = 왼쪽(반시계) → 회전 +
        if taste_g != 0:
            out["Win"][:, taste_in0 + TASTE_VIS + s, 0] = taste_g if s >= nsec // 2 else -taste_g
    out["Wout"][:, ACT_TURN, 0] = turn
    out["bout"][:, ACT_THRUST] = thrust
    out["bout"][:, ACT_REPRO] = 1.0
    out["bout"][:, ACT_SLEEP] = 1.0
    for a in (ACT_GIVE, ACT_DEPOSIT, ACT_ATTACK):
        out["bout"][:, a] = -1.0
    out["bout"][:, ACT_SIG0] = out["bout"][:, ACT_SIG1] = 0.0
    out["n_hidden"][:] = 1
    out["eta"][:] = eta
    out["alpha"][:, 0] = alpha
    out["rw"][:, 0] = rw_ate
    out["rw"][:, 1] = rw_energy
    out["rw"][:, 7] = rw_bias       # 평소 대사 비용(에너지 변화 × 2)을 메워 먹지 않을 때 보상이 0 근처이게
    return out


def _flip(s, alive, nsec: int = 8) -> float:
    """시야→회전 실효 가중치가 A를 쫓는 쪽인 정도(1 = 여덟 칸 모두 A를 쫓음, -1 = 모두 B를 쫓음, 0 = 어느 쪽도 아님).
    A를 쫓는 배선은 왼쪽 칸(높은 번호) +, 오른쪽 칸 − 이다(make_hand)."""
    if not alive.any():
        return float("nan")     # 다 죽은 창은 평균에서 뺀다(0으로 넣으면 "어느 쪽도 아님"으로 읽힌다)
    eff = s.Win[alive, :nsec, 0] + s.alpha[alive, 0:1] * s.Hin[alive, :nsec, 0]
    sg = np.where(np.arange(nsec) >= nsec // 2, 1.0, -1.0)
    return float(np.mean(np.sign(eff) * sg))


def _pref(s, alive, nsec: int = 8) -> tuple[float, float]:
    """시야→회전 실효 가중치가 A를 쫓는 배선 쪽으로 기운 정도. (기운 비율 −1~+1의 개체 평균, 가중치 크기의 평균)
    기운 비율 = Σ(부호 맞춘 가중치) / Σ|가중치|. 가중치가 전부 0인 개체는 0으로 센다."""
    if not alive.any():
        return float("nan"), float("nan")
    eff = (s.Win[alive, :nsec, 0] + s.alpha[alive, 0:1] * s.Hin[alive, :nsec, 0]).astype(np.float64)
    sg = np.where(np.arange(nsec) >= nsec // 2, 1.0, -1.0)
    proj = (eff * sg).sum(axis=1)
    tot = np.abs(eff).sum(axis=1)
    return float(np.mean(np.where(tot > 1e-9, proj / np.maximum(tot, 1e-9), 0.0))), float(np.mean(tot) / nsec)


def _ratio(rows) -> float:
    """독을 먹은 수 / 먹은 수. 먹은 것이 없으면 nan."""
    e, p = sum(r["eaten"] for r in rows), sum(r["poisoned"] for r in rows)
    return p / e if e else float("nan")


def _placed(s, good: int) -> tuple[int, float]:
    """들판에 놓인 먹이(살아 있고 아무도 들지 않은 종류 A·B) 수와 그 가운데 지금 독인 종류의 비율."""
    on = s.f_alive & (s.f_holder < 0) & ((s.f_kind == k.KIND_A) | (s.f_kind == k.KIND_B))
    n = int(on.sum())
    return n, float((s.f_kind[on] != good).mean()) if n else float("nan")


def curve(cfg, genomes: dict, n: int, steps: int, seed: int, every: int = 250, lab: bool = True) -> dict:
    """같은 개체들이 구간을 넘기며 배우는지 본다. lab이면 LAB 세계에서 번식을 끈다(유전체의 번식 편향을 −5로).
    lab이 아니면 받은 설정 그대로 돌리고 번식도 그대로 둔다."""
    rng = np.random.default_rng(seed)
    acfg = measure.arena_config(with_set(cfg, LAB) if lab else cfg, seed, n)
    genomes = {name: v.copy() for name, v in genomes.items()}
    if lab:
        genomes["bout"][:, ACT_REPRO] = -5.0
    world = World(acfg, populate=False)
    for _ in range(int(0.5 * acfg.food_cap * acfg.food_energy / max(acfg.food_inflow, 1e-9))):
        k.spawn_food(0, world.s, world.fp, world.ip)
    rows = rng.choice(len(genomes["n_hidden"]), size=n, replace=len(genomes["n_hidden"]) < n)
    for i in range(n):
        measure.inject_agent(world, genomes, int(rows[i]), 0, float(rng.random() * acfg.world_w),
                             float(rng.random() * acfg.world_h), float(rng.random() * 2 * math.pi), 0)
    c = world.s.counters
    period = acfg.kind_flip_period
    series = []
    pe, pp = int(c[k.C_EATEN]), int(c[k.C_POISONED])
    for _ in range(0, steps, every):
        world.run(every)
        e, p = int(c[k.C_EATEN]), int(c[k.C_POISONED])
        alive = world.s.alive
        good = int(k.good_kind(acfg.seed, world.step_no - 1, period))
        idx, size = _pref(world.s, alive)
        placed, placed_poison = _placed(world.s, good)
        series.append({"step": world.step_no, "pop": int(alive.sum()), "eaten": e - pe, "poisoned": p - pp,
                       "good": good, "right": idx if good == 0 else -idx, "size": size,
                       "placed": placed, "placed_poison": placed_poison})
        pe, pp = e, p
    spans = []
    for a in range(0, steps, period):
        rows_ = [r for r in series if a < r["step"] <= a + period]
        if not rows_:
            continue
        third = max(1, len(rows_) // 3)
        spans.append({"good": "AB"[rows_[0]["good"]], "first": _ratio(rows_[:third]), "last": _ratio(rows_[-third:]),
                      "all": _ratio(rows_), "right_end": rows_[-1]["right"], "size_end": rows_[-1]["size"],
                      "meals": sum(r["eaten"] for r in rows_) / max(1, rows_[-1]["pop"]), "pop": rows_[-1]["pop"]})
    later = spans[1:] or spans      # 첫 구간은 배우기 전이라 뺀다
    return {"spans": spans, "series": series,
            "last_mean": float(np.nanmean([x["last"] for x in later])),
            "right_mean": float(np.nanmean([x["right_end"] for x in later]))}


def after_flips(series: list, period: int, seed: int, win: int = 300) -> list[dict]:
    """실제 바뀜(good이 앞 줄과 달라진 줄)마다 바뀐 뒤의 독 비율을 낸다(curve의 series를 받는다).
    줄 하나는 앞 줄 스텝부터 그 줄 스텝 직전까지 먹은 것이다. 바뀐 스텝(flip)은 그 사이 구간 경계 가운데
    좋은 종류가 실제로 달라진 곳이다(다시 정해졌지만 그대로인 경계는 바뀜이 아니다). 씨앗으로 보아 그런 경계가
    없으면 씨앗이나 주기를 잘못 넘긴 것이라 멈춘다.
    first = flip부터 win스텝 안에서 시작한 줄들의 독 비율, rest = 그 뒤 다음 바뀜 줄 전까지(없으면 끝까지)."""
    at = [i for i in range(1, len(series)) if series[i]["good"] != series[i - 1]["good"]]
    out = []
    for m, i in enumerate(at):
        lo, hi = series[i - 1]["step"], series[i]["step"]
        real = [b for b in range(-(-lo // period) * period, hi, period)
                if b > 0 and k.good_kind(seed, b, period) != k.good_kind(seed, b - 1, period)]
        if not real:
            raise ValueError(f"스텝 {lo}~{hi} 사이에 실제 바뀜이 없다: 씨앗({seed})이나 주기({period})가 이 줄들과 다르다")
        flip = real[-1]
        seg = range(i, at[m + 1] if m + 1 < len(at) else len(series))
        out.append({"flip": flip,
                    "first": _ratio([series[j] for j in seg if series[j - 1]["step"] < flip + win]),
                    "rest": _ratio([series[j] for j in seg if series[j - 1]["step"] >= flip + win])})
    return out


def taste_verdict(rest: dict) -> dict:
    """판정 0을 세계마다 낸다. rest는 (lab, 맛 켬) → 바뀜별 rest 목록. 전체 통과 = 두 세계 모두 통과."""
    worlds = []
    for lab in (True, False):
        on_rest, off_rest = (float(np.nanmean(rest[(lab, v)])) if rest.get((lab, v)) else float("nan")
                             for v in (True, False))
        ok = bool(on_rest < TASTE_PASS and abs(off_rest - TASTE_CTRL[0]) <= TASTE_CTRL[1])
        worlds.append({"lab": lab, "pass": ok, "on_rest": on_rest, "off_rest": off_rest})
    return {"pass": all(w["pass"] for w in worlds), "worlds": worlds}


def solo(cfg, genomes: dict, n: int, steps: int, seed: int, every: int = 500) -> dict:
    rng = np.random.default_rng(seed)
    acfg = measure.arena_config(cfg, seed, n)
    world = World(acfg, populate=False)
    for _ in range(int(0.5 * acfg.food_cap * acfg.food_energy / max(acfg.food_inflow, 1e-9))):
        k.spawn_food(0, world.s, world.fp, world.ip)
    rows = rng.choice(len(genomes["n_hidden"]), size=n, replace=len(genomes["n_hidden"]) < n)
    for i in range(n):
        measure.inject_agent(world, genomes, int(rows[i]), 0, float(rng.random() * acfg.world_w),
                             float(rng.random() * acfg.world_h), float(rng.random() * 2 * math.pi),
                             int(rng.integers(0, max(1, acfg.max_age // 2))))
    c = world.s.counters
    series = []
    pe, pp = int(c[k.C_EATEN]), int(c[k.C_POISONED])
    for _ in range(0, steps, every):
        world.run(every)
        e, p = int(c[k.C_EATEN]), int(c[k.C_POISONED])
        alive = world.s.alive
        series.append({"step": world.step_no, "pop": int(alive.sum()), "eaten": e - pe, "poisoned": p - pp,
                       "good": int(k.good_kind(acfg.seed, world.step_no - 1, acfg.kind_flip_period)),
                       "hin_abs": float(np.abs(world.s.Hin[alive, :8, 0]).mean()) if alive.any() else 0.0,
                       "flip": _flip(world.s, alive),
                       "mod": float(world.s.mod[alive].mean()) if alive.any() else 0.0,
                       "de": float((world.s.energy[alive] - world.s.e_prev[alive]).mean()) if alive.any() else 0.0})
        pe, pp = e, p
        if not alive.any():
            break
    keep = [r for r in series if r["step"] > steps / 4] or series
    eat = sum(r["eaten"] for r in keep)
    poi = sum(r["poisoned"] for r in keep)
    out = {"ratio": poi / eat if eat else float("nan"), "eaten": eat, "pop_end": series[-1]["pop"],
           "pop_mean": float(np.mean([r["pop"] for r in keep])), "series": series}
    for name, good in (("a", 0), ("b", 1)):     # 좋은 종류가 A인 구간과 B인 구간을 따로 본다(처음부터 전부)
        rows = [r for r in series if r["good"] == good]
        e_, p_ = sum(r["eaten"] for r in rows), sum(r["poisoned"] for r in rows)
        out[f"ratio_{name}"] = p_ / e_ if e_ else float("nan")
        flips = [r["flip"] for r in rows if not math.isnan(r["flip"])]
        out[f"flip_{name}"] = float(np.mean(flips)) if flips else float("nan")
        out[f"pop_{name}"] = float(np.mean([r["pop"] for r in rows])) if rows else float("nan")
    return out


def with_set(cfg, text: str):
    """"이름=값,이름=값"을 설정에 덮어쓴다."""
    d = cfg.to_dict()
    for kv in filter(None, text.split(",")):
        key, val = kv.split("=")
        d[key] = type(d[key])(float(val)) if not isinstance(d[key], bool) else val == "true"
    return type(cfg).from_dict(d)


def main():
    mode, ck, steps, out = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
    if mode == "taste":     # 저장 설정에 맛 블록 크기와 장치를 덧붙여 읽는다(늘어난 입력 칸은 0)
        world, _ = ckpt.load(ck, ckpt.read_config(ck).with_overrides(["preset=taste"]))
        side = measure.side_from_world(world, str(ck))
    else:
        side = measure.load_side(ck)
    cfg = side["cfg"]
    if os.environ.get("HAND_SET"):       # 예: HAND_SET=reward_baseline_rate=0
        cfg = with_set(cfg, os.environ["HAND_SET"])
        side = dict(side, cfg=cfg)
    n = 400
    res = {"checkpoint": ck.name, "steps": steps, "variants": VARIANTS}
    plain = [name for name, kw in VARIANTS.items() if not kw.get("taste_g")]    # 맛 블록 번호 없이 만드는 변형
    if mode == "solo":
        for name in plain:
            kw = VARIANTS[name]
            gnm = make_hand(side["genomes"], n, **kw)
            res[name] = [solo(cfg, gnm, n, steps, 100 + r) for r in range(2)]
        res["evolved"] = [solo(cfg, side["genomes"], n, steps, 100 + r) for r in range(2)]
        for name in plain + ["evolved"]:
            print(name, [(round(x["ratio"], 3), x["eaten"], round(x["pop_mean"]), x["pop_end"])
                         for x in res[name]], flush=True)
            for x in res[name]:
                x["series"] = x["series"][-6:]
    elif mode == "diag":
        # 구간마다 좋은 종류와 함께 본다(좋은 종류가 0=A면 독 = B를 먹은 것)
        for name in sys.argv[5:] or plain:
            gnm = make_hand(side["genomes"], n, **VARIANTS[name]) if name in VARIANTS else side["genomes"]
            seed = int(os.environ.get("HAND_SEED", "100"))
            x = solo(cfg, gnm, n, steps, seed, every=250)
            print(name)
            for r in x["series"]:
                good = k.good_kind(seed, r["step"] - 1, cfg.kind_flip_period)
                print(f"  {r['step']:6d} good={good} pop={r['pop']:4d} eat={r['eaten']:5d} "
                      f"poi={r['poisoned']:5d} hin={r['hin_abs']:.3f} flip={r['flip']:+.2f} mod={r['mod']:+.4f}", flush=True)
    elif mode == "grid":
        names = os.environ.get("HAND_VARIANTS", "fixed,fast,blank").split(",")
        seeds = [int(x) for x in os.environ.get("HAND_SEEDS", "104,105").split(",")]
        res["grid"] = []
        for sets in sys.argv[5:] or [""]:
            gcfg = with_set(cfg, sets)
            for name in names:
                gnm = make_hand(side["genomes"], n, **VARIANTS[name]) if name in VARIANTS else side["genomes"]
                for seed in seeds:
                    x = solo(gcfg, gnm, n, steps, seed, every=250)
                    row = {"set": sets, "variant": name, "seed": seed,
                           **{key: x[key] for key in ("ratio", "ratio_a", "ratio_b", "flip_a", "flip_b",
                                                      "pop_a", "pop_b", "pop_end")},
                           "goods": "".join("AB"[r["good"]] for r in x["series"][5::6]),
                           "flips": [round(r["flip"], 2) for r in x["series"][5::6]],
                           "ratios": [round(r["poisoned"] / r["eaten"], 2) if r["eaten"] else None
                                      for r in x["series"][5::6]]}
                    res["grid"].append(row)
                    print(f"{sets or '-':42s} {name:10s} {seed} 독A={row['ratio_a']:.2f} 독B={row['ratio_b']:.2f} "
                          f"뒤집힘A={row['flip_a']:+.2f} B={row['flip_b']:+.2f} 개체A={row['pop_a']:.0f} "
                          f"B={row['pop_b']:.0f} 끝={row['pop_end']}", flush=True)
                    out.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    elif mode == "curve":
        names = os.environ.get("HAND_VARIANTS", "blank1,soft").split(",")
        seeds = [int(x) for x in os.environ.get("HAND_SEEDS", "102,100").split(",")]
        n = int(os.environ.get("HAND_N", "400"))
        res["n"] = n
        res["lab"] = LAB
        res["curve"] = []
        for sets in sys.argv[5:] or [""]:
            gcfg = with_set(cfg, sets)
            for name in names:
                gnm = make_hand(side["genomes"], n, **VARIANTS[name])
                for seed in seeds:
                    x = curve(gcfg, gnm, n, steps, seed)
                    res["curve"].append({"set": sets, "variant": name, "seed": seed, **x})
                    spans = "  ".join(f"{s['good']} {s['first']:.2f}→{s['last']:.2f} ({s['right_end']:+.2f})"
                                      for s in x["spans"])
                    print(f"{sets or '-':58s} {name:12s} {seed} n={n} | {spans} | 끝 1/3 평균 {x['last_mean']:.3f} "
                          f"옳은 쪽 {x['right_mean']:+.2f} 끼니 {x['spans'][-1]['meals']:.0f}", flush=True)
                    out.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    elif mode == "taste":
        seeds = [int(x) for x in os.environ.get("HAND_SEEDS", "102,100").split(",")]
        n = int(os.environ.get("HAND_N", "400"))
        gnm = make_hand(side["genomes"], n, taste_in0=cfg.taste_in0, **VARIANTS["taste"])
        res.update(n=n, lab=LAB, taste=[])
        rest = {}
        for lab in (True, False):
            for on in (True, False):
                tcfg = cfg if on else with_set(cfg, "taste_enabled=false")
                for seed in seeds:
                    x = curve(tcfg, gnm, n, steps, seed, every=100, lab=lab)
                    flips = after_flips(x["series"], tcfg.kind_flip_period, seed)
                    row = {"lab": lab, "taste": on, "seed": seed, "flips": flips,
                           "first_mean": float(np.nanmean([f["first"] for f in flips])) if flips else float("nan"),
                           "rest_mean": float(np.nanmean([f["rest"] for f in flips])) if flips else float("nan"),
                           "placed_mean": float(np.mean([r["placed"] for r in x["series"]])),
                           "placed_poison_mean": float(np.nanmean([r["placed_poison"] for r in x["series"]])),
                           "series": x["series"]}
                    res["taste"].append(row)
                    rest.setdefault((lab, on), []).extend(f["rest"] for f in flips)
                    text = "  ".join(f"{f['flip']} {f['first']:.2f}→{f['rest']:.2f}" for f in flips)
                    print(f"{'LAB' if lab else '그대로':6s} {'맛 켬' if on else '맛 끔':5s} {seed} n={n} | {text} | "
                          f"first 평균 {row['first_mean']:.3f} rest 평균 {row['rest_mean']:.3f} | "
                          f"놓인 먹이 {row['placed_mean']:.0f} 그 가운데 독 {row['placed_poison_mean']:.2f}", flush=True)
                    out.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
        res["verdict0"] = v = taste_verdict(rest)
        for w in v["worlds"]:
            print(f"판정 0 {'LAB' if w['lab'] else '그대로':6s}: {'통과' if w['pass'] else '실패'} | "
                  f"맛 켬 rest 평균 {w['on_rest']:.3f} (< {TASTE_PASS}) "
                  f"맛 끔 {w['off_rest']:.3f} ({TASTE_CTRL[0]} ± {TASTE_CTRL[1]})", flush=True)
        print(f"판정 0: {'통과' if v['pass'] else '실패'} (두 세계 모두 통과해야 통과)", flush=True)
    elif mode == "assay":
        hand = dict(side, path="hand-learn",
                    genomes=make_hand(side["genomes"], 400, **VARIANTS[os.environ.get("HAND_ASSAY", "learn")]))
        side = dict(side, path=ck.name)
        r = measure.assay_sides(side, hand, 400, 8000, 4, 3000)
        res["assay"] = r
        print(r["verdict"], r["r_per_1000"], r["final_counts"], flush=True)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
