"""성장 측정: 두 체크포인트의 개체를 새 세계에 같은 수로 풀고 어느 쪽 후손이 더 늘었는지 본다.

대결이 없는 세계라서 "이긴다"는 후손이 더 빨리 느는 것이다(Lenski식 selection rate).
    r = d/dt ln(N_B / N_A)      (양수면 B가 이김)
- 겨루는 동안에는 변이를 끈다(겨루는 중에 진화하면 무엇을 쟀는지 모르게 된다).
- 수용력 근처의 수로 시작하고 나이를 흩어 놓는다. 빈 세계에서 불어나는 능력이 아니라
  본 실행과 비슷한 경쟁 상태에서의 차이를 재려는 것이다.
- 두 무리는 같은 자리 쌍에서 시작한다(먹이 자리 근처에 떨어지는 운을 양쪽이 같이 받게).
- 처음 1/3은 버리고 ln(N_B/N_A)의 기울기로 r을 낸다.
- 같은 무리를 둘로 나눠 겨루게 한 값(A 대 A, B 대 B)을 같이 낸다. 이 값의 평균이 0에서 멀면
  측정 자체가 한쪽으로 기운 것이다.

    python -m kkumteul.measure assay  --a runs/main/checkpoints/ckpt_X.npz --b .../ckpt_Y.npz
    python -m kkumteul.measure assay  --a mine.pack.npz --b yours.pack.npz     (생물 묶음끼리, `pack.py`)
    python -m kkumteul.measure ladder --run runs/main --count 5

한계(독립 검토 2026-11-04): 같은 무리끼리도 r의 표준편차가 1,000스텝당 0.08쯤 된다.
6회 반복으로는 6,000스텝 뒤 개체 수 비가 약 1.4배 이상 벌어지는 차이만 가를 수 있다.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from . import checkpoint as ckpt
from . import kernels as k
from .config import Config
from .world import World

GENOME = ("Win", "Wrec", "bh", "Wout", "bout", "n_hidden", "sigma", "color")
_T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23,
        12: 2.18, 15: 2.13, 20: 2.09, 30: 2.04}


def t95(df: int) -> float:
    """양측 95% t 임계값."""
    if df < 1:
        return float("inf")
    best = max((d for d in _T95 if d <= df), default=1)
    return _T95[best] if df <= 30 else 1.96


def extract_genomes(world: World) -> dict:
    a = np.flatnonzero(world.s.alive)
    g = {name: getattr(world.s, name)[a].copy() for name in GENOME}
    g["gen"] = world.s.gen[a].copy()
    return g


def load_side(path: Path) -> dict:
    """체크포인트나 생물 묶음(pack) 하나에서 겨루기에 필요한 것만 읽어 둔다(나중에 파일이 지워져도 되게)."""
    from . import pack   # pack이 이 모듈을 쓰므로 여기서 불러온다
    if pack.is_pack(path):
        return pack.load_side(path)
    world, meta = ckpt.load(Path(path))
    g = extract_genomes(world)
    if not len(g["n_hidden"]):
        raise ValueError(f"살아 있는 개체가 없는 체크포인트다: {path}")
    return {"path": str(path), "step": meta["step"], "genomes": g, "cfg": world.cfg,
            "gen_mean": float(g["gen"].mean())}


def check_same_shape(a: dict, b: dict) -> None:
    """두 쪽의 유전체 배열 모양(개체 수 빼고)이 같은지 본다. 다르면 같은 세계에 넣을 수 없다."""
    for name in GENOME:
        sa, sb = a["genomes"][name].shape[1:], b["genomes"][name].shape[1:]
        if sa != sb:
            raise ValueError(f"두 쪽의 유전체 모양이 달라 겨룰 수 없다: {name}이 A는 {sa}, B는 {sb}다. "
                             "감각 칸 수와 뇌 최대 크기 설정(vision_sectors, n_spare_in, n_spare_out, h_max)이 "
                             "같은 세계에서 나온 생물끼리만 겨룰 수 있다")
    cfg = b["cfg"]
    want = (cfg.n_in, cfg.h_max)
    if b["genomes"]["Win"].shape[1:] != want:
        raise ValueError(f"겨룰 세계의 설정과 유전체 모양이 안 맞는다: Win이 {b['genomes']['Win'].shape[1:]}, "
                         f"설정은 {want}다")


def arena_config(cfg: Config, seed: int, n_total: int) -> Config:
    d = cfg.to_dict()
    d.update(seed=seed, mutation_enabled=False, min_agents=0, passive=False, init_agents=0)
    if n_total > d["max_agents"]:
        raise ValueError("겨루기 개체 수가 max_agents보다 많다")
    return Config.from_dict(d)


def _inject(world: World, genomes: dict, row: int, tag: int, x: float, y: float, heading: float,
            age: int) -> None:
    s = world.s
    cfg = world.cfg
    j = int(np.flatnonzero(~s.alive)[0])
    aid = int(s.counters[k.C_NEXT_ID])
    s.counters[k.C_NEXT_ID] += 1
    s.counters[k.C_NALIVE] += 1
    s.alive[j] = True
    s.ids[j], s.parent[j], s.gen[j], s.tag[j] = aid, -1, 0, tag
    s.birth_step[j] = world.step_no - 1   # 넣은 스텝에 바로 번식하지 못하게 하는 표시와 겹치지 않게
    s.x[j], s.y[j] = x, y
    s.vx[j] = s.vy[j] = 0.0
    s.heading[j] = heading
    s.energy[j] = cfg.e_start
    s.ledger[k.L_INJECT] += cfg.e_start
    s.age[j] = age
    s.n_off[j] = 0
    s.eaten[j] = s.touch[j] = s.ate[j] = 0.0
    s.h[j] = 0.0
    for name in GENOME:
        getattr(s, name)[j] = genomes[name][row]


def compete(cfg: Config, ga: dict, gb: dict, n_each: int, steps: int, seed: int,
            sample_every: int = 500) -> dict:
    """A(tag 0)와 B(tag 1)를 n_each씩 새 세계에 풀고 개체 수 변화를 본다."""
    rng = np.random.default_rng(seed)
    acfg = arena_config(cfg, seed, 2 * n_each)
    world = World(acfg, populate=False)
    # 빈 세계에 먹이를 절반쯤 미리 깔아 둔다
    for _ in range(int(0.5 * acfg.food_cap * acfg.food_energy / max(acfg.food_inflow, 1e-9))):
        k.spawn_food(0, world.s, world.fp, world.ip)
    ra = rng.choice(len(ga["n_hidden"]), size=n_each, replace=len(ga["n_hidden"]) < n_each)
    rb = rng.choice(len(gb["n_hidden"]), size=n_each, replace=len(gb["n_hidden"]) < n_each)
    px = rng.random(n_each) * acfg.world_w
    py = rng.random(n_each) * acfg.world_h
    ages = rng.integers(0, max(1, acfg.max_age // 2), size=n_each)
    for i in range(n_each):
        # 한 자리에 A 하나 B 하나. 칸 번호가 한쪽에 유리하지 않게 넣는 순서를 번갈아 바꾼다
        pair = [(ga, ra[i], 0), (gb, rb[i], 1)]
        if rng.random() < 0.5:
            pair.reverse()
        for g, row, tag in pair:
            _inject(world, g, int(row), tag, float(px[i]), float(py[i]),
                    float(rng.random() * 2 * math.pi), int(ages[i]))
    series = []
    for t in range(0, steps, sample_every):
        world.run(min(sample_every, steps - t))
        tags = world.s.tag[world.s.alive]
        series.append([world.step_no, int((tags == 0).sum()), int((tags == 1).sum())])
    world.drain_events()
    arr = np.asarray(series, dtype=float)
    # 한쪽이 전멸한 뒤에는 비가 더 변하지 않는다. 전멸한 표본까지만 쓴다
    # (그 뒤까지 넣으면 가장 크게 진 판의 기울기가 0에 가깝게 나온다).
    dead = np.flatnonzero((arr[:, 1] == 0) | (arr[:, 2] == 0))
    upto = arr[:dead[0] + 1] if len(dead) else arr
    keep = upto[upto[:, 0] > steps / 3.0]
    if len(keep) < 3:
        keep = upto     # 버릴 구간 안에서 승부가 났으면 처음부터 쓴다
    y = np.log((keep[:, 2] + 0.5) / (keep[:, 1] + 0.5))
    if len(keep) >= 3:
        r = float(np.polyfit(keep[:, 0], y, 1)[0])
    else:               # 표본이 모자라면 시작(1:1)에서 그 시점까지의 평균 기울기
        r = float(y[-1] / keep[-1, 0])
    return {"n_a": int(arr[-1, 1]), "n_b": int(arr[-1, 2]), "r": r, "series": series,
            "extinct": "a" if arr[-1, 1] == 0 else ("b" if arr[-1, 2] == 0 else "")}


def _split(g: dict, rng: np.random.Generator) -> tuple[dict, dict]:
    n = len(g["n_hidden"])
    p = rng.permutation(n)
    h = n // 2
    return ({key: v[p[:h]] for key, v in g.items()}, {key: v[p[h:]] for key, v in g.items()})


def _stats(vals: list[float]) -> dict:
    v = np.asarray(vals, dtype=float)
    sd = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
    return {"mean": float(v.mean()), "sd": sd, "sem": sd / math.sqrt(len(v)) if len(v) > 1 else float("nan"),
            "n": len(v)}


def assay_sides(a: dict, b: dict, n_each: int = 400, steps: int = 8000, reps: int = 6,
                seed: int = 1000, neutral: bool = True) -> dict:
    check_same_shape(a, b)
    cfg = b["cfg"]
    runs = [compete(cfg, a["genomes"], b["genomes"], n_each, steps, seed + i) for i in range(reps)]
    out = {
        "a": {key: a[key] for key in ("path", "step", "gen_mean")},
        "b": {key: b[key] for key in ("path", "step", "gen_mean")},
        "n_each": n_each, "steps": steps, "reps": reps,
        "r_per_1000": _stats([x["r"] * 1000 for x in runs]),
        "final_counts": [[x["n_a"], x["n_b"]] for x in runs],
        "a_extinct_runs": sum(x["extinct"] == "a" for x in runs),
        "b_extinct_runs": sum(x["extinct"] == "b" for x in runs),
    }
    if neutral:
        rng = np.random.default_rng(seed + 7919)
        neu = []
        for i in range(reps):
            g1, g2 = _split(a["genomes"] if i % 2 == 0 else b["genomes"], rng)
            neu.append(compete(cfg, g1, g2, n_each, steps, seed + 500 + i)["r"] * 1000)
        out["neutral_r_per_1000"] = _stats(neu)
    out["verdict"] = verdict(out)
    return out


def assay(path_a: Path, path_b: Path, n_each: int = 400, steps: int = 8000, reps: int = 6,
          seed: int = 1000, neutral: bool = True) -> dict:
    return assay_sides(load_side(path_a), load_side(path_b), n_each, steps, reps, seed, neutral)


def verdict(res: dict) -> str:
    """임시 기준: 반복 사이 t 검정(양측 95%). 같은 무리끼리의 평균이 0에서 멀면 측정 자체를 의심한다."""
    r = res["r_per_1000"]
    if res.get("a_extinct_runs") == res["reps"]:
        return f"B가 이김(A가 {res['reps']}회 모두 전멸)"
    if res.get("b_extinct_runs") == res["reps"]:
        return f"A가 이김(B가 {res['reps']}회 모두 전멸)"
    if not np.isfinite(r["sem"]):
        return "가를 수 없음(반복 부족)"
    edge = t95(r["n"] - 1) * r["sem"]
    nz = res.get("neutral_r_per_1000")
    warn = ""
    if nz and np.isfinite(nz["sem"]) and abs(nz["mean"]) > t95(nz["n"] - 1) * nz["sem"]:
        warn = " (주의: 같은 무리끼리도 한쪽으로 기울었다)"
    if r["mean"] > edge:
        return "B가 이김" + warn
    if r["mean"] < -edge:
        return "A가 이김" + warn
    return "가를 수 없음" + warn


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def ladder(run_dir: Path, count: int, n_each: int, steps: int, reps: int, seed: int,
           out_path: Path | None = None) -> dict:
    """한 실행의 체크포인트를 고르게 뽑아 모든 쌍을 겨루게 한다(행 = 옛것, 열 = 새것)."""
    cps = [c for c in ckpt.list_checkpoints(Path(run_dir) / "checkpoints") if c[0] > 0]
    if len(cps) < 2:
        raise ValueError("체크포인트가 2개 이상 있어야 한다")
    pick = sorted({int(round(i)) for i in np.linspace(0, len(cps) - 1, min(count, len(cps)))})
    # 살아 있는 실행은 도는 중에 옛 체크포인트를 지운다. 고른 것을 먼저 다 읽어 둔다
    sides = [load_side(cps[i][1]) for i in pick]
    n = len(sides)
    mat = np.zeros((n, n))
    cells = []
    res: dict = {}
    for i in range(n):
        for j in range(i + 1, n):
            r = assay_sides(sides[i], sides[j], n_each, steps, reps, seed, neutral=False)["r_per_1000"]
            mat[i, j] = r["mean"]
            mat[j, i] = -r["mean"]
            cells.append({"old": sides[i]["step"], "new": sides[j]["step"], **r,
                          "significant": bool(np.isfinite(r["sem"]) and abs(r["mean"]) > t95(r["n"] - 1) * r["sem"])})
            res = _ladder_summary(sides, mat, cells)
            if out_path:
                Path(out_path).write_text(json.dumps(res, indent=1))   # 칸마다 저장(중간에 죽어도 남게)
    return res


def _ladder_summary(sides: list[dict], mat: np.ndarray, cells: list[dict]) -> dict:
    n = len(sides)
    # 순환과 진보 가르기(Balduzzi 2019의 이산판): 등급 f_i = 행 평균, 추이 성분 T_ij = f_i - f_j
    win = -mat                       # win[i, j] > 0 이면 i가 j를 이김
    f = win.mean(axis=1)
    trans = f[:, None] - f[None, :]
    total = float((win ** 2).sum())
    steps = np.array([s["step"] for s in sides], dtype=float)
    return {
        "steps": [s["step"] for s in sides],
        "r_new_minus_old_per_1000": mat.tolist(),
        "rating": f.tolist(),
        # 추이 성분 비율은 방향을 모른다(퇴보만 해도 1). 순수 잡음에서도 기대값이 2/n이다.
        "transitive_share": float((trans ** 2).sum() / total) if total > 0 else None,
        "transitive_share_if_noise": 2.0 / n,
        # 등급이 시간에 따라 오르는가(+1이면 늘 새것이 강함, -1이면 늘 옛것이 강함)
        "rating_time_spearman": _spearman(steps, f),
        "cells": cells,
        "cells_done": len(cells), "cells_total": n * (n - 1) // 2,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="성장 측정")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("assay")
    a.add_argument("--a", required=True, type=Path, help="옛 체크포인트(또는 처음 보는 상대). 생물 묶음(pack)도 된다")
    a.add_argument("--b", required=True, type=Path, help="지금 체크포인트. 생물 묶음(pack)도 된다. 겨루는 세계는 B의 설정을 쓴다")
    la = sub.add_parser("ladder")
    la.add_argument("--run", required=True, type=Path)
    la.add_argument("--count", type=int, default=5)
    for p in (a, la):
        p.add_argument("--n", type=int, default=400, help="한쪽 개체 수(둘을 합쳐 수용력 근처가 되게)")
        p.add_argument("--steps", type=int, default=8000)
        p.add_argument("--reps", type=int, default=6)
        p.add_argument("--seed", type=int, default=1000)
        p.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "assay":
        res = assay(args.a, args.b, args.n, args.steps, args.reps, args.seed)
        r = res["r_per_1000"]
        print(f"A: step {res['a']['step']} (평균 세대 {res['a']['gen_mean']:.1f})  "
              f"B: step {res['b']['step']} (평균 세대 {res['b']['gen_mean']:.1f})")
        print(f"r(B-A) / 1000스텝 = {r['mean']:+.4f} ± {r['sem']:.4f} (표준오차, {r['n']}회)")
        if "neutral_r_per_1000" in res:
            nz = res["neutral_r_per_1000"]
            print(f"같은 무리끼리(잡음): 평균 {nz['mean']:+.4f}, 표준편차 {nz['sd']:.4f}")
        print("끝 개체 수 [A, B]:", res["final_counts"])
        print("판정:", res["verdict"])
        if args.out:
            args.out.write_text(json.dumps(res, indent=1))
    else:
        res = ladder(args.run, args.count, args.n, args.steps, args.reps, args.seed, args.out)
        print("스텝:", res["steps"])
        print("r(새것-옛것)/1000스텝, 행=옛것 열=새것")
        for row in res["r_new_minus_old_per_1000"]:
            print("  " + " ".join(f"{v:+7.3f}" for v in row))
        sig = sum(c["significant"] for c in res["cells"])
        print(f"t 검정(95%)을 넘은 칸: {sig} / {len(res['cells'])}")
        print("등급(클수록 강함):", [round(v, 3) for v in res["rating"]])
        print(f"등급과 시간의 순위 상관: {res['rating_time_spearman']:+.2f} (+1이면 늘 새것이 강함)")
        print(f"추이 성분 비율: {res['transitive_share']:.3f} (순수 잡음이면 {res['transitive_share_if_noise']:.2f}쯤. "
              "방향은 순위 상관으로 본다)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
