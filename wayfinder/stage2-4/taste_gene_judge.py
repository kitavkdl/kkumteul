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

observe (단서, 판정 아님):
    끼니 계수기는 체크포인트에 저장되지 않으므로 체크포인트 설정 그대로 --steps(기본 3,000) 더 돌려 센다.
    무리 크기, 실제 배움 속도 평균·사분위(돌리기 전), 그동안 먹힌 알 가운데 독 비율(세계 계수기 C_POISONED / C_EATEN 차이),
    끝에 산 개체의 첫 10끼 독 비율(n_poison10 / min(n_meal, 10) 합 — 읽은 뒤 태어난 개체는 태어난 뒤부터, 그 전부터 산
    개체는 읽은 뒤부터 센 끼니). 독 비율 0.5 근처면 가려 먹지 못하는 것이다.

    PYTHONPATH=. python wayfinder/stage2-4/taste_gene_judge.py rate-zero <체크포인트.npz> [--gene taste|eta]
        [--n 400] [--steps 8000] [--reps 6] [--seed 6000] [--json <출력.json>]
    PYTHONPATH=. python wayfinder/stage2-4/taste_gene_judge.py observe <체크포인트.npz> [--steps 3000] [--json <출력.json>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from kkumteul import checkpoint as ckpt  # noqa: E402
from kkumteul import kernels as k  # noqa: E402
from kkumteul import knockout, measure  # noqa: E402
from kkumteul.config import FIRST_MEALS  # noqa: E402


def zeroed_side(side: dict, gene: str = "taste") -> dict:
    """배움을 없앤 한 벌(유전체 사본). taste: taste_g = −taste_rate(실제 속도 0), eta: eta = 0."""
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
    else:
        raise ValueError(f"모르는 유전자: {gene} (taste 또는 eta)")
    return dict(side, genomes=g, path=f"{side['path']} ({gene} 배움 0)")


def rate_zero(path: Path, gene: str = "taste", n: int = 400, steps: int = 8000, reps: int = 6,
              seed: int = 6000) -> dict:
    side = measure.load_side(path)
    cfg = side["cfg"]
    a = zeroed_side(side, gene)
    res = measure.assay_sides(a, side, n, steps, reps, seed, neutral=False)
    res["gene"] = gene
    res["neutral_r_per_1000"] = knockout._neutral(side, cfg, n, steps, reps, seed)
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
    e0, p0 = int(s.counters[k.C_EATEN]), int(s.counters[k.C_POISONED])
    world.run(steps)
    a = s.alive
    eaten = int(s.counters[k.C_EATEN]) - e0
    out["eaten"] = eaten
    out["poison_frac"] = float((int(s.counters[k.C_POISONED]) - p0) / eaten) if eaten else None
    first = int(np.minimum(s.n_meal[a], FIRST_MEALS).sum())
    out["poison_frac_first10"] = float(s.n_poison10[a].sum() / first) if first else None
    out["pop_end"] = int(a.sum())
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rate-zero")
    r.add_argument("ckpt", type=Path)
    r.add_argument("--gene", choices=("taste", "eta"), default="taste")
    r.add_argument("--n", type=int, default=400)
    r.add_argument("--steps", type=int, default=8000)
    r.add_argument("--reps", type=int, default=6)
    r.add_argument("--seed", type=int, default=6000)
    r.add_argument("--json", type=Path)
    o = sub.add_parser("observe")
    o.add_argument("ckpt", type=Path)
    o.add_argument("--steps", type=int, default=3000)
    o.add_argument("--json", type=Path)
    args = p.parse_args(argv)
    if args.cmd == "rate-zero":
        out = rate_zero(args.ckpt, args.gene, args.n, args.steps, args.reps, args.seed)
    else:
        out = observe(args.ckpt, args.steps)
    text = json.dumps(out, ensure_ascii=False, indent=1, default=float)
    if args.json:
        args.json.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
