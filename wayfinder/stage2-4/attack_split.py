"""공격이 퍼진 것을 가른다: 같은 체크포인트의 같은 생물을 두 벌로 만들어 한쪽만 공격을 막고 겨룬다.

    python wayfinder/stage2-4/attack_split.py <체크포인트.npz> [--n 400] [--steps 8000] [--reps 6] [--share 1000]

- 막은 쪽(A): 공격 출력 편향 bout[:, ACT_ATTACK]을 -10으로. 공격은 tanh(출력) > 0일 때만 일어나므로
  이 정도면 다른 입력이 아무리 밀어도 거의 켜지지 않는다. 나머지 유전체는 그대로다.
- 원래 쪽(B): 손대지 않는다.
- 공격 규칙(체크포인트의 설정)과, 공격을 끈 규칙(conflict_enabled=false)에서 겨룬다.
  끈 규칙에서는 공격 출력이 쓰이지 않으므로 승부가 나지 않아야 한다(뻔한 대조).
- --share: 체크포인트에서 이만큼 돌리며 개체마다 공격을 낸 스텝 수를 세어, 몇 마리가 공격을 도맡는지 본다.
결과(JSON)는 표준 출력으로.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kkumteul import checkpoint as ckpt  # noqa: E402
from kkumteul import measure  # noqa: E402
from kkumteul.config import ACT_ATTACK  # noqa: E402

CALM_BIAS = -10.0


def sides(path: Path):
    world, _ = ckpt.load(path)
    orig = measure.side_from_world(world, str(path))
    g = {k: v.copy() for k, v in orig["genomes"].items()}
    g["bout"][:, ACT_ATTACK] = CALM_BIAS
    calm = dict(orig, genomes=g, path=f"{path} (공격 막음)")
    return world, calm, orig


def share(path: Path, steps: int) -> dict:
    """체크포인트에서 steps만큼 돌리며 개체마다 공격을 낸 스텝 수를 센다."""
    world, _ = ckpt.load(path)
    s = world.s
    cnt: dict[int, int] = {}
    alive_steps: dict[int, int] = {}
    attacks0 = None
    for _ in range(steps):
        world.run(1)
        live = np.flatnonzero(s.alive)
        on = live[s.act[live, ACT_ATTACK] > 0.0]
        for j in live.tolist():
            alive_steps[int(s.ids[j])] = alive_steps.get(int(s.ids[j]), 0) + 1
        for j in on.tolist():
            cnt[int(s.ids[j])] = cnt.get(int(s.ids[j]), 0) + 1
    world.drain_events()
    ids = list(alive_steps)
    c = np.array([cnt.get(i, 0) for i in ids], dtype=float)
    a = np.array([alive_steps[i] for i in ids], dtype=float)
    order = np.argsort(-c)
    tot = c.sum()
    top = lambda q: float(c[order[:max(1, int(round(q * len(c))))]].sum() / tot) if tot else 0.0  # noqa: E731
    rate = c / a
    return {
        "steps": steps, "agents_seen": len(ids), "attack_steps": int(tot),
        "attack_steps_per_agent_step": float(tot / a.sum()),
        "never_attacked_frac": float((c == 0).mean()),
        "share_top1pct": top(0.01), "share_top10pct": top(0.10), "share_top25pct": top(0.25),
        # 오래 본 개체(100스텝 넘게 산 것)만으로 공격하는 스텝의 비율 분포
        "rate_quantiles_lived100": {q: float(np.quantile(rate[a > 100], q)) for q in (0.5, 0.75, 0.9, 0.99)} if (a > 100).any() else {},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=3000)
    ap.add_argument("--share", type=int, default=1000)
    args = ap.parse_args(argv)

    out = {"ckpt": str(args.ckpt), "calm_bias": CALM_BIAS}
    if args.share:
        out["share"] = share(args.ckpt, args.share)
        print(json.dumps({"share": out["share"]}, ensure_ascii=False), file=sys.stderr, flush=True)
    world, calm, orig = sides(args.ckpt)
    out["step"] = int(world.step_no)
    cfg_on = orig["cfg"]
    cfg_off = cfg_on.with_overrides(["conflict_enabled=false"])
    for name, cfg in (("attack_rules", cfg_on), ("no_attack_rules", cfg_off)):
        a = dict(calm, cfg=cfg)
        b = dict(orig, cfg=cfg)
        res = measure.assay_sides(a, b, args.n, args.steps, args.reps, args.seed)
        res["verdict"] = measure.verdict(res, "막은 쪽이", "원래(공격하는) 쪽이")
        out[name] = res
        print(json.dumps({name: {k: res[k] for k in ("r_per_1000", "neutral_r_per_1000", "final_counts", "verdict")}},
                         ensure_ascii=False), file=sys.stderr, flush=True)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
