"""가려 먹기를 "놓인 먹이"로 잰다: 체크포인트에서 독이 바뀌는 순간 앞부터 다시 돌려 구간별로 본다.

    python wayfinder/stage2-4/placed.py <체크포인트.npz>... [--steps 3000] [--win 300] [--every 2] [--set poison=0]

--set을 주면 같은 체크포인트를 그 설정으로 바꿔 돌린다(대조: 독 0이면 독이 바뀌어도 아무 일이 없다).

먹은 것 가운데 독 = 구간에서 먹은 것 가운데 독의 비율. 난 먹이가 다 먹히는 세계에서는 가려 먹어도 0.5에 묶인다.
놓인 먹이 가운데 독 = 구간 동안 바닥에 놓인 먹이 가운데 지금 독인 종류의 비율(장면 평균).
  독이 바뀐 뒤 이 값이 다시 절반 위로 올라오면(독이 남게 되면) 독을 따라 가려 먹는 것이고,
  뒤집힌 채로 머물면 독이 아니라 모양(종류)을 가리는 것이다(results.md "화면을 짓다 알게 된 것").
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kkumteul.replay import record  # noqa: E402


def windows(rep, win: int) -> list[dict]:
    steps = rep.steps
    out = []
    lo = int(steps[0])
    while lo < steps[-1]:
        idx = np.nonzero((steps >= lo) & (steps < lo + win))[0]
        if len(idx) < 2:
            break
        eat = int(rep.c_eaten[idx].sum())          # 장면마다 그 사이에 먹은 수(누적 아님)
        poi = int(rep.c_poisoned[idx].sum())
        good = np.bincount(rep.good_kind[idx].astype(np.int64), minlength=2)
        out.append({
            "from": lo, "good": "A" if good[0] >= good[1] else "B", "mixed": bool(good.min() > 0),
            "eaten": eat, "eaten_poison": round(poi / eat, 3) if eat else None,
            "placed": round(float(rep.n_food[idx].mean())),
            "placed_poison": round(float(rep.blind[idx].mean()), 3),
        })
        lo += win
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", type=Path, nargs="+")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--win", type=int, default=300)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--set", action="append", default=[], metavar="key=value")
    a = ap.parse_args()
    for c in a.ckpts:
        rep = record(c, steps=a.steps, every=a.every, from_flip=True, overrides=a.set)
        print(f"== {c.parent.parent.name} @ {rep.meta.get('ckpt_step')} {' '.join(a.set)}  (구간 {a.win}스텝, 좋은 먹이 A·B는 그 구간에 독이 아닌 쪽)")
        for n in rep.meta.get("notes", []):
            print("  주의:", n)
        flips = [f["step"] for f in rep.meta.get("flips", [])]
        print("  독이 바뀐 스텝:", flips)
        print(f"  {'시작':>9} {'좋은':>4} {'먹은 수':>7} {'먹은것 독':>9} {'놓인 수':>7} {'놓인것 독':>9}")
        for w in windows(rep, a.win):
            mark = " (바뀜)" if w["mixed"] else ""
            print(f"  {w['from']:>9} {w['good']:>4} {w['eaten']:>7} {str(w['eaten_poison']):>9} {w['placed']:>7} {w['placed_poison']:>9}{mark}")


if __name__ == "__main__":
    main()
