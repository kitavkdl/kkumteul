"""실행의 stats.jsonl을 창(기본 10만 스텝)으로 묶어 본다: 개체 수, 하한 비율, 재주입, 독 비율, 뇌, 학습률.

    python wayfinder/stage2-4/windows.py runs/k4-s2 [--from 1300000] [--win 100000] [--floor 105]

하한 비율 = 창 안의 통계 줄 가운데 개체 수가 floor 이하인 줄의 비율.
독 비율 = 창에서 먹은 것 가운데 독의 비율(누적 카운터의 차).
"""
import argparse
import json
from pathlib import Path


def windows(run: Path, start: int, win: int, floor: int) -> list[dict]:
    rows = [json.loads(x) for x in open(run / "stats.jsonl") if x.strip()]
    rows = [r for r in rows if r["step"] >= start]
    out = []
    prev = rows[0] if rows else None
    lo = start
    while prev is not None:
        hi = lo + win
        part = [r for r in rows if lo < r["step"] <= hi]
        if not part:
            break
        last = part[-1]
        eat = last["eaten"] - prev["eaten"]
        poi = last["poisoned"] - prev["poisoned"]
        out.append({
            "end": last["step"],
            "pop": round(sum(r["pop"] for r in part) / len(part)),
            "floor_frac": round(sum(r["pop"] <= floor for r in part) / len(part), 2),
            "reseeds": last["reseeds"] - prev["reseeds"],
            "poison_ratio": round(poi / eat, 3) if eat else None,
            "hidden": round(last.get("hidden_mean", float("nan")), 2),
            "eta": round(last["eta_mean"], 4) if "eta_mean" in last else None,
            "gen": round(last["gen_mean"]),
        })
        prev = last
        lo = hi
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--from", dest="start", type=int, default=1300000)
    ap.add_argument("--win", type=int, default=100000)
    ap.add_argument("--floor", type=int, default=105)
    a = ap.parse_args()
    for w in windows(a.run, a.start, a.win, a.floor):
        print(json.dumps(w, ensure_ascii=False))


if __name__ == "__main__":
    main()
