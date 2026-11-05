"""실행의 stats.jsonl을 창(기본 10만 스텝)으로 묶어 본다: 개체 수, 하한 비율, 재주입, 독 비율, 뇌, 학습률.

    python wayfinder/stage2-4/windows.py runs/k4-s2 [--from 1300000] [--win 100000] [--floor 105]

하한 비율 = 창 안의 통계 줄 가운데 개체 수가 floor 이하인 줄의 비율.
독 비율 = 창에서 먹은 것 가운데 독의 비율(누적 카운터의 차). 먹이 종류가 없는 세계는 null.
  주의: 난 먹이가 다 먹히는 세계에서는 가려 먹어도 0.5에 묶인다(results.md "화면을 짓다 알게 된 것").
공격 세계에는 창당 공격 횟수(1,000스텝당)와 사망 가운데 맞아 죽은 비율이 붙는다.
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
        d = lambda k: last.get(k, 0) - prev.get(k, 0)     # 누적 카운터의 차(없는 필드는 0)
        eat = d("eaten")
        w = {
            "end": last["step"],
            "pop": round(sum(r["pop"] for r in part) / len(part)),
            "floor_frac": round(sum(r["pop"] <= floor for r in part) / len(part), 2),
            "reseeds": d("reseeds"),
            "poison_ratio": round(d("poisoned") / eat, 3) if eat and "poisoned" in last else None,
            "hidden": round(last.get("hidden_mean", float("nan")), 2),
            "eta": round(last["eta_mean"], 4) if "eta_mean" in last else None,
            "gen": round(last["gen_mean"]),
        }
        if "attacks" in last:                               # 공격 세계(preset=conflict)
            deaths = sum(d(k) for k in ("deaths_starve", "deaths_age", "deaths_disaster", "deaths_killed"))
            w["attacks_per_1k"] = round(d("attacks") / max(last["step"] - prev["step"], 1) * 1000)
            w["killed_frac"] = round(d("deaths_killed") / deaths, 3) if deaths else None
        out.append(w)
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
