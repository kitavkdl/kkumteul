"""실행 폴더의 기록(사건·통계)만으로 내는 상시 지표.

    python -m kkumteul.analyze runs/main

- 세대: 살아 있는 개체의 평균 계보 깊이. 현실 하루에 몇 세대가 쌓이는지도 낸다.
- 검산: 출생 때 부모 나이의 평균(T_parent). 정상 상태면 d(평균 세대)/d스텝 ≈ 1/T_parent.
- 출생 비: 자연 출생 / (자연 출생 + 재주입). 1.0에 가까워야 "스스로 유지되는 세계"다.
- Fitness Transmission(Miconi 2008): 부모와 자식의 손주 수 상관. 번식 성공이 상속되면 양수가 된다.
  기준선은 0이 아니다. 유전 변이가 전혀 없어도 평균 +0.01쯤 나온다(독립 검토 2026-11-04 실측).
  대조 실행(`run branch --passive`)에 같은 명령을 돌린 값과 비교해서 읽는다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import events
from . import kernels as k


MAX_EVENT_BYTES = 1 << 30   # 한 번에 읽는 기록의 상한(1GB)


def load_run(run_dir: Path, last_steps: int | None = None) -> dict:
    """실행 기록을 읽는다. last_steps를 주면 마지막 그만큼의 스텝만 읽는다(긴 실행용)."""
    run_dir = Path(run_dir)
    rows = []
    p = run_dir / "stats.jsonl"
    if p.exists():
        with open(p) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    cfg = json.loads((run_dir / "config.json").read_text())
    min_step = 0
    if last_steps is not None and rows:
        min_step = max(0, rows[-1]["step"] - last_steps)
        rows = [r for r in rows if r["step"] >= min_step]
    seg = cfg.get("event_segment_steps", 2000000)
    return {
        "config": cfg,
        "meta": json.loads((run_dir / "run.json").read_text()),
        "births": events.load(run_dir, "births", seg, min_step, MAX_EVENT_BYTES),
        "deaths": events.load(run_dir, "deaths", seg, min_step, MAX_EVENT_BYTES),
        "stats": rows,
        "min_step": min_step,
    }


def bred_kind(meta: dict) -> int:
    """이 실행에서 "낳아서 생긴 출생"의 종류. 대조 실행은 출생이 전부 BIRTH_PASSIVE다."""
    return k.BIRTH_PASSIVE if meta.get("kind") == "passive" else k.BIRTH_NATURAL


def lineage_tables(births: np.ndarray, kind: int = k.BIRTH_NATURAL):
    """id 순으로 정렬된 출생 기록에서 (부모의 줄 번호, 자식 수, 손주 수)를 만든다. 기록에 없는 부모는 -1."""
    ids = births["id"]
    pidx = np.searchsorted(ids, births["parent"])
    pidx[pidx >= len(ids)] = 0
    known = ids[pidx] == births["parent"]
    pidx = np.where(known, pidx, -1)
    natural = births["kind"] == kind
    ok = natural & (pidx >= 0)
    n_child = np.bincount(pidx[ok], minlength=len(ids))
    n_grand = np.zeros(len(ids), dtype=np.int64)
    np.add.at(n_grand, pidx[ok], n_child[ok])
    return pidx, n_child, n_grand, ok


def fitness_transmission(births: np.ndarray, window: int, settle: int, kind: int = k.BIRTH_NATURAL) -> list[dict]:
    """구간별 부모-자식 손주 수 상관. 손주 수가 아직 덜 쌓인 끝부분(settle 스텝)은 뺀다."""
    if len(births) < 10:
        return []
    pidx, n_child, n_grand, ok = lineage_tables(births, kind)
    step = births["step"]
    last = int(step.max()) - settle
    out = []
    t0 = int(step.min())
    while t0 + window <= last:
        sel = np.flatnonzero(ok & (step >= t0) & (step < t0 + window))
        if len(sel) >= 30:
            c = sel
            p = pidx[sel]
            n_c = n_grand[c].astype(float)
            # 의존성 제거: 부모의 손주 수에서 이 자식이 낳은 몫을 뺀다
            n_p = (n_grand[p] - n_child[c]).astype(float)
            # 정규화: 같은 구간 다른 부모들의 손주 합으로 나눈다
            uniq = np.unique(p)
            total = float(n_grand[uniq].sum())
            denom = total - n_grand[p]
            n_p = np.where(denom > 0, n_p / np.maximum(denom, 1e-12), 0.0)
            if n_p.std() > 0 and n_c.std() > 0:
                out.append({"start": t0, "pairs": int(len(sel)),
                            "corr": float(np.corrcoef(n_p, n_c)[0, 1])})
        t0 += window
    return out


def parent_age(births: np.ndarray, kind: int = k.BIRTH_NATURAL) -> float | None:
    pidx, _, _, ok = lineage_tables(births, kind)
    if not ok.any():
        return None
    return float((births["step"][ok] - births["step"][pidx[ok]]).mean())


def report(run_dir: Path, last_steps: int | None = None) -> dict:
    d = load_run(run_dir, last_steps)
    cfg, births, deaths, stats = d["config"], d["births"], d["deaths"], d["stats"]
    kind = bred_kind(d["meta"])
    out: dict = {"run": str(run_dir), "kind": d["meta"].get("kind"), "from_step": d["min_step"]}
    if not stats:
        return out
    first, last = stats[0], stats[-1]
    steps = last["step"] - first["step"]
    # 꺼져 있던 시간을 빼려고 줄마다 적힌 속도로 돈 시간을 더한다
    wall = sum((r.get("w_steps") or cfg["stats_every"]) / r["sps"] for r in stats[1:] if r.get("sps"))
    out["step"] = last["step"]
    out["pop"] = last["pop"]
    out["pop_min"] = min(r["pop"] for r in stats)
    out["pop_max"] = max(r["pop"] for r in stats)
    out["steps_per_sec"] = round(steps / wall, 1) if wall > 0 else None
    out["gen_mean"] = last.get("gen_mean")
    d_gen = (last.get("gen_mean") or 0) - (first.get("gen_mean") or 0)
    out["steps_per_generation"] = round(steps / d_gen, 1) if d_gen > 0 else None
    out["generations_per_day"] = round(d_gen / wall * 86400, 1) if wall > 0 and d_gen > 0 else None
    t_par = parent_age(births, kind)
    out["parent_age_at_birth"] = round(t_par, 1) if t_par else None
    natural = int((births["kind"] == kind).sum())
    reseed = int((births["kind"] == k.BIRTH_RESEED).sum())
    out["births_natural"] = natural
    out["reseeds"] = reseed
    out["birth_ratio"] = round(natural / (natural + reseed), 5) if natural + reseed else None
    rs = births["step"][births["kind"] == k.BIRTH_RESEED]
    out["steps_since_last_reseed"] = int(last["step"] - rs.max()) if len(rs) else int(steps)
    if len(deaths):
        out["life_mean"] = round(float(deaths["age"].mean()), 1)
        out["death_starve_frac"] = round(float((deaths["cause"] == k.DEATH_STARVE).mean()), 3)
    out["hidden_mean_first_last"] = [first.get("hidden_mean"), last.get("hidden_mean")]
    out["energy_residual_rel"] = last.get("e_residual_rel")
    ft = fitness_transmission(births, 2 * cfg["max_age"], 3 * cfg["max_age"], kind)
    if ft:
        c = np.array([x["corr"] for x in ft])
        out["fitness_transmission"] = {
            "windows": len(c), "mean": round(float(c.mean()), 4),
            "positive_frac": round(float((c > 0).mean()), 3),
            "last5_mean": round(float(c[-5:].mean()), 4),
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="실행 기록 분석")
    ap.add_argument("run", type=Path)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--last-steps", type=int, default=None, help="마지막 이만큼의 스텝만 본다(긴 실행용)")
    args = ap.parse_args(argv)
    r = report(args.run, args.last_steps)
    if args.json:
        print(json.dumps(r, indent=1))
        return 0
    label = {
        "from_step": "분석 시작 스텝", "step": "지금 스텝", "pop": "개체 수", "pop_min": "개체 수 최소", "pop_max": "개체 수 최대",
        "steps_per_sec": "초당 스텝", "gen_mean": "평균 세대(계보 깊이)",
        "steps_per_generation": "한 세대에 걸린 스텝", "generations_per_day": "현실 하루당 세대",
        "parent_age_at_birth": "출생 때 부모 나이(검산용)", "births_natural": "출생(재주입 제외)",
        "reseeds": "재주입", "birth_ratio": "출생 비(1.0이 목표)",
        "steps_since_last_reseed": "마지막 재주입 뒤 스텝", "life_mean": "평균 수명(스텝)",
        "death_starve_frac": "굶어 죽은 비율", "hidden_mean_first_last": "평균 활성 뉴런(처음, 지금)",
        "energy_residual_rel": "에너지 장부 오차(상대)", "fitness_transmission": "Fitness Transmission(대조 실행 값과 비교해서 읽는다)",
    }
    for key, val in r.items():
        print(f"{label.get(key, key)}: {val}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
