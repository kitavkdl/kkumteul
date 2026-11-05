"""장치 실험 비교: 같은 체크포인트에서 갈라 "장치를 켠 세계"와 "끈 세계"를 돌린 뒤, 두 세계의 생물을 겨루게 한다.

    python3 -m kkumteul.compare --control runs/x-ctrl --treatment runs/x-s2 [--at STEP] [--out card.json]

겨루기는 두 가지 규칙에서 한다.
- 켠 규칙(treatment의 설정): 장치가 있는 세계에서, 장치와 함께 진화한 쪽이 더 잘 사는가.
- 끈 규칙(control의 설정): 장치가 없는 세계로 데려가도 더 잘 사는가(장치 없이도 통하는 능력인가).

켠 규칙에서만 이기면 "장치를 쓰게 됐다"는 쪽의 증거이고, 양쪽에서 다 이기거나 다 지면 장치와 무관한 차이다.
형질의 추세(학습률, 신호 세기, 거름 양)는 참고로만 적는다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import checkpoint as ckpt
from . import measure

TRAITS = ("pop", "gen_mean", "hidden_mean", "speed_mean", "eta_mean", "alpha_abs_mean", "hebb_abs_mean",
          "asleep_frac", "signal_abs_mean", "gifts", "soil_total", "fertile_food", "trail_mean", "reseeds")
STAGE_FLAGS = ("mind_enabled", "social_enabled", "niche_enabled")


def _stats_rows(run_dir: Path) -> list[dict]:
    rows = []
    with open(Path(run_dir) / "stats.jsonl") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _checkpoint_at(run_dir: Path, at: int | None) -> Path:
    cps = ckpt.list_checkpoints(Path(run_dir) / "checkpoints")
    if at is not None:
        cps = [c for c in cps if c[0] == at]
    if not cps:
        raise FileNotFoundError(f"체크포인트를 못 찾았다: {run_dir} at={at}")
    return cps[-1][1]


def compare(control: Path, treatment: Path, at: int | None = None, n_each: int = 400, steps: int = 8000,
            reps: int = 6, seed: int = 2000) -> dict:
    a = measure.load_side(_checkpoint_at(control, at))
    b = measure.load_side(_checkpoint_at(treatment, at))
    if a["step"] != b["step"]:
        raise ValueError(f"두 실행의 스텝이 다르다({a['step']} != {b['step']}). --at으로 같은 스텝을 고른다")
    measure.check_same_shape(a, b)
    on = measure.assay_sides(a, b, n_each, steps, reps, seed)             # 켠 규칙: B = 켠 쪽
    off = measure.assay_sides(b, a, n_each, steps, reps, seed + 100)      # 끈 규칙: B = 끈 쪽 → 부호를 뒤집어 읽는다
    r_off = {**off["r_per_1000"], "mean": -off["r_per_1000"]["mean"]}
    out = {
        "step": a["step"],
        "control": str(control), "treatment": str(treatment),
        "flags": {f: [getattr(a["cfg"], f), getattr(b["cfg"], f)] for f in STAGE_FLAGS},
        "treatment_minus_control_r_per_1000": {
            "in_treatment_rules": on["r_per_1000"], "verdict_treatment_rules": _name(on["verdict"], "끈 쪽이", "켠 쪽이"),
            "neutral_treatment_rules": on.get("neutral_r_per_1000"),
            "in_control_rules": r_off, "verdict_control_rules": _name(off["verdict"], "켠 쪽이", "끈 쪽이"),
            "neutral_control_rules": off.get("neutral_r_per_1000"),
        },
        "final_counts_treatment_rules": on["final_counts"],          # [끈 쪽, 켠 쪽]
        "final_counts_control_rules": [[y, x] for x, y in off["final_counts"]],
        "traits": {},
    }
    ra, rb = _stats_rows(control), _stats_rows(treatment)
    for name, rows in (("control", ra), ("treatment", rb)):
        first, last = rows[0], rows[-1]
        out["traits"][name] = {key: [first.get(key), last.get(key)] for key in TRAITS
                               if first.get(key) is not None or last.get(key) is not None}
    out["reading"] = _reading(out)
    return out


def _name(v: str, a_label: str, b_label: str) -> str:
    """겨루기 판정 문구의 A·B를 '켠 쪽/끈 쪽'으로 바꾼다(끈 규칙 겨루기는 A=켠 쪽, B=끈 쪽으로 돌린다)."""
    return v.replace("A가", a_label).replace("B가", b_label)


def _reading(out: dict) -> str:
    t = out["treatment_minus_control_r_per_1000"]
    on_v = t["verdict_treatment_rules"]
    off_v = t["verdict_control_rules"]
    on_win = on_v.startswith("켠 쪽이 이김")
    on_lose = on_v.startswith("끈 쪽이 이김")
    off_win = off_v.startswith("켠 쪽이 이김")
    off_lose = off_v.startswith("끈 쪽이 이김")
    if on_win and not off_win:
        return "켠 규칙에서만 켠 쪽이 이겼다. 장치와 함께 진화한 것이 장치가 있을 때 값을 한다."
    if on_win and off_win:
        return "양쪽 규칙에서 켠 쪽이 이겼다. 장치와 무관한 능력 차이일 수 있다(장치를 썼다는 증거로는 약하다)."
    if on_lose and off_lose:
        return "양쪽 규칙에서 끈 쪽이 이겼다. 장치를 켠 세계의 진화가 뒤처졌다(느린 계산·비용 때문일 수 있다)."
    if on_lose:
        return "켠 규칙에서조차 끈 쪽이 이겼다. 장치와 함께 진화한 것이 값을 못 한다."
    if off_lose and not on_lose:
        return "끈 규칙에서는 끈 쪽이 이기고 켠 규칙에서는 가를 수 없다. 켠 쪽이 장치에 기대게 됐을 수 있다."
    return "가를 수 없다."


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="장치 실험 비교")
    ap.add_argument("--control", required=True, type=Path)
    ap.add_argument("--treatment", required=True, type=Path)
    ap.add_argument("--at", type=int, default=None, help="견줄 체크포인트의 스텝(없으면 각자 가장 최근)")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    res = compare(args.control, args.treatment, args.at, args.n, args.steps, args.reps, args.seed)
    t = res["treatment_minus_control_r_per_1000"]
    print(f"스텝 {res['step']}, 장치(끈 쪽, 켠 쪽): {res['flags']}")
    for label, key, vk in (("켠 규칙", "in_treatment_rules", "verdict_treatment_rules"),
                           ("끈 규칙", "in_control_rules", "verdict_control_rules")):
        r = t[key]
        print(f"{label}: r(켠 쪽 − 끈 쪽)/1000스텝 = {r['mean']:+.4f} ± {r['sem']:.4f} ({r['n']}회) → {t[vk]}")
    print("끝 개체 수 [끈 쪽, 켠 쪽] 켠 규칙:", res["final_counts_treatment_rules"])
    print("끝 개체 수 [끈 쪽, 켠 쪽] 끈 규칙:", res["final_counts_control_rules"])
    print("형질(처음 → 끝, 참고용)")
    for name in ("control", "treatment"):
        print(f"  {name}: " + ", ".join(f"{key} {_fmt(v[0])}→{_fmt(v[1])}" for key, v in res["traits"][name].items()))
    print("읽기:", res["reading"])
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


def _fmt(v) -> str:
    if v is None:
        return "–"
    return f"{v:.4g}" if isinstance(v, float) else str(v)


if __name__ == "__main__":
    sys.exit(main())
