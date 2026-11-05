"""장치 실험 비교: 같은 체크포인트에서 갈라 "장치를 켠 세계"와 "끈 세계"를 돌린 뒤, 두 세계의 생물을 겨루게 한다.

    python3 -m kkumteul.compare --control runs/x-ctrl --treatment runs/x-s2 [--at STEP] [--out card.json]

겨루기는 두 가지 규칙에서 한다.
- 켠 규칙(treatment의 설정): 장치가 있는 세계에서, 장치와 함께 진화한 쪽이 더 잘 사는가.
- 끈 규칙(control의 설정): 장치가 없는 세계로 데려가도 더 잘 사는가(장치 없이도 통하는 능력인가).

켠 규칙에서만 이기면 "켠 규칙에 적응했다"는 뜻이다. 그것만으로 장치를 쓰게 됐다고 할 수는 없다
(2026-11-05 첫 측정에서 켠 쪽은 켠 규칙에서 이겼지만 학습률은 0으로 줄어 있었다). 양쪽에서 다 이기거나 다 지면 장치와 무관한 차이다.
형질의 추세(학습률, 신호 세기, 거름 양)는 참고로만 적는다.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

from . import checkpoint as ckpt
from . import measure

TRAITS = ("pop", "gen_mean", "hidden_mean", "speed_mean", "eta_mean", "alpha_abs_mean", "hebb_abs_mean",
          "asleep_frac", "signal_abs_mean", "gifts", "soil_total", "fertile_food", "trail_mean", "reseeds",
          "attacks", "poisoned", "food_replaced")
STAGE_FLAGS = ("mind_enabled", "social_enabled", "niche_enabled", "food_kinds_enabled", "conflict_enabled")


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
        "config_diff": _config_diff(a["cfg"], b["cfg"]),              # 값이 다른 설정 전부 [끈 쪽, 켠 쪽]
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
        # '끝'은 겨룬 체크포인트의 시점이다(실행이 그 뒤로 더 돌았어도 그때의 형질을 적는다)
        upto = [r for r in rows if r.get("step", 0) <= out["step"]] or rows
        first, last = upto[0], upto[-1]
        out["traits"][name] = {key: [first.get(key), last.get(key)] for key in TRAITS
                               if first.get(key) is not None or last.get(key) is not None}
    out["reading"] = _reading(out)
    return out


def _name(v: str, a_label: str, b_label: str) -> str:
    """겨루기 판정 문구의 A·B를 '켠 쪽/끈 쪽'으로 바꾼다(끈 규칙 겨루기는 A=켠 쪽, B=끈 쪽으로 돌린다)."""
    return v.replace("A가", a_label).replace("B가", b_label)


def _config_diff(ca, cb) -> dict:
    da, db = ca.to_dict(), cb.to_dict()
    return {key: [da.get(key), db.get(key)] for key in sorted(set(da) | set(db)) if da.get(key) != db.get(key)}


_WARN = re.compile(r"\(주의:[^)]*\)")
_LIMIT = " 실행이 한 쌍뿐이라 장치의 효과와 계통의 운을 가르지 못한다(반복은 겨루기판의 잡음만 잰다)."


def _rules_differ(t: dict) -> bool:
    """두 규칙에서 잰 r(켠 쪽 − 끈 쪽)의 차이가 잡음보다 큰가. 한쪽만 유의하다는 것은 차이의 증거가 아니다."""
    a, b = t["in_treatment_rules"], t["in_control_rules"]
    se = math.hypot(a["sem"], b["sem"])
    if not math.isfinite(se):
        return False
    return abs(a["mean"] - b["mean"]) > measure.t95(min(a["n"], b["n"]) - 1) * se


def _reading(out: dict) -> str:
    t = out["treatment_minus_control_r_per_1000"]
    on_v = t["verdict_treatment_rules"]
    off_v = t["verdict_control_rules"]
    on_win = on_v.startswith("켠 쪽이 이김")
    on_lose = on_v.startswith("끈 쪽이 이김")
    off_win = off_v.startswith("켠 쪽이 이김")
    off_lose = off_v.startswith("끈 쪽이 이김")
    adapted = ("장치를 '쓰게 됐다'는 뜻은 아니다(장치가 주는 입력과 제약에 맞춰졌을 뿐일 수 있다). "
               "유전자에서 선택의 흔적을 따로 본다.")
    if on_win and off_lose:
        text = "켠 규칙에서는 켠 쪽이, 끈 규칙에서는 끈 쪽이 이겼다. 저마다 자기 규칙에 적응했다는 뜻이다. " + adapted
    elif on_win and off_win:
        text = "양쪽 규칙에서 켠 쪽이 이겼다. 장치와 무관한 능력 차이일 수 있다(장치를 썼다는 증거로는 약하다)."
    elif on_win:
        if _rules_differ(t):
            text = ("켠 규칙에서 켠 쪽이 이겼고 끈 규칙에서는 가를 수 없다. 두 규칙의 차이는 잡음보다 크다. "
                    "켠 규칙에 적응했다는 뜻이다. " + adapted)
        else:
            text = ("켠 규칙에서 켠 쪽이 이겼고 끈 규칙에서는 가를 수 없다. 두 규칙 사이의 차이는 잡음 안이라 "
                    "'켠 규칙에서만 이긴다'고 말할 수 없다.")
    elif on_lose and off_lose:
        text = "양쪽 규칙에서 끈 쪽이 이겼다. 장치를 켠 세계의 진화가 뒤처졌다(느린 계산·비용 때문일 수 있다)."
    elif on_lose:
        text = "켠 규칙에서조차 끈 쪽이 이겼다. 장치와 함께 진화한 것이 값을 못 한다."
    elif off_lose:
        text = "끈 규칙에서는 끈 쪽이 이기고 켠 규칙에서는 가를 수 없다. 켠 쪽이 장치에 기대게 됐을 수 있다."
    elif off_win:
        text = ("끈 규칙에서는 켠 쪽이 이기고 켠 규칙에서는 가를 수 없다. 장치 없이도 통하는 차이다"
                "(장치를 썼다는 증거는 아니다).")
    else:
        text = "가를 수 없다."
    for label, v in (("켠 규칙", on_v), ("끈 규칙", off_v)):
        for w in _WARN.findall(v):
            text += f" {label} 겨루기 {w}"
    return text + _LIMIT


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
    print(f"다른 설정(끈 쪽, 켠 쪽): {res['config_diff']}")
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
