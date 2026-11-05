"""실험 카드: 같은 체크포인트에서 대조(각본 없음)와 각본을 갈라 돌리고, 결과를 정해진 양식으로 적는다.

    python3 -m kkumteul.experiment --from runs/base --scenario scenarios/drought.json --steps 200000 --name my-drought

하는 일
1. `--from`의 체크포인트에서 두 갈래를 만든다: `runs/exp-<이름>-control`(각본 없음), `runs/exp-<이름>-treatment`(각본).
   난수 흐름은 둘이 같다. 두 갈래를 차례로 `--steps` 스텝씩 돌린다.
2. 두 갈래의 기록을 분석하고(`analyze.report`), 각본을 겪은 쪽의 생존자(B)와 대조 쪽의 생존자(A)를
   바탕 설정의 새 세계에서 겨루게 한다(`measure.assay_sides`). 한쪽이 20마리 아래면 겨루지 않고 그렇게 적는다.
3. `experiments/<이름>/card.json`과 `card.md`를 쓴다. 카드에는 기계마다 달라지는 값(시각, 속도, 경로)을 넣지 않는다.

도중에 멈췄으면 같은 명령을 다시 치면 이어 간다.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import signal
import sys
from pathlib import Path

from . import SOURCE_HASH, __version__
from . import analyze
from . import checkpoint as ckpt
from . import measure
from .config import Config
from .run import Runner
from .scenario import Scenario

CARD_VERSION = 1
MIN_SURVIVORS = 20          # 생존자가 이보다 적으면 겨루지 않는다
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

HOW_TO_READ = [
    "뇌 크기 같은 형질의 추세는 성장의 증거가 아니다. 선택이 없어도 변한다. 판정은 겨루기로 한다.",
    "재주입이 있었다면 그 개체군은 스스로 유지되지 못한 것이다. 그 갈래의 수치는 '버텼다'가 아니라 '채워 넣었다'로 읽는다.",
    "겨루기의 r은 1,000스텝당 ln(각본 쪽 수 ÷ 대조 쪽 수)의 기울기다. 양수면 각본을 겪은 쪽이 평소 세계에서 더 잘 는다.",
    "같은 무리끼리 겨룬 값(잡음)이 0에서 멀면 측정 자체가 기운 것이다. 한쪽이 전멸한 판은 r의 크기가 아니라 '전멸'만 읽는다.",
    "한 번, 한 씨앗의 결과다. 다른 씨앗의 세계에서도 같은지는 따로 돌려 봐야 한다.",
]


def _clean(v):
    """카드에 적을 측정값: 실수는 유효 숫자 5자리로 줄이고 NaN/Inf는 null로 바꾼다."""
    if isinstance(v, float):
        return float(f"{v:.5g}") if math.isfinite(v) else None
    if isinstance(v, dict):
        return {key: _clean(x) for key, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    return v


def _find_source(src: Path, at) -> Path:
    """출발점 체크포인트: npz면 그 파일, 실행 폴더면 at 스텝의 것(없으면 가장 최근)."""
    src = Path(src)
    if src.suffix == ".npz":
        return src
    cands = ckpt.list_checkpoints(src / "checkpoints")
    if at is not None:
        cands = [c for c in cands if c[0] == at]
    if not cands:
        raise FileNotFoundError(f"체크포인트를 못 찾았다: {src} at={at}")
    return cands[-1][1]


def _run_arm(src: Path, run_dir: Path, steps: int, scenario, quiet: bool, stop: dict) -> dict:
    """한 갈래를 만들거나(이미 있으면 이어서) 목표 스텝까지 돌린다. src는 출발점 체크포인트(npz)다."""
    side = run_dir / "experiment.json"
    if (run_dir / "run.json").exists():
        if not side.exists():
            raise FileExistsError(f"실험이 만든 폴더가 아니다: {run_dir}. 다른 --name을 쓴다")
        made = json.loads(side.read_text())
        meta = json.loads((run_dir / "run.json").read_text())
        same = (made["steps"] == steps
                and made.get("scenario_sha256") == (scenario.sha256() if scenario else None)
                and meta["parent_npz_sha256"] == json.loads(src.with_suffix(".json").read_text())["npz_sha256"])
        if not same:
            raise FileExistsError(f"같은 이름으로 다른 조건(길이·각본·출발점)의 실험이 이미 있다: {run_dir}. "
                                  "다른 --name을 쓰거나 그 폴더를 지운다")
        r = Runner.resume(run_dir)
    else:
        r = Runner.branch(src, run_dir, None, [], False, False, scenario)
        made = {"steps": steps, "scenario_sha256": scenario.sha256() if scenario else None,
                "start_summary": r.world.summary()}
        ckpt.atomic_write_bytes(side, json.dumps(made, indent=1).encode())
    stop["runner"] = r
    try:
        left = int(r.meta["parent_step"]) + steps - r.world.step_no
        if left > 0 and not stop["flag"]:
            r.loop(left, quiet)
    finally:
        stop["runner"] = None
        r.close()
    return {"meta": r.meta, "start": made["start_summary"], "done": r.world.step_no >= int(r.meta["parent_step"]) + steps}


def _arm_summary(run_dir: Path, start: dict) -> dict:
    """한 갈래의 요약. start는 갈라낸 순간의 세계 요약, 끝은 마지막 체크포인트의 요약이다."""
    rep = analyze.report(run_dir)
    last = ckpt.latest(run_dir / "checkpoints")
    end = json.loads(last.with_suffix(".json").read_text())["summary"]
    ft = rep.get("fitness_transmission")
    g0, g1 = start.get("gen_mean"), end.get("gen_mean")
    return {
        "pop_min": min(rep.get("pop_min", end["pop"]), end["pop"]),
        "pop_max": max(rep.get("pop_max", end["pop"]), end["pop"]),
        "pop_final": end["pop"],
        "reseeds": end["reseeds"] - start["reseeds"],
        "birth_ratio": rep.get("birth_ratio"),
        "disaster_deaths": end.get("deaths_disaster", 0) - start.get("deaths_disaster", 0),
        "generations_gained": None if g0 is None or g1 is None else g1 - g0,
        "hidden_mean_first_last": [start.get("hidden_mean"), end.get("hidden_mean")],
        "fitness_transmission_mean": ft["mean"] if ft else None,
        "fitness_transmission_windows": ft["windows"] if ft else 0,
        "energy_residual_rel": rep.get("energy_residual_rel"),
    }


def _assay(control_dir: Path, treat_dir: Path, base: Config, arms: dict, n_each: int, steps: int,
           reps: int, seed: int) -> dict:
    low = [name for name in ("control", "treatment") if arms[name]["pop_final"] < MIN_SURVIVORS]
    if low:
        label = {"control": "대조", "treatment": "각본"}
        return {"skipped": " · ".join(f"{label[name]} 쪽 생존자가 {arms[name]['pop_final']}마리라 겨루지 않았다"
                                      f"({MIN_SURVIVORS}마리 미만)" for name in low)}
    a = measure.load_side(ckpt.latest(control_dir / "checkpoints"))
    b = measure.load_side(ckpt.latest(treat_dir / "checkpoints"))
    # 겨루는 세계는 각본이 바꾸기 전의 바탕 설정이다. 카드에 기계의 경로가 남지 않게 이름으로 바꾼다
    a["cfg"] = b["cfg"] = base
    a["path"], b["path"] = "control", "treatment"
    n = min(n_each, base.max_agents // 2)
    res = measure.assay_sides(a, b, n, steps, reps, seed)
    out = {"a": "control", "b": "treatment", "n_each": n, "steps": steps, "reps": reps, "seed": seed,
           "r_per_1000": res["r_per_1000"], "neutral_r_per_1000": res.get("neutral_r_per_1000"),
           "final_counts": res["final_counts"], "a_extinct_runs": res["a_extinct_runs"],
           "b_extinct_runs": res["b_extinct_runs"], "verdict": res["verdict"]}
    return out


def _verdict(assay: dict) -> str:
    if "skipped" in assay:
        return "겨루지 않음: " + assay["skipped"]
    v = assay["verdict"]
    if v.startswith("B가 이김"):
        return "각본을 겪은 쪽이 이김" + v[len("B가 이김"):]
    if v.startswith("A가 이김"):
        return "대조 쪽이 이김" + v[len("A가 이김"):]
    return v


def run_experiment(src: Path, scenario_path: Path, steps: int, name: str, out_dir: Path = Path("experiments"),
                   runs_dir: Path = Path("runs"), at: int | None = None, reps: int = 4, assay_steps: int = 8000,
                   n_each: int = 400, assay_seed: int = 1000, quiet: bool = True, stop: dict | None = None):
    """실험을 돌리고 카드(dict)를 돌려준다. 도중에 멈추라는 신호를 받았으면 None을 돌려준다."""
    if not _NAME.match(name):
        raise ValueError("--name은 영문자·숫자·점·밑줄·붙임표만 쓴다(64자까지, 첫 글자는 영문자나 숫자)")
    if steps < 1:
        raise ValueError("--steps는 1 이상이어야 한다")
    stop = stop if stop is not None else {"flag": False, "runner": None}
    scn = Scenario.load(scenario_path)
    src = _find_source(src, at)
    control_dir = Path(runs_dir) / f"exp-{name}-control"
    treat_dir = Path(runs_dir) / f"exp-{name}-treatment"
    say = (lambda *a: None) if quiet else (lambda *a: print(*a, flush=True))
    say(f"[1/3] 대조 갈래(각본 없음)를 {steps}스텝 돌린다: {control_dir}")
    c = _run_arm(src, control_dir, steps, None, quiet, stop)
    if not c["done"]:
        return None
    say(f"[2/3] 각본 갈래({scn.name})를 {steps}스텝 돌린다: {treat_dir}")
    t = _run_arm(src, treat_dir, steps, scn, quiet, stop)
    if not t["done"]:
        return None
    base = Config.load(treat_dir / "config.json")
    arms = {"control": _arm_summary(control_dir, c["start"]), "treatment": _arm_summary(treat_dir, t["start"])}
    say(f"[3/3] 겨루기: 각본을 겪은 쪽(B) 대 대조 쪽(A), 한쪽 {min(n_each, base.max_agents // 2)}마리, "
        f"{assay_steps}스텝, {reps}회")
    assay = _assay(control_dir, treat_dir, base, arms, n_each, assay_steps, reps, assay_seed)
    default = Config().to_dict()
    card = {
        "card_version": CARD_VERSION,
        "name": name,
        "scenario": {**scn.to_dict(), "sha256": scn.sha256()},
        "parent": {"checkpoint": t["meta"]["parent_checkpoint"], "npz_sha256": t["meta"]["parent_npz_sha256"],
                   "step": t["meta"]["parent_step"]},
        "steps": steps,
        "seed": base.seed,
        "config_diff_from_default": {key: v for key, v in base.to_dict().items() if default[key] != v},
        "code_version": __version__,
        "source_hash": SOURCE_HASH,
        "arms": _clean(arms),
        "assay": _clean(assay),
        "verdict": _verdict(assay),
        "how_to_read": HOW_TO_READ,
    }
    out = Path(out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    ckpt.atomic_write_bytes(out / "card.json", (json.dumps(card, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    ckpt.atomic_write_bytes(out / "card.md", card_markdown(card).encode("utf-8"))
    return card


def _fmt(v, digits: int = 2) -> str:
    if v is None:
        return "없음"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return f"{v:,}" if isinstance(v, int) else str(v)


def card_markdown(card: dict) -> str:
    c, t = card["arms"]["control"], card["arms"]["treatment"]
    scn = card["scenario"]
    rows = [
        ("개체 수 최소 / 최대 / 끝", lambda a: f"{_fmt(a['pop_min'])} / {_fmt(a['pop_max'])} / {_fmt(a['pop_final'])}"),
        ("재주입(0이어야 스스로 유지된 것)", lambda a: _fmt(a["reseeds"])),
        ("재난으로 죽은 수", lambda a: _fmt(a["disaster_deaths"])),
        ("쌓인 세대(평균 계보 깊이 증가)", lambda a: _fmt(a["generations_gained"], 1)),
        ("평균 활성 뉴런 처음 → 끝", lambda a: f"{_fmt(a['hidden_mean_first_last'][0])} → {_fmt(a['hidden_mean_first_last'][1])}"),
        ("Fitness Transmission 평균(구간 수)", lambda a: f"{_fmt(a['fitness_transmission_mean'], 4)} ({a['fitness_transmission_windows']})"),
    ]
    out = [
        f"# 실험 카드: {card['name']}",
        "",
        f"- 각본: **{scn['name']}** — {scn['description'] or '(설명 없음)'}",
        f"- 출발점: `{card['parent']['checkpoint']}` ({_fmt(card['parent']['step'])}스텝, sha256 `{card['parent']['npz_sha256']}`)",
        f"- 돌린 길이: 두 갈래 모두 {_fmt(card['steps'])}스텝. 세계 씨앗 {card['seed']}, 난수 흐름은 두 갈래가 같다",
        f"- 코드 {card['code_version']}, 계산 코드 해시 `{card['source_hash'][:16]}`, 각본 해시 `{scn['sha256'][:16]}`",
    ]
    if card["config_diff_from_default"]:
        diff = ", ".join(f"`{key}={v}`" for key, v in sorted(card["config_diff_from_default"].items()))
        out.append(f"- 기본 설정과 다른 값: {diff}")
    out += ["", "## 두 갈래", "", "| | 대조(각본 없음) | 각본 |", "|---|---|---|"]
    out += [f"| {label} | {fn(c)} | {fn(t)} |" for label, fn in rows]
    out += ["", "## 겨루기", "", "각본을 겪은 쪽의 생존자(B)와 대조 쪽의 생존자(A)를 바탕 설정의 새 세계에 같은 수로 풀었다.", ""]
    a = card["assay"]
    if "skipped" in a:
        out.append(f"- {a['skipped']}")
    else:
        r = a["r_per_1000"]
        out.append(f"- 한쪽 {a['n_each']}마리, {_fmt(a['steps'])}스텝, {a['reps']}회(씨앗 {a['seed']})")
        out.append(f"- r(각본 − 대조) / 1,000스텝 = {_fmt(r['mean'], 4)} ± {_fmt(r['sem'], 4)} (표준오차)")
        nz = a.get("neutral_r_per_1000")
        if nz:
            out.append(f"- 같은 무리끼리(잡음): 평균 {_fmt(nz['mean'], 4)}, 표준편차 {_fmt(nz['sd'], 4)}")
        out.append("- 끝 개체 수 [대조, 각본]: " + ", ".join(f"[{x}, {y}]" for x, y in a["final_counts"]))
    out += ["", f"**판정: {card['verdict']}**", "", "## 읽는 법", ""]
    out += [f"- {line}" for line in card["how_to_read"]]
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="실험 카드: 같은 체크포인트에서 대조와 각본을 갈라 돌리고 결과를 적는다")
    ap.add_argument("--from", dest="src", required=True, type=Path, help="출발점: 실행 폴더(가장 최근 체크포인트)나 체크포인트 npz")
    ap.add_argument("--scenario", required=True, type=Path, help="각본 json (scenarios/ 참고)")
    ap.add_argument("--steps", required=True, type=int, help="두 갈래를 각각 돌릴 스텝 수")
    ap.add_argument("--name", required=True, help="실험 이름(폴더 이름이 된다)")
    ap.add_argument("--out", type=Path, default=Path("experiments"), help="카드를 쓸 폴더")
    ap.add_argument("--runs", type=Path, default=Path("runs"), help="두 갈래의 실행 폴더를 만들 곳")
    ap.add_argument("--at", type=int, default=None, help="갈라낼 체크포인트의 스텝(없으면 가장 최근)")
    ap.add_argument("--reps", type=int, default=4, help="겨루기 반복 수")
    ap.add_argument("--assay-steps", type=int, default=8000, help="겨루기 한 판의 스텝 수")
    ap.add_argument("--n", type=int, default=400, help="겨루기에서 한쪽 개체 수")
    ap.add_argument("--assay-seed", type=int, default=1000, help="겨루기 난수 씨앗")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    stop = {"flag": False, "runner": None}

    def on_signal(signum, frame):
        stop["flag"] = True
        if stop["runner"] is not None:
            stop["runner"].stop = True
    old = {sig: signal.signal(sig, on_signal) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        card = run_experiment(args.src, args.scenario, args.steps, args.name, args.out, args.runs, args.at,
                              args.reps, args.assay_steps, args.n, args.assay_seed, args.quiet, stop)
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)
    if card is None:
        print("멈췄다. 같은 명령을 다시 치면 이어 간다.")
        return 1
    out = args.out / args.name
    print(f"판정: {card['verdict']}")
    print(f"카드: {out / 'card.md'}, {out / 'card.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
