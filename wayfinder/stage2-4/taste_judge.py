"""맛 기억 판정 도구: 같은 체크포인트를 설정 하나만 바꿔 다시 돌려 켬/끔을 견준다.

flip-diff (판정 가: 맛 기억이 독을 따라 가려 먹게 하나):
    같은 체크포인트를 taste_enabled=true(켬)와 false(끔)로 — 같은 읽기 길로 — 첫 실제 바뀜 앞에서부터
    다시 돌려(replay.record) 장면마다 놓인 먹이 가운데 독(blind)을 적는다. 실제 바뀜마다
    "바뀐 직후 win스텝 → 다음 바뀜 직전 win스텝"의 변화(delta)를 내고, 같은 바뀜끼리 켬 − 끔을 짝지어
    바뀜 단위로 부트스트랩한다. 독을 따라 가려 먹으면 독 쪽이 다시 쌓여 delta가 오른다.
    --expect gain(기본): 평균 > 0.05, 95% 구간 아래 끝 > 0, 쓸 수 있는 바뀜 8번 이상이면 통과.
    --expect zero(장치를 끈 채 진화한 대조의 체크포인트): |평균| ≤ 0.02이면 통과(읽는 배선이 없으니 차이 0).
    바뀜이 8번 아래면 판정하지 않는다(30,000스텝에서 쓸 수 있는 바뀜은 평균 약 9번이라 모자랄 수 있다).
    독 0 대조(poison=0)는 쓰지 않는다: eat는 poison이 0이어도 독 이름표 분기를 타 그 알의 에너지가 사라지고
    맛 기억도 그 모양을 독으로 적어, 대조가 장치와 함께 지워진다.

first-meals (판정 나: 물려받으면 새끼가 첫 끼니부터 독을 덜 먹나):
    씨앗마다 같은 체크포인트를 taste_inherit=1(켬)과 0(끔)으로 읽어 한 스텝씩 돌리며, 그동안 태어난
    새끼의 첫 10끼 가운데 독을 센다. 한 스텝은 먹기 → 죽음 → 출생 순이고 죽은 칸은 같은 스텝의 출생이
    다시 써서 계수기가 0이 되므로, kernels.eat를 감싸(eat_tap) 먹기 직후에 읽는다(10끼를 못 채우고 죽은
    새끼도 먹은 만큼 들어간다). 재주입이 일어난 스텝(C_RESEED가 는 스텝)에 생긴 새끼는 모두 빼고 그 수를 적는다.
    새끼는 태어난 때가 마지막 실제 바뀜(실행 시작 전까지 거슬러 센다)에서 win스텝 넘게 지났으면 old
    (부모 기억이 맞는 때), win스텝 안이면 fresh(부모 기억이 틀린 때)다. 씨앗을 바꿔 읽으므로, 읽는 스텝의
    좋은 종류가 원래 씨앗의 바로 앞 스텝과 다르면 읽는 스텝도 실제 바뀜으로 친다. old끼리 끔 − 켬의 독 비율 차이를
    (씨앗, 태어난 바뀜 구간) 묶음 단위로 부트스트랩한다(같은 묶음의 새끼는 혈통과 바뀜을 함께 겪어 독립이 아니다).
    통과: old 차이 > 0.05이고 95% 구간 아래 끝 > 0(양수 = 물려받은 쪽이 독을 덜 먹음). 무리 크기와
    평균 에너지는 단서로 적는다. 선택 없는 대조 실행의 체크포인트는 받지 않는다. 각본은 적용하지 않는다
    (각본이 걸린 실행이면 경고를 낸다).

부트스트랩은 10,000번 다시 뽑기, np.random.default_rng(0), 2.5·97.5 백분위다. 끼니 합이 0이라 비율이 없는
회차는 백분위에서 빼고 그 수를 적는다.

    PYTHONPATH=. python wayfinder/stage2-4/taste_judge.py flip-diff <체크포인트.npz> [--steps 30000] [--every 10]
        [--win 300] [--expect zero|gain] [--json <출력.json>]
    PYTHONPATH=. python wayfinder/stage2-4/taste_judge.py first-meals <체크포인트.npz> [--steps 30000]
        [--seeds 1,2,3] [--win 300] [--json <출력.json>]
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np

from kkumteul import checkpoint as ckpt
from kkumteul import kernels as k
from kkumteul import replay
from kkumteul.config import FIRST_MEALS

GAIN_MEAN = 0.05      # 판정 가(②): 켬 − 끔의 바뀜 평균이 넘어야 하는 값
ZERO_TOL = 0.02       # 판정 가(①): 대조 체크포인트의 켬 − 끔이 이 안이어야 한다
MIN_FLIPS = 8         # 판정 가: 쓸 수 있는 바뀜이 이만큼은 있어야 판정한다
OLD_DIFF = 0.05       # 판정 나: old 새끼의 첫 10끼 독 비율이 끔 − 켬으로 넘어야 하는 값
REPS = 10000


# =========================== 순수 계산 ===========================
def flip_deltas(steps: np.ndarray, good: np.ndarray, blind: np.ndarray, win: int = 300) -> list[dict]:
    """실제 바뀜(good이 바뀐 장면)마다 바뀐 직후 win스텝과 다음 바뀜 직전 win스텝의 blind 평균과 그 차이.
    다음 바뀜이 없거나, 두 구간이 겹치거나(바뀜 간격 < 2·win), 구간에 장면이 없으면 넣지 않는다."""
    steps, good, blind = np.asarray(steps), np.asarray(good), np.asarray(blind, dtype=np.float64)
    flips = steps[1:][good[1:] != good[:-1]]
    out = []
    for f, nxt in zip(flips[:-1], flips[1:]):
        if nxt - f < 2 * win:
            continue
        early = blind[(steps >= f) & (steps < f + win)]
        late = blind[(steps >= nxt - win) & (steps < nxt)]
        if len(early) == 0 or len(late) == 0:
            continue
        e, lt = float(early.mean()), float(late.mean())
        out.append({"flip": int(f), "early": e, "late": lt, "delta": lt - e})
    return out


def paired_diff(on: list[dict], off: list[dict]) -> np.ndarray:
    """같은 바뀜 스텝끼리 짝지은 delta_on − delta_off(on의 순서). 한쪽에만 있는 바뀜은 뺀다."""
    by = {d["flip"]: d["delta"] for d in off}
    return np.array([d["delta"] - by[d["flip"]] for d in on if d["flip"] in by], dtype=np.float64)


def _resample(sizes: list[int], stat: Callable[..., np.ndarray], reps: int, seed: int) -> tuple[float, float, int]:
    """무리마다 단위를 제자리 뽑기로 다시 뽑아 stat의 2.5·97.5 백분위를 낸다.
    stat은 무리마다 (reps, 그 무리 크기) 번호 배열을 받아 다시 뽑은 값 reps개를 돌려준다.
    값이 nan인 회차(끼니 합 0)는 백분위에서 빠진다. 그 수를 셋째로 돌려준다."""
    rng = np.random.default_rng(seed)
    idx = [rng.integers(0, n, size=(reps, n)) for n in sizes]
    v = stat(*idx)
    lo, hi = np.nanpercentile(v, [2.5, 97.5])
    return float(lo), float(hi), int(np.isnan(v).sum())


def bootstrap_ci(x: np.ndarray, reps: int = REPS, seed: int = 0) -> tuple[float, float, float]:
    """(평균, 2.5%, 97.5%). 단위는 x의 원소(판정 가는 바뀜)."""
    x = np.asarray(x, dtype=np.float64)
    lo, hi, _ = _resample([len(x)], lambda i: x[i].mean(axis=1), reps, seed)
    return float(x.mean()), lo, hi


def last_flip_at_or_before(seed: int, t: int, period: int, lookback: int = 200) -> int | None:
    """t를 넘지 않는 마지막 실제 바뀜 스텝: 구간 경계 b·period에서 good_kind(b) != good_kind(b−1)인 가장 큰 것.
    good_kind는 씨앗과 스텝만의 함수라 실행 시작 전까지 거슬러 센다. 경계 lookback개 안에 없으면 None."""
    b = int(t) // period
    for _ in range(lookback):
        if b < 1:
            return None
        if k.good_kind(seed, b * period, period) != k.good_kind(seed, b * period - 1, period):
            return b * period
        b -= 1
    return None


def run_last_flip(orig_seed: int, seed: int, t0: int, t: int, period: int) -> int | None:
    """씨앗 orig_seed로 t0까지 산 체크포인트를 씨앗 seed로 읽어 돌릴 때, 개체가 겪은 t 이하 마지막 실제 바뀜.
    새 일정의 마지막 바뀜이 기본이다. 읽는 스텝 t0의 좋은 종류(새 씨앗)가 원래 씨앗의 t0 − 1과 다르면 t0도
    개체에게는 실제 바뀜이라 둘 중 늦은 쪽을 낸다. 같은 씨앗이면 이런 바뀜은 없다."""
    f = last_flip_at_or_before(seed, t, period)
    if 1 <= t0 <= t and k.good_kind(orig_seed, t0 - 1, period) != k.good_kind(seed, t0, period):
        f = t0 if f is None else max(f, t0)
    return f


def child_book(n: int) -> dict:
    """새끼 장부. slot_row·slot_id: 칸마다 그 칸에 사는 새끼의 줄 번호와 개체 번호(칸이 다른 개체로 바뀌었는지
    가린다). meals·poison·born: 줄마다 첫 끼니 수, 그 가운데 독, 태어난 스텝. n은 쓴 줄 수, excluded는 재주입
    스텝이라 올리지 않은 새끼 수."""
    return {"slot_row": np.full(n, -1, dtype=np.int64), "slot_id": np.full(n, -1, dtype=np.int64),
            "meals": np.zeros(1024, dtype=np.int64), "poison": np.zeros(1024, dtype=np.int64),
            "born": np.zeros(1024, dtype=np.int64), "n": 0, "excluded": 0}


def book_births(book: dict, new: np.ndarray, new_ids: np.ndarray, t: int, reseeded: bool) -> None:
    """스텝 t에 칸 new(개체 번호 new_ids)에서 태어난 새끼를 새 줄로 올린다. 재주입이 난 스텝이면 올리지 않고 센다."""
    if reseeded:
        book["excluded"] += len(new)
        return
    if not len(new):
        return
    while book["n"] + len(new) > len(book["born"]):
        for key in ("meals", "poison", "born"):
            book[key] = np.concatenate([book[key], np.zeros_like(book[key])])
    rows_new = np.arange(book["n"], book["n"] + len(new))
    book["slot_row"][new], book["slot_id"][new] = rows_new, new_ids
    book["meals"][rows_new], book["poison"][rows_new] = 0, 0    # 태어날 때 계수기는 0이다
    book["born"][rows_new] = t
    book["n"] += len(new)


def book_meals(book: dict, alive: np.ndarray, ids: np.ndarray, n_meal: np.ndarray, n_poison10: np.ndarray) -> None:
    """먹기 직후 계수기로 줄을 고친다. 칸의 지금 주인이 그 줄의 새끼(개체 번호가 같다)인 칸만 고치므로
    죽은 새끼의 줄은 먹은 만큼 남고, 칸을 다시 쓴 다른 개체의 끼니는 앞 새끼 줄에 들어가지 않는다."""
    m = (book["slot_row"] >= 0) & alive & (ids == book["slot_id"])
    book["meals"][book["slot_row"][m]] = np.minimum(n_meal[m], FIRST_MEALS)
    book["poison"][book["slot_row"][m]] = n_poison10[m]


def child_rows(book: dict, orig_seed: int, seed: int, t0: int, period: int) -> list[dict]:
    """장부의 줄마다 마지막 실제 바뀜에서 지난 스텝(since_flip), 묶음(씨앗, 그 바뀜), 끼니, 독."""
    n = book["n"]
    rows = []
    for t, me, po in zip(book["born"][:n].tolist(), book["meals"][:n].tolist(), book["poison"][:n].tolist()):
        f = run_last_flip(orig_seed, seed, t0, t, period)
        rows.append({"since_flip": None if f is None else t - f, "cluster": (seed, f), "meals": me, "poison": po})
    return rows


def split_children(rows: list[dict], win: int = 300) -> dict:
    """새끼를 마지막 실제 바뀜에서 지난 스텝으로 가른다: win 넘게(또는 바뀜을 못 찾음)는 old, 0~win은 fresh."""
    out: dict[str, list[dict]] = {"old": [], "fresh": []}
    for r in rows:
        sf = r["since_flip"]
        out["old" if sf is None or sf > win else "fresh"].append(r)
    return out


def pooled_ratio(rows: list[dict]) -> float:
    """Σpoison / Σmeals. 끼니가 0인 새끼도 분모·분자에 0으로 들어간다. 끼니가 하나도 없으면 nan."""
    m = sum(r["meals"] for r in rows)
    return sum(r["poison"] for r in rows) / m if m > 0 else float("nan")


def _cluster_sums(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """묶음(cluster)마다 (Σpoison, Σmeals)."""
    acc: dict = {}
    for r in rows:
        p, m = acc.get(r["cluster"], (0, 0))
        acc[r["cluster"]] = (p + r["poison"], m + r["meals"])
    v = np.array(list(acc.values()), dtype=np.float64).reshape(-1, 2)
    return v[:, 0], v[:, 1]


def ratio_diff_ci(a: list[dict], b: list[dict], reps: int = REPS,
                  seed: int = 0) -> tuple[float, float, float, int]:
    """pooled_ratio(a) − pooled_ratio(b)의 (값, 2.5%, 97.5%, 버린 회차 수). 두 무리를 각자 cluster 묶음째 다시 뽑는다.
    한쪽이라도 끼니 합이 0인 회차는 비율이 없어 버린다."""
    pa, ma = _cluster_sums(a)
    pb, mb = _cluster_sums(b)

    def stat(ia, ib):
        with np.errstate(invalid="ignore", divide="ignore"):
            return pa[ia].sum(axis=1) / ma[ia].sum(axis=1) - pb[ib].sum(axis=1) / mb[ib].sum(axis=1)

    lo, hi, dropped = _resample([len(pa), len(pb)], stat, reps, seed)
    return pooled_ratio(a) - pooled_ratio(b), lo, hi, dropped


@contextlib.contextmanager
def eat_tap(world, on_eat: Callable):
    """이 블록 안에서는 그 세계의 먹기(kernels.eat)가 끝날 때마다 on_eat(s)를 부른다.
    World.step이 k.eat를 모듈에서 찾으므로 감싸기가 먹힌다. 블록을 나가면 원래 함수로 돌려 놓는다.
    한 프로세스 안에서만 쓴다."""
    orig = k.eat

    def g(step, s, fp, ip):
        orig(step, s, fp, ip)
        if s is world.s:
            on_eat(s)

    k.eat = g
    try:
        yield
    finally:
        k.eat = orig


# =========================== 판정 가 ===========================
def flip_diff(path, steps: int = 30000, every: int = 10, win: int = 300, expect: str = "gain") -> dict:
    sides = {}
    for name, val in (("on", "true"), ("off", "false")):
        # 켬도 덮어쓰기를 준다: 빈 덮어쓰기는 ckpt.load(path)로 가서 끔 갈래와 읽기 길이 다르다
        rep = replay.record(path, steps, every, from_flip=True, overrides=[f"taste_enabled={val}"])
        sides[name] = flip_deltas(rep.steps, rep.good_kind, rep.blind, win)
    d = paired_diff(sides["on"], sides["off"])
    off = {x["flip"]: x for x in sides["off"]}
    rows = [{"flip": x["flip"], "on_early": x["early"], "on_late": x["late"], "on_delta": x["delta"],
             "off_early": off[x["flip"]]["early"], "off_late": off[x["flip"]]["late"],
             "off_delta": off[x["flip"]]["delta"]} for x in sides["on"] if x["flip"] in off]
    for r, v in zip(rows, d):
        r["diff"] = float(v)
    mean = lo = hi = None
    if len(d):
        mean, lo, hi = bootstrap_ci(d)
    verdict = None
    if len(d) >= MIN_FLIPS:
        verdict = (mean > GAIN_MEAN and lo > 0) if expect == "gain" else abs(mean) <= ZERO_TOL
    return {"ckpt": str(path), "steps": steps, "every": every, "win": win, "expect": expect,
            "n_flips": len(d), "flips": rows, "mean": mean, "lo": lo, "hi": hi, "verdict": verdict}


def _print_flip_diff(res: dict) -> None:
    print(f"쓸 수 있는 바뀜 {res['n_flips']}번 (win {res['win']}스텝, 놓인 먹이 가운데 독의 변화)")
    print(f"{'바뀜':>8} {'켬 delta':>9} {'끔 delta':>9} {'켬−끔':>8}")
    for r in res["flips"]:
        print(f"{r['flip']:>8} {r['on_delta']:>9.4f} {r['off_delta']:>9.4f} {r['diff']:>8.4f}")
    if res["mean"] is not None:
        print(f"평균 {res['mean']:.4f}, 95% 구간 [{res['lo']:.4f}, {res['hi']:.4f}]")
    if res["n_flips"] < MIN_FLIPS:
        print(f"바뀜 모자람({res['n_flips']} < {MIN_FLIPS}) — --steps를 늘려 다시. 판정하지 않음.")
    elif res["expect"] == "gain":
        print(f"판정(평균 > {GAIN_MEAN}, 아래 끝 > 0): {'통과' if res['verdict'] else '못 넘음'}")
    else:
        print(f"판정(|평균| ≤ {ZERO_TOL}): {'통과' if res['verdict'] else '못 넘음'}")


# =========================== 판정 나 ===========================
def _children(path, saved, seed: int, inherit: int, steps: int) -> dict:
    """한 씨앗·한 물려받기 설정으로 steps스텝 돌리며 새끼의 첫 10끼를 센다."""
    world, _ = ckpt.load(path, saved.with_overrides([f"taste_inherit={inherit}", f"seed={seed}"]))
    s, cfg = world.s, world.cfg
    t0 = world.step_no
    book = child_book(s.alive.shape[0])
    pop, energy = [], []

    def on_eat(st):
        # 먹기 뒤에는 그 스텝에 더 먹는 일이 없으므로 이 값이 그 스텝의 끝값이다
        book_meals(book, st.alive, st.ids, st.n_meal, st.n_poison10)

    with eat_tap(world, on_eat):
        for _ in range(steps):
            t = world.step_no
            r0 = int(s.counters[k.C_RESEED])
            world.step()
            world.drain_events()
            new = np.flatnonzero(s.alive & (s.birth_step == t))
            book_births(book, new, s.ids[new], t, int(s.counters[k.C_RESEED]) > r0)
            a = s.alive
            pop.append(int(a.sum()))
            energy.append(float(s.energy[a].mean()) if a.any() else 0.0)
    rows = child_rows(book, saved.seed, cfg.seed, t0, cfg.kind_flip_period)
    return {"rows": rows, "excluded": book["excluded"], "pop_mean": float(np.mean(pop)) if pop else 0.0,
            "energy_mean": float(np.mean(energy)) if energy else 0.0}


def _ratio_ci(a: list[dict], b: list[dict]) -> dict:
    """끔(a) − 켬(b). 어느 쪽이든 끼니가 없으면 값을 내지 않는다."""
    if not a or not b or not sum(r["meals"] for r in a) or not sum(r["meals"] for r in b):
        return {"diff": None, "lo": None, "hi": None, "dropped": None}
    v, lo, hi, dropped = ratio_diff_ci(a, b)
    return {"diff": v, "lo": lo, "hi": hi, "dropped": dropped}


def _has_scenario(path) -> bool:
    """체크포인트의 실행에 각본이 걸려 있었나. 다시 보기(replay)와 같은 길로 실행 폴더 run.json을 본다."""
    try:
        meta = json.loads((Path(path).parent.parent / "run.json").read_text())
    except (OSError, ValueError):
        return False
    return isinstance(meta, dict) and meta.get("scenario_start") is not None


def first_meals(path, steps: int = 30000, seeds=(1, 2, 3), win: int = 300) -> dict:
    saved = ckpt.read_config(Path(path))
    if saved.passive:     # 덮어쓴 설정이 아니라 저장된 설정으로 가린다
        raise ValueError("선택 없는 대조 실행의 체크포인트는 first-meals로 읽을 수 없다"
                         "(대조 일정 없이 돌리면 전혀 다른 세계가 된다). 본 실행의 체크포인트를 쓴다.")
    sides, split = {}, {}
    for name, inherit in (("on", 1), ("off", 0)):
        runs = [_children(path, saved, sd, inherit, steps) for sd in seeds]
        rows = [r for x in runs for r in x["rows"]]
        sp = split_children(rows, win)
        split[name] = sp
        sides[name] = {"n_children": len(rows), "reseed_excluded": sum(x["excluded"] for x in runs),
                       "n_old": len(sp["old"]), "n_fresh": len(sp["fresh"]),
                       "old_ratio": _num(pooled_ratio(sp["old"])), "fresh_ratio": _num(pooled_ratio(sp["fresh"])),
                       "pop_mean": float(np.mean([x["pop_mean"] for x in runs])),
                       "energy_mean": float(np.mean([x["energy_mean"] for x in runs])),
                       "rows": [[r["cluster"][0], r["cluster"][1], r["since_flip"], r["meals"], r["poison"]]
                                for r in rows]}
    old = _ratio_ci(split["off"]["old"], split["on"]["old"])
    fresh = _ratio_ci(split["off"]["fresh"], split["on"]["fresh"])
    verdict = None if old["diff"] is None else bool(old["diff"] > OLD_DIFF and old["lo"] > 0)
    return {"ckpt": str(path), "steps": steps, "seeds": list(seeds), "win": win,
            "scenario_ignored": _has_scenario(path), "sides": sides, "old": old, "fresh": fresh, "verdict": verdict}


def _num(v: float) -> float | None:
    """nan(끼니 없음)은 None으로(JSON에 그대로 쓸 수 있게)."""
    return None if v != v else v


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.4f}"


def _print_first_meals(res: dict) -> None:
    if res["scenario_ignored"]:
        print("경고: 이 실행에는 각본이 걸려 있었지만 first-meals는 각본을 적용하지 않고 저장된 설정 그대로 돌린다.")
    print(f"새끼의 첫 {FIRST_MEALS}끼 독 비율 (씨앗 {res['seeds']}, {res['steps']}스텝, 마지막 바뀜에서 "
          f"{res['win']}스텝 넘게 지나 태어남 = old)")
    for name, label in (("on", "물려받기 켬"), ("off", "물려받기 끔")):
        x = res["sides"][name]
        print(f"{label}: 새끼 {x['n_children']}(old {x['n_old']}, fresh {x['n_fresh']}, 재주입이라 뺌 "
              f"{x['reseed_excluded']}), old {_fmt(x['old_ratio'])}, fresh {_fmt(x['fresh_ratio'])}, "
              f"무리 {x['pop_mean']:.1f}, 평균 에너지 {x['energy_mean']:.3f}")
    for key, label in (("old", "old"), ("fresh", "fresh")):
        d = res[key]
        dropped = f" (끼니 합 0이라 버린 다시 뽑기 {d['dropped']}회)" if d["dropped"] else ""
        print(f"{label} 끔 − 켬: {_fmt(d['diff'])}, 95% 구간 [{_fmt(d['lo'])}, {_fmt(d['hi'])}]{dropped}")
    if res["verdict"] is None:
        print("old 새끼의 끼니가 없어 판정하지 않음.")
    else:
        print(f"판정(old 차이 > {OLD_DIFF}, 아래 끝 > 0): {'통과' if res['verdict'] else '못 넘음'}")


# =========================== 명령줄 ===========================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="taste_judge.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("flip-diff")
    f.add_argument("ckpt", type=Path)
    f.add_argument("--steps", type=int, default=30000)
    f.add_argument("--every", type=int, default=10)
    f.add_argument("--win", type=int, default=300)
    f.add_argument("--expect", choices=("zero", "gain"), default="gain")
    f.add_argument("--json", type=Path, default=None)
    m = sub.add_parser("first-meals")
    m.add_argument("ckpt", type=Path)
    m.add_argument("--steps", type=int, default=30000)
    m.add_argument("--seeds", default="1,2,3")
    m.add_argument("--win", type=int, default=300)
    m.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "flip-diff":
        res = flip_diff(a.ckpt, a.steps, a.every, a.win, a.expect)
        _print_flip_diff(res)
    else:
        seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
        res = first_meals(a.ckpt, a.steps, seeds, a.win)
        _print_first_meals(res)
    if a.json is not None:
        a.json.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
