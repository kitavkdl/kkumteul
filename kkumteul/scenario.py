"""각본: 세계에 주는 조건(재난·기후 변화)을 json 파일 하나로 적는다.

    {"name": "가뭄", "description": "...", "update_every": 100, "events": [ ... ]}

시간은 모두 각본이 시작된 스텝부터 센 스텝 수다(시작 스텝은 실행 폴더 `run.json`의 `scenario_start`).
사건의 종류는 다섯 가지다. 자세한 설명과 예는 `scenarios/README.md`에 있다.

    {"at": T, "set": {"food_inflow": 4.0}}                    T부터 설정값을 바꾼다
    {"at": T, "for": D, "scale": {"food_inflow": 0.2}}        [T, T+D) 동안 값에 배율을 곱한다
    {"from": T0, "to": T1, "scale_to": {"basal_cost": 1.5}}   배율이 T0의 1에서 T1의 목표까지 곧게 변하고 그 뒤로 머문다
    {"every": P, "amp": {"food_inflow": 0.5}, "phase": 0.0}   계절: 배율 = 1 + amp * sin(2pi * (t/P + phase))
    {"at": T, "strike": {"radius": 60, "kill": 1.0, "burn_food": true, "x": null, "y": null}}
    {"every": P, "first": T, "strike": {...}}                 운석: 원 안의 개체를 죽이고 먹이를 없앤다

지키는 것
- 어떤 스텝의 설정은 (바탕 설정, 각본, 그 스텝)만으로 정해진다. 그래서 체크포인트에서 이어 돌린 결과가
  끊지 않고 돌린 결과와 같다. 서서히 변하는 배율(scale_to, amp)은 `update_every` 스텝마다 한 번만 다시 계산하고,
  set과 scale의 시작·끝은 정확히 그 스텝에 반영한다.
- 배열 크기를 바꾸는 설정과 실행·기록용 설정은 건드릴 수 없다. 정수 설정은 배율을 곱한 뒤 반올림한다.
  결과는 `Config.validate()`를 통과해야 한다.
- 운석 자리를 적지 않으면 (세계의 씨앗, 떨어지는 스텝)으로 정한다. 난수는 `rng.py`의 해시만 쓴다.

    python3 -m kkumteul.scenario scenarios/drought.json      # 각본을 검사하고 시간에 따른 값을 보여 준다
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import fields
from pathlib import Path

from . import rng
from .config import Config

# 실행 중에 바꾸면 배열 크기가 달라지는 설정(`World.set_config`가 막는 것과 같은 목록)
FIXED_SIZE = ("max_agents", "food_cap", "h_max", "vision_sectors", "n_spare_in", "n_spare_out",
              "world_w", "world_h", "cell", "max_animals")
# 세계의 조건이 아니라 실행·기록 방법인 설정
RUN_ONLY = ("seed", "threads", "max_steps_per_sec", "passive", "init_agents",
            "stats_every", "frame_every", "clip_every", "clip_len", "frames_per_segment",
            "frame_ring_segments", "keyframe_every", "ckpt_every", "ckpt_keep_last", "ckpt_keep_every",
            "event_segment_steps", "event_keep_segments", "min_free_gb")
LOCKED = frozenset(FIXED_SIZE + RUN_ONLY)

MAX_EVENTS = 1000          # 운석의 난수 번호에 사건 번호를 섞으므로 상한을 둔다
_KEY_BASE = 1024

_TOP_KEYS = {"name", "description", "update_every", "events"}
_ACTIONS = ("set", "scale", "scale_to", "amp", "strike")
# 사건 종류별 (꼭 있어야 하는 키, 있어도 되는 키). "note"는 어디에나 적을 수 있는 메모다.
_KEYS = {
    "set": ({"at", "set"}, set()),
    "scale": ({"at", "for", "scale"}, set()),
    "scale_to": ({"from", "to", "scale_to"}, set()),
    "amp": ({"every", "amp"}, {"phase", "from"}),
    "strike_at": ({"at", "strike"}, set()),
    "strike_every": ({"every", "strike"}, {"first", "until"}),
}
_STRIKE_KEYS = {"radius", "kill", "burn_food", "x", "y"}


def _field_types() -> dict:
    return {f.name: str(f.type) for f in fields(Config)}


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _as_int(v, what: str, lo: int) -> int:
    if not _is_num(v) or v != int(v):
        raise ValueError(f"{what}는 정수여야 한다: {v!r}")
    v = int(v)
    if v < lo:
        raise ValueError(f"{what}는 {lo} 이상이어야 한다: {v}")
    return v


def _as_num(v, what: str, lo: float | None = None, hi: float | None = None) -> float:
    if not _is_num(v):
        raise ValueError(f"{what}는 수여야 한다: {v!r}")
    v = float(v)
    if lo is not None and v < lo:
        raise ValueError(f"{what}는 {lo:g} 이상이어야 한다: {v:g}")
    if hi is not None and v > hi:
        raise ValueError(f"{what}는 {hi:g} 이하여야 한다: {v:g}")
    return v


def _values(where: str, d, action: str, types: dict) -> dict:
    """설정 이름 -> 값(또는 배율) 사전을 검사해 돌려준다."""
    if not isinstance(d, dict) or not d:
        raise ValueError(f"{where}: {action}에는 설정 이름과 값을 하나 이상 적는다")
    out = {}
    for name, v in d.items():
        if name not in types:
            raise ValueError(f"{where}: 모르는 설정: {name}")
        if name in LOCKED:
            raise ValueError(f"{where}: {name}은 각본으로 바꿀 수 없다(배열 크기나 실행·기록 방법을 정하는 설정)")
        ty = types[name]
        if action == "set":
            if ty == "bool":
                if not isinstance(v, bool):
                    raise ValueError(f"{where}: {name}은 true/false여야 한다: {v!r}")
                out[name] = v
            elif ty == "int":
                out[name] = _as_int(v, f"{where}: {name}", -(1 << 62))
            else:
                out[name] = _as_num(v, f"{where}: {name}")
        else:
            if ty == "bool":
                raise ValueError(f"{where}: {name}은 켜고 끄는 설정이라 배율을 곱할 수 없다(set을 쓴다)")
            if action == "amp":
                out[name] = _as_num(v, f"{where}: {name}의 진폭", -1.0, 1.0)
            else:
                out[name] = _as_num(v, f"{where}: {name}의 배율", 0.0)
    return out


def _strike(where: str, d) -> dict:
    if not isinstance(d, dict):
        raise ValueError(f"{where}: strike는 {{\"radius\": ...}} 꼴이어야 한다")
    unknown = set(d) - _STRIKE_KEYS
    if unknown:
        raise ValueError(f"{where}: strike에 모르는 키: {sorted(unknown)}")
    if "radius" not in d:
        raise ValueError(f"{where}: strike에는 radius가 있어야 한다")
    radius = _as_num(d["radius"], f"{where}: radius")
    if radius <= 0:
        raise ValueError(f"{where}: radius는 0보다 커야 한다")
    burn = d.get("burn_food", True)
    if not isinstance(burn, bool):
        raise ValueError(f"{where}: burn_food는 true/false여야 한다")
    out = {"radius": radius, "kill": _as_num(d.get("kill", 1.0), f"{where}: kill", 0.0, 1.0),
           "burn_food": burn, "x": None, "y": None}
    for axis in ("x", "y"):
        if d.get(axis) is not None:
            out[axis] = _as_num(d[axis], f"{where}: {axis}")
    return out


def _event(i: int, ev, types: dict) -> dict:
    where = f"events[{i}]"
    if not isinstance(ev, dict):
        raise ValueError(f"{where}: 사건은 {{...}} 꼴이어야 한다")
    acts = [a for a in _ACTIONS if a in ev]
    if len(acts) != 1:
        raise ValueError(f"{where}: set, scale, scale_to, amp, strike 가운데 정확히 하나가 있어야 한다")
    act = acts[0]
    kind = act if act != "strike" else ("strike_every" if "every" in ev else "strike_at")
    need, optional = _KEYS[kind]
    keys = set(ev) - {"note"}
    if keys - need - optional:
        raise ValueError(f"{where}: {act} 사건에 모르는 키: {sorted(keys - need - optional)}")
    if need - keys:
        raise ValueError(f"{where}: {act} 사건에 빠진 키: {sorted(need - keys)}")
    out: dict = {"kind": kind}
    if "note" in ev:
        if not isinstance(ev["note"], str):
            raise ValueError(f"{where}: note는 글이어야 한다")
        out["note"] = ev["note"]
    if kind == "set":
        out["at"] = _as_int(ev["at"], f"{where}: at", 0)
        out["values"] = _values(where, ev["set"], "set", types)
    elif kind == "scale":
        out["at"] = _as_int(ev["at"], f"{where}: at", 0)
        out["for"] = _as_int(ev["for"], f"{where}: for", 1)
        out["values"] = _values(where, ev["scale"], "scale", types)
    elif kind == "scale_to":
        out["from"] = _as_int(ev["from"], f"{where}: from", 0)
        out["to"] = _as_int(ev["to"], f"{where}: to", 1)
        if out["to"] <= out["from"]:
            raise ValueError(f"{where}: to는 from보다 커야 한다")
        out["values"] = _values(where, ev["scale_to"], "scale_to", types)
    elif kind == "amp":
        out["every"] = _as_int(ev["every"], f"{where}: every", 2)
        out["phase"] = _as_num(ev.get("phase", 0.0), f"{where}: phase")
        out["from"] = _as_int(ev.get("from", 0), f"{where}: from", 0)
        out["values"] = _values(where, ev["amp"], "amp", types)
    elif kind == "strike_at":
        out["at"] = _as_int(ev["at"], f"{where}: at", 0)
        out["strike"] = _strike(where, ev["strike"])
    else:
        out["every"] = _as_int(ev["every"], f"{where}: every", 1)
        out["first"] = _as_int(ev.get("first", out["every"]), f"{where}: first", 0)
        out["until"] = None if ev.get("until") is None else _as_int(ev["until"], f"{where}: until", 1)
        out["strike"] = _strike(where, ev["strike"])
    return out


def strike_key(t_abs: int, index: int) -> int:
    """운석 하나의 난수 번호. 떨어지는 스텝과 각본 안의 사건 번호로 정한다(2 이상)."""
    return 2 + index + _KEY_BASE * int(t_abs)


def strike_position(seed: int, key: int, world_w: float, world_h: float) -> tuple[float, float]:
    """자리를 적지 않은 운석이 떨어지는 곳. (씨앗, 난수 번호)만으로 정해진다."""
    return (float(rng.u01(seed, rng.S_SCENARIO, key, 0)) * world_w,
            float(rng.u01(seed, rng.S_SCENARIO, key, 1)) * world_h)


class Scenario:
    def __init__(self, name: str, description: str, update_every: int, events: list[dict]):
        self.name = name
        self.description = description
        self.update_every = update_every
        self.events = events
        self._types = _field_types()
        self._sets = sorted((e for e in events if e["kind"] == "set"), key=lambda e: e["at"])
        self._mults = [e for e in events if e["kind"] in ("scale", "scale_to", "amp")]
        self._strikes = [(i, e) for i, e in enumerate(events) if e["kind"].startswith("strike")]
        touched = set()
        edges = set()
        for e in events:
            touched.update(e.get("values", ()))
            if e["kind"] == "set":
                edges.add(e["at"])
            elif e["kind"] == "scale":
                edges.update((e["at"], e["at"] + e["for"]))
        self.touched = tuple(sorted(touched))    # 이 각본이 건드리는 설정 이름
        self.edges = frozenset(edges)            # 설정이 정확히 그 스텝에 바뀌어야 하는 때

    # ---- 읽고 쓰기 ----
    @classmethod
    def from_dict(cls, d: dict, default_name: str = "") -> "Scenario":
        if not isinstance(d, dict):
            raise ValueError("각본은 {...} 꼴이어야 한다")
        unknown = set(d) - _TOP_KEYS
        if unknown:
            raise ValueError(f"각본에 모르는 키: {sorted(unknown)}")
        name = d.get("name", default_name)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("각본에는 name(이름)이 있어야 한다")
        desc = d.get("description", "")
        if not isinstance(desc, str):
            raise ValueError("description은 글이어야 한다")
        every = _as_int(d.get("update_every", 100), "update_every", 1)
        evs = d.get("events")
        if not isinstance(evs, list):
            raise ValueError("각본에는 events 목록이 있어야 한다(비어 있어도 된다)")
        if len(evs) > MAX_EVENTS:
            raise ValueError(f"사건은 {MAX_EVENTS}개까지 적을 수 있다")
        types = _field_types()
        return cls(name.strip(), desc, every, [_event(i, e, types) for i, e in enumerate(evs)])

    @classmethod
    def load(cls, path) -> "Scenario":
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            try:
                d = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"각본 파일이 json이 아니다: {path.name}: {e}") from None
        return cls.from_dict(d, default_name=path.stem)

    def to_dict(self) -> dict:
        evs = []
        for e in self.events:
            kind = e["kind"]
            if kind == "set":
                out = {"at": e["at"], "set": dict(e["values"])}
            elif kind == "scale":
                out = {"at": e["at"], "for": e["for"], "scale": dict(e["values"])}
            elif kind == "scale_to":
                out = {"from": e["from"], "to": e["to"], "scale_to": dict(e["values"])}
            elif kind == "amp":
                out = {"every": e["every"], "amp": dict(e["values"]), "phase": e["phase"]}
                if e["from"]:
                    out["from"] = e["from"]
            elif kind == "strike_at":
                out = {"at": e["at"], "strike": dict(e["strike"])}
            else:
                out = {"every": e["every"], "first": e["first"], "strike": dict(e["strike"])}
                if e["until"] is not None:
                    out["until"] = e["until"]
            if "note" in e:
                out["note"] = e["note"]
            evs.append(out)
        return {"name": self.name, "description": self.description, "update_every": self.update_every,
                "events": evs}

    def to_bytes(self) -> bytes:
        return (json.dumps(self.to_dict(), ensure_ascii=False, indent=1) + "\n").encode("utf-8")

    def sha256(self) -> str:
        """각본 내용의 해시(적는 순서·빈칸과 무관하게 같은 각본이면 같다)."""
        return hashlib.sha256(json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    # ---- 어떤 스텝의 설정 ----
    def values(self, base: Config, t_rel: int) -> dict:
        """각본이 건드리는 설정들의 t_rel 스텝째 값. (바탕 설정, 각본, t_rel)만으로 정해진다."""
        vals = {n: getattr(base, n) for n in self.touched}
        if t_rel < 0:
            return vals
        for e in self._sets:
            if t_rel >= e["at"]:
                vals.update(e["values"])
        tq = t_rel - t_rel % self.update_every   # 서서히 변하는 배율은 이 스텝의 값으로 묶는다
        mult: dict = {}
        for e in self._mults:
            kind = e["kind"]
            if kind == "scale":
                if not (e["at"] <= t_rel < e["at"] + e["for"]):
                    continue
                for n, f in e["values"].items():
                    mult[n] = mult.get(n, 1.0) * f
            elif kind == "scale_to":
                t0, t1 = e["from"], e["to"]
                if tq <= t0:
                    continue
                for n, target in e["values"].items():
                    m = target if tq >= t1 else 1.0 + (target - 1.0) * ((tq - t0) / (t1 - t0))
                    mult[n] = mult.get(n, 1.0) * m
            else:
                if tq < e["from"]:
                    continue
                s = math.sin(2.0 * math.pi * ((tq % e["every"]) / e["every"] + e["phase"]))
                for n, a in e["values"].items():
                    mult[n] = mult.get(n, 1.0) * (1.0 + a * s)
        for n, m in mult.items():
            v = vals[n] * m
            vals[n] = int(round(v)) if self._types[n] == "int" else v
        return vals

    def effective_config(self, base: Config, t_rel: int) -> Config:
        """t_rel 스텝째(각본 시작부터 센 스텝)에 세계가 쓸 설정."""
        if not self.touched:
            return base
        vals = self.values(base, t_rel)
        try:
            return Config.from_dict({**base.to_dict(), **vals})
        except ValueError as e:
            raise ValueError(f"각본 '{self.name}'의 {t_rel}스텝째 설정이 맞지 않는다: {e}") from None

    def strikes_at(self, t_rel: int) -> list[dict]:
        """t_rel 스텝째에 떨어지는 운석들. 각 항목에 각본 안의 사건 번호(index)가 붙는다."""
        out = []
        for i, e in self._strikes:
            if e["kind"] == "strike_at":
                hit = t_rel == e["at"]
            else:
                hit = (t_rel >= e["first"] and (t_rel - e["first"]) % e["every"] == 0
                       and (e["until"] is None or t_rel < e["until"]))
            if hit:
                out.append({**e["strike"], "index": i})
        return out

    def strike_times(self, upto: int) -> list[int]:
        """upto 스텝째까지 운석이 떨어지는 때."""
        ts = set()
        for _, e in self._strikes:
            if e["kind"] == "strike_at":
                if e["at"] <= upto:
                    ts.add(e["at"])
            else:
                end = upto if e["until"] is None else min(upto, e["until"] - 1)
                ts.update(range(e["first"], end + 1, e["every"]))
        return sorted(ts)

    def horizon(self) -> int:
        """마지막으로 무언가 바뀌는 때(끝없이 되풀이되는 사건은 두 주기까지)."""
        h = self.update_every
        for e in self.events:
            kind = e["kind"]
            if kind in ("set", "strike_at"):
                h = max(h, e["at"])
            elif kind == "scale":
                h = max(h, e["at"] + e["for"])
            elif kind == "scale_to":
                h = max(h, e["to"])
            elif kind == "amp":
                h = max(h, e["from"] + 2 * e["every"])
            else:
                h = max(h, e["until"] if e["until"] is not None else e["first"] + 2 * e["every"])
        return h

    def sample_times(self, limit: int = 2000) -> list[int]:
        """값이 달라질 수 있는 때를 고르게 뽑는다(검사·미리 보기용)."""
        u = self.update_every
        h = self.horizon()
        grid = u * max(1, math.ceil(h / u / limit))
        ts = set(range(0, h + grid + 1, grid))
        for t in self.edges:
            ts.update((t - 1, t))
        for e in self._mults:
            if e["kind"] == "scale_to":
                ts.update((e["from"], e["to"] - e["to"] % u, e["to"] - e["to"] % u + u))
            elif e["kind"] == "amp":
                # 한 주기에서 가장 높을 때와 가장 낮을 때 근처
                for q in (0.25, 0.75):
                    t = e["from"] - e["from"] % e["every"] + e["every"] + int(((q - e["phase"]) % 1.0) * e["every"])
                    ts.update((t - t % u, t - t % u + u))
        return sorted(t for t in ts if t >= 0)

    def check(self, base: Config) -> None:
        """이 바탕 설정에서 각본이 끝까지 맞는 설정을 내는지 미리 본다. 안 맞으면 ValueError."""
        seen = set()
        for t in self.sample_times():
            vals = self.values(base, t)
            key = tuple(vals.values())
            if key in seen:
                continue
            seen.add(key)
            self.effective_config(base, t)

    # ---- 세계에 적용 ----
    def apply(self, world, base_cfg: Config, t_abs: int, start: int, force: bool = False) -> int:
        """world.step() 바로 앞에서 부른다. 설정이 실제로 달라졌을 때만 world.set_config()를 부르고,
        이 스텝에 떨어질 운석을 떨어뜨린다. force는 이어 돌리기 시작할 때 한 번 쓴다(설정을 무조건 다시 맞춘다).
        운석으로 죽은 수를 돌려준다."""
        t = t_abs - start
        if force or (self.touched and (t % self.update_every == 0 or t in self.edges)):
            cfg = self.effective_config(base_cfg, t)
            if cfg != world.cfg:
                world.set_config(cfg)
        killed = 0
        if self._strikes and t >= 0:
            for st in self.strikes_at(t):
                key = strike_key(t_abs, st["index"])
                x, y = st["x"], st["y"]
                if x is None or y is None:
                    rx, ry = strike_position(world.cfg.seed, key, world.cfg.world_w, world.cfg.world_h)
                    x = rx if x is None else x
                    y = ry if y is None else y
                killed += world.strike(x, y, st["radius"], st["kill"], st["burn_food"], key)
        return killed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="각본을 검사하고 시간에 따른 설정값을 보여 준다")
    ap.add_argument("scenario", type=Path, help="각본 json 파일")
    ap.add_argument("--config", type=Path, help="바탕으로 쓸 설정 json(없으면 기본 설정)")
    ap.add_argument("--set", action="append", default=[], metavar="key=value", help="바탕 설정에서 바꿀 값")
    ap.add_argument("--rows", type=int, default=12, help="보여 줄 줄 수")
    args = ap.parse_args(argv)
    base = (Config.load(args.config) if args.config else Config()).with_overrides(args.set)
    scn = Scenario.load(args.scenario)
    scn.check(base)
    print(f"이름: {scn.name}")
    if scn.description:
        print(f"설명: {scn.description}")
    print(f"사건 {len(scn.events)}개, 건드리는 설정: {', '.join(scn.touched) or '없음'}")
    h = scn.horizon()
    if scn.touched:
        print("스텝(각본 시작부터)  " + "  ".join(scn.touched))
        for i in range(args.rows + 1):
            t = int(round(h * i / args.rows))
            vals = scn.values(base, t)
            print(f"{t:>12}  " + "  ".join(f"{vals[n]:.6g}" if isinstance(vals[n], float) else str(vals[n])
                                           for n in scn.touched))
    hits = [(t, s) for t in scn.strike_times(h) for s in scn.strikes_at(t)]
    if hits:
        print(f"운석 {len(hits)}번(끝없이 되풀이되면 두 주기까지만 센다). 처음 몇 번:")
        for t, s in hits[:8]:
            where = "씨앗과 스텝으로 정함" if s["x"] is None or s["y"] is None else f"({s['x']}, {s['y']})"
            print(f"  {t}스텝째: 반지름 {s['radius']}, 죽을 확률 {s['kill']}, 먹이 {'없앰' if s['burn_food'] else '둠'}, 자리 {where}")
    print("검사 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
