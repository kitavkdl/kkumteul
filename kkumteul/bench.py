"""속도·램 실측(T15). 개체 수와 뇌 크기를 바꿔 가며 초당 스텝과 메모리를 잰다.

    python -m kkumteul.bench [--n 500 1000 2000 5000] [--h 16 64] [--threads 0] [--steps 1500]

개체 수 n에 맞춰 세계 넓이·먹이 유입을 같이 늘려 밀도를 같게 둔다.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import resource
import sys
import time

import numba
import numpy as np

from . import kernels as k
from .config import Config
from .world import World


def make_config(n: int, h: int) -> Config:
    base = Config()
    side = max(8 * 12, int(round(base.world_w * math.sqrt(n / 1000.0) / base.cell)) * int(base.cell))
    d = base.to_dict()
    d.update(world_w=float(side), world_h=float(side), max_agents=2 * n, init_agents=n,
             min_agents=min(100, n // 4), food_inflow=base.food_inflow * n / 1000.0,
             food_cap=max(200, int(base.food_cap * n / 1000)), n_patches=max(2, int(16 * n / 1000)),
             h_max=max(base.h_max, h), h_init=h, p_hidden_change=0.0)
    return Config.from_dict(d)


def state_bytes(world: World) -> int:
    return int(sum(getattr(world.s, f).nbytes for f in world.s._fields))


def rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1 << 20) if platform.system() == "Darwin" else r / 1024.0


def kernel_breakdown(world: World, steps: int) -> dict:
    s, fp, ip = world.s, world.fp, world.ip
    gw, gh, inv = int(ip[k.Q_GW]), int(ip[k.Q_GH]), fp[k.P_INV_CELL]
    acc = {"grid": 0.0, "sense": 0.0, "think": 0.0, "move+collide+eat": 0.0, "birth+death": 0.0}
    for _ in range(steps):
        t = world.step_no
        k.spawn_food(t, s, fp, ip)
        a = time.perf_counter()
        k.build_grid(s.x, s.y, s.alive, inv, gw, gh, s.a_start, s.a_items, s.cursor)
        k.build_grid(s.fx, s.fy, s.f_alive, inv, gw, gh, s.f_start, s.f_items, s.cursor)
        b = time.perf_counter()
        k.sense(t, s, fp, ip)
        c = time.perf_counter()
        k.think(s)
        d = time.perf_counter()
        k.move(s, fp, ip)
        k.build_grid(s.x, s.y, s.alive, inv, gw, gh, s.a_start, s.a_items, s.cursor)
        k.collide(s, fp, ip)
        k.eat(s, fp, ip)
        e = time.perf_counter()
        k.process_deaths(t, s, fp, ip)
        nb = k.process_births(t, s, fp, ip)
        k.reseed(t, s, fp, ip, nb)
        f = time.perf_counter()
        world.step_no = t + 1
        s.counters[k.C_STEP] = world.step_no
        acc["grid"] += b - a
        acc["sense"] += c - b
        acc["think"] += d - c
        acc["move+collide+eat"] += e - d
        acc["birth+death"] += f - e
    tot = sum(acc.values())
    return {key: round(100 * v / tot, 1) for key, v in acc.items()}


def bench_one(n: int, h: int, steps: int, warm: int) -> dict:
    cfg = make_config(n, h)
    w = World(cfg)
    w.run(warm)
    w.drain_events()
    t0 = time.perf_counter()
    pops = []
    for i in range(steps):
        w.step()
        if i % 100 == 0:
            pops.append(int(w.s.counters[k.C_NALIVE]))
    dt = time.perf_counter() - t0
    w.drain_events()
    share = kernel_breakdown(w, 200)
    return {"n_target": n, "h": h, "world": cfg.world_w, "pop_mean": round(float(np.mean(pops)), 1),
            "steps_per_sec": round(steps / dt, 1), "ms_per_step": round(1000 * dt / steps, 3),
            "state_mb": round(state_bytes(w) / 1e6, 1), "rss_mb": round(rss_mb(), 1), "share_pct": share}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="속도·램 실측")
    ap.add_argument("--n", type=int, nargs="+", default=[500, 1000, 2000, 5000])
    ap.add_argument("--h", type=int, nargs="+", default=[16, 64])
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--warm", type=int, default=300)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    if args.threads > 0:
        numba.set_num_threads(args.threads)
    World(make_config(200, 16)).run(5)   # 컴파일
    info = {"machine": platform.machine(), "system": platform.system(), "python": platform.python_version(),
            "numpy": np.__version__, "numba": numba.__version__, "threads": numba.get_num_threads()}
    print(json.dumps(info))
    rows = []
    print(f"{'개체':>6} {'뇌':>4} {'세계':>6} {'실제 개체':>9} {'초당 스텝':>9} {'ms/스텝':>8} {'상태MB':>7} {'RSS MB':>7}  비중(%)")
    for n in args.n:
        for h in args.h:
            r = bench_one(n, h, args.steps, args.warm)
            rows.append(r)
            print(f"{n:>6} {h:>4} {r['world']:>6.0f} {r['pop_mean']:>9.0f} {r['steps_per_sec']:>9.0f} "
                  f"{r['ms_per_step']:>8.3f} {r['state_mb']:>7.1f} {r['rss_mb']:>7.0f}  {r['share_pct']}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"info": info, "rows": rows}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
