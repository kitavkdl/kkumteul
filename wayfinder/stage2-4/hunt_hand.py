"""사냥 손 개체: "사냥감 곁에서만 치면 이득인가"를 진화 없이 가른다(결과 문서 "결과 — 갈래 넷"의 다음 후보 ②).

물어뜯기 갈래 넷에서 사냥은 생기지 않았다. 공격 출력이 사냥감 곁이 아닌 곳에서도 나가 비용만 들고(친 개체-스텝은
공격 > 0 개체-스텝의 0.3%), 조건부로 치는 형질이 생기기 전에 깎인다고 읽었다(추정). 그래서 조건부로 치는 배선을
손으로 심어, 할 줄 알면 이득인지부터 본다.

체크포인트의 생물을 두 벌로 고친다. 둘 다 빈 숨은 칸 하나(n_hidden 자리)를 더한다(뇌 크기 비용을 같게):
    h_new = tanh(G × (사냥감 가까움×cos − θ))   — 다른 입력·되먹임·다른 출력과는 연결 0
    A(손 사냥꾼): 공격 출력 = tanh(2 × h_new − 1), 원래 숨은 칸 → 공격 연결은 0. 사냥감이 바로 앞 닿는 거리 안이면
                  공격 약 0.7, 아니면 공격 < 0(안 침, 비용 없음).
    B(안 치는 쪽): 새 칸은 공격에 안 닿고, 공격 출력 = tanh(−1) < 0 — 공격을 아예 안 낸다.
그 밖의 행동(다가가기 등)은 원래 생물 그대로다. 겨루기는 measure.assay_sides(A = tag 0, B = tag 1, r > 0이면 B가 이김).
물어뜯기 값마다(--bites) 겨루고, 같은 설정의 진단 판 하나에서 사냥 계수기(친 횟수, 쓰러뜨림, 머릿수)를 센다.
B는 공격을 안 내므로 진단 판의 사냥은 모두 A의 것이다.

닿는 거리: 중심 거리 prey_radius + agent_radius + 0.5 = 3.5 → 가까움 1 − (3.5 − 2)/16 = 0.906.
θ = 0.87이면 바로 앞 중심 거리 약 4.1 안에서 친다(조금 넘는 0.6만큼은 빗나가 비용만 든다).

    PYTHONPATH=<코드> python wayfinder/stage2-4/hunt_hand.py <체크포인트.npz> [--bites 0,0.6,1.0]
        [--n 400 --steps 8000 --reps 6] [--json <출력.json>]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from kkumteul import kernels as k
from kkumteul import measure
from kkumteul.config import ACT_ATTACK, EXT_PREY
from kkumteul.world import World

THETA = 0.87     # 사냥감 가까움×cos 문턱
GAIN = 40.0      # 문턱 날카로움
W_ATK = 2.0      # 새 칸 → 공격
B_ATK = -1.0     # 공격 편향


def plant(g: dict, col: int, hunter: bool) -> tuple[dict, int]:
    """유전체 묶음을 복사해 새 숨은 칸을 심는다. 숨은 칸이 가득 찬 개체는 빼고, 뺀 수를 돌려준다."""
    h_max = g["bh"].shape[1]
    keep = g["n_hidden"] < h_max
    g = {key: np.array(v[keep], copy=True) for key, v in g.items()}
    rows = np.arange(len(g["n_hidden"]))
    kk = g["n_hidden"].astype(np.int64)
    g["Win"][rows, :, kk] = 0.0
    g["Win"][rows, col, kk] = GAIN
    g["bh"][rows, kk] = -GAIN * THETA
    g["Wrec"][rows, kk, :] = 0.0
    g["Wrec"][rows, :, kk] = 0.0
    g["Wout"][rows, :, kk] = 0.0
    g["Wout"][:, ACT_ATTACK, :] = 0.0
    if hunter:
        g["Wout"][rows, ACT_ATTACK, kk] = W_ATK
    g["bout"][:, ACT_ATTACK] = B_ATK
    g["n_hidden"] = g["n_hidden"] + 1
    return g, int((~keep).sum())


def diagnose(cfg, ga: dict, gb: dict, n_each: int, steps: int, seed: int) -> dict:
    """measure.compete와 같은 꼴의 판 하나를 돌리고 사냥 계수기를 읽는다."""
    rng = np.random.default_rng(seed)
    acfg = measure.arena_config(cfg, seed, 2 * n_each)
    world = World(acfg, populate=False)
    for _ in range(int(0.5 * acfg.food_cap * acfg.food_energy / max(acfg.food_inflow, 1e-9))):
        k.spawn_food(0, world.s, world.fp, world.ip)
    ra = rng.choice(len(ga["n_hidden"]), size=n_each, replace=len(ga["n_hidden"]) < n_each)
    rb = rng.choice(len(gb["n_hidden"]), size=n_each, replace=len(gb["n_hidden"]) < n_each)
    px = rng.random(n_each) * acfg.world_w
    py = rng.random(n_each) * acfg.world_h
    ages = rng.integers(0, max(1, acfg.max_age // 2), size=n_each)
    for i in range(n_each):
        for g, row, tag in ((ga, ra[i], 0), (gb, rb[i], 1)):
            measure.inject_agent(world, g, int(row), tag, float(px[i]), float(py[i]),
                                 float(rng.random() * 2 * math.pi), int(ages[i]))
    c0 = world.s.counters.copy()
    atk_steps = saw = near = 0
    col = cfg.ext_in0 + EXT_PREY
    for _ in range(steps):
        world.step()
        a = world.s.alive & (world.s.tag == 0)
        atk_steps += int((world.s.act[a, ACT_ATTACK] > 0).sum())
        x = world.s.obs[a, col]
        saw += int((x != 0).sum())
        near += int((x > THETA).sum())
    world.drain_events()
    c = world.s.counters - c0
    tags = world.s.tag[world.s.alive]
    hits = int(c[k.C_PREY_HITS])
    return {"prey_hits": hits, "prey_killed": int(c[k.C_PREY_KILLED]), "hunters": int(c[k.C_HUNTERS]),
            "a_attack_steps": atk_steps, "a_saw_prey_steps": saw, "a_near_steps": near,
            "prey_alive_end": int(world.s.an_alive.sum()), "hit_frac": hits / atk_steps if atk_steps else None,
            "n_a": int((tags == 0).sum()), "n_b": int((tags == 1).sum())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("--bites", default="0,0.6,1.0")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--reps", type=int, default=6)
    p.add_argument("--seed", type=int, default=6000)
    p.add_argument("--json", type=Path)
    a = p.parse_args()
    side = measure.load_side(a.ckpt)
    cfg0 = side["cfg"]
    if not cfg0.prey_enabled:
        sys.exit("사냥감이 없는 세계다")
    col = cfg0.ext_in0 + EXT_PREY
    ga, drop_a = plant(side["genomes"], col, hunter=True)
    gb, drop_b = plant(side["genomes"], col, hunter=False)
    print(f"{a.ckpt}: 산 개체 {len(side['genomes']['n_hidden'])}, 숨은 칸이 가득 차 뺀 개체 {drop_a}. "
          f"사냥감 입력 칸 {col}, 문턱 {THETA}", flush=True)
    out = {"ckpt": str(a.ckpt), "theta": THETA, "gain": GAIN, "w_atk": W_ATK, "b_atk": B_ATK,
           "dropped": drop_a, "runs": []}
    for b in [float(x) for x in a.bites.split(",")]:
        cfg = cfg0.with_overrides([f"prey_bite={b}"])
        A = dict(side, cfg=cfg, genomes=ga, path=f"{a.ckpt} (손 사냥꾼)")
        B = dict(side, cfg=cfg, genomes=gb, path=f"{a.ckpt} (안 침)")
        res = measure.assay_sides(A, B, a.n, a.steps, a.reps, a.seed)
        diag = diagnose(cfg, ga, gb, a.n, a.steps, a.seed + 99)
        r = res["r_per_1000"]
        line = (f"물어뜯기 {b}: r {r['mean']:+.3f} ± {r['sem']:.3f}(중립 {res['neutral_r_per_1000']['mean']:+.3f}), "
                f"승자 {res['winner'] or '가를 수 없음'} — 사냥꾼 전멸 {res['a_extinct_runs']}, 안 치는 쪽 전멸 "
                f"{res['b_extinct_runs']} | 진단: 친 개체-스텝 {diag['prey_hits']}, 쓰러뜨림 {diag['prey_killed']}, "
                f"머릿수 {diag['hunters']}, 공격 > 0 개체-스텝 {diag['a_attack_steps']}, "
                f"사냥감 보임 {diag['a_saw_prey_steps']}, 문턱 넘음 {diag['a_near_steps']}, 끝 사냥감 {diag['prey_alive_end']}")
        print(line, flush=True)
        out["runs"].append({"bite": b, "assay": res, "diag": diag})
    if a.json:
        a.json.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
