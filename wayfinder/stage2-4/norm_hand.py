"""손 집행자: "표시된 이웃을 보고 돌아서 치는" 배선을 손으로 심어, 집행이 할 줄 알면 이득인지를 진화 없이 가른다
(규범의 씨앗 설계 시험 17 앞쪽, 계획 과제 15). 본보기는 hunt_hand.py의 plant.

유전체 묶음을 두 벌로 고친다. 둘 다 빈 숨은 칸 둘(n_hidden 자리부터)을 더하고 입력 쪽 배선까지 같다(뇌 크기 비용을 같게):
    h_atk  = tanh(G × ((표시 시야 앞 두 칸 3·4의 평균) − θ))       — 다른 입력·되먹임과는 연결 0
    h_turn = tanh(G_T × (오른쪽 네 칸 4~7의 합 − 왼쪽 네 칸 0~3의 합))
    집행자(태그 1·2·3이 같은 몸): 공격 출력 = tanh(W_ATK × h_atk + B_ATK), 회전 출력에 W_TURN × h_turn을 더한다.
    거주민(태그 0): 새 칸은 출력에 안 닿는다. 공격 행은 편향 −1·출력 행 0(measure.fix_resident_attack과 같다)이라
    충동이 없으면 공격이 0이다. 집행자도 원래 숨은 칸 → 공격 연결은 0(공격은 손 배선만 낸다).
그 밖의 행동은 원래 생물 그대로다. 마당은 태그 2에 norm_block_tag = 2(표시된 대상을 못 침), 태그 3에 norm_block3_tag = 3
(피해자가 자기일 때만 침)을 건다 — 같은 몸에서 집행 능력만 뺀 쌍둥이(measure.compete_multi).

회전 부호: kernels.move는 heading += turn × max_turn이고 표시 시야 칸 번호는 (이웃 방향 − heading)이 클수록 크다
(stages._see_norm). 그래서 오른쪽(큰 번호) 칸에 보이면 회전 +로 돌아 표적이 가운데로 온다(시험이 양쪽을 확인한다).
θ·G: 닿는 거리(중심 거리 2.5, 가까움 1 − 1.5/16 = 0.906)에서 밝기 ≥ 0.5(경과 ≤ τ/2)인 표시를 정면에 두면 친다
(θ = 0.906 × 0.5). 한계: 정면이 아니면(한 칸만 걸침) 밝기 1이어야 넘고, 밝기와 거리를 곱으로만 보므로 밝은 표시는
닿는 거리 밖(가까움 ≥ θ, 중심 거리 약 9.7 안)에서도 쳐서 빗나가며 비용만 들거나 그때 닿아 있는 표시 안 된 이웃을
가로챌 수 있다. 늙은(밝기 < 0.5) 표시에는 덜 친다. 손 배선은 임의다(설계 약점 6) — 결과 문서에 적는다.

    PYTHONPATH=<코드> python wayfinder/stage2-4/norm_hand.py <체크포인트.npz> [--n-res 300 --n-each 60]
        [--steps 3000 --reps 4 --seed 7000] [--set key=value ...] [--json <출력.json>]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from kkumteul import measure
from kkumteul.config import ACT_ATTACK, ACT_TURN, N_NORM_VIS, NORM_VIS, Config

N_NEW = 2            # 새 숨은 칸 수(공격 문턱, 회전)
THETA = 0.906 * 0.5  # 앞 두 칸 평균의 문턱: 닿는 거리 정면에서 밝기 ≥ 0.5
GAIN = 40.0          # 문턱 날카로움
W_ATK = 2.0          # 공격 칸 → 공격
B_ATK = -1.0         # 공격 편향(새 개체와 같다)
G_TURN = 2.0         # 좌우 차의 날카로움
W_TURN = 3.0         # 회전 칸 → 회전
FRONT = (3, 4)       # 시야 가운데 두 칸(8칸 기준)


def plant_enforcer(g: dict, cfg: Config, enforcer: bool) -> tuple[dict, int]:
    """유전체 묶음을 복사해 새 숨은 칸 둘을 심는다(집행자면 출력까지). 숨은 칸이 모자란 개체는 빼고, 뺀 수를 돌려준다."""
    if cfg.vision_sectors != N_NORM_VIS:
        raise ValueError(f"손 집행자는 표시 시야가 {N_NORM_VIS}칸이어야 한다(vision_sectors={cfg.vision_sectors})")
    c0 = cfg.norm_in0 + NORM_VIS
    if g["Win"].shape[1] < c0 + N_NORM_VIS:
        raise ValueError("유전체에 규범 블록 입력 칸이 없다: measure.norm_arena_side로 norm_size 크기로 읽는다")
    h_max = g["bh"].shape[1]
    keep = g["n_hidden"] + N_NEW <= h_max
    g = {key: np.array(v[keep], copy=True) for key, v in g.items()}
    rows = np.arange(len(g["n_hidden"]))
    ka = g["n_hidden"].astype(np.int64)      # 공격 문턱 칸
    kt = ka + 1                              # 회전 칸
    for kk in (ka, kt):
        g["Win"][rows, :, kk] = 0.0
        g["Wrec"][rows, kk, :] = 0.0
        g["Wrec"][rows, :, kk] = 0.0
        g["Wout"][rows, :, kk] = 0.0
    for sec in FRONT:
        g["Win"][rows, c0 + sec, ka] = GAIN / len(FRONT)
    g["bh"][rows, ka] = -GAIN * THETA
    for sec in range(N_NORM_VIS):
        g["Win"][rows, c0 + sec, kt] = G_TURN if sec >= N_NORM_VIS // 2 else -G_TURN
    g["bh"][rows, kt] = 0.0
    g["Wout"][:, ACT_ATTACK, :] = 0.0
    g["bout"][:, ACT_ATTACK] = B_ATK
    if enforcer:
        g["Wout"][rows, ACT_ATTACK, ka] = W_ATK
        g["Wout"][rows, ACT_TURN, kt] = W_TURN
    g["n_hidden"] = g["n_hidden"] + N_NEW
    return g, int((~keep).sum())


def arena(g: dict, cfg: Config, n_res: int, n_each: int) -> tuple[list, int, Config]:
    """겨루기 마당 묶음: [(거주민, 0, n_res), (집행자, 1, n_each), (같은 집행자, 2, n_each), (같은 집행자, 3, n_each)],
    뺀 개체 수, 마당 설정(norm 켬, 태그 2 막기 전부, 태그 3 막기 3자만). 거주민과 집행자는 같은 개체 집합에서 만든다."""
    acfg = cfg.with_overrides(["norm_enabled=true", "norm_block_tag=2", "norm_block3_tag=3"])
    res, dropped = plant_enforcer(g, acfg, enforcer=False)
    enf, _ = plant_enforcer(g, acfg, enforcer=True)
    return [(res, 0, n_res), (enf, 1, n_each), (enf, 2, n_each), (enf, 3, n_each)], dropped, acfg


def main() -> None:
    p = argparse.ArgumentParser(description="손 집행자 겨루기(태그 0 거주민, 1 집행자, 2 막기 전부, 3 막기 3자만)")
    p.add_argument("ckpt", type=Path)
    p.add_argument("--n-res", type=int, default=300)
    p.add_argument("--n-each", type=int, default=60)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--reps", type=int, default=4)
    p.add_argument("--seed", type=int, default=7000)
    p.add_argument("--set", action="append", default=[], metavar="key=value", help="마당 설정에 덧붙인다(예: norm_impulse=0.015)")
    p.add_argument("--json", type=Path)
    a = p.parse_args()
    side = measure.norm_arena_side(a.ckpt, ["conflict_enabled=true", *a.set])
    groups, dropped, acfg = arena(side["genomes"], side["cfg"], a.n_res, a.n_each)
    print(f"{a.ckpt}: 산 개체 {len(side['genomes']['n_hidden'])}, 숨은 칸이 모자라 뺀 개체 {dropped}, θ {THETA:.3f}",
          flush=True)
    out = {"ckpt": str(a.ckpt), "theta": THETA, "gain": GAIN, "w_atk": W_ATK, "b_atk": B_ATK, "g_turn": G_TURN,
           "w_turn": W_TURN, "dropped": dropped, "sets": a.set, "runs": []}
    for r in range(a.reps):
        res = measure.compete_multi(acfg, groups, a.steps, a.seed + r)
        net = res["net_income_by_tag"]
        line = " | ".join(f"태그 {t}: 산 {res['alive_by_tag'][t]}, 벌 {res['norm_by_tag'][t]['pun']:.0f}, "
                          f"순수입/스텝 {net[t]['net_per_step'] if net[t]['net_per_step'] is None else round(net[t]['net_per_step'], 5)}"
                          for t in sorted(res["alive_by_tag"]))
        print(f"판 {r}: {line}", flush=True)
        out["runs"].append(res)
    if a.json:
        a.json.write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    sys.exit(main())
