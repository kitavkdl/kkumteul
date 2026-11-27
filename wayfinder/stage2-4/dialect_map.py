"""신호 방언 지도 — 체크포인트를 읽기만 하고 신호의 공간 자기상관(Moran I)을 잰다.

develop 2바퀴 실측 3(창발 소통 G3-A). 세계를 돌리지 않는다.

잰 것
- 관찰 신호 `sig`(체크포인트에 저장된 마지막 신호 두 칸): 맥락(곁의 먹이 등)이 섞인다.
- 탐침 신호: 모든 개체의 뇌에 같은 입력(은닉 0, 입력 0 하나 + 입력 칸마다 1.0)을 넣어 낸 신호 두 칸.
  맥락을 뺀 "그 개체가 신호를 내는 방식". 이것의 주성분 PC1·PC2와 바탕(입력 0) 두 칸을 쓴다.
- 색(유전 표지, 물려받고 변이)의 Moran I — 친족 구조가 얼마나 공간에 뭉쳐 있는지의 잣대.
- 친족 통제: 이웃 쌍 가운데 유전체 거리(뇌 가중치 전체)가 중앙값보다 먼 쌍만으로 다시 잰 Moran I.

Moran I(거리 띠 가중치 0/1, 원환 거리). 무작위 맞바꿈(자리 고정, 값 섞기)으로 영가설 분포를 만든다.

사용: python -m wayfinder.stage2-4.dialect_map 은 안 되므로 파일 경로로 부른다.
  python wayfinder/stage2-4/dialect_map.py <ckpt.npz> [--out out.json] [--perm 199]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ACT_SIG0 = 4  # kkumteul/config.py
BANDS = ((0.0, 8.0), (8.0, 24.0), (24.0, 64.0), (64.0, 128.0))


def torus_pairs(x, y, w, h, rmax):
    """원환 거리 rmax 안의 쌍 (i<j)과 거리. 격자로 거른다."""
    n = len(x)
    cell = rmax
    gw, gh = max(1, int(w // cell)), max(1, int(h // cell))
    cx = np.minimum((x / w * gw).astype(int), gw - 1)
    cy = np.minimum((y / h * gh).astype(int), gh - 1)
    buckets: dict[tuple[int, int], list[int]] = {}
    for i in range(n):
        buckets.setdefault((cx[i], cy[i]), []).append(i)
    I, J = [], []
    for (bx, by), idx in buckets.items():
        idx = np.asarray(idx)
        cand = []
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                key = ((bx + ox) % gw, (by + oy) % gh)
                if key in buckets:
                    cand.append(np.asarray(buckets[key]))
        cand = np.unique(np.concatenate(cand))
        ii, jj = np.meshgrid(idx, cand, indexing="ij")
        m = ii < jj
        I.append(ii[m]); J.append(jj[m])
    I = np.concatenate(I); J = np.concatenate(J)
    dx = np.abs(x[I] - x[J]); dx = np.minimum(dx, w - dx)
    dy = np.abs(y[I] - y[J]); dy = np.minimum(dy, h - dy)
    d = np.hypot(dx, dy)
    keep = d < rmax
    return I[keep], J[keep], d[keep]


def moran(v, I, J):
    """대칭 0/1 가중치(쌍 목록)의 Moran I. 쌍이 없으면 nan."""
    if len(I) == 0:
        return float("nan")
    z = v - v.mean()
    s2 = float((z * z).mean())
    if s2 <= 0:
        return float("nan")
    return float((z[I] * z[J]).mean() / s2)


def moran_perm(v, I, J, rng, n_perm):
    obs = moran(v, I, J)
    null = np.array([moran(rng.permutation(v), I, J) for _ in range(n_perm)])
    p = float((1 + np.sum(null >= obs)) / (n_perm + 1))
    return {"I": round(obs, 4), "null_mean": round(float(null.mean()), 4),
            "null_sd": round(float(null.std()), 4), "p_upper": round(p, 4), "pairs": int(len(I))}


def probe_signals(z, a):
    """같은 입력을 모두에게: 은닉 0에서 한 걸음. 반환 (n, 2*(1+n_in))."""
    Win, bh, Wout, bout, nh = z["Win"][a], z["bh"][a], z["Wout"][a], z["bout"][a], z["n_hidden"][a]
    n, n_in, H = Win.shape
    mask = (np.arange(H)[None, :] < nh[:, None]).astype(np.float64)
    probes = [np.zeros(n_in)] + [np.eye(n_in)[j] for j in range(n_in)]
    cols = []
    for x in probes:
        hid = np.tanh(bh + np.einsum("j,njk->nk", x, Win)) * mask
        out = np.tanh(bout[:, ACT_SIG0:ACT_SIG0 + 2] + np.einsum("nok,nk->no", Wout[:, ACT_SIG0:ACT_SIG0 + 2], hid))
        cols.append(out)
    return np.concatenate(cols, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out")
    ap.add_argument("--perm", type=int, default=199)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ck = Path(args.ckpt)
    z = np.load(ck)
    meta = json.loads(ck.with_suffix(".json").read_text())
    cfg = meta["config"]
    w, h = float(cfg["world_w"]), float(cfg["world_h"])
    a = np.flatnonzero(z["alive"])
    x, y = z["x"][a].astype(np.float64), z["y"][a].astype(np.float64)
    rng = np.random.default_rng(args.seed)

    sig = z["sig"][a].astype(np.float64)
    prof = probe_signals(z, a)
    pc = prof - prof.mean(0)
    u, sv, vt = np.linalg.svd(pc, full_matrices=False)
    pcs = u[:, :2] * sv[:2]
    var_frac = (sv ** 2 / np.sum(sv ** 2))[:2]
    color = z["color"][a].astype(np.float64)
    genome = np.concatenate([z["Win"][a].reshape(len(a), -1), z["Wout"][a].reshape(len(a), -1),
                             z["bh"][a], z["bout"][a]], axis=1).astype(np.float32)

    values = {"sig0_obs": sig[:, 0], "sig1_obs": sig[:, 1],
              "probe_base0": prof[:, 0], "probe_base1": prof[:, 1],
              "probe_pc1": pcs[:, 0], "probe_pc2": pcs[:, 1],
              "color_r": color[:, 0], "color_g": color[:, 1], "color_b": color[:, 2]}

    I0, J0, d0 = torus_pairs(x, y, w, h, BANDS[-1][1])
    gd = np.empty(len(I0))
    for s0 in range(0, len(I0), 20000):  # 쌍이 백만 단위라 나눠서(메모리)
        sl = slice(s0, s0 + 20000)
        gd[sl] = np.sqrt(np.sum((genome[I0[sl]] - genome[J0[sl]]) ** 2, axis=1))
    near = d0 < 24.0
    gmed = float(np.median(gd[near])) if near.any() else float("nan")

    out = {"ckpt": ck.name, "step": meta["step"], "n_alive": int(len(a)),
           "probe_var_frac_pc1_pc2": [round(float(v), 3) for v in var_frac],
           "sig_obs_sd": [round(float(v), 4) for v in sig.std(0)],
           "probe_pc_sd": [round(float(v), 4) for v in pcs.std(0)],
           "genome_dist_near_median": round(gmed, 3),
           "genome_dist_vs_space_corr": round(float(np.corrcoef(gd, d0)[0, 1]), 4),
           "bands": {}}
    for lo, hi in BANDS:
        m = (d0 >= lo) & (d0 < hi)
        mf = m & (gd > gmed)
        key = f"{int(lo)}-{int(hi)}"
        out["bands"][key] = {"genome_dist_mean": round(float(gd[m].mean()), 3) if m.any() else None,
                             "all": {}, "far_kin": {}}
        for name, v in values.items():
            out["bands"][key]["all"][name] = moran_perm(v, I0[m], J0[m], rng, args.perm)
            if name.startswith("sig"):  # 탐침은 유전체의 함수라 유전체로 쌍을 고르면 거꾸로 치우친다 → 아래 부분 상관으로
                out["bands"][key]["far_kin"][name] = moran_perm(v, I0[mf], J0[mf], rng, args.perm)
    # 부분 맨텔: 128 안의 쌍에서 |값 차이|와 공간 거리의 상관을, 둘 다 유전체 거리로 선형 회귀한 나머지로.
    # 음수 = 가까울수록 닮음(유전체 거리를 뺀 뒤에도).
    X = np.column_stack([np.ones_like(gd), gd])
    def resid(t):
        beta, *_ = np.linalg.lstsq(X, t, rcond=None)
        return t - X @ beta
    rd = resid(d0)
    out["partial_mantel_vs_space_given_genome"] = {}
    out["raw_mantel_vs_space"] = {}
    for name, v in values.items():
        dv = np.abs(v[I0] - v[J0])
        if dv.std() == 0:
            continue
        out["raw_mantel_vs_space"][name] = round(float(np.corrcoef(dv, d0)[0, 1]), 4)
        out["partial_mantel_vs_space_given_genome"][name] = round(float(np.corrcoef(resid(dv), rd)[0, 1]), 4)
    txt = json.dumps(out, ensure_ascii=False, indent=1)
    if args.out:
        Path(args.out).write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
