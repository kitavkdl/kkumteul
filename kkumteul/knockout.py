"""한쪽 감각 막기: 생물이 어떤 감각(신호 듣기, 닮음, 남이 먹는 것 보기, 흔적, 시야, 맛 기억)을 실제로 쓰는지 가른다.

신호 세기 같은 형질의 추세는 그 감각을 쓴다는 증거가 못 된다(쓰이지 않는 가중치도 떠돈다).
그래서 같은 생물을 두 벌로 만들어 한 벌(A, tag 0)만 그 감각을 못 받게 하고 겨루게 한다.
- A가 지면: 그 감각이 생존에 값을 한다(쓰인다).
- 가를 수 없으면: 쓰인다는 증거가 없다(이득이 측정 잡음보다 작을 수는 있다).
막는 방법은 유전체를 고치지 않고 **입력을 0으로** 만드는 것이다. 감각 함수가 끝난 뒤 A의 그 입력 칸을 지운다.
가중치를 0으로 하는 방식은 평생 학습(Hin)이 그 칸으로 다시 자랄 수 있어서 쓰지 않았다.

대조 둘:
- 그 감각이 없는 규칙(듣기·닮음·먹는 것 보기는 사회 감각을 끈 규칙, 흔적은 틈새를 끈 규칙)에서 같은 겨루기.
  두 벌이 똑같이 움직이므로 가를 수 없어야 한다.
- 시야를 막은 쪽은 뻔히 져야 한다(이 장치가 제대로 막는지 보는 양성 대조).
- 맛 블록(taste: 기억으로 물든 시야 8칸 + 기억 값 2칸, taste_vis, taste_val)은 맛 장치를 끈 규칙에서 같은 겨루기.
  시야 막기는 크기가 맞는 세계에서 물든 시야 8칸도 함께 지운다.
- 맛 장치를 켠 규칙에서 맛 블록을 막는 음성 대조는 체크포인트가 장치를 끈 규칙이면 뜻이 없다(칸이 이미 0이다).
  그때는 `--set taste_enabled=true`로 체크포인트 규칙을 덮어쓴다.

    python -m kkumteul.knockout split <체크포인트> --channel hear [--n 400 --steps 8000 --reps 6] [--set 키=값 ...]
    python -m kkumteul.knockout describe <체크포인트> [--steps 500]

describe는 겨루기 없이 신호를 내는 쪽과 듣는 쪽을 묘사한다(판정이 아니라 단서):
- 듣는 쪽: 들을 것이 있었나(입력이 0이 아닌 비율), 행동이 들은 것에 얼마나 기대나
  (같은 순간의 뇌를 들은 것 있이·없이 두 번 계산해 추력·회전이 바뀌는 크기. 시야를 지웠을 때와 견준다).
- 내는 쪽: 신호가 개체마다 고정된 꼬리표인가(개체 사이 분산), 상황에 따라 바뀌는 말인가(개체 안 분산),
  상황(에너지, 방금 먹음, 냄새, 호르몬)과 얼마나 같이 움직이나(상관).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np

from . import checkpoint as ckpt
from . import kernels as k
from . import measure
from . import stages as st
from .config import (ACT_SIG0, ACT_THRUST, ACT_TURN, IN_HEAR, IN_HORM, IN_KIN, IN_SAWEAT, IN_TRAIL,
                     N_INTERNAL, N_TASTE_IN, N_TASTE_VIS, TASTE_VAL, TASTE_VIS, Config)

CHANNELS = ("hear", "kin", "saweat", "social", "trail", "vision", "taste", "taste_vis", "taste_val")
# 그 감각이 없는 규칙(대조). 시야는 양성 대조라 없앤 규칙이 없다
OFF_RULES = {"hear": ["social_enabled=false"], "kin": ["social_enabled=false"],
             "saweat": ["social_enabled=false"], "social": ["social_enabled=false"],
             "trail": ["niche_enabled=false"], "vision": None,
             "taste": ["taste_enabled=false"], "taste_vis": ["taste_enabled=false"],
             "taste_val": ["taste_enabled=false"]}


def _base(cfg: Config) -> int:
    return 2 * cfg.vision_sectors + N_INTERNAL


def columns(cfg: Config, channel: str) -> list[int]:
    """감각 이름 → 입력 칸 번호."""
    b = _base(cfg)
    hear = list(range(b + IN_HEAR, b + IN_HEAR + 4))
    table = {"hear": hear, "kin": [b + IN_KIN], "saweat": [b + IN_SAWEAT],
             "social": hear + [b + IN_KIN, b + IN_SAWEAT],
             "trail": list(range(b + IN_TRAIL, b + IN_TRAIL + 3)),
             "vision": list(range(2 * cfg.vision_sectors))}
    if channel not in table and channel not in CHANNELS:
        raise ValueError(f"모르는 감각: {channel} (가능: {', '.join(CHANNELS)})")
    # 맛 블록은 크기가 맞는 세계에만 있다. 장치가 꺼져 있어도 칸 목록은 같다(그 칸은 0일 뿐이다)
    if cfg.n_in >= cfg.taste_in0 + N_TASTE_IN:
        t0 = cfg.taste_in0
        table["taste"] = list(range(t0, t0 + N_TASTE_IN))
        table["taste_vis"] = list(range(t0 + TASTE_VIS, t0 + TASTE_VIS + N_TASTE_VIS))
        table["taste_val"] = [t0 + TASTE_VAL, t0 + TASTE_VAL + 1]
        table["vision"] = table["vision"] + table["taste_vis"]
    if channel not in table:
        raise ValueError(f"이 세계에는 {channel} 칸이 없다(맛 블록 크기가 아니다: preset=taste_size)")
    return table[channel]


@contextlib.contextmanager
def knockout(cfg: Config, channel: str, tag: int = 0):
    """이 블록 안에서는 세계의 감각 함수가 끝날 때마다 tag가 같은 개체의 그 입력 칸을 0으로 지운다.
    감각 함수(kernels.sense, stages.social_sense, stages.trail_sense)는 World.step만 부르고,
    셋 다 생각(think) 전에 돈다. 블록을 나가면 원래 함수로 돌려 놓는다. 한 프로세스 안에서만 쓴다."""
    cols = np.asarray(columns(cfg, channel), dtype=np.int64)
    orig = (k.sense, st.social_sense, st.trail_sense)

    def wipe(s):
        rows = np.flatnonzero(s.alive & (s.tag == tag))
        if len(rows):
            s.obs[np.ix_(rows, cols)] = 0.0

    def wrap(f, state_pos):
        def g(*args):
            f(*args)
            wipe(args[state_pos])
        return g

    k.sense = wrap(orig[0], 1)            # sense(step, s, fp, ip)
    st.social_sense = wrap(orig[1], 0)    # social_sense(s, fs, isp)
    st.trail_sense = wrap(orig[2], 0)
    try:
        yield cols
    finally:
        k.sense, st.social_sense, st.trail_sense = orig


def _neutral(side: dict, cfg: Config, n: int, steps: int, reps: int, seed: int) -> dict:
    """막지 않은 같은 무리를 둘로 나눠 겨룬 값(측정 잡음). 막기 블록 밖에서 돈다."""
    rng = np.random.default_rng(seed + 7919)
    vals = []
    for i in range(reps):
        g1, g2 = measure._split(side["genomes"], rng)
        vals.append(measure.compete(cfg, g1, g2, n, steps, seed + 500 + i)["r"] * 1000)
    return measure._stats(vals)


def split(path: Path, channel: str, n: int = 400, steps: int = 8000, reps: int = 6, seed: int = 4000,
          rules: list[str] | None = None, log=None, base_overrides: list[str] | None = None) -> dict:
    """같은 생물 두 벌, A(tag 0)만 channel을 막고 겨룬다. 체크포인트 규칙과 대조 규칙에서.
    base_overrides를 주면 체크포인트 규칙을 그것으로 덮어쓴 설정을 체크포인트 규칙으로 쓴다(대조 규칙도 그 위에 얹는다)."""
    side = measure.load_side(path)
    base = side["cfg"].with_overrides(base_overrides) if base_overrides else side["cfg"]
    out = {"ckpt": str(path), "step": side["step"], "channel": channel,
           "columns": columns(base, channel), "n_each": n, "steps": steps, "reps": reps}
    if base_overrides:
        out["base_overrides"] = base_overrides
    plan = [("ckpt_rules", base)]
    off = OFF_RULES[channel] if rules is None else rules
    if off:
        plan.append(("control_rules", base.with_overrides(off)))
        out["control_overrides"] = off
    a_label, b_label = f"막은 쪽({channel})이", "원래 쪽이"
    for name, cfg in plan:
        a = dict(side, cfg=cfg, path=f"{path} ({channel} 막음)")
        b = dict(side, cfg=cfg)
        with knockout(cfg, channel, tag=0):
            res = measure.assay_sides(a, b, n, steps, reps, seed, neutral=False)
        res["neutral_r_per_1000"] = _neutral(side, cfg, n, steps, reps, seed)
        res["winner"] = measure.winner(res)
        res["verdict"] = measure.verdict(res, a_label, b_label)
        out[name] = res
        if log:
            print(json.dumps({name: {kk: res[kk] for kk in ("r_per_1000", "neutral_r_per_1000", "final_counts",
                                                          "verdict")}}, ensure_ascii=False), file=log, flush=True)
    return out


# ---- 묘사(판정 아님) ----
def _forward(s, rows, obs, h_prev, mind: bool) -> np.ndarray:
    """같은 순간의 뇌를 다시 계산한다(kernels._think_par / stages.think_plastic의 식). 출력 전체를 돌려준다."""
    Win = s.Win[rows].astype(np.float64)
    Wrec = s.Wrec[rows].astype(np.float64)
    if mind:
        al = s.alpha[rows][:, None, :].astype(np.float64)
        Win = Win + al * s.Hin[rows]
        Wrec = Wrec + al * s.Hrec[rows]
    nh = s.n_hidden[rows]
    H = Win.shape[2]
    mask = (np.arange(H)[None, :] < nh[:, None]).astype(np.float64)
    pre = s.bh[rows] + np.einsum("ri,rih->rh", obs, Win) + np.einsum("rj,rjh->rh", h_prev * mask, Wrec)
    hid = np.tanh(pre) * mask
    return np.tanh(s.bout[rows] + np.einsum("roh,rh->ro", s.Wout[rows].astype(np.float64), hid))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def describe(path: Path, steps: int = 500) -> dict:
    world, _ = ckpt.load(Path(path))
    s, cfg = world.s, world.cfg
    if not cfg.social_enabled:
        return {"ckpt": str(path), "note": "사회 감각이 꺼진 세계라 신호가 없다"}
    mind = cfg.mind_enabled
    b = _base(cfg)
    hear = columns(cfg, "hear")
    vis = columns(cfg, "vision")
    mv = [ACT_THRUST, ACT_TURN]
    heard_any, d_hear, d_vis, fit_err = [], [], [], []
    sig_rows = []         # (id, sig0, sig1, 에너지, 방금 먹음, 냄새, 스트레스, 도파민, 지루함, 피로)
    for _ in range(steps):
        h_prev = s.h.copy()
        ids_before = s.ids.copy()
        alive_before = s.alive.copy()
        world.run(1)
        # 이 스텝 내내 같은 개체가 살아 있었고(새로 태어난 자리 제외) 잠들지 않은 개체만
        rows = np.flatnonzero(s.alive & alive_before & (s.ids == ids_before) & (s.asleep <= 0.0))
        if not len(rows):
            continue
        obs = s.obs[rows].astype(np.float64)
        hp = h_prev[rows].astype(np.float64)
        full = _forward(s, rows, obs, hp, mind)
        fit_err.append(float(np.abs(full[:, ACT_SIG0:ACT_SIG0 + 2] - s.act[rows, ACT_SIG0:ACT_SIG0 + 2]).max()))
        o2 = obs.copy(); o2[:, hear] = 0.0
        o3 = obs.copy(); o3[:, vis] = 0.0
        d_hear.append(np.abs(_forward(s, rows, o2, hp, mind)[:, mv] - full[:, mv]).mean(0))
        d_vis.append(np.abs(_forward(s, rows, o3, hp, mind)[:, mv] - full[:, mv]).mean(0))
        heard_any.append(float((np.abs(obs[:, hear]).max(1) > 0.05).mean()))
        sig_rows.append(np.column_stack([s.ids[rows], s.sig[rows, 0], s.sig[rows, 1], obs[:, b + 0], obs[:, b + 5],
                                         obs[:, b + 6], obs[:, b + IN_HORM:b + IN_HORM + 4]]))
    world.drain_events()
    X = np.concatenate(sig_rows)
    ids = X[:, 0].astype(np.int64)
    ctx_names = ["energy", "ate", "smell", "stress", "dopamine", "boredom", "fatigue"]
    speaker = {}
    for c, name in ((1, "sig0"), (2, "sig1")):
        v = X[:, c]
        uniq, inv = np.unique(ids, return_inverse=True)
        cnt = np.bincount(inv)
        mean_i = np.bincount(inv, v) / cnt
        within = float(((v - mean_i[inv]) ** 2).mean())
        total = float(v.var())
        speaker[name] = {
            "mean_abs": float(np.abs(v).mean()),
            "total_var": total,
            # 1에 가까우면 개체마다 거의 고정된 꼬리표, 0에 가까우면 순간순간 바뀌는 신호
            "between_individual_share": float(1.0 - within / total) if total > 1e-12 else float("nan"),
            "corr_with": {nm: _corr(v, X[:, 3 + i]) for i, nm in enumerate(ctx_names)},
        }
    dh = np.mean(d_hear, 0); dv = np.mean(d_vis, 0)
    return {
        "ckpt": str(path), "step": int(world.step_no), "steps": steps, "agent_steps": int(len(X)),
        "forward_check_max_err": float(max(fit_err)) if fit_err else float("nan"),
        "listener": {
            "heard_something_frac": float(np.mean(heard_any)),
            "mean_abs_change_if_deaf": {"thrust": float(dh[0]), "turn": float(dh[1])},
            "mean_abs_change_if_blind": {"thrust": float(dv[0]), "turn": float(dv[1])},
            "deaf_over_blind": {"thrust": float(dh[0] / dv[0]) if dv[0] > 0 else float("nan"),
                                "turn": float(dh[1] / dv[1]) if dv[1] > 0 else float("nan")},
        },
        "speaker": speaker,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m kkumteul.knockout")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("split")
    p.add_argument("ckpt", type=Path)
    p.add_argument("--channel", choices=CHANNELS, required=True)
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--reps", type=int, default=6)
    p.add_argument("--seed", type=int, default=4000)
    p.add_argument("--control", nargs="*", default=None, help="대조 규칙 덮어쓰기(주지 않으면 감각마다 정한 것)")
    p.add_argument("--set", dest="base_set", action="append", default=None, metavar="키=값",
                   help="체크포인트 규칙 덮어쓰기(여러 번 가능). 예: --set taste_enabled=true")
    d = sub.add_parser("describe")
    d.add_argument("ckpt", type=Path)
    d.add_argument("--steps", type=int, default=500)
    a = ap.parse_args(argv)
    if a.cmd == "split":
        out = split(a.ckpt, a.channel, a.n, a.steps, a.reps, a.seed, a.control, log=sys.stderr,
                    base_overrides=a.base_set)
    else:
        out = describe(a.ckpt, a.steps)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
