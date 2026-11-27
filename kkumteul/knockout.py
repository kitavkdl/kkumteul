"""한쪽 감각 막기: 생물이 어떤 감각(신호 듣기, 닮음, 남이 먹는 것 보기, 흔적, 시야, 맛 기억, 규범 표시)을 실제로 쓰는지 가른다.

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
- 규범 블록(norm: 표시 시야 8칸 + 내 밝기·곁 밝기·방금 벌 + 내 삼가기 기억, norm_vis, norm_scalars, norm_mem)은
  규범 장치를 끈 규칙(NORM_OFF)에서 같은 겨루기. 시야 막기는 크기가 맞는 세계에서 표시 시야 8칸도 함께 지운다.
  감각 함수는 넷(kernels.sense, stages.social_sense, stages.trail_sense, stages.norm_sense)이고 zero 막기는 넷 뒤 모두에서 지운다
  (norm_sense를 안 감싸면 규범 칸이 도로 채워져 막기가 조용히 무동작이 된다).

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
from .config import (ACT_ATTACK, ACT_SIG0, ACT_THRUST, ACT_TURN, IN_HEAR, IN_HORM, IN_KIN, IN_SAWEAT, IN_TRAIL,
                     N_INTERNAL, N_NORM_IN, N_NORM_VIS, N_TASTE_IN, N_TASTE_VIS, NORM_LOCAL, NORM_MEM, NORM_SAWPUN,
                     NORM_SELF, NORM_VIS, TASTE_VAL, TASTE_VIS, Config)

CHANNELS = ("hear", "kin", "saweat", "social", "trail", "vision", "taste", "taste_vis", "taste_val",
            "norm", "norm_vis", "norm_scalars", "norm_mem")
# 맛 장치를 끄는 한 묶음. 맛 기억·배움 속도 유전자·보고 배우기·보여 주기(teach)는 서로 기대므로(설정 검사) 함께 꺼야 한다
# (feed_enabled는 맛과 무관해 안 넣는다). 판정 도구(taste_judge·hand_learner)도 이 목록을 쓴다 — 여기 한 곳만 고치면 된다
TASTE_OFF = ["taste_enabled=false", "taste_gene_enabled=false", "soc_taste_enabled=false", "teach_enabled=false"]
# 규범 장치를 끄는 한 묶음. 의존 키(표시 원천·잡음·관찰 모드·막기)를 함께 기본으로 되돌려야 그것들을 켠 체크포인트
# 규칙에 한꺼번에 덮어써도 설정 검사에 안 걸린다. norm_impulse(위반 공급)는 감각이 아니라 안 건드린다
NORM_OFF = ["norm_enabled=false", "norm_mode=full", "norm_mark_src=take", "norm_mark_noise=0",
            "norm_mem_obs_mode=outcome", "norm_block_tag=-1", "norm_block3_tag=-1"]
# 불활성 대조(입력·조준·비용은 같고 벌·위축만 없음): 의존 키가 없는 한 줄
NORM_INERT = ["norm_mode=inert"]
# 그 감각이 없는 규칙(대조). 시야는 양성 대조라 없앤 규칙이 없다
OFF_RULES = {"hear": ["social_enabled=false"], "kin": ["social_enabled=false"],
             "saweat": ["social_enabled=false"], "social": ["social_enabled=false"],
             "trail": ["niche_enabled=false"], "vision": None,
             "taste": list(TASTE_OFF), "taste_vis": list(TASTE_OFF), "taste_val": list(TASTE_OFF),
             "norm": list(NORM_OFF), "norm_vis": list(NORM_OFF), "norm_scalars": list(NORM_OFF),
             "norm_mem": list(NORM_OFF)}


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
    # 규범 블록(맛 블록 바로 뒤)도 크기가 맞는 세계에만 있다
    if cfg.n_in >= cfg.norm_in0 + N_NORM_IN:
        c0 = cfg.norm_in0
        table["norm"] = list(range(c0, c0 + N_NORM_IN))
        table["norm_vis"] = list(range(c0 + NORM_VIS, c0 + NORM_VIS + N_NORM_VIS))
        table["norm_scalars"] = [c0 + NORM_SELF, c0 + NORM_LOCAL, c0 + NORM_SAWPUN]
        table["norm_mem"] = [c0 + NORM_MEM]
        table["vision"] = table["vision"] + table["norm_vis"]
    if channel not in table:
        if channel.startswith("norm"):
            raise ValueError(f"이 세계에는 {channel} 칸이 없다(규범 블록 크기가 아니다: preset=norm_size)")
        raise ValueError(f"이 세계에는 {channel} 칸이 없다(맛 블록 크기가 아니다: preset=taste_size)")
    return table[channel]


MODES = ("zero", "shuffle")
# 그 칸을 쓰는 감각 함수(섞기는 그 함수 뒤에서만 한 번 돈다)
WRITER = {"hear": "social", "kin": "social", "saweat": "social", "social": "social", "trail": "trail",
          "vision": "sense", "taste": "sense", "taste_vis": "sense", "taste_val": "sense",
          "norm": "norm", "norm_vis": "norm", "norm_scalars": "norm", "norm_mem": "norm"}


class KnockoutCols(np.ndarray):
    """막은 칸 번호 배열. `.stats`에 섞은 호출 수(calls), 고른 행 수(rows), 실제로 값이 바뀐 행 수(changed)."""
    stats: dict


def _shuffle_rows(s, cols: np.ndarray, tag: int, rng: np.random.Generator, frac: float = 1.0,
                  stats: dict | None = None) -> None:
    """tag가 같은 산 개체들 사이에서 cols 묶음을 행째로 바꿔 끼운다(한 번에 한 순열, 묶음 안 칸은 같이 움직임).
    frac < 1이면 그 몫만 무작위로 골라 그들끼리 바꾼다(나머지는 제 것을 듣는다)."""
    rows = np.flatnonzero(s.alive & (s.tag == tag))
    if frac < 1.0 and len(rows):
        rows = np.sort(rng.choice(rows, size=int(round(frac * len(rows))), replace=False))
    if stats is not None:
        stats["calls"] += 1
        stats["rows"] += int(len(rows))
    if len(rows) < 2:
        return
    block = s.obs[np.ix_(rows, cols)]
    new = block[rng.permutation(len(rows))]
    if stats is not None:
        stats["changed"] += int(np.any(new != block, axis=1).sum())
    s.obs[np.ix_(rows, cols)] = new


@contextlib.contextmanager
def knockout(cfg: Config, channel: str, tag: int = 0, mode: str = "zero", seed: int = 0, frac: float = 1.0):
    """이 블록 안에서는 tag가 같은 개체의 그 입력 칸을 막는다. 블록을 나가면 원래 함수로 돌려 놓는다.
    - zero: 세계의 감각 함수가 끝날 때마다 0으로 지운다(감각 함수 넷 뒤 모두 — 옛 결과와 같은 동작).
    - shuffle: 그 칸을 쓰는 감각 함수 뒤에서만, 스텝마다 한 번, 같은 tag의 산 개체끼리 행째로 바꿔 끼운다.
      입력의 크기·분포는 그대로이고 듣는 개체의 처지와의 짝만 깨진다. 난수는 지역 생성기(seed) —
      세계 난수 흐름은 그대로다.
    감각 함수(kernels.sense, stages.social_sense, stages.trail_sense, stages.norm_sense)는 World.step만 부르고,
    넷 다 생각(think) 전에 돈다. 한 프로세스 안에서만 쓴다.

    주의(시야 막기 + 규범 세계): 시야 칸 묶음에 표시 시야가 들어 있어, sense 뒤에 지운 표시 시야를 norm_sense가 도로
    채워도 norm_sense 뒤에 한 번 더 지운다(zero는 넷 뒤 모두에서 같은 칸을 지운다)."""
    if mode not in MODES:
        raise ValueError(f"모르는 막기 방식: {mode} (가능: {', '.join(MODES)})")
    cols = np.asarray(columns(cfg, channel), dtype=np.int64)
    out = cols.view(KnockoutCols)
    out.stats = {"calls": 0, "rows": 0, "changed": 0}
    orig = (k.sense, st.social_sense, st.trail_sense, st.norm_sense)
    rng = np.random.default_rng(seed)

    def wipe(s):
        rows = np.flatnonzero(s.alive & (s.tag == tag))
        if len(rows):
            s.obs[np.ix_(rows, cols)] = 0.0

    def shuf(s):
        _shuffle_rows(s, cols, tag, rng, frac, out.stats)

    def wrap(f, state_pos, act):
        def g(*args):
            f(*args)
            act(args[state_pos])
        return g

    if mode == "zero":
        k.sense = wrap(orig[0], 1, wipe)            # sense(step, s, fp, ip)
        st.social_sense = wrap(orig[1], 0, wipe)    # social_sense(s, fs, isp)
        st.trail_sense = wrap(orig[2], 0, wipe)
        st.norm_sense = wrap(orig[3], 1, wipe)      # norm_sense(step, s, fs, isp)
    else:
        w = WRITER[channel]
        if w == "sense":
            k.sense = wrap(orig[0], 1, shuf)
        elif w == "social":
            st.social_sense = wrap(orig[1], 0, shuf)
        elif w == "norm":
            st.norm_sense = wrap(orig[3], 1, shuf)
        else:
            st.trail_sense = wrap(orig[2], 0, shuf)
    try:
        yield out
    finally:
        k.sense, st.social_sense, st.trail_sense, st.norm_sense = orig


def _neutral(side: dict, cfg: Config, n: int, steps: int, reps: int, seed: int, twin: bool = False) -> dict:
    """막지 않은 같은 무리를 둘로 나눠 겨룬 값(측정 잡음). 막기 블록 밖에서 돈다.
    twin이면 나누지 않고 같은 무리 두 벌을 쌍둥이로 겨룬다(빈 막기 대조)."""
    rng = np.random.default_rng(seed + 7919)
    vals = []
    for i in range(reps):
        if twin:
            g = side["genomes"]
            vals.append(measure.compete(cfg, g, g, n, steps, seed + 500 + i, twin=True)["r"] * 1000)
            continue
        g1, g2 = measure._split(side["genomes"], rng)
        vals.append(measure.compete(cfg, g1, g2, n, steps, seed + 500 + i)["r"] * 1000)
    return measure._stats(vals)


def split(path: Path, channel: str, n: int = 400, steps: int = 8000, reps: int = 6, seed: int = 4000,
          rules: list[str] | None = None, log=None, base_overrides: list[str] | None = None,
          mode: str = "zero", frac: float = 1.0, twin: bool = False) -> dict:
    """같은 생물 두 벌, A(tag 0)만 channel을 막고 겨룬다. 체크포인트 규칙과 대조 규칙에서.
    base_overrides를 주면 체크포인트 규칙을 그것으로 덮어쓴 설정을 체크포인트 규칙으로 쓴다(대조 규칙도 그 위에 얹는다).
    mode="shuffle"이면 0으로 지우는 대신 A끼리 그 칸을 바꿔 끼운다(frac은 스텝마다 섞는 A의 몫)."""
    side = measure.load_side(path)
    base = side["cfg"].with_overrides(base_overrides) if base_overrides else side["cfg"]
    out = {"ckpt": str(path), "step": side["step"], "channel": channel, "mode": mode,
           "columns": columns(base, channel), "n_each": n, "steps": steps, "reps": reps}
    if mode != "zero":
        out["frac"] = frac
    if twin:
        out["twin"] = True
    if base_overrides:
        out["base_overrides"] = base_overrides
    plan = [("ckpt_rules", base)]
    off = OFF_RULES[channel] if rules is None else rules
    if off:
        plan.append(("control_rules", base.with_overrides(off)))
        out["control_overrides"] = off
    a_label = f"막은 쪽({channel})이" if mode == "zero" else f"섞은 쪽({channel})이"
    b_label = "원래 쪽이"
    for name, cfg in plan:
        a = dict(side, cfg=cfg, path=f"{path} ({channel} 막음)")
        b = dict(side, cfg=cfg)
        with knockout(cfg, channel, tag=0, mode=mode, seed=seed + 31, frac=frac) as ko_cols:
            res = measure.assay_sides(a, b, n, steps, reps, seed, neutral=False, twin=twin)
        if mode != "zero":
            st_ = ko_cols.stats
            res["shuffle_stats"] = dict(st_)
            # 섞을 짝이 없으면(A가 줄어 한 마리) 값이 그대로다. 실제로 바뀐 행의 몫
            res["changed_row_frac"] = st_["changed"] / st_["rows"] if st_["rows"] else 0.0
        res["neutral_r_per_1000"] = _neutral(side, cfg, n, steps, reps, seed, twin=twin)
        nsd = res["neutral_r_per_1000"].get("sd", float("nan"))
        # 중립 흔들림으로 가를 수 있는 가장 작은 r(어림: t95 × 중립 sd / √반복, 임시)
        res["min_detectable_r_per_1000"] = float(measure.t95(reps - 1) * nsd / np.sqrt(reps)) if reps > 1 else None
        res["winner"] = measure.winner(res)
        res["verdict"] = measure.verdict(res, a_label, b_label)
        out[name] = res
        if log:
            print(json.dumps({name: {kk: res[kk] for kk in ("r_per_1000", "neutral_r_per_1000", "final_counts",
                                                          "min_detectable_r_per_1000", "verdict")}},
                             ensure_ascii=False), file=log, flush=True)
    return out


# ---- 묘사(판정 아님) ----
def _forward(s, rows, obs, h_prev, mind: bool, Hin=None, Hrec=None, Win=None) -> np.ndarray:
    """같은 순간의 뇌를 다시 계산한다(kernels._think_par / stages.think_plastic의 식). 출력 전체를 돌려준다.
    평생 학습은 생각한 뒤에 Hin·Hrec을 고치므로, 그 스텝의 생각과 맞추려면 스텝 전에 뜬 사본(전체 배열)을 준다.
    Win(rows만큼, 타고난 것 + 배운 것을 이미 더한 값)을 주면 그것을 그대로 쓴다(가중치를 섞은 기준선용).
    노드 섭동의 잡음은 넣지 않는다."""
    Wrec = s.Wrec[rows].astype(np.float64)
    if mind:
        al = s.alpha[rows][:, None, :].astype(np.float64)
        Wrec = Wrec + al * (s.Hrec if Hrec is None else Hrec)[rows]
    if Win is None:
        Win = s.Win[rows].astype(np.float64)
        if mind:
            Win = Win + al * (s.Hin if Hin is None else Hin)[rows]
    nh = s.n_hidden[rows]
    H = Win.shape[2]
    mask = (np.arange(H)[None, :] < nh[:, None]).astype(np.float64)
    pre = s.bh[rows] + np.einsum("ri,rih->rh", obs, Win) + np.einsum("rj,rjh->rh", h_prev * mask, Wrec)
    hid = np.tanh(pre) * mask
    return np.tanh(s.bout[rows] + np.einsum("roh,rh->ro", s.Wout[rows].astype(np.float64), hid))


CTX_NAMES = ("energy", "ate", "smell", "stress", "dopamine", "boredom", "fatigue")


def context_columns(cfg: Config) -> list[int]:
    """내는 쪽 상황 칸: 에너지·방금 먹음·냄새(내부 감각, 시야 뒤 0·5·6번), 호르몬 넷(단계 입력 앞 넷).
    (11/07 바로잡음: 전에는 셋을 단계 입력 시작에서 세어 스트레스 호르몬·들은 신호 두 칸을 집었다.)"""
    iv = 2 * cfg.vision_sectors
    b = _base(cfg)
    return [iv + 0, iv + 5, iv + 6] + list(range(b + IN_HORM, b + IN_HORM + 4))


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
    ctx_cols = context_columns(cfg)
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
        sig_rows.append(np.column_stack([s.ids[rows], s.sig[rows, 0], s.sig[rows, 1], obs[:, ctx_cols]]))
    world.drain_events()
    X = np.concatenate(sig_rows)
    ids = X[:, 0].astype(np.int64)
    ctx_names = CTX_NAMES
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


# ---- 상호정보량(설계 docs/superpowers/specs/2026-11-07-signal-measure-design.md) ----
def _codes(x, bins: int) -> tuple[np.ndarray, list[int]]:
    """분위 칸 부호. 2차원이면 열마다 칸을 나눠 결합 부호를 만든다. 같은 값(동점)은 같은 칸에 든다
    (분위 경계가 겹치면 칸이 줄어든다). 실제로 차지한 칸 수도 돌려준다."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    code = np.zeros(len(x), dtype=np.int64)
    used = []
    for c in range(x.shape[1]):
        col = x[:, c]
        edges = np.unique(np.quantile(col, np.linspace(0.0, 1.0, bins + 1)[1:-1])) if len(col) else np.zeros(0)
        b = np.searchsorted(edges, col, side="right")
        code = code * (len(edges) + 1) + b
        used.append(int(len(np.unique(b))))
    return code, used


def _mi_codes(cx: np.ndarray, cy: np.ndarray) -> float:
    n = len(cx)
    if n == 0:
        return 0.0
    _, ix = np.unique(cx, return_inverse=True)
    _, iy = np.unique(cy, return_inverse=True)
    nx, ny = ix.max() + 1, iy.max() + 1
    if nx < 2 or ny < 2:
        return 0.0
    joint = np.bincount(ix * ny + iy, minlength=nx * ny).astype(np.float64) / n
    px = np.bincount(ix, minlength=nx) / n
    py = np.bincount(iy, minlength=ny) / n
    pxy = joint.reshape(nx, ny)
    m = pxy > 0
    return float((pxy[m] * np.log2(pxy[m] / (px[:, None] * py[None, :])[m])).sum())


def mi_binned(x, y, bins: int = 8) -> float:
    """분위 칸으로 나눈 표본 상호정보량(비트). 치우침은 빼지 않는다(mi_diff가 같은 크기의 영가설로 뺀다)."""
    return _mi_codes(_codes(x, bins)[0], _codes(y, bins)[0])


def _fold_of(ids: np.ndarray, folds: int, seed: int) -> np.ndarray:
    uniq, inv = np.unique(ids, return_inverse=True)
    perm = np.random.default_rng(seed).permutation(len(uniq))
    return (perm % folds)[inv]


def _signflip_p(d: np.ndarray) -> float:
    """겹 값들의 평균이 0보다 큰가: 부호 뒤집기 정확 검정(한쪽). 겹이 12개 넘으면 무작위 4,096번."""
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    if n == 0 or np.all(d == 0):
        return 1.0
    if n <= 12:
        signs = ((np.arange(2 ** n)[:, None] >> np.arange(n)[None, :]) & 1) * 2 - 1
    else:
        signs = np.random.default_rng(0).choice([-1, 1], size=(4096, n))
    return float(((signs * d).mean(1) >= d.mean() - 1e-15).mean())


def _interval(d: np.ndarray) -> dict:
    d = np.asarray(d, dtype=np.float64)
    m = float(d.mean())
    sem = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
    t = measure.t95(len(d) - 1) if len(d) > 1 else float("nan")
    lo, hi = m - t * sem, m + t * sem
    return {"mean": m, "sem": sem, "lo": lo, "hi": hi, "above_zero": bool(lo > 0.0), "p": _signflip_p(d),
            "folds": [float(v) for v in d]}


def mi_diff(x, y, y_nulls: list, ids: np.ndarray, folds: int = 10, bins: int = 8, seed: int = 0,
            xbins: int | None = None) -> dict:
    """개체 id로 겹을 나눠(같은 개체는 한 겹에만) 겹마다 I(x; y) − 평균 I(x; 영가설 y)을 구한다.
    겹 값들의 평균, t 95% 구간, 부호 뒤집기 p. 아래 끝 > 0이면 above_zero."""
    fold = _fold_of(ids, folds, seed)
    cx, ux = _codes(x, xbins or bins)
    cy, uy = _codes(y, bins)
    cn = [_codes(yn, bins)[0] for yn in y_nulls]
    obs_v, null_v, d = [], [], []
    for f in range(folds):
        m = fold == f
        if not m.any():
            continue
        o = _mi_codes(cx[m], cy[m])
        nl = float(np.mean([_mi_codes(cx[m], c[m]) for c in cn]))
        obs_v.append(o); null_v.append(nl); d.append(o - nl)
    out = _interval(np.asarray(d))
    out.update(mi=float(np.mean(obs_v)), mi_null=float(np.mean(null_v)), bins_used_x=ux, bins_used_y=uy)
    return out


def _id_key(ids: np.ndarray, seed: int) -> np.ndarray:
    """개체 id → 고정된 무작위 순서 열쇠(splitmix64)."""
    off = np.uint64((int(seed) * 0x9E3779B97F4A7C15) % (1 << 64))
    with np.errstate(over="ignore"):
        z = ids.astype(np.uint64) + off
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return z


def neighbor_index(ids: np.ndarray, steps: np.ndarray, seed: int, strata: np.ndarray | None = None) -> np.ndarray:
    """영가설 짝: 같은 스텝(그리고 같은 층) 안에서 개체를 고정된 무작위 순서로 늘어놓고 바로 다음 개체(돌아감).
    순서가 스텝마다 같아 같은 짝이 이어지므로, 개체 꼬리표 × 오래 가는 상황이 만드는 부풂이 영가설에도 같이 생긴다."""
    key = _id_key(ids, seed)
    grp = steps.astype(np.int64) * 7 + (0 if strata is None else strata.astype(np.int64))
    order = np.lexsort((key, grp))
    g = grp[order]
    partner = np.empty(len(ids), dtype=np.int64)
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    ends = np.r_[starts[1:], len(g)]
    for a, b in zip(starts, ends):
        idx = order[a:b]
        partner[idx] = np.roll(idx, -1)
    return partner


def mi_gain(x, y, ids, steps, bins: int = 8, folds: int = 10, n_null: int = 5, seed: int = 0) -> dict:
    """I(x; y)가 영가설(같은 스텝의 고정 순서 옆 개체의 y)보다 큰가. n_null개의 순서로 영가설을 평균한다."""
    y = np.asarray(y)
    nulls = [y[neighbor_index(np.asarray(ids), np.asarray(steps), seed + 101 * i)] for i in range(n_null)]
    return mi_diff(x, y, nulls, np.asarray(ids), folds, bins, seed)


def holm(ps: list[float], alpha: float = 0.05) -> list[bool]:
    order = np.argsort(ps)
    ok = [False] * len(ps)
    m = len(ps)
    for rank, i in enumerate(order):
        if ps[i] <= alpha / (m - rank):
            ok[i] = True
        else:
            break
    return ok


def reseed(world, seed: int) -> None:
    """체크포인트를 이어 돌 때 세계 씨앗만 바꾼다(반복 사이 구간용). 배열 크기는 그대로다."""
    world.set_config(world.cfg.with_overrides([f"seed={int(seed)}"]))


def _within(v: np.ndarray, ids: np.ndarray) -> np.ndarray:
    _, inv = np.unique(ids, return_inverse=True)
    cnt = np.bincount(inv)
    if v.ndim == 1:
        return v - (np.bincount(inv, v) / cnt)[inv]
    return np.column_stack([_within(v[:, c], ids) for c in range(v.shape[1])])


def info_world(world, steps: int = 2000, stride: int = 20, bins: int = 8, folds: int = 10, n_null: int = 5,
               seed: int = 0, alpha: float = 0.05) -> dict:
    """겨루기 없이 세계를 돌리며 신호의 상호정보량을 잰다(설계 ②). 판정이 아니라 반응의 크기다:
    "쓰임"은 섞기 겨루기에서 지고 이 값이 기준선보다 클 때만 말한다."""
    s, cfg = world.s, world.cfg
    if not cfg.social_enabled:
        return {"note": "사회 감각이 꺼진 세계라 신호가 없다"}
    mind = cfg.mind_enabled
    hear = np.asarray(columns(cfg, "hear"))
    vis = np.asarray(columns(cfg, "vision"))
    ctx = context_columns(cfg)
    ctx_names = list(CTX_NAMES) if mind else list(CTX_NAMES[:3])
    ctx = ctx[:len(ctx_names)]
    acts = {"move": [ACT_THRUST, ACT_TURN]}
    n_out = s.act.shape[1]
    if cfg.conflict_enabled and ACT_ATTACK < n_out:
        acts["attack"] = [ACT_ATTACK]
    rec = {k_: [] for k_ in ("id", "step", "sig", "ctx", "heard", "vstr")}
    resp = {name: {v: [] for v in ("orig", "shuf", "strat", "base", "base_shuf")} for name in acts}
    fit_err = []
    t0 = int(world.step_no)
    for t in range(steps):
        if t % stride:
            world.run(1)
            continue
        h_prev = s.h.copy()
        Hin = s.Hin.copy() if mind else None
        Hrec = s.Hrec.copy() if mind else None
        ids_before, alive_before = s.ids.copy(), s.alive.copy()
        world.run(1)
        rows = np.flatnonzero(s.alive & alive_before & (s.ids == ids_before) & (s.asleep <= 0.0))
        if len(rows) < 3:
            continue
        obs = s.obs[rows].astype(np.float64)
        hp = h_prev[rows].astype(np.float64)
        ids = s.ids[rows].astype(np.int64)
        stp = np.full(len(rows), t, dtype=np.int64)
        vstr = np.abs(obs[:, vis]).sum(1)
        strata = np.searchsorted(np.quantile(vstr, [1 / 3, 2 / 3]), vstr, side="right")
        nb = neighbor_index(ids, stp, seed)
        nb_s = neighbor_index(ids, stp, seed, strata)
        o_sh = obs.copy(); o_sh[:, hear] = obs[nb][:, hear]
        o_st = obs.copy(); o_st[:, hear] = obs[nb_s][:, hear]
        Win = s.Win[rows].astype(np.float64)
        if mind:
            Win = Win + s.alpha[rows][:, None, :].astype(np.float64) * Hin[rows]
        Wb = Win.copy(); Wb[:, hear, :] = Win[nb][:, hear, :]     # 듣기 가중치를 개체끼리 섞은 뇌(기준선)
        full = _forward(s, rows, obs, hp, mind, Hin, Hrec)
        fit_err.append(float(np.abs(full[:, ACT_SIG0:ACT_SIG0 + 2] - s.act[rows, ACT_SIG0:ACT_SIG0 + 2]).max()))
        sh = _forward(s, rows, o_sh, hp, mind, Hin, Hrec)
        stt = _forward(s, rows, o_st, hp, mind, Hin, Hrec)
        bo = _forward(s, rows, obs, hp, mind, Hin, Hrec, Win=Wb)
        bs = _forward(s, rows, o_sh, hp, mind, Hin, Hrec, Win=Wb)
        for name, cols_ in acts.items():
            for v, arr in (("orig", full), ("shuf", sh), ("strat", stt), ("base", bo), ("base_shuf", bs)):
                resp[name][v].append(arr[:, cols_])
        rec["id"].append(ids); rec["step"].append(stp)
        rec["sig"].append(s.sig[rows].astype(np.float64)); rec["ctx"].append(obs[:, ctx])
        rec["heard"].append(obs[:, hear]); rec["vstr"].append(vstr)
    world.drain_events()
    R = {k_: np.concatenate(v) for k_, v in rec.items()}
    ids, stp = R["id"], R["step"]
    out = {"step": t0, "steps": steps, "stride": stride, "seed": seed, "bins": bins, "folds": folds,
           "samples": int(len(ids)), "individuals": int(len(np.unique(ids))),
           "forward_check_max_err": float(max(fit_err)) if fit_err else float("nan"),
           "heard_something_frac": float((np.abs(R["heard"]).max(1) > 0.05).mean())}
    # 내는 쪽
    sig = R["sig"]
    sigvars = {"sig0": sig[:, 0], "sig1": sig[:, 1], "sum": sig[:, 0] + sig[:, 1], "diff": sig[:, 0] - sig[:, 1]}
    pairs, within = {}, {}
    for sn, sv in sigvars.items():
        for ci, cn in enumerate(ctx_names):
            cv = R["ctx"][:, ci]
            pairs[f"{sn}|{cn}"] = mi_gain(sv, cv, ids, stp, bins, folds, n_null, seed)
            within[f"{sn}|{cn}"] = mi_gain(_within(sv, ids), _within(cv, ids), ids, stp, bins, folds, n_null, seed)
    names = list(pairs)
    hp_ = holm([pairs[n]["p"] for n in names], alpha)
    hw_ = holm([within[n]["p"] for n in names], alpha)
    for n, a_, b_ in zip(names, hp_, hw_):
        pairs[n]["holm"] = a_ and pairs[n]["above_zero"]
        within[n]["holm"] = b_ and within[n]["above_zero"]
    judged = [f"{a_}|{b_}" for a_ in ("sig0", "sig1") for b_ in ("energy", "ate")]
    passing = [n for n in judged if pairs[n]["holm"] and within[n]["holm"]]
    sp_all = float(np.var(sig, axis=0).sum())
    sp_in = float(np.var(_within(sig, ids), axis=0).sum())
    out["speaker"] = {"pairs": pairs, "within": within, "judged_pairs": judged, "passing": passing,
                      "verdict_pass": bool(passing),
                      "n_holm_pass": int(sum(pairs[n]["holm"] for n in names)),
                      "n_within_holm_pass": int(sum(within[n]["holm"] for n in names)),
                      "between_individual_share": float(1 - sp_in / sp_all) if sp_all > 1e-12 else float("nan"),
                      "note": "단서이지 쓰임의 증거가 아니다(설계 판단표 규칙 5)"}
    # 듣는 쪽: 들은 4칸 결합(칸마다 3분위) ↔ 행동 결합(칸마다 4분위)
    H = R["heard"]
    Hsd = np.column_stack([H[:, 0] + H[:, 1], H[:, 0] - H[:, 1], H[:, 2] + H[:, 3], H[:, 2] - H[:, 3]])
    lis = {}
    for name in acts:
        A = {v: np.concatenate(resp[name][v]) for v in resp[name]}
        ab = 4 if A["orig"].shape[1] > 1 else bins
        r_ = mi_diff(H, A["orig"], [A["shuf"]], ids, folds, ab, seed, xbins=3)
        r_st = mi_diff(H, A["orig"], [A["strat"]], ids, folds, ab, seed, xbins=3)
        r_b = mi_diff(H, A["base"], [A["base_shuf"]], ids, folds, ab, seed, xbins=3)
        r_sd = mi_diff(Hsd, A["orig"], [A["shuf"]], ids, folds, ab, seed, xbins=3)
        gap = _interval(np.asarray(r_["folds"]) - np.asarray(r_b["folds"]))
        lis[name] = {"response": r_, "response_strat": r_st, "baseline_shuffled_weights": r_b,
                     "response_sumdiff": r_sd, "response_minus_baseline": gap,
                     "mean_abs_change_if_shuffled": float(np.abs(A["orig"] - A["shuf"]).mean())}
    out["listener"] = lis
    out["listener_note"] = ("반응 = I(들은 것; 행동) − I(들은 것; 들은 칸만 옆 개체 것으로 바꿔 다시 계산한 행동). "
                            "떠돌이 가중치도 0보다 크게 낸다 — 기준선(듣기 가중치를 개체끼리 섞은 뇌)보다 큰지 본다.")
    return out


def info(path: Path, steps: int = 2000, stride: int = 20, seed: int | None = None, bins: int = 8,
         folds: int = 10, n_null: int = 5) -> dict:
    world, _ = ckpt.load(Path(path))
    if seed is not None:
        reseed(world, seed)
    out = info_world(world, steps, stride, bins, folds, n_null, seed=0 if seed is None else seed)
    out["ckpt"] = str(path)
    out["world_seed"] = int(world.cfg.seed)
    return out


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
    p.add_argument("--mode", choices=MODES, default="zero", help="zero: 0으로 지우기, shuffle: A끼리 바꿔 끼우기")
    p.add_argument("--frac", type=float, default=1.0, help="shuffle일 때 스텝마다 섞는 A의 몫(약한 양성 대조: 0.5)")
    p.add_argument("--twin", action="store_true",
                   help="쌍둥이 겨루기: 한 자리의 A·B가 같은 개체, 중립은 막지 않은 두 벌(빈 막기 대조)")
    i = sub.add_parser("info")
    i.add_argument("ckpt", type=Path)
    i.add_argument("--steps", type=int, default=2000)
    i.add_argument("--stride", type=int, default=20)
    i.add_argument("--seed", type=int, default=None, help="세계 씨앗을 바꿔 돈다(반복 사이 구간용)")
    i.add_argument("--bins", type=int, default=8)
    i.add_argument("--folds", type=int, default=10)
    i.add_argument("--n-null", type=int, default=5)
    d = sub.add_parser("describe")
    d.add_argument("ckpt", type=Path)
    d.add_argument("--steps", type=int, default=500)
    a = ap.parse_args(argv)
    if a.cmd == "split":
        out = split(a.ckpt, a.channel, a.n, a.steps, a.reps, a.seed, a.control, log=sys.stderr,
                    base_overrides=a.base_set, mode=a.mode, frac=a.frac, twin=a.twin)
    elif a.cmd == "info":
        out = info(a.ckpt, a.steps, a.stride, a.seed, a.bins, a.folds, a.n_null)
    else:
        out = describe(a.ckpt, a.steps)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
