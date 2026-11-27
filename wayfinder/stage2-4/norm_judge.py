"""규범의 씨앗(가로채기·벌주기) 판정 도구 — 명령줄. 순수 계산은 norm_gate.py, 손 집행자는 norm_hand.py.

설계 docs/superpowers/specs/2026-11-26-norms-seed-design.md "판정", 계획 docs/superpowers/plans/2026-11-26-norms-seed.md 과제 17.
체인(norm_chain.sh)이 부르는 하위 명령과 원자료 이름(모두 기록 폴더 아래):
    threshold <체크포인트> --json threshold.json          관문 −2b: 공격 행 다시 뽑은 시작의 문턱 위 개체 비율(시뮬레이션 없음)
    observe <체크포인트> --steps N --json w-*.json        세계 점검 창(−1)과 단서: 유병률·θ_t·take_loss_frac·p2·p3·구간표 등
    arena-check <체크포인트>                              관문 −1e: 충동 0 마당에서 거주민 공격 0
    calibrate --base <체크포인트> --from w-full-a.json --out calib.json   표지 원천 유병률 맞추기(sig 문턱 이분 탐색, none 잡음 확률)
    hand-dose <체크포인트> --cell 이름 --arena std|mix [--calib calib.json] --json h-<칸>-<마당>.json   판정 0 한 칸
    gate-world --dir <폴더> --tags a b                    관문 −1(종료 코드 0 통과, 3 세계 못 섬, 4 딜레마 크기 밖)
    gate-hand --dir <폴더>                                관문 α·β·γ·δ·−2(0 통과, 3 가를 수 없음, 4 도구 깨짐, 5 미판정)
    block-compete <끝 체크포인트> --json j-block-*.json   판정 나: 태그 C(1, 원래)·A(2, 전부 못 침)·B(3, 3자만 못 침)
    bottom <끝 체크포인트> --json j-bottom-*.json         판정 다·라: 같은 끝 무리의 표준·섞은 마당 계수기
    final --dir <폴더> --tags a b                         주 판정 문장(0 통과, 3 가를 수 없음, 4 깨짐, 5 미판정)

    PYTHONPATH=<코드> python wayfinder/stage2-4/norm_judge.py <명령> ...
겨루기 마당 설정은 체크포인트 설정 + `conflict_enabled=true` + `--set`(체인이 norm_impulse·norm_mark_tau를 넘긴다) + 칸 설정.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import norm_gate as ng  # noqa: E402
import norm_hand as nh  # noqa: E402
from kkumteul import checkpoint as ckpt  # noqa: E402
from kkumteul import kernels as k  # noqa: E402
from kkumteul import measure  # noqa: E402
from kkumteul import stages as st  # noqa: E402
from kkumteul.config import ACT_ATTACK, ACT_TURN, N_NORM_VIS, NORM_VIS  # noqa: E402

DEFAULT_SETS = ["norm_impulse=0.015"]
MIX_SETS = ["norm_mix_period=200", "norm_mix_frac=0.2"]
ON = ["norm_enabled=true"]


def _cell(arena, phi, sets=(), calib=None):
    return {"arena": arena, "phi": phi, "sets": list(sets), "calib": calib}


# 판정 0의 칸(설계 표). arena: base(전원 거주민) / gap(마당 갑: 태그 1·2 = φ씩, norm_block_tag=2) /
# eul(마당 을: 태그 1·3 = φ씩, norm_block3_tag=3) / all1·all2·all3(전원 그 태그). calib: sig·noise면 calib.json에서 값을 읽는다.
# off는 표시 장치를 끄고 계수기만 켠다(norm_measure — 위반 피해율을 같은 정의로 읽는다), noimp는 위반 공급 0.
CELLS = {
    "base": _cell("base", 0.0, ON),
    "hand-phi0.1": _cell("gap", 0.1, ON),
    "hand-phi0.3": _cell("gap", 0.3, ON),
    "hand-phi1.0": _cell("all1", 1.0, ON),
    "hand3-phi0.3": _cell("eul", 0.3, ON),
    "hand-phi0.3-inert": _cell("gap", 0.3, ON + ["norm_mode=inert"]),
    "hand-phi0.3-nearest": _cell("gap", 0.3, ON + ["norm_aim=nearest"]),
    "hand-phi0.3-ret0": _cell("gap", 0.3, ON + ["norm_pun_return=0"]),
    "hand-phi0.3-ret0.5": _cell("gap", 0.3, ON + ["norm_pun_return=0.5"]),
    "hand-phi0.3-inh0": _cell("gap", 0.3, ON + ["norm_inhibit=0"]),
    "hand-phi0.3-inh0.4": _cell("gap", 0.3, ON + ["norm_inhibit=0.4"]),
    "hand-phi0.3-L1": _cell("gap", 0.3, ON + ["norm_impulse_len=1"]),
    "hand-phi0.3-L10": _cell("gap", 0.3, ON + ["norm_impulse_len=10"]),
    "hand-phi0.3-tau1": _cell("gap", 0.3, ON + ["norm_mark_tau=1"]),
    "hand-phi0.3-sig-ret0": _cell("gap", 0.3, ON + ["norm_mark_src=sig", "norm_pun_return=0"], "sig"),
    "hand-phi0.3-sig-ret0.1": _cell("gap", 0.3, ON + ["norm_mark_src=sig", "norm_pun_return=0.1"], "sig"),
    "hand-phi0.3-none-q-ret0": _cell("gap", 0.3, ON + ["norm_mark_src=none", "norm_pun_return=0"], "noise"),
    "hand-phi0.3-none-q-ret0.1": _cell("gap", 0.3, ON + ["norm_mark_src=none", "norm_pun_return=0.1"], "noise"),
    "hand-phi0.3-noise0.0005": _cell("gap", 0.3, ON + ["norm_mark_noise=0.0005"]),
    "hand-phi0.3-noise0.002": _cell("gap", 0.3, ON + ["norm_mark_noise=0.002"]),
    "off": _cell("gap", 0.3, ["norm_enabled=false", "norm_measure=true"]),
    "noimp": _cell("gap", 0.3, ON + ["norm_impulse=0"]),
    "grp-1": _cell("all1", 1.0, ON),
    "grp-2": _cell("all2", 1.0, ON),
    "grp-3": _cell("all3", 1.0, ON),
}
CORE, DELTA, REST = ng.CORE, ng.DELTA, ng.REST     # 묶음은 norm_gate에 한 번만 적는다(gate-hand가 CORE·DELTA를 요구)


def _dump(obj, path: Path | None) -> None:
    text = json.dumps(obj, indent=1, ensure_ascii=False, default=float)
    if path:
        Path(path).write_text(text)
    else:
        print(text)


def _load_json(path: Path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


# ======================= 마당 =======================
def cell_sets(cell: str, calib: dict | None) -> list[str]:
    """칸 설정 + calib(sig 문턱·none 잡음 확률). calib가 필요한데 없으면 오류."""
    c = CELLS[cell]
    sets = list(c["sets"])
    if c["calib"]:
        if not calib:
            raise ValueError(f"{cell}은 --calib(calib.json)가 필요하다")
        sets.append(f"norm_sig_theta={calib['sig_theta']}" if c["calib"] == "sig"
                    else f"norm_mark_noise={calib['noise_q']}")
    return sets


def arena_groups(cell: str, g: dict, cfg, n: int, arena: str = "std", calib: dict | None = None):
    """칸 하나의 마당: (groups, 마당 설정, 뺀 개체 수). 막기 키는 장치가 켜져 있을 때만 건다(꺼진 칸은 검사가 막는다)."""
    c = CELLS[cell]
    sets = cell_sets(cell, calib) + (MIX_SETS if arena == "mix" else [])
    acfg = cfg.with_overrides(sets)
    res, dropped = nh.plant_enforcer(g, acfg, enforcer=False)
    enf, _ = nh.plant_enforcer(g, acfg, enforcer=True)
    kind, m = c["arena"], int(round(c["phi"] * n))
    if kind == "base":
        groups = [(res, 0, n)]
    elif kind == "gap":
        groups = [(res, 0, n - 2 * m), (enf, 1, m), (enf, 2, m)]
    elif kind == "eul":
        groups = [(res, 0, n - 2 * m), (enf, 1, m), (enf, 3, m)]
    else:
        groups = [(enf, int(kind[-1]), n)]
    groups = [x for x in groups if x[2] > 0]
    tags = {t for _, t, _ in groups}
    if acfg.norm_enabled:
        extra = []
        if 2 in tags:
            extra.append("norm_block_tag=2")
        if 3 in tags:
            extra.append("norm_block3_tag=3")
        acfg = acfg.with_overrides(extra) if extra else acfg
    return groups, acfg, dropped


def _energy_by_tag(world, tags) -> dict:
    s = world.s
    out = {}
    for t in tags:
        m = s.alive & (s.tag == t)
        out[t] = float(s.energy[m].mean()) if m.any() else None
    return out


def run_arena(acfg, groups, steps: int, seed: int) -> dict:
    """compete_multi 한 판 + 판정이 쓰는 값(위반 피해율·표시당 집행 확률·태그별 끝 평균 에너지·구간표)."""
    res, world = measure.compete_multi(acfg, groups, steps, seed, return_world=True)
    tags = sorted({int(t) for _, t, _ in groups})
    res["damage_rate"] = ng.damage_rate(res["norm_by_tag"])
    res["damage_rate_by_tag"] = {t: ng.damage_rate(res["norm_by_tag"][t]) for t in tags}
    res["punished_per_episode"] = ng.punished_per_episode(res["norm_by_tag"])
    res["energy_mean_by_tag"] = _energy_by_tag(world, tags)
    res["hist"] = world.s.norm_hist.tolist()
    res["ledger_rel"] = _ledger_rel(world)
    return res


def _ledger_rel(world) -> float:
    b = world.energy_balance()
    den = b["spawn"] + b["inject"]
    return abs(b["residual"]) / den if den > 0 else 0.0


def hand_dose(path: Path, cell: str, arena: str = "std", n: int = 800, steps: int = 8000, reps: int = 6,
              seed: int = 9000, sets=None, calib: dict | None = None, log=None) -> dict:
    """판정 0 한 칸: 같은 씨앗(seed + r)으로 반복, base 칸만 중립(거주민을 태그 0·1로 반씩, seed + 500 + i)."""
    sets = DEFAULT_SETS if sets is None else list(sets)
    side = measure.norm_arena_side(path, ["conflict_enabled=true", *sets])
    groups, acfg, dropped = arena_groups(cell, side["genomes"], side["cfg"], n, arena, calib)
    out = {"ckpt": str(path), "cell": cell, "arena": arena, "n": n, "steps": steps, "reps": reps, "seed": seed,
           "sets": sets, "cell_sets": cell_sets(cell, calib), "dropped": dropped,
           "groups": [[int(t), int(c)] for _, t, c in groups], "runs": []}
    for r in range(reps):
        res = run_arena(acfg, groups, steps, seed + r)
        out["runs"].append(res)
        if log:
            print(f"{cell}-{arena} 판 {r}: 위반 피해율 {res['damage_rate']}, 산 {res['alive_by_tag']}", file=log, flush=True)
    if cell == "base":
        res_g = groups[0][0]
        neu = []
        for i in range(reps):
            r_ = run_arena(acfg, [(res_g, 0, n // 2), (res_g, 1, n - n // 2)], steps, seed + 500 + i)
            neu.append(ng.net_income_diff(r_, 1, 0))
        out["neutral"] = neu
    s = ng.cell_summary(out)
    out["summary"] = {"damage_rate": ng.level_intervals(s["dr"]), "punished_per_episode": ng.level_intervals(s["epp"]),
                      "prevalence": ng.level_intervals(s["prev"]), "both_extinct": s["both_extinct"]}
    tags = {t for t, _ in out["groups"]}
    if {1, 2} <= tags:
        out["summary"]["n_ind"] = ng.n_ind(out["runs"], 1, 2)
    if {1, 3} <= tags:
        out["summary"]["n_ind3"] = ng.n_ind(out["runs"], 1, 3)
    out["labels"] = {"hist": ng.LABEL_HIST, "imposed": ng.LABEL_IMPOSED}
    return out


# ======================= 관찰 창 =======================
def _income_now(world) -> float:
    """먹은 양의 누계(죽은 개체 EATEN_DEAD + 산 개체 eaten). 창 끝 − 창 시작 = 창 안 먹이 수입."""
    s = world.s
    return float(s.norm_tab[:, k.NT_EATEN_DEAD].sum()) + float(s.eaten[s.alive].sum())


def vis_check(world) -> tuple[int, int, int]:
    """−1d 표본: (깬 개체 수, NORM_VIS가 0이 아닌 개체 수, 시야 안에 표시된 이웃이 있는 개체 수). 지금 위치로 격자를 다시
    짓고 norm_sense를 한 번 불러(감각 칸만 쓴다) 독립 계산(거리 ≤ vision_range, 가까움 > 0, 시야각 안)과 견준다."""
    s, c = world.s, world.cfg
    t = world.step_no
    k.build_grid(s.x, s.y, s.alive, 1.0 / c.cell, c.grid_w, c.grid_h, s.a_start, s.a_items, s.cursor)
    st.norm_sense(t, s, world.fs, world.isp)
    awake = np.flatnonzero(s.alive & (s.asleep <= 0.0))
    tau = c.norm_mark_tau
    marked = np.flatnonzero(s.alive & (s.mark_t >= 0) & (t - s.mark_t <= tau) & (t - s.mark_t >= 0))
    c0 = c.norm_in0 + NORM_VIS
    nz = int((np.abs(s.obs[awake, c0:c0 + N_NORM_VIS]) > 0).any(axis=1).sum())
    present = 0
    if len(marked):
        bright = 1.0 - (t - s.mark_t[marked]) / (tau + 1.0)
        mk = marked[bright > 0]
        for i in awake:
            dx = s.x[mk] - s.x[i]
            dy = s.y[mk] - s.y[i]
            dx -= c.world_w * np.round(dx / c.world_w)
            dy -= c.world_h * np.round(dy / c.world_h)
            d = np.hypot(dx, dy)
            ok = (mk != i) & (d <= c.vision_range) & (1.0 - np.maximum(d - c.agent_radius, 0) / c.vision_range > 0)
            if not ok.any():
                continue
            ang = np.arctan2(dy[ok], dx[ok]) - s.heading[i]
            ang = (ang + math.pi) % (2 * math.pi) - math.pi
            rad = c.agent_radius
            delta = np.where(d[ok] <= rad, math.pi, np.arcsin(np.minimum(1.0, rad / np.maximum(d[ok], 1e-9))))
            if ((ang + delta >= -0.5 * c.vision_fov) & (ang - delta <= 0.5 * c.vision_fov)).any():
                present += 1
    return len(awake), nz, present


def observe_world(world, steps: int, sample_every: int = 250, late_tau: int | None = None) -> dict:
    """세계를 steps 더 돌리며 창을 읽는다. 창 시작에 reset_norm_tables. 장치가 켜졌을 때만 표시·벌 값을 낸다."""
    cfg = world.cfg
    world.reset_norm_tables()
    s = world.s
    inc0 = _income_now(world)
    rs0 = int(s.counters[k.C_RESEED])
    pops, nzs, prs, awk = [], 0, 0, 0
    delays = []
    on = bool(world.isp[st.I_NORM_ON])
    done = 0
    while done < steps:
        if late_tau is not None and on:
            t0 = world.step_no
            mt = s.mark_t.copy()
            world.run(1)
            hit = np.flatnonzero(s.alive & (s.pun_t == t0) & (mt >= 0))
            delays += [int(t0 - mt[i]) for i in hit]
            done += 1
        else:
            n = min(sample_every, steps - done)
            world.run(n)
            done += n
        if done % sample_every == 0 or done == steps:
            pops.append(int(s.alive.sum()))
            if on and cfg.vision_sectors == N_NORM_VIS:
                a, nz, pr = vis_check(world)
                awk, nzs, prs = awk + a, nzs + nz, prs + pr
    tab = s.norm_tab
    tot = tab.sum(axis=0)
    alive_steps = float(tot[k.NT_ALIVE_STEPS])
    income = _income_now(world) - inc0
    out = {"step": world.step_no, "steps": steps, "pop_mean": float(np.mean(pops)) if pops else 0.0,
           "pop_min": int(min(pops)) if pops else 0, "reseeds": int(s.counters[k.C_RESEED]) - rs0,
           "ledger_rel": _ledger_rel(world), "norm_on": on,
           "norm_measure": bool(cfg.norm_measure), "income": income}
    if cfg.norm_measure or cfg.norm_enabled:
        nt = {name: float(tot[c]) for c, name in enumerate(k.NT_NAMES)}
        out["norm_tab"] = nt
        out["theta_t"] = float(tot[k.NT_EXPO]) / alive_steps if alive_steps else None
        out["take_loss_frac"] = float(tot[k.NT_TAKE_DMG_RECV]) / income if income > 0 else None
        out["damage_rate"] = ng.damage_rate(nt)
    if on:
        nt = out["norm_tab"]
        out["prevalence"] = float(tot[k.NT_MARKED_STEPS]) / alive_steps if alive_steps else None
        out["punished_per_episode"] = ng.punished_per_episode(nt)
        out.update(ng.bottom_rates(nt))
        out["hist"] = s.norm_hist.tolist()
        out["hist_label"] = ng.LABEL_HIST
        out["saved_est"] = ng.saved_est(s.norm_hist, nt["take_dmg_recv"], nt["take"])
        out["vis_awake_samples"] = awk
        out["vis_nonzero_frac"] = nzs / awk if awk else None
        out["vis_present_frac"] = prs / awk if awk else None
        out.update(_sim_pairs(world))
        if late_tau is not None:
            early = sum(1 for d in delays if 0 <= d < 10)
            late = sum(1 for d in delays if 10 <= d <= late_tau)
            out["late_frac"] = late / early if early else None
            out["pun_delays_n"] = len(delays)
    return out


def _sim_pairs(world, max_pairs: int = 4000, seed: int = 11) -> dict:
    """이웃 쌍(soc_radius 안) 대 아무 쌍의 색 닮음 분포(10칸 히스토그램). 친척 경계를 사후에 정하는 자료(과제 8)."""
    s, c = world.s, world.cfg
    a = np.flatnonzero(s.alive)
    if len(a) < 2:
        return {"sim_neighbor_hist": None, "sim_random_hist": None}
    rng = np.random.default_rng(seed)
    col = s.color[a].astype(float)
    x, y = s.x[a], s.y[a]
    ii = rng.integers(0, len(a), size=max_pairs)
    jj = rng.integers(0, len(a), size=max_pairs)
    keep = ii != jj
    ii, jj = ii[keep], jj[keep]
    sim = 1.0 - np.linalg.norm(col[ii] - col[jj], axis=1) / math.sqrt(3.0)
    nb_sims = []
    for p in range(len(a)):
        dx = x - x[p]
        dy = y - y[p]
        dx -= c.world_w * np.round(dx / c.world_w)
        dy -= c.world_h * np.round(dy / c.world_h)
        q = np.flatnonzero((np.hypot(dx, dy) <= c.soc_radius) & (np.arange(len(a)) > p))
        if len(q):
            nb_sims.append(1.0 - np.linalg.norm(col[q] - col[p], axis=1) / math.sqrt(3.0))
        if sum(len(v) for v in nb_sims) >= max_pairs:
            break
    nb = np.concatenate(nb_sims) if nb_sims else np.zeros(0)
    edges = np.linspace(0, 1, 11)
    return {"sim_neighbor_hist": np.histogram(nb, edges)[0].tolist(), "sim_random_hist": np.histogram(sim, edges)[0].tolist(),
            "sim_edges": edges.tolist()}


def observe(path: Path, steps: int = 5000, sets=(), late: bool = False) -> dict:
    cfg = ckpt.read_config(Path(path)).with_overrides(list(sets)) if sets else None
    world, _ = ckpt.load(Path(path), cfg)
    out = observe_world(world, steps, late_tau=world.cfg.norm_mark_tau if late else None)
    out["ckpt"] = str(path)
    return out


# ======================= 관문 −2b·−1e·유병률 맞추기 =======================
def threshold(path: Path, b0: float = -1.0, sets=()) -> dict:
    """관문 −2b: 바탕을 norm 크기로 읽어 갈래와 같은 공격 행 다시 뽑기(편향 b0)를 건 뒤, 저장된 h로 pre-act와 문턱 위
    비율. b0 = −0.3(기울기 시작판)의 비율도 같이 낸다. NORM_VIS → 공격·회전 실효 가중치 분포."""
    cfg = ckpt.read_config(Path(path)).with_overrides(["preset=norm_size", *sets])
    out = {"ckpt": str(path)}
    for b in (b0, -0.3):
        world, _ = ckpt.load(Path(path), cfg)
        world.reseed_attack_row(b)
        s = world.s
        a = np.flatnonzero(s.alive)
        pre = s.bout[a, ACT_ATTACK] + np.einsum("nh,nh->n", s.Wout[a, ACT_ATTACK, :], s.h[a, :])
        c0 = cfg.norm_in0 + NORM_VIS
        win = s.Win[a, c0:c0 + N_NORM_VIS, :].astype(float)
        w_atk = np.einsum("nch,nh->nc", win, s.Wout[a, ACT_ATTACK, :].astype(float))
        w_turn = np.einsum("nch,nh->nc", win, s.Wout[a, ACT_TURN, :].astype(float))
        res = ng.threshold_above(pre, w_atk, w_turn)
        if b == b0:
            out.update(res)
            out["b0"] = b0
        else:
            out["b0_slope"] = {"b0": b, "frac_above": res["frac_above"]}
        if b0 == -0.3:
            break
    return out


def arena_check(path: Path, n: int = 400, steps: int = 2000, seed: int = 8100, sets=None) -> dict:
    """관문 −1e: 위반 공급 0(norm_impulse=0)의 마당(거주민만, 공격 행 고정)에서 가로채기·벌이 0."""
    sets = DEFAULT_SETS if sets is None else list(sets)
    side = measure.norm_arena_side(path, ["conflict_enabled=true", *sets, "norm_enabled=true", "norm_impulse=0"])
    res_g, _ = nh.plant_enforcer(side["genomes"], side["cfg"], enforcer=False)
    n = min(n, side["cfg"].max_agents)
    r = measure.compete_multi(side["cfg"], [(res_g, 0, n)], steps, seed)
    tot = ng.sum_rows(r["norm_by_tag"])
    return {"ckpt": str(path), "n": n, "steps": steps, "take": tot["take"], "pun": tot["pun"],
            "ok": bool(tot["take"] == 0 and tot["pun"] == 0)}


def _short_prevalence(path: Path, sets, steps: int) -> float:
    cfg = ckpt.read_config(Path(path)).with_overrides(["preset=norm_size", *sets])
    world, _ = ckpt.load(Path(path), cfg)
    world.reseed_attack_row(-1.0)
    o = observe_world(world, steps)
    return float(o.get("prevalence") or 0.0)


def calibrate(base: Path, full_json: dict, steps: int = 3000, sets=None, log=None) -> dict:
    """표지 원천 유병률 맞추기(설계 7절): take(full) 세계의 측정 유병률 P를 목표로 sig 문턱을 이분 탐색(짧은 창 최대 6번),
    none의 잡음 확률은 q0 = P ÷ (τ + 1)에서 한 번 확인 측정해 비례로 고친다. 근사: 3만 스텝 갈래 끝이 아니라 바탕에서 바로
    짧은 창을 읽는다(유병률이 오르는 중일 수 있다 — 세계 점검 갈래가 맞춘 값으로 다시 잰다)."""
    sets = DEFAULT_SETS if sets is None else list(sets)
    target = float(full_json["prevalence"])
    common = ["conflict_enabled=true", *sets, "norm_enabled=true", "norm_pun_return=0"]
    tau = ckpt.read_config(Path(base)).with_overrides(["preset=norm_size", *common]).norm_mark_tau

    def fn(theta):
        p = _short_prevalence(base, common + ["norm_mark_src=sig", f"norm_sig_theta={theta}"], steps)
        if log:
            print(f"sig θ {theta:.4f}: 유병률 {p:.4f}(목표 {target:.4f})", file=log, flush=True)
        return p
    sig = ng.calibrate_sig_theta(fn, target, -1.0, 1.0)
    q0 = ng.noise_q0(target, tau)
    p0 = _short_prevalence(base, common + ["norm_mark_src=none", f"norm_mark_noise={q0}"], steps)
    q1 = min(1.0, ng.noise_q_fix(q0, target, p0))
    return {"sig_theta": sig["sig_theta"], "noise_q": q1, "noise_q0": q0, "target": target,
            "measured": {"take": target, "sig": sig["measured"], "none": p0}, "sig_tries": sig["tries"],
            "sig_ok": sig["ok"], "steps": steps}


# ======================= 판정 나·다·라 =======================
BLOCK_SETS = ["norm_enabled=true", "norm_mode=full", "norm_mark_src=take", "norm_mark_noise=0", "norm_mix_period=0",
              "norm_mem_obs_mode=outcome", "norm_block_tag=-1", "norm_block3_tag=-1"]


def _r_series(series, a, b) -> float:
    """measure.compete와 같은 꼴의 r: ln((n_b + 0.5) ÷ (n_a + 0.5))의 기울기(뒤 2/3, 전멸 전까지) × 1000."""
    arr = np.array([[t, ng.tagget(c, a) or 0, ng.tagget(c, b) or 0] for t, c in series], dtype=float)
    dead = np.flatnonzero((arr[:, 1] == 0) | (arr[:, 2] == 0))
    upto = arr[:dead[0] + 1] if len(dead) else arr
    keep = upto[upto[:, 0] > upto[0, 0] + (arr[-1, 0] - arr[0, 0]) / 3.0]
    if len(keep) < 3:
        keep = upto
    y = np.log((keep[:, 2] + 0.5) / (keep[:, 1] + 0.5))
    if len(keep) >= 3:
        return float(np.polyfit(keep[:, 0], y, 1)[0]) * 1000
    return float((y[-1] - y[0]) / max(keep[-1, 0] - keep[0, 0], 1)) * 1000


def block_compete(path: Path, n: int = 800, steps: int = 8000, reps: int = 6, seed: int = 9500, log=None) -> dict:
    """판정 나: 끝 무리를 태그 C(1, 원래)·A(2, norm_block_tag)·B(3, norm_block3_tag)로. 마당 A형(C 대 A)·B형(C 대 B)을
    표준·섞은 마당에서, 순수 마당(전원 C·A·B)으로 N_grp·N_grp3, 중립(C 대 C', 막기 없음)으로 min_detectable.
    장치를 끈 끝 무리(off)도 같은 마당에서 견주도록 norm_enabled=true와 막기 키를 덮어쓴다(inert 끝 무리도 full 마당)."""
    side = measure.norm_arena_side(path, ["conflict_enabled=true", *BLOCK_SETS])
    g, cfg = side["genomes"], side["cfg"]
    half = n // 2
    plan = {
        "A_std": ([(g, 1, half), (g, 2, n - half)], ["norm_block_tag=2"]),
        "B_std": ([(g, 1, half), (g, 3, n - half)], ["norm_block3_tag=3"]),
        "A_mix": ([(g, 1, half), (g, 2, n - half)], ["norm_block_tag=2", *MIX_SETS]),
        "B_mix": ([(g, 1, half), (g, 3, n - half)], ["norm_block3_tag=3", *MIX_SETS]),
        "all_C": ([(g, 1, n)], []),
        "all_A": ([(g, 2, n)], ["norm_block_tag=2"]),
        "all_B": ([(g, 3, n)], ["norm_block3_tag=3"]),
        "neutral": ([(g, 1, half), (g, 2, n - half)], []),
    }
    out = {"ckpt": str(path), "n": n, "steps": steps, "reps": reps, "seed": seed, "arenas": {}}
    for name, (groups, extra) in plan.items():
        acfg = cfg.with_overrides(extra) if extra else cfg
        runs = []
        for r in range(reps):
            sd = seed + (500 if name == "neutral" else 0) + r
            runs.append(run_arena(acfg, groups, steps, sd))
            if log:
                print(f"{name} 판 {r}: 산 {runs[-1]['alive_by_tag']}", file=log, flush=True)
        out["arenas"][name] = runs
    out["summary"] = block_summary(out["arenas"])
    return out


def block_summary(ar: dict) -> dict:
    """block-compete 마당들 → 주 판정이 읽는 요약."""
    def sel(runs):
        return ng.level_intervals([ng.selectivity(ng.tagget(r["norm_by_tag"], 1)) for r in runs])

    def rs(runs, b):
        return [_r_series(r["series"], b, 1) for r in runs]     # C(1)가 b보다 잘 크면 양
    neu = rs(ar["neutral"], 2) if "neutral" in ar else []
    out = {
        "n_ind": ng.n_ind(ar["A_std"], 1, 2), "n_ind3": ng.n_ind(ar["B_std"], 1, 3),
        "n_ind_mix": ng.n_ind(ar["A_mix"], 1, 2) if "A_mix" in ar else None,
        "n_ind3_mix": ng.n_ind(ar["B_mix"], 1, 3) if "B_mix" in ar else None,
        "n_grp": ng.n_grp(ar["all_C"], ar["all_A"], 1, 2) if "all_A" in ar else None,
        "n_grp3": ng.n_grp(ar["all_C"], ar["all_B"], 1, 3) if "all_B" in ar else None,
        "selectivity": sel(ar["A_std"] + ar["B_std"]),
        "r_CA": ng.level_intervals(rs(ar["A_std"], 2)), "r_CB": ng.level_intervals(rs(ar["B_std"], 3)),
        "min_detectable": ng.detection_limit(neu), "neutral_r": ng.level_intervals(neu),
        "dr_diff_CA": ng.level_intervals([(r["damage_rate_by_tag"].get(1) or r["damage_rate_by_tag"].get("1") or 0)
                                          - (ng.tagget(r["damage_rate_by_tag"], 2) or 0) for r in ar["A_std"]]),
    }
    return out


def bottom_cmd(path: Path, n: int = 800, steps: int = 8000, reps: int = 6, seed: int = 9700, log=None) -> dict:
    """판정 다·라: 같은 끝 무리(전원 태그 1, 체크포인트의 표지 원천 그대로)를 표준·섞은 마당에서 같은 씨앗 짝으로.
    반복마다 벌의 늦음(경과 = 벌 받은 스텝 − 표시가 붙은 스텝)을 세어 late_frac = 경과 10~τ ÷ 0~10."""
    side = measure.norm_arena_side(path, ["conflict_enabled=true", "norm_enabled=true", "norm_mix_period=0",
                                          "norm_block_tag=-1", "norm_block3_tag=-1"])
    g, cfg = side["genomes"], side["cfg"]
    out = {"ckpt": str(path), "n": n, "steps": steps, "reps": reps, "seed": seed, "std": [], "mix": []}
    for arena in ("std", "mix"):
        acfg = cfg.with_overrides(MIX_SETS) if arena == "mix" else cfg
        for r in range(reps):
            res, world = measure.compete_multi(acfg, [(g, 1, n)], 0, seed + r, return_world=True)
            o = observe_world_arena(world, steps, acfg.norm_mark_tau)
            res = {"norm_by_tag": world.norm_table(), "alive_by_tag": {1: int(world.s.alive.sum())},
                   "mark_prevalence": o.get("prevalence"), "late_frac": o.get("late_frac"),
                   "pun_delays_n": o.get("pun_delays_n")}
            out[arena].append(res)
            if log:
                print(f"bottom {arena} 판 {r}: 산 {res['alive_by_tag']}, 늦은 집행 비 {res['late_frac']}", file=log, flush=True)
    out["summary"] = ng.bottom(out["std"], out["mix"])
    out["src"] = cfg.norm_mark_src
    return out


def observe_world_arena(world, steps: int, tau: int) -> dict:
    """겨루기 마당(compete_multi가 계수표를 비운 세계)을 한 스텝씩 돌리며 벌의 늦음을 센다(계수표는 지우지 않는다)."""
    s = world.s
    delays = []
    for _ in range(steps):
        t0 = world.step_no
        mt = s.mark_t.copy()
        world.run(1)
        hit = np.flatnonzero(s.alive & (s.pun_t == t0) & (mt >= 0))
        delays += [int(t0 - mt[i]) for i in hit]
    tot = s.norm_tab.sum(axis=0)
    al = float(tot[k.NT_ALIVE_STEPS])
    early = sum(1 for d in delays if 0 <= d < 10)
    late = sum(1 for d in delays if 10 <= d <= tau)
    return {"prevalence": float(tot[k.NT_MARKED_STEPS]) / al if al else None,
            "late_frac": late / early if early else None, "pun_delays_n": len(delays)}


# ======================= 파일을 읽는 관문 =======================
W_NAMES = ("meas", "inert", "full", "noimp", "sig", "none")


def gate_world_dir(directory: Path, tags) -> dict:
    obs = {name: [o for o in (_load_json(directory / f"w-{name}-{t}.json") for t in tags) if o] for name in W_NAMES}
    out = ng.gate_world(obs, _load_json(directory / "arena-check.json"), need=len(tags))
    out["missing_files"] = [f"w-{n}-{t}" for n in W_NAMES for t in tags if not (directory / f"w-{n}-{t}.json").exists()]
    return out


def read_cells(directory: Path, arena: str = "std") -> dict:
    cells = {}
    for name in CELLS:
        res = _load_json(directory / f"h-{name}-{arena}.json")
        if res:
            cells[name] = ng.cell_summary(res)
    return cells


def gate_hand_dir(directory: Path) -> dict:
    """관문 α·β·γ·δ·−2. δ의 50% 규칙은 같은 폴더의 gate-world.json(관문 −1 결과)에서 읽는다(없으면 δ 미판정)."""
    cells, mix = read_cells(directory, "std"), read_cells(directory, "mix")
    world = _load_json(directory / "gate-world.json")
    return ng.gate_hand(cells, mix, world=world)


def final_dir(directory: Path, tags) -> dict:
    """주 판정(씨앗마다) + 판정 다·라. 원자료가 없는 항목은 "미판정"으로 적고 통과로 안 센다."""
    out = {"by_seed": {}, "missing": []}

    def summ(name):
        d = _load_json(directory / f"{name}.json")
        if d is None:
            out["missing"].append(name)
            return None
        return d.get("summary")

    sums = {t: {arm: summ(f"j-block-{arm}-{t}") for arm in ("on", "inert", "off", "noimp")} for t in tags}
    codes = []
    for i, t in enumerate(tags):
        s = sums[t]
        knock = None
        kn_on, kn_in = _load_json(directory / f"j-knock-norm-on-{t}.json"), _load_json(directory / f"j-knock-norm-inert-{t}.json")
        if kn_on and kn_in:
            knock = {"on": kn_on["ckpt_rules"]["r_per_1000"]["mean"], "inert": kn_in["ckpt_rules"]["r_per_1000"]["mean"],
                     "limit": max(kn_on["ckpt_rules"].get("min_detectable_r_per_1000") or 0,
                                  kn_in["ckpt_rules"].get("min_detectable_r_per_1000") or 0)}
        other = sums[tags[1 - i]]["on"] if len(tags) == 2 else None
        mj = ng.main_judgment(s["on"], {"inert": s["inert"], "off": s["off"]}, s["noimp"], knock, other)
        bot_on, bot_arb = _load_json(directory / f"j-bottom-on-{t}.json"), _load_json(directory / f"j-bottom-arb-{t}.json")
        bot_mix = _load_json(directory / f"j-bottom-mix-{t}.json")
        out["by_seed"][t] = {
            "main": mj,
            "bottom": bot_on.get("summary") if bot_on else "미판정",
            "bottom_mix_branch": (bot_mix.get("summary") if bot_mix else "미판정"),
            "arbitrary": ng.arbitrary(bot_on["std"], bot_arb["std"]) if bot_on and bot_arb else "미판정",
        }
        codes.append(mj["code"])
    out["labels"] = {"arb": ng.LABEL_ARB, "clue": ng.LABEL_CLUE, "imposed": ng.LABEL_IMPOSED}
    out["code"] = 0 if 0 in codes else (5 if 5 in codes else (4 if 4 in codes else 3))
    return out


# ======================= 명령줄 =======================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="norm_judge.py", description="규범의 씨앗 판정 도구")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, sets=True):
        p.add_argument("--json", type=Path)
        if sets:
            p.add_argument("--set", action="append", default=None, metavar="key=value")
    p = sub.add_parser("threshold")
    p.add_argument("ckpt", type=Path)
    p.add_argument("--b0", type=float, default=-1.0)
    common(p)
    p = sub.add_parser("observe")
    p.add_argument("ckpt", type=Path)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--late", action="store_true", help="벌의 늦음(평판)도 센다(한 스텝씩이라 느리다)")
    common(p)
    p = sub.add_parser("arena-check")
    p.add_argument("ckpt", type=Path)
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--steps", type=int, default=2000)
    common(p)
    p = sub.add_parser("calibrate")
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--from", dest="src", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--set", action="append", default=None, metavar="key=value")
    p = sub.add_parser("hand-dose")
    p.add_argument("ckpt", type=Path)
    p.add_argument("--cell", choices=sorted(CELLS), required=True)
    p.add_argument("--arena", choices=("std", "mix"), default="std")
    p.add_argument("--n", type=int, default=800)
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--reps", type=int, default=6)
    p.add_argument("--seed", type=int, default=9000)
    p.add_argument("--calib", type=Path)
    common(p)
    for name in ("gate-world", "final"):
        p = sub.add_parser(name)
        p.add_argument("--dir", type=Path, required=True)
        p.add_argument("--tags", nargs="+", default=["a", "b"])
    p = sub.add_parser("gate-hand")
    p.add_argument("--dir", type=Path, required=True)
    for name in ("block-compete", "bottom"):
        p = sub.add_parser(name)
        p.add_argument("ckpt", type=Path)
        p.add_argument("--n", type=int, default=800)
        p.add_argument("--steps", type=int, default=8000)
        p.add_argument("--reps", type=int, default=6)
        common(p, sets=False)
    a = ap.parse_args(argv)
    code = 0
    if a.cmd == "threshold":
        out = threshold(a.ckpt, a.b0, a.set or ())
        print(f"관문 −2b: 문턱 위 {out['frac_above']:.4f}, 기울기 시작판 {out['slope_start']}", file=sys.stderr)
    elif a.cmd == "observe":
        out = observe(a.ckpt, a.steps, a.set or (), a.late)
        print(f"관찰: 무리 {out['pop_mean']:.0f}, θ_t {out.get('theta_t')}, 손실 {out.get('take_loss_frac')}, "
              f"유병률 {out.get('prevalence')}", file=sys.stderr)
    elif a.cmd == "arena-check":
        out = arena_check(a.ckpt, a.n, a.steps, sets=a.set)
        code = 0 if out["ok"] else 3
    elif a.cmd == "calibrate":
        src = _load_json(a.src)
        if not src or src.get("prevalence") is None:
            print(f"{a.src}: take(full) 세계의 유병률이 없다", file=sys.stderr)
            return 3
        out = calibrate(a.base, src, a.steps, a.set, log=sys.stderr)
        _dump(out, a.out)
        return 0
    elif a.cmd == "hand-dose":
        calib = _load_json(a.calib) if a.calib else None
        out = hand_dose(a.ckpt, a.cell, a.arena, a.n, a.steps, a.reps, a.seed, a.set, calib, log=sys.stderr)
    elif a.cmd == "gate-world":
        out = gate_world_dir(a.dir, a.tags)
        code = out["code"]
    elif a.cmd == "gate-hand":
        out = gate_hand_dir(a.dir)
        code = out["code"]
    elif a.cmd == "block-compete":
        out = block_compete(a.ckpt, a.n, a.steps, a.reps, log=sys.stderr)
    elif a.cmd == "bottom":
        out = bottom_cmd(a.ckpt, a.n, a.steps, a.reps, log=sys.stderr)
    else:
        out = final_dir(a.dir, a.tags)
        code = out["code"]
    _dump(out, getattr(a, "json", None))
    return code


if __name__ == "__main__":
    sys.exit(main())
