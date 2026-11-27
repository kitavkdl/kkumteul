"""성장 측정: 두 체크포인트의 개체를 새 세계에 같은 수로 풀고 어느 쪽 후손이 더 늘었는지 본다.

대결이 없는 세계라서 "이긴다"는 후손이 더 빨리 느는 것이다(Lenski식 selection rate).
    r = d/dt ln(N_B / N_A)      (양수면 B가 이김)
- 겨루는 동안에는 변이를 끈다(겨루는 중에 진화하면 무엇을 쟀는지 모르게 된다).
- 수용력 근처의 수로 시작하고 나이를 흩어 놓는다. 빈 세계에서 불어나는 능력이 아니라
  본 실행과 비슷한 경쟁 상태에서의 차이를 재려는 것이다.
- 두 무리는 같은 자리 쌍에서 시작한다(먹이 자리 근처에 떨어지는 운을 양쪽이 같이 받게).
- 처음 1/3은 버리고 ln(N_B/N_A)의 기울기로 r을 낸다.
- 같은 무리를 둘로 나눠 겨루게 한 값(A 대 A, B 대 B)을 같이 낸다. 이 값의 평균이 0에서 멀면
  측정 자체가 한쪽으로 기운 것이다.

    python -m kkumteul.measure assay  --a runs/main/checkpoints/ckpt_X.npz --b .../ckpt_Y.npz
    python -m kkumteul.measure assay  --a mine.pack.npz --b yours.pack.npz     (생물 묶음끼리, `pack.py`)
    python -m kkumteul.measure ladder --run runs/main --count 5

한계(독립 검토 2026-11-04): 같은 무리끼리도 r의 표준편차가 1,000스텝당 0.08쯤 된다.
6회 반복으로는 6,000스텝 뒤 개체 수 비가 약 1.4배 이상 벌어지는 차이만 가를 수 있다.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from . import checkpoint as ckpt
from . import kernels as k
from .config import ACT_ATTACK, FIRST_MEALS, Config
from .world import World

GENOME = ("Win", "Wrec", "bh", "Wout", "bout", "n_hidden", "sigma", "color", "rw", "eta", "alpha", "taste_g", "soc_g",
          "show_g", "feed_g")
# 나중에 생긴 유전체 배열: 이것이 없는 옛 묶음은 0으로 채워 읽는다(pack.read)
OPTIONAL_GENOME = ("taste_g", "soc_g", "show_g", "feed_g")
_T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23,
        12: 2.18, 15: 2.13, 20: 2.09, 30: 2.04}


def t95(df: int) -> float:
    """양측 95% t 임계값."""
    if df < 1:
        return float("inf")
    best = max((d for d in _T95 if d <= df), default=1)
    return _T95[best] if df <= 30 else 1.96


def extract_genomes(world: World) -> dict:
    a = np.flatnonzero(world.s.alive)
    g = {name: getattr(world.s, name)[a].copy() for name in GENOME}
    g["gen"] = world.s.gen[a].copy()
    return g


def load_side(path: Path) -> dict:
    """체크포인트나 생물 묶음(pack) 하나에서 겨루기에 필요한 것만 읽어 둔다(나중에 파일이 지워져도 되게)."""
    from . import pack   # pack이 이 모듈을 쓰므로 여기서 불러온다
    if pack.is_pack(path):
        return pack.load_side(path)
    world, _ = ckpt.load(Path(path))
    return side_from_world(world, str(path))


def side_from_world(world: World, path: str = "") -> dict:
    """살아 있는 세계에서 겨루기·이주에 쓰는 한쪽을 만든다(체크포인트나 묶음에서 읽은 것과 같은 꼴)."""
    g = extract_genomes(world)
    if not len(g["n_hidden"]):
        raise ValueError(f"살아 있는 개체가 없다: {path}")
    return {"path": path, "step": world.step_no, "genomes": g, "cfg": kind_aligned(world.cfg, world.step_no),
            "gen_mean": float(g["gen"].mean())}


def kind_aligned(cfg: Config, step: int) -> Config:
    """겨루기 마당이 원래 세계의 독 순서를 잇게 한 설정. 마당은 씨앗을 새로 받고 스텝 0부터 도므로, 그대로 두면
    어느 종류가 독인지를 마당 씨앗으로 다시 뽑는다 — 독이 오래 안 바뀌는 세계에서 자란 무리는 독이 뒤바뀐 마당에
    들어가 양쪽 다 전멸했다(2026-11-26, 고정 갈래). 먹이 종류가 꺼져 있으면 그대로 돌려준다."""
    if not cfg.food_kinds_enabled:
        return cfg
    d = cfg.to_dict()
    d.update(kind_seed=k.kind_seed_of(cfg), kind_step0=cfg.kind_step0 + int(step))
    return Config.from_dict(d)


def check_same_shape(a: dict, b: dict) -> None:
    """두 쪽의 유전체 배열 모양(개체 수 빼고)이 같은지 본다. 다르면 같은 세계에 넣을 수 없다."""
    for name in GENOME:
        sa, sb = a["genomes"][name].shape[1:], b["genomes"][name].shape[1:]
        if sa != sb:
            raise ValueError(f"두 쪽의 유전체 모양이 달라 겨룰 수 없다: {name}이 A는 {sa}, B는 {sb}다. "
                             "감각 칸 수와 뇌 최대 크기 설정(vision_sectors, n_spare_in, n_spare_out, h_max)이 "
                             "같은 세계에서 나온 생물끼리만 겨룰 수 있다")
    cfg = b["cfg"]
    want = (cfg.n_in, cfg.h_max)
    if b["genomes"]["Win"].shape[1:] != want:
        raise ValueError(f"겨룰 세계의 설정과 유전체 모양이 안 맞는다: Win이 {b['genomes']['Win'].shape[1:]}, "
                         f"설정은 {want}다")


def arena_config(cfg: Config, seed: int, n_total: int) -> Config:
    d = cfg.to_dict()
    d.update(seed=seed, mutation_enabled=False, min_agents=0, passive=False, init_agents=0)
    if n_total > d["max_agents"]:
        raise ValueError("겨루기 개체 수가 max_agents보다 많다")
    return Config.from_dict(d)


def inject_agent(world: World, genomes: dict, row: int, tag: int, x: float, y: float, heading: float,
                 age: int, gen: int = 0) -> int:
    """유전체 묶음의 row번째 개체를 세계의 빈 칸에 갓 태어난 상태로 넣고 그 개체의 id를 돌려준다.
    에너지는 e_start를 주고 장부(L_INJECT)에 적는다. 겨루기와 이주가 함께 쓴다."""
    s = world.s
    cfg = world.cfg
    j = int(np.flatnonzero(~s.alive)[0])
    aid = int(s.counters[k.C_NEXT_ID])
    s.counters[k.C_NEXT_ID] += 1
    s.counters[k.C_NALIVE] += 1
    s.alive[j] = True
    s.ids[j], s.parent[j], s.gen[j], s.tag[j] = aid, -1, gen, tag
    s.birth_step[j] = world.step_no - 1   # 넣은 스텝에 바로 번식하지 못하게 하는 표시와 겹치지 않게
    s.x[j], s.y[j] = x, y
    s.vx[j] = s.vy[j] = 0.0
    s.heading[j] = heading
    s.energy[j] = cfg.e_start
    s.ledger[k.L_INJECT] += cfg.e_start
    s.age[j] = age
    s.n_off[j] = 0
    s.eaten[j] = s.touch[j] = s.ate[j] = 0.0
    s.h[j] = 0.0
    # 평생 상태는 갓 태어난 개체와 같게 둔다(안 그러면 켠 세계의 생물을 학습이 꺼진 채로 재게 된다)
    s.e_prev[j] = cfg.e_start
    s.mod[j] = s.exp_r[j] = s.fatigue[j] = s.asleep[j] = 0.0
    s.last_proto[j] = 0
    s.a_held[j] = -1
    s.n_meal[j] = s.n_poison[j] = s.n_poison10[j] = 0
    s.meal_k[j] = 0
    s.meal_o[j] = 0.0
    s.n_seen[j] = s.n_seen_poison[j] = s.n_seen_kin[j] = 0
    s.seen_age_sum[j] = 0
    # 부모의 학습 보조 투자 계수(저장 안 함): 앎은 −1("아직 모름"), 나머지는 0(par_slot은 -1 = 부모 칸 없음, juv_pass가 스텝마다 다시 채운다)
    s.know_age[j] = -1
    s.know_src[j] = 0
    s.n_seen_par[j] = s.n_seen_par_poison[j] = s.n_taught[j] = s.n_taught_knower[j] = 0
    s.par_near_steps[j] = s.par_alive_steps[j] = s.juv_alive_steps[j] = 0
    s.adult_near_sum[j] = 0
    s.par_slot[j] = -1
    s.jscr_i[j] = 0
    s.jscr_d[j] = 0.0
    s.tflag[j] = 0
    # 규범: 표시·벌 없음, 삼가기 기억 0, 피해자는 −2(미지)
    s.mark_t[j] = s.pun_t[j] = -1
    s.mem[j] = 0.0
    s.mark_vslot[j] = s.mark_vid[j] = -2
    s.mark_pun[j] = 0
    s.pun_role[j] = -1
    for name in ("horm", "sig", "epost", "epre_h", "epre_in", "Hin", "Hrec", "proto", "link",
                 "elig_in", "elig_rec", "taste_m"):
        getattr(s, name)[j] = 0.0
    for name in GENOME:
        getattr(s, name)[j] = genomes[name][row]
    return aid


def plant_income(world: World, frac: float, tag: int, prev_id: np.ndarray, prev_eaten: np.ndarray) -> float:
    """심은 효과 눈금자: 태그 tag의 산 개체에 지난 호출 뒤 먹은 양(eaten 증가분)의 frac만큼 에너지를 더 준다.
    명시적 주입이라 장부 L_INJECT에 적는다(에너지 장부 규칙). e_max를 넘기지 않는다. 칸의 id가 바뀌었으면(새로 태어남)
    그 칸의 eaten 전체가 증가분이다. prev_id·prev_eaten을 지금 값으로 덮어써 다음 호출의 기준으로 삼는다."""
    s = world.s
    idx = np.flatnonzero(s.alive & (s.tag == tag))
    added = 0.0
    if len(idx) and frac > 0.0:
        same = s.ids[idx] == prev_id[idx]
        d = np.maximum(s.eaten[idx] - np.where(same, prev_eaten[idx], 0.0), 0.0)
        room = np.maximum(world.cfg.e_max - s.energy[idx], 0.0)
        add = np.minimum(frac * d, room).astype(s.energy.dtype)
        s.energy[idx] += add
        added = float(add.sum(dtype=np.float64))
        s.ledger[k.L_INJECT] += added
    prev_id[:] = s.ids
    prev_eaten[:] = s.eaten
    return added


def compete(cfg: Config, ga: dict, gb: dict, n_each: int, steps: int, seed: int,
            sample_every: int = 500, twin: bool = False, plant: float = 0.0, plant_every: int = 50) -> dict:
    """A(tag 0)와 B(tag 1)를 n_each씩 새 세계에 풀고 개체 수 변화를 본다.
    twin=True: 한 자리의 A와 B가 같은 줄(rb = ra)이다 — 같은 무리의 두 벌(막기 겨루기)에서 개체 차이 잡음을 지운다.
      두 유전체 묶음의 개체 수가 같아야 한다. 난수 소비는 twin=False와 같다(rb를 뽑고 버린다 → 자리·나이가 같다).
    plant > 0: 심은 효과 눈금자 — B가 먹은 양의 plant만큼을 plant_every스텝마다 더 준다(plant_income, L_INJECT)."""
    if twin and len(ga["n_hidden"]) != len(gb["n_hidden"]):
        raise ValueError("쌍둥이 겨루기는 두 묶음의 개체 수가 같아야 한다")
    rng = np.random.default_rng(seed)
    acfg = arena_config(cfg, seed, 2 * n_each)
    world = World(acfg, populate=False)
    world.reset_grad_table()      # 졸업표는 이 겨루기에서 태어난 새끼만(inject_agent로 들어온 개체는 기록에 안 든다)
    # 빈 세계에 먹이를 절반쯤 미리 깔아 둔다
    for _ in range(int(0.5 * acfg.food_cap * acfg.food_energy / max(acfg.food_inflow, 1e-9))):
        k.spawn_food(0, world.s, world.fp, world.ip)
    ra = rng.choice(len(ga["n_hidden"]), size=n_each, replace=len(ga["n_hidden"]) < n_each)
    rb = rng.choice(len(gb["n_hidden"]), size=n_each, replace=len(gb["n_hidden"]) < n_each)
    if twin:
        rb = ra
    px = rng.random(n_each) * acfg.world_w
    py = rng.random(n_each) * acfg.world_h
    ages = rng.integers(0, max(1, acfg.max_age // 2), size=n_each)
    for i in range(n_each):
        # 한 자리에 A 하나 B 하나. 칸 번호가 한쪽에 유리하지 않게 넣는 순서를 번갈아 바꾼다
        pair = [(ga, ra[i], 0), (gb, rb[i], 1)]
        if rng.random() < 0.5:
            pair.reverse()
        for g, row, tag in pair:
            inject_agent(world, g, int(row), tag, float(px[i]), float(py[i]),
                         float(rng.random() * 2 * math.pi), int(ages[i]))
    series = []
    planted = 0.0
    if plant > 0.0:
        prev_id, prev_eaten = world.s.ids.copy(), world.s.eaten.copy()
    for t in range(0, steps, sample_every):
        n = min(sample_every, steps - t)
        if plant > 0.0:
            done = 0
            while done < n:
                m = min(plant_every, n - done)
                world.run(m)
                done += m
                planted += plant_income(world, plant, 1, prev_id, prev_eaten)
        else:
            world.run(n)
        tags = world.s.tag[world.s.alive]
        series.append([world.step_no, int((tags == 0).sum()), int((tags == 1).sum())])
    world.drain_events()
    first10 = _first10(world)
    arr = np.asarray(series, dtype=float)
    # 한쪽이 전멸한 뒤에는 비가 더 변하지 않는다. 전멸한 표본까지만 쓴다
    # (그 뒤까지 넣으면 가장 크게 진 판의 기울기가 0에 가깝게 나온다).
    dead = np.flatnonzero((arr[:, 1] == 0) | (arr[:, 2] == 0))
    upto = arr[:dead[0] + 1] if len(dead) else arr
    keep = upto[upto[:, 0] > steps / 3.0]
    if len(keep) < 3:
        keep = upto     # 버릴 구간 안에서 승부가 났으면 처음부터 쓴다
    y = np.log((keep[:, 2] + 0.5) / (keep[:, 1] + 0.5))
    if len(keep) >= 3:
        r = float(np.polyfit(keep[:, 0], y, 1)[0])
    else:               # 표본이 모자라면 시작(1:1)에서 그 시점까지의 평균 기울기
        r = float(y[-1] / keep[-1, 0])
    return {"n_a": int(arr[-1, 1]), "n_b": int(arr[-1, 2]), "r": r, "series": series,
            "extinct": extinct_label(int(arr[-1, 1]), int(arr[-1, 2])), "first10": first10,
            "grad_by_tag": world.grad_table(),   # {태그 행: {GT_NAMES 이름: 값}}, 0=A·1=B(장치가 꺼진 세계는 전부 0)
            "plant_injected": planted}


# ---- 규범의 씨앗: 겨루기 마당 준비와 다태그 겨루기 ----
def norm_arena_side(path: Path, sets=()) -> dict:
    """체크포인트를 규범 블록 크기(preset=norm_size)로 올려 읽어 겨루기 한쪽을 만든다(늘어난 입력 열의 가중치는 0).
    cfg 없이 load_side로 읽으면 옛 크기 그대로라 norm 크기 세계와 check_same_shape를 못 통과한다.
    sets = 덧붙일 설정("key=value" 목록, 예: conflict_enabled=true)."""
    path = Path(path)
    cfg = ckpt.read_config(path).with_overrides(["preset=norm_size", *sets])
    world, _ = ckpt.load(path, cfg)
    return side_from_world(world, str(path))


def fix_resident_attack(genomes: dict) -> dict:
    """거주민 유전체의 사본: 공격 편향 −1, 공격 출력 행 0(편향만 고치면 Wout·h가 자라 양수가 되는 길이 남는다).
    충동이 없는 마당에서 거주민의 공격(가로채기·벌)이 0이 된다(관문 −1e의 전제)."""
    g = {key: v.copy() for key, v in genomes.items()}
    g["bout"][:, ACT_ATTACK] = -1.0
    g["Wout"][:, ACT_ATTACK, :] = 0.0
    return g


def _norm_rows_of(tags: np.ndarray) -> np.ndarray:
    """태그를 규범 계수표 행으로(0~3은 그 행, 그 밖은 마지막 행에 접는다 — 커널의 _norm_row와 같다)."""
    last = k.TAG_ROWS_NORM - 1
    return np.where((tags >= 0) & (tags < last), tags, last)


def net_income_by_tag(world: World) -> dict:
    """규범 계수표 행(태그)마다 개체 스텝당 순수입과 그 짝.
    net_per_step = (EATEN_DEAD + 산 개체 eaten 합 + TAKE_GAIN + PUN_FINE − TAKE_DMG_RECV − PUN_DMG_RECV − ATK_COST)
    ÷ ALIVE_STEPS. 같은 몸 쌍둥이끼리 수명 차이를 스텝으로 정규화한 값이고, 한 생 값은 × 평균 수명으로 환산한다.
    알려진 편향: 맞아 죽어 수명이 줄어드는 몫은 안 들어간다(죽으면 개체 스텝이 멈출 뿐) — 그래서
    eaten_per_life_done(끝난 생의 평균 먹은 양)과 life_mean(ALIVE_STEPS ÷ (끝난 생 + 산 개체))을 같이 낸다.
    산 개체 eaten은 그 생 전체의 값이라 계수표를 생 중간에 지운 세계(겨루기 마당이 아닌 곳)에서는 앞부분이 섞인다.
    분모가 0이면 None."""
    s = world.s
    tab = s.norm_tab
    rows = _norm_rows_of(s.tag[s.alive])
    eaten = s.eaten[s.alive]
    out = {}
    for r in range(tab.shape[0]):
        m = rows == r
        live_eaten = float(eaten[m].sum())
        n_alive = int(m.sum())
        steps = float(tab[r, k.NT_ALIVE_STEPS])
        lives = float(tab[r, k.NT_LIVES_DONE])
        net = (float(tab[r, k.NT_EATEN_DEAD]) + live_eaten + float(tab[r, k.NT_TAKE_GAIN]) + float(tab[r, k.NT_PUN_FINE])
               - float(tab[r, k.NT_TAKE_DMG_RECV]) - float(tab[r, k.NT_PUN_DMG_RECV]) - float(tab[r, k.NT_ATK_COST]))
        out[r] = {
            "net_per_step": net / steps if steps > 0 else None,
            "eaten_per_life_done": float(tab[r, k.NT_EATEN_DEAD]) / lives if lives > 0 else None,
            "life_mean": steps / (lives + n_alive) if lives + n_alive > 0 else None,
            "alive_steps": steps, "lives_done": lives, "alive_now": n_alive,
        }
    return out


def compete_multi(cfg: Config, groups, steps: int, seed: int, sample_every: int = 500,
                  return_world: bool = False):
    """여러 태그를 한 마당에: groups = [(유전체, 태그, 개체 수), ...]. 변이는 끈다(arena_config).
    같은 seed면 칸(넣는 차례)마다 같은 자리·방향·나이로 시작한다(태그 구성이 다른 마당끼리 같은 씨앗 짝으로 견준다).
    태그를 칸에 무작위로 섞어 넣어 칸 번호(처리 순서)가 한 태그에 유리하지 않게 한다.
    결과: 태그별 규범 계수표(norm_by_tag, NT_NAMES 사전)·역할표(role_table, 마당 전체 — 역할표는 태그로 안 나뉜다)·
    산 개체 수·끝 시점 eaten 합·표시 유병률·개체 스텝당 순수입(net_income_by_tag)·개체 수 표본(series)."""
    groups = list(groups)
    n_total = sum(int(c) for _, _, c in groups)
    rng = np.random.default_rng(seed)
    acfg = arena_config(cfg, seed, n_total)
    world = World(acfg, populate=False)
    world.reset_grad_table()
    world.reset_norm_tables()
    for _ in range(int(0.5 * acfg.food_cap * acfg.food_energy / max(acfg.food_inflow, 1e-9))):
        k.spawn_food(0, world.s, world.fp, world.ip)
    # 칸 배치(자리·방향·나이)를 먼저 뽑는다: 태그 구성과 무관하게 씨앗만으로 정해진다
    px = rng.random(n_total) * acfg.world_w
    py = rng.random(n_total) * acfg.world_h
    hd = rng.random(n_total) * 2 * math.pi
    ages = rng.integers(0, max(1, acfg.max_age // 2), size=n_total)
    entries = []
    for g, tag, count in groups:
        n_g = len(g["n_hidden"])
        rows = rng.choice(n_g, size=int(count), replace=n_g < int(count))
        entries += [(g, int(row), int(tag)) for row in rows]
    order = rng.permutation(n_total)
    for slot in range(n_total):
        g, row, tag = entries[order[slot]]
        inject_agent(world, g, row, tag, float(px[slot]), float(py[slot]), float(hd[slot]), int(ages[slot]))
    tag_list = sorted({int(tag) for _, tag, _ in groups})

    def counts() -> dict:
        tags = world.s.tag[world.s.alive]
        return {tag: int((tags == tag).sum()) for tag in tag_list}
    series = [[world.step_no, counts()]]
    t = 0
    while t < steps:
        n = min(sample_every, steps - t)
        world.run(n)
        t += n
        series.append([world.step_no, counts()])
    world.drain_events()
    s = world.s
    live = s.alive
    tab = s.norm_tab
    tot = tab.sum(axis=0)
    norm = world.norm_table()
    rows_of = {tag: int(_norm_rows_of(np.array([tag]))[0]) for tag in tag_list}
    res = {
        "n_total": n_total, "steps": steps, "seed": seed,
        "alive_by_tag": counts(),
        "eaten_by_tag": {tag: float(s.eaten[live & (s.tag == tag)].sum()) for tag in tag_list},
        "norm_by_tag": {tag: norm[rows_of[tag]] for tag in tag_list},
        "role_table": world.norm_role_table(),
        "mark_prevalence": float(tot[k.NT_MARKED_STEPS] / tot[k.NT_ALIVE_STEPS]) if tot[k.NT_ALIVE_STEPS] > 0 else 0.0,
        "mark_prevalence_by_tag": {tag: (float(tab[r, k.NT_MARKED_STEPS] / tab[r, k.NT_ALIVE_STEPS])
                                         if tab[r, k.NT_ALIVE_STEPS] > 0 else None) for tag, r in rows_of.items()},
        "net_income_by_tag": {tag: net_income_by_tag(world)[r] for tag, r in rows_of.items()},
        "series": series,
    }
    return (res, world) if return_world else res


def _first10(world: World) -> dict:
    """끝에 산 개체의 첫 FIRST_MEALS끼 기록을 태그별로 합친다: {태그: [독 합, 먹은 끼니 합]}.
    (독 합 = n_poison10, 끼니 합 = min(n_meal, FIRST_MEALS). 배운 개체가 첫 끼부터 덜 먹는지 보는 값.)"""
    s = world.s
    live = s.alive
    out = {}
    for tag in (0, 1):
        m = live & (s.tag == tag)
        out[tag] = [int(s.n_poison10[m].sum()), int(np.minimum(s.n_meal[m], FIRST_MEALS).sum())]
    return out


def extinct_label(n_a: int, n_b: int) -> str:
    """끝에 누가 전멸했는가. 둘 다 전멸한 판은 어느 쪽의 승패도 아니므로 따로 적는다
    (누가 먼저 무너졌는지는 전멸한 표본까지만 쓰는 r이 말한다)."""
    if n_a == 0 and n_b == 0:
        return "both"
    return "a" if n_a == 0 else ("b" if n_b == 0 else "")


def _split(g: dict, rng: np.random.Generator) -> tuple[dict, dict]:
    n = len(g["n_hidden"])
    p = rng.permutation(n)
    h = n // 2
    return ({key: v[p[:h]] for key, v in g.items()}, {key: v[p[h:]] for key, v in g.items()})


def _stats(vals: list[float]) -> dict:
    v = np.asarray(vals, dtype=float)
    sd = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
    return {"mean": float(v.mean()), "sd": sd, "sem": sd / math.sqrt(len(v)) if len(v) > 1 else float("nan"),
            "n": len(v)}


def assay_sides(a: dict, b: dict, n_each: int = 400, steps: int = 8000, reps: int = 6,
                seed: int = 1000, neutral: bool = True, twin: bool = False) -> dict:
    """twin=True: 쌍둥이 겨루기(compete 참고). 중립은 반으로 가르는 대신 A 묶음 두 벌을 쌍둥이로(빈 막기 대조)."""
    check_same_shape(a, b)
    cfg = b["cfg"]
    runs = [compete(cfg, a["genomes"], b["genomes"], n_each, steps, seed + i, twin=twin) for i in range(reps)]
    out = {
        "a": {key: a[key] for key in ("path", "step", "gen_mean")},
        "b": {key: b[key] for key in ("path", "step", "gen_mean")},
        "n_each": n_each, "steps": steps, "reps": reps,
        "r_per_1000": _stats([x["r"] * 1000 for x in runs]),
        "r_list": [x["r"] * 1000 for x in runs],          # 반복별 r(판정 도구가 두 판 사이를 비교할 때 쓴다)
        "first10_by_tag": {str(tag): [sum(x["first10"][tag][i] for x in runs) for i in (0, 1)]
                           for tag in (0, 1)},            # 태그별 첫 10끼 [독 합, 끼니 합](반복을 합침)
        "final_counts": [[x["n_a"], x["n_b"]] for x in runs],
        "grad_by_tag_list": [x["grad_by_tag"] for x in runs],   # 반복 순서대로 태그별 졸업표
        "a_extinct_runs": sum(x["extinct"] == "a" for x in runs),
        "b_extinct_runs": sum(x["extinct"] == "b" for x in runs),
        "both_extinct_runs": sum(x["extinct"] == "both" for x in runs),
    }
    if twin:
        out["twin"] = True
    if neutral:
        rng = np.random.default_rng(seed + 7919)
        neu = []
        for i in range(reps):
            if twin:
                g = a["genomes"] if i % 2 == 0 else b["genomes"]
                neu.append(compete(cfg, g, g, n_each, steps, seed + 500 + i, twin=True)["r"] * 1000)
                continue
            g1, g2 = _split(a["genomes"] if i % 2 == 0 else b["genomes"], rng)
            neu.append(compete(cfg, g1, g2, n_each, steps, seed + 500 + i)["r"] * 1000)
        out["neutral_r_per_1000"] = _stats(neu)
    out["winner"] = winner(out)
    out["verdict"] = verdict(out)
    return out


def ruler(side: dict, fracs=(0.0, 0.005, 0.01, 0.02, 0.05), n_each: int = 400, steps: int = 8000, reps: int = 6,
          seed: int = 1000, twin: bool = True, sample_every: int = 500) -> dict:
    """심은 효과 눈금자: 같은 무리 두 벌을 겨루며 B에만 먹은 양의 frac을 더 준다. 칸마다 r(1,000스텝당)과
    원점을 지나지 않는 직선 기울기 r_per_unit_plant(r ÷ frac). 판정에서 "가를 수 없음"을 "섭취의 몇 % 미만"으로
    옮길 때 plant_equiv로 쓴다. 씨앗은 칸마다 같다(같은 자리·나이에서 frac만 바뀐다)."""
    cfg, g = side["cfg"], side["genomes"]
    cells, xs, ys = [], [], []
    for f in fracs:
        rs = [compete(cfg, g, g, n_each, steps, seed + i, sample_every=sample_every, twin=twin, plant=float(f))["r"] * 1000
              for i in range(reps)]
        cells.append({"plant": float(f), "r_per_1000": _stats(rs), "r_list": rs})
        xs += [float(f)] * len(rs)
        ys += rs
    slope = float(np.polyfit(xs, ys, 1)[0]) if len(set(xs)) > 1 else None
    return {"path": side.get("path", ""), "n_each": n_each, "steps": steps, "reps": reps, "twin": twin,
            "cells": cells, "r_per_unit_plant": slope}


def plant_equiv(r_per_1000: float, ruler_res: dict) -> float | None:
    """r(1,000스텝당)을 눈금자 기울기로 나눠 "B가 먹은 양의 몇 배를 더 받은 것과 같은가"(0.01 = 섭취의 1%)."""
    sl = ruler_res.get("r_per_unit_plant")
    return r_per_1000 / sl if sl else None


def assay(path_a: Path, path_b: Path, n_each: int = 400, steps: int = 8000, reps: int = 6,
          seed: int = 1000, neutral: bool = True) -> dict:
    return assay_sides(load_side(path_a), load_side(path_b), n_each, steps, reps, seed, neutral)


def winner(res: dict) -> str:
    """겨루기의 승자: "a", "b", 가를 수 없으면 "". 임시 기준: 반복 사이 t 검정(양측 95%), 한쪽이 매번 전멸하면 그쪽이 진다.
    판정을 코드에서 가를 때는 문장(verdict)이 아니라 이 값을 쓴다."""
    r = res["r_per_1000"]
    if res.get("a_extinct_runs") == res["reps"]:
        return "b"
    if res.get("b_extinct_runs") == res["reps"]:
        return "a"
    if not np.isfinite(r["sem"]):
        return ""
    edge = t95(r["n"] - 1) * r["sem"]
    if r["mean"] > edge:
        return "b"
    if r["mean"] < -edge:
        return "a"
    return ""


def verdict(res: dict, a_label: str = "A가", b_label: str = "B가") -> str:
    """winner를 사람이 읽는 문장으로 적는다. 같은 무리끼리의 평균이 0에서 멀면 측정 자체를 의심한다.
    두 쪽을 다른 이름으로 부르려면 a_label, b_label을 준다(문장을 나중에 고쳐 쓰지 않는다)."""
    r = res["r_per_1000"]
    if res.get("a_extinct_runs") == res["reps"]:
        return f"{b_label} 이김({a_label} {res['reps']}회 모두 전멸)"
    if res.get("b_extinct_runs") == res["reps"]:
        return f"{a_label} 이김({b_label} {res['reps']}회 모두 전멸)"
    if not np.isfinite(r["sem"]):
        return "가를 수 없음(반복 부족)"
    nz = res.get("neutral_r_per_1000")
    warn = ""
    if res.get("both_extinct_runs"):
        warn = f" (주의: {res['both_extinct_runs']}회는 양쪽 다 전멸)"
    if nz and np.isfinite(nz["sem"]) and abs(nz["mean"]) > t95(nz["n"] - 1) * nz["sem"]:
        warn += " (주의: 같은 무리끼리도 한쪽으로 기울었다)"
    return {"b": f"{b_label} 이김", "a": f"{a_label} 이김", "": "가를 수 없음"}[winner(res)] + warn


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def ladder(run_dir: Path, count: int, n_each: int, steps: int, reps: int, seed: int,
           out_path: Path | None = None) -> dict:
    """한 실행의 체크포인트를 고르게 뽑아 모든 쌍을 겨루게 한다(행 = 옛것, 열 = 새것)."""
    cps = [c for c in ckpt.list_checkpoints(Path(run_dir) / "checkpoints") if c[0] > 0]
    if len(cps) < 2:
        raise ValueError("체크포인트가 2개 이상 있어야 한다")
    pick = sorted({int(round(i)) for i in np.linspace(0, len(cps) - 1, min(count, len(cps)))})
    # 살아 있는 실행은 도는 중에 옛 체크포인트를 지운다. 고른 것을 먼저 다 읽어 둔다
    sides = [load_side(cps[i][1]) for i in pick]
    n = len(sides)
    mat = np.zeros((n, n))
    cells = []
    res: dict = {}
    for i in range(n):
        for j in range(i + 1, n):
            r = assay_sides(sides[i], sides[j], n_each, steps, reps, seed, neutral=False)["r_per_1000"]
            mat[i, j] = r["mean"]
            mat[j, i] = -r["mean"]
            cells.append({"old": sides[i]["step"], "new": sides[j]["step"], **r,
                          "significant": bool(np.isfinite(r["sem"]) and abs(r["mean"]) > t95(r["n"] - 1) * r["sem"])})
            res = _ladder_summary(sides, mat, cells)
            if out_path:
                Path(out_path).write_text(json.dumps(res, indent=1))   # 칸마다 저장(중간에 죽어도 남게)
    return res


def _ladder_summary(sides: list[dict], mat: np.ndarray, cells: list[dict]) -> dict:
    n = len(sides)
    # 순환과 진보 가르기(Balduzzi 2019의 이산판): 등급 f_i = 행 평균, 추이 성분 T_ij = f_i - f_j
    win = -mat                       # win[i, j] > 0 이면 i가 j를 이김
    f = win.mean(axis=1)
    trans = f[:, None] - f[None, :]
    total = float((win ** 2).sum())
    steps = np.array([s["step"] for s in sides], dtype=float)
    return {
        "steps": [s["step"] for s in sides],
        "r_new_minus_old_per_1000": mat.tolist(),
        "rating": f.tolist(),
        # 추이 성분 비율은 방향을 모른다(퇴보만 해도 1). 순수 잡음에서도 기대값이 2/n이다.
        "transitive_share": float((trans ** 2).sum() / total) if total > 0 else None,
        "transitive_share_if_noise": 2.0 / n,
        # 등급이 시간에 따라 오르는가(+1이면 늘 새것이 강함, -1이면 늘 옛것이 강함)
        "rating_time_spearman": _spearman(steps, f),
        "cells": cells,
        "cells_done": len(cells), "cells_total": n * (n - 1) // 2,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="성장 측정")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("assay")
    a.add_argument("--a", required=True, type=Path, help="옛 체크포인트(또는 처음 보는 상대). 생물 묶음(pack)도 된다")
    a.add_argument("--b", required=True, type=Path, help="지금 체크포인트. 생물 묶음(pack)도 된다. 겨루는 세계는 B의 설정을 쓴다")
    la = sub.add_parser("ladder")
    la.add_argument("--run", required=True, type=Path)
    la.add_argument("--count", type=int, default=5)
    ru = sub.add_parser("ruler", help="심은 효과 눈금자: 같은 무리 두 벌, B에만 먹은 양의 frac을 더 준다")
    ru.add_argument("--ckpt", required=True, type=Path)
    ru.add_argument("--fracs", type=float, nargs="+", default=[0.0, 0.005, 0.01, 0.02, 0.05])
    ru.add_argument("--no-twin", action="store_true", help="쌍둥이 대신 따로 뽑은 줄(옛 겨루기 꼴)")
    for p in (a, la, ru):
        p.add_argument("--n", type=int, default=400, help="한쪽 개체 수(둘을 합쳐 수용력 근처가 되게)")
        p.add_argument("--steps", type=int, default=8000)
        p.add_argument("--reps", type=int, default=6)
        p.add_argument("--seed", type=int, default=1000)
        p.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "ruler":
        res = ruler(load_side(args.ckpt), args.fracs, args.n, args.steps, args.reps, args.seed, twin=not args.no_twin)
        for c in res["cells"]:
            r = c["r_per_1000"]
            print(f"섭취의 {c['plant'] * 100:5.2f}% 더: r = {r['mean']:+.4f} ± {r['sd']:.4f}(표준편차, {r['n']}회)")
        sl = res["r_per_unit_plant"]
        print(f"기울기: 섭취의 1%당 r {sl / 100:+.4f}" if sl is not None else "기울기: 칸이 하나라 없음")
        if args.out:
            args.out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
        return 0
    if args.cmd == "assay":
        res = assay(args.a, args.b, args.n, args.steps, args.reps, args.seed)
        r = res["r_per_1000"]
        print(f"A: step {res['a']['step']} (평균 세대 {res['a']['gen_mean']:.1f})  "
              f"B: step {res['b']['step']} (평균 세대 {res['b']['gen_mean']:.1f})")
        print(f"r(B-A) / 1000스텝 = {r['mean']:+.4f} ± {r['sem']:.4f} (표준오차, {r['n']}회)")
        if "neutral_r_per_1000" in res:
            nz = res["neutral_r_per_1000"]
            print(f"같은 무리끼리(잡음): 평균 {nz['mean']:+.4f}, 표준편차 {nz['sd']:.4f}")
        print("끝 개체 수 [A, B]:", res["final_counts"])
        print("판정:", res["verdict"])
        if args.out:
            args.out.write_text(json.dumps(res, indent=1))
    else:
        res = ladder(args.run, args.count, args.n, args.steps, args.reps, args.seed, args.out)
        print("스텝:", res["steps"])
        print("r(새것-옛것)/1000스텝, 행=옛것 열=새것")
        for row in res["r_new_minus_old_per_1000"]:
            print("  " + " ".join(f"{v:+7.3f}" for v in row))
        sig = sum(c["significant"] for c in res["cells"])
        print(f"t 검정(95%)을 넘은 칸: {sig} / {len(res['cells'])}")
        print("등급(클수록 강함):", [round(v, 3) for v in res["rating"]])
        print(f"등급과 시간의 순위 상관: {res['rating_time_spearman']:+.2f} (+1이면 늘 새것이 강함)")
        print(f"추이 성분 비율: {res['transitive_share']:.3f} (순수 잡음이면 {res['transitive_share_if_noise']:.2f}쯤. "
              "방향은 순위 상관으로 본다)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
