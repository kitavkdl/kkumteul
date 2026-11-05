"""numba 커널. 상태는 numpy 배열 묶음(State)이고 계산은 여기서 한다.

재현성 규칙
- 병렬(prange) 커널은 자기 개체 칸에만 쓴다. 합계는 순차 커널에서만 낸다.
- fastmath를 쓰지 않는다.
- 난수는 rng의 상태 없는 해시만 쓴다.
- 대사 에너지가 늘어나는 길은 먹기와 명시적 주입(처음·재주입)뿐이다. 충돌·경계는 에너지를 건드리지 않는다.
"""
import math
from collections import namedtuple

import numpy as np
from numba import njit, prange

from .config import ACT_ATTACK, ACT_CARRY, ACT_DEPOSIT, ACT_GIVE, ACT_REPRO, ACT_THRUST, ACT_TURN, N_REWARD
from .rng import (S_BIRTH_ORDER, S_BIRTH_POS, S_FOOD, S_KIND, S_INIT_AGE, S_INIT_GENOME, S_INIT_POS, S_MUT_GENE,
                  S_MUT_W, S_NOISE, S_PASSIVE, S_PATCH, S_RESEED, normal, u01)

# ---- 상태 묶음 ----
AGENT_FIELDS = [
    # (이름, dtype, 뒤 차원, 저장 여부)
    ("alive", np.bool_, (), True),
    ("ids", np.int64, (), True),
    ("parent", np.int64, (), True),
    ("gen", np.int32, (), True),
    ("tag", np.int32, (), True),
    ("birth_step", np.int64, (), True),
    ("x", np.float64, (), True),
    ("y", np.float64, (), True),
    ("vx", np.float64, (), True),
    ("vy", np.float64, (), True),
    ("heading", np.float64, (), True),
    ("energy", np.float64, (), True),
    ("age", np.int32, (), True),
    ("n_off", np.int32, (), True),
    ("eaten", np.float64, (), True),
    ("touch", np.float32, (), True),
    ("ate", np.float32, (), True),
    ("n_hidden", np.int32, (), True),
    ("sigma", np.float32, (), True),
    ("color", np.float32, ("3",), True),
    ("Win", np.float32, ("I", "H"), True),    # 입력 j -> 은닉 k
    ("Wrec", np.float32, ("H", "H"), True),   # 은닉 j -> 은닉 k
    ("bh", np.float32, ("H",), True),
    ("Wout", np.float32, ("O", "H"), True),   # 은닉 k -> 출력 o
    ("bout", np.float32, ("O",), True),
    ("h", np.float32, ("H",), True),
    # 2단계 유전자: 보상 가중치, 학습률, 뉴런별 가소성
    ("rw", np.float32, ("NR",), True),
    ("eta", np.float32, (), True),
    ("alpha", np.float32, ("H",), True),
    # 2~3단계 평생 상태(태어날 때 0). PI·PH·NP·ND는 마음이 꺼져 있으면 1이다
    ("horm", np.float32, ("4",), True),       # 허기, 스트레스, 도파민, 지루함
    ("mod", np.float32, (), True),            # 이번 스텝의 학습 조절 신호
    ("exp_r", np.float32, (), True),          # 기대 보상
    ("e_prev", np.float64, (), True),
    ("fatigue", np.float32, (), True),
    ("asleep", np.float32, (), True),
    ("epost", np.float32, ("PH",), True),
    ("epre_h", np.float32, ("PH",), True),
    ("epre_in", np.float32, ("PI",), True),
    ("Hin", np.float32, ("PI", "PH"), True),  # 살면서 배운 것(입력 -> 은닉)
    ("Hrec", np.float32, ("PH", "PH"), True),
    ("proto", np.float32, ("NP", "ND"), True),
    ("link", np.float32, ("NP", "NP"), True),
    ("last_proto", np.int32, (), True),
    ("sig", np.float32, ("2",), True),        # 지난 스텝에 낸 신호
    # 노드 섭동(perturb_sigma > 0)의 연결별 흔적: 무엇이 들어올 때 어느 뉴런을 흔들었는가. EI·EH는 꺼져 있으면 1이다
    ("elig_in", np.float32, ("EI", "EH"), True),
    ("elig_rec", np.float32, ("EH", "EH"), True),
    ("hnew", np.float32, ("H",), False),
    ("obs", np.float32, ("I",), False),
    ("act", np.float32, ("O",), False),
    ("dispx", np.float64, (), False),
    ("dispy", np.float64, (), False),
]
FOOD_FIELDS = [
    ("fx", np.float64, True),
    ("fy", np.float64, True),
    ("f_alive", np.bool_, True),
    ("f_energy", np.float64, True),
    ("f_kind", np.int8, True),       # 먹이 종류 0/1, 2는 늘 먹을 수 있는 것(사체)
    ("f_born", np.int64, True),      # 놓인 스텝(상한에서 가장 오래된 것을 고를 때, 부패의 나이를 셀 때 쓴다)
]
# ---- 먹이 종류 (f_kind) ----
KIND_A = 0           # 시야에서 양수로 보이는 종류
KIND_B = 1           # 시야에서 음수로 보이는 종류
KIND_CORPSE = 2      # 죽은 개체가 남긴 먹이. 어느 쪽이 독이든 늘 먹을 수 있고 시야에서는 양수로 보인다
TOUCH_HIT = 2.0      # 이번 스텝에 공격을 맞은 개체의 닿음 감각(그냥 닿으면 1.0)

OTHER_SAVED = ["counters", "ledger", "budget", "trail", "soil"]
# 크기가 설정에 따라 달라도 되는 평생 상태(체크포인트와 크기가 다르면 0에서 시작한다)
LIFETIME_FLEX = ("epost", "epre_h", "epre_in", "Hin", "Hrec", "proto", "link", "elig_in", "elig_rec")
# 노드 섭동을 켰을 때만 크게 잡는 배열
PERTURB_FLEX = ("elig_in", "elig_rec")
OTHER_SCRATCH = ["a_start", "a_items", "f_start", "f_items", "cursor", "bbuf", "dbuf", "dbuf_f"]

State = namedtuple(
    "State",
    [f[0] for f in AGENT_FIELDS] + [f[0] for f in FOOD_FIELDS] + OTHER_SAVED + OTHER_SCRATCH,
)
SAVED_FIELDS = ([f[0] for f in AGENT_FIELDS if f[3]] + [f[0] for f in FOOD_FIELDS if f[2]]
                + OTHER_SAVED)

# ---- counters 칸 ----
C_NEXT_ID = 0
C_FOOD_SERIAL = 1
C_NFOOD = 2
C_FOOD_CURSOR = 3
C_BIRTHS = 4
C_DEATH_STARVE = 5
C_DEATH_AGE = 6
C_RESEED = 7
C_BLOCKED = 8
C_EATEN = 9
C_NALIVE = 10
C_DEATH_PASSIVE = 11
C_STEP = 12
C_DEATH_DISASTER = 13   # 각본의 재난으로 죽은 수
C_GIFTS = 14            # 건넨 횟수 (stages.give)
C_FERTILE = 15          # 거름 땅에 난 먹이 수 (stages.niche_world)
C_ATTACKS = 16          # 공격이 맞은 횟수 (stages.attack)
C_POISONED = 17
C_FOOD_REPLACED = 18    # 상한에서 새 먹이에 자리를 내주고 사라진 먹이 수(food_cap_replace)
C_DEATH_KILLED = 19     # 공격을 맞은 스텝에 에너지가 바닥나 죽은 수
C_MIGRANTS = 20         # 다른 세계에서 옮겨 심은 개체 수(pack.migrate). 0이 아니면 이주가 있었던 세계다
C_SPOILED = 21          # 썩어서 사라진 먹이 수(spoil_food)
N_COUNTERS = 24         # 번호는 이 파일에만 둔다. 남는 칸은 다음 장치가 쓴다(늘리지 않아도 되게 비워 둔 것)

# ---- ledger 칸 (에너지 장부) ----
L_SPAWN = 0      # 먹이로 세계에 들어온 에너지
L_BASAL = 1      # 기초 대사로 사라진 에너지
L_BRAIN = 2      # 뇌 유지비
L_MOVE = 3       # 이동·회전 비용
L_BIRTH = 4      # 번식 때 사라진 에너지
L_LOST = 5       # 세계에서 사라진 에너지: 죽을 때 먹이로 못 돌아간 것(굶어 죽으면 음수 잔액 포함),
                 # 독·재난·상한에서 밀려나 없어진 먹이, 썩어 줄어든 먹이, 건네기·공격·거름의 손실
L_INJECT = 6     # 처음·재주입·대조 실행에서 개체에게 직접 준 에너지
L_DISCARD = 7    # 먹이 상한 때문에 버린 유입(세계에 들어오지 않음)
N_LEDGER = 8

# ---- 사건 종류 ----
BIRTH_NATURAL = 0
BIRTH_INITIAL = 1
BIRTH_RESEED = 2
BIRTH_INJECT = 3     # 겨루기 실험에서 넣은 개체
BIRTH_PASSIVE = 4
DEATH_STARVE = 0
DEATH_AGE = 1
DEATH_PASSIVE = 2
DEATH_DISASTER = 3       # 각본의 재난(운석)
DEATH_KILLED = 4         # 공격을 맞은 스텝에 에너지가 바닥남

# ---- 실수 설정 칸 ----
(P_W, P_H, P_INV_CELL, P_INFLOW, P_E_FOOD, P_PATCH_SIGMA, P_UNIFORM, P_A_RAD, P_F_RAD,
 P_THRUST, P_BACK, P_DRAG, P_VMAX, P_MAX_TURN, P_PUSH, P_E_START, P_E_MAX, P_E_REPRO,
 P_CHILD_FRAC, P_BIRTH_COST, P_BASAL, P_BRAIN, P_BRAIN_ACT, P_THRUST_COST, P_TURN_COST,
 P_FOV, P_VRANGE, P_VNOISE, P_SMELL_LEN, P_SMELL_GAIN, P_INIT_WIN, P_INIT_WREC, P_INIT_WOUT,
 P_BIAS_THRUST, P_BIAS_REPRO, P_SIGMA_INIT, P_SIGMA_MIN, P_SIGMA_MAX, P_SIGMA_TAU,
 P_P_HIDDEN, P_COLOR_SIGMA, P_SYN, P_BACK_COST, P_ETA_INIT, P_ETA_MAX, P_ALPHA_INIT, P_POISON,
 P_KIND_RATIO, P_SPOIL_RATE, P_SPOIL_MIN, N_FP) = range(51)

# ---- 정수 설정 칸 ----
(Q_SEED, Q_GW, Q_GH, Q_FOOD_CAP, Q_N_PATCH, Q_PATCH_LIFE, Q_MAX_AGE, Q_MIN_AGENTS,
 Q_SECTORS, Q_H_MAX, Q_H_INIT, Q_H_MIN, Q_REPRO_BRAIN, Q_REPRO_MIN_AGE, Q_MUT, Q_CORPSE,
 Q_PASSIVE, Q_KINDS, Q_KIND_PERIOD, Q_CAP_REPLACE, Q_SPOIL_AFTER, N_IP) = range(22)


def pack_params(cfg):
    fp = np.zeros(N_FP, dtype=np.float64)
    ip = np.zeros(N_IP, dtype=np.int64)
    fp[P_W] = cfg.world_w
    fp[P_H] = cfg.world_h
    fp[P_INV_CELL] = 1.0 / cfg.cell
    fp[P_INFLOW] = cfg.food_inflow
    fp[P_E_FOOD] = cfg.food_energy
    fp[P_PATCH_SIGMA] = cfg.patch_sigma
    fp[P_UNIFORM] = cfg.food_uniform_frac
    fp[P_A_RAD] = cfg.agent_radius
    fp[P_F_RAD] = cfg.food_radius
    fp[P_THRUST] = cfg.thrust_force
    fp[P_BACK] = cfg.back_frac
    fp[P_DRAG] = cfg.drag
    fp[P_VMAX] = cfg.v_max
    fp[P_MAX_TURN] = cfg.max_turn
    fp[P_PUSH] = cfg.collide_max_push
    fp[P_E_START] = cfg.e_start
    fp[P_E_MAX] = cfg.e_max
    fp[P_E_REPRO] = cfg.e_repro
    fp[P_CHILD_FRAC] = cfg.child_frac
    fp[P_BIRTH_COST] = cfg.birth_cost
    fp[P_BASAL] = cfg.basal_cost
    fp[P_BRAIN] = cfg.brain_cost
    fp[P_BRAIN_ACT] = cfg.brain_activity_cost
    fp[P_THRUST_COST] = cfg.thrust_cost
    fp[P_TURN_COST] = cfg.turn_cost
    fp[P_FOV] = cfg.vision_fov
    fp[P_VRANGE] = cfg.vision_range
    fp[P_VNOISE] = cfg.vision_noise
    fp[P_SMELL_LEN] = cfg.smell_length
    fp[P_SMELL_GAIN] = cfg.smell_gain
    fp[P_INIT_WIN] = cfg.init_w_in
    fp[P_INIT_WREC] = cfg.init_w_rec
    fp[P_INIT_WOUT] = cfg.init_w_out
    fp[P_BIAS_THRUST] = cfg.init_bias_thrust
    fp[P_BIAS_REPRO] = cfg.init_bias_repro
    fp[P_SIGMA_INIT] = cfg.sigma_init
    fp[P_SIGMA_MIN] = cfg.sigma_min
    fp[P_SIGMA_MAX] = cfg.sigma_max
    fp[P_SIGMA_TAU] = cfg.sigma_tau
    fp[P_P_HIDDEN] = cfg.p_hidden_change
    fp[P_COLOR_SIGMA] = cfg.color_sigma
    fp[P_SYN] = cfg.synapse_cost
    fp[P_BACK_COST] = cfg.back_cost_mult
    fp[P_ETA_INIT] = cfg.eta_init
    fp[P_ETA_MAX] = cfg.eta_max
    fp[P_ALPHA_INIT] = cfg.alpha_init
    fp[P_POISON] = cfg.poison
    fp[P_KIND_RATIO] = cfg.kind_ratio
    fp[P_SPOIL_RATE] = cfg.spoil_rate
    fp[P_SPOIL_MIN] = cfg.spoil_min_frac * cfg.food_energy
    ip[Q_SEED] = cfg.seed
    ip[Q_GW] = cfg.grid_w
    ip[Q_GH] = cfg.grid_h
    ip[Q_FOOD_CAP] = cfg.food_cap
    ip[Q_N_PATCH] = cfg.n_patches
    ip[Q_PATCH_LIFE] = cfg.patch_life
    ip[Q_MAX_AGE] = cfg.max_age
    ip[Q_MIN_AGENTS] = cfg.min_agents
    ip[Q_SECTORS] = cfg.vision_sectors
    ip[Q_H_MAX] = cfg.h_max
    ip[Q_H_INIT] = cfg.h_init
    ip[Q_H_MIN] = cfg.h_min
    ip[Q_REPRO_BRAIN] = int(cfg.repro_brain_controlled)
    ip[Q_REPRO_MIN_AGE] = cfg.repro_min_age
    ip[Q_MUT] = int(cfg.mutation_enabled)
    ip[Q_CORPSE] = int(cfg.corpse_to_food)
    ip[Q_PASSIVE] = int(cfg.passive)
    ip[Q_KINDS] = int(cfg.food_kinds_enabled)
    ip[Q_KIND_PERIOD] = cfg.kind_flip_period
    ip[Q_CAP_REPLACE] = int(cfg.food_cap_replace)
    ip[Q_SPOIL_AFTER] = cfg.spoil_after
    return fp, ip


def alloc_state(cfg) -> State:
    n = cfg.max_agents
    mind = cfg.mind_enabled
    perturb = mind and cfg.perturb_sigma > 0.0
    dims = {"3": 3, "4": 4, "2": 2, "I": cfg.n_in, "H": cfg.h_max, "O": cfg.n_out, "NR": N_REWARD,
            "PI": cfg.n_in if mind else 1, "PH": cfg.h_max if mind else 1,
            "EI": cfg.n_in if perturb else 1, "EH": cfg.h_max if perturb else 1,
            "NP": cfg.n_proto if mind else 1, "ND": 2 * cfg.vision_sectors if mind else 1}
    d = {}
    for name, dt, extra, _ in AGENT_FIELDS:
        d[name] = np.zeros((n,) + tuple(dims[e] for e in extra), dtype=dt)
    nf = cfg.food_slots
    for name, dt, _ in FOOD_FIELDS:
        d[name] = np.zeros(nf, dtype=dt)
    ncell = cfg.grid_w * cfg.grid_h
    d["counters"] = np.zeros(N_COUNTERS, dtype=np.int64)
    d["ledger"] = np.zeros(N_LEDGER, dtype=np.float64)
    d["budget"] = np.zeros(2, dtype=np.float64)   # [0] 보통 먹이, [1] 거름 땅에 날 먹이
    d["trail"] = np.zeros(ncell, dtype=np.float64)
    d["soil"] = np.zeros(ncell, dtype=np.float64)
    d["a_start"] = np.zeros(ncell + 1, dtype=np.int32)
    d["a_items"] = np.zeros(n, dtype=np.int32)
    d["f_start"] = np.zeros(ncell + 1, dtype=np.int32)
    d["f_items"] = np.zeros(nf, dtype=np.int32)
    d["cursor"] = np.zeros(ncell, dtype=np.int32)
    d["bbuf"] = np.zeros((2 * n + 8, 6), dtype=np.int64)
    d["dbuf"] = np.zeros((2 * n + 8, 7), dtype=np.int64)
    d["dbuf_f"] = np.zeros(2 * n + 8, dtype=np.float64)
    return State(**d)


# ---- 작은 도우미 ----
@njit(cache=True, inline="always")
def _wrap(v, size):
    v = v - math.floor(v / size) * size
    if v >= size:
        v = 0.0
    return v


@njit(cache=True, inline="always")
def _wdiff(d, size):
    half = 0.5 * size
    if d > half:
        d -= size
    elif d < -half:
        d += size
    return d


@njit(cache=True)
def patch_center(seed, k, step, n_patches, patch_life, w, h):
    """먹이 자리 k의 지금 위치. step만으로 정해진다(저장할 상태 없음)."""
    e = (step + (k * patch_life) // n_patches) // patch_life
    return u01(seed, S_PATCH, k, 2 * e) * w, u01(seed, S_PATCH, k, 2 * e + 1) * h


@njit(cache=True)
def good_kind(seed, step, period):
    """지금 먹어도 되는 먹이 종류(0 또는 1). 구간마다 무작위로 다시 정해진다."""
    return int(u01(seed, S_KIND, step // period, 1) * 2.0)


@njit(cache=True, inline="always")
def pick_kind(seed, a, b, ratio):
    """새로 나는 먹이의 종류. ratio는 KIND_B가 나는 비율이다. 난수 번호 (a, b)는 부르는 쪽이 서로 겹치지 않게 준다."""
    if u01(seed, S_KIND, a, b) < ratio:
        return KIND_B
    return KIND_A


@njit(cache=True)
def _place_food(s, px, py, e, kind, step):
    """빈 먹이 칸에 놓는다. 자리가 없으면 False."""
    nslots = s.fx.shape[0]
    cur = s.counters[C_FOOD_CURSOR]
    for _ in range(nslots):
        if cur >= nslots:
            cur = 0
        if not s.f_alive[cur]:
            s.fx[cur] = px
            s.fy[cur] = py
            s.f_energy[cur] = e
            s.f_kind[cur] = kind
            s.f_born[cur] = step
            s.f_alive[cur] = True
            s.counters[C_NFOOD] += 1
            s.counters[C_FOOD_CURSOR] = cur + 1
            return True
        cur += 1
    return False


@njit(cache=True)
def _drop_food_slot(s, j):
    """먹이 칸 j를 비운다(f_alive 끄고 먹이 수를 하나 줄인다). 에너지 장부는 부르는 쪽이 쓴다."""
    s.f_alive[j] = False
    s.counters[C_NFOOD] -= 1


@njit(cache=True)
def _evict_oldest(s):
    """가장 먼저 놓인 먹이 하나를 없앤다(같은 스텝에 놓인 것끼리는 낮은 칸부터, 사체도 가리지 않는다).
    없앤 에너지는 장부의 L_LOST로 간다. 없앨 먹이가 없으면 False."""
    best = -1
    born = 0
    for j in range(s.fx.shape[0]):
        if s.f_alive[j] and (best < 0 or s.f_born[j] < born):
            best = j
            born = s.f_born[j]
    if best < 0:
        return False
    s.f_alive[best] = False
    s.counters[C_NFOOD] -= 1
    s.counters[C_FOOD_REPLACED] += 1
    s.ledger[L_LOST] += s.f_energy[best]
    return True


@njit(cache=True)
def make_room(s, cap, replace):
    """새 먹이 한 알을 놓아도 되는가. 상한에 닿았으면 replace일 때만 가장 오래된 것을 치우고 True."""
    if s.counters[C_NFOOD] < cap:
        return True
    if replace == 0:
        return False
    return _evict_oldest(s)


# ---- 먹이 ----
@njit(cache=True)
def spawn_food(step, s, fp, ip, mult=1.0):
    seed = ip[Q_SEED]
    w = fp[P_W]
    h = fp[P_H]
    e_food = fp[P_E_FOOD]
    n_patches = ip[Q_N_PATCH]
    s.budget[0] += fp[P_INFLOW] * mult
    while s.budget[0] >= e_food:
        if not make_room(s, ip[Q_FOOD_CAP], ip[Q_CAP_REPLACE]):
            s.ledger[L_DISCARD] += s.budget[0]
            s.budget[0] = 0.0
            break
        serial = s.counters[C_FOOD_SERIAL]
        s.counters[C_FOOD_SERIAL] += 1
        if n_patches == 0 or u01(seed, S_FOOD, serial, 0) < fp[P_UNIFORM]:
            px = u01(seed, S_FOOD, serial, 1) * w
            py = u01(seed, S_FOOD, serial, 2) * h
        else:
            k = int(u01(seed, S_FOOD, serial, 3) * n_patches)
            cx, cy = patch_center(seed, k, step, n_patches, ip[Q_PATCH_LIFE], w, h)
            px = _wrap(cx + fp[P_PATCH_SIGMA] * normal(seed, S_FOOD, serial, 4), w)
            py = _wrap(cy + fp[P_PATCH_SIGMA] * normal(seed, S_FOOD, serial, 5), h)
        kind = KIND_A
        if ip[Q_KINDS] != 0:
            kind = pick_kind(seed, serial, 0, fp[P_KIND_RATIO])
        if _place_food(s, px, py, e_food, kind, step):
            s.ledger[L_SPAWN] += e_food
            s.budget[0] -= e_food
        else:
            s.ledger[L_DISCARD] += s.budget[0]
            s.budget[0] = 0.0
            break


@njit(cache=True)
def spoil_food(step, s, fp, ip, rate_mult=1.0):
    """난 지 spoil_after 스텝이 지난 먹이(사체 포함)를 스텝마다 spoil_rate * rate_mult 비율로 줄인다.
    줄어든 에너지는 장부의 L_LOST로 가고, 남은 것이 하한(spoil_min_frac * food_energy) 아래면 남은 것도
    L_LOST로 넣고 칸을 비운다. 순차 커널이다."""
    after = ip[Q_SPOIL_AFTER]
    rate = fp[P_SPOIL_RATE] * rate_mult
    floor = fp[P_SPOIL_MIN]
    for j in range(s.fx.shape[0]):
        if not s.f_alive[j] or step - s.f_born[j] < after:
            continue
        lost = s.f_energy[j] * rate
        s.f_energy[j] -= lost
        s.ledger[L_LOST] += lost
        if s.f_energy[j] < floor:
            s.ledger[L_LOST] += s.f_energy[j]
            _drop_food_slot(s, j)
            s.counters[C_SPOILED] += 1


# ---- 격자 ----
@njit(cache=True)
def build_grid(x, y, alive, inv_cell, gw, gh, start, items, cursor):
    ncell = gw * gh
    for c in range(ncell + 1):
        start[c] = 0
    n = x.shape[0]
    for i in range(n):
        if alive[i]:
            cx = int(x[i] * inv_cell)
            cy = int(y[i] * inv_cell)
            if cx >= gw:
                cx = gw - 1
            if cy >= gh:
                cy = gh - 1
            start[cy * gw + cx + 1] += 1
    for c in range(ncell):
        start[c + 1] += start[c]
    for c in range(ncell):
        cursor[c] = start[c]
    for i in range(n):
        if alive[i]:
            cx = int(x[i] * inv_cell)
            cy = int(y[i] * inv_cell)
            if cx >= gw:
                cx = gw - 1
            if cy >= gh:
                cy = gh - 1
            c = cy * gw + cx
            items[cursor[c]] = i
            cursor[c] += 1


# ---- 감각 ----
@njit(cache=True, inline="always")
def _see(obs, i, ch, nsec, ddx, ddy, d, rad, hi, fov, vrange, sgn):
    """물체 하나를 시야 칸에 반영한다. 칸마다 가장 가까운 것 하나만 남는다(가림).
    sgn이 -1이면 그 값을 음수로 적는다(먹이 종류 1). 가까움의 비교는 절댓값으로 한다."""
    ang = math.atan2(ddy, ddx) - hi
    ang = ang - 6.283185307179586 * math.floor((ang + 3.141592653589793) / 6.283185307179586)
    if d <= rad:
        delta = 3.141592653589793
    else:
        delta = math.asin(rad / d)
    half = 0.5 * fov
    lo = ang - delta
    hi_a = ang + delta
    if hi_a < -half or lo > half:
        return
    s_lo = int(math.floor((lo + half) / fov * nsec))
    s_hi = int(math.floor((hi_a + half) / fov * nsec))
    if s_lo < 0:
        s_lo = 0
    if s_hi > nsec - 1:
        s_hi = nsec - 1
    gap = d - rad
    if gap < 0.0:
        gap = 0.0
    c = np.float32(1.0 - gap / vrange)
    if c <= 0.0:
        return
    other = 1 - ch
    for sec in range(s_lo, s_hi + 1):
        cur = abs(obs[i, sec])
        if obs[i, nsec + sec] > cur:
            cur = obs[i, nsec + sec]
        if c > cur:
            obs[i, ch * nsec + sec] = c * sgn
            obs[i, other * nsec + sec] = 0.0


@njit(cache=True, parallel=True)
def _sense_par(step, fp, ip, alive, x, y, heading, vx, vy, energy, age, touch, ate, obs,
               fx, fy, f_alive, f_kind, a_start, a_items, f_start, f_items):
    # 주의: prange 안에서 상태 묶음을 거쳐 배열에 쓰면 numba가 그 쓰기를 조용히 버린다.
    # 병렬 커널은 반드시 배열을 직접 받는다.
    n = x.shape[0]
    n_in = obs.shape[1]
    seed = ip[Q_SEED]
    gw = ip[Q_GW]
    gh = ip[Q_GH]
    nsec = ip[Q_SECTORS]
    w = fp[P_W]
    hgt = fp[P_H]
    inv_cell = fp[P_INV_CELL]
    fov = fp[P_FOV]
    vrange = fp[P_VRANGE]
    vr2 = vrange * vrange
    a_rad = fp[P_A_RAD]
    f_rad = fp[P_F_RAD]
    noise = fp[P_VNOISE]
    sl2 = fp[P_SMELL_LEN] * fp[P_SMELL_LEN]
    reach = int(math.ceil(vrange * inv_cell))
    base = 2 * nsec
    for i in prange(n):
        if not alive[i]:
            continue
        for k in range(n_in):
            obs[i, k] = 0.0
        xi = x[i]
        yi = y[i]
        hi = heading[i]
        cx = int(xi * inv_cell)
        cy = int(yi * inv_cell)
        if cx >= gw:
            cx = gw - 1
        if cy >= gh:
            cy = gh - 1
        smell = 0.0
        for oy in range(-reach, reach + 1):
            yy = (cy + oy) % gh
            for ox in range(-reach, reach + 1):
                xx = (cx + ox) % gw
                c = yy * gw + xx
                for p in range(f_start[c], f_start[c + 1]):
                    j = f_items[p]
                    if not f_alive[j]:
                        continue
                    ddx = _wdiff(fx[j] - xi, w)
                    ddy = _wdiff(fy[j] - yi, hgt)
                    d2 = ddx * ddx + ddy * ddy
                    if d2 > vr2:
                        continue
                    smell += 1.0 / (1.0 + d2 / sl2)
                    _see(obs, i, 0, nsec, ddx, ddy, math.sqrt(d2), f_rad, hi, fov, vrange,
                         np.float32(-1.0) if f_kind[j] == KIND_B else np.float32(1.0))
                for p in range(a_start[c], a_start[c + 1]):
                    j = a_items[p]
                    if j == i:
                        continue
                    ddx = _wdiff(x[j] - xi, w)
                    ddy = _wdiff(y[j] - yi, hgt)
                    d2 = ddx * ddx + ddy * ddy
                    if d2 > vr2:
                        continue
                    _see(obs, i, 1, nsec, ddx, ddy, math.sqrt(d2), a_rad, hi, fov, vrange, np.float32(1.0))
        if noise > 0.0:
            for k in range(base):
                obs[i, k] += noise * normal(seed, S_NOISE, i * 64 + k, step)
        ch = math.cos(hi)
        sh = math.sin(hi)
        vmax = fp[P_VMAX]
        e = energy[i] / fp[P_E_REPRO]
        if e > 2.0:
            e = 2.0
        obs[i, base + 0] = e
        obs[i, base + 1] = age[i] / ip[Q_MAX_AGE]
        obs[i, base + 2] = (vx[i] * ch + vy[i] * sh) / vmax
        obs[i, base + 3] = (-vx[i] * sh + vy[i] * ch) / vmax
        obs[i, base + 4] = touch[i]
        obs[i, base + 5] = ate[i]
        obs[i, base + 6] = math.tanh(smell * fp[P_SMELL_GAIN])
        obs[i, base + 7] = 1.0


@njit(cache=True)
def sense(step, s, fp, ip):
    _sense_par(step, fp, ip, s.alive, s.x, s.y, s.heading, s.vx, s.vy, s.energy, s.age, s.touch,
               s.ate, s.obs, s.fx, s.fy, s.f_alive, s.f_kind, s.a_start, s.a_items, s.f_start, s.f_items)


# ---- 뇌 ----
@njit(cache=True, parallel=True)
def _think_par(alive, n_hidden, obs, h, hnew, act, Win, Wrec, bh, Wout, bout):
    n = alive.shape[0]
    n_in = obs.shape[1]
    n_out = act.shape[1]
    for i in prange(n):
        if not alive[i]:
            continue
        nh = n_hidden[i]
        for k in range(nh):
            hnew[i, k] = bh[i, k]
        for j in range(n_in):
            xj = obs[i, j]
            if xj != 0.0:
                for k in range(nh):
                    hnew[i, k] += Win[i, j, k] * xj
        for j in range(nh):
            hj = h[i, j]
            for k in range(nh):
                hnew[i, k] += Wrec[i, j, k] * hj
        for k in range(nh):
            h[i, k] = math.tanh(hnew[i, k])
        for o in range(n_out):
            acc = bout[i, o]
            for k in range(nh):
                acc += Wout[i, o, k] * h[i, k]
            act[i, o] = math.tanh(acc)


@njit(cache=True)
def write_ext2(alive, obs, col, a, b):
    """살아 있는 개체의 입력 두 칸(col, col+1)에 같은 값 a, b를 쓴다(계절 sin·cos 같은 세계 공통 감각)."""
    for i in range(alive.shape[0]):
        if alive[i]:
            obs[i, col] = a
            obs[i, col + 1] = b


@njit(cache=True)
def think(s):
    _think_par(s.alive, s.n_hidden, s.obs, s.h, s.hnew, s.act, s.Win, s.Wrec, s.bh, s.Wout, s.bout)


# ---- 움직임과 대사 ----
@njit(cache=True)
def move(s, fp, ip, basal_mult=1.0):
    n = s.x.shape[0]
    w = fp[P_W]
    hgt = fp[P_H]
    vmax = fp[P_VMAX]
    keep = 1.0 - fp[P_DRAG]
    h_max = ip[Q_H_MAX]
    n_io = s.obs.shape[1] + s.act.shape[1]
    syn_max = h_max * h_max + h_max * n_io
    for i in range(n):
        if not s.alive[i]:
            continue
        thrust = float(s.act[i, ACT_THRUST])
        turn = float(s.act[i, ACT_TURN])
        hd = s.heading[i] + turn * fp[P_MAX_TURN]
        hd = hd - 6.283185307179586 * math.floor(hd / 6.283185307179586)
        s.heading[i] = hd
        f = thrust * fp[P_THRUST]
        if thrust < 0.0:
            f *= fp[P_BACK]
        vx = (s.vx[i] + f * math.cos(hd)) * keep
        vy = (s.vy[i] + f * math.sin(hd)) * keep
        sp = math.sqrt(vx * vx + vy * vy)
        if sp > vmax:
            vx *= vmax / sp
            vy *= vmax / sp
        s.vx[i] = vx
        s.vy[i] = vy
        s.x[i] = _wrap(s.x[i] + vx, w)
        s.y[i] = _wrap(s.y[i] + vy, hgt)
        # 비용은 명령한 출력에 매긴다(환급 없음)
        nh = s.n_hidden[i]
        c_move = fp[P_THRUST_COST] * abs(thrust) + fp[P_TURN_COST] * abs(turn)
        if thrust < 0.0:
            c_move += fp[P_THRUST_COST] * abs(thrust) * (fp[P_BACK_COST] - 1.0)
        c_brain = fp[P_BRAIN] * nh / h_max + fp[P_SYN] * (nh * nh + nh * n_io) / syn_max
        if fp[P_BRAIN_ACT] > 0.0:
            a = 0.0
            for k in range(nh):
                a += abs(s.h[i, k])
            c_brain += fp[P_BRAIN_ACT] * a / h_max
        basal = fp[P_BASAL] * basal_mult
        s.energy[i] -= basal + c_brain + c_move
        s.ledger[L_BASAL] += basal
        s.ledger[L_BRAIN] += c_brain
        s.ledger[L_MOVE] += c_move
        s.age[i] += 1


@njit(cache=True, parallel=True)
def _collide_find(fp, ip, alive, x, y, dispx, dispy, touch, a_start, a_items):
    n = x.shape[0]
    gw = ip[Q_GW]
    gh = ip[Q_GH]
    w = fp[P_W]
    hgt = fp[P_H]
    inv_cell = fp[P_INV_CELL]
    rad = fp[P_A_RAD]
    dmin = 2.0 * rad
    dmin2 = dmin * dmin
    max_push = fp[P_PUSH]
    for i in prange(n):
        dispx[i] = 0.0
        dispy[i] = 0.0
        touch[i] = 0.0
        if not alive[i]:
            continue
        xi = x[i]
        yi = y[i]
        cx = int(xi * inv_cell)
        cy = int(yi * inv_cell)
        if cx >= gw:
            cx = gw - 1
        if cy >= gh:
            cy = gh - 1
        px = 0.0
        py = 0.0
        hit = False
        for oy in range(-1, 2):
            yy = (cy + oy) % gh
            for ox in range(-1, 2):
                xx = (cx + ox) % gw
                c = yy * gw + xx
                for p in range(a_start[c], a_start[c + 1]):
                    j = a_items[p]
                    if j == i:
                        continue
                    ddx = _wdiff(x[j] - xi, w)
                    ddy = _wdiff(y[j] - yi, hgt)
                    d2 = ddx * ddx + ddy * ddy
                    if d2 >= dmin2:
                        continue
                    hit = True
                    d = math.sqrt(d2)
                    pen = 0.5 * (dmin - d)
                    if d > 1e-9:
                        px -= ddx / d * pen
                        py -= ddy / d * pen
                    elif i < j:
                        px -= pen
                    else:
                        px += pen
        if hit:
            m = math.sqrt(px * px + py * py)
            if m > max_push:
                px *= max_push / m
                py *= max_push / m
            dispx[i] = px
            dispy[i] = py
            touch[i] = 1.0


@njit(cache=True)
def collide(s, fp, ip):
    """겹친 원을 위치만 밀어낸다. 속도·에너지는 건드리지 않는다."""
    _collide_find(fp, ip, s.alive, s.x, s.y, s.dispx, s.dispy, s.touch, s.a_start, s.a_items)
    n = s.x.shape[0]
    for i in range(n):
        if s.touch[i] > 0.0:
            s.x[i] = _wrap(s.x[i] + s.dispx[i], fp[P_W])
            s.y[i] = _wrap(s.y[i] + s.dispy[i], fp[P_H])


# ---- 먹기 ----
@njit(cache=True)
def eat(step, s, fp, ip):
    n = s.x.shape[0]
    kinds = ip[Q_KINDS] != 0
    good = 0
    if kinds:
        good = good_kind(ip[Q_SEED], step, ip[Q_KIND_PERIOD])
    gw = ip[Q_GW]
    gh = ip[Q_GH]
    w = fp[P_W]
    hgt = fp[P_H]
    inv_cell = fp[P_INV_CELL]
    r = fp[P_A_RAD] + fp[P_F_RAD]
    r2 = r * r
    e_max = fp[P_E_MAX]
    for i in range(n):
        s.ate[i] = 0.0
        if not s.alive[i] or s.energy[i] >= e_max:
            continue
        xi = s.x[i]
        yi = s.y[i]
        cx = int(xi * inv_cell)
        cy = int(yi * inv_cell)
        if cx >= gw:
            cx = gw - 1
        if cy >= gh:
            cy = gh - 1
        full = False
        for oy in range(-1, 2):
            if full:
                break
            yy = (cy + oy) % gh
            for ox in range(-1, 2):
                if full:
                    break
                xx = (cx + ox) % gw
                c = yy * gw + xx
                for p in range(s.f_start[c], s.f_start[c + 1]):
                    j = s.f_items[p]
                    if not s.f_alive[j]:
                        continue
                    ddx = _wdiff(s.fx[j] - xi, w)
                    ddy = _wdiff(s.fy[j] - yi, hgt)
                    if ddx * ddx + ddy * ddy > r2:
                        continue
                    s.f_alive[j] = False
                    s.counters[C_NFOOD] -= 1
                    s.counters[C_EATEN] += 1
                    if kinds and s.f_kind[j] != KIND_CORPSE and s.f_kind[j] != good:
                        # 독: 먹이의 에너지는 사라지고 먹은 쪽도 에너지를 잃는다
                        s.energy[i] -= fp[P_POISON]
                        s.ledger[L_LOST] += s.f_energy[j] + fp[P_POISON]
                        s.counters[C_POISONED] += 1
                        s.ate[i] = 1.0
                        continue
                    s.energy[i] += s.f_energy[j]
                    s.eaten[i] += s.f_energy[j]
                    s.ate[i] = 1.0
                    if s.energy[i] >= e_max:
                        full = True
                        break


# ---- 유전체 ----
@njit(cache=True)
def _reset_body(s, j, aid, parent, gen, tag, step, px, py, hd, e):
    s.alive[j] = True
    s.ids[j] = aid
    s.parent[j] = parent
    s.gen[j] = gen
    s.tag[j] = tag
    s.birth_step[j] = step
    s.x[j] = px
    s.y[j] = py
    s.vx[j] = 0.0
    s.vy[j] = 0.0
    s.heading[j] = hd
    s.energy[j] = e
    s.age[j] = 0
    s.n_off[j] = 0
    s.eaten[j] = 0.0
    s.touch[j] = 0.0
    s.ate[j] = 0.0
    for k in range(s.h.shape[1]):
        s.h[j, k] = 0.0
    # 평생 상태는 물려받지 않는다
    for k in range(4):
        s.horm[j, k] = 0.0
    s.mod[j] = 0.0
    s.exp_r[j] = 0.0
    s.e_prev[j] = e
    s.fatigue[j] = 0.0
    s.asleep[j] = 0.0
    s.last_proto[j] = 0
    s.sig[j, 0] = 0.0
    s.sig[j, 1] = 0.0
    for k in range(s.epost.shape[1]):
        s.epost[j, k] = 0.0
        s.epre_h[j, k] = 0.0
    for a in range(s.epre_in.shape[1]):
        s.epre_in[j, a] = 0.0
        for k in range(s.Hin.shape[2]):
            s.Hin[j, a, k] = 0.0
    for a in range(s.Hrec.shape[1]):
        for k in range(s.Hrec.shape[2]):
            s.Hrec[j, a, k] = 0.0
    for a in range(s.proto.shape[1]):
        for k in range(s.proto.shape[2]):
            s.proto[j, a, k] = 0.0
        for k in range(s.link.shape[2]):
            s.link[j, a, k] = 0.0
    for a in range(s.elig_in.shape[1]):
        for k in range(s.elig_in.shape[2]):
            s.elig_in[j, a, k] = 0.0
    for a in range(s.elig_rec.shape[1]):
        for k in range(s.elig_rec.shape[2]):
            s.elig_rec[j, a, k] = 0.0
    s.counters[C_NALIVE] += 1


@njit(cache=True)
def random_genome(s, j, aid, fp, ip):
    seed = ip[Q_SEED]
    n_in = s.Win.shape[1]
    hm = s.Win.shape[2]
    n_out = s.Wout.shape[1]
    q = 0
    for a in range(n_in):
        for k in range(hm):
            s.Win[j, a, k] = fp[P_INIT_WIN] * normal(seed, S_INIT_GENOME, aid, q)
            q += 1
    for a in range(hm):
        for k in range(hm):
            s.Wrec[j, a, k] = fp[P_INIT_WREC] * normal(seed, S_INIT_GENOME, aid, q)
            q += 1
    for o in range(n_out):
        for k in range(hm):
            s.Wout[j, o, k] = fp[P_INIT_WOUT] * normal(seed, S_INIT_GENOME, aid, q)
            q += 1
    for k in range(hm):
        s.bh[j, k] = 0.0
    for o in range(n_out):
        s.bout[j, o] = 0.0
    s.bout[j, ACT_THRUST] = fp[P_BIAS_THRUST]
    s.bout[j, ACT_REPRO] = fp[P_BIAS_REPRO]
    s.bout[j, ACT_GIVE] = -1.0       # 주기·쌓기·공격은 처음에는 꺼진 쪽으로 기울여 둔다
    s.bout[j, ACT_ATTACK] = -1.0
    s.bout[j, ACT_DEPOSIT] = -1.0
    if n_out > ACT_CARRY:            # 기본 크기(9칸)에서는 쓰지 않는다: numba는 범위를 검사하지 않아 다음 개체를 덮는다
        s.bout[j, ACT_CARRY] = -1.0
    s.n_hidden[j] = ip[Q_H_INIT]
    s.sigma[j] = fp[P_SIGMA_INIT]
    for c in range(3):
        s.color[j, c] = u01(seed, S_INIT_GENOME, aid, q)
        q += 1
    # 2단계 유전자. 타고난 보상: 먹으면 좋고, 에너지가 늘면 좋고, 새 연결이 생기면 조금 좋다
    for f in range(N_REWARD):
        s.rw[j, f] = 0.1 * normal(seed, S_INIT_GENOME, aid, q)
        q += 1
    s.rw[j, 0] += 1.0
    s.rw[j, 1] += 0.5
    s.rw[j, 3] += 0.2
    s.rw[j, 4] += 0.5
    s.rw[j, 5] -= 0.2
    s.rw[j, 6] += 0.2
    s.eta[j] = fp[P_ETA_INIT]
    for k in range(hm):
        s.alpha[j, k] = fp[P_ALPHA_INIT] * normal(seed, S_INIT_GENOME, aid, q)
        q += 1


@njit(cache=True)
def copy_genome(s, i, j, cid, fp, ip):
    """부모 칸 i의 유전체를 자식 칸 j로 복사하면서 변이시킨다."""
    seed = ip[Q_SEED]
    mut = ip[Q_MUT] != 0
    n_in = s.Win.shape[1]
    hm = s.Win.shape[2]
    n_out = s.Wout.shape[1]
    sig = float(s.sigma[i])
    nh_p = s.n_hidden[i]
    nh = nh_p
    if mut:
        sig = sig * math.exp(fp[P_SIGMA_TAU] * normal(seed, S_MUT_GENE, cid, 0))
        if sig < fp[P_SIGMA_MIN]:
            sig = fp[P_SIGMA_MIN]
        if sig > fp[P_SIGMA_MAX]:
            sig = fp[P_SIGMA_MAX]
        if u01(seed, S_MUT_GENE, cid, 1) < fp[P_P_HIDDEN]:
            if u01(seed, S_MUT_GENE, cid, 2) < 0.5:
                nh += 1
            else:
                nh -= 1
            if nh < ip[Q_H_MIN]:
                nh = ip[Q_H_MIN]
            if nh > hm:
                nh = hm
    s.sigma[j] = sig
    s.n_hidden[j] = nh
    amp = sig if mut else 0.0
    q = 0
    for a in range(n_in):
        for k in range(hm):
            v = s.Win[i, a, k]
            if mut:
                v += amp * normal(seed, S_MUT_W, cid, q)
            s.Win[j, a, k] = v
            q += 1
    for a in range(hm):
        for k in range(hm):
            v = s.Wrec[i, a, k]
            if mut:
                v += amp * normal(seed, S_MUT_W, cid, q)
            s.Wrec[j, a, k] = v
            q += 1
    for o in range(n_out):
        for k in range(hm):
            v = s.Wout[i, o, k]
            if mut:
                v += amp * normal(seed, S_MUT_W, cid, q)
            s.Wout[j, o, k] = v
            q += 1
    for k in range(hm):
        v = s.bh[i, k]
        if mut:
            v += amp * normal(seed, S_MUT_W, cid, q)
        s.bh[j, k] = v
        q += 1
    for o in range(n_out):
        v = s.bout[i, o]
        if mut:
            v += amp * normal(seed, S_MUT_W, cid, q)
        s.bout[j, o] = v
        q += 1
    for c in range(3):
        v = s.color[i, c]
        if mut:
            v += fp[P_COLOR_SIGMA] * normal(seed, S_MUT_W, cid, q)
            if v < 0.0:
                v = 0.0
            if v > 1.0:
                v = 1.0
        s.color[j, c] = v
        q += 1
    # 2단계 유전자(뒤에 덧붙여 앞쪽 난수 번호가 바뀌지 않게 한다)
    for f in range(N_REWARD):
        v = s.rw[i, f]
        if mut:
            v += amp * normal(seed, S_MUT_W, cid, q)
        s.rw[j, f] = v
        q += 1
    v = float(s.eta[i])
    if mut:
        v = v * math.exp(0.1 * normal(seed, S_MUT_W, cid, q))
        if v > fp[P_ETA_MAX]:
            v = fp[P_ETA_MAX]
    s.eta[j] = v
    q += 1
    for k in range(hm):
        v = s.alpha[i, k]
        if mut:
            v += amp * normal(seed, S_MUT_W, cid, q)
        s.alpha[j, k] = v
        q += 1
    if nh > nh_p:
        # 새로 켜진 뉴런은 처음에는 아무 데도 영향을 주지 않는다(중립)
        k = nh - 1
        for t in range(hm):
            s.Wrec[j, k, t] = 0.0
        for o in range(n_out):
            s.Wout[j, o, k] = 0.0


# ---- 죽음 ----
@njit(cache=True)
def _remove(s, i, cause, step, recycle, nd):
    s.alive[i] = False
    s.counters[C_NALIVE] -= 1
    e = s.energy[i]
    if recycle and e > 0.0 and _place_food(s, s.x[i], s.y[i], e, KIND_CORPSE, step):
        pass
    else:
        s.ledger[L_LOST] += e
    s.energy[i] = 0.0
    s.dbuf[nd, 0] = s.ids[i]
    s.dbuf[nd, 1] = step
    s.dbuf[nd, 2] = cause
    s.dbuf[nd, 3] = s.age[i]
    s.dbuf[nd, 4] = s.n_off[i]
    s.dbuf[nd, 5] = s.gen[i]
    s.dbuf[nd, 6] = s.n_hidden[i]
    s.dbuf_f[nd] = s.eaten[i]


@njit(cache=True)
def process_deaths(step, s, fp, ip):
    n = s.x.shape[0]
    nd = 0
    passive = ip[Q_PASSIVE] != 0
    corpse = ip[Q_CORPSE] != 0
    for i in range(n):
        if not s.alive[i]:
            continue
        if passive:
            # 대조 실행: 자연사는 없다. 굶으면 에너지를 채워 준다(장부에 기록)
            if s.energy[i] <= 0.0:
                add = fp[P_E_START] - s.energy[i]
                s.energy[i] += add
                s.ledger[L_INJECT] += add
            continue
        if s.energy[i] <= 0.0:
            if s.touch[i] == TOUCH_HIT:
                # 이번 스텝에 공격을 맞았고 에너지가 바닥났다. 굶어 죽음과 따로 센다
                # (맞은 뒤 제 공격 비용으로 바닥난 드문 경우도 여기에 든다)
                _remove(s, i, DEATH_KILLED, step, False, nd)
                s.counters[C_DEATH_KILLED] += 1
            else:
                _remove(s, i, DEATH_STARVE, step, False, nd)
                s.counters[C_DEATH_STARVE] += 1
            nd += 1
        elif s.age[i] >= ip[Q_MAX_AGE]:
            _remove(s, i, DEATH_AGE, step, corpse, nd)
            s.counters[C_DEATH_AGE] += 1
            nd += 1
    return nd


# ---- 출생 ----
@njit(cache=True)
def _log_birth(s, nb, cid, parent, step, gen, nh, kind):
    s.bbuf[nb, 0] = cid
    s.bbuf[nb, 1] = parent
    s.bbuf[nb, 2] = step
    s.bbuf[nb, 3] = gen
    s.bbuf[nb, 4] = nh
    s.bbuf[nb, 5] = kind


@njit(cache=True)
def _child(s, i, j, step, fp, ip, e_child, kind, nb):
    seed = ip[Q_SEED]
    cid = s.counters[C_NEXT_ID]
    s.counters[C_NEXT_ID] += 1
    ang = 6.283185307179586 * u01(seed, S_BIRTH_POS, cid, 0)
    dist = 2.0 * fp[P_A_RAD]
    px = _wrap(s.x[i] + dist * math.cos(ang), fp[P_W])
    py = _wrap(s.y[i] + dist * math.sin(ang), fp[P_H])
    hd = 6.283185307179586 * u01(seed, S_BIRTH_POS, cid, 1)
    _reset_body(s, j, cid, s.ids[i], s.gen[i] + 1, s.tag[i], step, px, py, hd, e_child)
    copy_genome(s, i, j, cid, fp, ip)
    s.n_off[i] += 1
    s.counters[C_BIRTHS] += 1
    _log_birth(s, nb, cid, s.ids[i], step, s.gen[j], s.n_hidden[j], kind)


@njit(cache=True)
def process_births(step, s, fp, ip):
    n = s.x.shape[0]
    nb = 0
    if ip[Q_PASSIVE] != 0:
        return 0
    brain = ip[Q_REPRO_BRAIN] != 0
    free = 0
    # 훑는 시작 칸을 스텝마다 바꾼다. 늘 0번 칸부터 훑으면 빈 칸이 모자랄 때(상한 포화)
    # 낮은 칸의 부모만 번식해서 칸 번호가 혈통처럼 이어진다(독립 검토 2026-11-04).
    start = int(u01(ip[Q_SEED], S_BIRTH_ORDER, step, 0) * n)
    for q in range(n):
        i = start + q
        if i >= n:
            i -= n
        if not s.alive[i] or s.birth_step[i] == step:
            continue
        if s.energy[i] < fp[P_E_REPRO] or s.age[i] < ip[Q_REPRO_MIN_AGE]:
            continue
        if brain and s.act[i, ACT_REPRO] <= 0.0:
            continue
        while free < n and s.alive[free]:
            free += 1
        if free >= n:
            s.counters[C_BLOCKED] += 1
            continue
        e_child = s.energy[i] * fp[P_CHILD_FRAC]
        cost = fp[P_BIRTH_COST]
        rest = s.energy[i] - e_child
        if cost > rest:
            cost = rest
        s.energy[i] = rest - cost
        s.ledger[L_BIRTH] += cost
        _child(s, i, free, step, fp, ip, e_child, BIRTH_NATURAL, nb)
        nb += 1
    return nb


@njit(cache=True)
def _nth_alive(s, r):
    n = s.x.shape[0]
    c = 0
    for i in range(n):
        if s.alive[i]:
            if c == r:
                return i
            c += 1
    return -1


@njit(cache=True)
def add_random_agent(s, step, fp, ip, kind, tag, stream, nb):
    """무작위 유전체 개체 하나를 빈 칸에 넣는다. 넣었으면 True."""
    n = s.x.shape[0]
    seed = ip[Q_SEED]
    j = -1
    for t in range(n):
        if not s.alive[t]:
            j = t
            break
    if j < 0:
        return False
    aid = s.counters[C_NEXT_ID]
    s.counters[C_NEXT_ID] += 1
    px = u01(seed, stream, aid, 0) * fp[P_W]
    py = u01(seed, stream, aid, 1) * fp[P_H]
    hd = 6.283185307179586 * u01(seed, stream, aid, 2)
    _reset_body(s, j, aid, -1, 0, tag, step, px, py, hd, fp[P_E_START])
    s.ledger[L_INJECT] += fp[P_E_START]
    random_genome(s, j, aid, fp, ip)
    _log_birth(s, nb, aid, -1, step, 0, s.n_hidden[j], kind)
    return True


@njit(cache=True)
def populate_initial(s, n_init, max_age, fp, ip):
    seed = ip[Q_SEED]
    nb = 0
    for _ in range(n_init):
        if add_random_agent(s, 0, fp, ip, BIRTH_INITIAL, 0, S_INIT_POS, nb):
            nb += 1
    # 처음 무리가 한꺼번에 늙어 죽지 않게 나이를 흩어 놓는다
    n = s.x.shape[0]
    for j in range(n):
        if s.alive[j]:
            s.age[j] = int(u01(seed, S_INIT_AGE, s.ids[j], 0) * 0.5 * max_age)
    return nb


@njit(cache=True)
def reseed(step, s, fp, ip, nb):
    """개체 수가 하한 아래면 채운다. 살아 있는 개체가 있으면 그중 무작위 하나의 변이 복제본을,
    아무도 없으면 무작위 유전체를 넣는다. 넣은 수를 돌려준다."""
    added = 0
    if ip[Q_PASSIVE] != 0:
        return 0
    n = s.x.shape[0]
    seed = ip[Q_SEED]
    while s.counters[C_NALIVE] < ip[Q_MIN_AGENTS]:
        na = s.counters[C_NALIVE]
        if na > 0:
            i = _nth_alive(s, int(u01(seed, S_RESEED, step, added) * na))
            j = -1
            for t in range(n):
                if not s.alive[t]:
                    j = t
                    break
            if i < 0 or j < 0:
                break
            s.ledger[L_INJECT] += fp[P_E_START]
            _child(s, i, j, step, fp, ip, fp[P_E_START], BIRTH_RESEED, nb + added)
            s.counters[C_BIRTHS] -= 1
            s.n_off[i] -= 1
        elif not add_random_agent(s, step, fp, ip, BIRTH_RESEED, 0, S_RESEED, nb + added):
            break
        s.counters[C_RESEED] += 1
        added += 1
    return added


# ---- 대조 실행 ----
@njit(cache=True)
def passive_events(step, s, fp, ip, n_births, n_deaths):
    """기록된 출생·사망 횟수를 무작위 개체에 적용한다(선택 없는 대조)."""
    seed = ip[Q_SEED]
    n = s.x.shape[0]
    nd = 0
    nb = 0
    for k in range(n_deaths):
        na = s.counters[C_NALIVE]
        if na <= 1:
            break
        i = _nth_alive(s, int(u01(seed, S_PASSIVE, step, 2 * k) * na))
        if i < 0:
            break
        _remove(s, i, DEATH_PASSIVE, step, False, nd)
        s.counters[C_DEATH_PASSIVE] += 1
        nd += 1
    for k in range(n_births):
        na = s.counters[C_NALIVE]
        if na <= 0 or na >= n:
            break
        i = _nth_alive(s, int(u01(seed, S_PASSIVE, step, 2 * k + 1) * na))
        j = -1
        for t in range(n):
            if not s.alive[t]:
                j = t
                break
        if i < 0 or j < 0:
            break
        s.ledger[L_INJECT] += fp[P_E_START]
        _child(s, i, j, step, fp, ip, fp[P_E_START], BIRTH_PASSIVE, nb)
        nb += 1
    return nb, nd


# ---- 점검용 ----
@njit(cache=True)
def totals(s):
    """(살아 있는 수, 개체 에너지 합, 먹이 수, 먹이 에너지 합)"""
    na = 0
    ea = 0.0
    for i in range(s.x.shape[0]):
        if s.alive[i]:
            na += 1
            ea += s.energy[i]
    nf = 0
    ef = 0.0
    for j in range(s.fx.shape[0]):
        if s.f_alive[j]:
            nf += 1
            ef += s.f_energy[j]
    return na, ea, nf, ef


# ---- 재난 (각본) ----
# 이 아래는 각본(scenario.py)이 쓰는 커널이다. 파일 위쪽의 import 줄을 건드리지 않으려고 여기서 불러온다.
from .rng import S_SCENARIO  # noqa: E402


@njit(cache=True)
def strike(step, s, fp, ip, x, y, radius, kill_prob, burn_food, stream_seed, nd):
    """운석: (x, y)를 중심으로 반지름 radius인 원 안의 개체를 kill_prob 확률로 죽이고,
    burn_food면 원 안의 먹이도 없앤다. 세계가 반대편으로 이어지므로 거리는 가까운 쪽으로 잰다.

    사라진 에너지(개체의 남은 에너지, 먹이의 에너지)는 모두 장부의 L_LOST에 적는다.
    사망 기록은 dbuf의 nd번 줄부터 쓰고, 죽은 수를 돌려준다.
    누가 죽는지는 (씨앗, 개체 id, stream_seed)만으로 정해진다(칸 번호·스레드 수와 무관).
    prange를 쓰지 않는 순차 커널이다(상태 묶음을 거쳐 배열에 쓴다).
    """
    seed = ip[Q_SEED]
    w = fp[P_W]
    hgt = fp[P_H]
    r2 = radius * radius
    killed = 0
    for i in range(s.x.shape[0]):
        if not s.alive[i]:
            continue
        ddx = _wdiff(s.x[i] - x, w)
        ddy = _wdiff(s.y[i] - y, hgt)
        if ddx * ddx + ddy * ddy > r2:
            continue
        if kill_prob < 1.0 and u01(seed, S_SCENARIO, s.ids[i], stream_seed) >= kill_prob:
            continue
        _remove(s, i, DEATH_DISASTER, step, False, nd + killed)
        s.counters[C_DEATH_DISASTER] += 1
        killed += 1
    if burn_food:
        for j in range(s.fx.shape[0]):
            if not s.f_alive[j]:
                continue
            ddx = _wdiff(s.fx[j] - x, w)
            ddy = _wdiff(s.fy[j] - y, hgt)
            if ddx * ddx + ddy * ddy > r2:
                continue
            s.f_alive[j] = False
            s.counters[C_NFOOD] -= 1
            s.ledger[L_LOST] += s.f_energy[j]
    return killed
