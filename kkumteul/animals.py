"""동물(큰 사냥감 등). 개체·먹이와 따로 고정 크기 배열(an_*)에 두고 순차 커널로 다룬다.

사냥감은 유입의 한 몫(budget[2])에서 태어나고, 다 차 있으면 넘친 몫은 풀 배수를 곱해 풀 몫(budget[0])으로 간다.
먹이 자리(목초지) 둘레를 천천히 맴돌다 다른 먹이 자리로 옮겨 간다. 개체와 부딪치지 않는다.
개체가 치면 다치고(한 스텝 피해는 공격자 prey_sat_n 몫에서 포화) 친 개체들에게 반격을 나눠 주며, 쉬면 아물고, 쓰러지면 고기가 된다.
난수 주소는 (칸, 스텝)으로만 정해 다른 동물·개체의 상태에 기대지 않는다.
"""
import math

import numpy as np
from numba import njit

from . import kernels as k
from .config import ACT_ATTACK
from .rng import S_ANIMAL_POS, S_MEAT, S_WANDER, normal, u01

# ---- 동물 종류 (an_kind) ----
ANIMAL_PREY = 0
ANIMAL_PRED = 1
ANIMAL_SCAV = 2

# ---- 실수 설정 칸 ----
(AF_W, AF_H, AF_INFLOW, AF_PREY_FRAC, AF_MEAT_E, AF_HP, AF_REGEN, AF_STRIKE, AF_RADIUS, AF_REACH,
 AF_GRAZE_V, AF_TRAVEL_V, AF_GRAZE_R, AF_ATK_DMG, AF_ATK_COST, AF_WANDER_TURN, N_FA) = range(17)

# ---- 정수 설정 칸 ----
(AI_SEED, AI_N_PREY, AI_REST, AI_SAT_N, AI_MEAT, AI_LIFE, AI_STAY, AI_CONFLICT, AI_N_PATCH,
 AI_PATCH_LIFE, N_IA) = range(11)


def pack_animal_params(cfg):
    fa = np.zeros(N_FA, dtype=np.float64)
    ia = np.zeros(N_IA, dtype=np.int64)
    fa[AF_W] = cfg.world_w
    fa[AF_H] = cfg.world_h
    fa[AF_INFLOW] = cfg.food_inflow
    fa[AF_PREY_FRAC] = cfg.prey_inflow_frac
    fa[AF_MEAT_E] = cfg.prey_meat * cfg.food_energy
    fa[AF_HP] = cfg.prey_hp
    fa[AF_REGEN] = cfg.prey_regen
    fa[AF_STRIKE] = cfg.prey_strike
    fa[AF_RADIUS] = cfg.prey_radius
    fa[AF_REACH] = cfg.prey_radius + cfg.agent_radius + 0.5     # 공격이 닿는 중심 거리(건넴·공격의 닿음 여유 0.5)
    fa[AF_GRAZE_V] = cfg.prey_graze_speed * cfg.v_max
    fa[AF_TRAVEL_V] = cfg.prey_travel_speed * cfg.v_max
    fa[AF_GRAZE_R] = cfg.prey_graze_r
    fa[AF_ATK_DMG] = cfg.attack_damage
    fa[AF_ATK_COST] = cfg.attack_cost
    fa[AF_WANDER_TURN] = cfg.prey_wander_turn
    ia[AI_SEED] = cfg.seed
    ia[AI_N_PREY] = cfg.n_prey
    ia[AI_REST] = cfg.prey_rest
    ia[AI_SAT_N] = cfg.prey_sat_n
    ia[AI_MEAT] = cfg.prey_meat
    ia[AI_LIFE] = cfg.prey_life
    ia[AI_STAY] = cfg.prey_stay
    ia[AI_CONFLICT] = int(cfg.conflict_enabled)
    ia[AI_N_PATCH] = cfg.n_patches
    ia[AI_PATCH_LIFE] = cfg.patch_life
    return fa, ia


# ---- 유입 나누기와 출생 ----
@njit(cache=True)
def prey_inflow(s, fa, ia, mult):
    """유입의 사냥감 몫을 budget[2]에 쌓는다. 한 마리 몫(meat_e)을 넘친 만큼은 풀 배수 mult를 곱해
    풀 몫(budget[0])으로 보낸다(사냥이 없으면 해 전체 유입이 사냥감 없는 세계와 같게). 순차 커널이다."""
    meat_e = fa[AF_MEAT_E]
    s.budget[2] += fa[AF_INFLOW] * fa[AF_PREY_FRAC]
    if s.budget[2] > meat_e:
        s.budget[0] += (s.budget[2] - meat_e) * mult
        s.budget[2] = meat_e


@njit(cache=True)
def _pick_target(seed, stream, a, b0, x, y, step, fa, ia):
    """다음 목적지. 난수로 먹이 자리 번호 k를 하나 뽑고(b0), 그 중심이 (x, y)에서 2 × graze_r 안이면
    k+1, k+2, …(번호를 돌며) 처음으로 먼 것을 쓴다. 모두 가까우면 k를 쓴다.
    먹이 자리가 없으면 세계 안 균등한 점(b0+1, b0+2)이다. 재시도가 없어 쓰는 난수 주소가 정해져 있다."""
    w = fa[AF_W]
    h = fa[AF_H]
    n = ia[AI_N_PATCH]
    if n == 0:
        return u01(seed, stream, a, b0 + 1) * w, u01(seed, stream, a, b0 + 2) * h
    k0 = int(u01(seed, stream, a, b0) * n)
    near = 2.0 * fa[AF_GRAZE_R]
    for i in range(n):
        kk = (k0 + i) % n
        cx, cy = k.patch_center(seed, kk, step, n, ia[AI_PATCH_LIFE], w, h)
        if math.hypot(k._wdiff(cx - x, w), k._wdiff(cy - y, h)) > near:
            return cx, cy
    return k.patch_center(seed, k0, step, n, ia[AI_PATCH_LIFE], w, h)


@njit(cache=True)
def spawn_prey(step, s, fa, ia):
    """산 사냥감이 n_prey보다 적고 사냥감 몫이 한 마리 몫 이상이고 빈 칸이 있으면 첫 빈 칸에 하나 낳는다.
    몫은 사냥감 몸(an_energy)으로 옮기고 장부의 L_SPAWN에 적는다. 낳은 수(0 또는 1)를 돌려준다."""
    meat_e = fa[AF_MEAT_E]
    if s.budget[2] < meat_e:
        return 0
    n_alive = 0
    slot = -1
    for j in range(s.an_alive.shape[0]):
        if s.an_alive[j]:
            if s.an_kind[j] == ANIMAL_PREY:
                n_alive += 1
        elif slot < 0:
            slot = j
    if n_alive >= ia[AI_N_PREY] or slot < 0:
        return 0
    seed = ia[AI_SEED]
    x = u01(seed, S_ANIMAL_POS, step, 5 * slot) * fa[AF_W]
    y = u01(seed, S_ANIMAL_POS, step, 5 * slot + 1) * fa[AF_H]
    tx, ty = _pick_target(seed, S_ANIMAL_POS, step, 5 * slot + 2, x, y, step, fa, ia)
    s.an_alive[slot] = True
    s.an_kind[slot] = ANIMAL_PREY
    s.an_x[slot] = x
    s.an_y[slot] = y
    s.an_heading[slot] = math.atan2(k._wdiff(ty - y, fa[AF_H]), k._wdiff(tx - x, fa[AF_W]))
    s.an_hp[slot] = fa[AF_HP]
    s.an_energy[slot] = meat_e
    s.an_tx[slot] = tx
    s.an_ty[slot] = ty
    s.an_timer[slot] = 0
    s.an_rest[slot] = ia[AI_REST]
    s.an_born[slot] = step
    s.an_natt[slot] = 0
    s.an_dmg[slot] = 0.0
    s.budget[2] -= meat_e
    s.ledger[k.L_SPAWN] += meat_e
    return 1


# ---- 떠돌기·수명 ----
@njit(cache=True)
def move_animals(step, s, fa, ia):
    """사냥감마다: 수명이 다했으면 없앤다(에너지는 장부의 L_LOST, 고기는 남기지 않는다).
    아니면 떠돈다. 맴도는 중(an_timer > 0)에는 방향을 흔들며 graze 속도로 움직이고, 목적지까지 감긴 거리가
    graze_r보다 크면 그 스텝의 방향은 목적지 쪽(+ 흔들기)이다. 다 맴돌면 다른 먹이 자리를 고른다.
    가는 중(an_timer == 0)에는 목적지 쪽으로 travel 속도로 가고, graze_r 안에 들면 prey_stay스텝 맴돌기 시작한다.
    난수 주소: S_WANDER, a = 칸, 흔들기 b = 4 × step, 새 목적지 b0 = 4 × step + 1. 순차 커널이다."""
    seed = ia[AI_SEED]
    w = fa[AF_W]
    h = fa[AF_H]
    graze_r = fa[AF_GRAZE_R]
    turn = fa[AF_WANDER_TURN]
    for j in range(s.an_alive.shape[0]):
        if not s.an_alive[j] or s.an_kind[j] != ANIMAL_PREY:
            continue
        if step - s.an_born[j] >= ia[AI_LIFE]:
            s.ledger[k.L_LOST] += s.an_energy[j]
            s.an_energy[j] = 0.0
            s.an_alive[j] = False
            continue
        x = s.an_x[j]
        y = s.an_y[j]
        dx = k._wdiff(s.an_tx[j] - x, w)
        dy = k._wdiff(s.an_ty[j] - y, h)
        d = math.hypot(dx, dy)
        if s.an_timer[j] > 0:
            jitter = (2.0 * u01(seed, S_WANDER, j, 4 * step) - 1.0) * turn
            if d > graze_r:
                hd = math.atan2(dy, dx) + jitter     # 가두기: 맴도는 자리를 벗어났으면 목적지 쪽으로
            else:
                hd = s.an_heading[j] + jitter
            v = fa[AF_GRAZE_V]
            nx = x + v * math.cos(hd)
            ny = y + v * math.sin(hd)
            s.an_timer[j] -= 1
            if s.an_timer[j] == 0:
                tx, ty = _pick_target(seed, S_WANDER, j, 4 * step + 1, nx, ny, step, fa, ia)
                s.an_tx[j] = tx
                s.an_ty[j] = ty
        else:
            hd = math.atan2(dy, dx)
            v = fa[AF_TRAVEL_V]
            if d <= v:
                nx = x + dx
                ny = y + dy
            else:
                nx = x + v * math.cos(hd)
                ny = y + v * math.sin(hd)
            if math.hypot(k._wdiff(s.an_tx[j] - nx, w), k._wdiff(s.an_ty[j] - ny, h)) <= graze_r:
                s.an_timer[j] = ia[AI_STAY]
        s.an_heading[j] = math.atan2(math.sin(hd), math.cos(hd))     # -pi~pi로 감는다
        s.an_x[j] = k._wrap(nx, w)
        s.an_y[j] = k._wrap(ny, h)


# ---- 감각 ----
@njit(cache=True)
def sense_animals(alive, x, y, heading, obs, col, an_alive, an_kind, an_x, an_y, kind, radius, vrange, fov, w, h):
    """산 개체마다 kind 종류의 산 동물 가운데 보이는 가장 가까운 것 하나의 방향을 입력 두 칸(col, col+1)에
    가까움 × (cos, sin)으로 쓴다(각도는 머리 기준 -pi~pi). 없으면 0, 0이다.
    보인다 = 틈(감긴 거리 - radius)이 vrange보다 작고, 몸 한쪽 끝이 시야각 안에 든다
    (|각도| - asin(radius / 거리) <= fov / 2, 거리가 radius 이하면 늘 보인다).
    가까움 = 1 - 틈 / vrange(겹치면 틈 0으로 1). 순차 커널이다."""
    half = 0.5 * fov
    for i in range(alive.shape[0]):
        if not alive[i]:
            continue
        best_d = -1.0
        best_a = 0.0
        for j in range(an_alive.shape[0]):
            if not an_alive[j] or an_kind[j] != kind:
                continue
            dx = k._wdiff(an_x[j] - x[i], w)
            dy = k._wdiff(an_y[j] - y[i], h)
            d = math.hypot(dx, dy)
            if d - radius >= vrange:
                continue
            ang = math.atan2(dy, dx) - heading[i]
            ang = ang - 2.0 * math.pi * math.floor((ang + math.pi) / (2.0 * math.pi))
            if d > radius and abs(ang) - math.asin(radius / d) > half:
                continue
            if best_d < 0.0 or d < best_d:
                best_d = d
                best_a = ang
        if best_d < 0.0:
            obs[i, col] = 0.0
            obs[i, col + 1] = 0.0
        else:
            gap = best_d - radius
            if gap < 0.0:
                gap = 0.0
            c = 1.0 - gap / vrange
            obs[i, col] = c * math.cos(best_a)
            obs[i, col + 1] = c * math.sin(best_a)


# ---- 사냥: 상처·회복·반격·고기 ----
@njit(cache=True)
def wound(s, a, dmg):
    """동물 a가 dmg만큼 다친다. 쉰 스텝 수는 0으로 돌아간다. 종류와 무관하다."""
    s.an_hp[a] -= dmg
    s.an_rest[a] = 0


@njit(cache=True)
def heal(s, a, hp_max, regen, rest):
    """맞지 않은 스텝마다 쉰 스텝 수를 하나 늘리고(rest에서 멈춘다), rest스텝을 쉬었으면 regen만큼 아문다(hp_max까지).
    종류와 무관하다."""
    if s.an_rest[a] < rest:
        s.an_rest[a] += 1
    if s.an_rest[a] >= rest:
        hp = s.an_hp[a] + regen
        s.an_hp[a] = hp_max if hp > hp_max else hp


@njit(cache=True)
def _scatter_meat(step, s, fa, ia, a):
    """쓰러진 사냥감 a의 몸 에너지를 고기 prey_meat알로 나눠 사냥감 자리 둘레에 뿌린다.
    자리는 사냥감 자리 + prey_radius × 정규 난수(S_MEAT, a = 늘리기 전 C_PREY_KILLED, b = 2q·2q+1)이고 감는다.
    고기는 들판 아닌 물건이라 저장 칸에 자리가 있어야 놓인다. 놓지 못한 알은 장부의 L_LOST로 간다."""
    n = ia[AI_MEAT]
    e = s.an_energy[a] / n
    seed = ia[AI_SEED]
    serial = s.counters[k.C_PREY_KILLED]
    r = fa[AF_RADIUS]
    for q in range(n):
        px = k._wrap(s.an_x[a] + r * normal(seed, S_MEAT, serial, 2 * q), fa[AF_W])
        py = k._wrap(s.an_y[a] + r * normal(seed, S_MEAT, serial, 2 * q + 1), fa[AF_H])
        if not (k.store_room(s, 1) and k._place_food(s, px, py, e, k.KIND_MEAT, step, 1)):
            s.ledger[k.L_LOST] += e


@njit(cache=True)
def attack_animals(step, s, fa, ia):
    """공격 출력을 낸 개체가 닿는 가장 가까운 산 동물(종류 무관)을 친다. 쓰러뜨린 사냥감 수를 돌려준다.
    1. 공격 출력 a > 0이고 에너지 > 0이면 비용 attack_cost × a(빗나가도 든다, L_MOVE)를 매긴다. 공격이 켜진 세계에서는
       st.attack이 이미 매겼으므로 여기서는 매기지 않는다. 감긴 중심 거리 AF_REACH 안의 가장 가까운 산 동물을 a_hit에 적는다.
    2. 맞은 사냥감은 그 스텝 피해의 합을 prey_sat_n × attack_damage까지만 입는다(포화).
    3. 사냥감을 친 개체는 반격 prey_strike를 그 사냥감의 공격자 수로 나눠 잃고(L_LOST) 닿음 감각이 TOUCH_HIT가 된다.
       반격은 쓰러지는 스텝에도 든다. 에너지가 바닥나면 process_deaths가 DEATH_KILLED로 센다.
    4. 체력이 0 이하인 사냥감은 쓰러져 고기가 되고, 그 스텝의 공격자 수를 C_HUNTERS에 더한다.
    5. 이번 스텝에 맞지 않은 산 사냥감은 쉬며 아문다. 순차 커널이다."""
    n_an = s.an_alive.shape[0]
    for j in range(n_an):
        s.an_natt[j] = 0
        s.an_dmg[j] = 0.0
    w = fa[AF_W]
    h = fa[AF_H]
    reach2 = fa[AF_REACH] * fa[AF_REACH]
    dmg1 = fa[AF_ATK_DMG]
    pay = ia[AI_CONFLICT] == 0
    n = s.x.shape[0]
    for i in range(n):
        s.a_hit[i] = -1
        if not s.alive[i]:
            continue
        act = float(s.act[i, ACT_ATTACK])
        if act <= 0.0 or s.energy[i] <= 0.0:
            continue
        if pay:
            cost = fa[AF_ATK_COST] * act
            s.energy[i] -= cost
            s.ledger[k.L_MOVE] += cost
        best = -1
        bd = reach2
        for j in range(n_an):
            if not s.an_alive[j]:
                continue
            dx = k._wdiff(s.an_x[j] - s.x[i], w)
            dy = k._wdiff(s.an_y[j] - s.y[i], h)
            d2 = dx * dx + dy * dy
            if d2 < bd:
                bd = d2
                best = j
        if best >= 0:
            s.a_hit[i] = best
            s.an_natt[best] += 1
            s.an_dmg[best] += dmg1 * act
    cap = ia[AI_SAT_N] * dmg1
    for j in range(n_an):
        if s.an_alive[j] and s.an_kind[j] == ANIMAL_PREY and s.an_natt[j] > 0:
            d = s.an_dmg[j]
            wound(s, j, cap if d > cap else d)
    strike = fa[AF_STRIKE]
    for i in range(n):
        j = s.a_hit[i]
        if j < 0 or s.an_kind[j] != ANIMAL_PREY:
            continue
        loss = strike / s.an_natt[j]
        s.energy[i] -= loss
        s.ledger[k.L_LOST] += loss
        s.touch[i] = k.TOUCH_HIT
    killed = 0
    for j in range(n_an):
        if not s.an_alive[j] or s.an_kind[j] != ANIMAL_PREY:
            continue
        if s.an_hp[j] <= 0.0:
            _scatter_meat(step, s, fa, ia, j)    # 고기 자리의 난수 번호는 늘리기 전 C_PREY_KILLED
            s.counters[k.C_PREY_KILLED] += 1
            s.counters[k.C_HUNTERS] += s.an_natt[j]
            s.an_alive[j] = False
            s.an_energy[j] = 0.0
            killed += 1
        elif s.an_natt[j] == 0:
            heal(s, j, fa[AF_HP], fa[AF_REGEN], ia[AI_REST])
    return killed


@njit(cache=True)
def animal_energy(s):
    """산 동물의 몸에 든 에너지 합(장부에서 세계 안의 에너지로 센다)."""
    e = 0.0
    for j in range(s.an_alive.shape[0]):
        if s.an_alive[j]:
            e += s.an_energy[j]
    return e
