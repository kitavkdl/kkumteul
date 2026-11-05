"""2~4단계 장치의 커널. 모두 설정으로 끄고 켠다(기본은 꺼짐).

- 2단계 마음: 호르몬, 보상으로 조절되는 평생 학습(가소성), 연결 기반 호기심, 잠
- 3단계 사회: 신호 내기·듣기, 혈연 알아보기, 이웃이 먹는 것 보기, 에너지 주기
- 4단계 판이 바뀌는 세계: 다닌 자리가 길이 되고, 쌓은 거름이 먹이를 끌어온다

여기 있는 것은 "장치"다. 생물이 그것을 쓰게 되는지는 진화가 정한다.
규칙은 kernels.py와 같다: 병렬 커널은 배열을 직접 받고 자기 칸에만 쓴다, 에너지는 장부에 적는다.
"""
import math

import numpy as np
from numba import njit, prange

from .config import (ACT_ATTACK, ACT_DEPOSIT, ACT_GIVE, ACT_REPRO, ACT_SIG0, ACT_SLEEP, ACT_THRUST, ACT_TURN,
                     IN_HEAR, IN_HORM, IN_KIN, IN_SAWEAT, IN_TRAIL, N_INTERNAL)
from .kernels import (C_ATTACKS, C_FERTILE, C_GIFTS, C_NFOOD, KIND_A, L_BASAL, L_BRAIN, L_DISCARD, L_LOST,
                      L_MOVE, L_SPAWN, TOUCH_HIT, _place_food, _wdiff, _wrap, make_room, pick_kind)
from .rng import S_ATTACK_ORDER, S_FERTILE, S_PERTURB, normal, u01

# 거름 먹이의 종류를 뽑는 난수 번호의 밑. S_KIND 흐름에서 보통 먹이는 (일련번호, 0), 어느 쪽이 독인지는
# (구간, 1)을 쓴다. 거름 먹이는 (스텝, 이 값 + 그 스텝의 몇 번째인지)를 써서 둘과 겹치지 않게 한다.
KIND_RNG_FERTILE = 1000003

# ---- 실수 설정 칸 ----
(F_W, F_H, F_INV_CELL, F_E_REPRO, F_BASAL, F_THRUST_COST, F_A_RAD, F_E_FOOD, F_INFLOW,
 F_TRACE, F_HEBB_DECAY, F_PLAST_COST, F_ETA_MAX, F_PROTO_LR, F_LINK_CAP,
 F_K_HUNGER, F_K_STRESS, F_K_DOPA, F_K_BORED, F_K_BASE, F_FATIGUE, F_RECOVER, F_SLEEP_MIN,
 F_SLEEP_BASAL, F_SLEEP_FORGET, F_SLEEP_KEEP,
 F_HEAR, F_SIG_COST, F_GIVE_RATE, F_GIVE_LOSS, F_GIVE_MIN,
 F_WEAR, F_TRAIL_DECAY, F_TRAIL_SAT, F_PATH_BONUS, F_DEP_RATE, F_DEP_LOSS, F_SOIL_DECAY,
 F_SOIL_REF, F_FERT_FRAC, F_SOIL_USE, F_CELL, F_ATK_DMG, F_ATK_GAIN, F_ATK_COST, F_KIND_RATIO,
 F_PERTURB, N_FS) = range(48)
# ---- 정수 설정 칸 ----
(I_SEED, I_GW, I_GH, I_SECTORS, I_H_MAX, I_MIND, I_SOCIAL, I_NICHE, I_FOOD_CAP, I_STAGE_IN, I_KINDS,
 I_CAP_REPLACE, N_IS) = range(13)


def pack_stage_params(cfg):
    fs = np.zeros(N_FS, dtype=np.float64)
    isp = np.zeros(N_IS, dtype=np.int64)
    fs[F_W], fs[F_H], fs[F_INV_CELL], fs[F_CELL] = cfg.world_w, cfg.world_h, 1.0 / cfg.cell, cfg.cell
    fs[F_E_REPRO], fs[F_BASAL], fs[F_THRUST_COST] = cfg.e_repro, cfg.basal_cost, cfg.thrust_cost
    fs[F_A_RAD], fs[F_E_FOOD], fs[F_INFLOW] = cfg.agent_radius, cfg.food_energy, cfg.food_inflow
    fs[F_TRACE], fs[F_HEBB_DECAY], fs[F_PLAST_COST] = cfg.trace_decay, cfg.hebb_decay, cfg.plast_cost
    fs[F_ETA_MAX], fs[F_PROTO_LR], fs[F_LINK_CAP] = cfg.eta_max, cfg.proto_lr, cfg.link_cap
    fs[F_K_HUNGER], fs[F_K_STRESS] = cfg.hormone_rate_hunger, cfg.hormone_rate_stress
    fs[F_K_DOPA], fs[F_K_BORED] = cfg.hormone_rate_dopamine, cfg.hormone_rate_boredom
    fs[F_K_BASE], fs[F_FATIGUE], fs[F_RECOVER] = cfg.reward_baseline_rate, cfg.fatigue_rate, cfg.sleep_recover
    fs[F_SLEEP_MIN], fs[F_SLEEP_BASAL] = cfg.sleep_min_fatigue, cfg.sleep_basal
    fs[F_SLEEP_FORGET], fs[F_SLEEP_KEEP] = cfg.sleep_forget, cfg.sleep_keep
    fs[F_HEAR], fs[F_SIG_COST] = cfg.hear_range, cfg.signal_cost
    fs[F_GIVE_RATE], fs[F_GIVE_LOSS], fs[F_GIVE_MIN] = cfg.give_rate, cfg.give_loss, cfg.give_min_energy
    fs[F_WEAR], fs[F_TRAIL_DECAY], fs[F_TRAIL_SAT] = cfg.wear_rate, cfg.trail_decay, cfg.trail_sat
    fs[F_PATH_BONUS], fs[F_DEP_RATE], fs[F_DEP_LOSS] = cfg.path_bonus, cfg.deposit_rate, cfg.deposit_loss
    fs[F_SOIL_DECAY], fs[F_SOIL_REF], fs[F_FERT_FRAC] = cfg.soil_decay, cfg.soil_ref, cfg.fert_frac
    fs[F_SOIL_USE] = cfg.soil_use
    fs[F_ATK_DMG], fs[F_ATK_GAIN], fs[F_ATK_COST] = cfg.attack_damage, cfg.attack_gain, cfg.attack_cost
    fs[F_KIND_RATIO] = cfg.kind_ratio
    fs[F_PERTURB] = cfg.perturb_sigma
    isp[I_SEED], isp[I_GW], isp[I_GH] = cfg.seed, cfg.grid_w, cfg.grid_h
    isp[I_SECTORS], isp[I_H_MAX] = cfg.vision_sectors, cfg.h_max
    isp[I_MIND], isp[I_SOCIAL], isp[I_NICHE] = int(cfg.mind_enabled), int(cfg.social_enabled), int(cfg.niche_enabled)
    isp[I_FOOD_CAP] = cfg.food_cap
    isp[I_KINDS] = int(cfg.food_kinds_enabled)
    isp[I_CAP_REPLACE] = int(cfg.food_cap_replace)
    isp[I_STAGE_IN] = 2 * cfg.vision_sectors + N_INTERNAL
    return fs, isp


@njit(cache=True, inline="always")
def _cell_of(x, y, inv_cell, gw, gh):
    cx = int(x * inv_cell)
    cy = int(y * inv_cell)
    if cx >= gw:
        cx = gw - 1
    if cy >= gh:
        cy = gh - 1
    return cy * gw + cx


# =========================== 3단계: 사회 (감각) ===========================
@njit(cache=True, parallel=True)
def _social_sense_par(fs, isp, alive, x, y, heading, color, ate, sig, obs, a_start, a_items):
    n = x.shape[0]
    gw = isp[I_GW]
    gh = isp[I_GH]
    w = fs[F_W]
    hgt = fs[F_H]
    inv_cell = fs[F_INV_CELL]
    rng = fs[F_HEAR]
    r2 = rng * rng
    reach = int(math.ceil(rng * inv_cell))
    base = isp[I_STAGE_IN]
    for i in prange(n):
        if not alive[i]:
            continue
        xi = x[i]
        yi = y[i]
        ch = math.cos(heading[i])
        sh = math.sin(heading[i])
        cx = int(xi * inv_cell)
        cy = int(yi * inv_cell)
        if cx >= gw:
            cx = gw - 1
        if cy >= gh:
            cy = gh - 1
        h0f = 0.0
        h0b = 0.0
        h1f = 0.0
        h1b = 0.0
        saw = 0.0
        near = -1
        near_d2 = r2
        for oy in range(-reach, reach + 1):
            yy = (cy + oy) % gh
            for ox in range(-reach, reach + 1):
                xx = (cx + ox) % gw
                c = yy * gw + xx
                for p in range(a_start[c], a_start[c + 1]):
                    j = a_items[p]
                    if j == i:
                        continue
                    ddx = _wdiff(x[j] - xi, w)
                    ddy = _wdiff(y[j] - yi, hgt)
                    d2 = ddx * ddx + ddy * ddy
                    if d2 > r2:
                        continue
                    wgt = 1.0 - math.sqrt(d2) / rng
                    if ddx * ch + ddy * sh >= 0.0:
                        h0f += sig[j, 0] * wgt
                        h1f += sig[j, 1] * wgt
                    else:
                        h0b += sig[j, 0] * wgt
                        h1b += sig[j, 1] * wgt
                    if ate[j] * wgt > saw:
                        saw = ate[j] * wgt
                    if d2 < near_d2:
                        near_d2 = d2
                        near = j
        obs[i, base + IN_HEAR + 0] = math.tanh(h0f)
        obs[i, base + IN_HEAR + 1] = math.tanh(h0b)
        obs[i, base + IN_HEAR + 2] = math.tanh(h1f)
        obs[i, base + IN_HEAR + 3] = math.tanh(h1b)
        kin = 0.0
        if near >= 0:
            dc = 0.0
            for c3 in range(3):
                t = color[i, c3] - color[near, c3]
                dc += t * t
            kin = 1.0 - math.sqrt(dc) / 1.7320508075688772
        obs[i, base + IN_KIN] = kin
        obs[i, base + IN_SAWEAT] = saw


@njit(cache=True)
def social_sense(s, fs, isp):
    _social_sense_par(fs, isp, s.alive, s.x, s.y, s.heading, s.color, s.ate, s.sig, s.obs,
                      s.a_start, s.a_items)


# =========================== 4단계: 흔적 (감각) ===========================
@njit(cache=True, parallel=True)
def _trail_sense_par(fs, isp, alive, x, y, heading, trail, obs):
    n = x.shape[0]
    gw = isp[I_GW]
    gh = isp[I_GH]
    w = fs[F_W]
    hgt = fs[F_H]
    inv_cell = fs[F_INV_CELL]
    sat = fs[F_TRAIL_SAT]
    look = fs[F_CELL]
    base = isp[I_STAGE_IN]
    for i in prange(n):
        if not alive[i]:
            continue
        hd = heading[i]
        c0 = _cell_of(x[i], y[i], inv_cell, gw, gh)
        xl = _wrap(x[i] + look * math.cos(hd - 0.6), w)
        yl = _wrap(y[i] + look * math.sin(hd - 0.6), hgt)
        xr = _wrap(x[i] + look * math.cos(hd + 0.6), w)
        yr = _wrap(y[i] + look * math.sin(hd + 0.6), hgt)
        t0 = trail[c0] / sat
        tl = trail[_cell_of(xl, yl, inv_cell, gw, gh)] / sat
        tr = trail[_cell_of(xr, yr, inv_cell, gw, gh)] / sat
        obs[i, base + IN_TRAIL + 0] = 1.0 if t0 > 1.0 else t0
        obs[i, base + IN_TRAIL + 1] = 1.0 if tl > 1.0 else tl
        obs[i, base + IN_TRAIL + 2] = 1.0 if tr > 1.0 else tr


@njit(cache=True)
def trail_sense(s, fs, isp):
    _trail_sense_par(fs, isp, s.alive, s.x, s.y, s.heading, s.trail, s.obs)


# =========================== 2단계: 마음 ===========================
@njit(cache=True, parallel=True)
def _mind_par(fs, isp, alive, obs, energy, e_prev, ate, touch, horm, mod, exp_r, fatigue, asleep,
              proto, link, last_proto, rw):
    """호르몬·호기심·보상 신호를 갱신하고 호르몬을 뇌 입력에 넣는다. 지난 스텝의 결과(먹음, 닿음,
    에너지 변화)를 보고 이번 스텝의 학습 조절 신호(mod)를 정한다."""
    n = alive.shape[0]
    nv = 2 * isp[I_SECTORS]
    base = isp[I_STAGE_IN]
    n_proto = proto.shape[1]
    lr = fs[F_PROTO_LR]
    cap = fs[F_LINK_CAP]
    for i in prange(n):
        if not alive[i]:
            continue
        cur = 0.0
        if asleep[i] > 0.0:
            # 자는 동안: 피로가 풀리고, 보지 못하고, 약한 연결은 흐려진다
            fatigue[i] -= fs[F_RECOVER]
            for k in range(nv):
                obs[i, k] = 0.0
            for a in range(n_proto):
                for b in range(n_proto):
                    link[i, a, b] *= 0.999
            if fatigue[i] <= 0.0:
                fatigue[i] = 0.0
                asleep[i] = 0.0
        else:
            # 감각을 가장 가까운 패턴에 묶고, 그 패턴을 조금 끌어온다(원본을 다 저장하지 않는다)
            best = 0
            bd = 1e30
            for p in range(n_proto):
                d = 0.0
                for k in range(nv):
                    t = obs[i, k] - proto[i, p, k]
                    d += t * t
                if d < bd:
                    bd = d
                    best = p
            for k in range(nv):
                proto[i, best, k] += lr * (obs[i, k] - proto[i, best, k])
            # 호기심: 지난 패턴과 이번 패턴 사이의 연결이 새로울수록 값이 크다.
            # 잡음은 패턴으로 묶인 뒤라 연결이 금방 다 차서 값이 사라진다(노이즈 TV에 갇히지 않는다)
            lp = last_proto[i]
            c = link[i, lp, best]
            cur = 1.0 / (1.0 + c)
            c += 1.0
            if c > cap:
                c = cap
            link[i, lp, best] = c
            last_proto[i] = best
            fatigue[i] += fs[F_FATIGUE]
            if fatigue[i] > 1.0:
                fatigue[i] = 1.0
        de = energy[i] - e_prev[i]
        e_prev[i] = energy[i]
        hunger = 1.0 - energy[i] / fs[F_E_REPRO]
        if hunger < 0.0:
            hunger = 0.0
        if hunger > 1.0:
            hunger = 1.0
        horm[i, 0] += fs[F_K_HUNGER] * (hunger - horm[i, 0])
        horm[i, 1] += fs[F_K_STRESS] * (touch[i] - horm[i, 1])
        horm[i, 3] += fs[F_K_BORED] * ((1.0 - cur) - horm[i, 3])
        des = de * 2.0
        if des > 1.0:
            des = 1.0
        if des < -1.0:
            des = -1.0
        # 보상은 개체마다 다르고(유전), 허기에 따라 같은 먹이의 값이 달라진다
        r = (rw[i, 0] * ate[i] + rw[i, 1] * des + rw[i, 2] * touch[i] + rw[i, 3] * cur
             + rw[i, 4] * horm[i, 0] * ate[i] + rw[i, 5] * horm[i, 1]
             + rw[i, 6] * obs[i, base + IN_SAWEAT] + rw[i, 7])
        m = math.tanh(r - exp_r[i])           # 기대보다 좋았던 정도
        exp_r[i] += fs[F_K_BASE] * (r - exp_r[i])
        horm[i, 2] += fs[F_K_DOPA] * (m - horm[i, 2])
        mod[i] = m
        obs[i, base + IN_HORM + 0] = horm[i, 1]
        obs[i, base + IN_HORM + 1] = horm[i, 2]
        obs[i, base + IN_HORM + 2] = horm[i, 3]
        obs[i, base + IN_HORM + 3] = fatigue[i]


@njit(cache=True)
def mind_update(s, fs, isp):
    _mind_par(fs, isp, s.alive, s.obs, s.energy, s.e_prev, s.ate, s.touch, s.horm, s.mod, s.exp_r,
              s.fatigue, s.asleep, s.proto, s.link, s.last_proto, s.rw)


@njit(cache=True, parallel=True)
def _think_plastic_par(fs, seed, step, h_max, alive, ids, n_hidden, obs, h, hnew, act, Win, Wrec, bh, Wout, bout,
                       alpha, eta, mod, fatigue, asleep, Hin, Hrec, epre_in, epre_h, epost, elig_in, elig_rec):
    """뇌 계산 + 평생 학습. 쓰는 가중치 = 타고난 것 + alpha * 살면서 배운 것.

    학습 규칙은 둘이다. 둘 다 "학습률 × 보상 신호 × 흔적"만큼 배운 것(Hin, Hrec)을 고치고, 흔적이 다르다.
    - 헤브(perturb_sigma = 0, 기본): 흔적 = 입력의 흔적(epre) × 뉴런 활동의 흔적(epost).
      보상이 오면 함께 활동한 짝이 세진다.
    - 노드 섭동(perturb_sigma > 0): 은닉 뉴런이 움직임 출력(추력, 회전)으로 보내는 값을 잡음으로 흔들고,
      연결마다 "그 순간 들어온 값 × 흔든 양"을 쌓는다(elig_in, elig_rec). 보상이 오면 흔든 쪽으로 옮긴다.
      뉴런이 조용해도(출력 0) 배울 수 있다. 흔든 양은 그 순간의 입력과만 짝짓는다: 왼쪽으로 흔들면
      그 뒤에는 먹이가 오른쪽에 보이므로, 나중에 본 것과 짝지으면 거꾸로 배운다.
    """
    n = alive.shape[0]
    n_in = obs.shape[1]
    n_out = act.shape[1]
    lam = fs[F_TRACE]
    keep = 1.0 - fs[F_HEBB_DECAY]
    sigma = fs[F_PERTURB]
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
                    hnew[i, k] += (Win[i, j, k] + alpha[i, k] * Hin[i, j, k]) * xj
        for j in range(nh):
            hj = h[i, j]
            for k in range(nh):
                hnew[i, k] += (Wrec[i, j, k] + alpha[i, k] * Hrec[i, j, k]) * hj
        for k in range(nh):
            hnew[i, k] = math.tanh(hnew[i, k])
        if asleep[i] > 0.0:
            # 잠: 약한 학습 흔적은 지우고 센 것만 남긴다
            fg = 1.0 - fs[F_SLEEP_FORGET]
            thr = fs[F_SLEEP_KEEP]
            for j in range(n_in):
                for k in range(nh):
                    if abs(Hin[i, j, k]) < thr:
                        Hin[i, j, k] *= fg
            for j in range(nh):
                for k in range(nh):
                    if abs(Hrec[i, j, k]) < thr:
                        Hrec[i, j, k] *= fg
        else:
            # 피곤할수록 덜 배운다. 보상 신호가 클수록 방금의 활동 짝이 더 세게 남는다
            # 보상 신호(mod)는 지난 스텝 행동의 결과다. 그래서 흔적(epre, epost)은 지난 스텝까지의
            # 활동만 담은 채로 쓰고, 이번 스텝의 활동은 학습 뒤에 흔적에 더한다.
            # 배운 것은 배우지 않는 스텝에도 조금씩 흐려진다.
            g = eta[i] * (1.0 - fatigue[i]) * mod[i]
            if sigma > 0.0:
                # 노드 섭동: 지난 스텝까지 쌓인 연결별 흔적만큼 옮긴다. 쓰는 가중치가 alpha × 배운 것이라
                # alpha가 음수인 뉴런은 부호를 뒤집어야 흔든 쪽으로 간다(alpha가 0이면 배울 것이 없다)
                for k in range(nh):
                    gk = g
                    if alpha[i, k] < 0.0:
                        gk = -g
                    elif alpha[i, k] == 0.0:
                        gk = 0.0
                    for j in range(n_in):
                        v = Hin[i, j, k] * keep + gk * elig_in[i, j, k]
                        if v > 1.0:
                            v = 1.0
                        if v < -1.0:
                            v = -1.0
                        Hin[i, j, k] = v
                    for j in range(nh):
                        v = Hrec[i, j, k] * keep + gk * elig_rec[i, j, k]
                        if v > 1.0:
                            v = 1.0
                        if v < -1.0:
                            v = -1.0
                        Hrec[i, j, k] = v
            else:
                for j in range(n_in):
                    gp = g * epre_in[i, j]
                    for k in range(nh):
                        v = Hin[i, j, k] * keep + gp * epost[i, k]
                        if v > 1.0:
                            v = 1.0
                        if v < -1.0:
                            v = -1.0
                        Hin[i, j, k] = v
                for j in range(nh):
                    gp = g * epre_h[i, j]
                    for k in range(nh):
                        v = Hrec[i, j, k] * keep + gp * epost[i, k]
                        if v > 1.0:
                            v = 1.0
                        if v < -1.0:
                            v = -1.0
                        Hrec[i, j, k] = v
        if sigma > 0.0:
            # 노드 섭동. 흔든 값은 움직임 출력(추력, 회전)으로만 간다. 잠·번식·신호 같은 다른 출력은 흔들지
            # 않은 값을 읽는다(흔들면 잠드는 빈도가 달라져 켠 세계와 끈 세계의 비교가 흐려진다).
            # 되먹임으로 남는 은닉 상태(h)도 흔들지 않은 값이라 되먹임 연결의 흔적에 잡음이 섞이지 않는다
            # (섞이면 자기 연결에 잡음끼리의 곱이 쌓인다).
            # 출력에는 tanh 뒤에서 실제로 바뀐 만큼(d)이 가고, 흔적에는 "잡음 × 기울기(1 − c²)"(e)를 쌓는다.
            # 둘은 잡음이 작으면 같다. d는 tanh가 굽어 있어 평균이 0이 아니고(약 −σ²·c·(1−c²)), 그대로 쌓으면
            # 보상과 상관없이 가중치가 한쪽으로 밀린다. e는 평균이 0이고, 포화한 뉴런에서는 d처럼 작아진다.
            # 자는 동안은 흔들지 않는다(행동이 꺼져 있어 흔들어도 결과가 없다). 흔적은 흐려지기만 한다.
            # 이 규칙에서는 epost를 "이번 스텝에 흔적에 넣는 양"을 담는 칸으로 쓰고 epre_in, epre_h는 쓰지 않는다.
            awake = asleep[i] <= 0.0
            d_thrust = 0.0
            d_turn = 0.0
            for k in range(nh):
                c = float(hnew[i, k])
                e = 0.0
                if awake:
                    xi = sigma * normal(seed, S_PERTURB, ids[i], step * h_max + k)
                    # tanh(a + b) = (tanh a + tanh b) / (1 + tanh a · tanh b): tanh 앞에 잡음을 더한 것과 같다
                    tn = math.tanh(xi)
                    d = (c + tn) / (1.0 + c * tn) - c
                    d_thrust += Wout[i, ACT_THRUST, k] * d
                    d_turn += Wout[i, ACT_TURN, k] * d
                    e = (1.0 - c * c) * xi
                epost[i, k] = e
            for j in range(n_in):
                xj = obs[i, j]
                for k in range(nh):
                    elig_in[i, j, k] = lam * elig_in[i, j, k] + xj * epost[i, k]
            for j in range(nh):
                hj = h[i, j]        # 지난 스텝의 흔들지 않은 상태
                for k in range(nh):
                    elig_rec[i, j, k] = lam * elig_rec[i, j, k] + hj * epost[i, k]
            for k in range(nh):
                h[i, k] = hnew[i, k]
            for o in range(n_out):
                acc = bout[i, o]
                for k in range(nh):
                    acc += Wout[i, o, k] * h[i, k]
                if o == ACT_THRUST:
                    acc += d_thrust
                elif o == ACT_TURN:
                    acc += d_turn
                act[i, o] = math.tanh(acc)
            continue
        for k in range(nh):
            epost[i, k] = lam * epost[i, k] + hnew[i, k]
        for j in range(n_in):
            epre_in[i, j] = lam * epre_in[i, j] + obs[i, j]
        for j in range(nh):
            epre_h[i, j] = lam * epre_h[i, j] + h[i, j]
        for k in range(nh):
            h[i, k] = hnew[i, k]
        for o in range(n_out):
            acc = bout[i, o]
            for k in range(nh):
                acc += Wout[i, o, k] * h[i, k]
            act[i, o] = math.tanh(acc)


@njit(cache=True)
def think_plastic(step, s, fs, isp):
    if fs[F_PERTURB] > 0.0 and (s.elig_in.shape[1] != s.obs.shape[1] or s.elig_in.shape[2] != s.h.shape[1]
                                or s.elig_rec.shape[1] != s.h.shape[1]):
        # 커널은 배열의 범위를 보지 않는다. 설정만 켜지고 배열이 작으면 다른 개체의 칸에 쓰게 된다
        raise ValueError("perturb_sigma > 0인데 연결별 흔적 배열이 작다(설정은 World.set_config로 바꾼다)")
    _think_plastic_par(fs, isp[I_SEED], step, isp[I_H_MAX], s.alive, s.ids, s.n_hidden, s.obs, s.h, s.hnew, s.act,
                       s.Win, s.Wrec, s.bh, s.Wout, s.bout, s.alpha, s.eta, s.mod, s.fatigue, s.asleep,
                       s.Hin, s.Hrec, s.epre_in, s.epre_h, s.epost, s.elig_in, s.elig_rec)


@njit(cache=True)
def after_think(s, fs, isp):
    """뇌 출력 뒤에 한다: 잠들기, 자는 개체의 행동 끄기, 신호 기록과 그 비용."""
    n = s.x.shape[0]
    mind = isp[I_MIND] != 0
    social = isp[I_SOCIAL] != 0
    n_out = s.act.shape[1]
    for i in range(n):
        if not s.alive[i]:
            continue
        if mind:
            if s.asleep[i] <= 0.0 and (s.fatigue[i] >= 1.0 or
                                       (s.act[i, ACT_SLEEP] > 0.0 and s.fatigue[i] > fs[F_SLEEP_MIN])):
                s.asleep[i] = 1.0
            if s.asleep[i] > 0.0:
                for o in range(n_out):
                    s.act[i, o] = 0.0
                s.act[i, ACT_REPRO] = -1.0
        if social:
            a0 = float(s.act[i, ACT_SIG0])
            a1 = float(s.act[i, ACT_SIG0 + 1])
            s.sig[i, 0] = a0
            s.sig[i, 1] = a1
            c = fs[F_SIG_COST] * (abs(a0) + abs(a1))
            s.energy[i] -= c
            s.ledger[L_MOVE] += c


@njit(cache=True)
def after_move(s, fs, isp):
    """움직인 뒤에 한다: 잠의 대사 할인, 학습 비용, 길의 이동 비용 할인, 흔적 남기기, 거름 쌓기."""
    n = s.x.shape[0]
    mind = isp[I_MIND] != 0
    niche = isp[I_NICHE] != 0
    gw = isp[I_GW]
    gh = isp[I_GH]
    h_max = isp[I_H_MAX]
    for i in range(n):
        if not s.alive[i]:
            continue
        if mind:
            if s.asleep[i] > 0.0:
                back = fs[F_BASAL] * (1.0 - fs[F_SLEEP_BASAL])
                s.energy[i] += back
                s.ledger[L_BASAL] -= back
            c = fs[F_PLAST_COST] * (s.eta[i] / fs[F_ETA_MAX]) * s.n_hidden[i] / h_max
            s.energy[i] -= c
            s.ledger[L_BRAIN] += c
        if niche:
            cell = _cell_of(s.x[i], s.y[i], fs[F_INV_CELL], gw, gh)
            tn = s.trail[cell] / fs[F_TRAIL_SAT]
            if tn > 1.0:
                tn = 1.0
            thrust = abs(float(s.act[i, ACT_THRUST]))
            back = fs[F_THRUST_COST] * thrust * fs[F_PATH_BONUS] * tn   # 닦인 길은 덜 든다
            s.energy[i] += back
            s.ledger[L_MOVE] -= back
            sp = math.sqrt(s.vx[i] * s.vx[i] + s.vy[i] * s.vy[i])
            s.trail[cell] += fs[F_WEAR] * sp
            dep = float(s.act[i, ACT_DEPOSIT])
            if dep > 0.0 and s.energy[i] > fs[F_GIVE_MIN]:
                amt = fs[F_DEP_RATE] * dep
                s.energy[i] -= amt
                kept = amt * (1.0 - fs[F_DEP_LOSS])
                s.soil[cell] += kept
                s.ledger[L_LOST] += amt - kept


# =========================== 3단계: 주기 ===========================
@njit(cache=True, inline="always")
def _reach2(fs):
    """건네기와 공격이 닿는 거리의 제곱. 두 장치가 같은 거리를 쓴다."""
    reach = 2.0 * fs[F_A_RAD] + 0.5
    return reach * reach


@njit(cache=True)
def _nearest_touching(s, fs, isp, i, need_energy):
    """i에게 닿아 있는(거리 안의) 가장 가까운 살아 있는 이웃의 칸. 없으면 -1.
    need_energy면 에너지가 0 이하인 이웃은 건너뛴다(빼앗을 것이 없다)."""
    gw = isp[I_GW]
    gh = isp[I_GH]
    w = fs[F_W]
    hgt = fs[F_H]
    inv_cell = fs[F_INV_CELL]
    xi = s.x[i]
    yi = s.y[i]
    cx = int(xi * inv_cell)
    cy = int(yi * inv_cell)
    if cx >= gw:
        cx = gw - 1
    if cy >= gh:
        cy = gh - 1
    best = -1
    bd = _reach2(fs)
    for oy in range(-1, 2):
        yy = (cy + oy) % gh
        for ox in range(-1, 2):
            xx = (cx + ox) % gw
            c = yy * gw + xx
            for p in range(s.a_start[c], s.a_start[c + 1]):
                j = s.a_items[p]
                if j == i or not s.alive[j]:
                    continue
                if need_energy and s.energy[j] <= 0.0:
                    continue
                ddx = _wdiff(s.x[j] - xi, w)
                ddy = _wdiff(s.y[j] - yi, hgt)
                d2 = ddx * ddx + ddy * ddy
                if d2 < bd:
                    bd = d2
                    best = j
    return best


@njit(cache=True)
def give(s, fs, isp):
    """주기 출력이 켜진 개체가 닿아 있는 가장 가까운 이웃에게 에너지를 건넨다."""
    n = s.x.shape[0]
    for i in range(n):
        if not s.alive[i]:
            continue
        g = float(s.act[i, ACT_GIVE])
        if g <= 0.0 or s.energy[i] <= fs[F_GIVE_MIN]:
            continue
        best = _nearest_touching(s, fs, isp, i, False)
        if best < 0:
            continue
        amt = fs[F_GIVE_RATE] * g
        got = amt * (1.0 - fs[F_GIVE_LOSS])
        s.energy[i] -= amt
        s.energy[best] += got
        s.ledger[L_LOST] += amt - got
        s.counters[C_GIFTS] += 1


# =========================== 다툼: 공격 ===========================
@njit(cache=True)
def attack(step, s, fs, isp):
    """공격 출력이 켜진 개체가 닿아 있는 가장 가까운 이웃의 에너지를 깎고 그 일부를 얻는다.
    맞은 개체는 닿음 감각이 TOUCH_HIT(2)가 된다(그냥 닿으면 1). 얻는 양은 깎은 양보다 적어 에너지가 생기지 않는다.
    훑는 시작 칸은 스텝마다 바꾼다. 늘 0번 칸부터 훑으면 낮은 칸의 개체가 늘 먼저 쳐서, 맞대결의 승패에
    칸 번호(나이·세대와 얽혀 있다)가 섞인다."""
    n = s.x.shape[0]
    start = int(u01(isp[I_SEED], S_ATTACK_ORDER, step, 0) * n)
    for q in range(n):
        i = start + q
        if i >= n:
            i -= n
        if not s.alive[i]:
            continue
        a = float(s.act[i, ACT_ATTACK])
        if a <= 0.0 or s.energy[i] <= 0.0:
            continue
        cost = fs[F_ATK_COST] * a
        s.energy[i] -= cost
        s.ledger[L_MOVE] += cost
        best = _nearest_touching(s, fs, isp, i, True)
        if best < 0:
            continue
        dmg = fs[F_ATK_DMG] * a
        if dmg > s.energy[best]:
            dmg = s.energy[best]          # 가진 것보다 더 빼앗을 수는 없다
        got = dmg * fs[F_ATK_GAIN]
        s.energy[best] -= dmg
        s.energy[i] += got
        s.ledger[L_LOST] += dmg - got
        s.touch[best] = TOUCH_HIT
        s.counters[C_ATTACKS] += 1


# =========================== 4단계: 땅 ===========================
@njit(cache=True)
def niche_world(step, s, fs, isp):
    """흔적과 거름이 흐려지고, 먹이 유입의 일부가 거름 있는 칸에 난다(생물이 만든 것이 판을 바꾼다)."""
    ncell = s.trail.shape[0]
    gw = isp[I_GW]
    keep_t = 1.0 - fs[F_TRAIL_DECAY]
    decay_s = fs[F_SOIL_DECAY]
    total = 0.0
    weight = 0.0
    for c in range(ncell):
        s.trail[c] *= keep_t
        v = s.soil[c]
        if v > 0.0:
            lost = v * decay_s
            v -= lost
            s.soil[c] = v
            s.ledger[L_LOST] += lost
            total += v
            tn = s.trail[c] / fs[F_TRAIL_SAT]
            if tn > 1.0:
                tn = 1.0
            weight += v * (1.0 + tn)     # 거름과 닦인 길이 함께 있으면 더 잘 난다
    if total <= 0.0:
        s.budget[0] += s.budget[1]
        s.budget[1] = 0.0
        return
    frac = total / fs[F_SOIL_REF]
    if frac > 1.0:
        frac = 1.0
    share = fs[F_INFLOW] * fs[F_FERT_FRAC] * frac
    # 보통 먹이 몫에서 떼어 온다(세계에 들어오는 먹이의 합은 그대로)
    s.budget[0] -= share
    s.budget[1] += share
    e_food = fs[F_E_FOOD]
    cell = fs[F_CELL]
    k = 0
    while s.budget[1] >= e_food:
        if s.counters[C_NFOOD] >= isp[I_FOOD_CAP] and isp[I_CAP_REPLACE] == 0:
            s.ledger[L_DISCARD] += s.budget[1]
            s.budget[1] = 0.0
            break
        target = u01(isp[I_SEED], S_FERTILE, step, 3 * k) * weight
        acc = 0.0
        pick = -1
        for c in range(ncell):
            v = s.soil[c]
            if v > 0.0:
                tn = s.trail[c] / fs[F_TRAIL_SAT]
                if tn > 1.0:
                    tn = 1.0
                acc += v * (1.0 + tn)
                if acc >= target:
                    pick = c
                    break
        if pick < 0:
            break
        px = (pick % gw + u01(isp[I_SEED], S_FERTILE, step, 3 * k + 1)) * cell
        py = (pick // gw + u01(isp[I_SEED], S_FERTILE, step, 3 * k + 2)) * cell
        kind = KIND_A
        if isp[I_KINDS] != 0:
            kind = pick_kind(isp[I_SEED], step, KIND_RNG_FERTILE + k, fs[F_KIND_RATIO])
        # 놓을 자리가 정해진 뒤에 자리를 낸다(먼저 치우면 놓지도 않고 먹이만 없앨 수 있다)
        if (not make_room(s, isp[I_FOOD_CAP], isp[I_CAP_REPLACE])
                or not _place_food(s, _wrap(px, fs[F_W]), _wrap(py, fs[F_H]), e_food, kind, step)):
            s.ledger[L_DISCARD] += s.budget[1]
            s.budget[1] = 0.0
            break
        s.ledger[L_SPAWN] += e_food
        s.budget[1] -= e_food
        s.counters[C_FERTILE] += 1
        use = fs[F_SOIL_USE]
        if use > s.soil[pick]:
            use = s.soil[pick]
        tn = s.trail[pick] / fs[F_TRAIL_SAT]
        if tn > 1.0:
            tn = 1.0
        weight -= use * (1.0 + tn)
        s.soil[pick] -= use
        s.ledger[L_LOST] += use
        k += 1
        if weight <= 1e-12:
            break
