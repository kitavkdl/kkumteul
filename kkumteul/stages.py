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

from .config import (ACT_ATTACK, ACT_DEPOSIT, ACT_GIVE, ACT_REPRO, ACT_SIG0, ACT_SIG1, ACT_SLEEP, ACT_THRUST,
                     ACT_TURN, AIM_MARKED, IN_HEAR, IN_HORM, IN_KIN, IN_SAWEAT, IN_TRAIL, N_INTERNAL, N_NORM_IN,
                     N_NORM_VIS, N_TASTE_IN, N_TASTE_VIS, NORM_AIMS, NORM_FULL, NORM_LOCAL, NORM_MEM, NORM_MODES,
                     NORM_OBS_MODES, NORM_SAWPUN, NORM_SELF, NORM_SRCS, NORM_VIS,
                     OBS_BLIND, OBS_OUTCOME, SRC_SIG, SRC_TAKE, TASTE_VIS)
from .kernels import (C_ATTACKS, C_FERTILE, C_GIFTS, C_NFIELD, C_NFOOD, KIND_A, L_BASAL, L_BRAIN, L_DISCARD, L_LOST,
                      L_MOVE, L_SPAWN, NT_ALIVE_STEPS, NT_ATK_COST, NT_ATK_VOL_MARKED, NT_ATK_VOL_PLAIN, NT_EXPO,
                      NT_OPP_MARKED, NT_OPP_OTHER, NT_OPP_OTHER_LINE, NT_OPP_OTHER_SIM0, NT_OPP_PLAIN, NT_OPP_SELF,
                      NT_PUN, NT_PUN_2ND, NT_PUN_3_KIN, NT_PUN_3_LINE, NT_PUN_3_SIM0, NT_PUN_3_STR, NT_PUN_COST,
                      NT_PUN_DMG_RECV, NT_PUN_FINE, NT_PUN_KILL, NT_PUN_NOVICT, NT_PUN_UNKNOWN, NT_PUN_VGONE,
                      RT_N_PUN, RT_PUN_COST, RT_PUN_FINE,
                      NT_TAKE, NT_TAKE_DMG_RECV, NT_TAKE_GAIN, NT_TAKE_IMP, NT_TAKE_KILL, NT_TAKE_KIN, NT_TAKE_LINE,
                      NT_MARKED_STEPS, NT_SAW_PUN, NT_TAKE_ON_MARKED, NT_TAKE_SIM0, NT_VIOL_DMG_RECV,
                      TOUCH_HIT, _kin_class, _norm_close_episode, _norm_row, _place_food, _wdiff, _wrap, make_room, pick_kind)
from .rng import (S_ATTACK_ORDER, S_FERTILE, S_NORM_BLIND, S_NORM_IMPULSE, S_NORM_MIX, S_NORM_NOISE, S_PERTURB,
                  normal, u01)

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
 F_PERTURB,
 # 규범의 씨앗(뒤에만 덧붙인다)
 F_NORM_PUN_RET, F_NORM_INHIBIT, F_NORM_MEM_RATE, F_NORM_MEM_OBS, F_NORM_MEM_TAU, F_NORM_IMPULSE, F_NORM_NOISE,
 F_NORM_SIG_THETA, F_NORM_TAU, F_NORM_MIX_FRAC, F_NORM_RADIUS, F_NORM_VRANGE, F_NORM_FOV,
 F_FIGHT_EXTRA, N_FS) = range(62)      # F_FIGHT_EXTRA: 맞공격 추가 손실(뒤에 덧붙임)
# ---- 정수 설정 칸 ----
(I_SEED, I_GW, I_GH, I_SECTORS, I_H_MAX, I_MIND, I_SOCIAL, I_NICHE, I_FOOD_CAP, I_STAGE_IN, I_KINDS,
 I_CAP_REPLACE, I_FIELD_CAP, I_TASTE_VIS0,
 # 규범의 씨앗(뒤에만 덧붙인다). I_NORM_ON은 norm_enabled이고 입력 칸이 충분할 때만 1, IN0·VIS0은 꺼지면 −1
 I_NORM_MEASURE, I_NORM_ON, I_NORM_MODE, I_NORM_SRC, I_NORM_AIM, I_NORM_OBS_MODE, I_NORM_IMP_LEN, I_NORM_BLOCK,
 I_NORM_BLOCK3, I_NORM_MIX_PERIOD, I_NORM_IN0, I_NORM_VIS0, N_IS) = range(27)


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
    isp[I_FIELD_CAP] = int(cfg.field_cap)
    isp[I_STAGE_IN] = 2 * cfg.vision_sectors + N_INTERNAL
    # 물든 시야 칸의 첫 절대 번호(자는 개체의 시야와 함께 지운다). 맛 장치가 꺼져 있으면 -1
    taste_on = cfg.taste_enabled and cfg.n_in >= cfg.taste_in0 + N_TASTE_IN
    isp[I_TASTE_VIS0] = cfg.taste_in0 + TASTE_VIS if taste_on else -1
    # 규범의 씨앗
    fs[F_NORM_PUN_RET], fs[F_NORM_INHIBIT] = cfg.norm_pun_return, cfg.norm_inhibit
    fs[F_NORM_MEM_RATE], fs[F_NORM_MEM_OBS], fs[F_NORM_MEM_TAU] = cfg.norm_mem_rate, cfg.norm_mem_obs, cfg.norm_mem_tau
    fs[F_NORM_IMPULSE], fs[F_NORM_NOISE] = cfg.norm_impulse, cfg.norm_mark_noise
    fs[F_NORM_SIG_THETA], fs[F_NORM_TAU], fs[F_NORM_MIX_FRAC] = cfg.norm_sig_theta, cfg.norm_mark_tau, cfg.norm_mix_frac
    fs[F_NORM_RADIUS], fs[F_NORM_VRANGE], fs[F_NORM_FOV] = cfg.soc_radius, cfg.vision_range, cfg.vision_fov
    fs[F_FIGHT_EXTRA] = cfg.fight_extra
    norm_on = cfg.norm_enabled and cfg.n_in >= cfg.norm_in0 + N_NORM_IN
    isp[I_NORM_MEASURE], isp[I_NORM_ON] = int(cfg.norm_measure), int(norm_on)
    isp[I_NORM_MODE], isp[I_NORM_SRC] = NORM_MODES[cfg.norm_mode], NORM_SRCS[cfg.norm_mark_src]
    isp[I_NORM_AIM], isp[I_NORM_OBS_MODE] = NORM_AIMS[cfg.norm_aim], NORM_OBS_MODES[cfg.norm_mem_obs_mode]
    isp[I_NORM_IMP_LEN], isp[I_NORM_MIX_PERIOD] = cfg.norm_impulse_len, cfg.norm_mix_period
    isp[I_NORM_BLOCK], isp[I_NORM_BLOCK3] = cfg.norm_block_tag, cfg.norm_block3_tag
    isp[I_NORM_IN0] = cfg.norm_in0 if norm_on else -1
    isp[I_NORM_VIS0] = cfg.norm_in0 + NORM_VIS if norm_on else -1
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
@njit(cache=True, inline="always")
def _color_sim(color, i, j):
    """색 유전자 닮음 1 − √(색 거리²)/√3 (같으면 1). 혈연 알아보기(IN_KIN)와 규범 계수기가 같은 식을 쓴다."""
    dc = 0.0
    for c3 in range(3):
        t = color[i, c3] - color[j, c3]
        dc += t * t
    return 1.0 - math.sqrt(dc) / 1.7320508075688772


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
            kin = _color_sim(color, i, near)
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
    tv0 = isp[I_TASTE_VIS0]
    nv0 = isp[I_NORM_VIS0]
    for i in prange(n):
        if not alive[i]:
            continue
        cur = 0.0
        if asleep[i] > 0.0:
            # 자는 동안: 피로가 풀리고, 보지 못하고, 약한 연결은 흐려진다
            fatigue[i] -= fs[F_RECOVER]
            for k in range(nv):
                obs[i, k] = 0.0
            if tv0 >= 0:
                # 물든 시야는 시야에서 나온 것이라 같이 지운다. 기억 값 칸은 시야가 아니라 둔다
                for k in range(N_TASTE_VIS):
                    obs[i, tv0 + k] = 0.0
            if nv0 >= 0:
                # 표시 시야도 시야에서 나온 것이라 지운다. 내 밝기·곁 밝기·방금 벌·내 기억은 둔다
                for k in range(N_NORM_VIS):
                    obs[i, nv0 + k] = 0.0
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
def after_move(s, fs, isp, basal_mult=1.0):
    """움직인 뒤에 한다: 잠의 대사 할인(뭉쳐 자기로 줄인 기초 대사 기준), 학습 비용, 길의 이동 비용 할인, 흔적 남기기, 거름 쌓기."""
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
                back = fs[F_BASAL] * basal_mult * (1.0 - s.huddle_cut[i]) * (1.0 - fs[F_SLEEP_BASAL])
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


# =========================== 규범의 씨앗: 도우미 ===========================
@njit(cache=True, inline="always")
def _sim_bin(sim):
    """닮음 구간: < 0.88, 0.88~0.92, 0.92~0.96, ≥ 0.96 → 0~3. 친척 경계는 분석에서 사후에 고른다(문턱 설정 키 없음)."""
    if sim < 0.88:
        return 0
    if sim < 0.92:
        return 1
    if sim < 0.96:
        return 2
    return 3


@njit(cache=True, inline="always")
def _is_marked(t, mt, tau):
    """스텝 t에 표시돼 있는가: 붙은 스텝과 그 뒤 τ스텝(모두 τ + 1스텝). 시각에서 계산하므로 감쇠 패스가 없다."""
    return mt >= 0 and t - mt <= tau


@njit(cache=True, inline="always")
def _bright(t, mt, tau):
    """표시의 밝기: 경과 0·1이면 1, 경과 τ이면 1/τ, 표시가 없으면 0(1 − max(0, 경과 − 1) ÷ τ)."""
    if not _is_marked(t, mt, tau):
        return 0.0
    e = t - mt
    if e <= 1:
        return 1.0
    return 1.0 - (e - 1) / tau


@njit(cache=True)
def _attach_mark(s, i, step, vslot, vid, tau):
    """i에게 표시를 붙인다. 표시가 없던 개체면 새 에피소드(이 에피소드의 벌 여부 mark_pun을 0으로), 아직 표시된 동안
    또 붙으면 같은 에피소드의 연장이다(시각·피해자만 새로; 에피소드는 마지막으로 붙은 때부터 만료까지 한 번 센다).
    피해자 칸·id: 가로채기면 맞은 이, sig·잡음이면 −1(없음)."""
    mt = s.mark_t[i]
    if not _is_marked(step, mt, tau):
        if mt >= 0 and step - mt == tau + 1:
            # 만료 스텝(경과 τ + 1)에 다시 붙는다: norm_account가 이 스텝 끝에 닫을 앞 에피소드를 여기서 닫는다
            # (mark_t를 덮어쓰면 그쪽에서 못 본다). 그보다 오래된 것은 만료 스텝에 이미 닫혔다
            _norm_close_episode(s, i)
        s.mark_pun[i] = 0
    s.mark_t[i] = step
    s.mark_vslot[i] = vslot
    s.mark_vid[i] = vid


@njit(cache=True)
def norm_mark_events(step, s, fs, isp):
    """뇌 출력이 정해진 뒤(after_think 뒤) 한 번: sig 원천의 표시와, 어떤 원천이든 엉뚱한 표시 잡음.
    sig: 신호 2번 출력(이미 tanh를 지난 값을 그대로)이 norm_sig_theta 위인 깬 개체. full 모드는 그 트리거 값에만
    (1 − norm_inhibit × mem)을 곱한다(실제 신호 출력·비용·이웃이 듣는 신호는 그대로 — 모형 단순화).
    잡음: 산 개체마다 확률 norm_mark_noise(상태 없는 난수 S_NORM_NOISE)."""
    n = s.x.shape[0]
    tau = int(fs[F_NORM_TAU])
    sig_src = isp[I_NORM_SRC] == SRC_SIG
    full = isp[I_NORM_MODE] == NORM_FULL
    theta = fs[F_NORM_SIG_THETA]
    inh = fs[F_NORM_INHIBIT]
    q = fs[F_NORM_NOISE]
    seed = isp[I_SEED]
    for i in range(n):
        if not s.alive[i]:
            continue
        if sig_src and s.asleep[i] <= 0.0:
            v = float(s.act[i, ACT_SIG1])
            if full:
                v *= 1.0 - inh * s.mem[i]
            if v > theta:
                _attach_mark(s, i, step, -1, -1, tau)
        if q > 0.0 and u01(seed, S_NORM_NOISE, step, s.ids[i]) < q:
            _attach_mark(s, i, step, -1, -1, tau)


@njit(cache=True)
def norm_tick(s, fs, isp):
    """삼가기 기억의 감쇠: 스텝마다 mem = max(0, mem − 1/norm_mem_tau), 0.5/τ 아래면 0으로 자른다(float32 잔여 없이
    τ번째에 정확히 0). 감각(norm_sense) 앞에 한 번 돈다."""
    n = s.x.shape[0]
    tau = fs[F_NORM_MEM_TAU]
    dec = 1.0 / tau
    floor = 0.5 / tau
    for i in range(n):
        if not s.alive[i]:
            continue
        m = float(s.mem[i])
        if m <= 0.0:
            continue
        m -= dec
        if m < floor:
            m = 0.0
        s.mem[i] = m


@njit(cache=True)
def norm_mix(step, s, fs, isp):
    """섞기: 산 개체마다 확률 norm_mix_frac로 세계 안 무작위 자리로 옮기고 속도를 0으로 한다(에너지·나이·표시·기억은 그대로).
    난수는 한 개체에 셋 — u01(seed, S_NORM_MIX, step, 3 × id + k), k = 0 옮길지, 1 x, 2 y(서로 다른 해시라 x·y가 상관되지
    않는다). world.step 맨 앞(첫 build_grid 앞)에서 step > 0이고 주기의 배수일 때만 부른다.
    한계: 친족 곁(혈연 선택)과 같은 상대와의 반복 만남(맞대응) 두 몫을 함께 깬다 — 이 장치로는 둘을 따로 못 가른다."""
    n = s.x.shape[0]
    seed = isp[I_SEED]
    f = fs[F_NORM_MIX_FRAC]
    w = fs[F_W]
    h = fs[F_H]
    for i in range(n):
        if not s.alive[i]:
            continue
        b = 3 * s.ids[i]
        if u01(seed, S_NORM_MIX, step, b) >= f:
            continue
        s.x[i] = u01(seed, S_NORM_MIX, step, b + 1) * w
        s.y[i] = u01(seed, S_NORM_MIX, step, b + 2) * h
        s.vx[i] = 0.0
        s.vy[i] = 0.0


@njit(cache=True, inline="always")
def _impulse_on(seed, step, aid, L, p):
    """떨리는 손: 개체 aid가 스텝 step에 충동 구간 안인가. 구간은 step // L마다 새로 정하는 상태 없는 난수라 저장할 것이 없고,
    한 구간은 정확히 L스텝 간다. L은 계산의 입력이 아니라 "상습 위반자" 모형의 가정이다(설계 2절·약점 10:
    벌의 억지 이득이 구간에 남은 스텝 수로 정해지므로 첫걸음 기울기가 이 가정에 달려 있다)."""
    return u01(seed, S_NORM_IMPULSE, step // L, aid) < p


@njit(cache=True)
def _scan_touching(s, fs, isp, i, t):
    """훑기: i에게 닿는(_nearest_touching과 같은 거리·같은 순회·같은 동률 규칙, 에너지 0 이하 건너뜀) 가장 가까운 이웃 nb와
    가장 가까운 표시된 이웃 nm을 한 번에 찾는다. 없으면 -1. 노출·기회·선택성 분모는 모두 여기서 센다."""
    gw = isp[I_GW]
    gh = isp[I_GH]
    w = fs[F_W]
    hgt = fs[F_H]
    inv_cell = fs[F_INV_CELL]
    tau = int(fs[F_NORM_TAU])
    xi = s.x[i]
    yi = s.y[i]
    cx = int(xi * inv_cell)
    cy = int(yi * inv_cell)
    if cx >= gw:
        cx = gw - 1
    if cy >= gh:
        cy = gh - 1
    nb = -1
    nm = -1
    bd = _reach2(fs)
    bm = bd
    for oy in range(-1, 2):
        yy = (cy + oy) % gh
        for ox in range(-1, 2):
            xx = (cx + ox) % gw
            c = yy * gw + xx
            for p in range(s.a_start[c], s.a_start[c + 1]):
                j = s.a_items[p]
                if j == i or not s.alive[j]:
                    continue
                if s.energy[j] <= 0.0:
                    continue
                ddx = _wdiff(s.x[j] - xi, w)
                ddy = _wdiff(s.y[j] - yi, hgt)
                d2 = ddx * ddx + ddy * ddy
                if d2 < bd:
                    bd = d2
                    nb = j
                if d2 < bm and _is_marked(t, s.mark_t[j], tau):
                    bm = d2
                    nm = j
    return nb, nm


@njit(cache=True, inline="always")
def _victim_of(s, m):
    """표시된 개체 m의 피해자 칸. 피해자 없음(−1: sig·잡음) → −1, 미지(−2: 이어 읽음 등) → −2,
    죽었거나 그 칸이 다른 개체로 바뀜 → −3(칸 번호와 id를 같이 보고 가린다)."""
    vid = s.mark_vid[m]
    if vid == -1:
        return -1
    if vid < 0:
        return -2
    v = s.mark_vslot[m]
    if v < 0 or v >= s.x.shape[0] or not s.alive[v] or s.ids[v] != vid:
        return -3
    return v


@njit(cache=True)
def _count_opp(s, tab, ri, i, nm):
    """훑기에서 닿는 표시된 이웃 nm이 있을 때: 그 피해자가 나(OPP_SELF)인지 산 남(OPP_OTHER + 닮음 구간·계보)인지.
    피해자 없음·미지·죽음은 어느 분모에도 안 든다(PUN_2ND·PUN_3_*의 분자와 같은 규칙이라 p2·p3 ≤ 1)."""
    v = _victim_of(s, nm)
    if v < 0:
        return
    if v == i:
        tab[ri, NT_OPP_SELF] += 1.0
        return
    tab[ri, NT_OPP_OTHER] += 1.0
    tab[ri, NT_OPP_OTHER_SIM0 + _sim_bin(_color_sim(s.color, i, v))] += 1.0
    if _kin_class(s, i, v) > 0:
        tab[ri, NT_OPP_OTHER_LINE] += 1.0


@njit(cache=True)
def _count_pun_role(s, tab, ri, i, t):
    """벌(집행자 i, 벌 받은 t)의 역할을 세고 돌려준다: 0 2자(피해자 = 집행자), 1 3자 친족(계보이거나 닮음 상위 두 구간),
    2 3자 남, −1 가를 수 없음(피해자 없음 PUN_NOVICT, 미지 PUN_UNKNOWN, 죽음·칸 바뀜 PUN_VGONE)."""
    v = _victim_of(s, t)
    if v == -1:
        tab[ri, NT_PUN_NOVICT] += 1.0
        return -1
    if v == -2:
        tab[ri, NT_PUN_UNKNOWN] += 1.0
        return -1
    if v == -3:
        tab[ri, NT_PUN_VGONE] += 1.0
        return -1
    if v == i:
        tab[ri, NT_PUN_2ND] += 1.0
        return 0
    sb = _sim_bin(_color_sim(s.color, i, v))
    line = _kin_class(s, i, v) > 0
    tab[ri, NT_PUN_3_SIM0 + sb] += 1.0
    if line:
        tab[ri, NT_PUN_3_LINE] += 1.0
    if line or sb >= 2:
        tab[ri, NT_PUN_3_KIN] += 1.0
        return 1
    tab[ri, NT_PUN_3_STR] += 1.0
    return 2


@njit(cache=True, inline="always")
def _hist_cell(s, i, t):
    """구간표(norm_hist)의 행·구간. 벌 받은 적 있으면 행 = 그 벌의 집행자 역할(pun_role 0~2; −1이면 뺌), 경과 = t − pun_t.
    벌 받은 적 없이 표시된 적 있으면 행 3, 경과 = t − mark_t. 경과 0(그 스텝에 막 벌·표시)은 뺀다(처리 순서 인공물).
    구간: [1, 10), [10, 50), [50, 200), [200, ∞). 해당 없으면 (−1, −1)."""
    if s.pun_t[i] >= 0:
        row = s.pun_role[i]
        if row < 0:
            return -1, -1
        e = t - s.pun_t[i]
    elif s.mark_t[i] >= 0:
        row = 3
        e = t - s.mark_t[i]
    else:
        return -1, -1
    if e <= 0:
        return -1, -1
    if e < 10:
        return row, 0
    if e < 50:
        return row, 1
    if e < 200:
        return row, 2
    return row, 3


@njit(cache=True)
def watch_pun(step, s, fs, isp):
    """이번 스텝에 벌 받은 개체(pun_t == step)를 soc_radius 안의 산 이웃이 본다(attack 뒤 다시 지은 격자를 쓴다).
    순차 전용(prange 금지). 적용 순서 = 벌 받은 개체의 칸 번호 오름차순. 벌 받은 본인은 여기서 안 오른다.
    본 벌 수(SAW_PUN, 관찰자의 태그 행)는 모든 모드에서 같게 센다.
    - outcome: 본 벌마다 mem += norm_mem_obs × (1 − mem)
    - blind: 이 스텝의 사건과 무관한 비율 p(이전 스텝까지 산 개체 하나가 한 스텝에 본 벌 수의 지수평활, 시간 규모
      norm_mem_tau, s.norm_blind = [합, 무게]이고 p = 합 ÷ 무게라 시작 편향이 없다)로 산 개체마다 동전(S_NORM_BLIND,
      (step, id))을 던져 같은 식으로 올린다. 벌이 없는 스텝에도 같은 비율이라 시간·공간 정보가 없고 장기 평균 상승
      횟수는 outcome과 같다. 이 스텝의 관찰은 동전을 던진 뒤에 평활에 섞는다. 누적 평균이 아니라 최근 ~τ스텝을 보므로
      마당 초반의 비율이 끝까지 끌려가지 않는다. 처음 한 스텝(관찰한 스텝이 없음)만 p = 0이다(이어 읽은 직후도 같다:
      저장하지 않는다).
    - off: 기억을 안 바꾼다."""
    n = s.x.shape[0]
    gw = isp[I_GW]
    gh = isp[I_GH]
    w = fs[F_W]
    hgt = fs[F_H]
    inv_cell = fs[F_INV_CELL]
    rad = fs[F_NORM_RADIUS]
    rad2 = rad * rad
    obs = fs[F_NORM_MEM_OBS]
    mode = isp[I_NORM_OBS_MODE]
    tab = s.norm_tab
    blind = s.norm_blind
    if mode == OBS_BLIND:
        p = 0.0
        if blind[1] > 0.0:
            p = blind[0] / blind[1]
        if p > 0.0:
            seed = isp[I_SEED]
            for i in range(n):
                if s.alive[i] and u01(seed, S_NORM_BLIND, step, s.ids[i]) < p:
                    s.mem[i] += obs * (1.0 - s.mem[i])
    saw_now = 0.0
    for j in range(n):
        if not s.alive[j] or s.pun_t[j] != step:
            continue
        xj = s.x[j]
        yj = s.y[j]
        cx = int(xj * inv_cell)
        cy = int(yj * inv_cell)
        if cx >= gw:
            cx = gw - 1
        if cy >= gh:
            cy = gh - 1
        for oy in range(-1, 2):
            # 격자가 3칸보다 좁으면 같은 칸을 두 번 훑지 않게 겹치는 쪽을 건너뛴다(watch_meals와 같다)
            if (oy == 1 and gh <= 2) or (oy == -1 and gh == 1):
                continue
            yy = (cy + oy) % gh
            for ox in range(-1, 2):
                if (ox == 1 and gw <= 2) or (ox == -1 and gw == 1):
                    continue
                xx = (cx + ox) % gw
                c = yy * gw + xx
                for p2 in range(s.a_start[c], s.a_start[c + 1]):
                    i = s.a_items[p2]
                    if i == j or not s.alive[i]:
                        continue
                    ddx = _wdiff(s.x[i] - xj, w)
                    ddy = _wdiff(s.y[i] - yj, hgt)
                    if ddx * ddx + ddy * ddy > rad2:
                        continue
                    tab[_norm_row(s.tag[i]), NT_SAW_PUN] += 1.0
                    saw_now += 1.0
                    if mode == OBS_OUTCOME:
                        s.mem[i] += obs * (1.0 - s.mem[i])
    if mode == OBS_BLIND:
        n_alive = 0
        for i in range(n):
            if s.alive[i]:
                n_alive += 1
        rate = 0.0
        if n_alive > 0:
            rate = saw_now / n_alive
        a = 1.0 / fs[F_NORM_MEM_TAU]
        blind[0] = (1.0 - a) * blind[0] + a * rate
        blind[1] = (1.0 - a) * blind[1] + a


@njit(cache=True)
def norm_account(step, s, fs, isp):
    """스텝 끝(attack 뒤, process_deaths 앞) 한 번: 표시된 개체 스텝(MARKED_STEPS, 유병률의 분자)과 이 스텝에 만료되는
    (경과 τ + 1) 표시 에피소드 닫기. 유병률의 분모 ALIVE_STEPS는 attack이 센다(그 사이에 죽는 개체가 없다)."""
    n = s.x.shape[0]
    tau = int(fs[F_NORM_TAU])
    tab = s.norm_tab
    for i in range(n):
        if not s.alive[i]:
            continue
        mt = s.mark_t[i]
        if mt < 0:
            continue
        e = step - mt
        if e <= tau:
            tab[_norm_row(s.tag[i]), NT_MARKED_STEPS] += 1.0
        elif e == tau + 1:
            _norm_close_episode(s, i)


@njit(cache=True, inline="always")
def _see_norm(obs, i, col0, nsec, ddx, ddy, d, rad, hi, fov, vrange, val):
    """표시된 이웃 하나를 표시 시야 칸(col0부터 nsec칸)에 반영한다. 칸 범위는 kernels._see와 같은 식(복사; _see는 안 고친다).
    값 = 가까움 × val(밝기). 칸마다 최댓값만 남긴다(순서 무관). 후보가 표시된 이웃뿐이라 표시 안 된 가까운 이웃이
    가리지 않는다. 본 시야·other 채널은 건드리지 않는다(_see를 재사용하면 같은 이웃이 이미 같은 가까움으로 들어 있어
    c > cur가 거짓이 되어 칸이 조용히 0이 된다)."""
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
    v = c * val
    for sec in range(s_lo, s_hi + 1):
        if v > obs[i, col0 + sec]:
            obs[i, col0 + sec] = v


@njit(cache=True, parallel=True)
def _norm_sense_par(fs, isp, t_now, alive, x, y, heading, mark_t, pun_t, mem, obs, a_start, a_items):
    """규범 감각 블록(12칸): 표시 시야 8칸(가까움 × 밝기), 내 밝기, hear_range 안 이웃의 거리 가중 평균 밝기,
    hear_range 안에 1스텝 전 벌 받은 이웃(거리 가중 최댓값), 내 삼가기 기억. 상태는 읽기만 하고 obs[i]에만 쓴다
    (prange 안에서 상태 묶음을 거쳐 쓰지 않도록 배열을 직접 받는다)."""
    n = x.shape[0]
    gw = isp[I_GW]
    gh = isp[I_GH]
    w = fs[F_W]
    hgt = fs[F_H]
    inv_cell = fs[F_INV_CELL]
    nsec = isp[I_SECTORS]
    if nsec > N_NORM_VIS:
        nsec = N_NORM_VIS
    fov = fs[F_NORM_FOV]
    vrange = fs[F_NORM_VRANGE]
    vr2 = vrange * vrange
    a_rad = fs[F_A_RAD]
    hear = fs[F_HEAR]
    h2 = hear * hear
    tau = int(fs[F_NORM_TAU])
    c0 = isp[I_NORM_IN0]
    far = vrange
    if hear > far:
        far = hear
    reach = int(math.ceil(far * inv_cell))
    for i in prange(n):
        if not alive[i]:
            continue
        for q in range(N_NORM_IN):
            obs[i, c0 + q] = 0.0
        xi = x[i]
        yi = y[i]
        hi = heading[i]
        cx = int(xi * inv_cell)
        cy = int(yi * inv_cell)
        if cx >= gw:
            cx = gw - 1
        if cy >= gh:
            cy = gh - 1
        wsum = 0.0
        bsum = 0.0
        saw = 0.0
        for oy in range(-reach, reach + 1):
            yy = (cy + oy) % gh
            for ox in range(-reach, reach + 1):
                xx = (cx + ox) % gw
                c = yy * gw + xx
                for p in range(a_start[c], a_start[c + 1]):
                    j = a_items[p]
                    if j == i or not alive[j]:
                        continue
                    ddx = _wdiff(x[j] - xi, w)
                    ddy = _wdiff(y[j] - yi, hgt)
                    d2 = ddx * ddx + ddy * ddy
                    b = _bright(t_now, mark_t[j], tau)
                    if b > 0.0 and d2 <= vr2:
                        _see_norm(obs, i, c0 + NORM_VIS, nsec, ddx, ddy, math.sqrt(d2), a_rad, hi, fov, vrange, b)
                    if d2 > h2:
                        continue
                    wgt = 1.0 - math.sqrt(d2) / hear
                    wsum += wgt
                    bsum += wgt * b
                    if pun_t[j] >= 0 and t_now - pun_t[j] == 1 and wgt > saw:
                        saw = wgt
        obs[i, c0 + NORM_SELF] = _bright(t_now, mark_t[i], tau)
        if wsum > 0.0:
            obs[i, c0 + NORM_LOCAL] = bsum / wsum
        obs[i, c0 + NORM_SAWPUN] = saw
        obs[i, c0 + NORM_MEM] = mem[i]


@njit(cache=True)
def norm_sense(step, s, fs, isp):
    """규범 감각(상태 s가 둘째 인자: knockout이 같은 꼴로 감싼다). 격자는 스텝 처음에 지은 것(k.sense와 같다)."""
    _norm_sense_par(fs, isp, step, s.alive, s.x, s.y, s.heading, s.mark_t, s.pun_t, s.mem, s.obs,
                    s.a_start, s.a_items)


# =========================== 다툼: 공격 ===========================
@njit(cache=True)
def attack(step, s, fs, isp):
    """공격 출력이 켜진 개체가 닿아 있는 가장 가까운 이웃의 에너지를 깎고 그 일부를 얻는다.
    맞은 개체는 닿음 감각이 TOUCH_HIT(2)가 된다(그냥 닿으면 1). 얻는 양은 깎은 양보다 적어 에너지가 생기지 않는다.
    훑는 시작 칸은 스텝마다 바꾼다. 늘 0번 칸부터 훑으면 낮은 칸의 개체가 늘 먼저 쳐서, 맞대결의 승패에
    칸 번호(나이·세대와 얽혀 있다)가 섞인다.

    규범 계수(norm_measure·norm_enabled)가 켜지면 산 모든 개체의 차례마다 먼저 이웃을 훑는다(훑기는 읽기만 해서
    세계 동작은 같다). 에너지가 0 이하인 개체는 차례가 없다: ALIVE_STEPS에는 세고 EXPO·기회 계수에는 안 센다.
    둘 다 꺼져 있으면 옛 공격과 똑같은 분기만 돈다."""
    n = s.x.shape[0]
    start = int(u01(isp[I_SEED], S_ATTACK_ORDER, step, 0) * n)
    count = isp[I_NORM_MEASURE] != 0 or isp[I_NORM_ON] != 0
    tab = s.norm_tab
    seed = isp[I_SEED]
    imp_p = fs[F_NORM_IMPULSE]
    imp_len = isp[I_NORM_IMP_LEN]
    on = isp[I_NORM_ON] != 0
    tau = int(fs[F_NORM_TAU])
    mark_take = on and isp[I_NORM_SRC] == SRC_TAKE
    aim_marked = isp[I_NORM_AIM] == AIM_MARKED
    full = isp[I_NORM_MODE] == NORM_FULL
    blk = isp[I_NORM_BLOCK]
    blk3 = isp[I_NORM_BLOCK3]
    # 맞공격 추가 손실(fight_extra): 이 스텝에 각 개체가 친 대상의 칸을 적어 두고, 내가 친 대상이 이미 이 스텝에 나를
    # 쳤으면(서로 친 쌍) 둘 다 fx를 더 잃는다. 개체마다 차례가 한 번이라 한 쌍은 한 스텝에 한 번만. 난수 없음.
    # 꺼져 있으면(0) 길이 1짜리 빈 배열만 만들고 아무것도 안 한다(옛 동작·해시 그대로)
    fx = fs[F_FIGHT_EXTRA]
    hit_tgt = np.full(n if fx > 0.0 else 1, -1, dtype=np.int64)
    for q in range(n):
        i = start + q
        if i >= n:
            i -= n
        if not s.alive[i]:
            continue
        ri = _norm_row(s.tag[i])
        nb = -1
        nm = -1
        hr = -1
        hb = -1
        if count:
            tab[ri, NT_ALIVE_STEPS] += 1.0
            if s.energy[i] <= 0.0:
                continue
            nb, nm = _scan_touching(s, fs, isp, i, step)
            if nb >= 0:
                tab[ri, NT_EXPO] += 1.0
                if on:
                    # 구간표: 차례를 받는 지금의 칸(이 차례에 친 뒤 표시가 붙어도 행은 그대로)
                    hr, hb = _hist_cell(s, i, step)
                    if hr >= 0:
                        s.norm_hist[hr, hb, 0] += 1.0
                if nm >= 0:
                    tab[ri, NT_OPP_MARKED] += 1.0
                    _count_opp(s, tab, ri, i, nm)
                else:
                    tab[ri, NT_OPP_PLAIN] += 1.0
        a = float(s.act[i, ACT_ATTACK])
        # 1. 떨리는 손: 충동 구간이면 출력을 max(a, 1)로 강제한다(이 변수만; 뇌 출력 s.act는 안 바뀌어 뇌는 못 말린다).
        #    자는 개체는 강제하지 않는다(after_think가 끈 출력을 우회하지 않게). 꺼져 있으면 난수도 안 부른다
        imp = False
        if imp_p > 0.0 and s.asleep[i] <= 0.0:
            imp = _impulse_on(seed, step, s.ids[i], imp_len, imp_p)
            if imp and a < 1.0:
                a = 1.0
        if a <= 0.0 or s.energy[i] <= 0.0:
            continue
        # 2. 대상 고르기(설계 순서는 비용 → 대상이나 비용 앞으로 옮겼다: 벌주기 막기가 "비용도 안 든다"를 요구해 취소 점검이
        #    비용 앞에 있어야 한다. 대상은 훑기 결과만 읽고 내 에너지에 기대지 않아 결과가 같다).
        #    충동 = 가장 가까운 nb(조준 안 씀). 자발은 marked 조준이면 닿는 표시된 이웃 nm, 없으면 nb
        if count:
            best = nb       # 훑기의 nb는 _nearest_touching과 같은 칸이다
            if on and not imp and aim_marked and nm >= 0:
                best = nm
        else:
            best = _nearest_touching(s, fs, isp, i, True)
        # 맞은 순간 대상이 표시돼 있는가(비용 앞에서 정한다: 아래 취소 점검이 쓴다. 이 차례 안에서는 안 바뀐다)
        marked_t = on and best >= 0 and _is_marked(step, s.mark_t[best], tau)
        # 2b. 벌주기 막기(겨루기 쌍둥이): 자발 공격이 표시된 대상을 겨누면 취소한다(비용·계수 없음, 대체 대상으로도 안 넘어간다).
        #     norm_block_tag 태그는 모두, norm_block3_tag 태그는 피해자가 내가 아닌 것(피해자 없음·미지 포함)만. 충동은 표시를
        #     모르므로(위반 공급) 막지 않는다
        if marked_t and not imp:
            tg = s.tag[i]
            if blk >= 0 and tg == blk:
                continue
            if blk3 >= 0 and tg == blk3 and s.mark_vid[best] != s.ids[i]:
                continue
        # 3. 비용: 위축 전 출력 a로 낸다(빗나가도 든다)
        cost = fs[F_ATK_COST] * a
        s.energy[i] -= cost
        s.ledger[L_MOVE] += cost
        if count:
            tab[ri, NT_ATK_COST] += cost
            if not imp and nb >= 0:
                # 자발 공격(충동 구간 밖, 뇌 출력 > 0)만 선택성 분자에 센다
                if nm >= 0:
                    tab[ri, NT_ATK_VOL_MARKED] += 1.0
                else:
                    tab[ri, NT_ATK_VOL_PLAIN] += 1.0
        if best < 0:
            continue
        # 4. 분기: 충동은 늘 가로채기. 자발은 full 모드에서 맞은 순간 표시돼 있는 대상이면 벌. inert는 모두 가로채기
        pun = marked_t and full and not imp
        a_hit = a
        if on and full and not pun:
            # 위축: 가로채기(충동 포함)의 피해·얻는 몫에만 (1 − norm_inhibit × mem)을 곱한다. 비용은 위에서 위축 전 a로
            # 냈고, 벌(벌주는 쪽은 삼가지 않는다)과 inert에는 곱하지 않는다
            a_hit = a * (1.0 - fs[F_NORM_INHIBIT] * s.mem[i])
        dmg = fs[F_ATK_DMG] * a_hit
        if dmg > s.energy[best]:
            dmg = s.energy[best]          # 가진 것보다 더 빼앗을 수는 없다
        if pun:
            got = dmg * fs[F_NORM_PUN_RET]    # 벌금: 옮겨 가는 몫(≤ attack_gain), 나머지는 L_LOST
            if got > dmg * fs[F_ATK_GAIN]:
                got = dmg * fs[F_ATK_GAIN]    # 설정 검사가 막지만 커널에서도 상한(새는 값을 못 쓰게)
        else:
            got = dmg * fs[F_ATK_GAIN]
        s.energy[best] -= dmg
        s.energy[i] += got
        s.ledger[L_LOST] += dmg - got
        s.touch[best] = TOUCH_HIT
        s.counters[C_ATTACKS] += 1
        if pun:
            s.pun_t[best] = step              # 집행자는 표시되지 않는다
            if s.mark_pun[best] != 2:
                s.mark_pun[best] = 1          # 이 표시 에피소드 안에 벌 받음(2 = 이어 읽기 전 에피소드는 미지로 둔다)
            s.mem[best] += fs[F_NORM_MEM_RATE] * (1.0 - s.mem[best])     # 삼가기 기억(float32, [0, 1] 안)
        elif mark_take:
            _attach_mark(s, i, step, best, s.ids[best], tau)     # 가로채기가 맞으면 친 쪽에만 표시
        if count:
            rb = _norm_row(s.tag[best])
            killed = s.energy[best] <= 0.0
            if pun:
                tab[ri, NT_PUN] += 1.0
                role = _count_pun_role(s, tab, ri, i, best)
                s.pun_role[best] = role
                if role >= 0:
                    s.norm_tab_role[role, RT_PUN_FINE] += got
                    s.norm_tab_role[role, RT_PUN_COST] += cost
                    s.norm_tab_role[role, RT_N_PUN] += 1.0
                tab[rb, NT_PUN_DMG_RECV] += dmg
                tab[ri, NT_PUN_FINE] += got
                tab[ri, NT_PUN_COST] += cost
                if killed:
                    tab[ri, NT_PUN_KILL] += 1.0
            else:
                tab[ri, NT_TAKE] += 1.0
                if hr >= 0:
                    s.norm_hist[hr, hb, 1] += 1.0
                if imp:
                    tab[ri, NT_TAKE_IMP] += 1.0
                if marked_t:
                    tab[ri, NT_TAKE_ON_MARKED] += 1.0    # inert·충동이 표시된 대상을 침
                tab[rb, NT_TAKE_DMG_RECV] += dmg
                tab[ri, NT_TAKE_GAIN] += got
                sb = _sim_bin(_color_sim(s.color, i, best))
                line = _kin_class(s, i, best) > 0
                tab[ri, NT_TAKE_SIM0 + sb] += 1.0
                if line:
                    tab[ri, NT_TAKE_LINE] += 1.0
                if line:
                    tab[ri, NT_TAKE_KIN] += 1.0     # 친족 = 계보만(부모·자식·형제). 닮음은 SIM 구간으로 따로 본다
                if killed:
                    tab[ri, NT_TAKE_KILL] += 1.0
            if not marked_t:
                # 위반 피해율의 분자: 때린 순간 표시가 없던 대상이 받은 피해(모드와 무관한 정의)
                tab[rb, NT_VIOL_DMG_RECV] += dmg
        # 5. 맞공격 추가 손실(fight_extra, 계수 뒤: 이 차례의 피해·죽음 계수는 맞은 순간의 값 그대로).
        #    best가 이 스텝 앞 차례에 나를 쳤으면 서로 친 쌍 → 둘 다 fx를 더 잃는다(가진 것까지, L_LOST로)
        if fx > 0.0:
            hit_tgt[i] = best
            if hit_tgt[best] == i:
                x = fx if fx < s.energy[i] else s.energy[i]
                if x > 0.0:
                    s.energy[i] -= x
                    s.ledger[L_LOST] += x
                x = fx if fx < s.energy[best] else s.energy[best]
                if x > 0.0:
                    s.energy[best] -= x
                    s.ledger[L_LOST] += x


# =========================== 4단계: 땅 ===========================
@njit(cache=True)
def niche_world(step, s, fs, isp, mult=1.0):
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
    share = fs[F_INFLOW] * mult * fs[F_FERT_FRAC] * frac
    # 보통 먹이 몫에서 떼어 온다(세계에 들어오는 먹이의 합은 그대로)
    s.budget[0] -= share
    s.budget[1] += share
    e_food = fs[F_E_FOOD]
    cell = fs[F_CELL]
    field_cap = isp[I_FIELD_CAP]
    k = 0
    while s.budget[1] >= e_food:
        n_food = s.counters[C_NFIELD] if field_cap != 0 else s.counters[C_NFOOD]
        if n_food >= isp[I_FOOD_CAP] and isp[I_CAP_REPLACE] == 0:
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
        if (not make_room(s, isp[I_FOOD_CAP], isp[I_CAP_REPLACE], field_cap)
                or not _place_food(s, _wrap(px, fs[F_W]), _wrap(py, fs[F_H]), e_food, kind, step, field_cap)):
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
