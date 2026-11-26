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

from .config import (ACT_ATTACK, ACT_CARRY, ACT_DEPOSIT, ACT_GIVE, ACT_REPRO, ACT_THRUST, ACT_TURN, FIRST_MEALS, N_REWARD,
                     N_TASTE_IN, N_TASTE_VIS, SOC_BLIND, SOC_INERT, SOC_MODES, SOC_WHAT, TASTE_VAL, TASTE_VIS,
                     TEACH_ANY, TEACH_COSTONLY, TEACH_DRIFT, TEACH_KNOWER, TEACH_MODES, TEACH_WHENS, WHEN_MEAL)
from .rng import (S_BIRTH_ORDER, S_BIRTH_POS, S_FOOD, S_KIND, S_INIT_AGE, S_INIT_GENOME, S_INIT_POS, S_MUT_GENE,
                  S_MUT_W, S_NOISE, S_PASSIVE, S_PATCH, S_RESEED, S_SOC_BLIND, S_SOC_GENE, S_TASTE_GENE, S_TEACH_GENE, S_FEED_GENE, normal,
                  soc_pair_key, u01)

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
    ("a_held", np.int32, (), True),           # 든 물건의 먹이 칸 번호(-1이면 빈손)
    ("taste_m", np.float32, ("2",), True),    # 맛 기억: +로 보이는 먹이, -로 보이는 먹이(-1~+1, 태어날 때 0)
    ("taste_g", np.float32, (), True),        # 맛 배움 속도 유전자: taste_rate에서 벗어난 몫(물려받음, 장치 꺼지면 안 쓰임)
    ("soc_g", np.float32, (), True),          # 보고 배우기 속도 유전자: 0이면 안 배움(물려받음, 장치 꺼지면 안 쓰임)
    ("show_g", np.float32, (), True),         # 보여 주기 강도 유전자: 0이면 안 보여 줌(물려받음, 장치 꺼지면 안 쓰임)
    ("feed_g", np.float32, (), True),         # 먹이 대주기 강도 유전자: 0이면 안 건넴(물려받음, 장치 꺼지면 안 쓰임)
    # 규범의 씨앗(평생 상태, 유전체가 아니다. 기본값이면 해시에 안 들어간다)
    ("mark_t", np.int32, (), True),           # 표시가 마지막으로 붙은 스텝(−1이면 없음). 밝기는 이 시각에서 계산한다
    ("pun_t", np.int32, (), True),            # 마지막으로 벌 받은 스텝(−1이면 없음)
    ("mem", np.float32, (), True),            # 삼가기 기억 [0, 1]
    ("hnew", np.float32, ("H",), False),
    ("obs", np.float32, ("I",), False),
    ("act", np.float32, ("O",), False),
    ("dispx", np.float64, (), False),
    ("dispy", np.float64, (), False),
    ("a_hit", np.int32, (), False),           # 이번 스텝에 친 동물 칸(-1이면 없음)
    ("huddle_cut", np.float64, (), False),    # 이번 스텝 기초 대사를 줄이는 비율(뭉쳐 자기, 겨울 밖이면 0)
    # 측정용 개체별 계수(태어날 때 0, 저장하지 않는다)
    ("n_meal", np.int32, (), False),          # 먹은 알 수
    ("n_poison", np.int32, (), False),        # 그 가운데 독
    ("n_poison10", np.int32, (), False),      # 첫 FIRST_MEALS끼 가운데 독
    # 보고 배우기용(태어날 때 0, 저장하지 않는다)
    ("meal_k", np.int8, (), False),           # 이번 스텝에 먹은 끼니의 보이는 모양 칸(0/1, 끼니가 없으면 0)
    ("meal_o", np.float32, (), False),        # 그 끼니의 결과(독 -1, 아니면 +1)
    ("n_seen", np.int32, (), False),          # 본 끼니 수
    ("n_seen_poison", np.int32, (), False),   # 그 가운데 독
    ("n_seen_kin", np.int32, (), False),      # 그 가운데 가까운 피붙이의 끼니
    ("seen_age_sum", np.int64, (), False),    # 본 시연자의 나이 합
    # 부모의 학습 보조 투자용(저장하지 않는다). know_age의 "아직 모름"은 −1이라 HASH_DEFAULTS가 채운다
    ("know_age", np.int32, (), False),        # 독 모양을 처음 "안" 나이(−1이면 아직 모름)
    ("know_src", np.int8, (), False),         # 그 앎을 만든 사건의 출처(1 자기 끼니, 2 부모, 3 부모 아닌 친족, 4 비친족)
    ("n_seen_par", np.int32, (), False),      # 부모의 끼니를 본 수
    ("n_seen_par_poison", np.int32, (), False),   # 그 가운데 독
    ("n_taught", np.int32, (), False),        # 보여 주기 boost가 붙은 관찰 수
    ("n_taught_knower", np.int32, (), False),     # 그 가운데 시연자가 독을 아는 쪽이었던 수
    ("par_near_steps", np.int32, (), False),  # 부모가 곁이었던 스텝 수
    ("par_alive_steps", np.int32, (), False), # 부모가 산 스텝 수
    ("juv_alive_steps", np.int32, (), False), # 미숙기를 센 스텝 수
    ("adult_near_sum", np.int64, (), False),  # 곁의 부모 아닌 어른 이웃 수의 합
    ("par_slot", np.int32, (), False),        # 스크래치: 부모의 산 칸(−1이면 없음)
    ("jscr_i", np.int32, (), False),          # 스크래치(정수)
    ("jscr_d", np.float64, (), False),        # 스크래치(실수)
    ("tflag", np.int8, (), False),            # 스크래치: 이번 스텝 보여 주기 표시
    # 규범의 씨앗 계수용(저장하지 않는다). 피해자 칸·id는 −2 = 미지(이어 읽은 직후·주입·재사용), −1 = 없음(sig·잡음)
    ("mark_vslot", np.int32, (), False),      # 이 표시의 피해자 칸
    ("mark_vid", np.int64, (), False),        # 이 표시의 피해자 id(칸이 다른 개체로 바뀌었는지 가린다)
    ("mark_pun", np.int8, (), False),         # 이 표시 에피소드 안에 벌 받았는가(0/1)
    ("pun_role", np.int8, (), False),         # 벌 받았을 때 집행자의 역할(0 2자, 1 3자 친족, 2 3자 남, −1 없음)
]
FOOD_FIELDS = [
    ("fx", np.float64, True),
    ("fy", np.float64, True),
    ("f_alive", np.bool_, True),
    ("f_energy", np.float64, True),
    ("f_kind", np.int8, True),       # 먹이 종류 0/1, 2는 늘 먹을 수 있는 것(사체), 3은 고기(쓰러진 사냥감)
    ("f_born", np.int64, True),      # 놓인 스텝(상한에서 가장 오래된 것을 고를 때, 부패의 나이를 셀 때 쓴다)
    ("f_holder", np.int32, True),    # 이 물건을 든 개체 칸(-1이면 아무도 들지 않음)
    ("f_dropped", np.int8, True),    # 1이면 들판을 떠난 적 있는 물건(들 때 켜고, 내려놓아도 1로 남는다)
]
# 동물(사냥감 등). 개체·먹이와 따로 늘 max_animals칸이고, 산 칸만 뜻이 있다
ANIMAL_FIELDS = [
    # (이름, dtype, 저장 여부)
    ("an_alive", np.bool_, True),
    ("an_kind", np.int8, True),       # 동물 종류(animals.ANIMAL_*)
    ("an_x", np.float64, True),
    ("an_y", np.float64, True),
    ("an_heading", np.float64, True),
    ("an_hp", np.float64, True),      # 남은 체력
    ("an_energy", np.float64, True),  # 몸에 든 에너지(쓰러지면 고기가 된다). 장부에서는 세계 안의 에너지다
    ("an_tx", np.float64, True),      # 떠도는 목적지
    ("an_ty", np.float64, True),
    ("an_timer", np.int32, True),     # 0보다 크면 목적지 둘레를 맴도는 중(남은 스텝), 0이면 목적지로 가는 중
    ("an_rest", np.int32, True),      # 마지막으로 맞은 뒤 지난 스텝 수(prey_rest에서 멈춘다)
    ("an_born", np.int64, True),      # 태어난 스텝(수명을 셀 때 쓴다)
    ("an_natt", np.int32, False),     # 이번 스텝의 공격자 수
    ("an_dmg", np.float64, False),    # 이번 스텝에 받은 피해
]
ANIMAL_SAVED = tuple(f[0] for f in ANIMAL_FIELDS if f[2])
# ---- 먹이 종류 (f_kind) ----
KIND_A = 0           # 시야에서 양수로 보이는 종류
KIND_B = 1           # 시야에서 음수로 보이는 종류
KIND_CORPSE = 2      # 죽은 개체가 남긴 먹이. 어느 쪽이 독이든 늘 먹을 수 있고 시야에서는 양수로 보인다
KIND_MEAT = 3        # 쓰러진 큰 사냥감이 남긴 고기. 사체처럼 늘 먹을 수 있고 들판 알이 아니며, 사체보다 빨리 썩는다
ALWAYS_EDIBLE = (KIND_CORPSE, KIND_MEAT)    # 독이 아니고 들판 알 수에도 세지 않는 종류(한 곳에서만 정한다)
TOUCH_HIT = 2.0      # 이번 스텝에 공격을 맞은 개체의 닿음 감각(그냥 닿으면 1.0)

OTHER_SAVED = ["counters", "ledger", "budget", "trail", "soil"]
# 크기가 설정에 따라 달라도 되는 평생 상태(체크포인트와 크기가 다르면 0에서 시작한다)
LIFETIME_FLEX = ("epost", "epre_h", "epre_in", "Hin", "Hrec", "proto", "link", "elig_in", "elig_rec")
# 노드 섭동을 켰을 때만 크게 잡는 배열
PERTURB_FLEX = ("elig_in", "elig_rec")
# 나중에 생긴 저장 필드의 기본값. 값이 모두 기본값이면 상태 해시에 넣지 않고(그 필드가 생기기 전 해시와 같게),
# 이 필드가 없는 옛 체크포인트를 읽으면 기본값으로 둔다(alloc_state가 이 값으로 채운다). 새 저장 필드는 여기에 더한다.
HASH_DEFAULTS = {"elig_in": 0, "elig_rec": 0, "a_held": -1, "f_holder": -1, "f_dropped": 0, "taste_m": 0, "taste_g": 0, "soc_g": 0,
                 "show_g": 0, "feed_g": 0, "know_age": -1, "mark_t": -1, "pun_t": -1, "mem": 0}
# 저장하지 않는 필드 가운데 기본값이 0이 아닌 것(alloc_state가 채운다)
UNSAVED_DEFAULTS = {"mark_vslot": -2, "mark_vid": -2, "pun_role": -1}
OTHER_SCRATCH = ["a_start", "a_items", "f_start", "f_items", "cursor", "bbuf", "dbuf", "dbuf_f", "grad_tab",
                 "norm_tab", "norm_tab_role", "norm_hist"]

# ---- 졸업표(grad_tab[TAG_ROWS][N_GT], 저장하지 않는 세계 배열): 행은 태그(범위 밖은 접는다), 칸은 아래 GT_* ----
TAG_ROWS = 8
(GT_GRAD, GT_DEAD, GT_N1, GT_M1, GT_K1, GT_N2, GT_M2, GT_K2, GT_M3, GT_K3,
 GT_SRC1, GT_SRC2, GT_SRC3, GT_SRC4, GT_PAR_NEAR, GT_PAR_ALIVE, GT_ADULT_NEAR, GT_ALIVE_STEPS,
 GT_SEEN, GT_SEEN_PAR, GT_SEEN_PAR_POISON, GT_TAUGHT, GT_TAUGHT_KNOWER, GT_MEALS, GT_POISON_MEALS,
 GT_E_GRAD, GT_TEACH_COST, GT_FEED_GIVEN, N_GT) = range(29)
# 판정 도구가 쓰는 칸 이름표(GT_* 순서와 같다)
GT_NAMES = ("grad", "dead", "n1", "m1", "k1", "n2", "m2", "k2", "m3", "k3",
            "src1", "src2", "src3", "src4", "par_near", "par_alive", "adult_near", "alive_steps",
            "seen", "seen_par", "seen_par_poison", "taught", "taught_knower", "meals", "poison_meals",
            "e_grad", "teach_cost", "feed_given")

# ---- 규범 계수표(norm_tab[TAG_ROWS_NORM][N_NT], 저장하지 않는 세계 배열) ----
# 행 = 행위 개체(공격자·집행자·관찰자)의 태그, 단 *_RECV 칸은 받은 개체의 태그. 태그 0~3은 그 행, 그 밖(음수 포함)은
# 행 4에 접는다(tag는 int32이고 numba는 범위를 안 본다). 섬(4단계)의 무리별 칸은 이 판 밖이다.
TAG_ROWS_NORM = 5
(NT_TAKE, NT_TAKE_KIN, NT_TAKE_SIM0, NT_TAKE_SIM1, NT_TAKE_SIM2, NT_TAKE_SIM3, NT_TAKE_LINE, NT_TAKE_IMP,
 NT_TAKE_ON_MARKED, NT_PUN, NT_TAKE_KILL, NT_PUN_KILL, NT_PUN_2ND, NT_PUN_3_KIN, NT_PUN_3_STR,
 NT_PUN_3_SIM0, NT_PUN_3_SIM1, NT_PUN_3_SIM2, NT_PUN_3_SIM3, NT_PUN_3_LINE, NT_PUN_NOVICT, NT_PUN_UNKNOWN,
 NT_PUN_VGONE, NT_EXPO, NT_OPP_SELF, NT_OPP_OTHER, NT_OPP_OTHER_SIM0, NT_OPP_OTHER_SIM1, NT_OPP_OTHER_SIM2,
 NT_OPP_OTHER_SIM3, NT_OPP_OTHER_LINE, NT_OPP_MARKED, NT_OPP_PLAIN, NT_ATK_VOL_MARKED, NT_ATK_VOL_PLAIN,
 NT_MARK_EPIS0, NT_MARK_EPIS1, NT_MARK_EPIS2, NT_MARK_EPIS_PUN0, NT_MARK_EPIS_PUN1, NT_MARK_EPIS_PUN2,
 NT_ALIVE_STEPS, NT_MARKED_STEPS, NT_SAW_PUN, NT_TAKE_DMG_RECV, NT_PUN_DMG_RECV, NT_VIOL_DMG_RECV,
 NT_TAKE_GAIN, NT_PUN_FINE, NT_PUN_COST, NT_ATK_COST, NT_EATEN_DEAD, NT_LIVES_DONE, N_NT) = range(54)
# 칸 뜻: TAKE = 벌이 아닌 공격이 맞음(KIN 친족 = 닮음 상위 두 구간이거나 계보, SIM 닮음 구간, LINE 계보 = 부모·자식·형제),
# ATK_COST = 모든 공격 비용(빗나감 포함), PUN_COST = 그 가운데 벌의 몫,
# VIOL_DMG_RECV = 때린 순간 표시가 없던 대상이 받은 피해(위반 피해율의 분자; 표시가 없는 측정 세계는 모든 피해)
NT_NAMES = ("take", "take_kin", "take_sim0", "take_sim1", "take_sim2", "take_sim3", "take_line", "take_imp",
            "take_on_marked", "pun", "take_kill", "pun_kill", "pun_2nd", "pun_3_kin", "pun_3_str",
            "pun_3_sim0", "pun_3_sim1", "pun_3_sim2", "pun_3_sim3", "pun_3_line", "pun_novict", "pun_unknown",
            "pun_vgone", "expo", "opp_self", "opp_other", "opp_other_sim0", "opp_other_sim1", "opp_other_sim2",
            "opp_other_sim3", "opp_other_line", "opp_marked", "opp_plain", "atk_vol_marked", "atk_vol_plain",
            "mark_epis0", "mark_epis1", "mark_epis2", "mark_epis_pun0", "mark_epis_pun1", "mark_epis_pun2",
            "alive_steps", "marked_steps", "saw_pun", "take_dmg_recv", "pun_dmg_recv", "viol_dmg_recv",
            "take_gain", "pun_fine", "pun_cost", "atk_cost", "eaten_dead", "lives_done")
# 집행 역할별 표(norm_tab_role[3][N_RT]): 행 = 2자·3자 친족·3자 남. 지킨 피해의 추정은 판정 도구가 norm_hist로 낸다
(RT_PUN_FINE, RT_PUN_COST, RT_N_PUN, N_RT) = range(4)
RT_ROLES = ("self", "kin", "stranger")

State = namedtuple(
    "State",
    [f[0] for f in AGENT_FIELDS] + [f[0] for f in FOOD_FIELDS] + [f[0] for f in ANIMAL_FIELDS]
    + OTHER_SAVED + OTHER_SCRATCH,
)
SAVED_FIELDS = ([f[0] for f in AGENT_FIELDS if f[3]] + [f[0] for f in FOOD_FIELDS if f[2]]
                + list(ANIMAL_SAVED) + OTHER_SAVED)

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
C_NFIELD = 22           # 들판 알 수(든 적 없는 알, 사체 제외). 먹이 상한이 들판 알만 셀 때(Config.field_cap)만 맞춘다
C_CARRIES = 23
# 아래 칸은 확장 크기 세계에만 있다. 꼬리가 모두 0이면 상태 해시에 넣지 않는다.
C_STORE_EVICTED = 24    # 저장소 자리가 차서 밀려난 먹이 수
C_DEATH_PREDATED = 25   # 포식자에게 죽은 수
C_SCAVENGED = 26        # 시체를 먹은 수
C_PREY_KILLED = 27      # 사냥감이 잡혀 죽은 수
C_ANIMAL_REPELLED = 28  # 동물이 쫓겨난 수
C_CHASES = 29           # 쫓은 횟수
C_MATINGS = 30          # 짝짓기 수
C_DEATH_CONTEST = 31    # 겨루다 죽은 수
C_INFECTIONS = 32       # 감염 수
C_DEATH_INFECTED = 33   # 감염으로 죽은 수
# 34·35: 암수용 예비
C_RESCUES = 36          # 구해 준 횟수
C_HUNTERS = 37          # 쓰러뜨린 사냥감마다 그 스텝의 공격자 수를 더한다(C_HUNTERS / C_PREY_KILLED = 한 마리에 붙은 평균 머릿수)
C_PREY_HITS = 38        # 사냥감을 친 개체-스텝 수(animals.attack_animals)
# 39: 예비
N_COUNTERS_BASE = 24    # 번호는 이 파일에만 둔다. 기본 크기 세계의 계수기 길이
N_COUNTERS_EXT = 40     # 확장 크기 세계(Config.ext_size)의 계수기 길이
N_COUNTERS = N_COUNTERS_BASE    # 바깥 코드 호환용 이름
N_BUDGET_BASE = 2       # [0] 보통 먹이, [1] 거름 땅에 날 먹이
N_BUDGET_EXT = 3        # 확장 크기 세계는 한 칸이 더 있다

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
 P_KIND_RATIO, P_SPOIL_RATE, P_SPOIL_MIN, P_CARRY_COST, P_MEAT_SPOIL, P_HUDDLE_CAP, P_HUDDLE_EXP,
 P_TASTE_RATE, P_TASTE_DECAY, P_TASTE_INHERIT, P_TASTE_GENE_SIGMA, P_TASTE_COST,
 P_SOC_RATE, P_SOC_COST, P_SOC_GENE_SIGMA, P_SOC_RADIUS,
 P_JUV_EAT_FLOOR, P_JUV_MOVE_FLOOR, P_JUV_POISON_MULT, P_JUV_KNOW_THETA, P_TEACH_GAIN, P_TEACH_COST,
 P_TEACH_GENE_SIGMA, P_FEED_RATE, P_FEED_GENE_SIGMA, P_FEED_LOSS, P_FEED_MIN, N_FP) = range(75)

# ---- 정수 설정 칸 ----
(Q_SEED, Q_GW, Q_GH, Q_FOOD_CAP, Q_N_PATCH, Q_PATCH_LIFE, Q_MAX_AGE, Q_MIN_AGENTS,
 Q_SECTORS, Q_H_MAX, Q_H_INIT, Q_H_MIN, Q_REPRO_BRAIN, Q_REPRO_MIN_AGE, Q_MUT, Q_CORPSE,
 Q_PASSIVE, Q_KINDS, Q_KIND_PERIOD, Q_CAP_REPLACE, Q_SPOIL_AFTER, Q_FIELD_CAP, Q_TASTE_IN0, Q_TASTE_GENE,
 Q_KIND_SEED, Q_KIND_STEP0, Q_SOC, Q_SOC_MODE,
 Q_JUV, Q_JUV_COUNT, Q_JUV_STEPS, Q_TEACH, Q_TEACH_MODE, Q_TEACH_WHEN, Q_FEED, N_IP) = range(36)


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
    fp[P_CARRY_COST] = cfg.carry_cost_mult
    fp[P_MEAT_SPOIL] = cfg.meat_spoil_mult
    fp[P_HUDDLE_CAP] = cfg.huddle_cap
    fp[P_HUDDLE_EXP] = cfg.huddle_exp
    fp[P_TASTE_RATE] = cfg.taste_rate
    fp[P_TASTE_DECAY] = cfg.taste_decay
    fp[P_TASTE_INHERIT] = cfg.taste_inherit
    fp[P_TASTE_GENE_SIGMA] = cfg.taste_gene_sigma
    fp[P_TASTE_COST] = cfg.taste_cost
    fp[P_SOC_RATE] = cfg.soc_rate
    fp[P_SOC_COST] = cfg.soc_cost
    fp[P_SOC_GENE_SIGMA] = cfg.soc_gene_sigma
    fp[P_SOC_RADIUS] = cfg.soc_radius
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
    ip[Q_KIND_SEED] = kind_seed_of(cfg)
    ip[Q_KIND_STEP0] = cfg.kind_step0
    ip[Q_CAP_REPLACE] = int(cfg.food_cap_replace)
    ip[Q_SPOIL_AFTER] = cfg.spoil_after
    ip[Q_FIELD_CAP] = int(cfg.field_cap)
    # 맛 블록의 첫 입력 번호. 장치가 꺼져 있거나 입력 칸이 모자라면 -1(맛 블록에 쓰지 않는다)
    taste_on = cfg.taste_enabled and cfg.n_in >= cfg.taste_in0 + N_TASTE_IN
    ip[Q_TASTE_IN0] = cfg.taste_in0 if taste_on else -1
    # 맛 배움 속도 유전자: 맛 블록이 실제로 쓰일 때만(아니면 배움도 비용도 없다)
    ip[Q_TASTE_GENE] = int(cfg.taste_gene_enabled and taste_on)
    # 보고 배우기: 맛 블록이 실제로 쓰일 때만(아니면 배움도 비용도 없다)
    ip[Q_SOC] = int(cfg.soc_taste_enabled and taste_on)
    ip[Q_SOC_MODE] = SOC_MODES[cfg.soc_taste_mode]
    # 부모의 학습 보조 투자. 곁 계수기·졸업표는 juv_enabled나 juv_measure 둘 중 하나면 돈다(Q_JUV_COUNT)
    soc_on = bool(ip[Q_SOC])
    juv_on = cfg.juv_enabled
    ip[Q_JUV] = int(juv_on)
    ip[Q_JUV_COUNT] = int(juv_on or cfg.juv_measure)
    ip[Q_JUV_STEPS] = cfg.juv_steps
    ip[Q_TEACH] = int(cfg.teach_enabled and juv_on and soc_on)
    ip[Q_TEACH_MODE] = TEACH_MODES[cfg.teach_mode]
    ip[Q_TEACH_WHEN] = TEACH_WHENS[cfg.teach_cost_when]
    ip[Q_FEED] = int(cfg.feed_enabled and juv_on)
    fp[P_JUV_EAT_FLOOR] = cfg.juv_eat_floor
    fp[P_JUV_MOVE_FLOOR] = cfg.juv_move_floor
    fp[P_JUV_POISON_MULT] = cfg.juv_poison_mult
    fp[P_JUV_KNOW_THETA] = cfg.juv_know_theta
    fp[P_TEACH_GAIN] = cfg.teach_gain
    fp[P_TEACH_COST] = cfg.teach_cost
    fp[P_TEACH_GENE_SIGMA] = cfg.teach_gene_sigma
    fp[P_FEED_RATE] = cfg.feed_rate
    fp[P_FEED_GENE_SIGMA] = cfg.feed_gene_sigma
    fp[P_FEED_LOSS] = cfg.give_loss
    fp[P_FEED_MIN] = cfg.give_min_energy
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
    for name, dt, _ in ANIMAL_FIELDS:
        d[name] = np.zeros(cfg.max_animals, dtype=dt)    # 모두 0: 산 동물이 없다
    ncell = cfg.grid_w * cfg.grid_h
    ext = cfg.ext_size
    d["counters"] = np.zeros(N_COUNTERS_EXT if ext else N_COUNTERS_BASE, dtype=np.int64)
    d["ledger"] = np.zeros(N_LEDGER, dtype=np.float64)
    d["budget"] = np.zeros(N_BUDGET_EXT if ext else N_BUDGET_BASE, dtype=np.float64)
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
    d["grad_tab"] = np.zeros((TAG_ROWS, N_GT), dtype=np.float64)
    d["norm_tab"] = np.zeros((TAG_ROWS_NORM, N_NT), dtype=np.float64)
    d["norm_tab_role"] = np.zeros((3, N_RT), dtype=np.float64)
    # 구간표: 행 0~2 = 벌 받은 개체를 벌 준 집행자의 역할, 행 3 = 표시됐으나 벌 안 받음; 열 = 벌 뒤 경과 구간 넷;
    # 마지막 축 = [노출, 가로채기]
    d["norm_hist"] = np.zeros((4, 4, 2), dtype=np.float64)
    for name, v in HASH_DEFAULTS.items():
        d[name][...] = v    # 옛 체크포인트에 없는 필드는 이 값 그대로 남는다(a_held가 0이면 0번 칸을 든 꼴이 된다)
    for name, v in UNSAVED_DEFAULTS.items():
        d[name][...] = v
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


def kind_seed_of(cfg):
    """독 순서를 정하는 씨앗(kind_seed가 -1이면 seed)."""
    return cfg.seed if cfg.kind_seed < 0 else cfg.kind_seed


def cfg_good_kind(cfg, step):
    """세계 설정 기준으로 이 스텝에 먹어도 되는 종류. 커널 eat과 같은 셈(kind_seed·kind_step0 반영)."""
    return int(good_kind(kind_seed_of(cfg), step + cfg.kind_step0, cfg.kind_flip_period))


@njit(cache=True, inline="always")
def pick_kind(seed, a, b, ratio):
    """새로 나는 먹이의 종류. ratio는 KIND_B가 나는 비율이다. 난수 번호 (a, b)는 부르는 쪽이 서로 겹치지 않게 준다."""
    if u01(seed, S_KIND, a, b) < ratio:
        return KIND_B
    return KIND_A


@njit(cache=True, inline="always")
def juv_ramp(a, floor, steps):
    """미숙기 비율: floor에서 시작해 a가 steps에 이르면 1. 연속·선형이다(문턱 아님).
    a는 부르는 쪽이 맞춘다: move는 증가 전 나이(age), eat·계수는 age - 1."""
    if a >= steps:
        return 1.0
    return floor + (1.0 - floor) * (a / steps)


@njit(cache=True, inline="always")
def always_edible(kind):
    """이 종류는 어느 쪽이 독이든 늘 먹을 수 있고 들판 알이 아닌가(사체·고기)."""
    for e in ALWAYS_EDIBLE:
        if kind == e:
            return True
    return False


def always_edible_mask(kinds):
    """always_edible의 배열 판(파이썬). 다시 보기와 시험이 쓴다."""
    return np.isin(kinds, ALWAYS_EDIBLE)


@njit(cache=True, inline="always")
def is_field(s, j):
    """먹이 칸 j가 들판 알인가: 들판을 떠난 적 없고 늘 먹을 수 있는 종류(사체·고기)가 아니다."""
    return s.f_dropped[j] == 0 and not always_edible(s.f_kind[j])


@njit(cache=True)
def _place_food(s, px, py, e, kind, step, field_cap):
    """빈 먹이 칸에 아무도 들지 않은 들판 물건으로 놓는다. field_cap이면 들판 알 수도 센다. 자리가 없으면 False."""
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
            s.f_holder[cur] = -1
            s.f_dropped[cur] = 0
            s.f_alive[cur] = True
            s.counters[C_NFOOD] += 1
            if field_cap != 0 and is_field(s, cur):
                s.counters[C_NFIELD] += 1
            s.counters[C_FOOD_CURSOR] = cur + 1
            return True
        cur += 1
    return False


@njit(cache=True)
def _drop_food_slot(s, j, field_cap):
    """먹이 칸 j를 비운다. 먹이 칸을 없애는 길은 모두 여기를 지난다.
    든 개체가 있으면 손을 풀고, field_cap이고 들판 알이면 들판 알 수도 줄인다. 에너지 장부는 부르는 쪽이 쓴다."""
    h = s.f_holder[j]
    if h >= 0:
        s.a_held[h] = -1
        s.f_holder[j] = -1
    if field_cap != 0 and is_field(s, j):
        s.counters[C_NFIELD] -= 1
    s.f_alive[j] = False
    s.counters[C_NFOOD] -= 1


@njit(cache=True)
def _evict_oldest(s, field_cap):
    """가장 먼저 놓인 먹이 하나를 없앤다(같은 스텝에 놓인 것끼리는 낮은 칸부터).
    field_cap이면 들판 알만 고르고(든 알·내려놓은 알·사체는 두고), 아니면 사체도 가리지 않는다.
    없앤 에너지는 장부의 L_LOST로 간다. 없앨 먹이가 없으면 False."""
    best = -1
    born = 0
    for j in range(s.fx.shape[0]):
        if not s.f_alive[j] or (field_cap != 0 and not is_field(s, j)):
            continue
        if best < 0 or s.f_born[j] < born:
            best = j
            born = s.f_born[j]
    if best < 0:
        return False
    _drop_food_slot(s, best, field_cap)
    s.counters[C_FOOD_REPLACED] += 1
    s.ledger[L_LOST] += s.f_energy[best]
    return True


@njit(cache=True)
def make_room(s, cap, replace, field_cap):
    """새 먹이 한 알을 놓아도 되는가. 상한에 닿았으면 replace일 때만 가장 오래된 것을 치우고 True.
    field_cap이면 들판 알 수로, 아니면 칸 점유 수로 상한과 견준다."""
    n = s.counters[C_NFIELD] if field_cap != 0 else s.counters[C_NFOOD]
    if n < cap:
        return True
    if replace == 0:
        return False
    return _evict_oldest(s, field_cap)


@njit(cache=True)
def evict_stored(s, field_cap):
    """들지 않은 들판 아닌 물건(내려놓은 알·사체) 가운데 가장 먼저 놓인 것(같으면 낮은 칸)을 치운다.
    에너지는 장부의 L_LOST로 간다. 치울 것이 없으면 False."""
    best = -1
    born = 0
    for j in range(s.fx.shape[0]):
        if not s.f_alive[j] or s.f_holder[j] >= 0 or is_field(s, j):
            continue
        if best < 0 or s.f_born[j] < born:
            best = j
            born = s.f_born[j]
    if best < 0:
        return False
    _drop_food_slot(s, best, field_cap)
    s.counters[C_STORE_EVICTED] += 1
    s.ledger[L_LOST] += s.f_energy[best]
    return True


@njit(cache=True)
def store_room(s, field_cap):
    """들판 아닌 물건 하나를 더 둘 자리가 있는가. 들판 아닌 물건은 개체 칸 수(max_agents)까지만 둔다.
    찼으면 evict_stored로 자리를 내 보고 그 결과를 돌려준다. 들판 알 수를 세는 세계(field_cap)에서만 부른다."""
    if s.counters[C_NFOOD] - s.counters[C_NFIELD] < s.x.shape[0]:
        return True
    return evict_stored(s, field_cap)


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
        if not make_room(s, ip[Q_FOOD_CAP], ip[Q_CAP_REPLACE], ip[Q_FIELD_CAP]):
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
        if _place_food(s, px, py, e_food, kind, step, ip[Q_FIELD_CAP]):
            s.ledger[L_SPAWN] += e_food
            s.budget[0] -= e_food
        else:
            s.ledger[L_DISCARD] += s.budget[0]
            s.budget[0] = 0.0
            break


@njit(cache=True)
def spoil_food(step, s, fp, ip, rate_mult=1.0):
    """난 지 spoil_after 스텝이 지난 먹이(사체 포함)를 스텝마다 spoil_rate * rate_mult 비율로 줄인다.
    고기(KIND_MEAT)는 여기에 meat_spoil_mult를 더 곱한다. 줄어든 에너지는 장부의 L_LOST로 가고, 남은 것이 하한(spoil_min_frac * food_energy) 아래면 남은 것도
    L_LOST로 넣고 칸을 비운다. 순차 커널이다."""
    after = ip[Q_SPOIL_AFTER]
    rate = fp[P_SPOIL_RATE] * rate_mult
    floor = fp[P_SPOIL_MIN]
    for j in range(s.fx.shape[0]):
        if not s.f_alive[j] or step - s.f_born[j] < after:
            continue
        r = rate
        if s.f_kind[j] == KIND_MEAT:
            r = rate * fp[P_MEAT_SPOIL]
        lost = s.f_energy[j] * r
        s.f_energy[j] -= lost
        s.ledger[L_LOST] += lost
        if s.f_energy[j] < floor:
            s.ledger[L_LOST] += s.f_energy[j]
            _drop_food_slot(s, j, ip[Q_FIELD_CAP])
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
               fx, fy, f_alive, f_kind, f_holder, a_start, a_items, f_start, f_items, taste_m):
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
    t0 = ip[Q_TASTE_IN0]
    taste = t0 >= 0 and t0 + N_TASTE_IN <= n_in
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
                    if not f_alive[j] or f_holder[j] == i:
                        continue    # 자기가 든 물건은 보지도 냄새 맡지도 않는다(곁 개체에게는 보인다)
                    ddx = _wdiff(fx[j] - xi, w)
                    ddy = _wdiff(fy[j] - yi, hgt)
                    d2 = ddx * ddx + ddy * ddy
                    h = f_holder[j]
                    if h >= 0:
                        # 남이 든 알은 시야에서만 든 개체 머리 앞(몸 반지름 + 알 반지름)에 있는 것으로 본다.
                        # 든 개체가 이쪽을 보면 알이 앞에 보이고, 등을 돌리면 몸 뒤에 가려진다. 냄새는 저장된 자리 그대로
                        if d2 <= vr2:
                            smell += 1.0 / (1.0 + d2 / sl2)
                        reach_h = a_rad + f_rad
                        ddx = _wdiff(x[h] + math.cos(heading[h]) * reach_h - xi, w)
                        ddy = _wdiff(y[h] + math.sin(heading[h]) * reach_h - yi, hgt)
                        d2 = ddx * ddx + ddy * ddy
                        if d2 > vr2:
                            continue
                    else:
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
        if taste:
            # 맛 블록: 먹이 시야 칸을 그 모양(부호)의 기억으로 물들인다. 잡음 전 부호로 모양을 읽는다.
            # 꺼져 있으면 위에서 0으로 비운 그대로 둔다
            m_pos = taste_m[i, 0]
            m_neg = taste_m[i, 1]
            for sec in range(min(nsec, N_TASTE_VIS)):     # 물든 시야 칸을 넘어 기억 값 칸에 쓰지 않는다
                v = obs[i, sec]
                if v > 0.0:
                    obs[i, t0 + TASTE_VIS + sec] = v * m_pos
                elif v < 0.0:
                    obs[i, t0 + TASTE_VIS + sec] = -v * m_neg
            obs[i, t0 + TASTE_VAL + 0] = m_pos
            obs[i, t0 + TASTE_VAL + 1] = m_neg
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
               s.ate, s.obs, s.fx, s.fy, s.f_alive, s.f_kind, s.f_holder, s.a_start, s.a_items, s.f_start, s.f_items,
               s.taste_m)


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
def write_held(alive, a_held, obs, col):
    """살아 있는 개체의 입력 칸 col에 손 감각을 쓴다: 무엇을 들고 있으면 1, 빈손이면 0."""
    for i in range(alive.shape[0]):
        if alive[i]:
            obs[i, col] = 1.0 if a_held[i] >= 0 else 0.0


@njit(cache=True)
def think(s):
    _think_par(s.alive, s.n_hidden, s.obs, s.h, s.hnew, s.act, s.Win, s.Wrec, s.bh, s.Wout, s.bout)


# ---- 움직임과 대사 ----
@njit(cache=True)
def huddle(s, fp, ip, on):
    """뭉쳐 자기: 산 개체마다 닿은(중심 거리 2 * agent_radius + 0.5 안) 다른 산 개체 수 n을 세어
    기초 대사를 줄이는 비율 huddle_cut = min(huddle_cap, 1 - (1 + n) ** -huddle_exp)를 적는다.
    개체 격자는 스텝 처음에 세운 것을 쓴다(그 뒤 move 전까지 자리가 바뀌지 않는다). on이 아니면(겨울 밖) 모두 0이다."""
    n = s.x.shape[0]
    if not on:
        for i in range(n):
            s.huddle_cut[i] = 0.0
        return
    gw = ip[Q_GW]
    gh = ip[Q_GH]
    w = fp[P_W]
    hgt = fp[P_H]
    inv_cell = fp[P_INV_CELL]
    reach = 2.0 * fp[P_A_RAD] + 0.5
    reach2 = reach * reach
    cap = fp[P_HUDDLE_CAP]
    ex = fp[P_HUDDLE_EXP]
    for i in range(n):
        if not s.alive[i]:
            s.huddle_cut[i] = 0.0
            continue
        xi = s.x[i]
        yi = s.y[i]
        cx = int(xi * inv_cell)
        cy = int(yi * inv_cell)
        if cx >= gw:
            cx = gw - 1
        if cy >= gh:
            cy = gh - 1
        cnt = 0
        for oy in range(-1, 2):
            yy = (cy + oy) % gh
            for ox in range(-1, 2):
                xx = (cx + ox) % gw
                c = yy * gw + xx
                for p in range(s.a_start[c], s.a_start[c + 1]):
                    j = s.a_items[p]
                    if j == i or not s.alive[j]:
                        continue
                    ddx = _wdiff(s.x[j] - xi, w)
                    ddy = _wdiff(s.y[j] - yi, hgt)
                    if ddx * ddx + ddy * ddy < reach2:
                        cnt += 1
        cut = 1.0 - (1.0 + cnt) ** (-ex)
        if cut > cap:
            cut = cap
        s.huddle_cut[i] = cut


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
    juv = ip[Q_JUV] != 0
    juv_steps = ip[Q_JUV_STEPS]
    for i in range(n):
        if not s.alive[i]:
            continue
        # 미숙기 속도 상한: move는 age를 올리기 전이라 age가 곧 a
        vm = vmax
        if juv and s.age[i] < juv_steps:
            vm = vmax * juv_ramp(s.age[i], fp[P_JUV_MOVE_FLOOR], juv_steps)
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
        if sp > vm:
            vx *= vm / sp
            vy *= vm / sp
        s.vx[i] = vx
        s.vy[i] = vy
        s.x[i] = _wrap(s.x[i] + vx, w)
        s.y[i] = _wrap(s.y[i] + vy, hgt)
        # 비용은 명령한 출력에 매긴다(환급 없음)
        nh = s.n_hidden[i]
        c_move = fp[P_THRUST_COST] * abs(thrust) + fp[P_TURN_COST] * abs(turn)
        if thrust < 0.0:
            c_move += fp[P_THRUST_COST] * abs(thrust) * (fp[P_BACK_COST] - 1.0)
        if s.a_held[i] >= 0:
            c_move *= fp[P_CARRY_COST]      # 들고 있으면 이동·회전이 더 든다. 든 물건의 자리는 carry가 맞춘다
        c_brain = fp[P_BRAIN] * nh / h_max + fp[P_SYN] * (nh * nh + nh * n_io) / syn_max
        if fp[P_BRAIN_ACT] > 0.0:
            a = 0.0
            for k in range(nh):
                a += abs(s.h[i, k])
            c_brain += fp[P_BRAIN_ACT] * a / h_max
        basal = fp[P_BASAL] * basal_mult * (1.0 - s.huddle_cut[i])
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


# ---- 들기 ----
@njit(cache=True)
def carry(step, s, fp, ip):
    """들고 내려놓기. collide 뒤, eat 앞에 부르는 순차 커널이다.
    먼저 든 물건을 모두 든 개체의 자리로 옮긴다. 이어 step % n 칸부터 산 개체를 돌며(늘 낮은 칸이 먼저 들지 않게)
    들기 출력이 0보다 크고 빈손이면 닿는 거리 안의 들지 않은 물건(들판 알·내려놓은 알·사체) 가운데 가장 가까운 것을 들고,
    0 이하이고 들고 있으면 그 자리에 내려놓는다. 자는 개체는 든 채로 둔다(자는 동안 출력이 0이 되어도 놓지 않는다).
    들판 알을 들려면 저장 칸에 자리가 있어야 한다(store_room). 에너지는 옮기지 않는다.
    먹이 격자는 스텝 처음에 세운 것을 쓴다(든 물건 말고는 먹이가 움직이지 않는다). 기본 크기 세계에서는 하는 일이 없다."""
    if s.act.shape[1] <= ACT_CARRY:
        return
    n = s.x.shape[0]
    for i in range(n):
        j = s.a_held[i]
        if j >= 0:
            s.fx[j] = s.x[i]
            s.fy[j] = s.y[i]
    gw = ip[Q_GW]
    gh = ip[Q_GH]
    w = fp[P_W]
    hgt = fp[P_H]
    inv_cell = fp[P_INV_CELL]
    r = fp[P_A_RAD] + fp[P_F_RAD]
    r2 = r * r
    field_cap = ip[Q_FIELD_CAP]
    start = step % n
    for q in range(n):
        i = start + q
        if i >= n:
            i -= n
        if not s.alive[i] or s.asleep[i] > 0.0:
            continue
        if s.act[i, ACT_CARRY] <= 0.0:
            j = s.a_held[i]
            if j >= 0:
                s.f_holder[j] = -1
                s.a_held[i] = -1
            continue
        if s.a_held[i] >= 0:
            continue
        xi = s.x[i]
        yi = s.y[i]
        cx = int(xi * inv_cell)
        cy = int(yi * inv_cell)
        if cx >= gw:
            cx = gw - 1
        if cy >= gh:
            cy = gh - 1
        best = -1
        bd = r2
        for oy in range(-1, 2):
            yy = (cy + oy) % gh
            for ox in range(-1, 2):
                xx = (cx + ox) % gw
                c = yy * gw + xx
                for p in range(s.f_start[c], s.f_start[c + 1]):
                    j = s.f_items[p]
                    if not s.f_alive[j] or s.f_holder[j] >= 0:
                        continue
                    ddx = _wdiff(s.fx[j] - xi, w)
                    ddy = _wdiff(s.fy[j] - yi, hgt)
                    d2 = ddx * ddx + ddy * ddy
                    if d2 <= bd and (best < 0 or d2 < bd):
                        best = j
                        bd = d2
        if best < 0:
            continue
        if is_field(s, best):
            # 들판 알은 들면 저장 칸 물건이 된다. 자리가 없으면 들지 않는다(store_room은 들지 않은 들판 아닌 것만 치운다)
            if not store_room(s, field_cap):
                continue
            s.counters[C_NFIELD] -= 1
            s.f_dropped[best] = 1
        s.a_held[i] = best
        s.f_holder[best] = i
        s.counters[C_CARRIES] += 1


# ---- 먹기 ----
@njit(cache=True)
def eat(step, s, fp, ip):
    n = s.x.shape[0]
    kinds = ip[Q_KINDS] != 0
    good = 0
    if kinds:
        good = good_kind(ip[Q_KIND_SEED], step + ip[Q_KIND_STEP0], ip[Q_KIND_PERIOD])
    gw = ip[Q_GW]
    gh = ip[Q_GH]
    w = fp[P_W]
    hgt = fp[P_H]
    inv_cell = fp[P_INV_CELL]
    r = fp[P_A_RAD] + fp[P_F_RAD]
    r2 = r * r
    e_max = fp[P_E_MAX]
    field_cap = ip[Q_FIELD_CAP]
    taste = ip[Q_TASTE_IN0] >= 0
    rate0 = fp[P_TASTE_RATE]
    rate = rate0
    gene = taste and ip[Q_TASTE_GENE] != 0
    cost = fp[P_TASTE_COST]
    soc = taste and ip[Q_SOC] != 0
    soc_rate0 = fp[P_SOC_RATE]
    soc_cost = fp[P_SOC_COST]
    keep = 1.0 - fp[P_TASTE_DECAY]
    juv = ip[Q_JUV] != 0
    juv_steps = ip[Q_JUV_STEPS]
    juv_count = ip[Q_JUV_COUNT] != 0
    kp = _poison_slot(step, ip)
    for i in range(n):
        s.ate[i] = 0.0
        s.meal_o[i] = 0.0       # 이번 스텝의 끼니 기록(보고 배우기용). 죽은 칸·배불러 건너뛰는 개체도 비운다
        if not s.alive[i]:
            continue
        # 미숙기: eat는 move가 age를 올린 뒤라 a = age - 1이 move 시점 나이. 들판 알 효율 m, 독 배수 pm
        m = 1.0
        pm = 1.0
        if juv:
            a_juv = s.age[i] - 1
            if a_juv < juv_steps:
                m = juv_ramp(a_juv, fp[P_JUV_EAT_FLOOR], juv_steps)
                pm = fp[P_JUV_POISON_MULT]
        if taste:
            # 잊기: 배부른 개체도 스텝마다 기억이 0 쪽으로 줄어든다
            s.taste_m[i, 0] *= keep
            s.taste_m[i, 1] *= keep
            if gene:
                # 실제 배움 속도 = clamp(taste_rate + taste_g, 0, 1). 빨리 배우는 만큼 스텝마다 값을 치른다(뇌의 값)
                rate = rate0 + s.taste_g[i]
                if rate < 0.0:
                    rate = 0.0
                elif rate > 1.0:
                    rate = 1.0
                c_learn = cost * rate
                s.energy[i] -= c_learn
                s.ledger[L_BRAIN] += c_learn
            if soc:
                # 보고 배우는 속도의 값: 실제 속도 clamp(soc_rate + soc_g, 0, 1)에 비례해 스텝마다 치른다(뇌의 값).
                # 배불러 아래에서 continue하는 개체도 낸다. inert 모드도 낸다(비용은 watch_meals에 두지 않는다)
                rate_s = soc_rate0 + s.soc_g[i]
                if rate_s < 0.0:
                    rate_s = 0.0
                elif rate_s > 1.0:
                    rate_s = 1.0
                c_soc = soc_cost * rate_s
                s.energy[i] -= c_soc
                s.ledger[L_BRAIN] += c_soc
        if s.energy[i] >= e_max:
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
                    if not s.f_alive[j] or s.f_holder[j] >= 0:
                        continue    # 든 물건은 먹지 않는다(먹으려면 내려놓는다)
                    ddx = _wdiff(s.fx[j] - xi, w)
                    ddy = _wdiff(s.fy[j] - yi, hgt)
                    if ddx * ddx + ddy * ddy > r2:
                        continue
                    _drop_food_slot(s, j, field_cap)
                    s.counters[C_EATEN] += 1
                    fk = s.f_kind[j]
                    poison = kinds and not always_edible(fk) and fk != good
                    if s.n_meal[i] < FIRST_MEALS and poison:
                        s.n_poison10[i] += 1
                    s.n_meal[i] += 1
                    if poison:
                        s.n_poison[i] += 1
                    if taste:
                        # 먹은 순간: 보인 모양의 칸이 결과(독 -1, 아니면 +1) 쪽으로 움직인다
                        mc = 1 if fk == KIND_B else 0
                        o = -1.0 if poison else 1.0
                        s.taste_m[i, mc] += rate * (o - s.taste_m[i, mc])
                        if juv_count and s.age[i] <= juv_steps:
                            _note_know(s, i, kp, 1, fp)     # 앎 도우미: 미숙기 새끼의 자기 끼니(읽기 전용 계수)
                        if soc:
                            # 끼니 기록: 독 우선, 아니면 마지막 좋은 알(독 분기의 continue보다 앞에서 적는다)
                            if poison:
                                s.meal_k[i] = mc
                                s.meal_o[i] = -1.0
                            elif s.meal_o[i] != -1.0:
                                s.meal_k[i] = mc
                                s.meal_o[i] = 1.0
                    if poison:
                        # 독: 먹이의 에너지는 사라지고 먹은 쪽도 에너지를 잃는다
                        # (미숙기 새끼는 독 배수 pm만큼 더 잃는다. 어른·꺼짐은 pm = 1.0이라 비트가 같다)
                        s.energy[i] -= fp[P_POISON] * pm
                        s.ledger[L_LOST] += s.f_energy[j] + fp[P_POISON] * pm
                        s.counters[C_POISONED] += 1
                        s.ate[i] = 1.0
                        continue
                    # 들판 알은 미숙기 효율 m을 곱해 얻고 못 얻은 몫은 L_LOST로 간다(장부). 사체·고기는 효율 1
                    gain = s.f_energy[j]
                    if not always_edible(fk):
                        gain = s.f_energy[j] * m
                    s.energy[i] += gain
                    s.eaten[i] += gain
                    s.ledger[L_LOST] += s.f_energy[j] - gain
                    s.ate[i] = 1.0
                    if s.energy[i] >= e_max:
                        full = True
                        break


# ---- 보고 배우기 ----
@njit(cache=True)
def watch_meals(step, s, fp, ip):
    """이번 스텝에 끼니를 먹은 개체(meal_o != 0)를 반지름 안의 이웃이 보고 맛 기억을 움직인다(사회 학습).

    순차 전용: prange 금지, 상태 묶음 s를 거쳐 쓴다. 적용 순서 = 시연자 j의 칸 번호 오름차순
    (같은 관찰자의 한 칸을 여러 시연자가 차례로 갱신하는 지수평활은 비가환이라 순서가 값을 가른다).
    먹은 j 자신의 기억은 여기서 안 바뀐다(자기 끼니는 eat의 개인 학습). id < 2^32를 가정한다(눈가림 쌍 키).
    모드: outcome = 시연자의 (모양, 결과)대로, what = 결과를 늘 +1로(무엇을 먹었나만), inert = 기억을 안 바꾼다(보기만),
    blind = 결과·속도·때는 outcome과 같되 갱신할 모양 칸을 동전으로(정보량만 0). 보기 계수기는 모든 모드에서 같다.

    부모의 보여 주기(ip[Q_TEACH]): 시연자 j가 어른(age > juv_steps)이고 관찰자 i가 미숙기(age <= juv_steps)인 쌍에서
    자격이 있으면(kin·costonly·drift = j가 i의 부모, knower = 부모이고 j가 그 스텝 독 모양을 앎, any = 모두)
    실제 속도 = clamp(soc_rate + soc_g[i] + boost, 0, 1), boost = teach_gain × clamp(show_g[j], 0, 1)
    (costonly·drift는 boost 0; 갱신식은 그대로). boost가 붙은 관찰마다 n_taught(시연자가 아는 쪽이면 n_taught_knower도).
    teach_cost_when == "meal"이면 자격 있는 관찰이 하나라도 있었던 시연자 j가 그 스텝 한 번
    teach_cost × clamp(show_g[j], 0, 1)을 energy에서 빼 L_BRAIN·GT_TEACH_COST에 더한다(drift 제외, boost 0이어도 자격이 있으면).
    미숙기 관찰자는 앎 도우미 _note_know로 know_age·know_src를, 부모 끼니를 보면 n_seen_par(독이면 n_seen_par_poison)를 센다
    (juv_enabled나 juv_measure일 때; 세계 동작을 안 바꾸는 읽기 전용 계수).
    """
    n = s.x.shape[0]
    gw = ip[Q_GW]
    gh = ip[Q_GH]
    w = fp[P_W]
    hgt = fp[P_H]
    inv_cell = fp[P_INV_CELL]
    rad = fp[P_SOC_RADIUS]
    rad2 = rad * rad
    rate0 = fp[P_SOC_RATE]
    mode = ip[Q_SOC_MODE]
    juv_count = ip[Q_JUV_COUNT] != 0
    juv_steps = ip[Q_JUV_STEPS]
    teach = ip[Q_TEACH] != 0
    tmode = ip[Q_TEACH_MODE]
    meal_cost = teach and ip[Q_TEACH_WHEN] == WHEN_MEAL and tmode != TEACH_DRIFT
    kp = _poison_slot(step, ip)
    for j in range(n):
        if not s.alive[j] or s.meal_o[j] == 0.0:
            continue
        xj = s.x[j]
        yj = s.y[j]
        cx = int(xj * inv_cell)
        cy = int(yj * inv_cell)
        if cx >= gw:
            cx = gw - 1
        if cy >= gh:
            cy = gh - 1
        o_j = s.meal_o[j]
        mc_j = s.meal_k[j]
        j_adult = s.age[j] > juv_steps
        j_knows = False
        if teach and j_adult:
            j_knows = _knows_now(s, j, kp, fp)
        served = False
        for oy in range(-1, 2):
            # 격자가 3칸보다 좁으면 같은 칸을 두 번 훑지 않게 겹치는 쪽을 건너뛴다
            if (oy == 1 and gh <= 2) or (oy == -1 and gh == 1):
                continue
            yy = (cy + oy) % gh
            for ox in range(-1, 2):
                if (ox == 1 and gw <= 2) or (ox == -1 and gw == 1):
                    continue
                xx = (cx + ox) % gw
                c = yy * gw + xx
                for p in range(s.a_start[c], s.a_start[c + 1]):
                    i = s.a_items[p]
                    if i == j or not s.alive[i]:
                        continue
                    ddx = _wdiff(s.x[i] - xj, w)
                    ddy = _wdiff(s.y[i] - yj, hgt)
                    if ddx * ddx + ddy * ddy > rad2:
                        continue
                    s.n_seen[i] += 1
                    if o_j < 0.0:
                        s.n_seen_poison[i] += 1
                    kc = _kin_class(s, i, j)
                    if kc != 0:
                        s.n_seen_kin[i] += 1
                    s.seen_age_sum[i] += s.age[j]
                    i_juv = juv_count and s.age[i] <= juv_steps
                    if i_juv and kc == 1:
                        s.n_seen_par[i] += 1
                        if o_j < 0.0:
                            s.n_seen_par_poison[i] += 1
                    # 보여 주기 자격과 boost(시연자 어른, 관찰자 미숙기)
                    boost = 0.0
                    if teach and j_adult and s.age[i] <= juv_steps:
                        if tmode == TEACH_ANY:
                            elig = True
                        elif tmode == TEACH_KNOWER:
                            elig = kc == 1 and j_knows
                        else:
                            elig = kc == 1
                        if elig:
                            served = True
                            if tmode != TEACH_COSTONLY and tmode != TEACH_DRIFT:
                                g = s.show_g[j]
                                if g < 0.0:
                                    g = 0.0
                                elif g > 1.0:
                                    g = 1.0
                                boost = fp[P_TEACH_GAIN] * g
                    if mode == SOC_INERT:
                        continue        # inert: 기억을 안 바꾼다
                    rate = rate0 + s.soc_g[i] + boost
                    if rate < 0.0:
                        rate = 0.0
                    elif rate > 1.0:
                        rate = 1.0
                    o = o_j
                    if mode == SOC_WHAT:
                        o = 1.0
                    mc = mc_j
                    if mode == SOC_BLIND:
                        # 눈가림: 같은 관찰·같은 결과·같은 속도, 모양 칸만 (스텝, 관찰자·시연자 쌍)의 동전으로
                        mc = int(u01(ip[Q_SEED], S_SOC_BLIND, step, soc_pair_key(s.ids[i], s.ids[j])) * 2.0)
                    s.taste_m[i, mc] += rate * (o - s.taste_m[i, mc])
                    if boost > 0.0:
                        s.n_taught[i] += 1
                        if j_knows:
                            s.n_taught_knower[i] += 1
                    if i_juv:
                        # 앎 도우미: 부모 끼니 2, 그 밖의 친족 3, 남 4
                        src = 4
                        if kc == 1:
                            src = 2
                        elif kc == 2:
                            src = 3
                        _note_know(s, i, kp, src, fp)
        if meal_cost and served and s.show_g[j] > 0.0:
            g = s.show_g[j]
            if g > 1.0:
                g = 1.0
            cost = fp[P_TEACH_COST] * g
            s.energy[j] -= cost
            s.ledger[L_BRAIN] += cost
            row = s.tag[j]
            if row < 0:
                row = 0
            elif row > TAG_ROWS - 1:
                row = TAG_ROWS - 1
            s.grad_tab[row, GT_TEACH_COST] += cost


@njit(cache=True, inline="always")
def _kin_class(s, i, j):
    """i가 본 j의 친족 분류: 0 남, 1 j가 i의 부모, 2 그 밖의 친족(i가 j의 부모, 같은 부모 >= 0)."""
    if s.parent[i] == s.ids[j]:
        return 1
    if s.parent[j] == s.ids[i] or (s.parent[i] >= 0 and s.parent[i] == s.parent[j]):
        return 2
    return 0


@njit(cache=True, inline="always")
def _note_know(s, i, kp, src, fp):
    """새끼 i가 독 모양(칸 kp)을 처음 "안" 때만 그 나이와 출처(src)를 적는다(한 번만; 읽기 전용 계수)."""
    if kp >= 0 and s.know_age[i] < 0 and s.taste_m[i, kp] <= -fp[P_JUV_KNOW_THETA]:
        s.know_age[i] = s.age[i]
        s.know_src[i] = src


@njit(cache=True, inline="always")
def _poison_slot(step, ip):
    """이 스텝에 독인 모양의 맛 기억 칸(= 1 - good_kind). 먹이 종류가 꺼진 세계는 -1(독 칸이 없다)."""
    if ip[Q_KINDS] == 0:
        return -1
    return 1 - good_kind(ip[Q_KIND_SEED], step + ip[Q_KIND_STEP0], ip[Q_KIND_PERIOD])


@njit(cache=True, inline="always")
def _knows(s, i, kp, fp):
    """새끼 i가 독 모양(칸 kp)을 "아는가": 기억이 -juv_know_theta 아래. 한 번 안 새끼(know_age >= 0)는 계속 안다(누적)."""
    if kp < 0:
        return False
    return s.know_age[i] >= 0 or s.taste_m[i, kp] <= -fp[P_JUV_KNOW_THETA]


@njit(cache=True, inline="always")
def _knows_now(s, j, kp, fp):
    """시연자 j(어른)가 그 스텝 독 모양(칸 kp)을 "아는가": 그 스텝 기억만 본다(누적 know_age는 안 봄)."""
    if kp < 0:
        return False
    return s.taste_m[j, kp] <= -fp[P_JUV_KNOW_THETA]


@njit(cache=True)
def juv_pass(step, s, fp, ip):
    """미숙기 새끼의 곁 머무름 계수기와 졸업표(읽기 전용 측정: 세계 동작을 안 바꾼다).

    순차 전용, prange 금지. 격자를 안 쓰고 산 개체를 선형으로 훑는다(칸 경계 0.5 안쪽 이웃을 놓치지 않고
    s.a_start·s.a_items를 안 건드린다). O(J×N). eat 뒤에 부른다: 나이는 move가 이미 올린 값이다.
    기록 조건 = 부모가 있고 태어난 스텝부터 계수기를 쭉 센 새끼(juv_alive_steps == age)만 졸업표에 든다.
    """
    n = s.x.shape[0]
    steps = ip[Q_JUV_STEPS]
    w = fp[P_W]
    hgt = fp[P_H]
    r2 = fp[P_SOC_RADIUS] * fp[P_SOC_RADIUS]
    kp = _poison_slot(step, ip)
    for i in range(n):
        s.par_slot[i] = -1
    for i in range(n):
        if not s.alive[i] or s.age[i] > steps:
            continue
        s.juv_alive_steps[i] += 1
        p = s.parent[i]
        q = -1
        near = False
        n_adult = 0
        for j in range(n):
            if j == i or not s.alive[j]:
                continue
            ddx = _wdiff(s.x[j] - s.x[i], w)
            ddy = _wdiff(s.y[j] - s.y[i], hgt)
            close = ddx * ddx + ddy * ddy <= r2
            if p >= 0 and s.ids[j] == p:
                q = j
                if close:
                    near = True
            elif close and s.age[j] > steps:
                n_adult += 1
        s.par_slot[i] = q
        if q >= 0:
            s.par_alive_steps[i] += 1
            if near:
                s.par_near_steps[i] += 1
        s.adult_near_sum[i] += n_adult
        if p < 0 or s.juv_alive_steps[i] != s.age[i]:
            continue
        row = s.tag[i]
        if row < 0:
            row = 0
        elif row > TAG_ROWS - 1:
            row = TAG_ROWS - 1
        a = s.age[i]
        if a == steps // 4 or a == steps // 2:
            if kp >= 0:
                if a == steps // 4:
                    c_n, c_m, c_k = GT_N1, GT_M1, GT_K1
                else:
                    c_n, c_m, c_k = GT_N2, GT_M2, GT_K2
                s.grad_tab[row, c_n] += 1.0
                s.grad_tab[row, c_m] += -s.taste_m[i, kp]
                if _knows(s, i, kp, fp):
                    s.grad_tab[row, c_k] += 1.0
        if a == steps:
            tab = s.grad_tab
            tab[row, GT_GRAD] += 1.0
            if kp >= 0:
                tab[row, GT_M3] += -s.taste_m[i, kp]
                if _knows(s, i, kp, fp):
                    tab[row, GT_K3] += 1.0
            src = s.know_src[i]
            if src >= 1 and src <= 4:
                tab[row, GT_SRC1 + src - 1] += 1.0
            tab[row, GT_PAR_NEAR] += s.par_near_steps[i]
            tab[row, GT_PAR_ALIVE] += s.par_alive_steps[i]
            tab[row, GT_ADULT_NEAR] += s.adult_near_sum[i]
            tab[row, GT_ALIVE_STEPS] += s.juv_alive_steps[i]
            tab[row, GT_SEEN] += s.n_seen[i]
            tab[row, GT_SEEN_PAR] += s.n_seen_par[i]
            tab[row, GT_SEEN_PAR_POISON] += s.n_seen_par_poison[i]
            tab[row, GT_TAUGHT] += s.n_taught[i]
            tab[row, GT_TAUGHT_KNOWER] += s.n_taught_knower[i]
            tab[row, GT_MEALS] += s.n_meal[i]
            tab[row, GT_POISON_MEALS] += s.n_poison[i]
            tab[row, GT_E_GRAD] += s.energy[i]


@njit(cache=True)
def teach_pass(step, s, fp, ip):
    """보여 주기 비용(teach_cost_when == "near"): 어른 시연자가 곁의 미숙기 새끼 때문에 스텝당 한 번 내는 비용.

    순차 전용, prange 금지. 격자를 안 쓴다(juv_pass가 채운 par_slot과 선형 탐색). juv_pass 뒤에 부른다.
    - kin·costonly·drift·knower: 어른 p가 자기 미숙기 새끼 i(par_slot[i] == p)가 soc_radius 안일 때 표시
      (knower는 p가 독 모양을 알 때만). any: 곁(soc_radius 안)의 미숙기 새끼마다 그 곁의 어른 모두를 표시.
    - 표시된 어른 q(칸 번호 오름차순)가 drift가 아니고 show_g[q] > 0이면 c = teach_cost × clamp(show_g[q], 0, 1)을
      energy에서 빼 L_BRAIN에 더한다(새 에너지원 없음) + 태그별 GT_TEACH_COST. 새끼가 몇이든 어른 한 번.
    - 미숙기(age <= juv_steps) 개체는 시연자가 아니다.
    """
    n = s.x.shape[0]
    steps = ip[Q_JUV_STEPS]
    mode = ip[Q_TEACH_MODE]
    w = fp[P_W]
    hgt = fp[P_H]
    r2 = fp[P_SOC_RADIUS] * fp[P_SOC_RADIUS]
    kp = _poison_slot(step, ip)
    for q in range(n):
        s.tflag[q] = 0
    for i in range(n):
        if not s.alive[i] or s.age[i] > steps:
            continue
        if mode == TEACH_ANY:
            for j in range(n):
                if j == i or not s.alive[j] or s.age[j] <= steps:
                    continue
                ddx = _wdiff(s.x[j] - s.x[i], w)
                ddy = _wdiff(s.y[j] - s.y[i], hgt)
                if ddx * ddx + ddy * ddy <= r2:
                    s.tflag[j] = 1
        else:
            q = s.par_slot[i]
            if q < 0 or not s.alive[q] or s.age[q] <= steps:
                continue
            if mode == TEACH_KNOWER and not _knows_now(s, q, kp, fp):
                continue
            ddx = _wdiff(s.x[q] - s.x[i], w)
            ddy = _wdiff(s.y[q] - s.y[i], hgt)
            if ddx * ddx + ddy * ddy <= r2:
                s.tflag[q] = 1
    if mode == TEACH_DRIFT:
        return
    for q in range(n):
        if s.tflag[q] != 1 or s.show_g[q] <= 0.0:
            continue
        g = s.show_g[q]
        if g > 1.0:
            g = 1.0
        c = fp[P_TEACH_COST] * g
        s.energy[q] -= c
        s.ledger[L_BRAIN] += c
        row = s.tag[q]
        if row < 0:
            row = 0
        elif row > TAG_ROWS - 1:
            row = TAG_ROWS - 1
        s.grad_tab[row, GT_TEACH_COST] += c


@njit(cache=True)
def feed_pass(step, s, fp, ip):
    """먹이 대주기(돌봄 대조): 어른 p가 닿는 거리(2 x agent_radius + 0.5) 안의 자기 미숙기 새끼 가운데
    가장 가까운 하나에게 스텝당 feed_rate x clamp(feed_g[p], 0, 1)을 건넨다.

    순차 전용, prange 금지. 격자를 안 쓴다(juv_pass가 채운 par_slot과 선형 탐색). juv_pass 뒤에 부른다.
    stages.give와 같은 식: 받는 쪽은 (1 - give_loss) 배, 손실은 L_LOST(새 에너지원 없음), 주는 쪽 에너지가
    give_min_energy 이하이면 못 준다(그 위에서는 주고 나서 아래로 내려갈 수 있다). 받는 쪽 e_max 상한은 없다.
    가장 가까운 새끼가 같은 거리면 칸 번호가 낮은 쪽. 어른은 칸 번호 오름차순으로 건넨다. 태그별 GT_FEED_GIVEN.
    teach·soc 장치와 무관하게 동작한다(juv_enabled만 요구).
    """
    n = s.x.shape[0]
    steps = ip[Q_JUV_STEPS]
    w = fp[P_W]
    hgt = fp[P_H]
    reach = 2.0 * fp[P_A_RAD] + 0.5
    reach2 = reach * reach
    for q in range(n):
        s.jscr_i[q] = -1
        s.jscr_d[q] = 1e30
    for i in range(n):
        if not s.alive[i] or s.age[i] > steps:
            continue
        q = s.par_slot[i]
        if q < 0 or not s.alive[q] or s.age[q] <= steps or s.feed_g[q] <= 0.0:
            continue
        ddx = _wdiff(s.x[q] - s.x[i], w)
        ddy = _wdiff(s.y[q] - s.y[i], hgt)
        d2 = ddx * ddx + ddy * ddy
        if d2 <= reach2 and d2 < s.jscr_d[q]:
            s.jscr_d[q] = d2
            s.jscr_i[q] = i
    for q in range(n):
        i = s.jscr_i[q]
        if i < 0 or s.energy[q] <= fp[P_FEED_MIN]:
            continue
        g = s.feed_g[q]
        if g > 1.0:
            g = 1.0
        amt = fp[P_FEED_RATE] * g
        got = amt * (1.0 - fp[P_FEED_LOSS])
        s.energy[q] -= amt
        s.energy[i] += got
        s.ledger[L_LOST] += amt - got
        row = s.tag[q]
        if row < 0:
            row = 0
        elif row > TAG_ROWS - 1:
            row = TAG_ROWS - 1
        s.grad_tab[row, GT_FEED_GIVEN] += amt


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
    s.a_held[j] = -1
    s.taste_m[j, 0] = 0.0
    s.taste_m[j, 1] = 0.0
    s.taste_g[j] = 0.0
    s.n_meal[j] = 0
    s.n_poison[j] = 0
    s.n_poison10[j] = 0
    s.soc_g[j] = 0.0
    s.show_g[j] = 0.0
    s.feed_g[j] = 0.0
    s.meal_k[j] = 0
    s.meal_o[j] = 0.0
    s.n_seen[j] = 0
    s.n_seen_poison[j] = 0
    s.n_seen_kin[j] = 0
    s.seen_age_sum[j] = 0
    # 부모의 학습 보조 투자 계수(저장 안 함): 앎은 -1("아직 모름"), 부모 칸도 -1(없음), 나머지는 0
    s.know_age[j] = -1
    s.know_src[j] = 0
    s.n_seen_par[j] = 0
    s.n_seen_par_poison[j] = 0
    s.n_taught[j] = 0
    s.n_taught_knower[j] = 0
    s.par_near_steps[j] = 0
    s.par_alive_steps[j] = 0
    s.juv_alive_steps[j] = 0
    s.adult_near_sum[j] = 0
    s.par_slot[j] = -1
    s.jscr_i[j] = 0
    s.jscr_d[j] = 0.0
    s.tflag[j] = 0
    # 규범: 표시·벌·삼가기 기억은 물려받지 않는다(norm_mem_inherit는 과제 19). 피해자는 −2(미지)
    s.mark_t[j] = -1
    s.pun_t[j] = -1
    s.mem[j] = 0.0
    s.mark_vslot[j] = -2
    s.mark_vid[j] = -2
    s.mark_pun[j] = 0
    s.pun_role[j] = -1
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
    # 맛 배움 속도 유전자(맨 뒤에 덧붙임, 자기 난수 용도라 앞의 흐름은 그대로). 장치가 꺼져 있으면 변이 없이 베낀다.
    # 선형 변이(0 근처가 덫이 되지 않게), 실제 속도가 [0, 1] 안에 들게 [−taste_rate, 1 − taste_rate]로 자른다
    g = float(s.taste_g[i])
    if mut and ip[Q_TASTE_GENE] != 0:
        g += fp[P_TASTE_GENE_SIGMA] * normal(seed, S_TASTE_GENE, cid, 0)
        lo = -fp[P_TASTE_RATE]
        hi = 1.0 - fp[P_TASTE_RATE]
        if g < lo:
            g = lo
        elif g > hi:
            g = hi
    s.taste_g[j] = g
    # 보고 배우기 속도 유전자(맨 뒤, 자기 난수 용도 S_SOC_GENE라 앞의 흐름은 그대로). 꺼져 있으면 변이 없이 베낀다.
    # 실제 속도 = clamp(soc_rate + soc_g, 0, 1)이라 soc_g를 [−soc_rate, 1 − soc_rate]로 자른다
    g2 = float(s.soc_g[i])
    if mut and ip[Q_SOC] != 0:
        g2 += fp[P_SOC_GENE_SIGMA] * normal(seed, S_SOC_GENE, cid, 0)
        lo2 = -fp[P_SOC_RATE]
        hi2 = 1.0 - fp[P_SOC_RATE]
        if g2 < lo2:
            g2 = lo2
        elif g2 > hi2:
            g2 = hi2
    s.soc_g[j] = g2
    # 보여 주기·먹이 대주기 강도 유전자(맨 뒤, 자기 난수 용도 S_TEACH_GENE·S_FEED_GENE). 꺼져 있으면 변이 없이 베낀다.
    # 강도는 [0, 1]이라 그 안으로 자른다. (cross_genome이 생기면 show_g·feed_g는 낱값 유전자로 변이 목록과 섞기 목록 둘 다에 넣는다)
    g3 = float(s.show_g[i])
    if mut and ip[Q_TEACH] != 0:
        g3 += fp[P_TEACH_GENE_SIGMA] * normal(seed, S_TEACH_GENE, cid, 0)
        if g3 < 0.0:
            g3 = 0.0
        elif g3 > 1.0:
            g3 = 1.0
    s.show_g[j] = g3
    g4 = float(s.feed_g[i])
    if mut and ip[Q_FEED] != 0:
        g4 += fp[P_FEED_GENE_SIGMA] * normal(seed, S_FEED_GENE, cid, 0)
        if g4 < 0.0:
            g4 = 0.0
        elif g4 > 1.0:
            g4 = 1.0
    s.feed_g[j] = g4


# ---- 죽음 ----
@njit(cache=True)
def _remove(s, i, cause, step, recycle, nd, field_cap):
    s.alive[i] = False
    s.counters[C_NALIVE] -= 1
    j = s.a_held[i]
    if j >= 0:
        # 들고 있던 물건은 죽은 자리에 내려놓는다(f_dropped는 들 때 이미 1이다)
        s.fx[j] = s.x[i]
        s.fy[j] = s.y[i]
        s.f_holder[j] = -1
        s.a_held[i] = -1
    e = s.energy[i]
    # 사체는 들판 아닌 물건이다. 들판 알만 세는 세계에서는 저장 칸에 자리가 있어야 남는다
    if (recycle and e > 0.0 and (field_cap == 0 or store_room(s, field_cap))
            and _place_food(s, s.x[i], s.y[i], e, KIND_CORPSE, step, field_cap)):
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
            # 미숙기 사망 기록(졸업률의 분모). 나이 == juv_steps인 새끼는 이 스텝 juv_pass가 이미 졸업으로 세었으니 뺀다.
            # 노쇠 사망은 미숙기일 수 없어 안 넣는다. strike(운석) 사망은 안 센다
            if (ip[Q_JUV_COUNT] != 0 and s.parent[i] >= 0 and s.age[i] < ip[Q_JUV_STEPS]
                    and s.juv_alive_steps[i] == s.age[i]):
                row = s.tag[i]
                if row < 0:
                    row = 0
                elif row > TAG_ROWS - 1:
                    row = TAG_ROWS - 1
                s.grad_tab[row, GT_DEAD] += 1.0
            if s.touch[i] == TOUCH_HIT:
                # 이번 스텝에 공격을 맞았고 에너지가 바닥났다. 굶어 죽음과 따로 센다
                # (맞은 뒤 제 공격 비용으로 바닥난 드문 경우도 여기에 든다)
                _remove(s, i, DEATH_KILLED, step, False, nd, ip[Q_FIELD_CAP])
                s.counters[C_DEATH_KILLED] += 1
            else:
                _remove(s, i, DEATH_STARVE, step, False, nd, ip[Q_FIELD_CAP])
                s.counters[C_DEATH_STARVE] += 1
            nd += 1
        elif s.age[i] >= ip[Q_MAX_AGE]:
            _remove(s, i, DEATH_AGE, step, corpse, nd, ip[Q_FIELD_CAP])
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
    if kind == BIRTH_NATURAL and ip[Q_TASTE_IN0] >= 0:
        # 자연 출생만 부모의 맛 기억을 물려받는다(재주입·대조 새끼는 _reset_body가 지운 0 그대로)
        inh = fp[P_TASTE_INHERIT]
        s.taste_m[j, 0] = inh * s.taste_m[i, 0]
        s.taste_m[j, 1] = inh * s.taste_m[i, 1]
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
        _remove(s, i, DEATH_PASSIVE, step, False, nd, ip[Q_FIELD_CAP])
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
        _remove(s, i, DEATH_DISASTER, step, False, nd + killed, ip[Q_FIELD_CAP])
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
            _drop_food_slot(s, j, ip[Q_FIELD_CAP])
            s.ledger[L_LOST] += s.f_energy[j]
    return killed
