"""설정. 모든 값은 임시 결정(사용자 확인 전)이고 여기서만 바꾼다.

근거는 `reports/인공생명 1단계 구현 결정.md`와 각 티켓에 있다.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields


@dataclass
class Config:
    # --- 실행 ---
    seed: int = 1
    threads: int = 0              # 0이면 numba 기본값
    max_steps_per_sec: float = 0.0  # 0이면 제한 없음

    # --- 세계 (T08) ---
    world_w: float = 512.0        # 반대편으로 이어지는 세계(torus)
    world_h: float = 512.0
    cell: float = 8.0             # 이웃 탐색 격자 한 칸. world_w, world_h는 cell의 배수

    # --- 먹이 (T08) ---
    food_inflow: float = 9.0      # 스텝당 세계에 들어오는 먹이 에너지
    food_energy: float = 1.0      # 먹이 한 알의 에너지
    food_cap: int = 2000          # 먹이 알 수 상한(넘치는 유입은 버린다). 개체군 저장량보다 작게
    food_cap_replace: bool = False  # 상한에 닿으면 유입을 버리는 대신 가장 오래된 먹이를 새 먹이로 바꾼다
    food_radius: float = 0.5
    n_patches: int = 16           # 먹이가 뭉쳐 나는 자리 수
    patch_sigma: float = 18.0     # 자리 하나의 퍼짐
    patch_life: int = 20000       # 자리가 다른 곳으로 옮겨 가는 주기(스텝)
    food_uniform_frac: float = 0.15  # 자리와 무관하게 아무 데나 나는 비율

    # --- 개체 수 (T09) ---
    max_agents: int = 3000        # 배열 크기 = 계산 보호용 상한. 꽉 차면 출생이 막힌다.
                                  # 먹이 유입 ÷ 최소 소비(약 0.0031)보다 크게 잡아 평소에는 닿지 않게 한다
    init_agents: int = 400
    min_agents: int = 100         # 이 아래로 떨어지면 살아 있는 개체의 변이 복제본을 넣는다(횟수 기록)

    # --- 몸과 움직임 (T10) ---
    agent_radius: float = 1.0
    thrust_force: float = 0.1     # 추력 출력 1일 때 스텝당 가속
    back_frac: float = 0.5        # 뒤로 갈 때 힘 비율
    drag: float = 0.1             # 스텝마다 속도에 곱하는 (1 - drag)
    v_max: float = 1.0            # 속도 상한(터널링 방지: 반지름 이하)
    max_turn: float = 0.3         # 회전 출력 1일 때 스텝당 방향 변화(rad)
    collide_max_push: float = 0.5  # 겹침을 한 스텝에 밀어내는 거리 상한

    # --- 에너지 (T09, T10, T13) ---
    e_start: float = 2.0          # 처음·재주입 개체의 에너지
    e_max: float = 8.0            # 이 이상이면 더 먹지 않는다
    e_repro: float = 4.0          # 번식 문턱
    child_frac: float = 0.5       # 번식 때 부모 에너지 중 자식에게 가는 비율
    birth_cost: float = 0.3       # 번식 때 사라지는 에너지
    basal_cost: float = 0.003     # 스텝당 기초 대사
    brain_cost: float = 0.006     # 스텝당 뉴런 유지비 = brain_cost * (활성 뉴런 수 / h_max). 최대 뇌가 기초 대사의 2배
    synapse_cost: float = 0.0006  # 스텝당 시냅스 유지비 = synapse_cost * (활성 시냅스 수 / 최대 시냅스 수)
    brain_activity_cost: float = 0.0  # 스텝당 활동 비용 = 이 값 * 평균 |활성| (기본은 끔)
    thrust_cost: float = 0.006    # 스텝당 이동 비용 = thrust_cost * |추력|. 전속이 기초 대사의 2배
    back_cost_mult: float = 1.5   # 뒤로 갈 때 이동 비용 배율
    turn_cost: float = 0.0005     # 스텝당 회전 비용 = turn_cost * |회전|
    max_age: int = 3000           # 최대 수명(스텝). 가득 찬 에너지로 굶기까지가 수명의 약 0.3배
    corpse_to_food: bool = True   # 늙어 죽으면 남은 에너지가 먹이로 남는다

    # --- 번식 (T09) ---
    repro_brain_controlled: bool = True  # 번식 출력이 0보다 클 때만 번식
    repro_min_age: int = 200

    # --- 감각 (T11) ---
    vision_sectors: int = 8       # 시야를 나눈 칸 수(낮은 해상도 = 흐릿함)
    vision_fov: float = 2.094     # 시야각(rad), 120도
    vision_range: float = 16.0
    vision_noise: float = 0.0     # 시야 값에 더하는 잡음 표준편차(기본은 끔: 흐릿함은 낮은 해상도로 만든다)
    smell_length: float = 12.0    # 냄새가 줄어드는 거리 척도
    smell_gain: float = 0.25
    n_spare_in: int = 13          # 2~4단계 입력 자리: 호르몬 4, 듣기 4, 혈연 1, 이웃이 먹음 1, 흔적 3

    # --- 뇌 (T12) ---
    h_max: int = 64               # 은닉 뉴런 최대 수(배열 크기)
    h_init: int = 16              # 처음 개체의 활성 뉴런 수
    h_min: int = 1
    n_spare_out: int = 6          # 2~4단계 출력 자리: 잠, 신호 2, 주기, 쌓기, 예비 1
    init_w_in: float = 0.1
    init_w_rec: float = 0.1
    init_w_out: float = 0.1
    init_bias_thrust: float = 1.0  # 처음에는 앞으로 가려는 쪽으로 기울여 둔다
    init_bias_repro: float = 1.0   # 처음에는 번식하려는 쪽으로 기울여 둔다

    # --- 변이 (T12) ---
    mutation_enabled: bool = True
    sigma_init: float = 0.02      # 가중치 변이 크기(유전자 자리는 있지만 기본은 고정)
    sigma_min: float = 0.003
    sigma_max: float = 0.2
    sigma_tau: float = 0.0        # 변이 크기 자체가 변하는 정도(log 정규). 0이면 고정
    p_hidden_change: float = 0.03  # 출생 때 활성 뉴런 수가 +-1 될 확률
    color_sigma: float = 0.02     # 색 유전자 변이(화면에서 계통을 구분하는 용도)

    # --- 2단계: 마음 (기본은 꺼짐) ---
    mind_enabled: bool = False    # 호르몬, 보상으로 조절되는 평생 학습, 호기심, 잠
    eta_init: float = 0.01        # 처음 개체의 학습률 유전자
    eta_max: float = 0.2
    alpha_init: float = 0.05      # 처음 개체의 뉴런별 가소성 유전자 표준편차
    trace_decay: float = 0.8      # 활동 흔적이 남는 정도(스텝마다 곱함)
    hebb_decay: float = 0.001     # 배운 것이 스텝마다 흐려지는 정도
    plast_cost: float = 0.002     # 스텝당 학습 비용 = plast_cost * (학습률 / eta_max) * (활성 뉴런 수 / h_max)
    n_proto: int = 8              # 개체가 기억하는 감각 패턴 수
    proto_lr: float = 0.05        # 패턴이 새 경험 쪽으로 움직이는 정도
    link_cap: float = 20.0        # 패턴 사이 연결 세기 상한
    hormone_rate_hunger: float = 0.02
    hormone_rate_stress: float = 0.05
    hormone_rate_dopamine: float = 0.2
    hormone_rate_boredom: float = 0.01
    reward_baseline_rate: float = 0.05  # 기대 보상이 따라가는 속도
    fatigue_rate: float = 0.002   # 깨어 있을 때 스텝당 쌓이는 피로
    sleep_recover: float = 0.01   # 잘 때 스텝당 풀리는 피로
    sleep_min_fatigue: float = 0.3  # 이보다 덜 피곤하면 잠들 수 없다
    sleep_basal: float = 0.6      # 자는 동안 기초 대사 배율
    sleep_forget: float = 0.01    # 자는 동안 약한 학습 흔적이 스텝마다 지워지는 정도
    sleep_keep: float = 0.3       # 이보다 센 학습 흔적은 자는 동안 지워지지 않는다
    perturb_sigma: float = 0.0    # 0보다 크면 학습 규칙이 노드 섭동으로 바뀐다: 은닉 뉴런이 움직임 출력(추력, 회전)으로
                                  # 보내는 값을 이 크기의 잡음으로 흔들고, 흔든 뒤 보상이 오면 그쪽으로 배운다.
                                  # 잠·번식·신호 출력은 흔들지 않는다. 0이면 헤브 학습(기본).
                                  # 실행 중에 0과 양수 사이를 넘나들면(각본 포함) 학습 흔적이 0에서 다시 시작한다

    # --- 3단계: 사회 (기본은 꺼짐) ---
    social_enabled: bool = False  # 신호 내기·듣기, 혈연 알아보기, 이웃이 먹는 것 보기, 에너지 주기
    hear_range: float = 24.0
    signal_cost: float = 0.0005   # 스텝당 신호 비용 = signal_cost * (|신호1| + |신호2|)
    give_rate: float = 0.02       # 주기 출력 1일 때 스텝당 건네는 에너지
    give_loss: float = 0.1        # 건네는 중에 사라지는 비율
    give_min_energy: float = 1.0  # 이보다 에너지가 적으면 주지 못한다

    # --- 먹이 종류 (기본은 꺼짐. 어느 단계 묶음에도 들어 있지 않다) ---
    food_kinds_enabled: bool = False  # 먹이가 두 종류이고 한쪽은 독이다. 어느 쪽이 독인지는 때때로 바뀐다
    kind_flip_period: int = 1500  # 이 스텝마다 어느 종류가 독인지 다시 정한다(무작위라 안 바뀔 때도 있다)
    kind_seed: int = -1           # 어느 종류가 독인지 정하는 씨앗. -1이면 seed. 겨루기 마당이 원래 세계의 독 순서를 잇게 할 때 쓴다
    kind_step0: int = 0           # 독 순서를 셀 때 스텝에 더하는 값(겨루기 마당이 체크포인트 스텝부터 이어 세게)
    poison: float = 1.0           # 독인 먹이를 먹으면 잃는 에너지(먹이 자체의 에너지도 사라진다)
    kind_ratio: float = 0.5       # 새로 나는 먹이 가운데 음수로 보이는 종류(종류 1)의 비율. 보통 먹이와 거름 먹이에 같이 쓴다

    # --- 다툼 (기본은 꺼짐. 어느 단계 묶음에도 들어 있지 않다) ---
    conflict_enabled: bool = False  # 공격: 닿은 이웃의 에너지를 깎고 일부를 얻는다(포식·처벌의 수단)
    attack_damage: float = 0.2    # 공격 출력 1일 때 스텝당 상대가 잃는 에너지
    attack_gain: float = 0.5      # 깎은 에너지 중 공격한 쪽이 얻는 비율(나머지는 사라진다)
    attack_cost: float = 0.02     # 공격 출력 1일 때 스텝당 드는 비용(빗나가도 든다)

    # --- 혹독한 세계 장치 (기본은 꺼짐. 켜려면 확장 크기: preset=ext) ---
    season_enabled: bool = False  # 겨울: 해마다 해의 끝 winter_len 스텝 동안 풀이 줄고 기초 대사가 는다.
                                  # 해의 위상(sin, cos)이 입력 두 칸으로 들어온다.
                                  # 해 위상은 절대 스텝 기준이라, 각본이 중간에 켜면 겨울 중간에서 시작할 수 있다
    spoil_enabled: bool = False   # 부패: 난 지 spoil_after 스텝이 지난 먹이(사체 포함)가 스텝마다 spoil_rate 비율로 줄고,
                                  # spoil_min_frac * food_energy 아래로 내려가면 사라진다. 줄어든 에너지는 장부의 L_LOST로 간다.
                                  # 난 스텝(f_born)이 없던 옛 체크포인트는 0으로 채워 읽으므로, 켜자마자 모든 먹이가 오래된 것이 된다
    season_period: int = 4000     # 한 해의 길이(스텝). 임시 결정
    winter_len: int = 1200        # 겨울의 길이(해의 끝 이만큼). 해 길이의 3할. 임시 결정
    winter_mult: float = 0.3      # 겨울의 풀(먹이 유입) 배수. 겨울 밖은 해 평균이 1이 되도록 1보다 크게 잡힌다. 임시 결정
    autumn_len: int = 0           # 가을 열매철: 겨울 바로 앞 이만큼의 스텝 동안 풀이 autumn_mult배로 몰린다. 0이면 꺼짐.
                                  # 겨울이 켜져 있어야 한다. 해 평균 1은 그대로 지켜진다(나머지 철 배수가 그만큼 낮아진다)
    autumn_mult: float = 1.0      # 가을의 풀(먹이 유입) 배수. 1 이상. 지구의 가을 열매 몰림. 임시 결정
    winter_basal_mult: float = 1.5  # 겨울의 기초 대사 배수. 지구 온혈동물의 겨울/여름 1.0~1.5배
                                  # (docs/research/earth-references.md). 임시 결정
    spoil_after: int = 400        # 먹이가 난 뒤 이만큼 지나야 썩기 시작한다. 임시 결정
    spoil_rate: float = 0.00115   # 썩는 동안 스텝마다 줄어드는 비율(반감기 약 600스텝). 임시 결정
    spoil_min_frac: float = 0.05  # 남은 에너지가 food_energy의 이 비율 아래면 먹이가 사라진다. 임시 결정
    spoil_winter_mult: float = 0.2  # 겨울의 부패 속도 배수. 지구는 여름이 겨울보다 3~7배 빨리 썩는다
                                  # (docs/research/earth-references.md). 겨울이 꺼져 있으면 쓰이지 않는다. 임시 결정
    carry_enabled: bool = False   # 들기: 개체가 먹이 한 알을 들고 다니고 내려놓는다(물건 = 먹이 칸 하나, 손 하나).
                                  # 켜면 먹이 상한은 들판 알(든 적 없는 알, 사체 제외)만 센다.
                                  # 들판 아닌 물건이 max_agents칸을 채우면 들지 않은 것 가운데 가장 오래된 것이 밀려난다
    carry_cost_mult: float = 1.3  # 들고 있을 때 이동 비용 배수. 지구 참고값 없음. 임시 결정
    meat_spoil_mult: float = 3.0  # 고기는 부패 속도에 이 배수를 곱한다. 지구: 씨앗·견과 ≪ 과일 ≪ 고기
                                  # (DeVault et al. 2004 [2차]). 임시 결정
    prey_enabled: bool = False    # 큰 사냥감: 유입의 prey_inflow_frac 몫에서 태어나 먹이 자리를 떠돈다.
                                  # 켜면 먹이 상한은 들판 알만 센다. 실행 중에 끄면 산 사냥감은 사라지고(에너지는 장부의 L_LOST)
                                  # 쌓여 있던 사냥감 몫은 풀 몫으로 돌아간다
    max_animals: int = 64         # 동물 배열 크기(사냥감과 다른 동물이 함께 쓴다). 실행 중에 바꿀 수 없다
    n_prey: int = 4               # 산 사냥감 수의 상한. 드물게 두어 찾아다니게 한다. 임시 결정
    prey_hp: float = 20.0         # 사냥감 체력. 공격 피해 0.2/스텝으로 혼자 100스텝, 넷이면 25스텝에 쓰러진다. 임시 결정
    prey_regen: float = 0.05      # 쉰 뒤 스텝당 아무는 체력. 임시 결정
    prey_rest: int = 100          # 이만큼 맞지 않아야 아물기 시작한다(스텝). 임시 결정
    prey_strike: float = 0.12     # 사냥감 반격의 스텝당 총량. 그 스텝의 공격자 수로 나눠 각자 잃는다.
                                  # 혼자 100스텝이면 12 + 공격 비용 2 = 14로 얻는 것보다 크다. 임시 결정
    prey_sat_n: int = 4           # 한 스텝 피해는 공격자 이만큼의 몫까지만 쌓인다. 지구: 엘크 사냥 4마리에서 포화
                                  # (MacNulty et al. 2012, 2014). 임시 결정
    prey_meat: int = 24           # 쓰러지면 남기는 고기 알 수. 넷이 나누면 각 6(배를 채우는 양). 임시 결정
    prey_inflow_frac: float = 0.3  # 유입 가운데 사냥감 몫. 겨울에도 줄지 않아 사냥이 되면 겨울 먹이가 0.3배에서
                                  # 약 0.51배로 는다. 사냥감이 다 차 있으면 넘친 몫은 풀 배수를 곱해 풀로 간다. 임시 결정
    prey_bite: float = 0.0        # 물어뜯기: 친 개체가 입힌 피해(포화 뒤 몫)만큼 사냥감 몸 에너지의 이 비율을 곧바로 얻는다.
                                  # 뜯긴 몸은 아물 때 사냥감 몫(budget[2])에서 다시 찬다. 0이면 꺼짐(설계 일곱째 판). 임시 결정
    prey_life: int = 12000        # 사냥감 수명(스텝). 다하면 고기 없이 사라진다(장부의 L_LOST). 임시 결정
    prey_stay: int = 1500         # 한 먹이 자리에서 맴도는 스텝 수. 임시 결정
    prey_graze_speed: float = 0.1  # 맴돌 때 속도(v_max 배수). 임시 결정
    prey_travel_speed: float = 0.3  # 다른 먹이 자리로 옮겨 갈 때 속도(v_max 배수). 임시 결정
    prey_radius: float = 2.0      # 사냥감 반지름. 개체(1)보다 크다. 임시 결정, 개체 지름과 같은 크기
    prey_graze_r: float = 18.0    # 목적지에서 이 거리 안을 맴돈다. 임시 결정, 먹이 자리 퍼짐(patch_sigma)과 같은 크기
    prey_wander_turn: float = 0.3  # 맴돌 때 스텝마다 방향을 흔드는 폭(rad, ±이 값 안 균등).
                                  # 임시 결정, 개체의 한 스텝 최대 회전(max_turn)과 같은 크기
    huddle_enabled: bool = False  # 뭉쳐 자기: 겨울에만 닿은(중심 거리 2 * agent_radius + 0.5 안) 다른 개체 수 n으로
                                  # 기초 대사를 min(huddle_cap, 1 - (1 + n) ** -huddle_exp)만큼 줄인다.
                                  # 붙어 있으면 생기는 물리 효과다. 겨울(season_enabled)이 켜져 있어야 한다
    huddle_cap: float = 0.4       # 줄이는 비율의 상한. 지구: 겨울 대사 6~53% 절약, 상한 0.3~0.5
                                  # (Gilbert et al. 2010 [2차]). 임시 결정
    huddle_exp: float = 0.33      # 머릿수 지수. 지구: 한 마리 대사가 대략 N^-0.33(Gilbert et al. 2010 [2차]).
                                  # 셋이 붙으면 약 0.37, 넷 이상은 상한에서 멈춘다. 임시 결정

    # --- 맛 기억 (기본은 꺼짐. 켜려면 맛 블록 크기: preset=taste) ---
    taste_enabled: bool = False   # 개체마다 보이는 모양 둘(+로 보이는 먹이, -로 보이는 먹이)의 먹은 결과를 기억한다.
                                  # 기억으로 물든 시야 8칸과 기억 값 2칸이 확장 블록 뒤 맛 블록으로 들어온다.
                                  # 시야 칸 수가 8이어야 한다. 실행 중에 끄면 기억을 0으로 지운다
    taste_rate: float = 0.5       # 먹은 순간 기억이 결과(독 -1, 아니면 +1) 쪽으로 움직이는 비율.
                                  # 한 번이면 반, 두 번이면 3/4("한두 끼면 바뀐다"). 임시 결정
    taste_decay: float = 0.002    # 스텝마다 기억이 0 쪽으로 줄어드는 비율. 반감기 약 350스텝으로 독을 다시 정하는
                                  # 주기 1500의 4분의 1, 독이 실제로 바뀌는 평균 간격 약 3,000의 8분의 1.
                                  # 잊어야 피하던 모양을 다시 맛본다. 임시 결정
    taste_inherit: float = 0.0    # 자연 출생 때 새끼가 받는 기억 = taste_inherit * 부모 기억. 0이면 0에서 시작. 임시 결정
    # 맛 배움 속도 유전자 (기본은 꺼짐, 맛 기억이 켜져 있어야 한다. 설계 2026-11-07-individual-learning-design.md)
    taste_gene_enabled: bool = False  # 켜면 실제 배움 속도 = clamp(taste_rate + 개체의 taste_g, 0, 1).
                                      # taste_g는 물려받는 유전자(처음·옛 체크포인트는 0), 진화가 배움을 늘리고 줄일 수 있다
    taste_gene_sigma: float = 0.02    # 출생 때 taste_g에 더하는 변이 폭(선형 — 0 근처가 덫이 되지 않게). 30만 스텝(약 400세대)
                                      # 떠돌기만으로 약 0.4 퍼진다. 임시 결정
    taste_cost: float = 0.0005        # 스텝당 비용 = taste_cost × 실제 배움 속도(장치를 켰을 때만). 속도 0.5면 한 생(약 1,100스텝)
                                      # 약 0.28 — 끼니 한 번(1.0)의 4분의 1쯤, 독을 피한 이득(끼당 1.3 차이)보다 훨씬 작다. 임시 결정

    # 보고 배우기 (사회 학습, 기본은 꺼짐, 맛 기억이 켜져 있어야 한다. 설계 2026-11-07-social-learning-design.md)
    soc_taste_enabled: bool = False   # 켜면 가까이서 다른 개체의 끼니(어느 모양을 먹고 독이었나)를 보고 맛 기억이 조금 움직인다.
                                      # 속도는 개체마다 물려받는 유전자 soc_g(처음·옛 체크포인트는 0)
    soc_taste_mode: str = "outcome"   # outcome(결과를 보고 배움: 판정 대상), blind(눈가림 대조: 모양 칸을 동전으로 정함),
                                      # inert(끼니 기록·변이·비용은 있고 기억은 안 바뀜: 드리프트 기준선), what(결과는 늘 +1: 시험용)
    soc_rate: float = 0.05            # 보고 배울 때 기억이 결과 쪽으로 움직이는 비율의 바탕(직접 먹을 때 taste_rate보다 작다). 임시 결정
    soc_gene_sigma: float = 0.02      # 출생 때 soc_g에 더하는 변이 폭(taste_gene_sigma와 같은 꼴). 임시 결정
    soc_cost: float = 0.0005          # 스텝당 비용 = soc_cost × 보고 배우는 속도(장치를 켰을 때만, 뇌 유지비 칸에 더한다). 임시 결정
    soc_radius: float = 8.0           # 끼니를 볼 수 있는 거리. cell 이하여야 한다(이웃 격자 한 칸 안). 임시 결정

    # 부모의 학습 보조 투자 ("가르침" 후보, 기본은 모두 꺼짐. 설계 2026-11-26-parent-teaching-design.md)
    # 미숙기: 갓 태어난 새끼는 먹이를 덜 얻고 느리다. 연속으로 오른다(ramp = floor + (1 - floor) * min(1, a / juv_steps))
    juv_enabled: bool = False         # 켜면 미숙기의 효율·속도 상한·독 배수가 적용되고 곁 계수기·졸업표가 돈다
    juv_measure: bool = False         # 미숙기 동작은 안 바꾸고 곁 계수기·졸업표만 세는 읽기 전용 측정 모드
    juv_steps: int = 200              # 미숙기 길이(스텝). repro_min_age와 같게. 임시 결정
    juv_eat_floor: float = 0.5        # 나이 0의 들판 알 효율(얻는 에너지 = food_energy * ramp). (0, 1]. 임시 결정
    juv_move_floor: float = 0.3       # 나이 0의 속도 상한 비율(v_max * ramp). (0, 1]. 임시 결정
    juv_poison_mult: float = 1.0      # 미숙기 새끼가 독을 먹을 때 잃는 에너지 배수(1 이상). 기본은 안 건드림. 임시 결정
    juv_know_theta: float = 0.2       # "독 모양을 안다"의 문턱: 독 칸 기억이 -theta 이하. (0, 1]. 임시 결정
    juv_follow: float = 0.0           # 따름 반응(셋째 판 F1): 미숙기 새끼를 부모 쪽으로 끄는 세기 [0, 1]. 0이면 꺼짐. 임시 결정
    juv_follow_r: float = 3.0         # 부모와 이보다 멀 때만 끈다(먹는 반지름 1.5의 두 배). 임시 결정
    # 보여 주기: 어른이 끼니를 먹을 때 곁의 미숙기 관찰자가 보고 배우는 속도에 teach_gain * 강도를 더한다
    # (soc_taste_mode의 이름과 겹치지 않게 costonly·drift를 쓴다)
    teach_enabled: bool = False       # juv_enabled·soc_taste_enabled(outcome)·food_kinds_enabled가 필요하다
    teach_mode: str = "kin"           # kin(자기 새끼에게만: 판정 대상), costonly(비용만), any(아무 미숙기에게),
                                      # knower(독을 아는 부모만), drift(효과도 비용도 없음: 기준선)
    teach_gain: float = 0.5           # 강도 1일 때 관찰 속도에 더하는 몫. [0, 1]. 임시 결정
    teach_cost: float = 0.0005        # 스텝당 비용 = teach_cost * 강도(뇌 유지비 칸에 더한다). 임시 결정
    teach_cost_when: str = "near"     # near(곁에 미숙기가 있는 스텝마다), meal(boost가 붙는 끼니마다)
    teach_gene_sigma: float = 0.02    # 출생 때 show_g에 더하는 변이 폭. 임시 결정
    # 먹이 대주기(돌봄 대조): 미숙기 새끼에게 에너지를 건넨다
    feed_enabled: bool = False        # juv_enabled가 필요하다. 건네는 손실은 give_loss, 최소 에너지는 give_min_energy
    feed_rate: float = 0.02           # 강도 1일 때 스텝당 건네는 양의 바탕. 임시 결정
    feed_gene_sigma: float = 0.02     # 출생 때 feed_g에 더하는 변이 폭. 임시 결정

    # 규범의 씨앗: 가로채기와 벌주기 (기본은 모두 꺼짐·불활성. 설계 2026-11-26-norms-seed-design.md)
    # 켜려면 입력 크기 preset=norm_size(대조 갈래도 같은 크기). 값은 모두 임시 결정
    norm_measure: bool = False        # 세계 동작은 안 바꾸고 계수기만(가로챈 횟수·피해자·친족 비율·노출). conflict_enabled 필요
    norm_enabled: bool = False        # 표시와 벌 물리. conflict_enabled·social_enabled·norm_size·vision_sectors 8 필요
    norm_mode: str = "full"           # full(표시된 대상을 친 자발 공격은 벌) / inert(모든 공격이 가로채기, 위축·기억 갱신 없음)
    norm_mark_src: str = "take"       # 표시가 붙는 원천: take(가로챔) / sig(신호 2번 출력 > norm_sig_theta) / none(잡음만)
    norm_mark_tau: int = 50           # 표시가 보이는 스텝(붙은 스텝 + τ). 평균 속도 0.15 × 50 ≈ 7.5 ≈ soc_radius
    norm_mark_noise: float = 0.0      # 개체 스텝마다 엉뚱한 표시가 붙는 확률. 정보량 기울기 칸에서 0.0005·0.002
    norm_sig_theta: float = 0.5       # sig 원천의 문턱(출력은 이미 tanh를 지난 값). 유병률 맞춤 관문에서 조정
    norm_pun_return: float = 0.1      # 벌의 집행자가 얻는 몫. = attack_cost ÷ attack_damage(한 번 맞힐 때 본전). [0, attack_gain]
    norm_aim: str = "marked"          # 자발 공격의 대상: marked(닿는 표시된 이웃이 있으면 그쪽) / nearest(가장 가까운 이웃)
    norm_inhibit: float = 0.8         # 위축: 가로채기 출력 × (1 − norm_inhibit × mem). 한 번 벌 받으면 0.2배. [0, 1)
    norm_mem_tau: int = 50            # 삼가기 기억이 스텝마다 1/τ씩 준다. 표시와 같은 시간 규모
    norm_mem_rate: float = 1.0        # 벌 받으면 mem += rate × (1 − mem). 맞은 본인은 한 번에 최대
    norm_mem_obs: float = 0.2         # 곁 이웃이 벌 받는 것을 보면 mem += obs × (1 − mem). soc_rate 0.05보다 큼(벌은 드문 사건)
    norm_mem_obs_mode: str = "outcome"    # outcome(본 벌로 배움) / blind(눈가림 대조) / off(보고 배우기 없음)
    norm_mem_inherit: float = 0.0     # 새끼의 mem 시작 몫. 줄일 후보(설계 약점 9). 구현 전(과제 19)이라 0만 허용
    norm_impulse: float = 0.0         # 떨리는 손: 개체마다 L스텝 구간이 충동 상태일 확률. 켤 때 0.015(맞은 피해 ≈ 수입의 14%).
                                      # norm_enabled 없이 conflict_enabled만으로 쓸 수 있다(끔·off 대조에도 위반이 있어야 한다)
    norm_impulse_len: int = 5         # 충동 구간 길이 L. 계산의 입력이 아니라 "상습 위반자" 모형의 가정이다. 대조 1·10
    norm_mix_period: int = 0          # 섞기: 이 주기마다 개체를 무작위 자리로 옮긴다(0이면 꺼짐). 켤 때 200
    norm_mix_frac: float = 0.2        # 섞기 때 옮기는 비율. 줄일 후보(설계 약점 9)
    norm_block_tag: int = -1          # 겨루기: 이 태그의 개체는 표시된 대상을 못 친다(비용도 안 든다). -1이면 꺼짐
    norm_block3_tag: int = -1         # 겨루기: 이 태그의 개체는 피해자가 자기가 아닌 표시된 대상만 못 친다. -1이면 꺼짐

    # --- 4단계: 판이 바뀌는 세계 (기본은 꺼짐) ---
    niche_enabled: bool = False   # 다닌 자리가 길이 되고, 쌓은 거름이 먹이를 끌어온다
    wear_rate: float = 0.002      # 지나갈 때 그 칸에 남는 흔적 = wear_rate * 속도.
                                  # 세계 평균이 포화의 0.1~0.2배에 머물게 잡았다(0.02면 세계 전체가 길이 된다)
    trail_decay: float = 0.0005   # 흔적이 스텝마다 흐려지는 정도
    trail_sat: float = 5.0        # 흔적이 이만큼 쌓이면 "다 닦인 길"
    path_bonus: float = 0.5       # 다 닦인 길에서 이동 비용이 줄어드는 비율
    deposit_rate: float = 0.01    # 쌓기 출력 1일 때 스텝당 땅에 넣는 에너지
    deposit_loss: float = 0.1     # 쌓는 중에 사라지는 비율
    soil_decay: float = 0.0002    # 거름이 스텝마다 사라지는 정도
    soil_ref: float = 50.0        # 세계 전체 거름이 이만큼이면 먹이 유입의 fert_frac이 거름 땅에 난다
    fert_frac: float = 0.5
    soil_use: float = 1.0         # 먹이 한 알이 날 때 그 칸에서 줄어드는 거름(1이면 넣은 만큼만 끌어온다)

    # --- 대조 실행 (T14) ---
    passive: bool = False         # True면 자연 출생·사망을 끄고 기록된 사건을 무작위 개체에 적용

    # --- 기록 ---
    stats_every: int = 5000
    frame_every: int = 500        # 드문 프레임(전체 흐름용)
    clip_every: int = 200000      # 이 주기로 한 스텝씩 촘촘히 녹화한다(행동을 눈으로 보는 용도)
    clip_len: int = 400           # 촘촘히 녹화하는 길이(스텝)
    frames_per_segment: int = 500
    frame_ring_segments: int = 40  # 최근 이만큼만 남긴다
    keyframe_every: int = 200000   # 드문 프레임은 계속 남긴다
    ckpt_every: int = 1000000
    ckpt_keep_last: int = 12
    ckpt_keep_every: int = 50000000  # 이 배수 스텝의 체크포인트는 지우지 않는다
    event_segment_steps: int = 2000000  # 출생·사망 기록 파일을 이 스텝 수마다 새로 연다
    event_keep_segments: int = 0   # 최근 이만큼의 기록 파일만 남긴다(0이면 전부 남김)
    min_free_gb: float = 2.0       # 디스크 여유가 이보다 적으면 저장하고 멈춘다

    # ---- 파생 값 ----
    @property
    def n_in(self) -> int:
        return 2 * self.vision_sectors + N_INTERNAL + self.n_spare_in

    @property
    def n_out(self) -> int:
        return N_ACTIONS + self.n_spare_out

    @property
    def needs_ext(self) -> bool:
        """확장 입출력 칸(preset=ext)이 필요한 장치가 하나라도 켜져 있는가."""
        return (self.season_enabled or self.spoil_enabled or self.carry_enabled or self.prey_enabled
                or self.huddle_enabled)

    @property
    def field_cap(self) -> bool:
        """먹이 상한을 들판 알 수(C_NFIELD)로 재는가. 꺼져 있으면 칸 점유 수(C_NFOOD)로 잰다.
        들판 아닌 물건(든 알, 고기)이 생기는 장치가 켜져 있으면 켠다."""
        return self.carry_enabled or self.prey_enabled

    @property
    def animals_on(self) -> bool:
        """동물 배열을 쓰는 장치가 하나라도 켜져 있는가."""
        return self.prey_enabled

    @property
    def ext_size(self) -> bool:
        """입출력 자리가 확장 크기(preset=ext)인가. 계수기·예산 배열 길이가 이것을 따른다."""
        return (self.n_spare_in >= N_STAGE_IN + N_EXT_IN
                and self.n_spare_out >= N_STAGE_OUT + N_EXT_OUT)

    @property
    def ext_in0(self) -> int:
        """확장 입력 블록의 첫 입력 번호."""
        return 2 * self.vision_sectors + N_INTERNAL + N_STAGE_IN

    @property
    def taste_in0(self) -> int:
        """맛 블록의 첫 입력 번호(확장 블록 바로 뒤)."""
        return self.ext_in0 + N_EXT_IN

    @property
    def norm_in0(self) -> int:
        """규범 블록의 첫 입력 번호(맛 블록 바로 뒤)."""
        return self.taste_in0 + N_TASTE_IN

    @property
    def grid_w(self) -> int:
        return int(round(self.world_w / self.cell))

    @property
    def grid_h(self) -> int:
        return int(round(self.world_h / self.cell))

    @property
    def food_slots(self) -> int:
        return self.food_cap + self.max_agents

    def validate(self) -> None:
        gw, gh = self.grid_w, self.grid_h
        if abs(gw * self.cell - self.world_w) > 1e-9 or abs(gh * self.cell - self.world_h) > 1e-9:
            raise ValueError("world_w, world_h는 cell의 배수여야 한다")
        reach = int(math.ceil(self.vision_range / self.cell))
        if 2 * reach + 1 > min(gw, gh):
            raise ValueError("세계가 시야 범위에 비해 너무 작다")
        if self.cell < 2 * self.agent_radius or self.cell < self.agent_radius + self.food_radius:
            raise ValueError("cell은 개체 지름 이상이어야 한다")
        if self.v_max > self.agent_radius:
            raise ValueError("v_max는 개체 반지름 이하여야 한다(터널링 방지)")
        if not (1 <= self.h_min <= self.h_init <= self.h_max):
            raise ValueError("h_min <= h_init <= h_max 여야 한다")
        if self.init_agents > self.max_agents:
            raise ValueError("init_agents > max_agents")
        if not (0.0 < self.child_frac < 1.0) or self.birth_cost < 0 or self.e_repro <= 0:
            raise ValueError("child_frac은 0과 1 사이, birth_cost는 0 이상, e_repro는 양수여야 한다")
        for name in ("basal_cost", "brain_cost", "synapse_cost", "brain_activity_cost", "thrust_cost",
                     "turn_cost", "food_inflow", "food_energy", "e_start", "vision_noise"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name}은 0 이상이어야 한다")
        if not (1 <= self.winter_len < self.season_period):
            raise ValueError("winter_len은 1 이상 season_period 미만이어야 한다")
        if not (0.0 <= self.winter_mult <= 1.0) or self.winter_basal_mult < 1.0:
            raise ValueError("winter_mult는 0~1, winter_basal_mult는 1 이상이어야 한다")
        if self.autumn_len < 0:
            raise ValueError("autumn_len은 0 이상이어야 한다")
        if self.autumn_len > 0:
            if not self.season_enabled:
                raise ValueError("autumn_len은 겨울(season_enabled)이 켜져 있어야 쓸 수 있다")
            if self.autumn_len + self.winter_len >= self.season_period:
                raise ValueError("autumn_len + winter_len은 season_period 미만이어야 한다")
            if self.autumn_mult < 1.0:
                raise ValueError("autumn_mult는 1 이상이어야 한다")
            rest = (self.season_period - self.winter_len * self.winter_mult - self.autumn_len * self.autumn_mult) \
                / (self.season_period - self.winter_len - self.autumn_len)
            if rest < 0.0:
                raise ValueError("autumn_mult가 너무 커서 나머지 철의 풀 배수가 음수가 된다")
        if self.spoil_after < 0 or not (0.0 <= self.spoil_rate <= 1.0):
            raise ValueError("spoil_after는 0 이상, spoil_rate는 0~1이어야 한다")
        if not (0.0 <= self.spoil_min_frac < 1.0) or not (0.0 <= self.spoil_winter_mult <= 1.0):
            raise ValueError("spoil_min_frac는 0 이상 1 미만, spoil_winter_mult는 0~1이어야 한다")
        if self.back_cost_mult < 1.0:
            raise ValueError("back_cost_mult는 1 이상이어야 한다(환급 금지)")
        if self.carry_cost_mult < 1.0:
            raise ValueError("carry_cost_mult는 1 이상이어야 한다(들면 이동이 싸지면 안 된다)")
        if self.meat_spoil_mult < 1.0 or self.spoil_rate * self.meat_spoil_mult > 1.0:
            raise ValueError("meat_spoil_mult는 1 이상이고 spoil_rate * meat_spoil_mult는 1 이하여야 한다(에너지가 음수가 되지 않게)")
        for name in ("stats_every", "frame_every", "frames_per_segment", "frame_ring_segments",
                     "ckpt_every", "ckpt_keep_last", "patch_life", "event_segment_steps", "max_age"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name}은 1 이상이어야 한다")
        if self.vision_sectors < 1 or self.vision_sectors > 32:
            raise ValueError("vision_sectors는 1~32여야 한다")
        if self.vision_fov > math.pi:
            raise ValueError("vision_fov는 pi 이하여야 한다(뒤쪽에 걸친 물체를 놓친다)")
        if self.n_spare_in < N_STAGE_IN or self.n_spare_out < N_STAGE_OUT:
            raise ValueError(f"n_spare_in은 {N_STAGE_IN} 이상, n_spare_out은 {N_STAGE_OUT} 이상이어야 한다")
        if self.needs_ext and (self.n_spare_in < N_STAGE_IN + N_EXT_IN or self.n_spare_out < N_STAGE_OUT + N_EXT_OUT):
            raise ValueError("혹독한 세계·암수 장치는 확장 크기가 필요하다: preset=ext")
        if self.field_cap and not self.ext_size:
            raise ValueError("들판 계수기는 확장 크기 세계에서만 쓴다(계수기 칸이 모자란다)")
        if self.max_animals < 1 or not (0 <= self.n_prey <= self.max_animals):
            raise ValueError("max_animals는 1 이상, n_prey는 0 이상 max_animals 이하여야 한다")
        for name in ("prey_regen", "prey_rest", "prey_strike"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name}은 0 이상이어야 한다")
        if not (0.0 <= self.prey_bite <= 1.0):
            raise ValueError("prey_bite는 0 이상 1 이하여야 한다")
        for name in ("prey_hp", "prey_radius", "prey_graze_r"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name}은 0보다 커야 한다")
        for name in ("prey_sat_n", "prey_meat", "prey_life", "prey_stay"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name}은 1 이상이어야 한다")
        if not (0.0 <= self.prey_inflow_frac < 1.0):
            raise ValueError("prey_inflow_frac는 0 이상 1 미만이어야 한다")
        if not (0.0 < self.prey_graze_speed <= 1.0) or not (0.0 < self.prey_travel_speed <= 1.0):
            raise ValueError("prey_graze_speed, prey_travel_speed는 0보다 크고 1 이하여야 한다(v_max 배수)")
        if not (0.0 <= self.prey_wander_turn < math.pi / 2):
            raise ValueError("prey_wander_turn은 0 이상 pi/2 미만이어야 한다(맴도는 자리에 가두기가 pi/2 아래에서만 보장된다)")
        if self.huddle_enabled and not self.season_enabled:
            raise ValueError("huddle_enabled는 겨울(season_enabled)이 켜져 있어야 쓸 수 있다")
        if not (0.0 <= self.huddle_cap < 1.0) or self.huddle_exp < 0.0:
            raise ValueError("huddle_cap는 0 이상 1 미만, huddle_exp는 0 이상이어야 한다")
        if self.taste_enabled and self.n_spare_in < N_STAGE_IN + N_EXT_IN + N_TASTE_IN:
            raise ValueError("맛 기억 장치는 맛 블록 크기가 필요하다: preset=taste")
        if self.taste_enabled and self.vision_sectors != N_TASTE_VIS:
            raise ValueError(f"맛 기억 장치는 vision_sectors가 {N_TASTE_VIS}이어야 한다(맛 블록의 물든 시야 칸 수가 고정이다)")
        for name in ("taste_rate", "taste_decay", "taste_inherit"):
            if not (0.0 <= getattr(self, name) <= 1.0):
                raise ValueError(f"{name}은 0과 1 사이여야 한다")
        if self.taste_gene_enabled and not self.taste_enabled:
            raise ValueError("taste_gene_enabled는 맛 기억(taste_enabled)이 켜져 있어야 쓸 수 있다: preset=taste")
        if self.taste_gene_sigma < 0 or self.taste_cost < 0:
            raise ValueError("taste_gene_sigma와 taste_cost는 0 이상이어야 한다")
        if self.soc_taste_enabled and not self.taste_enabled:
            raise ValueError("soc_taste_enabled는 맛 기억(taste_enabled)이 켜져 있어야 쓸 수 있다: preset=taste")
        if self.soc_taste_mode not in SOC_MODES:
            raise ValueError(f"soc_taste_mode는 {sorted(SOC_MODES)} 중 하나여야 한다: {self.soc_taste_mode!r}")
        if not (0.0 <= self.soc_rate <= 1.0):
            raise ValueError("soc_rate는 0과 1 사이여야 한다")
        if self.soc_cost < 0 or self.soc_gene_sigma < 0:
            raise ValueError("soc_cost와 soc_gene_sigma는 0 이상이어야 한다")
        if not (0.0 < self.soc_radius <= self.cell):
            raise ValueError("soc_radius는 0보다 크고 cell 이하여야 한다(이웃 격자 한 칸 안)")
        if self.juv_steps < 1:
            raise ValueError("juv_steps는 1 이상이어야 한다")
        for name in ("juv_eat_floor", "juv_move_floor", "juv_know_theta"):
            if not (0.0 < getattr(self, name) <= 1.0):
                raise ValueError(f"{name}은 0보다 크고 1 이하여야 한다")
        if not (0.0 <= self.juv_follow <= 1.0):
            raise ValueError("juv_follow는 0과 1 사이여야 한다")
        if self.juv_follow > 0.0 and not self.juv_enabled:
            raise ValueError("juv_follow는 미숙기(juv_enabled)가 켜져 있어야 쓸 수 있다")
        if self.juv_follow_r <= 0.0:
            raise ValueError("juv_follow_r는 0보다 커야 한다")
        if self.juv_poison_mult < 1.0:
            raise ValueError("juv_poison_mult는 1 이상이어야 한다(1 미만이면 독이 에너지를 만든다)")
        if not (0.0 <= self.teach_gain <= 1.0):
            raise ValueError("teach_gain은 0과 1 사이여야 한다")
        for name in ("teach_cost", "teach_gene_sigma", "feed_rate", "feed_gene_sigma"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name}은 0 이상이어야 한다")
        if self.teach_mode not in TEACH_MODES:
            raise ValueError(f"teach_mode는 {sorted(TEACH_MODES)} 중 하나여야 한다: {self.teach_mode!r}")
        if self.teach_cost_when not in TEACH_WHENS:
            raise ValueError(f"teach_cost_when은 {sorted(TEACH_WHENS)} 중 하나여야 한다: {self.teach_cost_when!r}")
        if self.teach_enabled:
            if not self.juv_enabled:
                raise ValueError("teach_enabled는 미숙기(juv_enabled)가 켜져 있어야 쓸 수 있다")
            if not self.soc_taste_enabled:
                raise ValueError("teach_enabled는 보고 배우기(soc_taste_enabled)가 켜져 있어야 쓸 수 있다")
            if self.soc_taste_mode != "outcome":
                raise ValueError("teach_enabled는 soc_taste_mode가 outcome이어야 쓸 수 있다(what·blind·inert면 뜻이 없다)")
            if not self.food_kinds_enabled:
                raise ValueError("teach_enabled는 먹이 종류(food_kinds_enabled)가 켜져 있어야 쓸 수 있다: preset=kinds")
        if self.feed_enabled and not self.juv_enabled:
            raise ValueError("feed_enabled는 미숙기(juv_enabled)가 켜져 있어야 쓸 수 있다")
        self._validate_norm()
        for name in ("give_loss", "deposit_loss", "path_bonus", "fert_frac", "sleep_basal", "trace_decay",
                     "hormone_rate_hunger", "hormone_rate_stress", "hormone_rate_dopamine",
                     "hormone_rate_boredom", "reward_baseline_rate", "proto_lr", "sleep_forget",
                     "trail_decay", "soil_decay", "hebb_decay", "perturb_sigma"):
            if not (0.0 <= getattr(self, name) <= 1.0):
                raise ValueError(f"{name}은 0과 1 사이여야 한다")
        if self.eta_max <= 0 or self.sleep_recover <= 0:
            raise ValueError("eta_max와 sleep_recover는 0보다 커야 한다")
        if not (0.0 <= self.attack_gain <= 1.0):
            raise ValueError("attack_gain은 0과 1 사이여야 한다(1을 넘으면 에너지가 생긴다)")
        if self.kind_flip_period < 1:
            raise ValueError("kind_flip_period는 1 이상이어야 한다")
        if self.kind_step0 < 0 or self.kind_seed < -1:
            raise ValueError("kind_step0은 0 이상, kind_seed는 -1 이상이어야 한다")
        if not (0.0 <= self.kind_ratio <= 1.0):
            raise ValueError("kind_ratio는 0과 1 사이여야 한다")
        for name in ("soil_use", "link_cap", "give_min_energy", "alpha_init", "attack_damage", "attack_cost", "poison"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name}은 0 이상이어야 한다")
        for name in ("eta_init", "eta_max", "plast_cost", "signal_cost", "give_rate", "deposit_rate",
                     "wear_rate", "trail_decay", "soil_decay", "hebb_decay", "fatigue_rate", "sleep_recover"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name}은 0 이상이어야 한다")
        if self.n_proto < 2 or self.hear_range <= 0 or self.trail_sat <= 0 or self.soil_ref <= 0:
            raise ValueError("n_proto는 2 이상, hear_range·trail_sat·soil_ref는 양수여야 한다")
        if 2 * int(math.ceil(self.hear_range / self.cell)) + 1 > min(gw, gh):
            raise ValueError("세계가 듣는 범위에 비해 너무 작다")

    def _validate_norm(self) -> None:
        """규범의 씨앗 설정 검사. 의존 키는 기본값이 아닐 때만 요구한다(기본값은 늘 있으므로 옛 설정이 통과하게)."""
        if not (0.0 <= self.norm_pun_return <= self.attack_gain):
            raise ValueError("norm_pun_return은 0과 attack_gain 사이여야 한다(넘으면 벌이 평소 공격보다 많이 얻는다)")
        for name in ("norm_inhibit", "norm_mem_rate", "norm_mem_obs", "norm_mem_inherit", "norm_impulse",
                     "norm_mark_noise", "norm_mix_frac"):
            if not (0.0 <= getattr(self, name) <= 1.0):
                raise ValueError(f"{name}은 0과 1 사이여야 한다")
        if self.norm_inhibit >= 1.0:
            # mem = 1(맞은 직후)이면 가로채기 피해·얻는 몫이 0이 되어 비용만 내고 0 피해로 표시가 붙는다
            raise ValueError("norm_inhibit은 1보다 작아야 한다")
        for name in ("norm_mark_tau", "norm_mem_tau", "norm_impulse_len"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name}은 1 이상이어야 한다")
        if self.norm_mix_period < 0:
            raise ValueError("norm_mix_period는 0 이상이어야 한다")
        for name, table in (("norm_mode", NORM_MODES), ("norm_mark_src", NORM_SRCS), ("norm_aim", NORM_AIMS),
                            ("norm_mem_obs_mode", NORM_OBS_MODES)):
            if getattr(self, name) not in table:
                raise ValueError(f"{name}은 {sorted(table)} 중 하나여야 한다: {getattr(self, name)!r}")
        if self.norm_mem_inherit > 0.0:
            # 부모의 가르침 연결과 함께 구현한다. 그때까지 설정이 조용히 무동작하지 않게 막는다
            raise ValueError("norm_mem_inherit > 0은 아직 구현 전이다(규범 계획 과제 19)")
        if self.norm_measure and not self.conflict_enabled:
            raise ValueError("norm_measure는 공격(conflict_enabled)이 켜져 있어야 쓸 수 있다")
        if self.norm_impulse > 0.0 and not self.conflict_enabled:
            raise ValueError("norm_impulse는 공격(conflict_enabled)이 켜져 있어야 쓸 수 있다")
        if self.norm_enabled:
            if not self.conflict_enabled:
                raise ValueError("norm_enabled는 공격(conflict_enabled)이 켜져 있어야 쓸 수 있다")
            if not self.social_enabled:
                raise ValueError("norm_enabled는 사회(social_enabled)가 켜져 있어야 쓸 수 있다")
            if self.n_in < self.norm_in0 + N_NORM_IN:
                raise ValueError("norm_enabled는 규범 블록 크기가 필요하다: preset=norm_size")
            if self.vision_sectors != N_NORM_VIS:
                raise ValueError(f"norm_enabled는 vision_sectors가 {N_NORM_VIS}이어야 한다(표시 시야 칸 수가 고정이다)")
        else:
            for name, off in (("norm_mode", "full"), ("norm_mark_src", "take"), ("norm_mem_obs_mode", "outcome")):
                if getattr(self, name) != off:
                    raise ValueError(f"{name}={getattr(self, name)}은 norm_enabled가 켜져 있어야 쓸 수 있다")
            if self.norm_mark_noise > 0.0:
                raise ValueError("norm_mark_noise는 norm_enabled가 켜져 있어야 쓸 수 있다")
            if self.norm_block_tag >= 0 or self.norm_block3_tag >= 0:
                raise ValueError("norm_block_tag·norm_block3_tag는 norm_enabled가 켜져 있어야 쓸 수 있다")
        if self.norm_mix_period > 0:
            if self.carry_enabled:
                raise ValueError("norm_mix_period는 들기(carry_enabled)와 같이 못 켠다(든 먹이 위치가 낡는다)")
            if self.teach_enabled:
                raise ValueError("norm_mix_period는 보여 주기(teach_enabled)와 같이 못 켠다(곁 머무름이 깨진다)")

    # ---- 입출력 ----
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"모르는 설정: {sorted(unknown)}")
        cfg = cls(**d)
        cfg.validate()
        return cfg

    def save(self, path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1, sort_keys=True)

    @classmethod
    def load(cls, path) -> "Config":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def with_overrides(self, pairs) -> "Config":
        """["key=value", ...]를 적용한 새 설정."""
        d = self.to_dict()
        types = {f.name: str(f.type) for f in fields(self)}
        for p in pairs or []:
            k, _, v = p.partition("=")
            k = k.strip()
            if k == "preset":   # 단계 묶음: preset=stage2 처럼 쓴다
                d.update(preset(v.strip()))
                continue
            if k not in d:
                raise ValueError(f"모르는 설정: {k}")
            if types[k] == "bool":
                d[k] = v.strip().lower() in ("1", "true", "yes", "on")
            elif types[k] == "str":
                d[k] = v.strip()
            elif types[k] == "int":
                f = float(v)
                if f != int(f):
                    raise ValueError(f"{k}는 정수여야 한다: {v}")
                d[k] = int(f)
            else:
                d[k] = float(v)
        return Config.from_dict(d)


# 입력 중 시야가 아닌 것: 에너지, 나이, 전진 속도, 옆 속도, 닿음, 먹음, 냄새, 상수 1
N_INTERNAL = 8
# 출력: 추력, 회전, 번식
N_ACTIONS = 3
ACT_THRUST = 0
ACT_TURN = 1
ACT_REPRO = 2
# 2~4단계 출력(비워 둔 자리를 쓴다)
ACT_SLEEP = 3
ACT_SIG0 = 4
ACT_SIG1 = 5
ACT_GIVE = 6
ACT_DEPOSIT = 7
ACT_ATTACK = 8
N_STAGE_OUT = 6      # 잠, 신호 2, 주기, 쌓기, 공격
# 2~4단계 입력: 시야·내부 감각 뒤에 이 순서로 붙는다
IN_HORM = 0      # 스트레스, 도파민, 지루함, 피로
IN_HEAR = 4      # 신호1 앞·뒤, 신호2 앞·뒤
IN_KIN = 8       # 가장 가까운 이웃이 나와 닮은 정도
IN_SAWEAT = 9    # 가까운 이웃이 방금 먹었는가
IN_TRAIL = 10    # 흔적: 내 자리, 왼쪽 앞, 오른쪽 앞
N_STAGE_IN = 13
# 확장 블록: 기본 입출력 뒤에 늘 같은 자리에 둔다(꺼진 장치의 칸은 0). 기본 크기는 건드리지 않는다.
N_EXT_IN = 13        # 혹독한 세계 9칸(들기 1, 계절 2, 사냥감·포식자·약탈자 방향 각 2) + 암수 4칸
N_EXT_OUT = 1        # 들기
ACT_CARRY = N_ACTIONS + N_STAGE_OUT   # 9
# 확장 입력 안 자리(절대 번호 = Config.ext_in0 + EXT_*)
EXT_CARRY = 0
EXT_SEASON = 1       # sin, cos
EXT_PREY = 3
EXT_PRED = 5
EXT_SCAV = 7
EXT_SEX = 9          # 네 칸
# 맛 블록: 확장 블록 바로 뒤(절대 번호 = Config.taste_in0 + TASTE_*). 맛 기억 장치가 꺼져 있으면 칸은 0
# 보고 배우기 모드(soc_taste_mode의 이름 -> 커널이 읽는 번호)
SOC_OUTCOME, SOC_BLIND, SOC_INERT, SOC_WHAT = 0, 1, 2, 3
SOC_MODES = {"outcome": SOC_OUTCOME, "blind": SOC_BLIND, "inert": SOC_INERT, "what": SOC_WHAT}
# 보여 주기 모드(teach_mode의 이름 -> 커널이 읽는 번호). soc_taste_mode의 이름과 겹치지 않게 costonly·drift를 쓴다
TEACH_KIN, TEACH_COSTONLY, TEACH_ANY, TEACH_KNOWER, TEACH_DRIFT = 0, 1, 2, 3, 4
TEACH_MODES = {"kin": TEACH_KIN, "costonly": TEACH_COSTONLY, "any": TEACH_ANY, "knower": TEACH_KNOWER,
               "drift": TEACH_DRIFT}
# 보여 주기 비용을 치르는 때
WHEN_NEAR, WHEN_MEAL = 0, 1
TEACH_WHENS = {"near": WHEN_NEAR, "meal": WHEN_MEAL}
N_TASTE_IN = 10      # 기억으로 물든 시야 8칸 + 기억 값 2칸
TASTE_VIS = 0        # 물든 시야 여덟 칸(vision_sectors와 같은 수)
TASTE_VAL = 8        # 기억 값 두 칸: +로 보이는 먹이, -로 보이는 먹이
N_TASTE_VIS = TASTE_VAL - TASTE_VIS   # 물든 시야 칸 수(8). 맛 기억 장치는 vision_sectors가 이 수여야 한다
# 규범 블록: 맛 블록 바로 뒤(절대 번호 = Config.norm_in0 + NORM_*). 장치가 꺼져 있으면 칸은 0
N_NORM_IN = 12       # 표시 시야 8칸 + 내 밝기 + 곁의 평균 밝기 + 방금 벌 + 내 삼가기 기억
N_NORM_VIS = 8       # 표시 시야 칸 수. norm_enabled는 vision_sectors가 이 수여야 한다
NORM_VIS = 0         # 표시된 이웃의 방향별 가까움 × 밝기(여덟 칸)
NORM_SELF = 8        # 내 밝기
NORM_LOCAL = 9       # hear_range 안 이웃의 평균 밝기(거리 가중)
NORM_SAWPUN = 10     # hear_range 안에 방금(1스텝 전) 벌 받은 이웃(거리 가중 최댓값)
NORM_MEM = 11        # 내 삼가기 기억
# 규범 모드(이름 -> 커널이 읽는 번호). inert는 사회 학습의 inert와 같은 구실이라 같은 이름을 쓴다
NORM_FULL, NORM_INERT = 0, 1
NORM_MODES = {"full": NORM_FULL, "inert": NORM_INERT}
SRC_TAKE, SRC_SIG, SRC_NONE = 0, 1, 2
NORM_SRCS = {"take": SRC_TAKE, "sig": SRC_SIG, "none": SRC_NONE}
AIM_MARKED, AIM_NEAREST = 0, 1
NORM_AIMS = {"marked": AIM_MARKED, "nearest": AIM_NEAREST}
OBS_OUTCOME, OBS_BLIND, OBS_OFF = 0, 1, 2
NORM_OBS_MODES = {"outcome": OBS_OUTCOME, "blind": OBS_BLIND, "off": OBS_OFF}
FIRST_MEALS = 10     # 측정: 개체의 첫 이만큼 끼니에서 독 비율을 센다(n_poison10)
N_REWARD = 8     # 보상 특징: 먹음, 에너지 변화, 닿음, 호기심, 허기*먹음, 스트레스, 이웃이 먹음, 상수


def preset(name: str) -> dict:
    """단계 묶음. 앞 단계의 장치는 켠 채로 다음 단계를 더한다."""
    table = {
        "stage1": dict(mind_enabled=False, social_enabled=False, niche_enabled=False),
        "stage2": dict(mind_enabled=True, social_enabled=False, niche_enabled=False),
        "stage3": dict(mind_enabled=True, social_enabled=True, niche_enabled=False),
        "stage4": dict(mind_enabled=True, social_enabled=True, niche_enabled=True),
        "conflict": dict(conflict_enabled=True),     # 다른 묶음에 덧붙여 쓴다: preset=stage3 preset=conflict
        "kinds": dict(food_kinds_enabled=True),      # 먹이 두 종류와 독. 덧붙여 쓴다: preset=stage2 preset=kinds
        "ext": dict(n_spare_in=N_STAGE_IN + N_EXT_IN, n_spare_out=N_STAGE_OUT + N_EXT_OUT),  # 크기만 올린다(대조 갈래용)
    }
    table["harsh"] = {**table["ext"], "season_enabled": True, "spoil_enabled": True}  # 확장 묶음에 덧붙이므로 표 뒤에 만든다
    table["harsh2"] = {**table["harsh"], "autumn_len": 800, "autumn_mult": 2.0, "carry_enabled": True,
                       "prey_enabled": True, "huddle_enabled": True}
    # 맛 블록 크기만 올린다(대조 갈래용). 대조도 같은 크기여야 변이 난수 자리와 시냅스 유지비가 같다
    table["taste_size"] = {**table["ext"], "n_spare_in": N_STAGE_IN + N_EXT_IN + N_TASTE_IN}
    table["taste"] = {**table["taste_size"], "taste_enabled": True}
    # 규범 블록 크기만 올린다(모든 규범 갈래가 같은 크기를 쓴다: 변이 난수 자리와 시냅스 유지비가 같게)
    table["norm_size"] = {**table["taste_size"], "n_spare_in": N_STAGE_IN + N_EXT_IN + N_TASTE_IN + N_NORM_IN}
    table["norm"] = {**table["norm_size"], "norm_enabled": True}
    if name not in table:
        raise ValueError(f"모르는 preset: {name} (stage1~stage4, conflict, kinds, ext, harsh, harsh2, taste, taste_size, "
                         "norm_size, norm)")
    return table[name]
