"""세계 하나. 한 스텝의 순서와 사건 기록을 맡는다."""
from __future__ import annotations

import hashlib
import math

import numpy as np

from . import animals as an
from . import kernels as k
from . import stages as st
from .config import ACT_ATTACK, ACT_CARRY, EXT_CARRY, EXT_PREY, EXT_SEASON, Config, N_REWARD

# 2~4단계에서 새로 생긴 저장 필드. 옛 체크포인트에는 없어도 된다(그 밖의 필드가 없으면 깨진 것이다)
NEW_SINCE_STAGE1 = frozenset({"rw", "eta", "alpha", "horm", "mod", "exp_r", "e_prev", "fatigue", "asleep",
                              "epost", "epre_h", "epre_in", "Hin", "Hrec", "proto", "link", "last_proto",
                              "sig", "trail", "soil", "f_kind", "f_born", "a_held", "f_holder", "f_dropped"})

BIRTH_DTYPE = np.dtype([("id", "<i8"), ("parent", "<i8"), ("step", "<i8"),
                        ("gen", "<i4"), ("n_hidden", "<i4"), ("kind", "<i4")])
DEATH_DTYPE = np.dtype([("id", "<i8"), ("step", "<i8"), ("cause", "<i4"), ("age", "<i4"),
                        ("n_off", "<i4"), ("gen", "<i4"), ("n_hidden", "<i4"), ("eaten", "<f4")])


def attack_newly_on(old: Config, new: Config) -> bool:
    """old에서 new로 바꾸며 공격 출력이 처음으로 쓰이게 되는가: 공격(conflict_enabled)이나 동물 장치가 새로 켜진다.
    꺼진 세계에서는 그 출력이 변이로 떠돌았으므로, 이때 편향을 새 개체와 같게 맞춘다(World.calm_attack)."""
    return (new.conflict_enabled and not old.conflict_enabled) or (new.animals_on and not old.animals_on)


class World:
    def __init__(self, cfg: Config, populate: bool = True):
        cfg.validate()
        self.cfg = cfg
        self.fp, self.ip = k.pack_params(cfg)
        self.fs, self.isp = st.pack_stage_params(cfg)
        self.fa, self.ia = an.pack_animal_params(cfg)
        self.s = k.alloc_state(cfg)
        self.step_no = 0
        self.births: list[np.ndarray] = []   # 아직 파일에 안 쓴 사건
        self.deaths: list[np.ndarray] = []
        self.passive_births = None           # 대조 실행용: (스텝 배열, 횟수 배열)
        self.passive_deaths = None
        self.lifetime_reset: list[str] = []  # 크기를 올려 읽을 때 0에서 다시 시작한 평생 학습 배열(내용이 있던 것만)
        if cfg.field_cap:
            self.recount_field()
        if populate:
            nb = k.populate_initial(self.s, cfg.init_agents, cfg.max_age, self.fp, self.ip)
            self._take_births(nb)

    # ---- 설정 바꾸기(분기) ----
    def set_config(self, cfg: Config) -> None:
        """배열 크기를 바꾸지 않는 설정만 바꿀 수 있다."""
        old = self.cfg
        for name in ("max_agents", "food_cap", "h_max", "vision_sectors", "n_spare_in",
                     "n_spare_out", "world_w", "world_h", "cell", "max_animals"):
            if getattr(old, name) != getattr(cfg, name):
                raise ValueError(f"{name}은 실행 중에 바꿀 수 없다(배열 크기가 달라진다)")
        cfg.validate()
        if cfg.mind_enabled != old.mind_enabled or cfg.n_proto != old.n_proto:
            # 마음을 켜거나 끄면 평생 학습 배열의 크기가 바뀐다. 새로 만들고 0에서 시작한다
            fresh = k.alloc_state(cfg)
            self.s = self.s._replace(**{name: getattr(fresh, name) for name in k.LIFETIME_FLEX})
            self.s.last_proto[:] = 0
            self.s.e_prev[:] = self.s.energy    # 켠 첫 스텝에 엉뚱한 "에너지 변화"가 보상에 들어가지 않게
        elif (cfg.perturb_sigma > 0.0) != (old.perturb_sigma > 0.0):
            # 노드 섭동을 켜거나 끄면 연결별 흔적 배열의 크기만 바뀐다. 그때까지 배운 것(Hin, Hrec)은 그대로 둔다
            fresh = k.alloc_state(cfg)
            self.s = self.s._replace(**{name: getattr(fresh, name) for name in k.PERTURB_FLEX})
            # 두 규칙이 epost를 다른 뜻으로 쓴다(활동 흔적 / 이번 스텝에 흔든 양). 바꾸는 순간의 값이 섞이지 않게 비운다
            self.s.epost[:] = 0.0
            self.s.epre_in[:] = 0.0
            self.s.epre_h[:] = 0.0
        self.cfg = cfg
        self.fp, self.ip = k.pack_params(cfg)
        self.fs, self.isp = st.pack_stage_params(cfg)
        self.fa, self.ia = an.pack_animal_params(cfg)
        if old.carry_enabled and not cfg.carry_enabled:
            self.release_all()
        if old.prey_enabled and not cfg.prey_enabled:
            # 산 사냥감은 사라지고(에너지는 L_LOST), 쌓여 있던 사냥감 몫은 풀 몫으로 돌아간다
            self.clear_animals(an.ANIMAL_PREY)
            self.s.budget[0] += self.s.budget[2]
            self.s.budget[2] = 0.0
        if cfg.field_cap and not old.field_cap:
            self.recount_field()    # 꺼져 있던 동안 들판 알 수를 맞추지 않았다
        if old.field_cap and not cfg.field_cap:
            self.s.counters[k.C_NFIELD] = 0     # 꺼진 동안은 맞추지 않으니 남은 값이 낡지 않게 비운다
        if old.huddle_enabled and not cfg.huddle_enabled:
            self.s.huddle_cut[:] = 0.0          # 꺼진 동안 move가 지난 겨울의 절감을 쓰지 않게 비운다
        if old.taste_enabled and not cfg.taste_enabled:
            self.s.taste_m[:] = 0.0             # 꺼진 세계에 낡은 기억이 남아 해시·물려주기에 섞이지 않게 지운다

    # ---- 들기: 물건 칸 ----
    def recount_field(self) -> None:
        """들판 알 수(C_NFIELD)를 살아 있는 들판 알(들판을 떠난 적 없고 사체·고기가 아닌 것) 수로 다시 센다."""
        s = self.s
        s.counters[k.C_NFIELD] = int((s.f_alive & (s.f_dropped == 0) & ~k.always_edible_mask(s.f_kind)).sum())

    def clear_animals(self, kind: int) -> float:
        """그 종류의 산 동물을 모두 없앤다. 몸에 든 에너지는 장부의 L_LOST로 가고, 그 합을 돌려준다."""
        s = self.s
        gone = s.an_alive & (s.an_kind == kind)
        e = 0.0
        for j in np.flatnonzero(gone):
            e += float(s.an_energy[j])
        s.ledger[k.L_LOST] += e
        s.an_energy[gone] = 0.0
        s.an_alive[gone] = False
        return e

    def release_all(self) -> None:
        """모든 손을 푼다. 든 물건은 든 개체의 자리에 내려놓는다(f_dropped는 그대로 1이다)."""
        s = self.s
        items = np.flatnonzero(s.f_holder >= 0)
        holders = s.f_holder[items]
        s.fx[items] = s.x[holders]
        s.fy[items] = s.y[holders]
        s.f_holder[:] = -1
        s.a_held[:] = -1

    def calm_new_outputs(self, old_n_out: int) -> list[str]:
        """크기를 올려 읽어 새로 생긴 출력 자리(old_n_out부터)의 편향을 새로 만든 개체의 기본값으로 맞춘다.
        0으로 채운 편향은 기본값과 같다. 다만 나르기(ACT_CARRY)만 새 개체가 -1에서 시작하므로 이것만 고친다.
        손댄 것을 돌려준다(없으면 빈 목록)."""
        if old_n_out <= ACT_CARRY < self.cfg.n_out:
            self.s.bout[self.s.alive, ACT_CARRY] = -1.0
            return ["bout[ACT_CARRY]=-1"]
        return []

    def calm_attack(self) -> list[str]:
        """산 개체의 공격 출력 편향을 새로 만든 개체와 같은 -1로 맞춘다. 손댄 것을 돌려준다."""
        self.s.bout[self.s.alive, ACT_ATTACK] = -1.0
        return ["bout[ACT_ATTACK]=-1"]

    # ---- 사건 버퍼 ----
    def _take_births(self, nb: int) -> None:
        if nb:
            b = self.s.bbuf[:nb]
            rec = np.empty(nb, dtype=BIRTH_DTYPE)
            rec["id"], rec["parent"], rec["step"] = b[:, 0], b[:, 1], b[:, 2]
            rec["gen"], rec["n_hidden"], rec["kind"] = b[:, 3], b[:, 4], b[:, 5]
            self.births.append(rec)

    def _take_deaths(self, nd: int) -> None:
        if nd:
            d = self.s.dbuf[:nd]
            rec = np.empty(nd, dtype=DEATH_DTYPE)
            rec["id"], rec["step"], rec["cause"], rec["age"] = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
            rec["n_off"], rec["gen"], rec["n_hidden"] = d[:, 4], d[:, 5], d[:, 6]
            rec["eaten"] = self.s.dbuf_f[:nd]
            self.deaths.append(rec)

    def drain_events(self):
        b = np.concatenate(self.births) if self.births else np.empty(0, dtype=BIRTH_DTYPE)
        d = np.concatenate(self.deaths) if self.deaths else np.empty(0, dtype=DEATH_DTYPE)
        self.births, self.deaths = [], []
        return b, d

    # ---- 계절 ----
    def in_winter(self, t: int) -> bool:
        """스텝 t가 겨울인가: 해의 끝 winter_len 스텝. 겨울이 꺼져 있으면 늘 False."""
        cfg = self.cfg
        return cfg.season_enabled and t % cfg.season_period >= cfg.season_period - cfg.winter_len

    def in_autumn(self, t: int) -> bool:
        """스텝 t가 가을인가: 겨울 바로 앞 autumn_len 스텝. 겨울이 꺼져 있으면 늘 False."""
        cfg = self.cfg
        if not cfg.season_enabled:
            return False
        u = t % cfg.season_period
        return cfg.season_period - cfg.winter_len - cfg.autumn_len <= u < cfg.season_period - cfg.winter_len

    def season(self, t: int) -> tuple[float, float, float, float]:
        """스텝 t의 계절: (풀 배수, 기초 대사 배수, 해 위상 sin, cos).
        겨울은 해의 끝 winter_len 스텝, 가을은 그 바로 앞 autumn_len 스텝이다(갈라 켜는 스텝이 해의 배수면 봄에서 시작한다).
        가을·겨울 밖의 풀 배수는 해 평균이 1이 되게 잡는다. 꺼져 있으면 (1, 1, 0, 0)."""
        cfg = self.cfg
        if not cfg.season_enabled:
            return 1.0, 1.0, 0.0, 0.0
        P, W, A = cfg.season_period, cfg.winter_len, cfg.autumn_len
        ang = 2.0 * math.pi * ((t % P) / P)
        if self.in_winter(t):
            return float(cfg.winter_mult), float(cfg.winter_basal_mult), math.sin(ang), math.cos(ang)
        if self.in_autumn(t):
            return float(cfg.autumn_mult), 1.0, math.sin(ang), math.cos(ang)
        rest = (P - float(cfg.winter_mult) * W - float(cfg.autumn_mult) * A) / (P - W - A)
        return rest, 1.0, math.sin(ang), math.cos(ang)

    def spoil_mult(self, t: int) -> float:
        """스텝 t의 부패 속도 배수: 겨울이면 spoil_winter_mult, 아니면 1."""
        return float(self.cfg.spoil_winter_mult) if self.in_winter(t) else 1.0

    # ---- 한 스텝 ----
    def step(self) -> None:
        s, fp, ip = self.s, self.fp, self.ip
        fs, isp = self.fs, self.isp
        cfg = self.cfg
        mind, social, niche = cfg.mind_enabled, cfg.social_enabled, cfg.niche_enabled
        t = self.step_no
        gw, gh = int(ip[k.Q_GW]), int(ip[k.Q_GH])
        inv_cell = fp[k.P_INV_CELL]
        grass, basal, sn, cs = self.season(t)
        g = grass       # 풀 몫의 배수. 사냥감이 켜지면 유입의 사냥감 몫을 먼저 떼어 낸다
        if cfg.prey_enabled:
            an.prey_inflow(s, self.fa, self.ia, grass)
            g = grass * (1.0 - cfg.prey_inflow_frac)
        if niche:
            st.niche_world(t, s, fs, isp, g)
        k.spawn_food(t, s, fp, ip, g)
        if cfg.spoil_enabled:
            k.spoil_food(t, s, fp, ip, self.spoil_mult(t))
        if cfg.prey_enabled:
            an.spawn_prey(t, s, self.fa, self.ia)
        k.build_grid(s.x, s.y, s.alive, inv_cell, gw, gh, s.a_start, s.a_items, s.cursor)
        k.build_grid(s.fx, s.fy, s.f_alive, inv_cell, gw, gh, s.f_start, s.f_items, s.cursor)
        k.sense(t, s, fp, ip)
        if social:
            st.social_sense(s, fs, isp)
        if niche:
            st.trail_sense(s, fs, isp)
        if cfg.season_enabled:
            k.write_ext2(s.alive, s.obs, cfg.ext_in0 + EXT_SEASON, sn, cs)
        if cfg.carry_enabled:
            k.write_held(s.alive, s.a_held, s.obs, cfg.ext_in0 + EXT_CARRY)
        if cfg.prey_enabled:
            an.sense_animals(s.alive, s.x, s.y, s.heading, s.obs, cfg.ext_in0 + EXT_PREY, s.an_alive, s.an_kind,
                             s.an_x, s.an_y, an.ANIMAL_PREY, cfg.prey_radius, cfg.vision_range, cfg.vision_fov,
                             cfg.world_w, cfg.world_h)
        if mind:
            st.mind_update(s, fs, isp)
            st.think_plastic(t, s, fs, isp)
        else:
            k.think(s)
        if mind or social:
            st.after_think(s, fs, isp)
        if cfg.huddle_enabled:
            k.huddle(s, fp, ip, self.in_winter(t))
        k.move(s, fp, ip, basal)
        if mind or niche:
            st.after_move(s, fs, isp, basal)
        if cfg.animals_on:
            an.move_animals(t, s, self.fa, self.ia)
        k.build_grid(s.x, s.y, s.alive, inv_cell, gw, gh, s.a_start, s.a_items, s.cursor)
        k.collide(s, fp, ip)
        if cfg.carry_enabled:
            k.carry(t, s, fp, ip)
        k.eat(t, s, fp, ip)
        if social:
            st.give(s, fs, isp)
        if cfg.conflict_enabled:
            st.attack(t, s, fs, isp)
        if cfg.prey_enabled:
            an.attack_animals(t, s, self.fa, self.ia)
        nd = k.process_deaths(t, s, fp, ip)
        self._take_deaths(nd)
        nb = k.process_births(t, s, fp, ip)
        if self.cfg.passive:
            nb, nd = k.passive_events(t, s, fp, ip, self._passive_count(self.passive_births, t),
                                      self._passive_count(self.passive_deaths, t))
            self._take_deaths(nd)
        nb += k.reseed(t, s, fp, ip, nb)
        self._take_births(nb)
        self.step_no = t + 1
        s.counters[k.C_STEP] = self.step_no

    def run(self, n: int) -> None:
        for _ in range(n):
            self.step()

    # ---- 재난 (각본) ----
    def strike(self, x: float, y: float, radius: float, kill: float = 1.0, burn_food: bool = True,
               key: int = 0) -> int:
        """운석. (x, y) 둘레 반지름 radius 안의 개체를 kill 확률로 죽이고 burn_food면 먹이도 없앤다.

        스텝과 스텝 사이에 부른다. 사망 기록의 스텝은 이제 돌 스텝(step_no)이다.
        key는 누가 죽는지 정하는 난수 번호다(같은 스텝에 여러 번 떨어뜨릴 때 서로 다르게 준다).
        죽은 수를 돌려준다.
        """
        cfg = self.cfg
        nd = k.strike(self.step_no, self.s, self.fp, self.ip, float(x) % cfg.world_w, float(y) % cfg.world_h,
                      float(radius), float(kill), bool(burn_food), int(key), 0)
        self._take_deaths(nd)
        return int(nd)

    @staticmethod
    def _passive_count(table, t: int) -> int:
        if table is None:
            return 0
        steps, counts = table
        i = np.searchsorted(steps, t)
        if i < len(steps) and steps[i] == t:
            return int(counts[i])
        return 0

    def set_passive_schedule(self, births: np.ndarray, deaths: np.ndarray) -> None:
        """본 실행의 사건 기록에서 스텝별 자연 출생·사망 횟수를 뽑아 대조 실행 일정으로 쓴다."""
        # 재주입도 따라 한다(재주입은 원래 살아 있는 무작위 개체의 복제라 대조 실행의 출생과 뜻이 같다).
        # 빼면 본 실행은 하한에서 버티는데 대조 실행만 무너진다.
        nat = births[(births["kind"] == k.BIRTH_NATURAL) | (births["kind"] == k.BIRTH_RESEED)]
        self.passive_births = np.unique(nat["step"], return_counts=True)
        self.passive_deaths = np.unique(deaths["step"], return_counts=True)

    # ---- 저장 ----
    def state_arrays(self) -> dict:
        return {name: getattr(self.s, name) for name in k.SAVED_FIELDS}

    def load_arrays(self, arrays: dict) -> bool:
        """저장된 배열을 읽는다. 옛 체크포인트(입출력 자리가 적거나 2~4단계 배열이 없는 것)도 받는다:
        모자란 입출력 자리는 0 가중치로 채우고, 없는 유전자는 기본값으로, 평생 상태는 0에서 시작한다."""
        s = self.s
        missing = []
        exact = True        # 저장된 것과 모양까지 똑같이 읽었는가(아니면 저장 때의 상태 해시와 견줄 수 없다)
        for name in k.SAVED_FIELDS:
            dst = getattr(s, name)
            if name not in arrays:
                if name in k.HASH_DEFAULTS or name in k.ANIMAL_SAVED:
                    # 이 필드가 생기기 전의 체크포인트. 기본값(동물은 빈 배열)에서 시작하고, 기본값뿐이거나 산 동물이
                    # 없으면 해시에 넣지 않으므로 저장 때의 상태 해시와 그대로 견줄 수 있다(빠진 것으로 치지 않는다)
                    continue
                if name not in NEW_SINCE_STAGE1:
                    raise ValueError(f"체크포인트에 {name}이 없다")
                missing.append(name)
                continue
            src = arrays[name]
            if dst.shape == src.shape:
                dst[...] = src
                continue
            was_exact, exact = exact, False
            if name in k.LIFETIME_FLEX:
                if src.any():
                    self.lifetime_reset.append(name)
                dst[...] = 0
            elif (src.ndim == dst.ndim and all(a <= b for a, b in zip(src.shape, dst.shape))
                  and (name in ("budget", "counters") or (name in ("Win", "Wout", "bout")
                                            and src.shape[0] == dst.shape[0]))):
                dst[...] = 0
                dst[tuple(slice(0, n) for n in src.shape)] = src
                if name in ("budget", "counters"):
                    exact = was_exact   # 늘어난 꼬리는 0이고 해시가 0 꼬리를 넣지 않으므로 저장 때와 같다
            else:
                raise ValueError(f"{name}: 크기가 다르다 {src.shape} != {dst.shape}")
        if "eta" in missing:
            s.eta[:] = self.cfg.eta_init
        if "alpha" in missing:
            # 전부 0이면 갈라낸 직후 한 세대 동안 학습이 아무 효과도 낼 수 없다
            gen = np.random.default_rng(self.cfg.seed + 977)
            s.alpha[:] = (self.cfg.alpha_init * gen.standard_normal(s.alpha.shape)).astype(np.float32)
        if "rw" in missing:
            s.rw[:] = 0.0
            s.rw[:, :N_REWARD] = np.array([1.0, 0.5, 0.0, 0.2, 0.5, -0.2, 0.2, 0.0], dtype=np.float32)
        # f_born이 없던 옛 체크포인트의 먹이는 난 스텝이 0으로 채워진다. 부패를 켜면 모두 오래된 먹이로 취급된다
        if "e_prev" in missing:
            s.e_prev[:] = s.energy
        if self.cfg.field_cap:
            self.recount_field()
        self.step_no = int(s.counters[k.C_STEP])
        return exact and not missing

    def state_hash(self) -> str:
        """저장되는 상태 전체의 해시. 재현성 점검에 쓴다."""
        m = hashlib.sha256()
        s = self.s
        alive = s.alive
        m.update(str(self.step_no).encode())
        for name, _, _, saved in k.AGENT_FIELDS:
            if saved:
                # 죽은 칸에 남은 값은 뜻이 없으므로 살아 있는 칸만 본다
                v = getattr(s, name)[alive]
                if name in k.HASH_DEFAULTS and (v == k.HASH_DEFAULTS[name]).all():
                    continue    # 기본값뿐이면 넣지 않는다: 그 장치를 안 쓰는 세계의 해시가 이 필드가 생기기 전과 같게
                m.update(np.ascontiguousarray(v).tobytes())
        fa = s.f_alive
        for name, _, saved in k.FOOD_FIELDS:
            if saved:
                v = getattr(s, name)[fa]
                if name in k.HASH_DEFAULTS and (v == k.HASH_DEFAULTS[name]).all():
                    continue
                m.update(np.ascontiguousarray(v).tobytes())
        aa = s.an_alive
        if aa.any():    # 산 동물이 없으면 넣지 않는다(동물 장치를 안 쓰는 세계의 해시가 동물 배열이 생기기 전과 같게)
            for name in k.ANIMAL_SAVED:
                v = getattr(s, name)[aa]
                if name in k.HASH_DEFAULTS and (v == k.HASH_DEFAULTS[name]).all():
                    continue
                m.update(np.ascontiguousarray(v).tobytes())
        # 늘어난 꼬리가 모두 0이면 넣지 않는다(기본 길이 세계·옛 체크포인트의 해시와 같게)
        c = s.counters
        m.update((c[:k.N_COUNTERS_BASE] if not c[k.N_COUNTERS_BASE:].any() else c).tobytes())
        m.update(s.ledger.tobytes())
        b = s.budget
        m.update((b[:k.N_BUDGET_BASE] if not b[k.N_BUDGET_BASE:].any() else b).tobytes())
        if s.trail.any() or s.soil.any():   # 땅이 비어 있으면 넣지 않는다(1~3단계 해시가 그대로 유효하게)
            m.update(s.trail.tobytes())
            m.update(s.soil.tobytes())
        return m.hexdigest()

    # ---- 요약 ----
    def summary(self) -> dict:
        s = self.s
        a = s.alive
        n = int(a.sum())
        c = s.counters
        out = {
            "step": self.step_no,
            "pop": n,
            "food": int(c[k.C_NFOOD]),
            "births": int(c[k.C_BIRTHS]),
            "deaths_starve": int(c[k.C_DEATH_STARVE]),
            "deaths_age": int(c[k.C_DEATH_AGE]),
            "deaths_passive": int(c[k.C_DEATH_PASSIVE]),
            "deaths_disaster": int(c[k.C_DEATH_DISASTER]),
            "reseeds": int(c[k.C_RESEED]),
            "blocked": int(c[k.C_BLOCKED]),
            "eaten": int(c[k.C_EATEN]),
        }
        if n:
            out.update({
                "energy_mean": float(s.energy[a].mean()),
                "age_mean": float(s.age[a].mean()),
                "gen_mean": float(s.gen[a].mean()),
                "gen_max": int(s.gen[a].max()),
                "hidden_mean": float(s.n_hidden[a].mean()),
                "hidden_max": int(s.n_hidden[a].max()),
                "sigma_mean": float(s.sigma[a].mean()),
                "speed_mean": float(np.hypot(s.vx[a], s.vy[a]).mean()),
            })
            cfg = self.cfg
            if cfg.mind_enabled:
                out.update({
                    "eta_mean": float(s.eta[a].mean()),
                    "alpha_abs_mean": float(np.abs(s.alpha[a]).mean()),
                    "asleep_frac": float((s.asleep[a] > 0).mean()),
                    "hebb_abs_mean": float(np.abs(s.Hrec[a]).mean()),
                    "dopamine_mean": float(s.horm[a, 2].mean()),
                })
            if cfg.social_enabled:
                out.update({
                    "signal_abs_mean": float(np.abs(s.sig[a]).mean()),
                    "gifts": int(c[k.C_GIFTS]),
                })
            if cfg.conflict_enabled:
                out["attacks"] = int(c[k.C_ATTACKS])
                out["deaths_killed"] = int(c[k.C_DEATH_KILLED])
            if cfg.food_cap_replace:
                out["food_replaced"] = int(c[k.C_FOOD_REPLACED])
            if cfg.food_kinds_enabled:
                out["poisoned"] = int(c[k.C_POISONED])
                out["good_kind"] = int(k.good_kind(cfg.seed, self.step_no, cfg.kind_flip_period))
            if cfg.taste_enabled:
                out["taste_plus"] = float(s.taste_m[a, 0].mean())     # +로 보이는 먹이의 기억 평균
                out["taste_minus"] = float(s.taste_m[a, 1].mean())    # -로 보이는 먹이의 기억 평균
                if cfg.food_kinds_enabled:
                    # 지금 독인 모양 쪽 기억의 평균. 방금 돈 스텝(step_no - 1) 기준이라
                    # 다음 스텝 값인 good_kind와 독이 바뀌는 순간에는 어긋날 수 있다
                    bad = 1 - int(k.good_kind(cfg.seed, max(self.step_no - 1, 0), cfg.kind_flip_period))
                    out["taste_on_poison"] = float(s.taste_m[a, 1 if bad == k.KIND_B else 0].mean())
            if c[k.C_MIGRANTS] > 0 or (s.tag[a] != 0).any():
                # 이주해 온 생물과 그 후손의 비율. 이주가 있었던 세계는 그 계통이 다 사라져도 0으로 적는다
                out["migrant_frac"] = float((s.tag[a] != 0).mean())
            if cfg.niche_enabled:
                out.update({
                    "trail_mean": float(s.trail.mean()),
                    "soil_total": float(s.soil.sum()),
                    "fertile_food": int(c[k.C_FERTILE]),
                })
        # 아래 키는 개체 수와 상관없이 낸다(모두 죽은 순간에도 열이 끊기지 않게)
        if self.cfg.huddle_enabled:
            out["huddle_cut_mean"] = float(s.huddle_cut[a].mean()) if n else 0.0
        if self.cfg.prey_enabled:
            out["prey_alive"] = int((s.an_alive & (s.an_kind == an.ANIMAL_PREY)).sum())
            out["prey_killed"] = int(c[k.C_PREY_KILLED])
            out["prey_hunters"] = int(c[k.C_HUNTERS])     # 쓰러뜨린 사냥감마다 붙어 있던 공격자 수의 합
            out["prey_hits"] = int(c[k.C_PREY_HITS])      # 사냥감을 친 개체-스텝 수
            pm = s.an_alive & (s.an_kind == an.ANIMAL_PREY)
            out["prey_body_mean"] = float(s.an_energy[pm].mean()) if pm.any() else 0.0   # 산 사냥감 몸 평균(뜯기면 준다)
            out["deaths_killed"] = int(c[k.C_DEATH_KILLED])
        return out

    def energy_balance(self) -> dict:
        """에너지 장부 점검. residual이 0에 가까워야 한다(새는 곳 감시)."""
        na, ea, nf, ef = k.totals(self.s)
        L = self.s.ledger
        soil = float(self.s.soil.sum())    # 땅에 쌓인 거름도 세계 안의 에너지다
        animals = float(an.animal_energy(self.s))   # 동물 몸에 든 에너지도 세계 안의 에너지다
        expected = (L[k.L_SPAWN] + L[k.L_INJECT] - L[k.L_BASAL] - L[k.L_BRAIN]
                    - L[k.L_MOVE] - L[k.L_BIRTH] - L[k.L_LOST])
        return {
            "agents": float(ea), "food": float(ef), "soil": soil, "animals": animals,
            "expected": float(expected), "residual": float(ea + ef + soil + animals - expected),
            "spawn": float(L[k.L_SPAWN]), "inject": float(L[k.L_INJECT]),
            "basal": float(L[k.L_BASAL]), "brain": float(L[k.L_BRAIN]),
            "move": float(L[k.L_MOVE]), "birth": float(L[k.L_BIRTH]),
            "lost": float(L[k.L_LOST]), "discard": float(L[k.L_DISCARD]),
        }

    def check_sane(self) -> None:
        s = self.s
        a = s.alive
        for name in ("x", "y", "vx", "vy", "heading", "energy"):
            v = getattr(s, name)[a]
            if not np.isfinite(v).all():
                raise FloatingPointError(f"{name}에 NaN/Inf")
        if a.any():
            if (s.x[a] < 0).any() or (s.x[a] >= self.cfg.world_w).any() \
                    or (s.y[a] < 0).any() or (s.y[a] >= self.cfg.world_h).any():
                raise ValueError("세계 밖으로 나간 개체가 있다")
            if not np.isfinite(s.h[a]).all():
                raise FloatingPointError("뇌 상태에 NaN/Inf")
        if int(a.sum()) != int(s.counters[k.C_NALIVE]):
            raise ValueError("살아 있는 수가 안 맞는다")
