"""세계 하나. 한 스텝의 순서와 사건 기록을 맡는다."""
from __future__ import annotations

import hashlib
import math

import numpy as np

from . import kernels as k
from . import stages as st
from .config import ACT_CARRY, EXT_SEASON, Config, N_REWARD

# 2~4단계에서 새로 생긴 저장 필드. 옛 체크포인트에는 없어도 된다(그 밖의 필드가 없으면 깨진 것이다)
NEW_SINCE_STAGE1 = frozenset({"rw", "eta", "alpha", "horm", "mod", "exp_r", "e_prev", "fatigue", "asleep",
                              "epost", "epre_h", "epre_in", "Hin", "Hrec", "proto", "link", "last_proto",
                              "sig", "trail", "soil", "f_kind", "f_born"})

BIRTH_DTYPE = np.dtype([("id", "<i8"), ("parent", "<i8"), ("step", "<i8"),
                        ("gen", "<i4"), ("n_hidden", "<i4"), ("kind", "<i4")])
DEATH_DTYPE = np.dtype([("id", "<i8"), ("step", "<i8"), ("cause", "<i4"), ("age", "<i4"),
                        ("n_off", "<i4"), ("gen", "<i4"), ("n_hidden", "<i4"), ("eaten", "<f4")])


class World:
    def __init__(self, cfg: Config, populate: bool = True):
        cfg.validate()
        self.cfg = cfg
        self.fp, self.ip = k.pack_params(cfg)
        self.fs, self.isp = st.pack_stage_params(cfg)
        self.s = k.alloc_state(cfg)
        self.step_no = 0
        self.births: list[np.ndarray] = []   # 아직 파일에 안 쓴 사건
        self.deaths: list[np.ndarray] = []
        self.passive_births = None           # 대조 실행용: (스텝 배열, 횟수 배열)
        self.passive_deaths = None
        if populate:
            nb = k.populate_initial(self.s, cfg.init_agents, cfg.max_age, self.fp, self.ip)
            self._take_births(nb)

    # ---- 설정 바꾸기(분기) ----
    def set_config(self, cfg: Config) -> None:
        """배열 크기를 바꾸지 않는 설정만 바꿀 수 있다."""
        old = self.cfg
        for name in ("max_agents", "food_cap", "h_max", "vision_sectors", "n_spare_in",
                     "n_spare_out", "world_w", "world_h", "cell"):
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

    def calm_new_outputs(self, old_n_out: int) -> list[str]:
        """크기를 올려 읽어 새로 생긴 출력 자리(old_n_out부터)의 편향을 새로 만든 개체의 기본값으로 맞춘다.
        0으로 채운 편향은 대부분 기본값과 같지만, 나르기(ACT_CARRY)는 새 개체가 -1에서 시작한다.
        손댄 것을 돌려준다(없으면 빈 목록)."""
        if old_n_out <= ACT_CARRY < self.cfg.n_out:
            self.s.bout[self.s.alive, ACT_CARRY] = -1.0
            return ["bout[ACT_CARRY]=-1"]
        return []

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
    def season(self, t: int) -> tuple[float, float, float, float]:
        """스텝 t의 계절: (풀 배수, 기초 대사 배수, 해 위상 sin, cos).
        겨울은 해의 끝 winter_len 스텝이다(갈라 켜는 스텝이 해의 배수면 봄에서 시작한다).
        겨울 밖의 풀 배수는 해 평균이 1이 되게 잡는다. 꺼져 있으면 (1, 1, 0, 0)."""
        cfg = self.cfg
        if not cfg.season_enabled:
            return 1.0, 1.0, 0.0, 0.0
        P, W = cfg.season_period, cfg.winter_len
        r = t % P
        ang = 2.0 * math.pi * (r / P)
        if r >= P - W:
            return cfg.winter_mult, cfg.winter_basal_mult, math.sin(ang), math.cos(ang)
        return (P - cfg.winter_mult * W) / (P - W), 1.0, math.sin(ang), math.cos(ang)

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
        if niche:
            st.niche_world(t, s, fs, isp, grass)
        k.spawn_food(t, s, fp, ip, grass)
        k.build_grid(s.x, s.y, s.alive, inv_cell, gw, gh, s.a_start, s.a_items, s.cursor)
        k.build_grid(s.fx, s.fy, s.f_alive, inv_cell, gw, gh, s.f_start, s.f_items, s.cursor)
        k.sense(t, s, fp, ip)
        if social:
            st.social_sense(s, fs, isp)
        if niche:
            st.trail_sense(s, fs, isp)
        if cfg.season_enabled:
            k.write_ext2(s.alive, s.obs, cfg.ext_in0 + EXT_SEASON, sn, cs)
        if mind:
            st.mind_update(s, fs, isp)
            st.think_plastic(t, s, fs, isp)
        else:
            k.think(s)
        if mind or social:
            st.after_think(s, fs, isp)
        k.move(s, fp, ip, basal)
        if mind or niche:
            st.after_move(s, fs, isp, basal)
        k.build_grid(s.x, s.y, s.alive, inv_cell, gw, gh, s.a_start, s.a_items, s.cursor)
        k.collide(s, fp, ip)
        k.eat(t, s, fp, ip)
        if social:
            st.give(s, fs, isp)
        if cfg.conflict_enabled:
            st.attack(t, s, fs, isp)
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
                if name in k.PERTURB_FLEX:
                    # 이 배열이 생기기 전의 체크포인트. 0에서 시작하고, 0뿐이면 해시에 넣지 않으므로
                    # 저장 때의 상태 해시와 그대로 견줄 수 있다(빠진 것으로 치지 않는다)
                    continue
                if name not in NEW_SINCE_STAGE1:
                    raise ValueError(f"체크포인트에 {name}이 없다")
                missing.append(name)
                continue
            src = arrays[name]
            if dst.shape == src.shape:
                dst[...] = src
                continue
            exact = False
            if name in k.LIFETIME_FLEX:
                dst[...] = 0
            elif (src.ndim == dst.ndim and all(a <= b for a, b in zip(src.shape, dst.shape))
                  and (name in ("budget", "counters") or (name in ("Win", "Wout", "bout")
                                            and src.shape[0] == dst.shape[0]))):
                dst[...] = 0
                dst[tuple(slice(0, n) for n in src.shape)] = src
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
        if "e_prev" in missing:
            s.e_prev[:] = s.energy
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
                if name in k.PERTURB_FLEX and not v.any():
                    continue    # 노드 섭동을 안 쓰는 세계의 해시가 이 필드가 생기기 전과 같게
                m.update(np.ascontiguousarray(v).tobytes())
        fa = s.f_alive
        for name, _, saved in k.FOOD_FIELDS:
            if saved:
                m.update(np.ascontiguousarray(getattr(s, name)[fa]).tobytes())
        m.update(s.counters.tobytes())
        m.update(s.ledger.tobytes())
        m.update(s.budget.tobytes())
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
            if c[k.C_MIGRANTS] > 0 or (s.tag[a] != 0).any():
                # 이주해 온 생물과 그 후손의 비율. 이주가 있었던 세계는 그 계통이 다 사라져도 0으로 적는다
                out["migrant_frac"] = float((s.tag[a] != 0).mean())
            if cfg.niche_enabled:
                out.update({
                    "trail_mean": float(s.trail.mean()),
                    "soil_total": float(s.soil.sum()),
                    "fertile_food": int(c[k.C_FERTILE]),
                })
        return out

    def energy_balance(self) -> dict:
        """에너지 장부 점검. residual이 0에 가까워야 한다(새는 곳 감시)."""
        na, ea, nf, ef = k.totals(self.s)
        L = self.s.ledger
        soil = float(self.s.soil.sum())    # 땅에 쌓인 거름도 세계 안의 에너지다
        expected = (L[k.L_SPAWN] + L[k.L_INJECT] - L[k.L_BASAL] - L[k.L_BRAIN]
                    - L[k.L_MOVE] - L[k.L_BIRTH] - L[k.L_LOST])
        return {
            "agents": float(ea), "food": float(ef), "soil": soil, "expected": float(expected),
            "residual": float(ea + ef + soil - expected),
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
