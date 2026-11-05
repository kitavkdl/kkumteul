"""세계 하나. 한 스텝의 순서와 사건 기록을 맡는다."""
from __future__ import annotations

import hashlib

import numpy as np

from . import kernels as k
from .config import Config

BIRTH_DTYPE = np.dtype([("id", "<i8"), ("parent", "<i8"), ("step", "<i8"),
                        ("gen", "<i4"), ("n_hidden", "<i4"), ("kind", "<i4")])
DEATH_DTYPE = np.dtype([("id", "<i8"), ("step", "<i8"), ("cause", "<i4"), ("age", "<i4"),
                        ("n_off", "<i4"), ("gen", "<i4"), ("n_hidden", "<i4"), ("eaten", "<f4")])


class World:
    def __init__(self, cfg: Config, populate: bool = True):
        cfg.validate()
        self.cfg = cfg
        self.fp, self.ip = k.pack_params(cfg)
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
        self.cfg = cfg
        self.fp, self.ip = k.pack_params(cfg)

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

    # ---- 한 스텝 ----
    def step(self) -> None:
        s, fp, ip = self.s, self.fp, self.ip
        t = self.step_no
        gw, gh = int(ip[k.Q_GW]), int(ip[k.Q_GH])
        inv_cell = fp[k.P_INV_CELL]
        k.spawn_food(t, s, fp, ip)
        k.build_grid(s.x, s.y, s.alive, inv_cell, gw, gh, s.a_start, s.a_items, s.cursor)
        k.build_grid(s.fx, s.fy, s.f_alive, inv_cell, gw, gh, s.f_start, s.f_items, s.cursor)
        k.sense(t, s, fp, ip)
        k.think(s)
        k.move(s, fp, ip)
        k.build_grid(s.x, s.y, s.alive, inv_cell, gw, gh, s.a_start, s.a_items, s.cursor)
        k.collide(s, fp, ip)
        k.eat(s, fp, ip)
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

    def load_arrays(self, arrays: dict) -> None:
        for name in k.SAVED_FIELDS:
            dst = getattr(self.s, name)
            src = arrays[name]
            if dst.shape != src.shape:
                raise ValueError(f"{name}: 크기가 다르다 {src.shape} != {dst.shape}")
            dst[...] = src
        self.step_no = int(self.s.counters[k.C_STEP])

    def state_hash(self) -> str:
        """저장되는 상태 전체의 해시. 재현성 점검에 쓴다."""
        m = hashlib.sha256()
        s = self.s
        alive = s.alive
        m.update(str(self.step_no).encode())
        for name, _, _, saved in k.AGENT_FIELDS:
            if saved:
                # 죽은 칸에 남은 값은 뜻이 없으므로 살아 있는 칸만 본다
                m.update(np.ascontiguousarray(getattr(s, name)[alive]).tobytes())
        fa = s.f_alive
        for name, _, saved in k.FOOD_FIELDS:
            if saved:
                m.update(np.ascontiguousarray(getattr(s, name)[fa]).tobytes())
        m.update(s.counters.tobytes())
        m.update(s.ledger.tobytes())
        m.update(s.budget.tobytes())
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
        return out

    def energy_balance(self) -> dict:
        """에너지 장부 점검. residual이 0에 가까워야 한다(새는 곳 감시)."""
        na, ea, nf, ef = k.totals(self.s)
        L = self.s.ledger
        expected = (L[k.L_SPAWN] + L[k.L_INJECT] - L[k.L_BASAL] - L[k.L_BRAIN]
                    - L[k.L_MOVE] - L[k.L_BIRTH] - L[k.L_LOST])
        return {
            "agents": float(ea), "food": float(ef), "expected": float(expected),
            "residual": float(ea + ef - expected),
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
