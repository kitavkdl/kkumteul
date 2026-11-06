"""다시 보기: 체크포인트에서 세계를 다시 돌리며 장면을 전부 적어, 자료가 통째로 든 HTML 한 파일로 내보낸다.

    python -m kkumteul.replay --ckpt runs/<실행>/checkpoints/ckpt_<스텝>.npz [--out FILE] [--steps 3000]
                              [--every 2] [--from-flip] [--title 이름] [--artifact]

서버 없이 폰에서도 열리고, 파일이나 링크로 남에게 보낼 수 있다. 설계와 까닭은 `docs/visualization.md`.

지키는 것
- 체크포인트와 실행 폴더는 읽기만 한다. 다시 돌리는 것은 메모리 안의 사본이다.
- 각본이 있는 실행은 실행기와 같은 방식으로 매 스텝 앞에서 각본을 적용한다.
- 선택 없는 대조 실행(passive)의 체크포인트는 받지 않는다.
- 누가 독을 먹었는지는 에너지 변화로 짐작하지 않고, 그 스텝에 사라진 먹이의 종류로 가린다.
- 내보낸 파일에는 경로를 넣지 않고, 바깥에서 불러오는 것이 없다.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import SOURCE_HASH
from . import checkpoint as ckpt
from . import kernels as k
from . import scenario as scn_mod
from .config import ACT_GIVE, Config
from .scenario import Scenario

# 장면의 개체 표시(한 바이트): 아래 두 비트씩은 그 사이에 먹은 좋은 먹이 수와 독 수(3에서 멈춘다)
F_ASLEEP = 16
F_GAVE = 32
F_HIT = 64
F_CARRY = 128       # 무엇을 들고 있다

# 먹이가 사라진 까닭
WHY_PRESENT = 0     # 끝까지 남아 있었다
WHY_GOOD = 1        # 누가 먹었다(먹어도 되는 것)
WHY_POISON = 2      # 누가 먹었다(독)
WHY_UNKNOWN = 3     # 먹혔지만 누가 먹었는지 모른다
WHY_EVICTED = 4     # 먹이 상한 장치가 새 먹이에 자리를 내줬다
WHY_BURNT = 5       # 운석에 사라졌다
WHY_SPOILED = 6     # 썩어서 사라졌다
WHY_PICKED = 7      # 누가 들어서 들판에서 빠졌다(내려놓으면 그 자리에서 새 행이 열린다)
WHY_STORED_OUT = 8  # 저장 칸(들판 아닌 물건 자리)이 차서 치워졌다

BORN_BEFORE = 255   # 다시 보기가 시작될 때 이미 살아 있던 개체의 출생 종류
NO_CAUSE = 255      # 다시 보기가 끝날 때까지 살아 있었다

MAX_BYTES = 15_000_000   # 아티팩트 상한(16MB) 아래로 지킨다
MAX_EVERY = 40           # 장면 간격을 이보다 넓히지는 않는다(움직임을 따라갈 수 없다)
HISTORY_POINTS = 600
SIG_MIN_FLOOR = 0.5      # 신호 고리 문턱의 하한. 실제 문턱은 기록마다 가장 센 신호 5%(95번째 백분위수)
FLIP_LEAD = 300          # --from-flip: 바뀌는 순간보다 이만큼(또는 구간의 1/4) 앞에서 시작한다

AGENT_DTYPE = np.dtype([("id", "<f8"), ("parent", "<i4"), ("gen", "<i4"), ("kind", "u1"), ("n_hidden", "u1"),
                        ("eta", "<f4"), ("color", "u1", (3,)), ("born", "<i4"), ("died", "<i4"), ("cause", "u1"),
                        ("n_off", "<u2"), ("eaten", "<f4"), ("ate_good", "<u2"), ("ate_poison", "<u2"),
                        ("kids", "<u2")])
FOOD_DTYPE = np.dtype([("x", "<u2"), ("y", "<u2"), ("kind", "u1"), ("f_in", "<u4"), ("f_out", "<u4"),
                       ("why", "u1")])

TEMPLATE = Path(__file__).with_name("replay.html")
_DATA_MARK = "<!--__REPLAY_DATA__-->"
_HEAD_RE = re.compile(r"<!--HEAD-->(.*?)<!--/HEAD-->", re.S)
_META_RE = re.compile(r'<script type="application/json" id="replay-meta">(.*?)</script>', re.S)
_BIN_RE = re.compile(r'<script type="application/octet-stream" id="replay-bin">(.*?)</script>', re.S)


def _q16(v: np.ndarray, size: float) -> np.ndarray:
    return np.minimum(v / size * 65536.0, 65535).astype("<u2")


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def intro_sentences(cfg: Config, scn: Scenario | None = None) -> list[str]:
    """이 세계가 무엇을 하는 세계인지 두세 문장. 꺼진 장치는 말하지 않는다."""
    out = ["작은 생물들이 먹이를 찾아 먹고, 새끼를 낳고, 죽습니다. 어떻게 살지는 아무도 가르치지 않습니다."]
    if cfg.food_kinds_enabled:
        out.append(f"이 세계에는 먹이가 두 가지이고 한쪽은 독입니다. 어느 쪽이 독인지는 "
                   f"{cfg.kind_flip_period:,}스텝마다 다시 정해집니다.")
    has = []
    if cfg.mind_enabled:
        has.append("살면서 배우는 장치와 잠")
    if cfg.social_enabled:
        has.append("서로 내고 듣는 신호와 에너지 건네기")
    if cfg.conflict_enabled:
        has.append("공격")
    if cfg.niche_enabled:
        has.append("다닌 자리에 생기는 길과 거름 쌓기")
    if has:
        out.append("이 세계의 생물에게는 " + ", ".join(has) + "가 주어져 있습니다. 쓸지 말지는 진화가 정합니다.")
    if scn is not None:
        desc = f": {scn.description.strip()}" if scn.description.strip() else ""
        out.append(f"이 세계에는 각본 '{scn.name}'이 걸려 있습니다{desc}")
    return out


class Replay:
    """다시 돌린 장면 전체. 시험과 내보내기가 같이 쓴다."""

    def __init__(self, start, every, off, cols, agents, food, per_frame, seen, meta, world_w, world_h):
        self.start = start
        self.every = every
        self.off = off                    # 장면 f의 개체는 cols의 off[f]:off[f+1]
        self.cols = cols                  # idx, x, y, head, energy, flags, (sig)
        self.agents = agents
        self.food = food
        self.n_frames = len(off) - 1
        self.steps = start + every * np.arange(self.n_frames, dtype=np.int64)
        self.pop = np.diff(off).astype(np.int64)
        self.good_kind = per_frame["good"]
        self.n_food = per_frame["food"]
        self.c_eaten = per_frame["eaten"]
        self.c_poisoned = per_frame["poisoned"]
        self.blind = per_frame["blind"]
        self.gifts = per_frame["gifts"]
        self.attacks = per_frame["attacks"]
        self.seen = seen
        self.meta = meta
        self._w, self._h = world_w, world_h

    def frame(self, f: int) -> SimpleNamespace:
        a, b = int(self.off[f]), int(self.off[f + 1])
        c = self.cols
        sig = c["sig"][a:b].astype(np.float32) / 127.0 if "sig" in c else None
        return SimpleNamespace(idx=c["idx"][a:b], x=c["x"][a:b] / 65536.0 * self._w,
                               y=c["y"][a:b] / 65536.0 * self._h, head=c["head"][a:b],
                               energy=c["energy"][a:b], flags=c["flags"][a:b], sig=sig)


class Recorder:
    """세계를 한 스텝씩 돌리며 장면과 사건을 적는다. 세계는 메모리 안의 사본이어야 한다(상태가 바뀐다)."""

    def __init__(self, world, every: int = 2, scenario: Scenario | None = None,
                 base_cfg: Config | None = None, scenario_start: int = 0):
        if world.cfg.passive:
            raise ValueError("선택 없는 대조 실행은 다시 보기로 만들 수 없다(대조 일정 없이 돌리면 전혀 다른 세계가 된다)")
        self.w = world
        self.every = max(1, int(every))
        self.start = world.step_no
        self.scn = scenario
        self.base = base_cfg if base_cfg is not None else world.cfg
        self.scn_start = int(scenario_start)
        self._force = True
        world.drain_events()             # 시작 전의 사건은 버린다. 처음 개체는 상태에서 읽는다
        s = world.s
        cfg = world.cfg
        # 개체 표
        self.id2idx: dict[int, int] = {}
        self.rows: list[list] = []        # [id, parent_id, gen, kind, n_hidden, eta, color, born, died, cause, n_off, eaten]
        self.ate_good: list[int] = []
        self.ate_poison: list[int] = []
        self.kids: list[int] = []
        for j in np.flatnonzero(s.alive).tolist():
            self._add_agent(j, BORN_BEFORE, int(s.birth_step[j]))
        # 먹이 표
        self.food_rows: list[list] = []   # [x, y, kind, 처음 있던 상태, 처음 없던 상태(-1이면 남아 있다), 까닭]
        self.slot_food = np.full(s.fx.shape[0], -1, dtype=np.int64)
        for j in np.flatnonzero(s.f_alive & (s.f_holder < 0)).tolist():    # 든 물건은 내려놓을 때 행이 열린다
            self._add_food(j, self.start)
        # 장면
        self.frames: list[dict] = []
        self.per: dict[str, list] = {n: [] for n in ("good", "food", "eaten", "poisoned", "blind", "gifts", "attacks")}
        self.seen = {"good": 0, "poison": 0, "unknown": 0, "evicted": 0, "burnt": 0, "gave": 0, "spoiled": 0,
                     "picked": 0, "dropped": 0, "stored_out": 0}
        self._acc_good: dict[int, int] = {}
        self._acc_poison: dict[int, int] = {}
        self._acc_gave: set[int] = set()
        self._acc_hit: set[int] = set()
        self._acc = dict(eaten=0, poisoned=0, gifts=0, attacks=0)
        # 좋은 종류, 각본, 먹이 자리
        self.flips: list[dict] = []
        self._good = self._good_at(max(self.start - 1, 0), cfg)
        self.changes: list[dict] = []
        self.strikes: list[dict] = []
        self._scn_prev = {n: getattr(cfg, n) for n in scenario.touched} if scenario is not None else {}
        self._scn_at_start = dict(self._scn_prev)
        self.patches: list[dict] = []
        self._patch_key = None
        self._capture()

    # ---- 표에 넣기 ----
    def _add_agent(self, slot: int, kind: int, born: int, parent: int | None = None, gen: int | None = None) -> int:
        s = self.w.s
        aid = int(s.ids[slot])
        col = np.clip(s.color[slot] * 255.0, 0, 255).astype(np.uint8)
        self.id2idx[aid] = len(self.rows)
        self.rows.append([aid, int(s.parent[slot]) if parent is None else parent,
                          int(s.gen[slot]) if gen is None else gen, kind, int(s.n_hidden[slot]),
                          float(s.eta[slot]), col, born, -1, NO_CAUSE, 0, 0.0])
        self.ate_good.append(0)
        self.ate_poison.append(0)
        self.kids.append(0)
        return len(self.rows) - 1

    def _add_food(self, slot: int, state: int) -> None:
        s = self.w.s
        self.slot_food[slot] = len(self.food_rows)
        self.food_rows.append([float(s.fx[slot]), float(s.fy[slot]), int(s.f_kind[slot]), state, -1, WHY_PRESENT])

    def _food_gone(self, slot: int, state: int, why: int) -> None:
        n = int(self.slot_food[slot])
        if n >= 0:
            self.food_rows[n][4] = state
            self.food_rows[n][5] = why
            self.slot_food[slot] = -1

    def _frame_of(self, state: int) -> int:
        """이 상태(step_no)가 처음 보이는 장면 번호."""
        return _ceil_div(state - self.start, self.every)

    @staticmethod
    def _good_at(step: int, cfg: Config) -> int:
        return int(k.good_kind(cfg.seed, step, cfg.kind_flip_period)) if cfg.food_kinds_enabled else 0

    def _food_snapshot(self):
        s = self.w.s
        return (s.f_alive.copy(), s.f_born.copy(), s.f_kind.copy(), s.fx.copy(), s.fy.copy(), s.f_energy.copy(),
                s.f_holder.copy(), s.f_dropped.copy())

    # ---- 한 스텝 ----
    def advance(self, n: int) -> None:
        for _ in range(int(n)):
            self._step()

    def _apply_scenario(self, t: int, pre):
        """실행기와 같은 자리(스텝 바로 앞)에서 각본을 적용한다. 운석에 사라진 먹이와 풀린 손(들기를 끄거나
        든 개체가 운석에 죽음)을 가려 적고 새 먹이 사진을 돌려준다."""
        w, scn = self.w, self.scn
        t_rel = t - self.scn_start
        hits = scn.strikes_at(t_rel) if t_rel >= 0 else []
        killed = scn.apply(w, self.base, t, self.scn_start, self._force)
        self._force = False
        cur = {n: getattr(w.cfg, n) for n in scn.touched}
        diff = {n: v for n, v in cur.items() if v != self._scn_prev[n]}
        if diff:
            self.changes.append({"step": t, "frame": self._frame_of(t + 1), "values": diff})
            self._scn_prev = cur
        held = bool((pre[6] >= 0).any())
        if not hits and not held:
            return pre
        cfg = w.cfg
        for n, st in enumerate(hits):
            key = scn_mod.strike_key(t, st["index"])
            x, y = st["x"], st["y"]
            if x is None or y is None:
                rx, ry = scn_mod.strike_position(cfg.seed, key, cfg.world_w, cfg.world_h)
                x = rx if x is None else x
                y = ry if y is None else y
            # 같은 스텝에 여럿 떨어지면 죽은 수는 첫 운석에 몰아 적는다(각본은 합계만 돌려준다)
            self.strikes.append({"step": t, "frame": self._frame_of(t + 1), "x": x, "y": y, "radius": st["radius"],
                                 "burn": bool(st["burn_food"]), "killed": int(killed) if n == 0 else 0})
        post = self._food_snapshot()
        for j in np.flatnonzero(pre[0] & ~post[0]).tolist():
            self._food_gone(j, t + 1, WHY_BURNT)
            self.seen["burnt"] += 1
        # 각본 앞에 들려 있다가 풀린 물건은 그 자리에 내려놓은 것으로 적는다(스텝 앞 사진도 풀린 뒤의 것이 된다)
        for j in np.flatnonzero(pre[0] & post[0] & (pre[6] >= 0) & (post[6] < 0)).tolist():
            self._add_food(j, t + 1)
            self.seen["dropped"] += 1
        return post

    def _step(self) -> None:
        w = self.w
        s = w.s
        t = w.step_no
        pre = self._food_snapshot()
        if self.scn is not None:
            pre = self._apply_scenario(t, pre)
        cfg = w.cfg
        good = self._good_at(t, cfg)
        if good != self._good:
            self.flips.append({"step": t, "frame": self._frame_of(t + 1), "good": good})
            self._good = good
        c0 = s.counters.copy()
        w.step()
        dc = s.counters - c0
        self._acc["eaten"] += int(dc[k.C_EATEN])
        self._acc["poisoned"] += int(dc[k.C_POISONED])
        self._acc["gifts"] += int(dc[k.C_GIFTS])
        self._acc["attacks"] += int(dc[k.C_ATTACKS])
        self._read_agents(t)
        self._read_food(t, pre, cfg, good, int(dc[k.C_FOOD_REPLACED]),
                        int(dc[k.C_STORE_EVICTED]) if len(dc) > k.C_STORE_EVICTED else 0)
        if cfg.social_enabled and dc[k.C_GIFTS] > 0:
            self._read_givers(t, cfg)
        if cfg.conflict_enabled and dc[k.C_ATTACKS] > 0:
            for j in np.flatnonzero(s.alive & (s.touch == k.TOUCH_HIT)).tolist():
                self._acc_hit.add(self.id2idx[int(s.ids[j])])
        if (w.step_no - self.start) % self.every == 0:
            self._capture()

    def _read_agents(self, t: int) -> None:
        """이 스텝의 출생·사망 기록으로 개체 표를 고친다(운석으로 죽은 것도 여기에 들어 있다)."""
        s = self.w.s
        births, deaths = self.w.drain_events()
        for d in deaths:
            n = self.id2idx.get(int(d["id"]))
            if n is not None:
                r = self.rows[n]
                r[8], r[9], r[10], r[11] = int(d["step"]), int(d["cause"]), int(d["n_off"]), float(d["eaten"])
        if len(births):
            slot_of = {int(s.ids[j]): j for j in np.flatnonzero(s.alive & (s.birth_step == t)).tolist()}
            for b in births:
                j = slot_of.get(int(b["id"]))
                if j is None:
                    continue    # 태어난 스텝에는 죽지 않으므로 오지 않는 길이다
                self._add_agent(j, int(b["kind"]), int(b["step"]), int(b["parent"]), int(b["gen"]))
                p = self.id2idx.get(int(b["parent"]))
                if p is not None and b["kind"] == k.BIRTH_NATURAL:
                    self.kids[p] += 1

    def _read_food(self, t: int, pre, cfg: Config, good: int, n_evicted: int, n_stored_out: int = 0) -> None:
        """스텝 전후의 먹이를 (칸, 놓인 스텝, 종류, 위치)로 견주어, 사라진 것과 새로 난 것을 가린다.
        들기가 켜진 세계에서는 물건이 움직이므로 위치는 보지 않고, 들림과 내려놓음을 따로 적는다:
        들리면 그 행을 닫고(WHY_PICKED), 내려놓으면 그 자리에서 새 행을 연다."""
        s = self.w.s
        p_alive, p_born, p_kind, p_x, p_y, p_energy, p_holder, p_dropped = pre
        carry = cfg.carry_enabled
        same = s.f_alive & p_alive & (s.f_born == p_born) & (s.f_kind == p_kind)
        if not carry:
            same &= (s.fx == p_x) & (s.fy == p_y)
        gone = np.flatnonzero(p_alive & ~same)
        if len(gone):
            if n_evicted > 0:
                # 상한 장치는 가장 먼저 놓인 것부터(같으면 낮은 칸부터) 치운다. 먹기보다 앞에 일어난다.
                # 들기가 켜진 세계에서는 들판 알만 치운다
                was = np.flatnonzero(p_alive)
                if carry:
                    was = was[(p_dropped[was] == 0) & (p_kind[was] != k.KIND_CORPSE)]
                oldest = was[np.lexsort((was, p_born[was]))][:n_evicted]
                for j in oldest.tolist():
                    self._food_gone(j, t + 1, WHY_EVICTED)
                    self.seen["evicted"] += 1
                gone = np.setdiff1d(gone, oldest)
        rotten = np.zeros(len(gone), dtype=bool)
        if len(gone) and cfg.spoil_enabled:
            # 썩음은 먹기보다 앞에 일어나고, 남은 에너지가 하한 아래로 떨어진 먹이를 비운다(커널과 같은 계산)
            left = p_energy[gone] - p_energy[gone] * (cfg.spoil_rate * self.w.spoil_mult(t))
            rotten = (t - p_born[gone] >= cfg.spoil_after) & (left < cfg.spoil_min_frac * cfg.food_energy)
        if len(gone) and n_stored_out > 0:
            # 저장 칸이 차서 치운 것(C_STORE_EVICTED는 들기 커널(먹기 앞)과 사망 처리(먹기 뒤)에서 오른다): 들지 않은 들판 아닌 물건 가운데 가장 먼저 놓인 것부터.
            # 사라진 것 가운데서 고른다. 그보다 앞에 썩은 것은 빼고, 이 스텝에 내려놓였다가 치워진 것도 받게
            # 스텝 앞에 들려 있던 것(f_dropped가 1이다)도 후보로 둔다
            cand = gone[~rotten & ((p_dropped[gone] == 1) | (p_kind[gone] == k.KIND_CORPSE))]
            out = cand[np.lexsort((cand, p_born[cand]))][:n_stored_out]
            for j in out.tolist():
                self._food_gone(j, t + 1, WHY_STORED_OUT)
                self.seen["stored_out"] += 1
            keep = ~np.isin(gone, out)
            gone, rotten = gone[keep], rotten[keep]
        if len(gone) and cfg.spoil_enabled:
            for j in gone[rotten].tolist():
                self._food_gone(j, t + 1, WHY_SPOILED)
                self.seen["spoiled"] += 1
            gone = gone[~rotten]
        if len(gone):
            # 나머지는 먹힌 것이다. 이 스텝에 먹은 개체(ate) 가운데 닿는 거리 안의, 커널이 먼저 훑는 낮은 칸의 개체가 먹었다
            eaters = np.flatnonzero(s.ate > 0)
            who = np.full(len(gone), -1, dtype=np.int64)
            if len(eaters):
                r = cfg.agent_radius + cfg.food_radius
                gx, gy = p_x[gone], p_y[gone]
                if carry:
                    # 이 스텝에 내려놓였다가 먹힌 물건은 내려놓은 자리에서 먹혔다. 비운 칸에는 그 자리가 남아 있다
                    # (같은 스텝에 다른 물건이 그 칸에 들어왔으면 스텝 앞 자리로 어림한다)
                    freed = ~s.f_alive[gone]
                    gx = np.where(freed, s.fx[gone], gx)
                    gy = np.where(freed, s.fy[gone], gy)
                dx = gx[:, None] - s.x[eaters][None, :]
                dy = gy[:, None] - s.y[eaters][None, :]
                dx -= cfg.world_w * np.round(dx / cfg.world_w)
                dy -= cfg.world_h * np.round(dy / cfg.world_h)
                near = dx * dx + dy * dy <= r * r + 1e-9
                hit = near.any(axis=1)
                who[hit] = eaters[near.argmax(axis=1)[hit]]
            for j, e in zip(gone.tolist(), who.tolist()):
                n = self.id2idx.get(int(s.ids[e])) if e >= 0 else None
                if n is None:
                    self._food_gone(j, t + 1, WHY_UNKNOWN)
                    self.seen["unknown"] += 1
                    continue
                kind = int(p_kind[j])
                if cfg.food_kinds_enabled and kind != k.KIND_CORPSE and kind != good:
                    self._food_gone(j, t + 1, WHY_POISON)
                    self.seen["poison"] += 1
                    self.ate_poison[n] += 1
                    self._acc_poison[n] = self._acc_poison.get(n, 0) + 1
                else:
                    self._food_gone(j, t + 1, WHY_GOOD)
                    self.seen["good"] += 1
                    self.ate_good[n] += 1
                    self._acc_good[n] = self._acc_good.get(n, 0) + 1
        for j in np.flatnonzero(s.f_alive & ~same).tolist():
            self._add_food(j, t + 1)
        if carry:
            # 같은 물건의 손이 바뀐 것. 든 개체가 들자마자 죽으면 손은 그대로 비어 있고 자리만 옮겨진다(들고 내려놓음)
            # 같은 자리에서 들렸다 놓인 것은 자리도 손도 그대로라, 앞 스텝에 놓여 있다가 이번에 들판을 떠난 표시(f_dropped)로도 본다
            moved = same & ((s.fx != p_x) | (s.fy != p_y))
            lifted = same & (p_dropped == 0) & (s.f_dropped == 1)
            for j in np.flatnonzero(same & (p_holder < 0) & ((s.f_holder >= 0) | moved | lifted)).tolist():
                self._food_gone(j, t + 1, WHY_PICKED)
                self.seen["picked"] += 1
            for j in np.flatnonzero(same & (s.f_holder < 0) & ((p_holder >= 0) | moved | lifted)).tolist():
                self._add_food(j, t + 1)
                self.seen["dropped"] += 1

    def _read_givers(self, t: int, cfg: Config) -> None:
        """에너지를 건넨 개체를 어림한다: 주기 출력이 켜져 있고 닿는 거리 안에 이웃이 있던 개체.
        건넬 때의 에너지는 남아 있지 않아 끝난 뒤의 값으로 어림한다(세계의 계수기와 조금 다를 수 있다)."""
        s = self.w.s
        then = s.alive & (s.birth_step != t)
        g = s.act[:, ACT_GIVE].astype(np.float64)
        cand = np.flatnonzero(then & (g > 0) & (s.energy + cfg.give_rate * g > cfg.give_min_energy))
        if not len(cand):
            return
        others = np.flatnonzero(then)
        reach = 2.0 * cfg.agent_radius + 0.5
        ox, oy = s.x[others], s.y[others]
        for a in range(0, len(cand), 256):
            c = cand[a:a + 256]
            dx = s.x[c][:, None] - ox[None, :]
            dy = s.y[c][:, None] - oy[None, :]
            dx -= cfg.world_w * np.round(dx / cfg.world_w)
            dy -= cfg.world_h * np.round(dy / cfg.world_h)
            d2 = dx * dx + dy * dy
            d2[c[:, None] == others[None, :]] = np.inf
            for j in c[(d2 < reach * reach).any(axis=1)].tolist():
                self._acc_gave.add(self.id2idx[int(s.ids[j])])
                self.seen["gave"] += 1

    # ---- 장면 ----
    def _capture(self) -> None:
        w = self.w
        s, cfg = w.s, w.cfg
        a = np.flatnonzero(s.alive)
        idx = np.fromiter((self.id2idx[i] for i in s.ids[a].tolist()), dtype=np.int64, count=len(a))
        order = np.argsort(idx)
        a, idx = a[order], idx[order]
        flags = np.where(s.asleep[a] > 0, F_ASLEEP, 0).astype(np.uint8)
        flags[s.a_held[a] >= 0] |= F_CARRY
        if self._acc_good or self._acc_poison or self._acc_gave or self._acc_hit:
            for n, i in enumerate(idx.tolist()):
                f = min(self._acc_good.get(i, 0), 3) | (min(self._acc_poison.get(i, 0), 3) << 2)
                if i in self._acc_gave:
                    f |= F_GAVE
                if i in self._acc_hit:
                    f |= F_HIT
                flags[n] |= f
        fr = {
            "idx": idx,
            "x": _q16(s.x[a], cfg.world_w), "y": _q16(s.y[a], cfg.world_h),
            "head": (np.mod(s.heading[a], 2.0 * math.pi) / (2.0 * math.pi) * 256.0).astype(np.int64).astype(np.uint8),
            "energy": np.clip(s.energy[a] / cfg.e_max * 255.0, 0, 255).astype(np.uint8),
            "flags": flags,
        }
        if cfg.social_enabled:
            fr["sig"] = np.clip(np.round(s.sig[a] * 127.0), -127, 127).astype(np.int8)
        self.frames.append(fr)
        fa = s.f_alive
        n_food = int(fa.sum())
        bad = 0
        if cfg.food_kinds_enabled:
            kinds = s.f_kind[fa]
            bad = int(((kinds != k.KIND_CORPSE) & (kinds != self._good)).sum())
        self.per["good"].append(self._good)
        self.per["food"].append(n_food)
        self.per["blind"].append(bad / max(n_food, 1))
        for name in ("eaten", "poisoned", "gifts", "attacks"):
            self.per[name].append(self._acc[name])
            self._acc[name] = 0
        self._acc_good, self._acc_poison, self._acc_gave, self._acc_hit = {}, {}, set(), set()
        # 먹이가 뭉쳐 나는 자리: 옮겨 갈 때만 다시 적는다
        if cfg.n_patches > 0:
            t = w.step_no
            key = (cfg.n_patches, cfg.patch_life, cfg.patch_sigma,
                   tuple((t + (i * cfg.patch_life) // cfg.n_patches) // cfg.patch_life for i in range(cfg.n_patches)))
            if key != self._patch_key:
                self._patch_key = key
                pts = [[round(float(v), 1) for v in k.patch_center(cfg.seed, i, t, cfg.n_patches, cfg.patch_life,
                                                                    cfg.world_w, cfg.world_h)]
                       for i in range(cfg.n_patches)]
                self.patches.append({"frame": len(self.frames) - 1, "sigma": cfg.patch_sigma, "pts": pts})

    # ---- 마무리 ----
    def _stars(self, agents: np.ndarray, end: int) -> list[dict]:
        """따라가 볼 만한 개체 넷: 가장 많이 먹은, 가장 오래 산, 자식을 가장 많이 낳은, 독을 가장 많이 먹은."""
        if not len(agents):
            return []
        last = np.where(agents["died"] >= 0, agents["died"], end)
        seen_for = last - np.maximum(agents["born"], self.start)      # 다시 보기 안에서 보이는 길이
        picks = [("가장 많이 먹은 개체", agents["ate_good"].astype(np.int64), "끼"),
                 ("가장 오래 산 개체", (last - agents["born"]).astype(np.int64), "스텝"),
                 ("자식을 가장 많이 낳은 개체", agents["kids"].astype(np.int64), "마리"),
                 ("독을 가장 많이 먹은 개체", agents["ate_poison"].astype(np.int64), "번")]
        out, used = [], set()
        for why, score, unit in picks:
            # 같은 점수면 화면에 오래 보이는 개체를 고른다
            for n in np.lexsort((-seen_for, -score)).tolist():
                if n not in used and score[n] > 0:
                    used.add(n)
                    out.append({"agent": int(n), "why": why, "n": int(score[n]), "unit": unit})
                    break
        return out

    @staticmethod
    def _sig_min(cols) -> float | None:
        """신호 고리 문턱. 쓰이지 않는 신호도 0~1 사이를 떠다녀서 고정 문턱(0.5)이면 절반이 고리를 단다
        (공격 세계 20만 스텝에서 52%). 그래서 이 기록에서 가장 센 5%만 고리를 달게 한다."""
        if "sig" not in cols or not len(cols["sig"]):
            return None
        mg = np.abs(cols["sig"].astype(np.float32).reshape(-1, 2) / 127.0).max(axis=1)
        return round(max(SIG_MIN_FLOOR, float(np.percentile(mg, 95))), 4)

    def finish(self) -> Replay:
        w = self.w
        s, cfg = w.s, w.cfg
        end = w.step_no
        n_frames = len(self.frames)
        agents = np.zeros(len(self.rows), dtype=AGENT_DTYPE)
        for n, r in enumerate(self.rows):
            agents[n] = (r[0], self.id2idx.get(r[1], -1), r[2], r[3], min(r[4], 255), r[5], r[6],
                         r[7], r[8], r[9], min(r[10], 65535), r[11],
                         min(self.ate_good[n], 65535), min(self.ate_poison[n], 65535), min(self.kids[n], 65535))
        for j in np.flatnonzero(s.alive).tolist():       # 끝까지 산 개체의 자식 수와 먹은 양은 상태에서 읽는다
            n = self.id2idx[int(s.ids[j])]
            agents["n_off"][n] = min(int(s.n_off[j]), 65535)
            agents["eaten"][n] = float(s.eaten[j])
        rows = []
        for x, y, kind, a, b, why in self.food_rows:
            f_in = self._frame_of(a)
            f_out = n_frames if b < 0 else self._frame_of(b)
            if f_in < f_out:                 # 장면 사이에만 있던 먹이는 뺀다
                rows.append((min(int(x / cfg.world_w * 65536.0), 65535), min(int(y / cfg.world_h * 65536.0), 65535),
                             kind, f_in, f_out, why))
        food = np.array(rows, dtype=FOOD_DTYPE) if rows else np.zeros(0, dtype=FOOD_DTYPE)
        food = food[np.argsort(food["f_in"], kind="stable")]
        off = np.concatenate([[0], np.cumsum([len(f["idx"]) for f in self.frames])]).astype("<u4")
        cols = {name: np.concatenate([f[name] for f in self.frames]) for name in self.frames[0]}
        cols["idx"] = cols["idx"].astype("<u2" if len(agents) < 65536 else "<u4")
        per = {"good": np.array(self.per["good"], dtype=np.uint8), "blind": np.array(self.per["blind"], dtype="<f4")}
        for name in ("food", "eaten", "poisoned", "gifts", "attacks"):
            per[name] = np.array(self.per[name], dtype="<u4")
        info = None
        if self.scn is not None:
            base = {n: getattr(self.base, n) for n in self.scn.touched}
            info = {"name": self.scn.name, "description": self.scn.description, "base": base,
                    "at_start": {n: v for n, v in self._scn_at_start.items() if v != base[n]},
                    "changes": self.changes, "strikes": self.strikes}
        meta = {
            "v": 1, "title": None, "ckpt_step": self.start, "start": self.start, "every": self.every,
            "n_frames": n_frames, "end": end, "w": cfg.world_w, "h": cfg.world_h,
            "on": {"mind": cfg.mind_enabled, "social": cfg.social_enabled, "niche": cfg.niche_enabled,
                   "conflict": cfg.conflict_enabled, "kinds": cfg.food_kinds_enabled, "carry": cfg.carry_enabled},
            "cfg": {"agent_radius": cfg.agent_radius, "food_radius": cfg.food_radius, "vision_fov": cfg.vision_fov,
                    "vision_range": cfg.vision_range, "e_max": cfg.e_max, "e_repro": cfg.e_repro,
                    "max_age": cfg.max_age, "kind_flip_period": cfg.kind_flip_period, "poison": cfg.poison,
                    "food_energy": cfg.food_energy, "eta_max": cfg.eta_max, "h_max": cfg.h_max},
            "intro": intro_sentences(cfg, self.scn), "flips": self.flips, "patches": self.patches,
            "stars": self._stars(agents, end), "scenario": info, "notes": [], "history": None, "orig": [],
            "seen": dict(self.seen), "sig_min": self._sig_min(cols),
        }
        return Replay(self.start, self.every, off, cols, agents, food, per, dict(self.seen), meta,
                      cfg.world_w, cfg.world_h)


# =========================== 체크포인트에서 ===========================
def _run_name(path: Path) -> str:
    return path.parent.parent.name if path.parent.name == "checkpoints" else path.stem


def _read_scenario(run_dir: Path, world, notes: list[str], overrides: list[str] | None = None):
    """실행 폴더의 각본과 바탕 설정. 각본이 없으면 (None, 세계의 설정, 0).
    바탕 설정은 저장된 config.json에 덮어쓰기를 얹고, 배열 크기(입출력 자리)는 실제 세계의 것으로 맞춘다.
    각본이 첫 스텝에 바탕 설정으로 세계를 다시 맞추기 때문에, 바탕이 세계와 크기가 다르면 죽고
    덮어쓰기가 빠져 있으면 조용히 지워진다."""
    try:
        meta = json.loads((run_dir / "run.json").read_text())
    except (OSError, ValueError):
        return None, world.cfg, 0
    if not isinstance(meta, dict) or meta.get("scenario_start") is None:
        return None, world.cfg, 0
    try:
        scn = Scenario.load(run_dir / "scenario.json")
        d = json.loads((run_dir / "config.json").read_text())
        # 옛 실행이나 갈라 켠 실행은 저장된 크기가 세계와 다르다. 체크포인트를 읽은 세계의 크기에 맞춘다
        d["n_spare_in"], d["n_spare_out"] = world.cfg.n_spare_in, world.cfg.n_spare_out
        base = Config.from_dict(d).with_overrides(overrides)
    except (OSError, ValueError, TypeError):
        notes.append("이 실행에는 각본이 걸려 있었지만 읽을 수 없어, 다시 돌릴 때는 각본 적용 안 함.")
        return None, world.cfg, 0
    return scn, base, int(meta["scenario_start"])


def _read_history(run_dir: Path, ckpt_step: int, end: int, scn, base, scn_start):
    """실행 폴더의 stats.jsonl에서 긴 흐름(체크포인트까지)과, 다시 돌린 구간의 원래 개체 수를 읽는다."""
    try:
        f = open(run_dir / "stats.jsonl", "rb")
    except OSError:
        return None, []
    rows, orig = [], []
    with f:
        for line in f:
            if not line.endswith(b"\n"):
                break                       # 쓰다 만 마지막 줄
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if not isinstance(r, dict) or not isinstance(r.get("step"), int) or "pop" not in r:
                continue
            if r["step"] <= ckpt_step:
                if rows and r["step"] <= rows[-1]["step"]:
                    continue
                rows.append({n: r.get(n) for n in ("step", "pop", "eaten", "poisoned", "hidden_mean", "w_steps")})
            elif r["step"] <= end:
                orig.append([r["step"], int(r["pop"])])
    if not rows:
        return None, orig
    g = _ceil_div(len(rows), HISTORY_POINTS)
    hist = {"step": [], "pop": [], "hidden": [], "poison": []}
    prev = None
    for a in range(0, len(rows), g):
        part = rows[a:a + g]
        last = part[-1]
        hist["step"].append(last["step"])
        hist["pop"].append(round(sum(r["pop"] for r in part) / len(part), 1))
        hid = [r["hidden_mean"] for r in part if r["hidden_mean"] is not None]
        hist["hidden"].append(round(sum(hid) / len(hid), 3) if hid else None)
        ratio = None
        if last["eaten"] is not None and last["poisoned"] is not None:
            if prev is not None and prev["eaten"] is not None and prev["poisoned"] is not None:
                de = last["eaten"] - prev["eaten"]
                ratio = (last["poisoned"] - prev["poisoned"]) / de if de > 0 else None
            elif prev is None and g == 1 and last["w_steps"] == last["step"] and last["eaten"] > 0:
                ratio = last["poisoned"] / last["eaten"]      # 0스텝에서 시작한 실행의 첫 줄
        hist["poison"].append(None if ratio is None else round(min(max(ratio, 0.0), 1.0), 4))
        prev = last
    if scn is not None:
        ref = {n: getattr(base, n) for n in scn.touched}
        hist["scn"] = [int(scn.values(base, t - scn_start) != ref) for t in hist["step"]]
        t0 = hist["step"][0]
        hist["strikes"] = [scn_start + t for t in scn.strike_times(ckpt_step - scn_start)
                           if scn_start + t >= t0][-200:]
    return hist, orig


def _next_flip(cfg: Config, t0: int) -> int | None:
    """t0 뒤에 좋은 종류가 실제로 바뀌는 첫 스텝. 40구간 안에 없으면 None."""
    if not cfg.food_kinds_enabled:
        return None
    p = cfg.kind_flip_period
    b = (t0 // p + 1) * p
    for _ in range(40):
        if k.good_kind(cfg.seed, b, p) != k.good_kind(cfg.seed, b - 1, p):
            return b
        b += p
    return None


def default_steps(cfg: Config) -> int:
    """기본 길이. 먹이 종류 세계는 독이 다시 정해지는 순간을 두 번 담는다."""
    if cfg.food_kinds_enabled:
        return int(min(max(2 * cfg.kind_flip_period, 1500), 6000))
    return 1500


def record(ckpt_path, steps: int | None = None, every: int = 2, from_flip: bool = False,
           title: str | None = None, overrides: list[str] | None = None) -> Replay:
    """체크포인트를 읽어 steps 스텝을 다시 돌리며 장면을 적는다. 체크포인트와 실행 폴더는 읽기만 한다.
    overrides(["poison=0", ...])를 주면 그 설정으로 바꿔 돌린다(같은 체크포인트에서 설정 하나만 바꾼 대조)."""
    path = Path(ckpt_path)
    overrides = list(overrides or [])
    if overrides:     # 설정을 먼저 만들어 읽는다(preset=ext면 확장 크기로)
        world, cmeta = ckpt.load(path, ckpt.read_config(path).with_overrides(overrides))
        world.calm_new_outputs(cmeta["enlarged_out"][0])
    else:
        world, cmeta = ckpt.load(path)
    if cmeta["config"].get("passive"):     # 덮어쓴 설정이 아니라 저장된 설정으로 가린다
        raise ValueError("선택 없는 대조 실행의 체크포인트는 다시 보기로 만들 수 없다"
                         "(대조 일정 없이 돌리면 전혀 다른 세계가 된다). 본 실행의 체크포인트를 쓴다.")
    run_dir = path.parent.parent
    notes: list[str] = []
    if cmeta["lifetime_reset"]:
        notes.append("크기를 올려 읽어 개체의 평생 학습 상태(" + ", ".join(cmeta["lifetime_reset"])
                     + ")는 0에서 다시 시작했습니다.")
    if overrides:
        notes.append("원래 실행과 설정을 바꿔 다시 돌렸습니다: " + ", ".join(overrides) + ".")
    if cmeta.get("source_hash") not in (None, SOURCE_HASH):
        notes.append("이 체크포인트는 지금과 다른 판의 코드로 저장됐습니다. 다시 돌린 장면이 원래 실행이 "
                     "그 뒤에 겪은 것과 스텝 단위로 같지는 않을 수 있습니다.")
    scn, base, scn_start = _read_scenario(run_dir, world, notes, overrides)
    ckpt_step = world.step_no
    if steps is None:
        steps = default_steps(world.cfg)
    if from_flip:
        b = _next_flip(world.cfg, ckpt_step)
        if b is None:
            notes.append("독이 바뀌는 순간을 찾지 못해 체크포인트에서 바로 시작했습니다.")
        else:
            lead = min(FLIP_LEAD, world.cfg.kind_flip_period // 4)
            force = True
            while world.step_no < b - lead:           # 장면을 적지 않고 거기까지 간다
                if scn is not None:
                    scn.apply(world, base, world.step_no, scn_start, force)
                    force = False
                world.step()
                world.drain_events()
    rec = Recorder(world, every, scn, base, scn_start)
    rec.advance(int(steps))
    rep = rec.finish()
    hist, orig = _read_history(run_dir, ckpt_step, world.step_no, scn, base, scn_start)
    rep.meta.update(title=title or _run_name(path), ckpt_step=ckpt_step, notes=notes, history=hist, orig=orig,
                    overrides=overrides)
    return rep


# =========================== 내보내기 ===========================
def pack(rep: Replay) -> tuple[dict, bytes]:
    """장면 자료를 이진 덩어리 하나로 묶는다. 돌려주는 설명(meta)의 arrays에 이름별 [자료형, 자리, 개수]가 있다."""
    arrays: dict[str, np.ndarray] = {"fr_off": rep.off}
    for name, v in rep.cols.items():
        arrays["fr_" + name] = v.reshape(-1)
    a = rep.agents
    for name, col in (("id", "id"), ("parent", "parent"), ("gen", "gen"), ("kind", "kind"), ("hidden", "n_hidden"),
                      ("eta", "eta"), ("color", "color"), ("born", "born"), ("died", "died"), ("cause", "cause"),
                      ("off", "n_off"), ("eaten", "eaten"), ("good", "ate_good"), ("poison", "ate_poison"),
                      ("kids", "kids")):
        arrays["ag_" + name] = np.ascontiguousarray(a[col]).reshape(-1)
    for name, col in (("x", "x"), ("y", "y"), ("kind", "kind"), ("in", "f_in"), ("out", "f_out"), ("why", "why")):
        arrays["fd_" + name] = np.ascontiguousarray(rep.food[col])
    arrays.update(f_good=rep.good_kind, f_food=rep.n_food, f_eaten=rep.c_eaten, f_poisoned=rep.c_poisoned,
                  f_blind=rep.blind, f_gifts=rep.gifts, f_attacks=rep.attacks)
    chunks, directory, pos = [], {}, 0
    for name, v in arrays.items():
        v = np.ascontiguousarray(v)
        v = v.astype(v.dtype.newbyteorder("<"), copy=False)
        code = v.dtype.kind + str(v.dtype.itemsize)       # u1, i1, u2, u4, i4, f4, f8
        raw = v.tobytes()
        directory[name] = [code, pos, int(v.size)]
        pad = (-len(raw)) % 8                             # 자리마다 8바이트 경계에 맞춘다(화면이 바로 배열로 읽는다)
        chunks.append(raw + b"\0" * pad)
        pos += len(raw) + pad
    meta = dict(rep.meta)
    meta["arrays"] = directory
    return meta, b"".join(chunks)


def export_html(rep: Replay, standalone: bool = True) -> str:
    """화면 틀(replay.html)에 자료를 넣어 한 파일로 만든다. standalone이 아니면 아티팩트용으로 내용만 낸다."""
    meta, blob = pack(rep)
    page = TEMPLATE.read_text(encoding="utf-8")
    if _DATA_MARK not in page:
        raise ValueError("화면 틀에 자료 자리가 없다")
    data = ('<script type="application/json" id="replay-meta">'
            + json.dumps(meta, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
            + '</script>\n<script type="application/octet-stream" id="replay-bin">'
            + base64.b64encode(blob).decode("ascii") + "</script>")
    page = page.replace(_DATA_MARK, data)
    m = _HEAD_RE.search(page)
    head = m.group(1).strip() if m else ""
    body = _HEAD_RE.sub("", page).strip()
    if not standalone:
        return head + "\n" + body + "\n"
    return ('<!doctype html>\n<html lang="ko">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            + head + "\n</head>\n<body>\n" + body + "\n</body>\n</html>\n")


def read_embedded(html: str) -> tuple[dict, dict]:
    """내보낸 파일에서 자료를 다시 꺼낸다(시험과 점검용)."""
    meta = json.loads(_META_RE.search(html).group(1).replace("<\\/", "</"))
    blob = base64.b64decode(_BIN_RE.search(html).group(1))
    arrays = {name: np.frombuffer(blob, dtype="<" + code, count=n, offset=pos)
              for name, (code, pos, n) in meta["arrays"].items()}
    return meta, arrays


def write(ckpt_path, out, steps: int | None = None, every: int = 2, from_flip: bool = False,
          title: str | None = None, standalone: bool = True, max_bytes: int = MAX_BYTES) -> Replay:
    """다시 보기 파일을 쓴다. 크기가 상한을 넘으면 장면 간격을 넓혀 다시 적는다."""
    every = max(1, int(every))
    while True:
        rep = record(ckpt_path, steps, every, from_flip, title)
        html = export_html(rep, standalone).encode("utf-8")
        if len(html) <= max_bytes:
            break
        frames = sum(v.nbytes for v in rep.cols.values()) * 4 // 3      # 장면 간격에 따라 줄어드는 몫
        fixed = len(html) - frames
        wider = max(every + 1, math.ceil(every * frames / max(max_bytes - fixed, 1) * 1.02)) if fixed < max_bytes else 0
        if not 0 < wider <= MAX_EVERY:
            raise ValueError(f"다시 보기 파일의 크기({len(html):,}바이트)를 상한({max_bytes:,}바이트) 아래로 "
                             "줄일 수 없다. --steps를 줄인다.")
        every = wider
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ckpt.atomic_write_bytes(out, html)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="체크포인트에서 세계를 다시 돌려, 혼자 열리는 다시 보기 파일(HTML)을 만든다")
    ap.add_argument("--ckpt", type=Path, required=True, help="체크포인트 npz(같은 이름의 json이 옆에 있어야 한다)")
    ap.add_argument("--out", type=Path, help="쓸 파일(기본: runs/replay-<실행 이름>.html)")
    ap.add_argument("--steps", type=int, help="다시 돌릴 스텝 수(기본: 먹이 종류 세계는 독이 두 번 다시 정해지는 길이)")
    ap.add_argument("--every", type=int, default=2, help="몇 스텝마다 장면을 적을지(기본 2)")
    ap.add_argument("--from-flip", action="store_true", help="독이 바뀌는 다음 순간의 조금 앞에서 시작한다")
    ap.add_argument("--title", help="화면에 적을 세계 이름(기본: 실행 폴더 이름)")
    ap.add_argument("--artifact", action="store_true", help="뼈대(<html>, <body>) 없이 내용만 쓴다(아티팩트로 올릴 때)")
    ap.add_argument("--max-mb", type=float, default=MAX_BYTES / 1e6, help="파일 크기 상한(MB)")
    args = ap.parse_args(argv)
    out = args.out or Path("runs") / f"replay-{_run_name(args.ckpt)}.html"
    try:
        rep = write(args.ckpt, out, args.steps, args.every, args.from_flip, args.title,
                    not args.artifact, int(args.max_mb * 1e6))
    except (ValueError, OSError) as e:
        print(f"못 만들었다: {e}", file=sys.stderr)
        return 1
    seen = rep.seen
    eaten = int(rep.c_eaten.sum())
    found = seen["good"] + seen["poison"] + seen["unknown"]
    print(f"{out} ({out.stat().st_size / 1e6:.1f}MB): {rep.meta['start']:,}스텝부터 {rep.meta['end'] - rep.start:,}스텝, "
          f"장면 {rep.n_frames}장(간격 {rep.every}), 개체 {len(rep.agents)}마리, 먹이 {len(rep.food)}알")
    print(f"먹은 수 {eaten}(독 {int(rep.c_poisoned.sum())}) 가운데 누가 먹었는지 가린 것 {seen['good'] + seen['poison']}, "
          f"모름 {seen['unknown']}, 같은 스텝에 났다 먹혀 못 본 것 {eaten - found}")
    for n in rep.meta["notes"]:
        print("주의:", n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
