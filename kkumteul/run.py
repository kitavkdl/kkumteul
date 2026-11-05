"""실행기. 세계를 계속 돌리면서 사건·통계·화면용 프레임·체크포인트를 실행 폴더에 쓴다.

    python -m kkumteul.run new    --run runs/main [--set key=value ...] [--steps N]
    python -m kkumteul.run resume --run runs/main [--steps N]
    python -m kkumteul.run branch --from runs/main --run runs/what-if [--at STEP]
                                  [--set key=value ...] [--new-stream] [--passive]
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np

from . import checkpoint as ckpt
from . import events
from . import kernels as k
from .config import Config
from .world import World

FRAME_MAGIC = b"WSF2"
FRAME_HEADER = struct.Struct("<4sIqIIff")   # magic, 전체 바이트, step, 개체 수, 먹이 수, 세계 너비, 높이
FRAME_TAIL = struct.Struct("<HHBBH")        # 격자 너비, 높이, 켜진 단계(비트: 1 마음, 2 사회, 4 판), 예비, 예비


def encode_frame(world: World) -> bytes:
    s = world.s
    a = np.flatnonzero(s.alive)
    f = np.flatnonzero(s.f_alive)
    n, m = len(a), len(f)
    cfg = world.cfg
    body = np.empty((n, 8), dtype="<f4")
    body[:, 0] = s.x[a]
    body[:, 1] = s.y[a]
    body[:, 2] = s.heading[a]
    body[:, 3] = s.energy[a]
    body[:, 4] = s.age[a]
    body[:, 5] = s.gen[a]
    body[:, 6] = s.n_hidden[a]
    body[:, 7] = s.n_off[a]
    ids = (s.ids[a] & 0xFFFFFFFF).astype("<u4")   # 화면용이라 아래 32비트만(43억 번째 출생 뒤 돌아온다)
    col = np.empty((n, 4), dtype=np.uint8)
    col[:, :3] = np.clip(s.color[a] * 255.0, 0, 255).astype(np.uint8)
    col[:, 3] = np.clip(s.tag[a], 0, 255).astype(np.uint8)
    food = np.empty((m, 2), dtype="<u2")
    food[:, 0] = np.minimum(s.fx[f] / cfg.world_w * 65536.0, 65535).astype("<u2")
    food[:, 1] = np.minimum(s.fy[f] / cfg.world_h * 65536.0, 65535).astype("<u2")
    # 2~4단계 볼거리: 개체마다 잠·신호 2개·학습률, 그리고 칸마다 길과 거름(판이 켜져 있을 때만)
    extra = np.zeros((n, 4), dtype=np.uint8)
    extra[:, 0] = np.where(s.asleep[a] > 0, 255, 0)
    extra[:, 1] = np.clip((s.sig[a, 0] + 1.0) * 127.5, 0, 255).astype(np.uint8)
    extra[:, 2] = np.clip((s.sig[a, 1] + 1.0) * 127.5, 0, 255).astype(np.uint8)
    extra[:, 3] = np.clip(s.eta[a] / max(cfg.eta_max, 1e-9) * 255.0, 0, 255).astype(np.uint8)
    flags = int(cfg.mind_enabled) | (int(cfg.social_enabled) << 1) | (int(cfg.niche_enabled) << 2)
    tail = FRAME_TAIL.pack(cfg.grid_w, cfg.grid_h, flags, 0, 0)
    grids = b""
    if cfg.niche_enabled:
        trail = np.clip(s.trail / cfg.trail_sat * 255.0, 0, 255).astype(np.uint8)
        soil = np.clip(np.sqrt(s.soil) * 64.0, 0, 255).astype(np.uint8)   # 거름 16이면 가득
        grids = trail.tobytes() + soil.tobytes()
    total = (FRAME_HEADER.size + body.nbytes + ids.nbytes + col.nbytes + food.nbytes
             + extra.nbytes + FRAME_TAIL.size + len(grids))
    head = FRAME_HEADER.pack(FRAME_MAGIC, total, world.step_no, n, m, cfg.world_w, cfg.world_h)
    return b"".join([head, body.tobytes(), ids.tobytes(), col.tobytes(), food.tobytes(),
                     extra.tobytes(), tail, grids])


def _sd_notify(msg: str) -> None:
    """systemd에 살아 있음을 알린다(Type=notify, WatchdogSec). systemd 밖에서는 아무 일도 안 한다."""
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return
    if addr.startswith("@"):
        addr = "\0" + addr[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.sendto(msg.encode(), addr)
    except OSError:
        pass


def acquire_lock(run_dir: Path):
    """실행 폴더 하나에는 실행기 하나만. 둘이 같이 쓰면 기록 파일이 어긋난다."""
    run_dir.mkdir(parents=True, exist_ok=True)
    f = open(run_dir / ".lock", "w")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        raise RuntimeError(f"이 실행 폴더를 다른 실행기가 쓰고 있다: {run_dir}")
    return f


def parent_run_dir(run_dir: Path, meta: dict) -> Path:
    """분기·대조 실행의 부모 폴더. 폴더째 다른 기계로 옮긴 경우를 위해 상대 경로도 본다."""
    p = Path(meta["parent_run"])
    if not (p / "run.json").exists() and meta.get("parent_run_rel"):
        q = (run_dir / meta["parent_run_rel"]).resolve()
        if (q / "run.json").exists():
            return q
    return p


class Runner:
    def __init__(self, run_dir: Path, world: World, run_meta: dict, lock):
        self.dir = run_dir
        self.world = world
        self.meta = run_meta
        self.lock = lock
        self.ckpt_dir = run_dir / "checkpoints"
        self.frame_dir = run_dir / "frames"
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.f_stats = open(run_dir / "stats.jsonl", "ab")
        self.f_key = open(run_dir / "keyframes.bin", "ab")
        self.f_seg = None
        self.seg_frames = 0
        self.stop = False
        self.stop_reason = ""
        self.dense_until = -1
        self.passive_last_step = -1
        self.win_start_wall = time.time()
        self.win_start_step = world.step_no
        self.win_counters = world.s.counters.copy()
        self.win_life_sum = 0.0
        self.win_life_n = 0
        self.win_eaten_sum = 0.0

    # ---- 만들기 ----
    @classmethod
    def new(cls, run_dir: Path, cfg: Config, meta: dict | None = None) -> "Runner":
        if (run_dir / "run.json").exists():
            raise FileExistsError(f"이미 있는 실행이다: {run_dir}")
        lock = acquire_lock(run_dir)
        world = World(cfg)
        m = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "kind": "new"}
        m.update(meta or {})
        r = cls._create(run_dir, world, m, lock)
        r.checkpoint()
        return r

    @classmethod
    def _create(cls, run_dir: Path, world: World, meta: dict, lock) -> "Runner":
        world.cfg.save(run_dir / "config.json")
        ckpt.atomic_write_bytes(run_dir / "run.json", json.dumps(meta, indent=1).encode())
        r = cls(run_dir, world, meta, lock)
        r._load_passive()
        return r

    @classmethod
    def resume(cls, run_dir: Path) -> "Runner":
        meta = json.loads((run_dir / "run.json").read_text())
        lock = acquire_lock(run_dir)
        world = cmeta = None
        # 가장 최근 체크포인트가 깨졌으면 그 앞 것으로 내려간다
        for step, path in reversed(ckpt.list_checkpoints(run_dir / "checkpoints")):
            try:
                world, cmeta = ckpt.load(path)
                break
            except Exception as e:   # noqa: BLE001 - 어떤 식으로 깨졌든 다음 것을 본다
                print(f"주의: 체크포인트 {path.name}을 못 읽었다({e}). 그 앞 것을 쓴다.", flush=True)
        if world is None:
            lock.close()
            raise FileNotFoundError(f"쓸 수 있는 체크포인트가 없다: {run_dir}")
        # 체크포인트 뒤에 쓴 기록은 같은 내용으로 다시 만들어지므로 잘라 낸다
        seg = cmeta.get("event_seg", 0)
        events.truncate_after(run_dir, "births", seg, cmeta.get("births_bytes", 0))
        events.truncate_after(run_dir, "deaths", seg, cmeta.get("deaths_bytes", 0))
        for name, key in (("stats.jsonl", "stats_bytes"), ("keyframes.bin", "key_bytes")):
            p = run_dir / name
            if p.exists() and key in cmeta and p.stat().st_size > cmeta[key]:
                with open(p, "r+b") as f:
                    f.truncate(cmeta[key])
        for p in (run_dir / "frames").glob("seg_*.bin"):
            if int(p.stem.split("_")[1]) > world.step_no:
                p.unlink()
        r = cls(run_dir, world, meta, lock)
        r._load_passive()
        return r

    @classmethod
    def branch(cls, src: Path, run_dir: Path, at: int | None, overrides: list[str],
               new_stream: bool, passive: bool) -> "Runner":
        if (run_dir / "run.json").exists():
            raise FileExistsError(f"이미 있는 실행이다: {run_dir}")
        src = Path(src)
        if src.suffix == ".npz":
            path = src
            src_run = src.parent.parent
        else:
            src_run = src
            cands = ckpt.list_checkpoints(src / "checkpoints")
            if at is not None:
                cands = [c for c in cands if c[0] == at]
            if not cands:
                raise FileNotFoundError(f"체크포인트를 못 찾았다: {src} at={at}")
            path = cands[-1][1]
        world, cmeta = ckpt.load(path)
        old = world.cfg
        cfg = old.with_overrides(overrides)
        d = cfg.to_dict()
        if new_stream and d["seed"] == old.seed:
            d["seed"] = old.seed + 1000003
        if passive:
            d["passive"] = True
        cfg = Config.from_dict(d)
        world.set_config(cfg)
        diff = {key: [getattr(old, key), val] for key, val in d.items() if getattr(old, key) != val}
        lock = acquire_lock(run_dir)
        meta = {
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "kind": "passive" if passive else "branch",
            "parent_run": str(src_run.resolve()),
            "parent_run_rel": os.path.relpath(src_run.resolve(), run_dir.resolve()),
            "parent_checkpoint": path.name,
            "parent_npz_sha256": cmeta["npz_sha256"],
            "parent_step": cmeta["step"],
            "config_diff": diff,
            "rng": "new_stream" if d["seed"] != old.seed else "same_stream",
        }
        r = cls._create(run_dir, world, meta, lock)
        r.checkpoint()
        return r

    def _load_passive(self) -> None:
        if not self.world.cfg.passive:
            return
        parent = parent_run_dir(self.dir, self.meta)
        pcfg = json.loads((parent / "config.json").read_text())
        seg_steps = pcfg.get("event_segment_steps", self.world.cfg.event_segment_steps)
        start = int(self.meta["parent_step"])
        b = events.load(parent, "births", seg_steps, start)
        d = events.load(parent, "deaths", seg_steps, start)
        self.world.set_passive_schedule(b, d)
        self.passive_last_step = int(max(b["step"].max(initial=start), d["step"].max(initial=start)))

    # ---- 기록 ----
    def _cur_seg(self) -> int:
        return max(self.world.step_no - 1, 0) // self.world.cfg.event_segment_steps

    def flush_events(self) -> None:
        cfg = self.world.cfg
        b, d = self.world.drain_events()
        events.append(self.dir, "births", b, cfg.event_segment_steps)
        events.append(self.dir, "deaths", d, cfg.event_segment_steps)
        if len(d):
            self.win_life_sum += float(d["age"].sum())
            self.win_life_n += len(d)
            self.win_eaten_sum += float(d["eaten"].sum())

    def write_stats(self) -> dict:
        w = self.world
        now = time.time()
        c = w.s.counters
        dc = c - self.win_counters
        steps = w.step_no - self.win_start_step
        row = w.summary()
        bal = w.energy_balance()
        row.update({
            "wall": round(now, 2),
            "sps": round(steps / max(now - self.win_start_wall, 1e-9), 1),
            "w_steps": int(steps),
            "w_births": int(dc[k.C_BIRTHS]),
            "w_deaths": int(dc[k.C_DEATH_STARVE] + dc[k.C_DEATH_AGE] + dc[k.C_DEATH_PASSIVE]),
            "w_starve": int(dc[k.C_DEATH_STARVE]),
            "w_age": int(dc[k.C_DEATH_AGE]),
            "w_reseeds": int(dc[k.C_RESEED]),
            "w_blocked": int(dc[k.C_BLOCKED]),
            "w_eaten": int(dc[k.C_EATEN]),
            "w_life_mean": round(self.win_life_sum / self.win_life_n, 1) if self.win_life_n else None,
            "w_eaten_per_life": round(self.win_eaten_sum / self.win_life_n, 3) if self.win_life_n else None,
            "e_residual_rel": bal["residual"] / max(bal["spawn"] + bal["inject"], 1.0),
            "e_discard": bal["discard"],
        })
        self.f_stats.write((json.dumps(row) + "\n").encode())
        self.f_stats.flush()
        self.win_start_wall, self.win_start_step = now, w.step_no
        self.win_counters = c.copy()
        self.win_life_sum = self.win_eaten_sum = 0.0
        self.win_life_n = 0
        return row

    def write_frame(self) -> None:
        cfg = self.world.cfg
        data = encode_frame(self.world)
        if self.f_seg is None or self.seg_frames >= cfg.frames_per_segment:
            if self.f_seg:
                self.f_seg.close()
            self.f_seg = open(self.frame_dir / f"seg_{self.world.step_no:012d}.bin", "wb")
            self.seg_frames = 0
            segs = sorted(self.frame_dir.glob("seg_*.bin"))
            for old in segs[:-cfg.frame_ring_segments]:
                old.unlink()
        self.f_seg.write(data)
        self.f_seg.flush()
        self.seg_frames += 1
        if cfg.keyframe_every > 0 and self.world.step_no % cfg.keyframe_every == 0:
            self.f_key.write(data)
            self.f_key.flush()

    def checkpoint(self) -> Path:
        cfg = self.world.cfg
        self.flush_events()
        for f in (self.f_stats, self.f_key):
            f.flush()
            os.fsync(f.fileno())
        seg = self._cur_seg()
        extra = {
            "event_seg": seg,
            "births_bytes": events.sync(self.dir, "births", seg),
            "deaths_bytes": events.sync(self.dir, "deaths", seg),
            "stats_bytes": self.f_stats.tell(), "key_bytes": self.f_key.tell(),
            "wall": time.time(),
        }
        p = ckpt.save(self.world, self.ckpt_dir, extra)
        ckpt.thin(self.ckpt_dir, cfg.ckpt_keep_last, cfg.ckpt_keep_every)
        events.prune(self.dir, "births", seg, cfg.event_keep_segments)
        events.prune(self.dir, "deaths", seg, cfg.event_keep_segments)
        return p

    def disk_ok(self) -> bool:
        free = shutil.disk_usage(self.dir).free
        return free >= self.world.cfg.min_free_gb * 1e9

    # ---- 돌리기 ----
    def loop(self, max_steps: int | None = None, quiet: bool = False) -> None:
        w = self.world
        cfg = w.cfg
        end = None if max_steps is None else w.step_no + max_steps
        if cfg.passive and (end is None or end > self.passive_last_step + 1):
            end = self.passive_last_step + 1   # 대조 실행은 본 실행 기록이 끝나는 곳까지만
        t0 = time.time()
        s0 = w.step_no
        _sd_notify("READY=1")
        while not self.stop and (end is None or w.step_no < end):
            w.step()
            t = w.step_no
            if (t % cfg.frame_every == 0 or t <= self.dense_until
                    or (cfg.clip_every > 0 and t % cfg.clip_every < cfg.clip_len)):
                self.write_frame()
            if t % cfg.stats_every == 0:
                self.flush_events()
                w.check_sane()
                row = self.write_stats()
                _sd_notify("WATCHDOG=1")
                req = self.dir / "record.request"
                if req.exists():
                    try:
                        n = int(req.read_text().strip() or "600")
                    except ValueError:
                        n = 600
                    req.unlink()
                    self.dense_until = t + min(max(n, 1), 5000)
                if row["w_blocked"] > 0:
                    print(f"주의: step {t}에서 개체 수 상한(max_agents={cfg.max_agents})에 닿아 출생 "
                          f"{row['w_blocked']}번이 막혔다. 상한을 올려야 한다.", flush=True)
                if not self.disk_ok():
                    self.stop = True
                    self.stop_reason = f"디스크 여유가 {cfg.min_free_gb}GB 아래로 내려가 멈춘다"
                if not quiet:
                    print(f"step {t} pop {row['pop']} food {row['food']} gen {row.get('gen_mean', 0):.1f} "
                          f"hidden {row.get('hidden_mean', 0):.2f} reseeds {row['reseeds']} "
                          f"{(t - s0) / max(time.time() - t0, 1e-9):.0f} steps/s", flush=True)
            if t % cfg.ckpt_every == 0:
                self.checkpoint()
            if cfg.max_steps_per_sec > 0:
                ahead = (t - s0) / cfg.max_steps_per_sec - (time.time() - t0)
                if ahead > 0:
                    time.sleep(ahead)
        self.checkpoint()
        if self.stop_reason:
            (self.dir / "STOPPED.txt").write_text(
                f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} step {w.step_no}: {self.stop_reason}\n")
            print(self.stop_reason, flush=True)
        _sd_notify("STOPPING=1")

    def close(self) -> None:
        for f in (self.f_stats, self.f_key, self.f_seg, self.lock):
            if f:
                f.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="세계 시뮬레이션 실행기")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("new", "resume", "branch"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True, type=Path)
        p.add_argument("--steps", type=int, default=None, help="이만큼 돌리고 멈춘다(없으면 계속)")
        p.add_argument("--quiet", action="store_true")
        if name != "resume":
            p.add_argument("--set", action="append", default=[], metavar="key=value")
        if name == "new":
            p.add_argument("--config", type=Path, help="바탕으로 쓸 설정 json")
        if name == "branch":
            p.add_argument("--from", dest="src", required=True, type=Path)
            p.add_argument("--at", type=int, default=None, help="갈라낼 체크포인트의 스텝(없으면 가장 최근)")
            p.add_argument("--new-stream", action="store_true", help="난수 흐름을 새로 한다")
            p.add_argument("--passive", action="store_true", help="선택 없는 대조 실행으로 만든다")
    args = ap.parse_args(argv)

    if args.cmd == "new":
        base = Config.load(args.config) if args.config else Config()
        runner = Runner.new(args.run, base.with_overrides(args.set))
    elif args.cmd == "resume":
        runner = Runner.resume(args.run)
    else:
        runner = Runner.branch(args.src, args.run, args.at, args.set, args.new_stream, args.passive)

    cfg = runner.world.cfg
    if cfg.threads > 0:
        import numba
        numba.set_num_threads(min(cfg.threads, numba.config.NUMBA_NUM_THREADS))

    def on_signal(signum, frame):
        runner.stop = True
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    try:
        runner.loop(args.steps, args.quiet)
    finally:
        runner.close()
    print(f"멈춤: step {runner.world.step_no}, 실행 폴더 {args.run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
