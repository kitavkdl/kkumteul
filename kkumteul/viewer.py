"""관찰 화면 서버. 시뮬레이션과 따로 돌고, 실행 폴더의 파일만 읽는다(시뮬레이션을 멈추거나 느리게 하지 않는다).

    python -m kkumteul.viewer --runs runs [--port 8765] [--host 127.0.0.1]

유일하게 쓰는 것은 "촘촘히 녹화" 요청 파일(record.request) 하나다.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HEAD = struct.Struct("<4sIq")
SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
STAT_KEYS = ("step", "pop", "food", "gen_mean", "gen_max", "hidden_mean", "energy_mean",
             "speed_mean", "sps", "w_births", "w_deaths", "w_reseeds", "w_life_mean",
             "w_eaten_per_life", "reseeds", "wall")


class FrameIndex:
    """실행 하나의 프레임 위치표. 파일이 자란 만큼만 다시 읽는다."""

    def __init__(self, run_dir: Path):
        self.dir = run_dir
        self.files: dict[Path, tuple[int, list]] = {}   # 경로 -> (읽은 바이트, [(step, offset, length)])
        self.lock = threading.Lock()

    def refresh(self) -> list[tuple[int, Path, int, int]]:
        with self.lock:
            paths = sorted((self.dir / "frames").glob("seg_*.bin"))
            key = self.dir / "keyframes.bin"
            if key.exists():
                paths.append(key)
            for gone in [p for p in self.files if p not in paths]:
                del self.files[gone]
            for p in paths:
                try:
                    size = p.stat().st_size
                except FileNotFoundError:
                    continue
                done, entries = self.files.get(p, (0, []))
                if size < done:
                    done, entries = 0, []
                if size > done:
                    with open(p, "rb") as f:
                        while done + HEAD.size <= size:
                            f.seek(done)
                            magic, total, step = HEAD.unpack(f.read(HEAD.size))
                            if magic not in (b"WSF1", b"WSF2") or total < HEAD.size or done + total > size:
                                break
                            entries.append((step, done, total))
                            done += total
                    self.files[p] = (done, entries)
            merged: dict[int, tuple[int, Path, int, int]] = {}
            for p, (_, entries) in self.files.items():
                for step, off, length in entries:
                    merged[step] = (step, p, off, length)
            return [merged[s] for s in sorted(merged)]


class StatsCache:
    """stats.jsonl을 자란 만큼만 읽어 둔다. 줄이 많아지면 둘 중 하나씩 버려 메모리를 묶어 둔다."""
    LIMIT = 4000

    def __init__(self, path: Path):
        self.path = path
        self.offset = 0
        self.rows: list[dict] = []
        self.stride = 1       # 지금은 원래 줄 stride개마다 하나를 남긴다
        self.seen = 0
        self.last: dict | None = None
        self.lock = threading.Lock()

    def refresh(self) -> None:
        with self.lock:
            try:
                size = self.path.stat().st_size
            except FileNotFoundError:
                return
            if size < self.offset:   # 이어 돌리면서 잘린 경우: 처음부터 다시
                self.offset, self.rows, self.stride, self.seen, self.last = 0, [], 1, 0, None
            if size == self.offset:
                return
            with open(self.path, "rb") as f:
                f.seek(self.offset)
                chunk = f.read(size - self.offset)
            end = chunk.rfind(b"\n")
            if end < 0:
                return
            self.offset += end + 1
            for line in chunk[:end].splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                row = {key: r.get(key) for key in STAT_KEYS}
                self.last = row
                if self.seen % self.stride == 0:
                    self.rows.append(row)
                self.seen += 1
                if len(self.rows) > self.LIMIT:
                    self.rows = self.rows[::2]
                    self.stride *= 2

    def sample(self, max_rows: int) -> list[dict]:
        self.refresh()
        with self.lock:
            rows = self.rows
            if len(rows) > max_rows:
                step = len(rows) / max_rows
                rows = [rows[int(i * step)] for i in range(max_rows)]
            else:
                rows = list(rows)
            if self.last is not None and (not rows or rows[-1] is not self.last):
                rows.append(self.last)
            return rows


class App:
    def __init__(self, runs_root: Path):
        self.root = runs_root.resolve()
        self.indexes: dict[str, FrameIndex] = {}
        self.stat_caches: dict[str, StatsCache] = {}

    def run_dir(self, name: str) -> Path | None:
        if not name or not SAFE_NAME.match(name) or name in (".", ".."):
            return None
        p = self.root / name
        return p if (p / "run.json").exists() else None

    def runs(self) -> list[dict]:
        out = []
        if self.root.is_dir():
            for p in sorted(self.root.iterdir()):
                if (p / "run.json").exists() and SAFE_NAME.match(p.name):
                    meta = json.loads((p / "run.json").read_text())
                    out.append({"name": p.name, "kind": meta.get("kind"), "parent": meta.get("parent_run"),
                                "last": self._last_stat(p)})
        return out

    @staticmethod
    def _last_stat(p: Path):
        f = p / "stats.jsonl"
        if not f.exists() or f.stat().st_size == 0:
            return None
        with open(f, "rb") as fh:
            fh.seek(max(0, f.stat().st_size - 4096))
            lines = fh.read().splitlines()
        for line in reversed(lines):
            try:
                row = json.loads(line)
                return {key: row.get(key) for key in ("step", "pop", "gen_mean", "sps")}
            except ValueError:
                continue
        return None

    def index(self, name: str) -> FrameIndex:
        if name not in self.indexes:
            self.indexes[name] = FrameIndex(self.root / name)
        return self.indexes[name]

    def stats(self, name: str, p: Path, max_rows: int) -> list[dict]:
        if name not in self.stat_caches:
            self.stat_caches[name] = StatsCache(p / "stats.jsonl")
        return self.stat_caches[name].sample(max_rows)


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _run(self, q):
            name = (q.get("run") or [""])[0]
            p = app.run_dir(name)
            if p is None:
                self._json({"error": "없는 실행"}, 404)
            return name, p

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/":
                html = Path(__file__).with_name("viewer.html").read_bytes()
                return self._send(200, html, "text/html; charset=utf-8")
            if u.path == "/api/runs":
                return self._json(app.runs())
            if u.path == "/api/index":
                name, p = self._run(q)
                if p is None:
                    return
                entries = app.index(name).refresh()
                try:
                    after = int((q.get("after") or ["-1"])[0])
                except ValueError:
                    after = -1
                steps = [e[0] for e in entries]
                out = {"steps": [s for s in steps if s > after], "count": len(steps),
                       "first": steps[0] if steps else None, "last": steps[-1] if steps else None}
                if after < 0:   # 처음 한 번만 설정을 같이 보낸다
                    out["config"] = json.loads((p / "config.json").read_text())
                    out["meta"] = json.loads((p / "run.json").read_text())
                return self._json(out)
            if u.path == "/api/frame":
                name, p = self._run(q)
                if p is None:
                    return
                try:
                    want = int((q.get("step") or ["-1"])[0])
                except ValueError:
                    return self._json({"error": "step"}, 400)
                entries = app.index(name).refresh()
                if not entries:
                    return self._json({"error": "프레임 없음"}, 404)
                hit = entries[-1]
                if want >= 0:
                    for e in entries:
                        if e[0] <= want:
                            hit = e
                        else:
                            break
                    if hit[0] > want:
                        hit = entries[0]
                try:
                    with open(hit[1], "rb") as f:
                        f.seek(hit[2])
                        data = f.read(hit[3])
                except FileNotFoundError:
                    return self._json({"error": "지워진 프레임"}, 404)
                return self._send(200, data, "application/octet-stream")
            if u.path == "/api/stats":
                name, p = self._run(q)
                if p is None:
                    return
                try:
                    n = min(max(int((q.get("max") or ["400"])[0]), 10), 2000)
                except ValueError:
                    n = 400
                return self._json(app.stats(name, p, n))
            self._json({"error": "없는 주소"}, 404)

        def do_POST(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == "/api/record":
                name, p = self._run(q)
                if p is None:
                    return
                try:
                    n = min(max(int((q.get("n") or ["600"])[0]), 1), 5000)
                except ValueError:
                    n = 600
                (p / "record.request").write_text(str(n))
                return self._json({"ok": True, "n": n})
            self._json({"error": "없는 주소"}, 404)

    return Handler


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="관찰 화면 서버")
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1",
                    help="다른 기기(폰)에서 보려면 0.0.0.0. 그러면 같은 망의 누구나 화면을 보고 "
                         "'촘촘히 녹화'를 요청할 수 있다(다른 것은 못 바꾼다)")
    args = ap.parse_args(argv)
    app = App(args.runs)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"관찰 화면: http://{args.host}:{args.port}/  (실행 폴더: {app.root})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
