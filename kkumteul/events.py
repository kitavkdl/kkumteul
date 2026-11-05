"""출생·사망 기록 파일. 스텝 구간마다 파일을 나눠 덧붙인다(`events/births_000012.bin`).

한 파일로 끝없이 쌓으면 며칠 뒤에는 분석이 램에 안 들어간다. 나눠 두면 필요한 구간만 읽고,
오래된 구간은 지우거나 옮길 수 있다.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

from .world import BIRTH_DTYPE, DEATH_DTYPE

DTYPES = {"births": BIRTH_DTYPE, "deaths": DEATH_DTYPE}
_NAME = re.compile(r"^(births|deaths)_(\d+)\.bin$")


def seg_path(run_dir: Path, kind: str, seg: int) -> Path:
    return Path(run_dir) / "events" / f"{kind}_{seg:06d}.bin"


def segments(run_dir: Path, kind: str) -> list[tuple[int, Path]]:
    d = Path(run_dir) / "events"
    out = []
    if d.is_dir():
        for p in d.iterdir():
            m = _NAME.match(p.name)
            if m and m.group(1) == kind:
                out.append((int(m.group(2)), p))
    return sorted(out)


def append(run_dir: Path, kind: str, rec: np.ndarray, seg_steps: int) -> None:
    if not len(rec):
        return
    (Path(run_dir) / "events").mkdir(exist_ok=True)
    seg = rec["step"] // seg_steps
    for sidx in np.unique(seg):
        with open(seg_path(run_dir, kind, int(sidx)), "ab") as f:
            f.write(rec[seg == sidx].tobytes())


def sync(run_dir: Path, kind: str, seg: int) -> int:
    """구간 파일을 디스크에 내려쓰고 크기를 돌려준다(없으면 0)."""
    p = seg_path(run_dir, kind, seg)
    if not p.exists():
        return 0
    with open(p, "rb") as f:
        os.fsync(f.fileno())
    return p.stat().st_size


def truncate_after(run_dir: Path, kind: str, seg: int, size: int) -> None:
    """체크포인트 뒤에 쓴 기록을 잘라 낸다: seg보다 뒤 파일은 지우고 seg 파일은 size로 줄인다."""
    for sidx, p in segments(run_dir, kind):
        if sidx > seg:
            p.unlink()
        elif sidx == seg and p.stat().st_size > size:
            with open(p, "r+b") as f:
                f.truncate(size)


def prune(run_dir: Path, kind: str, cur_seg: int, keep: int) -> None:
    if keep <= 0:
        return
    for sidx, p in segments(run_dir, kind):
        if sidx <= cur_seg - keep:
            p.unlink()


def load(run_dir: Path, kind: str, seg_steps: int, min_step: int = 0, max_bytes: int | None = None) -> np.ndarray:
    """min_step 이후의 기록을 읽는다. 끝이 잘린 레코드는 버린다."""
    dt = DTYPES[kind]
    parts = []
    total = 0
    for sidx, p in segments(run_dir, kind):
        if (sidx + 1) * seg_steps <= min_step:
            continue
        a = np.fromfile(p, dtype=dt, count=p.stat().st_size // dt.itemsize)
        total += a.nbytes
        if max_bytes is not None and total > max_bytes:
            raise MemoryError(f"기록이 너무 크다({total >> 20} MB 초과). --last-steps로 구간을 줄인다")
        parts.append(a)
    if not parts:
        return np.empty(0, dtype=dt)
    out = np.concatenate(parts)
    return out[out["step"] >= min_step] if min_step > 0 else out
