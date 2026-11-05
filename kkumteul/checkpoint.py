"""체크포인트. 압축 없는 npz + json을 임시 이름에 쓰고 fsync 뒤 rename한다(쓰다 꺼져도 옛 파일은 온전)."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

from . import SOURCE_HASH, STATE_VERSION, __version__
from .config import N_STAGE_IN, N_STAGE_OUT, Config
from .world import World

_NAME = re.compile(r"^ckpt_(\d+)\.npz$")


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def file_sha256(path: Path) -> str:
    m = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            m.update(chunk)
    return m.hexdigest()


def save(world: World, ckpt_dir: Path, extra: dict | None = None) -> Path:
    """세계를 저장하고 npz 경로를 돌려준다. extra는 사건 파일 위치 같은 덧붙임 정보."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    step = world.step_no
    path = ckpt_dir / f"ckpt_{step:012d}.npz"
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **world.state_arrays())
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    meta = {
        "state_version": STATE_VERSION,
        "code_version": __version__,
        "source_hash": SOURCE_HASH,
        "step": step,
        "config": world.cfg.to_dict(),
        "state_hash": world.state_hash(),
        "npz_sha256": file_sha256(path),
        "summary": world.summary(),
    }
    meta.update(extra or {})
    atomic_write_bytes(path.with_suffix(".json"), json.dumps(meta, indent=1).encode())
    return path


def load(path: Path, cfg: Config | None = None, verify: bool = True) -> tuple[World, dict]:
    """체크포인트에서 세계를 되살린다. cfg를 주면 그 설정으로(분기), 없으면 저장된 설정으로."""
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    if verify and meta.get("npz_sha256") and file_sha256(path) != meta["npz_sha256"]:
        raise ValueError(f"체크포인트 파일이 기록과 다르다(깨졌거나 바뀌었다): {path}")
    if meta["state_version"] != STATE_VERSION:
        raise ValueError(f"상태 형식이 다르다: {meta['state_version']} != {STATE_VERSION}")
    cd = dict(meta["config"])
    # 옛 체크포인트는 입출력 자리가 적다. 지금 크기로 올리고 배열은 읽을 때 0으로 채운다
    cd["n_spare_in"] = max(cd.get("n_spare_in", 0), N_STAGE_IN)
    cd["n_spare_out"] = max(cd.get("n_spare_out", 0), N_STAGE_OUT)
    upgraded = cd != meta["config"]
    saved_cfg = Config.from_dict(cd)
    world = World(saved_cfg, populate=False)
    with np.load(path) as z:
        exact = world.load_arrays({k: z[k] for k in z.files})
    # 옛 형식(필드가 없거나 크기가 작던 것)을 채워 읽은 경우는 저장 때의 상태 해시와 견줄 수 없다.
    # 그때도 파일 자체는 위의 sha256이 지킨다.
    if verify and exact and not upgraded and world.state_hash() != meta["state_hash"]:
        raise ValueError(f"체크포인트가 깨졌다: {path}")
    if cfg is not None:
        world.set_config(cfg)
    if meta.get("source_hash") not in (None, SOURCE_HASH):
        print(f"주의: {path.name}은 지금과 다른 계산 코드로 저장됐다. 이어 돌린 결과가 "
              "끊지 않고 돌린 결과와 달라질 수 있다.", flush=True)
    return world, meta


def list_checkpoints(ckpt_dir: Path) -> list[tuple[int, Path]]:
    out = []
    if ckpt_dir.is_dir():
        for p in ckpt_dir.iterdir():
            m = _NAME.match(p.name)
            if m and p.with_suffix(".json").exists():
                out.append((int(m.group(1)), p))
    return sorted(out)


def latest(ckpt_dir: Path) -> Path | None:
    c = list_checkpoints(ckpt_dir)
    return c[-1][1] if c else None


def thin(ckpt_dir: Path, keep_last: int, keep_every: int) -> list[Path]:
    """최근 keep_last개와 keep_every 배수 스텝만 남기고 지운다. 지운 경로를 돌려준다."""
    c = list_checkpoints(ckpt_dir)
    removed = []
    for step, p in c[:-keep_last] if keep_last > 0 else c:
        if keep_every > 0 and step % keep_every == 0:
            continue
        p.unlink()
        p.with_suffix(".json").unlink(missing_ok=True)
        removed.append(p)
    return removed
