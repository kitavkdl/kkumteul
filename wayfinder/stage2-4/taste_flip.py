"""맛 칸 가중치 손대기: 체크포인트의 맛 블록 입력 가중치(Win의 맛 칸 열)만 바꿔 새 실행 폴더에 저장한다.

대조 갈래 b(맛 기억을 읽도록 진화한 적 없음)에 기억을 켜면 독을 따라갔다(판정 가 +0.08, 문턱 0 ± 0.02).
그것이 "떠돌이 가중치가 우연히 맞는 쪽으로 기운 것"이면 부호를 뒤집은 생물은 반대로(−) 가야 하고,
"기억 입력이 들어오기만 하면 생기는 효과"면 부호와 상관없이 +로 남는다. 같은 판정 도구를 새 폴더에 쓴다.

    flip: 맛 칸 열의 부호를 뒤집는다.   zero: 맛 칸 열을 0으로(기억 입력이 생각에 닿지 않는다 — 막기와 같은 대조).
새끼는 부모의 Win을 물려받으므로 손댄 것은 다시 돌리는 동안 이어진다(돌연변이로 조금씩 흐트러질 뿐).
원래 실행 폴더의 run.json·config.json·stats.jsonl은 복사하고 run.json에 손댄 것을 적는다. 원래 폴더는 읽기만 한다.

    PYTHONPATH=<코드> python wayfinder/stage2-4/taste_flip.py <체크포인트.npz> <새 실행 폴더> --mode flip|zero
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from kkumteul import checkpoint as ckpt
from kkumteul.config import N_TASTE_IN


def edit(src: Path, out_dir: Path, mode: str) -> Path:
    world, meta = ckpt.load(src)
    cfg = world.cfg
    t0 = cfg.taste_in0
    if cfg.n_in < t0 + N_TASTE_IN:
        raise ValueError("맛 블록 크기가 아닌 세계다(preset=taste_size)")
    before = world.s.Win[:, t0:t0 + N_TASTE_IN, :].copy()
    if mode == "flip":
        world.s.Win[:, t0:t0 + N_TASTE_IN, :] *= -1.0
    elif mode == "zero":
        world.s.Win[:, t0:t0 + N_TASTE_IN, :] = 0.0
    else:
        raise ValueError(mode)
    alive = world.s.alive
    print(f"맛 칸 {t0}~{t0 + N_TASTE_IN - 1}, 산 개체 {int(alive.sum())}, "
          f"평균 |가중치| {float(np.abs(before[alive]).mean()):.3f} → "
          f"{float(np.abs(world.s.Win[alive, t0:t0 + N_TASTE_IN, :]).mean()):.3f}")
    out_dir.mkdir(parents=True, exist_ok=False)
    src_run = src.parent.parent
    for name in ("config.json", "stats.jsonl"):
        if (src_run / name).exists():
            shutil.copy2(src_run / name, out_dir / name)
    run = json.loads((src_run / "run.json").read_text()) if (src_run / "run.json").exists() else {}
    run.setdefault("genome_edits", []).append(f"Win[:, {t0}:{t0 + N_TASTE_IN}, :] {mode} (taste_flip.py, {src})")
    (out_dir / "run.json").write_text(json.dumps(run, indent=1, ensure_ascii=False))
    return ckpt.save(world, out_dir / "checkpoints", extra={"edited_from": str(src), "edit": mode})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--mode", choices=("flip", "zero"), required=True)
    a = p.parse_args()
    print(edit(a.ckpt, a.out, a.mode))


if __name__ == "__main__":
    main()
