"""생물 묶음(pack): 유전체만 담은 작은 파일. 서로 주고받아 겨루게 한다.

체크포인트는 세계 전체(먹이, 장부, 모든 개체의 몸 상태)라 크고, 그 실행의 기록과 묶여 있다.
묶음에는 뽑은 개체들의 유전체(`measure.GENOME`의 배열과 세대 수)와, 겨룰 세계를 다시 만드는 데 필요한
설정, 붙인 이름, 뽑아 온 스텝만 들어 있다.

    python3 -m kkumteul.pack export --from runs/main --out mine.pack.npz [--n 400] [--seed 1] [--label "..."]
    python3 -m kkumteul.pack info mine.pack.npz
    python3 -m kkumteul.measure assay --a mine.pack.npz --b yours.pack.npz

- `--from`에는 실행 폴더(가장 최근 체크포인트를 쓴다)나 체크포인트 npz를 준다.
- 각본을 쓰는 실행에서 뽑으면 설정은 각본이 바꾸기 전의 바탕 설정을 담는다(겨루는 세계는 평소 조건이어야 하므로).
- 묶음에는 만든 사람의 경로나 기계 이름이 들어가지 않는다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np

from . import SOURCE_HASH, STATE_VERSION, __version__
from . import checkpoint as ckpt
from .config import Config
from . import measure
from .measure import GENOME, extract_genomes

MARK = "kkumteul_pack"    # 이 이름의 배열이 있으면 묶음이다(값은 형식 번호)
PACK_VERSION = 1


def find_checkpoint(src: Path) -> Path:
    """실행 폴더면 가장 최근 체크포인트, npz면 그 파일."""
    src = Path(src)
    if src.suffix == ".npz":
        return src
    p = ckpt.latest(src / "checkpoints")
    if p is None:
        raise FileNotFoundError(f"체크포인트를 못 찾았다: {src}")
    return p


def base_config(ckpt_path: Path, saved: Config) -> Config:
    """체크포인트가 각본을 쓰는 실행의 것이면 그 실행의 바탕 설정을, 아니면 저장된 설정을 돌려준다."""
    run_dir = Path(ckpt_path).parent.parent
    if (run_dir / "scenario.json").exists() and (run_dir / "config.json").exists():
        return Config.load(run_dir / "config.json")
    return saved


def export(src: Path, out: Path, n: int = 400, seed: int = 1, label: str = "") -> dict:
    """체크포인트에서 살아 있는 개체 n마리(모자라면 전부)의 유전체를 뽑아 묶음으로 쓴다."""
    path = find_checkpoint(src)
    world, meta = ckpt.load(path)
    g = extract_genomes(world)
    alive = len(g["n_hidden"])
    if alive == 0:
        raise ValueError(f"살아 있는 개체가 없는 체크포인트다: {path.name}")
    if n < 1:
        raise ValueError("--n은 1 이상이어야 한다")
    if alive > n:
        pick = np.sort(np.random.default_rng(seed).choice(alive, size=n, replace=False))
        g = {key: v[pick] for key, v in g.items()}
    info = {
        "label": label or path.stem,
        "source_step": int(meta["step"]),
        "source_npz_sha256": meta.get("npz_sha256"),
        "source_alive": alive,
        "n": int(len(g["n_hidden"])),
        "sample_seed": int(seed),
        "world_seed": int(world.cfg.seed),
        "state_version": STATE_VERSION,
        "code_version": __version__,
        "source_hash": SOURCE_HASH,
    }
    cfg = base_config(path, world.cfg)
    arrays = dict(g)
    arrays[MARK] = np.array(PACK_VERSION, dtype=np.int64)
    arrays["config_json"] = np.array(json.dumps(cfg.to_dict(), sort_keys=True))
    arrays["info_json"] = np.array(json.dumps(info, ensure_ascii=False, sort_keys=True))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out)
    return info


def is_pack(path) -> bool:
    try:
        with np.load(Path(path)) as z:
            return MARK in z.files
    except (OSError, ValueError, zipfile.BadZipFile):
        return False


def read(path: Path) -> tuple[dict, Config, dict]:
    """(유전체, 설정, 정보)를 읽는다."""
    with np.load(Path(path)) as z:
        if MARK not in z.files:
            raise ValueError(f"생물 묶음이 아니다: {Path(path).name}")
        version = int(z[MARK])
        if version != PACK_VERSION:
            raise ValueError(f"묶음 형식이 다르다: {version} != {PACK_VERSION}")
        genomes = {name: z[name] for name in GENOME + ("gen",)}
        cfg = Config.from_dict(json.loads(str(z["config_json"])))
        info = json.loads(str(z["info_json"]))
    n = len(genomes["n_hidden"])
    if n == 0 or any(len(v) != n for v in genomes.values()):
        raise ValueError(f"묶음이 깨졌다(개체 수가 배열마다 다르거나 0이다): {Path(path).name}")
    return genomes, cfg, info


def load_side(path: Path) -> dict:
    """겨루기(`measure.assay_sides`)에 넣을 한쪽. 체크포인트에서 읽은 것과 같은 꼴이다."""
    genomes, cfg, info = read(path)
    return {"path": str(path), "step": info["source_step"], "genomes": genomes, "cfg": cfg,
            "gen_mean": float(genomes["gen"].mean()), "label": info["label"]}


def describe(path: Path) -> dict:
    genomes, cfg, info = read(path)
    out = dict(info)
    out.update({
        "file_bytes": Path(path).stat().st_size,
        "gen_mean": round(float(genomes["gen"].mean()), 2),
        "hidden_mean": round(float(genomes["n_hidden"].mean()), 2),
        "shapes": {name: list(genomes[name].shape[1:]) for name in GENOME},
        "world": [cfg.world_w, cfg.world_h],
        "food_inflow": cfg.food_inflow,
        "max_agents": cfg.max_agents,
    })
    return out


def migrate(src: Path, pack_path: Path, run_dir: Path, n: int = 50, seed: int = 1, tag: int = 1) -> dict:
    """이주: 실행 src의 가장 최근 체크포인트에서 새 실행을 가르고, 묶음의 생물 n마리를 무작위 자리에 넣는다.
    넣은 생물과 그 후손은 표지(tag)를 물려받아, 나중에 누가 남았는지 셀 수 있다."""
    from . import kernels as k
    from .run import Runner
    from .world import BIRTH_DTYPE
    side = load_side(pack_path)
    runner = Runner.branch(Path(src), Path(run_dir), None, [], False, False)
    try:
        world = runner.world
        measure.check_same_shape(side, {"genomes": measure.extract_genomes(world), "path": str(src),
                                        "cfg": world.cfg})
        free = int((~world.s.alive).sum())
        if n > free:
            raise ValueError(f"빈 자리가 모자란다(넣을 수 {n}, 빈 자리 {free})")
        rng = np.random.default_rng(seed)
        g = side["genomes"]
        rows = rng.choice(len(g["n_hidden"]), size=n, replace=len(g["n_hidden"]) < n)
        cfg = world.cfg
        rec = np.zeros(n, dtype=BIRTH_DTYPE)
        for q, row in enumerate(rows):
            aid = int(world.s.counters[k.C_NEXT_ID])
            measure._inject(world, g, int(row), tag, float(rng.random() * cfg.world_w),
                            float(rng.random() * cfg.world_h), float(rng.random() * 2 * np.pi), 0)
            rec[q] = (aid, -1, world.step_no, 0, int(g["n_hidden"][row]), k.BIRTH_INJECT)
        world.births.append(rec)
        info = {"pack": Path(pack_path).name, "n": int(n), "seed": int(seed), "tag": int(tag),
                "pack_label": side.get("label"), "at_step": world.step_no}
        runner.meta["migrants"] = info
        ckpt.atomic_write_bytes(runner.dir / "run.json", json.dumps(runner.meta, indent=1).encode())
        runner.checkpoint()      # 같은 스텝의 체크포인트를 이주가 끝난 상태로 다시 쓴다
        return info
    finally:
        runner.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="생물 묶음: 유전체만 담은 파일을 만들고 본다")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="체크포인트에서 묶음을 만든다")
    ex.add_argument("--from", dest="src", required=True, type=Path, help="체크포인트 npz나 실행 폴더(가장 최근 체크포인트)")
    ex.add_argument("--out", required=True, type=Path, help="쓸 묶음 파일(.npz)")
    ex.add_argument("--n", type=int, default=400, help="뽑을 개체 수(살아 있는 수가 모자라면 전부)")
    ex.add_argument("--seed", type=int, default=1, help="뽑기 난수 씨앗")
    ex.add_argument("--label", default="", help="묶음에 붙일 이름(없으면 체크포인트 파일 이름)")
    mg = sub.add_parser("migrate", help="이주: 실행에서 갈라낸 새 실행에 묶음의 생물을 넣는다")
    mg.add_argument("--from", dest="src", required=True, type=Path, help="받는 쪽 실행 폴더")
    mg.add_argument("--pack", required=True, type=Path, help="넣을 생물 묶음")
    mg.add_argument("--run", required=True, type=Path, help="새로 만들 실행 폴더(여기서 이어 돌린다)")
    mg.add_argument("--n", type=int, default=50, help="넣을 개체 수")
    mg.add_argument("--seed", type=int, default=1)
    inf = sub.add_parser("info", help="묶음의 내용을 요약해 보여 준다")
    inf.add_argument("pack", type=Path)
    inf.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "export":
        info = export(args.src, args.out, args.n, args.seed, args.label)
        print(f"묶음을 썼다: {args.out} ({args.out.stat().st_size / 1e6:.1f}MB)")
        print(f"이름 {info['label']}, {info['source_step']}스텝에서 살아 있던 {info['source_alive']}마리 가운데 {info['n']}마리")
        return 0
    if args.cmd == "migrate":
        info = migrate(args.src, args.pack, args.run, args.n, args.seed)
        print(f"이주시켰다: {info['pack']}의 {info['n']}마리를 {info['at_step']}스텝의 세계에 넣었다(표지 {info['tag']}).")
        print(f"이어 돌리기: python3 -m kkumteul.run resume --run {args.run}")
        return 0
    d = describe(args.pack)
    if args.json:
        print(json.dumps(d, ensure_ascii=False, indent=1))
        return 0
    print(f"이름: {d['label']}")
    print(f"개체 수: {d['n']} (뽑아 온 스텝 {d['source_step']}, 그때 살아 있던 수 {d['source_alive']}, 뽑기 씨앗 {d['sample_seed']})")
    print(f"평균 세대(계보 깊이): {d['gen_mean']}, 평균 활성 뉴런: {d['hidden_mean']}")
    print(f"세계: {d['world'][0]:g} x {d['world'][1]:g}, 먹이 유입 {d['food_inflow']:g}, 개체 수 상한 {d['max_agents']}, 세계 씨앗 {d['world_seed']}")
    print("유전체 모양(개체 하나): " + ", ".join(f"{name} {tuple(shape)}" for name, shape in d["shapes"].items()))
    print(f"원본 체크포인트 sha256: {d['source_npz_sha256']}")
    print(f"코드 {d['code_version']}, 계산 코드 해시 {d['source_hash'][:12]}, 파일 크기 {d['file_bytes'] / 1e6:.1f}MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
