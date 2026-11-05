"""핵심 성질 시험: 재현성, 저장·복구, 에너지 보존, 분기, 대조 실행.

    python3 -m pytest -q
"""
import json
import sys
from pathlib import Path

import numba
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kkumteul import analyze, measure  # noqa: E402
from kkumteul import checkpoint as ckpt  # noqa: E402
from kkumteul import events  # noqa: E402
from kkumteul import kernels as k  # noqa: E402
from kkumteul.config import Config  # noqa: E402
from kkumteul.run import FRAME_HEADER, FRAME_MAGIC, Runner, encode_frame  # noqa: E402
from kkumteul.world import World  # noqa: E402

SEG = 700   # 시험에서는 기록 파일이 여러 개로 나뉘게 작게 잡는다


def ev(run_dir, kind):
    return events.load(run_dir, kind, SEG)


def small(**kw) -> Config:
    base = dict(world_w=256.0, world_h=256.0, max_agents=600, init_agents=150, food_cap=500,
                food_inflow=2.2, n_patches=4, stats_every=200, frame_every=50, clip_every=300, clip_len=20,
                ckpt_every=400, keyframe_every=200, min_agents=20, event_segment_steps=SEG)
    base.update(kw)
    return Config.from_dict({**Config().to_dict(), **base})


def test_same_seed_same_world():
    a, b = World(small()), World(small())
    a.run(600)
    b.run(600)
    assert a.state_hash() == b.state_hash()
    c = World(small(seed=2))
    c.run(600)
    assert c.state_hash() != a.state_hash()


def test_thread_count_does_not_matter():
    n0 = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        a = World(small())
        a.run(400)
        numba.set_num_threads(max(2, min(4, n0)))
        b = World(small())
        b.run(400)
    finally:
        numba.set_num_threads(n0)
    assert a.state_hash() == b.state_hash()


def test_world_lives_and_energy_is_conserved():
    w = World(small())
    w.run(3000)
    w.check_sane()
    s = w.summary()
    assert s["births"] > 0, "아무도 번식하지 못했다"
    assert s["pop"] > 10
    bal = w.energy_balance()
    scale = bal["spawn"] + bal["inject"]
    assert abs(bal["residual"]) / scale < 1e-9, bal
    # 먹기와 주입 말고는 에너지가 생기지 않는다
    assert bal["agents"] + bal["food"] <= scale + 1e-6


def test_parallel_kernels_actually_write():
    """prange 안에서 상태 묶음을 거치면 쓰기가 사라지는 numba 함정이 다시 생기지 않았는지."""
    w = World(small())
    w.run(50)
    a = w.s.alive
    assert np.abs(w.s.act[a]).sum() > 0
    assert np.abs(w.s.h[a]).sum() > 0
    assert np.hypot(w.s.vx[a], w.s.vy[a]).mean() > 0.05
    base = 2 * w.cfg.vision_sectors
    assert (w.s.obs[a][:, base + 7] == 1.0).all()


def test_checkpoint_roundtrip(tmp_path):
    a = World(small())
    a.run(300)
    p = ckpt.save(a, tmp_path)
    a.run(300)
    b, meta = ckpt.load(p)
    assert meta["step"] == 300
    b.run(300)
    assert a.state_hash() == b.state_hash()


def test_checkpoint_detects_corruption(tmp_path):
    a = World(small())
    a.run(50)
    p = ckpt.save(a, tmp_path)
    meta = json.loads(p.with_suffix(".json").read_text())
    meta["state_hash"] = "0" * 64
    p.with_suffix(".json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        ckpt.load(p)


def test_runner_resume_matches_uninterrupted(tmp_path):
    cfg = small()
    one = Runner.new(tmp_path / "one", cfg)
    one.loop(1000, quiet=True)
    one.close()

    two = Runner.new(tmp_path / "two", cfg)
    two.loop(500, quiet=True)
    # 체크포인트 뒤에 기록이 더 쌓인 채 꺼진 상황을 흉내 낸다
    two.world.run(130)
    two.flush_events()
    two.write_stats()
    two.close()
    two = Runner.resume(tmp_path / "two")
    assert two.world.step_no == 500
    two.loop(500, quiet=True)
    two.close()

    assert one.world.state_hash() == two.world.state_hash()
    for kind in ("births", "deaths"):
        assert len(events.segments(tmp_path / "two", kind)) >= 2, "기록이 여러 파일로 나뉘어야 한다"
        assert ev(tmp_path / "one", kind).tobytes() == ev(tmp_path / "two", kind).tobytes()
    b = ev(tmp_path / "two", "births")
    assert (np.diff(b["id"]) > 0).all(), "출생 기록의 id는 늘어나기만 해야 한다"


def test_branch_same_stream_replays_and_new_stream_diverges(tmp_path):
    cfg = small()
    main = Runner.new(tmp_path / "main", cfg)
    main.loop(400, quiet=True)
    main.loop(400, quiet=True)
    main.close()

    same = Runner.branch(tmp_path / "main", tmp_path / "same", 400, [], False, False)
    same.loop(400, quiet=True)
    same.close()
    assert same.world.state_hash() == main.world.state_hash()

    other = Runner.branch(tmp_path / "main", tmp_path / "other", 400, [], True, False)
    other.loop(400, quiet=True)
    other.close()
    assert other.world.state_hash() != main.world.state_hash()
    meta = json.loads((tmp_path / "other" / "run.json").read_text())
    assert meta["rng"] == "new_stream" and meta["parent_step"] == 400

    what_if = Runner.branch(tmp_path / "main", tmp_path / "whatif", 400, ["brain_cost=0.02"], False, False)
    assert json.loads((tmp_path / "whatif" / "run.json").read_text())["config_diff"] == {"brain_cost": [0.006, 0.02]}
    what_if.close()


def test_passive_control_follows_recorded_events(tmp_path):
    cfg = small()
    main = Runner.new(tmp_path / "main", cfg)
    main.loop(1200, quiet=True)
    main.close()
    b = ev(tmp_path / "main", "births")
    d = ev(tmp_path / "main", "deaths")
    nat = int(((b["kind"] != k.BIRTH_INITIAL) & (b["step"] >= 400)).sum())
    dead = int((d["step"] >= 400).sum())
    assert nat > 0 and dead > 0

    pas = Runner.branch(tmp_path / "main", tmp_path / "passive", 400, [], True, True)
    pas.loop(None, quiet=True)
    pas.close()
    pb = ev(tmp_path / "passive", "births")
    pd_ = ev(tmp_path / "passive", "deaths")
    assert set(np.unique(pb["kind"])) <= {k.BIRTH_PASSIVE}
    assert set(np.unique(pd_["cause"])) <= {k.DEATH_PASSIVE}
    # 같은 수의 출생·사망이 일어난다(개체가 바닥나는 드문 경우만 빼고)
    assert abs(len(pb) - nat) <= 0.02 * nat + 2
    assert abs(len(pd_) - dead) <= 0.02 * dead + 2
    bal = pas.world.energy_balance()
    assert abs(bal["residual"]) / (bal["spawn"] + bal["inject"]) < 1e-9


def test_frame_format():
    w = World(small())
    w.run(20)
    data = encode_frame(w)
    magic, total, step, n, m, ww, wh = FRAME_HEADER.unpack_from(data)
    assert magic == FRAME_MAGIC and total == len(data) and step == 20
    assert n == int(w.s.alive.sum()) and m == int(w.s.f_alive.sum())
    assert total == FRAME_HEADER.size + n * 40 + m * 4


def test_mutation_off_copies_genome_exactly():
    w = World(small(mutation_enabled=False))
    w.run(1500)
    s = w.s
    a = np.flatnonzero(s.alive & (s.gen > 0))
    assert len(a) > 0
    assert (s.n_hidden[s.alive] == w.cfg.h_init).all()
    assert np.allclose(s.sigma[s.alive], w.cfg.sigma_init)


def test_config_rejects_unknown_and_bad_values():
    with pytest.raises(ValueError):
        Config().with_overrides(["no_such_key=1"])
    with pytest.raises(ValueError):
        Config().with_overrides(["v_max=5"])
    with pytest.raises(ValueError):
        Config().with_overrides(["world_w=500"])


def test_reseed_clones_living_and_is_counted():
    w = World(small(food_inflow=0.05, min_agents=40))
    w.run(1500)
    s = w.summary()
    assert s["reseeds"] > 0 and s["pop"] >= 40
    b, _ = w.drain_events()
    rs = b[b["kind"] == k.BIRTH_RESEED]
    assert len(rs) == s["reseeds"]
    assert (rs["parent"] >= 0).all(), "살아 있는 개체가 있으면 그 복제본을 넣는다"
    assert s["births"] == int((b["kind"] == k.BIRTH_NATURAL).sum()), "재주입은 자연 출생으로 세지 않는다"
    bal = w.energy_balance()
    assert abs(bal["residual"]) / (bal["spawn"] + bal["inject"]) < 1e-9


def test_passive_control_follows_reseeds_too(tmp_path):
    """본 실행이 재주입으로 버틸 때 대조 실행만 무너지면 둘을 비교할 수 없다."""
    cfg = small(food_inflow=0.05, min_agents=40)
    main = Runner.new(tmp_path / "main", cfg)
    main.loop(1200, quiet=True)
    main.close()
    assert main.world.summary()["reseeds"] > 0
    pas = Runner.branch(tmp_path / "main", tmp_path / "passive", 0, [], True, True)
    pas.loop(None, quiet=True)
    pas.close()
    assert abs(pas.world.summary()["pop"] - main.world.summary()["pop"]) <= 3


def test_run_folder_lock(tmp_path):
    a = Runner.new(tmp_path / "r", small())
    with pytest.raises(RuntimeError):
        Runner.resume(tmp_path / "r")
    a.close()
    b = Runner.resume(tmp_path / "r")
    b.close()


def test_resume_skips_corrupt_latest_checkpoint(tmp_path):
    r = Runner.new(tmp_path / "r", small())
    r.loop(800, quiet=True)
    r.close()
    cps = ckpt.list_checkpoints(tmp_path / "r" / "checkpoints")
    assert cps[-1][0] == 800 and cps[-2][0] == 400
    meta = json.loads(cps[-1][1].with_suffix(".json").read_text())
    meta["state_hash"] = "0" * 64
    cps[-1][1].with_suffix(".json").write_text(json.dumps(meta))
    r2 = Runner.resume(tmp_path / "r")
    assert r2.world.step_no == 400
    r2.loop(400, quiet=True)
    r2.close()
    assert r2.world.state_hash() == r.world.state_hash()
    assert ev(tmp_path / "r", "births")["step"].max() < 800


def test_birth_order_is_not_slot_order_when_full():
    """상한이 꽉 찼을 때 늘 0번 칸부터 훑으면 낮은 칸 부모만 번식한다."""
    w = World(small(max_agents=120, init_agents=120, food_inflow=3.0, min_agents=0))
    w.run(6000)
    assert w.summary()["blocked"] > 0, "이 시험은 상한에 닿아야 뜻이 있다"
    b, _ = w.drain_events()
    nat = b[b["kind"] == k.BIRTH_NATURAL]
    # 출생 순간의 부모 칸은 기록에 없으므로, 살아 있는 개체의 칸별 자식 수로 본다
    s = w.s
    slots = np.flatnonzero(s.alive)
    low = s.n_off[slots[slots < 60]].sum()
    high = s.n_off[slots[slots >= 60]].sum()
    assert len(nat) > 200
    assert high > 0.25 * (low + high), (low, high)


def test_analyze_and_assay_run(tmp_path):
    r = Runner.new(tmp_path / "r", small(max_age=300, repro_min_age=50))
    r.loop(4000, quiet=True)
    r.close()
    rep = analyze.report(tmp_path / "r")
    assert rep["birth_ratio"] is not None and rep["steps_per_generation"] > 0
    assert rep["fitness_transmission"]["windows"] >= 2
    part = analyze.report(tmp_path / "r", last_steps=1500)
    assert part["from_step"] == 2500
    cps = ckpt.list_checkpoints(tmp_path / "r" / "checkpoints")
    res = measure.assay(cps[1][1], cps[-1][1], n_each=40, steps=1500, reps=3, seed=5)
    assert len(res["final_counts"]) == 3 and np.isfinite(res["r_per_1000"]["mean"])
    lad = measure.ladder(tmp_path / "r", 3, 30, 1000, 2, 7, tmp_path / "ladder.json")
    assert lad["cells_done"] == lad["cells_total"] == 3
    assert json.loads((tmp_path / "ladder.json").read_text())["steps"] == lad["steps"]


def test_config_types_and_energy_rules():
    c = Config().with_overrides(["max_steps_per_sec=0.5"])
    assert c.max_steps_per_sec == 0.5
    for bad in ("child_frac=1.5", "child_frac=0", "birth_cost=-1", "stats_every=0", "frame_ring_segments=0",
                "vision_sectors=40", "vision_fov=4", "max_agents=2.5", "back_cost_mult=0.5"):
        with pytest.raises(ValueError):
            Config().with_overrides([bad])


def test_assay_reports_extinction_as_a_loss(tmp_path):
    """한쪽이 일찍 전멸한 판이 '차이 없음'으로 읽히면 안 된다."""
    r = Runner.new(tmp_path / "r", small())
    r.loop(4000, quiet=True)
    r.close()
    a = measure.load_side(ckpt.list_checkpoints(tmp_path / "r" / "checkpoints")[-1][1])
    weak = {"path": "weak", "step": 0, "gen_mean": 0.0, "cfg": a["cfg"],
            "genomes": {key: v.copy() for key, v in a["genomes"].items()}}
    weak["genomes"]["bout"][:, 0] = -5.0     # 늘 뒤로만 가고
    weak["genomes"]["bout"][:, 2] = -5.0     # 번식하지 않는 무리
    weak["genomes"]["Wout"][:] = 0.0
    res = measure.assay_sides(weak, a, n_each=40, steps=6000, reps=3, seed=3, neutral=False)
    assert res["a_extinct_runs"] == 3
    assert "전멸" in res["verdict"] and res["verdict"].startswith("B가 이김")
    assert res["r_per_1000"]["mean"] > 0.3
