"""겨루기 마당이 원래 세계의 독 순서를 잇는지(2026-11-26: 고정 갈래 겨루기가 독이 뒤바뀐 마당에서 양쪽 전멸)."""
import numpy as np
import pytest

from kkumteul import kernels as k
from kkumteul import measure
from kkumteul.config import Config


def _cfg(**kw):
    base = dict(food_kinds_enabled=True, kind_flip_period=10**9, seed=1)
    base.update(kw)
    return Config(**base)


def test_default_is_old_rule():
    """kind_seed −1·kind_step0 0이면 예전 셈(seed, step)과 같다 — 옛 세계의 해시·재현이 그대로."""
    cfg = _cfg(kind_flip_period=7)
    for step in range(0, 200, 3):
        assert k.cfg_good_kind(cfg, step) == int(k.good_kind(cfg.seed, step, 7))
    ip = k.pack_params(cfg)[1]
    assert ip[k.Q_KIND_SEED] == cfg.seed and ip[k.Q_KIND_STEP0] == 0


@pytest.mark.parametrize("period", [10**9, 500])
def test_arena_keeps_original_poison_order(period):
    """마당 씨앗이 무엇이든 마당 스텝 t의 좋은 종류 = 원래 세계 (체크포인트 스텝 + t)의 좋은 종류."""
    orig = _cfg(kind_flip_period=period, seed=1)
    step = 2_600_000
    aligned = measure.kind_aligned(orig, step)
    for s in range(6000, 6012):
        arena = measure.arena_config(aligned, s, 800)
        assert arena.seed == s
        for t in (0, 1, 499, 500, 7999):
            assert k.cfg_good_kind(arena, t) == k.cfg_good_kind(orig, step + t)
        ip = k.pack_params(arena)[1]
        assert ip[k.Q_KIND_SEED] == 1 and ip[k.Q_KIND_STEP0] == step


def test_unaligned_arena_would_flip():
    """고치기 전 꼴(마당 씨앗으로 다시 뽑기)이면 씨앗 열둘 가운데 원래와 다른 마당이 생긴다 — 위 시험이 장치를 가른다."""
    orig = _cfg(seed=1)
    step = 2_600_000
    want = k.cfg_good_kind(orig, step)
    got = [k.cfg_good_kind(measure.arena_config(orig, s, 800), 0) for s in range(6000, 6012)]
    assert any(g != want for g in got)


def test_kinds_off_unchanged():
    cfg = Config()
    assert measure.kind_aligned(cfg, 12345) is cfg


def test_eat_uses_aligned_kind():
    """커널 eat이 kind_seed·kind_step0을 쓴다: 좋은 종류가 반대가 되는 kind_step0을 주면 같은 세계에서
    독으로 센 수가 달라진다(먹이가 나는 자리·종류는 kind_step0과 무관)."""
    from kkumteul.world import World
    base = _cfg(seed=5, init_agents=150, max_agents=400)
    g0 = k.cfg_good_kind(base, 0)
    step0 = next(n * 10**9 for n in range(1, 64) if k.cfg_good_kind(_cfg(seed=5, kind_step0=n * 10**9), 0) != g0)
    counts = []
    for cfg in (base, _cfg(seed=5, init_agents=150, max_agents=400, kind_step0=step0)):
        w = World(cfg)
        w.run(300)
        counts.append((int(w.s.counters[k.C_POISONED]), int(w.s.counters[k.C_EATEN])))
    assert counts[0][1] > 0 and counts[1][1] > 0
    assert counts[0][0] != counts[1][0]


# ---- 체크포인트 밖의 길: 묶음 겨루기·실험 카드 겨루기·손 개체 도구(2026-11-26 검토 지적) ----
def _small_kinds(**kw):
    base = dict(food_kinds_enabled=True, kind_flip_period=500, seed=3, world_w=256.0, world_h=256.0,
                init_agents=60, max_agents=200)
    base.update(kw)
    return Config(**base)


def test_pack_side_is_aligned_like_checkpoint(tmp_path):
    """묶음에서 읽은 한쪽도 체크포인트에서 읽은 것처럼 뽑아 온 스텝의 독 순서를 잇는다."""
    from kkumteul import checkpoint as ckpt, pack
    from kkumteul.world import World
    w = World(_small_kinds())
    w.run(30)
    cp = ckpt.save(w, tmp_path / "c")
    pack.export(cp, tmp_path / "p.npz", n=20)
    side = measure.load_side(tmp_path / "p.npz")
    want = measure.load_side(cp)["cfg"]
    assert side["cfg"].kind_step0 == 30 and side["cfg"].kind_seed == k.kind_seed_of(w.cfg)
    assert side["cfg"] == want
    assert pack.read(tmp_path / "p.npz")[1] == w.cfg, "파일에 적힌 설정은 그대로(읽을 때만 잇는다)"


def test_experiment_assay_keeps_poison_order(monkeypatch, tmp_path):
    """실험 카드 겨루기는 바탕 설정으로 덮어쓰지만 독 순서는 체크포인트 스텝에서 잇는다."""
    from kkumteul import checkpoint as ckpt, experiment
    from kkumteul.world import World
    base = _small_kinds()
    for name in ("ctl", "trt"):
        w = World(base)
        w.run(40)
        ckpt.save(w, tmp_path / name / "checkpoints")
    seen = []

    def fake(a, b, n, steps, reps, seed, neutral=False):
        seen.append((a["cfg"], b["cfg"]))
        return {"r_per_1000": {}, "final_counts": [], "a_extinct_runs": 0, "b_extinct_runs": 0,
                "both_extinct_runs": 0, "winner": ""}

    monkeypatch.setattr(measure, "assay_sides", fake)
    monkeypatch.setattr(measure, "verdict", lambda *a, **kw: "")
    arms = {"control": {"pop_final": 100}, "treatment": {"pop_final": 100}}
    experiment._assay(tmp_path / "ctl", tmp_path / "trt", base, arms, 20, 100, 1, 1)
    a, b = seen[0]
    assert a == b == measure.kind_aligned(base, 40)
    assert k.cfg_good_kind(measure.arena_config(a, 9, 40), 0) == k.cfg_good_kind(base, 40)


def test_hand_learner_after_flips_with_offset_order():
    """kind_step0이 있는 마당에선 바뀜 경계가 주기의 배수가 아니다 — good을 넘기면 실제 바뀜을 찾는다."""
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "wayfinder" / "stage2-4" / "hand_learner.py"
    spec = importlib.util.spec_from_file_location("hand_learner_t", p)
    hl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hl)
    period, every = 250, 50
    acfg = next(c for s0 in range(1, 400, 7)
                for c in [_small_kinds(kind_flip_period=period, kind_step0=s0)]
                if any(k.cfg_good_kind(c, t) != k.cfg_good_kind(c, t - 1) for t in range(1, 1500)))
    flips = [t for t in range(1, 1500) if k.cfg_good_kind(acfg, t) != k.cfg_good_kind(acfg, t - 1)]
    assert all((t + acfg.kind_step0) % period == 0 for t in flips)
    series = [{"step": s, "good": k.cfg_good_kind(acfg, s - 1), "eaten": 10, "poisoned": 1}
              for s in range(every, 1501, every)]
    out = hl.after_flips(series, period, acfg.seed, good=lambda t: k.cfg_good_kind(acfg, t))
    assert [o["flip"] for o in out] == flips


def _hl():
    import importlib.util
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "wayfinder" / "stage2-4" / "hand_learner.py"
    spec = importlib.util.spec_from_file_location("hand_learner_t2", p)
    hl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hl)
    return hl


@pytest.mark.parametrize("fn", ["curve", "solo"])
def test_hand_learner_rows_use_aligned_order(fn):
    """curve·solo의 줄마다 good = 마당 설정 기준(kind_seed·kind_step0 반영) 좋은 종류.
    옛 셈(마당 씨앗·스텝만)과 갈리는 kind_step0을 골라, 정렬을 빼먹으면 이 시험이 깨지게 한다."""
    from kkumteul.world import World
    hl = _hl()
    period, steps, every, n, seed = 100, 600, 50, 20, 4
    w = World(_small_kinds(kind_flip_period=period))
    side = measure.side_from_world(w)

    def arena(c):
        return hl.curve_arena(c, seed, n, lab=False) if fn == "curve" else measure.arena_config(c, seed, n)

    def naive_differs(c):
        a = arena(c)
        return any(k.cfg_good_kind(a, t) != int(k.good_kind(a.seed, t, period)) for t in range(steps))

    cfg = next(c for s0 in range(1, 2000, 37)
               for c in [_small_kinds(kind_flip_period=period, kind_step0=s0)] if naive_differs(c))
    acfg = arena(cfg)
    out = hl.curve(cfg, side["genomes"], n, steps, seed, every=every, lab=False) if fn == "curve" \
        else hl.solo(cfg, side["genomes"], n, steps, seed, every=every)
    rows = out["series"]
    assert rows
    assert all(r["good"] == k.cfg_good_kind(acfg, r["step"] - 1) for r in rows)


def test_pack_to_compete_arena_follows_original_order(monkeypatch, tmp_path):
    """묶음 → load_side → compete 전체를 지나, compete가 실제로 만든 마당 세계의 커널 독 순서가
    원래 세계(뽑아 온 스텝 + 마당 스텝)의 독 순서와 같다. 가짜 assay가 아니라 진짜 compete를 돌린다."""
    from kkumteul import checkpoint as ckpt, pack
    from kkumteul.world import World
    period, at, steps = 60, 37, 300
    orig = _small_kinds(kind_flip_period=period)
    w = World(orig)
    w.run(at)
    cp = ckpt.save(w, tmp_path / "c")
    pack.export(cp, tmp_path / "p.npz", n=20)
    side = measure.load_side(tmp_path / "p.npz")
    built = []

    class Spy(World):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            built.append(self)

    monkeypatch.setattr(measure, "World", Spy)
    seed = 11
    # 마당 씨앗이 원래 씨앗과 달라 옛 셈이면 순서가 갈린다는 것을 먼저 확인(시험이 헛돌지 않게)
    assert any(int(k.good_kind(seed, t, period)) != k.cfg_good_kind(orig, at + t) for t in range(steps))
    measure.compete(side["cfg"], side["genomes"], side["genomes"], 10, steps, seed, sample_every=100)
    assert built, "compete가 마당 세계를 만들지 않았다"
    arena = built[0]
    ip = arena.ip
    assert ip[k.Q_KIND_SEED] == k.kind_seed_of(arena.cfg) and ip[k.Q_KIND_STEP0] == arena.cfg.kind_step0
    assert all(int(k.good_kind(int(ip[k.Q_KIND_SEED]), t + int(ip[k.Q_KIND_STEP0]), period))
               == k.cfg_good_kind(orig, at + t) for t in range(steps))
