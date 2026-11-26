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
