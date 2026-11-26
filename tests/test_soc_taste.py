"""사회 학습(보고 배우기) 시험. 과제 1: 설정·난수·필드·옛 호환(동작 변화 없음). 과제 2: 속도 유전자 변이·비용. 과제 3: 끼니 기록·watch_meals·세계 연결.

설계 docs/superpowers/specs/2026-11-07-social-learning-design.md, 계획 docs/superpowers/plans/2026-11-26-social-learning.md.

    python3 -m pytest -q tests/test_soc_taste.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_harsh import _place  # noqa: E402
from test_stages import rel_residual, small  # noqa: E402
from test_taste import _natural_child, taste_world  # noqa: E402
from kkumteul import config as cfgmod  # noqa: E402
from kkumteul import kernels as k  # noqa: E402
from kkumteul import measure  # noqa: E402
from kkumteul import pack  # noqa: E402
from kkumteul import rng  # noqa: E402
from kkumteul.world import World  # noqa: E402


def soc_world(**kw) -> World:
    return World(small("stage2", "kinds", "taste", soc_taste_enabled=True, **kw))


# ---- 설정 ----
def test_soc_config_defaults_and_checks():
    c = small("taste")
    assert c.soc_taste_enabled is False and c.soc_taste_mode == "outcome"
    assert c.soc_rate == 0.05 and c.soc_gene_sigma == 0.02 and c.soc_cost == 0.0005 and c.soc_radius == 8.0
    assert small("taste", soc_taste_enabled=True).soc_taste_enabled
    for mode in ("outcome", "blind", "inert", "what"):
        assert small("taste", soc_taste_mode=mode).soc_taste_mode == mode
    with pytest.raises(ValueError):
        small("taste_size", soc_taste_enabled=True)        # 맛 기억이 꺼져 있으면 뜻이 없다
    for bad in (dict(soc_radius=c.cell + 0.5), dict(soc_radius=0.0), dict(soc_radius=-1.0),
                dict(soc_rate=-0.1), dict(soc_rate=1.1), dict(soc_cost=-0.1), dict(soc_gene_sigma=-0.1),
                dict(soc_taste_mode="nope")):
        with pytest.raises(ValueError):
            small("taste", **bad)
    assert small("taste", soc_radius=c.cell).soc_radius == c.cell     # 한 칸까지는 된다


def test_soc_mode_constants():
    assert (cfgmod.SOC_OUTCOME, cfgmod.SOC_BLIND, cfgmod.SOC_INERT, cfgmod.SOC_WHAT) == (0, 1, 2, 3)
    assert cfgmod.SOC_MODES == {"outcome": 0, "blind": 1, "inert": 2, "what": 3}


def test_soc_mode_string_override():
    c = small("taste").with_overrides(["soc_taste_enabled=true", "soc_taste_mode=blind"])
    assert c.soc_taste_enabled is True and c.soc_taste_mode == "blind"
    c2 = c.with_overrides(["soc_rate=0.1", "soc_taste_mode=inert"])
    assert c2.soc_rate == 0.1 and c2.soc_taste_mode == "inert"
    with pytest.raises(ValueError):
        small("taste").with_overrides(["soc_taste_mode=nope"])


def test_soc_params_packed():
    c = small("taste", soc_taste_enabled=True, soc_taste_mode="blind", soc_rate=0.07, soc_cost=0.001,
              soc_gene_sigma=0.03, soc_radius=6.0)
    fp, ip = k.pack_params(c)
    assert ip[k.Q_SOC] == 1 and ip[k.Q_SOC_MODE] == cfgmod.SOC_BLIND
    assert fp[k.P_SOC_RATE] == 0.07 and fp[k.P_SOC_COST] == 0.001
    assert fp[k.P_SOC_GENE_SIGMA] == 0.03 and fp[k.P_SOC_RADIUS] == 6.0
    fp0, ip0 = k.pack_params(small("taste"))
    assert ip0[k.Q_SOC] == 0 and ip0[k.Q_SOC_MODE] == cfgmod.SOC_OUTCOME
    assert k.N_FP == len(fp0) and k.N_IP == len(ip0)


# ---- 필드 ----
def test_soc_fields_registered():
    names = [f[0] for f in k.AGENT_FIELDS]
    assert names.index("soc_g") == names.index("taste_g") + 1
    saved = {f[0]: f[3] for f in k.AGENT_FIELDS}
    assert saved["soc_g"] is True and k.HASH_DEFAULTS["soc_g"] == 0
    for name in ("meal_k", "meal_o", "n_seen", "n_seen_poison", "n_seen_kin", "seen_age_sum"):
        assert name in names and saved[name] is False
    assert "soc_g" in k.SAVED_FIELDS and "meal_k" not in k.SAVED_FIELDS
    w = World(small("stage2", "kinds", "taste"))
    for name in ("soc_g", "meal_k", "meal_o", "n_seen", "n_seen_poison", "n_seen_kin", "seen_age_sum"):
        assert (getattr(w.s, name) == 0).all(), name
    assert w.s.seen_age_sum.dtype == np.int64 and w.s.meal_k.dtype == np.int8


def test_soc_in_genome():
    assert "soc_g" in measure.GENOME
    assert measure.OPTIONAL_GENOME == ("taste_g", "soc_g")


# ---- 묶음 호환 ----
def _bundle(tmp_path, drop):
    """살아 있는 세계의 유전체로 묶음을 만들고, drop에 든 배열을 뺀 묶음을 다시 쓴다."""
    w = taste_world("taste")
    w.run(20)
    g = measure.extract_genomes(w)
    arrays = dict(g)
    arrays[pack.MARK] = np.array(pack.PACK_VERSION, dtype=np.int64)
    arrays["config_json"] = np.array(json.dumps(w.cfg.to_dict(), sort_keys=True))
    arrays["info_json"] = np.array(json.dumps({"label": "t", "n": len(g["n_hidden"])}, sort_keys=True))
    n = len(g["n_hidden"])
    full = tmp_path / "full.npz"
    np.savez(full, **arrays)
    cut = tmp_path / "cut.npz"
    np.savez(cut, **{kk: v for kk, v in arrays.items() if kk not in drop})
    return full, cut, n


def test_pack_reads_bundle_without_soc_g(tmp_path):
    full, cut, n = _bundle(tmp_path, ("soc_g",))
    g, _, _ = pack.read(cut)
    assert (g["soc_g"] == 0).all() and len(g["soc_g"]) == n and g["soc_g"].dtype == np.float32
    assert set(g) == set(measure.GENOME) | {"gen"}
    g_full, _, _ = pack.read(full)
    assert set(g_full) == set(g)


def test_pack_reads_bundle_without_taste_and_soc_g(tmp_path):
    _, cut, n = _bundle(tmp_path, ("taste_g", "soc_g"))
    g, _, _ = pack.read(cut)
    assert (g["taste_g"] == 0).all() and (g["soc_g"] == 0).all() and len(g["taste_g"]) == n
    # 다른 유전체 배열이 빠지면 여전히 오류다
    _, cut2, _ = _bundle(tmp_path, ("soc_g", "Wout"))
    with pytest.raises(ValueError, match="빠졌다"):
        pack.read(cut2)


# ---- 해시·체크포인트 ----
def test_soc_off_world_hash_unchanged():
    # 새 필드는 모두 기본값이라 상태 해시에서 건너뛰어진다. 꺼진 세계가 박은 해시(test_taste.py)와 그대로 같은 것은
    # 그 시험의 통과로 확인하고, 여기서는 soc_g가 0인 채 남고 해시가 재현되는 것만 본다.
    w = taste_world("taste")
    w.run(200)
    assert (w.s.soc_g == 0).all()
    h = w.state_hash()
    w2 = taste_world("taste")
    w2.run(200)
    assert w2.state_hash() == h


def test_old_checkpoint_loads_soc_g_zero():
    w = soc_world()
    w.run(30)
    a = np.flatnonzero(w.s.alive)
    w.s.soc_g[a] = np.linspace(-0.1, 0.2, len(a)).astype(np.float32)
    arrays = {kk: v.copy() for kk, v in w.state_arrays().items()}
    assert "soc_g" in arrays
    # 저장·복원은 값을 보존한다(0이 아닌 값)
    w1 = World(w.cfg, populate=False)
    w1.step_no = w.step_no
    assert w1.load_arrays({kk: v.copy() for kk, v in arrays.items()})
    assert (w1.s.soc_g == w.s.soc_g).all() and np.abs(w1.s.soc_g).max() > 0
    # soc_g가 없는 옛 체크포인트는 0으로 읽힌다
    del arrays["soc_g"]
    w2 = World(w.cfg, populate=False)
    w2.step_no = w.step_no
    assert w2.load_arrays(arrays)
    assert (w2.s.soc_g == 0).all()


# ---- 눈가림 동전의 묶음 키 ----
def test_soc_blind_key_unique():
    assert rng.SOC_ID_STRIDE == 1 << 32
    ids = list(range(0, 130)) + [2 ** 16 - 1, 2 ** 16, 2 ** 31 - 1, 2 ** 31, 2 ** 32 - 2, 2 ** 32 - 1]
    keys = {}
    for o in ids:
        for e in ids:
            key = int(rng.soc_pair_key(o, e))
            assert key not in keys, (o, e, keys[key])
            keys[key] = (o, e)
    assert len(keys) == len(ids) ** 2
    # 관찰자와 시연자를 맞바꾸면 다른 키다
    assert int(rng.soc_pair_key(3, 5)) != int(rng.soc_pair_key(5, 3))


# ---- 과제 2: 보고 배우는 속도 유전자(변이·비용) ----
# 비용은 eat만 직접 불러 잰다: 세계를 돌리면 에너지 감각 입력이 달라져 이동 비용이 갈라져 비용만 떼어 잴 수 없다.
def lone_soc(on: bool, **kw) -> World:
    w = taste_world("taste", soc_taste_enabled=on, init_agents=1, min_agents=0, food_inflow=0.0, **kw)
    w.s.energy[0] = 5.0
    return w


def _eat_n(w, n):
    for _ in range(n):
        k.eat(w.step_no, w.s, w.fp, w.ip)


def _free_slot(w) -> int:
    return int(np.flatnonzero(~w.s.alive)[0])


def _copy_many(w, n, start_cid=10_000):
    """부모 칸 0의 유전체를 서로 다른 아이디(난수 키)로 n번 빈 칸에 베껴 soc_g 값들을 돌려준다."""
    j = _free_slot(w)
    out = []
    for cid in range(start_cid, start_cid + n):
        w.s.soc_g[j] = 99.0
        k.copy_genome(w.s, 0, j, cid, w.fp, w.ip)
        out.append(float(w.s.soc_g[j]))
    return np.array(out)


def test_soc_gene_copied_when_off():
    w = lone_soc(False)
    w.s.soc_g[0] = 0.1
    j = _natural_child(w)                      # 자연 출생: 변이 없이 그대로
    assert w.s.soc_g[j] == pytest.approx(0.1) and w.s.soc_g[0] == pytest.approx(0.1)
    # 다시 심기(RESEED)와 수동(PASSIVE) 출생도 같은 길(copy_genome)이라 베낀다
    for kind in (k.BIRTH_RESEED, k.BIRTH_PASSIVE):
        jj = _free_slot(w)
        k._child(w.s, 0, jj, w.step_no, w.fp, w.ip, 1.0, kind, 0)
        assert w.s.soc_g[jj] == pytest.approx(0.1)
        w.s.alive[jj] = False
    # 변이를 끈 켜진 장치도 그대로
    w = lone_soc(True, mutation_enabled=False)
    w.s.soc_g[0] = 0.1
    assert w.s.soc_g[_natural_child(w)] == pytest.approx(0.1)


def test_soc_gene_mutates_and_clips_when_on():
    w = lone_soc(True, soc_gene_sigma=0.02)
    w.s.soc_g[0] = 0.1
    vals = _copy_many(w, 400)
    assert abs(vals.mean() - 0.1) < 0.005
    assert 0.017 < vals.std() < 0.023
    # 난수는 S_SOC_GENE(자식 id, 0): 같은 키로 직접 뽑은 값과 같다
    for n, cid in enumerate((10_000, 10_001, 10_017)):
        z = rng.normal(w.cfg.seed, rng.S_SOC_GENE, cid, 0)
        assert vals[cid - 10_000] == pytest.approx(0.1 + 0.02 * z, abs=1e-6), (n, cid)
    # 범위 [−soc_rate, 1 − soc_rate]
    for g0 in (0.9, -0.04):
        w = lone_soc(True, soc_gene_sigma=1.0)
        w.s.soc_g[0] = g0
        v = _copy_many(w, 300)
        assert v.min() >= -w.cfg.soc_rate - 1e-6 and v.max() <= 1.0 - w.cfg.soc_rate + 1e-6
        assert np.isclose(v.min(), -w.cfg.soc_rate) and np.isclose(v.max(), 1.0 - w.cfg.soc_rate)
    # 한 번 부른 자연 출생도 변한다
    w = lone_soc(True)
    w.s.soc_g[0] = 0.1
    j = _natural_child(w)
    assert w.s.soc_g[j] != pytest.approx(0.1) and abs(w.s.soc_g[j] - 0.1) < 0.2


def test_soc_gene_zero_for_initial_and_random_agent():
    w = soc_world()
    assert (w.s.soc_g == 0).all()
    w = lone_soc(True)
    w.s.soc_g[~w.s.alive] = 0.3               # 빈 칸에 찌꺼기가 남아 있어도
    nb = k.add_random_agent(w.s, w.step_no, w.fp, w.ip, k.BIRTH_RESEED, 0, rng.S_RESEED, 0)
    assert nb
    new = np.flatnonzero(w.s.alive)[-1]
    assert w.s.soc_g[new] == 0.0


def test_reset_body_clears_soc_fields():
    w = lone_soc(True)
    j = _free_slot(w)
    w.s.soc_g[j] = 0.3
    w.s.meal_o[j] = -1.0
    w.s.meal_k[j] = 1
    w.s.n_seen[j] = w.s.n_seen_poison[j] = w.s.n_seen_kin[j] = 5
    w.s.seen_age_sum[j] = 77
    k._reset_body(w.s, j, 999, -1, 0, 0, w.step_no, 10.0, 10.0, 0.0, 1.0)
    assert w.s.soc_g[j] == 0.0 and w.s.meal_o[j] == 0.0
    assert w.s.n_seen[j] == w.s.n_seen_poison[j] == w.s.n_seen_kin[j] == 0 and w.s.seen_age_sum[j] == 0


def test_soc_cost_per_step_and_none_when_off():
    on = lone_soc(True, soc_cost=0.01)
    off = lone_soc(False, soc_cost=0.01)
    on.s.soc_g[0] = 0.45                       # 실제 속도 0.05 + 0.45 = 0.5 → 스텝당 0.005
    off.s.soc_g[0] = 0.45
    _eat_n(on, 10)
    _eat_n(off, 10)
    assert off.s.energy[0] - on.s.energy[0] == pytest.approx(10 * 0.005, rel=1e-5)
    assert on.s.ledger[k.L_BRAIN] - off.s.ledger[k.L_BRAIN] == pytest.approx(10 * 0.005, rel=1e-5)


def test_soc_cost_zero_rate_is_free():
    on = lone_soc(True, soc_cost=0.01)
    off = lone_soc(False)
    on.s.soc_g[0] = -0.05                      # 실제 속도 0
    _eat_n(on, 10)
    _eat_n(off, 10)
    assert on.s.energy[0] == pytest.approx(off.s.energy[0])
    assert on.s.ledger[k.L_BRAIN] == pytest.approx(off.s.ledger[k.L_BRAIN])
    on.s.soc_g[0] = -0.5                       # 더 음수여도 0으로 잘린다(비용이 음수가 되지 않는다)
    _eat_n(on, 3)
    _eat_n(off, 3)
    assert on.s.energy[0] == pytest.approx(off.s.energy[0])


def test_soc_cost_paid_by_full_agent():
    # 배불러(energy >= e_max) continue하는 개체도 비용을 낸다: 비용이 eat 맨 위 산 개체 루프에 있다
    on = lone_soc(True, soc_cost=0.01)
    off = lone_soc(False, soc_cost=0.01)
    for w in (on, off):
        w.s.soc_g[0] = 0.45
        w.s.energy[0] = w.cfg.e_max + 1.0
    _eat_n(on, 4)
    _eat_n(off, 4)
    assert off.s.energy[0] - on.s.energy[0] == pytest.approx(4 * 0.005, rel=1e-5)


def test_soc_cost_independent_of_taste_gene():
    # 맛 유전자가 꺼져 있어도 soc_taste_enabled면 soc_cost를 낸다. 두 장치는 따로 비용을 낸다
    soc = lone_soc(True, soc_cost=0.01)
    assert not soc.cfg.taste_gene_enabled
    soc.s.soc_g[0] = 0.45
    base = lone_soc(False, soc_cost=0.01)
    both = lone_soc(True, soc_cost=0.01, taste_gene_enabled=True, taste_cost=0.02)
    both.s.soc_g[0] = 0.45
    both.s.taste_g[0] = 0.0                    # 맛 속도 0.5 → 스텝당 0.01
    for w in (soc, base, both):
        _eat_n(w, 10)
    assert base.s.energy[0] - soc.s.energy[0] == pytest.approx(10 * 0.005, rel=1e-5)
    assert base.s.energy[0] - both.s.energy[0] == pytest.approx(10 * (0.005 + 0.01), rel=1e-5)


def test_ledger_closes_with_soc_cost():
    # 비용이 장부(L_BRAIN)에 안 들어가면 잔차가 남는다(속도·비용을 크게 잡아 가릴 수 있게)
    w = soc_world(soc_cost=0.01, soc_rate=0.5)
    w.run(1500)
    assert rel_residual(w) < 1e-9
    assert w.s.ledger[k.L_BRAIN] > 0


def test_soc_gene_checkpoint_roundtrip(tmp_path):
    from kkumteul import checkpoint as ckpt
    w = soc_world()
    w.run(20)
    a = np.flatnonzero(w.s.alive)
    w.s.soc_g[a] = np.linspace(-0.04, 0.3, len(a)).astype(np.float32)
    path = ckpt.save(w, tmp_path)
    w2, _ = ckpt.load(path)
    assert (w2.s.soc_g == w.s.soc_g).all() and np.abs(w2.s.soc_g).max() > 0
    assert w2.state_hash() == w.state_hash()


def test_summary_soc_rate_mean():
    w = soc_world()
    w.run(10)
    a = w.s.alive
    w.s.soc_g[a] = 0.2
    assert w.summary()["soc_rate_mean"] == pytest.approx(0.25)
    w.s.soc_g[a] = -0.2                        # 실제 속도 −0.15 → 0으로 잘린 평균
    assert w.summary()["soc_rate_mean"] == pytest.approx(0.0)
    assert "soc_rate_mean" not in taste_world("taste").summary()


def test_set_config_keeps_soc_g():
    # 분기(set_config)가 soc_g를 지우지 않는다(taste_g와 같은 선례): 켜고 끄는 분기가 한 몸의 값을 보존
    w = soc_world()
    w.run(10)
    a = w.s.alive
    w.s.soc_g[a] = 0.1
    w.set_config(w.cfg.with_overrides(["soc_taste_enabled=false"]))
    assert np.allclose(w.s.soc_g[a], 0.1)


# ---- 과제 3: 끼니 기록·watch_meals·세계 연결·관찰 계수기 ----
def _bad(w) -> int:
    return 1 - k.good_kind(w.cfg.seed, w.step_no, w.cfg.kind_flip_period)


def _shape_cell(kind) -> int:
    return 1 if kind == k.KIND_B else 0


def _feed_seq(w, i, kinds):
    """개체 i 자리에 kinds 순서대로 먹이를 놓고(먼저 놓은 것이 먼저 먹힌다) eat를 한 번 부른다."""
    s, cfg = w.s, w.cfg
    s.energy[i] = 2.0
    for kind in kinds:
        _place(w, e=1.0, kind=kind, x=float(s.x[i]), y=float(s.y[i]))
    k.build_grid(s.fx, s.fy, s.f_alive, 1.0 / cfg.cell, cfg.grid_w, cfg.grid_h, s.f_start, s.f_items, s.cursor)
    k.eat(w.step_no, s, w.fp, w.ip)


@pytest.mark.parametrize("poison_first", [True, False])
def test_meal_record_poison_first(poison_first):
    w = lone_soc(True)
    bad = _bad(w)
    good = 1 - bad
    _feed_seq(w, 0, [bad, good] if poison_first else [good, bad])
    assert w.s.n_meal[0] == 2 and w.s.n_poison[0] == 1          # 둘 다 먹었다(양성 확인)
    assert w.s.meal_o[0] == -1.0 and w.s.meal_k[0] == _shape_cell(bad)


def test_meal_record_last_good_when_no_poison():
    w = lone_soc(True)
    bad = _bad(w)
    good = 1 - bad
    _feed_seq(w, 0, [good])
    assert w.s.meal_o[0] == 1.0 and w.s.meal_k[0] == _shape_cell(good)
    _feed_seq(w, 0, [k.KIND_CORPSE])                            # 사체는 + 모양 칸 0
    assert w.s.meal_o[0] == 1.0 and w.s.meal_k[0] == 0


def test_meal_record_cleared_when_no_meal():
    w = lone_soc(True)
    _feed_seq(w, 0, [1 - _bad(w)])
    assert w.s.meal_o[0] == 1.0
    _eat_n(w, 1)                                                # 다음 스텝에 못 먹음
    assert w.s.meal_o[0] == 0.0
    # 배불러 continue하는 개체도 0
    w.s.meal_o[0] = 1.0
    w.s.energy[0] = w.cfg.e_max + 1.0
    _eat_n(w, 1)
    assert w.s.meal_o[0] == 0.0
    # 죽은 칸도 0
    j = _free_slot(w)
    w.s.meal_o[j] = -1.0
    _eat_n(w, 1)
    assert w.s.meal_o[j] == 0.0


def test_meal_record_off_when_soc_off():
    w = lone_soc(False)
    _feed_seq(w, 0, [_bad(w)])
    assert w.s.n_meal[0] == 1 and w.s.meal_o[0] == 0.0


def watch_world(n, mode="outcome", **kw) -> World:
    """개체 n마리를 서로 멀리(40 간격) 놓은 세계. 맛 기억·계수기는 0, 보고 배우는 속도는 soc_rate(0.5)."""
    kw.setdefault("soc_rate", 0.5)
    w = taste_world("taste", soc_taste_enabled=True, soc_taste_mode=mode, init_agents=n, min_agents=0,
                    food_inflow=0.0, **kw)
    s = w.s
    assert int(s.alive.sum()) == n
    for i in range(n):
        s.x[i] = 20.0 + 40.0 * i
        s.y[i] = 100.0
        s.soc_g[i] = 0.0
        s.parent[i] = -1
    s.taste_m[:] = 0.0
    s.meal_o[:] = 0.0
    return w


def _at(w, i, x, y):
    w.s.x[i] = x
    w.s.y[i] = y


def _meal(w, j, mc, o):
    w.s.meal_k[j] = mc
    w.s.meal_o[j] = o


def _grid(w):
    s, cfg = w.s, w.cfg
    k.build_grid(s.x, s.y, s.alive, 1.0 / cfg.cell, cfg.grid_w, cfg.grid_h, s.a_start, s.a_items, s.cursor)


def _watch(w):
    _grid(w)
    k.watch_meals(w.step_no, w.s, w.fp, w.ip)


def test_watch_meals_moves_observer_toward_outcome():
    w = watch_world(2)
    s = w.s
    _at(w, 0, 100.0, 100.0)
    _at(w, 1, 103.0, 100.0)
    s.taste_m[0] = (0.2, 0.4)
    s.taste_m[1] = (0.1, 0.1)
    s.soc_g[0] = 0.1                                            # 실제 속도 0.5 + 0.1
    _meal(w, 1, 1, -1.0)
    _watch(w)
    assert s.n_seen[0] == 1                                     # 양성 확인: 보았다
    assert s.taste_m[0, 1] == pytest.approx(0.4 + 0.6 * (-1.0 - 0.4))
    assert s.taste_m[0, 0] == pytest.approx(0.2)                # 다른 칸은 그대로
    assert s.taste_m[1] == pytest.approx((0.1, 0.1))            # 먹은 j 자신은 이 커널로 안 바뀜
    assert s.n_seen[1] == 0


@pytest.mark.parametrize("xo,xj,seen", [
    (20.0, 27.9, True), (20.0, 28.1, False),                    # 반지름 8 바로 안·밖
    (7.9, 8.1, True), (8.1, 7.9, True),                         # 격자 칸 경계 건너편
    (0.5, 255.5, True), (255.5, 0.5, True),                     # 토러스 반대편(세계 256)
    (0.5, 247.9, False),                                        # 반대편이라도 멀면 안 보임
])
def test_watch_meals_radius_edges(xo, xj, seen):
    w = watch_world(2)
    _at(w, 0, xo, 100.0)
    _at(w, 1, xj, 100.0)
    _meal(w, 1, 0, 1.0)
    _watch(w)
    assert int(w.s.n_seen[0]) == int(seen)
    assert (w.s.taste_m[0, 0] != 0.0) == seen


def test_watch_meals_skips_dead_and_nonmeal():
    w = watch_world(4)
    s = w.s
    _at(w, 0, 100.0, 100.0)
    _at(w, 1, 103.0, 100.0)             # 끼니 있음
    _at(w, 2, 100.0, 103.0)             # 끼니 없음
    _at(w, 3, 97.0, 100.0)              # 끼니 있지만 죽음
    _meal(w, 1, 0, 1.0)
    _meal(w, 3, 0, 1.0)
    _grid(w)
    s.alive[3] = False                  # 격자를 지은 뒤 죽인다: 커널이 스스로 산 개체인지 본다
    k.watch_meals(w.step_no, s, w.fp, w.ip)
    assert s.n_seen[0] == 1 and s.taste_m[0, 0] == pytest.approx(0.5)
    # 죽은 관찰자는 안 바뀐다
    w2 = watch_world(2)
    _at(w2, 0, 100.0, 100.0)
    _at(w2, 1, 103.0, 100.0)
    _meal(w2, 1, 0, 1.0)
    _grid(w2)
    w2.s.alive[0] = False
    k.watch_meals(w2.step_no, w2.s, w2.fp, w2.ip)
    assert w2.s.n_seen[0] == 0 and w2.s.taste_m[0, 0] == 0.0


@pytest.mark.parametrize("mode,expect", [("outcome", 0.4 + 0.5 * (-1.0 - 0.4)), ("what", 0.4 + 0.5 * (1.0 - 0.4)),
                                         ("inert", 0.4)])
def test_watch_meals_modes(mode, expect):
    w = watch_world(2, mode)
    s = w.s
    _at(w, 0, 100.0, 100.0)
    _at(w, 1, 103.0, 100.0)
    s.taste_m[0] = (0.2, 0.4)
    _meal(w, 1, 1, -1.0)                # 독을 먹었다
    _watch(w)
    assert s.n_seen[0] == 1 and s.n_seen_poison[0] == 1         # 보기는 모든 모드에서 같다
    assert s.taste_m[0, 1] == pytest.approx(expect)
    assert s.taste_m[0, 0] == pytest.approx(0.2)


def test_watch_meals_zero_rate_observer():
    w = watch_world(2)
    s = w.s
    _at(w, 0, 100.0, 100.0)
    _at(w, 1, 103.0, 100.0)
    s.soc_g[0] = -0.5                   # 실제 속도 0
    s.taste_m[0] = (0.2, 0.4)
    _meal(w, 1, 1, -1.0)
    _watch(w)
    assert s.n_seen[0] == 1             # 보기는 보았다
    assert s.taste_m[0] == pytest.approx((0.2, 0.4))


def _crowd(w):
    """관찰자 0과 겹쳐 보이는 시연자 둘(1, 2: 같은 모양 칸에 독·좋은 알)과 서로를 보는 자리."""
    _at(w, 0, 100.0, 100.0)
    _at(w, 1, 103.0, 100.0)
    _at(w, 2, 100.0, 103.0)
    w.s.taste_m[0] = (0.3, 0.0)
    _meal(w, 1, 0, -1.0)
    _meal(w, 2, 0, 1.0)


def test_watch_meals_order_deterministic():
    hashes = []
    for _ in range(2):
        w = watch_world(3)
        _crowd(w)
        _watch(w)
        hashes.append(w.state_hash())
    assert hashes[0] == hashes[1]
    # 한 관찰자의 한 칸을 시연자 1 < 2가 차례로 갱신하는 값이 칸 번호순 식과 같다(지수평활은 비가환)
    w = watch_world(3)
    _crowd(w)
    _watch(w)
    m = 0.3
    m += 0.5 * (-1.0 - m)               # 시연자 1(독)
    m += 0.5 * (1.0 - m)                # 시연자 2(좋은 알)
    assert w.s.taste_m[0, 0] == pytest.approx(m)
    assert w.s.n_seen[0] == 2


def test_seen_counters():
    w = watch_world(5)
    s = w.s
    s.ids[:5] = (10, 11, 77, 13, 14)
    s.parent[0] = 77                    # 관찰자의 부모는 칸 2의 개체
    s.parent[1] = 10                    # 칸 1은 관찰자의 자식
    s.parent[2] = -1
    s.parent[3] = 77                    # 칸 3은 형제(같은 부모)
    s.parent[4] = -1                    # 칸 4는 남(부모 없음끼리는 형제가 아니다)
    s.parent[0] = 77
    for j, age in zip((1, 2, 3, 4), (5, 7, 11, 13)):
        s.age[j] = age
    _at(w, 0, 100.0, 100.0)
    _at(w, 1, 103.0, 100.0)
    _at(w, 2, 100.0, 104.0)
    _at(w, 3, 96.0, 100.0)
    _at(w, 4, 100.0, 96.0)
    _meal(w, 1, 0, -1.0)
    _meal(w, 2, 0, 1.0)
    _meal(w, 3, 0, 1.0)
    _meal(w, 4, 1, -1.0)
    _watch(w)
    assert s.n_seen[0] == 4
    assert s.n_seen_poison[0] == 2
    assert s.n_seen_kin[0] == 3
    assert s.seen_age_sum[0] == 5 + 7 + 11 + 13


def test_world_step_rebuilds_grid_before_watch(monkeypatch):
    w = soc_world(soc_rate=0.5)
    w.run(5)
    s, cfg = w.s, w.cfg
    gw, gh, inv = cfg.grid_w, cfg.grid_h, 1.0 / cfg.cell
    seen = []
    real = k.watch_meals

    def spy(step, st, fp, ip):
        start2 = np.zeros_like(st.a_start)
        items2 = np.zeros_like(st.a_items)
        cur2 = np.zeros_like(st.cursor)
        k.build_grid(st.x, st.y, st.alive, inv, gw, gh, start2, items2, cur2)
        n = int(start2[-1])
        seen.append(bool((st.a_start == start2).all() and (st.a_items[:n] == items2[:n]).all()))
        return real(step, st, fp, ip)

    monkeypatch.setattr(k, "watch_meals", spy)
    w.run(5)
    assert seen == [True] * 5
    # 장치가 꺼진 세계: watch_meals도, 추가 build_grid도 없다
    calls = {"watch": 0, "grid": 0}
    real_grid = k.build_grid

    def spy_watch(*a):
        calls["watch"] += 1
        return real(*a)

    def spy_grid(*a):
        calls["grid"] += 1
        return real_grid(*a)

    monkeypatch.setattr(k, "watch_meals", spy_watch)
    monkeypatch.setattr(k, "build_grid", spy_grid)
    off = taste_world("taste")
    off.run(3)
    grid_off = calls["grid"]
    assert calls["watch"] == 0
    on = soc_world()
    calls["grid"] = 0
    on.run(3)
    assert calls["watch"] == 3 and calls["grid"] == grid_off + 3


def test_soc_off_world_never_calls_watch(monkeypatch):
    n = []
    real = k.watch_meals
    monkeypatch.setattr(k, "watch_meals", lambda *a: (n.append(1), real(*a))[1])
    w = taste_world("taste")
    w.run(30)
    assert n == []
    w_gene_only = taste_world("taste", taste_gene_enabled=True)
    w_gene_only.run(10)
    assert n == []


# ---- 과제 4: 눈가림 모드(blind) ----
def _blind_coin(w, i, j, step):
    """watch_meals가 쓰는 것과 같은 식: 모양 칸 = 동전(씨앗, 눈가림 용도, 스텝, 관찰자·시연자 묶음 키)."""
    s = w.s
    return int(rng.u01(w.cfg.seed, rng.S_SOC_BLIND, step, rng.soc_pair_key(int(s.ids[i]), int(s.ids[j]))) * 2.0)


def _blind_pair(mode="blind", o=-1.0, mc=1):
    w = watch_world(2, mode)
    _at(w, 0, 100.0, 100.0)
    _at(w, 1, 103.0, 100.0)
    _meal(w, 1, mc, o)
    return w


def test_blind_reproducible():
    hashes = []
    for _ in range(2):
        w = _blind_pair()
        _grid(w)
        for step in (5, 6, 7, 8):
            k.watch_meals(step, w.s, w.fp, w.ip)
        hashes.append(w.state_hash())
    assert hashes[0] == hashes[1]
    # 같은 (스텝, 쌍)이면 같은 칸, 식과도 같다
    for step in (0, 1, 2, 3, 40, 41):
        w = _blind_pair()
        _grid(w)
        k.watch_meals(step, w.s, w.fp, w.ip)
        c = _blind_coin(w, 0, 1, step)
        assert w.s.taste_m[0, c] == pytest.approx(-0.5)
        assert w.s.taste_m[0, 1 - c] == 0.0
        assert (w.s.taste_m[1] == 0.0).all()        # 먹은 이 자신은 안 바뀐다


def test_blind_balanced():
    w = _blind_pair()
    _grid(w)
    n = 4000
    hits = 0
    for step in range(n):
        w.s.taste_m[:] = 0.0
        k.watch_meals(step, w.s, w.fp, w.ip)
        assert (w.s.taste_m[0] != 0.0).sum() == 1   # 한 번에 한 칸만
        hits += int(w.s.taste_m[0, 1] != 0.0)
    assert abs(hits - n / 2) < 5.0 * (n * 0.25) ** 0.5
    # 관찰자·시연자 쌍이 바뀌면 같은 스텝에서도 다른 동전이 나올 수 있다(쌍 키가 쌍마다 다르다)
    assert any(int(rng.u01(w.cfg.seed, rng.S_SOC_BLIND, s_, rng.soc_pair_key(1, 2)) * 2.0)
               != int(rng.u01(w.cfg.seed, rng.S_SOC_BLIND, s_, rng.soc_pair_key(2, 1)) * 2.0) for s_ in range(64))


def test_blind_same_observations_as_outcome():
    res = {}
    for mode in ("outcome", "blind"):
        w = watch_world(3, mode)
        _crowd(w)                                   # 시연자 1(독), 2(좋은 알)을 관찰자 0이 봄
        w.s.taste_m[0] = (0.3, 0.1)
        _grid(w)
        k.watch_meals(9, w.s, w.fp, w.ip)
        s = w.s
        res[mode] = (s.n_seen.copy(), s.n_seen_poison.copy(), s.n_seen_kin.copy(), s.seen_age_sum.copy())
    o, b = res["outcome"], res["blind"]
    for a, c in zip(o, b):                                  # 보기 자체(계수기 넷, 개체 전부)는 같다
        assert (a == c).all()
    assert o[0][0] == 2 and o[1][0] == 1
    # blind: 결과(독 −1, 좋은 알 +1)는 그대로 쓰되 칸만 동전으로 — 두 번의 갱신이 각자 동전 칸에 들어간다
    w = watch_world(3, "blind")
    _crowd(w)
    w.s.taste_m[0] = (0.3, 0.1)
    expect = np.array([0.3, 0.1])
    for j, oj in ((1, -1.0), (2, 1.0)):
        c = _blind_coin(w, 0, j, 9)
        expect[c] += 0.5 * (oj - expect[c])
    _grid(w)
    k.watch_meals(9, w.s, w.fp, w.ip)
    assert w.s.taste_m[0] == pytest.approx(expect)
