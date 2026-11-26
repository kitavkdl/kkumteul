"""사회 학습(보고 배우기) 시험. 과제 1: 설정·난수·필드·옛 호환(동작 변화 없음). 과제 2: 속도 유전자 변이·비용.

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
