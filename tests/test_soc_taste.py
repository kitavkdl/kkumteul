"""사회 학습(보고 배우기) 과제 1 시험: 설정·난수·필드·옛 호환. 이 과제에서는 동작이 바뀌지 않는다.

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

from test_stages import small  # noqa: E402
from test_taste import taste_world  # noqa: E402
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
