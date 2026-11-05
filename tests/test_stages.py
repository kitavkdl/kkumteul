"""2~4단계 장치 시험: 켜도 세계가 살고, 에너지가 보존되고, 재현되고, 1단계는 그대로다.

    python3 -m pytest -q tests/test_stages.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kkumteul import checkpoint as ckpt  # noqa: E402
from kkumteul import kernels as k  # noqa: E402
from kkumteul import stages as st  # noqa: E402
from kkumteul.config import (ACT_DEPOSIT, ACT_GIVE, IN_HEAR, IN_HORM, IN_KIN, IN_TRAIL, N_INTERNAL,  # noqa: E402
                             Config)
from kkumteul.run import Runner  # noqa: E402
from kkumteul.world import World  # noqa: E402


def small(*presets, **kw) -> Config:
    base = dict(world_w=256.0, world_h=256.0, max_agents=600, init_agents=150, food_cap=500,
                food_inflow=2.2, n_patches=4, stats_every=200, frame_every=50, clip_every=300,
                clip_len=20, ckpt_every=400, keyframe_every=200, min_agents=20, event_segment_steps=700)
    base.update(kw)
    cfg = Config.from_dict({**Config().to_dict(), **base})
    return cfg.with_overrides([f"preset={p}" for p in presets])


def rel_residual(w: World) -> float:
    b = w.energy_balance()
    return abs(b["residual"]) / (b["spawn"] + b["inject"])


@pytest.mark.parametrize("preset", ["stage2", "stage3", "stage4"])
def test_stage_world_lives_conserves_energy_and_repeats(preset):
    a = World(small(preset))
    a.run(1500)
    a.check_sane()
    s = a.summary()
    assert s["births"] > 0 and s["pop"] > 20
    assert rel_residual(a) < 1e-9
    b = World(small(preset))
    b.run(1500)
    assert a.state_hash() == b.state_hash()


def test_stage_flags_off_do_not_change_stage1():
    """장치를 끈 채로는 그 설정값을 아무리 바꿔도 세계가 같아야 한다."""
    a = World(small())
    a.run(500)
    b = World(small(eta_init=0.1, give_rate=0.5, wear_rate=1.0, hear_range=40.0, fatigue_rate=0.05))
    b.run(500)
    for name in ("x", "y", "energy", "Win", "Wrec", "n_hidden", "ids"):
        assert np.array_equal(getattr(a.s, name)[a.s.alive], getattr(b.s, name)[b.s.alive]), name
    assert not a.s.trail.any() and not a.s.soil.any()
    assert a.s.Hin.shape[1:] == (1, 1), "마음이 꺼져 있으면 학습 배열을 크게 잡지 않는다"


def test_mind_hormones_sleep_and_learning_happen():
    w = World(small("stage2"))
    w.run(1200)
    s = w.s
    a = s.alive
    base = 2 * w.cfg.vision_sectors + N_INTERNAL
    assert np.abs(s.horm[a]).sum() > 0
    assert (s.asleep[a] > 0).any(), "피로가 쌓이면 잠드는 개체가 있어야 한다"
    assert np.abs(s.Hrec[a]).max() > 0, "보상이 있으면 학습 흔적이 남아야 한다"
    assert np.abs(s.Hrec[a]).max() <= 1.0
    assert (s.fatigue[a] >= 0).all() and (s.fatigue[a] <= 1).all()
    awake = a & (s.asleep <= 0)
    assert np.abs(s.obs[awake][:, base + IN_HORM:base + IN_HORM + 4]).sum() > 0
    # 자는 개체는 움직이지 않으려 한다
    asleep = a & (s.asleep > 0)
    assert np.all(s.act[asleep][:, 0] == 0)


def test_newborn_does_not_inherit_what_parent_learned():
    w = World(small("stage2"))
    w.run(1500)
    s = w.s
    newborn = s.alive & (s.age <= 1) & (s.gen > 0)
    if newborn.any():
        assert np.abs(s.Hrec[newborn]).max() < 0.05
        assert (s.fatigue[newborn] < 0.01).all()


def test_social_hearing_kin_and_giving():
    cfg = small("stage3", init_agents=2, min_agents=0, food_inflow=0.0)
    w = World(cfg)
    s = w.s
    i, j = np.flatnonzero(s.alive)[:2]
    s.x[i], s.y[i], s.heading[i] = 100.0, 100.0, 0.0
    s.x[j], s.y[j] = 110.0, 100.0          # i의 앞쪽 10 거리
    s.vx[:] = s.vy[:] = 0.0
    s.sig[j, 0], s.sig[j, 1] = 1.0, -1.0
    s.color[i] = s.color[j] = 0.5
    gw, gh = w.cfg.grid_w, w.cfg.grid_h
    k.build_grid(s.x, s.y, s.alive, 1.0 / w.cfg.cell, gw, gh, s.a_start, s.a_items, s.cursor)
    s.obs[:] = 0
    st.social_sense(s, w.fs, w.isp)
    base = 2 * w.cfg.vision_sectors + N_INTERNAL
    assert s.obs[i, base + IN_HEAR + 0] > 0.3      # 신호1을 앞에서 들음
    assert s.obs[i, base + IN_HEAR + 1] == 0       # 뒤에서는 안 들림
    assert s.obs[i, base + IN_HEAR + 2] < -0.3     # 신호2는 음수
    assert s.obs[i, base + IN_KIN] == pytest.approx(1.0)
    s.color[j] = 0.0
    st.social_sense(s, w.fs, w.isp)
    assert s.obs[i, base + IN_KIN] < 0.6           # 색이 다르면 덜 닮았다
    # 주기: 닿아 있을 때만, 준 만큼에서 손실을 뺀 양이 간다
    s.x[j] = 101.5
    k.build_grid(s.x, s.y, s.alive, 1.0 / w.cfg.cell, gw, gh, s.a_start, s.a_items, s.cursor)
    s.act[:] = 0
    s.act[i, ACT_GIVE] = 1.0
    ei, ej, lost = s.energy[i], s.energy[j], s.ledger[k.L_LOST]
    st.give(s, w.fs, w.isp)
    assert s.energy[i] == pytest.approx(ei - cfg.give_rate)
    assert s.energy[j] == pytest.approx(ej + cfg.give_rate * (1 - cfg.give_loss))
    assert s.ledger[k.L_LOST] == pytest.approx(lost + cfg.give_rate * cfg.give_loss)
    s.x[j] = 130.0
    k.build_grid(s.x, s.y, s.alive, 1.0 / w.cfg.cell, gw, gh, s.a_start, s.a_items, s.cursor)
    ei = s.energy[i]
    st.give(s, w.fs, w.isp)
    assert s.energy[i] == ei, "떨어져 있으면 주지 못한다"


def test_niche_trails_soil_and_fertile_food():
    cfg = small("stage4")
    w = World(cfg)
    w.run(800)
    s = w.s
    assert s.trail.max() > 0, "다닌 자리에 흔적이 남아야 한다"
    assert rel_residual(w) < 1e-9
    # 거름을 한 칸에 몰아 주면 그 칸에 먹이가 난다
    w2 = World(small("stage4", init_agents=0, min_agents=0, n_patches=0, food_uniform_frac=1.0))
    cell = 5 * w2.cfg.grid_w + 7
    w2.s.soil[cell] = 200.0
    w2.s.ledger[k.L_INJECT] += 200.0     # 시험용으로 넣은 에너지를 장부에 적는다
    w2.run(300)
    assert w2.s.counters[st.C_FERTILE] > 20
    fa = w2.s.f_alive
    cx = (w2.s.fx[fa] // w2.cfg.cell).astype(int)
    cy = (w2.s.fy[fa] // w2.cfg.cell).astype(int)
    assert ((cx == 7) & (cy == 5)).sum() > 20
    assert w2.s.soil[cell] < 200.0
    assert rel_residual(w2) < 1e-9


def test_branch_from_stage1_into_later_stages(tmp_path):
    """1단계 세계에서 갈라 다음 단계 장치를 켤 수 있어야 한다(장치의 효과를 견주는 길)."""
    main = Runner.new(tmp_path / "main", small())
    main.loop(400, quiet=True)
    main.close()
    for preset in ("stage2", "stage4"):
        r = Runner.branch(tmp_path / "main", tmp_path / preset, 400, [f"preset={preset}"], False, False)
        assert r.world.s.Hin.shape[1] == r.world.cfg.n_in
        r.loop(600, quiet=True)
        r.close()
        assert r.world.summary()["pop"] > 20 and rel_residual(r.world) < 1e-9
        diff = json.loads((tmp_path / preset / "run.json").read_text())["config_diff"]
        assert diff["mind_enabled"] == [False, True]
        # 갈라낸 실행도 이어 돌리면 끊지 않은 것과 같다
        again = Runner.resume(tmp_path / preset)
        assert again.world.state_hash() == r.world.state_hash()
        again.close()


def test_old_checkpoint_with_fewer_inputs_loads(tmp_path):
    """입출력 자리가 적던 옛 체크포인트: 모자란 자리는 0 가중치로 채워 읽는다."""
    w = World(small())
    w.run(100)
    p = ckpt.save(w, tmp_path)
    old_in, old_out = w.cfg.n_in - 9, w.cfg.n_out - 4
    with np.load(p) as z:
        arrays = {name: z[name] for name in z.files}
    arrays["Win"] = arrays["Win"][:, :old_in, :]
    arrays["Wout"] = arrays["Wout"][:, :old_out, :]
    arrays["bout"] = arrays["bout"][:, :old_out]
    arrays["budget"] = arrays["budget"][:1]
    for name in ("rw", "eta", "alpha", "trail", "soil", "horm", "Hin", "sig"):
        del arrays[name]
    np.savez(p, **arrays)
    meta = json.loads(p.with_suffix(".json").read_text())
    meta["config"]["n_spare_in"], meta["config"]["n_spare_out"] = 4, 2
    meta["npz_sha256"] = ckpt.file_sha256(p)
    p.with_suffix(".json").write_text(json.dumps(meta))
    w2, _ = ckpt.load(p)
    a = w2.s.alive
    assert w2.cfg.n_in == w.cfg.n_in
    assert np.array_equal(w2.s.Win[a][:, :old_in, :], w.s.Win[a][:, :old_in, :])
    assert not w2.s.Win[a][:, old_in:, :].any()
    assert (w2.s.eta[a] == np.float32(w.cfg.eta_init)).all() and w2.s.rw[a][:, 0].min() == 1.0
    assert np.abs(w2.s.alpha[a]).mean() > 0.01, "가소성 유전자가 전부 0이면 갈라낸 뒤 학습이 효과를 못 낸다"
    w2.run(200)
    w2.check_sane()


def test_preset_names():
    assert Config().with_overrides(["preset=stage3"]).social_enabled
    assert not Config().with_overrides(["preset=stage3"]).niche_enabled
    with pytest.raises(ValueError):
        Config().with_overrides(["preset=stage9"])


def test_hash_and_checkpoint_cover_the_ground(tmp_path):
    w = World(small("stage4"))
    w.run(300)
    h = w.state_hash()
    w.s.trail[5] += 1.0
    assert w.state_hash() != h, "길이 바뀌면 상태 해시도 바뀌어야 한다"
    w.s.trail[5] -= 1.0
    p = ckpt.save(w, tmp_path)
    with np.load(p) as z:
        arrays = {name: z[name] for name in z.files}
    arrays["soil"] = arrays["soil"] + 3.0
    np.savez(p, **arrays)
    with pytest.raises(ValueError):
        ckpt.load(p)                       # 파일이 기록(sha256)과 다르다
    del arrays["energy"]
    np.savez(p, **arrays)
    meta = json.loads(p.with_suffix(".json").read_text())
    meta["npz_sha256"] = ckpt.file_sha256(p)
    p.with_suffix(".json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        ckpt.load(p)                       # 꼭 있어야 하는 필드가 없다


def test_assay_keeps_learning_genes_and_fresh_lifetime_state(tmp_path):
    from kkumteul import measure
    r = Runner.new(tmp_path / "r", small("stage2"))
    r.loop(800, quiet=True)
    r.close()
    side = measure.load_side(ckpt.latest(tmp_path / "r" / "checkpoints"))
    assert {"rw", "eta", "alpha"} <= set(side["genomes"])
    acfg = measure.arena_config(side["cfg"], 3, 10)
    arena = World(acfg, populate=False)
    measure._inject(arena, side["genomes"], 0, 0, 10.0, 10.0, 0.0, 5)
    j = int(np.flatnonzero(arena.s.alive)[0])
    assert arena.s.eta[j] > 0 and np.abs(arena.s.rw[j]).sum() > 0 and np.abs(arena.s.alpha[j]).sum() > 0
    assert arena.s.e_prev[j] == acfg.e_start and not arena.s.Hrec[j].any()


def test_first_agents_do_not_give_or_deposit_by_default():
    w = World(small("stage4"))
    w.run(300)
    s = w.summary()
    assert s["gifts"] < 0.2 * 300 * s["pop"], "처음 개체는 주기 쪽으로 기울어 있지 않아야 한다"
    assert w.s.trail.mean() < 0.5 * w.cfg.trail_sat, "길이 세계 전체에서 포화하면 길이 아니다"


def test_stage_config_validation():
    for bad in ("eta_max=0", "sleep_recover=0", "soil_use=-0.5", "trail_decay=1.5", "hormone_rate_dopamine=3",
                "n_spare_out=5"):
        with pytest.raises(ValueError):
            Config().with_overrides([bad])


def test_compare_control_and_treatment(tmp_path):
    from kkumteul import compare
    main = Runner.new(tmp_path / "main", small())
    main.loop(400, quiet=True)
    main.close()
    for name, sets in (("ctrl", []), ("s2", ["preset=stage2"])):
        r = Runner.branch(tmp_path / "main", tmp_path / name, 400, sets, False, False)
        r.loop(800, quiet=True)
        r.close()
    res = compare.compare(tmp_path / "ctrl", tmp_path / "s2", at=1200, n_each=30, steps=1500, reps=3, seed=1)
    assert res["flags"]["mind_enabled"] == [False, True] and res["step"] == 1200
    t = res["treatment_minus_control_r_per_1000"]
    assert np.isfinite(t["in_treatment_rules"]["mean"]) and np.isfinite(t["in_control_rules"]["mean"])
    assert "A가" not in t["verdict_control_rules"] and "B가" not in t["verdict_treatment_rules"]
    assert res["reading"]
