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
from kkumteul.config import (ACT_ATTACK, ACT_DEPOSIT, ACT_GIVE, IN_HEAR, IN_HORM, IN_KIN, IN_TRAIL, N_INTERNAL,  # noqa: E402
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
    assert w2.s.counters[k.C_FERTILE] > 20
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
    arrays["counters"] = arrays["counters"][:16]
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
    measure.inject_agent(arena, side["genomes"], 0, 0, 10.0, 10.0, 0.0, 5)
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
    assert t["winner_treatment_rules"] in ("treatment", "control", "")
    assert t["winner_control_rules"] in ("treatment", "control", "")
    # 켜고 끄는 장치는 설정에서 모은다. 새 장치도 카드에 빠지지 않는다
    assert {"food_kinds_enabled", "conflict_enabled", "mind_enabled"} <= set(res["flags"])
    assert "mutation_enabled" not in res["flags"]
    assert res["reading"]


def test_attack_takes_energy_without_creating_any():
    cfg = small("stage3", "conflict", init_agents=2, min_agents=0, food_inflow=0.0)
    w = World(cfg)
    s = w.s
    i, j = np.flatnonzero(s.alive)[:2]
    s.x[i], s.y[i] = 100.0, 100.0
    s.x[j], s.y[j] = 101.5, 100.0
    k.build_grid(s.x, s.y, s.alive, 1.0 / cfg.cell, cfg.grid_w, cfg.grid_h, s.a_start, s.a_items, s.cursor)
    s.act[:] = 0
    s.act[i, ACT_ATTACK] = 1.0
    ei, ej, lost, moved = s.energy[i], s.energy[j], s.ledger[k.L_LOST], s.ledger[k.L_MOVE]
    st.attack(0, s, w.fs, w.isp)
    assert s.energy[j] == pytest.approx(ej - cfg.attack_damage)
    assert s.energy[i] == pytest.approx(ei - cfg.attack_cost + cfg.attack_damage * cfg.attack_gain)
    assert s.ledger[k.L_LOST] == pytest.approx(lost + cfg.attack_damage * (1 - cfg.attack_gain))
    assert s.ledger[k.L_MOVE] == pytest.approx(moved + cfg.attack_cost)
    assert s.touch[j] == 2.0, "맞은 개체는 닿음 감각이 2가 된다"
    # 가진 것보다 더 빼앗지 못한다
    s.energy[j] = 0.05
    st.attack(1, s, w.fs, w.isp)
    assert s.energy[j] == pytest.approx(0.0)
    # 멀리 있으면 비용만 든다
    s.x[j] = 150.0
    s.energy[j] = 1.0
    k.build_grid(s.x, s.y, s.alive, 1.0 / cfg.cell, cfg.grid_w, cfg.grid_h, s.a_start, s.a_items, s.cursor)
    ei = s.energy[i]
    st.attack(2, s, w.fs, w.isp)
    assert s.energy[i] == pytest.approx(ei - cfg.attack_cost) and s.energy[j] == 1.0


def test_conflict_world_lives_and_conserves_energy():
    cfg = small("stage3", "conflict")
    a = World(cfg)
    a.s.bout[:, ACT_ATTACK] = 0.5      # 처음 개체가 공격하게 해서 장치를 실제로 돌린다
    a.run(1500)
    a.check_sane()
    assert a.summary()["attacks"] > 0 and a.summary()["pop"] > 10
    assert rel_residual(a) < 1e-9
    with pytest.raises(ValueError):
        Config().with_overrides(["attack_gain=1.5"])


def test_food_kinds_poison_and_sign():
    cfg = small("kinds", init_agents=1, min_agents=0, food_inflow=0.0)
    w = World(cfg)
    s = w.s
    i = int(np.flatnonzero(s.alive)[0])
    good = k.good_kind(cfg.seed, 0, cfg.kind_flip_period)
    s.x[i], s.y[i], s.heading[i] = 100.0, 100.0, 0.0
    s.vx[:] = s.vy[:] = 0.0
    # 앞쪽 5 거리에 독인 종류 하나
    bad = 1 - good
    s.fx[0], s.fy[0], s.f_energy[0], s.f_kind[0], s.f_alive[0] = 105.0, 100.0, 1.0, bad, True
    s.counters[k.C_NFOOD] = 1
    s.ledger[k.L_SPAWN] += 1.0
    gw, gh, inv = cfg.grid_w, cfg.grid_h, 1.0 / cfg.cell
    k.build_grid(s.x, s.y, s.alive, inv, gw, gh, s.a_start, s.a_items, s.cursor)
    k.build_grid(s.fx, s.fy, s.f_alive, inv, gw, gh, s.f_start, s.f_items, s.cursor)
    k.sense(0, s, w.fp, w.ip)
    vis = s.obs[i, :cfg.vision_sectors]
    assert (vis != 0).any()
    assert (np.sign(vis[vis != 0]) == (-1 if bad == 1 else 1)).all(), "먹이 종류 1은 음수로 보인다"
    # 독을 먹으면 에너지를 잃고 장부에 적힌다
    s.x[i] = 104.6
    k.build_grid(s.x, s.y, s.alive, inv, gw, gh, s.a_start, s.a_items, s.cursor)
    e0, lost = s.energy[i], s.ledger[k.L_LOST]
    k.eat(0, s, w.fp, w.ip)
    assert s.energy[i] == pytest.approx(e0 - cfg.poison)
    assert s.ledger[k.L_LOST] == pytest.approx(lost + 1.0 + cfg.poison)
    assert not s.f_alive[0] and s.counters[k.C_POISONED] == 1
    assert rel_residual(w) < 1e-9
    # 꺼져 있으면 종류는 전부 0이고 독이 없다
    w1 = World(small())
    w1.run(300)
    assert not w1.s.f_kind[w1.s.f_alive & (w1.s.f_kind != 2)].any()


def test_food_kinds_world_runs_and_conserves_energy():
    w = World(small("stage2", "kinds"))
    w.run(2000)
    w.check_sane()
    sm = w.summary()
    assert sm["poisoned"] > 0 and rel_residual(w) < 1e-9
    kinds = w.s.f_kind[w.s.f_alive]
    assert (kinds == 0).any() and (kinds == 1).any()
    b = World(small("stage2", "kinds"))
    b.run(2000)
    assert b.state_hash() == w.state_hash()


def test_checkpoint_from_before_new_fields_still_resumes(tmp_path):
    """장치를 더 넣기 전에 저장한 체크포인트(필드가 없거나 칸 수가 적은 것)도 이어 돌릴 수 있어야 한다."""
    r = Runner.new(tmp_path / "r", small("stage3"))
    r.loop(400, quiet=True)
    r.close()
    for step, p in ckpt.list_checkpoints(tmp_path / "r" / "checkpoints"):
        with np.load(p) as z:
            arrays = {name: z[name] for name in z.files}
        arrays["counters"] = arrays["counters"][:16]
        del arrays["f_kind"]
        arrays.pop("f_born", None)
        np.savez(p, **arrays)
        meta = json.loads(p.with_suffix(".json").read_text())
        meta["npz_sha256"] = ckpt.file_sha256(p)
        p.with_suffix(".json").write_text(json.dumps(meta))
    again = Runner.resume(tmp_path / "r")
    assert again.world.step_no == 400
    again.loop(200, quiet=True)
    again.close()


def test_food_cap_replace_swaps_oldest_and_conserves_energy():
    """먹이가 상한에 닿았을 때: 꺼져 있으면 유입을 버리고, 켜면 가장 오래된 먹이를 새 먹이로 바꾼다."""
    kw = dict(init_agents=1, min_agents=0, food_cap=50, food_inflow=3.0)
    off = World(small(**kw))
    on = World(small(food_cap_replace=True, **kw))
    for t in range(40):                     # 스텝마다 3알씩 120알. 상한 50은 17번째 스텝에 찬다
        k.spawn_food(t, off.s, off.fp, off.ip)
        k.spawn_food(t, on.s, on.fp, on.ip)
    assert off.s.counters[k.C_NFOOD] == 50 and on.s.counters[k.C_NFOOD] == 50
    assert off.s.counters[k.C_FOOD_REPLACED] == 0 and off.s.ledger[k.L_DISCARD] > 0
    assert off.s.f_born[off.s.f_alive].min() == 0, "꺼져 있으면 처음 놓인 먹이가 그대로 남는다"
    assert on.s.counters[k.C_FOOD_REPLACED] == 70 and on.s.ledger[k.L_DISCARD] == 0
    assert on.s.ledger[k.L_LOST] == pytest.approx(70 * on.cfg.food_energy)
    born = on.s.f_born[on.s.f_alive]
    assert born.min() == 23 and (born == 23).sum() == 2, "남은 것은 가장 나중에 놓인 50알이다"
    assert on.summary()["food_replaced"] == 70 and "food_replaced" not in off.summary()
    assert rel_residual(on) < 1e-9 and rel_residual(off) < 1e-9


def test_food_cap_replace_changes_nothing_below_cap():
    a = World(small(food_cap=3000))
    b = World(small(food_cap=3000, food_cap_replace=True))
    a.run(300)
    b.run(300)
    assert a.summary()["food"] < 3000 and a.state_hash() == b.state_hash()


@pytest.mark.parametrize("presets", [("kinds",), ("stage4", "kinds")])
def test_food_cap_replace_world_runs_and_conserves_energy(presets):
    cfg = small(*presets, food_cap=150, food_cap_replace=True)
    w = World(cfg)
    if "stage4" in presets:
        w.s.bout[:, ACT_DEPOSIT] = 0.5     # 거름을 놓게 해서 거름 땅 먹이 쪽 분기도 돌린다
    w.run(2000)
    w.check_sane()
    sm = w.summary()
    assert sm["food_replaced"] > 0 and sm["food"] <= 150 + cfg.max_agents
    if "stage4" in presets:
        assert sm["fertile_food"] > 0
    assert rel_residual(w) < 1e-9
    b = World(cfg)
    if "stage4" in presets:
        b.s.bout[:, ACT_DEPOSIT] = 0.5
    b.run(2000)
    assert b.state_hash() == w.state_hash()


def test_compare_reading_is_not_overconfident():
    """한쪽 규칙만 유의하다는 것으로 '켠 규칙에서만 이긴다'고 하지 않고, 경고와 한계를 읽기에 싣는다."""
    from kkumteul import compare

    def card(on_mean, on_v, off_mean, off_v, sem=0.05):
        st_ = lambda m: {"mean": m, "sd": sem * 6 ** 0.5, "sem": sem, "n": 6}  # noqa: E731
        who = lambda v: "treatment" if v.startswith("켠 쪽이 이김") else ("control" if v.startswith("끈 쪽이 이김") else "")  # noqa: E731
        return {"treatment_minus_control_r_per_1000": {
            "in_treatment_rules": st_(on_mean), "verdict_treatment_rules": on_v, "winner_treatment_rules": who(on_v),
            "in_control_rules": st_(off_mean), "verdict_control_rules": off_v, "winner_control_rules": who(off_v)}}

    near = compare._reading(card(0.20, "켠 쪽이 이김", 0.10, "가를 수 없음"))
    assert "말할 수 없다" in near and "한 쌍뿐" in near
    far = compare._reading(card(0.60, "켠 쪽이 이김", 0.00, "가를 수 없음"))
    assert "잡음보다 크다" in far and "쓰게 됐다" in far
    cross = compare._reading(card(0.60, "켠 쪽이 이김", -0.60, "끈 쪽이 이김"))
    assert "자기 규칙에 적응" in cross
    only_off = compare._reading(card(0.02, "가를 수 없음", 0.50, "켠 쪽이 이김"))
    assert "장치 없이도" in only_off, "끈 규칙에서만 켠 쪽이 이기는 경우를 '가를 수 없다'로 뭉개지 않는다"
    warned = compare._reading(card(0.60, "켠 쪽이 이김 (주의: 같은 무리끼리도 한쪽으로 기울었다)", 0.0, "가를 수 없음"))
    assert "켠 규칙 겨루기 (주의: 같은 무리끼리도" in warned
    assert compare._reading(card(0.0, "가를 수 없음", 0.0, "가를 수 없음")).startswith("가를 수 없다.")


def test_branch_turning_conflict_on_calms_attack_bias(tmp_path):
    """공격이 꺼진 세계에서는 공격 출력이 변이로 떠돈다. 갈라내며 켜면 새 개체처럼 -1로 맞추고 기록한다."""
    main = Runner.new(tmp_path / "main", small("stage3"))
    main.world.s.bout[:, ACT_ATTACK] = 0.7      # 떠돈 편향을 흉내 낸다
    main.loop(400, quiet=True)
    main.close()
    on = Runner.branch(tmp_path / "main", tmp_path / "on", 400, ["preset=conflict"], False, False)
    assert (on.world.s.bout[on.world.s.alive, ACT_ATTACK] == -1.0).all() and on.meta["genome_edits"]
    on.close()
    plain = Runner.branch(tmp_path / "main", tmp_path / "plain", 400, [], False, False)
    assert (plain.world.s.bout[plain.world.s.alive, ACT_ATTACK] > 0).any() and plain.meta["genome_edits"] == []
    plain.close()
    ctrl = Runner.branch(tmp_path / "main", tmp_path / "ctrl", 400, [], False, False, calm_attack=True)
    assert (ctrl.world.s.bout[ctrl.world.s.alive, ACT_ATTACK] == -1.0).all()
    ctrl.close()


def test_compare_reading_follows_the_winner_field_not_the_sentence():
    """읽기는 승자 필드로 가른다. 판정 문장의 낱말이 바뀌어도 읽기가 '가를 수 없다'로 떨어지지 않는다."""
    from kkumteul import compare, measure
    st_ = {"mean": 0.6, "sd": 0.1, "sem": 0.05, "n": 6}
    res = {"reps": 6, "a_extinct_runs": 0, "b_extinct_runs": 0, "both_extinct_runs": 0, "r_per_1000": st_}
    assert measure.winner(res) == "b" and measure.verdict(res, "끈 쪽이", "켠 쪽이") == "켠 쪽이 이김"
    assert measure.winner({**res, "r_per_1000": {**st_, "mean": -0.6}}) == "a"
    assert measure.winner({**res, "r_per_1000": {**st_, "mean": 0.01}}) == ""
    assert measure.winner({**res, "a_extinct_runs": 6, "r_per_1000": {**st_, "mean": -0.6}}) == "b"
    card = {"treatment_minus_control_r_per_1000": {
        "in_treatment_rules": st_, "verdict_treatment_rules": "문구가 달라져도", "winner_treatment_rules": "treatment",
        "in_control_rules": {**st_, "mean": -0.6}, "verdict_control_rules": "상관없다", "winner_control_rules": "control"}}
    assert "자기 규칙에 적응" in compare._reading(card)


def test_kind_ratio_sets_the_share_of_the_negative_kind():
    """먹이 종류의 비율은 설정(kind_ratio)이다. 0이면 전부 종류 0, 1이면 전부 종류 1이 난다."""
    for ratio, want in ((0.0, k.KIND_A), (1.0, k.KIND_B)):
        w = World(small("kinds", kind_ratio=ratio, init_agents=2, min_agents=0))
        w.run(150)
        kinds = w.s.f_kind[w.s.f_alive & (w.s.f_kind != k.KIND_CORPSE)]
        assert len(kinds) > 50 and (kinds == want).all()
    w = World(small("kinds", init_agents=2, min_agents=0))
    w.run(150)
    kinds = w.s.f_kind[w.s.f_alive & (w.s.f_kind != k.KIND_CORPSE)]
    assert 0.35 < (kinds == k.KIND_B).mean() < 0.65
    with pytest.raises(ValueError):
        small("kinds", kind_ratio=1.5).validate()
    # 거름 땅에 나는 먹이도 같은 비율을 쓴다
    w4 = World(small("stage4", "kinds", kind_ratio=1.0, init_agents=2, min_agents=0))
    w4.s.soil[:] = 5.0
    w4.s.ledger[k.L_SPAWN] += float(w4.s.soil.sum())
    w4.run(150)
    assert w4.s.counters[k.C_FERTILE] > 5
    kinds = w4.s.f_kind[w4.s.f_alive & (w4.s.f_kind != k.KIND_CORPSE)]
    assert (kinds == k.KIND_B).all()


def _two_facing(cfg):
    """모든 칸이 살아 있는 작은 세계에서 0번과 가운데 칸의 개체를 맞붙여 놓는다. 나머지는 멀리 둔다."""
    w = World(cfg)
    s = w.s
    n = cfg.max_agents
    assert int(s.alive.sum()) == n
    i, j = 0, n // 2
    for q in range(n):
        s.x[q], s.y[q] = 20.0 + 25.0 * q, 200.0
    s.x[i], s.y[i] = 100.0, 100.0
    s.x[j], s.y[j] = 101.5, 100.0
    k.build_grid(s.x, s.y, s.alive, 1.0 / cfg.cell, cfg.grid_w, cfg.grid_h, s.a_start, s.a_items, s.cursor)
    s.act[:] = 0
    s.touch[:] = 0
    return w, i, j


def test_death_by_attack_is_counted_apart_from_starving():
    cfg = small("stage3", "conflict", max_agents=8, init_agents=8, min_agents=0, food_inflow=0.0)
    w, i, j = _two_facing(cfg)
    s = w.s
    s.act[i, ACT_ATTACK] = 1.0
    s.energy[j] = 0.05                  # 한 대 맞으면 바닥난다
    s.energy[3] = -0.1                  # 맞지 않고 굶은 개체
    st.attack(0, s, w.fs, w.isp)
    assert s.energy[j] == 0.0 and s.touch[j] == k.TOUCH_HIT
    nd = k.process_deaths(0, s, w.fp, w.ip)
    w._take_deaths(nd)
    _, d = w.drain_events()
    cause = {int(r["id"]): int(r["cause"]) for r in d}
    assert cause == {int(s.ids[j]): k.DEATH_KILLED, int(s.ids[3]): k.DEATH_STARVE}
    assert s.counters[k.C_DEATH_KILLED] == 1 and s.counters[k.C_DEATH_STARVE] == 1
    assert w.summary()["deaths_killed"] == 1


def test_attack_order_does_not_always_favor_the_low_slot():
    """서로 한 대면 쓰러지는 둘이 맞붙으면 먼저 치는 쪽이 산다. 훑는 시작 칸을 스텝마다 돌리므로 양쪽 다 이길 때가 있다."""
    cfg = small("stage3", "conflict", max_agents=8, init_agents=8, min_agents=0, food_inflow=0.0)
    w, i, j = _two_facing(cfg)
    s = w.s
    s.act[i, ACT_ATTACK] = s.act[j, ACT_ATTACK] = 1.0
    wins = {i: 0, j: 0}
    for t in range(60):
        s.energy[i] = s.energy[j] = 0.1
        st.attack(t, s, w.fs, w.isp)
        assert (s.energy[i] > 0) != (s.energy[j] > 0), "먼저 맞은 쪽은 반격하지 못한다"
        wins[i if s.energy[i] > 0 else j] += 1
    assert wins[i] > 5 and wins[j] > 5, wins
    # 같은 스텝이면 같은 순서다(재현성)
    s.energy[i] = s.energy[j] = 0.1
    st.attack(7, s, w.fs, w.isp)
    first = s.energy[i] > 0
    s.energy[i] = s.energy[j] = 0.1
    st.attack(7, s, w.fs, w.isp)
    assert (s.energy[i] > 0) == first
