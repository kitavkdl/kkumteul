"""혹독한 세계 장치 시험: 확장 칸, 갈라 켜기, 장치별 장부.

    python3 -m pytest -q tests/test_harsh.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_stages import rel_residual, small  # noqa: E402
from kkumteul import checkpoint as ckpt  # noqa: E402
from kkumteul import replay  # noqa: E402
from kkumteul import kernels as k  # noqa: E402
from kkumteul import stages as st  # noqa: E402
from kkumteul.config import ACT_CARRY, EXT_SEASON, N_EXT_IN, N_EXT_OUT, N_INTERNAL, N_STAGE_IN, N_STAGE_OUT  # noqa: E402
from kkumteul.run import Runner  # noqa: E402
from kkumteul.world import World  # noqa: E402


def test_ext_preset_sizes():
    c = small("ext")
    assert (c.n_spare_in, c.n_spare_out) == (N_STAGE_IN + N_EXT_IN, N_STAGE_OUT + N_EXT_OUT) == (26, 7)
    assert c.n_out > ACT_CARRY and c.ext_in0 == 2 * c.vision_sectors + N_INTERNAL + N_STAGE_IN


def test_harsh_device_needs_ext_size():
    with pytest.raises(ValueError, match="ext"):
        small(season_enabled=True)          # 기본 크기에 장치만 켬
    small("ext", season_enabled=True)       # 확장 크기면 된다
    with pytest.raises(ValueError, match="ext"):
        small(spoil_enabled=True)           # 부패도 마찬가지다
    for kw in (dict(season_enabled=True), dict(spoil_enabled=True)):
        with pytest.raises(ValueError, match="ext"):
            small(n_spare_in=N_STAGE_IN + N_EXT_IN, n_spare_out=N_STAGE_OUT + N_EXT_OUT - 1, **kw)   # 출력이 한 칸 모자람
        with pytest.raises(ValueError, match="ext"):
            small(n_spare_in=N_STAGE_IN + N_EXT_IN - 1, n_spare_out=N_STAGE_OUT + N_EXT_OUT, **kw)   # 입력이 한 칸 모자람
        small(n_spare_in=N_STAGE_IN + N_EXT_IN, n_spare_out=N_STAGE_OUT + N_EXT_OUT, **kw)           # 딱 맞으면 된다


def test_ext_world_gives_new_agents_carry_bias_off():
    w = World(small("ext"))
    assert (w.s.bout[w.s.alive, ACT_CARRY] == -1.0).all()


def test_default_size_has_no_carry_slot():
    c = small()
    assert c.n_out == ACT_CARRY and not c.needs_ext   # 기본 9칸이라 ACT_CARRY 칸이 없다


def test_harsh_preset_turns_devices_on():
    c = small("harsh")
    assert c.season_enabled and c.spoil_enabled and c.needs_ext


# ---- 갈라 켜기: 기본 크기 체크포인트에서 확장 크기로 ----
def _main(tmp_path, *presets):
    r = Runner.new(tmp_path / "main", small(*presets)); r.loop(400, quiet=True); r.close()


def _ckpt_at(run_dir, step):
    return dict(ckpt.list_checkpoints(run_dir / "checkpoints"))[step]


def test_branch_into_ext_from_default_size(tmp_path):
    _main(tmp_path, "stage4")
    src, _ = ckpt.load(_ckpt_at(tmp_path / "main", 400))
    r = Runner.branch(tmp_path / "main", tmp_path / "ext", 400, ["preset=ext"], False, False)
    a, n_in0, n_out0 = r.world.s.alive, src.cfg.n_in, src.cfg.n_out
    assert r.world.cfg.n_spare_in == 26 and r.world.s.Win.shape[1] == r.world.cfg.n_in
    assert np.array_equal(r.world.s.Win[a][:, :n_in0], src.s.Win[a][:, :n_in0])
    assert not r.world.s.Win[a][:, n_in0:].any() and not r.world.s.Wout[a][:, n_out0:].any()
    assert (r.world.s.bout[a, ACT_CARRY] == -1.0).all() and "bout[ACT_CARRY]=-1" in r.meta["genome_edits"]
    # 마음이 켜진 체크포인트라 평생 학습 배열이 0에서 다시 시작한다는 것이 갈래의 기록에 남는다
    assert any(e.startswith("lifetime_reset:") and "Hin" in e for e in r.meta["genome_edits"])
    r.loop(400, quiet=True); r.close()
    assert rel_residual(r.world) < 1e-9


def test_branch_ext_is_idempotent(tmp_path):
    _main(tmp_path, "ext")
    r = Runner.branch(tmp_path / "main", tmp_path / "again", 400, ["preset=ext"], False, False)
    assert (r.world.cfg.n_spare_in, r.world.cfg.n_spare_out) == (26, 7) and r.meta["genome_edits"] == []
    r.close()


def test_ext_branch_resumes_exactly(tmp_path):
    _main(tmp_path)
    r = Runner.branch(tmp_path / "main", tmp_path / "ext", 400, ["preset=ext"], False, False)
    r.loop(400, quiet=True); r.close()
    again = Runner.resume(tmp_path / "ext")
    assert again.world.state_hash() == r.world.state_hash(); again.close()


def test_harsh_branch_resumes_exactly_across_winter_and_spoilage(tmp_path):
    # 마음·사회·거름이 켜진 체크포인트에서 겨울·부패를 갈라 켜고, 중간에 닫았다 이어 돌려도 쉬지 않고 돈 것과 같다
    _main(tmp_path, "stage4")
    ov = ["preset=harsh", "season_period=400", "winter_len=200", "spoil_after=60", "spoil_rate=0.02"]
    a = Runner.branch(tmp_path / "main", tmp_path / "a", 400, ov, False, False)
    a.loop(400, quiet=True); a.close()
    a = Runner.resume(tmp_path / "a")
    a.loop(400, quiet=True)
    b = Runner.branch(tmp_path / "main", tmp_path / "b", 400, ov, False, False)
    b.loop(800, quiet=True)
    assert a.world.step_no == b.world.step_no == 1200
    assert a.world.state_hash() == b.world.state_hash()
    assert b.world.s.counters[k.C_SPOILED] > 0
    a.close(); b.close()


def test_replay_record_with_ext_override(tmp_path):
    _main(tmp_path)
    rep = replay.record(_ckpt_at(tmp_path / "main", 400), steps=50, overrides=["preset=ext"])
    assert rep is not None and not any("평생 학습" in n for n in rep.meta["notes"])   # 마음이 꺼진 실행이라 잃은 것이 없다


def test_replay_record_with_ext_override_tells_lifetime_reset(tmp_path):
    _main(tmp_path, "stage4")
    rep = replay.record(_ckpt_at(tmp_path / "main", 400), steps=20, overrides=["preset=ext"])
    assert any("평생 학습" in n and "Hin" in n for n in rep.meta["notes"])


# ---- 겨울 ----
def test_season_schedule_keeps_yearly_mean():
    w = World(small("ext", season_enabled=True))
    P, W = w.cfg.season_period, w.cfg.winter_len
    g = [w.season(t)[0] for t in range(P)]
    assert abs(np.mean(g) - 1.0) < 1e-12
    assert w.season(0)[0] > 1.0
    assert w.season(P - 1)[:2] == (pytest.approx(0.3), 1.5)      # 해의 마지막 스텝은 겨울
    assert w.season(P - W - 1)[1] == 1.0 and w.season(P - W)[:2] == (pytest.approx(0.3), 1.5)
    assert w.season(P)[0] == w.season(0)[0]                        # 해가 되풀이된다


def test_season_off_is_neutral():
    assert World(small("ext")).season(3999) == (1.0, 1.0, 0.0, 0.0)


def test_season_config_validated():
    for bad in (dict(winter_len=0), dict(winter_len=4000), dict(winter_mult=1.5), dict(winter_basal_mult=0.9)):
        with pytest.raises(ValueError):
            small("ext", season_enabled=True, **bad)


def test_winter_cuts_grass_and_raises_basal():
    on = World(small("ext", season_enabled=True, season_period=400, winter_len=200))
    off = World(small("ext"))
    for w in (on, off):
        w.run(400)
    assert on.s.ledger[k.L_BASAL] > off.s.ledger[k.L_BASAL]
    assert rel_residual(on) < 1e-9
    # 겨울만 도는 구간(200~400)에서는 들어오는 풀도 준다
    cold = World(small("ext", season_enabled=True, season_period=400, winter_len=200))
    base = World(small("ext"))
    for w in (cold, base):
        w.step_no = 200
        w.run(200)
    assert cold.s.ledger[k.L_SPAWN] < 0.5 * base.s.ledger[k.L_SPAWN]


def test_autumn_off_keeps_season_values():
    w = World(small("harsh"))
    assert w.season(100)[0] == pytest.approx((4000 - 1200 * 0.3) / 2800)   # 1.3, 가을 없는 겨울과 같다
    assert not w.in_autumn(2500)


def test_autumn_mast_keeps_year_mean():
    w = World(small("harsh2"))
    P = w.cfg.season_period
    g = [w.season(t)[0] for t in range(P)]
    assert sum(g) / P == pytest.approx(1.0, rel=1e-12)
    assert w.season(2000)[0] == pytest.approx(2.0) and w.in_autumn(2000)    # 2000..2799 가을
    assert not w.in_autumn(1999) and not w.in_autumn(2800)
    assert w.season(100)[0] == pytest.approx(1.02)
    assert w.season(2000)[1] == 1.0                                          # 가을 기초 대사는 그대로


def test_autumn_validate():
    with pytest.raises(ValueError):
        small("harsh", autumn_len=2800)                    # 가을 + 겨울이 해 전체
    with pytest.raises(ValueError):
        small("harsh", autumn_len=800, autumn_mult=0.5)    # 가을 배수는 1 이상
    with pytest.raises(ValueError):
        small("harsh", autumn_len=800, autumn_mult=5.0)    # 나머지 철 배수가 음수
    with pytest.raises(ValueError):
        small("ext", autumn_len=800, autumn_mult=2.0)      # 겨울이 꺼진 세계에 가을만


def test_autumn_ledger_closes():
    # 세 해(1,800스텝)를 돌려도 장부가 닫힌다. 가을 150스텝(300~449)은 나머지 철(0~299)보다 풀이 많이 들어온다
    kw = dict(season_period=600, winter_len=150, autumn_len=150)
    w = World(small("harsh2", **kw))
    w.run(1800)
    assert rel_residual(w) < 1e-9
    fall = World(small("harsh2", **kw))
    rest = World(small("harsh2", **kw))
    fall.step_no = 300
    for x in (fall, rest):
        x.run(150)
    assert fall.in_autumn(300) and not rest.in_autumn(0)
    assert fall.s.ledger[k.L_SPAWN] > 1.5 * rest.s.ledger[k.L_SPAWN]


def _one_step_basal(w, t):
    """스텝 t를 한 번 돌리고, 그 스텝에 기초 대사로 장부에 오른 양과 move 직전의 산 개체 수를 돌려준다
    (move는 출생·사망보다 앞이라 개체 수는 스텝 시작 때와 같다)."""
    w.step_no = t
    n = int(w.s.alive.sum())
    before = float(w.s.ledger[k.L_BASAL])
    w.step()
    return float(w.s.ledger[k.L_BASAL]) - before, n


def test_winter_basal_multiplier_reaches_move():
    # 겨울 한 스텝의 기초 대사는 개체당 basal_cost * winter_basal_mult, 봄은 basal_cost 그대로다
    # (개체 수가 달라 저절로 커지는 장부 합계로는 가를 수 없어 개체당 값을 본다)
    cfg = small("ext", season_enabled=True, season_period=400, winter_len=200)
    spring, n_s = _one_step_basal(World(cfg), 100)
    winter, n_w = _one_step_basal(World(cfg), 399)
    assert spring == pytest.approx(n_s * cfg.basal_cost, rel=1e-12)
    assert winter == pytest.approx(n_w * cfg.basal_cost * cfg.winter_basal_mult, rel=1e-12)
    assert cfg.winter_basal_mult > 1.0


def test_winter_basal_multiplier_reaches_sleep_refund():
    # 자는 개체는 기초 대사의 (1 - sleep_basal)을 돌려받는다. 겨울에는 올라간 기초 대사 기준으로 돌려받는다
    cfg = small("stage2", "ext", season_enabled=True, season_period=400, winter_len=200)
    w = World(cfg)
    asleep = np.flatnonzero(w.s.alive)[::2]
    w.s.asleep[asleep] = 1.0
    w.s.fatigue[:] = 0.0
    w.s.fatigue[asleep] = 50.0            # 한 스텝에 깨지 않게
    winter, n = _one_step_basal(w, 399)
    m = cfg.basal_cost * cfg.winter_basal_mult
    assert 0 < len(asleep) < n
    assert winter == pytest.approx(n * m - len(asleep) * m * (1.0 - cfg.sleep_basal), rel=1e-12)


def test_season_input_is_written():
    w = World(small("ext", season_enabled=True))
    w.run(5)
    a = w.s.alive
    c = w.cfg.ext_in0 + EXT_SEASON
    _, _, sn, cs = w.season(w.step_no - 1)
    assert np.allclose(w.s.obs[a, c], sn) and np.allclose(w.s.obs[a, c + 1], cs)


def test_winter_with_niche_conserves_energy():
    w = World(small("stage4", "ext", season_enabled=True, season_period=400, winter_len=200))
    w.run(1200)
    w.check_sane()
    assert rel_residual(w) < 1e-9


@pytest.mark.parametrize("mult", [0.3, 1.0])
def test_niche_share_follows_season_mult(mult):
    # 거름 몫이 겨울 배수를 따르는지: niche_world를 한 번 불러 budget 이동량을 센다.
    # (장부 잔차에는 budget이 들어가지 않아 위 보존 시험만으로는 이것을 가리지 못한다)
    w = World(small("stage4", "ext", season_enabled=True))
    w.s.soil[:] = 0.0
    w.s.soil[0] = 0.1 * w.cfg.soil_ref          # 몫이 먹이 한 알(1.0)보다 작게 잡아 먹이가 놓이지 않게 한다
    w.s.budget[:] = 0.0
    st.niche_world(0, w.s, w.fs, w.isp, mult)
    frac = float(w.s.soil.sum()) / w.cfg.soil_ref
    share = w.cfg.food_inflow * mult * w.cfg.fert_frac * frac
    assert share < w.cfg.food_energy
    assert w.s.budget[1] == pytest.approx(share, rel=1e-6)
    assert w.s.budget[0] == pytest.approx(-share, rel=1e-6)


@pytest.mark.parametrize("t", [100, 399])
def test_world_step_passes_season_mult_to_niche_world(t):
    # World.step이 그 스텝의 풀 배수를 niche_world에 넘기는지(위 시험은 niche_world만 직접 부른다)
    w = World(small("stage4", "ext", season_enabled=True, season_period=400, winter_len=200))
    w.step_no = t
    w.s.soil[:] = 0.0
    w.s.soil[0] = 0.1 * w.cfg.soil_ref
    w.s.budget[:] = 0.0
    mult = w.season(t)[0]
    frac = 0.1 * (1.0 - w.cfg.soil_decay)                    # niche_world가 거름을 줄인 뒤의 몫
    share = w.cfg.food_inflow * mult * w.cfg.fert_frac * frac
    assert share < w.cfg.food_energy and mult != pytest.approx(1.0)
    w.step()
    assert w.s.budget[1] == pytest.approx(share, rel=1e-9)


# ---- 부패 ----
def _empty_world(*presets, **kw):
    """먹는 개체가 없는 세계(init_agents=0, min_agents=0). 먹이는 시험이 직접 놓는다."""
    return World(small(*presets, init_agents=0, min_agents=0, food_inflow=0.0, **kw))


def _place(w, e=1.0, born=0, kind=k.KIND_A, x=10.0, y=10.0):
    """먹이 하나를 놓고 칸 번호를 돌려준다. 에너지 장부에도 들어온 것으로 적는다."""
    before = w.s.f_alive.copy()
    assert k._place_food(w.s, x, y, e, kind, born, int(w.cfg.field_cap))
    w.s.ledger[k.L_SPAWN] += e
    return int(np.flatnonzero(w.s.f_alive & ~before)[0])


def test_food_spoils_after_delay_with_half_life():
    w = _empty_world("ext", spoil_enabled=True)
    j = _place(w)
    w.run(400)
    assert w.s.f_energy[j] == pytest.approx(1.0)                       # 난 지 400스텝까지는 그대로
    w.run(600)
    assert w.s.f_energy[j] == pytest.approx((1 - 0.00115) ** 600, rel=1e-9)
    assert 0.49 < w.s.f_energy[j] < 0.51                                # 반감기 약 600
    w.run(2500)
    assert not w.s.f_alive[j] and w.s.counters[k.C_SPOILED] == 1 and w.s.counters[k.C_NFOOD] == 0
    assert rel_residual(w) < 1e-9
    assert w.s.ledger[k.L_LOST] == pytest.approx(1.0, rel=1e-9)         # 줄어든 양과 남은 것이 모두 L_LOST로


def test_spoil_off_leaves_food_alone():
    w = _empty_world("ext")
    j = _place(w)
    w.run(1500)
    assert w.s.f_energy[j] == 1.0 and w.s.counters[k.C_SPOILED] == 0


def test_corpse_spoils_too():
    w = _empty_world("ext", spoil_enabled=True)
    j = _place(w, born=-1000, kind=k.KIND_CORPSE)
    w.run(1)
    assert w.s.f_energy[j] == pytest.approx(1 - 0.00115, rel=1e-12)


def test_drop_food_slot_clears_slot_and_count():
    w = _empty_world("ext")
    j = _place(w)
    k._drop_food_slot(w.s, j, int(w.cfg.field_cap))
    assert not w.s.f_alive[j] and w.s.counters[k.C_NFOOD] == 0


@pytest.mark.parametrize("t0,mult", [(0, 1.0), (199, 1.0), (200, 0.2), (399, 0.2)])
def test_spoil_slows_in_winter(t0, mult):
    w = _empty_world("ext", spoil_enabled=True, season_enabled=True, season_period=400, winter_len=200)
    j = _place(w, born=-1000)
    w.step_no = t0
    w.run(1)
    assert w.s.f_energy[j] == pytest.approx(1.0 - 0.00115 * mult, rel=1e-12)   # 겨울(200~399)에는 0.2배
    assert rel_residual(w) < 1e-9


def test_spoil_without_season_ignores_winter_mult():
    w = _empty_world("ext", spoil_enabled=True, season_period=400, winter_len=200)
    j = _place(w, born=-1000)
    w.step_no = 300
    w.run(1)
    assert w.s.f_energy[j] == pytest.approx(1.0 - 0.00115, rel=1e-12)           # 겨울이 꺼져 있으면 늘 1배


def test_spoil_config_validated():
    for bad in (dict(spoil_after=-1), dict(spoil_rate=1.5), dict(spoil_rate=-0.1),
                dict(spoil_min_frac=1.0), dict(spoil_winter_mult=1.5)):
        with pytest.raises(ValueError):
            small("ext", spoil_enabled=True, **bad)


def test_spoil_world_runs_and_conserves_energy():
    w = World(small("stage4", "ext", spoil_enabled=True))
    w.run(1500)
    w.check_sane()
    assert w.summary()["pop"] > 20 and rel_residual(w) < 1e-9


def test_fast_spoil_drops_food_and_conserves_energy():
    # 기본값으로는 1500스텝 안에 사라지는 먹이가 없다. 빠르게 썩혀 칸이 비워지는 길까지 장부로 본다
    w = World(small("stage4", "ext", spoil_enabled=True, spoil_after=100, spoil_rate=0.01))
    w.run(1500)
    w.check_sane()
    assert w.s.counters[k.C_SPOILED] > 20 and rel_residual(w) < 1e-9
    assert w.s.counters[k.C_NFOOD] == int(w.s.f_alive.sum())


def test_spoil_with_cap_replace_conserves_energy():
    w = World(small("ext", spoil_enabled=True, food_cap_replace=True))
    w.run(1500)
    assert rel_residual(w) < 1e-9
    # 상한에 닿은 세계에서도 마찬가지다
    w = World(small("ext", spoil_enabled=True, food_cap_replace=True, food_cap=150))
    w.run(1500)
    assert w.s.counters[k.C_FOOD_REPLACED] > 0 and rel_residual(w) < 1e-9


def test_spoil_and_cap_replace_in_one_step_both_go_to_lost():
    # 상한(4)이 찬 세계에 새 먹이가 하나 나는 스텝: 가장 오래된 하나는 밀려나고, 하한 아래로 내려간 둘은 썩어 사라진다
    w = World(small("ext", spoil_enabled=True, food_cap_replace=True, food_cap=4, init_agents=0, min_agents=0,
                    food_inflow=1.0))
    old = [_place(w, born=-1000), _place(w)]
    rotten = [_place(w, e=0.05001, born=-400), _place(w, e=0.05001, born=-400)]
    w.run(1)
    c = w.s.counters
    assert c[k.C_FOOD_REPLACED] == 1 and c[k.C_SPOILED] == 2 and c[k.C_NFOOD] == int(w.s.f_alive.sum()) == 2
    assert not w.s.f_alive[old[0]] and not w.s.f_alive[rotten[0]] and not w.s.f_alive[rotten[1]]
    assert w.s.f_alive[old[1]]
    assert w.s.ledger[k.L_LOST] == pytest.approx(1.0 + 2 * 0.05001, rel=1e-9)   # 밀려난 하나 + 썩은 둘의 전부
    assert rel_residual(w) < 1e-9


@pytest.mark.parametrize("year", [None, 150])
def test_replay_tells_spoiled_food_from_eaten(year):
    # year가 있으면 겨울이 켜진 짧은 해를 지난다
    kw = dict(season_enabled=True, season_period=year, winter_len=year // 2) if year else {}
    w = World(small("ext", spoil_enabled=True, spoil_after=60, spoil_rate=0.02, **kw))
    w.run(300)
    c0 = w.s.counters.copy()
    rec = replay.Recorder(w, every=5)
    rec.advance(300)
    rep = rec.finish()
    dc = w.s.counters - c0
    eaten, spoiled = int(dc[k.C_EATEN]), int(dc[k.C_SPOILED])
    assert eaten > 50 and spoiled > 5
    assert rep.seen["spoiled"] == spoiled
    assert int((rep.food["why"] == replay.WHY_SPOILED).sum()) == spoiled
    assert rep.seen["unknown"] <= 0.02 * eaten


@pytest.mark.parametrize("t,why", [(399, "good"), (100, "spoiled")])
def test_replay_uses_winter_spoil_speed_to_tell_spoiled_from_eaten(t, why):
    # 에너지가 하한(0.05) 바로 위인 오래된 먹이를 개체가 먹는 스텝: 겨울(느린 부패)에는 하한 위에 남아 먹힌 것이고,
    # 봄(보통 부패)에는 먹기 전에 하한 아래로 내려가 썩은 것이다. 다시 보기가 겨울의 부패 속도로 가려야 맞다
    w = World(small("ext", spoil_enabled=True, season_enabled=True, season_period=400, winter_len=200,
                    spoil_rate=0.02, init_agents=1, min_agents=0, food_inflow=0.0))
    w.step_no = t
    a = int(np.flatnonzero(w.s.alive)[0])
    w.s.x[a], w.s.y[a] = 10.0, 10.0
    _place(w, e=0.0505, born=-1000)
    rec = replay.Recorder(w, every=1)
    rec.advance(1)
    rep = rec.finish()
    assert rep.seen["good"] + rep.seen["spoiled"] + rep.seen["unknown"] == 1
    assert rep.seen[why] == 1


# ---- 공유 저장 틀: 해시 기본값 규칙, 번호표, 확장 세계 계수기 길이 ----
# 장치를 넣기 전 코드(0945b10)가 낸 300스텝 해시. 확장 크기 세계에 새 필드·늘린 계수기가 들어와도 장치를 끈 동안은 같아야 한다.
PINNED_EXT_HASH_300 = {
    ("ext",): "f99625e844a559b3d3518c2dc18ca85fb83642f4bc71046d3c74afdea5ec3fdf",
    ("harsh",): "edfcfe6ec79c8676c656d64cf4a01145c9a9b245e3cad99c9ffb10d9ec53a3ef",
    ("harsh", "stage4", "conflict", "kinds"): "09def17063df81639d606807ce530c6de55df7e87e8f00f6ac95c0f6dc990ed7",
}


@pytest.mark.parametrize("presets", list(PINNED_EXT_HASH_300))
def test_ext_worlds_keep_pinned_hash(presets):
    w = World(small(*presets)); w.run(300)
    assert w.state_hash() == PINNED_EXT_HASH_300[presets]


def test_counter_and_budget_length_follow_ext_size():
    base = World(small())
    assert len(base.s.counters) == k.N_COUNTERS_BASE and len(base.s.budget) == k.N_BUDGET_BASE   # 기본 크기
    w = World(small("ext"))
    assert len(w.s.counters) == k.N_COUNTERS_EXT and len(w.s.budget) == k.N_BUDGET_EXT


def test_ext_hash_ignores_zero_tail():
    w = World(small("harsh")); w.run(50)
    h = w.state_hash()
    w.s = w.s._replace(counters=w.s.counters[:24].copy(), budget=w.s.budget[:2].copy())
    assert w.state_hash() == h          # 꼬리가 0이면 옛 길이 해시와 같다
    w2 = World(small("harsh")); w2.run(50)
    w2.s.counters[k.C_RESCUES] = 1
    assert w2.state_hash() != h         # 꼬리에 값이 있으면 들어간다


def test_hash_skips_default_valued_field_but_not_changed_one():
    w = World(small("harsh")); w.run(20)
    h = w.state_hash()
    w.s.elig_in[...] = 0
    assert w.state_hash() == h          # 기본값 그대로면 건너뛴다
    a = np.flatnonzero(w.s.alive)[0]
    w.s.elig_in[a].flat[0] = 1.0
    assert w.state_hash() != h


def test_old_ext_checkpoint_loads_exact(tmp_path):
    # 확장 세계를 저장한 뒤 체크포인트의 counters를 24칸, budget을 2칸으로 잘라 옛 체크포인트처럼 만든다.
    # 새 코드로 읽으면 채워 읽기가 되고, load_arrays가 True(exact)를 돌려주며, checkpoint.load의 저장 해시 검증을 통과한다.
    w = World(small("harsh")); w.run(60)
    path = ckpt.save(w, tmp_path)
    with np.load(path) as z:
        arrays = {n: z[n] for n in z.files}
    arrays["counters"] = arrays["counters"][:24].copy()
    arrays["budget"] = arrays["budget"][:2].copy()
    np.savez(path, **arrays)
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text())
    meta["npz_sha256"] = ckpt.file_sha256(path)
    meta_path.write_text(json.dumps(meta))
    w2, _ = ckpt.load(path)                       # 저장 해시 검증을 켠 채로 읽는다
    assert len(w2.s.counters) == k.N_COUNTERS_EXT and len(w2.s.budget) == k.N_BUDGET_EXT
    assert w2.state_hash() == w.state_hash()
    w3 = World(small("harsh"), populate=False)
    assert w3.load_arrays(arrays) is True


# ---- 들기: 물건 칸과 장부 ----
def hold(w, i, j):
    """개체 i가 물건 j를 든 꼴로 맞춘다. 들기처럼 들판 알이면 들판 수를 하나 줄이고 f_dropped를 켠다."""
    s = w.s
    if s.f_dropped[j] == 0 and s.f_kind[j] != k.KIND_CORPSE:
        s.counters[k.C_NFIELD] -= 1
    s.a_held[i], s.f_holder[j], s.f_dropped[j] = j, i, 1


def unhold(w, i):
    """개체 i가 든 물건을 놓는다(f_dropped는 1로 남는다)."""
    s = w.s
    j = int(s.a_held[i])
    s.f_holder[j] = -1
    s.a_held[i] = -1


def _place_stored(w, e=1.0, born=0, kind=k.KIND_A, x=100.0, y=100.0):
    """내려놓은 물건(들판을 떠난 적 있는 알) 하나를 놓고 칸 번호를 돌려준다."""
    j = _place(w, e, born, kind, x, y)
    if w.s.f_kind[j] != k.KIND_CORPSE:
        w.s.counters[k.C_NFIELD] -= 1
    w.s.f_dropped[j] = 1
    return j


def _n_field(s):
    return int((s.f_alive & (s.f_dropped == 0) & (s.f_kind != k.KIND_CORPSE)).sum())


def _check_hands(s):
    """손과 물건이 서로를 가리키는가."""
    held = np.flatnonzero(s.a_held >= 0)
    assert s.alive[held].all() and (s.f_holder[s.a_held[held]] == held).all() and s.f_alive[s.a_held[held]].all()
    items = np.flatnonzero(s.f_holder >= 0)
    assert (s.a_held[s.f_holder[items]] == items).all()
    assert (s.f_dropped[items] == 1).all()


def _eat_once(w):
    """먹이 격자를 세우고 eat만 한 번 부른다(개체가 움직이지 않게)."""
    s, fp, ip = w.s, w.fp, w.ip
    k.build_grid(s.fx, s.fy, s.f_alive, fp[k.P_INV_CELL], int(ip[k.Q_GW]), int(ip[k.Q_GH]),
                 s.f_start, s.f_items, s.cursor)
    k.eat(w.step_no, s, fp, ip)


def _shuffle_hands(w, rng):
    """든 손의 절반쯤을 풀고, 빈손 개체 몇에게 들지 않은 먹이를 쥐여 준다."""
    s = w.s
    for i in np.flatnonzero(s.a_held >= 0):
        if rng.random() < 0.5:
            unhold(w, int(i))
    free_hands = np.flatnonzero(s.alive & (s.a_held < 0))
    items = np.flatnonzero(s.f_alive & (s.f_holder < 0))
    n = min(10, len(free_hands), len(items))
    for i, j in zip(rng.choice(free_hands, n, replace=False), rng.choice(items, n, replace=False)):
        hold(w, int(i), int(j))


def test_carry_needs_ext_size():
    with pytest.raises(ValueError, match="ext"):
        small(carry_enabled=True)
    c = small("ext", carry_enabled=True)
    assert c.needs_ext and c.field_cap and not small("ext").field_cap


def test_carry_fields_default_and_hash_unchanged():
    w = World(small("ext"))
    assert (w.s.a_held == -1).all() and (w.s.f_holder == -1).all() and (w.s.f_dropped == 0).all()
    w.run(20); h = w.state_hash()
    i = int(np.flatnonzero(w.s.alive)[0])
    w.s.a_held[i] = 5
    assert w.state_hash() != h          # 기본값이 아니면 해시에 들어간다
    w.s.a_held[i] = -1
    assert w.state_hash() == h          # 모두 기본값이면 빠진다


def test_old_checkpoint_without_carry_fields_loads_exact(tmp_path):
    # 확장 세계를 저장한 뒤 체크포인트에서 a_held, f_holder, f_dropped를 빼고 읽는다 →
    # 기본값으로 채워지고 load_arrays가 True, 저장 해시 검증 통과
    w = World(small("harsh")); w.run(60)
    path = ckpt.save(w, tmp_path)
    with np.load(path) as z:
        arrays = {n: z[n] for n in z.files}
    for name in ("a_held", "f_holder", "f_dropped"):
        del arrays[name]
    np.savez(path, **arrays)
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text())
    meta["npz_sha256"] = ckpt.file_sha256(path)
    meta_path.write_text(json.dumps(meta))
    w2, _ = ckpt.load(path)                       # 저장 해시 검증을 켠 채로 읽는다
    assert (w2.s.a_held == -1).all() and (w2.s.f_holder == -1).all() and (w2.s.f_dropped == 0).all()
    assert w2.state_hash() == w.state_hash()
    w3 = World(small("harsh"), populate=False)
    assert w3.load_arrays(arrays) is True
    assert (w3.s.a_held == -1).all() and (w3.s.f_holder == -1).all()


def test_field_counter_matches_every_step():
    # carry 켠 ext 세계 + stage4(거름 niche_world) + corpse 켬 + spoil_enabled + food_cap_replace + 작은 food_cap,
    # 각본 없이 매 스텝 strike를 가끔 손으로 부른다(burn_food). 400스텝 동안 매 스텝
    # C_NFIELD == (f_alive & f_dropped==0 & f_kind != KIND_CORPSE).sum(), C_NFOOD == f_alive.sum(). 잔차 < 1e-9
    w = World(small("stage4", "ext", carry_enabled=True, corpse_to_food=True, spoil_enabled=True, spoil_after=60,
                    spoil_rate=0.02, food_cap_replace=True, food_cap=120, max_age=300))
    w.s.soil[::7] = 2.0                         # 처음 개체는 거름을 쌓지 않으므로 거름 땅을 깔아 거름 먹이가 나게 한다
    w.s.ledger[k.L_INJECT] += float(w.s.soil.sum())
    rng = np.random.default_rng(5)
    for t in range(400):
        if t % 20 == 0:
            _shuffle_hands(w, rng)
        if t % 37 == 11:
            w.strike(rng.random() * 256.0, rng.random() * 256.0, 40.0, kill=0.5, burn_food=True, key=t)
        w.step()
        s = w.s
        assert s.counters[k.C_NFIELD] == _n_field(s), t
        assert s.counters[k.C_NFOOD] == int(s.f_alive.sum()), t
        _check_hands(s)
        assert (s.a_held[~s.alive] == -1).all()
    c = w.s.counters
    # 시험이 지나간 길: 상한 교체, 부패, 늙어 죽음(사체), 운석, 거름 먹이, 내려놓은 물건
    assert c[k.C_FOOD_REPLACED] > 0 and c[k.C_SPOILED] > 0 and c[k.C_DEATH_AGE] > 0 and c[k.C_DEATH_DISASTER] > 0
    assert c[k.C_FERTILE] > 0
    assert (w.s.f_alive & (w.s.f_dropped == 1)).any()
    assert rel_residual(w) < 1e-9


def test_eat_releases_and_counts_field():
    # 들판 알을 먹으면 C_NFIELD가 하나 줄고, 내려놓은 알(f_dropped=1)을 먹으면 C_NFIELD는 그대로
    w = World(small("ext", carry_enabled=True, init_agents=2, min_agents=0, food_inflow=0.0))
    s = w.s
    a, b = np.flatnonzero(s.alive)[:2]
    s.ledger[k.L_LOST] += s.energy[a] - 1.0     # 시험이 깎는 에너지를 장부에도 적는다
    s.x[a], s.y[a], s.energy[a] = 10.0, 10.0, 1.0
    s.x[b], s.y[b] = 200.0, 200.0
    field = _place(w, x=10.0, y=10.0)
    _place(w, x=150.0, y=150.0)                 # 멀리 있는 들판 알 하나(먹히지 않는다)
    assert s.counters[k.C_NFIELD] == 2 and s.counters[k.C_NFOOD] == 2
    _eat_once(w)
    assert not s.f_alive[field] and s.counters[k.C_NFIELD] == 1 and s.counters[k.C_NFOOD] == 1
    stored = _place_stored(w, x=10.0, y=10.0)
    assert s.counters[k.C_NFIELD] == 1 and s.counters[k.C_NFOOD] == 2
    _eat_once(w)
    assert not s.f_alive[stored] and s.counters[k.C_NFIELD] == 1 and s.counters[k.C_NFOOD] == 1
    # 든 물건의 칸을 비우면 손도 풀린다
    j = _place(w, x=200.0, y=200.0)
    hold(w, b, j)
    k._drop_food_slot(s, j, 1)
    assert s.a_held[b] == -1 and s.f_holder[j] == -1 and s.counters[k.C_NFIELD] == 1
    s.ledger[k.L_LOST] += s.f_energy[j]
    assert rel_residual(w) < 1e-9


def test_held_item_is_not_eaten():
    # 든 알 곁에 배고픈 다른 개체를 두고 한 스텝 → 알이 그대로 f_alive, f_holder 그대로
    w = World(small("ext", carry_enabled=True, init_agents=2, min_agents=0, food_inflow=0.0))
    s = w.s
    a, b = np.flatnonzero(s.alive)[:2]
    for i in (a, b):
        s.x[i], s.y[i], s.energy[i] = 10.0, 10.0, 1.0
    j = _place(w, x=10.0, y=10.0)
    hold(w, a, j)
    _eat_once(w)
    assert s.f_alive[j] and s.f_holder[j] == a and s.a_held[a] == j
    assert s.ate[a] == 0.0 and s.ate[b] == 0.0 and s.counters[k.C_EATEN] == 0


def test_death_drops_item_and_slot_reuse_is_empty_handed():
    # 물건을 든 개체를 굶겨 죽인다 → 물건 f_holder == -1, f_dropped == 1, 자리 = 죽은 자리, f_alive
    # 이어 그 칸에 새 개체가 나면(process_births/reseed) a_held == -1. strike로 죽인 경우도 같다
    w = World(small("ext", carry_enabled=True, init_agents=5, min_agents=5, food_inflow=0.0))
    s = w.s
    assert s.alive[:5].all()
    j = _place(w, x=50.0, y=50.0)
    hold(w, 0, j)
    s.ledger[k.L_LOST] += s.energy[0] + 0.5     # 시험이 깎는 에너지를 장부에도 적는다
    s.x[0], s.y[0], s.energy[0] = 120.0, 80.0, -0.5
    nd = k.process_deaths(w.step_no, s, w.fp, w.ip)
    assert nd == 1 and not s.alive[0]
    assert s.f_alive[j] and s.f_holder[j] == -1 and s.f_dropped[j] == 1 and s.a_held[0] == -1
    assert (s.fx[j], s.fy[j]) == (120.0, 80.0)
    assert s.counters[k.C_NFIELD] == 0          # 내려놓은 알은 들판으로 돌아가지 않는다
    s.a_held[0] = 7                             # 빈 칸에 남은 값이 새 개체에게 넘어가지 않는지 보려고 일부러 더럽힌다
    assert k.reseed(w.step_no, s, w.fp, w.ip, 0) == 1
    assert s.alive[0] and s.a_held[0] == -1
    # strike로 죽인 경우
    j2 = _place(w, x=60.0, y=60.0)
    hold(w, 1, j2)
    s.x[1], s.y[1] = 30.0, 200.0
    others = np.flatnonzero(s.alive)
    assert (np.hypot(s.x[others] - 30.0, s.y[others] - 200.0)[others != 1] > 1.0).all()
    assert w.strike(30.0, 200.0, 0.5, kill=1.0, burn_food=False) == 1
    assert not s.alive[1] and s.a_held[1] == -1
    assert s.f_alive[j2] and s.f_holder[j2] == -1 and s.f_dropped[j2] == 1 and (s.fx[j2], s.fy[j2]) == (30.0, 200.0)
    assert rel_residual(w) < 1e-9


def test_cap_counts_field_only():
    # food_cap=50, 들판 알 50 + 내려놓은 알 30(C_NFOOD 80): 다음 유입은 놓이지 않고 L_DISCARD로 간다(C_NFIELD == cap).
    # 들판 알 하나를 먹이면 다음 유입 한 알이 놓인다. 내려놓은 알 30은 상한에 세지 않는다
    # (반대로 들판 알 20 + 내려놓은 알 30이면 C_NFOOD가 cap을 넘어도 유입이 놓인다). 장부 잔차 < 1e-9
    def inflow(w, n):
        """유입 n알어치를 넣고 버려진 양을 돌려준다."""
        before = float(w.s.ledger[k.L_DISCARD])
        w.s.budget[0] = n * w.cfg.food_energy
        k.spawn_food(w.step_no, w.s, w.fp, w.ip, 0.0)
        return float(w.s.ledger[k.L_DISCARD]) - before

    w = World(small("ext", carry_enabled=True, food_cap=50, init_agents=1, min_agents=0, food_inflow=0.0))
    s = w.s
    a = int(np.flatnonzero(s.alive)[0])
    s.x[a], s.y[a] = 200.0, 200.0
    for _ in range(50):
        _place(w, x=10.0, y=10.0)
    for _ in range(30):
        _place_stored(w)
    assert s.counters[k.C_NFIELD] == 50 and s.counters[k.C_NFOOD] == 80
    assert inflow(w, 1) == pytest.approx(1.0)
    assert s.counters[k.C_NFIELD] == 50 and s.counters[k.C_NFOOD] == 80
    s.ledger[k.L_INJECT] += w.cfg.e_max - 0.5 - s.energy[a]        # 시험이 더하는 에너지를 장부에도 적는다
    s.x[a], s.y[a], s.energy[a] = 10.0, 10.0, w.cfg.e_max - 0.5     # 들판 알 한 알을 먹으면 배가 찬다
    _eat_once(w)
    assert s.counters[k.C_EATEN] == 1 and s.counters[k.C_NFIELD] == 49 and s.counters[k.C_NFOOD] == 79
    assert inflow(w, 1) == 0.0
    assert s.counters[k.C_NFIELD] == 50 and s.counters[k.C_NFOOD] == 80
    assert inflow(w, 1) == pytest.approx(1.0)
    assert rel_residual(w) < 1e-9

    w = World(small("ext", carry_enabled=True, food_cap=50, init_agents=0, min_agents=0, food_inflow=0.0))
    s = w.s
    for _ in range(20):
        _place(w, x=10.0, y=10.0)
    for _ in range(30):
        _place_stored(w)
    assert s.counters[k.C_NFOOD] == 50 and s.counters[k.C_NFIELD] == 20
    assert inflow(w, 10) == 0.0
    assert s.counters[k.C_NFOOD] == 60 > w.cfg.food_cap and s.counters[k.C_NFIELD] == 30
    assert rel_residual(w) < 1e-9


def test_cap_replace_skips_stored_items():
    # food_cap_replace=True + spoil_enabled: 가장 오래된 것이 내려놓은 알·든 알이어도 교체는 들판 알을 고른다
    w = World(small("ext", carry_enabled=True, food_cap_replace=True, spoil_enabled=True, food_cap=4,
                    init_agents=1, min_agents=0, food_inflow=0.0))
    s = w.s
    a = int(np.flatnonzero(s.alive)[0])
    stored = _place_stored(w, born=-1000)
    held = _place(w, born=-900)
    hold(w, a, held)
    field = [_place(w, born=b) for b in (-500, -400, -300, -200)]
    assert s.counters[k.C_NFIELD] == 4 and s.counters[k.C_NFOOD] == 6
    s.budget[0] = w.cfg.food_energy
    k.spawn_food(w.step_no, s, w.fp, w.ip, 0.0)
    assert s.counters[k.C_FOOD_REPLACED] == 1 and s.ledger[k.L_DISCARD] == 0.0
    assert s.f_alive[stored] and s.f_alive[held] and s.f_holder[held] == a
    assert not s.f_alive[field[0]] and s.f_alive[field[1:]].all()
    assert s.counters[k.C_NFIELD] == 4 == _n_field(s) and s.counters[k.C_NFOOD] == 6
    assert rel_residual(w) < 1e-9


def test_store_room_evicts_oldest_unheld():
    # 들판 아닌 물건이 max_agents개면 store_room이 가장 오래된 "들지 않은" 것(내려놓은 알이나 사체)을 L_LOST로 치우고
    # C_STORE_EVICTED += 1. 들판 아닌 물건이 모두 든 것이면 False. 사체가 칸을 채운 경우도 사체를 치워 자리를 낸다
    def world(n_agents):
        return World(small("ext", carry_enabled=True, max_agents=10, init_agents=n_agents, min_agents=0,
                           food_inflow=0.0))

    # 든 알(가장 오래됨) 하나 + 내려놓은 알 여덟 + 사체 하나 = 10
    w = world(1)
    s = w.s
    a = int(np.flatnonzero(s.alive)[0])
    held = _place(w, born=-100)
    hold(w, a, held)
    stored = [_place_stored(w, born=b) for b in (-30, -50, -50, -20, -10, -5, -4, -3)]
    corpse = _place(w, born=-40, kind=k.KIND_CORPSE)
    field = _place(w, born=-1000)                   # 들판 알은 저장 칸에 세지 않는다
    assert s.counters[k.C_NFOOD] - s.counters[k.C_NFIELD] == 10
    lost0 = float(s.ledger[k.L_LOST])
    assert k.store_room(s, 1)
    assert not s.f_alive[stored[1]] and s.f_alive[stored[2]]     # 같은 스텝이면 낮은 칸
    assert s.f_alive[held] and s.f_alive[corpse] and s.f_alive[field]
    assert s.counters[k.C_STORE_EVICTED] == 1 and s.ledger[k.L_LOST] - lost0 == pytest.approx(1.0)
    assert k.store_room(s, 1) and s.counters[k.C_STORE_EVICTED] == 1   # 자리가 있으면 치우지 않는다
    assert rel_residual(w) < 1e-9

    # 들판 아닌 물건이 모두 든 것
    w = world(10)
    s = w.s
    for i in range(10):
        hold(w, i, _place(w, born=-i))
    n_food = int(s.counters[k.C_NFOOD])
    assert not k.store_room(s, 1)
    assert s.counters[k.C_STORE_EVICTED] == 0 and s.counters[k.C_NFOOD] == n_food
    assert not k.evict_stored(s, 1)

    # 사체가 칸을 채운 경우
    w = world(0)
    s = w.s
    corpses = [_place(w, born=b, kind=k.KIND_CORPSE) for b in (5, 3, 9, 1, 7, 2, 8, 6, 4, 10)]
    assert s.counters[k.C_NFIELD] == 0 and s.counters[k.C_NFOOD] == 10
    assert k.store_room(s, 1)
    assert not s.f_alive[corpses[3]] and s.counters[k.C_STORE_EVICTED] == 1 and s.counters[k.C_NFOOD] == 9
    assert rel_residual(w) < 1e-9


def test_carry_toggle_recounts_and_releases():
    # carry 꺼진 ext 세계를 돌린 뒤 set_config로 켬 → C_NFIELD == 들판 알 수(사체 제외)
    # 손을 몇 개 맞춘 뒤 set_config로 끔 → a_held 모두 -1, f_holder 모두 -1, f_dropped는 그대로
    w = World(small("ext"))
    w.run(100)
    _place(w, kind=k.KIND_CORPSE)
    assert w.s.counters[k.C_NFIELD] == 0 and w.s.counters[k.C_NFOOD] > 1
    w.set_config(small("ext", carry_enabled=True))
    assert w.s.counters[k.C_NFIELD] == _n_field(w.s) == w.s.counters[k.C_NFOOD] - 1
    _shuffle_hands(w, np.random.default_rng(3))
    assert (w.s.a_held >= 0).sum() == 10
    dropped = w.s.f_dropped.copy()
    assert dropped.any()
    w.set_config(small("ext"))
    assert (w.s.a_held == -1).all() and (w.s.f_holder == -1).all()
    assert np.array_equal(w.s.f_dropped, dropped)


def test_carry_ledger_closes():
    # carry 켠 ext 세계 300스텝(매 50스텝 hold/unhold 도우미로 손을 무작위로 맞추고 푼다) → |residual|/(spawn+inject) < 1e-9
    w = World(small("ext", carry_enabled=True))
    rng = np.random.default_rng(11)
    for t in range(300):
        if t % 50 == 0:
            _shuffle_hands(w, rng)
        w.step()
    w.check_sane()
    _check_hands(w.s)
    assert w.s.counters[k.C_NFIELD] == _n_field(w.s)
    assert rel_residual(w) < 1e-9
