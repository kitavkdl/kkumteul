"""참여용 장치 시험: 각본(재난·기후 변화), 운석, 생물 묶음, 실험 카드.

    python3 -m pytest -q tests/test_scenario.py
"""
import dataclasses
import json
import signal
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kkumteul import checkpoint as ckpt  # noqa: E402
from kkumteul import events, experiment, measure, pack, run, scenario  # noqa: E402
from kkumteul import kernels as k  # noqa: E402
from kkumteul.config import Config  # noqa: E402
from kkumteul.run import Runner  # noqa: E402
from kkumteul.scenario import Scenario  # noqa: E402
from kkumteul.world import World  # noqa: E402

SEG = 700   # 시험에서는 기록 파일이 여러 개로 나뉘게 작게 잡는다
ROOT = Path(__file__).resolve().parents[1]


def ev(run_dir, kind):
    return events.load(run_dir, kind, SEG)


def small(**kw) -> Config:
    base = dict(world_w=256.0, world_h=256.0, max_agents=600, init_agents=150, food_cap=500,
                food_inflow=2.2, n_patches=4, stats_every=200, frame_every=50, clip_every=300, clip_len=20,
                ckpt_every=400, keyframe_every=200, min_agents=20, event_segment_steps=SEG)
    base.update(kw)
    return Config.from_dict({**Config().to_dict(), **base})


def scn(events_, **kw) -> Scenario:
    return Scenario.from_dict({"name": "시험", "events": events_, **kw})


def stats_rows(run_dir):
    return [json.loads(line) for line in (run_dir / "stats.jsonl").read_text().splitlines() if line.strip()]


def torus_dist(x, y, cx, cy, w, h):
    dx = np.abs(x - cx)
    dy = np.abs(y - cy)
    return np.hypot(np.minimum(dx, w - dx), np.minimum(dy, h - dy))


# 여러 사건 종류와 그 경계가 다 들어 있는 각본(이어 돌리기 시험용)
MIXED = {
    "name": "섞음", "description": "시험용", "update_every": 50,
    "events": [
        {"at": 300, "for": 350, "scale": {"food_inflow": 0.4}},
        {"from": 100, "to": 900, "scale_to": {"basal_cost": 1.5}},
        {"every": 400, "amp": {"thrust_cost": 0.3}, "phase": 0.1},
        {"at": 520, "set": {"max_age": 2500}},
        {"at": 450, "strike": {"radius": 80, "kill": 0.7}},
        {"every": 300, "first": 250, "strike": {"radius": 40, "burn_food": False}},
    ],
}


# ---- 각본 읽기와 검사 ----
BAD = [
    ({"name": "x", "events": [], "extra": 1}, "모르는 키"),
    ({"name": "x"}, "events"),
    ({"name": "", "events": []}, "name"),
    ({"name": "x", "events": [], "update_every": 0}, "update_every"),
    ({"name": "x", "events": [{"at": 1}]}, "정확히 하나"),
    ({"name": "x", "events": [{"at": 1, "set": {"food_inflow": 1}, "scale": {"food_inflow": 1}}]}, "정확히 하나"),
    ({"name": "x", "events": [{"at": 1, "set": {"food_inflow": 1}, "for": 3}]}, "모르는 키"),
    ({"name": "x", "events": [{"at": 1, "scale": {"food_inflow": 0.5}}]}, "빠진 키"),
    ({"name": "x", "events": [{"at": 1, "set": {"no_such_thing": 1}}]}, "모르는 설정"),
    ({"name": "x", "events": [{"at": 1, "set": {"max_agents": 10}}]}, "바꿀 수 없다"),
    ({"name": "x", "events": [{"at": 1, "set": {"seed": 3}}]}, "바꿀 수 없다"),
    ({"name": "x", "events": [{"at": 1, "for": 5, "scale": {"stats_every": 2}}]}, "바꿀 수 없다"),
    ({"name": "x", "events": [{"at": 1, "for": 5, "scale": {"corpse_to_food": 0.5}}]}, "배율"),
    ({"name": "x", "events": [{"at": 1, "set": {"corpse_to_food": 1}}]}, "true/false"),
    ({"name": "x", "events": [{"at": 1, "set": {"max_age": 2.5}}]}, "정수"),
    ({"name": "x", "events": [{"at": 1, "set": {"food_inflow": "많이"}}]}, "수여야"),
    ({"name": "x", "events": [{"at": 1, "for": 5, "scale": {"food_inflow": -0.5}}]}, "이상"),
    ({"name": "x", "events": [{"at": -1, "for": 5, "scale": {"food_inflow": 0.5}}]}, "이상"),
    ({"name": "x", "events": [{"at": 1, "for": 0, "scale": {"food_inflow": 0.5}}]}, "이상"),
    ({"name": "x", "events": [{"at": 1.5, "for": 5, "scale": {"food_inflow": 0.5}}]}, "정수"),
    ({"name": "x", "events": [{"from": 10, "to": 10, "scale_to": {"food_inflow": 0.5}}]}, "from보다"),
    ({"name": "x", "events": [{"every": 100, "amp": {"food_inflow": 1.5}}]}, "이하"),
    ({"name": "x", "events": [{"every": 100, "amp": {}}]}, "하나 이상"),
    ({"name": "x", "events": [{"at": 1, "strike": {"kill": 1.0}}]}, "radius"),
    ({"name": "x", "events": [{"at": 1, "strike": {"radius": 0}}]}, "radius"),
    ({"name": "x", "events": [{"at": 1, "strike": {"radius": 5, "kill": 1.5}}]}, "이하"),
    ({"name": "x", "events": [{"at": 1, "strike": {"radius": 5, "depth": 2}}]}, "모르는 키"),
    ({"name": "x", "events": [{"at": 1, "strike": {"radius": 5, "burn_food": "yes"}}]}, "true/false"),
    ({"name": "x", "events": [{"every": 0, "strike": {"radius": 5}}]}, "이상"),
    ({"name": "x", "events": ["가뭄"]}, "꼴"),
]


@pytest.mark.parametrize("bad, word", BAD)
def test_scenario_rejects_bad_input(bad, word):
    with pytest.raises(ValueError, match=word):
        Scenario.from_dict(bad)


def test_scenario_roundtrip_and_file(tmp_path):
    s = Scenario.from_dict(MIXED)
    assert Scenario.from_dict(s.to_dict()).to_dict() == s.to_dict()
    assert s.touched == ("basal_cost", "food_inflow", "max_age", "thrust_cost")
    assert s.edges == {300, 650, 520}
    p = tmp_path / "my-scenario.json"
    p.write_text(json.dumps({"events": [{"at": 5, "set": {"food_inflow": 1.0}, "note": "메모"}]}), encoding="utf-8")
    loaded = Scenario.load(p)
    assert loaded.name == "my-scenario", "이름이 없으면 파일 이름을 쓴다"
    assert loaded.to_dict()["events"][0]["note"] == "메모"
    assert loaded.sha256() == Scenario.from_dict(loaded.to_dict()).sha256()
    (tmp_path / "broken.json").write_text("{not json")
    with pytest.raises(ValueError, match="json"):
        Scenario.load(tmp_path / "broken.json")


def test_scenario_check_rejects_configs_that_fail_validation():
    base = small()
    for bad in ([{"at": 10, "set": {"v_max": 5.0}}],                     # 반지름보다 빠른 속도
                [{"at": 10, "for": 5, "scale": {"max_age": 0.0001}}],    # 반올림하면 0
                [{"at": 10, "set": {"child_frac": 1.5}}]):
        s = scn(bad)
        with pytest.raises(ValueError, match="맞지 않는다"):
            s.check(base)
    scn([{"at": 10, "for": 5, "scale": {"max_age": 0.5}}]).check(base)


def test_locked_list_covers_everything_world_refuses():
    """World.set_config가 막는 설정(배열 크기)이 각본의 금지 목록에 빠짐없이 있어야 한다."""
    cfg = small()
    w = World(cfg, populate=False)
    for f in dataclasses.fields(Config):
        v = getattr(cfg, f.name)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        try:
            w.set_config(dataclasses.replace(cfg, **{f.name: v + 1}))
        except ValueError as e:
            if "배열 크기" in str(e):
                assert f.name in scenario.LOCKED, f.name
        w.set_config(cfg)


# ---- 사건 종류별 값 ----
def test_set_changes_value_from_its_step_on():
    base = small()
    s = scn([{"at": 100, "set": {"food_inflow": 4.0, "corpse_to_food": False}},
             {"at": 300, "set": {"food_inflow": 1.0}}])
    assert s.effective_config(base, 0) == base
    assert s.effective_config(base, 99).food_inflow == 2.2
    c = s.effective_config(base, 100)
    assert c.food_inflow == 4.0 and c.corpse_to_food is False
    assert s.effective_config(base, 299).food_inflow == 4.0
    assert s.effective_config(base, 300).food_inflow == 1.0
    assert s.effective_config(base, 10**9).food_inflow == 1.0
    # 건드리지 않은 설정은 그대로다
    assert dataclasses.replace(c, food_inflow=2.2, corpse_to_food=True) == base


def test_pulse_scales_only_inside_its_window_and_rounds_integers():
    base = small()
    s = scn([{"at": 250, "for": 100, "scale": {"food_inflow": 0.5, "max_age": 0.3333}}])
    for t, inflow, age in ((249, 2.2, 3000), (250, 1.1, 1000), (349, 1.1, 1000), (350, 2.2, 3000)):
        c = s.effective_config(base, t)
        assert c.food_inflow == pytest.approx(inflow) and c.max_age == age and isinstance(c.max_age, int)
    # 겹친 배율은 곱해진다. set으로 바꾼 값 위에도 곱해진다
    s2 = scn([{"at": 0, "for": 100, "scale": {"food_inflow": 0.5}},
              {"at": 50, "for": 100, "scale": {"food_inflow": 0.5}},
              {"at": 60, "set": {"food_inflow": 8.0}}])
    assert s2.effective_config(base, 40).food_inflow == pytest.approx(1.1)
    assert s2.effective_config(base, 55).food_inflow == pytest.approx(0.55)
    assert s2.effective_config(base, 60).food_inflow == pytest.approx(2.0)
    assert s2.effective_config(base, 120).food_inflow == pytest.approx(4.0)
    assert s2.effective_config(base, 150).food_inflow == 8.0


def test_ramp_goes_straight_to_target_and_stays():
    base = small()
    s = scn([{"from": 1000, "to": 3000, "scale_to": {"basal_cost": 1.5}}], update_every=100)
    b = base.basal_cost
    assert s.effective_config(base, 0).basal_cost == b
    assert s.effective_config(base, 1000).basal_cost == b
    assert s.effective_config(base, 1500).basal_cost == pytest.approx(b * 1.125)
    assert s.effective_config(base, 2000).basal_cost == pytest.approx(b * 1.25)
    assert s.effective_config(base, 2999).basal_cost == pytest.approx(b * (1 + 0.5 * 1900 / 2000))
    assert s.effective_config(base, 3000).basal_cost == b * 1.5
    assert s.effective_config(base, 500000).basal_cost == b * 1.5


def test_cycle_follows_a_sine():
    base = small()
    s = scn([{"every": 400, "amp": {"food_inflow": 0.5}}], update_every=100)
    f = base.food_inflow
    assert s.effective_config(base, 0).food_inflow == f, "시작할 때는 바탕 값 그대로다"
    assert s.effective_config(base, 100).food_inflow == pytest.approx(f * 1.5)
    assert s.effective_config(base, 200).food_inflow == pytest.approx(f)
    assert s.effective_config(base, 300).food_inflow == pytest.approx(f * 0.5)
    assert s.effective_config(base, 400 * 1000).food_inflow == f, "주기의 배수에서는 정확히 바탕 값이다"
    shifted = scn([{"every": 400, "amp": {"food_inflow": 0.5}, "phase": 0.25}], update_every=100)
    assert shifted.effective_config(base, 0).food_inflow == pytest.approx(f * 1.5)
    late = scn([{"every": 400, "amp": {"food_inflow": 0.5}, "from": 800}], update_every=100)
    assert late.effective_config(base, 700).food_inflow == f
    assert late.effective_config(base, 900).food_inflow == pytest.approx(f * 1.5)


def test_strike_schedule():
    s = scn([{"at": 450, "strike": {"radius": 30}},
             {"every": 300, "first": 250, "until": 900, "strike": {"radius": 10, "kill": 0.5, "x": 3, "y": 4}},
             {"every": 500, "strike": {"radius": 5, "burn_food": False}}])
    assert s.touched == () and s.strikes_at(0) == []
    assert s.strikes_at(450) == [{"radius": 30.0, "kill": 1.0, "burn_food": True, "x": None, "y": None, "index": 0}]
    assert [x["index"] for x in s.strikes_at(250)] == [1]
    assert s.strikes_at(550)[0]["x"] == 3.0 and s.strikes_at(550)[0]["kill"] == 0.5
    assert s.strikes_at(850) and not s.strikes_at(1150), "until부터는 떨어지지 않는다"
    assert not s.strikes_at(0) and [x["index"] for x in s.strikes_at(500)] == [2], "first가 없으면 한 주기 뒤부터"
    assert s.strike_times(1000) == [250, 450, 500, 550, 850, 1000]


def test_effective_config_is_pure_and_quantised():
    base = small()
    s = Scenario.from_dict(MIXED)
    u = s.update_every
    order = list(range(0, 1200))
    np.random.default_rng(0).shuffle(order)
    seen = {t: s.effective_config(base, t) for t in order}          # 아무 순서로 물어도
    for t in range(0, 1200):
        assert seen[t] == s.effective_config(base, t)               # 같은 답이 나온다
        if t and t % u != 0 and t not in s.edges:
            assert seen[t] == seen[t - 1], f"{t}: 묶음 경계나 사건 경계가 아닌데 값이 바뀌었다"
    changed = [t for t in range(1, 1200) if seen[t] != seen[t - 1]]
    assert {300, 650, 520} <= set(changed), "set과 scale의 경계는 정확히 그 스텝에 반영된다"
    assert len(changed) > 15, "ramp와 계절은 update_every마다 바뀐다"


def test_apply_touches_world_config_only_when_it_changes():
    base = small()
    w = World(base)
    calls = []
    orig = w.set_config
    w.set_config = lambda cfg: (calls.append(w.step_no), orig(cfg))
    s = scn([{"at": 120, "for": 100, "scale": {"food_inflow": 0.5}}], update_every=10)
    for t in range(400):
        s.apply(w, base, w.step_no, 0, force=(t == 0))
        w.step()
    assert calls == [120, 220]
    assert w.cfg == base
    # 각본을 중간부터 시작해도(start) 시간은 시작 스텝부터 센다
    calls.clear()
    for _ in range(300):
        s.apply(w, base, w.step_no, 400)
        w.step()
    assert calls == [520, 620]


# ---- 운석 ----
def test_strike_kills_and_burns_only_inside_the_circle_across_the_edge():
    cfg = small()
    w = World(cfg)
    w.run(300)
    w.drain_events()
    s = w.s
    cx, cy, r = 5.0, 5.0, 40.0
    alive = np.flatnonzero(s.alive)
    food = np.flatnonzero(s.f_alive)
    assert len(alive) > 20 and len(food) > 20
    # 세계의 반대편 끝에 있어도 원 안이면 맞는다. 경계 바로 밖은 안 맞는다
    for i, (x, y) in zip(alive[:5], ((250.0, 250.0), (3.0, 252.0), (20.0, 10.0), (5.0 + 41.0, 5.0), (128.0, 128.0))):
        s.x[i], s.y[i] = x, y
    for j, (x, y) in zip(food[:4], ((254.0, 1.0), (1.0, 254.0), (5.0, 5.0 + 41.0), (200.0, 60.0))):
        s.fx[j], s.fy[j] = x, y
    inside_a = s.alive & (torus_dist(s.x, s.y, cx, cy, cfg.world_w, cfg.world_h) <= r)
    inside_f = s.f_alive & (torus_dist(s.fx, s.fy, cx, cy, cfg.world_w, cfg.world_h) <= r)
    assert inside_a[alive[:3]].all() and not inside_a[alive[3:5]].any()
    assert inside_f[food[:2]].all() and not inside_f[food[2:4]].any()
    alive_before, food_before = s.alive.copy(), s.f_alive.copy()
    ids_hit = set(s.ids[inside_a].tolist())
    lost_before = float(s.ledger[k.L_LOST])
    e_lost = float(s.energy[inside_a].sum() + s.f_energy[inside_f].sum())

    killed = w.strike(cx, cy, r, 1.0, True)
    assert killed == int(inside_a.sum()) > 0
    assert (s.alive == (alive_before & ~inside_a)).all()
    assert (s.f_alive == (food_before & ~inside_f)).all()
    assert int(s.counters[k.C_NFOOD]) == int(s.f_alive.sum())
    assert w.summary()["deaths_disaster"] == killed
    w.check_sane()
    assert float(s.ledger[k.L_LOST]) - lost_before == pytest.approx(e_lost)
    bal = w.energy_balance()
    scale = bal["spawn"] + bal["inject"]
    assert abs(bal["residual"]) / scale < 1e-9, bal
    _, d = w.drain_events()
    assert len(d) == killed and (d["cause"] == k.DEATH_DISASTER).all() and (d["step"] == 300).all()
    assert set(d["id"].tolist()) == ids_hit

    w.run(300)      # 그 뒤로도 세계가 멀쩡히 돌고 장부가 맞는다
    w.check_sane()
    bal = w.energy_balance()
    assert abs(bal["residual"]) / (bal["spawn"] + bal["inject"]) < 1e-9, bal


def test_strike_options_and_determinism():
    a, b, c = World(small()), World(small()), World(small())
    for w in (a, b, c):
        w.run(200)
    n_food = int(a.s.f_alive.sum())
    inside = int((a.s.alive & (torus_dist(a.s.x, a.s.y, 100, 100, 256, 256) <= 110)).sum())
    assert inside >= 30
    ka = a.strike(100, 100, 110, 0.5, False, key=9)
    kb = b.strike(100 + 256 * 3, 100 - 256, 110, 0.5, False, key=9)     # 세계 밖 좌표는 감아서 같은 자리
    kc = c.strike(100, 100, 110, 0.5, False, key=10)
    assert ka == kb and a.state_hash() == b.state_hash(), "같은 씨앗·같은 번호면 같은 개체가 죽는다"
    assert 0.25 * inside < ka < 0.75 * inside
    assert int(a.s.f_alive.sum()) == n_food, "burn_food가 꺼져 있으면 먹이는 그대로다"
    assert c.state_hash() != a.state_hash(), "번호가 다르면 죽는 개체가 달라진다"
    assert kc > 0 and a.strike(100, 100, 110, 0.0, False) == 0
    # 세계 전체를 덮는 운석: 모두 죽고, 다음 스텝에 하한까지 다시 채워진다(재주입으로 기록된다)
    pop = a.summary()["pop"]
    assert a.strike(0, 0, 1000, 1.0, True) == pop > 0
    assert a.summary()["pop"] == 0 and a.summary()["food"] == 0
    a.step()
    assert a.summary()["pop"] == a.cfg.min_agents and a.summary()["reseeds"] == a.cfg.min_agents
    bal = a.energy_balance()
    assert abs(bal["residual"]) / (bal["spawn"] + bal["inject"]) < 1e-9


def test_strike_position_comes_from_seed_and_step():
    p = scenario.strike_position(1, scenario.strike_key(500, 0), 256.0, 256.0)
    assert p == scenario.strike_position(1, scenario.strike_key(500, 0), 256.0, 256.0)
    assert 0 <= p[0] < 256 and 0 <= p[1] < 256
    assert p != scenario.strike_position(2, scenario.strike_key(500, 0), 256.0, 256.0)
    assert p != scenario.strike_position(1, scenario.strike_key(501, 0), 256.0, 256.0)
    assert p != scenario.strike_position(1, scenario.strike_key(500, 1), 256.0, 256.0)
    # 자리를 적지 않은 운석은 같은 세계에서 늘 같은 곳에 떨어진다
    s = scn([{"at": 100, "strike": {"radius": 60}}])
    a, b = World(small()), World(small())
    for w in (a, b):
        for _ in range(150):
            s.apply(w, w.cfg, w.step_no, 0)
            w.step()
    assert a.summary()["deaths_disaster"] > 0 and a.state_hash() == b.state_hash()


# ---- 실행기 ----
def test_empty_scenario_is_the_same_world_as_no_scenario(tmp_path):
    cfg = small()
    plain = Runner.new(tmp_path / "plain", cfg)
    plain.loop(800, quiet=True)
    plain.close()
    empty = Runner.new(tmp_path / "empty", cfg, scenario=scn([]))
    empty.loop(800, quiet=True)
    empty.close()
    assert plain.world.state_hash() == empty.world.state_hash()
    for kind in ("births", "deaths"):
        assert ev(tmp_path / "plain", kind).tobytes() == ev(tmp_path / "empty", kind).tobytes()
    assert "scenario" not in json.loads((tmp_path / "plain" / "run.json").read_text())
    assert not (tmp_path / "plain" / "scenario.json").exists()
    assert "scn" not in stats_rows(tmp_path / "plain")[-1]
    assert stats_rows(tmp_path / "plain")[-1]["w_disaster"] == 0
    assert stats_rows(tmp_path / "empty")[-1]["scn"] == {}


def test_scenario_run_resumed_matches_uninterrupted(tmp_path):
    cfg = small()
    s = Scenario.from_dict(MIXED)
    one = Runner.new(tmp_path / "one", cfg, scenario=s)
    one.loop(1000, quiet=True)
    one.close()

    # 경계마다 끊는다: 300(가뭄 시작), 450(운석이 떨어질 스텝), 650(가뭄 끝), 그리고 아무 데(777)
    two = Runner.new(tmp_path / "two", cfg, scenario=s)
    done = 0
    for upto in (300, 450, 650, 777, 1000):
        two.loop(upto - done, quiet=True)
        done = upto
        if upto == 450:
            # 체크포인트 뒤에 기록이 더 쌓인 채 꺼진 상황을 흉내 낸다
            two.world.run(60)
            two.flush_events()
            two.write_stats()
        two.close()
        if upto < 1000:
            two = Runner.resume(tmp_path / "two")
            assert two.world.step_no == upto and two.scenario is not None and two.scenario_start == 0
            assert two.base_cfg == cfg, "이어 돌릴 때 바탕은 config.json이다(체크포인트의 설정이 아니다)"
    assert one.world.state_hash() == two.world.state_hash()
    for kind in ("births", "deaths"):
        assert ev(tmp_path / "one", kind).tobytes() == ev(tmp_path / "two", kind).tobytes()

    # 실행 중에 주기적으로 쓴 체크포인트(800: 모든 사건의 한가운데)에서도 같다
    last = ckpt.list_checkpoints(tmp_path / "one" / "checkpoints")[-1][1]
    assert last.name.endswith("1000.npz")
    last.unlink()
    last.with_suffix(".json").unlink()
    again = Runner.resume(tmp_path / "one")
    assert again.world.step_no == 800 and again.world.cfg != cfg
    again.loop(200, quiet=True)
    again.close()
    assert again.world.state_hash() == one.world.state_hash()
    assert ev(tmp_path / "one", "deaths").tobytes() == ev(tmp_path / "two", "deaths").tobytes()

    # 각본이 실제로 일을 했는지
    d = ev(tmp_path / "one", "deaths")
    hit = d[d["cause"] == k.DEATH_DISASTER]
    assert len(hit) > 0 and set(np.unique(hit["step"])) <= {250, 450, 550, 850}
    assert one.world.summary()["deaths_disaster"] == len(hit)
    # (끊긴 실행은 끊긴 곳에서 통계 구간을 새로 세므로 구간 합은 끊지 않은 쪽에서 본다)
    assert sum(r["w_disaster"] for r in stats_rows(tmp_path / "one")) == len(hit)
    rows = stats_rows(tmp_path / "two")
    by_step = {r["step"]: r["scn"] for r in rows}
    assert by_step[200]["food_inflow"] == cfg.food_inflow and by_step[200]["max_age"] == 3000
    assert by_step[400]["food_inflow"] == pytest.approx(cfg.food_inflow * 0.4)
    assert by_step[800]["food_inflow"] == cfg.food_inflow and by_step[800]["max_age"] == 2500
    assert by_step[1000]["basal_cost"] == cfg.basal_cost * 1.5
    assert set(by_step[400]) == {"basal_cost", "food_inflow", "max_age", "thrust_cost"}
    bal = one.world.energy_balance()
    assert abs(bal["residual"]) / (bal["spawn"] + bal["inject"]) < 1e-9
    # 실행 폴더에 남는 것: 바탕 설정, 각본 사본, 시작 스텝
    assert Config.load(tmp_path / "one" / "config.json") == cfg
    assert Scenario.load(tmp_path / "one" / "scenario.json").to_dict() == s.to_dict()
    meta = json.loads((tmp_path / "one" / "run.json").read_text())
    assert meta["scenario"] == "섞음" and meta["scenario_start"] == 0 and meta["scenario_sha256"] == s.sha256()

    # 각본 없이 돌린 세계와는 달라야 한다
    plain = World(cfg)
    plain.run(1000)
    assert plain.state_hash() != one.world.state_hash()


def test_branch_with_scenario_starts_counting_at_the_branch_step(tmp_path):
    cfg = small()
    main = Runner.new(tmp_path / "main", cfg)
    main.loop(400, quiet=True)
    main.close()
    path = tmp_path / "pulse.json"
    path.write_text(json.dumps({"name": "짧은 가뭄", "update_every": 20,
                                "events": [{"at": 100, "for": 250, "scale": {"food_inflow": 0.25}}]}),
                    encoding="utf-8")
    control = Runner.branch(tmp_path / "main", tmp_path / "control", None, [], False, False)
    control.loop(400, quiet=True)
    control.close()
    treat = Runner.branch(tmp_path / "main", tmp_path / "treat", None, [], False, False, path)
    assert treat.scenario_start == 400
    treat.loop(100, quiet=True)
    assert treat.world.cfg == cfg, "각본 시작부터 100스텝이 되기 전에는 바탕 그대로다"
    treat.loop(1, quiet=True)
    assert treat.world.cfg.food_inflow == pytest.approx(cfg.food_inflow * 0.25)
    treat.loop(299, quiet=True)
    treat.close()
    meta = json.loads((tmp_path / "treat" / "run.json").read_text())
    assert meta["scenario"] == "짧은 가뭄" and meta["scenario_start"] == 400 and meta["parent_step"] == 400
    assert meta["rng"] == "same_stream" and meta["config_diff"] == {}
    assert control.world.state_hash() != treat.world.state_hash()
    assert control.world.summary()["food"] != treat.world.summary()["food"] or \
        control.world.summary()["pop"] != treat.world.summary()["pop"]

    # 각본을 쓰는 실행의 한가운데(가뭄 중)에서 갈라내도 갈래의 바탕은 부모의 바탕 설정이다
    mid = ckpt.list_checkpoints(tmp_path / "treat" / "checkpoints")
    mid = [p for step, p in mid if step == 501][0]
    assert json.loads(mid.with_suffix(".json").read_text())["config"]["food_inflow"] != cfg.food_inflow
    child = Runner.branch(mid, tmp_path / "child", None, ["brain_cost=0.02"], False, False)
    assert child.scenario is None and child.world.cfg.food_inflow == cfg.food_inflow
    assert json.loads((tmp_path / "child" / "run.json").read_text())["config_diff"] == {"brain_cost": [0.006, 0.02]}
    child.close()

    # 맞지 않는 각본은 실행 폴더를 만들기 전에 거절한다
    with pytest.raises(ValueError):
        Runner.branch(tmp_path / "main", tmp_path / "bad", None, [], False, False,
                      scn([{"at": 5, "set": {"v_max": 9.0}}]))
    assert not (tmp_path / "bad").exists()


def test_run_command_line_with_scenario(tmp_path, capsys):
    small().save(tmp_path / "cfg.json")
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"name": "명령줄", "events": [{"at": 50, "strike": {"radius": 70}}]}), encoding="utf-8")
    old = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        assert run.main(["new", "--run", str(tmp_path / "r"), "--config", str(tmp_path / "cfg.json"),
                         "--scenario", str(path), "--steps", "200", "--quiet"]) == 0
        assert run.main(["resume", "--run", str(tmp_path / "r"), "--steps", "100", "--quiet"]) == 0
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)
    meta = json.loads((tmp_path / "r" / "run.json").read_text())
    assert meta["scenario"] == "명령줄" and meta["scenario_start"] == 0
    world, cmeta = ckpt.load(ckpt.latest(tmp_path / "r" / "checkpoints"))
    assert cmeta["step"] == 300 and cmeta["summary"]["deaths_disaster"] > 0
    assert scenario.main([str(path), "--config", str(tmp_path / "cfg.json")]) == 0
    assert "검사 통과" in capsys.readouterr().out


def test_example_scenarios_are_valid():
    files = sorted((ROOT / "scenarios").glob("*.json"))
    names = {p.stem for p in files}
    assert {"drought", "ice-age", "seasons", "meteor", "meteor-shower", "warming", "famine-and-recovery"} <= names
    for p in files:
        s = Scenario.load(p)
        s.check(Config())
        assert len(s.description) > 30, f"{p.name}: 무엇을 하고 무엇을 볼지 적는다"
        assert s.events, p.name
        assert s.effective_config(Config(), 0) == Config(), f"{p.name}: 시작 스텝에는 바탕 그대로여야 비교가 쉽다"


# ---- 생물 묶음 ----
@pytest.fixture(scope="module")
def grown(tmp_path_factory):
    """조금 진화한 작은 세계 하나(여러 시험이 같이 쓴다)."""
    d = tmp_path_factory.mktemp("grown")
    r = Runner.new(d / "base", small(max_age=300, repro_min_age=50))
    r.loop(1600, quiet=True)
    r.close()
    return d / "base"


def test_pack_export_info_and_assay(grown, tmp_path, capsys):
    cps = ckpt.list_checkpoints(grown / "checkpoints")
    alive = json.loads(cps[-1][1].with_suffix(".json").read_text())["summary"]["pop"]
    assert alive > 50
    assert pack.main(["export", "--from", str(grown), "--out", str(tmp_path / "new.npz"), "--n", "50",
                      "--label", "새것"]) == 0
    info_old = pack.export(cps[1][1], tmp_path / "old.npz", n=100000, seed=3)
    assert info_old["n"] == info_old["source_alive"] and info_old["source_step"] == cps[1][0]
    assert pack.is_pack(tmp_path / "new.npz") and not pack.is_pack(cps[-1][1])
    assert not pack.is_pack(tmp_path / "nothing.npz")

    g, cfg, info = pack.read(tmp_path / "new.npz")
    assert info["label"] == "새것" and info["n"] == 50 and info["source_step"] == 1600
    assert info["source_npz_sha256"] == ckpt.file_sha256(cps[-1][1])
    assert set(g) == set(measure.GENOME) | {"gen"} and all(len(v) == 50 for v in g.values())
    assert cfg == small(max_age=300, repro_min_age=50)
    assert str(tmp_path.parent) not in json.dumps(info), "묶음에 만든 사람의 경로가 들어가면 안 된다"
    # 같은 씨앗으로 뽑으면 같은 개체가 뽑히고, 체크포인트의 유전체와 같다
    pack.export(grown, tmp_path / "again.npz", n=50, seed=1)
    g2, _, _ = pack.read(tmp_path / "again.npz")
    assert all((g[key] == g2[key]).all() for key in g)
    full = measure.load_side(cps[-1][1])["genomes"]
    rows = {v.tobytes() for v in full["Wout"]}
    assert all(v.tobytes() in rows for v in g["Wout"])

    assert pack.main(["info", str(tmp_path / "new.npz")]) == 0
    out = capsys.readouterr().out
    assert "새것" in out and "개체 수: 50" in out
    assert pack.describe(tmp_path / "old.npz")["shapes"]["Wrec"] == [64, 64]

    side = measure.load_side(tmp_path / "new.npz")
    assert side["step"] == 1600 and side["label"] == "새것" and len(side["genomes"]["n_hidden"]) == 50
    res = measure.assay(tmp_path / "old.npz", tmp_path / "new.npz", n_each=30, steps=600, reps=2, seed=5,
                        neutral=False)
    assert len(res["final_counts"]) == 2 and np.isfinite(res["r_per_1000"]["mean"])
    # 묶음과 체크포인트를 섞어도 된다
    mixed = measure.assay(cps[1][1], tmp_path / "new.npz", n_each=30, steps=300, reps=1, seed=5, neutral=False)
    assert mixed["a"]["step"] == cps[1][0] and mixed["b"]["step"] == 1600


def test_pack_from_scenario_run_carries_the_base_config(tmp_path):
    cfg = small()
    r = Runner.new(tmp_path / "r", cfg, scenario=scn([{"at": 10, "set": {"food_inflow": 1.0}}]))
    r.loop(100, quiet=True)
    r.close()
    assert r.world.cfg.food_inflow == 1.0
    pack.export(tmp_path / "r", tmp_path / "p.npz", n=20)
    assert pack.read(tmp_path / "p.npz")[1] == cfg


def test_assay_refuses_genomes_of_different_shape(tmp_path):
    for name, kw in (("a", {}), ("b", {"h_max": 32})):
        w = World(small(**kw))
        w.run(20)
        pack.export(ckpt.save(w, tmp_path / name), tmp_path / f"{name}.npz", n=30)
    with pytest.raises(ValueError, match="모양이 달라"):
        measure.assay(tmp_path / "a.npz", tmp_path / "b.npz", n_each=20, steps=100, reps=1)
    with pytest.raises(ValueError, match="묶음이 아니다"):
        pack.read(ckpt.latest(tmp_path / "a"))


# ---- 실험 카드 ----
def test_experiment_end_to_end(grown, tmp_path, capsys):
    path = tmp_path / "hard-times.json"
    path.write_text(json.dumps({
        "name": "힘든 시절", "description": "먹이가 줄고 운석이 떨어진다", "update_every": 50,
        "events": [{"at": 200, "for": 600, "scale": {"food_inflow": 0.5}},
                   {"at": 400, "strike": {"radius": 70, "kill": 0.8}}]}), encoding="utf-8")
    args = ["--from", str(grown), "--scenario", str(path), "--steps", "1200", "--name", "hard-times",
            "--out", str(tmp_path / "experiments"), "--runs", str(tmp_path / "runs"),
            "--reps", "2", "--assay-steps", "900", "--n", "40", "--quiet"]
    assert experiment.main(args) == 0
    out = tmp_path / "experiments" / "hard-times"
    text = (out / "card.json").read_text(encoding="utf-8")
    card = json.loads(text)
    md = (out / "card.md").read_text(encoding="utf-8")

    assert card["name"] == "hard-times" and card["steps"] == 1200 and card["seed"] == 1
    assert card["scenario"]["name"] == "힘든 시절" and card["scenario"]["events"][0]["scale"] == {"food_inflow": 0.5}
    assert card["scenario"]["sha256"] == Scenario.load(path).sha256()
    last = ckpt.latest(grown / "checkpoints")
    assert card["parent"] == {"checkpoint": last.name, "npz_sha256": ckpt.file_sha256(last), "step": 1600}
    assert len(card["source_hash"]) == 64 and card["config_diff_from_default"]["world_w"] == 256.0
    c, t = card["arms"]["control"], card["arms"]["treatment"]
    for arm in (c, t):
        assert arm["pop_min"] <= arm["pop_final"] <= arm["pop_max"]
        assert arm["generations_gained"] > 0 and len(arm["hidden_mean_first_last"]) == 2
        assert {"reseeds", "disaster_deaths", "fitness_transmission_mean", "birth_ratio"} <= set(arm)
    assert c["disaster_deaths"] == 0 and t["disaster_deaths"] > 0
    assert c["hidden_mean_first_last"][0] == t["hidden_mean_first_last"][0], "두 갈래는 같은 곳에서 출발한다"
    assay = card["assay"]
    assert assay["a"] == "control" and assay["b"] == "treatment" and assay["n_each"] == 40
    assert len(assay["final_counts"]) == 2 and assay["verdict"]
    assert card["verdict"] and "B가" not in card["verdict"] and "A가" not in card["verdict"]
    assert "읽는 법" in md and "힘든 시절" in md and card["verdict"] in md and "재주입" in md
    assert str(tmp_path.parent) not in text and str(tmp_path.parent) not in md, "카드에 기계의 경로가 남으면 안 된다"
    assert "NaN" not in text and len(text) < 6000, "카드는 작아야 한다"

    # 두 갈래의 실행 폴더: 대조에는 각본이 없고, 각본 쪽은 갈라낸 스텝부터 센다
    runs = tmp_path / "runs"
    assert not (runs / "exp-hard-times-control" / "scenario.json").exists()
    meta = json.loads((runs / "exp-hard-times-treatment" / "run.json").read_text())
    assert meta["scenario_start"] == 1600 and meta["rng"] == "same_stream"

    # 같은 명령을 다시 치면 다시 돌리지 않고 같은 카드가 나온다
    assert experiment.main(args) == 0
    assert (out / "card.json").read_text(encoding="utf-8") == text
    assert "판정" in capsys.readouterr().out
    # 겨루기 옵션만 바꾸면 두 갈래는 그대로 두고 겨루기만 다시 한다
    assert experiment.main(args[:-1] + ["--assay-seed", "77", "--quiet"]) == 0
    again = json.loads((out / "card.json").read_text(encoding="utf-8"))
    assert again["arms"] == card["arms"] and again["assay"]["seed"] == 77
    # 같은 이름으로 다른 조건을 주면 거절한다
    with pytest.raises(FileExistsError):
        experiment.run_experiment(grown, path, 500, "hard-times", tmp_path / "experiments", tmp_path / "runs")
    with pytest.raises(FileExistsError, match="출발점"):
        experiment.run_experiment(grown, path, 1200, "hard-times", tmp_path / "experiments", tmp_path / "runs",
                                  at=400)
    with pytest.raises(ValueError):
        experiment.run_experiment(grown, path, 500, "../밖", tmp_path / "experiments", tmp_path / "runs")


def test_experiment_reports_extinction_instead_of_an_assay(tmp_path):
    r = Runner.new(tmp_path / "base", small(min_agents=0))
    r.loop(400, quiet=True)
    r.close()
    doom = scn([{"at": 299, "strike": {"radius": 1000}}])
    path = tmp_path / "doom.json"
    path.write_bytes(doom.to_bytes())
    card = experiment.run_experiment(tmp_path / "base", path, 300, "doom", tmp_path / "experiments",
                                     tmp_path / "runs", reps=1, assay_steps=200, n_each=20)
    assert card["arms"]["treatment"]["pop_final"] == 0 and card["arms"]["control"]["pop_final"] > 20
    assert "skipped" in card["assay"] and "각본 쪽 생존자가 0마리" in card["assay"]["skipped"]
    assert card["verdict"].startswith("겨루지 않음")
    assert card["arms"]["treatment"]["generations_gained"] is None
    md = (tmp_path / "experiments" / "doom" / "card.md").read_text(encoding="utf-8")
    assert "겨루지 않" in md


def test_migrate_injects_tagged_agents_into_a_branch(tmp_path):
    """이주: 받는 세계에서 갈라낸 실행에 묶음의 생물이 표지를 달고 들어가고, 그 실행을 이어 돌릴 수 있다."""
    from kkumteul import events
    from kkumteul import kernels as kk
    from kkumteul import pack
    from kkumteul.run import Runner
    a = Runner.new(tmp_path / "a", small())
    a.loop(600, quiet=True)
    a.close()
    b = Runner.new(tmp_path / "b", small(seed=7))
    b.loop(600, quiet=True)
    b.close()
    pack.export(tmp_path / "b", tmp_path / "b.pack.npz", n=30, seed=1, label="섬 B")
    before = a.world.summary()["pop"]
    info = pack.migrate(tmp_path / "a", tmp_path / "b.pack.npz", tmp_path / "a2", n=25, seed=3)
    assert info["n"] == 25 and info["at_step"] == a.world.step_no
    r = Runner.resume(tmp_path / "a2")
    w = r.world
    alive = w.s.alive
    assert int(alive.sum()) == before + 25 and int((w.s.tag[alive] == 1).sum()) == 25
    assert w.summary()["migrant_frac"] == pytest.approx(25 / (before + 25))
    r.loop(400, quiet=True)
    r.close()
    bal = w.energy_balance()
    assert abs(bal["residual"]) / (bal["spawn"] + bal["inject"]) < 1e-9
    births = events.load(tmp_path / "a2", "births", w.cfg.event_segment_steps)
    assert int((births["kind"] == kk.BIRTH_INJECT).sum()) == 25
    assert (np.diff(births["id"]) > 0).all()
