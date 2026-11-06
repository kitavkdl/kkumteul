"""다시 보기(replay) 시험: 체크포인트에서 다시 돌린 장면이 세계와 맞는지, 내보낸 파일이 혼자 열리는지.

    python3 -m pytest -q tests/test_replay.py
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_harsh import _place  # noqa: E402
from test_stages import small as ext_small  # noqa: E402
from kkumteul import animals as an  # noqa: E402
from kkumteul import checkpoint as ckpt  # noqa: E402
from kkumteul import kernels as k  # noqa: E402
from kkumteul import replay  # noqa: E402
from kkumteul.config import ACT_ATTACK, ACT_CARRY, ACT_GIVE, ACT_REPRO, ACT_THRUST, Config  # noqa: E402
from kkumteul.run import Runner  # noqa: E402
from kkumteul.scenario import Scenario  # noqa: E402
from kkumteul.world import World  # noqa: E402


def small(*presets, **kw) -> Config:
    base = dict(world_w=256.0, world_h=256.0, max_agents=600, init_agents=150, food_cap=500,
                food_inflow=2.2, n_patches=4, stats_every=200, frame_every=50, clip_every=300,
                clip_len=20, ckpt_every=400, keyframe_every=200, min_agents=20, event_segment_steps=700)
    base.update(kw)
    cfg = Config.from_dict({**Config().to_dict(), **base})
    return cfg.with_overrides([f"preset={p}" for p in presets])


def ckpt_of(run_dir: Path, step: int) -> Path:
    return run_dir / "checkpoints" / f"ckpt_{step:012d}.npz"


def listing(run_dir: Path) -> list:
    return sorted((str(p.relative_to(run_dir)), p.stat().st_size, p.stat().st_mtime_ns)
                  for p in run_dir.rglob("*") if p.is_file() and p.name != ".lock")


@pytest.fixture(scope="module")
def kinds_run(tmp_path_factory) -> Path:
    """먹이 종류를 켠 작은 세계를 400스텝 돌린 실행 폴더."""
    d = tmp_path_factory.mktemp("replay") / "kinds-world"
    r = Runner.new(d, small("stage2", "kinds", poison=0.3, kind_flip_period=150, max_age=300, repro_min_age=50))
    r.loop(400, quiet=True)
    r.close()
    return d


# ---------------------------------------------------------------- 장면이 세계와 맞는가
def test_frames_follow_the_world_step_by_step(kinds_run):
    path = ckpt_of(kinds_run, 400)
    rep = replay.record(path, steps=120, every=2)
    w, _ = ckpt.load(path)
    assert rep.start == 400 and rep.every == 2 and rep.n_frames == 61
    assert list(rep.steps[:3]) == [400, 402, 404] and rep.steps[-1] == 520
    for f in range(rep.n_frames):
        fr = rep.frame(f)
        a = np.flatnonzero(w.s.alive)
        ids = rep.agents["id"][fr.idx]
        assert len(ids) == len(a) == rep.pop[f]
        assert sorted(ids) == sorted(w.s.ids[a]), "장면의 개체는 그 스텝에 살아 있는 개체와 같다"
        by_id = dict(zip(w.s.ids[a].tolist(), a.tolist()))
        slot = np.array([by_id[int(i)] for i in ids])
        assert np.abs(fr.x - w.s.x[slot]).max() <= 256.0 / 65536 + 1e-9
        assert np.abs(fr.y - w.s.y[slot]).max() <= 256.0 / 65536 + 1e-9
        assert ((fr.flags & replay.F_ASLEEP) > 0).tolist() == (w.s.asleep[slot] > 0).tolist()
        fa = w.s.f_alive
        on = rep.food[(rep.food["f_in"] <= f) & (f < rep.food["f_out"])]
        assert len(on) == int(fa.sum()) == rep.n_food[f], "먹이 표에서 이 장면에 있는 먹이 수 = 세계의 먹이 수"
        assert sorted(on["kind"].tolist()) == sorted(w.s.f_kind[fa].tolist())
        if f < rep.n_frames - 1:
            w.run(2)
    # 다시 돌리는 동안 나고 죽은 개체는 장면 사이에 있었더라도 모두 표에 있다
    births, deaths = w.drain_events()
    row = {int(i): n for n, i in enumerate(rep.agents["id"])}
    assert len(row) == len(rep.agents), "개체 번호는 겹치지 않는다"
    assert len(births) > 0 and len(deaths) > 0
    for b in births:
        r = rep.agents[row[int(b["id"])]]
        assert r["born"] == b["step"] and r["kind"] == b["kind"] and r["gen"] == b["gen"]
        if b["parent"] >= 0:
            assert r["parent"] >= 0 and rep.agents["id"][r["parent"]] == b["parent"], "부모도 표에 있다"
    for d in deaths:
        r = rep.agents[row[int(d["id"])]]
        assert r["died"] == d["step"] and r["cause"] == d["cause"]
    alive_end = rep.agents["died"] < 0
    assert int(alive_end.sum()) == rep.pop[-1]


def test_eating_is_read_from_the_food_side_and_matches_counters(kinds_run):
    path = ckpt_of(kinds_run, 400)
    rep = replay.record(path, steps=300, every=2)
    w, _ = ckpt.load(path)
    c0 = w.s.counters.copy()
    w.run(300)
    dc = w.s.counters - c0
    eaten, poisoned = int(dc[k.C_EATEN]), int(dc[k.C_POISONED])
    assert eaten > 50 and poisoned > 5      # 시험이 뜻이 있으려면 먹은 것과 독이 넉넉해야 한다
    # 화면의 숫자(먹은 수, 독 수)는 세계의 계수기에서 온다
    assert int(rep.c_eaten.sum()) == eaten and int(rep.c_poisoned.sum()) == poisoned
    # 누가 무엇을 먹었는지는 사라진 먹이에서 가린다. 같은 스텝에 났다가 먹힌 먹이만 빠진다
    seen = rep.seen
    found = seen["good"] + seen["poison"] + seen["unknown"]
    print(f"먹은 수 {eaten}(독 {poisoned}) / 먹이 쪽에서 가린 것 좋음 {seen['good']} 독 {seen['poison']} "
          f"모름 {seen['unknown']} / 빠진 것 {eaten - found}")
    assert 0 <= eaten - found <= 0.02 * eaten
    assert abs(seen["poison"] - poisoned) <= 0.02 * eaten
    assert seen["unknown"] <= 0.02 * eaten
    assert int(rep.agents["ate_good"].sum()) == seen["good"]
    assert int(rep.agents["ate_poison"].sum()) == seen["poison"]
    why = rep.food["why"]
    assert int((why == replay.WHY_GOOD).sum()) + int((why == replay.WHY_POISON).sum()) <= found
    # 지금 좋은 종류는 먹기 커널이 그 스텝에 쓴 값이다(장면의 스텝 - 1)
    cfg = w.cfg
    for f in (1, 30, 75, 76, 150):
        assert rep.good_kind[f] == k.good_kind(cfg.seed, int(rep.steps[f]) - 1, cfg.kind_flip_period)
    # 눈 감고 먹었을 때의 독 비율: 그때 세계에 놓인 먹이 가운데 독인 것
    fa = w.s.f_alive
    good = k.good_kind(cfg.seed, w.step_no - 1, cfg.kind_flip_period)
    bad = int((~k.always_edible_mask(w.s.f_kind[fa]) & (w.s.f_kind[fa] != good)).sum())
    assert rep.blind[-1] == pytest.approx(bad / max(int(fa.sum()), 1), abs=1e-6)


def test_one_poison_and_one_good_food_are_told_apart():
    cfg = small("kinds", init_agents=1, min_agents=0, food_inflow=0.0, poison=0.3)
    w = World(cfg)
    s = w.s
    i = int(np.flatnonzero(s.alive)[0])
    good = k.good_kind(cfg.seed, 0, cfg.kind_flip_period)
    s.x[i], s.y[i], s.heading[i] = 100.0, 100.0, 0.0
    s.vx[:] = s.vy[:] = 0.0
    for j, (dx, kind) in enumerate(((0.4, 1 - good), (-0.4, good))):
        s.fx[j], s.fy[j], s.f_energy[j], s.f_kind[j], s.f_alive[j] = 100.0 + dx, 100.0, 1.0, kind, True
    s.counters[k.C_NFOOD] = 2
    s.ledger[k.L_SPAWN] += 2.0
    rec = replay.Recorder(w, every=1)
    rec.advance(1)
    rep = rec.finish()
    me = rep.agents[rep.agents["id"] == s.ids[i]][0]
    assert me["ate_good"] == 1 and me["ate_poison"] == 1
    fr = rep.frame(1)
    assert len(fr.flags) == 1 and fr.flags[0] & 3 == 1 and (fr.flags[0] >> 2) & 3 == 1
    food = rep.food
    assert len(food) == 2 and (food["f_in"] == 0).all() and (food["f_out"] == 1).all()
    by_kind = {int(r["kind"]): int(r["why"]) for r in food}
    assert by_kind[good] == replay.WHY_GOOD and by_kind[1 - good] == replay.WHY_POISON
    assert rep.seen == {"good": 1, "poison": 1, "unknown": 0, "evicted": 0, "burnt": 0, "gave": 0, "spoiled": 0,
                        "picked": 0, "dropped": 0, "stored_out": 0}
    assert int(rep.c_eaten[1]) == 2 and int(rep.c_poisoned[1]) == 1


def test_agent_that_lives_one_step_between_frames_is_still_in_the_table():
    w = World(small("stage2", "kinds", max_age=300, repro_min_age=50))
    w.run(300)
    rec = replay.Recorder(w, every=3)
    victim = None
    for _ in range(600):
        rec.advance(1)
        if (w.step_no - rec.start) % 3 == 1:
            born = np.flatnonzero(w.s.alive & (w.s.birth_step == w.step_no - 1))
            if len(born):
                victim = int(w.s.ids[born[0]])
                w.s.energy[born[0]] = -1.0     # 다음 스텝에 굶어 죽는다(장면과 장면 사이에만 산다)
                break
    assert victim is not None, "600스텝 안에 맞는 때의 출생이 없었다"
    rec.advance(5)
    rep = rec.finish()
    r = rep.agents[rep.agents["id"] == victim]
    assert len(r) == 1
    assert r["died"][0] - r["born"][0] == 1 and r["cause"][0] == k.DEATH_STARVE
    assert r["parent"][0] >= 0
    for f in range(rep.n_frames):
        assert victim not in rep.agents["id"][rep.frame(f).idx]


def test_signals_gifts_and_hits_are_marked():
    # 건네기: 공격이 없는 세계에서는 표시한 수가 세계의 계수기와 거의 같다
    # (공격이 섞이면 건넨 뒤 맞아 죽거나 에너지가 깎인 개체를 놓쳐 더 벌어진다. 화면 표시용 어림이다)
    w = World(small("stage3"))
    w.s.bout[w.s.alive, ACT_GIVE] = 1.0
    c0 = w.s.counters.copy()
    rec = replay.Recorder(w, every=2)
    rec.advance(200)
    rep = rec.finish()
    gifts = int((w.s.counters - c0)[k.C_GIFTS])
    assert gifts > 50 and int(rep.gifts.sum()) == gifts
    print(f"건넨 횟수 {gifts}, 화면에 표시한 주는 개체 {rep.seen['gave']}")
    assert abs(rep.seen["gave"] - gifts) <= 0.05 * gifts
    flags = np.concatenate([rep.frame(f).flags for f in range(rep.n_frames)])
    assert (flags & replay.F_GAVE).any() and not (flags & replay.F_HIT).any()
    # 공격: 맞은 개체에 표시가 선다
    w = World(small("stage3", "conflict"))
    w.s.bout[w.s.alive, ACT_ATTACK] = 1.0
    c0 = w.s.counters.copy()
    rec = replay.Recorder(w, every=2)
    rec.advance(200)
    rep = rec.finish()
    attacks = int((w.s.counters - c0)[k.C_ATTACKS])
    assert attacks > 0 and int(rep.attacks.sum()) == attacks
    flags = np.concatenate([rep.frame(f).flags for f in range(rep.n_frames)])
    assert (flags & replay.F_HIT).any()
    fr = rep.frame(rep.n_frames - 1)
    al = np.flatnonzero(w.s.alive)
    by_id = dict(zip(w.s.ids[al].tolist(), al.tolist()))
    slot = np.array([by_id[int(i)] for i in rep.agents["id"][fr.idx]])
    assert fr.sig is not None and fr.sig.shape == (len(slot), 2)
    assert np.abs(fr.sig - w.s.sig[slot]).max() <= 1.0 / 127 + 1e-6
    # 고리 문턱: 이 기록에서 가장 센 신호 5%(95번째 백분위수), 그래도 0.5보다 낮지는 않다
    mg = np.abs(rep.cols["sig"].astype(np.float32).reshape(-1, 2) / 127.0).max(axis=1)
    assert rep.meta["sig_min"] == pytest.approx(max(replay.SIG_MIN_FLOOR, float(np.percentile(mg, 95))), abs=1e-4)
    # 사회가 꺼진 세계에는 신호 자료가 없다
    plain = replay.Recorder(World(small()), every=2)
    plain.advance(4)
    prep = plain.finish()
    assert prep.frame(1).sig is None and prep.meta["sig_min"] is None


# ---------------------------------------------------------------- 어디서부터, 무엇을 읽는가
def test_from_flip_starts_a_little_before_the_next_switch(kinds_run):
    path = ckpt_of(kinds_run, 400)
    w, _ = ckpt.load(path)
    cfg = w.cfg
    period = cfg.kind_flip_period
    b = (400 // period + 1) * period
    while k.good_kind(cfg.seed, b, period) == k.good_kind(cfg.seed, b - 1, period):
        b += period
    start = max(400, b - min(300, period // 4))
    rep = replay.record(path, steps=200, every=2, from_flip=True)
    assert rep.start == start and rep.meta["ckpt_step"] == 400
    w.run(start - 400)
    assert rep.pop[0] == int(w.s.alive.sum())
    change = [int(rep.steps[f]) for f in range(1, rep.n_frames) if rep.good_kind[f] != rep.good_kind[f - 1]]
    assert change and b + 1 <= change[0] <= b + 2, "바뀐 뒤의 첫 장면"
    flips = rep.meta["flips"]
    assert flips and flips[0]["step"] == b and flips[0]["good"] == k.good_kind(cfg.seed, b, period)


def test_overrides_change_the_replayed_world_and_are_noted(kinds_run):
    """독 0 같은 대조는 같은 체크포인트를 설정 하나만 바꿔 다시 돌린다(placed.py --set)."""
    path = ckpt_of(kinds_run, 400)
    rep = replay.record(path, steps=120, every=2, overrides=["poison=0"])
    w, _ = ckpt.load(path)
    w.set_config(w.cfg.with_overrides(["poison=0"]))
    pops = []
    for _ in range(rep.n_frames):
        pops.append(int(w.s.alive.sum()))
        w.run(2)
    assert list(rep.pop) == pops, "덮어쓴 설정으로 돈 세계와 장면이 같다"
    assert rep.meta["overrides"] == ["poison=0"]
    assert any("poison=0" in n for n in rep.meta["notes"])
    plain = replay.record(path, steps=120, every=2)
    assert plain.meta["overrides"] == [] and not any("설정을 바꿔" in n for n in plain.meta["notes"])


def test_passive_control_checkpoint_is_refused(tmp_path):
    w = World(small(passive=True))
    path = ckpt.save(w, tmp_path / "checkpoints")
    with pytest.raises(ValueError, match="대조 실행"):
        replay.record(path, steps=10)
    with pytest.raises(ValueError, match="대조 실행"):      # 덮어쓰기로 passive를 끄고 피할 수는 없다
        replay.record(path, steps=10, overrides=["passive=0"])


def test_scenario_run_is_replayed_with_its_scenario(tmp_path):
    scn = Scenario.from_dict({"name": "시험 가뭄", "description": "먹이가 줄고 운석이 떨어진다", "events": [
        {"at": 420, "for": 40, "scale": {"food_inflow": 0.5}},
        {"at": 411, "strike": {"radius": 70, "kill": 1.0, "burn_food": True, "x": 128, "y": 128}}]})
    run_dir = tmp_path / "scn"
    r = Runner.new(run_dir, small(), scenario=scn)
    r.loop(400, quiet=True)
    r.close()
    rep = replay.record(ckpt_of(run_dir, 400), steps=100, every=2)
    info = rep.meta["scenario"]
    assert info["name"] == "시험 가뭄" and info["description"] == "먹이가 줄고 운석이 떨어진다"
    assert len(info["strikes"]) == 1
    st = info["strikes"][0]
    assert st["step"] == 411 and st["frame"] == 6 and st["killed"] > 0 and st["radius"] == 70
    assert (st["x"], st["y"]) == (128, 128)
    assert [(c["step"], c["values"]) for c in info["changes"]] == [
        (420, {"food_inflow": pytest.approx(1.1)}), (460, {"food_inflow": pytest.approx(2.2)})]
    assert rep.seen["burnt"] > 0
    assert int((rep.agents["cause"] == k.DEATH_DISASTER).sum()) == st["killed"]
    # 실행기가 같은 체크포인트에서 이어 돌린 것과 같은 세계다
    r = Runner.resume(run_dir)
    r.loop(100, quiet=True)
    w = r.world
    last = rep.frame(rep.n_frames - 1)
    assert sorted(rep.agents["id"][last.idx]) == sorted(w.s.ids[w.s.alive])
    assert rep.n_food[-1] == int(w.s.f_alive.sum())
    r.close()
    # 각본 파일을 읽을 수 없으면 그렇다고 적는다
    (run_dir / "scenario.json").unlink()
    rep2 = replay.record(ckpt_of(run_dir, 400), steps=10, every=2)
    assert rep2.meta["scenario"] is None and any("각본 적용 안 함" in n for n in rep2.meta["notes"])


def test_scenario_run_replays_with_overrides(tmp_path):
    # 각본이 걸린 실행의 체크포인트를 설정을 바꿔(크기 포함) 다시 보면, 첫 스텝에 각본이 바탕 설정을 다시 맞춰도
    # 덮어쓰기가 남는다(크기가 달라 죽거나, 덮어쓰기가 조용히 지워지면 안 된다)
    scn = Scenario.from_dict({"name": "시험 가뭄", "description": "먹이가 줄어든다", "events": [
        {"at": 420, "for": 40, "scale": {"food_inflow": 0.5}}]})
    run_dir = tmp_path / "scn"
    r = Runner.new(run_dir, small(), scenario=scn)
    r.loop(400, quiet=True)
    r.close()
    rep = replay.record(ckpt_of(run_dir, 400), steps=100, every=2, overrides=["preset=ext", "poison=0"])
    assert rep.meta["cfg"]["poison"] == 0.0                         # 덮어쓴 값이 각본이 맞춘 설정에도 남는다
    assert rep.meta["scenario"]["name"] == "시험 가뭄"
    assert [c["step"] for c in rep.meta["scenario"]["changes"]] == [420, 460]
    assert any("poison=0" in n for n in rep.meta["notes"])
    # 비교: 덮어쓰지 않으면 원래 값이다
    assert replay.record(ckpt_of(run_dir, 400), steps=10, every=2).meta["cfg"]["poison"] == 1.0


def test_history_reads_only_whole_rows_up_to_the_checkpoint(kinds_run, tmp_path):
    run_dir = tmp_path / "copy"
    shutil.copytree(kinds_run, run_dir, ignore=shutil.ignore_patterns(".lock"))
    with open(run_dir / "stats.jsonl", "ab") as f:
        f.write(json.dumps({"step": 600, "pop": 123, "eaten": 9, "poisoned": 1, "hidden_mean": 2.0}).encode() + b"\n")
        f.write(json.dumps({"step": 9000, "pop": 7}).encode() + b"\n")
        f.write(b'{"step": 9200, "pop": 1')          # 쓰다 만 마지막 줄
    rep = replay.record(ckpt_of(run_dir, 400), steps=300, every=2)
    hist = rep.meta["history"]
    assert hist["step"] == [200, 400] and len(hist["pop"]) == 2 and len(hist["hidden"]) == 2
    assert hist["poison"][0] is not None and 0.0 <= hist["poison"][1] <= 1.0
    assert rep.meta["orig"] == [[600, 123]], "다시 돌린 구간에 있는 원래 기록은 따로 적는다"
    # 기록이 없으면 긴 흐름을 뺀다
    (run_dir / "stats.jsonl").unlink()
    rep = replay.record(ckpt_of(run_dir, 400), steps=10, every=2)
    assert rep.meta["history"] is None and rep.meta["orig"] == []


def test_intro_and_title_come_from_the_config_and_folder(kinds_run):
    rep = replay.record(ckpt_of(kinds_run, 400), steps=10, every=2)
    intro = " ".join(rep.meta["intro"])
    assert "독" in intro and "가르치지 않" in intro
    assert rep.meta["title"] == "kinds-world"
    assert replay.record(ckpt_of(kinds_run, 400), steps=10, title="내 세계").meta["title"] == "내 세계"
    plain = replay.Recorder(World(small()), every=2)
    plain.advance(4)
    assert "독" not in " ".join(plain.finish().meta["intro"])
    stars = rep.meta["stars"]
    assert 1 <= len(stars) <= 4 and all(0 <= s["agent"] < len(rep.agents) and s["why"] for s in stars)


def test_intro_names_harsh_devices_only_when_on():
    words = ("겨울", "가을", "붙어 있으면", "큰 짐승", "고기", "들고 다니기")
    plain = " ".join(replay.intro_sentences(ext_small("ext")))
    assert not [w for w in words if w in plain]
    harsh = " ".join(replay.intro_sentences(ext_small("harsh")))
    assert "겨울" in harsh and not [w for w in words[1:] if w in harsh]
    harsh2 = " ".join(replay.intro_sentences(ext_small("harsh2")))
    assert all(w in harsh2 for w in words)
    assert "4,000스텝" in harsh2 and "1,200스텝" in harsh2 and "800스텝" in harsh2


# ---------------------------------------------------------------- 내보낸 파일
def test_export_is_one_self_contained_file_and_touches_nothing(kinds_run, tmp_path):
    path = ckpt_of(kinds_run, 400)
    sha, before = ckpt.file_sha256(path), listing(kinds_run)
    out = tmp_path / "out.html"
    rep = replay.write(path, out, steps=100, every=2)
    assert ckpt.file_sha256(path) == sha, "체크포인트는 읽기만 한다"
    assert listing(kinds_run) == before, "실행 폴더에 아무것도 쓰지 않는다"
    html = out.read_text(encoding="utf-8")
    assert html.lstrip().lower().startswith("<!doctype html>")
    for pat in (r"\bsrc\s*=", r"\bhref\s*=", r"url\(", r"fetch\(", r"XMLHttpRequest", r"@import",
                r"\bimport\s*\(", r"https?://(?!www\.w3\.org/)"):
        assert not re.search(pat, html), f"바깥에서 불러오는 것이 있다: {pat}"
    assert str(tmp_path) not in html and str(kinds_run.parent) not in html and "/home/" not in html
    assert "kinds-world" in html
    assert "__REPLAY_DATA__" not in html
    # 파일에 든 자료를 다시 꺼내면 적은 것과 같다
    meta, arrays = replay.read_embedded(html)
    assert meta["n_frames"] == rep.n_frames == 51 and meta["every"] == 2 and meta["start"] == 400
    fr = [rep.frame(f) for f in range(rep.n_frames)]
    assert arrays["fr_off"].tolist() == np.concatenate([[0], np.cumsum([len(x.idx) for x in fr])]).tolist()
    assert arrays["fr_idx"].tolist() == np.concatenate([x.idx for x in fr]).tolist()
    assert arrays["fr_flags"].tolist() == np.concatenate([x.flags for x in fr]).tolist()
    assert arrays["ag_id"].tolist() == rep.agents["id"].tolist()
    assert arrays["fd_in"].tolist() == rep.food["f_in"].tolist()
    assert arrays["f_good"].tolist() == rep.good_kind.tolist()
    # 아티팩트용은 뼈대 없이 내용만
    art = tmp_path / "art.html"
    replay.write(path, art, steps=20, every=2, standalone=False)
    body = art.read_text(encoding="utf-8").lower()
    assert "<!doctype" not in body and "<html" not in body and "<body" not in body and "<head" not in body
    assert "<title>" in body and "<canvas" in body


def test_export_widens_the_frame_gap_to_stay_under_the_size_limit(kinds_run, tmp_path):
    path = ckpt_of(kinds_run, 400)
    out = tmp_path / "a.html"
    rep = replay.write(path, out, steps=200, every=2)
    full = out.stat().st_size
    assert rep.every == 2
    # 상한은 "장면이 차지하는 몫의 1/3을 덜어야 맞는" 크기로 잡는다(화면 틀은 간격을 넓혀도 줄지 않는다)
    frames = sum(v.nbytes for v in rep.cols.values()) * 4 // 3
    limit = full - frames // 3
    rep = replay.write(path, out, steps=200, every=2, max_bytes=limit)
    assert rep.every > 2 and out.stat().st_size <= limit
    with pytest.raises(ValueError, match="크기"):
        replay.write(path, tmp_path / "b.html", steps=200, every=2, max_bytes=2000)
    assert not (tmp_path / "b.html").exists()


def test_command_line_writes_under_runs_by_default(kinds_run, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert replay.main(["--ckpt", str(ckpt_of(kinds_run, 400)), "--steps", "40"]) == 0
    out = tmp_path / "runs" / "replay-kinds-world.html"
    assert out.exists() and out.stat().st_size > 10000
    assert "replay-kinds-world.html" in capsys.readouterr().out


# ---------------------------------------------------------------- 화면의 계산 (node가 있으면)
NODE = shutil.which("node")
SCREEN_CHECK = Path(__file__).with_name("replay_screen_check.js")


@pytest.mark.skipif(NODE is None, reason="node가 없어 화면의 계산을 돌릴 수 없다")
def test_screen_logic_agrees_with_the_recording(kinds_run, tmp_path):
    """화면이 자료에서 스스로 만드는 것(일생 자막, 사건, 최근 구간의 숫자)이 적은 자료와 맞는지 node로 본다."""
    out = tmp_path / "screen.html"
    rep = replay.write(ckpt_of(kinds_run, 400), out, steps=300, every=2)
    run = subprocess.run([NODE, str(SCREEN_CHECK), str(out)], capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr[-2000:]
    got = json.loads(run.stdout)
    ag, n_frames, n_agents = rep.agents, rep.n_frames, len(rep.agents)
    assert got["nF"] == n_frames and got["nA"] == n_agents
    idx = rep.cols["idx"].astype(np.int64)
    flags = rep.cols["flags"].astype(np.int64)
    row_frame = np.repeat(np.arange(n_frames), rep.pop)
    first = np.full(n_agents, n_frames)
    np.minimum.at(first, idx, row_frame)
    first[first == n_frames] = -1
    last = np.full(n_agents, -1)
    np.maximum.at(last, idx, row_frame)
    good = np.bincount(idx, weights=flags & 3, minlength=n_agents).astype(np.int64)
    poison = np.bincount(idx, weights=(flags >> 2) & 3, minlength=n_agents).astype(np.int64)
    natural = (ag["parent"] >= 0) & (ag["kind"] == k.BIRTH_NATURAL)
    kids = np.bincount(ag["parent"][natural], minlength=n_agents)
    assert good.sum() > 0 and poison.sum() > 0 and kids.sum() > 0 and (ag["died"] >= 0).sum() > 0
    assert kids.tolist() == ag["kids"].tolist()

    # 개체마다: 처음·마지막으로 보이는 장면, 일생 자막의 끼니 수·독 수·자식 수, 나고 죽은 스텝
    for n, a in enumerate(got["agents"]):
        assert (a["first"], a["last"]) == (first[n], last[n]), n
        assert (a["good"], a["poison"], a["kids"]) == (good[n], poison[n], kids[n]), n
        assert a["sorted"] and a["texts"] and a["inRange"], n
        assert a["born"] == (None if ag["kind"][n] == replay.BORN_BEFORE else int(ag["born"][n])), n
        assert a["died"] == (None if ag["died"][n] < 0 else int(ag["died"][n])), n
        assert a["cause"] == (None if ag["died"][n] < 0 else int(ag["cause"][n])), n
    # 보이는 장면은 "born + 1부터 died까지의 상태"와 같다
    seen = first >= 0
    lo = np.maximum(0, -(-(ag["born"] + 1 - rep.start) // rep.every))
    hi = np.where(ag["died"] >= 0, (ag["died"] - rep.start) // rep.every, n_frames - 1)
    assert first[seen].tolist() == lo[seen].tolist() and last[seen].tolist() == hi[seen].tolist()
    assert not (lo <= hi)[~seen].any(), "장면에 안 보인 개체는 장면 사이에만 살았다"

    # 한 장면에서 개체 찾기
    mid = got["mid"]
    a0, a1 = int(rep.off[mid]), int(rep.off[mid + 1])
    assert got["found"] == [[int(i), int(x), int(y)] for i, x, y in
                            zip(rep.cols["idx"][a0:a1], rep.cols["x"][a0:a1], rep.cols["y"][a0:a1])]

    # 최근 구간(100스텝 = 50장면)의 숫자
    for f, w in got["windows"].items():
        f = int(f)
        lo_f = max(0, f - 49)
        assert w["eaten"] == int(rep.c_eaten[lo_f:f + 1].sum())
        assert w["poisoned"] == int(rep.c_poisoned[lo_f:f + 1].sum())
        assert w["gifts"] == int(rep.gifts[lo_f:f + 1].sum()) and w["attacks"] == int(rep.attacks[lo_f:f + 1].sum())
        assert w["blind"] == pytest.approx(float(rep.blind[lo_f:f + 1].mean()), abs=1e-6)

    # 지도에 뜨는 사건: 장면 순으로 놓이고, 수가 자료와 같다
    ev, t = got["events"], got["evTypes"]
    assert got["evSorted"]
    assert ev.get(str(t["good"]), 0) == int(((flags & 3) > 0).sum())
    assert ev.get(str(t["poison"]), 0) == int(((flags & 12) > 0).sum())
    assert ev.get(str(t["gave"]), 0) == int(((flags & replay.F_GAVE) > 0).sum())
    assert ev.get(str(t["hit"]), 0) == int(((flags & replay.F_HIT) > 0).sum())
    assert ev.get(str(t["born"]), 0) == int(((ag["kind"] != replay.BORN_BEFORE) & seen).sum())
    assert ev.get(str(t["died"]), 0) == int(((ag["died"] >= 0) & seen & (last + 1 <= n_frames - 1)).sum())

    # 사람 말 한 줄
    # 한 구간만 보면 "덜 먹는다"까지만 말하고, 독이 바뀌기 앞 구간과 견주어 독을 가리는지 모양을 가리는지 가른다
    tones = [v["tone"] for v in got["verdicts"]]
    assert tones == ["flat", "maybe", "bad", "few", "plain", "good", "shape", "shape", "maybe"]
    assert "가려 먹지 않습니다" in got["verdicts"][0]["text"] and all(v["text"] for v in got["verdicts"])
    # 구간: 독이 바뀐 장면부터 새 구간이고, 숫자는 그다음 장면부터 센다(바뀐 장면의 계수기에는 앞 구간의 끼니가 섞인다)
    cuts = [0] + [e["frame"] for e in rep.meta["flips"] if 0 < e["frame"] <= n_frames - 1]
    assert len(cuts) >= 2, "시험 세계는 다시 보는 동안 독이 한 번은 바뀐다"
    for f, seg in zip([0, 1, mid, n_frames - 1], got["segs"]):
        j = max(i for i, c in enumerate(cuts) if c <= f)
        assert seg["a"] == cuts[j] + (1 if j else 0)
        assert seg["prev"] == (None if j == 0 else {"a": cuts[j - 1] + (1 if j > 1 else 0), "b": cuts[j] - 1})
    assert got["frameOf"] == [0, 0, 1, 1, 2, n_frames - 1]
    dark, navy, white = got["lifted"]
    lum = lambda c: 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]   # noqa: E731
    assert lum(dark) >= 100 and lum(navy) >= 100 and white == [250, 250, 250]
    assert navy[2] > navy[0], "어두운 색을 밝혀도 빛깔은 남는다"


# ---------------------------------------------------------------- 들기
def _carry_cfg(**kw) -> Config:
    return small("ext", **kw).with_overrides(["carry_enabled=True"])


def test_replay_tracks_carried_item(tmp_path):
    # 마음 꺼진 carry 세계, 개체 하나 곁에 들판 알 하나. 몇 스텝 들고 움직인 뒤 내려놓게 출력을 맞추고 적는다 →
    # 그 알의 첫 행이 WHY_PICKED로 닫히고, 내려놓은 자리에서 행 하나가 열린다(합계 두 행), 든 동안 그 개체 장면의 깃발에 F_CARRY,
    # seen["unknown"] == 0, seen["picked"] == 1, seen["dropped"] == 1
    w = World(_carry_cfg(init_agents=1, min_agents=0, food_inflow=0.0))
    s = w.s
    i = int(np.flatnonzero(s.alive)[0])
    s.x[i], s.y[i], s.heading[i] = 100.0, 100.0, 0.0
    s.vx[i] = s.vy[i] = 0.0
    s.ledger[k.L_INJECT] += w.cfg.e_max + 1.0 - s.energy[i]
    s.energy[i] = w.cfg.e_max + 1.0             # 배가 불러 먹지 않는다
    s.Wout[i] = 0.0
    s.bout[i] = 0.0
    s.bout[i, ACT_REPRO] = -5.0
    s.bout[i, ACT_CARRY] = 5.0
    s.bout[i, ACT_THRUST] = 2.0
    assert k._place_food(s, 100.5, 100.0, 1.0, k.KIND_A, 0, 1)
    s.ledger[k.L_SPAWN] += 1.0
    j = int(np.flatnonzero(s.f_alive)[0])
    rec = replay.Recorder(w, every=1)
    rec.advance(5)                              # 첫 스텝에 들고 네 스텝 들고 간다
    assert s.a_held[i] == j
    s.bout[i, ACT_CARRY] = -5.0
    s.bout[i, ACT_THRUST] = 0.0
    rec.advance(1)                              # 여섯째 스텝에 내려놓는다
    drop_x, drop_y = float(s.fx[j]), float(s.fy[j])
    assert s.a_held[i] == -1 and drop_x > 101.0     # 처음 자리(100.5)에서 옮겨 내려놓았다
    rec.advance(3)
    rep = rec.finish()
    food = rep.food
    assert len(food) == 2
    first, second = food[0], food[1]
    assert (first["f_in"], first["f_out"], first["why"]) == (0, 1, replay.WHY_PICKED)
    assert (second["f_in"], second["f_out"], second["why"]) == (6, rep.n_frames, replay.WHY_PRESENT)
    assert second["x"] == min(int(drop_x / w.cfg.world_w * 65536.0), 65535)
    assert second["y"] == min(int(drop_y / w.cfg.world_h * 65536.0), 65535)
    flags = [int(rep.frame(f).flags[0]) for f in range(rep.n_frames)]
    assert [bool(fl & replay.F_CARRY) for fl in flags] == [False] + [True] * 5 + [False] * 4
    assert rep.seen["unknown"] == 0 and rep.seen["picked"] == 1 and rep.seen["dropped"] == 1
    assert rep.seen["stored_out"] == 0 and rep.meta["on"]["carry"]


def test_replay_reads_item_lifted_and_put_back_at_the_same_spot():
    # 스텝 앞에 들판에 놓여 있던 알이 스텝 뒤에는 손이 비어 있고 자리도 그대로이지만 f_dropped가 0에서 1로 바뀌었다
    # (들었다가 같은 자리에 내려놓음) → 들림 한 번과 내려놓음 한 번이 적힌다
    w = World(_carry_cfg(init_agents=1, min_agents=0, food_inflow=0.0))
    s = w.s
    assert k._place_food(s, 100.5, 100.0, 1.0, k.KIND_A, 0, 1)
    s.ledger[k.L_SPAWN] += 1.0
    j = int(np.flatnonzero(s.f_alive)[0])
    rec = replay.Recorder(w, every=1)
    pre = rec._food_snapshot()
    assert pre[7][j] == 0 and pre[6][j] == -1
    s.f_dropped[j] = 1                          # 손과 자리는 그대로, 들었다 놓은 흔적만 남았다
    rec._read_food(w.step_no, pre, w.cfg, 0, 0)
    assert rec.seen["picked"] == 1 and rec.seen["dropped"] == 1 and rec.seen["unknown"] == 0


def test_replay_store_eviction_is_not_unknown(tmp_path):
    # max_agents를 작게 해 저장 칸이 차게 만든 carry 세계를 적는다 → C_STORE_EVICTED 증가분만큼 WHY_STORED_OUT, unknown 늘지 않음
    w = World(_carry_cfg(max_agents=30, init_agents=12, min_agents=0))
    s = w.s
    gen = np.random.default_rng(4)
    for n in range(30):                         # 저장 칸을 내려놓은 알로 채운다(오래된 것부터 밀려난다)
        assert k._place_food(s, float(gen.random() * 256.0), float(gen.random() * 256.0), 1.0, k.KIND_A, -100 + n, 1)
        s.ledger[k.L_SPAWN] += 1.0
    stored = np.flatnonzero(s.f_alive & (s.f_born < 0))
    s.f_dropped[stored] = 1
    w.recount_field()
    assert s.counters[k.C_NFOOD] - s.counters[k.C_NFIELD] == 30
    s.bout[s.alive, ACT_CARRY] = 5.0           # 모두 들려고 한다
    c0 = s.counters.copy()
    rec = replay.Recorder(w, every=1)
    rec.advance(150)
    rep = rec.finish()
    dc = s.counters - c0
    n_out = int(dc[k.C_STORE_EVICTED])
    assert n_out > 0 and int(dc[k.C_CARRIES]) >= n_out
    assert rep.seen["stored_out"] == n_out and int((rep.food["why"] == replay.WHY_STORED_OUT).sum()) == n_out
    assert rep.seen["unknown"] == 0
    assert rep.seen["picked"] == int(dc[k.C_CARRIES])


def test_replay_scenario_turning_carry_off_records_release(tmp_path):
    # 각본이 들기를 끈 스텝의 풀림(release_all)은 내려놓음으로 적힌다: 든 알마다 든 자리에서 행이 다시 열린다
    cfg = _carry_cfg(init_agents=40, min_agents=0, food_inflow=0.0)
    w = World(cfg)
    s = w.s
    s.Wout[:, ACT_CARRY] = 0.0
    s.bout[:, ACT_CARRY] = -5.0                 # 다른 개체는 들지 않는다
    for i in np.flatnonzero(s.alive)[:5].tolist():
        assert k._place_food(s, float(s.x[i]), float(s.y[i]), 1.0, k.KIND_A, 0, 1)
        s.ledger[k.L_SPAWN] += 1.0
        s.bout[i, ACT_CARRY] = 5.0
        s.Wout[i, ACT_REPRO] = 0.0
        s.bout[i, ACT_REPRO] = -5.0             # 낳지 않고 배가 불러 먹지 않는다(내려놓은 알이 남는다)
        s.ledger[k.L_INJECT] += cfg.e_max + 1.0 - s.energy[i]
        s.energy[i] = cfg.e_max + 1.0
    scn = Scenario.from_dict({"name": "들기 끔", "events": [{"at": 3, "set": {"carry_enabled": False}}]})
    rec = replay.Recorder(w, every=1, scenario=scn, base_cfg=cfg, scenario_start=0)
    rec.advance(2)
    held = int((s.a_held >= 0).sum())
    assert held == 5
    rec.advance(4)
    assert not w.cfg.carry_enabled and (s.a_held == -1).all()
    rep = rec.finish()
    assert rep.seen["dropped"] == held and rep.seen["unknown"] == 0
    reopened = rep.food[(rep.food["f_in"] == 4) & (~k.always_edible_mask(rep.food["kind"]))]
    assert len(reopened) == held and (reopened["why"] == replay.WHY_PRESENT).all()


# ---------------------------------------------------------------- 고기·동물
def _hunting_world() -> World:
    """먹이 종류가 켜진 사냥감 세계. 200스텝 뒤 산 개체 모두의 공격 편향을 +3으로 두고, 산 사냥감마다 개체 곁으로 옮긴다."""
    w = World(ext_small("ext", prey_enabled=True, prey_hp=0.5, food_kinds_enabled=True))
    w.run(200)
    s = w.s
    s.bout[s.alive, ACT_ATTACK] = 3.0
    ag = np.flatnonzero(s.alive)
    for q, j in enumerate(np.flatnonzero(s.an_alive)):
        s.an_x[j], s.an_y[j] = s.x[ag[q]], s.y[ag[q]]
    return w


def test_replay_records_animals_and_meat():
    # 사냥감을 개체 곁에 두고 공격 편향을 +3으로 둔 세계를 Recorder로 120스텝 적는다(장면마다 한 스텝)
    steps = 120
    ref = _hunting_world()                       # 같은 세계를 따로 돌려, 장면마다 산 동물 수를 센다
    live = [(int(ref.s.an_alive.sum()))]
    c0 = ref.s.counters.copy()
    for _ in range(steps):
        ref.step()
        live.append(int(ref.s.an_alive.sum()))
    kills = int((ref.s.counters - c0)[k.C_PREY_KILLED])
    w = _hunting_world()
    rec = replay.Recorder(w, every=1)
    rec.advance(steps)
    rep = rec.finish()
    assert kills > 0 and max(live) > 0
    an_off = rep.animals["an_off"]
    assert an_off.dtype == np.dtype("<u4") and len(an_off) == rep.n_frames + 1
    assert np.diff(an_off).tolist() == live, "장면마다 동물 수 = 그 장면의 산 동물 수"
    n = int(an_off[-1])
    for name in ("an_x", "an_y", "an_kind", "an_hp"):
        assert len(rep.animals[name]) == n
    assert rep.animals["an_x"].dtype == np.dtype("<u2") and rep.animals["an_kind"].dtype == np.dtype("u1")
    assert (rep.animals["an_kind"] == an.ANIMAL_PREY).all()
    hp = rep.animals["an_hp"]
    assert hp.dtype == np.dtype("u1") and hp.max() == 255 and hp.min() < 255, "체력은 prey_hp에 대한 비율"
    # 마지막 장면의 동물 자리는 세계의 것과 같다
    last = np.flatnonzero(w.s.an_alive)
    got = rep.animals["an_x"][an_off[-2]:an_off[-1]]
    assert sorted(got.tolist()) == sorted(replay._q16(w.s.an_x[last], w.cfg.world_w).tolist())
    # 쓰러지면 고기 24알이 열리고, 먹은 고기는 독이 아니라 좋은 먹이로 적힌다
    meat = rep.food[rep.food["kind"] == k.KIND_MEAT]
    assert len(meat) == 24 * kills
    why = meat["why"]
    assert int((why == replay.WHY_GOOD).sum()) > 0
    assert int((why == replay.WHY_POISON).sum()) == 0 and int((why == replay.WHY_UNKNOWN).sum()) == 0
    assert rep.seen["poison"] == int((rep.food["why"] == replay.WHY_POISON).sum())
    assert rep.seen["unknown"] == 0
    assert rep.meta["on"]["prey"] and not rep.meta["on"]["huddle"] and rep.meta["cfg"]["prey_radius"] == w.cfg.prey_radius
    meta, _ = replay.pack(rep)
    for name in ("an_off", "an_x", "an_y", "an_kind", "an_hp"):
        assert name in meta["arrays"]
    assert meta["arrays"]["an_x"][2] == n
    # 세계의 먹이 집계: 고기는 독이 아니므로 눈 감고 먹은 독 비율에 들지 않는다
    fa = w.s.f_alive
    bad = int((~k.always_edible_mask(w.s.f_kind[fa]) & (w.s.f_kind[fa] != rec._good)).sum())
    assert rep.blind[-1] == pytest.approx(bad / max(int(fa.sum()), 1), abs=1e-6)


def test_replay_tells_spoiled_meat_from_eaten():
    # 하한(0.05) 바로 위의 오래된 고기 한 알 곁에 개체 하나. 고기는 부패가 3배라 한 스텝에 하한 아래로 떨어지고
    # (e 0.052 -> 0.0489), 배수를 빼면 0.0510으로 남아 먹힌다. 다시 보기는 썩음으로 적어야 한다
    w = World(ext_small("ext", prey_enabled=True, n_prey=0, spoil_enabled=True, spoil_after=60, spoil_rate=0.02,
                        init_agents=1, min_agents=0, food_inflow=0.0))
    a = int(np.flatnonzero(w.s.alive)[0])
    w.s.x[a], w.s.y[a] = 10.0, 10.0
    j = _place(w, e=0.052, born=-1000, kind=k.KIND_MEAT)
    rec = replay.Recorder(w, every=1)
    rec.advance(1)
    rep = rec.finish()
    assert not w.s.f_alive[j] and w.s.counters[k.C_SPOILED] == 1
    assert rep.seen["spoiled"] == 1 and rep.seen["good"] == 0 and rep.seen["unknown"] == 0
    assert int((rep.food["why"] == replay.WHY_SPOILED).sum()) == 1


def test_replay_without_animals_has_empty_animal_arrays(kinds_run):
    rep = replay.record(ckpt_of(kinds_run, 400), steps=40, every=2)
    assert (rep.animals["an_off"] == 0).all() and len(rep.animals["an_off"]) == rep.n_frames + 1
    for name in ("an_x", "an_y", "an_kind", "an_hp"):
        assert len(rep.animals[name]) == 0
    meta, _ = replay.pack(rep)
    for name in ("an_off", "an_x", "an_y", "an_kind", "an_hp"):
        assert name in meta["arrays"]
    assert meta["arrays"]["an_x"][2] == 0 and meta["arrays"]["an_off"][2] == rep.n_frames + 1
    assert not meta["on"]["prey"] and not meta["on"]["huddle"]
