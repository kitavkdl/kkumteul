"""맛 기억 판정 도구 시험: 바뀜별 구간 값, 짝지은 차이, 부트스트랩, 새끼 가르기, 먹기 엿보기.

    python3 -m pytest -q tests/test_taste_judge.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "wayfinder" / "stage2-4"))

from test_harsh import _place  # noqa: E402
from test_stages import small  # noqa: E402
from test_taste import taste_world  # noqa: E402
from kkumteul import checkpoint as ckpt  # noqa: E402
from kkumteul import kernels as k  # noqa: E402
from kkumteul.run import Runner  # noqa: E402
from kkumteul.config import FIRST_MEALS  # noqa: E402
from kkumteul.world import World  # noqa: E402
from taste_judge import (book_births, book_meals, bootstrap_ci, child_book, child_rows, eat_tap,  # noqa: E402
                         first_meals, flip_deltas, last_flip_at_or_before, main, paired_diff, pooled_ratio,
                         ratio_diff_ci, run_last_flip, split_children)


def test_flip_deltas_windows():
    steps = np.arange(0, 6000, 10)
    good = np.where(steps < 1500, 0, np.where(steps < 4500, 1, 0))
    blind = np.where((steps >= 1500) & (steps < 1800), 0.3, 0.6)
    d = flip_deltas(steps, good, blind)
    assert [x["flip"] for x in d] == [1500]           # 4500의 바뀜은 다음 바뀜이 없어 빠진다
    assert d[0]["early"] == pytest.approx(0.3) and d[0]["late"] == pytest.approx(0.6)
    assert d[0]["delta"] == pytest.approx(0.3)


def test_flip_deltas_skips_short_gaps():
    # 바뀜 1000, 1500(간격 500 < 2 × 300), 3000, 5000. 1000은 다음 바뀜이 가까워 빠지고 1500, 3000은 남는다
    steps = np.arange(0, 6000, 10)
    good = np.select([steps < 1000, steps < 1500, steps < 3000, steps < 5000], [0, 1, 0, 1], 0)
    blind = np.full(len(steps), 0.5)
    d = flip_deltas(steps, good, blind)
    assert [x["flip"] for x in d] == [1500, 3000]
    assert flip_deltas(steps, good, blind, win=200)[0]["flip"] == 1000    # 간격 500 ≥ 2 × 200


def test_paired_diff_matches_by_flip():
    on = [{"flip": 1500, "delta": 0.2}, {"flip": 4500, "delta": 0.1}]
    off = [{"flip": 4500, "delta": 0.0}, {"flip": 1500, "delta": 0.05}]
    assert paired_diff(on, off).tolist() == pytest.approx([0.15, 0.1])
    assert paired_diff(on, off[:1]).tolist() == pytest.approx([0.1])      # 한쪽에만 있는 바뀜은 뺀다


def test_bootstrap_ci_brackets_mean_and_is_repeatable():
    x = np.array([0.1, 0.12, 0.08, 0.11])
    a = bootstrap_ci(x)
    assert a == bootstrap_ci(x)
    assert a[1] <= a[0] <= a[2] and a[0] == pytest.approx(x.mean())
    assert a[1] < a[2]


def test_split_children_and_pooled_ratio():
    rows = [{"since_flip": 500, "meals": 10, "poison": 2}, {"since_flip": 100, "meals": 10, "poison": 6},
            {"since_flip": 900, "meals": 3, "poison": 1}, {"since_flip": None, "meals": 0, "poison": 0}]
    sp = split_children(rows)
    assert len(sp["old"]) == 3 and len(sp["fresh"]) == 1
    assert pooled_ratio(sp["old"]) == pytest.approx(3 / 13)    # 10끼 못 채운 새끼도 먹은 만큼 들어간다
    assert pooled_ratio(sp["fresh"]) == pytest.approx(0.6)
    edge = split_children([{"since_flip": 300, "meals": 1, "poison": 0}, {"since_flip": 301, "meals": 1, "poison": 0},
                           {"since_flip": 0, "meals": 1, "poison": 0}])
    assert len(edge["fresh"]) == 2 and len(edge["old"]) == 1


def test_last_flip_looks_back_before_run_start():
    period = 1500

    def flips_at(sd, b):
        return k.good_kind(sd, b * period, period) != k.good_kind(sd, b * period - 1, period)

    # 경계 b0에서 실제로 바뀌고, 그다음 두 경계(b0+1, b0+2)는 다시 정했지만 그대로인 씨앗
    b0 = 3
    sd = next(sd for sd in range(1, 10 ** 6)
              if flips_at(sd, b0) and not flips_at(sd, b0 + 1) and not flips_at(sd, b0 + 2))
    f = b0 * period
    assert last_flip_at_or_before(sd, f + 100, period) == f
    assert last_flip_at_or_before(sd, f, period) == f                        # 바로 그 스텝도 넣는다
    assert last_flip_at_or_before(sd, (b0 + 2) * period + 100, period) == f  # 안 바뀐 경계 둘을 건너뛴다
    assert (last_flip_at_or_before(sd, f - 1, period) or -1) < f
    assert last_flip_at_or_before(sd, (b0 + 2) * period + 100, period, lookback=2) is None


def test_ratio_diff_ci_resamples_clusters():
    # 무리 a: 묶음 하나는 새끼 100마리 모두 독 10/10, 다른 묶음 100마리는 0/10. 무리 b: 묶음 둘 다 5/10
    a = ([{"cluster": (1, 0), "meals": 10, "poison": 10}] * 100
         + [{"cluster": (1, 3000), "meals": 10, "poison": 0}] * 100)
    b = ([{"cluster": (1, 0), "meals": 10, "poison": 5}] * 100
         + [{"cluster": (2, 0), "meals": 10, "poison": 5}] * 100)
    v, lo, hi, dropped = ratio_diff_ci(a, b)
    assert v == pytest.approx(0.0) and dropped == 0
    assert lo <= -0.4 and hi >= 0.4          # 묶음 둘을 다시 뽑으므로 끝이 ±0.5 근처(새끼 단위였다면 좁다)
    assert ratio_diff_ci(a, b) == (v, lo, hi, dropped)
    # 끼니 0인 묶음만 뽑힌 회차는 비율이 없어 버린다. 버린 수를 함께 돌려준다(묶음 둘 중 하나만 뽑힐 확률 1/4)
    z = [{"cluster": (1, 0), "meals": 10, "poison": 5}, {"cluster": (1, 1), "meals": 0, "poison": 0}]
    dropped = ratio_diff_ci(z, b, reps=4000)[3]
    assert 800 < dropped < 1200


def test_eat_tap_sees_meals_before_death():
    w = taste_world("taste", init_agents=1, min_agents=0, food_inflow=0.0)
    s = w.s
    assert s.alive[0] and s.alive.sum() == 1
    s.energy[0] = 0.5
    bad = 1 - k.good_kind(w.cfg.seed, w.step_no, w.cfg.kind_flip_period)
    _place(w, kind=bad, x=float(s.x[0]), y=float(s.y[0]))
    seen = []
    orig = k.eat
    with eat_tap(w, lambda st: seen.append((bool(st.alive[0]), int(st.n_meal[0]), int(st.n_poison10[0])))):
        w.run(1)
    assert k.eat is orig                     # 블록을 나가면 되돌린다
    assert seen == [(True, 1, 1)]            # 먹기 직후: 아직 살아 있고 독 한 끼
    assert not s.alive[0]                    # 스텝 뒤에는 죽어 있다
    w.run(1)
    assert len(seen) == 1                    # 블록 밖에서는 부르지 않는다


def test_commands_run_on_tiny_checkpoint(tmp_path, capsys):
    # 아주 작은 체크포인트로 두 하위 명령을 끝까지 부른다(인자·읽기 길만 본다. 판정 값은 보지 않는다)
    r = Runner.new(tmp_path / "r", small("stage2", "kinds", "taste", kind_flip_period=50))
    r.loop(200, quiet=True)
    r.close()
    p = ckpt.latest(tmp_path / "r" / "checkpoints")
    out = tmp_path / "flip.json"
    assert main(["flip-diff", str(p), "--steps", "400", "--every", "5", "--win", "10", "--json", str(out)]) == 0
    res = json.loads(out.read_text())
    assert {"n_flips", "flips", "mean", "lo", "hi", "verdict"} <= set(res) and res["expect"] == "gain"
    assert res["n_flips"] == len(res["flips"])
    out2 = tmp_path / "meals.json"
    assert main(["first-meals", str(p), "--steps", "60", "--seeds", "1,2", "--win", "10", "--json", str(out2)]) == 0
    res2 = json.loads(out2.read_text())
    assert set(res2["sides"]) == {"on", "off"}
    for side in res2["sides"].values():
        assert {"n_children", "reseed_excluded", "n_old", "n_fresh", "pop_mean", "energy_mean"} <= set(side)
    assert "old" in res2 and "fresh" in res2 and "verdict" in res2
    assert "바뀜" in capsys.readouterr().out


def test_run_last_flip_counts_hidden_flip_at_read_step():
    # 씨앗을 바꿔 읽으면 읽는 스텝 t0에서 좋은 종류가 원래 일정(t0 − 1)과 달라질 수 있다. 그러면 t0이 바뀜이다
    period, t0, orig = 1500, 3700, 1

    def g(sd, t):
        return k.good_kind(sd, t, period)

    def flips_at(sd, b):
        return g(sd, b * period) != g(sd, b * period - 1)

    before = g(orig, t0 - 1)
    # 둘 다 새 일정의 3000 경계는 그대로라 t0 직후 새끼의 새 일정상 마지막 바뀜은 1500 이하(멀다)
    differ = next(sd for sd in range(2, 10 ** 6) if g(sd, t0) != before and not flips_at(sd, 2) and flips_at(sd, 3))
    same = next(sd for sd in range(2, 10 ** 6) if g(sd, t0) == before and not flips_at(sd, 2))
    t = t0 + 100
    assert (last_flip_at_or_before(differ, t, period) or -1) <= period
    assert run_last_flip(orig, differ, t0, t, period) == t0
    assert run_last_flip(orig, differ, t0, 3 * period + 100, period) == 3 * period    # 새 일정의 더 늦은 바뀜
    assert run_last_flip(orig, same, t0, t, period) == last_flip_at_or_before(same, t, period)
    assert run_last_flip(orig, orig, t0, t, period) == last_flip_at_or_before(orig, t, period)
    b = child_book(2)
    book_births(b, np.array([0]), np.array([7]), t, False)
    rows = child_rows(b, orig, differ, t0, period)
    assert rows[0]["since_flip"] == 100 and rows[0]["cluster"] == (differ, t0)
    assert len(split_children(rows, win=300)["fresh"]) == 1
    assert len(split_children(child_rows(b, orig, same, t0, period), win=300)["old"]) == 1


def _book_rows(b) -> list[tuple[int, int, int]]:
    n = b["n"]
    return list(zip(b["born"][:n].tolist(), b["meals"][:n].tolist(), b["poison"][:n].tolist()))


def test_child_book_keeps_dead_masks_reused_slots_and_skips_reseed():
    n = 6
    b = child_book(n)
    alive = np.ones(n, dtype=bool)
    ids = np.arange(100, 100 + n)
    meal = np.zeros(n, dtype=np.int64)
    poison = np.zeros(n, dtype=np.int64)
    book_births(b, np.array([2, 4]), ids[[2, 4]], 5, False)
    meal[[2, 4]], poison[[2, 4]] = (3, 12), (1, 4)
    book_meals(b, alive, ids, meal, poison)
    assert _book_rows(b) == [(5, 3, 1), (5, FIRST_MEALS, 4)]
    # 칸 2의 새끼가 죽는다: 그 줄은 먹은 만큼 남는다
    alive[2], meal[2], poison[2] = False, 0, 0
    book_meals(b, alive, ids, meal, poison)
    assert _book_rows(b) == [(5, 3, 1), (5, FIRST_MEALS, 4)]
    # 같은 스텝의 출생이 칸 2를 다시 쓴다(번호 200): 새 줄이 생기고 앞 새끼 줄은 그대로
    alive[2], ids[2] = True, 200
    book_births(b, np.array([2]), ids[[2]], 9, False)
    meal[2], poison[2] = 2, 2
    book_meals(b, alive, ids, meal, poison)
    assert _book_rows(b) == [(5, 3, 1), (5, FIRST_MEALS, 4), (9, 2, 2)]
    # 재주입이 난 스텝: 그 스텝 새끼 둘(칸 4를 다시 쓴 번호 300, 칸 5의 301)은 올리지 않고 세기만 한다.
    # 칸 4의 앞 새끼 줄은 번호가 달라 가려져 바뀌지 않는다
    ids[4], ids[5] = 300, 301
    book_births(b, np.array([4, 5]), ids[[4, 5]], 11, True)
    meal[4], poison[4], meal[5], poison[5] = 1, 1, 1, 0
    book_meals(b, alive, ids, meal, poison)
    assert _book_rows(b) == [(5, 3, 1), (5, FIRST_MEALS, 4), (9, 2, 2)] and b["excluded"] == 2
    # 줄이 1024를 넘어도 앞 줄을 지키며 늘어난다
    many = child_book(2000)
    for t in range(1100):
        book_births(many, np.array([t]), np.array([t]), t, False)
    assert many["n"] == 1100 and many["born"][1099] == 1099 and many["born"][0] == 0


def test_first_meals_refuses_passive_checkpoint(tmp_path):
    w = World(small("stage2", "kinds", "taste", passive=True))
    path = ckpt.save(w, tmp_path / "checkpoints")
    with pytest.raises(ValueError, match="대조 실행"):
        first_meals(path, steps=10, seeds=(1,))


def test_first_meals_warns_about_scenario(tmp_path, capsys):
    # 각본이 걸린 실행(run.json의 scenario_start)이면 first-meals는 각본을 적용하지 않는다고 알린다
    w = World(small("stage2", "kinds", "taste"))
    path = ckpt.save(w, tmp_path / "r" / "checkpoints")
    assert main(["first-meals", str(path), "--steps", "3", "--seeds", "1"]) == 0
    assert "각본" not in capsys.readouterr().out
    (tmp_path / "r" / "run.json").write_text(json.dumps({"scenario_start": 0}))
    assert main(["first-meals", str(path), "--steps", "3", "--seeds", "1"]) == 0
    assert "각본" in capsys.readouterr().out
