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
from taste_judge import (bootstrap_ci, eat_tap, flip_deltas, last_flip_at_or_before, main,  # noqa: E402
                         paired_diff, pooled_ratio, ratio_diff_ci, split_children)


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
    v, lo, hi = ratio_diff_ci(a, b)
    assert v == pytest.approx(0.0)
    assert lo <= -0.4 and hi >= 0.4          # 묶음 둘을 다시 뽑으므로 끝이 ±0.5 근처(새끼 단위였다면 좁다)
    assert ratio_diff_ci(a, b) == (v, lo, hi)


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
