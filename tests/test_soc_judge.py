"""사회 학습 판정 도구 시험(과제 6): `taste_gene_judge.py --gene soc`, 옵션, 관문(gate), observe의 soc 칸.

설계 docs/superpowers/specs/2026-11-07-social-learning-design.md 4절, 계획 docs/superpowers/plans/2026-11-26-social-learning.md.
시뮬레이션 없이 도는 것(관문 순수 계산·옵션 적용)이 대부분이고, observe만 작은 세계를 잠깐 돌린다.

    python3 -m pytest -q tests/test_soc_judge.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_stages import small  # noqa: E402
from kkumteul import checkpoint as ckpt  # noqa: E402
from kkumteul import kernels as k  # noqa: E402
from kkumteul import measure  # noqa: E402
from kkumteul.world import World  # noqa: E402


def _judge():
    p = Path(__file__).resolve().parents[1] / "wayfinder" / "stage2-4" / "taste_gene_judge.py"
    spec = importlib.util.spec_from_file_location("taste_gene_judge_soc", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def soc_world(**kw) -> World:
    return World(small("stage2", "kinds", "taste", soc_taste_enabled=True, taste_gene_enabled=True, **kw))


# ---- zeroed_side ----
def test_zeroed_side_soc():
    tj = _judge()
    w = soc_world()
    w.s.soc_g[w.s.alive] = 0.03
    side = measure.side_from_world(w)
    z = tj.zeroed_side(side, "soc")
    assert np.allclose(z["genomes"]["soc_g"], -w.cfg.soc_rate)          # 실제 속도 0
    assert np.allclose(side["genomes"]["soc_g"], 0.03)                  # 원본은 그대로
    assert np.allclose(z["genomes"]["taste_g"], side["genomes"]["taste_g"])   # 개인 속도는 안 건드린다
    off = measure.side_from_world(World(small("stage2", "kinds", "taste", taste_gene_enabled=True)))
    with pytest.raises(ValueError, match="보고 배우기가 꺼진"):
        tj.zeroed_side(off, "soc")


def test_zeroed_side_unknown_gene_message():
    tj = _judge()
    side = measure.side_from_world(soc_world())
    with pytest.raises(ValueError) as e:
        tj.zeroed_side(side, "nope")
    for name in ("taste", "eta", "soc"):
        assert name in str(e.value)


# ---- 옵션 ----
def test_apply_options():
    tj = _judge()
    w = soc_world()
    w.s.taste_g[w.s.alive] = 0.1
    a = tj.zeroed_side(measure.side_from_world(w), "soc")
    b = measure.side_from_world(w)
    a2, b2 = tj.apply_options(a, b, sets=["soc_taste_mode=blind", "soc_cost=0"])
    assert a2["cfg"].soc_taste_mode == "blind" and b2["cfg"].soc_taste_mode == "blind"
    assert a2["cfg"].soc_cost == 0.0 and b2["cfg"].soc_cost == 0.0
    assert b["cfg"].soc_taste_mode == "outcome"                          # 원래 옆은 그대로
    # 속도 d: B쪽 soc_g = d - soc_rate(실제 속도 d), A쪽은 그대로
    a3, b3 = tj.apply_options(a, b, soc_d=0.25)
    assert np.allclose(b3["genomes"]["soc_g"], 0.25 - b["cfg"].soc_rate)
    assert np.allclose(a3["genomes"]["soc_g"], a["genomes"]["soc_g"])
    assert np.allclose(b["genomes"]["soc_g"], 0.0)                       # 원본 배열을 안 건드림
    # 개인 학습 없음: 두 쪽 모두 taste_g = -taste_rate
    a4, b4 = tj.apply_options(a, b, no_individual=True)
    for s in (a4, b4):
        assert np.allclose(s["genomes"]["taste_g"], -b["cfg"].taste_rate)
    assert np.allclose(b["genomes"]["taste_g"], 0.1)
    # 소수점 값 d는 옵션을 `soc_rate`를 덮어쓴 뒤의 값으로 계산한다
    a5, b5 = tj.apply_options(a, b, sets=["soc_rate=0.1"], soc_d=0.5)
    assert np.allclose(b5["genomes"]["soc_g"], 0.5 - 0.1)
    # 개인 학습 유전자가 꺼진 설정에서 --no-individual은 뜻이 없다
    nog = measure.side_from_world(World(small("stage2", "kinds", "taste", soc_taste_enabled=True)))
    with pytest.raises(ValueError):
        tj.apply_options(nog, nog, no_individual=True)


def test_parse_sets_and_negative_control_skips_enabled_check():
    tj = _judge()
    w = soc_world()
    side = measure.side_from_world(w)
    # 음성 대조: soc_taste_enabled=false를 --set으로 주면 0 벌 만들기가 "꺼짐" 오류를 내지 않는다
    a, b = tj.build_pair(side, "soc", sets=["soc_taste_enabled=false"])
    assert a["cfg"].soc_taste_enabled is False and b["cfg"].soc_taste_enabled is False
    assert np.allclose(a["genomes"]["soc_g"], -b["cfg"].soc_rate)
    # 켜지 않은 체크포인트에 옵션도 없으면 오류
    off = measure.side_from_world(World(small("stage2", "kinds", "taste", taste_gene_enabled=True)))
    with pytest.raises(ValueError, match="보고 배우기가 꺼진"):
        tj.build_pair(off, "soc")
    # 옵션으로 켜면 된다
    a, b = tj.build_pair(off, "soc", sets=["soc_taste_enabled=true"])
    assert b["cfg"].soc_taste_enabled is True


# ---- 관문(순수 계산) ----
def _res(rlist, md=None, both=0, a_ext=0, b_ext=0):
    d = {"r_per_1000": measure._stats(rlist), "r_list": list(rlist), "reps": len(rlist),
         "both_extinct_runs": both, "a_extinct_runs": a_ext, "b_extinct_runs": b_ext}
    if md is not None:
        d["min_detectable_r_per_1000"] = md
    return d


NOISE = [3.0, -2.0, 4.0, 1.0, 2.0, -1.0]
BIG = [30.0, 25.0, 40.0, 35.0, 28.0, 32.0]


def _pass_results(**over):
    r = {"outcome-d0.05": _res(NOISE, md=10.0),
         "blind-d0.05": _res(NOISE),
         "outcome-d0.25": _res(BIG),
         "blind-d0.25": _res([2.0, -3.0, 5.0, 0.0, 1.0, -1.0]),
         "off-d1.0": _res([1.0, -2.0, 3.0, -1.0, 2.0, 0.0])}
    r.update(over)
    return r


def test_gate_tag_pass():
    tj = _judge()
    g = tj.gate_tag(_pass_results())
    assert g["status"] == "pass" and g["only_noind"] is False
    assert g["min_detectable"] == 10.0
    assert g["d"]["0.25"]["pass"] is True and g["d"]["0.05"]["pass"] is False
    assert g["control"]["ok"] is True


def test_gate_tag_requires_five_of_six_and_welch():
    tj = _judge()
    four = [30.0, 25.0, -1.0, -2.0, 28.0, 32.0]                         # 평균은 양이어도 4판만 양
    g = tj.gate_tag(_pass_results(**{"outcome-d0.25": _res(four)}))
    assert g["status"] == "undetectable" and g["d"]["0.25"]["c1"] is False
    five = [30.0, 25.0, 40.0, -1.0, 28.0, 32.0]                         # 5판 양: 통과
    assert tj.gate_tag(_pass_results(**{"outcome-d0.25": _res(five)}))["d"]["0.25"]["c1"] is True
    # outcome이 blind와 같으면 ②가 안 된다
    g = tj.gate_tag(_pass_results(**{"blind-d0.25": _res(BIG)}))
    assert g["status"] == "undetectable" and g["d"]["0.25"]["c2"] is False
    # 차이가 양이고 Welch는 되지만 min_detectable 밑이면 ②가 안 된다
    g = tj.gate_tag(_pass_results(**{"outcome-d0.05": _res(NOISE, md=500.0)}))
    assert g["status"] == "undetectable" and g["d"]["0.25"]["c2"] is False


def test_gate_tag_undetectable():
    tj = _judge()
    r = _pass_results(**{"outcome-d0.25": _res(NOISE), "blind-d0.25": _res(NOISE)})
    g = tj.gate_tag(r)
    assert g["status"] == "undetectable" and g["control"]["ok"] is True
    assert not any(v["pass"] for v in g["d"].values())


def test_gate_tag_broken():
    tj = _judge()
    g = tj.gate_tag(_pass_results(**{"off-d1.0": _res([40.0, 35.0, 42.0, 38.0, 41.0, 39.0])}))
    assert g["status"] == "broken" and "대조" in g["reason"]             # 음성 대조가 0에서 멂
    g = tj.gate_tag(_pass_results(**{"off-d1.0": _res(NOISE, both=3)}))
    assert g["status"] == "broken"                                       # 양쪽 전멸 3판
    g = tj.gate_tag(_pass_results(**{"off-d1.0": _res(NOISE, both=2)}))
    assert g["status"] == "pass"                                         # 2판까지는 허용
    r = _pass_results()
    del r["off-d1.0"]
    assert tj.gate_tag(r)["status"] == "broken"                          # 음성 대조가 없으면 판정을 쓰지 않는다
    r = _pass_results()
    del r["outcome-d0.05"]
    g = tj.gate_tag(r)
    assert g["status"] == "broken" and "min_detectable" in g["reason"]


def test_gate_tag_noind_and_nocost_are_read_only_except_noind_pass():
    tj = _judge()
    undetectable = _pass_results(**{"outcome-d0.25": _res(NOISE), "blind-d0.25": _res(NOISE)})
    only_noind = dict(undetectable, **{"outcome-noind-d1.0": _res(BIG), "blind-noind-d1.0": _res(NOISE)})
    g = tj.gate_tag(only_noind)
    assert g["status"] == "pass" and g["only_noind"] is True
    nocost = dict(undetectable, **{"outcome-nocost-d0.25": _res(BIG), "blind-d0.25": _res(NOISE)})
    g = tj.gate_tag(nocost)
    assert g["status"] == "undetectable"                                 # 비용 없는 판은 읽기용
    assert g["nocost"]["0.25"]["r_outcome"] == pytest.approx(np.mean(BIG))


def test_gate_combines_tags_exit_codes():
    tj = _judge()
    ok = _pass_results()
    und = _pass_results(**{"outcome-d0.25": _res(NOISE), "blind-d0.25": _res(NOISE)})
    bad = _pass_results(**{"off-d1.0": _res(NOISE, both=4)})
    noind = dict(und, **{"outcome-noind-d1.0": _res(BIG), "blind-noind-d1.0": _res(NOISE)})
    c = tj.gate_tags({"a": ok, "b": und})
    assert c["exit_code"] == 0 and c["status"] == "pass" and c["only_noind"] is False
    assert tj.gate_tags({"a": und, "b": und})["exit_code"] == 3
    c = tj.gate_tags({"a": bad, "b": und})
    assert c["exit_code"] == 4 and c["status"] == "broken"
    assert tj.gate_tags({"a": bad, "b": ok})["exit_code"] == 0           # 한 바탕이라도 통과면 0
    c = tj.gate_tags({"a": noind, "b": und})
    assert c["exit_code"] == 0 and c["only_noind"] is True
    c = tj.gate_tags({"a": noind, "b": ok})
    assert c["exit_code"] == 0 and c["only_noind"] is False              # 비용 있는 판이 통과한 바탕이 있음
    assert set(c["tags"]) == {"a", "b"}


def test_gate_cli_reads_files(tmp_path, capsys):
    tj = _judge()
    for tag in ("a", "b"):
        for name, res in _pass_results().items():
            (tmp_path / f"j-dose-{name}-{tag}.json").write_text(json.dumps(res))
    out = tmp_path / "gate.json"
    code = tj.main(["gate", "--dir", str(tmp_path), "--tags", "a", "b", "--json", str(out)])
    assert code == 0
    got = json.loads(out.read_text())
    assert got["status"] == "pass" and set(got["tags"]) == {"a", "b"}
    # 파일이 하나도 없는 바탕은 읽을 것이 없어 "도구 깨짐"
    code = tj.main(["gate", "--dir", str(tmp_path), "--tags", "zz"])
    assert code == 4


# ---- observe의 soc 칸 ----
def test_observe_reports_soc_fields(tmp_path):
    tj = _judge()
    w = soc_world()
    w.run(200)
    w.s.soc_g[w.s.alive] = 0.05
    path = ckpt.save(w, tmp_path)
    o = tj.observe(path, steps=300)
    assert o["soc_rate"]["mean"] == pytest.approx(min(1.0, w.cfg.soc_rate + 0.05))
    for key in ("q25", "median", "q75"):
        assert key in o["soc_rate"]
    w2, _ = ckpt.load(path)
    w2.run(300)
    a = w2.s.alive
    assert o["n_seen"] == int(w2.s.n_seen[a].sum()) and o["n_seen"] > 0
    assert o["n_seen_poison"] == int(w2.s.n_seen_poison[a].sum())
    assert o["n_seen_kin"] == int(w2.s.n_seen_kin[a].sum())
    assert o["seen_age_sum"] == int(w2.s.seen_age_sum[a].sum())
    assert o["seen_poison_frac"] == pytest.approx(o["n_seen_poison"] / o["n_seen"])
    assert o["seen_kin_frac"] == pytest.approx(o["n_seen_kin"] / o["n_seen"])
    assert o["seen_per_1000_steps"] == pytest.approx(o["n_seen"] / (int(a.sum()) * 300) * 1000)


def test_observe_has_no_soc_fields_when_off(tmp_path):
    tj = _judge()
    w = World(small("stage2", "kinds", "taste"))
    w.run(100)
    path = ckpt.save(w, tmp_path)
    o = tj.observe(path, steps=100)
    assert "soc_rate" not in o and "n_seen" not in o
