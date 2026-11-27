import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "wayfinder" / "stage2-4"))
import norm_twin_compare as tc  # noqa: E402


def _w(folder: Path, name: str, d: dict):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.json").write_text(json.dumps(d))


def test_metrics_values_and_missing_tab():
    m = tc.metrics({"pop_mean": 900.0, "norm_tab": {"hit": 200.0, "hit_hawk": 50.0, "atk_vol_marked": 3.0,
                                                    "atk_vol_plain": 7.0, "expo": 1000.0}})
    assert m["hawk_frac"] == 0.25 and m["vol_rate"] == 0.01 and m["pop_mean"] == 900.0
    m0 = tc.metrics({"pop_mean": 10.0})            # off 갈래: norm_tab 없음 → 빈칸
    assert m0["hawk_frac"] is None and m0["vol_rate"] is None
    mz = tc.metrics({"norm_tab": {"hit": 0.0, "expo": 0.0}})   # 0으로 나누지 않는다
    assert mz["hawk_frac"] is None and mz["vol_rate"] is None


def test_compare_pairs_by_name_and_keeps_one_sided(tmp_path):
    p, h = tmp_path / "plain", tmp_path / "hawk"
    tab = {"hit": 10.0, "hit_hawk": 1.0, "atk_vol_marked": 0.0, "atk_vol_plain": 2.0, "expo": 100.0}
    _w(p, "j-obs-on-a", {"pop_mean": 1.0, "norm_tab": tab})
    _w(h, "j-obs-on-a", {"pop_mean": 2.0, "norm_tab": dict(tab, hit_hawk=5.0)})
    _w(p, "w-meas-a", {"pop_mean": 3.0, "norm_tab": tab})
    _w(p, "h-base-std", {"x": 1})                   # 판정 0 결과는 읽지 않는다
    rows = {r["name"]: r for r in tc.compare(p, h)}
    assert set(rows) == {"j-obs-on-a", "w-meas-a"}
    assert rows["j-obs-on-a"]["plain"]["hawk_frac"] == 0.1 and rows["j-obs-on-a"]["hawk"]["hawk_frac"] == 0.5
    assert rows["w-meas-a"]["hawk"] is None
    out = tmp_path / "o.json"
    assert tc.main([str(p), str(h), "--json", str(out)]) == 0
    assert len(json.loads(out.read_text())) == 2
