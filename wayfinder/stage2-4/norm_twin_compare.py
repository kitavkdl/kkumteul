"""규범의 씨앗 두 판(지금 구조 / 매-비둘기) 견주기 — 같은 이름의 관찰 결과를 나란히 놓는다.

읽는 것: 두 기록 폴더(runs/norm-exp-<꼬리표>)의 `w-*.json`(세계 점검 끝 5,000스텝 창)과 `j-obs-*.json`(진화 갈래 끝 무리).
내는 값(norm_tab이 있는 갈래만 — 장치를 끈 off 갈래는 계수가 없어 빈칸):
  hawk_frac = hit_hawk ÷ hit        맞은 공격 중 맞은 쪽 뇌 공격 출력 > 0(매)의 비율
  vol_rate  = (atk_vol_marked + atk_vol_plain) ÷ expo   닿는 이웃이 있던 차례당 자발 공격(충동 밖, 뇌 출력 > 0)
  pop_mean  창 평균 개체 수
판정 규칙은 results.md "매-비둘기 판 다시 설계" 절을 따른다. 이 도구는 값만 낸다(해석 안 함).

  python wayfinder/stage2-4/norm_twin_compare.py <지금 구조 기록 폴더> <매-비둘기 기록 폴더> [--json 출력]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _div(a, b):
    return a / b if b else None


def metrics(d: dict) -> dict:
    nt = d.get("norm_tab")
    out = {"pop_mean": d.get("pop_mean"), "hawk_frac": None, "vol_rate": None, "hits": None}
    if nt:
        out["hits"] = nt.get("hit", 0.0)
        out["hawk_frac"] = _div(nt.get("hit_hawk", 0.0), nt.get("hit", 0.0))
        out["vol_rate"] = _div(nt.get("atk_vol_marked", 0.0) + nt.get("atk_vol_plain", 0.0), nt.get("expo", 0.0))
    return out


def _load(folder: Path) -> dict:
    res = {}
    for pat in ("w-*.json", "j-obs-*.json"):
        for p in sorted(folder.glob(pat)):
            try:
                res[p.stem] = metrics(json.loads(p.read_text()))
            except (OSError, ValueError) as e:
                print(f"읽기 실패 {p}: {e}", file=sys.stderr)
    return res


def compare(plain: Path, hawk: Path) -> list[dict]:
    a, b = _load(plain), _load(hawk)
    rows = []
    for name in sorted(set(a) | set(b)):
        rows.append({"name": name, "plain": a.get(name), "hawk": b.get(name)})
    return rows


def _f(v, w=8, p=4):
    return f"{v:{w}.{p}f}" if isinstance(v, (int, float)) else " " * (w - 1) + "-"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("plain", type=Path)
    ap.add_argument("hawk", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args(argv)
    rows = compare(a.plain, a.hawk)
    print(f"{'이름':24s} {'매비율(지금)':>12s} {'매비율(매비)':>12s} {'자발(지금)':>10s} {'자발(매비)':>10s} {'개체(지금)':>10s} {'개체(매비)':>10s}")
    for r in rows:
        p, h = r["plain"] or {}, r["hawk"] or {}
        print(f"{r['name']:24s} {_f(p.get('hawk_frac'), 12)} {_f(h.get('hawk_frac'), 12)} {_f(p.get('vol_rate'), 10)} "
              f"{_f(h.get('vol_rate'), 10)} {_f(p.get('pop_mean'), 10, 0)} {_f(h.get('pop_mean'), 10, 0)}")
    if a.json:
        a.json.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
