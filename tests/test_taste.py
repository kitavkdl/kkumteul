"""맛 기억 장치 시험: 설정·묶음, 저장 필드, 끄면 지우기.

    python3 -m pytest -q tests/test_taste.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_stages import rel_residual, small  # noqa: E402,F401
from kkumteul import checkpoint as ckpt  # noqa: E402,F401
from kkumteul import kernels as k  # noqa: E402
from kkumteul import measure  # noqa: E402
from kkumteul.config import (ACT_TURN, FIRST_MEALS, N_EXT_IN, N_STAGE_IN, N_TASTE_IN, TASTE_VAL,  # noqa: E402,F401
                             TASTE_VIS, preset)
from kkumteul.world import World  # noqa: E402

# 맛 장치를 넣기 전 코드가 같은 크기(n_spare_in=36, n_spare_out=7) 세계를 300스텝 돌려 낸 해시.
# taste_size 묶음(장치 끈 같은 크기 세계)은 이 값을 그대로 내야 한다. 장치를 끈 세계의 계산을 일부러 바꿀 때만 고친다.
PINNED_TASTE_SIZE_HASH_300 = {
    (): "209278e89cb28d47c96dff4d06f9d6115de5949170ef7842c6ceb9e65486ae18",
    ("stage2", "kinds"): "cbd3d3d4d9e35ea02424d1fe18c453ea8f3b56f1aadc76c4ee39013d96bc1af2",
}


def taste_world(*presets, **kw) -> World:
    return World(small("stage2", "kinds", *presets, **kw))


def test_taste_config_and_presets():
    c = small("taste")
    assert c.taste_enabled and c.n_spare_in == N_STAGE_IN + N_EXT_IN + N_TASTE_IN
    assert c.taste_in0 == c.ext_in0 + N_EXT_IN and c.n_in >= c.taste_in0 + N_TASTE_IN
    z = small("taste_size")
    assert not z.taste_enabled and z.n_spare_in == c.n_spare_in and z.n_spare_out == c.n_spare_out
    with pytest.raises(ValueError, match="taste"):
        preset("nope")                      # 오류 메시지가 taste, taste_size를 알려 준다
    with pytest.raises(ValueError):
        small("ext", taste_enabled=True)    # 크기가 모자람
    with pytest.raises(ValueError):
        small("taste", vision_sectors=6)
    for name in ("taste_rate", "taste_decay", "taste_inherit"):
        with pytest.raises(ValueError):
            small(**{name: 1.5})
        with pytest.raises(ValueError):
            small(**{name: -0.1})


@pytest.mark.parametrize("presets", list(PINNED_TASTE_SIZE_HASH_300))
def test_taste_size_world_hash_pinned(presets):
    # taste_size 묶음 = 장치 끈 같은 크기 세계(장치를 넣기 전 코드로 뽑은 값)
    w = World(small(*presets, "taste_size"))
    w.run(300)
    assert w.state_hash() == PINNED_TASTE_SIZE_HASH_300[presets]


def test_taste_m_saved_and_old_checkpoint_reads_zero(tmp_path):
    # 저장 → 읽기 → 같은 해시. 0이 아닌 기억은 해시에 들어간다. taste_m 없는 배열 묶음을 읽으면 0
    # (장치 끈 크기 세계에서 한다: 켠 세계는 먹으면서 기억이 0이 아니게 되어 h0가 기억 0 상태가 아니다)
    w = taste_world("taste_size")
    w.run(50)
    h0 = w.state_hash()
    w.s.taste_m[w.s.alive] = 0.25
    assert w.state_hash() != h0
    arrays = {kk: v.copy() for kk, v in w.state_arrays().items()}
    w2 = World(w.cfg, populate=False)
    w2.step_no = w.step_no
    assert w2.load_arrays(arrays) and w2.state_hash() == w.state_hash()
    del arrays["taste_m"]
    w3 = World(w.cfg, populate=False)
    w3.step_no = w.step_no
    assert w3.load_arrays(arrays)           # 빠진 것으로 치지 않는다(HASH_DEFAULTS)
    assert (w3.s.taste_m == 0).all() and w3.state_hash() == h0


def test_set_config_off_clears_taste():
    w = taste_world("taste")
    w.s.taste_m[:] = 0.5
    w.set_config(w.cfg.with_overrides(["taste_enabled=false"]))
    assert (w.s.taste_m == 0).all() and w.ip[k.Q_TASTE_IN0] == -1


def test_inject_agent_starts_with_blank_taste():
    w = taste_world("taste")
    w.s.taste_m[:] = 0.5
    w.s.n_meal[:] = 7
    w.s.n_poison[:] = 3
    w.s.n_poison10[:] = 3
    side = measure.side_from_world(w)
    j_id = measure.inject_agent(w, side["genomes"], 0, 1, 10.0, 10.0, 0.0, 0)
    j = int(np.flatnonzero(w.s.ids == j_id)[0])
    assert (w.s.taste_m[j] == 0).all() and w.s.n_meal[j] == 0
    assert w.s.n_poison[j] == 0 and w.s.n_poison10[j] == 0
