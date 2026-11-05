"""노드 섭동 학습 규칙(perturb_sigma) 시험.

은닉 뉴런이 출력으로 보내는 값을 조금 흔들고, 흔든 뒤에 보상이 오면 그 흔든 쪽으로 가중치를 옮긴다.
무엇을 보고 있을 때 흔들었는지는 연결마다 따로 적어 둔다(elig_in, elig_rec).
perturb_sigma=0(기본)이면 예전 규칙(보상으로 조절하는 헤브 학습) 그대로다.

    python3 -m pytest -q tests/test_perturb.py
"""
import json
import math
import sys
from pathlib import Path

import numba
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kkumteul import checkpoint as ckpt  # noqa: E402
from kkumteul import kernels as k  # noqa: E402
from kkumteul import measure  # noqa: E402
from kkumteul import stages as st  # noqa: E402
from kkumteul.config import ACT_THRUST, ACT_TURN, Config  # noqa: E402
from kkumteul.run import Runner  # noqa: E402
from kkumteul.world import World  # noqa: E402

J = 0             # 시험에 쓰는 입력 칸
J2 = 1            # 또 다른 입력 칸
O = ACT_TURN      # 은닉 뉴런 0을 읽는 출력
O2 = ACT_THRUST   # 은닉 뉴런 1을 읽는 출력
ZEROED = ("Win", "Wrec", "bh", "Wout", "bout", "alpha", "eta", "h", "obs", "Hin", "Hrec",
          "epost", "epre_in", "epre_h", "elig_in", "elig_rec", "mod", "fatigue", "asleep")


def small(*presets, **kw) -> Config:
    base = dict(world_w=256.0, world_h=256.0, max_agents=600, init_agents=150, food_cap=500,
                food_inflow=2.2, n_patches=4, stats_every=200, frame_every=50, clip_every=300,
                clip_len=20, ckpt_every=400, keyframe_every=200, min_agents=20, event_segment_steps=700)
    base.update(kw)
    cfg = Config.from_dict({**Config().to_dict(), **base})
    return cfg.with_overrides([f"preset={p}" for p in presets])


def blank(n=4, nh=1, **kw) -> World:
    """모두 깨어 있고 뇌와 평생 상태가 전부 0인 개체 n마리. 시험마다 필요한 값만 채운다."""
    w = World(small("stage2", max_agents=n, init_agents=n, min_agents=0, food_inflow=0.0, **kw))
    assert w.s.alive.all()
    for name in ZEROED:
        getattr(w.s, name)[...] = 0
    w.s.n_hidden[:] = nh
    return w


def think(w: World, step: int = 0) -> None:
    st.think_plastic(step, w.s, w.fs, w.isp)


def nudge_of(w: World, out: int = O, k: int = 0) -> np.ndarray:
    """출력으로 간 은닉 값이 흔들린 양. 출력 가중치가 1이고 편향이 0이면 출력 = tanh(흔들린 은닉 값)이다."""
    return np.arctanh(w.s.act[:, out].astype(np.float64)) - w.s.h[:, k]


def test_sigma_zero_keeps_the_reward_modulated_hebb_rule():
    w = blank()
    s = w.s
    s.Win[:, J, 0] = 0.5
    s.obs[:, J] = 1.0
    s.Wout[:, O, 0] = 1.0
    s.alpha[:, 0] = -2.0            # 헤브 규칙은 alpha의 부호를 보지 않는다
    s.eta[:] = 0.1
    s.mod[:] = 1.0
    s.epre_in[:, J] = 2.0
    s.epost[:, 0] = 0.5
    think(w)
    hc = math.tanh(0.5)
    assert s.h[:, 0] == pytest.approx(hc, abs=1e-6)
    assert s.act[:, O] == pytest.approx(math.tanh(hc), abs=1e-6)
    assert s.Hin[:, J, 0] == pytest.approx(0.1, abs=1e-6)          # 학습률 0.1 × 보상 1 × 입력 흔적 2 × 출력 흔적 0.5
    assert s.epost[:, 0] == pytest.approx(0.8 * 0.5 + hc, abs=1e-6)  # 출력 흔적에는 뉴런의 활동이 쌓인다
    assert s.epre_in[:, J] == pytest.approx(0.8 * 2.0 + 1.0, abs=1e-6)


def test_nudge_reaches_the_action_but_not_the_remembered_state():
    w = blank(perturb_sigma=0.2)
    s = w.s
    s.Win[:, J, 0] = 0.5
    s.obs[:, J] = 1.0
    s.Wout[:, O, 0] = 1.0
    s.alpha[:, 0] = 2.0
    think(w)
    hc = math.tanh(0.5)
    assert s.h[:, 0] == pytest.approx(hc, abs=1e-6), "되먹임으로 남는 은닉 상태는 흔들지 않은 값이어야 한다"
    nudge = nudge_of(w)
    assert np.abs(nudge).max() > 1e-3, "출력으로 가는 값은 흔들려야 한다"
    assert len(set(np.round(nudge, 6))) == len(nudge), "개체마다 다르게 흔들려야 한다"
    assert s.elig_in[:, J, 0] == pytest.approx(nudge, abs=1e-5), "연결의 흔적 = 그때 본 것(1.0) × 흔든 양"


def test_trace_holds_the_change_that_reached_the_output_not_the_raw_noise():
    """포화한 뉴런은 흔들어도 출력이 거의 안 바뀐다. 흔적에는 넣은 잡음이 아니라 실제로 바뀐 양이 들어간다."""
    w = blank(perturb_sigma=0.2)
    s = w.s
    s.bh[:, 0] = 3.5                # tanh(3.5) = 0.998, 기울기 0.0036
    s.obs[:, J] = 1.0
    s.Wout[:, O, 0] = 1.0
    s.alpha[:, 0] = 2.0
    think(w)
    assert s.elig_in[:, J, 0] == pytest.approx(nudge_of(w), abs=2e-6)
    assert np.abs(s.elig_in[:, J, 0]).max() < 0.02     # 잡음 그대로라면 크기가 0.2 안팎이다
    assert np.abs(s.elig_in[:, J, 0]).max() > 0.0


def test_nudge_is_paired_with_what_was_seen_at_that_moment_not_afterwards():
    """왼쪽으로 흔들면 그 뒤에는 먹이가 오른쪽에 보인다. 뒤에 보인 것과 짝지으면 거꾸로 배운다."""
    w = blank(perturb_sigma=0.2)
    s = w.s
    s.Wout[:, O, 0] = 1.0
    s.alpha[:, 0] = 2.0
    s.eta[:] = 0.1
    s.obs[:, J] = 1.0
    think(w, 0)
    d0 = nudge_of(w)
    s.obs[:, J] = 0.0
    s.obs[:, J2] = 1.0              # 흔든 다음에 보이는 것이 바뀌었다
    think(w, 1)
    d1 = nudge_of(w)
    s.mod[:] = 1.0
    think(w, 2)
    assert s.Hin[:, J, 0] == pytest.approx(0.1 * 0.8 * d0, abs=1e-6)   # 스텝 0에 본 것은 스텝 0의 흔듦과만(한 스텝 흐려져 0.8)
    assert s.Hin[:, J2, 0] == pytest.approx(0.1 * d1, abs=1e-6)        # 스텝 1에 본 것은 스텝 1의 흔듦과만


def test_recurrent_trace_pairs_the_nudge_with_the_unshaken_previous_state():
    w = blank(nh=2, perturb_sigma=0.2)
    s = w.s
    s.bh[:, 1] = 0.5                # 뉴런 1은 늘 tanh(0.5)로 켜져 있다
    s.Wout[:, O, 0] = 1.0           # 출력 O는 뉴런 0만 읽는다
    s.alpha[:, :2] = 2.0
    s.eta[:] = 0.1
    think(w, 0)
    s.bh[:, 1] = -1.0               # 스텝 1에는 뉴런 1의 상태가 바뀐다. 짝은 지난 상태(tanh 0.5)여야 한다
    think(w, 1)
    d1 = nudge_of(w)                # 스텝 1에 뉴런 0이 흔들린 양
    s.mod[:] = 1.0
    think(w, 2)
    assert s.Hrec[:, 1, 0] == pytest.approx(0.1 * math.tanh(0.5) * d1, abs=1e-6)   # 뉴런 1 → 0: 지난 상태 × 흔든 양
    assert not s.Hrec[:, 0, 0].any(), "흔들린 뉴런의 자기 연결에 잡음끼리의 곱이 쌓이면 안 된다"


@pytest.mark.parametrize("alpha", [2.0, -2.0])
def test_reward_pulls_the_output_toward_the_nudge_whatever_the_sign_of_alpha(alpha):
    w = blank(perturb_sigma=0.2)
    s = w.s
    s.obs[:, J] = 1.0
    s.Wout[:, O, 0] = 1.0
    s.alpha[:, 0] = alpha
    s.eta[:] = 0.1
    think(w, 0)                     # 흔적이 비어 있어 아직 배울 것이 없다
    assert not s.Hin.any()
    nudge = nudge_of(w)
    s.mod[:] = 1.0                  # 흔든 다음 스텝에 보상이 왔다
    think(w, 1)
    # 학습률 0.1 × 보상 1 × 입력 흔적 1 × 흔든 양, 부호는 alpha를 따른다(쓰는 가중치 = 타고난 것 + alpha × 배운 것)
    assert s.Hin[:, J, 0] == pytest.approx(0.1 * math.copysign(1.0, alpha) * nudge, abs=1e-6)
    assert (alpha * s.Hin[:, J, 0] * nudge > 0).all(), "쓰는 가중치는 흔든 쪽으로 움직여야 한다"


def test_sleeping_agents_are_not_nudged():
    w = blank(perturb_sigma=0.2)
    s = w.s
    s.Win[:, J, 0] = 0.5
    s.obs[:, J] = 1.0
    s.Wout[:, O, 0] = 1.0
    s.alpha[:, 0] = 2.0
    s.asleep[:] = 1.0
    s.elig_in[:, J, 0] = 0.5
    think(w)
    assert s.act[:, O] == pytest.approx(math.tanh(math.tanh(0.5)), abs=1e-6)
    assert s.elig_in[:, J, 0] == pytest.approx(0.8 * 0.5, abs=1e-6)     # 흐려지기만 하고 더해지는 것이 없다


def test_trace_arrays_are_small_unless_perturbation_is_on():
    assert World(small("stage2")).s.elig_in.shape[1:] == (1, 1)
    assert World(small("stage2")).s.elig_rec.shape[1:] == (1, 1)
    w = World(small("stage2", perturb_sigma=0.1))
    assert w.s.elig_in.shape[1:] == (w.cfg.n_in, w.cfg.h_max)
    assert w.s.elig_rec.shape[1:] == (w.cfg.h_max, w.cfg.h_max)


def test_newborn_and_injected_agents_start_with_empty_traces():
    w = blank(perturb_sigma=0.2)
    s = w.s
    genomes = measure.extract_genomes(w)
    s.elig_in[:] = 1.0
    s.elig_rec[:] = 1.0
    s.alive[:2] = False
    s.counters[k.C_NALIVE] -= 2
    k._reset_body(s, 0, 900, -1, 0, 0, 0, 1.0, 1.0, 0.0, 2.0)     # 0번 칸에 태어난다
    assert not s.elig_in[0].any() and not s.elig_rec[0].any()
    measure.inject_agent(w, genomes, 0, 0, 5.0, 5.0, 0.0, 1)       # 남은 빈 칸(1번)에 넣는다
    assert s.alive[1] and not s.elig_in[1].any() and not s.elig_rec[1].any()
    assert s.elig_in[2].all(), "다른 개체의 흔적은 건드리지 않는다"


def test_nudges_have_the_set_size_and_are_independent():
    w = blank(n=600, nh=2, perturb_sigma=0.1)
    s = w.s
    s.Wout[:, O, 0] = 1.0
    s.Wout[:, O2, 1] = 1.0
    think(w, 7)
    a0, a1 = nudge_of(w, O, 0), nudge_of(w, O2, 1)
    think(w, 8)
    b0 = nudge_of(w, O, 0)
    assert 0.085 < a0.std() < 0.115 and 0.085 < a1.std() < 0.115
    assert abs(a0.mean()) < 0.02
    assert abs(np.corrcoef(a0, a1)[0, 1]) < 0.2, "뉴런마다 따로 흔들려야 한다"
    assert abs(np.corrcoef(a0, b0)[0, 1]) < 0.2, "스텝마다 새로 흔들려야 한다"


def rel_residual(w: World) -> float:
    b = w.energy_balance()
    return abs(b["residual"]) / (b["spawn"] + b["inject"])


def test_perturbed_world_lives_conserves_energy_and_repeats():
    a = World(small("stage2", perturb_sigma=0.1))
    a.run(1500)
    a.check_sane()
    s = a.summary()
    assert s["births"] > 0 and s["pop"] > 20
    assert rel_residual(a) < 1e-9
    b = World(small("stage2", perturb_sigma=0.1))
    b.run(1500)
    assert a.state_hash() == b.state_hash()
    c = World(small("stage2"))
    c.run(1500)
    assert c.state_hash() != a.state_hash(), "켜면 세계가 달라져야 한다"


def test_perturbed_world_does_not_depend_on_thread_count():
    n0 = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        a = World(small("stage2", perturb_sigma=0.1))
        a.run(400)
        numba.set_num_threads(max(2, min(4, n0)))
        b = World(small("stage2", perturb_sigma=0.1))
        b.run(400)
    finally:
        numba.set_num_threads(n0)
    assert a.state_hash() == b.state_hash()


def test_perturbed_run_resumes_to_the_same_world(tmp_path):
    cfg = small("stage2", perturb_sigma=0.1)
    one = Runner.new(tmp_path / "one", cfg)
    one.loop(800, quiet=True)
    one.close()
    two = Runner.new(tmp_path / "two", cfg)
    two.loop(400, quiet=True)
    two.close()
    two = Runner.resume(tmp_path / "two")
    assert two.world.step_no == 400
    two.loop(400, quiet=True)
    two.close()
    assert one.world.state_hash() == two.world.state_hash()


def test_branch_can_turn_perturbation_on(tmp_path):
    main = Runner.new(tmp_path / "main", small("stage2"))
    main.loop(400, quiet=True)
    main.close()
    r = Runner.branch(tmp_path / "main", tmp_path / "p", 400, ["perturb_sigma=0.1"], False, False)
    s = r.world.s
    assert s.elig_in.shape[1:] == (r.world.cfg.n_in, r.world.cfg.h_max)
    assert np.abs(s.Hrec[s.alive]).max() > 0, "그때까지 배운 것은 갈라낼 때 지워지지 않는다"
    r.loop(200, quiet=True)
    r.close()
    s = r.world.s
    r.world.check_sane()
    assert np.abs(s.elig_in[s.alive]).max() > 0


def test_perturbation_refuses_to_run_on_trace_arrays_of_the_wrong_size():
    """설정만 켜지고 배열이 작으면 커널이 배열 밖에 쓴다(numba는 범위를 보지 않는다). 돌기 전에 막는다."""
    w = World(small("stage2"))
    w.fs[st.F_PERTURB] = 0.1        # set_config를 거치지 않고 켠 상황
    with pytest.raises(ValueError):
        w.step()


def test_turning_perturbation_off_again_clears_the_shared_trace():
    w = World(small("stage2", perturb_sigma=0.2))
    w.run(50)
    assert np.abs(w.s.epost[w.s.alive]).max() > 0
    w.set_config(w.cfg.with_overrides(["perturb_sigma=0"]))
    assert w.s.elig_in.shape[1:] == (1, 1)
    assert not w.s.epost.any(), "흔든 양이 헤브 규칙의 활동 흔적으로 읽히면 안 된다"


def test_checkpoint_from_before_the_trace_arrays_is_still_verified(tmp_path):
    """이 배열이 생기기 전에 저장한 체크포인트도 상태 해시로 검증한다(없는 것은 0과 같다)."""
    w = World(small("stage2"))
    w.run(200)
    p = ckpt.save(w, tmp_path)
    with np.load(p) as z:
        arrays = {name: z[name] for name in z.files if name not in ("elig_in", "elig_rec")}

    def rewrite():
        np.savez(p, **arrays)
        meta = json.loads(p.with_suffix(".json").read_text())
        meta["npz_sha256"] = ckpt.file_sha256(p)
        p.with_suffix(".json").write_text(json.dumps(meta))

    rewrite()
    again, _ = ckpt.load(p)
    assert again.state_hash() == w.state_hash()
    arrays["energy"] = arrays["energy"] + 1.0     # 내용이 바뀌었는데 파일 지문만 맞춘 경우
    rewrite()
    with pytest.raises(ValueError, match="깨졌다"):
        ckpt.load(p)


@pytest.mark.parametrize("bad", ["perturb_sigma=-0.1", "perturb_sigma=1.5"])
def test_perturb_sigma_out_of_range_is_rejected(bad):
    assert Config().with_overrides(["perturb_sigma=0.1"]).perturb_sigma == 0.1
    with pytest.raises(ValueError, match="perturb_sigma은 0과 1 사이"):
        Config().with_overrides([bad])
