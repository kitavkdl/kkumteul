"""한쪽 감각 막기(knockout): 겨루기에서 한 무리만 어떤 입력을 못 받게 한다."""
import numpy as np
import pytest

from kkumteul import kernels as k
from kkumteul import knockout as ko
from kkumteul import stages as st
from kkumteul.config import IN_HEAR, IN_KIN, IN_SAWEAT, IN_TRAIL, N_INTERNAL, Config
from kkumteul.world import World


def small(*presets, **kw) -> Config:
    base = dict(world_w=256.0, world_h=256.0, max_agents=300, init_agents=20, food_cap=300,
                food_inflow=2.2, n_patches=4, min_agents=0, seed=5)
    base.update(kw)
    cfg = Config.from_dict({**Config().to_dict(), **base})
    return cfg.with_overrides([f"preset={p}" for p in presets])


def test_channel_columns():
    cfg = small("stage4")
    base = 2 * cfg.vision_sectors + N_INTERNAL
    assert ko.columns(cfg, "hear") == list(range(base + IN_HEAR, base + IN_HEAR + 4))
    assert ko.columns(cfg, "kin") == [base + IN_KIN]
    assert ko.columns(cfg, "saweat") == [base + IN_SAWEAT]
    assert ko.columns(cfg, "trail") == list(range(base + IN_TRAIL, base + IN_TRAIL + 3))
    assert ko.columns(cfg, "social") == ko.columns(cfg, "hear") + [base + IN_KIN, base + IN_SAWEAT]
    assert ko.columns(cfg, "vision") == list(range(2 * cfg.vision_sectors))
    with pytest.raises(ValueError):
        ko.columns(cfg, "smellovision")


def _facing_pair(cfg):
    w = World(cfg)
    s = w.s
    i, j = np.flatnonzero(s.alive)[:2]
    s.alive[np.flatnonzero(s.alive)[2:]] = False
    s.x[i], s.y[i], s.heading[i] = 100.0, 100.0, 0.0
    s.x[j], s.y[j], s.heading[j] = 110.0, 100.0, np.pi
    s.sig[i] = s.sig[j] = (1.0, -1.0)
    gw, gh = w.cfg.grid_w, w.cfg.grid_h
    k.build_grid(s.x, s.y, s.alive, 1.0 / w.cfg.cell, gw, gh, s.a_start, s.a_items, s.cursor)
    return w, i, j


def test_knocked_out_side_hears_nothing_other_side_hears():
    cfg = small("stage3")
    w, i, j = _facing_pair(cfg)
    s = w.s
    s.tag[i], s.tag[j] = 0, 1
    hear = ko.columns(cfg, "hear")
    with ko.knockout(cfg, "hear", tag=0):
        st.social_sense(s, w.fs, w.isp)
    assert np.all(s.obs[i, hear] == 0.0), "막은 쪽(tag 0)은 아무것도 못 듣는다"
    assert np.abs(s.obs[j, hear]).max() > 0.3, "다른 쪽은 그대로 듣는다"
    kin = ko.columns(cfg, "kin")[0]
    assert s.obs[i, kin] != 0.0, "듣기만 막았으니 닮음 감각은 남는다"


def test_patch_is_undone_and_nothing_changes_outside():
    cfg = small("stage3")
    orig = (k.sense, st.social_sense, st.trail_sense)
    with ko.knockout(cfg, "hear", tag=0):
        assert st.social_sense is not orig[1]
    assert (k.sense, st.social_sense, st.trail_sense) == orig
    w, i, j = _facing_pair(cfg)
    w.s.tag[i] = 0
    st.social_sense(w.s, w.fs, w.isp)
    assert np.abs(w.s.obs[i, ko.columns(cfg, "hear")]).max() > 0.3


def test_vision_knockout_through_a_whole_step():
    """세계 한 스텝을 통째로 돌려도(감각 → 생각) 막은 쪽의 시야 입력은 0이다."""
    cfg = small("stage4", init_agents=40)
    w = World(cfg)
    s = w.s
    live = np.flatnonzero(s.alive)
    s.tag[live[::2]] = 0
    s.tag[live[1::2]] = 1
    vis = ko.columns(cfg, "vision")
    with ko.knockout(cfg, "vision", tag=0):
        w.run(3)
    live = np.flatnonzero(s.alive)
    a = live[s.tag[live] == 0]
    b = live[s.tag[live] == 1]
    assert np.all(s.obs[np.ix_(a, vis)] == 0.0)
    assert np.abs(s.obs[np.ix_(b, vis)]).sum() > 0.0


def test_knockout_of_a_sense_that_is_off_changes_nothing():
    """규칙에 없는 감각(사회 감각이 꺼진 세계의 듣기)을 막아도 세계가 똑같이 흘러간다."""
    cfg = small("stage2")
    assert not cfg.social_enabled
    w1, w2 = World(cfg), World(cfg)
    w1.s.tag[:] = 0
    w2.s.tag[:] = 0
    w1.run(20)
    with ko.knockout(cfg, "hear", tag=0):
        w2.run(20)
    assert w1.state_hash() == w2.state_hash()


# ---- 맛 블록 채널 ----
def test_taste_channels_columns_and_rules():
    c = small("stage2", "kinds", "taste")
    t0 = c.taste_in0
    assert ko.columns(c, "taste") == list(range(t0, t0 + 10))
    assert ko.columns(c, "taste_vis") == list(range(t0, t0 + 8))
    assert ko.columns(c, "taste_val") == [t0 + 8, t0 + 9]
    assert ko.columns(c, "vision") == list(range(2 * c.vision_sectors)) + list(range(t0, t0 + 8))
    # 장치를 끈 같은 크기 세계에서도 칸 목록은 같다
    z = small("stage2", "kinds", "taste_size")
    assert ko.columns(z, "vision") == ko.columns(c, "vision")
    assert ko.columns(small("stage2"), "vision") == list(range(2 * 8))     # 맛 블록 없는 세계는 그대로
    for ch in ("taste", "taste_vis", "taste_val"):
        # 유전자 켠 세계도 설정 검사에 걸리지 않게 둘 다 끈다(설계 2026-11-07 개인 학습 둘째 판 규칙 축 2)
        # 보고 배우기(soc)까지 셋을 함께 끈다(과제 5: 장치를 켠 세계도 설정 검사에 걸리지 않게)
        assert ko.OFF_RULES[ch] == ko.TASTE_OFF
        assert "soc_taste_enabled=false" in ko.OFF_RULES[ch]
        with pytest.raises(ValueError):
            ko.columns(small("stage2"), ch)


def _taste_world_two_tags():
    cfg = small("stage2", "kinds", "taste", init_agents=60)
    w = World(cfg)
    s = w.s
    live = np.flatnonzero(s.alive)
    s.tag[live[::2]] = 0
    s.tag[live[1::2]] = 1
    s.taste_m[:] = 0.5
    # 개체마다 바로 앞에 먹이 하나(보이는 먹이가 있어야 물든 시야 칸이 0이 아니다)
    s.f_alive[:] = False
    for n, i in enumerate(live):
        s.x[i], s.y[i], s.heading[i] = 20.0 + (n % 10) * 20.0, 20.0 + (n // 10) * 20.0, 0.0
        assert k._place_food(s, s.x[i] + 10.0, s.y[i] + 3.0, 1.0, k.KIND_A, 0, int(cfg.field_cap))
        s.ledger[k.L_SPAWN] += 1.0
    return cfg, w


def test_taste_knockout_wipes_block():
    cfg, w = _taste_world_two_tags()
    s = w.s
    t0 = cfg.taste_in0
    blk = list(range(t0, t0 + 10))
    ids0 = s.ids.copy()
    with ko.knockout(cfg, "taste", tag=0):
        w.step()
    same = s.alive & (s.ids == ids0)
    a, b = np.flatnonzero(same & (s.tag == 0)), np.flatnonzero(same & (s.tag == 1))
    assert len(a) and len(b)
    assert np.all(s.obs[np.ix_(a, blk)] == 0.0), "막은 쪽(tag 0)의 맛 블록 10칸은 0"
    assert np.all(s.obs[np.ix_(b, [t0 + 8, t0 + 9])] == 0.5), "다른 쪽은 기억 값을 그대로 받는다"
    assert np.abs(s.obs[np.ix_(b, blk)]).sum() > 0.0
    assert np.abs(s.obs[np.ix_(a, list(range(8)))]).sum() > 0.0, "맛 블록만 막았으니 보통 시야는 남는다"


def test_vision_knockout_also_wipes_tainted_sight():
    cfg, w = _taste_world_two_tags()
    s = w.s
    t0 = cfg.taste_in0
    ids0 = s.ids.copy()
    with ko.knockout(cfg, "vision", tag=0):
        w.step()
    same = s.alive & (s.ids == ids0)
    a, b = np.flatnonzero(same & (s.tag == 0)), np.flatnonzero(same & (s.tag == 1))
    assert np.all(s.obs[np.ix_(a, list(range(2 * cfg.vision_sectors)))] == 0.0)
    assert np.all(s.obs[np.ix_(a, list(range(t0, t0 + 8)))] == 0.0), "막은 쪽의 물든 시야 8칸도 0"
    assert np.abs(s.obs[np.ix_(b, list(range(t0, t0 + 8)))]).sum() > 0.0, "다른 쪽의 물든 시야는 남는다"
    assert np.all(s.obs[np.ix_(a, [t0 + 8, t0 + 9])] == 0.5), "기억 값 칸은 시야 막기가 건드리지 않는다"


def test_split_base_overrides_reach_rules(monkeypatch, tmp_path):
    from kkumteul import measure
    cfg = small("stage2", "kinds", "taste_size")
    assert not cfg.taste_enabled
    side = {"path": "x", "step": 0, "genomes": {}, "cfg": cfg, "gen_mean": 0.0}
    seen = []

    def fake_assay(a, b, n, steps, reps, seed, neutral=False):
        seen.append(a["cfg"])
        return {"r_per_1000": {}, "final_counts": []}

    monkeypatch.setattr(measure, "load_side", lambda p: side)
    monkeypatch.setattr(measure, "assay_sides", fake_assay)
    monkeypatch.setattr(ko, "_neutral", lambda *a, **kw: {})
    monkeypatch.setattr(measure, "winner", lambda r: "")
    monkeypatch.setattr(measure, "verdict", lambda *a, **kw: "")
    ko.split(tmp_path / "c", "taste", rules=[], base_overrides=["taste_enabled=true"])
    assert seen[0].taste_enabled
    seen.clear()
    ko.split(tmp_path / "c", "taste", rules=[])
    assert not seen[0].taste_enabled


def test_taste_off_includes_teach():
    assert "teach_enabled=false" in ko.TASTE_OFF
    for ch in ("taste", "taste_vis", "taste_val"):
        assert "teach_enabled=false" in ko.OFF_RULES[ch]


def test_knockout_taste_split_with_teach_on_passes_config_check():
    # 가르침(teach)은 미숙기·보고 배우기·먹이 종류에 기대므로, 맛 끔 묶음이 teach도 함께 꺼야 설정 검사에 안 걸린다
    c = small("stage2", "kinds", "taste").with_overrides(
        ["taste_gene_enabled=true", "soc_taste_enabled=true", "juv_enabled=true", "teach_enabled=true"])
    assert c.teach_enabled and c.soc_taste_enabled and c.juv_enabled
    off = c.with_overrides(ko.TASTE_OFF)                     # 한꺼번에 적용해도 ValueError가 안 난다
    assert not (off.taste_enabled or off.taste_gene_enabled or off.soc_taste_enabled or off.teach_enabled)
    assert off.juv_enabled                                   # 미숙기는 맛과 무관해 그대로 남는다
    for ch in ("taste", "taste_vis", "taste_val"):
        off = c.with_overrides(ko.OFF_RULES[ch])
        assert not off.teach_enabled and not off.soc_taste_enabled


# ---- 규범의 씨앗: norm_sense 넷째 감각 함수와 norm 채널 ----
NORM_CH = ("norm", "norm_vis", "norm_scalars", "norm_mem")


def _norm_cfg(**kw):
    c = small("stage4", "conflict", "norm", init_agents=80)
    return Config.from_dict({**c.to_dict(), **kw}) if kw else c     # 규범 키는 묶음(preset) 뒤에 얹는다


def test_norm_channel_columns():
    c = _norm_cfg()
    c0 = c.norm_in0
    assert ko.columns(c, "norm") == list(range(c0, c0 + 12))
    assert ko.columns(c, "norm_vis") == list(range(c0, c0 + 8))
    assert ko.columns(c, "norm_scalars") == [c0 + 8, c0 + 9, c0 + 10]
    assert ko.columns(c, "norm_mem") == [c0 + 11]
    for ch in NORM_CH:
        assert ch in ko.CHANNELS and ko.WRITER[ch] == "norm"
        for z in (small("stage2"), small("stage2", "kinds", "taste_size")):
            with pytest.raises(ValueError, match="norm_size"):
                ko.columns(z, ch)
    # 장치를 끈 같은 크기 세계도 칸 목록은 같다
    assert ko.columns(small("stage4", "conflict", "norm_size"), "norm") == ko.columns(c, "norm")


def test_vision_channel_includes_norm_vis():
    c = _norm_cfg()
    t0, c0 = c.taste_in0, c.norm_in0
    assert ko.columns(c, "vision") == (list(range(2 * c.vision_sectors)) + list(range(t0, t0 + 8))
                                       + list(range(c0, c0 + 8)))
    # 규범 블록이 없는 맛 크기 세계는 그대로
    z = small("stage2", "kinds", "taste_size")
    assert ko.columns(z, "vision") == list(range(16)) + list(range(z.taste_in0, z.taste_in0 + 8))


def _norm_world_two_tags():
    cfg = _norm_cfg()
    w = World(cfg)
    s = w.s
    live = np.flatnonzero(s.alive)
    s.tag[live[::2]] = 0
    s.tag[live[1::2]] = 1
    # 짝지어 마주 보게: 둘씩 붙여 놓고 모두 표시·기억을 준다(표시 시야·곁 밝기·내 기억이 0이 아니게)
    for n, i in enumerate(live):
        px, py = 20.0 + (n // 2 % 10) * 22.0, 20.0 + (n // 20) * 22.0
        s.x[i], s.y[i] = px + (n % 2) * 6.0, py
        s.heading[i] = 0.0 if n % 2 == 0 else np.pi
    s.mark_t[live] = 0
    s.mem[live] = 0.5
    return cfg, w


def test_knockout_zero_mode_wipes_norm_columns():
    cfg, w = _norm_world_two_tags()
    cols = ko.columns(cfg, "norm")
    # 막기 전 양성 확인: 막지 않으면 두 쪽 모두 0이 아닌 칸이 있다
    cfg2, w2 = _norm_world_two_tags()
    w2.step()
    s2 = w2.s
    for tg in (0, 1):
        rows = np.flatnonzero(s2.alive & (s2.tag == tg))
        assert np.abs(s2.obs[np.ix_(rows, cols)]).sum() > 0
    s = w.s
    ids0 = s.ids.copy()
    with ko.knockout(cfg, "norm", tag=0):
        w.step()
    same = s.alive & (s.ids == ids0)
    a, b = np.flatnonzero(same & (s.tag == 0)), np.flatnonzero(same & (s.tag == 1))
    assert len(a) and len(b)
    assert np.all(s.obs[np.ix_(a, cols)] == 0.0), "막은 쪽(tag 0)의 규범 블록 12칸은 0"
    assert np.abs(s.obs[np.ix_(b, cols)]).sum() > 0
    assert np.abs(s.obs[np.ix_(a, list(range(16)))]).sum() > 0, "규범 블록만 막았으니 보통 시야는 남는다"


def test_knockout_shuffle_norm_channels_no_keyerror():
    for ch in NORM_CH:
        cfg, w = _norm_world_two_tags()
        with ko.knockout(cfg, ch, tag=0, mode="shuffle", seed=1) as kc:
            w.step()
        assert kc.stats["calls"] == 1


def test_knockout_restores_all_four():
    cfg = _norm_cfg()
    orig = (k.sense, st.social_sense, st.trail_sense, st.norm_sense)
    with ko.knockout(cfg, "norm", tag=0):
        assert st.norm_sense is not orig[3] and k.sense is not orig[0]
    assert (k.sense, st.social_sense, st.trail_sense, st.norm_sense) == orig
    with pytest.raises(RuntimeError):
        with ko.knockout(cfg, "norm_mem", tag=0, mode="shuffle"):
            assert st.norm_sense is not orig[3]
            raise RuntimeError("boom")
    assert (k.sense, st.social_sense, st.trail_sense, st.norm_sense) == orig


def test_knockout_norm_off_rules_pass_config_check():
    c = _norm_cfg(norm_mark_src="sig", norm_mark_noise=0.01, norm_mem_obs_mode="blind", norm_block_tag=0,
                  norm_block3_tag=1, norm_impulse=0.02, norm_mode="inert")
    for ch in NORM_CH:
        assert ko.OFF_RULES[ch] == ko.NORM_OFF
        off = c.with_overrides(ko.OFF_RULES[ch])        # 한꺼번에 적용해도 ValueError가 안 난다
        assert not off.norm_enabled and off.norm_mark_src == "take" and off.norm_mode == "full"
        assert off.norm_block_tag == -1 and off.norm_block3_tag == -1 and off.norm_mark_noise == 0.0
        assert off.norm_impulse == 0.02, "위반 공급은 감각이 아니라 안 건드린다"
    assert "norm_impulse" not in " ".join(ko.NORM_OFF)
    # 불활성 규칙은 의존 키가 없는 한 줄이다
    assert ko.NORM_INERT == ["norm_mode=inert"] and _norm_cfg().with_overrides(ko.NORM_INERT).norm_mode == "inert"
