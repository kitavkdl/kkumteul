# 규범의 씨앗(가로채기·벌주기) 구현 계획 (2026-11-26 밤)

설계: `docs/superpowers/specs/2026-11-26-norms-seed-design.md` 둘째 판(검토 반영). 이 계획은 그 설계를 **과제 열아홉 개(+ 전제 확인 0)**로 쪼개 한 명이 순서대로 구현할 수 있게 한다.
**사회 학습(`docs/superpowers/plans/2026-11-26-social-learning.md`)과 부모의 가르침(`docs/superpowers/plans/2026-11-26-parent-teaching.md`) 구현이 모두 커밋되고 시험이 초록인 코드 위에 얹는다**(같은 `kernels.py`·`world.py`·`config.py`·`measure.py`·`knockout.py`를 건드린다). 그 계획들이 이미 붙인 번호·이름은 건드리지 않고 **그 맨 뒤**에 이어 붙인다. 이 계획이 그 번호를 이렇게 전제한다(과제 0에서 구현 시점의 실제 값을 다시 읽어 맞춘다 — 다르면 이 문서의 번호를 고치고 시작한다):
- `rng.py`: 마지막이 `S_FEED_GENE = 35`(가르침). 이 계획은 **36~39**(`S_NORM_IMPULSE`·`S_NORM_NOISE`·`S_NORM_MIX`·`S_NORM_BLIND`).
- `kernels.py`: `P_FEED_MIN`·`N_FP = range(75)`, `Q_FEED`·`N_IP = range(36)`(가르침). 이 계획은 그 **뒤**에 `Q_NORM`·`Q_NORM_TAU`(과제 9)와 `P_NORM_MEM_INHERIT`(과제 19)만 덧붙인다. 가르침의 졸업표 `grad_tab`·`TAG_ROWS = 8`·`GT_*`·`N_GT = 29`는 건드리지 않고, 이 계획의 계수 배열은 이름이 다른 `norm_*`다.
- `stages.py`: `N_FS = range(48)`·`N_IS = range(15)`(아직 아무도 안 건드림). 이 계획이 `F_NORM_*`·`I_NORM_*`를 덧붙인다(과제 2).
- `measure.py`: `GENOME`·`OPTIONAL_GENOME == ("taste_g", "soc_g", "show_g", "feed_g")`. **이 계획은 유전체 배열을 더하지 않는다**(`mark_t`·`pun_t`·`mem`은 상태 필드, 유전체가 아니다) → `GENOME`·`OPTIONAL_GENOME`·`pack.py`는 안 고친다(설계 "바꾸는 파일"의 `pack.py` 줄은 새 유전체가 없어 필요 없음. 과제 2의 시험이 묶음 읽기가 그대로인 것만 확인).
값은 모두 **임시 결정**(설계 그대로, `config.py`에서만 바꾼다). 과제마다 TDD: **시험 먼저 → 빨강 확인 → 구현 → 초록 확인 → 커밋**.

공통 규칙(과제마다 지킨다)
- 시험 명령: `python3 -m pytest -q tests/<파일>`. 서버에서는 메모리 상한 도구로 감싸고(`--max 2G`), `NUMBA_NUM_THREADS=2`, `nice`로 낮춘다. `kernels.py`·`stages.py`·`config.py`를 고치면 numba 캐시가 지문(`_cache._SOURCES`)으로 새로 지어지므로 첫 실행이 느리다. 새 시험은 `tests/test_norm.py`(새)에 모은다.
- **꺼짐 해시 시험이 초록이어야 다음 과제로 간다**(설계 "구현 순서"의 규칙). 박은 해시를 쓰는 기존 시험이 그대로 통과해야 한다: `tests/test_stages.py`(`PINNED_HASH_300`, 특히 `("stage4", "conflict", "kinds")` — `attack`을 지나는 세계), `tests/test_taste.py`, `tests/test_harsh.py`, `tests/test_prey.py`, `tests/test_soc_taste.py`, `tests/test_teach.py`.
- 에너지가 생기는 길은 먹기와 명시적 주입뿐이다. 이 장치의 에너지 줄은 모두 `energy[대상] -= dmg; energy[나] += got; L_LOST += dmg − got`(벌금 `got`는 **옮겨 가는 몫**이고 나머지는 `L_LOST`), 비용은 기존대로 `L_MOVE`. 위축·강제·표시·삼가기 기억·섞기는 에너지를 안 건드린다. 설정 검사가 `norm_pun_return ∈ [0, attack_gain]`을 막아 새는 값을 못 쓰게 한다. 장부 시험이 과제 6·16·18의 끝에서 통과해야 한다.
- `rng.py`의 용도 번호는 **뒤에 덧붙인다**(36~39). 기존 번호·호출 순서·인자 배치를 바꾸지 않는다. 새 난수는 `u01(seed, 용도, a, b)` 상태 없는 해시라 번호·인자만 안 겹치면 된다.
- numba: `attack`(기존)과 새 `norm_*` 패스(`norm_tick`·`norm_mark_events`·`watch_pun`·`norm_account`·`norm_mix`)는 **순차 `@njit(cache=True)`**(`prange` 금지, 상태 묶음 `s`를 거쳐 읽고 쓴다). **병렬은 `_norm_sense_par` 하나뿐**이고 배열을 직접 받아 **읽기만 하며 `obs[i]`에만 쓴다**(prange 안에서 상태 묶음을 거쳐 쓰면 쓰기가 조용히 사라진다). 새 커널은 `kernels.py`·`stages.py`에만 둔다.
- **설정 칸**: `attack`은 `fs`/`isp`(`stages.pack_stage_params`)에서 설정을 읽는다 → 이 계획의 새 설정은 `F_NORM_*`·`I_NORM_*`로 둔다. `kernels.py`의 `P_*/Q_*`는 `process_deaths`·`_child`가 읽을 때만(과제 9·19). `attack(step, s, fs, isp)`의 서명은 **바꾸지 않는다**(기존 시험이 직접 부른다).
- 새 장치는 설정으로 끄고 켠다. 꺼져 있으면 `norm_*` 패스 호출 0·표시 갱신 0·계수기 0·입력 블록 0이고 새 저장 필드는 `HASH_DEFAULTS`라 해시에 안 들어간다(박은 해시 시험은 **처음부터 꺼진 세계**로). "꺼짐 = 호출 0"은 시험에서 함수를 감싸 세어 확인한다.
- 시험용 세계는 `small("stage4", "conflict", "norm", ...)`꼴(`tests/test_stages.py`의 `small` 패턴, 개체 수백 마리). 손 시험은 `test_attack_takes_energy_without_creating_any`(`tests/test_stages.py`) 패턴: 개체를 직접 놓고 `k.build_grid` 뒤 `st.attack(step, s, w.fs, w.isp)`를 부른다. **양성 확인 먼저**(이웃이 실제로 보이는 것부터 확인한 뒤 음성 시험).
- 공개 레포: 문서·주석에 로컬 절대 경로·서버 구성·다른 프로젝트 이야기를 쓰지 않는다(레포 상대 경로만, 체인 스크립트 안의 경로는 변수).
- 고친 파일만 `git add`, 커밋하면 바로 push(프로젝트 규칙).

## 과제 0. 전제 확인 (코드 안 고침)
이 계획은 **사회 학습 과제 1~7과 가르침 과제 1~8이 커밋되고 시험이 초록**이어야 시작한다(가르침 9의 판정 도구는 이 계획이 안 건드려 안 기다려도 된다).
- `git log --oneline | grep -E "사회 학습|부모의 가르침"`로 커밋 확인. `git status`가 깨끗해야 한다(다른 에이전트의 미커밋 변경과 섞지 않는다). `grep -n "def watch_meals\|def juv_pass\|def teach_pass\|def feed_pass" kkumteul/kernels.py`, `grep -n "grad_tab\|TAG_ROWS" kkumteul/kernels.py`, `grep -n "teach_enabled" kkumteul/config.py`가 모두 있어야 한다. 없으면 **멈춘다**.
- 시험 한 번: `python3 -m pytest -q tests/test_soc_taste.py tests/test_teach.py tests/test_taste_gene.py tests/test_taste.py tests/test_stages.py tests/test_knockout.py` 통과.
- **번호 읽기(구현 시점의 값을 다시 읽어 이 문서와 맞춘다)**: `grep -n "^S_" kkumteul/rng.py`(마지막이 35가 아니면 그 뒤 번호로 36~39를 밀고 이 문서·시험의 번호를 고친다), `grep -n "N_FP\|N_IP" kkumteul/kernels.py`(마지막 `P_*`·`Q_*`), `grep -n "range(48)\|range(15)" kkumteul/stages.py`(`N_FS`·`N_IS`), `grep -n "feed_g\|know_age" kkumteul/kernels.py`(`AGENT_FIELDS`에서 저장 필드의 마지막 자리 — `mark_t`·`pun_t`·`mem`은 그 바로 뒤), `grep -n "^OTHER_SCRATCH\|HASH_DEFAULTS" kkumteul/kernels.py`.
- **`world.step`의 실제 호출 순서를 읽는다**: `grep -n "k\.\|st\.\|an\." kkumteul/world.py`(가르침이 `juv_pass`·`teach_pass`·`feed_pass`를 어디 끼웠는지). 이 계획의 호출 자리는 "**`st.attack` 바로 뒤**", "**`social_sense`·`trail_sense` 뒤**", "**스텝 맨 앞(`niche_world` 앞)**", "**`after_think` 뒤**"처럼 상대 위치로만 적었다. 가르침 호출과 어긋나면 이 계획의 위치를 고친다.
- 바탕 체크포인트: 설계의 B1(`world4`)의 `config.json`에서 `conflict_enabled`가 꺼짐, `vision_sectors` 8, `n_spare_in` 13, `carry_enabled` 꺼짐, `mind_enabled`·`social_enabled`·`niche_enabled` 켜짐이어야 하고(설계 "지금 아는 것"), 체크포인트 파일이 있어야 체인을 돌린다(없으면 체인 금지; 코드·시험은 상관없다).
- 사회 학습·가르침 체인이 아직 도는 중이면 이 계획의 **체인은 돌리지 않는다**(한 번에 하나). 구현·시험은 상관없다.

---

## 과제 1. 설정·검사·상수·preset (동작 변화 없음)
**바꾸는 파일**: `kkumteul/config.py`, 새 `tests/test_norm.py`.

**시험 먼저**(`tests/test_norm.py`)
- `test_norm_config_defaults`: 설정 키가 정확히 이 **21개**이고 모두 꺼짐·불활성 기본: `norm_measure=False`, `norm_enabled=False`, `norm_mode="full"`, `norm_mark_src="take"`, `norm_mark_tau=50`, `norm_mark_noise=0.0`, `norm_sig_theta=0.5`, `norm_pun_return=0.1`, `norm_aim="marked"`, `norm_inhibit=0.8`, `norm_mem_tau=50`, `norm_mem_rate=1.0`, `norm_mem_obs=0.2`, `norm_mem_obs_mode="outcome"`, `norm_mem_inherit=0.0`, `norm_impulse=0.0`, `norm_impulse_len=5`, `norm_mix_period=0`, `norm_mix_frac=0.2`, `norm_block_tag=-1`, `norm_block3_tag=-1`. (설계 표는 20줄이고 `norm_mix_frac`는 본문과 검사 목록에만 있다 → 21개로 센다. 줄일 후보 `norm_mem_inherit`·`norm_mix_frac`는 `config.py` 주석에 "줄일 후보(설계 약점 9)" 한 줄.)
- `test_norm_config_range_checks`: `norm_pun_return ∉ [0, attack_gain]`, `norm_inhibit ∉ [0, 1]`, `norm_mem_rate`·`norm_mem_obs`·`norm_mem_inherit`·`norm_impulse`·`norm_mark_noise`·`norm_mix_frac ∉ [0, 1]`, `norm_mark_tau`·`norm_mem_tau`·`norm_impulse_len < 1`, `norm_mix_period < 0`, 모르는 `norm_mode`·`norm_mark_src`·`norm_aim`·`norm_mem_obs_mode`면 `ValueError`.
- `test_norm_config_dependencies`: `norm_measure`는 `conflict_enabled` 요구. `norm_enabled`는 `conflict_enabled`·`social_enabled`·입력 크기(`n_in ≥ norm_in0 + N_NORM_IN`)·`vision_sectors == N_NORM_VIS`(8) 요구. **기본값이 아닐 때만 `norm_enabled` 요구**: `norm_mark_src ≠ take`, `norm_mark_noise > 0`, `norm_mem_obs_mode ≠ outcome`, `norm_block_tag ≥ 0`, `norm_block3_tag ≥ 0`, `norm_mode ≠ full`(문자 그대로 "`norm_mem_obs_mode`는 `norm_enabled`를 요구한다"로 쓰면 기본값이 늘 있어 모든 옛 설정이 실패한다 — **기본 설정이 검사를 통과하는 것**을 시험에 박는다). `norm_mark_src=sig`는 `social_enabled`. **`norm_impulse > 0`은 `conflict_enabled`만 요구**(`norm_enabled` 없이 쓰는 위반 공급). `norm_mix_period > 0`은 `carry_enabled` 꺼짐과 `teach_enabled` 꺼짐을 요구.
- `test_norm_mem_inherit_guard`: 과제 19 전에는 `norm_mem_inherit > 0`이 `ValueError`("구현 전 — 과제 19")(설정이 조용히 무동작하는 것을 막는 임시 가드. 과제 19가 이 가드와 이 시험을 지운다).
- `test_norm_string_overrides`: `with_overrides(["preset=norm", "norm_mode=inert", "norm_mark_src=sig", "norm_aim=nearest", "norm_mem_obs_mode=blind"])`가 문자열 키를 받는다(사회 학습·가르침이 만든 문자열 지원 재사용).
- `test_norm_presets_and_sizes`: `preset=norm_size` = `taste_size`에 `n_spare_in`을 `N_NORM_IN`(12)만큼 더한 값(= 13 + 13 + 10 + 12 = 48 → `vision_sectors` 8에서 `n_in == 72`), `preset=norm` = `norm_size` + `norm_enabled=True`. `Config.norm_in0 == taste_in0 + N_TASTE_IN`. `NORM_VIS=0`(여덟 칸)·`NORM_SELF=8`·`NORM_LOCAL=9`·`NORM_SAWPUN=10`·`NORM_MEM=11`. `preset("모르는 이름")` 오류 문구에 `norm_size`·`norm`이 들어 있다.
- `test_old_configs_validate`: `Config()`, 기존 모든 `preset(...)` 조합(`stage1~4`, `conflict`, `kinds`, `ext`, `harsh`, `harsh2`, `taste`, `taste_size`), `norm_*` 키가 없는 옛 설정 dict의 `Config.from_dict`가 `validate`를 통과한다(회귀).
- 빨강 확인: `python3 -m pytest -q tests/test_norm.py`.

**구현**
1. `config.py`: 위 21개 키(주석에 "기본 꺼짐. 설계 2026-11-26-norms-seed-design.md"와 각 키의 근거 한 줄, 설계 표 "근거(셈)" 그대로). 상수 `N_NORM_IN = 12`, `N_NORM_VIS = 8`, `NORM_VIS = 0`, `NORM_SELF = 8`, `NORM_LOCAL = 9`, `NORM_SAWPUN = 10`, `NORM_MEM = 11`. 모드 사전(이름 → 커널이 읽는 번호): `NORM_FULL=0, NORM_INERT=1`(`NORM_MODES`), `SRC_TAKE=0, SRC_SIG=1, SRC_NONE=2`(`NORM_SRCS`), `AIM_MARKED=0, AIM_NEAREST=1`(`NORM_AIMS`), `OBS_OUTCOME=0, OBS_BLIND=1, OBS_OFF=2`(`NORM_OBS_MODES`). 사회 학습·가르침의 모드 이름과 겹치지 않는지 확인(`inert`는 사회 학습과 같은 구실이라 같은 이름을 쓴다).
2. `Config.norm_in0` 속성(`taste_in0 + N_TASTE_IN`). `validate()`: 위 범위·의존·가드. 의존 키는 위 "기본값이 아닐 때만" 꼴(선례: `taste_gene_enabled`가 `taste_enabled`를 요구하는 줄).
3. `preset()`: `table["norm_size"] = {**table["taste_size"], "n_spare_in": N_STAGE_IN + N_EXT_IN + N_TASTE_IN + N_NORM_IN}`, `table["norm"] = {**table["norm_size"], "norm_enabled": True}`(표 뒤에 만든다). 오류 문구의 허용 이름 목록에 추가.

**초록 확인**: `python3 -m pytest -q tests/test_norm.py tests/test_stages.py tests/test_taste.py tests/test_teach.py`. 커밋: "규범의 씨앗 1: 설정·검사·상수·preset(동작 변화 없음)".

---

## 과제 2. 난수·저장 필드·계수 배열·설정 칸·옛 호환 (동작 변화 없음)
**바꾸는 파일**: `kkumteul/rng.py`, `kkumteul/kernels.py`, `kkumteul/stages.py`(칸 추가와 `pack_stage_params`만), `kkumteul/world.py`(`set_config`·표 API), `kkumteul/measure.py`(`inject_agent`만), `tests/test_norm.py`, `tests/test_taste_gene.py`, `tests/test_stages.py`(시험 한 줄 확인만).

**시험 먼저**
- `tests/test_taste_gene.py::test_new_rng_stream_appended`: 가르침이 고친 꼴(`max == 35`)에 `S_NORM_IMPULSE == 36`, `S_NORM_NOISE == 37`, `S_NORM_MIX == 38`, `S_NORM_BLIND == 39`를 더하고 `max(vals) == 39`로 올린다(모든 `S_*`가 서로 다르다).
- `tests/test_norm.py`
  - `test_norm_fields_registered`: `AGENT_FIELDS`에서 `mark_t`(int32)·`pun_t`(int32)·`mem`(float32)이 **저장**, 저장 필드 맨 끝(`feed_g` 뒤). 저장 안 함: `mark_vslot`(int32)·`mark_vid`(int64)(= 설계의 `mark_victim` 칸·id 쌍). `HASH_DEFAULTS`: `mark_t` −1, `pun_t` −1, `mem` 0. `alloc_state`의 새 상태에서 `mark_t`·`pun_t` 전부 −1, `mem` 0, `mark_vslot`·`mark_vid` 전부 **−2(피해자 미지)**.
  - `test_norm_scratch_arrays`: `OTHER_SCRATCH`에 `norm_tab`(`(TAG_ROWS_NORM = 5, N_NT)`, float64: 행은 태그 0·1·2·3과 접힌 칸 4), `norm_tab_role`(`(3, N_RT)`: 2자·3자 친족·3자 남, float64), `norm_hist`(구간표용, 과제 9)가 있고 `SAVED_FIELDS`에는 없음. 가르침의 `grad_tab`은 그대로.
  - `test_norm_off_world_hash_unchanged`: 처음부터 꺼진 세계(`small("stage4", "conflict", "kinds")` 등)의 해시가 기존 박은 해시 시험과 같다(`tests/test_stages.py::test_worlds_without_harsh_keep_pinned_hash`가 그대로 통과).
  - `test_state_hash_skips_default_norm_fields`: 모두 기본값이면 해시에 안 들어가고(그 필드가 생기기 전과 같음), 하나라도 기본값이 아니면 들어간다.
  - `test_old_checkpoint_loads_norm_defaults`: 저장한 `.npz`에서 `mark_t`·`pun_t`·`mem`을 뺀 옛 묶음을 `load_arrays`로 읽으면 −1·−1·0이고 해시가 저장 때와 같다(`exact`가 True). 0 아닌 값이 저장·복원된다. `pack.py`로 낸 묶음 읽기가 그대로다(유전체 배열이 안 늘었다).
  - `test_inject_agent_and_reset_body_init_norm_state`: `measure.inject_agent`로 넣은 개체와 `_reset_body`로 재사용한 칸이 `mark_t = −1`, `pun_t = −1`, `mem = 0`, `mark_vslot = mark_vid = −2`.
  - `test_set_config_off_clears_norm_state`: `norm_enabled`를 True → False로 `set_config`하면 산 개체의 `mark_t`·`pun_t`가 −1, `mem`이 0, 피해자 칸이 −2로 지워진다(꺼지면 `norm_*` 패스가 안 불려 감쇠가 없어 낡은 표시가 되살아나는 것을 막는다 — 설계 "끄기·안전", 판단표 9). 다시 켜도 낡은 표시가 없다. `norm_impulse`는 지우는 것과 무관하다. `norm_measure`만 끄는 것은 지울 것이 없다.
  - `test_norm_stage_params_packed`: `pack_stage_params`가 새 `F_NORM_*`·`I_NORM_*`에 설정값을 담고, 기존 칸의 값·순서는 그대로다(`N_FS`·`N_IS`만 늘었다).
- 빨강 확인.

**구현**
1. `rng.py`: `S_NORM_IMPULSE = 36`("떨리는 손의 구간 상태"), `S_NORM_NOISE = 37`("엉뚱한 표시 잡음"), `S_NORM_MIX = 38`("섞기의 옮김·위치"), `S_NORM_BLIND = 39`("삼가기 눈가림 동전" — **설계는 36~38만 적었으나 `norm_mem_obs_mode=blind`가 난수를 쓰므로 4번째가 필요하다**. 인자 배치는 과제 12에서 정한다).
2. `kernels.py`: `AGENT_FIELDS`에 `mark_t`·`pun_t`·`mem`(저장, `feed_g` 뒤)과 `mark_vslot`·`mark_vid`(저장 안 함). `HASH_DEFAULTS`에 `"mark_t": -1, "pun_t": -1, "mem": 0`. `alloc_state`: `mark_t`·`pun_t` −1, `mark_vslot`·`mark_vid` −2(저장 안 하는 필드의 기본값 채우기 줄이 필요하면 거기에), `OTHER_SCRATCH`에 `"norm_tab"`·`"norm_tab_role"`·`"norm_hist"`와 `alloc_state`의 `np.zeros`. 상수 `TAG_ROWS_NORM = 5`(접기 `min(max(tag, 0), 4)`), 계수 칸 `NT_*`(태그별 계수·에너지, 설계 9절 이름 그대로: `TAKE, TAKE_IMP, TAKE_ON_MARKED, PUN, TAKE_KILL, PUN_KILL, PUN_2ND, PUN_3_KIN, PUN_3_STR, PUN_3_SIM0..3, PUN_3_LINE, PUN_NOVICT, PUN_UNKNOWN, PUN_VGONE, EXPO, OPP_SELF, OPP_OTHER, OPP_OTHER_SIM0..3, OPP_OTHER_LINE, OPP_MARKED, OPP_PLAIN, ATK_VOL_MARKED, ATK_VOL_PLAIN, MARK_EPIS0..2, MARK_EPIS_PUN0..2, ALIVE_STEPS, MARKED_STEPS, SAW_PUN, TAKE_DMG_RECV, PUN_DMG_RECV, TAKE_GAIN, PUN_FINE, PUN_COST, ATK_COST, EATEN_DEAD, LIVES_DONE, N_NT`; `ATK_COST` = 모든 공격 비용(빗나감 포함), `PUN_COST` = 그 가운데 벌의 몫)와 이름표 `NT_NAMES`(소문자 튜플, 판정 도구가 씀), 역할 표 칸 `RT_*`(`PUN_FINE, PUN_COST, SAVED_EST, N_RT`)과 `RT_ROLES = ("self", "kin", "stranger")`. **행 규칙**: 행 = 행위 개체(공격자·집행자·관찰자)의 태그, 단 `*_RECV` 칸은 받은 개체의 태그. `_reset_body`: `mark_t = pun_t = −1`, `mem = 0`, `mark_vslot = mark_vid = −2`.
3. `stages.py`: `F_NORM_*`(`PUN_RET, INHIBIT, MEM_RATE, MEM_OBS, MEM_TAU, IMPULSE, NOISE, SIG_THETA, TAU, MIX_FRAC, RADIUS(= soc_radius 복사), VRANGE, FOV`)와 `I_NORM_*`(`MEASURE, ON, MODE, SRC, AIM, OBS_MODE, IMP_LEN, BLOCK, BLOCK3, MIX_PERIOD, IN0(= norm_in0, 꺼졌으면 −1), VIS0(= norm_in0 + NORM_VIS, 꺼졌으면 −1)`)를 `N_FS`·`N_IS` 앞에 덧붙이고 `range(...)` 개수를 올린다. `pack_stage_params`가 채운다(`I_NORM_ON = int(norm_enabled and n_in이 충분)`). 아직 아무도 안 읽는다.
4. `world.py`: `set_config`에서 `old.norm_enabled and not cfg.norm_enabled`이면 산 개체의 `mark_t`·`pun_t`를 −1, `mem`을 0, `mark_vslot`·`mark_vid`를 −2로(`taste_m` 지우기 옆, 주석에 "유전체가 아니므로 후손에게는 안 간다"). `World.norm_table()`(`{태그: {NT_NAMES: 값}}`, 행 4는 "접힌 칸")·`norm_role_table()`·`reset_norm_tables()` API(과제 3부터 채워진다).
5. `measure.py`: `inject_agent`의 초기화에 `mark_t = pun_t = −1`, `mem = 0`, `mark_vslot = mark_vid = −2`.

**초록 확인**: `python3 -m pytest -q tests/test_norm.py tests/test_taste_gene.py tests/test_stages.py tests/test_taste.py tests/test_harsh.py tests/test_teach.py tests/test_soc_taste.py`. 커밋: "규범의 씨앗 2: 난수·저장 필드·계수 배열·옛 호환(동작 변화 없음)".

---

## 과제 3. 훑기와 `norm_measure` 계수기 (세계 동작 불변 측정)
**바꾸는 파일**: `kkumteul/stages.py`(`attack`의 훑기, `_scan_touching`), `kkumteul/world.py`(summary), `tests/test_norm.py`. (`bench.py`는 안 고친다: 설계 "바꾸는 파일"은 적었지만 `kernel_breakdown`이 1단계 커널만 직접 불러 `stages.attack`을 아예 안 지난다 — 훑기 비용은 아래 시험이 `time.perf_counter`로 `st.attack`을 직접 잰다.)

**시험 먼저**(손 시험: `build_grid` 먼저, 이웃이 실제 보이는 양성 확인을 먼저)
- `test_scan_matches_nearest_touching`: 무작위로 놓은 작은 세계의 모든 산 개체 i에서 훑기가 돌려주는 가장 가까운 닿은 이웃 `nb`가 `_nearest_touching(s, fs, isp, i, True)`와 **같은 칸**이다(같은 거리 2.5, 에너지 ≤ 0 이웃 건너뜀, 자기 제외, 토러스 반대편, 거리 동률의 처리 규칙까지). 손으로 `mark_t`를 붙이면 `nm`(가장 가까운 **표시된** 닿은 이웃)이 표시된 쪽 중 가장 가깝다.
- `test_expo_counts_gap_neighbors_and_ignores_touch`: 거리 2.0~2.5의 이웃(겹치지 않아 `collide`의 `touch = 1`이 안 켜지는 것)도 `EXPO`에 세고, `attack`이 맞은 개체의 `touch`를 `TOUCH_HIT`로 덮어도 분모가 안 흔들린다(설계 판단표 5·22: `touch`를 안 쓴다).
- `test_measure_counts_take_by_actor_tag`: `norm_measure`만 켠 세계에서 손으로 놓은 쌍이 한 번 맞히면 `TAKE`·`TAKE_DMG_RECV`(받은 쪽 태그 행)·`TAKE_GAIN`(친 쪽 태그 행)이 오르고, 친족(부모·자식·형제)이면 친족 계수가, 닮음 구간이면 그 구간 계수가 오른다(`atk-a` 바닥값 용도: 가로챈 횟수·피해자·친족 비율·노출).
- `test_norm_tag_fold_no_overflow`: 태그 −3·4·99인 개체의 계수가 `norm_tab` 밖 배열을 안 덮고 행 4에 접힌다(`tag`는 int32, numba는 범위를 안 검사한다).
- `test_measure_only_hash_equals_off`: `norm_measure`만 켠 세계(같은 `preset=norm_size`)의 300스텝 해시가 꺼진 세계와 같다(동작을 안 바꾸고 계수기는 저장 안 함).
- `test_scan_not_called_when_both_off`: `norm_measure`·`norm_enabled`가 둘 다 꺼진 세계에서 훑기 호출 0(감싸서 셈) — 옛 `attack`과 같은 분기만 돈다.
- `test_scan_cost_budget`(선택, 어림 확인용): 개체 800마리·밀도 현실 값에서 켠 `attack` 한 번이 꺼진 `attack`의 몇 배인지 `time.perf_counter`로 재고 느리면(설계 "다음 검토에서 먼저 볼 곳": 훑기 비용) 결과 문서에 적는다 — 통과/실패 단정은 안 둔다.
- 빨강 확인.

**구현**
1. `_scan_touching(s, fs, isp, i, t)`: `_nearest_touching`과 같은 3×3 칸 순회로 한 번에 `nb`(가장 가까운 닿은 이웃)와 `nm`(가장 가까운 표시된 닿은 이웃, `mark_t`가 아직 없는 이 과제에서는 항상 −1)을 돌려준다. 표시됨 판정은 과제 5의 `_is_marked(t, mark_t, tau)`(여기서는 `mark_t < 0`이면 거짓).
2. `attack`: `isp[I_NORM_MEASURE] != 0 or isp[I_NORM_ON] != 0`이면 **산 모든 개체의 차례마다 먼저** 훑는다(그 개체가 `a ≤ 0`이어도). 에너지 ≤ 0인 개체는 차례가 없다(`ALIVE_STEPS`에는 세되 `EXPO`·`OPP_*`에는 안 센다 — 문서에 이 규칙을 적는다). `EXPO`(내 차례에 `nb`가 있음), `ALIVE_STEPS`를 센다. 꺼져 있으면 이 분기를 안 타고 옛 `attack`과 똑같다. 맞힌 공격은(표시·벌이 아직 없는 이 과제에서는 모두) `TAKE`·`TAKE_DMG_RECV`·`TAKE_GAIN`·친족/닮음 계수를 센다. 색 닮음 식은 `_social_sense_par`의 `IN_KIN`(`1 − √(색 거리)/√3`)과 **같은 식을 함수로 뽑아** 쓴다.
3. `world.summary`: `norm_measure`일 때 `norm_expo_frac`(= `EXPO ÷ ALIVE_STEPS`, 설계의 `θ_t`)·`norm_take_per_step`.

**초록 확인**: `python3 -m pytest -q tests/test_norm.py tests/test_stages.py -k "norm or attack or pinned"`. 커밋: "규범의 씨앗 3: 훑기·norm_measure 계수기(동작 불변)".

---

## 과제 4. 위반 공급: 떨리는 손 (`norm_impulse`)
**바꾸는 파일**: `kkumteul/stages.py`(`attack`의 충동 분기), `tests/test_norm.py`.

**시험 먼저**
- `test_impulse_interval_is_stateless_and_length_L`: `u01(seed, S_NORM_IMPULSE, step // L, id) < p`인 개체가 충동 상태이고, 같은 (seed, 구간, id)는 같은 결과, 충동 상태는 구간 경계(`step // L`이 같은 L스텝)에서 정확히 L스텝 이어진다(저장할 것이 없다). L = 1·5·10에서 확인.
- `test_impulse_probability_close_to_p`: 큰 표본(수만 개체 구간)에서 충동 구간 시작 비율이 p에 가깝다(허용 오차), 개체 스텝 중 충동 상태 비율이 p(L과 무관)다.
- `test_impulse_forces_attack_and_hits_nearest`: 뇌 출력이 −1이어도 충동 상태면 `a = max(a, 1)`로 공격하고 비용 `attack_cost × a`를 내며 **가장 가까운 닿은 이웃**을 가로채기로 친다. 닿은 이웃이 없으면 빗나가 비용만 든다. **이웃이 표시돼 있어도(`mark_t` 손으로 붙임) 가로채기로 친다**(벌이 아니다 — "충동은 누가 표시됐는지 모른다", 설계 약점 13). 조준(`norm_aim=marked`)에 안 끌려간다.
- `test_impulse_does_not_touch_brain_output`: `attack` 호출 뒤에도 `s.act[i, ACT_ATTACK]`가 그대로다(뇌 출력은 안 바뀌고 `attack` 안에서만 덮는다 — 그래서 뇌는 이 충동을 말릴 수 없다, 설계 한계).
- `test_impulse_skips_sleeping_agents`: `asleep > 0`인 개체는 충동이어도 강제되지 않는다. 마음이 꺼진 세계는 `asleep`이 늘 0이라 영향이 없다.
- `test_impulse_counts_all_attacks_in_interval`: 충동 구간 안의 공격은(뇌 출력이 이미 0 위였어도) 모두 `TAKE_IMP`로 세고, "자발" 공격 계수(`ATK_VOL_*`)는 충동 구간 **밖**에서 뇌 출력이 0 위인 공격만 센다.
- `test_impulse_world_needs_conflict_only_and_changes_hash`: `norm_enabled` 없이 `conflict_enabled`만으로 쓸 수 있다(끔·`off` 대조에도 위반이 있어야 한다). `norm_impulse = 0`이면 꺼진 세계와 해시가 같고, `> 0`이면 달라진다(동작이 바뀌므로 당연).
- 빨강 확인.

**구현**: `attack` 안 순서 1번(설계 4절). 충동 판정 `_impulse_on(isp, fs, step, id, asleep)`(`u01`, `isp[I_NORM_IMP_LEN]`, `fs[F_NORM_IMPULSE]`)을 `attack` 첫머리에서 계산한다. 충동이면 `a = max(a, 1.0)`(이 변수만; `s.act`는 안 건드림), 대상은 `nb`, 분기는 가로채기(과제 6에서 분기 코드가 들어오므로 이 과제에서는 기존 가로채기 줄을 그대로 쓰되 충동 표지를 남긴다). `isp[I_NORM_IMPULSE_ON]`처럼 따로 칸을 두지 말고 `fs[F_NORM_IMPULSE] > 0`으로 가른다(꺼져 있으면 `u01` 호출 0). **L은 계산의 입력이 아니라 모형의 가정**(상습 위반자 — 설계 2절, 약점 10)이라 주석에 쓰고 결과 문서 첫머리에 쓴다(과제 18).

**초록 확인**: `python3 -m pytest -q tests/test_norm.py tests/test_stages.py`. 커밋: "규범의 씨앗 4: 떨리는 손(norm_impulse)".

---

## 과제 5. 표시: `mark_t`·밝기·원천 (`take`·`sig`·`none` + 잡음)
**바꾸는 파일**: `kkumteul/stages.py`(`attack`의 표시 줄, `_is_marked`·`_bright`, `norm_mark_events`), `kkumteul/world.py`(`step`), `tests/test_norm.py`.

**시험 먼저**
- `test_mark_visibility_window_and_brightness`: 원천 사건이 있는 스텝 t에 `mark_t = t`. **표시됨 ⇔ `mark_t ≥ 0` 그리고 `경과 ≤ τ`**(τ + 1스텝 동안), 밝기 = `1 − max(0, 경과 − 1) ÷ τ`(경과 0·1이면 1, 경과 τ이면 1/τ). **τ = 5·49·50에서 경과 τ + 1에 정확히 표시 없음**(float32 잔여가 없다 — float32로 `1`에서 `1/τ`를 τ번 빼면 0에 안 닿던 첫째 판의 오류, 설계 판단표 4). `τ = 1`이어도 다음 스텝 감각에는 밝기 1.
- `test_take_marks_only_the_taker`: `take` 원천: 벌이 아닌 공격이 **맞았을 때** 공격자에게만 `mark_t = t`(`mark_vslot`·`mark_vid` = 맞은 이의 칸·id). 빗나간 공격(이웃 없음)은 안 붙고, 맞은 쪽은 안 붙는다. 같은 원천이 또 나면 `mark_t`가 갱신된다.
- `test_mark_source_sig`: `sig` 원천은 `tanh(sig1 출력) > norm_sig_theta`인 스텝에 붙고(아무에게도 해가 없는 행동, `ACT_SIG1`), 그 표시의 피해자는 −1("없음")이다. `social_enabled` 요구.
- `test_mark_noise_stateless`: `norm_mark_noise = q`이면 어떤 원천이든 개체 스텝마다 확률 q로 엉뚱한 표시(피해자 −1)가 붙고, 난수는 `S_NORM_NOISE`의 `u01(seed, S_NORM_NOISE, step, id)`라 같은 시나리오 두 번이 같다. `none` + q = 순수 임의 꼬리표.
- `test_mark_events_run_after_think_before_attack`: `norm_mark_events`가 `after_think`(출력이 정해진 뒤) 뒤에 한 번 돈다. 꺼진 세계·`norm_enabled=false`에서 호출 0.
- `test_marks_not_set_by_measure_only`: `norm_measure`만 켠 세계에서는 표시가 안 붙는다(계수기만).
- 빨강 확인.

**구현**
1. `_is_marked(t, mark_t, tau)`(= `mark_t ≥ 0 and t − mark_t ≤ τ`), `_bright(t, mark_t, tau)`: `@njit(cache=True, inline="always")`. 감쇠 패스가 없다(시각에서 계산).
2. `attack`: `isp[I_NORM_ON]`이고 원천이 `take`일 때, 가로채기가 맞은 직후 공격자의 `mark_t = step`, `mark_vslot = best`, `mark_vid = ids[best]`. **`step`은 `attack`의 인자 그대로**(`world.step_no`).
3. `norm_mark_events(step, s, fs, isp)` 순차 커널: `sig` 원천(`act[i, ACT_SIG1]`에서 `tanh`를 `F_NORM_SIG_THETA`와 비교)과 잡음(`F_NORM_NOISE`) — 어떤 원천이든 잡음은 여기서. 표시 대상은 산 개체뿐.
4. `world.step`: `after_think` 블록 뒤에 `if self.isp[st.I_NORM_ON]: st.norm_mark_events(t, s, fs, isp)`(마음·사회가 꺼진 세계에서도 도는 자리 — `after_think`가 `mind or social`일 때만 도는 것과 무관하게 `norm_enabled`일 때만).

**초록 확인**: `python3 -m pytest -q tests/test_norm.py tests/test_stages.py`. 커밋: "규범의 씨앗 5: 표시(mark_t)와 원천".

---

## 과제 6. 조준·벌 분기·벌금·집행자 비표시·`inert`
**바꾸는 파일**: `kkumteul/stages.py`(`attack`), `tests/test_norm.py`.

**시험 먼저**(설계 시험 5·6·15)
- `test_pun_branch_pays_fine_not_take_gain`: **자발** 공격(충동 아님)이 `full` 모드에서 **표시된 대상**을 치면 벌: 대상이 잃는 양은 같고(`attack_damage × a`, 가진 만큼까지) 집행자가 얻는 몫이 `dmg × norm_pun_return`(0.1), 나머지는 `L_LOST`, 집행자는 **표시되지 않는다**, 맞은 대상은 `pun_t = t`와 `TOUCH_HIT`. 표시 안 된 대상이면 평소 가로채기(`attack_gain` 0.5)이고 공격자가 표시된다.
- `test_pun_return_equal_to_attack_gain_is_plain_attack`: 같은 조건에서 `norm_pun_return = attack_gain`이면 집행자의 에너지·장부가 평소 공격과 같다(손익 표 4행: 0.5는 "벌이 아니라 표시된 자를 노린 약탈", 설계 1절).
- `test_inert_mode_all_attacks_are_takes`: `norm_mode=inert`면 모든 공격이 가로채기(표시된 대상을 쳐도 가로채기이고 공격자가 표시된다), 위축·삼가기 기억 갱신이 없다. 입력·조준·비용은 `full`과 같다(표시가 입력으로 들어와 뇌가 읽을 수 있다는 점까지).
- `test_aim_marked_prefers_marked_neighbor`: `norm_aim=marked`(기본)에서 가까운 비표시 이웃과 먼 표시 이웃이 닿는 거리 안에 둘 다 있으면 표시된 쪽을 친다. 표시가 없으면 가장 가까운 쪽(기존과 같음). `nearest`는 기존 동작. **충동은 항상 `nearest`**(과제 4 시험이 확인한 것과 같은 코드 경로).
- `test_wrong_target_becomes_take_and_marks_me`: 엉뚱한 이웃(표시 안 된)을 치면 가로채기가 되어 내가 표시된다(헛나감의 값이 저절로 생긴다).
- `test_kill_counters_pun_vs_take`: 대상 에너지가 바닥나면 `TAKE_KILL`·`PUN_KILL`로 따로 센다(벌에 맞은 개체도 `TOUCH_HIT`라 `process_deaths`의 `DEATH_KILLED`에 가로채기 사망과 섞이므로 벌로 죽은 수를 따로 읽는다).
- `test_norm_world_lives_and_ledger_closes`: `small("stage4", "conflict", "norm", norm_impulse=0.015)`를 1,500스텝 돌려 `rel_residual(w) < 1e-9`이고 `NT_PUN > 0`. **벌 분기의 벌금 줄을 틀리게 하면(예: `got`를 `attack_gain`으로) 실패하는 것을 구현 뒤 한 번 확인하고 되돌린다.**
- `test_inert_isolated_hash_equals_off`(설계 시험 2): `norm_enabled=true` + `inert` + `norm_impulse=0` + `norm_mark_src=none` + `norm_mark_noise=0`인 세계의 해시가 꺼진 세계(**같은 `preset=norm_size`, 같은 `soc_*`·`taste_*` 설정**)와 같다. (과제 12에서 `attack` 뒤 격자 재구성이 들어오면 같은 시험을 다시 돌려 여전히 같은지 본다.)
- 빨강 확인.

**구현**: `attack`을 설계 4절 순서 0~6대로 정리한다. 0 훑기(과제 3), 1 충동 여부(과제 4), 2 비용 `attack_cost × a`(`L_MOVE`, 기존), 3 대상 고르기(충동 = `nb`, 자발은 `AIM_NEAREST`면 `nb`, `AIM_MARKED`면 `nm`이 있으면 `nm` 없으면 `nb`), 4 분기(충동 = 가로채기, 자발 + `full` + 대상이 표시돼 있으면 벌, `inert`는 모두 가로채기), 가로채기는 기존 에너지 줄 + 표시(과제 5), 벌은 `dmg`는 같고 `got = dmg × F_NORM_PUN_RET`, 집행자 비표시, `pun_t[대상] = step`, `TOUCH_HIT`, 계수. `norm_enabled`가 꺼져 있으면 이 분기들을 못 타고 옛 `attack` 본문 그대로(해시 시험이 잡는다).

**초록 확인**: `python3 -m pytest -q tests/test_norm.py tests/test_stages.py tests/test_taste.py`. 커밋: "규범의 씨앗 6: 조준·벌 분기·벌금·inert".

---

## 과제 7. 삼가기 기억 `mem`·위축·`norm_tick`
**바꾸는 파일**: `kkumteul/stages.py`(`attack`의 위축·기억 갱신, `norm_tick`), `kkumteul/world.py`(`step`), `tests/test_norm.py`.

**시험 먼저**(설계 시험 7)
- `test_mem_rises_on_punishment_and_decays`: 벌 받으면 `mem += norm_mem_rate × (1 − mem)`(기본 1.0: 한 번에 최대), `norm_tick`이 스텝마다 `mem = max(0, mem − 1/norm_mem_tau)`로 줄이고 **`< 0.5/τ`면 0으로 자른다**(float32 잔여가 없다: τ = 50·49·5에서 정확히 0에 닿는다).
- `test_inhibition_multiplies_takes_only`: 가로채기 출력에만 `a_eff = a × (1 − norm_inhibit × mem[i])`가 곱해진다(**충동 가로채기에도** 곱해진다 — 벌 받은 충동 상태는 곧 가만해진다, 설계 4절). **벌에는 안 곱해진다**(벌주는 쪽은 삼가지 않는다). `inert`는 위축·기억 갱신이 없다.
- `test_inhibit_zero_leaves_output_unchanged_memory_still_rises`: `norm_inhibit = 0`이면 가로채기 출력이 안 바뀌고 `mem`만 오른다(뇌가 `NORM_MEM` 입력으로 읽을 때만 행동을 바꾸는 측정 통로).
- `test_norm_tick_runs_before_norm_sense_slot`: `world.step`에서 `norm_tick`이 `social_sense`·`trail_sense` 뒤, 생각 앞에 한 번 돈다. 꺼진 세계에서 호출 0.
- `test_mem_not_inherited_by_default`: 새끼의 `mem`은 0(`norm_mem_inherit`는 과제 19).
- 빨강 확인.

**구현**: `attack` 가로채기 가지에서 출력 `a`에 위축을 곱해 에너지·비용 줄에 쓴다(벌 가지는 곱하지 않음). 벌 받은 대상의 `mem += F_NORM_MEM_RATE × (1 − mem)`. `norm_tick(s, fs, isp)` 순차: `mem = max(0, mem − 1/τ)`, `< 0.5/τ`면 0. `world.step`: `k.sense`·`social_sense`·`trail_sense` 뒤 `if isp[I_NORM_ON]: st.norm_tick(s, fs, isp)`. (`mem` 갱신은 float32에서 `1 − mem`을 곱해 [0, 1] 안에 머문다.)

**초록 확인**: `python3 -m pytest -q tests/test_norm.py tests/test_stages.py`. 커밋: "규범의 씨앗 7: 삼가기 기억·위축·norm_tick".

---

## 과제 8. 계수기 ①: 집행 역할·친족·에너지 표
**바꾸는 파일**: `kkumteul/stages.py`(`attack`의 계수), `kkumteul/world.py`(표 API·summary), `tests/test_norm.py`.

**시험 먼저**(설계 시험 11 앞쪽)
- `test_pun_role_counts`: 벌이 일어날 때 (집행자 i, 벌 받은 t, `mark_vid[t]`로 찾은 피해자)를 보고 `PUN_2ND`(피해자 = 집행자 자신), 피해자가 있고 자신이 아니면 `PUN_3_KIN`·`PUN_3_STR`로 가른다. 친족은 **두 가지로 잰다**: ⒜ 닮음 구간별 `PUN_3_SIM[0..3]`(경계 < 0.88, 0.88~0.92, 0.92~0.96, ≥ 0.96; `1 − √(색 거리)/√3`), ⒝ 계보 `PUN_3_LINE`(피해자가 집행자의 부모·자식·형제 — `parent`·`ids`로 가림). **문턱을 박은 설정 키는 없다**(`test_no_kin_theta_key`: `Config`에 `norm_kin_theta`가 없다). 구현 첫 일로 시작 체크포인트(B1)의 이웃 닮음 분포(이웃 쌍 대 아무 쌍)를 재서 "친척" 경계를 사후에 고르는 한 줄 도구를 과제 17의 `observe`에 둔다.
- `test_pun_unknown_novict_vgone`: `PUN_NOVICT`(표시가 위반에서 오지 않음: `sig` 원천·잡음, 피해자 −1), **`PUN_UNKNOWN`**(피해자 미지 −2: 체크포인트를 이어 읽은 직후 이미 표시돼 있던 개체; 읽기·`inject_agent`·`_reset_body` 뒤의 피해자는 −2), `PUN_VGONE`(피해자가 이미 죽었거나 그 칸이 다른 개체로 바뀜 — **칸 번호와 id를 같이** 저장해 가린다: 칸 재사용 시나리오 포함). 둘 다(−1·−2)는 3자 분모에서 빠진다. `PUN_NOVICT ÷ PUN`은 판정에 안 쓴다(항등식, 판단표 14).
- `test_opportunity_counts_self_other_sim_line`: 훑기에서 내 차례에 닿는 표시된 이웃 `nm`이 있으면 그 피해자가 나(`OPP_SELF`)인지 남(`OPP_OTHER`, 남이면 `OPP_OTHER_SIM[구간]`·`OPP_OTHER_LINE`)인지로 센다. 같은 훑기·같은 거리 2.5라 `p2 = PUN_2ND ÷ OPP_SELF`, `p3 = (PUN_3_KIN + PUN_3_STR) ÷ OPP_OTHER`가 1을 넘지 않는다(분모가 분자의 기회 전체).
- `test_selectivity_denominators`: `OPP_MARKED`(`nm` 있음)와 `OPP_PLAIN`(닿는 이웃은 있으나 전부 표시 안 됨), 그때의 **자발 공격**(충동 아님, 뇌 출력 > 0) `ATK_VOL_MARKED`·`ATK_VOL_PLAIN`. 선택성 = (`ATK_VOL_MARKED ÷ OPP_MARKED`) ÷ (`ATK_VOL_PLAIN ÷ OPP_PLAIN`)이 합성 값에서 손셈과 같다.
- `test_energy_tables_by_tag_and_role`: 태그별 `TAKE_DMG_RECV`·`PUN_DMG_RECV`·`TAKE_GAIN`·`PUN_FINE`·`PUN_COST`(벌 비용 몫 `attack_cost`)·`ATK_COST`가 에너지 줄과 같고(합 = 장부와 대조: 같은 스텝에 `L_MOVE` 증가 = `ATK_COST` 합, `L_LOST` 증가 = 피해 − 얻은 몫의 합), `norm_tab_role[역할]`에 벌금·벌 비용·지킨 피해의 추정이 역할(2자·3자 친족·3자 남) 안에서 쌓인다. 태그 접기(행 4)가 여기서도 안 넘친다.
- `test_summary_norm_fields`: `world.summary()`에 `norm_pun_rate_2nd`·`norm_pun_rate_3rd`(`p2`·`p3`)가 켠 세계에서만 있다.
- 빨강 확인.

**구현**: `attack`의 벌·가로채기 가지에서 위 계수를 센다. 닮음 식은 과제 3의 함수. 계보 판정 `_kin_line(s, i, j)`(`parent[i] == ids[j]`, `parent[j] == ids[i]`, 같은 `parent ≥ 0`; 가르침이 `_kin_class`를 이미 만들었으면 **재사용**하고 새로 안 만든다 — 과제 0에서 `grep -n "_kin_class" kkumteul/kernels.py`). 에너지 표는 에너지 줄 바로 옆에서 더한다(장부를 안 건드린다). `world.norm_table()`·`norm_role_table()`이 채워진다.

**초록 확인**: `python3 -m pytest -q tests/test_norm.py tests/test_stages.py tests/test_soc_taste.py tests/test_teach.py`. 커밋: "규범의 씨앗 8: 집행 역할·친족·에너지 표 계수기".

---

## 과제 9. 계수기 ②: 구간표·표시 에피소드·유병률·죽을 때 닫기
**바꾸는 파일**: `kkumteul/stages.py`(`norm_account`, `attack`의 구간 계수), `kkumteul/kernels.py`(`process_deaths` 훅, `Q_NORM`·`Q_NORM_TAU`, `pack_params`), `kkumteul/world.py`(`step`, summary), `tests/test_norm.py`.

**시험 먼저**(설계 시험 11 뒤쪽)
- `test_interval_table_boundaries_and_exclusions`: 노출 개체 스텝(`EXPO`)과 가로채기 횟수를 **벌 받은 뒤 경과 스텝**(`t − pun_t`) 구간(1~10, 10~50, 50~200, 200~·없음)으로 `norm_hist`에 센다. 경계가 설계 그대로고, **벌 받은 바로 그 스텝(경과 0)에 자기 차례가 오는 개체는 표에서 뺀다**(처리 순서에 따라 같은 스텝에 먼저 맞은 개체의 노출만 늘어나는 인공물). 표시됐으나 **벌 받지 않은** 개체의 같은 구간표가 짝으로 쌓인다. 구간별 가로채기율 = 횟수 ÷ 노출.
- `test_interval_table_is_manipulation_check_only`: 문서(`norm_judge.py`·결과 문서 문구 상수)에 "엔진 위축이 있으면 규범 없이도 낮게 나온다 — 조작 확인용이지 뇌가 배운 삼감의 증거가 아니다"가 구간표 출력 라벨로 붙는다(과제 17에서 출력 문구 시험).
- `test_prevalence_and_episode_counts`: `norm_account`(스텝 끝 순차 패스)가 표시 유병률 계수(`MARKED_STEPS ÷ ALIVE_STEPS`)를 세고, 표시가 **만료되는 스텝**(`t − mark_t == τ + 1`)에 `MARK_EPIS[종류]`를 1 올리고 `pun_t ≥ mark_t`이면 `MARK_EPIS_PUN[종류]`도 올린다. 종류 = 그 표시의 피해자 값으로 0 `take`(≥ 0), 1 피해자 없음(`sig`·잡음, −1), 2 미지(−2). 같은 원천이 또 나 `mark_t`가 갱신되면 새 에피소드로 센다.
- `test_episode_closed_at_death_no_double_count`: **설계에 없던 구멍을 이 계획이 막는다**: 표시 에피소드를 "만료 때만" 세면, 표시된 채 죽는 개체(벌에 맞아 죽는 개체가 바로 그렇다)의 에피소드가 빠져 `MARK_EPIS_PUN ÷ MARK_EPIS`가 낮게 치우친다. 그래서 `process_deaths`의 `_remove` 호출 앞에서 `ip[Q_NORM] != 0`이고 아직 표시됨(`t − mark_t ≤ τ`)이면 그 에피소드를 닫아 센다. 만료 스텝에 이미 센 것(`t − mark_t == τ + 1`)은 안 센다. 죽은 개체가 만료 스텝과 같은 스텝에 죽어도 한 번만 센다. 노쇠·굶음·공격 사망 모두.
- `test_death_hook_records_eaten_for_net_income`: `process_deaths` 훅이 죽는 개체의 `eaten`을 `NT_EATEN_DEAD`(태그 행)에 더하고 `NT_LIVES_DONE`을 1 올린다(과제 14·17의 개체 스텝당 순수입 계산이 쓴다 — 설계는 "생애 에너지"라고만 적어 계수 정의가 없었다).
- `test_account_off_not_called`: 꺼진 세계·`norm_measure`만 켠 세계에서 `norm_account`와 `process_deaths` 훅 호출 0. 훅이 `process_deaths`의 기존 시험·해시에 안 닿는다.
- `test_summary_prevalence`: `world.summary()["norm_mark_prevalence"]`·`norm_punished_per_episode`가 켠 세계에서만 있다.
- 빨강 확인.

**구현**
1. `kernels.py`: `Q_NORM`(= `int(cfg.norm_enabled)`)·`Q_NORM_TAU`(= `cfg.norm_mark_tau`)를 `N_IP` 앞(가르침의 `Q_FEED` 뒤)에 덧붙여 개수 +2, `pack_params`가 채운다. `process_deaths`: `_remove` 호출 앞에 `if ip[Q_NORM] != 0: _norm_death_hook(step, s, i, ip)`(에피소드 닫기 + `EATEN_DEAD`·`LIVES_DONE`). 주석에 "`strike`(운석) 사망은 안 센다"(가르침과 같은 줄).
2. `stages.py`: `norm_account(step, s, fs, isp)` 순차(유병률 계수 + 만료 에피소드 + 구간표 경과 계수의 분모·분자 중 `attack` 밖에서 세는 것). `attack`의 구간 계수는 훑기 줄 옆.
3. `world.step`: `st.attack` 바로 뒤(과제 12의 격자 재구성·`watch_pun` 자리 뒤)에 `if isp[I_NORM_ON]: st.norm_account(t, s, fs, isp)`, **`process_deaths` 앞**.
4. `world.summary`: 표시 유병률, 표시당 집행 확률(`MARK_EPIS_PUN ÷ MARK_EPIS`).

**초록 확인**: `python3 -m pytest -q tests/test_norm.py tests/test_stages.py tests/test_harsh.py tests/test_prey.py`. 커밋: "규범의 씨앗 9: 구간표·표시 에피소드·유병률(죽을 때 닫기)".

---

## 과제 10. 감각 블록: 표시 시야·이웃의 표시·방금 벌·내 기억 (`_see_norm`)
**바꾸는 파일**: `kkumteul/stages.py`(`_see_norm`, `_norm_sense_par`, `norm_sense`, `_mind_par`의 자는 개체 지우기), `kkumteul/world.py`(`step`), `tests/test_norm.py`.

**시험 먼저**(설계 시험 10)
- `test_norm_vis_direction_and_brightness`: 손 시나리오(표시된 이웃 하나)에서 `NORM_VIS`가 그 이웃이 걸치는 **시야 방향 칸**에서만 0이 아니고 값 = 가까움 × 밝기다. 같은 이웃이 표시되지 않았으면 전부 0.
- `test_norm_vis_equals_closeness_times_brightness`: 한 이웃일 때 `NORM_VIS[sec]` = 본 시야 채널 1의 `[sec]`(가까움 `c`) × `_bright(...)`(칸 값이 조용히 0이 되는 함정 — 같은 이웃이 본 시야에 같은 `c`로 이미 들어 있어 `_see`를 그대로 재사용하면 `c > cur`가 거짓이 된다, 설계 5절 (a)).
- `test_norm_vis_unmarked_near_does_not_occlude_marked_far`: 같은 칸에서 **표시 안 된 가까운 이웃이 표시된 먼 이웃을 안 가린다**(설계 5절 (b): 후보는 표시된 이웃뿐, 칸마다 값의 **최댓값**만 남기며 순서 무관).
- `test_norm_vis_leaves_ordinary_vision_untouched`: 본 시야 채널 0·1과 `other` 채널이 안 지워진다(NORM 블록이 있어도 `obs[i, 0:2×nsec]`가 꺼진 세계와 같다).
- `test_see_norm_sector_span_matches_see`: `_see_norm`이 덮는 칸의 범위가 `_see`와 같은 식이다(같은 거리·반지름·시야각에서 칸 집합이 같음; 시야 기하는 복사하되 `_see`는 **고치지 않는다**).
- `test_norm_vis_torus_and_self_excluded`: 토러스 반대편 이웃이 보이고 자기 자신은 안 본다(`NORM_SELF`는 별도).
- `test_norm_scalars`: `NORM_SELF` = 내 밝기, `NORM_LOCAL` = `hear_range` 안 이웃의 거리 가중 평균 밝기(무리 기본값), `NORM_SAWPUN` = `hear_range` 안 이웃 가운데 `t_now − pun_t == 1`인 것의 거리 가중 **최댓값**, `NORM_MEM` = `mem`. 손셈과 같다.
- `test_sleeping_norm_vis_zero_scalars_kept`: 마음이 켜진 세계에서 자는 개체(`asleep > 0`)의 `NORM_VIS`는 0(시야에서 나온 것이므로 `I_NORM_VIS0`을 `_mind_par`가 물든 시야(`I_TASTE_VIS0`)와 함께 지운다)이고 `NORM_SELF`·`NORM_LOCAL`·`NORM_SAWPUN`·`NORM_MEM`은 그대로 둔다.
- `test_norm_block_all_zero_when_off`: 꺼짐이면 블록 12칸이 전부 0이고 `norm_sense` 호출 0(감싸서 셈).
- `test_norm_sense_thread_determinism`: `numba.set_num_threads(1)`·`(4)`에서 같은 `obs`(결정성: 감쇠·쓰기를 병렬 커널 안에서 하지 않는다).
- `test_norm_sense_only_reads_state`: 감각 호출 전후 `mark_t`·`pun_t`·`mem`이 그대로(`obs[i]`에만 쓴다).
- 빨강 확인.

**구현**
1. `_see_norm(obs, i, col0, nsec, ddx, ddy, d, rad, hi, fov, vrange, val)`: 시야 칸 범위 계산은 `kernels._see`의 식을 복사하고, 값 = `c × val`(`c` = 1 − 간격/vrange, `val` = 밝기), **칸마다 최댓값만** 쓰며 본 시야·`other` 채널은 건드리지 않는다. 표시된 이웃만 후보.
2. `_norm_sense_par(fs, isp, t_now, alive, x, y, heading, mark_t, pun_t, mem, obs, a_start, a_items)` — `@njit(cache=True, parallel=True)`, **배열을 직접 받는다**. 격자(`a_start`·`a_items`)는 스텝 처음에 지은 것(`k.sense`가 쓰는 것과 같다). 시야 거리·시야각은 `F_NORM_VRANGE`·`F_NORM_FOV`, 듣는 거리는 `F_HEAR`. `norm_sense(step, s, fs, isp)` 래퍼(인자 위치: 상태 `s`가 **둘째**, `knockout`이 같은 꼴로 감싼다).
3. `_mind_par`: `nv0 = isp[I_NORM_VIS0]`로 자는 개체의 NORM_VIS 여덟 칸을 지운다(`I_TASTE_VIS0` 줄 옆).
4. `world.step`: `norm_tick`(과제 7) 바로 뒤 `if isp[I_NORM_ON]: st.norm_sense(t, s, fs, isp)`.

**초록 확인**: `python3 -m pytest -q tests/test_norm.py tests/test_stages.py tests/test_taste.py`. 커밋: "규범의 씨앗 10: 감각 블록(_see_norm)".

---

## 과제 11. 감각 막기(`knockout`)에 `norm_sense` 넷째 감각 함수와 채널
**바꾸는 파일**: `kkumteul/knockout.py`, `tests/test_knockout.py`.

**시험 먼저**(`tests/test_knockout.py`, 설계 시험 17 뒤쪽)
- `test_knockout_zero_mode_wipes_norm_columns`: zero 모드는 감각 함수 **넷**(`k.sense`, `st.social_sense`, `st.trail_sense`, `st.norm_sense`) 뒤에서 지운다. `norm_sense`를 안 감싸면 NORM 칸이 도로 채워져 막기가 조용히 무동작(가를 수 없음으로 오판)이므로, 켠 세계에서 `norm` 채널을 막으면 그 태그 개체의 NORM 열이 0인 것을 확인한다(**막기 전 양성 확인**: 막지 않으면 0이 아닌 칸이 있다).
- `test_knockout_shuffle_norm_channels_no_keyerror`: `shuffle`에서 `WRITER["norm"|"norm_vis"|"norm_scalars"|"norm_mem"] == "norm"`이라 KeyError 없이 돈다.
- `test_knockout_restores_all_four`: 블록을 나가면 네 함수가 모두 원래대로다(예외가 나도).
- `test_norm_channel_columns`: `CHANNELS`에 `norm`(블록 12칸 전부), `norm_vis`(0~7), `norm_scalars`(8~10), `norm_mem`(11)이 있고 열 번호가 `norm_in0 + NORM_*`다. 장치가 없는 설정에서 `columns()`가 맛 채널처럼 오류 문구를 낸다.
- `test_vision_channel_includes_norm_vis`: `columns(cfg, "vision")`이 `norm_vis`를 합친다(양성 대조 "시야 막기"가 표시 시야도 막는다 — 맛 시야 `taste_vis`가 이미 합쳐지는 선례).
- `test_knockout_norm_off_rules_pass_config_check`: `OFF_RULES`의 `norm`·`norm_vis`·`norm_scalars`·`norm_mem`이 **의존 키를 함께 기본으로 되돌린다**(`norm_enabled=false`, `norm_mode=full`, `norm_mark_src=take`, `norm_mark_noise=0`, `norm_mem_obs_mode=outcome`, `norm_block_tag=-1`, `norm_block3_tag=-1`)라 `norm_enabled`·`sig`·막기 키를 켠 체크포인트 설정에 `with_overrides`로 한꺼번에 적용해도 `ValueError`가 안 난다(맛이 `taste_enabled=false`와 `taste_gene_enabled=false`를 같이 둔 이유). **`norm_impulse`는 건드리지 않는다**(위반 공급은 감각이 아니다). `norm_mode=inert`는 의존 키가 없는 하나의 규칙이다.
- 빨강 확인.

**구현**: `knockout.py`의 `orig = (...)`를 넷으로, `wrap`에 `norm_sense`(상태 위치 1 — `norm_sense(step, s, fs, isp)`)를 더하고 복원도 넷, `WRITER`에 넷, `CHANNELS`·`columns()`에 새 채널, `OFF_RULES`에 `NORM_OFF = [...]`(`TASTE_OFF` 옆, 한 곳에서 정의해 도구가 같이 쓴다). 모듈 머리 설명의 "감각 함수 셋"을 넷으로 고친다. `argparse`의 `--channel choices=CHANNELS`는 자동으로 따라온다.

**초록 확인**: `python3 -m pytest -q tests/test_knockout.py tests/test_norm.py tests/test_taste.py`. 커밋: "규범의 씨앗 11: 감각 막기에 norm_sense와 norm 채널".

---

## 과제 12. `watch_pun`·눈가림·`attack` 뒤 격자 재구성
**바꾸는 파일**: `kkumteul/stages.py`(`watch_pun`), `kkumteul/world.py`(`step`), `tests/test_norm.py`.

**시험 먼저**(설계 시험 8·19; 사회 학습 시험 2번과 같은 이유로 **`build_grid`를 먼저 부르고 양성 확인 먼저** — 칸 경계의 이웃이 격자 다시 짓기로 보인다)
- `test_watch_pun_raises_neighbor_mem_within_radius`: 이번 스텝에 `pun_t == t`인 개체를 `soc_radius` 안의 이웃이 보면 `mem += norm_mem_obs × (1 − mem)`(0.2). 반지름 밖·죽은 이웃은 그대로, 벌 받은 본인은 `norm_mem_rate`로 이미 올랐고 여기서 또 안 오른다. **칸 경계 0.5 안쪽 쌍**(예: x = 7.8과 8.2), 거리 `soc_radius` 바로 안(7.99)·바로 밖(8.01), 토러스 반대편이 정확히 센다.
- `test_watch_pun_order_is_deterministic`: 적용 순서 = 벌 받은 개체 칸 번호 오름차순(같은 관찰자의 `mem`을 여러 벌이 차례로 올리는 지수평활은 비가환이라 순서가 값을 가른다). 같은 시나리오 두 번이 같은 해시.
- `test_mem_obs_mode_blind_and_off`: `outcome` = 본 벌로 올린다. **`blind` = 벌을 본 스텝과 상관없이 같은 기대 비율로** 올린다 — **설계는 "같은 확률"이라고만 적고 정의가 없어 이 계획이 정한다**: 그 스텝에 (관찰자, 본 벌) 쌍의 수를 `m`, 산 개체 수를 `n`이라 하면 개체마다 `u01(seed, S_NORM_BLIND, step, id) < m ÷ n`이면 `mem += norm_mem_obs × (1 − mem)`(관찰 횟수의 기대는 같고 정보는 0). 시험: 같은 시나리오 두 번 같음, 벌이 없는 스텝은 `m = 0`이라 안 오름, 큰 표본에서 평균 상승 횟수가 `outcome`과 같다. `off` = 안 바뀐다.
- `test_saw_pun_counter`: 본 벌 수(`SAW_PUN`)가 모든 모드에서 같게 센다.
- `test_grid_rebuild_after_attack_only_when_norm_on`: `k.build_grid`와 `st.attack`·`st.watch_pun`을 감싸 호출 순서를 기록: `norm_enabled=false`면 `attack` 뒤 격자 호출 0, 켜지면 **정확히 한 번**(`attack` 뒤·`watch_pun` 앞).
- `test_attack_neighbor_set_unchanged_by_rebuild`: `attack`이 보는 이웃 집합이 꺼진 세계와 같다(**격자를 `attack` 앞에 지으면 달라지는 것을 한 번 확인**하고 뒤로 되돌린다 — 앞에 두면 불활성 = 꺼짐 해시 주장이 깨진다, 설계 순서절).
- `test_inert_isolated_hash_equals_off_still_holds`: 과제 6의 불활성 해시 시험이 격자 재구성을 켠 뒤에도 같다(격자 배열이 해시 밖인지 확인).
- `test_watch_pun_off_not_called`.
- 빨강 확인.

**구현**: `watch_pun(step, s, fs, isp)` 순차(`prange` 금지): 이웃 순회는 `kernels.watch_meals`의 3×3 칸 순회와 좁은 격자 중복 방지(`oy == 1 and gh <= 2` 건너뛰기)를 그대로. 반지름은 `F_NORM_RADIUS`. `blind`는 두 패스(쌍 수 세기 → 동전). `world.step`: `st.attack` 호출 바로 뒤에 `if cfg.norm_enabled: k.build_grid(...); st.watch_pun(t, s, fs, isp); st.norm_account(...)`. 이 순서가 `process_deaths` 앞. **`soc_taste_enabled`가 켜지면 `eat` 뒤에 이미 격자를 다시 지어 `give`·`attack`이 새 격자를 본다**(`soc` 갈래) → `soc_*`·`taste_*`는 모든 규범 갈래에서 같게 박는다(체인의 사전 검사, 과제 18).

**초록 확인**: `python3 -m pytest -q tests/test_norm.py tests/test_stages.py tests/test_soc_taste.py`. 커밋: "규범의 씨앗 12: watch_pun·눈가림·attack 뒤 격자 재구성".

---

## 과제 13. 벌주기 막기 (`norm_block_tag`·`norm_block3_tag`)
**바꾸는 파일**: `kkumteul/stages.py`(`attack`의 대상 점검), `tests/test_norm.py`.

**시험 먼저**(설계 시험 13)
- `test_block_tag_cannot_punish_marked_and_pays_nothing`: `norm_block_tag = 2`인 태그의 개체는 **자발 공격**이 표시된 대상을 겨눌 때 공격이 취소된다(에너지·비용·장부·계수 모두 변화 없음). 표시 안 된 이웃은 평소대로 친다.
- `test_block3_allows_self_defense_only`: `norm_block3_tag = 3`인 태그는 표시된 대상이 **나에게 가로챈 것**(`mark_vid[대상] == ids[나]`)이면 친다(2자 허용), 남이 피해자(또는 피해자 없음 −1·미지 −2)이면 못 친다.
- `test_block_minus_one_has_no_effect`: 기본 −1은 해시·행동이 꺼진 세계와 같다.
- `test_block_does_not_cancel_impulse`: 충동(떨리는 손)의 가로채기는 막기와 무관하다(충동은 표시를 모른다 — 벌이 아니라 위반 공급이므로. **설계가 충동의 막기 여부를 안 적어 이 계획이 정한다**).
- `test_blocked_opportunity_still_counted`: 막혀서 못 친 기회도 `OPP_*`·선택성 분모에는 센다(기회는 있었다), `ATK_VOL_*`에는 안 센다(공격이 안 나갔다).
- `test_block_requires_norm_enabled`(과제 1 검사의 재확인).
- 빨강 확인.

**구현**: `attack` 순서 3(대상 고르기)와 4(분기) 사이: 자발이고 대상이 표시돼 있고 `isp[I_NORM_BLOCK] == tag[i]`면 취소, `isp[I_NORM_BLOCK3] == tag[i]`이고 `mark_vid[대상] != ids[i]`면 취소(비용 내기 **앞**에서 — "비용도 안 든다"). 취소 때는 대체 대상(가장 가까운 비표시 이웃)으로 넘어가지 않는다("집행 능력만 뺀 쌍둥이").

**초록 확인**: `python3 -m pytest -q tests/test_norm.py tests/test_stages.py`. 커밋: "규범의 씨앗 13: 벌주기 막기(전부·3자만)".

---

## 과제 14. 겨루기 마당 준비·다태그 겨루기·공격 행 다시 뽑기
**바꾸는 파일**: `kkumteul/measure.py`(마당 준비 함수·다태그 겨루기·순수입 계산), `kkumteul/world.py`(`reseed_attack_row`), `kkumteul/run.py`(`branch` 옵션), `tests/test_norm.py`, `tests/test_stages.py`(갈래 시험 한 줄 확인만).

**시험 먼저**(설계 시험 18과 "마당 준비")
- `test_arena_side_resized_to_norm_size`: 합성 체크포인트(world4 꼴의 작은 세계, `conflict`·`norm_size` 없음)를 저장하고 `measure.norm_arena_side(path)`(= `ckpt.load(path, cfg + preset=norm_size)` + `side_from_world`; 선례 `wayfinder/stage2-4/hand_learner.py`의 `taste` 모드)가 만든 쪽이 `n_in`이 올라가 `check_same_shape`를 통과한다. **같은 체크포인트를 cfg 없이 `measure.load_side`로 읽으면 못 통과**하는 것도 확인(`n_in` 37 대 72 꼴 — 첫째 판의 가장 큰 구멍).
- `test_resident_attack_row_fixed_and_arena_has_no_violence`: 마당용 유전체에서 `bout[:, ACT_ATTACK] == −1`, `Wout[:, ACT_ATTACK, :] == 0`이고, `norm_impulse = 0`인 겨루기 마당(거주민만)을 몇백 스텝 돌리면 `TAKE == 0`·`PUN == 0`(**관문 −1e**: 음성 대조의 전제). 합성 체크포인트의 공격 편향을 **+7로 일부러 틀리게** 해서 편향만 고치면(`world.calm_attack` 꼴) 가로채기가 남아 실패하고 출력 행까지 0으로 하면 통과하는 것을 한 번 확인한다(`Wout·h`가 자라며 양수가 되는 길이 있다).
- `test_compete_multi_groups_and_same_seed_pairs`: `measure.compete_multi(cfg, groups, steps, seed)`(`groups = [(genomes, tag, count), ...]`)가 태그별 개체 수대로 넣고(`n_total > max_agents`면 오류), 같은 `seed`는 같은 위치·방향·나이 배치와 같은 결과를 낸다(칸마다 같은 마당 난수 — 같은 씨앗 짝으로 칸끼리 비교해 검출 한계를 줄인다), 넣는 순서를 섞어 칸 번호가 한 태그에 유리하지 않다. 기존 `measure.compete`·`assay_sides`는 **그대로**다(기존 시험 통과).
- `test_compete_multi_reports_norm_tables`: 결과에 태그별 `norm_by_tag`(`NT_NAMES` 사전)·`role_by_tag`·태그별 산 개체 수·끝 시점 `eaten` 합·`mark_prevalence`가 들어 있다. 겨루기 세계는 `mutation_enabled = False`(`arena_config`가 이미 끈다).
- `test_net_income_per_step_by_tag_hand_values`: `measure.net_income_by_tag(world)` = (`EATEN_DEAD` + 산 개체 `eaten` 합 + `TAKE_GAIN` + `PUN_FINE` − `TAKE_DMG_RECV` − `PUN_DMG_RECV` − `ATK_COST`) ÷ `ALIVE_STEPS`(개체 스텝당 순수입, 태그별)가 합성 값에서 손셈과 같다. **설계의 "생애 에너지 차"(`N_ind`)를 개체 스텝당 순수입의 차로 정의하고(같은 몸 쌍둥이끼리라 수명 차이는 스텝으로 정규화), 한 생 값은 × 평균 수명(1,560스텝, 설계 "판정 −2")으로 환산해 설계의 +0.05~+0.3과 견준다.**
- `test_reseed_attack_row_matches_new_agent_distribution`: `World.reseed_attack_row(b0=-1.0, seed)`가 산 개체의 `bout[:, ACT_ATTACK] = b0`(−1, 새 개체의 값)와 `Wout[:, ACT_ATTACK, :]`을 새 개체의 초기 분포(`init_wout` × 표준정규)로 다시 뽑는다. 다른 출력 행·다른 배열은 그대로이고(전후 비교), 평균 0·표준편차 ≈ `init_wout`(통계 허용 오차). 난수는 `numpy` 생성기(`cfg.seed`에서 파생, 선례: `load_arrays`의 `alpha` 다시 뽑기)라 `rng.py` 용도 번호를 안 쓴다. `b0 = −0.3`(기울기 시작판, 관문 −2b가 쓴다)도 받는다.
- `test_branch_reseed_option_recorded_and_applies_to_all`: `Runner.branch(..., reseed_attack=True, attack_b0=-1.0)`와 CLI `--reseed-attack-row [--attack-b0 −0.3]`가 갈래 시작에 이 편집을 걸고 **meta의 `edits`에 남긴다**. 갈래가 새로 `conflict`를 켜 `attack_newly_on`의 `calm_attack`(편향 −1)이 이미 걸려도 그 **뒤에** 걸려 겹쳐도 결과가 정의된다. `--reseed-attack-row` 없는 갈래는 이전과 똑같다.
- `test_branch_resizes_to_norm_size`: 옛 크기 체크포인트에서 `preset=norm_size`로 갈라도 `ckpt.load`가 크기를 올려 읽고(새 입력 열의 가중치는 0) 이어 돈다(`Runner.branch`가 `enlarged_in`을 처리).
- 빨강 확인.

**구현**
1. `measure.py`: `norm_arena_side(path, sets=())`, `fix_resident_attack(genomes)`(사본, 편향 −1·출력 행 0), `compete_multi(...)`, `net_income_by_tag(world)`. `inject_agent`는 과제 2에서 이미 고쳤다.
2. `world.py`: `reseed_attack_row(b0, seed)`. `run.py`: `Runner.branch`에 `reseed_attack`·`attack_b0`와 CLI 옵션. `edits` 목록에 `"reseed_attack_row(b0=...)"`.

**초록 확인**: `python3 -m pytest -q tests/test_norm.py tests/test_stages.py tests/test_taste.py`. 커밋: "규범의 씨앗 14: 마당 준비·다태그 겨루기·공격 행 다시 뽑기".

---

## 과제 15. 손 집행자와 양성 확인
**바꾸는 파일**: 새 `wayfinder/stage2-4/norm_hand.py`, `tests/test_norm.py`.

**시험 먼저**(설계 시험 17 앞쪽; 작은 마당)
- `test_hand_enforcer_turns_to_marked_neighbor_and_hits`: 손으로 짠 집행자(**태그 1**)가 표시된 이웃(`mark_t` 손으로 붙임)이 시야 한쪽에 있을 때 그 쪽으로 돌고 앞에 닿으면 공격을 낸다(**양성 확인을 먼저** — 이웃이 보이고 돌고 치는 것까지 몇 스텝 안에).
- `test_hand_enforcer_blind_when_norm_vis_blocked`: 같은 시나리오에서 `knockout(cfg, "norm_vis", tag=1)`로 `NORM_VIS`를 막으면 **못 한다**(양성 대조: 이 배선 누락이 막기 채널의 `norm_sense` 감싸기 누락을 잡는다).
- `test_hand_tags_share_one_body`: 태그 1·2·3은 **같은 유전체**(마당에서 태그만 다르다: 태그 2는 `norm_block_tag = 2`, 태그 3은 `norm_block3_tag = 3`)이고 거주민(태그 0)도 빈 숨은 칸 하나를 더 갖는다(뇌 크기 비용을 같게 — `hunt_hand.py`의 `plant`가 A·B 둘 다 한 칸을 더하는 이유와 같다).
- `test_hand_resident_attack_zero`: 거주민은 공격 행이 고정돼 마당에서 공격이 0이고(과제 14), 손 집행자의 공격은 손 배선이 정한다.
- `test_hand_skips_full_hidden_agents_and_reports_count`: 숨은 칸이 가득 찬 개체는 빼고 뺀 수를 돌려준다(`hunt_hand.plant`와 같은 규칙).
- `test_hand_requires_vision_sectors_8`.
- 빨강 확인.

**구현**: `hunt_hand.py`의 `plant`를 본보기로 `plant_enforcer(g, enforcer: bool)`: 빈 숨은 칸 하나 `h_new = tanh(G × (표시 시야 앞쪽 칸 값 − θ))`, 집행자는 공격 출력 = `tanh(2 h_new − 1)`, 회전 출력에 `NORM_VIS` 방향 칸별 가중치(가장 센 방향 쪽으로 돈다 — 회전 부호는 `kernels.move`의 `ACT_TURN` 규칙을 읽고 위 양성 시험으로 확인). 앞쪽 칸 = 시야 가운데 두 칸. **θ·G는 "밝기 ≥ 0.5(경과 ≤ τ/2)인 표시를 닿는 거리(중심 거리 2.5, 가까움 0.906)에서 칠 것"으로 정하고** 밝기가 낮은 늙은 표시에서는 덜 칠 수 있다는 한계를 결과 문서에 적는다(설계 약점 6: 손 배선이 임의다). 거주민은 같은 빈 칸을 더하되 공격에 안 닿는다.

**초록 확인**: `python3 -m pytest -q tests/test_norm.py tests/test_knockout.py`. 커밋: "규범의 씨앗 15: 손 집행자와 양성 확인".

---

## 과제 16. 섞기 (`norm_mix`: 친족·반복 만남 빼기)
**바꾸는 파일**: `kkumteul/stages.py`(`norm_mix`), `kkumteul/world.py`(`step`), `tests/test_norm.py`.
(설계의 구현 순서는 손 집행자·판정 도구 뒤 ⑥이지만, 판정 도구의 "섞은 마당" 칸이 이 장치를 쓰므로 **판정 도구(과제 17) 앞**으로 옮겼다.)

**시험 먼저**(설계 시험 12)
- `test_mix_only_on_period_steps_with_probability_f`: `norm_mix_period = T`일 때 정확히 `step > 0 and step % T == 0`인 스텝에만 옮기고(스텝 0은 안 옮김), 개체마다 확률 `norm_mix_frac`(f)로 옮긴다(큰 표본, 허용 오차).
- `test_mix_xy_independent_and_uniform`: 난수는 한 개체에 셋: `u01(seed, S_NORM_MIX, step, 3 × id + k)`, k = 0(옮길지), 1(x), 2(y)(서로 다른 해시라 x·y가 상관되지 않는다 — 인자 배치를 문서화). 옮겨진 개체의 x·y 표본 상관 |r| < 0.05, 세계 안 균등.
- `test_mix_conserves_energy_and_zeroes_velocity`: 옮긴 개체의 속도는 0, 에너지·장부는 그대로(에너지 합 불변: 위치만 바뀜), 다른 필드(나이·표시·기억)는 그대로.
- `test_mix_runs_before_first_build_grid`: `world.step`에서 `norm_mix`가 **스텝 맨 앞**(`niche_world`·`spawn_food`·첫 `build_grid` 앞)이라 낡은 격자를 보는 커널이 없다(호출 순서를 감싸 기록).
- `test_mix_deterministic_same_hash`: 같은 시나리오 두 번이 같은 해시, `norm_mix_period = 0`이면 `norm_mix` 호출 0, 꺼진 세계와 해시가 같다.
- `test_mix_breaks_parent_child_adjacency`: 부모·자식이 닿는 거리(< 2.5)에 있던 쌍이 한 번 섞은 뒤에도 닿을 비율이 ≈ (1 − f)²(어림, 둘 다 안 옮긴 경우)로 내려간다(효과의 방향 확인).
- `test_mix_incompatible_with_carry_and_teach`(과제 1 검사의 재확인: `carry_enabled`·`teach_enabled`와 같이 못 켠다 — 들기는 `collide` 뒤에야 먹이 위치를 동기화해 순간이동 뒤 낡은 위치를 보고, 가르침의 곁 출생·`par_near_frac`는 순간이동이 깬다).
- 빨강 확인.

**구현**: `norm_mix(step, s, fs, isp)` 순차: 산 개체마다 위 난수대로 확률 f면 `x`·`y`를 세계 안 무작위로, `vx = vy = 0`. `world.step` 맨 앞에 `if isp[I_NORM_MIX_PERIOD] > 0 and t > 0 and t % period == 0: st.norm_mix(...)`. 임시 결정 T = 200, f = 0.2(한 생 약 7.8번 섞여 이웃의 약 82%가 낯선 개체로 바뀐다, 설계 6절). 두 몫(친족·반복 만남)을 따로 못 가른다는 한계를 주석에 적는다.

**초록 확인**: `python3 -m pytest -q tests/test_norm.py tests/test_stages.py tests/test_harsh.py`. 커밋: "규범의 씨앗 16: 섞기(norm_mix)".

---

## 과제 17. 판정 도구: 순수 계산(`norm_gate.py`)과 명령(`norm_judge.py`)
**바꾸는 파일**: 새 `wayfinder/stage2-4/norm_gate.py`(순수 계산), 새 `wayfinder/stage2-4/norm_judge.py`(명령줄), 새 `tests/test_norm_judge.py`. (`taste_gene_judge.py`는 안 고친다 — 가르침 과제 9와 같은 파일을 두 곳에서 건드리지 않는다. `measure.t95`·`knockout._neutral`·`taste_gene_judge`의 Welch 패턴은 가져다 쓴다.)

**시험 먼저**(`tests/test_norm_judge.py`; 시뮬레이션 없이 도는 것을 최대로)
- `test_cells_table_complete`: `norm_judge.CELLS`가 설계 판정 0 표의 칸을 이름·설정·마당 종류·φ로 정의한다: `hand-phi0.1/0.3/1.0`, `hand3-phi0.3`, `hand-phi0.3-inert/-nearest/-ret0/-ret0.5/-inh0/-inh0.4/-L1/-L10/-tau1/-sig(ret0·ret0.1)/-none-q(ret0·ret0.1)/-noise0.0005/-noise0.002`, `off`(`norm_enabled=false`, 충동은 켠 채), `noimp`(`norm_impulse=0`), 기준선 `base`(φ = 0, 전원 거주민), 전원 마당 `grp-1/grp-2/grp-3`(φ = 1.0). **설계의 `sig`/`none-q`는 "각각 `ret0`·`ret0.1`"이라 적었는데 판정 δ는 `ret0`에서 읽으므로 두 `ret` 값을 모두 도는 네 칸으로 읽는다**(설계 작성 의도와 어긋나면 고친다). 핵심 칸 목록(`CORE`)과 나머지(`REST`) 구분.
- `test_arena_composition_and_settings`: 마당 갑(φ 0.1·0.3: 태그 1 = φ·800, 태그 2 = φ·800, 나머지 거주민 태그 0, `norm_block_tag = 2`)·마당 을(φ 0.3: 태그 1·태그 3 = 240씩, `norm_block3_tag = 3`)·전원 마당(전원 태그 1 / 전원 태그 2 / 전원 태그 3)·`base`의 개체 수와 설정이 설계 그대로. 섞은 마당은 같은 구성에 `norm_mix_period = 200`·`norm_mix_frac = 0.2`. 모든 칸은 `mutation_enabled = False`이고 같은 칸 안 태그끼리 한 마당에서 읽는다.
- `test_damage_rate_definition_mode_independent`: **위반 피해율 = 때린 순간 표시가 없던 대상이 받은 피해 ÷ 개체 스텝**이 합성 계수에서 손셈과 같고, `inert`에서 표시 있던 대상을 친 피해(`TAKE_ON_MARKED`)는 빠진다(`full`·`inert`·`off`를 같은 정의로 견준다, 판단표 18).
- `test_net_income_levels_hand_values`: `N_ind` = 태그 1 − 태그 2의 개체 스텝당 순수입(마당 갑), `N_ind3` = 태그 1 − 태그 3(마당 을), `N_grp`·`N_grp3` = 전원 마당 쌍의 끝 개체 수 로그 비와 평균 에너지 차(같은 씨앗 반복 6쌍), 한 생 환산(× 1,560), 반복별 t 구간(`measure.t95`, df = n − 1)이 손셈과 같다. 2차 딜레마의 크기는 `N_grp3 > 0 ≥ N_ind3`일 때 `N_grp3`.
- `test_gate_alpha_beta_gamma_delta`: **관문 α**(작동): `hand-phi0.3`의 위반 피해율이 `base`보다 Welch로 낮고 7% 이상 낮으며, φ 단조(6반복 중 5반복 이상 `0.1 ≥ 0.3 ≥ 1.0`), `inert`·`off`가 `base` ± 검출 한계(양쪽 전멸 2판 이하) → "작동", 음성 대조가 깨지면 "도구 깨짐". **관문 β**(가·나·다): (가) `N_ind` 아래끝 > 0, (나) `N_ind3` 아래끝 > 0 — **"벌주기 씨앗이 이득이 된다"는 (나)가 0 위일 때만**, (다) `N_grp3` 아래끝 > 0, 2차 딜레마 = (나) ≤ 0 이고 (다) > 0. 관문 −2의 검출 가능성이 1 아래인 칸은 β를 안 쓴다("검출 한계 아래"). **관문 γ**(첫걸음): `hand-phi0.1`이 `base`보다 낮고 `inert`와 가를 수 있을 때만 "만족", `L1` 칸이 `L5`와 같으면 "기울기는 억지가 아닌 곳에서", `L1`이 0이면 "상습 위반자 가정에서". **관문 δ**(임의 대 기능): `ret0`에서 `sig`·`none-q`의 표시당 집행 확률이 `take`보다 Welch로 낮고 `N_ind`도 작으면 "기능에서 값을 얻는다", 같으면 "표시 자체가 약탈 대상"; 세 원천의 끝 유병률이 2배를 넘게 갈리면 "유병률이 달라 가를 수 없음". 합성 결과로 "작동 + 이득"·"작동 + 이득을 가를 수 없음"·"작동 아님"·"도구 깨짐" 네 경우.
- `test_detectability_minus2_and_threshold`: **관문 −2**: 예상 순이득(`N`)과 검출 한계(음성 대조 칸 반복별 순이득 차의 sd × t95 ÷ √n)의 비가 1 아래인 d는 β를 안 쓴다. 진화 선택 강도 `N × 순이득 ÷ 한 생 수입(9.8)`이 1 이상인지(N = 무리 크기 입력). **관문 −2b**: 저장된 `h`로 근사한 `pre-act > 0`인 개체 비율(문턱 위 비율)과 `NORM_VIS` → 공격·회전 출력 가중치 분포를 계산하고, 1% 아래면 기울기 시작판(`b0 = −0.3`) 플래그를 올린다.
- `test_gate_world_minus1`: −1a(켠 무리 ≥ 끈 무리의 0.9·재주입 급증/멸종 없음·장부 통과), −1b(`θ_t` = `EXPO ÷ 개체 스텝`, `take_loss_frac` = 맞은 피해 ÷ 먹이 수입이 0.08~0.15면 통과, 밖이면 `norm_impulse` 조정 권고 — 최대 세 번), −1c(표시 유병률 0.5%~10%, 세 표지 원천 유병률이 서로 50% 이내, 넘으면 τ 50 → 25 → 12 권고), −1d(표시된 개체 근처 개체의 `NORM_VIS`가 0이 아닌 비율이 시야 안 표시 존재 비율과 맞음), −1e(거주민 공격 0)가 합성 값에서 각각 통과·실패로 갈린다. 종료 코드: 0 통과, 3 세계 못 섬(−1a·−1c·−1d·−1e), 4 딜레마 크기 밖(−1b).
- `test_bottom_judgment`: `p2 = PUN_2ND ÷ OPP_SELF`, `p3 = (PUN_3_KIN + PUN_3_STR) ÷ OPP_OTHER`, **친족 부풀림** = (`PUN_3_KIN ÷ (PUN_3_KIN + PUN_3_STR)`) ÷ (3자 기회의 친족 비율), 2자 대 3자 몫을 **같은 끝 무리의 표준 마당·섞은 마당** 값의 차로 낸다(서로 다른 두 진화 갈래의 계수기 차는 "단서"로만 — 판정 문장에 못 쓴다). 닮음 구간 ⒜와 계보 ⒝가 **같은 쪽으로 읽힐 때만** 친족 부풀림을 쓴다. 평판(늦은 집행 비율: 경과 10~τ ÷ 0~10)은 `tau1` 칸과 견준다.
- `test_main_judgment_sentence_and_codes`: **판정 나**(주): 같은 끝 무리를 태그 A(전부 못 침)·B(3자만 못 침)·C(원래)로 나눈 겨루기 `r(C−A)`·`r(C−B)`·`min_detectable`, `N_ind`·`N_ind3`(표준·섞은 마당), 위반 피해율 태그별 차, **선택성**(`ATK_VOL_MARKED ÷ OPP_MARKED` ÷ `ATK_VOL_PLAIN ÷ OPP_PLAIN`)의 구간 아래끝. 주 판정 문장 "**`N_ind3`(또는 `N_ind`)의 구간 아래끝이 0 위, 선택성 아래끝이 1 위, 감각 막기 겨루기가 `inert` 끝 무리와 가를 수 있는 방향, `inert` 끝 무리 null과 구분됨, 음성 대조(위반 공급 끔)가 0 ± 한계, 두 씨앗이 같은 쪽**"일 때만 "그 수준(2자 맞대응 / 3자 집행)의 벌주기 씨앗이 개인에게 이득"이다. `N_grp3`만 0 위면 "모두가 하면 집단이 이득인데 개인이 이득이 아닌 칸(2차 딜레마가 열려 있음)"으로 **따로** 쓴다(OR로 묶지 않음). A쪽이 비용을 안 내 유리하므로 이기거나 비기면 "벌주기 값이 이득보다 크거나 같다"까지만. 합성 결과에서 이 조합들과 한 바탕이라도 통과/모두 "가를 수 없음"/깨짐 종료 코드(0·3·4).
- `test_output_labels_fixed`: 출력에 한계 문구가 붙는다 — 구간표에 "조작 확인용(엔진 위축이 있으면 규범 없이도 낮게 나온다)", ④ "부여된 위축 아래서", 판정 라에 "이득 구조 확인이지 관습이 아니다", 결과 문서 머리 한계 문구 상수(과제 18이 그대로 옮긴다).
- `test_observe_window_resets_and_reports`: 작은 세계 `observe --steps N`이 창 시작에 `reset_norm_tables()`하고 표시 유병률·`θ_t`·`take_loss_frac`·표시당 집행 확률·`p2`·`p3`·구간표·이웃 쌍 대 아무 쌍의 색 닮음 분포(친척 경계를 사후에 정하는 자료, 과제 8)를 낸다(장치를 켠 설정일 때만).
- 빨강 확인.

**구현**
1. `norm_gate.py`(순수): 위 계산 — `damage_rate`, `net_income_diff`, `level_intervals`, `gate_alpha`, `gate_beta`, `gate_gamma`, `gate_delta`, `detectability`, `threshold_above`, `gate_world`, `bottom`, `main_judgment`, `combine_tags`. 문턱은 코드 상수 한 곳(`THRESHOLDS`)에 두고 결과 문서의 예측 절과 **같은 값**이다.
2. `norm_judge.py` 하위 명령: `threshold <체크포인트>`(−2b, 시뮬레이션 없이 저장된 `h`로), `observe <체크포인트> --steps N`, `arena-check <체크포인트>`(−1e), `hand-dose <체크포인트> --cell 이름 --arena std|mix`(판정 0 한 칸: 마당 구성대로 `compete_multi`를 반복 6번 + 같은 씨앗 짝, 중립 6은 `base` 칸에서만), `gate-world --dir <폴더> --tags a b`(−1), `gate-hand --dir <폴더>`(α·β·γ·δ·−2), `block-compete <끝 체크포인트>`(판정 나: 태그 A/B/C 겨루기 + 선택성 + `inert`·`off` 끝 무리 null 인자), `bottom <끝 체크포인트>`(판정 다·라), `final --dir <폴더> --tags a b`(주 판정 문장과 종료 코드). 원자료 이름 규칙은 과제 18의 체인.
3. **문서의 실제 사용 예를 시간 제한을 걸고 직접 돌린다**(시험 통과만으로 "됐다"고 하지 않는다): 시험 보조 함수로 만든 작은 세계의 체크포인트(`preset=norm`)를 저장하고
   `PYTHONPATH=. timeout 600 python3 wayfinder/stage2-4/norm_judge.py hand-dose <작은 체크포인트> --cell hand-phi0.3 --arena std --n 60 --steps 400 --reps 2 --json <임시 폴더>/x.json`
   결과 JSON에 태그별 `norm_by_tag`와 위반 피해율이 들어 있고 `gate-hand`가 합성이 아닌 이 파일을 읽어 오류 없이 끝나는지까지 본다. 서버의 큰 실행은 건드리지 않는다. 멈추면 내 코드부터 의심한다.

**초록 확인**: `python3 -m pytest -q tests/test_norm_judge.py tests/test_norm.py tests/test_knockout.py`. 커밋: "규범의 씨앗 17: 판정 도구(norm_gate·norm_judge)".

---

## 과제 18. 체인 스크립트 초안·예측 기록·전체 시험
**바꾸는 파일**: 새 `wayfinder/stage2-4/norm_chain.sh`, `wayfinder/stage2-4/results.md`("규범의 씨앗 — 예측" 절 — 체인을 돌리기 **전에**), `docs/status.md`(막힌 점·다음 후보 칸 한 줄과 마지막 갱신일).

**순서 규칙(예측은 실행 전에)**: ① 아래 3의 예측 절을 `results.md`에 적고 **커밋·push한 뒤에** ② 체인을 돌린다. 체인 스크립트는 첫머리에서 `results.md`에 "규범의 씨앗 — 예측" 절이 없으면 멈춘다(아래 초안의 `grep`). 문턱은 실행 중에 바꾸지 않는다(바꾸면 결과 문서에 "예측 바꿈"을 날짜와 함께 따로 적는다).

**작업**
1. `norm_chain.sh`를 아래 초안대로 쓴다. `bash -n wayfinder/stage2-4/norm_chain.sh`와 `DRY=1`로 명령이 맞는지 본다(경로는 변수, 레포 상대 경로만).
2. 시간 어림은 **추정**이고 미측정이다: 구현 끝에 작은 갈래(3,000스텝)와 겨루기 한 칸(800마리 × 8,000스텝)의 실제 시간을 재서 스크립트 머리 주석을 고친다. 판정 0 핵심 칸 열한 개를 표준·섞은 마당에서 반복 6씩 돌리는 것이 가장 크다 — 시간이 모자라면 `CORE`만 먼저 돌리고 `REST`는 뒤로 미룬다(사회 학습 체인이 핵심 넷을 두 바탕 먼저 돌린 것과 같은 방식).
3. `results.md`에 **"규범의 씨앗 — 예측" 절**을 쓴다(실행 전 날짜·시각 ET 표시): (a) **첫 줄에 한계를 한 번에**: "①②와 벌의 물리·위축은 부여, ⑤⑥과 벌주기 성향은 측정, ④의 뇌 삼감과 소통·전달은 이 판 밖. 위반은 엔진이 주는 떨리는 손(뇌가 못 말림)이고 표시는 정직·즉시·전원 공개의 엔진 방송이다. 이 판에서는 '규범'이라는 말을 쓰지 않는다(쓰는 말: 벌주기 씨앗, 집행, 맞대응 방어)." (b) 설계 판정 절의 **예측 표 그대로**(`θ_t` 0.2~0.5, 유병률 5~13%(τ = 50) 등 — 숫자를 옮길 뿐 새로 정하지 않는다), 억지 셈(s = 1~3, 위축 0.8, 구간 L = 5)과 "L은 계산의 입력이 아니라 모형의 가정", (c) **판정 규칙 옮기기**: 관문 −1a~e·−2·−2b·α·β(가·나·다)·γ·δ, 진화 판정 문장(주 판정 나)·판정 다·라, 단서의 정의, τ 조정 규칙(50 → 25 → 12), `norm_impulse` 조정(최대 세 번), 문턱 상수(`norm_gate.THRESHOLDS`와 같은 값), (d) "서로 다른 두 진화 갈래의 계수기 차는 단서"(형질 추세 금지 규칙), (e) 이번 판 밖(섬·5단계·다음 후보)은 한 줄.
4. `docs/status.md`에 이 단계의 막힌 점·다음 후보 칸 한 줄과 마지막 갱신일을 고친다(관찰에는 잰 값만, 해석에는 추정만).
5. 전체 시험을 한 번: `python3 -m pytest -q`(메모리 상한 도구 `--max 2G`, `NUMBA_NUM_THREADS=2`, `nice`). 실패는 이 과제에서 고친다. 박은 해시 시험이 **모두 그대로**여야 한다.
6. **제출 전 직접 실행**: `norm_chain.sh`를 `DRY=1`로 돌려 명령 목록을 본다(실제 체인은 서버 상황 — 사회 학습·가르침 체인이 안 돌고 있는가 — 을 보고 정한다. **한 번에 하나, 2스레드, 메모리 상한 1,500M, `nice -n 15`.**)
7. 체인 결과를 읽을 때: 판정 문장은 설계 그대로. 결과 문서에 판정을 적을 때 `docs/status.md`의 관찰·기각 칸과 마지막 갱신일도 같이 고친다. `observe`의 값으로 설계의 셈(`θ_t`, 유병률 8%)을 실측으로 확인한다.
커밋: "규범의 씨앗 18: 체인 스크립트 초안·예측 기록". (예측 절만 먼저 별도 커밋으로 올려도 된다 — 체인을 돌리기 전에 올라가 있어야 한다.)

### 체인 스크립트 초안 (`wayfinder/stage2-4/norm_chain.sh`)
```bash
#!/bin/bash
# 규범의 씨앗(가로채기·벌주기) 실험 체인. 설계: docs/superpowers/specs/2026-11-26-norms-seed-design.md 둘째 판.
# 예측·문턱: wayfinder/stage2-4/results.md "규범의 씨앗 — 예측"(실행 전에 적어 둔다. 없으면 이 스크립트는 멈춘다).
# 앞 실험(사회 학습·가르침 체인 포함)이 끝난 뒤 한 번에 하나(2스레드, nice).
# 순서: 바탕 확인 → 관문 −2b(문턱) → 세계 점검(−1) → 판정 0(용량 반응, 핵심 칸 먼저) → 관문(α·β·γ·δ·−2) → 진화 갈래 → 판정 나·다·라·단서.
#
#   ROOT=<레포 원래 폴더> CODE=<커밋한 코드 사본 폴더> [BASE=<바탕 체크포인트>] [WAIT_FOR=<pgrep -f 패턴>] [DRY=1] [OUT=<기록 폴더>] \
#     <메모리 상한 도구> --max 1500M --bg norm-chain --log "$ROOT/runs/norm-exp/chain.log" -- bash wayfinder/stage2-4/norm_chain.sh
# 상한은 묶음 전체에 걸린다. 칸·갈래·판정마다 done/json 표시가 있어 다시 불러도 이어 간다.
# DRY=1이면 명령만 찍고(표준출력) 아무것도 돌리지 않는다. 이때 기록 폴더는 임시 폴더(OUT으로 바꿀 수 있다).
# 시간 어림(미측정 — 과제 18에서 작은 갈래·겨루기 한 칸을 재서 고친다): 세계 점검 여덟 갈래, 판정 0 핵심 열한 칸 × 표준·섞은 마당,
#   진화 갈래 한 개당 30만 스텝(1단계 세계보다 입력이 72칸이라 더 느릴 것). 관문에서 닫히면 판정 0까지만 돈다.
set -u
R=${ROOT:?ROOT=레포 원래 폴더}
C=${CODE:?CODE=코드 사본 폴더}
if [ -n "${DRY:-}" ]; then
  E=${OUT:-$(mktemp -d)}
  exec 3>&1                      # DRY 줄은 명령별 리다이렉트(> 로그)와 상관없이 화면(표준출력)으로
else
  E=${OUT:-$R/runs/norm-exp}
fi
PY=$R/.venv/bin/python
J=wayfinder/stage2-4/norm_judge.py
BASE=${BASE:-$R/runs/world4/checkpoints/ckpt_000089611957.npz}   # 바탕 B1(마음·사회·판, 공격 선택압 없음)
bn=$(basename "$BASE" .npz); S0=$((10#${bn#ckpt_}))
W_END=$(printf 'ckpt_%012d.npz' $((S0 + 30000)))     # 세계 점검 갈래(3만 스텝) 끝
END=$(printf 'ckpt_%012d.npz' $((S0 + 300000)))      # 진화 갈래(30만 스텝) 끝
export NUMBA_NUM_THREADS=${NUMBA_NUM_THREADS:-2} PYTHONPATH=.
mkdir -p "$E"
note() { echo "$(TZ=America/New_York date '+%m-%d %H:%M ET') $*" | tee -a "$E/progress.txt"; }
runit() {  # DRY=1이면 명령만 찍는다
  if [ -n "${DRY:-}" ]; then echo "DRY: $*" >&3; return 0; fi
  "$@"
}

if [ -n "${WAIT_FOR:-}" ] && [ -z "${DRY:-}" ]; then
  while pgrep -f "$WAIT_FOR" >/dev/null; do sleep 30; done
  note "앞 실험 끝남 확인 — 체인 시작"
fi
cd "$C" || exit 1
grep -q "규범의 씨앗 — 예측" wayfinder/stage2-4/results.md || { note "예측 절이 results.md에 없음 — 체인 중단"; exit 1; }

pre_check() {  # 바탕 체크포인트 설정이 설계가 기대는 세계인지(다르면 멈춘다)
  [ -f "$BASE" ] || { note "바탕 체크포인트 없음: $BASE"; return 1; }
  [ -n "${DRY:-}" ] && return 0
  "$PY" - "${BASE%.npz}.json" <<'PYEOF' || return 1
import json, sys
c = json.load(open(sys.argv[1]))["config"]
want = {"mind_enabled": True, "social_enabled": True, "niche_enabled": True, "conflict_enabled": False,
        "vision_sectors": 8, "n_spare_in": 13, "carry_enabled": False, "taste_enabled": False,
        "soc_taste_enabled": False}
bad = {k: (c.get(k), v) for k, v in want.items() if c.get(k) != v}
if bad:
    print("체크포인트 설정이 기대와 다름:", bad)
    sys.exit(1)
PYEOF
}

# 모든 갈래·점검에 같은 설정을 박는다(soc_*·taste_*는 갈래마다 섞지 않는다: 격자 재구성이 attack 이웃을 바꾼다)
G="preset=conflict preset=norm_size norm_impulse=0.015"

# ---- 관문 −2b(진화 시작점의 공격 문턱): 시뮬레이션 없이 저장된 h로 문턱 위 개체 비율을 잰다
if [ ! -s "$E/threshold.json" ]; then
  runit "$PY" $J threshold "$BASE" --json "$E/threshold.json" > "$E/threshold.log" 2>&1
  note "관문 −2b: $(tail -1 "$E/threshold.log" 2>/dev/null | cut -c1-160)"
fi
# 문턱 위가 1% 아래면(threshold.json의 slope_start) 진화 갈래에 기울기 시작판(--attack-b0 -0.3) 넷을 더한다(아래 4번)
SLOPE=$("$PY" -c "import json; print(int(json.load(open('$E/threshold.json')).get('slope_start', 0)))" 2>/dev/null || echo 0)

# ---- 세계 점검(판정 −1): 같은 체크포인트에서 갈라 3만 스텝, 끝에서 5,000스텝 창을 읽는다(씨앗 a·b)
wbranch() {  # 씨앗 이름 설정...   (이름: meas=끔+계수만, inert, full, noimp=위반 공급 끔)
  local seed=$1 name=$2; shift 2
  local dir=$R/runs/n-w-$name-$seed ck out=$E/w-$name-$seed
  local extra=(); [ "$seed" = b ] && extra=(--new-stream)
  local kv sets=(); for kv in "$@"; do sets+=(--set "$kv"); done   # 반복 변수를 local로(호출자의 s를 덮어쓰지 않게)
  if [ ! -f "$dir/done" ]; then
    note "세계 $name-$seed 시작 ($*)"
    runit nice -n 15 "$PY" -m kkumteul.run branch --from "$BASE" --run "$dir" --steps 30000 --quiet \
      --reseed-attack-row "${extra[@]}" "${sets[@]}" > "$dir.log" 2>&1
    runit touch "$dir/done"
  fi
  ck=$dir/checkpoints/$W_END
  [ -s "$out.json" ] && return 0
  [ -f "$ck" ] || { note "세계 $name-$seed: 체크포인트 없음 $ck"; return 0; }
  runit nice -n 15 "$PY" $J observe "$ck" --steps 5000 --json "$out.json" > "$out.log" 2>&1
  note "세계 $name-$seed 읽음: $(tail -1 "$out.log" 2>/dev/null | cut -c1-160)"
}

# ---- 판정 0: 용량 반응(진화 없이 겨루기만). 칸 정의는 norm_judge.CELLS에 있다
CORE=(base hand-phi0.1 hand-phi0.3 hand-phi1.0 grp-1 grp-2 grp-3 hand3-phi0.3 hand-phi0.3-inert hand-phi0.3-ret0
      hand-phi0.3-ret0.5 hand-phi0.3-L1 off noimp)
REST=(hand-phi0.3-nearest hand-phi0.3-inh0 hand-phi0.3-inh0.4 hand-phi0.3-L10 hand-phi0.3-tau1
      hand-phi0.3-sig-ret0 hand-phi0.3-sig-ret0.1 hand-phi0.3-none-q-ret0 hand-phi0.3-none-q-ret0.1
      hand-phi0.3-noise0.0005 hand-phi0.3-noise0.002)
dose() {  # 칸 마당(std|mix)
  local cell=$1 arena=$2 out=$E/h-$1-$2
  [ -s "$out.json" ] && { note "용량 반응 $cell-$arena 이미 있음"; return 0; }
  note "용량 반응 $cell-$arena 시작"
  runit nice -n 15 "$PY" $J hand-dose "$BASE" --cell "$cell" --arena "$arena" --n 800 --steps 8000 --reps 6 \
    --json "$out.json" > "$out.log" 2>&1
  local rc=$?
  note "용량 반응 $cell-$arena 끝 ($rc): $(tail -1 "$out.log" 2>/dev/null | cut -c1-160)"
}
dose_list() {  # 칸 이름들
  local c
  for c in "$@"; do dose "$c" std; dose "$c" mix; done
}

run_branch() {  # 이름 씨앗(a=같은 흐름, b=--new-stream) 설정...
  local name=$1 seed=$2; shift 2
  local dir=$R/runs/n-$name-$seed
  [ -f "$dir/done" ] && { note "$name-$seed 이미 끝남, 건너뜀"; return 0; }
  [ -f "$dir/run.json" ] && { note "$name-$seed 중간에 끊긴 흔적 — 건너뜀"; return 0; }
  local extra=(); [ "$seed" = b ] && extra=(--new-stream)
  local kv sets=(); for kv in "$@"; do sets+=(--set "$kv"); done   # 반복 변수를 local로(호출자의 s를 덮어쓰지 않게)
  note "$name-$seed 시작 ($*)"
  runit nice -n 15 "$PY" -m kkumteul.run branch --from "$BASE" --run "$dir" --steps 300000 --quiet \
    --reseed-attack-row "${extra[@]}" "${sets[@]}" > "$dir.log" 2>&1
  local rc=$?
  runit touch "$dir/done"
  note "$name-$seed 끝 (종료 코드 $rc): $(tail -1 "$dir.log" 2>/dev/null | cut -c1-160)"
}

judge() {  # 갈래-씨앗 무엇(block|bottom|obs|kin|knock-norm|knock-vis)
  local tag=$1 what=$2
  local ck=$R/runs/n-$tag/checkpoints/$END out=$E/j-$what-$tag
  [ -s "$out.json" ] && { note "판정 $what-$tag 이미 있음"; return 0; }
  [ -f "$ck" ] || { note "판정 $what-$tag: 체크포인트 없음 $ck"; return 0; }
  note "판정 $what-$tag 시작"
  case $what in
    block)      runit nice -n 15 "$PY" $J block-compete "$ck" --json "$out.json" > "$out.log" 2>&1 ;;
    bottom)     runit nice -n 15 "$PY" $J bottom "$ck" --json "$out.json" > "$out.log" 2>&1 ;;
    obs)        runit nice -n 15 "$PY" $J observe "$ck" --steps 5000 --json "$out.json" > "$out.log" 2>&1 ;;
    knock-norm) runit nice -n 15 "$PY" -m kkumteul.knockout split "$ck" --channel norm > "$out.json" 2> "$out.log" ;;
    knock-vis)  runit nice -n 15 "$PY" -m kkumteul.knockout split "$ck" --channel vision > "$out.json" 2> "$out.log" ;;
  esac
  local rc=$?
  note "판정 $what-$tag 끝 ($rc): $(tail -1 "$out.log" 2>/dev/null | cut -c1-160)"
}

pre_check || { note "바탕 확인 실패 — 체인 중단"; exit 1; }

# 1. 세계 점검(−1)
for t in a b; do
  wbranch $t meas  $G norm_measure=true
  wbranch $t inert $G norm_enabled=true norm_mode=inert
  wbranch $t full  $G norm_enabled=true
  wbranch $t noimp $G norm_enabled=true norm_impulse=0
done
runit "$PY" $J arena-check "$BASE" > "$E/arena-check.json" 2> "$E/arena-check.log"
runit "$PY" $J gate-world --dir "$E" --tags a b > "$E/gate-world.json" 2> "$E/gate-world.log"
rc=$?
note "관문 −1 종료 코드 $rc (0 통과, 3 세계 못 섬, 4 딜레마 크기 밖): $(tr -d '\n ' < "$E/gate-world.json" 2>/dev/null | cut -c1-200)"
if [ "$rc" != 0 ] && [ -z "${DRY:-}" ]; then
  note "관문 −1 못 넘음 — 판정 0을 돌리지 않고 닫는다(τ·norm_impulse 조정은 결과 문서를 고쳐 쓰고 정한다)"
  exit 0
fi

# 2. 판정 0: 핵심 칸 먼저, 나머지는 뒤에
dose_list "${CORE[@]}"
runit "$PY" $J gate-hand --dir "$E" > "$E/gate.json" 2> "$E/gate.log"
rc=$?
note "관문(α·β·γ·δ·−2) 종료 코드 $rc (0 통과, 3 가를 수 없음, 4 도구 깨짐): $(tr -d '\n ' < "$E/gate.json" 2>/dev/null | cut -c1-200)"
dose_list "${REST[@]}"
if [ "$rc" != 0 ] && [ -z "${DRY:-}" ]; then
  note "관문 못 넘음 — 진화 갈래를 돌리지 않고 닫는다(가를 수 없음·도구 깨짐은 결과 문서에 그대로 적는다)"
  exit 0
fi

# 3. 진화 갈래: 같은 체크포인트에서 갈라 30만 스텝. 먼저 핵심 넷(켬·불활성·끔·위반 공급 끔), 시간이 남으면 나머지 넷
ON="$G norm_enabled=true norm_pun_return=0.1"
for s in a b; do
  run_branch on    $s $ON
  run_branch inert $s $G norm_enabled=true norm_mode=inert
  run_branch off   $s $G norm_enabled=false
  run_branch noimp $s $G norm_enabled=true norm_impulse=0
done
for s in a b; do
  run_branch altr  $s $G norm_enabled=true norm_pun_return=0           # 순이타(2차 딜레마가 가장 아픈 칸)
  run_branch mix   $s $ON norm_mix_period=200
  run_branch arb   $s $G norm_enabled=true norm_mark_src=sig norm_pun_return=0   # 임의-관습
  run_branch blind $s $ON norm_mem_obs_mode=blind
done
if [ "$SLOPE" = 1 ]; then   # 문턱 위가 1% 아래였을 때만: 기울기 시작판(공격 편향 b0 = −0.3)
  for s in a b; do
    for arm in on inert off noimp; do
      note "기울기 시작판 $arm-$s: run_branch에 --attack-b0 -0.3을 더해 별도 이름(slope-$arm)으로 돈다(구현 때 채운다)"
    done
  done
fi

# 4. 판정: on 갈래는 판정 나(주)·다·라·감각 막기, 나머지는 단서(observe)만
for s in a b; do
  judge on-$s block
  judge on-$s bottom
  judge on-$s knock-norm
  judge on-$s knock-vis
  for arm in inert off noimp altr mix arb blind; do judge $arm-$s obs; done
done
runit "$PY" $J final --dir "$E" --tags a b > "$E/final.json" 2> "$E/final.log"
note "최종 판정 종료 코드 $?: $(tr -d '\n ' < "$E/final.json" 2>/dev/null | cut -c1-200)"
note "체인 끝"
```
(체인은 갈래 폴더 `runs/n-<갈래>-<씨앗>`, 판정 원자료 `runs/norm-exp/h-*.json`·`w-*.json`·`j-*.json`을 쓴다 — 모두 `runs/` 아래라 git에 안 올라간다. 칸·갈래마다 `done`/json 표시가 있어 끊겨도 다시 불러 이어 간다. 기울기 시작판의 실제 명령은 과제 14의 `--attack-b0` 옵션으로 구현 때 채운다 — 초안에서는 자리만 둔다.)

---

## 과제 19. 부모의 가르침 연결과 삼가기 물려받기 (둘째 단계, 조건부)
**시작 조건**: 과제 1~18이 끝나고 첫째 단계 결과 문서가 "벌주기 씨앗이 작동"일 때만(설계 8절 2는 "부모의 가르침 구현 뒤 둘째 단계"). 아니면 이 과제를 건너뛴다(설정 키 `norm_mem_inherit`는 결과가 0이면 접는다 — 설계 약점 9).
**바꾸는 파일**: `kkumteul/config.py`(과제 1의 가드 제거), `kkumteul/kernels.py`(`_child`의 `mem` 시작값, `P_NORM_MEM_INHERIT`), `kkumteul/stages.py`(`watch_pun`의 부모 boost, `attack`의 집행자 기록), `tests/test_norm.py`.

**시험 먼저**
- `test_mem_inherit_natural_births_only`: `norm_mem_inherit = r`이면 **자연 출생** 새끼의 `mem`이 `r ×` 부모 `mem`(맛 기억 물려받기 `taste_inherit`와 같은 꼴, `_child`), 재주입·겨루기 새끼는 `_reset_body`의 0 그대로. 기본 0이면 해시가 그대로. `norm_enabled`를 요구. 과제 1의 가드 시험(`test_norm_mem_inherit_guard`)을 지운다. **맛 기억 물려받기가 +0.03에 그쳤으므로 효과가 작을 것을 예측하고, 결과가 0이면 키를 접는다**(결과 문서에).
- `test_parent_shown_pun_boost`: 미숙기 새끼(`age ≤ juv_steps`)가 본 벌 가운데 집행자 또는 벌 받은 쪽이 **부모**인 것에만 관찰 속도가 `clamp(norm_mem_obs + teach_gain × clamp(show_g[부모], 0, 1), 0, 1)`이 된다(`kin` 모드와 같은 구조, 시연자 = 부모. **설계는 "키운다"고만 적어 가르침의 boost 식 `soc_rate + soc_g + teach_gain × show_g`와 같은 합 꼴로 이 계획이 정한다**). 다른 미숙기·어른 관찰자·부모 아닌 시연자는 안 키운다.
- `test_parent_shown_pun_modes`: `costonly`·`drift` 눈가림도 같은 이름으로 둔다: `costonly`·`drift`는 boost 0(비용은 가르침의 `teach_pass`/`watch_meals`가 이미 처리, 이 과제는 비용을 새로 걷지 않는다), `teach_enabled = false`면 `watch_pun`이 과제 12와 같다(회귀).
- `test_pun_by_recorded_unsaved`: 집행자를 알려면 `attack`의 벌 분기가 벌 받은 개체의 `pun_by`(집행자 칸·id, 저장 안 함, 기본 −1)를 남긴다(설계에 이 필드가 없어 이 계획이 더한다. `_reset_body`·`inject_agent`·읽은 직후는 −1 = 모름).
- `test_mix_and_teach_still_incompatible`(과제 1 검사의 재확인).
- `test_post_graduation_take_rate_by_parent_nearness`: 졸업 직후 첫 `juv_steps`스텝의 노출·가로채기를 부모 곁 비율(`par_near_steps ÷ par_alive_steps`) ≥ 0.5 / < 0.5로 나눠 `norm_tab`의 네 칸(저장 안 함)에 센다 — "부모 곁에서 자란 새끼가 다 자란 뒤 조건부 삼감을 더 일찍 보이는가"를 읽는 값(태그별).
- 빨강 확인.

**구현**: `P_NORM_MEM_INHERIT`를 `P_*`의 맨 뒤(번호는 과제 0에서 읽은 마지막 뒤)에 덧붙이고 `pack_params`가 채운다. `_child`의 자연 출생 가지에서 `s.mem[j] = P × s.mem[i]`. `watch_pun`의 부모 판정은 가르침의 `_kin_class(s, i, j) == 1`(j가 i의 부모)를 재사용한다. 새 난수는 없다.

**초록 확인**: `python3 -m pytest -q tests/test_norm.py tests/test_teach.py tests/test_soc_taste.py tests/test_stages.py`. 커밋: "규범의 씨앗 19: 삼가기 물려받기와 부모가 보여 준 벌(둘째 단계)".

---

## 설계의 "이 판의 약점"을 어느 과제에서 막는가
약점은 대부분 **측정으로만 확인되는 한계**라, 코드로 막을 수 있는 것은 시험으로, 나머지는 결과 문서 문구·체인 칸으로 막는다.
| 약점(설계 번호) | 코드·시험으로 막는 과제 | 문서·체인 칸 |
|---|---|---|
| 1 위반은 세계가 공급, 뇌가 못 말림 | 4(`test_impulse_does_not_touch_brain_output`) | 18의 예측 절 첫 줄(한계), 17의 출력 라벨 시험 |
| 2 벌의 물리·위축은 부여, 뇌 삼감 못 잼 | 7(`test_inhibit_zero_...`) | 17(구간표 "조작 확인용" 라벨), 체인 `hand-phi0.3-inh0` 칸(확인용) |
| 3 `θ_t` 미측정 | 3(`EXPO` 계수·summary), 17(`observe`·관문 −1b) | 체인 세계 점검, 예측 표 0.2~0.5 |
| 4 표시가 정직·즉시·전원 공개(평판 과대) | 5(표시는 가로챈 개체에게만, 시각 방식) | 체인 `tau1` 칸, 17(평판 = 늦은 집행 비율) |
| 5 친족만으로 협력이 서 있을 수 있음, 태그 상속 혼입 | 8(닮음 구간 + 계보 병행), 16(섞기) | 17(섞은 마당에서 `N_ind` 다시 읽기), 체인 표준·섞은 마당 이중 칸 |
| 6 손 집행자 배선이 임의 | 15(양성 확인·`NORM_VIS` 막으면 못 함·θ 규칙) | 18(결과 문서에 어긋남 그대로 적기) |
| 7 효과 크기·검출 한계·진화의 공격 문턱 | 14(`reseed_attack_row`·`--attack-b0`) | 17(관문 −2·−2b), 체인 `threshold`·기울기 시작판 자리 |
| 8 임의 대 기능은 이득 구조 확인 | 5(`sig` 원천), 9(표시당 집행 확률) | 17(관문 δ의 `ret0` 규칙·라벨), 체인 `sig`·`none-q` 칸 |
| 9 설정 키 21개(줄일 후보) | 1(키 목록 고정 시험·주석) | 19(`norm_mem_inherit` 결과가 0이면 접음) |
| 10 상습 위반자 모형(L)이 첫걸음 기울기를 정함 | 4(L은 가정이라는 주석) | 체인 `L1`·`L10` 칸, 17(관문 γ 규칙), 18의 예측 절 |
| 11 섬·5단계는 방향만 | 2(`norm_tab`의 섬 자리는 주석만) | 18(이번 판 밖 한 줄) |
| 12 친족 분류가 거침 | 8(`test_no_kin_theta_key`·구간+계보) | 17(같은 쪽일 때만 쓰는 규칙, 닮음 분포 `observe`) |
| 13 충동은 표시를 모름 | 4(`test_impulse_forces_attack_and_hits_nearest`), 13(막기는 충동에 안 닿음) | 18(결과 문서에 모형 단순화로 적기) |

## 설계 "검토 판단표" 26건을 어느 과제가 지키는가
| 판단표 # | 요지 | 지키는 과제·시험 |
|---|---|---|
| 1 | B1 거주민이 "공격 없음"이 아님(편향 +7.27) | 14(`test_resident_attack_row_fixed...`: 합성 +7 체크포인트), 17(−1e `arena-check`) |
| 2 | 사회 학습 격자 재구성·`watch_pun` 낡은 격자 | 12(`test_grid_rebuild...`·`test_attack_neighbor_set_unchanged...`), 18(`soc_*` 갈래 동일) |
| 3 | `knockout`이 감각 함수 셋만 감쌈 | 11(넷째 `norm_sense`) |
| 4 | float32 `mark` 감쇠 잔여 | 5(`mark_t` 시각 방식, τ = 5·49·50 시험), 7(`mem` 0 자르기) |
| 5·22 | 노출 `touch`가 공격 거리와 안 맞음·분모 | 3(`test_expo_counts_gap_neighbors...`), 8(`OPP_*` 같은 훑기) |
| 6 | 병렬 커널 데이터 경쟁 | 10(읽기만·스레드 1/4 결정성), 공통 규칙 |
| 7 | `_see` 일반화 함정 | 10(`_see_norm`·`test_norm_vis_...`) |
| 8 | 자는 개체의 충동·NORM_VIS·`fs`/`isp` | 2(칸), 4(자는 개체), 10(`I_NORM_VIS0`) |
| 9 | 런타임 끄기 자기모순 | 2(`test_set_config_off_clears_norm_state`) |
| 10 | 의존 검사 문구·`OFF_RULES` | 1(기본값이 아닐 때만), 11(의존 키를 함께 되돌림) |
| 11 | 계수기 표식 충돌·태그 접기·벌 사망·섞기 난수 | 2(−2 미지), 3(접기), 6(`PUN_KILL`), 8(칸+id), 16(`3 × id + k`) |
| 12 | `norm_mix`가 가르침·들기와 충돌 | 1(검사), 16(`test_mix_incompatible...`) |
| 13 | 장부·해시·rng 확인 | 모든 과제의 공통 규칙, 6(장부 시험 + 벌금 줄 틀리게 해 실패 확인), 18(전체 시험·박은 해시) |
| 14 | `PUN_NOVICT ÷ PUN` 항등식 | 8(`PUN_NOVICT`·`PUN_UNKNOWN`은 분모 제외), 9(표시당 집행 확률), 17(δ) |
| 15 | 2자 반격·3자·`N_ind3`·`N_grp3`·역할별 | 8(역할별 표), 13(3자 막기), 17(`N_ind3`·`N_grp3`·β 가·나·다) |
| 16 | 선택성 지표·비자발 벌 | 4(충동은 가로채기), 8(`ATK_VOL_*`), 17(선택성) |
| 17 | 혈연 분리·태그 상속 혼입 | 8(구간 + 계보), 16(섞기), 17(섞은 마당 `N_ind`) |
| 18 | `inert` 정의·`norm_impulse`가 `norm_enabled` 안 요구 | 1(검사), 6(`inert` 해시 시험), 17(`off` 갈래 = 충동은 켠 채) |
| 19 | L 가정·진화 쪽 공격 문턱 | 4(L 주석), 14(`reseed_attack_row`), 17(−2b·γ), 체인 `L1`·`L10` |
| 20 | ④가 엔진 위축만으로 통과 | 7(`norm_inhibit=0`은 기억만), 17(라벨), 체인 `inh0` 칸 |
| 21 | 임의 대 기능은 보수 구조로 정해짐 | 5(`sig` 원천), 17(δ 한계 문구) |
| 23 | 표시 유병률 예측·B2 불가 | 9(유병률 계수), 17(−1c τ 조정 권고), 18(예측 5~13%) |
| 24 | 수명 1,000 대 1,560·Welch √2·구간 이중 계산 | 14(한 생 환산 × 1,560), 17(검출 한계 √n·Welch), 18(예측 표) |
| 25 | 사용자가 원한 소통이 없음 | 18(한계 문구 첫 줄) |
| 26 | `vision_sectors` 기본 8 등 확인 | 0(바탕 설정 확인), 1(`vision_sectors == 8` 검사) |

## 이 계획이 설계와 코드를 맞추며 정한 것(검토 때 먼저 볼 곳)
1. 설정 키는 설계 표 20줄 + `norm_mix_frac` = **21개**(과제 1). 난수 용도는 36~38에 더해 **`S_NORM_BLIND = 39`**가 필요하다(과제 2). `blind`의 "같은 확률"은 (본 벌 쌍 수 ÷ 산 개체 수)로 정했다(과제 12).
2. 표시 에피소드는 만료 때만 세면 표시된 채 죽는 개체(벌에 맞아 죽는 개체)가 빠져 표시당 집행 확률이 치우친다 → **`process_deaths`에서 닫는다**(과제 9, `Q_NORM`·`Q_NORM_TAU`).
3. 설계는 `bench.py`·`pack.py`를 "바꾸는 파일"에 적었으나 `bench`의 `kernel_breakdown`은 `stages.attack`을 안 지나고 새 유전체 배열이 없어 `pack.py`도 안 바뀐다 → 둘 다 안 고친다(과제 3·2).
4. `N_ind`("생애 에너지 차")를 **개체 스텝당 순수입의 태그 차**로 정의하고 죽는 개체의 `eaten`을 `process_deaths` 훅으로 모은다(과제 9·14). 설계의 한 생 값은 × 1,560으로 환산해 견준다.
5. 구현 순서를 바꿨다: 섞기(설계 ⑥)를 **판정 도구 앞**(과제 16)으로. 섞은 마당 칸이 도구에 들어가기 때문이다. `watch_pun`의 부모 boost용 집행자 기록(`pun_by`)과 boost 식은 과제 19에서 정했다(설계가 안 적음). 충동은 `norm_block_tag`에 안 막힌다(과제 13).
6. 설계의 `sig`·`none-q` "각각 `ret0`·`ret0.1`"은 두 값 모두 도는 네 칸으로 읽었다(과제 17). `norm_hand`의 회전 부호·θ·G는 양성 시험으로 구현 때 정한다.
7. 호출 위치는 "`st.attack` 바로 뒤"처럼 상대 위치로만 적었다. 가르침 구현이 `world.step`에 끼운 자리와 어긋나면 과제 0에서 고친다.

## 과제 요약
0. 전제 확인(사회 학습 1~7·가르침 1~8 커밋·초록, 번호 읽기, 호출 순서 읽기, B1 설정)
1. 설정 21개·검사·상수·preset(`norm_size` = 12칸 크기, 동작 변화 없음)
2. 난수 36~39·저장 필드(`mark_t`·`pun_t`·`mem`)·계수 배열·`F_`/`I_` 칸·옛 호환·`set_config` 끄기 지우기
3. 훑기(`_scan_touching`)와 `norm_measure` 계수기(세계 동작 불변)
4. 떨리는 손(`norm_impulse`: 충동 구간·강제·가까운 이웃·가로채기)
5. 표시(`mark_t` 시각 방식·밝기·`take`/`sig`/`none` + 잡음)
6. 조준·벌 분기·벌금(`got = dmg × norm_pun_return`)·집행자 비표시·`inert`·장부
7. 삼가기 기억 `mem`·위축·`norm_tick`
8. 계수기 ①: 집행 역할(2자·3자 친족·3자 남)·닮음 구간 + 계보·기회·선택성 분모·에너지 표
9. 계수기 ②: 구간표·표시 에피소드(죽을 때 닫기)·유병률·`Q_NORM`
10. 감각 블록 12칸(`_see_norm`·병렬 읽기 전용·자는 개체)
11. `knockout`에 `norm_sense` 넷째 감각 함수와 `norm` 채널·끄기 규칙
12. `watch_pun`·눈가림·`attack` 뒤 격자 재구성(불활성 해시 재확인)
13. 벌주기 막기(`norm_block_tag`·`norm_block3_tag`)
14. 겨루기 마당 준비(크기 올림·거주민 공격 행 고정)·다태그 겨루기·순수입·`reseed_attack_row`
15. 손 집행자(`norm_hand.py`)와 양성 확인(`NORM_VIS` 막으면 못 함)
16. 섞기(`norm_mix`)
17. 판정 도구(`norm_gate.py`·`norm_judge.py`: 관문 −1·−2·α·β·γ·δ·판정 나·다·라)
18. 체인 스크립트 초안·예측 기록(실행 전)·전체 시험·직접 실행
19. 삼가기 물려받기와 부모가 보여 준 벌(둘째 단계, 조건부)

