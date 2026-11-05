# 새 장치 독립 검토 (2026-11-05 새벽, 서버 세션)

대상: 공격(`stages.py`의 `attack`), 먹이 종류와 독(`kernels.py`), 이주(`pack.migrate`), 비교 도구(`compare.py`). 범위는 `git diff f9d50c0...1ef202f`.
맥락 없는 검토자 셋(규칙·명세·결함, 서로 모르게)이 지적을 올리고, 다른 모델의 검증자가 지적마다 코드를 열어 반박을 시도했다. 검토자·검증자는 코드를 실행하지 않았다.

**처리 상태: 아직 반영 전이다.** 지적마다 코드로 다시 확인해 "반영 / 기각+근거"를 이 파일에 적는다. 축끼리 섞거나 순위를 다시 매기지 않는다.

## 규칙 (6건)

### 규칙-1. 먹이 종류 비율 0.5가 커널 안에 박혀 있고 config.py에 없다
- 종류: 위반 · 자리: `kkumteul/kernels.py:327` · 검증: **확인됨**
- 지적: 종류 0과 1이 반반으로 나는 것은 세계 규칙에 속하는 수인데 두 군데(kernels.py:327, stages.py:606)에 리터럴로만 있다. 새 먹이 장치의 설정값은 kind_flip_period, poison만 config.py에 들어갔다. 종류 비율을 바꿔 보려는 참여자는 커널 두 곳을 고쳐야 하고, 한 곳만 고치면 일반 먹이와 거름 먹이의 비율이 어긋난다. CONTRIBUTING이 "설정값은 config.py" 전제로 참여를 안내하는 점에서도 걸린다. 공격 사거리의 `+ 0.5`(stages.py:497)와 맞은 쪽 touch 값 2.0(stages.py:542)도 같은 부류의 리터럴이지만, 사거리는 기존 give(stages.py:443)를 그대로 따른 것이라 위반 목록에서는 뺐다.
- 인용: 레포 CLAUDE.md 표: "`kkumteul/` | 코드. 설정값은 전부 `config.py`" / 코드: `if ip[Q_KINDS] != 0 and u01(seed, S_KIND, serial, 0) < 0.5:` (kernels.py:327), 같은 모양이 stages.py:606 `u01(isp[I_SEED], S_KIND, step, 1000003 + k) < 0.5`
- 검증 근거: kernels.py:327 `if ip[Q_KINDS] != 0 and u01(seed, S_KIND, serial, 0) < 0.5:`, stages.py:606 `... u01(isp[I_SEED], S_KIND, step, 1000003 + k) < 0.5:` 둘 다 리터럴이다. config.py에서 종류 관련 필드는 food_kinds_enabled(130), kind_flip_period(131)뿐이고 비율 필드는 없다. kernels.py:215-216 pack에도 Q_KINDS, Q_KIND_PERIOD만 넘긴다. 레포 CLAUDE.md의 '설정값은 전부 config.py'와 어긋난다. 다만 CONTRIBUTING.md에는 '설정값은 config.py'라는 문장이 그대로 있지는 않다(81행 '새 장치는 설정으로 끄고 켤 수 있게'가 가장 가깝다). 그래서 지적의 이 보조 근거는 조금 과장됐지만 핵심 주장은 맞다.
- 고치는 방향(검증자): config에 kind_ratio(기본 0.5)를 넣고 P_ 칸으로 넘긴다. 두 곳은 같은 도우미 함수 하나로 뽑게 한다.
- 처리: (미정)

### 규칙-2. Duplicated Code: attack이 give의 가장 가까운 이웃 찾기를 통째로 복사했다
- 종류: 판단 사항 · 자리: `kkumteul/stages.py:497` · 검증: **확인됨**
- 지적: 약 30줄의 순회(셀 좌표 계산, 3x3 이웃, 환형 거리, 최근접 선택)가 give와 attack에서 글자까지 같고, 달라진 것은 후보 조건 한 줄(attack은 `s.energy[j] <= 0.0` 제외)과 그 뒤의 처리뿐이다. 사거리 공식 `2.0 * F_A_RAD + 0.5`도 두 번 적혀 있어, 사거리를 바꾸면 한쪽만 바뀐다. 이 장치의 목적이 "주기와 공격을 같은 세계에서 비교"이므로 두 장치의 도달 거리가 어긋나면 비교가 흐려진다. 같은 이유로 먹이 종류 뽑기(`kind = 0; if ... < 0.5: kind = 1`)도 kernels.py:326-328과 stages.py:605-607에 두 벌로 있고 난수 b 인자 규약만 다르다(`0` 대 `1000003 + k`).
- 인용: give: `reach = 2.0 * fs[F_A_RAD] + 0.5` ... `for oy in range(-1, 2): ... if d2 < bd: bd = d2; best = j` (stages.py:443-475) / attack: 같은 `reach`, 같은 3x3 셀 순회, 같은 `_wdiff` 거리 비교 (stages.py:497-530)
- 검증 근거: stages.py:443 give `reach = 2.0 * fs[F_A_RAD] + 0.5`, stages.py:497 attack에도 같은 줄이 있다. 셀 좌표 계산, 3x3 순회, `_wdiff` 거리, `d2 < bd` 최근접 선택이 give 449-475와 attack 503-530에서 글자까지 같다. 다른 점은 후보 조건 `or s.energy[j] <= 0.0`과 그 뒤 처리뿐이다. 먹이 종류 뽑기도 kernels.py:326-328과 stages.py:605-607 두 벌이고 b 인자가 각각 `0`, `1000003 + k`다.
- 고치는 방향(검증자): @njit 도우미 `_nearest_neighbor(s, fs, isp, i, need_energy)`와 `_reach(fs)`로 뽑아낸다. 먹이 종류 뽑기도 도우미 하나로 묶는다. 단, 난수 호출 순서와 인자는 그대로 둔다.
- 처리: (미정)

### 규칙-3. Primitive Obsession: 먹이 종류가 이름 없는 정수 0/1/2이고 부호와 사체 값이 리터럴로 흩어져 있다
- 종류: 판단 사항 · 자리: `kkumteul/kernels.py:707` · 검증: **확인됨**
- 지적: "사체는 늘 먹을 수 있다"는 개념이 숫자 2 하나로만 표현되어 eat, _remove, 시험(`f_kind != 2`)에 각각 따로 박혀 있다. 이름이 있는 상수가 없어 2를 읽는 사람이 FOOD_FIELDS의 주석("2는 늘 먹을 수 있는 것(사체)")까지 가야 뜻을 안다. 종류 1이 시야에서 음수가 된다는 약속(부호 sgn)과 touch 값 2.0(맞음)도 주석과 docstring에만 있다. `1000003`은 일반 먹이 난수 번호와 겹치지 않으려는 오프셋 같지만 이유가 적혀 있지 않다.
- 인용: `if kinds and s.f_kind[j] != 2 and s.f_kind[j] != good:` (kernels.py:707) / `_place_food(s, s.x[i], s.y[i], e, 2)` (kernels.py:930) / `np.float32(-1.0) if f_kind[j] == 1 else np.float32(1.0)` (kernels.py:462) / `kind = 1` (kernels.py:328) / `s.touch[best] = 2.0` (stages.py:542) / `1000003 + k` (stages.py:606)
- 검증 근거: kernels.py:707 `s.f_kind[j] != 2 and s.f_kind[j] != good`, kernels.py:930 `_place_food(s, s.x[i], s.y[i], e, 2)`, kernels.py:462 `np.float32(-1.0) if f_kind[j] == 1 else np.float32(1.0)`, kernels.py:328 `kind = 1`, stages.py:542 `s.touch[best] = 2.0`, stages.py:606 `1000003 + k`를 모두 확인했다. 이름 있는 상수(KIND_CORPSE 같은 것)는 없고, 뜻은 FOOD_FIELDS 주석(kernels.py:79 '2는 늘 먹을 수 있는 것(사체)')과 attack docstring에만 적혀 있다. 1000003의 까닭을 적은 주석도 없다.
- 고치는 방향(검증자): 모듈 상수 KIND_A=0, KIND_B=1, KIND_CORPSE=2, TOUCH_HIT=2.0, KIND_RNG_OFFSET=1000003(까닭 주석 포함)을 두고 바꿔 쓴다. numba는 전역 정수 상수를 그대로 쓸 수 있다.
- 처리: (미정)

### 규칙-4. Primitive Obsession: compare가 겨루기 판정을 한국어 문장 접두어로 해석한다
- 종류: 판단 사항 · 자리: `kkumteul/compare.py:88` · 검증: **확인됨**
- 지적: 판정이 열거형이나 필드가 아니라 문장이고, compare는 그 문장의 "A가"/"B가"를 치환한 뒤 접두어로 다시 읽는다. measure.verdict의 문구가 바뀌면 `_reading`이 오류 없이 마지막 줄 "가를 수 없음."으로 떨어진다. 현재 코드에서도 분기가 다 차지 않는다. 켠 규칙은 "가를 수 없음"이고 끈 규칙만 "켠 쪽이 이김"이면 어느 `if`에도 걸리지 않아 "가를 수 없음."이 나온다(`off_win`은 `on_win`과 함께일 때만 쓰인다).
- 인용: `return v.replace("A가", a_label).replace("B가", b_label)` (compare.py:81) / `on_win = on_v.startswith("켠 쪽이 이김")` (compare.py:88-91) / measure.py의 `return "B가 이김" + warn`
- 검증 근거: compare.py:81 `return v.replace("A가", a_label).replace("B가", b_label)`, compare.py:88-91 `startswith("켠 쪽이 이김")` 등. measure.py:210-228 verdict는 문자열('B가 이김', 'A가 이김', '가를 수 없음'+warn)만 돌려준다. 지적한 분기 누락도 실제로 있다. on_win=False, on_lose=False, off_win=True이면 `on_win and ...` 두 개, `on_lose and off_lose`, `on_lose`, `off_lose and not on_lose`(off_lose=False) 어디에도 걸리지 않는다. 그래서 '가를 수 없다.'가 나온다(compare.py:104). '끈 규칙에서만 켠 쪽이 이김' 경우가 해석 없이 빠진다.
- 고치는 방향(검증자): measure.verdict가 winner('A'|'B'|None) 필드를 따로 돌려주게 하고 compare는 그 필드로 가른다. `off_win and not on_win` 분기를 더한다.
- 처리: (미정)

### 규칙-5. Shotgun Surgery: 새 장치를 하나 넣으면 흩어진 목록을 함께 고쳐야 하고, 이번에 compare의 목록이 빠졌다
- 종류: 판단 사항 · 자리: `kkumteul/compare.py:25` · 검증: **확인됨**
- 지적: conflict_enabled와 food_kinds_enabled를 켠 실행과 끈 실행을 compare로 견주면 flags가 모두 [False, False]로 찍혀 무엇이 달라진 비교인지 카드에 드러나지 않는다. TRAITS에도 attacks, poisoned, migrant_frac이 없다. CLAUDE.md의 "단계마다 무엇이 효과를 냈는지 가를 수 있어야 한다"를 받치는 도구인데, 장치를 추가할 때 고칠 곳(config 필드·검증·preset, kernels의 P_/Q_ 칸 번호, stages의 F_/I_ 칸 번호, 두 군데 pack 함수, world.summary, compare의 두 목록)이 손으로 맞추는 목록이라 빠지기 쉽다. 이번에 그 하나가 빠졌다. 카운터 번호도 두 파일에 나뉘어 있어(14·15·16은 stages.py, 17은 kernels.py) 다음에 넣는 사람이 두 파일을 다 봐야 하고, N_COUNTERS를 16에서 24로 늘려 쓰지 않는 칸이 6개 남았다.
- 인용: `STAGE_FLAGS = ("mind_enabled", "social_enabled", "niche_enabled")` (compare.py:25) / `"flags": {f: [getattr(a["cfg"], f), getattr(b["cfg"], f)] for f in STAGE_FLAGS}` (compare.py:59) / `C_ATTACKS = 16` (stages.py:23), `C_POISONED = 17` (kernels.py:108), `N_COUNTERS = 24` (kernels.py:109) / `ip[Q_KINDS]`(kernels.py)와 `isp[I_KINDS]`(stages.py:62)
- 검증 근거: compare.py:25 STAGE_FLAGS = (mind_enabled, social_enabled, niche_enabled)에 config.py:130 food_kinds_enabled와 135 conflict_enabled가 없다. 그래서 compare.py:59 flags에 두 장치가 드러나지 않는다. compare.py:23 TRAITS에도 world.py:279/281/284가 내는 attacks, poisoned, migrant_frac이 없다. 카운터 번호는 kernels.py:94-108(0-13, 17)과 stages.py:21-23(14-16)에 나뉘어 있다. kernels.py:109 N_COUNTERS=24이고 18-23은 아무 데서도 쓰지 않는다(6칸).
- 고치는 방향(검증자): STAGE_FLAGS를 config의 *_enabled 필드에서 자동으로 모으거나 두 장치를 더한다. TRAITS에 attacks, poisoned, migrant_frac을 넣는다. 카운터 번호는 한 파일에 모은다.
- 처리: (미정)

### 규칙-6. Feature Envy: pack.migrate가 measure의 비공개 함수와 내부 사이드 모양을 직접 만진다
- 종류: 판단 사항 · 자리: `kkumteul/pack.py:165` · 검증: **확인됨**
- 지적: 이주 로직이 겨루기 모듈 쪽의 `_inject`(밑줄로 비공개 표시)를 불러 개체를 세계에 넣고, check_same_shape를 통과시키려고 `load_side`가 만드는 dict 모양을 손으로 흉내 낸다. measure의 사이드 모양이나 `_inject` 인자가 바뀌면 pack.migrate가 깨지는데 둘 사이에 계약이 없다. 같은 파일에서 `extract_genomes`를 이름으로도 가져오고 `measure.extract_genomes`로도 부르는 혼용도 있다. 이주 시험(test_migrate_injects_tagged_agents_into_a_branch)은 있지만 그 결합을 지키는 것은 아니다.
- 인용: `measure.check_same_shape(side, {"genomes": measure.extract_genomes(world), "path": str(src), "cfg": world.cfg})` (pack.py:153) / `measure._inject(world, g, int(row), tag, float(rng.random() * cfg.world_w), ...)` (pack.py:165) / `from . import measure` (pack.py:29)와 `from .measure import GENOME, extract_genomes` (pack.py:30)
- 검증 근거: pack.py:165 `measure._inject(...)`는 measure.py:91의 밑줄 붙은 비공개 함수를 부른다. pack.py:153은 measure.check_same_shape(measure.py:68)에 넘기려고 {'genomes','path','cfg'} dict를 손으로 만든다(measure.load_side가 만드는 모양을 흉내 낸 것). pack.py:30은 `from .measure import GENOME, extract_genomes`로 가져와 59행에서 이름으로 부르고, 153행에서는 `measure.extract_genomes`로 불러 둘이 섞여 있다. 참고로 pack.py:149의 load_side는 pack.py:121에 있는 pack 자신의 함수다.
- 고치는 방향(검증자): measure에 공개 함수 inject_agent()와 side_from_world(world)를 두고 pack은 그것만 쓴다. extract_genomes 부르는 방식은 하나로 맞춘다.
- 처리: (미정)

## 명세 (6건)

### 명세-1. 독립 검토 확인 문구가 검토받지 않은 새 장치(공격·먹이 종류)까지 덮는다
- 종류: 설명과 다름 · 자리: `docs/stages-2-4.md:56` · 검증: **확인됨**
- 지적: 이 문장은 공격(attack)과 먹이 종류·독 절이 추가된 같은 문서의 '확인된 것과 아닌 것' 아래에 범위 한정 없이 놓여 있다. 그런데 검토 기록 code-review-2026-11-05.md는 공격·먹이 종류·이주·비교 도구가 들어오기 전 커밋(412edb6)에서 만들어졌고, 이 파일에서 공격이나 독을 다룬 줄은 찾지 못했다(grep에는 '독립'만 걸림). sessions/tasks.md도 '새로 넣은 것(공격, 먹이 종류, 이주, 비교 도구) 독립 검토'를 아직 할 일로 남겨 두었다. 따라서 독립 검토로 확인됐다는 말은 새 장치에는 사실이 아니다. 새 장치에 대해 확인된 것은 시험뿐이고, 인수인계 문서도 '돌아가고 에너지가 새지 않는다는 것만 확인했다'고 적었다.
- 인용: - 확인(독립 검토): 에너지가 새거나 생기는 길이 없다. 검토 지적과 처리는 `wayfinder/stage2-4/code-review-2026-11-05.md`.
- 검증 근거: docs/stages-2-4.md:56에 범위를 한정하지 않은 '확인(독립 검토): 에너지가 새거나 생기는 길이 없다'가 있다. 공격 절은 :40~44, 먹이 종류 절은 :31~32에 있다. git log를 보면 검토 기록 code-review-2026-11-05.md는 412edb6에서 들어왔고, 그 뒤 커밋에서 compare(1624237), 공격(866bbed), 이주(2397836), 먹이 종류(24f9764)가 들어왔다. 검토 파일에서 공격·독·kind·attack·poison을 grep하면 1행 제목의 '독립'만 걸린다. sessions/tasks.md:18은 '새로 넣은 것(공격, 먹이 종류, 이주, 비교 도구) 독립 검토'를 아직 안 한 일([ ])로 두었다. 같은 문서 :44도 공격은 '돌려 본 것은 시험뿐이다'라고 적었다.
- 고치는 방향(검증자): :56을 '확인(독립 검토, 412edb6 시점의 2~4단계 장치만)'으로 좁히고, 공격·먹이 종류·이주·compare는 시험으로만 확인했다고 적는다.
- 처리: (미정)

### 명세-2. 사체 먹이(종류 2)는 독이 아니고 양수로 보이는데, 문서는 이 예외를 말하지 않는다
- 종류: 설명과 다름 · 자리: `kkumteul/kernels.py:707` · 검증: **확인됨**
- 지적: 문서와 config 주석은 먹이가 두 종류이고 한쪽이 독이라고만 쓴다. 코드는 세 번째 종류를 둔다. 죽은 개체가 남기는 먹이(_remove가 _place_food(..., 2)로 놓는 것)는 종류 2이고, 어느 쪽이 독이든 늘 안전하다. 시야 부호는 sgn이 -1인 종류 1만 음수이고 나머지는 양수라서, 사체는 종류 0과 똑같이 양수로 보인다. corpse_to_food 기본값이 True이므로 기본 설정에서 사체 먹이가 흔하다. 종류 0이 독인 구간에 양수로 보이는 사체는 안전하다. 그래서 '양수=독'이라는 규칙이 문서가 설명한 것처럼 깨끗하게 성립하지 않는다. 학습이 값을 하는지 재는 실험(tasks.md의 kinds 실험)을 읽을 때 반드시 알아야 할 동작인데 docs/stages-2-4.md에는 적혀 있지 않다.
- 인용: if kinds and s.f_kind[j] != 2 and s.f_kind[j] != good:  /  docs: "먹이가 두 종류로 나고(시야에서 한 종류는 양수, 다른 종류는 음수로 보인다) 한쪽은 독이다."
- 검증 근거: kernels.py:707의 조건 `s.f_kind[j] != 2 and s.f_kind[j] != good` 때문에 종류 2는 독이 되지 않는다. kernels.py:79 주석도 '2는 늘 먹을 수 있는 것(사체)'이라고 적었다. kernels.py:930의 _remove는 _place_food(..., 2)로 사체를 종류 2로 놓는다. kernels.py:463에서 시야 부호는 f_kind==1일 때만 -1이고 종류 2는 종류 0처럼 +1로 보인다. docs/stages-2-4.md:31~32는 '두 종류, 한쪽은 독'이라고만 썼고 '사체'·'종류 2'는 문서에 없다. 다만 사체는 늙어 죽을 때만 생긴다(kernels.py:966의 DEATH_AGE 경로, config.py:63의 '늙어 죽으면'). 그래서 '흔하다'는 정도는 실행해 봐야 안다. 종류 2 먹이가 차지하는 비율을 세면 가려진다.
- 고치는 방향(검증자): 문서의 먹이 종류 절에 '늙어 죽은 개체가 남긴 먹이는 세 번째 종류로, 늘 안전하고 시야에서는 양수로 보인다'를 덧붙인다. 필요하면 사체에 따로 부호를 주는 방안을 검토한다.
- 처리: (미정)

### 명세-3. 독을 먹어도 '먹었음(ate)' 감각이 1이 되어, 보상 항과 이웃이 먹음 입력에 독 섭취가 먹기로 잡힌다
- 종류: 넘침 · 자리: `kkumteul/kernels.py:712` · 검증: **확인됨**
- 지적: 문서에는 '독을 먹으면 에너지 1을 잃는다'만 있고 ate 신호를 어떻게 하는지는 적혀 있지 않다. 코드는 독을 먹은 개체의 ate를 1로 둔다. stages.py의 보상은 r = rw[i,0]*ate[i] + rw[i,1]*des + ... 이고 rw[i,4]*horm[i,0]*ate[i]도 있으므로, 독 섭취도 'ate' 보상 항을 켠다. 이웃이 먹음 입력(IN_SAWEAT)도 독 섭취를 먹음으로 본다. 독의 해로움은 에너지 변화량 항(des)을 통해서만 보상에 반영된다. 이는 문서가 말한 '살면서 어느 쪽이 먹을 만한지 배우는' 세계의 학습 신호에 영향을 주는 적히지 않은 설계 결정이다. 의도한 것이라면 문서에 한 줄 필요하다.
- 인용: s.counters[C_POISONED] += 1                         s.ate[i] = 1.0                         continue
- 검증 근거: kernels.py:712에서 독 분기 안에 `s.ate[i] = 1.0`을 둔 뒤 continue한다. stages.py:262~263의 보상은 `rw[i,0]*ate[i]`와 `rw[i,4]*horm[i,0]*ate[i]`를 담는다. stages.py:132~133과 :149를 보면 이웃의 ate가 IN_SAWEAT(config.py:316 '가까운 이웃이 방금 먹었는가')로 들어가고, :264에서 보상에도 들어간다. docs/stages-2-4.md:32에는 ate 처리가 적혀 있지 않다. 이렇게 만든 것이 의도인지는 코드만 봐서는 알 수 없다.
- 고치는 방향(검증자): 의도라면 문서에 한 줄을 적는다. 아니라면 독은 ate를 0으로 두거나 따로 '독 먹음' 감각을 만든다. 이 경우 옛 체크포인트와 결과가 달라지는지도 확인한다.
- 처리: (미정)

### 명세-4. compare의 '켠 규칙에서만 이김' 자동 해석이 실제 결과 해석과 어긋난다
- 종류: 설명과 다름 · 자리: `kkumteul/compare.py:91` · 검증: **반박됨**
- 지적: 이 도구가 낸 결과 파일 wayfinder/stage2-4/results/compare-s2.json의 reading에 위 문구가 그대로 들어 있다(켠 규칙 +, 끈 규칙 −인 판). 같은 실험을 사람이 정리한 wayfinder/stage2-4/results.md는 '켠 세계의 생물은 켠 규칙에 적응했다. 그러나 그것은 장치를 "쓰게 된" 것이 아니다'라고 결론짓고, 끈 쪽이 켠 규칙에서 전멸한 까닭으로 '호르몬 입력에 뇌가 흔들린 것'을 든다. 도구가 '값을 한다'고 읽어 주는 패턴이 바로 이 문서가 장치 사용의 증거가 아니라고 한 패턴이다. 프로젝트 규칙('겨루기와 대조 실행으로 판정한다')과 docs/stages-2-4.md의 '결과를 읽을 때'도 같은 경계심을 요구하는데, reading 문구는 그 단서 없이 단정한다. 자동 해석 문구가 설명한 용도(장치를 썼다는 증거)와 실제로 가르는 것(그 규칙에 적응했는가)이 다르다.
- 인용: if on_win and not off_win:         return "켠 규칙에서만 켠 쪽이 이겼다. 장치와 함께 진화한 것이 장치가 있을 때 값을 한다."  /  docstring: "켠 규칙에서만 이기면 '장치를 쓰게 됐다'는 쪽의 증거이고"
- 검증 근거: 지금 코드(HEAD, compare.py의 마지막 수정 커밋 b5efaed)의 compare.py:91~93은 '켠 규칙에서만 켠 쪽이 이겼다. 켠 규칙에 적응했다는 뜻이다. 장치를 '쓰게 됐다'는 뜻은 아니다(...)'이다. docstring(:8~10)도 '그것만으로 장치를 쓰게 됐다고 할 수는 없다'로 이미 고쳐져 있다. 검토자가 인용한 '장치와 함께 진화한 것이 장치가 있을 때 값을 한다'는 코드에 없다. 그 문구는 옛 코드로 만든 결과 파일 wayfinder/stage2-4/results/compare-s2.json:145의 reading에만 남아 있다. 그러니 도구의 해석이 어긋난다는 지적은 지금 코드에는 맞지 않는다. 결과 파일 한 곳에 옛 문구가 남은 것만 사실이다.
- 고치는 방향(검증자): compare-s2.json의 reading이 옛 문구라는 주석을 달거나, 지금 _reading 문구로 다시 만든다.
- 처리: (미정)

### 명세-5. compare의 장치 표시(flags)에 새로 생긴 장치 둘(공격, 먹이 종류)이 없다
- 종류: 빠짐 · 자리: `kkumteul/compare.py:25` · 검증: **확인됨**
- 지적: README는 compare를 '장치 실험 비교: 같은 체크포인트에서 가른 끈 세계와 켠 세계의 생물을 양쪽 규칙에서 겨루게 한다'고 설명하고, 같은 절에서 덧붙이는 묶음으로 kinds와 conflict를 소개한다. 그런데 결과의 flags와 터미널 출력 '장치(끈 쪽, 켠 쪽)'는 세 단계 플래그만 담는다. 공격을 켠 세계와 끈 세계(예: preset=stage3 대 preset=stage3 preset=conflict)를 견주면 어느 플래그도 달라지지 않아 flags가 [False, False]/[True, True]로 모두 같게 찍힌다. 어떤 장치를 켠 비교였는지가 결과 카드에 남지 않는다. 형질 목록 TRAITS에도 attacks, poisoned가 없다(이쪽은 명세에 직접 적혀 있지 않아 참고).
- 인용: STAGE_FLAGS = ("mind_enabled", "social_enabled", "niche_enabled")
- 검증 근거: compare.py:25는 `STAGE_FLAGS = ("mind_enabled", "social_enabled", "niche_enabled")`이고, :60의 flags는 이 셋만 담는다. config.py:130의 food_kinds_enabled와 공격 플래그(preset conflict)는 빠져 있다. 그래서 공격이나 먹이 종류만 다르게 켠 비교는 flags가 양쪽 똑같이 찍힌다. TRAITS(:23~24)에도 attacks·poisoned가 없다.
- 고치는 방향(검증자): STAGE_FLAGS에 food_kinds_enabled와 공격 플래그를 더하거나, 두 cfg에서 값이 다른 필드를 모두 적는다.
- 처리: (미정)

### 명세-6. compare의 형질 '처음 → 끝'이 --at으로 고른 체크포인트가 아니라 stats 파일의 첫 행과 마지막 행이다
- 종류: 설명과 다름 · 자리: `kkumteul/compare.py:72` · 검증: **확인됨**
- 지적: 겨루기는 --at 스텝의 체크포인트 생물로 한다. 그러나 출력의 형질(처음 → 끝, 참고용)은 stats.jsonl 전체의 맨 앞과 맨 뒤 행이다. 실행이 --at보다 더 멀리 돌아간 경우(예: 체크포인트 110만 스텝 이후에도 계속 돌린 실행) 겨룬 생물의 시점과 보고된 '끝' 형질의 시점이 달라진다. 지금 tasks.md의 사용례는 갈래가 --at 지점에서 끝나므로 어긋나지 않지만, 도구가 --at을 제공하는 이상 설명과 다르게 동작할 수 있다.
- 인용: first, last = rows[0], rows[-1]  /  --at: "견줄 체크포인트의 스텝(없으면 각자 가장 최근)"
- 검증 근거: compare.py:72의 `first, last = rows[0], rows[-1]`은 stats.jsonl 전체의 첫 행과 마지막 행이다. --at(:44~49 _checkpoint_at, :108 help '견줄 체크포인트의 스텝')은 겨룰 생물을 고를 때만 쓰이고, 형질을 뽑는 데는 쓰이지 않는다. 실행이 --at보다 더 멀리 돌았다면 '끝' 형질은 겨룬 시점보다 뒤의 값이 된다.
- 고치는 방향(검증자): rows에서 step이 out['step'] 이하인 행 중 마지막 것을 '끝'으로 쓴다.
- 처리: (미정)

## 결함 (11건)

### 결함-1. 양쪽이 다 전멸한 판을 'A 전멸'로만 적어, 겨루기 판정이 뒤집힐 수 있다
- 종류: 중요 · 자리: `kkumteul/measure.py:163` · 검증: **확인됨**
- 지적: 끝 표본에서 A가 0이면 B도 0이든 아니든 무조건 "a"로 적는다. 가혹한 판(독·공격 규칙에서 수용력 근처로 시작해 둘 다 무너지는 경우)에서 B가 먼저 죽고 A가 나중에 죽어도 "a"다. 그러면 verdict()가 r(기울기)을 보기도 전에 `a_extinct_runs == reps`로 "B가 이김(A가 6회 모두 전멸)"을 낸다. 같은 판의 r은 먼저 죽은 B가 졌다고 말한다. compare.py는 이 문자열을 '켠 쪽/끈 쪽'으로 바꿔 읽으므로, 켠 규칙에서는 '켠 쪽이 이김', 끈 규칙에서는 '끈 쪽이 이김'이 나와 `_reading`이 "켠 규칙에서만 켠 쪽이 이겼다. 장치와 함께 진화한 것이 값을 한다"를 낼 수 있다. 장치를 켜서 둘 다 무너지는 조건이 바로 공격·독 실험이라 이 경로가 열려 있다. 둘 다 전멸한 판은 '무승부'로 따로 세야 한다.
- 인용: "extinct": "a" if arr[-1, 1] == 0 else ("b" if arr[-1, 2] == 0 else "")
- 검증 근거: kkumteul/measure.py:163 `"extinct": "a" if arr[-1, 1] == 0 else (...)`. 끝 표본에서 A가 0이면 B가 0이어도 "a"다. measure.py:213-214 verdict()는 r을 보기 전에 `a_extinct_runs == reps`이면 "B가 이김(A가 N회 모두 전멸)"을 돌려준다. 반면 r(measure.py:146-160)은 처음 전멸한 표본까지만 써서 먼저 죽은 쪽이 졌다고 말할 수 있다. compare.py:61,63 _name은 "A가/B가"를 켠/끈 쪽으로 바꾸고, :88 startswith("켠 쪽이 이김")로 읽으므로 이 문자열이 그대로 이김으로 셈된다. 단, 지적에 인용된 _reading 문구("장치와 함께 진화한 것이 값을 한다")는 지금 코드(compare.py:93)와 다르다. 지금은 "켠 규칙에 적응했다"라고 쓴다. 그래도 틀린 판정이 그대로 전달된다는 핵심은 맞다. 실제로 두 쪽이 다 전멸하는 일이 얼마나 자주 생기는지는 실행해 봐야 안다.
- 고치는 방향(검증자): compete에서 둘 다 0이면 "both"로 적고(또는 각자 처음 0이 된 표본 스텝을 견줘 먼저 죽은 쪽을 적고), assay_sides·verdict에서 무승부를 따로 센다.
- 처리: (미정)

### 결함-2. compare의 읽기가 '켠 규칙에서만 이김'을 유의성 차이로 판정하고, 계통 1개씩으로 장치 효과를 단정한다
- 종류: 중요 · 자리: `kkumteul/compare.py:92` · 검증: **확인됨**
- 지적: (1) off_win이 False라는 것은 끈 규칙에서 '가를 수 없음'일 수도 있다. 한쪽은 유의하고 다른 쪽은 유의하지 않다는 사실만으로 두 규칙 사이에 차이가 있다고 말하는 오류이고, 두 규칙의 r 차이를 검정하지 않는다. (2) 6회 반복은 같은 두 개체군(켠 실행 1개, 끈 실행 1개)에서 400마리씩 다시 뽑은 것이라 겨루기판의 잡음만 잰다. 실행 사이 분산(진화 계통의 운)은 들어 있지 않아, 장치 효과와 계통 차이를 가를 수 없는데 문구는 인과로 쓴다. (3) verdict 문자열 끝의 "(주의: 같은 무리끼리도 한쪽으로 기울었다)"는 startswith로만 읽혀 무시된다. (4) 켠 규칙은 '가를 수 없음'이고 끈 규칙에서 켠 쪽이 이기는 경우(off_win and not on_win)는 어느 분기에도 안 걸려 맨 끝 "가를 수 없다."로 떨어진다. 장치 없이도 켠 쪽이 낫다는 정보가 사라진다.
- 인용: if on_win and not off_win:         return "켠 규칙에서만 켠 쪽이 이겼다. 장치와 함께 진화한 것이 장치가 있을 때 값을 한다."
- 검증 근거: (1) compare.py:92 `if on_win and not off_win`: off_win이 False이면 끈 규칙의 '가를 수 없음'도 여기에 들어간다. 두 규칙의 r 차이를 검정하는 코드는 없다. (2) measure.py:191 runs는 같은 a/b 유전체에서 seed만 바꿔 뽑는다(measure.py:126-127 rng.choice). compare.py:48-49는 양쪽 실행마다 체크포인트를 하나씩 쓴다. 그래서 실행 사이 분산은 측정되지 않는다. (3) measure.py:222-227에서 warn 접미사가 붙는다. compare.py:88-91은 startswith로만 읽어 경고를 무시한다. (4) on_win=F, on_lose=F, off_win=T이면 compare.py:92-102의 어느 분기에도 걸리지 않고 :103 "가를 수 없다."로 떨어진다. 인용 문구는 지금 코드(:93-94, "적응했다는 뜻이다. 장치를 '쓰게 됐다'는 뜻은 아니다")보다 단정적이다. 지금 문구는 조금 누그러졌다. 하지만 (1)~(4)의 논리 결함은 그대로다.
- 고치는 방향(검증자): 두 규칙의 r 차이에 대한 검정(또는 같은 부호인지 확인)을 넣는다. 경고 접미사는 읽기 문장에 이어 붙인다. off_win and not on_win 분기를 추가한다. 실행 반복(서로 다른 seed의 계통)이 없다는 한계를 문구에 적는다.
- 처리: (미정)

### 결함-3. 옛 체크포인트에서 공격을 켜면 '공격하지 않는 쪽으로 기울임'이 적용되지 않는다
- 종류: 중요 · 자리: `kkumteul/kernels.py:799` · 검증: **확인됨**
- 지적: -1 편향은 random_genome으로 새로 만든 개체에만 들어간다. 이 커밋 전에 저장된 체크포인트(runs/main 등)에서 `preset=conflict`로 가르면 world.load_arrays는 배열을 그대로 읽거나 0으로 채우기만 한다. 예전에 '예비 1'이던 8번 출력의 bout은 0이고, Wout은 init_w_out=0.1짜리 무작위 값에서 중립 변이로 떠돌았다. 그래서 개체의 약 절반이 첫 스텝부터 양의 공격 출력(대략 0.1~0.2)을 낸다. docs/stages-2-4.md의 "처음 개체는 공격하지 않는 쪽으로 기울여 두었다"는 새 실행에서만 맞다. 갈래로 돌린 켠 쪽은 처음부터 약한 공격자가 절반 섞인 채 시작하므로, 대조와의 차이를 '공격이 생겼다'로 읽으면 안 된다. 주기·쌓기에서 이미 겪은 문제(검토 기록 5번)와 같은 종류다.
- 인용: s.bout[j, ACT_ATTACK] = -1.0
- 검증 근거: kernels.py:799 `s.bout[j, ACT_ATTACK] = -1.0`은 random_genome 안에만 있다. 866bbed 직전 버전(git show 866bbed^:kkumteul/kernels.py:768-771)은 GIVE와 DEPOSIT만 -1이고 8번(config 이력에 '예비 1')은 0이다. world.py:171-197 load_arrays는 bout 모양이 같으면 그대로 복사한다. run.py:219-269 branch와 config.py:329 preset conflict도 bout를 고치지 않는다. kernels.py:880-883에 따르면 bout는 변이로 떠돌고, 출력은 tanh(bout + Wout·h)(kernels.py:523-526, stages.py:352-355)다. stages.py:502-504는 a>0이면 공격한다. 따라서 옛 체크포인트에서 갈라 conflict를 켜면 상당수 개체가 첫 스텝부터 양의 공격 출력을 낸다. docs/stages-2-4.md:43의 '공격하지 않는 쪽으로 기울여 두었다'는 새 실행에만 맞다. '약 절반', '0.1~0.2'라는 수치는 코드만으로 확인할 수 없다. 옛 체크포인트를 읽어 act[:,8]>0인 비율을 세 보면 가려진다.
- 고치는 방향(검증자): 갈래를 만들 때 conflict_enabled가 새로 켜지면(config_diff) 살아 있는 개체의 bout[:,ACT_ATTACK]를 -1로 맞추는 옵션을 넣고, 무엇을 했는지 run.json에 적는다. 대조 갈래에도 똑같이 적용한다.
- 처리: (미정)

### 결함-4. 공격으로 죽은 개체가 '굶어 죽음'으로만 집계되어 공격 실험의 사망 통계를 읽을 수 없다
- 종류: 사소 · 자리: `kkumteul/stages.py:539` · 검증: **확인됨**
- 지적: 공격이 에너지를 0까지 깎으면 process_deaths가 `energy <= 0`으로 DEATH_STARVE로 처리한다. 사망 원인 코드도, 죽인 횟수 카운터도 없다. C_ATTACKS는 '맞은 횟수'뿐이다. 통계의 w_starve와 사건 기록 cause가 기아사와 맞아 죽음을 섞는다. 공격 세계가 무너졌는지, 포식이 생겼는지 가르는 일(tasks.md 17번)에서 w_starve 증가를 기아로 오독할 수 있다. 맞아 죽은 개체는 사체 먹이도 남기지 않는다(recycle=False).
- 인용: s.energy[best] -= dmg         s.energy[i] += got         s.ledger[L_LOST] += dmg - got
- 검증 근거: stages.py:535-541은 에너지를 깎고 C_ATTACKS만 1 늘린다. 이 값은 '때린 횟수'다(맞은 개체마다 따로 세지 않는다). kernels.py:961-963에 따르면 energy<=0이면 원인을 가리지 않고 DEATH_STARVE, C_DEATH_STARVE로 처리하고, recycle=False라 사체가 남지 않는다. 맞아 죽었을 때 에너지는 정확히 0(stages.py:536-537의 dmg 상한)이므로, 사체를 남기더라도 남길 에너지가 없다. 따로 쓰는 사망 원인 코드나 '죽인 횟수' 카운터는 없다.
- 고치는 방향(검증자): attack에서 상대를 0으로 만든 때를 표시해 두고(예: last_hit 배열 또는 C_KILLS), process_deaths에서 DEATH_KILLED를 따로 기록한다.
- 처리: (미정)

### 결함-5. compare 카드가 켠/끈 장치를 mind·social·niche만 적어, 공격·먹이 종류 비교는 장치가 안 보인다
- 종류: 사소 · 자리: `kkumteul/compare.py:25` · 검증: **확인됨**
- 지적: 겨루기판 설정(arena_config)은 conflict_enabled·food_kinds_enabled를 포함해 전체 설정을 그대로 복사하므로 장치는 실제로 실린다. 그러나 `flags`와 출력의 "장치(끈 쪽, 켠 쪽)" 줄은 이 세 개만 보여 준다. 대조=stage3, 실험=stage3+conflict로 견주면 카드에 두 쪽이 같은 값으로 찍혀 무엇을 견줬는지 남지 않는다. 두 실행의 나머지 설정(seed, 먹이 유입 등)이 같은지 점검하지 않는다. 또 traits의 '끝'은 stats.jsonl의 마지막 줄(compare.py:72)이라 --at으로 고른 스텝이 아니라 실행이 끝난 시점일 수 있고, 양쪽 마지막 스텝도 다를 수 있다.
- 인용: STAGE_FLAGS = ("mind_enabled", "social_enabled", "niche_enabled")
- 검증 근거: compare.py:25 STAGE_FLAGS는 세 개뿐이고, :59 flags와 :119 출력도 이것만 쓴다. measure.py:83-88 arena_config는 cfg.to_dict() 전체를 복사하므로 conflict_enabled와 food_kinds_enabled도 실린다. 두 실행의 나머지 설정이 같은지는 점검하지 않는다. compare.py:50-52는 스텝과 유전체 모양만 본다(measure.py:69-80). compare.py:70-74의 traits '끝'은 stats.jsonl의 rows[-1]이라 --at과 상관없다.
- 고치는 방향(검증자): 두 cfg의 to_dict() 차이를 통째로 카드에 적고, 설정 중 장치 플래그(conflict_enabled, food_kinds_enabled 포함)를 보여 준다. traits는 --at 스텝에 가장 가까운 줄을 쓴다.
- 처리: (미정)

### 결함-6. migrate가 검사 실패 시 반쯤 만든 실행 폴더를 남긴다
- 종류: 사소 · 자리: `kkumteul/pack.py:150` · 검증: **확인됨**
- 지적: Runner.branch는 run.json·config.json·체크포인트를 먼저 쓰고, 유전체 모양 검사(check_same_shape)와 빈 자리 검사(`빈 자리가 모자란다`)는 그 뒤에 한다. 묶음이 다른 크기의 세계에서 온 경우나 n이 큰 경우에 ValueError가 나도 run_dir에는 이주자가 없는 갈래 실행이 남는다. 같은 --run으로 다시 하면 FileExistsError이고, 그 폴더를 `resume`으로 돌리면 이주자 없는 세계가 조용히 돈다. run.json에 migrants 기록도 없다.
- 인용: runner = Runner.branch(Path(src), Path(run_dir), None, [], False, False)     try:         world = runner.world         measure.check_same_shape(side, ...
- 검증 근거: pack.py:150 Runner.branch가 먼저 불린다. run.py:267-268에서 _create 뒤에 r.checkpoint()가 불려 run.json·체크포인트가 써진다. 그 뒤 pack.py:153-157의 check_same_shape와 '빈 자리가 모자란다' ValueError가 나면 finally에서 runner.close()만 하고 폴더는 지우지 않는다. 다시 실행하면 run.py:222-223에서 FileExistsError가 난다. migrants 기록은 pack.py:170-171에서 성공했을 때만 써진다.
- 고치는 방향(검증자): 묶음을 읽고 모양·자리 검사를 branch보다 먼저 하거나(체크포인트를 ckpt.load로 먼저 읽어 검사), 실패하면 만든 run_dir을 지운다.
- 처리: (미정)

### 결함-7. 이주자의 세대(gen)가 0으로 초기화되어 gen_mean 계열 통계가 왜곡된다
- 종류: 사소 · 자리: `kkumteul/measure.py:100` · 검증: **확인됨**
- 지적: 겨루기용 _inject를 그대로 쓰므로 묶음에 있는 gen은 버려지고 이주자는 gen 0이 된다. 받는 세계의 gen_mean은 수천 세대인데 이주자와 그 후손은 0에서 다시 세어, 이주 뒤의 gen_mean·gen_max가 내려간다. experiment.py의 generations_gained(=gen_mean 끝−처음)와 analyze.py의 평균 세대가 이주 실행에서는 의미가 달라진다. 이주 실행의 gen_mean을 대조 실행과 그대로 견주면 안 된다.
- 인용: s.ids[j], s.parent[j], s.gen[j], s.tag[j] = aid, -1, 0, tag
- 검증 근거: measure.py:100 `s.gen[j] ... = 0`. pack.py:164-165 migrate는 이 _inject를 그대로 쓰고, g["gen"]은 BIRTH 기록에도 넣지 않는다(rec의 gen=0). 묶음의 gen은 버려진다. 후손은 부모 gen+1로 세므로 이주자 계통은 0에서 다시 센다.
- 고치는 방향(검증자): migrate에서는 _inject 뒤에 s.gen[j] = g['gen'][row]로 되돌리거나, _inject에 keep_gen 인자를 둔다. 겨루기 쪽은 지금대로 둔다.
- 처리: (미정)

### 결함-8. migrant_frac 필드가 이주자가 모두 사라지면 통계 줄에서 빠진다
- 종류: 사소 · 자리: `kkumteul/world.py:283` · 검증: **확인됨**
- 지적: 이주자 계통이 전멸하면 그 줄부터 키가 없어져 값 0이 아니라 '없음'이 된다. 이주한 적 없는 실행과 구분되지 않고, stats.jsonl을 표나 그림으로 읽는 쪽은 결측 처리를 해야 한다. 이주 실험에서 가장 중요한 사건인 '이주자 소멸'이 필드 부재로 나타난다. 또 tag가 0이 아닌 모든 개체를 세므로, 두 번째 이주(같은 기본 tag=1)와 첫 이주를 가를 수 없다. 해당 필드를 이주를 한 번이라도 한 실행(run.json의 migrants)이면 항상 적는 쪽이 안전하다.
- 인용: if (s.tag[a] != 0).any():      # 이주해 온 생물과 그 후손의 비율                 out["migrant_frac"] = float((s.tag[a] != 0).mean())
- 검증 근거: world.py:283-284 `if (s.tag[a] != 0).any(): out["migrant_frac"] = ...`. 이주자 계통이 전멸하면 키가 빠진다. tag가 0이 아니면 모두 센다. pack.py:144의 migrate tag 기본값은 1이고, CLI(pack.py mg 서브파서)에는 --tag 옵션이 없어 두 번째 이주도 tag 1이 된다.
- 고치는 방향(검증자): run.json에 migrants가 있으면(또는 C_* 카운터로 이주한 적이 있으면) migrant_frac를 항상 적는다. 이주마다 다른 tag를 쓰도록 CLI에 --tag를 연다.
- 처리: (미정)

### 결함-9. 시야의 먹이 부호가 설정(Q_KINDS)이 아니라 저장된 f_kind만 보고 정해진다
- 종류: 사소 · 자리: `kkumteul/kernels.py:462` · 검증: **확인됨**
- 지적: eat의 독 판정은 `ip[Q_KINDS] != 0`로 막는데 _sense_par는 막지 않는다. food_kinds_enabled=True로 돌던 실행에서 갈라 kinds를 끈 대조 갈래를 만들면, 갈라낸 시점에 남아 있던 종류 1 먹이가 먹힐 때까지 음수로 보인다. 독은 없는데 관측만 달라, 같은 체크포인트에서 켜고 끄는 비교의 대조가 깨끗하지 않다. 반대로 옛 체크포인트(f_kind 없음)에서 kinds를 켜면 기존 먹이와 사체가 모두 종류 0이 된다(사체가 2가 아님).
- 인용: _see(obs, i, 0, nsec, ddx, ddy, math.sqrt(d2), f_rad, hi, fov, vrange,                          np.float32(-1.0) if f_kind[j] == 1 else np.float32(1.0))
- 검증 근거: kernels.py:462-463의 _sense_par는 `f_kind[j] == 1`만 보고 ip[Q_KINDS]를 받지도 않는다(:411 인자 목록). eat은 kernels.py:662,707에서 `kinds = ip[Q_KINDS] != 0`으로 막는다. kinds를 끈 갈래에서 새로 나는 먹이는 종류 0이다(:327). 그래서 갈라낼 때 남아 있던 종류 1 먹이만 계속 음수로 보인다. f_kind는 world.py:15의 NEW_SINCE_STAGE1에 있어서, 옛 체크포인트에서는 load_arrays가 missing으로 남기고 0으로 둔다. 기존 사체도 종류 0(2가 아님)이 된다.
- 고치는 방향(검증자): _sense_par에 kinds 플래그를 넘겨 꺼져 있으면 부호를 1로 고정한다. 또는 kinds를 끄는 갈래를 만들 때 f_kind[f_kind==1]=0으로 맞춘다.
- 처리: (미정)

### 결함-10. 사체(kind 2)가 시야에서 종류 0과 같은 양수로 보인다
- 종류: 사소 · 자리: `kkumteul/kernels.py:930` · 검증: **확인됨**
- 지적: 종류 2(사체)는 늘 먹을 수 있는데 _sense_par는 `f_kind[j] == 1`만 음수로 만들어 사체가 종류 0과 구분되지 않는다. corpse_to_food 기본이 True이므로 늙어 죽은 개체의 에너지가 곳곳에 양수 먹이로 놓인다. 종류 0이 독인 구간에서는 '양수=독'을 배운 개체가 사체를 먹어 손해를 보고, 사체를 먹은 개체는 독을 먹은 것처럼 보인다. 배움이 값을 하는지 재려는 실험(tasks.md 16번)에서 신호에 사체만큼의 잡음이 낀다. 사체를 따로 부호화하거나 kinds 세계에서는 사체를 끄는 설정이 필요하다.
- 인용: if recycle and e > 0.0 and _place_food(s, s.x[i], s.y[i], e, 2):
- 검증 근거: kernels.py:930은 사체를 kind 2로 둔다. 시야(kernels.py:463)는 1만 -1, 나머지(0과 2)는 +1이다. config.py:63 corpse_to_food=True가 기본값이고, kernels.py:965-966에서 늙어 죽으면 recycle된다. eat(kernels.py:707)에서 kind 2는 독 판정에서 빠져 언제나 먹을 수 있다. 다만 '양수=독을 배운 개체가 사체를 먹어 손해를 본다'는 서술은 정확하지 않다. 사체는 해가 없으니 그런 개체는 사체를 피해서 먹을 기회를 잃을 뿐이다. 신호에 잡음이 낀다는 핵심은 맞다.
- 고치는 방향(검증자): 사체를 시야에서 따로 부호화하거나(예: 0 또는 별도 채널), kinds 세계에서는 corpse_to_food를 끄는 설정을 프리셋에 넣는다.
- 처리: (미정)

### 결함-11. 공격은 칸 번호 순서로 순차 처리되어 같은 스텝 안에서 낮은 칸의 공격자가 먼저 친다
- 종류: 사소 · 자리: `kkumteul/stages.py:499` · 검증: **확인됨**
- 지적: 에너지는 보존되지만, 먼저 처리되는 개체가 상대의 에너지를 먼저 깎는다. 맞은 개체의 에너지가 0이 되면 그 개체의 차례에 `s.energy[i] <= 0.0`으로 건너뛰어 반격이 사라진다. 출생 칸은 first-fit이라 칸 번호가 나이·세대와 상관이 있는데, process_births는 같은 문제(검토 2026-11-04)로 시작 칸을 스텝마다 돌리지만 attack은 돌리지 않는다. 피해량(0.2)이 에너지(수 단위)에 비해 작아 영향은 작지만, 약한 개체끼리의 맞대결이나 포식 실험에서는 칸 번호가 승패에 섞인다.
- 인용: for i in range(n):         if not s.alive[i]:             continue         a = float(s.act[i, ACT_ATTACK])
- 검증 근거: stages.py:499 `for i in range(n):`는 시작 칸을 돌리지 않는다. :503 `if a <= 0.0 or s.energy[i] <= 0.0: continue`이므로, 먼저 맞아 0이 된 개체는 반격하지 못한다. 반면 kernels.py:1010 process_births는 `start = int(u01(..., S_BIRTH_ORDER, step, 0) * n)`으로 시작 칸을 스텝마다 돌린다. 영향이 얼마나 큰지는 실행해 봐야 안다.
- 고치는 방향(검증자): process_births처럼 새 rng 용도(뒤에 덧붙임)로 시작 칸을 돌리거나, 피해를 모아 두었다가 스텝 끝에 한꺼번에 적용한다.
- 처리: (미정)
