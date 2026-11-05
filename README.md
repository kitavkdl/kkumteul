# 꿈틀 (kkumteul)

아무도 가르치지 않았는데 생물들이 스스로 행동·전략·신호를 만들어 내는지 보는 세계 시뮬레이션이다.
감각도 기억도 수명도 제한된 작은 존재들이 2D 세계에서 먹고, 낳고, 죽는다. 정해 준 것은 물리와 에너지 규칙뿐이고 나머지는 진화에 맡긴다.

| 단계 | 내용 | 상태 |
|---|---|---|
| 1 세계와 몸 | 2D 세계, 몸, 죽음/번식, 뇌 에너지 비용, 감각 한계 | 구현됨. 맥 시범 실행에서 재주입 없이 유지되고 진화함 |
| 2 마음 | 호르몬 보상, 평생 학습, 호기심, 잠 | 장치를 넣음(기본은 꺼짐). 생물이 쓰게 되는지는 아직 못 잼 |
| 3 사회 | 신호, 혈연, 이웃 보기, 에너지 주기 | 장치를 넣음(기본은 꺼짐). 위와 같음 |
| 4 판이 바뀌는 세계 | 길, 거름, 재난·기후 각본 | 장치를 넣음(기본은 꺼짐). 위와 같음 |
| 항상 | 성장 측정, 저장/분기, 관찰 화면 | 구현됨 |

- 왜 만드는지와 설계 생각: [docs/design-notes-2026-10-28.md](docs/design-notes-2026-10-28.md)
- 2~4단계 장치: [docs/stages-2-4.md](docs/stages-2-4.md)
- 1단계 결정과 근거: [wayfinder/stage1/map.md](wayfinder/stage1/map.md), 조사 보고서 [wayfinder/stage1/reports/](wayfinder/stage1/reports/)
- 첫 실행 결과: [wayfinder/stage1/research/first-run-mac.md](wayfinder/stage1/research/first-run-mac.md)
- 같이 실험하기: [CONTRIBUTING.md](CONTRIBUTING.md)

이 레포의 문서는 만든 사람("대표님")과 Claude가 일하며 남긴 기록을 그대로 둔 것이다.


**모든 숫자는 임시 결정(대표님 확인 전)이다.** 값은 `kkumteul/config.py` 한 곳에 있고 실행할 때 `--set 이름=값`으로 바꾼다.

필요한 것: Python 3.10 이상, numpy, numba. (2026-11-04 맥에서 Python 3.10.12, numpy 2.2.6, numba 0.67.0으로 확인)

## 쓰는 법

모든 명령은 이 폴더(이 레포)에서 친다. 실행 결과는 `runs/<이름>/`에 쌓이고 git에는 올리지 않는다.

```bash
# 새 세계를 만들어 계속 돌린다 (Ctrl+C나 SIGTERM이면 저장하고 멈춘다)
python3 -m kkumteul.run new --run runs/main

# 꺼졌던 세계를 마지막 체크포인트에서 이어 돌린다
python3 -m kkumteul.run resume --run runs/main

# 같은 시점에서 갈라 "~였다면?"을 본다 (예: 뇌 비용을 3배로)
python3 -m kkumteul.run branch --from runs/main --run runs/costly-brain --set brain_cost=0.018

# 선택 없는 대조 실행: 본 실행의 출생·사망 횟수를 무작위 개체에 그대로 적용한다
python3 -m kkumteul.run branch --from runs/main --at 0 --run runs/main-passive --passive --new-stream

# 관찰 화면 (폰에서 보려면 --host 0.0.0.0)
python3 -m kkumteul.viewer --runs runs --port 8765

# 기록만으로 내는 지표: 하루당 세대, 출생 비, Fitness Transmission 등 (긴 실행은 --last-steps로 끝부분만)
python3 -m kkumteul.analyze runs/main
python3 -m kkumteul.analyze runs/main --last-steps 5000000

# 성장 측정: 옛 체크포인트 개체와 지금 개체를 새 세계에 같은 수로 풀어 겨루게 한다
python3 -m kkumteul.measure assay --a runs/main/checkpoints/ckpt_000001000000.npz --b runs/main/checkpoints/ckpt_000008000000.npz
python3 -m kkumteul.measure ladder --run runs/main --count 5

# 속도·램 실측
python3 -m kkumteul.bench

# 시험
python3 -m pytest -q
```

## 세계의 규칙 (한 스텝)

1. 먹이가 일정한 양만큼 들어온다. 대부분 몇 군데 자리에 뭉쳐 나고, 자리는 가끔 옮겨 간다.
2. 개체가 느낀다: 앞쪽 120도·거리 16을 8칸으로 나눈 시야(칸마다 가장 가까운 먹이·개체의 가까움), 자기 에너지·나이·속도, 닿음, 먹음, 주변 먹이 냄새.
3. 뇌(순환 신경망)가 추력·회전·번식 출력을 낸다.
4. 움직인다. 겹친 개체는 위치만 밀려난다. 먹이에 닿으면 먹는다.
5. 에너지를 쓴다: 기초 대사 + 뇌 유지비(활성 뉴런·시냅스 수에 비례) + 이동 비용(명령한 추력에 비례).
6. 에너지가 0 이하이거나 최대 수명을 넘으면 죽는다. 늙어 죽은 개체의 남은 에너지는 먹이로 남는다.
7. 에너지가 문턱을 넘고 번식 출력이 켜져 있으면 번식한다. 자식 에너지는 부모에게서만 나오고, 유전체는 변이하며 복사된다.
8. 개체 수가 하한 아래로 떨어지면 살아 있는 개체의 변이 복제본을 넣고 그 횟수를 기록한다(재주입). 재주입이 있던 구간은 성장 판정에서 뺀다.

개체 수는 따로 조절하지 않는다. 먹이 유입 ÷ 개체당 소비가 수용력을 정한다.

## 코드 구성

| 파일 | 하는 일 |
|---|---|
| `kkumteul/config.py` | 모든 설정값과 검사 |
| `kkumteul/rng.py` | 상태 없는 난수(씨앗·용도·번호를 섞어 만든다) |
| `kkumteul/kernels.py` | numba 계산 커널: 격자, 감각, 뇌, 움직임, 충돌, 먹기, 죽음, 번식, 대조 실행 |
| `kkumteul/world.py` | 세계 하나: 한 스텝의 순서, 사건 기록, 에너지 장부 점검 |
| `kkumteul/checkpoint.py` | 저장·복구(임시 파일에 쓰고 이름을 바꾼다), 오래된 체크포인트 정리 |
| `kkumteul/run.py` | 실행기: 새로 시작·이어 돌리기·분기, 통계·프레임·체크포인트 기록, 실행 폴더 잠금, 디스크 여유 감시 |
| `kkumteul/events.py` | 출생·사망 기록 파일(스텝 구간마다 나눔) |
| `kkumteul/_cache.py` | 계산 소스가 바뀌면 numba 캐시를 지운다 |
| `kkumteul/measure.py` | 성장 측정(겨루기, 사다리) |
| `kkumteul/analyze.py` | 기록 분석(세대, 출생 비, Fitness Transmission) |
| `kkumteul/viewer.py`, `viewer.html` | 관찰 화면 |
| `kkumteul/bench.py` | 속도·램 실측 |
| `deploy/kkumteul.service` | 24시간 가동용 서비스 등록 초안(아직 설치 안 함) |

## 실행 폴더 안

| 파일 | 내용 |
|---|---|
| `config.json`, `run.json` | 설정, 만든 방법(분기면 부모 체크포인트 해시와 설정 차이) |
| `checkpoints/ckpt_<스텝>.npz` + `.json` | 세계 전체 상태. 100만 스텝마다 쓰고, 최근 12개와 5천만 스텝마다 하나를 남긴다 |
| `events/births_<번호>.bin`, `deaths_<번호>.bin` | 출생·사망 기록(덧붙이기 전용, 200만 스텝마다 새 파일). 형식은 `world.py`의 `BIRTH_DTYPE`, `DEATH_DTYPE` |
| `stats.jsonl` | 5,000스텝마다 한 줄 요약 |
| `STOPPED.txt` | 스스로 멈췄을 때 그 까닭(디스크 여유 부족) |
| `frames/seg_*.bin`, `keyframes.bin` | 관찰 화면용 프레임. 최근 것만 남기고(고리), 드문 프레임은 계속 남긴다 |

## 지켜야 하는 것

- **병렬 커널은 배열을 직접 받는다.** numba는 `prange` 안에서 상태 묶음(namedtuple)을 거쳐 배열에 쓰면 오류 없이 그 쓰기를 버린다(2026-11-04에 겪음). `tests/test_core.py`의 `test_parallel_kernels_actually_write`가 지킨다.
- **에너지가 생기는 길은 먹기와 명시적 주입뿐이다.** 충돌·경계는 위치만 바꾼다. `World.energy_balance()`의 오차가 0에 가까운지 통계에 계속 적는다.
- **난수는 `rng.py`의 해시만 쓴다.** 그래서 같은 씨앗이면 스레드 수와 무관하게 같은 세계가 나오고, 체크포인트에서 이어 돌린 결과가 끊지 않고 돌린 결과와 비트까지 같다. 다른 기계(맥과 서버) 사이에서는 같다고 보장하지 않는다.
- `rng.py`의 용도 번호와 `kernels.py`의 난수 호출 순서를 바꾸면 옛 체크포인트에서 이어 돌린 결과가 달라진다.

## 디스크가 차는 속도 (맥, 전속 약 1,000스텝/초 기준, 독립 검토 실측)

| 항목 | 하루 증가 | 줄이는 설정 |
|---|---|---|
| 출생·사망 기록 | 약 6.7GB (스텝당 79바이트) | `event_keep_segments`(최근 몇 개만 남김), 또는 `max_steps_per_sec`로 속도를 늦춤 |
| 프레임(고리) | 늘지 않음, 최대 약 1GB | `frame_ring_segments`, `frame_every`, `clip_every` |
| 영구 보존 체크포인트 | 약 0.1GB | `ckpt_keep_every` |
| 드문 프레임, 통계 | 약 0.03GB | `keyframe_every`, `stats_every` |

기록 보존 기한의 기본값은 "전부 남김"이다. 디스크 여유가 `min_free_gb`(2GB) 아래로 내려가면 저장하고 스스로 멈춘다.

## 측정값을 읽을 때

- 겨루기의 r은 같은 무리끼리도 흔들린다(1,000스텝당 표준편차 0.08쯤). 6회 반복으로는 꽤 큰 차이만 가를 수 있다.
- 한쪽이 전멸한 판은 r의 크기에 뜻이 없다. "전멸"만 읽는다.
- 추이 성분 비율은 방향을 모른다. 순수 잡음에서도 2/n(5개면 0.4)이 나온다. "등급과 시간의 순위 상관"과 같이 본다.
- Fitness Transmission은 유전 변이가 없어도 평균 +0.01쯤 나온다. 대조 실행에 같은 분석을 돌린 값과 비교한다.
- 뇌 크기·속도 같은 형질의 추세는 성장의 증거로 쓰지 않는다(선택이 없어도 변한다).

독립 검토의 지적과 처리는 `wayfinder/stage1/research/code-review-2026-11-04.md`에 있다.

## 아직 없는 것

- 포식·공격, 유성 생식, 몸 형태 진화: 넣지 않았다.
- 체크포인트를 시간별→일별로 솎는 정책, 멈췄을 때 알림, 성장 측정을 주기적으로 자동으로 돌리기: 실행 자리가 정해진 뒤에 붙인다.
