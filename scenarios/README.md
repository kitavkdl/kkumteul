# 각본 (scenarios)

각본은 세계에 주는 조건을 적은 json 파일 하나다. 가뭄, 빙하기, 계절, 운석 같은 것을 코드를 고치지 않고 줄 수 있다.
같은 체크포인트에서 각본만 다르게 갈라 돌리면 "그 일이 없었다면?"과 나란히 비교할 수 있다.

## 쓰는 법

```bash
# 각본을 검사하고 시간에 따라 값이 어떻게 변하는지 본다(세계를 돌리지 않는다)
python3 -m kkumteul.scenario scenarios/drought.json

# 새 세계를 각본과 함께 시작한다
python3 -m kkumteul.run new --run runs/dry --scenario scenarios/drought.json

# 돌던 세계의 체크포인트에서 갈라 각본을 준다(각본의 시간은 갈라낸 스텝부터 센다)
python3 -m kkumteul.run branch --from runs/base --run runs/base-dry --scenario scenarios/drought.json

# 대조(각본 없음)와 각본을 같이 돌리고 결과 카드까지 쓴다
python3 -m kkumteul.experiment --from runs/base --scenario scenarios/drought.json --steps 200000 --name my-drought
```

각본은 실행 폴더에 `scenario.json`으로 베껴진다. 원본 파일을 고치거나 지워도 그 실행은 처음 받은 각본으로 이어 돈다.

## 들어 있는 각본

| 파일 | 무슨 일이 생기나 |
|---|---|
| `drought.json` | 가뭄: 1만 스텝째부터 3만 스텝 동안 먹이 유입 0.3배 |
| `famine-and-recovery.json` | 기근과 회복: 4만 스텝 동안 0.1배, 이어서 2만 스텝 동안 1.5배 |
| `ice-age.json` | 빙하기: 기초 대사 비용이 20만 스텝에 걸쳐 1.6배까지 오른다 |
| `seasons.json` | 계절: 먹이 유입이 2만 스텝 주기로 ±60% |
| `warming.json` | 기후 변화: 먹이가 서서히 줄고(30만 스텝에 0.6배) 계절이 중간부터 거세진다 |
| `meteor.json` | 운석: 2만 스텝째에 큰 것 하나(반지름 180) |
| `meteor-shower.json` | 운석 소나기: 4천 스텝마다 작은 것(반지름 45), 12만 스텝째까지 |

숫자는 기본 설정(512 x 512 세계, 먹이 유입 9)에 맞춰 잡은 임시 값이다. 파일의 `description`에 무엇을 볼지 적혀 있다.

## 형식

```json
{
 "name": "가뭄",
 "description": "무슨 일이 생기고 무엇을 볼지",
 "update_every": 100,
 "events": [
  {"at": 10000, "for": 30000, "scale": {"food_inflow": 0.3}}
 ]
}
```

| 키 | 뜻 | 없으면 |
|---|---|---|
| `name` | 각본 이름. 실행 폴더의 `run.json`과 실험 카드에 적힌다 | 파일 이름 |
| `description` | 설명 | 빈 글 |
| `update_every` | 서서히 변하는 값(`scale_to`, `amp`)을 몇 스텝마다 다시 계산할지 | 100 |
| `events` | 사건 목록(아래 다섯 가지). 비어 있어도 된다 | 꼭 있어야 한다 |

여기에 없는 키나 모르는 설정 이름이 있으면 읽을 때 오류가 난다(오타를 조용히 넘기지 않으려는 것이다).
사건마다 `"note": "메모"`를 붙일 수 있다. 세계에는 아무 영향이 없다.

**시간은 모두 각본이 시작된 스텝부터 센 스텝 수다.** 새 세계라면 0스텝부터, 갈라낸 세계라면 갈라낸 스텝부터다.
시작 스텝은 실행 폴더 `run.json`의 `scenario_start`에 적힌다.

## 사건 다섯 가지

### 1. `set` — 값을 바꾼다

```json
{"at": 50000, "set": {"food_inflow": 4.0, "corpse_to_food": false}}
```

`at` 스텝째부터 설정값이 적은 값으로 바뀌고 그대로 간다. 켜고 끄는 설정(true/false)은 `set`으로만 바꿀 수 있다.
같은 설정에 `set`이 여럿이면 더 늦은 것이 이긴다.

### 2. `scale` — 한동안 배율을 곱한다

```json
{"at": 10000, "for": 30000, "scale": {"food_inflow": 0.3}}
```

`at`부터 `for` 스텝 동안(`at` 이상 `at + for` 미만) 값에 배율을 곱한다. 끝나면 원래대로 돌아온다. 배율은 0 이상이다.

### 3. `scale_to` — 서서히 변한다

```json
{"from": 0, "to": 200000, "scale_to": {"basal_cost": 1.6}}
```

배율이 `from`에서 1로 시작해 `to`에서 목표값이 될 때까지 곧은 선으로 변하고, 그 뒤로는 목표값에 머문다.

### 4. `amp` — 계절

```json
{"every": 20000, "amp": {"food_inflow": 0.6}, "phase": 0.0}
```

배율 = 1 + amp × sin(2π × (t / every + phase)). `every`가 한 해의 길이다. `amp`는 -1에서 1 사이다.
`phase`가 0이면 배율 1(봄)에서 시작해 1/4 주기에 가장 높다. 0.25면 가장 높을 때(한여름)에서, 0.75면 가장 낮을 때에서 시작한다.
`"from": T`를 붙이면 T 스텝째부터만 적용된다(주기의 배수로 잡으면 배율 1에서 매끄럽게 시작한다).

### 5. `strike` — 운석

```json
{"at": 20000, "strike": {"radius": 60, "kill": 1.0, "burn_food": true, "x": null, "y": null}}
{"every": 4000, "first": 4000, "until": 120000, "strike": {"radius": 45, "kill": 0.9}}
```

원 안의 개체가 `kill` 확률로 죽고, `burn_food`가 true면 원 안의 먹이도 사라진다.
세계는 반대편으로 이어져 있으므로 가장자리에 떨어지면 반대편까지 덮는다.

| 키 | 뜻 | 없으면 |
|---|---|---|
| `at` | 한 번 떨어지는 스텝 | (`every`를 쓸 때는 적지 않는다) |
| `every`, `first`, `until` | `first` 스텝째부터 `every` 스텝마다 되풀이한다. `until`을 적으면 그 스텝부터는 그친다 | `first`는 `every`, `until`은 끝없음 |
| `radius` | 원의 반지름(세계의 길이 단위. 기본 세계는 512 x 512) | 꼭 있어야 한다 |
| `kill` | 원 안의 개체 하나가 죽을 확률(0~1) | 1.0 |
| `burn_food` | 원 안의 먹이를 없앨지 | true |
| `x`, `y` | 떨어지는 자리 | null: 세계의 씨앗과 떨어지는 스텝으로 정해진다 |

- 자리를 적지 않아도 무작위가 아니다. 같은 씨앗의 세계에서 같은 스텝에 떨어지는 운석은 늘 같은 자리에 떨어진다. 누가 죽는지도 마찬가지다.
- 운석으로 죽은 개체는 사망 기록에 원인 3(`DEATH_DISASTER`)으로 남고, 남은 에너지는 먹이로 돌아가지 않는다. 사라진 에너지는 에너지 장부에 적힌다.
- 개체 수가 하한(`min_agents`, 기본 100) 아래로 떨어지면 세계가 살아 있는 개체의 복제본을 넣는다(재주입). 모두 죽었으면 무작위 유전체를 넣는다.
  재주입이 생겼다면 그 개체군은 그 재난을 스스로 버티지 못한 것이다.

## 값이 정해지는 규칙

1. 바탕 값에서 시작한다(실행 폴더의 `config.json`. 각본은 이 파일을 바꾸지 않는다).
2. 그 스텝까지 온 `set`을 시간 순서로 적용한다.
3. 그 스텝에 걸려 있는 배율(`scale`, `scale_to`, `amp`)을 모두 곱한다. 같은 설정에 여러 사건이 걸리면 배율끼리 곱해진다.
4. 정수 설정(`max_age` 등)은 곱한 뒤 반올림한다.
5. 결과가 설정 검사(`Config.validate()`)를 통과해야 한다. 실행을 만들 때 각본 전체를 미리 훑어 보고, 안 맞으면 시작하기 전에 거절한다.

어떤 스텝의 설정은 (바탕 설정, 각본, 그 스텝)만으로 정해진다. 그래서 중간에 꺼졌다 이어 돌린 결과가 끊지 않고 돌린 결과와 같다.
`set`과 `scale`의 시작·끝은 정확히 그 스텝에 반영되고, `scale_to`와 `amp`는 `update_every` 스텝마다 계단처럼 바뀐다.

## 건드릴 수 있는 설정

`kkumteul/config.py`의 설정 가운데 아래를 뺀 것은 모두 된다.

- 배열 크기를 정하는 것: `max_agents`, `food_cap`, `h_max`, `vision_sectors`, `n_spare_in`, `n_spare_out`, `world_w`, `world_h`, `cell`
- 세계의 조건이 아니라 실행·기록 방법인 것: `seed`, `threads`, `max_steps_per_sec`, `passive`, `init_agents`와 기록 주기·보존 설정(`stats_every`, `ckpt_every` 등)

자주 쓸 만한 것:

| 설정 | 기본값 | 바꾸면 |
|---|---|---|
| `food_inflow` | 9.0 | 스텝당 들어오는 먹이. 세계가 먹여 살릴 수 있는 개체 수를 정한다 |
| `n_patches`, `patch_sigma`, `patch_life` | 16, 18.0, 20000 | 먹이 자리의 수, 퍼짐, 자리가 옮겨 가는 주기(작게 하면 먹이 자리가 자주 옮겨 간다) |
| `food_uniform_frac` | 0.15 | 자리와 무관하게 아무 데나 나는 먹이의 비율 |
| `basal_cost` | 0.003 | 가만히 있어도 드는 에너지(추위) |
| `brain_cost`, `synapse_cost` | 0.006, 0.0006 | 뇌 유지비 |
| `thrust_cost`, `turn_cost` | 0.006, 0.0005 | 움직이는 비용(거친 땅) |
| `max_age` | 3000 | 최대 수명 |
| `e_repro`, `birth_cost`, `repro_min_age` | 4.0, 0.3, 200 | 번식 문턱, 번식 때 사라지는 에너지, 번식할 수 있는 나이 |
| `vision_range`, `vision_noise` | 16.0, 0.0 | 보이는 거리, 시야 잡음(안개) |
| `mutation_enabled` | true | `set`으로 false를 주면 그때부터 변이가 꺼진다 |
| `min_agents` | 100 | 재주입 하한. 0으로 두면 멸종할 수 있다 |

## 실행 폴더에 남는 것

| 자리 | 내용 |
|---|---|
| `scenario.json` | 받은 각본의 사본 |
| `run.json` | `scenario`(이름), `scenario_start`(시작 스텝), `scenario_sha256`(각본 내용의 해시) |
| `config.json` | 바탕 설정(각본이 바꾸기 전) |
| `stats.jsonl` | 줄마다 `scn`(각본이 건드리는 설정의 그때 값)과 `w_disaster`(그 구간에 운석으로 죽은 수), `deaths_disaster`(누적) |
| 체크포인트의 `.json` | 그 스텝에 실제로 쓰이던 설정(각본이 바꾼 값) |

## 내 각본 쓰기

1. 가까운 각본 파일을 베껴 이름과 설명을 고친다. 설명에는 무슨 일이 생기는지와 **무엇을 볼지**를 적는다.
2. `python3 -m kkumteul.scenario 내각본.json`으로 검사하고 값의 흐름을 본다.
3. 시작 스텝(0)에서는 바탕 값과 같게 두는 편이 좋다. 그래야 대조와 같은 곳에서 출발한다. 처음 얼마 동안은 아무 일도 없게 두면 "일이 생기기 전"도 같이 볼 수 있다.
4. `python3 -m kkumteul.experiment ...`로 대조와 함께 돌린다. 재주입이 많이 생기면 너무 가혹한 것이다. 배율을 줄이거나 기간을 줄여 본다.
5. 재미있는 결과가 나왔으면 각본 파일을 `scenarios/`에, 실험 카드를 `experiments/`에 올린다(`experiments/README.md`).

각본으로 적을 수 없는 조건(예: 세계의 한쪽만 추워진다, 먹이 자리가 한 방향으로 흘러간다)은 코드를 고쳐야 한다. `CONTRIBUTING.md`의 "코드를 고칠 때"를 본다.
