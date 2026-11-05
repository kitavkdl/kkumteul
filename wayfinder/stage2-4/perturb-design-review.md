# 노드 섭동 학습 규칙 — 설계 검토 (2026-11-05 03:05 ET, 구현 전)

배경: `results.md` "네 번째 시도". 지금 규칙(보상 조절 헤브)으로는 손으로 만든 개체도 가려 먹기를 못 배운다.
계획: `perturb_sigma`(기본 0 = 지금 규칙). >0이면 은닉 pre-activation에 noise = σ·N(0,1)
(`rng.normal(seed, S_PERTURB=18, ids[i], step*h_max+k)`)를 더하고, 출력 흔적을 `epost = lam*epost + noise`로 쓴다.
고칠 파일: config.py, rng.py(맨 뒤 18), stages.py(F_PERTURB, 커널 인자 seed·step·ids), world.py(호출), tests.
검토는 맥락 없는 서브에이전트 둘(sonnet). 규칙 축과 명세 축을 따로 보고 순위를 섞지 않았다.

## 명세 축 (목표를 이루는가)
| # | 지적 | 처리 |
|---|---|---|
| 1 치명 | α 부호가 뉴런마다 제각각이다(α_init·N(0,1)). α<0 뉴런은 보상을 내리는 쪽으로 배운다. 손 개체는 α>0이라 이 문제가 가려진다 | **반영 예정**: 섭동 모드 갱신에 sign(α[k])를 곱한다 |
| 2 치명 | tanh 앞에 넣은 noise는 포화한 뉴런에서 행동을 거의 안 바꾸는데, 흔적에는 noise가 다 들어간다 | **반영 예정**: epost에 `tanh(pre+noise) − tanh(pre)`(실제로 바뀐 양)를 쓴다 |
| 3 치명 | epre_h(noise 섞인 h)와 epost(noise)의 곱으로 Hrec[k,k]에 σ²·mod 편향이 쌓인다 | **반영 예정**: 섭동 모드에서 epre_h는 noise 없는 h로 쌓는다 |
| 4 심각 | 흔적 lam 0.8은 약 5스텝이다. 먹이까지 수십 스텝이면 대부분의 noise가 보상과 이어지지 않는다 | **먼저 잰다**: 손 개체로 σ × trace_decay(0.8/0.95) 격자 |
| 5 심각 | 먹은 뒤 약 20스텝 동안 mod<0이 이어지는데, 그동안 mod·x·noise는 잡음 랜덤워크다 | 평균 0이라 치우침은 없다. 격자에서 잡음이 크면 "|mod| 문턱 이상일 때만 갱신"을 더한다 |
| 6 심각 | 잠잘 때 |Hin|<0.3은 스텝마다 1%씩 지워진다. 손 개체의 "Hin 0.2면 뒤집힘"이 이 구간이다 | 판정 실험에서는 sleep_forget=0도 함께 본다 |
| 7 | 진화한 α(sd 0.05)로는 α·Hin이 0.15를 못 넘어 Win의 부호를 못 뒤집는다 | 손 개체로 판정하고, 진화 세계에서는 α가 커지는지 본다(추세는 증거가 아님) |
| - | 부호 자체(Δ ∝ mod·x·δ)와 한 스텝 정렬은 맞다 | — |

## 규칙 축 (레포 규칙·코드 냄새)
치명적인 것은 없다.
1. 해시 불변 시험이 비어 있을 수 있다. HEAD에서 stage2 몇백 스텝의 `state_hash()`를 뽑아 상수로 박는다. σ=0일 때는 기존 줄(stages.py:312 tanh, :361 epost)을 그대로 타게 분기한다(+0.0으로도 비트가 달라질 수 있다).
2. epost는 저장되는 평생 상태다. 켜는 순간 옛 흔적이 섞인다. 켜기 전후 비교는 같은 체크포인트에서 갈라 한다.
3. 잠든 개체에도 noise가 들어간다. 의도를 정해 주석으로 남긴다(제안: 잠든 동안은 noise 0).
4. stages.py import에 `normal, S_PERTURB`를 더한다. seed는 prange 밖에서 `isp[I_SEED]`, ids는 래퍼에서 풀어 넘긴다.
5. 옛 run.json은 기본값 0.0으로 읽힌다. 새 run.json은 옛 코드에서 "모르는 설정"이 된다(어쩔 수 없음).
6. config 검증 목록(config.py:243)에 넣는다. preset에는 넣지 않는다. `docs/stages-2-4.md`에 한 줄 더한다.
7. σ>0 장부 시험과 재현 시험을 `test_stage_world_lives_conserves_energy_and_repeats` 꼴로 더한다.
