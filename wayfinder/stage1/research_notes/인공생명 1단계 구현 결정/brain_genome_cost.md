# 뇌(신경 제어기) 구조 · 유전체 · 변이 · 생애 학습 · 뇌 에너지 비용 — 1단계 구현용 조사 노트

조사일 2026-11-04. 표기 규칙: **[직접]** = 논문 본문·소스 코드를 이번에 직접 열어 확인, **[요약]** = 검색 요약이나 초록으로만 확인(본문 못 엶), **[실측]** = 이번에 직접 돌려 잰 값, **[기억]** = 열어 확인하지 못해 Gaps에만 적은 내용. 수치 인용 전 [요약]·[기억]은 다시 확인할 것.

## 1. 구조 선택지와 증거 — 비에피소드 자연선택 세계에서 실제로 쓰인 뇌와 크기, 나온 행동

### Takeaway
자연선택만으로 도는 세계들은 예외 없이 **작은 뇌**(은닉 4~64 유닛, 파라미터 수천 개 수준)로 시작했고, 가변 구조(NEAT류)를 쓰는 곳도 배치 계산 때는 결국 "최대 크기로 채운 조밀 행렬 + 빈칸 표시"로 바꿔 계산한다. 따라서 1단계 기본값은 "최대 크기 고정 배열 + 마스크"인 단순 순환망이 증거와 가장 잘 맞는다.

### Cited Findings
- **Polyworld(1994) 뇌**: 입력 = 시각(R,G,B 색마다 뉴런 수를 유전자가 정함, "typically range between 1 and 16 neurons per color") + 자기 체력(health-energy) + 난수. 출력 = 7개 행동(eat, mate, fight, move, turn, focus, light). 내부 뉴런 그룹 수 1~5, 그룹마다 흥분성·억제성 뉴런 각 1~16개. 뉴런 식 `x_i = Σ_j a_j s_ij`, `a_i(t+1) = 1/(1+e^{-α x_i})`. [직접] — [Yaeger 1994](https://shinyverse.org/al4ai/papers/Yaeger.Polyworld.pdf)
- **Polyworld 현재 기본 설정 파일**: 내부 뉴런 그룹 0~20(legacy 5), 그룹당 흥분성 0~16·억제성 0~16, 색당 시각 뉴런 1~16, 연결 밀도 0.0~1.0, 뉴런 모델 기본 `F`(firing rate; 그 밖에 Tau/Gain, Spiking). [직접] — [polyworld etc/worldfile.wfs](https://github.com/polyworld/polyworld/blob/master/etc/worldfile.wfs)
- **Polyworld 2009 실험 규모**: 30,000 스텝(약 400세대), 개체 90~300, 내부 그룹 0~5·그룹당 흥분성 16 + 억제성 16까지, **최대 217 뉴런 · 45,584 시냅스**. [직접(WebFetch 요약 모델 경유)] — [Yaeger 2009, PMC2801533](https://pmc.ncbi.nlm.nih.gov/articles/PMC2801533)
- **Polyworld에서 나온 행동**: "frenetic joggers"(전속 직진하며 항상 먹고 짝짓기 시도), "indolent cannibals"(부모·자식 곁을 안 떠나며 서로 짝짓고 싸우고 죽은 개체를 먹음), "edge runners"(세계 가장자리를 따라 달림). indolent cannibals는 "부모가 자기 에너지를 자식에게 넘겨야 한다"는 규칙이 생기기 전에 가장 흔했다 — 공짜 에너지원을 착취한 것. [직접] — [Yaeger 1994](https://shinyverse.org/al4ai/papers/Yaeger.Polyworld.pdf)
- **Inria eco-evo(Hamon 등 2023)**: 15×15 관측 → conv 2단(3×3, stride 2, 특징맵 4개 → 8개) → 이전 행동·먹었는지 여부와 이어붙임 → **LSTM 은닉 크기 4** → dense 8(tanh) → 5개 이산 행동(정지·상하좌우, softmax temperature 1/50). **총 2,445 파라미터**. 격자 400×200, 시작 330마리, 최대 1,000마리, 1,000,000 스텝을 GPU 1대에서 약 20분. 나온 행동: 자원·개체수의 포식자-피식자식 진동, 초기에는 많이 움직이는 개체가 우세하다가 덜 움직이는 "sustainable foragers"와 "opportunistic travelers"가 공존. [직접(WebFetch 요약 모델 경유)] — [arXiv 2302.09334v3](https://arxiv.org/html/2302.09334v3)
- **JaxLife 뇌(소스)**: agent/bot/terrain 인코더(Dense, 크기 `HSIZE`) → self-attention(4 heads) + cross-attention(4 heads) → `LSTMCell(HSIZE)` → 출력 `6 + MESSAGE_SIZE*2 + 5`개. 기본 `--hsize 64`, `MESSAGE_SIZE 16`(→ 출력 43개), `--num-agents 128`. [직접] — [JaxLife lstm_brain.py](https://github.com/luchris429/JaxLife/blob/main/src/agent_brains/lstm_brain.py), [main.py](https://github.com/luchris429/JaxLife/blob/main/src/main.py)
- **Biosim4**: 내부 뉴런 최대 5개(`maxNumberNeurons = 5`), 유전체 = 유전자 24개(`genomeInitialLengthMin/Max = 24`, 유전자 하나가 연결 하나), `genomeMaxLength = 300`, 개체 3,000, 세대당 300 스텝. **세대제 + 생존 기준(challenge) 선택**이라 연속 자연선택 세계가 아니고, 에너지 개념이 설정 파일에 없다. [직접] — [biosim4.ini](https://github.com/davidrmiller/biosim4/blob/main/biosim4.ini)
- **NEAT(2002)**: 은닉 노드 0개(입력이 출력에 바로 연결)에서 시작해 구조를 조금씩 늘린다. XOR 해답은 평균 은닉 2.35개·활성 연결 7.48개. 저자 설명: 노드·연결을 더하면 처음에는 적합도가 보통 떨어져 새 구조가 한 세대도 못 버티므로 **종 분화(speciation)로 보호**해야 한다. [직접] — [Stanley & Miikkulainen 2002](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
- **TensorNEAT(2024)의 가변 구조 배치 방법**: 노드 텐서 `[P, |N|max, len(n)]`, 연결 텐서 `[P, |C|max, len(c)]`를 **NaN으로 채워** 같은 모양으로 만든다. 실험값 `max_nodes = 50`, `max_conns = 100`. 추론 전에 위상 정렬 순서 `N_order`와 **`|N|max × |N|max` 확장 연결 행렬 `C_exp`**(없는 연결은 NaN)로 한 번 변환한 뒤 노드 순서대로 계산. 논문이 자세히 다루는 것은 feedforward이고 recurrent는 언급만 한다. [직접(WebFetch 요약 모델 경유)] — [arXiv 2404.01817](https://arxiv.org/html/2404.01817)
- **TensorNEAT의 CPU 성능**: 개체 10,000·100세대 Brax 과제에서 NEAT-Python 대비 GPU(RTX 4090)는 59.8~544배 빠르지만, **같은 CPU(AMD EPYC 7543)에서는 1.07~9.40배**뿐(Swimmer 2.88×, Hopper 1.07×, Halfcheetah 9.40×). [직접(요약 모델 경유)] — [arXiv 2404.01817](https://arxiv.org/html/2404.01817)
- **Heesom-Green 등(ALIFE 2025)**: NEAT로 feedforward ANN을 진화(초기: 은닉 1개 완전 연결, 크기 N_S = 연결 수 + 비입력 노드 수 = 254). 입력 243(9×9×3 RGB), 출력 5. 재귀 연결이 없는 것을 저자 스스로 한계로 적었다. [직접] — [arXiv 2511.20018](https://arxiv.org/abs/2511.20018)

### Inferences
- "가변 구조냐 고정 구조냐"는 배치 계산 관점에서는 차이가 작다. TensorNEAT도 결국 개체마다 `Nmax×Nmax` 조밀 행렬을 만들어 계산하므로, 처음부터 `[N, Hmax, Hmax]` 가중치 + 마스크로 두는 것과 계산 모양이 같다. NEAT의 고유 가치는 혁신 번호(교차용)와 종 분화인데, 무성 생식·자연선택만 쓰는 1단계에는 둘 다 필요 없다.
- 비교 대상 중 연속 자연선택 세계(Polyworld, Inria, JaxLife)는 모두 순환 상태를 가진 뇌(순환 연결 또는 LSTM)를 썼다. 시야가 제한된 세계에서는 기억 없이는 "방금 본 먹이 쪽으로 계속 가기"도 어렵기 때문으로 보인다. 1단계 기본값 후보: Elman형 `h' = tanh(W_in x + W_rec h + b)`, `y = tanh(W_out h')` — LSTM/GRU보다 게이트 수가 없어 파라미터가 1/3~1/4이고 변이에 덜 민감하며, 나중에 Hebbian 항을 붙이기 쉽다(5번 참조).
- Inria가 은닉 4·2,445 파라미터로 집단 수준 행동까지 냈으므로, 1단계 은닉 크기는 8~32면 충분하고 최대치(Hmax)만 넉넉히(64) 잡아 두면 된다.

### Gaps
- **Bibites 뇌 구조**: 위키(the-bibites.fandom.com `Brain`, `Nodes`, `Synapses`)가 HTTP 402로 열리지 않았다. 검색 요약으로만 확인한 것: 입력·출력 뉴런은 고정이고 은닉 뉴런만 변이, 시냅스는 enabled/disabled 두 상태, 입력을 곱으로 모으는 특수 노드 종류가 있음. 출력 목록(알 낳기가 뇌 출력인지)·기본 뇌는 확인 못 함. [요약] — [Brain](https://the-bibites.fandom.com/wiki/Brain), [Synapses](https://the-bibites.fandom.com/wiki/Synapses), [Nodes](https://the-bibites.fandom.com/wiki/Nodes)
- **CTRNN(Beer)**: 이번에 원문을 열지 못했다. [기억] 식은 `τ_i dy_i/dt = −y_i + Σ_j w_ji σ(y_j + θ_j) + I_i`이고 유전체는 뉴런마다 τ, θ와 가중치. 오일러 적분 `y += dt/τ · (...)`으로 numpy 배치화는 Elman과 같은 비용(행렬곱 1회). 연속 시간 물리와 잘 맞는 장점이 있으나 비에피소드 자연선택 세계에서 쓴 사례는 이번 조사에서 찾지 못했다.
- **Geb(Channon), Avida류, HyperNEAT·발생적(indirect) 인코딩**: 조사하지 못했다. 간접 인코딩이 수천 개체·CPU 조건의 연속 세계에서 쓰인 근거는 찾지 못함.
- Biosim4 유전자 비트 배치(32비트: source 종류 1 + id 7, sink 종류 1 + id 7, 가중치 16비트)와 가중치 나눗수는 [기억]이며 README에 없어서 확인 못 함(소스 `genome.cpp` 필요).
- JaxLife의 총 파라미터 수는 논문·소스 어디에도 숫자로 적혀 있지 않다.

## 2. 뇌 크기를 위아래로 진화시키면서 배치 계산을 유지하는 법 — CPU 시간·메모리

### Takeaway
개체마다 다른 가중치를 `[N, Hmax, Hmax]` 배열에 두고 한 번에 곱하는 방식은 N=5,000·H=64에서도 **스텝당 1.4~2.4 ms, 메모리 116 MB**(Apple M3 실측)라서 뇌 계산은 병목이 아니다. 크기 진화는 "활성 뉴런 수 유전자(앞에서부터 n_h개만 사용)"가 가장 싸고, numba 루프에서는 작은 뇌가 실제로 계산 시간도 덜 쓴다.

### Cited Findings
- **[실측] 배치 순환망 한 스텝**(개체마다 다른 가중치, float32, 입력 I=24, 출력 O=6, `h' = tanh(W_in x + W_rec h + b)`, `y = tanh(W_out h')`). 기계: Apple M3 8코어, numpy 2.2.6(Accelerate BLAS), numba 0.67.0. 최솟값 기준(ms). 스크립트: (세션 임시 폴더)(임시 폴더라 사라질 수 있어 아래에 핵심 코드를 적음). **목표 기계(N5 Air, 12코어 AVX2)에서 다시 재야 한다.**

  | N | H | 가중치 메모리(MB) | `np.matmul(W_rec, h)`만 | `einsum('nij,nj->ni')`만 | numpy 전체 스텝 | numpy 전체 스텝 + 뉴런 마스크 | numba 8스레드 전체 | numba, 뉴런 절반만 활성 | numba 1스레드 |
  |---|---|---|---|---|---|---|---|---|---|
  | 1000 | 16 | 2.9 | 0.033 | 0.040 | 0.120 | 0.124 | 0.101 | 0.075 | 0.299 |
  | 1000 | 32 | 7.7 | 0.064 | 0.104 | 0.187 | 0.194 | 0.159 | 0.108 | 0.557 |
  | 1000 | 64 | 23.2 | 0.203 | 0.329 | 0.396 | 0.412 | 0.306 | 0.191 | 1.290 |
  | 1000 | 128 | 77.6 | 1.084 | 1.272 | 1.531 | 1.558 | 0.882 | 0.475 | 3.012 |
  | 2000 | 64 | 46.4 | 0.409 | 0.653 | 0.843 | 0.920 | — | — | — |
  | 5000 | 16 | 14.3 | 0.159 | 0.194 | 0.583 | 0.598 | 0.332 | 0.218 | 1.465 |
  | 5000 | 32 | 38.5 | 0.330 | 0.551 | 0.936 | 0.966 | 0.600 | 0.424 | 2.836 |
  | 5000 | 64 | 116.0 | 1.213 | 1.749 | 2.378 | 2.439 | 1.373 | 1.004 | 6.463 |
  | 5000 | 128 | 388.2 | 5.441 | 6.378 | 7.646 | 7.790 | 4.695 | 2.600 | 15.577 |

  — 출처: 이번 세션 직접 측정(위 스크립트)
- **[실측] 변이 비용**: 자식 100마리 × 파라미터 6,080개(H=64, I=24, O=6: `H² + H·I + O·H + H`)에 "10% 확률로 N(0, 0.1) 더하기"를 numpy로 하면 4.70 ms. — 출처: 같은 측정
- **핵심 코드(재측정용)**:
  ```python
  # numpy: Wr [N,H,H], Wi [N,H,I], Wo [N,O,H], b [N,H], h [N,H], x [N,I], mask [N,H] (0/1)
  hn = np.tanh(np.matmul(Wi, x[:, :, None])[:, :, 0]
               + np.matmul(Wr, (h * mask)[:, :, None])[:, :, 0] + b) * mask
  y = np.tanh(np.matmul(Wo, hn[:, :, None])[:, :, 0])

  # numba: 개체마다 활성 은닉 수 nh[n] (앞에서부터 k개만 쓰는 prefix 방식)
  @njit(parallel=True, fastmath=True)
  def step_nb(Wi, Wr, Wo, b, h, x, nh, out_h, out_y):
      for n in prange(Wr.shape[0]):
          k = nh[n]
          for i in range(k):
              s = b[n, i]
              for j in range(Wi.shape[2]): s += Wi[n, i, j] * x[n, j]
              for j in range(k):           s += Wr[n, i, j] * h[n, j]
              out_h[n, i] = np.tanh(s)
          for o in range(Wo.shape[1]):
              s = 0.0
              for j in range(k): s += Wo[n, o, j] * out_h[n, j]
              out_y[n, o] = np.tanh(s)
  ```
- **TensorNEAT의 채움 방식**: 최대 노드 50·최대 연결 100으로 고정하고 빈자리는 NaN, 계산은 `|N|max × |N|max` 확장 행렬로 한다 — 가변 구조도 배치에서는 "최대 크기 조밀 행렬"이 된다. [직접(요약 모델 경유)] — [arXiv 2404.01817](https://arxiv.org/html/2404.01817)
- **JaxLife**: 개체 배열 크기는 `NUM_AGENTS`로 고정하고 `alive` 불리언으로 살아 있는 칸을 표시, 죽은 칸(`~alive`)에 새 자식을 채운다(`world.py`의 `jnp.logical_and(~agent_states.alive, ...)`). [직접] — [JaxLife world.py](https://github.com/luchris429/JaxLife/blob/main/src/world/world.py)
- **Polyworld의 크기 진화 방식**: 뉴런 수를 직접 유전자로 둔다(색당 시각 뉴런 수, 내부 그룹 수, 그룹당 흥분성·억제성 뉴런 수, 그룹 사이 연결 밀도). 즉 "크기·밀도 숫자 유전자"로 위아래 모두 변한다. [직접] — [Yaeger 1994](https://shinyverse.org/al4ai/papers/Yaeger.Polyworld.pdf)

### Inferences
- **numpy 마스크 방식은 메모리·시간이 뇌 크기와 무관하게 Hmax 기준으로 든다**(표에서 마스크 유무 차이 3% 이내). "작은 뇌는 계산도 싸다"를 실제로 얻으려면 numba에서 `nh[n]`까지만 도는 prefix 방식이 맞다(H=64·N=5000에서 절반 활성이면 1.37 → 1.00 ms). 다만 어느 쪽이든 절대 시간이 작아서, 1단계에서는 구현이 단순한 쪽을 골라도 된다.
- 크기 진화 표현 세 가지 비교:
  1. **활성 뉴런 수 유전자 `n_h`(prefix)** — 유전자 1개, 뉴런 추가/삭제 변이는 `n_h ± 1`. 새로 켜지는 뉴런의 나가는 가중치(`W_rec[:, new]`, `W_out[:, new]`)를 0으로 두면 행동이 바뀌지 않는 중립 변이가 된다. 단점: 중간 뉴런만 골라 지울 수 없다(항상 마지막 것이 꺼짐).
  2. **뉴런별 마스크 `[N, Hmax]`** — 아무 뉴런이나 켜고 끌 수 있지만 numba에서도 조건 분기가 들어가 이득이 줄고, 꺼진 뉴런의 가중치가 유전체에 남아 "숨은 유전 변이"가 된다(장점일 수도 있음).
  3. **연결 마스크 `[N, Hmax, Hmax]`** — 시냅스 수 비용을 정확히 매길 수 있지만 메모리가 2배(불리언이면 1.25배), 변이 연산이 복잡. Polyworld의 "연결 밀도" 유전자처럼 숫자 하나로 대신하는 방법도 있다.
- 메모리는 float32 기준 H=64·N=5000이 116 MB, H=128이 388 MB라 7 GB 여유에서 문제없다. 나중에 가소성용 `alpha`·`Hebb` 배열(각각 `W_rec`와 같은 크기)을 더해도 H=64면 +약 160 MB.
- `np.matmul`은 3차원 스택에서 개체별 작은 행렬곱을 반복하는 방식이라 코어 수 이득이 작을 것으로 보이고(확인 안 함), 서버의 OpenBLAS에서는 작은 행렬 호출 오버헤드가 Accelerate와 다를 수 있다. numba `prange`는 8스레드에서 1스레드 대비 3.5~4.7배였다(표). 12코어 서버에서는 numba 쪽이 유리할 가능성이 높다.
- 초당 30스텝을 목표로 하면 스텝 예산이 33 ms인데, 뇌는 최악(N=5000, H=128, numpy)에도 7.6 ms다. 병목은 감각(이웃 찾기)·물리 쪽일 것이다.

### Gaps
- 목표 기계(N5 Air, x86 AVX2, 12코어)에서의 실측이 없다. 위 표는 Apple M3(ARM, Accelerate) 값이다.
- 희소 연결(CSR 등)로 개체별 다른 구조를 배치 계산하는 방식의 CPU 실측 자료는 찾지 못했다. 연결 밀도가 매우 낮을 때만 이득일 것으로 추정되나 근거 없음.
- EvoJAX/evosax 문서의 배치 신경진화 관행은 조사하지 못했다(JAX `vmap`으로 개체별 파라미터 pytree를 쌓는 방식이라는 것은 [기억]).

## 3. 변이 — 실제 쓰이는 비율·시그마, 자기 적응 변이율, 구조 변이, 교차와 문제점

### Takeaway
연속 세계 선례는 "출생 때 모든 가중치에 작은 가우시안(σ = 0.01~0.02)"이라는 가장 단순한 방식을 쓰고, Polyworld만 변이율 자체를 유전자로 두되 상·하한으로 묶는다. 구조 변이는 가중치 변이보다 훨씬 드물게(NEAT: 노드 0.03, 연결 0.05) 두며, 크기 비용이 없으면 뇌가 선택과 무관하게 불어난다.

### Cited Findings
- **JaxLife**: 자식 = 부모의 모든 파라미터 + `normal(0, 1) * MUTATION_STD`, 기본 `--mut-std 0.01`. 무성 생식, 교차 없음. [직접] — [JaxLife agent.py](https://github.com/luchris429/JaxLife/blob/main/src/world/agent.py), [main.py](https://github.com/luchris429/JaxLife/blob/main/src/main.py)
- **Inria eco-evo**: 자식 가중치 = 부모 + N(0, σ), **σ = 0.02**(표에는 "variance"라고 적혀 식과 표기가 어긋남). 무성·국소 생식(자식이 부모 칸에 나타남). [직접(요약 모델 경유)] — [arXiv 2302.09334v3](https://arxiv.org/html/2302.09334v3)
- **Polyworld — 변이율이 유전자**: 1994 논문은 변이율 유전자 범위 0.01~0.1, 교차점 수 2~8로 "reasonable limits" 안에 묶었다고 적는다. 현재 설정 기본값은 `MinMutationRate 0.01`/`MaxMutationRate 0.05`(비트 단위일 때; 바이트 단위면 0.08/0.4), `MinCrossoverPoints 2`/`MaxCrossoverPoints 100`(legacy 8), 바이트 변이의 표준편차는 `2^MutationStdevPower`(범위 0.0~6.0). 유성 생식(교차 + 변이). [직접] — [Yaeger 1994](https://shinyverse.org/al4ai/papers/Yaeger.Polyworld.pdf), [worldfile.wfs](https://github.com/polyworld/polyworld/blob/master/etc/worldfile.wfs), [Genome.cc](https://github.com/polyworld/polyworld/blob/master/src/library/genome/Genome.cc)
- **Polyworld — 다양성 상실 사례**: 장벽도 miscegenation function(유전적으로 먼 개체끼리 짝짓기 억제)도 없는 세계에서는 초기 다양성이 금방 섞여 한 종(frenetic joggers)이 끝까지 지배했다. [직접] — [Yaeger 1994](https://shinyverse.org/al4ai/papers/Yaeger.Polyworld.pdf)
- **NEAT 설정값**: 개체 150(가장 어려운 과제만 1,000). 유전체의 80%가 가중치 변이를 받고, 그때 각 가중치는 90% 확률로 균등 섭동·10% 확률로 새 난수. **노드 추가 0.03, 연결 추가 0.05**(큰 개체군에서는 0.3). 자식의 25%는 교차 없이 변이만. 부모 한쪽에서 꺼진 유전자는 75% 확률로 꺼진 채 유전. 종간 교배율 0.001. 15세대 동안 최고 적합도가 안 오르는 종은 번식 금지. 호환 거리 `δ = c1·E/N + c2·D/N + c3·W̄`, c1 = c2 = 1.0, c3 = 0.4, δt = 3.0. 활성 함수 `1/(1+e^{-4.9x})`. 저자: "links need to be added significantly more often than nodes". [직접] — [Stanley & Miikkulainen 2002](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
- **NEAT — 구조 변이의 문제**: 작은 구조가 큰 구조보다 빨리 최적화되고, 노드·연결 추가는 처음에 적합도를 떨어뜨리므로 새 구조는 보호(종 분화) 없이는 살아남기 어렵다. 교차는 혁신 번호(historical marking)로 유전자를 맞춰야 서로 다른 구조끼리 가능하다. [직접] — [Stanley & Miikkulainen 2002](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
- **Biosim4**: `pointMutationRate = 0.001`(유전자당 1비트 뒤집힘 확률), 유전자 삽입·삭제율은 0.0("Ignored for now"), `sexualReproduction = true`. [직접] — [biosim4.ini](https://github.com/davidrmiller/biosim4/blob/main/biosim4.ini)
- **Heesom-Green 등 2025(NEAT)**: 가중치 변이율 0.1, 노드 추가/삭제 0.5/0.0, 연결 추가/삭제 0.5/0.5, 초기 가중치·편향 N(0, 1). 노드 삭제율을 0으로 둔 이유는 "끊어진 노드는 어차피 제거되므로". [직접] — [arXiv 2511.20018](https://arxiv.org/abs/2511.20018)
- **비용이 없으면 뇌가 불어난다(bloat)**: 같은 논문에서 에너지 비용이 없으면(NEC) 환경과 무관하게 모든 조건에서 ANN 크기가 세대에 따라 커졌다(세대-크기 Spearman 0.45~0.60, 모두 유의). 저자: "absence of energy constraints likely allowed unrestricted ANN growth". [직접] — [arXiv 2511.20018](https://arxiv.org/abs/2511.20018)
- **선택이 없어도 복잡도는 오른다**: Polyworld에서 자연선택을 끈 "passive" 대조군(실제 실행과 같은 시점에 무작위 개체를 죽이고 무작위 둘을 교배)도 복잡도가 올라 약 t = 7,000에 자연선택 실행의 평균에 닿고, 끝에는 평균적으로 더 높아졌다. 유전체 비트가 약 50% 켜짐에 이르는 t ≈ 10,000에서 멈춘다. [직접(요약 모델 경유)] — [Yaeger 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2801533)
- **Bibites의 뇌 변이**: 출생 때 뇌 변이가 몇 번 일어날지를 유전자(위키 쪽에 따라 "Brain mutation Chance" 또는 "Average Brain Mutations" + "Brain Mutation Variance")가 정하고, 시냅스·뉴런마다 change/flip/toggle/add/remove 변이 확률 설정이 있다. 기본값은 확인 못 함. [요약] — [Bibites wiki: Brain](https://the-bibites.fandom.com/wiki/Brain), [Genes](https://the-bibites.fandom.com/wiki/Genes)
- **Bibites — 비용을 낮추면 생기는 문제**: 뉴런·시냅스 출생 비용과 유지비를 너무 낮추면 뇌가 크고 복잡해질 수 있지만, 쓸모없는 뉴런·시냅스가 쌓여 긴 시뮬레이션이 느려질 수 있다고 위키가 경고한다. [요약] — [Neuron Birth Cost](https://the-bibites.fandom.com/wiki/Neuron_Birth_Cost)
- **Najarro & Risi의 ES 설정(참고)**: 섭동 σ = 0.1(감쇠 0.999), 학습률 0.2(감쇠 0.995), 개체 200/500. 자연선택이 아니라 적합도 기반 ES이므로 σ를 그대로 가져올 수는 없다. [직접(요약 모델 경유)] — [arXiv 2007.02686](https://ar5iv.labs.arxiv.org/html/2007.02686)

### Inferences
- 1단계 기본값 제안(선례의 중간): 무성 생식, 출생 때 **모든 가중치에 N(0, σ)**, σ는 개체의 유전자로 두되 [0.005, 0.1] 같은 상·하한으로 묶고(Polyworld 방식) 로그 공간에서 변이(`σ' = σ·exp(τ·N(0,1))`). 시작값은 0.02(Inria)~0.01(JaxLife). 자기 적응 σ는 "대부분의 변이가 해롭다"는 이유로 하한 쪽으로 끌려가는 경향이 알려져 있으므로 하한이 0이면 진화가 멈출 수 있다 — 하한을 반드시 둔다(이 경향 자체는 이번에 출처를 확인하지 못한 일반론).
- 가중치 수가 수천 개일 때 "전부 섭동"은 한 번의 출생에 기대 변화량이 `σ·√P`로 커진다(P = 6,080, σ = 0.02면 가중치 벡터 거리 약 1.6). 뇌 최대 크기를 키울수록 σ를 낮추거나 "가중치당 변이 확률 p(예: 0.1) × 더 큰 σ" 방식(NEAT·Heesom-Green의 변이율 0.1과 같은 꼴)을 쓰는 편이 안전하다.
- 구조 변이(뉴런 수 ±1)는 가중치 변이보다 한두 자릿수 드물게(NEAT 0.03 수준) 두고, 추가는 반드시 **행동이 안 바뀌는 중립 변이**로 만든다(새 뉴런의 나가는 가중치 0). NEAT가 종 분화로 풀던 "새 구조는 처음에 손해" 문제를, 종 분화 없이 피하는 가장 싼 방법이다.
- 교차는 1단계에서 빼는 것이 맞다. 고정 구조 망끼리의 교차는 같은 기능이 다른 은닉 뉴런 순서로 표현되는 경쟁 관례(competing conventions) 문제가 있고, NEAT는 이를 혁신 번호로 푼다. 무성 생식 계통에서는 문제가 생기지 않는다. 유성 생식을 나중에 넣는다면 Polyworld의 miscegenation function처럼 "유전적으로 가까운 개체끼리만"이라는 제한이 필요하다.
- 크기 비용이 0이면 불어남(bloat)이 기본 동작이다(Heesom-Green NEC, Yaeger passive, Bibites 경고). 따라서 "뇌가 커졌다"를 지능의 증거로 읽으려면 비용이 있는 조건이어야 한다.

### Gaps
- 돌연변이 붕괴(mutational meltdown)를 연속 ALife 세계에서 정량적으로 보고한 1차 자료는 찾지 못했다.
- 자기 적응 변이율이 하한으로 수렴한다는 현상의 1차 출처(진화 전략 문헌)는 이번에 열지 못했다. [기억]
- NEAT 논문의 competing conventions 절(2.2절)은 이번에 그 쪽을 직접 읽지 않았다(읽은 범위: 3.2~4.2절). 개념 설명은 [기억].
- Bibites 변이 기본 확률값, Polyworld 유전체의 실제 비트 길이는 확인 못 함.

## 4. 초기화 — "쓸 만한 행동이 나오기 전에 초기 개체군이 죽는다" 문제를 어떻게 피하나

### Takeaway
선례의 해법은 세 가지다: (a) 개체 수가 하한 아래로 내려가면 시스템이 개체를 만들어 넣는다(Polyworld: 임시 적합도 상위 개체의 자식 또는 무작위 개체, JaxLife: 전멸하면 무작위로 재시작), (b) 번식을 뇌의 결정이 아닌 자동 규칙으로 둔다(Inria), (c) 초기 환경을 너그럽게 한다. Polyworld의 정의대로 "더는 만들어 넣지 않아도 출생만으로 유지되는 시점"을 자립의 기준으로 삼으면 된다.

### Cited Findings
- **Polyworld의 부트스트랩**: 출생만으로 수가 유지되는 상태를 Successful Behavior Strategy(SBS)라 부르고, **SBS가 나오기 전까지는 임시(ad hoc) 적합도 함수를 쓰는 steady-state GA로 돈다**. 개체 수가 하한 아래로 떨어지면 (1) 무작위 개체를 만들거나 (2) "N fittest" 표에서 둘을 골라 자식을 만들거나 (3) 드물게 역대 최고 개체를 그대로 되돌려 넣는다. 임시 적합도는 먹기·짝짓기·수명 다 채우기·에너지를 남기고 죽기·움직이기에 점수를 준다. SBS가 나온 뒤에는 "there is no fitness function except survival". [직접] — [Yaeger 1994](https://shinyverse.org/al4ai/papers/Yaeger.Polyworld.pdf)
- **Polyworld 기본 수치**: `MinAgents 90`, `InitAgents 180`, `MaxAgents 300`, `NumberFittest 30`, `MateWait 25`(짝짓기 뒤 대기 스텝). 추가로 `EnergyBasedPopulationControl`(기본 True)이 개체 수가 너무 적거나 많으면 **에너지 소비량에 배율을 곱해** 조절한다(agent.cc 주석: "if population is getting too low or too high, adjust energy consumption"). [직접] — [worldfile.wfs](https://github.com/polyworld/polyworld/blob/master/etc/worldfile.wfs), [agent.cc](https://github.com/polyworld/polyworld/blob/master/src/library/agent/agent.cc)
- **Polyworld는 성공한 실행만 보고한다**: "A simulation is considered 'successful' if and only if some number of species emerge with a SBS" — 관찰 보고는 성공한 시뮬레이션만 다룬다고 명시. 즉 실패하는 실행이 있었다. [직접] — [Yaeger 1994](https://shinyverse.org/al4ai/papers/Yaeger.Polyworld.pdf)
- **JaxLife**: 논문 "We reinitialize the population—with random networks—if all agents die". 소스에서는 `NUM_AGENTS 128` 칸 중 `MIN_AGENTS = NUM_AGENTS // 4 = 32`마리로 시작하고, `ONLY_RESET_WHEN_NO_AGENTS = True`면 살아 있는 수가 0일 때만 `MIN_AGENTS`마리를 무작위로 넣는다(False면 `MIN_AGENTS`에 못 미치는 만큼 계속 채움). 초기 에너지 32, 기본 유지비 `age/256 + 0.25`/스텝. [직접] — [arXiv 2409.00853](https://arxiv.org/html/2409.00853), [JaxLife world.py](https://github.com/luchris429/JaxLife/blob/main/src/world/world.py), [main.py](https://github.com/luchris429/JaxLife/blob/main/src/main.py)
- **Inria eco-evo**: 재주입 장치 없이 무작위 가중치 330마리로 시작. 번식이 자동(에너지가 E_min 위에서 T_repro = 140 스텝 유지되면 번식)이고, 죽음도 에너지가 E_min 아래로 T_death = 200 스텝 지속되거나 나이 650 초과일 때라 굶어도 200 스텝의 유예가 있다. 초기 에너지 = 최대 에너지 = 3, 스텝당 −0.025, 자원 하나 +1. 자원이 완전히 사라지지 않도록 자발적 재생 확률 c = 0.00005를 둔다. 안정성은 하이퍼파라미터 조정으로 얻었고 **일부 시드에서는 개체군이 붕괴**했다(요약 모델은 "5개 중 3개"라고 답했으나 본문 재확인 필요). [직접(요약 모델 경유)] — [arXiv 2302.09334v3](https://arxiv.org/html/2302.09334v3)
- **Polyworld의 첫 성공 전략은 뇌를 거의 안 쓴다**: frenetic joggers는 "ran straight ahead at full speed, always wanting to mate and always wanting to eat" — 환경이 충분히 너그러우면 상수 출력만으로 자립한다. [직접] — [Yaeger 1994](https://shinyverse.org/al4ai/papers/Yaeger.Polyworld.pdf)

### Inferences
- 1단계에서 권할 조합: **(1) 하한 재주입**(개체 수 < N_min이면 "최근 오래 산/많이 낳은 개체 표"에서 골라 변이한 복제본을 넣고, 표가 비었으면 무작위), **(2) 재주입 횟수를 기록**해 "최근 X 스텝 동안 재주입 0회"를 자립 판정으로 사용(Polyworld의 SBS 정의). 재주입이 있는 동안은 "자연선택만"이 아니라는 점을 로그에 남긴다.
- 무작위 뇌가 살아남으려면 "아무 일도 안 하는 출력"이 죽음이 아니어야 한다. 출력 편향(bias)의 초기값을 "앞으로 조금 가기 + 먹기 켜짐" 쪽으로 주는 것(손으로 심은 본능)은 Polyworld의 frenetic joggers가 저절로 찾은 해를 미리 주는 것과 같고, 이후 진화가 덮어쓸 수 있다. 가중치 초기 표준편차를 작게(예: 0.1) 두면 초기 행동이 편향에 의해 결정된다.
- Inria식 "유예 기간"(에너지가 0이 돼도 바로 죽지 않음)이나 초기 에너지를 최대로 채워 시작하는 것은 초기 사망 파동을 늦춰 준다. 단, Polyworld의 indolent cannibals 교훈대로 자식 에너지는 반드시 부모에게서 빠져나가야 한다(공짜 에너지 금지).
- Polyworld식 "개체 수에 따라 에너지 소비 배율 조절"은 전멸·폭증을 막는 부드러운 안전장치지만 선택압을 인위적으로 바꾼다. 1단계에서는 꺼 두고 먹이 재생량으로만 조절하는 편이 해석이 쉽다.

### Gaps
- Bibites의 기본 개체("basic bibite")가 손으로 심은 뇌를 갖는지는 위키를 열지 못해 확인하지 못했다.
- "쉬운 초기 환경에서 시작해 점차 어렵게"를 연속 세계에서 정량 비교한 자료는 찾지 못했다. 관련 근거는 6번의 arXiv 2609.15569(초록만 확인: 정적 환경에서 먼저 진화한 뒤 변하는 환경으로 옮긴 쪽이 더 큰 망을 가짐)뿐이다.
- Inria의 붕괴 시드 수는 본문 재확인 필요.

## 5. 생애 학습 — 1단계에 넣을지, 나중에 붙이려면 구조에 무엇이 있어야 하나

### Takeaway
증거는 "진화만으로 시작하고 가소성은 나중에"를 지지한다: 가소성은 파라미터를 2~5배로 늘리고(탐색이 느려짐), 익숙한 조건에서는 고정 가중치가 더 잘하며, 이득은 조건이 바뀔 때만 뚜렷하다. 단, 지금 배열 설계에 "시냅스별 가소성 계수·흔적(trace) 자리"와 "조절 신호용 출력 자리"만 비워 두면 나중에 다시 쓰지 않고 붙일 수 있다.

### Cited Findings
- **Polyworld의 Hebb 규칙(1994)**: `s_ij(t+1) = s_ij(t) + η_ckl · (a_i(t+1) − 0.5) · (a_j(t) − 0.5)`. η는 연결 종류(e-e, e-i, i-i, i-e)와 그룹 쌍(l → k)마다 유전자로 정해지고, 선택적으로 곱셈 감쇠 항을 둘 수 있다. 편향도 같은 규칙으로 갱신(편향 학습률 보통 0.0~0.2). 현재 기본값 `MaxLearningRate 0.1`, `MinLearningRate 0.0`. 유전자는 가중치가 아니라 구조와 학습률만 담는다 — 가중치는 생애 동안 Hebb 학습으로 형성. [직접] — [Yaeger 1994](https://shinyverse.org/al4ai/papers/Yaeger.Polyworld.pdf), [worldfile.wfs](https://github.com/polyworld/polyworld/blob/master/etc/worldfile.wfs)
- **Najarro & Risi 2020(ABCD 규칙)**: `Δw_ij = η_w · (A_w·o_i·o_j + B_w·o_i + C_w·o_j + D_w)` — **시냅스마다 계수 5개**(A, B, C, D, η). 가중치는 에피소드마다 U[−0.1, 0.1]로 무작위 초기화하고 계수만 진화. 사족 보행: 층 [128, 64, 8], 시냅스 12,288개 → Hebbian 계수 61,440개. 결과(이동 거리, 100회): 손상 없음 Hebbian 1051 ± 113 vs 고정 가중치 1604 ± 171, 학습 때 본 손상 1019 ± 116 vs 1431 ± 54, **처음 보는 손상 452 ± 95 vs 68 ± 56**. 가중치는 약 30~80 스텝 만에 수렴. 저자 한계: 본 조건에서는 고정 가중치가 더 낫고, 파라미터 공간이 커져 탐색 수렴이 느려진다. 초기 가중치까지 같이 진화시키는 것은 효율을 높이지 못했다. [직접(요약 모델 경유)] — [arXiv 2007.02686](https://ar5iv.labs.arxiv.org/html/2007.02686)
- **Miconi 등 2018(differentiable plasticity)**: `x_j(t) = tanh( Σ_i [ w_ij + α_ij · Hebb_ij(t) ] · x_i(t−1) )`, 흔적 갱신 `Hebb_ij(t+1) = η · x_i(t−1) · x_j(t) + (1−η) · Hebb_ij(t)`, Oja 변형 `Hebb_ij(t+1) = Hebb_ij(t) + η · x_j(t) · (x_i(t−1) − x_j(t) · Hebb_ij(t))`. `w`, `α`는 연결마다, `η`는 전체 공유 스칼라 하나, `Hebb`는 생애 시작 때 0. **파라미터 수는 약 2배**(연결당 α 하나 추가). 저자: 단순 RNN 코드에 4줄 이내 추가. 연결별 α가 공유 α보다 성능이 좋았다. [직접(요약 모델 경유)] — [arXiv 1804.02464](https://ar5iv.labs.arxiv.org/html/1804.02464)
- **Soltoggio 등 2008(neuromodulation)**: T-maze 학습 문제에서 조절 뉴런(modulatory neuron)을 허용한 진화가 금지한 경우보다 학습을 더 잘했고, 조절 뉴런은 진화 과정에서 저절로 나타났다. 박사 논문은 제어 문제가 복잡할수록 이 이점이 커진다고 보고. [요약] — [EPFL Infoscience 121548](https://infoscience.epfl.ch/record/121548?ln=en)
- **Hinton & Nowlan 1987(Baldwin effect)**: 유전자 값 1(연결 있음)/0(없음)/?(학습이 정하는 스위치), 연결 20개, 개체 1,000. 2017년 재분석: 학습이 있으면 50세대 뒤 평균 적합도 약 11.6, 학습이 없으면 기대 적합도 약 1. 같은 재분석의 단서: 원 실험이 보여 준 것은 "학습 능력이 쉽게 선택된다"는 것이지 완전한 Baldwin effect의 증명은 아니다. [요약] — [Hinton 페이지](https://www.cs.toronto.edu/~hinton/absps/evolution.htm), [arXiv 1702.08411](https://arxiv.org/pdf/1702.08411)
- **Heesom-Green 등 2025**: 생애 학습(PPO 1,000 에피소드)으로 바뀐 가중치는 자식에게 유전하지 않는 Baldwin 방식. 진화는 구조·초기 가중치·RL 하이퍼파라미터만 다룬다. [직접] — [arXiv 2511.20018](https://arxiv.org/abs/2511.20018)

### Inferences
- **1단계는 진화만**이 맞다. 근거: (1) 가소성은 유전체를 2배(Miconi)~5배(Najarro)로 키워 자연선택의 탐색을 늦춘다, (2) Najarro 결과에서 조건이 안 바뀌면 고정 가중치가 50% 더 잘한다 — 1단계 세계가 단순·정적이면 가소성은 비용만 되고 선택되지 않을 것, (3) Polyworld처럼 가중치를 전부 생애 학습에 맡기면 태어난 직후 무능한 기간이 생겨 부트스트랩이 더 어려워진다.
- **나중에 붙일 수 있게 지금 해 둘 것**(Miconi 식이 Elman RNN에 그대로 더해지는 형태라 가장 싸다):
  1. 순방향 계산을 `W_eff = W + alpha * Hebb` 꼴로 쓸 수 있게 함수 경계를 잡는다. 1단계에서는 `alpha`, `Hebb` 배열을 만들지 않거나 0으로 둔다. `alpha = 0`이면 지금과 완전히 같은 동작이므로 2단계 전환이 중립 변이로 시작된다.
  2. 개체별 "생애 상태"(은닉 상태 `h`, 나중의 `Hebb`, 호르몬 수치)와 "유전체"(가중치, `alpha`, `η`, σ, 크기 유전자)를 **다른 배열 묶음으로 분리**한다. 출생 때 유전체만 복사·변이하고 생애 상태는 0으로 초기화.
  3. 출력 벡터에 **쓰지 않는 예비 출력 1~2개**를 처음부터 둔다(나중에 조절 신호 `m`: `Hebb += m · η · pre · post` 형태의 게이트로 사용). 입력에도 예비 자리(나중에 호르몬·보상 신호)를 둔다. 예비 입력은 0, 예비 출력은 무시하면 1단계 동작에 영향 없음.
  4. 유전체를 "이름 붙은 배열들의 묶음(dict/구조체)"으로 두고 변이 함수가 배열마다 σ 배율을 받게 하면, 새 유전자 종류 추가가 한 줄이 된다.
- 조절(neuromodulation)은 "언제 배울지"를 뇌가 정하게 해 주므로 3단계(호르몬·보상) 설계와 직접 이어진다. Soltoggio 결과는 조절 뉴런이 허용만 되면 진화가 스스로 쓴다는 것이므로, 자리를 비워 두는 전략과 맞는다.
- 가소성의 계산 비용: `Hebb` 갱신은 개체당 외적 `H×H` 한 번(행렬곱 한 번과 같은 크기)이라 2번 표의 시간이 대략 2배가 되는 정도로 추정.

### Gaps
- **Backpropamine(Miconi 등 2019)**: 원문을 열지 못했다. [기억] 망이 내는 조절 신호 `M(t)`로 `Hebb` 갱신을 게이트하고(`Hebb += M(t) · x_i · x_j`를 자른 값), 적격 흔적(eligibility trace)을 쓰는 "retroactive" 변형이 있다. 출처 후보: https://openreview.net/forum?id=r1lrAiA5Ym
- Soltoggio 2008의 정확한 규칙 식([기억]: `Δw = tanh(m/2) · η · (A·pre·post + B·pre + C·post + D)`)과 T-maze 수치는 본문을 열지 못해 확인 못 함.
- Hinton & Nowlan의 초기 대립유전자 비율(0.25/0.25/0.5)과 학습 시행 1,000회는 [기억]이며 이번에 확인되지 않았다.
- 비에피소드 자연선택 세계에서 "진화만 vs 진화 + 가소성"을 직접 비교한 실험은 찾지 못했다(Polyworld는 처음부터 Hebb 학습 포함이라 대조군이 없음).

## 6. 뇌 에너지 비용 — 정확한 식, 생물학적 근거, 비용 수준에 따른 뇌 크기 결과

### Takeaway
선례의 뇌 비용은 거의 모두 **구조 비용**(뉴런 수·시냅스 수에 비례, 활동량과 무관)이고, Polyworld 기본값에서는 최대 크기 뇌가 고정 대사의 약 2.2배·전속 이동과 같은 수준의 비용을 낸다. 실험 결과는 일관되게 "비용이 있으면 뇌가 작아지되 성능은 유지되고, 큰 뇌는 순 에너지 섭취가 높은(쉬운·안정적인) 환경에서만 유지된다"이다.

### Cited Findings
- **Polyworld 뇌 비용 식(소스)**:
  ```
  brainEnergy = EnergyUseNeurons  * numNeurons  / maxNeurons
              + EnergyUseSynapses * numSynapses / maxSynapses
  ```
  기본값 `EnergyUseNeurons = 0.1`(주석 "was 1.84; where did that come from?"), `EnergyUseSynapses = 0.01`(주석 "was 0.34"). `maxNeurons`/`maxSynapses`는 세계 설정에서 가능한 최댓값. 뇌가 만들어질 때 한 번 계산되는 **상수**이며 활동량 항은 없다. [직접] — [GroupsBrain.cc](https://github.com/polyworld/polyworld/blob/master/src/library/brain/groups/GroupsBrain.cc), [worldfile.wfs](https://github.com/polyworld/polyworld/blob/master/etc/worldfile.wfs)
- **Polyworld 스텝당 총 에너지 식(소스)**:
  ```
  energyused = eat*EnergyUseEat(0.01) + mate*EnergyUseMate(0.1) + fight*EnergyUseFight(1.0)
             + speed*EnergyUseMove(0.1)*maxSpeed유전자 + |dyaw|*EnergyUseTurn(0.1)*maxSpeed유전자
             + brainEnergy + EnergyUseFixed(0.05)   (+ light, give, carry 항)
  denergy = energyused * Strength() * (1 + AgeEnergyMultiplier * age)
  denergy *= (개체 수 기반 배율들) * EnergyUseMultiplier(1.0)
  ```
  → 최대 크기 뇌의 비용은 0.1 + 0.01 = **0.11/스텝**, 고정 대사는 0.05/스텝, 전속 이동은 0.1 × maxSpeed/스텝. [직접] — [agent.cc](https://github.com/polyworld/polyworld/blob/master/src/library/agent/agent.cc), [worldfile.wfs](https://github.com/polyworld/polyworld/blob/master/etc/worldfile.wfs)
- **Polyworld 1994의 설명**: 신경 처리 에너지는 뉴런 수와 시냅스 수에 선형이며, 세계 전체의 최댓값 대비 비율에 "neuron-to-energy", "synapse-to-energy" 계수를 곱한다. 계산 시간으로 벌주는 대신 에너지로 벌준다는 설계 의도: "The penalty is more properly derived from the additional energy use associated with these additional neural structures". 또 "there are clear energy conservation benefits to being small and weak"라고 적는다. [직접] — [Yaeger 1994](https://shinyverse.org/al4ai/papers/Yaeger.Polyworld.pdf)
- **JaxLife에는 뇌 크기 비용이 없다**: 스텝당 비용은 `life_cost = age / AGE_COST(256) + LIFE_COST(0.25)` + 행동별 비용(지형 칸이 배율을 정함; `MOVE_COST 0.5`, `EAT_COST 0.1`, `PUSH_COST 0.25`, `HIT_COST 0.4`, 번식 `MAX_REPRODUCE_COST 32`). 뇌 크기는 모든 개체가 같으므로 비용 항이 될 수 없다. 나이가 들수록 유지비가 선형으로 오르는 것이 수명 제한 역할. [직접] — [JaxLife agent.py](https://github.com/luchris429/JaxLife/blob/main/src/world/agent.py), [main.py](https://github.com/luchris429/JaxLife/blob/main/src/main.py)
- **Inria eco-evo에도 뇌 비용이 없다**: 스텝당 −0.025 고정, 자원 +1, 최대 에너지 3. [직접(요약 모델 경유)] — [arXiv 2302.09334v3](https://arxiv.org/html/2302.09334v3)
- **Bibites**: 알을 낳을 때 은닉 뉴런 하나당 **Neuron Birth Cost 250 mE(0.25 E)**, 시냅스 하나당 **Synapse Birth Cost 50 mE(0.05 E)**(기본값; 위키 쪽이 오래돼 바뀌었을 수 있음). 작은 뇌에서는 미미하지만 큰 망에서는 알 낳는 비용의 큰 비중을 차지한다고 설명. 별도의 유지비(upkeep)도 있다고 언급되나 식은 확인 못 함. [요약] — [Neuron Birth Cost](https://the-bibites.fandom.com/wiki/Neuron_Birth_Cost), [Synapse Birth Cost](https://the-bibites.fandom.com/wiki/Synapse_Birth_Cost)
- **Heesom-Green 등 2025의 비용 식**: 비용 없음(NEC) `E = −0.01`/스텝, 비용 있음(EC) **`E = −0.01 × N_S(현재) / N_S(0세대)`**, N_S = 연결 수 + 비입력 노드 수, 0세대 N_S = 254. 에피소드 100스텝, 먹이 +1, 독 −1. [직접] — [arXiv 2511.20018](https://arxiv.org/abs/2511.20018)
- **같은 논문의 결과**: (1) 비용이 없으면 계절 수(환경 변동)와 무관하게 망이 커지고 환경 간 크기 차이가 없다(Kruskal-Wallis p > 0.05). (2) 비용이 있으면 **모든 환경에서 망이 더 작다**(NEC > EC, p = 0.00~0.01)인데 **과제 성능은 차이가 없다**(p = 0.27~0.56). (3) 비용이 있을 때 계절이 많을수록 망이 작다(계절 수와 N_S의 Spearman ρ = −0.4322). 그림 2b에서 읽은 400세대 평균 N_S(대략값): NEC 약 330~345, EC는 1계절 약 308·2계절 약 300·3계절 약 280·4계절 약 254(약 150세대까지 커졌다가 초기 크기로 되돌아감). (4) 매개 분석: 계절이 늘면 성능(순 에너지 섭취)이 떨어지고(−1.46, −2.02, −3.56), 성능이 높을수록 망이 크다(계수 14.50). 저자 결론: 큰 망은 **순 에너지 섭취가 높은 환경(정적 환경)**에서 유지된다 — Expensive Brain Hypothesis 지지, Cognitive Buffer Hypothesis 반박. [직접] — [arXiv 2511.20018](https://arxiv.org/abs/2511.20018)
- **Expensive Brain Hypothesis의 정의(같은 논문 서론, Isler & van Schaik 2009 인용)**: 뇌 크기 증가는 순 에너지 섭취 증가 또는 다른 장기로 가는 에너지 감소로 메워져야 한다. [직접] — [arXiv 2511.20018](https://arxiv.org/abs/2511.20018)
- **"Big Brains and Changing Environments: Cause or Consequence?"(2026)**: 계절성 먹이 찾기 과제의 신경진화에서, 변하는 환경의 큰 망은 주로 **그 전에 정적 환경에서 진화한 이력**에서 나온다("larger neural networks in dynamic environments arise mainly from prior static evolution"). 그런 망이 예측 못 한 변화에서도 더 잘한다. 비용 구현 방식·수치는 초록에 없음. [요약(초록만)] — [arXiv 2609.15569](https://arxiv.org/abs/2609.15569)
- **Yaeger 2009 — 자연선택은 복잡도를 "충분할 때까지만" 올린다**: 자연선택 실행은 초기에 대조군보다 복잡도가 빨리 오르지만(유의성은 약 t = 4,000까지), "good enough" 해가 퍼지면 정체하고 그 뒤 자연선택은 "weakly stabilizing force"로 작용한다. 몇몇 실행에서는 더 잘 적응한 망이 나타나 다시 오른 뒤 또 정체(단속적 도약). 결론: 진화는 복잡도를 항상 편향되게 이끌지만 방향은 증가·정체·감소 모두 가능하다. [직접(요약 모델 경유)] — [Yaeger 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2801533)
- **생물학 — 신호 에너지 예산(Attwell & Laughlin 2001, 설치류 회백질)**: 신호 에너지의 **47%가 활동전위, 34%가 시냅스 후 효과(글루탐산), 13%가 휴지 전위 유지, 3%가 글루탐산 재활용**. 에너지 사용량은 발화율에 강하게 의존하고, 저자는 에너지 효율을 위해 **동시에 활성인 뉴런이 15% 이하**인 분산 부호를 제안. [직접(초록, Europe PMC 경유)] — [PubMed 11598490](https://pubmed.ncbi.nlm.nih.gov/11598490/)

### Inferences
- **1단계 기본 비용 식 제안**(Polyworld 식을 정규화 없이 풀어 쓴 형태, 스텝당):
  `E_brain = c_n · n_active + c_s · n_syn` (구조 비용, 상수) `[+ c_a · Σ|h|]` (활동 비용, 선택 사항)
  - `n_active` = 활성 은닉 뉴런 수, `n_syn` = 활성 뉴런 사이의 연결 수(prefix 방식이면 `n_active² + n_active·(I + O)`; 연결 마스크가 없으면 대신 `Σ|w|`를 쓰면 가중치를 0으로 줄이는 것이 시냅스 삭제와 같아져 희소화 압력이 생긴다).
  - **눈금 맞추기**: Polyworld 기본 비율을 따르면 "최대 크기 뇌 비용 ≈ 고정 대사의 2배 ≈ 전속 이동 비용", 그리고 뉴런 항 : 시냅스 항 = 10 : 1(최대 크기 기준). 즉 `c_n · Hmax ≈ 2 · E_fixed`, `c_s · n_syn_max ≈ 0.2 · E_fixed`.
  - 생물학 비율(Attwell & Laughlin: 활동 의존 약 81~84%, 휴지 13%)을 따르자면 활동 비용 항의 비중이 커야 하지만, ALife 선례는 모두 구조 비용만 쓴다. 활동 비용(`c_a · Σ|h|` 또는 `Σ|Δh|`)은 "가만히 있을 때는 생각도 덜 한다"는 희소 부호 압력을 주는 2차 장치로 남겨 두고 1단계는 구조 비용만으로 시작하는 것이 해석이 쉽다.
- **비용 수준에 대한 예측**(Heesom-Green + Yaeger 2009 + Bibites 경고를 합치면):
  - 비용 0: 뇌가 선택과 무관하게 불어난다 → "커졌다"가 의미 없음.
  - 비용이 크기에 비례하고 환경이 단순: 뇌가 "충분한 최소 크기"까지 줄고 성능은 유지된다 → 사용자가 이미 아는 "보통 뇌가 줄어든다"와 일치하며, 이것은 실패가 아니라 비용이 제대로 작동한다는 신호.
  - **큰 뇌가 남는 조건**: 순 에너지 섭취가 넉넉할 것(먹이가 풍부하거나, 큰 뇌로만 얻을 수 있는 고에너지 먹이가 있을 것). 환경이 어렵고 변덕스러운 것만으로는 뇌가 커지지 않고 오히려 줄어든다(섭취가 줄기 때문). 따라서 "똑똑함이 값을 하는 곳에서만 이득"을 보려면, 어려운 환경이 아니라 **똑똑해야만 열리는 풍부한 자원**(숨은 먹이, 움직이는 먹이, 기억이 필요한 재생 주기 등)을 세계에 둬야 한다.
  - 2609.15569 초록의 시사점: 쉬운·정적 환경에서 먼저 뇌를 키운 뒤 환경을 변하게 하는 순서가 반대 순서보다 큰 뇌를 남긴다 → 4번의 "너그러운 초기 환경"과 같은 방향.
- 출생 비용(Bibites식: 뉴런·시냅스 수에 비례해 알 비용 증가)은 유지비와 별개의 손잡이다. 유지비는 수명 내내, 출생 비용은 번식 속도에만 작용한다. 1단계는 유지비 하나로 시작하고, 뇌 크기 분포가 유지비에 지나치게 민감하면 출생 비용으로 나눠 싣는 것을 고려.
- 비용 계수는 유전자가 아니라 세계 상수로 두고, 실행 중 바꿀 수 있는 설정값으로 노출해 "비용 수준 → 평균 뇌 크기" 곡선을 직접 그려 보는 것이 가장 값진 첫 실험이 된다(Heesom-Green도 "varying energy costs to identify thresholds"를 진행 중 과제로 적음).

### Gaps
- **사람 뇌 "체중의 2%, 에너지의 약 20%"**와 **expensive tissue hypothesis(Aiello & Wheeler 1995)**는 널리 인용되는 내용이지만 이번 세션에서 1차 출처를 열어 확인하지 못했다. [기억]
- Attwell & Laughlin의 활동전위·소포당 ATP 개수는 초록에 없어 확인 못 함(본문 필요).
- **Bibites 뇌 유지비(upkeep) 식**은 위키가 열리지 않아(HTTP 402) 얻지 못했다. 출생 비용 수치도 검색 요약 기준이며 위키 쪽이 3~4년 전 내용이다.
- **Werld의 뇌 비용 식**은 이번에 다시 열지 않았다(앞선 조사에서 다룸). 검색 요약상 Werld 저장소가 뉴런 대사 비용을 언급한다는 것만 확인. [요약] — [github.com/nocodemf/werld](https://github.com/nocodemf/werld)
- Polyworld에서 `EnergyUseNeurons`/`EnergyUseSynapses` 값을 바꿔 가며 뇌 크기가 어떻게 변하는지 본 실험 보고는 찾지 못했다. 옛 값 1.84/0.34의 유래는 소스 주석도 모른다고 적혀 있다.
- 활동량 기반 뇌 비용(Σ|activation| 등)을 쓴 ALife 시스템의 1차 자료는 찾지 못했다.

## 7. 출력·행동 인터페이스 — 연속 vs 이산, 번식을 뇌가 정할지 자동으로 할지

### Takeaway
연속 2D 세계의 선례(Polyworld, JaxLife)는 **동시에 여러 개가 켜지는 연속 출력**을 쓰고 출력 크기가 곧 행동 세기이자 에너지 비용의 배율이다. 번식은 "뇌 출력 + 에너지 문턱"(Polyworld, JaxLife)과 "완전 자동"(Inria) 두 방식이 모두 작동했으며, 자동 쪽이 부트스트랩이 쉽고 뇌 출력 쪽이 번식 시점·투자량 전략의 진화를 허용한다.

### Cited Findings
- **Polyworld**: 출력 7개(eat, mate, fight, move, turn, focus, light)가 각각 0~1 활성값이며, 에너지 식에서 `eat·0.01 + mate·0.1 + fight·1.0 + speed·0.1·maxSpeed + |dyaw|·0.1·maxSpeed`로 **출력값에 비례해 비용**을 낸다(먹기 자체도 비용). 짝짓기는 두 개체가 모두 mate 출력을 내야 하고 `MateWait 25` 스텝 대기, 자식 에너지는 부모가 유전자(`MateEnergyFraction`)가 정한 비율만큼 내준다. [직접] — [agent.cc](https://github.com/polyworld/polyworld/blob/master/src/library/agent/agent.cc), [worldfile.wfs](https://github.com/polyworld/polyworld/blob/master/etc/worldfile.wfs), [Yaeger 1994](https://shinyverse.org/al4ai/papers/Yaeger.Polyworld.pdf)
- **JaxLife**: 연속 출력이 동시에 실행되고 크기가 효과 세기("the strength of the action's effect is determined by the magnitude of the corresponding entry"). Action 필드: `x_move, y_move, push(음수는 당김), eat, hit, reproduce, message_other, self_message[16], other_message[16], terrain_* 5개, read_terrain_bit`. **번식은 뇌 출력**이며 조건은 `reproduce` 출력이 켜지고 에너지 > `MAX_REPRODUCE_COST(32) × 지형 배율 × REPRODUCE_FACTOR(2)`. [직접] — [JaxLife structs.py](https://github.com/luchris429/JaxLife/blob/main/src/world/structs.py), [agent.py](https://github.com/luchris429/JaxLife/blob/main/src/world/agent.py), [arXiv 2409.00853](https://arxiv.org/html/2409.00853)
- **Inria eco-evo**: 이산 5행동(정지·상하좌우) softmax, 먹기는 칸에 올라서면 자동. **번식은 자동**: 에너지가 E_min 위로 140 스텝 유지되면 부모 칸에 자식 생성, 부모는 남아 다시 번식 가능. 연구실 실험에서 번식을 켜면 개체군 평균 채집 효율이 유의하게 올랐다. [직접(요약 모델 경유)] — [arXiv 2302.09334v3](https://arxiv.org/html/2302.09334v3)
- **Biosim4**: 이산 격자, 세대제라 번식 행동이 없다(세대 끝에 생존 기준을 통과한 개체가 부모). [직접] — [biosim4.ini](https://github.com/davidrmiller/biosim4/blob/main/biosim4.ini)
- **Polyworld에서 나온 "항상 켜 두기" 해**: frenetic joggers는 mate·eat 출력을 항상 켠 채 직진했다 — 뇌 제어 번식에서 가장 먼저 발견되는 해는 "상수 출력"이다. [직접] — [Yaeger 1994](https://shinyverse.org/al4ai/papers/Yaeger.Polyworld.pdf)

### Inferences
- **연속 출력 권장**: `tanh` 출력을 그대로 추진력(−1~1 또는 0~1)·회전(−1~1)으로 쓰고, 먹기·번식은 0~1 "의지"로 읽어 문턱(예: > 0.5)을 넘으면 실행. 출력 크기에 비례한 행동 비용을 물리면(Polyworld) "쓸데없이 항상 전속"이 자연히 억제된다. 이산 softmax(Inria)는 격자 세계에는 맞지만 연속 물리에서는 동시 행동(가면서 돌기)을 표현하지 못한다.
- **번식 방식 비교**:
  | | 자동(에너지 문턱) | 뇌 출력 + 에너지 문턱 |
  |---|---|---|
  | 선례 | Inria | Polyworld, JaxLife, (Bibites — 미확인) |
  | 부트스트랩 | 쉽다: 먹기만 배우면 번식이 따라옴 | 어렵다: 먹기와 번식 출력을 동시에 찾아야 함 |
  | 진화 여지 | 번식 시점·자식 투자량 전략이 못 나옴 | r/K식 전략, 안전한 곳에서 낳기 등이 가능 |
  | 흔한 첫 해 | — | 번식 출력 상수로 켜 두기(frenetic joggers) |
- 절충안: **번식 출력의 초기 편향을 "켜짐"으로 두는 뇌 제어 방식**. 시작은 사실상 자동(에너지 문턱만 넘으면 번식)이고, 진화가 필요할 때 끄는 법을 배운다. 4번의 "손으로 심은 본능"과 같은 장치이며 Polyworld가 저절로 도달한 첫 해를 미리 주는 셈이다. 자식에게 줄 에너지 비율을 유전자로 두는 것(Polyworld `MateEnergyFraction`)은 유전자 1개로 투자 전략 진화를 연다.
- 번식 뒤 대기 시간(Polyworld `MateWait 25`)이나 번식 비용이 없으면 문턱을 넘는 순간마다 연속 번식해 폭증·붕괴 진동이 커진다. Inria가 "문턱 위 140 스텝 유지"를 요구한 것도 같은 역할이다.
- 5번의 예비 출력(나중의 조절 신호·호르몬 분비)을 처음부터 출력 벡터에 포함해 두면 인터페이스 변경 없이 단계 확장이 된다.

### Gaps
- Bibites의 출력 목록(알 낳기 의지 출력의 존재와 이름, 예: "Want2Lay")은 위키를 열지 못해 확인 못 함. [기억]으로는 번식(알 낳기)이 뇌 출력으로 제어된다.
- 같은 세계에서 "자동 번식 vs 뇌 제어 번식"을 직접 비교한 실험은 찾지 못했다. 위 표의 부트스트랩 난이도 비교는 추론이다.
- JaxLife의 `reproduce` 출력 문턱(양수면 켜짐인지)은 소스의 해당 줄 전체를 확인하지 않았다(206~211행 일부만 확인).
