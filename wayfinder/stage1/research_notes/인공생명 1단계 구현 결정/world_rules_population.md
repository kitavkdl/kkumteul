# 세계 규칙·개체 수 조절·죽음/번식·"세대" 정의 (연속·비에피소드 2D 인공생명, 신경망 개체)

읽는 법
- **[직접]** = 원문 PDF를 `pdftotext`로 풀어 직접 읽음 (Inria 2302.09334, Polyworld Yaeger 1994, JaxLife 2409.00853 본문).
- **[요약기]** = 원문 파일(소스 코드·설정)을 WebFetch로 열었지만 작은 요약 모델이 값을 뽑아 줌. 값은 원문에서 온 것이나, 구현에 넣기 전 해당 줄을 한 번 더 확인할 것.
- **[스니펫]** = 페이지를 열지 못하고 검색 결과 요약만 봄 (The Bibites 위키는 HTTP 402로 전부 막힘, Tierra·ECJ도 스니펫).
- 수식·변수명·URL은 원문 그대로 둠.

---

## 1. 세계 크기와 경계 (torus vs 벽), 개체 밀도

### Takeaway
비교 대상 대부분은 "개체 수 ≈ 칸(또는 면적)의 1% 안팎, 시야가 세계의 0.3% 안팎"인 성긴 세계를 쓴다. 벽은 Polyworld에서 "edge runners"(가장자리만 도는 종)와 구석 뭉침을, torus는 "frenetic joggers"(직진만 하는 종)를 낳았다 — 둘 다 인공물이 있고, 1단계는 torus + 먹이를 뭉쳐 두어 직진 전략을 불리하게 만드는 쪽이 구현·해석 모두 단순하다.

### Cited Findings
- Polyworld 기본값: `WorldSize` 100.0, `Edges` 기본 `B`(Blocked). 선택지는 `B`(Blocked), `W`(Wraparound), `T`(Tabletop — 떨어지면 죽음), `S`(Sticky and Blocked). 개체 수 `MinAgents` 90 / `InitAgents` 180 / `MaxAgents` 300, 먹이 `MaxFood` 120. **[요약기]** — [polyworld etc/worldfile.wfs](https://raw.githubusercontent.com/polyworld/polyworld/master/etc/worldfile.wfs)
- Polyworld 1994 원문: 가장자리는 "act as barriers (the usual), wrap around, or aren't there at all"; 마지막 경우 가장자리를 넘은 개체는 즉시 죽는다. **[직접]** — [Yaeger 1994, §9 The New Context](https://shinyverse.org/larryy/Yaeger.ALife3.pdf)
- torus 인공물 — "frenetic joggers": 장벽 없음 + miscegenation function 없음 + wrap-around 세계에서 "basically just ran straight ahead at full speed, always wanting to mate and always wanting to eat"인 집단이 나와 세계를 끝까지 지배했다. 세계가 "benign enough"해서 직진만으로 먹이·짝을 충분히 만났기 때문. **[직접]** — [Yaeger 1994, §10](https://shinyverse.org/larryy/Yaeger.ALife3.pdf)
- 벽 인공물 — "edge runners": 세계를 "approximately one-dimensional curve"로 줄여 가장자리만 계속 돈다. 길 위에 죽은 개체가 먹이가 되고 짝도 쉽게 만나므로 꽤 좋은 전략. 장벽으로 막아도 가장자리·장벽 근처에 모이는 행동이 남는다. **[직접]** — [Yaeger 1994, §10](https://shinyverse.org/larryy/Yaeger.ALife3.pdf)
- 벽 인공물 — 구석 뭉침: "indolent cannibals"는 에너지 균형을 고친 뒤에도 "near corners of the world, where some organisms with limited motor skills naturally end up congregating"에서 가끔 다시 나타났다. **[직접]** — [Yaeger 1994, §10](https://shinyverse.org/larryy/Yaeger.ALife3.pdf)
- 떨어지면 죽는 가장자리(Tabletop)에서는 "dervishes"(빠르게 도는 전략)가 진화해 수백 세대 유지되었다. **[직접]** — [Yaeger 1994, §10](https://shinyverse.org/larryy/Yaeger.ALife3.pdf)
- Inria eco-evo: grid 400x200(=80,000칸), 시작 330 개체, 최대 1000 개체, 시야 `w_o` = 15 (15x15, 사방 7칸). 벽 칸은 지나가려는 개체를 죽인다("walls (in blue) that kill agents trying to traverse them"). **[직접]** — [arXiv 2302.09334, App. B.1·B.4](https://arxiv.org/abs/2302.09334)
- JaxLife 기본값: `--num-agents` 128, `--num-bots` 32, `--map-size` 128, `MIN_AGENTS = num_agents // 4`. **[요약기]** — [JaxLife src/main.py](https://raw.githubusercontent.com/luchris429/JaxLife/main/src/main.py)
- JaxLife 규모 실험: 32 개체에서는 "we do not observe any meaningful emergent patterns", 256 개체에서는 에너지 사용의 봉우리·골이 더 뚜렷(급증 뒤 급락). **[직접]** — [arXiv 2409.00853, Scaling 절·Fig. 7](https://arxiv.org/abs/2409.00853)
- Avida 기본값: `WORLD_X 60`, `WORLD_Y 60`, `WORLD_GEOMETRY 2`(Toroidal Grid; 1 = Bounded Grid). **[요약기]** — [avida.cfg](https://raw.githubusercontent.com/devosoft/avida/master/avida-core/support/config/avida.cfg)
- The Bibites: `Simulation Size`는 세계의 최대 크기(단위 u)이며 두 배로 하면 접근 가능한 면적과 개체 수가 약 4배, 부하도 약 4배. **[스니펫]** — [Bibites wiki: Simulation Size](https://the-bibites.fandom.com/wiki/Simulation_Size)

### Inferences
- 밀도 환산(위 수치로 계산): Inria 최대 1000/80,000칸 = 칸의 1.25%(시작 0.41%); 시야 225칸 = 세계의 0.28%, 최대 인구일 때 시야 안 다른 개체 기대값 ≈ 2.8(시작 때 ≈ 0.9). JaxLife 128/16,384칸 = 0.78%. Polyworld 300/(100×100) = 0.03 개체/단위². Avida는 3,600칸이 곧 최대 개체 수(칸당 1개체).
- "시야 안에 다른 개체가 평균 1~3" 정도가 되도록 세계 크기를 정하면 위 선례와 같은 밀도대다. 예: 감각 반경 r, 목표 개체 수 N이면 세계 면적 A ≈ N·πr²/2 (기대 이웃 2 기준).
- torus를 쓰되 먹이를 균일하게 뿌리면 Polyworld의 "직진만 하는 종"이 그대로 재현될 가능성이 크다. 먹이를 patch로 뭉치거나(§2) 이동 에너지 비용을 속도에 비례시키면 직진 전략의 이득이 줄어든다.
- 벽을 쓰면 가장자리·구석이 "먹이와 짝이 모이는 1차원 통로"가 되어 행동 해석이 흐려진다. numpy에서는 torus가 `pos %= size` 한 줄이고, 거리 계산만 최소 이미지(minimum image)로 바꾸면 된다.
- JaxLife 결과로 보면 개체 수 수십(32)은 너무 적고, 100 단위 이상이 필요하다. 반대로 상한 근처(256)는 진동이 커진다.

### Gaps
- Inria 격자의 바깥 테두리가 torus인지 벽인지 논문에 적혀 있지 않다(벽 칸이 개체를 죽인다는 것만 있음). [EcoEvoJax 코드](https://github.com/flowersteam/EcoEvoJax)를 열어 보지 못함.
- JaxLife의 경계 처리(wrap/clip)는 `main.py`에 없고 `src/world.py` 추정 경로는 404였다. 미확인.
- The Bibites의 기본 `Simulation Size`·void(경계 밖) 규칙 수치는 위키가 HTTP 402로 막혀 확인 못 함.
- Polyworld 개체의 몸 크기·시야 거리 대비 `WorldSize` 100의 비율은 확인 못 함.
- Geb(Channon), Evosphere, Neural MMO, Biomaker CA의 세계 크기·경계는 이번에 원문을 열지 못했다(검색 결과 없음).

---

## 2. 먹이: 생성·재성장 규칙

### Takeaway
확인된 규칙은 세 갈래다 — (a) Inria: 이웃 먹이에 비례하는 확률적 재성장 + 아주 낮은 자발 성장(완전 소멸 방지), (b) Polyworld: 상한까지 스텝당 확률로 무작위(또는 patch 안) 생성 + 죽은 개체가 먹이가 됨, (c) The Bibites: 총 에너지가 고정된 닫힌 계. "이웃 비례 재성장"은 양의 되먹임이라 붕괴하기 쉽고(Inria 5개 중 3개 seed 붕괴), 자발 성장 항과 상한이 안정 장치 역할을 한다.

### Cited Findings
- Inria 재성장 식: `p(x, y) = p_I(x, y) · c(x) + c`. `p_I(x, y) = 1_{I=1} * 0.002`(원문 표기 "⊮𝐼=1 ∗ 0.002"), `I`는 "the 4 direct neighbors", 기후값 `c(x) = (α^x + 1)/(α + 1)`, `α = 200`, 자발 성장 `c = 0.00005`. 한 칸에 자원은 최대 1개. **[직접]** — [arXiv 2302.09334, App. B.1 식 (1)·B.4](https://arxiv.org/abs/2302.09334)
- Inria 시작 자원: "Starting resources : 16 0000 randomly placed resources (there can only be at most 1 resources at one pixel)". **[직접]** — [arXiv 2302.09334, App. B.4](https://arxiv.org/abs/2302.09334)
- Inria가 자발 성장 항을 둔 이유: 자원이 "do not disappear too easily, which can be problematic in reset-free environments". 또 이웃 의존 재성장은 "positive feedback loop, with reduction in resources leading to further reductions". **[직접]** — [arXiv 2302.09334, App. B.1](https://arxiv.org/abs/2302.09334)
- Inria 위도 모형: 세계를 가로 줄무늬 N개 niche로 나누고 아래쪽(저위도)일수록 재성장률이 높다. **[직접]** — [arXiv 2302.09334, Fig. 1·App. B.1](https://arxiv.org/abs/2302.09334)
- Inria에서 관찰된 먹이–인구 동역학: 작은 시간 척도에서 "predator-prey Lokta-Volterra dynamic"; 초기(0~200K 스텝)는 자원이 풍부해 많이 움직이는 행동이 쉬운 첫 전략이 되고, 이후 자원이 줄면서 적게 움직이는 개체가 늘었다. 초기 자원 감소는 "near tragedy-of-the-commons". **[직접]** — [arXiv 2302.09334, §2.1·App. D.1](https://arxiv.org/abs/2302.09334)
- Polyworld 1994: "Food is grown at a specifiable rate up to a specifiable maximum number of grown food items." 먹이 수는 지정한 최소·최대 사이로 유지할 수 있고, 먹이 한 조각의 에너지는 최소·최대 food energy 사이에서 무작위. 개체가 죽으면 그 개체의 food-value-energy(또는 지정한 minimum-food-energy-at-death)만큼의 먹이가 된다. **[직접]** — [Yaeger 1994, §9](https://shinyverse.org/larryy/Yaeger.ALife3.pdf)
- Polyworld 기본값: `FoodGrowthRate` 0.1(범위 0.0~10.0; 소수부 = 스텝당 확률, 정수부 = 스텝당 추가 개수), `FoodGrowthModel` 기본 `MaxIndependent`(소수부를 그대로 확률로 씀; 옛 방식 `MaxRelative`는 소수부 × (maxGrown − current)), `MinFood` 0(옛 기본 90), `MaxFood` 120, `InitFood` 90, `MinFoodEnergy` 200.0, `MaxFoodEnergy` 1000.0, `FoodMaxLifeSpan` 0(= 무한), `AgentsAreFood` True. 기본 Domain에는 `FoodFraction` 0.2와 0.8인 patch 두 개. **[요약기]** — [polyworld etc/worldfile.wfs](https://raw.githubusercontent.com/polyworld/polyworld/master/etc/worldfile.wfs)
- Polyworld 에너지 보존 감시: (food-energy in − food-energy out)/(in + out) 비율을 그리며, 에너지가 보존되는 세계라면 0.0으로 수렴해야 한다고 적혀 있다. **[직접]** — [Yaeger 1994, §9](https://shinyverse.org/larryy/Yaeger.ALife3.pdf)
- JaxLife 지형: 칸마다 현재 에너지와 스텝당 energy gain이 있고, 날씨·기후 계가 매 스텝 기본 지형을 조금씩 바꾼다. 기본값 `--max-terrain-energy` 8, `--max-terrain-energy-gain` 1, `--terrain-alpha` 0.005, `--weather-change-speed` 0.001, `TERRAIN_PERIOD` 182.5, `TERRAIN_GAIN_SCALING` 1.5, 초기 지형 `perlin`(`--perlin_res` 4). **[직접](본문) + [요약기](수치)** — [arXiv 2409.00853, Terrain 절](https://arxiv.org/abs/2409.00853); [JaxLife src/main.py](https://raw.githubusercontent.com/luchris429/JaxLife/main/src/main.py)
- The Bibites: 닫힌 계이며 총 에너지 `TE = Biomass Density * Simulation Size²`. 시작 pellet 수 ≈ `(Biomass Density * Simulation Size^2 - 5000E) / Average Pellet Size`. 기본 밀도는 bibite가 진화할 만큼 에너지를 보존하면서 경쟁도 있게 하는 값이라고 설명. **[스니펫]** — [Bibites wiki: Global Biomass Density](https://the-bibites.fandom.com/wiki/Global_Biomass_Density)

### Inferences
- Inria의 "16 0000"은 80,000칸에 칸당 최대 1개라는 조건과 맞으려면 16,000개(칸의 20%)로 읽어야 한다. 160,000은 불가능.
- Inria 수치로 계산한 수급: 개체 하나의 유지 소비는 0.025/스텝 = 먹이 1개/40스텝. 최대 1000 개체면 25개/스텝이 필요. 자발 성장만으로는 빈 지도에서 80,000 × 0.00005 = 4개/스텝. 즉 자발 성장만으로 버틸 수 있는 인구는 약 160 개체 — 이것이 "바닥 수용력"이고, 그 위는 이웃 재성장(먹이를 남겨 둬야 생김)에 달려 있다. 1단계에서도 "자발 성장만으로 먹여 살릴 수 있는 인구"를 목표 최소 인구로 잡으면 멸종 방지 장치가 된다.
- `c(x)`는 x를 0~1로 정규화했다고 보면 x=0에서 2/201 ≈ 0.01, x=1에서 1 — 위도에 따라 재성장률이 약 100배 차이 난다(x의 정규화는 코드 미확인).
- "안정적이지만 압박이 있는" 조건을 한 식으로 쓰면: 먹이 유입률 F(에너지/스텝)과 개체당 유지 비용 m이면 수용력 K ≈ F/m. 고정 유입(Polyworld식 상한 + 일정 생성률)은 K가 명확하고, 이웃 비례 재성장(Inria식)은 K가 개체 행동에 따라 변해 진동·붕괴가 생긴다. 1단계는 "일정 유입 + 상한 + patch 배치"가 가장 예측 가능하다.
- Polyworld 기본값의 비율: 먹이 상한 120개 × 평균 600 에너지 = 72,000, 개체 최대 에너지 500~1000 × 최대 300 개체 — 먹이에 저장된 에너지가 인구 전체 저장량보다 작게 잡혀 있다(풍요 방지).

### Gaps
- `p_I`의 지시함수가 "이웃이 정확히 1개일 때"인지 "1개 이상일 때"인지 원문 표기만으로는 알 수 없다. EcoEvoJax 코드 확인 필요.
- The Bibites의 기본 biomass density 값, pellet 크기, pellet 재생성(죽은 개체·배설 → pellet) 속도는 위키가 막혀 확인 못 함.
- JaxLife 지형 에너지의 정확한 갱신식(alpha·period가 식에서 어떻게 쓰이는지)은 `world.py`를 열지 못해 미확인.
- Polyworld `MaintainFood()` 본문(스텝당 생성 코드)은 파일이 길어 읽힌 범위 밖이었다. `docs/FoodPatchNotes.pdf`도 열지 않음.
- Echo(Holland), Evosphere, Biomaker CA, Neural MMO의 먹이 재성장 수치는 원문을 찾지 못함. 계절(seasonal) 먹이의 구체적 선례 수치도 JaxLife의 `TERRAIN_PERIOD` 182.5 외에는 없음.

---

## 3. 적합도 함수 없는 개체 수 조절

### Takeaway
실제로 쓰인 장치는 다섯 가지다: 유한한 먹이 유입(기본), 상한에서 출생 차단, 하한에서 재시드(무작위/저장 유전체), 인구에 따라 에너지 소비를 스케일(Polyworld 최신 기본), 상한에서 솎아내기(smite/reaper). Polyworld는 이 중 넷을 차례로 덧붙였고 최신 기본은 "인구 의존 에너지 비용"이다 — 하한·상한에 닿기 전에 부드럽게 밀어내는 방식이 가장 뒤에 도입되어 기본이 되었다.

### Cited Findings
- Inria: "this system required some tuning of the hyperparameters in order to find a stable environment, as exponential growth of both food and population can easily lead to collapse (and even did after several generations in 3 out of the 5 seeds launched)". 조절 장치는 먹이 한계 + `Max population : 1000 agents` + `Maximum age : 650`뿐이고 재시드는 없다. **[직접]** — [arXiv 2302.09334, §2·App. B.4](https://arxiv.org/abs/2302.09334)
- Inria에서 인구는 "bursts and crashes"를 보이고 "over-population leads to near-extinction of resources"가 관찰되었다. **[직접]** — [arXiv 2302.09334, 결론부](https://arxiv.org/abs/2302.09334)
- Polyworld 1994 — 하한 재시드: Successful Behavior Strategy(SBS, 출생만으로 수를 유지)가 나오기 전까지 "on-line GA(Steady State GA)" 모드. 개체 수가 최소 아래로 떨어지면 (1) 무작위 개체 생성, (2) "the offspring of two organisms from a table of the N fittest", (3) 드물게 역대 최고 개체를 그대로 복귀(elitist) 중 하나. ad hoc fitness는 먹기·짝짓기·수명 채우기·남은 에너지로 죽기·이동을 보상. "Once an SBS has emerged, there is no fitness function except survival." 어떤 실행은 첫 seed에서 SBS를 얻고, 어떤 실행은 끝내 못 얻어 "unsuccessful"로 분류. **[직접]** — [Yaeger 1994, §2 Overview](https://shinyverse.org/larryy/Yaeger.ALife3.pdf)
- Polyworld 1994 — 상한: "a maximum number of organisms is specifiable, in order to keep the problem computationally tractable"; 짝짓기 결과는 "the maximum number of organisms permitted in the world"에 의해 막힐 수 있다. **[직접]** — [Yaeger 1994, §8 Behavior·§9](https://shinyverse.org/larryy/Yaeger.ALife3.pdf)
- Polyworld 현재 코드 — 출생 차단: `GetMateDenialStatus`가 `domain.numAgents >= domain.maxNumAgents`, `fNumberAlive >= fMaxNumAgents`를 검사. **[요약기]** — [Simulation.cc](https://raw.githubusercontent.com/polyworld/polyworld/master/src/library/sim/Simulation.cc)
- Polyworld 현재 코드 — Smite: `SmiteMode` 기본 `O`(Off), 선택지 `R`(Random)·`L`(Least Fit); `SmiteFrac` 0.1, `SmiteAgeFrac` 0.25. 짝이 맺어졌는데 domain이 `maxNumAgents`에 닿아 있으면 실행. 후보 조건: domain이 상한 근처이고 `c->Age() >= (fSmiteAgeFrac * c->MaxAge())`이며 적합도가 평균 미만. `L` 모드는 부모와 fittest 목록을 빼고 가장 덜 적합한 후보를, `R` 모드는 부모를 뺀 무작위 개체를 죽인다. **[요약기]** — [worldfile.wfs](https://raw.githubusercontent.com/polyworld/polyworld/master/etc/worldfile.wfs); [Simulation.cc](https://raw.githubusercontent.com/polyworld/polyworld/master/src/library/sim/Simulation.cc)
- Polyworld 현재 코드 — 인구 의존 에너지 비용: `EnergyBasedPopulationControl` 기본 True(옛 기본 False). `PopControlMinFixedRange` 0.25, `PopControlMaxFixedRange` 0.75, `PopControlMinScaleFactor` 0.0, `PopControlMaxScaleFactor` 5.0. 인구가 고정 구간 안이면 변화 없음; 구간 아래면 `scaleFactor = 1.0 - (1.0 - fPopControlMinScaleFactor) * fraction`(0에서 자름); 구간 위면 `fractionReduced = pow(fraction, 4.0)`, `1.0 + (fPopControlMaxScaleFactor - 1.0) * fractionReduced`. **[요약기]** — [worldfile.wfs](https://raw.githubusercontent.com/polyworld/polyworld/master/etc/worldfile.wfs); [Simulation.cc](https://raw.githubusercontent.com/polyworld/polyworld/master/src/library/sim/Simulation.cc)
- Polyworld 현재 코드 — 그 전 방식 `ApplyLowPopulationAdvantage`(기본 False, 에너지 기반 조절이 꺼져 있을 때만): `fLowPopulationAdvantageFactor = 1.0 - (initNumAgents - numAgents) / (initNumAgents - minNumAgents)`를 [0, 1]로 자름. 코드 주석: "10% buffer, to help prevent reaching actual min value and invoking GA." **[요약기]** — [Simulation.cc](https://raw.githubusercontent.com/polyworld/polyworld/master/src/library/sim/Simulation.cc)
- Polyworld 현재 코드 — 하한 재시드 세부: `numToCreate = max(0, minNumAgents - numAgents)`; fittest 목록이 차 있으면 `fFitness1Frequency`번째마다 최상위 유전체 복사("revive 1 fittest"), `fFitness2Frequency`번째마다 fittest 둘을 교차("mate 2 from array of fittest"), 나머지는 무작위("just generate a random, hopeful monster"). 스키마 기본 `NumberFittest` 30, `EliteFrequency` 100, `PairFrequency` 2, `EndOnPopulationCrash` False. **[요약기]** — [Simulation.cc](https://raw.githubusercontent.com/polyworld/polyworld/master/src/library/sim/Simulation.cc); [worldfile.wfs](https://raw.githubusercontent.com/polyworld/polyworld/master/etc/worldfile.wfs)
- JaxLife: "We reinitialize the population—with random networks—if all agents die." 256 개체 실험에서는 살아 있는 수가 크게 떨어졌지만 "the population does not entirely die out, and the automatic reinitialization does not occur". 설정에 `MIN_AGENTS: num_agents // 4`, `ONLY_RESET_WHEN_NO_AGENTS: True`. **[직접](본문) + [요약기](설정)** — [arXiv 2409.00853](https://arxiv.org/abs/2409.00853); [JaxLife src/main.py](https://raw.githubusercontent.com/luchris429/JaxLife/main/src/main.py)
- Avida: 인구 조절은 공간 덮어쓰기 — `BIRTH_METHOD 0`("Which organism should be replaced when a birth occurs? 0 = Random organism in neighborhood"). `POPULATION_CAP 0`(0 = 상한 없음), `POP_CAP_ELDEST`는 "Will kill oldest organism in population". **[요약기]** — [avida.cfg](https://raw.githubusercontent.com/devosoft/avida/master/avida-core/support/config/avida.cfg)
- Tierra: reaper queue — 메모리가 80% 차면 queue 끝의 개체를 죽이고, 실행 오류를 낸 개체는 queue에서 한 칸 앞으로(벌점). Ray의 말: "The Reaper kills either the oldest creature or the most screwed-up creature". **[스니펫, 2차 자료]** — [Kurth 강의 노트](https://wkurth.grogra.de/cb/html/al2_v06c.pdf); [Kelly, Out of Control ch.15](https://kk.org/outofcontrol/ch15-a.html)
- The Bibites: 총 에너지가 고정된 닫힌 계(§2) + `Bibite Spawn Cap` 설정 존재. **[스니펫]** — [Bibites wiki: Bibite Spawn Cap](https://the-bibites.fandom.com/wiki/Bibite_Spawn_Cap); [Global Biomass Density](https://the-bibites.fandom.com/wiki/Global_Biomass_Density)

### Inferences
- Polyworld의 조절 방식은 스키마·코드에 층층이 남아 있다: (1994) 상한 출생 차단 + 하한 on-line GA → Smite(상한에서 덜 적합한 늙은 개체 제거) → `ApplyLowPopulationAdvantage`(하한 근처에서 유리하게) → `EnergyBasedPopulationControl`(현재 기본). 뒤로 갈수록 "딱딱한 경계에 닿기 전에 연속적으로 미는" 방식이다. 하한 GA는 ad hoc fitness라는 인위적 선택을 끌어들이므로 "닿지 않게 하는 것"이 목적임을 코드 주석("prevent … invoking GA")이 보여 준다.
- 기본값을 대입하면(구간 경계가 min~max 범위의 0.25·0.75 지점이라고 가정 — 이름에서 추정, 코드 미확인): Min 90 / Max 300이면 142~248 사이는 비용 ×1, 90에서는 ×0(에너지를 전혀 안 씀 → 굶어 죽지 않음), 300에서는 ×5. 위쪽은 4제곱이라 구간 바로 위에서는 거의 안 변하고 상한 근처에서 급증한다.
- 상한에서 출생만 막으면(Inria·Polyworld 초기) 이미 살아 있는 개체가 자리를 차지해 선택이 "오래 버티기" 쪽으로 쏠린다. Polyworld가 Smite에 나이 조건(`SmiteAgeFrac` 0.25)을 넣은 것, Tierra·Avida가 가장 늙은 개체를 죽이는 것 모두 같은 문제에 대한 답이다.
- Inria 붕괴의 원인은 논문 표현대로 "먹이와 인구가 둘 다 지수적으로 자라는 것"이다. 먹이가 이웃에 비례해 자라므로 다 먹히면 재성장 자체가 사라지고(§2), 자발 성장만으로 먹일 수 있는 인구(계산상 약 160)는 번식 가능한 최소 집단을 유지하기에 빠듯하다. 재시드가 없으니 한 번의 붕괴가 영구 멸종이 된다.
- 1단계 기본안으로 묶으면: (1) 일정한 먹이 에너지 유입(수용력 K를 직접 정함), (2) 자식 에너지는 부모에게서만(§5), (3) 연속적인 인구 의존 유지비(Polyworld식, 선택 사항), (4) 계산 보호용 hard cap(K의 2배쯤, 평소엔 닿지 않게), (5) 하한 재시드는 "저장해 둔 최근 유전체에서"(무작위 재시드는 진화를 0으로 되돌림 — JaxLife 방식), 재시드 발생 횟수를 기록해 0이 되는지를 건강 지표로 삼는다(Polyworld의 born/(born+created) 비율과 같은 생각).

### Gaps
- Polyworld가 조절 방식을 바꾼 이유를 설명하는 논문·커밋 기록은 찾지 못했다(스키마의 "legacy" 표시와 코드로만 확인). Griffith & Yaeger, Yaeger/Griffith/Sporns 후속 논문은 열지 못함.
- `EnergyScaleFactor`의 `fraction` 정의와 구간 경계 계산식은 요약기를 거친 설명뿐 — 원문 줄 확인 필요.
- JaxLife에서 `MIN_AGENTS`가 실제로 무엇을 하는지(하한 재시드인지 초기 생존 수인지)는 `world.py`를 못 열어 미확인. 고정 슬롯이 꽉 찼을 때 번식이 막히는지도 미확인.
- The Bibites의 `Bibite Spawn Cap` 기본값과 "최소 개체 수 유지(빈 세계에 기본 bibite 재투입)" 규칙 수치는 확인 못 함.
- 밀도 의존 번식 비용(개체가 많을수록 번식이 비싸짐)을 쓴 선례의 수치는 찾지 못했다. Polyworld는 번식 비용이 아니라 전체 에너지 소비를 스케일한다.

---

## 4. 죽음: 에너지 고갈·최대 수명·포식

### Takeaway
확인한 시스템은 모두 "에너지 고갈 + 나이 제한"을 함께 쓴다. 나이 제한은 딱딱한 최대 수명(Inria 650스텝, Polyworld 유전자 500~1000스텝, Avida `AGE_LIMIT`)이거나 나이에 따라 유지비가 오르는 부드러운 형태(JaxLife, The Bibites)다. 수명 제한을 뺀 실험 결과를 직접 보고한 자료는 찾지 못했다.

### Cited Findings
- Inria 죽음 규칙: "An agent dies once its energy level has been below a threshold E_min for at least T_death time-steps or if its age is bigger than a certain value L_max." 수치: `Time to die T_death : 200 timesteps`, `Energy death E_death : 0`, `Energy decay : 0.025`, `Max energy E_max = Starting energy E_0 : 3`, `Increase in energy when eating a resource : 1`, `Maximum age : 650`. **[직접]** — [arXiv 2302.09334, App. B.3·B.4](https://arxiv.org/abs/2302.09334)
- Inria 수명 지표: "Expectancy : Average of age of agent that died during a large time window of 500 timesteps." 진화가 진행되며 이 값이 올라가다 평탄해진다. **[직접]** — [arXiv 2302.09334, App. C·D.1.1](https://arxiv.org/abs/2302.09334)
- Polyworld 1994: 최대 수명은 유전자이며 "a few hundred to a few thousand 'time-steps' for life span" 범위로 제한. 평균 수명 약 500 time-steps. 싸움(fighting)으로 죽일 수 있고 죽은 개체는 먹이가 된다. **[직접]** — [Yaeger 1994, §2·§3](https://shinyverse.org/larryy/Yaeger.ALife3.pdf)
- Polyworld 기본값: `MinLifeSpan` 500, `MaxLifeSpan` 1000, `DieAtMaxAge` True, `AgeEnergyMultiplier` 0.0, `MinAgentMaxEnergy` 500.0, `MaxAgentMaxEnergy` 1000.0. 에너지 비용 `EnergyUseFixed` 0.05, `EnergyUseNeurons` 0.1, `EnergyUseSynapses` 0.01, `EnergyUseMove` 0.1, `EnergyUseTurn` 0.1, `EnergyUseEat` 0.01, `EnergyUseMate` 0.1, `EnergyUseFight` 1.0. **[요약기]** — [worldfile.wfs](https://raw.githubusercontent.com/polyworld/polyworld/master/etc/worldfile.wfs)
- Polyworld 뇌 에너지: 뉴런 수·시냅스 수에 선형이며, 세계 설정의 최대 뉴런·시냅스 수로 나눈 상대값에 "neuron-to-energy"·"synapse-to-energy" 계수를 곱한다. 스텝당 총 소비 = Σ(행동 뉴런 활성 0~1 × 그 행동의 계수) + 신경 에너지. strength 유전자(보통 0.5~2.0)가 총 소비를 스케일. **[직접]** — [Yaeger 1994, §4 Physiology and Metabolism](https://shinyverse.org/larryy/Yaeger.ALife3.pdf)
- Polyworld 현재 코드의 자연사 조건: `c->GetEnergy().isDepleted()` 또는 `c->Age() >= c->MaxAge()` 또는 (가장자리가 막히지도 이어지지도 않을 때) 세계 밖으로 나감, patch에 의한 죽음. 인구 하한을 깨는 죽음은 건너뛴다(무작위 `fAllowMinDeaths` 통과 시 예외). **[요약기]** — [Simulation.cc](https://raw.githubusercontent.com/polyworld/polyworld/master/src/library/sim/Simulation.cc)
- JaxLife: "Agents also age, and have to use more energy to survive as they become older." 설정 `AGE_COST: 256.0`, `AGE_SPEED: 1.0`; 딱딱한 최대 나이는 설정에 없다. **[직접](본문) + [요약기](설정)** — [arXiv 2409.00853, Agents 절](https://arxiv.org/abs/2409.00853); [JaxLife src/main.py](https://raw.githubusercontent.com/luchris429/JaxLife/main/src/main.py)
- Avida: `DEATH_METHOD`("When should death by old age occur?" 1 = `AGE_LIMIT`(+deviation)개 명령 실행 시, 2 = genome_length × `AGE_LIMIT` 명령 실행 시), `AGE_LIMIT 20`, `AGE_DEVIATION 0`. **[요약기]** — [avida.cfg](https://raw.githubusercontent.com/devosoft/avida/master/avida-core/support/config/avida.cfg)
- The Bibites: `Ageing Threshold`를 넘으면 `Age Strength Penalty`·`Age Metabolism Penalty`가 적용되는 부드러운 노화. **[스니펫]** — [Bibites wiki: Ageing](https://the-bibites.fandom.com/wiki/Ageing)

### Inferences
- Inria 수치로 계산: 가득 찬 에너지 3에서 아무것도 안 먹으면 3/0.025 = 120스텝에 0, 거기서 200스텝을 더 버티므로 굶어 죽기까지 약 320스텝. 최대 수명 650스텝의 약 절반 — "굶는 시간"과 "수명"이 같은 자릿수다.
- 모든 선례가 나이 제한을 둔 이유는 §3의 상한 문제와 같다: 인구가 상한·수용력에 닿으면 출생이 줄고, 그때 죽지 않는 개체가 자리를 영구히 차지해 세대 교체(=진화 속도)가 멈춘다. 최대 수명은 "세대 길이의 상한"을 직접 정하는 손잡이이기도 하다(§6).
- 부드러운 노화(JaxLife·Bibites식: 나이에 따라 유지비 증가)는 딱딱한 수명보다 인위적 절벽이 없고, 먹이가 풍부하면 수명이 늘어나는 식으로 환경과 엮인다. 다만 세대 길이가 예측하기 어려워지므로 "10세대/일"을 맞추려면 딱딱한 상한을 같이 두는 편이 안전하다(Polyworld: 유전자 수명 + `DieAtMaxAge` True + `AgeEnergyMultiplier` 선택).
- 포식은 Polyworld에서 "죽은 개체 = 먹이" + 싸움으로 구현되었고, 에너지 회계가 맞지 않을 때 "indolent cannibals" 문제를 키웠다(§5). 1단계에서는 포식을 빼고 에너지 고갈 + 최대 수명만으로 시작해도 선례와 어긋나지 않는다(Inria가 그 구성).

### Gaps
- "최대 수명이 없을 때 죽지 않는 지배 개체가 생긴다"는 것을 실험으로 보인 1차 자료는 찾지 못했다. 위 추론은 상한 출생 차단 구조에서 나오는 논리적 귀결이지 인용된 관찰이 아니다.
- JaxLife의 나이 비용 식(`AGE_COST`·`AGE_SPEED`가 어떻게 들어가는지) 미확인.
- The Bibites의 노화 기본 수치 미확인(위키 차단).
- Inria에서 에너지가 0 아래로 내려가는지(음수 허용인지 0에서 잘리는지), `E_min`과 `E_death`가 같은 값인지 논문만으로는 불분명.

---

## 5. 번식: 무성/유성, 에너지 문턱·전달, 자식 배치, 변이

### Takeaway
신경망 가중치를 직접 진화시키는 최근 시스템(Inria, JaxLife)은 모두 무성생식 + 가우시안 변이(σ = 0.01~0.02)를 쓰고, 자식은 부모 자리에 놓는다. 유성생식을 쓴 Polyworld에서 가장 중요한 교훈은 유성/무성 자체가 아니라 "자식 에너지를 부모에게서 빼지 않으면 공짜 에너지를 착취하는 종이 나온다"는 것이다.

### Cited Findings
- Inria 번식: "an agent needs to maintain its energy level above a threshold E_min for at least T_repr time steps. Once this happens the agent produces an off-spring and is a candidate for reproduction again." 죽지 않고 여러 번 낳을 수 있다. "Reproduction is asexual: an agent's weights are mutated by adding noise sampled from N(0, σ)". 수치: `Time to reproduce T_reproduce : 140 timesteps`, `Mutation variance σ : 0.02`. 자식은 "appears on the same cell as its parent". 신경망은 2445 parameters. **[직접]** — [arXiv 2302.09334, App. B.3·B.4·B.5](https://arxiv.org/abs/2302.09334)
- Inria의 번식 기준은 적합도가 아니라 "minimal criterion"(에너지 수준을 일정 시간 유지)이다. **[직접]** — [arXiv 2302.09334, App. B.3](https://arxiv.org/abs/2302.09334)
- JaxLife: "Since crossover for fixed-topology neural networks is challenging (Haflidason and Neville, 2009; Pretorius and Pillay, 2024), our agents reproduce asexually, and must have a minimum amount of energy before they can execute the REPRODUCE action. Reproduction copies the agent's weights to a child agent, and random perturbations are also added to these weights." **[직접]** — [arXiv 2409.00853, Reproduction 절](https://arxiv.org/abs/2409.00853)
- JaxLife 수치: `INITIAL_ENERGY` 32.0, `--max-reproduce-cost` 32.0, `REPRODUCE_FACTOR` 2(번식하려면 번식 비용의 몇 배를 가져야 하는지), `--mut-std` 0.01. **[요약기]** — [JaxLife src/main.py](https://raw.githubusercontent.com/luchris429/JaxLife/main/src/main.py)
- Polyworld 1994 번식: 두 개체가 겹쳐 있고 둘 다 mating 행동을 문턱 이상 표현해야 한다. 유전자가 "the fraction of an organism's remaining energy that it will donate to its offspring upon birth"를 정하고, 자식의 출생 에너지는 "the sum of these contributions from the two parents". 변이율 0.01~0.1, 교차점 2~8개. **[직접]** — [Yaeger 1994, §3 Genetics·§8](https://shinyverse.org/larryy/Yaeger.ALife3.pdf)
- Polyworld 기본값: `MinEnergyFractionToOffspring` 0.2, `MaxEnergyFractionToOffspring` 0.8, `MateWait` 25, `MinMateEnergyFraction` 0.0, `PreBirthCycles` 25, `MinMutationRate` 0.01 / `MaxMutationRate` 0.05(Bit 해상도), `EnergyUseMate` 0.1. **[요약기]** — [worldfile.wfs](https://raw.githubusercontent.com/polyworld/polyworld/master/etc/worldfile.wfs)
- Polyworld의 실패 사례 — 공짜 에너지: "indolent cannibals"(부모·자식 곁을 떠나지 않고 서로 짝짓고 싸우고 죽이고 먹는 종)는 "most prevalent in simulations run before the parents were required to transfer their own energies to the offspring; the organisms of these worlds were exploiting an essentially free energy source. With proper energy balancing, this behavior was reduced to only an occasional flare-up". 죽은 개체가 남기는 먹이를 없애도 나타났으므로 주된 이득은 "the ready availability of mates"였다. **[직접]** — [Yaeger 1994, §10](https://shinyverse.org/larryy/Yaeger.ALife3.pdf)
- Polyworld의 유성생식 부작용: miscegenation function과 장벽이 없으면 초기 다양성이 "quickly redistributed and blended into a single species"가 되었다. **[직접]** — [Yaeger 1994, §10](https://shinyverse.org/larryy/Yaeger.ALife3.pdf)
- Avida: 기본은 분열(무성). 유성 옵션으로 `RECOMBINATION_PROB 1.0`("Probability of recombination in div-sex"), `TWO_FOLD_COST_SEX`(0 = 자식 둘 다 태어남, 1 = 재조합 자식 하나만) 등. 변이 `COPY_MUT_PROB 0.0075`(복사당), `DIVIDE_INS_PROB 0.05`, `DIVIDE_DEL_PROB 0.05`. **[요약기]** — [avida.cfg](https://raw.githubusercontent.com/devosoft/avida/master/avida-core/support/config/avida.cfg)
- 유성 vs 무성 실험 증거(로봇 뇌): Luo, Longhi, Eiben(2023)은 모듈형 로봇의 뇌 번식에서 Darwinian 체계에서는 유성생식이 더 나았고 Lamarckian 체계에서는 무성생식이 더 나았다고 보고. **[스니펫]** — [arXiv 2303.12594](https://arxiv.org/abs/2303.12594)
- The Bibites 0.6: 알을 한 번에 큰 에너지로 낳지 않고 점진적으로 만들어 womb에 저장. **[스니펫]** — [Bibites 0.6.0 devlog](https://thebibites.itch.io/the-bibites/devlog/809631/the-bibites-060-organs-and-science)

### Inferences
- Inria 논문에는 번식의 에너지 비용이 적혀 있지 않다(문턱 위로 140스텝 유지가 조건일 뿐). 즉 자식의 시작 에너지 `E_0 = 3`은 부모에게서 빠지지 않는 것으로 읽힌다 — Polyworld가 "free energy source"라고 부른 구조와 같고, 이것이 Inria의 지수적 인구 증가·붕괴에 기여했을 수 있다. 1단계에서는 반드시 `child.E = f · parent.E; parent.E -= child.E`(+ 선택적 고정 번식 비용)로 닫아야 한다.
- JaxLife 수치로 읽으면: 번식 비용 최대 32, 그 2배(64)를 가져야 번식, 자식 시작 에너지 32 — "가진 것의 절반을 떼어 준다"에 해당한다. Polyworld 기본(0.2~0.8, 부모 둘의 합)과 합치면 "문턱 = 자식 몫의 2배, 전달 비율 ≈ 0.5"가 선례의 중심값이다.
- 고정 위상 신경망에서는 교차가 어렵다는 것이 JaxLife의 명시적 이유이고, 유성생식이 이 규모(수백~수천 개체, 직접 가중치 부호화)에서 진화를 더 좋게 한다는 증거는 찾지 못했다. Luo et al.의 결과는 조건에 따라 뒤집힌다. 1단계는 무성으로 시작하는 것이 선례와 맞다. 유성을 넣으면 짝 찾기라는 추가 문제가 생기고(Polyworld에서 "짝이 가까이 있다"가 정착·뭉침 행동의 주된 이득이었음), 저밀도에서 멸종 위험이 커진다.
- 자식을 부모 자리에 놓으면(Inria·Avida 이웃 배치) 친족이 한곳에 몰려 국소 경쟁이 세진다. 연속 공간에서는 부모 몸 반지름 1~2배 거리의 무작위 방향이 자연스러운 대응이다(충돌 겹침 방지).
- 변이 크기: 가중치당 N(0, σ)에서 σ = 0.02(Inria, 2445 parameters) / 0.01(JaxLife). 이 범위를 기본으로 두고, Polyworld처럼 변이율 자체를 유전자로 두는 것은 나중 단계로 미뤄도 된다.

### Gaps
- Inria 코드에서 번식 때 부모 에너지가 줄어드는지 직접 확인하지 못했다(논문 서술만으로 한 추론).
- JaxLife에서 자식이 받는 에너지와 부모가 잃는 에너지의 정확한 식, 자식 배치 위치는 `world.py` 미확인.
- The Bibites의 번식 방식(무성 여부), 알 에너지·부화 시간 기본값은 확인 못 함.
- 연속 2D 세계의 신경망 개체에서 유성 vs 무성을 직접 비교한 실험은 찾지 못했다(가장 가까운 것이 로봇 뇌 실험).

---

## 6. "세대"의 정의와 세는 법

### Takeaway
연속 시스템에서 쓰이는 정의는 네 가지다: (a) 계보 깊이(자식 = 부모 + 1)의 살아 있는 개체 평균, (b) 출생 수 ÷ 인구(steady-state GA의 "인구만큼의 평가 = 1세대"), (c) 인구학의 세대 시간(부모가 자식을 낳을 때의 평균 나이), (d) 수명 기반 회전 시간. Yaeger는 사실상 (c)에 가까운 "첫 자식까지의 시간"으로 세대를 셌다. "하루 10세대"는 (a)의 평균 계보 깊이가 하루에 10 오르는 것으로 재는 것이 가장 직접적이다.

### Cited Findings
- Avida — 계보 깊이 방식: `GENERATION_INC_METHOD` "0 = Only increase generation of offspring on divide. 1 = Increase generation of both parent and offspring". 시간 단위 update는 `AVE_TIME_SLICE 30`("Average number of CPU-cycles per org per update"). **[요약기]** — [avida.cfg](https://raw.githubusercontent.com/devosoft/avida/master/avida-core/support/config/avida.cfg)
- Avida에서 세대는 겹친다: "generations are overlapping in Avida (as they are in most biological populations) so that the same genotype often straddles one (or even more) generations". **[스니펫]** — [arXiv 1408.3651](https://arxiv.org/pdf/1408.3651)
- Steady-state GA(ECJ): "A 'generation' is defined as a Population's worth of evaluations." **[스니펫]** — [ECJ SteadyStateEvolutionState 문서](https://people.cs.gmu.edu/~eclab/projects/ecj/docs/classdocs/ec/steadystate/SteadyStateEvolutionState.html)
- 인구학의 세대 시간 정의 세 가지: (1) 인구가 R₀배가 되는 시간 `T = ln R₀ / r`; (2) 부모와 자식의 평균 나이 차 `T = ∫₀^∞ x e^(−rx) ℓ(x) m(x) dx`; (3) cohort 세대 시간 `T = ∫₀^∞ x ℓ(x) m(x) dx / ∫₀^∞ ℓ(x) m(x) dx`(분모 = R₀). ℓ(x)는 생존 함수, m(x)는 나이별 출산율. — [Wikipedia: Generation time](https://en.wikipedia.org/wiki/Generation_time)
- Polyworld 1994의 세대 환산: "With an average life span of about 500 time-steps, and a time-to-first-offspring of about 100 timesteps, this means that 500 generations can be run at this complexity in about 1 week." 조건은 300 개체 이상, 개체당 최대 약 200 뉴런, time-step당 약 13초(SGI Iris 4D/240-GTX). **[직접]** — [Yaeger 1994, §2](https://shinyverse.org/larryy/Yaeger.ALife3.pdf)
- Polyworld의 "자립" 지표: born / (born + created) 비율 — 0.0에서 시작해 성공한 실행은 1.0으로 수렴. **[직접]** — [Yaeger 1994, §9](https://shinyverse.org/larryy/Yaeger.ALife3.pdf)
- Inria: 비에피소드 방식은 "removing the need for a concept of generation"이라고 하고 세대를 정의하지 않는다. 대신 수명 지표(500스텝 창에서 죽은 개체의 평균 나이)를 쓴다. 그런데도 본문은 "collapse … after several generations"라고 쓴다(정의 없이). **[직접]** — [arXiv 2302.09334, §2·App. B.3·C](https://arxiv.org/abs/2302.09334)

### Inferences
- Yaeger의 환산을 검산하면: 1주 = 604,800초 ÷ 13초 = 약 46,500스텝, ÷ 500세대 = 세대당 약 93스텝 ≈ "time-to-first-offspring 약 100스텝". 즉 Yaeger의 "세대"는 평균 수명(500)이 아니라 첫 자식까지의 시간이다. 수명으로 세면 같은 실행이 약 93세대밖에 안 된다 — 정의에 따라 5배 차이가 난다.
- 구현용 정의 네 가지와 계산법:
  - **G_depth (권장 주 지표)**: 개체마다 `gen` 저장, 시조 0, 출생 시 `child.gen = parent.gen + 1`(Avida 방식 0). 보고값 = 살아 있는 개체의 `mean(gen)`. "하루 10세대" = 24시간 동안 `mean(gen)`이 10 증가. 재시드된 개체는 가져온 유전체의 `gen`을 물려받게 해야 지표가 0으로 꺼지지 않는다.
  - **G_births**: 누적 출생 수 ÷ 평균 인구(ECJ 방식). 번식 전에 죽는 자식이 많으면 G_depth보다 크게 나온다 — 두 값의 비가 "태어난 개체 중 번식까지 가는 비율"의 간접 지표가 된다.
  - **T_parent**: 출생 사건마다 부모 나이를 기록해 창(window) 평균 — cohort 세대 시간 (3)의 표본 추정. 세대 수 = 경과 스텝 ÷ T_parent. 정상 상태에서는 d(mean gen)/dt ≈ 1/T_parent가 되어야 하므로 G_depth와 서로 검산된다.
  - **T_life**: 창 안에서 죽은 개체의 평균 나이(Inria의 Expectancy). 회전 시간이지 세대 시간이 아니다(Yaeger 예에서 5배 차이).
- "하루 10세대" 목표를 스텝으로 바꾸면: 한 세대에 쓸 수 있는 실제 시간은 86,400 ÷ 10 = 8,640초(2.4시간). 시뮬레이션이 초당 S스텝이면 T_parent ≈ 8,640·S 스텝. 예: S = 30이면 약 26만 스텝, S = 100이면 약 86만 스텝. 선례의 세대 길이(Polyworld 약 100스텝, Inria 최대 수명 650스텝)와 비교하면 이 목표는 세대당 스텝이 수백~수천 배 길다 — 즉 속도가 병목이 아니라 "수명·번식 문턱을 얼마나 길게 잡을지"가 설계 선택이며, 같은 계산 자원으로 하루 수백 세대도 가능하다. 목표가 "최소 10세대"인지 "정확히 10세대쯤"인지 정해야 한다.
- 최대 수명 L과 번식 가능 시점이 T_parent의 범위를 정한다: (번식 가능한 최소 나이) ≤ T_parent ≤ L. 목표 세대 길이를 맞추는 가장 직접적인 손잡이는 번식 에너지 문턱(먹이 유입 대비)과 최대 수명이다.

### Gaps
- Avida의 `GENERATION_INC_METHOD` 기본값과 "average generation" 통계의 정확한 계산식은 확인 못 함(설정 주석만 확인).
- ECJ 문서는 검색 스니펫으로만 확인.
- Geb(Channon)·Tierra·The Bibites가 세대를 어떻게 세어 보고하는지는 원문을 찾지 못했다(The Bibites는 개체별 generation 숫자를 보여 주는 것으로 알려져 있으나 출처를 열지 못해 인용하지 않음).
- 유성생식일 때 계보 깊이를 어떻게 정하는지(두 부모 중 max인지 평균인지)에 대한 표준 출처는 찾지 못했다.

---

## 7. 시간 단계: 스텝·수명·물리 안정성

### Takeaway
선례의 수명은 수백~천 스텝 규모이고(Inria 최대 650, Polyworld 500~1000), 번식 대기·굶는 시간은 그 1/5~1/3이다. 손으로 쓰는 물리는 semi-implicit Euler(속도 먼저, 위치 나중)를 고정 dt로 쓰는 것이 권고되며, explicit Euler는 dt = 1/100에서도 감쇠 스프링에서 에너지가 늘어난다.

### Cited Findings
- Inria 시간 수치: 총 1e6 timesteps, `Maximum age` 650, `T_reproduce` 140, `T_death` 200, 에너지 감소 0.025/스텝. "1e6 timesteps took 20 minutes on a single GPU". **[직접]** — [arXiv 2302.09334, App. B.4](https://arxiv.org/abs/2302.09334)
- Polyworld 시간 수치: 평균 수명 약 500 time-steps, 첫 자식까지 약 100 time-steps(1994 원문). 기본값 `MinLifeSpan` 500, `MaxLifeSpan` 1000, `MateWait` 25, `PreBirthCycles` 25, `MaxSteps` 30000. **[직접] + [요약기]** — [Yaeger 1994, §2](https://shinyverse.org/larryy/Yaeger.ALife3.pdf); [worldfile.wfs](https://raw.githubusercontent.com/polyworld/polyworld/master/etc/worldfile.wfs)
- Polyworld 속도(1994): 300 개체 이상·최대 약 200 뉴런에서 time-step당 약 13초, 약 100 개체에서 프레임당 약 4초. **[직접]** — [Yaeger 1994, §2](https://shinyverse.org/larryy/Yaeger.ALife3.pdf)
- JaxLife: 2^16 timesteps 실행(설정 `NUM_WORLD_STEPS` 512 × `NUM_OUTER_STEPS` 128), 단일 NVIDIA A100. **[직접](본문) + [요약기](설정)** — [arXiv 2409.00853, Experiments 절](https://arxiv.org/abs/2409.00853); [JaxLife src/main.py](https://raw.githubusercontent.com/luchris429/JaxLife/main/src/main.py)
- Avida: 수명 상한은 genome_length × `AGE_LIMIT`(20) 명령, update 하나는 개체당 평균 30 CPU-cycle. **[요약기]** — [avida.cfg](https://raw.githubusercontent.com/devosoft/avida/master/avida-core/support/config/avida.cfg)
- The Bibites: 배속을 10배로 하면 노화도 10배 빨라진다(노화가 시뮬레이션 시간 기준). **[스니펫]** — [Bibites wiki: Ageing](https://the-bibites.fandom.com/wiki/Ageing)
- 적분 순서: explicit Euler는 `position += velocity * dt; velocity += acceleration * dt;`, semi-implicit Euler는 `velocity += acceleration * dt; position += velocity * dt;`. 감쇠 스프링 예에서 explicit Euler는 dt = 1/100에서 불안정("it gains energy over time!"), semi-implicit Euler는 같은 dt에서 정확한 해에 매우 가깝게 안정. 저자 권고: "My recommendation is semi-implicit euler." — [Gaffer On Games: Integration Basics](https://gafferongames.com/post/integration_basics/)

### Inferences
- 선례의 비율: 번식 대기/최대 수명 = 140/650 ≈ 0.22(Inria), 첫 자식/평균 수명 = 100/500 = 0.2(Polyworld). 굶어 죽는 시간/최대 수명 ≈ 320/650 ≈ 0.5(Inria, §4 계산). 1단계에서 수명 L을 정하면 번식 가능 시점 ≈ 0.2 L, 굶는 시간 ≈ 0.3~0.5 L이 선례와 같은 모양이다.
- §6의 계산대로 "하루 10세대"는 세대당 수십만 스텝을 허용하므로, 수명을 선례(수백~천 스텝)보다 훨씬 길게 잡을 수 있다. 수명을 길게 잡는다면 굶는 시간·번식 대기도 같은 비율로 늘려야 선택 압력의 모양이 유지된다(에너지 감소율 = 저장량 ÷ 굶는 시간).
- 격자 선례(Inria, JaxLife, Avida)는 1스텝 = 1칸 이동이라 물리 안정성 문제가 없다. 연속 공간에서 dt의 제약은 (1) 충돌: 한 스텝 이동 거리 v_max·dt가 몸 반지름보다 작아야 서로 뚫고 지나가지 않음, (2) 마찰·반발을 스프링처럼 구현할 때의 적분 안정성이다. (2)는 semi-implicit Euler로 완화되고, 마찰은 `v *= (1 − μ·dt)` 꼴에서 μ·dt < 1이어야 한다.
- 뇌는 매 물리 스텝마다 돌릴 필요가 없다. 물리 dt를 작게, 뇌·감각 갱신을 k 스텝마다 하면 CPU 비용(뇌 순전파가 대부분)을 줄이면서 물리 안정성을 지킬 수 있다. 수명·세대는 "뇌 스텝" 기준으로 세어야 선례 수치와 비교가 된다.

### Gaps
- 연속 2D 물리 인공생명(The Bibites 등)의 실제 물리 dt, 초당 스텝 수, 수명(초·스텝) 기본값은 확인 못 함(위키 차단).
- Polyworld의 물리 스텝과 뇌 스텝의 관계(1:1인지)는 확인 못 함.
- numpy·CPU 12코어에서 수백~수천 개체의 초당 스텝 수 실측 선례는 찾지 못했다 — 직접 재야 한다.
- "시뮬레이션 1초 = 몇 스텝"을 명시한 선례는 이번에 연 자료에는 없었다(모두 스텝 단위로만 서술).
