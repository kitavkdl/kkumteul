# 몸·움직임·충돌/먹기·제한된 감각 — 2D 연속 공간 인공생명 1단계 (numpy/numba, CPU)

읽는 법
- "직접 읽음" = 이번 조사에서 원문(논문 본문·소스 코드)을 열어 숫자를 확인한 것. "요약만 봄" = 검색 요약으로만 보고 원문을 열지 못한 것(숫자를 그대로 믿지 말 것).
- Evolvio 숫자는 carykh 원본 저장소(`carykh/evolvio`, `carykh/EvolvioColor` 모두 404)가 아니라 Java 포팅 포크 `FSMaxB/evolvio`에서 읽었다. 원본과 상수가 같은지는 확인하지 못했다.
- The Bibites 공식 위키(fandom)는 자동 접근이 막혀(HTTP 402) 열지 못했다. Bibites 내용은 v0.3.0 세이브 파일(gist) 하나에서 읽은 유전자·뉴런 이름과 값뿐이다. 수식은 없다.
- 성능 수치 중 "직접 측정"은 이번 조사에서 이 맥북(Apple M3, numba 0.67 스레드 8, numpy 2.2.6, scipy 1.15.3)에서 돌린 것이다. 대상 서버(12코어 AVX2)에서 잰 값이 아니다. 스크립트는 임시 폴더 (세션 임시 폴더)에 있고 세션이 끝나면 사라질 수 있다(핵심 알고리즘은 아래 본문에 적어 둠).

## 1. 몸 모양: 원 하나 대 여러 마디, 그리고 유전되는 몸 형질

### Takeaway
열어 본 1단계급 시스템(Evolvio, Bibites, biosim4, JaxLife, Olson 2013, eco-evo 격자)은 전부 "몸 = 점·격자 한 칸·원 하나"이고, 여러 마디 몸은 Framsticks·ALIEN처럼 형태 진화 자체가 목적인 시스템에만 있다. 유전 형질로는 크기·힘·최고 속도·시야(각/반경 또는 시각 뉴런 수)·색이 반복해서 나온다(Polyworld, Bibites).

### Cited Findings
- Polyworld 유전자 중 처음 8개가 생리(physiology)를 정한다. 목록 앞부분: size, strength, maximum speed, ID(색), mutation rate, number of crossover points, lifespan, fraction of energy to offspring, 그 뒤에 "number of neurons devoted to red component of vision" 등 시각 뉴런 수(직접 읽음) — [Yaeger 1994 텍스트본](http://textfiles.com/computers/CYBERSPACE/yaegerxa.txt)
- Polyworld에서 크기는 에너지 저장 한도를 선형으로 정한다: `ECap = minECap + (size - minSize) * (maxECap - minECap) / (maxSize - minSize)`. 같은 선형 관계로 크기가 전진·회전 에너지 소모(maximum-size-penalty)와 싸움 이점(maximum-size-advantage)을 정한다(직접 읽음) — [Yaeger 1994](http://textfiles.com/computers/CYBERSPACE/yaegerxa.txt)
- Polyworld의 strength는 한 스텝 총 에너지 소모 전체에 곱해지고 "typically 0.5 to 2.0" 범위이며, 공격자의 strength가 피해량에도 곱해진다. 논문은 "작고 약하면 에너지가 절약되고, 크고 강하면 포식에 유리하다"는 맞교환을 틈새(niche)를 만들려는 의도로 설명한다(직접 읽음) — [Yaeger 1994](http://textfiles.com/computers/CYBERSPACE/yaegerxa.txt)
- Polyworld에서 최고 속도 유전자는 몸의 길이·너비 표시에만 반영된다("Purely for the purposes of display", 길이×√maxspeed, 너비÷√maxspeed). 즉 모양은 보이기 위한 것이고 물리적 몸은 단순하다(직접 읽음) — [Yaeger 1994](http://textfiles.com/computers/CYBERSPACE/yaegerxa.txt)
- Evolvio(포크)는 몸이 원 하나이고 반지름·질량이 유전자가 아니라 현재 에너지에서 나온다: `ENERGY_DENSITY = 1/(0.2*0.2*PI)`, `getRadius() = sqrt(energy/ENERGY_DENSITY/PI)`(= 0.2·√energy), `getMass() = energy/ENERGY_DENSITY*density`. 주석: "set so when a creature is of minimum size, it equals one"(직접 읽음, 포크) — [FSMaxB/evolvio SoftBody.java](https://github.com/FSMaxB/evolvio/blob/master/src/main/java/io/evolvio/SoftBody.java)
- The Bibites v0.3.0 세이브 파일의 유전자 24개: LayTime, BroodTime, HatchTime, SizeRatio, SpeedRatio, ColorR/G/B, Strength, MutationAmountSigma, AverageMutationNumber, BrainMutationSigma, BrainAverageMutation, ViewAngle(표본 값 195.9), ViewRadius(93.4), ClockSpeed, PheroSense(163.6), Diet, ImmuneSystemStrength, HerdSeparationWeight, HerdAlignmentWeight, HerdCohesionWeight, HerdVelocityWeight, HerdSeparationDistance. 물리 몸은 강체 하나(`rb2d` px, py, vx, vy, m=0.659, r)이고 `body.d2Size`와 `rb2d.m`이 같은 값이다(직접 읽음) — [Bibites 세이브 파일 gist](https://gist.github.com/AndisGrossteins/449d1ac2bb7eeae198160a856d199d80)
- biosim4는 몸이 격자 한 칸이다(`Grid`는 16비트 인덱스 2D 배열, 0=빈칸). 기본값 sizeX=128, sizeY=128, population=3000, stepsPerGeneration=300. 몸 형질 유전자는 없고 게놈은 뇌 배선만 담는다(직접 읽음) — [biosim4 README](https://github.com/davidrmiller/biosim4/blob/main/README.md), [biosim4.ini](https://github.com/davidrmiller/biosim4/blob/main/biosim4.ini)
- JaxLife 에이전트는 연속 (x, y) 위치만 있고 MOVE_X, MOVE_Y로 움직인다. 몸 크기·충돌 처리는 논문에 없다(직접 읽음) — [JaxLife 논문](https://arxiv.org/html/2409.00853v1)
- Olson 외(2013) 포식자–먹이 모델은 방향이 있는 점 에이전트다(먹이 50, 포식자 1, 512×512 닫힌 공간)(직접 읽음) — [Olson et al. arXiv 1209.3330](https://arxiv.org/pdf/1209.3330)
- Framsticks는 막대(stick)들을 굽힘·회전 근육 두 종류로 이은 몸이고 근육이 에너지를 쓴다(요약만 봄) — [Framsticks 소개 논문(Emerald)](https://www.emerald.com/k/article/264074/Framsticks)
- ALIEN v4는 여러 세포(센서·근육·공격 세포 등)로 된 몸이고, 공식 문서 저장소는 "v4의 새 모델을 설명하지 않는 옛 문서"라고 스스로 밝힌다(요약만 봄) — [alien-docs](https://github.com/chrxh/alien-docs), [ALIEN v4.0 릴리스 노트](https://chrxh.itch.io/alien/devlog/582289/version-40-released)

### Inferences
- 원 하나가 잃는 것(위 사례들의 차이에서 추론): (1) 형태 진화(팔다리·입 위치·몸 비율)가 없다 — Polyworld도 모양은 표시용일 뿐이다. (2) 마찰 없는 원끼리 충돌은 회전을 만들지 않아 "밀기·붙잡기·몸으로 막기" 같은 접촉 행동의 여지가 좁다. (3) 몸 부위별 기능 분화(ALIEN의 센서/근육/공격 세포)가 없다. 반대로 얻는 것은 충돌 검사가 거리 비교 한 번이고 배치 계산이 쉽다는 점이다.
- 원 하나로 시작하더라도 "앞"은 필요하다: heading θ를 상태로 두고 입·시야·공격 범위를 앞쪽에 두면(Evolvio의 FIGHT_RANGE, biosim4의 `lastMoveDir` 앞 칸 공격, Olson의 정면 180° 망막) 방향 있는 몸의 효과 대부분을 얻는다.
- 1단계 기본 유전 형질 후보(Polyworld·Bibites 공통분모): 반지름(size), 최대 추력 또는 속도 계수, 시야 각·시야 거리(또는 광선 수), 색. 크기를 에너지에서 파생시키는 Evolvio 방식은 유전자 수를 줄이지만 "작게 태어나 자란다"와 "크기 진화"를 섞어 버린다. Polyworld처럼 크기 → 저장 한도·이동 비용·싸움 이점에 선형으로 묶는 편이 맞교환이 분명하다.
- Bibites의 ViewAngle 표본 값 195.9는 단위가 도(°)로 보이고 ViewRadius 93.4는 세계 단위로 보이지만 위키를 못 열어 단위는 추정이다.

### Gaps
- Bibites에서 SizeRatio·SpeedRatio·ViewAngle·ViewRadius가 에너지 비용에 어떻게 들어가는지(수식) 확인 못 함 — 공식 위키 접근 불가(HTTP 402).
- Geb(Channon), Critters, Framsticks 2D 모드의 몸 표현은 원문을 열지 못했다.
- carykh 원본 Evolvio 저장소를 찾지 못해 포크 상수가 원본과 같은지 미확인.
- Polyworld의 minSize/maxSize, maximum-size-penalty 등 실제 숫자는 논문에 없고 worldfile에 있다(이번에 worldfile은 열지 않음).

## 2. 움직임 모델: 갱신 식, 적분, 끌림, 이동 에너지 비용

### Takeaway
열어 본 시스템은 "전진 추력 + 회전" 두 출력이 표준이고(Polyworld, Evolvio, Bibites), 속도에 매 스텝 마찰 계수를 곱하는 식의 감쇠를 쓴다. 적분은 속도를 먼저 갱신하고 그 속도로 위치를 갱신하는 semi-implicit Euler가 권장이며, 이동 비용은 대부분 "출력 활성(=힘) × 상수"의 선형이다. 생물학의 단순화는 "거리당·질량당 비용 ∝ M^-0.32(달리기), 유체 저항 일률 ∝ v³"이다.

### Cited Findings
- Evolvio(포크) 가속: `multiplied = amount*timeStep/getMass(); vx += cos(rotation)*multiplied; vy += sin(rotation)*multiplied;` 비용은 전진이면 `amount*ACCELERATION_ENERGY*timeStep`, 후진이면 `|amount*ACCELERATION_BACK_ENERGY*timeStep|`. 상수 ACCELERATION_ENERGY=0.03, ACCELERATION_BACK_ENERGY=0.05(직접 읽음, 포크) — [FSMaxB/evolvio Creature.java](https://github.com/FSMaxB/evolvio/blob/master/src/main/java/io/evolvio/Creature.java)
- Evolvio(포크) 회전: `vr += 0.04*amount*timeStep/getMass();` 비용 `|amount*TURN_ENERGY*energy*timeStep|`, TURN_ENERGY=0.01. 회전 비용은 현재 에너지(=몸 크기)에 비례한다(직접 읽음, 포크) — [Creature.java](https://github.com/FSMaxB/evolvio/blob/master/src/main/java/io/evolvio/Creature.java)
- Evolvio(포크) 위치·마찰: `px = xBodyBound(px+vx*timeStep); vx *= (1-FRICTION/getMass());` (vy, vr도 같음), `rotation += vr;`, FRICTION=0.03. 마찰 곱은 timeStep으로 스케일되지 않고, 경계는 위치만 [radius, boardWidth−radius]로 자른다(직접 읽음, 포크) — [SoftBody.java](https://github.com/FSMaxB/evolvio/blob/master/src/main/java/io/evolvio/SoftBody.java), [Creature.java](https://github.com/FSMaxB/evolvio/blob/master/src/main/java/io/evolvio/Creature.java)
- Evolvio(포크) 기초대사: `loseEnergy(energy*METABOLISM_ENERGY*timeStep)`, METABOLISM_ENERGY=0.004. 물 타일 위에서는 스텝마다 `SWIM_ENERGY*energy`(0.008)를 더 잃는다. 대사가 현재 에너지에 비례하므로 큰 몸이 더 낸다(직접 읽음, 포크) — [Creature.java](https://github.com/FSMaxB/evolvio/blob/master/src/main/java/io/evolvio/Creature.java)
- Polyworld 이동: 전진량은 moving 뉴런 활성에 비례, 회전량은 turning 뉴런 활성에 비례(속도 직접 제어). 한 스텝 총 소모 = Σ(행동 뉴런 활성 0~1 × 그 행동의 에너지 변환 계수; 먹기·짝짓기·싸움·이동·회전·초점·발광) + 신경 소모(뉴런 수·시냅스 수에 선형) + 고정 소모, 이 합에 strength를 곱한다. 이동·회전 비용에는 크기 벌점이 선형으로 붙는다(직접 읽음) — [Yaeger 1994](http://textfiles.com/computers/CYBERSPACE/yaegerxa.txt)
- Bibites v0.3.0 출력 뉴런: Accelerate(TanH), Rotate(TanH), Herding, Want2Lay, Want2Eat, Want2Sex, Grab, ClkReset, PhereOut1~3, Want2Grow, Want2Heal, Want2Attack, ImmuneSystem. 움직임은 가속·회전 두 출력이다(직접 읽음) — [Bibites 세이브 파일 gist](https://gist.github.com/AndisGrossteins/449d1ac2bb7eeae198160a856d199d80)
- biosim4 이동: 여러 이동 뉴런의 X·Y 성분을 합쳐 tanh로 −1..1로 만들고, 절댓값을 확률로 써서(`prob2bool`) 축마다 −1/0/1칸 이동을 정한다. 에너지 비용은 없다. 반응성 곡선: `pow((r - 2.0), -2.0*k) - pow(2.0, -2.0*k) * (1.0 - r)`, k=responsivenessCurveKFactor=2(직접 읽음) — [biosim4 executeActions.cpp](https://github.com/davidrmiller/biosim4/blob/main/src/executeActions.cpp), [biosim4.ini](https://github.com/davidrmiller/biosim4/blob/main/biosim4.ini)
- Olson 외(2013): 먹이의 행동은 4개 이산값 — 정지, 전진 1단위, 왼쪽 8° 돌며 전진 1, 오른쪽 8° 돌며 전진 1. 포식자는 먹이보다 3배 빠르고 스텝당 6°만 돈다(속도–민첩성 맞교환을 손으로 넣음)(직접 읽음) — [Olson et al.](https://arxiv.org/pdf/1209.3330)
- JaxLife: "Energy is consumed when performing any action"이고 소모율은 현재 지형 칸에 따라 달라진다. 힘·속도·끌림·dt 수치는 논문에 없다(직접 읽음) — [JaxLife 논문](https://arxiv.org/html/2409.00853v1)
- eco-evo 격자(Hamon 외): 행동 5개(정지, 상하좌우), 에너지는 스텝당 0.025 선형 감소, 먹으면 +1, 최대 3, 수명 한도 650스텝. 움직임 자체의 추가 비용은 없다(직접 읽음, ar5iv 요약 변환본) — [Eco-evolutionary Dynamics of Non-episodic Neuroevolution](https://ar5iv.labs.arxiv.org/html/2302.09334)
- 적분 순서: explicit Euler는 `position += velocity*dt; velocity += acceleration*dt;`이고 감쇠 스프링(dt=1/100, k=15)에서 "gains energy over time". 두 줄 순서만 바꾼 semi-implicit(symplectic) Euler `velocity += acceleration*dt; position += velocity*dt;`는 같은 계에서 안정적이고 평균적으로 에너지를 보존한다. 충분히 뻣뻣한 스프링에서는 dt를 줄여도 explicit Euler 불안정이 남는다(직접 읽음) — [Gaffer On Games, Integration Basics](https://gafferongames.com/post/integration_basics/)
- Catto: "smaller time steps are more effective than more iterations"(반복 횟수보다 서브스텝) — [Solver2D](https://box2d.org/posts/2024/02/solver2d/)
- 생물학(달리기): 순 대사 이동 비용(거리당·질량당)은 (body mass)^-0.32에 비례. 달리기 대사 일률은 속도에 선형(Taylor 외 1982)이라 순 이동 비용은 속도와 거의 무관하다(직접 읽음, 요약 변환) — [Alexander 2005, J Exp Biol 208:1645](https://journals.biologists.com/jeb/article/208/9/1645/9370/Models-and-the-scaling-of-energy-costs-for)
- 생물학(헤엄): 일률 ∝ (body mass)^0.5 × (speed)^2.5(층류 경계층) 또는 (body mass)^0.6 × (speed)^2.8(난류). 기생 저항 일률 = 0.5ρAv³C_D. 기하 상사면 표면적 A ∝ M^0.67(직접 읽음, 요약 변환) — [Alexander 2005](https://journals.biologists.com/jeb/article/208/9/1645/9370/Models-and-the-scaling-of-energy-costs-for)
- 생물학(비행): 최대 항속 속도에서 기계 일률 ∝ M^1.02, 대사 일률 ∝ M^0.83 — [Alexander 2005](https://journals.biologists.com/jeb/article/208/9/1645/9370/Models-and-the-scaling-of-energy-costs-for)

### Inferences
- 1단계 기본 갱신 식 제안(semi-implicit Euler, 배치 배열 그대로):
  - 입력: 추력 `a_f ∈ [-1,1]`, 회전 `a_t ∈ [-1,1]` (Polyworld·Evolvio·Bibites 공통 2출력)
  - `F = F_max * a_f * (cos θ, sin θ) + F_contact`
  - `v ← (v + dt * F / m) * exp(-c_lin * dt)`
  - `ω ← (ω + dt * τ_max * a_t / I) * exp(-c_ang * dt)`; `θ ← θ + dt * ω`; `x ← x + dt * v`
  - 끌림을 `exp(-c*dt)`로 곱하면 dt를 바꿔도 거동이 같다. Evolvio의 `v *= (1 - FRICTION/m)`은 dt와 무관하게 스텝마다 곱하므로 dt를 바꾸면 물리가 달라지고, `m < FRICTION`이면 계수가 음수가 되어 속도 부호가 뒤집힌다(코드에서 읽은 식으로부터의 추론).
  - 선형 끌림에서 종단 속도는 대략 `v_T ≈ F_max / (m * c_lin)`. 원하는 최고 속도에서 `c_lin`을 거꾸로 정하면 된다.
- 회전을 각속도·관성 없이 "θ += a_t * turn_rate * dt"로 직접 제어하는 Polyworld·Olson 방식이 가장 단순하고 뇌가 배우기 쉽다. 속도–회전 맞교환을 원하면 Olson처럼 "빠른 종은 덜 돈다"를 `turn_rate ∝ 1/속도계수`로 넣을 수 있다.
- 이동 비용 법칙 선택지와 근거:
  - (a) `cost = k * |a_f| * dt` (힘에 선형, 질량 무관) — Evolvio. 가장 단순. 무거운 몸은 같은 비용으로 덜 가속된다.
  - (b) `cost = k * |a_f| * size_factor * strength * dt` — Polyworld(크기 벌점 선형, strength 곱).
  - (c) 육상형 생물학 단순화: 거리당 비용 ∝ m^0.68(= m × m^-0.32), 즉 `cost ≈ k * m^0.68 * |v| * dt`.
  - (d) 유체형: `cost ≈ k * r * |v|^3 * dt`(A ∝ r, 2D에서 단면은 지름).
  - 제안: 비용은 실현된 속도가 아니라 **명령한 추력**에 매긴다(`k * F_max * |a_f| * dt`에 `m^α` 계수). 그러면 남에게 밀리거나 충돌로 생긴 속도에 비용·환급이 생기지 않아 에너지 장부가 물리 버그와 분리된다. F_max를 크기에 따라 키우고 비용 지수 α를 0.68 근처로 두면 "큰 몸은 거리당 총비용은 크지만 질량당은 싸다"는 생물학 경향을 흉내 낸다.
- 후진 비용을 전진보다 비싸게(Evolvio 0.05 대 0.03) 두는 것은 "앞"을 의미 있게 만드는 싼 장치다.
- dt: 열어 본 ALife 시스템 중 물리 dt를 명시한 곳은 없다. 게임 물리 글의 예시는 dt=1/100(Gaffer), 60 Hz(Catto)다. 1단계에서는 "한 스텝 최대 이동 거리 ≤ 최소 반지름의 절반"이 되도록 dt·최고 속도를 묶는 편이 수치보다 중요하다(3절 터널링).

### Gaps
- Bibites의 가속·회전 식, 끌림 계수, 이동 에너지 식 — 위키 접근 불가로 미확인. 세이브 파일에 drag/angularDrag 항목은 없었다.
- Evolvio의 `timeStep` 실제 값(연 단위 시간과의 관계)은 Board 코드를 열지 않아 미확인.
- Schmidt-Nielsen 1972 원문(PDF)은 받았지만 스캔본이라 회귀식 계수를 글자로 뽑지 못했다. 위 지수는 Alexander 2005 리뷰에서 온 것이다.
- 기초대사의 체중 스케일링(Kleiber 3/4 법칙 등)은 이번에 출처를 확인하지 않았다.
- 뇌 에너지 비용의 구체 숫자: Polyworld는 "뉴런 수·시냅스 수에 선형, 전역 변환 계수"라는 구조만 논문에 있고 값은 worldfile에 있다.

## 3. 원끼리 충돌: 배치 해소, 이웃 탐색 성능, 알려진 악용

### Takeaway
이웃 탐색은 수천 개 규모에서는 병목이 아니다(직접 측정: 10,000개에서 numba 셀 리스트 0.23 ms, cKDTree 1.9 ms, 순수 numpy 6.2 ms). 진짜 위험은 손으로 짠 물리의 에너지 누수다 — 진화는 적분 오차·충돌 보정력·큰 dt를 "공짜 에너지"로 쓴 기록이 여러 번 있으므로, 대사 에너지 장부를 운동 에너지와 분리하고 속도 상한·관통 보정 상한을 처음부터 넣어야 한다.

### Cited Findings
- Evolvio(포크) 충돌: 겹치면(`distance < combinedRadius`) `force = combinedRadius*COLLISION_FORCE; vx += ((px-collider.px)/distance)*force/getMass();` COLLISION_FORCE=0.1. 밀어내는 힘은 겹침 깊이에 비례하지 않고 dt로 스케일되지 않으며 스텝마다 더해진다(직접 읽음, 포크) — [SoftBody.java](https://github.com/FSMaxB/evolvio/blob/master/src/main/java/io/evolvio/SoftBody.java)
- Evolvio(포크) 광역 탐색: 타일마다 그 타일에 걸친 몸의 목록(`softBodiesInPositions[x][y]`)을 유지하고, 몸이 덮는 타일 범위(반지름×FIGHT_RANGE)가 바뀔 때만 목록을 고친다(균일 격자) — [SoftBody.java](https://github.com/FSMaxB/evolvio/blob/master/src/main/java/io/evolvio/SoftBody.java)
- Polyworld 광역 탐색: 생물·먹이·장벽을 한 축(x)으로 정렬한 이중 연결 리스트로 유지한다. "minimal maintenance overhead, yet rules out most non-intersections very well"이며 프레임 간 위치 일관성을 이용하는 정렬을 쓴다(1D sweep-and-prune)(직접 읽음) — [Yaeger 1994](http://textfiles.com/computers/CYBERSPACE/yaegerxa.txt)
- Catto soft constraint(접촉을 감쇠 스프링으로): `omega = 2*pi*hertz; a1 = 2*zeta + omega*timeStep; a2 = timeStep*omega*a1; a3 = 1/(1+a2); biasRate = omega/a1; massCoeff = a2*a3; impulseCoeff = a3;` 충격량 `-massCoeff*meff*(vn + biasRate*contactSeparation) - impulseCoeff*accumulatedImpulse`. 예시 값 zeta=1.0, hertz=5.0. "we should keep the Hertz at 30Hz or below for a 60Hz time step"(직접 읽음) — [Solver2D](https://box2d.org/posts/2024/02/solver2d/)
- Baumgarte 보정 `-meff*(vn + 0.2*contactSeparation/timeStep)`은 "a cheap, yet crude way to deal with overlap"이고 "can lead to energy creation and jitter if not tuned carefully". 위치만 고치는 NGS는 "doesn't affect kinetic energy"(위치 충격량 `-effectiveMass*(0.2*min(0, contactSeparation))`)(직접 읽음) — [Solver2D](https://box2d.org/posts/2024/02/solver2d/)
- Box2D v2.4.1 안전 상수: `b2_linearSlop 0.005`(허용 겹침), `b2_maxLinearCorrection 0.2`(한 번에 고치는 최대 거리), `b2_baumgarte 0.2`, `b2_maxTranslation 2.0`(한 스텝 최대 이동), `b2_maxRotation 0.5*pi`(한 스텝 최대 회전). 단위는 미터 기준(직접 읽음) — [box2d b2_common.h](https://github.com/erincatto/box2d/blob/v2.4.1/include/box2d/b2_common.h)
- 악용 1(적분 오차): Karl Sims의 헤엄 진화에서 단순 Euler 적분의 오차가 빠른 움직임에서 쌓였고, 생물들이 작은 몸 부위를 빠르게 떨어 "free energy"를 얻어 비현실적 속도로 나아갔다(직접 읽음) — [Lehman et al., The Surprising Creativity of Digital Evolution](https://arxiv.org/pdf/1803.03453)
- 악용 2(충돌 버그): Sims의 뛰기 진화에서 자기 몸 두 부위의 모서리를 특정하게 부딪히면 오류가 나 공중으로 튀어 올랐다(직접 읽음) — [Lehman et al.](https://arxiv.org/pdf/1803.03453)
- 악용 3(큰 dt + 관통 보정력): Cheney 외 연체 로봇 시뮬레이터는 세포 수가 적으면 dt를 키웠다. 생물이 몸을 몇 세포로 줄여 큰 dt를 얻었고, 바닥을 스텝 사이에 뚫고 들어간 뒤 물리 엔진의 보정력을 "free" energy로 받아 진동하며 이동했다. 고친 방법: 바닥 접촉 감쇠를 올리고, 최소 몸 크기를 올리고, dt 계산을 고쳤다(직접 읽음) — [Lehman et al.](https://arxiv.org/pdf/1803.03453)
- 악용 4(위치 에너지): Sims의 이동 진화에서 키 크고 뻣뻣한 몸이 넘어지면서 초기 위치 에너지로 속도를 얻었다. 고친 방법: 시작할 때 초기 자세의 위치 에너지를 풀어 주는 시간을 둠(직접 읽음) — [Lehman et al.](https://arxiv.org/pdf/1803.03453)
- 악용 5(에너지 장부): Polyworld의 "indolent cannibals"는 부모가 자식에게 자기 에너지를 넘기도록 강제하기 전의 세계에서 가장 흔했고 "exploiting an essentially free energy source"였다. 에너지 균형을 맞춘 뒤에는 세계 구석에서 가끔 나타나는 정도로 줄었다(직접 읽음) — [Yaeger 1994](http://textfiles.com/computers/CYBERSPACE/yaegerxa.txt)
- 악용 6(경계): Polyworld의 "edge runners"는 세계 가장자리를 따라 빙빙 돈다. 죽은 개체가 길 위에 먹이로 남고 짝도 쉽게 만나기 때문이다. 장벽이 있어도 가장자리·장벽에 모이는 형태로 남는다(직접 읽음) — [Yaeger 1994](http://textfiles.com/computers/CYBERSPACE/yaegerxa.txt)
- 직접 측정(이 맥북 M3, 대상 서버 아님): 반지름 1, 면적 점유율 0.05, 균일 무작위 위치, 접촉 쌍 찾기(셀 크기 = 지름), 7회 중앙값.

  | N | 세계 변 | 접촉 쌍 | scipy cKDTree(build + query_pairs, ndarray) | 순수 numpy 패딩 셀 표 | numba 셀 리스트(1스레드) |
  |---|---|---|---|---|---|
  | 1,000 | 251 | 89 | 0.12 ms | 0.44 ms | 0.021 ms |
  | 3,000 | 434 | 288 | 0.38 ms | 1.76 ms | 0.065 ms |
  | 10,000 | 793 | 1,042 | 1.90 ms | 6.17 ms | 0.227 ms |

  세 구현의 쌍 개수가 일치함을 확인했다 — 출처: 이번 조사의 벤치마크 스크립트(위 "읽는 법"의 임시 경로), 재현용 알고리즘은 아래 Inferences
- 다른 사람의 측정(요약만 봄): freud 논문은 입자당 이웃 약 12개인 무작위 점에서 자기 AABB 트리가 수천 점 이상에서 cKDTree를 앞선다고 보고한다 — [freud 논문](https://arxiv.org/pdf/1906.06317). 파이썬 포럼에는 3D 점 10,000개에 `query_pairs(r=0.05)`가 호출당 약 2.8초 걸렸다는 일화가 있다(3D, 조건 불명) — [discuss.python.org](https://discuss.python.org/t/computing-and-benchmarking-a-neighbour-list-search/10376)

### Inferences
- 측정에 쓴 알고리즘(재현용):
  - numba 셀 리스트: `cid = cx*G + cy` → 셀별 개수 세기 → `start = cumsum(cnt)` → counting sort로 `order` 채움 → 각 i에 대해 주변 3×3 셀의 `order[start[c]:start[c+1]]`를 돌며 `j > i`이고 거리² < (2R)²인 쌍 기록.
  - 순수 numpy 패딩 셀 표: `order = argsort(cid, stable)`; `rank = arange(N) - searchsorted(sc, sc, 'left')`; `table[sc, rank] = order`(모양 (셀 수+1, K), 빈칸 −1, 마지막 행은 범위 밖용 빈 셀) → `cand = table[이웃 9셀]`을 (N, 9K)로 펴서 한 번에 거리 계산. K(셀당 최대 개수)가 커지면 낭비가 커지므로 밀도가 고르지 않으면 불리하다.
- 수천 개 규모에서 선택: 충돌만 보면 cKDTree로도 충분하다(10k에서 1.9 ms). 다만 시야(4절)가 같은 격자를 다시 쓰므로 numba 셀 리스트 하나를 만들어 충돌·시야·먹기 판정이 공유하는 구성이 가장 싸다. 24시간 돌릴 것이면 numba를 처음부터 쓰는 편이 낫다(순수 numpy 대비 충돌 27배, 시야 4~20배 — 아래 4절 표).
- 배치 친화적 충돌 해소 후보:
  - (a) 위치 투영(PBD식): 겹침 `pen = r_i + r_j - d`를 `n = (x_i - x_j)/d` 방향으로 질량 역수 비율로 나눠 위치만 옮긴다(`np.add.at`으로 누적). Catto의 NGS와 같은 성격이라 운동 에너지를 건드리지 않는다. 한 번에 고치는 거리에 상한(Box2D는 0.2), 허용 겹침(0.005)을 둔다.
  - (b) 접근 속도 제거: 법선 방향 상대속도가 접근 중일 때만 그 성분을 없앤다(반발 계수 0~1). 반발 계수 ≤ 1이면 운동 에너지가 늘지 않는다.
  - (c) 벌점 스프링: `F = k * pen * n - c * v_n * n`. 구현이 가장 쉽지만 k·dt가 크면 explicit하게는 에너지가 생긴다(Gaffer의 스프링 예). Catto 식의 hertz·zeta 매개변수화를 쓰고 hertz를 스텝 주파수의 절반 이하로 둔다.
  - Evolvio식 "겹치면 일정한 속도 증분"은 깊이·dt와 무관하게 매 스텝 운동량을 넣으므로 그대로 베끼지 않는 편이 낫다.
- 악용 방어 목록(위 사례에서 도출):
  1. 대사 에너지와 운동 에너지를 다른 장부로 둔다. 충돌·보정·경계 처리로 대사 에너지가 생기는 경로를 0개로 만든다(먹기·광합성류 입력만 +).
  2. 이동 비용은 명령한 추력에 매기고 환급(음수 비용)을 금지한다.
  3. 속도 상한: `|v|*dt ≤ 0.5 * r_min`(터널링 방지; Box2D는 스텝당 이동 2.0 상한).
  4. 관통 보정 상한과 접촉 감쇠(Cheney 사례의 수정과 같은 방향), dt는 개체 상태와 무관하게 고정.
  5. 최소 몸 크기 하한(질량 → 0에서 `F/m`, `1 - c/m` 폭주 방지).
  6. 경계: 위치만 자르면(Evolvio) 벽에 붙어 미끄러지는 edge runner가 유리해진다. 토러스(wrap)로 하거나, 벽이면 법선 속도를 0으로 만들고 벽 근처 먹이 밀도가 높아지지 않게 한다. 토러스로 하면 Polyworld에서는 "frenetic joggers"(직진만 하는 종)가 나왔다는 점도 같이 기억한다.
  7. 세계 전체 에너지 합(먹이 + 개체 + 시체)을 매 N스텝 기록해 새는 곳을 감시한다(Polyworld도 에너지 보존 지표를 그래프로 그렸다).
  8. NaN/Inf 검사와 좌표 범위 검사를 주기적으로 한다(Lehman의 오버플로 사례).

### Gaps
- 대상 서버(12코어 AVX2)에서의 측정값 없음 — 위 표는 Apple M3 값이라 절대 수치는 다를 수 있다(상대 순위는 유지될 가능성이 높다는 것은 추정).
- 밀집 상황(면적 점유율 0.5 이상, 무리 짓기)에서의 성능·안정성은 재지 않았다.
- numba 셀 리스트와 cKDTree를 같은 조건으로 비교한 제3자 공개 벤치마크는 찾지 못했다.
- Bibites(Unity 물리)·JaxLife의 충돌 처리 방식은 확인 못 함.

## 4. 먹기와 공격: 접촉 기반 먹기, 에너지 이전, 시체, 처음부터 포식이 있을 때의 동역학

### Takeaway
열어 본 연속 공간 시스템은 모두 "겹침(접촉) + 행동 뉴런 활성"으로 먹기와 공격을 판정한다. Polyworld·Evolvio·Bibites는 처음부터 개체 간 공격이 있고, biosim4는 기본값으로 꺼 두었으며, JaxLife·eco-evo 격자는 없다. Polyworld에서는 포식이 "공짜 에너지원"이 되지 않도록 에너지 저장을 둘로 나눈 설계(health-energy / food-value-energy)와 부모→자식 에너지 이전이 핵심이었다.

### Cited Findings
- Polyworld 먹기: 먹으려면 위치가 먹이와 겹쳐야 하고, 먹는 양은 eating 뉴런 활성에 비례하며 활성이 임계값을 넘어야 한다. 먹기에도 에너지가 든다(직접 읽음) — [Yaeger 1994](http://textfiles.com/computers/CYBERSPACE/yaegerxa.txt)
- Polyworld 싸움: 공격자가 상대와 겹쳐야 하고 한쪽만 fight를 내도 성립한다. 피해 = (fight 뉴런 활성 × 공격자의 현재 health-energy × strength × size)에 전역 attack-to-energy 계수를 곱한 값. 양쪽 다 싸우면 역할을 바꿔 한 번 더 계산한다. 싸움 의지는 몸의 빨간색 성분으로 표시되어 다른 개체에게도 보인다(직접 읽음) — [Yaeger 1994](http://textfiles.com/computers/CYBERSPACE/yaegerxa.txt)
- Polyworld 에너지 두 종류: health-energy와 food-value-energy. 둘 다 먹으면 차고 행동·신경 활동으로 준다. 공격받으면 health-energy만 준다. health-energy가 0이면 죽고, 죽으면 food-value-energy만큼의 에너지를 가진 먹이가 된다(또는 지정한 최소값). 그래서 "죽였지만 먹을 값어치는 높게 남는" 포식이 자연스럽다. 뇌 입력으로 들어가는 것은 health-energy다(직접 읽음) — [Yaeger 1994](http://textfiles.com/computers/CYBERSPACE/yaegerxa.txt)
- Polyworld 번식 에너지: "fraction of energy to offspring"이 유전자다. 부모가 자식에게 자기 에너지를 넘기도록 하기 전에는 indolent cannibals(제자리에서 짝짓고 싸우고 죽이고 먹는 종)가 가장 흔했다. 저자는 시체 먹이를 완전히 없애도 이 종이 나타나는 것을 보고, 주된 이득은 먹이가 아니라 짝을 쉽게 구하는 것이라고 판단했다(직접 읽음) — [Yaeger 1994](http://textfiles.com/computers/CYBERSPACE/yaegerxa.txt)
- Polyworld 포식 동역학: 가장자리 없는 세계에서 안정한 "dervishes" 종이 수백 세대 유지되는 동안에도 "evolution continued to explore optimum degrees of predation"이었고, 싸움 표현 수준의 파동이 여러 집단을 휩쓸었다(직접 읽음) — [Yaeger 1994](http://textfiles.com/computers/CYBERSPACE/yaegerxa.txt)
- Evolvio(포크) 먹기: `amount = attemptedAmount/(1.0+speed)`("The faster you're moving, the less efficiently you can eat"), `foodToEat = tile.foodLevel*(1-pow(1-EAT_SPEED, amount*timeStep))`, EAT_SPEED=0.9. 입 색과 먹이 색의 차이로 `multiplier = 1 - foodDistance/FOOD_SENSITIVITY`가 정해지고 음수면 오히려 에너지를 잃는다. 먹기 시도 비용 `attemptedAmount*EAT_ENERGY*timeStep`, EAT_ENERGY=0.04. 음수 출력은 토하기(`dropEnergy`)다(직접 읽음, 포크) — [Creature.java](https://github.com/FSMaxB/evolvio/blob/master/src/main/java/io/evolvio/Creature.java)
- Evolvio(포크) 싸움: 성숙(MATURE_AGE=0.01년) 후에만 가능. 비용 `fightLevel*FIGHT_ENERGY*energy*timeStep`(FIGHT_ENERGY=0.03). 범위는 `getRadius()*FIGHT_RANGE + 상대 반지름`(FIGHT_RANGE=2.0) 안의 생물. 피해는 상대의 `dropEnergy(fightLevel*INJURED_ENERGY*timeStep)`(INJURED_ENERGY=0.25)이고, dropEnergy는 그 에너지를 상대가 덮은 타일의 먹이로 떨어뜨린다. 공격자에게 직접 넘어가지 않고 땅에 떨어진 것을 먹어야 한다(직접 읽음, 포크) — [Creature.java](https://github.com/FSMaxB/evolvio/blob/master/src/main/java/io/evolvio/Creature.java)
- Evolvio(포크) 번식 조건: Birth 출력 > 0, 나이 ≥ MATURE_AGE, energy > SAFE_SIZE(1.25). 자식에게 줄 수 있는 에너지는 `energy - SAFE_SIZE`(직접 읽음, 포크) — [Creature.java](https://github.com/FSMaxB/evolvio/blob/master/src/main/java/io/evolvio/Creature.java)
- Bibites v0.3.0: 출력에 Want2Eat, Want2Attack, Grab, Want2Heal, Want2Grow가 있고 입력에 MeatConcentrationWeight/Angle, NVisibleMeat, AttackedDamage가 있다. 몸에 energy/maxEnergy(44.3/98.9)와 health/maxHealth(65.9/65.9)가 따로 있고 Diet 유전자(0.495)가 있다. 고기(meat)가 식물 펠릿과 별도 범주로 처음부터 있다(직접 읽음) — [Bibites 세이브 파일 gist](https://gist.github.com/AndisGrossteins/449d1ac2bb7eeae198160a856d199d80)
- biosim4: KILL_FORWARD는 enum에서 NUM_ACTIONS 뒤에 있어 기본 빌드에서 꺼져 있고 `killEnable = false`가 기본이다. 켜면 출력이 임계 0.5를 넘을 때 그 값을 확률로 써서 `loc + lastMoveDir`(바로 앞 칸)의 개체를 죽음 대기열에 넣는다. 에너지 이전은 없다(직접 읽음) — [sensors-actions.h](https://github.com/davidrmiller/biosim4/blob/main/src/sensors-actions.h), [executeActions.cpp](https://github.com/davidrmiller/biosim4/blob/main/src/executeActions.cpp), [biosim4.ini](https://github.com/davidrmiller/biosim4/blob/main/biosim4.ini)
- JaxLife: 에이전트 행동은 MOVE_X, MOVE_Y, EAT, REPRODUCE, 메시지, 지형 바꾸기이고 에이전트 간 공격은 논문에 없다. 먹기는 에너지가 있는 지형 칸에 EAT를 하는 것이다(직접 읽음) — [JaxLife 논문](https://arxiv.org/html/2409.00853v1)
- eco-evo 격자(Hamon 외): 포식 없음. 자원만 먹는데도 개체 수와 자원이 포식자–먹이(Lotka–Volterra)형 진동을 보였고, 한 시드에서는 지도 대부분의 자원이 고갈되는 공유지의 비극에 가까운 상태가 나왔다(직접 읽음, ar5iv 요약 변환본) — [Eco-evolutionary Dynamics](https://ar5iv.labs.arxiv.org/html/2302.09334)
- Olson 외(2013) 공격 규칙: 포식자가 보이는 먹이의 5 virtual meter 안에 들어가면 자동 공격. 공격은 10스텝에 한 번(handling time). 혼란 효과가 켜지면 `Pcapture = 1/ANV`(ANV = 포식자 시야 안에 있고 목표 먹이의 30 virtual meter 안에 있는 먹이 수). 이 압력만으로 먹이의 무리 짓기가 진화했다. handling time이 아예 없을 때만 결과가 달라졌다(직접 읽음) — [Olson et al.](https://arxiv.org/pdf/1209.3330)
- Neural MMO 1.x: 공격 방식 Melee/Range/Mage가 처음부터 있고, 음식·물이 0이면 피해를 받기 시작하며 체력 0이면 죽는다(요약만 봄) — [Neural MMO 논문](https://arxiv.org/pdf/1903.00784)

### Inferences
- 1단계 먹기 기본 제안: 접촉(중심 거리 < r_agent + r_food) + Want2Eat류 출력 > 임계. 섭취량 = `min(먹이 에너지, bite_rate * 활성 * dt)`. Evolvio의 `1/(1+speed)` 감쇠를 넣으면 "달리면서 쓸어 먹기"(Polyworld의 frenetic joggers형 전략)를 약하게 만든다.
- 공격을 1단계에 넣는다면 Polyworld의 두 장치를 같이 넣어야 한다: (1) 체력(health)과 몸값(food-value)을 분리, (2) 번식 시 부모 에너지가 실제로 자식에게 이전(공짜 생성 금지). 그리고 공격에도 비용(Evolvio: 에너지 비례, Polyworld: 활성×계수×strength)과 쿨다운(Olson의 handling time 10스텝)을 둔다. 쿨다운이 0일 때만 동역학이 달라졌다는 Olson의 관찰은 "쿨다운은 있기만 하면 된다"는 뜻으로 읽을 수 있다.
- 에너지 이전 방식 두 가지: (a) 직접 이전(피해량의 일부가 공격자에게) — 장부가 단순하지만 접촉 공격이 곧 먹기라 포식이 너무 싸다. (b) 떨어뜨리기/시체(Evolvio: 타일에 떨어짐, Polyworld: 죽으면 먹이 조각) — 한 단계가 더 있어 청소부(scavenger) 틈새가 생기고, 시체도 일반 먹이 배열에 넣으면 구현이 같다. (b)가 1단계에 더 맞다.
- 포식을 처음부터 켜면 생기는 일(Polyworld 근거): 제자리 근친 포식 집단이 에너지 누수 지점을 찾아낸다. 누수가 없으면 이 행동은 드물어진다. 따라서 "포식을 켜기 전에 에너지 보존 검사를 통과"를 조건으로 삼을 만하다. biosim4가 kill을 기본으로 꺼 둔 것, JaxLife·eco-evo가 아예 넣지 않은 것은 포식 없이도 개체 수–자원 진동이 충분히 나오기 때문으로 보인다(eco-evo 결과).
- 시체 색을 살아 있는 개체·식물과 다르게 하면(Bibites의 meat 범주, Polyworld의 색 기반 시각) 감각만으로 구분할 수 있어 별도 "종류" 채널 없이도 된다.

### Gaps
- Bibites의 공격 피해·고기 에너지 환산·Diet 유전자 효율 식 미확인(위키 접근 불가).
- Polyworld의 attack-to-energy 계수, 먹기 임계값의 실제 숫자는 worldfile에 있고 이번에 열지 않았다.
- "포식을 처음부터 넣은 경우 대 나중에 넣은 경우"를 직접 비교한 통제 실험은 찾지 못했다. 위 동역학은 Polyworld 저자의 관찰 서술이다.
- ALIEN의 공격 세포 에너지 이전 규칙은 릴리스 노트 요약만 보았고 수식을 찾지 못했다.

## 5. 시각: 광선 망막, 1D 색 띠, 범주별 집계, 격자 패치 — 매개변수와 배치 구현, "흐릿하게" 만드는 법

### Takeaway
시각 설계는 네 갈래다: (1) 1D 색 띠(Polyworld: 22×22 렌더의 한 줄을 시각 뉴런 수만큼 축소, 수평 시야 20°~120°), (2) 부채꼴 이진 망막(Olson: 15°×12칸=180°, 거리 100/200), (3) 소수의 점 표본(Evolvio: 광선 3개), (4) 범주별 집계(Bibites: 종류마다 "무게·각도·개수"). numba로는 3,000개체 × 광선 16개가 1스레드 약 7 ms, 8스레드 약 1.4 ms였고(직접 측정, M3), 순수 numpy는 약 28 ms였다. "흐릿함"은 열어 본 시스템에서 잡음이 아니라 낮은 해상도·이진화·집계로 만든다.

### Cited Findings
- Polyworld 시각: 각 개체 시점에서 22×22 픽셀로 렌더한 뒤 "pixel row just above vertical center"를 그 개체의 시각 뉴런 수에 맞게 안티에일리어싱해 넣는다(1차원 색 띠). 수직 시야는 10° 고정, 수평 시야는 focusing 뉴런 활성으로 최소~최대("typically 20° to 120°") 사이에서 스스로 조절한다. 색 성분(R/G/B)별 시각 뉴런 수가 유전자다(직접 읽음) — [Yaeger 1994](http://textfiles.com/computers/CYBERSPACE/yaegerxa.txt)
- Polyworld는 시각 뉴런 수에 대해 "similar opposing pressures between energy expenditure and visual acuity"가 있다고 적는다(뉴런·시냅스 수에 비례하는 신경 에너지 비용 때문). 즉 시각 해상도가 비용이 붙은 진화 형질이다(직접 읽음) — [Yaeger 1994](http://textfiles.com/computers/CYBERSPACE/yaegerxa.txt)
- Polyworld는 색을 통신에도 쓴다: 싸움 의지 → 빨강, 짝짓기 의지 → 파랑, ID 유전자 → 초록 계열, lighting 뉴런 → 앞면 폴리곤 밝기 0~255. 저자는 시각으로 이동을 제어하는 증거는 관찰했지만 발광 통신의 사용 증거는 찾지 못했다고 적었다(직접 읽음) — [Yaeger 1994](http://textfiles.com/computers/CYBERSPACE/yaegerxa.txt)
- Olson 외(2013) 망막: 층 하나가 센서 12개, 센서 하나의 시야 15°(합 180° 정면), 거리 한도 먹이 100 / 포식자 200 virtual meter(세계는 512×512). 각 칸은 그 종류의 개체가 하나라도 있으면 1인 이진값이고 개수·거리는 알려 주지 않는다. 먹이는 2층(동종 감지, 포식자 감지), 포식자는 1층(먹이만). 뇌는 Markov Network이고 한 평가는 2,000스텝이다(직접 읽음) — [Olson et al.](https://arxiv.org/pdf/1209.3330)
- Evolvio(포크) 시각: 광선 3개 `visionAngles = {0, -0.4, 0.4}`(rad), `visionDistances = {0, 1.42, 1.42}`(타일 단위), MAX_VISION_DISTANCE=10. 각 광선의 끝점에서 색(hue, saturation, brightness)을 읽어 9개 입력이 되고, 도중의 다른 몸이 가리면 그 몸의 색을 본다(광선이 지나는 타일의 몸 목록만 후보로 모음). 입력 12개: 0Hue,0Sat,0Bri,1Hue,1Sat,1Bri,2Hue,2Sat,2Bri,Size,Mem,Const. 출력 12개: BHue,MHue,Accel.,Turn,Eat,Fight,Birth,How funny?,How popular?,How generous?,Mem,Const. 뇌는 3층×12(직접 읽음, 포크) — [Creature.java](https://github.com/FSMaxB/evolvio/blob/master/src/main/java/io/evolvio/Creature.java)
- Bibites v0.3.0 시각 입력(범주별 집계): BibiteConcentrationWeight, BibiteConcentrationAngle, NVisibleBibites, PelletConcentrationWeight, PelletConcentrationAngle, NVisiblePellets, MeatConcentrationWeight, MeatConcentrationAngle, NVisibleMeat, ClosestBibiteR/G/B. 시야는 ViewAngle·ViewRadius 유전자로 정해진다(직접 읽음; 집계 식은 미확인) — [Bibites 세이브 파일 gist](https://gist.github.com/AndisGrossteins/449d1ac2bb7eeae198160a856d199d80)
- biosim4 "시각"에 해당하는 것: LONGPROBE_POP_FWD / LONGPROBE_BAR_FWD(정면 한 줄을 longProbeDistance=16칸까지 탐침), POPULATION(반경 populationSensorRadius=2.5 안의 밀도), POPULATION_FWD / POPULATION_LR(전후·좌우 축 밀도), BARRIER_FWD / BARRIER_LR(shortProbeBarrierDistance=4). 탐침 거리는 SET_LONGPROBE_DIST 행동으로 스스로 바꾼다(`1 + level * maxLongProbeDistance`)(직접 읽음) — [sensors-actions.h](https://github.com/davidrmiller/biosim4/blob/main/src/sensors-actions.h), [biosim4.ini](https://github.com/davidrmiller/biosim4/blob/main/biosim4.ini), [executeActions.cpp](https://github.com/davidrmiller/biosim4/blob/main/src/executeActions.cpp)
- JaxLife 관측: 주변 영역의 지형 속성 + 가장 가까운 N^view_agents 에이전트와 N^view_bots 로봇의 속성 + 자기 속성. N 값과 영역 크기는 논문에 숫자가 없다. 개체별 인코더 → self-attention → 자기 임베딩과 cross-attention → LSTM(직접 읽음) — [JaxLife 논문](https://arxiv.org/html/2409.00853v1)
- eco-evo 격자 관측: 15×15 정사각(각 방향 7칸), 채널은 자원·다른 에이전트(수 포함)·벽. 합성곱 2층 → LSTM(은닉 4) → 밀집 8, 매개변수 총 2,445개(직접 읽음, ar5iv 요약 변환본) — [Eco-evolutionary Dynamics](https://ar5iv.labs.arxiv.org/html/2302.09334)
- Neural MMO: 매 틱 주변 15×15 타일 크롭과 그 안의 에이전트를 관측. 2.0은 15×15 타일 + 시야 안 최대 100개 개체, 지도 128×128(요약만 봄) — [Neural MMO 논문](https://arxiv.org/pdf/1903.00784), [Neural MMO 2.0 위키](https://neuralmmo.github.io/_build/html/rst/game_wiki.html)
- Beer의 minimally cognitive 에이전트: 지름 30의 원형 몸, π/6 호에 고르게 퍼진 거리 광선 7개, 광선 길이 265(다른 문헌은 220~0을 [0,10]으로 사상), 경기장 400×275(요약만 봄; 원 논문 미확인, 숫자 불일치 있음) — [Multifunctionality in embodied agents](https://arxiv.org/pdf/1802.03891)
- 광선–원 교차 식(표준): 원점 o, 단위 방향 u, 원 중심 c, 반지름 r에서 `d = u·(c−o) ± sqrt((u·(c−o))² − (|c−o|² − r²))`, 판별식이 음수면 빗나감(이번에 페이지를 열지는 않았고, 아래 측정에서 numpy 구현과 numba 구현의 결과가 일치함을 확인) — [Wikipedia: Line–sphere intersection](https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection)
- 직접 측정(이 맥북 M3, 대상 서버 아님): 반지름 1, 면적 점유율 0.05, 시야 거리 15, 시야각 120°, 격자 셀 크기 = 시야 거리(주변 3×3 셀 후보), 광선이 무언가에 맞은 비율 약 0.37.

  | N | 광선 수 | 순수 numpy(패딩 후보 표, (N,R,C) 브로드캐스트) | numba 1스레드 | numba prange 8스레드 |
  |---|---|---|---|---|
  | 1,000 | 8 / 16 / 32 | 3.7 / 6.4 / 12.0 ms | 1.07 / 2.19 / 3.49 ms | 0.30 / 0.48 / 0.75 ms |
  | 3,000 | 8 / 16 / 32 | 16.1 / 27.5 / 49.3 ms | 4.15 / 7.06 / 10.99 ms | 0.83 / 1.39 / 2.23 ms |
  | 10,000 | 8 / 16 / 32 | 54.6 / 92.5 / 167.3 ms | 14.24 / 23.97 / 37.08 ms | 2.69 / 4.61 / 7.46 ms |

  — 출처: 이번 조사의 벤치마크 스크립트(위 "읽는 법"의 임시 경로)

### Inferences
- 측정에 쓴 벡터화 식(재현용): 후보 `cand`(N, C)에 대해 `m = pos[cand] - pos[:,None,:]`; 광선 방향 `(dx, dy)`(N, R); `b = m_x*dx + m_y*dy`(N, R, C); `c2 = |m|² - r²`(N, C); `disc = b² - c2`; `t = b - sqrt(max(disc, 0))`; 유효 조건 `disc ≥ 0 & t > 0 & t ≤ range & cand ≠ self`; `t.argmin(-1)`로 가장 가까운 것. numba판은 같은 식을 3×3 셀 루프로 돌리고 `b ≤ 0`이면 건너뛴다.
- 비용 감각: 검사 횟수 = N × R × C(C = 시야 안 후보 수). 위 조건에서 numba 1스레드는 광선–원 검사 한 번에 수 ns 수준이다. 광선 수에 선형이므로 3,000개체·16광선·1스레드 7 ms면 물리 스텝당 예산의 대부분이 시각(과 뇌)이다. 시야 거리를 2배로 하면 후보 수가 4배가 된다.
- 광선보다 싼 대안 — 부채꼴 집계(Olson·Bibites형): 시야 안 후보 각각의 상대 각도 φ와 거리 d를 구해 칸 번호 `k = floor((φ + FOV/2) / (FOV/K))`에 `minimum.at`(가장 가까운 거리) 또는 `add.at`(개수·가중합)으로 모은다. 비용이 O(N × C)라 광선 수에 무관하다. 가림(occlusion)은 "칸마다 가장 가까운 것만 본다"로 근사된다. 원의 각 폭 `asin(r/d)`를 무시하면 가까운 큰 물체가 한 칸에만 잡히는 오차가 있다.
- 1단계 기본 시각 제안(근거가 된 사례 표시):
  - 정면 부채꼴 K칸(K는 유전 가능, 기본 5~8; Olson은 12칸×15°, Evolvio는 3개, Polyworld는 색별 뉴런 수 유전자).
  - 칸마다 채널 2~4개: 가장 가까운 물체까지의 근접도 `1 - d/range`(없으면 0)와 그 물체의 색 1~3성분(Polyworld·Evolvio는 색만 주고 의미는 뇌가 배운다).
  - 시야각 FOV와 거리 range도 유전 형질로 두고(Bibites의 ViewAngle·ViewRadius), 비용은 "입력 뉴런 수에 비례하는 뇌 비용"(Polyworld 방식) + `range`에 비례하는 유지비를 더한다. FOV × range²(본 면적)에 비례시키는 것도 방법이다 — 비용 식 자체는 출처 없는 설계 제안이다.
- "좁고 흐릿하게" 만드는 수단(열어 본 사례에서):
  - 이진화·개수 숨김(Olson: 칸에 몇 마리 있는지 모름),
  - 저해상도 축소(Polyworld: 22픽셀 한 줄을 뉴런 몇 개로 평균),
  - 점 표본(Evolvio: 광선 끝점 3곳의 색만),
  - 집계(Bibites: 범주별 "무게·각도·개수"),
  - 거리 한도(Olson 100/200, biosim4 16칸),
  - 주의 맞교환(Polyworld: 넓게 볼지 좁게 볼지 스스로 정함; 뉴런 수가 고정이라 넓게 보면 각 해상도가 떨어진다 — 이 부분은 구조에서 나오는 추론).
  - 잡음은 열어 본 어떤 시스템도 시각에 명시적으로 넣지 않았다(Aharonov-Barki의 "냄새 탐침이 자원 위가 아니면 무작위 값"이 유일한 잡음 사례 — 7절). 넣는다면 `근접도 + N(0, σ·d/range)`처럼 거리에 따라 커지는 잡음이 "멀수록 흐릿"에 맞다(설계 제안).
- 격자 패치 관측(eco-evo 15×15, Neural MMO 15×15)은 연속 공간에서는 래스터화 비용이 들고 방향성이 없어 "좁은 시야" 목표와 덜 맞는다. 다만 냄새장(6절)을 격자로 두면 그쪽에서 자연스럽게 쓴다.

### Gaps
- Bibites의 ConcentrationWeight·Angle 계산식(거리·크기 가중, 가림 여부)과 시야 유전자의 에너지 비용 미확인. 검색에 걸린 "가장 중요한 물체의 각도" 설명 글은 Bibites가 아니라 다른 게임(NeuraQuarium) 개발자의 글이었다.
- ALIEN 센서의 탐지 각·거리·밀도 매개변수는 프로그램 내 도움말에만 있어 확인 못 함.
- 시야 형질(FOV·거리·광선 수)에 에너지 비용을 붙여 진화시킨 정량 실험 결과(어느 값으로 수렴했는지)는 찾지 못했다. Polyworld는 압력이 있다는 서술뿐이다.
- 대상 서버에서의 시각 비용, 그리고 뇌(신경망) 추론 비용과의 비율은 재지 않았다.
- 선분 장벽·벽에 대한 광선 교차는 다루지 않았다(원만 측정).

## 6. 다른 감각: 접촉, 냄새(확산 격자), 내부 상태, 소리·신호 자리, 최소 감각 세트

### Takeaway
비교 대상들이 공통으로 주는 최소 세트는 "외부 감각 한 종류(시각 또는 근접 탐침) + 자기 에너지 + 상수(또는 난수)"이고, 그 위에 나이, 자기 속도·마지막 이동 방향, 피해 감지, 시계/진동자, 페로몬이 얹힌다. 냄새는 biosim4처럼 격자 스칼라장의 "주변 값 + 전후 차 + 좌우 차" 세 입력이 가장 싸다.

### Cited Findings
- Polyworld 입력: 시각(색 띠) + 현재 내부 에너지(health-energy) + 난수 값. 출력 7개: eating, mating, fighting, moving, turning, focusing, lighting(직접 읽음) — [Yaeger 1994](http://textfiles.com/computers/CYBERSPACE/yaegerxa.txt)
- Evolvio(포크) 입력 12개: 시각 9(색 3성분 × 3곳) + Size(에너지) + Mem(자기 출력을 되먹임하는 기억 1칸) + Const(직접 읽음, 포크) — [Creature.java](https://github.com/FSMaxB/evolvio/blob/master/src/main/java/io/evolvio/Creature.java)
- Bibites v0.3.0 입력 29개: Constant, EnergyRatio, Maturity, LifeRatio, Speed, IsGrabbingObjects, AttackedDamage, 시각 12개(5절), Tic, Minute, TimeAlive, PheroSense1~3, Phero1~3Angle, InfectionRate(직접 읽음) — [Bibites 세이브 파일 gist](https://gist.github.com/AndisGrossteins/449d1ac2bb7eeae198160a856d199d80)
- biosim4 센서 21개(전부 0.0~1.0): LOC_X, LOC_Y, BOUNDARY_DIST_X, BOUNDARY_DIST, BOUNDARY_DIST_Y, GENETIC_SIM_FWD(앞 개체와의 유전 유사도), LAST_MOVE_DIR_X, LAST_MOVE_DIR_Y, LONGPROBE_POP_FWD, LONGPROBE_BAR_FWD, POPULATION, POPULATION_FWD, POPULATION_LR, OSC1(진동자), AGE, BARRIER_FWD, BARRIER_LR, RANDOM, SIGNAL0, SIGNAL0_FWD, SIGNAL0_LR(직접 읽음) — [sensors-actions.h](https://github.com/davidrmiller/biosim4/blob/main/src/sensors-actions.h)
- biosim4 페로몬: 격자 위에 겹친 한 층, 값은 부호 없는 8비트(0~255). 방출하면 작은 이웃 범위의 값이 올라가고 255에서 막히며, 보충되지 않으면 시간에 따라 줄어든다. 감지 반경 signalSensorRadius=2.0, 방출은 출력이 임계 0.5를 넘을 때 확률적으로 `signals.increment(0, loc)`(직접 읽음) — [biosim4 README](https://github.com/davidrmiller/biosim4/blob/main/README.md), [biosim4.ini](https://github.com/davidrmiller/biosim4/blob/main/biosim4.ini), [executeActions.cpp](https://github.com/davidrmiller/biosim4/blob/main/src/executeActions.cpp)
- JaxLife 통신: 자기 메시지가 가장 가까운 N^view_agents 에이전트에게 전달되어 그들의 other-message를 갱신한다. 자기 속성(에너지, 나이 등)도 관측에 들어간다(직접 읽음) — [JaxLife 논문](https://arxiv.org/html/2409.00853v1)
- eco-evo 격자: 관측 외에 "직전 행동"과 "방금 자원을 먹었는지" 플래그를 합성곱 출력에 이어 붙여 LSTM에 넣는다(고유감각과 섭취 감각에 해당)(직접 읽음, ar5iv 요약 변환본) — [Eco-evolutionary Dynamics](https://ar5iv.labs.arxiv.org/html/2302.09334)
- Aharonov-Barki 외의 접촉형 감각(somatosensor): 탐침 4개가 자기 칸과 앞의 세 칸을 감지(빈칸 / 자원 있음(먹이·독 구분 없음) / 경기장 경계 / 먹이 구역 경계), 다섯 번째 "냄새" 탐침은 바로 밑의 자원이 먹이인지 독인지 구분하지만 자원 위가 아니면 무작위 값을 낸다(직접 읽음) — [Aharonov-Barki, Beker, Ruppin 2001 (preprint PDF)](https://neuro.bstu.by/ai/To-dom/My_research/failed%201%20subitem/For-research/Agents/pcmj12.pdf)
- Framsticks 수용기: S(냄새)는 먹이와 다른 개체를 구분하지 않고 감지하고, T(접촉), G가 있다. 문헌은 냄새와 압력 수용기만으로 세계에서 길을 찾기에 충분하다고 적는다(요약만 봄) — [Framsticks 소개 논문(Emerald)](https://www.emerald.com/k/article/264074/Framsticks)
- Neural MMO 에이전트 속성 관측: health, food, water, lifetime, position, freeze 플래그(요약만 봄) — [Neural MMO 논문](https://arxiv.org/pdf/1903.00784)

### Inferences
- 최소 감각 세트 비교(위 사실의 정리):

  | 시스템 | 외부 감각 | 내부 감각 | 기억·시계 보조 | 신호 |
  |---|---|---|---|---|
  | Polyworld | 1D 색 띠 | 에너지, 난수 | 없음(순환 연결) | 색·발광 |
  | Evolvio(포크) | 색 표본 3곳 | 크기(에너지), 상수 | Mem 1칸 | 몸 색 |
  | Bibites 0.3.0 | 범주별 집계 12 | 에너지비, 성숙도, 수명비, 속도, 붙잡는 중, 받은 피해, 감염 | Tic, Minute, TimeAlive | 페로몬 3종 |
  | biosim4 | 정면 탐침, 밀도, 장벽 | 위치, 나이, 마지막 이동 방향 | 진동자, 난수 | 페로몬 1층 |
  | eco-evo 격자 | 15×15 패치 | 직전 행동, 먹었는지 | LSTM | 없음 |
  | JaxLife | 가까운 N개 + 지형 | 자기 속성 | LSTM | 메시지 |

- 1단계 권장 최소 세트(공통분모): (1) 시각 부채꼴(5절), (2) 에너지 비율 E/E_max, (3) 자기 속도(전진 성분·회전 성분 — Bibites의 Speed, biosim4의 LAST_MOVE_DIR), (4) 접촉/피해 플래그(Bibites AttackedDamage, eco-evo "먹었는지"), (5) 나이 비율, (6) 상수 1, (7) 선택: 진동자 또는 난수(biosim4 OSC1·RANDOM, Polyworld 난수 — 순환 뇌가 없을 때 탐색을 돕는 싼 장치).
- 절대 위치(biosim4 LOC_X/Y)는 주지 않는 편이 목표("탐색·기억이 필요해지게")에 맞다. Aharonov-Barki에서 위치 센서가 있으면 기억 대신 위치 지도로 풀었다(7절).
- 접촉 감각은 충돌 단계의 부산물로 공짜다: 접촉 쌍 목록에서 `np.add.at`으로 "접촉 수", "접촉 방향(자기 기준 각의 cos·sin 합)", "닿은 것이 먹이/개체/벽" 플래그를 만든다.
- 냄새(확산 격자) 구현 스케치 — biosim4 구조를 연속 공간에 옮김: 거친 격자 S[gx, gy](float32)에 먹이·시체·개체가 매 스텝 값을 더하고, `S *= (1 - decay)` 후 3×3 평균(또는 5점 라플라시안 `S += D * lap(S)`, 안정 조건 D ≤ 0.25)으로 퍼뜨린다. 감지는 자기 위치 값 + 앞뒤 차 + 좌우 차 3입력(biosim4의 SIGNAL0, SIGNAL0_FWD, SIGNAL0_LR와 같은 꼴). 격자 연산이라 개체 수와 무관하게 싸다. 확산 식과 안정 조건은 출처를 확인하지 않은 표준 수치해석 지식이다.
- 소리·신호 자리: 지금은 출력 1~3개(방출)와 입력 3개(값·전후·좌우)를 예약만 하고 0으로 두면 된다. Polyworld의 발광은 논문 시점까지 사용 증거가 없었으므로, 신호 채널을 넣는다고 통신이 저절로 나오지는 않는다.

### Gaps
- biosim4 페로몬의 감쇠율·방출량 숫자는 signals.cpp를 열지 않아 미확인.
- Bibites 페로몬 감지식(PheroSense 유전자와 거리의 관계)·시계 입력(Tic, Minute)의 정확한 정의 미확인.
- 확산 격자를 쓰는 ALife 시스템의 실제 계수(확산률·감쇠율) 사례를 원문으로 확인하지 못했다.
- Framsticks 수용기의 수식(거리 감쇠 등)은 공식 문서 주소가 404라 확인 못 함.

## 7. 감각 제한이 진화한 행동을 바꾼다는 증거

### Takeaway
직접 읽은 증거는 세 가지다: (1) 위치 정보를 빼면 진화한 신경망에 단기 기억 기제가 스스로 생긴다(Aharonov-Barki 외 2001), (2) 시야를 좁힐수록 기억 없는 에이전트는 무너지고 단순한 기억만 더해도 크게 회복한다(Forager 2026, 진화가 아니라 강화학습), (3) 포식자의 지각 한계(혼란 효과)만으로 먹이의 무리 짓기가 진화한다(Olson 외 2013). 다만 "시야를 좁히면 통신이 진화한다"는 직접 증거는 이번에 찾지 못했다.

### Cited Findings
- Aharonov-Barki 외: 격자 경기장, 독은 전체에 흩어져 있고 먹이는 남서쪽 "food zone"에만 있다. 과제의 어려움은 "limited sensory information"에서 나온다. 뇌는 완전 순환 신경망(15~50 뉴런), 개체군 100, 한 생애 150스텝, 운동은 전진·좌우 90° 회전·먹기(먹기는 이동 없는 한 스텝을 쓴다)(직접 읽음) — [Aharonov-Barki et al. 2001 preprint](https://neuro.bstu.by/ai/To-dom/My_research/failed%201%20subitem/For-research/Agents/pcmj12.pdf)
- 같은 논문: SP형(접촉 감각 + 위치 센서)과 S형(접촉 감각만)을 비교. 위치 정보가 있으면 command neuron이 환경 지도를 반영하는 위치 의존 세포처럼 동작하고, 위치 정보가 없으면 "spontaneously evolving short-term memory mechanism"이 생겨 그 뉴런의 겉보기 위치 민감성을 만든다. 저자들은 이 기억 기제에 대한 2-매개변수 확률 모형을 제안한다(직접 읽음: 초록·모형 절) — [Aharonov-Barki et al. 2001 preprint](https://neuro.bstu.by/ai/To-dom/My_research/failed%201%20subitem/For-research/Agents/pcmj12.pdf)
- 같은 논문의 냄새 탐침은 자원 위가 아니면 무작위 값을 내므로 "requires sensory integration"(여러 번 읽어 통합해야 함)이라고 적는다. 잡음 섞인 감각이 시간 통합을 요구한 사례다(직접 읽음) — [Aharonov-Barki et al. 2001 preprint](https://neuro.bstu.by/ai/To-dom/My_research/failed%201%20subitem/For-research/Agents/pcmj12.pdf)
- Forager(Tang 외, arXiv 2605.01131v2; 강화학습 시험대): 시야(FOV) 3, 5, 7, 9, 11, 13, 15를 시험. 가장 큰 시야에서는 DQN이 Search Oracle과 비슷했고, 시야가 줄수록 나빠져 FOV 7에서는 "가장 가까운 것 찾기" 기준선과 비슷, FOV 3·5에서는 기준선보다 못하고 "stumbled around the world lacking reward-seeking behavior"였다(생물군 사이 간격 너머를 볼 수 없었음)(직접 읽음, 본문 앞 100,000자 범위) — [Forager](https://arxiv.org/html/2605.01131v2)
- Forager: 지난 보상의 지수 흔적(trace)이라는 단순한 기억만 더해도 "both agents exhibit significant performance improvements". GRU를 붙인 DRQN은 DQN보다 낫지만 단순 흔적 기억보다 못했고, 순환 PPO(RTU-PPO)가 학습 에이전트 중 가장 좋았다(직접 읽음) — [Forager](https://arxiv.org/html/2605.01131v2)
- Olson 외: 포식자의 포획 확률이 시야 안 먹이 수에 반비례(`Pcapture = 1/ANV`)하는 지각 제약만으로 먹이의 무리 짓기가 진화했다(1,200세대, 180회 반복). 저자들은 이 압력이 거꾸로 포식자 시각계의 구조에도 압력을 줄 수 있다고 논한다(정면 지향·고해상도)(직접 읽음) — [Olson et al.](https://arxiv.org/pdf/1209.3330)
- Olson 외: 먹이 망막은 칸마다 이진값만 주지만, 저자들은 이 제약이 "counting mechanism"의 진화를 막지 않는다고 명시한다. 낮은 해상도 감각 위에서 뇌가 통합 기능을 만들 여지를 남긴 설계다(직접 읽음) — [Olson et al.](https://arxiv.org/pdf/1209.3330)
- eco-evo 격자(Hamon 외): 15×15 시야와 LSTM(은닉 4)을 가진 개체에서 두 전략이 공존했다 — 거의 직선으로 다니며 지나는 자원을 먹는 "opportunistic traveler"와 자원 근처에 머물며 탐욕스럽게 먹지 않는 "sustainable" 채집자(시험한 50개체 중 9). 생애 내 적응을 LSTM의 기억 덕으로 보는 것은 저자들의 가설이고 직접 측정은 아니다(직접 읽음, ar5iv 요약 변환본) — [Eco-evolutionary Dynamics](https://ar5iv.labs.arxiv.org/html/2302.09334)
- Polyworld: 감각이 좁은 색 띠뿐인 세계에서 반복해 나온 종은 "직진만 하는 종"(frenetic joggers), "제자리 종"(indolent cannibals), "가장자리 도는 종"(edge runners), "빠르게 도는 종"(dervishes)이었다. 저자는 시각으로 이동을 제어하는 증거는 관찰했다고 적는다(직접 읽음) — [Yaeger 1994](http://textfiles.com/computers/CYBERSPACE/yaegerxa.txt)
- Gallagher(GECCO 1999)는 시각·이동·단기 기억을 통합하는 연속시간 순환망을 진화시켰고 단기 기억이 필요한 과제를 포함한다(요약만 봄) — [GECCO 1999 AA-005](https://gpbib.pmacs.upenn.edu/gecco1999/AA-005.pdf)

### Inferences
- 설계 함의 1: "감각을 좁히면 기억이 생긴다"는 조건부다. Aharonov-Barki에서 기억이 나온 것은 (a) 감각이 국소적이고, (b) 세계에 기억할 가치가 있는 구조(먹이 구역)가 있고, (c) 뇌가 순환 연결을 가졌기 때문이다. Polyworld에서 좁은 시각이 주로 낳은 것은 기억이 아니라 "감각이 필요 없는 반사 전략"(직진, 가장자리 돌기, 제자리)이었다. 즉 먹이가 균일하게 흩어져 있으면 좁은 시야는 무작위 보행으로 풀린다. 먹이를 패치·구역·계절로 뭉쳐 "기억할 것"을 만들어야 한다.
- 설계 함의 2: 뇌에 최소한의 기억 경로가 있어야 한다. Forager에서는 지수 흔적 같은 아주 단순한 기억이 GRU보다 효과적이었다. 진화 설정에서도 Evolvio의 "Mem 1칸 되먹임"이나 감각의 지수 이동평균 입력처럼 싼 기억을 기본으로 주는 편이 순환망이 저절로 기억을 발명하기를 기다리는 것보다 빠를 가능성이 있다(강화학습 결과를 진화로 옮긴 추정).
- 설계 함의 3: 시야가 "과제의 공간 규모보다 좁아야" 효과가 난다. Forager에서 FOV 3·5가 무너진 것은 생물군 사이 간격을 볼 수 없었기 때문이다. 시야 거리 < 먹이 패치 간 거리가 되도록 세계를 잡으면 탐색이 필요해지고, 시야 거리 ≥ 패치 간 거리면 반사 행동으로 충분하다.
- 설계 함의 4: 잡음은 시간 통합(=기억의 가장 약한 형태)을 요구한다(Aharonov-Barki의 냄새 탐침). 시각을 이진·저해상도로만 하지 말고 약한 잡음을 섞으면 "여러 번 보고 판단"이 유리해진다.
- 설계 함의 5: 다른 개체의 지각 한계가 사회 행동을 만든다(Olson). 포식을 넣을 때 포식자의 시야·목표 고정 한계를 물리로 넣으면(여럿이 보이면 명중률 하락) 무리 짓기 압력이 생긴다. 다만 이것은 손으로 넣은 규칙이고, 1단계에서 창발로 기대할 것은 아니다.
- 통신: Polyworld는 발광 채널을 주었지만 사용 증거를 찾지 못했다고 적었다. 좁은 시야가 통신을 낳으려면 "한 개체가 아는 것을 다른 개체가 알면 이득"인 상황(친족 근처에 머무름, 먹이 패치가 여럿이 먹을 만큼 큼)이 따로 필요해 보인다(추론; 직접 증거 없음).

### Gaps
- 시야 범위·시야각을 체계적으로 바꿔 가며 "진화한" 개체의 기억 사용량(순환 연결 사용, 은닉 상태 정보량)을 잰 ALife 실험은 찾지 못했다. Forager는 강화학습이고 Aharonov-Barki는 위치 센서 유무 두 조건뿐이다.
- 감각 제한 → 통신 진화의 직접 증거 미발견.
- Aharonov-Barki에서 진화 개체가 "최선의 기억 없는 설계 알고리즘"보다 얼마나 나았는지의 수치는 그림에만 있어 읽지 못했다.
- Forager 본문의 마지막 약 7,000자(부록 일부)는 읽지 않았다. 학회는 미확인(arXiv 판).
- Beer(1996) 원 논문, Gallagher(1999) 원문은 열지 않았다.
- Polyworld 후속 연구(복잡도 증가 측정 등)에서 시각 뉴런 수가 실제로 어떻게 진화했는지는 조사하지 않았다.
