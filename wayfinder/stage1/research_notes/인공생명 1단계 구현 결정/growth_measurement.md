# 성장 측정: 열린·비에피소드 인공생명 세계가 "변하기만 하는지 / 도는지 / 떠내려가는지 / 자라는지" 가르는 법

조사일 2026-11-04. 표기: **[1차]** = 원문(PDF/HTML)을 직접 열어 읽음, **[2차]** = 검색 요약·다른 논문의 인용으로만 확인(원문 못 엶).
원문을 못 연 것: Cliff & Miller 1995, Rosin & Belew 1997, Channon 2001/2003/2006(Keele 저장소 403, workshops.alife.org 인증서 오류), Cartlidge & Bullock 2004(403), Lenski 1991, Wiser·Ribeck·Lenski 2013, Taylor et al. 2016, Banzhaf et al. 2016, Adams et al. 2017, AlphaStar Nature 논문, Dolson et al. 2020, Stout & Spector 2005. 이들은 아래에서 [2차]로 표시했다.

---

## 1. 과거 대 현재 프로토콜 (CIAO, Master Tournament, Hall of Fame, Elo/TrueSkill) — 정확히 어떻게 돌리고, 순환과 진짜 진보를 어떻게 가르나

### Takeaway
"지금 개체가 옛 개체를 이긴다"는 **historical progress**만 증명하고 **global progress**(처음 보는 상대에게도 강해짐)는 증명하지 못한다 — Miconi는 이것을 "training set을 test set으로 쓰는 오류"라고 부른다. 순환 여부는 (a) 세대×세대 행렬에서 "시차가 클수록 더 크게 이기는가", (b) 평가 행렬의 transitive 성분과 cyclic 성분의 비로 가른다.

### Cited Findings
- **세 가지 진보의 정의.** local progress = 현재 상대에 대한 우위, historical progress = 과거 상대에 대한 우위, global progress = 새로운(본 적 없는) 상대를 포함한 전체 상대 공간에 대한 우위. "prolonged progress against competitors does not necessarily imply global progress" — [1차] [Simione & Nolfi 2019, arXiv:1909.08303](https://arxiv.org/pdf/1909.08303); 원 정의는 [1차] [Miconi 2008 박사논문 3장](https://etheses.bham.ac.uk/id/eprint/148/1/Miconi08PhD.pdf)
- **CIAO (Current Individual vs Ancestral Opponents).** 각 세대의 최고 개체를 "경쟁 집단의 모든 과거 세대 최고 개체"와 사후 평가한 자료. 과거 상대만 쓰므로 historical progress만 잰다 — [1차] [Nolfi & Floreano 1998](https://infoscience.epfl.ch/record/63899) (Cliff & Miller의 방법을 이렇게 요약), [1차] [Simione & Nolfi 2019](https://arxiv.org/pdf/1909.08303). Cliff & Miller 1995 원문은 [2차].
- **Master Tournament.** "testing the performance of the best individual of each generation against each best competitor of all generations". Nolfi & Floreano의 설정: 두 집단 각 100개체, 100세대 공진화, 각 세대 최고 개체를 상대편 100세대의 최고 개체와 **한 번씩** 대결(점수 0~100), **10회 반복 실험**의 평균과 최고 실험을 따로 그림 — [1차] [Nolfi & Floreano 1998](https://infoscience.epfl.ch/record/63899)
- **이상적 무기경쟁 그림 대 실제.** 이상형: 세대×세대 정사각형에서 "포식자가 더 최근 세대인 삼각형"은 전부 검정(포식자 승), 반대 삼각형은 전부 흰색. 실제 표준 공진화: "The distribution of black and white spots does not differ significantly in the two sub-parts of the square", 그리고 "individuals of later generations do not necessarily score well against competitors of much earlier generations". 10회 평균에서는 "performance does not increase at all throughout generations" — [1차] [Nolfi & Floreano 1998](https://infoscience.epfl.ch/record/63899)
- **순환의 구조.** 전략 A(집단1)→B를 이김, C(집단2)가 A를 이김, D가 C를 이김, B가 D를 이김 → A·B로 되돌아가 무한 반복. 공진화 실험에서 "The most common outcome is a limit cycle dynamic in which qualitatively similar strategies are abandoned and rediscovered over and over again" — [1차] [Simione & Nolfi 2019](https://arxiv.org/pdf/1909.08303)
- **Hall of Fame (Rosin & Belew 1997).** "The best individual from every generation, is retained for future testing." Nolfi & Floreano의 구현: 각 개체를 **이전 모든 세대에서 무작위로 뽑은 10 상대**와 평가(표준 조건은 직전 10세대의 최고 개체 10). 결과: Master Tournament 점수는 꾸준히 오름. 그러나 표준 공진화 최고 10개체 대 HoF 최고 10개체 교차 대결에서는, 감각이 풍부한 설정에서 "individuals obtained with standard co-evolution tend to outperform individuals obtained with 'Hall of Fame' co-evolution"; 단순 설정에서도 차이가 "not as great as one could expect from the trends of the Master Tournaments" — [1차] [Nolfi & Floreano 1998](https://infoscience.epfl.ch/record/63899) (Rosin & Belew 원문은 이 논문 속 인용으로만 확인 [2차])
- **HoF의 한계 요약.** archive 기법은 "guarantees historical progress but not global progress"; archive 상대에 과적합하고 다른 상대에 일반화가 약함 — [1차] [Simione & Nolfi 2019](https://arxiv.org/pdf/1909.08303)
- **Master Tournament도 독립 검증이 아니다.** 미래 세대 상대는 과거 상대의 후손이므로 "for each species, half the test set is composed of the training set, and the rest is derived from it". 단일 실행 자료만 쓰는 방법(Master Tournament, CIAO, Dominance Tournament)은 "essentially measure historical progress" — [1차] [Miconi 2008, 6장](https://etheses.bham.ac.uk/id/eprint/148/1/Miconi08PhD.pdf); 같은 주장 초록 [Miconi 2009 EuroGP](https://gpbib.cs.ucl.ac.uk/gp-html/Miconi_2009_eurogp.html)
- **Coarse-grained Master Tournament(비용 절감).** N세대 전부가 아니라 등간격으로 k개 챔피언만 뽑아 k×k 행렬. 예: 500세대를 10세대마다 뽑아 50×50. "by sampling 10% of the generations, computing costs are divided by 100". 잃는 것은 단기·소규모 정보이고 "과거보다 나은가"라는 큰 흐름은 보존 — [1차] [Miconi 2008, 6.4](https://etheses.bham.ac.uk/id/eprint/148/1/Miconi08PhD.pdf)
- **장기 실험의 실제 수치(Simione & Nolfi).** 150,000세대, **10,000세대마다** 양 집단을 뽑아 과거·미래 세대 상대와 사후 평가, **5회 반복**, 비교는 **one-tailed t-test (N=5)**. 진보의 판정 기준: historical = "현재 상대보다 옛 상대에게 유의하게 더 잘함"(예: 150,000세대 포식자가 동시대 상대에 약 0.25, 10,000세대 상대에 약 0.6), global = "마지막 세대 상대에 대해 이전 세대보다 유의하게 더 잘함". Vanilla(archive·필터 없음) 조건은 "did not produce neither historical neither general progress" — [1차] [Simione & Nolfi 2019](https://arxiv.org/pdf/1909.08303)
- **Red Queen 때문에 동시대 적합도 곡선은 쓸모없다.** 적합도가 안 변하는 구간은 "stagnation"일 수도, 양쪽이 같이 좋아지는 "progress"일 수도 있고, 한쪽 적합도 상승은 그쪽의 개선일 수도 상대의 퇴보일 수도 있다 — [1차] [Simione & Nolfi 2019](https://arxiv.org/pdf/1909.08303)
- **CIAO 그림의 모호성.** 순환적이거나 무작위인 공진화도 진보와 비슷해 보이는 "tartan" 무늬를 만들 수 있어, 그림만으로 판단하지 말고 문제별 분석을 같이 하라고 권고 — [2차] [Cartlidge & Bullock 2004 (검색 요약으로만 확인)](https://eprints.soton.ac.uk/261441/2/ciao.pdf)
- **Elo/TrueSkill의 한계(자기대국 RL).** "Tracking reward is an insufficient evaluation metric in multi-agent settings"; 보상은 "ambiguous in indicating whether agents are improving evenly or have stagnated". Elo·TrueSkill은 이전 정책 판본 대비 향상을 더 믿을 만하게 재지만 "these metrics still do not give insight into whether improved performance stems from new adaptations" — [1차] [Baker et al. 2019, arXiv:1909.07528 (ar5iv HTML)](https://ar5iv.labs.arxiv.org/html/1909.07528)
- **Transitive / cyclic 분해(Balduzzi et al. 2019).** 대칭 영합 게임을 반대칭 함수 φ(v,w) = −φ(w,v)로 쓰면, 모든 게임 = transitive game ⊕ cyclic game(직교 분해). transitive: φ(v,w) = f(v) − f(w) (등급 f의 차이). cyclic: 모든 v에 대해 ∫_W φ(v,w) dw = 0. Elo 모형은 P(v≻w) = σ(f(v) − f(w)), σ(x) = 1/(1+e^(−αx)) 꼴이라 등급 차로 쓸 수 있는 성분만 표현한다. cyclic 게임에서는 "the performance of individual agents is meaningless" — [1차] [Balduzzi et al. 2019, arXiv:1901.08106](https://arxiv.org/abs/1901.08106)
- **평가 행렬과 순환 지표.** 개체 n개의 평가 행렬 A_P = [φ(w_i, w_j)] (n×n 반대칭). 단일 순환(n개체)의 rank는 n이 짝수면 n−2, 홀수면 n−1. Schur 분해가 "the analog of PCA for antisymmetric matrices". 지배당하는 것처럼 보이는 개체가 부분적으로만 관측된 순환의 일부일 수 있다는 경고 — [1차] [Balduzzi et al. 2019](https://arxiv.org/abs/1901.08106)
- **집단 단위 비교.** relative population performance v(P,Q) = pᵀ A_{P,Q} q, (p,q)는 A_{P,Q}의 Nash 균형; effective diversity d(P) = pᵀ ⌊A_P⌋₊ p — [1차] [Balduzzi et al. 2019](https://arxiv.org/abs/1901.08106)
- **AlphaStar league.** main agents / main exploiters / league exploiters 세 역할, prioritized fictitious self-play(상대 승률로 표집 가중)로 "가위바위보식 순환"을 피하려 함 — [2차] [DeepMind 블로그·2차 해설 (Nature 논문의 Elo/payoff 평가 절차는 확인 못 함)](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/)

### Inferences
- 사용자 안("100세대 전 개체와 지금 개체를 같이 풀어 지금이 이기면 성장")은 CIAO의 **한 칸**(시차 100 하나)만 보는 것이다. 한 칸으로는 순환과 진보를 못 가른다: 순환 주기가 100세대와 어긋나면 "이김"이, 맞으면 "비김"이 나온다. 최소한 **여러 시차**(예: 10, 30, 100, 300세대 전)를 함께 봐야 하고, 진보라면 **시차가 클수록 격차가 커지는 단조 관계**가 나와야 한다(Simione & Nolfi의 historical progress 판정과 같은 논리).
- 순환 판정의 구현 가능한 수치화(Balduzzi 분해의 이산판): 체크포인트 k개의 대결 결과로 반대칭 행렬 A를 만들고, 등급 f_i = (1/k)·Σ_j A_ij (행 평균)을 구한다. transitive 성분 T_ij = f_i − f_j, cyclic 잔차 C = A − T. **transitive 비율 = ‖T‖²_F / ‖A‖²_F**. 1에 가까우면 "등급 하나로 설명되는 진보", 낮으면 가위바위보 구조. 그리고 f가 체크포인트 시간에 대해 증가하는지(Spearman 상관)를 본다. numpy 몇 줄이고 추가 시뮬레이션이 필요 없다.
- 비용: k개 체크포인트 전수 대결은 k(k−1)/2 쌍. Miconi의 coarse-graining을 따라 k를 작게(예: 8~12) 유지하고, 새 체크포인트가 생길 때 그 행만 추가로 채우면 된다.
- Elo/TrueSkill은 1단계에 과하다. 대결 수가 적고, 순환 성분이 있으면 등급이 무의미해지므로, 위의 행렬+transitive 비율이 더 정직하다.

### Gaps
- Cliff & Miller 1995의 CIAO 원 절차(대결 횟수, 음영 계산법)는 원문을 못 열어 Nolfi & Floreano·Simione & Nolfi의 요약에 의존했다.
- Rosin & Belew 1997의 HoF 표집 크기·"shared sampling"/"competitive fitness sharing" 세부는 확인 못 했다.
- AlphaStar의 체크포인트 간 Elo·payoff matrix 평가 절차(대국 수 등)는 1차 자료로 확인 못 했다.
- 순환과 진보를 가르는 **정량적 문턱값**(예: transitive 비율 몇 % 이상)은 어떤 문헌에서도 찾지 못했다 — 문턱은 중립 대조(3번)에서 얻은 잡음 바닥으로 직접 정해야 한다.

---

## 2. 명시적 대결이 없는 세계로 옮기기 — "이김 = 자손을 남김"일 때의 경쟁 분석과 공식

### Takeaway
Lenski식 경쟁 분석이 그대로 옮겨진다: 두 유형을 같은 수로 새 표준 세계에 섞어 넣고, 각자의 실현 Malthusian parameter m = ln(N_f/N_i)/T를 구해 **비(W)** 또는 **차(r, selection rate)**를 본다. 개체 수가 줄 수 있는 인공생명 세계에서는 W가 깨지므로 **r을 써야 한다**. 대결 없이 계보 기록만으로 선택이 작동하는지 보는 통계(Fitness Transmission)도 있다.

### Cited Findings
- **Relative fitness 공식.** w = ln(A_f/A_i) / ln(B_f/B_i), A·B는 두 경쟁자의 개체 수, i·f는 시작·끝. "실현된 성장률의 비"이며 배가 횟수의 비와 같다 — [1차] [Wiser & Lenski 2015, PLoS ONE](https://pmc.ncbi.nlm.nih.gov/articles/PMC4427439)
- **프로토콜(Barrick Lab, Lenski 실험실 절차를 옮긴 것).** (1) 되살리기 → (2) **preconditioning**: 두 균주를 각각 시험 환경에서 24시간 길러 생리 상태를 맞춤 → (3) 두 균주를 같은 희석(각 200배)으로 섞어 시작, 즉시 초기 수를 셈 → (4) 정확히 24시간 뒤 최종 수를 셈. M_A = N_A(f)/N_A(i), M_B = N_B(f)/N_B(i), **W = log(M_A)/log(M_B)** = "the ratio of their Malthusian parameters". 더 정밀하게 하려면 여러 날 이어서 하되 "그 사이에 진화가 일어날 수 있음"을 경고 — [1차] [Barrick Lab: Co-culture Competition Assays](https://barricklab.org/twiki/bin/view/Lab/ProceduresLongTermCompetitions)
- **W가 깨지는 경우.** (1) 한쪽 또는 양쪽 개체 수가 경쟁 중에 **감소**하면 W가 음수가 될 수 있음, (2) 두 유형의 적합도 차가 매우 클 때. 이때는 selection rate r을 쓰라고 권고 — [1차] [Barrick Lab](https://barricklab.org/twiki/bin/view/Lab/ProceduresLongTermCompetitions)
- **Selection rate.** m_A = ln[A(1)/A(0)]/day (실현 Malthusian parameter), m_B도 같게 정의, **r = m_A − m_B** (직접 경쟁 중 두 Malthusian parameter의 차). W는 차가 아니라 비 — [2차] [Lenski, "Comment on the use of selection rate versus relative fitness" (검색 발췌로 확인, 현재 주소는 다른 곳으로 넘어감)](https://lenski.mmg.msu.edu/ecoli/srvsrf.html)
- **표지(marker) 중립성 대조.** 표지만 다른 같은 유형끼리 경쟁시켜 표지가 중립임을 확인(예: Ara+ 복귀 돌연변이가 중립임을 다일 경쟁으로 보임) — [1차] [Barrick Lab](https://barricklab.org/twiki/bin/view/Lab/ProceduresLongTermCompetitions)
- **적합도 격차가 커지면 측정 오차가 커진다.** "As the relative fitness of the two competitors diverges, the measurement error increases" — 덜 적합한 쪽이 너무 적어져 세기 어려워서. 대안: 시작 비율을 1:4로 바꾸기(ASR), 중간 적합도의 공통 경쟁자 쓰기(DCC: 0~50,000세대 기대값의 기하평균 근처인 5,000세대 클론). 적합도 1~1.8 범위에서는 세 방법의 평균 궤적이 사실상 같았고 정밀도 차이도 유의하지 않아(Traditional 대 DCC p = 0.8802) 전통 방법을 권고, 1.8을 크게 넘으면 ASR·DCC가 나을 수 있음 — [1차] [Wiser & Lenski 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4427439)
- **Avida에서의 대응물.** 표지를 단 유형과 안 단 유형을 한 집단에서 경쟁시키며 세대마다 각 유형의 평균·최대 적합도와 **상대 빈도**를 기록 — [2차] [Wilke, arXiv:physics/0302046 (검색 요약)](https://arxiv.org/pdf/physics/0302046). Avida는 집단을 저장·복원할 수 있어 "direct comparisons of evolved and ancestral genotypes"가 가능 — [2차] [Ostrowski, Ofria & Lenski (검색 요약)](https://core.ac.uk/download/194690722.pdf)
- **적합도 함수 없는 세계의 "챔피언".** 열린 진화에서는 "maximum number of children is the closest approximation to the notion of 'champion'" — [1차] [Miconi 2008, 10장 각주](https://etheses.bham.ac.uk/id/eprint/148/1/Miconi08PhD.pdf)
- **Fitness Transmission(계보만으로 선택 검출).** 정의: "the statistical correlation between the fitness of children and parents". 적합도 = **손주 수(NOGC)** — 자식 수만 쓰면 "자식은 많지만 그 자식이 못 사는" 형질을 걸러내지 못함. 계산법: (1) 계보 기록을 시간 구간으로 나눔, (2) 구간 안에 태어난 모든 자식 C_i와 그 부모 P_i의 쌍을 모음, (3) 각자의 손주 수 N(P_i), N(C_i)(이 구간 이후 출생만), (4) 의존성 제거: N(P_i)에서 C_i의 자식 수를 뺌 → N′(P_i), (5) 정규화: N′(P_i)를 다른 모든 부모의 손주 합으로 나눔(C_i와 그 후손 제외) → N″(P_i), (6) Pearson 상관 Corr(N″(P_i), N(C_i))이 그 구간의 값. 값 자체는 낮으니 "the enduring presence of this correlation over time"이 지표 — [1차] [Miconi 2008, 9장](https://etheses.bham.ac.uk/id/eprint/148/1/Miconi08PhD.pdf)
- **Evosphere 적용.** 구간 = 20,000 timestep(생물 최대 수명의 2배), 5구간 이동평균. 초기에 FT가 거의 0인 단계가 있다가, 지속적 진동 운동(이동)이 나타나는 순간 FT가 급등하고 kill 수 급증과 시점이 일치 — [1차] [Miconi 2008, 10.4.2](https://etheses.bham.ac.uk/id/eprint/148/1/Miconi08PhD.pdf)
- **FT의 검증과 한계.** 선택 있는 GA 50회 중 46회에서 장기 평균 FT > 0 (이항 검정 p ≈ 10⁻¹⁰), 선택 유·무 비교 two-tailed Mann-Whitney p < 2×10⁻⁶. 한계: 절대적 안정화 선택(최적에서 벗어나면 번식 0)은 무작위 선택과 구별 불가. FT는 성능·새로움의 척도가 아니라 "선택이 작동 중인가"만 답한다 — [1차] [Miconi 2008, 9.5–9.7](https://etheses.bham.ac.uk/id/eprint/148/1/Miconi08PhD.pdf)

### Inferences
- **1단계용 경쟁 분석(arena assay) 설계안** (Lenski 절차의 직역):
  1. 체크포인트: 본 실행에서 일정 간격으로 살아 있는 개체의 유전체를 무작위 표본(또는 전부) 얼려 저장.
  2. 새 표준 세계(본 실행과 같은 물리·먹이 규칙, 고정된 seed 목록)에 조상 표본 A와 현재 표본 C를 **같은 수**로, 같은 초기 에너지·나이·무작위 위치로 넣는다(preconditioning의 대응: 초기 상태를 같게).
  3. 유형 표지(상속되는 정수 tag)를 달고, **돌연변이는 끈다**(경쟁 중 진화 방지 — Barrick 경고의 대응). T는 몇 세대(수명의 수 배)로 짧게.
  4. 지표: m_X = ln(N_X(T)/N_X(0))/T, **r = m_C − m_A**. 시작 수가 같으면 r = ln(N_C(T)/N_A(T))/T로 단순해진다. 한쪽이 0이 되면 +1 보정(또는 "소멸"로 따로 기록). W(비)는 개체 수가 줄 때 음수가 되므로 쓰지 않는다.
  5. 보조 지표: 유형별 총 에너지 섭취량, 평균 수명, T 시점 계통 점유율 C/(A+C).
- **중립 표지 대조가 곧 잡음 바닥이다.** A 대 A(같은 표본을 두 tag로 나눔)를 같은 횟수 돌려 r의 분포를 얻고, C 대 A의 r이 그 분포 밖에 있을 때만 "차이 있음"으로 본다.
- Wiser & Lenski의 교훈을 옮기면, 현재 개체가 너무 강해져 조상이 몇 스텝 만에 사라지면 측정이 거칠어진다. 그때는 (a) 시작 비율을 조상 4 : 현재 1로, 또는 (b) 중간 세대 체크포인트를 "공통 경쟁자"로 쓴다.
- 경쟁 분석은 "상대적 우위"만 준다. **단독 분석**(각 체크포인트 표본만 넣은 세계에서 에너지 섭취 효율·수명·존속 시간을 무작위 뇌 기준선과 비교)을 같이 해야 "상대를 잘 괴롭혀서 이김"과 "세계를 더 잘 삶"을 가를 수 있다(Simione & Nolfi의 Red Queen 지적에 대응).
- Fitness Transmission은 추가 시뮬레이션 없이 본 실행의 출생 기록(자식 id, 부모 id, 시각)만으로 계산되므로 **상시 지표**로 적합하다. 구간 길이는 Miconi를 따라 최대 수명의 2배.

### Gaps
- Lenski, Rose, Simpson & Tadler 1991의 원문 표기와 반복 수는 직접 확인 못 했다(공식은 Wiser & Lenski 2015와 Barrick Lab 절차로 확인).
- Avida 문헌에서 "조상 대 진화형 직접 경쟁"의 표준 반복 수·기간·계산식을 명시한 1차 자료는 찾지 못했다.
- 공간이 있고 개체 수가 수백 수준인 인공생명 세계에서 Lenski식 분석을 그대로 쓴 선례(반복 수·T 선택 근거 포함)는 찾지 못했다. Miconi의 Evosphere도 과거 대 현재 경쟁 분석은 하지 않고 FT만 썼다.
- 유성생식(두 부모)일 때 FT의 "부모"를 어떻게 잡는지(양친 평균 등)는 발췌한 부분에서 확인하지 못했다.

---

## 3. 중립/대조("shadow") 실행 — 정확히 어떻게 만들고 무엇을 비교하나

### Takeaway
대조는 두 계열이 있다. (1) **Bedau의 neutral shadow**: 세계 없이 "출생·사망 사건의 시각과 횟수, 돌연변이율"만 본 실행에서 베끼고, 누가 낳고 누가 죽는지만 균등 무작위로 바꾼 장부상의 집단 — 싸다. (2) **Yaeger의 lockstep(passive) 실행**: 같은 초기 조건의 Polyworld 전체를 돌리되 자연 출생·사망을 막고 본 실행의 출생·사망 기록을 무작위 개체에 강제 — 본 실행만큼 비싸지만 뇌·행동 지표를 직접 비교할 수 있다.

### Cited Findings
- **Evolutionary activity 통계의 정의(Bedau, Snyder & Packard 1998).**
  - activity increment: Δ_i(t) = 1 (component i가 t에 존재) / 0 (아니면); activity counter a_i(t) = Σ_{k≤t} Δ_i(k) — 사실상 component의 나이.
  - diversity: D(t) = #{i : a_i(t) > 0}
  - component activity distribution: C(t,a) = Σ_i δ(a − a_i(t))
  - total cumulative activity: A_cum(t) = Σ_i a_i(t); mean cumulative activity: Ā_cum(t) = A_cum(t)/D(t)
  - new activity: A_new(t) = (1/D(t)) · Σ_{i: a0 ≤ a_i(t) ≤ a1} a_i(t)
  - component는 allele부터 분류군까지 무엇이든 가능하며 이 논문은 genotype 전체와 화석 과(family)를 썼다. counter는 상속되어 계통 전체의 누적값을 유지 — [1차] [Bedau, Snyder & Packard 1998](https://people.reed.edu/~mab/papers/alife6.pdf)
- **Neutral shadow의 구성(Echo).** shadow "creature"는 공간 위치도 자원 섭취도 상호작용도 없고 genotype과 출생·번식·사망 시각만 가진다. "The timing and number of birth and death events in the neutral shadow are directly copied from those in the normal Echo run, as is the neutral shadow's mutation rate." 본 실행에서 출생이 일어나면 shadow에서 **균등 확률로 뽑은 부모**가 번식(돌연변이가 없으면 부모 genotype 상속, 있으면 새 고유 genotype). 본 실행에서 사망이 일어나면 shadow에서 **균등 확률로 뽑은 개체**가 죽음. 본 실행이 shadow에 영향을 주지만 반대는 없음 — [1차] [Bedau, Snyder & Packard 1998](https://people.reed.edu/~mab/papers/alife6.pdf)
- **a0, a1 정하는 법.** 본 실행과 shadow의 activity 분포를 시간 방향으로 합쳐(collapse) 각자의 총 count로 나눈 뒤, 두 분포가 **교차하는 가장 낮은 activity 값 a′**를 찾는다(그 값에서 count가 어느 쪽에서 나왔을 확률이 같음). a0, a1 = a′ ± 0.05 × (a_max − a′). 예: a′ = 2.1×10⁵ → a0 = 1.7×10⁵, a1 = 2.5×10⁵ — [1차] [Bedau, Snyder & Packard 1998](https://people.reed.edu/~mab/papers/alife6.pdf)
- **정규화와 판정.** "we normalize Echo's activity data by subtracting the neutral shadow's new or cumulative activity from that of Echo. If the result is negligible or negative, then Echo's normalized activity is nil." 등급: Class 1 = D 유계, A_new 0, Ā_cum 0(적응 활동 없음); Class 2 = D 유계, A_new 양, Ā_cum 유계; Class 3 = D 무계, A_new 양, Ā_cum 유계. 조작적 정의: f가 unbounded ⇔ lim_{t→∞} sup(f(t))/t > 0; f가 positive ⇔ lim_{t→∞} (∫₀ᵗ f)/t > 0 — [1차] [Bedau, Snyder & Packard 1998](https://people.reed.edu/~mab/papers/alife6.pdf)
- **Echo의 결과와 "돌연변이율 함정".** Echo는 class 1 또는 2, 화석 기록만 class 3. 돌연변이율이 매우 높으면(μ≈1) "virtually every genotype has only one instance"가 되어 본 실행과 shadow가 사실상 같아지고 A_new ≈ 0 → class 1. μ≈0도 class 1. 상호작용 확률·대사세가 매우 낮아도 돌연변이율과 무관하게 class 1. 적응 활동이 유의하게 나온 영역은 0.5 ≤ ι ≤ 1.0, 0.15 ≤ τ ≤ 0.4. shadow는 선택이 없어 genotype 수가 본 실행보다 흔히 더 많다(diversity가 shadow에서 더 높음) — [1차] [Bedau, Snyder & Packard 1998](https://people.reed.edu/~mab/papers/alife6.pdf)
- **저자들이 인정한 약점.** 충분히 오래 관찰해야 하고, component와 activity 정의를 먼저 정해야 해서 진짜 새로운 종류의 component 출현은 직접 못 잡는다("emergence" 문제) — [1차] [Bedau, Snyder & Packard 1998](https://people.reed.edu/~mab/papers/alife6.pdf)
- **Channon의 개선(component-normalised activity).** shadow를 **snapshot마다 본 실행과 똑같게 되돌리고**, snapshot 사이의 변화만 본 실행과 비교해 component 수준에서 비적응 활동을 빼낸다. 이 수정된 검정에서 Geb는 normalised median/total/mean activity가 unbounded, normalised new activity가 양 → 무한 진화 동역학으로 분류 — [2차] [Channon 2003 (Keele 저장소 초록·검색 요약, 본문 못 엶)](https://keele-repository.worktribe.com/output/403342/improving-and-still-passing-the-alife-test-component-normalised-activity-statistics-classify-evolution-in-geb-as-unbounded); [2차] [Channon 2006 Springer 초록](https://link.springer.com/article/10.1007/s10710-006-9009-3)
- **Channon 절차의 최근 적용례(공식 포함).** de Pinho & Sinapayen 2026: shadow는 본 모형과 병렬로 돌며 "each birth in the real model causes a birth in the shadow model", 사망도 같음, 대상 component는 균등 무작위. **1000 timestep마다** shadow의 집단과 activity 이력을 본 모형과 같게 재설정. Δ_i^N(t) = Δ_i^R(t) − Δ_i^S(t); a_i^N(t) = Σ_τ Δ_i^N(τ) (본 모형에 존재할 때). 통계: total normalised cumulative activity(합), median normalised cumulative activity(중앙값), new activity. 판정: total·median이 unbounded이고 new activity가 지속적으로 양이면 통과. 20회 실행 × 200만 timestep. 결과: 정규화 전 total cumulative activity는 40% 실행에서 unbounded였지만 **정규화하면 20회 모두 bounded(오히려 감소), new activity는 0** → "열려 있지 않다". 한계로 "곡선을 눈으로 분류함", "component를 gene(속성 값)으로 고정함"을 듦 — [1차] [de Pinho & Sinapayen 2026, arXiv:2603.01701](https://arxiv.org/abs/2603.01701)
- **Polyworld의 passive(lockstep) 실행(Yaeger, Griffith & Sporns 2008).** 같은 초기 조건에서 시작해 driven 실행의 출생·사망 기록을 재생. 자연 출생·사망은 없음. 기록된 출생마다 "the birth is produced by two agents chosen at random from the population", 기록된 사망마다 "a random agent is killed and removed from the population". 따라서 "population statistics are forced to be identical between the paired natural-selection and lockstep runs". 교차·돌연변이 횟수와 개체의 생애 경험·시각 입력 통계는 통계적으로만 비슷(동일하지 않음) — 돌연변이율 자체가 유전자에 들어 있어서 — [1차] [Yaeger, Griffith & Sporns 2008, arXiv:1112.4906 (ar5iv HTML)](https://ar5iv.labs.arxiv.org/html/1112.4906)
- **무엇을 비교했나.** 개체 일생의 뇌 활동에서 계산한 neural complexity(Tononi 근사: C(X) = H(X) − Σ_i H(x_i | X − x_i), 입력 뉴런 제외), 1,000 timestep 구간별 집단 평균. **10쌍의 seed(20 실행)**, 30,000 timestep(약 400세대로 추정, 범위 100~1,200). 각 시점에서 **paired Student's t-test**, 자유도 9, 임계값 1.833(p < 0.05) — [1차] [Yaeger et al. 2008](https://ar5iv.labs.arxiv.org/html/1112.4906)
- **결과.** 처음 약 4,000 step: driven이 유의하게 더 빨리 증가. 이후 driven은 정체하고 passive가 t≈7,000에 추월, **t≈8,000부터 끝(30,000)까지 passive가 유의하게 더 높음**(평균 driven 0.275 대 passive 0.33). driven 10회 중 7회는 정체 유지, 3회는 t=20,000 이후 2차 상승으로 passive 수준 도달. passive가 정체하는 이유는 유전체 bit가 50/50에 가까워져 분산이 최대가 되기 때문. 결론: 단일 종·생태 지위 규모에서 진화는 항상 driven이지만 방향이 늘 "증가"는 아니다 — [1차] [Yaeger et al. 2008](https://ar5iv.labs.arxiv.org/html/1112.4906)
- **shadow로도 못 거르는 가짜 양성(Miconi의 비판).** Bedau–Packard 통계는 "persistence의 초과 분산"을 재므로, **상속되지 않는** 운(무작위로 뽑힌 개체가 오래 살고 많이 낳음, 또는 출생 때 적합도를 유전체와 무관하게 무작위 부여)만 있어도 shadow 정규화 후 양의 신호가 나온다 — [1차] [Miconi 2008, 9.2](https://etheses.bham.ac.uk/id/eprint/148/1/Miconi08PhD.pdf)
- **검증 시도.** Stout & Spector 2005는 이 통계를 깨 보려는 실험을 했고 "has so far proved robust to our attempts to break it"라고 보고 — [2차] [Stout & Spector 2005 (초록 수준)](https://researchr.org/publication/StoutS05). Miconi의 반례와 결론이 엇갈린다.
- **MODES의 평가.** shadow run은 "challenging to implement in many systems and requires researchers to be able to isolate all selective events in the system" → 대안으로 persistence filter 제안. 다만 "Whereas shadow runs filter out the effect of neutral processes, the persistence filter does not do so entirely"이며 둘이 다른 정보를 잡으므로 번갈아 쓸 가치가 있다고 함 — [1차] [Dolson et al. 2019 (MODES)](https://par.nsf.gov/servlets/purl/10104628)

### Inferences
- **1단계에서 쓸 대조 두 가지와 비용.**
  - (A) *장부 shadow*(Bedau식): 본 실행의 사건 기록(출생 시각, 사망 시각)만 읽어, 가상의 id 목록에서 무작위 부모·무작위 희생자를 뽑는다. 세계 시뮬레이션이 없어 사건 수에 비례하는 비용 → **상시 가능**. 얻는 것: "선택이 없을 때 계통·genotype이 얼마나 오래 버티는가"의 기준선(activity, 계통 수 감소 속도, 합류 시간).
  - (B) *lockstep 세계*(Yaeger식): 물리·뇌까지 다 돌리므로 본 실행과 같은 CPU. 얻는 것: 뇌 크기·행동 지표의 "선택 없는 추세". **가끔, 짧게**(예: 본 실행 체크포인트에서 시작해 수십 세대) 돌리는 일정형 측정으로 적합.
- **가장 중요한 실무 함정: component 정의.** 신경망 가중치가 실수값이고 매 출생마다 변이하면 "genotype 동일성" 기준 component는 전부 1개체짜리가 되어 Bedau가 보인 μ≈1 경우(본 실행 = shadow, class 1)가 **인공적으로** 나온다. 1단계에서는 component를 (i) 창시자 id 기반 **계통(clade) tag**, (ii) 이산 형질(뉴런 수, 감각 수 등 구조 유전자 값 — de Pinho & Sinapayen 방식), (iii) 양자화한 행동 서술자 중에서 골라야 한다.
- **Yaeger 결과의 올바른 독해.** "대조 실행에서 복잡도가 더 높았다"는 것은 측정이 틀렸다는 뜻이 아니라, *복잡도 같은 형질 추세는 방향이 어느 쪽이든 선택의 증거가 아니며, driven과 passive의 차이(부호 불문)가 선택의 증거*라는 뜻이다. 따라서 뇌 크기·복잡도 추세는 "성장" 판정 기준이 아니라 **대조와의 차이로만** 해석해야 하고, "성장" 자체는 2번의 경쟁 분석으로 잰다.
- shadow 정규화 후 양의 신호는 Miconi의 반례(상속 안 되는 운) 때문에 그것만으로는 부족하다. **Fitness Transmission(상속되는 번식 성공)**과 함께 볼 때 "선택이 작동한다"고 말할 수 있다.

### Gaps
- Channon 2001/2003/2006 본문(Geb에서 component = gene의 정확한 정의, snapshot 간격, "unbounded" 판정의 수치 기준)은 열지 못했다. 위 공식은 de Pinho & Sinapayen 2026이 옮긴 형태다.
- Yaeger의 lockstep에서 무작위로 뽑힌 부모·희생자의 **나이·에너지 조건**(예: 갓 태어난 개체도 부모가 될 수 있는지)은 확인하지 못했다.
- Bedau의 "unbounded/positive" 정의는 t→∞ 극한이라 유한 실행에서의 판정 규칙(통계 검정)이 없다 — de Pinho & Sinapayen도 눈으로 분류했다고 한계를 적었다.

---

## 4. 열린 진화(open-endedness) 지표 — 무엇이 작은 기계에서 상시 계산 가능한가

### Takeaway
MODES 네 지표 중 **change·novelty·ecology는 계통 추적만 있으면 싸게** 계산된다(집합 연산과 Shannon entropy). **complexity는 knockout(자리마다 적합도 재평가)이 필요해 비싸고**, 적합도 함수가 없는 세계에서는 곧바로 적용되지 않는다. 형식적 정의(Adams 등)와 범주 논의(York/Tokyo)는 1단계 판정 기준으로 쓰기에는 너무 멀다.

### Cited Findings
- **Persistence filter.** 시점 A의 개체마다 계통 id를 붙이고, t세대 뒤(A+t)에 **후손이 남아 있는 계통의 시점-A component만** "의미 있다"고 센다. 지표 집계가 t세대 늦어지는 대신 곧 사라질 해로운 변이로 수치가 부풀지 않는다 — [1차] [Dolson et al. 2019](https://par.nsf.gov/servlets/purl/10104628)
- **t 고르기.** 해로운 변이만 거르려면 작은 t로 충분(선행 연구는 사실상 t = 1을 보조 필터로 씀). 중립 변이까지 거르려면 합류 시간(coalescence time)보다 크게: 선택 없는 잘 섞인 무성 집단 N개체의 **중앙 합류 시간은 2N 세대**이나 분포가 지수형이라 완벽 보장은 비현실적; 선택이 있으면 합류가 훨씬 빨라 "relatively low values of t yield still effective filters". **t는 세대 단위**로 재고, 집단 간 비교는 같은 t로. 실험에서는 **t = 집단 크기**를 썼고 "seems to yield an adequate filter" — [1차] [Dolson et al. 2019](https://par.nsf.gov/servlets/purl/10104628)
- **네 지표의 식.** F = 현재 시점에 필터를 통과한 component 집합, F′ = 직전 시점의 통과 집합, S = 지금까지 통과한 적 있는 모든 component.
  - change = |{c ∈ F : c ∉ F′}|
  - novelty = |{c ∈ F : c ∉ S}|
  - complexity = 필터 통과 component 중 **informative site 수의 최댓값**(자리 하나씩 knockout해 적합도가 떨어지는 자리 수)
  - ecology = −Σ_{c∈F} P(c) log₂ P(c) (Shannon entropy; evolutionary activity의 D에 해당)
  — [1차] [Dolson et al. 2019](https://par.nsf.gov/servlets/purl/10104628)
- **해석 기준.** 선택압이 하나뿐인 빈 환경에서는 합류 구간마다 한 genotype이 휩쓰므로 기대 change ≈ 1. 통과 개체가 하나면 그 개체가 전체의 공통 조상선 위에 있다는 뜻, **여럿이 통과하면 공존을 돕는 생태 동역학의 증거**. t를 키우면 change가 줄어든다 — [1차] [Dolson et al. 2019](https://par.nsf.gov/servlets/purl/10104628)
- **비용 절감책.** novelty는 "통과한 적 있는 모든 taxon"을 기억해야 해 메모리가 가장 큼 → Bloom filter(거짓 음성은 지표를 보수적으로 만들 뿐). 계통수는 살아 있는 후손이 없는 taxon을 가지치기하고 "현재 세대 − t" 이전에 죽은 taxon을 버리면 무한히 커지지 않음. "With these optimizations, MODES metrics can be implemented with minimal overhead" — [1차] [Dolson et al. 2019](https://par.nsf.gov/servlets/purl/10104628)
- **MODES 실험 규모.** Avida 집단 크기 500/1000/2000, 조건당 **30회 반복**, Kruskal–Wallis 뒤 사후 Wilcoxon, 그림은 bootstrap 95% 신뢰구간. NK landscape에서 fitness sharing만이 ecology를 안정적으로 만들었음 — [1차] [Dolson et al. 2019](https://par.nsf.gov/servlets/purl/10104628)
- **범주(York → Tokyo).** York(Taylor et al. 2016): 1. 적응적 새로움의 지속 생성((a) 새 적응, (b) 새 종류의 개체, (c) major transition, (d) evolvability의 진화), 2. 복잡도의 지속 성장((a) 개체, (b) 상호작용). Tokyo(Packard et al. 2019): 1. 흥미로운 새 개체·상호작용, 2. evolvability의 진화, 3. major transitions, 4. semantic evolution. "there is not a simple single test for the phenomenon"; type 1의 "interesting"은 모호·주관적이라고 스스로 지적 — [1차] [Packard et al. 2019, arXiv:1909.04430](https://arxiv.org/pdf/1909.04430) (Taylor et al. 2016 원문은 [2차])
- **적응 없이도 "열림"은 나온다.** "algorithms like novelty search generate ongoing novelty without any evident adaptive pressure" — 새로움 지표만으로는 성장을 말할 수 없다 — [1차] [Packard et al. 2019](https://arxiv.org/pdf/1909.04430)
- **Adams et al. 2017의 형식 정의.** unbounded evolution = 고립계의 기대 Poincaré 재귀 시간 안에 패턴이 반복되지 않음; innovation = 고립계에서 관측되지 않는 궤적; 둘 다여야 OEE. 셀룰러 오토마타로 검증 — [2차] [Adams et al. 2017 (검색 요약)](https://www.arxiv.org/abs/1607.01750)
- **Geb의 확장성 검정(Channon 2019).** 세계 길이(=집단 크기 상한)와 개체당 최대 뉴런 수를 **함께** 키우면 최대 개체 복잡도가 무한정 늘 수 있는 것으로 보이고, 하나만 키우면 점근적으로 유계. 실행 시간은 "years", 실행당 번식 수십억 회 — [2차] [Channon 2019, Artificial Life, DOI 10.1162/artl_a_00285 (초록 수준)](https://www.bohrium.com/paper-details/maximum-individual-complexity-is-indefinitely-scalable-in-geb/812769391648178177-2887)

### Inferences
- **상시 계산 가능(계통 기록만 필요):** change, novelty, ecology(component = 계통 tag 또는 이산 형질), diversity D(t), Fitness Transmission, 장부 shadow 대비 activity. 모두 출생·사망 사건당 O(1) 갱신 + 주기적 집합 연산.
- **일정형(가끔):** complexity(knockout 대신 쓸 수 있는 대용: "뉴런/시냅스를 하나씩 끊은 복제 개체를 단독 분석에 넣어 에너지 섭취가 유의하게 떨어지는 연결 수" — 적합도 함수가 없으므로 단독 분석 결과를 적합도 대용으로 써야 하고, 한 번에 수십~수백 회 arena가 필요해 비싸다).
- **1단계 판정에서 제외:** Adams의 Poincaré 기준, Tokyo 범주, Channon의 무한 확장성 검정(연 단위) — 1단계의 질문은 "열려 있나"가 아니라 "선택이 작동하고 적응이 쌓이나"다.
- 집단이 수백이면 t = 집단 크기(세대)라는 MODES의 권고는 "하루 10세대" 속도에서 **수십 일**의 지연을 뜻한다. 1단계에서는 해로운 변이만 거르는 짧은 t(예: 5~10세대)와 긴 t 두 가지를 같이 기록해 두는 것이 현실적이다.

### Gaps
- "assembly"/압축 기반 복잡도(예: assembly index, Lempel–Ziv로 뇌·행동열을 압축)로 인공생명 세계의 성장을 잰 1차 자료는 이번 조사에서 열지 못했다 — 근거 없이 채택하지 말 것.
- Banzhaf et al. 2016(variation/innovation/emergence), Stepney의 정의는 원문 확인 못 함.
- 2020~2026년 ALIFE 학회·Artificial Life 학술지의 "열림 측정" 논문은 de Pinho & Sinapayen 2026 하나만 직접 확인했다. 그 밖의 최신 지표(예: 계통수 모양으로 생태·선택압을 추정하는 연구)는 제목 수준으로만 보았다.

---

## 5. 싸게 기록할 수 있는 행동·생태 지표와 계통 추적 도구

### Takeaway
선행 연구가 "실행이 살아 있는가"를 보는 데 실제로 쓴 계기판은 놀랄 만큼 단순하다: Polyworld는 **출생/(출생+생성) 비**와 **먹이 에너지 수지**를, Evosphere는 집단 크기·kill 수·FT를 썼다. 계통 추적은 phylotrackpy(pip, ALife 표준 CSV)로 바로 붙일 수 있다.

### Cited Findings
- **Polyworld의 상시 그래프.** (1) 집단 크기(전체·영역별), (2) ad hoc 적합도의 과거 최대·현재 최대·현재 평균, (3) **"born"(짝짓기로 태어남) 수 / ("born" + "created"(시스템이 만들어 넣음)) 비** — "will start at 0.0 and asymptote to 1.0 for successful simulations ... it will peak well below 1.0 for unsuccessful simulations", (4) (먹이 에너지 유입 − 유출)/(유입 + 유출), −1~1, 에너지가 보존되는 세계면 0으로 수렴. (3)·(4)가 "important gauges of the course of the simulation" — [1차] [Yaeger 1994, PolyWorld: Life in a New Context (Artificial Life III; 이 세션에서 받아 둔 사본으로 확인, 링크는 프로젝트 페이지)](https://shinyverse.org/larryy/PolyWorld.html)
- **Evosphere가 본 것.** 종별 집단 크기, 평균 몸 부피, 구간별 kill 수, Fitness Transmission(5구간 이동평균); FT 급등 시점이 kill 수 증가·집단 크기 분화와 일치 — [1차] [Miconi 2008, 10장](https://etheses.bham.ac.uk/id/eprint/148/1/Miconi08PhD.pdf)
- **Hide-and-seek의 행동 통계와 그 한계.** 추적한 것: 상자·경사로의 최대 이동량, 에피소드 끝에 잠긴 물체 수(3 seed 평균). 그러나 "the choice to track absolute movement does not illuminate which direction agents moved", 환경이 커질수록 충분한 지표 설계가 "difficult and costly" — [1차] [Baker et al. 2019](https://ar5iv.labs.arxiv.org/html/1909.07528)
- **Yaeger 2008이 뇌에서 잰 것.** 개체 일생의 뉴런 활동 기록에서 neural complexity를 계산해 1,000 step 구간 집단 평균; 유전체 쪽은 genomic consistency(선택 있으면 크고 안정, 없으면 무작위 편집으로 최소) — [1차] [Yaeger et al. 2008](https://ar5iv.labs.arxiv.org/html/1112.4906)
- **phylotrackpy 사용법.** `pip install phylotrackpy`; `from phylotrackpy import systematics`; `syst = systematics.Systematics(lambda org: org.genotype)`(taxon 함수가 개체→문자열); 출생 `org.taxon = syst.add_org(org, parent.taxon)`(첫 개체는 부모 인자 없이); 사망 `syst.remove_org(org.taxon)`; 저장 `syst.snapshot("phylo.csv")`(ALife data standard 형식), 불러오기 `load_from_file`. 옵션 `store_active`, `store_ancestors`, `store_all`(계통수가 "very large very fast") — [1차] [phylotrackpy quickstart](https://phylotrackpy.readthedocs.io/en/latest/quickstart.html)
- **phylotrack이 주는 통계와 자료 형식.** 평균 phylogenetic depth, 평균 origin time, MRCA origin time, Shannon diversity, Colless-like index, Sackin index, phylogenetic diversity(Faith 1992), evolutionary distinctiveness, 쌍별 거리 등. 표준 CSV 열: 고유 id, ancestor list, origin time, destruction time. C++ header-only 라이브러리의 Pybind11 래퍼 — [2차] [Phylotrack 논문 arXiv:2405.09389 (검색 요약; 통계 함수 이름은 API 문서로 확인 필요)](https://arxiv.org/pdf/2405.09389)
- **계통 지표의 선행 제안.** lineage length, mutation accumulation, phenotypic volatility 세 종류의 계통 측정 — [2차] [Dolson et al. 2018/2020 "Interpreting the tape of life" (검색 요약)](https://par.nsf.gov/servlets/purl/10308959)
- **MODES의 가지치기 규칙**(계통 기록이 안 터지게): 살아 있는 후손이 없는 taxon 제거 + (현재 − t)세대 이전에 죽은 taxon 제거 — [1차] [Dolson et al. 2019](https://par.nsf.gov/servlets/purl/10104628)

### Inferences
- **1단계 상시 기록 권장 목록**(전부 사건 기록 또는 주기적 집계로 계산, 추가 시뮬레이션 없음):
  - *사건 기록(append-only)*: 출생(시각, 자식 id, 부모 id, 세대 깊이, 유전체 해시/구조 요약), 사망(시각, id, 원인, 수명, 평생 에너지 섭취, 자식 수).
  - *주기 집계*: 집단 크기와 그 분산, 구간당 출생·사망 수, **출생/(출생+재파종) 비**(Polyworld 지표 (3) — 재파종 장치를 두는 경우), 평균 수명, 평균·최대 세대 깊이, 개체당 에너지 섭취 효율(섭취/소비), 세계 에너지 수지, 뇌 크기 분포(뉴런·시냅스 수의 분위수), 살아남은 창시 계통 수와 MRCA 시각, 유전체 쌍별 거리(무작위 표본 수십 쌍), 구간별 Fitness Transmission, MODES change/novelty/ecology.
  - *행동 서술자(표본 개체 수십의 궤적에서)*: 평균 속력, 경로 굴곡도(이동 거리/변위), 먹이 근처 체류 시간 비율, 정지 시간 비율.
- phylotrackpy의 taxon 함수는 실수값 유전체에 그대로 쓰면 개체마다 taxon이 달라진다(3번의 component 함정과 같은 문제). taxon = "구조 유전자 튜플" 또는 "창시자 id" 같이 **거칠게** 정의해야 통계가 의미를 가진다. 의존성을 늘리기 싫으면 표준 CSV 네 열(id, ancestor_list, origin_time, destruction_time)만 직접 써 두고 나중에 도구로 읽어도 된다.
- Baker의 경고대로 행동 통계는 "무엇이 바뀌었나"를 보여 줄 뿐 "나아졌나"는 말해 주지 않는다. 성장 판정은 2번의 분석으로, 행동 통계는 그 **설명**(무엇이 달라져서 이겼나)으로 쓴다.

### Gaps
- 경로 굴곡도·속력·먹이 근처 체류 시간 같은 행동 서술자를 인공생명 세계의 적응 지표로 **검증한** 1차 자료는 찾지 못했다(목록은 요청서의 후보를 Baker 등의 일반 원칙에 비춰 정리한 것).
- phylotrackpy의 통계 함수 정확한 이름·시그니처와 macOS(ARM) wheel 제공 여부는 quickstart에 없어 확인 못 했다.
- 유성생식(두 부모) 계통을 phylotrackpy가 어떻게 다루는지(부모 taxon 하나만 받는 API로 보임) 확인 필요.

---

## 6. 일반화 검정 — "본 적 없는 상대"와 "본 적 없는 환경"

### Takeaway
선행 연구의 일반화 검정은 세 가지다: (1) **독립 반복 실행에서 나온 상대**를 검증 집합으로 쓰기(Miconi의 cross-validation, Simione & Nolfi), (2) 조건이 다른 실험끼리 교차 대결, (3) 과제를 바꾼 transfer 검정(Baker). 독립 실행 **2개만으로도** 단일 실행 분석이 못 보는 차이가 드러났다.

### Cited Findings
- **Cross-validation의 원리.** "Because opponents from different runs do not coevolve together, they do not influence each other's evolution, and therefore offer test elements distinct from the training set." N개 실행을 골라 각 집단의 연속된 챔피언을 다른 모든 집단의 모든 챔피언과 대결. 비용은 (kN)²/2로 커지지만 "we can still gain valuable information from very low values of N" — **2개 실행**으로 시연 — [1차] [Miconi 2008, 6.5](https://etheses.bham.ac.uk/id/eprint/148/1/Miconi08PhD.pdf)
- **단일 실행 분석이 놓친 것.** 자기 실행 안에서 모든 상대를 이긴("clean sheet") 챔피언들 가운데 일부만 다른 실행의 상대에게도 강했다(최대 150점 근처 대 훨씬 낮은 점수). 한 집단은 자기 Master Tournament에서는 완승 기록이 있었지만 교차 검증에서는 다른 실행의 집단보다 일관되게 약했다 — "a single-record analysis could not have detected this" — [1차] [Miconi 2008, 6.5](https://etheses.bham.ac.uk/id/eprint/148/1/Miconi08PhD.pdf)
- **신뢰도는 상대 수에 비례.** "The reliability of this estimation is of course directly related to the number of other individuals against which this cross-validation is performed" — [1차] [Miconi 2008, 6.5](https://etheses.bham.ac.uk/id/eprint/148/1/Miconi08PhD.pdf)
- **Simione & Nolfi의 두 방식.** (a) global progress는 "다른 반복 실험에서 진화한 상대"를 validation set으로 써서 잴 수 있다. (b) 조건 간 비교: Standard 조건 5회 반복의 150,000세대 개체를 Simplified·Vanilla 조건 5회 반복의 같은 세대 상대와 대결 → Standard 포식자는 Simplified(p < .05)·Vanilla(p < .001) 먹이에게 자기 조건 먹이보다 유의하게 더 잘함 — [1차] [Simione & Nolfi 2019](https://arxiv.org/pdf/1909.08303)
- **검증 상대를 선택에 쓰는 알고리즘.** 진화 중인 개체를 훈련 상대 n과 따로 떼어 둔 validation 상대에 모두 평가해, "훈련 상대에는 잘하지만 validation 상대에는 못해진" 기회주의적 변이를 버림 → 이 필터가 있어야 장기 global progress가 나왔다(n = 10, 진화 단계 100세대, 5회 반복) — [1차] [Simione & Nolfi 2019](https://arxiv.org/pdf/1909.08303)
- **Transfer 검정(Baker et al.).** hide-and-seek로 사전학습한 정책을 5개 단일 개체 과제(Object Counting, Lock and Return, Sequential Lock, Construction from Blueprint, Shelter Construction)에 미세조정하고, 기준선 2개(처음부터 학습, count-based 내재 동기 사전학습)와 비교. seed 3개(분산 큰 과제는 6개), 90% 신뢰구간. 결과는 3개 과제에서 약간 우세, 1개 열세, 1개는 최종 성능 같고 더 느림 — 차이는 "not drastic". 창발 단계가 뒤일수록 Lock and Return 성능이 올라 6단계 정책이 1단계보다 약 20% 좋음 — [1차] [Baker et al. 2019](https://ar5iv.labs.arxiv.org/html/1909.07528)
- **무작위 상대의 한계.** 무작위 생성 상대와 평가하면 항상 새 상대라 global progress를 직접 촉진하지만 "the efficacy of the opponents does not increase across generations"라 강한 상대를 이기는 개체는 못 만든다 — [1차] [Simione & Nolfi 2019](https://arxiv.org/pdf/1909.08303)

### Inferences
- **1단계의 "본 적 없는 상대" 구현:** 본 실행 M과 seed만 다른 **독립 실행 V**를 하나 더 둔다(같은 CPU를 나눠 쓰므로 속도가 반으로 줄어듦 — 또는 V를 더 작은 세계로). 검정: M의 체크포인트(옛것, 지금 것)를 각각 **V의 현재 개체**와 arena 경쟁시켜 r을 구한다. "M의 지금 것이 M의 옛것보다 V 상대에게 더 잘한다"면 global progress의 증거(Simione & Nolfi의 global 판정을 옮긴 것). Miconi의 결과로 보아 V 하나(N=2)로도 정보가 있다.
- **"본 적 없는 환경" 구현:** 단독 분석과 경쟁 분석을 (i) 본 실행과 다른 먹이 배치 seed, (ii) 먹이 밀도·분포가 다른 변형 세계 2~3개에서 반복. 본 세계에서만 이기고 변형 세계에서 지면 "그 지형에 과적합".
- **무작위 뇌 기준선**은 싸고 유용하지만 Simione & Nolfi의 지적대로 "약한 상대"라서, 1단계 초반(선택이 작동하는지)까지만 변별력이 있다. 그 뒤로는 옛 체크포인트와 V 상대가 기준선이 되어야 한다.
- 검증용 실행 V의 개체를 M에 **섞어 넣지 않는다**(검증 집합 오염 — Miconi의 training/test 논리).

### Gaps
- 적합도 함수 없는 인공생명 세계에서 "다른 seed의 병렬 실행 개체를 검증 상대로" 쓴 구체 선례는 찾지 못했다(Miconi·Simione & Nolfi는 둘 다 명시적 대결 과제).
- 변형 환경을 몇 개, 얼마나 다르게 만들어야 하는지에 대한 정량 지침은 찾지 못했다.

---

## 7. 통계 — 반복 수, 분산, 쓰이는 검정

### Takeaway
선행 연구의 반복 수는 **5~12**(실험 단위)가 보통이고 MODES·Miconi의 장난감 모형만 30~50이다. 검정은 단순하다: paired/one-tailed t-test, Mann–Whitney, Kruskal–Wallis+Wilcoxon, bootstrap 신뢰구간. 대장균의 "12회 반복으로 ±1%"는 개체 수가 수억이라 가능한 것이므로, 수백 개체 세계에서는 표류 잡음을 중립 대조로 **직접 재서** 반복 수를 정해야 한다.

### Cited Findings
- **LTEE.** 12회 반복 + 3일 경쟁이면 근접한 두 균주의 상대 적합도를 **±1%(95% 신뢰구간)** 정밀도로 잴 수 있다 — [1차] [Barrick Lab](https://barricklab.org/twiki/bin/view/Lab/ProceduresLongTermCompetitions)
- **Wiser & Lenski 2015.** 15 시점 × 3 방법 × **10 반복 block** = 450회 분석. 정밀도 척도는 각 시점의 **변동계수(CV = SD/평균)**. 시간이 CV에 매우 유의한 영향(p < 0.0001) — 적합도 격차가 커질수록 잡음 증가. 분석: block을 무선 요인으로 둔 ANOVA, 100만 회 bootstrap — [1차] [Wiser & Lenski 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4427439)
- **Yaeger et al. 2008.** 10쌍 seed, 시점별 **paired t-test**(df = 9, 임계 1.833, p < 0.05) — [1차] [Yaeger et al. 2008](https://ar5iv.labs.arxiv.org/html/1112.4906)
- **Simione & Nolfi 2019.** 5회 반복, **one-tailed t-test (N = 5)** — [1차] [Simione & Nolfi 2019](https://arxiv.org/pdf/1909.08303)
- **Nolfi & Floreano 1998.** 10회 반복; "variability in different replication is high"라고 두 번 명시 — [1차] [Nolfi & Floreano 1998](https://infoscience.epfl.ch/record/63899)
- **MODES.** 조건당 30회 반복, Kruskal–Wallis → 사후 Wilcoxon, bootstrap 95% 신뢰구간 — [1차] [Dolson et al. 2019](https://par.nsf.gov/servlets/purl/10104628)
- **Miconi(FT).** 50회 실행; (i) 장기 평균이 0보다 큰 실행 수에 대한 이항(부호) 검정, (ii) 선택 유·무 두 표본의 two-tailed Mann–Whitney — [1차] [Miconi 2008, 9.5](https://etheses.bham.ac.uk/id/eprint/148/1/Miconi08PhD.pdf)
- **Baker et al. 2019.** seed 3개(분산 큰 과제 6개), 90% 신뢰구간 — [1차] [Baker et al. 2019](https://ar5iv.labs.arxiv.org/html/1909.07528)
- **de Pinho & Sinapayen 2026.** 20회 실행, 곡선을 눈으로 bounded/unbounded/unclear 분류(저자들이 한계로 명시) — [1차] [de Pinho & Sinapayen 2026](https://arxiv.org/abs/2603.01701)

### Inferences
- **표류 잡음의 대략 크기(출처 없는 표준 집단유전학 근사 — 추론임).** 중립인 두 유형을 빈도 p로 섞어 유효 크기 N_e 집단에서 T세대 두면 빈도의 분산은 대략 p(1−p)·T/N_e. arena 200개체, p = 0.5, T = 5세대면 표준편차 ≈ √(0.25×5/200) ≈ 0.08 — **한 arena로는 점유율 50% 대 58%도 구별 못 한다**. 20회 반복이면 표준오차 ≈ 0.018. 실제 값은 공간 구조·번식 분산에 따라 더 클 수 있으므로 A 대 A 중립 대조로 직접 잰다.
- **권장 절차:** (1) 먼저 A 대 A 중립 arena를 20~30회 돌려 r의 표준편차 s₀를 얻는다(한 번만 하면 됨, 세계 규칙이 바뀌면 다시). (2) 검출하고 싶은 최소 효과 δ에 대해 반복 수 n ≈ (2.8·s₀/δ)²(양측 5%, 검정력 80%의 통상 근사 — 추론). (3) 본 측정은 seed를 짝지어(같은 먹이 배치 seed에서 C 대 A와 A 대 A) paired 비교 — Yaeger의 paired 설계와 같은 이유.
- **검정 선택:** arena 반복 r값들에 대해 one-sample 부호 검정 또는 Wilcoxon signed-rank(분포 가정 없음), 효과 크기는 bootstrap 95% 신뢰구간으로 보고. 5회 반복의 t-test(Simione & Nolfi)는 선례가 있지만 최소한이다.
- **다중 비교 주의:** 체크포인트 쌍을 많이 보면 우연히 유의한 칸이 생긴다. 개별 칸의 p값보다 **시차에 대한 단조 추세**(Spearman)와 transitive 비율 같은 요약 통계 하나로 판정하는 편이 안전하다.
- **CPU 예산:** arena 하나가 "200개체 × 5세대"라면 본 실행 하루치(10세대)의 일부에 해당. 20회 반복 × 시차 4개 = 80 arena를 매일 돌리는 것은 과하므로, **주 1회** 또는 **본 실행이 100세대 진행할 때마다** 한 묶음으로 일정화하는 것이 현실적이다.

### Gaps
- 공간이 있는 소규모(수백 개체) 인공생명 경쟁 분석의 **실측 분산**을 보고한 자료는 찾지 못했다 — 위 표류 추정은 문헌값이 아니라 근사 계산이다.
- 순환 여부에 대한 정식 통계 검정(예: transitive 비율의 귀무 분포)은 문헌에서 찾지 못했다. 중립 대조 행렬(같은 체크포인트끼리의 arena로 채운 행렬)에서 얻은 transitive 비율 분포를 귀무 기준으로 쓰는 방안은 추론이다.

---

## 8. 1단계 "살아 있는 세계" 성공·실패 기준과 첫 적응까지 걸리는 세대 수

### Takeaway
선행 프로젝트의 1단계 성공 기준은 "**재파종 없이 번식만으로 집단이 유지된다**"(Polyworld의 Successful Behavior Strategy)이고, 그다음 단계가 "선택이 작동한다"(FT > 0, 대조와의 차이), 마지막이 "적응이 쌓인다"(과거 대비·독립 상대 대비 우위)다. 첫 적응 신호는 Polyworld에서 수십 세대 규모, 공진화 로봇에서 초반 수십 세대에 나타났다는 정황이 있으나 정밀한 공통 수치는 없다.

### Cited Findings
- **Polyworld의 성공 정의.** "Populations of organisms that have evolved a set of behaviors which allow them to replenish their numbers through births, with no further creations (after some point in time), are said to exhibit a Successful Behavior Strategy (SBS)". "A simulation is considered 'successful' if and only if some number of species emerge with a SBS; these populations are capable of sustaining their numbers through their mating behaviors, and thus organism creations cease." — [1차] [Yaeger 1994 (사본으로 확인; 링크는 프로젝트 페이지)](https://shinyverse.org/larryy/PolyWorld.html)
- **SBS 이전의 보조 장치.** SBS가 나오기 전에는 ad hoc 적합도 함수(먹기·짝짓기·수명 채우기·에너지 남기고 죽기·움직이기에 보상)를 쓰는 steady-state GA로 돌리며 최소 개체 수를 보장(부족하면 무작위 개체 생성, 최상위 N의 자손 생성, 드물게 역대 최고 개체 복귀). "Once an SBS has emerged, there is no fitness function except survival." 일부 실행은 첫 seed 집단에서 바로 SBS에 도달하고, "Others never acquire an SBS, and are considered unsuccessful simulations" — [1차] [Yaeger 1994](https://shinyverse.org/larryy/PolyWorld.html)
- **성공의 계기판 판독.** born/(born+created) 비가 1.0에 점근하면 성공, 1.0보다 한참 아래에서 꺾이면 실패 — [1차] [Yaeger 1994](https://shinyverse.org/larryy/PolyWorld.html)
- **Evosphere의 "적응 진화 시작" 판정.** 초기에 FT가 0 근처인 단계("creatures are so ill-adapted that any damage they inflict or suffer arises mostly through chance") → 지속적 이동이 나타나며 FT 급등, kill 수 증가와 동시. 저자는 육안 관찰과 FT 두 가지 모두로 적응 진화를 확인했다고 적음 — [1차] [Miconi 2008, 10장](https://etheses.bham.ac.uk/id/eprint/148/1/Miconi08PhD.pdf)
- **Polyworld에서 선택 효과가 보이는 시간.** driven 실행이 passive보다 유의하게 빨리 복잡도가 느는 구간은 처음 약 4,000 timestep(전체 30,000 step ≈ 약 400세대 추정, 범위 100~1,200) — [1차] [Yaeger et al. 2008](https://ar5iv.labs.arxiv.org/html/1112.4906)
- **그 뒤는 정체가 흔하다.** driven 10회 중 7회는 복잡도가 정체 상태로 30,000 step 끝까지 유지, 3회만 뒤늦게 2차 상승 — [1차] [Yaeger et al. 2008](https://ar5iv.labs.arxiv.org/html/1112.4906)
- **공진화의 초반 진보와 이후 붕괴.** 표준 공진화(100세대)에서 양쪽이 나아지는 구간 뒤에 "sudden drops in performance"가 오고, 10회 평균으로는 세대에 따른 향상이 없음 — [1차] [Nolfi & Floreano 1998](https://infoscience.epfl.ch/record/63899). "진보는 처음 20세대 정도에만" 이라는 요약은 EPFL 강의 자료의 표현 — [2차] [EPFL 강의 슬라이드(검색 요약)](https://baibook.epfl.ch/slides/Coevolution.pdf)
- **장기 진보는 특별한 장치가 있어야.** archive·필터 없는 Vanilla 조건은 150,000세대 동안 historical·global progress 모두 없음 — [1차] [Simione & Nolfi 2019](https://arxiv.org/pdf/1909.08303)
- **Echo의 교훈(매개변수 영역).** 상호작용 확률과 대사세가 너무 낮으면 돌연변이율과 무관하게 적응 활동이 0(class 1); 돌연변이율이 0 근처거나 1 근처여도 class 1 — "살아는 있는데 적응은 없는" 세계가 흔한 실패 양상 — [1차] [Bedau, Snyder & Packard 1998](https://people.reed.edu/~mab/papers/alife6.pdf)
- **"열림" 검정을 통과 못 하는 것이 보통.** 2026년의 종분화 시뮬레이션도 정규화하면 20회 모두 bounded, new activity 0 — [1차] [de Pinho & Sinapayen 2026](https://arxiv.org/abs/2603.01701)

### Inferences
- **1단계 성공 기준안(사다리식 — 아래 단이 안 되면 위 단을 볼 필요 없음).**
  - **L0 존속:** 재파종(시스템이 개체를 만들어 넣기) 없이 집단이 유지된다. 지표: 출생/(출생+재파종) → 1.0에 점근하고 그 상태가 최소 수십 세대(예: 50세대 = 5일) 지속. 근거: Polyworld SBS.
  - **L1 선택 작동:** 구간별 Fitness Transmission의 이동평균이 지속적으로 0보다 크고(구간들의 부호 검정), 장부 shadow와 계통 존속 통계가 다르다. 근거: Miconi, Bedau.
  - **L2 기준선 초과:** 현재 체크포인트가 단독 분석에서 무작위 뇌·0세대 표본보다 에너지 섭취 효율·존속 시간이 높고, 경쟁 분석에서 r > 0(중립 대조 분포 밖). 근거: Lenski 분석의 직역.
  - **L3 누적(= "성장"):** 여러 시차의 조상에 대해 r > 0이고 시차가 클수록 r이 커진다(단조), 체크포인트 행렬의 transitive 비율이 중립 대조보다 높다. 근거: Simione & Nolfi의 historical progress 판정, Balduzzi 분해.
  - **L4 일반화:** 독립 실행 V의 상대, 변형 환경에서도 "지금 것 > 옛것". 근거: Miconi cross-validation.
- **실패·경고 판정안.**
  - *죽은 세계:* 출생 비가 1.0 한참 아래에서 꺾이거나 재파종 없이는 멸종(Polyworld의 "unsuccessful").
  - *떠내려감(drift):* L0은 되지만 FT ≈ 0, 모든 시차에서 r이 중립 대조 범위 안. Echo의 class 1에 해당 — 선택압이 약하거나(대사 비용·경쟁이 너무 낮음) 돌연변이가 너무 크거나 작다는 신호.
  - *순환:* r의 부호가 시차에 따라 뒤집히고(가까운 조상은 이기는데 먼 조상에게 짐) transitive 비율이 낮다.
  - *국소 정체:* L2까지는 되나 L3의 r이 시차에 대해 평평해짐 — Yaeger의 "10회 중 7회 정체"처럼 흔하며 실패라기보다 "1단계의 한계"로 해석.
  - *형질 추세만 있음:* 뇌 크기·복잡도가 늘지만 lockstep/shadow 대조에서도 같은 만큼(또는 더) 는다 → 선택이 아니라 표류.
- **시간 척도 추론.** Yaeger의 4,000/30,000 step × 약 400세대 ≈ **50세대 안팎**에서 선택 효과가 통계적으로 보였다(세대 추정 범위가 100~1,200이라 12~150세대로 넓게 잡아야 함). "하루 10세대"면 **L1·L2 신호는 1~2주 안에** 기대할 수 있고, 2~3주가 지나도 FT ≈ 0이면 세계 매개변수(대사 비용, 먹이 밀도, 돌연변이 크기)를 의심하는 것이 합리적이다. "100세대 전과 비교"는 10일 간격이므로 L3 판정에는 최소 **한 달 이상**(시차 3~4개)이 필요하다.
- Polyworld도 초기에는 ad hoc 적합도로 집단을 떠받쳤다. 1단계에서 초기 멸종이 잦으면 "재파종 장치를 켜고 출생 비를 지켜보다가 1.0에 가까워지면 끄는" 방식이 선례에 부합하되, 재파종이 켜진 동안의 기록은 성장 판정에서 제외해야 한다(외부에서 유전체가 계속 들어오면 계통·경쟁 분석이 오염됨).

### Gaps
- "재파종 없이 N세대 유지"의 **N에 대한 문헌 기준**은 없다(Polyworld는 "어느 시점 이후 생성이 멎음"이라고만 정의). 위 50세대는 제안값이다.
- "진화한 먹이 찾기가 무작위 뇌 기준선을 몇 % 넘어야 하는가"의 문턱을 제시한 1차 자료는 찾지 못했다.
- LTEE의 초기 적응 속도(예: 처음 2,000세대의 적합도 상승 폭)는 Lenski 1991·Wiser et al. 2013 원문을 열지 못해 수치를 싣지 않았다.
- Evosphere에서 FT가 급등하기까지 걸린 세대 수는 본문 발췌에서 수치를 확인하지 못했다(그림으로만 제시).
- Geb·Tierra·Avida 계열의 "1단계 성공" 기준(자기 유지 집단의 최소 지속 시간 등)은 이번 조사에서 1차 자료로 확인하지 못했다.
