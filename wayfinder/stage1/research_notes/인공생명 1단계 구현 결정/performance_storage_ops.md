# 인공생명 1단계 구현 공학: 성능·재현성·체크포인트·저장·운영·관찰 화면 (CPU 전용 Python, 2026-11-04 조사)

읽는 법
- 출처 표시가 "(요약으로만 확인)"인 항목은 검색 결과 요약만 보았고 원문을 열지 못한 것이다. 수치를 그대로 쓰기 전에 원문 확인이 필요하다.
- "직접 측정" 항목은 이번 조사에서 개발용 맥(Apple M3, 성능 코어 4 + 효율 코어 4, Python 3.10.12, numpy 2.2.6(BLAS는 Accelerate), numba 0.67.0)에서 돌린 값이다. 스크립트는 이 노트와 같은 폴더의 `bench_alife.py`에 있다. 운영 기기(Ryzen 7 255 VM, 12 vCPU, Linux, OpenBLAS)에서는 재지 않았으므로, 같은 스크립트를 VM에서 한 번 돌려야 최종 수치가 된다.
- 버전에 민감한 내용에는 버전과 날짜를 붙였다.

## 1. 계산 스택: numpy / numba / JAX-CPU / PyTorch-CPU / Taichi / Cython / Rust 중 무엇을 쓸까

### Takeaway
10/29 결정("Python + numpy 배치 배열")은 자료 구조(struct-of-arrays)로는 맞지만 계산까지 numpy만으로 하면 안 된다. 직접 재 보니 이웃 탐색과 시야(ray–circle)는 numpy 전체 broadcast가 numba 격자 루프보다 43~325배 느리고 N=5000에서는 임시 배열이 GB 단위라 메모리 상한에 걸린다. 반면 개체별 RNN 행렬곱은 numpy `matmul`도 충분히 빠르다. 권고는 "상태는 numpy 배열, 무거운 세 커널은 numba `@njit`(필요하면 `prange`)"이고, JAX-CPU·Taichi·Rust로 갈 근거는 찾지 못했다.

### Cited Findings

버전·플랫폼 지원
- Numba 최신 안정판은 0.68.1(2026-10-26)이고 Python 3.10 이상 3.16 미만, NumPy "1.22 <= version < 1.27 ; 2.0 <= version < 2.6", llvmlite 0.50.x, LLVM 22.x를 지원한다. 지원 플랫폼에 Linux x86_64, Linux arm64, macOS 11.0 이상 Apple Silicon이 모두 있다. `@jit` 자동 병렬화는 64비트 플랫폼에서만 된다 — [Numba installing](https://numba.readthedocs.io/en/stable/user/installing.html)
- Python 3.14 지원은 Numba 0.63 계열에서 들어왔다(요약으로만 확인) — [Numba 0.63.0b1 공지](https://numba.discourse.group/t/ann-numba-0-63-0b1-llvmlite-0-46-0b1-python-3-14-support/3068)
- Numba 0.67.0은 안정판 문서에 2026-08-11로 적혀 있고 NumPy 2.5를 지원한다. 검색 요약은 "0.67.0 wheel이 처음에는 Python 3.14용만 제공된다"고 했으나(요약으로만 확인), 이 맥의 Python 3.10.12에 numba 0.67.0이 설치되어 동작하므로 그 문장은 현재 상태와 맞지 않는다 — [Numba 0.67.0 notes](https://numba.readthedocs.io/en/stable/release/0.67.0-notes.html)
- NumPy 현재 문서는 v2.5 Manual이다 — [NumPy parallel random](https://numpy.org/doc/stable/reference/random/parallel.html)
- Taichi 최신은 1.7.4(2025-07-31)이고 wheel은 CPython 3.9~3.13, Linux x86-64(glibc 2.27+), macOS 11+ ARM64, Windows x86-64용이다. 직전 판(1.7.3)은 2024-12-23으로 출시 간격이 약 7개월이고, 페이지에 Python 3.14 wheel은 없다 — [PyPI taichi](https://pypi.org/project/taichi/)
- zarr-python 현재 문서는 "Zarr requires Python 3.12 or higher"라고 적는다(이 맥의 pyenv Python은 3.10.12) — [zarr docs](https://zarr.readthedocs.io/en/stable/)

numba의 성능 특성(공식 문서의 측정값, i7-4790 기준)
- 명시적 루프 함수 `ident_loops`는 `@njit` 없이 25.2 s, 붙이면 0.670 s로 numpy 벡터화 버전(0.581 s)과 같은 수준이 된다 — [Numba performance tips](https://numba.readthedocs.io/en/stable/user/performance-tips.html)
- `fastmath`: 합 축약 `do_sum` 35.2 ms → `do_sum_fast` 17.8 ms. fastmath가 없으면 루프가 "must accumulate in strict order"라서 SIMD로 못 묶는다 — [Numba performance tips](https://numba.readthedocs.io/en/stable/user/performance-tips.html)
- `parallel=True`: `ident_parallel` 112 ms로 numpy 대비 약 5배. 병렬 합은 9.81 ms, 병렬 + fastmath는 5.37 ms — [Numba performance tips](https://numba.readthedocs.io/en/stable/user/performance-tips.html)
- Intel SVML은 `intel-cmplr-lib-rt` 패키지가 있을 때만 쓰이고, 기본은 1 ULP, `fastmath=True`면 4 ULP 정확도 버전을 쓴다(초월함수 결과가 설정에 따라 달라진다는 뜻) — [Numba performance tips](https://numba.readthedocs.io/en/stable/user/performance-tips.html)
- numba 안의 `np.dot`·`@` 등 선형대수는 1-D/2-D 연속 배열만 지원하고 "The implementation of these functions needs SciPy to be installed" — [Numba supported NumPy features](https://numba.readthedocs.io/en/stable/reference/numpysupported.html)
- `NUMBA_CPU_NAME=generic`으로 두면 `cache=True` 캐시가 같은 아키텍처·OS 안에서 옮길 수 있는 코드가 된다. `NUMBA_CPU_FEATURES`는 `+feature1,-feature2` 형식으로 LLVM에 넘긴다. `NUMBA_OPT` 기본값은 3 — [Numba envvars](https://numba.readthedocs.io/en/stable/reference/envvars.html)

스레드와 과다 구독(oversubscription)
- `NUMBA_NUM_THREADS` 기본값은 감지된 CPU 코어 수이고, numba를 import할 때 읽혀 고정된다. `set_num_threads(n)`은 그 이하로만 줄일 수 있다. 문서는 여러 프로세스를 기본 스레드 수로 돌리면 과다 구독이 된다고 경고하고 프로세스별 제한을 권한다 — [Numba threading layers](https://numba.readthedocs.io/en/stable/user/threading-layer.html)
- threading layer 탐색 순서는 tbb → omp → workqueue이고 workqueue만 항상 있다. Linux의 omp(GNU OpenMP)는 fork-safe가 아니다. `safe` 설정은 tbb 설치가 필요하다 — [Numba threading layers](https://numba.readthedocs.io/en/stable/user/threading-layer.html)
- `prange` 기본 스케줄은 반복 구간을 거의 같은 크기로 나누는 정적 방식이고, 동적 스케줄(`set_parallel_chunksize`)은 tbb에서만 된다. `prange` 안에서 list·set·dict를 동시에 고치면 "corruption or an access violation" 가능성이 높다 — [Numba parallel](https://numba.readthedocs.io/en/stable/user/parallel.html)
- threadpoolctl은 OpenMP와 주요 BLAS(MKL, OpenBLAS, FlexiBLAS)의 스레드 수를 실행 중에 제한한다(`threadpool_limits(limits=1, user_api='blas')`). OpenBLAS pthreads 설정은 프로세스 전체에, libgomp/libomp는 호출 스레드에만 적용된다. OpenMP 병렬 구간 안에서 BLAS를 순차로 쓰려면 `limits="sequential_blas_under_openmp"`가 따로 있다. 문서에 BLIS와 Apple Accelerate는 지원 목록에 없다 — [threadpoolctl README](https://github.com/joblib/threadpoolctl)
- NumPy를 소스에서 빌드할 때 BLAS 탐색 순서는 보통 "MKL, Accelerate, OpenBLAS, FlexiBLAS, BLIS, plain libblas/liblapack"이다(wheel이 무엇을 싣는지는 이 페이지에 없다) — [NumPy BLAS and LAPACK](https://numpy.org/doc/stable/building/blas_lapack.html)

작은 행렬 배치 곱
- 쌓인 작은 행렬 곱에서 numpy는 행렬마다 루프를 돌며 BLAS를 부르고, 아주 작은 행렬에서는 BLAS·numpy 호출 오버헤드가 대부분이라는 설명과, (5000,4,4)@(5000,4,1)에서 버전 간 2~3배 회귀 보고가 있다(요약으로만 확인) — [numpy issue #7569](https://www.github.com/numpy/numpy/issues/7569)

다른 프레임워크의 CPU 성적
- JAX 공식 FAQ: CPU에서 개별 배열 연산을 마이크로벤치마크하면 "you can generally expect NumPy to outperform JAX due to its lower per-operation dispatch overhead", JIT으로 묶은 연산열은 JAX가 유리할 수 있다 — [JAX FAQ](https://docs.jax.dev/en/latest/faq.html)
- QuantEcon 강의 PR(2025-11-01~02): CPU에 묶인 Monte Carlo(McCall 모형, 100,000회)에서 numba 순차 0.1259 s, numba 병렬 0.0234 s(8스레드, 5.39배), JAX 0.1529 s. warm-up 실수를 고친 뒤 24스레드 기기에서 numba 병렬 0.0100 s 대 JAX(CPU 강제) 0.0494 s로 "Numba is 4.92x faster than JAX". 라이브러리 버전과 CPU 모델은 적혀 있지 않다 — [QuantEcon lecture-python.myst PR #663](https://github.com/QuantEcon/lecture-python.myst/pull/663)
- pyhpc-benchmarks(NumPy, Numba, Aesara, JAX, TensorFlow, PyTorch, CuPy, Taichi 비교)의 결론: "The performance of JAX is very competitive, both on GPU and CPU", "Numba is a great choice on CPU if you don't mind writing explicit for loops", PyTorch는 "its CPU performance is not great for tasks with many slicing operations". CPU 실행은 단일 코어(`taskset -c 0`) 기준이고 README에 표 수치·날짜·버전은 없다(그래프가 이미지라 수치를 읽지 못함) — [pyhpc-benchmarks](https://github.com/dionhaefner/pyhpc-benchmarks)

직접 측정 (맥 M3, float32, 5회 중 최소, 단위 ms; 조건: 밀도 0.001개/단위², 반지름 3~7, 시야 거리 100, 시야각 2 rad, ray 16개, 격자 칸 25, 시야 안에 평균 약 31개체) — 출처 `bench_alife.py`(이 폴더)

| 커널 | N=1000,H=32 | N=1000,H=64 | N=1000,H=128 | N=5000,H=32 | N=5000,H=64 | N=5000,H=128 |
|---|---|---|---|---|---|---|
| (a) numpy `matmul` (N,H,H)@(N,H,1) + tanh (Accelerate) | 0.090 | 0.253 | 1.213 | 0.491 | 1.504 | 6.097 |
| (a) numpy `einsum('nij,nj->ni')` + tanh | 0.129 | 0.373 | 1.371 | 0.689 | 1.995 | 7.281 |
| (a) numba 순차, 엄격한 합 순서 | 0.591 | 2.209 | 9.113 | 3.328 | 11.753 | 45.783 |
| (a) numba 순차, fastmath | 0.266 | 0.728 | 2.040 | 1.598 | 4.027 | 10.933 |
| (a) numba 순차, 열 누적(axpy) 순서, fastmath 없음 | 0.258 | 0.631 | 1.822 | 1.582 | 3.488 | 9.243 |
| (a) numba prange axpy, 4스레드 | 0.111 | 0.260 | 0.929 | 0.497 | 1.202 | 4.945 |
| (a) numba prange axpy, 8스레드 | 0.111 | 0.206 | 0.707 | 0.334 | 0.920 | 3.784 |
| (b) 격자(cell list) 만들기, numba | 0.003 | 0.003 | 0.003 | 0.014 | 0.014 | 0.014 |
| (b) 충돌 힘, numba 순차 (격자 3×3칸) | 0.019 | 0.020 | 0.020 | 0.228 | 0.208 | 0.210 |
| (b) 충돌 힘, numpy N×N broadcast | 2.538 | 2.649 | 2.694 | 74.028 | 74.132 | 74.279 |
| (c) 시야 N×16 ray, numba 순차 (격자) | 1.133 | 1.119 | 1.171 | 7.077 | 6.853 | 6.910 |
| (c) 시야, numba prange 4스레드 | 0.402 | 0.394 | 0.407 | 2.107 | 2.037 | 2.081 |
| (c) 시야, numba prange 8스레드 | 0.285 | 0.289 | 0.309 | 1.502 | 1.456 | 1.502 |
| (c) 시야, numpy N×R×N broadcast | 49.185 | 50.119 | 50.156 | 재지 않음 | 재지 않음 | 재지 않음 |

- N=5000에서 numpy 시야 broadcast는 원소가 5000×16×5000 = 4억 개(float32 임시 배열 하나에 1.6 GB)라 돌리지 않았다 — `bench_alife.py`
- 개체별 가중치 `W_hh`(N×H×H float32)의 크기: N=1000,H=128은 66 MB, N=5000,H=64는 82 MB, N=5000,H=128은 328 MB — `bench_alife.py`
- `prange`를 1스레드로 돌리면 순차판보다 조금 느리다(충돌 0.020 → 0.062 ms, 시야 1.13 → 1.39 ms). 병렬 구간 진입 비용이 수십 µs 있다는 뜻이다 — `bench_alife.py`

### Inferences
- 세 커널의 성격이 다르다. RNN 행렬곱은 규칙적인 밀집 연산이라 numpy `matmul`(이 맥에서는 Accelerate)이 numba 순차보다 1.5~3배 빠르고, 8스레드 numba와는 H=32에서 비슷하며 H=64 이상에서는 8스레드 numba가 1.2~1.7배 빠르다. 이웃 탐색과 시야는 "가까운 것만 본다"는 가지치기가 핵심인데 numpy는 이를 표현하지 못해 N² 또는 N²·R을 통째로 계산한다. 그래서 충돌은 134배(N=1000)·325배(N=5000), 시야는 43배(N=1000) 차이가 났다.
- 따라서 10/29 결정은 "numpy 배열을 상태 저장소로, numba 커널을 계산기로"로 고쳐 읽는 것이 맞다. 순수 numpy만 고집하면 N=5000에서 시야 계산이 메모리 상한(약 7 GB 여유)에 먼저 걸린다.
- 운영 기기는 Linux라서 numpy wheel의 BLAS가 Accelerate가 아니다(보통 OpenBLAS). 작은 행렬을 행렬마다 BLAS로 부르는 구조라 맥에서 본 `matmul` 우위가 그대로 옮겨진다는 보장이 없다. RNN도 numba axpy 루프로 써 두면 두 기기에서 같은 코드 경로가 되고 BLAS 스레드 문제도 사라진다. BLAS를 쓴다면 numba 스레드와 겹치지 않게 BLAS는 1스레드로 묶어야 한다.
- H가 크면 병목은 코어 수가 아니라 메모리 대역폭이다. N=5000,H=128은 매 스텝 328 MB를 읽는데 8스레드로도 3.8 ms였다(초당 약 87 GB를 읽는 셈). 운영 기기의 메모리 대역폭이 이보다 낮으면 스레드를 늘려도 이 구간은 빨라지지 않는다. 가중치를 float32로 두고 은닉 크기를 16~64로 시작하는 편이 안전하다.
- "열 누적(axpy) 순서" 루프는 fastmath 없이도 SIMD로 묶여 fastmath판과 같은 속도가 나오면서 엄격 순서 스칼라 루프와 비트 단위로 같았다(3절). 속도 때문에 fastmath를 켤 이유가 없다.
- JAX-CPU는 numba 병렬보다 빠르다는 증거가 없고(QuantEcon 사례는 4.9배 느림), 개체 수가 출생·사망으로 변하는 세계는 고정 shape JIT과 잘 맞지 않는다. Taichi는 Python 3.14 wheel이 없고 출시 간격이 길다. 둘 다 1단계에서 고를 이유가 없다.

### Gaps
- PyTorch-CPU, Cython, Mojo, Rust(PyO3)를 이 세 커널로 비교한 공개 측정은 찾지 못했다. pyhpc-benchmarks의 수치 표는 이미지라 읽지 못했다.
- 운영 기기(Zen 4급, 12 vCPU, AVX-512 가능)에서의 값은 없다. numba(LLVM)가 그 CPU에서 512비트 벡터를 실제로 쓰는지, OpenBLAS의 작은 행렬 성능이 어떤지는 VM에서 `bench_alife.py`를 돌려야 안다.
- Linux용 numpy wheel이 어떤 BLAS와 스레드 방식을 싣는지는 이번에 연 문서에 없었다(`np.show_config()`와 `threadpoolctl.threadpool_info()`로 VM에서 확인할 것).
- 이 맥의 Accelerate가 `matmul`에서 여러 스레드를 썼는지는 확인하지 못했다.

## 2. 약 12코어에서 1,000~5,000 개체일 때 초당 몇 스텝이 가능한가

### Takeaway
비교 가능한 시뮬레이터가 공개한 CPU 스텝 속도는 사실상 찾지 못했다. 직접 잰 커널 합계로는 맥 M3 8스레드에서 N=1000이면 초당 약 900~2,100스텝, N=5000이면 약 185~510스텝이 상한이고, 실제 루프는 Python 조율·출생/사망·기록 때문에 이보다 낮다.

### Cited Findings
- 직접 측정, 세 커널 합계(RNN axpy + 격자 만들기 + 충돌 + 시야)로 환산한 초당 스텝 — `bench_alife.py`

| 설정 | 1스레드 | 4스레드 | 8스레드 |
|---|---|---|---|
| N=1000, H=32 | 567 | 1,758 | 2,142 |
| N=1000, H=64 | 485 | 1,406 | 1,855 |
| N=1000, H=128 | 309 | 720 | 921 |
| N=5000, H=32 | 114 | 363 | 511 |
| N=5000, H=64 | 95 | 296 | 401 |
| N=5000, H=128 | 62 | 140 | 185 |

- JaxLife 논문은 "a single NVIDIA A100 GPU"에서 개체 128 + 로봇 32로 2^16(65,536) 스텝을 돌렸고, 규모 실험은 개체 32·64·128·256이다. 초당 스텝 수와 CPU 성능은 본문에 없다. 개체 신경망은 개체·로봇·지형 인코더 → self-attention → cross-attention → LSTM 구조이고 LSTM 크기는 적혀 있지 않다 — [JaxLife (arXiv 2409.00853)](https://arxiv.org/html/2409.00853)
- Biosim4는 C++이고 "The innermost loop is thread-safe so that it can be parallelized by OpenMP"라고만 적으며 README에 속도 수치는 없다 — [biosim4 README](https://github.com/davidrmiller/biosim4)
- Biosim4 기본 설정은 population = 3000, stepsPerGeneration = 300이다(포크 저장소의 ini, 요약으로만 확인) — [biosim4.ini (fork)](https://github.com/surfingdoggo/biosim4/blob/main/biosim4.ini)
- Vivarium(flowersteam)은 Jax-MD 기반 2D 강체 물리에 Braitenberg식 개체(모터 2, 센서 2)이고 README에 개체 수·FPS 수치가 없다 — [flowersteam/vivarium](https://github.com/flowersteam/vivarium)
- 참고용 JAX-CPU 사례: NetForge RL은 6코어/12스레드 CPU에서 JAX 코어가 배치 4096으로 2.5×10^5 환경스텝/s, 단일 인스턴스 Python 엔진은 약 144 환경스텝/s라고 보고한다(요약으로만 확인, 인공생명이 아니고 작업량이 다름) — [NetForge RL (arXiv 2604.09523)](https://arxiv.org/pdf/2604.09523)

### Inferences
- 초당 400스텝이면 하루 약 3,460만 스텝, 초당 100스텝이면 약 864만 스텝이다. "끝나지 않는 세계"에서는 스텝당 비용보다 하루 누적 스텝과 저장량(4절)이 설계를 좌우한다.
- 표는 커널만 잰 상한이다. 시야 조건(31개체가 보이는 밀도, ray 16개)이 바뀌면 시야 비용이 거의 비례해 변한다. 시야가 커널 합계의 약 60~80%를 차지하는 구간(N=5000,H≤64, 8스레드)에서는 ray 수와 시야 거리가 가장 큰 조절 손잡이다.
- 4→8스레드에서 이득이 작은 것은 이 맥의 뒤 4개가 효율 코어이기 때문이기도 하다. 운영 기기는 낮은 우선순위로 돌려야 하므로 numba 스레드를 4~8로 잡고 측정해 정하는 것이 현실적이다.
- N=1000 정도에서 병렬 구간 진입 비용(수십 µs)이 충돌 커널 자체보다 크다. 작은 커널은 순차로 두고 시야·RNN만 `prange`로 돌리는 편이 낫다.

### Gaps
- Polyworld, evojax-CPU, JaxLife-CPU, Bibites의 공개 스텝 속도는 찾지 못했다.
- 운영 VM에서의 실측이 없다.

## 3. 결정적 재현성: RNG 설계와 부동소수점

### Takeaway
같은 기기·같은 라이브러리 빌드에서 체크포인트를 다시 돌려 비트까지 같게 만드는 것은 가능하다. 조건은 (1) 난수를 스레드와 무관하게 뽑고, (2) 각 `prange` 반복이 자기 칸에만 쓰고, (3) fastmath·병렬 축약·BLAS 스레드 변동을 피하고, (4) 라이브러리 버전을 고정하는 것이다. 맥과 x86 Linux 사이의 비트 일치는 초월함수(tanh, sin, cos)와 BLAS·SIMD 경로 차이 때문에 보장할 수 없다.

### Cited Findings

NumPy 난수
- NumPy의 약속은 엄격한 조건부다: 같은 BitGenerator·같은 seed·같은 호출 순서와 인자, "on the same build of numpy, in the same environment, on the same machine"일 때 같은 수열이 나온다 — [NumPy random compatibility policy](https://numpy.org/doc/stable/reference/random/compatibility.html)
- 호출 크기도 계약의 일부다: "Calling rng.random() 5 times is not guaranteed to give the same numbers as rng.random(5)" — [NumPy random compatibility policy](https://numpy.org/doc/stable/reference/random/compatibility.html)
- `Generator` 메서드의 수열은 기능 판(X.Y)에서 바뀔 수 있고 `default_rng`의 기본 BitGenerator도 바뀔 수 있다. BitGenerator 클래스는 판 간 호환 보장이 더 강하다 — [NumPy random compatibility policy](https://numpy.org/doc/stable/reference/random/compatibility.html)
- "different CPUs implement floating point arithmetic differently, and this can cause differences in certain edge cases that cascade to the rest of the stream" — [NumPy random compatibility policy](https://numpy.org/doc/stable/reference/random/compatibility.html)
- 독립 스트림을 만드는 공식 방법: `SeedSequence(12345).spawn(10)` 또는 `default_rng(12345).spawn(10)`. 자식이 다시 spawn할 수 있어 조율 없이 나무 모양으로 쪼갤 수 있다. n개 스트림의 충돌 확률은 비관적으로 약 n²·2^-128 — [NumPy parallel random](https://numpy.org/doc/stable/reference/random/parallel.html)
- 정수 묶음 seed도 된다: `default_rng([worker_id, root_seed])`. 변하는 ID를 앞에 두라고 권한다. `root_seed + worker_id`는 "UNSAFE! Do not do this!" — [NumPy parallel random](https://numpy.org/doc/stable/reference/random/parallel.html)
- 사람이 고른 seed 대신 `secrets.randbits`로 128비트 seed를 만들라고 권한다 — [NumPy parallel random](https://numpy.org/doc/stable/reference/random/parallel.html)
- Philox는 counter 기반이다: `Philox(seed=None, counter=None, key=None)`, 256비트 counter + 128비트 key, "Using different keys produces independent sequences", `advance(delta)`로 임의 위치로 건너뛸 수 있고, "Philox makes a guarantee that a fixed seed will always produce the same random integer stream" — [NumPy Philox](https://numpy.org/doc/stable/reference/random/bit_generators/philox.html)
- 직접 확인: `Generator(Philox(key=12345)).bit_generator.state`는 `{'bit_generator': 'Philox', 'state': {'counter': [0,0,0,0], 'key': [12345,0]}, 'buffer': [...], 'buffer_pos': 4, 'has_uint32': 0, 'uinteger': 0}` 꼴의 dict이고, 이 dict를 다른 Generator에 넣으면 이후 수열이 같았다 — `bench_alife.py`와 같은 환경에서 실행(numpy 2.2.6)

numba 안의 난수와 병렬
- numba 안의 `np.random.*`·`random.*`는 numpy와 별개의 내부 Mersenne Twister 상태를 쓰고 시작할 때 OS 엔트로피로 초기화된다. Python 쪽에서 seed를 줘도 numba 쪽은 seed되지 않는다. "Each thread and each process will produce independent streams of random numbers" — [Numba supported Python features](https://numba.readthedocs.io/en/stable/reference/pysupported.html)
- numba 0.56부터 `Generator` 객체를 jit 함수에 넘길 수 있지만 "The current Numba support for Generator is not thread-safe"이고 병렬 코드에서는 쓰지 말라고 한다. 결과가 numpy와 Linux-x86_64·macOS에서 수십 ULP, linux-aarch64에서 수천 ULP까지 다를 수 있다 — [Numba supported NumPy features](https://numba.readthedocs.io/en/stable/reference/numpysupported.html)
- `prange`에서 "There's no guarantee about which indexes are assigned to which executing threads" — [Numba parallel](https://numba.readthedocs.io/en/stable/user/parallel.html)
- `prange` 축약은 `+ - * /`·`max`·`min`에 자동 추론되고, 배열 조각이나 원소로 축약하면 컴파일러가 못 잡는 경쟁 조건이 생길 수 있다(`y[i % 4] += x[i]` 등). 문서는 부동소수점 축약 순서가 결과에 주는 영향을 말하지 않는다 — [Numba parallel](https://numba.readthedocs.io/en/stable/user/parallel.html)
- `fastmath`는 결과를 바꾼다. 문서 예에서 `add_assoc(0, np.inf)`가 `fastmath=False`면 `nan`, `True`면 `0.0` — [Numba performance tips](https://numba.readthedocs.io/en/stable/user/performance-tips.html)

부동소수점의 기기 간 차이
- "the IEEE standard does not guarantee that the same program will deliver identical results on all conforming systems"(Peter Markstein). 같은 기기에서 기록된 입력을 다시 재생해 같은 결과를 얻는 것은 가능하다. 초월함수는 구현마다 값이 조금 다르다. 재현성을 원하면 자동 FMA 생성을 피하라 — [Gaffer On Games: Floating Point Determinism](https://gafferongames.com/post/floating_point_determinism/)
- Rapier 물리 엔진은 기본이 "locally deterministic"(같은 기기·같은 버전·같은 컴파일러)이고, 기기 간 결정성에는 별도 기능(`enhanced-determinism`), IEEE 754-2008 준수, 8폭 SIMD 금지, sin·cos 같은 함수의 결정적 구현 사용, 그리고 물체를 넣고 빼는 순서의 동일성이 필요하다. 병렬 솔버는 스레드 수와 무관하게 순차 결과와 같도록 만들어져 함께 쓸 수 있다 — [Rapier determinism](https://rapier.rs/docs/user_guides/rust/determinism/)
- NumPy ufunc은 실행 시 CPU 기능에 따라 다른 SIMD 루프를 고른다. AVX512 경로의 sin·cos 등은 최대 4 ULP 오차이고, 구현 변경 때 고급 Intel CPU에서만 결과가 달라진 사례가 있다. 원인 추적에는 `NPY_DISABLE_CPU_FEATURES`를 쓴다(모두 요약으로만 확인) — [numpy-discussion: Precision changes to sin/cos](https://mail.python.org/archives/list/numpy-discussion@python.org/thread/C6EYZZSR4EWGVKHAZXLE7IBILRMNVK7L); [numpy issue #20356](https://www.github.com/numpy/numpy/issues/20356)
- PyTorch 사용자는 AVX2만 쓰도록 지정해도 AVX512 기기와 결과가 달랐고, AVX512를 끈 VM 안에서만 일치했다고 보고했다(요약으로만 확인) — [PyTorch forum](https://discuss.pytorch.org/t/reproducibility-and-floating-point-arithmetics-with-out-avx512/220674)

직접 측정 (맥 M3, numba 0.67.0) — `bench_alife.py`
- 각 반복이 `out[i]`·`dist[i, :]`처럼 자기 개체 칸에만 쓰는 `prange` 시야 커널은 1·4·8스레드 결과가 순차판과 비트 단위로 같았다(6개 설정 모두).
- RNN을 fastmath로 컴파일하면 엄격 순서판과 비트가 달랐다(최대 절대 차 3.7e-7 ~ 1.2e-6). 열 누적(axpy) 순서판은 fastmath 없이 엄격 순서판과 비트 단위로 같았다.
- numpy 시야 구현과 numba 시야 구현은 같은 수식인데 ray 거리 최대 차이가 1.4~3.9 단위였다. 각도 계산의 float32/float64 차이(약 1e-7)가 원 가장자리를 스치는 ray의 맞음/빗나감을 뒤집은 것으로 추정한다(개수는 세지 않음).
- 난수 비용(N=1000, 개체당 4개): 개체마다 `Generator(Philox(key=[i, 777]))`를 새로 만들면 개체당 6.3 µs(스텝당 6.3 ms, 커널 합계보다 큼). Generator 하나로 `random((N,4), dtype=float32)`를 뽑으면 Philox 9.2 µs, PCG64 7.7 µs. numba 안에서 (seed, 개체 id, 스텝, k)를 섞는 무상태 해시(splitmix64 마무리 함수)로 뽑으면 1스레드 29 µs, 8스레드 45 µs이고 스레드 수와 무관하게 결과가 같았다.

### Inferences
- 같은 기기에서 비트 일치 재생에 필요한 것(구현 규칙):
  1. 난수는 두 길 중 하나로만 쓴다. (가) 주 스레드에서 Generator 하나로 정해진 순서·정해진 크기로 뽑아 배열로 커널에 넘긴다. (나) 커널 안에서 `f(root_seed, 개체 고유 id, 스텝, 용도 번호)` 꼴의 무상태 counter 기반 해시를 쓴다. numba 내장 `np.random`과 `Generator`를 `prange` 안에서 쓰면 안 된다(스레드별 상태라 스케줄에 따라 달라진다).
  2. (가)를 쓸 때는 호출 크기가 개체 수에 따라 달라지므로 "개체 수가 같으면 같은 수열"까지만 보장된다. 변종 비교(5절)에는 (나)가 더 잘 맞는다. 개체 id와 스텝이 같으면 개체 수가 달라져도 같은 난수를 받기 때문이다.
  3. `prange` 반복은 자기 개체 칸에만 쓰고, 여러 개체에 걸친 합(총 에너지 등)은 순차 루프나 numpy로 따로 계산한다. 병렬 축약은 스레드 수에 따라 덧셈 순서가 달라질 수 있다(문서가 순서를 보장하지 않음).
  4. fastmath를 끄고, 속도는 루프 순서(axpy)로 얻는다.
  5. 개체를 배열에 넣고 빼는 순서(출생·사망 처리, 빈 칸 재사용, 격자 정렬)를 결정적으로 만든다. dict·set 순회 순서나 스레드 완료 순서에 기대지 않는다.
  6. BLAS를 거치는 연산을 쓰면 BLAS 스레드 수를 고정한다(1 권장). 스레드 수에 따라 분할이 달라질 수 있다.
  7. numpy·numba·llvmlite·Python 버전을 lock 파일로 고정하고 체크포인트에 기록한다. numpy 기능 판이 바뀌면 Generator 수열이 바뀔 수 있다.
- 기기 간(맥 arm64 ↔ Linux x86-64)에는 보장할 수 없는 것: `tanh`·`sin`·`cos`·`exp`의 마지막 비트(libm·SVML·numpy SIMD 경로가 다름), BLAS 결과(Accelerate 대 OpenBLAS), 컴파일러가 고른 벡터 폭에 따른 fastmath 축약 순서. 한 ray의 맞음/빗나감이 뒤집히면 이후 세계가 통째로 갈라지므로 "거의 같다"는 유지되지 않는다.
- 그래서 계약을 이렇게 정하는 것이 현실적이다: "운영 VM에서 만든 체크포인트는 운영 VM에서 비트 단위로 재생된다. 맥에서는 통계적으로 같은 세계가 나온다." 기기 간 비트 일치가 꼭 필요해지면 초월함수를 직접 쓴 다항식·유리식 근사(덧셈·곱셈·나눗셈·sqrt만 사용)로 바꾸는 길이 있다(Rapier가 쓰는 방법과 같은 생각).
- VM의 CPU type이 host이므로 호스트 CPU나 CPU 기능 노출이 바뀌면 numba가 다른 기계어를 만들 수 있다. 체크포인트에 CPU 모델·기능 문자열을 같이 적어 두면 불일치 원인을 가릴 수 있다. `NUMBA_CPU_NAME`·`NUMBA_CPU_FEATURES`를 고정하는 것도 방법이다.
- 재현성 회귀 시험을 처음부터 넣는다: 체크포인트에서 1,000스텝을 두 번 돌려 상태 배열의 해시가 같은지, 스레드 1개와 8개에서 같은지 비교한다.

### Gaps
- numba `prange` 축약의 덧셈 순서가 스레드 수·threading layer에 따라 실제로 달라지는지 명시한 공식 문장은 찾지 못했다(문서가 침묵). 안전하게 피하는 쪽으로 적었다.
- LLVM이 fastmath 없이 `a*b+c`를 FMA로 합치지 않는다는 점은 이 맥에서 axpy판과 스칼라판이 비트 일치한 것으로만 간접 확인했다. x86-64에서는 확인하지 않았다.
- OpenBLAS가 스레드 수에 따라 결과 비트가 달라지는지에 대한 1차 자료는 열지 못했다.

## 4. 체크포인트 설계와 저장량: 형식, 원자적 쓰기, 계보 저장, 보존 정책, ZFS

### Takeaway
체크포인트는 "압축하지 않은 `npz`(또는 safetensors) + JSON 메타데이터"를 임시 이름에 쓰고 fsync한 뒤 rename하는 방식이면 충분하다. 가중치는 압축이 거의 안 되므로(직접 측정 7% 감소에 3초) 압축은 ZFS(lz4)에 맡기고, 저장량은 전체 스냅샷 간격과 관찰용 프레임 기록 간격이 결정한다.

### Cited Findings

원자적·내구적 쓰기
- 파일을 안전하게 바꾸는 순서: "create a new temp file (on the same file system!)" → "write data to the temp file" → "fsync() the temp file" → "rename the temp file to the appropriate name" → "fsync() the containing directory". rename은 읽는 쪽이 옛 것 아니면 새 것 중 하나만 보게 하고, 페이지 캐시의 더티 페이지는 언제 디스크에 갈지 정해져 있지 않아 fsync가 필요하며, 새 파일은 디렉터리도 fsync해야 할 수 있다 — [LWN: Ensuring data reaches disk](https://lwn.net/Articles/457667/)

형식
- safetensors는 pickle과 달리 안전하고 zero-copy로 빠른 텐서 저장 형식이라고 소개한다. 문서 페이지의 형식 그림과 비교표는 이번에 내용이 비어 보여 확인하지 못했다(헤더 구조, numpy API, 메타데이터 제한은 미확인) — [safetensors docs](https://huggingface.co/docs/safetensors/index)
- zarr-python은 Zarr 형식 2와 3을 지원하는 "chunked, compressed, N-dimensional arrays" 구현이고 Python 3.12 이상이 필요하다 — [zarr docs](https://zarr.readthedocs.io/en/stable/)
- 직접 측정(맥 M3 SSD, fsync 없음, N=5000·H=64 상태 124.2 MB: `W_hh` + `W_ih`(H×32) + `h` + 위치): `np.savez` 0.056 s에 124.2 MB, `np.savez_compressed` 3.064 s에 115.3 MB(7% 감소), `np.load` 후 전부 읽기 0.018 s. 가중치는 정규분포 난수라 진화한 가중치의 압축률과는 다를 수 있다 — 이 조사에서 실행(numpy 2.2.6)
- RNG 상태는 `bit_generator.state` dict로 꺼내고 넣을 수 있다(3절 직접 확인) — [NumPy Philox](https://numpy.org/doc/stable/reference/random/bit_generators/philox.html)

계보(phylogeny)
- ALife 데이터 표준의 계보 표는 필수 열이 `id`(음이 아닌 정수)와 `ancestor_list`(같은 파일 안의 id 목록)이고, 관례 열이 `origin_time`, `destruction_time`이며 열을 더 붙일 수 있다 — [ALife Data Standards: phylogeny](https://alife-data-standards.github.io/alife-data-standards/phylogeny.html)
- phylotrackpy(C++ Empirical Systematics Manager의 Python 래퍼)는 출생마다 계보를 기록하고, 기본 설정(`store_outside=False`)에서 살아 있는 분류군과 그 조상만 남기는 가지치기가 자동으로 된다. 스냅샷 파일은 ALife 표준 형식(id, 조상 목록, 생긴 시각, 사라진 시각)이다. `remove_before`로 특정 시점 전에 죽은 분류군을 버릴 수 있으나 위상 지표가 제한된다. 더 가벼운 방법으로 hstrat을 안내한다 — [phylotrackpy introduction](https://phylotrackpy.readthedocs.io/en/latest/introduction.html)

ZFS
- 순차 작업용 데이터셋은 "Set recordsize=1M", 압축은 모르겠으면 LZ4, "incompressible data will be stored without compression", SQLite는 별도 데이터셋에 recordsize 64K, atime은 `relatime=on` 또는 `atime=off` 권장. recordsize 변경은 새 파일에만 적용된다. 이 페이지는 스냅샷은 다루지 않는다 — [OpenZFS Workload Tuning](https://openzfs.github.io/openzfs-docs/Performance%20and%20Tuning/Workload%20Tuning.html)
- fsync가 많은 작업에서 기계식 저장소라면 SLOG 장치 추가를 고려하라고 한다 — [OpenZFS Workload Tuning](https://openzfs.github.io/openzfs-docs/Performance%20and%20Tuning/Workload%20Tuning.html)

다른 시뮬레이터의 저장 방식
- ALIEN은 `.sim` 파일로 저장하고 "Autosave with save points and flashbacks to return to interesting moments" 기능이 있다 — [chrxh/alien](https://github.com/chrxh/alien)
- Biosim4는 세대마다 `logs/epoch.txt`에 한 줄을 덧붙이고 선택한 세대만 영상으로 만든다 — [biosim4 README](https://github.com/davidrmiller/biosim4)

### Inferences
- 권장 구조(1단계): 체크포인트 하나 = 디렉터리 하나.
  - `state.npz`: 압축 없는 `np.savez`. 개체 배열 전부(위치·속도·방향·에너지·나이·id·부모 id·가중치·은닉 상태), 환경 배열, 빈 칸 목록. 읽을 때 `allow_pickle=False`.
  - `meta.json`: `schema_version`, 스텝 번호, 벽시계 시각, 설정 전체와 그 해시, 부모 체크포인트(실행 id·스텝·`state.npz`의 sha256), RNG 상태(`bit_generator.state`를 리스트로 바꾼 것 또는 root seed와 counter), numpy·numba·llvmlite·Python 버전, CPU 모델과 기능, numba 스레드 수.
  - 쓰기 순서: `ckpt_000123.tmp/`에 두 파일을 쓰고 각각 fsync → 디렉터리를 `ckpt_000123/`으로 rename → 부모 디렉터리 fsync. 읽을 때는 `meta.json`이 있고 해시가 맞는 가장 최근 것을 고른다. 전원이 끊겨 `.tmp`만 남은 것은 시작할 때 지운다.
- pickle은 쓰지 않는다. 코드가 바뀌면 못 읽고, 임의 코드 실행 위험이 있으며, 스키마 버전 관리가 안 된다. HDF5·zarr는 "하나의 큰 배열을 조각으로 덧붙이는" 용도에 강하지만 전체 상태를 통째로 바꿔 쓰는 체크포인트에는 이점이 작고 의존성이 늘어난다(zarr는 Python 3.12 이상).
- 저장량 계산(개체당 상태 = `W_hh` H²·4 + `W_ih` H·I·4 + 기타. H=64, 입력 32면 약 25 KB):
  - N=2000이면 전체 스냅샷 약 50 MB, N=5000이면 약 124 MB(직접 측정값과 일치).
  - N=2000을 1시간마다 저장하면 하루 1.2 GB, 한 달 36 GB. "최근 24시간은 시간별, 그 뒤 30일은 일별, 그 뒤는 주별 또는 월별"로 솎으면 1년에 약 24 + 30 + 약 50개 = 100개 남짓, 약 5 GB 수준이다.
  - 전원 차단 대비용 "최신 체크포인트"는 5~10분마다 덮어쓰는 두 개(번갈아 쓰기)를 따로 둔다. 124 MB를 쓰는 데 0.06초(fsync 제외)이므로 쓰는 비용보다 fsync 대기와 디스크 수명이 고려 대상이다.
- 관찰용 프레임(5절·7절의 시간 되감기용)은 체크포인트와 다른 기록이다. 개체당 16바이트(x·y를 uint16으로 양자화, 방향 uint8, 반지름·에너지·종 표시, id uint32)로 잡으면 N=2000에서 프레임당 32 KB. 초당 1프레임이면 하루 2.8 GB, 10초당 1프레임이면 하루 276 MB다. 매 스텝 기록은 불가능하므로 "k스텝마다 1프레임 + 오래된 구간은 간격을 넓혀 솎기"가 필요하다.
- 계보는 작다. 출생 한 건을 (id uint64, 부모 id uint64, 출생 스텝 uint64, 사망 스텝 uint64) 32바이트로 적으면 1,000만 건이 320 MB이고, 열 방식 압축(parquet 등)을 쓰면 id와 스텝이 단조 증가라 더 줄어든다. 1단계는 "덧붙이기 전용 이진 로그(출생 레코드, 사망 레코드) + 주기적으로 parquet 또는 SQLite로 묶기"면 충분하고, 표 형식은 ALife 표준 열 이름(`id`, `ancestor_list`, `origin_time`, `destruction_time`)을 따르면 phylotrackpy 등 도구와 맞는다. 전체 계보를 메모리에 들고 있지 말고 파일에만 쌓는다.
- 통계(개체 수, 평균 에너지, 출생·사망 수)는 N스텝마다 한 줄씩 고정 폭 이진 또는 CSV로 덧붙인다. 하루 수만 줄 수준이라 부담이 없다.
- ZFS는 보완재다. 체크포인트 데이터셋은 recordsize=1M + lz4, 스냅샷은 "앱이 rename까지 마친 상태"를 통째로 보존해 실수로 지운 것을 되살리는 용도로 쓴다. 앱 수준 원자적 쓰기를 대신하지는 못한다(스냅샷 시점에 쓰는 중이던 `.tmp`는 그대로 반쪽이다). 큰 파일을 자주 덮어쓰면 스냅샷이 옛 블록을 붙잡아 공간이 불어나므로, 자주 덮어쓰는 "최신 체크포인트"는 스냅샷 주기가 긴 데이터셋에 두거나 스냅샷 보존 기간을 짧게 한다.

### Gaps
- HDF5(h5py)의 덧붙이기·손상 내성, parquet·DuckDB·SQLite의 측정 비교는 조사하지 못했다.
- safetensors의 형식 세부(헤더, 메타데이터, numpy API)는 문서 본문이 비어 보여 확인하지 못했다.
- VM의 가상 디스크가 TrueNAS ZFS 위에 어떻게 놓였는지(zvol인지, NFS인지, 캐시 모드)에 따라 VM 안의 fsync가 실제 디스크까지 가는지가 달라진다. 이 부분은 자료가 아니라 기지 설정을 확인해야 한다.
- 진화한 가중치의 실제 압축률은 모른다(난수 가중치로만 측정).

## 5. 분기 실험: 같은 순간에서 갈라 "만약"을 비교하기

### Takeaway
분기는 "부모 체크포인트의 해시 + 설정 차이 + RNG 처리 방침"을 적은 새 실행 디렉터리로 표현하면 된다. 이 주제만 다룬 권위 있는 자료는 찾지 못했고, 아래는 NumPy의 스트림 분할 규칙과 3절의 재현성 조건에서 끌어낸 설계다.

### Cited Findings
- `SeedSequence`와 `Generator`는 `spawn()`으로 독립 스트림을 나무 모양으로 만들 수 있고 "make local decisions about when and how to split up streams without coordination between processes"가 가능하다. 한 단계만 spawn하면 모든 상태가 유일함이 보장된다 — [NumPy parallel random](https://numpy.org/doc/stable/reference/random/parallel.html)
- 정수 묶음 seed `[worker_id, root_seed]`로도 독립 스트림을 만들 수 있고, ID는 유일하고 결정적으로 만들어져야 한다 — [NumPy parallel random](https://numpy.org/doc/stable/reference/random/parallel.html)
- Philox는 key가 다르면 독립 수열이고 key를 직접 지정할 수 있다. 단 "avoid reusing stream IDs" — [NumPy parallel random](https://numpy.org/doc/stable/reference/random/parallel.html)
- 여러 numba 프로세스를 기본 스레드 수로 함께 돌리면 과다 구독이 되므로 프로세스별로 스레드를 제한하라고 한다 — [Numba threading layers](https://numba.readthedocs.io/en/stable/user/threading-layer.html)
- ALIEN은 저장 지점(save points)과 되돌아가기(flashbacks)를 제공한다 — [chrxh/alien](https://github.com/chrxh/alien)

### Inferences
- 실행 디렉터리 구조: `runs/<run_id>/{config.json, lineage.json, ckpt/, frames/, stats/, births.log}`. `lineage.json`에 `parent_run`, `parent_step`, `parent_state_sha256`, `config_diff`(바뀐 키만), `rng_policy`를 적는다. 분기 나무는 이 파일들만 읽어 복원할 수 있다.
- RNG 방침은 두 가지를 구분해 이름 붙인다.
  - `same_stream`: 체크포인트의 RNG 상태를 그대로 이어 쓴다. 설정이 같으면 부모와 비트 단위로 같은 미래가 나온다(재현 시험 겸용). 설정만 바꾼 변종은 "같은 난수, 다른 규칙" 비교가 된다.
  - `new_stream`: root seed에 분기 번호를 섞는다(`SeedSequence([branch_id, root_seed])` 또는 Philox key 변경). "같은 규칙, 다른 운" 비교가 된다. 분기 번호는 재사용하지 않는다.
- 3절의 무상태 해시 난수(`f(seed, 개체 id, 스텝, 용도)`)를 쓰면 변종에서 개체 수가 달라져도 살아 있는 같은 개체는 같은 스텝에 같은 난수를 받는다. 변종 간 차이가 "운"이 아니라 "바꾼 규칙"에서 나온 것임을 가리기 쉽다. Generator 하나를 순서대로 소비하는 방식은 개체 하나만 늘어도 이후 모든 난수가 어긋난다.
- 분기 뒤에 태어나는 개체의 id는 분기마다 겹칠 수 있으므로 전역 비교 키는 `(run_id, 개체 id)`로 한다.
- 자원: 분기는 systemd 템플릿 유닛(`alife@<run_id>.service`)으로 띄우고 하나의 slice에 묶어 slice에 CPU·메모리 상한을 건다(6절). 본 세계 1개 + 분기 1~2개가 동시에 돌 때 각 프로세스의 `NUMBA_NUM_THREADS`를 합이 vCPU 수를 넘지 않게 나눈다. 메모리는 프로세스당 상태 크기의 3~4배(상태 + 커널 임시 버퍼 + 체크포인트 쓰는 순간의 사본)를 잡는다.
- 비교는 "분기 시점부터 M스텝 뒤"의 통계 줄과 프레임을 나란히 놓는 것으로 충분하다. 분기를 오래 돌릴 필요가 없으면 스텝 상한을 설정에 넣어 스스로 끝나게 한다.

### Gaps
- 인공생명 장기 실행에서 분기·비교 실험을 다룬 공개 설계 문서나 사례는 찾지 못했다(ALIEN의 기능 소개 한 줄뿐).
- 변종 비교에 필요한 반복 수와 통계적 판단 기준은 이 조사 범위 밖이다.

## 6. 운영: systemd 서비스, 자원 제한, 종료·전원 차단 대비, 장기 실행 함정

### Takeaway
systemd 유닛 하나로 낮은 우선순위(`Nice`, `CPUWeight`), 메모리 상한(`MemoryHigh` + `MemoryMax`), 실패 시 재시작, 멈춤 감지(`WatchdogSec`)를 모두 걸 수 있다. SIGTERM을 받으면 체크포인트를 쓰고 0으로 끝내고, 전원 차단은 주기적 원자적 체크포인트로 대비한다.

### Cited Findings

CPU
- `CPUWeight=`는 1~10000(커널 기본 100)이고 `idle` 값은 같은 cgroup과 형제에 다른 실행할 프로세스가 없을 때만 CPU를 받게 한다(cgroup v2 전용, systemd 232부터) — [systemd.resource-control](https://www.freedesktop.org/software/systemd/man/latest/systemd.resource-control.html)
- `CPUQuota=`는 CPU 하나 기준 백분율이고 100%를 넘으면 여러 CPU 분량이다. 주기는 `CPUQuotaPeriodSec=`(기본 100 ms). `AllowedCPUs=`로 쓸 CPU 번호를 제한할 수 있다(244부터) — [systemd.resource-control](https://www.freedesktop.org/software/systemd/man/latest/systemd.resource-control.html)
- 커널 쪽 대응: `cpu.weight`(1~10000, 기본 100), `cpu.max`("$MAX $PERIOD", 기본 "max 100000"), `cpu.idle`=1이면 cgroup 전체가 SCHED_IDLE처럼 동작 — [Linux cgroup v2](https://docs.kernel.org/admin-guide/cgroup-v2.html)
- `Nice=`는 -20(최고)~19(최저). `CPUSchedulingPolicy=`는 `other`, `batch`, `idle`, `fifo`, `rr`, `ext` 중 하나. `IOSchedulingClass=`는 `realtime`, `best-effort`, `idle`이고 커널 기본은 `best-effort` 우선순위 4 — [systemd.exec](https://www.freedesktop.org/software/systemd/man/latest/systemd.exec.html)

메모리
- `MemoryHigh=`를 넘으면 "the processes are heavily slowed down and memory is taken away aggressively"이고 문서는 이것을 주된 메모리 제어 수단으로 부른다. `MemoryMax=`를 넘고 줄일 수 없으면 "out-of-memory killer is invoked inside the unit"이며 마지막 방어선으로 쓰라고 한다. `MemorySwapMax=`로 swap 사용도 제한할 수 있다 — [systemd.resource-control](https://www.freedesktop.org/software/systemd/man/latest/systemd.resource-control.html)
- cgroup v2 메모리 계산에는 사용자 메모리(페이지 캐시와 익명 메모리 포함), 커널 자료 구조, TCP 버퍼가 들어간다. "Going over the high limit never invokes the OOM killer." `memory.events`에 low·high·max·oom·oom_kill 횟수가 쌓이고 `memory.pressure`로 압박 정보(PSI)를 볼 수 있다 — [Linux cgroup v2](https://docs.kernel.org/admin-guide/cgroup-v2.html)
- `OOMPolicy=`는 `continue`·`stop`·`kill`이고 `stop`·`kill`이면 서비스가 `oom-kill` 실패 상태가 된 뒤 `Restart=`가 적용될 수 있다. `OOMScoreAdjust=`는 -1000~1000 — [systemd.service](https://www.freedesktop.org/software/systemd/man/latest/systemd.service.html); [systemd.exec](https://www.freedesktop.org/software/systemd/man/latest/systemd.exec.html)

재시작·종료·감시
- `Restart=on-failure`는 비정상 종료 코드, 비정상 신호, 시간 초과, watchdog 만료, OOM 종료에서 재시작하고 정상 종료에서는 하지 않는다. systemd가 직접 멈춘 경우에는 재시작하지 않는다. `RestartSec=` 기본 100 ms. `RestartSteps=`와 `RestartMaxDelaySec=`를 함께 주면 재시작 간격이 지수적으로 늘어난다(예: 10s, 20s, 40s, 80s, 160s). 재시작은 `StartLimitIntervalSec=` 제한을 받는다 — [systemd.service](https://www.freedesktop.org/software/systemd/man/latest/systemd.service.html)
- `ExecStop=`이 없으면 멈출 때 SIGTERM이 바로 가고, `TimeoutStopSec=` 안에 끝나지 않으면 SIGKILL로 강제 종료된다 — [systemd.service](https://www.freedesktop.org/software/systemd/man/latest/systemd.service.html)
- `WatchdogSec=`(기본 0, 꺼짐)을 켜면 서비스가 `WATCHDOG=1`을 주기적으로 보내야 하고, 놓치면 실패 상태가 되어 SIGABRT로 종료된다. 제한 시간은 `WATCHDOG_USEC` 환경 변수로 전달된다. `Type=notify`면 `READY=1`을 보내야 시작 완료로 본다. `NotifyAccess=`를 따로 안 주면 `main`으로 설정된다 — [systemd.service](https://www.freedesktop.org/software/systemd/man/latest/systemd.service.html)
- `sd_notify` 문서에 라이브러리 없이 쓰는 Python 구현 예가 있다: `NOTIFY_SOCKET` 환경 변수의 경로로 `AF_UNIX` 데이터그램 소켓을 연결해 문자열을 보낸다. `STATUS=...`로 한 줄 상태 문자열을, `STOPPING=1`로 종료 시작을 알릴 수 있다 — [sd_notify](https://www.freedesktop.org/software/systemd/man/latest/sd_notify.html)

스레드 수와 자원 제한의 관계
- `NUMBA_NUM_THREADS`는 설정하지 않으면 감지된 CPU 코어 수이고 OpenMP·MKL 스레드 설정과는 별개다 — [Numba envvars](https://numba.readthedocs.io/en/stable/reference/envvars.html)

장기 실행 Python의 메모리
- glibc malloc은 스레드마다 arena를 더 만들고(64비트에서 코어 수의 8배까지), 해제한 메모리를 모두 OS에 돌려주지 않는다. `MALLOC_ARENA_MAX`, `MALLOC_TRIM_THRESHOLD_`, `malloc_trim(0)`, jemalloc(`LD_PRELOAD`)이 흔한 대응이지만 효과는 사례마다 다르다(한 사례는 `MALLOC_ARENA_MAX=2`와 `malloc_trim`이 효과가 없어 jemalloc으로 바꿈, 다른 사례는 jemalloc으로 2.68 GiB → 약 460 MiB. 모두 웹 서비스 사례이고 요약으로만 확인) — [CPython issue msg316359](https://bugs.python.org/msg316359); [discuss.python.org: ThreadPoolExecutor와 glibc](https://discuss.python.org/t/memory-leak-when-using-concurrent-futures-threadpoolexecutor-with-glibc-but-possibly-not-with-jemalloc/108311); [DEV: Your memory leak might be standard Python behavior](https://dev.to/sondar4/your-memory-leak-might-be-standard-python-behavior-484o)

### Inferences
- 유닛 초안(값은 VM 실측 뒤 조정):
  ```ini
  [Unit]
  Description=alife world %i
  After=network.target
  StartLimitIntervalSec=600
  StartLimitBurst=5

  [Service]
  Type=notify
  ExecStart=/opt/alife/.venv/bin/python -m alife.run --run %i --resume latest
  Environment=NUMBA_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
  Nice=19
  CPUSchedulingPolicy=batch
  IOSchedulingClass=idle
  CPUWeight=20
  CPUQuota=600%
  MemoryHigh=3G
  MemoryMax=4G
  MemorySwapMax=0
  Restart=on-failure
  RestartSec=10
  RestartSteps=4
  RestartMaxDelaySec=160
  WatchdogSec=120
  TimeoutStopSec=90
  KillSignal=SIGTERM

  [Install]
  WantedBy=multi-user.target
  ```
- `CPUQuota`를 걸 때는 numba 스레드 수를 quota 이하로 맞춘다(600%면 6스레드). 스레드가 quota보다 많으면 주기(100 ms)마다 할당량을 다 쓰고 전부 멈췄다 다시 도는 식이 되어 스텝 시간이 들쭉날쭉해진다. "다른 일이 없을 때는 다 쓰고, 있으면 양보"가 목적이면 `CPUQuota` 없이 `CPUWeight`(또는 `idle`)와 `Nice=19`만 쓰는 편이 낫다.
- 메모리 상한에는 페이지 캐시도 잡힌다. 큰 체크포인트나 프레임 파일을 쓰면 cgroup 사용량이 일시적으로 오르지만 캐시는 회수 대상이라 `MemoryHigh` 근처에서 회수·감속이 먼저 일어난다. `MemoryHigh`를 상태 크기의 3~4배 위에, `MemoryMax`를 그보다 조금 위에 둔다. `memory.events`의 `high`·`oom_kill` 횟수를 건강 점검에 넣는다.
- 종료 처리: SIGTERM 핸들러는 플래그만 세우고, 주 루프가 스텝 경계에서 체크포인트를 쓴 뒤 `sys.exit(0)`한다(정상 종료라 `on-failure` 재시작이 안 걸린다). 체크포인트 쓰기 + fsync가 `TimeoutStopSec` 안에 끝나야 한다. 커널 호출 중에는 신호 처리가 미뤄지므로 한 스텝이 길어야 수십 ms인 구조를 유지한다.
- 전원 차단: SIGTERM이 오지 않으므로 마지막 주기 체크포인트로 돌아간다. 잃는 양 = 체크포인트 간격이다. 시작할 때 `--resume latest`가 유효한 최신 체크포인트를 찾고 `.tmp` 찌꺼기를 지운다(4절). 덧붙이기 로그(프레임·계보·통계)는 체크포인트 스텝 이후 부분을 잘라 내고 이어 써야 재개 뒤 기록이 겹치지 않는다. 그러려면 체크포인트 메타에 각 로그 파일의 그 시점 바이트 길이를 적어 둔다.
- watchdog: 주 루프가 N스텝마다(또는 5~10초마다) `WATCHDOG=1`과 `STATUS=step=... agents=... sps=...`를 보낸다. 무한 루프·교착 시 systemd가 죽이고 다시 띄운다. 첫 실행 때 numba 컴파일이 수십 초 걸릴 수 있으므로 `READY=1`은 컴파일 뒤에 보내거나 `cache=True`를 쓴다.
- 건강 알림: `systemctl status`의 STATUS 줄, 스텝 속도가 기준 이하로 떨어짐, 개체 수 0(멸종), 디스크 사용률, `memory.events` 증가를 별도 타이머 유닛이 읽어 기존 알림 경로로 보낸다. 시뮬레이션 프로세스가 직접 알림을 보내지 않게 하면 루프가 외부 장애에 묶이지 않는다.
- 메모리 증가를 막는 코드 습관: 배열은 최대 개체 수로 한 번만 할당하고 살아 있음 마스크·빈 칸 목록으로 재사용한다. numba 커널 안에서 반복마다 `np.empty`를 부르지 않고 스레드별 작업 버퍼를 밖에서 넘긴다(이번 벤치마크의 시야 커널은 반복마다 작은 배열을 만들고 있어 개선 여지가 있다). numpy 식은 `out=`과 제자리 연산으로 임시 배열을 줄인다. Python 객체로 된 개체 목록·기록 리스트를 무한히 쌓지 않는다.
- 로그: stdout은 journald로 보내고 요약만 낮은 빈도로 찍는다. 통계·프레임은 자체 이진 파일로 쓰고 보존 정책(4절)으로 솎는다. journald 용량 상한은 시스템 설정에서 따로 건다.
- RSS가 서서히 오르면 먼저 측정(`tracemalloc`, RSS 대 배열 총량 비교)한 뒤 `MALLOC_ARENA_MAX`나 jemalloc을 시도한다. 사례마다 효과가 달라 미리 정할 수 없다.

### Gaps
- numpy 배열 위주 장기 프로세스의 메모리 단편화를 직접 다룬 자료는 찾지 못했다(찾은 것은 웹 서비스 사례).
- `systemd.exec` 문서 뒤쪽(`Environment=`, `StandardOutput=`, 로그 속도 제한)은 페이지가 길어 읽지 못했다.
- `CPUQuota`와 스레드 수 불일치로 인한 지연은 알려진 현상에 근거한 추론이고 이번에 1차 자료로 확인하지 않았다.
- VM 안의 systemd 버전(Ubuntu 판)에 따라 `RestartSteps` 등 일부 지시어가 없을 수 있다. VM에서 `systemctl --version` 확인이 필요하다.

## 7. 관찰 화면: 시뮬레이션과 분리된 브라우저 뷰어

### Takeaway
시뮬레이션은 압축된 이진 프레임을 파일에 덧붙이기만 하고, 별도의 작은 웹 서버 프로세스가 그 파일을 읽어 WebSocket(실시간)과 HTTP(되감기)로 내보내는 구조가 루프를 뷰어 장애·지연에서 떼어 놓는다. 수천 개체는 브라우저 쪽에서 부담이 아니며(PixiJS v8은 M3에서 100만 입자 60fps), 실제 제약은 프레임 저장량과 휴대폰 회선이다.

### Cited Findings
- Vivarium(flowersteam)은 시뮬레이션 서버(`run_server.py`)와 클라이언트를 분리하고 "uses gRPC to communicate between server and clients", 웹 화면은 Panel(`panel serve scripts/run_interface.py`)로 만들었으며 Jupyter 노트북이 제어기로 붙을 수 있다. Python 3.10~3.12, 스스로 "very alpha"라고 적는다 — [flowersteam/vivarium](https://github.com/flowersteam/vivarium)
- Neural MMO는 THREE.js 기반 웹 클라이언트(렌더러 모듈 이름 Embyr)를 썼고 로컬에서 `http://localhost:8080/forge/embyr/`로 띄웠다. NeurIPS 2022 대회에서는 리플레이를 압축 파일(`.lzma`/`.replay`, 변환 스크립트는 `pickle.loads(lz4.block.decompress(data))`)로 저장해 별도 웹 뷰어에 올려 재생했고 Unity 클라이언트도 따로 있었다. WebSocket 계층에 대한 설명은 찾지 못했다(모두 요약으로만 확인) — [AIcrowd: replay webviewer tutorial](https://www.aicrowd.com/showcase/replay-webviewer-tutorial); [neural-mmo README (shwang fork)](https://github.com/shwang/neural-mmo)
- ALIEN은 브라우저가 아니라 네이티브 앱이다(CUDA 물리 + Vulkan 렌더링, RTX 20 이상 또는 AMD RDNA2 이상 필요). 관찰 기능으로 "Inspection windows for every object, creature and genome", "Evolution dashboard with lineages, population statistics and long-term plots", 저장 지점과 되돌아가기를 둔다 — [chrxh/alien](https://github.com/chrxh/alien)
- Biosim4는 실시간 뷰어 없이 선택한 세대만 영상 파일로 만들고 세대별 통계를 텍스트 로그로 남긴다 — [biosim4 README](https://github.com/davidrmiller/biosim4)
- PixiJS v8 ParticleContainer: 저자의 MacBook Pro M3에서 일반 Sprite는 "200,000 at 60fps", ParticleContainer는 "1,000,000 at 60fps!". 기본은 위치만 매 프레임 GPU에 올리고(dynamic) 크기·회전·색은 static으로 둘 수 있다(2024-10-03 글) — [PixiJS blog: ParticleContainer v8](https://pixijs.com/blog/particlecontainer-v8)
- SSE(Server-Sent Events)는 서버 → 클라이언트 단방향이고 "a simple stream of text data which must be encoded using UTF-8"이다. HTTP/2가 아니면 브라우저·도메인당 동시 연결이 6개로 제한된다. 끊기면 자동으로 다시 연결한다 — [MDN: Using server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events)

### Inferences
- 권장 구조:
  1. 시뮬레이션 프로세스: k스텝마다 프레임 레코드를 `frames/000042.bin`에 덧붙인다(고정 헤더: 매직·스키마 버전·스텝·개체 수 → 개체별 고정 폭 레코드). 프레임의 파일 위치를 적은 색인(`frames.idx`: 스텝, 파일 번호, 바이트 위치)도 덧붙인다. 파일은 일정 크기나 시간마다 새로 연다. 뷰어 쪽으로 가는 소켓·큐·잠금이 없으므로 뷰어가 죽거나 느려도 루프는 영향을 받지 않는다.
  2. 뷰어 서버(별도 systemd 유닛, 읽기 전용): 색인을 따라가며 새 프레임을 WebSocket 이진 메시지로 내보내고(실시간), `GET /frame?step=...` 또는 구간 요청으로 과거 프레임을 준다(되감기). 통계 파일을 읽어 개체 수 차트용 JSON을 준다. 단일 개체 상세(가중치, 계보)는 최신 체크포인트나 계보 로그에서 요청이 올 때만 읽는다.
  3. 브라우저: `ArrayBuffer`를 `DataView`·typed array로 바로 해석해 그린다. 개체 수천이면 Canvas2D로 시작해도 되고, 확대·축소·색 구분이 늘면 PixiJS(WebGL)로 옮긴다. 차트는 가벼운 라이브러리 하나면 된다.
- 실시간 프레임에 SSE는 맞지 않는다. 텍스트 전용이라 이진 프레임을 base64로 부풀려야 한다. 통계 숫자 몇 개만 흘리는 용도라면 SSE로도 충분하다. 개체 선택 같은 클라이언트 → 서버 요청은 일반 HTTP 요청으로 처리하면 WebSocket을 양방향으로 쓸 필요도 없다.
- 프레임 형식 예(개체당 16바이트): `x uint16, y uint16`(세계 크기를 65535로 양자화), `heading uint8`, `radius uint8`, `energy uint8`, `flags uint8`, `id uint32`, `lineage_color uint32`. N=2000이면 32 KB. 휴대폰 회선에서 초당 2프레임이면 64 KB/s이고, 서버가 클라이언트 화면 크기에 맞춰 프레임을 건너뛰어 보내면 더 줄일 수 있다.
- 실시간 화면과 기록은 분리해서 생각한다. 실시간은 서버가 "가장 최근 프레임"만 보내면 되므로 기록 간격과 무관하게 부드럽게 볼 수 있다(시뮬레이션이 최신 프레임을 고정 위치 파일이나 공유 메모리에 덮어쓰는 방식). 되감기는 저장된 프레임 간격만큼만 가능하고, 더 촘촘한 재현이 필요하면 가장 가까운 체크포인트에서 분기 실행(5절)으로 다시 돌리면 된다. 3절의 결정성이 여기서 저장 공간을 아껴 준다.
- 프레임 기록 비용: 기록 자체는 배열 몇 개를 양자화해 `write`하는 일이라 수십 µs~수백 µs 수준일 것이고, 부담은 디스크 양이다(4절: 초당 1프레임에 하루 2.8 GB). 기본값을 "10초당 1프레임 + 최근 1시간만 초당 1프레임"처럼 두 단계로 두는 것을 권한다.
- 단일 개체 관찰에 필요한 것: 프레임에 개체 id가 있어야 화면에서 누른 점을 개체로 되돌릴 수 있고, 상세는 뷰어 서버가 최신 체크포인트에서 그 개체의 행을 잘라 준다. 신경망 활동까지 실시간으로 보려면 "관찰 대상 id"를 파일 하나에 적어 두고 시뮬레이션이 그 개체의 은닉 상태만 추가 기록하게 하는 식의 느슨한 연결이 루프 독립성을 지킨다.
- 선례에서 빌릴 것: ALIEN의 화면 구성(개체별 관찰 창, 계보·개체 수·장기 그래프 대시보드, 저장 지점으로 돌아가기), Neural MMO의 "리플레이 파일을 웹 뷰어가 재생" 방식, Vivarium의 서버–화면 프로세스 분리. Vivarium처럼 뷰어가 시뮬레이션 서버에 직접 질의하는 구조는 뷰어가 루프를 붙잡을 수 있어 24시간 운영에는 파일 경유가 더 안전하다.
- 외부 접근은 기지의 기존 방식(VPN 또는 리버스 프록시 뒤)에 얹고, 뷰어 서버는 읽기 전용으로 둔다.

### Gaps
- The Bibites(Unity 네이티브로 알고 있으나 이번에 확인하지 않음)와 Lenia 웹 포트는 조사하지 못했다.
- Canvas2D로 수천 개 원을 그릴 때의 프레임 속도 측정값, 휴대폰 브라우저에서의 값은 찾지 못했다(PixiJS 수치는 M3 노트북 기준).
- Neural MMO 웹 클라이언트의 전송 방식(WebSocket 여부)과 내부 구조는 원문 코드를 열지 못했다.
- 프레임 기록 비용은 추정이고 측정하지 않았다.
