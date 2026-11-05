"""Batched per-agent RNN step benchmark: every agent has its own weights.
h' = tanh(W_in @ x + W_rec @ h + b), W_rec shape [N, H, H], float32.
"""
import time, os, sys
import numpy as np

I, O = 24, 6
rng = np.random.default_rng(0)

def bench(fn, reps):
    fn()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
    return min(t) * 1e3, float(np.median(t)) * 1e3

print("numpy", np.__version__, "threads env", os.environ.get("VECLIB_MAXIMUM_THREADS"), os.environ.get("OMP_NUM_THREADS"))
print(f"{'N':>5} {'H':>4} {'mem_MB':>8} {'matmul_ms':>10} {'einsum_ms':>10} {'full_step_ms':>12} {'masked_ms':>10}")
for N in (1000, 2000, 5000):
    for H in (16, 32, 64, 128):
        Wr = rng.standard_normal((N, H, H), dtype=np.float32) * 0.1
        Wi = rng.standard_normal((N, H, I), dtype=np.float32) * 0.1
        Wo = rng.standard_normal((N, O, H), dtype=np.float32) * 0.1
        b = np.zeros((N, H), dtype=np.float32)
        h = rng.standard_normal((N, H), dtype=np.float32)
        x = rng.standard_normal((N, I), dtype=np.float32)
        mask = (rng.random((N, H)) < 0.5).astype(np.float32)  # active-neuron mask
        mem = (Wr.nbytes + Wi.nbytes + Wo.nbytes + b.nbytes) / 2**20
        reps = 20 if N * H * H < 2e7 else 6
        f_mm = lambda: np.matmul(Wr, h[:, :, None])[:, :, 0]
        f_es = lambda: np.einsum('nij,nj->ni', Wr, h)
        def f_full():
            hn = np.tanh(np.matmul(Wi, x[:, :, None])[:, :, 0] + np.matmul(Wr, h[:, :, None])[:, :, 0] + b)
            return np.tanh(np.matmul(Wo, hn[:, :, None])[:, :, 0])
        def f_masked():
            hn = np.tanh(np.matmul(Wi, x[:, :, None])[:, :, 0] + np.matmul(Wr, (h * mask)[:, :, None])[:, :, 0] + b) * mask
            return np.tanh(np.matmul(Wo, hn[:, :, None])[:, :, 0])
        a, _ = bench(f_mm, reps); e, _ = bench(f_es, reps); f, _ = bench(f_full, reps); m, _ = bench(f_masked, reps)
        print(f"{N:>5} {H:>4} {mem:>8.1f} {a:>10.3f} {e:>10.3f} {f:>12.3f} {m:>10.3f}")

try:
    import numba
    from numba import njit, prange
    @njit(parallel=True, fastmath=True, cache=False)
    def step_nb(Wi, Wr, Wo, b, h, x, nh, out_h, out_y):
        N, H, _ = Wr.shape
        for n in prange(N):
            k = nh[n]  # per-agent active hidden count: loop only over the used block
            for i in range(k):
                s = b[n, i]
                for j in range(Wi.shape[2]):
                    s += Wi[n, i, j] * x[n, j]
                for j in range(k):
                    s += Wr[n, i, j] * h[n, j]
                out_h[n, i] = np.tanh(s)
            for o in range(Wo.shape[1]):
                s = 0.0
                for j in range(k):
                    s += Wo[n, o, j] * out_h[n, j]
                out_y[n, o] = np.tanh(s)
    print("\nnumba", numba.__version__, "threads", numba.get_num_threads())
    print(f"{'N':>5} {'H':>4} {'numba_full_ms':>14} {'numba_halfsize_ms':>18} {'numba_1thread_ms':>17}")
    for N in (1000, 5000):
        for H in (16, 32, 64, 128):
            Wr = rng.standard_normal((N, H, H)).astype(np.float32) * 0.1
            Wi = rng.standard_normal((N, H, I)).astype(np.float32) * 0.1
            Wo = rng.standard_normal((N, O, H)).astype(np.float32) * 0.1
            b = np.zeros((N, H), dtype=np.float32)
            h = rng.standard_normal((N, H)).astype(np.float32)
            x = rng.standard_normal((N, I)).astype(np.float32)
            oh = np.zeros((N, H), np.float32); oy = np.zeros((N, O), np.float32)
            nh_full = np.full(N, H, np.int64); nh_half = np.full(N, H // 2, np.int64)
            step_nb(Wi, Wr, Wo, b, h, x, nh_full, oh, oy)
            a, _ = bench(lambda: step_nb(Wi, Wr, Wo, b, h, x, nh_full, oh, oy), 10)
            c, _ = bench(lambda: step_nb(Wi, Wr, Wo, b, h, x, nh_half, oh, oy), 10)
            nt = numba.get_num_threads(); numba.set_num_threads(1)
            d, _ = bench(lambda: step_nb(Wi, Wr, Wo, b, h, x, nh_full, oh, oy), 5)
            numba.set_num_threads(nt)
            print(f"{N:>5} {H:>4} {a:>14.3f} {c:>18.3f} {d:>17.3f}")
except Exception as ex:
    print("numba bench failed:", ex)

# mutation cost: gaussian perturbation of one child genome vs whole population
N, H = 5000, 64
P = H * H + H * I + O * H + H
g = rng.standard_normal((N, P), dtype=np.float32)
t0 = time.perf_counter(); child = g[:100] + (rng.random((100, P)) < 0.1) * rng.standard_normal((100, P), dtype=np.float32) * 0.1; t1 = time.perf_counter()
print(f"\nmutating 100 children with P={P} params: {(t1 - t0) * 1e3:.2f} ms")
