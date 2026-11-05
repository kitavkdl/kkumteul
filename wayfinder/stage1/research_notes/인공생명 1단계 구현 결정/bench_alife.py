"""Micro-benchmark: per-agent RNN matmul, cell-list neighbour search, ray-circle vision.
numpy vs numba (serial / prange / fastmath). float32, struct-of-arrays."""
import time, sys, platform, math
import numpy as np, numba
from numba import njit, prange

print("python", sys.version.split()[0], platform.machine(), "numpy", np.__version__, "numba", numba.__version__,
      "threading_layer_default_threads", numba.config.NUMBA_NUM_THREADS)


def bench(f, *a, rep=5):
    f(*a)
    ts = []
    for _ in range(rep):
        t = time.perf_counter(); f(*a); ts.append(time.perf_counter() - t)
    return min(ts) * 1e3


# ---------- (a) per-agent recurrent matmul: out[n] = tanh(W[n] @ h[n]) ----------
def rnn_matmul(W, h, out):
    out[:] = np.tanh(np.matmul(W, h[:, :, None])[:, :, 0])

def rnn_einsum(W, h, out):
    out[:] = np.tanh(np.einsum('nij,nj->ni', W, h))

def make_rnn(parallel, fastmath):
    @njit(parallel=parallel, fastmath=fastmath)
    def f(W, h, out):
        N, H, _ = W.shape
        for n in prange(N):
            for i in range(H):
                s = np.float32(0.0)
                for j in range(H):
                    s += W[n, i, j] * h[n, j]
                out[n, i] = np.tanh(s)
    return f

def make_rnn_axpy(parallel):
    # Wt[n, j, i] = W[n, i, j]; column-accumulate so SIMD needs no reassociation (no fastmath)
    @njit(parallel=parallel)
    def f(Wt, h, out):
        N, H, _ = Wt.shape
        for n in prange(N):
            acc = np.zeros(H, np.float32)
            for j in range(H):
                hj = h[n, j]
                for i in range(H):
                    acc[i] += Wt[n, j, i] * hj
            for i in range(H):
                out[n, i] = np.tanh(acc[i])
    return f

rnn_ser, rnn_ser_fm = make_rnn(False, False), make_rnn(False, True)
rnn_par, rnn_par_fm = make_rnn(True, False), make_rnn(True, True)
axpy_ser, axpy_par = make_rnn_axpy(False), make_rnn_axpy(True)


# ---------- (b) uniform grid / cell list ----------
@njit
def build_cells(x, y, cell, nx, ny):
    N = x.shape[0]
    cid = np.empty(N, np.int32)
    count = np.zeros(nx * ny + 1, np.int32)
    for i in range(N):
        cx = min(int(x[i] / cell), nx - 1); cy = min(int(y[i] / cell), ny - 1)
        c = cy * nx + cx; cid[i] = c; count[c + 1] += 1
    start = np.cumsum(count).astype(np.int32)
    order = np.empty(N, np.int32)
    fill = start[:-1].copy()
    for i in range(N):
        c = cid[i]; order[fill[c]] = i; fill[c] += 1
    return cid, start, order

def make_collide(parallel):
    @njit(parallel=parallel)
    def f(x, y, rad, cell, nx, ny, start, order, fx, fy):
        N = x.shape[0]
        for i in prange(N):
            cx = min(int(x[i] / cell), nx - 1); cy = min(int(y[i] / cell), ny - 1)
            ax = np.float32(0.0); ay = np.float32(0.0)
            for yy in range(max(cy - 1, 0), min(cy + 2, ny)):
                for xx in range(max(cx - 1, 0), min(cx + 2, nx)):
                    c = yy * nx + xx
                    for k in range(start[c], start[c + 1]):
                        j = order[k]
                        if j == i:
                            continue
                        dx = x[i] - x[j]; dy = y[i] - y[j]
                        d2 = dx * dx + dy * dy
                        rr = rad[i] + rad[j]
                        if d2 < rr * rr and d2 > 0.0:
                            d = math.sqrt(d2)
                            p = (rr - d) / d
                            ax += dx * p; ay += dy * p
            fx[i] = ax; fy[i] = ay
    return f

col_ser, col_par = make_collide(False), make_collide(True)

def collide_numpy(x, y, rad, fx, fy):
    dx = x[:, None] - x[None, :]; dy = y[:, None] - y[None, :]
    d2 = dx * dx + dy * dy
    rr = rad[:, None] + rad[None, :]
    m = (d2 < rr * rr) & (d2 > 0)
    d = np.sqrt(np.where(m, d2, 1.0))
    p = np.where(m, (rr - d) / d, 0.0)
    fx[:] = (dx * p).sum(1); fy[:] = (dy * p).sum(1)


# ---------- (c) ray-circle vision, N agents x R rays, range D ----------
def make_vision(parallel):
    @njit(parallel=parallel)
    def f(x, y, rad, theta, R, fov, D, cell, nx, ny, start, order, dist):
        N = x.shape[0]
        reach = int(math.ceil(D / cell))
        for i in prange(N):
            cx = min(int(x[i] / cell), nx - 1); cy = min(int(y[i] / cell), ny - 1)
            dxr = np.empty(R, np.float32); dyr = np.empty(R, np.float32)
            for r in range(R):
                a = theta[i] - 0.5 * fov + fov * r / (R - 1)
                dxr[r] = math.cos(a); dyr[r] = math.sin(a); dist[i, r] = D
            for yy in range(max(cy - reach, 0), min(cy + reach + 1, ny)):
                for xx in range(max(cx - reach, 0), min(cx + reach + 1, nx)):
                    c = yy * nx + xx
                    for k in range(start[c], start[c + 1]):
                        j = order[k]
                        if j == i:
                            continue
                        ox = x[j] - x[i]; oy = y[j] - y[i]
                        oc2 = ox * ox + oy * oy
                        lim = D + rad[j]
                        if oc2 > lim * lim:
                            continue
                        r2 = rad[j] * rad[j]
                        for r in range(R):
                            b = ox * dxr[r] + oy * dyr[r]
                            if b <= 0.0:
                                continue
                            disc = r2 - (oc2 - b * b)
                            if disc >= 0.0:
                                t = b - math.sqrt(disc)
                                if t > 0.0 and t < dist[i, r]:
                                    dist[i, r] = t
    return f

vis_ser, vis_par = make_vision(False), make_vision(True)

def vision_numpy(x, y, rad, theta, R, fov, D, dist):
    a = theta[:, None] - 0.5 * fov + fov * np.arange(R, dtype=np.float32)[None, :] / (R - 1)
    dxr = np.cos(a); dyr = np.sin(a)                       # (N,R)
    ox = x[None, :] - x[:, None]; oy = y[None, :] - y[:, None]   # (N,N) i->j
    oc2 = ox * ox + oy * oy
    b = ox[:, None, :] * dxr[:, :, None] + oy[:, None, :] * dyr[:, :, None]   # (N,R,N)
    disc = (rad * rad)[None, None, :] - (oc2[:, None, :] - b * b)
    ok = (b > 0) & (disc >= 0)
    t = np.where(ok, b - np.sqrt(np.where(ok, disc, 0.0)), np.inf)
    t[t <= 0] = np.inf
    dist[:] = np.minimum(t.min(2), D)


def run(N, H, R=16, threads=(1, 4, 8)):
    rng = np.random.default_rng(1)
    W = (rng.standard_normal((N, H, H)) / np.sqrt(H)).astype(np.float32)
    Wt = np.ascontiguousarray(W.transpose(0, 2, 1))
    h = rng.standard_normal((N, H)).astype(np.float32)
    out = np.empty((N, H), np.float32); out2 = np.empty_like(out)
    L = 1000.0 * math.sqrt(N / 1000.0)
    x = (rng.random(N) * L).astype(np.float32); y = (rng.random(N) * L).astype(np.float32)
    rad = (3 + 4 * rng.random(N)).astype(np.float32)
    theta = (rng.random(N) * 2 * np.pi).astype(np.float32)
    cell = np.float32(25.0); nx = ny = int(math.ceil(L / cell))
    D = np.float32(100.0); fov = np.float32(2.0)
    fx = np.empty(N, np.float32); fy = np.empty(N, np.float32)
    dist = np.empty((N, R), np.float32)
    print(f"\n=== N={N} H={H} R={R}  W bytes={W.nbytes/1e6:.0f} MB  L={L:.0f} ===")
    numba.set_num_threads(1)
    res = {}
    res['a numpy matmul'] = bench(rnn_matmul, W, h, out)
    res['a numpy einsum'] = bench(rnn_einsum, W, h, out)
    res['a numba serial strict'] = bench(rnn_ser, W, h, out)
    res['a numba serial fastmath'] = bench(rnn_ser_fm, W, h, out2)
    res['a numba serial axpy(no fastmath)'] = bench(axpy_ser, Wt, h, out2)
    rnn_ser(W, h, out); axpy_ser(Wt, h, out2)
    print("axpy bit-identical to strict scalar:", np.array_equal(out, out2))
    rnn_ser_fm(W, h, out2)
    print("fastmath bit-identical to strict scalar:", np.array_equal(out, out2),
          "max abs diff", float(np.abs(out - out2).max()))
    cid, start, order = build_cells(x, y, cell, nx, ny)
    res['b build cell list'] = bench(build_cells, x, y, cell, nx, ny)
    res['b collide numba serial'] = bench(col_ser, x, y, rad, cell, nx, ny, start, order, fx, fy)
    if N <= 5000:
        res['b collide numpy NxN'] = bench(collide_numpy, x, y, rad, fx, fy, rep=3)
    res['c vision numba serial'] = bench(vis_ser, x, y, rad, theta, R, fov, D, cell, nx, ny, start, order, dist)
    d_ser = dist.copy()
    if N <= 1000:
        res['c vision numpy NxRxN'] = bench(vision_numpy, x, y, rad, theta, R, fov, D, dist, rep=3)
        print("vision numpy vs numba max abs diff", float(np.abs(dist - d_ser).max()))
    for t in threads:
        numba.set_num_threads(t)
        res[f'a numba prange strict t={t}'] = bench(rnn_par, W, h, out)
        res[f'a numba prange fastmath t={t}'] = bench(rnn_par_fm, W, h, out)
        res[f'a numba prange axpy t={t}'] = bench(axpy_par, Wt, h, out2)
        res[f'b collide numba prange t={t}'] = bench(col_par, x, y, rad, cell, nx, ny, start, order, fx, fy)
        res[f'c vision numba prange t={t}'] = bench(vis_par, x, y, rad, theta, R, fov, D, cell, nx, ny, start, order, dist)
        print(f"t={t} vision prange bit-identical to serial:", np.array_equal(dist, d_ser))
    for k, v in res.items():
        print(f"{k:38s} {v:9.3f} ms")
    for t in threads:
        tot = res[f'a numba prange axpy t={t}'] + res['b build cell list'] + res[f'b collide numba prange t={t}'] + res[f'c vision numba prange t={t}']
        print(f"sum(a axpy + b + c) t={t}: {tot:.3f} ms -> {1000/tot:.0f} steps/s (kernels only)")


for N, H in [(1000, 32), (1000, 64), (1000, 128), (5000, 32), (5000, 64), (5000, 128)]:
    run(N, H)
