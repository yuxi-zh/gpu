import tvm
import numpy as np

M = 1024
K = 1024
L = 1024
N = 1024

A = tvm.placeholder((M, K), name="A")
B = tvm.placeholder((K, L), name="B")
C = tvm.placeholder((L, N), name="C")
k = tvm.reduce_axis((0, K), name="k")
l = tvm.reduce_axis((0, L), name="l")
T = tvm.compute((K, N), lambda tx, ty: tvm.sum(B[tx, l] * C[l, ty], axis=l), name="T")
D = tvm.compute((M, N), lambda dx, dy: tvm.sum(A[dx, k] * T[k, dy], axis=k), name="D")

schedule = tvm.create_schedule(D.op)

TX = 16
TY = 16
SK = 16
SL = 16

BX = M / TX
BY = N / TY

VT = 4

blk_x = tvm.thread_axis("blockIdx.x")
blk_y = tvm.thread_axis("blockIdx.y")
thd_x = tvm.thread_axis((0, TX), "threadIdx.x")
thd_y = tvm.thread_axis((0, TY), "threadIdx.y")

dx, dy = D.op.axis
dxo, dyo, dxi, dyi = schedule[D].tile(dx, dy, BX, BY)
ko, ki = schedule[D].split(k, factor=SK)
dxi, dxii = schedule[D].split(dxi, nparts=BX/TX)
dyi, dyii = schedule[D].split(dyi, nparts=BY/TY)
dxiii, dxii = schedule[D].split(dxii, factor=TX)
dyiii, dyii = schedule[D].split(dyii, factor=TY)
schedule[D].reorder(dxo, dyo, dxi, dyi, dxii, dyii, ko, ki, dxiii, dyiii)
vthd_x = tvm.thread_axis((0, BX/TX), "vthread", name="vx")
vthd_y = tvm.thread_axis((0, BY/TY), "vthread", name="vy")
schedule[D].bind(dxo, blk_x)
schedule[D].bind(dyo, blk_y)
schedule[D].bind(dxi, vthd_x)
schedule[D].bind(dyi, vthd_y)
schedule[D].bind(dxii, thd_x)
schedule[D].bind(dyii, thd_y)

AS = schedule.cache_read(A, 'shared', [D])
schedule[AS].compute_at(schedule[D], ko)
asx, asy = AS.op.axis
asxo, asxi = schedule[AS].split(asx, factor=TX)
asyo, asyi = schedule[AS].split(asy, factor=TY)
schedule[AS].reorder(asxi, asyi, asxo, asyo)
schedule[AS].bind(asxi, thd_x)
schedule[AS].bind(asyi, thd_y)

tx, ty = T.op.axis
schedule[T].compute_at(schedule[D], ko)
lo, li = schedule[T].split(l, factor=SL)
schedule[T].reorder(lo, ty, tx, li)

BS = schedule.cache_read(B, 'shared', [T])
schedule[BS].compute_at(schedule[T], lo)
bsx, bsy = BS.op.axis
bsxi, bsx = schedule[BS].split(bsx, factor=TX)
bsyi, bsy = schedule[BS].split(bsy, factor=TY)
schedule[BS].reorder(bsx, bsy, bsxi, bsyi)
schedule[BS].bind(bsx, thd_x)
schedule[BS].bind(bsy, thd_y)

CS = schedule.cache_read(C, 'shared', [T])
schedule[CS].compute_at(schedule[T], lo)
csx, csy = CS.op.axis
csxi, csx = schedule[CS].split(csx, factor=TX)
csyi, csy = schedule[CS].split(csy, factor=TY)
schedule[CS].reorder(csx, csy, csxi, csyi)
schedule[CS].bind(csx, thd_x)
schedule[CS].bind(csy, thd_y)

TS = schedule.cache_read(T, 'shared', [D])
schedule[TS].compute_at(schedule[D], ko)
tsx, tsy = TS.op.axis
tsxi, tsx = schedule[TS].split(tsx, factor=TX)
tsyi, tsy = schedule[TS].split(tsy, factor=TY)
schedule[TS].reorder(tsx, tsy, tsxi, tsyi)
schedule[TS].bind(tsx, thd_x)
schedule[TS].bind(tsy, thd_y)

AL = schedule.cache_read(AS, 'local', [D])
schedule[AL].compute_at(schedule[D], ki)

TL = schedule.cache_read(TS, 'local', [D])
schedule[TL].compute_at(schedule[D], ki)

lower_func = tvm.lower(schedule, [A, B, C, D], simple_mode=True)
print(lower_func)

build_func = tvm.build(schedule, [A, B, C, D], target='cuda', name="K1")
print(build_func.imported_modules[0].get_source())

ctx = tvm.context("cuda", 0)
high = 1024
a = tvm.nd.array(np.random.uniform(high=high, size=M*K).astype(A.dtype).reshape((M,K)), ctx)
b = tvm.nd.array(np.random.uniform(high=high, size=K*L).astype(B.dtype).reshape((K,L)), ctx)
c = tvm.nd.array(np.random.uniform(high=high, size=L*N).astype(C.dtype).reshape((L,N)), ctx)
d = tvm.nd.array(np.zeros(M,N, dtype=D.dtype).reshape((M,N)), ctx)
evaluator = build_func.time_evaluator(build_func.entry_name, ctx, number=1)
print('time: %f ms' % (evaluator(a, b, c, d).mean * 1e3))