import tvm
import numpy as np

M = 1024
K = 1024
N = 1024

A = tvm.placeholder((M, K), name="A")
B = tvm.placeholder((K, N), name="B")
k = tvm.reduce_axis((0, K), name="k")
C = tvm.compute((M, N), lambda x, y: tvm.sum(A[x, k] * B[k, y], axis=k), name="C")

schedule = tvm.create_schedule(C.op)

lower_func = tvm.lower(schedule, [A, B, C], simple_mode=True)
print(lower_func)

TX = 16
TY = 16
SK = 16

BX = M / TX
BY = N / TY

blk_x = tvm.thread_axis("blockIdx.x")
blk_y = tvm.thread_axis("blockIdx.y")
thd_x = tvm.thread_axis((0, TX), "threadIdx.x")
thd_y = tvm.thread_axis((0, TY), "threadIdx.y")

cxo, cyo, cxi, cyi = schedule[C].tile(C.op.axis[0], C.op.axis[1], BX, BY)
ko, ki = schedule[C].split(k, factor=SK)
cxi, cxii = schedule[C].split(cxi, nparts=BX/TX)
cyi, cyii = schedule[C].split(cyi, nparts=BY/TY)
cxiii, cxii = schedule[C].split(cxii, factor=TX)
cyiii, cyii = schedule[C].split(cyii, factor=TY)
schedule[C].reorder(cxo, cyo, cxi, cyi, cxii, cyii, ko, ki, cxiii, cyiii)
vthd_x = tvm.thread_axis((0, BX/TX), "vthread", name="vx")
vthd_y = tvm.thread_axis((0, BY/TY), "vthread", name="vy")
schedule[C].bind(cxo, blk_x)
schedule[C].bind(cyo, blk_y)
schedule[C].bind(cxi, vthd_x)
schedule[C].bind(cyi, vthd_y)
schedule[C].bind(cxii, thd_x)
schedule[C].bind(cyii, thd_y)

# cxyi = schedule[C].fuse(cxi, cyi)
# schedule[C].unroll(cxyi)

AS = schedule.cache_read(A, 'shared', [C])
schedule[AS].compute_at(schedule[C], ko)
asx, asy = AS.op.axis
asxo, asxi = schedule[AS].split(asx, factor=TX)
asyo, asyi = schedule[AS].split(asy, factor=TY)
schedule[AS].reorder(asxi, asyi, asxo, asyo)
schedule[AS].bind(asxi, thd_x)
schedule[AS].bind(asyi, thd_y)
asxyo = schedule[AS].fuse(asxo, asyo)
schedule[AS].unroll(asxyo)

BS = schedule.cache_read(B, 'shared', [C])
schedule[BS].compute_at(schedule[C], ko)
bsx, bsy = BS.op.axis
bsxo, bsxi = schedule[BS].split(bsx, factor=TX)
bsyo, bsyi = schedule[BS].split(bsy, factor=TY)
schedule[BS].reorder(bsxi, bsyi, bsxo, bsyo)
schedule[BS].bind(bsxi, thd_x)
schedule[BS].bind(bsyi, thd_y)
bsxyo = schedule[BS].fuse(bsxo, bsyo)
schedule[BS].unroll(bsxyo)

AL = schedule.cache_read(AS, 'local', [C])
schedule[AL].compute_at(schedule[C], ki)

BL = schedule.cache_read(BS, 'local', [C])
schedule[BL].compute_at(schedule[C], ki)

lower_func = tvm.lower(schedule, [A, B, C], simple_mode=True)
print(lower_func)

ctx = tvm.context("cuda", 0)
high = 1024
a = tvm.nd.array(np.random.uniform(high=high, size=M*K).astype(A.dtype).reshape((M,K)), ctx)
b = tvm.nd.array(np.random.uniform(high=high, size=K*N).astype(B.dtype).reshape((K,N)), ctx)
d = tvm.nd.array(np.zeros((M,N)).astype(D.dtype).reshape((M,N)), ctx)
evaluator = build_func.time_evaluator(build_func.entry_name, ctx, number=1)
print('time: %f ms' % (evaluator(a, b, c, d).mean * 1e3))