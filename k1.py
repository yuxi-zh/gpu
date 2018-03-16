import tvm

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

lower_func = tvm.lower(schedule, [A, B, C, D], simple_mode=True)
print(lower_func)

TX = 16
TY = 16
SK = 16
SL = 16

blk_x = tvm.thread_axis("blockIdx.x")
blk_y = tvm.thread_axis("blockIdx.y")
thd_x = tvm.thread_axis((0, TX), "threadIdx.x")
thd_y = tvm.thread_axis((0, TY), "threadIdx.y")


dx, dy = D.op.axis
dxo, dyo, dxi, dyi = schedule[D].tile(dx, dy, TX, TY)
ko, ki = schedule[D].split(k, factor=SK)
schedule[D].reorder(dxo, dyo, dxi, dyi, ko, ki)
schedule[D].bind(dxo, blk_x)
schedule[D].bind(dyo, blk_y)
schedule[D].bind(dxi, thd_x)
schedule[D].bind(dyi, thd_y)

schedule[T].compute_at(schedule[D], dyi)

# txo, tyo, txi, tyi = schedule[T].tile(T.op.axis[0], T.op.axis[1], TX, TY)
lo, li = schedule[T].split(l, factor=SL)
# schedule[T].reorder(txo, tyo, txi, tyi, lo, li)
# schedule[T].bind(txi, thd_x)
# schedule[T].bind(tyi, thd_y)

lower_func = tvm.lower(schedule, [A, B, C, D], simple_mode=True)
print(lower_func)

# SB = schedule.cache_read(B, 'shared', [T])
# schedule[SB].compute_at(schedule[T], lo)
# sbx, sby = SB.op.axis
# sbxo, sbxi = schedule[SB].split(sbx, nparts=TX)
# sbyo, sbyi = schedule[SB].split(sby, nparts=TY)
# schedule[SB].bind(sbxo, thd_x)
# schedule[SB].bind(sbyo, thd_y)

# SC = schedule.cache_read(C, 'shared', [T])
# schedule[SC].compute_at(schedule[T], lo)
# scx, scy = SC.op.axis
# scxo, scxi = schedule[SC].split(scx, nparts=TX)
# scyo, scyi = schedule[SC].split(scy, nparts=TY)
# schedule[SC].bind(scxo, thd_x)
# schedule[SC].bind(scyo, thd_y)

# # ST = schedule.cache_write(T, 'shared')

# lower_func = tvm.lower(schedule, [A, B, C, D], simple_mode=True)
# print(lower_func)

build_func = tvm.build(schedule, [A, B, C, D], target='llvm', name="K1")
print(build_func.imported_modules[0].get_source())
