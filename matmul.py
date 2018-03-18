import tvm

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

schedule[C].bind(cxo, blk_x)
schedule[C].bind(cyo, blk_y)

cxi, cxii = schedule[C].split(cxi, factor=TX)
cyi, cyii = schedule[C].split(cyi, factor=TY)
schedule[C].reorder(cxo, cyo, cxii, cyii, ko, ki, cxi, cyi)

schedule[C].bind(cxii, thd_x)
schedule[C].bind(cyii, thd_y)

cxyi = schedule[C].fuse(cxi, cyi)
schedule[C].unroll(cxyi)

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

# AL = schedule.cache_read(AS, 'local', [C])
# schedule[AL].compute_at(schedule[C], ki)

# BL = schedule.cache_read(BS, 'local', [C])
# schedule[BL].compute_at(schedule[C], ki)

# lower_func = tvm.lower(schedule, [A, B, C], simple_mode=True)
# print(lower_func)

build_func = tvm.build(schedule, [A, B, C], target='cuda', name="mm")
print(build_func.imported_modules[0].get_source())
with open('mm_{}_{}_{}_{}_{}_{}.cu'.format(M, K, N, TX, TY, SK),'w') as source:
	source.write(build_func.imported_modules[0].get_source())