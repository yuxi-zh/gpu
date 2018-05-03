import tvm
import math
import fractions
import numpy as np

def lcm(a, b):
	return abs(a * b) / fractions.gcd(a, b) if a and b else 0

def conv2d_nchw(input_shape, filter_shape, stride, padding, thread):

	if input_shape[1] != filter_shape[2]:
		raise ValueError('Input and filter have different channels')

	N, C, HI, WI = input_shape
	HF, WF, C, M = filter_shape
	HS, WS = stride

	I = tvm.placeholder((N, C, HI, WI), name="input")
	F = tvm.placeholder((HF, WF, C, M), name="filter")

	HO = (HI - HF) / HS + 1
	WO = (WI - WF) / WS + 1

	target = 'cuda'
	ctx = tvm.context(target, 0)
	I2COL = tvm.compute((N, C * HF * WF, HO * WO), lambda nn, yy, xx:
		I[nn, yy / (HF * WF), 
		(xx / HO) * HS + (yy % (HF * WF)) / WF, 
		(xx % HO) * WS + (yy % (HF * WF)) % WF], name="I2COL")
	schedule = tvm.create_schedule(I2COL.op)
	# blk_z = tvm.thread_axis('blockIdx.z')
	# blk_y = tvm.thread_axis('blockIdx.y')
	# blk_x = tvm.thread_axis('blockIdx.x')
	# thd_y = tvm.thread_axis((0, thread), 'threadIdx.y')
	# thd_x = tvm.thread_axis((0, thread), 'threadIdx.x')
	# n, c, s = schedule[I2COL].op.axis
	# so, si = schedule[I2COL].split(s, 4)
	# co, ci = schedule[I2COL].split(c, HF * WF)
	# # schedule[I2COL].reorder(by, bx, tx, c, s)
	# schedule[I2COL].bind(n, blk_z)
	# schedule[I2COL].bind(co, blk_y)
	# schedule[I2COL].bind(so, blk_x)
	# schedule[I2COL].bind(ci, thd_x)
	# SI2COL = schedule.cache_read(I, 'shared', [I2COL])

	lower = tvm.lower(schedule, [I, I2COL], simple_mode=True)
	print(lower)
	build = tvm.build(schedule, [I, I2COL], target=target)
	_I = tvm.nd.array(np.random.rand(*input_shape).astype(I.dtype), ctx)
	_I2COL = tvm.nd.array(np.zeros((N, C * HF * WF, HO * WO), dtype=I2COL.dtype), ctx)
	build(_I, _I2COL)
	evaluator = build.time_evaluator(build.entry_name, ctx, number=10)
	print('time = {} ms'.format(evaluator(_I, _I2COL).mean * 1e3))

def conv1d_ncw(input_shape, filter_shape, stride, padding, thread):

	if input_shape[1] != filter_shape[1]:
		raise ValueError('Input and filter have different channels')

	N, C, LI = input_shape
	K, C, M = filter_shape

	I = tvm.placeholder((N, C, LI), name='input')
	F = tvm.placeholder((K, C, M), name='filter')

	LO = (LI - K) / stride + 1
	
	target = 'cuda'
	ctx = tvm.context(target, 0)

	I2COL = tvm.compute((N, C * K, LO), lambda nn, yy, xx: 
		I[nn, yy / K, xx * stride + yy % K], name='I2COL')
	schedule = tvm.create_schedule(I2COL.op)
	blk_x = tvm.thread_axis('blockIdx.x')
	thd_x = tvm.thread_axis((0, thread), 'threadIdx.x')
	n, c, l = schedule[I2COL].op.axis
	fused_index = schedule[I2COL].fuse(c, l)
	_, tx = schedule[I2COL].split(fused_index, thread)
	schedule[I2COL].bind(n, blk_x)
	schedule[I2COL].bind(tx, thd_x)
	lower = tvm.lower(schedule, [I, I2COL], simple_mode=True)
	build = tvm.build(schedule, [I, I2COL], target=target)
	_I = tvm.nd.array(np.random.rand(*input_shape).astype(I.dtype), ctx)
	_I2COL = tvm.nd.array(np.zeros((N, C * K, LO), dtype=I2COL.dtype), ctx)
	build(_I, _I2COL)
	evaluator = build.time_evaluator(build.entry_name, ctx, number=10)
	print('time = {} ms'.format(evaluator(_I, _I2COL).mean * 1e3))

	F2COL = tvm.compute((C * K, M), lambda yy, xx:
		F[yy % K, yy / K, xx], name="F2COL")
	schedule = tvm.create_schedule(F2COL.op)
	blk_z = tvm.thread_axis('blockIdx.z')
	blk_y = tvm.thread_axis('blockIdx.y')
	blk_x = tvm.thread_axis('blockIdx.x')
	thd_x = tvm.thread_axis((0, thread), 'threadIdx.x')
	c, l = schedule[F2COL].op.axis
	fused_index = schedule[F2COL].fuse(c, l)
	bx, tx = schedule[F2COL].split(fused_index, thread)
	schedule[F2COL].bind(bx, blk_x)
	schedule[F2COL].bind(tx, thd_x)
	lower = tvm.lower(schedule, [F, F2COL], simple_mode=True)
	build = tvm.build(schedule, [F, F2COL], target=target)
	_F = tvm.nd.array(np.random.rand(*filter_shape).astype(F.dtype), ctx)
	_F2COL = tvm.nd.array(np.zeros((C * K, M), dtype=F2COL.dtype), ctx)
	build(_F, _F2COL)
	evaluator = build.time_evaluator(build.entry_name, ctx, number=10)
	print('time = {} ms'.format(evaluator(_F, _F2COL).mean * 1e3))

	I2COL = tvm.placeholder((N, C * K, LO), name="I2COL")
	F2COL = tvm.placeholder((C * K, M), name="F2COL")
	rk = tvm.reduce_axis((0, C * K), name="rc")
	CONV1D = tvm.compute((N, LO, M), lambda nn, yy, xx:
		tvm.sum(I2COL[nn, rk, yy] * F2COL[rk, xx], axis=rk), name="CONV1D")
	schedule = tvm.create_schedule(CONV1D.op)

	SI2COL = schedule.cache_read(I2COL, "shared", [CONV1D])
	SF2COL = schedule.cache_read(F2COL, "shared", [CONV1D])
	LI2COL = schedule.cache_read(SI2COL, "local", [CONV1D])
	LF2COL = schedule.cache_read(SF2COL, "local", [CONV1D])
	LCONV1D = schedule.cache_write(CONV1D, "local")

	block_factor = 8 * thread
	blk_x = tvm.thread_axis("blockIdx.x")
	blk_y = tvm.thread_axis("blockIdx.y")
	blk_z = tvm.thread_axis("blockIdx.z")
	thd_x = tvm.thread_axis((0, thread), "threadIdx.x")
	thd_y = tvm.thread_axis((0, thread), "threadIdx.y")
	thd_xz = tvm.thread_axis((0, 2), "vthread", name="vx")
	thd_yz = tvm.thread_axis((0, 2), "vthread", name="vy")

	bz, y, x = schedule[CONV1D].op.axis
	by, yi = schedule[CONV1D].split(y, factor=block_factor)
	bx, xi = schedule[CONV1D].split(x, factor=block_factor)
	schedule[CONV1D].bind(bz, blk_z)
	schedule[CONV1D].bind(by, blk_y)
	schedule[CONV1D].bind(bx, blk_x)
	schedule[CONV1D].reorder(by, bx, yi, xi)

	tyz, yi = schedule[CONV1D].split(yi, nparts=2)
	ty, yi = schedule[CONV1D].split(yi, nparts=thread)
	txz, xi = schedule[CONV1D].split(xi, nparts=2)
	tx, xi = schedule[CONV1D].split(xi, nparts=thread)
	schedule[CONV1D].bind(tyz, thd_yz)
	schedule[CONV1D].bind(txz, thd_xz)
	schedule[CONV1D].bind(ty, thd_y)
	schedule[CONV1D].bind(tx, thd_x)
	schedule[CONV1D].reorder(tyz, txz, ty, tx, yi, xi)
	schedule[LCONV1D].compute_at(schedule[CONV1D], tx)

	_, yo, xo = LCONV1D.op.axis
	ko, ki = schedule[LCONV1D].split(rk, factor=8)
	kt, ki = schedule[LCONV1D].split(ki, factor=1)
	schedule[LCONV1D].reorder(ko, kt, ki, yo, xo)
	schedule[SI2COL].compute_at(schedule[LCONV1D], ko)
	schedule[SF2COL].compute_at(schedule[LCONV1D], ko)
	schedule[LCONV1D].unroll(kt)
	schedule[LI2COL].compute_at(schedule[LCONV1D], kt)
	schedule[LF2COL].compute_at(schedule[LCONV1D], kt)
	# Schedule for A's shared memory load
	ty, xi = schedule[SI2COL].split(schedule[SI2COL].op.axis[0], nparts=thread)
	_, xi = schedule[SI2COL].split(schedule[SI2COL].op.axis[1], factor=thread * 4)
	tx, xi = schedule[SI2COL].split(xi, nparts=thread)
	schedule[SI2COL].bind(ty, thd_y)
	schedule[SI2COL].bind(tx, thd_x)
	schedule[SI2COL].vectorize(xi)
	# Schedule for B' shared memory load
	ty, xi = schedule[SF2COL].split(schedule[SF2COL].op.axis[0], nparts=thread)
	_, xi = schedule[SF2COL].split(schedule[SF2COL].op.axis[1], factor=thread * 4)
	tx, xi = schedule[SF2COL].split(xi, nparts=thread)
	schedule[SF2COL].bind(ty, thd_y)
	schedule[SF2COL].bind(tx, thd_x)
	schedule[SF2COL].vectorize(xi)

	lower = tvm.lower(schedule, [I2COL, F2COL, CONV1D], simple_mode=True)
	build = tvm.build(schedule, [I2COL, F2COL, CONV1D], target=target)
	_CONV1D = tvm.nd.array(np.zeros((N, LO, M), dtype=CONV1D.dtype), ctx)
	_I2COL = tvm.nd.array(np.random.rand(N, C * K, LO).astype(I2COL.dtype), ctx)
	_F2COL = tvm.nd.array(np.random.rand(C * K, M).astype(_F2COL.dtype), ctx)
	build(_I2COL, _F2COL, _CONV1D)
	evaluator = build.time_evaluator(build.entry_name, ctx, number=10)
	print('time = {} ms'.format(evaluator(_I2COL, _F2COL, _CONV1D).mean * 1e3))

def main():
	# conv1d_ncw([1, 3, 5], [3, 3, 3], 1, [0, 0], 8)
	# conv1d_ncw([1, 3, 224], [7, 3, 64], 2, [3, 3], 8)
	# conv1d_ncw([16, 3, 224], [7, 3, 64], 2, [3, 3], 8)
	# conv1d_ncw([32, 3, 224], [7, 3, 64], 2, [3, 3], 8)

	# conv2d_nchw([1, 3, 4, 4], [3, 3, 3, 3], [1, 1], [0, 0, 0, 0], 8)
	conv2d_nchw([1, 3, 224, 224], [7, 7, 3, 64], [2, 2], [3, 3, 3, 3], 32)
	conv2d_nchw([16, 3, 224, 224], [7, 7, 3, 64], [2, 2], [3, 3, 3, 3], 32)
	conv2d_nchw([32, 3, 224, 224], [7, 7, 3, 64], [2, 2], [3, 3, 3, 3], 32)

if __name__ == '__main__':
	main()