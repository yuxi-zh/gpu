import os
import sys
import tvm
import topi
import argparse
import numpy as np
from topi.util import get_const_tuple

def test_conv(batch, in_channel, in_size, num_filter, ft_size, st_size, padding, layout):
	
	in_height = in_width = in_size
	ft_height = ft_width = ft_size
	st_height = st_width = st_size

	if layout is 'HWCN':
		A = tvm.placeholder((in_height, in_width, in_channel, batch), name='A')
	elif layout is 'NHWC':
		A = tvm.placeholder((batch, in_height, in_width, in_channel), name='A')
	elif layout is 'NCHW':
		A = tvm.placeholder((batch, in_channel, in_height, in_width), name='A')
	else:
		raise ValueError('No definition for layout {}'.format(layout))
	
	if layout is 'NCHW':
		W = tvm.placeholder((num_filter, in_channel, ft_height, ft_width), name='W')
	elif layout is 'NHWC':
		W = tvm.placeholder((num_filter, ft_height, ft_width, in_channel), name='W')
	elif layout is 'HWCN':
		W = tvm.placeholder((ft_height, ft_width, in_channel, num_filter), name='W')

	ctx = tvm.context('cuda', 0)

	a_shape = get_const_tuple(A.shape)
	w_shape = get_const_tuple(W.shape)
	a_np = np.random.uniform(size=a_shape).astype(A.dtype)
	w_np = np.random.uniform(size=w_shape).astype(W.dtype)

	a = tvm.nd.array(a_np, ctx)
	w = tvm.nd.array(w_np, ctx)
	
	def build_and_run(sch, target, B):
		with tvm.build_config(auto_unroll_max_step=1400, unroll_explicit=False):
			b_shape = get_const_tuple(B.shape)
			func = tvm.build(sch, [A, W, B], target)
        	b = tvm.nd.array(np.zeros(get_const_tuple(B.shape), dtype=B.dtype), ctx)
        	func(a, w, b)
        	evaluator = func.time_evaluator(func.entry_name, ctx, number=1)
        	return evaluator(a, w, b).mean * 1e3		
	try:
		with tvm.target.cuda() as target:
			B0 = topi.nn.conv2d(A, W, [st_height, st_width], padding, layout)
			if layout is 'HWCN':
				sch = topi.cuda.schedule_conv2d_hwcn(B0)
			elif layout is 'NCHW':
				sch = topi.cuda.schedule_conv2d_nchw(B0)
			else:
				raise ValueError('No schedule for layout {}'.format(layout))
			tvm_time = build_and_run(sch, target, B0)
	except Exception, e:
		print >> sys.stderr, e
		tvm_time = float('nan')

	try:
		with tvm.target.cuda('-libs=cudnn') as target:
			B1 = topi.cuda.conv2d_cuda(A, W, [st_height, st_width], padding, layout)
			sch = topi.cuda.schedule_extern(B1)
			cudnn_time = build_and_run(sch, target, B1)
	except Exception, e:
		print >> sys.stderr, e
		cudnn_time = float('nan')

	sys.stdout.write(',%g, %g, %2.2f' % (tvm_time, cudnn_time, tvm_time / cudnn_time))

def main():

	args = [[3, 224, 64, 7, 2, 3],
			[64, 56, 64, 3, 1, 1],
			[64, 56, 64, 1, 1, 0],
			[64, 56, 128, 3, 2, 1],
			[64, 56, 128, 1, 2, 0],
			[128, 28, 128, 3, 1, 1],
			[128, 28, 256, 3, 2, 1],
			[128, 28, 256, 1, 2, 0],
			[256, 14, 256, 3, 1, 1],
			[256, 14, 512, 3, 2, 1],
			[256, 14, 512, 1, 2, 0],
			[512, 7, 512, 3, 1, 1],
			[128, 122, 128, 3, 1, 1],
			[1, 224, 64, 5, 1, 2],
			[64, 224, 64, 3, 1, 1],
			[64, 224, 32, 3, 1, 1],
			[32, 224, 9, 3, 1, 1]]
	for layout in ['NCHW', 'HWCN']:
		for arg in args:
			for batch in [1, 16, 32]:
				param = [batch] + arg + [layout]
				sys.stdout.write(','.join([str(p) for p in [layout, batch] + arg]))
				try:
					test_conv(*param)
				except Exception, e:
					print >> sys.stderr, e
					continue
				sys.stdout.write('\n')

if __name__ == '__main__':
	main()

