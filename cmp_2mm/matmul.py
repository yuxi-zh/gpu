"""Example code to do square matrix multiplication."""
import tvm
import os
from tvm.contrib import nvcc
from tvm.contrib import spirv
import numpy as np

TASK="gemm"
USE_MANUAL_CODE = False

@tvm.register_func
def tvm_callback_cuda_compile(code):
    ptx =  nvcc.compile_cuda(code, target="ptx")
    return ptx

def write_code(code, fname):
    with open(fname, "w") as f:
        f.write(code)

@tvm.register_func
def tvm_callback_cuda_postproc(code):
    if not os.path.exists("perf"):
        os.mkdir("perf")
    write_code(code, "perf/%s_generated.cu" % TASK)
    if USE_MANUAL_CODE:
        code = open("perf/%s_manual.cu" % TASK).read()
    return code

def gemm(m, n, l, 
                transa=False, transb=True, 
                thd_x=8, thd_y=8,
                scale_x=8, scale_y=8, 
                k_factor=8):

    ashape = (m, l) if not transa else (l, m)
    bshape = (l, n) if not transb else (n, l)

    A = tvm.placeholder(ashape, name='A')
    B = tvm.placeholder(bshape, name='B')
    k = tvm.reduce_axis((0, l), name='k')

    _A = tvm.compute((l, m), lambda y, x: A[x, y], name="TA") if not transa else A
    _B = tvm.compute((l, n), lambda y, x: B[x, y], name="TB") if transb else B
    C = tvm.compute((m, n), lambda y, x: tvm.sum(_A[k, y] * _B[k, x], axis=k), name="C")

    s = tvm.create_schedule(C.op)

    if not transa:
        s[_A].compute_inline()
    if transb:
        s[_B].compute_inline()

    AA = s.cache_read(_A, "shared", [C])
    BB = s.cache_read(_B, "shared", [C])
    AL = s.cache_read(AA, "local", [C])
    BL = s.cache_read(BB, "local", [C])
    CC = s.cache_write(C, "local")

    block_factor_x = scale_x * thd_x
    block_factor_y = scale_y * thd_y
    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    thread_x = tvm.thread_axis((0, thd_x), "threadIdx.x")
    thread_y = tvm.thread_axis((0, thd_y), "threadIdx.y")
    thread_xz = tvm.thread_axis((0, 2), "vthread", name="vx")
    thread_yz = tvm.thread_axis((0, 2), "vthread", name="vy")

    by, yi = s[C].split(C.op.axis[0], factor=block_factor_y)
    bx, xi = s[C].split(C.op.axis[1], factor=block_factor_x)
    s[C].bind(by, block_y)
    s[C].bind(bx, block_x)
    s[C].reorder(by, bx, yi, xi)

    tyz, yi = s[C].split(yi, nparts=2)
    ty, yi = s[C].split(yi, nparts=thd_y)
    txz, xi = s[C].split(xi, nparts=2)
    tx, xi = s[C].split(xi, nparts=thd_x)
    s[C].bind(tyz, thread_yz)
    s[C].bind(txz, thread_xz)
    s[C].bind(ty, thread_y)
    s[C].bind(tx, thread_x)
    s[C].reorder(tyz, txz, ty, tx, yi, xi)
    s[CC].compute_at(s[C], tx)

    yo, xo = CC.op.axis
    ko, ki = s[CC].split(k, factor=k_factor)
    kt, ki = s[CC].split(ki, factor=1)
    s[CC].reorder(ko, kt, ki, yo, xo)
    s[AA].compute_at(s[CC], ko)
    s[BB].compute_at(s[CC], ko)
    s[CC].unroll(kt)
    s[AL].compute_at(s[CC], kt)
    s[BL].compute_at(s[CC], kt)
    
    if transa:
        xo, xi = s[AA].split(AA.op.axis[1], factor=thd_x * 4)
        tx, xi = s[AA].split(xi, nparts=thd_x)
        ty, _ = s[AA].split(AA.op.axis[0], nparts=thd_y)    
        s[AA].vectorize(xi)
    else:
        tx, _ = s[AA].split(AA.op.axis[1], nparts=thd_x)
        ty, _ = s[AA].split(AA.op.axis[0], nparts=thd_y)

    s[AA].bind(ty, thread_y)
    s[AA].bind(tx, thread_x)

    if not transb:
        xo, xi = s[BB].split(BB.op.axis[1], factor=thd_x * 4)
        tx, xi = s[BB].split(xi, nparts=thd_x)
        ty, _ = s[BB].split(BB.op.axis[0], nparts=thd_y)
        s[BB].vectorize(xi)
    else:
        tx, _ = s[BB].split(BB.op.axis[1], nparts=thd_x)
        ty, _ = s[BB].split(BB.op.axis[0], nparts=thd_y)
    
    s[BB].bind(ty, thread_y)
    s[BB].bind(tx, thread_x)

    with tvm.build_config(auto_unroll_max_step=128, unroll_explicit=False):
        lower_func = tvm.lower(s, [A, B, C], simple_mode=True)
        # build_func = tvm.build(s, [A, B, C], 'cuda')
        build_func = None
    
    return lower_func, build_func

def check(f, m, n, l, transa, transb):
    ctx = tvm.context('cuda', 0)
    ashape = (m, l) if not transa else (l, m)
    bshape = (l, n) if not transb else (n, l)
    a_np = np.random.uniform(size=ashape).astype('float32')
    b_np = np.random.uniform(size=bshape).astype('float32')
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(b_np, ctx)
    c = tvm.nd.array(np.zeros((m, n), dtype=C.dtype), ctx)
    lhs = a_np if not transa else a_np.T
    rhs = b_np if not transb else b_np.T
    np.testing.assert_allclose(c.asnumpy(), np.dot(lhs, rhs), rtol=1e-5)
    timer_f = f.time_evaluator(f.entry_name, ctx, number=50)
    t = timer_f(a, b, c).mean
    GFLOPS = (2 * m * n * l) / (t * 1e3) / 1e6
    print(transa, transb, '%4d %4d %4d %f %f' % (m, n, l, t, GFLOPS))

from itertools import product

if __name__ == "__main__":

    transas = [True, False]
    transbs = [False, True]
    scales = [64, 128, 512, 1024, 2014]

    for ta, tb, scale in product(transas, transbs, scales):
        lower, build = gemm(scale, scale, scale, ta, tb)
        check(build, scale, scale, scale, ta, tb)

    # test_gemm(512, 1, 512, 1, 1, 1, 8, 8);
    # test_gemm(512, 2, 512, 1, 8, 1, 1, 8);
    # test_gemm(512, 4, 512, 4, 8, 1, 1, 8);
    # test_gemm(512, 8, 512, 1, 8, 8, 8, 8);
    # test_gemm(512, 16, 512, 2, 4, 8, 8, 8);
    # test_gemm(512, 32, 512, 4, 8, 8, 8, 8);
    # test_gemm(512, 64, 512, 8, 8, 8, 8, 8);
