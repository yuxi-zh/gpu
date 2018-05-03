#ifndef __TVM_CONV2D_H__
#define __TVM_CONV2D_H__

#ifndef a_n
#error "a_n undefined!"
#endif

#ifndef a_h
#error "a_h undefined!"
#endif

#ifndef a_w
#error "a_w undefined!"
#endif

#ifndef a_c
#error "a_c undefined!"
#endif

#ifndef w_h
#error "w_h undefined!"
#endif

#ifndef w_w
#error "w_w undefined!"
#endif

#ifndef w_c
#error "w_c undefined!"
#else
#if w_c != a_c
#error "w_c != a_c"
#endif
#endif

#ifndef w_f
#error "w_f undefined!"
#endif

#ifndef s_h
#error "s_h undefined!"
#endif

#ifndef s_w
#error "s_w undefined!"
#endif

// block size of dimension x, must be power of 2 
#ifndef blkdim_x
#error "blkdim_x undefined!"
#endif

// block size of dimension y, must be power of 2
#ifndef blkdim_y
#error "blkdim_y undefined!"
#endif

#ifndef padding
#error "padding undefined!"
#else
#if (padding != SAME) || (padding != VALID)
#error "unknown padding!"
#endif
#endif

#define ceildiv(x, y) (((x) + (y) - 1) / (y))
#define roundup(x, y) (ceildiv((x), (y)) * (y))

#if padding == SAME
#define o_h (ceildiv(a_h, s_h))
#define o_w (ceildiv(a_w, s_w))
#define p_t (o_h / 2)
#define p_b ((o_h - 1) * (s_h) + (w_h) - (a_h) - (p_t))
#define p_l (o_w / 2)
#define p_r ((o_w - 1) * (s_w) + (w_w) - (a_w) - (p_l))
#elif padding == VALID
#error "VALID padding is not supported now!"
#endif

#ifndef TEST_DEVICE
#define blk_y (__blockIdx.y)
#define blk_x (__blockIdx.x)
#else
#define blk_y (blockidx.y)
#define blk_x (blockidx.x)
#endif

#define thd_y (threadIdx.y)
#define thd_x (threadIdx.x)
#define laneid ((thd_y * blkdim_y + thd_x) & 0x1f)

#endif