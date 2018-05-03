
#include "conv2d.h"

#ifndef TEST_DEVICE
__device__ void Conv2D(float* a, float* w, float* o, int3 __blockIdx)
#else
__global__ void Conv2D(float* a, float* w, float* o)
#endif
{
	float *ab = a + blk_y * a_h * a_w * a_c;

	// cache ouput in shared memory
	constexpr int _a_h_r = 1 << ((int)log2(static_cast<float>(blkdim_y)) / 2);
	constexpr int _a_w_r = 1 << (((int)log2(static_cast<float>(blkdim_y)) + 1) / 2);
	constexpr int _a_h = roundup(a_h, _a_h_r);
	constexpr int _a_w = roundup(a_w, _a_w_r);
	__shared__ float so[_a_h][_a_w][blkdim_y];

	for (int iy = 0; iy < a_h; iy++) {
		for (int ix = 0; ix < a_w; ix++) {
			for (int ic = 0; ic < ceildiv(a_c, blkdim_x); ic++) {
				const int _w_h = roundup(w_h, _w_h_r);
				const int _w_w = roundup(w_h, _w_w_r); 
				// cooperatively load shared padded input
				__shared__ float ap[_w_h][_w_w][blkdim_x];
				// assume _w_h * _w_w >= blkdim_y
				// assume blkdim_y is power of 2
				for (int fs = 0; fs < (_w_h * _w_w) / blkdim_y; fs++) {
					int ty = (fs * blkdim_y + thd_y) / _w_h;
					int tx = (fs * blkdim_y + thd_y) % _w_h;
					int tc = ic * blkdim_x + thd_x;
#if padding == SAME
					// map (ty, tx, tc) to (aty, atx, atc)
					int aty = ty * s_h - p_t;
					int atx = tx * s_w - p_l;
					// pad input data
					if ((aty < 0 || aty >= a_h) || (atx < 0 || atx >= a_w)) {
						ap[ty][tx][tc] = 0;
					} else {
						ap[ty][tx][tc] = ab[(((aty * a_w) + atx) * blkdim_x) + thd_x];
					}
#else
					int aty = ty * s_h;
					int atx = tx * s_w;
					ap[ty][tx][tc] = ab[(((aty * a_w) + atx) * blkdim_x) + thd_x];
#endif
				}
				// cooperatively load shared filter
				__shared__ float ft[_w_h][_w_w][blkdim_x][blkdim_y];
				for (int ty = 0; ty < _w_h; ty++) {
					for (int tx = 0; tx < _w_w; tx++) {
						ft[ty][tx][tc][tf] = w[((((ty * a_w) + tx) * blkdim_x) + thd_x) * blkdim_y + thd_y];
					}
				}
				// do convolution computation
				for (int ry = 0; ry < _w_h; ry++) {
					for (int rx = 0; rx < _w_w; rx++) {
						float reduce_c = ap[ry][rx][thd_x] * ft[ry][rx][thd_x][thd_y];
						// assume blkdim_x is power of 2, and not greater than 32
						for (int rc = 1; rc <= int(log2(static_cast<float>(blkdim_x))); rc *= 2) {
							float tmp1 = __shfl_up_sync(0xffffffff, reduce_c, i, blkdim_x);
							if (((laneid & blkdim_x) & (i-1)) == 0) {
								reduce_c += tmp1;
							}
						}
						if (thd_x == 0) {
							so[iy][ix][thd_y] += reduce_c;
						}
					}
				}
			}
		}
	}
	
	// assume (_a_h * _a_w) can be devided by blkdim.x without reminder
	for (int fs = 0; fs < (_a_h * _a_w) / blkdim_x; iy++) {
		int iy = (fs * blkdim_x + thd_x) / _a_h;
		int ix = (fs * blkdim_x + thd_x) / _a_w;
		if (iy < a_h && ix < a_w) {
			int tf = blk_x * blkdim_x + thd_y;
			o[(((iy * o_w) + ix) * w_f) + tf] = so[iy][ix][thd_y];
		}
	}
}

