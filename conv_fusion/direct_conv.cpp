#include "direct_conv.h"

#include <vector>

using namespace std;

#define ConvParamsDeclaration() \
    float* Sum, float* X, float* O, float* I, float* F, \
    float alpha, float beta, unsigned flags, \
    unsigned N, unsigned K, unsigned D, unsigned H, unsigned W, \
    unsigned WN, unsigned HWN, unsigned DHWN, unsigned C, \
    unsigned KRST, unsigned RST, unsigned RS, unsigned T, \
    unsigned R, unsigned S, \
    unsigned magic_RS, unsigned shift_RS, unsigned magic_S, unsigned shift_S, \
    int pad_d, int pad_h, int pad_w, \
    unsigned str_d, unsigned str_h, unsigned str_w, \
    unsigned dil_d, unsigned dil_h, unsigned dil_w, \
    unsigned P2, unsigned Q, unsigned PQk, unsigned Qk, unsigned k, \
    unsigned magic_PQk, unsigned shift_PQk, unsigned magic_Qk, unsigned shift_Qk, \
    unsigned magic_k, unsigned shift_k, \
    unsigned QN, unsigned PQN, unsigned MPQN, \
    unsigned gridNw, unsigned gridQNw, unsigned gridPQNw, unsigned gridMPQNw

#define ConfigParams(MacroOp) \
	MacroOp(N), MacroOp(C), MacroOp(K), \
	MacroOp(H), MacroOp(W), MacroOp(S), MacroOp(R),\
	MacroOp(str_h), MacroOp(str_w), \
	MacroOp(pad_h), MacroOp(pad_w), \
	MacroOp(dil_h), MacroOp(dil_w) \

#define ConvParams(MacroOp) \
    MacroOp(Sum), MacroOp(X), MacroOp(O), MacroOp(I), MacroOp(F), \
    MacroOp(alpha), MacroOp(beta), MacroOp(flags), \
    MacroOp(N), MacroOp(K), MacroOp(D), MacroOp(H), \
    MacroOp(W), MacroOp(WN), MacroOp(HWN), MacroOp(DHWN), MacroOp(C), \
    MacroOp(KRST), MacroOp(RST), MacroOp(RS), \
    MacroOp(T), MacroOp(R), MacroOp(S), \
    MacroOp(magic_RS), MacroOp(shift_RS), \
    MacroOp(magic_S), MacroOp(shift_S), \
    MacroOp(pad_d), MacroOp(pad_h), MacroOp(pad_w), \
    MacroOp(str_d), MacroOp(str_h), MacroOp(str_w), \
    MacroOp(dil_d), MacroOp(dil_h), MacroOp(dil_w), \
    MacroOp(P2), MacroOp(Q), MacroOp(PQk), MacroOp(Qk), \
    MacroOp(k), MacroOp(magic_PQk), MacroOp(shift_PQk), \
    MacroOp(magic_Qk), MacroOp(shift_Qk), \
    MacroOp(magic_k), MacroOp(shift_k), \
    MacroOp(QN), MacroOp(PQN), MacroOp(MPQN), \
    MacroOp(gridNw), MacroOp(gridQNw), MacroOp(gridPQNw), MacroOp(gridMPQw)

#define CopyConfig(Name) \
    Name = args[# Name]

#define PushBackParam(Name) \
	{ \
	    Parameter param(); \
	    memcpy(param.value, Name, sizeof(Name)); \
	    params.push_back(param); \
	}

void InitLarge(map<string, unsigned> args)
{
	ConvParamsDeclaration();
	ConfigParams(CopyConfig);

	unsigned blockN, blockK;
	for (blockN : {128, 64}) {
		if (N % blockN == 0) {
			break;
		}
	}
	
	vector<unsigned> tileK;
	tileK = blockN == 128 ? {128, 64, 32} : {128, 64};
	for (blockK : tileK) {
		unsigned mod = K % blockK;
		if (mod == 0 || mod > blockK - 32) {
			break;
		}
	}

	ostringstream os;
	os << clss << "_direct_fprop_" << blockK << "x" << blockN;
	name = os.str();
	threads = kernel_specs[name]["threads"];

	gridK = CeilDiv(K, blockK);
	gridN = CeilDiv(N. blockN);
	RS = args["R"] * args["S"];
	TRS = RS;
	TRSK = TRS * args["K"];
	
	k = ClosestDivisor(gridK, 128 / blockK);
	P2 = args[P] / 2;
	Q2 = args[Q] * 2;
	Qk = Q2 * k;
	PQk = args["P"] * args["Q"] * k;

	magic_PQk = Magic64(PQK);
	magic_Qk = Magic64(Qk);
	magic_k = Magic32(Qk, k);
	magic_RS = Magic32(TRS + 32, RS);
	magic_S = Magic32(RS + 32, S);

	bsum_warps = blockN / 64;
	gridNw = gridN * bsum_warps;
	gridQNw = Q * gridNw;
	gridPQNw = P * gridQNw;
	gridMPQNw = M * gridPQNw;
	gridMPQ = M * P * W;
	grid = (gridMPQ * k, gridK / k, gridN);

	WN = W * N;
	HWN = H * W * N;
	DHWN = D * H * W * N;
	QN = Q * N;
	PQN = P * Q * N;
	MPQN = M * P * Q * N;

	ConvParams(PushBack);
}

void InitSmall(map<string, unsigned> args)
{
	ConvParamsDeclaration();
	ConfigParams(CopyConfig);

	assert(N % 4 == 0 || N == 1 || N == 2);

	unsigned blockN;
	for (blockN : {32, 16, 8, 4, 2, 1}) {
		if (N % blockN == 0) {
			break;
		}
	}

	map<unsigned, vector<unsigned> > sb_params_in, sb_params_out;

    if (P == 1) {
        // # 1D conv
        sb_params_in = {
            // #blkN: supM, shfM, supP, shfP, supQ, shfQ, supN, shfN
            {32 , { 0x000, 0,   0x000, 0,   0x000, 0,    7,   5  }}, // # 1x1  nnn(nn)
            {16 , { 0x000, 0,   0x000, 0,   0x102, 1,    3,   4  }}, // # 1x2  xnn(nn)
            {8  , { 0x000, 0,   0x000, 0,   0x201, 2,    1,   3  }}, // # 2x2  xxn(nn)
            {4  , { 0x000, 0,   0x000, 0,   0x300, 3,    0,   2  }}, // # 2x4  xxx(nn)
            {2  , { 0x000, 0,   0x000, 0,   0x300, 4,    0,   1  }}, // # 4x4  xxx(xn)
            {1  , { 0x000, 0,   0x000, 0,   0x300, 5,    0,   0  }}, // # 4x8  xxx(xx)
        }
        sb_params_out = {
            // #blkN:  supM,  supP,  supQ, supN
            {32 , { 0x000, 0x000, 0x000, 31 }}, // # 1x1  nnnnn
            {16 , { 0x000, 0x000, 0x104, 15 }}, // # 1x2  xnnnn
            {8  , { 0x000, 0x000, 0x203,  7 }}, // # 2x2  xxnnn
            {4  , { 0x000, 0x000, 0x302,  3 }}, // # 2x4  xxxnn
            {2  , { 0x000, 0x000, 0x401,  1 }}, // # 4x4  xxxxn
            {1  , { 0x000, 0x000, 0x500,  0 }}, // # 4x8  xxxxx
        }
    else {
        sb_params_in = {
            // #blkN: supM, shfM, supP, shfP, supQ, shfQ, supN, shfN
            {32 , { 0x000, 0,   0x000, 0,   0x000, 0,    7,   5  }}, // # 1x1  nnn(nn)
            {16 , { 0x000, 0,   0x000, 0,   0x102, 1,    3,   4  }}, // # 1x2  xnn(nn)
            {8  , { 0x000, 0,   0x102, 1,   0x101, 1,    1,   3  }}, // # 2x2  yxn(nn)
            {4  , { 0x000, 0,   0x102, 1,   0x200, 2,    0,   2  }}, // # 2x4  yxx(nn)
            {2  , { 0x000, 0,   0x201, 2,   0x100, 2,    0,   1  }}, // # 4x4  yyx(xn)
            {1  , { 0x000, 0,   0x201, 2,   0x100, 3,    0,   0  }}, // # 4x8  yyx(xx)
        }
        sb_params_out = {
            // #blkN:  supM,  supP,  supQ, supN
            {32 , { 0x000, 0x000, 0x000, 31 }}, // # 1x1  nnnnn
            {16 , { 0x000, 0x000, 0x104, 15 }}, // # 1x2  xnnnn
            {8  , { 0x000, 0x104, 0x103,  7 }}, // # 2x2  yxnnn
            {4  , { 0x000, 0x104, 0x202,  3 }}, // # 2x4  yxxnn
            {2  , { 0x000, 0x203, 0x201,  1 }}, // # 4x4  yyxxn
            {1  , { 0x000, 0x203, 0x300,  0 }}, // # 4x8  yyxxx
        }
    }

	blockM  = 1 << shiftM;
    blockP  = 1 << shiftP;
    blockQ  = 1 << shiftQ;
    gridM   = CeilDiv(M, blockM);
    gridP   = CeilDiv(P, blockP);
    gridQ   = CeilDiv(Q, blockQ);
    gridN   = CeilDiv(N, blockN);
    gridK   = CeilDiv(K, 64);
    gridP2  = gridP / 2;
    gridQ2  = gridQ * 2;

    RS       = R * S;
    TRS      = T * RS;
    TRSK     = K * TRS;
    n        = ClosestDivisor(gridN, 2);
    k        = ClosestDivisor(gridK, 2);
    nk       = n * k;
    Qnk      = gridQ2 * nk;
    PQnk     = gridP * gridQ * nk;

    magic_PQnk = Magic64(PQnk);
    magic_Qnk  = Magic64(Qnk);
    magic_nk   = Magic32(Qnk, nk);
    magic_k    = Magic32(nk,   k);
    magic_RS   = Magic32(TRS, RS);
    magic_S    = Magic32(RS,   S);

	gridMPQ = gridM * gridP * gridQ
	grid    = (gridMPQ * nk, gridK / k, gridN / n)
}

DirectConv::DirectConv(ConfigParams(ConfigParamsType))
{

	if (N % 64 == 0 && K % vect_size == 0) {
		InitLarge(args);
	} else {
		InitSmall(args);
	}
}

int PrepareAuxiliaryParameter()
{

}

int GenCUDAModule(bool vblock, 
	GetParamOffsetInConstMem offset, CUmodule &module)
{

}
