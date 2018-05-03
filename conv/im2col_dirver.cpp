#include <vector>

using namespace std;

extern
void drive_im2col(vector<int> input_shape, vector<int> filter_shape, vector<int> stride, vector<int> padding, int thread);

int main(int argc, char const *argv[]) {
	drive_im2col({1, 3, 224, 224}, {7, 7, 3, 64}, {2, 2}, {3, 3}, 8);
	drive_im2col({16, 3, 224, 224}, {7, 7, 3, 64}, {2, 2}, {3, 3}, 8);
	drive_im2col({32, 3, 224, 224}, {7, 7, 3, 64}, {2, 2}, {3, 3}, 8);
	drive_im2col({1, 64, 56, 56}, {3, 3, 64, 64}, {1, 1}, {1, 1}, 8);
	drive_im2col({16, 64, 56, 56}, {3, 3, 64, 64}, {1, 1}, {1, 1}, 8);
	drive_im2col({32, 64, 56, 56}, {3, 3, 64, 64}, {1, 1}, {1, 1}, 8);
	drive_im2col({1, 64, 56, 56}, {1, 1, 64, 64}, {1, 1}, {0, 0}, 8);
	drive_im2col({16, 64, 56, 56}, {1, 1, 64, 64}, {1, 1}, {0, 0}, 8);
	drive_im2col({32, 64, 56, 56}, {1, 1, 64, 64}, {1, 1}, {0, 0}, 8);
	drive_im2col({1, 64, 56, 56}, {3, 3, 64, 128}, {2, 2}, {1, 1}, 8);
	drive_im2col({16, 64, 56, 56}, {3, 3, 64, 128}, {2, 2}, {1, 1}, 8);
	drive_im2col({32, 64, 56, 56}, {3, 3, 64, 128}, {2, 2}, {1, 1}, 8);
	drive_im2col({1, 64, 56, 56}, {1, 1, 64, 128}, {2, 2}, {0, 0}, 8);
	drive_im2col({16, 64, 56, 56}, {1, 1, 64, 128}, {2, 2}, {0, 0}, 8);
	drive_im2col({32, 64, 56, 56}, {1, 1, 64, 128}, {2, 2}, {0, 0}, 8);
	drive_im2col({1, 128, 28, 28}, {3, 3, 128, 128}, {1, 1}, {1, 1}, 8);
	drive_im2col({16, 128, 28, 28}, {3, 3, 128, 128}, {1, 1}, {1, 1}, 8);
	drive_im2col({32, 128, 28, 28}, {3, 3, 128, 128}, {1, 1}, {1, 1}, 8);
	drive_im2col({1, 128, 28, 28}, {3, 3, 128, 256}, {2, 2}, {1, 1}, 8);
	drive_im2col({16, 128, 28, 28}, {3, 3, 128, 256}, {2, 2}, {1, 1}, 8);
	drive_im2col({32, 128, 28, 28}, {3, 3, 128, 256}, {2, 2}, {1, 1}, 8);
	drive_im2col({1, 128, 28, 28}, {1, 1, 128, 256}, {2, 2}, {0, 0}, 8);
	drive_im2col({16, 128, 28, 28}, {1, 1, 128, 256}, {2, 2}, {0, 0}, 8);
	drive_im2col({32, 128, 28, 28}, {1, 1, 128, 256}, {2, 2}, {0, 0}, 8);
	drive_im2col({1, 256, 14, 14}, {3, 3, 256, 256}, {1, 1}, {1, 1}, 8);
	drive_im2col({16, 256, 14, 14}, {3, 3, 256, 256}, {1, 1}, {1, 1}, 8);
	drive_im2col({32, 256, 14, 14}, {3, 3, 256, 256}, {1, 1}, {1, 1}, 8);
	drive_im2col({1, 256, 14, 14}, {3, 3, 256, 512}, {2, 2}, {1, 1}, 8);
	drive_im2col({16, 256, 14, 14}, {3, 3, 256, 512}, {2, 2}, {1, 1}, 8);
	drive_im2col({32, 256, 14, 14}, {3, 3, 256, 512}, {2, 2}, {1, 1}, 8);
	drive_im2col({1, 256, 14, 14}, {1, 1, 256, 512}, {2, 2}, {0, 0}, 8);
	drive_im2col({16, 256, 14, 14}, {1, 1, 256, 512}, {2, 2}, {0, 0}, 8);
	drive_im2col({32, 256, 14, 14}, {1, 1, 256, 512}, {2, 2}, {0, 0}, 8);
	drive_im2col({1, 512, 7, 7}, {3, 3, 512, 512}, {1, 1}, {1, 1}, 8);
	drive_im2col({16, 512, 7, 7}, {3, 3, 512, 512}, {1, 1}, {1, 1}, 8);
	drive_im2col({32, 512, 7, 7}, {3, 3, 512, 512}, {1, 1}, {1, 1}, 8);
	drive_im2col({1, 128, 122, 122}, {3, 3, 128, 128}, {1, 1}, {1, 1}, 8);
	drive_im2col({16, 128, 122, 122}, {3, 3, 128, 128}, {1, 1}, {1, 1}, 8);
	drive_im2col({32, 128, 122, 122}, {3, 3, 128, 128}, {1, 1}, {1, 1}, 8);
	drive_im2col({1, 1, 224, 224}, {5, 5, 1, 64}, {1, 1}, {2, 2}, 8);
	drive_im2col({16, 1, 224, 224}, {5, 5, 1, 64}, {1, 1}, {2, 2}, 8);
	drive_im2col({32, 1, 224, 224}, {5, 5, 1, 64}, {1, 1}, {2, 2}, 8);
	drive_im2col({1, 64, 224, 224}, {3, 3, 64, 64}, {1, 1}, {1, 1}, 8);
	drive_im2col({16, 64, 224, 224}, {3, 3, 64, 64}, {1, 1}, {1, 1}, 8);
	drive_im2col({32, 64, 224, 224}, {3, 3, 64, 64}, {1, 1}, {1, 1}, 8);
	drive_im2col({1, 64, 224, 224}, {3, 3, 64, 32}, {1, 1}, {1, 1}, 8);
	drive_im2col({16, 64, 224, 224}, {3, 3, 64, 32}, {1, 1}, {1, 1}, 8);
	drive_im2col({32, 64, 224, 224}, {3, 3, 64, 32}, {1, 1}, {1, 1}, 8);
	drive_im2col({1, 32, 224, 224}, {3, 3, 32, 9}, {1, 1}, {1, 1}, 8);
	drive_im2col({16, 32, 224, 224}, {3, 3, 32, 9}, {1, 1}, {1, 1}, 8);
	drive_im2col({32, 32, 224, 224}, {3, 3, 32, 9}, {1, 1}, {1, 1}, 8);

	return 0;
}