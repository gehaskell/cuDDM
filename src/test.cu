#include <cuda.h>
#include <cufft.h>

#include <opencv2/opencv.hpp>
using namespace cv;

// Test file to see where we are going wrong

__global__ void Test_AbsDifference(cufftReal *d_diff, float *d_frame1, float *d_frame2, int width, int height) {
	int x = threadIdx.x + blockIdx.x * 32;
	int y = threadIdx.y + blockIdx.y * 32;

	if (x <= width-1 && y <= height-1) {
		int pos_offset = y * width + x;
		d_diff[pos_offset] = abs(d_frame1[pos_offset] - d_frame2[pos_offset]);
	}
	return;
}

__global__ void Test_processFFT(cufftComplex *d_data, float *d_fft, int width, int height) {
	// Takes output of cuFFT R2C operation, normalises it (i.e. divides by px count), takes the magnitude and adds it to the accum_array

	int size = width * height;

	int j = threadIdx.x + blockIdx.x * 32;
	int i = threadIdx.y + blockIdx.y * 32;

	float mag;
	if (j <= width-1 && i <= height-1) {
		int pos_offset = i * width + j;
		int sym_w = width / 2 + 1; // to deal with complex (hermitian) symmetry

		if (j >= sym_w) {
			// real ->  d_data[i*sym_w+(width-j)].x
			// img  -> -d_data[i*sym_w+(width-j)].y
			mag = cuCabsf(d_data[i*sym_w+(width-j)]);

		} else {
			// real -> d_data[i*sym_w+j].x
			// img  -> d_data[i*sym_w+j].y
			mag = cuCabsf(d_data[i*sym_w+j]);
		}

		// add to fft_accum
		d_fft[pos_offset] = mag;
	}
}


int _main() {
	VideoCapture cap("/home/ghaskell/projects_Git/cuDDM/data/colloid_0.5um_vid.mp4");

	Mat img1;
	Mat img2;

	int w = 5;
	int h = 5;
	int delta = 3;

	float* h_frame1 = new float[w * h];
	float* h_frame2 = new float[w * h];

	cap >> img1;
	while (delta >= 0) {
		cap >> img2;
		delta--;
	}

	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			h_frame1[y * w + x] = (float) img1.data[((img1.step)/img1.elemSize1())* y + img1.channels() * x];
			h_frame2[y * w + x] = (float) img2.data[((img2.step)/img2.elemSize1())* y + img2.channels() * x];
		}
	}

	float *d_frame1, *d_frame2;
	cudaMalloc((void **) &d_frame1, w * h * sizeof(float));
	cudaMalloc((void **) &d_frame2, w * h * sizeof(float));
	cudaMemcpy(d_frame1, h_frame1,  w * h * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_frame2, h_frame2,  w * h * sizeof(float), cudaMemcpyHostToDevice);

	cufftReal *d_diff_local;
	cudaMalloc((void **) &d_diff_local, w * h * sizeof(cufftReal));
	cufftReal *h_diff_local = new cufftReal[w * h];

	dim3 blockDim(32, 32, 1);
	dim3 gridDim((int)ceil(w/32.0), (int)ceil(h/32.0), 1);

	Test_AbsDifference<<<gridDim, blockDim>>>(d_diff_local, d_frame1, d_frame2, w, h);
	cudaMemcpy(h_diff_local, d_diff_local,  w * h * sizeof(cufftReal), cudaMemcpyDeviceToHost);

	// FFT
	cufftHandle plan;
	if ((cufftPlan2d(&plan, w, h, CUFFT_R2C)) != CUFFT_SUCCESS) {
		std::cout << "cufft plan error" << std::endl;
	}
	cufftComplex *d_fft_local; // should i malloc this????
	cudaMalloc((void **) &d_fft_local, w * (h/2 + 1) * sizeof(cufftComplex));
	if ((cufftExecR2C(plan, d_diff_local, d_fft_local)) != CUFFT_SUCCESS) {
		std::cout << "cufft exec error" << std::endl;
	}

	float *d_fft;
	cudaMalloc((void **) &d_fft, w * h * sizeof(float));
	float *h_fft = new float[w*h];
	Test_processFFT<<<gridDim, blockDim>>>(d_fft_local, d_fft, w, h);
	cudaMemcpy(h_fft, d_fft,  w * h * sizeof(float), cudaMemcpyDeviceToHost);


	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			std::cout << h_frame1[y*w + x] << ","<< h_frame2[y*w + x] << ","<< h_diff_local[y*w + x] << "," << h_fft[y*w + x]<< " " << std::endl;
		}
	}





	std::cout << "END" << std::endl;


}
