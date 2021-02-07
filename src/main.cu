#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <opencv2/opencv.hpp>


using namespace cv;

typedef cufftReal Real;
typedef cufftComplex Complex;

struct VideoInfo {
    int width;
    int height;
    int frame_count;
};


void FFT_test() {
	unsigned int nx, ny;
	nx = ny = 64;

	Real *idata;
	Complex *odata;

	cudaMalloc((void **) &idata,  sizeof(Real) * nx * ny); 	// In data
	cudaMalloc((void **) &odata, sizeof(Complex) * nx * (ny/2 + 1)); // Out data

	cufftHandle plan;
	cufftPlan2d(&plan, nx, ny, CUFFT_R2C);
	cufftExecR2C(plan, idata, odata);




	Real *data;

	Complex *odata;

}

void LoadVideoToBuffer(float *d_ptr, int frame_count, VideoCapture cap, int w, int h) {
	// No bounds check! assume that w, h smaller than mat
	int num_elements = w * h;

	Mat vid_frame, data_frame;

	float *h_ptr = (float*) malloc(num_elements * frame_count * sizeof(float));

	for (int frame_idx; frame_idx < frame_count; frame_idx++) {
		cap >> vid_frame;

		if (vid_frame.empty()) {
			std::cout << "Loaded frame is empty." << std::endl;
		}

		vid_frame.convertTo(data_frame, CV_32F); // covert to float array

		int cols = data_frame.cols, rows = data_frame.rows;

		if (w > cols || h > rows) {
			std::cout << "Issue: specified width / height > cols / rows." << std::endl;
		}

		for (int y = 0; y < rows; y++) {
			for (int x = 0; x < cols; x++) {
				h_ptr[frame_idx * num_elements + y * w + x] = data_frame.at<float>(y,x);
			}
		}
	}
	cudaMemcpy(d_ptr, h_ptr, num_elements * frame_count * sizeof(float), cudaMemcpyHostToDevice);
}

__global__ void AbsDifference(float *d_buffer, float *d_diff, int frame1, int frame2, int tau_idx, int width, int height) {
	int size = width * height;

	int x = threadIdx.x + blockIdx.x * 32;
	int y = threadIdx.y + blockIdx.y * 32;


	if (x <= width-1 && y <= height-1) {
		int pos_offset = y * width + x;
		d_diff[tau_idx * size + pos_offset] = abs(d_buffer[frame1 * size + pos_offset] - d_buffer[frame2 * size + pos_offset]);
	}

	return;
}

__global__ void processFFT(Complex *d_data, float *d_fft_accum, int tau_idx, int width, int height) {
	// Takes output of cuFFT R2C operation, normalises it (i.e. divides by px count), takes the magnitude and adds it to the accum_array

	int size = width * height;

	int j = threadIdx.x + blockIdx.x * 32;
	int i = threadIdx.y + blockIdx.y * 32;

	if (j <= width-1 && i <= height-1) {
		int pos_offset = i * width + j;
		int sym_w = width / 2 + 1; // to deal with complex (hermitian) symmetry

		float val;
		if (j >= sym_w) {
			// real ->  d_data[i*sym_w+(width-j)].x
			// img  -> -d_data[i*sym_w+(width-j)].y
			val = (float) (d_data[i*sym_w+(width-j)].x * d_data[i*sym_w+(width-j)].x + d_data[i*sym_w+(width-j)].y * d_data[i*sym_w+(width-j)].y); // magnitude
		} else {
			// real -> d_data[i*sym_w+j].x
			// img  -> d_data[i*sym_w+j].y
			val = (float) (d_data[i*sym_w+j].x * d_data[i*sym_w+j].x  + d_data[i*sym_w+j].y * d_data[i*sym_w+j].y); // magnitude
		}

		val = val / (float)(size*size); // as magnitude normalise by square of size

		// add to fft_accum
		d_fft_accum[tau_idx * size + pos_offset] = d_fft_accum[tau_idx * size + pos_offset] + val;

	}
}

void analyseChunk() {
	return;
}

void RunDDM() {
	int tau_count = 10;

	// Initialise video parameters
	unsigned int w, h, f;
	h = w = 512;
	f = 60;

	// Initialise buffer parameters
	int buff_frames, chunk_frames;
	buff_frames = 40; // as we are running load / analyse in serial buff size = chunk size
	chunk_frames = 40;

	int num_data = w * h;
	int buff_size = num_data * buff_frames * sizeof(float);
	int work_size = num_data * tau_count * sizeof(float);

	float *d_buffer, *d_fftAccum, *d_absDiff;
	Complex *d_fftWork;

	// allocate device memory for buffer and output
	cudaMalloc((void **) &d_buffer, buff_size);
	cudaMalloc((void **) &d_fftAccum, work_size);
	// TODO allocate the other working arrays

	// At the moment we run each operation in series - can parallelise later

	while (f > chunk_frames) {
		// LoadVideoToBuffer
		// analyseChunk
	}




}



int maiin(int argc, char **argv)
{
	/*
	unsigned int tauCount = 10;

	// Initialise video parameters
	unsigned int w, h, f;
	h = w = 512;
	f = 60;

	// Initialise buffer parameters
	unsigned int buffFrame, chunkFrame;
	buffFrame = 200;
	chunkFrame = 40;

	unsigned int numData = w * h;
	unsigned int buffSize = numData * buffFrame * sizeof(float);
	unsigned int workSize = numData * tauCount * sizeof(float);

	float *d_buffer, *d_fftAccum;

	// allocate device memory for buffer and output
	cudaMalloc((void **) &d_buffer, buffSize);
	cudaMalloc((void **) &d_fftAccum, workSize);

	*/






}

