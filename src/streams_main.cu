//TODO: clean up print statements - switch errors to fprintf
//TODO: probably causes memory leak

#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <chrono>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

using namespace cv;

__global__ void AbsDifference(cufftReal *d_diff, float *d_frame1, float *d_frame2, int width, int height) {
	int x = threadIdx.x + blockIdx.x * 32;
	int y = threadIdx.y + blockIdx.y * 32;

	if (x <= width-1 && y <= height-1) {
		int pos_offset = y * width + x;
		d_diff[pos_offset] = abs(d_frame1[pos_offset] - d_frame2[pos_offset]);
	}
	return;
}


__global__ void processFFT(cufftComplex *d_data, float *d_fft, int tau_idx, int width, int height) {
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
			mag = cuCabsf(d_data[i*sym_w+(width-j)]) / (float)size;

		} else {
			// real -> d_data[i*sym_w+j].x
			// img  -> d_data[i*sym_w+j].y
			mag = cuCabsf(d_data[i*sym_w+j]) / (float)size;
		}

		// add to fft_accum
		d_fft[tau_idx * size + pos_offset] += mag*mag;
	}
}


bool LoadVideoToBuffer(float *h_ptr, int frame_count, VideoCapture cap, int w, int h) {
	//printf("load video (%d frames) (w: %d, h: %d)\n", frame_count, w, h);

	// No bounds check! assume that w, h smaller than mat
	int num_elements = w * h;

	Mat input_img; //, grayscale_img;

	// There is some problems with the image type we are using - though some effort was put into switching to a
	// more generic image format, more thought is required therefore switch to just dealing with 3 channel uchars
	// look at http://ninghang.blogspot.com/2012/11/list-of-mat-type-in-opencv.html and
	// https://docs.opencv.org/3.4/d3/d63/classcv_1_1Mat.html#aa5d20fc86d41d59e4d71ae93daee9726 for more info.


	for (int frame_idx = 0; frame_idx < frame_count; frame_idx++) {
		//std::cout << "Loaded frame " << frame_idx << std::endl;

		cap >> input_img;

		if (input_img.empty()) {
			fprintf(stderr,"Video frame is empty");
			return false;
		}

		//input_img.convertTo(grayscale_img, CV_32FC1); // covert to grayscale image

		if (input_img.type() != 16) {
			std::cout << "Non standard image format detected, may cause unexpected behaviour, image type : " << input_img.type() << std::endl;
			return false;
		}

	    //imshow("Input", input_img);
	    //waitKey(0);

		int cols = input_img.cols, rows = input_img.rows;

		if (w > cols || h > rows) {
			std::cout << "Issue: specified width / height > cols / rows." << std::endl;
		}

		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				// Using img.at<>8
				h_ptr[frame_idx * num_elements + y * w + x] =  (float) input_img.data[((input_img.step)/input_img.elemSize1())* y + input_img.channels() * x];
			}
		}
	}
	return true;
}


void processChunk(cudaStream_t stream, float *d_ptr, int frame_count, float *d_out, int *tau_vector, int tau_count, int width, int height, int repeat_count = 20, float *debug_buff=NULL) {
	// debug_buffer is a width * height *sizeof(float) buffer which can be printed
	//	if (debug_buff != NULL) {
	//		cudaMemcpy(debug_buff, <device ptr>, width*height*sizeof(float), cudaMemcpyDeviceToHost);
	//		return;
	//	}
	// d_out size: tau_count * width * height * sizeof(float)

	int w = width;
	int h = height;

	//printf("chunk analysis (%d frames).\n", frame_count);

	// Max 1024 (32 x 32) threads per block hence multiple blocks to operate on a frame
	// Max number of thread blocks is 65536)

	dim3 blockDim(32, 32, 1);
	dim3 gridDim((int) ceil(width/32.0), (int) ceil(height/32.0), 1);

	if ((int) ceil(width/32.0) * (int) ceil(height/32.0) > 65536) {
		fprintf(stderr, "Image too big, not enough thread blocks (%d).\n", (int) ceil(width/32.0) * (int) ceil(height/32.0));
	}


	cufftReal *d_local_absdiff;
	cudaMalloc((void **) &d_local_absdiff, w * h * sizeof(cufftReal));

	cufftComplex *d_local_fft;
	cudaMalloc((void **) &d_local_fft, w * (h / 2 + 1) * sizeof(cufftComplex));

	// cuFFT plan
	cufftHandle plan;
	if ((cufftPlan2d(&plan, w, h, CUFFT_R2C)) != CUFFT_SUCCESS) {
		fprintf(stderr, "cuFFT Error: Plan failure.\n");
	}
	cufftSetStream(plan, stream);


	// Main loop
	int tau, idx1, idx2;
	float *d_frame1, *d_frame2;

	for (int repeat = 0; repeat < repeat_count; repeat++) {
		for (int tau_idx = 0; tau_idx < tau_count; tau_idx++) {
			tau = tau_vector[tau_idx];

			idx1 = rand() % (frame_count - tau);
			idx2 = idx1 + tau;
			//std::cout << "tau: " << tau << " idxs: " << idx1 << ", " << idx2 << std::endl;

			d_frame1 = d_ptr + (idx1 * w * h);	// float pointer to frame 1
			d_frame2 = d_ptr + (idx2 * w * h);

			AbsDifference<<<gridDim, blockDim, 0, stream >>>(d_local_absdiff, d_frame1, d_frame2, w, h); // find absolute difference

			//FFT execute
			if ((cufftExecR2C(plan, d_local_absdiff, d_local_fft)) != CUFFT_SUCCESS) {
				std::cout << "cuFFT Exec Error\n" << std::endl;
			}

			processFFT<<<gridDim, blockDim, 0, stream>>>(d_local_fft, d_out, tau_idx, w, h); // process FFT (i.e. normalise and add to accumulator)
		}
	}
	cudaFree(d_local_absdiff); cudaFree(d_local_fft);
	cufftDestroy(plan);

	return;
}


int main(){
	for (int x = 0; x < 5; x++) {
		auto t1 = std::chrono::high_resolution_clock::now();

		int w = 1024;
		int h = 1024;
		int total_frames = 500;
		int buffer_frames = 50;
		int tau_count = 15;
		int tau_vector [tau_count] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
		VideoCapture cap("/home/ghaskell/projects_Git/cuDDM/data/test.mp4");

		// Initialisation
		int iterations = total_frames / buffer_frames;
		bool read_ok;

		float *h_buffer1, *h_buffer2;
		float *d_buffer1, *d_buffer2;
		float *d_out;

		int buffer_size = sizeof(float) * buffer_frames * w * h;

		gpuErrchk(cudaHostAlloc((void **) &h_buffer1, buffer_size, cudaHostAllocDefault));
		gpuErrchk(cudaHostAlloc((void **) &h_buffer2, buffer_size, cudaHostAllocDefault));
		gpuErrchk(cudaMalloc((void **) &d_buffer1, buffer_size));
		gpuErrchk(cudaMalloc((void **) &d_buffer2, buffer_size));
		gpuErrchk(cudaMalloc((void **) &d_out, sizeof(float) * tau_count* w * h));

		cudaStream_t stream1, stream2;
		cudaStreamCreate(&stream1); cudaStreamCreate(&stream2);

		float *d_data = d_buffer1;
		float *h_data = h_buffer1;

		float *d_next = d_buffer2;
		float *h_next = h_buffer2;

		cudaStream_t *work_stream = &stream1;
		cudaStream_t *next_stream = &stream2;

		read_ok = LoadVideoToBuffer(h_data, buffer_frames, cap, w, h); // puts chunk data into pinned host memory

		while (read_ok && iterations > 0) {
			gpuErrchk(cudaMemcpyAsync(d_data, h_data, buffer_size, cudaMemcpyHostToDevice, *work_stream)); // copy buffer to device

			// PROCESS FRAME - use work stream
			processChunk(*work_stream, d_data, buffer_frames, d_out, tau_vector, tau_count, w, h);

			gpuErrchk(cudaStreamSynchronize(*next_stream)); // prevent overrun

			read_ok = LoadVideoToBuffer(h_next, buffer_frames, cap, w, h);

			// Swap working and secondary streams
			float *tmp = h_data;
			h_data = h_next;
			h_next = tmp;

			tmp = d_data;
			d_data = d_next;
			d_next = tmp;

			cudaStream_t *st_tmp = work_stream;
			work_stream = next_stream;
			next_stream = st_tmp;

			printf("Interation complete (Iterations = %d))\n", iterations);
			iterations--;

		}
		printf("Done\n");

		auto t2 = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
		std::cout << (float)duration/1000000.0 << std::endl;

	}
}
