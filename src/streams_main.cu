#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace cv;

//bool validate_image(unsigned char *img) {
//  validated_frame++;
//  for (int i = 0; i < IMGSZ; i++) if (img[i] != validated_frame) {printf("image validation failed at %d, was: %d, should be: %d\n",i, img[i], validated_frame); return false;}
//  return true;
//}

void CUDART_CB my_callback(cudaStream_t stream, cudaError_t status, void* data) {
    validate_image((unsigned char *)data);
}

//
//bool capture_image(unsigned char *img){
//
//  for (int i = 0; i < IMGSZ; i++) img[i] = cur_frame;
//  if (++cur_frame == NUM_FRAMES) {cur_frame--; return true;}
//  return false;
//}


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


void LoadVideoToBuffer(float *h_ptr, int frame_count, VideoCapture cap, int w, int h) {
	std::cout << "Load frame " << frame_count << " (w: " <<  w << " h: " << h << ")" << std::endl;

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
			std::cout << "Loaded frame is empty." << std::endl;
		}

		//input_img.convertTo(grayscale_img, CV_32FC1); // covert to grayscale image

		if (input_img.type() != 16) {
			std::cout << "Non standard image format detected, may cause unexpected behaviour, image type : " << input_img.type() << std::endl;
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


void processChunk(cudaStream_t stream, float *d_ptr, int frame_count, float *d_out, int *tau_vector, int tau_count, int width, int height, float *debug_buff=NULL) {
	// debug_buffer is a width * height *sizeof(float) buffer which can be printed
	//	if (debug_buff != NULL) {
	//		cudaMemcpy(debug_buff, <device ptr>, width*height*sizeof(float), cudaMemcpyDeviceToHost);
	//		return;
	//	}
	// d_out size: tau_count * width * height * sizeof(float)

	int w = width;
	int h = height;

	printf("Chunk Analysis Start (%d frames)", frame_count);

	// Max 1024 (32 x 32) threads per block hence multiple blocks to operate on a frame
	// Max number of thread blocks is 65536)

	dim3 blockDim(32, 32, 1);
	dim3 gridDim((int) ceil(width/32.0), (int) ceil(height/32.0), 1);

	if ((int) ceil(width/32.0) * (int) ceil(height/32.0)) {
		fprintf(stderr, "Image too big, not enough thread blocks (%d)", (int) ceil(width/32.0) * (int) ceil(height/32.0));
	}


	cufftReal *d_local_absdiff;
	cudaMalloc((void **) &d_local_absdiff, w * h * sizeof(cufftReal));

	cufftComplex *d_local_fft;
	cudaMalloc((void **) &d_local_fft, w * (h / 2 + 1) * sizeof(cufftComplex));

	// cuFFT plan
	cufftHandle plan;
	if ((cufftPlan2d(&plan, w, h, CUFFT_R2C)) != CUFFT_SUCCESS) {
		fprintf(stderr, "cuFFT Error: Plan failure");
	}
	cufftSetStream(plan, stream);


	// Main loop
	int tau, idx1, idx2;
	float *d_frame1, *d_frame2;

	for (int repeats = 0; repeats < 20; repeats++) {
		for (int tau_idx = 0; tau_idx < tau_count; tau_idx++) {
			tau = tau_vector[tau_idx];

			idx1 = rand() % (frame_count - tau);
			idx2 = idx1 + tau;
			// std::cout << "tau: " << tau << " idxs: " << idx1 << ", " << idx2 << std::endl;

			d_frame1 = d_ptr + (idx1 * w * h);	// float pointer to frame 1
			d_frame2 = d_ptr + (idx2 * w * h);

			AbsDifference<<<gridDim, blockDim, 0, stream >>>(d_local_absdiff, d_frame1, d_frame2, w, h); // find absolute difference

			// FFT execute
			if ((cufftExecR2C(plan, d_local_absdiff, d_local_fft)) != CUFFT_SUCCESS) {
				std::cout << "cuFFT Exec Error" << std::endl;
			}

			processFFT<<<gridDim, blockDim, 0, stream>>>(d_local_fft, d_out, tau_idx, w, h); // process FFT (i.e. normalise and add to accumulator)
		}
	}
	cudaFree(d_local_absdiff); cudaFree(d_local_fft);
	cufftDestroy(plan);

	return;
}


bool load_chunk(float *t) {
	false;
}


int main(){
	int w = 256;
	int h = 245;
	int num_frames_buffer = 200;
	int tau_count = 15;


	// Initialisation
	bool done = false;

	float *h_buffer1, *h_buffer2;
	float *d_buffer1, *d_buffer2;

	size_t buffer_size = sizeof(float) * num_frames_buffer * w * h;

	cudaHostAlloc(&h_buffer1, buffer_size, cudaHostAllocDefault);
	cudaHostAlloc(&h_buffer2, buffer_size, cudaHostAllocDefault);
	cudaMalloc(&d_buffer1, buffer_size);
	cudaMalloc(&d_buffer2, buffer_size);

	cudaStream_t stream1, stream2;
	float *d_data = d_buffer1;
	float *h_data = h_buffer1;

	float *d_next = d_buffer2;
	float *h_next = h_buffer2;

	cudaStream_t *work_stream = &stream1;
	cudaStream_t *next_stream = &stream2;

	done = load_chunk(h_data); // puts chunk data into pinned host memory

	while (!done) {
		cudaMemcpyAsync(d_data, h_data, buffer_size, cudaMemcpyHostToDevice, *work_stream); // copy buffer to device

		// PROCESS FRAME - use work stream
		//
		// img_proc_kernel<<<nBLK, nTPB, 0, *curst>>>(d_cur); // process frame

	    cudaStreamAddCallback(*work_stream, &my_callback, (void *)h_data, 0);
	    cudaStreamSynchronize(*next_stream); // prevent overrun

		done = load_chunk(h_next); // capture nxt image while GPU is processing cur

	    float *tmp = h_data;
	    h_data = h_next;
	    h_next = tmp;   // ping - pong

	    tmp = d_data;
	    d_data = d_next;
	    d_next = tmp;

	    cudaStream_t *st_tmp = work_stream;
	    work_stream = next_stream;
	    next_stream = st_tmp;

	}
}
