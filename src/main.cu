#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace cv;

struct VideoInfo {
    int width;
    int height;
    int frame_count;
};

void LoadVideoToBuffer(float *d_ptr, int frame_count, VideoCapture cap, int w, int h) {
	std::cout << "Load frame " << frame_count << " (w: " <<  w << " h: " << h << ")" << std::endl;

	// No bounds check! assume that w, h smaller than mat
	int num_elements = w * h;

	Mat input_img, grayscale_img;
	float *h_ptr = new float[num_elements * frame_count];

	for (int frame_idx = 0; frame_idx < frame_count; frame_idx++) {
		//std::cout << "Loaded frame " << frame_idx << std::endl;

		cap >> input_img;

		if (input_img.empty()) {
			std::cout << "Loaded frame is empty." << std::endl;
		}

		input_img.convertTo(grayscale_img, CV_32FC1); // covert to grayscale image

		int cols = grayscale_img.cols, rows = grayscale_img.rows;

		if (w > cols || h > rows) {
			std::cout << "Issue: specified width / height > cols / rows." << std::endl;
		}

		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				h_ptr[frame_idx * num_elements + y * w + x] = grayscale_img.at<float>(y,x) / 255.0;
			}
		}
	}
	cudaMemcpy(d_ptr, h_ptr, num_elements * frame_count * sizeof(float), cudaMemcpyHostToDevice);
}

__global__ void AbsDifference(float *d_buffer, cufftReal *d_diff, int frame1, int frame2, int tau_idx, int width, int height) {
	int size = width * height;

	int x = threadIdx.x + blockIdx.x * 32;
	int y = threadIdx.y + blockIdx.y * 32;

	if (x <= width-1 && y <= height-1) {
		int pos_offset = y * width + x;
		d_diff[tau_idx * size + pos_offset] = (cufftReal) abs(d_buffer[frame1 * size + pos_offset] - d_buffer[frame2 * size + pos_offset]);
	}

	return;
}

__global__ void processFFT(cufftComplex *d_data, float *d_fft_accum, int tau_idx, int width, int height) {
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

void analyseChunk(float* d_buffer, float *d_fft_accum, int tau_count, int chunk_frame_count, int *tau_vector, int width, int height) {
	std::cout << "Analyse Chunk" << std::endl;
	cufftComplex *d_fft_local;
	cudaMalloc((void **) &d_fft_local, width * (height/2 + 1) * tau_count * sizeof(cufftComplex));

	cufftReal *d_diff_local;
	cudaMalloc((void **) &d_diff_local, width * height * tau_count * sizeof(cufftReal));

	dim3 blockDim(32, 32, 1);
	dim3 gridDim((int)ceil(width/32.0), (int)ceil(height/32.0), 1);

	int tau, frame1, frame2;
	for (int tau_idx = 0; tau_idx < tau_count; tau_idx++) {
        tau = tau_vector[tau_idx];

        frame1 = rand() % (chunk_frame_count - tau);
        frame2 = frame1 + tau;

		std::cout << " Abs Diff" << std::endl;
        AbsDifference<<<gridDim, blockDim>>>(d_buffer, d_diff_local, frame1, frame2, tau_idx, width, height);


        // FFT
		std::cout << " FFt Diff" << std::endl;
        cufftHandle plan;
        if ((cufftPlan2d(&plan, height, width, CUFFT_R2C)) != CUFFT_SUCCESS) {
        	std::cout << "cufft plan error" << std::endl;
        }
        if ((cufftExecR2C(plan, (cufftReal*)d_diff_local, (cufftComplex*)d_fft_local)) != CUFFT_SUCCESS) {
        	std::cout << "cufft exec error" << std::endl;
        }

		std::cout << " Process FFt" << std::endl;
        processFFT<<<gridDim, blockDim>>>(d_fft_local, d_fft_accum, tau_idx, width, height);
	}
}

void RunDDM(float *out, VideoCapture cap, int width, int height, int frame_count, int tau_count, int* tau_vector) {
	std::cout<<"Start"<<std::endl;

	// Initialise buffer parameters
	int buff_frames, chunk_frames;
	buff_frames = 20; // as we are running load / analyse in serial buff size = chunk size
	chunk_frames = 20;

	int num_data = width * height;
	int buff_size = num_data * buff_frames * sizeof(float);
	int work_size = num_data * tau_count * sizeof(float);

	float *d_buffer, *d_fftAccum;

	// allocate device memory for buffer and output
	cudaMalloc((void **) &d_buffer, buff_size);
	cudaMalloc((void **) &d_fftAccum, work_size);

	// At the moment we run each operation in series - can parallelise later

	while (frame_count >= chunk_frames) {
		std::cout << frame_count << " Frames left" << std::endl;
		LoadVideoToBuffer(d_buffer, chunk_frames, cap, width, height);
		analyseChunk(d_buffer, d_fftAccum, tau_count, chunk_frames, tau_vector, width, height);
		frame_count -= chunk_frames;
	}

	cudaMemcpy(out, d_fftAccum, work_size, cudaMemcpyDeviceToHost);
	std::cout<<"Done"<<std::endl;
}



int main(int argc, char **argv)
{
	VideoCapture cap("/home/ghaskell/projects_Git/cuDDM/data/colloid_0.2um_vid.mp4");

	int tau_count = 10;
	int tau_vector [14] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
	int width = 128;
	int height = 128;
	int frame_count = 400;

	float * out = new float [width * height * tau_count];
	//RunDDM(float *out, VideoCapture cap, int width, int height, int frame_count, int tau_count, int* tau_vector)
	RunDDM(out, cap, width, height, frame_count, tau_count, &tau_vector[0]);

	std::ofstream myfile("/home/ghaskell/projects_Git/cuDDM/data/data2.txt");

	if (myfile.is_open()) {
		for (int t = 0; t < tau_count; t++) {
			for (int x = 0; x < width*height; x++) {
				myfile << out[t*width*height+ x] <<" ";
			}
			myfile << std::endl;
		}
	}

	myfile.close();

	std::cout << "DONE" << std::endl;

}


