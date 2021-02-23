//TODO: clean up print statements - switch errors to fprintf
//TODO: probably causes memory leak

#include "movie_reader.h"

#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <nvToolsExt.h>

#define blockSize_x 16
#define blockSize_y 16

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

__global__ void AbsDifference(cufftReal *d_diff, unsigned char *d_frame1, unsigned char *d_frame2,
		int img_width, int img_height,
		int out_width, int out_height,
		int bytes_per_px=1)
	{
	// If more than one byte per pixel, then we just take first channel
	int x = threadIdx.x + blockIdx.x * blockSize_x;
	int y = threadIdx.y + blockIdx.y * blockSize_y;

	if (x <= out_width-1 && y <= out_height-1) {
		d_diff[y * out_width + x] = (cufftReal)abs(d_frame1[(y * img_width + x) * bytes_per_px] - d_frame2[(y * img_width + x)  * bytes_per_px]) ;
	}
	return;
}


__global__ void processFFT(cufftComplex *d_data, float *d_fft, int tau_idx, int width, int height) {
	// Takes output of cuFFT R2C operation, normalises it (i.e. divides by px count), takes the magnitude and adds it to the accum_array
	//TODO look at cuff abs to make better
	int size = (width * height);
	float size_recip = 1.0 / (float)(size);

	int j = threadIdx.x + blockIdx.x * blockSize_x;
	int i = threadIdx.y + blockIdx.y * blockSize_y;

	float inten;
	if (j <= width-1 && i <= height-1) {
		int pos_offset = i * width + j;
		int sym_w = width / 2 + 1; // to deal with complex (hermitian) symmetry
		cufftComplex val;

		if (j >= sym_w) {
			// real ->  d_data[i*sym_w+(width-j)].x
			// img  -> -d_data[i*sym_w+(width-j)].y
			val =  d_data[i*sym_w+(width-j)];
			inten = (size_recip * val.x) * (size_recip * val.x) + (size_recip * val.y) * (size_recip * val.y);

		} else {
			// real -> d_data[i*sym_w+j].x
			// img  -> d_data[i*sym_w+j].y
			val = d_data[i*sym_w+j];
			inten = (size_recip * val.x) * (size_recip * val.x) + (size_recip * val.y) * (size_recip * val.y);
		}

		// add to fft_accum
		d_fft[tau_idx * size + pos_offset] += inten;
	}
}


bool LoadVideoToBuffer(float *h_ptr, int frame_count, VideoCapture cap, int w, int h) {
	nvtxRangePush(__FUNCTION__); // to track video loading times in nvvp

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

    nvtxRangePop();
	return true;
}


void processChunk(cudaStream_t stream, unsigned char *d_ptr,
		int frame_count,
		float *d_out,
		int *tau_vector,
		int tau_count,
		cufftReal *d_abs_workspace,
		cufftComplex *d_fft_workspace,
		cufftHandle plan,
		int img_width, int img_height,
		int out_width, int out_height,
		int repeat_count = 50, float *debug_buff=NULL) {

	// debug_buffer is a width * height *sizeof(float) buffer which can be printed
	//	if (debug_buff != NULL) {
	//		cudaMemcpy(debug_buff, <device ptr>, width*height*sizeof(float), cudaMemcpyDeviceToHost);
	//		return;
	//	}
	// d_out size: tau_count * width * height * sizeof(float)

	//printf("chunk analysis (%d frames).\n", frame_count);

	// Max 1024 (32 x 32) threads per block hence multiple blocks to operate on a frame
	// Max number of thread blocks is 65536)

	dim3 blockDim(blockSize_x, blockSize_y, 1);
	int grid_x = (int) ceil(out_width/(float)blockSize_x);
	int grid_y = (int) ceil(out_height/(float)blockSize_y);

	dim3 gridDim(grid_x, grid_y, 1);

	if (gridDim.x * gridDim.y * gridDim.z > 65536) {
		fprintf(stderr, "Image too big, not enough thread blocks (%d).\n", gridDim.x * gridDim.y * gridDim.z);
	}

	cufftSetStream(plan, stream);

	// Main loop
	int tau, frame_idx;
	unsigned char *d_frame1, *d_frame2;
	cufftComplex *d_local_fft;
	cufftReal *d_local_absdiff;

	for (int repeat = 0; repeat < repeat_count; repeat++) {
		for (int tau_idx = 0; tau_idx < tau_count; tau_idx++) {
			tau = tau_vector[tau_idx];

			d_local_fft = d_fft_workspace + (tau_idx * out_width * (out_height / 2 + 1));
			d_local_absdiff = d_abs_workspace + (tau_idx * out_width * out_height);

			frame_idx = rand() % (frame_count - tau);

			//std::cout << "tau: " << tau << " idxs: " << idx1 << ", " << idx2 << std::endl;

			d_frame1 = d_ptr + (frame_idx * img_width * img_height);	// float pointer to frame 1
			d_frame2 = d_ptr + ((frame_idx+tau) * img_width * img_height);

			AbsDifference<<<gridDim, blockDim, 0, stream >>>(d_local_absdiff, d_frame1, d_frame2, img_width, img_height, out_width, out_height); // find absolute difference

			//FFT execute
			if ((cufftExecR2C(plan, d_local_absdiff, d_local_fft)) != CUFFT_SUCCESS) {
				std::cout << "cuFFT Exec Error\n" << std::endl;
			}

			processFFT<<<gridDim, blockDim, 0, stream>>>(d_local_fft, d_out, tau_idx, out_width, out_height); // process FFT (i.e. normalise and add to accumulator)

		}
	}

	return;
}


void HARDCODEanalyseFFTHost(float *d_in, int norm_factor, int *tau_vector, int tau_count, int width, int height) {
    int w = width; int h = height;

	// Generate q - vectors - Hard Coded
	int q_count = 25;

	float q_squared[q_count];
	float q_vector[q_count];

	for (int i = 0; i < q_count; i++) {
		//std::cout << 50 * ((float)i /20.0) << std::endl;
		q_vector[i] = 75 * ((float)(i+1) /q_count);
		q_squared[i] = q_vector[i] * q_vector[i];
	}

	// Generate masks
    int *px_count = new int[q_count](); // () initialises to zero
    float *masks = new float[w * h * q_count];

    float half_w, half_h;
    half_h = height / 2.0;
    half_w = width / 2.0;
    float r_sqr, ratio;

    // First Generate the radius masks
    int shift_x, shift_y;
    for (int q_idx = 0; q_idx < q_count; q_idx++) {
        for (int x = 0; x < w; x++)
        {
            for (int y = 0; y < h; y++)
            {
                // Perform manual FFT shift
                shift_x = (x + (int)half_w) % w;
                shift_y = (y + (int)half_h) % h;

                // Distance relative to centre
                shift_x -= half_w;
                shift_y -= half_h;

                r_sqr = shift_x * shift_x + shift_y * shift_y;
                ratio = r_sqr / q_squared[q_idx];

                if (1 <= ratio && ratio <= 1.69) { // we want values from 1.0 * q to 1.2 * q
                    masks[q_idx*w*h + y*w + x] = 1.0;
                    px_count[q_idx] += 1;
                } else {
                    masks[q_idx*w*h + y*w + x] = 0.0;
                }
            }
        }
    }


    // Start analysis
    float val;
	float * iq_tau = new float[tau_count * q_count]();

    for (int tau_idx = 0; tau_idx < tau_count; tau_idx++) {

        for (int q_idx = 0; q_idx < q_count; q_idx++) {
        	val = 0;

        	if (px_count[q_idx] != 0) { // If the mask has no values iq_tau must be zero

        		for (int i = 0; i < w*h; i++) { 	// iterate through all pixels
                	val += d_in[w * h * tau_idx + i] * masks[w * h * q_idx + i] ;
                }
                // Also should divide by chunk count
                val /= (float)px_count[q_idx]; // could be potential for overflow here
                val /= (float)norm_factor;
        	} else {
        		printf("q %d has zero mask pixels\n", q_idx);
        	}

        	iq_tau[q_idx * tau_count + tau_idx] = val;
        }
    }

	// outputting iqtau
    std::ofstream myfile("/home/ghaskell/projects_Git/cuDDM_streams/data/iqt.txt");

    if (myfile.is_open()) {
    	for (int i = 0; i < q_count; i++) {
    		myfile << q_vector[i] << " ";
    	}
		myfile << "\n";
    	for (int i = 0; i < tau_count; i++) {
    		myfile << tau_vector[i] << " ";
    	}
		myfile << "\n";

		for (int q_idx = 0; q_idx < q_count; q_idx++) {
	    	for (int t_idx = 0; t_idx < tau_count; t_idx++) {
	    		myfile << iq_tau[q_idx * tau_count + t_idx] << " ";
	    	}
			myfile << "\n";
		}

		myfile.close();
		printf("outputted to /home/ghaskell/projects_Git/cuDDM_streams/data/iqt.txt");
    } else {
    	std::cout << "Unable to open file" << std::endl;
    	return;
    }
}



int main(){
	auto start_time = std::chrono::high_resolution_clock::now();

	FILE *moviefile;
	if ( !(moviefile = fopen("/media/ghaskell/Slow Drive/colloid_vids/blue_colloids_1um11feb160000.movie", "rb" ))) {
		fprintf(stderr, "Couldn't open movie file.\n" );
		exit( EXIT_FAILURE );
	}

	video_info_struct vid_info = initFile(moviefile);
	int img_width = vid_info.size_x;
	int img_height = vid_info.size_y;
	int bpp = vid_info.bpp;

	int out_width = 1024;
	int out_height = 1024;

	int buffer_frames = 25;
	int total_frames = 2000;
	int tau_count = 15;
	int tau_vector [tau_count] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
	int repeat_count = 50;

	// Initialisation
	int iterations = total_frames / buffer_frames;
	int kiterations = iterations;
	bool read_ok = true;

	unsigned char *h_buffer1, *h_buffer2;
	unsigned char *d_buffer1, *d_buffer2;
	float *h_out, *d_out;

	int buffer_size = sizeof(unsigned char) * buffer_frames * img_width * img_height * bpp; // size of each raw image frame * number of frames

	gpuErrchk(cudaHostAlloc((void **) &h_buffer1, buffer_size, cudaHostAllocDefault));
	gpuErrchk(cudaHostAlloc((void **) &h_buffer2, buffer_size, cudaHostAllocDefault));

	gpuErrchk(cudaMalloc((void **) &d_buffer1, buffer_size));
	gpuErrchk(cudaMalloc((void **) &d_buffer2, buffer_size));

	gpuErrchk(cudaMalloc((void **) &d_out, sizeof(float) * tau_count * out_width * out_height));
	h_out = new float[tau_count * out_width * out_height];

	unsigned char *d_data = d_buffer1;
	unsigned char *h_data = h_buffer1;

	unsigned char *d_next = d_buffer2;
	unsigned char *h_next = h_buffer2;

	cudaStream_t stream1, stream2;
	cudaStreamCreate(&stream1); cudaStreamCreate(&stream2);
	cudaStream_t *work_stream = &stream1;
	cudaStream_t *next_stream = &stream2;

	// Workspace
	cufftReal *d_abs_workspace1;
	cufftReal *d_abs_workspace2;
	cudaMalloc((void **) &d_abs_workspace1, tau_count * out_width * out_height * sizeof(cufftReal));
	cudaMalloc((void **) &d_abs_workspace2, tau_count * out_width * out_height * sizeof(cufftReal));

	cufftComplex *d_fft_workspace1;
	cufftComplex *d_fft_workspace2;
	cudaMalloc((void **) &d_fft_workspace1, tau_count * out_width * (out_height / 2 + 1) * sizeof(cufftComplex));
	cudaMalloc((void **) &d_fft_workspace2, tau_count * out_width * (out_height / 2 + 1) * sizeof(cufftComplex));

	cufftComplex *d_fft_current = d_fft_workspace1;
	cufftComplex *d_fft_next = d_fft_workspace2;

	cufftReal *d_abs_current = d_abs_workspace1;
	cufftReal *d_abs_next = d_abs_workspace1;

	// cuFFT plan
	cufftHandle plan;
	if ((cufftPlan2d(&plan, out_width, out_height, CUFFT_R2C)) != CUFFT_SUCCESS) {
		fprintf(stderr, "cuFFT Error: Plan failure.\n");
	}

	// Main loop

	//read_ok = LoadVideoToBuffer(h_data, buffer_frames, cap, w, h); // puts chunk data into pinned host memory

	nvtxRangePush("Loading frames to file");
	loadFileToHost(moviefile, h_data, vid_info, buffer_frames);
    nvtxRangePop();

	while (read_ok && iterations > 0) {
		gpuErrchk(cudaMemcpyAsync(d_data, h_data, buffer_size, cudaMemcpyHostToDevice, *work_stream)); // copy buffer to device

		// PROCESS FRAME - use work stream
		processChunk(*work_stream, d_data, buffer_frames, d_out, tau_vector, tau_count, d_abs_current, d_fft_current, plan, img_width, img_height, out_width, out_height, repeat_count); // repeat count optional

		gpuErrchk(cudaStreamSynchronize(*next_stream)); // prevent overrun

		//read_ok = LoadVideoToBuffer(h_next, buffer_frames, cap, w, h); // load next while GPU processing current
		nvtxRangePush("Loading frames to file");
		loadFileToHost(moviefile, h_data, vid_info, buffer_frames);
	    nvtxRangePop();

		// Swap working and secondary streams
		unsigned char *tmp = h_data;
		h_data = h_next;
		h_next = tmp;

		tmp = d_data;
		d_data = d_next;
		d_next = tmp;

		cudaStream_t *st_tmp = work_stream;
		work_stream = next_stream;
		next_stream = st_tmp;

		cufftComplex *fft_tmp = d_fft_current;
		d_fft_current = d_fft_next;
		d_fft_next = fft_tmp;

		cufftReal *abs_tmp = d_abs_current;
		d_abs_current = d_abs_next;
		d_abs_next = abs_tmp;

		printf("chunk complete (%d \\ %d)\n", kiterations- iterations + 1, kiterations);
		iterations--;

	}

	gpuErrchk(cudaMemcpy(h_out, d_out, sizeof(float) * tau_count* out_width * out_height, cudaMemcpyDeviceToHost));

	cudaFree(h_buffer1); cudaFree(h_buffer2);
	cudaFree(d_buffer1); cudaFree(d_buffer2);
	cudaFree(d_out);
	cudaFree(d_abs_workspace1); cudaFree(d_abs_workspace2);
	cudaFree(d_fft_workspace1); cudaFree(d_fft_workspace2);
	cufftDestroy(plan);

	auto t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - start_time ).count();
	std::cout << "END (time elapsed: " << (float)duration/1000000.0 << " seconds.)"<< std::endl;

	HARDCODEanalyseFFTHost(h_out, repeat_count*kiterations, tau_vector, tau_count, out_width, out_height);




}
