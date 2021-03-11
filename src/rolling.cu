// Copyright 2021 George Haskell (gh455)

#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <nvToolsExt.h>
#include <stdbool.h>

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <string>

#include "movie_reader.hpp"
#include "verbose.hpp"

#define blockSize_x 16
#define blockSize_y 16

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ void parseBuffer(unsigned char *d_buffer, float *d_parsed, frame_info info, int channel_count, int frame_count) {
	// Must be launched with enough threads to cover input image

	int x = threadIdx.x + blockIdx.x * blockSize_x;
	int y = threadIdx.y + blockIdx.y * blockSize_y;

	if (x < info.out_width && y < info.out_height) {
		for (int frame_idx = 0; frame_idx < frame_count; frame_idx++) {
			d_parsed[frame_idx * info.out_width * info.out_height + y * info.out_width + x] =
					static_cast<float>(d_buffer[channel_count * (frame_idx * info.in_height * info.in_width + (y+info.y_offset) * info.in_width + (x+info.x_offset))]);
		}
	}
}


__global__ void processFFT(cufftComplex *d_fft_frame1, cufftComplex *d_fft_frame2, float *d_fft_accum, int tau_idx, float fft_norm, int width, int height) {
	int x = threadIdx.x + blockIdx.x * blockSize_x;
	int y = threadIdx.y + blockIdx.y * blockSize_y;

	if (x < (width/2 + 1) && y < height) {
		int pos = y * (width/2 + 1) + x;

		cufftComplex val;
		val.x = fft_norm * (d_fft_frame1[pos].x - d_fft_frame2[pos].x);
		val.y = fft_norm * (d_fft_frame1[pos].y - d_fft_frame2[pos].y);

		d_fft_accum[tau_idx * height * (width/2 + 1) + pos] += val.x * val.x + val.y * val.y;
	}
}

__global__ void combine_accum(float *d_A, float *d_B, int width, int height, int tau_count) {
	int x = threadIdx.x + blockIdx.x * blockSize_x;
	int y = threadIdx.y + blockIdx.y * blockSize_y;

	if (x < (width/2 + 1) && y < height) {
		int pos = y * (width/2+1) + x;
		for (int t = 0; t < tau_count; t++)
			d_A[t * height * (width/2+1) + pos] += d_B[t * height * (width/2+1) + pos];
	}
}


void parseChunk(unsigned char *d_raw_in,
				cufftComplex *d_fft_out,
				float *d_workspace,
				int frame_count,
				cufftHandle fft_plan,
				frame_info info,
				int bytes_per_px,
				cudaStream_t stream) {

	dim3 blockDim(blockSize_x, blockSize_y, 1);

	int x_dim = static_cast<int>(ceil(info.out_width / static_cast<float>(blockSize_x)));
	int y_dim = static_cast<int>(ceil(info.out_height / static_cast<float>(blockSize_y)));
	dim3 gridDim(x_dim, y_dim, 1);

	cufftSetStream(fft_plan, stream);

	parseBuffer<<<gridDim, blockDim, 0, stream >>>(d_raw_in, d_workspace, info, bytes_per_px, frame_count);

	// FFT execute
	if ((cufftExecR2C(fft_plan, d_workspace, d_fft_out)) != CUFFT_SUCCESS) {
		fprintf(stderr, "[cuFFT Exec Error]");
	}
	return;
}

void analyseChunk(cufftComplex *d_start,
				  cufftComplex *d_end,
				  float *d_fft_accum,
				  int frame_count,
				  int frame_offset,
				  frame_info frame,
				  int tau_count,
				  int *tau_vector,
				  int repeat_count,
				  int step_count,
				  cudaStream_t stream) {

	int frame_width = frame.out_width;
	int frame_height = frame.out_height;

	int frame_size = frame_height * (frame_width / 2 + 1);
	int px_count = frame_width * frame_height;
	float fft_norm = 1.0 / static_cast<float>(px_count);

	dim3 blockDim(blockSize_x, blockSize_y, 1);
	int x_dim = static_cast<int>(ceil((frame_width / 2 + 1) / static_cast<float>(blockSize_x)));
	int y_dim = static_cast<int>(ceil(frame_height / static_cast<float>(blockSize_y)));
	dim3 gridDim(x_dim, y_dim, 1);

	int tau, frame1_idx, frame2_idx;
	cufftComplex *d_frame1, *d_frame2;

	bool rand_idx; // should we gen random frame indices of just use all
	if (step_count > 1) {
		if (step_count > repeat_count) {
			rand_idx = true;
		} else {
			verbose("Setting step size smaller than repeat count will repeat the same calculation.\n");
			verbose("Setting repeat_count, step_count to 1\n");
			repeat_count = 1;
			step_count = 1;

			rand_idx = false;
		}
	} else if (step_count == 1) {
		if (repeat_count > 1) {
			verbose("Setting step size smaller than repeat count will repeat the same calculation.\n");
			verbose("Setting repeat_count to 1\n");
			repeat_count = 1;
		}
		rand_idx = false;
	}

	for (int repeat = 0; repeat < repeat_count; repeat++) {
		for (int tau_idx = 0; tau_idx < tau_count; tau_idx++) {
			tau = tau_vector[tau_idx];

			frame1_idx = rand_idx ? rand() % (frame_count - tau) + frame_offset : 0; // if not using random index, just pick 0th frame
			frame2_idx = frame1_idx + tau;

			d_frame1 = (frame1_idx < frame_count) ? d_start + frame1_idx * frame_size : d_end + (frame1_idx - frame_count) * frame_size;

			d_frame2 = (frame2_idx < frame_count) ? d_start + frame2_idx * frame_size : d_end + (frame2_idx - frame_count) * frame_size;

			processFFT<<<gridDim, blockDim, 0, stream >>>(d_frame1, d_frame2, d_fft_accum, tau_idx, fft_norm, frame_width, frame_height);
		}
	}
}


void analyseFFTHost(float *d_in, int norm_factor, float *q_vector, int q_count, int *tau_vector, int tau_count, int width, int height) {
    int w = width; int h = height;

	float q_squared[q_count];
	for (int i = 0; i < q_count; i++) {
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
        for (int x = 0; x < w; x++) {
            for (int y = 0; y < h; y++) {
                // Perform manual FFT shift
                shift_x = (x + static_cast<int>(half_w)) % w;
                shift_y = (y + static_cast<int>(half_h)) % h;

                // Distance relative to centre
                shift_x -= half_w;
                shift_y -= half_h;

                r_sqr = shift_x * shift_x + shift_y * shift_y;
                ratio = r_sqr / q_squared[q_idx];

                if (1 <= ratio && ratio <= 1.44) { // we want values from 1.0 * q to 1.2 * q
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

        		for (int x_i = 0; x_i < (w/2+1); x_i++) {
        			for (int y_i = 0; y_i < h; y_i++) {
        				val += d_in[(w/2+1) * h * tau_idx + (w/2+1)*y_i + x_i] * masks[w * h * q_idx + w*y_i + x_i];
        			}
        		}

                // Also should divide by chunk count
        		val *= 2; // account for symmetry
                val /= static_cast<float>(px_count[q_idx]); // could be potential for overflow here
                val /= static_cast<float>(norm_factor);
        	} else {
        		printf("q %d has zero mask pixels\n", q_idx);
        	}

        	iq_tau[q_idx * tau_count + tau_idx] = val;
        }
    }

	// outputting iqtau
    std::ofstream myfile("/home/ghaskell/projects_Git/cuDDM_streams/out/iqt.txt");

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
		printf("I(Q, tau) written to /home/ghaskell/projects_Git/cuDDM_streams/out/iqt.txt");
    } else {
    	std::cout << "Unable to open file" << std::endl;
    	return;
    }
}


void run(std::string filename,
		bool movie_file,
		int out_width,
		int out_height,
		int x_offset,
		int y_offset,
		int total_frames,
		int repeat_count,
		int frame_step,
		int q_count,
		float *q_vector,
		int tau_count,
		int *tau_vector) {

	auto start_time = std::chrono::high_resolution_clock::now();
	verbose("Analysis Start.\n");

	//////////////////
	//  PARAMETERS  //
	//////////////////

	video_info_struct vid_info;
	FILE *moviefile;
	cv::VideoCapture cap;

	int img_width, img_height, bytes_per_px;

	if (movie_file) { // if we have a .moviefile folder we open with own reader
		if (!(moviefile = fopen(filename.c_str(), "rb" ))) {
			fprintf(stderr, "Couldn't open movie file.\n");
			exit(EXIT_FAILURE);
		}
		vid_info = initFile(moviefile);

		img_width = vid_info.size_x;
		img_height = vid_info.size_y;
		bytes_per_px = vid_info.bpp;
	} else {
		cap = cv::VideoCapture(filename);

		img_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
		img_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
		bytes_per_px = 1;
	}

	int buffer_frame_count = 300; // MUST BE MULTIPLE OF 3

	if (buffer_frame_count % 3 != 0) {
		printf("Buffer frame count is not a multiple of chunk frame count.");
	}

	int chunk_frame_count = buffer_frame_count / 3;

	//////////////////////////
	//  MEMORY ALLOCATIONS  //
	//////////////////////////

	size_t buffer_size = sizeof(unsigned char) * buffer_frame_count * img_width * img_height * bytes_per_px; // #frames * size of raw frame
	size_t chunk_size = sizeof(unsigned char) * chunk_frame_count * img_width * img_height * bytes_per_px; // #frames * size of raw frame
	size_t parsed_size = sizeof(cufftComplex) * buffer_frame_count * (out_width/2+1) * out_height;  // #frames * size of raw frame

	unsigned char *d_buffer;
	unsigned char *h_chunk1;
	unsigned char *h_chunk2;
	cufftComplex *d_parsed_fft;

	gpuErrchk(cudaMalloc((void **) &d_buffer, buffer_size));
	gpuErrchk(cudaHostAlloc((void **) &h_chunk1, chunk_size, cudaHostAllocDefault)); // As we do async memory copies - must be pinned memory
	gpuErrchk(cudaHostAlloc((void **) &h_chunk2, chunk_size, cudaHostAllocDefault)); // As we do async memory copies - must be pinned memory
	gpuErrchk(cudaMalloc((void **) &d_parsed_fft, parsed_size));


	// parsed frame workspace

	float *d_workspace1, *d_workspace2;

	size_t workspace_size = sizeof(float) * chunk_frame_count * out_width * out_height;

	gpuErrchk(cudaMalloc((void **) &d_workspace1, workspace_size));
	gpuErrchk(cudaMalloc((void **) &d_workspace2, workspace_size));

	// FFT intensity accumulators for both streams (must also initialise to zero)

	float *d_int_accum1, *d_int_accum2;

	size_t accum_size = sizeof(float) * tau_count * out_height * (out_width/2 + 1);

	gpuErrchk(cudaMalloc((void **) &d_int_accum1, accum_size));
	gpuErrchk(cudaMalloc((void **) &d_int_accum2, accum_size));

	gpuErrchk(cudaMemset(d_int_accum1, 0, accum_size));
	gpuErrchk(cudaMemset(d_int_accum2, 0, accum_size));

	// Host output (does not need to be pinned)

	float *h_out;

	h_out = new float[accum_size];

	////////////////
	//  FFT PLAN  //
	////////////////

	// plan to perform buffer_frame_number C2R,
	cufftHandle plan;

	int batch_count = chunk_frame_count;
	int rank = 2;
    int n[2] = {out_width, out_height};

	int idist = out_width * out_height;
	int odist = out_height * (out_width / 2 + 1);

    int inembed[] = {out_height, out_width};
    int onembed[] = {out_height, out_width / 2 + 1};

    int istride = 1;
    int ostride = 1;

    if (cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batch_count) != CUFFT_SUCCESS) {
    	fprintf(stderr, "cuFFT Error: Plan failure.\n");
    }

	//////////////////
	//  FINAL INIT  //
	//////////////////

	frame_info frame;
	frame.out_width = out_width;
	frame.out_height = out_height;
	frame.in_width = img_width;
	frame.in_height = img_height;
	frame.x_offset = x_offset;
	frame.y_offset = y_offset;

	// iteration counts

	int total_chunks = total_frames / chunk_frame_count;
	int chunks_remaining = total_chunks;

	// streams

	cudaStream_t stream1, stream2;

	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);

	cudaStream_t *stream_current = &stream1;
	cudaStream_t *stream_next = &stream2;

	/////////////////
	//  MAIN LOOP  //
	/////////////////

	float *d_workspace_current = d_workspace1;
	float *d_workspace_next = d_workspace2;

	float *d_accum_current = d_int_accum1;
	float *d_accum_next = d_int_accum2;

	unsigned char *h_chunk_current = h_chunk1;
	unsigned char *h_chunk_next = h_chunk2;

	unsigned char *d_idle  = d_buffer;
	unsigned char *d_ready = d_buffer + 1 * chunk_frame_count * img_width * img_height * bytes_per_px;
	unsigned char *d_used  = d_buffer + 2 * chunk_frame_count * img_width * img_height * bytes_per_px;

	cufftComplex *d_start = d_parsed_fft;
	cufftComplex *d_end   = d_parsed_fft + 1 * chunk_frame_count * (out_width/2+1) * out_height;
	cufftComplex *d_junk  = d_parsed_fft + 2 * chunk_frame_count * (out_width/2+1) * out_height;

	unsigned char *uc_tmp;
	float *f_tmp;
	cufftComplex *complex_tmp;

	// Initialise CPU memory (h_ready / idle)
	if (movie_file) {
		loadMovieToHost(moviefile, h_chunk_next, vid_info, chunk_frame_count);
		loadMovieToHost(moviefile, h_chunk_current, vid_info, chunk_frame_count);
	} else {
		loadCaptureToHost(cap, h_chunk_next, frame, chunk_frame_count);
		loadCaptureToHost(cap, h_chunk_next, frame, chunk_frame_count); // puts chunk data into pinned host memory
	}

	gpuErrchk(cudaMemcpyAsync(d_idle, h_chunk_next, chunk_size, cudaMemcpyHostToDevice, *stream_current));
	parseChunk(d_idle, d_start, d_workspace_current, chunk_frame_count, plan, frame, bytes_per_px, *stream_current);
	gpuErrchk(cudaStreamSynchronize(*stream_current));

	while (chunks_remaining > 0) {

		gpuErrchk(cudaMemcpyAsync(d_ready, h_chunk_current, chunk_size, cudaMemcpyHostToDevice, *stream_current));

		// k(unsigned char *d_raw_in, cufftComplex *d_fft_out, float *d_workspace,
		// int frame_count, cufftHandle fft_plan, frame_info info, int bytes_per_px, cudaStream_t stream)
		parseChunk(d_ready, d_end, d_workspace_current, chunk_frame_count, plan, frame, bytes_per_px, *stream_current);

		for (int frame_offset = 0; frame_offset < chunk_frame_count; frame_offset += frame_step) {
			// Frame #(total_chunks - chunks_remaining) * chunk_frame_count + frame_offset)
			analyseChunk(d_start, d_end, d_accum_current, chunk_frame_count, frame_offset, frame, tau_count, tau_vector, repeat_count, frame_step, *stream_current);
		}

		// prevent overrun
		gpuErrchk(cudaStreamSynchronize(*stream_next));

		if (chunks_remaining > 2) {
			if (movie_file)
				loadMovieToHost(moviefile, h_chunk_next, vid_info, chunk_frame_count);
			else
				loadCaptureToHost(cap, h_chunk_next, frame, chunk_frame_count); // puts chunk data into pinned host memory
			// capture nxt image while GPU is processing cur
		}

		//// PTR swap

		// host
		uc_tmp = h_chunk_current;
		h_chunk_current = h_chunk_next;
		h_chunk_next = uc_tmp;

		uc_tmp = d_used;
		d_used = d_ready;
		d_ready = d_idle;
		d_idle = uc_tmp;

		complex_tmp = d_junk;
		d_junk = d_start;
		d_start = d_end;
		d_end = complex_tmp;

		f_tmp = d_workspace_current;
		d_workspace_current = d_workspace_next;
		d_workspace_next = f_tmp;

		f_tmp = d_accum_current;
		d_accum_current = d_accum_next;
		d_accum_next = f_tmp;

		cudaStream_t *stream_tmp = stream_current;
		stream_current = stream_next;
		stream_next = stream_tmp;

		// End of iteration
		printf("[Chunk complete (%d \\ %d)]\n", total_chunks - chunks_remaining + 1, total_chunks);
		chunks_remaining--;
	}

	// sync device
	cudaDeviceSynchronize();

	// add both intensity accumulators together
	int grid_x = ceil((img_width/2+1) / static_cast<float>(blockSize_x));
	int grid_y = ceil(out_height / static_cast<float>(blockSize_y));

	dim3 blockDim(blockSize_x, blockSize_y, 1);
	dim3 gridDim(grid_x, grid_y, 1);
	combine_accum<<<gridDim, blockDim>>>(d_accum_current, d_accum_next, out_width, out_height, tau_count);

	// copy to host
	gpuErrchk(cudaMemcpy(h_out, d_accum_current, accum_size, cudaMemcpyDeviceToHost));

	cudaFree(h_chunk1);
	cudaFree(h_chunk2);
	cudaFree(d_buffer);
	cudaFree(d_parsed_fft);
	cudaFree(d_int_accum1);
	cudaFree(d_int_accum2);
	cudaFree(d_workspace1);
	cudaFree(d_workspace2);

	cufftDestroy(plan);

	if (movie_file) {
		fclose(moviefile);
	} else {
		cap.release();
	}


	auto end_time = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
	printf("Finished frame analysis, time elapsed %f seconds, %f frames / second)\n", (float)duration/1000000.0, (float) (total_frames * 1000000) / (float) duration);

	int norm_factor = repeat_count * total_chunks * (chunk_frame_count / frame_step);

	analyseFFTHost(h_out, norm_factor, q_vector, q_count, tau_vector, tau_count, out_width, out_height);
}


int main() {
    setVerbose(true);
    verbose("Verbose is on\n");

	int total_frames = 1000;
	int repeats = 1;
	int frame_step = 1;

	int q_count = 25;
	float q_vector[q_count];
	for (int i = 0; i < q_count; i++) {
		q_vector[i] = (i + 2)*3;
	}

	// 64, 128, 256, 512, 1024
	// 1000 frames, 15 tau, 25 q, frame_step

	bool movie_file;
	std::string filename;

	printf("DDM starting\n");

	//filename = "/media/ghaskell/Slow Drive/colloid_vids/red_colloids_0_5um.11Feb2021_15.27.31.movie";
	filename = "/media/ghaskell/Slow Drive/colloid_vids/red_colloids_0_5um11Feb2021152731.mp4";
	movie_file = false;

	int tau_count = 20;
	int tau_vector[tau_count];
	tau_vector[0] = 1;
	tau_vector[1] = 2;
	tau_vector[2] = 3;

	for (int j = 0; j < tau_count-3; j++) {
		tau_vector[j+3] = (j+1)*4;
	}

// std::string filename,
// bool movie_file,
// int out_width,
// int out_height,
// int x_offset,
// int y_offset,
// int total_frames,
// int repeat_count,
// int frame_step,
// int q_count,
// float *q_vector,
// int tau_count,
// int *tau_vector

	run(filename, movie_file, 1024, 1024, 0, 0, total_frames, repeats, frame_step, q_count, q_vector, tau_count, tau_vector);
}

