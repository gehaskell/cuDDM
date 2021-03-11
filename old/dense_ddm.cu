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

struct frame_info {
	int out_width;
	int out_height;
	int in_width;
	int in_height;
	int x_offset = 0;
	int y_offset = 0;
};

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

using namespace cv;

__global__ void parseBuffer(unsigned char *d_buffer, float *d_parsed, frame_info info, int channel_count, int frame_count) {
	// Must be launched with enough threads to cover input image

	int x = threadIdx.x + blockIdx.x * blockSize_x;
	int y = threadIdx.y + blockIdx.y * blockSize_y;


//	if (info.out_width + info.x_offset > info.in_width || info.out_height + info.y_offset > info.in_height) {
//		printf("[Image Parse Error] x / y offset too large./n");
//	}

	if (x < info.out_width && y < info.out_height) {
		for (int frame_idx = 0; frame_idx < frame_count; frame_idx++) {
			d_parsed[frame_idx * info.out_width * info.out_height + y * info.out_width + x] =
					(float) d_buffer[channel_count * (frame_idx * info.in_height * info.in_width + (y+info.y_offset) * info.in_width + (x+info.x_offset))];
		}
	}
}


__global__ void processFFT(cufftComplex *d_fft_frame1, cufftComplex *d_fft_frame2, float *d_fft_accum, int tau_idx, int width, int height) {
	int x = threadIdx.x + blockIdx.x * blockSize_x;
	int y = threadIdx.y + blockIdx.y * blockSize_y;

	if (x < (width/2 + 1) && y < height) {
		int px_count = width * height;
		float fft_norm = 1.0 / (float)px_count;

		int pos = y * (width/2 + 1) + x;

		cufftComplex val;
		val.x = d_fft_frame1[pos].x - d_fft_frame2[pos].x;
		val.y = d_fft_frame1[pos].y - d_fft_frame2[pos].y;
		d_fft_accum[tau_idx * height * (width/2 + 1) + pos] += (fft_norm * val.x)*(fft_norm * val.x) + (fft_norm * val.y)*(fft_norm * val.y);
	}
}

__global__ void combine_accum(float *d_A, float *d_B, int width, int height, int tau_count) {
	int x = threadIdx.x + blockIdx.x * blockSize_x;
	int y = threadIdx.y + blockIdx.y * blockSize_y;

	if (x < (width/2 + 1) && y < height) {
		for (int t = 0; t < tau_count; t++)
			d_A[t * height * (width/2+1) + y * (width/2+1) + x] += d_B[t * height * (width/2+1) + y * (width/2+1) + x];

	}
}


bool LoadVideoToBuffer(unsigned char *h_ptr, int frame_count, VideoCapture cap, int w, int h) {
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
			fprintf(stderr, "Video Load Error: specified dimensions too large, %d > %d or %d > %d", w, cols, h, rows);
		}

		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				// Using img.at<>8
				h_ptr[frame_idx * num_elements + y * w + x] =  (unsigned char) input_img.data[((input_img.step)/input_img.elemSize1())* y + input_img.channels() * x];
			}
		}

//		for (int i = 0; i < 20; i++) printf("%d ", h_ptr[frame_idx * num_elements+i]);
//		printf("\n");

	}

    nvtxRangePop();
	return true;
}


void chunk_analysis(unsigned char *d_buff,
		float *d_parsed_buff,
		cufftComplex *d_fft_frames,
		float *d_fft_accum,
		cudaStream_t stream,
		int frame_count,
		int *tau_vector,
		int tau_count,
		cufftHandle fft_plan,
		frame_info frame_info,
		int bytes_per_px=1,
		int repeat_count=20) {

	// for r < repeat count
	// 	for t in tau_vector
	//    frame_idx = rand() % (frame_count - tau);
	//    d_accum[tau_idx] = intensity of difference (frame1, frame2)

	int out_width = frame_info.out_width;
	int out_height = frame_info.out_height;

	if (frame_info.out_width + frame_info.x_offset > frame_info.in_width || frame_info.out_height + frame_info.y_offset > frame_info.in_height) {
		fprintf(stderr, "[Image Parse Error] x / y offset too large. (Image width / height: %d / %d)\n", frame_info.in_width , frame_info.in_height );
	}

	dim3 blockDim(blockSize_x, blockSize_y, 1);

	int tmp_x, tmp_y;

	// out frame
	tmp_x = (int) ceil(out_width/(float)blockSize_x);
	tmp_y = (int) ceil(out_height/(float)blockSize_y);
	dim3 out_frame_grid(tmp_x, tmp_y, 1);

	// fft frame
	tmp_x = (int) ceil((out_width/2+1)/(float)blockSize_x);
	dim3 fftFrame(tmp_x, tmp_y, 1);

	cufftSetStream(fft_plan, stream);

	parseBuffer<<<out_frame_grid, blockDim, 0, stream >>>(d_buff, d_parsed_buff, frame_info, bytes_per_px, frame_count);

	//FFT execute
	if ((cufftExecR2C(fft_plan, d_parsed_buff, d_fft_frames)) != CUFFT_SUCCESS) {
		fprintf(stderr, "[cuFFT Exec Error]");
	}

	// Main loop
	int tau, frame_idx;
	cufftComplex *d_frame1, *d_frame2;
	int frame_size = out_height * (out_width/2 + 1);

	for (int repeat = 0; repeat < repeat_count; repeat++) {
		for (int tau_idx = 0; tau_idx < tau_count; tau_idx++) {
			tau = tau_vector[tau_idx];

			frame_idx = rand() % (frame_count - tau);

			d_frame1 = d_fft_frames + frame_idx * frame_size;
			d_frame2 = d_frame1 + tau * frame_size;

			processFFT<<<fftFrame, blockDim, 0, stream >>>(d_frame1, d_frame2, d_fft_accum, tau_idx, out_width, out_height);

		}
	}

	return;
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
        				val += d_in[(w/2+1) * h * tau_idx + (w/2+1)*y_i + x_i] * masks[w * h * q_idx + w*y_i + x_i] ;
        			}
        		}

                // Also should divide by chunk count
        		val *= 2; // account for symmetry
                val /= (float)px_count[q_idx]; // could be potential for overflow here
                val /= (float)norm_factor;
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


int main() {
	auto start_time = std::chrono::high_resolution_clock::now();

	//////////////////
	//  PARAMETERS  //
	//////////////////

	bool movie_file = true;

	FILE *moviefile;
	if ( !(moviefile = fopen("/media/ghaskell/Slow Drive/colloid_vids/red_colloids_0_5um.11Feb2021_15.27.31.movie", "rb" ))) {
		fprintf(stderr, "Couldn't open movie file.\n" );
		exit( EXIT_FAILURE );
	}

	VideoCapture cap("/media/ghaskell/Slow Drive/colloid_vids/red_colloids_0_5um11Feb2021152731.mp4");
	//VideoCapture cap(0);


	video_info_struct vid_info = initFile(moviefile);

	int img_width, img_height, bytes_per_px;

	if (movie_file) {


		img_width = vid_info.size_x;
		img_height = vid_info.size_y;
		bytes_per_px = vid_info.bpp;
	} else {
		img_width = cap.get(CAP_PROP_FRAME_WIDTH);
		img_height = cap.get(CAP_PROP_FRAME_HEIGHT);
		bytes_per_px = 1;
	}

	int out_width = 1024;
	int out_height = 1024;

	int x_offset = 0;
	int y_offset = 0;

	int buffer_frame_count = 100;
	int total_frames = 1400;
	int repeats = 20;

	int tau_count = 10;
	int tau_vector [tau_count];
	for (int j = 0; j < tau_count; j++) {
		tau_vector[j] = (j+2)*8;
	}

	int q_count = 25;
	float q_vector[q_count];
	for (int i = 0; i < q_count; i++) {
		q_vector[i] = (i + 2)*3;
	}

	//////////////////////////
	//  MEMORY ALLOCATIONS  //
	//////////////////////////

	// host & device frame buffer for both streams

	unsigned char *h_buffer1, *h_buffer2;
	unsigned char *d_buffer1, *d_buffer2;

	size_t buffer_size = sizeof(unsigned char) * buffer_frame_count * img_width * img_height * bytes_per_px; // #frames * size of raw frame

	gpuErrchk(cudaHostAlloc((void **) &h_buffer1, buffer_size, cudaHostAllocDefault)); // As we do async memory copies - must be pinned memory
	gpuErrchk(cudaHostAlloc((void **) &h_buffer2, buffer_size, cudaHostAllocDefault));

	gpuErrchk(cudaMalloc((void **) &d_buffer1, buffer_size));
	gpuErrchk(cudaMalloc((void **) &d_buffer2, buffer_size));

	// parsed frame buffer

	float *d_parsed1, *d_parsed2;

	size_t parsed_size = sizeof(float) * buffer_frame_count * out_width * out_height; // #frames * size of raw frame

	gpuErrchk(cudaMalloc((void **) &d_parsed1, parsed_size));
	gpuErrchk(cudaMalloc((void **) &d_parsed2, parsed_size));

	// FFT workspace

	cufftComplex *d_fft1, *d_fft2;

	size_t fft_size = sizeof(cufftComplex) * buffer_frame_count * out_height * (out_width / 2 + 1);

	gpuErrchk(cudaMalloc((void **) &d_fft1, fft_size));
	gpuErrchk(cudaMalloc((void **) &d_fft2, fft_size));

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

	////////////////////////
	//  SHUFFLE POINTERs  //
	////////////////////////

	unsigned char *d_buff_current = d_buffer1;
	unsigned char *d_buff_next = d_buffer2;

	unsigned char *h_buff_current = h_buffer1;
	unsigned char *h_buff_next = h_buffer2;

	float *d_parsed_current = d_parsed1;
	float *d_parsed_next = d_parsed2;

	cufftComplex *d_fft_current = d_fft1;
	cufftComplex *d_fft_next = d_fft2;

	float *d_accum_current = d_int_accum1;
	float *d_accum_next = d_int_accum2;

	////////////////
	//  FFT PLAN  //
	////////////////

	// plan to perform buffer_frame_number C2R,
	cufftHandle plan;

	int batch_count = buffer_frame_count;
	int rank = 2;
    int n[2] = {out_width, out_height};

	int idist = out_width * out_height;
	int odist = out_height * (out_width / 2 + 1);

    int inembed[] = {out_height, out_width};
    int onembed[] = {out_height, out_width / 2 + 1};

    int istride = 1;
    int ostride = 1;

    // TODO Implement cuFFT error checker
    if (cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batch_count) != CUFFT_SUCCESS) {fprintf(stderr, "cuFFT Error: Plan failure.\n");}

	//////////////////
	//  FINAL INIT  //
	//////////////////

	frame_info frame;
	frame.out_width = out_width;
	frame.out_height = out_height;
	frame.in_width = img_width;
	frame.in_height= img_height;
	frame.x_offset = x_offset;
	frame.y_offset = y_offset;

	// iteration counts

	int total_iterations = total_frames / buffer_frame_count;
	int iterations_remaining = total_iterations;

	// streams

	cudaStream_t stream1, stream2;

	cudaStreamCreate(&stream1);
	cudaStreamCreate(&stream2);

	cudaStream_t *stream_current = &stream1;
	cudaStream_t *stream_next = &stream2;

	/////////////////
	//  MAIN LOOP  //
	/////////////////

	if (movie_file)
		loadFileToHost(moviefile, h_buff_current, vid_info, buffer_frame_count);
	else
		LoadVideoToBuffer(h_buff_current, buffer_frame_count, cap, img_width, img_height); // puts chunk data into pinned host memory

	while (iterations_remaining > 1) {

		// copy host data to device (async)
		gpuErrchk(cudaMemcpyAsync(d_buff_current, h_buff_current, buffer_size, cudaMemcpyHostToDevice, *stream_current));

		// process chunk
		chunk_analysis(d_buff_current, d_parsed_current,d_fft_current,d_accum_current, *stream_current, buffer_frame_count, tau_vector, tau_count,
				plan, frame, bytes_per_px, repeats);

		// prevent overrun
		gpuErrchk(cudaStreamSynchronize(*stream_next));

		// load next chunk to CPU while GPU processing current

		if (movie_file)
			loadFileToHost(moviefile, h_buff_current, vid_info, buffer_frame_count);
		else
			LoadVideoToBuffer(h_buff_current, buffer_frame_count, cap, img_width, img_height); // puts chunk data into pinned host memory

		// pointer swap

		unsigned char *tmp = h_buff_current;
		h_buff_current = h_buff_next;
		h_buff_next = tmp;

		tmp = d_buff_current;
		d_buff_current = d_buff_next;
		d_buff_next = tmp;

		float *parsed_tmp = d_parsed_current;
		d_parsed_current = d_parsed_next;
		d_parsed_next = parsed_tmp;

		cufftComplex *fft_tmp = d_fft_current;
		d_fft_current = d_fft_next;
		d_fft_next = fft_tmp;

		float *accum_tmp = d_accum_current;
		d_accum_current = d_accum_next;
		d_accum_next = accum_tmp;

		cudaStream_t *stream_tmp = stream_current;
		stream_current = stream_next;
		stream_next = stream_tmp;

		// End of iteration
		printf("[Chunk complete (%d \\ %d)]\n", total_iterations - iterations_remaining + 1, total_iterations);
		iterations_remaining--;

	}

	// final iteration
	gpuErrchk(cudaMemcpyAsync(d_buff_current, h_buff_current, buffer_size, cudaMemcpyHostToDevice, *stream_current));

	chunk_analysis(d_buff_current, d_parsed_current,d_fft_current,d_accum_current, *stream_current, buffer_frame_count, tau_vector, tau_count,
			plan, frame, bytes_per_px, repeats);

	printf("[Chunk complete (%d \\ %d)]\n", total_iterations - iterations_remaining + 1, total_iterations);

	// sync device
	cudaDeviceSynchronize();

	// add both intensity accumulators together
	int grid_x = (int) ceil((img_width/2+1)/(float)blockSize_x);
	int grid_y = (int) ceil(out_height/(float)blockSize_y);

	dim3 blockDim(blockSize_x, blockSize_y, 1);
	dim3 gridDim(grid_x, grid_y, 1);
	combine_accum<<<gridDim, blockDim>>>(d_accum_current, d_accum_next, out_width, out_height, tau_count);

	// copy to host
	gpuErrchk(cudaMemcpy(h_out, d_accum_current, accum_size, cudaMemcpyDeviceToHost));

	cudaFree(h_buffer1); cudaFree(h_buffer2);
	cudaFree(d_buffer1); cudaFree(d_buffer2);
	cudaFree(d_parsed1); cudaFree(d_parsed1);
	cudaFree(d_fft1); cudaFree(d_fft2);
	cudaFree(d_int_accum1); cudaFree(d_int_accum2);
	cufftDestroy(plan);

	auto end_time = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
	std::cout << "END (time elapsed: " << (float)duration/1000000.0 << " seconds.)"<< std::endl;

	analyseFFTHost(h_out, repeats*total_iterations, q_vector, q_count, tau_vector, tau_count, out_width, out_height);

}
