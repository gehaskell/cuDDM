////TODO: clean up print statements - switch errors to fprintf
////TODO: implement bytes per pixel correctly
//
//#include "movie_reader.h"
//
//#include <iostream>
//#include <stdio.h>
//#include <cuda_runtime.h>
//#include <cufft.h>
//#include <fstream>
//#include <opencv2/opencv.hpp>
//#include <chrono>
//#include <nvToolsExt.h>
//
//#define blockSize_x 16
//#define blockSize_y 16
//
//#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
//
//inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
//{
//   if (code != cudaSuccess)
//   {
//      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
//      if (abort) exit(code);
//   }
//}
//
//using namespace cv;
//
//__global__ void frameDifference(cufftReal *d_diff, unsigned char *d_frame1, unsigned char *d_frame2,
//		int img_width, int img_height,
//		int out_width, int out_height,
//		int bytes_per_px=1)
//	{
//	// If more than one byte per pixel, then we just take first channel
//	int x = threadIdx.x + blockIdx.x * blockSize_x;
//	int y = threadIdx.y + blockIdx.y * blockSize_y;
//
//	if (x <= out_width-1 && y <= out_height-1) {
//		d_diff[y * out_width + x] = ((cufftReal)d_frame1[(y * img_width + x) * bytes_per_px] - (cufftReal)d_frame2[(y * img_width + x)  * bytes_per_px]) ;
//	}
//	return;
//}
//
//
//__global__ void processFFT(cufftComplex *d_input, float *d_output, int tau_idx, int width, int height) {
//	// Takes output of cuFFT R2C operation, normalises (i.e. divides each value by pixel count), finds the intensity and adds to accumulator
//
//	int px_count = width * height;
//	float fft_norm = 1.0 / (float)px_count;
//
//	int j = threadIdx.x + blockIdx.x * blockSize_x;
//	int i = threadIdx.y + blockIdx.y * blockSize_y;
//
//	float intensity;
//	if (j < width && i < height) {
//		int sym_w = width / 2 + 1; // to deal with complex (hermitian) symmetry
//		cufftComplex val;
//
//		if (j >= sym_w) {
//			// real ->  d_data[i*sym_w+(width-j)].x
//			// img  -> -d_data[i*sym_w+(width-j)].y
//			val =  d_input[i*sym_w+(width-j)];
//			intensity = (fft_norm * val.x)*(fft_norm * val.x) + (fft_norm * val.y)*(fft_norm * val.y);
//
//		} else {
//			// real -> d_data[i*sym_w+j].x
//			// img  -> d_data[i*sym_w+j].y
//			val = d_input[i*sym_w+j];
//			intensity = (fft_norm * val.x)*(fft_norm * val.x) + (fft_norm * val.y)*(fft_norm * val.y);
//		}
//
//		// add to fft_accum
//		//d_output[tau_idx * px_count + i * width + j] += intensity;
//		atomicAdd((d_output + tau_idx * px_count + i * width + j), intensity);
//	}
//}
//
//__global__ void array_add(float *d_A, float *d_B, int w, int h, int tau_count) {
//	int x = threadIdx.x + blockIdx.x * blockSize_x;
//	int y = threadIdx.y + blockIdx.y * blockSize_y;
//
//	if (x <= w-1 && y <= h-1) {
//		for (int t = 0; t < tau_count; t++)
//			d_A[t*w*h + y*w + x] += d_B[t*w*h + y*w + x];
//	}
//}
//
//
//bool LoadVideoToBuffer(unsigned char *h_ptr, int frame_count, VideoCapture cap, int w, int h) {
//	nvtxRangePush(__FUNCTION__); // to track video loading times in nvvp
//
//	//printf("load video (%d frames) (w: %d, h: %d)\n", frame_count, w, h);
//
//	// No bounds check! assume that w, h smaller than mat
//	int num_elements = w * h;
//
//	Mat input_img; //, grayscale_img;
//
//	// There is some problems with the image type we are using - though some effort was put into switching to a
//	// more generic image format, more thought is required therefore switch to just dealing with 3 channel uchars
//	// look at http://ninghang.blogspot.com/2012/11/list-of-mat-type-in-opencv.html and
//	// https://docs.opencv.org/3.4/d3/d63/classcv_1_1Mat.html#aa5d20fc86d41d59e4d71ae93daee9726 for more info.
//
//
//	for (int frame_idx = 0; frame_idx < frame_count; frame_idx++) {
//		//std::cout << "Loaded frame " << frame_idx << std::endl;
//
//		cap >> input_img;
//
//		if (input_img.empty()) {
//			fprintf(stderr,"Video frame is empty");
//			return false;
//		}
//
//		//input_img.convertTo(grayscale_img, CV_32FC1); // covert to grayscale image
//
//		if (input_img.type() != 16) {
//			std::cout << "Non standard image format detected, may cause unexpected behaviour, image type : " << input_img.type() << std::endl;
//			return false;
//		}
//
//	    //imshow("Input", input_img);
//	    //waitKey(0);
//
//		int cols = input_img.cols, rows = input_img.rows;
//
//		if (w > cols || h > rows) {
//			fprintf(stderr, "Video Load Error: specified dimensions too large, %d > %d or %d > %d", w, cols, h, rows);
//		}
//
//		for (int y = 0; y < h; y++) {
//			for (int x = 0; x < w; x++) {
//				// Using img.at<>8
//				h_ptr[frame_idx * num_elements + y * w + x] =  (unsigned char) input_img.data[((input_img.step)/input_img.elemSize1())* y + input_img.channels() * x];
//			}
//		}
//
////		for (int i = 0; i < 20; i++) printf("%d ", h_ptr[frame_idx * num_elements+i]);
////		printf("\n");
//
//	}
//
//    nvtxRangePop();
//	return true;
//}
//
//
//void processChunk(cudaStream_t stream, unsigned char *d_ptr,
//		int frame_count,
//		float *d_out,
//		int *tau_vector,
//		int tau_count,
//		cufftReal *d_diff_workspace,
//		cufftComplex *d_fft_workspace,
//		cufftHandle plan,
//		int img_width, int img_height,
//		int out_width, int out_height,
//		int repeat_count = 50, float *debug_buff=NULL) {
//
//	// debug_buffer is a width * height *sizeof(float) buffer which can be printed
//	//	if (debug_buff != NULL) {
//	//		cudaMemcpy(debug_buff, <device ptr>, width*height*sizeof(float), cudaMemcpyDeviceToHost);
//	//		return;
//	//	}
//	// d_out size: tau_count * width * height * sizeof(float)
//
//	//printf("chunk analysis (%d frames).\n", frame_count);
//
//	// Max 1024 (32 x 32) threads per block hence multiple blocks to operate on a frame
//	// Max number of thread blocks is 65536)
//
//	dim3 blockDim(blockSize_x, blockSize_y, 1);
//	int grid_x = (int) ceil(out_width/(float)blockSize_x);
//	int grid_y = (int) ceil(out_height/(float)blockSize_y);
//
//	dim3 gridDim(grid_x, grid_y, 1);
//
//	if (gridDim.x * gridDim.y * gridDim.z > 65536) {
//		fprintf(stderr, "Image too big, not enough thread blocks (%d).\n", gridDim.x * gridDim.y * gridDim.z);
//	}
//
//	cufftSetStream(plan, stream);
//
//	// Main loop
//	int tau, frame_idx;
//	unsigned char *d_frame1, *d_frame2;
//	cufftComplex *d_local_fft;
//	cufftReal *d_local_diff;
//
//	for (int tau_idx = 0; tau_idx < tau_count; tau_idx++) {
//		for (int repeat = 0; repeat < repeat_count; repeat++) {
//
//			tau = tau_vector[tau_idx];
//
//			d_local_fft = d_fft_workspace + (tau_idx * out_width * (out_height / 2 + 1));
//			d_local_diff = d_diff_workspace + (tau_idx * out_width * out_height);
//
//			frame_idx = rand() % (frame_count - tau);
////			frame_idx = 0;
//
//			d_frame1 = d_ptr + (frame_idx * img_width * img_height);	// float pointer to frame 1
//			d_frame2 = d_ptr + ((frame_idx+tau) * img_width * img_height);
//
//			frameDifference<<<gridDim, blockDim, 0, stream >>>(d_local_diff, d_frame1, d_frame2, img_width, img_height, out_width, out_height); // find absolute difference
//
////			printf("%d  ", tau);
////			float *tmp = new float[20];
////			cudaMemcpy(tmp, d_local_diff, 20*sizeof(float), cudaMemcpyDeviceToHost);
////			for (int i = 0; i < 20; i++) printf("%f ", tmp[i]);
////			printf("\n");
//
//			//FFT execute
//			if ((cufftExecR2C(plan, d_local_diff, d_local_fft)) != CUFFT_SUCCESS) {
//				std::cout << "cuFFT Exec Error\n" << std::endl;
//			}
//
//			processFFT<<<gridDim, blockDim, 0, stream>>>(d_local_fft, d_out, tau_idx, out_width, out_height); // process FFT (i.e. normalise and add to accumulator)
//
////			printf("%d  ", tau);
////			cudaMemcpy(tmp, d_out+out_width*out_height*tau_idx, 20*sizeof(float), cudaMemcpyDeviceToHost);
////			for (int i = 0; i < 20; i++) printf("%f ", tmp[i]);
////			printf("\n ");
//
//		}
//	}
//
//	return;
//}
//
//
//void analyseFFTHost(float *d_in, int norm_factor, float *q_vector, int q_count, int *tau_vector, int tau_count, int width, int height) {
//    int w = width; int h = height;
//
//	float q_squared[q_count];
//	for (int i = 0; i < q_count; i++) {
//		q_squared[i] = q_vector[i] * q_vector[i];
//	}
//
//	// Generate masks
//    int *px_count = new int[q_count](); // () initialises to zero
//    float *masks = new float[w * h * q_count];
//
//    float half_w, half_h;
//    half_h = height / 2.0;
//    half_w = width / 2.0;
//    float r_sqr, ratio;
//
//    // First Generate the radius masks
//    int shift_x, shift_y;
//    for (int q_idx = 0; q_idx < q_count; q_idx++) {
//        for (int x = 0; x < w; x++)
//        {
//            for (int y = 0; y < h; y++)
//            {
//                // Perform manual FFT shift
//                shift_x = (x + (int)half_w) % w;
//                shift_y = (y + (int)half_h) % h;
//
//                // Distance relative to centre
//                shift_x -= half_w;
//                shift_y -= half_h;
//
//                r_sqr = shift_x * shift_x + shift_y * shift_y;
//                ratio = r_sqr / q_squared[q_idx];
//
//                if (1 <= ratio && ratio <= 1.44) { // we want values from 1.0 * q to 1.2 * q
//                    masks[q_idx*w*h + y*w + x] = 1.0;
//                    px_count[q_idx] += 1;
//                } else {
//                    masks[q_idx*w*h + y*w + x] = 0.0;
//                }
//            }
//        }
//    }
//
//
//    // Start analysis
//    float val;
//	float * iq_tau = new float[tau_count * q_count]();
//
//    for (int tau_idx = 0; tau_idx < tau_count; tau_idx++) {
//
//        for (int q_idx = 0; q_idx < q_count; q_idx++) {
//        	val = 0;
//
//        	if (px_count[q_idx] != 0) { // If the mask has no values iq_tau must be zero
//
//        		for (int i = 0; i < w*h; i++) { 	// iterate through all pixels
//                	val += d_in[w * h * tau_idx + i] * masks[w * h * q_idx + i] ;
//                }
//                // Also should divide by chunk count
//                val /= (float)px_count[q_idx]; // could be potential for overflow here
//                val /= (float)norm_factor;
//        	} else {
//        		printf("q %d has zero mask pixels\n", q_idx);
//        	}
//
//        	iq_tau[q_idx * tau_count + tau_idx] = val;
//        }
//    }
//
//	// outputting iqtau
//    std::ofstream myfile("/home/ghaskell/projects_Git/cuDDM_streams/out/iqt.txt");
//
//    if (myfile.is_open()) {
//    	for (int i = 0; i < q_count; i++) {
//    		myfile << q_vector[i] << " ";
//    	}
//		myfile << "\n";
//    	for (int i = 0; i < tau_count; i++) {
//    		myfile << tau_vector[i] << " ";
//    	}
//		myfile << "\n";
//
//		for (int q_idx = 0; q_idx < q_count; q_idx++) {
//	    	for (int t_idx = 0; t_idx < tau_count; t_idx++) {
//	    		myfile << iq_tau[q_idx * tau_count + t_idx] << " ";
//	    	}
//			myfile << "\n";
//		}
//
//		myfile.close();
//		printf("I(Q, tau) written to /home/ghaskell/projects_Git/cuDDM_streams/out/iqt.txt");
//    } else {
//    	std::cout << "Unable to open file" << std::endl;
//    	return;
//    }
//}
//
//
//
//int main(){
//	auto start_time = std::chrono::high_resolution_clock::now();
//
//	//////////////////
//	//  PARAMETERS  //
//	//////////////////
//
//	bool movie_file = false;
//
//	VideoCapture cap("/media/ghaskell/Slow Drive/colloid_vids/red_colloids_0_5um11Feb2021152731.mp4");
//
//	FILE *moviefile;
//	if ( !(moviefile = fopen("/media/ghaskell/Slow Drive/colloid_vids/blue_colloids_1um11feb160000.movie", "rb" ))) {
//		fprintf(stderr, "Couldn't open movie file.\n" );
//		exit( EXIT_FAILURE );
//	}
//	video_info_struct vid_info = initFile(moviefile);
//
//	int img_width, img_height, bytes_per_px;
//
//	if (movie_file) {
//		img_width = vid_info.size_x; // incompatibility due to different matrix major (hence y - width / x - height)
//		img_height = vid_info.size_y;
//		bytes_per_px = vid_info.bpp;
//	} else {
//		img_width = cap.get(CAP_PROP_FRAME_WIDTH);
//		img_height = cap.get(CAP_PROP_FRAME_HEIGHT);
//		bytes_per_px = 1;
//	}
//
//
//
//	int out_width = 1024;
//	int out_height = 1024;
//
//	int buffer_frame_count = 50;
//	int total_frames = 1500;
//	int repeat_count = 100;
//
//	int tau_count = 15;
//	int tau_vector [tau_count];
//	for (int j = 0; j < tau_count; j++) {
//		tau_vector[j] = (j+2);
//	}
//
//	int q_count = 25;
//	float q_vector[q_count];
//	for (int i = 0; i < q_count; i++) {
//		q_vector[i] = (i + 2)*3;
//	}
//
//	//////////////////////////
//	//  MEMORY ALLOCATIONS  //
//	//////////////////////////
//
//	// host & device frame buffer for both streams
//
//	unsigned char *h_buffer1, *h_buffer2;
//	unsigned char *d_buffer1, *d_buffer2;
//
//	size_t buffer_size = sizeof(unsigned char) * buffer_frame_count * img_width * img_height * bytes_per_px; // #frames * size of raw frame
//
//	gpuErrchk(cudaHostAlloc((void **) &h_buffer1, buffer_size, cudaHostAllocDefault)); // As we do async memory copies - must be pinned memory
//	gpuErrchk(cudaHostAlloc((void **) &h_buffer2, buffer_size, cudaHostAllocDefault));
//
//	gpuErrchk(cudaMalloc((void **) &d_buffer1, buffer_size));
//	gpuErrchk(cudaMalloc((void **) &d_buffer2, buffer_size));
//
//	// workspace (diff & FFT) for both streams
//
//	cufftReal *d_diff1, *d_diff2;
//	cufftComplex *d_fft1, *d_fft2;
//
//	size_t diff_size = sizeof(cufftReal) * tau_count * out_width * out_height ;
//	size_t fft_size = sizeof(cufftComplex) * tau_count * out_width * (out_height / 2 + 1);
//
//	gpuErrchk(cudaMalloc((void **) &d_diff1, diff_size));
//	gpuErrchk(cudaMalloc((void **) &d_diff2, diff_size));
//
//	gpuErrchk(cudaMalloc((void **) &d_fft1, fft_size));
//	gpuErrchk(cudaMalloc((void **) &d_fft2, fft_size));
//
//	// FFT intensity accumulators for both streams (must also initialise to zero)
//
//	float *d_int_accum1, *d_int_accum2;
//
//	size_t accum_size = sizeof(float) * tau_count * out_width * out_height;
//
//	gpuErrchk(cudaMalloc((void **) &d_int_accum1, accum_size));
//	gpuErrchk(cudaMalloc((void **) &d_int_accum2, accum_size));
//
//	gpuErrchk(cudaMemset(d_int_accum1, 0, accum_size));
//	gpuErrchk(cudaMemset(d_int_accum2, 0, accum_size));
//
//	// Host output (does not need to be pinned)
//
//	float *h_out;
//
//	h_out = new float[tau_count * out_width * out_height];
//
//	////////////////////////
//	//  SHUFFLE POINTERs  //
//	////////////////////////
//
//	unsigned char *d_buff_current = d_buffer1;
//	unsigned char *d_buff_next = d_buffer2;
//
//	unsigned char *h_buff_current = h_buffer1;
//	unsigned char *h_buff_next = h_buffer2;
//
//	cufftReal *d_diff_current = d_diff1;
//	cufftReal *d_diff_next = d_diff2;
//
//	cufftComplex *d_fft_current = d_fft1;
//	cufftComplex *d_fft_next = d_fft2;
//
//	float *d_accum_current = d_int_accum1;
//	float *d_accum_next = d_int_accum2;
//
//	//////////////////
//	//  FINAL INIT  //
//	//////////////////
//
//	// iteration counts
//
//	int total_iterations = total_frames / buffer_frame_count;
//	int iterations_remaining = total_iterations;
//
//	// cuFFT plan
//
//	cufftHandle plan;
//	if ((cufftPlan2d(&plan, out_width, out_height, CUFFT_R2C)) != CUFFT_SUCCESS) {
//		fprintf(stderr, "cuFFT Error: Plan failure.\n");
//	}
//
//	// streams
//
//	cudaStream_t stream1, stream2;
//
//	cudaStreamCreate(&stream1);
//	cudaStreamCreate(&stream2);
//
//	cudaStream_t *stream_current = &stream1;
//	cudaStream_t *stream_next = &stream2;
//
//	//////////////////
//	//  MAIN LOOP  //
//	//////////////////
//
//	if (movie_file)
//		loadFileToHost(moviefile, h_buff_current, vid_info, buffer_frame_count);
//	else
//		LoadVideoToBuffer(h_buff_current, buffer_frame_count, cap, img_width, img_height); // puts chunk data into pinned host memory
//
//
//
////	printf("%d, %d, %d, %d, %d\n", h_buff_current[1918], h_buff_current[1919],h_buff_current[1920],h_buff_current[1921], h_buff_current[1922]);
////	printf("%d, %d, %d, %d, %d\n", h_buff_current[1198] ,h_buff_current[1199],h_buff_current[1200],h_buff_current[1201], h_buff_current[1202]);
////	exit(10);
//
//
//	while (iterations_remaining > 1) {
//
//		// copy host data to device (async)
//		gpuErrchk(cudaMemcpyAsync(d_buff_current, h_buff_current, buffer_size, cudaMemcpyHostToDevice, *stream_current));
//
//		// process chunk
//		processChunk(*stream_current, d_buff_current, buffer_frame_count, d_accum_current,
//				tau_vector, tau_count, d_diff_current, d_fft_current, plan, img_width, img_height, out_width, out_height, repeat_count);
//
//		// prevent overrun
//		gpuErrchk(cudaStreamSynchronize(*stream_next));
//
//		// load next chunk to CPU while GPU processing current
//
//		if (movie_file)
//			loadFileToHost(moviefile, h_buff_current, vid_info, buffer_frame_count);
//		else
//			LoadVideoToBuffer(h_buff_current, buffer_frame_count, cap, img_width, img_height); // puts chunk data into pinned host memory
//
//	    // pointer swap
//
//	    unsigned char *tmp = h_buff_current;
//	    h_buff_current = h_buff_next;
//	    h_buff_next = tmp;
//
//	    tmp = d_buff_current;
//	    d_buff_current = d_buff_next;
//	    d_buff_next = tmp;
//
//		cufftReal *diff_tmp = d_diff_current;
//		d_diff_current = d_diff_next;
//		d_diff_next = diff_tmp;
//
//		cufftComplex *fft_tmp = d_fft_current;
//		d_fft_current = d_fft_next;
//		d_fft_next = fft_tmp;
//
//		float *accum_tmp = d_accum_current;
//		d_accum_current = d_accum_next;
//		d_accum_next = accum_tmp;
//
//		cudaStream_t *stream_tmp = stream_current;
//		stream_current = stream_next;
//		stream_next = stream_tmp;
//
//		// End of iteration
//		printf("[Chunk complete (%d \\ %d)]\n", total_iterations - iterations_remaining + 1, total_iterations);
//		iterations_remaining--;
//
//	}
//
//	// final iteration
//	gpuErrchk(cudaMemcpyAsync(d_buff_current, h_buff_current, buffer_size, cudaMemcpyHostToDevice, *stream_current));
//	processChunk(*stream_current, d_buff_current, buffer_frame_count, d_accum_current,
//			tau_vector, tau_count, d_diff_current, d_fft_current, plan, img_width, img_height, out_width, out_height, repeat_count);
//	printf("[Chunk complete (%d \\ %d)]\n", total_iterations - iterations_remaining + 1, total_iterations);
//
//	// sync device
//    cudaDeviceSynchronize();
//
//	// add both intensity accumulators together
//	int grid_x = (int) ceil(out_width/(float)blockSize_x);
//	int grid_y = (int) ceil(out_height/(float)blockSize_y);
//
//	dim3 blockDim(blockSize_x, blockSize_y, 1);
//	dim3 gridDim(grid_x, grid_y, 1);
//	array_add<<<gridDim, blockDim>>>(d_accum_current, d_accum_next, out_width, out_height, tau_count);
//
//	// copy to host
//	gpuErrchk(cudaMemcpy(h_out, d_accum_current, accum_size, cudaMemcpyDeviceToHost));
//
//	cudaFree(h_buffer1); cudaFree(h_buffer2);
//	cudaFree(d_buffer1); cudaFree(d_buffer2);
//	cudaFree(d_diff1); cudaFree(d_diff2);
//	cudaFree(d_fft1); cudaFree(d_fft2);
//	cudaFree(d_int_accum1); cudaFree(d_int_accum2);
//	cufftDestroy(plan);
//
//	auto end_time = std::chrono::high_resolution_clock::now();
//	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
//	std::cout << "END (time elapsed: " << (float)duration/1000000.0 << " seconds.)"<< std::endl;
//
//	analyseFFTHost(h_out, repeat_count*total_iterations, q_vector, q_count, tau_vector, tau_count, out_width, out_height);
//
//
//
//
//}
