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

struct QMaskStruct {
    // encapsulates data surrounding q-vector mask

    int q_count; // count of q vectors
    float * mask_fft; // FFT of q vector mask
    float * q_vector; // average q associated with each mask
    int * px_count; // normalisation factor for each mask
};

void write_to_file(float * ptr, int width, int height) {
	std::ofstream myfile("/home/ghaskell/projects_Git/cuDDM/data/data2.txt");
	float * print_buff = new float[width * height];
	if (myfile.is_open()) {
		for (int x = 0; x < width*height; x++) {
			myfile << ptr[x] <<" ";
		}
		myfile << std::endl;
	}
	myfile.close();
}


void LoadVideoToBuffer(float *d_ptr, int frame_count, VideoCapture cap, int w, int h) {
	std::cout << "Load frame " << frame_count << " (w: " <<  w << " h: " << h << ")" << std::endl;

	// No bounds check! assume that w, h smaller than mat
	int num_elements = w * h;

	Mat input_img; //, grayscale_img;
	float *h_ptr = new float[w * h * frame_count];

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
				h_ptr[frame_idx * num_elements + y * w + x] =  (float) input_img.data[((input_img.step)/input_img.elemSize1())* y + input_img.channels() * x];
			}
		}
	}
	cudaMemcpy(d_ptr, h_ptr, num_elements * frame_count * sizeof(float), cudaMemcpyHostToDevice);
}

//__global__ void AbsDifference(float *d_buffer, cufftReal *d_diff, int frame1, int frame2, int width, int height) {
//	int size = width * height;
//
//	int x = threadIdx.x + blockIdx.x * 32;
//	int y = threadIdx.y + blockIdx.y * 32;
//
//	if (x <= width-1 && y <= height-1) {
//		int pos_offset = y * width + x;
//		d_diff[pos_offset] = abs(d_buffer[frame1 * size + pos_offset] - d_buffer[frame2 * size + pos_offset]);
//	}
//
//	return;
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

void analyseChunk(float *d_chunk_ptr, int frame_count, float *d_out, int *tau_vector, int tau_count, int width, int height, float *debug_buff=NULL) {
	// debug_buffer is a width * height *sizeof(float) buffer which can be printed
	//	if (debug_buff != NULL) {
	//		cudaMemcpy(debug_buff, <device ptr>, width*height*sizeof(float), cudaMemcpyDeviceToHost);
	//		return;
	//	}
	// d_out size: tau_count * width * height * sizeof(float)

	int w = width;
	int h = height;

	std::cout << "Chunk Analysis Start (" << frame_count << " frames)" <<  std::endl;

	// Initialise workspace
	// these buffers are fit for one frame, if we do all taus at once then should modify

	cufftReal *d_diff_local;
	cudaMalloc((void **) &d_diff_local, w * h * sizeof(cufftReal));

	cufftComplex *d_fft_local;
	cudaMalloc((void **) &d_fft_local, w * (h / 2 + 1) * sizeof(cufftComplex));

	// Max 1024 (32 x 32) threads per block hence multiple blocks to operate on a frame
	dim3 blockDim(32, 32, 1);
	dim3 gridDim((int)ceil(width/32.0), (int)ceil(height/32.0), 1);

	// cuFFT plan
	cufftHandle plan;
	if ((cufftPlan2d(&plan, w, h, CUFFT_R2C)) != CUFFT_SUCCESS) {
		std::cout << "cuFFT Plan Error" << std::endl;
	}

	// Main loop

	int tau, idx1, idx2;
	float *d_frame1, *d_frame2;

	for (int repeats = 0; repeats < 10; repeats++) {
		for (int tau_idx = 0; tau_idx < tau_count; tau_idx++) {
			tau = tau_vector[tau_idx];

			idx1 = rand() % (frame_count - tau);
			idx2 = idx1 + tau;
			std::cout << "tau: " << tau << " idxs: " << idx1 << ", " << idx2 << std::endl;

			d_frame1 = d_chunk_ptr + (idx1 * w * h);	// float pointer to frame 1
			d_frame2 = d_chunk_ptr + (idx2 * w * h);

			AbsDifference<<<gridDim, blockDim>>>(d_diff_local, d_frame1, d_frame2, w, h); // find absolute difference

			// FFT execute
			if ((cufftExecR2C(plan, d_diff_local, d_fft_local)) != CUFFT_SUCCESS) {
				std::cout << "cuFFT Exec Error" << std::endl;
			}

			processFFT<<<gridDim, blockDim>>>(d_fft_local, d_out, tau_idx, w, h); // process FFT (i.e. normalise and add to accumulator)
		}
	}

//	int tau, frame1, frame2;
//
//	for (int tau_idx = 0; tau_idx < tau_count; tau_idx++) {
//		for (int repeats = 0; repeats < 40; repeats++) {
//			tau = tau_vector[tau_idx];
//
//			frame1 = rand() % (chunk_frame_count - tau);
//			frame2 = frame1 + tau;
//
//
//
//			std::cout << " Abs Diff" << std::endl;
//			AbsDifference<<<gridDim, blockDim>>>(d_buffer, d_diff_local, frame1, frame2, width, height);
//
//			// FFT
//			std::cout << " FFt Diff" << std::endl;
//			cufftHandle plan;
//			if ((cufftPlan2d(&plan, height, width, CUFFT_R2C)) != CUFFT_SUCCESS) {
//				std::cout << "cufft plan error" << std::endl;
//			}
//
//			if ((cufftExecR2C(plan, (cufftReal*)d_diff_local, (cufftComplex*)d_fft_local)) != CUFFT_SUCCESS) {
//				std::cout << "cufft exec error" << std::endl;
//			}
//
//			std::cout << " Process FFt" << std::endl;
//			processFFT<<<gridDim, blockDim>>>(d_fft_local, d_fft_accum, tau_idx, width, height);
//			//cudaMemcpy(print_buffer, d_fft_accum, width*height*sizeof(float), cudaMemcpyDeviceToHost);
//		}
//	}

}

void RunDDM(float *out, VideoCapture cap, int width, int height, int frame_count, int tau_count, int* tau_vector, float * print_buffer) {
	std::cout<<"Start"<<std::endl;

	// Initialise buffer parameters
	int buff_frames, chunk_frames;
	buff_frames = 30; // as we are running load / analyse in serial buff size = chunk size
	chunk_frames = 30;

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
		analyseChunk(d_buffer, chunk_frames, d_fftAccum, tau_vector, tau_count, width, height, print_buffer);
		frame_count -= chunk_frames;
	}

	cudaMemcpy(out, d_fftAccum, work_size, cudaMemcpyDeviceToHost);
	std::cout<<"Done"<<std::endl;
}

void HARDCODEanalyseFFTHost(float *d_in, float *d_out, int *tau_vector, int tau_count, int width, int height, float *debug_buff=NULL) {
    int w = width; int h = height;

	// Generate q - vectors - Hard Coded
	int q_count = 20;
	float q_squared[20] = 	{ 1.        ,   3.6472384 ,   7.94985584,  13.90785234,  21.52122788, 30.78998247,  41.71411612,
    							 54.29362881,  68.52852055,  84.41879134, 101.96444118, 121.16547007, 142.021878 , 164.53366499,
								188.70083102, 214.52337611, 242.00130024, 271.13460343, 301.92328566, 334.36734694};
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
//    // Mask generation end
//
//    // Start analysis
//    float *tau_frame;
//    float val;
//
//    for (int tau_idx = 0; tau_idx < tau_count; tau_idx++) {
//        tau_frame = d_in + (w * h * tau_idx);
//
//        for (int q_idx = 0; q_idx < q_count; q_idx++) {
//        	val = 0;
//        	if (px_count[q_idx] != 0) { // If the mask has no values iq_tau must be zero
//                for (int i = 0; i < w*h; i++) { 	// iterate through all pixels
//                	val += d_in[w * h * tau_idx + i] * masks[w * h * tau_idx + i];
//                }
//                // Also should divide by chunk count
//                val /= (float)px_count[q_idx]); // could be potential for overflow here
//        	}
//
//        	iq_tau[q_idx * tau_count + tau_idx] = val;
//        }
//    }
//
//


}






void analyseFFTHost(float *h_in, float *iq_tau, int number_chunks, int tau_count, int* tau_vector, int width, int height) {
	// Handles generation of masks
	std::cout << "Final analysis start" << std::endl;

	int frame_size = width * height;


//	int smallest_size = (width < height) ? width : height;
//	int q_count = (int)(log2(smallest_size) + log2(2.0/3.0) - 1);
//    float *q_vector = new float[q_count];
//    float *q_sq_vector = new float[q_count];
//    int current_q = 3;
//    for (int i=0; i < q_count; i++) {
//        std::cout << current_q << std::endl;
//        q_vector[i] =  current_q;
//        q_sq_vector[i] = q_vector[i] * q_vector[i];
//        current_q *= 2;
//    }

	int q_count = 20;
    float q_vector[20] = { 1.        ,  1.90977444,  2.81954887,  3.72932331,  4.63909774,
							  5.54887218,  6.45864662,  7.36842105,  8.27819549,  9.18796992,
							 10.09774436, 11.0075188 , 11.91729323, 12.82706767, 13.73684211,
							 14.64661654, 15.55639098, 16.46616541, 17.37593985, 18.28571429};

    float q_sq_vector[20] = {  1.        ,   3.6472384 ,   7.94985584,  13.90785234,  21.52122788,
								  30.78998247,  41.71411612,  54.29362881,  68.52852055,  84.41879134,
								 101.96444118, 121.16547007, 142.021878  , 164.53366499, 188.70083102,
								 214.52337611, 242.00130024, 271.13460343, 301.92328566, 334.36734694};

    int *px_count = new int[q_count]();
    float *masks = new float[frame_size * q_count];

    float half_w, half_h;
    half_h = height / 2.0;
    half_w = width / 2.0;
    float r_sqr, ratio;

    // First Generate the radius masks
    int shift_x, shift_y;
    for (int q_idx = 0; q_idx < q_count; q_idx++) {
        for (int x = 0; x < width; x++)
        {
            for (int y = 0; y < height; y++)
            {
                // We want the x and y values to be FFT shifted, we can perform this manually
                shift_x = (x + (int)half_w) % width;
                shift_y = (y + (int)half_h) % height;

                r_sqr = (shift_x - half_w) * (shift_x - half_w) + (shift_y - half_h) * (shift_y - half_h);
                ratio = r_sqr / q_sq_vector[q_idx];
                if (1 <= ratio && ratio <= 1.44) {
                    masks[q_idx*frame_size + y*width + x] = 1.0;
                    px_count[q_idx] += 1;
                } else {
                    masks[q_idx*frame_size + y*width + x] = 0.0;
                }
            }
        }
    }
    // Mask generation end

    // Start analysis
    float * tau_frame;
    float val;

    for (int tau_idx = 0; tau_idx < tau_count; tau_idx++) {
        tau_frame = h_in + (frame_size * tau_idx);

        for (int q_idx = 0; q_idx < q_count; q_idx++) {
        	val = 0;
        	if (!(px_count[q_idx] == 0)) { // If the mask has no values iq_tau must be zero
                for (int i = 0; i < frame_size; i++) { 	// iterate through all pixels
                	val += tau_frame[i] * masks[q_idx * frame_size + i];
                }
                val /= ((float)number_chunks * (float)px_count[q_idx]); // could be potential for overflow here
        	}

        	iq_tau[q_idx * tau_count + tau_idx] = val;
        }
    }

}

int main(int argc, char **argv)
{
	VideoCapture cap("/home/ghaskell/projects_Git/cuDDM/data/colloid_0.5um_vid.mp4");

	int tau_count = 11;
	int tau_vector [tau_count] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
	int width = 512;
	int height = 512;
	int frame_count = 400;


	float * out = new float [width * height * tau_count];
	float * print_buffer = new float[width * height];
	RunDDM(out, cap, width, height, frame_count, tau_count, &tau_vector[0], print_buffer);
	write_to_file(out, width, height);


	// HARD CODED - BAD - only works for 1024
	int q_count = 20;
    float q_vector[20] = { 1.        ,  1.90977444,  2.81954887,  3.72932331,  4.63909774,
							  5.54887218,  6.45864662,  7.36842105,  8.27819549,  9.18796992,
							 10.09774436, 11.0075188 , 11.91729323, 12.82706767, 13.73684211,
							 14.64661654, 15.55639098, 16.46616541, 17.37593985, 18.28571429};

	float * iq_tau = new float[tau_count * q_count]();

	analyseFFTHost(out, iq_tau, 20, tau_count, tau_vector, width, height);

	for (int i = 0; i < tau_count * q_count; i++) {
		std::cout << iq_tau[i] << std::endl;
	}

	// outputting iqtau
    std::ofstream myfile("/home/ghaskell/projects_Git/cuDDM/data/iqt.txt");

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
	    		myfile << out[q_idx * tau_count + t_idx] << " ";
	    	}
			myfile << "\n";
		}

		myfile.close();
    } else {
    	std::cout << "Unable to open file";
    	return 0;
    }


//	std::ofstream myfile("/home/ghaskell/projects_Git/cuDDM/data/data2.txt");
//	float * print_buff = new float[width * height];
//	if (myfile.is_open()) {
//		for (int t = 0; t < 1; tau_count++) {
//			for (int x = 0; x < width*height; x++) {
//				myfile << out[t*width*height+ x] <<" ";
//			}
//			myfile << std::endl;
//		}
//	}
//	myfile.close();

	std::cout << "DONE" << std::endl;

}


