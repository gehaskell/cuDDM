#define INFO(x) std::cout << "[INFO] " << x << std::endl;

#include <iostream>
#include <stdlib.h>
#include <simple_fft/fft_settings.h>
#include <simple_fft/fft.h>
#include <opencv2/opencv.hpp>

using namespace cv;

const char * FFT_ERROR = NULL;

struct BufferHandle {
    float *start;
    float *end;
    float *data;
    float *load;
};

struct VideoInfo {
    int w;
    int h;
    int frame_count;
};

struct FloatArray {
    float *data;
    int size;
};

struct IntArray {
    int *data;
    int size;
};


struct QMaskStruct {
    // encapsulates data surrounding q-vector mask

    int q_count; // count of q vectors
    float * mask_fft; // FFT of q vector mask
    float * q_vector; // average q associated with each mask
    int * px_count; // normalisation factor for each mask
};


void PrintFrame(int w, int h, float * frame_ptr) {
    // Prints out a given frame
    for (int x = 0; x < w; x++) {
        for (int y = 0; y < h; y ++) {
				std::cout << frame_ptr[y * w + x] << " ";
        }
        std::cout << std::endl;
    }
    return;
}


FloatArray GenRandomData(VideoInfo info, int particle_count, int walk_size) {
    // Generates test data using basic random walk algorthim
    int w, h, size;
    w = info.w;
    h = info.h;
    size = w * h;

    float * out = new float[w * h * info.frame_count]();
    int * particles = new int[particle_count * 2]();

    // assing random starting postion for particles - dont care about overlap
    // particle will be 2 x 2 pixels hence stopping 1 before the edge of grid
    for (int i = 0; i < particle_count; i++) {
        particles[2 * i] = rand() % (w - 1);
        particles[2 * i + 1] = rand() % (h - 1);
    }

    // f for frame
    int foff, p_x, p_y;
    for (int f = 0; f < info.frame_count; f++) {
        foff = f * size;

        for (int i = 0; i < particle_count; i++) {
            p_x = particles[2 * i];
            p_y = particles[2 * i + 1];

            if (0 < p_x && p_x < w - 1 && 0 < p_y && p_y < h-1) {
                out[foff + p_y * h + p_x] = 1;
                out[foff + (p_y+1) * h + p_x] = 1;
                out[foff + p_y * h + (p_x+1)] = 1;
                out[foff + (p_y+1) * h + (p_x+1)] = 1;
            }

            particles[2 * i] += walk_size * ((rand() % 3) - 1);
            particles[2 * i + 1] += walk_size * ((rand() % 3) - 1);
        }
    }
    delete []particles;
    FloatArray output = {out, w * h * info.frame_count};
    return output;
}


BufferHandle LoadToBuffer(BufferHandle buffer, VideoInfo info, int frame_count, FloatArray to_load) {
    int frame_size = info.w * info.h;  // number of elements per frame
    int load_size = frame_size * frame_count;  // total number of elements to load

    // Arg checks - start

    // data must have enough elements to read in
    if (to_load.size < load_size) {
        std::cout << to_load.size << " < " << load_size << " Not enough elements in load array." << std::endl;
    }

    // Arg checks - end

    // First check if we have enough room in buffer - if we dont reset to start of buffer
    if (buffer.load + load_size > buffer.end) {
        std::cout << "Not enough room in buffer, returning to start" << std::endl;
        buffer.load = buffer.start;
    }

    // Copy data into buffer
    std::cout << "Copying " << load_size <<  " elements (" << frame_count << " frames) into " << buffer.load << std::endl;
    std::copy(to_load.data, to_load.data + load_size, buffer.load);

    // Update load pointer
    buffer.load += load_size;
    return buffer;
}


BufferHandle LoadVideoToBuffer(BufferHandle buffer, VideoInfo info, int frame_count, VideoCapture cap) {
    INFO("Entered video load.")

	int frame_size = info.w * info.h;  // number of elements per frame
    int load_size = frame_size * frame_count;  // total number of elements to load

	// First check if we have enough room in buffer - if we don't reset to start of buffer
    if (buffer.load + load_size > buffer.end) {
        std::cout << "Not enough room in buffer, returning to start" << std::endl;
        buffer.load = buffer.start;
    }
	Mat frame;
	for (int i = 0; i < frame_count; i++) {
		// Load next frame of capture into Mat object
		INFO("Started to read frame")
		cap >> frame;
		INFO("Read frame")

		if (frame.empty()) {
			std::cout << "Attempted to read empty frame." << std::endl;
			break;
		} else {
		    std::cout << "Copying frame " << i+1 << " into buffer." << std::endl;

			// Copying frame data can be complicated, the use of Mat.step is used to account for any padding bytes.
		    // If the frame has multiple channels we simply take the first channel

			uchar *frame_array = frame.data;
			for (int row = 0; row < info.h; row++) {
				for (int col = 0; col < info.w; col++) {
					// (mat.step)/mat.elemSize1() is the actual row length in (double) elements
					buffer.load[i*frame_size + row * info.w + col] = (float)frame_array[((frame.step)/frame.elemSize1())* col + frame.channels() * row] / 256;
				}
			}
			//PrintFrame(info.w, info.h, &buffer.load[i*frame_size]);
		}
	}

    // Update load pointer
    buffer.load += load_size;
    INFO("Returning from video load.")
    return buffer;
}


bool ABS_DIFF(float * out, float * A, float * B, int frame_size) {
    for (int i = 0; i < frame_size; i++) {
        out[i] += abs(A[i] - B[i]);
    }
    return true;
}


bool ABS_MAG_ACCUM(float * out, float * in, int frame_size) {
    // divide norm factor, abs, and square
    float norm_factor = 1 / (float)(frame_size * frame_size);

    for (int i = 0; i < frame_size; i++) {
        out[i] += (in[i] * in[i]) * norm_factor;
    }

    return true;
}

// Working //
QMaskStruct GenRadiusMasks(int q_count, int width, int height) {

	std::cout << "[INFO]	Mask Generation Started." << std::endl;
	int frame_size = width * height;
	float *test = new float[1000];
	INFO("allocated test")
    std::cout << frame_size << " " << q_count << std::endl;
    float *masks = new float[frame_size * q_count];
    std::cout << "[INFO]	Allocated space for masks." << std::endl;

    float *q_vector = new float[q_count];
    float *q_sq_vector = new float[q_count];
    int * px_count = new int[q_count]();

    for (int i=0; i < q_count; i++) {
        q_vector[i] =  1.0 + ((float)width / 1.5 - 1.0) * ((float)i / (float)(q_count - 1));
        q_sq_vector[i] = q_vector[i] * q_vector[i];
    }

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

        // Print masks - for debug
        std::cout << "q: " << q_idx << std::endl;
        PrintFrame(width, height, masks+q_idx*frame_size);
    }

    QMaskStruct mask_out;
    mask_out.mask_fft = masks;
    mask_out.px_count = px_count;
    mask_out.q_vector = q_vector;
    mask_out.q_count = q_count;
    return mask_out;
}


BufferHandle DoChunkAnalysis(BufferHandle buffer, VideoInfo info, FloatArray fft_abs_out, IntArray tau_array, int chunk_frame_count) {
    INFO("Entered chunk analysis.")
    int frame_size = info.w * info.h;  // number of elements per frame

    int *tau_vector = tau_array.data;

    float *abs_diff = new float[tau_array.size * frame_size]();
    float *fft_diff = new float[tau_array.size * frame_size]();

    int tau;
    float  *idx1, *idx2;
    float *local_abs_diff, *local_fft_diff;

    for (int tau_idx = 0; tau_idx < tau_array.size; tau_idx++)
    {

        tau = tau_vector[tau_idx];
	    std::cout << "Analysing tau " << tau << std::endl;

        // get pointers to frames corresponding to tau
        local_abs_diff = abs_diff + (tau * frame_size);
        local_fft_diff = fft_diff + (tau * frame_size);

        idx1 = frame_size * (rand() % (chunk_frame_count - tau)) + buffer.data;
        idx2 = idx1 + tau;

        ABS_DIFF(local_abs_diff, idx1, idx2, frame_size);

        // FFT(A,B,n,m,error) -- FFT from A to B where A and B are matrices (2D arrays) with n rows and m columns
        simple_fft::FFT(local_abs_diff, local_fft_diff, info.w, info.h, FFT_ERROR);

        ABS_MAG_ACCUM(fft_abs_out.data + (tau * frame_size), local_fft_diff, frame_size);
        //INFO("Finished loop.")
    }
    if (buffer.data + chunk_frame_count * frame_size  > buffer.end) {
    	buffer.data = buffer.start;
    } else {
        buffer.data += chunk_frame_count * frame_size;
    }
    INFO("Returning from chunk analysis.")
    return buffer;

}


FloatArray RunCircVideoDDM(VideoInfo info, IntArray tau_array, VideoCapture cap) {
    // Buffer information
	int buffer_frame_count = 1000;
    int chunk_frame_count = 40;
    int video_length = info.frame_count;
    int frame_size = info.w * info.h;

    // Initialise Buffer
    float * buffer_start = new float [info.w * info.h * buffer_frame_count]();
    float * buffer_end = buffer_start + (info.w * info.h * buffer_frame_count);
    BufferHandle buffer = {buffer_start, buffer_end, buffer_start, buffer_start};

    // Initialise Output
    int fft_output_size = frame_size * tau_array.size;
    float * fft_output_data = new float[fft_output_size]();
    FloatArray fft_output = {fft_output_data, fft_output_size};

    // Main Loop
    Mat frame;
    int chunks_analysed = 0;
    while (info.frame_count >= chunk_frame_count) {
        chunks_analysed++;
        buffer = LoadVideoToBuffer(buffer, info, chunk_frame_count, cap);
        buffer = DoChunkAnalysis(buffer, info, fft_output, tau_array, chunk_frame_count);
        info.frame_count -= chunk_frame_count;
    }
    std::cout << "[INFO]	Main Loop ended." << std::endl;

    delete []buffer_start;

    // Analysis
    float norm_factor = 1.0 / (float) chunks_analysed;
    QMaskStruct mask = GenRadiusMasks(5, info.w, info.h);
    std::cout << "[INFO]	Generated masks." << std::endl;

    float * iq_tau = new float[tau_array.size * mask.q_count]();
    float * tau_frame;

    for (int tau_idx = 0; tau_idx < tau_array.size; tau_idx++) {
        tau_frame = fft_output_data + (frame_size * tau_idx);
        //PrintFrame(info.w, info.h, tau_frame);

        for (int q_idx = 0; q_idx < mask.q_count; q_idx++) {
            if (mask.px_count[q_idx] == 0) {
                iq_tau[q_idx * tau_array.size + tau_idx] = 1;
            } else {
                std::cout << "q: " << q_idx << "\t t: " << tau_idx << std::endl;
                //PrintFrame(info.w, info.h, mask.mask_fft+q_idx*frame_size);

                for (int i = 0; i < frame_size; i++) {
                	iq_tau[q_idx * tau_array.size + tau_idx] += (float)mask.mask_fft[q_idx*frame_size + i] * tau_frame[i] * norm_factor;
                }
            }
        }
    }
    //delete []buffer_start;
    //delete []fft_data;
    FloatArray iq_tau_arr = {iq_tau, tau_array.size * mask.q_count};
    return iq_tau_arr;
}


int main(int argc, char **argv) {
    VideoInfo info = {16, 16, 50};
	VideoCapture cap(0);

    int tau_vec [10] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    IntArray tau_arr = {&tau_vec[0], 10};

    FloatArray out = RunCircVideoDDM(info, tau_arr, cap);

    for (int i = 0; i < out.size; i++)
    {
       std::cout << out.data[i] << std::endl;
    }


}