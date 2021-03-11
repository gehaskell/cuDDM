#include "movie_reader.hpp"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <nvToolsExt.h>
#include <opencv2/opencv.hpp>
#include <string>

// Common camera defines
#define CAMERA_MOVIE_MAGIC 0x496d6554 // TemI
#define CAMERA_MOVIE_VERSION 1
#define CAMERA_TYPE_IIDC	1
#define CAMERA_TYPE_ANDOR	2
#define CAMERA_TYPE_XIMEA	3
#define CAMERA_PIXELMODE_MONO_8		1
#define CAMERA_PIXELMODE_MONO_16BE	2 // Big endian
#define CAMERA_PIXELMODE_MONO_16LE	3 // Little endian

#define IIDC_MOVIE_HEADER_LENGTH 172
#define ANDOR_MOVIE_HEADER_LENGTH 128
#define XIMEA_MOVIE_HEADER_LENGTH 240

video_info_struct initFile(FILE *moviefile) {
	// Find start of bin data using magic value
	// Extract general frame information
	// Extract camera specific frame information + seek back to start of binary data

	// Find start of bin data using magic value
	long magic_offset = 0;
	bool found = false;
	uint32_t magic_val;

	while (fread(&magic_val, sizeof(uint32_t), 1, moviefile) == 1) {
		if (magic_val == CAMERA_MOVIE_MAGIC) {
			found = true;
			break;
		}
		magic_offset++;
		fseek(moviefile, magic_offset, SEEK_SET);
	}

	fseek(moviefile, magic_offset, SEEK_SET);

//	if (found) {
//		printf("Found magic. \n");
//	} else {
//		printf("Not Found magic. \n");
//	}

	// Extract general frame information
	camera_save_struct camera_frame;
	if (fread( &camera_frame, sizeof( struct camera_save_struct ), 1, moviefile ) != 1) {
		fprintf(stderr, "Corrupted header at offset %lu\n", ftell(moviefile));
		exit(EXIT_FAILURE);
	}

		// Check read is as expected
	if (camera_frame.magic != CAMERA_MOVIE_MAGIC) {
		fprintf(stderr, "Wrong magic at offset %lu\n", ftell(moviefile));
		exit(EXIT_FAILURE);
	}

	if (camera_frame.version != CAMERA_MOVIE_VERSION) {
		fprintf(stderr, "Unsupported movie version %u\n", camera_frame.version);
		exit(EXIT_FAILURE);
	}

		// Go to the beginning of the frame for easier reading
	fseek(moviefile, -sizeof(struct camera_save_struct), SEEK_CUR);

	// Extract camera specific frame information
	uint32_t size_x, size_y;

	switch (camera_frame.type) {
		case CAMERA_TYPE_IIDC:
			struct iidc_save_struct iidc_frame;
			if (fread(&iidc_frame, IIDC_MOVIE_HEADER_LENGTH, 1, moviefile ) != 1) {
				fprintf(stderr, "Corrupted header at offset %lu\n", ftell(moviefile));
				exit(EXIT_FAILURE);
			}
			size_x = iidc_frame.i_size_x;
			size_y = iidc_frame.i_size_y;

			fseek(moviefile, -IIDC_MOVIE_HEADER_LENGTH, SEEK_CUR);
			break;

		case CAMERA_TYPE_ANDOR:
			struct andor_save_struct andor_frame;
			if (fread(&andor_frame, ANDOR_MOVIE_HEADER_LENGTH, 1, moviefile ) != 1) {
				fprintf(stderr, "Corrupted header at offset %lu\n", ftell(moviefile));
				exit(EXIT_FAILURE);
			}
			size_x = (andor_frame.a_x_end - andor_frame.a_x_start + 1) / andor_frame.a_x_bin;
			size_y = (andor_frame.a_y_end - andor_frame.a_y_start + 1) / andor_frame.a_y_bin;

			fseek(moviefile, -ANDOR_MOVIE_HEADER_LENGTH, SEEK_CUR);
			break;

		case CAMERA_TYPE_XIMEA:
			struct ximea_save_struct ximea_frame;
			if (fread( &ximea_frame, XIMEA_MOVIE_HEADER_LENGTH, 1, moviefile) != 1) {
				fprintf(stderr, "Corrupted header at offset %lu\n", ftell(moviefile));
				exit(EXIT_FAILURE);
			}
			size_x = ximea_frame.x_size_x;
			size_y = ximea_frame.x_size_y;

			fseek(moviefile, -XIMEA_MOVIE_HEADER_LENGTH, SEEK_CUR);
			break;

		default:
			fprintf(stderr, "Unsupported camera.\n" );
			exit( EXIT_FAILURE );
			break;
	}

	int bpp;
	if (camera_frame.pixelmode == CAMERA_PIXELMODE_MONO_8)
		bpp = 1;
	else
		bpp = 2;


	video_info_struct out;
	out.size_x = (int) size_x;
	out.size_y = (int) size_y;
	out.type = camera_frame.type;
	out.pixelmode = camera_frame.pixelmode;
	out.length_data = camera_frame.length_data;
	out.bpp = bpp;

	//printf("[Initialised frame] frame x-size: %d, frame y-size %d, pixel-mode %d\n", out.size_x, out.size_y, camera_frame.pixelmode);
	return out;
}

void loadMovieToHost (FILE* moviefile, unsigned char *h_buffer, video_info_struct vid_info, int frame_count) {
	nvtxRangePush(__FUNCTION__); // to track video loading times in nvvp

	// Read header
	// Read data

	int frame_index = 0;
	// Data depth

	int bpp = vid_info.bpp;


	while (frame_index < frame_count) {
		//printf("Loading frame %d\n", frame_index);
		// Read header
		switch (vid_info.type) {
			case CAMERA_TYPE_IIDC:
				struct iidc_save_struct iidc_frame;
				if (fread(&iidc_frame, IIDC_MOVIE_HEADER_LENGTH, 1, moviefile ) != 1) {
					fprintf(stderr, "Corrupted header at offset %lu\n", ftell(moviefile));
					exit(EXIT_FAILURE);
				}

				break;

			case CAMERA_TYPE_ANDOR:
				struct andor_save_struct andor_frame;
				if (fread(&andor_frame, ANDOR_MOVIE_HEADER_LENGTH, 1, moviefile ) != 1) {
					fprintf(stderr, "Corrupted header at offset %lu\n", ftell(moviefile));
					exit(EXIT_FAILURE);
				}

				break;

			case CAMERA_TYPE_XIMEA:
				struct ximea_save_struct ximea_frame;
				if (fread( &ximea_frame, XIMEA_MOVIE_HEADER_LENGTH, 1, moviefile) != 1) {
					fprintf(stderr, "Corrupted header at offset %lu\n", ftell(moviefile));
					exit(EXIT_FAILURE);
				}

				break;

			default:
				fprintf(stderr, "Unsupported camera.\n" );
				exit( EXIT_FAILURE );
				break;
		}

		unsigned char *h_current = h_buffer + vid_info.size_x * vid_info.size_y * bpp * frame_index;

		// Read data
		if (fread(h_current, vid_info.length_data, 1, moviefile ) != 1 ) {
			fprintf(stderr, "Corrupted data at offset %lu\n", ftell(moviefile));
			exit(EXIT_FAILURE);
		}

		//printf("Read Frame (%d * %d * %d = %d)\n", (int)vid_info.size_x, (int)vid_info.size_y, bpp, vid_info.length_data);
		frame_index++;
	}
    nvtxRangePop();
}


void loadCaptureToHost(cv::VideoCapture cap, unsigned char *h_buffer, frame_info info, int frame_count) {
	// TODO as we have true frame size we could run analysis on just the bit we will be using
	nvtxRangePush(__FUNCTION__); // to track video loading times in nvvp

	//printf("load video (%d frames) (w: %d, h: %d)\n", frame_count, w, h);

	int num_elements = info.in_width * info.in_height;

	cv::Mat input_img; //, grayscale_img;

	// There is some problems with the image type we are using - though some effort was put into switching to a
	// more generic image format, more thought is required therefore switch to just dealing with 3 channel uchars
	// look at http://ninghang.blogspot.com/2012/11/list-of-mat-type-in-opencv.html and
	// https://docs.opencv.org/3.4/d3/d63/classcv_1_1Mat.html#aa5d20fc86d41d59e4d71ae93daee9726 for more info.


	for (int frame_idx = 0; frame_idx < frame_count; frame_idx++) {
		//std::cout << "Loaded frame " << frame_idx << std::endl;

		cap >> input_img;

		if (input_img.empty()) {
			fprintf(stderr,"Video frame is empty");
		}

		//input_img.convertTo(grayscale_img, CV_32FC1); // covert to grayscale image

		if (input_img.type() != 16) {
			fprintf(stderr,"Non standard image format detected, may cause unexpected behaviour, image type : %d", input_img.type());
		}

	    // imshow("Input", input_img);
	    // waitKey(0);

		int cols = input_img.cols, rows = input_img.rows;

		int in_width = info.in_width;
		int in_height = info.in_height;

		int out_width = info.out_width;
		int out_height= info.out_height;

		// we take short cut by only analysing the pixels we will actually need - this means that the buffer will be padded with junk values
		for (int y = 0; y < out_height; y++) {
			for (int x = 0; x < out_width; x++) {
				// Using img.at<>8
				h_buffer[frame_idx * num_elements + y * in_width + x] =  (unsigned char) input_img.data[((input_img.step)/input_img.elemSize1())* y + input_img.channels() * x];
			}
		}
	}

    nvtxRangePop();
}


