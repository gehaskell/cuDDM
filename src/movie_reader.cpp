#include "movie_reader.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

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

	if (found) {
		printf("Found magic. \n");
	} else {
		printf("Not Found magic. \n");
	}

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

	printf("[Initialised frame] x-size: %d, y-size %d, pixel-mode %d\n", out.size_x, out.size_y, camera_frame.pixelmode);
	return out;
}

void loadFileToHost (FILE* moviefile, unsigned char *h_buffer, video_info_struct vid_info, int frame_count) {
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

}

