#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <stdbool.h>
#include <libgen.h>

//
// Common camera defines
//
#define CAMERA_MOVIE_MAGIC 0x496d6554 // TemI
#define CAMERA_MOVIE_VERSION 1
#define CAMERA_TYPE_IIDC	1
#define CAMERA_TYPE_ANDOR	2
#define CAMERA_TYPE_XIMEA	3
#define CAMERA_PIXELMODE_MONO_8		1
#define CAMERA_PIXELMODE_MONO_16BE	2 // Big endian
#define CAMERA_PIXELMODE_MONO_16LE	3 // Little endian

//
// IIDC defines
//
#define IIDC_MOVIE_HEADER_LENGTH 172
// Feature modes
#define IIDC_FEATURE_MODE_OFF ( 1<<0 )
#define IIDC_FEATURE_MODE_RELATIVE ( 1<<1 )
#define IIDC_FEATURE_MODE_ABSOLUTE ( 1<<2 )
#define IIDC_FEATURE_MODE_AUTO ( 1<<3 )
#define IIDC_FEATURE_MODE_ONEPUSH ( 1<<4 )
#define IIDC_FEATURE_MODE_ADVANCED ( 1<<5 )
// Trigger
#define IIDC_TRIGGER_INTERNAL  -1
#define IIDC_TRIGGER_EXTERNAL0 0
#define IIDC_TRIGGER_EXTERNAL1 1
#define IIDC_TRIGGER_EXTERNAL15 7

//
// Andor defines
//
#define ANDOR_MOVIE_HEADER_LENGTH 128
// VS Speeds
#define ANDOR_VALUE_VS_SPEED_MIN 4
#define ANDOR_VALUE_VS_SPEED_MAX 0
#define ANDOR_VALUE_VS_SPEED_0_3 0
#define ANDOR_VALUE_VS_SPEED_0_5 1
#define ANDOR_VALUE_VS_SPEED_0_9 2
#define ANDOR_VALUE_VS_SPEED_1_7 3
#define ANDOR_VALUE_VS_SPEED_3_3 4
// VS Amplitudes
#define ANDOR_VALUE_VS_AMPLITUDE_MIN 0
#define ANDOR_VALUE_VS_AMPLITUDE_MAX 4
#define ANDOR_VALUE_VS_AMPLITUDE_0 0
#define ANDOR_VALUE_VS_AMPLITUDE_1 1
#define ANDOR_VALUE_VS_AMPLITUDE_2 2
#define ANDOR_VALUE_VS_AMPLITUDE_3 3
#define ANDOR_VALUE_VS_AMPLITUDE_4 4
// Shutter
#define ANDOR_VALUE_SHUTTER_AUTO 0
#define ANDOR_VALUE_SHUTTER_OPEN 1
#define ANDOR_VALUE_SHUTTER_CLOSE 2
// Cooler
#define ANDOR_VALUE_COOLER_OFF 0
#define ANDOR_VALUE_COOLER_ON 1
// Cooler mode
#define ANDOR_VALUE_COOLER_MODE_RETURN 0
#define ANDOR_VALUE_COOLER_MODE_MAINTAIN 1
// Fan
#define ANDOR_VALUE_FAN_FULL 0
#define ANDOR_VALUE_FAN_LOW 1
#define ANDOR_VALUE_FAN_OFF 2
// ADC
#define ANDOR_VALUE_ADC_14BIT 0
#define ANDOR_VALUE_ADC_16BIT 1
// Amplifier
#define ANDOR_VALUE_AMPLIFIER_EM 0
#define ANDOR_VALUE_AMPLIFIER_CON 1
// Preamp gain
#define ANDOR_VALUE_PREAMP_GAIN_1_0 0
#define ANDOR_VALUE_PREAMP_GAIN_2_4 1
#define ANDOR_VALUE_PREAMP_GAIN_5_1 2
// Trigger
#define ANDOR_VALUE_TRIGGER_INTERNAL  0
#define ANDOR_VALUE_TRIGGER_EXTERNAL 1
#define ANDOR_VALUE_TRIGGER_FAST_EXTERNAL -1 // Combination of external and SetFastExtTrigger
#define ANDOR_VALUE_TRIGGER_EXTERNAL_START 6
#define ANDOR_VALUE_TRIGGER_EXTERNAL_EXPOSURE  7
#define ANDOR_VALUE_TRIGGER_SOFTWARE  10

//
// Ximea defines
//
#define XIMEA_MOVIE_HEADER_LENGTH 240
#define XIMEA_TRIGGER_INTERNAL 0
#define XIMEA_TRIGGER_EXTERNAL 1
#define XIMEA_TRIGGER_SOFTWARE 3

//
// Common camera frame struct
//
struct camera_save_struct {
	//
	// Common stuff
	//
	uint32_t magic; // 'AndO'
	uint32_t version;
	uint32_t type; // Camera type
	uint32_t pixelmode; // Pixel mode
	uint32_t length_header; // Header data in bytes ( Everything except image data )
	uint32_t length_data; // Total data length in bytes;
};

//
// IIDC movie frame struct
//
union iidc_save_feature_value {
	uint32_t value;
	float absvalue;
};

struct iidc_save_struct {
	//
	// Common stuff
	//
	uint32_t magic; // 'TemI'
	uint32_t version;
	uint32_t type; // Camera type
	uint32_t pixelmode; // Pixel mode
	uint32_t length_header; // Header data in bytes ( Everything except image data )
	uint32_t length_data; // Total data length in bytes;

	//
	// Camera specific stuff
	//
	// Camera properties
	uint64_t i_guid;
	uint32_t i_vendor_id;
	uint32_t i_model_id;

	// Frame properties
	uint32_t i_video_mode;
	uint32_t i_color_coding;

	uint64_t i_timestamp; // microseconds

	uint32_t i_size_x_max; // Sensor size
	uint32_t i_size_y_max;
	uint32_t i_size_x; // Selected region
	uint32_t i_size_y;
	uint32_t i_pos_x;
	uint32_t i_pos_y;

	uint32_t i_pixnum; // Number of pixels
	uint32_t i_stride; // Number of bytes per image line
	uint32_t i_data_depth;  // Number of bits per pixel.

	uint32_t i_image_bytes; // Number of bytes used for the image (image data only, no padding)
	uint64_t i_total_bytes; // Total size of the frame buffer in bytes. May include packet multiple padding and intentional padding (vendor specific)

	// Features
	uint32_t i_brightness_mode; // Current mode
	union iidc_save_feature_value i_brightness; // Can be also float if mode is IIDC_FEATURE_MODE_ABSOLUTE (1<<2)

	uint32_t i_exposure_mode;
	union iidc_save_feature_value i_exposure;

	uint32_t i_gamma_mode;
	union iidc_save_feature_value i_gamma;

	uint32_t i_shutter_mode;
	union iidc_save_feature_value i_shutter;

	uint32_t i_gain_mode;
	union iidc_save_feature_value i_gain;

	uint32_t i_temperature_mode;
	union iidc_save_feature_value i_temperature;

	uint32_t i_trigger_delay_mode;
	union iidc_save_feature_value i_trigger_delay;

	int32_t i_trigger_mode;

	// Advanced features
	uint32_t i_avt_channel_balance_mode;
	int32_t i_avt_channel_balance;

	// Image data
	uint8_t *data;
} __attribute__((__packed__));

//
// Andor movie frame struct
//
struct andor_save_struct {
	//
	// Common stuff
	//
	uint32_t magic; // 'TemI'
	uint32_t version;
	uint32_t type; // Camera type
	uint32_t pixelmode; // Pixel mode
	uint32_t length_header; // Header data in bytes ( Everything except image data )
	uint32_t length_data; // Total data length in bytes;

	//
	// Camera specific stuff
	//
	// Timestamp
	uint64_t a_timestamp_sec;
	uint64_t a_timestamp_nsec;

	// Frame properties
	int32_t a_x_size_max; // Sensor size
	int32_t a_y_size_max;
	int32_t a_x_start; // Selected size and positions
	int32_t a_x_end;
	int32_t a_y_start;
	int32_t a_y_end;
	int32_t a_x_bin;
	int32_t a_y_bin;

	// Camera settings
	int32_t a_ad_channel; // ADC
	int32_t a_amplifier; // EM or classical preamplifier
	int32_t a_preamp_gain; // Preamplifier gain
	int32_t a_em_gain; // EM gain
	int32_t a_hs_speed; // HS speed
	int32_t a_vs_speed; // VS speed
	int32_t a_vs_amplitude; // VS amplitude
	float a_exposure; // Exposure time in seconds
	int32_t a_shutter; // Shutter
	int32_t a_trigger; // Trigger
	int32_t a_temperature; // Temperature
	int32_t a_cooler; // Cooler
	int32_t a_cooler_mode; // Cooler mode
	int32_t a_fan; // Fan

	//
	// Image data
	//
	uint8_t *data;
} __attribute__((__packed__));

//
// Ximea movie frame struct
//
struct ximea_save_struct {
	//
	// Common stuff
	//
	uint32_t magic; // 'TemI'
	uint32_t version;
	uint32_t type; // Camera type
	uint32_t pixelmode; // Pixel mode
	uint32_t length_header; // Header data in bytes ( Everything except image data )
	uint32_t length_data; // Total data length in bytes;

	//
	// Camera specific stuff
	//
	char x_name[100]; // Camera name
	uint32_t x_serial_number; // Serial number

	// Timestamp
	uint64_t x_timestamp_sec;
	uint64_t x_timestamp_nsec;

	// Sensor
	uint32_t x_size_x_max; // Sensor size
	uint32_t x_size_y_max;
	uint32_t x_size_x; // Selected region
	uint32_t x_size_y;
	uint32_t x_pos_x;
	uint32_t x_pos_y;

	//
	// Features
	//
	uint32_t x_exposure; // Exposure [us]
	float x_gain; // Gain [dB]
	uint32_t x_downsampling; // Downsampling, 1 1x1, 2 2x2
	uint32_t x_downsampling_type; // 0 binning, 1 skipping
	uint32_t x_bpc; // Bad Pixels Correction, 0 disabled, 1 enabled
	uint32_t x_lut; // Look up table, 0 disabled, 1 enabled
	uint32_t x_trigger; // Trigger

	// Automatic exposure/gain
	uint32_t x_aeag; // 0 disabled, 1 enabled
	float x_aeag_exposure_priority; // Priority of exposure versus gain 0.0 1.0
	uint32_t x_aeag_exposure_max_limit; // Maximum exposure time [us]
	float x_aeag_gain_max_limit; // Maximum gain [dB]
	uint32_t x_aeag_average_intensity; // Average intensity level [%]

	// High dynamic range
	uint32_t x_hdr; // 0 disabled, 1 enabled
	uint32_t x_hdr_t1; // Exposure time of the first slope [us]
	uint32_t x_hdr_t2; // Exposure time of the second slope [us]
	uint32_t x_hdr_t3; // Exposure time of the third slope [us]
	uint32_t x_hdr_kneepoint1; // Kneepoint 1 [%]
	uint32_t x_hdr_kneepoint2; // Kneepoint 2 [%]

	//
	// Image data
	//
	uint8_t *data;
} __attribute__((__packed__));

#define NAMELENGTH 100

camera_save_struct loadToHost(char *h_image_buffer, int frame_count, int w, int h) {

	// Setup
	char filename[] = "FILENAME - FILL IN";
	FILE *moviefile; // file-stream for movie file

	// Read moviefile
	if ( !(moviefile = fopen(filename, "rb" ))) {
		printf( "Couldn't open movie file.\n" );
		exit( EXIT_FAILURE );
	}

	// Find the beginning of binary data, it won't work if "TemI" is written in the header.
	long offset = 0;
	bool found = false;
	uint32_t magic;

	while (fread( &magic, sizeof(uint32_t), 1, moviefile ) == 1 ) {
		if (magic == CAMERA_MOVIE_MAGIC) {
			found = true;
			break;
		}
		offset++;
	}

	fseek(moviefile, offset, SEEK_SET);
	camera_save_struct camera_frame;

	// Main read
	// If we have found magic value
	if (found) {
		int frame_index = 0;

		// Read the camera frame information directly into struct
		while (frame_index<frame_count && fread( &camera_frame, sizeof( struct camera_save_struct ), 1, moviefile ) == 1 ) { // lazy evaluation prevents reading too many frames
			// Check read is as expected
			if (camera_frame.magic != CAMERA_MOVIE_MAGIC) {
				offset = ftell(moviefile);
				fprintf(stderr, "Wrong magic at offset %lu\n", offset );
				exit( EXIT_FAILURE );
			}

			if (camera_frame.version != CAMERA_MOVIE_VERSION) {
				fprintf(stderr, "Unsupported version %u\n", camera_frame.version );
				exit( EXIT_FAILURE );
			}

			// Go to the beginning of the frame for easier reading
			fseek(moviefile, -sizeof( struct camera_save_struct ), SEEK_CUR );


			struct andor_save_struct andor_frame;
			struct ximea_save_struct ximea_frame;


			// Read the header
			uint32_t size_x, size_y;

			switch (camera_frame.type) {
				case CAMERA_TYPE_IIDC:
					struct iidc_save_struct iidc_frame;
					if ( fread( &iidc_frame, IIDC_MOVIE_HEADER_LENGTH, 1, moviefile ) != 1 ) {
						offset = ftell( moviefile );
						printf( "Corrupted header at offset %lu\n", offset );
						exit( EXIT_FAILURE );
					}
					size_x = iidc_frame.i_size_x;
					size_y = iidc_frame.i_size_y;
					//printf( "%lu\n", iidc_frame.i_timestamp );
					break;
				case CAMERA_TYPE_ANDOR:
					if ( fread( &andor_frame, ANDOR_MOVIE_HEADER_LENGTH, 1, moviefile ) != 1 ) {
						offset = ftell( moviefile );
						printf( "Corrupted header at offset %lu\n", offset );
						exit( EXIT_FAILURE );
					}
					size_x = ( andor_frame.a_x_end - andor_frame.a_x_start + 1 ) / andor_frame.a_x_bin;
					size_y = ( andor_frame.a_y_end - andor_frame.a_y_start + 1 ) / andor_frame.a_y_bin;
					break;
				case CAMERA_TYPE_XIMEA:
					if ( fread( &ximea_frame, XIMEA_MOVIE_HEADER_LENGTH, 1, moviefile ) != 1 ) {
							offset = ftell( moviefile );
							printf( "Corrupted header at offset %lu\n", offset );
							exit( EXIT_FAILURE );
						}
					size_x = ximea_frame.x_size_x;
					size_y = ximea_frame.x_size_y;
					break;
				default:
					printf( "Unsupported camera.\n" );
					exit( EXIT_FAILURE );
					break;

			}
			// Data depth
			int bpp;
			if (camera_frame.pixelmode == CAMERA_PIXELMODE_MONO_8)
				bpp = 1;
			else
				bpp = 2;

			char *h_current = h_image_buffer + size_x * size_y * frame_index;

			// Read the data
			printf("%d * %d * %d = %d", size_x, size_y, bpp, camera_frame.length_data);

 			if (fread(h_current, w * h * bpp, 1, moviefile ) != 1 ) {
				offset = ftell( moviefile );
				fprintf(stderr, "Corrupted data at offset %lu\n", offset);
				exit(EXIT_FAILURE);
			}

			// Convert to little endian data if needed
			if ( camera_frame.pixelmode == CAMERA_PIXELMODE_MONO_16BE ) {
				char c;
				for (int j = 0; j < size_x * size_y; j++ ) {
					c = h_current[2*j];
					h_current[2*j] = h_current[2*j+1];
					h_current[2*j+1] = c;
				}
			}

			frame_index++;
		}
	}
	return camera_frame;
}

