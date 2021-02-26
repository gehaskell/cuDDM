#include <stdint.h>
#include <stdio.h>

#ifndef MOVIE_READER_H
#define MOVIE_READER_H

//
// Common camera frame struct
////
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

struct video_info_struct {
	int size_x;
	int size_y;
	uint32_t type;
	uint32_t pixelmode;
	uint32_t length_data; // Total data length in bytes;
	int bpp;
};

void loadFileToHost (FILE* moviefile, unsigned char *h_buffer, video_info_struct vid_info, int frame_count);
video_info_struct initFile(FILE *moviefile);


#endif
