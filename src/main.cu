#include <cuda.h>

void AnalyseBuffer()

void RunDDM() {
	//
	chunks_per_buffer = 5;
	buffer_frame_length = 30;

	// Initialise workspace
	int w, int h, int frame_count;
	frame_count = chunks_per_buffer * buffer_frame_length;

	int mem_size = sizeof(float) * w * h * frame_count;

	float *d_data1, *d_data2;
	checkCudaErrors(cudaMalloc((void**) &d_data1, mem_size));
	checkCudaErrors(cudaMalloc((void**) &d_data2, mem_size));

	cudaStream_t stream1, stream2;
	cudaStreamCreate(stream1);
	cudaStreamCreate(stream2);

	cudaEvent_t data1_read, data2_read;

	LoadBuffer(stream1, d_data1);
	for (;;) {
		Analyse(stream1, d_data1);
		LoadBuffer(stream2, d_data2);

		//cudaStreamWaitEvent(stream,event)
		//cudaStreamWaitEvent(stream,event)

		LoadBuffer(stream1, d_data1);
		Analyse(stream2, d_data2);

		//cudaStreamWaitEvent(stream,event)
		//cudaStreamWaitEvent(stream,event)
	}
	Analyse(stream1, d_data1);
}

int main(int argc, char **argv) {

}
