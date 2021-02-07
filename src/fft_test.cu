#include <cufft.h>
#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>
#define FFTSIZE 512
#define DEBUG 0

typedef size_t sf_count_t;

void process_data(float *h_in_data_dynamic, sf_count_t samples, int channels) {
  int nSamples = (int)samples;
  int DATASIZE = FFTSIZE;
  int batch = nSamples / DATASIZE;

  cufftHandle plan;

  cufftReal *d_in_data;
  cudaMalloc((void**)&d_in_data, sizeof(cufftReal) * nSamples);
  cudaMemcpy(d_in_data, (cufftReal*)h_in_data_dynamic, sizeof(cufftReal) * nSamples, cudaMemcpyHostToDevice);

  cufftComplex *data;
  cudaMalloc((void**)&data, sizeof(cufftComplex) * batch * (DATASIZE/2 + 1));

  cufftComplex *hostOutputData = (cufftComplex*)malloc((DATASIZE / 2 + 1) * batch * sizeof(cufftComplex));

  if (cudaGetLastError() != cudaSuccess) {
    fprintf(stderr, "Cuda error: Failed to allocate\n");
    return;
  }

  int rank = 1;                           // --- 1D FFTs
  int n[] = { DATASIZE };                 // --- Size of the Fourier transform
  int istride = 1, ostride = 1;           // --- Distance between two successive input/output elements
  int idist = DATASIZE, odist = (DATASIZE / 2) + 1; // --- Distance between batches
  int inembed[] = { 0 };                  // --- Input size with pitch (ignored for 1D transforms)
  int onembed[] = { 0 };                  // --- Output size with pitch (ignored for 1D transforms)

  if(cufftPlanMany(&plan, rank, n,
              inembed, istride, idist,
              onembed, ostride, odist, CUFFT_R2C, batch) != CUFFT_SUCCESS){
    fprintf(stderr, "CUFFT error: Plan failed");
    return;
  }

/* Use the CUFFT plan to transform the signal in place. */
  if (cufftExecR2C(plan, d_in_data, data) != CUFFT_SUCCESS) {
    fprintf(stderr, "CUFFT error: ExecR2C Forward failed");
    return;
  }

  cudaMemcpy(hostOutputData, data, ((DATASIZE / 2) + 1) * batch * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
  if (cudaGetLastError() != cudaSuccess) {
    fprintf(stderr, "Cuda error: Failed results copy\n");
    return;
  }

  float *spectrum = (float *)malloc((DATASIZE/2)*sizeof(float));
  for (int j = 0; j < (DATASIZE/2); j++) spectrum[j] = 0.0f;
  for (int i=0; i < batch; i++)
    for (int j=0; j < (DATASIZE / 2 + 1); j++){
#if DEBUG
        printf("%i %i %f %f\n", i, j, hostOutputData[i*(DATASIZE / 2 + 1) + j].x, hostOutputData[i*(DATASIZE / 2 + 1) + j].y);
#endif
        // compute spectral magnitude
        // note that cufft induces a scale factor of FFTSIZE
        if (j < (DATASIZE/2)) spectrum[j] += sqrt(pow(hostOutputData[i*(DATASIZE/2 +1) +j].x, 2) + pow(hostOutputData[i*(DATASIZE/2 +1) +j].y, 2))/(float)(batch*DATASIZE);
        }
  //assumes Fs is half of FFTSIZE, or we could pass Fs separately
  printf("Spectrum\n Hz:   Magnitude:\n");
  for (int j = 0; j < (DATASIZE/2); j++) printf("%.3f %.3f\n", j/2.0f, spectrum[j]);

  cufftDestroy(plan);
  cudaFree(data);
  cudaFree(d_in_data);
}

int main(){

  const int nsets = 20;
  const float sampling_rate = FFTSIZE/2;
  const float amplitude = 1.0;
  const float fc1 = 6.0;
  const float fc2 = 4.5;
  float *my_data;

  my_data = (float *)malloc(nsets*FFTSIZE*sizeof(float));
  //generate synthetic data that is a mix of 2 sine waves at fc1 and fc2 Hz
  for (int i = 0; i < nsets*FFTSIZE; i++)
    my_data[i] = amplitude*sin(fc1*(6.283/sampling_rate)*i)
               + amplitude*sin(fc2*(6.283/sampling_rate)*i);

  process_data(my_data, nsets*FFTSIZE, 1);
  return 0;
}
