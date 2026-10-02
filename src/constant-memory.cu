#include<stdio.h>
#include "../include/convolution.h"
#include<cuda_runtime.h>
#include "../include/cuda_utils.cuh"
#include "../include/filter.cuh"

#define RADIUS (FILTER_SIZE-1)/2
#define BLOCK_SIZE 32

__constant__ float filter_d[CHANNELS*FILTER_ELEMENTS];

__global__ void convolkernel(const float *d_input, float* d_output, int width, int height)
{
    int outcol = blockIdx.x*blockDim.x + threadIdx.x;
    int outrow = blockIdx.y*blockDim.y + threadIdx.y;

    if(outrow>=0 && outrow<height && outcol>=0 && outcol<width)
    {
        float Pvalue = 0.0f;
        for(int frow=0;frow<FILTER_SIZE;frow++)
        {
            for(int fcol=0;fcol<FILTER_SIZE;fcol++)
            {
                int row = outrow - RADIUS + frow;
                int col = outcol - RADIUS + fcol;
                if(row>=0 && row<height && col>=0 && col<width)
                {
                    int r = row * width + col;
                    int g = r + width * height;
                    int b = g + width * height;
                    int filterIndex = frow * FILTER_SIZE + fcol;
                    Pvalue += d_input[r] * filter_d[filterIndex];
                    Pvalue += d_input[g] * filter_d[filterIndex + FILTER_ELEMENTS];
                    Pvalue += d_input[b] * filter_d[filterIndex + 2 * FILTER_ELEMENTS];
                
                }
            }
        }
        d_output[outrow*width+outcol] = Pvalue;

    }
}
float convolutionConstant(
    const float* d_input,
    float* d_output,
    int width,
    int height,
    int channels
)
{
    CUDA_CHECK(cudaMemcpyToSymbol(filter_d, FILTER1, CHANNELS * FILTER_ELEMENTS * sizeof(float)));

    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaDeviceSynchronize());

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    convolkernel<<<gridSize,blockSize>>>(d_input, d_output, width, height);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &milliseconds,
        start,
        stop
    ));

    CUDA_CHECK(cudaGetLastError());

    return milliseconds;

}