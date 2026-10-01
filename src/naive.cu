#include<stdio.h>
#include "../include/convolution.h"
#include<cuda_runtime.h>
#include "../include/cuda_utils.cuh"
#include "../include/filter.cuh"

#define RADIUS (FILTER_SIZE-1)/2
#define BLOCK_SIZE 32

__global__ void convolutionKernel(const float* d_input, float* d_output, const float* d_filter,
    int width, int height, int channels)
{
    int outrow = blockIdx.y * blockDim.y + threadIdx.y;
    int outcol = blockIdx.x * blockDim.x + threadIdx.x;

    if (outrow >=0 && outrow < height && outcol >=0 && outcol < width)
    {
        float Pvalue = 0.0f;
        for (int frow=0; frow<FILTER_SIZE; frow++)
        {
            for (int fcol=0; fcol<FILTER_SIZE; fcol++)
            {
                int inrow = outrow + frow - RADIUS;
                int incol = outcol + fcol - RADIUS;
                if (inrow >=0 && inrow<height && incol >=0 && incol<width)
                {
                    int r = inrow * width + incol;
                    int g = r + width * height;
                    int b = g + width * height;
                    int filterIndex = frow * FILTER_SIZE + fcol;
                    Pvalue += d_input[r] * d_filter[filterIndex];
                    Pvalue += d_input[g] * d_filter[filterIndex + FILTER_ELEMENTS];
                    Pvalue += d_input[b] * d_filter[filterIndex + 2 * FILTER_ELEMENTS];
                }
            }
        }
        d_output[outrow * width + outcol] = Pvalue;
    }
}

float convolutionNaive(
    const float* d_input,
    float* d_output,
    int width,
    int height,
    int channels
) {
    
    
    float* d_filter;
    CUDA_CHECK(cudaMalloc(&d_filter, CHANNELS * FILTER_ELEMENTS * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_filter, FILTER1, CHANNELS * FILTER_ELEMENTS * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaDeviceSynchronize());

    convolutionKernel<<<gridSize, blockSize>>>(d_input, d_output, d_filter, width, height, channels);
    
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
    CUDA_CHECK(cudaFree(d_filter));

    return milliseconds;
}
