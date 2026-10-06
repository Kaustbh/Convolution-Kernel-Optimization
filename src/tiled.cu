#include<stdio.h>
#include "../include/convolution.h"
#include<cuda_runtime.h>
#include "../include/cuda_utils.cuh"
#include "../include/filter.cuh"

#define RADIUS (FILTER_SIZE-1)/2
#define INPUT_TILE_SIZE 32
#define OUTPUT_TILE_SIZE (INPUT_TILE_SIZE - 2 * RADIUS)

__constant__ float filter_d[CHANNELS*FILTER_ELEMENTS];

__global__ void convoltile(const float* d_input, float* d_output, int width, int height)
{
    int outrow = blockIdx.y * OUTPUT_TILE_SIZE + threadIdx.y - RADIUS;
    int outcol = blockIdx.x * OUTPUT_TILE_SIZE + threadIdx.x - RADIUS;

    __shared__ float m_s[INPUT_TILE_SIZE*INPUT_TILE_SIZE*CHANNELS];

    int local_idx = threadIdx.y * INPUT_TILE_SIZE + threadIdx.x;
    int s_plane = INPUT_TILE_SIZE * INPUT_TILE_SIZE;
    if(outrow>=0 && outrow<height && outcol>=0 && outcol<width)
    {
        m_s[local_idx] = d_input[outrow * width + outcol];
        m_s[local_idx + s_plane] = d_input[outrow * width + outcol + width * height];
        m_s[local_idx + 2 * s_plane] = d_input[outrow * width + outcol + 2 * width * height];
    }
    else
    {
        m_s[local_idx] = 0.0f;
        m_s[local_idx + s_plane] = 0.0f;
        m_s[local_idx + 2 * s_plane] = 0.0f;
    }
    __syncthreads();

    int tilecol = threadIdx.x - RADIUS;
    int tilerow = threadIdx.y - RADIUS;

    if (outrow >= 0 && outrow < height && outcol >= 0 && outcol < width) {
        if (tilecol >= 0 && tilecol < OUTPUT_TILE_SIZE && tilerow >= 0 && tilerow < OUTPUT_TILE_SIZE) {
            float Pvalue = 0.0f;

            for (int frow = 0; frow < FILTER_SIZE; frow++) {
                for (int fcol = 0; fcol < FILTER_SIZE; fcol++) {
                    int rf = frow * FILTER_SIZE + fcol;
                    int s_row = threadIdx.y + frow - RADIUS;
                    int s_col = threadIdx.x + fcol - RADIUS;
                    int s_idx = s_row * INPUT_TILE_SIZE + s_col;

                    Pvalue += m_s[s_idx] * filter_d[rf];
                    Pvalue += m_s[s_idx + s_plane] * filter_d[rf + FILTER_ELEMENTS];
                    Pvalue += m_s[s_idx + 2 * s_plane] * filter_d[rf + 2 * FILTER_ELEMENTS];
                }
            }
            d_output[outrow * width + outcol] = Pvalue;
        }
    }
}

float convolutionShared(
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
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));

    dim3 blockSize(INPUT_TILE_SIZE, INPUT_TILE_SIZE);
    dim3 gridSize(
    (width + OUTPUT_TILE_SIZE - 1) / OUTPUT_TILE_SIZE,
    (height + OUTPUT_TILE_SIZE - 1) / OUTPUT_TILE_SIZE);


    convoltile<<<gridSize,blockSize>>>(d_input, d_output, width, height);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaDeviceSynchronize());
    float milliseconds = 0.0f;

    CUDA_CHECK(cudaEventElapsedTime(
        &milliseconds,
        start,
        stop
    ));

    CUDA_CHECK(cudaGetLastError());

    return milliseconds;

}