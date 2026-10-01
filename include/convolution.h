#pragma once

float convolutionNaive(
    const float* d_input,
    float* d_output,
    int width,
    int height,
    int channels
);

float convolutionConstant(
    const float* d_input,
    float* d_output,
    const float* d_filter,
    int width,
    int height,
    int channels
);

float convolutionShared(
    const float* d_input,
    float* d_output,
    const float* d_filter,
    int width,
    int height,
    int channels
);

float convolutionL2(
    const float* d_input,
    float* d_output,
    int width,
    int height,
    int channels
);