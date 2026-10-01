#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "imageloader.h"

#include <iostream>
#include <stdexcept>

Image loadImageCHW(const char* filename)
{
    int width;
    int height;
    int channels;

    // Force stb_image to give us exactly 3 channels: RGB.
    unsigned char* pixels = stbi_load(
        filename,
        &width,
        &height,
        &channels,
        3
    );

    if (!pixels) {
        throw std::runtime_error(
            std::string("Failed to load image: ") +
            stbi_failure_reason()
        );
    }

    constexpr int C = 3;
    const int planeSize = width * height;

    // CHW:
    // R plane -> C*H*W
    // G plane -> C*H*W
    // B plane -> C*H*W
    float* chw = new float[C * planeSize];

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {

            int pixel = y * width + x;

            // HWC location in stb_image:
            //
            // RGB RGB RGB RGB ...
            //
            int hwcIndex = pixel * C;

            // CHW locations:
            int rIndex = 0 * planeSize + pixel;
            int gIndex = 1 * planeSize + pixel;
            int bIndex = 2 * planeSize + pixel;

            chw[rIndex] = static_cast<float>(pixels[hwcIndex + 0])/255.0f;
            chw[gIndex] = static_cast<float>(pixels[hwcIndex + 1])/255.0f;
            chw[bIndex] = static_cast<float>(pixels[hwcIndex + 2])/255.0f;
        }
    }

    stbi_image_free(pixels);

    Image image;
    image.width = width;
    image.height = height;
    image.channels = C;
    image.data = chw;

    return image;
}


void freeImage(Image& image)
{
    delete[] image.data;

    image.data = nullptr;
    image.width = 0;
    image.height = 0;
    image.channels = 0;
}