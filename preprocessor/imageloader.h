#pragma once

struct Image {
    int width;
    int height;
    int channels;

    // Planar CHW layout:
    // [RRRR...][GGGG...][BBBB...]
    float* data;
};

Image loadImageCHW(const char* filename);
void freeImage(Image& image);