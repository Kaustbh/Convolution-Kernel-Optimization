#include "preprocessor/imageloader.h"
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include "include/convolution.h"
#include "include/filter.cuh"
#include "include/convolution_config.cuh"
#include "include/cuda_utils.cuh"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void printHelp(const char* program)
{
    std::cout
        << "CUDA Image Convolution\n\n"

        << "Usage:\n"
        << "  " << program
        << " [options]\n\n"

        << "Options:\n"
        << "  --kernel <name>     Kernel to execute\n"
        << "                      naive\n"
        << "                      constant\n"
        << "                      shared\n"
        << "                      l2\n"
        << "                      all\n\n"

        << "  --image <path>      Input image path\n"
        << "                      Default: images/demon-slayer.jpeg\n\n"

        << "  --output <path>     Output image path\n"
        << "                      Default: images/output.png\n\n"

        << "  --help              Show this help message\n\n"

        << "Examples:\n"
        << "  " << program << " --kernel naive\n"
        << "  " << program << " --kernel shared\n"
        << "  " << program
        << " --kernel l2 --image images/test.jpg\n"
        << "  " << program
        << " --kernel all --image images/test.jpg\n";
}

int main(int argc, char* argv[])
{   
    std::string kernel;
    std::string imagePath = "images/3.jpg";
    std::string outputPath = "output/3.jpg";

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h")
        {
            printHelp(argv[0]);
            return 0;
        }
        else if (arg == "--kernel")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "Error: --kernel requires a value\n";
                return 1;
            }

            kernel = argv[++i];
        }
        else if (arg == "--image")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "Error: --image requires a path\n";
                return 1;
            }

            imagePath = argv[++i];
        }
        else if (arg == "--output")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "Error: --output requires a path\n";
                return 1;
            }

            outputPath = argv[++i];
        }
        else
        {
            std::cerr << "Unknown argument: "
                      << arg << '\n';

            std::cerr << "Use --help for usage information.\n";

            return 1;
        }
    }

    if (kernel.empty())
    {
        std::cerr
            << "Error: --kernel is required.\n"
            << "Use --help for usage information.\n";

        return 1;
    }

    if (kernel != "naive" &&
        kernel != "constant" &&
        kernel != "shared" &&
        kernel != "l2" &&
        kernel != "all")
    {
        std::cerr
            << "Error: unknown kernel '"
            << kernel << "'\n";

        return 1;
    }

    std::cout << "Kernel : " << kernel << '\n';
    std::cout << "Image  : " << imagePath << '\n';
    std::cout << "Output : " << outputPath << '\n';

    Image image = loadImageCHW(imagePath.c_str());

    std::cout << "Image loaded successfully\n";
    std::cout << "Width    : " << image.width << '\n';
    std::cout << "Height   : " << image.height << '\n';
    std::cout << "Channels : " << image.channels << '\n';

    int width = image.width;
    int height = image.height;
    int channels = image.channels;
    int planeSize = width * height;

    std::cout << "\nFirst 5 R values: ";
    for (int i = 0; i < 5; ++i)
        std::cout << image.data[i] << ' ';

    std::cout << "\nFirst 5 G values: ";
    for (int i = 0; i < 5; ++i)
        std::cout << image.data[planeSize + i] << ' ';

    std::cout << "\nFirst 5 B values: ";
    for (int i = 0; i < 5; ++i)
        std::cout << image.data[2 * planeSize + i] << ' ';

    std::cout << '\n';

    float* input_d, * output_d;
    size_t inputsize = width * height * channels * sizeof(float);
    size_t outputsize = width * height * sizeof(float);

    CUDA_CHECK(cudaMalloc(&input_d, inputsize));
    CUDA_CHECK(cudaMalloc(&output_d, outputsize));
    CUDA_CHECK(cudaMemcpy(input_d, image.data, inputsize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(output_d, 0, outputsize));

    float time = 0.0f;
    if (kernel == "naive")
    {
        std::cout << "\nExecuting naive kernel...\n";
        time = convolutionNaive(input_d, output_d, width, height, channels);
    }
    else if (kernel == "constant")
    {
        std::cout << "\nExecuting constant kernel...\n";
        time = convolutionConstant(input_d, output_d, width, height, channels);
    }
    else if (kernel == "shared")
    {
        std::cout << "\nExecuting shared kernel...\n";
        time = convolutionShared(input_d, output_d, width, height, channels);
    }
    else if (kernel == "l2")
    {
        std::cout << "\nExecuting L2 kernel...\n";
        time = convolutionL2(input_d, output_d, width, height, channels);
    }
    else if (kernel == "all")
    {
        std::cout << "\nExecuting all kernels...\n";

        float naive = convolutionNaive(
        input_d, output_d,
        width, height, channels
    );

    float constant = convolutionConstant(
        input_d, output_d,
        width, height, channels
    );

    float tiled = convolutionShared(
        input_d, output_d,
        width, height, channels
    );

    float l2 = convolutionL2(
        input_d, output_d,
        width, height, channels
    );

    CUDA_CHECK(cudaFree(input_d));
    CUDA_CHECK(cudaFree(output_d));

    freeImage(image);

    std::cout << "\n";
    std::cout << "========== Benchmark ==========\n";
    std::cout << "Naive    : " << naive << " ms\n";
    std::cout << "Constant : " << constant << " ms\n";
    std::cout << "Tiled    : " << tiled << " ms\n";
    std::cout << "L2       : " << l2 << " ms\n";
    std::cout << "===============================\n";

    return 0;

    }

    float* output_h = new float[width * height];
    CUDA_CHECK(cudaMemcpy(output_h, output_d, outputsize, cudaMemcpyDeviceToHost));

    // --------------------------------------------------
    // Find output range
    // --------------------------------------------------

    float minVal = output_h[0];
    float maxVal = output_h[0];

    for (int i = 1; i < width * height; ++i)
    {
        minVal = std::min(minVal, output_h[i]);
        maxVal = std::max(maxVal, output_h[i]);
    }

    std::cout << "\nOutput range: "
              << minVal << " -> "
              << maxVal << '\n';


    // --------------------------------------------------
    // Convert float -> uint8
    // --------------------------------------------------

    unsigned char* outputImage =
        new unsigned char[width * height];

    for (int i = 0; i < width * height; ++i)
    {
        // float normalized =
        //     (output_h[i] - minVal) /
        //     (maxVal - minVal);

        // float normalized =
        //     output_h[i];

        // outputImage[i] =
        //     static_cast<unsigned char>(
        //         normalized * 255.0f
        //     );

        float val = output_h[i];

    // 1. Take the absolute value to convert negative edge energy into positive highlights
    val = std::fabs(val); 

    // 2. STABILITY CLAMP: Securely lock the value between 0.0f and 1.0f 
    if (val < 0.0f) val = 0.0f;
    if (val > 1.0f) val = 1.0f;

    // 3. Scale safely to standard 8-bit unsigned integer space
    outputImage[i] = static_cast<unsigned char>(val * 255.0f);
    }


    // --------------------------------------------------
    // Save output image
    // --------------------------------------------------

    stbi_write_png(
        outputPath.c_str(),
        width,
        height,
        1,
        outputImage,
        width
    );

    std::cout << "Time required for executing the kernel\n" << time << " ms\n";
    std::cout << "Output saved to\n" << outputPath << '\n';

    std::cout << "For Profiling the kernel \n";
    std::cout << "Run: ncu --set full -o covol"<<kernel<<" convolution.exe --kernel "<<kernel;


    // --------------------------------------------------
    // Cleanup
    // --------------------------------------------------

    CUDA_CHECK(cudaFree(input_d));
    CUDA_CHECK(cudaFree(output_d));
    delete[] output_h;
    delete[] outputImage;

    freeImage(image);

    return 0;
}