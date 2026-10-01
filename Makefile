NVCC = nvcc

# CXXFLAGS = -std=c++17 -O3 -Iinclude

SOURCES = \
    main.cpp \
    preprocessor/imageloader.cpp \
    src/naive.cu \
#     src/constant.cu \
#     src/shared.cu \
#     src/l2.cu

TARGET = convolution.exe

all:
	$(NVCC) $(SOURCES) -o $(TARGET)

clean:
	rm -f $(TARGET)