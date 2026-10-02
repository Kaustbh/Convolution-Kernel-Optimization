NVCC = nvcc

# CXXFLAGS = -std=c++17 -O3 -Iinclude

SOURCES = \
    main.cpp \
    preprocessor/imageloader.cpp \
    src/naive.cu \
    src/constant-memory.cu \
    src/tiled.cu \
    src/l2cache.cu

TARGET = convolution.exe

all:
	$(NVCC) $(SOURCES) -o $(TARGET)

clean:
	rm -f $(TARGET)