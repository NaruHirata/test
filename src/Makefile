CC      = gcc -Wno-unused-result
NVCC    = /usr/local/cuda/bin/nvcc
CXX     = $(CC)
TARGET  := GFandSlope
INSTALL_DIR = ../

# Try compute_cap first
COMPUTE_CAP := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1)
GPU_NAME := $(shell nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)
SM_ARCH := $(shell echo $(COMPUTE_CAP) | tr -d '.')

NVOPTIMIZE = -O4

OPTIMIZE = -O
CXXFLAGS 	= $(OPTIMIZE) $(DEFINES) $(INCLUDE)
LIB	=  -lm

default: $(TARGET)

OBJS = GFandSlope.o GF_host.o GF.o model_IO.o

$(TARGET) : $(OBJS)
	$(NVCC) -o $@ $(OBJS) $(INC) $(LIB) 

GFandSlope.o : GFandSlope.c GFhost.h
	$(CXX) $(CXXFLAGS) -c $(INC) $(LIB) GFandSlope.c

model_IO.o : model_IO.c GFhost.h GFvec3.h
	$(CXX) $(CXXFLAGS) -DSP -c $(INC) $(LIB) model_IO.c

GF_host.o : GF_host.c GFhost.h  GFvec3.h
	$(CXX) $(CXXFLAGS) -c $(INC) $(LIB) GF_host.c

GF.o : GF.cu cudaGF.h
	@echo Detected GPU name: $(GPU_NAME)
	@echo Detected GPU compute capability: sm_$(SM_ARCH)
	$(NVCC) GF.cu $(NVOPTIMIZE)  -lineinfo -c --ptxas-options=-v -arch=sm_$(SM_ARCH)

install: $(TARGET)
	cp $(TARGET) $(INSTALL_DIR)
	@echo "Installation complete: $(TARGET) copied to $(INSTALL_DIR)"

clean:;
	@rm -f $(TARGET) $(OBJS) core *~ *.trace *.a *.o *.so run
