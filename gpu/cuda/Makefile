# build on a CUDA machine: make (needs nvcc, default /usr/local/cuda/bin/nvcc, and gcc). mass8.c is ../../uniform/mass8.c.
NVCC ?= /usr/local/cuda/bin/nvcc
ARCH ?= sm_75
U = ../../uniform
all: mass8g mass8
cpu.o: cpu.c $(U)/mass8.c
	gcc -O2 -march=native -DSPLITMIX -I$(U) -c cpu.c -o cpu.o
mass8g: mass8g.cu cpu.o
	$(NVCC) -O3 -arch=$(ARCH) -o mass8g mass8g.cu cpu.o
mass8: $(U)/mass8.c
	gcc -O2 -march=native -DSPLITMIX -o mass8 $(U)/mass8.c
