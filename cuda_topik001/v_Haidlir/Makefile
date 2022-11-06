.DEFAULT_GOAL := all

all: build_a run_a build_b run_b

build_a:
    @echo "Building cuda file for question A"
    nvcc -o Topik001-A-vHAN-1.o Topik001-A-vHAN-1.cu

build_b:
    @echo "Building cuda file for question B"
    nvcc -o incrementArrays.o incrementArrays.cu

run_a:
    @echo "Run cuda file for question A"
    ./Topik001-A-vHAN-1.o 1 128 128
    ./Topik001-A-vHAN-1.o 2 64 128
    ./Topik001-A-vHAN-1.o 4 32 128
    ./Topik001-A-vHAN-1.o 8 16 128
    ./Topik001-A-vHAN-1.o 16 4 128
    ./Topik001-A-vHAN-1.o 32 4 128
    ./Topik001-A-vHAN-1.o 64 4 128
    ./Topik001-A-vHAN-1.o 128 4 128
    ./Topik001-A-vHAN-1.o 1 128 128
    ./Topik001-A-vHAN-1.o 1 256 128
    ./Topik001-A-vHAN-1.o 1 512 128
    ./Topik001-A-vHAN-1.o 1 1024 128
    ./Topik001-A-vHAN-1.o 1 2048 128

run_b:
    @echo "Run cuda file for question B"
    ./incrementArrays.o 512 512
    ./incrementArrays.o 512 1024
    ./incrementArrays.o 512 2048
    ./incrementArrays.o 1024 512
    ./incrementArrays.o 1024 1024
    ./incrementArrays.o 1024 2048
    ./incrementArrays.o 2096 512
    ./incrementArrays.o 2096 1024
    ./incrementArrays.o 2096 2048
