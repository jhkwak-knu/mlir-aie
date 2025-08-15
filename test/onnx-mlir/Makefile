# VITIS related variables
AIETOOLS_DIR ?= $(shell realpath $(dir $(shell which xchesscc))/../)
AIE2_INCLUDE_DIR ?= ${AIETOOLS_DIR}/data/aie2p/lib
AIE_ARCH ?= aie2p
CHESSCC_FLAGS ?= --aiearch ${AIE_ARCH} -f -p me -P ${AIE2_INCLUDE_DIR} -I ${AIETOOLS_DIR}/include
AIECC_FLAGS ?= --aie-generate-xclbin --no-compile-host --aie-generate-npu-insts --npu-insts-name=insts.bin

# INPUT related variables
SRC_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
MLIR_SRCS ?= ${SRC_DIR}/onnx_matmul.mlir
KERNEL_SRCS ?= ${SRC_DIR}/kernel.cc
PASSES ?= --convert-onnx-to-aie
TARGET ?= onnx_matmul

all: build/final.xclbin build/insts.bin

build/aie.mlir: ${MLIR_SRCS}
	mkdir -p ${@D}
	aie-opt $< ${PASSES} > host.mlir
	mv aie.mlir host.mlir ${@D}

build/kernel.o: ${KERNEL_SRCS}
	mkdir -p ${@D}
	xchesscc ${CHESSCC_FLAGS} -c $< -o $@

build/final.xclbin: build/aie.mlir build/kernel.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py ${AIECC_FLAGS} --xclbin-name=${@F} $(<:%=../%)

${TARGET}: test.cpp
	rm -rf _build
	mkdir -p _build
	cd _build && cmake ${SRC_DIR} -DTARGET_NAME=${TARGET}
	cd _build && cmake --build . --config Release
	cp _build/${TARGET} $@

run: ${TARGET} build/final.xclbin build/insts.bin
	./$< -x build/final.xclbin -i build/insts.bin -k MLIR_AIE -v 1 > log.txt

clean:
	rm -rf build _build ${TARGET} log.txt
