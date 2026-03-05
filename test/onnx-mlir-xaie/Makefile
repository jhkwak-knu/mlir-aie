# VITIS related variables
AIETOOLS_DIR ?= $(shell realpath $(dir $(shell which xchesscc))/../)
AIE_INCLUDE_DIR ?= ${AIETOOLS_DIR}/data/aie_ml/lib
AIE_INSTALL_DIR ?= $(shell realpath $(dir $(shell which aie-opt))/../)

CHESSCC_FLAGS ?= --aiearch aie-ml -f -p me -P ${AIE_INCLUDE_DIR} -I ${AIETOOLS_DIR}/include
AIECC_FLAGS += --host-target=x86_64-amd-linux-gnu --compile-host --aie-generate-cdo
AIECC_HOST_FLAGS += \
	-I./build/aie.mlir.prj \
	-I${AIE_INSTALL_DIR}/include/xaiengine \
	-I${AIE_INSTALL_DIR}/include/test_lib \
	-L${AIE_INSTALL_DIR}/lib \
	-Wl,--start-group \
	-lcdo_driver_mlir_aie \
	-lxaienginecdo_static \
	-lmemory_allocator_ion_static \
	-ltest_lib_static \
	-Wl,--end-group

# INPUT related variables
SRC_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
MLIR_SRC ?= ${SRC_DIR}/onnx_matmul.mlir
KERNEL_SRC ?= ${SRC_DIR}/kernel.cc
HOST_SRC ?= ${SRC_DIR}/test.cpp
PASSES ?= --convert-onnx-to-aie
TARGET ?= onnx_matmul

all: ${TARGET}

build/aie.mlir: ${MLIR_SRC}
	mkdir -p ${@D}
	aie-opt $< ${PASSES} > host.mlir
	mv aie.mlir host.mlir ${@D}

build/kernel.o: ${KERNEL_SRC}
	mkdir -p ${@D}
	xchesscc ${CHESSCC_FLAGS} -c $< -o ${@}

build/aie.mlir.prj/aie_inc.cpp: build/aie.mlir build/kernel.o
	mkdir -p build
	cd build && aiecc.py -j4 ${AIECC_FLAGS} $(notdir $<)

${TARGET}: ${HOST_SRC} build/aie.mlir.prj/aie_inc.cpp
	clang++ -std=c++17 --target=x86_64-amd-linux-gnu -D__AIEARCH__=10 $< -o $@ ${AIECC_HOST_FLAGS} 

clean:
	rm -rf build ${TARGET}
