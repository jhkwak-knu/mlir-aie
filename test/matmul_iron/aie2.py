# section-4/section-4b/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import numpy as np
import argparse
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_

import aie.utils.trace as trace_utils

from tc_loader import *


def my_matmul(config, trace_size):
    elem_dtype = np.float32                          # TODO: read from config.data["elemType"]
    num_cols = config.data["numLastSpm"] // 4

    # Map num_cols → AIEDevice
    col2dev = {
        1: AIEDevice.npu2_1col,
        2: AIEDevice.npu2_2col,
        3: AIEDevice.npu2_3col,
        4: AIEDevice.npu2_4col,
        5: AIEDevice.npu2_5col,
        6: AIEDevice.npu2_6col,
        7: AIEDevice.npu2_7col,
        8: AIEDevice.npu2,
    }

    if num_cols not in col2dev:
        raise ValueError(f"[ERROR] Unsupported num_cols derived from numLastSpm: {num_cols}")

    dev = col2dev[num_cols]

    @device(dev)
    def device_body():
        lhs_ty = np.ndarray[(config.data["levels"][0]["TM"] * config.data["levels"][0]["TK"],), np.dtype[elem_dtype]]
        rhs_ty = np.ndarray[(config.data["levels"][0]["TN"] * config.data["levels"][0]["TK"],), np.dtype[elem_dtype]]
        res_ty = np.ndarray[(config.data["levels"][0]["TM"] * config.data["levels"][0]["TN"],), np.dtype[elem_dtype]]

        # Tile declarations
        ShimTile = tile(0, 0)
        ComputeTile2 = tile(0, 2)
        ComputeTile3 = tile(0, 3)
        ComputeTile4 = tile(0, 4)
        ComputeTile5 = tile(0, 5)

        # AIE-array data movement with object fifos
        of_lhs1 = object_fifo("lhs1", ShimTile, ComputeTile2, 1, lhs_ty)
        of_lhs2 = object_fifo("lhs2", ShimTile, ComputeTile3, 1, lhs_ty)
        of_lhs3 = object_fifo("lhs3", ShimTile, ComputeTile4, 1, lhs_ty)
        of_lhs4 = object_fifo("lhs4", ShimTile, ComputeTile5, 1, lhs_ty)
        of_rhs1 = object_fifo("rhs1", ShimTile, [ComputeTile2, ComputeTile3, ComputeTile4, ComputeTile5], 1, rhs_ty)
        of_res1 = object_fifo("res1", ComputeTile2, ShimTile, 1, res_ty)
        of_res2 = object_fifo("res2", ComputeTile3, ShimTile, 1, res_ty)
        of_res3 = object_fifo("res3", ComputeTile4, ShimTile, 1, res_ty)
        of_res4 = object_fifo("res4", ComputeTile5, ShimTile, 1, res_ty)

        # AIE Core Function declarations
        extern_kernel = external_func(
            "extern_kernel",
            inputs=[lhs_ty, rhs_ty, res_ty, np.int32, np.int32, np.int32, np.int8],
        )

        # Set up compute tiles
        # Compute tile 2
        @core(ComputeTile2, "kernel.o")
        def core_body():
            # Effective while(1)
            for _ in range_(sys.maxsize):
                acc = 0
                elem_rhs = of_rhs1.acquire(ObjectFifoPort.Consume, 1)
                # Number of sub-vector "tile" iterations
                for _ in range_(config.data["levels"][0]["TPm"]):
                    elem_res = of_res1.acquire(ObjectFifoPort.Produce, 1)
                    elem_lhs = of_lhs1.acquire(ObjectFifoPort.Consume, 1)
                    extern_kernel(elem_lhs, elem_rhs, elem_res, config.data["levels"][0]["TM"], config.data["levels"][0]["TN"], config.data["levels"][0]["TK"], acc)
                    of_lhs1.release(ObjectFifoPort.Consume, 1)
                    of_res1.release(ObjectFifoPort.Produce, 1)
                of_rhs1.release(ObjectFifoPort.Consume, 1)

    #     # Set up a packet-switched flow from core to shim for tracing information
    #     tiles_to_trace = [ComputeTile2, ShimTile]
    #     if trace_size > 0:
    #         trace_utils.configure_packet_tracing_flow(tiles_to_trace, ShimTile)

    #     # To/from AIE-array data movement
    #     @runtime_sequence(tensor_ty, scalar_ty, tensor_ty)
    #     def sequence(A, F, C):
    #         if trace_size > 0:
    #             trace_utils.configure_packet_tracing_aie2(
    #                 tiles_to_trace=tiles_to_trace,
    #                 shim=ShimTile,
    #                 trace_size=trace_size,
    #             )

    #         in_task = shim_dma_single_bd_task(
    #             of_in, A, sizes=[1, 1, 1, tensor_size], issue_token=True
    #         )
    #         in_factor_task = shim_dma_single_bd_task(
    #             of_factor, F, sizes=[1, 1, 1, 1], issue_token=True
    #         )
    #         out_task = shim_dma_single_bd_task(
    #             of_out, C, sizes=[1, 1, 1, tensor_size], issue_token=True
    #         )

    #         dma_start_task(in_task, in_factor_task, out_task)
    #         dma_await_task(in_task, in_factor_task, out_task)

    #         trace_utils.gen_trace_done_aie2(ShimTile)


if len(sys.argv) < 1:
    raise ValueError(
        "[ERROR] Need at least 1 arguments (path)"
    )


p = argparse.ArgumentParser()
p.add_argument("-p", "--path", required=True, dest="path", help="AIE configuration file path")
p.add_argument(
    "-t",
    "--trace_size",
    required=False,
    dest="trace_size",
    default=0,
    help="Trace buffer size",
)
opts = p.parse_args(sys.argv[1:])

file_path = opts.path
trace_size = int(opts.trace_size)

tc = load_tc_json(file_path)

with mlir_mod_ctx() as ctx:
    my_matmul(tc, trace_size)
    res = ctx.module.operation.verify()
    if res == True:
        print(ctx.module)
    else:
        print(res)
