/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ NNPACompilerOptions.hpp ---------------------===//
//
// Copyright 2022-2024 The IBM Research Authors.
//
// =============================================================================
//
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_NNPA_COMPILER_OPTIONS_H
#define ONNX_MLIR_NNPA_COMPILER_OPTIONS_H

#include "llvm/Support/CommandLine.h"

// clang-format off

#define INSTRUMENTSTAGE_ENUM_NNPA                                               \
    ,                                                                          \
    ZHigh,                                                                     \
    ZLow

// clang-format on

#define INSTRUMENTSTAGE_CL_ENUM_NNPA                                           \
  clEnumVal(Onnx, "Profile for onnx ops. For NNPA, profile onnx ops before "   \
                  "lowering to zhigh."),                                       \
      clEnumVal(ZHigh, "NNPA profiling for onnx and zhigh ops."),              \
      clEnumVal(ZLow, "NNPA profiling for zlow ops.")

#define PROFILEIR_CL_ENUM_NNPA                                                 \
  , clEnumVal(ZHigh, "Profile operations in ZHighIR generated by "             \
                     "--EmitZHighIR.")

#define OPTREPORT_ENUM_NNPA , NNPAUnsupportedOps

#define OPTREPORT_CL_ENUM_NNPA                                                 \
  , clEnumVal(NNPAUnsupportedOps,                                              \
        "Provide report on why ONNX ops did not run on NNPA.")

namespace onnx_mlir {
typedef enum {
  EmitZNONE,
  EmitZLowIR,
  EmitZHighIR,
} NNPAEmissionTargetType;

typedef enum {
  QualifyingOps,    /* Any ops that qualify for NNPA will go on NNPA. */
  FasterOps,        /* Only qualifying ops that are faster on NNPA */
  FasterOpsWSU,     /* FasterOps with With Stick and Unstick (WSU) cost.*/
  MuchFasterOpsWSU, /* FasterOpsWSU only if significantly faster. */
} NNPAPlacementHeuristic;

// Quantization type
typedef enum {
  symWeight,
  asymWeight,
  symActivation,
  asymActivation,
  autoQuantOpt,
} NNPAQuantOptions;

extern llvm::cl::OptionCategory OnnxMlirOptions;
extern llvm::cl::OptionCategory OnnxMlirCommonOptions;
extern llvm::cl::opt<onnx_mlir::NNPAEmissionTargetType> nnpaEmissionTarget;
extern llvm::cl::opt<bool> nnpaDisableZHighToOnnx;
extern llvm::cl::opt<bool> nnpaEnableZHighDecomposeStickUnstick;
extern llvm::cl::opt<bool> nnpaDisableCompilerStickUnstick;
extern llvm::cl::opt<bool> nnpaEnableScalarBcastBinary;
extern llvm::cl::opt<NNPAPlacementHeuristic> nnpaPlacementHeuristic;
extern llvm::cl::opt<bool> profileZHighIR;
extern llvm::cl::opt<std::string> nnpaLoadDevicePlacementFile;
extern llvm::cl::opt<std::string> nnpaSaveDevicePlacementFile;
extern llvm::cl::opt<bool> nnpaDisableSaturation;
extern llvm::cl::opt<bool> nnpaUseDynamicQuantizeLinearOnCPU;
extern llvm::cl::opt<bool> nnpaUseDynamicQuantizeLinearOnCPUForScaleOffset;
extern std::vector<NNPAQuantOptions> nnpaQuantDynamic;
extern std::vector<std::string> nnpaQuantOpTypes;

} // namespace onnx_mlir
#endif
