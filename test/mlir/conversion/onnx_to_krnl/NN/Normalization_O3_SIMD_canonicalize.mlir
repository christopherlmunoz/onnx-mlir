// RUN: onnx-mlir-opt -O3 --mtriple=s390x-ibm-loz --march=z16 --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// use --mtriple=s390x-ibm-loz --march=z16 to enable SIMD as we now need a machine
// can also use --march=x86-64 instead.

// -----

// It should make the substitution with the fast algo
func.func @layernorm_4D_with_scale_bias(%arg0: tensor<2x64x32x8xf32>, %arg1: tensor<32x8xf32>, %arg2: tensor<32x8xf32>) -> tensor<*xf32> {
  %0 = "onnx.NoValue"() {value} : () -> none
  %Y, %Mean, %InvStdDev = "onnx.LayerNormalization"(%arg0, %arg1, %arg2) {axis = -2 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<2x64x32x8xf32>, tensor<32x8xf32>, tensor<32x8xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
  func.return %Y : tensor<*xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @layernorm_4D_with_scale_bias
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x64x32x8xf32>, [[PARAM_1_:%.+]]: memref<32x8xf32>, [[PARAM_2_:%.+]]: memref<32x8xf32>) -> memref<2x64x32x8xf32> {
// CHECK-DAG:       [[CST_2_dot_560000_:%.+]] = arith.constant 2.560000e+02 : f32
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<0.000000e+00> : vector<16xf32>
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_9_dot_99999974_:%.+]] = arith.constant 9.99999974E-6 : f32
// CHECK-DAG:       [[CST_128_:%.+]] = arith.constant 128 : index
// CHECK-DAG:       [[CST_256_:%.+]] = arith.constant 256 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2xindex>
// CHECK:           affine.store [[CST_128_]], [[RES_]][0] : memref<2xindex>
// CHECK:           affine.store [[CST_256_]], [[RES_]][1] : memref<2xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_]]) : (memref<2x64x32x8xf32>, memref<2xindex>) -> memref<128x256xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_256_]], [[RES_1_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_4_:%.+]] = memref.reshape [[PARAM_1_]]([[RES_1_]]) : (memref<32x8xf32>, memref<1xindex>) -> memref<256xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_256_]], [[RES_2_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_6_:%.+]] = memref.reshape [[PARAM_2_]]([[RES_2_]]) : (memref<32x8xf32>, memref<1xindex>) -> memref<256xf32>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<2x64x32x8xf32>
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<2xindex>
// CHECK:           affine.store [[CST_128_]], [[RES_4_]][0] : memref<2xindex>
// CHECK:           affine.store [[CST_256_]], [[RES_4_]][1] : memref<2xindex>
// CHECK-DAG:       [[VAR_reshape_9_:%.+]] = memref.reshape [[RES_3_]]([[RES_4_]]) : (memref<2x64x32x8xf32>, memref<2xindex>) -> memref<128x256xf32>
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<2x64x1x1xf32>
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<2xindex>
// CHECK:           affine.store [[CST_128_]], [[RES_6_]][0] : memref<2xindex>
// CHECK:           affine.store [[CST_1_]], [[RES_6_]][1] : memref<2xindex>
// CHECK-DAG:       [[VAR_reshape_12_:%.+]] = memref.reshape [[RES_5_]]([[RES_6_]]) : (memref<2x64x1x1xf32>, memref<2xindex>) -> memref<128x1xf32>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc() {{.*}}: memref<2x64x1x1xf32>
// CHECK-DAG:       [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<2xindex>
// CHECK:           affine.store [[CST_128_]], [[RES_8_]][0] : memref<2xindex>
// CHECK:           affine.store [[CST_1_]], [[RES_8_]][1] : memref<2xindex>
// CHECK-DAG:       [[VAR_reshape_15_:%.+]] = memref.reshape [[RES_7_]]([[RES_8_]]) : (memref<2x64x1x1xf32>, memref<2xindex>) -> memref<128x1xf32>
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc() {{.*}}: memref<4x16xf32>
// CHECK-DAG:       [[RES_10_:%.+]] = memref.alloc() {{.*}}: memref<4x16xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]] 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 128){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__0_]]) : (!krnl.loop) -> index
// CHECK:             vector.store [[VAR_cst_0_]], [[RES_9_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x16xf32>, vector<16xf32>
// CHECK:             vector.store [[VAR_cst_0_]], [[RES_10_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x16xf32>, vector<16xf32>
// CHECK:             vector.store [[VAR_cst_0_]], [[RES_9_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x16xf32>, vector<16xf32>
// CHECK:             vector.store [[VAR_cst_0_]], [[RES_10_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x16xf32>, vector<16xf32>
// CHECK:             vector.store [[VAR_cst_0_]], [[RES_9_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x16xf32>, vector<16xf32>
// CHECK:             vector.store [[VAR_cst_0_]], [[RES_10_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x16xf32>, vector<16xf32>
// CHECK:             vector.store [[VAR_cst_0_]], [[RES_9_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x16xf32>, vector<16xf32>
// CHECK:             vector.store [[VAR_cst_0_]], [[RES_10_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x16xf32>, vector<16xf32>
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 256 step 16 {
// CHECK:               [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_1_]], [[I_1_]]{{.}} : memref<128x256xf32>, vector<16xf32>
// CHECK-DAG:           [[VAR_53_:%.+]] = arith.mulf [[LOAD_VAR_reshape_MEM_]], [[LOAD_VAR_reshape_MEM_]] : vector<16xf32>
// CHECK-DAG:           [[LOAD_RES_9_MEM_:%.+]] = vector.load [[RES_9_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x16xf32>, vector<16xf32>
// CHECK:               [[VAR_55_:%.+]] = arith.addf [[LOAD_RES_9_MEM_]], [[LOAD_VAR_reshape_MEM_]] : vector<16xf32>
// CHECK:               vector.store [[VAR_55_]], [[RES_9_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x16xf32>, vector<16xf32>
// CHECK:               [[LOAD_RES_10_MEM_:%.+]] = vector.load [[RES_10_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x16xf32>, vector<16xf32>
// CHECK:               [[VAR_57_:%.+]] = arith.addf [[LOAD_RES_10_MEM_]], [[VAR_53_]] : vector<16xf32>
// CHECK:               vector.store [[VAR_57_]], [[RES_10_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x16xf32>, vector<16xf32>
// CHECK:               [[VAR_58_:%.+]] = arith.addi [[VAR_1_]], [[CST_1_]] : index
// CHECK:               [[LOAD_VAR_reshape_MEM_1_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_58_]], [[I_1_]]{{.}} : memref<128x256xf32>, vector<16xf32>
// CHECK-DAG:           [[VAR_60_:%.+]] = arith.mulf [[LOAD_VAR_reshape_MEM_1_]], [[LOAD_VAR_reshape_MEM_1_]] : vector<16xf32>
// CHECK-DAG:           [[LOAD_RES_9_MEM_1_:%.+]] = vector.load [[RES_9_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x16xf32>, vector<16xf32>
// CHECK:               [[VAR_62_:%.+]] = arith.addf [[LOAD_RES_9_MEM_1_]], [[LOAD_VAR_reshape_MEM_1_]] : vector<16xf32>
// CHECK:               vector.store [[VAR_62_]], [[RES_9_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x16xf32>, vector<16xf32>
// CHECK:               [[LOAD_RES_10_MEM_1_:%.+]] = vector.load [[RES_10_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x16xf32>, vector<16xf32>
// CHECK:               [[VAR_64_:%.+]] = arith.addf [[LOAD_RES_10_MEM_1_]], [[VAR_60_]] : vector<16xf32>
// CHECK:               vector.store [[VAR_64_]], [[RES_10_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x16xf32>, vector<16xf32>
// CHECK:               [[VAR_65_:%.+]] = arith.addi [[VAR_1_]], [[CST_2_]] : index
// CHECK:               [[LOAD_VAR_reshape_MEM_2_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_65_]], [[I_1_]]{{.}} : memref<128x256xf32>, vector<16xf32>
// CHECK-DAG:           [[VAR_67_:%.+]] = arith.mulf [[LOAD_VAR_reshape_MEM_2_]], [[LOAD_VAR_reshape_MEM_2_]] : vector<16xf32>
// CHECK-DAG:           [[LOAD_RES_9_MEM_2_:%.+]] = vector.load [[RES_9_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x16xf32>, vector<16xf32>
// CHECK:               [[VAR_69_:%.+]] = arith.addf [[LOAD_RES_9_MEM_2_]], [[LOAD_VAR_reshape_MEM_2_]] : vector<16xf32>
// CHECK:               vector.store [[VAR_69_]], [[RES_9_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x16xf32>, vector<16xf32>
// CHECK:               [[LOAD_RES_10_MEM_2_:%.+]] = vector.load [[RES_10_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x16xf32>, vector<16xf32>
// CHECK:               [[VAR_71_:%.+]] = arith.addf [[LOAD_RES_10_MEM_2_]], [[VAR_67_]] : vector<16xf32>
// CHECK:               vector.store [[VAR_71_]], [[RES_10_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x16xf32>, vector<16xf32>
// CHECK:               [[VAR_72_:%.+]] = arith.addi [[VAR_1_]], [[CST_3_]] : index
// CHECK:               [[LOAD_VAR_reshape_MEM_3_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_72_]], [[I_1_]]{{.}} : memref<128x256xf32>, vector<16xf32>
// CHECK-DAG:           [[VAR_74_:%.+]] = arith.mulf [[LOAD_VAR_reshape_MEM_3_]], [[LOAD_VAR_reshape_MEM_3_]] : vector<16xf32>
// CHECK-DAG:           [[LOAD_RES_9_MEM_3_:%.+]] = vector.load [[RES_9_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x16xf32>, vector<16xf32>
// CHECK:               [[VAR_76_:%.+]] = arith.addf [[LOAD_RES_9_MEM_3_]], [[LOAD_VAR_reshape_MEM_3_]] : vector<16xf32>
// CHECK:               vector.store [[VAR_76_]], [[RES_9_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x16xf32>, vector<16xf32>
// CHECK:               [[LOAD_RES_10_MEM_3_:%.+]] = vector.load [[RES_10_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x16xf32>, vector<16xf32>
// CHECK:               [[VAR_78_:%.+]] = arith.addf [[LOAD_RES_10_MEM_3_]], [[VAR_74_]] : vector<16xf32>
// CHECK:               vector.store [[VAR_78_]], [[RES_10_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x16xf32>, vector<16xf32>
// CHECK:             }
// CHECK-DAG:         [[LOAD_RES_9_MEM_4_:%.+]] = vector.load [[RES_9_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x16xf32>, vector<16xf32>
// CHECK-DAG:         [[LOAD_RES_10_MEM_4_:%.+]] = vector.load [[RES_10_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x16xf32>, vector<16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_4_:%.+]] = vector.reduction <add>, [[LOAD_RES_9_MEM_4_]] : vector<16xf32> into f32
// CHECK-DAG:         [[VAR_5_:%.+]] = vector.reduction <add>, [[LOAD_RES_10_MEM_4_]] : vector<16xf32> into f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.divf [[VAR_4_]], [[CST_2_dot_560000_]] : f32
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.divf [[VAR_5_]], [[CST_2_dot_560000_]] : f32
// CHECK:             [[VAR_8_:%.+]] = arith.mulf [[VAR_6_]], [[VAR_6_]] : f32
// CHECK:             [[VAR_9_:%.+]] = arith.subf [[VAR_7_]], [[VAR_8_]] : f32
// CHECK:             [[VAR_10_:%.+]] = arith.addf [[VAR_9_]], [[CST_9_dot_99999974_]] : f32
// CHECK:             [[VAR_11_:%.+]] = math.sqrt [[VAR_10_]] : f32
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_11_]] : f32
// CHECK-DAG:         [[LOAD_RES_9_MEM_5_:%.+]] = vector.load [[RES_9_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x16xf32>, vector<16xf32>
// CHECK-DAG:         [[LOAD_RES_10_MEM_5_:%.+]] = vector.load [[RES_10_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x16xf32>, vector<16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_15_:%.+]] = vector.reduction <add>, [[LOAD_RES_9_MEM_5_]] : vector<16xf32> into f32
// CHECK-DAG:         [[VAR_16_:%.+]] = vector.reduction <add>, [[LOAD_RES_10_MEM_5_]] : vector<16xf32> into f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_17_:%.+]] = arith.divf [[VAR_15_]], [[CST_2_dot_560000_]] : f32
// CHECK-DAG:         [[VAR_18_:%.+]] = arith.divf [[VAR_16_]], [[CST_2_dot_560000_]] : f32
// CHECK:             [[VAR_19_:%.+]] = arith.mulf [[VAR_17_]], [[VAR_17_]] : f32
// CHECK:             [[VAR_20_:%.+]] = arith.subf [[VAR_18_]], [[VAR_19_]] : f32
// CHECK:             [[VAR_21_:%.+]] = arith.addf [[VAR_20_]], [[CST_9_dot_99999974_]] : f32
// CHECK:             [[VAR_22_:%.+]] = math.sqrt [[VAR_21_]] : f32
// CHECK-DAG:         [[VAR_23_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_22_]] : f32
// CHECK-DAG:         [[LOAD_RES_9_MEM_6_:%.+]] = vector.load [[RES_9_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x16xf32>, vector<16xf32>
// CHECK-DAG:         [[LOAD_RES_10_MEM_6_:%.+]] = vector.load [[RES_10_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x16xf32>, vector<16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_26_:%.+]] = vector.reduction <add>, [[LOAD_RES_9_MEM_6_]] : vector<16xf32> into f32
// CHECK-DAG:         [[VAR_27_:%.+]] = vector.reduction <add>, [[LOAD_RES_10_MEM_6_]] : vector<16xf32> into f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_28_:%.+]] = arith.divf [[VAR_26_]], [[CST_2_dot_560000_]] : f32
// CHECK-DAG:         [[VAR_29_:%.+]] = arith.divf [[VAR_27_]], [[CST_2_dot_560000_]] : f32
// CHECK:             [[VAR_30_:%.+]] = arith.mulf [[VAR_28_]], [[VAR_28_]] : f32
// CHECK:             [[VAR_31_:%.+]] = arith.subf [[VAR_29_]], [[VAR_30_]] : f32
// CHECK:             [[VAR_32_:%.+]] = arith.addf [[VAR_31_]], [[CST_9_dot_99999974_]] : f32
// CHECK:             [[VAR_33_:%.+]] = math.sqrt [[VAR_32_]] : f32
// CHECK-DAG:         [[VAR_34_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_33_]] : f32
// CHECK-DAG:         [[LOAD_RES_9_MEM_7_:%.+]] = vector.load [[RES_9_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x16xf32>, vector<16xf32>
// CHECK-DAG:         [[LOAD_RES_10_MEM_7_:%.+]] = vector.load [[RES_10_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x16xf32>, vector<16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_37_:%.+]] = vector.reduction <add>, [[LOAD_RES_9_MEM_7_]] : vector<16xf32> into f32
// CHECK-DAG:         [[VAR_38_:%.+]] = vector.reduction <add>, [[LOAD_RES_10_MEM_7_]] : vector<16xf32> into f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_39_:%.+]] = arith.divf [[VAR_37_]], [[CST_2_dot_560000_]] : f32
// CHECK-DAG:         [[VAR_40_:%.+]] = arith.divf [[VAR_38_]], [[CST_2_dot_560000_]] : f32
// CHECK:             [[VAR_41_:%.+]] = arith.mulf [[VAR_39_]], [[VAR_39_]] : f32
// CHECK:             [[VAR_42_:%.+]] = arith.subf [[VAR_40_]], [[VAR_41_]] : f32
// CHECK:             [[VAR_43_:%.+]] = arith.addf [[VAR_42_]], [[CST_9_dot_99999974_]] : f32
// CHECK:             [[VAR_44_:%.+]] = math.sqrt [[VAR_43_]] : f32
// CHECK:             [[VAR_45_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_44_]] : f32
// CHECK:             affine.for [[I_2_:%.+]] = 0 to 256 step 16 {
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_4_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_1_]], [[I_2_]]{{.}} : memref<128x256xf32>, vector<16xf32>
// CHECK-DAG:           [[VAR_53_1_:%.+]] = vector.broadcast [[VAR_6_]] : f32 to vector<16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_RES_9_MEM_8_:%.+]] = arith.subf [[LOAD_VAR_reshape_MEM_4_]], [[VAR_53_1_]] : vector<16xf32>
// CHECK-DAG:           [[VAR_55_1_:%.+]] = vector.broadcast [[VAR_12_]] : f32 to vector<16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_RES_10_MEM_8_:%.+]] = arith.mulf [[LOAD_RES_9_MEM_8_]], [[VAR_55_1_]] : vector<16xf32>
// CHECK-DAG:           [[VAR_57_1_:%.+]] = vector.load [[VAR_reshape_4_]]{{.}}[[I_2_]]{{.}} : memref<256xf32>, vector<16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_58_1_:%.+]] = arith.mulf [[LOAD_RES_10_MEM_8_]], [[VAR_57_1_]] : vector<16xf32>
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_1_:%.+]] = vector.load [[VAR_reshape_6_]]{{.}}[[I_2_]]{{.}} : memref<256xf32>, vector<16xf32>
// CHECK:               [[VAR_60_1_:%.+]] = arith.addf [[VAR_58_1_]], [[LOAD_VAR_reshape_MEM_1_]] : vector<16xf32>
// CHECK:               vector.store [[VAR_60_1_]], [[VAR_reshape_9_]]{{.}}[[VAR_1_]], [[I_2_]]{{.}} : memref<128x256xf32>, vector<16xf32>
// CHECK:               [[LOAD_RES_9_MEM_1_:%.+]] = arith.addi [[VAR_1_]], [[CST_1_]] : index
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_5_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[LOAD_RES_9_MEM_1_]], [[I_2_]]{{.}} : memref<128x256xf32>, vector<16xf32>
// CHECK-DAG:           [[LOAD_RES_10_MEM_1_:%.+]] = vector.broadcast [[VAR_17_]] : f32 to vector<16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_64_1_:%.+]] = arith.subf [[LOAD_VAR_reshape_MEM_5_]], [[LOAD_RES_10_MEM_1_]] : vector<16xf32>
// CHECK-DAG:           [[VAR_65_1_:%.+]] = vector.broadcast [[VAR_23_]] : f32 to vector<16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_2_:%.+]] = arith.mulf [[VAR_64_1_]], [[VAR_65_1_]] : vector<16xf32>
// CHECK-DAG:           [[VAR_67_1_:%.+]] = vector.load [[VAR_reshape_4_]]{{.}}[[I_2_]]{{.}} : memref<256xf32>, vector<16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_RES_9_MEM_2_:%.+]] = arith.mulf [[LOAD_VAR_reshape_MEM_2_]], [[VAR_67_1_]] : vector<16xf32>
// CHECK-DAG:           [[VAR_69_1_:%.+]] = vector.load [[VAR_reshape_6_]]{{.}}[[I_2_]]{{.}} : memref<256xf32>, vector<16xf32>
// CHECK:               [[LOAD_RES_10_MEM_2_:%.+]] = arith.addf [[LOAD_RES_9_MEM_2_]], [[VAR_69_1_]] : vector<16xf32>
// CHECK:               vector.store [[LOAD_RES_10_MEM_2_]], [[VAR_reshape_9_]]{{.}}[[LOAD_RES_9_MEM_1_]], [[I_2_]]{{.}} : memref<128x256xf32>, vector<16xf32>
// CHECK:               [[VAR_71_1_:%.+]] = arith.addi [[VAR_1_]], [[CST_2_]] : index
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_6_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_71_1_]], [[I_2_]]{{.}} : memref<128x256xf32>, vector<16xf32>
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_3_:%.+]] = vector.broadcast [[VAR_28_]] : f32 to vector<16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_74_1_:%.+]] = arith.subf [[LOAD_VAR_reshape_MEM_6_]], [[LOAD_VAR_reshape_MEM_3_]] : vector<16xf32>
// CHECK-DAG:           [[LOAD_RES_9_MEM_3_:%.+]] = vector.broadcast [[VAR_34_]] : f32 to vector<16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_76_1_:%.+]] = arith.mulf [[VAR_74_1_]], [[LOAD_RES_9_MEM_3_]] : vector<16xf32>
// CHECK-DAG:           [[LOAD_RES_10_MEM_3_:%.+]] = vector.load [[VAR_reshape_4_]]{{.}}[[I_2_]]{{.}} : memref<256xf32>, vector<16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_78_1_:%.+]] = arith.mulf [[VAR_76_1_]], [[LOAD_RES_10_MEM_3_]] : vector<16xf32>
// CHECK-DAG:           [[LOAD_VAR_reshape_6_MEM_:%.+]] = vector.load [[VAR_reshape_6_]]{{.}}[[I_2_]]{{.}} : memref<256xf32>, vector<16xf32>
// CHECK:               [[VAR_80_:%.+]] = arith.addf [[VAR_78_1_]], [[LOAD_VAR_reshape_6_MEM_]] : vector<16xf32>
// CHECK:               vector.store [[VAR_80_]], [[VAR_reshape_9_]]{{.}}[[VAR_71_1_]], [[I_2_]]{{.}} : memref<128x256xf32>, vector<16xf32>
// CHECK:               [[VAR_81_:%.+]] = arith.addi [[VAR_1_]], [[CST_3_]] : index
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_7_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_81_]], [[I_2_]]{{.}} : memref<128x256xf32>, vector<16xf32>
// CHECK-DAG:           [[VAR_83_:%.+]] = vector.broadcast [[VAR_39_]] : f32 to vector<16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_84_:%.+]] = arith.subf [[LOAD_VAR_reshape_MEM_7_]], [[VAR_83_]] : vector<16xf32>
// CHECK-DAG:           [[VAR_85_:%.+]] = vector.broadcast [[VAR_45_]] : f32 to vector<16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_86_:%.+]] = arith.mulf [[VAR_84_]], [[VAR_85_]] : vector<16xf32>
// CHECK-DAG:           [[LOAD_VAR_reshape_4_MEM_:%.+]] = vector.load [[VAR_reshape_4_]]{{.}}[[I_2_]]{{.}} : memref<256xf32>, vector<16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_88_:%.+]] = arith.mulf [[VAR_86_]], [[LOAD_VAR_reshape_4_MEM_]] : vector<16xf32>
// CHECK-DAG:           [[LOAD_VAR_reshape_6_MEM_1_:%.+]] = vector.load [[VAR_reshape_6_]]{{.}}[[I_2_]]{{.}} : memref<256xf32>, vector<16xf32>
// CHECK:               [[VAR_90_:%.+]] = arith.addf [[VAR_88_]], [[LOAD_VAR_reshape_6_MEM_1_]] : vector<16xf32>
// CHECK:               vector.store [[VAR_90_]], [[VAR_reshape_9_]]{{.}}[[VAR_81_]], [[I_2_]]{{.}} : memref<128x256xf32>, vector<16xf32>
// CHECK:             }
// CHECK:             krnl.store [[VAR_6_]], [[VAR_reshape_12_]]{{.}}[[VAR_1_]], [[CST_0_]]{{.}} : memref<128x1xf32>
// CHECK:             [[VAR_46_:%.+]] = arith.addi [[VAR_1_]], [[CST_1_]] : index
// CHECK:             krnl.store [[VAR_17_]], [[VAR_reshape_12_]]{{.}}[[VAR_46_]], [[CST_0_]]{{.}} : memref<128x1xf32>
// CHECK:             [[VAR_47_:%.+]] = arith.addi [[VAR_1_]], [[CST_2_]] : index
// CHECK:             krnl.store [[VAR_28_]], [[VAR_reshape_12_]]{{.}}[[VAR_47_]], [[CST_0_]]{{.}} : memref<128x1xf32>
// CHECK:             [[VAR_48_:%.+]] = arith.addi [[VAR_1_]], [[CST_3_]] : index
// CHECK:             krnl.store [[VAR_39_]], [[VAR_reshape_12_]]{{.}}[[VAR_48_]], [[CST_0_]]{{.}} : memref<128x1xf32>
// CHECK:             krnl.store [[VAR_12_]], [[VAR_reshape_15_]]{{.}}[[VAR_1_]], [[CST_0_]]{{.}} : memref<128x1xf32>
// CHECK:             [[VAR_49_:%.+]] = arith.addi [[VAR_1_]], [[CST_1_]] : index
// CHECK:             krnl.store [[VAR_23_]], [[VAR_reshape_15_]]{{.}}[[VAR_49_]], [[CST_0_]]{{.}} : memref<128x1xf32>
// CHECK:             [[VAR_50_:%.+]] = arith.addi [[VAR_1_]], [[CST_2_]] : index
// CHECK:             krnl.store [[VAR_34_]], [[VAR_reshape_15_]]{{.}}[[VAR_50_]], [[CST_0_]]{{.}} : memref<128x1xf32>
// CHECK:             [[VAR_51_:%.+]] = arith.addi [[VAR_1_]], [[CST_3_]] : index
// CHECK:             krnl.store [[VAR_45_]], [[VAR_reshape_15_]]{{.}}[[VAR_51_]], [[CST_0_]]{{.}} : memref<128x1xf32>
// CHECK:           }
// CHECK:           return [[RES_3_]] : memref<2x64x32x8xf32>
// CHECK:         }
}

// -----

// collapsed range is not a multiple of 4, cannot do simd: Update, it is now supported.

func.func @layernorm_4D_with_scale_bias_no_SIMD(%arg0: tensor<2x64x31x3xf32>, %arg1: tensor<31x3xf32>, %arg2: tensor<31x3xf32>) -> tensor<*xf32> {
  %0 = "onnx.NoValue"() {value} : () -> none
  %Y, %Mean, %InvStdDev = "onnx.LayerNormalization"(%arg0, %arg1, %arg2) {axis = -2 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<2x64x31x3xf32>, tensor<31x3xf32>, tensor<31x3xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
  func.return %Y : tensor<*xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 + 2)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 + 3)>
// CHECK-LABEL:  func.func @layernorm_4D_with_scale_bias_no_SIMD
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x64x31x3xf32>, [[PARAM_1_:%.+]]: memref<31x3xf32>, [[PARAM_2_:%.+]]: memref<31x3xf32>) -> memref<2x64x31x3xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<1.000000e+00> : vector<32xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<9.300000e+01> : vector<4xf32>
// CHECK-DAG:       [[VAR_cst_1_:%.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
// CHECK-DAG:       [[CST_92_:%.+]] = arith.constant 92 : index
// CHECK-DAG:       [[CST_90_:%.+]] = arith.constant 90 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_93_:%.+]] = arith.constant 93 : index
// CHECK-DAG:       [[CST_11904_:%.+]] = arith.constant 11904 : index
// CHECK-DAG:       [[CST_128_:%.+]] = arith.constant 128 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_64_:%.+]] = arith.constant 64 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [1], value = dense<9.99999974E-6> : tensor<1xf32>}> : () -> memref<1xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x64x1x1xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<3xindex>
// CHECK:           affine.store [[CST_2_]], [[RES_1_]][0] : memref<3xindex>
// CHECK:           affine.store [[CST_64_]], [[RES_1_]][1] : memref<3xindex>
// CHECK:           affine.store [[CST_93_]], [[RES_1_]][2] : memref<3xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_1_]]) : (memref<2x64x31x3xf32>, memref<3xindex>) -> memref<2x64x93xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2xindex>
// CHECK:           affine.store [[CST_2_]], [[RES_2_]][0] : memref<2xindex>
// CHECK:           affine.store [[CST_64_]], [[RES_2_]][1] : memref<2xindex>
// CHECK-DAG:       [[VAR_reshape_4_:%.+]] = memref.reshape [[RES_]]([[RES_2_]]) : (memref<2x64x1x1xf32>, memref<2xindex>) -> memref<2x64xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]]#1 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[BLOCK_TILE__0_]]) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 64){
// CHECK-DAG:         [[VAR_7_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[BLOCK_TILE__0_]]) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<4x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_8_:%.+]] = affine.apply [[MAP_0_]]([[VAR_7_]]#1)
// CHECK-DAG:         [[VAR_9_:%.+]] = affine.apply [[MAP_1_]]([[VAR_7_]]#1)
// CHECK-DAG:         [[VAR_10_:%.+]] = affine.apply [[MAP_2_]]([[VAR_7_]]#1)
// CHECK:             vector.store [[VAR_cst_1_]], [[RES_3_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_cst_1_]], [[RES_3_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_cst_1_]], [[RES_3_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_cst_1_]], [[RES_3_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:             scf.for [[I_2_:%.+]] = [[CST_0_]] to [[CST_90_]] step [[CST_4_]] {
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_7_]]#0, [[VAR_7_]]#1, [[I_2_]]{{.}} : memref<2x64x93xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_1_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_7_]]#0, [[VAR_8_]], [[I_2_]]{{.}} : memref<2x64x93xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_2_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_7_]]#0, [[VAR_9_]], [[I_2_]]{{.}} : memref<2x64x93xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_3_:%.+]] = vector.load [[VAR_reshape_]]{{.}}[[VAR_7_]]#0, [[VAR_10_]], [[I_2_]]{{.}} : memref<2x64x93xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_3_MEM_:%.+]] = vector.load [[RES_3_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_3_MEM_1_:%.+]] = vector.load [[RES_3_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_3_MEM_2_:%.+]] = vector.load [[RES_3_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_3_MEM_3_:%.+]] = vector.load [[RES_3_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_45_:%.+]] = arith.addf [[LOAD_RES_3_MEM_]], [[LOAD_VAR_reshape_MEM_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_46_:%.+]] = arith.addf [[LOAD_RES_3_MEM_1_]], [[LOAD_VAR_reshape_MEM_1_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_47_:%.+]] = arith.addf [[LOAD_RES_3_MEM_2_]], [[LOAD_VAR_reshape_MEM_2_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_48_:%.+]] = arith.addf [[LOAD_RES_3_MEM_3_]], [[LOAD_VAR_reshape_MEM_3_]] : vector<4xf32>
// CHECK:               vector.store [[VAR_45_]], [[RES_3_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               vector.store [[VAR_46_]], [[RES_3_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               vector.store [[VAR_47_]], [[RES_3_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               vector.store [[VAR_48_]], [[RES_3_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:             }
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_4_:%.+]] = memref.load [[VAR_reshape_]]{{.}}[[VAR_7_]]#0, [[VAR_7_]]#1, [[CST_92_]]{{.}} : memref<2x64x93xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_5_:%.+]] = memref.load [[VAR_reshape_]]{{.}}[[VAR_7_]]#0, [[VAR_8_]], [[CST_92_]]{{.}} : memref<2x64x93xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_6_:%.+]] = memref.load [[VAR_reshape_]]{{.}}[[VAR_7_]]#0, [[VAR_9_]], [[CST_92_]]{{.}} : memref<2x64x93xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_7_:%.+]] = memref.load [[VAR_reshape_]]{{.}}[[VAR_7_]]#0, [[VAR_10_]], [[CST_92_]]{{.}} : memref<2x64x93xf32>
// CHECK-DAG:         [[LOAD_RES_3_MEM_4_:%.+]] = memref.load [[RES_3_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>
// CHECK-DAG:         [[LOAD_RES_3_MEM_5_:%.+]] = memref.load [[RES_3_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>
// CHECK-DAG:         [[LOAD_RES_3_MEM_6_:%.+]] = memref.load [[RES_3_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>
// CHECK-DAG:         [[LOAD_RES_3_MEM_7_:%.+]] = memref.load [[RES_3_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_19_:%.+]] = arith.addf [[LOAD_RES_3_MEM_4_]], [[LOAD_VAR_reshape_MEM_4_]] : f32
// CHECK-DAG:         [[VAR_20_:%.+]] = arith.addf [[LOAD_RES_3_MEM_5_]], [[LOAD_VAR_reshape_MEM_5_]] : f32
// CHECK-DAG:         [[VAR_21_:%.+]] = arith.addf [[LOAD_RES_3_MEM_6_]], [[LOAD_VAR_reshape_MEM_6_]] : f32
// CHECK-DAG:         [[VAR_22_:%.+]] = arith.addf [[LOAD_RES_3_MEM_7_]], [[LOAD_VAR_reshape_MEM_7_]] : f32
// CHECK:             memref.store [[VAR_19_]], [[RES_3_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>
// CHECK:             memref.store [[VAR_20_]], [[RES_3_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>
// CHECK:             memref.store [[VAR_21_]], [[RES_3_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>
// CHECK:             memref.store [[VAR_22_]], [[RES_3_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>
// CHECK-DAG:         [[LOAD_RES_3_MEM_8_:%.+]] = vector.load [[RES_3_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:         [[LOAD_RES_3_MEM_9_:%.+]] = vector.load [[RES_3_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:         [[LOAD_RES_3_MEM_10_:%.+]] = vector.load [[RES_3_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:         [[LOAD_RES_3_MEM_11_:%.+]] = vector.load [[RES_3_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_27_:%.+]] = vector.shuffle [[LOAD_RES_3_MEM_8_]], [[LOAD_RES_3_MEM_9_]] [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_28_:%.+]] = vector.shuffle [[LOAD_RES_3_MEM_8_]], [[LOAD_RES_3_MEM_9_]] [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_29_:%.+]] = arith.addf [[VAR_28_]], [[VAR_27_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_30_:%.+]] = vector.shuffle [[LOAD_RES_3_MEM_10_]], [[LOAD_RES_3_MEM_11_]] [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_31_:%.+]] = vector.shuffle [[LOAD_RES_3_MEM_10_]], [[LOAD_RES_3_MEM_11_]] [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK:             [[VAR_32_:%.+]] = arith.addf [[VAR_31_]], [[VAR_30_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_33_:%.+]] = vector.shuffle [[VAR_29_]], [[VAR_32_]] [0, 1, 4, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_34_:%.+]] = vector.shuffle [[VAR_29_]], [[VAR_32_]] [2, 3, 6, 7] : vector<4xf32>, vector<4xf32>
// CHECK:             [[VAR_35_:%.+]] = arith.addf [[VAR_34_]], [[VAR_33_]] : vector<4xf32>
// CHECK:             [[VAR_36_:%.+]] = arith.divf [[VAR_35_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK:             vector.store [[VAR_36_]], [[VAR_reshape_4_]]{{.}}[[VAR_7_]]#0, [[VAR_7_]]#1] : memref<2x64xf32>, vector<4xf32>
// CHECK:           }
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<2x64x1x1xf32>
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_128_]], [[RES_5_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_7_:%.+]] = memref.reshape [[RES_]]([[RES_5_]]) : (memref<2x64x1x1xf32>, memref<1xindex>) -> memref<128xf32>
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_128_]], [[RES_6_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_9_:%.+]] = memref.reshape [[RES_]]([[RES_6_]]) : (memref<2x64x1x1xf32>, memref<1xindex>) -> memref<128xf32>
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_128_]], [[RES_7_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_11_:%.+]] = memref.reshape [[RES_4_]]([[RES_7_]]) : (memref<2x64x1x1xf32>, memref<1xindex>) -> memref<128xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_1_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__1_]]) with ([[LOOP_1_]] -> [[I_3_:%.+]] = 0 to 128){
// CHECK:               [[VAR_8_1_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__1_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[VAR_9_1_:%.+]] = vector.load [[VAR_reshape_7_]]{{.}}[[VAR_8_1_]]{{.}} : memref<128xf32>, vector<32xf32>
// CHECK-DAG:           [[VAR_10_1_:%.+]] = vector.load [[VAR_reshape_9_]]{{.}}[[VAR_8_1_]]{{.}} : memref<128xf32>, vector<32xf32>
// CHECK:               [[LOAD_VAR_reshape_MEM_4_:%.+]] = arith.mulf [[VAR_9_1_]], [[VAR_10_1_]] : vector<32xf32>
// CHECK:               vector.store [[LOAD_VAR_reshape_MEM_4_]], [[VAR_reshape_11_]]{{.}}[[VAR_8_1_]]{{.}} : memref<128xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_8_:%.+]] = memref.alloc() {{.*}}: memref<2x64x31x3xf32>
// CHECK-DAG:       [[RES_9_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_11904_]], [[RES_9_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_14_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_9_]]) : (memref<2x64x31x3xf32>, memref<1xindex>) -> memref<11904xf32>
// CHECK-DAG:       [[RES_10_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_11904_]], [[RES_10_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_16_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_10_]]) : (memref<2x64x31x3xf32>, memref<1xindex>) -> memref<11904xf32>
// CHECK-DAG:       [[RES_11_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_11904_]], [[RES_11_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_18_:%.+]] = memref.reshape [[RES_8_]]([[RES_11_]]) : (memref<2x64x31x3xf32>, memref<1xindex>) -> memref<11904xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__2_:%.+]], [[BLOCK_IN__2_:%.+]] = krnl.block [[LOOP_2_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__2_]]) with ([[LOOP_2_]] -> [[I_4_:%.+]] = 0 to 11904){
// CHECK:               [[VAR_8_2_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__2_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[VAR_9_1_:%.+]] = vector.load [[VAR_reshape_14_]]{{.}}[[VAR_8_2_]]{{.}} : memref<11904xf32>, vector<32xf32>
// CHECK-DAG:           [[VAR_10_1_:%.+]] = vector.load [[VAR_reshape_16_]]{{.}}[[VAR_8_2_]]{{.}} : memref<11904xf32>, vector<32xf32>
// CHECK:               [[LOAD_VAR_reshape_MEM_4_1_:%.+]] = arith.mulf [[VAR_9_1_]], [[VAR_10_1_]] : vector<32xf32>
// CHECK:               vector.store [[LOAD_VAR_reshape_MEM_4_1_]], [[VAR_reshape_18_]]{{.}}[[VAR_8_2_]]{{.}} : memref<11904xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_12_:%.+]] = memref.alloc() {{.*}}: memref<2x64x1x1xf32>
// CHECK-DAG:       [[RES_13_:%.+]] = memref.alloc() {{.*}}: memref<3xindex>
// CHECK:           affine.store [[CST_2_]], [[RES_13_]][0] : memref<3xindex>
// CHECK:           affine.store [[CST_64_]], [[RES_13_]][1] : memref<3xindex>
// CHECK:           affine.store [[CST_93_]], [[RES_13_]][2] : memref<3xindex>
// CHECK-DAG:       [[VAR_reshape_21_:%.+]] = memref.reshape [[RES_8_]]([[RES_13_]]) : (memref<2x64x31x3xf32>, memref<3xindex>) -> memref<2x64x93xf32>
// CHECK-DAG:       [[RES_14_:%.+]] = memref.alloc() {{.*}}: memref<2xindex>
// CHECK:           affine.store [[CST_2_]], [[RES_14_]][0] : memref<2xindex>
// CHECK:           affine.store [[CST_64_]], [[RES_14_]][1] : memref<2xindex>
// CHECK-DAG:       [[VAR_reshape_23_:%.+]] = memref.reshape [[RES_12_]]([[RES_14_]]) : (memref<2x64x1x1xf32>, memref<2xindex>) -> memref<2x64xf32>
// CHECK-DAG:       [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:           [[BLOCK_TILE__3_:%.+]], [[BLOCK_IN__3_:%.+]] = krnl.block [[LOOP_3_]]#1 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[BLOCK_TILE__3_]]) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = 0 to 2, [[LOOP_3_]]#1 -> [[I_6_:%.+]] = 0 to 64){
// CHECK-DAG:         [[VAR_7_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[BLOCK_TILE__3_]]) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[RES_15_:%.+]] = memref.alloc() {{.*}}: memref<4x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_8_3_:%.+]] = affine.apply [[MAP_0_]]([[VAR_7_1_]]#1)
// CHECK-DAG:         [[VAR_9_2_:%.+]] = affine.apply [[MAP_1_]]([[VAR_7_1_]]#1)
// CHECK-DAG:         [[VAR_10_2_:%.+]] = affine.apply [[MAP_2_]]([[VAR_7_1_]]#1)
// CHECK:             vector.store [[VAR_cst_1_]], [[RES_15_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_cst_1_]], [[RES_15_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_cst_1_]], [[RES_15_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:             vector.store [[VAR_cst_1_]], [[RES_15_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:             scf.for [[I_7_:%.+]] = [[CST_0_]] to [[CST_90_]] step [[CST_4_]] {
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_8_:%.+]] = vector.load [[VAR_reshape_21_]]{{.}}[[VAR_7_1_]]#0, [[VAR_7_1_]]#1, [[I_7_]]{{.}} : memref<2x64x93xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_1_:%.+]] = vector.load [[VAR_reshape_21_]]{{.}}[[VAR_7_1_]]#0, [[VAR_8_3_]], [[I_7_]]{{.}} : memref<2x64x93xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_2_:%.+]] = vector.load [[VAR_reshape_21_]]{{.}}[[VAR_7_1_]]#0, [[VAR_9_2_]], [[I_7_]]{{.}} : memref<2x64x93xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_3_:%.+]] = vector.load [[VAR_reshape_21_]]{{.}}[[VAR_7_1_]]#0, [[VAR_10_2_]], [[I_7_]]{{.}} : memref<2x64x93xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_3_MEM_12_:%.+]] = vector.load [[RES_15_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_3_MEM_1_:%.+]] = vector.load [[RES_15_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_3_MEM_2_:%.+]] = vector.load [[RES_15_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_3_MEM_3_:%.+]] = vector.load [[RES_15_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_45_1_:%.+]] = arith.addf [[LOAD_RES_3_MEM_12_]], [[LOAD_VAR_reshape_MEM_8_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_46_1_:%.+]] = arith.addf [[LOAD_RES_3_MEM_1_]], [[LOAD_VAR_reshape_MEM_1_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_47_1_:%.+]] = arith.addf [[LOAD_RES_3_MEM_2_]], [[LOAD_VAR_reshape_MEM_2_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_48_1_:%.+]] = arith.addf [[LOAD_RES_3_MEM_3_]], [[LOAD_VAR_reshape_MEM_3_]] : vector<4xf32>
// CHECK:               vector.store [[VAR_45_1_]], [[RES_15_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               vector.store [[VAR_46_1_]], [[RES_15_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               vector.store [[VAR_47_1_]], [[RES_15_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               vector.store [[VAR_48_1_]], [[RES_15_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:             }
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_4_1_:%.+]] = memref.load [[VAR_reshape_21_]]{{.}}[[VAR_7_1_]]#0, [[VAR_7_1_]]#1, [[CST_92_]]{{.}} : memref<2x64x93xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_5_:%.+]] = memref.load [[VAR_reshape_21_]]{{.}}[[VAR_7_1_]]#0, [[VAR_8_3_]], [[CST_92_]]{{.}} : memref<2x64x93xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_6_:%.+]] = memref.load [[VAR_reshape_21_]]{{.}}[[VAR_7_1_]]#0, [[VAR_9_2_]], [[CST_92_]]{{.}} : memref<2x64x93xf32>
// CHECK-DAG:         [[LOAD_VAR_reshape_MEM_7_:%.+]] = memref.load [[VAR_reshape_21_]]{{.}}[[VAR_7_1_]]#0, [[VAR_10_2_]], [[CST_92_]]{{.}} : memref<2x64x93xf32>
// CHECK-DAG:         [[LOAD_RES_3_MEM_4_:%.+]] = memref.load [[RES_15_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>
// CHECK-DAG:         [[LOAD_RES_3_MEM_5_:%.+]] = memref.load [[RES_15_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>
// CHECK-DAG:         [[LOAD_RES_3_MEM_6_:%.+]] = memref.load [[RES_15_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>
// CHECK-DAG:         [[LOAD_RES_3_MEM_7_:%.+]] = memref.load [[RES_15_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_19_1_:%.+]] = arith.addf [[LOAD_RES_3_MEM_4_]], [[LOAD_VAR_reshape_MEM_4_1_]] : f32
// CHECK-DAG:         [[VAR_20_1_:%.+]] = arith.addf [[LOAD_RES_3_MEM_5_]], [[LOAD_VAR_reshape_MEM_5_]] : f32
// CHECK-DAG:         [[VAR_21_1_:%.+]] = arith.addf [[LOAD_RES_3_MEM_6_]], [[LOAD_VAR_reshape_MEM_6_]] : f32
// CHECK-DAG:         [[VAR_22_1_:%.+]] = arith.addf [[LOAD_RES_3_MEM_7_]], [[LOAD_VAR_reshape_MEM_7_]] : f32
// CHECK:             memref.store [[VAR_19_1_]], [[RES_15_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>
// CHECK:             memref.store [[VAR_20_1_]], [[RES_15_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>
// CHECK:             memref.store [[VAR_21_1_]], [[RES_15_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>
// CHECK:             memref.store [[VAR_22_1_]], [[RES_15_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>
// CHECK-DAG:         [[LOAD_RES_3_MEM_8_:%.+]] = vector.load [[RES_15_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:         [[LOAD_RES_3_MEM_9_:%.+]] = vector.load [[RES_15_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:         [[LOAD_RES_3_MEM_10_:%.+]] = vector.load [[RES_15_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:         [[LOAD_RES_3_MEM_11_:%.+]] = vector.load [[RES_15_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_27_1_:%.+]] = vector.shuffle [[LOAD_RES_3_MEM_8_]], [[LOAD_RES_3_MEM_9_]] [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_28_1_:%.+]] = vector.shuffle [[LOAD_RES_3_MEM_8_]], [[LOAD_RES_3_MEM_9_]] [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_29_1_:%.+]] = arith.addf [[VAR_28_1_]], [[VAR_27_1_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_30_1_:%.+]] = vector.shuffle [[LOAD_RES_3_MEM_10_]], [[LOAD_RES_3_MEM_11_]] [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_31_1_:%.+]] = vector.shuffle [[LOAD_RES_3_MEM_10_]], [[LOAD_RES_3_MEM_11_]] [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK:             [[VAR_32_1_:%.+]] = arith.addf [[VAR_31_1_]], [[VAR_30_1_]] : vector<4xf32>
// CHECK-DAG:         [[VAR_33_1_:%.+]] = vector.shuffle [[VAR_29_1_]], [[VAR_32_1_]] [0, 1, 4, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:         [[VAR_34_1_:%.+]] = vector.shuffle [[VAR_29_1_]], [[VAR_32_1_]] [2, 3, 6, 7] : vector<4xf32>, vector<4xf32>
// CHECK:             [[VAR_35_1_:%.+]] = arith.addf [[VAR_34_1_]], [[VAR_33_1_]] : vector<4xf32>
// CHECK:             [[VAR_36_1_:%.+]] = arith.divf [[VAR_35_1_]], [[VAR_cst_0_]] : vector<4xf32>
// CHECK:             vector.store [[VAR_36_1_]], [[VAR_reshape_23_]]{{.}}[[VAR_7_1_]]#0, [[VAR_7_1_]]#1] : memref<2x64xf32>, vector<4xf32>
// CHECK:           }
// CHECK-DAG:       [[RES_16_:%.+]] = memref.alloc() {{.*}}: memref<2x64x1x1xf32>
// CHECK-DAG:       [[RES_17_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_128_]], [[RES_17_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_28_:%.+]] = memref.reshape [[RES_12_]]([[RES_17_]]) : (memref<2x64x1x1xf32>, memref<1xindex>) -> memref<128xf32>
// CHECK-DAG:       [[RES_18_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_128_]], [[RES_18_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_30_:%.+]] = memref.reshape [[RES_4_]]([[RES_18_]]) : (memref<2x64x1x1xf32>, memref<1xindex>) -> memref<128xf32>
// CHECK-DAG:       [[RES_19_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_128_]], [[RES_19_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_32_:%.+]] = memref.reshape [[RES_16_]]([[RES_19_]]) : (memref<2x64x1x1xf32>, memref<1xindex>) -> memref<128xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__4_:%.+]], [[BLOCK_IN__4_:%.+]] = krnl.block [[LOOP_4_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__4_]]) with ([[LOOP_4_]] -> [[I_8_:%.+]] = 0 to 128){
// CHECK:               [[VAR_8_4_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__4_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[VAR_9_2_:%.+]] = vector.load [[VAR_reshape_28_]]{{.}}[[VAR_8_4_]]{{.}} : memref<128xf32>, vector<32xf32>
// CHECK-DAG:           [[VAR_10_2_:%.+]] = vector.load [[VAR_reshape_30_]]{{.}}[[VAR_8_4_]]{{.}} : memref<128xf32>, vector<32xf32>
// CHECK:               [[LOAD_VAR_reshape_MEM_4_1_1_:%.+]] = arith.subf [[VAR_9_2_]], [[VAR_10_2_]] : vector<32xf32>
// CHECK:               vector.store [[LOAD_VAR_reshape_MEM_4_1_1_]], [[VAR_reshape_32_]]{{.}}[[VAR_8_4_]]{{.}} : memref<128xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_20_:%.+]] = memref.alloc() {{.*}}: memref<2x64x31x3xf32>
// CHECK-DAG:       [[RES_21_:%.+]] = memref.alloc() {{.*}}: memref<3xindex>
// CHECK:           affine.store [[CST_2_]], [[RES_21_]][0] : memref<3xindex>
// CHECK:           affine.store [[CST_64_]], [[RES_21_]][1] : memref<3xindex>
// CHECK:           affine.store [[CST_93_]], [[RES_21_]][2] : memref<3xindex>
// CHECK-DAG:       [[VAR_reshape_35_:%.+]] = memref.reshape [[PARAM_0_]]([[RES_21_]]) : (memref<2x64x31x3xf32>, memref<3xindex>) -> memref<2x64x93xf32>
// CHECK-DAG:       [[RES_22_:%.+]] = memref.alloc() {{.*}}: memref<3xindex>
// CHECK:           affine.store [[CST_2_]], [[RES_22_]][0] : memref<3xindex>
// CHECK:           affine.store [[CST_64_]], [[RES_22_]][1] : memref<3xindex>
// CHECK:           affine.store [[CST_1_]], [[RES_22_]][2] : memref<3xindex>
// CHECK-DAG:       [[VAR_reshape_37_:%.+]] = memref.reshape [[RES_]]([[RES_22_]]) : (memref<2x64x1x1xf32>, memref<3xindex>) -> memref<2x64x1xf32>
// CHECK-DAG:       [[RES_23_:%.+]] = memref.alloc() {{.*}}: memref<3xindex>
// CHECK:           affine.store [[CST_2_]], [[RES_23_]][0] : memref<3xindex>
// CHECK:           affine.store [[CST_64_]], [[RES_23_]][1] : memref<3xindex>
// CHECK:           affine.store [[CST_93_]], [[RES_23_]][2] : memref<3xindex>
// CHECK-DAG:       [[VAR_reshape_39_:%.+]] = memref.reshape [[RES_20_]]([[RES_23_]]) : (memref<2x64x31x3xf32>, memref<3xindex>) -> memref<2x64x93xf32>
// CHECK-DAG:       [[LOOP_5_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_5_]]#0, [[LOOP_5_]]#1) with ([[LOOP_5_]]#0 -> [[I_9_:%.+]] = 0 to 2, [[LOOP_5_]]#1 -> [[I_10_:%.+]] = 0 to 64){
// CHECK-DAG:         [[VAR_7_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_5_]]#0, [[LOOP_5_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_6_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__5_:%.+]], [[BLOCK_IN__5_:%.+]] = krnl.block [[LOOP_6_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__5_]]) with ([[LOOP_6_]] -> [[I_11_:%.+]] = 0 to 62){
// CHECK:               [[VAR_10_3_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__5_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_4_1_1_:%.+]] = vector.load [[VAR_reshape_35_]]{{.}}[[VAR_7_2_]]#0, [[VAR_7_2_]]#1, [[VAR_10_3_]]{{.}} : memref<2x64x93xf32>, vector<32xf32>
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_5_1_:%.+]] = krnl.load [[VAR_reshape_37_]]{{.}}[[VAR_7_2_]]#0, [[VAR_7_2_]]#1, [[CST_0_]]{{.}} : memref<2x64x1xf32>
// CHECK:               [[LOAD_VAR_reshape_MEM_6_1_:%.+]] = vector.broadcast [[LOAD_VAR_reshape_MEM_5_1_]] : f32 to vector<32xf32>
// CHECK:               [[LOAD_VAR_reshape_MEM_7_1_:%.+]] = arith.subf [[LOAD_VAR_reshape_MEM_4_1_1_]], [[LOAD_VAR_reshape_MEM_6_1_]] : vector<32xf32>
// CHECK:               vector.store [[LOAD_VAR_reshape_MEM_7_1_]], [[VAR_reshape_39_]]{{.}}[[VAR_7_2_]]#0, [[VAR_7_2_]]#1, [[VAR_10_3_]]{{.}} : memref<2x64x93xf32>, vector<32xf32>
// CHECK:             }
// CHECK:             [[LOOP_7_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_7_]]) with ([[LOOP_7_]] -> [[I_12_:%.+]] = 64 to 93){
// CHECK:               [[VAR_10_4_:%.+]] = krnl.get_induction_var_value([[LOOP_7_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_4_1_1_1_:%.+]] = krnl.load [[VAR_reshape_35_]]{{.}}[[VAR_7_2_]]#0, [[VAR_7_2_]]#1, [[VAR_10_4_]]{{.}} : memref<2x64x93xf32>
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_5_1_:%.+]] = krnl.load [[VAR_reshape_37_]]{{.}}[[VAR_7_2_]]#0, [[VAR_7_2_]]#1, [[CST_0_]]{{.}} : memref<2x64x1xf32>
// CHECK:               [[LOAD_VAR_reshape_MEM_6_1_:%.+]] = arith.subf [[LOAD_VAR_reshape_MEM_4_1_1_1_]], [[LOAD_VAR_reshape_MEM_5_1_]] : f32
// CHECK:               krnl.store [[LOAD_VAR_reshape_MEM_6_1_]], [[VAR_reshape_39_]]{{.}}[[VAR_7_2_]]#0, [[VAR_7_2_]]#1, [[VAR_10_4_]]{{.}} : memref<2x64x93xf32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_24_:%.+]] = memref.alloc() {{.*}}: memref<2x64x1x1xf32>
// CHECK-DAG:       [[RES_25_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_128_]], [[RES_25_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_42_:%.+]] = memref.reshape [[RES_16_]]([[RES_25_]]) : (memref<2x64x1x1xf32>, memref<1xindex>) -> memref<128xf32>
// CHECK-DAG:       [[RES_26_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_128_]], [[RES_26_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_44_:%.+]] = memref.reshape [[RES_24_]]([[RES_26_]]) : (memref<2x64x1x1xf32>, memref<1xindex>) -> memref<128xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_8_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__6_:%.+]], [[BLOCK_IN__6_:%.+]] = krnl.block [[LOOP_8_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__6_]]) with ([[LOOP_8_]] -> [[I_13_:%.+]] = 0 to 128){
// CHECK:               [[VAR_8_5_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__6_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOOP_7_:%.+]] = vector.load [[VAR_reshape_42_]]{{.}}[[VAR_8_5_]]{{.}} : memref<128xf32>, vector<32xf32>
// CHECK-DAG:           [[VAR_10_4_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[CST_0_]]{{.}} : memref<1xf32>
// CHECK:               [[LOAD_VAR_reshape_MEM_4_1_1_1_:%.+]] = vector.broadcast [[VAR_10_4_]] : f32 to vector<32xf32>
// CHECK:               [[LOAD_VAR_reshape_MEM_5_1_1_:%.+]] = arith.addf [[LOOP_7_]], [[LOAD_VAR_reshape_MEM_4_1_1_1_]] : vector<32xf32>
// CHECK:               vector.store [[LOAD_VAR_reshape_MEM_5_1_1_]], [[VAR_reshape_44_]]{{.}}[[VAR_8_5_]]{{.}} : memref<128xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_27_:%.+]] = memref.alloc() {{.*}}: memref<2x64x1x1xf32>
// CHECK-DAG:       [[RES_28_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_128_]], [[RES_28_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_47_:%.+]] = memref.reshape [[RES_24_]]([[RES_28_]]) : (memref<2x64x1x1xf32>, memref<1xindex>) -> memref<128xf32>
// CHECK-DAG:       [[RES_29_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_128_]], [[RES_29_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_49_:%.+]] = memref.reshape [[RES_27_]]([[RES_29_]]) : (memref<2x64x1x1xf32>, memref<1xindex>) -> memref<128xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_9_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__7_:%.+]], [[BLOCK_IN__7_:%.+]] = krnl.block [[LOOP_9_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__7_]]) with ([[LOOP_9_]] -> [[I_14_:%.+]] = 0 to 128){
// CHECK:               [[VAR_8_6_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__7_]]) : (!krnl.loop) -> index
// CHECK:               [[LOOP_7_1_:%.+]] = vector.load [[VAR_reshape_47_]]{{.}}[[VAR_8_6_]]{{.}} : memref<128xf32>, vector<32xf32>
// CHECK:               [[VAR_10_5_:%.+]] = math.sqrt [[LOOP_7_1_]] : vector<32xf32>
// CHECK:               vector.store [[VAR_10_5_]], [[VAR_reshape_49_]]{{.}}[[VAR_8_6_]]{{.}} : memref<128xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_30_:%.+]] = memref.alloc() {{.*}}: memref<2x64x1x1xf32>
// CHECK-DAG:       [[RES_31_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_128_]], [[RES_31_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_52_:%.+]] = memref.reshape [[RES_27_]]([[RES_31_]]) : (memref<2x64x1x1xf32>, memref<1xindex>) -> memref<128xf32>
// CHECK-DAG:       [[RES_32_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_128_]], [[RES_32_]][0] : memref<1xindex>
// CHECK:           [[VAR_reshape_54_:%.+]] = memref.reshape [[RES_30_]]([[RES_32_]]) : (memref<2x64x1x1xf32>, memref<1xindex>) -> memref<128xf32>
// CHECK:           krnl.iterate() with (){
// CHECK:             [[LOOP_10_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__8_:%.+]], [[BLOCK_IN__8_:%.+]] = krnl.block [[LOOP_10_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__8_]]) with ([[LOOP_10_]] -> [[I_15_:%.+]] = 0 to 128){
// CHECK:               [[VAR_8_7_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__8_]]) : (!krnl.loop) -> index
// CHECK:               [[LOOP_7_1_:%.+]] = vector.load [[VAR_reshape_52_]]{{.}}[[VAR_8_7_]]{{.}} : memref<128xf32>, vector<32xf32>
// CHECK:               [[VAR_10_6_:%.+]] = arith.divf [[VAR_cst_]], [[LOOP_7_1_]] : vector<32xf32>
// CHECK:               vector.store [[VAR_10_6_]], [[VAR_reshape_54_]]{{.}}[[VAR_8_7_]]{{.}} : memref<128xf32>, vector<32xf32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_33_:%.+]] = memref.alloc() {{.*}}: memref<2x64x31x3xf32>
// CHECK-DAG:       [[RES_34_:%.+]] = memref.alloc() {{.*}}: memref<3xindex>
// CHECK:           affine.store [[CST_2_]], [[RES_34_]][0] : memref<3xindex>
// CHECK:           affine.store [[CST_64_]], [[RES_34_]][1] : memref<3xindex>
// CHECK:           affine.store [[CST_93_]], [[RES_34_]][2] : memref<3xindex>
// CHECK-DAG:       [[VAR_reshape_57_:%.+]] = memref.reshape [[RES_20_]]([[RES_34_]]) : (memref<2x64x31x3xf32>, memref<3xindex>) -> memref<2x64x93xf32>
// CHECK-DAG:       [[RES_35_:%.+]] = memref.alloc() {{.*}}: memref<3xindex>
// CHECK:           affine.store [[CST_2_]], [[RES_35_]][0] : memref<3xindex>
// CHECK:           affine.store [[CST_64_]], [[RES_35_]][1] : memref<3xindex>
// CHECK:           affine.store [[CST_1_]], [[RES_35_]][2] : memref<3xindex>
// CHECK-DAG:       [[VAR_reshape_59_:%.+]] = memref.reshape [[RES_30_]]([[RES_35_]]) : (memref<2x64x1x1xf32>, memref<3xindex>) -> memref<2x64x1xf32>
// CHECK-DAG:       [[RES_36_:%.+]] = memref.alloc() {{.*}}: memref<3xindex>
// CHECK:           affine.store [[CST_2_]], [[RES_36_]][0] : memref<3xindex>
// CHECK:           affine.store [[CST_64_]], [[RES_36_]][1] : memref<3xindex>
// CHECK:           affine.store [[CST_93_]], [[RES_36_]][2] : memref<3xindex>
// CHECK-DAG:       [[VAR_reshape_61_:%.+]] = memref.reshape [[RES_33_]]([[RES_36_]]) : (memref<2x64x31x3xf32>, memref<3xindex>) -> memref<2x64x93xf32>
// CHECK-DAG:       [[LOOP_11_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_11_]]#0, [[LOOP_11_]]#1) with ([[LOOP_11_]]#0 -> [[I_16_:%.+]] = 0 to 2, [[LOOP_11_]]#1 -> [[I_17_:%.+]] = 0 to 64){
// CHECK-DAG:         [[VAR_7_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_11_]]#0, [[LOOP_11_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_12_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__9_:%.+]], [[BLOCK_IN__9_:%.+]] = krnl.block [[LOOP_12_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__9_]]) with ([[LOOP_12_]] -> [[I_18_:%.+]] = 0 to 62){
// CHECK:               [[VAR_10_7_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__9_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_4_1_1_1_1_:%.+]] = vector.load [[VAR_reshape_57_]]{{.}}[[VAR_7_3_]]#0, [[VAR_7_3_]]#1, [[VAR_10_7_]]{{.}} : memref<2x64x93xf32>, vector<32xf32>
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_5_1_1_:%.+]] = krnl.load [[VAR_reshape_59_]]{{.}}[[VAR_7_3_]]#0, [[VAR_7_3_]]#1, [[CST_0_]]{{.}} : memref<2x64x1xf32>
// CHECK:               [[LOAD_VAR_reshape_MEM_6_1_1_:%.+]] = vector.broadcast [[LOAD_VAR_reshape_MEM_5_1_1_]] : f32 to vector<32xf32>
// CHECK:               [[LOAD_VAR_reshape_MEM_7_1_:%.+]] = arith.mulf [[LOAD_VAR_reshape_MEM_4_1_1_1_1_]], [[LOAD_VAR_reshape_MEM_6_1_1_]] : vector<32xf32>
// CHECK:               vector.store [[LOAD_VAR_reshape_MEM_7_1_]], [[VAR_reshape_61_]]{{.}}[[VAR_7_3_]]#0, [[VAR_7_3_]]#1, [[VAR_10_7_]]{{.}} : memref<2x64x93xf32>, vector<32xf32>
// CHECK:             }
// CHECK:             [[LOOP_13_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_13_]]) with ([[LOOP_13_]] -> [[I_19_:%.+]] = 64 to 93){
// CHECK:               [[VAR_10_8_:%.+]] = krnl.get_induction_var_value([[LOOP_13_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_4_1_1_1_1_:%.+]] = krnl.load [[VAR_reshape_57_]]{{.}}[[VAR_7_3_]]#0, [[VAR_7_3_]]#1, [[VAR_10_8_]]{{.}} : memref<2x64x93xf32>
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_5_1_1_1_:%.+]] = krnl.load [[VAR_reshape_59_]]{{.}}[[VAR_7_3_]]#0, [[VAR_7_3_]]#1, [[CST_0_]]{{.}} : memref<2x64x1xf32>
// CHECK:               [[LOAD_VAR_reshape_MEM_6_1_1_:%.+]] = arith.mulf [[LOAD_VAR_reshape_MEM_4_1_1_1_1_]], [[LOAD_VAR_reshape_MEM_5_1_1_1_]] : f32
// CHECK:               krnl.store [[LOAD_VAR_reshape_MEM_6_1_1_]], [[VAR_reshape_61_]]{{.}}[[VAR_7_3_]]#0, [[VAR_7_3_]]#1, [[VAR_10_8_]]{{.}} : memref<2x64x93xf32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_37_:%.+]] = memref.alloc() {{.*}}: memref<2x64x31x3xf32>
// CHECK-DAG:       [[RES_38_:%.+]] = memref.alloc() {{.*}}: memref<3xindex>
// CHECK:           affine.store [[CST_2_]], [[RES_38_]][0] : memref<3xindex>
// CHECK:           affine.store [[CST_64_]], [[RES_38_]][1] : memref<3xindex>
// CHECK:           affine.store [[CST_93_]], [[RES_38_]][2] : memref<3xindex>
// CHECK-DAG:       [[VAR_reshape_64_:%.+]] = memref.reshape [[RES_33_]]([[RES_38_]]) : (memref<2x64x31x3xf32>, memref<3xindex>) -> memref<2x64x93xf32>
// CHECK-DAG:       [[RES_39_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_93_]], [[RES_39_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_66_:%.+]] = memref.reshape [[PARAM_1_]]([[RES_39_]]) : (memref<31x3xf32>, memref<1xindex>) -> memref<93xf32>
// CHECK-DAG:       [[RES_40_:%.+]] = memref.alloc() {{.*}}: memref<3xindex>
// CHECK:           affine.store [[CST_2_]], [[RES_40_]][0] : memref<3xindex>
// CHECK:           affine.store [[CST_64_]], [[RES_40_]][1] : memref<3xindex>
// CHECK:           affine.store [[CST_93_]], [[RES_40_]][2] : memref<3xindex>
// CHECK-DAG:       [[VAR_reshape_68_:%.+]] = memref.reshape [[RES_37_]]([[RES_40_]]) : (memref<2x64x31x3xf32>, memref<3xindex>) -> memref<2x64x93xf32>
// CHECK-DAG:       [[LOOP_14_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_14_]]#0, [[LOOP_14_]]#1) with ([[LOOP_14_]]#0 -> [[I_20_:%.+]] = 0 to 2, [[LOOP_14_]]#1 -> [[I_21_:%.+]] = 0 to 64){
// CHECK-DAG:         [[VAR_7_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_14_]]#0, [[LOOP_14_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_15_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__10_:%.+]], [[BLOCK_IN__10_:%.+]] = krnl.block [[LOOP_15_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__10_]]) with ([[LOOP_15_]] -> [[I_22_:%.+]] = 0 to 62){
// CHECK:               [[VAR_10_9_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__10_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_4_1_1_1_1_1_:%.+]] = vector.load [[VAR_reshape_64_]]{{.}}[[VAR_7_4_]]#0, [[VAR_7_4_]]#1, [[VAR_10_9_]]{{.}} : memref<2x64x93xf32>, vector<32xf32>
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_5_1_1_1_:%.+]] = vector.load [[VAR_reshape_66_]]{{.}}[[VAR_10_9_]]{{.}} : memref<93xf32>, vector<32xf32>
// CHECK:               [[LOAD_VAR_reshape_MEM_6_1_1_1_:%.+]] = arith.mulf [[LOAD_VAR_reshape_MEM_4_1_1_1_1_1_]], [[LOAD_VAR_reshape_MEM_5_1_1_1_]] : vector<32xf32>
// CHECK:               vector.store [[LOAD_VAR_reshape_MEM_6_1_1_1_]], [[VAR_reshape_68_]]{{.}}[[VAR_7_4_]]#0, [[VAR_7_4_]]#1, [[VAR_10_9_]]{{.}} : memref<2x64x93xf32>, vector<32xf32>
// CHECK:             }
// CHECK:             [[LOOP_16_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_16_]]) with ([[LOOP_16_]] -> [[I_23_:%.+]] = 64 to 93){
// CHECK:               [[VAR_10_10_:%.+]] = krnl.get_induction_var_value([[LOOP_16_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_4_1_1_1_1_1_:%.+]] = krnl.load [[VAR_reshape_64_]]{{.}}[[VAR_7_4_]]#0, [[VAR_7_4_]]#1, [[VAR_10_10_]]{{.}} : memref<2x64x93xf32>
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_5_1_1_1_1_:%.+]] = krnl.load [[VAR_reshape_66_]]{{.}}[[VAR_10_10_]]{{.}} : memref<93xf32>
// CHECK:               [[LOAD_VAR_reshape_MEM_6_1_1_1_:%.+]] = arith.mulf [[LOAD_VAR_reshape_MEM_4_1_1_1_1_1_]], [[LOAD_VAR_reshape_MEM_5_1_1_1_1_]] : f32
// CHECK:               krnl.store [[LOAD_VAR_reshape_MEM_6_1_1_1_]], [[VAR_reshape_68_]]{{.}}[[VAR_7_4_]]#0, [[VAR_7_4_]]#1, [[VAR_10_10_]]{{.}} : memref<2x64x93xf32>
// CHECK:             }
// CHECK:           }
// CHECK-DAG:       [[RES_41_:%.+]] = memref.alloc() {{.*}}: memref<2x64x31x3xf32>
// CHECK-DAG:       [[RES_42_:%.+]] = memref.alloc() {{.*}}: memref<3xindex>
// CHECK:           affine.store [[CST_2_]], [[RES_42_]][0] : memref<3xindex>
// CHECK:           affine.store [[CST_64_]], [[RES_42_]][1] : memref<3xindex>
// CHECK:           affine.store [[CST_93_]], [[RES_42_]][2] : memref<3xindex>
// CHECK-DAG:       [[VAR_reshape_71_:%.+]] = memref.reshape [[RES_37_]]([[RES_42_]]) : (memref<2x64x31x3xf32>, memref<3xindex>) -> memref<2x64x93xf32>
// CHECK-DAG:       [[RES_43_:%.+]] = memref.alloc() {{.*}}: memref<1xindex>
// CHECK:           affine.store [[CST_93_]], [[RES_43_]][0] : memref<1xindex>
// CHECK-DAG:       [[VAR_reshape_73_:%.+]] = memref.reshape [[PARAM_2_]]([[RES_43_]]) : (memref<31x3xf32>, memref<1xindex>) -> memref<93xf32>
// CHECK-DAG:       [[RES_44_:%.+]] = memref.alloc() {{.*}}: memref<3xindex>
// CHECK:           affine.store [[CST_2_]], [[RES_44_]][0] : memref<3xindex>
// CHECK:           affine.store [[CST_64_]], [[RES_44_]][1] : memref<3xindex>
// CHECK:           affine.store [[CST_93_]], [[RES_44_]][2] : memref<3xindex>
// CHECK-DAG:       [[VAR_reshape_75_:%.+]] = memref.reshape [[RES_41_]]([[RES_44_]]) : (memref<2x64x31x3xf32>, memref<3xindex>) -> memref<2x64x93xf32>
// CHECK-DAG:       [[LOOP_17_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_17_]]#0, [[LOOP_17_]]#1) with ([[LOOP_17_]]#0 -> [[I_24_:%.+]] = 0 to 2, [[LOOP_17_]]#1 -> [[I_25_:%.+]] = 0 to 64){
// CHECK-DAG:         [[VAR_7_5_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_17_]]#0, [[LOOP_17_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_18_:%.+]] = krnl.define_loops 1
// CHECK:             [[BLOCK_TILE__11_:%.+]], [[BLOCK_IN__11_:%.+]] = krnl.block [[LOOP_18_]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:             krnl.iterate([[BLOCK_TILE__11_]]) with ([[LOOP_18_]] -> [[I_26_:%.+]] = 0 to 62){
// CHECK:               [[VAR_10_11_:%.+]] = krnl.get_induction_var_value([[BLOCK_TILE__11_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_4_1_1_1_1_1_1_:%.+]] = vector.load [[VAR_reshape_71_]]{{.}}[[VAR_7_5_]]#0, [[VAR_7_5_]]#1, [[VAR_10_11_]]{{.}} : memref<2x64x93xf32>, vector<32xf32>
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_5_1_1_1_1_:%.+]] = vector.load [[VAR_reshape_73_]]{{.}}[[VAR_10_11_]]{{.}} : memref<93xf32>, vector<32xf32>
// CHECK:               [[LOAD_VAR_reshape_MEM_6_1_1_1_1_:%.+]] = arith.addf [[LOAD_VAR_reshape_MEM_4_1_1_1_1_1_1_]], [[LOAD_VAR_reshape_MEM_5_1_1_1_1_]] : vector<32xf32>
// CHECK:               vector.store [[LOAD_VAR_reshape_MEM_6_1_1_1_1_]], [[VAR_reshape_75_]]{{.}}[[VAR_7_5_]]#0, [[VAR_7_5_]]#1, [[VAR_10_11_]]{{.}} : memref<2x64x93xf32>, vector<32xf32>
// CHECK:             }
// CHECK:             [[LOOP_19_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_19_]]) with ([[LOOP_19_]] -> [[I_27_:%.+]] = 64 to 93){
// CHECK:               [[VAR_10_12_:%.+]] = krnl.get_induction_var_value([[LOOP_19_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_4_1_1_1_1_1_1_:%.+]] = krnl.load [[VAR_reshape_71_]]{{.}}[[VAR_7_5_]]#0, [[VAR_7_5_]]#1, [[VAR_10_12_]]{{.}} : memref<2x64x93xf32>
// CHECK-DAG:           [[LOAD_VAR_reshape_MEM_5_1_1_1_1_1_:%.+]] = krnl.load [[VAR_reshape_73_]]{{.}}[[VAR_10_12_]]{{.}} : memref<93xf32>
// CHECK:               [[LOAD_VAR_reshape_MEM_6_1_1_1_1_:%.+]] = arith.addf [[LOAD_VAR_reshape_MEM_4_1_1_1_1_1_1_]], [[LOAD_VAR_reshape_MEM_5_1_1_1_1_1_]] : f32
// CHECK:               krnl.store [[LOAD_VAR_reshape_MEM_6_1_1_1_1_]], [[VAR_reshape_75_]]{{.}}[[VAR_7_5_]]#0, [[VAR_7_5_]]#1, [[VAR_10_12_]]{{.}} : memref<2x64x93xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_41_]] : memref<2x64x31x3xf32>
// CHECK:         }
}

// -----

// arg1 is defined for every outer loop, arg2 is defined for 64 of the 128 outer loops.
func.func @layernorm_4D_with_scale_bias_with_high_dims(%arg0: tensor<2x64x32x8xf32>, %arg1: tensor<2x64x32x8xf32>, %arg2: tensor<64x32x8xf32>) -> tensor<*xf32> {
  %0 = "onnx.NoValue"() {value} : () -> none
  %Y, %Mean, %InvStdDev = "onnx.LayerNormalization"(%arg0, %arg1, %arg2) {axis = -2 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<2x64x32x8xf32>, tensor<2x64x32x8xf32>, tensor<64x32x8xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
  func.return %Y : tensor<*xf32>

// CHECK-LABEL:  func.func @layernorm_4D_with_scale_bias_with_high_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x64x32x8xf32>, [[PARAM_1_:%.+]]: memref<2x64x32x8xf32>, [[PARAM_2_:%.+]]: memref<64x32x8xf32>) -> memref<2x64x32x8xf32> {
// CHECK-DAG:       [[VAR_reshape_4_:%.+]] = memref.reshape [[PARAM_1_]]([[RES_1_:%.+]]) : (memref<2x64x32x8xf32>, memref<2xindex>) -> memref<128x256xf32>
// CHECK-DAG:       [[VAR_reshape_6_:%.+]] = memref.reshape [[PARAM_2_]]([[RES_2_:%.+]]) : (memref<64x32x8xf32>, memref<2xindex>) -> memref<64x256xf32>
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_:%.+]] 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_:%.+]] -> [[I_0_:%.+]] = 0 to 128){
// CHECK:             affine.for [[I_2_:%.+]] = 0 to 256 step 16 {
// CHECK:             affine.for [[I_2_:%.+]] = 0 to 256 step 16 {
}
