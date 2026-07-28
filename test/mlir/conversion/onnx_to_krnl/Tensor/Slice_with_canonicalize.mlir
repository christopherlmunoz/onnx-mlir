// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

// Slice where all the parameters are constant.
func.func @test_slice_constant_default_axes(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = "onnx.NoValue"() {value} : () -> none
  %starts = onnx.Constant dense<[1, 0]> : tensor<2xi64>
  %ends = onnx.Constant dense<[2, 3]> : tensor<2xi64>
  %steps = onnx.Constant dense<[1, 2]> : tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, none, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-LABEL:  func @test_slice_constant_default_axes
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4xf32>) -> memref<1x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply [[MAP_0_]]([[IV]]#0)
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply [[MAP_1_]]([[IV]]#1)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[VAR_7_]]{{.}} : memref<2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]][[[IV]]#0, [[IV]]#1] : memref<1x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x2xf32>
// CHECK:         }
}

// -----

func.func @test_slice_constant_default_steps(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = onnx.Constant dense<[0, 1]> : tensor<2xi64>
  %starts = onnx.Constant dense<[1, 0]> : tensor<2xi64>
  %ends = onnx.Constant dense<[2, 3]> : tensor<2xi64>
  %steps = "onnx.NoValue"() {value} : () -> none
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, none) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-LABEL:  func @test_slice_constant_default_steps
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4xf32>) -> memref<1x3xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x3xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[VAR_6_:%.+]] = affine.apply [[MAP_0_]]([[IV]]#0)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[IV]]#1] : memref<2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]][[[IV]]#0, [[IV]]#1] : memref<1x3xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x3xf32>
// CHECK:         }
}

// -----

func.func @test_slice_all_constant(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = onnx.Constant dense<[0, 1]> : tensor<2xi64>
  %starts = onnx.Constant dense<[1, 0]> : tensor<2xi64>
  %ends = onnx.Constant dense<[2, 3]> : tensor<2xi64>
  %steps = onnx.Constant dense<[1, 2]> : tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-LABEL:  func @test_slice_all_constant
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4xf32>) -> memref<1x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply [[MAP_0_]]([[IV]]#0)
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply [[MAP_1_]]([[IV]]#1)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[VAR_7_]]{{.}} : memref<2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]][[[IV]]#0, [[IV]]#1] : memref<1x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x2xf32>
// CHECK:         }
}

// -----

func.func @test_slice_all_constant_negative(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = onnx.Constant dense<[0, -1]> : tensor<2xi64>
  %starts = onnx.Constant dense<[1, 0]> : tensor<2xi64>
  %ends = onnx.Constant dense<[2, -1]> : tensor<2xi64>
  %steps = onnx.Constant dense<[1, 2]> : tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-LABEL:  func @test_slice_all_constant_negative
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4xf32>) -> memref<1x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply [[MAP_0_]]([[IV]]#0)
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply [[MAP_1_]]([[IV]]#1)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[VAR_7_]]{{.}} : memref<2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]][[[IV]]#0, [[IV]]#1] : memref<1x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x2xf32>
// CHECK:         }
}

// -----

func.func @test_slice_all_constant_end_outofbound(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = onnx.Constant dense<[0, 1]> : tensor<2xi64>
  %starts = onnx.Constant dense<[1, 0]> : tensor<2xi64>
  %ends = onnx.Constant dense<[5, 3]> : tensor<2xi64>
  %steps = onnx.Constant dense<[1, 2]> : tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-LABEL:  func @test_slice_all_constant_end_outofbound
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4xf32>) -> memref<1x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply [[MAP_0_]]([[IV]]#0)
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply [[MAP_1_]]([[IV]]#1)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[VAR_7_]]{{.}} : memref<2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]][[[IV]]#0, [[IV]]#1] : memref<1x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x2xf32>
// CHECK:         }
}

// -----

func.func @test_slice_all_constant_negative_steps(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = onnx.Constant dense<[0, 1]> : tensor<2xi64>
  %starts = onnx.Constant dense<[1, 3]> : tensor<2xi64>
  %ends = onnx.Constant dense<[2, 0]> : tensor<2xi64>
  %steps = onnx.Constant dense<[1, -2]> : tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 * -2 + 3)>
// CHECK-LABEL:  func @test_slice_all_constant_negative_steps
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4xf32>) -> memref<1x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply [[MAP_0_]]([[IV]]#0)
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply [[MAP_1_]]([[IV]]#1)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[VAR_7_]]{{.}} : memref<2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]][[[IV]]#0, [[IV]]#1] : memref<1x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x2xf32>
// CHECK:         }
}

// -----

// Slice where the data is dyn sized along a non-sliced dim
func.func @dyntest_slice_constant_dynshape_not_spliced(%arg0 : tensor<?x4x5xf32>) -> tensor<*xf32> {
  // %data = onnx.Constant dense<[ [ [ 0, 1, 2, 3, 4 ], [ 10, 11, 12, 13, 14 ], [ 20, 21, 22, 23, 24 ], [ 30, 31, 32, 33, 34 ] ], [ [ 100, 101, 102, 103, 104 ], [ 110, 111, 112, 113, 114 ], [ 120, 121, 122, 123, 124 ], [ 130, 131, 132, 133, 134 ] ], [ [ 200, 201, 202, 203, 204 ], [ 210, 211, 212, 213, 214 ], [ 220, 221, 222, 223, 224 ], [ 230, 231, 232, 233, 234 ] ] ] > : tensor<3x4x5xi64>
  // slice * 1-3 1-4 with neg numbers
  %axes = onnx.Constant dense<[2, 1]> : tensor<2xi64>
  %starts = onnx.Constant dense<[1, 1]> : tensor<2xi64>
  %ends = onnx.Constant dense<[-1, -1]> : tensor<2xi64>
  %steps = onnx.Constant dense<[1, 1]> : tensor<2xi64>
  %res = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<?x4x5xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%res) : (tensor<*xf32>) -> ()

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-LABEL:  func @dyntest_slice_constant_dynshape_not_spliced
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x4x5xf32>) -> memref<?x2x3xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[DIM_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x4x5xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[DIM_0_]]) {{.*}} : memref<?x2x3xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[DIM_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 3){
// CHECK:             [[IV:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply [[MAP_1_]]([[IV]]#1)
// CHECK-DAG:         [[VAR_8_:%.+]] = affine.apply [[MAP_1_]]([[IV]]#2)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]][[[IV]]#0, [[VAR_7_]], [[VAR_8_]]{{.}} : memref<?x4x5xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]][[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<?x2x3xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x2x3xf32>
// CHECK:         }
}

// -----

// Check where all is dynamic except input size and axis. The code was verified
// using a procedure simioar to mlir-run and by manually adding code to print the
// output as a vector

func.func @compute_slice_all_dyn(%arg0 : tensor<2xi64>, %arg1 : tensor<2xi64>, %arg2 : tensor<2xi64>) -> tensor<3x?x?xi64> {
   %data = onnx.Constant dense<[ [ [ 0, 1, 2, 3, 4 ], [ 10, 11, 12, 13, 14 ], [ 20, 21, 22, 23, 24 ], [ 30, 31, 32, 33, 34 ] ], [ [ 100, 101, 102, 103, 104 ], [ 110, 111, 112, 113, 114 ], [ 120, 121, 122, 123, 124 ], [ 130, 131, 132, 133, 134 ] ], [ [ 200, 201, 202, 203, 204 ], [ 210, 211, 212, 213, 214 ], [ 220, 221, 222, 223, 224 ], [ 230, 231, 232, 233, 234 ] ] ] > : tensor<3x4x5xi64>

  // slice * 1-3 1-4 with neg numbers
  %axes = onnx.Constant dense<[2, 1]> : tensor<2xi64>
  %res = "onnx.Slice"(%data, %arg0, %arg1, %axes, %arg2) : (tensor<3x4x5xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<3x?x?xi64>
  return %res : tensor<3x?x?xi64>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 + 5)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 + 4)>
// CHECK-LABEL:  func.func @compute_slice_all_dyn
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2xi64>, [[PARAM_1_:%.+]]: memref<2xi64>, [[PARAM_2_:%.+]]: memref<2xi64>) -> memref<3x?x?xi64> {
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_2147483647_:%.+]] = arith.constant 2147483647 : index
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[CST_minus_2147483648_:%.+]] = arith.constant -2147483648 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() <{name = "constant_{{[0-9]+}}", shape = [3, 4, 5], value = dense<{{.}}{{.}}[0, 1, 2, 3, 4], [10, 11, 12, 13, 14], [20, 21, 22, 23, 24], [30, 31, 32, 33, 34]{{.}}, {{.}}[100, 101, 102, 103, 104], [110, 111, 112, 113, 114], [120, 121, 122, 123, 124], [130, 131, 132, 133, 134]{{.}}, {{.}}[200, 201, 202, 203, 204], [210, 211, 212, 213, 214], [220, 221, 222, 223, 224], [230, 231, 232, 233, 234]{{.}}{{.}}> : tensor<3x4x5xi64>}> : () -> memref<3x4x5xi64>
// CHECK:           [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[CST_0_]]{{.}} : memref<2xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.index_cast [[LOAD_PARAM_0_MEM_]] : i64 to index
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_0_]]{{.}} : memref<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:       [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[CST_0_]]{{.}} : memref<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.index_cast [[LOAD_PARAM_2_MEM_]] : i64 to index
// CHECK-DAG:       [[VAR_7_:%.+]] = arith.cmpi slt, [[VAR_2_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_8_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_2_]]{{.}}
// CHECK:           [[VAR_9_:%.+]] = arith.select [[VAR_7_]], [[VAR_8_]], [[VAR_2_]] : index
// CHECK:           [[VAR_10_:%.+]] = arith.maxsi [[VAR_9_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_11_:%.+]] = arith.minsi [[VAR_10_]], [[CST_4_]] : index
// CHECK-DAG:       [[VAR_12_:%.+]] = arith.maxsi [[VAR_9_]], [[CST_0_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_13_:%.+]] = arith.minsi [[VAR_12_]], [[CST_5_]] : index
// CHECK-DAG:       [[VAR_14_:%.+]] = arith.cmpi slt, [[VAR_6_]], [[CST_0_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_15_:%.+]] = arith.select [[VAR_14_]], [[VAR_11_]], [[VAR_13_]] : index
// CHECK-DAG:       [[VAR_16_:%.+]] = arith.cmpi slt, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_17_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_4_]]{{.}}
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_18_:%.+]] = arith.select [[VAR_16_]], [[VAR_17_]], [[VAR_4_]] : index
// CHECK-DAG:       [[VAR_19_:%.+]] = arith.cmpi sle, [[VAR_4_]], [[CST_minus_2147483648_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_20_:%.+]] = arith.select [[VAR_19_]], [[CST_minus_1_]], [[VAR_18_]] : index
// CHECK-DAG:       [[VAR_21_:%.+]] = arith.cmpi sge, [[VAR_4_]], [[CST_2147483647_]] : index
// CHECK:           [[VAR_22_:%.+]] = arith.select [[VAR_21_]], [[CST_5_]], [[VAR_20_]] : index
// CHECK:           [[VAR_23_:%.+]] = arith.maxsi [[VAR_22_]], [[CST_minus_1_]] : index
// CHECK-DAG:       [[VAR_24_:%.+]] = arith.minsi [[VAR_23_]], [[CST_5_]] : index
// CHECK-DAG:       [[VAR_25_:%.+]] = arith.maxsi [[VAR_22_]], [[CST_0_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_26_:%.+]] = arith.minsi [[VAR_25_]], [[CST_5_]] : index
// CHECK-DAG:       [[VAR_27_:%.+]] = arith.cmpi slt, [[VAR_6_]], [[CST_0_]] : index
// CHECK:           [[VAR_28_:%.+]] = arith.select [[VAR_27_]], [[VAR_24_]], [[VAR_26_]] : index
// CHECK:           [[VAR_29_:%.+]] = arith.subi [[VAR_28_]], [[VAR_15_]] : index
// CHECK:           [[VAR_30_:%.+]] = arith.ceildivsi [[VAR_29_]], [[VAR_6_]] : index
// CHECK-DAG:       [[VAR_31_:%.+]] = arith.maxsi [[VAR_30_]], [[CST_0_]] : index
// CHECK-DAG:       [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[CST_1_]]{{.}} : memref<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_33_:%.+]] = arith.index_cast [[LOAD_PARAM_0_MEM_1_]] : i64 to index
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_1_]]{{.}} : memref<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_35_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_1_]] : i64 to index
// CHECK-DAG:       [[LOAD_PARAM_2_MEM_1_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[CST_1_]]{{.}} : memref<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_37_:%.+]] = arith.index_cast [[LOAD_PARAM_2_MEM_1_]] : i64 to index
// CHECK-DAG:       [[VAR_38_:%.+]] = arith.cmpi slt, [[VAR_33_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_39_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_33_]]{{.}}
// CHECK:           [[VAR_40_:%.+]] = arith.select [[VAR_38_]], [[VAR_39_]], [[VAR_33_]] : index
// CHECK:           [[VAR_41_:%.+]] = arith.maxsi [[VAR_40_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_42_:%.+]] = arith.minsi [[VAR_41_]], [[CST_3_]] : index
// CHECK-DAG:       [[VAR_43_:%.+]] = arith.maxsi [[VAR_40_]], [[CST_0_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_44_:%.+]] = arith.minsi [[VAR_43_]], [[CST_4_]] : index
// CHECK-DAG:       [[VAR_45_:%.+]] = arith.cmpi slt, [[VAR_37_]], [[CST_0_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_46_:%.+]] = arith.select [[VAR_45_]], [[VAR_42_]], [[VAR_44_]] : index
// CHECK-DAG:       [[VAR_47_:%.+]] = arith.cmpi slt, [[VAR_35_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_48_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_35_]]{{.}}
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_49_:%.+]] = arith.select [[VAR_47_]], [[VAR_48_]], [[VAR_35_]] : index
// CHECK-DAG:       [[VAR_50_:%.+]] = arith.cmpi sle, [[VAR_35_]], [[CST_minus_2147483648_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_51_:%.+]] = arith.select [[VAR_50_]], [[CST_minus_1_]], [[VAR_49_]] : index
// CHECK-DAG:       [[VAR_52_:%.+]] = arith.cmpi sge, [[VAR_35_]], [[CST_2147483647_]] : index
// CHECK:           [[VAR_53_:%.+]] = arith.select [[VAR_52_]], [[CST_4_]], [[VAR_51_]] : index
// CHECK:           [[VAR_54_:%.+]] = arith.maxsi [[VAR_53_]], [[CST_minus_1_]] : index
// CHECK-DAG:       [[VAR_55_:%.+]] = arith.minsi [[VAR_54_]], [[CST_4_]] : index
// CHECK-DAG:       [[VAR_56_:%.+]] = arith.maxsi [[VAR_53_]], [[CST_0_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_57_:%.+]] = arith.minsi [[VAR_56_]], [[CST_4_]] : index
// CHECK-DAG:       [[VAR_58_:%.+]] = arith.cmpi slt, [[VAR_37_]], [[CST_0_]] : index
// CHECK:           [[VAR_59_:%.+]] = arith.select [[VAR_58_]], [[VAR_55_]], [[VAR_57_]] : index
// CHECK:           [[VAR_60_:%.+]] = arith.subi [[VAR_59_]], [[VAR_46_]] : index
// CHECK:           [[VAR_61_:%.+]] = arith.ceildivsi [[VAR_60_]], [[VAR_37_]] : index
// CHECK:           [[VAR_62_:%.+]] = arith.maxsi [[VAR_61_]], [[CST_0_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_62_]], [[VAR_31_]]) {{.*}}: memref<3x?x?xi64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[VAR_62_]], [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to [[VAR_31_]]){
// CHECK:             [[VAR_64_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[VAR_65_:%.+]] = arith.muli [[VAR_37_]], [[VAR_64_]]#1 : index
// CHECK-DAG:         [[VAR_66_:%.+]] = arith.addi [[VAR_65_]], [[VAR_46_]] : index
// CHECK-DAG:         [[VAR_67_:%.+]] = arith.muli [[VAR_6_]], [[VAR_64_]]#2 : index
// CHECK:             [[VAR_68_:%.+]] = arith.addi [[VAR_67_]], [[VAR_15_]] : index
// CHECK:             [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[VAR_64_]]#0, [[VAR_66_]], [[VAR_68_]]{{.}} : memref<3x4x5xi64>
// CHECK:             krnl.store [[LOAD_VAR_0_MEM_]], [[RES_]]{{.}}[[VAR_64_]]#0, [[VAR_64_]]#1, [[VAR_64_]]#2] : memref<3x?x?xi64>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x?x?xi64>
// CHECK:         }
}

