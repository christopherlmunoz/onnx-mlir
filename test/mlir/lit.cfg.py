import os
import sys
import re
import platform
import subprocess

import lit.util
import lit.formats
from lit.llvm import llvm_config
from lit.llvm.subst import FindTool
from lit.llvm.subst import ToolSubst

# name: The name of this test suite.
config.name = "Open Neural Network Frontend"

# Prefer the lit internal shell, matching upstream MLIR's lit.cfg.py: it gives
# better failure output, and LLVM-23 deprecated execute_external=True (raises
# unless force_execute_external is set). LIT_USE_INTERNAL_SHELL=0 opts out.
use_lit_shell = True
lit_shell_env = os.environ.get("LIT_USE_INTERNAL_SHELL")
if lit_shell_env:
    use_lit_shell = lit.util.pythonize_bool(lit_shell_env)

config.test_format = lit.formats.ShTest(execute_external=not use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir", ".json", ".onnxtext"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.onnx_mlir_obj_root, "test", "mlir")

llvm_config.use_default_substitutions()

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.onnx_mlir_tools_dir, config.mlir_tools_dir, config.llvm_tools_dir]

tools = [
    "onnx-mlir",
    "onnx-mlir-opt",
    "mlir-opt",
    "mlir-translate",
    "binary-decoder",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

# %onnx-mlir-home expands to the parent of the tools dir (i.e. the Debug/ or
# Release/ prefix under build/).  build-run-onnx-lib.sh expects ONNX_MLIR_HOME
# to point there so it can locate bin/, lib/, and the source tree.
config.substitutions.append(
    ("%onnx-mlir-home", os.path.dirname(os.path.normpath(config.onnx_mlir_tools_dir)))
)

# This is based on the same code in llvm and it is meant to determine what
# the supported targets for llvm & friends are - this allow us to filter test
# execution based on the available targets
for arch in config.targets_to_build.split():
    config.available_features.add(arch.lower())
