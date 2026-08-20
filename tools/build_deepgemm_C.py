# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build DeepGEMM's TORCH_LIBRARY extension and copy vendored artifacts.

DeepGEMM now registers ops via TORCH_LIBRARY into ``deep_gemm._C_extension``
(abi3) and exposes the legacy API through ``deep_gemm/_C.py``. This driver
delegates to DeepGEMM's ``setup.py build_ext --inplace`` and copies the shim
plus extension into the cmake output directory.

Usage: python build_deepgemm_C.py <DEEPGEMM_SRC_DIR> <OUTPUT_DIR>
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

if len(sys.argv) != 3:
    sys.exit(f"usage: {sys.argv[0]} <SRC> <OUT>")

src = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
_pkg = src / "deep_gemm"
out.mkdir(parents=True, exist_ok=True)

is_windows = sys.platform == "win32"

if not (_pkg / "_C.py").is_file():
    sys.exit(
        f"DeepGEMM source at {src} is missing deep_gemm/_C.py; "
        "expected TORCH_LIBRARY migration layout"
    )

# Avoid DeepGEMM's clean-git assertion when vendoring a local dirty tree.
env = os.environ.copy()
env.pop("DG_SKIP_CUDA_BUILD", None)
if cuda_include_overlay := env.get("VLLM_CUDA_INCLUDE_OVERLAY"):
    env["DG_CUDA_INCLUDE_OVERLAY"] = cuda_include_overlay

cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]
if is_windows:
    # Honor CMake-triggered header rebuilds, as the old Windows driver did.
    cmd.append("--force")

print(f"[build_deepgemm_C] building in {src} with {sys.executable}", flush=True)
subprocess.check_call(cmd, cwd=src, env=env)

shim = _pkg / "_C.py"
shutil.copy2(shim, out / shim.name)

extension = _pkg / ("_C_extension.pyd" if is_windows else "_C_extension.abi3.so")
if not extension.is_file():
    sys.exit(f"DeepGEMM build did not produce {extension}")
pattern = "_C_extension*.pyd" if is_windows else "_C_extension*.so"
for stale in out.glob(pattern):
    stale.unlink()
shutil.copy2(extension, out / extension.name)
print(f"[build_deepgemm_C] installed {extension.name} -> {out}", flush=True)
