# Copyright 2023 The Navix Authors.

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Hardware-detection functions `AlgorithmEntry` auto-populates its
`cpu_type`/`ram_bytes`/`gpu_type`/`cuda_version`/`cudnn_version` fields
from. See `navix.benchmarks` (this package's `__init__.py`) for the
full design."""
from __future__ import annotations

import importlib.metadata
import os
import platform
import re
import subprocess
from typing import Optional

import jax


def cpu_type() -> str:
    """Reads the CPU's model name.

    Returns:
        str: `/proc/cpuinfo`'s `model name` on Linux,
        `sysctl -n machdep.cpu.brand_string` on macOS, or
        `platform.processor()`/`platform.machine()` as a fallback on
        any other platform, or if the platform-specific lookup fails.
    """
    system = platform.system()
    if system == "Linux":
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif system == "Darwin":
        try:
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    return platform.processor() or platform.machine()


def ram_bytes() -> int:
    """Reads total system RAM.

    Returns:
        int: Total system RAM, in bytes. POSIX-only (Linux/macOS).
    """
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def gpu_type() -> Optional[str]:
    """Reads the specific GPU model JAX is running on.

    Returns:
        Optional[str]: The GPU model (e.g. distinguishes an SXM from a
        PCIe variant of the same chip), or `None` if JAX isn't running
        on a GPU.
    """
    device = jax.devices()[0]
    return device.device_kind if device.platform == "gpu" else None


def cuda_version() -> Optional[str]:
    """Reads the CUDA version jaxlib is running on.

    Returns:
        Optional[str]: The installed `nvidia-cuda-runtime-cuXX`
        package version (the same one a `pip install jaxlib[cudaXX]`
        pins), falling back to `nvidia-smi`'s reported CUDA version if
        that package isn't found. `None` if JAX isn't running on a
        GPU.
    """
    if jax.devices()[0].platform != "gpu":
        return None
    for cuda_major in ("12", "11"):
        try:
            return importlib.metadata.version(f"nvidia-cuda-runtime-cu{cuda_major}")
        except importlib.metadata.PackageNotFoundError:
            continue
    try:
        output = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True).stdout
        match = re.search(r"CUDA Version:\s*([\d.]+)", output)
        return match.group(1) if match else None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def cudnn_version() -> Optional[str]:
    """Reads the cuDNN version jaxlib is running on.

    Returns:
        Optional[str]: The installed `nvidia-cudnn-cuXX` package
        version (the same one a `pip install jaxlib[cudaXX]` pins).
        `None` if JAX isn't running on a GPU.
    """
    if jax.devices()[0].platform != "gpu":
        return None
    for cuda_major in ("12", "11"):
        try:
            return importlib.metadata.version(f"nvidia-cudnn-cu{cuda_major}")
        except importlib.metadata.PackageNotFoundError:
            continue
    return None
