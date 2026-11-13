"""GPU/CUDA environment setup — must run before `import torch` anywhere.

Discovered during implementation: this host's shell environment sets
`LD_LIBRARY_PATH` to include `/usr/local/cuda-11.0/lib64` (a shared,
system-wide path this harness must not modify — see §0A "never modify
shared CUDA/driver state"). That path's `libcublas.so.11` is from the
CUDA 11.0 toolkit and is missing symbols (e.g. `cublasSetWorkspace_v2`)
that the isolated environment's torch==2.1.2+cu118 build requires. Because
it appears earlier in `LD_LIBRARY_PATH` than torch's own bundled
`torch/lib/libcublas.so.11`, the dynamic linker resolves the wrong one and
`import torch` fails with an `undefined symbol` ImportError.

Fix, scoped entirely to this benchmark's own process (never touches the
user's shell profile or any shared/global environment file): prepend
torch's own lib directory to `LD_LIBRARY_PATH` for this process. Since
`LD_LIBRARY_PATH` must be correct at process exec time (mutating
`os.environ` after the interpreter has already started is not reliably
honored by the dynamic linker for subsequent dlopen calls on every
platform), this re-execs the current process once with the corrected
environment if it is not already fixed.

Call `ensure_cuda_lib_path()` as the first statement of any entrypoint that
will (transitively) `import torch` — before that import.
"""

from __future__ import annotations

import importlib.util
import os
import sys

_MARKER = "_INDICUAD_LD_LIBRARY_PATH_FIXED"


def _torch_lib_dir() -> str | None:
    """Locate torch's bundled lib/ directory without importing torch (which
    would fail before the fix is applied)."""
    try:
        spec = importlib.util.find_spec("torch")
    except Exception:  # noqa: BLE001
        return None
    if spec is None or spec.origin is None:
        return None
    lib_dir = os.path.join(os.path.dirname(spec.origin), "lib")
    return lib_dir if os.path.isdir(lib_dir) else None


def ensure_cuda_lib_path(cuda_visible_devices: str | None = None) -> None:
    """Fix LD_LIBRARY_PATH (see module docstring) and, optionally, pin
    CUDA_VISIBLE_DEVICES — both must be correct *before* `import torch`
    triggers any CUDA driver initialization, so both are applied in the same
    re-exec when either needs changing. Safe to call unconditionally at the
    top of any script; a no-op once already fixed."""
    already_fixed = os.environ.get(_MARKER) == "1"
    cvd_needs_set = cuda_visible_devices is not None and os.environ.get("CUDA_VISIBLE_DEVICES") != cuda_visible_devices

    if already_fixed and not cvd_needs_set:
        return

    lib_dir = _torch_lib_dir()
    current = os.environ.get("LD_LIBRARY_PATH", "")
    entries = current.split(":") if current else []
    lib_path_needs_set = lib_dir is not None and entries[:1] != [lib_dir]

    if not lib_path_needs_set and not cvd_needs_set:
        os.environ[_MARKER] = "1"
        return

    env = dict(os.environ)
    if lib_dir is not None:
        env["LD_LIBRARY_PATH"] = lib_dir + (":" + current if current else "")
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    env[_MARKER] = "1"
    os.execve(sys.executable, [sys.executable] + sys.argv, env)
