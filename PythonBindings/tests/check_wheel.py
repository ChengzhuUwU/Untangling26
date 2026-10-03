"""Check installed native runtime dependencies without requiring a GPU.

cibuildwheel runs this inside its isolated test environment. Real repair tests
must additionally run on a machine with a supported compute device.
"""

import ctypes
from pathlib import Path
import sys

import lcs_py


def main():
    runtime = Path(lcs_py.__file__).resolve().parent
    expected = "metal" if sys.platform == "darwin" else "vk"
    # LuisaCompute uses MODULE libraries (.so) for Metal too; only its shared
    # runtime libraries use the macOS .dylib suffix.
    suffix = ".dll" if sys.platform == "win32" else ".so"
    modules = list(runtime.glob(f"*backend-{expected}{suffix}"))
    if len(modules) != 1:
        raise RuntimeError(f"Expected one {expected} backend beside lcs_py, found {modules}")
    # Loading the module catches missing indirect dependencies such as glslang.
    # It does not create a device or require a GPU on hosted CI runners.
    ctypes.CDLL(str(modules[0]))
    print(f"Native solver and {expected} backend dependencies load successfully")


if __name__ == "__main__":
    main()
