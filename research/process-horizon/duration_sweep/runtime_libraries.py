"""Verify the native libraries loaded into the current macOS process."""

import ctypes
import sys
from pathlib import Path


class RuntimeLibraryError(RuntimeError):
    """The requested solver libraries do not match the loaded native images."""


def validate_loaded_libraries(expected, loaded):
    """Require exactly one loaded image for each requested library name."""
    expected_paths = [Path(path).resolve() for path in expected]
    loaded_paths = [Path(path).resolve() for path in loaded]
    if len({path.name for path in expected_paths}) != len(expected_paths):
        raise RuntimeLibraryError("requested solver library names are duplicated")
    for path in expected_paths:
        actual = [item for item in loaded_paths if item.name == path.name]
        if actual != [path]:
            raise RuntimeLibraryError(f"expected {path}; loaded {actual}")
    return expected_paths


def require_loaded_libraries(expected):
    """Inspect dyld images and reject a runtime that differs from explicit inputs."""
    if sys.platform != "darwin":
        raise RuntimeLibraryError("this research runtime requires macOS dyld inspection")
    dyld = ctypes.CDLL(None)
    dyld._dyld_image_count.restype = ctypes.c_uint32
    dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    dyld._dyld_get_image_name.restype = ctypes.c_char_p
    loaded = [
        Path(value.decode())
        for index in range(dyld._dyld_image_count())
        if (value := dyld._dyld_get_image_name(index))
    ]
    return validate_loaded_libraries(expected, loaded)
