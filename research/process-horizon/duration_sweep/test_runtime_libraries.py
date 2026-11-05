"""The sweep must use the requested native libraries, never a similarly named copy."""

import pytest
from runtime_libraries import RuntimeLibraryError, validate_loaded_libraries


def test_accepts_only_requested_library_paths(tmp_path):
    ifopt = tmp_path / "ifopt" / "libtrajopt_ifopt.dylib"
    sqp = tmp_path / "sqp" / "libtrajopt_sqp.dylib"
    assert validate_loaded_libraries([ifopt, sqp], [ifopt, sqp]) == [ifopt, sqp]
    with pytest.raises(RuntimeLibraryError):
        validate_loaded_libraries([ifopt, sqp], [ifopt, tmp_path / "old" / sqp.name])
    with pytest.raises(RuntimeLibraryError):
        validate_loaded_libraries([ifopt, sqp], [ifopt, sqp, tmp_path / "other" / sqp.name])
    with pytest.raises(RuntimeLibraryError):
        validate_loaded_libraries([ifopt, sqp], [ifopt])
