# Experimental native quintic spline binding

This standalone research module binds `QuinticSplineModel`, its native Ifopt jerk cost and sampled position constraints. It does not replace the existing Python implementation or integrate the feature into the app. The model uses a fixed clock in seconds and explicit boundary velocities and accelerations; these are modeled inputs, not measured controller states. An independent collision check remains required.

The binding releases the GIL for spline-map evaluation. `QuinticSplineModel.evaluate(times_s, positions, max_order)` accepts a NumPy time vector and evaluates position and all derivatives through `max_order` in one native batch, without per-sample Python objects. The affine `map` API remains available for optimizer Jacobians. The native PIQP/SQP binding used in these experiments releases it for `solve`; to keep this solve path native, construct the jerk cost and sampled position constraints from this module and use the native uncapped dynamic collision constraint. A Python callback registered with SQP would still reacquire the GIL. This module requires the locally built PR 363 PIQP binding and corrected native TrajOpt collision/SQP libraries; it is not a packaged release.

The following reproduces the current macOS development build from the root of this isolated TrajOpt worktree. `INTEGRATION` is the existing app integration worktree, `RESEARCH` is the earlier research worktree used only for its installed dependency environment, and `PACKAGED` is the pre-existing nanobind dependency environment. These paths are local experiment inputs, not packaged dependencies.

```bash
INTEGRATION=/Users/jelle/Code/CADCAM/tesseract_qt_robodk/.worktrees/r62-smooth-windows-main
RESEARCH=/Users/jelle/Code/CADCAM/tesseract_qt_robodk/.worktrees/r62-smooth-windows
PACKAGED=/Users/jelle/Code/CADCAM/tesseract_nanobind-upstream-main/.pixi/envs/packaged
UPSTREAM=/Users/jelle/Code/CADCAM/tesseract_nanobind-upstream-main/.pixi/envs/upstream
MANIFEST="$RESEARCH/.superpowers/r62-env/pyproject.toml"
PREFIX="$RESEARCH/.superpowers/r62-master/install;$PACKAGED;$UPSTREAM"

pixi run --as-is --manifest-path "$MANIFEST" cmake -S trajopt_ifopt -B build-ifopt \
  -DCMAKE_BUILD_TYPE=Release -DTRAJOPT_ENABLE_TESTING=ON \
  -DCMAKE_PREFIX_PATH="$PREFIX" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L$PACKAGED/lib" -DCMAKE_EXE_LINKER_FLAGS="-L$PACKAGED/lib"
pixi run --as-is --manifest-path "$MANIFEST" cmake --build build-ifopt \
  --target trajopt_ifopt_quintic_spline_model_unit \
  trajopt_ifopt_quintic_spline_jerk_cost_unit \
  trajopt_ifopt_quintic_spline_position_limits_unit \
  trajopt_ifopt_dynamic_spline_collision_constraint_unit -j 4
pixi run --as-is --manifest-path "$MANIFEST" env TESSERACT_RESOURCE_PATH="$PWD" \
  ctest --test-dir build-ifopt -R 'QuinticSpline|DynamicSplineCollision' --output-on-failure
pixi run --as-is --manifest-path "$MANIFEST" cmake --install build-ifopt --prefix build-ifopt/install

PY_PREFIX="$PWD/build-ifopt/install;$PACKAGED/lib/python3.12/site-packages/nanobind;$PREFIX"
pixi run --as-is --manifest-path "$MANIFEST" cmake -S trajopt_ifopt/python -B build-python \
  -DCMAKE_PREFIX_PATH="$PY_PREFIX" -DPython_EXECUTABLE="$PACKAGED/bin/python" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_MODULE_LINKER_FLAGS="-L$PACKAGED/lib"
pixi run --as-is --manifest-path "$MANIFEST" cmake --build build-python -j 4
PYTEST_XDIST_AUTO_NUM_WORKERS=2 pixi run --as-is --manifest-path "$MANIFEST" env \
  PYTHONPATH="$PWD/build-python:$INTEGRATION/.superpowers/piqp-comparison/overlay:." \
  DYLD_LIBRARY_PATH="$PWD/build-ifopt:$INTEGRATION/.superpowers/piqp-comparison/build-sqp:$RESEARCH/.superpowers/r62-master/install/lib" \
  pytest -q -n auto --testmon --testmon-noselect --noconftest trajopt_ifopt/python/test_quintic_spline.py
```

The C++ tests cover SciPy reference values, nonuniform knots, nonzero boundary derivatives, analytic polynomial reproduction, jerk quadrature, Jacobians, invalid inputs and the existing dynamic collision rows. The Python tests cover SciPy parity, a native Ifopt/PIQP solve and thread progress during a GIL-released native evaluation. These tests do not establish full-tour planning, physical timing, continuous collision clearance or an end-to-end speedup.
