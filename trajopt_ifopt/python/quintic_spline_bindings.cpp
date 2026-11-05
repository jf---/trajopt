#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <trajopt_ifopt/spline/quintic_spline_jerk_cost.h>
#include <trajopt_ifopt/spline/quintic_spline_model.h>
#include <trajopt_ifopt/spline/quintic_spline_position_limits.h>
#include <trajopt_ifopt/variable_sets/var.h>

namespace nb = nanobind;
using namespace nb::literals;
namespace ti = trajopt_ifopt;

namespace
{
std::vector<ti::Seconds> seconds(const std::vector<double>& values)
{
  std::vector<ti::Seconds> result;
  result.reserve(values.size());
  for (double value : values)
    result.emplace_back(value);
  return result;
}
}  // namespace

NB_MODULE(_quintic_spline, m)
{
  nb::module_::import_("tesseract_robotics.trajopt_ifopt");
  nb::exception<ti::QuinticSplineError>(m, "QuinticSplineError");
  nb::class_<ti::SplineAffineMap>(m, "SplineAffineMap")
      .def_ro("basis", &ti::SplineAffineMap::basis)
      .def_ro("offset", &ti::SplineAffineMap::offset);
  nb::class_<ti::JerkQuadrature>(m, "JerkQuadrature")
      .def_ro("map", &ti::JerkQuadrature::map)
      .def_ro("weights", &ti::JerkQuadrature::weights);
  nb::class_<ti::QuinticSplineModel>(m, "QuinticSplineModel")
      .def_static(
          "build",
          [](const std::vector<double>& times,
             const Eigen::VectorXd& v0,
             const Eigen::VectorXd& a0,
             const Eigen::VectorXd& v1,
             const Eigen::VectorXd& a1) { return ti::QuinticSplineModel::build(seconds(times), v0, a0, v1, a1); },
          "times_s"_a,
          "velocity_start"_a,
          "acceleration_start"_a,
          "velocity_end"_a,
          "acceleration_end"_a,
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "map",
          [](const ti::QuinticSplineModel& self, const std::vector<double>& times, unsigned order) {
            return self.map(seconds(times), order);
          },
          "times_s"_a,
          "order"_a,
          nb::call_guard<nb::gil_scoped_release>())
      .def("jerk_quadrature", &ti::QuinticSplineModel::jerkQuadrature, nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("knot_count", &ti::QuinticSplineModel::knotCount)
      .def_prop_ro("axis_count", &ti::QuinticSplineModel::axisCount);
  nb::class_<ti::QuinticSplineJerkCost, ti::ConstraintSet>(m, "QuinticSplineJerkCost")
      .def(nb::init<const ti::QuinticSplineModel&,
                    std::vector<std::shared_ptr<const ti::Var>>,
                    Eigen::VectorXd,
                    std::string>(),
           "model"_a,
           "knots"_a,
           "axis_weights"_a,
           "name"_a = "ContinuousSplineJerk");
  nb::class_<ti::QuinticSplinePositionLimits, ti::ConstraintSet>(m, "QuinticSplinePositionLimits")
      .def(
          "__init__",
          [](ti::QuinticSplinePositionLimits* self,
             const ti::QuinticSplineModel& model,
             std::vector<std::shared_ptr<const ti::Var>> knots,
             const std::vector<double>& times,
             std::vector<ti::Bounds> limits,
             std::string name) {
            new (self) ti::QuinticSplinePositionLimits(
                model, std::move(knots), seconds(times), std::move(limits), std::move(name));
          },
          "model"_a,
          "knots"_a,
          "times_s"_a,
          "limits"_a,
          "name"_a = "SplineInterpolatedJointLimits");
}
