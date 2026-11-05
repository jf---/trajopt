#include "../include/trajopt_ifopt/spline/spline_fixed_join_progress_velocity.h"
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <trajopt_ifopt/spline/quintic_spline_jerk_cost.h>
#include "../include/trajopt_ifopt/spline/quintic_spline_jerk_fourth_cost.h"
#include "../include/trajopt_ifopt/spline/quintic_spline_jerk_peak_constraint.h"
#include <trajopt_ifopt/spline/quintic_spline_model.h>
#include <trajopt_ifopt/spline/spline_cartesian_line_constraint.h>
#include <trajopt_ifopt/spline/spline_cartesian_progress_constraint.h>
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
      .def(
          "evaluate",
          [](const ti::QuinticSplineModel& self,
             const Eigen::Ref<const Eigen::VectorXd>& times,
             const Eigen::MatrixXd& positions,
             unsigned max_order) {
            std::vector<ti::Seconds> clock;
            clock.reserve(static_cast<std::size_t>(times.size()));
            for (Eigen::Index i = 0; i < times.size(); ++i)
              clock.emplace_back(times[i]);
            return self.evaluate(clock, positions, max_order);
          },
          "times_s"_a,
          "positions"_a,
          "max_order"_a,
          nb::call_guard<nb::gil_scoped_release>())
      .def("jerk_quadrature", &ti::QuinticSplineModel::jerkQuadrature, nb::call_guard<nb::gil_scoped_release>())
      .def("jerk_fourth_quadrature", &ti::QuinticSplineModel::jerkFourthQuadrature, nb::call_guard<nb::gil_scoped_release>())
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
  nb::class_<ti::QuinticSplineJerkFourthCost, ti::ConstraintSet>(m, "QuinticSplineJerkFourthCost")
      .def(nb::init<const ti::QuinticSplineModel&,
                    std::vector<std::shared_ptr<const ti::Var>>,
                    Eigen::VectorXd,
                    std::string>(),
           "model"_a,
           "knots"_a,
           "axis_weights"_a,
           "name"_a = "ContinuousSplineJerkFourth");
  nb::class_<ti::QuinticSplineJerkPeakConstraint, ti::ConstraintSet>(m, "QuinticSplineJerkPeakConstraint")
      .def("__init__",
           [](ti::QuinticSplineJerkPeakConstraint* self,
              const ti::QuinticSplineModel& model,
              std::vector<std::shared_ptr<const ti::Var>> knots,
              std::shared_ptr<const ti::Var> peak,
              const std::vector<double>& witnesses,
              Eigen::VectorXd normalization,
              std::string name) {
             new (self) ti::QuinticSplineJerkPeakConstraint(model, std::move(knots), std::move(peak),
                                                            seconds(witnesses), std::move(normalization),
                                                            std::move(name));
           },
           "model"_a, "knots"_a, "peak"_a, "witnesses_s"_a, "normalization"_a, "name"_a = "SplineJerkPeak");
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
  nb::class_<ti::SplineCartesianLineConstraint, ti::ConstraintSet>(m, "SplineCartesianLineConstraint")
      .def(nb::init<ti::CartLineInfo,
                    std::vector<std::shared_ptr<const ti::Var>>,
                    Eigen::VectorXd,
                    Eigen::VectorXd,
                    Eigen::VectorXd,
                    std::string>(),
           "info"_a,
           "knots"_a,
           "basis"_a,
           "offset"_a,
           "coefficients"_a,
           "name"_a = "SplineCartesianLine");
  nb::class_<ti::SplineFixedJoinProgressVelocity, ti::ConstraintSet>(m, "SplineFixedJoinProgressVelocity")
      .def(nb::init<std::vector<std::shared_ptr<const ti::Var>>,
                    Eigen::VectorXd,
                    Eigen::VectorXd,
                    Eigen::VectorXd,
                    std::string>(),
           nb::arg("knots"),
           nb::arg("velocity_basis"),
           nb::arg("velocity_offset"),
           nb::arg("progress_gradient"),
           nb::arg("name") = "FixedJoinProgressVelocity");
  nb::class_<ti::SplineCartesianProgressConstraint, ti::ConstraintSet>(m, "SplineCartesianProgressConstraint")
      .def(nb::init<ti::CartLineInfo,
                    std::vector<std::shared_ptr<const ti::Var>>,
                    Eigen::MatrixXd,
                    Eigen::MatrixXd,
                    bool,
                    bool,
                    std::string>(),
           "info"_a,
           "knots"_a,
           "basis"_a,
           "offset"_a,
           "include_start"_a,
           "include_end"_a,
           "name"_a = "SplineCartesianProgress");
}
