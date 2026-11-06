#include "spline_timed_progress.h"
#include "../include/trajopt_ifopt/spline/affine_spline_constraint.h"
#include "../include/trajopt_ifopt/spline/cubic_spline_stage.h"
#include "../include/trajopt_ifopt/spline/spline_fixed_join_progress_velocity.h"
#include "../include/trajopt_ifopt/spline/quintic_spline_derivative_limits.h"
#include "../include/trajopt_ifopt/spline/monotone_fixture_axis.h"
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <trajopt_ifopt/spline/quintic_spline_jerk_cost.h>
#include "../include/trajopt_ifopt/spline/quintic_spline_jerk_fourth_cost.h"
#include "../include/trajopt_ifopt/spline/quintic_spline_jerk_peak_constraint.h"
#include <trajopt_ifopt/spline/quintic_spline_model.h>
#include <trajopt_ifopt/spline/spline_cartesian_line_constraint.h>
#include <trajopt_ifopt/spline/spline_timed_cartesian_pose.h>
#include <trajopt_ifopt/spline/spline_timed_cartesian_velocity.h>
#include <trajopt_ifopt/spline/spline_cartesian_progress_constraint.h>
#include <trajopt_ifopt/spline/spline_cartesian_path_progress_constraint.h>
#include <trajopt_ifopt/spline/quintic_spline_position_limits.h>
#include <trajopt_ifopt/variable_sets/var.h>
#include <trajopt_sqp/qp_problem.h>

namespace nb = nanobind;
using namespace nb::literals;
namespace ti = trajopt_ifopt;

namespace
{
class PiecewiseQpDiagnosticError : public std::runtime_error
{
public:
  using std::runtime_error::runtime_error;
};

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
  nb::module_::import_("tesseract_robotics.trajopt_sqp");
  nb::exception<PiecewiseQpDiagnosticError>(m, "PiecewiseQpDiagnosticError");
  m.def("qp_diagnostic", [](const trajopt_sqp::QPProblem& problem, const Eigen::VectorXd& candidate) {
    const auto& matrix = problem.getConstraintMatrix();
    const auto& hessian = problem.getHessian();
    const auto& gradient = problem.getGradient();
    const auto& lower = problem.getBoundsLower();
    const auto& upper = problem.getBoundsUpper();
    if (candidate.size() != matrix.cols() || lower.size() != matrix.rows() || upper.size() != matrix.rows())
      throw PiecewiseQpDiagnosticError("QP diagnostic dimensions do not match");
    const Eigen::VectorXd projected = matrix * candidate;
    const double residual = projected.size() == 0 ? 0.0 :
        ((lower - projected).cwiseMax(projected - upper)).cwiseMax(0.0).maxCoeff();
    const auto sparse_max = [](const auto& value) {
      double result = 0.0;
      for (Eigen::Index i = 0; i < value.nonZeros(); ++i)
        result = std::max(result, std::abs(value.valuePtr()[i]));
      return result;
    };
    nb::dict result;
    result["qp_rows"] = matrix.rows();
    result["qp_columns"] = matrix.cols();
    result["primal_max_violation"] = residual;
    result["hessian_max_abs"] = sparse_max(hessian);
    result["constraint_max_abs"] = sparse_max(matrix);
    result["gradient_max_abs"] = gradient.size() == 0 ? 0.0 : gradient.cwiseAbs().maxCoeff();
    return result;
  });
  nb::exception<ti::MonotoneFixtureAxisError>(m, "MonotoneFixtureAxisError");
  nb::class_<ti::MonotoneFixtureAxis>(m, "MonotoneFixtureAxis")
      .def_static(
          "build",
          [](const std::vector<double>& times, const std::vector<double>& targets) {
            return ti::MonotoneFixtureAxis::build(seconds(times), targets);
          },
          "times_s"_a,
          "targets"_a)
      .def(
          "evaluate_many",
          [](const ti::MonotoneFixtureAxis& self, const std::vector<double>& times) {
            Eigen::MatrixXd result(static_cast<Eigen::Index>(times.size()), 4);
            {
              nb::gil_scoped_release release;
              for (std::size_t i = 0; i < times.size(); ++i)
              {
                const auto sample = self.evaluate(ti::Seconds(times[i]));
                result.row(static_cast<Eigen::Index>(i)) << sample.position, sample.velocity, sample.acceleration,
                    sample.jerk;
              }
            }
            return result;
          },
          "times_s"_a);
  nb::exception<ti::QuinticSplineError>(m, "QuinticSplineError");
  nb::exception<ti::CubicSplineStageError>(m, "CubicSplineStageError");
  nb::exception<ti::AffineSplineConstraintError>(m, "AffineSplineConstraintError");
  nb::class_<ti::SplineAffineMap>(m, "SplineAffineMap")
      .def_ro("basis", &ti::SplineAffineMap::basis)
      .def_ro("offset", &ti::SplineAffineMap::offset);
  nb::class_<ti::JerkQuadrature>(m, "JerkQuadrature")
      .def_ro("map", &ti::JerkQuadrature::map)
      .def_ro("weights", &ti::JerkQuadrature::weights);
  nb::class_<ti::CubicSplineStage>(m, "CubicSplineStage")
      .def_static(
          "build",
          [](const std::vector<double>& times, Eigen::Index axes) {
            return ti::CubicSplineStage::build(seconds(times), axes);
          },
          "times_s"_a,
          "axis_count"_a,
          nb::call_guard<nb::gil_scoped_release>())
      .def(
          "map",
          [](const ti::CubicSplineStage& self, const std::vector<double>& samples, unsigned order) {
            return self.map(seconds(samples), order);
          },
          "times_s"_a,
          "order"_a,
          nb::call_guard<nb::gil_scoped_release>());
  nb::class_<ti::AffineSplineConstraint, ti::ConstraintSet>(m, "AffineSplineConstraint")
      .def(nb::init<std::vector<std::shared_ptr<const ti::Var>>,
                    Eigen::MatrixXd,
                    Eigen::MatrixXd,
                    std::vector<ti::Bounds>,
                    Eigen::VectorXd,
                    std::string>(),
           "variables"_a,
           "basis"_a,
           "offset"_a,
           "bounds"_a,
           "coefficients"_a,
           "name"_a = "AffineSpline");
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
      .def_static(
          "build_variable_boundary",
          [](const std::vector<double>& times,
             Eigen::Index axis_count,
             std::optional<Eigen::VectorXd> v0,
             std::optional<Eigen::VectorXd> a0,
             std::optional<Eigen::VectorXd> v1,
             std::optional<Eigen::VectorXd> a1) {
            return ti::QuinticSplineModel::buildVariableBoundary(
                seconds(times), axis_count, std::move(v0), std::move(a0), std::move(v1), std::move(a1));
          },
          "times_s"_a,
          "axis_count"_a,
          "velocity_start"_a,
          "acceleration_start"_a,
          "velocity_end"_a,
          "acceleration_end"_a,
          nb::call_guard<nb::gil_scoped_release>())
      .def_static(
          "build_variable_acceleration",
          [](const std::vector<double>& times,
             const Eigen::VectorXd& v0,
             std::optional<Eigen::VectorXd> a0,
             const Eigen::VectorXd& v1,
             std::optional<Eigen::VectorXd> a1) {
            return ti::QuinticSplineModel::buildVariableAcceleration(
                seconds(times), v0, std::move(a0), v1, std::move(a1));
          },
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
      .def("jerk_fourth_quadrature",
           &ti::QuinticSplineModel::jerkFourthQuadrature,
           nb::call_guard<nb::gil_scoped_release>())
      .def_prop_ro("knot_count", &ti::QuinticSplineModel::knotCount)
      .def_prop_ro("variable_count", &ti::QuinticSplineModel::variableCount)
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
      .def(
          "__init__",
          [](ti::QuinticSplineJerkPeakConstraint* self,
             const ti::QuinticSplineModel& model,
             std::vector<std::shared_ptr<const ti::Var>> knots,
             std::shared_ptr<const ti::Var> peak,
             const std::vector<double>& witnesses,
             Eigen::VectorXd normalization,
             std::string name) {
            new (self) ti::QuinticSplineJerkPeakConstraint(model,
                                                           std::move(knots),
                                                           std::move(peak),
                                                           seconds(witnesses),
                                                           std::move(normalization),
                                                           std::move(name));
          },
          "model"_a,
          "knots"_a,
          "peak"_a,
          "witnesses_s"_a,
          "normalization"_a,
          "name"_a = "SplineJerkPeak");
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
  nb::class_<ti::QuinticSplineDerivativeLimits, ti::ConstraintSet>(m, "QuinticSplineDerivativeLimits")
      .def(
          "__init__",
          [](ti::QuinticSplineDerivativeLimits* self,
             const ti::QuinticSplineModel& model,
             std::vector<std::shared_ptr<const ti::Var>> knots,
             const std::vector<double>& times,
             std::vector<ti::Bounds> limits,
             std::string name) {
            new (self) ti::QuinticSplineDerivativeLimits(
                model, std::move(knots), seconds(times), std::move(limits), std::move(name));
          },
          "model"_a,
          "knots"_a,
          "times_s"_a,
          "sample_axis_limits"_a,
          "name"_a = "SplineInterpolatedVelocityLimits");
  nb::class_<ti::SplineTimedCartesianVelocity, ti::ConstraintSet>(m, "SplineTimedCartesianVelocity")
      .def("__init__", [](ti::SplineTimedCartesianVelocity* self,
                          ti::CartLineInfo info,
                          const Eigen::Matrix4d& start_matrix,
                          const Eigen::Matrix4d& end_matrix,
                          std::vector<std::shared_ptr<const ti::Var>> knots,
                          Eigen::VectorXd start_basis, Eigen::VectorXd start_offset,
                          Eigen::VectorXd end_basis, Eigen::VectorXd end_offset,
                          double interval_s, double speed_scale_m_s,
                          double reference_time_s, double weight, std::string name) {
        Eigen::Isometry3d start_target = Eigen::Isometry3d::Identity();
        Eigen::Isometry3d end_target = Eigen::Isometry3d::Identity();
        start_target.matrix() = start_matrix;
        end_target.matrix() = end_matrix;
        new (self) ti::SplineTimedCartesianVelocity(std::move(info), start_target, end_target, std::move(knots),
            std::move(start_basis), std::move(start_offset), std::move(end_basis), std::move(end_offset),
            ti::Seconds(interval_s), ti::WorkpieceSpeedScale{speed_scale_m_s}, ti::Seconds(reference_time_s),
            weight, std::move(name));
      }, "info"_a, "start_workpiece_pose"_a, "end_workpiece_pose"_a, "knots"_a,
         "start_basis"_a, "start_offset"_a, "end_basis"_a, "end_offset"_a,
         "interval_s"_a, "speed_scale_m_s"_a, "reference_time_s"_a, "weight"_a,
         "name"_a = "SplineTimedCartesianVelocity");
  nb::class_<ti::SplineTimedCartesianPose, ti::ConstraintSet>(m, "SplineTimedCartesianPose")
      .def("__init__", [](ti::SplineTimedCartesianPose* self,
                          const ti::CartLineInfo& info,
                          const Eigen::Matrix4d& target_matrix,
                          std::vector<std::shared_ptr<const ti::Var>> knots,
                          Eigen::VectorXd basis, Eigen::VectorXd offset,
                          double position_scale_m, double orientation_scale_rad,
                          double weight, std::string name) {
        if (!info.source_frame_offset.isApprox(Eigen::Isometry3d::Identity(), 0.0))
          throw ti::TimedCartesianPoseError("timed pose currently requires identity source offset");
        Eigen::Isometry3d target_pose = Eigen::Isometry3d::Identity();
        target_pose.matrix() = target_matrix;
        new (self) ti::SplineTimedCartesianPose(info.manip, info.source_frame, info.target_frame,
            target_pose, std::move(knots), std::move(basis), std::move(offset),
            ti::WorkpiecePositionScale{position_scale_m}, ti::OrientationScale{orientation_scale_rad}, weight, std::move(name));
      }, "info"_a, "target_workpiece_pose"_a, "knots"_a, "basis"_a, "offset"_a,
         "position_scale_m"_a, "orientation_scale_rad"_a, "weight"_a, "name"_a = "SplineTimedCartesianPose");
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
  nb::class_<ti::SplineCartesianPathProgressConstraint, ti::ConstraintSet>(m, "SplineCartesianPathProgressConstraint")
      .def(nb::init<ti::CartLineInfo,
                    std::vector<std::shared_ptr<const ti::Var>>,
                    Eigen::MatrixXd,
                    Eigen::MatrixXd,
                    Eigen::MatrixXd,
                    Eigen::MatrixXd,
                    Eigen::VectorXd,
                    bool,
                    bool,
                    std::string>(),
           "info"_a, "knots"_a, "basis"_a, "offset"_a,
           "path_points"_a, "path_tangents"_a, "arc_lengths"_a,
           "include_start"_a, "include_end"_a, "name"_a = "SplineCartesianPathProgress");
  nb::exception<ti::SplineTimedProgressError>(m, "SplineTimedProgressError", PyExc_ValueError);
  nb::class_<ti::SplineTimedProgress, ti::ConstraintSet>(m, "SplineTimedProgress")
      .def(nb::init<ti::CartLineInfo,
                    std::vector<std::shared_ptr<const ti::Var>>,
                    Eigen::MatrixXd,
                    Eigen::MatrixXd,
                    Eigen::VectorXd,
                    Eigen::VectorXd,
                    std::string>(),
           "info"_a,
           "knots"_a,
           "basis"_a,
           "offset"_a,
           "targets"_a,
           "coefficients"_a,
           "name"_a = "SplineTimedProgress");
}
