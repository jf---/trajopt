#include <trajopt_ifopt/spline/quintic_spline_model.h>

#include <Eigen/LU>
#include <unsupported/Eigen/Splines>
#include <array>
#include <cmath>
#include <utility>

namespace trajopt_ifopt
{
QuinticSplineError::~QuinticSplineError() = default;

namespace
{
constexpr int DEGREE = 5;
constexpr std::size_t MINIMUM_TIMES = 6;
using BasisSpline = Eigen::Spline<double, 1, DEGREE>;

Eigen::RowVectorXd evaluateBasis(const Eigen::RowVectorXd& knots, double time, unsigned order)
{
  const Eigen::Index span = BasisSpline::Span(time, DEGREE, knots);
  const auto derivatives = BasisSpline::BasisFunctionDerivatives(time, static_cast<Eigen::Index>(order), DEGREE, knots);
  Eigen::RowVectorXd row = Eigen::RowVectorXd::Zero(knots.size() - DEGREE - 1);
  row.segment(span - DEGREE, DEGREE + 1) = derivatives.row(static_cast<Eigen::Index>(order));
  return row;
}
}  // namespace

QuinticSplineModel::QuinticSplineModel(std::vector<Seconds> times,
                                       Eigen::RowVectorXd knots,
                                       Eigen::MatrixXd position_coefficients,
                                       Eigen::MatrixXd boundary_coefficients)
  : times_(std::move(times))
  , knots_(std::move(knots))
  , position_coefficients_(std::move(position_coefficients))
  , boundary_coefficients_(std::move(boundary_coefficients))
{
}

QuinticSplineModel QuinticSplineModel::build(std::vector<Seconds> times,
                                             const Eigen::VectorXd& velocity_start,
                                             const Eigen::VectorXd& acceleration_start,
                                             const Eigen::VectorXd& velocity_end,
                                             const Eigen::VectorXd& acceleration_end)
{
  return buildVariableAcceleration(
      std::move(times), velocity_start, acceleration_start, velocity_end, acceleration_end);
}

QuinticSplineModel QuinticSplineModel::buildVariableAcceleration(std::vector<Seconds> times,
                                                                 const Eigen::VectorXd& velocity_start,
                                                                 std::optional<Eigen::VectorXd> acceleration_start,
                                                                 const Eigen::VectorXd& velocity_end,
                                                                 std::optional<Eigen::VectorXd> acceleration_end)
{
  if (times.size() < MINIMUM_TIMES || velocity_start.size() == 0 || velocity_end.size() != velocity_start.size() ||
      !velocity_start.allFinite() || !velocity_end.allFinite() ||
      (acceleration_start &&
       (acceleration_start->size() != velocity_start.size() || !acceleration_start->allFinite())) ||
      (acceleration_end && (acceleration_end->size() != velocity_start.size() || !acceleration_end->allFinite())))
    throw QuinticSplineError("invalid spline knot count or boundary derivatives");
  for (std::size_t i = 0; i < times.size(); ++i)
    if (!std::isfinite(times[i].count()) || (i > 0 && times[i] <= times[i - 1]))
      throw QuinticSplineError("spline clock must be finite and strictly increasing");
  const double first = times.front().count();
  const double duration = times.back().count() - first;
  if (!std::isfinite(duration) || duration <= 0)
    throw QuinticSplineError("spline duration must be finite and positive");

  const Eigen::Index n = static_cast<Eigen::Index>(times.size());
  Eigen::RowVectorXd knots(n + 2 * DEGREE);
  for (int i = 0; i < DEGREE; ++i)
    knots[i] = 0.0;
  for (Eigen::Index i = 0; i < n; ++i)
    knots[DEGREE + i] = (times[static_cast<std::size_t>(i)].count() - first) / duration;
  for (int i = 0; i < DEGREE; ++i)
    knots[DEGREE + n + i] = 1.0;
  if (!knots.allFinite())
    throw QuinticSplineError("normalized clock is not finite");

  const Eigen::Index controls = n + 4;
  Eigen::MatrixXd system(controls, controls);
  system.row(0) = evaluateBasis(knots, 0.0, 1);
  system.row(1) = evaluateBasis(knots, 0.0, 2);
  for (Eigen::Index i = 0; i < n; ++i)
    system.row(i + 2) = evaluateBasis(knots, knots[DEGREE + i], 0);
  system.row(n + 2) = evaluateBasis(knots, 1.0, 1);
  system.row(n + 3) = evaluateBasis(knots, 1.0, 2);
  Eigen::FullPivLU<Eigen::MatrixXd> factor(system);
  if (!factor.isInvertible())
    throw QuinticSplineError("spline interpolation system is singular");
  const Eigen::Index free_count =
      static_cast<Eigen::Index>(!acceleration_start) + static_cast<Eigen::Index>(!acceleration_end);
  Eigen::MatrixXd position_rhs = Eigen::MatrixXd::Zero(controls, n + free_count);
  position_rhs.block(2, 0, n, n).setIdentity();
  Eigen::Index variable = n;
  if (!acceleration_start)
    position_rhs(1, variable++) = duration * duration;
  if (!acceleration_end)
    position_rhs(n + 3, variable++) = duration * duration;
  Eigen::MatrixXd boundary_rhs = Eigen::MatrixXd::Zero(controls, velocity_start.size());
  boundary_rhs.row(0) = duration * velocity_start.transpose();
  if (acceleration_start)
    boundary_rhs.row(1) = duration * duration * acceleration_start->transpose();
  boundary_rhs.row(n + 2) = duration * velocity_end.transpose();
  if (acceleration_end)
    boundary_rhs.row(n + 3) = duration * duration * acceleration_end->transpose();
  return QuinticSplineModel(std::move(times), std::move(knots), factor.solve(position_rhs), factor.solve(boundary_rhs));
}

Eigen::RowVectorXd QuinticSplineModel::controlBasis(double normalized_time, unsigned order) const
{
  return evaluateBasis(knots_, normalized_time, order);
}

SplineAffineMap QuinticSplineModel::map(const std::vector<Seconds>& times, unsigned order) const
{
  if (order > DEGREE || times.empty())
    throw QuinticSplineError("spline evaluation requires times and derivative order at most five");
  const double first = times_.front().count();
  const double last = times_.back().count();
  const double duration = last - first;
  const Eigen::Index count = static_cast<Eigen::Index>(times.size());
  Eigen::MatrixXd basis(count, position_coefficients_.cols());
  Eigen::MatrixXd offset(count, boundary_coefficients_.cols());
  for (Eigen::Index i = 0; i < count; ++i)
  {
    const double time = times[static_cast<std::size_t>(i)].count();
    if (!std::isfinite(time) || time < first || time > last)
      throw QuinticSplineError("spline evaluation time is outside the modeled clock");
    const Eigen::RowVectorXd row = controlBasis((time - first) / duration, order) / std::pow(duration, order);
    basis.row(i).noalias() = row * position_coefficients_;
    offset.row(i).noalias() = row * boundary_coefficients_;
  }
  return { std::move(basis), std::move(offset) };
}

std::vector<Eigen::MatrixXd> QuinticSplineModel::evaluate(const std::vector<Seconds>& times,
                                                          const Eigen::MatrixXd& positions,
                                                          unsigned max_order) const
{
  if (max_order > DEGREE || times.empty() || positions.rows() != variableCount() || positions.cols() != axisCount() ||
      !positions.allFinite())
    throw QuinticSplineError("invalid batched spline times, positions or derivative order");
  const double first = times_.front().count();
  const double last = times_.back().count();
  const double duration = last - first;
  const Eigen::MatrixXd controls = position_coefficients_ * positions + boundary_coefficients_;
  std::vector<Eigen::MatrixXd> result;
  result.reserve(max_order + 1);
  for (unsigned order = 0; order <= max_order; ++order)
    result.emplace_back(static_cast<Eigen::Index>(times.size()), axisCount());
  for (std::size_t i = 0; i < times.size(); ++i)
  {
    const double time = times[i].count();
    if (!std::isfinite(time) || time < first || time > last)
      throw QuinticSplineError("spline evaluation time is outside the modeled clock");
    const double parameter = (time - first) / duration;
    const Eigen::Index span = BasisSpline::Span(parameter, DEGREE, knots_);
    const auto derivatives =
        BasisSpline::BasisFunctionDerivatives(parameter, static_cast<Eigen::Index>(max_order), DEGREE, knots_);
    const auto local_controls = controls.middleRows(span - DEGREE, DEGREE + 1);
    double time_scale = 1.0;
    for (unsigned order = 0; order <= max_order; ++order)
    {
      result[order].row(static_cast<Eigen::Index>(i)).noalias() =
          time_scale * derivatives.row(static_cast<Eigen::Index>(order)).matrix() * local_controls;
      time_scale /= duration;
    }
  }
  return result;
}

JerkQuadrature QuinticSplineModel::jerkQuadrature() const
{
  const double node = std::sqrt(3.0 / 5.0);
  const std::array<double, 3> nodes = { -node, 0.0, node };
  const std::array<double, 3> weights = { 5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0 };
  const std::size_t count = 3 * (times_.size() - 1);
  std::vector<Seconds> samples;
  samples.reserve(count);
  Eigen::VectorXd quadrature_weights(static_cast<Eigen::Index>(count));
  for (std::size_t interval = 0; interval + 1 < times_.size(); ++interval)
  {
    const double start = times_[interval].count();
    const double span = times_[interval + 1].count() - start;
    for (std::size_t j = 0; j < nodes.size(); ++j)
    {
      samples.emplace_back(start + span * (nodes[j] + 1.0) / 2.0);
      quadrature_weights[static_cast<Eigen::Index>(3 * interval + j)] = span * weights[j] / 2.0;
    }
  }
  return { map(samples, 3), std::move(quadrature_weights) };
}
JerkQuadrature QuinticSplineModel::jerkFourthQuadrature() const
{
  const double inner = std::sqrt(5.0 - 2.0 * std::sqrt(10.0 / 7.0)) / 3.0;
  const double outer = std::sqrt(5.0 + 2.0 * std::sqrt(10.0 / 7.0)) / 3.0;
  const double outer_weight = (322.0 - 13.0 * std::sqrt(70.0)) / 900.0;
  const double inner_weight = (322.0 + 13.0 * std::sqrt(70.0)) / 900.0;
  const std::array<double, 5> nodes = { -outer, -inner, 0.0, inner, outer };
  const std::array<double, 5> weights = { outer_weight, inner_weight, 128.0 / 225.0, inner_weight, outer_weight };
  const std::size_t count = nodes.size() * (times_.size() - 1);
  std::vector<Seconds> samples;
  samples.reserve(count);
  Eigen::VectorXd quadrature_weights(static_cast<Eigen::Index>(count));
  for (std::size_t interval = 0; interval + 1 < times_.size(); ++interval)
  {
    const double start = times_[interval].count();
    const double span = times_[interval + 1].count() - start;
    for (std::size_t j = 0; j < nodes.size(); ++j)
    {
      samples.emplace_back(start + span * (nodes[j] + 1.0) / 2.0);
      quadrature_weights[static_cast<Eigen::Index>(nodes.size() * interval + j)] = span * weights[j] / 2.0;
    }
  }
  return { map(samples, 3), std::move(quadrature_weights) };
}
}  // namespace trajopt_ifopt
