#include <trajopt_ifopt/spline/cubic_spline_stage.h>

#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <utility>

namespace trajopt_ifopt
{
CubicSplineStageError::~CubicSplineStageError() = default;

CubicSplineStage::CubicSplineStage(std::vector<Seconds> times,
                                   Eigen::Index axis_count,
                                   Eigen::MatrixXd second_derivatives)
  : times_(std::move(times)), axis_count_(axis_count), second_derivatives_(std::move(second_derivatives))
{
}

CubicSplineStage CubicSplineStage::build(std::vector<Seconds> times, Eigen::Index axis_count)
{
  if (times.size() < 2 || axis_count <= 0)
    throw CubicSplineStageError("at least two times and one axis required");
  for (std::size_t i = 0; i < times.size(); ++i)
    if (!std::isfinite(times[i].count()) ||
        (i > 0 && (times[i] <= times[i - 1] || !std::isfinite((times[i] - times[i - 1]).count()))))
      throw CubicSplineStageError("stage clock must be finite and strictly increasing");
  const Eigen::Index n = static_cast<Eigen::Index>(times.size());
  Eigen::MatrixXd system = Eigen::MatrixXd::Zero(n, n);
  Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(n, n + 2);
  const double first = (times[1] - times[0]).count();
  system(0, 0) = 2 * first;
  system(0, 1) = first;
  rhs(0, 0) = -6 / first;
  rhs(0, 1) = 6 / first;
  rhs(0, n) = -6;
  for (Eigen::Index i = 1; i + 1 < n; ++i)
  {
    const double left = (times[static_cast<std::size_t>(i)] - times[static_cast<std::size_t>(i - 1)]).count();
    const double right = (times[static_cast<std::size_t>(i + 1)] - times[static_cast<std::size_t>(i)]).count();
    system(i, i - 1) = left;
    system(i, i) = 2 * (left + right);
    system(i, i + 1) = right;
    rhs(i, i - 1) = 6 / left;
    rhs(i, i) = -6 * (1 / left + 1 / right);
    rhs(i, i + 1) = 6 / right;
  }
  const double last = (times.back() - times[times.size() - 2]).count();
  system(n - 1, n - 2) = last;
  system(n - 1, n - 1) = 2 * last;
  rhs(n - 1, n - 2) = 6 / last;
  rhs(n - 1, n - 1) = -6 / last;
  rhs(n - 1, n + 1) = 6;
  Eigen::FullPivLU<Eigen::MatrixXd> factor(system);
  if (!factor.isInvertible())
    throw CubicSplineStageError("stage cubic interpolation system is singular");
  return CubicSplineStage(std::move(times), axis_count, factor.solve(rhs));
}

SplineAffineMap CubicSplineStage::map(const std::vector<Seconds>& samples, unsigned order) const
{
  if (samples.empty() || order > 3)
    throw CubicSplineStageError("samples and derivative order at most three required");
  Eigen::MatrixXd basis(static_cast<Eigen::Index>(samples.size()), variableCount());
  for (std::size_t j = 0; j < samples.size(); ++j)
  {
    const double time = samples[j].count();
    if (!std::isfinite(time) || samples[j] < times_.front() || samples[j] > times_.back())
      throw CubicSplineStageError("stage sample outside modeled clock");
    const auto upper = std::upper_bound(times_.begin(), times_.end(), samples[j]);
    const Eigen::Index i = std::min(static_cast<Eigen::Index>(upper - times_.begin() - 1), knotCount() - 2);
    const double h = (times_[static_cast<std::size_t>(i + 1)] - times_[static_cast<std::size_t>(i)]).count();
    const double x = (samples[j] - times_[static_cast<std::size_t>(i)]).count();
    Eigen::RowVectorXd left = Eigen::RowVectorXd::Zero(variableCount());
    Eigen::RowVectorXd right = Eigen::RowVectorXd::Zero(variableCount());
    left[i] = 1;
    right[i + 1] = 1;
    const Eigen::RowVectorXd b =
        (right - left) / h - h * (2 * second_derivatives_.row(i) + second_derivatives_.row(i + 1)) / 6;
    const Eigen::RowVectorXd c = second_derivatives_.row(i) / 2;
    const Eigen::RowVectorXd d = (second_derivatives_.row(i + 1) - second_derivatives_.row(i)) / (6 * h);
    Eigen::RowVectorXd row(variableCount());
    switch (order)
    {
      case 0:
        row = left + b * x + c * x * x + d * x * x * x;
        break;
      case 1:
        row = b + 2 * c * x + 3 * d * x * x;
        break;
      case 2:
        row = 2 * c + 6 * d * x;
        break;
      default:
        row = 6 * d;
        break;
    }
    basis.row(static_cast<Eigen::Index>(j)) = row;
  }
  return { std::move(basis), Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(samples.size()), axis_count_) };
}
}  // namespace trajopt_ifopt
