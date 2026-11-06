#pragma once

#include <trajopt_ifopt/spline/quintic_spline_model.h>
#include <Eigen/Core>
#include <stdexcept>
#include <vector>

namespace trajopt_ifopt
{
class TRAJOPT_API CubicSplineStageError : public std::runtime_error
{
public:
  using std::runtime_error::runtime_error;
  ~CubicSplineStageError() override;
};

/** Stage-local C2 cubic interpolation with optimized endpoint velocities.
 * Independent rows are knot positions, start velocity, then end velocity.
 * Times are modeled seconds; axis quantities use their native units.
 */
class TRAJOPT_API CubicSplineStage
{
public:
  static CubicSplineStage build(std::vector<Seconds> times, Eigen::Index axis_count);
  Eigen::Index knotCount() const { return static_cast<Eigen::Index>(times_.size()); }
  Eigen::Index variableCount() const { return knotCount() + 2; }
  Eigen::Index axisCount() const { return axis_count_; }
  SplineAffineMap map(const std::vector<Seconds>& samples, unsigned order) const;

private:
  CubicSplineStage(std::vector<Seconds> times, Eigen::Index axis_count, Eigen::MatrixXd second_derivatives);
  std::vector<Seconds> times_;
  Eigen::Index axis_count_;
  Eigen::MatrixXd second_derivatives_;
};
}  // namespace trajopt_ifopt
