#pragma once

#include <trajopt_ifopt/constraints/cartesian_line_constraint.h>
#include <trajopt_ifopt/spline/quintic_spline_model.h>
#include <trajopt_ifopt/spline/spline_timed_cartesian_pose.h>
#include <Eigen/Geometry>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace trajopt_ifopt
{
struct WorkpieceSpeedScale
{
  double meters_per_second;
};

class TimedCartesianVelocityError : public std::invalid_argument
{
public:
  using std::invalid_argument::invalid_argument;
};

/** Soft residual of average workpiece-frame TCP velocity from generated targets.
 * Each sample is an actual spline pose. Their workpiece-frame position errors
 * are differenced and divided by the modeled interval and reference speed.
 * A squared-cost consumer uses coefficients weight * interval / reference_time.
 * This is neither an instantaneous velocity bound nor a process tolerance.
 */
class SplineTimedCartesianVelocity : public ConstraintSet
{
public:
  SplineTimedCartesianVelocity(CartLineInfo info,
                               Eigen::Isometry3d start_target,
                               Eigen::Isometry3d end_target,
                               std::vector<std::shared_ptr<const Var>> knots,
                               Eigen::VectorXd start_basis,
                               Eigen::VectorXd start_offset,
                               Eigen::VectorXd end_basis,
                               Eigen::VectorXd end_offset,
                               Seconds interval,
                               WorkpieceSpeedScale speed,
                               Seconds reference_time,
                               double weight,
                               std::string name = "SplineTimedCartesianVelocity");
  int update() override { return rows_; }
  Eigen::VectorXd getValues() const override;
  std::vector<Bounds> getBounds() const override;
  Eigen::VectorXd getCoefficients() const override;
  Jacobian getJacobian() const override;

private:
  void initVariableDependedQuantities(const Variables::Ptr& variables) override;
  Eigen::Matrix3d start_rotation_;
  Eigen::Matrix3d end_rotation_;
  double residual_scale_;
  double coefficient_;
  std::shared_ptr<SplineTimedCartesianPose> start_;
  std::shared_ptr<SplineTimedCartesianPose> end_;
};
}  // namespace trajopt_ifopt
