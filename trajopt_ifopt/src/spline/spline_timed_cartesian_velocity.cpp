#include <trajopt_ifopt/spline/spline_timed_cartesian_velocity.h>
#include <tesseract/kinematics/joint_group.h>
#include <cmath>
#include <utility>

namespace trajopt_ifopt
{
SplineTimedCartesianVelocity::SplineTimedCartesianVelocity(CartLineInfo info,
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
                                                           std::string name)
  : ConstraintSet(std::move(name), 3)
  , start_rotation_(start_target.linear())
  , end_rotation_(end_target.linear())
  , residual_scale_(0)
  , coefficient_(0)
{
  if (!info.manip || !info.manip->isActiveLinkId(info.source_frame) ||
      !info.source_frame_offset.isApprox(Eigen::Isometry3d::Identity(), 0.0) || !std::isfinite(interval.count()) ||
      interval.count() <= 0 || !std::isfinite(speed.meters_per_second) || speed.meters_per_second <= 0 ||
      !std::isfinite(reference_time.count()) || reference_time.count() <= 0 || !std::isfinite(weight) || weight <= 0)
    throw TimedCartesianVelocityError("active TCP and positive finite modeled interval, scales and weight required");
  residual_scale_ = 1.0 / (interval.count() * speed.meters_per_second);
  coefficient_ = weight * interval.count() / reference_time.count();
  if (!std::isfinite(residual_scale_) || residual_scale_ <= 0 || !std::isfinite(coefficient_) || coefficient_ <= 0)
    throw TimedCartesianVelocityError("modeled velocity objective normalization overflow or underflow");
  start_ = std::make_shared<SplineTimedCartesianPose>(info.manip,
                                                      info.source_frame,
                                                      info.target_frame,
                                                      start_target,
                                                      knots,
                                                      std::move(start_basis),
                                                      std::move(start_offset),
                                                      WorkpiecePositionScale{ 1.0 },
                                                      OrientationScale{ 1.0 },
                                                      1.0,
                                                      getName() + "_start");
  end_ = std::make_shared<SplineTimedCartesianPose>(std::move(info.manip),
                                                    info.source_frame,
                                                    info.target_frame,
                                                    end_target,
                                                    std::move(knots),
                                                    std::move(end_basis),
                                                    std::move(end_offset),
                                                    WorkpiecePositionScale{ 1.0 },
                                                    OrientationScale{ 1.0 },
                                                    1.0,
                                                    getName() + "_end");
}

void SplineTimedCartesianVelocity::initVariableDependedQuantities(const Variables::Ptr& variables)
{
  if (!variables)
    throw TimedCartesianVelocityError("timed Cartesian velocity requires linked variables");
  start_->linkWithVariables(variables);
  end_->linkWithVariables(variables);
  non_zeros_ = rows_ * variables->getRows();
}

Eigen::VectorXd SplineTimedCartesianVelocity::getValues() const
{
  const Eigen::VectorXd start_error = start_->getValues();
  const Eigen::VectorXd end_error = end_->getValues();
  return residual_scale_ * (end_rotation_ * end_error.head<3>() - start_rotation_ * start_error.head<3>());
}
std::vector<Bounds> SplineTimedCartesianVelocity::getBounds() const { return std::vector<Bounds>(3, BoundZero); }
Eigen::VectorXd SplineTimedCartesianVelocity::getCoefficients() const
{
  return Eigen::VectorXd::Constant(3, coefficient_);
}
Jacobian SplineTimedCartesianVelocity::getJacobian() const
{
  if (!variables_)
    throw TimedCartesianVelocityError("timed Cartesian velocity must be linked before evaluation");
  const Jacobian start = start_->getJacobian();
  const Jacobian end = end_->getJacobian();
  std::vector<Eigen::Triplet<double>> entries;
  entries.reserve(static_cast<std::size_t>((start.nonZeros() + end.nonZeros()) * 3));
  for (Eigen::Index row = 0; row < 3; ++row)
  {
    for (Jacobian::InnerIterator item(start, row); item; ++item)
      for (Eigen::Index target_row = 0; target_row < 3; ++target_row)
        entries.emplace_back(
            target_row, item.col(), -residual_scale_ * start_rotation_(target_row, row) * item.value());
    for (Jacobian::InnerIterator item(end, row); item; ++item)
      for (Eigen::Index target_row = 0; target_row < 3; ++target_row)
        entries.emplace_back(target_row, item.col(), residual_scale_ * end_rotation_(target_row, row) * item.value());
  }
  Jacobian result(rows_, variables_->getRows());
  result.setFromTriplets(entries.begin(), entries.end());
  return result;
}
}  // namespace trajopt_ifopt
