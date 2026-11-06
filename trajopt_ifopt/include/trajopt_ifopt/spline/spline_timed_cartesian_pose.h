#pragma once

#include <trajopt_ifopt/constraints/cartesian_position_constraint.h>
#include <Eigen/Geometry>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace trajopt_ifopt
{
class NodesVariables;
class Var;

class TimedCartesianPoseError : public std::invalid_argument
{
public:
  using std::invalid_argument::invalid_argument;
};

class TimedCartesianPoseLinkError : public std::logic_error
{
public:
  using std::logic_error::logic_error;
};

// Objective normalization scales, not process acceptance tolerances.
struct WorkpiecePositionScale
{
  double meters;
};
struct OrientationScale
{
  double radians;
};

/** Residual between the actual spline TCP and a modeled-clock target pose.
 * The target is an offset in the target link frame. Translation is measured in
 * metres and orientation as the target-relative angle-axis vector in radians.
 * The separately supplied scales make a dimensionless squared objective.
 */
class SplineTimedCartesianPose : public ConstraintSet
{
public:
  SplineTimedCartesianPose(std::shared_ptr<const tesseract::kinematics::JointGroup> manip,
                           tesseract::common::LinkId source_frame,
                           tesseract::common::LinkId target_frame,
                           Eigen::Isometry3d target_pose,
                           std::vector<std::shared_ptr<const Var>> knots,
                           Eigen::VectorXd basis,
                           Eigen::VectorXd offset,
                           WorkpiecePositionScale position_scale,
                           OrientationScale orientation_scale,
                           double weight,
                           std::string name = "SplineTimedCartesianPose");
  int update() override { return rows_; }
  Eigen::VectorXd getValues() const override;
  std::vector<Bounds> getBounds() const override;
  Eigen::VectorXd getCoefficients() const override;
  Jacobian getJacobian() const override;

private:
  void initVariableDependedQuantities(const Variables::Ptr& variables) override;
  void updateSample() const;
  std::vector<std::shared_ptr<const Var>> knots_;
  Eigen::VectorXd basis_;
  Eigen::VectorXd offset_;
  std::shared_ptr<NodesVariables> sample_variables_;
  std::shared_ptr<CartPosConstraint> native_;
};
}  // namespace trajopt_ifopt
