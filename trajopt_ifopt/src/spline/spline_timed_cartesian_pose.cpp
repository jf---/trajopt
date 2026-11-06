#include <trajopt_ifopt/spline/spline_timed_cartesian_pose.h>
#include <trajopt_ifopt/variable_sets/node.h>
#include <trajopt_ifopt/variable_sets/nodes_variables.h>
#include <trajopt_ifopt/variable_sets/var.h>
#include <tesseract/kinematics/joint_group.h>
#include <set>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace trajopt_ifopt
{
SplineTimedCartesianPose::SplineTimedCartesianPose(std::shared_ptr<const tesseract::kinematics::JointGroup> manip,
                                                   tesseract::common::LinkId source_frame,
                                                   tesseract::common::LinkId target_frame,
                                                   Eigen::Isometry3d target_pose,
                                                   std::vector<std::shared_ptr<const Var>> knots,
                                                   Eigen::VectorXd basis,
                                                   Eigen::VectorXd offset,
                                                   WorkpiecePositionScale position_scale,
                                                   OrientationScale orientation_scale,
                                                   double weight,
                                                   std::string name)
  : ConstraintSet(std::move(name), 6), knots_(std::move(knots)), basis_(std::move(basis)), offset_(std::move(offset))
{
  if (!manip || knots_.empty() || !knots_.front() || basis_.size() != static_cast<Eigen::Index>(knots_.size()) ||
      offset_.size() != manip->numJoints() || !basis_.allFinite() || !offset_.allFinite() ||
      !target_pose.matrix().allFinite() || !std::isfinite(position_scale.meters) || position_scale.meters <= 0 ||
      !std::isfinite(orientation_scale.radians) || orientation_scale.radians <= 0 || !std::isfinite(weight) ||
      weight <= 0)
    throw TimedCartesianPoseError("invalid timed Cartesian pose target, normalization, basis, or knot layout");
  // Rotation validity allows only binary64 rounding in a rigid transform.
  constexpr double ROTATION_ROUNDING_BAND = 128 * std::numeric_limits<double>::epsilon();
  const Eigen::Matrix3d rotation = target_pose.linear();
  const Eigen::RowVector4d homogeneous(0.0, 0.0, 0.0, 1.0);
  if ((rotation.transpose() * rotation - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff() > ROTATION_ROUNDING_BAND ||
      std::abs(rotation.determinant() - 1.0) > ROTATION_ROUNDING_BAND ||
      (target_pose.matrix().row(3) - homogeneous).cwiseAbs().maxCoeff() > ROTATION_ROUNDING_BAND)
    throw TimedCartesianPoseError("timed Cartesian target must be a rigid pose");
  const auto& names = knots_.front()->name();
  std::set<Eigen::Index> indices;
  for (const auto& knot : knots_)
  {
    if (!knot || !knot->getParent() || !knot->getParent()->getParent() || knot->size() != offset_.size() ||
        knot->name() != names || knot->getIndex() < 0)
      throw TimedCartesianPoseError("timed Cartesian pose knot axes, ownership or indices differ");
    for (Eigen::Index axis = 0; axis < knot->size(); ++axis)
      if (!indices.insert(knot->getIndex() + axis).second)
        throw TimedCartesianPoseError("timed Cartesian pose knot variable indices overlap");
  }
  auto node = std::make_unique<Node>("timed_pose_sample");
  auto position =
      node->addVar("position", names, offset_, std::vector<Bounds>(static_cast<std::size_t>(offset_.size()), NoBound));
  std::vector<std::unique_ptr<Node>> nodes;
  nodes.push_back(std::move(node));
  sample_variables_ = std::make_shared<NodesVariables>("timed_pose_sample", std::move(nodes));
  Eigen::VectorXd coefficients(6);
  coefficients.head<3>().setConstant(weight / (position_scale.meters * position_scale.meters));
  coefficients.tail<3>().setConstant(weight / (orientation_scale.radians * orientation_scale.radians));
  if (!coefficients.allFinite() || (coefficients.array() <= 0.0).any())
    throw TimedCartesianPoseError("timed Cartesian pose objective scaling overflow or underflow");
  native_ = std::make_shared<CartPosConstraint>(std::move(position),
                                                coefficients,
                                                std::vector<Bounds>(6, Bounds(0.0, 0.0)),
                                                std::move(manip),
                                                std::move(source_frame),
                                                std::move(target_frame),
                                                Eigen::Isometry3d::Identity(),
                                                target_pose,
                                                getName());
  if (native_->getRows() != rows_)
    throw TimedCartesianPoseError("timed Cartesian pose weights are too small for native residual rows");
  native_->use_numeric_differentiation = false;
  native_->linkWithVariables(sample_variables_);
  non_zeros_ = static_cast<Eigen::Index>(rows_) * offset_.size() * static_cast<Eigen::Index>(knots_.size());
}

void SplineTimedCartesianPose::initVariableDependedQuantities(const Variables::Ptr& variables)
{
  if (!variables)
    throw TimedCartesianPoseError("timed Cartesian pose requires linked variables");
  for (const auto& knot : knots_)
    if (knot->getParent()->getParent() != variables.get() || knot->getIndex() + knot->size() > variables->getRows())
      throw TimedCartesianPoseError("timed Cartesian pose knots do not belong to linked variables");
}

void SplineTimedCartesianPose::updateSample() const
{
  if (!variables_)
    throw TimedCartesianPoseLinkError("timed Cartesian pose must be linked before evaluation");
  Eigen::VectorXd sample = offset_;
  for (std::size_t i = 0; i < knots_.size(); ++i)
    sample.noalias() += basis_[static_cast<Eigen::Index>(i)] * knots_[i]->value();
  sample_variables_->setVariables(sample);
}

Eigen::VectorXd SplineTimedCartesianPose::getValues() const
{
  updateSample();
  return native_->getValues();
}
std::vector<Bounds> SplineTimedCartesianPose::getBounds() const { return native_->getBounds(); }
Eigen::VectorXd SplineTimedCartesianPose::getCoefficients() const { return native_->getCoefficients(); }
Jacobian SplineTimedCartesianPose::getJacobian() const
{
  updateSample();
  const Jacobian local = native_->getJacobian();
  std::vector<Eigen::Triplet<double>> entries;
  entries.reserve(static_cast<std::size_t>(local.nonZeros()) * knots_.size());
  for (Eigen::Index outer = 0; outer < local.outerSize(); ++outer)
    for (Jacobian::InnerIterator item(local, outer); item; ++item)
      for (std::size_t k = 0; k < knots_.size(); ++k)
        entries.emplace_back(
            item.row(), knots_[k]->getIndex() + item.col(), basis_[static_cast<Eigen::Index>(k)] * item.value());
  Jacobian result(rows_, variables_->getRows());
  result.setFromTriplets(entries.begin(), entries.end());
  return result;
}
}  // namespace trajopt_ifopt
