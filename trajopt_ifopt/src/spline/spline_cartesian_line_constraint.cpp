#include <trajopt_ifopt/spline/spline_cartesian_line_constraint.h>
#include <trajopt_ifopt/variable_sets/node.h>
#include <trajopt_ifopt/variable_sets/nodes_variables.h>
#include <trajopt_ifopt/variable_sets/var.h>
#include <tesseract/kinematics/joint_group.h>
#include <set>
#include <stdexcept>
#include <utility>

namespace trajopt_ifopt
{
SplineCartesianLineConstraint::SplineCartesianLineConstraint(CartLineInfo info,
                                                             std::vector<std::shared_ptr<const Var>> knots,
                                                             Eigen::VectorXd basis,
                                                             Eigen::VectorXd offset,
                                                             Eigen::VectorXd coefficients,
                                                             std::string name)
  : ConstraintSet(std::move(name), static_cast<int>(info.indices.size()))
  , knots_(std::move(knots))
  , basis_(std::move(basis))
  , offset_(std::move(offset))
{
  if (!info.manip || knots_.empty() || !knots_.front() || basis_.size() != static_cast<Eigen::Index>(knots_.size()) ||
      offset_.size() != info.manip->numJoints() || !basis_.allFinite() || !offset_.allFinite())
    throw std::invalid_argument("invalid spline line basis, offset, kinematics or knot layout");
  const auto& names = knots_.front()->name();
  std::set<Eigen::Index> indices;
  for (const auto& knot : knots_)
  {
    if (!knot || !knot->getParent() || !knot->getParent()->getParent() || knot->size() != offset_.size() ||
        knot->name() != names || knot->getIndex() < 0)
      throw std::invalid_argument("spline line knot axes, ownership or indices differ");
    for (Eigen::Index axis = 0; axis < knot->size(); ++axis)
      if (!indices.insert(knot->getIndex() + axis).second)
        throw std::invalid_argument("spline line knot variable indices overlap");
  }
  auto node = std::make_unique<Node>("spline_line_sample");
  auto position =
      node->addVar("position", names, offset_, std::vector<Bounds>(static_cast<std::size_t>(offset_.size()), NoBound));
  std::vector<std::unique_ptr<Node>> nodes;
  nodes.push_back(std::move(node));
  sample_variables_ = std::make_shared<NodesVariables>("spline_line_sample", std::move(nodes));
  native_ = std::make_shared<CartLineConstraint>(std::move(info), std::move(position), coefficients, getName());
  native_->use_numeric_differentiation = false;
  native_->linkWithVariables(sample_variables_);
  non_zeros_ = static_cast<Eigen::Index>(rows_) * offset_.size() * static_cast<Eigen::Index>(knots_.size());
}

void SplineCartesianLineConstraint::initVariableDependedQuantities(const Variables::Ptr& variables)
{
  if (!variables)
    throw std::invalid_argument("spline line requires linked variables");
  for (const auto& knot : knots_)
    if (knot->getParent()->getParent() != variables.get() || knot->getIndex() + knot->size() > variables->getRows())
      throw std::invalid_argument("spline line knots do not belong to linked variables");
}

void SplineCartesianLineConstraint::updateSample() const
{
  if (!variables_)
    throw std::logic_error("spline line must be linked before evaluation");
  Eigen::VectorXd sample = offset_;
  for (std::size_t i = 0; i < knots_.size(); ++i)
    sample.noalias() += basis_[static_cast<Eigen::Index>(i)] * knots_[i]->value();
  sample_variables_->setVariables(sample);
}

Eigen::VectorXd SplineCartesianLineConstraint::getValues() const
{
  updateSample();
  return native_->getValues();
}
std::vector<Bounds> SplineCartesianLineConstraint::getBounds() const { return native_->getBounds(); }
Eigen::VectorXd SplineCartesianLineConstraint::getCoefficients() const { return native_->getCoefficients(); }
Jacobian SplineCartesianLineConstraint::getJacobian() const
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
