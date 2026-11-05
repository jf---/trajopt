/** @file dynamic_spline_collision_constraint.cpp
 *  @brief Apply an uncapped native collision constraint to an affine spline sample.
 */
#include <trajopt_ifopt/constraints/collision/dynamic_spline_collision_constraint.h>
#include <trajopt_ifopt/constraints/collision/discrete_collision_constraint.h>
#include <trajopt_ifopt/constraints/collision/discrete_collision_evaluators.h>
#include <trajopt_ifopt/variable_sets/node.h>
#include <trajopt_ifopt/variable_sets/nodes_variables.h>
#include <trajopt_ifopt/variable_sets/var.h>

#include <set>
#include <stdexcept>
#include <utility>

namespace trajopt_ifopt
{
DynamicSplineCollisionConstraint::DynamicSplineCollisionConstraint(
    std::shared_ptr<DiscreteCollisionEvaluator> evaluator,
    std::vector<std::shared_ptr<const Var>> knots,
    Eigen::VectorXd basis,
    Eigen::VectorXd offset,
    std::string name)
  : ConstraintSet(std::move(name), true), knots_(std::move(knots)), basis_(std::move(basis)), offset_(std::move(offset))
{
  if (!evaluator || knots_.empty() || basis_.size() != static_cast<Eigen::Index>(knots_.size()) ||
      !basis_.allFinite() || !offset_.allFinite() || offset_.size() == 0 || !knots_.front())
    throw std::invalid_argument("invalid spline collision evaluator, basis, offset or knot layout");

  const auto& names = knots_.front()->name();
  std::set<Eigen::Index> indices;
  for (const auto& knot : knots_)
  {
    if (!knot || !knot->getParent() || !knot->getParent()->getParent() || knot->size() != offset_.size() ||
        knot->name() != names || knot->getIndex() < 0)
      throw std::invalid_argument("spline knot axes, ownership or indices differ");
    for (Eigen::Index axis = 0; axis < knot->size(); ++axis)
      if (!indices.insert(knot->getIndex() + axis).second)
        throw std::invalid_argument("spline knot variable indices overlap");
  }

  auto node = std::make_unique<Node>("spline_collision_sample");
  const std::vector<Bounds> bounds(static_cast<std::size_t>(offset_.size()), NoBound);
  auto position = node->addVar("position", names, offset_, bounds);
  std::vector<std::unique_ptr<Node>> nodes;
  nodes.push_back(std::move(node));
  sample_variables_ = std::make_shared<NodesVariables>("spline_collision_sample", std::move(nodes));
  native_ = std::make_shared<DiscreteCollisionConstraintD>(std::move(evaluator), std::move(position), getName());
  native_->linkWithVariables(sample_variables_);
}

void DynamicSplineCollisionConstraint::initVariableDependedQuantities(const Variables::Ptr& variables)
{
  if (!variables)
    throw std::invalid_argument("spline collision requires linked knot variables");
  for (const auto& knot : knots_)
    if (knot->getParent()->getParent() != variables.get() || knot->getIndex() + knot->size() > variables->getRows())
      throw std::invalid_argument("spline knots do not belong to the linked variables");
  updated_ = false;
  // TrajOptQPProblem inspects bounds as soon as this set is linked.
  update();
}

int DynamicSplineCollisionConstraint::update()
{
  if (!variables_)
    throw std::logic_error("spline collision must be linked before update");
  Eigen::VectorXd sample = offset_;
  for (std::size_t i = 0; i < knots_.size(); ++i)
    sample.noalias() += basis_[static_cast<Eigen::Index>(i)] * knots_[i]->value();
  sample_variables_->setVariables(sample);
  rows_ = native_->update();
  non_zeros_ = native_->getNonZeros() * static_cast<Eigen::Index>(knots_.size());
  updated_ = true;
  return rows_;
}

void DynamicSplineCollisionConstraint::requireUpdated() const
{
  if (!updated_)
    throw std::logic_error("spline collision must be updated before reading contacts");
}

Eigen::VectorXd DynamicSplineCollisionConstraint::getValues() const
{
  requireUpdated();
  return native_->getValues();
}

std::vector<Bounds> DynamicSplineCollisionConstraint::getBounds() const
{
  requireUpdated();
  return native_->getBounds();
}

Eigen::VectorXd DynamicSplineCollisionConstraint::getCoefficients() const
{
  requireUpdated();
  return native_->getCoefficients();
}

Jacobian DynamicSplineCollisionConstraint::getJacobian() const
{
  requireUpdated();
  const Jacobian local = native_->getJacobian();
  std::vector<Eigen::Triplet<double>> entries;
  entries.reserve(static_cast<std::size_t>(local.nonZeros()) * knots_.size());
  for (Eigen::Index outer = 0; outer < local.outerSize(); ++outer)
    for (Jacobian::InnerIterator item(local, outer); item; ++item)
      for (std::size_t k = 0; k < knots_.size(); ++k)
        entries.emplace_back(item.row(), knots_[k]->getIndex() + item.col(),
                             basis_[static_cast<Eigen::Index>(k)] * item.value());
  Jacobian result(rows_, variables_->getRows());
  result.setFromTriplets(entries.begin(), entries.end());
  return result;
}
}  // namespace trajopt_ifopt
