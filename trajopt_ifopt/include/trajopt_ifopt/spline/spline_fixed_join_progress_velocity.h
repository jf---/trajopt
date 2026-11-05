#pragma once
#include <trajopt_ifopt/core/constraint_set.h>
#include <trajopt_ifopt/variable_sets/node.h>
#include <trajopt_ifopt/variable_sets/nodes_variables.h>
#include <trajopt_ifopt/variable_sets/var.h>
#include <Eigen/Core>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace trajopt_ifopt
{
/** Linear progress velocity at an explicitly fixed configuration.
 * The supplied gradient is evaluated at that configuration by the caller.
 * The caller must enforce that configuration at the join; otherwise this
 * constraint is not the actual Cartesian progress velocity.
 */
class SplineFixedJoinProgressVelocity : public ConstraintSet
{
public:
  SplineFixedJoinProgressVelocity(std::vector<std::shared_ptr<const Var>> knots,
                                  Eigen::VectorXd velocity_basis,
                                  Eigen::VectorXd velocity_offset,
                                  Eigen::VectorXd progress_gradient,
                                  std::string name = "FixedJoinProgressVelocity")
    : ConstraintSet(std::move(name), 1)
    , knots_(std::move(knots))
    , basis_(std::move(velocity_basis))
    , offset_(std::move(velocity_offset))
    , gradient_(std::move(progress_gradient))
  {
    if (knots_.empty() || !knots_.front() || basis_.size() != static_cast<Eigen::Index>(knots_.size()) ||
        offset_.size() == 0 || gradient_.size() != offset_.size() || !basis_.allFinite() || !offset_.allFinite() ||
        !gradient_.allFinite() || gradient_.squaredNorm() == 0)
      throw std::invalid_argument("invalid fixed join progress velocity input");
    const auto& names = knots_.front()->name();
    std::set<Eigen::Index> indices;
    for (const auto& knot : knots_)
    {
      if (!knot || !knot->getParent() || !knot->getParent()->getParent() || knot->size() != offset_.size() ||
          knot->name() != names || knot->getIndex() < 0)
        throw std::invalid_argument("fixed join knots have incompatible ownership or axes");
      for (Eigen::Index axis = 0; axis < knot->size(); ++axis)
        if (!indices.insert(knot->getIndex() + axis).second)
          throw std::invalid_argument("fixed join knot indices overlap");
    }
    non_zeros_ = static_cast<Eigen::Index>(knots_.size()) * offset_.size();
  }
  int update() override { return rows_; }
  Eigen::VectorXd getValues() const override
  {
    if (!variables_)
      throw std::logic_error("fixed join velocity must be linked");
    Eigen::VectorXd velocity = offset_;
    for (std::size_t i = 0; i < knots_.size(); ++i)
      velocity.noalias() += basis_[static_cast<Eigen::Index>(i)] * knots_[i]->value();
    return Eigen::VectorXd::Constant(1, gradient_.dot(velocity));
  }
  std::vector<Bounds> getBounds() const override { return { BoundZero }; }
  Eigen::VectorXd getCoefficients() const override { return Eigen::VectorXd::Ones(1); }
  Jacobian getJacobian() const override
  {
    if (!variables_)
      throw std::logic_error("fixed join velocity must be linked");
    std::vector<Eigen::Triplet<double>> entries;
    entries.reserve(static_cast<std::size_t>(non_zeros_));
    for (std::size_t i = 0; i < knots_.size(); ++i)
      for (Eigen::Index axis = 0; axis < offset_.size(); ++axis)
        entries.emplace_back(0, knots_[i]->getIndex() + axis, basis_[static_cast<Eigen::Index>(i)] * gradient_[axis]);
    Jacobian result(1, variables_->getRows());
    result.setFromTriplets(entries.begin(), entries.end());
    return result;
  }

private:
  void initVariableDependedQuantities(const Variables::Ptr& variables) override
  {
    if (!variables)
      throw std::invalid_argument("fixed join velocity requires linked variables");
    for (const auto& knot : knots_)
      if (knot->getParent()->getParent() != variables.get() || knot->getIndex() + knot->size() > variables->getRows())
        throw std::invalid_argument("fixed join knots do not belong to linked variables");
  }
  std::vector<std::shared_ptr<const Var>> knots_;
  Eigen::VectorXd basis_;
  Eigen::VectorXd offset_;
  Eigen::VectorXd gradient_;
};
}  // namespace trajopt_ifopt
