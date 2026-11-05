/** @file dynamic_spline_collision_constraint.h
 *  @brief Uncapped discrete collisions on an affine spline sample.
 */
#ifndef TRAJOPT_IFOPT_DYNAMIC_SPLINE_COLLISION_CONSTRAINT_H
#define TRAJOPT_IFOPT_DYNAMIC_SPLINE_COLLISION_CONSTRAINT_H

#include <trajopt_ifopt/core/constraint_set.h>
#include <Eigen/Core>
#include <memory>
#include <string>
#include <vector>

namespace trajopt_ifopt
{
class DiscreteCollisionConstraintD;
class DiscreteCollisionEvaluator;
class NodesVariables;
class Var;

/** @brief Apply the native dynamic collision rows to q = offset + sum(basis_i * knot_i).
 *
 * The native contact evaluator supplies every row, coefficient, bound and local
 * gradient. The chain rule maps each row to the linked knot variables.
 * This constrains the supplied sample only, not the continuous spline.
 */
class DynamicSplineCollisionConstraint : public ConstraintSet
{
public:
  DynamicSplineCollisionConstraint(std::shared_ptr<DiscreteCollisionEvaluator> evaluator,
                                   std::vector<std::shared_ptr<const Var>> knots,
                                   Eigen::VectorXd basis,
                                   Eigen::VectorXd offset,
                                   std::string name = "DynamicSplineCollision");
  int update() override;
  Eigen::VectorXd getValues() const override;
  std::vector<Bounds> getBounds() const override;
  Eigen::VectorXd getCoefficients() const override;
  Jacobian getJacobian() const override;

private:
  void initVariableDependedQuantities(const Variables::Ptr& variables) override;
  void requireUpdated() const;
  std::vector<std::shared_ptr<const Var>> knots_;
  Eigen::VectorXd basis_;
  Eigen::VectorXd offset_;
  std::shared_ptr<NodesVariables> sample_variables_;
  std::shared_ptr<DiscreteCollisionConstraintD> native_;
  bool updated_{ false };
};
}  // namespace trajopt_ifopt
#endif
