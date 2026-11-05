#pragma once

#include <trajopt_ifopt/constraints/cartesian_line_constraint.h>
#include <Eigen/Core>
#include <memory>
#include <string>
#include <vector>

namespace trajopt_ifopt
{
class NodesVariables;
class Var;

/** Apply native Cartesian line residuals to one actual affine spline sample.
 * The line is defined in the target-link frame by CartLineInfo. Ordering,
 * process endpoints and the curve between sample times require separate checks.
 */
class SplineCartesianLineConstraint : public ConstraintSet
{
public:
  SplineCartesianLineConstraint(CartLineInfo info,
                                std::vector<std::shared_ptr<const Var>> knots,
                                Eigen::VectorXd basis,
                                Eigen::VectorXd offset,
                                Eigen::VectorXd coefficients,
                                std::string name = "SplineCartesianLine");
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
  std::shared_ptr<CartLineConstraint> native_;
};
}  // namespace trajopt_ifopt
