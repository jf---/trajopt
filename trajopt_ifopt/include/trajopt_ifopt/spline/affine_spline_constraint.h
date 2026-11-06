#pragma once

#include <trajopt_ifopt/core/constraint_set.h>
#include <trajopt_common/macros.h>
#include <Eigen/Core>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace trajopt_ifopt
{
class Var;
class TRAJOPT_API AffineSplineConstraintError : public std::runtime_error
{
public:
  using std::runtime_error::runtime_error;
  ~AffineSplineConstraintError() override;
};

/** Native Ifopt rows for an affine map of stage-local spline variables.
 * Rows are ordered sample then axis. Units follow the mapped quantity.
 */
class TRAJOPT_API AffineSplineConstraint : public ConstraintSet
{
public:
  AffineSplineConstraint(std::vector<std::shared_ptr<const Var>> variables,
                         Eigen::MatrixXd basis,
                         Eigen::MatrixXd offset,
                         std::vector<Bounds> bounds,
                         Eigen::VectorXd coefficients,
                         std::string name = "AffineSpline");
  Eigen::VectorXd getValues() const override;
  std::vector<Bounds> getBounds() const override;
  Eigen::VectorXd getCoefficients() const override;
  Jacobian getJacobian() const override;
  int update() override;

protected:
  void initVariableDependedQuantities(const Variables::Ptr& variables) override;

private:
  std::vector<std::shared_ptr<const Var>> independent_;
  Eigen::MatrixXd basis_;
  Eigen::MatrixXd offset_;
  std::vector<Bounds> bounds_;
  Eigen::VectorXd coefficients_;
};
}  // namespace trajopt_ifopt
