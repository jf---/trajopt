#pragma once

#include <trajopt_ifopt/core/constraint_set.h>
#include <trajopt_ifopt/spline/quintic_spline_model.h>
#include <memory>
#include <string>
#include <vector>

namespace trajopt_ifopt
{
class Var;

/** Fixed-clock analytic jerk residual; use with TrajOpt's squared-cost penalty. */
class QuinticSplineJerkCost : public ConstraintSet
{
public:
  QuinticSplineJerkCost(const QuinticSplineModel& model,
                        std::vector<std::shared_ptr<const Var>> knots,
                        Eigen::VectorXd axis_weights,
                        std::string name = "ContinuousSplineJerk");
  Eigen::VectorXd getValues() const override;
  std::vector<Bounds> getBounds() const override;
  Jacobian getJacobian() const override;
  Eigen::VectorXd getCoefficients() const override;
  int update() override;

protected:
  void initVariableDependedQuantities(const Variables::Ptr& variables) override;

private:
  std::vector<std::shared_ptr<const Var>> knots_;
  SplineAffineMap map_;
  Eigen::VectorXd coefficients_;
};
}  // namespace trajopt_ifopt
