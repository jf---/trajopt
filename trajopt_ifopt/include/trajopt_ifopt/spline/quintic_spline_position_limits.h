#pragma once

#include <trajopt_ifopt/core/constraint_set.h>
#include <trajopt_ifopt/spline/quintic_spline_model.h>
#include <memory>
#include <string>
#include <vector>

namespace trajopt_ifopt
{
class Var;

/** Joint bounds on finitely many actual spline samples; no between-sample certificate. */
class QuinticSplinePositionLimits : public ConstraintSet
{
public:
  QuinticSplinePositionLimits(const QuinticSplineModel& model,
                              std::vector<std::shared_ptr<const Var>> knots,
                              std::vector<Seconds> samples,
                              std::vector<Bounds> limits,
                              std::string name = "SplineInterpolatedJointLimits");
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
  std::vector<Bounds> bounds_;
};
}  // namespace trajopt_ifopt
