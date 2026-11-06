#pragma once

#include <trajopt_ifopt/core/constraint_set.h>
#include <trajopt_ifopt/spline/quintic_spline_model.h>
#include <memory>
#include <string>
#include <vector>

namespace trajopt_ifopt
{
class Var;

/** Velocity bounds at explicit actual-spline sample times; not an interval certificate. */
class QuinticSplineDerivativeLimits : public ConstraintSet
{
public:
  QuinticSplineDerivativeLimits(const QuinticSplineModel& model,
                                std::vector<std::shared_ptr<const Var>> knots,
                                std::vector<Seconds> samples,
                                std::vector<Bounds> sample_axis_limits,
                                std::string name = "SplineInterpolatedVelocityLimits");
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
  std::vector<Eigen::Index> row_samples_;
  std::vector<Eigen::Index> row_axes_;
};
}  // namespace trajopt_ifopt
