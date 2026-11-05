#pragma once

#include <trajopt_ifopt/core/constraint_set.h>
#include <trajopt_ifopt/spline/quintic_spline_model.h>
#include <memory>
#include <string>
#include <vector>

namespace trajopt_ifopt
{
class Var;

/** Signed jerk epigraph at fixed clock witnesses. The scale is T_ref^2 / sourced axis speed. */
class QuinticSplineJerkPeakConstraint : public ConstraintSet
{
public:
  QuinticSplineJerkPeakConstraint(const QuinticSplineModel& model,
                                  std::vector<std::shared_ptr<const Var>> knots,
                                  std::shared_ptr<const Var> peak,
                                  std::vector<Seconds> witnesses,
                                  Eigen::VectorXd normalization,
                                  std::string name = "SplineJerkPeak");
  Eigen::VectorXd getValues() const override;
  std::vector<Bounds> getBounds() const override;
  Jacobian getJacobian() const override;
  Eigen::VectorXd getCoefficients() const override;
  int update() override;

protected:
  void initVariableDependedQuantities(const Variables::Ptr& variables) override;

private:
  std::vector<std::shared_ptr<const Var>> knots_;
  std::shared_ptr<const Var> peak_;
  SplineAffineMap map_;
  Eigen::VectorXd normalization_;
};
}  // namespace trajopt_ifopt
