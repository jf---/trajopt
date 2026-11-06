#pragma once

#include <trajopt_ifopt/spline/spline_cartesian_progress_constraint.h>
#include <stdexcept>
#include <utility>

namespace trajopt_ifopt
{
class SplineTimedProgressError : public std::invalid_argument
{
public:
  using std::invalid_argument::invalid_argument;
};

/** Residual of actual Cartesian line progress against an explicit modeled clock. */
class SplineTimedProgress : public ConstraintSet
{
public:
  SplineTimedProgress(CartLineInfo info,
                      std::vector<std::shared_ptr<const Var>> knots,
                      Eigen::MatrixXd basis,
                      Eigen::MatrixXd offset,
                      Eigen::VectorXd targets,
                      Eigen::VectorXd coefficients,
                      std::string name)
    : ConstraintSet(std::move(name), static_cast<int>(basis.rows()))
    , progress_(std::move(info), std::move(knots), std::move(basis), std::move(offset), false, false, "progress_source")
    , targets_(std::move(targets))
    , coefficients_(std::move(coefficients))
  {
    if (targets_.size() != rows_ || coefficients_.size() != rows_ || !targets_.allFinite() ||
        !coefficients_.allFinite() || (coefficients_.array() < 0).any())
      throw SplineTimedProgressError("invalid timed progress targets or coefficients");
  }
  int update() override { return rows_; }
  Eigen::VectorXd getValues() const override { return progress_.getValues().head(rows_) - targets_; }
  std::vector<Bounds> getBounds() const override { return std::vector<Bounds>(static_cast<std::size_t>(rows_), BoundZero); }
  Eigen::VectorXd getCoefficients() const override { return coefficients_; }
  Jacobian getJacobian() const override { return progress_.getJacobian().topRows(rows_); }

private:
  void initVariableDependedQuantities(const Variables::Ptr& variables) override { progress_.linkWithVariables(variables); }
  SplineCartesianProgressConstraint progress_;
  Eigen::VectorXd targets_;
  Eigen::VectorXd coefficients_;
};
}  // namespace trajopt_ifopt
