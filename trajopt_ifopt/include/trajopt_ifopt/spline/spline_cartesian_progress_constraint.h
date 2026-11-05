#pragma once
#include <trajopt_ifopt/constraints/cartesian_line_constraint.h>
#include <Eigen/Core>
#include <memory>
#include <string>
#include <vector>

namespace trajopt_ifopt
{
class Var;
/** Sample unclamped line progress and ordered progress differences on an affine spline.
 * Bound the first/last sample only when the process segment explicitly includes that endpoint.
 * Pose, collision and behavior between sample times remain separate constraints/checks.
 */
class SplineCartesianProgressConstraint : public ConstraintSet
{
public:
  SplineCartesianProgressConstraint(CartLineInfo info,
                                    std::vector<std::shared_ptr<const Var>> knots,
                                    Eigen::MatrixXd basis,
                                    Eigen::MatrixXd offset,
                                    bool include_start,
                                    bool include_end,
                                    std::string name = "SplineCartesianProgress");
  int update() override { return rows_; }
  Eigen::VectorXd getValues() const override;
  std::vector<Bounds> getBounds() const override;
  Eigen::VectorXd getCoefficients() const override;
  Jacobian getJacobian() const override;

private:
  void initVariableDependedQuantities(const Variables::Ptr& variables) override;
  void evaluate(Eigen::VectorXd& progress, Eigen::MatrixXd* gradients) const;
  CartLineInfo info_;
  std::vector<std::shared_ptr<const Var>> knots_;
  Eigen::MatrixXd basis_;
  Eigen::MatrixXd offset_;
  Eigen::Vector3d line_;
  double line_squared_;
  bool include_start_;
  bool include_end_;
};
}  // namespace trajopt_ifopt
