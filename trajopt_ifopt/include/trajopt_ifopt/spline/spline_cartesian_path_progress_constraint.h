#pragma once
#include <trajopt_ifopt/constraints/cartesian_line_constraint.h>
#include <Eigen/Core>
#include <memory>
#include <string>
#include <vector>

namespace trajopt_ifopt
{
class Var;
/** Sample ordered local-tangent progress along a generated workpiece-frame path.
 * Bound the first/last sample only when the process segment explicitly includes that endpoint.
 * Pose, collision and behavior between sample times remain separate constraints/checks.
 */
class SplineCartesianPathProgressConstraint : public ConstraintSet
{
public:
  SplineCartesianPathProgressConstraint(CartLineInfo info,
                                    std::vector<std::shared_ptr<const Var>> knots,
                                    Eigen::MatrixXd basis,
                                    Eigen::MatrixXd offset,
                                    Eigen::MatrixXd path_points,
                                    Eigen::MatrixXd path_tangents,
                                    Eigen::VectorXd arc_lengths,
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
  Eigen::MatrixXd path_points_;
  Eigen::MatrixXd path_tangents_;
  Eigen::VectorXd arc_lengths_;
  double total_length_;
  bool include_start_;
  bool include_end_;
};
}  // namespace trajopt_ifopt
