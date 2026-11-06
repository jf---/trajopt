#include <trajopt_ifopt/spline/affine_spline_constraint.h>
#include <trajopt_ifopt/variable_sets/node.h>
#include <trajopt_ifopt/variable_sets/nodes_variables.h>
#include <trajopt_ifopt/variable_sets/var.h>
#include <gtest/gtest.h>

using namespace trajopt_ifopt;

TEST(AffineSplineConstraint, ValuesAndJacobiansUseEverySharedVariable)
{
  std::vector<std::unique_ptr<Node>> nodes;
  std::vector<std::shared_ptr<const Var>> independent;
  for (int i = 0; i < 3; ++i)
  {
    nodes.push_back(std::make_unique<Node>("node_" + std::to_string(i)));
    independent.push_back(nodes.back()->addVar(
        "value", { "track", "arm" }, Eigen::Vector2d(.2 * i, -.1 * i), std::vector<Bounds>(2, NoBound)));
  }
  auto variables = std::make_shared<NodesVariables>("all", std::move(nodes));
  Eigen::MatrixXd basis(2, 3);
  basis << 1, 2, -1, .5, 0, 2;
  Eigen::MatrixXd offset(2, 2);
  offset << .1, -.2, .3, -.4;
  const std::vector<Bounds> bounds(4, BoundZero);
  AffineSplineConstraint constraint(independent, basis, offset, bounds, Eigen::Vector4d::Ones(), "affine");
  EXPECT_THROW(
      AffineSplineConstraint(independent, basis, offset, std::vector<Bounds>(), Eigen::Vector4d::Ones(), "invalid"),
      AffineSplineConstraintError);
  constraint.linkWithVariables(variables);
  Eigen::MatrixXd values(3, 2);
  for (int i = 0; i < 3; ++i)
    values.row(i) = independent[static_cast<std::size_t>(i)]->value().transpose();
  const Eigen::MatrixXd expected = basis * values + offset;
  EXPECT_NEAR(constraint.getValues()[0], expected(0, 0), 1e-12);
  EXPECT_NEAR(constraint.getValues()[3], expected(1, 1), 1e-12);
  const Eigen::MatrixXd analytic = constraint.getJacobian();
  const Eigen::VectorXd initial = variables->getValues();
  constexpr double STEP = 1e-6;
  for (Eigen::Index column = 0; column < initial.size(); ++column)
  {
    Eigen::VectorXd x = initial;
    x[column] += STEP;
    variables->setVariables(x);
    const Eigen::VectorXd plus = constraint.getValues();
    x[column] -= 2 * STEP;
    variables->setVariables(x);
    const Eigen::VectorXd minus = constraint.getValues();
    EXPECT_TRUE(((plus - minus) / (2 * STEP)).isApprox(analytic.col(column), 1e-6));
  }
}
