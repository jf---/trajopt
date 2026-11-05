#include <trajopt_ifopt/spline/quintic_spline_jerk_cost.h>
#include <trajopt_ifopt/variable_sets/node.h>
#include <trajopt_ifopt/variable_sets/nodes_variables.h>
#include <trajopt_ifopt/variable_sets/var.h>
#include <gtest/gtest.h>

#include <memory>
#include <vector>

using namespace trajopt_ifopt;

TEST(QuinticSplineJerkCost, MatchesModelAndFiniteDifferenceJacobians)
{
  const std::vector<Seconds> clock = { Seconds(0),   Seconds(.2),  Seconds(.65), Seconds(1.1),
                                       Seconds(1.7), Seconds(2.3), Seconds(3) };
  Eigen::Vector2d v0(.35, -.4), a0(-.18, .12), v1(-.22, .32), a1(.27, -.14);
  auto model = QuinticSplineModel::build(clock, v0, a0, v1, a1);
  std::vector<std::unique_ptr<Node>> nodes;
  std::vector<std::shared_ptr<const Var>> vars;
  const std::vector<Bounds> bounds(2, NoBound);
  for (int i = 0; i < 7; ++i)
  {
    nodes.push_back(std::make_unique<Node>("knot_" + std::to_string(i)));
    nodes.back()->addVar("auxiliary", { "auxiliary" }, Eigen::VectorXd::Zero(1), { NoBound });
    vars.push_back(nodes.back()->addVar("position", { "track", "arm" }, Eigen::Vector2d(i * .12, i * -.09), bounds));
  }
  auto variables = std::make_shared<NodesVariables>("knots", std::move(nodes));
  QuinticSplineJerkCost cost(model, vars, Eigen::Vector2d(2.0, 3.0));
  cost.linkWithVariables(variables);
  const auto quadrature = model.jerkQuadrature();
  Eigen::MatrixXd knot_values(7, 2);
  for (int i = 0; i < 7; ++i)
    knot_values.row(i) = vars[static_cast<std::size_t>(i)]->value().transpose();
  const Eigen::MatrixXd expected = quadrature.map.basis * knot_values + quadrature.map.offset;
  const Eigen::VectorXd values = cost.getValues();
  const Eigen::VectorXd coefficients = cost.getCoefficients();
  EXPECT_EQ(cost.getRows(), 36);
  for (int sample = 0; sample < 18; ++sample)
    for (int axis = 0; axis < 2; ++axis)
    {
      EXPECT_NEAR(values[sample * 2 + axis], expected(sample, axis), 1e-10);
      EXPECT_NEAR(coefficients[sample * 2 + axis], quadrature.weights[sample] * (axis == 0 ? 2.0 : 3.0), 1e-14);
    }
  const Eigen::MatrixXd analytic = cost.getJacobian();
  const Eigen::VectorXd initial = variables->getValues();
  const double step = 1e-6;
  for (Eigen::Index i = 0; i < initial.size(); ++i)
  {
    Eigen::VectorXd x = initial;
    x[i] += step;
    variables->setVariables(x);
    const Eigen::VectorXd plus = cost.getValues();
    x[i] -= 2 * step;
    variables->setVariables(x);
    const Eigen::VectorXd minus = cost.getValues();
    EXPECT_TRUE(((plus - minus) / (2 * step)).isApprox(analytic.col(i), 1e-7));
  }
  variables->setVariables(initial);
}
