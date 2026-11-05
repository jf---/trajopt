#include <trajopt_ifopt/spline/quintic_spline_jerk_fourth_cost.h>
#include <trajopt_ifopt/variable_sets/node.h>
#include <trajopt_ifopt/variable_sets/nodes_variables.h>
#include <trajopt_ifopt/variable_sets/var.h>
#include <gtest/gtest.h>
#include <cmath>
#include <memory>
#include <vector>

using namespace trajopt_ifopt;

TEST(QuinticSplineJerkFourthCost, IntegratesPolynomialAndMatchesJacobian)
{
  const std::vector<Seconds> clock = { Seconds(0),   Seconds(.11), Seconds(.27), Seconds(.46),
                                       Seconds(.69), Seconds(.86), Seconds(1) };
  const Eigen::Vector2d v0(0, 0), a0(0, 0), v1(5, 6), a1(20, 12);
  const auto model = QuinticSplineModel::build(clock, v0, a0, v1, a1);
  std::vector<std::unique_ptr<Node>> nodes;
  std::vector<std::shared_ptr<const Var>> vars;
  for (std::size_t i = 0; i < clock.size(); ++i)
  {
    const double t = clock[i].count();
    nodes.push_back(std::make_unique<Node>("knot_" + std::to_string(i)));
    nodes.back()->addVar("auxiliary", { "auxiliary" }, Eigen::VectorXd::Zero(1), { NoBound });
    vars.push_back(nodes.back()->addVar("position",
                                        { "track", "arm" },
                                        Eigen::Vector2d(std::pow(t, 5), 2 * std::pow(t, 3)),
                                        std::vector<Bounds>(2, NoBound)));
  }
  auto variables = std::make_shared<NodesVariables>("knots", std::move(nodes));
  QuinticSplineJerkFourthCost cost(model, vars, Eigen::Vector2d(2.0, 3.0));
  cost.linkWithVariables(variables);
  const auto values = cost.getValues();
  const auto weights = cost.getCoefficients();
  ASSERT_EQ(cost.getRows(), 60);
  double objective = 0;
  for (Eigen::Index i = 0; i < values.size(); ++i)
    objective += weights[i] * values[i] * values[i];
  const double expected = 2.0 * std::pow(60.0, 4) / 9.0 + 3.0 * std::pow(12.0, 4);
  constexpr double kObjectiveRelativeTolerance = 1e-9;
  EXPECT_NEAR(objective, expected, expected * kObjectiveRelativeTolerance);
  const Eigen::MatrixXd analytic = cost.getJacobian();
  const Eigen::VectorXd initial = variables->getValues();
  constexpr double kStep = 1e-6;
  constexpr double kJacobianRelativeTolerance = 1e-5;
  for (Eigen::Index i = 0; i < initial.size(); ++i)
  {
    Eigen::VectorXd x = initial;
    x[i] += kStep;
    variables->setVariables(x);
    const Eigen::VectorXd plus = cost.getValues();
    x[i] -= 2 * kStep;
    variables->setVariables(x);
    const Eigen::VectorXd minus = cost.getValues();
    EXPECT_TRUE(((plus - minus) / (2 * kStep)).isApprox(analytic.col(i), kJacobianRelativeTolerance));
  }
  variables->setVariables(initial);
}
