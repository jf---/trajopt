#include <trajopt_ifopt/spline/quintic_spline_derivative_limits.h>
#include <trajopt_ifopt/variable_sets/node.h>
#include <trajopt_ifopt/variable_sets/nodes_variables.h>
#include <trajopt_ifopt/variable_sets/var.h>
#include <gtest/gtest.h>
#include <limits>
#include <cmath>
#include <memory>
#include <vector>

using namespace trajopt_ifopt;

TEST(QuinticSplineDerivativeLimits, ValuesVariableBoundsAndAnalyticJacobian)
{
  const std::vector<Seconds> times = { Seconds(0),   Seconds(.2),  Seconds(.65), Seconds(1.1),
                                       Seconds(1.7), Seconds(2.3), Seconds(3) };
  const std::vector<Seconds> samples = { Seconds(.1), Seconds(.74), Seconds(1.9), Seconds(2.9) };
  Eigen::Vector2d v0(.35, -.4), a0(-.18, .12), v1(-.22, .32), a1(.27, -.14);
  auto model = QuinticSplineModel::build(times, v0, a0, v1, a1);
  std::vector<std::unique_ptr<Node>> nodes;
  std::vector<std::shared_ptr<const Var>> vars;
  for (int i = 0; i < 7; ++i)
  {
    nodes.push_back(std::make_unique<Node>("knot_" + std::to_string(i)));
    nodes.back()->addVar("auxiliary", { "auxiliary" }, Eigen::VectorXd::Zero(1), { NoBound });
    vars.push_back(nodes.back()->addVar(
        "position", { "track", "fixture" }, Eigen::Vector2d(i * .12, i * -.09), { NoBound, NoBound }));
  }
  auto variables = std::make_shared<NodesVariables>("knots", std::move(nodes));
  const double inf = std::numeric_limits<double>::infinity();
  std::vector<Bounds> limits;
  for (int i = 0; i < 4; ++i)
  {
    limits.emplace_back(-1.8, 1.8);
    limits.emplace_back(i < 2 ? 0.0 : -inf, i < 2 ? inf : 0.0);
  }
  QuinticSplineDerivativeLimits term(model, vars, samples, limits);
  term.linkWithVariables(variables);
  Eigen::MatrixXd positions(7, 2);
  for (int i = 0; i < 7; ++i)
    positions.row(i) = vars[static_cast<std::size_t>(i)]->value().transpose();
  const auto map = model.map(samples, 1);
  const Eigen::MatrixXd expected = map.basis * positions + map.offset;
  const auto output = term.getValues();
  const auto bounds = term.getBounds();
  ASSERT_EQ(term.getRows(), 12);
  int row = 0;
  for (int sample = 0; sample < 4; ++sample)
    for (int axis = 0; axis < 2; ++axis)
    {
      const auto& bound = limits[static_cast<std::size_t>(sample * 2 + axis)];
      if (std::isfinite(bound.getLower()))
      {
        EXPECT_NEAR(output[row], expected(sample, axis), 1e-12);
        EXPECT_DOUBLE_EQ(bounds[static_cast<std::size_t>(row++)].getLower(), bound.getLower());
      }
    }
  for (int sample = 0; sample < 4; ++sample)
    for (int axis = 0; axis < 2; ++axis)
    {
      const auto& bound = limits[static_cast<std::size_t>(sample * 2 + axis)];
      if (std::isfinite(bound.getUpper()))
      {
        EXPECT_NEAR(output[row], expected(sample, axis), 1e-12);
        EXPECT_DOUBLE_EQ(bounds[static_cast<std::size_t>(row++)].getUpper(), bound.getUpper());
      }
    }
  const Eigen::MatrixXd jacobian = term.getJacobian();
  const auto initial = variables->getValues();
  for (Eigen::Index i = 0; i < initial.size(); ++i)
  {
    Eigen::VectorXd x = initial;
    x[i] += 1e-6;
    variables->setVariables(x);
    const auto plus = term.getValues();
    x[i] -= 2e-6;
    variables->setVariables(x);
    const auto minus = term.getValues();
    EXPECT_TRUE(((plus - minus) / 2e-6).isApprox(jacobian.col(i), 1e-7));
  }
  variables->setVariables(initial);
}
