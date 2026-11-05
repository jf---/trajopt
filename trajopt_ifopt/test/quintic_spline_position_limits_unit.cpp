#include <trajopt_ifopt/spline/quintic_spline_position_limits.h>
#include <trajopt_ifopt/variable_sets/node.h>
#include <trajopt_ifopt/variable_sets/nodes_variables.h>
#include <trajopt_ifopt/variable_sets/var.h>
#include <gtest/gtest.h>

#include <memory>
#include <vector>

using namespace trajopt_ifopt;

TEST(QuinticSplinePositionLimits, MatchesAffineMapBoundsAndFiniteDifferences)
{
  const std::vector<Seconds> clock = { Seconds(0),   Seconds(.2),  Seconds(.65), Seconds(1.1),
                                       Seconds(1.7), Seconds(2.3), Seconds(3) };
  const std::vector<Seconds> samples = { Seconds(.1), Seconds(.74), Seconds(1.9), Seconds(2.9) };
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
  QuinticSplinePositionLimits limits(model, vars, samples, { Bounds(-1, 2), Bounds(-3, 4) });
  limits.linkWithVariables(variables);
  const auto map = model.map(samples, 0);
  Eigen::MatrixXd positions(7, 2);
  for (int i = 0; i < 7; ++i)
    positions.row(i) = vars[static_cast<std::size_t>(i)]->value().transpose();
  const Eigen::MatrixXd expected = map.basis * positions + map.offset;
  const Eigen::VectorXd value = limits.getValues();
  const auto output_bounds = limits.getBounds();
  EXPECT_EQ(limits.getRows(), 16);
  for (int sample = 0; sample < 4; ++sample)
    for (int axis = 0; axis < 2; ++axis)
    {
      const int row = sample * 2 + axis;
      EXPECT_NEAR(value[row], expected(sample, axis), 1e-12);
      EXPECT_NEAR(value[row + 8], expected(sample, axis), 1e-12);
      EXPECT_DOUBLE_EQ(output_bounds[static_cast<std::size_t>(row)].getLower(), axis == 0 ? -1 : -3);
      EXPECT_DOUBLE_EQ(output_bounds[static_cast<std::size_t>(row + 8)].getUpper(), axis == 0 ? 2 : 4);
    }
  const Eigen::MatrixXd analytic = limits.getJacobian();
  const Eigen::VectorXd initial = variables->getValues();
  const double step = 1e-6;
  for (Eigen::Index i = 0; i < initial.size(); ++i)
  {
    Eigen::VectorXd x = initial;
    x[i] += step;
    variables->setVariables(x);
    const Eigen::VectorXd plus = limits.getValues();
    x[i] -= 2 * step;
    variables->setVariables(x);
    const Eigen::VectorXd minus = limits.getValues();
    EXPECT_TRUE(((plus - minus) / (2 * step)).isApprox(analytic.col(i), 1e-7));
  }
  variables->setVariables(initial);
}
