#include <trajopt_ifopt/spline/quintic_spline_jerk_peak_constraint.h>
#include <trajopt_ifopt/variable_sets/node.h>
#include <trajopt_ifopt/variable_sets/nodes_variables.h>
#include <trajopt_ifopt/variable_sets/var.h>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

using namespace trajopt_ifopt;

TEST(QuinticSplineJerkPeakConstraint, BothSignsUnitsAndAnalyticJacobian)
{
  const std::vector<Seconds> clock = { Seconds(0),   Seconds(.11), Seconds(.27), Seconds(.46),
                                       Seconds(.69), Seconds(.86), Seconds(1) };
  const auto model = QuinticSplineModel::build(
      clock, Eigen::Vector2d(.2, -.3), Eigen::Vector2d(.1, .2), Eigen::Vector2d(-.4, .6), Eigen::Vector2d(.3, -.1));
  std::vector<std::unique_ptr<Node>> nodes;
  std::vector<std::shared_ptr<const Var>> positions;
  for (std::size_t i = 0; i < clock.size(); ++i)
  {
    nodes.push_back(std::make_unique<Node>("knot_" + std::to_string(i)));
    positions.push_back(nodes.back()->addVar(
        "position", { "track", "arm" }, Eigen::Vector2d(.1 * i, -.07 * i), std::vector<Bounds>(2, NoBound)));
  }
  nodes.push_back(std::make_unique<Node>("peak"));
  auto peak =
      nodes.back()->addVar("peak", { "track", "arm" }, Eigen::Vector2d(2.0, 3.0), std::vector<Bounds>(2, NoBound));
  auto variables = std::make_shared<NodesVariables>("knots", std::move(nodes));
  const std::vector<Seconds> witnesses = { Seconds(.05), Seconds(.35), Seconds(.91) };
  const Eigen::Vector2d normalization(2.0, .5);
  QuinticSplineJerkPeakConstraint cost(model, positions, peak, witnesses, normalization);
  cost.linkWithVariables(variables);
  EXPECT_THROW(QuinticSplineJerkPeakConstraint(model, positions, peak, witnesses, Eigen::Vector2d(0.0, 1.0)),
               QuinticSplineError);
  EXPECT_THROW(QuinticSplineJerkPeakConstraint(model, positions, peak, { Seconds(-.01) }, normalization),
               QuinticSplineError);
  Eigen::MatrixXd q(7, 2);
  for (int i = 0; i < 7; ++i)
    q.row(i) = positions[static_cast<std::size_t>(i)]->value().transpose();
  const auto map = model.map(witnesses, 3);
  const Eigen::MatrixXd jerk = map.basis * q + map.offset;
  const auto values = cost.getValues();
  ASSERT_EQ(values.size(), 12);
  constexpr double kValueBand = 1e-10;
  for (int t = 0; t < 3; ++t)
    for (int axis = 0; axis < 2; ++axis)
    {
      const int row = 2 * (t * 2 + axis);
      EXPECT_NEAR(values[row], normalization[axis] * jerk(t, axis) - peak->value()[axis], kValueBand);
      EXPECT_NEAR(values[row + 1], -normalization[axis] * jerk(t, axis) - peak->value()[axis], kValueBand);
      EXPECT_DOUBLE_EQ(cost.getBounds()[static_cast<std::size_t>(row)].getUpper(), 0.0);
      EXPECT_DOUBLE_EQ(cost.getBounds()[static_cast<std::size_t>(row + 1)].getUpper(), 0.0);
    }
  const Eigen::MatrixXd analytic = cost.getJacobian();
  const Eigen::VectorXd initial = variables->getValues();
  constexpr double kStep = 1e-6;
  constexpr double kJacobianRelativeBand = 1e-6;
  for (Eigen::Index i = 0; i < initial.size(); ++i)
  {
    Eigen::VectorXd x = initial;
    x[i] += kStep;
    variables->setVariables(x);
    const Eigen::VectorXd plus = cost.getValues();
    x[i] -= 2 * kStep;
    variables->setVariables(x);
    const Eigen::VectorXd minus = cost.getValues();
    EXPECT_TRUE(((plus - minus) / (2 * kStep)).isApprox(analytic.col(i), kJacobianRelativeBand));
  }
  variables->setVariables(initial);
}
