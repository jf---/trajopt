#include <trajopt_ifopt/spline/quintic_spline_model.h>
#include <trajopt_ifopt/spline/quintic_spline_jerk_cost.h>
#include <trajopt_ifopt/spline/quintic_spline_jerk_fourth_cost.h>
#include <trajopt_ifopt/spline/quintic_spline_jerk_peak_constraint.h>
#include <trajopt_ifopt/spline/quintic_spline_position_limits.h>
#include <trajopt_ifopt/variable_sets/node.h>
#include <trajopt_ifopt/variable_sets/nodes_variables.h>
#include <trajopt_ifopt/variable_sets/var.h>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <vector>

using namespace trajopt_ifopt;

TEST(QuinticSplineFreeAcceleration, AllConsumersIncludeBothAccelerationColumns)
{
  const std::vector<Seconds> clock = { Seconds(0),   Seconds(.11), Seconds(.27), Seconds(.46),
                                       Seconds(.69), Seconds(.86), Seconds(1) };
  const auto model = QuinticSplineModel::buildVariableAcceleration(
      clock, Eigen::Vector2d(.2, -.3), std::nullopt, Eigen::Vector2d(-.4, .6), std::nullopt);
  std::vector<std::unique_ptr<Node>> nodes;
  std::vector<std::shared_ptr<const Var>> independent;
  for (int i = 0; i < 9; ++i)
  {
    nodes.push_back(std::make_unique<Node>("independent_" + std::to_string(i)));
    independent.push_back(nodes.back()->addVar(
        "value", { "track", "arm" }, Eigen::Vector2d(.1 * i, -.07 * i), std::vector<Bounds>(2, NoBound)));
  }
  nodes.push_back(std::make_unique<Node>("peak"));
  auto peak =
      nodes.back()->addVar("peak", { "track", "arm" }, Eigen::Vector2d(2.0, 3.0), std::vector<Bounds>(2, NoBound));
  auto variables = std::make_shared<NodesVariables>("all", std::move(nodes));
  const std::vector<Seconds> samples = { Seconds(.05), Seconds(.35), Seconds(.91) };
  QuinticSplineJerkCost jerk(model, independent, Eigen::Vector2d::Ones());
  QuinticSplineJerkFourthCost fourth(model, independent, Eigen::Vector2d::Ones());
  QuinticSplineJerkPeakConstraint peak_constraint(model, independent, peak, samples, Eigen::Vector2d(2.0, .5));
  QuinticSplinePositionLimits limits(model, independent, samples, std::vector<Bounds>(2, Bounds(-10, 10)));
  for (ConstraintSet* constraint : { static_cast<ConstraintSet*>(&jerk),
                                     static_cast<ConstraintSet*>(&fourth),
                                     static_cast<ConstraintSet*>(&peak_constraint),
                                     static_cast<ConstraintSet*>(&limits) })
  {
    constraint->linkWithVariables(variables);
    const Eigen::MatrixXd analytic = constraint->getJacobian();
    const Eigen::VectorXd initial = variables->getValues();
    constexpr double STEP = 1e-6;
    for (Eigen::Index column = 14; column < 18; ++column)
    {
      Eigen::VectorXd x = initial;
      x[column] += STEP;
      variables->setVariables(x);
      const Eigen::VectorXd plus = constraint->getValues();
      x[column] -= 2 * STEP;
      variables->setVariables(x);
      const Eigen::VectorXd minus = constraint->getValues();
      EXPECT_TRUE(((plus - minus) / (2 * STEP)).isApprox(analytic.col(column), 1e-6));
    }
    variables->setVariables(initial);
  }
}
