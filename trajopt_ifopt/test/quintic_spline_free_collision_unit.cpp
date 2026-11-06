/** Free endpoint acceleration changes the same native collision sample and Jacobian. */
#include <trajopt_common/macros.h>
TRAJOPT_IGNORE_WARNINGS_PUSH
#include <gtest/gtest.h>
#include <Eigen/Core>
#include <tesseract/collision/types.h>
#include <tesseract/common/resource_locator.h>
#include <tesseract/environment/commands/add_link_command.h>
#include <tesseract/environment/commands/change_link_collision_enabled_command.h>
#include <tesseract/environment/environment.h>
#include <tesseract/geometry/geometries.h>
#include <tesseract/kinematics/joint_group.h>
#include <tesseract/scene_graph/link.h>
TRAJOPT_IGNORE_WARNINGS_POP

#include <trajopt_ifopt/constraints/collision/discrete_collision_constraint.h>
#include <trajopt_ifopt/constraints/collision/discrete_collision_evaluators.h>
#include <trajopt_ifopt/constraints/collision/dynamic_spline_collision_constraint.h>
#include <trajopt_ifopt/spline/quintic_spline_model.h>
#include <trajopt_ifopt/variable_sets/node.h>
#include <trajopt_ifopt/variable_sets/nodes_variables.h>
#include <trajopt_ifopt/variable_sets/var.h>
#include <filesystem>
#include <memory>
#include <limits>
#include <optional>
#include <vector>

using namespace trajopt_ifopt;

TEST(QuinticSplineFreeAcceleration, NativeCollisionSampleAndJacobianUseFreeVariables)
{
  using namespace tesseract::environment;
  using namespace tesseract::scene_graph;
  auto env = std::make_shared<Environment>();
  auto locator = std::make_shared<tesseract::common::GeneralResourceLocator>();
  ASSERT_TRUE(env->init(std::filesystem::path(std::string(TRAJOPT_DATA_DIR) + "/spherebot.urdf"),
                        std::filesystem::path(std::string(TRAJOPT_DATA_DIR) + "/spherebot.srdf"),
                        locator));
  for (const char* name : { "test_sphere_link", "test_sphere_link2", "test_sphere_link3" })
    ASSERT_TRUE(env->applyCommand(std::make_shared<ChangeLinkCollisionEnabledCommand>(name, false)));
  Link obstacle("free_acceleration_obstacle");
  auto shape = std::make_shared<Collision>();
  shape->origin = Eigen::Isometry3d::Identity();
  shape->origin.translation().x() = 0.77;
  shape->geometry = std::make_shared<tesseract::geometry::Sphere>(0.2);
  obstacle.collision.push_back(std::move(shape));
  ASSERT_TRUE(env->applyCommand(std::make_shared<AddLinkCommand>(obstacle)));
  const auto manip = env->getJointGroup("manipulator");
  trajopt_common::TrajOptCollisionConfig config(
      0.0, 1.0, tesseract::collision::ContactRequest(tesseract::collision::ContactTestType::ALL));
  config.collision_margin_buffer = 0.1;
  auto evaluator = std::make_shared<SingleTimestepCollisionEvaluator>(manip, env, config, false);
  const std::vector<std::string> names = tesseract::common::toNames(manip->getJointIds());
  const std::vector<Seconds> clock = { Seconds(0),   Seconds(.11), Seconds(.27), Seconds(.46),
                                       Seconds(.69), Seconds(.86), Seconds(1) };
  const auto model = QuinticSplineModel::buildVariableAcceleration(
      clock, Eigen::Vector2d::Zero(), std::nullopt, Eigen::Vector2d::Zero(), std::nullopt);
  const auto map = model.map({ Seconds(.35) }, 0);
  std::vector<std::unique_ptr<Node>> nodes;
  std::vector<std::shared_ptr<const Var>> independent;
  for (int i = 0; i < 9; ++i)
  {
    nodes.push_back(std::make_unique<Node>("independent_" + std::to_string(i)));
    Eigen::Vector2d value = Eigen::Vector2d::Zero();
    if (i == 7)
      value << 0.4, -0.2;
    if (i == 8)
      value << -0.3, 0.1;
    independent.push_back(nodes.back()->addVar("value", names, value, std::vector<Bounds>(2, NoBound)));
  }
  auto variables = std::make_shared<NodesVariables>("free_spline", std::move(nodes));
  DynamicSplineCollisionConstraint spline(
      evaluator, independent, map.basis.row(0).transpose(), map.offset.row(0).transpose());
  spline.linkWithVariables(variables);
  ASSERT_GT(spline.update(), 0);
  Eigen::MatrixXd independent_values(9, 2);
  for (int i = 0; i < 9; ++i)
    independent_values.row(i) = independent[static_cast<std::size_t>(i)]->value().transpose();
  const Eigen::VectorXd sample = (map.basis * independent_values + map.offset).row(0).transpose();
  std::vector<std::unique_ptr<Node>> direct_nodes;
  direct_nodes.push_back(std::make_unique<Node>("direct"));
  auto direct_var = direct_nodes.back()->addVar("position", names, sample, std::vector<Bounds>(2, NoBound));
  auto direct_variables = std::make_shared<NodesVariables>("direct", std::move(direct_nodes));
  DiscreteCollisionConstraintD direct(evaluator, direct_var, "direct_collision");
  direct.linkWithVariables(direct_variables);
  ASSERT_EQ(direct.update(), spline.getRows());
  EXPECT_TRUE(spline.getValues().isApprox(direct.getValues(), 1e-9));
  const Eigen::MatrixXd analytic = spline.getJacobian();
  const Eigen::MatrixXd direct_jacobian = direct.getJacobian();
  const Eigen::VectorXd initial = variables->getValues();
  constexpr double STEP = 1e-6;
  for (Eigen::Index column = 14; column < 18; ++column)
  {
    EXPECT_TRUE(analytic.col(column).isApprox(map.basis(0, column / 2) * direct_jacobian.col(column % 2), 1e-9));
    Eigen::VectorXd x = initial;
    x[column] += STEP;
    variables->setVariables(x);
    ASSERT_EQ(spline.update(), direct.getRows());
    const Eigen::VectorXd plus = spline.getValues();
    x[column] -= 2 * STEP;
    variables->setVariables(x);
    ASSERT_EQ(spline.update(), direct.getRows());
    const Eigen::VectorXd minus = spline.getValues();
    // The central difference subtracts metre-scale binary64 distances before division by STEP.
    constexpr double ROUNDING_ULPS = 8.0;
    constexpr double RELATIVE_BAND = 1e-4;
    const double absolute_band = ROUNDING_ULPS * std::numeric_limits<double>::epsilon() / STEP;
    const Eigen::VectorXd numerical = (plus - minus) / (2 * STEP);
    for (Eigen::Index row = 0; row < numerical.size(); ++row)
      EXPECT_NEAR(
          numerical[row], analytic(row, column), absolute_band + RELATIVE_BAND * std::abs(analytic(row, column)));
  }
}
