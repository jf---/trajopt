/** @file discrete_collision_ranking_unit.cpp
 *  @brief A fixed-row discrete constraint must retain the most severe real contact.
 */
#include <trajopt_common/macros.h>
TRAJOPT_IGNORE_WARNINGS_PUSH
#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <tesseract/collision/types.h>
#include <tesseract/common/resource_locator.h>
#include <tesseract/common/types.h>
#include <tesseract/environment/commands/add_link_command.h>
#include <tesseract/environment/commands/change_link_collision_enabled_command.h>
#include <tesseract/environment/environment.h>
#include <tesseract/geometry/geometries.h>
#include <tesseract/kinematics/joint_group.h>
#include <tesseract/scene_graph/joint.h>
#include <tesseract/scene_graph/link.h>
TRAJOPT_IGNORE_WARNINGS_POP

#include <algorithm>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <trajopt_common/collision_types.h>
#include <trajopt_ifopt/constraints/collision/discrete_collision_constraint.h>
#include <trajopt_ifopt/constraints/collision/discrete_collision_evaluators.h>
#include <trajopt_ifopt/core/problem.h>
#include <trajopt_ifopt/utils/ifopt_utils.h>
#include <trajopt_ifopt/variable_sets/node.h>
#include <trajopt_ifopt/variable_sets/nodes_variables.h>
#include <trajopt_ifopt/variable_sets/var.h>

namespace
{
// Sphere distances are measured in metres; this checks equality of the same
// reported contact within a precision much smaller than the 0.01 m spacing.
constexpr double CONTACT_ERROR_TOLERANCE = 1e-8;
// A 1 micrometre joint displacement is small relative to the 10 mm gap
// between contact severities, so the active rows do not switch during FD.
constexpr double FINITE_DIFFERENCE_STEP = 1e-6;
constexpr double JACOBIAN_TOLERANCE = 1e-4;
constexpr std::size_t ROW_CAP = 3;

std::string obstacleName(bool active_first)
{
  const tesseract::common::LinkId moving("spherebot_link");
  for (int i = 0; i < 100; ++i)
  {
    const std::string candidate = "ranking_obstacle_" + std::to_string(i);
    const tesseract::common::LinkIdPair pair(moving, tesseract::common::LinkId(candidate));
    if ((pair.first() == moving) == active_first)
      return candidate;
  }
  throw std::runtime_error("could not construct the requested link ordering");
}

std::shared_ptr<tesseract::environment::Environment> makeEnvironment(bool active_first)
{
  using namespace tesseract::environment;
  using namespace tesseract::scene_graph;
  auto env = std::make_shared<Environment>();
  const auto locator = std::make_shared<tesseract::common::GeneralResourceLocator>();
  const std::filesystem::path urdf(std::string(TRAJOPT_DATA_DIR) + "/spherebot.urdf");
  const std::filesystem::path srdf(std::string(TRAJOPT_DATA_DIR) + "/spherebot.srdf");
  if (!env->init(urdf, srdf, locator))
    throw std::runtime_error("spherebot environment did not initialize");
  for (const char* name : { "test_sphere_link", "test_sphere_link2", "test_sphere_link3" })
  {
    if (!env->applyCommand(std::make_shared<ChangeLinkCollisionEnabledCommand>(name, false)))
      throw std::runtime_error("could not disable unrelated spherebot obstacle");
  }

  Link obstacle(obstacleName(active_first));
  // The first three 0.2 m spheres are clear but within the 0.1 m query buffer;
  // the last penetrates the 0.5 m robot sphere by 0.1 m at the zero joint state.
  for (const double x : { 0.77, 0.76, 0.75, 0.60 })
  {
    auto shape = std::make_shared<Collision>();
    shape->origin = Eigen::Isometry3d::Identity();
    shape->origin.translation().x() = x;
    shape->geometry = std::make_shared<tesseract::geometry::Sphere>(0.2);
    obstacle.collision.push_back(std::move(shape));
  }
  if (active_first)
  {
    Joint attachment("ranking_joint");
    attachment.type = JointType::FIXED;
    attachment.parent_link_id = tesseract::common::LinkId("spherebot_linkX");
    attachment.child_link_id = obstacle.getId();
    if (!env->applyCommand(std::make_shared<AddLinkCommand>(obstacle, attachment)))
      throw std::runtime_error("could not add moving multi-shape obstacle");
  }
  else if (!env->applyCommand(std::make_shared<AddLinkCommand>(obstacle)))
    throw std::runtime_error("could not add static multi-shape obstacle");
  return env;
}
}  // namespace

class DiscreteCollisionRanking : public testing::TestWithParam<std::tuple<bool, bool>>
{
};

TEST_P(DiscreteCollisionRanking, KeepsWorstActualContactInLimitedRows)  // NOLINT
{
  const auto [active_first, dynamic_environment] = GetParam();
  auto env = makeEnvironment(active_first);
  auto manip = env->getJointGroup("manipulator");
  trajopt_common::TrajOptCollisionConfig config(
      0.0, 1.0, tesseract::collision::ContactRequest(tesseract::collision::ContactTestType::ALL));
  config.collision_margin_buffer = 0.1;
  auto evaluator =
      std::make_shared<trajopt_ifopt::SingleTimestepCollisionEvaluator>(manip, env, config, dynamic_environment);

  const Eigen::Vector2d q = Eigen::Vector2d::Zero();
  trajopt_common::CollisionCacheData data;
  evaluator->calcCollisions(data, q, ROW_CAP);
  ASSERT_GT(data.gradient_results_sets.size(), ROW_CAP);
  const std::size_t active_index = active_first ? 0 : 1;
  for (const auto& contact : data.gradient_results_sets)
  {
    ASSERT_TRUE(contact.max_error[active_index].has_error[0]);
    if (!active_first)
      ASSERT_FALSE(contact.max_error[0].has_error[0]);
    else
      ASSERT_TRUE(contact.max_error[1].has_error[0]);
  }
  const auto worst =
      std::max_element(data.gradient_results_sets.begin(),
                       data.gradient_results_sets.end(),
                       [](const auto& a, const auto& b) { return a.getMaxErrorT0() < b.getMaxErrorT0(); });
  ASSERT_GT(worst->getMaxErrorT0(), 0.0);
  EXPECT_NEAR(data.gradient_results_sets.front().getMaxErrorT0(), worst->getMaxErrorT0(), CONTACT_ERROR_TOLERANCE);
  for (std::size_t i = 1; i < data.gradient_results_sets.size(); ++i)
    EXPECT_GE(data.gradient_results_sets[i - 1].getMaxErrorT0(), data.gradient_results_sets[i].getMaxErrorT0());

  auto node = std::make_unique<trajopt_ifopt::Node>("Joints");
  const std::vector<trajopt_ifopt::Bounds> bounds(2, trajopt_ifopt::NoBound);
  auto var = node->addVar("position", tesseract::common::toNames(manip->getJointIds()), q, bounds);
  std::vector<std::unique_ptr<trajopt_ifopt::Node>> nodes;
  nodes.push_back(std::move(node));
  auto vars = std::make_shared<trajopt_ifopt::NodesVariables>("trajectory", std::move(nodes));
  trajopt_ifopt::Problem nlp(vars);
  auto constraint = std::make_shared<trajopt_ifopt::DiscreteCollisionConstraint>(evaluator, var, ROW_CAP);
  nlp.addConstraintSet(constraint);
  nlp.setVariables(q.data());
  const Eigen::VectorXd values = constraint->getValues();
  EXPECT_NEAR(values[0], worst->getMaxErrorT0(), CONTACT_ERROR_TOLERANCE);
  EXPECT_GT(values[0], 0.0);
  std::vector<double> independently_ranked;
  for (const auto& contact : data.gradient_results_sets)
    independently_ranked.push_back(contact.getMaxErrorT0());
  std::sort(independently_ranked.begin(), independently_ranked.end(), std::greater<double>());
  for (std::size_t i = 0; i < ROW_CAP; ++i)
    EXPECT_NEAR(values[static_cast<Eigen::Index>(i)], independently_ranked[i], CONTACT_ERROR_TOLERANCE);

  const Eigen::MatrixXd analytic = nlp.getJacobianOfConstraints().toDense();
  const Eigen::MatrixXd numerical =
      trajopt_ifopt::calcNumericalConstraintGradient(q.data(), nlp, FINITE_DIFFERENCE_STEP).toDense();
  EXPECT_LT((analytic - numerical).cwiseAbs().maxCoeff(), JACOBIAN_TOLERANCE);
}

INSTANTIATE_TEST_SUITE_P(RealManager, DiscreteCollisionRanking, testing::Combine(testing::Bool(), testing::Bool()));
