/** @file dynamic_spline_collision_constraint_unit.cpp
 * @brief Compare all dynamic spline collision rows against the native discrete constraint.
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

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <trajopt_ifopt/constraints/collision/discrete_collision_constraint.h>
#include <trajopt_ifopt/constraints/collision/discrete_collision_evaluators.h>
#include <trajopt_ifopt/constraints/collision/dynamic_spline_collision_constraint.h>
#include <trajopt_ifopt/variable_sets/node.h>
#include <trajopt_ifopt/variable_sets/nodes_variables.h>
#include <trajopt_ifopt/variable_sets/var.h>

namespace
{
constexpr std::size_t KNOT_COUNT = 7;
constexpr Eigen::Index AXES = 2;
// Real Bullet contact values in metres; much smaller than 10 mm sphere spacing.
constexpr double CONTACT_TOLERANCE = 1e-8;
// One micrometre joint displacement preserves the fixture's contact ordering.
constexpr double DIFFERENCE_STEP = 1e-6;
// Error in the centred finite-difference derivative away from contact switches.
constexpr double JACOBIAN_TOLERANCE = 1e-4;
// Four shape contacts exceed the old three-row cap.
constexpr int OLD_ROW_CAP = 3;

std::string obstacleName(bool active_first)
{
  const tesseract::common::LinkId moving("spherebot_link");
  for (int i = 0; i < 100; ++i)
  {
    const std::string candidate = "spline_obstacle_" + std::to_string(i);
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
    if (!env->applyCommand(std::make_shared<ChangeLinkCollisionEnabledCommand>(name, false)))
      throw std::runtime_error("could not disable unrelated obstacle");

  Link obstacle(obstacleName(active_first));
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
    Joint attachment("spline_obstacle_joint");
    attachment.type = JointType::FIXED;
    attachment.parent_link_id = tesseract::common::LinkId("spherebot_linkX");
    attachment.child_link_id = obstacle.getId();
    if (!env->applyCommand(std::make_shared<AddLinkCommand>(obstacle, attachment)))
      throw std::runtime_error("could not add moving obstacle");
  }
  else if (!env->applyCommand(std::make_shared<AddLinkCommand>(obstacle)))
    throw std::runtime_error("could not add static obstacle");
  return env;
}

class DynamicSplineCollisionFixture : public testing::TestWithParam<std::tuple<bool, bool>>
{
protected:
  std::shared_ptr<tesseract::environment::Environment> env;
  std::shared_ptr<trajopt_ifopt::SingleTimestepCollisionEvaluator> evaluator;
  std::shared_ptr<trajopt_ifopt::NodesVariables> knots;
  std::shared_ptr<trajopt_ifopt::NodesVariables> scratch;
  std::shared_ptr<trajopt_ifopt::DynamicSplineCollisionConstraint> spline;
  std::shared_ptr<trajopt_ifopt::DiscreteCollisionConstraintD> native;
  Eigen::VectorXd basis;
  Eigen::Vector2d offset;
  Eigen::VectorXd near_knots;

  void SetUp() override
  {
    const auto [active_first, dynamic_environment] = GetParam();
    env = makeEnvironment(active_first);
    const auto manip = env->getJointGroup("manipulator");
    trajopt_common::TrajOptCollisionConfig config(
        0.0, 1.0, tesseract::collision::ContactRequest(tesseract::collision::ContactTestType::ALL));
    config.collision_margin_buffer = 0.1;
    evaluator = std::make_shared<trajopt_ifopt::SingleTimestepCollisionEvaluator>(
        manip, env, config, dynamic_environment);

    const std::vector<std::string> names = tesseract::common::toNames(manip->getJointIds());
    const std::vector<trajopt_ifopt::Bounds> bounds(static_cast<std::size_t>(AXES), trajopt_ifopt::NoBound);
    std::vector<std::unique_ptr<trajopt_ifopt::Node>> nodes;
    std::vector<std::shared_ptr<const trajopt_ifopt::Var>> positions;
    for (std::size_t i = 0; i < KNOT_COUNT; ++i)
    {
      nodes.push_back(std::make_unique<trajopt_ifopt::Node>("spline_knot_" + std::to_string(i)));
      const double index = static_cast<double>(i);
      positions.push_back(nodes.back()->addVar("position", names, Eigen::Vector2d(0.001 * index, -0.002 * index), bounds));
    }
    knots = std::make_shared<trajopt_ifopt::NodesVariables>("spline_knots", std::move(nodes));
    near_knots = knots->getValues();
    basis.resize(static_cast<Eigen::Index>(KNOT_COUNT));
    basis << 0.05, -0.10, 0.30, 0.60, 0.20, -0.10, 0.05;
    offset << 0.005, -0.003;
    spline = std::make_shared<trajopt_ifopt::DynamicSplineCollisionConstraint>(evaluator, positions, basis, offset);
    spline->linkWithVariables(knots);

    std::vector<std::unique_ptr<trajopt_ifopt::Node>> native_nodes;
    native_nodes.push_back(std::make_unique<trajopt_ifopt::Node>("native_sample"));
    auto native_var = native_nodes.back()->addVar("position", names, sample(near_knots), bounds);
    scratch = std::make_shared<trajopt_ifopt::NodesVariables>("native_sample", std::move(native_nodes));
    native = std::make_shared<trajopt_ifopt::DiscreteCollisionConstraintD>(evaluator, native_var, "native");
    native->linkWithVariables(scratch);
  }

  Eigen::Vector2d sample(const Eigen::VectorXd& values) const
  {
    Eigen::Vector2d q = offset;
    for (std::size_t i = 0; i < KNOT_COUNT; ++i)
      q.noalias() += basis[static_cast<Eigen::Index>(i)] * values.segment<AXES>(static_cast<Eigen::Index>(i) * AXES);
    return q;
  }

  void updateBoth(const Eigen::VectorXd& values)
  {
    knots->setVariables(values);
    scratch->setVariables(sample(values));
    spline->update();
    native->update();
  }

  void expectParity()
  {
    ASSERT_EQ(spline->getRows(), native->getRows());
    EXPECT_TRUE(spline->isDynamic());
    EXPECT_EQ(spline->getValues().size(), native->getValues().size());
    EXPECT_TRUE(spline->getValues().isApprox(native->getValues(), CONTACT_TOLERANCE));
    EXPECT_TRUE(spline->getCoefficients().isApprox(native->getCoefficients(), CONTACT_TOLERANCE));
    const auto direct_bounds = spline->getBounds();
    const auto native_bounds = native->getBounds();
    ASSERT_EQ(direct_bounds.size(), native_bounds.size());
    for (std::size_t i = 0; i < direct_bounds.size(); ++i)
    {
      EXPECT_EQ(direct_bounds[i].getLower(), native_bounds[i].getLower());
      EXPECT_EQ(direct_bounds[i].getUpper(), native_bounds[i].getUpper());
    }
    const Eigen::MatrixXd direct_jacobian = spline->getJacobian().toDense();
    const Eigen::MatrixXd native_jacobian = native->getJacobian().toDense();
    ASSERT_EQ(direct_jacobian.rows(), native_jacobian.rows());
    ASSERT_EQ(direct_jacobian.cols(), static_cast<Eigen::Index>(KNOT_COUNT) * AXES);
    for (std::size_t i = 0; i < KNOT_COUNT; ++i)
      EXPECT_TRUE(direct_jacobian.middleCols<AXES>(static_cast<Eigen::Index>(i) * AXES)
                      .isApprox(basis[static_cast<Eigen::Index>(i)] * native_jacobian, CONTACT_TOLERANCE));
  }
};

TEST_P(DynamicSplineCollisionFixture, MatchesEveryNativeContactAndFiniteDifferences)  // NOLINT
{
  updateBoth(near_knots);
  ASSERT_GT(spline->getRows(), OLD_ROW_CAP);
  expectParity();
  const Eigen::MatrixXd analytic = spline->getJacobian().toDense();
  Eigen::MatrixXd numerical = Eigen::MatrixXd::Zero(analytic.rows(), analytic.cols());
  for (Eigen::Index column = 0; column < near_knots.size(); ++column)
  {
    Eigen::VectorXd plus = near_knots;
    plus[column] += DIFFERENCE_STEP;
    knots->setVariables(plus);
    spline->update();
    ASSERT_EQ(spline->getRows(), analytic.rows());
    const Eigen::VectorXd high = spline->getValues();
    Eigen::VectorXd minus = near_knots;
    minus[column] -= DIFFERENCE_STEP;
    knots->setVariables(minus);
    spline->update();
    ASSERT_EQ(spline->getRows(), analytic.rows());
    numerical.col(column) = (high - spline->getValues()) / (2.0 * DIFFERENCE_STEP);
  }
  updateBoth(near_knots);
  EXPECT_LT((analytic - numerical).cwiseAbs().maxCoeff(), JACOBIAN_TOLERANCE);
  expectParity();
}

TEST_P(DynamicSplineCollisionFixture, ContactRowsDisappearAndReappearOnUpdates)  // NOLINT
{
  updateBoth(near_knots);
  const int initial_rows = spline->getRows();
  ASSERT_GT(initial_rows, OLD_ROW_CAP);
  expectParity();

  // All spline basis weights sum to one: setting each y knot to 0.5 m
  // leaves only the closest sphere inside the 0.1 m contact query buffer.
  Eigen::VectorXd partial = near_knots;
  for (std::size_t i = 0; i < KNOT_COUNT; ++i)
    partial[static_cast<Eigen::Index>(i) * AXES + 1] = 0.5;
  updateBoth(partial);
  EXPECT_GT(spline->getRows(), 0);
  EXPECT_LT(spline->getRows(), initial_rows);
  expectParity();

  // At a 2 m transverse separation, none of these spheres are in the query buffer.
  Eigen::VectorXd distant = near_knots;
  for (std::size_t i = 0; i < KNOT_COUNT; ++i)
    distant[static_cast<Eigen::Index>(i) * AXES + 1] = 2.0;
  updateBoth(distant);
  EXPECT_EQ(spline->getRows(), 0);
  expectParity();

  updateBoth(near_knots);
  EXPECT_EQ(spline->getRows(), initial_rows);
  expectParity();
  updateBoth(distant);
  EXPECT_EQ(spline->getRows(), 0);
  updateBoth(near_knots);
  EXPECT_EQ(spline->getRows(), initial_rows);
  expectParity();
}

INSTANTIATE_TEST_SUITE_P(RealManager,
                         DynamicSplineCollisionFixture,
                         testing::Combine(testing::Bool(), testing::Bool()));

}  // namespace
