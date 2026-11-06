#include <trajopt_ifopt/spline/spline_cartesian_path_progress_constraint.h>
#include <trajopt_ifopt/spline/quintic_spline_model.h>
#include <trajopt_ifopt/variable_sets/node.h>
#include <trajopt_ifopt/variable_sets/nodes_variables.h>
#include <trajopt_ifopt/variable_sets/var.h>
#include <tesseract/environment/environment.h>
#include <tesseract/common/resource_locator.h>
#include <tesseract/common/utils.h>
#include <tesseract/kinematics/joint_group.h>
#include <gtest/gtest.h>

using namespace trajopt_ifopt;
TEST(SplineCartesianPathProgressConstraint, OrderedEndpointsAndAnalyticJacobian)
{
  auto env = std::make_shared<tesseract::environment::Environment>();
  auto locator = std::make_shared<tesseract::common::GeneralResourceLocator>();
  ASSERT_TRUE(env->init(std::filesystem::path(std::string(TRAJOPT_DATA_DIR) + "/arm_around_table.urdf"),
                        std::filesystem::path(std::string(TRAJOPT_DATA_DIR) + "/pr2.srdf"),
                        locator));
  auto manip = env->getJointGroup("right_arm");
  const auto n = manip->numJoints();
  const auto names = tesseract::common::toNames(manip->getJointIds());
  std::vector<std::unique_ptr<Node>> nodes;
  std::vector<std::shared_ptr<const Var>> knots;
  for (int i = 0; i < 7; i++)
  {
    auto node = std::make_unique<Node>("p" + std::to_string(i));
    node->addVar("aux", { "aux" }, Eigen::VectorXd::Zero(1), { NoBound });
    Eigen::VectorXd q = Eigen::VectorXd::Constant(n, 0.4 + 0.02 * i);
    knots.push_back(node->addVar("position", names, q, std::vector<Bounds>(static_cast<std::size_t>(n), NoBound)));
    nodes.push_back(std::move(node));
  }
  auto variables = std::make_shared<NodesVariables>("knots", std::move(nodes));
  const std::vector<Seconds> clock = { Seconds(0),   Seconds(.3),  Seconds(.7), Seconds(1.2),
                                       Seconds(1.8), Seconds(2.5), Seconds(3.1) };
  auto model = QuinticSplineModel::build(
      clock, Eigen::VectorXd::Zero(n), Eigen::VectorXd::Zero(n), Eigen::VectorXd::Zero(n), Eigen::VectorXd::Zero(n));
  auto map = model.map({ Seconds(.5), Seconds(1.4), Seconds(2.8) }, 0);
  auto fk = manip->calcFwdKin(knots[3]->value());
  for (const std::string target : { "base_link", "r_forearm_link" })
  {
    Eigen::Isometry3d a = fk.at(target).inverse() * fk.at("r_gripper_tool_frame"), b = a;
    a.translation() += Eigen::Vector3d(-.4, .2, .1);
    b.translation() += Eigen::Vector3d(.4, -.2, -.1);
    CartLineInfo info(manip, "r_gripper_tool_frame", target, a, b);
    Eigen::MatrixXd targets(3, 3), tangents(3, 3);
    targets.row(0) = a.translation().transpose();
    targets.row(1) = ((a.translation() + b.translation()) / 2 + Eigen::Vector3d(0.0, 0.1, 0.0)).transpose();
    targets.row(2) = b.translation().transpose();
    tangents.row(0) = (targets.row(1) - targets.row(0)).normalized();
    tangents.row(1) = (targets.row(2) - targets.row(0)).normalized();
    tangents.row(2) = (targets.row(2) - targets.row(1)).normalized();
    Eigen::VectorXd arc(3);
    arc << 0.0, (targets.row(1) - targets.row(0)).norm(),
        (targets.row(1) - targets.row(0)).norm() + (targets.row(2) - targets.row(1)).norm();
    SplineCartesianPathProgressConstraint progress(info, knots, map.basis, map.offset, targets, tangents, arc, true, true);
    progress.linkWithVariables(variables);
    ASSERT_EQ(progress.getRows(), 8);
    auto bounds = progress.getBounds();
    EXPECT_DOUBLE_EQ(bounds[0].getLower(), 0);
    EXPECT_DOUBLE_EQ(bounds[0].getUpper(), 0);
    EXPECT_DOUBLE_EQ(bounds[1].getLower(), 0);
    EXPECT_DOUBLE_EQ(bounds[4].getUpper(), 1);
    EXPECT_DOUBLE_EQ(bounds[5].getLower(), 1);
    EXPECT_DOUBLE_EQ(bounds[5].getUpper(), 1);
    EXPECT_DOUBLE_EQ(bounds[6].getLower(), 0);
    EXPECT_DOUBLE_EQ(bounds[7].getLower(), 0);
    Eigen::MatrixXd jac = progress.getJacobian();
    Eigen::VectorXd initial = variables->getValues();
    const double step = 1e-6;
    for (Eigen::Index i = 0; i < initial.size(); i++)
    {
      auto x = initial;
      x[i] += step;
      variables->setVariables(x);
      Eigen::VectorXd plus = progress.getValues();
      x[i] -= 2 * step;
      variables->setVariables(x);
      Eigen::VectorXd minus = progress.getValues();
      // Central finite-difference error combines O(step^2) truncation and binary64 roundoff / step.
      constexpr double derivative_error_bound = 1e-7;
      EXPECT_LT((((plus - minus) / (2 * step)) - jac.col(i)).cwiseAbs().maxCoeff(), derivative_error_bound)
          << target << " axis " << i;
    }
    variables->setVariables(initial);
    arc[2] = 0.0;
    EXPECT_THROW(SplineCartesianPathProgressConstraint(info, knots, map.basis, map.offset, targets, tangents, arc, true, true),
                 std::invalid_argument);
  }
}
