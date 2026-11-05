#include <trajopt_ifopt/spline/spline_cartesian_line_constraint.h>
#include <trajopt_ifopt/spline/quintic_spline_model.h>
#include <trajopt_ifopt/constraints/cartesian_line_constraint.h>
#include <trajopt_ifopt/variable_sets/node.h>
#include <trajopt_ifopt/variable_sets/nodes_variables.h>
#include <trajopt_ifopt/variable_sets/var.h>
#include <tesseract/environment/environment.h>
#include <tesseract/common/resource_locator.h>
#include <tesseract/common/utils.h>
#include <tesseract/kinematics/joint_group.h>
#include <gtest/gtest.h>

using namespace trajopt_ifopt;

TEST(SplineCartesianLineConstraint, ActualSplinePoseAndChainJacobian)
{
  auto env = std::make_shared<tesseract::environment::Environment>();
  auto locator = std::make_shared<tesseract::common::GeneralResourceLocator>();
  ASSERT_TRUE(env->init(std::filesystem::path(std::string(TRAJOPT_DATA_DIR) + "/arm_around_table.urdf"),
                        std::filesystem::path(std::string(TRAJOPT_DATA_DIR) + "/pr2.srdf"),
                        locator));
  const auto manip = env->getJointGroup("right_arm");
  const auto n = manip->numJoints();
  const auto names = tesseract::common::toNames(manip->getJointIds());
  std::vector<std::unique_ptr<Node>> nodes;
  std::vector<std::shared_ptr<const Var>> knots;
  for (int i = 0; i < 7; ++i)
  {
    auto node = std::make_unique<Node>("sample-" + std::to_string(i));
    node->addVar("auxiliary", { "extra" }, Eigen::VectorXd::Zero(1), { NoBound });
    Eigen::VectorXd q = Eigen::VectorXd::Constant(n, 0.4 + 0.015 * i);
    knots.push_back(node->addVar("position", names, q, std::vector<Bounds>(static_cast<std::size_t>(n), NoBound)));
    nodes.push_back(std::move(node));
  }
  auto variables = std::make_shared<NodesVariables>("trajectory", std::move(nodes));
  const std::vector<Seconds> clock{ Seconds(0),   Seconds(.3),  Seconds(.7), Seconds(1.2),
                                    Seconds(1.8), Seconds(2.5), Seconds(3.1) };
  auto model = QuinticSplineModel::build(
      clock, Eigen::VectorXd::Zero(n), Eigen::VectorXd::Zero(n), Eigen::VectorXd::Zero(n), Eigen::VectorXd::Zero(n));
  auto sample = model.map({ Seconds(.9) }, 0);
  Eigen::VectorXd q = sample.offset.row(0).transpose();
  for (int i = 0; i < 7; ++i)
    q += sample.basis(0, i) * knots[static_cast<std::size_t>(i)]->value();
  const auto fk = manip->calcFwdKin(q);
  Eigen::Isometry3d start = fk.at("base_link").inverse() * fk.at("r_gripper_tool_frame");
  Eigen::Isometry3d finish = start;
  start.translation() += Eigen::Vector3d(-.5, .1, -.05);
  finish.translation() += Eigen::Vector3d(.5, -.1, .05);
  CartLineInfo info(manip, "r_gripper_tool_frame", "base_link", start, finish);
  SplineCartesianLineConstraint line(
      info, knots, sample.basis.row(0).transpose(), sample.offset.row(0).transpose(), Eigen::VectorXd::Ones(6));
  line.linkWithVariables(variables);
  auto sample_node = std::make_unique<Node>("actual");
  auto actual_var =
      sample_node->addVar("position", names, q, std::vector<Bounds>(static_cast<std::size_t>(n), NoBound));
  std::vector<std::unique_ptr<Node>> actual_nodes;
  actual_nodes.push_back(std::move(sample_node));
  auto actual_vars = std::make_shared<NodesVariables>("actual", std::move(actual_nodes));
  CartLineConstraint actual(info, actual_var, Eigen::VectorXd::Ones(6));
  actual.use_numeric_differentiation = false;
  actual.linkWithVariables(actual_vars);
  ASSERT_TRUE(line.getValues().isApprox(actual.calcValues(q), 1e-12));
  const Eigen::MatrixXd analytic = line.getJacobian();
  const Eigen::VectorXd initial = variables->getValues();
  const double step = 1e-6;
  for (Eigen::Index i = 0; i < initial.size(); ++i)
  {
    Eigen::VectorXd x = initial;
    x[i] += step;
    variables->setVariables(x);
    const Eigen::VectorXd plus = line.getValues();
    x[i] -= 2 * step;
    variables->setVariables(x);
    const Eigen::VectorXd minus = line.getValues();
    EXPECT_TRUE(((plus - minus) / (2 * step)).isApprox(analytic.col(i), 1e-5)) << "column " << i;
  }
  variables->setVariables(initial);
}
