#include <trajopt_ifopt/spline/spline_timed_cartesian_pose.h>
#include <trajopt_ifopt/spline/quintic_spline_model.h>
#include <trajopt_ifopt/variable_sets/node.h>
#include <trajopt_ifopt/variable_sets/nodes_variables.h>
#include <trajopt_ifopt/variable_sets/var.h>
#include <tesseract/environment/environment.h>
#include <tesseract/common/resource_locator.h>
#include <tesseract/common/utils.h>
#include <tesseract/kinematics/joint_group.h>
#include <gtest/gtest.h>
#include <limits>

using namespace trajopt_ifopt;

TEST(SplineTimedCartesianPose, ActualPoseAndAnalyticJacobianInMovingAndFixedFrames)
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
    knots.push_back(node->addVar("position",
                                 names,
                                 Eigen::VectorXd::Constant(n, 0.4 + 0.015 * i),
                                 std::vector<Bounds>(static_cast<std::size_t>(n), NoBound)));
    nodes.push_back(std::move(node));
  }
  auto variables = std::make_shared<NodesVariables>("trajectory", std::move(nodes));
  const std::vector<Seconds> clock{ Seconds(0),   Seconds(.3),  Seconds(.7), Seconds(1.2),
                                    Seconds(1.8), Seconds(2.5), Seconds(3.1) };
  const auto model = QuinticSplineModel::build(
      clock, Eigen::VectorXd::Zero(n), Eigen::VectorXd::Zero(n), Eigen::VectorXd::Zero(n), Eigen::VectorXd::Zero(n));
  const auto sample = model.map({ Seconds(.9) }, 0);
  Eigen::VectorXd q = sample.offset.row(0).transpose();
  for (int i = 0; i < 7; ++i)
    q += sample.basis(0, i) * knots[static_cast<std::size_t>(i)]->value();
  const auto fk = manip->calcFwdKin(q);
  for (const std::string frame : { "base_link", "r_forearm_link" })
  {
    Eigen::Isometry3d target = fk.at(frame).inverse() * fk.at("r_gripper_tool_frame");
    SplineTimedCartesianPose exact(manip,
                                   "r_gripper_tool_frame",
                                   frame,
                                   target,
                                   knots,
                                   sample.basis.row(0).transpose(),
                                   sample.offset.row(0).transpose(),
                                   WorkpiecePositionScale{ 0.2 },
                                   OrientationScale{ 1.0 },
                                   2.0);
    exact.linkWithVariables(variables);
    EXPECT_LT(exact.getValues().norm(), 1e-10);
    EXPECT_NEAR(exact.getCoefficients()[0], 50.0, 1e-12);
    EXPECT_NEAR(exact.getCoefficients()[3], 2.0, 1e-12);
    EXPECT_THROW((SplineTimedCartesianPose(manip,
                                           "r_gripper_tool_frame",
                                           frame,
                                           target,
                                           knots,
                                           sample.basis.row(0).transpose(),
                                           sample.offset.row(0).transpose(),
                                           WorkpiecePositionScale{ 0.0 },
                                           OrientationScale{ 1.0 },
                                           2.0)),
                 std::invalid_argument);
    EXPECT_THROW((SplineTimedCartesianPose(manip,
                                           "r_gripper_tool_frame",
                                           frame,
                                           target,
                                           knots,
                                           sample.basis.row(0).transpose(),
                                           sample.offset.row(0).transpose(),
                                           WorkpiecePositionScale{ std::numeric_limits<double>::max() },
                                           OrientationScale{ 1.0 },
                                           2.0)),
                 TimedCartesianPoseError);
    Eigen::Isometry3d malformed = target;
    malformed.linear() *= 2.0;
    EXPECT_THROW((SplineTimedCartesianPose(manip,
                                           "r_gripper_tool_frame",
                                           frame,
                                           malformed,
                                           knots,
                                           sample.basis.row(0).transpose(),
                                           sample.offset.row(0).transpose(),
                                           WorkpiecePositionScale{ 0.2 },
                                           OrientationScale{ 1.0 },
                                           2.0)),
                 std::invalid_argument);
    target.translation() += Eigen::Vector3d(0.03, -0.02, 0.01);
    target.linear() *= Eigen::AngleAxisd(0.18, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    SplineTimedCartesianPose cost(manip,
                                  "r_gripper_tool_frame",
                                  frame,
                                  target,
                                  knots,
                                  sample.basis.row(0).transpose(),
                                  sample.offset.row(0).transpose(),
                                  WorkpiecePositionScale{ 0.2 },
                                  OrientationScale{ 1.0 },
                                  2.0);
    cost.linkWithVariables(variables);
    const Eigen::MatrixXd analytic = cost.getJacobian();
    const Eigen::VectorXd initial = variables->getValues();
    constexpr double step = 1e-6;                 // Central finite-difference joint perturbation.
    constexpr double jacobian_error_band = 1e-7;  // Numerical derivative error in mixed pose units per joint unit.
    for (Eigen::Index i = 0; i < initial.size(); ++i)
    {
      Eigen::VectorXd x = initial;
      x[i] += step;
      variables->setVariables(x);
      const Eigen::VectorXd plus = cost.getValues();
      x[i] -= 2 * step;
      variables->setVariables(x);
      const Eigen::VectorXd minus = cost.getValues();
      EXPECT_LT((((plus - minus) / (2 * step)) - analytic.col(i)).norm(), jacobian_error_band)
          << frame << " column " << i;
    }
    variables->setVariables(initial);
  }
}
