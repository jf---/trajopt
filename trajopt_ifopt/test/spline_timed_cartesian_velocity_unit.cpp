#include <trajopt_ifopt/spline/spline_timed_cartesian_velocity.h>
#include <trajopt_ifopt/costs/squared_cost.h>
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

TEST(SplineTimedCartesianVelocity, WorkpieceResidualAndJacobian)
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
  const auto map = model.map({ Seconds(.9), Seconds(1.37) }, 0);
  std::vector<Eigen::VectorXd> q(2, Eigen::VectorXd::Zero(n));
  for (int k = 0; k < 2; ++k)
  {
    q[static_cast<std::size_t>(k)] = map.offset.row(k).transpose();
    for (int i = 0; i < 7; ++i)
      q[static_cast<std::size_t>(k)] += map.basis(k, i) * knots[static_cast<std::size_t>(i)]->value();
  }
  constexpr double dt_s = .47;
  constexpr double speed_m_s = .2;
  for (const std::string frame : { "base_link", "r_forearm_link" })
  {
    const auto fk0 = manip->calcFwdKin(q[0]);
    const auto fk1 = manip->calcFwdKin(q[1]);
    const Eigen::Isometry3d actual0 = fk0.at(frame).inverse() * fk0.at("r_gripper_tool_frame");
    const Eigen::Isometry3d actual1 = fk1.at(frame).inverse() * fk1.at("r_gripper_tool_frame");
    Eigen::Isometry3d target0 = actual0;
    Eigen::Isometry3d target1 = actual1;
    target0.translation() += Eigen::Vector3d(.02, -.01, .03);
    target1.translation() += Eigen::Vector3d(-.01, .04, .02);
    target0.linear() *= Eigen::AngleAxisd(.3, Eigen::Vector3d::UnitX()).toRotationMatrix();
    target1.linear() *= Eigen::AngleAxisd(-.2, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    CartLineInfo info(manip, "r_gripper_tool_frame", frame, target0, target1);
    SplineTimedCartesianVelocity cost(info,
                                      target0,
                                      target1,
                                      knots,
                                      map.basis.row(0).transpose(),
                                      map.offset.row(0).transpose(),
                                      map.basis.row(1).transpose(),
                                      map.offset.row(1).transpose(),
                                      Seconds(dt_s),
                                      WorkpieceSpeedScale{ speed_m_s },
                                      Seconds(2.0),
                                      3.0);
    cost.linkWithVariables(variables);
    const Eigen::Vector3d expected =
        ((actual1.translation() - target1.translation()) - (actual0.translation() - target0.translation())) /
        (dt_s * speed_m_s);
    EXPECT_LT((cost.getValues() - expected).norm(), 1e-10);
    EXPECT_NEAR(cost.getCoefficients()[0], 3.0 * dt_s / 2.0, 1e-12);
    EXPECT_THROW((SplineTimedCartesianVelocity(info,
                                               target0,
                                               target1,
                                               knots,
                                               map.basis.row(0).transpose(),
                                               map.offset.row(0).transpose(),
                                               map.basis.row(1).transpose(),
                                               map.offset.row(1).transpose(),
                                               Seconds(0.0),
                                               WorkpieceSpeedScale{ speed_m_s },
                                               Seconds(2.0),
                                               3.0)),
                 TimedCartesianVelocityError);
    const Eigen::MatrixXd analytic = cost.getJacobian();
    const Eigen::VectorXd initial = variables->getValues();
    constexpr double step = 1e-6;             // Central finite-difference joint perturbation.
    constexpr double derivative_band = 1e-7;  // Difference derivative error in inverse joint units.
    for (Eigen::Index i = 0; i < initial.size(); ++i)
    {
      Eigen::VectorXd x = initial;
      x[i] += step;
      variables->setVariables(x);
      const Eigen::VectorXd plus = cost.getValues();
      x[i] -= 2 * step;
      variables->setVariables(x);
      const Eigen::VectorXd minus = cost.getValues();
      EXPECT_LT((((plus - minus) / (2 * step)) - analytic.col(i)).norm(), derivative_band) << frame << " column " << i;
    }
    variables->setVariables(initial);
    auto term = std::make_shared<SplineTimedCartesianVelocity>(info,
                                                               target0,
                                                               target1,
                                                               knots,
                                                               map.basis.row(0).transpose(),
                                                               map.offset.row(0).transpose(),
                                                               map.basis.row(1).transpose(),
                                                               map.offset.row(1).transpose(),
                                                               Seconds(dt_s),
                                                               WorkpieceSpeedScale{ speed_m_s },
                                                               Seconds(2.0),
                                                               3.0);
    term->linkWithVariables(variables);
    SquaredCost consumer(term, term->getCoefficients());
    EXPECT_NEAR(consumer.getCost(), (3.0 * dt_s / 2.0) * expected.squaredNorm(), 1e-10);
    const Eigen::MatrixXd gradient = consumer.getJacobian();
    for (Eigen::Index i = 0; i < initial.size(); ++i)
    {
      Eigen::VectorXd x = initial;
      x[i] += step;
      variables->setVariables(x);
      const double plus = consumer.getCost();
      x[i] -= 2 * step;
      variables->setVariables(x);
      const double minus = consumer.getCost();
      EXPECT_NEAR((plus - minus) / (2 * step), gradient(0, i), derivative_band) << frame << " column " << i;
    }
    variables->setVariables(initial);
  }
}
