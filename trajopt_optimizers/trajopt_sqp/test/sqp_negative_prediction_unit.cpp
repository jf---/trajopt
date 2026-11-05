/** @file sqp_negative_prediction_unit.cpp
 *  @brief SQP must not report a materially worse QP model as convergence.
 */
#include <trajopt_common/macros.h>
TRAJOPT_IGNORE_WARNINGS_PUSH
#include <gtest/gtest.h>
#include <Eigen/Core>
TRAJOPT_IGNORE_WARNINGS_POP

#include <cmath>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <trajopt_ifopt/constraints/joint_position_constraint.h>
#include <trajopt_ifopt/variable_sets/node.h>
#include <trajopt_ifopt/variable_sets/nodes_variables.h>
#include <trajopt_ifopt/variable_sets/var.h>
#include <trajopt_sqp/osqp_eigen_solver.h>
#include <trajopt_sqp/trajopt_qp_problem.h>
#include <trajopt_sqp/trust_region_sqp_solver.h>
#include <trajopt_sqp/types.h>

#include "scripted_qp_solver.h"

namespace
{
std::shared_ptr<trajopt_sqp::TrajOptQPProblem> makeProblem(const Eigen::Vector2d& target = Eigen::Vector2d(0.8, -0.3))
{
  auto node = std::make_unique<trajopt_ifopt::Node>("Joints");
  const std::vector<std::string> names{ "j0", "j1" };
  const std::vector<trajopt_ifopt::Bounds> bounds(2, trajopt_ifopt::Bounds(-1.0, 1.0));
  const std::shared_ptr<const trajopt_ifopt::Var> var =
      node->addVar("position", names, Eigen::Vector2d::Zero(), bounds);
  std::vector<std::unique_ptr<trajopt_ifopt::Node>> nodes;
  nodes.push_back(std::move(node));
  auto variables = std::make_shared<trajopt_ifopt::NodesVariables>("trajectory", std::move(nodes));
  auto qp = std::make_shared<trajopt_sqp::TrajOptQPProblem>(variables);
  qp->addCostSet(std::make_shared<trajopt_ifopt::JointPosConstraint>(target, var, Eigen::VectorXd::Ones(1), "Target"),
                 trajopt_sqp::CostPenaltyType::kSquared);
  qp->setup();
  return qp;
}

/** Inject an inaccurate QP solution while exercising the real SQP and real convex problem. */
class IncorrectTrialSolver : public trajopt_sqp::test::ScriptedQPSolver
{
public:
  explicit IncorrectTrialSolver(int incorrect_trials)
    : ScriptedQPSolver(std::make_shared<trajopt_sqp::OSQPEigenSolver>()), incorrect_trials_(incorrect_trials)
  {
  }

  Eigen::VectorXd getSolution() override
  {
    Eigen::VectorXd solution = ScriptedQPSolver::getSolution();
    if (solves <= incorrect_trials_)
    {
      // Inside the current box: 0.05 on the first solve, 0.005 after one 0.1 shrink.
      const double offset = 0.05 * std::pow(0.1, solves - 1);
      solution[0] = -offset;
      solution[1] = offset;
    }
    return solution;
  }

private:
  int incorrect_trials_;
};

class SlowIncorrectTrialSolver : public IncorrectTrialSolver
{
public:
  SlowIncorrectTrialSolver() : IncorrectTrialSolver(100) {}
  bool solve() override
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    return IncorrectTrialSolver::solve();
  }
};

class SmallNegativeTrialSolver : public trajopt_sqp::test::ScriptedQPSolver
{
public:
  SmallNegativeTrialSolver() : ScriptedQPSolver(std::make_shared<trajopt_sqp::OSQPEigenSolver>()) {}

  Eigen::VectorXd getSolution() override
  {
    Eigen::VectorXd solution = ScriptedQPSolver::getSolution();
    solution[0] = -0.000001;
    solution[1] = 0.000001;
    return solution;
  }
};

class DelayedIncorrectTrialSolver : public trajopt_sqp::test::ScriptedQPSolver
{
public:
  DelayedIncorrectTrialSolver() : ScriptedQPSolver(std::make_shared<trajopt_sqp::OSQPEigenSolver>()) {}

  Eigen::VectorXd getSolution() override
  {
    Eigen::VectorXd solution = ScriptedQPSolver::getSolution();
    if (solves == 1)
      accepted = solution.head(2);
    else
    {
      const double offset = 0.05 * std::pow(0.1, solves - 2);
      solution[0] = accepted[0] - offset;
      solution[1] = accepted[1] + offset;
    }
    return solution;
  }

  Eigen::Vector2d accepted{ Eigen::Vector2d::Zero() };
};
}  // namespace

TEST(SQPNegativePrediction, RetriesAndRecoversWhenNextQPSolutionIsValid)  // NOLINT
{
  auto qp = std::make_shared<IncorrectTrialSolver>(1);
  trajopt_sqp::TrustRegionSQPSolver solver(qp);
  solver.solve(makeProblem());
  EXPECT_GT(qp->solves, 1);
  EXPECT_EQ(solver.getStatus(), trajopt_sqp::SQPStatus::kConverged);
  EXPECT_NEAR(solver.getResults().best_var_vals[0], 0.8, 1e-3);
  EXPECT_NEAR(solver.getResults().best_var_vals[1], -0.3, 1e-3);
}

TEST(SQPNegativePrediction, PersistentInvalidModelDoesNotBecomeFeasibleSuccess)  // NOLINT
{
  auto qp = std::make_shared<IncorrectTrialSolver>(100);
  trajopt_sqp::TrustRegionSQPSolver solver(qp);
  solver.params.min_trust_box_size = 0.009;
  solver.solve(makeProblem());
  EXPECT_GT(qp->solves, 1);
  EXPECT_EQ(solver.getStatus(), trajopt_sqp::SQPStatus::kInvalidPredictedImprovement);
  EXPECT_NEAR(solver.getResults().best_var_vals[0], 0.0, 1e-12);
  EXPECT_NEAR(solver.getResults().best_var_vals[1], 0.0, 1e-12);
}

TEST(SQPNegativePrediction, SmallNonnegativePredictionRemainsConverged)  // NOLINT
{
  auto qp = std::make_shared<trajopt_sqp::OSQPEigenSolver>();
  trajopt_sqp::TrustRegionSQPSolver solver(qp);
  solver.solve(makeProblem(Eigen::Vector2d(0.0001, 0.0)));
  EXPECT_GE(solver.getResults().approx_merit_improve, 0.0);
  EXPECT_LT(solver.getResults().approx_merit_improve, solver.params.min_approx_improve);
  EXPECT_EQ(solver.getStatus(), trajopt_sqp::SQPStatus::kConverged);
}

TEST(SQPNegativePrediction, SmallNegativePredictionWithinToleranceRemainsConverged)  // NOLINT
{
  auto qp = std::make_shared<SmallNegativeTrialSolver>();
  trajopt_sqp::TrustRegionSQPSolver solver(qp);
  solver.solve(makeProblem());
  EXPECT_EQ(qp->solves, 1);
  EXPECT_LT(solver.getResults().approx_merit_improve, 0.0);
  EXPECT_GE(solver.getResults().approx_merit_improve, -solver.params.min_approx_improve);
  EXPECT_EQ(solver.getStatus(), trajopt_sqp::SQPStatus::kConverged);
}

TEST(SQPNegativePrediction, InvalidRetriesRetainPreviouslyAcceptedStep)  // NOLINT
{
  auto qp = std::make_shared<DelayedIncorrectTrialSolver>();
  trajopt_sqp::TrustRegionSQPSolver solver(qp);
  solver.params.min_trust_box_size = 0.014;
  solver.solve(makeProblem());
  EXPECT_GT(qp->solves, 2);
  EXPECT_EQ(solver.getStatus(), trajopt_sqp::SQPStatus::kInvalidPredictedImprovement);
  EXPECT_GT(qp->accepted.norm(), 0.01);
  EXPECT_NEAR(solver.getResults().best_var_vals[0], qp->accepted[0], 1e-12);
  EXPECT_NEAR(solver.getResults().best_var_vals[1], qp->accepted[1], 1e-12);
}

TEST(SQPNegativePrediction, RetryRespectsOverallTimeLimit)  // NOLINT
{
  auto qp = std::make_shared<SlowIncorrectTrialSolver>();
  trajopt_sqp::TrustRegionSQPSolver solver(qp);
  solver.params.max_time = 0.1;
  solver.solve(makeProblem());
  EXPECT_EQ(qp->solves, 1);
  EXPECT_EQ(solver.getStatus(), trajopt_sqp::SQPStatus::kTimeLimit);
}
