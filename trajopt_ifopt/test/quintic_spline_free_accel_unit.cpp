#include <trajopt_ifopt/spline/quintic_spline_model.h>
#include <gtest/gtest.h>

#include <optional>
#include <vector>

using trajopt_ifopt::QuinticSplineModel;
using trajopt_ifopt::Seconds;

namespace
{
const std::vector<Seconds> TIMES = { Seconds(0.0), Seconds(0.2), Seconds(0.65), Seconds(1.1),
                                     Seconds(1.7), Seconds(2.3), Seconds(3.0) };
const std::vector<Seconds> SAMPLES = { Seconds(0.0), Seconds(0.13), Seconds(0.41),
                                       Seconds(1.4), Seconds(2.4),  Seconds(3.0) };
Eigen::MatrixXd positions()
{
  Eigen::MatrixXd x(7, 2);
  x << 0.1, -0.2, -0.2, 0.3, 0.4, 0.2, 0.25, -0.1, 0.7, 0.5, 0.3, 0.8, 0.9, 0.2;
  return x;
}
}  // namespace

TEST(QuinticSplineFreeAcceleration, FreeBothMatchesClampedReferenceAndJacobians)
{
  Eigen::Vector2d v0(0.35, -0.1), a0(-0.18, 0.23), v1(-0.22, 0.15), a1(0.27, -0.11);
  const auto fixed = QuinticSplineModel::build(TIMES, v0, a0, v1, a1);
  const auto free = QuinticSplineModel::buildVariableAcceleration(TIMES, v0, std::nullopt, v1, std::nullopt);
  EXPECT_EQ(free.knotCount(), 7);
  EXPECT_EQ(free.variableCount(), 9);
  Eigen::MatrixXd variables(9, 2);
  variables.topRows(7) = positions();
  variables.row(7) = a0.transpose();
  variables.row(8) = a1.transpose();
  for (unsigned order = 0; order <= 3; ++order)
  {
    const auto a = free.map(SAMPLES, order);
    const auto b = fixed.map(SAMPLES, order);
    EXPECT_TRUE((a.basis * variables + a.offset).isApprox(b.basis * positions() + b.offset, 1e-10));
    for (Eigen::Index column = 7; column < 9; ++column)
    {
      const double step = 1e-6;
      Eigen::MatrixXd plus = variables;
      Eigen::MatrixXd minus = variables;
      plus(column, 0) += step;
      minus(column, 0) -= step;
      const Eigen::VectorXd derivative =
          ((a.basis * plus + a.offset) - (a.basis * minus + a.offset)).col(0) / (2 * step);
      EXPECT_TRUE(derivative.isApprox(a.basis.col(column), 1e-6));
    }
  }
  const auto batch = free.evaluate(SAMPLES, variables, 3);
  EXPECT_EQ(batch.size(), 4);
  EXPECT_EQ(free.jerkQuadrature().weights.size(), 18);
  EXPECT_TRUE(batch[2].row(0).isApprox(a0.transpose(), 1e-10));
  EXPECT_TRUE(batch[2].row(batch[2].rows() - 1).isApprox(a1.transpose(), 1e-10));
}

TEST(QuinticSplineFreeAcceleration, EachSideCanBeFixedOrFree)
{
  Eigen::Vector2d v0(0.35, -0.1), a0(-0.18, 0.23), v1(-0.22, 0.15), a1(0.27, -0.11);
  const auto fixed = QuinticSplineModel::build(TIMES, v0, a0, v1, a1);
  for (bool free_start : { false, true })
  {
    const auto spline =
        QuinticSplineModel::buildVariableAcceleration(TIMES,
                                                      v0,
                                                      free_start ? std::nullopt : std::optional<Eigen::VectorXd>(a0),
                                                      v1,
                                                      free_start ? std::optional<Eigen::VectorXd>(a1) : std::nullopt);
    EXPECT_EQ(spline.variableCount(), 8);
    Eigen::MatrixXd variables(8, 2);
    variables.topRows(7) = positions();
    variables.row(7) = free_start ? a0.transpose() : a1.transpose();
    for (unsigned order = 0; order <= 3; ++order)
    {
      const auto actual = spline.map(SAMPLES, order);
      const auto expected = fixed.map(SAMPLES, order);
      EXPECT_TRUE(
          (actual.basis * variables + actual.offset).isApprox(expected.basis * positions() + expected.offset, 1e-10));
    }
  }
}
