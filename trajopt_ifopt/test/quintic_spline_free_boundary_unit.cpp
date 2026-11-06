#include <trajopt_ifopt/spline/quintic_spline_model.h>
#include <gtest/gtest.h>

#include <optional>
#include <vector>

using trajopt_ifopt::QuinticSplineModel;
using trajopt_ifopt::Seconds;

TEST(QuinticSplineFreeBoundary, VelocityAndAccelerationMatchFixedCurveAtEitherEnd)
{
  const std::vector<Seconds> times = { Seconds(0),   Seconds(.2),  Seconds(.65), Seconds(1.1),
                                       Seconds(1.7), Seconds(2.3), Seconds(3) };
  const std::vector<Seconds> samples = {
    Seconds(0), Seconds(.13), Seconds(.41), Seconds(1.4), Seconds(2.4), Seconds(3)
  };
  Eigen::MatrixXd positions(7, 2);
  positions << .1, -.2, -.2, .3, .4, .2, .25, -.1, .7, .5, .3, .8, .9, .2;
  const Eigen::Vector2d v0(.35, -.1), a0(-.18, .23), v1(-.22, .15), a1(.27, -.11);
  const auto fixed = QuinticSplineModel::build(times, v0, a0, v1, a1);
  for (bool free_start : { false, true })
  {
    const auto free =
        QuinticSplineModel::buildVariableBoundary(times,
                                                  2,
                                                  free_start ? std::nullopt : std::optional<Eigen::VectorXd>(v0),
                                                  free_start ? std::nullopt : std::optional<Eigen::VectorXd>(a0),
                                                  free_start ? std::optional<Eigen::VectorXd>(v1) : std::nullopt,
                                                  free_start ? std::optional<Eigen::VectorXd>(a1) : std::nullopt);
    ASSERT_EQ(free.variableCount(), 9);
    Eigen::MatrixXd variables(9, 2);
    variables.topRows(7) = positions;
    variables.row(7) = free_start ? v0.transpose() : v1.transpose();
    variables.row(8) = free_start ? a0.transpose() : a1.transpose();
    for (unsigned order = 0; order <= 3; ++order)
    {
      const auto map = free.map(samples, order);
      const auto reference = fixed.map(samples, order);
      EXPECT_TRUE((map.basis * variables + map.offset).isApprox(reference.basis * positions + reference.offset, 1e-10));
      for (Eigen::Index col = 7; col < 9; ++col)
      {
        constexpr double STEP = 1e-6;
        Eigen::MatrixXd plus = variables, minus = variables;
        plus(col, 0) += STEP;
        minus(col, 0) -= STEP;
        const Eigen::VectorXd numeric =
            ((map.basis * plus + map.offset) - (map.basis * minus + map.offset)).col(0) / (2 * STEP);
        EXPECT_TRUE(numeric.isApprox(map.basis.col(col), 1e-6));
      }
    }
    const auto batch = free.evaluate(samples, variables, 3);
    EXPECT_TRUE(batch[1].row(free_start ? 0 : 5).isApprox((free_start ? v0 : v1).transpose(), 1e-10));
    EXPECT_TRUE(batch[2].row(free_start ? 0 : 5).isApprox((free_start ? a0 : a1).transpose(), 1e-10));
  }
}

TEST(QuinticSplineFreeBoundary, AllBoundaryVariablesAndInvalidAxisCount)
{
  const std::vector<Seconds> times = { Seconds(0),   Seconds(.2),  Seconds(.65), Seconds(1.1),
                                       Seconds(1.7), Seconds(2.3), Seconds(3) };
  const auto free =
      QuinticSplineModel::buildVariableBoundary(times, 2, std::nullopt, std::nullopt, std::nullopt, std::nullopt);
  EXPECT_EQ(free.variableCount(), 11);
  EXPECT_EQ(free.axisCount(), 2);
  EXPECT_THROW(
      QuinticSplineModel::buildVariableBoundary(times, 0, std::nullopt, std::nullopt, std::nullopt, std::nullopt),
      trajopt_ifopt::QuinticSplineError);
  EXPECT_THROW(QuinticSplineModel::buildVariableBoundary(
                   times, 2, Eigen::Vector3d::Zero(), std::nullopt, std::nullopt, std::nullopt),
               trajopt_ifopt::QuinticSplineError);
}
