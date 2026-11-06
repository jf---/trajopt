#include <trajopt_ifopt/spline/cubic_spline_stage.h>
#include <gtest/gtest.h>

#include <vector>

using trajopt_ifopt::CubicSplineStage;
using trajopt_ifopt::Seconds;

namespace
{
double p(double t) { return t * t * t - 2 * t * t + 3 * t + 1; }
double dp(double t) { return 3 * t * t - 4 * t + 3; }
double ddp(double t) { return 6 * t - 4; }
}  // namespace

TEST(CubicSplineStage, ReproducesNonuniformCubicAndAnalyticDerivatives)
{
  const std::vector<Seconds> knots = { Seconds(0), Seconds(.19), Seconds(.54), Seconds(1.3), Seconds(2.1) };
  const std::vector<Seconds> samples = { Seconds(0),   Seconds(.05), Seconds(.19),
                                         Seconds(.71), Seconds(1.9), Seconds(2.1) };
  const auto model = CubicSplineStage::build(knots, 2);
  ASSERT_EQ(model.variableCount(), 7);
  Eigen::MatrixXd variables(7, 2);
  for (int i = 0; i < 5; ++i)
    variables.row(i) << p(knots[static_cast<std::size_t>(i)].count()),
        2 * p(knots[static_cast<std::size_t>(i)].count());
  variables.row(5) << dp(0), 2 * dp(0);
  variables.row(6) << dp(2.1), 2 * dp(2.1);
  for (unsigned order = 0; order <= 3; ++order)
  {
    const auto map = model.map(samples, order);
    const Eigen::MatrixXd values = map.basis * variables + map.offset;
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(samples.size()); ++i)
    {
      const double t = samples[static_cast<std::size_t>(i)].count();
      const double expected = order == 0 ? p(t) : order == 1 ? dp(t) : order == 2 ? ddp(t) : 6.0;
      EXPECT_NEAR(values(i, 0), expected, 1e-10);
      EXPECT_NEAR(values(i, 1), 2 * expected, 1e-10);
    }
    constexpr double STEP = 1e-6;
    for (Eigen::Index column = 0; column < variables.rows(); ++column)
    {
      Eigen::MatrixXd plus = variables, minus = variables;
      plus(column, 0) += STEP;
      minus(column, 0) -= STEP;
      const Eigen::VectorXd numeric =
          ((map.basis * plus + map.offset) - (map.basis * minus + map.offset)).col(0) / (2 * STEP);
      EXPECT_TRUE(numeric.isApprox(map.basis.col(column), 1e-5));
    }
  }
}

TEST(CubicSplineStage, RejectsInvalidClockOrAxisCount)
{
  EXPECT_THROW(CubicSplineStage::build({ Seconds(0), Seconds(0) }, 1), trajopt_ifopt::CubicSplineStageError);
  EXPECT_THROW(CubicSplineStage::build({ Seconds(0), Seconds(1) }, 0), trajopt_ifopt::CubicSplineStageError);
  const auto model = CubicSplineStage::build({ Seconds(0), Seconds(1) }, 1);
  EXPECT_THROW(model.map({ Seconds(1.1) }, 0), trajopt_ifopt::CubicSplineStageError);
  EXPECT_THROW(model.map({ Seconds(.5) }, 4), trajopt_ifopt::CubicSplineStageError);
}
