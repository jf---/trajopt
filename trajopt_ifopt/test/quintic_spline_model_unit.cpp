#include <trajopt_ifopt/spline/quintic_spline_model.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <vector>

using trajopt_ifopt::QuinticSplineError;
using trajopt_ifopt::QuinticSplineModel;
using trajopt_ifopt::Seconds;

namespace
{
std::vector<Seconds> knotTimes()
{
  return { Seconds(0.0), Seconds(0.2), Seconds(0.65), Seconds(1.1), Seconds(1.7), Seconds(2.3), Seconds(3.0) };
}

Eigen::MatrixXd positions()
{
  Eigen::MatrixXd x(7, 1);
  x << 0.1, -0.2, 0.4, 0.25, 0.7, 0.3, 0.9;
  return x;
}

QuinticSplineModel model()
{
  return QuinticSplineModel::build(knotTimes(),
                                   Eigen::VectorXd::Constant(1, 0.35),
                                   Eigen::VectorXd::Constant(1, -0.18),
                                   Eigen::VectorXd::Constant(1, -0.22),
                                   Eigen::VectorXd::Constant(1, 0.27));
}
}  // namespace

TEST(QuinticSplineModel, MatchesScipyNonuniformClampedQuintic)
{
  const auto spline = model();
  const std::vector<Seconds> times = { Seconds(0.0), Seconds(0.13), Seconds(0.41), Seconds(1.0),
                                       Seconds(1.4), Seconds(2.4),  Seconds(3.0) };
  // scipy.interpolate.make_interp_spline(k=5), endpoint v=(.35,-.22), a=(-.18,.27).
  const std::array<std::array<double, 7>, 4> expected = { { { { 0.1,
                                                                -0.012050471224162723,
                                                                -0.33001187556647288,
                                                                0.40109077666431747,
                                                                0.30317573364251421,
                                                                0.35655220128521609,
                                                                0.9 } },
                                                            { { 0.35,
                                                                -2.3707812045706911,
                                                                1.9044133419543046,
                                                                -1.6596909738820338,
                                                                1.3995309717741025,
                                                                0.9358375086281473,
                                                                -0.22 } },
                                                            { { -0.18,
                                                                -17.077850589744493,
                                                                21.404110741881283,
                                                                0.14090331999435851,
                                                                4.6431804502602469,
                                                                6.6922215232769648,
                                                                0.27 } },
                                                            { { -820.50975880257306,
                                                                314.88970834817167,
                                                                -113.50762565407391,
                                                                91.55203645552028,
                                                                -50.38573592025436,
                                                                -26.992685180084752,
                                                                81.157447006956446 } } } };
  for (unsigned order = 0; order < expected.size(); ++order)
  {
    auto map = spline.map(times, order);
    Eigen::MatrixXd actual = map.basis * positions() + map.offset;
    for (std::size_t i = 0; i < times.size(); ++i)
      EXPECT_NEAR(actual(static_cast<Eigen::Index>(i), 0), expected[order][i], 2e-8);
  }
  auto jerk = spline.jerkQuadrature();
  Eigen::VectorXd residual = (jerk.map.basis * positions() + jerk.map.offset).col(0);
  EXPECT_NEAR((jerk.weights.array() * residual.array().square()).sum(), 39418.052631782746, 1e-5);
}

TEST(QuinticSplineModel, AnalyticJacobiansMatchFiniteDifferences)
{
  const auto spline = model();
  const std::vector<Seconds> times = { Seconds(0.11), Seconds(0.77), Seconds(2.71) };
  const double h = 1e-6;
  for (unsigned order = 0; order <= 3; ++order)
  {
    auto map = spline.map(times, order);
    for (Eigen::Index k = 0; k < positions().rows(); ++k)
    {
      auto plus = positions();
      plus(k, 0) += h;
      auto minus = positions();
      minus(k, 0) -= h;
      Eigen::VectorXd numerical = ((map.basis * plus + map.offset) - (map.basis * minus + map.offset)).col(0) / (2 * h);
      EXPECT_TRUE(numerical.isApprox(map.basis.col(k), 1e-6));
    }
  }
  const auto jerk = spline.jerkQuadrature();
  EXPECT_EQ(jerk.weights.size(), 18);
  EXPECT_TRUE((jerk.weights.array() > 0).all());
  EXPECT_NEAR(jerk.weights.sum(), 3.0, 1e-13);
}

TEST(QuinticSplineModel, ReproducesTwoAxisQuinticOnShiftedClock)
{
  auto times = knotTimes();
  for (auto& t : times)
    t += Seconds(2.0);
  const auto q = [](double t) { return std::pow(t, 5) - 2 * std::pow(t, 3) + 3 * t * t + 4 * t + 1; };
  const auto v = [](double t) { return 5 * std::pow(t, 4) - 6 * t * t + 6 * t + 4; };
  const auto a = [](double t) { return 20 * std::pow(t, 3) - 12 * t + 6; };
  const auto j = [](double t) { return 60 * t * t - 12; };
  Eigen::MatrixXd knots(7, 2);
  for (Eigen::Index i = 0; i < knots.rows(); ++i)
  {
    const double value = q(times[static_cast<std::size_t>(i)].count() - 2.0);
    knots(i, 0) = value;
    knots(i, 1) = -2 * value + 0.75;
  }
  Eigen::Vector2d v0(v(0), -2 * v(0));
  Eigen::Vector2d a0(a(0), -2 * a(0));
  Eigen::Vector2d v1(v(3), -2 * v(3));
  Eigen::Vector2d a1(a(3), -2 * a(3));
  const auto spline = QuinticSplineModel::build(times, v0, a0, v1, a1);
  const std::vector<Seconds> query = { Seconds(2.0), Seconds(2.4), Seconds(3.3), Seconds(5.0) };
  for (unsigned order = 0; order < 4; ++order)
  {
    const auto affine = spline.map(query, order);
    const Eigen::MatrixXd values = affine.basis * knots + affine.offset;
    for (std::size_t i = 0; i < query.size(); ++i)
    {
      const double t = query[i].count() - 2.0;
      const double expected = order == 0 ? q(t) : order == 1 ? v(t) : order == 2 ? a(t) : j(t);
      EXPECT_NEAR(values(static_cast<Eigen::Index>(i), 0), expected, 2e-8);
      EXPECT_NEAR(values(static_cast<Eigen::Index>(i), 1), -2 * expected + (order == 0 ? 0.75 : 0.0), 4e-8);
    }
  }
  const auto quadrature = spline.jerkQuadrature();
  const Eigen::MatrixXd jerks = quadrature.map.basis * knots + quadrature.map.offset;
  EXPECT_NEAR((quadrature.weights.array() * jerks.col(0).array().square()).sum(), 162432.0, 1e-5);
  EXPECT_NEAR((quadrature.weights.array() * jerks.col(1).array().square()).sum(), 649728.0, 4e-5);
}

TEST(QuinticSplineModel, RejectsInvalidInputs)
{
  auto bad = knotTimes();
  bad[2] = bad[1];
  EXPECT_THROW(
      QuinticSplineModel::build(
          bad, Eigen::VectorXd::Zero(1), Eigen::VectorXd::Zero(1), Eigen::VectorXd::Zero(1), Eigen::VectorXd::Zero(1)),
      QuinticSplineError);
  EXPECT_THROW(model().map({ Seconds(-0.01) }, 0), QuinticSplineError);
  EXPECT_THROW(model().map({ Seconds(0.0) }, 6), QuinticSplineError);
  EXPECT_THROW(QuinticSplineModel::build(knotTimes(),
                                         Eigen::VectorXd::Constant(1, std::numeric_limits<double>::quiet_NaN()),
                                         Eigen::VectorXd::Zero(1),
                                         Eigen::VectorXd::Zero(1),
                                         Eigen::VectorXd::Zero(1)),
               QuinticSplineError);
}
