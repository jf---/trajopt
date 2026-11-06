#include <trajopt_ifopt/spline/monotone_fixture_axis.h>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <vector>

using trajopt_ifopt::MonotoneFixtureAxis;
using trajopt_ifopt::MonotoneFixtureAxisError;
using trajopt_ifopt::Seconds;

namespace
{
// Binary64 numerical comparison for position and derivatives in the test units.
constexpr double TEST_TOLERANCE = 1e-10;
// Time step in seconds for independent central differences inside one stage.
constexpr double DIFFERENCE_STEP_S = 1e-5;
// Truncation error of the central-difference derivative tests.
constexpr double DERIVATIVE_TOLERANCE = 1e-6;
// The one-sided acceleration samples straddle the knot by DIFFERENCE_STEP_S.
constexpr double JOIN_NEIGHBOR_ACCELERATION_TOLERANCE = 1e-2;
}  // namespace

TEST(MonotoneFixtureAxis, PreservesTargetsAndCarriesVelocityThroughAlignedJoin)
{
  const auto axis = MonotoneFixtureAxis::build({ Seconds(0), Seconds(1), Seconds(3) }, { 0, 1, 2 });
  EXPECT_DOUBLE_EQ(axis.evaluate(Seconds(0)).position, 0);
  const auto join = axis.evaluate(Seconds(1));
  EXPECT_DOUBLE_EQ(join.position, 1);
  EXPECT_NEAR(join.velocity, 0.5, TEST_TOLERANCE);
  EXPECT_NEAR(join.acceleration, 0, TEST_TOLERANCE);
  EXPECT_DOUBLE_EQ(axis.evaluate(Seconds(3)).position, 2);
  EXPECT_NEAR(axis.evaluate(Seconds(0)).velocity, 0, TEST_TOLERANCE);
  EXPECT_NEAR(axis.evaluate(Seconds(3)).velocity, 0, TEST_TOLERANCE);
  const double h = DIFFERENCE_STEP_S;
  EXPECT_NEAR(axis.evaluate(Seconds(1 - h)).velocity, axis.evaluate(Seconds(1 + h)).velocity, DERIVATIVE_TOLERANCE);
  EXPECT_NEAR(axis.evaluate(Seconds(1 - h)).acceleration,
              axis.evaluate(Seconds(1 + h)).acceleration,
              JOIN_NEIGHBOR_ACCELERATION_TOLERANCE);
  for (int i = 0; i <= 300; ++i)
  {
    const auto sample = axis.evaluate(Seconds(3.0 * i / 300));
    EXPECT_GE(sample.velocity, -TEST_TOLERANCE);
    EXPECT_GE(sample.position, -TEST_TOLERANCE);
    EXPECT_LE(sample.position, 2 + TEST_TOLERANCE);
  }
}

TEST(MonotoneFixtureAxis, StopsOnlyAtReversalAndHold)
{
  const auto axis = MonotoneFixtureAxis::build({ Seconds(0), Seconds(1), Seconds(2), Seconds(4) }, { 0, 2, 1, 1 });
  EXPECT_NEAR(axis.evaluate(Seconds(1)).velocity, 0, TEST_TOLERANCE);
  EXPECT_NEAR(axis.evaluate(Seconds(2)).velocity, 0, TEST_TOLERANCE);
  for (int i = 0; i <= 100; ++i)
  {
    EXPECT_GE(axis.evaluate(Seconds(i / 100.0)).velocity, -TEST_TOLERANCE);
    EXPECT_LE(axis.evaluate(Seconds(1 + i / 100.0)).velocity, TEST_TOLERANCE);
    const auto hold = axis.evaluate(Seconds(2 + 2 * i / 100.0));
    EXPECT_NEAR(hold.position, 1, TEST_TOLERANCE);
    EXPECT_NEAR(hold.velocity, 0, TEST_TOLERANCE);
    EXPECT_NEAR(hold.acceleration, 0, TEST_TOLERANCE);
    EXPECT_NEAR(hold.jerk, 0, TEST_TOLERANCE);
  }
}

TEST(MonotoneFixtureAxis, DerivativesMatchIndependentFiniteDifferences)
{
  const auto axis = MonotoneFixtureAxis::build({ Seconds(0), Seconds(0.7), Seconds(2.4) }, { 0, 0.4, 1.7 });
  const double h = DIFFERENCE_STEP_S;
  const auto left = axis.evaluate(Seconds(0.32 - h));
  const auto mid = axis.evaluate(Seconds(0.32));
  const auto right = axis.evaluate(Seconds(0.32 + h));
  EXPECT_NEAR((right.position - left.position) / (2 * h), mid.velocity, DERIVATIVE_TOLERANCE);
  EXPECT_NEAR((right.velocity - left.velocity) / (2 * h), mid.acceleration, DERIVATIVE_TOLERANCE);
  EXPECT_NEAR((right.acceleration - left.acceleration) / (2 * h), mid.jerk, DERIVATIVE_TOLERANCE);
}

TEST(MonotoneFixtureAxis, RejectsInvalidClockTargetsAndEvaluation)
{
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(MonotoneFixtureAxis::build({ Seconds(0) }, { 1 }), MonotoneFixtureAxisError);
  EXPECT_THROW(MonotoneFixtureAxis::build({ Seconds(0), Seconds(1) }, { 1 }), MonotoneFixtureAxisError);
  EXPECT_THROW(MonotoneFixtureAxis::build({ Seconds(0), Seconds(0) }, { 1, 2 }), MonotoneFixtureAxisError);
  EXPECT_THROW(MonotoneFixtureAxis::build({ Seconds(1), Seconds(0) }, { 1, 2 }), MonotoneFixtureAxisError);
  EXPECT_THROW(MonotoneFixtureAxis::build({ Seconds(0), Seconds(inf) }, { 1, 2 }), MonotoneFixtureAxisError);
  EXPECT_THROW(MonotoneFixtureAxis::build({ Seconds(0), Seconds(1) }, { 1, nan }), MonotoneFixtureAxisError);
  const auto axis = MonotoneFixtureAxis::build({ Seconds(1), Seconds(2) }, { 1, 2 });
  EXPECT_THROW(axis.evaluate(Seconds(0)), MonotoneFixtureAxisError);
  EXPECT_THROW(axis.evaluate(Seconds(3)), MonotoneFixtureAxisError);
  EXPECT_THROW(axis.evaluate(Seconds(nan)), MonotoneFixtureAxisError);
}
