#include <gtest/gtest.h>
#include <trajopt_common/collision_types.h>
#include <trajopt_ifopt/constraints/collision/weighted_average_methods.h>

namespace
{
using trajopt_common::GradientResults;
using trajopt_common::GradientResultsSet;
using tesseract::collision::ContinuousCollisionType;

// Synthetic unitless gradients use exactly representable inputs.
constexpr double GRADIENT_TOLERANCE = 1e-12;

GradientResults contact(double error, bool two_active)
{
  GradientResults result;
  result.error_with_buffer = error;
  for (std::size_t link = 0; link < (two_active ? 2U : 1U); ++link)
  {
    for (auto* gradient : { &result.gradients[link], &result.cc_gradients[link] })
    {
      gradient->has_gradient = true;
      gradient->gradient = Eigen::VectorXd::Constant(1, link == 0 ? 1.0 : 2.0);
      gradient->scale = 1.0;
      gradient->cc_type = ContinuousCollisionType::CCType_Between;
    }
  }
  return result;
}

GradientResultsSet twoContacts(bool first_has_two, bool second_has_two)
{
  GradientResultsSet result;
  result.is_continuous = true;
  result.add(contact(2.0, first_has_two));
  auto second = contact(1.0, second_has_two);
  second.gradients[0].gradient[0] = 3.0;
  second.cc_gradients[0].gradient[0] = 3.0;
  if (second_has_two)
  {
    second.gradients[1].gradient[0] = 4.0;
    second.cc_gradients[1].gradient[0] = 4.0;
  }
  result.add(second);
  return result;
}
}  // namespace

TEST(CollisionPairWeight, BothActiveLinksCountOncePerContactAtBothEndpoints)
{
  auto result = twoContacts(true, true);
  const double expected = (3.0 + 0.5 * 7.0) / 1.5;
  EXPECT_NEAR(trajopt_ifopt::getWeightedAvgGradientT0(result, 2.0, 1)[0], expected, GRADIENT_TOLERANCE);
  EXPECT_NEAR(trajopt_ifopt::getWeightedAvgGradientT1(result, 2.0, 1)[0], expected, GRADIENT_TOLERANCE);
}

TEST(CollisionPairWeight, MixedActiveLinkCountDoesNotReweightContacts)
{
  auto result = twoContacts(true, false);
  const double expected = (3.0 + 0.5 * 3.0) / 1.5;
  EXPECT_NEAR(trajopt_ifopt::getWeightedAvgGradientT0(result, 2.0, 1)[0], expected, GRADIENT_TOLERANCE);
  EXPECT_NEAR(trajopt_ifopt::getWeightedAvgGradientT1(result, 2.0, 1)[0], expected, GRADIENT_TOLERANCE);
}

TEST(CollisionPairWeight, SingleActiveLinkPreservesExistingWeightedBehavior)
{
  auto result = twoContacts(false, false);
  const double expected = (1.0 + 0.5 * 3.0) / 1.5;
  EXPECT_NEAR(trajopt_ifopt::getWeightedAvgGradientT0(result, 2.0, 1)[0], expected, GRADIENT_TOLERANCE);
  EXPECT_NEAR(trajopt_ifopt::getWeightedAvgGradientT1(result, 2.0, 1)[0], expected, GRADIENT_TOLERANCE);
}

TEST(CollisionPairWeight, IgnoresContactsWithNoDerivativeAtTheChosenEndpoint)
{
  auto result = twoContacts(true, true);
  for (auto& gradient : result.results[1].gradients)
    gradient.cc_type = ContinuousCollisionType::CCType_Time1;
  for (auto& gradient : result.results[0].cc_gradients)
    gradient.cc_type = ContinuousCollisionType::CCType_Time0;
  EXPECT_NEAR(trajopt_ifopt::getWeightedAvgGradientT0(result, 2.0, 1)[0], 3.0, GRADIENT_TOLERANCE);
  EXPECT_NEAR(trajopt_ifopt::getWeightedAvgGradientT1(result, 2.0, 1)[0], 7.0, GRADIENT_TOLERANCE);
}
