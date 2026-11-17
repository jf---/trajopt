#include <trajopt_common/macros.h>
TRAJOPT_IGNORE_WARNINGS_PUSH
#include <set>
#include <unordered_set>
#include <gtest/gtest.h>
#include <omp.h>
#include <tesseract/common/resource_locator.h>
#include <tesseract/environment/environment.h>
#include <tesseract/collision/discrete_contact_manager.h>
#include <tesseract/collision/continuous_contact_manager.h>
#include <tesseract/kinematics/joint_group.h>
TRAJOPT_IGNORE_WARNINGS_POP

#include <trajopt/contact_manager_pool.hpp>
#include <trajopt_common/collision_types.h>

using namespace trajopt;
using namespace tesseract::environment;
using namespace tesseract::collision;
using namespace tesseract::common;

// Collision margin of the test config (m); any positive value, the tests compare exact sums.
static const double TEST_MARGIN = 0.2;
// Margin buffer added on top of the margin for the check (m).
static const double TEST_MARGIN_BUFFER = 0.05;

class ContactManagerPoolTest : public testing::Test
{
public:
  Environment::Ptr env_ = std::make_shared<Environment>();
  std::vector<LinkId> active_links_;
  trajopt_common::TrajOptCollisionConfig config_{ TEST_MARGIN, 1.0 };

  void SetUp() override
  {
    const std::filesystem::path urdf_file(std::string(TRAJOPT_DATA_DIR) + "/spherebot.urdf");
    const std::filesystem::path srdf_file(std::string(TRAJOPT_DATA_DIR) + "/spherebot.srdf");
    const ResourceLocator::Ptr locator = std::make_shared<GeneralResourceLocator>();
    ASSERT_TRUE(env_->init(urdf_file, srdf_file, locator));
    active_links_ = env_->getJointGroup("manipulator")->getActiveLinkIds();
    config_.collision_margin_buffer = TEST_MARGIN_BUFFER;
  }
};

TEST_F(ContactManagerPoolTest, same_thread_same_manager)  // NOLINT
{
  auto pool = DiscreteContactManagerPool::create(*env_, active_links_, config_);
  EXPECT_EQ(pool->numManagers(), 1U);  // the prototype
  DiscreteContactManager& a = pool->get();
  DiscreteContactManager& b = pool->get();
  EXPECT_EQ(&a, &b);
  EXPECT_EQ(pool->numManagers(), 2U);
}

template <typename PoolT>
void checkOneManagerPerThread(const typename PoolT::Ptr& pool)
{
  // Twice the default team size: a planner may ask for more threads than omp_get_max_threads().
  const int n_threads = 2 * omp_get_max_threads();
  std::set<const void*> managers;
  int team_size = 0;
#pragma omp parallel num_threads(n_threads) shared(managers, team_size, pool)
  {
    const void* m = &pool->get();
#pragma omp critical
    {
      managers.insert(m);
      team_size = omp_get_num_threads();
    }
  }
  EXPECT_EQ(team_size, n_threads);
  EXPECT_EQ(managers.size(), static_cast<std::size_t>(n_threads));
  EXPECT_EQ(pool->numManagers(), static_cast<std::size_t>(n_threads) + 1);
}

TEST_F(ContactManagerPoolTest, one_manager_per_thread_discrete)  // NOLINT
{
  checkOneManagerPerThread<DiscreteContactManagerPool>(
      DiscreteContactManagerPool::create(*env_, active_links_, config_));
}

TEST_F(ContactManagerPoolTest, one_manager_per_thread_continuous)  // NOLINT
{
  config_.collision_check_config.type = CollisionEvaluatorType::CONTINUOUS;
  checkOneManagerPerThread<ContinuousContactManagerPool>(
      ContinuousContactManagerPool::create(*env_, active_links_, config_));
}

TEST_F(ContactManagerPoolTest, margin_data_before_buffer)  // NOLINT
{
  auto pool = DiscreteContactManagerPool::create(*env_, active_links_, config_);
  EXPECT_EQ(pool->getCollisionMarginData().getMaxCollisionMargin(), TEST_MARGIN);
  EXPECT_EQ(pool->get().getCollisionMarginData().getMaxCollisionMargin(), TEST_MARGIN + TEST_MARGIN_BUFFER);
}

TEST_F(ContactManagerPoolTest, configured_like_upstream)  // NOLINT
{
  auto pool = DiscreteContactManagerPool::create(*env_, active_links_, config_);
  const std::unordered_set<LinkId> expected(active_links_.begin(), active_links_.end());
  EXPECT_EQ(pool->get().getActiveCollisionObjects(), expected);
}

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
