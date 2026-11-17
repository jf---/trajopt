#include <trajopt_common/macros.h>
TRAJOPT_IGNORE_WARNINGS_PUSH
#include <algorithm>
#include <random>
#include <set>
#include <tuple>
#include <gtest/gtest.h>
#include <omp.h>
#include <tesseract/common/resource_locator.h>
#include <tesseract/environment/environment.h>
#include <tesseract/collision/discrete_contact_manager.h>
#include <tesseract/collision/continuous_contact_manager.h>
#include <tesseract/kinematics/joint_group.h>
TRAJOPT_IGNORE_WARNINGS_POP

#include <trajopt/collision_terms.hpp>
#include <trajopt/contact_manager_pool.hpp>
#include <trajopt/problem_description.hpp>
#include <trajopt/utils.hpp>
#include <trajopt_sco/optimizers.hpp>
#include <trajopt_common/collision_types.h>
#include "trajopt_test_utils.hpp"

using namespace trajopt;
using namespace tesseract::environment;
using namespace tesseract::collision;
using namespace tesseract::common;

// Collision margin (m); large against the spherebot obstacle so random states produce contacts.
static const double TEST_MARGIN = 0.2;
// Evaluators per A/B run: enough that one shared manager serves many evaluators' call histories.
static const int NUM_EVALUATORS = 20;
// Evaluation rounds per evaluator: each round is a new state, so every manager is reused across evaluators.
static const int NUM_ROUNDS = 3;
// Fixed seed so a failure reproduces.
static const unsigned SEED = 20261101U;
// Half-width of the sampled joint box (m): spherebot's obstacles (r = 0.5) sit within 0.75 m of the origin, so
// states within 1.5 m reach them; the joint limits (+-20 m) would almost never.
static const double STATE_HALF_WIDTH = 1.5;
// Timesteps of the term-level problem: the waypoint count of the puzzle piece auxiliary axes example.
static const int TERM_TEST_STEPS = 437;
// SQP iterations for the term-level problem: one iteration convexifies and evaluates every term.
static const int TERM_TEST_MAX_ITER = 1;

namespace
{
using ContactKey = std::tuple<std::string, std::string, int, int, int, int, double>;

ContactKey key(const ContactResult& r)
{
  return { r.link_names[0], r.link_names[1], r.shape_id[0], r.shape_id[1], r.subshape_id[0], r.subshape_id[1],
           r.distance };
}

/** @brief Contacts in a canonical order, so that only the broadphase visiting order may differ between A and B */
ContactResultVector sorted(const ContactResultMap& m)
{
  ContactResultVector v;
  m.flattenCopyResults(v);
  std::sort(v.begin(), v.end(), [](const ContactResult& a, const ContactResult& b) { return key(a) < key(b); });
  return v;
}

/** @brief Bitwise comparison of the fields trajopt reads from a contact */
void expectSameContacts(const ContactResultMap& a, const ContactResultMap& b)
{
  const ContactResultVector va = sorted(a);
  const ContactResultVector vb = sorted(b);
  ASSERT_EQ(va.size(), vb.size());
  for (std::size_t i = 0; i < va.size(); ++i)
  {
    EXPECT_EQ(key(va[i]), key(vb[i]));
    EXPECT_EQ(va[i].normal, vb[i].normal);
    for (std::size_t j = 0; j < 2; ++j)
    {
      EXPECT_EQ(va[i].nearest_points[j], vb[i].nearest_points[j]);
      EXPECT_EQ(va[i].nearest_points_local[j], vb[i].nearest_points_local[j]);
      EXPECT_EQ(va[i].cc_time[j], vb[i].cc_time[j]);
      EXPECT_EQ(va[i].cc_type[j], vb[i].cc_type[j]);
    }
  }
}
}  // namespace

class CollisionEvaluatorPoolTest : public testing::Test
{
public:
  Environment::Ptr env_ = std::make_shared<Environment>();
  tesseract::kinematics::JointGroup::ConstPtr manip_;

  void SetUp() override
  {
    const std::filesystem::path urdf_file(std::string(TRAJOPT_DATA_DIR) + "/spherebot.urdf");
    const std::filesystem::path srdf_file(std::string(TRAJOPT_DATA_DIR) + "/spherebot.srdf");
    const ResourceLocator::Ptr locator = std::make_shared<GeneralResourceLocator>();
    ASSERT_TRUE(env_->init(urdf_file, srdf_file, locator));
    manip_ = env_->getJointGroup("manipulator");
  }

  Eigen::VectorXd randomState(std::mt19937& rng) const
  {
    const Eigen::MatrixX2d& limits = manip_->getLimits().joint_limits;
    Eigen::VectorXd q(limits.rows());
    for (Eigen::Index i = 0; i < limits.rows(); ++i)
      q(i) = std::uniform_real_distribution<double>(std::max(limits(i, 0), -STATE_HALF_WIDTH),
                                                    std::min(limits(i, 1), STATE_HALF_WIDTH))(rng);
    return q;
  }

  trajopt_common::TrajOptCollisionConfig config(CollisionEvaluatorType type) const
  {
    trajopt_common::TrajOptCollisionConfig c(TEST_MARGIN, 1.0);
    c.collision_check_config.type = type;
    return c;
  }

  /**
   * @brief Evaluate the same states with A (one pool shared by all evaluators) and B (one pool per evaluator, the
   * ownership upstream had before pooling), visiting the evaluators in a shuffled interleaved order.
   */
  template <typename EvalT, typename MakeA, typename MakeB, typename Calc>
  void runAB(MakeA make_a, MakeB make_b, Calc calc)
  {
    std::mt19937 rng(SEED);
    std::vector<std::shared_ptr<EvalT>> a, b;
    for (int i = 0; i < NUM_EVALUATORS; ++i)
    {
      a.push_back(make_a(i));
      b.push_back(make_b(i));
    }

    std::vector<int> order;
    for (int r = 0; r < NUM_ROUNDS; ++r)
      for (int i = 0; i < NUM_EVALUATORS; ++i)
        order.push_back(i);
    std::shuffle(order.begin(), order.end(), rng);

    std::size_t total_contacts = 0;
    for (const int i : order)
    {
      const Eigen::VectorXd q0 = randomState(rng);
      const Eigen::VectorXd q1 = randomState(rng);
      ContactResultMap ra, rb;
      calc(*a[static_cast<std::size_t>(i)], q0, q1, ra);
      calc(*b[static_cast<std::size_t>(i)], q0, q1, rb);
      expectSameContacts(ra, rb);
      total_contacts += ra.count();
    }
    EXPECT_GT(total_contacts, 0U) << "no contacts: the comparison is vacuous";
  }
};

TEST_F(CollisionEvaluatorPoolTest, ab_equivalence_discrete)  // NOLINT
{
  const auto cfg = config(CollisionEvaluatorType::DISCRETE);
  auto shared = DiscreteContactManagerPool::create(*env_, manip_->getActiveLinkIds(), cfg);
  auto make = [&](const DiscreteContactManagerPool::Ptr& pool, int i) {
    // Every other evaluator takes the dynamic environment path
    return std::make_shared<SingleTimestepCollisionEvaluator>(
        manip_, env_, cfg, pool, sco::VarVector{}, CollisionExpressionEvaluatorType::SINGLE_TIME_STEP, i % 2 == 1);
  };
  runAB<SingleTimestepCollisionEvaluator>(
      [&](int i) { return make(shared, i); },
      [&](int i) { return make(DiscreteContactManagerPool::create(*env_, manip_->getActiveLinkIds(), cfg), i); },
      [](SingleTimestepCollisionEvaluator& e, const Eigen::VectorXd& q0, const Eigen::VectorXd&, ContactResultMap& r) {
        e.CalcCollisions(q0, r);
      });
}

TEST_F(CollisionEvaluatorPoolTest, ab_equivalence_lvs_discrete)  // NOLINT
{
  const auto cfg = config(CollisionEvaluatorType::LVS_DISCRETE);
  auto shared = DiscreteContactManagerPool::create(*env_, manip_->getActiveLinkIds(), cfg);
  auto make = [&](const DiscreteContactManagerPool::Ptr& pool) {
    return std::make_shared<DiscreteCollisionEvaluator>(manip_,
                                                        env_,
                                                        cfg,
                                                        pool,
                                                        sco::VarVector{},
                                                        sco::VarVector{},
                                                        CollisionExpressionEvaluatorType::START_FREE_END_FREE);
  };
  runAB<DiscreteCollisionEvaluator>(
      [&](int) { return make(shared); },
      [&](int) { return make(DiscreteContactManagerPool::create(*env_, manip_->getActiveLinkIds(), cfg)); },
      [](DiscreteCollisionEvaluator& e, const Eigen::VectorXd& q0, const Eigen::VectorXd& q1, ContactResultMap& r) {
        e.CalcCollisions(q0, q1, r);
      });
}

TEST_F(CollisionEvaluatorPoolTest, ab_equivalence_cast)  // NOLINT
{
  // LVS_CONTINUOUS with random endpoints: segments far longer than the longest valid segment, so the evaluator
  // casts several sub-segments per call
  const auto cfg = config(CollisionEvaluatorType::LVS_CONTINUOUS);
  auto shared = ContinuousContactManagerPool::create(*env_, manip_->getActiveLinkIds(), cfg);
  auto make = [&](const ContinuousContactManagerPool::Ptr& pool) {
    return std::make_shared<CastCollisionEvaluator>(manip_,
                                                    env_,
                                                    cfg,
                                                    pool,
                                                    sco::VarVector{},
                                                    sco::VarVector{},
                                                    CollisionExpressionEvaluatorType::START_FREE_END_FREE);
  };
  runAB<CastCollisionEvaluator>(
      [&](int) { return make(shared); },
      [&](int) { return make(ContinuousContactManagerPool::create(*env_, manip_->getActiveLinkIds(), cfg)); },
      [](CastCollisionEvaluator& e, const Eigen::VectorXd& q0, const Eigen::VectorXd& q1, ContactResultMap& r) {
        e.CalcCollisions(q0, q1, r);
      });
}

TEST_F(CollisionEvaluatorPoolTest, null_pool_throws)  // NOLINT
{
  const auto discrete = config(CollisionEvaluatorType::DISCRETE);
  const auto lvs = config(CollisionEvaluatorType::LVS_DISCRETE);
  const auto cast = config(CollisionEvaluatorType::LVS_CONTINUOUS);
  const auto free = CollisionExpressionEvaluatorType::START_FREE_END_FREE;
  EXPECT_THROW(SingleTimestepCollisionEvaluator(  // NOLINT
                   manip_,
                   env_,
                   discrete,
                   nullptr,
                   sco::VarVector{},
                   CollisionExpressionEvaluatorType::SINGLE_TIME_STEP),
               std::invalid_argument);
  EXPECT_THROW(  // NOLINT
      DiscreteCollisionEvaluator(manip_, env_, lvs, nullptr, sco::VarVector{}, sco::VarVector{}, free),
      std::invalid_argument);
  EXPECT_THROW(  // NOLINT
      CastCollisionEvaluator(manip_, env_, cast, nullptr, sco::VarVector{}, sco::VarVector{}, free),
      std::invalid_argument);
}

/** @brief A collision term over TERM_TEST_STEPS timesteps, solved multi-threaded */
class CollisionTermPoolTest : public testing::TestWithParam<CollisionEvaluatorType>
{
public:
  Environment::Ptr env_ = std::make_shared<Environment>();

  void SetUp() override
  {
    const std::filesystem::path urdf_file(std::string(TRAJOPT_DATA_DIR) + "/arm_around_table.urdf");
    const std::filesystem::path srdf_file(std::string(TRAJOPT_DATA_DIR) + "/pr2.srdf");
    const ResourceLocator::Ptr locator = std::make_shared<GeneralResourceLocator>();
    ASSERT_TRUE(env_->init(urdf_file, srdf_file, locator));
  }
};

TEST_P(CollisionTermPoolTest, term_shares_one_pool)  // NOLINT
{
  Json::Value root = readJsonFile(std::string(TRAJOPT_DATA_DIR) + "/config/arm_around_table.json");
  root["basic_info"]["n_steps"] = TERM_TEST_STEPS;
  root["init_info"] = Json::Value(Json::objectValue);
  root["init_info"]["type"] = "stationary";
  root["costs"][1]["params"]["evaluator_type"] = static_cast<int>(GetParam());
  root["costs"][1]["params"]["fixed_steps"] = Json::Value(Json::arrayValue);
  root["costs"][1]["params"]["fixed_steps"].append(0);
  root["constraints"][0]["params"]["first_step"] = TERM_TEST_STEPS - 1;
  root["constraints"][0]["params"]["last_step"] = TERM_TEST_STEPS - 1;

  std::unordered_map<std::string, double> ipos;
  ipos["torso_lift_joint"] = 0;
  ipos["r_shoulder_pan_joint"] = -1.832;
  ipos["r_shoulder_lift_joint"] = -0.332;
  ipos["r_upper_arm_roll_joint"] = -1.011;
  ipos["r_elbow_flex_joint"] = -1.437;
  ipos["r_forearm_roll_joint"] = -1.1;
  ipos["r_wrist_flex_joint"] = -1.926;
  ipos["r_wrist_roll_joint"] = 3.074;
  env_->setState(ipos);

  ProblemConstructionInfo pci(env_);
  pci.fromJson(root);
  pci.basic_info.convex_solver = sco::ModelType::OSQP;
  const TrajOptProb::Ptr prob = ConstructProblem(pci);
  ASSERT_TRUE(!!prob);

  auto opt = std::make_shared<sco::BasicTrustRegionSQPMultiThreaded>(prob);
  opt->getParameters().num_threads = omp_get_max_threads();
  opt->getParameters().max_iter = TERM_TEST_MAX_ITER;
  opt->initialize(trajToDblVec(prob->GetInitTraj()));
  opt->optimize();

  std::set<const CollisionEvaluator*> evaluators;
  std::set<std::size_t> manager_counts;
  for (const auto& cost : prob->getCosts())
  {
    auto c = std::dynamic_pointer_cast<CollisionCost>(cost);
    if (c == nullptr)
      continue;
    evaluators.insert(c->getEvaluator().get());
    manager_counts.insert(c->getEvaluator()->numContactManagers());
  }
  // One evaluator per step (DISCRETE: every non-fixed step; continuous: every segment)
  EXPECT_GE(evaluators.size(), static_cast<std::size_t>(TERM_TEST_STEPS - 1));
  // All evaluators of the term report the same pool size, because they share one pool
  ASSERT_EQ(manager_counts.size(), 1U);
  // prototype + one manager per OpenMP thread
  const std::size_t max_managers_per_term = static_cast<std::size_t>(omp_get_max_threads()) + 1;
  EXPECT_LE(*manager_counts.begin(), max_managers_per_term);
  EXPECT_GE(*manager_counts.begin(), 2U);
}

INSTANTIATE_TEST_SUITE_P(EvaluatorTypes,  // NOLINT
                         CollisionTermPoolTest,
                         testing::Values(CollisionEvaluatorType::DISCRETE,
                                         CollisionEvaluatorType::LVS_DISCRETE,
                                         CollisionEvaluatorType::LVS_CONTINUOUS));

int main(int argc, char** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
