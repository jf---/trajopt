#pragma once
#include <trajopt_common/macros.h>
TRAJOPT_IGNORE_WARNINGS_PUSH
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <tesseract/collision/fwd.h>
#include <tesseract/common/collision_margin_data.h>
#include <tesseract/environment/fwd.h>
TRAJOPT_IGNORE_WARNINGS_POP

#include <trajopt_common/collision_types.h>

namespace trajopt
{
/**
 * @brief One contact manager per OpenMP thread, shared by all collision evaluators of a term.
 *
 * Each evaluator used to clone the environment's contact manager, so a term over N timesteps held N full manager
 * clones for the whole solve. Every evaluation sets the transforms of all active links before its contact test, so
 * evaluators with the same configuration can share a manager as long as no two threads use it at once.
 *
 * The pool holds a configured prototype that is never used for contact tests, and hands each OpenMP thread its own
 * clone of it, created on that thread's first call to get(). The managers alive are therefore the prototype plus one
 * per thread that evaluated the term, independent of the number of timesteps.
 */
template <typename ManagerT>
class ContactManagerPool
{
public:
  using Ptr = std::shared_ptr<ContactManagerPool<ManagerT>>;
  using ConstPtr = std::shared_ptr<const ContactManagerPool<ManagerT>>;

  /**
   * @brief Clone the environment's contact manager and configure it as the pool's prototype
   * @param env The environment, cloned now: the pool sees the environment as of this call
   * @param active_links The links to set active
   * @param collision_config The contact manager config to apply and the margin buffer to add
   */
  static Ptr create(const tesseract::environment::Environment& env,
                    const std::vector<std::string>& active_links,
                    const trajopt_common::TrajOptCollisionConfig& collision_config);

  explicit ContactManagerPool(std::unique_ptr<ManagerT> prototype,
                              tesseract::common::CollisionMarginData margin_data);

  /** @brief The manager of the calling OpenMP thread, cloned from the prototype on that thread's first call */
  ManagerT& get();

  /** @brief The collision margin data after applying the config, before adding the margin buffer */
  const tesseract::common::CollisionMarginData& getCollisionMarginData() const;

  /** @brief The number of managers this pool holds: the prototype plus one per thread that called get() */
  std::size_t numManagers() const;

private:
  std::unique_ptr<const ManagerT> prototype_;
  tesseract::common::CollisionMarginData margin_data_;
  mutable std::mutex mutex_;
  std::map<int, std::unique_ptr<ManagerT>> managers_;
};

using DiscreteContactManagerPool = ContactManagerPool<tesseract::collision::DiscreteContactManager>;
using ContinuousContactManagerPool = ContactManagerPool<tesseract::collision::ContinuousContactManager>;
}  // namespace trajopt
