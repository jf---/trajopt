#include <trajopt_common/macros.h>
TRAJOPT_IGNORE_WARNINGS_PUSH
#include <omp.h>
#include <type_traits>
#include <tesseract/collision/discrete_contact_manager.h>
#include <tesseract/collision/continuous_contact_manager.h>
#include <tesseract/environment/environment.h>
TRAJOPT_IGNORE_WARNINGS_POP

#include <trajopt/contact_manager_pool.hpp>

namespace trajopt
{
template <typename ManagerT>
typename ContactManagerPool<ManagerT>::Ptr
ContactManagerPool<ManagerT>::create(const tesseract::environment::Environment& env,
                                     const std::vector<std::string>& active_links,
                                     const trajopt_common::TrajOptCollisionConfig& collision_config)
{
  std::unique_ptr<ManagerT> prototype;
  if constexpr (std::is_same_v<ManagerT, tesseract::collision::DiscreteContactManager>)
    prototype = env.getDiscreteContactManager();
  else
    prototype = env.getContinuousContactManager();

  prototype->setActiveCollisionObjects(active_links);
  prototype->applyContactManagerConfig(collision_config.contact_manager_config);
  // Must make a copy after applying the config but before we increment the margin data
  tesseract::common::CollisionMarginData margin_data = prototype->getCollisionMarginData();

  // Increase the default by the buffer
  prototype->incrementCollisionMargin(collision_config.collision_margin_buffer);

  return std::make_shared<ContactManagerPool<ManagerT>>(std::move(prototype), std::move(margin_data));
}

template <typename ManagerT>
ContactManagerPool<ManagerT>::ContactManagerPool(std::unique_ptr<ManagerT> prototype,
                                                 tesseract::common::CollisionMarginData margin_data)
  : prototype_(std::move(prototype)), margin_data_(std::move(margin_data))
{
  if (prototype_ == nullptr)
    throw std::invalid_argument("ContactManagerPool, prototype contact manager is null");
}

template <typename ManagerT>
ManagerT& ContactManagerPool<ManagerT>::get()
{
  const int thread = omp_get_thread_num();
  const std::lock_guard<std::mutex> lock(mutex_);
  std::unique_ptr<ManagerT>& manager = managers_[thread];
  if (manager == nullptr)
    manager = prototype_->clone();

  return *manager;
}

template <typename ManagerT>
const tesseract::common::CollisionMarginData& ContactManagerPool<ManagerT>::getCollisionMarginData() const
{
  return margin_data_;
}

template <typename ManagerT>
std::size_t ContactManagerPool<ManagerT>::numManagers() const
{
  const std::lock_guard<std::mutex> lock(mutex_);
  return managers_.size() + 1;
}

template class ContactManagerPool<tesseract::collision::DiscreteContactManager>;
template class ContactManagerPool<tesseract::collision::ContinuousContactManager>;
}  // namespace trajopt
