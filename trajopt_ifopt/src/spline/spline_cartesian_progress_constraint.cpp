#include <trajopt_ifopt/spline/spline_cartesian_progress_constraint.h>
#include <trajopt_ifopt/variable_sets/node.h>
#include <trajopt_ifopt/variable_sets/nodes_variables.h>
#include <trajopt_ifopt/variable_sets/var.h>
#include <tesseract/kinematics/joint_group.h>
#include <tesseract/common/utils.h>
#include <set>
#include <stdexcept>
#include <utility>

namespace trajopt_ifopt
{
SplineCartesianProgressConstraint::SplineCartesianProgressConstraint(CartLineInfo info,
                                                                     std::vector<std::shared_ptr<const Var>> knots,
                                                                     Eigen::MatrixXd basis,
                                                                     Eigen::MatrixXd offset,
                                                                     bool include_start,
                                                                     bool include_end,
                                                                     std::string name)
  : ConstraintSet(std::move(name), static_cast<int>(3 * basis.rows() - 1))
  , info_(std::move(info))
  , knots_(std::move(knots))
  , basis_(std::move(basis))
  , offset_(std::move(offset))
  , line_(info_.target_frame_offset2.translation() - info_.target_frame_offset1.translation())
  , line_squared_(line_.squaredNorm())
  , include_start_(include_start)
  , include_end_(include_end)
{
  if (!info_.manip || knots_.empty() || !knots_.front() || basis_.rows() < 2 ||
      basis_.cols() != static_cast<Eigen::Index>(knots_.size()) || offset_.rows() != basis_.rows() ||
      offset_.cols() != info_.manip->numJoints() || !basis_.allFinite() || !offset_.allFinite() ||
      !std::isfinite(line_squared_) || line_squared_ <= 0)
    throw std::invalid_argument("invalid spline progress line, samples or knots");
  const auto& names = knots_.front()->name();
  std::set<Eigen::Index> indices;
  for (const auto& knot : knots_)
  {
    if (!knot || !knot->getParent() || !knot->getParent()->getParent() || knot->size() != offset_.cols() ||
        knot->name() != names || knot->getIndex() < 0)
      throw std::invalid_argument("spline progress knot axes or ownership differ");
    for (Eigen::Index axis = 0; axis < knot->size(); ++axis)
      if (!indices.insert(knot->getIndex() + axis).second)
        throw std::invalid_argument("spline progress knot variable indices overlap");
  }
  non_zeros_ = static_cast<Eigen::Index>(rows_) * offset_.cols() * static_cast<Eigen::Index>(knots_.size());
}

void SplineCartesianProgressConstraint::initVariableDependedQuantities(const Variables::Ptr& variables)
{
  if (!variables)
    throw std::invalid_argument("spline progress requires linked variables");
  for (const auto& knot : knots_)
    if (knot->getParent()->getParent() != variables.get() || knot->getIndex() + knot->size() > variables->getRows())
      throw std::invalid_argument("spline progress knots do not belong to linked variables");
}

void SplineCartesianProgressConstraint::evaluate(Eigen::VectorXd& progress, Eigen::MatrixXd* gradients) const
{
  if (!variables_)
    throw std::logic_error("spline progress must be linked before evaluation");
  progress.resize(basis_.rows());
  if (gradients != nullptr)
    gradients->resize(basis_.rows(), offset_.cols());
  for (Eigen::Index i = 0; i < basis_.rows(); ++i)
  {
    Eigen::VectorXd q = offset_.row(i).transpose();
    for (std::size_t k = 0; k < knots_.size(); ++k)
      q.noalias() += basis_(i, static_cast<Eigen::Index>(k)) * knots_[k]->value();
    const auto transforms = info_.manip->calcFwdKin(q);
    const Eigen::Isometry3d source_link = transforms.at(info_.source_frame);
    const Eigen::Isometry3d source = source_link * info_.source_frame_offset;
    const Eigen::Isometry3d target = transforms.at(info_.target_frame);
    const Eigen::Vector3d relative = target.inverse() * source.translation();
    progress[i] = line_.dot(relative - info_.target_frame_offset1.translation()) / line_squared_;
    if (gradients != nullptr)
    {
      Eigen::MatrixXd twists = Eigen::MatrixXd::Zero(6, offset_.cols());
      if (info_.manip->isActiveLinkId(info_.source_frame))
      {
        twists = info_.manip->calcJacobian(q, info_.source_frame);
        tesseract::common::jacobianChangeRefPoint(twists, source.translation() - source_link.translation());
      }
      if (info_.manip->isActiveLinkId(info_.target_frame))
      {
        Eigen::MatrixXd target_twists = info_.manip->calcJacobian(q, info_.target_frame);
        tesseract::common::jacobianChangeRefPoint(target_twists, source.translation() - target.translation());
        twists -= target_twists;
      }
      gradients->row(i) = line_.transpose() * target.linear().transpose() * twists.topRows(3) / line_squared_;
    }
  }
}

Eigen::VectorXd SplineCartesianProgressConstraint::getValues() const
{
  Eigen::VectorXd progress;
  evaluate(progress, nullptr);
  Eigen::VectorXd result(rows_);
  result.head(progress.size()) = progress;
  result.segment(progress.size(), progress.size()) = progress;
  result.tail(progress.size() - 1) = progress.tail(progress.size() - 1) - progress.head(progress.size() - 1);
  return result;
}
std::vector<Bounds> SplineCartesianProgressConstraint::getBounds() const
{
  std::vector<Bounds> bounds(static_cast<std::size_t>(rows_), BoundGreaterZero);
  for (Eigen::Index i = 0; i < basis_.rows(); ++i)
    bounds[static_cast<std::size_t>(basis_.rows() + i)] = Bounds(-double(INFINITY), 1);
  if (include_start_)
    bounds.front() = BoundZero;
  if (include_end_)
    bounds[static_cast<std::size_t>(2 * basis_.rows() - 1)] = Bounds(1, 1);
  return bounds;
}
Eigen::VectorXd SplineCartesianProgressConstraint::getCoefficients() const { return Eigen::VectorXd::Ones(rows_); }
Jacobian SplineCartesianProgressConstraint::getJacobian() const
{
  Eigen::VectorXd progress;
  Eigen::MatrixXd gradients;
  evaluate(progress, &gradients);
  std::vector<Eigen::Triplet<double>> entries;
  entries.reserve(static_cast<std::size_t>(non_zeros_));
  for (Eigen::Index i = 0; i < basis_.rows(); ++i)
    for (std::size_t k = 0; k < knots_.size(); ++k)
      for (Eigen::Index axis = 0; axis < offset_.cols(); ++axis)
      {
        const Eigen::Index col = knots_[k]->getIndex() + axis;
        const double value = basis_(i, static_cast<Eigen::Index>(k)) * gradients(i, axis);
        entries.emplace_back(i, col, value);
        entries.emplace_back(basis_.rows() + i, col, value);
        if (i > 0)
          entries.emplace_back(2 * basis_.rows() + i - 1, col, value);
        if (i + 1 < basis_.rows())
          entries.emplace_back(2 * basis_.rows() + i, col, -value);
      }
  Jacobian result(rows_, variables_->getRows());
  result.setFromTriplets(entries.begin(), entries.end());
  return result;
}
}  // namespace trajopt_ifopt
