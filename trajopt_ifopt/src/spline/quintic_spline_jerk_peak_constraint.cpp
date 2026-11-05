#include <trajopt_ifopt/spline/quintic_spline_jerk_peak_constraint.h>
#include <trajopt_ifopt/variable_sets/node.h>
#include <trajopt_ifopt/variable_sets/nodes_variables.h>
#include <trajopt_ifopt/variable_sets/var.h>
#include <cmath>
#include <set>
#include <utility>

namespace trajopt_ifopt
{
QuinticSplineJerkPeakConstraint::QuinticSplineJerkPeakConstraint(const QuinticSplineModel& model,
                                                                 std::vector<std::shared_ptr<const Var>> knots,
                                                                 std::shared_ptr<const Var> peak,
                                                                 std::vector<Seconds> witnesses,
                                                                 Eigen::VectorXd normalization,
                                                                 std::string name)
  : ConstraintSet(std::move(name),
                  static_cast<int>(2 * static_cast<Eigen::Index>(witnesses.size()) * model.axisCount()))
  , knots_(std::move(knots))
  , peak_(std::move(peak))
  , map_(model.map(witnesses, 3))
  , normalization_(std::move(normalization))
{
  if (witnesses.empty() || static_cast<Eigen::Index>(knots_.size()) != model.knotCount() ||
      normalization_.size() != model.axisCount() || !normalization_.allFinite() ||
      !(normalization_.array() > 0).all() || !peak_ || peak_->size() != model.axisCount() || !peak_->getParent() ||
      !peak_->getParent()->getParent() || peak_->getIndex() < 0)
    throw QuinticSplineError("invalid jerk peak witnesses, axis normalization or variables");
  std::set<Eigen::Index> indices;
  for (Eigen::Index axis = 0; axis < peak_->size(); ++axis)
    indices.insert(peak_->getIndex() + axis);
  for (const auto& knot : knots_)
  {
    if (!knot || !knot->getParent() || !knot->getParent()->getParent() || knot->size() != model.axisCount() ||
        knot->getIndex() < 0 || knot->name() != peak_->name())
      throw QuinticSplineError("jerk peak knot layout differs");
    for (Eigen::Index axis = 0; axis < knot->size(); ++axis)
      if (!indices.insert(knot->getIndex() + axis).second)
        throw QuinticSplineError("jerk peak variable indices overlap");
  }
}

void QuinticSplineJerkPeakConstraint::initVariableDependedQuantities(const Variables::Ptr& variables)
{
  if (!variables || peak_->getParent()->getParent() != variables.get() ||
      peak_->getIndex() + peak_->size() > variables->getRows())
    throw QuinticSplineError("jerk peak variable must belong to linked variables");
  for (const auto& knot : knots_)
    if (knot->getParent()->getParent() != variables.get() || knot->getIndex() + knot->size() > variables->getRows())
      throw QuinticSplineError("jerk peak knots must belong to linked variables");
}

Eigen::VectorXd QuinticSplineJerkPeakConstraint::getValues() const
{
  if (!variables_)
    throw QuinticSplineError("jerk peak constraint must be linked before evaluation");
  Eigen::MatrixXd positions(static_cast<Eigen::Index>(knots_.size()), map_.offset.cols());
  for (std::size_t k = 0; k < knots_.size(); ++k)
    positions.row(static_cast<Eigen::Index>(k)) = knots_[k]->value().transpose();
  const Eigen::MatrixXd jerk = map_.basis * positions + map_.offset;
  Eigen::VectorXd result(rows_);
  for (Eigen::Index sample = 0; sample < jerk.rows(); ++sample)
    for (Eigen::Index axis = 0; axis < jerk.cols(); ++axis)
    {
      const Eigen::Index row = 2 * (sample * jerk.cols() + axis);
      const double value = normalization_[axis] * jerk(sample, axis);
      result[row] = value - peak_->value()[axis];
      result[row + 1] = -value - peak_->value()[axis];
    }
  return result;
}

std::vector<Bounds> QuinticSplineJerkPeakConstraint::getBounds() const
{
  return std::vector<Bounds>(static_cast<std::size_t>(rows_), Bounds(-double(INFINITY), 0.0));
}
Eigen::VectorXd QuinticSplineJerkPeakConstraint::getCoefficients() const { return Eigen::VectorXd::Ones(rows_); }
int QuinticSplineJerkPeakConstraint::update() { return rows_; }

Jacobian QuinticSplineJerkPeakConstraint::getJacobian() const
{
  if (!variables_)
    throw QuinticSplineError("jerk peak constraint must be linked before evaluating its Jacobian");
  std::vector<Eigen::Triplet<double>> entries;
  entries.reserve(static_cast<std::size_t>(rows_) * (knots_.size() + 1));
  for (Eigen::Index sample = 0; sample < map_.basis.rows(); ++sample)
    for (Eigen::Index axis = 0; axis < map_.offset.cols(); ++axis)
    {
      const Eigen::Index row = 2 * (sample * map_.offset.cols() + axis);
      for (std::size_t knot = 0; knot < knots_.size(); ++knot)
      {
        const double value = normalization_[axis] * map_.basis(sample, static_cast<Eigen::Index>(knot));
        entries.emplace_back(row, knots_[knot]->getIndex() + axis, value);
        entries.emplace_back(row + 1, knots_[knot]->getIndex() + axis, -value);
      }
      entries.emplace_back(row, peak_->getIndex() + axis, -1.0);
      entries.emplace_back(row + 1, peak_->getIndex() + axis, -1.0);
    }
  Jacobian result(rows_, variables_->getRows());
  result.setFromTriplets(entries.begin(), entries.end());
  return result;
}
}  // namespace trajopt_ifopt
