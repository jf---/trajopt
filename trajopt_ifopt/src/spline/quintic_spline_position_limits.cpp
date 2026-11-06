#include <trajopt_ifopt/spline/quintic_spline_position_limits.h>
#include <trajopt_ifopt/variable_sets/node.h>
#include <trajopt_ifopt/variable_sets/nodes_variables.h>
#include <trajopt_ifopt/variable_sets/var.h>
#include <cmath>
#include <limits>
#include <set>
#include <utility>

namespace trajopt_ifopt
{
QuinticSplinePositionLimits::QuinticSplinePositionLimits(const QuinticSplineModel& model,
                                                         std::vector<std::shared_ptr<const Var>> knots,
                                                         std::vector<Seconds> samples,
                                                         std::vector<Bounds> limits,
                                                         std::string name)
  : ConstraintSet(std::move(name), static_cast<int>(2 * static_cast<Eigen::Index>(samples.size()) * model.axisCount()))
  , knots_(std::move(knots))
{
  if (static_cast<Eigen::Index>(knots_.size()) != model.variableCount() ||
      static_cast<Eigen::Index>(limits.size()) != model.axisCount() || samples.empty() || knots_.empty() ||
      !knots_.front())
    throw QuinticSplineError("position constraint knot, sample or limit layout invalid");
  for (std::size_t i = 0; i < samples.size(); ++i)
    if (!std::isfinite(samples[i].count()) || (i > 0 && samples[i] <= samples[i - 1]))
      throw QuinticSplineError("position constraint sample times must increase");
  for (const auto& bound : limits)
    if (!std::isfinite(bound.getLower()) || !std::isfinite(bound.getUpper()) || bound.getLower() > bound.getUpper())
      throw QuinticSplineError("position constraint bounds must be finite and ordered");
  const auto& names = knots_.front()->name();
  std::set<Eigen::Index> indices;
  for (const auto& knot : knots_)
  {
    if (!knot || !knot->getParent() || !knot->getParent()->getParent() || knot->size() != model.axisCount() ||
        knot->name() != names || knot->getIndex() < 0)
      throw QuinticSplineError("position constraint knot axes or ownership differ");
    for (Eigen::Index axis = 0; axis < knot->size(); ++axis)
      if (!indices.insert(knot->getIndex() + axis).second)
        throw QuinticSplineError("position constraint knot variable indices overlap");
  }
  map_ = model.map(samples, 0);
  const double infinity = std::numeric_limits<double>::infinity();
  bounds_.reserve(static_cast<std::size_t>(rows_));
  for (std::size_t i = 0; i < samples.size(); ++i)
    for (const auto& bound : limits)
      bounds_.emplace_back(bound.getLower(), infinity);
  for (std::size_t i = 0; i < samples.size(); ++i)
    for (const auto& bound : limits)
      bounds_.emplace_back(-infinity, bound.getUpper());
  non_zeros_ = static_cast<Eigen::Index>(rows_) * model.variableCount();
}

void QuinticSplinePositionLimits::initVariableDependedQuantities(const Variables::Ptr& variables)
{
  if (!variables)
    throw QuinticSplineError("position constraint needs linked variables");
  for (const auto& knot : knots_)
    if (knot->getParent()->getParent() != variables.get() || knot->getIndex() + knot->size() > variables->getRows())
      throw QuinticSplineError("position constraint knots do not belong to linked variables");
}

Eigen::VectorXd QuinticSplinePositionLimits::getValues() const
{
  if (!variables_)
    throw QuinticSplineError("position constraint must be linked");
  Eigen::MatrixXd positions(static_cast<Eigen::Index>(knots_.size()), map_.offset.cols());
  for (std::size_t k = 0; k < knots_.size(); ++k)
    positions.row(static_cast<Eigen::Index>(k)) = knots_[k]->value().transpose();
  const Eigen::MatrixXd values = map_.basis * positions + map_.offset;
  Eigen::VectorXd result(rows_);
  for (Eigen::Index sample = 0; sample < values.rows(); ++sample)
    for (Eigen::Index axis = 0; axis < values.cols(); ++axis)
    {
      const auto row = sample * values.cols() + axis;
      result[row] = values(sample, axis);
      result[row + rows_ / 2] = values(sample, axis);
    }
  return result;
}
std::vector<Bounds> QuinticSplinePositionLimits::getBounds() const { return bounds_; }
Eigen::VectorXd QuinticSplinePositionLimits::getCoefficients() const { return Eigen::VectorXd::Ones(rows_); }
int QuinticSplinePositionLimits::update() { return rows_; }

Jacobian QuinticSplinePositionLimits::getJacobian() const
{
  if (!variables_)
    throw QuinticSplineError("position constraint must be linked before evaluating Jacobian");
  std::vector<Eigen::Triplet<double>> entries;
  entries.reserve(static_cast<std::size_t>(non_zeros_));
  for (Eigen::Index sample = 0; sample < map_.basis.rows(); ++sample)
    for (Eigen::Index axis = 0; axis < map_.offset.cols(); ++axis)
      for (std::size_t knot = 0; knot < knots_.size(); ++knot)
      {
        const auto row = sample * map_.offset.cols() + axis;
        const auto column = knots_[knot]->getIndex() + axis;
        const auto value = map_.basis(sample, static_cast<Eigen::Index>(knot));
        entries.emplace_back(row, column, value);
        entries.emplace_back(row + rows_ / 2, column, value);
      }
  Jacobian result(rows_, variables_->getRows());
  result.setFromTriplets(entries.begin(), entries.end());
  return result;
}
}  // namespace trajopt_ifopt
