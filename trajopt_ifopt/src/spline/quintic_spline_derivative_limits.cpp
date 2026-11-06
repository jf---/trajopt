#include <trajopt_ifopt/spline/quintic_spline_derivative_limits.h>
#include <trajopt_ifopt/variable_sets/node.h>
#include <trajopt_ifopt/variable_sets/nodes_variables.h>
#include <trajopt_ifopt/variable_sets/var.h>
#include <cmath>
#include <limits>
#include <set>
#include <utility>

namespace trajopt_ifopt
{
namespace
{
int finiteBoundCount(const std::vector<Bounds>& limits)
{
  int count = 0;
  for (const auto& bound : limits)
    count += static_cast<int>(std::isfinite(bound.getLower())) + static_cast<int>(std::isfinite(bound.getUpper()));
  return count;
}
}  // namespace
QuinticSplineDerivativeLimits::QuinticSplineDerivativeLimits(const QuinticSplineModel& model,
                                                             std::vector<std::shared_ptr<const Var>> knots,
                                                             std::vector<Seconds> samples,
                                                             std::vector<Bounds> limits,
                                                             std::string name)
  : ConstraintSet(std::move(name), finiteBoundCount(limits)), knots_(std::move(knots))
{
  if (static_cast<Eigen::Index>(knots_.size()) != model.variableCount() ||
      static_cast<Eigen::Index>(limits.size()) != static_cast<Eigen::Index>(samples.size()) * model.axisCount() ||
      samples.empty() || knots_.empty() || !knots_.front() || rows_ == 0)
    throw QuinticSplineError("velocity constraint knot, sample or bound layout invalid");
  for (std::size_t i = 0; i < samples.size(); ++i)
    if (!std::isfinite(samples[i].count()) || (i > 0 && samples[i] <= samples[i - 1]))
      throw QuinticSplineError("velocity constraint sample times must increase");
  for (const auto& bound : limits)
    if (std::isnan(bound.getLower()) || std::isnan(bound.getUpper()) || bound.getLower() > bound.getUpper() ||
        bound.getLower() == std::numeric_limits<double>::infinity() ||
        bound.getUpper() == -std::numeric_limits<double>::infinity())
      throw QuinticSplineError("velocity constraint bounds must be ordered");
  const auto& names = knots_.front()->name();
  std::set<Eigen::Index> indices;
  for (const auto& knot : knots_)
  {
    if (!knot || !knot->getParent() || !knot->getParent()->getParent() || knot->size() != model.axisCount() ||
        knot->name() != names || knot->getIndex() < 0)
      throw QuinticSplineError("velocity constraint knot axes or ownership differ");
    for (Eigen::Index axis = 0; axis < knot->size(); ++axis)
      if (!indices.insert(knot->getIndex() + axis).second)
        throw QuinticSplineError("velocity constraint knot variable indices overlap");
  }
  map_ = model.map(samples, 1);
  const double infinity = std::numeric_limits<double>::infinity();
  bounds_.reserve(static_cast<std::size_t>(rows_));
  for (std::size_t index = 0; index < limits.size(); ++index)
    if (std::isfinite(limits[index].getLower()))
    {
      bounds_.emplace_back(limits[index].getLower(), infinity);
      row_samples_.push_back(static_cast<Eigen::Index>(index) / model.axisCount());
      row_axes_.push_back(static_cast<Eigen::Index>(index) % model.axisCount());
    }
  for (std::size_t index = 0; index < limits.size(); ++index)
    if (std::isfinite(limits[index].getUpper()))
    {
      bounds_.emplace_back(-infinity, limits[index].getUpper());
      row_samples_.push_back(static_cast<Eigen::Index>(index) / model.axisCount());
      row_axes_.push_back(static_cast<Eigen::Index>(index) % model.axisCount());
    }
  non_zeros_ = static_cast<Eigen::Index>(rows_) * model.variableCount();
}

void QuinticSplineDerivativeLimits::initVariableDependedQuantities(const Variables::Ptr& variables)
{
  if (!variables)
    throw QuinticSplineError("velocity constraint needs linked variables");
  for (const auto& knot : knots_)
    if (knot->getParent()->getParent() != variables.get() || knot->getIndex() + knot->size() > variables->getRows())
      throw QuinticSplineError("velocity constraint knots do not belong to linked variables");
}

Eigen::VectorXd QuinticSplineDerivativeLimits::getValues() const
{
  if (!variables_)
    throw QuinticSplineError("velocity constraint must be linked");
  Eigen::MatrixXd positions(static_cast<Eigen::Index>(knots_.size()), map_.offset.cols());
  for (std::size_t k = 0; k < knots_.size(); ++k)
    positions.row(static_cast<Eigen::Index>(k)) = knots_[k]->value().transpose();
  const Eigen::MatrixXd values = map_.basis * positions + map_.offset;
  Eigen::VectorXd result(rows_);
  for (Eigen::Index row = 0; row < rows_; ++row)
    result[row] = values(row_samples_[static_cast<std::size_t>(row)], row_axes_[static_cast<std::size_t>(row)]);
  return result;
}
std::vector<Bounds> QuinticSplineDerivativeLimits::getBounds() const { return bounds_; }
Eigen::VectorXd QuinticSplineDerivativeLimits::getCoefficients() const { return Eigen::VectorXd::Ones(rows_); }
int QuinticSplineDerivativeLimits::update() { return rows_; }

Jacobian QuinticSplineDerivativeLimits::getJacobian() const
{
  if (!variables_)
    throw QuinticSplineError("velocity constraint must be linked before evaluating Jacobian");
  std::vector<Eigen::Triplet<double>> entries;
  entries.reserve(static_cast<std::size_t>(non_zeros_));
  for (Eigen::Index row = 0; row < rows_; ++row)
    for (std::size_t knot = 0; knot < knots_.size(); ++knot)
    {
      const auto sample = row_samples_[static_cast<std::size_t>(row)];
      const auto axis = row_axes_[static_cast<std::size_t>(row)];
      const auto column = knots_[knot]->getIndex() + axis;
      const auto value = map_.basis(sample, static_cast<Eigen::Index>(knot));
      entries.emplace_back(row, column, value);
    }
  Jacobian result(rows_, variables_->getRows());
  result.setFromTriplets(entries.begin(), entries.end());
  return result;
}
}  // namespace trajopt_ifopt
