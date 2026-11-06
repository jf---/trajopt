#include <trajopt_ifopt/spline/quintic_spline_jerk_fourth_cost.h>
#include <trajopt_ifopt/variable_sets/node.h>
#include <trajopt_ifopt/variable_sets/nodes_variables.h>
#include <trajopt_ifopt/variable_sets/var.h>
#include <set>
#include <utility>

namespace trajopt_ifopt
{
QuinticSplineJerkFourthCost::QuinticSplineJerkFourthCost(const QuinticSplineModel& model,
                                                         std::vector<std::shared_ptr<const Var>> knots,
                                                         Eigen::VectorXd axis_weights,
                                                         std::string name)
  : ConstraintSet(std::move(name), static_cast<int>(5 * (model.knotCount() - 1) * model.axisCount()))
  , knots_(std::move(knots))
{
  if (static_cast<Eigen::Index>(knots_.size()) != model.variableCount() || axis_weights.size() != model.axisCount() ||
      !axis_weights.allFinite() || !(axis_weights.array() > 0).all() || knots_.empty() || !knots_.front())
    throw QuinticSplineError("jerk cost knot layout or axis weights invalid");
  const auto& names = knots_.front()->name();
  std::set<Eigen::Index> indices;
  for (const auto& knot : knots_)
  {
    if (!knot || !knot->getParent() || !knot->getParent()->getParent() || knot->size() != model.axisCount() ||
        knot->name() != names || knot->getIndex() < 0)
      throw QuinticSplineError("jerk cost knot axes or ownership differ");
    for (Eigen::Index axis = 0; axis < knot->size(); ++axis)
      if (!indices.insert(knot->getIndex() + axis).second)
        throw QuinticSplineError("jerk cost knot variable indices overlap");
  }
  auto quadrature = model.jerkFourthQuadrature();
  map_ = std::move(quadrature.map);
  coefficients_.resize(rows_);
  for (Eigen::Index sample = 0; sample < quadrature.weights.size(); ++sample)
    for (Eigen::Index axis = 0; axis < model.axisCount(); ++axis)
      coefficients_[sample * model.axisCount() + axis] = quadrature.weights[sample] * axis_weights[axis];
  non_zeros_ = static_cast<Eigen::Index>(rows_) * model.variableCount();
}

void QuinticSplineJerkFourthCost::initVariableDependedQuantities(const Variables::Ptr& variables)
{
  if (!variables)
    throw QuinticSplineError("jerk cost needs linked knot variables");
  for (const auto& knot : knots_)
    if (knot->getParent()->getParent() != variables.get() || knot->getIndex() + knot->size() > variables->getRows())
      throw QuinticSplineError("jerk cost knots do not belong to linked variables");
}

Eigen::VectorXd QuinticSplineJerkFourthCost::getValues() const
{
  if (!variables_)
    throw QuinticSplineError("jerk cost must be linked before evaluation");
  Eigen::MatrixXd positions(static_cast<Eigen::Index>(knots_.size()), map_.offset.cols());
  for (std::size_t k = 0; k < knots_.size(); ++k)
    positions.row(static_cast<Eigen::Index>(k)) = knots_[k]->value().transpose();
  const Eigen::MatrixXd residuals = map_.basis * positions + map_.offset;
  Eigen::VectorXd result(rows_);
  for (Eigen::Index sample = 0; sample < residuals.rows(); ++sample)
    for (Eigen::Index axis = 0; axis < residuals.cols(); ++axis)
      result[sample * residuals.cols() + axis] = residuals(sample, axis) * residuals(sample, axis);
  return result;
}

std::vector<Bounds> QuinticSplineJerkFourthCost::getBounds() const
{
  return std::vector<Bounds>(static_cast<std::size_t>(rows_), Bounds(0, 0));
}
Eigen::VectorXd QuinticSplineJerkFourthCost::getCoefficients() const { return coefficients_; }
int QuinticSplineJerkFourthCost::update() { return rows_; }

Jacobian QuinticSplineJerkFourthCost::getJacobian() const
{
  if (!variables_)
    throw QuinticSplineError("jerk cost must be linked before evaluating its Jacobian");
  Eigen::MatrixXd positions(static_cast<Eigen::Index>(knots_.size()), map_.offset.cols());
  for (std::size_t k = 0; k < knots_.size(); ++k)
    positions.row(static_cast<Eigen::Index>(k)) = knots_[k]->value().transpose();
  const Eigen::MatrixXd jerk = map_.basis * positions + map_.offset;
  std::vector<Eigen::Triplet<double>> entries;
  entries.reserve(static_cast<std::size_t>(non_zeros_));
  for (Eigen::Index sample = 0; sample < map_.basis.rows(); ++sample)
    for (Eigen::Index axis = 0; axis < map_.offset.cols(); ++axis)
      for (std::size_t knot = 0; knot < knots_.size(); ++knot)
        entries.emplace_back(sample * map_.offset.cols() + axis,
                             knots_[knot]->getIndex() + axis,
                             2.0 * jerk(sample, axis) * map_.basis(sample, static_cast<Eigen::Index>(knot)));
  Jacobian result(rows_, variables_->getRows());
  result.setFromTriplets(entries.begin(), entries.end());
  return result;
}
}  // namespace trajopt_ifopt
