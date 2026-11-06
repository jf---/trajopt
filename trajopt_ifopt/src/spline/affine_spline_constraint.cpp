#include <trajopt_ifopt/spline/affine_spline_constraint.h>
#include <trajopt_ifopt/variable_sets/node.h>
#include <trajopt_ifopt/variable_sets/nodes_variables.h>
#include <trajopt_ifopt/variable_sets/var.h>
#include <set>
#include <utility>

namespace trajopt_ifopt
{
AffineSplineConstraintError::~AffineSplineConstraintError() = default;
AffineSplineConstraint::AffineSplineConstraint(std::vector<std::shared_ptr<const Var>> variables,
                                               Eigen::MatrixXd basis,
                                               Eigen::MatrixXd offset,
                                               std::vector<Bounds> bounds,
                                               Eigen::VectorXd coefficients,
                                               std::string name)
  : ConstraintSet(std::move(name), static_cast<int>(offset.size()))
  , independent_(std::move(variables))
  , basis_(std::move(basis))
  , offset_(std::move(offset))
  , bounds_(std::move(bounds))
  , coefficients_(std::move(coefficients))
{
  if (basis_.rows() <= 0 || offset_.cols() <= 0 || basis_.rows() != offset_.rows() ||
      basis_.cols() != static_cast<Eigen::Index>(independent_.size()) || !basis_.allFinite() || !offset_.allFinite() ||
      static_cast<Eigen::Index>(bounds_.size()) != rows_ || coefficients_.size() != rows_ ||
      !coefficients_.allFinite() || (coefficients_.array() < 0).any())
    throw AffineSplineConstraintError("invalid affine spline map, bounds or coefficients");
  std::set<Eigen::Index> indices;
  for (const auto& variable : independent_)
  {
    if (!variable || !variable->getParent() || !variable->getParent()->getParent() ||
        variable->size() != offset_.cols() || variable->getIndex() < 0)
      throw AffineSplineConstraintError("invalid affine spline variable or ownership");
    for (Eigen::Index axis = 0; axis < variable->size(); ++axis)
      if (!indices.insert(variable->getIndex() + axis).second)
        throw AffineSplineConstraintError("affine spline variable indices overlap");
  }
}

void AffineSplineConstraint::initVariableDependedQuantities(const Variables::Ptr& variables)
{
  if (!variables)
    throw AffineSplineConstraintError("affine spline needs linked variables");
  for (const auto& variable : independent_)
    if (variable->getParent()->getParent() != variables.get() ||
        variable->getIndex() + variable->size() > variables->getRows())
      throw AffineSplineConstraintError("affine spline variable ownership differs");
}

Eigen::VectorXd AffineSplineConstraint::getValues() const
{
  if (!variables_)
    throw AffineSplineConstraintError("affine spline must be linked before evaluation");
  Eigen::MatrixXd input(static_cast<Eigen::Index>(independent_.size()), offset_.cols());
  for (std::size_t i = 0; i < independent_.size(); ++i)
    input.row(static_cast<Eigen::Index>(i)) = independent_[i]->value().transpose();
  const Eigen::MatrixXd mapped = basis_ * input + offset_;
  Eigen::VectorXd result(rows_);
  for (Eigen::Index i = 0; i < mapped.rows(); ++i)
    for (Eigen::Index axis = 0; axis < mapped.cols(); ++axis)
      result[i * mapped.cols() + axis] = mapped(i, axis);
  return result;
}
std::vector<Bounds> AffineSplineConstraint::getBounds() const { return bounds_; }
Eigen::VectorXd AffineSplineConstraint::getCoefficients() const { return coefficients_; }
int AffineSplineConstraint::update() { return rows_; }
Jacobian AffineSplineConstraint::getJacobian() const
{
  if (!variables_)
    throw AffineSplineConstraintError("affine spline must be linked before evaluating its Jacobian");
  std::vector<Eigen::Triplet<double>> entries;
  entries.reserve(static_cast<std::size_t>(rows_) * independent_.size());
  for (Eigen::Index i = 0; i < basis_.rows(); ++i)
    for (Eigen::Index axis = 0; axis < offset_.cols(); ++axis)
      for (std::size_t j = 0; j < independent_.size(); ++j)
        entries.emplace_back(
            i * offset_.cols() + axis, independent_[j]->getIndex() + axis, basis_(i, static_cast<Eigen::Index>(j)));
  Jacobian result(rows_, variables_->getRows());
  result.setFromTriplets(entries.begin(), entries.end());
  return result;
}
}  // namespace trajopt_ifopt
