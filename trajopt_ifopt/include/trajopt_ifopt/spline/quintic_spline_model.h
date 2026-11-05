#pragma once

#include <Eigen/Core>
#include <trajopt_common/macros.h>
#include <chrono>
#include <stdexcept>
#include <vector>

namespace trajopt_ifopt
{
using Seconds = std::chrono::duration<double>;

class TRAJOPT_API QuinticSplineError : public std::runtime_error
{
public:
  using std::runtime_error::runtime_error;
  ~QuinticSplineError() override;
};

/** Affine map from knot positions to one or more position/derivative samples. */
struct SplineAffineMap
{
  Eigen::MatrixXd basis;   // sample x knot
  Eigen::MatrixXd offset;  // sample x axis
};

struct JerkQuadrature
{
  SplineAffineMap map;
  Eigen::VectorXd weights;  // seconds
};

/** Fixed-clock degree-5 interpolant. Boundary derivatives use axis units per second and per second squared. */
class QuinticSplineModel
{
public:
  static QuinticSplineModel build(std::vector<Seconds> times,
                                  const Eigen::VectorXd& velocity_start,
                                  const Eigen::VectorXd& acceleration_start,
                                  const Eigen::VectorXd& velocity_end,
                                  const Eigen::VectorXd& acceleration_end);

  Eigen::Index knotCount() const { return position_coefficients_.cols(); }
  Eigen::Index axisCount() const { return boundary_coefficients_.cols(); }

  /** Return q^(order)(t) = basis * knot_positions + offset. */
  SplineAffineMap map(const std::vector<Seconds>& times, unsigned order) const;

  /** Three-point Gauss-Legendre rule on every knot interval integrates squared jerk exactly. */
  JerkQuadrature jerkQuadrature() const;

private:
  QuinticSplineModel(std::vector<Seconds> times,
                     Eigen::RowVectorXd knots,
                     Eigen::MatrixXd position_coefficients,
                     Eigen::MatrixXd boundary_coefficients);

  Eigen::RowVectorXd controlBasis(double normalized_time, unsigned order) const;
  std::vector<Seconds> times_;
  Eigen::RowVectorXd knots_;
  Eigen::MatrixXd position_coefficients_;
  Eigen::MatrixXd boundary_coefficients_;
};
}  // namespace trajopt_ifopt
