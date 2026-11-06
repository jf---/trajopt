#pragma once

#include <Eigen/Core>
#include <trajopt_common/macros.h>
#include <chrono>
#include <optional>
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

  /** Missing acceleration is an optimized variable, ordered after all position knots. */
  static QuinticSplineModel buildVariableAcceleration(std::vector<Seconds> times,
                                                      const Eigen::VectorXd& velocity_start,
                                                      std::optional<Eigen::VectorXd> acceleration_start,
                                                      const Eigen::VectorXd& velocity_end,
                                                      std::optional<Eigen::VectorXd> acceleration_end);

  /** Missing derivatives are variables after the position knots, ordered v0, a0, v1, a1.
   * Axis positions use their native units; derivatives use those units per second and per second squared.
   */
  static QuinticSplineModel buildVariableBoundary(std::vector<Seconds> times,
                                                  Eigen::Index axis_count,
                                                  std::optional<Eigen::VectorXd> velocity_start,
                                                  std::optional<Eigen::VectorXd> acceleration_start,
                                                  std::optional<Eigen::VectorXd> velocity_end,
                                                  std::optional<Eigen::VectorXd> acceleration_end);

  Eigen::Index knotCount() const { return static_cast<Eigen::Index>(times_.size()); }
  Eigen::Index variableCount() const { return position_coefficients_.cols(); }
  Eigen::Index axisCount() const { return boundary_coefficients_.cols(); }

  /** Return q^(order)(t) = basis * knot_positions + offset. */
  SplineAffineMap map(const std::vector<Seconds>& times, unsigned order) const;

  /** Evaluate positions and all derivatives through max_order from one batched clock. */
  std::vector<Eigen::MatrixXd> evaluate(const std::vector<Seconds>& times,
                                        const Eigen::MatrixXd& positions,
                                        unsigned max_order) const;

  /** Three-point Gauss-Legendre rule on every knot interval integrates squared jerk exactly. */
  JerkQuadrature jerkQuadrature() const;

  /** Five-point Gauss-Legendre per knot interval integrates jerk to the fourth power. */
  JerkQuadrature jerkFourthQuadrature() const;

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
