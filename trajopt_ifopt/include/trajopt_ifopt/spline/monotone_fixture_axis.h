#pragma once

#include <trajopt_ifopt/spline/quintic_spline_model.h>

#include <array>
#include <stdexcept>
#include <vector>

namespace trajopt_ifopt
{
class TRAJOPT_API MonotoneFixtureAxisError : public std::runtime_error
{
public:
  using std::runtime_error::runtime_error;
  ~MonotoneFixtureAxisError() override;
};

/** Values in the input axis units and those units per second, second squared and second cubed. */
struct FixtureAxisSample
{
  double position;
  double velocity;
  double acceleration;
  double jerk;
};

/** C2 quintics through ordered axis targets on an explicit modeled clock.
 *
 * This curve imposes zero velocity at both outer ends, reversing targets and
 * holds. Same-direction internal targets carry the signed minimum neighboring
 * secant velocity. Every target has zero acceleration. These are modeled
 * boundary choices, not observed controller velocities or physical limits.
 */
class MonotoneFixtureAxis
{
public:
  static MonotoneFixtureAxis build(std::vector<Seconds> times, std::vector<double> targets);

  /** Evaluate the position and first three time derivatives within the supplied clock. */
  FixtureAxisSample evaluate(Seconds time) const;

private:
  MonotoneFixtureAxis(std::vector<Seconds> times, std::vector<std::array<double, 6>> controls);

  std::vector<Seconds> times_;
  std::vector<std::array<double, 6>> controls_;
};
}  // namespace trajopt_ifopt
