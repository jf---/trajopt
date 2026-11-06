#include <trajopt_ifopt/spline/monotone_fixture_axis.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

namespace trajopt_ifopt
{
MonotoneFixtureAxisError::~MonotoneFixtureAxisError() = default;

MonotoneFixtureAxis::MonotoneFixtureAxis(std::vector<Seconds> times, std::vector<std::array<double, 6>> controls)
  : times_(std::move(times)), controls_(std::move(controls))
{
}

MonotoneFixtureAxis MonotoneFixtureAxis::build(std::vector<Seconds> times, std::vector<double> targets)
{
  if (times.size() < 2 || times.size() != targets.size())
    throw MonotoneFixtureAxisError("two or more clocked axis targets are required");
  for (std::size_t i = 0; i < times.size(); ++i)
    if (!std::isfinite(times[i].count()) || !std::isfinite(targets[i]) || (i && times[i] <= times[i - 1]))
      throw MonotoneFixtureAxisError("clock and axis targets must be finite and strictly ordered");

  std::vector<double> durations(times.size() - 1);
  std::vector<double> secants(times.size() - 1);
  for (std::size_t i = 0; i < durations.size(); ++i)
  {
    durations[i] = (times[i + 1] - times[i]).count();
    secants[i] = (targets[i + 1] - targets[i]) / durations[i];
    if (!std::isfinite(durations[i]) || durations[i] <= 0 || !std::isfinite(secants[i]))
      throw MonotoneFixtureAxisError("clock interval or axis secant is not finite");
  }

  std::vector<double> velocities(times.size(), 0.0);
  for (std::size_t i = 1; i + 1 < times.size(); ++i)
  {
    const double a = secants[i - 1];
    const double b = secants[i];
    if ((a > 0 && b > 0) || (a < 0 && b < 0))
      velocities[i] = std::copysign(std::min(std::abs(a), std::abs(b)), a);
  }

  std::vector<std::array<double, 6>> controls;
  controls.reserve(durations.size());
  for (std::size_t i = 0; i < durations.size(); ++i)
  {
    const double left = durations[i] * velocities[i] / 5;
    const double right = durations[i] * velocities[i + 1] / 5;
    // The quintic Bernstein controls impose endpoint q, qdot and zero qddot.
    // Each endpoint velocity has the interval's direction and magnitude no
    // larger than its secant. Thus the normalized first and last offsets are
    // at most 1/5 and 2/5 of the interval displacement. The six controls are
    // ordered, so their first differences (the derivative's Bernstein
    // controls) have the interval's direction everywhere: no reversal.
    const std::array<double, 6> points = {
      targets[i],    targets[i] + left, targets[i] + 2 * left, targets[i + 1] - 2 * right, targets[i + 1] - right,
      targets[i + 1]
    };
    for (double point : points)
      if (!std::isfinite(point))
        throw MonotoneFixtureAxisError("axis polynomial controls are not finite");
    controls.push_back(points);
  }
  return MonotoneFixtureAxis(std::move(times), std::move(controls));
}

FixtureAxisSample MonotoneFixtureAxis::evaluate(Seconds time) const
{
  if (!std::isfinite(time.count()) || time < times_.front() || time > times_.back())
    throw MonotoneFixtureAxisError("evaluation time lies outside the modeled clock");
  const auto next = std::upper_bound(times_.begin(), times_.end(), time);
  const auto index = next == times_.end() ? controls_.size() - 1 : static_cast<std::size_t>(next - times_.begin() - 1);
  const double duration = (times_[index + 1] - times_[index]).count();
  const double u = (time - times_[index]).count() / duration;
  auto controls = controls_[index];
  std::array<double, 4> values{};
  for (std::size_t order = 0; order < values.size(); ++order)
  {
    const auto degree = controls.size() - 1 - order;
    auto work = controls;
    for (std::size_t width = degree; width > 0; --width)
      for (std::size_t j = 0; j < width; ++j)
        work[j] = (1 - u) * work[j] + u * work[j + 1];
    values[order] = work[0];
    if (!std::isfinite(values[order]))
      throw MonotoneFixtureAxisError("axis derivative is not finite");
    if (order + 1 < values.size())
      for (std::size_t j = 0; j < degree; ++j)
        controls[j] = static_cast<double>(degree) * (controls[j + 1] - controls[j]) / duration;
  }
  return { values[0], values[1], values[2], values[3] };
}
}  // namespace trajopt_ifopt
