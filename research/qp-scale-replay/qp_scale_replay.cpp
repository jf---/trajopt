#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <trajopt_sqp/qp_problem.h>
#include <trajopt_sqp/piqp_solver.h>
#include <stdexcept>
#include <cmath>
#include <limits>

namespace nb = nanobind;
class PiecewiseQpDiagnosticError : public std::runtime_error
{
public:
  using std::runtime_error::runtime_error;
};

NB_MODULE(_qp_scale_replay, m)
{
  nb::module_::import_("tesseract_robotics.trajopt_sqp");
  nb::exception<PiecewiseQpDiagnosticError>(m, "PiecewiseQpDiagnosticError");
  m.def("compare", [](const trajopt_sqp::QPProblem& problem, const Eigen::VectorXd& best, const Eigen::VectorXd& box) {
    const auto& hessian = problem.getHessian();
    const auto& gradient = problem.getGradient();
    const auto& matrix = problem.getConstraintMatrix();
    const auto& lower = problem.getBoundsLower();
    const auto& upper = problem.getBoundsUpper();
    if (matrix.cols() != hessian.cols() || gradient.size() != matrix.cols() || lower.size() != matrix.rows() ||
        upper.size() != matrix.rows() || best.size() != box.size() || best.size() > matrix.cols())
      throw PiecewiseQpDiagnosticError("QP replay dimensions differ");
    nb::list comparisons;
    Eigen::VectorXd reference_solution;
    for (const bool scale_cost : { false, true })
    {
      trajopt_sqp::PIQPSolver replay;
      replay.settings.preconditioner_scale_cost = scale_cost;
      if (!replay.init(matrix.cols(), matrix.rows()) || !replay.updateHessianMatrix(hessian) ||
          !replay.updateGradient(gradient) || !replay.updateLinearConstraintsMatrix(matrix) ||
          !replay.updateBounds(lower, upper))
        throw PiecewiseQpDiagnosticError("QP replay initialization failed");
      const bool solved = replay.solve();
      nb::dict item;
      item["scale_cost"] = scale_cost;
      if (!solved)
      {
        // The wrapper can reject crossed bounds before PIQP initializes its result.
        item["status"] = "wrapper_failed";
        comparisons.append(item);
        continue;
      }
      const auto& info = replay.solver().result().info;
      item["status"] = piqp::status_to_string(info.status);
      item["iterations"] = info.iter;
      item["piqp_primal_residual"] = info.primal_res;
      item["piqp_dual_residual"] = info.dual_res;
      item["piqp_objective"] = info.primal_obj;
      if (solved)
      {
        const Eigen::VectorXd x = replay.getSolution();
        if (x.size() != matrix.cols() || !x.allFinite())
          throw PiecewiseQpDiagnosticError("QP replay returned invalid solution");
        const Eigen::VectorXd projected = matrix * x;
        const double violation =
            projected.size() == 0 ? 0.0 : ((lower - projected).cwiseMax(projected - upper)).cwiseMax(0.0).maxCoeff();
        const double trust =
            best.size() == 0 ? 0.0 : ((x.head(best.size()) - best).cwiseAbs() - box).cwiseMax(0.0).maxCoeff();
        double objective = 0.0;
        double compensation = 0.0;
        double absolute_terms = 0.0;
        const auto add_term = [&](double term) {
          absolute_terms += std::abs(term);
          const double adjusted = term - compensation;
          const double next = objective + adjusted;
          compensation = (next - objective) - adjusted;
          objective = next;
        };
        for (Eigen::Index i = 0; i < x.size(); ++i)
          add_term(gradient[i] * x[i]);
        for (Eigen::Index outer = 0; outer < hessian.outerSize(); ++outer)
          for (trajopt_ifopt::Jacobian::InnerIterator value(hessian, outer); value; ++value)
            if (value.col() >= value.row())
              add_term(value.value() * x[value.row()] * x[value.col()] * (value.row() == value.col() ? 1.0 : 2.0));
        item["objective"] = objective;
        item["objective_abs_term_sum"] = absolute_terms;
        item["objective_term_roundoff_scale"] = std::numeric_limits<double>::epsilon() * absolute_terms;
        item["max_constraint_violation"] = violation;
        item["max_trust_box_violation"] = trust;
        if (!scale_cost)
          reference_solution = x;
        else if (reference_solution.size() == x.size())
        {
          const Eigen::VectorXd delta = x - reference_solution;
          const Eigen::VectorXd sum = x + reference_solution;
          double difference = 0.0;
          double difference_compensation = 0.0;
          double difference_absolute_terms = 0.0;
          const auto add_difference = [&](double term) {
            difference_absolute_terms += std::abs(term);
            const double adjusted = term - difference_compensation;
            const double next = difference + adjusted;
            difference_compensation = (next - difference) - adjusted;
            difference = next;
          };
          for (Eigen::Index i = 0; i < x.size(); ++i)
            add_difference(gradient[i] * delta[i]);
          for (Eigen::Index outer = 0; outer < hessian.outerSize(); ++outer)
            for (trajopt_ifopt::Jacobian::InnerIterator value(hessian, outer); value; ++value)
              if (value.col() == value.row())
                add_difference(value.value() * delta[value.row()] * sum[value.row()]);
              else if (value.col() > value.row())
                add_difference(value.value() *
                               (delta[value.row()] * sum[value.col()] + delta[value.col()] * sum[value.row()]));
          item["objective_delta_from_default"] = difference;
          item["objective_delta_abs_term_sum"] = difference_absolute_terms;
          item["objective_delta_term_roundoff_scale"] =
              std::numeric_limits<double>::epsilon() * difference_absolute_terms;
        }
      }
      comparisons.append(item);
    }
    return comparisons;
  });
}
