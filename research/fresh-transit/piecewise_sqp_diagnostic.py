"""Observe successful SQP trials without changing their variables or constraints."""

import _quintic_spline as native
from r62_master_sqp_observer import capture_master_trace
from tesseract_robotics.trajopt_sqp import SQPCallback


def collision_row_counts(constraints):
    """Read the current dynamic row counts without updating the constraints."""
    return {str(item.getName()): int(item.getRows()) for item in constraints}


class PiecewiseSqpTrace(SQPCallback):
    """Copy each successful QP trial; native failed QPs do not invoke callbacks."""

    def __init__(self, collision_constraints):
        super().__init__()
        self.constraints = collision_constraints
        self.records = []

    def execute(self, problem, result):
        trace = capture_master_trace(problem, result)
        trace.update(
            collision_rows_last_evaluation=collision_row_counts(self.constraints),
            qp=native.qp_diagnostic(problem, result.new_var_vals),
        )
        self.records.append(trace)
        return True
