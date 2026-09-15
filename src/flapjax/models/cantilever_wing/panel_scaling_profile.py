from __future__ import annotations

import copy
import csv
import time
from collections.abc import Callable
from pathlib import Path

"""
Obtain data on evaluation time of the primal, matrix-free and dense adjoint problems, and how these change with problem
size. This creates plots for the time per step of these problems for just the time series evaluation, omitting pre- and
post-processing steps which have fixed cost irrespective of number of time steps. By default this script is chosen to
run on one core only. This may take ~1 hour to run, as the compile cost is rather large for so many different cases, as
well as the large cases taking in the order of seconds to complete a single step.
"""

# jax imports need to be after setting environment variables
import jax
from jax import Array
from jax import numpy as jnp

from flapjax.coupled import (
    AeroelasticCase,
    CoupledAeroelastic,
)
from flapjax.coupled.data_structures import (
    AeroelasticDesignVariables,
    AeroelasticFullStates,
)
from flapjax.models.cantilever_wing.cantilever_wing import generate_cantilever_wing
from flapjax.structure.utils import get_solve_dofs
from flapjax.utils.data_structures import ConvergenceSettings

N_TSTEPS: int = 10  # average time over this many time steps
METRICS: tuple[str, ...] = ("primal", "adj_mf_precond", "adj_dense")
CSV_FIELDS: tuple[str, ...] = (
    "m",
    "n",
    "m_star",
    "bound_panels",
    "wake_panels",
    "metric",
    "time_s",
    "precond_build_s",
)


FIXED_ITER = ConvergenceSettings(
    max_n_iter=3,
    rel_disp_tol=None,
    abs_disp_tol=None,
    rel_force_tol=None,
    abs_force_tol=None,
)

STATIC_CONVERGENCE = ConvergenceSettings(
    max_n_iter=100,
    rel_disp_tol=1e-5,
    abs_disp_tol=1e-8,
    rel_force_tol=1e-5,
    abs_force_tol=1e-8,
)


def build_wing(m: int, n: int, m_star: int) -> CoupledAeroelastic:
    return generate_cantilever_wing(n_nodes=n + 1, m=m, m_star=m_star)


def objective(
    states: AeroelasticFullStates,
    _dv: AeroelasticDesignVariables,
    _,
) -> Array:
    # sample objective, chosen as it has nonzero value at all timesteps
    return states.structure.f_elem[0, 5]


def make_adjoint_step(
    wing: CoupledAeroelastic,
    case: AeroelasticCase,
    matrix_free: bool,
    n_tstep_adjoint: int,
    gmres_precond: bool = True,
    preconditioner: Callable[[Array], Array] | None = None,
) -> Callable:
    # return a function which takes no arguments and computes the adjont

    @jax.jit
    def step() -> object:
        return wing.dynamic_adjoint(
            case=case,
            objective=objective,
            matrix_free=matrix_free,
            gmres_mode="incremental",
            gmres_precond=gmres_precond,
            preconditioner=preconditioner,
            grads_to_compute=None,
            approx_grads=True,
            i_ts_adjoint_range=(None, n_tstep_adjoint),
            include_initial_state_grad=False,
        )

    return step


def time_fn(fn: Callable, n_warmup: int = 1, n_loops: int = 5) -> float:
    # function to time average evaluation
    for _ in range(n_warmup):
        jax.block_until_ready(fn())
    t0 = time.perf_counter()
    for _ in range(n_loops):
        jax.block_until_ready(fn())
    return (time.perf_counter() - t0) / n_loops


def append_row(row: dict[str, object], path: Path) -> None:
    # write outputs to file. This is done as the script runs as it prevents data loss if the script crashes, which can
    # occur if choosing too large a model and running out of RAM
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_FIELDS))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        fh.flush()


def run_config(m: int, n: int, m_star: int, metric: str, csv_path: Path) -> None:
    wing = build_wing(m=m, n=n, m_star=m_star)

    # One initial-condition slot plus N_STEPS evolved timesteps.
    n_tstep_case = N_TSTEPS + 1

    static_sol = wing.static_solve(
        f_ext_dead=None,
        f_ext_follower=None,
        prescribed_dofs=jnp.arange(6),
        horseshoe=True,
    )

    wing.fsi_convergence_settings = FIXED_ITER
    wing.structure.struct_convergence_settings = FIXED_ITER

    base: dict[str, int | float] = {
        "m": m,
        "n": n,
        "m_star": m_star,
        "bound_panels": m * n,
        "wake_panels": m_star * n,
        "precond_build_s": 0.0,
    }

    if metric == "primal":
        prescribed_dofs = tuple(range(6))
        primal_wing = copy.deepcopy(wing)

        @jax.jit
        def primal_step() -> AeroelasticCase:
            return primal_wing.dynamic_solve(
                init_case=static_sol,
                prescribed_dofs=prescribed_dofs,
                n_tstep=n_tstep_case,
            )

        t_total = time_fn(primal_step, n_warmup=1, n_loops=2)
        append_row(
            {**base, "metric": "primal", "time_s": t_total / N_TSTEPS}, path=csv_path
        )
        return

    # both adjoint methods require a dynamic solution input
    case = wing.dynamic_solve(
        init_case=static_sol,
        prescribed_dofs=jnp.arange(6),
        n_tstep=n_tstep_case,
    )

    if metric == "adj_mf_precond":
        # build the frozen-wake preconditioner seperately
        solve_dofs = get_solve_dofs(
            n_dof=wing.structure.n_dof,
            prescribed_dofs=case.structure.prescribed_dofs,
        )
        dv_full = wing.get_design_variables(case=case, grads_to_compute=None)

        def _build_precond() -> Callable[[Array], Array]:
            return wing.make_frozen_wake_preconditioner(
                case=case,
                dv_full=dv_full,
                solve_dofs=solve_dofs,
                approx_grads=True,
            )

        # warmup call
        _build_precond()

        # time compiled evaluation
        t0 = time.perf_counter()
        preconditioner = _build_precond()
        base["precond_build_s"] = time.perf_counter() - t0

        # warmup call for adjoint
        step = make_adjoint_step(
            wing=wing,
            case=case,
            matrix_free=True,
            n_tstep_adjoint=N_TSTEPS,
            gmres_precond=True,
            preconditioner=preconditioner,
        )
    elif metric == "adj_dense":
        step = make_adjoint_step(
            wing=wing,
            case=case,
            matrix_free=False,
            n_tstep_adjoint=N_TSTEPS,  # warmup call
        )
    else:
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")

    t_total = time_fn(step, n_warmup=1, n_loops=2)
    append_row({**base, "metric": metric, "time_s": t_total / N_TSTEPS}, csv_path)


def run_sweep(
    label: str,
    sweep_param: str,
    sweep_values: list[int],
    fixed: dict[str, int],
    csv_path: Path,
    skip_dense_above: int,
) -> None:
    # run for all values in parameter sweep

    for metric in METRICS:
        print(f"\n{label} sweep - {metric} evaluation ###", flush=True)
        for val in sweep_values:
            if metric == "adj_dense" and val >= skip_dense_above:
                continue  # skip large dense adjoints as RAM will get exhausted
            params = {**fixed, sweep_param: val}
            print(
                f"m={params['m']}, n={params['n']}, m_star={params['m_star']}",
                flush=True,
            )
            run_config(metric=metric, csv_path=csv_path, **params)


def plot_sweeps(bound_csv_: Path, wake_csv_: Path, out_path: Path) -> None:
    # log-log plot of step time vs. problem size
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _load(csv_source: Path, x_col: str) -> dict[str, list[tuple[int, float]]]:
        # load data from file. This approach was chosen as it allows this function to process previously run cases
        by_metric: dict[str, list[tuple[int, float]]] = {}
        with csv_source.open() as fh:
            for row in csv.DictReader(fh):
                by_metric.setdefault(row["metric"], []).append(
                    (int(row[x_col]), float(row["time_s"]))
                )
        for series in by_metric.values():
            series.sort()
        return by_metric

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), sharey=True)
    for ax, path, x_key, title in (
        (axes[0], bound_csv_, "bound_panels", "Bound-panel sweep"),
        (axes[1], wake_csv_, "wake_panels", "Wake-panel sweep"),
    ):
        for metric, pts in _load(path, x_key).items():
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", label=metric)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(x_key)
        ax.set_title(title)
        ax.grid(True, which="both", ls=":", lw=0.5)
        ax.legend()
    axes[0].set_ylabel("time per step (s)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    out_dir = Path(__file__).parent / "panel_scaling_output"
    bound_csv = out_dir / "bound_sweep.csv"
    wake_csv = out_dir / "wake_sweep.csv"

    # fresh CSVs each run
    # Comment these two lines out to resume a previous partial sweep.
    bound_csv.unlink(missing_ok=True)
    wake_csv.unlink(missing_ok=True)

    # baseline parameters
    n_fixed = 20
    m_fixed = 12
    m_star_fixed = 50

    # Bound sweep: vary m, keep n and m_star fixed
    run_sweep(
        label="bound",
        sweep_param="m",
        sweep_values=[8, 12, 16, 24, 32, 48, 64],
        fixed={"n": n_fixed, "m_star": m_star_fixed},
        csv_path=bound_csv,
        skip_dense_above=16,
    )

    # wake sweep: vary m_star, keep m and n fixed
    run_sweep(
        label="wake",
        sweep_param="m_star",
        sweep_values=[20, 30, 40, 60, 80, 120, 160, 240, 320, 480, 640],
        fixed={"m": m_fixed, "n": n_fixed},
        csv_path=wake_csv,
        skip_dense_above=120,
    )

    plot_sweeps(bound_csv, wake_csv, out_dir / "panel_scaling.png")
