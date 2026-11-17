from __future__ import annotations

"""
Profile the relative cost of the primal and time-domain adjoint solves for a short 1-cosine gust case on the
cantilever wing, and see how both costs scale with the number of structural nodes.
"""

import copy
import csv
import time
from collections.abc import Callable
from pathlib import Path

import jax
from jax import Array
from jax import numpy as jnp

from flapjax.aero.flowfields import OneMinusCosineFlowField
from flapjax.coupled import AeroelasticCase, CoupledAeroelastic
from flapjax.coupled.data_structures import (
    AeroelasticDesignVariables,
    AeroelasticFullStates,
)
from flapjax.models.cantilever_wing.cantilever_wing import generate_cantilever_wing
from flapjax.utils.data_structures import ConvergenceSettings

N_TSTEP: int = (
    30  # evolved timesteps in the short gust case (per-step cost averages over these)
)
M_AERO: int = 4
M_STAR_AERO: int = 8
U_INF: Array = jnp.array((10.0, 0.0, 0.1))
GUST_LENGTH: float = 2.0
GUST_AMPLITUDE: float = 0.3
GUST_X0: Array = jnp.array((-5.0, 0.0, 0.0))
SKIP_DENSE_ABOVE: int = 25  # dense adjoint memory cost grows too fast to be practical beyond this many nodes

METRICS: tuple[str, ...] = ("primal", "adj_gmres", "adj_dense")
CSV_FIELDS: tuple[str, ...] = (
    "n_nodes",
    "n_dof",
    "n_span_aero",
    "metric",
    "time_s",
    "wrbm_l2",
    "d_wrbm_l2_d_gust_amplitude",
)

# stricter tolerances for better adjoint accuracy
STATIC_CONVERGENCE = ConvergenceSettings(
    max_n_iter=100,
    rel_disp_tol=1e-5,
    abs_disp_tol=1e-8,
    rel_force_tol=1e-5,
    abs_force_tol=1e-8,
)

DYNAMIC_CONVERGENCE = ConvergenceSettings(
    max_n_iter=100,
    rel_disp_tol=1e-5,
    abs_disp_tol=1e-8,
    rel_force_tol=1e-5,
    abs_force_tol=1e-8,
)


def build_wing(n_nodes: int) -> CoupledAeroelastic:
    wing = generate_cantilever_wing(
        n_nodes=n_nodes, m=M_AERO, m_star=M_STAR_AERO, u_inf=U_INF
    )
    wing.aero.flowfield = OneMinusCosineFlowField(
        u_inf=U_INF,
        rho=1.225,
        relative_motion=True,
        gust_length=GUST_LENGTH,
        gust_amplitude=GUST_AMPLITUDE,
        gust_x0=GUST_X0,
    )
    wing.structure.spectral_radius = 0.8
    wing.aero.gamma_dot_relaxation = 0.7
    return wing


def wrbm_sq_objective(
    states: AeroelasticFullStates, _dv: AeroelasticDesignVariables, _i_ts: int
) -> Array:
    # extract wing root bending moment squared, nonzero for all time steps
    return states.structure.f_elem[0, 5] ** 2


def time_fn(fn: Callable, n_warmup: int = 1, n_loops: int = 2) -> float:
    for _ in range(n_warmup):
        jax.block_until_ready(fn())
    t0 = time.perf_counter()
    for _ in range(n_loops):
        jax.block_until_ready(fn())
    return (time.perf_counter() - t0) / n_loops


def append_row(row: dict[str, object], path: Path) -> None:
    # write outputs to file as the script runs
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_FIELDS))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        fh.flush()


def make_adjoint_step(
    wing: CoupledAeroelastic,
    case: AeroelasticCase,
    matrix_free: bool,
    gmres_precond: bool = True,
    preconditioner: Callable[[Array], Array] | None = None,
) -> Callable:
    def step() -> object:
        return wing.dynamic_adjoint(
            case=case,
            objective=wrbm_sq_objective,
            matrix_free=matrix_free,
            gmres_mode="incremental",
            gmres_precond=gmres_precond,
            preconditioner=preconditioner,
            grads_to_compute=None,  # gradients for all design variables
            approx_grads=True,
            include_initial_state_grad=False,  # skip initial state sensitivities
        )

    return step


def run_config(n_nodes: int, csv_path: Path) -> None:
    wing = build_wing(n_nodes)
    n_tstep_case = N_TSTEP + 1

    wing.structure.struct_convergence_settings = STATIC_CONVERGENCE
    static_sol = wing.static_solve(
        f_ext_dead=None,
        f_ext_follower=None,
        prescribed_dofs=jnp.arange(6),
        horseshoe=True,
    )

    wing.fsi_convergence_settings = DYNAMIC_CONVERGENCE
    wing.structure.struct_convergence_settings = DYNAMIC_CONVERGENCE

    base: dict[str, int | float] = {
        "n_nodes": n_nodes,
        "n_dof": wing.structure.n_dof,
        "n_span_aero": n_nodes - 1,
        "wrbm_l2": float("nan"),
        "d_wrbm_l2_d_gust_amplitude": float("nan"),
    }

    primal_wing = copy.deepcopy(wing)
    prescribed_dofs = tuple(range(6))

    @jax.jit
    def primal_step() -> AeroelasticCase:
        return primal_wing.dynamic_solve(
            init_case=static_sol,
            prescribed_dofs=prescribed_dofs,
            n_tstep=n_tstep_case,
        )

    t_primal = time_fn(primal_step)

    case = wing.dynamic_solve(
        init_case=static_sol,
        prescribed_dofs=prescribed_dofs,
        n_tstep=n_tstep_case,
    )
    wrbm_l2 = float(jnp.sqrt(jnp.sum(case.structure.f_elem[:, 0, 5] ** 2)))

    append_row(
        {**base, "metric": "primal", "time_s": t_primal / N_TSTEP, "wrbm_l2": wrbm_l2},
        path=csv_path,
    )
    print(
        f"  primal:          {t_primal / N_TSTEP:.4e} s/step  |  WRBM L2 = {wrbm_l2:.4e}"
    )

    for metric in METRICS[1:]:
        if metric == "adj_dense" and n_nodes >= SKIP_DENSE_ABOVE:
            continue

        row: dict[str, int | float | str] = dict(base)
        if metric == "adj_gmres":
            step = make_adjoint_step(
                wing=wing,
                case=case,
                matrix_free=True,
                gmres_precond=True,
            )
        elif metric == "adj_dense":
            step = make_adjoint_step(wing=wing, case=case, matrix_free=False)
        else:
            raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")

        t_adj = time_fn(step)
        grad, j_per_ts, _ = step()
        s = jnp.sum(j_per_ts)
        l2 = jnp.sqrt(s)
        d_l2_d_gust = (
            grad.aero.flowfield["gust_amplitude"].sum() / (2.0 * l2)
            if grad.aero.flowfield is not None
            else float("nan")
        )

        row["metric"] = metric
        row["time_s"] = t_adj / N_TSTEP
        row["wrbm_l2"] = float(l2)
        row["d_wrbm_l2_d_gust_amplitude"] = float(d_l2_d_gust)
        append_row(row, path=csv_path)
        print(
            f"  {metric:16s} {t_adj / N_TSTEP:.4e} s/step  |  WRBM L2 = {float(l2):.4e}  |  "
            f"d(WRBM L2)/d(gust amplitude) = {float(d_l2_d_gust):.4e}"
        )


def run_sweep(n_nodes_values: list[int], csv_path: Path) -> None:
    for n_nodes in n_nodes_values:
        print(
            f"\nn_nodes={n_nodes} (n_dof={n_nodes * 6}, spanwise aero panels={n_nodes - 1})",
            flush=True,
        )
        run_config(n_nodes=n_nodes, csv_path=csv_path)


def plot_sweep(csv_path: Path, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_metric: dict[str, list[tuple[int, float]]] = {}
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            by_metric.setdefault(row["metric"], []).append(
                (int(row["n_nodes"]), float(row["time_s"]))
            )
    for series in by_metric.values():
        series.sort()

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))

    for metric, pts in by_metric.items():
        xs, ys = zip(*pts)
        axes[0].plot(xs, ys, marker="o", label=metric)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("number of structural nodes")
    axes[0].set_ylabel("time per step (s)")
    axes[0].set_title("Absolute cost")
    axes[0].grid(True, which="both", ls=":", lw=0.5)
    axes[0].legend()

    primal_by_n = dict(by_metric.get("primal", []))
    for metric, pts in by_metric.items():
        if metric == "primal":
            continue
        xs, ys = zip(*[(n, t / primal_by_n[n]) for n, t in pts if n in primal_by_n])
        axes[1].plot(xs, ys, marker="o", label=metric)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("number of structural nodes")
    axes[1].set_ylabel("time relative to primal")
    axes[1].set_title("Relative cost (adjoint / primal)")
    axes[1].grid(True, which="both", ls=":", lw=0.5)
    axes[1].legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    out_dir = Path(__file__).parent / "structural_scaling_output"
    csv_path_ = out_dir / "structural_sweep.csv"

    # fresh csv each run, comment out to resume a sweep
    csv_path_.unlink(missing_ok=True)

    run_sweep(n_nodes_values=[6, 11, 16, 21, 31, 41], csv_path=csv_path_)

    plot_sweep(csv_path_, out_dir / "structural_scaling.png")
