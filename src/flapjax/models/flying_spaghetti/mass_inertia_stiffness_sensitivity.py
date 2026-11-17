"""
Sensitivity of the flying-spaghetti tip position w.r.t. mass per unit length,
rotational inertia per unit length, and bending stiffness, via the structural
discrete adjoint (``BeamStructure.dynamic_adjoint``), cross-checked against
central finite differences (FD).
"""

from __future__ import annotations

import time

import jax
from jax import Array
from jax import numpy as jnp

from flapjax.models.flying_spaghetti.flying_spaghetti import generate_flying_spaghetti
from flapjax.structure import StructureDesignVariables, StructureFullStates
from flapjax.structure.data_structures import StructureCase
from flapjax.structure.gradients.data_structures import StructureGradsToCompute
from flapjax.utils.print_utils import set_verbosity

set_verbosity("warning")

# case parameters
N_NODES = 21
DT = 0.01
T_END = 10.0
SPECTRAL_RADIUS = 0.7
N_TSTEP = int(jnp.ceil(T_END / DT)) + 1

# baseline cross-sectional properties
EA = GAY = GAZ = 10e4
GJ = 500.0
EIY = EIZ = 500.0
M_BAR0 = 1.0
J0 = jnp.array((20.0, 10.0, 10.0))

# relative step size for central finite differences
FD_STEP = 1.0e-3


def build_m_cs(mass_scale: float, inertia_scale: float) -> Array:
    return jnp.diag(
        jnp.array(
            (
                mass_scale * M_BAR0,
                mass_scale * M_BAR0,
                mass_scale * M_BAR0,
                inertia_scale * J0[0],
                inertia_scale * J0[1],
                inertia_scale * J0[2],
            )
        )
    )


def build_k_cs(bending_scale: float) -> Array:
    return jnp.diag(
        jnp.array((EA, GAY, GAZ, GJ, bending_scale * EIY, bending_scale * EIZ))
    )


def tip_x_cost(
    states: StructureFullStates,
    _dv: StructureDesignVariables,
    i_ts: int,
) -> Array:
    # x-coordinate of the free tip node (last node) at the final timestep only
    return jax.lax.select(i_ts == N_TSTEP - 1, states.x[-1, 0], 0.0)


def solve(
    struct,
    f_dead_2d: Array,
    mass_scale: float,
    inertia_scale: float,
    bending_scale: float,
) -> StructureCase:
    struct.set_design_variables(
        coords=struct.x0_reference,
        k_cs=build_k_cs(bending_scale),
        m_cs=build_m_cs(mass_scale, inertia_scale),
        m_lumped=None,
    )
    return struct.dynamic_solve(
        init_state=None,
        n_tstep=N_TSTEP,
        dt=DT,
        f_ext_follower=None,
        f_ext_dead=f_dead_2d,
        f_ext_aero=None,
        prescribed_dofs=(),
    )


def tip_x_final(
    struct,
    f_dead_2d: Array,
    mass_scale: float,
    inertia_scale: float,
    bending_scale: float,
) -> float:
    sol = solve(struct, f_dead_2d, mass_scale, inertia_scale, bending_scale)
    return float(sol.x[-1, -1, 0])


def central_fd(
    struct,
    f_dead_2d: Array,
    base: tuple[float, float, float],
    i_var: int,
    label: str,
) -> float:
    plus = list(base)
    minus = list(base)
    plus[i_var] += FD_STEP
    minus[i_var] -= FD_STEP

    t0 = time.perf_counter()
    cost_plus = tip_x_final(struct, f_dead_2d, *plus)
    cost_minus = tip_x_final(struct, f_dead_2d, *minus)
    print(f"  FD [{label}]: {time.perf_counter() - t0:.1f}s")

    return (cost_plus - cost_minus) / (2.0 * FD_STEP)


def main() -> None:
    t = jnp.arange(N_TSTEP) * DT
    struct, f_dead_2d, _ = generate_flying_spaghetti(
        n_nodes=N_NODES, t=t, spectral_radius=SPECTRAL_RADIUS
    )

    base_scales = (1.0, 1.0, 1.0)  # (mass_scale, inertia_scale, bending_scale)

    # baseline primal solve, reused for the adjoint
    t0 = time.perf_counter()
    sol0 = solve(struct, f_dead_2d, *base_scales)
    cost0 = float(sol0.x[-1, -1, 0])
    print(
        f"Baseline primal solve: {time.perf_counter() - t0:.1f}s, tip x = {cost0:.6f}"
    )

    # discrete adjoint: single backward sweep gives d(tip_x)/d(m_cs) and d(tip_x)/d(k_cs)
    t0 = time.perf_counter()
    grads, _ = struct.dynamic_adjoint(
        structure=sol0,
        objective=tip_x_cost,
        p_q0_p_x=None,
        approx_grads=False,
        grads_to_compute=StructureGradsToCompute(
            x0=False,
            orientation_euler=False,
            k_cs=True,
            m_cs=True,
            m_lumped=False,
            f_ext_follower=False,
            f_ext_dead=False,
            thrust_t=False,
        ),
    )
    print(f"Adjoint sweep: {time.perf_counter() - t0:.1f}s")

    assert grads.m_cs is not None and grads.k_cs is not None
    m_cs_grad = grads.m_cs[0, 0]  # (6, 6), d(tip_x)/d(m_cs[i, i])
    k_cs_grad = grads.k_cs[0, 0]  # (6, 6), d(tip_x)/d(k_cs[i, i])

    # chain rule back to the scalar scale factors
    adjoint_grads = {
        "mass_scale": float(
            (m_cs_grad[0, 0] + m_cs_grad[1, 1] + m_cs_grad[2, 2]) * M_BAR0
        ),
        "inertia_scale": float(
            m_cs_grad[3, 3] * J0[0] + m_cs_grad[4, 4] * J0[1] + m_cs_grad[5, 5] * J0[2]
        ),
        "bending_scale": float(k_cs_grad[4, 4] * EIY + k_cs_grad[5, 5] * EIZ),
    }

    # finite-difference cross-check
    fd_grads = {}
    for i_var, label in enumerate(("mass_scale", "inertia_scale", "bending_scale")):
        fd_grads[label] = central_fd(struct, f_dead_2d, base_scales, i_var, label)

    print()
    print(f"{'design variable':<16} {'adjoint':>14} {'FD':>14} {'rel. error':>12}")
    for label in ("mass_scale", "inertia_scale", "bending_scale"):
        a = adjoint_grads[label]
        f = fd_grads[label]
        rel_err = abs(a - f) / abs(f)
        print(f"{label:<16} {a:>14.6e} {f:>14.6e} {rel_err:>12.3%}")


if __name__ == "__main__":
    main()
