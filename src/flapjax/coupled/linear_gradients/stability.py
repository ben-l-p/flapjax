from __future__ import annotations

from collections.abc import Callable

import jax
import numpy as np
from jax import Array
from jax import numpy as jnp

from flapjax.coupled.coupled import BaseCoupledAeroelastic
from flapjax.coupled.data_structures import (
    AeroelasticCase,
    AeroelasticDesignVariables,
    AeroelasticFullStates,
)
from flapjax.coupled.gradients.coupled import CoupledAeroelastic
from flapjax.coupled.gradients.data_structures import AeroelasticGradsToCompute
from flapjax.structure.utils import get_solve_dofs, transform_nodal_vect
from flapjax.utils.utils import pytree_clone

type StabilityObjective = Callable[
    [AeroelasticFullStates, AeroelasticDesignVariables, Array], Array
]


def build_reference_case(
    system: CoupledAeroelastic,
    dv: AeroelasticDesignVariables,
    varphi: Array,
    case: AeroelasticCase,
) -> tuple[AeroelasticCase, BaseCoupledAeroelastic]:
    r"""
    Rebuild the linearisation solution and aeroelastic object as a function of the design variables and static
    deformation.
    :param system: Coupled aeroelastic system.
    :param dv: Aeroelastic design variables.
    :param varphi: Structural configuration, ``(n_nodes, 6)``.
    :param case: A converged case used for linearisation.
    :return: Solution object for reference equilibrium and the inner coupled aeroelastic object with passed design
     variables.
    """
    inner = system.case_from_dv(dv)
    hg = inner.structure.compute_hg_from_varphi(varphi=varphi)
    aero_sol = inner.aero.static_solve(
        t=case.aero.t,
        hg=hg,
        horseshoe=False,
        cs_ang=inner.aero.cs_ang0,
    )

    # aero forcing, transformed to local frame
    f_ext_aero_global = aero_sol.project_forcing_to_beam(
        i_ts=0,
        rmat=hg[:, :3, :3],
        x0_aero=inner.aero.zeta_b0,
        include_unsteady=False,
    )
    f_ext_aero_local = transform_nodal_vect(
        f_ext_aero_global, jnp.transpose(hg[:, :3, :3], (0, 2, 1))
    )

    # find the structural states so that they propogate the derivatives correctly
    d_arr = inner.structure.make_d(hg)
    eps_arr = inner.structure.make_eps(d=d_arr)

    struct_case = pytree_clone(case.structure)
    struct_case.hg = hg
    struct_case.varphi = varphi
    struct_case.d = d_arr
    struct_case.eps = eps_arr
    struct_case.f_ext_aero = f_ext_aero_local

    return AeroelasticCase(structure=struct_case, aero=aero_sol), inner


def assemble_a(
    system: CoupledAeroelastic,
    dv: AeroelasticDesignVariables,
    varphi: Array,
    case: AeroelasticCase,
    n_struct_modes: int | None = None,
    batch_size: int | None = 4,
) -> Array:
    r"""
    Assemble the discrete-time aeroelastic system matrix as a function of deformation and design variables.
    :param system: The coupled aeroelastic system.
    :param dv: Aeroelastic design variables.
    :param varphi: Structural configuration, ``(n_nodes, 6)``.
    :param case: A converged aeroelastic case around which to linearise.
    :param n_struct_modes: Structural mode count. Currently not supported.
    :param batch_size: Batch size for constructing the Jacobian.
    :return: Discrete-time system matrix, ``(n_states, n_states)``.
    """
    if n_struct_modes is not None:
        raise NotImplementedError(
            "Modal reduction for structural system with derivatives not supported"
        )

    # find the aeroelastic object and the reference case for linearisation
    ref, inner = build_reference_case(system, dv, varphi, case)

    # linearise system and return the system matrix
    linear = inner.linearise(
        reference=ref,
        skip_checks=True,
        batch_size=batch_size,
        n_struct_modes=n_struct_modes,
    )
    return linear.sys.a


def eig_left_right(a: Array) -> tuple[Array, Array, Array]:
    r"""
    Compute right and left eigenvectors of a square matrix, matched by nearest
    eigenvalue.
    :param a: Square matrix, ``(n, n)``.
    :return: Tuple of eigenvalues, right eigenvectors, and left eigenvectors, with ordering matched by nearest
    eigenvalue.
    """
    lam_r, phi_r = jnp.linalg.eig(a)
    lam_l, phi_l = jnp.linalg.eig(a.T)
    n = lam_r.shape[0]

    def body(i, carry):
        # function to loop to perform matching of left and right eigenvectors by nearest eigenvalue
        used, idx_ = carry
        d = jnp.where(used, jnp.inf, jnp.abs(lam_r[i] - lam_l))
        j = jnp.argmin(d)
        return used.at[j].set(True), idx_.at[i].set(j)

    used0 = jnp.zeros(n, dtype=bool)
    idx0 = jnp.zeros(n, dtype=int)
    _, idx = jax.lax.fori_loop(0, n, body, (used0, idx0))
    return lam_r, phi_r, phi_l[:, idx]


def stability_adjoint(
    system: CoupledAeroelastic,
    case: AeroelasticCase,
    objective: StabilityObjective,
    grads_to_compute: AeroelasticGradsToCompute | None = None,
    batch_size: int = 32,
) -> tuple[Array, AeroelasticDesignVariables]:
    r"""
    Compute the sensitivities of some objective that refers to the aeroelastic continuous-time eigenvalues, full system
    states and design variables, with respect to the design variables.
    :param system: The coupled aeroelastic system.
    :param case: Converged static solution around which to linearise.
    :param objective: Function which takes the full aeroelastic states, aeroelastic design variables, and the full
    continuous-time eigenvalue vector, and returns a real-valued objective.
    :param grads_to_compute: Which design variable gradients to request.
    :param batch_size: Batch size for Jacobian materialisation.
    :return: Primal value of the objective, and its gradient with respect to the design variables.
    """
    n_struct_modes = None  # model structure not implemented

    if grads_to_compute is None:
        grads_to_compute = AeroelasticGradsToCompute()

    # extract base parameters
    varphi_eq = case.structure.varphi
    dv_ref = system.get_design_variables(case=case, grads_to_compute=grads_to_compute)
    dt = system.aero.dt
    n_dof = system.structure.n_dof
    solve_dofs = jnp.array(
        get_solve_dofs(
            n_dof=n_dof,
            prescribed_dofs=case.structure.prescribed_dofs,
        )
    )

    # assemble system matrix and compute eigendecomposition
    a_matrix = assemble_a(
        system,
        dv_ref,
        varphi_eq,
        case,
        n_struct_modes=n_struct_modes,
    )
    lam_d, phi_r, phi_l = eig_left_right(a_matrix)
    lam_c = jnp.log(lam_d) / dt  # continuous time eigenvalues

    def _objective_of_dv_lam_varphi(
        dv_: AeroelasticDesignVariables, lam_c_: Array, varphi_flat_: Array
    ) -> Array:
        states_, _ = system.aeroelastic_states_res_from_dv_varphi(
            dv=dv_,
            varphi=varphi_flat_.reshape(-1, 6),
            thrust=case.structure.thrust,
            t=case.aero.t,
            i_ts=0,
            use_horseshoe=False,
        )
        return objective(states_, dv_, lam_c_)

    # VJP of the objective against it's arguments
    j_val, vjp_j = jax.vjp(
        _objective_of_dv_lam_varphi, dv_ref, lam_c, varphi_eq.ravel()
    )

    # allow for arbitrary shape
    j_shape = j_val.shape
    n_f = max(1, int(np.prod(j_shape)))
    d_j_d_x_direct_b, d_j_d_lambda_c_b, d_j_d_varphi_direct_b = jax.vmap(vjp_j)(
        jnp.eye(n_f).reshape((n_f,) + j_shape)
    )  # sensitivities through objective direct path

    # build one A-cotangent per output row, with filter to ignore defective modes
    c_denom = jnp.einsum("ij,ij->j", phi_l, phi_r)
    denom_ok = jnp.abs(c_denom) > 1e-12

    def _make_a_bar_row(dj_row: Array) -> Array:
        w_d = jnp.conj(dj_row) / (dt * lam_d)
        coeff = jnp.where(denom_ok, w_d / c_denom, 0.0)
        return jnp.real(phi_l @ jnp.diag(coeff) @ phi_r.T)

    a_bar_b = jax.vmap(_make_a_bar_row)(d_j_d_lambda_c_b)  # (n_f, N, N)

    # create a JVP for the system matrix construction
    def _a_fn(dv_, varphi_):
        return assemble_a(
            system,
            dv_,
            varphi_,
            case,
            n_struct_modes=n_struct_modes,
        )

    _, vjp_a = jax.vjp(_a_fn, dv_ref, varphi_eq)
    dv_bar_a_b, varphi_bar_a_b = jax.vmap(vjp_a)(a_bar_b)  # leading (n_f,)

    # static residual Jacobian
    def _residual_of_varphi(varphi_vec: Array) -> Array:
        return system.aeroelastic_states_res_from_dv_varphi(
            dv=dv_ref,
            varphi=varphi_vec.reshape(-1, 6),
            thrust=case.structure.thrust,
            t=case.aero.t,
            i_ts=0,
            use_horseshoe=False,
        )[1]

    _, vjp_res_v = jax.vjp(_residual_of_varphi, varphi_eq.ravel())
    p_res_p_varphi = jax.lax.map(
        lambda cot: vjp_res_v(cot)[0], jnp.eye(n_dof), batch_size=batch_size
    )

    # combine the sensitivities with respect to varphi from the full states in the objective and the path through the
    # system matrix
    varphi_bar_b = jnp.real(varphi_bar_a_b.reshape(n_f, -1)) + jnp.real(
        d_j_d_varphi_direct_b.reshape(n_f, -1)
    )  # (n_f, n_dof)
    varphi_bar_free_b = varphi_bar_b[:, solve_dofs]  # (n_f, n_free)

    j_res_free = p_res_p_varphi[jnp.ix_(solve_dofs, solve_dofs)]
    mu_free_b = jnp.linalg.solve(j_res_free.T, varphi_bar_free_b.T).T  # (n_f, n_free)
    mu_full_b = (
        jnp.zeros((n_f, n_dof), dtype=mu_free_b.dtype).at[:, solve_dofs].set(mu_free_b)
    )

    _, vjp_res_dv = jax.vjp(
        lambda dv_: system.aeroelastic_states_res_from_dv_varphi(
            dv=dv_,
            varphi=varphi_eq,
            thrust=case.structure.thrust,
            t=case.aero.t,
            i_ts=0,
            use_horseshoe=False,
        )[1],
        dv_ref,
    )
    (dv_bar_via_eq_pos_b,) = jax.vmap(vjp_res_dv)(mu_full_b)

    # handles sign flip for floating point types
    def _neg_if_float(x):
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating):
            return -x
        return x

    dv_bar_via_eq_b = jax.tree.map(_neg_if_float, dv_bar_via_eq_pos_b)

    # combine contributions
    def _sum_real(direct, via_a, via_eq):
        if hasattr(direct, "dtype") and jnp.issubdtype(direct.dtype, jnp.floating):
            return direct + jnp.real(via_a) + via_eq
        return direct

    total_b = jax.tree.map(_sum_real, d_j_d_x_direct_b, dv_bar_a_b, dv_bar_via_eq_b)

    # reshape back to original shape
    def _to_j_shape(leaf):
        if not hasattr(leaf, "shape") or leaf.ndim == 0:
            return leaf
        return leaf.reshape(j_shape + leaf.shape[1:])

    total = jax.tree.map(_to_j_shape, total_b)

    return j_val, total
