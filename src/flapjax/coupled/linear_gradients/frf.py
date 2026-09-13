from __future__ import annotations

from collections.abc import Callable

import jax
import numpy as np
from jax import Array
from jax import numpy as jnp

from flapjax.aero.frequency_flowfields import FrequencyFlowField
from flapjax.coupled.data_structures import (
    AeroelasticCase,
    AeroelasticDesignVariables,
    AeroelasticFullStates,
)
from flapjax.coupled.gradients.coupled import CoupledAeroelastic
from flapjax.coupled.gradients.data_structures import AeroelasticGradsToCompute
from flapjax.coupled.linear.data_structures import AeroelasticOutputUnflattened
from flapjax.coupled.linear_gradients.stability import build_reference_case
from flapjax.structure.utils import get_solve_dofs

type FRFObjective = Callable[
    [AeroelasticFullStates, AeroelasticDesignVariables, AeroelasticOutputUnflattened],
    Array,
]


def gust_penetration_vector(
    omega: Array,
    vertex_x: Array,
    u_inf: float | Array,
) -> Array:
    r"""
    Build the frequency-dependent gust-to-upwash mapping vector with delay.

    :param omega: Sampling frequencies in rad/s, ``(n_freq,)``.
    :param vertex_x: Streamwise coordinate of each grid vertex, ``(n_vertices,)``.
    :param u_inf: Freestream velocity magnitude in m/s.
    :return: Delayed vector, ``(n_freq, 3 * n_vertices)``.
    """
    n_verts = vertex_x.shape[0]
    phase = jnp.exp(-1j * omega[:, None] * vertex_x[None, :] / u_inf)
    g_full = jnp.zeros((omega.shape[0], 3 * n_verts), dtype=complex)
    g_full = g_full.at[:, 2::3].set(phase)
    return g_full


def _compute_gust_frf_base(
    system: CoupledAeroelastic,
    dv: AeroelasticDesignVariables,
    varphi: Array,
    case: AeroelasticCase,
    omega: Array,
    frf_flowfield: FrequencyFlowField | None = None,
    batch_size: int | None = 4,
) -> Array:
    ref, inner = build_reference_case(system, dv, varphi, case)
    linear = inner.linearise(
        reference=ref, skip_checks=True, batch_size=batch_size, n_struct_modes=None
    )
    return linear.frf_base(omega=omega, flowfield=frf_flowfield)


def compute_gust_frf(
    system: CoupledAeroelastic,
    dv: AeroelasticDesignVariables,
    varphi: Array,
    case: AeroelasticCase,
    omega: Array,
    frf_flowfield: FrequencyFlowField | None = None,
    batch_size: int | None = 4,
) -> AeroelasticOutputUnflattened:
    r"""
    Compute the gust frequency response function using JVP through the coupled step.

    When ``frf_flowfield`` is provided the result is PSD-weighted (each
    frequency row scaled by ``sqrt(psd(omega))``). When ``None``, the raw
    transfer function is returned.

    :param system: The coupled aeroelastic system.
    :param dv: Aeroelastic design variables.
    :param varphi: Structural configuration, ``(n_nodes, 6)``.
    :param case: Converged aeroelastic case around which to linearise.
    :param omega: Angular frequencies in rad/s, ``(n_freq,)``.
    :param frf_flowfield: Frequency-domain turbulence spectrum. If ``None``,
        the raw transfer function is returned.
    :param batch_size: Batch size for Jacobian materialisation of the A matrix.
    :return: Gust FRF with ``q`` and ``q_dot`` fields.
    """
    ref, inner = build_reference_case(system, dv, varphi, case)
    linear = inner.linearise(
        reference=ref, skip_checks=True, batch_size=batch_size, n_struct_modes=None
    )
    return linear.frf(omega=omega, flowfield=frf_flowfield)


def gust_frf_adjoint(
    system: CoupledAeroelastic,
    case: AeroelasticCase,
    omega: Array,
    objective: FRFObjective,
    frf_flowfield: FrequencyFlowField | None = None,
    grads_to_compute: AeroelasticGradsToCompute | None = None,
    batch_size: int = 32,
) -> tuple[Array, AeroelasticDesignVariables]:
    r"""
    Compute sensitivities of an objective that depends on the gust FRF with
    respect to design variables.
    :param system: The coupled aeroelastic system.
    :param case: Converged static solution around which to linearise.
    :param omega: Frequencies to sample in rad/s, ``(n_freq,)``.
    :param objective: Function ``(full_states, design_variables, H_gust) -> scalar``
        where ``H_gust`` is the complex gust FRF, ``(n_freq, n_outputs)``.
    :param frf_flowfield: Frequency-domain turbulence spectrum. If ``None``,
        the raw transfer function is passed to the objective.
    :param grads_to_compute: Which design variable gradients to request.
    :param batch_size: Batch size for Jacobian materialisation.
    :return: Primal objective value and its gradient w.r.t. design variables.
    """
    if grads_to_compute is None:
        grads_to_compute = AeroelasticGradsToCompute()

    varphi_eq = case.structure.varphi
    dv_ref = system.get_design_variables(case=case, grads_to_compute=grads_to_compute)
    n_dof = system.structure.n_dof
    solve_dofs = jnp.array(
        get_solve_dofs(
            n_dof=n_dof,
            prescribed_dofs=case.structure.prescribed_dofs,
        )
    )

    # compute primal gust FRF
    h_gust = _compute_gust_frf_base(
        system, dv_ref, varphi_eq, case, omega, frf_flowfield
    )

    n_out = h_gust.shape[1] // 2

    def _unpack_h(h_: Array) -> AeroelasticOutputUnflattened:
        return AeroelasticOutputUnflattened(q=h_[:, :n_out], q_dot=h_[:, n_out:])

    def _objective_of_dv_h_varphi(
        dv_: AeroelasticDesignVariables, h_: Array, varphi_flat_: Array
    ) -> Array:
        # differentiable object w.r.t. its arguments
        states_, _ = system.aeroelastic_states_res_from_dv_varphi(
            dv=dv_,
            varphi=varphi_flat_.reshape(-1, 6),
            thrust=case.structure.thrust,
            t=case.aero.t,
            i_ts=0,
            use_horseshoe=False,
        )
        return objective(states_, dv_, _unpack_h(h_))

    j_val, vjp_j = jax.vjp(_objective_of_dv_h_varphi, dv_ref, h_gust, varphi_eq.ravel())

    j_shape = j_val.shape
    n_f = max(1, int(np.prod(j_shape)))
    d_j_d_x_direct_b, d_j_d_h_b, d_j_d_varphi_direct_b = jax.vmap(vjp_j)(
        jnp.eye(n_f).reshape((n_f,) + j_shape)
    )

    def _gust_frf_fn(dv_: AeroelasticDesignVariables, varphi_: Array) -> Array:
        # differentiate through FRF computation
        return _compute_gust_frf_base(system, dv_, varphi_, case, omega, frf_flowfield)

    _, vjp_gust = jax.vjp(_gust_frf_fn, dv_ref, varphi_eq)
    dv_bar_gust_b, varphi_bar_gust_b = jax.vmap(vjp_gust)(d_j_d_h_b)

    def _residual_of_varphi(varphi_vec: Array) -> Array:
        # differentiate full states w.r.t. static deformation
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

    varphi_bar_b = jnp.real(varphi_bar_gust_b.reshape(n_f, -1)) + jnp.real(
        d_j_d_varphi_direct_b.reshape(n_f, -1)
    )
    varphi_bar_free_b = varphi_bar_b[:, solve_dofs]

    j_res_free = p_res_p_varphi[jnp.ix_(solve_dofs, solve_dofs)]
    mu_free_b = jnp.linalg.solve(j_res_free.T, varphi_bar_free_b.T).T
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

    def _neg_if_float(x):
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating):
            return -x
        return x

    dv_bar_via_eq_b = jax.tree.map(_neg_if_float, dv_bar_via_eq_pos_b)

    def _sum_real(direct, via_gust, via_eq):
        if hasattr(direct, "dtype") and jnp.issubdtype(direct.dtype, jnp.floating):
            return direct + jnp.real(via_gust) + via_eq
        return direct

    total_b = jax.tree.map(_sum_real, d_j_d_x_direct_b, dv_bar_gust_b, dv_bar_via_eq_b)

    def _to_j_shape(leaf):
        if not hasattr(leaf, "shape") or leaf.ndim == 0:
            return leaf
        return leaf.reshape(j_shape + leaf.shape[1:])

    total = jax.tree.map(_to_j_shape, total_b)

    return j_val, total
