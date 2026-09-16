from __future__ import annotations

from copy import deepcopy

from jax import Array
from jax import numpy as jnp

from flapjax.aero.flowfields import ConstantFlowField
from flapjax.aero.frequency_flowfields import VonKarmanFlowField
from flapjax.aero.gradients.data_structures import AeroGradsToCompute
from flapjax.coupled import (
    AeroelasticDesignVariables,
    AeroelasticFullStates,
    AeroelasticGradsToCompute,
    CoupledAeroelastic,
    compute_gust_frf,
    gust_frf_adjoint,
)
from flapjax.coupled.linear.data_structures import AeroelasticOutputUnflattened
from flapjax.models.pazy.straight.pazy_wing import generate_pazy_wing
from flapjax.structure.gradients.data_structures import StructureGradsToCompute
from flapjax.utils.data_structures import ConvergenceSettings

M = 8
M_STAR = 40
U_INF_REF = 53.0
RHO = 1.225
AOA = jnp.deg2rad(3.0)
OMEGA = jnp.array([10.0, 50.0, 100.0])
FRF_FLOWFIELD = VonKarmanFlowField(sigma=1.0, length_scale=100.0, u_inf=U_INF_REF)


def _build_wing(u_inf_mag: float) -> CoupledAeroelastic:
    wing = generate_pazy_wing(
        flowfield=ConstantFlowField(
            u_inf=jnp.array((u_inf_mag, 0.0, 0.0)),
            rho=RHO,
            relative_motion=True,
        ),
        aoa=AOA,
        m=M,
        m_star=M_STAR,
        node_multiplier=1,
        skin=True,
    )
    conv = ConvergenceSettings(
        max_n_iter=50,
        abs_disp_tol=1e-10,
        rel_disp_tol=1e-8,
        abs_force_tol=1e-9,
        rel_force_tol=1e-8,
    )
    wing.structure.struct_convergence_settings = conv
    wing.fsi_convergence_settings = conv
    return wing


def _apply_dv(
    wing: CoupledAeroelastic,
    u_inf_mag: float | Array,
    k_cs: Array,
) -> None:
    wing.set_design_variables(
        coords=wing.structure.x0,
        k_cs=k_cs,
        m_cs=wing.structure.m_cs,
        m_lumped=None,
        dt=wing.aero.dt,
        flowfield=ConstantFlowField(
            u_inf=jnp.array((u_inf_mag, 0.0, 0.0)),
            rho=RHO,
            relative_motion=True,
        ),
        x0_aero=wing.aero.zeta_b0,
        remove_checks=True,
    )


def _gust_objective(
    states: AeroelasticFullStates,
    dv: AeroelasticDesignVariables,
    h_gust: AeroelasticOutputUnflattened,
) -> Array:
    """Sum of squared magnitudes of the gust FRF over all frequencies."""
    del states, dv
    return jnp.sum(jnp.abs(h_gust.q) ** 2) + jnp.sum(jnp.abs(h_gust.q_dot) ** 2)


def compute_objective_gust(
    base_wing: CoupledAeroelastic,
    u_inf_mag: float,
    k_cs: Array,
    omega: Array,
) -> float:
    wing = deepcopy(base_wing)
    _apply_dv(wing, u_inf_mag=u_inf_mag, k_cs=k_cs)
    static_sol = wing.static_solve(prescribed_dofs=tuple(range(6)), horseshoe=False)
    dv = wing.get_design_variables(case=static_sol, grads_to_compute=None)
    h = compute_gust_frf(
        wing, dv, static_sol.structure.varphi, static_sol, omega, FRF_FLOWFIELD
    )
    states = static_sol.get_full_states(i_ts=None)
    return float(_gust_objective(states=states, dv=dv, h_gust=h))


class TestGustFRFAdjoint:
    grads_to_compute: AeroelasticGradsToCompute = AeroelasticGradsToCompute(
        structure=StructureGradsToCompute(k_cs=True),
        aero=AeroGradsToCompute(x0_aero=False, flowfield=True),
    )

    @classmethod
    def setup_class(cls):
        cls.wing = _build_wing(U_INF_REF)
        _apply_dv(cls.wing, u_inf_mag=U_INF_REF, k_cs=cls.wing.structure.k_cs)
        cls.k_cs_base: Array = cls.wing.structure.k_cs
        cls.static_sol = cls.wing.static_solve(
            prescribed_dofs=tuple(range(6)), horseshoe=False
        )
        cls.j_adj, cls.dj_ddv = gust_frf_adjoint(
            system=cls.wing,
            case=cls.static_sol,
            omega=OMEGA,
            objective=_gust_objective,
            frf_flowfield=FRF_FLOWFIELD,
            grads_to_compute=cls.grads_to_compute,
        )

    @classmethod
    def test_primal_matches_pipeline(cls):
        """Adjoint and direct pipeline should yield the same primal objective."""
        j_pipe = compute_objective_gust(cls.wing, U_INF_REF, cls.k_cs_base, OMEGA)
        assert jnp.isclose(cls.j_adj, j_pipe, rtol=1e-4), (
            f"Primal mismatch: adjoint={float(cls.j_adj)}, pipeline={j_pipe}"
        )

    @classmethod
    def test_u_inf_gradient(cls):
        """FD check on freestream u_inf."""
        h = 1e-3 * U_INF_REF
        j_plus = compute_objective_gust(cls.wing, U_INF_REF + h, cls.k_cs_base, OMEGA)
        j_minus = compute_objective_gust(cls.wing, U_INF_REF - h, cls.k_cs_base, OMEGA)
        fd_grad = (j_plus - j_minus) / (2.0 * h)

        assert cls.dj_ddv.aero.flowfield is not None
        adj_grad = float(jnp.real(cls.dj_ddv.aero.flowfield["u_inf"][0]))

        rel_err = abs(fd_grad - adj_grad) / abs(fd_grad)

        assert rel_err < 1e-2, (
            f"u_inf_x gradient mismatch: adjoint={adj_grad}, FD={fd_grad}, "
            f"rel_err={rel_err:.3e}"
        )

    @classmethod
    def test_k_bending_gradient(cls):
        """FD check on out-of-plane bending stiffness k_cs[:, 4, 4]."""
        eps = 1e-3
        k_plus = cls.k_cs_base.at[:, 4, 4].add(eps)
        k_minus = cls.k_cs_base.at[:, 4, 4].add(-eps)

        j_plus = compute_objective_gust(cls.wing, U_INF_REF, k_plus, OMEGA)
        j_minus = compute_objective_gust(cls.wing, U_INF_REF, k_minus, OMEGA)
        fd_grad = (j_plus - j_minus) / (2.0 * eps)

        assert cls.dj_ddv.structure.k_cs is not None
        adj_grad = float(cls.dj_ddv.structure.k_cs[:, 4, 4].sum())

        rel_err = abs(fd_grad - adj_grad) / abs(fd_grad)

        assert rel_err < 5e-2, (
            f"k_cs[:, 4, 4] gradient mismatch: adjoint={adj_grad}, FD={fd_grad}, "
            f"rel_err={rel_err:.3e}"
        )
