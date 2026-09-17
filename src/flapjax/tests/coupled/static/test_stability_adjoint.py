from __future__ import annotations

from copy import deepcopy

from jax import Array
from jax import numpy as jnp

from flapjax.aero.flowfields import ConstantFlowField
from flapjax.aero.gradients.data_structures import AeroGradsToCompute
from flapjax.coupled import (
    AeroelasticDesignVariables,
    AeroelasticFullStates,
    AeroelasticGradsToCompute,
    CoupledAeroelastic,
    stability_adjoint,
)
from flapjax.models.pazy.straight.pazy_wing import generate_pazy_wing
from flapjax.structure.gradients.data_structures import StructureGradsToCompute
from flapjax.utils.data_structures import ConvergenceSettings

M = 8
M_STAR = 40
U_INF_REF = 53.0
RHO = 1.225
AOA = jnp.deg2rad(3.0)
RHO_KS = 20.0


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
    # strict tolerances
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


def _objective(
    states: AeroelasticFullStates,
    dv: AeroelasticDesignVariables,
    lambda_c: Array,
) -> Array:
    """KS aggregation of Re(lambda_c) over all continuous-time eigenvalues."""
    del states, dv  # unused
    return jnp.log(jnp.sum(jnp.exp(RHO_KS * jnp.real(lambda_c)))) / RHO_KS


def compute_objective(
    base_wing: CoupledAeroelastic, u_inf_mag: float, k_cs: Array
) -> float:
    wing = deepcopy(base_wing)
    _apply_dv(wing, u_inf_mag=u_inf_mag, k_cs=k_cs)
    static_sol = wing.static_solve(prescribed_dofs=tuple(range(6)), horseshoe=False)
    linear_sol = wing.linearise(
        reference=static_sol,
        skip_checks=True,
        batch_size=4,
        n_struct_modes=None,
    )
    lam_c = linear_sol.modal()
    dv = wing.get_design_variables(case=static_sol, grads_to_compute=None)
    states = static_sol.get_full_states(i_ts=None)
    return float(_objective(states=states, dv=dv, lambda_c=lam_c))


class TestStabilityAdjoint:
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
        cls.j_adj, cls.dj_ddv = stability_adjoint(
            system=cls.wing,
            case=cls.static_sol,
            objective=_objective,
            grads_to_compute=cls.grads_to_compute,
        )

    @classmethod
    def test_primal_matches_pipeline(cls):
        """The two solution paths (adjoint and FD) should yield the same primal objective value."""
        j_pipe = compute_objective(cls.wing, U_INF_REF, cls.k_cs_base)
        assert jnp.isclose(cls.j_adj, j_pipe, rtol=1e-5), (
            f"Primal mismatch: adjoint={float(cls.j_adj)}, pipeline={j_pipe}"
        )

    @classmethod
    def test_u_inf_gradient(cls):
        """FD check on freestream u_inf. Note that the timestep length is kept constant."""
        h = 1e-3 * U_INF_REF
        j_plus = compute_objective(cls.wing, U_INF_REF + h, cls.k_cs_base)
        j_minus = compute_objective(cls.wing, U_INF_REF - h, cls.k_cs_base)
        fd_grad = (j_plus - j_minus) / (2.0 * h)

        assert cls.dj_ddv.aero.flowfield is not None
        adj_grad = float(jnp.real(cls.dj_ddv.aero.flowfield["u_inf"][0]))

        rel_err = abs(fd_grad - adj_grad) / max(abs(fd_grad), 1e-30)
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

        j_plus = compute_objective(cls.wing, U_INF_REF, k_plus)
        j_minus = compute_objective(cls.wing, U_INF_REF, k_minus)
        fd_grad = (j_plus - j_minus) / (2.0 * eps)

        assert cls.dj_ddv.structure.k_cs is not None
        adj_grad = float(cls.dj_ddv.structure.k_cs[:, 4, 4].sum())

        rel_err = abs(fd_grad - adj_grad) / abs(fd_grad)
        assert rel_err < 5e-2, (
            f"k_cs[:, 4, 4] gradient mismatch: adjoint={adj_grad}, FD={fd_grad}, "
            f"rel_err={rel_err:.3e}"
        )
