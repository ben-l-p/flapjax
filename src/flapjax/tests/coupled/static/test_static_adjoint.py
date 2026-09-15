from typing import Literal

from jax import Array
from jax import numpy as jnp

from flapjax.aero.gradients.data_structures import AeroGradsToCompute
from flapjax.coupled.data_structures import (
    AeroelasticDesignVariables,
    AeroelasticFullStates,
)
from flapjax.coupled.gradients.data_structures import AeroelasticGradsToCompute
from flapjax.models.cantilever_wing.cantilever_wing import generate_cantilever_wing
from flapjax.structure.gradients.data_structures import StructureGradsToCompute
from flapjax.utils.data_structures import ConvergenceSettings

# Small discretisation so tests run quickly
n_nodes = 6
m = 4
m_star = 16
u_inf_base = jnp.array((10.0, 0.0, 0.1))
k_cs_base = jnp.diag(jnp.array((1e6, 1e6, 1e6, 4e2, 4e2, 4e2)))


def _objective(states: AeroelasticFullStates, *_) -> Array:
    """Scalar objective: z-component of root internal force."""
    return states.structure.f_elem[0, 2]


def _solve(u_inf: Array, k_cs: Array):
    wing = generate_cantilever_wing(
        n_nodes=n_nodes, m=m, m_star=m_star, u_inf=u_inf, k_cs=k_cs
    )

    # strict convergence
    conv_settings = ConvergenceSettings(
        max_n_iter=25,
        abs_disp_tol=1e-9,
        rel_disp_tol=1e-7,
        abs_force_tol=1e-9,
        rel_force_tol=1e-7,
    )
    wing.structure.struct_convergence_settings = conv_settings
    wing.fsi_convergence_settings = conv_settings

    sol = wing.static_solve(
        f_ext_dead=None,
        f_ext_follower=None,
        prescribed_dofs=jnp.arange(6),
        horseshoe=True,
    )
    return wing, sol


class TestForwardStaticAeroelasticAdjoint:
    ad_mode: Literal["forward", "reverse"] = "forward"

    grads_to_compute: AeroelasticGradsToCompute = AeroelasticGradsToCompute(
        structure=StructureGradsToCompute(k_cs=True),
        aero=AeroGradsToCompute(flowfield=True),
    )

    @classmethod
    def setup_class(cls):
        cls.wing, cls.sol = _solve(u_inf=u_inf_base, k_cs=k_cs_base)
        cls.grad: AeroelasticDesignVariables = cls.wing.static_adjoint(
            case=cls.sol,
            objective=_objective,
            ad_mode=cls.ad_mode,
            grads_to_compute=cls.grads_to_compute,
        )[0]
        cls.objective_val: Array = _objective(
            cls.sol.get_full_states(),
            cls.wing.get_design_variables(
                case=cls.sol, grads_to_compute=cls.grads_to_compute
            ),
        )

    @classmethod
    def test_u_inf_gradient(cls):
        """Adjoint gradient w.r.t. u_inf verified by finite differences."""

        eps = 1e-2
        new_wing, new_sol = _solve(u_inf=u_inf_base.at[0].add(eps), k_cs=k_cs_base)

        new_obj = _objective(
            new_sol.get_full_states(),
            new_wing.get_design_variables(
                case=new_sol, grads_to_compute=cls.grads_to_compute
            ),
        )

        fd_grad = (new_obj - cls.objective_val) / eps

        assert cls.grad.aero.flowfield is not None
        adj_grad = cls.grad.aero.flowfield["u_inf"][0]

        assert jnp.allclose(fd_grad, adj_grad, rtol=1e-2), (
            f"Gradient mismatch with respect to u_inf[0], Adjoint={adj_grad}, FD={fd_grad}"
        )

    @classmethod
    def test_k_cs_gradient(cls):
        """Adjoint gradient w.r.t. k_cs verified by finite differences."""
        eps = 1e-1

        new_wing, new_sol = _solve(u_inf=u_inf_base, k_cs=k_cs_base.at[3, 3].add(eps))

        new_obj = _objective(
            new_sol.get_full_states(),
            new_wing.get_design_variables(
                case=new_sol, grads_to_compute=AeroelasticGradsToCompute()
            ),
        )

        fd_grad = (new_obj - cls.objective_val) / eps
        if cls.grad.structure.k_cs is None:
            raise ValueError("k_cs is None")

        adj_grad = cls.grad.structure.k_cs[:, 3, 3].sum()

        assert jnp.allclose(fd_grad, adj_grad, rtol=6e-3), (
            f"Gradient mismatch with respect to k_cs[:, 3, 3], Adjoint={adj_grad}, FD={fd_grad}"
        )


class TestReverseStaticAeroelasticAdjoint(TestForwardStaticAeroelasticAdjoint):
    ad_mode = "reverse"
