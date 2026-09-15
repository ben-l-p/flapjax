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


class TestDynamicEquilibriumAdjoint:
    @staticmethod
    def _run(
        matrix_free: bool,
        plot=False,
    ):
        m = 2
        n = 4
        m_star = 3
        c_ref = 0.2
        b_ref = 1.0
        u_inf = jnp.array((10.0, 0.0, 0.1))
        k_cs = jnp.diag(jnp.array((1e2, 1e2, 1.0, 1.0, 1.0, 1.0)))

        wing = generate_cantilever_wing(
            m=m,
            m_star=m_star,
            c_ref=c_ref,
            b_ref=b_ref,
            k_cs=k_cs,
            ea=0.25,
            n_nodes=n + 1,
            u_inf=u_inf,
        )

        n_tstep = 100

        def static_objective(states: AeroelasticFullStates, *_, **__) -> Array:
            return states.structure.f_elem[0, 3]

        def dynamic_objective(
            states: AeroelasticFullStates,
            dv: AeroelasticDesignVariables,
            i_ts: int | Array | None,
        ) -> Array:
            return static_objective(states, dv, i_ts=i_ts) / n_tstep

        # set tolerance to zero, rather than none, to prevent error messages
        wing.structure.struct_convergence_settings = ConvergenceSettings(
            max_n_iter=100,
            rel_disp_tol=0.0,
            abs_disp_tol=0.0,
            rel_force_tol=0.0,
            abs_force_tol=0.0,
        )
        wing.fsi_convergence_settings = ConvergenceSettings(
            max_n_iter=40,
            rel_disp_tol=0.0,
            abs_disp_tol=0.0,
            rel_force_tol=0.0,
            abs_force_tol=0.0,
        )
        wing.aero.include_unsteady_force = False
        wing.aero.gamma_dot_relaxation = 0.7
        wing.structure.spectral_radius = 1.0

        static_sol = wing.static_solve(prescribed_dofs=jnp.arange(6))

        dynamic_sol = wing.dynamic_solve(
            init_case=static_sol,
            prescribed_dofs=jnp.arange(6),
            n_tstep=n_tstep,
        )

        if plot:
            dynamic_sol.plot(directory="./test_outputs/dynamic_coupled_adjoint")

        grads_to_compute: AeroelasticGradsToCompute = AeroelasticGradsToCompute(
            structure=StructureGradsToCompute(m_cs=True, k_cs=True),
            aero=AeroGradsToCompute(flowfield=True),
        )

        static_grad, static_adj = wing.static_adjoint(
            case=static_sol,
            objective=static_objective,
            grads_to_compute=grads_to_compute,
        )

        dynamic_grad, *_ = wing.dynamic_adjoint(
            case=dynamic_sol,
            objective=dynamic_objective,
            matrix_free=matrix_free,
            p_varphi_p_x=-static_adj,
            approx_grads=False,
            grads_to_compute=grads_to_compute,
            gmres_precond=matrix_free,
        )

        assert (
            dynamic_grad.aero.flowfield is not None
            and static_grad.aero.flowfield is not None
            and dynamic_grad.aero.flowfield is not None
            and dynamic_grad.structure.m_cs is not None
            and dynamic_grad.structure.k_cs is not None
            and static_grad.structure.k_cs is not None
        )

        assert jnp.allclose(
            dynamic_grad.aero.flowfield["u_inf"], static_grad.aero.flowfield["u_inf"]
        ), "Mismatch in u_inf gradient"

        assert jnp.allclose(
            dynamic_grad.aero.flowfield["rho"], static_grad.aero.flowfield["rho"]
        ), "Mismatch in rho gradient"

        assert jnp.allclose(dynamic_grad.structure.m_cs, 0.0, atol=1e-6), (
            "Nonzero mass gradient"
        )

        assert jnp.allclose(dynamic_grad.structure.k_cs, static_grad.structure.k_cs), (
            "Mismatch in stiffness gradient"
        )

    def test_dynamic_equilibrium_adjoint_dense(self):
        self._run(matrix_free=False)

    def test_dynamic_equilibrium_adjoint_matrix_free_preconditioned(self):
        self._run(matrix_free=True)
