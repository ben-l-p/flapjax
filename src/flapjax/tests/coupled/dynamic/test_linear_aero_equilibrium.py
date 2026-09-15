import pytest
from jax import numpy as jnp

from flapjax.aero.linear.linear_uvlm import LinearWakeType
from flapjax.coupled import NonlinearBeamLinearAero
from flapjax.models.cantilever_wing.cantilever_wing import generate_cantilever_wing
from flapjax.utils.data_structures import ConvergenceSettings


@pytest.mark.parametrize("wake_type", ["frozen", "prescribed", "free"])
class TestLinearAeroEquilibrium:
    def test_linear_aero_equilibrium(
        self, wake_type: LinearWakeType, plot: bool = False
    ):
        r"""
        Cantilever wing initialised at its static aeroelastic equilibrium and stepped forward in time with the nonlinear
        beam and linear UVLM. This should remain at the equilibrium for all wake types.
        """

        m = 8
        n = 20
        m_star = 20
        c_ref = 1.0
        u_inf = jnp.array((10.0, 0.0, 1.5))
        k_cs = jnp.diag(jnp.array((1e6, 1e6, 1e6, 1e3, 1e3, 1e3)))

        wing = generate_cantilever_wing(
            m=m,
            m_star=m_star,
            c_ref=c_ref,
            k_cs=k_cs,
            ea=0.25,
            n_nodes=n + 1,
            u_inf=u_inf,
        )
        wing.structure.spectral_radius = 0.8

        conv_settings = ConvergenceSettings(
            max_n_iter=25,
            abs_disp_tol=1e-9,
            rel_disp_tol=1e-7,
            abs_force_tol=1e-9,
            rel_force_tol=1e-7,
        )
        wing.structure.struct_convergence_settings = conv_settings

        n_tstep = 100

        static_sol = wing.static_solve(prescribed_dofs=jnp.arange(6))

        linear_aero = wing.aero.linearise(
            reference=static_sol.aero,
            wake_type=wake_type,
            bound_upwash=False,
            wake_upwash=False,
            unsteady_force=True,
        )
        linear_wing = NonlinearBeamLinearAero(
            structure=wing.structure,
            aero=linear_aero,
            fsi_convergence_settings=conv_settings,
        )

        dynamic_sol = linear_wing.dynamic_solve(
            init_case=static_sol,
            prescribed_dofs=jnp.arange(6),
            n_tstep=n_tstep,
        )

        if plot:
            dynamic_sol.plot(directory="./test_outputs/linear_aero_equilibrium")

        gamma_b_err = (
            dynamic_sol.aero.gamma_b[0] - dynamic_sol.aero.gamma_b[0][[0], ...]
        )
        assert jnp.allclose(jnp.abs(gamma_b_err).max(), 0.0, atol=5e-4), (
            "Bound circulation varies from equilibrium"
        )

        varphi_diff = (
            dynamic_sol.structure.varphi - dynamic_sol.structure.varphi[[0], ...]
        )
        assert jnp.allclose(jnp.abs(varphi_diff).max(), 0.0, atol=5e-4), (
            "Structural deformation varies from equilibrium"
        )
