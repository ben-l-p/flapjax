from copy import deepcopy

from jax import numpy as jnp

from flapjax.aero.flowfields import OneMinusCosineFlowField
from flapjax.models.simple_hale.simple_hale import generate_simple_hale


class TestHaleRelativeMotion:
    @staticmethod
    def test_dynamic_relative_motion():
        """Dynamic gust response from a trimmed HALE should be identical for relative_motion True and False."""
        u_inf = jnp.array((10.0, 0.0, 0.0))
        u_inf_mag = 10.0
        rho = 1.225
        gust_length = 2.5
        n_tstep = 30

        flowfield = OneMinusCosineFlowField(
            u_inf=u_inf,
            rho=rho,
            relative_motion=True,
            gust_length=gust_length,
            gust_amplitude=0.2 * u_inf_mag,
            gust_x0=jnp.array((-2.0 * gust_length, 0.0, 0.0)),
        )

        hale_false = generate_simple_hale(flowfield=deepcopy(flowfield))
        static_false, _ = hale_false.trim(
            prescribed_dofs=jnp.arange(6),
            zero_force_dofs=(0, 2, 4),
            trim_cs="elevator",
            thrust_nodes="thrust",
            trim_orientation="y",
            horseshoe=False,
        )
        dynamic_init_false = hale_false.initialise_dynamic(
            static_case=static_false, prescribed_dofs=()
        )
        dynamic_false = hale_false.dynamic_solve(
            init_case=dynamic_init_false, prescribed_dofs=(), n_tstep=n_tstep
        )

        hale_true = generate_simple_hale(flowfield=deepcopy(flowfield))
        static_true, _ = hale_true.trim(
            prescribed_dofs=jnp.arange(6),
            zero_force_dofs=(0, 2, 4),
            trim_cs="elevator",
            thrust_nodes="thrust",
            trim_orientation="y",
            horseshoe=False,
        )
        dynamic_init_true = hale_true.initialise_dynamic(
            static_case=static_true, prescribed_dofs=()
        )
        hale_true.aero.flowfield.relative_motion = True
        dynamic_init_true.structure.v = dynamic_init_true.structure.v.at[:, :3].set(0.0)

        dynamic_true = hale_true.dynamic_solve(
            init_case=dynamic_init_true, prescribed_dofs=(), n_tstep=n_tstep
        )

        # compare strains
        eps_err = jnp.max(
            jnp.abs(dynamic_true.structure.eps - dynamic_false.structure.eps)
        )
        assert eps_err < 5e-4, f"Max strain difference: {float(eps_err)}"

        # compare bound circulation
        for i, (gamma_true, gamma_false) in enumerate(
            zip(dynamic_true.aero.gamma_b, dynamic_false.aero.gamma_b)
        ):
            gamma_err = jnp.max(jnp.abs(gamma_true - gamma_false))
            assert gamma_err < 0.05, (
                f"Max gamma_b difference on surface {i}: {float(gamma_err)}"
            )
