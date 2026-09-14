from jax import numpy as jnp

from flapjax.aero.data_structures import GridDiscretisation
from flapjax.aero.flowfields import ConstantFlowField
from flapjax.aero.utils import make_rectangular_grid
from flapjax.aero.uvlm import UVLM


class TestRotInvariance:
    @staticmethod
    def test_rot_invariance_no_wake():
        r"""
        Test that the solution is rotation invariant for a no-wake case of a square wing, subject to flows in both
        the X and Y directions with positive angle of attack.
        """

        mn = 5
        width = 1.0
        disc = GridDiscretisation(mn, mn, 0)

        hg = jnp.zeros((mn + 1, 4, 4))
        x_grid = make_rectangular_grid(mn, mn, width, 0.0)
        beam_coords = jnp.zeros((mn + 1, 3))
        beam_coords = beam_coords.at[:, 1].set(jnp.linspace(0.0, width, mn + 1))
        hg = hg.at[:, :3, :3].set(jnp.eye(3)[None, :, :])
        hg = hg.at[:, :3, 3].set(beam_coords)

        uvlm = UVLM(
            grid_shapes=[disc],
            dof_mapping=jnp.arange(0, mn + 1),
            variable_wake_disc=False,
        )

        cases = []
        for _i_u_inf, u_inf in enumerate(
            [jnp.array((0.0, 10.0, 3.0)), jnp.array((10.0, 0.0, 3.0))]
        ):
            flowfield = ConstantFlowField(u_inf, 1.225, True)
            uvlm.set_design_variables(
                dt=1.0, flowfield=flowfield, zeta_b0=x_grid, hg0=hg
            )
            cases.append(uvlm.static_solve())

        if not jnp.allclose(cases[0].gamma_b[0], cases[1].gamma_b[0]):
            raise ValueError(
                "Gamma distribution is not equal for both flow directions in no-wake case."
            )

        if not jnp.allclose(
            f_tot := jnp.sum(cases[0].f_steady[0]),
            0.0,
            atol=1e-5,
            rtol=1e-4,
        ):
            raise ValueError(f"Total force in flow is not zero: {f_tot}")

        if not jnp.allclose(
            cases[0].f_steady[0],
            jnp.transpose(cases[1].f_steady[0], (1, 0, 2))[..., (1, 0, 2)],
            atol=1e-5,
            rtol=1e-4,
        ):
            raise ValueError(
                "Steady force distribution is not equal in both flow directions in no-wake case."
            )
