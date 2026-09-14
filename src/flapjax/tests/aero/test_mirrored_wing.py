from jax import numpy as jnp

from flapjax.aero.data_structures import GridDiscretisation
from flapjax.aero.flowfields import ConstantFlowField
from flapjax.aero.utils import make_rectangular_grid
from flapjax.aero.uvlm import UVLM


class TestMirroredWing:
    @staticmethod
    def test_mirror():
        r"""
        Test that the solution is rotation invariant for a no-wake case of a square wing, subject to flows in both
        the X and Y directions with positive angle of attack.
        """

        m = 5
        n = 10
        m_star = 20
        c_ref = 1.0
        b_ref = 5.0
        u_inf = jnp.array((10.0, 0.0, 1.0))

        dt = c_ref / (jnp.linalg.norm(u_inf) * m)
        flowfield = ConstantFlowField(u_inf, 1.225, True)

        # mirrored case
        disc = GridDiscretisation(m=m, n=n, m_star=m_star)

        hg = jnp.zeros((n + 1, 4, 4))
        x_grid = make_rectangular_grid(m=m, n=n, ea=0.0, chord=c_ref)
        beam_coords = jnp.zeros((n + 1, 3))
        beam_coords = beam_coords.at[:, 1].set(jnp.linspace(0.0, b_ref, n + 1))
        hg = hg.at[:, :3, :3].set(jnp.eye(3)[None, :, :])
        hg = hg.at[:, :3, 3].set(beam_coords)

        uvlm_mirror = UVLM(
            grid_shapes=[disc],
            dof_mapping=jnp.arange(n + 1),
            variable_wake_disc=False,
            mirror_point=jnp.zeros(3),
            mirror_normal=jnp.array((0.0, 1.0, 0.0)),
        )

        uvlm_mirror.set_design_variables(
            dt=dt, flowfield=flowfield, zeta_b0=x_grid, hg0=hg
        )
        sol_mirror = uvlm_mirror.static_solve()

        # full wing
        disc = GridDiscretisation(m=m, n=2 * n, m_star=m_star)

        hg = jnp.zeros((2 * n + 1, 4, 4))
        x_grid = make_rectangular_grid(m=m, n=2 * n, ea=0.0, chord=c_ref)
        beam_coords = jnp.zeros((2 * n + 1, 3))
        beam_coords = beam_coords.at[:, 1].set(jnp.linspace(-b_ref, b_ref, 2 * n + 1))
        hg = hg.at[:, :3, :3].set(jnp.eye(3)[None, :, :])
        hg = hg.at[:, :3, 3].set(beam_coords)

        uvlm_full = UVLM(
            grid_shapes=[disc],
            dof_mapping=jnp.arange(2 * n + 1),
            variable_wake_disc=False,
        )

        uvlm_full.set_design_variables(
            dt=dt, flowfield=flowfield, zeta_b0=x_grid, hg0=hg
        )
        sol_full = uvlm_full.static_solve()

        # compare
        assert jnp.allclose(sol_mirror.gamma_b[0], sol_full.gamma_b[0][:, n:]), (
            "Mirrored bound solution does not match full solution"
        )

        assert jnp.allclose(sol_mirror.gamma_w[0], sol_full.gamma_w[0][:, n:]), (
            "Mirrored wake solution does not match full solution"
        )
