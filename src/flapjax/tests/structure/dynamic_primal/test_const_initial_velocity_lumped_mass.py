import pytest
from jax import numpy as jnp
from jax import vmap
from jax.scipy.linalg import block_diag

from flapjax.algebra.se3 import log_se3
from flapjax.structure import BeamStructure


@pytest.mark.parametrize("v_direction_index", [
    pytest.param(0, id="lin_x"),
    pytest.param(1, id="lin_y"),
    pytest.param(2, id="lin_z"),
    pytest.param(3, id="rot_x"),
    pytest.param(4, id="rot_y"),
    pytest.param(5, id="rot_z"),
])
class TestConstVelocityLumpedMass:
    def test_const_velocity_point_mass(self, v_direction_index):
        v = 10.0

        coords = jnp.zeros((1, 3))
        conn = jnp.zeros((0, 2), dtype=int)

        m = 0.1
        j = 10.0
        m_lump = block_diag(jnp.eye(3) * m, jnp.eye(3) * j)

        n_tstep = 50
        dt = 0.001

        struct = BeamStructure(
            num_nodes=1,
            connectivity=conn,
            y_vector=jnp.zeros((0, 3)),
            m_lumped_index=jnp.zeros((1,), dtype=int),
            spectral_radius=1.0,
        )
        struct.set_design_variables(
            coords, jnp.zeros((0, 6, 6)), None, m_lump[None, ...]
        )

        v_init = jnp.zeros((1, 6)).at[0, v_direction_index].set(v)

        init_cond = struct.reference_configuration(prescribed_dofs=()).to_dynamic()
        init_cond.v = v_init

        output = struct.dynamic_solve(
            init_state=init_cond,
            n_tstep=n_tstep,
            dt=dt,
            f_ext_follower=None,
            f_ext_dead=None,
            f_ext_aero=None,
        )

        x_expected = v * jnp.arange(n_tstep) * dt

        disp_measured = output.x[:, 0, :]
        theta_measured = vmap(log_se3)(output.hg[:, 0, :, :])[:, 3:]
        x_measured = jnp.concatenate((disp_measured, theta_measured), axis=-1)[
            :, v_direction_index
        ]

        assert jnp.allclose(x_measured, x_expected), (
            "Displacements/angles do not match expected values."
        )

        assert jnp.allclose(output.v[:, 0, v_direction_index], v), (
            "Velocities do not remain constant as expected."
        )

        assert jnp.allclose(output.v_dot, 0.0), (
            "Accelerations are not zero as expected."
        )
