import pytest
from jax import numpy as jnp
from jax.scipy.linalg import block_diag

from flapjax.structure import BeamStructure

PARAMS = [
    pytest.param(v, b, y, id=f"v_{['x','y','z'][v]}-beam_{['x','y','z'][b]}")
    for b, y in [(0, jnp.array([[0.0, 1.0, 0.0]])),
                 (1, jnp.array([[0.0, 0.0, 1.0]])),
                 (2, jnp.array([[1.0, 0.0, 0.0]]))]
    for v in range(3)
]


@pytest.mark.parametrize("v_direction_index, beam_direction_index, y_vect", PARAMS)
class TestConstLinearVelocity:
    def test_const_velocity_beam(self, v_direction_index, beam_direction_index, y_vect):
        v_mag: float = 50.0
        length = 3.14

        coords = jnp.zeros((2, 3)).at[1, beam_direction_index].set(length)
        conn = jnp.array([[0, 1]])

        k_cs = jnp.diag(jnp.full(6, 1e3))
        m_bar = 5.0 * jnp.eye(3)
        j_bar = 0.1 * jnp.eye(3)
        m_cs = block_diag(m_bar, j_bar)

        n_tstep = 500
        dt = 0.01

        struct = BeamStructure(
            2,
            conn,
            y_vect,
            None,
            spectral_radius=1.0,
        )
        struct.set_design_variables(coords, k_cs, m_cs)

        v_init = jnp.zeros((2, 6)).at[:, v_direction_index].set(v_mag)

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
        x_t = output.x[:, 0, v_direction_index]  # [n_tstep]

        expected_x_t = jnp.arange(n_tstep) * dt * v_mag  # [n_tstep]

        assert jnp.allclose(expected_x_t, x_t), (
            "Beam with constant initial velocity did not maintain constant velocity."
        )
