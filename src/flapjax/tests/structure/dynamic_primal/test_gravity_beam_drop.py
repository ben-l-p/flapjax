from typing import cast

import pytest
from jax import Array
from jax import numpy as jnp
from jax.scipy.linalg import block_diag

from flapjax.structure import BeamStructure

PARAMS = [
    pytest.param(g, b, y, id=f"g_{['x', 'y', 'z'][g]}-beam_{['x', 'y', 'z'][b]}")
    for b, y in [
        (0, jnp.array([[0.0, 1.0, 0.0]])),
        (1, jnp.array([[0.0, 0.0, 1.0]])),
        (2, jnp.array([[1.0, 0.0, 0.0]])),
    ]
    for g in range(3)
]


@pytest.mark.parametrize("g_direction_index, beam_direction_index, y_vect", PARAMS)
class TestGravityBeamDrop:
    def test_beam_drop(self, g_direction_index, beam_direction_index, y_vect):
        length = 3.14
        g = -9.81

        g_vec = jnp.zeros(3).at[g_direction_index].set(g)

        coords = jnp.zeros((2, 3)).at[1, beam_direction_index].set(length)
        conn = jnp.array([[0, 1]])

        k_cs = jnp.diag(jnp.full(6, 1e3))
        m_bar = 5.0 * jnp.eye(3)
        j_bar = 0.1 * jnp.eye(3)
        m_cs = block_diag(m_bar, j_bar)

        n_tstep = 1000
        dt = 0.001

        struct = BeamStructure(
            num_nodes=2,
            connectivity=conn,
            y_vector=y_vect,
            gravity=g_vec,
            spectral_radius=1.0,
        )
        struct.set_design_variables(coords, k_cs, m_cs)

        init_cond = struct.reference_configuration(prescribed_dofs=()).to_dynamic()
        init_cond.v_dot = init_cond.v_dot.at[:, :3].set(g_vec[None, :])

        output = struct.dynamic_solve(
            init_state=init_cond,
            n_tstep=n_tstep,
            dt=dt,
            f_ext_follower=None,
            f_ext_dead=None,
            f_ext_aero=None,
        )

        expected_nodal_fg = 0.5 * m_bar[0, 0] * length * g
        expected_v = jnp.arange(1, n_tstep) * dt * g
        expected_x0 = 0.5 * g * (jnp.arange(1, n_tstep) * dt) ** 2
        expected_x1 = expected_x0 + coords[1, g_direction_index]

        assert jnp.allclose(
            expected_x0,
            output.x[1:, 0, g_direction_index],
        ), "Node 0 positions do not match expected values"
        assert jnp.allclose(
            expected_x1,
            output.x[1:, 1, g_direction_index],
        ), "Node 1 positions do not match expected values"
        assert jnp.allclose(expected_v, output.v[1:, 0, g_direction_index]), (
            "Node 0 velocities do not match expected values"
        )
        assert jnp.allclose(expected_v, output.v[1:, 1, g_direction_index]), (
            "Node 1 velocities do not match expected values"
        )
        assert jnp.allclose(g, output.v_dot[:, :, g_direction_index]), (
            "Accelerations do not match gravity"
        )
        assert jnp.allclose(
            cast(Array, output.f_grav)[1:, :, g_direction_index], expected_nodal_fg
        ), "Gravitational forces do not match expected values"
