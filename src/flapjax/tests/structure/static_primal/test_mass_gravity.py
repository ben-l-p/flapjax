import pytest
from jax import numpy as jnp

from flapjax.algebra.se3 import exp_se3
from flapjax.structure import BeamStructure

LENGTH = jnp.array(2.5)
M_BAR = 10.0
K_CS = jnp.eye(6) * 1e6
M_CS = jnp.zeros((6, 6)).at[:3, :3].set(M_BAR * jnp.eye(3))

GRAVITY_PARAMS = [
    pytest.param(0, jnp.array([[0.0, 1.0, 0.0]]), jnp.array([0.0, 0.0, -9.81]), id="x_beam-z_gravity"),
    pytest.param(0, jnp.array([[0.0, 1.0, 0.0]]), jnp.array([-9.81, 0.0, 0.0]), id="x_beam-x_gravity"),
    pytest.param(0, jnp.array([[0.0, 1.0, 0.0]]), jnp.array([0.0, -9.81, 0.0]), id="x_beam-y_gravity"),
    pytest.param(1, jnp.array([[0.0, 0.0, 1.0]]), jnp.array([0.0, 0.0, -9.81]), id="y_beam-z_gravity"),
    pytest.param(1, jnp.array([[0.0, 0.0, 1.0]]), jnp.array([-9.81, 0.0, 0.0]), id="y_beam-x_gravity"),
    pytest.param(1, jnp.array([[0.0, 0.0, 1.0]]), jnp.array([0.0, -9.81, 0.0]), id="y_beam-y_gravity"),
    pytest.param(2, jnp.array([[1.0, 0.0, 0.0]]), jnp.array([0.0, 0.0, -9.81]), id="z_beam-z_gravity"),
    pytest.param(2, jnp.array([[1.0, 0.0, 0.0]]), jnp.array([-9.81, 0.0, 0.0]), id="z_beam-x_gravity"),
    pytest.param(2, jnp.array([[1.0, 0.0, 0.0]]), jnp.array([0.0, -9.81, 0.0]), id="z_beam-y_gravity"),
]


def _beam(direction_index, y_vector, g_vec):
    coords = jnp.zeros((2, 3)).at[1, direction_index].set(LENGTH)
    struct = BeamStructure(
        num_nodes=2,
        connectivity=jnp.array([[0, 1]]),
        y_vector=y_vector,
        gravity=g_vec,
    )
    struct.set_design_variables(coords, K_CS, M_CS)
    return struct, coords


@pytest.mark.parametrize("direction_index, y_vector, g_vec", GRAVITY_PARAMS)
class TestTwoNodeGravity:
    r"""
    Test the strains and forces for a two-node beam element with prescribed displacements
    """

    def test_total_mass(self, direction_index, y_vector, g_vec):
        r"""
        Ensure total mass of beam is correct
        """
        struct, coords = _beam(direction_index, y_vector, g_vec)

        m_t = struct.make_m_t(struct.d0)

        expected_mass = M_BAR * LENGTH

        # consider all three directions, and divide by three
        matrix_mass = jnp.sum(
            m_t
            @ jnp.array((1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0))
            / 3.0
        )

        assert jnp.allclose(matrix_mass, expected_mass), (
            f"Total mass from mass matrix {matrix_mass} does not match expected {expected_mass}"
        )

    def test_gravity_forces(self, direction_index, y_vector, g_vec):
        r"""
        Ensure weight of beam is correct
        """
        struct, coords = _beam(direction_index, y_vector, g_vec)

        f_g = struct.assemble_vector_from_entries(
            struct._make_f_grav(
                struct.make_m_t(struct.d0), struct.hg0[:, :3, :3]
            )
        )

        expected_weight = M_BAR * LENGTH * g_vec
        matrix_weight = (
            struct.hg0[0, :3, :3] @ f_g[:3] + struct.hg0[1, :3, :3] @ f_g[6:9]
        )

        assert jnp.allclose(matrix_weight, expected_weight), (
            f"""Weight from gravity forces {matrix_weight} does not match expected {expected_weight}"""
        )

    def test_gravity_forces_deformed(self, direction_index, y_vector, g_vec):
        r"""
        Ensure weight of beam is correct
        """
        struct, coords = _beam(direction_index, y_vector, g_vec)

        # make beam curved around local y
        d = jnp.array((LENGTH, 0.0, 0.0, 0.0, jnp.pi / 2.0, 0.0))
        ha0 = jnp.eye(4).at[:3, :3].set(struct.o0[0, ...])
        hb = ha0 @ exp_se3(d) @ ha0.T

        hg = jnp.stack((jnp.eye(4), hb), axis=0)

        f_g = struct.assemble_vector_from_entries(
            struct._make_f_grav(struct.make_m_t(d[None, :]), hg[:, :3, :3])
        )

        expected_weight = M_BAR * LENGTH * g_vec
        matrix_weight = hg[0, :3, :3] @ f_g[:3] + hg[1, :3, :3] @ f_g[6:9]

        assert jnp.allclose(matrix_weight, expected_weight), (
            f"""Weight from gravity forces {matrix_weight} does not match expected {expected_weight}"""
        )
