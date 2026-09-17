import pytest
from jax import numpy as jnp

from flapjax.structure import BeamStructure
from flapjax.structure.constraints import MultibodyHinge


def _cantilever_beam(
    n_nodes: int = 6, length: float = 1.0, k_scale: float = 1.0, **kwargs
):
    conn = jnp.stack((jnp.arange(n_nodes - 1), jnp.arange(1, n_nodes)), axis=1).astype(
        int
    )
    beam = BeamStructure(
        num_nodes=n_nodes,
        connectivity=conn,
        y_vector=jnp.array((0.0, 0.0, 1.0)),
        **kwargs,
    )
    coords = jnp.stack(
        (
            jnp.linspace(0.0, length, n_nodes),
            jnp.zeros(n_nodes),
            jnp.zeros(n_nodes),
        ),
        axis=1,
    )
    beam.set_design_variables(
        coords=coords,
        k_cs=jnp.diag(jnp.array((1e6, 1e6, 1e6, 1e2, 1e2, 1e2)) * k_scale),
        m_cs=jnp.diag(jnp.array((1.0, 1.0, 1.0, 0.1, 0.1, 0.1))),
    )
    return beam


class TestAutoNodeCreation:
    r"""
    Test the automatic node creation for a beam structure with constraints.
    """

    def test_node_count_and_index(self):
        r"""
        Passing ``node_j=None`` should increment the beam's node count and assign the new index to the constraint.
        """
        hinge = MultibodyHinge(node_i=2, axis=jnp.array([0.0, 1.0, 0.0]))
        beam = BeamStructure(
            num_nodes=5,
            connectivity=jnp.array([[0, 1], [1, 2], [2, 3], [3, 4]]),
            y_vector=jnp.array((0.0, 0.0, 1.0)),
            constraints={"hinge": hinge},
        )
        assert beam.n_nodes == 6  # check node count is increased to 6
        assert hinge.node_j == 5  # check new node is added to end

    def test_coords_auto_extended(self):
        r"""
        Check that the reference node coordinates are correctly copied for added nodes.
        """
        hinge = MultibodyHinge(node_i=2, axis=jnp.array([0.0, 1.0, 0.0]))
        beam = BeamStructure(
            num_nodes=5,
            connectivity=jnp.array([[0, 1], [1, 2], [2, 3], [3, 4]]),
            y_vector=jnp.array((0.0, 0.0, 1.0)),
            constraints={"hinge": hinge},
        )
        coords = jnp.stack(
            (jnp.linspace(0, 1, 5), jnp.zeros(5), jnp.zeros(5)),
            axis=1,
        )
        beam.set_design_variables(
            coords=coords,
            k_cs=jnp.diag(jnp.array((1e6, 1e6, 1e6, 1e2, 1e2, 1e2))),
            m_cs=jnp.diag(jnp.array((1.0, 1.0, 1.0, 0.1, 0.1, 0.1))),
        )
        assert jnp.allclose(beam.hg0[5, :3, 3], beam.hg0[2, :3, 3])


L = 1.0
EI = 1e4
N_ELEM_PER_SEG = 4
N_NODES_PER_SEG = N_ELEM_PER_SEG + 1
N_NODES = 2 * N_NODES_PER_SEG

K_CS = jnp.diag(jnp.array([1e8, 1e8, 1e8, 1e6, EI, EI]))
M_CS = jnp.diag(jnp.array([1.0, 1.0, 1.0, 0.1, 0.1, 0.1]))

NODE_I = N_NODES_PER_SEG - 1
NODE_J = N_NODES_PER_SEG
PRESCRIBED_DOFS = tuple(range(6)) + tuple(range((N_NODES - 1) * 6, N_NODES * 6))


def _hinged_beam():
    left_conn = jnp.stack(
        (jnp.arange(N_NODES_PER_SEG - 1), jnp.arange(1, N_NODES_PER_SEG)),
        axis=1,
    )
    right_conn = jnp.stack(
        (
            jnp.arange(N_NODES_PER_SEG, 2 * N_NODES_PER_SEG - 1),
            jnp.arange(N_NODES_PER_SEG + 1, 2 * N_NODES_PER_SEG),
        ),
        axis=1,
    )
    conn = jnp.concatenate((left_conn, right_conn), axis=0).astype(int)

    hinge = MultibodyHinge(
        node_i=NODE_I, node_j=NODE_J, axis=jnp.array([0.0, 1.0, 0.0])
    )
    beam = BeamStructure(
        num_nodes=N_NODES,
        connectivity=conn,
        y_vector=jnp.array((0.0, 0.0, 1.0)),
        constraints={"hinge": hinge},
    )

    x_left = jnp.linspace(0.0, L / 2, N_NODES_PER_SEG)
    x_right = jnp.linspace(L / 2, L, N_NODES_PER_SEG)
    x = jnp.concatenate((x_left, x_right))
    coords = jnp.stack((x, jnp.zeros(N_NODES), jnp.zeros(N_NODES)), axis=1)
    beam.set_design_variables(coords=coords, k_cs=K_CS, m_cs=M_CS)
    return beam, hinge


FORCE_PARAMS = [
    pytest.param(500.0, id="F_500"),
    pytest.param(1000.0, id="F_1000"),
    pytest.param(2000.0, id="F_2000"),
]


@pytest.mark.parametrize("force", FORCE_PARAMS)
class TestHingedBeamEquilibrium:
    r"""
    Clamped-clamped beam with a Hinge at the midpoint under a dead force at the hinge. Analytical solution from
    Euler-Bernoulli beam theory.
    """

    def test_hinge_angle(self, force):
        beam, _ = _hinged_beam()
        f_ext_dead = jnp.zeros((N_NODES, 6)).at[NODE_I, 2].set(-force)

        result = beam.static_solve(
            prescribed_dofs=PRESCRIBED_DOFS,
            f_ext_dead=f_ext_dead,
            load_steps=10,
        )

        rmat_rel = result.hg[NODE_I, :3, :3].T @ result.hg[NODE_J, :3, :3]
        theta_actual = jnp.abs(jnp.arctan2(rmat_rel[2, 0], rmat_rel[0, 0]))

        theta_expected = force * L**2 / (8 * EI)
        assert jnp.isclose(theta_actual, theta_expected, rtol=0.05), (
            f"Hinge angle {float(theta_actual):.6f} rad "
            f"!= expected {float(theta_expected):.6f} rad"
        )

    def test_midpoint_deflection(self, force):
        beam, _ = _hinged_beam()
        f_ext_dead = jnp.zeros((N_NODES, 6)).at[NODE_I, 2].set(-force)

        result = beam.static_solve(
            prescribed_dofs=PRESCRIBED_DOFS,
            f_ext_dead=f_ext_dead,
            load_steps=10,
        )

        delta_actual = -result.hg[NODE_I, 2, 3]
        delta_expected = force * L**3 / (48 * EI)
        assert jnp.isclose(delta_actual, delta_expected, rtol=0.05), (
            f"Midpoint deflection {float(delta_actual):.6f} m "
            f"!= expected {float(delta_expected):.6f} m"
        )

    def test_constraint_satisfaction(self, force):
        beam, hinge = _hinged_beam()
        f_ext_dead = jnp.zeros((N_NODES, 6)).at[NODE_I, 2].set(-force)

        result = beam.static_solve(
            prescribed_dofs=PRESCRIBED_DOFS,
            f_ext_dead=f_ext_dead,
            load_steps=10,
        )

        violation = hinge.violation(result.hg[NODE_I], result.hg[NODE_J])
        assert jnp.allclose(violation, 0.0, atol=1e-8), (
            f"Constraint violation: {violation}"
        )
