import jax
from jax import Array
from jax import numpy as jnp
from jax.scipy.linalg import block_diag

from flapjax.structure import BeamStructure
from flapjax.structure.constraints import MultibodyHinge


def _hinged_beam(hinge_axis: Array, gravity: Array):
    r"""
    Two stiff beam segments connected with a hinge.
    """
    n_nodes = 6
    conn = jnp.array([[0, 1], [1, 2], [3, 4], [4, 5]])

    hinge = MultibodyHinge(node_i=2, node_j=3, axis=hinge_axis)

    beam = BeamStructure(
        num_nodes=n_nodes,
        connectivity=conn,
        y_vector=jnp.array((0.0, 0.0, 1.0)),
        constraints={"hinge": hinge},
        gravity=gravity,
        spectral_radius=1.0,
    )
    x = jnp.array([0.0, 0.25, 0.5, 0.5, 0.75, 1.0])
    coords = jnp.stack((x, jnp.zeros(n_nodes), jnp.zeros(n_nodes)), axis=1)
    k_cs = jnp.diag(jnp.array((1e6, 1e6, 1e6, 1e4, 1e4, 1e4)))
    m_cs = block_diag(5.0 * jnp.eye(3), 0.1 * jnp.eye(3))
    beam.set_design_variables(coords=coords, k_cs=k_cs, m_cs=m_cs)
    return beam, hinge


class TestMultibodyHingeDynamic:
    def test_gravity_drop_constraint_satisfaction(self):
        r"""
        Drop a hinged beam under gravity with the root clamped. The
        hinge constraint should remain satisfied at every time step.
        """
        g = jnp.array([0.0, 0.0, -9.81])
        beam, hinge = _hinged_beam(
            hinge_axis=jnp.array([0.0, 1.0, 0.0]),
            gravity=g,
        )
        n_tstep = 1000
        dt = 0.001

        init = beam.reference_configuration(
            prescribed_dofs=tuple(range(6))
        ).to_dynamic()
        init.v_dot = init.v_dot.at[:, :3].set(g[None, :])

        out = beam.dynamic_solve(
            init_state=init,
            n_tstep=n_tstep,
            dt=dt,
        )

        # noinspection argument-list
        viol = jax.vmap(hinge.violation)(out.hg[1:, 2], out.hg[1:, 3])
        viol_norms = jnp.linalg.norm(viol, axis=1)
        worst = jnp.argmax(viol_norms)
        assert jnp.all(viol_norms < 1e-6), (
            f"Constraint violation at step {worst + 1}: {float(viol_norms[worst]):.2e}"
        )

        # ensure both hinge nodes share translational coordinate
        pos_2 = out.hg[1:, 2, :3, 3]
        pos_3 = out.hg[1:, 3, :3, 3]
        err = jnp.linalg.norm(pos_2 - pos_3, axis=1)
        assert jnp.all(err < 1e-6), f"Max co-location error: {float(jnp.max(err)):.2e}"
