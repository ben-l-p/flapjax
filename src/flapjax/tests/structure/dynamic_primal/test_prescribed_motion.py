from jax import numpy as jnp

from flapjax.structure import BeamStructure
from flapjax.structure.constraints import PrescribedMotion


# create a beam that takes input nodal constraints
def _short_beam(n_nodes: int, constraints):
    conn = jnp.stack((jnp.arange(n_nodes - 1), jnp.arange(1, n_nodes)), axis=1).astype(
        int
    )
    beam = BeamStructure(
        num_nodes=n_nodes,
        connectivity=conn,
        y_vector=jnp.array((0.0, 0.0, 1.0)),
        constraints=constraints,
    )
    coords = jnp.stack(
        (
            jnp.linspace(0.0, 1.0, n_nodes),
            jnp.zeros(n_nodes),
            jnp.zeros(n_nodes),
        ),
        axis=1,
    )
    beam.set_design_variables(
        coords=coords,
        k_cs=jnp.diag(jnp.array((1e6, 1e6, 1e6, 1e2, 1e2, 1e2))),
        m_cs=jnp.diag(jnp.array((1.0, 1.0, 1.0, 0.1, 0.1, 0.1))),
    )
    return beam


class TestPrescribedMotion:
    r"""
    Verify that a node driven by a `PrescribedMotion` constraint converges to the reference trajectory. This uses heavy
    spring and damping coefficients to ensure that the node tracks the reference closely.
    """

    n_nodes = 4
    n_tstep = 80
    dt = 0.005

    @classmethod
    def _run_with_reference(cls, hg_ref_t):
        pm = PrescribedMotion(
            node_index=0,
            k=jnp.eye(6) * 1e6,
            hg_ref_t=hg_ref_t,
            c=jnp.eye(6) * 1e4,
        )
        beam = _short_beam(cls.n_nodes, constraints={"prescribed_motion": pm})
        return beam.dynamic_solve(
            init_state=None,
            prescribed_dofs=(),
            n_tstep=cls.n_tstep,
            dt=cls.dt,
        )

    @classmethod
    def test_constant_translational_offset(cls):
        r"""
        Perturb root coordinate for t>0, and ensure that it is tracked after some time.
        """
        offset = 0.02
        hg_ref_t = (
            jnp.broadcast_to(jnp.eye(4)[None, ...], (cls.n_tstep, 4, 4))
            .at[1:, 0, 3]
            .set(offset)
        )

        res = cls._run_with_reference(hg_ref_t)

        # settled position over the final 10 steps
        settled = res.hg[-10:, 0, 0, 3]
        assert jnp.all(jnp.abs(settled - offset) < 1e-4), (
            f"Root x should settle at {offset}, got trajectory tail {settled}"
        )

    @classmethod
    def test_ramp_trajectory(cls):
        r"""
        Linear ramp in x coordinate of the root node. The root should track the ramp with a small lag.
        """
        rate = 0.001  # per step, small enough that the tracking spring keeps up
        target_x = rate * jnp.arange(cls.n_tstep)
        hg_ref_t = (
            jnp.broadcast_to(jnp.eye(4)[None, ...], (cls.n_tstep, 4, 4))
            .at[:, 0, 3]
            .set(target_x)
        )

        res = cls._run_with_reference(hg_ref_t)

        # last 10 steps
        settled = res.hg[-10:, 0, 0, 3]
        target = target_x[-10:]
        err = jnp.abs(settled - target)
        assert jnp.all(err < 0.05 * jnp.abs(target)), (
            f"Ramp tracking error too high: settled={settled}, target={target}"
        )

    @classmethod
    def test_tracks_prescribed_rotation(cls):
        r"""
        Rotate the root about the y-axis by a fixed angle, and ensure that the rotation is achieved after some time.
        """
        theta = 0.05  # angle to achieve

        c_t, s_t = jnp.cos(theta), jnp.sin(theta)
        rot_y = (
            jnp.eye(4)
            .at[0, 0]
            .set(c_t)
            .at[0, 2]
            .set(s_t)
            .at[2, 0]
            .set(-s_t)
            .at[2, 2]
            .set(c_t)
        )

        hg_ref_t = (
            jnp.broadcast_to(jnp.eye(4)[None, ...], (cls.n_tstep, 4, 4))
            .at[1:]
            .set(rot_y)
        )

        res = cls._run_with_reference(hg_ref_t)

        # extract y-rotation from varphi
        theta_y = res.varphi[-1, 0, 4]
        assert jnp.isclose(theta_y, theta, atol=1e-3), (
            f"Prescribed y-rotation should settle to {theta}, got {theta_y}"
        )
