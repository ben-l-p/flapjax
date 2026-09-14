import pytest
from jax import numpy as jnp

from flapjax.structure import BeamStructure
from flapjax.structure.constraints import Hinge, SpringDamper


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


class TestSpringDamper:
    r"""
    A very stiff spring-damper at the root reproduces the result of clamping with eliminating degrees of freedom.
    """

    n_nodes = 6

    @classmethod
    def test_stiff_spring_reproduces_hard_clamp(cls):
        tip_force = jnp.zeros((cls.n_nodes, 6)).at[-1, 2].set(1.0)

        # hard constraint case
        beam_hard = _cantilever_beam(n_nodes=cls.n_nodes)
        res_hard = beam_hard.static_solve(
            prescribed_dofs=tuple(range(6)), f_ext_dead=tip_force
        )

        # soft constraint case with very stiff spring
        spring = SpringDamper(node_index=0, k=jnp.eye(6) * 1e10, hg_ref=jnp.eye(4))
        beam_soft = _cantilever_beam(n_nodes=cls.n_nodes, nodal_constraints=[spring])
        res_soft = beam_soft.static_solve(prescribed_dofs=(), f_ext_dead=tip_force)

        assert jnp.allclose(res_soft.hg[0, :3, 3], 0.0, atol=1e-8), (
            "Root coordinate should not be displaced"
        )
        assert jnp.allclose(
            res_hard.hg[-1, :3, 3], res_soft.hg[-1, :3, 3], rtol=1e-4
        ), (
            f"Tip deflection should match hard constraint enforcement, got hard={res_hard.hg[-1, :3, 3]}, "
            f"soft={res_soft.hg[-1, :3, 3]}"
        )

    @classmethod
    def test_spring_provides_finite_compliance(cls):
        r"""
        With a moderate spring stiffness the root moves with linear spring relations when the tip is subject to a
        linear dead force.
        """
        f = 1.0
        k = 1e4
        tip_force = jnp.zeros((cls.n_nodes, 6)).at[-1, 2].set(f)

        spring = SpringDamper(node_index=0, k=jnp.eye(6) * k, hg_ref=jnp.eye(4))
        beam = _cantilever_beam(n_nodes=cls.n_nodes, nodal_constraints=[spring])
        sol = beam.static_solve(prescribed_dofs=(), f_ext_dead=tip_force)

        # root z-displacement should equal force / stiffness
        expected_root_z = f / k
        assert jnp.isclose(sol.hg[0, 2, 3], expected_root_z, rtol=1e-3), (
            f"Root z should be F/k = {expected_root_z}, got {sol.hg[0, 2, 3]}"
        )


@pytest.mark.parametrize("hinge_dir_idx", [
    pytest.param(0, id="x"),
    pytest.param(1, id="y"),
    pytest.param(2, id="z"),
])
class TestHinge:
    r"""
    Hinge with axis aligned with :math:`\hat{\mathbf{y}}`: rotation about y is
    free (governed by ``k_axis``), rotations about x/z and translations are
    blocked.
    """

    def test_free_axis_rotates_under_moment(self, hinge_dir_idx):
        r"""
        Stiff beam attached to ground by hinge with torsional
        stiffness. Applying a moment should result in a linear relationship with angle about the hinge axis.
        """
        n_nodes = 3
        k_axis = 100.0
        moment = 1.0

        hinge = Hinge(
            node_index=0,
            axis=jnp.zeros(3).at[hinge_dir_idx].set(1.0),
            hg_ref=jnp.eye(4),
            k_translation=1e12,
            k_perpendicular=1e12,
            k_axis=k_axis,
        )

        beam = _cantilever_beam(n_nodes=n_nodes, k_scale=1e6, nodal_constraints=[hinge])

        f_ext = jnp.zeros((n_nodes, 6)).at[0, hinge_dir_idx + 3].set(moment)
        res = beam.static_solve(prescribed_dofs=(), f_ext_dead=f_ext)

        # rotation of node 0 about y
        theta_y = res.varphi[0, hinge_dir_idx + 3]
        expected = moment / k_axis
        assert jnp.isclose(theta_y, expected, rtol=1e-3), (
            f"Free-axis rotation should be M/k_axis = {expected}, got {theta_y}"
        )

    def test_constrained_axis_barely_rotates(self, hinge_dir_idx):
        r"""
        Same setup as the free-axis test but applying moment around a non-free axis, which should use the perpendicular
        stiffness.
        """
        n_nodes = 3
        k_axis = 100.0
        k_perp = 1e8
        moment = 1.0

        hinge = Hinge(
            node_index=0,
            axis=jnp.zeros(3).at[hinge_dir_idx].set(1.0),
            hg_ref=jnp.eye(4),
            k_translation=1e12,
            k_perpendicular=k_perp,
            k_axis=k_axis,
        )
        beam = _cantilever_beam(n_nodes=n_nodes, k_scale=1e6, nodal_constraints=[hinge])

        i_non_axis = (hinge_dir_idx + 1) % 3

        f_ext = jnp.zeros((n_nodes, 6)).at[0, i_non_axis + 3].set(moment)
        res = beam.static_solve(prescribed_dofs=(), f_ext_dead=f_ext)

        theta_x = res.varphi[0, i_non_axis + 3]
        expected = moment / k_perp
        assert jnp.isclose(theta_x, expected, rtol=1e-2), (
            f"Constrained-axis rotation should be ~M/k_perp = {expected}, got {theta_x}"
        )
