import pytest
from jax import numpy as jnp

from flapjax.algebra.base import chi
from flapjax.algebra.se3 import p
from flapjax.algebra.test_routines import const_curvature_beam
from flapjax.structure import BeamStructure

LENGTH = jnp.array(3.45)

BEAM_DIRECTIONS = [
    pytest.param(0, jnp.array([[0.0, 1.0, 0.0]]), id="x"),
    pytest.param(1, jnp.array([[0.0, 0.0, 1.0]]), id="y"),
    pytest.param(2, jnp.array([[1.0, 0.0, 0.0]]), id="z"),
]


def _beam(direction_index, y_vector):
    coords = jnp.zeros((2, 3)).at[1, direction_index].set(LENGTH)
    struct = BeamStructure(2, jnp.array([[0, 1]]), y_vector)
    return struct, coords


@pytest.mark.parametrize("direction_index, y_vector", BEAM_DIRECTIONS)
class TestTwoNodeBeamStrainsForces:
    r"""
    Test the strains and forces for a two-node beam element with prescribed displacements
    """

    def test_unloaded(self, direction_index, y_vector):
        r"""
        Ensure undeformed beam has zero strains and internal forces
        """
        struct, coords = _beam(direction_index, y_vector)

        k_coeffs = jnp.ones(6)
        struct.set_design_variables(coords, jnp.diag(k_coeffs)[None, :], None)
        d = jnp.zeros((1, 6)).at[0, 0].set(LENGTH)
        eps = struct.make_eps(d)
        assert jnp.allclose(eps, 0.0), (
            f"Axial strain calculation incorrect, expected zero strain, got {eps}"
        )
        f_int = struct.make_f_int(
            p(d[0, :], struct.ad_inv_o0[0, ...])[None, :], eps
        )[0, :]
        assert jnp.allclose(f_int, 0.0), (
            f"Internal force vector incorrect, expected zero force, got {f_int}"
        )

    def test_axial_strain(self, direction_index, y_vector):
        r"""
        Ensure axial strain and forces are calculated correctly.
        """
        struct, coords = _beam(direction_index, y_vector)

        k_coeffs = jnp.full(6, 1e5).at[0].set(4.56)
        struct.set_design_variables(coords, jnp.diag(k_coeffs)[None, :], None)
        dx = 0.1
        d = jnp.zeros((1, 6))
        d = d.at[0, 0].set(LENGTH + dx)
        eps = struct.make_eps(d)
        expected_eps = jnp.array((dx / LENGTH, 0.0, 0.0, 0.0, 0.0, 0.0))
        assert jnp.allclose(eps, expected_eps), (
            f"Axial strain calculation incorrect, expected {expected_eps}, got {eps}"
        )
        f_int = struct.make_f_int(
            p(d[0, :], struct.ad_inv_o0[0, ...])[None, :], eps
        )[0, :]
        expected_f_int = jnp.zeros(12)
        expected_f_int = expected_f_int.at[0].set(k_coeffs[0] * dx / LENGTH)
        expected_f_int = expected_f_int.at[6].set(-k_coeffs[0] * dx / LENGTH)
        expected_f_int = chi(chi(struct.o0[0, ...])) @ expected_f_int

        assert jnp.allclose(f_int, expected_f_int), (
            f"Axial force calculation incorrect, expected {expected_f_int}, got {f_int}"
        )

    def test_torsional_strain(self, direction_index, y_vector):
        r"""
        Ensure torsional strain and forces are calculated correctly.
        """
        struct, coords = _beam(direction_index, y_vector)

        k_coeffs = jnp.full(6, 1e5).at[3].set(4.56)
        struct.set_design_variables(coords, jnp.diag(k_coeffs)[None, :], None)
        theta_x = 0.1
        d = jnp.zeros((1, 6))
        d = d.at[0, 0].set(LENGTH)
        d = d.at[0, 3].set(theta_x)
        eps = struct.make_eps(d)
        expected_eps = jnp.array((0.0, 0.0, 0.0, theta_x / LENGTH, 0.0, 0.0))
        assert jnp.allclose(eps, expected_eps), (
            f"Torsional strain calculation incorrect, expected {expected_eps}, got {eps}"
        )
        f_int = struct.make_f_int(
            p(d[0, :], struct.ad_inv_o0[0, ...])[None, :], eps
        )[0, :]
        expected_f_int = jnp.zeros(12)
        expected_f_int = expected_f_int.at[3].set(k_coeffs[3] * theta_x / LENGTH)
        expected_f_int = expected_f_int.at[9].set(-k_coeffs[3] * theta_x / LENGTH)
        expected_f_int = chi(chi(struct.o0[0, ...])) @ expected_f_int
        assert jnp.allclose(f_int, expected_f_int), (
            f"Torsional force calculation incorrect, expected {expected_f_int}, got {f_int}"
        )

    def test_bending_strain_y(self, direction_index, y_vector):
        r"""
        Ensure y-bending strain and forces are calculated correctly.
        """
        struct, coords = _beam(direction_index, y_vector)

        eiy = 2.5

        k_coeffs = jnp.full(6, 1e5).at[4].set(eiy)
        struct.set_design_variables(coords, jnp.diag(k_coeffs)[None, :], None)

        kappa_y = 1.0

        d = jnp.zeros((1, 6))
        d = d.at[0, 0].set(LENGTH)
        d = d.at[0, 4].set(kappa_y * LENGTH)

        eps = struct.make_eps(d)
        expected_eps = jnp.array((0.0, 0.0, 0.0, 0.0, kappa_y, 0.0))
        assert jnp.allclose(eps, expected_eps), (
            f"Bending strain calculation incorrect, expected {expected_eps}, got {eps}"
        )

        f_int = struct.make_f_int(
            p(d[0, :], struct.ad_inv_o0[0, ...])[None, :], eps
        )[0, :]
        expected_f_int = jnp.zeros(12)
        expected_f_int = expected_f_int.at[4].set(eiy * kappa_y)
        expected_f_int = expected_f_int.at[10].set(-eiy * kappa_y)
        expected_f_int = chi(chi(struct.o0[0, ...])) @ expected_f_int

        assert jnp.allclose(f_int, expected_f_int), (
            f"Bending force calculation incorrect, expected {expected_f_int}, got {f_int}"
        )

    def test_bending_strain_z(self, direction_index, y_vector):
        r"""
        Ensure z-bending strain and forces are calculated correctly.
        """
        struct, coords = _beam(direction_index, y_vector)

        eiz = 2.5

        k_coeffs = jnp.full(6, 1e5).at[5].set(eiz)
        struct.set_design_variables(coords, jnp.diag(k_coeffs)[None, :], None)

        kappa_z = 1.0

        d = jnp.zeros((1, 6))
        d = d.at[0, 0].set(LENGTH)
        d = d.at[0, 5].set(kappa_z * LENGTH)

        eps = struct.make_eps(d)
        expected_eps = jnp.array((0.0, 0.0, 0.0, 0.0, 0.0, kappa_z))
        assert jnp.allclose(eps, expected_eps), (
            f"Bending strain calculation incorrect, expected {expected_eps}, got {eps}"
        )

        f_int = struct.make_f_int(
            p(d[0, :], struct.ad_inv_o0[0, ...])[None, :], eps
        )[0, :]
        expected_f_int = jnp.zeros(12)
        expected_f_int = expected_f_int.at[5].set(eiz * kappa_z)
        expected_f_int = expected_f_int.at[11].set(-eiz * kappa_z)
        expected_f_int = chi(chi(struct.o0[0, ...])) @ expected_f_int

        assert jnp.allclose(f_int, expected_f_int), (
            f"Bending force calculation incorrect, expected {expected_f_int}, got {f_int}"
        )

    def test_solve_unloaded(self, direction_index, y_vector):
        r"""
        Ensure undeformed beam has zero strains and internal forces when solved for no external loads.
        """
        struct, coords = _beam(direction_index, y_vector)

        k_coeffs = jnp.full(6, 4.56)
        struct.set_design_variables(coords, jnp.diag(k_coeffs)[None, :], None)
        result = struct.static_solve(
            f_ext_follower=None,
            f_ext_dead=None,
            f_ext_aero=None,
            prescribed_dofs=jnp.arange(6),
        )

        assert jnp.allclose(result.hg, struct.hg0), (
            f"Unloaded static solve contained deformation, expected {struct.hg0}, got {result.hg}"
        )
        assert jnp.allclose(
            result.d, exp := jnp.array((LENGTH, 0.0, 0.0, 0.0, 0.0, 0.0))
        ), (
            f"Incorrect configuration for unloaded static solve, expected {exp}, got {result.d}"
        )
        assert jnp.allclose(result.f_int, 0.0), (
            f"Internal force vector incorrect for unloaded static solve, expected zero force, got {result.f_int}"
        )

    def test_solve_axial_load(self, direction_index, y_vector):
        r"""
        Ensure axial load case is solved correctly.
        """
        struct, coords = _beam(direction_index, y_vector)

        k_coeffs = jnp.full(6, 1e5).at[0].set(4.56)
        struct.set_design_variables(coords, jnp.diag(k_coeffs)[None, :], None)
        load = 0.123
        f_ext = (
            jnp.zeros((2, 6))
            .at[1, :3]
            .set(struct.o0[0, ...] @ jnp.array([load, 0.0, 0.0]))
        )
        result = struct.static_solve(
            f_ext_follower=f_ext,
            f_ext_dead=None,
            f_ext_aero=None,
            prescribed_dofs=jnp.arange(6),
        )

        expected_eps = load / k_coeffs[0]
        expected_disp = expected_eps * LENGTH
        assert jnp.allclose(
            result.d,
            exp := jnp.array((LENGTH + expected_disp, 0.0, 0.0, 0.0, 0.0, 0.0)),
        ), (
            f"Incorrect configuration for axial load static solve, expected {exp}, got {result.d}"
        )

        f_int_rot = (
            chi(chi(struct.o0[0, ...].T)) @ result.f_int.flatten()
        ).reshape(-1, 6)

        assert jnp.allclose(f_int_rot[:, 1:], 0.0), (
            f"Internal force vector expected to have zero shear/moment/torsion components, got {f_int_rot[:, 1:]}"
        )

        assert jnp.isclose(f_int_rot[0, 0], load), (
            f"Internal axial force at fixed end incorrect, expected {load}, got {f_int_rot[0, 0]}"
        )

        assert jnp.isclose(f_int_rot[1, 0], -load), (
            f"Internal axial force at loaded end incorrect, expected {-load}, got {f_int_rot[1, 0]}"
        )

    def test_solve_torsional_load(self, direction_index, y_vector):
        r"""
        Ensure torsion load case is solved correctly.
        """
        struct, coords = _beam(direction_index, y_vector)

        k_coeffs = jnp.full(6, 1e5).at[3].set(4.56)
        struct.set_design_variables(coords, jnp.diag(k_coeffs)[None, :], None)
        load = 0.123
        f_ext = (
            jnp.zeros((2, 6))
            .at[1, 3:]
            .set(struct.o0[0, ...] @ jnp.array([load, 0.0, 0.0]))
        )
        result = struct.static_solve(
            f_ext_follower=f_ext,
            f_ext_dead=None,
            f_ext_aero=None,
            prescribed_dofs=jnp.arange(6),
        )

        expected_strain = load / k_coeffs[3]
        expected_disp = expected_strain * LENGTH
        assert jnp.allclose(
            result.d,
            exp := jnp.array((LENGTH, 0.0, 0.0, expected_disp, 0.0, 0.0)),
        ), (
            f"Incorrect configuration for torsional load static solve, expected {exp}, got {result.d}"
        )

        f_int_rot = (
            chi(chi(struct.o0[0, ...].T)) @ result.f_int.flatten()
        ).reshape(-1, 6)

        assert jnp.allclose(f_int_rot[:, jnp.array((0, 1, 2, 4, 5))], 0.0), (
            f"Internal force vector expected to have zero shear/moment/axial components, got {f_int_rot}"
        )

        assert jnp.isclose(f_int_rot[0, 3], load), (
            f"Internal axial force at fixed end incorrect, expected {load}, got {f_int_rot[0, 3]}"
        )

        assert jnp.isclose(f_int_rot[1, 3], -load), (
            f"Internal axial force at loaded end incorrect, expected {-load}, got {f_int_rot[1, 3]}"
        )

    def test_solve_bending_y_load(self, direction_index, y_vector):
        r"""
        Ensure z-bending load case is solved correctly.
        """
        struct, coords = _beam(direction_index, y_vector)

        k_coeffs = jnp.full(6, 1e-3).at[4].set(4.56)
        struct.set_design_variables(coords, jnp.diag(k_coeffs)[None, :], None)
        load = 1e-1
        f_ext = (
            jnp.zeros((2, 6))
            .at[1, 3:]
            .set(struct.o0[0, ...] @ jnp.array([0.0, load, 0.0]))
        )
        result = struct.static_solve(
            f_ext_follower=f_ext,
            f_ext_dead=None,
            f_ext_aero=None,
            prescribed_dofs=jnp.arange(6),
        )

        expected_curvature = load / k_coeffs[4]
        expected_angle = expected_curvature * LENGTH

        assert jnp.allclose(
            result.d,
            exp := jnp.array((LENGTH, 0.0, 0.0, 0.0, expected_angle, 0.0)),
            atol=2e-5,
        ), (
            f"Incorrect configuration for axial load static solve, expected {exp}, got {result.d}"
        )

        f_int_rot = (
            chi(chi(struct.o0[0, ...].T)) @ result.f_int.flatten()
        ).reshape(-1, 6)

        assert jnp.allclose(f_int_rot[:, jnp.array((0, 1, 2, 3, 5))], 0.0), (
            f"Internal force vector expected to have zero axial/shear/moment/z-moment components, got {f_int_rot}"
        )

        assert jnp.isclose(f_int_rot[0, 4], load), (
            f"Internal bending moment at fixed end incorrect, expected {load}, got {f_int_rot[0, 4]}"
        )

        assert jnp.isclose(f_int_rot[1, 4], -load), (
            f"Internal bending moment at loaded end incorrect, expected {-load}, got {f_int_rot[1, 4]}"
        )

        coord_tip = struct.o0[0, ...].T @ result.x[1, :]

        expected_coord_tip = const_curvature_beam(
            expected_curvature, LENGTH, direction="y"
        )

        assert jnp.allclose(
            coord_tip,
            expected_coord_tip,
            atol=1e-5,
        ), (
            f"Incorrect tip coordinate for bending y load static solve, expected {expected_coord_tip}, got {coord_tip}"
        )

    def test_solve_bending_z_load(self, direction_index, y_vector):
        r"""
        Ensure bending in z load case is solved correctly.
        """
        struct, coords = _beam(direction_index, y_vector)

        k_coeffs = jnp.full(6, 1e-3).at[5].set(4.56)
        struct.set_design_variables(coords, jnp.diag(k_coeffs)[None, :], None)
        load = 1e-1
        f_ext = (
            jnp.zeros((2, 6))
            .at[1, 3:]
            .set(struct.o0[0, ...] @ jnp.array([0.0, 0.0, load]))
        )
        result = struct.static_solve(
            f_ext_follower=f_ext,
            f_ext_dead=None,
            f_ext_aero=None,
            prescribed_dofs=jnp.arange(6),
        )

        expected_curvature = load / k_coeffs[5]
        expected_angle = expected_curvature * LENGTH

        assert jnp.allclose(
            result.d,
            exp := jnp.array((LENGTH, 0.0, 0.0, 0.0, 0.0, expected_angle)),
            atol=2e-5,
        ), (
            f"Incorrect configuration for axial load static solve, expected {exp}, got {result.d}"
        )

        f_int_rot = (
            chi(chi(struct.o0[0, ...].T)) @ result.f_int.flatten()
        ).reshape(-1, 6)

        assert jnp.allclose(f_int_rot[:, jnp.array((0, 1, 2, 3, 4))], 0.0), (
            f"Internal force vector expected to have zero axial/shear/y-moment components, got {f_int_rot}"
        )

        assert jnp.isclose(f_int_rot[0, 5], load), (
            f"Internal z-moment at fixed end incorrect, expected {load}, got {f_int_rot[0, 5]}"
        )

        assert jnp.isclose(f_int_rot[1, 5], -load), (
            f"Internal z-moment at loaded end incorrect, expected {-load}, got {f_int_rot[1, 5]}"
        )

        coord_tip = struct.o0[0, ...].T @ result.x[1, :3]

        expected_coord_tip = const_curvature_beam(
            expected_curvature, LENGTH, direction="z"
        )

        assert jnp.allclose(
            coord_tip,
            expected_coord_tip,
            atol=1e-5,
        ), (
            f"Incorrect tip coordinate for bending z load static solve, expected {expected_coord_tip}, got {coord_tip}"
        )

    def test_solve_shear_y_load(self, direction_index, y_vector):
        r"""
        Ensure shear load case is solved correctly.
        """
        struct, coords = _beam(direction_index, y_vector)

        k_coeffs = jnp.full(6, 1e5).at[1].set(4.56)
        struct.set_design_variables(coords, jnp.diag(k_coeffs)[None, :], None)
        load = 7.89
        f_ext = (
            jnp.zeros((2, 6))
            .at[1, :3]
            .set(struct.o0[0, ...] @ jnp.array([0.0, load, 0.0]))
        )
        result = struct.static_solve(
            f_ext_follower=f_ext,
            f_ext_dead=None,
            f_ext_aero=None,
            prescribed_dofs=jnp.concatenate((jnp.arange(6), jnp.arange(9, 12))),
        )

        expected_strain = load / k_coeffs[1]
        expected_disp = expected_strain * LENGTH
        expected_moment = -0.5 * load * LENGTH

        assert jnp.allclose(
            result.d,
            exp := jnp.array((LENGTH, expected_disp, 0.0, 0.0, 0.0, 0.0)),
        ), (
            f"Incorrect configuration for shear_y load static solve, expected {exp}, got {result.d}"
        )

        f_int_rot = (
            chi(chi(struct.o0[0, ...].T)) @ result.f_int.flatten()
        ).reshape(-1, 6)

        assert jnp.allclose(f_int_rot[:, jnp.array((0, 2, 3, 4))], 0.0), (
            f"Internal force vector expected to have zero axial/torsion/z_shear components, got {f_int_rot}"
        )

        assert jnp.isclose(f_int_rot[0, 1], load), (
            f"Internal shear_y force at fixed end incorrect, expected {load}, got {f_int_rot[0, 1]}"
        )

        assert jnp.isclose(f_int_rot[1, 1], -load), (
            f"Internal shear_y force at loaded end incorrect, expected {-load}, got {f_int_rot[1, 1]}"
        )

        assert jnp.isclose(f_int_rot[0, 5], -expected_moment), (
            f"Internal moment at fixed end incorrect, expected {-expected_moment}, got {f_int_rot[0, 5]}",
        )

        assert jnp.isclose(f_int_rot[1, 5], -expected_moment), (
            f"Internal moment at loaded end incorrect, expected {-expected_moment}, got {result.f_int[1, 5]}"
        )

    def test_solve_shear_z_load(self, direction_index, y_vector):
        r"""
        Ensure shear load case is solved correctly.
        """
        struct, coords = _beam(direction_index, y_vector)

        k_coeffs = jnp.full(6, 1e5).at[2].set(4.56)
        struct.set_design_variables(coords, jnp.diag(k_coeffs)[None, :], None)
        load = 7.89
        f_ext = (
            jnp.zeros((2, 6))
            .at[1, :3]
            .set(struct.o0[0, ...] @ jnp.array([0.0, 0.0, load]))
        )
        result = struct.static_solve(
            f_ext_follower=f_ext,
            f_ext_dead=None,
            f_ext_aero=None,
            prescribed_dofs=jnp.concatenate((jnp.arange(6), jnp.arange(9, 12))),
        )

        expected_strain = load / k_coeffs[2]
        expected_disp = expected_strain * LENGTH
        expected_moment = 0.5 * load * LENGTH

        assert jnp.allclose(
            result.d,
            exp := jnp.array((LENGTH, 0.0, expected_disp, 0.0, 0.0, 0.0)),
        ), (
            f"Incorrect configuration for shear_y load static solve, expected {exp}, got {result.d}"
        )

        f_int_rot = (
            chi(chi(struct.o0[0, ...].T)) @ result.f_int.flatten()
        ).reshape(-1, 6)

        assert jnp.allclose(f_int_rot[:, jnp.array((0, 1, 3, 5))], 0.0), (
            f"Internal force vector expected to have zero axial/torsion/y-shear components, got {f_int_rot}"
        )

        assert jnp.isclose(f_int_rot[0, 2], load), (
            f"Internal shear_z force at fixed end incorrect, expected {load}, got {f_int_rot[0, 2]}"
        )

        assert jnp.isclose(f_int_rot[1, 2], -load), (
            f"Internal shear_z force at loaded end incorrect, expected {-load}, got {f_int_rot[1, 2]}"
        )

        assert jnp.isclose(f_int_rot[0, 4], -expected_moment), (
            f"Internal moment at fixed end incorrect, expected {-expected_moment}, got {f_int_rot[0, 4]}",
        )

        assert jnp.isclose(f_int_rot[1, 4], -expected_moment), (
            f"Internal moment at loaded end incorrect, expected {-expected_moment}, got {f_int_rot[1, 4]}"
        )
