import pytest
from jax import numpy as jnp

from flapjax.algebra.base import chi
from flapjax.algebra.test_routines import const_curvature_beam
from flapjax.structure import BeamStructure

N_NODES = 10
N_ELEM = N_NODES - 1
CONN = jnp.zeros((N_ELEM, 2), dtype=int)
CONN = CONN.at[:, 0].set(jnp.arange(N_ELEM))
CONN = CONN.at[:, 1].set(jnp.arange(1, N_ELEM + 1))

BEAM_PARAMS = [
    pytest.param(0, 1, jnp.array(2.0), id="x"),
    pytest.param(1, 2, jnp.array(2.5), id="y"),
    pytest.param(2, 0, jnp.array(2.5), id="z"),
]


def _beam(direction_index, y_vect_col, length):
    coords = jnp.zeros((N_NODES, 3)).at[:, direction_index].set(jnp.linspace(0, length, N_NODES))
    y_vect = jnp.zeros((N_ELEM, 3)).at[:, y_vect_col].set(1.0)
    struct = BeamStructure(N_NODES, CONN, y_vect)
    return struct, coords


@pytest.mark.parametrize("direction_index, y_vect_col, length", BEAM_PARAMS)
class TestMultiElementStrainsForces:
    r"""
    Test the strains and forces for a two-node beam element with prescribed displacements
    """

    def test_unloaded(self, direction_index, y_vect_col, length):
        r"""
        Ensure undeformed beam has zero strains and internal forces
        """
        struct, coords = _beam(direction_index, y_vect_col, length)

        k_coeffs = jnp.full(6, 4.56)
        struct.set_design_variables(coords, jnp.diag(k_coeffs)[None, :], None)
        d = jnp.zeros((N_ELEM, 6)).at[:, 0].set(length / N_ELEM)
        eps = struct.make_eps(d)
        assert jnp.allclose(eps, 0.0), (
            f"Strain calculation incorrect, expected zero strain, got {eps}"
        )
        f_int = struct.make_f_int(struct.make_p_d(d), eps)[0]
        assert jnp.allclose(f_int, 0.0), (
            f"Internal force vector incorrect, expected zero, got {f_int}"
        )

    def test_axial_strain(self, direction_index, y_vect_col, length):
        r"""
        Ensure axial strain and forces are calculated correctly.
        """
        struct, coords = _beam(direction_index, y_vect_col, length)

        k_coeffs = jnp.full(6, 1e5).at[0].set(4.56)
        struct.set_design_variables(coords, jnp.diag(k_coeffs)[None, :], None)
        dx = 0.1
        d = jnp.zeros((N_ELEM, 6))
        d = d.at[:, 0].set((length + dx) / N_ELEM)
        eps = struct.make_eps(d)
        expected_eps = jnp.array((dx / length, 0.0, 0.0, 0.0, 0.0, 0.0))[None, :]
        expected_f = k_coeffs[0] * dx / length

        assert jnp.allclose(eps, expected_eps), (
            f"Axial strain calculation incorrect, expected {expected_eps}, got {eps}"
        )
        f_int = struct.assemble_vector_from_entries(
            struct.make_f_int(struct.make_p_d(d), eps)
        ).reshape(-1, 6)

        f_int_rot = jnp.einsum("ij,kj->ki", chi(struct.o0[0, ...].T), f_int)

        assert jnp.allclose(f_int_rot[0, 0], expected_f), (
            f"Axial force calculation at root incorrect, expected {expected_f}, got {f_int_rot[0, 0]}"
        )

        assert jnp.allclose(f_int_rot[-1, 0], -expected_f), (
            f"Axial force calculation at tip incorrect, expected {-expected_f}, got {f_int_rot[-1, 0]}"
        )

        index_zero = jnp.array(
            tuple(set(range(6 * N_NODES)) - {0, 6 * (N_NODES - 1)})
        )
        assert jnp.allclose(f_int_rot.ravel()[index_zero], 0.0), (
            f"Axial force in beam incorrect, expected zero, got {f_int_rot}"
        )

    def test_torsional_strain(self, direction_index, y_vect_col, length):
        r"""
        Ensure torsional strain and forces are calculated correctly.
        """
        struct, coords = _beam(direction_index, y_vect_col, length)

        k_coeffs = jnp.full(6, 1e5).at[3].set(4.56)
        struct.set_design_variables(coords, jnp.diag(k_coeffs)[None, :], None)
        dx = 0.1
        d = jnp.zeros((N_ELEM, 6))
        d = d.at[:, 0].set(length / N_ELEM)
        d = d.at[:, 3].set(dx / N_ELEM)

        eps = struct.make_eps(d)
        expected_strain = jnp.array((0.0, 0.0, 0.0, dx / length, 0.0, 0.0))[None, :]
        expected_f = k_coeffs[3] * dx / length

        assert jnp.allclose(eps, expected_strain), (
            f"Torsional strain calculation incorrect, expected {expected_strain}, got {eps}"
        )
        f_int = struct.assemble_vector_from_entries(
            struct.make_f_int(struct.make_p_d(d), eps)
        ).reshape(-1, 6)

        f_int_rot = jnp.einsum("ij,kj->ki", chi(struct.o0[0, ...].T), f_int)

        assert jnp.allclose(f_int_rot[0, 3], expected_f), (
            f"Torsional force calculation at root incorrect, expected {expected_f}, got {f_int_rot[0, 3]}"
        )

        assert jnp.allclose(f_int_rot[-1, 3], -expected_f), (
            f"Torsional force calculation at tip incorrect, expected {-expected_f}, got {f_int_rot[-1, 3]}"
        )

        index_zero = jnp.array(
            tuple(set(range(6 * N_NODES)) - {3, 6 * (N_NODES - 1) + 3})
        )
        assert jnp.allclose(f_int_rot.ravel()[index_zero], 0.0), (
            f"Torsional force in beam incorrect, expected zero, got {f_int_rot}"
        )

    def test_bending_y_strain(self, direction_index, y_vect_col, length):
        r"""
        Ensure y-bending strain and forces are calculated correctly.
        """
        struct, coords = _beam(direction_index, y_vect_col, length)

        k_coeffs = jnp.full(6, 1e5).at[4].set(4.56)
        struct.set_design_variables(coords, jnp.diag(k_coeffs)[None, :], None)
        dx = 0.1
        d = jnp.zeros((N_ELEM, 6))
        d = d.at[:, 0].set(length / N_ELEM)
        d = d.at[:, 4].set(dx / N_ELEM)

        eps = struct.make_eps(d)
        expected_bending_strain = jnp.array((0.0, 0.0, 0.0, 0.0, dx / length, 0.0))[
            None, :
        ]
        expected_f = k_coeffs[4] * dx / length

        assert jnp.allclose(eps, expected_bending_strain), (
            f"Bending strain calculation incorrect, expected {expected_bending_strain}, got {eps}"
        )
        f_int = struct.assemble_vector_from_entries(
            struct.make_f_int(struct.make_p_d(d), eps)
        ).reshape(-1, 6)
        f_int_rot = jnp.einsum("ij,kj->ki", chi(struct.o0[0, ...].T), f_int)

        assert jnp.allclose(f_int_rot[0, 4], expected_f), (
            f"Bending moment calculation at root incorrect, expected {expected_f}, got {f_int_rot[0, 4]}"
        )

        assert jnp.allclose(f_int_rot[-1, 4], -expected_f), (
            f"Bending calculation at tip incorrect, expected {-expected_f}, got {f_int_rot[-1, 4]}"
        )

        index_zero = jnp.array(
            tuple(set(range(6 * N_NODES)) - {4, 6 * (N_NODES - 1) + 4})
        )
        assert jnp.allclose(f_int_rot.ravel()[index_zero], 0.0), (
            f"Torsional force in beam incorrect, expected zero, got {f_int_rot}"
        )

    def test_bending_z_strain(self, direction_index, y_vect_col, length):
        r"""
        Ensure z-bending strain and forces are calculated correctly.
        """
        struct, coords = _beam(direction_index, y_vect_col, length)

        k_coeffs = jnp.full(6, 1e5).at[5].set(4.56)
        struct.set_design_variables(coords, jnp.diag(k_coeffs)[None, :], None)
        dx = 0.1
        d = jnp.zeros((N_ELEM, 6))
        d = d.at[:, 0].set(length / N_ELEM)
        d = d.at[:, 5].set(dx / N_ELEM)

        eps = struct.make_eps(d)
        expected_eps = jnp.array((0.0, 0.0, 0.0, 0.0, 0.0, dx / length))[None, :]
        expected_f = k_coeffs[5] * dx / length

        assert jnp.allclose(eps, expected_eps), (
            f"Bending strain calculation incorrect, expected {expected_eps}, got {eps}"
        )
        f_int = struct.assemble_vector_from_entries(
            struct.make_f_int(struct.make_p_d(d), eps)
        ).reshape(-1, 6)
        f_int_rot = jnp.einsum("ij,kj->ki", chi(struct.o0[0, ...].T), f_int)

        assert jnp.allclose(f_int_rot[0, 5], expected_f), (
            f"Bending moment calculation at root incorrect, expected {expected_f}, got {f_int_rot[0, 5]}"
        )

        assert jnp.allclose(f_int_rot[-1, 5], -expected_f), (
            f"Bending calculation at tip incorrect, expected {-expected_f}, got {f_int_rot[-1, 5]}"
        )

        index_zero = jnp.array(
            tuple(set(range(6 * N_NODES)) - {5, 6 * (N_NODES - 1) + 5})
        )
        assert jnp.allclose(f_int_rot.ravel()[index_zero], 0.0), (
            f"Torsional force in beam incorrect, expected zero, got {f_int_rot}"
        )

    def test_solve_unloaded(self, direction_index, y_vect_col, length):
        r"""
        Ensure undeformed beam has zero strains and internal forces when solved for no external loads.
        """
        struct, coords = _beam(direction_index, y_vect_col, length)

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
            result.d,
            exp := jnp.array((length / N_ELEM, 0.0, 0.0, 0.0, 0.0, 0.0))[
                None, :
            ],
        ), (
            f"Incorrect configuration for unloaded static solve, expected {exp}, got {result.d}"
        )
        assert jnp.allclose(result.f_int, 0.0), (
            f"Internal force vector incorrect for unloaded static solve, expected zero force, got {result.f_int}"
        )

    def test_solve_axial_load(self, direction_index, y_vect_col, length):
        r"""
        Ensure axial load case is solved correctly.
        """
        struct, coords = _beam(direction_index, y_vect_col, length)

        k_coeffs = jnp.full(6, 1e5).at[0].set(4.56)
        struct.set_design_variables(coords, jnp.diag(k_coeffs)[None, :], None)
        load = 1.23
        f_ext = (
            jnp.zeros((N_NODES, 6))
            .at[-1, :3]
            .set(struct.o0[0, ...] @ jnp.array([load, 0.0, 0.0]))
        )
        result = struct.static_solve(
            f_ext_follower=f_ext,
            f_ext_dead=None,
            f_ext_aero=None,
            prescribed_dofs=jnp.arange(6),
        )

        expected_eps = load / k_coeffs[0]
        expected_disp = expected_eps * length
        assert jnp.allclose(
            result.d,
            exp := jnp.array(
                ((length + expected_disp) / N_ELEM, 0.0, 0.0, 0.0, 0.0, 0.0)
            )[None, :],
        ), (
            f"Incorrect configuration for axial load static solve, expected {exp}, got {result.d}"
        )

        f_int = result.f_int
        f_int_rot = jnp.einsum("ij,kj->ki", chi(struct.o0[0, ...].T), f_int)

        zero_index = jnp.array(
            tuple(set(range(N_NODES * 6)) - {0, (N_NODES - 1) * 6})
        )

        assert jnp.allclose(f_int_rot.ravel()[zero_index], 0.0), (
            f"Internal force vector expected to have zero shear/moment/torsion components, got {f_int_rot}"
        )

        assert jnp.isclose(f_int_rot[0, 0], load), (
            f"Internal axial force at fixed end incorrect, expected {load}, got {f_int_rot[0, 0]}"
        )

        assert jnp.isclose(f_int_rot[-1, 0], -load), (
            f"Internal axial force at loaded end incorrect, expected {-load}, got {f_int_rot[-1, 0]}"
        )

    def test_solve_torsional_load(self, direction_index, y_vect_col, length):
        r"""
        Ensure torsional load case is solved correctly.
        """
        struct, coords = _beam(direction_index, y_vect_col, length)

        k_coeffs = jnp.full(6, 1e5).at[3].set(4.56)
        struct.set_design_variables(coords, jnp.diag(k_coeffs)[None, :], None)
        load = 1.23
        f_ext = (
            jnp.zeros((N_NODES, 6))
            .at[-1, 3:]
            .set(struct.o0[0, ...] @ jnp.array([load, 0.0, 0.0]))
        )
        result = struct.static_solve(
            f_ext_follower=f_ext,
            f_ext_dead=None,
            f_ext_aero=None,
            prescribed_dofs=jnp.arange(6),
        )

        expected_strain = load / k_coeffs[3]
        expected_disp = expected_strain * length
        assert jnp.allclose(
            result.d,
            exp := jnp.array(
                (
                    length / N_ELEM,
                    0.0,
                    0.0,
                    expected_disp / N_ELEM,
                    0.0,
                    0.0,
                )
            )[None, :],
        ), f"Incorrect configuration for static solve, expected {exp}, got {result.d}"

        f_int = result.f_int
        f_int_rot = jnp.einsum("ij,kj->ki", chi(struct.o0[0, ...].T), f_int)

        zero_index = jnp.array(
            tuple(set(range(N_NODES * 6)) - {3, (N_NODES - 1) * 6 + 3})
        )
        assert jnp.allclose(f_int_rot.ravel()[zero_index], 0.0), (
            f"Internal force vector expected to have zero axial/shear/moment components, got {f_int_rot}"
        )

        assert jnp.isclose(f_int_rot[0, 3], load), (
            f"Internal axial force at fixed end incorrect, expected {load}, got {f_int_rot[0, 3]}"
        )

        assert jnp.isclose(f_int_rot[-1, 3], -load), (
            f"Internal axial force at loaded end incorrect, expected {-load}, got {f_int_rot[-1, 3]}"
        )

    def test_solve_y_bending_load(self, direction_index, y_vect_col, length):
        r"""
        Ensure y-bending load case is solved correctly.
        """
        struct, coords = _beam(direction_index, y_vect_col, length)

        k_coeffs = jnp.full(6, 1e5).at[4].set(4.56)
        struct.set_design_variables(coords, jnp.diag(k_coeffs)[None, :], None)
        load = 1.23
        f_ext = (
            jnp.zeros((N_NODES, 6))
            .at[-1, 3:]
            .set(struct.o0[0, ...] @ jnp.array([0.0, load, 0.0]))
        )
        result = struct.static_solve(
            f_ext_follower=f_ext,
            f_ext_dead=None,
            f_ext_aero=None,
            prescribed_dofs=jnp.arange(6),
        )

        expected_strain = load / k_coeffs[4]
        expected_disp = expected_strain * length
        assert jnp.allclose(
            result.d,
            exp := jnp.array(
                (
                    length / N_ELEM,
                    0.0,
                    0.0,
                    0.0,
                    expected_disp / N_ELEM,
                    0.0,
                )
            )[None, :],
        ), f"Incorrect configuration for static solve, expected {exp}, got {result.d}"

        f_int = result.f_int
        f_int_rot = jnp.einsum("ij,kj->ki", chi(struct.o0[0, ...].T), f_int)

        zero_index = jnp.array(
            tuple(set(range(N_NODES * 6)) - {4, (N_NODES - 1) * 6 + 4})
        )
        assert jnp.allclose(f_int_rot.ravel()[zero_index], 0.0), (
            f"Internal force vector expected to have zero axial/shear/torsional components, got {f_int_rot}"
        )

        assert jnp.isclose(f_int_rot[0, 4], load), (
            f"Internal moment at fixed end incorrect, expected {load}, got {f_int_rot[0, 4]}"
        )

        assert jnp.isclose(f_int_rot[-1, 4], -load), (
            f"Internal moment at loaded end incorrect, expected {-load}, got {f_int_rot[-1, 4]}"
        )

        coord_tip = struct.o0[-1, :, :].T @ result.x[-1, :]

        expected_coord_tip = const_curvature_beam(
            expected_strain, length, direction="y"
        )

        assert jnp.allclose(
            coord_tip,
            expected_coord_tip,
            atol=1e-5,
        ), (
            f"Incorrect tip coordinate for bending y load static solve, expected {expected_coord_tip}, got {coord_tip}"
        )

    def test_solve_z_bending_load(self, direction_index, y_vect_col, length):
        r"""
        Ensure z-bending load case is solved correctly.
        """
        struct, coords = _beam(direction_index, y_vect_col, length)

        k_coeffs = jnp.full(6, 1e5).at[5].set(4.56)
        struct.set_design_variables(coords, jnp.diag(k_coeffs)[None, :], None)
        load = 1.23
        f_ext = (
            jnp.zeros((N_NODES, 6))
            .at[-1, 3:]
            .set(struct.o0[0, ...] @ jnp.array([0.0, 0.0, load]))
        )
        result = struct.static_solve(
            f_ext_follower=f_ext,
            f_ext_dead=None,
            f_ext_aero=None,
            prescribed_dofs=jnp.arange(6),
        )

        expected_strain = load / k_coeffs[5]
        expected_disp = expected_strain * length
        assert jnp.allclose(
            result.d,
            exp := jnp.array(
                (
                    length / N_ELEM,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    expected_disp / N_ELEM,
                )
            )[None, :],
        ), f"Incorrect configuration for static solve, expected {exp}, got {result.d}"

        f_int = result.f_int
        f_int_rot = jnp.einsum("ij,kj->ki", chi(struct.o0[0, ...].T), f_int)

        zero_index = jnp.array(
            tuple(set(range(N_NODES * 6)) - {5, (N_NODES - 1) * 6 + 5})
        )
        assert jnp.allclose(f_int_rot.ravel()[zero_index], 0.0), (
            f"Internal force vector expected to have zero axial/shear/torsional components, got {f_int_rot}"
        )

        assert jnp.isclose(f_int_rot[0, 5], load), (
            f"Internal moment at fixed end incorrect, expected {load}, got {f_int_rot[0, 5]}"
        )

        assert jnp.isclose(f_int_rot[-1, 5], -load), (
            f"Internal moment at loaded end incorrect, expected {-load}, got {f_int_rot[-1, 5]}"
        )

        coord_tip = struct.o0[-1, :, :].T @ result.x[-1, :]

        expected_coord_tip = const_curvature_beam(
            expected_strain, length, direction="z"
        )

        assert jnp.allclose(
            coord_tip,
            expected_coord_tip,
            atol=1e-5,
        ), (
            f"Incorrect tip coordinate for bending z load static solve, expected {expected_coord_tip}, got {coord_tip}"
        )

    def test_solve_shear_y_load(self, direction_index, y_vect_col, length):
        r"""
        Ensure y-shear load case is solved correctly.
        Note - this should not be expected to converge for z-bending moment, and as such has weaker tolerances.
        """
        struct, coords = _beam(direction_index, y_vect_col, length)

        k_coeffs = jnp.full(6, 1e5).at[1].set(4.56)
        k_coeffs = k_coeffs.at[0].set(1e2)  # allow the beam to stretch
        struct.set_design_variables(coords, jnp.diag(k_coeffs)[None, :], None)
        load = 1.23
        f_ext = (
            jnp.zeros((N_NODES, 6))
            .at[-1, :3]
            .set(struct.o0[0, ...] @ jnp.array([0.0, load, 0.0]))
        )
        result = struct.static_solve(
            f_ext_follower=f_ext,
            f_ext_dead=None,
            f_ext_aero=None,
            prescribed_dofs=jnp.concatenate(
                (jnp.arange(6), (N_NODES - 1) * 6 + jnp.arange(3, 6))
            ),
            load_steps=4,
        )

        expected_eps = load / k_coeffs[1]
        expected_disp = expected_eps * length
        expected_moment = -0.5 * load * length

        assert jnp.allclose(
            result.d,
            exp := jnp.array(
                (
                    length / N_ELEM,
                    expected_disp / N_ELEM,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
            )[None, :],
            atol=1e-5,
            rtol=1e-3,
        ), (
            f"Incorrect configuration for shear_y load static solve, expected {exp}, got {result.d}"
        )

        f_int = result.f_int
        f_int_rot = jnp.einsum("ij,kj->ki", chi(struct.o0[0, ...].T), f_int)

        assert jnp.isclose(f_int_rot[0, 1], load, atol=1e-4), (
            f"Internal shear_y force at fixed end incorrect, expected {load}, got {f_int_rot[0, 1]}"
        )

        assert jnp.isclose(f_int_rot[-1, 1], -load, atol=1e-4), (
            f"Internal shear_y force at loaded end incorrect, expected {-load}, got {f_int_rot[-1, 1]}"
        )

        assert jnp.isclose(f_int_rot[0, 5], -expected_moment, atol=1e-3), (
            f"Internal moment at fixed end incorrect, expected {-expected_moment}, got {f_int_rot[0, 5]}",
        )

        assert jnp.isclose(f_int_rot[-1, 5], -expected_moment, atol=1e-3), (
            f"Internal moment at loaded end incorrect, expected {-expected_moment}, got {f_int_rot[-1, 5]}"
        )

    def test_solve_shear_z_load(self, direction_index, y_vect_col, length):
        r"""
        Ensure z-shear load case is solved correctly.
        Note - this should not be expected to converge for z-bending moment, and as such has weaker tolerances.
        """
        struct, coords = _beam(direction_index, y_vect_col, length)

        k_coeffs = jnp.full(6, 1e5).at[2].set(4.56)
        k_coeffs = k_coeffs.at[0].set(1e2)  # allow the beam to stretch
        struct.set_design_variables(coords, jnp.diag(k_coeffs)[None, :], None)
        load = 1.23
        f_ext = (
            jnp.zeros((N_NODES, 6))
            .at[-1, :3]
            .set(struct.o0[0, ...] @ jnp.array([0.0, 0.0, load]))
        )
        result = struct.static_solve(
            f_ext_follower=f_ext,
            f_ext_dead=None,
            f_ext_aero=None,
            prescribed_dofs=jnp.concatenate(
                (jnp.arange(6), (N_NODES - 1) * 6 + jnp.arange(3, 6))
            ),
            load_steps=4,
        )

        expected_eps = load / k_coeffs[2]
        expected_disp = expected_eps * length
        expected_moment = 0.5 * load * length

        assert jnp.allclose(
            result.d,
            exp := jnp.array(
                (
                    length / N_ELEM,
                    0.0,
                    expected_disp / N_ELEM,
                    0.0,
                    0.0,
                    0.0,
                )
            )[None, :],
            atol=1e-5,
            rtol=1e-3,
        ), (
            f"Incorrect configuration for shear_y load static solve, expected {exp}, got {result.d}"
        )

        f_int = result.f_int
        f_int_rot = jnp.einsum("ij,kj->ki", chi(struct.o0[0, ...].T), f_int)

        assert jnp.isclose(f_int_rot[0, 2], load, atol=1e-4), (
            f"Internal shear_y force at fixed end incorrect, expected {load}, got {f_int_rot[0, 2]}"
        )

        assert jnp.isclose(f_int_rot[-1, 2], -load, atol=1e-4), (
            f"Internal shear_y force at loaded end incorrect, expected {-load}, got {f_int_rot[-1, 2]}"
        )

        assert jnp.isclose(f_int_rot[0, 4], -expected_moment, atol=1e-3), (
            f"Internal moment at fixed end incorrect, expected {-expected_moment}, got {f_int_rot[0, 4]}",
        )

        assert jnp.isclose(f_int_rot[-1, 4], -expected_moment, atol=1e-3), (
            f"Internal moment at loaded end incorrect, expected {-expected_moment}, got {f_int_rot[-1, 4]}"
        )
