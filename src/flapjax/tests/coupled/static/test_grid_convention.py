from __future__ import annotations

import jax.numpy as jnp

from flapjax.aero.data_structures import GridDiscretisation
from flapjax.aero.flowfields import ConstantFlowField
from flapjax.aero.utils import make_rectangular_grid
from flapjax.aero.uvlm import UVLM
from flapjax.coupled import CoupledAeroelastic
from flapjax.models.cantilever_wing.cantilever_wing import (
    K_CS_DEFAULT,
    M_CS_DEFAULT,
    U_INF_DEFAULT,
)
from flapjax.structure import BeamStructure

N_NODES = 6
M = 4
C_REF = 1.0


def _build_wing(transpose_grid: bool) -> CoupledAeroelastic:
    """The same cantilever wing built either with the standard
    (n = spanwise, beam-mapped) grid convention, or the transposed (m = spanwise, beam-mapped via
    beam_m). This transposed-type grid is to be used for fuselage aerodynamics where we wish to swap the grid to beam
    mapping axis.
    """

    n = N_NODES - 1
    n_elem = n

    y_vector = jnp.array((0.0, 0.0, 1.0))
    conn = jnp.zeros((n_elem, 2), dtype=int)
    conn = conn.at[:, 0].set(jnp.arange(n_elem))
    conn = conn.at[:, 1].set(jnp.arange(1, N_NODES))
    beam = BeamStructure(num_nodes=N_NODES, connectivity=conn, y_vector=y_vector)

    grid = make_rectangular_grid(M, n, C_REF, 0.25)
    if transpose_grid:
        gd = GridDiscretisation(m=n, n=M, m_star=0, beam_m=True)
        grid = jnp.swapaxes(grid, 0, 1)
    else:
        gd = GridDiscretisation(m=M, n=n, m_star=0)

    uvlm = UVLM(grid_shapes=[gd], dof_mapping=jnp.arange(N_NODES))
    wing = CoupledAeroelastic(beam, uvlm)

    beam_coords = jnp.zeros((N_NODES, 3)).at[:, 1].set(jnp.linspace(0, 5.0, N_NODES))
    dt = C_REF / (jnp.linalg.norm(U_INF_DEFAULT) * M)
    flowfield = ConstantFlowField(u_inf=U_INF_DEFAULT, rho=1.225, relative_motion=True)

    wing.set_design_variables(
        coords=beam_coords,
        k_cs=K_CS_DEFAULT,
        m_cs=M_CS_DEFAULT,
        m_lumped=None,
        dt=dt,
        flowfield=flowfield,
        delta_w=None,
        x0_aero=grid,
    )
    return wing


def _solve(transpose_grid: bool):
    wing = _build_wing(transpose_grid)
    # wing.structure.struct_convergence_settings = _TIGHT_CONVERGENCE
    # wing.fsi_convergence_settings = _TIGHT_CONVERGENCE
    return wing.static_solve(
        f_ext_dead=None,
        f_ext_follower=None,
        prescribed_dofs=tuple(range(6)),
        horseshoe=False,
    )


class TestGridConvention:
    """A wing built with the standard (n = spanwise, beam-mapped) grid convention and with the
    transposed (m = spanwise, beam-mapped via beam_m) convention should reach the same static
    aeroelastic equilibrium, since both describe the same physical wing under the same loading.
    """

    @classmethod
    def setup_class(cls):
        cls.sol_standard = _solve(transpose_grid=False)
        cls.sol_transposed = _solve(transpose_grid=True)

    def test_same_node_positions(self):
        if not jnp.allclose(
            self.sol_standard.structure.x,
            self.sol_transposed.structure.x,
            atol=1e-6,
            rtol=1e-5,
        ):
            raise ValueError(
                "Node positions differ between the standard and transposed grid conventions:\n"
                f"standard:\n{self.sol_standard.structure.x}\n"
                f"transposed:\n{self.sol_transposed.structure.x}"
            )

    def test_same_node_orientations(self):
        if not jnp.allclose(
            self.sol_standard.structure.rmat,
            self.sol_transposed.structure.rmat,
            atol=1e-6,
            rtol=1e-5,
        ):
            raise ValueError(
                "Node orientations differ between the standard and transposed grid conventions."
            )

    def test_same_total_force(self):
        f_standard = jnp.sum(self.sol_standard.aero.f_steady[0], axis=(0, 1))
        f_transposed = jnp.sum(self.sol_transposed.aero.f_steady[0], axis=(0, 1))
        if not jnp.allclose(f_standard, f_transposed, atol=1e-6, rtol=1e-5):
            raise ValueError(
                f"Total aerodynamic force differs: standard={f_standard}, transposed={f_transposed}"
            )
