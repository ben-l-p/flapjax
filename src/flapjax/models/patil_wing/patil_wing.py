from jax import Array
from jax import numpy as jnp

from flapjax.aero.data_structures import GridDiscretisation
from flapjax.aero.flowfields import ConstantFlowField
from flapjax.aero.utils import add_control_surface, make_rectangular_grid
from flapjax.aero.uvlm import UVLM
from flapjax.algebra.array_utils import ArrayList
from flapjax.coupled import CoupledAeroelastic
from flapjax.structure import BeamStructure

U_INF_DEFAULT = jnp.array((30.0, 0.0, 0.0))


# noinspection SpellCheckingInspection
def generate_patil_wing(
    sigma: float = 1.0,
    n_nodes: int = 49,
    m: int = 4,
    m_star: int = 4 * 20,
    u_inf=U_INF_DEFAULT,
    rho=0.088,
) -> CoupledAeroelastic:
    # noinspection GrazieStyle
    r"""
    Create the wing from “Nonlinear Aeroelasticity and Flight Dynamics of High-Altitude Long-Endurance Aircraft”,
    Patil, M. J., Hodges, D. H., and Cesnik, C. E. S. This wing is used for comparison with "Optimal Rolling Maneuvers
    with Very Flexible Wings", Salvatore Maraniello and Rafael Palacios.
    """

    # constants
    b_ref = 16.0
    c_ref = 1.0
    ea = 0.5

    m_cs = jnp.diag(jnp.array((0.75, 0.75, 0.75, 0.1, 0.1, 0.1)))
    k_cs = jnp.diag(
        jnp.array(
            (
                1e7,
                1e7,
                1e7,
                sigma * 1e4,
                5e6,
                sigma * 2e4,
            )
        )
    )

    m_lump = 75.0 - (
        2.0 * m_cs[0, 0] * b_ref
    )  # add a lumped mass to get total mass up to 75 kg
    m_lump_arr = jnp.zeros((6, 6)).at[:3, :3].set(jnp.eye(3) * m_lump)

    # beam non-design variables
    n_elem = n_nodes - 1
    n = n_nodes - 1
    y_vector = jnp.array((0.0, 0.0, 1.0))
    conn = jnp.zeros((n_elem, 2), dtype=int)
    conn = conn.at[:, 0].set(jnp.arange(n_elem))
    conn = conn.at[:, 1].set(jnp.arange(1, n_nodes))
    beam = BeamStructure(
        num_nodes=n_nodes,
        connectivity=conn,
        y_vector=y_vector,
        gravity=jnp.array((0.0, 0.0, -9.81)),
        m_lumped_index=jnp.array(n_elem // 2, dtype=int),
    )

    # create aero grid function with control surface
    def aero_grid_func(
        x0: ArrayList,
        *,
        left_aileron: Array,
        right_aileron: Array,
    ) -> ArrayList:
        # positive is surface up for both left and right
        arr = add_control_surface(
            grid=x0[0],
            angle=left_aileron,
            m_slice=slice(int(0.75 * m), None),
            n_slice=slice(0, n // 4 + 1),
        )

        arr = add_control_surface(
            grid=arr,
            angle=right_aileron,
            m_slice=slice(int(0.75 * m), None),
            n_slice=slice(-n // 4 - 1, None),
        )

        return ArrayList([arr])

    # aero non-design variables
    gd = GridDiscretisation(m=m, n=n, m_star=m_star)
    uvlm = UVLM(
        grid_shapes=[gd],
        dof_mapping=jnp.arange(n_nodes),
        grid_func=aero_grid_func,  # type: ignore
    )

    wing = CoupledAeroelastic(beam, uvlm)

    beam_coords = (
        jnp.zeros((n_nodes, 3)).at[:, 1].set(jnp.linspace(-b_ref, b_ref, n_nodes))
    )
    grid = make_rectangular_grid(m, n, c_ref, ea)
    dt = c_ref / (jnp.linalg.norm(u_inf) * m)
    flowfield = ConstantFlowField(u_inf=u_inf, rho=rho, relative_motion=True)

    wing.set_design_variables(
        coords=beam_coords,
        k_cs=k_cs,
        m_cs=m_cs,
        m_lumped=m_lump_arr[None, ...],
        dt=dt,
        flowfield=flowfield,
        delta_w=None,
        zeta_b0=grid,
        cs_angles_reference={
            "left_aileron": jnp.zeros(()),
            "right_aileron": jnp.zeros(()),
        },
        orientation_euler=jnp.array((0.0, jnp.deg2rad(5.0), 0.0)),
    )

    return wing
