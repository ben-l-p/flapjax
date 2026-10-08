from dataclasses import dataclass

import jax
from jax import Array
from jax import numpy as jnp
from jax.scipy.spatial.transform import Rotation

from flapjax.aero.data_structures import GridDiscretisation
from flapjax.aero.flowfields import FlowField
from flapjax.aero.utils import make_rectangular_grid
from flapjax.aero.uvlm import UVLM
from flapjax.algebra.so3 import vec_to_skew
from flapjax.coupled import CoupledAeroelastic
from flapjax.structure import BeamStructure
from flapjax.utils.data_structures import ConvergenceSettings
from flapjax.utils.print_utils import warn


@dataclass
class PazyParameters:
    n_keypoints: int
    coords: Array
    mass: Array
    cg_x: Array
    cg_y: Array
    cg_z: Array
    i_xx: Array
    i_yy: Array
    i_zz: Array
    i_xy: Array
    i_xz: Array
    i_yz: Array
    k11: Array
    k22: Array
    k33: Array
    k44: Array
    k12: Array
    k13: Array
    k14: Array
    k23: Array
    k24: Array
    k34: Array
    inertia_at_axis: bool = False


def generate_generic_pazy_wing(
    data: PazyParameters,
    m: int,
    m_star: int,
    node_multiplier: int,
    gravity: bool | Array,
    flowfield: FlowField,
    aoa: float | Array,
    sweep: float | Array | None,
    variable_disc_wake: bool,
    custom_dt: float | None = None,
    lumped_mass: bool = False,
    y_vector_override: Array | None = None,
) -> CoupledAeroelastic:
    # keypoint 3D coordinates at which cross-section properties are defined
    coords_data = jnp.array(data.coords)  # [N_KEYPOINT, 3]

    # arclength along the beam at each keypoint, used as the interpolation parameter
    seg_lengths_data = jnp.linalg.norm(jnp.diff(coords_data, axis=0), axis=1)
    s_data = jnp.concatenate([jnp.zeros(1), jnp.cumsum(seg_lengths_data)])

    # coordinates we wish to use, for which we will interpolate properties
    coords = coords_data
    if node_multiplier > 1:
        # subdivide each keypoint segment into node_multiplier equal parts in 3D
        frac = jnp.linspace(0.0, 1.0, node_multiplier + 1)[:-1]
        sub_nodes = (
            coords[:-1, None, :]
            + (coords[1:, None, :] - coords[:-1, None, :]) * frac[None, :, None]
        )
        coords = jnp.concatenate([sub_nodes.reshape(-1, 3), coords[-1:]], axis=0)
    n_nodes = coords.shape[0]
    n_elem = n_nodes - 1

    # arclength midpoints of the flapjax elements, used for property interpolation
    seg_lengths = jnp.linalg.norm(jnp.diff(coords, axis=0), axis=1)
    s_arr = jnp.concatenate([jnp.zeros(1), jnp.cumsum(seg_lengths)])
    elem_midpoints = 0.5 * (s_arr[:-1] + s_arr[1:])

    # mass/inertia data is provided per keypoint (one entry per node)
    n_mass = data.mass.shape[0]
    if n_mass != data.n_keypoints:
        raise ValueError(
            f"data.mass has length {n_mass}; expected {data.n_keypoints} (per-node)"
        )

    cg_data = jnp.stack((data.cg_y, -data.cg_x, data.cg_z), axis=1)  # [n_mass, 3]

    # inertia tensor at CG, in beam-local frame
    j_cg_data = jnp.zeros((n_mass, 3, 3))

    idx_keys = (
        (0, 0, "i_yy"),
        (1, 1, "i_xx"),
        (2, 2, "i_zz"),
        (0, 1, "i_xy"),
        (1, 2, "i_xz"),
        (0, 2, "i_yz"),
    )
    for i, j, key in idx_keys:
        j_cg_data = j_cg_data.at[:, i, j].set(getattr(data, key))
        j_cg_data = j_cg_data.at[:, j, i].set(getattr(data, key))

    # parallel-axis shift inertia from CG to the beam axis, at each keypoint
    skew_cg_data = jax.vmap(vec_to_skew)(cg_data)  # [n_mass, 3, 3]
    if (
        data.inertia_at_axis
    ):  # different Pazy data sets provide inertia either at the CG or at the beam axis
        j_axis_data = j_cg_data
    else:
        j_axis_data = j_cg_data - data.mass[:, None, None] * (
            skew_cg_data @ skew_cg_data
        )

    if lumped_mass:
        if node_multiplier != 1:
            warn(
                "lumped_mass=True with a node multiplier > 1 leads to nodes with zero mass"
            )

        # build 6x6 mass matrix at each point
        cg_data_ref = jnp.stack((data.cg_x, data.cg_y, data.cg_z), axis=1)
        j_inertia_ref = jnp.zeros((n_mass, 3, 3))
        for i, j, key in (
            # different ordering from above
            (0, 0, "i_xx"),
            (1, 1, "i_yy"),
            (2, 2, "i_zz"),
            (0, 1, "i_xy"),
            (0, 2, "i_xz"),
            (1, 2, "i_yz"),
        ):
            j_inertia_ref = j_inertia_ref.at[:, i, j].set(getattr(data, key))
            j_inertia_ref = j_inertia_ref.at[:, j, i].set(getattr(data, key))
        skew_cg_ref = jax.vmap(vec_to_skew)(cg_data_ref)

        if data.inertia_at_axis:
            j_axis_ref = j_inertia_ref
        else:
            j_axis_ref = j_inertia_ref - data.mass[:, None, None] * (
                skew_cg_ref @ skew_cg_ref
            )

        m_arr = data.mass[:, None, None]
        m_lumped_arr = jnp.zeros((n_mass, 6, 6))
        m_lumped_arr = m_lumped_arr.at[:, :3, :3].set(m_arr * jnp.eye(3))
        m_lumped_arr = m_lumped_arr.at[:, :3, 3:].set(-m_arr * skew_cg_ref)
        m_lumped_arr = m_lumped_arr.at[:, 3:, :3].set(m_arr * skew_cg_ref)
        m_lumped_arr = m_lumped_arr.at[:, 3:, 3:].set(j_axis_ref)

        # keypoints sit at every node_multiplier-th node of the refined mesh
        m_lumped_index = jnp.arange(n_mass) * node_multiplier

        m_cs = jnp.zeros((n_elem, 6, 6))
    else:
        # split each nodal lump across its adjacent half-elements
        lengths_data = jnp.concatenate(
            (
                0.5 * seg_lengths_data[:1],
                0.5 * (seg_lengths_data[:-1] + seg_lengths_data[1:]),
                0.5 * seg_lengths_data[-1:],
            )
        )  # [N_KEYPOINT]
        data_interp_coords = s_data

        m_bar_data = data.mass / lengths_data  # mass per unit length
        j_bar_data = (
            j_axis_data / lengths_data[:, None, None]
        )  # inertia per unit length

        m_bar = jnp.interp(elem_midpoints, data_interp_coords, m_bar_data)
        cg_elem = jnp.stack(
            [
                jnp.interp(elem_midpoints, data_interp_coords, cg_data[:, i])
                for i in range(3)
            ],
            axis=1,
        )
        j_bar = jnp.stack(
            [
                jnp.stack(
                    [
                        jnp.interp(
                            elem_midpoints, data_interp_coords, j_bar_data[:, i, k]
                        )
                        for k in range(3)
                    ],
                    axis=1,
                )
                for i in range(3)
            ],
            axis=1,
        )  # [n_elem, 3, 3]

        skew_cg_elem = jax.vmap(vec_to_skew)(cg_elem)
        m_bar_ = m_bar[:, None, None]
        m_cs = jnp.zeros((n_elem, 6, 6))
        m_cs = m_cs.at[:, :3, :3].set(m_bar_ * jnp.eye(3))
        m_cs = m_cs.at[:, :3, 3:].set(-m_bar_ * skew_cg_elem)
        m_cs = m_cs.at[:, 3:, :3].set(m_bar_ * skew_cg_elem)
        m_cs = m_cs.at[:, 3:, 3:].set(j_bar)

        m_lumped_arr = None
        m_lumped_index = None

    # stiffness properties, provided for non-shear deformation
    k_cs_data = jnp.zeros((data.n_keypoints - 1, 6, 6))

    idx_keys = (
        (0, 0, "k11"),
        (3, 3, "k22"),
        (4, 4, "k33"),
        (5, 5, "k44"),
        (0, 3, "k12"),
        (0, 4, "k13"),
        (0, 5, "k14"),
        (3, 4, "k23"),
        (3, 5, "k24"),
        (4, 5, "k34"),
    )
    for i, j, key in idx_keys:
        # redundant for diagonal terms
        k_cs_data = k_cs_data.at[:, i, j].set(getattr(data, key))
        k_cs_data = k_cs_data.at[:, j, i].set(getattr(data, key))

    # add large shear stiffness to diagonal terms to avoid singularity
    k_cs_data = k_cs_data.at[:, 1, 1].set(1e9)
    k_cs_data = k_cs_data.at[:, 2, 2].set(1e9)
    data_elem_midpoints = 0.5 * (
        s_data[:-1] + s_data[1:]
    )  # arclength midpoints of the data segments, used for interpolation

    # interpolate each stiffness component at the new element midpoints;
    k_cs = interp_cs_property(
        data=k_cs_data, data_coords=data_elem_midpoints, coords=elem_midpoints
    )

    # aerodynamic model
    c_ref = 0.1
    ea = 0.441
    aero_grid = make_rectangular_grid(m=m, n=n_nodes - 1, chord=c_ref, ea=ea)

    # make wing
    conn = jnp.stack(
        [jnp.arange(n_nodes - 1), jnp.arange(1, n_nodes)], axis=1
    )  # [n_elem, 2]
    if y_vector_override is not None:
        y_vector = jnp.broadcast_to(y_vector_override[None, :], (n_elem, 3))
    else:
        y_vector = jnp.zeros((n_elem, 3))  # [n_elem, 3]
        y_vector = y_vector.at[:, 0].set(-1.0)

    if custom_dt is None:
        dt = c_ref / (flowfield.u_inf_mag * m)  # time step based on CFL condition
    else:
        dt = custom_dt

    if isinstance(gravity, Array):
        gravity_ = gravity
    elif gravity:
        gravity_ = jnp.array((0.0, -9.81, 0.0))
    else:
        gravity_ = None

    if sweep is not None:
        # apply sweep as a postprocessing step to rotate the coordinates
        rmat = Rotation.from_euler("zyx", jnp.array((-sweep, 0.0, 0.0))).as_matrix()
        sweep_coords = jnp.einsum("ij, kj->ki", rmat, coords)
    else:
        sweep_coords = coords

    structure = BeamStructure(
        num_nodes=n_nodes,
        connectivity=conn,
        y_vector=y_vector,
        k_cs_index=jnp.arange(n_elem),
        m_cs_index=jnp.arange(n_elem),
        m_lumped_index=m_lumped_index,
        gravity=gravity_,
    )
    structure.struct_convergence_settings = ConvergenceSettings(
        max_n_iter=100,
        rel_disp_tol=1e-8,
        abs_disp_tol=1e-8,
        rel_force_tol=1e-5,
        abs_force_tol=1e-3,
    )

    aero = UVLM(
        grid_shapes=(GridDiscretisation(m=m, n=n_nodes - 1, m_star=m_star),),
        dof_mapping=jnp.arange(n_nodes),
        mirror_point=jnp.zeros(3),
        mirror_normal=jnp.array((0.0, 1.0, 0.0)),
    )
    wing = CoupledAeroelastic(structure=structure, aero=aero)
    wing.fsi_convergence_settings = ConvergenceSettings(
        max_n_iter=40,
        rel_disp_tol=1e-3,
        abs_disp_tol=1e-3,
        rel_force_tol=1e-2,
        abs_force_tol=1e-2,
    )

    if variable_disc_wake:
        m_base = 3 * m
        wake_delta = jnp.ones(m_star)
        wake_delta = wake_delta.at[m_base:].set(jnp.logspace(0.0, 1.3, m_star - m_base))
    else:
        wake_delta = None

    wing.set_design_variables(
        coords=sweep_coords,
        k_cs=k_cs,
        m_cs=m_cs,
        m_lumped=m_lumped_arr,
        dt=dt,
        flowfield=flowfield,
        zeta_b0=aero_grid,
        delta_w=None
        if wake_delta is None
        else wake_delta
        * dt
        * flowfield.u_inf_mag,  # all ones for base wake, increase values for variable disc.
        orientation_euler=jnp.array((0.0, aoa, 0.0)),
    )

    return wing


def interp_cs_property(data: Array, data_coords: Array, coords: Array) -> Array:
    r"""
    Helper function for interpolating cross-sectional structural properties along a beam.
    """
    return jnp.stack(
        [
            jnp.stack(
                [jnp.interp(coords, data_coords, data[:, i, j]) for j in range(6)],
                axis=1,
            )
            for i in range(6)
        ],
        axis=1,
    )
