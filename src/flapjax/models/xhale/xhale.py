from __future__ import annotations

from typing import Sequence

from jax import Array
from jax import numpy as jnp

from flapjax.aero.data_structures import GridDiscretisation
from flapjax.aero.flowfields import ConstantFlowField, FlowField
from flapjax.aero.utils import add_control_surface, make_rectangular_grid
from flapjax.aero.uvlm import UVLM
from flapjax.algebra.array_utils import ArrayList
from flapjax.algebra.so3 import exp_so3, vec_to_skew
from flapjax.coupled import CoupledAeroelastic
from flapjax.structure import BeamStructure

r"""
X-HALE aircraft model, ported version from SHARPy's ``sharpy_cases/XHALE/generate_xhale.py``
"""

FLOWFIELD_DEFAULT = ConstantFlowField(
    rho=1.225,
    u_inf=jnp.array((14.0, 0.0, 0.0)),
    relative_motion=True,
)

X_AXIS = jnp.array((1.0, 0.0, 0.0))
Y_AXIS = jnp.array((0.0, 1.0, 0.0))
Z_AXIS = jnp.array((0.0, 0.0, 1.0))

# component name: (mass per unit length, cg_y, cg_z, ixx, iyy, izz, iyz)
MASS_DATA: dict[str, tuple[float, ...]] = {
    "Linboard": (0.394, -0.0294, 0.0, 0.000809, 0.000012, 0.000797, 0.000006),
    "Loutboard": (0.394, -0.0294, 0.0, 0.000809, 0.000012, 0.000797, 0.000006),
    "Ldihedral": (0.5, -0.0214, 0.0, 0.000809, 0.000012, 0.000797, 0.000006),
    "Rinboard": (0.394, 0.0294, 0.0, 0.000809, 0.000012, 0.000797, 0.000006),
    "Routboard": (0.394, 0.0294, 0.0, 0.000809, 0.000012, 0.000797, 0.000006),
    "Rdihedral": (0.5, 0.0214, 0.0, 0.000809, 0.000012, 0.000797, 0.000006),
    "boom": (0.0429, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "tailL": (0.2614, -0.0144, 0.0, 0.00016, 0.000003, 0.000157, 0.0),
    "tailR": (0.2614, 0.0144, 0.0, 0.00016, 0.000003, 0.000157, 0.0),
    "Cfin": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "Lfin": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "Rfin": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "LLfin": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "RRfin": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    "Cvfin": (0.5092, 0.0, 0.0, 0.003187, 0.000093, 0.00328, 0.0),
    "Lvfin": (0.3208, 0.0, 0.0, 0.000817, 0.000059, 0.000876, 0.0),
    "Rvfin": (0.3208, 0.0, 0.0, 0.000817, 0.000059, 0.000876, 0.0),
}
MASS_NAMES: tuple[str, ...] = tuple(MASS_DATA.keys())
M_INDEX: dict[str, int] = {name: i for i, name in enumerate(MASS_NAMES)}

# component name: (ea, gay, gaz, gj, eiy, eiz, k13, k14, k34)
STIFFNESS_DATA: dict[str, tuple[float, ...]] = {
    "Linboard": (2.14e6, 2.14e7, 2.14e7, 59.3, 112.0, 6350.0, 0.0, 0.0, -46.3),
    "Loutboard": (2.14e6, 2.14e7, 2.14e7, 59.3, 112.0, 6350.0, 0.0, 0.0, -46.3),
    "Ldihedral": (2.14e6, 2.14e7, 2.14e7, 59.3, 112.0, 6350.0, 0.0, 0.0, -46.3),
    "Rinboard": (2.14e6, 2.14e7, 2.14e7, 59.3, 112.0, 6350.0, 0.0, 0.0, 46.3),
    "Routboard": (2.14e6, 2.14e7, 2.14e7, 59.3, 112.0, 6350.0, 0.0, 0.0, 46.3),
    "Rdihedral": (2.14e6, 2.14e7, 2.14e7, 59.3, 112.0, 6350.0, 0.0, 0.0, 46.3),
    "boom": (5.39e7, 5.39e8, 5.39e8, 5.39e7, 5.39e7, 5.39e7, 0.0, 0.0, 0.0),
    "tailL": (
        3.21e6,
        3.21e7,
        3.21e7,
        21.4,
        91.0,
        4270.0,
        -0.000371,
        -74400.0,
        0.00000226,
    ),
    "tailR": (
        3.21e6,
        3.21e7,
        3.21e7,
        21.4,
        91.0,
        4270.0,
        -0.000371,
        74400.0,
        0.00000226,
    ),
    "Cfin": (5.39e7, 5.39e8, 5.39e8, 5.39e7, 5.39e7, 5.39e7, 0.0, 0.0, 0.0),
    "Lfin": (5.39e7, 5.39e8, 5.39e8, 5.39e7, 5.39e7, 5.39e7, 0.0, 0.0, 0.0),
    "Rfin": (5.39e7, 5.39e8, 5.39e8, 5.39e7, 5.39e7, 5.39e7, 0.0, 0.0, 0.0),
}
STIFFNESS_NAMES: tuple[str, ...] = tuple(STIFFNESS_DATA.keys())
K_INDEX: dict[str, int] = {name: i for i, name in enumerate(STIFFNESS_NAMES)}

# lumped point masses making up each wing pod - each pod is 3 point masses, with data in the untwisted frame
# mass, cg_x, cg_y, cg_z, ixx, ixy, ixz, iyy, iyz, izz
CENTRE_POD: tuple[tuple[float, ...], ...] = (
    (0.3746, 0.0, 0.1, 0.0, 0.00115, 0.0, 0.0, 0.00089, 0.0, 0.00089),
    (
        1.0462,
        0.003974,
        0.0612,
        -0.0168,
        0.0148,
        0.000232,
        0.000023,
        0.00282,
        0.00045,
        0.00025,
    ),
    (0.0230, 0.0, 0.259011, -0.02266, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
)
INBOARD_POD: tuple[tuple[float, ...], ...] = (
    (0.548, -0.010000, 0.090410, 0.0, 0.00154, 0.0, 0.0, 0.00089, 0.0, 0.00089),
    (
        0.929,
        0.002138,
        0.040000,
        -0.013853,
        0.01130,
        -0.00121,
        0.000011,
        0.00321,
        0.000046,
        0.00848,
    ),
    (0.023, 0.0, 0.259011, -0.022660, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
)
OUTBOARD_POD: tuple[tuple[float, ...], ...] = (
    (0.571, -0.010000, 0.091000, 0.0, 0.00154, 0.0, 0.0, 0.00089, 0.0, 0.00089),
    (
        0.929,
        0.002138,
        0.040000,
        -0.013853,
        0.01130,
        -0.00121,
        0.000011,
        0.00321,
        0.000046,
        0.00848,
    ),
    (0.023, 0.0, 0.259011, -0.022660, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
)

# Aerodynamic chord and elastic axis location
AERO_WING = (0.2, 0.288)
AERO_TAIL = (0.11, 0.3235)
AERO_FIN = (0.38, 0.6093)
AERO_CVFIN = (0.78, 1.2256)
AERO_LRVFIN = (0.43, 1.1395)

# Main-wing mean camber line for EMX-07 section, given as (x/c, z/c)
EMX07_CAMBER_X = (
    0.0, 3.527358e-07, 2.736395e-03, 1.092258e-02, 2.445874e-02, 4.321506e-02,
    6.697159e-02, 9.546839e-02, 1.284056e-01, 1.654034e-01, 2.060818e-01,
    2.499709e-01, 2.966008e-01, 3.454613e-01, 3.960123e-01, 4.477137e-01,
    4.999755e-01, 5.522375e-01, 6.039397e-01, 6.544919e-01, 7.033541e-01,
    7.499865e-01, 7.938786e-01, 8.345606e-01, 8.715624e-01, 9.045041e-01,
    9.330056e-01, 9.567668e-01, 9.755279e-01, 9.890686e-01, 9.972590e-01,
    9.999992e-01, 1.0,
)
EMX07_CAMBER_Z = (
    0.0, 5.961561e-04, 1.136788e-03, 4.241672e-03, 7.896996e-03, 1.164179e-02,
    1.542114e-02, 1.902119e-02, 2.208612e-02, 2.422109e-02, 2.527606e-02,
    2.526599e-02, 2.410581e-02, 2.209060e-02, 1.946039e-02, 1.645021e-02,
    1.325003e-02, 1.003985e-02, 6.979703e-03, 4.224549e-03, 1.969342e-03,
    3.292383e-04, -8.407012e-04, -1.650622e-03, -2.095578e-03, -2.160548e-03,
    -1.880469e-03, -1.405372e-03, -8.752758e-04, -4.201971e-04, -1.101422e-04,
    5.238689e-09, 0.0,
)
EMX07_CAMBER = (jnp.array(EMX07_CAMBER_X), jnp.array(EMX07_CAMBER_Z))


MIRROR_Y = jnp.diag(jnp.array((1.0, -1.0, 1.0)))


def _mass_matrix(mass: Array | float, xcg: Array, inertia: Array) -> Array:
    r"""
    Build a 6x6 rigid-body mass matrix from a mass, cg offset and inertia tensor
    """
    m_chi_cg = mass * vec_to_skew(xcg)
    return jnp.block(
        [
            [mass * jnp.eye(3), -m_chi_cg],
            [m_chi_cg, inertia],
        ]
    )


def _cross_section_mass(name: str) -> Array:
    m, ycg, zcg, ixx, iyy, izz, iyz = MASS_DATA[name]
    xcg = jnp.array((0.0, ycg, zcg))
    inertia = jnp.array(((ixx, 0.0, 0.0), (0.0, iyy, iyz), (0.0, iyz, izz)))
    return _mass_matrix(m, xcg, inertia)


def _cross_section_stiffness(name: str, ga_mult: float, sigma: float) -> Array:
    ea, gay, gaz, gj, eiy, eiz, k13, k14, k34 = STIFFNESS_DATA[name]
    k = jnp.diag(jnp.array((ea, gay * ga_mult, gaz * ga_mult, gj, eiy, eiz)))
    k = k.at[0, 4].set(k13).at[4, 0].set(k13)
    k = k.at[0, 5].set(k14).at[5, 0].set(k14)
    k = k.at[4, 5].set(k34).at[5, 4].set(k34)
    return sigma * k


def _wing_root_triad(twist: Array | float) -> Array:
    r"""
    Local material triad for the (twisted) right-wing-root cross-section.
    """
    tangent = Y_AXIS
    normal = jnp.cross(tangent, -X_AXIS)
    normal /= jnp.linalg.norm(normal)
    binormal = -jnp.cross(tangent, normal)
    r = exp_so3(tangent * twist)
    return jnp.stack((tangent, r @ binormal, r @ normal), axis=1)


def _pod_lumped_masses(
    pod_data: tuple[tuple[float, ...], ...], triad: Array, mirror: bool
) -> Array:
    matrices = []
    for mass, xcg, ycg, zcg, ixx, ixy, ixz, iyy, iyz, izz in pod_data:
        p_local = jnp.array((xcg, ycg, zcg))
        inertia_local = jnp.array(((ixx, ixy, ixz), (ixy, iyy, iyz), (ixz, iyz, izz)))
        p_global = triad @ p_local
        inertia_global = triad @ inertia_local @ triad.T
        if mirror:
            p_global = MIRROR_Y @ p_global
            inertia_global = MIRROR_Y @ inertia_global @ MIRROR_Y
        matrices.append(_mass_matrix(mass, p_global, inertia_global))
    return jnp.stack(matrices, axis=0)


def generate_xhale(
    sigma: float = 1.0,
    ga_mult: float = 0.1,
    flowfield: FlowField = FLOWFIELD_DEFAULT,
    m_wing: int = 8,
    m_tail: int = 3,
    m_fin: int = 4,
    n_wing_section: int = 4,
    n_wing_dihedral: int = 8,
    n_boom_centre: int = 2,
    n_boom_outer: int = 2,
    n_tail: int = 2,
    n_fin: int = 2,
    n_vfin: int = 2,
    m_star: int = 160,
    alpha: Array | float = 0.0,
    roll: Array | float = 0.0,
    beta: Array | float = 0.0,
    elevator_angle: Array | float = 0.0,
    rudder_angle: Array | float = 0.0,
    left_aileron_angle: Array | float = 0.0,
    right_aileron_angle: Array | float = 0.0,
    gravity: Sequence[float] | Array = (0.0, 0.0, -9.81),
    relaxation_factor: float = 0.7,
    spectral_radius: float = 0.5,
    gamma_dot_relaxation: float = 0.5,
) -> CoupledAeroelastic:
    r"""
    Generate the X-HALE aircraft.

    :param sigma: Stiffness multiplier applied to all cross-sectional stiffness matrices.
    :param ga_mult: Shear stiffness multiplier.
    :param flowfield: FlowField object.
    :param m_wing: Number of chordwise aerodynamic panels for the main wing.
    :param m_tail: Number of chordwise aerodynamic panels for the tail surfaces.
    :param m_fin: Number of chordwise aerodynamic panels for the fin surfaces.
    :param n_wing_section: Number of beam elements per straight (inboard/outboard) wing section.
    :param n_wing_dihedral: Number of beam elements in the dihedral wing tip section.
    :param n_boom_centre: Number of beam elements in the centre tail boom.
    :param n_boom_outer: Number of beam elements in each outer tail booms.
    :param n_tail: Number of beam elements per tail half (up/down or left/right).
    :param n_fin: Number of beam elements per wing-mounted pod fin.
    :param n_vfin: Number of beam elements per tail-boom-mounted vertical fin.
    :param m_star: Number of trailing wake panels.
    :param alpha: Reference angle of attack.
    :param roll: Reference roll angle.
    :param beta: Reference sideslip angle.
    :param elevator_angle: Reference deflection angle applied to the 4 tails.
    :param rudder_angle: Reference deflection rudder angle.
    :param left_aileron_angle: Reference deflection angle for the left wingtip aileron.
    :param right_aileron_angle: Reference deflection angle for the right wingtip aileron.
    :param gravity: Gravity vector.
    :param relaxation_factor: Structural relaxation factor.
    :param spectral_radius: Spectral radius for the time integrator.
    :param gamma_dot_relaxation: Filtering parameter for the aerodynamic circulation time derivative.
    :return: X-HALE aircraft object.
    """
    # geometric parameters
    span_section = 1.0
    dihedral = 10.0 / 180.0 * jnp.pi
    length_centre_tail = 1.106
    length_outer_tail = 0.65
    span_tail = 0.24
    span_ctail_up = 0.24
    span_ctail_down = 0.145
    span_fin = 0.184
    span_vfin = 0.15
    twist = 5.0 / 180.0 * jnp.pi

    dt: float = AERO_WING[0] / (m_wing * float(jnp.linalg.norm(flowfield.u_inf)))

    # create structure
    node_coords: list[Array] = [jnp.zeros(3)]
    node_index: dict[str, int] = {"root": 0}
    branch_nodes: dict[str, list[int]] = {}
    conn: list[tuple[int, int]] = []
    y_vectors: list[Array] = []
    k_indices: list[int] = []
    m_indices: list[int] = []

    def add_branch(
        name: str,
        parent: str,
        offset: Array,
        n_elem: int,
        y_vector_raw: Array,
        twist_angle: Array | float,
        k_name: str,
        m_name: str,
    ) -> int:
        start = node_index[parent]
        start_coord = node_coords[start]
        tangent = offset / jnp.linalg.norm(offset)
        y_vector_ = exp_so3(tangent * twist_angle) @ y_vector_raw
        points = jnp.linspace(start_coord, start_coord + offset, n_elem + 1)[1:]
        chain = [start]
        prev = start
        for point in points:
            idx = len(node_coords)
            node_coords.append(point)
            conn.append((prev, idx))
            y_vectors.append(y_vector_)
            k_indices.append(K_INDEX[k_name])
            m_indices.append(M_INDEX[m_name])
            chain.append(idx)
            prev = idx
        branch_nodes[name] = chain
        node_index[name] = prev
        return prev

    # main wing: inboard, outboard and dihedral tip, both sides
    add_branch(
        "R0",
        "root",
        span_section * Y_AXIS,
        n_wing_section,
        -X_AXIS,
        twist,
        "Rinboard",
        "Rinboard",
    )
    add_branch(
        "R1",
        "R0",
        span_section * Y_AXIS,
        n_wing_section,
        -X_AXIS,
        twist,
        "Routboard",
        "Routboard",
    )
    add_branch(
        "R2",
        "R1",
        span_section * (jnp.cos(dihedral) * Y_AXIS + jnp.sin(dihedral) * Z_AXIS),
        n_wing_dihedral,
        -X_AXIS,
        twist,
        "Rdihedral",
        "Rdihedral",
    )
    add_branch(
        "L0",
        "root",
        -span_section * Y_AXIS,
        n_wing_section,
        X_AXIS,
        -twist,
        "Linboard",
        "Linboard",
    )
    add_branch(
        "L1",
        "L0",
        -span_section * Y_AXIS,
        n_wing_section,
        X_AXIS,
        -twist,
        "Loutboard",
        "Loutboard",
    )
    add_branch(
        "L2",
        "L1",
        span_section * (-jnp.cos(dihedral) * Y_AXIS + jnp.sin(dihedral) * Z_AXIS),
        n_wing_dihedral,
        X_AXIS,
        -twist,
        "Ldihedral",
        "Ldihedral",
    )

    # centreline: tail boom, ctail, ventral fin, tail vertical fin
    add_branch(
        "ctail_root",
        "root",
        length_centre_tail * X_AXIS,
        n_boom_centre,
        Y_AXIS,
        0.0,
        "boom",
        "boom",
    )
    add_branch(
        "ctail_up",
        "ctail_root",
        span_ctail_up * Z_AXIS,
        n_tail,
        -X_AXIS,
        0.0,
        "tailR",
        "tailR",
    )
    add_branch(
        "ctail_down",
        "ctail_root",
        -span_ctail_down * Z_AXIS,
        n_tail,
        X_AXIS,
        0.0,
        "tailL",
        "tailL",
    )
    add_branch(
        "cvfin",
        "ctail_root",
        -span_vfin * Z_AXIS,
        n_vfin,
        -X_AXIS,
        0.0,
        "Cfin",
        "Cvfin",
    )
    add_branch("cfin", "root", -span_fin * Z_AXIS, n_fin, -X_AXIS, 0.0, "Cfin", "Cfin")

    # 4 outer tail booms
    outer_specs = (
        ("R0", "Rfin", "Rfin"),
        ("R1", "Rfin", "RRfin"),
        ("L0", "Lfin", "Lfin"),
        ("L1", "Lfin", "LLfin"),
    )
    for wing_node, fin_k, fin_m in outer_specs:
        boom_name = f"{wing_node}_boom"
        add_branch(
            boom_name,
            wing_node,
            length_outer_tail * X_AXIS,
            n_boom_outer,
            Y_AXIS,
            0.0,
            "boom",
            "boom",
        )
        add_branch(
            f"{wing_node}_tail_up",
            boom_name,
            span_tail * Y_AXIS,
            n_tail,
            -X_AXIS,
            0.0,
            "tailR",
            "tailR",
        )
        add_branch(
            f"{wing_node}_tail_down",
            boom_name,
            -span_tail * Y_AXIS,
            n_tail,
            X_AXIS,
            0.0,
            "tailL",
            "tailL",
        )
        add_branch(
            f"{wing_node}_fin",
            wing_node,
            -span_fin * Z_AXIS,
            n_fin,
            -X_AXIS,
            0.0,
            fin_k,
            fin_m,
        )
        if wing_node in ("R0", "L0"):
            vfin_k = "Rfin" if wing_node == "R0" else "Lfin"
            vfin_m = "Rvfin" if wing_node == "R0" else "Lvfin"
            add_branch(
                f"{wing_node}_vfin",
                boom_name,
                -span_vfin * Z_AXIS,
                n_vfin,
                -X_AXIS,
                0.0,
                vfin_k,
                vfin_m,
            )

    coords = jnp.stack(node_coords, axis=0)
    connectivity = jnp.array(conn, dtype=int)
    y_vector = jnp.stack(y_vectors, axis=0)
    k_cs_index = jnp.array(k_indices, dtype=int)
    m_cs_index = jnp.array(m_indices, dtype=int)

    k_cs = jnp.stack(
        [_cross_section_stiffness(name, ga_mult, sigma) for name in STIFFNESS_NAMES],
        axis=0,
    )
    m_cs = jnp.stack([_cross_section_mass(name) for name in MASS_NAMES], axis=0)

    # lumped pod masses
    triad = _wing_root_triad(twist)
    m_lumped = jnp.concatenate(
        [
            _pod_lumped_masses(CENTRE_POD, triad, mirror=False),
            _pod_lumped_masses(INBOARD_POD, triad, mirror=False),
            _pod_lumped_masses(OUTBOARD_POD, triad, mirror=False),
            _pod_lumped_masses(INBOARD_POD, triad, mirror=True),
            _pod_lumped_masses(OUTBOARD_POD, triad, mirror=True),
        ],
        axis=0,
    )
    m_lumped_index = jnp.array(
        [node_index["root"]] * len(CENTRE_POD)
        + [node_index["R0"]] * len(INBOARD_POD)
        + [node_index["R1"]] * len(OUTBOARD_POD)
        + [node_index["L0"]] * len(INBOARD_POD)
        + [node_index["L1"]] * len(OUTBOARD_POD),
        dtype=int,
    )

    # thrust on each pod
    thrust_nodes = {
        "thrust_centre": node_index["root"],
        "thrust_right_inboard": node_index["R0"],
        "thrust_right_outboard": node_index["R1"],
        "thrust_left_inboard": node_index["L0"],
        "thrust_left_outboard": node_index["L1"],
    }
    thrust_direction = {k: -X_AXIS for k in thrust_nodes}

    structure = BeamStructure(
        num_nodes=coords.shape[0],
        connectivity=connectivity,
        y_vector=y_vector,
        k_cs_index=k_cs_index,
        m_cs_index=m_cs_index,
        m_lumped_index=m_lumped_index,
        gravity=gravity,
        thrust_nodes=thrust_nodes,
        thrust_direction=thrust_direction,
        relaxation_factor=relaxation_factor,
        spectral_radius=spectral_radius,
    )

    # aerodynamic surfaces
    def pair_mapping(down_name: str, up_name: str) -> Array:
        return jnp.array(
            branch_nodes[down_name][::-1] + branch_nodes[up_name][1:], dtype=int
        )

    wing_mapping = jnp.array(
        branch_nodes["L2"][1:][::-1]
        + branch_nodes["L1"][1:][::-1]
        + branch_nodes["L0"][::-1]
        + branch_nodes["R0"][1:]
        + branch_nodes["R1"][1:]
        + branch_nodes["R2"][1:],
        dtype=int,
    )

    surfaces: list[dict] = [
        {
            "name": "wing",
            "mapping": wing_mapping,
            "m": m_wing,
            "x0": make_rectangular_grid(
                m=m_wing,
                n=wing_mapping.shape[0] - 1,
                chord=AERO_WING[0],
                ea=AERO_WING[1],
                camber_line=EMX07_CAMBER,
                twist=twist,
            ),
        },
    ]
    ctail_mapping = pair_mapping("ctail_down", "ctail_up")
    surfaces.append(
        {
            "name": "ctail",
            "mapping": ctail_mapping,
            "m": m_tail,
            "x0": make_rectangular_grid(
                m=m_tail,
                n=ctail_mapping.shape[0] - 1,
                chord=AERO_TAIL[0],
                ea=AERO_TAIL[1],
            ),
        }
    )
    for wing_node in ("R0", "R1", "L0", "L1"):
        tail_mapping = pair_mapping(f"{wing_node}_tail_down", f"{wing_node}_tail_up")
        surfaces.append(
            {
                "name": f"tail_{wing_node}",
                "mapping": tail_mapping,
                "m": m_tail,
                "x0": make_rectangular_grid(
                    m=m_tail,
                    n=tail_mapping.shape[0] - 1,
                    chord=AERO_TAIL[0],
                    ea=AERO_TAIL[1],
                ),
            }
        )

    for name in ("cfin", "R0_fin", "L0_fin"):
        fin_mapping = jnp.array(branch_nodes[name], dtype=int)
        surfaces.append(
            {
                "name": name,
                "mapping": fin_mapping,
                "m": m_fin,
                "m_star": 0,  # forward-most of an interfering pair: no wake
                "x0": make_rectangular_grid(
                    m=m_fin,
                    n=fin_mapping.shape[0] - 1,
                    chord=AERO_FIN[0],
                    ea=AERO_FIN[1],
                ),
            }
        )
    for name in ("R1_fin", "L1_fin"):
        fin_mapping = jnp.array(branch_nodes[name], dtype=int)
        surfaces.append(
            {
                "name": name,
                "mapping": fin_mapping,
                "m": m_fin,
                "x0": make_rectangular_grid(
                    m=m_fin,
                    n=fin_mapping.shape[0] - 1,
                    chord=AERO_FIN[0],
                    ea=AERO_FIN[1],
                ),
            }
        )
    cvfin_mapping = jnp.array(branch_nodes["cvfin"], dtype=int)
    surfaces.append(
        {
            "name": "cvfin",
            "mapping": cvfin_mapping,
            "m": m_fin,
            "x0": make_rectangular_grid(
                m=m_fin,
                n=cvfin_mapping.shape[0] - 1,
                chord=AERO_CVFIN[0],
                ea=AERO_CVFIN[1],
            ),
        }
    )
    for wing_node in ("R0", "L0"):
        vfin_mapping = jnp.array(branch_nodes[f"{wing_node}_vfin"], dtype=int)
        surfaces.append(
            {
                "name": f"{wing_node}_vfin",
                "mapping": vfin_mapping,
                "m": m_fin,
                "x0": make_rectangular_grid(
                    m=m_fin,
                    n=vfin_mapping.shape[0] - 1,
                    chord=AERO_LRVFIN[0],
                    ea=AERO_LRVFIN[1],
                ),
            }
        )

    surface_names = [s["name"] for s in surfaces]
    idx_wing = surface_names.index("wing")
    idx_rudder = surface_names.index("ctail")
    idx_elevator_tails = [
        surface_names.index(name)
        for name in ("tail_R0", "tail_R1", "tail_L0", "tail_L1")
    ]

    aileron_m_slice = slice(int(0.75 * m_wing), None)
    elevator_m_slice = slice(0, None)
    rudder_m_slice = slice(0, None)

    def aero_grid_func(
        x0: ArrayList,
        *,
        left_aileron: Array,
        right_aileron: Array,
        elevator: Array,
        rudder: Array,
    ) -> ArrayList:
        grids = x0.to_list()
        wing_grid = add_control_surface(
            grids[idx_wing],
            left_aileron,
            aileron_m_slice,
            slice(0, n_wing_dihedral + 1),
        )
        wing_grid = add_control_surface(
            wing_grid,
            right_aileron,
            aileron_m_slice,
            slice(-(n_wing_dihedral + 1), None),
        )
        grids[idx_wing] = wing_grid
        for i in idx_elevator_tails:
            grids[i] = add_control_surface(
                grids[i], elevator, elevator_m_slice, slice(None)
            )
        grids[idx_rudder] = add_control_surface(
            grids[idx_rudder],
            rudder,
            rudder_m_slice,
            slice(None),
            hinge_axis=Z_AXIS,
        )
        return ArrayList(grids)

    aero = UVLM(
        grid_shapes=[
            GridDiscretisation(
                m=s["m"], n=len(s["mapping"]) - 1, m_star=s.get("m_star", m_star)
            )
            for s in surfaces
        ],
        dof_mapping=ArrayList([s["mapping"] for s in surfaces]),
        grid_func=aero_grid_func,  # type: ignore
        gamma_dot_relaxation=gamma_dot_relaxation,
    )

    aircraft = CoupledAeroelastic(aero=aero, structure=structure)
    aircraft.set_design_variables(
        coords=coords,
        k_cs=k_cs,
        m_cs=m_cs,
        m_lumped=m_lumped,
        dt=dt,
        flowfield=flowfield,
        x0_aero=[s["x0"] for s in surfaces],
        thrust_reference={k: jnp.zeros(()) for k in thrust_nodes},
        cs_angles_reference={
            "left_aileron": jnp.asarray(left_aileron_angle),
            "right_aileron": jnp.asarray(right_aileron_angle),
            "elevator": jnp.asarray(elevator_angle),
            "rudder": jnp.asarray(rudder_angle),
        },
        orientation_euler=jnp.array((roll, alpha, beta)),
    )

    return aircraft


if __name__ == "__main__":
    model = generate_xhale()

    trim_sol, trim_vars = model.trim(
        prescribed_dofs=tuple(range(6)),
        zero_force_dofs=(0, 2, 4),
        trim_orientation="y",
        thrust_nodes=[list(model.structure.thrust_reference.keys())],
        trim_cs="elevator",
        horseshoe=True,
        trim_relaxation=0.5,
    )

    trim_sol.plot("trimmed_xhale")
