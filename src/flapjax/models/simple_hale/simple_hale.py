from jax import Array
from jax import numpy as jnp

from flapjax.aero.data_structures import GridDiscretisation
from flapjax.aero.flowfields import (
    ConstantFlowField,
    FlowField,
    OneMinusCosineFlowField,
)
from flapjax.aero.utils import add_control_surface, make_rectangular_grid
from flapjax.aero.uvlm import UVLM
from flapjax.algebra.array_utils import ArrayList
from flapjax.algebra.so3 import exp_so3
from flapjax.coupled import CoupledAeroelastic
from flapjax.structure import BeamStructure
from flapjax.structure.constraints import MultibodyHinge
from flapjax.structure.data_structures import OptionalJacobians

FLOWFIELD_DEFAULT = ConstantFlowField(
    rho=1.225,
    u_inf=jnp.array((10.0, 0.0, 0.0)),
    relative_motion=True,
)


def generate_simple_hale(
    sigma_wing: float = 1.5,
    sigma_fuselage: float = 10.0,
    sigma_tail: float = 100.0,
    flowfield: FlowField = FLOWFIELD_DEFAULT,
    m_wing: int = 4,
    m_tail: int = 4,
    n_half_wing: int = 16,
    n_fuselage: int = 8,
    n_fin: int = 4,
    n_half_tail: int = 4,
    m_star: int = 20,
    alpha: Array | float = 0.0,
    roll: Array | float = 0.0,
    beta: Array | float = 0.0,
    flare_angle: Array | float | None = None,
    hinge_spring_stiffness: float = 0.0,
    hinge_damping: float = 0.0,
    dihedral_ang: Array | float = 20.0 / 180.0 * jnp.pi,
) -> CoupledAeroelastic:
    r"""
    Generate the simple HALE aircraft. Reference parameters are derived from the SHARPy case.
    :param sigma_wing: Stiffness multiplier for the wing structure.
    :param sigma_fuselage: Stiffness multiplier for the fuselage structure.
    :param sigma_tail: Stiffness multiplier for the tail structure.
    :param flowfield: FlowField object.
    :param m_wing: Number of chordwise aerodynamic panels for the wing.
    :param m_tail: Number of chordwise aerodynamic panels for the tail.
    :param n_half_wing: Number of spanwise panels for each wing.
    :param n_fuselage: Number of beam elements in the fuselage.
    :param n_fin: Number of height-wise panels in the tail fin.
    :param n_half_tail: Number of spanwise panels for each horizontal stabiliser half.
    :param m_star: Number of trailing wake panels.
    :param alpha: Reference angle of attack.
    :param roll: Reference roll angle.
    :param beta: Reference sideslip angle.
    :param flare_angle: If set, free-hinging wingtips will be used, where this sets the angle of the hinge axis relative
    to the wing plane. If None, no free-hinging wingtips are used.
    :param hinge_spring_stiffness: Rotational spring stiffness for the hinge joints (N·m/rad).
    :param hinge_damping: Rotational damping for the hinge joints (N·m·s/rad).
    :param dihedral_ang: Dihedral angle of the outer wing segment.
    :return: HALE aircraft object.
    """
    # geometry
    c_wing: float = 1.0
    c_tail: float = 0.5
    b_tail: float = 2.5
    b_wing: float = 16.0
    l_fuselage: float = 10.0
    height_tail: float = 2.5
    elastic_axis_wing: float = 0.3
    elastic_axis_tail: float = 0.5
    dihedral_fraction: float = 0.25

    # structural properties
    m_payload: float = 50.0
    m_bar_wing: float = 0.75
    m_bar_tail: float = 0.08
    m_bar_fuselage: float = 0.2
    j_bar_wing: float = 0.075
    j_bar_tail: float = 0.01
    j_bar_fuselage: float = 0.08
    gj: float = 1e4
    ea: float = 1e7
    ga: float = 1e5
    eiy = 2e4  # out-of-plane
    eiz = 4e6  # in-plane

    # discretisation
    n_outer_wing: int = int(n_half_wing * dihedral_fraction)
    n_inner_wing: int = n_half_wing - n_outer_wing

    # other variables
    dt: float = c_wing / (m_wing * float(jnp.linalg.norm(flowfield.u_inf)))

    multibody: bool = flare_angle is not None

    # dihedral folds about the hinge's own axis (misaligned from pure x by the flare angle) so that the
    # reference geometry is consistent with the hinge; falls back to the pure chordwise axis if unhinged
    left_hinge_axis = (
        jnp.array((jnp.cos(flare_angle), -jnp.sin(flare_angle), 0.0))
        if multibody
        else jnp.array((1.0, 0.0, 0.0))
    )

    # useful reference coordinates
    left_wing_junction_coord = jnp.array((0.0, (1.0 - dihedral_fraction) * b_wing, 0.0))
    left_wing_tip_coord = left_wing_junction_coord + exp_so3(
        left_hinge_axis * dihedral_ang
    ) @ jnp.array((0.0, dihedral_fraction * b_wing, 0.0))
    tail_base_coord = jnp.array((l_fuselage, 0.0, 0.0))
    tail_root_coord = jnp.array((l_fuselage, 0.0, height_tail))
    left_tail_tip_coord = tail_root_coord + jnp.array((0.0, b_tail, 0.0))

    # create beam coordinates
    left_wing_coords: Array = jnp.zeros((n_half_wing + 1, 3))  # includes zero node
    left_wing_coords = left_wing_coords.at[
        : n_inner_wing + 1, :
    ].set(  # inner straight wing
        jnp.outer(jnp.linspace(0.0, 1.0, n_inner_wing + 1), left_wing_junction_coord)
    )
    left_wing_coords = left_wing_coords.at[n_inner_wing:, :].set(  # add outer dihedral
        jnp.outer(
            jnp.linspace(0.0, 1.0, n_outer_wing + 1),
            left_wing_tip_coord - left_wing_junction_coord,
        )
        + left_wing_junction_coord[None, :]
    )

    right_wing_coords: Array = left_wing_coords[1:, :].at[:, 1].mul(-1.0)  # mirror wing

    fuselage_coords: Array = jnp.outer(
        jnp.linspace(0.0, 1.0, n_fuselage + 1)[1:], tail_base_coord
    )

    fin_coords: Array = tail_base_coord[None, :] + jnp.outer(
        jnp.linspace(0.0, 1.0, n_fin + 1)[1:], (tail_root_coord - tail_base_coord)
    )

    left_tail_coords: Array = tail_root_coord[None, :] + jnp.outer(
        jnp.linspace(0.0, 1.0, n_half_tail + 1)[1:],
        (left_tail_tip_coord - tail_root_coord),
    )

    right_tail_coords: Array = left_tail_coords[...].at[:, 1].mul(-1.0)  # mirror tail

    coords = jnp.concatenate(
        (
            left_wing_coords,
            right_wing_coords,
            fuselage_coords,
            fin_coords,
            left_tail_coords,
            right_tail_coords,
        )
    )

    # index of components in full coordinates
    left_wing_coord_index = jnp.arange(n_half_wing + 1)
    right_wing_coord_index = jnp.array(
        (0, *jnp.arange(n_half_wing) + left_wing_coord_index[-1] + 1)
    )
    fuselage_coord_index = jnp.array(
        (0, *jnp.arange(n_fuselage) + right_wing_coord_index[-1] + 1)
    )
    fin_coord_index = jnp.arange(n_fin + 1) + fuselage_coord_index[-1]
    left_tail_coord_index = jnp.arange(n_half_tail + 1) + fin_coord_index[-1]
    right_tail_coord_index = jnp.array(
        (fin_coord_index[-1], *(jnp.arange(n_fin) + left_tail_coord_index[-1] + 1))
    )

    # connectivity matrices
    conn_left_wing = jnp.stack(
        (left_wing_coord_index[:-1], left_wing_coord_index[1:]), axis=1
    )
    conn_right_wing = jnp.stack(
        (right_wing_coord_index[:-1], right_wing_coord_index[1:]), axis=1
    )
    conn_fuselage = jnp.stack(
        (fuselage_coord_index[:-1], fuselage_coord_index[1:]), axis=1
    )
    conn_fin = jnp.stack((fin_coord_index[:-1], fin_coord_index[1:]), axis=1)
    conn_left_tail = jnp.stack(
        (left_tail_coord_index[:-1], left_tail_coord_index[1:]), axis=1
    )
    conn_right_tail = jnp.stack(
        (right_tail_coord_index[:-1], right_tail_coord_index[1:]), axis=1
    )
    conn = jnp.concatenate(
        (
            conn_left_wing,
            conn_right_wing,
            conn_fuselage,
            conn_fin,
            conn_left_tail,
            conn_right_tail,
        ),
        axis=0,
    )

    # mass and stiffness matrices
    m_cs_wing = jnp.diag(
        jnp.array(
            (m_bar_wing, m_bar_wing, m_bar_wing, j_bar_wing, j_bar_wing, j_bar_wing)
        )
    )
    m_cs_fuselage = jnp.diag(
        jnp.array(
            (
                m_bar_fuselage,
                m_bar_fuselage,
                m_bar_fuselage,
                j_bar_fuselage,
                j_bar_fuselage,
                j_bar_fuselage,
            )
        )
    )
    m_cs_tail = jnp.diag(
        jnp.array(
            (m_bar_tail, m_bar_tail, m_bar_tail, j_bar_tail, j_bar_tail, j_bar_tail)
        )
    )
    k_cs_wing = jnp.diag(
        jnp.array((ea, ga, ga, gj * sigma_wing, eiy * sigma_wing, eiz * sigma_wing))
    )
    k_cs_fuselage = jnp.diag(
        jnp.array(
            (
                ea,
                ga,
                ga,
                gj * sigma_fuselage,
                eiy * sigma_fuselage,
                eiz * sigma_fuselage,
            )
        )
    )
    k_cs_tail = jnp.diag(
        jnp.array((ea, ga, ga, gj * sigma_tail, eiy * sigma_tail, eiz * sigma_tail))
    )
    m_lump = (
        jnp.zeros((6, 6))
        .at[:3, :3]
        .set(jnp.diag(jnp.array((m_payload, m_payload, m_payload))))
    )[None, :]

    # index of 0 for wing, 1 for fuselage and 2 for tail
    elem_type_index = jnp.concatenate(
        (
            jnp.zeros(2 * n_half_wing, dtype=int),
            jnp.full(n_fuselage, 1, dtype=int),
            jnp.full(n_fin + 2 * n_half_tail, 2, dtype=int),
        )
    )

    # columns are the y vector for the wing, fuselage and tails
    y_vectors = jnp.concatenate(
        (
            jnp.broadcast_to(jnp.array((1.0, 0.0, 0.0)), (2 * n_half_wing, 3)),
            jnp.broadcast_to(jnp.array((0.0, 1.0, 0.0)), (n_fuselage, 3)),
            jnp.broadcast_to(jnp.array((0.0, 1.0, 0.0)), (n_fin, 3)),
            jnp.broadcast_to(jnp.array((1.0, 0.0, 0.0)), (2 * n_half_tail, 3)),
        ),
        axis=0,
    )

    # multibody constraints for free-hinging wingtips
    if multibody:
        # add hinges at the wingtip base
        left_hinge = MultibodyHinge(
            node_i=int(left_wing_coord_index[n_inner_wing]),
            axis=left_hinge_axis,
            spring_stiffness=hinge_spring_stiffness,
            damping=hinge_damping,
        )
        right_hinge = MultibodyHinge(
            node_i=int(right_wing_coord_index[n_inner_wing]),
            axis=left_hinge_axis * jnp.array((1.0, -1.0, 1.0)),
            spring_stiffness=hinge_spring_stiffness,
            damping=hinge_damping,
        )
        constraints = {"left_hinge": left_hinge, "right_hinge": right_hinge}
    else:
        constraints = None

    structure = BeamStructure(
        num_nodes=coords.shape[0],
        connectivity=conn,
        y_vector=y_vectors,
        m_cs_index=elem_type_index,
        k_cs_index=elem_type_index,
        m_lumped_index=jnp.array(0, dtype=int),
        gravity=jnp.array((0.0, 0.0, -9.81)),
        thrust_nodes={"thrust": 0},
        thrust_direction={"thrust": jnp.array((-1.0, 0.0, 0.0))},
        relaxation_factor=0.7,
        spectral_radius=0.5,
        constraints=constraints,
        optional_jacobians=OptionalJacobians(
            d_f_ext_dead_d_n=multibody,
            d_f_grav_d_n=multibody,
        )
        if multibody
        else None,
    )

    # aerodynamic_grids
    # wing
    wing_x0 = make_rectangular_grid(
        m=m_wing, n=2 * n_half_wing, chord=c_wing, ea=elastic_axis_wing
    )

    # add aerodynamic changes for flare angle
    if multibody:
        idx_hinge = n_outer_wing
        hinge_strip = wing_x0[:, idx_hinge, :]

        delta_hinge = jnp.tan(flare_angle) * hinge_strip[:, 0]

        left_hinge_strip = hinge_strip.at[:, 1].set(-delta_hinge)
        right_hinge_strip = hinge_strip.at[:, 1].set(delta_hinge)

        wing_x0 = wing_x0.at[:, idx_hinge, :].set(left_hinge_strip)
        wing_x0 = wing_x0.at[:, -idx_hinge - 1, :].set(right_hinge_strip)

    wing_mapping = jnp.concatenate(
        (left_wing_coord_index[::-1], right_wing_coord_index[1:])
    )
    wing_grid_shape = GridDiscretisation(m=m_wing, n=2 * n_half_wing, m_star=m_star)

    # fin
    fin_x0 = make_rectangular_grid(
        m=m_tail, n=n_fin, chord=c_tail, ea=elastic_axis_tail
    )
    fin_mapping = fin_coord_index
    fin_grid_shape = GridDiscretisation(m=m_tail, n=n_fin, m_star=m_star)

    # tail
    tail_x0 = make_rectangular_grid(
        m=m_tail, n=2 * n_half_tail, chord=c_tail, ea=elastic_axis_tail
    )
    tail_mapping = jnp.concatenate(
        (left_tail_coord_index[::-1], right_tail_coord_index[1:])
    )
    tail_grid_shape = GridDiscretisation(m=m_tail, n=2 * n_half_tail, m_star=m_star)

    # control surface spanwise extents on the wing grid (left tip → root → right tip)
    n_left_aileron_grid = n_outer_wing + 1
    n_right_aileron_grid = n_outer_wing + 1

    # control surface chordwise extents
    elevator_m_slice = slice(0, None)
    rudder_m_slice = slice(m_tail - 1, None)
    aileron_m_slice = slice(int(0.75 * m_wing), None)

    def aero_grid_func(
        x0: ArrayList,
        *,
        left_aileron: Array,
        right_aileron: Array,
        elevator: Array,
        rudder: Array,
    ) -> ArrayList:
        wing_grid = add_control_surface(
            grid=x0[0],
            angle=left_aileron,
            m_slice=aileron_m_slice,
            n_slice=slice(0, n_left_aileron_grid),
        )
        wing_grid = add_control_surface(
            grid=wing_grid,
            angle=right_aileron,
            m_slice=aileron_m_slice,
            n_slice=slice(-n_right_aileron_grid, None),
        )
        fin_grid = add_control_surface(
            grid=x0[1],
            angle=rudder,
            m_slice=rudder_m_slice,
            n_slice=slice(None),
        )
        tail_grid = add_control_surface(
            grid=x0[2],
            angle=elevator,
            m_slice=elevator_m_slice,
            n_slice=slice(None),
        )
        return ArrayList([wing_grid, fin_grid, tail_grid])

    aero = UVLM(
        grid_shapes=(wing_grid_shape, fin_grid_shape, tail_grid_shape),
        dof_mapping=(wing_mapping, fin_mapping, tail_mapping),
        grid_func=aero_grid_func,  # type: ignore
        gamma_dot_relaxation=0.5,
    )
    wing = CoupledAeroelastic(aero=aero, structure=structure)
    wing.set_design_variables(
        coords=coords,
        k_cs=jnp.stack([k_cs_wing, k_cs_fuselage, k_cs_tail], axis=0),
        m_cs=jnp.stack([m_cs_wing, m_cs_fuselage, m_cs_tail], axis=0),
        m_lumped=m_lump,
        dt=dt,
        flowfield=flowfield,
        delta_w=None,
        x0_aero=[wing_x0, fin_x0, tail_x0],
        thrust_reference={"thrust": jnp.zeros(())},
        cs_angles_reference={
            "left_aileron": jnp.zeros(()),
            "right_aileron": jnp.zeros(()),
            "elevator": jnp.zeros(()),
            "rudder": jnp.zeros(()),
        },
        orientation_euler=jnp.array((roll, alpha, beta)),
    )

    return wing


if __name__ == "__main__":
    # trim the simple hale aircraft and run a gust case
    u_inf_mag: float = 10.0
    gust_intensity: float = 0.2
    gust_length: float = 1.0 * u_inf_mag  # 1 second gust duration
    physical_time: float = 10.0

    flowfield_ = OneMinusCosineFlowField(
        u_inf=jnp.array((u_inf_mag, 0.0, 0.0)),
        rho=1.225,
        relative_motion=True,
        gust_length=gust_length,
        gust_amplitude=gust_intensity * u_inf_mag,
        gust_x0=jnp.array((-2.0 * gust_length, 0.0, 0.0)),
    )

    # create aircraft model
    hale = generate_simple_hale(flowfield=flowfield_, sigma_wing=1.5)
    n_tstep = int(physical_time / float(hale.aero.dt)) + 1

    # trim the aircraft
    static_sol, trim_vars = hale.trim(
        prescribed_dofs=jnp.arange(6),
        zero_force_dofs=(0, 2, 4),  # balance drag, lift, and pitching moment
        trim_cs="elevator",
        thrust_nodes="thrust",
        trim_orientation="y",
        horseshoe=False,
    )

    # run gust case
    dynamic_init = hale.initialise_dynamic(static_case=static_sol, prescribed_dofs=())

    # noinspection bad-argument-type
    dynamic_sol = hale.dynamic_solve(
        init_case=dynamic_init, prescribed_dofs=(), n_tstep=n_tstep
    )

    # plot the transient
    dynamic_sol.plot("./simple_hale/")
