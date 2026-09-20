from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal, Protocol, overload

import jax
from jax import Array, vmap
from jax import numpy as jnp
from jax.lax import cond

if TYPE_CHECKING:
    from flapjax.aero import AeroCase
from flapjax.algebra.array_utils import ArrayList, neighbour_average, split_to_vertex
from flapjax.algebra.base import finite_difference
from flapjax.algebra.so3 import exp_so3
from flapjax.utils.constants import EPSILON, R_CUTOFF
from flapjax.utils.utils import index_to_arr

type KernelFunction = Callable[[Array, Array], Array]
type PolarFunction = Callable[[Array], tuple[Array, Array, Array]]


# Local "spanwise" axis shared by control-surface hinges (add_control_surface) and built-in geometric twist
# (make_rectangular_grid): both are rotations of the local chord/camber (x, z) section about this axis.
HINGE_AXIS_DEFAULT = jnp.array((0.0, 1.0, 0.0))


def make_rectangular_grid(
    m: int,
    n: int,
    chord: Array | float,
    ea: Array | float,
    camber_line: tuple[Array, Array] | None = None,
    twist: Array | float = 0.0,
) -> Array:
    r"""
    Create a rectangular aerodynamic grid.
    :param m: Number of panels in the chordwise direction.
    :param n: Number of panels in the spanwise direction.
    :param chord: Surface chord length.
    :param ea: Elastic axis location as fraction of chord.
    :param camber_line: Optional mean camber line as a pair ``(x/c, z/c)`` of equal-length vectors, giving
    the camber-line height at a set of chordwise stations, with ``x/c`` running from 0 (leading edge)
    to 1 (trailing edge). If None (default), the section is a flat plate.
    :param twist: Built-in geometric twist angle in radians, uniform over the whole grid, applied by rotating the
    local chord/camber section about the local spanwise axis (``HINGE_AXIS_DEFAULT``). This is a purely
    aerodynamic incidence offset: unlike a beam's ``y_vector``-defined twist (which only reorients the
    structural cross-section's stiffness/mass axes), it actually changes the panels' angle of attack, since a
    uniform twist produces no curvature for the structural solver to pick up on its own. Default 0 (no twist).
    :return: Local grid points for planar wing, ``(zeta_m, zeta_n, 3)``.
    """

    x_over_c = jnp.linspace(0.0, 1.0, m + 1)
    grid = jnp.zeros((m + 1, n + 1, 3))
    grid = grid.at[..., 0].set((x_over_c * chord - ea * chord)[:, None])
    if camber_line is not None:
        camber_x, camber_z = camber_line
        z_over_c = jnp.interp(x_over_c, camber_x, camber_z)
        grid = grid.at[..., 2].set((z_over_c * chord)[:, None])
    rmat = exp_so3(HINGE_AXIS_DEFAULT * twist)
    grid = jnp.einsum("ij,mnj->mni", rmat, grid)
    return grid


def add_control_surface(
    grid: Array,
    angle: Array,
    m_slice: Array | Sequence[int] | slice,
    n_slice: Array | Sequence[int] | slice,
    hinge_axis: Array = HINGE_AXIS_DEFAULT,
) -> Array:
    r"""
    Add a control surface to a panel grid.
    :param grid: Grid without deflection of this surface, ``(zeta_m, zeta_n, 3)``.
    :param angle: Angle in radians through which the control surface will be deflected.
    :param m_slice: Slice of chordwise strips to include in the control surface.
    :param n_slice: Slice of spanwise strips to include in the control surface.
    :param hinge_axis: Axis of the hinge surface in the local frame, ``(3, )``.
    :return: Deflected aerodynamic grid, ``(zeta_m, zeta_n, 3)``.
    """

    m_slice_arr: Array = index_to_arr(index=m_slice, n_entries=grid.shape[0])
    n_slice_arr: Array = index_to_arr(index=n_slice, n_entries=grid.shape[1])

    # grid for deflected surfaces
    grid_out = grid

    def inner_func(n_idx: Array) -> Array:
        hinge_point = grid[m_slice_arr[0], n_idx, :]  # (3, )

        crv = hinge_axis * angle  # cartesian rotation vector for surface, (3, ).
        rmat = exp_so3(crv)  # rotation matrix for rotating surface

        # transform coordinates to rotate control surface
        return (
            jnp.einsum(
                "ij,hj->hi",
                rmat,
                (grid[m_slice_arr, n_idx, :] - hinge_point[None, :]),
            )
            + hinge_point[None, :]
        )

    # update grid
    return grid_out.at[jnp.ix_(m_slice_arr, n_slice_arr, jnp.arange(3))].set(
        vmap(inner_func, in_axes=0, out_axes=1)(n_slice_arr)
    )


def compute_surf_c(zeta: Array) -> Array:
    r"""
    Compute the colocation points for a given grid of points on a single surface.
    :param zeta: Grid of points, ``(..., zeta_m, zeta_n, 3)``.
    :return: Colocation points, ``(..., m, n, 3)``.
    """
    return neighbour_average(zeta, axes=(-3, -2))


def compute_surf_nc(zeta: Array) -> Array:
    r"""
    Compute the surface normal vectors for a given grid of points on a single surface. These have length equal to the
    area of their corresponding panel.
    :param zeta: Grid of points, ``(..., zeta_m, zeta_n, 3)``.
    :return: Normal vectors, ``(..., m, n, 3)``.
    """
    diag1 = zeta[..., 1:, 1:, :] - zeta[..., :-1, :-1, :]
    diag2 = zeta[..., 1:, :-1, :] - zeta[..., :-1, 1:, :]
    return jnp.cross(diag1, diag2)


def compute_c(zetas: ArrayList) -> ArrayList:
    r"""
    Compute the colocation points for a list of surface grids.
    :param zetas: Grids of points, ``(n_surf, )(zeta_m, zeta_n, 3)``.
    :return: Colocation points ``(n_surf ,)(m, n, 3)``.
    """
    return ArrayList([compute_surf_c(zeta) for zeta in zetas])


def compute_nc(zetas: ArrayList) -> ArrayList:
    r"""
    Compute the surface normal vectors for a list of surface grids.
    :param zetas: Grids of points, ``(n_surf, )(zeta_m, zeta_n, 3)``.
    :return: Normal vectors ``(n_surf, )(m, n, 3)``.
    """
    return ArrayList([compute_surf_nc(zeta) for zeta in zetas])


def compute_steady_forcing(
    zeta_b: ArrayList,
    zeta_dot_b: ArrayList | None,
    gamma_b: ArrayList,
    gamma_w: ArrayList,
    rho: Array,
    v_func: Callable[[Array], Array],
    v_inputs: ArrayList | None,
) -> ArrayList:
    r"""
    Calculate steady aerodynamic forcing for all surfaces at specified time step.
    :param zeta_b: Bound grid coordinates, ``(n_surf, )(zeta_m, zeta_n, 3)``.
    :param zeta_dot_b: Bound grid velocities, ``(n_surf, )(zeta_m, zeta_n, 3)``.
    :param gamma_b: Bound grid circulation, ``(n_surf, )(m, n)``
    :param gamma_w: Wake grid circulation, ``(n_surf, )(m, n)``
    :param rho: Flow field density.
    :param v_func: Total velocity as a function of coordinate.
    :param v_inputs: Additive inputs for total velocity on bound grid vertex, used for the linear solver for custom
    perturbations.
    """

    f_steady = ArrayList([])

    if zeta_dot_b is None:
        zeta_dot_bs_: list[Array | None] = [None] * len(zeta_b)
    else:
        zeta_dot_bs_ = zeta_dot_b

    if v_inputs is None:
        v_inputs_ = [None] * len(zeta_b)
    else:
        v_inputs_ = v_inputs

    for zeta_b_surf, zeta_dot_b_surf, gamma_b_surf, gamma_w_surf, v_input_surf in zip(
        zeta_b, zeta_dot_bs_, gamma_b, gamma_w, v_inputs_
    ):
        # compute midpoints
        mp_chordwise = neighbour_average(zeta_b_surf, axes=0)  # (gamma_m, gamma_n+1, 3)
        mp_spanwise = neighbour_average(zeta_b_surf, axes=1)  # (gamma_m+1, gamma_n, 3)

        assert zeta_dot_b_surf is not None

        mp_dot_chordwise = neighbour_average(
            zeta_dot_b_surf, axes=0
        )  # (gamma_m, gamma_n+1, 3)
        mp_dot_spanwise = neighbour_average(
            zeta_dot_b_surf, axes=1
        )  # (gamma_m+1, gamma_n, 3)

        # relative flow velocities at midpoints
        v_rel_chordwise = (
            v_func(mp_chordwise) - mp_dot_chordwise
        )  # (gamma_m, gamma_n+1, 3)
        v_rel_spanwise = (
            v_func(mp_spanwise) - mp_dot_spanwise
        )  # (gamma_m+1, gamma_n, 3)

        # add any input_ velocities
        if v_input_surf is not None:
            v_rel_chordwise += neighbour_average(v_input_surf, axes=0)
            v_rel_spanwise += neighbour_average(v_input_surf, axes=1)

        # equivalent strengths of filaments
        gamma_chordwise = jnp.zeros(
            v_rel_chordwise.shape[:-1]
        )  # (gamma_m, gamma_n+1, 3)
        gamma_chordwise = gamma_chordwise.at[:, :-1].set(gamma_b_surf)
        gamma_chordwise = gamma_chordwise.at[:, 1:].add(-gamma_b_surf)
        gamma_spanwise = jnp.zeros(v_rel_spanwise.shape[:-1])  # (gamma_m+1, gamma_n, 3)
        gamma_spanwise = gamma_spanwise.at[:-1, :].set(-gamma_b_surf)
        gamma_spanwise = gamma_spanwise.at[1:, :].add(gamma_b_surf)

        # add first wake gamma
        if gamma_w_surf.shape[0] > 0:
            gamma_spanwise = gamma_spanwise.at[-1, :].add(-gamma_w_surf[0, :])

        # filament vectors (from zeta_b_fil, which may differ from the midpoint geometry)
        r_chordwise = (
            zeta_b_surf[1:, :, :] - zeta_b_surf[:-1, :, :]
        )  # (gamma_m, gamma_n+1, 3)
        r_spanwise = (
            zeta_b_surf[:, 1:, :] - zeta_b_surf[:, :-1, :]
        )  # (gamma_m+1, gamma_n, 3)

        # forces from each set of filaments
        f_chordwise = rho * jnp.einsum(
            "ij,ijk->ijk",
            gamma_chordwise,
            jnp.cross(v_rel_chordwise, r_chordwise),
        )  # (gamma_m, gamma_n+1, 3)
        f_spanwise = rho * jnp.einsum(
            "ij,ijk->ijk", gamma_spanwise, jnp.cross(v_rel_spanwise, r_spanwise)
        )  # (gamma_m+1, gamma_n, 3)

        f_steady.append(
            split_to_vertex(f_chordwise, 0) + split_to_vertex(f_spanwise, 1)
        )  # (gamma_m+1, gamma_n+1, 3)
    return f_steady


def strip_alpha(
    zeta_b: ArrayList,
    f_steady: ArrayList,
    v_func: Callable[[Array], Array],
    rho: Array,
) -> ArrayList:
    r"""
    Compute the per-strip effective angle of attack from the UVLM sectional forcing. For each spanwise strip, the strip
    total force is projected onto the local lift direction to obtain the strip lift coefficient. The angle of attack is
    found by using a lift slope of 2 pi.

    :param zeta_b: Bound grid coordinates, ``(n_surf, )(zeta_m, zeta_n, 3)``.
    :param f_steady: Steady forcing, ``(n_surf, )(zeta_m, zeta_n, 3)``.
    :param v_func: Reference velocity as a function of position, ``(..., 3) -> (..., 3)``.
    :param rho: Flow density.
    :return: Per-surface strip angle of attack, ``(n_surf, )(n_strip,)``.
    """
    alphas = ArrayList([])
    for zeta_surf, f_surf in zip(zeta_b, f_steady):
        # find mid-panel chord length
        zeta_le = zeta_surf[0, :, :]
        zeta_te = zeta_surf[-1, :, :]
        le_j = 0.5 * (zeta_le[:-1, :] + zeta_le[1:, :])
        te_j = 0.5 * (zeta_te[:-1, :] + zeta_te[1:, :])

        chord_vec = te_j - le_j  # (n, 3)
        c_len = jnp.linalg.norm(chord_vec, axis=-1)  # (n)

        span_vec = 0.5 * (
            (zeta_le[1:, :] + zeta_te[1:, :]) - (zeta_le[:-1, :] + zeta_te[:-1, :])
        )  # (n, 3)
        b_len = jnp.linalg.norm(span_vec, axis=-1)  # (n)
        e_s = span_vec / b_len[:, None]  # unit vector in span direction

        strip_mid = 0.5 * (le_j + te_j)  # centre of strip
        v_ref = v_func(strip_mid)  # evaluate velocity at mid-strip
        v_mag2 = jnp.sum(v_ref * v_ref, axis=-1)

        e_l = jnp.cross(v_ref, e_s)  # vector in lift direction
        e_l /= jnp.linalg.norm(e_l, axis=-1, keepdims=True)  # make unit vector

        f_strip = neighbour_average(f_surf, axes=1).sum(axis=0)
        q = 0.5 * rho * v_mag2

        # correct from 2 pi lift slope to custom input
        cl_uvlm = jnp.sum(f_strip * e_l, axis=-1) / (q * c_len * b_len)
        alphas.append(cl_uvlm / (2.0 * jnp.pi))
    return alphas


def apply_polar_correction(
    zeta_b: ArrayList,
    f_steady: ArrayList,
    v_func: Callable[[Array], Array],
    rho: Array,
    polars: Sequence[Sequence[PolarFunction] | None],
    alpha: ArrayList | None = None,
) -> tuple[ArrayList, ArrayList]:
    r"""
    Replace UVLM strip forcing with a sectional force built from tabulated airfoil polars.

    For each surface where polars are provided, and for each spanwise strip, we firstly compute the strip forcing in the
    global frame. This forcing can then be converted into the local strip coordinate frame. We compute the velocity at
    the mid-strip point, from which we can calculate the lift coefficient. By assuming a ``2*pi`` potential flow lift
    slope, we can correct the forcing from tabulate data of the airfoil polar for lift, drag and moment. To correct the
    output, we distribure the forcing back onto the grid in a way which satisfies the moment and force balance.
    :param zeta_b: Bound grid coordinates, ``(n_surf, )(zeta_m, zeta_n, 3)``.
    :param f_steady: Steady vertex forcing, ``(n_surf, )(zeta_m, zeta_n, 3)``.
    :param v_func: Reference velocity as a function of position, ``(..., 3) -> (..., 3)``.
    :param rho: Flow density.
    :param polars: Per-surface polar tables. Each entry is either ``None`` (no correction for
        that surface) or a sequence of length ``n`` of callables mapping
        ``alpha -> (cl, cd, cm)`` about the quarter-chord.
    :param alpha: Optional precomputed per-strip angle of attack, ``(n_surf, )(n_strip,)``. If
        ``None``, computed internally via :func:`strip_alpha` with the same ``v_func``.
    :return: Corrected vertex forcing, ``(n_surf, )(zeta_m, zeta_n, 3)`` and per-strip lift scale factors,
        ``(n_surf, )(n, )`` which can be used to scale the circulation strengths if requested.
    """
    if alpha is None:
        # compute the angles of attack for each strip if not passed
        alpha = strip_alpha(zeta_b=zeta_b, f_steady=f_steady, v_func=v_func, rho=rho)

    f_out = ArrayList([])
    lift_scale_out = ArrayList([])
    for i_surf, (zeta_surf, f_surf, polar_surf, alpha_surf) in enumerate(
        zip(zeta_b, f_steady, polars, alpha)
    ):
        n = zeta_surf.shape[1] - 1

        if polar_surf is None:
            # no correction to apply
            f_out.append(f_surf)
            lift_scale_out.append(jnp.ones(n))
            continue

        zeta_m = zeta_surf.shape[0]
        m = zeta_m - 1

        if len(polar_surf) != n:
            raise ValueError(
                f"Expected {n} polar callables for surface {i_surf}, got {len(polar_surf)}"
            )

        # find leading and trailing edges of centre of strip
        zeta_le = zeta_surf[0, :, :]
        zeta_te = zeta_surf[-1, :, :]
        le_j = 0.5 * (zeta_le[:-1, :] + zeta_le[1:, :])
        te_j = 0.5 * (zeta_te[:-1, :] + zeta_te[1:, :])

        chord_vec = te_j - le_j
        c_len = jnp.linalg.norm(chord_vec, axis=-1)  # chord of strips, (n, ).
        e_c = chord_vec / c_len[:, None]  # unit vector in chordwise direction, (n, 3)

        span_vec = 0.5 * (
            (zeta_le[1:, :] + zeta_te[1:, :]) - (zeta_le[:-1, :] + zeta_te[:-1, :])
        )
        b_len = jnp.linalg.norm(span_vec, axis=-1)  # span of strips, (n, ).
        e_s = span_vec / b_len[:, None]  # unit vector in spanwise direction, (n, 3)

        e_n = jnp.cross(e_c, e_s)
        e_n /= jnp.linalg.norm(
            e_n, axis=-1, keepdims=True
        )  # unit vector in normal direction, (n, 3)

        # compute dynamic pressure at centroid of strip
        strip_mid = 0.5 * (le_j + te_j)
        v_ref = v_func(strip_mid)
        v_mag2 = jnp.sum(v_ref * v_ref, axis=-1)
        v_mag = jnp.sqrt(v_mag2)
        q = 0.5 * rho * v_mag2

        e_l = jnp.cross(v_ref, e_s)
        e_l /= jnp.linalg.norm(
            e_l, axis=-1, keepdims=True
        )  # unit vector in flow direction, (n, 3)
        e_d = v_ref / v_mag[:, None]

        # extract from polar data
        cl_p = jnp.stack([polar_surf[j](alpha_surf[j])[0] for j in range(n)])
        cd_p = jnp.stack([polar_surf[j](alpha_surf[j])[1] for j in range(n)])
        cm_p = jnp.stack([polar_surf[j](alpha_surf[j])[2] for j in range(n)])

        # lift scale factor cl_polar / cl_uvlm with cl_uvlm = 2 pi alpha
        cl_uvlm = 2.0 * jnp.pi * alpha_surf
        lift_scale = jnp.where(jnp.abs(cl_uvlm) > EPSILON, cl_p / cl_uvlm, 1.0)
        lift_scale_out.append(lift_scale)

        qcb = q * c_len * b_len
        f_lump = qcb[:, None] * (cl_p[:, None] * e_l + cd_p[:, None] * e_d)
        f_couple = (qcb * cm_p)[:, None] * e_n  # rotate back into normal direction

        # chordwise triangular weights placing the lumped force at c/4
        chord_fracs = jnp.linspace(0.0, 1.0, zeta_m)
        w_chord = jnp.clip(1.0 - jnp.abs(chord_fracs - 0.25) * m, 0.0, 1.0)

        f_corrected = jnp.zeros_like(f_surf)

        # split force across the two bounding spanwise vertex columns
        lump_contrib = 0.5 * w_chord[:, None, None] * f_lump[None, :, :]
        f_corrected = f_corrected.at[:, :-1, :].add(lump_contrib)
        f_corrected = f_corrected.at[:, 1:, :].add(lump_contrib)

        # correct for moment with LE/TE opposing forces
        couple_contrib = 0.5 * f_couple
        f_corrected = f_corrected.at[0, :-1, :].add(couple_contrib)
        f_corrected = f_corrected.at[0, 1:, :].add(couple_contrib)
        f_corrected = f_corrected.at[-1, :-1, :].add(-couple_contrib)
        f_corrected = f_corrected.at[-1, 1:, :].add(-couple_contrib)

        f_out.append(f_corrected)

    return f_out, lift_scale_out


def propagate_surf_wake(
    gamma_b_nm1: Array,
    gamma_w_nm1: Array,
    zeta_b_n: Array,
    zeta_w_nm1: Array,
    delta_w: Array | None,
    v_func: Callable[[Array], Array],
    dt: Array,
    frozen_wake: bool,
    linearise_variable_wake: bool = False,
) -> tuple[Array | None, Array]:
    r"""
    Convect the wake at some given velocity for a single surface from timestep n-1 to timestep n. This step includes
    convection from the trailing edge and culling the downstream data.
    :param gamma_b_nm1: Bound circulation at time step n-1, ``(m, n)``.
    :param gamma_w_nm1: Wake circulation at time step n-1, ``(m_star, n)``.
    :param zeta_b_n: Bound grid at time step n, ``(zeta_m, zeta_n, 3)``.
    :param zeta_w_nm1: Wake grid at time step n-1, ``(zeta_m_star, zeta_n, 3)``.
    :param delta_w: Desired wake discretisation, ``(zeta_m_star, 3)``, or None for uniform.
    :param v_func: Function that computes the velocity as a function of coordinate, ``(3, )`` -> ``(3, )``.
    :param dt: Time step length.
    :param frozen_wake: If true, the grid stays constant with time. Used in the linearised case.
    :param linearise_variable_wake: If true, block gradients through the arc-length computation so that
        the re-discretisation is treated as a linear operator when differentiated.
    :return: New wake grid and circulation, ``(zeta_m_star, zeta_n, 3)``, ``(m_star, n)``.
    """

    # trailing edge positions and circulations
    zeta_te = zeta_b_n[-1, ...]  # (zeta_n, 3)
    gamma_te = gamma_b_nm1[-1, ...]  # (gamma_n)

    # variable wake discretisation also depends on the final element
    if delta_w is not None:
        zeta_base = zeta_w_nm1  # (zeta_w_m, zeta_n, 3)
        gamma_base = gamma_w_nm1  # (gamma_w_m, gamma_n)
    else:
        zeta_base = zeta_w_nm1[:-1, ...]  # (zeta_w_m - 1, zeta_n, 3)
        gamma_base = gamma_w_nm1[:-1, ...]  # (gamma_w_m - 1, gamma_n)

    # values at t=n+1 before re-discretisation
    gamma_w_n = jnp.concatenate(
        (gamma_te[None, ...], gamma_base), axis=0
    )  # (gamma_w_m+1 | gamma_w_m, gamma_n)

    # if the wake is free, this should be embedded here
    v = v_func(zeta_base)  # (zeta_w_m | zeta_w_m-1, zeta_n, 3)

    # wake coordinates at t=n+1 before re-discretisation
    zeta_w_n = jnp.concatenate(
        (zeta_te[None, :, :], zeta_base + dt * v), axis=0
    )  # (zeta_w_m+1 | zeta_w_m, zeta_n, 3)

    if delta_w is not None:
        # streamline coordinates before re-discretisation
        s_zeta_w = jnp.concatenate(
            (
                jnp.zeros((1, zeta_te.shape[0])),  # (1, zeta_n)
                jnp.cumsum(
                    jnp.linalg.norm(
                        zeta_w_n[1:, ...] - zeta_w_n[:-1, ...], axis=-1
                    ),  # (zeta_w_m+1, zeta_n)
                    axis=0,
                ),  # (zeta_w_m, zeta_n)
            ),
            axis=0,
        )  # distance along each wake filament for each point (zeta_w_m + 1, zeta_n]

        if linearise_variable_wake:
            s_zeta_w = jax.lax.stop_gradient(s_zeta_w)

        # consider gamma to be at midpoints of zeta
        s_gamma_w = neighbour_average(s_zeta_w, axes=(0, 1))

        # vertex coordinates along desired discretised streamline, (m_star + 1)
        s_zeta_w_discretisation = jnp.concatenate((jnp.zeros(1), jnp.cumsum(delta_w)))

        # midpoint coordinates along desired discretised streamline, (m_star)
        s_gamma_w_discretisation = neighbour_average(s_zeta_w_discretisation, axes=(0,))

        # re-discretise coordinates onto desired grid
        zeta_w_n = vmap(
            vmap(jnp.interp, in_axes=(None, 0, 0), out_axes=1),
            in_axes=(None, None, 1),
            out_axes=2,
        )(
            s_zeta_w_discretisation, s_zeta_w.T, jnp.transpose(zeta_w_n, (1, 2, 0))
        )  # (zeta_w_m, zeta_n, 3)

        # re-discretise gamma onto desired grid
        gamma_w_n = vmap(jnp.interp, in_axes=(None, 0, 0), out_axes=1)(
            s_gamma_w_discretisation, s_gamma_w.T, gamma_w_n.T
        )  # (zeta_w_m, zeta_n, 3)

    # logic for edge case where there is no wake
    if gamma_w_nm1.size == 0:
        gamma_w_n = jnp.zeros_like(gamma_w_nm1)

    if frozen_wake:
        return None, gamma_w_n
    else:
        return zeta_w_n, gamma_w_n


@overload
def propagate_wake(
    gamma_b_nm1: ArrayList,
    gamma_w_nm1: ArrayList,
    zeta_b_n: ArrayList,
    zeta_w_nm1: ArrayList,
    delta_w: Sequence[Array | None],
    v_func: Callable[[Array], Array],
    dt: Array,
    frozen_wake: Literal[True],
    linearise_variable_wake: bool,
) -> tuple[None, ArrayList]: ...


@overload
def propagate_wake(
    gamma_b_nm1: ArrayList,
    gamma_w_nm1: ArrayList,
    zeta_b_n: ArrayList,
    zeta_w_nm1: ArrayList,
    delta_w: Sequence[Array | None],
    v_func: Callable[[Array], Array],
    dt: Array,
    frozen_wake: Literal[False],
    linearise_variable_wake: bool,
) -> tuple[ArrayList, ArrayList]: ...


def propagate_wake(
    gamma_b_nm1: ArrayList,
    gamma_w_nm1: ArrayList,
    zeta_b_n: ArrayList,
    zeta_w_nm1: ArrayList,
    delta_w: Sequence[Array | None],
    v_func: Callable[[Array], Array],
    dt: Array,
    frozen_wake: bool,
    linearise_variable_wake: bool = False,
) -> tuple[ArrayList | None, ArrayList]:
    r"""
    Convect the wake for all surfaces.
    :param gamma_b_nm1: Bound circulation at time step n-1, ``(n_surf, )(m, n)``.
    :param gamma_w_nm1: Wake circulation at time step n-1, ``(n_surf, )(m_star, n)``.
    :param zeta_b_n: Bound grid at time step n, ``(n_surf, )(zeta_m, zeta_n, 3)``.
    :param zeta_w_nm1: Wake grid at time step n-1, ``(n_surf, )(zeta_m_star, zeta_n, 3)``.
    :param delta_w: Desired wake discretisation, ``(n_surf, )(zeta_m_star, 3)`` or None for uniform.
    :param v_func: Function that computes the velocity, ``(3, )`` -> ``(3, )``.
    :param dt: Time step length.
    :param frozen_wake: If true, the grid stays constant with time, useful in the linearised case.
    :param linearise_variable_wake: If true, block gradients through the arc-length computation so that
        the re-discretisation is treated as a linear operator when differentiated with jax.jvp.
    :return: New wake grid and circulation, ``(n_surf, )(zeta_m_star, zeta_n, 3)``, ``(n_surf, )(m_star, n)``.
    """

    n_surf = len(gamma_b_nm1)
    zeta_w_n: ArrayList | None = ArrayList([]) if not frozen_wake else None
    gamma_w_n = ArrayList([])

    for i_surf in range(n_surf):
        surf_zeta_w, surf_gamma_w = propagate_surf_wake(
            gamma_b_nm1=gamma_b_nm1[i_surf],
            gamma_w_nm1=gamma_w_nm1[i_surf],
            zeta_b_n=zeta_b_n[i_surf],
            zeta_w_nm1=zeta_w_nm1[i_surf],
            delta_w=delta_w[i_surf],
            v_func=v_func,
            dt=dt,
            frozen_wake=frozen_wake,
            linearise_variable_wake=linearise_variable_wake,
        )
        if zeta_w_n is not None:
            assert surf_zeta_w is not None
            zeta_w_n.append(surf_zeta_w)
        gamma_w_n.append(surf_gamma_w)
    return zeta_w_n, gamma_w_n


def biot_savart(x: Array, y: Array) -> Array:
    r"""
    Biot-Savart kernel without any smoothing or cutoff.
    :param x: Target point, ``(3, )``.
    :param y: Filament endpoints, ``(2, 3)``.
    :return: Influence at target point, ``(3, )``.
    """
    r0 = y[1, :] - y[0, :]
    r1 = x - y[0, :]
    r2 = x - y[1, :]
    r1_x_r2 = jnp.cross(r1, r2)
    diff_r = r1 / jnp.linalg.norm(r1) - r2 / jnp.linalg.norm(r2)
    return r1_x_r2 / (jnp.inner(r1_x_r2, r1_x_r2) * 4.0 * jnp.pi) * jnp.dot(r0, diff_r)


def make_unit_epsilon(r: Array) -> Array:
    r"""
    Differentiable function to obtain a unit vector that is defined for all ``r``. As ``r`` -> 0, the output approaches
    zero instead of being undefined. Autodiff of this form matches a custom JVP to floating-point noise but transposes
    to a much cheaper VJP under reverse-mode.
    :param r: Vector to be normalised, ``(3, )``.
    :return: Unit vector, ``(3, )``.
    """
    return r / jnp.sqrt(jnp.sum(r**2) + EPSILON**2)


def biot_savart_epsilon(x: Array, y: Array) -> Array:
    r"""
    Biot-Savart kernel with epsilon term added to remove singularity.

    :param x: Target point, ``(3, )``.
    :param y: Filament endpoints, ``(2, 3)``.
    :return: Influence at target point, ``(3, )``.
    """
    r0 = y[1, :] - y[0, :]
    r1 = x - y[0, :]
    r2 = x - y[1, :]
    inv_n1 = jax.lax.rsqrt(jnp.sum(r1 * r1) + EPSILON * EPSILON)
    inv_n2 = jax.lax.rsqrt(jnp.sum(r2 * r2) + EPSILON * EPSILON)
    diff_r = r1 * inv_n1 - r2 * inv_n2
    r1_x_r2 = jnp.cross(r1, r2)
    r0_sq = jnp.sum(r0 * r0)
    denom = jnp.sum(r1_x_r2 * r1_x_r2) + EPSILON * r0_sq * r0_sq
    return r1_x_r2 * (jnp.dot(r0, diff_r) / (4.0 * jnp.pi * denom))


def biot_savart_cutoff(x: Array, y: Array) -> Array:
    r"""
    Biot-Savart kernel with truncation radius to remove singularity.
    :param x: Target point, ``(3, )``.
    :param y: Filament endpoints, ``(2, 3)``.
    :return: Influence at target point, ``(3, )``.
    """
    r0 = y[1, :] - y[0, :]
    r1 = x - y[0, :]
    r2 = x - y[1, :]

    sm = jnp.inner(r0, r1) / jnp.inner(r0, y[1, :] - y[0, :])
    m = y[0, :] + sm * (y[1, :] - y[0, :])
    r = jnp.linalg.norm(x - m)  # radial distance

    def _kernel_value() -> Array:
        # Compute the standard Biot-Savart kernel, called only if r > R_CUTOFF
        r1_x_r2 = jnp.cross(r1, r2)
        r1_x_r2_unit2 = r1_x_r2 / (jnp.inner(r1_x_r2, r1_x_r2))
        diff_r = make_unit_epsilon(r1) - make_unit_epsilon(r2)
        return r1_x_r2_unit2 / (4.0 * jnp.pi) * jnp.dot(r0, diff_r)

    return cond((r > R_CUTOFF), _kernel_value, lambda: jnp.zeros(3))


def mirror_grid(zeta: Array, mirror_point: Array, mirror_normal: Array) -> Array:
    """
    Mirror a grid of points across a plane defined by a point and a normal vector.
    :param zeta: Grid of points, ``(zeta_m, zeta_n, 3)``.
    :param mirror_point: Point in mirror plane, ``(3, )``.
    :param mirror_normal: Normal vector of mirror plane, ``(3, )``. Should be normalised.
    :return: Mirrored grid of points, ``(zeta_m, zeta_n, 3)``.
    """
    diff = zeta - mirror_point[None, None, :]  # (zeta_m, zeta_n, 3)
    diff_n = jnp.einsum("ijk,k->ij", diff, mirror_normal)  # (zeta_m, zeta_n)
    return (
        zeta - 2.0 * diff_n[:, :, None] * mirror_normal[None, None, :]
    )  # (zeta_m, zeta_n, 3)


def project_forcing_to_beam(
    f_total: ArrayList,
    rmat: Array,
    dof_mapping: ArrayList,
    x0_aero: ArrayList,
) -> Array:
    r"""
    Project aerodynamic forcing at specified time step onto the beam grid. Returned forces are in the global frame.
    :param f_total: Total force on aerodynamic grid, ``(n_surf, )(m+1, n+1, 3)``
    :param rmat: Rotation matrix for each node relative to reference, ``(n_nodes, 3, 3)``.
    :param x0_aero: Reference coordinates for aerodynamic grid, ``(n_surf, )(zeta_m, zeta_n, 3)``.
    :param dof_mapping: Mapping between aero and beam discretisations.
    :return: Steady and unsteady forcing projected onto the beam grid, ``(n_nodes, 6)``
    """

    n_nodes = rmat.shape[0]
    result = jnp.zeros((n_nodes, 6))

    for i_surf in range(len(f_total)):
        # rotate relative distances to get moment arms
        this_rmat = rmat[dof_mapping[i_surf], ...]  # (zeta_n, 3, 3)
        r_x0 = jnp.einsum(
            "ijk,lik->lij", this_rmat, x0_aero[i_surf]
        )  # relative distance (zeta_n, zeta_m, 3)

        result = result.at[dof_mapping[i_surf], :3].set(
            f_total[i_surf].sum(axis=0)
        )  # forcing is sum along strip (zeta_n, 3)
        result = result.at[dof_mapping[i_surf], 3:].set(
            jnp.cross(r_x0, f_total[i_surf]).sum(axis=0)
        )  # moment is cross(r, f) summed along strip (zeta_n, 3)
    return result


def cs_ang_to_cs_vel(cs_ang_t: dict[str, Array], dt: float | Array) -> dict[str, Array]:
    r"""
    Approximate control surfaces velocities from the time series of their angles using finite differences.
    :param cs_ang_t: Time history of control surface angle, ``{name, (n_tstep, )}``.
    :param dt: Time step length.
    :return: Control surface velocity, ``{name, (n_tstep, )}``.
    """
    cs_vel_t = {}
    for k, v in cs_ang_t.items():
        n_tstep = v.shape[0]
        cs_vel_t[k] = vmap(
            lambda i_ts: finite_difference(
                i_=i_ts,
                data=v,  # noqa: B023
                delta=jnp.array(dt),
                axis=0,
            ),
            in_axes=0,
            out_axes=0,
        )(jnp.arange(n_tstep))
    return cs_vel_t


def cs_vel_to_cs_ang(cs_vel_t: dict[str, Array], dt: float | Array) -> dict[str, Array]:
    r"""
    Approximate control surfaces angles from the time series of their velocities using finite differences.
    :param cs_vel_t: Time history of control surface velocity, ``{name, (n_tstep, )}``.
    :param dt: Time step length.
    :return: Control surface angle, ``{name, (n_tstep, )}``.
    """
    cs_ang_t = {}
    for k, v in cs_vel_t.items():
        cs_ang_t[k] = jnp.cumsum(v, axis=0) * dt
    return cs_ang_t


class DynamicAeroSolver(Protocol):
    r"""
    Required methods for a dynamic aerodynamic solver that can be coupled into the structural solver.
    """

    include_unsteady_force: bool

    @property
    def zeta_b0(self) -> ArrayList: ...

    def case_solve(
        self,
        case: AeroCase,
        i_ts: int,
        hg_n: Array | None,
        hg_nm1: Array | None,
        hg_dot_n: Array | None,
        static: bool,
        horseshoe: bool,
        cs_ang_n: dict[str, Array],
        cs_ang_nm1: dict[str, Array] | None,
        cs_vel_n: dict[str, Array] | None,
    ) -> AeroCase: ...
