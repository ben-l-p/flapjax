from collections.abc import Sequence

import jax
from jax import Array, vmap
from jax import numpy as jnp

from flapjax.aero.utils import KernelFunction, mirror_grid
from flapjax.algebra.array_utils import ArrayList, block_axis


def compute_aic_grid(
    c: Array,
    n: Array | None,
    zeta: Array,
    kernel: KernelFunction,
    batch_size: int | None,
) -> Array:
    """
    Compute the aerodynamic influence coefficient (AIC) across grids of points. When normal is provided, fuses the dot
    product inside each map step so the trailing 3-component axis is never accumulated, saving memory.
    :param c: Collocation points, ``(c_m, c_n, 3)``.
    :param n: Normal vectors at collocation points, ``(c_m, c_n, 3)``, or None.
    :param zeta: Grid vertices, ``(zeta_m, zeta_n, 3)``.
    :param kernel: Kernel function to compute the influence.
    :param batch_size: Batch size for vectorising AIC computations.
    :return: ``(c_m, c_n, zeta_m, zeta_n, 3)`` if normal is None, else ``(c_m, c_n, zeta_m, zeta_n)``.
    """
    c_m, c_n = c.shape[:2]
    m_panels, n_panels = zeta.shape[0] - 1, zeta.shape[1] - 1

    m_vect_flat = jnp.stack((zeta[:-1, :, :], zeta[1:, :, :]), axis=-2).reshape(
        -1, 2, 3
    )
    n_vect_flat = jnp.stack((zeta[:, :-1, :], zeta[:, 1:, :]), axis=-2).reshape(
        -1, 2, 3
    )

    # account for the degenerate case where there are no source panels to prevent division by zero
    if not c_m or not c_n or not m_panels or not n_panels:
        return jnp.zeros((c_m, c_n, m_panels, n_panels))

    @jax.checkpoint
    def row(args: tuple) -> Array:
        # compute the influence of all spanwise (m) and chordwise (n) filaments before combining. This prevents any
        # duplicate computations.
        ci, ni = args
        m_influence = vmap(kernel, (None, 0), 0)(ci, m_vect_flat)
        m_influence_ni = jnp.dot(m_influence, ni).reshape(
            m_panels, n_panels + 1
        )  # [m, n+1]
        n_influence = vmap(kernel, (None, 0), 0)(ci, n_vect_flat)
        n_influence_ni = jnp.dot(n_influence, ni).reshape(
            m_panels + 1, n_panels
        )  # [m+1, n]
        return -jnp.diff(m_influence_ni, axis=1) + jnp.diff(
            n_influence_ni, axis=0
        )  # [m, n]

    return jax.lax.map(
        row,
        (c.reshape(-1, 3), n.reshape(-1, 3) if n is not None else None),
        batch_size=batch_size,
    ).reshape(c_m, c_n, m_panels, n_panels)


def compute_aic_sys(
    zetas: ArrayList,
    cs: ArrayList,
    ns: ArrayList,
    kernels: Sequence[KernelFunction],
    batch_size: int | None,
    mirror_point: Array | None,
    mirror_normal: Array | None,
) -> list[list[Array]]:
    """
    Compute the AIC matrix for a system of elements. Returns a list of AIC matrices, one for each element.
    :param zetas: List of source points to compute the AIC from, ``(n_source,)(zeta_m, zeta_n, 3)``.
    :param cs: List of target points to compute the AIC at, ``(n_target,)(c_m, c_n, 3)``.
    :param ns: Bound normal vectors, ``(c_m, c_n, 3)``. If None, no projection will be done.
    :param kernels: List of kernel functions to use for each source surface, ``(n_source, )``.
    :param batch_size: Batch size for vectorising AIC computations.
    :param mirror_normal: Normal vector to mirror across, ``(3, )``. If None, no mirroring will be done.
    :param mirror_point: Point on mirror plane, ``(3, )``. If None, no mirroring will be done.
    :return: Nested sequences of AIC matrices, ``(n_target,)(n_source, c_m, c_n, zeta_m, zeta_n, 3)``, or
    ``(n_target,)(n_source,)(c_m, c_n, zeta_m, zeta_n)`` if projected onto normals.
    """

    aic_mats = []
    for c, n in zip(cs, ns):
        aic_mats.append([])
        for zeta, kernel in zip(zetas, kernels):
            # compute the AIC matrix, [n_cx, n_cy, n_ex, n_ey, 3]
            aic_ = compute_aic_grid(
                c=c,
                n=n,
                zeta=zeta,
                kernel=kernel,
                batch_size=batch_size,
            )

            if mirror_point is not None and mirror_normal is not None:
                # add influence from mirrored grid, if specified
                zeta_mirror = mirror_grid(
                    zeta=zeta,
                    mirror_point=mirror_point,
                    mirror_normal=mirror_normal,
                )
                aic_ -= compute_aic_grid(
                    c=c, n=n, zeta=zeta_mirror, kernel=kernel, batch_size=batch_size
                )
            aic_mats[-1].append(aic_)
    return aic_mats


def reshape_aic_sys(aic_mat: Array) -> Array:
    r"""
    Reshape an AIC matrix such that the source and target dimensions are flattened.
    :param aic_mat: Input AIC matrix, ``(c_m, c_n, zeta_m, zeta_n)`` or ``(c_m, c_n, zeta_m, zeta_n, 3)``.
    :return: Reshaped AIC matrix, ``(c_m*c_n, zeta_m*zeta_n)`` or ``(c_m*c_n, zeta_m*zeta_n, 3)``.
    """
    shape = aic_mat.shape
    return aic_mat.reshape([shape[0] * shape[1], shape[2] * shape[3]])


def assemble_aic_sys(aic_mats: Sequence[Sequence[Array]]) -> Array:
    r"""
    Assemble a nested sequence of AIC matrices into a single AIC matrix.
    :param aic_mats: Nested sequence of AIC matrices, ``(n_target,)(n_source,)(c_m, c_n, zeta_m, zeta_n)`` or ``(n_target,)(n_source,)(c_m, c_n, zeta_m, zeta_n, 3)``.
    :return: Assembled AIC matrix, ``(c_tot, zeta_tot)`` or ``(c_tot, zeta_tot, 3)``.
    """
    aic_mats_reshaped = [
        [reshape_aic_sys(aic) for aic in aic_row] for aic_row in aic_mats
    ]
    return block_axis(aic_mats_reshaped, axes=(0, 1))


def compute_aic_solve(
    cs: ArrayList,
    ns: ArrayList,
    zetas_b: ArrayList,
    zetas_w: ArrayList | None,
    kernels_b: Sequence[KernelFunction],
    kernels_w: Sequence[KernelFunction] | None,
    batch_size: int | None,
    mirror_point: Array | None,
    mirror_normal: Array | None,
) -> Array:
    r"""
    Compute the AIC matrix used for the UVLM solve step.
    :param cs: List of target points to compute the AIC at, ``(n_target,)(c_m, c_n, 3)``.
    :param ns: Bound normal vectors, ``(n_target,)(c_m, c_n, 3)``. If None, no projection will be done.
    :param zetas_b: Bound aerodynamic grids, ``(n_source,)(zeta_m, zeta_n, 3)``.
    :param zetas_w: Wake aerodynamic grids, ``(n_source,)(zeta_m_star, zeta_n, 3)``. This is only passed in the static case,
    as in the dynamic case the wake influence is instead included in the boundary conditions.
    :param kernels_b: Bound grid kernels.
    :param kernels_w: Wake grid kernels.
    :param batch_size: Batch size for vectorising AIC computations.
    :param mirror_normal: Normal vector to mirror across, ``(3, )``. If None, no mirroring will be done.
    :param mirror_point: Point on mirror plane, ``(3, )``. If None, no mirroring will be done.
    :return: Square AIC matrix for the solve step, ``(c_tot, zeta_tot)``.
    """
    aic_b_mats = compute_aic_sys(
        cs=cs,
        ns=ns,
        zetas=zetas_b,
        kernels=kernels_b,
        batch_size=batch_size,
        mirror_point=mirror_point,
        mirror_normal=mirror_normal,
    )

    if zetas_w is not None:
        if kernels_w is None:
            raise ValueError("kernels_w must not be None")
        aic_w_mats = compute_aic_sys(
            cs=cs,
            ns=ns,
            zetas=zetas_w,
            kernels=kernels_w,
            batch_size=batch_size,
            mirror_point=mirror_point,
            mirror_normal=mirror_normal,
        )

        aic_b_mats = add_wake_influence(aic_b_mats, aic_w_mats)

    return assemble_aic_sys(aic_b_mats)


def add_wake_influence(
    aic_bs: list[list[Array]], aic_ws: list[list[Array]]
) -> list[list[Array]]:
    r"""
    Lump the wake influence onto the last column of the bound AIC matrices. This captures the steady Kutta condition
    by ensuring that the trailing edge panels have the same strength as all wake panels along a streamline.
    :param aic_bs: Bound influence matrices, ``(n_target,)(n_source,)(c_m, c_n, zeta_m, zeta_n, 3)``.
    :param aic_ws: Wake influence matrices, ``(n_target,)(n_source,)(c_m, c_n, zeta_m_star, zeta_n, 3)``.
    :return: Updated bound influence matrices, ``(n_target,)(n_source,)(c_m, c_n, zeta_m, zeta_n, 3)``.
    """
    for i in range(len(aic_bs)):
        for j in range(len(aic_bs[i])):
            aic_bs[i][j] = (
                aic_bs[i][j].at[:, :, -1, :].add(jnp.sum(aic_ws[i][j], axis=2))
            )
    return aic_bs


def v_ind_vmap(
    c: Array,
    zeta: Array,
    gamma: Array,
    kernel: KernelFunction,
    batch_size: int | None,
) -> Array:
    """
    Compute the induced velocity by the aerodynamic elements at some points in space for a single source-target panel
    system. This is done without materialising the full AIC matrix, instead directly computing its contraction with the
    circulation strength.
    :param c: Points at which to sample the velocity, ``(c_m, c_n, 3)``.
    :param zeta: Filament grid, ``(zeta_m, zeta_n, 2, 3)``.
    :param gamma: Circulation strengths, ``(zeta_m, zeta_n)``.
    :param kernel: Kernel function.
    :param batch_size: Batch size for vectorising AIC computations.
    :return: Induced velocity, ``(c_m, c_n, 3)``.
    """
    c_m, c_n = c.shape[:2]
    c_flat = c.reshape(-1, 3)
    zeta_flat = zeta.reshape(-1, 2, 3)
    gamma_flat = gamma.ravel()  # [zeta_m * zeta_n]

    # account for case where zeta is empty
    if zeta.size == 0:
        return jnp.zeros_like(c)

    @jax.checkpoint
    def row(ci: Array) -> Array:
        influence = vmap(kernel, (None, 0), 0)(ci, zeta_flat)  # [zeta_m * zeta_n, 3]

        if influence.shape[0] != gamma_flat.shape[0]:
            pass

        return jnp.einsum("lm,l->m", influence, gamma_flat)  # [3]

    result = jax.lax.map(row, c_flat, batch_size=batch_size)  # [c_m * c_n, 3]
    return result.reshape(c_m, c_n, 3)


def compute_v_ind[T: Array | ArrayList](
    cs: T,
    zetas: ArrayList,
    gammas: ArrayList,
    kernels: Sequence[KernelFunction],
    mirror_point: Array | None,
    mirror_normal: Array | None,
    batch_size: int | None,
) -> T:
    """
    Compute the induced velocity by multiple surfaces of aerodynamic elements at one or multiple grids of points in
    space. This is done without materialising the full AIC matrix, instead directly computing its contraction with the circulation strength.
    :param cs: Points at which to sample the velocity, ``(c_m, c_n, 3)`` or ``(n_target,)(c_m, c_n, 3)``.
    :param zetas: Filament grid, ``(n_source,)(zeta_m, zeta_n, 3)``.
    :param gammas: Circulation strengths, ``(n_source,)(zeta_m, zeta_n)``.
    :param kernels: Kernel function.
    :param mirror_point: Mirror point, ``(3, )``. If None, no mirroring will be done.
    :param mirror_normal: Normal mirror vector, ``(3, )``. If None, no mirroring will be done.
    :param batch_size: Batch size for vectorising AIC computations.
    :return: Array or ArrayList of induced velocity, ``(c_m, c_n, 3)`` or ``(n_target,)(c_m, c_n, 3)``.
    """

    # convert cs to an ArrayList. If it is an Array, we will convert back before returning.
    cs_: ArrayList = ArrayList([cs]) if isinstance(cs, Array) else cs

    v = ArrayList([])
    for c in cs_:
        v.append(jnp.zeros_like(c))
        for zeta, gamma, kernel in zip(zetas, gammas, kernels):
            m_vect = jnp.stack(
                (zeta[:-1, :, :], zeta[1:, :, :]), axis=-2
            )  # [m, n+1, 2, 3]
            n_vect = jnp.stack(
                (zeta[:, :-1, :], zeta[:, 1:, :]), axis=-2
            )  # [m+1, n, 2, 3]

            gamma_eff_m = jnp.diff(jnp.pad(gamma, ((0, 0), (1, 1))), axis=1)  # [m, n+1]
            gamma_eff_n = -jnp.diff(
                jnp.pad(gamma, ((1, 1), (0, 0))), axis=0
            )  # [m+1, n]

            v[-1] += v_ind_vmap(
                c, m_vect, gamma_eff_m, kernel, batch_size
            ) + v_ind_vmap(c, n_vect, gamma_eff_n, kernel, batch_size)

            if mirror_point is not None and mirror_normal is not None:
                zeta_mirror = mirror_grid(
                    zeta=zeta,
                    mirror_point=mirror_point,
                    mirror_normal=mirror_normal,
                )
                m_vect_mirror = jnp.stack(
                    (zeta_mirror[:-1, :, :], zeta_mirror[1:, :, :]), axis=-2
                )  # [m, n+1, 2, 3]
                n_vect_mirror = jnp.stack(
                    (zeta_mirror[:, :-1, :], zeta_mirror[:, 1:, :]), axis=-2
                )  # [m+1, n, 2, 3]

                v[-1] -= v_ind_vmap(
                    c, m_vect_mirror, gamma_eff_m, kernel, batch_size
                ) + v_ind_vmap(c, n_vect_mirror, gamma_eff_n, kernel, batch_size)

    return v[0] if isinstance(cs, Array) else v
