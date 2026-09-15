from __future__ import annotations

from typing import Literal

import jax
from jax import Array
from jax import numpy as jnp

from flapjax.algebra.integration import gauss_legendre, gauss_lobatto
from flapjax.algebra.se3 import ha_to_ha_check, ha_to_ha_hat, p, q, q_dot


def _split_connectivity(
    conn: list[list[int]], node_i: int, node_j: int
) -> list[list[int]]:
    r"""
    Split connectivity at ``node_i``: the first element containing ``node_i``
    keeps it, the second has ``node_i`` replaced by ``node_j``. Used for automatically adding constraint nodes.
    """
    touching = [i for i, (a, b) in enumerate(conn) if a == node_i or b == node_i]
    if len(touching) < 2:
        return conn
    if len(touching) > 2:
        raise ValueError(
            f"Node {node_i} appears in {len(touching)} elements; "
            f"_split_connectivity only supports nodes shared by exactly 2 elements."
        )
    conn = [list(e) for e in conn]
    e_idx = touching[1]
    if conn[e_idx][0] == node_i:
        conn[e_idx][0] = node_j
    else:
        conn[e_idx][1] = node_j
    return conn


def _check_connectivity(connectivity: Array, num_nodes: int) -> None:
    r"""
    Check connectivity array for validity
    :param connectivity: Connectivity array, ``(n_elem, 2)``
    :param num_nodes: Number of nodes in the structure
    :raises ValueError: If connectivity array contains invalid node indices, or is missing nodes
    """

    if num_nodes != 1:
        all_node_index = set(connectivity.ravel().tolist())
        expected_node_index = set(range(num_nodes))

        if all_node_index != expected_node_index:
            raise ValueError(
                f"Connectivity array either contains invalid node indices, or is missing nodes. Expected "
                f"indices: {expected_node_index}, but got: {all_node_index}."
            )
    else:
        if connectivity.size != 0:
            raise ValueError(
                f"Connectivity array must be empty for a structure with a single node, but got: {connectivity}."
            )


def _n_elem_per_node(connectivity: Array, n_nodes: int) -> Array:
    r"""
    Computes the number of elements connected to each node in the structure.
    :param connectivity: Connectivity array, ``(n_elem, 2)``.
    :return: Number of elements connected to each node, ``(num_nodes, )``.
    """

    return jnp.bincount(
        connectivity.ravel(),
        minlength=connectivity.shape[0],
        length=n_nodes,
    )


def _k_t_entry(
    d: Array,
    p_d: Array,
    length: Array,
    eps: Array,
    k: Array,
    ad_inv: Array,
    include_geometric: bool = True,
) -> Array:
    r"""
    Computes a stiffness matrix entry between two degrees of freedom. Formulation from Geometrically exact beam finite
    element formulated on the special Euclidean group SE(3), by Sonneville et al., 2013, Eq 76.
    :param d: Relative se(3) configuration vector, ``(6, )``.
    :param p_d: :math:`\mathbf{P}(\mathbf{d})` matrix, ``(6, 12)``.
    :param length: Element length, ``()``.
    :param eps: Beam strain vector, ``(6, )``.
    :param k: Beam cross-sectional stiffness matrix, ``(6, 6)``.
    :param ad_inv: Inverse grads matrix for element, ``(6, 6)``.
    :param include_geometric: Whether to include geometric stiffness contribution, bool.
    :return: Stiffness matrix entry, ``(12, 12)``.
    """

    k_t = p_d.T @ k @ p_d / length  # (12, 12)

    if include_geometric:
        e = jax.jacobian(lambda d__,: p(d__, ad_inv))(d)

        # (12, 6, 12)
        f = jnp.einsum("ijk, kl->ijl", e, p_d)

        # (12, 12)
        k_tg = jnp.einsum("jil,j->il", f, (k * length) @ eps)
        k_t += k_tg
    return k_t


def _integrate_m_l(
    m_cs: Array,
    d: Array,
    ad_inv: Array,
    length: Array,
    int_order: Literal[3, 4, 5],
) -> Array:
    r"""
    Approximates the integral :math:`\int_L \mathbf{Q}(s, \mathbf{d})^{\top} \mathcal{M}_{CS} \mathbf{Q}(s, \mathbf{d}) \ ds`
    :param m_cs: Cross sectional mass matrix, ``(6, 6)``.
    :param d: Configuration vector, ``(6, )``.
    :param ad_inv: Inverse grads matrix for element, ``(6, 6)``
    :param length: Element length, ()
    :param int_order: Order of integration, 3, 4, or 5.
    :return: Integrated mass matrix, ``(12, 12)``
    """

    def _inner_func(s_l) -> Array:
        q_mat = q(s_l, d, ad_inv)
        return q_mat.T @ m_cs @ q_mat

    f0 = jnp.zeros((12, 12)).at[:6, :6].set(ad_inv.T @ m_cs @ ad_inv)
    fl = jnp.zeros((12, 12)).at[6:, 6:].set(ad_inv.T @ m_cs @ ad_inv)

    return length * gauss_lobatto(
        _inner_func,
        jnp.array((0.0, 1.0)),
        jnp.stack((f0, fl), axis=0),
        int_order=int_order,
    )


def _integrate_c_t(
    m_cs: Array,
    v_ab: Array,
    d: Array,
    d_dot: Array,
    ad_inv: Array,
    length: Array,
    int_order: Literal[1, 2, 3],
    include_q_dot: bool,
) -> Array:
    r"""
    Approximate the integral :math:`C_T = C^L - \int_L \check{(\mathbf{MQv}_{AB})}^{\top} \mathbf{Q} \ ds` where
    :math:`C^L = \int_L \mathbf{Q}^{\top} ( \mathbf{M}_{cs} \dot{\mathbf{Q}} - \hat{\mathbf{Qv}_{AB}}^{\top}
    \mathbf{M}_{cs} \mathbf{Q} ) \ ds`
    :param m_cs: Cross sectional mass matrix, ``(6, 6)``.
    :param v_ab: Nodal local velocities, ``(12, )``.
    :param d: Configuration vector, ``(6, )``.
    :param d_dot: Configuration velocity vector, ``(6, )``.
    :param ad_inv: Inverse grads matrix for element, ``(6, 6)``
    :param length: Element length, ``()``
    :param int_order: Order of integration, 1, 2, or 3
    :param include_q_dot: Whether to compute the contribution of the derivative of q_dot with respect to velocity.
    :return: Stacked `[C_L, C_T]`, ``(2, 12, 12)``
    """

    if include_q_dot:

        def _g_iner_ab_integrand(s_l: Array, v: Array) -> Array:
            r"""
            Integrand for inertial forcing with zero acceleration
            """
            d_dot_ = p(d, ad_inv) @ v
            q_mat = q(s_l, d, ad_inv)
            q_dot_mat = q_dot(s_l, d, d_dot_, ad_inv)
            return q_mat.T @ (
                m_cs @ q_dot_mat @ v - ha_to_ha_hat(q_mat @ v).T @ m_cs @ q_mat @ v
            )

        def _g_iner_ab(v: Array) -> Array:
            r"""
            Integrate along the beam to find the inertial loads at each end due to velocity.
            """
            return length * gauss_legendre(
                lambda s_l_: _g_iner_ab_integrand(s_l_, v),
                jnp.array((0.0, 1.0)),
                int_order=int_order,
            )  # [12]

        def _c_l_integrand(s_l: Array) -> Array:
            r"""
            Integrand for linear contribution to inertial forcing.
            """
            q_mat = q(s_l, d, ad_inv)
            q_dot_mat = q_dot(s_l, d, d_dot, ad_inv)
            return q_mat.T @ (
                m_cs @ q_dot_mat - ha_to_ha_hat(q_mat @ v_ab).T @ m_cs @ q_mat
            )

        # obtain the tangent stiffness as the Jacobian of the inertial forcing with respect to velocity, which includes
        # the contribution from q_dot
        c_t_ = jax.jacobian(_g_iner_ab, argnums=0)(v_ab)  # (12,12)

        # obtain the linear contribution separately - these are combined when q_dot is omitted for simplicity
        c_l_ = length * gauss_legendre(
            _c_l_integrand,
            jnp.array((0.0, 1.0)),
            int_order=int_order,
        )  # (12, 12)

        return jnp.stack([c_l_, c_t_], axis=0)
    else:

        def _inner_func(s_l) -> Array:
            q_mat = q(s_l, d, ad_inv)
            q_dot_mat = q_dot(s_l, d, d_dot, ad_inv)
            c_l = q_mat.T @ (
                m_cs @ q_dot_mat - ha_to_ha_hat(q_mat @ v_ab).T @ m_cs @ q_mat
            )
            c_t = c_l - q_mat.T @ ha_to_ha_check(m_cs @ q_mat @ v_ab).T @ q_mat
            return jnp.stack((c_l, c_t), axis=0)

        return length * gauss_legendre(
            _inner_func,
            jnp.array((0.0, 1.0)),
            int_order=int_order,
        )  # (2, 12, 12)


def _make_c_t_lumped(m_lumped: Array, v: Array) -> Array:
    r"""
    Construct lumped damping matrix :math:`C_T` from lumped mass matrix :math:`M_{lumped}` and nodal velocities
    :math:`\mathbf{v}`
    :param m_lumped: Lumped mass matrix, ``(6, 6)``
    :param v: Nodal local velocities, (6)
    :return: Stacked gyroscopic matrices, ``(2, 6, 6)``
    """

    def _g_iner_func(v_: Array) -> Array:
        return -ha_to_ha_hat(v_).T @ m_lumped @ v_

    c_l = -ha_to_ha_hat(v).T @ m_lumped  # g_iner = c_l @ v

    c_t = jax.jacobian(_g_iner_func)(v)  # d_{g_iner}/d_v

    return jnp.stack((c_l, c_t), axis=0)


def get_solve_dofs(n_dof: int, prescribed_dofs: tuple[int, ...]) -> tuple[int, ...]:
    r"""
    Obtain the index of degrees of freedom to solve for, given the index of prescribed degrees of freedom.
    :param n_dof: Total number of degrees of freedom
    :param prescribed_dofs: Index of prescribed degrees of freedom
    :return: Index of degrees of freedom to solve for
    """
    return tuple(sorted(set(range(n_dof)) - set(prescribed_dofs)))


def input_dof_index_to_tuple(
    input_dof_index: Array | tuple[int, ...],
) -> tuple[int, ...]:
    if isinstance(input_dof_index, tuple):
        return input_dof_index
    elif isinstance(input_dof_index, Array):
        return tuple(input_dof_index.tolist())
    else:
        raise TypeError("Invalid input dof index")


def transform_nodal_vect(vect: Array, rmat: Array) -> Array:
    r"""
    Rotate a nodal vector quantity.
    :param vect: Nodal vectors, ``(..., n_nodes, 6)``
    :param rmat: Rotation matrix, ``(..., n_nodes, 3, 3)``
    :return: Rotated vectors, ``(..., n_nodes, 6)``
    """

    idx = "...ijk,...ik->...ij"
    vect_lin = jnp.einsum(idx, rmat, vect[..., :3])
    vect_rot = jnp.einsum(idx, rmat, vect[..., 3:])
    return jnp.concatenate((vect_lin, vect_rot), axis=-1)


_OPTIONAL_FORCE_FIELDS: tuple[str, ...] = (
    "f_ext_follower",
    "f_ext_dead",
    "f_ext_aero",
    "f_grav",
)


def apply_frame_transform(obj, rmat: Array, extra_fields: tuple[str, ...] = ()) -> None:
    r"""
    Rotate all force / velocity fields of a structure state object in place.
    """
    for name in _OPTIONAL_FORCE_FIELDS:
        if getattr(obj, name) is not None:
            setattr(obj, name, transform_nodal_vect(getattr(obj, name), rmat))
    for name in ("f_int", "f_res", *extra_fields):
        setattr(obj, name, transform_nodal_vect(getattr(obj, name), rmat))
