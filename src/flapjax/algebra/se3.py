import jax
from jax import Array
from jax import numpy as jnp
from jax.lax import cond

from flapjax.algebra.base import chi, exp_sum, log_sum, matrix2, t_inv_sum, t_sum
from flapjax.algebra.so3 import (
    alpha,
    beta,
    bracket_neg_so3,
    exp_so3,
    log_so3,
    skew_to_vec,
    t_inv_so3,
    t_so3,
    vec_to_skew,
)
from flapjax.utils.constants import BASE_SUMMATION_ORDER, SMALL_ANG_THRESH


def bracket_se3(a_vec: Array, b_vec: Array) -> Array:
    r"""
    Computes the Lie bracket of two se(3) elements, :math:`\tilde{a}\tilde{b} - \tilde{b}\tilde{a}`.
    :param a_vec: Lie algebra vector in se(3), ``(6, )``.
    :param b_vec: Lie algebra vector in se(3), ``(6, )``.
    :return: Lie bracket, ``(4, 4)``.
    """
    mat1 = ha_to_ha_tilde(a_vec)
    mat2 = ha_to_ha_tilde(b_vec)

    return mat1 @ mat2 - mat2 @ mat1


def bracket_neg_se3(a_vec: Array, b_vec: Array) -> Array:
    r"""
    Computes the negative Lie bracket of two se(3) elements, :math:`\tilde{a}\tilde{b} + \tilde{b}\tilde{a}`.
    :param a_vec: Lie algebra vector in se(3), ``(6, )``.
    :param b_vec: Lie algebra vector in se(3), ``(6, )``.
    :return: Negative lie bracket, ``(4, 4)``.
    """
    mat1 = ha_to_ha_tilde(a_vec)
    mat2 = ha_to_ha_tilde(b_vec)

    return mat1 @ mat2 + mat2 @ mat1


def t_u_omega_plus(ha: Array) -> Array:
    r"""
    Computes the :math:`\mathbf{T}_{U \omega+}` matrix, used for computing the tangent application for SE(3). Formulation
    from Geometrically exact beam finite element formulated on the special Euclidean group SE(3), by Sonneville et al., 2013, Eq A.12
    :param ha: Vector in se(3), ``(6, )``.
    :return: Operator, ``(3, 3)``.
    """
    # (6,) -> (3, 3)
    a = ha[:3]
    b = ha[3:]
    b_norm2 = jnp.inner(b, b)

    def t_u_omega_plus_full() -> Array:
        r"""Full computation of :math:`\mathbf{T}_{U \omega+}` for non-small angles."""
        alpha_ = alpha(b)
        beta_ = beta(b)

        return (
            -0.5 * beta_ * vec_to_skew(a)
            + (1.0 - alpha_) / b_norm2 * bracket_neg_so3(a, b)
            + jnp.inner(b, a)
            / b_norm2
            * (
                (beta_ - alpha_) * vec_to_skew(b)
                + (0.5 * beta_ - 3.0 * (1.0 - alpha_) / b_norm2)
                * matrix2(vec_to_skew(b))
            )
        )

    def t_u_omega_plus_small_angle() -> Array:
        r"""Computation of :math:`\mathbf{T}_{U \omega+}` when the rotation angle is small."""
        return -0.5 * vec_to_skew(a)

    return cond(
        b_norm2 > SMALL_ANG_THRESH, t_u_omega_plus_full, t_u_omega_plus_small_angle
    )


def t_u_omega_minus(ha: Array) -> Array:
    r"""
    Computes the :math:`\mathbf{T}_{U \omega-}` matrix, used for computing the inverse tangent application for se(3).
    Formulation from Geometrically exact beam finite element formulated on the special Euclidean group SE(3), by
    Sonneville et al., 2013, Eq A.14. This can be represented in terms of :math:`\mathbf{T}_{U \omega+}` and the
    inverse of the SO(3) tangent operator.
    :param ha: Vector in se(3), ``(6, )``
    :return: Operator, ``(3, 3)``.
    """

    t_inv_ = t_inv_so3(ha[3:])
    return -t_inv_ @ t_u_omega_plus(ha) @ t_inv_


def t_se3(ha: Array) -> Array:
    r"""
    Computes the tangent operator for se(3). Formulation from Geometrically exact beam finite element formulated on the
    special Euclidean group SE(3), by Sonneville et al., 2013, Eq A.11.
    :param ha: se(3) vector, ``(6, )``.
    :return: Tangent operator, ``(6, 6)``.
    """

    def t_se3_full() -> Array:
        t_ = t_so3(ha[3:])
        return jnp.block([[t_, t_u_omega_plus(ha)], [jnp.zeros((3, 3)), t_]])

    def t_se3_small_angle() -> Array:
        return t_sum(ha_to_ha_hat(ha), BASE_SUMMATION_ORDER)

    ang_mag2 = jnp.inner(ha[3:], ha[3:])
    return cond(ang_mag2 > SMALL_ANG_THRESH, t_se3_full, t_se3_small_angle)


def t_inv_se3(ha: Array) -> Array:
    r"""
    Computes the inverse tangent operator for se(3). Formulation from Geometrically exact beam finite element formulated
    on the special Euclidean group SE(3), by Sonneville et al., 2013, Eq A.13.
    :param ha: se(3) algebra vector, ``(6, )``.
    :return: Inverse angent operator, ``(6, 6)``.
    """

    def t_inv_se3_full() -> Array:
        t_ = t_inv_so3(ha[3:])
        return jnp.block([[t_, t_u_omega_minus(ha)], [jnp.zeros((3, 3)), t_]])

    def t_inv_se3_small_angle() -> Array:
        return t_inv_sum(ha_to_ha_hat(ha), BASE_SUMMATION_ORDER)

    ang_mag2 = jnp.inner(ha[3:], ha[3:])
    return cond(ang_mag2 > SMALL_ANG_THRESH, t_inv_se3_full, t_inv_se3_small_angle)


def log_se3(hg: Array) -> Array:
    r"""
    Computes the logarithm map from SE(3) to se(3). Formulation from Geometrically exact beam finite element formulated
    on the special Euclidean group SE(3), by Sonneville et al., 2013, Eq A.15.
    :param hg: SE(3) group element, ``(4, 4)``.
    :return: se(3) algebra vector, ``(6, )``.
    """

    omega = log_so3(hg[:3, :3])

    def log_se3_full() -> Array:
        return jnp.concatenate((t_inv_so3(omega).T @ hg[:3, 3], omega))

    def log_se3_small_angle() -> Array:
        return ha_tilde_to_ha(log_sum(hg, BASE_SUMMATION_ORDER))

    ang_mag2 = jnp.inner(omega, omega)
    return cond(ang_mag2 > SMALL_ANG_THRESH, log_se3_full, log_se3_small_angle)


def exp_se3(ha: Array) -> Array:
    r"""
    Computes the exponential map from se(3) to SE(3). Formulation from Geometrically exact beam finite element formulated
    on the special Euclidean group SE(3), by Sonneville et al., 2013, Eq A.10.
    :param ha: se(3) algebra vector, ``(6, )``.
    :return: SE(3) group element, ``(4, 4)``.
    """

    def exp_se3_full() -> Array:
        return jnp.block(
            [
                [exp_so3(ha[3:]), (t_so3(ha[3:]).T @ ha[:3])[:, None]],
                [jnp.zeros((1, 3)), jnp.ones((1, 1))],
            ]
        )

    def exp_se3_small_angle() -> Array:
        return exp_sum(ha_to_ha_tilde(ha), BASE_SUMMATION_ORDER)

    ang_mag2 = jnp.inner(ha[3:], ha[3:])
    return cond(ang_mag2 > SMALL_ANG_THRESH, exp_se3_full, exp_se3_small_angle)


def x_rmat_to_hg(x: Array, rmat: Array) -> Array:
    r"""
    Combines a translation vector and rotation matrix into an element of the SE(3) group.
    :param x: Translation vector, ``(3, )``.
    :param rmat: Rotation matrix, ``(3, 3)``.
    :return: SE(3) group element, ``(4, 4)``.
    """
    return jnp.block([[rmat, x[:, None]], [jnp.zeros((1, 3)), jnp.ones((1, 1))]])


def x_rmat_to_ha(x: Array, rmat: Array) -> Array:
    r"""
    Combines a translation vector and rotation matrix into an element of the se(3) algebra.
    :param x: Translation vector, ``(3, )``.
    :param rmat: Rotation matrix, ``(3, 3)``.
    :return: se(3) algebra vector, ``(6, )``.
    """
    return log_se3(x_rmat_to_hg(x, rmat))


def hg_to_x_rmat(hg: Array) -> tuple[Array, Array]:
    r"""
    Decomposes an SE(3) group element into a translation vector and rotation matrix.
    :param hg: SE(3) group element, ``(4, 4)``.
    :return: Translation vector, ``(3, )``, and rotation matrix, ``(3, 3)``.
    """
    return hg[:3, 3], hg[:3, :3]


def vect_product(hg: Array, x: Array) -> Array:
    r"""
    Computes the resulting vector of an SE(3) group element and a 3D translation vector.
    :param hg: SE(3) group element, ``(4, 4)``.
    :param x: Translation vector, ``(3, )``.
    :return: Resulting translation vector, ``(3, )``.
    """
    return hg[:3, :3] @ x + hg[:3, 3]


def hg_inv(hg: Array) -> Array:
    r"""
    Computes the inverse of an SE(3) group element.
    :param hg: SE(3) group element, ``(4, 4)``.
    :return: Inverse SE(3) group element, ``(4, 4)``.
    """
    x, rmat = hg_to_x_rmat(hg)
    return jnp.block(
        [[rmat.T, -(rmat.T @ x)[:, None]], [jnp.zeros((1, 3)), jnp.ones((1, 1))]]
    )


def ha_to_ha_tilde(ha: Array) -> Array:
    r"""
    Converts a se(3) vector into its matrix representation.
    :param ha: se(3) algebra vector, ``(6, )``.
    :return: se(3) algebra element in matrix form, ``(4, 4)``.
    """
    return jnp.block([[vec_to_skew(ha[3:]), ha[:3, None]], [jnp.zeros((1, 4))]])


def ha_tilde_to_ha(ha_tilde: Array) -> Array:
    r"""
    Converts a se(3) element matrix into its vector representation.
    :param ha_tilde: se(3) algebra element in matrix form, ``(4, 4)``.
    :return: se(3) algebra vector, ``(6, )``.
    """
    ha_u = ha_tilde[:3, 3]
    ha_omega = skew_to_vec(ha_tilde[:3, :3])
    return jnp.concatenate((ha_u, ha_omega), axis=-1)


def ha_to_ha_hat(ha: Array) -> Array:
    r"""
    Converts a se(3) vector into its hat matrix representation. Formulation from Geometrically exact beam finite element
    formulated on the special Euclidean group SE(3), by Sonneville et al., 2013, Eq 15.
    :param ha: se(3) algebra vector, ``(6, )``.
    :return: Hat matrix representation, ``(6, 6)``.
    """
    return jnp.block(
        [
            [vec_to_skew(ha[3:]), vec_to_skew(ha[:3])],
            [jnp.zeros((3, 3)), vec_to_skew(ha[3:])],
        ]
    )


def rmat_to_ha_hat(rmat: Array) -> Array:
    r"""
    Converts a rotation matrix into its hat matrix representation in se(3) with zero translation. Formulation from
    "A geometric local frame approach for flexible multibody systems", by Sonneville, 2015, Eq 1.50, p. 15.
    :param rmat: Rotation matrix, ``(3, 3)``.
    :return: Hat matrix representation, ``(6, 6)``.
    """
    return chi(rmat)


def ha_hat_to_ha(ha_hat: Array) -> Array:
    r"""
    Converts a se(3) hat matrix representation into se(3) vector. Formulation from Geometrically exact beam finite
    element formulated on the special Euclidean group SE(3), by Sonneville et al., 2013, Eq 15.
    :param ha_hat: Hat matrix representation, ``(6, 6)``.
    :return: se(3) vector, ``(6, )``.
    """
    ha_u = skew_to_vec(ha_hat[:3, 3:])
    ha_omega = skew_to_vec(ha_hat[:3, :3])
    return jnp.concatenate((ha_u, ha_omega), axis=-1)


def ha_to_ha_check(ha: Array) -> Array:
    r"""
    Converts a se(3) vector into its check matrix representation. Formulation from "Geometrically exact beam finite element formulated
    on the special Euclidean group SE(3)", by Sonneville, 2014, Eq 16.
    :param ha: se(3) algebra vector, ``(6, )``.
    :return: Check matrix representation, ``(6, 6)``.
    """
    return -jnp.block(
        [
            [jnp.zeros((3, 3)), vec_to_skew(ha[:3])],
            [vec_to_skew(ha[:3]), vec_to_skew(ha[3:])],
        ]
    )


def hg_to_ha_hat(hg: Array) -> Array:
    r"""
    Converts an SE(3) group element into its se(3) hat matrix representation.
    :param hg: SE(3) group element, ``(4, 4)``.
    :return: se(3) hat matrix representation, ``(6, 6)``.
    """
    rmat = hg[:3, :3]
    x = hg[:3, 3]

    return jnp.block(
        [
            [rmat, vec_to_skew(x) @ rmat],
            [jnp.zeros((3, 3)), rmat],
        ]
    )


def hg_to_d(hg1: Array, hg2: Array) -> Array:
    r"""
    Obtains the relative configuration vector between two SE(3) group elements. Formulation from Geometrically exact
    beam finite element formulated on the special Euclidean group SE(3), by Sonneville et al., 2013, Eq 56.
    :param hg1: Base SE(3) group element at s=0, ``(4, 4)``.
    :param hg2: Tip SE(3) group element at s=L, ``(4, 4)``.
    :return: se(3) relative configuration vector, ``(6, )``.
    """
    return log_se3(hg_inv(hg1) @ hg2)


def p(
    d: Array,
    ad_inv: Array,
) -> Array:
    r"""
    Computes the :math:`\mathbf{P}(\mathbf{d}) = \frac{d \mathbf{d}}{d \mathbf{h}_{AB}}` matrix. Formulation
    from "A geometric local frame approach for flexible multibody systems", by Sonneville, 2015, Eq 6.141, p. 90.
    :param d: Relative se(3) configuration vector, ``(6, )``.
    :param ad_inv: Adjoint action for rotation, ``(6, 6)``.
    :return: Matrix, ``(6, 12)``.
    """

    t_inv_pos = t_inv_se3(d)
    t_inv_neg = t_inv_pos - ha_to_ha_hat(d)  # t_inv(-d)

    return jnp.concatenate((-t_inv_neg @ ad_inv, t_inv_pos @ ad_inv), axis=1)


def t_star(s_l: Array, d: Array) -> Array:
    r"""
    Matrix which described perturbations in the algebra element along an element with respect to the algebra elements at
    both ends of the element, :math:`T^*(s, \mathbf{d}) = \frac{d \mathbf{h}(s)}{d \mathbf{h}_A}` or
    :math:`\frac{d \mathbf{h}(s)}{d \mathbf{h}_B}`. Formulation from Geometrically exact beam finite element
    formulated on the special Euclidean group SE(3), by Sonneville et al., 2013, Eq 70.
    :param s_l: Relative position along the element :math:`\frac{s}{l0} \in [0, 1]`, ().
    :param d: Relative se(3) configuration vector, ``(6, )``.
    :return: :math:`T^*(s, \mathbf{d})` matrix, ``(6, 6)``.
    """
    return s_l * t_se3(s_l * d) @ t_inv_se3(d)


def q(s_l: Array, d: Array, ad_inv: Array) -> Array:
    r"""
    Matrix which described perturbations in the algebra element along an element with respect to the algebra elements at
    both ends of the element, :math:`Q(s, \mathbf{d}) = [\mathbf{I}_{6 \times 6} - T^*(s, \mathbf{d}) &
    T^*(s, \mathbf{d})]`. Formulation from "A geometric local frame approach for flexible multibody systems",
    by Sonneville, 2015, Eq 6.145, p. 90.
    :param s_l: Relative position along the element :math:`\frac{s}{l0} \in [0, 1]`, ().
    :param d: Relative se(3) configuration vector, ``(6, )``.
    :param ad_inv: Adjoint action for base rotation, ``(6, 6)``.
    :return: :math:`Q(s, \mathbf{d})` matrix, ``(6, 12)``.
    """
    t_star_ = t_star(s_l, d)

    return jnp.concatenate(((jnp.eye(6) - t_star_) @ ad_inv, t_star_ @ ad_inv), axis=1)


def q_dot(s_l: Array, d: Array, d_dot: Array, ad_inv: Array) -> Array:
    r"""
    Time derivative of the matrix which described perturbations in the algebra element along an element with respect to
    the algebra elements at both ends of the element, :math:`\dot{Q}(s, \mathbf{d})`.
    :param s_l: Relative position along the element :math:`\frac{s}{l0} \in [0, 1]`, ().
    :param d: Relative se(3) configuration vector, ``(6, )``.
    :param d_dot: Time derivative of relative se(3) configuration vector, ``(6, )``.
    :param ad_inv: Adjoint action for base rotation, ``(6, 6)``.
    :return: Time derivative of :math:`Q(s, \mathbf{d})` matrix, ``(6, 12)``.
    """

    _, tangents = jax.jvp(
        lambda d_: q(s_l, d_, ad_inv),
        primals=[d],
        tangents=[d_dot],
    )

    return tangents
