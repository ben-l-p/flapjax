from jax import Array
from jax import numpy as jnp
from jax.lax import cond

from flapjax.algebra.base import clip_to_pi, log_sum, matrix2, t_inv_sum, t_sum
from flapjax.utils.constants import SMALL_ANG_THRESH


def vec_to_skew(vec: Array) -> Array:
    r"""
    Converts a 3D vector to a skew-symmetric matrix.
    :param vec: 3D vector, ``(3, )``.
    :return: Skew-symmetric matrix, ``(3, 3)``
    """
    return jnp.array(
        ((0.0, -vec[2], vec[1]), (vec[2], 0.0, -vec[0]), (-vec[1], vec[0], 0.0))
    )


def skew_to_vec(mat: Array) -> Array:
    r"""
    Converts a skew-symmetric matrix to a 3D vector. Note this refers to both skew symmetric entries for consistent
    gradients.
    :param mat: Skew-symmetric matrix, ``(3, 3)``
    :return: 3D vector, ``(3, )``.
    """
    a1 = mat[2, 1] - mat[1, 2]
    a2 = mat[0, 2] - mat[2, 0]
    a3 = mat[1, 0] - mat[0, 1]

    return 0.5 * jnp.array((a1, a2, a3))


def _safe_norm(b: Array) -> Array:
    r"""
    Norm of ``b`` that gives 1.0 for a zero-length vector, keeping the derivative defined.
    :param b: Input vector, ``(3, )``.
    :return: Norm of ``b``, ``()``.
    """
    b_norm2 = jnp.inner(b, b)
    return jnp.sqrt(jnp.where(b_norm2 > 0, b_norm2, 1.0))


def alpha(b: Array) -> Array:
    r"""
    Computes the alpha function for SO(3) operations, :math:`\alpha(\mathbf{b}) = \sin(\|\mathbf{b}\|) / \|\mathbf{b}\|`.
    Formulation from Geometrically exact beam finite element formulated on the special Euclidean group SE(3), by
    Sonneville et al., 2013, Eq A.4. Expressed with the sinc function to keep it valid even as the angle approaches zero.
    :param b: Input vector, ``(3, )``.
    :return: Alpha value, ().
    """
    return jnp.sinc(_safe_norm(b) / jnp.pi)


def beta(b: Array) -> Array:
    r"""
    Computes the beta function for SO(3) operations, :math:`\beta(\mathbf{b}) = 2(1 - \cos(\|\mathbf{b}\|)) /
    \|\mathbf{b}\|^2`. Formulation from Geometrically exact beam finite element formulated on the special Euclidean
    group SE(3), by Sonneville et al., 2013, Eq A.4. Expressed with the sinc function to keep it valid even as the angle
    approaches zero.
    :param b: Input vector, ``(3, )``.
    :return: Beta value, ()
    """
    return jnp.sinc(_safe_norm(b) / (2.0 * jnp.pi)) ** 2


def bound_h_omega(h_omega: Array) -> Array:
    r"""
    Bounds the angle of a rotation vector to be within `[-pi, pi]`.
    :param h_omega: Cartesian rotation vector, ``(3, )``.
    :return: Bounded Cartesian rotation vector, ``(3, )``.
    """
    ang = jnp.linalg.norm(h_omega)

    def nonzero_ang() -> Array:
        n = h_omega / ang
        bounded_ang = clip_to_pi(ang)
        return bounded_ang * n

    def small_ang() -> Array:
        return h_omega

    return cond(ang > SMALL_ANG_THRESH, nonzero_ang, small_ang)


def bracket_so3(vec1: Array, vec2: Array) -> Array:
    r"""
    Computes the Lie bracket of two so(3) elements represented as vectors.
    :param vec1: Vector 1, ``(3, )``.
    :param vec2: Vector 2, ``(3, )``.
    :return: Lie bracket, ``(3, 3)``.
    """
    mat1 = vec_to_skew(vec1)
    mat2 = vec_to_skew(vec2)

    return mat1 @ mat2 - mat2 @ mat1


def bracket_neg_so3(vec1: Array, vec2: Array) -> Array:
    r"""
    Computes the negative Lie bracket of two so(3) elements represented as vectors.
    :param vec1: Vector 1, ``(3, )``
    :param vec2: Vector 2, ``(3, )``
    :return: Negative Lie bracket, ``(3, 3)``
    """
    mat1 = vec_to_skew(vec1)
    mat2 = vec_to_skew(vec2)

    return mat1 @ mat2 + mat2 @ mat1


def t_so3(ha_omega: Array) -> Array:
    r"""
    Computes the tangent operator for SO(3) given a rotation vector. Includes a small angle approximation as the
    rotation approaches zero. Formulation from Geometrically exact beam finite element formulated on the special
    Euclidean group SE(3), by Sonneville et al., 2013, Eq A.6
    :param ha_omega: Rotation vector, ``(3, )``.
    :return: Tangent operator, ``(3, 3)``
    """

    def t_so3_full() -> Array:
        return (
            jnp.eye(3)
            - 0.5 * beta(ha_omega) * vec_to_skew(ha_omega)
            + (1.0 - alpha(ha_omega))
            / jnp.inner(ha_omega, ha_omega)
            * matrix2(vec_to_skew(ha_omega))
        )

    def t_so3_small_angle() -> Array:
        return t_sum(vec_to_skew(ha_omega), 2)

    ang_mag2 = jnp.inner(ha_omega, ha_omega)
    return cond(ang_mag2 > SMALL_ANG_THRESH, t_so3_full, t_so3_small_angle)


def t_inv_so3(ha_omega: Array) -> Array:
    r"""
    Computes the inverse tangent operator for SO(3) given a rotation vector. Includes a small angle approximation as the
    rotation approaches zero. Formulation from Geometrically exact beam finite element formulated on the special
    Euclidean group SE(3), by Sonneville et al., 2013, Eq A.7
    :param ha_omega: Rotation vector, ``(3, )``
    :return: Inverse tangent operator, ``(3, 3)``
    """

    def t_inv_so3_full() -> Array:
        return (
            jnp.eye(3)
            + 0.5 * vec_to_skew(ha_omega)
            + (1.0 - alpha(ha_omega) / beta(ha_omega))
            / jnp.inner(ha_omega, ha_omega)
            * matrix2(vec_to_skew(ha_omega))
        )

    def t_inv_so3_small_angle() -> Array:
        return t_inv_sum(vec_to_skew(ha_omega), 2)

    ang_mag2 = jnp.linalg.norm(ha_omega)
    return cond(ang_mag2 > SMALL_ANG_THRESH, t_inv_so3_full, t_inv_so3_small_angle)


def exp_so3(ha_omega: Array) -> Array:
    r"""
    Computes the exponential map from so(3) to SO(3) given a rotation vector.
    :param ha_omega: Rotation vector, ``(3, )``
    :return: Rotation matrix, ``(3, 3)``
    """
    return (
        jnp.eye(3)
        + alpha(ha_omega) * vec_to_skew(ha_omega)
        + 0.5 * beta(ha_omega) * matrix2(vec_to_skew(ha_omega))
    )


def log_so3(rmat: Array) -> Array:
    r"""
    Computes the logarithmic map from SO(3) to so(3) given a rotation matrix. Includes a small angle approximation as
    the angle approaches zero.
    :param rmat: Rotation matrix, ``(3, 3)``
    :return: Rotation vector, ``(3, )``.
    """

    cos_theta = 0.5 * (jnp.trace(rmat) - 1.0)

    def log_so3_full() -> Array:
        # acos is only computed here, where theta is large enough that its gradient is finite
        theta = jnp.acos(cos_theta)
        return skew_to_vec(theta / (2.0 * jnp.sin(theta)) * (rmat - rmat.T))

    def log_so3_small_angle() -> Array:
        return skew_to_vec(log_sum(rmat, 2))

    return cond(
        cos_theta < jnp.cos(SMALL_ANG_THRESH), log_so3_full, log_so3_small_angle
    )
