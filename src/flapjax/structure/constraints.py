from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import jax
from jax import Array
from jax import numpy as jnp

from flapjax.algebra.se3 import exp_se3, hg_inv, hg_to_d, t_se3
from flapjax.utils.utils import make_pytree


class SoftConstraint(ABC):
    r"""
    Base class for a single-node structural soft constraint (applied without Lagrange multipliers).
    """

    node_index: int  # node where the constraint is applied

    @abstractmethod
    def f_res(self, hg: Array, v: Array, i_ts: int) -> Array:
        r"""
        Compute the forcing residual contribution from the constraint
        :param hg: Node SE(3) coordinate, ``(4, 4)``.
        :param v: Node local velocity, ``(6, )``.
        :param i_ts: Time-step index.
        :return: Nodal force, ``(6, )``.
        """

    @abstractmethod
    def k_tangent(self, hg: Array, i_ts: int) -> Array:
        r"""
        Effective stiffness contribution :math:`-\partial \mathbf{f_{res}}/\partial \boldsymbol{\varphi}`.
        :param hg: Node SE(3) coordinate, ``(4, 4)``.
        :param i_ts: Time-step index.
        :return: Nodal stiffness contribution, ``(6, 6)``.
        """

    def c_tangent(self, hg: Array, i_ts: int) -> Array:
        r"""
        Effective damping contribution :math:`-\partial \mathbf{f_{res}}/\partial \mathbf{v}`.
        :param hg: Node SE(3) coordinate, ``(4, 4)``.
        :param i_ts: Time-step index.
        :return: Nodal damping contribution, ``(6, 6)``. Default implementation returns zero damping.
        """
        return jnp.zeros((6, 6))

    def resolve_hg_ref(self, hg0: Array) -> None:
        r"""
        Called once ``hg0`` is populated so subclasses can default their reference frame to the node's initial pose.
        :param hg0: Full nodal initial-frame array, ``(n_nodes, 4, 4)``.
        """


@make_pytree
class SpringDamper(SoftConstraint):
    r"""
    6-DOF spring-damper attached to a fixed reference frame.
    """

    _static: ClassVar[tuple[str, ...]] = ("node_index", "_hg_ref_from_hg0")

    def __init__(
        self,
        node_index: int,
        k: Array,
        hg_ref: Array | None = None,
        c: Array | None = None,
    ) -> None:
        r"""
        :param node_index: Index of the node to attach to.
        :param k: Stiffness matrix, ``(6, 6)``.
        :param hg_ref: Reference SE(3) frame, ``(4, 4)``, or ``None`` to default
        to the node's initial pose.
        :param c: Damping matrix, ``(6, 6)`` or ``None`` for no damping.
        """
        self.node_index: int = int(node_index)
        self.k: Array = k
        # sentinel: hg_ref is always a (4, 4) Array so the pytree structure is stable.
        # The static flag records whether the beam should overwrite it with hg0[node].
        self._hg_ref_from_hg0: bool = hg_ref is None
        self.hg_ref: Array = jnp.eye(4) if hg_ref is None else hg_ref
        self.c: Array = jnp.zeros((6, 6)) if c is None else c

    def resolve_hg_ref(self, hg0: Array) -> None:
        if self._hg_ref_from_hg0:
            self.hg_ref = hg0[self.node_index]

    def _varphi(self, hg_i: Array) -> Array:
        return hg_to_d(self.hg_ref, hg_i)

    def f_res(self, hg: Array, v: Array, i_ts: int) -> Array:
        return -self.k @ self._varphi(hg) - self.c @ v

    def k_tangent(self, hg: Array, i_ts: int) -> Array:
        return self.k @ t_se3(self._varphi(hg))

    def c_tangent(self, hg: Array, i_ts: int) -> Array:
        return self.c


@make_pytree
class PrescribedMotion(SoftConstraint):
    r"""
    Prescribe the trajectory of a node along a reference SE(3) history. The node is driven through a 6-DOF spring-damper
    to the reference trajectory.
    """

    _static: ClassVar[tuple[str, ...]] = ("node_index",)

    def __init__(
        self,
        node_index: int,
        k: Array,
        hg_ref_t: Array,
        c: Array | None = None,
    ) -> None:
        r"""
        :param node_index: Index of the node to drive.
        :param k: Tracking stiffness, ``(6, 6)``.
        :param hg_ref_t: Reference SE(3) trajectory, ``(n_tstep, 4, 4)``.
        :param c: Tracking damping, ``(6, 6)`` or ``None`` for no damping.
        """
        self.node_index: int = int(node_index)
        self.k: Array = k
        self.hg_ref_t: Array = hg_ref_t
        self.c: Array = jnp.zeros((6, 6)) if c is None else c

    def _varphi(self, hg_i: Array, i_ts: int) -> Array:
        return hg_to_d(self.hg_ref_t[i_ts], hg_i)

    def f_res(self, hg: Array, v: Array, i_ts: int) -> Array:
        return -self.k @ self._varphi(hg, i_ts) - self.c @ v

    def k_tangent(self, hg: Array, i_ts: int) -> Array:
        return self.k @ t_se3(self._varphi(hg, i_ts))

    def c_tangent(self, hg: Array, i_ts: int) -> Array:
        return self.c


class HardConstraint(ABC):
    r"""
    Base class for a constraint applied through Lagrange multipliers.

    Subclasses are either holonomic or non-holonomic

    Grounded constraints constrain a single node relative to a fixed reference frame stored on the object.
    They have no ``node_j`` and must provide a reference coordinate `hg_ref`.
    """

    node_i: int
    node_j: int | None
    is_holonomic: ClassVar[bool] = True
    is_grounded: ClassVar[bool] = False

    @property
    def hg_ref(self) -> Array:
        r"""Fixed reference SE(3) frame for grounded constraints, ``(4, 4)``."""
        raise NotImplementedError(
            f"{type(self).__name__} is not grounded and has no hg_ref"
        )

    def violation(self, hg_i: Array, hg_j: Array) -> Array:
        r"""
        Compute the position-level constraint violation vector. Must be overridden
        by holonomic constraints.
        :param hg_i: SE(3) frame of node i, ``(4, 4)``.
        :param hg_j: SE(3) frame of node j, ``(4, 4)``.
        :return: Constraint violation, ``(n_constraints,)``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} is non-holonomic and has no position-level constraint"
        )

    def vel_violation(self, hg_i: Array, hg_j: Array, v_i: Array, v_j: Array) -> Array:
        r"""
        Compute the velocity-level constraint violation vector. Must be overridden
        by non-holonomic constraints.
        :param hg_i: SE(3) frame of node i, ``(4, 4)``.
        :param hg_j: SE(3) frame of node j, ``(4, 4)``.
        :param v_i: Local velocity of node i, ``(6, )``.
        :param v_j: Local velocity of node j, ``(6, )``.
        :return: Velocity constraint violation, ``(n_constraints,)``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} is holonomic and has no velocity-level constraint"
        )

    @property
    @abstractmethod
    def n_constraints(self) -> int:
        r"""Number of scalar constraints."""

    def jacobian_local(
        self,
        hg_base_i: Array,
        hg_base_j: Array,
        phi_i: Array | None = None,
        phi_j: Array | None = None,
    ) -> tuple[Array, Array]:
        r"""
        Compute constraint Jacobian blocks via forward-mode AD.
        :param hg_base_i: Base SE(3) frame of node i, ``(4, 4)``.
        :param hg_base_j: Base SE(3) frame of node j, ``(4, 4)``.
        :param phi_i: Accumulated configuration increment for node i, ``(6, )``.
        :param phi_j: Accumulated configuration increment for node j, ``(6, )``.
        :return: Jacobian blocks ``(jac_i, jac_j)`` each of shape ``(n_constraints, 6)``.
        """
        _phi_i = jnp.zeros(6) if phi_i is None else phi_i
        _phi_j = jnp.zeros(6) if phi_j is None else phi_j

        def violation_delta_i(delta_i: Array) -> Array:
            return self.violation(
                hg_base_i @ exp_se3(_phi_i + delta_i),
                hg_base_j @ exp_se3(_phi_j),
            )

        def violation_delta_j(delta_j: Array) -> Array:
            return self.violation(
                hg_base_i @ exp_se3(_phi_i),
                hg_base_j @ exp_se3(_phi_j + delta_j),
            )

        jac_i = jax.jacfwd(violation_delta_i)(jnp.zeros(6))
        jac_j = jax.jacfwd(violation_delta_j)(jnp.zeros(6))
        return jac_i, jac_j

    def a_vel_local(
        self,
        hg_i: Array,
        hg_j: Array,
        v_i: Array,
        v_j: Array,
    ) -> tuple[Array, Array]:
        r"""
        Velocity Jacobian blocks :math:`\partial g_{vel}/\partial v_i` and
        :math:`\partial g_{vel}/\partial v_j` via forward-mode AD.
        :param hg_i: SE(3) frame of node i, ``(4, 4)``.
        :param hg_j: SE(3) frame of node j, ``(4, 4)``.
        :param v_i: Local velocity of node i, ``(6, )``.
        :param v_j: Local velocity of node j, ``(6, )``.
        :return: Jacobian blocks ``(av_i, av_j)`` each ``(n_constraints, 6)``.
        """

        def vel_violation_vi(vi: Array) -> Array:
            return self.vel_violation(hg_i, hg_j, vi, v_j)

        def vel_violation_vj(vj: Array) -> Array:
            return self.vel_violation(hg_i, hg_j, v_i, vj)

        return jax.jacfwd(vel_violation_vi)(v_i), jax.jacfwd(vel_violation_vj)(v_j)

    def a_phi_local(
        self,
        hg_base_i: Array,
        hg_base_j: Array,
        v_i: Array,
        v_j: Array,
        phi_i: Array | None = None,
        phi_j: Array | None = None,
    ) -> tuple[Array, Array]:
        r"""
        Configuration Jacobian blocks of :meth:`vel_violation` with respect to SE(3)
        configuration increments, via forward-mode AD.
        :param hg_base_i: Base SE(3) frame of node i, ``(4, 4)``.
        :param hg_base_j: Base SE(3) frame of node j, ``(4, 4)``.
        :param v_i: Local velocity of node i, ``(6, )``.
        :param v_j: Local velocity of node j, ``(6, )``.
        :param phi_i: Accumulated configuration increment for node i, ``(6, )``.
        :param phi_j: Accumulated configuration increment for node j, ``(6, )``.
        :return: Jacobian blocks ``(ap_i, ap_j)`` each ``(n_constraints, 6)``.
        """
        _phi_i = jnp.zeros(6) if phi_i is None else phi_i
        _phi_j = jnp.zeros(6) if phi_j is None else phi_j

        def vel_violation_delta_i(delta_i: Array) -> Array:
            return self.vel_violation(
                hg_base_i @ exp_se3(_phi_i + delta_i),
                hg_base_j @ exp_se3(_phi_j),
                v_i,
                v_j,
            )

        def vel_violation_delta_j(delta_j: Array) -> Array:
            return self.vel_violation(
                hg_base_i @ exp_se3(_phi_i),
                hg_base_j @ exp_se3(_phi_j + delta_j),
                v_i,
                v_j,
            )

        return jax.jacfwd(vel_violation_delta_i)(jnp.zeros(6)), jax.jacfwd(vel_violation_delta_j)(
            jnp.zeros(6)
        )

    def resolve_hg_ref(self, hg0: Array) -> None:
        r"""
        Called once ``hg0`` is populated so subclasses can default their reference frame to the initial relative pose.
        :param hg0: Full nodal initial-frame array, ``(n_nodes, 4, 4)``.
        """


def _hinge_projection(axis: Array) -> tuple[Array, Array]:
    r"""
    Build the 5x6 projection matrix that selects the 3 translational and
    2 perpendicular-rotation DOFs constrained by a hinge about ``axis``.

    :param axis: Hinge axis direction ``(3, )``, not necessarily unit-length.
    :return: ``(projection, axis_normalised)`` where ``projection`` is ``(5, 6)``.
    """
    axis_unit = axis / jnp.linalg.norm(axis)
    trial = jnp.where(
        jnp.abs(axis_unit[0]) < 0.9,
        jnp.array([1.0, 0.0, 0.0]),
        jnp.array([0.0, 1.0, 0.0]),
    )
    e1 = jnp.cross(axis_unit, trial)
    e1 /= jnp.linalg.norm(e1)
    e2 = jnp.cross(axis_unit, e1)

    projection = (
        jnp.zeros((5, 6)).at[:3, :3].set(jnp.eye(3)).at[3, 3:].set(e1).at[4, 3:].set(e2)
    )
    return projection, axis_unit


@make_pytree
class MultibodyHinge(HardConstraint):
    r"""
    Hinge joint between two nodes. Constrains the relative configuration to allow only rotation about the
    specified axis (5 scalar constraints: 3 translation + 2 perpendicular rotation).
    """

    _static: ClassVar[tuple[str, ...]] = ("node_i", "node_j", "_hg_ref_from_hg0")
    n_constraints: ClassVar[int] = 5

    def __init__(
        self,
        node_i: int,
        node_j: int | None = None,
        *,
        axis: Array,
        hg_rel_ref: Array | None = None,
    ) -> None:
        r"""
        :param node_i: Index of first node.
        :param node_j: Index of second node, or ``None`` to have the beam structure automatically create a co-located
        node.
        :param axis: Hinge axis direction ``(3, )``, given in the frame of ``hg_rel_ref``.
        :param hg_rel_ref: Reference relative SE(3) pose of the hinge ``(4, 4)``, or ``None`` to default to the initial
        relative pose.
        """
        self.node_i: int = int(node_i)
        self.node_j: int | None = None if node_j is None else int(node_j)

        self.projection, self.axis = _hinge_projection(axis)

        self._hg_ref_from_hg0: bool = hg_rel_ref is None
        self.hg_rel_ref: Array = jnp.eye(4) if hg_rel_ref is None else hg_rel_ref

    def resolve_hg_ref(self, hg0: Array) -> None:
        if self._hg_ref_from_hg0:
            self.hg_rel_ref = hg_inv(hg0[self.node_i]) @ hg0[self.node_j]

    def violation(self, hg_i: Array, hg_j: Array) -> Array:
        hg_rel = hg_inv(hg_i) @ hg_j
        d_error = hg_to_d(self.hg_rel_ref, hg_rel)
        return self.projection @ d_error


@make_pytree
class GroundedHinge(HardConstraint):
    r"""
    Hinge joint pinning a single node to a fixed point in space.
    """

    _static: ClassVar[tuple[str, ...]] = ("node_i", "_hg_ref_from_hg0")
    n_constraints: ClassVar[int] = 5
    is_grounded: ClassVar[bool] = True

    def __init__(
        self,
        node_i: int,
        *,
        axis: Array,
        hg_ref: Array | None = None,
    ) -> None:
        r"""
        :param node_i: Index of the constrained node.
        :param axis: Hinge axis direction ``(3, )``, given in the global frame.
        :param hg_ref: Reference SE(3) frame ``(4, 4)`` the node is pinned to,
            or ``None`` to default to the node's initial pose.
        """
        self.node_i: int = int(node_i)
        self.node_j: None = None

        self.projection, self.axis = _hinge_projection(axis)

        self._hg_ref_from_hg0: bool = hg_ref is None
        self._hg_ref: Array = jnp.eye(4) if hg_ref is None else hg_ref

    @property
    def hg_ref(self) -> Array:
        return self._hg_ref

    def resolve_hg_ref(self, hg0: Array) -> None:
        if self._hg_ref_from_hg0:
            self._hg_ref = hg0[self.node_i]

    def violation(self, hg_i: Array, hg_j: Array) -> Array:
        d_error = hg_to_d(hg_j, hg_i)
        return self.projection @ d_error

    def jacobian_local(
        self,
        hg_base_i: Array,
        hg_base_j: Array,
        phi_i: Array | None = None,
        phi_j: Array | None = None,
    ) -> tuple[Array, Array]:
        _phi_i = jnp.zeros(6) if phi_i is None else phi_i

        def violation_delta_i(delta_i: Array) -> Array:
            return self.violation(hg_base_i @ exp_se3(_phi_i + delta_i), hg_base_j)

        jac_i = jax.jacfwd(violation_delta_i)(jnp.zeros(6))
        return jac_i, jnp.zeros((self.n_constraints, 6))
