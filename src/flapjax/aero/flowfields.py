from __future__ import annotations

from typing import ClassVar

import jax
from jax import Array
from jax import numpy as jnp

from flapjax.algebra.array_utils import ArrayList, check_arr_shape
from flapjax.utils.print_utils import warn
from flapjax.utils.utils import make_pytree


class FlowField:
    r"""
    Base class for background flow field definitions. Allows for the definition of arbitrary flow fields which are
    functions of space and time.
    """

    _static: ClassVar[tuple[str, ...]] = ("relative_motion",)

    def __init__(
        self,
        u_inf: Array,
        rho: float | Array,
        relative_motion: bool,
        mach: float | Array = 0.0,
    ):
        r"""
        :param u_inf: Base flow velocity, ``(3, )``.
        :param rho: Flow density.
        :param relative_motion: If True, the air moves, if False, the plane moves.
        :param mach: Freestream Mach number, used to apply a Prandtl-Glauert compressibility correction.
        Defaults to 0 (incompressible).
        """
        check_arr_shape(u_inf, (3,), name="u_inf")
        self.u_inf: Array = u_inf
        self.rho: Array = jnp.array(rho)
        self.u_inf_mag: Array = jnp.linalg.norm(u_inf)
        self.u_inf_dir: Array = u_inf / self.u_inf_mag
        self.q_inf: Array = 0.5 * rho * self.u_inf_mag**2  # dynamic pressure
        self.relative_motion: bool = relative_motion

        if isinstance(mach, (int, float)) and mach >= 1.0:
            warn(
                "Prandtl-Glauert compressibility correction requires subsonic flow (Mach < 1)."
            )
        self.mach: Array = jnp.array(mach)
        self.beta: Array = jnp.sqrt(
            1.0 - self.mach**2
        )  # Prandtl-Glauert compressibility factor

    def __call__(self, x: Array, t: Array) -> Array:
        """
        Evaluate the flow field at a spatial and temporal coordinate.

        :param x: Spatial coordinate, ``(3, )``
        :param t: Time, ().
        :return: Sampled velocity, ``(3, )``.
        """
        raise NotImplementedError("__call__ method must be implemented in subclasses.")

    def vmap_call(self, x: Array, t: Array) -> Array:
        """
        Vectorized version of the __call__ method. This maps over all leading dimensions of x.
        :param x: Spatial coordinates, ``(..., 3)``
        :param t: Time, ()
        :return: Flow field values at the specified coordinates, ``(..., 3)``
        """
        n_vmap = x.ndim - 1
        func = self.__call__
        for i_dim in range(n_vmap):
            func = jax.vmap(func, in_axes=(i_dim, None), out_axes=i_dim)
        return func(x, t)

    def surf_vmap_call(self, xs: ArrayList, t: Array) -> ArrayList:
        """
        Vectorized version of the __call__ method over a list of surfaces.
        :param xs: Spatial coordinates, ``(n_surf,)(..., 3)``
        :param t: Time, ()
        :return: Flow field values at the specified coordinates, ``(n_surf, )(..., 3)``
        """
        return ArrayList([self.vmap_call(x, t) for x in xs])

    def to_design_variables(self) -> dict[str, Array]:
        r"""
        Extract the design variables associated with this flow field.
        :return: Dictionary of design variables.
        """
        return {"u_inf": self.u_inf, "rho": self.rho, "mach": self.mach}

    def from_design_variables(self, design_variables: dict[str, Array]) -> FlowField:
        r"""
        Create a new flow field from design variables as the inverse of ``self.to_design_variables()``.
        :param design_variables: Dictionary of design variables.
        :return: New FlowField object.
        """
        return self.__class__(**design_variables, relative_motion=self.relative_motion)


@make_pytree
class ConstantFlowField(FlowField):
    r"""
    Constant velocity flow field.
    """

    def __call__(self, x: Array, t: Array) -> Array:
        if self.relative_motion:
            return self.u_inf
        else:
            return jnp.zeros(3)


@make_pytree
class OneMinusCosineFlowField(FlowField):
    r"""
    One minus cosine gust flow field.
    """

    def __init__(
        self,
        u_inf: Array,
        rho: float | Array,
        relative_motion: bool,
        gust_length: float | Array,
        gust_amplitude: float | Array,
        gust_travel_direction: Array | None = None,
        gust_amplitude_direction: Array | None = None,
        gust_x0: Array | None = None,
        mach: float | Array = 0.0,
    ):
        r"""
        :param u_inf: Base flow velocity, ``(3, )``.
        :param rho: Flow density.
        :param relative_motion: If True, the air moves, if False, the plane moves.
        :param gust_length: Gust length.
        :param gust_amplitude: Gust amplitude.
        :param gust_travel_direction: Vector which defines the direction that the gust travels, ``(3, )``. Defaults to the
        freestream direction if None.
        :param gust_amplitude_direction: Vector which defines the direction that the gust amplitude acts, ``(3, )``. Defaults
        to the z-direction if None.
        :param gust_x0: Coordinate on the initial leading edge of the gust, ``(3, )``. Defaults to 0 if None.
        :param mach: Freestream Mach number, used to apply a Prandtl-Glauert compressibility correction.
        Defaults to 0 (incompressible).
        """
        super().__init__(u_inf, rho, relative_motion, mach=mach)

        # base gust parameters
        self.gust_amplitude: Array = jnp.array(gust_amplitude)
        self.gust_length: Array = jnp.array(gust_length)

        # direction of travel for the gust - use background flow direction as default
        # even for a gust frozen in place, this defines the orientation of the ridge
        self.gust_travel_direction: Array = (
            gust_travel_direction
            if gust_travel_direction is not None
            else self.u_inf_dir
        )
        check_arr_shape(self.gust_travel_direction, (3,), "gust_travel_direction")
        self.gust_travel_direction /= jnp.linalg.norm(self.gust_travel_direction)

        # lateral direction of the gust (direction in which the gust acts), default is in Z
        self.gust_amplitude_direction: Array = (
            jnp.array((0.0, 0.0, 1.0))
            if gust_amplitude_direction is None
            else gust_amplitude_direction
        )
        check_arr_shape(self.gust_amplitude_direction, (3,), "gust_amplitude")
        self.gust_amplitude_direction /= jnp.linalg.norm(self.gust_amplitude_direction)

        # base coordinate at the start of the gust at t=0
        self.gust_x0: Array = gust_x0 if gust_x0 is not None else jnp.zeros(3)
        check_arr_shape(self.gust_x0, (3,), "gust_x0")

    def __call__(self, x: Array, t: Array) -> Array:
        rel_x = x - self.gust_x0  # position relative to the gust start
        if self.relative_motion:
            rel_x -= self.u_inf * t  # add relative motion if applicable

        gust_x = jnp.dot(rel_x, self.gust_travel_direction)

        def _one_minus_cos(x_: Array) -> Array:
            return (
                self.gust_amplitude_direction
                * self.gust_amplitude
                * 0.5
                * (1.0 - jnp.cos(jnp.pi * x_ / self.gust_length))
            )

        u = jax.lax.select(
            (gust_x > 0) & (gust_x < 2.0 * self.gust_length),
            _one_minus_cos(gust_x),
            jnp.zeros(3),
        )

        if self.relative_motion:
            u += self.u_inf
        return u

    def to_design_variables(self) -> dict[str, Array]:
        return {
            "u_inf": self.u_inf,
            "rho": self.rho,
            "gust_amplitude": self.gust_amplitude,
            "gust_length": self.gust_length,
            "mach": self.mach,
        }

    def from_design_variables(
        self, design_variables: dict[str, Array]
    ) -> OneMinusCosineFlowField:
        return OneMinusCosineFlowField(
            **design_variables,
            relative_motion=self.relative_motion,
            gust_travel_direction=self.gust_travel_direction,
            gust_amplitude_direction=self.gust_amplitude_direction,
            gust_x0=self.gust_x0,
        )
