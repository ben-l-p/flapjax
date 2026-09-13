from __future__ import annotations

from typing import ClassVar

from jax import Array
from jax import numpy as jnp

from flapjax.utils.utils import make_pytree


class FrequencyFlowField:
    r"""
    Base class for turbulence spectra used in frequency-domain gust analysis.
    """

    _static: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        sigma: float | Array,
        length_scale: float | Array,
        u_inf: float | Array,
    ):
        r"""
        :param sigma: Turbulence intensity (RMS gust velocity), m/s.
        :param length_scale: Turbulence scale length, m.
        :param u_inf: Freestream velocity magnitude, m/s.
        """
        self.sigma: Array = jnp.array(sigma)
        self.length_scale: Array = jnp.array(length_scale)
        self.u_inf: Array = jnp.array(u_inf)

    def psd(self, omega: Array) -> Array:
        r"""
        Power spectral density of the vertical gust velocity.

        :param omega: Sampling requencies in rad/s, ``(n_freq, )``.
        :return: PSD values, ``(n_freq, )``.
        """
        raise NotImplementedError("psd must be implemented in subclasses.")


@make_pytree
class VonKarmanFlowField(FrequencyFlowField):
    r"""
    Von Kármán continuous turbulence spectrum.
    """

    def psd(self, omega: Array) -> Array:
        eta = 1.339 * self.length_scale * omega / self.u_inf
        eta2 = eta**2
        return (
            self.sigma**2
            * self.length_scale
            / (jnp.pi * self.u_inf)
            * (1.0 + 8.0 / 3.0 * eta2)
            / (1.0 + eta2) ** (11.0 / 6.0)
        )


@make_pytree
class DrydenFlowField(FrequencyFlowField):
    r"""
    Dryden continuous turbulence spectrum.
    """

    def psd(self, omega: Array) -> Array:
        eta = self.length_scale * omega / self.u_inf
        eta2 = eta**2
        return (
            self.sigma**2
            * self.length_scale
            / (jnp.pi * self.u_inf)
            * (1.0 + 3.0 * eta2)
            / (1.0 + eta2) ** 2
        )
