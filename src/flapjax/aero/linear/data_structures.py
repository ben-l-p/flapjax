from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from jax import Array
from jax import numpy as jnp

from flapjax.aero.data_structures import AeroCase
from flapjax.algebra.array_utils import ArrayList
from flapjax.plotting.pvd import write_pvd


@dataclass
class AeroInputUnflattened:
    zeta_b: ArrayList
    zeta_b_dot: ArrayList
    nu_b: ArrayList | None
    nu_w: ArrayList | None


@dataclass
class AeroStateUnflattened:
    gamma_b: ArrayList
    gamma_w: ArrayList
    gamma_b_nm1: ArrayList | None
    zeta_w: ArrayList | None
    zeta_b: ArrayList | None


@dataclass
class AeroOutputUnflattened:
    f_steady: ArrayList
    f_unsteady: ArrayList | None


class AeroLinearResult:
    def __init__(
        self,
        reference: AeroCase,
        u_t: AeroInputUnflattened,
        x_t: AeroStateUnflattened,
        y_t: AeroOutputUnflattened,
        u_t_tot: AeroInputUnflattened,
        x_t_tot: AeroStateUnflattened,
        y_t_tot: AeroOutputUnflattened,
        n_tstep: int,
        n_surf: int,
        t: Array,
        surf_b_names: list[str],
        surf_w_names: list[str],
    ) -> None:
        # system results, if simulated
        self.u_t: AeroInputUnflattened = u_t
        self.x_t: AeroStateUnflattened = x_t
        self.y_t: AeroOutputUnflattened = y_t
        self.u_t_tot: AeroInputUnflattened = u_t_tot
        self.x_t_tot: AeroStateUnflattened = x_t_tot
        self.y_t_tot: AeroOutputUnflattened = y_t_tot
        self.n_tstep: int = n_tstep
        self.n_surf: int = n_surf
        self.t: Array = t
        self.surf_b_names: list[str] = surf_b_names
        self.surf_w_names: list[str] = surf_w_names
        self.reference: AeroCase = reference

    def plot(
        self,
        directory: str | os.PathLike,
        index: slice | Sequence[int] | int | Array | None = None,
        plot_wake: bool = True,
    ) -> None:
        r"""
        Plot the aerodynamic grid at specified time steps.
        :param directory: Directory to save the plots to
        :param index: Index or slice of time steps to plot. If None, plot all time steps.
        :param plot_wake: If True, plot the wake grid
        """
        if isinstance(index, slice):
            index_ = jnp.arange(self.n_tstep)[index]
        elif isinstance(index, Sequence):
            index_ = jnp.array(index)
        elif isinstance(index, Array):
            index_ = index
        elif isinstance(index, int):
            index_ = (index,)
        elif index is None:
            index_ = jnp.arange(self.n_tstep)
        else:
            raise TypeError("index must be a slices, sequence of ints, or Array")

        directory_path = Path(directory).resolve()
        directory_path.mkdir(parents=True, exist_ok=True)

        paths: list[Sequence[Path]] = []
        for i_ts in index_:
            snapshot = self[i_ts]
            paths.append(snapshot.plot(directory, plot_wake=plot_wake))

        for i_surf in range(2 * self.n_surf):
            try:
                surf_paths = [paths[i][i_surf] for i in range(len(index_))]
                name = (self.surf_b_names + self.surf_w_names)[i_surf] + "_ts"
                write_pvd(directory, name, surf_paths, list(self.t[index_]))
            except IndexError:
                pass

    def __getitem__(self, i_ts: int) -> AeroCase:
        r"""
        Get initial_snapshot of aerodynamic surface at a single time step
        :param i_ts: Timestep index
        :return: AeroCase at specified time step
        """

        if i_ts < 0 or i_ts >= self.n_tstep:
            raise IndexError("Timestep index out of range")

        # always exist
        zeta_b_tot = self.u_t_tot.zeta_b.index_all(i_ts, ...)
        zeta_b_dot_tot = self.u_t_tot.zeta_b_dot.index_all(i_ts, ...)
        gamma_b_tot = self.x_t_tot.gamma_b.index_all(i_ts, ...)
        gamma_w_tot = self.x_t_tot.gamma_w.index_all(i_ts, ...)
        f_steady_tot = self.y_t_tot.f_steady.index_all(i_ts, ...)

        # put in the reference grid for the frozen wake case
        zeta_w_tot = (
            self.x_t_tot.zeta_w.index_all(i_ts, ...)
            if self.x_t_tot.zeta_w is not None
            else self.reference.zeta_w
        )
        f_unsteady_tot = (
            self.y_t_tot.f_unsteady.index_all(i_ts, ...)
            if self.y_t_tot.f_unsteady is not None
            else None
        )

        return AeroCase(
            zeta_b=zeta_b_tot,
            zeta_b_dot=zeta_b_dot_tot,
            zeta_w=zeta_w_tot,
            gamma_b=gamma_b_tot,
            gamma_b_dot=None,
            gamma_w=gamma_w_tot,
            f_steady=f_steady_tot,
            f_unsteady=f_unsteady_tot,
            cs_ang={},
            cs_vel={},
            surf_b_names=self.surf_b_names,
            surf_w_names=self.surf_w_names,
            i_ts=i_ts,
            t=self.t[i_ts],
            static_horseshoe=False,
            c=None,
            n=None,
            alpha=None,
            cl=None,
            cd=None,
            cm=None,
            kernels=self.reference.kernels,
            mirror_point=None,
            mirror_normal=None,
            flowfield=self.reference.flowfield,
            dof_mapping=self.reference.dof_mapping,
            free_wake=self.reference.free_wake,
            gamma_dot_relaxation=self.reference.gamma_dot_relaxation,
            batch_size=1,  # chosen default value
        )
