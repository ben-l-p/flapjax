from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from jax import Array, vmap
from jax import numpy as jnp

from flapjax.aero.aic import compute_v_ind
from flapjax.aero.flowfields import FlowField
from flapjax.aero.gradients.data_structures import AeroFullStates
from flapjax.aero.utils import (
    KernelFunction,
    compute_c,
    compute_nc,
    project_forcing_to_beam,
)
from flapjax.algebra.array_utils import ArrayList
from flapjax.algebra.base import finite_difference
from flapjax.plotting.aerogrid import plot_grid_to_vtk
from flapjax.plotting.pvd import write_pvd
from flapjax.utils.print_utils import warn
from flapjax.utils.utils import index_to_arr, make_pytree


@dataclass
class GridDiscretisation:
    r"""
    Data class to hold discretisation parameters for each aerodynamic grid.
    :param m: Number of panels in the chordwise direction.
    :param n: Number of panels in the spanwise direction.
    :param m_star: Number of wake panels in the chordwise direction.
    """

    m: int
    n: int
    m_star: int


@make_pytree
class AeroCase:
    r"""Contains an aerodynamic solution across one or many timesteps.

    A single instance may represent either:

    * **Snapshot** (single timestep): array leaves within have no leading time axis (e.g. ``zeta_b[i_surf].shape == (m+1, n+1, 3)``,
      ``gamma_b[i_surf].shape == (m, n)``). ``t`` is a scalar and ``i_ts`` is an integer
    * **Batched** (many timesteps): array leaves carry a leading ``n_tstep`` axis
      (e.g. ``zeta_b[i_surf].shape == (n_tstep, m+1, n+1, 3)``). ``t`` is
      ``(n_tstep,)`` and ``i_ts`` is a ``(n_tstep, )`` array of indices.

    Use ``is_batched`` to distinguish at runtime.
    """

    _static: ClassVar[tuple[str, ...]] = (
        "surf_b_names",
        "surf_w_names",
        "static_horseshoe",
        "free_wake",
        "kernels",
        "batch_size",
    )

    def __init__(
        self,
        zeta_b: ArrayList,
        zeta_b_dot: ArrayList,
        zeta_w: ArrayList,
        c: ArrayList | None,
        n: ArrayList | None,
        gamma_b: ArrayList,
        gamma_b_dot: ArrayList | None,
        gamma_w: ArrayList,
        f_steady: ArrayList,
        f_unsteady: ArrayList | None,
        alpha: ArrayList | None,
        cl: ArrayList | None,
        cd: ArrayList | None,
        cm: ArrayList | None,
        cs_ang: dict[str, Array],
        cs_vel: dict[str, Array],
        kernels: Sequence[KernelFunction],
        mirror_point: Array | None,
        mirror_normal: Array | None,
        flowfield: FlowField,
        surf_b_names: Sequence[str],
        surf_w_names: Sequence[str],
        t: Array,
        i_ts: Array | int,
        dof_mapping: ArrayList,
        static_horseshoe: bool,
        free_wake: bool,
        gamma_dot_relaxation: float | Array,
        batch_size: int | None,
    ) -> None:
        r"""
        :param zeta_b: Bound grid coordinates, batched: ``(n_surf, )(n_tstep, zeta_m, zeta_n, 3)`` /
            snapshot: ``(n_surf, )(zeta_m, zeta_n, 3)``.
        :param zeta_b_dot: Bound grid velocities, same layout as ``zeta_b``.
        :param zeta_w: Wake grid coordinates or ``None``.
        :param c: Bound collocation points or ``None``.
        :param n: Bound grid normals or ``None``.
        :param gamma_b: Bound circulation strengths, batched: ``(n_surf, )(n_tstep, m, n)`` /
            snapshot: ``(n_surf, )(m, n)``.
        :param gamma_b_dot: Bound circulation time derivatives or ``None``.
        :param gamma_w: Wake circulation strengths.
        :param f_steady: Steady force contributions.
        :param f_unsteady: Unsteady force contributions or ``None``.
        :param alpha: Per-strip effective angle of attack extracted from the UVLM sectional lift; batched:
            ``(n_surf, )(n_tstep, n)``, snapshot: ``(n_surf, )(n, )``, or ``None``.
        :param cl: Per-strip lift coefficient sampled from the airfoil polars, or ``None`` for no polars.
        :param cd: Per-strip drag coefficient sampled from the airfoil polars, or ``None`` for no polars.
        :param cm: Per-strip moment coefficient sampled from the airfoil polars, or ``None`` for no polars.
        :param cs_ang: Control surface angle time history, ``{name: (n_tstep,)}`` (batched) or ``{name: ()}`` (snapshot).
        :param cs_vel: Control surface velocity time history.
        :param kernels: Kernel functions for both bound and wake source grids.
        :param mirror_point: Point on mirror plane, ``(3, )`` or None.
        :param mirror_normal: Normal on mirror plane, ``(3, )`` or None.
        :param flowfield: ``FlowField`` object which includes background velocity and density.
        :param surf_b_names: Names of bound surfaces, ``(n_surf, )``.
        :param surf_w_names: Names of wake surfaces, ``(n_surf, )``.
        :param t: Time; batched: ``(n_tstep, )``, snapshot: scalar.
        :param i_ts: Timestep index; batched: ``(n_tstep, )``, snapshot: ``int``.
        :param dof_mapping: Map from aero grid to beam DOFs, ``(n_surf, )(zeta_n, )``.
        :param static_horseshoe: If true, a horseshoe formulation was used for the initial static solution.
        :param free_wake: Free-wake formulation flag.
        :param gamma_dot_relaxation: Circulation time derivative filter.
        :param batch_size: Batch size used for AIC vectorisation.
        """
        self.zeta_b: ArrayList = zeta_b
        self.zeta_b_dot: ArrayList = zeta_b_dot
        self.zeta_w: ArrayList = zeta_w
        self.c: ArrayList | None = c
        self.nc: ArrayList | None = n
        self.gamma_b: ArrayList = gamma_b
        self.gamma_b_dot: ArrayList | None = gamma_b_dot
        self.gamma_w: ArrayList = gamma_w
        self.f_steady: ArrayList = f_steady
        self.f_unsteady: ArrayList | None = f_unsteady
        self.alpha: ArrayList | None = alpha
        self.cl: ArrayList | None = cl
        self.cd: ArrayList | None = cd
        self.cm: ArrayList | None = cm
        self.cs_ang: dict[str, Array] = cs_ang
        self.cs_vel: dict[str, Array] = cs_vel
        self.t: Array = t
        self.i_ts: Array | int = i_ts

        self.kernels: Sequence[KernelFunction] = kernels
        self.mirror_point: Array | None = mirror_point
        self.mirror_normal: Array | None = mirror_normal
        self.flowfield: FlowField = flowfield
        self.surf_b_names: Sequence[str] = surf_b_names
        self.surf_w_names: Sequence[str] = surf_w_names
        self.dof_mapping: ArrayList = dof_mapping

        # settings
        self.static_horseshoe: bool = static_horseshoe
        self.free_wake: bool = free_wake
        self.gamma_dot_relaxation: float | Array = gamma_dot_relaxation
        self.batch_size: int | None = batch_size

    @property
    def n_surf(self) -> int:
        return len(self.surf_b_names)

    @property
    def is_batched(self) -> bool:
        # gamma_b[i_surf] shape: (m, n) snapshot, (n_tstep, m, n) batched
        return self.gamma_b[0].ndim == 3

    @property
    def n_tstep(self) -> int:
        if not self.is_batched:
            raise ValueError("n_tstep only defined for batched AeroCase")
        return self.gamma_b[0].shape[0]

    @property
    def f_unsteady(self) -> ArrayList:
        if self._f_unsteady is None:
            return ArrayList.zeros_like(self.f_steady)
        else:
            return self._f_unsteady

    @f_unsteady.setter
    def f_unsteady(self, value: ArrayList | None) -> None:
        self._f_unsteady = value

    @property
    def c(self) -> ArrayList:
        if self._c is None:
            self._c = compute_c(zetas=self.zeta_b)
        return self._c

    @c.setter
    def c(self, value: ArrayList | None) -> None:
        self._c = value

    @property
    def nc(self) -> ArrayList:
        if self._nc is None:
            self._nc = compute_nc(zetas=self.zeta_b)
        return self._nc

    @nc.setter
    def nc(self, value: ArrayList | None) -> None:
        self._nc = value

    @property
    def alpha(self) -> ArrayList:
        if self._alpha is None:
            # return zeros if None
            self._alpha = ArrayList.zeros_like(self.gamma_b)
        return self._alpha

    @alpha.setter
    def alpha(self, value: ArrayList | None) -> None:
        self._alpha = value

    @property
    def cl(self) -> ArrayList:
        if self._cl is None:
            # default to the flat-plate value
            self._cl = 2.0 * jnp.pi * self.alpha
        return self._cl

    @cl.setter
    def cl(self, value: ArrayList | None) -> None:
        self._cl = value

    @property
    def cd(self) -> ArrayList:
        if self._cd is None:
            self._cd = ArrayList.zeros_like(self.gamma_b)
        return self._cd

    @cd.setter
    def cd(self, value: ArrayList | None) -> None:
        self._cd = value

    @property
    def cm(self) -> ArrayList:
        if self._cm is None:
            self._cm = ArrayList.zeros_like(self.gamma_b)
        return self._cm

    @cm.setter
    def cm(self, value: ArrayList | None) -> None:
        self._cm = value

    @property
    def gamma_b_dot(self) -> ArrayList:
        if self._gamma_b_dot is None:
            # compute based on gamma_b
            if self.is_batched:
                dt = self.t[1] - self.t[0]
                idx = jnp.arange(self.n_tstep)

                def _per_surface(gb: Array) -> Array:
                    return vmap(
                        lambda i_ts: finite_difference(
                            i_=i_ts, data=gb, delta=dt, axis=0, order=1
                        ),
                        in_axes=0,
                        out_axes=0,
                    )(idx)

                self._gamma_b_dot = ArrayList([_per_surface(gb) for gb in self.gamma_b])
            else:
                self._gamma_b_dot = ArrayList.zeros_like(self.gamma_b)
        return self._gamma_b_dot

    @gamma_b_dot.setter
    def gamma_b_dot(self, value: ArrayList | None) -> None:
        self._gamma_b_dot = value

    def get_states(self, i_ts: int | Array | None = None) -> AeroFullStates:
        r"""
        Obtain the aerodynamic state at a given timestep (used in the adjoint solution).
        :param i_ts: Time step index (required for batched, ignored for snapshot).
        :return: Aero states.
        """
        if self.is_batched:
            if i_ts is None:
                raise ValueError("i_ts must be provided for batched AeroCase")
            assert self.gamma_b_dot is not None and self.zeta_w is not None
            return AeroFullStates(
                gamma_b=self.gamma_b.index_all(i_ts, ...),
                gamma_w=self.gamma_w.index_all(i_ts, ...),
                gamma_b_dot=self.gamma_b_dot.index_all(i_ts, ...),
                zeta_w=self.zeta_w.index_all(i_ts, ...),
            )
        assert self.gamma_b_dot is not None and self.zeta_w is not None
        return AeroFullStates(
            gamma_b=self.gamma_b,
            gamma_w=self.gamma_w,
            gamma_b_dot=self.gamma_b_dot,
            zeta_w=self.zeta_w,
        )

    def gamma_full(self, i_ts: int | None = None) -> ArrayList:
        r"""Concatenate bound and wake circulation strengths.
        :param i_ts: Time step index (required for batched, ignored for snapshot).
        :return: Circulation strength, ``(2 * n_surf,)(m | m_star, n)``.
        """
        if self.is_batched:
            if i_ts is None:
                raise ValueError("i_ts must be provided for batched AeroCase")
            return ArrayList(
                [
                    *self.gamma_b.index_all(i_ts, ...),
                    *self.gamma_w.index_all(i_ts, ...),
                ]
            )
        return ArrayList([*self.gamma_b, *self.gamma_w])

    def zeta_full(self, i_ts: int | None = None) -> ArrayList:
        r"""Concatenate bound and wake grids.
        :param i_ts: Time step index (required for batched, ignored for snapshot).
        :return: Grids, ``(2 * n_surf,)(zeta_m | zeta_m_star, zeta_n, 3)``.
        """
        if self.is_batched:
            if i_ts is None:
                raise ValueError("i_ts must be provided for batched AeroCase")
            assert self.zeta_w is not None
            return ArrayList(
                [
                    *self.zeta_b.index_all(i_ts, ...),
                    *self.zeta_w.index_all(i_ts, ...),
                ]
            )
        assert self.zeta_w is not None
        return ArrayList([*self.zeta_b, *self.zeta_w])

    def set_arraylist_at_ts(self, attr: str, values: ArrayList, i_ts: int) -> None:
        """Set an attribute at a given timestep on a batched AeroCase.
        :param attr: Name of the attribute to set.
        :param values: ArrayList of per-surface values (no leading time axis).
        :param i_ts: Time step index.
        """
        if not self.is_batched:
            raise TypeError("set_arraylist_at_ts only supported for batched AeroCase")
        arr = getattr(self, attr)
        for i_surf, val in enumerate(values):
            arr[i_surf] = arr[i_surf].at[i_ts, ...].set(val)

    def get_surf_snapshot(self, i_ts: int, i_surf: int) -> _AeroSurfacePlot:
        r"""Get single-surface plot data for a given ``(timestep, surface)`` pair on
        a batched AeroCase.
        """
        if not self.is_batched:
            raise TypeError("get_surf_snapshot only supported for batched AeroCase")
        assert self.zeta_w is not None
        assert self.gamma_b_dot is not None
        assert self.f_unsteady is not None
        return _AeroSurfacePlot(
            zeta_b=self.zeta_b[i_surf][i_ts, ...],
            zeta_b_dot=self.zeta_b_dot[i_surf][i_ts, ...],
            zeta_w=self.zeta_w[i_surf][i_ts, ...],
            gamma_b=self.gamma_b[i_surf][i_ts, ...],
            gamma_b_dot=self.gamma_b_dot[i_surf][i_ts, ...],
            gamma_w=self.gamma_w[i_surf][i_ts, ...],
            f_steady=self.f_steady[i_surf][i_ts, ...],
            f_unsteady=self.f_unsteady[i_surf][i_ts, ...],
            alpha=self.alpha[i_surf][i_ts, ...],
            cl=self.cl[i_surf][i_ts, ...],
            cd=self.cd[i_surf][i_ts, ...],
            cm=self.cm[i_surf][i_ts, ...],
            surf_b_name=self.surf_b_names[i_surf],
            surf_w_name=self.surf_w_names[i_surf],
            i_ts=i_ts,
        )

    def get_surface(self, idx: int) -> _AeroSurfacePlot:
        """Get single-surface plot data for the given surface on a snapshot
        AeroCase."""
        if self.is_batched:
            raise TypeError("get_surface only supported for snapshot AeroCase")
        assert self.zeta_w is not None
        assert self.gamma_b_dot is not None
        assert self.f_unsteady is not None
        return _AeroSurfacePlot(
            zeta_b=self.zeta_b[idx],
            zeta_b_dot=self.zeta_b_dot[idx],
            zeta_w=self.zeta_w[idx],
            gamma_b=self.gamma_b[idx],
            gamma_b_dot=self.gamma_b_dot[idx],
            gamma_w=self.gamma_w[idx],
            f_steady=self.f_steady[idx],
            f_unsteady=self.f_unsteady[idx],
            alpha=self.alpha[idx],
            cl=self.cl[idx],
            cd=self.cd[idx],
            cm=self.cm[idx],
            surf_b_name=self.surf_b_names[idx],
            surf_w_name=self.surf_w_names[idx],
            i_ts=int(self.i_ts),
        )

    def __setitem__(self, i_ts: int, snapshot: AeroCase) -> None:
        r"""Set the data at a given time step from a snapshot AeroCase."""
        if not self.is_batched:
            raise TypeError("__setitem__ only supported for batched AeroCase")
        if snapshot.is_batched:
            raise TypeError("snapshot AeroCase must not be batched")

        self.t = self.t.at[i_ts].set(snapshot.t)
        for i_surf in range(self.n_surf):
            self.zeta_b[i_surf] = (
                self.zeta_b[i_surf].at[i_ts, ...].set(snapshot.zeta_b[i_surf])
            )
            self.zeta_b_dot[i_surf] = (
                self.zeta_b_dot[i_surf].at[i_ts, ...].set(snapshot.zeta_b_dot[i_surf])
            )
            if self.zeta_w is not None and snapshot.zeta_w is not None:
                self.zeta_w[i_surf] = (
                    self.zeta_w[i_surf].at[i_ts, ...].set(snapshot.zeta_w[i_surf])
                )
            self.gamma_b[i_surf] = (
                self.gamma_b[i_surf].at[i_ts, ...].set(snapshot.gamma_b[i_surf])
            )
            if self.gamma_b_dot is not None and snapshot.gamma_b_dot is not None:
                self.gamma_b_dot[i_surf] = (
                    self.gamma_b_dot[i_surf]
                    .at[i_ts, ...]
                    .set(snapshot.gamma_b_dot[i_surf])
                )
            self.gamma_w[i_surf] = (
                self.gamma_w[i_surf].at[i_ts, ...].set(snapshot.gamma_w[i_surf])
            )
            self.f_steady[i_surf] = (
                self.f_steady[i_surf].at[i_ts, ...].set(snapshot.f_steady[i_surf])
            )
            if self.f_unsteady is not None and snapshot.f_unsteady is not None:
                self.f_unsteady[i_surf] = (
                    self.f_unsteady[i_surf]
                    .at[i_ts, ...]
                    .set(snapshot.f_unsteady[i_surf])
                )
            self.alpha[i_surf] = (
                self.alpha[i_surf].at[i_ts, ...].set(snapshot.alpha[i_surf])
            )
            self.cl[i_surf] = self.cl[i_surf].at[i_ts, ...].set(snapshot.cl[i_surf])
            self.cd[i_surf] = self.cd[i_surf].at[i_ts, ...].set(snapshot.cd[i_surf])
            self.cm[i_surf] = self.cm[i_surf].at[i_ts, ...].set(snapshot.cm[i_surf])

    def plot(
        self,
        directory: os.PathLike | str,
        plot_bound: bool = True,
        plot_wake: bool = True,
        index: int | Sequence[int] | Array | slice | None = None,
    ) -> Sequence[Path]:
        r"""Plot aerodynamic surfaces to VTU files (with per-surface PVD when
        batched).
        :param directory: Directory to save files.
        :param plot_bound: If True, plot the bound surfaces.
        :param plot_wake: If True, plot the wake surfaces.
        :param index: For batched, timestep indices to plot (all if None).
            Ignored for snapshots.
        :return: Sequence of paths to the saved PVD (batched) or VTU (snapshot) files.
        """
        directory_path = Path(directory)
        directory_path.mkdir(parents=True, exist_ok=True)

        if not self.is_batched:
            paths: list[Path] = []
            for i_surf in range(self.n_surf):
                paths.extend(
                    self.get_surface(idx=i_surf).plot(
                        directory, plot_bound=plot_bound, plot_wake=plot_wake
                    )
                )
            return paths

        index_ = index_to_arr(index=index, n_entries=self.n_tstep)
        pvd_paths: list[Path] = []
        for i_surf in range(self.n_surf):
            per_ts_paths: list[Sequence[Path]] = []
            for i_ts in index_:
                per_ts_paths.append(
                    self.get_surf_snapshot(i_ts=i_ts, i_surf=i_surf).plot(
                        directory, plot_bound=plot_bound, plot_wake=plot_wake
                    )
                )

            if plot_bound:
                bound_name = f"aero_dynamic_{self.surf_b_names[i_surf]}_ts"
                pvd_paths.append(
                    write_pvd(
                        directory=directory,
                        name=bound_name,
                        file_dirs=next(zip(*per_ts_paths)),
                        times=list(self.t[index_]),
                    )
                )

            if plot_wake:
                wake_name = f"aero_dynamic_{self.surf_w_names[i_surf]}_ts"
                pvd_paths.append(
                    write_pvd(
                        directory=directory,
                        name=wake_name,
                        file_dirs=list(zip(*per_ts_paths))[-1],
                        times=list(self.t[index_]),
                    )
                )
        return pvd_paths

    def project_forcing_to_beam(
        self,
        i_ts: int,
        rmat: Array,
        x0_aero: ArrayList,
        include_unsteady: bool,
    ) -> Array:
        r"""Project aerodynamic forcing at ``i_ts`` onto the beam grid (global frame).
        :param i_ts: Time step index (ignored for snapshot).
        :param rmat: Rotation matrix for each node relative to reference, ``(n_nodes, 3, 3)``.
        :param x0_aero: Reference coordinates for aerodynamic grid, ``(n_surf, )(zeta_m, zeta_n, 3)``.
        :param include_unsteady: If true, include unsteady forcing.
        :return: Steady and unsteady forcing projected onto the beam grid, ``(n_nodes, 6)``.
        """
        if self.is_batched:
            f_total = self.f_steady.index_all(i_ts, ...)
            if include_unsteady:
                assert self.f_unsteady is not None
                f_total += self.f_unsteady.index_all(i_ts, ...)
        else:
            f_total = self.f_steady
            if include_unsteady:
                assert self.f_unsteady is not None
                f_total = ArrayList([a + b for a, b in zip(f_total, self.f_unsteady)])

        return project_forcing_to_beam(
            f_total=f_total, rmat=rmat, x0_aero=x0_aero, dof_mapping=self.dof_mapping
        )

    def _t_at(self, i_ts: int | None) -> Array:
        if self.is_batched:
            if i_ts is None:
                raise ValueError("i_ts required for batched AeroCase")
            return self.t[i_ts]
        return self.t

    def get_v_background[T: Array | ArrayList](
        self, x_target: T, i_ts: int | None = None
    ) -> T:
        r"""Background velocity at specified points and time step."""
        t_val = self._t_at(i_ts)
        if isinstance(x_target, Array):
            return self.flowfield.vmap_call(x=x_target, t=t_val)
        elif isinstance(x_target, ArrayList):
            return self.flowfield.surf_vmap_call(xs=x_target, t=t_val)  # type: ignore
        raise NotImplementedError

    def get_v_ind[T: Array | ArrayList](
        self, x_target: T, i_ts: int | None = None
    ) -> T:
        r"""Induced velocity at specified points and time step."""
        return compute_v_ind(
            cs=x_target,
            zetas=self.zeta_full(i_ts),
            gammas=self.gamma_full(i_ts),
            kernels=self.kernels,
            mirror_normal=self.mirror_normal,
            mirror_point=self.mirror_point,
            batch_size=self.batch_size,
        )

    def get_v_tot[T: Array | ArrayList](
        self, x_target: T, i_ts: int | None = None
    ) -> T:
        r"""Total (induced + background) velocity at specified points and time step."""
        return self.get_v_ind(x_target=x_target, i_ts=i_ts) + self.get_v_background(
            x_target=x_target, i_ts=i_ts
        )

    def __getitem__(self, idx: int) -> AeroCase:
        r"""Extract a snapshot at time index ``idx`` from a batched AeroCase."""
        if not self.is_batched:
            raise TypeError("__getitem__ only supported for batched AeroCase")
        return AeroCase(
            zeta_b=self.zeta_b.index_all(idx, ...),
            zeta_b_dot=self.zeta_b_dot.index_all(idx, ...),
            zeta_w=self.zeta_w.index_all(idx, ...),
            c=self.c.index_all(idx, ...) if self.c is not None else None,
            n=self.nc.index_all(idx, ...) if self.nc is not None else None,
            gamma_b=self.gamma_b.index_all(idx, ...),
            gamma_b_dot=self.gamma_b_dot.index_all(idx, ...)
            if self.gamma_b_dot is not None
            else None,
            gamma_w=self.gamma_w.index_all(idx, ...),
            f_steady=self.f_steady.index_all(idx, ...),
            f_unsteady=self.f_unsteady.index_all(idx, ...)
            if self.f_unsteady is not None
            else None,
            alpha=self.alpha.index_all(idx, ...),
            cl=self.cl.index_all(idx, ...),
            cd=self.cd.index_all(idx, ...),
            cm=self.cm.index_all(idx, ...),
            cs_ang={k: jnp.atleast_1d(v)[idx, ...] for k, v in self.cs_ang.items()},
            cs_vel={k: jnp.atleast_1d(v)[idx, ...] for k, v in self.cs_vel.items()},
            surf_b_names=self.surf_b_names,
            surf_w_names=self.surf_w_names,
            t=self.t[idx],
            i_ts=idx,
            static_horseshoe=self.static_horseshoe,
            free_wake=self.free_wake,
            gamma_dot_relaxation=self.gamma_dot_relaxation,
            kernels=self.kernels,
            mirror_point=self.mirror_point,
            mirror_normal=self.mirror_normal,
            flowfield=self.flowfield,
            dof_mapping=self.dof_mapping,
            batch_size=self.batch_size,
        )

    def to_dynamic(self, i_ts: int, n_tstep: int) -> AeroCase:
        """Expand this snapshot into a batched AeroCase with ``n_tstep``
        timesteps, placing the current snapshot at index ``i_ts``.
        """
        if self.is_batched:
            raise TypeError("to_dynamic only supported for snapshot AeroCase")

        def _expand(arr_list: ArrayList) -> ArrayList:
            out = []
            for a in arr_list:
                out.append(jnp.zeros((n_tstep, *a.shape)).at[i_ts, ...].set(a))
            return ArrayList(out)

        return AeroCase(
            zeta_b=_expand(self.zeta_b),
            zeta_b_dot=_expand(self.zeta_b_dot),
            zeta_w=_expand(self.zeta_w),
            c=_expand(self.c),
            n=_expand(self.nc),
            gamma_b=_expand(self.gamma_b),
            gamma_b_dot=_expand(self.gamma_b_dot),
            gamma_w=_expand(self.gamma_w),
            f_steady=_expand(self.f_steady),
            f_unsteady=_expand(self.f_unsteady),
            alpha=_expand(self.alpha),
            cl=_expand(self.cl),
            cd=_expand(self.cd),
            cm=_expand(self.cm),
            cs_ang={k: jnp.full(n_tstep, v) for k, v in self.cs_ang.items()},
            cs_vel={k: jnp.full(n_tstep, v) for k, v in self.cs_vel.items()},
            kernels=self.kernels,
            mirror_point=self.mirror_point,
            mirror_normal=self.mirror_normal,
            flowfield=self.flowfield,
            surf_b_names=self.surf_b_names,
            surf_w_names=self.surf_w_names,
            t=jnp.zeros(n_tstep).at[i_ts].set(self.t),
            i_ts=jnp.arange(n_tstep),
            dof_mapping=self.dof_mapping,
            static_horseshoe=self.static_horseshoe,
            free_wake=self.free_wake,
            gamma_dot_relaxation=self.gamma_dot_relaxation,
            batch_size=self.batch_size,
        )

    @classmethod
    def initialise(cls, initial_snapshot: AeroCase, n_tstep: int) -> AeroCase:
        r"""Create a batched AeroCase from a snapshot placed at ``i_ts=0``."""
        return initial_snapshot.to_dynamic(i_ts=0, n_tstep=n_tstep)


class _AeroSurfacePlot:
    """Helper class for plotting a single surface at a single timestep to VTU."""

    def __init__(
        self,
        zeta_b: Array,
        zeta_b_dot: Array,
        zeta_w: Array,
        gamma_b: Array,
        gamma_b_dot: Array,
        gamma_w: Array,
        f_steady: Array,
        f_unsteady: Array,
        alpha: Array,
        cl: Array,
        cd: Array,
        cm: Array,
        surf_b_name: str,
        surf_w_name: str,
        i_ts: int,
    ) -> None:
        self.zeta_b = zeta_b
        self.zeta_b_dot = zeta_b_dot
        self.zeta_w = zeta_w
        self.gamma_b = gamma_b
        self.gamma_b_dot = gamma_b_dot
        self.gamma_w = gamma_w
        self.f_steady = f_steady
        self.f_unsteady = f_unsteady
        self.alpha = alpha
        self.cl = cl
        self.cd = cd
        self.cm = cm
        self.surf_b_name = surf_b_name
        self.surf_w_name = surf_w_name
        self.i_ts = i_ts

    def plot(
        self,
        directory: str | os.PathLike,
        plot_bound: bool = True,
        plot_wake: bool = True,
    ) -> Sequence[Path]:
        Path(directory).mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        if plot_bound:
            bound_filename = Path(directory).joinpath(self.surf_b_name)
            paths.append(
                plot_grid_to_vtk(
                    self.zeta_b,
                    bound_filename,
                    self.i_ts,
                    node_vector_data={
                        "f_steady": self.f_steady,
                        "f_unsteady": self.f_unsteady,
                        "zeta_dot": self.zeta_b_dot,
                    },
                    cell_scalar_data={
                        "gamma": self.gamma_b,
                        "gamma_dot": self.gamma_b_dot,
                        "alpha": jnp.broadcast_to(self.alpha, self.gamma_b.shape),
                        "cl": jnp.broadcast_to(self.cl, self.gamma_b.shape),
                        "cd": jnp.broadcast_to(self.cd, self.gamma_b.shape),
                        "cm": jnp.broadcast_to(self.cm, self.gamma_b.shape),
                    },
                )
            )
        if plot_wake:
            if not self.gamma_w.shape[0]:
                warn("No wake panels to plot, skipping.")
            else:
                wake_filename = Path(directory).joinpath(self.surf_w_name)
                paths.append(
                    plot_grid_to_vtk(
                        self.zeta_w,
                        wake_filename,
                        self.i_ts,
                        cell_scalar_data={"gamma": self.gamma_w},
                    )
                )
        return paths
