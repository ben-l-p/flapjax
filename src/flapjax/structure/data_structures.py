from __future__ import annotations

import os
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, overload

from jax import Array
from jax import numpy as jnp

from flapjax.plotting.beam import plot_beam_to_vtk
from flapjax.plotting.pvd import write_pvd
from flapjax.structure.gradients.data_structures import StructureFullStates
from flapjax.structure.utils import (
    apply_frame_transform,
    get_solve_dofs,
    input_dof_index_to_tuple,
)
from flapjax.utils.print_utils import warn
from flapjax.utils.utils import index_to_arr, make_pytree


@dataclass(frozen=True)
class OptionalJacobians:
    d_f_ext_dead_d_n: bool = False  # stiffness from dead loads
    d_f_grav_d_n: bool = False  # stiffness from gravitational loads
    d_f_gyr_d_q_dot: bool = (
        False  # derivative of gyroscopic forces with respect to Q_dot
    )
    d_f_int_d_p_d: bool = False  # geometric stiffness


@make_pytree
class StructureCase:
    """Object to hold the full state and forces of a structure analysis.

    A single instance may represent any of three flavours:

    * **Static**: dynamic-only fields (``v``, ``v_dot``, ``a``, ``f_iner_gyr``)
      are not set — their public properties return an all-zeros array of appropriate shape. Array shapes are ``(n_nodes, ...)``.
    * **Dynamic snapshot**: all dynamic fields populated for a single timestep. Array shapes are ``(n_nodes, ...)``.
    * **Dynamic trajectory**: all dynamic fields populated for multiple timesteps. ``t`` is a
      ``(n_tstep,)`` array and ``i_ts`` is ``None``. Array shapes are
      ``(n_tstep, n_nodes, ...)``.

    Use :attr:`is_dynamic` and :attr:`is_batched` to distinguish at runtime.
    """

    _static: ClassVar[tuple[str, ...]] = (
        "prescribed_dofs",
        "free_dofs",
        "local",
        "conn",
        "thrust_nodes",
        "thrust_direction",
        "i_ts",
    )

    def __init__(
        self,
        hg: Array,
        conn: tuple[tuple[int, int], ...],
        o0: Array,
        d: Array,
        eps: Array,
        varphi: Array,
        f_ext_follower: Array | None,
        f_ext_dead: Array | None,
        f_ext_aero: Array | None,
        f_grav: Array | None,
        f_int: Array,
        f_elem: Array,
        f_res: Array,
        t: Array,
        thrust: dict[str, Array],
        thrust_nodes: tuple[tuple[str, int], ...],
        thrust_direction: tuple[tuple[str, tuple[float, float, float]], ...],
        prescribed_dofs: tuple[int, ...] | Array,
        v: Array | None = None,
        v_dot: Array | None = None,
        a: Array | None = None,
        f_iner_gyr: Array | None = None,
        i_ts: int | None = None,
        local: bool = True,
        constraint_data: dict[str, dict[str, Array]] | None = None,
    ):
        self.hg: Array = hg
        self.conn: tuple[tuple[int, int], ...] = conn
        self.o0: Array = o0
        self.d: Array = d
        self.eps: Array = eps
        self.varphi: Array = varphi
        self._v: Array | None = v
        self._v_dot: Array | None = v_dot
        self._a: Array | None = a
        self.f_ext_follower: Array | None = f_ext_follower
        self.f_ext_dead: Array | None = f_ext_dead
        self.f_ext_aero: Array | None = f_ext_aero
        self.f_grav: Array | None = f_grav
        self.f_int: Array = f_int
        self.f_elem: Array = f_elem
        self._f_iner_gyr: Array | None = f_iner_gyr
        self.f_res: Array = f_res
        self.thrust: dict[str, Array] = thrust
        self.thrust_nodes: tuple[tuple[str, int], ...] = thrust_nodes
        self.thrust_direction: tuple[tuple[str, tuple[float, float, float]], ...] = (
            thrust_direction
        )
        self.t: Array = t
        self.i_ts: int | None = i_ts
        self.prescribed_dofs: tuple[int, ...] = input_dof_index_to_tuple(
            prescribed_dofs
        )
        self.free_dofs: tuple[int, ...] = get_solve_dofs(
            n_dof=varphi.shape[-2] * 6, prescribed_dofs=self.prescribed_dofs
        )
        self.local: bool = local
        self.constraint_data: dict[str, dict[str, Array]] = constraint_data if constraint_data is not None else {}

    @property
    def x(self) -> Array:
        return self.hg[..., :3, 3]

    @property
    def rmat(self) -> Array:
        return self.hg[..., :3, :3]

    @property
    def is_dynamic(self) -> bool:
        return self._v is not None

    def _zeros_nodal_6(self) -> Array:
        """Zero-filled ``(..., n_nodes, 6)`` array matching this case's batching."""
        return jnp.zeros((*self.hg.shape[:-2], 6))

    @property
    def v(self) -> Array:
        return self._v if self._v is not None else self._zeros_nodal_6()

    @v.setter
    def v(self, value: Array | None) -> None:
        self._v = value

    @property
    def v_dot(self) -> Array:
        return self._v_dot if self._v_dot is not None else self._zeros_nodal_6()

    @v_dot.setter
    def v_dot(self, value: Array | None) -> None:
        self._v_dot = value

    @property
    def a(self) -> Array:
        return self._a if self._a is not None else self._zeros_nodal_6()

    @a.setter
    def a(self, value: Array | None) -> None:
        self._a = value

    @property
    def f_iner_gyr(self) -> Array:
        return (
            self._f_iner_gyr if self._f_iner_gyr is not None else self._zeros_nodal_6()
        )

    @f_iner_gyr.setter
    def f_iner_gyr(self, value: Array | None) -> None:
        self._f_iner_gyr = value

    @property
    def is_batched(self) -> bool:
        return self.hg.ndim == 4

    @property
    def n_tstep(self) -> int:
        if not self.is_batched:
            raise ValueError("n_tstep only defined for batched Structure")
        return self.hg.shape[0]

    @overload
    def to_dynamic(self) -> StructureCase: ...

    @overload
    def to_dynamic(self, t: None) -> StructureCase: ...

    @overload
    def to_dynamic(self, t: Array) -> StructureCase: ...

    def to_dynamic(self, t: Array | None = None) -> StructureCase:
        """Convert static structure results to a dynamic snapshot (``t=None``) or
        a batched trajectory (``t`` provided), zeroing velocity/acceleration
        fields. Calling on a Structure that is already dynamic returns ``self``.
        """
        if self.is_dynamic:
            return self

        dyn_snapshot = StructureCase(
            hg=self.hg,
            conn=self.conn,
            o0=self.o0,
            d=self.d,
            eps=self.eps,
            varphi=self.varphi,
            v=self.v,
            v_dot=self.v_dot,
            a=self.a,
            f_ext_follower=self.f_ext_follower,
            f_ext_dead=self.f_ext_dead,
            f_ext_aero=self.f_ext_aero,
            f_grav=self.f_grav,
            f_int=self.f_int,
            f_elem=self.f_elem,
            f_iner_gyr=self.f_iner_gyr,
            f_res=self.f_res,
            thrust=self.thrust,
            thrust_nodes=self.thrust_nodes,
            thrust_direction=self.thrust_direction,
            t=jnp.array(0.0),
            i_ts=-1,
            prescribed_dofs=self.prescribed_dofs,
            constraint_data=self.constraint_data,
        )

        if t is None:
            return dyn_snapshot
        return StructureCase.initialise(
            initial_snapshot=dyn_snapshot,
            t=t,
            use_f_ext_aero=self.f_ext_aero is not None,
            use_f_ext_follower=self.f_ext_follower is not None,
            use_f_ext_dead=self.f_ext_dead is not None,
        )

    def to_static(self) -> StructureCase:
        """Return a static Structure, dropping velocity/acceleration
        fields. If already static, returns ``self``. For a batched trajectory,
        raises: use ``self[i_ts].to_static()`` to extract a single time step first.
        """
        if not self.is_dynamic:
            return self
        if self.is_batched:
            raise ValueError(
                "to_static() on a batched Structure is ambiguous; index a "
                "single time step first (e.g. `structure[i_ts].to_static()`)."
            )
        return StructureCase(
            hg=self.hg,
            conn=self.conn,
            o0=self.o0,
            d=self.d,
            eps=self.eps,
            varphi=self.varphi,
            f_ext_follower=self.f_ext_follower,
            f_ext_dead=self.f_ext_dead,
            f_ext_aero=self.f_ext_aero,
            f_grav=self.f_grav,
            f_int=self.f_int,
            f_elem=self.f_elem,
            f_res=self.f_res,
            t=self.t,
            thrust=self.thrust,
            thrust_nodes=self.thrust_nodes,
            thrust_direction=self.thrust_direction,
            prescribed_dofs=self.prescribed_dofs,
            constraint_data=self.constraint_data,
        )

    def __getitem__(self, i_ts: int) -> StructureCase:
        """Extract a dynamic snapshot at a specific time index from a batched
        trajectory."""
        if not self.is_batched:
            raise TypeError("__getitem__ only supported for batched Structure")
        assert self.t is not None
        return StructureCase(
            hg=self.hg[i_ts, ...],
            conn=self.conn,
            o0=self.o0,
            d=self.d[i_ts, ...],
            eps=self.eps[i_ts, ...],
            varphi=self.varphi[i_ts, ...],
            v=self.v[i_ts, ...],
            v_dot=self.v_dot[i_ts, ...],
            a=self.a[i_ts, ...],
            f_ext_follower=self.f_ext_follower[i_ts, ...]
            if self.f_ext_follower is not None
            else None,
            f_ext_dead=self.f_ext_dead[i_ts, ...]
            if self.f_ext_dead is not None
            else None,
            f_ext_aero=self.f_ext_aero[i_ts, ...]
            if self.f_ext_aero is not None
            else None,
            f_grav=self.f_grav[i_ts, ...] if self.f_grav is not None else None,
            f_int=self.f_int[i_ts, ...],
            f_elem=self.f_elem[i_ts, ...],
            f_iner_gyr=self.f_iner_gyr[i_ts, ...],
            f_res=self.f_res[i_ts, ...],
            thrust={k: v[i_ts, ...] for k, v in self.thrust.items()},
            thrust_nodes=self.thrust_nodes,
            thrust_direction=self.thrust_direction,
            t=self.t[i_ts],
            i_ts=i_ts,
            prescribed_dofs=self.prescribed_dofs,
            constraint_data={
                name: {k: v[i_ts, ...] for k, v in quantities.items()}
                for name, quantities in self.constraint_data.items()
            },
        )

    def get_full_states(self, i_ts: int | Array | None = None) -> StructureFullStates:
        if self.is_batched:
            if i_ts is None:
                raise ValueError("i_ts must be provided for batched Structure")
            return StructureFullStates(
                v=self._v[i_ts, ...] if self._v is not None else None,
                v_dot=self._v_dot[i_ts, ...] if self._v_dot is not None else None,
                varphi=self.varphi[i_ts, ...],
                hg=self.hg[i_ts, ...],
                eps=self.eps[i_ts, ...],
                f_elem=self.f_elem[i_ts, ...],
                f_res=self.f_res[i_ts, ...],
            )
        return StructureFullStates(
            v=self._v,
            v_dot=self._v_dot,
            varphi=self.varphi,
            hg=self.hg,
            eps=self.eps,
            f_elem=self.f_elem,
            f_res=self.f_res,
        )

    def get_minimal_states(self, i_ts: int | Array) -> StructureMinimalStates:
        if not self.is_batched:
            raise ValueError("get_minimal_states requires a batched Structure")
        return StructureMinimalStates(
            varphi=self.varphi[i_ts, ...],
            v=self.v[i_ts, ...],
            v_dot=self.v_dot[i_ts, ...],
            a=self.a[i_ts, ...],
            f_ext_aero=self.f_ext_aero[i_ts, ...]
            if self.f_ext_aero is not None
            else None,
        )

    @classmethod
    def initialise(
        cls,
        initial_snapshot: StructureCase,
        t: Array,
        use_f_ext_follower: bool,
        use_f_ext_dead: bool,
        use_f_ext_aero: bool,
    ) -> StructureCase:
        r"""
        Initialise a batched dynamic Structure from a single dynamic snapshot.
        :param initial_snapshot: Snapshot at initial time step (must be dynamic,
        i.e. ``is_dynamic == True`` and not batched).
        :param t: Time step array, ``(n_tstep, )``
        :param use_f_ext_follower: Whether to include follower force array
        :param use_f_ext_dead: Whether to include dead force array
        :param use_f_ext_aero: Whether to include aero force array
        :return: Batched Structure with arrays initialised to zero except for
        the first time step.
        """
        if not initial_snapshot.is_dynamic:
            raise ValueError(
                "initial_snapshot must be dynamic; call to_dynamic() first"
            )
        if initial_snapshot.is_batched:
            raise ValueError("initial_snapshot must be a single snapshot, not batched")

        n_node = initial_snapshot.hg.shape[0]
        n_elem = initial_snapshot.d.shape[0]
        n_tstep = t.shape[0]

        hg = jnp.zeros((n_tstep, n_node, 4, 4)).at[0, ...].set(initial_snapshot.hg)
        d = jnp.zeros((n_tstep, n_elem, 6)).at[0, ...].set(initial_snapshot.d)
        eps = jnp.zeros((n_tstep, n_elem, 6)).at[0, ...].set(initial_snapshot.eps)
        varphi = jnp.zeros((n_tstep, n_node, 6)).at[0, ...].set(initial_snapshot.varphi)
        v = jnp.zeros((n_tstep, n_node, 6)).at[0, ...].set(initial_snapshot.v)
        v_dot = jnp.zeros((n_tstep, n_node, 6)).at[0, ...].set(initial_snapshot.v_dot)
        a = jnp.zeros((n_tstep, n_node, 6)).at[0, ...].set(initial_snapshot.a)

        if use_f_ext_follower:
            f_ext_follower = jnp.zeros((n_tstep, n_node, 6))
            if initial_snapshot.f_ext_follower is not None:
                f_ext_follower = f_ext_follower.at[0, ...].set(
                    initial_snapshot.f_ext_follower
                )
        else:
            f_ext_follower = None

        if use_f_ext_dead:
            f_ext_dead = jnp.zeros((n_tstep, n_node, 6))
            if initial_snapshot.f_ext_dead is not None:
                f_ext_dead = f_ext_dead.at[0, ...].set(initial_snapshot.f_ext_dead)
        else:
            f_ext_dead = None

        if use_f_ext_aero:
            f_ext_aero = jnp.zeros((n_tstep, n_node, 6))
            if initial_snapshot.f_ext_aero is not None:
                f_ext_aero = f_ext_aero.at[0, ...].set(initial_snapshot.f_ext_aero)
        else:
            f_ext_aero = None

        f_grav = (
            jnp.zeros((n_tstep, n_node, 6)).at[0, ...].set(initial_snapshot.f_grav)
            if initial_snapshot.f_grav is not None
            else None
        )
        f_int = jnp.zeros((n_tstep, n_node, 6)).at[0, ...].set(initial_snapshot.f_int)
        f_elem = jnp.zeros((n_tstep, n_elem, 6)).at[0, ...].set(initial_snapshot.f_elem)
        f_iner_gyr = (
            jnp.zeros((n_tstep, n_node, 6)).at[0, ...].set(initial_snapshot.f_iner_gyr)
        )
        f_res = jnp.zeros((n_tstep, n_node, 6)).at[0, ...].set(initial_snapshot.f_res)

        thrust = {k: jnp.full(n_tstep, v) for k, v in initial_snapshot.thrust.items()}
        return cls(
            hg=hg,
            conn=initial_snapshot.conn,
            o0=initial_snapshot.o0,
            d=d,
            eps=eps,
            varphi=varphi,
            v=v,
            v_dot=v_dot,
            a=a,
            f_ext_follower=f_ext_follower,
            f_ext_dead=f_ext_dead,
            f_ext_aero=f_ext_aero,
            f_grav=f_grav,
            f_int=f_int,
            f_elem=f_elem,
            f_iner_gyr=f_iner_gyr,
            f_res=f_res,
            thrust=thrust,
            thrust_nodes=initial_snapshot.thrust_nodes,
            thrust_direction=initial_snapshot.thrust_direction,
            t=t,
            prescribed_dofs=initial_snapshot.prescribed_dofs,
        )

    def _transform(self, rmat: Array) -> None:
        r"""
        Transform orientation-dependent results between frames. Dynamic fields
        (``f_iner_gyr``, ``v``, ``v_dot``, ``a``) are transformed only when
        populated.
        :param rmat: Nodal rotations with shape matching ``rmat``.
        """
        extra: tuple[str, ...] = (
            ("f_iner_gyr", "v", "v_dot", "a") if self.is_dynamic else ()
        )
        apply_frame_transform(self, rmat, extra)

    def to_global(self) -> None:
        """Convert local structure results to global frame."""
        if not self.local:
            warn("Results already in global frame, skipping conversion.")
            return
        self.local = False
        self._transform(rmat=self.rmat)

    def to_local(self) -> None:
        """Convert global structure results to local frame."""
        if self.local:
            warn("Results already in local frame, skipping conversion.")
            return
        self.local = True
        if self.is_batched:
            rmat_t = jnp.transpose(self.rmat, (0, 1, 3, 2))
        else:
            rmat_t = jnp.transpose(self.rmat, (0, 2, 1))
        self._transform(rmat=rmat_t)

    @overload
    def plot(self, directory: os.PathLike | str, n_interp: int = 0) -> Path: ...

    @overload
    def plot(
        self,
        directory: os.PathLike | str,
        n_interp: int = 0,
        *,
        index: slice | Sequence[int] | int | Array | None = None,
    ) -> Path: ...

    def plot(
        self,
        directory: os.PathLike | str,
        n_interp: int = 0,
        *,
        index: slice | Sequence[int] | int | Array | None = None,
    ) -> Path:
        r"""
        Plot beam results to VTK/VTU files in the specified directory. For a
        batched Structure, a PVD is written alongside per-timestep VTUs.
        :param directory: Path to write files to.
        :param n_interp: Number of interpolation points to add between each element for smoother visualisation.
        :param index: For batched Structures only, time step indices to plot.
        """
        if self.is_batched:
            index_ = index_to_arr(index=index, n_entries=self.n_tstep)
            directory_path = Path(directory).resolve()
            directory_path.mkdir(parents=True, exist_ok=True)

            paths = [self[i_ts]._plot_single(directory, n_interp) for i_ts in index_]

            assert self.t is not None
            return write_pvd(directory, "beam_dynamic_ts", paths, list(self.t[index_]))

        if index is not None:
            raise ValueError("`index` is only used for batched Structure")

        if not self.is_dynamic:
            return self.to_dynamic()._plot_single(directory, n_interp)
        return self._plot_single(directory, n_interp)

    def _plot_single(self, directory: os.PathLike | str, n_interp: int) -> Path:
        """Create a single VTU for this snapshot."""
        # represent all vectors in the inertial frame
        data = deepcopy(self)
        data.to_global()

        # vectors making up local frame rotation matrices
        local_x = data.hg[:, :3, 0]
        local_y = data.hg[:, :3, 1]
        local_z = data.hg[:, :3, 2]

        # forcing data
        f_ext_follower = (
            data.f_ext_follower[:, :3] if data.f_ext_follower is not None else None
        )
        m_ext_follower = (
            data.f_ext_follower[:, 3:] if data.f_ext_follower is not None else None
        )
        f_ext_dead = data.f_ext_dead[:, :3] if data.f_ext_dead is not None else None
        m_ext_dead = data.f_ext_dead[:, 3:] if data.f_ext_dead is not None else None
        f_ext_aero = data.f_ext_aero[:, :3] if data.f_ext_aero is not None else None
        m_ext_aero = data.f_ext_aero[:, 3:] if data.f_ext_aero is not None else None
        f_ext_grav = data.f_grav[:, :3] if data.f_grav is not None else None
        m_ext_grav = data.f_grav[:, 3:] if data.f_grav is not None else None
        f_iner = data.f_iner_gyr[:, :3] if data.f_iner_gyr is not None else None
        m_iner = data.f_iner_gyr[:, 3:] if data.f_iner_gyr is not None else None
        f_int = data.f_int[:, :3]
        m_int = data.f_int[:, 3:]
        f_res = data.f_res[:, :3]
        m_res = data.f_res[:, 3:]

        # velocity and acceleration data
        v_lin = data.v[:, :3] if data.v is not None else None
        v_ang = data.v[:, 3:] if data.v is not None else None
        v_dot_lin = data.v_dot[:, :3] if data.v_dot is not None else None
        v_dot_ang = data.v_dot[:, 3:] if data.v_dot is not None else None

        # beam strain data
        eps_lin = data.eps[:, :3]
        eps_ang = data.eps[:, 3:]

        # element force data
        f_elem_lin = data.f_elem[:, :3]
        f_elem_ang = data.f_elem[:, 3:]

        # node and element numbers for plotting
        node_num = jnp.arange(data.hg.shape[0])
        elem_num = jnp.arange(len(data.conn))

        node_scalar_data = {"node_number": node_num}
        node_vector_data = {
            "local_x": local_x,
            "local_y": local_y,
            "local_z": local_z,
            "f_ext_follower": f_ext_follower,
            "m_ext_follower": m_ext_follower,
            "f_ext_dead": f_ext_dead,
            "m_ext_dead": m_ext_dead,
            "f_ext_aero": f_ext_aero,
            "m_ext_aero": m_ext_aero,
            "f_ext_grav": f_ext_grav,
            "m_ext_grav": m_ext_grav,
            "f_iner_gyr": f_iner,
            "m_iner": m_iner,
            "f_int": f_int,
            "m_int": m_int,
            "f_res": f_res,
            "m_res": m_res,
            "v_linear": v_lin,
            "v_angular": v_ang,
            "v_dot_linear": v_dot_lin,
            "v_dot_angular": v_dot_ang,
        }
        cell_scalar_data = {"element_number": elem_num}
        cell_vector_data = {
            "eps_linear": eps_lin,
            "eps_angular": eps_ang,
            "f_elem_linear": f_elem_lin,
            "f_elem_angular": f_elem_ang,
        }

        Path(directory).mkdir(parents=True, exist_ok=True)
        file_name = Path(directory).joinpath("beam")
        return plot_beam_to_vtk(
            hg=data.hg,
            conn=jnp.array(data.conn, dtype=int),
            o0=data.o0,
            n_interp=n_interp,
            filename=file_name,
            i_ts=data.i_ts,
            node_scalar_data=node_scalar_data,
            node_vector_data=node_vector_data,
            cell_scalar_data=cell_scalar_data,
            cell_vector_data=cell_vector_data,
        )


@make_pytree
class StructureMinimalStates:
    _static: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        varphi: Array | None,
        v: Array,
        v_dot: Array,
        a: Array,
        f_ext_aero: Array | None = None,
    ):
        self._varphi: Array | None = varphi
        self.v: Array = v
        self.v_dot: Array = v_dot
        self.a: Array = a
        self.f_ext_aero: Array | None = f_ext_aero

    @property
    def varphi(self) -> Array:
        if self._varphi is None:
            raise ValueError("varphi is None")
        return self._varphi

    @varphi.setter
    def varphi(self, varphi: Array) -> None:
        self._varphi = varphi

    @classmethod
    def from_mat(cls, stacked_mat: Array) -> StructureMinimalStates:
        return StructureMinimalStates(*stacked_mat.reshape(stacked_mat.shape[0], -1, 6))

    def to_mat(self) -> Array:
        out = jnp.stack((self.varphi, self.v, self.v_dot, self.a), 0)  # [4, n_nodes, 6]

        if self.f_ext_aero is not None:
            out = jnp.concatenate((out, self.f_ext_aero[None, ...]), 0)
        return out

    def ravel(self) -> Array:
        return self.to_mat().ravel()

    @property
    def n_states(self) -> int:
        return self.to_mat().size
