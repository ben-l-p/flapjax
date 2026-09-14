from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Self

from jax import Array
from jax import numpy as jnp

from flapjax.aero.gradients.data_structures import ApproxJacobianEntry
from flapjax.algebra.array_utils import ArrayListShape
from flapjax.plotting.beam import plot_beam_to_vtk
from flapjax.utils.data_structures import DesignVariables
from flapjax.utils.utils import make_pytree

if TYPE_CHECKING:
    from flapjax.structure.data_structures import StructureCase


@make_pytree
@dataclass
class StructureFullStates:
    _static: ClassVar[tuple[str, ...]] = ()

    v: Array | None
    v_dot: Array | None
    varphi: Array
    hg: Array
    eps: Array
    f_elem: Array
    f_res: Array  # residual forcing at nodes, added as it is used for trim

    @property
    def x(self) -> Array:
        return self.hg[:, :3, 3]

    @property
    def rmat(self) -> Array:
        return self.hg[:, :3, :3]


@dataclass(frozen=True)
class StructureGradsToCompute:
    x0: bool = False
    orientation_euler: bool = False
    k_cs: bool = True
    m_cs: bool = True
    m_lumped: bool = False
    f_ext_follower: bool = False
    f_ext_dead: bool = False
    thrust_t: bool = False


@dataclass
class VarphiApprox:
    varphi_nm1: ApproxJacobianEntry = None
    varphi_n: ApproxJacobianEntry = None
    v_nm1: ApproxJacobianEntry = None
    a_nm1: ApproxJacobianEntry = None
    a_n: ApproxJacobianEntry = None


@dataclass
class VApprox:
    v_nm1: ApproxJacobianEntry = None
    v_n: ApproxJacobianEntry = None
    a_nm1: ApproxJacobianEntry = None
    a_n: ApproxJacobianEntry = None


@dataclass
class VDotApprox:
    varphi_nm1: ApproxJacobianEntry = None
    varphi_n: ApproxJacobianEntry = None
    v_nm1: ApproxJacobianEntry = None
    v_n: ApproxJacobianEntry = None
    v_dot_nm1: ApproxJacobianEntry = None
    v_dot_n: ApproxJacobianEntry = None
    dv: ApproxJacobianEntry = None


@dataclass
class AApprox:
    v_dot_nm1: ApproxJacobianEntry = None
    v_dot_n: ApproxJacobianEntry = None
    a_nm1: ApproxJacobianEntry = None
    a_n: ApproxJacobianEntry = None


@dataclass
class StructureJacobianApproximations:
    varphi: VarphiApprox = field(default_factory=VarphiApprox)
    v: VApprox = field(default_factory=VApprox)
    v_dot: VDotApprox = field(default_factory=VDotApprox)
    a: AApprox = field(default_factory=AApprox)


@make_pytree
class StructureDesignVariables(DesignVariables):
    _static: ClassVar[tuple[str, ...]] = ("f_shape", "f_size", "n_x", "shapes")

    def __init__(
        self,
        x0: Array | None,
        orientation_euler: Array | None,
        k_cs: Array | None,
        m_cs: Array | None,
        m_lumped: Array | None,
        f_ext_follower: Array | None,
        f_ext_dead: Array | None,
        thrust_t: dict[str, Array] | None,
        f_shape: tuple[int, ...],
    ):
        super().__init__()
        self.x0: Array | None = x0
        self.orientation_euler: Array | None = orientation_euler
        self.k_cs: Array | None = k_cs
        self.m_cs: Array | None = m_cs
        self.m_lumped: Array | None = m_lumped
        self.f_ext_follower: Array | None = f_ext_follower
        self.f_ext_dead: Array | None = f_ext_dead
        self.thrust_t: dict[str, Array] | None = thrust_t
        self.f_shape: tuple[int, ...] = f_shape
        self.f_size: int = prod(f_shape)

        self.shapes: OrderedDict[
            str,
            tuple[int, ...]
            | ArrayListShape
            | OrderedDict[str, tuple[int, ...] | ArrayListShape]
            | None,
        ] = self.get_shapes()
        self.mapping, self.n_x = self.make_index_mapping()

    def __iadd__(self, other: StructureDesignVariables) -> Self:
        if self.x0 is not None:
            assert other.x0 is not None
            self.x0 += other.x0
        if self.orientation_euler is not None:
            assert other.orientation_euler is not None
            self.orientation_euler += other.orientation_euler
        if self.k_cs is not None:
            assert other.k_cs is not None
            self.k_cs += other.k_cs
        if self.m_cs is not None:
            assert other.m_cs is not None
            self.m_cs += other.m_cs
        if self.m_lumped is not None:
            assert other.m_lumped is not None
            self.m_lumped += other.m_lumped
        if self.f_ext_follower is not None:
            assert other.f_ext_follower is not None
            self.f_ext_follower += other.f_ext_follower
        if self.f_ext_dead is not None:
            assert other.f_ext_dead is not None
            self.f_ext_dead += other.f_ext_dead
        if self.thrust_t is not None:
            assert other.thrust_t is not None
            for k in self.thrust_t:
                self.thrust_t[k] += other.thrust_t[k]
        return self

    def premultiply_adj(self, adj: Array) -> StructureDesignVariables:
        return StructureDesignVariables(
            x0=jnp.einsum("ij,j...->i...", adj, self.x0)
            if self.x0 is not None
            else None,
            orientation_euler=jnp.einsum("ij,j...->i...", adj, self.orientation_euler)
            if self.orientation_euler is not None
            else None,
            k_cs=jnp.einsum("ij,j...->i...", adj, self.k_cs)
            if self.k_cs is not None
            else None,
            m_cs=jnp.einsum("ij,j...->i...", adj, self.m_cs)
            if self.m_cs is not None
            else None,
            m_lumped=jnp.einsum("ij,j...->i...", adj, self.m_lumped)
            if self.m_lumped is not None
            else None,
            f_ext_follower=jnp.einsum("ij,j...->i...", adj, self.f_ext_follower)
            if self.f_ext_follower is not None
            else None,
            f_ext_dead=jnp.einsum("ij,j...->i...", adj, self.f_ext_dead)
            if self.f_ext_dead is not None
            else None,
            thrust_t={
                k: jnp.einsum("ij,j...->i...", adj, v) for k, v in self.thrust_t.items()
            }
            if self.thrust_t is not None
            else None,
            f_shape=(adj.shape[1],),
        )

    def zeros_like(self) -> StructureDesignVariables:
        return StructureDesignVariables(
            x0=jnp.zeros_like(self.x0) if self.x0 is not None else None,
            orientation_euler=jnp.zeros_like(self.orientation_euler)
            if self.orientation_euler is not None
            else None,
            k_cs=jnp.zeros_like(self.k_cs) if self.k_cs is not None else None,
            m_cs=jnp.zeros_like(self.m_cs) if self.m_cs is not None else None,
            m_lumped=jnp.zeros_like(self.m_lumped)
            if self.m_lumped is not None
            else None,
            f_ext_follower=jnp.zeros_like(self.f_ext_follower)
            if self.f_ext_follower is not None
            else None,
            f_ext_dead=jnp.zeros_like(self.f_ext_dead)
            if self.f_ext_dead is not None
            else None,
            thrust_t={k: jnp.zeros_like(v) for k, v in self.thrust_t.items()}
            if self.thrust_t is not None
            else None,
            f_shape=self.f_shape,
        )

    def plot(
        self,
        case: StructureCase,
        directory: os.PathLike | str,
        n_interp: int = 0,
    ) -> Path:
        if self.f_size != 1:
            raise ValueError("Can only plot gradients for scalar objective functions.")

        d_f_d_f_follower = (
            jnp.einsum("ijk,...ik->ij", case.hg[:, :3, :3], self.f_ext_follower[:, :3])
            if self.f_ext_follower is not None
            else None
        )
        d_f_d_m_follower = (
            jnp.einsum("ijk,...ik->ij", case.hg[:, :3, :3], self.f_ext_follower[:, 3:])
            if self.f_ext_follower is not None
            else None
        )
        d_f_d_f_dead = (
            jnp.einsum("ijk,...ik->ij", case.hg[:, :3, :3], self.f_ext_dead[:, :3])
            if self.f_ext_dead is not None
            else None
        )
        d_f_d_m_dead = (
            jnp.einsum("ijk,...ik->ij", case.hg[:, :3, :3], self.f_ext_dead[:, 3:])
            if self.f_ext_dead is not None
            else None
        )
        d_f_d_x0 = (
            jnp.einsum("ijk,...ik->ij", case.hg[:, :3, :3], self.x0)
            if self.x0 is not None
            else None
        )

        node_num = jnp.arange(case.hg.shape[0])  # (n_nodes, )
        elem_num = jnp.arange(len(case.conn))  # (n_elems, )

        node_scalar_data = {"node_number": node_num}
        node_vector_data = {
            "d_f_d_f_follower": d_f_d_f_follower,
            "d_f_d_m_follower": d_f_d_m_follower,
            "d_f_d_f_dead": d_f_d_f_dead,
            "d_f_d_m_dead": d_f_d_m_dead,
            "d_f_d_x0": d_f_d_x0,
        }
        cell_scalar_data = {"element_number": elem_num}
        cell_vector_data = {}

        Path(directory).mkdir(parents=True, exist_ok=True)
        file_name = Path(directory).joinpath("beam_gradient")  # default file name
        return plot_beam_to_vtk(
            hg=case.hg,
            conn=jnp.array(case.conn),
            o0=case.o0,
            n_interp=n_interp,
            filename=file_name,
            i_ts=None,
            node_scalar_data=node_scalar_data,
            node_vector_data=node_vector_data,
            cell_scalar_data=cell_scalar_data,
            cell_vector_data=cell_vector_data,
        )

    def to_dict(self) -> dict[str, Array | dict[str, Array] | None]:
        return {
            "x0": self.x0,
            "orientation_euler": self.orientation_euler,
            "k_cs": self.k_cs,
            "m_cs": self.m_cs,
            "m_lumped": self.m_lumped,
            "f_ext_follower": self.f_ext_follower,
            "f_ext_dead": self.f_ext_dead,
            "thrust_t": self.thrust_t,
        }
