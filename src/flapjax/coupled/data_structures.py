from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Self, overload

from jax import Array
from jax import numpy as jnp

from flapjax.aero.data_structures import AeroCase
from flapjax.aero.gradients.data_structures import AeroDesignVariables, AeroFullStates
from flapjax.algebra.array_utils import ArrayList, ArrayListShape
from flapjax.coupled.gradients.data_structures import AeroelasticGradsToCompute
from flapjax.structure.data_structures import StructureCase, StructureMinimalStates
from flapjax.structure.gradients.data_structures import StructureDesignVariables
from flapjax.utils.data_structures import DesignVariables
from flapjax.utils.utils import make_pytree

if TYPE_CHECKING:
    from flapjax.coupled.coupled import BaseCoupledAeroelastic
    from flapjax.structure.data_structures import StructureFullStates
    from flapjax.structure.gradients.beam import BeamStructure


@make_pytree
class AeroelasticCase:
    """Coupled aeroelastic case, which is a wrapper around a ``StructureCase`` and a ``AeroCase`` pair.

    A single instance may represent any of three flavours (derived from the
    wrapped ``structure``):

    * **Static**: static snapshot ``structure`` and snapshot ``aero``.
    * **Dynamic snapshot**: dynamic snapshot``structure`` and snapshot
      ``aero``.
    * **Dynamic trajectory**: batched ``structure`` and batched ``aero``.

    Use :attr:`is_dynamic` and :attr:`is_batched` to distinguish at runtime.
    """

    _static: ClassVar[tuple[str, ...]] = ()

    def __init__(self, structure: StructureCase, aero: AeroCase):
        self.structure: StructureCase = structure
        self.aero: AeroCase = aero

    @property
    def is_dynamic(self) -> bool:
        return self.structure.is_dynamic

    @property
    def is_batched(self) -> bool:
        return self.structure.is_batched

    @overload
    def to_dynamic(self) -> AeroelasticCase: ...

    @overload
    def to_dynamic(self, t: None) -> AeroelasticCase: ...

    @overload
    def to_dynamic(self, t: Array) -> AeroelasticCase: ...

    def to_dynamic(self, t: Array | None = None) -> AeroelasticCase:
        """Convert a static snapshot to a dynamic snapshot (``t=None``) or a batched
        trajectory (``t`` provided). Calling on an already-dynamic case returns
        ``self``.
        """
        if self.is_dynamic:
            return self
        if t is None:
            return AeroelasticCase(
                structure=self.structure.to_dynamic(), aero=self.aero
            )
        return AeroelasticCase(
            structure=self.structure.to_dynamic(t),
            aero=self.aero.to_dynamic(i_ts=0, n_tstep=len(t)),
        )

    def to_static(self) -> AeroelasticCase:
        """Return a static AeroelasticCase, dropping the structure's
        velocity/acceleration fields.
        """
        if not self.is_dynamic:
            return self
        if self.is_batched:
            raise ValueError(
                "to_static() on a batched AeroelasticCase is ambiguous; index a "
                "single time step first (e.g. `case[i_ts].to_static()`)."
            )
        return AeroelasticCase(
            structure=self.structure.to_static(),
            aero=self.aero,
        )

    def __getitem__(self, i_ts: int) -> AeroelasticCase:
        """Extract a snapshot at a specific time index from a batched case."""
        if not self.is_batched:
            raise TypeError("__getitem__ only supported for batched AeroelasticCase")
        return AeroelasticCase(structure=self.structure[i_ts], aero=self.aero[i_ts])

    @classmethod
    def initialise(
        cls,
        initial_snapshot: AeroelasticCase,
        t: Array,
        use_f_ext_follower: bool,
        use_f_ext_dead: bool,
        structure: BeamStructure,
        zeta_b0: ArrayList,
    ) -> AeroelasticCase:
        """Build a batched AeroelasticCase from any single-timestep case."""
        if initial_snapshot.is_batched:
            if initial_snapshot.structure.n_tstep != 1:
                raise ValueError("initial_snapshot.structure.n_tstep != 1")
            if initial_snapshot.aero.n_tstep != 1:
                raise ValueError("initial_snapshot.aero.n_tstep != 1")
            init_struct: StructureCase = initial_snapshot.structure[0]
            init_aero: AeroCase = initial_snapshot.aero[0]
        elif not initial_snapshot.is_dynamic:
            init_struct = initial_snapshot.structure.to_dynamic(t=None)
            init_aero = initial_snapshot.aero
        else:
            init_struct = initial_snapshot.structure
            init_aero = initial_snapshot.aero

        struct_case = StructureCase.initialise(
            initial_snapshot=init_struct,
            t=t,
            use_f_ext_aero=True,
            use_f_ext_follower=use_f_ext_follower,
            use_f_ext_dead=use_f_ext_dead,
        )
        aero_case = AeroCase.initialise(initial_snapshot=init_aero, n_tstep=len(t))

        # compute aerodynamic forcing at timestep 0
        f_aero_init = aero_case.project_forcing_to_beam(
            i_ts=0,
            rmat=struct_case.hg[0, :, :3, :3],
            zeta_b0=zeta_b0,
            include_unsteady=False,
        )
        f_aero_local = structure.make_f_dead_ext(
            f_ext=f_aero_init, rmat=struct_case.hg[0, :, :3, :3]
        )

        if struct_case.f_ext_aero is None:
            raise ValueError("f_ext_aero cannot be None")

        struct_case.f_ext_aero = struct_case.f_ext_aero.at[0, ...].set(f_aero_local)

        return AeroelasticCase(structure=struct_case, aero=aero_case)

    def get_full_states(self, i_ts: int | Array | None = None) -> AeroelasticFullStates:
        r"""
        Get the full aeroelastic states. For a batched case, ``i_ts`` selects the
        timestep; for a snapshot, ``i_ts`` is ignored.
        """
        if self.is_batched:
            if i_ts is None:
                raise ValueError("i_ts must be provided for batched AeroelasticCase")
            return AeroelasticFullStates(
                structure=self.structure.get_full_states(i_ts=i_ts),
                aero=self.aero.get_states(i_ts=i_ts),
            )
        if self.structure.f_ext_aero is None:
            raise ValueError("f_ext_aero is None")
        return AeroelasticFullStates(
            structure=self.structure.get_full_states(),
            aero=self.aero.get_states(i_ts=0 if self.is_dynamic else None),
        )

    def get_minimal_states(self, i_ts: int | Array) -> AeroelasticMinimalStates:
        r"""Get minimal aeroelastic states at the given timestep (batched only)."""
        if not self.is_batched:
            raise TypeError(
                "get_minimal_states only supported for batched AeroelasticCase"
            )
        return AeroelasticMinimalStates(
            structure=self.structure.get_minimal_states(i_ts=i_ts),
            aero=self.aero.get_states(i_ts=i_ts),
        )

    def plot(
        self,
        directory: os.PathLike | str,
        index: int | Sequence[int] | Array | slice | None = None,
        n_interp: int = 0,
        plot_bound: bool = True,
        plot_wake: bool = True,
    ) -> tuple[Path, Sequence[Path]]:
        r"""Plot the aeroelastic case.
        :param directory: Directory to save the plots to.
        :param index: For batched cases, timestep indices to plot; ignored for
            snapshots.
        :param n_interp: Number of interpolation points for plotting the structure.
        :param plot_bound: Whether to plot the bound aerodynamic panels.
        :param plot_wake: Whether to plot the wake aerodynamic panels.
        :return: ``(structure_path, aero_paths)``. For batched cases the
            structure path is a PVD; for snapshots it is a single VTU.
        """
        if self.is_batched:
            struct_out: Path = self.structure.plot(
                directory=directory, n_interp=n_interp, index=index
            )
        else:
            struct_out = self.structure.plot(directory=directory, n_interp=n_interp)
        aero_out: Sequence[Path] = self.aero.plot(
            directory=directory,  # type: ignore
            plot_bound=plot_bound,
            plot_wake=plot_wake,
            index=index,
        )
        return struct_out, aero_out


@make_pytree
@dataclass
class AeroelasticFullStates:
    _static: ClassVar[tuple[str, ...]] = ()

    aero: AeroFullStates
    structure: StructureFullStates


@make_pytree
class AeroelasticMinimalStates:
    _static: ClassVar[tuple[str, ...]] = ()

    def __init__(self, structure: StructureMinimalStates, aero: AeroFullStates):
        self.structure: StructureMinimalStates = structure
        self.aero: AeroFullStates = aero

    @staticmethod
    def from_vector(
        vect: Array,
        n_dof: int,
        aero_shapes: OrderedDict[str, tuple[int, ...] | ArrayListShape | None],
    ) -> AeroelasticMinimalStates:
        struct = StructureMinimalStates.from_mat(vect[: 5 * n_dof].reshape(5, n_dof))
        aero = AeroFullStates.from_vector(vect[5 * n_dof :], aero_shapes)
        return AeroelasticMinimalStates(structure=struct, aero=aero)

    def ravel(self) -> Array:
        return jnp.concatenate([self.structure.ravel(), self.aero.ravel()])

    def to_free_dofs(self, solve_dofs_arr: Array) -> Array:
        # convert state object to vector with only free degrees of freedom
        struct_stack = self.structure.to_mat()  # (5, n_nodes, 6)
        struct_flat = struct_stack.reshape(struct_stack.shape[0], -1)  # (5, n_dof)
        struct_free = struct_flat[:4][:, solve_dofs_arr].ravel()  # (4 * n_solve)
        f_ext_aero_free = struct_flat[4, solve_dofs_arr]  # (n_solve)
        aero_flat = self.aero.ravel()  # (n_aero_states)
        return jnp.concatenate((struct_free, aero_flat, f_ext_aero_free))

    @property
    def n_states(self) -> int:
        return self.structure.n_states + self.aero.n_states


@make_pytree
class AeroelasticDesignVariables(DesignVariables):
    _static: ClassVar[tuple[str, ...]] = ("shapes", "n_x")

    def __init__(
        self,
        structure_dv: StructureDesignVariables,
        aero_dv: AeroDesignVariables,
    ):
        super().__init__()
        self.structure: StructureDesignVariables = structure_dv
        self.aero: AeroDesignVariables = aero_dv

        self.shapes: OrderedDict[
            str,
            tuple[int, ...]
            | ArrayListShape
            | OrderedDict[str, tuple[int, ...] | ArrayListShape]
            | None,
        ] = self.get_shapes()
        self.mapping, self.n_x = self.make_index_mapping()

    def to_dict(self) -> dict[str, Array | None]:
        return {
            **(self.structure.to_dict() if self.structure is not None else {}),
            **(self.aero.to_dict() if self.aero is not None else {}),
        }

    def split_adjoint(
        self, d_f_d_x: dict[str, Array | ArrayList | None], f_shape: tuple[int, ...]
    ) -> AeroelasticDesignVariables:
        struct_dv = StructureDesignVariables(
            **{k: v for k, v in d_f_d_x.items() if k in self.structure.to_dict()},
            f_shape=f_shape,
        )
        aero_dv = AeroDesignVariables(
            **{k: v for k, v in d_f_d_x.items() if k in self.aero.to_dict()},
            f_shape=f_shape,
        )
        return AeroelasticDesignVariables(structure_dv=struct_dv, aero_dv=aero_dv)

    def premultiply_adj(self, adj: Array) -> AeroelasticDesignVariables:
        return AeroelasticDesignVariables(
            structure_dv=self.structure.premultiply_adj(adj),
            aero_dv=self.aero.premultiply_adj(adj),
        )

    def __iadd__(self, other: AeroelasticDesignVariables) -> Self:
        self.structure += other.structure
        self.aero += other.aero
        return self

    def add_structure_dv(
        self, other: StructureDesignVariables
    ) -> AeroelasticDesignVariables:
        structure_dv = deepcopy(self.structure)
        structure_dv += other
        return AeroelasticDesignVariables(
            structure_dv=structure_dv,
            aero_dv=self.aero,
        )

    @staticmethod
    def zeros(
        system: BaseCoupledAeroelastic,
        case: AeroelasticCase,
        grads_to_compute: AeroelasticGradsToCompute | None,
        j_shape: tuple[int, ...],
    ) -> AeroelasticDesignVariables:
        return AeroelasticDesignVariables(
            structure_dv=StructureDesignVariables(
                x0=jnp.zeros((*j_shape, *system.structure.x0.shape))
                if grads_to_compute is None or grads_to_compute.structure.x0
                else None,
                orientation_euler=jnp.zeros((*j_shape, 3))
                if grads_to_compute is None
                or grads_to_compute.structure.orientation_euler
                else None,
                k_cs=jnp.zeros((*j_shape, *system.structure.k_cs.shape))
                if grads_to_compute is None or grads_to_compute.structure.k_cs
                else None,
                m_cs=jnp.zeros((*j_shape, *system.structure.m_cs.shape))
                if grads_to_compute is None or grads_to_compute.structure.m_cs
                else None,
                m_lumped=jnp.zeros((*j_shape, *system.structure.m_lumped.shape))
                if system.structure.use_lumped_mass
                and (grads_to_compute is None or grads_to_compute.structure.m_lumped)
                else None,
                f_ext_dead=jnp.zeros((*j_shape, *case.structure.f_ext_dead.shape))
                if case.structure.f_ext_dead is not None
                and (grads_to_compute is None or grads_to_compute.structure.f_ext_dead)
                else None,
                f_ext_follower=jnp.zeros(
                    (*j_shape, *case.structure.f_ext_follower.shape)
                )
                if case.structure.f_ext_follower is not None
                and (
                    grads_to_compute is None
                    or grads_to_compute.structure.f_ext_follower
                )
                else None,
                thrust_t={
                    k: jnp.zeros((*j_shape, *v.shape))
                    for k, v in case.structure.thrust.items()
                }
                if grads_to_compute is None or grads_to_compute.structure.thrust_t
                else None,
                f_shape=j_shape,
            ),
            aero_dv=AeroDesignVariables(
                zeta_b0=ArrayList(
                    [jnp.zeros((*j_shape, *arr.shape)) for arr in system.aero.zeta_b0]
                )
                if (grads_to_compute is None or grads_to_compute.aero.zeta_b0)
                else None,
                flowfield={
                    k: jnp.zeros((*j_shape, *v.shape))
                    for k, v in system.aero.flowfield.to_design_variables().items()
                }
                if (grads_to_compute is None or grads_to_compute.aero.flowfield)
                else None,
                cs_ang_t={
                    k: jnp.zeros((*j_shape, *v.shape))
                    for k, v in case.aero.cs_ang.items()
                }
                if (grads_to_compute is None or grads_to_compute.aero.cs_ang_t)
                else None,
                cs_vel_t={
                    k: jnp.zeros((*j_shape, *v.shape))
                    for k, v in case.aero.cs_vel.items()
                }
                if (grads_to_compute is None or grads_to_compute.aero.cs_vel_t)
                else None,
                f_shape=j_shape,
            ),
        )

    @classmethod
    def concatenate(
        cls, *dvs: AeroelasticDesignVariables
    ) -> AeroelasticDesignVariables:
        aero_dvs = AeroDesignVariables.concatenate(*[dv.aero for dv in dvs])
        structure_dv = StructureDesignVariables.concatenate(
            *[dv.structure for dv in dvs]
        )
        return AeroelasticDesignVariables(aero_dv=aero_dvs, structure_dv=structure_dv)

    def plot(
        self,
        case: AeroelasticCase,
        i_ts: int | None,
        directory: os.PathLike | str,
    ) -> Sequence[Path]:
        paths = []

        if case.is_batched:
            if i_ts is None:
                raise ValueError("Time step index must be specified")
            struct_snapshot: StructureCase = case.structure[i_ts]
            aero_snapshot: AeroCase = case.aero[i_ts]
        else:
            struct_snapshot = case.structure
            aero_snapshot = case.aero

        if self.structure is not None:
            paths.append(
                self.structure.plot(
                    case=struct_snapshot, directory=directory, n_interp=0
                )
            )
        if self.aero is not None:
            rmat_nodal = ArrayList(
                [struct_snapshot.hg[mp, :3, :3] for mp in case.aero.dof_mapping]
            )
            paths.extend(
                self.aero.plot(
                    aero_snapshot, directory=directory, rmat_nodal=rmat_nodal
                )
            )
        return paths
