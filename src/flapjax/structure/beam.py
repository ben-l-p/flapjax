from __future__ import annotations

import os
from collections.abc import Sequence
from functools import partial
from typing import TYPE_CHECKING, ClassVar, Literal, cast, overload

import equinox.internal as eqxi
import jax
from jax import Array, vmap
from jax import numpy as jnp
from jax.scipy.linalg import block_diag
from jax.scipy.spatial.transform import Rotation

from flapjax.aero.data_structures import AeroCase
from flapjax.algebra.array_utils import check_arr_dtype, check_arr_shape
from flapjax.algebra.se3 import (
    exp_se3,
    ha_to_ha_tilde,
    hg_to_d,
    p,
    rmat_to_ha_hat,
    t_se3,
)
from flapjax.algebra.so3 import vec_to_skew
from flapjax.plotting.modal import plot_modes_vtu
from flapjax.structure.constraints import HardConstraint, SoftConstraint
from flapjax.structure.data_structures import (
    OptionalJacobians,
    StructureCase,
    StructureMinimalStates,
)
from flapjax.structure.gradients.data_structures import (
    StructureDesignVariables,
    StructureGradsToCompute,
)
from flapjax.structure.linear.linear_beam import LinearBeam
from flapjax.structure.time_integration import TimeIntegrator
from flapjax.structure.utils import (
    _check_connectivity,
    _integrate_c_t,
    _integrate_m_l,
    _k_t_entry,
    _make_c_t_lumped,
    _n_elem_per_node,
    _split_connectivity,
    get_solve_dofs,
    transform_nodal_vect,
)
from flapjax.utils.constants import BASE_LEGENDRE_ORDER, BASE_LOBATTO_ORDER
from flapjax.utils.data_structures import ConvergenceSettings, ConvergenceStatus
from flapjax.utils.linear import conjugate_partner_mask
from flapjax.utils.print_utils import (
    get_verbosity,
    jax_print,
    map_verbosity_level,
    print_table_line,
    warn,
)
from flapjax.utils.utils import (
    check_type,
    make_pytree,
    nested_list_to_tuple,
)

if TYPE_CHECKING:
    from flapjax.aero.utils import DynamicAeroSolver
    from flapjax.aero.uvlm import UVLM
    from flapjax.coupled.data_structures import AeroelasticCase

DEFAULT_STRUCT_CONVERGENCE_SETTINGS = ConvergenceSettings(
    max_n_iter=25,
    rel_disp_tol=1e-5,
    abs_disp_tol=1e-7,
    rel_force_tol=1e-6,
    abs_force_tol=1e-8,
)


@make_pytree
class BaseBeamStructure:
    r"""
    Class to represent nonlinear beam structure model
    """

    _static: ClassVar[tuple[str, ...]] = (
        "n_nodes",
        "n_dof",
        "connectivity",
        "n_elem_per_node",
        "n_elem",
        "dof_per_elem",
        "y_vector_reference",
        "use_lumped_mass",
        "use_gravity",
        "k_cs_index",
        "m_cs_index",
        "m_lumped_index",
        "thrust_nodes",
        "thrust_direction",
        "optional_jacobians",
        "struct_convergence_settings",
        "relaxation_factor",
        "spectral_radius",
        "alpha_m",
        "beta_k",
        "_auto_node_sources",
    )

    @property
    def nodal_constraints(self) -> tuple[SoftConstraint, ...]:
        return tuple(
            c for c in self.constraints.values() if isinstance(c, SoftConstraint)
        )

    @property
    def multibody_constraints(self) -> tuple[HardConstraint, ...]:
        return tuple(
            c for c in self.constraints.values() if isinstance(c, HardConstraint)
        )

    @property
    def k_cs(self) -> Array:
        if self._k_cs is None:
            raise ValueError("k_cs has not been set")
        return self._k_cs

    @k_cs.setter
    def k_cs(self, value: Array | None) -> None:
        self._k_cs = value

    @property
    def m_cs(self) -> Array:
        if self._m_cs is None:
            raise ValueError("m_cs has not been set")
        return self._m_cs

    @m_cs.setter
    def m_cs(self, value: Array | None) -> None:
        self._m_cs = value

    @property
    def m_lumped(self) -> Array:
        if self._m_lumped is None:
            raise ValueError("m_lumped has not been set")
        return self._m_lumped

    @m_lumped.setter
    def m_lumped(self, value: Array | None) -> None:
        self._m_lumped = value

    @property
    def time_integrator(self) -> TimeIntegrator:
        if self._time_integrator is None:
            raise ValueError("time_integrator has not been set")
        return self._time_integrator

    @time_integrator.setter
    def time_integrator(self, value: TimeIntegrator | None) -> None:
        self._time_integrator = value

    def __init__(
        self,
        num_nodes: int,
        connectivity: Array,
        y_vector: Array,
        k_cs_index: Array | None = None,
        m_cs_index: Array | None = None,
        m_lumped_index: Array | None = None,
        gravity: Array | None = None,
        thrust_nodes: dict[str, int] | None = None,
        thrust_direction: dict[str, Array] | None = None,
        optional_jacobians: OptionalJacobians | None = None,
        relaxation_factor: float = 1.0,
        spectral_radius: float = 0.9,
        alpha_m: float = 0.0,
        beta_k: float = 0.0,
        struct_convergence_settings: ConvergenceSettings = DEFAULT_STRUCT_CONVERGENCE_SETTINGS,
        constraints: (dict[str, SoftConstraint | HardConstraint] | None) = None,
    ) -> None:
        r"""
        Initialise BaseBeamStructure class with all non-design parameters.
        :param num_nodes: Number of nodes in the structure.
        :param connectivity: Connectivity array, `(n_elem, 2)``.
        :param y_vector: Vector defining the y direction for each element, ``(n_elem, 3)``.
        :param k_cs_index: Array defining the index from the library of k_cs to use for each element, ``(n_elem, )``.
        If ``None``, all elements will use the first entry in the k_cs library.
        :param m_cs_index: Array defining the index from the library of m_cs to use for each element, ``(n_elem, )``.
        If ``None``, all elements will use the first entry in the m_cs library.
        :param m_lumped_index: Node index for nodes which are to have a lumped mass attached. The order is the same as
        that for the lumped mass data ``(n_lumped_mass, )``.
        :param gravity: Gravity vector in global reference frame, or None for no gravity_vec, ``(3, )``.
        :param thrust_nodes: Dictionary of thrust node names and their corresponding node indices, {keys, int}.
        :param thrust_direction: Dictionary of thrust node names and their corresponding thrust direction vectors,
        ``{keys, (3, )}``.
        :param optional_jacobians: Define which Jacobians contributions are to be used for solution.
        :param relaxation_factor: Relaxation factor which reduces the displacement update at each iteration. A value of
        1 is no relaxation, and a value of 0 is no update.
        :param spectral_radius: Spectral radius for structural time integrator, where a value of 0 is highly damped and
        a value of 1 is undamped.
        :param alpha_m: Mass-proportional Rayleigh damping coefficient.
        :param beta_k: Stiffness-proportional Rayleigh damping coefficient.
        :param struct_convergence_settings: Structure convergence settings.
        :param constraints: Named dict ``{name: constraint}``, or None, which add
        """

        check_type(num_nodes, int)

        if constraints is None:
            named_constraints: dict[str, SoftConstraint | HardConstraint] = {}
        elif isinstance(constraints, dict):
            named_constraints = constraints
        else:
            raise ValueError("Invalid constraint input")

        hard_constraints = tuple(
            c for c in named_constraints.values() if isinstance(c, HardConstraint)
        )

        check_arr_shape(connectivity, (None, 2), "connectivity")
        check_arr_dtype(connectivity, int, "connectivity")
        _check_connectivity(connectivity, num_nodes)

        auto_node_sources: list[int] = []
        conn_list: list[list[int]] = connectivity.tolist()

        # add extra nodes and alter connectivity to account for constraints with the auto-generate node behaviour
        for con in hard_constraints:
            if con.node_j is None and not con.is_grounded:
                node_j_new = num_nodes + len(auto_node_sources)
                con.node_j = node_j_new
                auto_node_sources.append(con.node_i)
                conn_list = _split_connectivity(conn_list, con.node_i, node_j_new)
        self._auto_node_sources: tuple[int, ...] = tuple(auto_node_sources)

        num_nodes += len(auto_node_sources)
        connectivity = jnp.array(conn_list, dtype=int)
        self.n_nodes: int = num_nodes
        self.n_dof: int = num_nodes * 6

        self.connectivity: tuple[tuple[int, int], ...] = nested_list_to_tuple(
            connectivity.tolist()
        )  # (n_elem, 2)
        self.n_elem_per_node: tuple[int] = tuple(
            _n_elem_per_node(connectivity=connectivity, n_nodes=num_nodes).tolist()
        )  # (n_nodes, )
        self.n_elem: int = connectivity.shape[0]

        self.dof_per_elem: tuple[tuple[float]] = nested_list_to_tuple(
            jnp.zeros((self.n_elem, 12), dtype=int)
            .at[:, :6]
            .set(6 * self.connectivity_arr[:, [0]] + jnp.arange(6)[None, :])
            .at[:, 6:]
            .set(6 * self.connectivity_arr[:, [1]] + jnp.arange(6)[None, :])
            .tolist()
        )

        # allow for a single y_vector to be broadcast to all elements
        if y_vector.shape == (3,):
            y_vector = y_vector[None, :]
        if y_vector.shape == (1, 3):
            y_vector = jnp.broadcast_to(y_vector, (self.n_elem, 3))

        # y vectors in reference unoriented configuration, and placeholder for oriented equivalent.
        check_arr_shape(y_vector, (self.n_elem, 3), "y_vector")
        self.y_vector_reference: tuple[tuple[tuple[float]]] = nested_list_to_tuple(
            y_vector.tolist()
        )
        self.y_vector: Array = jnp.zeros_like(jnp.array(y_vector))

        # initialise design variables with default values
        self.x0_reference: Array = jnp.zeros((num_nodes, 3))  # unoriented
        self.x0: Array = jnp.zeros((num_nodes, 3))  # oriented

        self.m_cs = None
        self.k_cs = None
        self.m_lumped = None
        self.use_lumped_mass: bool = m_lumped_index is not None

        # initialise auxiliary arrays
        self.o0: Array = jnp.zeros((self.n_elem, 3, 3))
        self.l0: Array = jnp.zeros(self.n_elem)
        self.d0: Array = jnp.zeros((self.n_elem, 6))

        # initialise undeformed algebra and group
        self.hg0_reference: Array = jnp.zeros((self.n_nodes, 4, 4))  # unoriented
        self.hg0: Array = jnp.zeros((self.n_nodes, 4, 4))  # oriented

        # grads inverse action for the reference rotations
        self.ad_inv_o0: Array = jnp.zeros((self.n_elem, 6, 6))

        # gravity_vec settings
        self.use_gravity: bool = gravity is not None and bool(jnp.any(gravity))
        if self.use_gravity:
            assert gravity is not None
            check_arr_shape(gravity, (3,), "gravity")
            self.gravity_vec: tuple[float, float, float] = tuple(gravity.tolist())
        else:
            self.gravity_vec = (0.0, 0.0, 0.0)

        # indexing
        if k_cs_index is None:
            k_cs_index_ = jnp.zeros(self.n_elem, dtype=int)
        else:
            check_arr_shape(k_cs_index, (self.n_elem,), "k_cs_index")
            check_arr_dtype(k_cs_index, int, "k_cs_index")
            k_cs_index_ = k_cs_index
        self.k_cs_index: tuple[int] = tuple(k_cs_index_.tolist())

        if m_cs_index is None:
            m_cs_index_ = jnp.zeros(self.n_elem, dtype=int)
        else:
            check_arr_shape(m_cs_index, (self.n_elem,), "m_cs_index")
            check_arr_dtype(m_cs_index, int, "m_cs_index")
            m_cs_index_ = m_cs_index
        self.m_cs_index: tuple[int] = tuple(m_cs_index_.tolist())

        self.m_lumped_index: tuple[int] | None = None
        if m_lumped_index is not None:
            check_arr_dtype(m_lumped_index, int, "m_lumped_index")
            if m_lumped_index.ndim not in (0, 1):
                raise ValueError("m_lumped_index.ndim must be 0 or 1.")
            self.m_lumped_index = tuple(jnp.atleast_1d(m_lumped_index).tolist())

        # add thrust
        self.thrust_nodes: tuple[tuple[str, int], ...] = ()
        self.thrust_direction: tuple[tuple[str, tuple[float, float, float]], ...] = ()
        if thrust_nodes is not None and thrust_direction is not None:
            if thrust_nodes.keys() != thrust_direction.keys():
                raise ValueError(
                    f"Mismatch in keys of thrust_nodes ({thrust_nodes.keys()}) and thrust_direction ({thrust_direction.keys()}))."
                )

            for k, v in thrust_direction.items():
                check_arr_shape(v, (3,), f"thrust_direction[{k}]")

            self.thrust_nodes = tuple([(k, v) for k, v in thrust_nodes.items()])
            self.thrust_direction = tuple(
                [
                    (k, nested_list_to_tuple((v / jnp.linalg.norm(v)).tolist()))
                    for k, v in thrust_direction.items()
                ]
            )  # make unit vectors
        elif thrust_nodes is not None or thrust_direction is not None:
            warn(
                "One of thrust_nodes or thrust_direction has not been passed. Running with no thrust nodes."
            )

        # set the reference thrust to be zero, which can be overwritten later
        self.thrust_reference: dict[str, Array] = {
            k: jnp.atleast_1d(1) for k in [k_ for k_, v in self.thrust_nodes]
        }

        # set the reference orientation, which can be overwritten later.
        self.orientation_euler: Array = jnp.zeros(3)
        self.orientation: Array = jnp.eye(3)

        self.optional_jacobians: OptionalJacobians = (
            optional_jacobians
            if optional_jacobians is not None
            else OptionalJacobians()
        )
        self.struct_convergence_settings: ConvergenceSettings = (
            struct_convergence_settings
        )
        self.relaxation_factor: float = relaxation_factor
        self.spectral_radius: float = spectral_radius
        self.alpha_m: float = float(alpha_m)
        self.beta_k: float = float(beta_k)

        self.time_integrator = None

        self.constraints: dict[str, SoftConstraint | HardConstraint] = named_constraints

    def set_design_variables(
        self,
        coords: Array,
        k_cs: Array,
        m_cs: Array | None,
        m_lumped: Array | None = None,
        orientation_euler: Array | None = None,
        thrust_reference: dict[str, Array | float] | None = None,
        *,
        remove_checks: bool = False,
    ) -> None:
        r"""
        Set design variables and compute initial configuration dependent quantities.
        :param coords: Node coordinates in the reference configuration, ``(n_nodes, 3)``.
        :param k_cs: Cross-section stiffness matrices, ``(n_entry, 6, 6)`` or ``(6, 6)``.
        :param m_cs: Cross-section mass matrices, ``(n_entry, 6, 6)`` or ``(6, 6)``.
        :param m_lumped: Lumped mass matrices at nodes, ``(n_entry, 6, 6)``.
        :param orientation_euler: Euler angles in radians which to rotate the reference configuration by, ``(3, )``. This rotation
        is performed about the origin, and will default to the identity is no Array is passed. These are rotated in
        z-y-x order.
        :param thrust_reference: Reference thrust magnitude, ``{keys, (1, )}``.
        :param remove_checks: Flag to ignore input checks, used when function is JIT compiled.
        """

        # orientation
        if orientation_euler is not None:
            check_arr_shape(orientation_euler, (3,), "orientation_euler")
            self.orientation_euler = orientation_euler
            self.orientation = Rotation.from_euler(
                seq="zyx", angles=orientation_euler
            ).as_matrix()

        # rotate the y vectors
        self.y_vector = jnp.einsum(
            "jk,ik->ij",
            self.orientation,
            self.y_vector_reference_arr,
        )

        # coordinates — auto-extend for nodes created by multibody constraints
        if self._auto_node_sources and coords.shape[0] == self.n_nodes - len(
            self._auto_node_sources
        ):
            coords = jnp.concatenate(
                [coords, coords[jnp.array(self._auto_node_sources)]],
                axis=0,
            )
        check_arr_shape(coords, (self.n_nodes, 3), "coords")
        self.x0_reference = coords
        self.x0 = jnp.einsum("jk,ik->ij", self.orientation, coords)

        # populate arrays
        if k_cs.ndim == 2:
            k_cs = k_cs[None, ...]
        check_arr_shape(k_cs, (None, 6, 6), "k_cs")

        if (
            not remove_checks
            and k_cs.shape[0] != jnp.unique_values(jnp.array(self.k_cs_index)).size
        ):
            warn(
                "Redundant values in k_cs which are not used for solution due to no corresponding entry in k_cs_index."
            )

        self.k_cs = k_cs
        if m_cs is None:
            if not remove_checks and self.use_gravity and m_lumped is None:
                warn(
                    "No mass matrices provided, but gravity is enabled. Assuming zero mass.",
                )
            m_cs_ = jnp.zeros((6, 6))
        else:
            m_cs_ = m_cs

        if m_cs_.ndim == 2:
            m_cs_ = m_cs_[None, ...]

        check_arr_shape(m_cs_, (None, 6, 6), "m_cs")

        if (
            not remove_checks
            and m_cs_.shape[0] != jnp.unique_values(jnp.array(self.m_cs_index)).size
            and m_cs is not None
        ):
            warn(
                "Redundant values in m_cs which are not used for solution due to no corresponding entry in "
                "m_cs_index."
            )

        self.m_cs = m_cs_

        # thrust
        if thrust_reference is not None:
            self.thrust_reference = {
                k: jnp.atleast_1d(v) for k, v in thrust_reference.items()
            }

            for k, v in self.thrust_reference.items():
                check_arr_shape(v, (1,), f"thrust_reference[{k}]")

        if m_lumped is not None:
            if not remove_checks:
                check_arr_shape(m_lumped, (None, 6, 6), "m_lumped")

                if self.m_lumped_index is None:
                    raise ValueError("m_lumped_index has not been set")

                if m_lumped.shape[0] != len(self.m_lumped_index):
                    raise ValueError(
                        "Number of entries in m_lumped does not match number of indices in m_lumped_index."
                    )

            self.m_lumped = m_lumped

        # obtain initial orientation and length
        x_elem = jnp.take(
            self.x0_reference, self.connectivity_arr, axis=0
        )  # (n_elem, 2, 3)
        dx = x_elem[:, 1, :] - x_elem[:, 0, :]  # (n_elem, 3)

        # ensure out-of-plane vector and beam vector are not collinear
        if not remove_checks and jnp.any(
            jnp.linalg.norm(jnp.cross(dx, self.y_vector_reference_arr, 1, 1), axis=-1)
            < 1e-6
        ):
            raise ValueError(
                "y_vector is collinear with beam element direction for at least one element. "
                "Please provide a different y_vector."
            )

        self.l0 = jnp.linalg.norm(dx, axis=-1)  # (n_elem,)
        self.d0 = self.d0.at[:, 0].set(self.l0)

        dx_unit = dx / self.l0[:, None]  # unit vector in beam direction, (n_elem, 3)
        dz = jnp.cross(
            dx_unit, self.y_vector_reference_arr, axis=-1
        )  # vector in plane(n_elem, 3)
        dz_unit = dz / jnp.linalg.norm(dz, axis=-1)[:, None]  # (n_elem, 3)

        dy_unit = jnp.cross(dz_unit, dx_unit)

        self.o0 = self.o0.at[..., 0].set(dx_unit)
        self.o0 = self.o0.at[..., 1].set(dy_unit)
        self.o0 = self.o0.at[..., 2].set(dz_unit)

        self.ad_inv_o0 = vmap(rmat_to_ha_hat)(jnp.transpose(self.o0, (0, 2, 1)))

        # set unoriented initial coordinates
        self.hg0_reference = jnp.broadcast_to(
            jnp.eye(4)[None, ...], (self.n_nodes, 4, 4)
        )  # (n_nodes, 4, 4)
        self.hg0_reference = self.hg0_reference.at[:, :3, 3].set(self.x0_reference)

        # set oriented initial coordinates
        self.hg0 = self.hg0.at[:, :3, :3].set(
            jnp.broadcast_to(self.orientation[None, ...], (self.n_nodes, 3, 3))
        )  # (n_nodes, 4, 4)
        self.hg0 = self.hg0.at[:, :3, 3].set(self.x0)
        self.hg0 = self.hg0.at[:, 3, 3].set(1.0)

        # add reference frames to the nodal constraints
        for con in self.nodal_constraints:
            con.resolve_hg_ref(self.hg0)
        for con in self.multibody_constraints:
            con.resolve_hg_ref(self.hg0)

    def get_design_variables(
        self,
        struct_case: StructureCase,
        thrust_t: dict[str, Array],
        grads_to_compute: StructureGradsToCompute | None,
    ) -> StructureDesignVariables:
        r"""
        Obtain the design variables for the structural problem. As the external forcing is defined for each solve, the
        chosen forcing is required as input.
        :param struct_case: Structural case
        :param thrust_t: Thrust time history, {keys, ``(n_tstep,)``}.
        :param grads_to_compute: Data structure which describes which design variables should be obtained. If none, all
        variables are obtained.
        :return: StructureDesignVariables dataclass containing design variables
        """

        # struct_case.f_ext_dead is stored in local frame: f_local = R^T @ f_global,
        # so recover f_global = R @ f_local
        hg = struct_case.hg
        if hg.ndim == 4:  # batched case: (n_tstep, n_nodes, 4, 4)
            rmat = hg[:, :, :3, :3]
        else:  # snapshot: (n_nodes, 4, 4)
            rmat = hg[:, :3, :3]
        f_ext_dead_global = (
            transform_nodal_vect(struct_case.f_ext_dead, rmat)
            if struct_case.f_ext_dead is not None
            else None
        )
        if isinstance(grads_to_compute, StructureGradsToCompute):
            return StructureDesignVariables(
                x0=self.x0 if grads_to_compute.x0 else None,
                orientation_euler=self.orientation_euler
                if grads_to_compute.orientation_euler
                else None,
                m_cs=self.m_cs if grads_to_compute.m_cs else None,
                k_cs=self.k_cs if grads_to_compute.k_cs else None,
                m_lumped=self._m_lumped if grads_to_compute.m_lumped else None,
                f_ext_dead=f_ext_dead_global if grads_to_compute.f_ext_dead else None,
                f_ext_follower=struct_case.f_ext_follower
                if grads_to_compute.f_ext_follower
                else None,
                thrust_t=thrust_t if grads_to_compute.thrust_t else None,
                f_shape=(),
            )
        else:
            return StructureDesignVariables(
                x0=self.x0,
                orientation_euler=self.orientation_euler,
                m_cs=self.m_cs,
                k_cs=self.k_cs,
                m_lumped=self._m_lumped,
                f_ext_dead=f_ext_dead_global,
                f_ext_follower=struct_case.f_ext_follower,
                thrust_t=thrust_t,
                f_shape=(),
            )

    def reference_configuration(
        self,
        prescribed_dofs: Sequence[int] | Array | slice | int = (),
        use_f_ext_follower: bool = True,
        use_f_ext_dead: bool = True,
        use_f_aero: bool = True,
        use_f_grav: bool = True,
    ) -> StructureCase:
        r"""
        Get the reference configuration of the structure.
        :param prescribed_dofs: Prescribed degrees of freedom, which are not solved for. Defaults to no prescribed DoFs.
        :param use_f_ext_follower: Whether to include follower forces in the reference configuration.
        :param use_f_ext_dead: Whether to include dead forces in the reference configuration.
        :param use_f_aero: Whether to include aerodynamic forces in the reference configuration.
        :param use_f_grav: Whether to include gravitational forces in the reference configuration.
        :return: Structure dataclass containing reference configuration.
        """
        prescribed_dofs = self.make_prescribed_dofs_tuple(prescribed_dofs)
        return StructureCase(
            hg=self.hg0,
            conn=self.connectivity,
            o0=self.o0,
            d=self.d0,
            eps=jnp.zeros((self.n_elem, 6)),
            varphi=jnp.zeros((self.n_nodes, 6)),
            f_ext_follower=jnp.zeros((self.n_nodes, 6)) if use_f_ext_follower else None,
            f_ext_dead=jnp.zeros((self.n_nodes, 6)) if use_f_ext_dead else None,
            f_ext_aero=jnp.zeros((self.n_nodes, 6)) if use_f_aero else None,
            f_grav=jnp.zeros((self.n_nodes, 6)) if use_f_grav else None,
            f_int=jnp.zeros((self.n_nodes, 6)),
            f_elem=jnp.zeros((self.n_elem, 6)),
            f_res=jnp.zeros((self.n_nodes, 6)),
            thrust=self.thrust_reference,
            thrust_direction=self.thrust_direction,
            thrust_nodes=self.thrust_nodes,
            local=True,
            prescribed_dofs=prescribed_dofs,
            t=jnp.zeros(1),
        )

    @property
    def connectivity_arr(self) -> Array:
        if len(self.connectivity):
            return jnp.array(self.connectivity, dtype=int)
        else:
            return jnp.zeros((0, 2), dtype=int)  # special case for no beam elements

    @property
    def y_vector_reference_arr(self) -> Array:
        if len(self.y_vector_reference):
            return jnp.array(self.y_vector_reference)
        else:
            return jnp.zeros((0, 3))

    @property
    def m_lumped_index_arr(self) -> Array:
        return jnp.atleast_1d(jnp.array(self.m_lumped_index))

    @property
    def dof_per_elem_arr(self) -> Array:
        if len(self.dof_per_elem):
            return jnp.array(self.dof_per_elem, dtype=int)
        else:
            return jnp.zeros((0, 12), dtype=int)

    def compute_varphi_from_hg(self, hg: Array) -> Array:
        r"""
        Calculate the twist vector from the reference configuration to hg
        :param hg: Deformed coordinates, ``(n_nodes, 4, 4)``
        :return: Vector of twists, ``(n_nodes, 6)``
        """
        return vmap(hg_to_d, (0, 0), 0)(self.hg0, hg)

    def compute_hg_from_varphi(self, varphi: Array) -> Array:
        exp_varphi = vmap(exp_se3)(varphi)  # [n_nodes, 4, 4]
        return jnp.einsum("ijk,ikl->ijl", self.hg0, exp_varphi)

    def assemble_matrix_from_entries(self, entries: Array) -> Array:
        r"""
        Assemble global matrix from element entries
        :param entries: Array of element matrix entries, ``(n_elem, 12, 12)``
        :return: System global matrix, ``(n_dof, n_dof)``
        """

        row_idx = jnp.broadcast_to(
            self.dof_per_elem_arr[:, :, None], (self.n_elem, 12, 12)
        )
        col_idx = jnp.broadcast_to(
            self.dof_per_elem_arr[:, None, :], (self.n_elem, 12, 12)
        )
        return (
            jnp.zeros((self.n_dof, self.n_dof))
            .at[row_idx.ravel(), col_idx.ravel()]
            .add(entries.ravel())
        )

    def assemble_vector_from_entries(self, entries: Array) -> Array:
        r"""
        Assemble global vector from element entries
        :param entries: Array of element vector entries, ``(n_elem, 12)``
        :return: System global vector, ``(n_dof, )``
        """

        vect = jnp.zeros(self.n_dof)
        vect = vect.at[self.dof_per_elem_arr[:, :6]].add(entries[:, :6])
        return vect.at[self.dof_per_elem_arr[:, 6:]].add(entries[:, 6:])

    def add_lumped_contributions_to_arr(self, arr: Array, lumped_arr: Array) -> Array:
        r"""
        Add lumped contributions to an array
        :param arr: Full array, ``(6*n_node, 6*n_node)``
        :param lumped_arr: Lumped contributions, ``(n_lump, 6, 6)``
        :return: In-place updated array, ``(6*n_node, 6*n_node)``
        """

        assert self.m_lumped_index is not None

        def add_block(carry, x):
            node_idx, block = x
            dofs = node_idx * 6 + jnp.arange(6)
            return carry.at[jnp.ix_(dofs, dofs)].add(block), None

        arr, _ = jax.lax.scan(
            add_block, arr, (jnp.array(self.m_lumped_index), lumped_arr)
        )
        return arr

    def add_lumped_contributions_to_vec(self, vec: Array, lumped_vec: Array) -> Array:
        r"""
        Add lumped contributions to an array
        :param vec: Full vector, ``(6*n_node, )``
        :param lumped_vec: Lumped contributions, ``(n_lump, 6)``
        :return: In-place updated vector, ``(6*n_node, )``.
        """

        assert self.m_lumped_index is not None

        idx = (
            jnp.array(self.m_lumped_index)[:, None] * 6 + jnp.arange(6)[None, :]
        ).ravel()  # (n_lump * 6,)

        return vec.at[idx].add(lumped_vec)

    def _make_load_steps_f(
        self, f: Array | None, weighting: Array, apply_alpha_weighting: bool
    ) -> Array | None:
        r"""
        This also includes the effect of the time integrator
        :param f: Forcing, ``(n_nodes, 6)``.
        :param weighting: Vector of load step weightings, ``(n_load_steps, )``.
        :param apply_alpha_weighting: If true, compute the forcing at the alpha step.
        :return:
        """
        if f is not None:
            if apply_alpha_weighting:
                f = (
                    jnp.zeros((f.shape[0], self.n_nodes, 6))
                    .at[1:, ...]
                    .set(
                        self.time_integrator.compute_f_alpha(
                            f_nm1=f[:-1, ...], f_n=f[1:, ...]
                        )
                    )
                )  # (n_tstep, n_nodes, 6)

            f_steps = jnp.einsum("i,...->i...", weighting, f)  # (load_steps, ...)
        else:
            f_steps = None
        return f_steps

    @staticmethod
    def make_f_ext_dead_tot(
        f_ext_dead: Array | None,
        f_ext_aero: Array | None,
        i_load_step: int | None,
    ) -> Array | None:
        idx = (i_load_step, ...) if i_load_step is not None else (...,)

        if f_ext_dead is None and f_ext_aero is None:
            return None
        elif f_ext_dead is None and f_ext_aero is not None:
            return f_ext_aero[idx]
        elif f_ext_dead is not None and f_ext_aero is None:
            return f_ext_dead[idx]
        else:
            assert f_ext_dead is not None and f_ext_aero is not None  # for type checker
            return f_ext_dead[idx] + f_ext_aero[idx]

    def make_k_t(
        self,
        d: Array,
        p_d: Array,
        eps: Array,
    ) -> Array:
        r"""
        Assemble tangent stiffness matrix as a function of the element relative configuration vectors
        :param d: Element relative configuration, ``(n_elem, 6)``.
        :param p_d: P(d) operator, ``(n_elem, 6, 12)``.
        :param eps: Element strains, ``(n_elem, 6)``.
        :return: Elementwise stiffness matrix entries, ``(n_elem, 12, 12)``.
        """
        # compute stiffness matrix entries
        return vmap(
            partial(
                _k_t_entry,
                include_geometric=self.optional_jacobians.d_f_int_d_p_d,
            ),
            (0, 0, 0, 0, 0, 0),
            0,
        )(
            d,
            p_d,
            self.l0,
            eps,
            self.k_cs[self.k_cs_index, ...],
            self.ad_inv_o0,
        )  # (n_elem, 12, 12)

    def _make_k_t_dead(self, rmat: Array, f_ext_dead: Array) -> Array:
        r"""
        Compute the contribution to the stiffness matrix from dead external forces.
        :param rmat: Rotation matrices at nodes, ``(n_node, 3, 3)``
        :param f_ext_dead: External dead forces in global reference, ``(n_node, 6)``
        :return: Stiffness matrix contribution from dead forces, ``(n_node, 6, 6)``
        """
        k_t = jnp.zeros((self.n_nodes, 6, 6))

        k_t = k_t.at[:, :3, 3:].set(
            -vmap(vec_to_skew)(jnp.einsum("ikj,ik->ij", rmat, f_ext_dead[:, :3]))
        )
        k_t = k_t.at[:, 3:, 3:].set(
            -vmap(vec_to_skew)(jnp.einsum("ikj,ik->ij", rmat, f_ext_dead[:, 3:]))
        )

        return k_t

    def _make_k_t_grav(self, d: Array, p_d: Array, rmat: Array, m_t: Array) -> Array:
        r"""
        Compute the contribution to the stiffness matrix from gravity forces.
        :param d: Element relative configuration, ``(n_elem, 6)``
        :param p_d: P(d) operator, ``(n_elem, 6, 12)``
        :param rmat: Nodal rotation matrices, ``(n_node, 3, 3)``
        :param m_t: Disassembled system mass matrix, ``(n_elem, 12, 12)``
        :return: Stiffness matrix contribution from gravity forces, ``(n_elem, 12, 12)``
        """

        # perturbations in mass matrix integration
        g_ab = jnp.zeros(12)
        g_ab = g_ab.at[:3].set(jnp.array(self.gravity_vec))
        g_ab = g_ab.at[6:9].set(jnp.array(self.gravity_vec))
        p_d_g = jnp.einsum("ijk,k->ij", p_d, g_ab)  # (n_elem, 6)

        # computes dm/dd @ p @ g_ab
        d_mg = vmap(
            lambda m_cs_, d_, ad_, l_, p_d_g_: jax.jvp(
                lambda d__: _integrate_m_l(
                    m_cs_, d__, ad_, l_, int_order=BASE_LOBATTO_ORDER
                ),
                primals=[d_],
                tangents=[p_d_g_],
            )[1],
            (0, 0, 0, 0, 0),
            0,
        )(self.m_cs[self.m_cs_index, ...], d, self.ad_inv_o0, self.l0, p_d_g)

        # perturbations in gravity direction, (n_nodes, 3, 3)
        d_g_d_omega = vmap(vec_to_skew, 0, 0)(
            jnp.einsum("ikj,k->ij", rmat, jnp.array(self.gravity_vec))
        )

        # adding terms of (n_elem, 12, 3)
        d_mg = d_mg.at[:, :, 3:6].add(
            jnp.einsum(
                "ijk,ikl->ijl",
                m_t[:, :, :3],
                d_g_d_omega[self.connectivity_arr[:, 0], :, :],
            )
        )
        d_mg = d_mg.at[:, :, 9:].add(
            jnp.einsum(
                "ijk,ikl->ijl",
                m_t[:, :, 6:9],
                d_g_d_omega[self.connectivity_arr[:, 1], :, :],
            )
        )

        return d_mg

    def _make_k_t_grav_lumped(self, rmat: Array) -> Array:
        r"""
        Compute the contribution to the stiffness matrix from gravity forces for the lumped masses.
        :param rmat: Nodal rotation matrices, ``(n_node, 3, 3)``
        :return: Stiffness contribution from gravity forces for lumped masses, ``(n_lumped, 6, 6)``
        """
        # (n_lumped, 3, 3)
        d_g_d_omega = vmap(vec_to_skew, 0, 0)(
            jnp.einsum(
                "ikj,k->ij", rmat[self.m_lumped_index, ...], jnp.array(self.gravity_vec)
            )
        )

        return (
            jnp.zeros_like(self.m_lumped)
            .at[:, :, 3:]
            .set(jnp.einsum("ijk,ikl->ijl", self.m_lumped[:, :, 3:], d_g_d_omega))
        )

    def make_k_t_full(
        self,
        d: Array,
        p_d: Array,
        eps: Array,
        f_ext_dead: Array | None,
        rmat: Array,
        m_t: Array | None,
    ) -> Array:
        r"""
        Compute the full tangent stiffness matrix, with contributions from stiffness, dead forces and gravity.
        :param d: Element relative configuration, ``(n_elem, 6)``.
        :param p_d: P(d) operator, ``(n_elem, 6, 12)``.
        :param eps: Strain vectors, ``(n_elem, 6)``.
        :param f_ext_dead: External dead forces in global reference, ``(n_node, 6)``.
        :param rmat: Nodal rotation matrices, ``(n_node, 3, 3)``.
        :param m_t: Disassembled system mass matrix, ``(n_elem, 12, 12)``.
        :return: Tangent stiffness matrix with all contributions, ``(n_dof, n_dof)``.
        """

        k_t = self.assemble_matrix_from_entries(self.make_k_t(d, p_d, eps))
        if f_ext_dead is not None and self.optional_jacobians.d_f_ext_dead_d_n:
            k_t += block_diag(*self._make_k_t_dead(rmat, f_ext_dead))

        if self.use_gravity and self.optional_jacobians.d_f_grav_d_n:
            if m_t is None:
                raise ValueError("m_t needs to be provided")
            k_t += self.assemble_matrix_from_entries(
                self._make_k_t_grav(d, p_d, rmat, m_t)
            )
            if self.use_lumped_mass:
                k_t_lumped = self._make_k_t_grav_lumped(rmat)
                k_t = self.add_lumped_contributions_to_arr(
                    arr=k_t, lumped_arr=k_t_lumped
                )
        return k_t

    def make_m_t(
        self, d: Array, int_order: Literal[3, 4, 5] = BASE_LOBATTO_ORDER
    ) -> Array:
        r"""
        Assemble tangent mass matrix as a function of the element relative configuration vectors. This does not include
        the lumped mass contribution.
        :param d: Element relative configuration, ``(n_elem, 6)``
        :param int_order: Integration order for mass matrix computation
        :return: Elementwise mass matrix, ``(n_elem, 12, 12)``
        """
        return vmap(partial(_integrate_m_l, int_order=int_order), (0, 0, 0, 0), 0)(
            self.m_cs[self.m_cs_index, ...], d, self.ad_inv_o0, self.l0
        )

    def make_nodal_m_k(
        self,
        case: StructureCase,
        int_order: Literal[3, 4, 5] = BASE_LOBATTO_ORDER,
    ) -> tuple[Array, Array]:
        r"""
        Create the global mass and stiffness matrices for a given static structure case. These can be used for modal
        analysis or other purposes. These matrices are the Jacobians of the local forcing residual with respect to
        global perturbations in acceleration and displacement, respectively.
        :param case: Static structure case for which to compute the global mass and stiffness matrices.
        :param int_order: Integration order for mass matrix computation.
        :return: Global mass and stiffness matrices, ``(n_free_dof, n_free_dof)``.
        """
        # extract variables from case
        d = case.d
        eps = self.make_eps(d=d)
        p_d = self.make_p_d(d=d)
        t_varphi = vmap(t_se3)(case.varphi)  # (n_node, 6, 6)
        rmat = case.hg[:, :3, :3]  # (n_node, 3, 3)

        # get dead external forcing as this has a stiffness contribution
        f_ext_dead_local: Array | None
        if case.f_ext_dead is not None and case.f_ext_aero is not None:
            f_ext_dead_local = case.f_ext_dead + case.f_ext_aero
        else:
            f_ext_dead_local = (
                case.f_ext_dead if case.f_ext_dead is not None else case.f_ext_aero
            )

        # convert to global frame, as it required for creating the stiffness matrix
        f_ext_dead: Array | None = (
            transform_nodal_vect(f_ext_dead_local, rmat)
            if f_ext_dead_local is not None
            else None
        )
        free_dofs = jnp.array(
            get_solve_dofs(n_dof=self.n_dof, prescribed_dofs=case.prescribed_dofs)
        )

        def transform_mat_to_global(mat: Array) -> Array:
            # function to rotate a forcing Jacobian matrix from the local frame to the global frame.
            mat_reshaped = mat.reshape(self.n_nodes, 6, self.n_dof)
            m_lin = jnp.einsum("nij,njk->nik", rmat, mat_reshaped[:, :3, :])
            m_rot = jnp.einsum("nij,njk->nik", rmat, mat_reshaped[:, 3:, :])
            return jnp.concatenate((m_lin, m_rot), axis=1).reshape(
                self.n_dof, self.n_dof
            )

        # mass
        m_t = self.assemble_matrix_from_entries(
            self.make_m_t(d=d, int_order=int_order)
        )  # (n_dof, n_dof)
        if self.use_lumped_mass:
            m_t = self.add_lumped_contributions_to_arr(
                arr=m_t, lumped_arr=self.m_lumped
            )

        m_modal_full = transform_mat_to_global(
            mat=jnp.einsum("ijk,jkl->ijl", m_t.reshape(self.n_dof, -1, 6), t_varphi)
        )

        m_modal = m_modal_full.reshape(self.n_dof, self.n_dof)[
            jnp.ix_(free_dofs, free_dofs)
        ]

        # stiffness
        k_t = self.make_k_t_full(
            d=case.d, p_d=p_d, eps=eps, f_ext_dead=f_ext_dead, rmat=rmat, m_t=m_t
        )

        k_modal_full = transform_mat_to_global(
            mat=jnp.einsum("ijk,jkl->ijl", k_t.reshape(self.n_dof, -1, 6), t_varphi)
        )

        k_modal = k_modal_full.reshape(self.n_dof, self.n_dof)[
            jnp.ix_(free_dofs, free_dofs)
        ]

        return m_modal, k_modal

    def make_modal_m_k(
        self,
        case: StructureCase,
        n_modes: int,
        remove_complex_conjugate: bool = True,
        int_order: Literal[3, 4, 5] = BASE_LOBATTO_ORDER,
    ) -> tuple[Array, Array, Array]:
        *_, modes, m_nodal, k_nodal = self.base_modal(
            case=case,
            int_order=int_order,
            n_modes=n_modes,
            remove_complex_conjugate=remove_complex_conjugate,
        )

        # modes has shape (n_modes, n_free_dof) (rows are mode shapes)
        m_modal = modes @ m_nodal @ modes.T  # (n_modes, n_modes)
        k_modal = modes @ k_nodal @ modes.T  # (n_modes, n_modes)

        return m_modal, k_modal, modes

    def modal(
        self,
        case: StructureCase,
        remove_complex_conjugate: bool = True,
        int_order: Literal[3, 4, 5] = BASE_LOBATTO_ORDER,
        n_modes: int = 20,
        freq_range: tuple[float | Array, float | Array] = (0.0, jnp.inf),
        damp_range: tuple[float | Array, float | Array] = (-jnp.inf, jnp.inf),
        vtu_directory: str | os.PathLike = "./modal",
        n_plot_vtu: int | None = None,
        aero: UVLM | None = None,
        n_phase: int = 8,
        n_interp: int = 0,
        max_disp: float = 0.2,
        max_ang: float = 0.2,
    ) -> tuple[Array, Array, Array]:
        r"""
        Perform modal analysis on the structure.
        :param case: The static structure case for which to perform modal analysis.
        :param remove_complex_conjugate: If true, keep only one mode from each complex conjugate pair.
        :param int_order: Integration order for mass matrix computation.
        :param n_modes: Number of modes to preserve.
        :param freq_range: Frequency range for filtering out modes.
        :param damp_range: Damping range for filtering out modes.
        :param vtu_directory: Directory to for saving the mode shapes to vtu files.
        :param n_plot_vtu: Number of modes to plot to vtu files. Will default to "./modal".
        :param aero: UVLM aerodynamic model. If passed, the vtu files will include the aerodynamic grid. If not, they
        will just be the beam structure.
        :param n_phase: Number of phases to use when plotting the modes to vtu files.
        :param n_interp: Number of times to interpolate between beam nodes for vtu plotting.
        :param max_disp: Maximum displacement of structure for plotted modes, used for scaling.
        :param max_ang: Maximum angle of structure for plotted modes in radians, used for scaling.
        :return: Tuple of natural frequencies (n_free_dof), damping ratios (n_free_dof), and mode shapes with no
         normalisation ``(n_modes, n_free_dof)``.
        """
        freqs, damping, modes, *_ = self.base_modal(
            case=case,
            freq_range=freq_range,
            damp_range=damp_range,
            int_order=int_order,
            n_modes=n_modes,
            remove_complex_conjugate=remove_complex_conjugate,
        )

        if n_plot_vtu is not None:
            q_full = (
                jnp.zeros((n_plot_vtu, self.n_nodes * 6))
                .at[:, case.free_dofs]
                .set(modes[:n_plot_vtu, :])
            )

            for _i_mode in range(n_plot_vtu):
                plot_modes_vtu(
                    reference=case,
                    directory=vtu_directory,
                    q_full=q_full.reshape(n_plot_vtu, self.n_nodes, 6),
                    freqs=freqs,
                    dampings=damping,
                    gamma_b_full=None,
                    gamma_w_full=None,
                    zeta_w_full=None,
                    uvlm=aero,
                    n_interp=n_interp,
                    n_phase=n_phase,
                    max_disp=max_disp,
                    max_ang=max_ang,
                    max_gamma=1e6,
                )

        return freqs, damping, modes

    @staticmethod
    def _base_modal_symmetric(
        m_nodal: Array,
        k_nodal: Array,
        freq_range: tuple[float | Array, float | Array],
        damp_range: tuple[float | Array, float | Array],
        alpha_m: float | Array = 0.0,
        beta_k: float | Array = 0.0,
    ) -> tuple[Array, Array, Array]:
        # Cholesky-transformed symmetric eigenproblem: K phi = omega^2 M phi
        # ensure symmetry
        m_sym = 0.5 * (m_nodal + m_nodal.T)
        k_sym = 0.5 * (k_nodal + k_nodal.T)

        # spectral shift so rigid-body modes are well identified (alpha can be tuned)
        alpha = 1e-3
        sigma = alpha * jnp.trace(k_sym) / jnp.trace(m_sym)
        k_shift = k_sym + sigma * m_sym

        l_chol = jnp.linalg.cholesky(m_sym)
        x = jnp.linalg.solve(l_chol, k_shift)
        a_sym = jnp.linalg.solve(l_chol, x.T).T
        a_sym = 0.5 * (a_sym + a_sym.T)
        mu, y = jnp.linalg.eigh(a_sym)
        modes = jax.scipy.linalg.solve_triangular(l_chol.T, y, lower=False)

        omega_sq = mu - sigma
        omega = jnp.sqrt(jnp.maximum(omega_sq, 0.0))
        freq_hz = omega / (2.0 * jnp.pi)

        # damping can be obtained directly from the natural of the system and the Raleigh coefficients
        # includex filter to not damp the rigid-body modes
        safe_omega = jnp.where(omega > 0.0, omega, 1.0)
        damping = jnp.where(
            omega > 0.0,
            0.5 * (alpha_m / safe_omega + beta_k * omega),
            0.0,
        )

        # eigh returns ascending eigenvalues, so frequency ordering is implicit
        in_range = (
            (freq_hz >= freq_range[0])
            & (freq_hz <= freq_range[1])
            & (damping >= damp_range[0])
            & (damping <= damp_range[1])
        )
        idx = jnp.argsort(~in_range, stable=True)
        return freq_hz[idx], damping[idx], modes[:, idx]

    @staticmethod
    def _base_modal_nonsymmetric(
        m_nodal: Array,
        k_nodal: Array,
        freq_range: tuple[float | Array, float | Array],
        damp_range: tuple[float | Array, float | Array],
        remove_complex_conjugate: bool,
    ) -> tuple[Array, Array, Array]:
        # non-symmetric fallback for follower-load-deformed references
        m_inv_k = jnp.linalg.solve(m_nodal, k_nodal)
        omega_sq, modes = jnp.linalg.eig(m_inv_k)

        s = jnp.sqrt(-omega_sq)
        s = jnp.where(jnp.imag(s) < 0, -s, s)
        s_mag = jnp.abs(s)
        freq_hz = s_mag / (2.0 * jnp.pi)
        damping = -jnp.real(s) / s_mag

        # sort by frequency, remove conjugates and remove out-of-range
        idx = jnp.argsort(freq_hz)
        if remove_complex_conjugate:
            partner = conjugate_partner_mask(
                freq_hz=freq_hz, damping=damping, tiebreaker=jnp.real(s)
            )
            idx = idx[jnp.argsort(partner[idx], stable=True)]

        ordered_freq = freq_hz[idx]
        out_damping = damping[idx]
        in_range = (
            (ordered_freq >= freq_range[0])
            & (ordered_freq <= freq_range[1])
            & (out_damping >= damp_range[0])
            & (out_damping <= damp_range[1])
        )
        range_idx = jnp.argsort(~in_range, stable=True)
        idx = idx[range_idx]
        ordered_freq = ordered_freq[range_idx]
        out_damping = out_damping[range_idx]
        out_modes = modes[:, idx].real
        # per-mode mass normalisation: phi^T M phi = 1
        m_diag = jnp.einsum("im,ij,jm->m", out_modes, m_nodal, out_modes)
        out_modes /= jnp.sqrt(jnp.abs(m_diag))
        return ordered_freq, out_damping, out_modes

    def base_modal(
        self,
        case: StructureCase,
        remove_complex_conjugate: bool,
        int_order: Literal[3, 4, 5] = BASE_LOBATTO_ORDER,
        n_modes: int = 20,
        freq_range: tuple[float | Array, float | Array] = (0.0, jnp.inf),
        damp_range: tuple[float | Array, float | Array] = (-jnp.inf, jnp.inf),
    ) -> tuple[Array, Array, Array, Array, Array]:
        m_nodal, k_nodal = self.make_nodal_m_k(case=case, int_order=int_order)

        # dispatch based on stiffness symmetry and mass positive-definiteness;
        # external loads can cause a non-conservative system, and lumped masses
        # can produce a Jacobian mass matrix with small negative eigenvalues
        k_asymmetry = jnp.linalg.norm(k_nodal - k_nodal.T) / jnp.maximum(
            jnp.linalg.norm(k_nodal), 1.0
        )
        m_sym = 0.5 * (m_nodal + m_nodal.T)
        m_pd = jnp.linalg.eigvalsh(m_sym)[0] > 0.0
        ordered_freq, out_damping, out_modes = jax.lax.cond(
            (k_asymmetry < 1e-10) & m_pd,
            lambda: self._base_modal_symmetric(
                m_nodal=m_nodal,
                k_nodal=k_nodal,
                freq_range=freq_range,
                damp_range=damp_range,
                alpha_m=self.alpha_m,
                beta_k=self.beta_k,
            ),
            lambda: self._base_modal_nonsymmetric(
                m_nodal,
                k_nodal,
                freq_range,
                damp_range,
                remove_complex_conjugate,
            ),
        )

        # write to console
        print_table_line(inner_width=39)
        jax_print(
            "| Mode | Frequency [Hz] | Damping Ratio |",
            verbose_level="normal",
        )
        print_table_line(inner_width=39)
        for i_mode in range(n_modes):
            jax_print(
                "| {mode:>4d} | {freq:>14.3f} | {damp:>13.6f} |",
                mode=i_mode + 1,
                freq=ordered_freq[i_mode],
                damp=out_damping[i_mode],
                verbose_level="normal",
            )
        print_table_line(inner_width=39)
        return (
            ordered_freq[:n_modes],
            out_damping[:n_modes],
            out_modes[:, :n_modes].T,
            m_nodal,
            k_nodal,
        )

    def linearise(
        self,
        reference: StructureCase,
        dt: float,
        n_modes: int | None = None,
        modal_inputs: bool = False,
        modal_outputs: bool = False,
        prescribed_dofs: Sequence[int] | Array | slice | int | None = None,
    ) -> LinearBeam:
        r"""
        Linearise the beam about a given static structure case. This creates a LinearBeam object which can be used for
        linear dynamic analysis.
        :param reference: Static structure case about which to linearise the beam.
        :param dt: Time step size, used for conversions between continuous and discrete time.
        :param n_modes: If not None, the linearised system uses modal state coordinates truncated to this many modes.
        :param modal_inputs: If True, external forcing inputs are provided as modal forces (requires n_modes).
        :param modal_outputs: If True, outputs are exposed as modal coordinates (requires n_modes).
        :param prescribed_dofs: If provided, overrides the prescribed DOFs from the reference case.
        :return: Continuous-time linearised beam object.
        """
        return LinearBeam(
            beam=self,
            reference=reference,
            dt=dt,
            n_modes=n_modes,
            modal_inputs=modal_inputs,
            modal_outputs=modal_outputs,
            prescribed_dofs=prescribed_dofs,
        )

    def _make_c_t(
        self,
        d: Array,
        d_dot: Array,
        v: Array,
        int_order: Literal[1, 2, 3] = BASE_LEGENDRE_ORDER,
    ) -> tuple[Array, Array]:
        r"""
        Assemble tangent gyroscopic matrix. This does not include the lumped mass contribution.
        :param d: Element relative configuration, ``(n_elem, 6)``
        :param d_dot: Element relative velocity, ``(n_elem, 6)``
        :param v: Velocities in local frames, ``(n_node, 6)``
        :param int_order: Integration order,
        :return: Elementwise gyroscopic C_L and C_T matrices, ``(n_elem, 12, 12)``, ``(n_elem, 12, 12)``
        """
        clt = vmap(
            partial(
                _integrate_c_t,
                int_order=int_order,
                include_q_dot=self.optional_jacobians.d_f_gyr_d_q_dot,
            ),
            (0, 0, 0, 0, 0, 0),
            0,
        )(
            self.m_cs[self.m_cs_index, ...],
            jnp.concatenate(
                (
                    v[self.connectivity_arr[:, 0], :],
                    v[self.connectivity_arr[:, 1], :],
                ),
                axis=-1,
            ),
            d,
            d_dot,
            self.ad_inv_o0,
            self.l0,
        )
        cl = clt[:, 0, ...]
        ct = clt[:, 1, ...]
        return cl, ct

    def _make_c_t_lumped(self, v: Array) -> tuple[Array, Array]:
        r"""
        Obtain the gyroscopic matrix contribution from the lumped masses.
        :param v: Nodal velocities in global frame, ``(n_node, 6)``
        :return: Gyroscopic L and T matrix entries from lumped masses, ``(n_lumped, 6, 6)``, ``(n_lumped, 6, 6)``
        """
        c_t_l = vmap(_make_c_t_lumped, (0, 0), 0)(
            self.m_lumped, v[self.m_lumped_index, ...]
        )  # (n_lumped, 2, 6, 6)
        return c_t_l[:, 0, :, :], c_t_l[:, 1, :, :]

    def _make_sys_matrix(
        self,
        m_t: Array,
        c_t: Array,
        c_t_lumped: Array | None,
        k_t: Array,
        t_n: Array,
        ti: TimeIntegrator,
    ) -> Array:
        r"""
        Create the system matrix for the static or dynamic analysis.
        :param m_t: Disassembled system mass matrix, ``(n_elem, 12, 12)``.
        :param c_t: Disassembled system gyroscopic matrix, ``(n_elem, 12, 12)``.
        :param c_t_lumped: Disassembled system lumped gyroscopic matrix, ``(n_lumped, 6, 6)``.
        :param k_t: System stiffness matrix, ``(n_dof, n_dof)``.
        :param t_n: Tangent operator T(varphi), ``(n_nodes, 6, 6)``.
        :param ti: Time integration parameters.
        :return: System matrix, ``(n_dof, n_dof)``.
        """

        # note that k_t is already assembled for convenience
        k_t_tan_n = jnp.einsum(
            "ijk,jkl->ijl", k_t.reshape(self.n_dof, -1, 6), t_n
        ).reshape(self.n_dof, self.n_dof)  # (n_dof, n_dof)

        mat = (
            self.assemble_matrix_from_entries(
                m_t * ti.beta_prime + c_t * ti.gamma_prime
            )
        ) + k_t_tan_n

        if self.use_lumped_mass:
            if c_t_lumped is None:
                raise ValueError("c_t_lumped needs to be passed")
            mat = self.add_lumped_contributions_to_arr(
                arr=mat, lumped_arr=self.m_lumped * ti.beta_prime
            )
            mat = self.add_lumped_contributions_to_arr(
                arr=mat, lumped_arr=c_t_lumped * ti.gamma_prime
            )

        # add Rayleigh structural damping contribution
        if self.beta_k != 0.0:
            mat += self.beta_k * ti.gamma_prime * k_t
        if self.alpha_m != 0.0:
            mat += (
                self.alpha_m * ti.gamma_prime * self.assemble_matrix_from_entries(m_t)
            )
            if self.use_lumped_mass:
                mat = self.add_lumped_contributions_to_arr(
                    arr=mat,
                    lumped_arr=self.alpha_m * ti.gamma_prime * self.m_lumped,
                )

        return mat

    def apply_nodal_constraint_tangent(
        self,
        mat: Array,
        hg: Array,
        i_ts: int,
        gamma_prime: float | Array | None,
    ) -> Array:
        r"""
        Add nodal constraint contributions to a system matrix.
        :param mat: System matrix to update, ``(n_dof, n_dof)``.
        :param hg: SE(3) coordiantes, ``(n_nodes, 4, 4)``.
        :param i_ts: Time-step index (0 for static solves).
        :param gamma_prime: Time-integrator gamma_prime for damping scaling, or ``None`` to skip damping.
        :return: Updated system matrix.
        """
        for con in self.nodal_constraints:
            node = con.node_index
            hg_i = hg[node]
            dofs = node * 6 + jnp.arange(6)
            mat = mat.at[jnp.ix_(dofs, dofs)].add(con.k_tangent(hg_i, i_ts))
            if gamma_prime is not None:
                mat = mat.at[jnp.ix_(dofs, dofs)].add(
                    gamma_prime * con.c_tangent(hg_i, i_ts)
                )

        # hard constraint tangent contributions (e.g. hinge spring-damper)
        for con in self.multibody_constraints:
            if con.has_f_res:
                dofs_i = con.node_i * 6 + jnp.arange(6)
                dofs_j = con.node_j * 6 + jnp.arange(6)
                dofs_ij = jnp.concatenate([dofs_i, dofs_j])
                k_12 = con.k_tangent(hg[con.node_i], hg[con.node_j])
                mat = mat.at[jnp.ix_(dofs_ij, dofs_ij)].add(k_12)
                if gamma_prime is not None:
                    # add damping terms
                    mat = mat.at[jnp.ix_(dofs_ij, dofs_ij)].add(
                        gamma_prime * con.c_tangent(hg[con.node_i], hg[con.node_j])
                    )

        return mat

    @property
    def n_multibody_constraints(self) -> int:
        r"""Total number of scalar Lagrange-multiplier constraints."""
        return sum(con.n_constraints for con in self.multibody_constraints)

    @property
    def n_holonomic_constraints(self) -> int:
        r"""Number of scalar holonomic (position-level) Lagrange-multiplier constraints."""
        return sum(
            con.n_constraints for con in self.multibody_constraints if con.is_holonomic
        )

    @property
    def n_nonholonomic_constraints(self) -> int:
        r"""Number of scalar non-holonomic (velocity-level) Lagrange-multiplier constraints."""
        return sum(
            con.n_constraints
            for con in self.multibody_constraints
            if not con.is_holonomic
        )

    def postprocess_constraints(self, hg: Array) -> dict[str, dict[str, Array]]:
        r"""
        Postprocess all constraints to extract derived quantities (e.g. hinge angles).
        :param hg: Nodal SE(3) frames, ``(n_nodes, 4, 4)`` or ``(n_tstep, n_nodes, 4, 4)``.
        :return: Nested dict ``{constraint_name: {quantity_name: Array}}``.
        """
        data: dict[str, dict[str, Array]] = {}
        for name, con in self.constraints.items():
            pp = con.postprocess(hg)
            if pp:
                data[name] = pp
        return data

    def _compute_holonomic_violation(self, hg: Array) -> Array:
        r"""
        Compute the stacked position-level constraint violation for holonomic constraints.
        :param hg: Current nodal SE(3) frames, ``(n_nodes, 4, 4)``.
        :return: Constraint violation, ``(n_holonomic_constraints,)``.
        """
        return jnp.concatenate(
            [
                con.violation(
                    hg[con.node_i], con.hg_ref if con.is_grounded else hg[con.node_j]
                )
                for con in self.multibody_constraints
                if con.is_holonomic
            ]
        )

    def _compute_holonomic_jacobian(
        self,
        hg: Array,
        solve_dofs: Array,
        phi: Array | None = None,
    ) -> Array:
        r"""
        Assemble the position-level constraint Jacobian for holonomic constraints.
        :param hg: Base SE(3) frames, ``(n_nodes, 4, 4)``.
        :param solve_dofs: Free DOF indices, ``(n_solve,)``.
        :param phi: Accumulated configuration increment, ``(n_nodes, 6)`` or ``None``.
        :return: Constraint Jacobian at solve DOFs, ``(n_holonomic_constraints, n_solve)``.
        """
        n_c = self.n_holonomic_constraints
        jac_full = jnp.zeros((n_c, self.n_dof))
        offset = 0
        for con in self.multibody_constraints:
            if not con.is_holonomic:
                continue
            phi_i = phi[con.node_i] if phi is not None else None
            hg_j = con.hg_ref if con.is_grounded else hg[con.node_j]
            phi_j = (
                None
                if con.is_grounded
                else (phi[con.node_j] if phi is not None else None)
            )
            jac_i, jac_j = con.jacobian_local(hg[con.node_i], hg_j, phi_i, phi_j)
            nc = con.n_constraints
            dofs_i = con.node_i * 6 + jnp.arange(6)
            jac_full = jac_full.at[offset : offset + nc, dofs_i].set(jac_i)
            if not con.is_grounded:
                dofs_j = con.node_j * 6 + jnp.arange(6)
                jac_full = jac_full.at[offset : offset + nc, dofs_j].set(jac_j)
            offset += nc
        return jac_full[:, solve_dofs]

    def _compute_nonholonomic_vel_violation(self, hg: Array, v: Array) -> Array:
        r"""
        Compute the stacked velocity-level constraint violation for non-holonomic constraints.
        :param hg: Current nodal SE(3) frames, ``(n_nodes, 4, 4)``.
        :param v: Current nodal velocities, ``(n_nodes, 6)``.
        :return: Velocity constraint violation, ``(n_nonholonomic_constraints,)``.
        """
        return jnp.concatenate(
            [
                con.vel_violation(
                    hg[con.node_i],
                    con.hg_ref if con.is_grounded else hg[con.node_j],
                    v[con.node_i],
                    jnp.zeros(6) if con.is_grounded else v[con.node_j],
                )
                for con in self.multibody_constraints
                if not con.is_holonomic
            ]
        )

    def _compute_nonholonomic_a_vel(
        self, hg: Array, v: Array, solve_dofs: Array
    ) -> Array:
        r"""
        Assemble velocity Jacobian :math:`\partial g_{vel}/\partial v` for
        non-holonomic constraints. Used for the constraint force direction.
        :param hg: Current SE(3) frames, ``(n_nodes, 4, 4)``.
        :param v: Current velocities, ``(n_nodes, 6)``.
        :param solve_dofs: Free DOF indices, ``(n_solve,)``.
        :return: Velocity Jacobian at solve DOFs, ``(n_nonholonomic_constraints, n_solve)``.
        """
        n_c = self.n_nonholonomic_constraints
        a_full = jnp.zeros((n_c, self.n_dof))
        offset = 0
        for con in self.multibody_constraints:
            if con.is_holonomic:
                continue
            hg_j = con.hg_ref if con.is_grounded else hg[con.node_j]
            v_j = jnp.zeros(6) if con.is_grounded else v[con.node_j]
            av_i, av_j = con.a_vel_local(hg[con.node_i], hg_j, v[con.node_i], v_j)
            nc = con.n_constraints
            dofs_i = con.node_i * 6 + jnp.arange(6)
            a_full = a_full.at[offset : offset + nc, dofs_i].set(av_i)
            if not con.is_grounded:
                dofs_j = con.node_j * 6 + jnp.arange(6)
                a_full = a_full.at[offset : offset + nc, dofs_j].set(av_j)
            offset += nc
        return a_full[:, solve_dofs]

    def _compute_nonholonomic_a_phi(
        self,
        hg: Array,
        v: Array,
        solve_dofs: Array,
        phi: Array | None = None,
    ) -> Array:
        r"""
        Assemble configuration Jacobian :math:`\partial g_{vel}/\partial \varphi`
        for non-holonomic constraints.
        :param hg: Base SE(3) frames, ``(n_nodes, 4, 4)``.
        :param v: Current velocities, ``(n_nodes, 6)``.
        :param solve_dofs: Free DOF indices, ``(n_solve,)``.
        :param phi: Accumulated configuration increment, ``(n_nodes, 6)`` or ``None``.
        :return: Configuration Jacobian at solve DOFs, ``(n_nonholonomic_constraints, n_solve)``.
        """
        n_c = self.n_nonholonomic_constraints
        a_full = jnp.zeros((n_c, self.n_dof))
        offset = 0
        for con in self.multibody_constraints:
            if con.is_holonomic:
                continue
            phi_i = phi[con.node_i] if phi is not None else None
            hg_j = con.hg_ref if con.is_grounded else hg[con.node_j]
            v_j = jnp.zeros(6) if con.is_grounded else v[con.node_j]
            phi_j = (
                None
                if con.is_grounded
                else (phi[con.node_j] if phi is not None else None)
            )
            ap_i, ap_j = con.a_phi_local(
                hg[con.node_i],
                hg_j,
                v[con.node_i],
                v_j,
                phi_i,
                phi_j,
            )
            nc = con.n_constraints
            dofs_i = con.node_i * 6 + jnp.arange(6)
            a_full = a_full.at[offset : offset + nc, dofs_i].set(ap_i)
            if not con.is_grounded:
                dofs_j = con.node_j * 6 + jnp.arange(6)
                a_full = a_full.at[offset : offset + nc, dofs_j].set(ap_j)
            offset += nc
        return a_full[:, solve_dofs]

    def solve_constrained(
        self,
        sys_mat_solve: Array,
        f_res_solve: Array,
        hg_eval: Array,
        solve_dofs: Array,
        hg_base: Array | None = None,
        phi: Array | None = None,
        v: Array | None = None,
        gamma_prime: float | Array | None = None,
    ) -> tuple[Array, Array, Array]:
        r"""
        Solve the augmented system with Lagrange multipliers, supporting both holonomic and non-holonomic constraints.
        :param sys_mat_solve: System matrix at solve DOFs, ``(n_solve, n_solve)``.
        :param f_res_solve: Force residual at solve DOFs, ``(n_solve,)``.
        :param hg_eval: SE(3) frames for constraint evaluation, ``(n_nodes, 4, 4)``.
        :param solve_dofs: Free DOF indices, ``(n_solve,)``.
        :param hg_base: Base frames for Jacobian computation (defaults to ``hg_eval``).
        :param phi: Accumulated configuration increment, ``(n_nodes, 6)``.
        :param v: Current nodal velocities for non-holonomic constraints, ``(n_nodes, 6)``.
        :param gamma_prime: Newmark parameter for non-holonomic constraints.
        :return: ``(delta_phi, lagrange_multipliers, constraint_violation)``.
        """
        hg_jac = hg_base if hg_base is not None else hg_eval

        n_s = sys_mat_solve.shape[0]
        n_h = self.n_holonomic_constraints
        n_nh = self.n_nonholonomic_constraints
        n_c = n_h + n_nh

        aug = jnp.zeros((n_s + n_c, n_s + n_c))
        aug = aug.at[:n_s, :n_s].set(sys_mat_solve)

        rhs_parts: list[Array] = [f_res_solve]
        violation_parts: list[Array] = []

        if n_h > 0:
            viol_h = self._compute_holonomic_violation(hg_eval)
            jac_h = self._compute_holonomic_jacobian(hg_jac, solve_dofs, phi)
            aug = aug.at[:n_s, n_s : n_s + n_h].set(-jac_h.T)
            aug = aug.at[n_s : n_s + n_h, :n_s].set(jac_h)
            rhs_parts.append(-viol_h)
            violation_parts.append(viol_h)

        if n_nh > 0:
            assert v is not None
            vel_viol_nh = self._compute_nonholonomic_vel_violation(hg_eval, v)
            a_vel_solve = self._compute_nonholonomic_a_vel(hg_eval, v, solve_dofs)
            a_phi_solve = self._compute_nonholonomic_a_phi(hg_jac, v, solve_dofs, phi)
            aug = aug.at[:n_s, n_s + n_h : n_s + n_c].set(-a_vel_solve.T)
            aug = aug.at[n_s + n_h : n_s + n_c, :n_s].set(
                a_phi_solve + gamma_prime * a_vel_solve
            )
            rhs_parts.append(-vel_viol_nh)
            violation_parts.append(vel_viol_nh)

        rhs = jnp.concatenate(rhs_parts)
        sol = jnp.linalg.solve(aug, rhs)

        return sol[:n_s], sol[n_s:], jnp.concatenate(violation_parts)

    def compute_centre_of_mass(self, hg: Array) -> Array:
        r"""
        Compute the centre of mass for an arbitrary system.
        :param hg: Node SE(3) coordinates, ``(n_node, 4, 4)`` or ``(n_tstep, n_node, 4, 4)``.
        :return: Centre of mass, (3) or ``(n_tstep, 3)``.
        """

        def inner_func(hg_: Array) -> Array:
            d = self.make_d(hg=hg_)
            m = self.assemble_matrix_from_entries(self.make_m_t(d=d))  # (n_dof, n_dof)
            if self.use_lumped_mass:
                m = self.add_lumped_contributions_to_arr(
                    arr=m, lumped_arr=self.m_lumped
                )
            m_lin = m[::6, ::6]
            return jnp.einsum("ij,jk->k", m_lin, hg_[:, :3, 3]) / m_lin.sum()  # (3, )

        if hg.ndim == 3:
            return inner_func(hg)  # single timestep, (3, ).
        elif hg.ndim == 4:
            return vmap(inner_func, 0, 0)(hg)  # multiple timesteps, (n_tstep, 3)
        else:
            raise ValueError("hg.ndim must be 3 or 4")

    def make_f_elem(self, eps: Array) -> Array:
        r"""
        Compute the forces within the elements as :math:`\mathbf{f}_{elem} = \mathcal{K}_{cs} \epsilon`.
        :param eps: Element strain vectors, ``(n_elem, 6)``.
        :return: Element forces, ``(n_elem, 6)``.
        """
        return jnp.einsum("ijk,ik->ij", self.k_cs[self.k_cs_index, ...], eps)

    def make_f_int(self, p_d: Array, eps: Array) -> Array:
        r"""
        Assemble global internal force vector as a function of the element relative configuration vectors.
        :param p_d: P(d) operator, ``(n_elem, 6, 12)``.
        :param eps: Element strain vectors, ``(n_elem, 6)``.
        :return: Internal forces, ``(n_elem, 12)``.
        """

        return -jnp.einsum("ikj,ikl,il->ij", p_d, self.k_cs[self.k_cs_index, ...], eps)

    def _make_f_grav(self, m_t: Array, rmat: Array) -> Array:
        r"""
        Compute the global gravitational force vector. This does not include the lumped mass contribution.
        :param m_t: Disassembled system mass matrix, ``(n_elem, 12, 12)``.
        :param rmat: Node rotations from reference, ``(n_node, 3, 3)``.
        :return: Gravity force element vector, ``(n_elem, 12)``.
        """
        f_rot = jnp.einsum(
            "ikj,k->ij", rmat, jnp.array(self.gravity_vec)
        )  # (n_node, 3)
        f_rot_tot = jnp.concatenate(
            (f_rot, jnp.zeros((self.n_nodes, 3))), axis=-1
        )  # (n_node, 6)
        return jnp.einsum(
            "ijk,ik->ij",
            m_t,
            jnp.concatenate(
                (
                    f_rot_tot[self.connectivity_arr[:, 0], :],
                    f_rot_tot[self.connectivity_arr[:, 1], :],
                ),
                axis=1,
            ),
        )  # (n_elem, 12)

    def _make_f_grav_lumped(self, rmat: Array) -> Array:
        r"""
        Compute the global gravitational force vector contribution from the lumped masses.
        :param rmat: Rotation matrices at nodes, ``(n_node, 3, 3)``
        :return: Lumped gravity force vector, ``(n_lumped, 6)``
        """
        f_rot = jnp.einsum(
            "ikj,k->ij", rmat[self.m_lumped_index_arr, ...], jnp.array(self.gravity_vec)
        )  # (n_lumped, 3)
        f_rot_tot = jnp.concatenate(
            (f_rot, jnp.zeros_like(f_rot)), axis=-1
        )  # (n_lumped, 6)
        return jnp.einsum("ijk,ik->ij", self.m_lumped, f_rot_tot)  # (n_lumped, 6)

    @staticmethod
    def make_f_dead_ext(f_ext: Array, rmat: Array) -> Array:
        r"""
        Compute the global external dead force vector.
        :param f_ext: External forces array of dead forces in global reference, ``(n_node, 6)``
        :param rmat: Deformation rotation matrices, ``(n_node, 3, 3)``
        :return: External forces, ``(n_node, 6)``
        """

        return transform_nodal_vect(f_ext, jnp.swapaxes(rmat, -1, -2))

    def split_vector_to_elements(self, vec: Array) -> Array:
        return jnp.concatenate(
            (
                vec[self.connectivity_arr[:, 0], :],
                vec[self.connectivity_arr[:, 1], :],
            ),
            axis=-1,
        )

    def _make_f_iner_gyr(
        self, m_l: Array, c_l: Array, v: Array, v_dot: Array
    ) -> tuple[Array, Array]:
        r"""
        Compute the global inertial force vector.
        :param m_l: Disassembled system mass matrix, ``(n_elem, 12, 12)``
        :param c_l: Disassembled system gyroscopic matrix, ``(n_elem, 12, 12)``
        :param v: Nodal velocities in local frame, ``(n_node, 6)``
        :param v_dot: Nodal accelerations in local frame, ``(n_node, 6)``
        :return: Inertial forces, ``(n_elem, 12)``
        """

        v_elem = self.split_vector_to_elements(v)
        v_dot_elem = jnp.concatenate(
            (
                v_dot[self.connectivity_arr[:, 0], :],
                v_dot[self.connectivity_arr[:, 1], :],
            ),
            axis=-1,
        )  # (n_elem, 12)

        return -jnp.einsum("ijk,ik->ij", m_l, v_dot_elem), -jnp.einsum(
            "ijk,ik->ij", c_l, v_elem
        )  # (n_elem, 12)

    def _make_f_rayleigh_damp(
        self, m_t: Array, k_t_assembled: Array | None, v: Array
    ) -> Array:
        r"""
        Compute the Rayleigh structural damping force, returned as a global DOF vector.
        :param m_t: Element mass matrices, ``(n_elem, 12, 12)``.
        :param k_t_assembled: Assembled global tangent stiffness matrix, ``(n_dof, n_dof)``.
        :param v: Nodal velocities in local frames, ``(n_nodes, 6)``.
        :return: Damping force vector, ``(n_dof, )``.
        """
        f_damp = jnp.zeros(self.n_dof)

        # stiffness proportional damping contribution
        if self.beta_k != 0.0:
            assert k_t_assembled is not None
            f_damp -= self.beta_k * (k_t_assembled @ v.ravel())

        # mass proportional damping contribution
        if self.alpha_m != 0.0:
            v_elem = self.split_vector_to_elements(v)  # (n_elem, 12)
            f_mass_elem = -self.alpha_m * jnp.einsum(
                "ijk,ik->ij", m_t, v_elem
            )  # (n_elem, 12)
            f_damp += self.assemble_vector_from_entries(f_mass_elem)
            if self.use_lumped_mass:
                f_mass_lumped = (
                    -self.alpha_m
                    * jnp.einsum(
                        "ijk,ik->ij", self.m_lumped, v[self.m_lumped_index_arr, ...]
                    ).ravel()
                )
                f_damp = self.add_lumped_contributions_to_vec(
                    vec=f_damp, lumped_vec=f_mass_lumped
                )

        return f_damp

    def _make_f_iner_gyr_lumped(
        self, c_l_lumped: Array, v: Array, v_dot: Array
    ) -> tuple[Array, Array]:
        r"""
        Obtain the contribution to the inertial forces from the lumped masses.
        :param c_l_lumped: Gyroscopic matrix from lumped masses, ``(n_lumped, 6, 6)``
        :param v: Nodal velocities in local frame, ``(n_node, 6)``
        :param v_dot: Nodal accelerations in local frame, ``(n_node, 6)``
        :return: Inertial forces from lumped masses, ``(n_lumped, 6)``
        """
        f_iner = -jnp.einsum(
            "ijk,ik->ij", self.m_lumped, v_dot[self.m_lumped_index, ...]
        )  # (n_lumped, 6)
        f_gyr = -jnp.einsum(
            "ijk,ik->ij", c_l_lumped, v[self.m_lumped_index, ...]
        )  # (n_lumped, 6)
        return f_iner, f_gyr

    def add_thrust_force(self, force: Array, thrust: dict[str, Array]) -> Array:
        r"""
        Add thrust acting at nodes onto full system forcing.
        :param force: Input forcing, ``(n_node, 6)``.
        :param thrust: Input thrust at the current step, ``{key: ()}``.
        :return: Updated forcing, ``(n_node, 6)``.
        """

        for k, v in thrust.items():
            node = dict(self.thrust_nodes)[k]
            direction = jnp.array(dict(self.thrust_direction)[k])
            force = force.at[node, :3].add(v * direction)
        return force

    def make_eps(self, d: Array) -> Array:
        r"""
        Compute the element strain vectors as a function of the element relative configuration vectors. Formulation from
        Geometrically exact beam finite element formulated on the special Euclidean group SE(3), by Sonneville et al.,
        2013, Eq 64.
        :param d: Element relative configuration, ``(n_elem, 6)``
        :return: Element strain vectors, ``(n_elem, 6)``
        """

        return (d - self.d0) / self.l0[:, None]

    def make_p_d(self, d: Array) -> Array:
        r"""
        Compute the P(d) operator as a function of the element relative configuration vectors.
        :param d: Relative configuration vectors, ``(n_elem, 6)``
        :return: P(d) operator, ``(n_elem, 6, 12)``
        """
        return vmap(p, (0, 0), 0)(d, self.ad_inv_o0)  # [n_elem, 6, 12]

    def make_d(self, hg: Array) -> Array:
        r"""
        Compute the element relative configuration vectors from the nodal homogeneous transformation matrices
        :param hg: Nodal homogeneous transformation matrices, ``(n_nodes, 4, 4)``
        :return: Element relative configuration vectors, ``(n_elem, 6)``
        """

        base_hg = jnp.zeros((self.n_elem, 4, 4))
        base_hg = base_hg.at[:, :3, :3].set(self.o0)
        base_hg = base_hg.at[:, 3, 3].set(1.0)

        haha0 = jnp.einsum(
            "ijk,ikl->ijl", hg[self.connectivity_arr[:, 0], :, :], base_hg
        )  # (n_elem, 4, 4)
        haha1 = jnp.einsum(
            "ijk,ikl->ijl", hg[self.connectivity_arr[:, 1], :, :], base_hg
        )  # (n_elem, 4, 4)

        return vmap(hg_to_d, (0, 0), 0)(haha0, haha1)  # (n_elem, 6)

    def _make_d_dot(self, p_d: Array, v: Array) -> Array:
        r"""
        Compute the time derivative of the element relative configuration vectors from the nodal velocities.
        :param p_d: P(d) operator, ``(n_elem, 6, 12)``
        :param v: Nodal velocities in local frame, ``(n_node, 6)``
        :return: Element relative velocity vectors, ``(n_elem, 6)``
        """

        v_elem = jnp.concatenate(
            (
                v[self.connectivity_arr[:, 0], :],
                v[self.connectivity_arr[:, 1], :],
            ),
            axis=-1,
        )  # (n_elem, 12)

        return jnp.einsum("ijk,ik->ij", p_d, v_elem)  # (n_elem, 6)

    @staticmethod
    def make_hg_dot(hg: Array, v: Array) -> Array:
        r"""
        Obtain the time derivative of the nodal coordinates.
        :param hg: Node coordinates, ``(n_node, 4, 4)``.
        :param v: Node local velocities, ``(n_node, 6)``
        :return: Coordinate time derivative, ``(n_node, 4, 4)``
        """
        return jnp.einsum(
            "ijk,ikl->ijl", hg, vmap(ha_to_ha_tilde, 0, 0)(v)
        )  # (n_nodes, 4, 4)

    @overload
    def resolve_forces(
        self,
        hg: Array,
        dynamic: Literal[True],
        f_ext_follower: Array | None,
        f_ext_dead: Array | None,
        f_ext_aero: Array | None,
        thrust: dict[str, Array],
        v: Array,
        v_dot: Array,
        approx_gradients: bool = False,
    ) -> tuple[
        Array,
        Array,
        Array | None,
        Array | None,
        Array | None,
        Array,
        Array,
        Array,
        Array,
    ]: ...

    @overload
    def resolve_forces(
        self,
        hg: Array,
        dynamic: Literal[False],
        f_ext_follower: Array | None,
        f_ext_dead: Array | None,
        f_ext_aero: Array | None,
        thrust: dict[str, Array],
        v: None,
        v_dot: None,
        approx_gradients: bool = False,
    ) -> tuple[
        Array,
        Array,
        Array | None,
        Array | None,
        Array | None,
        Array,
        None,
        None,
        Array,
    ]: ...

    def resolve_forces(
        self,
        hg: Array,
        dynamic: bool,
        f_ext_follower: Array | None,
        f_ext_dead: Array | None,
        f_ext_aero: Array | None,
        thrust: dict[str, Array],
        v: Array | None,
        v_dot: Array | None,
        approx_gradients: bool = False,
    ) -> tuple[
        Array,
        Array,
        Array | None,
        Array | None,
        Array | None,
        Array,
        Array | None,
        Array | None,
        Array,
    ]:
        r"""
        Obtain all components of the force from a final solution.
        :param hg: Nodal homogeneous transformation matrices, ``(n_nodes, 4, 4)``.
        :param dynamic: Whether to compute dynamic forces.
        :param f_ext_follower: External follower forces in local reference, ``(n_node, 6)``.
        :param f_ext_dead: External dead forces in global reference, ``(n_node, 6)``.
        :param f_ext_aero: External aero forces in global reference, ``(n_node, 6)``.
        :param thrust: Thrust forces at current step, {keys, ``()``}.
        :param v: Nodal velocities in global frame, ``(n_node, 6)``.
        :param v_dot: Nodal accelerations in global frame, ``(n_node, 6)``.
        :param approx_gradients: Whether to stop computing gradients of the inertial and gyroscopic forces with respect to
        the node coordinates, as these are small but nonzero values in practice.
        :return: Configuration vectors, strain vectors, Dead external forces, aero external forces, gravitational forces, internal forces,
        gyroscopic forces, inertial forces and residual forces.
        """

        def prop_grad(x: Array) -> Array:
            return jax.lax.stop_gradient(x) if approx_gradients else x

        d = self.make_d(hg)
        eps = self.make_eps(d)
        p_d = self.make_p_d(d)

        if dynamic or self.use_gravity:
            m_t = self.make_m_t(prop_grad(d))
        else:
            m_t = None

        if dynamic:
            assert v is not None

            d_dot = self._make_d_dot(p_d, v)
            c_l = self._make_c_t(prop_grad(d), prop_grad(d_dot), v)[0]
            c_l_lumped = self._make_c_t_lumped(v)[0] if self.use_lumped_mass else None
        else:
            d_dot, c_l, c_l_lumped = None, None, None

        this_f_res = self.add_thrust_force(
            force=jnp.zeros((self.n_nodes, 6)), thrust=thrust
        )

        if f_ext_dead is not None:
            this_f_ext_dead = self.make_f_dead_ext(f_ext_dead, hg[:, :3, :3])
            this_f_res += this_f_ext_dead
        else:
            this_f_ext_dead = None

        if f_ext_aero is not None:
            this_f_ext_aero = self.make_f_dead_ext(f_ext_aero, hg[:, :3, :3])
            this_f_res += this_f_ext_aero
        else:
            this_f_ext_aero = None

        if self.use_gravity:
            assert m_t is not None
            this_f_grav = self.assemble_vector_from_entries(
                self._make_f_grav(m_t, hg[:, :3, :3])
            ).reshape(-1, 6)
            if self.use_lumped_mass:
                f_grav_lumped = self._make_f_grav_lumped(hg[:, :3, :3])
                this_f_grav = self.add_lumped_contributions_to_vec(
                    vec=this_f_grav.ravel(), lumped_vec=f_grav_lumped.ravel()
                ).reshape(-1, 6)
            this_f_res += this_f_grav
        else:
            this_f_grav = None

        this_f_int = self.assemble_vector_from_entries(
            self.make_f_int(p_d, eps)
        ).reshape(-1, 6)
        this_f_res += this_f_int

        if dynamic:
            assert (
                m_t is not None
                and c_l is not None
                and v is not None
                and v_dot is not None
            )
            this_f_iner, this_f_gyr = self._make_f_iner_gyr(m_t, c_l, v, v_dot)
            this_f_iner = self.assemble_vector_from_entries(this_f_iner).reshape(-1, 6)
            this_f_gyr = self.assemble_vector_from_entries(this_f_gyr).reshape(-1, 6)

            if self.use_lumped_mass:
                assert c_l_lumped is not None
                f_iner_lumped, f_gyr_lumped = self._make_f_iner_gyr_lumped(
                    c_l_lumped, v, v_dot
                )
                this_f_iner = self.add_lumped_contributions_to_vec(
                    this_f_iner.ravel(), (f_iner_lumped + f_gyr_lumped).ravel()
                ).reshape(-1, 6)
            this_f_res += this_f_iner
        else:
            this_f_iner = None
            this_f_gyr = None

        if f_ext_follower is not None:
            this_f_res += f_ext_follower

        return (
            d,
            eps,
            this_f_ext_dead,
            this_f_ext_aero,
            this_f_grav,
            this_f_int,
            this_f_gyr,
            this_f_iner,
            this_f_res,
        )

    @overload
    def make_f_res(
        self,
        solve_dofs: Array | None,
        p_d: Array,
        eps: Array,
        hg: Array,
        f_ext_follower_n: Array | None,
        f_ext_dead_n: Array | None,
        thrust_n: dict[str, Array],
        dynamic: Literal[True],
        m_t: Array,
        c_l: Array,
        c_l_lumped: Array | None,
        v: Array,
        v_dot: Array,
        i_ts: int = 0,
        k_t_assembled: Array | None = None,
    ) -> tuple[Array, Array]: ...

    @overload
    def make_f_res(
        self,
        solve_dofs: Array | None,
        p_d: Array,
        eps: Array,
        hg: Array,
        f_ext_follower_n: Array | None,
        f_ext_dead_n: Array | None,
        thrust_n: dict[str, Array],
        dynamic: Literal[False],
        m_t: Array | None,
        c_l: None,
        c_l_lumped: None,
        v: None,
        v_dot: None,
        i_ts: int = 0,
        k_t_assembled: Array | None = None,
    ) -> tuple[Array, Array]: ...

    def make_f_res(
        self,
        solve_dofs: Array | None,
        p_d: Array,
        eps: Array,
        hg: Array,
        f_ext_follower_n: Array | None,
        f_ext_dead_n: Array | None,
        thrust_n: dict[str, Array],
        dynamic: bool,
        m_t,
        c_l,
        c_l_lumped,
        v,
        v_dot,
        i_ts: int = 0,
        k_t_assembled: Array | None = None,
    ) -> tuple[Array, Array]:
        r"""
        Compute the residual force vector for a given configuration and external forces, used in the nonlinear solve.
        This is the force imbalance that the nonlinear solver will seek to drive to zero. Additionally, returns an
        "absolute sum" of all forces, used for relative convergence checks.
        :param solve_dofs: Optional array of degrees of freedom to solve for ``(n_solve_dofs, )``.
        :param p_d: P(d) operator, ``(n_elem, 6, 12)``.
        :param eps: Element strain vectors, ``(n_elem, 6)``.
        :param hg: Nodal homogeneous transformation matrices, ``(n_nodes, 4, 4)``.
        :param f_ext_follower_n: Nodal follower forces, ``(n_nodes, 6)``.
        :param f_ext_dead_n: Nodal dead forces, ``(n_nodes, 6)``.
        :param thrust_n: Thrust magnitude, ``{key: ()}``.
        :param dynamic: Flag for whether to compute dynamic entries.
        :param m_t: Disassembled system mass matrix, ``(n_elem, 12, 12)``.
        :param c_l: Dissembled system gyroscopic matrix, ``(n_elem, 12, 12)``.
        :param c_l_lumped: Lumped gyroscopic matrix, ``(n_nodes, 6, 6)``.
        :param v: Nodal velocities, ``(n_nodes, 6)``.
        :param v_dot: Nodal accelerations, ``(n_node, 6)``.
        :param i_ts: Time-step index (0 for static solves).
        :param k_t_assembled: Assembled global tangent stiffness matrix ``(n_dof, n_dof)``, required for Rayleigh
        damping.
        :return: Residual force vector, ``(n_dof, )``, absolute sum of forces, ``(n_dof, )``.
        """

        f_res = self.make_f_int(p_d, eps)  # (n_elem, 12)
        f_abs_sum = jnp.abs(f_res)

        if self.use_gravity:
            f_grav = self._make_f_grav(m_t, hg[:, :3, :3])
            f_res += f_grav
            f_abs_sum += jnp.abs(f_grav)

        if dynamic:
            f_iner, f_gyr = self._make_f_iner_gyr(m_t, c_l, v, v_dot)
            f_res += f_iner + f_gyr
            f_abs_sum += jnp.abs(f_iner + f_gyr)

        f_res_vect = self.assemble_vector_from_entries(f_res)
        f_abs_sum_vect = self.assemble_vector_from_entries(f_abs_sum)

        # add external forcing contributions
        if f_ext_follower_n is not None:
            f_res_vect += f_ext_follower_n.reshape(self.n_dof).ravel()
            f_abs_sum_vect += jnp.abs(f_ext_follower_n.reshape(self.n_dof).ravel())
        if f_ext_dead_n is not None:
            f_dead = self.make_f_dead_ext(f_ext_dead_n, hg[:, :3, :3]).ravel()
            f_res_vect += f_dead
            f_abs_sum_vect += jnp.abs(f_dead)

        f_thrust = self.add_thrust_force(
            force=jnp.zeros((self.n_nodes, 6)), thrust=thrust_n
        ).ravel()
        f_res_vect += f_thrust
        f_abs_sum_vect += jnp.abs(f_thrust)

        if self.use_lumped_mass:
            if dynamic:
                f_iner_lumped, f_gyr_lumped = self._make_f_iner_gyr_lumped(
                    c_l_lumped, v, v_dot
                )
                f_iner_gyr_lumped = (f_iner_lumped + f_gyr_lumped).ravel()
                f_res_vect = self.add_lumped_contributions_to_vec(
                    f_res_vect, f_iner_gyr_lumped
                )
                f_abs_sum_vect = self.add_lumped_contributions_to_vec(
                    f_abs_sum_vect, jnp.abs(f_iner_gyr_lumped)
                )
            if self.use_gravity:
                f_grav_lumped = self._make_f_grav_lumped(hg[:, :3, :3]).ravel()
                f_res_vect = self.add_lumped_contributions_to_vec(
                    vec=f_res_vect, lumped_vec=f_grav_lumped
                )
                f_abs_sum_vect = self.add_lumped_contributions_to_vec(
                    vec=f_abs_sum_vect, lumped_vec=f_grav_lumped
                )

        # nodal constraint contributions
        for con in self.nodal_constraints:
            node = con.node_index
            v_node = v[node] if dynamic else jnp.zeros(6)
            f_constraint = con.f_res(hg[node], v_node, i_ts)
            dofs = node * 6 + jnp.arange(6)
            f_res_vect = f_res_vect.at[dofs].add(f_constraint)
            f_abs_sum_vect = f_abs_sum_vect.at[dofs].add(jnp.abs(f_constraint))

        # hard constraint force contributions (e.g. hinge spring-damper)
        for con in self.multibody_constraints:
            if con.has_f_res:
                v_i = v[con.node_i] if dynamic else jnp.zeros(6)
                v_j = v[con.node_j] if dynamic else jnp.zeros(6)
                f_i, f_j = con.f_res(hg[con.node_i], hg[con.node_j], v_i, v_j)
                dofs_i = con.node_i * 6 + jnp.arange(6)
                dofs_j = con.node_j * 6 + jnp.arange(6)
                f_res_vect = f_res_vect.at[dofs_i].add(f_i)
                f_res_vect = f_res_vect.at[dofs_j].add(f_j)
                f_abs_sum_vect = f_abs_sum_vect.at[dofs_i].add(jnp.abs(f_i))
                f_abs_sum_vect = f_abs_sum_vect.at[dofs_j].add(jnp.abs(f_j))

        # Rayleigh structural damping
        if dynamic and (self.alpha_m != 0.0 or self.beta_k != 0.0):
            if self.beta_k != 0.0 and k_t_assembled is None:
                raise ValueError(
                    "k_t_assembled must be provided when beta_k != 0 for dynamic residual."
                )
            f_damp = self._make_f_rayleigh_damp(
                m_t=m_t,
                k_t_assembled=k_t_assembled
                if k_t_assembled is not None
                else jnp.zeros((self.n_dof, self.n_dof)),
                v=v,
            )
            f_res_vect += f_damp
            f_abs_sum_vect += jnp.abs(f_damp)

        if solve_dofs is not None:
            return f_res_vect[solve_dofs], f_abs_sum_vect[
                solve_dofs
            ]  # (n_solve_dof, ), (n_solve_dof, )
        else:
            return f_res_vect, f_abs_sum_vect  # (n_dof, ), (n_dof, )

    @staticmethod
    def update_hg(hg: Array, phi: Array) -> Array:
        r"""
        Update the nodal homogeneous transformation matrices with the configuration increments.
        :param hg: Existing nodal homogeneous transformation matrices, ``(n_nodes, 4, 4)``
        :param phi: Perturbation to the configuration vector, ``(n_nodes, 6)``
        :return: Updated nodal homogeneous transformation matrices, ``(n_nodes, 4, 4)``
        """
        return jnp.einsum(
            "ijk,ikl->ijl",
            hg,
            vmap(exp_se3, 0, 0)(phi.reshape(-1, 6)),
        )

    def make_prescribed_dofs_tuple(
        self,
        prescribed_dofs: Sequence[int] | Array | slice | int,
    ) -> tuple[int, ...]:
        if isinstance(prescribed_dofs, slice):
            return tuple(jnp.arange(self.n_dof)[prescribed_dofs].tolist())
        elif isinstance(prescribed_dofs, Sequence):
            return tuple(prescribed_dofs)
        elif isinstance(prescribed_dofs, int):
            return (prescribed_dofs,)
        elif isinstance(prescribed_dofs, Array):
            return tuple(jnp.atleast_1d(prescribed_dofs).tolist())
        else:
            raise TypeError(
                "prescribed_dofs must be an int, slice, Sequence[int], or Array"
            )

    def static_solve(
        self,
        prescribed_dofs: Sequence[int] | Array | slice | int,
        f_ext_follower: Array | None = None,
        f_ext_dead: Array | None = None,
        f_ext_aero: Array | None = None,
        load_steps: int = 1,
        *,
        print_header: bool = True,
        postprocess_constraints: bool = True,
    ) -> StructureCase:
        r"""
        Perform static solve of the structure under external loads.
        :param f_ext_follower: External forces array of follower forces ``(n_node, 6)``.
        :param f_ext_dead: External forces array of dead loads ``(n_node, 6)``.
        :param f_ext_aero: External forces array of aerodynamic loads ``(n_node, 6)``.
        :param prescribed_dofs: Index of degrees of freedom which are prescribed (not solved for).
        :param load_steps: Number of load steps to apply the external loads over.
        :param print_header: If False, suppress the "Static Solve" table header and trailing line.
        :param postprocess_constraints: If True, apply constraint postprocessing to the final solution.
        :return: StructureCase object containing results of the static analysis.
        """

        if load_steps < 1:
            raise ValueError("load_steps must be at least 1")

        # check inputs
        if f_ext_follower is not None:
            check_arr_shape(f_ext_follower, (self.n_nodes, 6), "f_ext_follower")
        if f_ext_dead is not None:
            check_arr_shape(f_ext_dead, (self.n_nodes, 6), "f_ext_dead")

        if not (0.0 < self.relaxation_factor <= 1.0):
            raise ValueError("struct_relaxation_factor must be in the range (0, 1]")

        # degrees of freedom to solve for
        prescribed_dofs_: tuple[int, ...] = self.make_prescribed_dofs_tuple(
            prescribed_dofs
        )
        solve_dofs: Array = jnp.array(
            get_solve_dofs(n_dof=self.n_dof, prescribed_dofs=prescribed_dofs_)
        )

        # process external forces for load stepping
        load_step_weight: Array = jnp.linspace(0.0, 1.0, load_steps + 1)[
            1:
        ]  # (load_steps, )

        f_ext_follower_steps = self._make_load_steps_f(
            f_ext_follower, load_step_weight, apply_alpha_weighting=False
        )
        f_ext_dead_steps = self._make_load_steps_f(
            f_ext_dead, load_step_weight, apply_alpha_weighting=False
        )
        f_ext_aero_steps = self._make_load_steps_f(
            f_ext_aero, load_step_weight, apply_alpha_weighting=False
        )

        def _update(
            i_load_step: int,
            converge_status: ConvergenceStatus,
            hg_n: Array,
        ) -> tuple[int, ConvergenceStatus, Array]:
            # base parameters
            d_n = self.make_d(hg_n)  # (n_elem, 6)
            p_d_n = self.make_p_d(d_n)  # (n_elem, 6, 12)
            eps_n = self.make_eps(d_n)  # (n_elem, 6)
            m_t = self.make_m_t(d_n) if self.use_gravity else None  # (n_elem, 12, 12)

            # get total dead forces for this load step, (n_node, 6)
            total_f_ext_dead_step = self.make_f_ext_dead_tot(
                f_ext_dead_steps, f_ext_aero_steps, i_load_step
            )

            # assemble tangent stiffness matrix, (n_dof, n_dof)
            k_t_full_n = self.make_k_t_full(
                d=d_n,
                p_d=p_d_n,
                eps=eps_n,
                f_ext_dead=total_f_ext_dead_step,
                rmat=hg_n[:, :3, :3],
                m_t=m_t,
            )
            # apply nodal constraint contributions
            k_t_full_n = self.apply_nodal_constraint_tangent(
                mat=k_t_full_n, hg=hg_n, i_ts=0, gamma_prime=None
            )
            k_t_solve_n = k_t_full_n[jnp.ix_(solve_dofs, solve_dofs)]

            # compute residual forces, (n_solve_dofs, )
            f_res_solve_n, f_abs_sum_n = self.make_f_res(
                solve_dofs=solve_dofs,
                p_d=p_d_n,
                eps=eps_n,
                hg=hg_n,
                f_ext_follower_n=f_ext_follower_steps[i_load_step, ...]
                if f_ext_follower_steps is not None
                else None,
                f_ext_dead_n=total_f_ext_dead_step,
                thrust_n=self.thrust_reference,  # use reference thrust in static case
                dynamic=False,
                m_t=m_t,
                c_l=None,
                c_l_lumped=None,
                v=None,
                v_dot=None,
            )

            # solve for configuration increment, (n_solve_dofs, )
            if self.n_holonomic_constraints:
                d_varphi_np1, _, _ = self.solve_constrained(
                    sys_mat_solve=k_t_solve_n,
                    f_res_solve=f_res_solve_n,
                    hg_eval=hg_n,
                    solve_dofs=solve_dofs,
                )
                d_varphi_np1 *= self.relaxation_factor
            else:
                d_varphi_np1 = (
                    jnp.linalg.solve(k_t_solve_n, f_res_solve_n)
                    * self.relaxation_factor
                )

            # update configuration, (n_nodes, 4, 4)
            hg_np1_full = self.update_hg(
                hg_n, jnp.zeros(self.n_dof).at[solve_dofs].set(d_varphi_np1)
            )

            # algebra between undeformed and deformed shape, used to check relative convergence, (n_solve_dofs, )
            # this is relatively expensive to compute
            if self.struct_convergence_settings.rel_disp_tol is not None:
                h_full = vmap(hg_to_d, (0, 0), 0)(self.hg0, hg_np1_full).ravel()[
                    solve_dofs
                ]
            else:
                h_full = None

            # update convergence status
            converge_status.update(
                delta_disp=d_varphi_np1,
                total_disp=h_full,
                delta_force=f_res_solve_n,
                total_force=f_abs_sum_n,
            )

            if map_verbosity_level(get_verbosity()) >= map_verbosity_level("verbose"):
                converge_status.print_struct_message(
                    i_ts=None, t=None, i_load_step=i_load_step
                )

            return i_load_step, converge_status, hg_np1_full

        def convergence_loop(
            i_load_step: int,
            hg_init: Array,
        ) -> Array:
            r"""
            Convergence loop
            :param i_load_step: Index of load step.
            :param hg_init: Initial coordinates, ``(n_nodes, 4, 4)``.
            :return: Converged coordinates, ``(n_nodes, 4, 4)``.
            """
            _, convergence_status, hg_solve = eqxi.while_loop(
                lambda args_: ~args_[1].get_status(),
                lambda args_: _update(*args_),
                (
                    i_load_step,
                    ConvergenceStatus(
                        self.struct_convergence_settings,
                    ),
                    hg_init,
                ),
                max_steps=self.struct_convergence_settings.max_n_iter,
                kind="bounded",
            )

            if map_verbosity_level(get_verbosity()) >= map_verbosity_level("normal"):
                convergence_status.print_struct_message(
                    i_ts=None, t=None, i_load_step=i_load_step
                )

            return hg_solve

        if print_header and map_verbosity_level(get_verbosity()) >= map_verbosity_level(
            "normal"
        ):
            ConvergenceStatus.print_header(dynamic=False)

        # solve for each load step
        hg = jax.lax.fori_loop(
            0,
            load_steps,
            lambda *args: convergence_loop(*args),
            self.hg0,
        )

        if print_header and map_verbosity_level(get_verbosity()) >= map_verbosity_level(
            "normal"
        ):
            ConvergenceStatus.print_line(dynamic=False)

        # postprocess final results
        d, eps, f_ext_dead_local, f_ext_aero_local, f_grav, f_int, _, _, f_res = (
            self.resolve_forces(
                hg=hg,
                dynamic=False,
                f_ext_dead=f_ext_dead,
                f_ext_follower=f_ext_follower,
                f_ext_aero=f_ext_aero,
                thrust=self.thrust_reference,
                v=None,
                v_dot=None,
            )
        )
        varphi = self.compute_varphi_from_hg(hg)
        f_elem = self.make_f_elem(eps=eps)  # compute loads in each element

        result = StructureCase(
            hg=hg,
            conn=self.connectivity,
            o0=self.o0,
            d=d,
            eps=eps,
            varphi=varphi,
            f_int=f_int,
            f_elem=f_elem,
            f_ext_follower=f_ext_follower,
            f_ext_dead=f_ext_dead_local,
            f_ext_aero=f_ext_aero_local,
            f_grav=f_grav,
            f_res=f_res,
            thrust=self.thrust_reference,
            thrust_nodes=self.thrust_nodes,
            thrust_direction=self.thrust_direction,
            prescribed_dofs=prescribed_dofs_,
            t=jnp.zeros(1),
        )
        if postprocess_constraints:
            result.constraint_data = self.postprocess_constraints(hg)
        return result

    @overload
    def base_dynamic_solve(
        self,
        struct_case: StructureCase,
        struct_convergence_status: ConvergenceStatus,
        t: Array,
        solve_dofs: tuple[int, ...],
        load_steps: int,
        f_ext_dead: Array | None,
        f_ext_follower: Array | None,
        thrust_t: dict[str, Array],
        aero_obj: None,
        aero_case: None,
        fsi_convergence_status: None,
        cs_ang_t: None,
        cs_vel_t: None,
    ) -> StructureCase: ...

    @overload
    def base_dynamic_solve(
        self,
        struct_case: StructureCase,
        struct_convergence_status: ConvergenceStatus,
        t: Array,
        solve_dofs: tuple[int, ...],
        load_steps: int,
        f_ext_dead: Array | None,
        f_ext_follower: Array | None,
        thrust_t: dict[str, Array],
        aero_obj: DynamicAeroSolver,
        aero_case: AeroCase,
        fsi_convergence_status: ConvergenceStatus,
        cs_ang_t: dict[str, Array],
        cs_vel_t: dict[str, Array],
    ) -> AeroelasticCase: ...

    def base_dynamic_solve(
        self,
        struct_case: StructureCase,
        struct_convergence_status: ConvergenceStatus,
        t: Array,
        solve_dofs: tuple[int, ...],
        load_steps: int,
        f_ext_dead: Array | None,
        f_ext_follower: Array | None,
        thrust_t: dict[str, Array],
        aero_obj: DynamicAeroSolver | None,
        aero_case: AeroCase | None,
        fsi_convergence_status: ConvergenceStatus | None,
        cs_ang_t: dict[str, Array] | None,
        cs_vel_t: dict[str, Array] | None,
    ) -> StructureCase | AeroelasticCase:
        r"""
        Generic dynamic solver. Both the structural dynamic solve, and aeroelastic dynamic solve, are formed as wrappers
        of this
        """

        if not (0.0 < self.relaxation_factor <= 1.0):
            raise ValueError("Relaxation factor must be in range (0, 1]")

        n_tstep = len(t)

        include_aero: bool = aero_obj is not None

        # process external forces for load stepping
        load_step_weight: Array = jnp.linspace(0.0, 1.0, load_steps + 1)[
            1:
        ]  # (load_steps, )
        f_ext_follower_alpha_steps = self._make_load_steps_f(
            f_ext_follower, load_step_weight, apply_alpha_weighting=True
        )
        f_ext_dead_alpha_steps = self._make_load_steps_f(
            f_ext_dead, load_step_weight, apply_alpha_weighting=True
        )

        solve_dofs_arr: Array = jnp.array(solve_dofs)
        prescribed_dofs_arr: Array = jnp.array(
            sorted(set(range(self.n_dof)) - set(solve_dofs)), dtype=int
        )

        def _update(
            i_load_step: int,
            i_ts: int,
            struct_convergence_status_: ConvergenceStatus,
            hg_n: Array,
            phi_alpha: Array,
            q_alpha: StructureMinimalStates,
            f_ext_aero_alpha_steps: Array | None,
            thrust_alpha: dict[str, Array],
        ) -> tuple[
            int,
            int,
            ConvergenceStatus,
            Array,
            Array,
            StructureMinimalStates,
            Array | None,
            dict[str, Array],
        ]:
            r"""
            Solution update for a single iteration of the nonlinear solver at a given time step and load step.
            :param i_load_step: Load step index.
            :param i_ts: Time step index.
            :param struct_convergence_status_: ConvergenceStatus object for the current iteration, used to track
            convergence and print messages.
            :param hg_n: Transformation matrices at iteration varphi, ``(n_nodes, 4, 4)``.
            :param phi_alpha: Timestep increment to the alpha step, ``(n_nodes, 6)``.
            :param f_ext_aero_alpha_steps: Load steps for the external aerodynamic forcing, ``(n_steps, n_nodes, 6)``.
            :param thrust_alpha: Thrust magnitude at the alpha step, ``{keys: ()}``.
            :return: Load and time step indices, updated ConvergenceStatus object, updated transformation matrices,
            configuration, velocities and accelerations for iteration n+1.
            """

            hg_update = self.update_hg(hg_n, phi_alpha)  # (n_node, 4, 4)

            # base parameters
            d_n = self.make_d(hg_update)  # (n_elem, 6)
            p_d_n = self.make_p_d(d_n)  # (n_elem, 6, 12)
            eps_n = self.make_eps(d_n)  # (n_elem, 6)
            d_dot_n = self._make_d_dot(p_d_n, q_alpha.v)  # (n_elem, 6)
            t_n = vmap(t_se3, 0, 0)(phi_alpha)  # (n_node, 6, 6)

            # tangent matrices
            m_t = self.make_m_t(d_n)  # (n_elem, 12, 12)
            c_l, c_t = self._make_c_t(
                d_n, d_dot_n, q_alpha.v
            )  # (n_elem, 12, 12), (n_elem, 12, 12)

            total_f_ext_dead = self.make_f_ext_dead_tot(
                f_ext_dead=f_ext_dead_alpha_steps[:, i_ts, :, :]
                if f_ext_dead_alpha_steps is not None
                else None,
                f_ext_aero=f_ext_aero_alpha_steps,
                i_load_step=i_load_step,
            )  # (n_node, 6)

            k_t = self.make_k_t_full(
                d_n,
                p_d_n,
                eps_n,
                total_f_ext_dead,
                hg_update[:, :3, :3],
                m_t,
            )  # (n_dof, n_dof)

            # add lumped mass contributions if applicable
            if self.use_lumped_mass:
                c_l_lumped, c_t_lumped = self._make_c_t_lumped(
                    q_alpha.v
                )  # (n_node, 6, 6), (n_node, 6, 6)
            else:
                c_l_lumped, c_t_lumped = None, None

            # residual forces, (n_solve_dofs, )
            f_res_n_solve, f_abs_sum_n = self.make_f_res(
                solve_dofs=solve_dofs_arr,
                p_d=p_d_n,
                eps=eps_n,
                hg=hg_update,
                f_ext_follower_n=f_ext_follower_alpha_steps[i_load_step, i_ts, ...]
                if f_ext_follower_alpha_steps is not None
                else None,
                f_ext_dead_n=total_f_ext_dead,
                thrust_n=thrust_alpha,
                dynamic=True,
                m_t=m_t,
                c_l=c_l,
                c_l_lumped=c_l_lumped,
                v=q_alpha.v,
                v_dot=q_alpha.v_dot,
                i_ts=i_ts,
                k_t_assembled=k_t,
            )

            # system matrix, (n_dof, n_dof)
            sys_mat_full = self._make_sys_matrix(
                m_t=m_t,
                c_t=c_t,
                c_t_lumped=c_t_lumped,
                k_t=k_t,
                t_n=t_n,
                ti=self.time_integrator,
            )
            # add nodal constraint contributions
            sys_mat_full = self.apply_nodal_constraint_tangent(
                mat=sys_mat_full,
                hg=hg_update,
                i_ts=i_ts,
                gamma_prime=self.time_integrator.gamma_prime,
            )
            sys_mat = sys_mat_full[jnp.ix_(solve_dofs_arr, solve_dofs_arr)]

            # solve for configuration increment, (n_solve_dofs, )
            if self.multibody_constraints:
                d_n_np1, _, _ = self.solve_constrained(
                    sys_mat_solve=sys_mat,
                    f_res_solve=f_res_n_solve,
                    hg_eval=hg_update,
                    solve_dofs=solve_dofs_arr,
                    hg_base=hg_n,
                    phi=phi_alpha,
                    v=q_alpha.v,
                    gamma_prime=self.time_integrator.gamma_prime,
                )
                d_n_np1 *= self.relaxation_factor
            else:
                d_n_np1 = (
                    jnp.linalg.solve(sys_mat, f_res_n_solve) * self.relaxation_factor
                )
            phi_np1 = phi_alpha.ravel().at[solve_dofs_arr].add(d_n_np1).reshape(-1, 6)

            # update configuration, velocities and accelerations
            v_np1 = (
                q_alpha.v.ravel()
                .at[solve_dofs_arr]
                .add(self.time_integrator.gamma_prime * d_n_np1)
                .reshape(-1, 6)
            )
            v_dot_np1 = (
                q_alpha.v_dot.ravel()
                .at[solve_dofs_arr]
                .add(self.time_integrator.beta_prime * d_n_np1)
                .reshape(-1, 6)
            )

            # update convergence status
            struct_convergence_status_.update(
                delta_disp=d_n_np1,
                total_disp=phi_np1,
                delta_force=f_res_n_solve,
                total_force=f_abs_sum_n,
            )

            if map_verbosity_level(get_verbosity()) >= map_verbosity_level("verbose"):
                struct_convergence_status_.print_struct_message(
                    i_ts=i_ts, t=t[i_ts], i_load_step=i_load_step
                )

            q_alpha_update = StructureMinimalStates(
                varphi=None, v=v_np1, v_dot=v_dot_np1, a=q_alpha.a
            )

            return (
                i_load_step,
                i_ts,
                struct_convergence_status_,
                hg_n,
                phi_np1,
                q_alpha_update,
                f_ext_aero_alpha_steps,
                thrust_alpha,
            )

        @overload
        def time_step_loop(
            i_ts: int,
            struct_sol: StructureCase,
            struct_convergence_status_: ConvergenceStatus,
            aero_sol: None,
            fsi_convergence_status_: None,
            thrust_t_: dict[str, Array],
            cs_ang_t_: None,
            cs_vel_t_: None,
        ) -> tuple[
            StructureCase,
            ConvergenceStatus,
            None,
            None,
            dict[str, Array],
            None,
            None,
        ]: ...

        @overload
        def time_step_loop(
            i_ts: int,
            struct_sol: StructureCase,
            struct_convergence_status_: ConvergenceStatus,
            aero_sol: AeroCase,
            fsi_convergence_status_: ConvergenceStatus,
            thrust_t_: dict[str, Array],
            cs_ang_t_: dict[str, Array],
            cs_vel_t_: dict[str, Array],
        ) -> tuple[
            StructureCase,
            ConvergenceStatus,
            AeroCase,
            ConvergenceStatus,
            dict[str, Array],
            dict[str, Array],
            dict[str, Array],
        ]: ...

        def time_step_loop(
            i_ts: int,
            struct_sol: StructureCase,
            struct_convergence_status_: ConvergenceStatus,
            aero_sol: AeroCase | None,
            fsi_convergence_status_: ConvergenceStatus | None,
            thrust_t_: dict[str, Array],
            cs_ang_t_: dict[str, Array] | None,
            cs_vel_t_: dict[str, Array] | None,
        ) -> tuple[
            StructureCase,
            ConvergenceStatus,
            AeroCase | None,
            ConvergenceStatus | None,
            dict[str, Array],
            dict[str, Array] | None,
            dict[str, Array] | None,
        ]:
            r"""
            Performs analysis on a single time step, including load stepping
            :param i_ts: Index of time step to solve
            :param struct_sol: Solution object, with results up to time step i_ts-1.
            :param struct_convergence_status_: Convergence status object.
            :param aero_sol: Aero solution object, with results up to time step i_ts-1, if aero is included.
            :param fsi_convergence_status_: Convergence status object.
            :param thrust_t_: Thrust magnitude time history, ``{name: (n_tstep, )}``.
            :param cs_ang_t_: Control surface angle time history, ``{name: (n_tstep, )}``.
            :param cs_vel_t_: Control surface velocity time history, ``{name: (n_tstep, )}``.
            :return: Solution object with results up to time step i_ts.
            """

            # predictor step
            q_nm1 = struct_sol.get_minimal_states(i_ts - 1)
            phi_init, q_init = self.time_integrator.predict_q(q_nm1)
            phi_alpha_init, q_alpha_init = self.time_integrator.compute_q_alpha(
                q_nm1=q_nm1,
                q_n=q_init,
                phi_n=phi_init,
            )

            # prescribed DOFs should not be influenced by the time integration
            phi_alpha_init = (
                phi_alpha_init.ravel().at[prescribed_dofs_arr].set(0.0).reshape(-1, 6)
            )
            q_alpha_init.v = (
                q_alpha_init.v.ravel()
                .at[prescribed_dofs_arr]
                .set(q_nm1.v.ravel()[prescribed_dofs_arr])
                .reshape(-1, 6)
            )
            q_alpha_init.v_dot = (
                q_alpha_init.v_dot.ravel()
                .at[prescribed_dofs_arr]
                .set(q_nm1.v_dot.ravel()[prescribed_dofs_arr])
                .reshape(-1, 6)
            )
            q_alpha_init.a = (
                q_alpha_init.a.ravel()
                .at[prescribed_dofs_arr]
                .set(q_nm1.a.ravel()[prescribed_dofs_arr])
                .reshape(-1, 6)
            )

            q_alpha_init.varphi = None  # this value is not used during the loop

            # thrust force
            thrust_alpha: dict[str, Array] = {
                k: self.time_integrator.compute_f_alpha(f_nm1=v[i_ts - 1], f_n=v[i_ts])
                for k, v in thrust_t_.items()
            }
            thrust_n: dict[str, Array] = {k: v[i_ts] for k, v in thrust_t_.items()}

            if include_aero:
                assert (
                    aero_sol is not None
                    and fsi_convergence_status_ is not None
                    and struct_sol.f_ext_aero is not None
                    and cs_ang_t_ is not None
                    and cs_vel_t_ is not None
                )

                fsi_convergence_status_.reset_status()

                # f_ext_aero is stored in local frame, so we convert back to global
                # so that both operands of the alpha blend are in the same (global) frame.
                f_aero_nm1 = jnp.concatenate(
                    [
                        jnp.einsum(
                            "ijk,ik->ij",
                            struct_sol.hg[i_ts - 1, :, :3, :3],
                            struct_sol.f_ext_aero[i_ts - 1, :, :3],
                        ),
                        jnp.einsum(
                            "ijk,ik->ij",
                            struct_sol.hg[i_ts - 1, :, :3, :3],
                            struct_sol.f_ext_aero[i_ts - 1, :, 3:],
                        ),
                    ],
                    axis=-1,
                )

                # get control surface angles and velocities
                cs_ang_nm1 = {k: v[i_ts - 1] for k, v in cs_ang_t_.items()}
                cs_ang_n = {k: v[i_ts] for k, v in cs_ang_t_.items()}
                cs_vel_n = {k: v[i_ts] for k, v in cs_vel_t_.items()}
                assert fsi_convergence_status is not None
                (
                    _,
                    struct_sol,
                    aero_sol,
                    struct_convergence_status_,
                    fsi_convergence_status_,
                    phi_alpha,
                    q_alpha,
                    *_,
                ) = eqxi.while_loop(
                    lambda args_: ~cast(ConvergenceStatus, args_[4]).get_status(),
                    lambda args_: fsi_convergence_loop(*args_),
                    (
                        i_ts,
                        struct_sol,
                        aero_sol,
                        struct_convergence_status_,
                        fsi_convergence_status_,
                        phi_alpha_init,
                        q_alpha_init,
                        f_aero_nm1,  # this value is for the previous timesteps force, and is propagated unaltered
                        f_aero_nm1,  # first guess for forcing at alpha is to use value from i_ts=n-1
                        thrust_alpha,
                        cs_ang_n,
                        cs_ang_nm1,
                        cs_vel_n,
                    ),
                    max_steps=fsi_convergence_status.convergence_settings.max_n_iter,
                    kind="bounded",
                )

            else:
                # solve pure structural problem
                _, struct_convergence_status_, _, phi_alpha, q_alpha, *_ = (
                    load_step_loop(
                        i_ts=i_ts,
                        struct_convergence_status_=struct_convergence_status_,
                        hg_alpha=struct_sol.hg[i_ts - 1, ...],
                        phi_alpha=phi_alpha_init,
                        q_alpha=q_alpha_init,
                        f_ext_aero_steps=None,
                        thrust_alpha=thrust_alpha,
                    )
                )

            # print message where we only require one message per timestep
            if map_verbosity_level(get_verbosity()) == map_verbosity_level("normal"):
                struct_convergence_status_.print_struct_message(
                    i_ts=i_ts, t=struct_sol.t[i_ts], i_load_step=load_steps - 1
                )
                if include_aero and fsi_convergence_status_ is not None:
                    fsi_convergence_status_.print_fsi_message(
                        i_ts=i_ts, t=struct_sol.t[i_ts]
                    )

            # postprocess results for time step and store in solution object
            q_n, phi_n = self.time_integrator.compute_q_n_from_q_alpha(
                q_alpha=q_alpha,
                q_nm1=struct_sol.get_minimal_states(i_ts - 1),
                phi_alpha=phi_alpha,
            )

            # update pseudo-acceleration
            q_n.a = self.time_integrator.compute_a_n(
                a_nm1=struct_sol.a[i_ts - 1, ...],
                v_dot_nm1=struct_sol.v_dot[i_ts - 1, ...],
                v_dot_n=q_n.v_dot,
            )

            # final node coordinates
            hg_n = self.update_hg(struct_sol.hg[i_ts - 1, ...], phi_n)

            if include_aero:
                if (
                    aero_sol is None
                    or aero_obj is None
                    or fsi_convergence_status_ is None
                ):
                    raise ValueError("Missing aero arguments")

                f_ext_aero = aero_sol.project_forcing_to_beam(
                    i_ts=i_ts,
                    rmat=hg_n[:, :3, :3],
                    x0_aero=aero_obj.zeta_b0,
                    include_unsteady=aero_obj.include_unsteady_force,
                )

            else:
                f_ext_aero = None

            (
                d,
                eps,
                f_ext_dead_local,
                f_ext_aero_local,
                f_grav,
                f_int,
                f_gyr,
                f_iner,
                f_res,
            ) = self.resolve_forces(
                hg=hg_n,
                dynamic=True,
                f_ext_dead=f_ext_dead[i_ts, ...] if f_ext_dead is not None else None,
                f_ext_follower=f_ext_follower[i_ts, ...]
                if f_ext_follower is not None
                else None,
                thrust=thrust_n,
                f_ext_aero=f_ext_aero,
                v=q_n.v,
                v_dot=q_n.v_dot,
            )
            struct_sol.d = struct_sol.d.at[i_ts, ...].set(d)
            struct_sol.eps = struct_sol.eps.at[i_ts, ...].set(eps)
            struct_sol.v = struct_sol.v.at[i_ts, ...].set(q_n.v)
            struct_sol.v_dot = struct_sol.v_dot.at[i_ts, ...].set(q_n.v_dot)
            struct_sol.a = struct_sol.a.at[i_ts, ...].set(q_n.a)
            struct_sol.hg = struct_sol.hg.at[i_ts, ...].set(hg_n)
            struct_sol.varphi = struct_sol.varphi.at[i_ts, ...].set(
                vmap(hg_to_d, (0, 0), 0)(self.hg0, hg_n)
            )

            if f_ext_follower is not None and struct_sol.f_ext_follower is not None:
                struct_sol.f_ext_follower = struct_sol.f_ext_follower.at[i_ts, ...].set(
                    f_ext_follower[i_ts, ...]
                )
            if f_ext_dead is not None and struct_sol.f_ext_dead is not None:
                struct_sol.f_ext_dead = struct_sol.f_ext_dead.at[i_ts, ...].set(
                    f_ext_dead_local
                )

            if f_ext_aero is not None and struct_sol.f_ext_aero is not None:
                struct_sol.f_ext_aero = struct_sol.f_ext_aero.at[i_ts, ...].set(
                    f_ext_aero_local
                )

            if self.use_gravity:
                if struct_sol.f_grav is None:
                    raise ValueError("struct_sol.f_grav is None")
                struct_sol.f_grav = struct_sol.f_grav.at[i_ts, ...].set(f_grav)
            struct_sol.f_int = struct_sol.f_int.at[i_ts, ...].set(f_int)
            struct_sol.f_elem = struct_sol.f_elem.at[i_ts, ...].set(
                self.make_f_elem(eps=eps)
            )
            struct_sol.f_iner_gyr = struct_sol.f_iner_gyr.at[i_ts, ...].set(
                f_iner + f_gyr
            )
            struct_sol.f_res = struct_sol.f_res.at[i_ts, ...].set(f_res)

            if include_aero and aero_sol is not None:
                assert cs_ang_t_ is not None and cs_vel_t_ is not None
                cs_ang_n = {k: v[i_ts] for k, v in cs_ang_t_.items()}
                cs_vel_n = {k: v[i_ts] for k, v in cs_vel_t_.items()}
                aero_sol.cs_ang = {
                    k: v.at[i_ts].set(cs_ang_n[k]) for k, v in aero_sol.cs_ang.items()
                }
                aero_sol.cs_vel = {
                    k: v.at[i_ts].set(cs_vel_n[k]) for k, v in aero_sol.cs_vel.items()
                }

            return (
                struct_sol,
                struct_convergence_status_,
                aero_sol,
                fsi_convergence_status_,
                thrust_t_,
                cs_ang_t_,
                cs_vel_t_,
            )

        def fsi_convergence_loop(
            i_ts: int,
            struct_sol: StructureCase,
            aero_sol: AeroCase,
            struct_convergence_status_: ConvergenceStatus,
            fsi_convergence_status_: ConvergenceStatus,
            phi_alpha_init: Array,
            q_alpha_init: StructureMinimalStates,
            f_aero_nm1: Array,
            f_aero_alpha_prev: Array,
            thrust_alpha: dict[str, Array],
            cs_ang_n: dict[str, Array],
            cs_ang_nm1: dict[str, Array],
            cs_vel_n: dict[str, Array],
        ) -> tuple[
            int,
            StructureCase,
            AeroCase,
            ConvergenceStatus,
            ConvergenceStatus,
            Array,
            StructureMinimalStates,
            Array,
            Array,
            dict[str, Array],
            dict[str, Array],
            dict[str, Array],
            dict[str, Array],
        ]:
            # obtain coordinates at timestep (not alpha)
            phi_n = self.time_integrator.compute_phi_from_phi_alpha(
                phi_alpha=phi_alpha_init
            )
            v_n = self.time_integrator.compute_v_from_v_alpha(
                v_alpha=q_alpha_init.v, v_nm1=struct_sol.v[i_ts - 1, ...]
            )

            hg_n = self.update_hg(hg=struct_sol.hg[i_ts - 1, ...], phi=phi_n)
            hg_dot = self.make_hg_dot(hg=hg_n, v=v_n)

            if aero_obj is None or struct_sol.f_ext_aero is None:
                raise ValueError("Missing aero parameters")

            # evaluate aerodynamic forcing on beam
            aero_sol = aero_obj.case_solve(
                case=aero_sol,
                i_ts=i_ts,
                hg_n=hg_n,
                hg_nm1=struct_sol.hg[i_ts - 1, ...],
                hg_dot_n=hg_dot,
                static=False,
                horseshoe=False,
                cs_ang_n=cs_ang_n,
                cs_ang_nm1=cs_ang_nm1,
                cs_vel_n=cs_vel_n,
            )

            f_aero_n = aero_sol.project_forcing_to_beam(
                i_ts=i_ts,
                rmat=hg_n[:, :3, :3],
                x0_aero=aero_obj.zeta_b0,
                include_unsteady=aero_obj.include_unsteady_force,
            )

            # aerodynamic force at alpha point, subsequently divided into load steps
            f_aero_alpha = self.time_integrator.compute_f_alpha(
                f_nm1=f_aero_nm1, f_n=f_aero_n
            )

            f_aero_alpha_steps = self._make_load_steps_f(
                f=f_aero_alpha, weighting=load_step_weight, apply_alpha_weighting=False
            )

            # reset convergence status
            struct_convergence_status_.reset_status()

            # solve structural problem for given aero load
            _, struct_convergence_status_, _, phi_alpha, q_alpha, *_ = load_step_loop(
                i_ts,
                struct_convergence_status_,
                struct_sol.hg[i_ts - 1, ...],
                phi_alpha_init,
                q_alpha_init,
                f_aero_alpha_steps,
                thrust_alpha,
            )

            # update the FSI convergence object
            # note that for convenience we use the alpha properties
            fsi_convergence_status_.update(
                delta_disp=(phi_alpha_init - phi_alpha).ravel()[solve_dofs_arr],
                total_disp=phi_alpha.ravel()[solve_dofs_arr],
                delta_force=(f_aero_alpha - f_aero_alpha_prev).ravel()[solve_dofs_arr],
                total_force=f_aero_alpha.ravel()[solve_dofs_arr],
            )

            if map_verbosity_level(get_verbosity()) >= map_verbosity_level("verbose"):
                fsi_convergence_status_.print_fsi_message(i_ts=i_ts, t=t[i_ts])

            return (
                i_ts,
                struct_sol,
                aero_sol,
                struct_convergence_status_,
                fsi_convergence_status_,
                phi_alpha,
                q_alpha,
                f_aero_nm1,
                f_aero_alpha,
                thrust_alpha,
                cs_ang_n,
                cs_ang_nm1,
                cs_vel_n,
            )

        def struct_convergence_loop(
            i_load_step: int,
            i_ts: int,
            struct_convergence_status_: ConvergenceStatus,
            hg_alpha: Array,
            phi_alpha: Array,
            q_alpha: StructureMinimalStates,
            f_ext_aero_steps: Array | None,
            thrust_alpha: dict[str, Array],
        ) -> tuple[
            int,
            ConvergenceStatus,
            Array,
            Array,
            StructureMinimalStates,
            Array | None,
            dict[str, Array],
        ]:
            r"""
            Convergence loop within each load step of a time step.
            :param i_load_step: Load step index.
            :param i_ts: Time step index.
            :param struct_convergence_status_: ConvergenceStatus object to update with convergence information during load
            stepping.
            :param hg_alpha: Node transformations at the beginning of the load step, ``(n_nodes, 4, 4)``.
            :param phi_alpha: Node configuration increments in algebra space, ``(n_nodes, 6)``.
            :param q_alpha: Minimal states at intermediate alpha step.
            :param f_ext_aero_steps: Optional aerodynamic forcing alpha load steps ``(n_steps, n_nodes, 6)``.
            :param thrust_alpha: Thrust at the alpha step, ``{key: ()}``.
            :return: Time step index, convergence status, and updated configuration, velocities, accelerations, and
            optional aerodynamic forcing.
            """

            struct_convergence_status_.reset_status()

            _, _, struct_convergence_status_, hg_solve, phi_alpha, q_alpha, _, _ = (
                eqxi.while_loop(
                    lambda args_: ~args_[2].get_status(),
                    lambda args_: _update(*args_),
                    (
                        i_load_step,
                        i_ts,
                        struct_convergence_status_,
                        hg_alpha,
                        phi_alpha,
                        q_alpha,
                        f_ext_aero_steps,
                        thrust_alpha,
                    ),
                    max_steps=self.struct_convergence_settings.max_n_iter,
                    kind="bounded",
                )
            )

            if map_verbosity_level(get_verbosity()) >= map_verbosity_level("verbose"):
                struct_convergence_status_.print_struct_message(
                    i_ts=i_ts, t=t[i_ts], i_load_step=i_load_step
                )

            return (
                i_ts,
                struct_convergence_status_,
                hg_solve,
                phi_alpha,
                q_alpha,
                f_ext_aero_steps,
                thrust_alpha,
            )

        def load_step_loop(
            i_ts: int,
            struct_convergence_status_: ConvergenceStatus,
            hg_alpha: Array,
            phi_alpha: Array,
            q_alpha: StructureMinimalStates,
            f_ext_aero_steps: Array | None,
            thrust_alpha: dict[str, Array],
        ) -> tuple[
            int,
            ConvergenceStatus,
            Array,
            Array,
            StructureMinimalStates,
            Array | None,
        ]:
            r"""
            Performs load stepping iterations for a given time step. Load stepping is not performed for thrust.
            :param i_ts: Timestep index for which to perform load stepping.
            :param struct_convergence_status_: ConvergenceStatus object to update with load stepping convergence information.
            :param hg_alpha: SE(3) nodal transformation matrices at the beginning of the load step, ``(n_nodes, 4, 4)``.
            :param phi_alpha: Nodal updates to the configuration in the algebra space, ``(n_nodes, 6)``.
            :param q_alpha: Minimal states at intermediate alpha step.
            :param f_ext_aero_steps: Optional aerodynamic forcing alpha load steps ``(n_steps, n_nodes, 6)``.
            :param thrust_alpha: Thrust at the alpha step, ``{key: ()}``.
            :return: Time step index, updated ConvergenceStatus object, and updated configuration, velocities and accelerations after load stepping
            """
            return jax.lax.fori_loop(
                0,
                load_steps,
                lambda i_load_step, args: struct_convergence_loop(i_load_step, *args),
                (
                    i_ts,
                    struct_convergence_status_,
                    hg_alpha,
                    phi_alpha,
                    q_alpha,
                    f_ext_aero_steps,
                    thrust_alpha,
                ),
            )

        struct_case, _, aero_case, *_ = jax.lax.fori_loop(
            1,
            n_tstep,
            lambda i_ts, args: time_step_loop(i_ts, *args),
            (
                struct_case,
                struct_convergence_status,
                aero_case,
                fsi_convergence_status,
                thrust_t,
                cs_ang_t,
                cs_vel_t,
            ),
        )

        struct_case.constraint_data = self.postprocess_constraints(struct_case.hg)

        if include_aero:
            if aero_case is None:
                raise ValueError("aero_case cannot be None")

            from flapjax.coupled.data_structures import (
                AeroelasticCase,
            )  # import here to prevent circular references

            return AeroelasticCase(structure=struct_case, aero=aero_case)
        else:
            return struct_case

    def dynamic_solve(
        self,
        init_state: StructureCase | None,
        n_tstep: int,
        dt: Array | float,
        prescribed_dofs: Sequence[int] | Array | slice | int | None = None,
        f_ext_follower: Array | None = None,
        f_ext_dead: Array | None = None,
        f_ext_aero: Array | None = None,
        thrust_t: dict[str, Array] | None = None,
        load_steps: int = 1,
    ) -> StructureCase:
        r"""
        Perform dynamic solve of the structure under external loads
        :param init_state: Initial state of the structure, either static or a
        dynamic snapshot. If None, the reference configuration is used with zero
        velocities.
        :param prescribed_dofs: Degrees of freedom which are prescribed (not solved for). If None, inherit
        from the initial state.
        :param n_tstep: Number of time steps to simulate.
        :param dt: Time step length.
        :param f_ext_follower: Following external forces array, ``(n_tstep, n_node, 6)``, ``(n_node, 6)`` or None for zero external follower forces.
        :param f_ext_dead: Dead external forces array, ``(n_tstep, n_node, 6)``, ``(n_node, 6)`` or None for zero external dead forces.
        :param f_ext_aero: Aerodynamic external forces array, ``(n_tstep, n_node, 6)``, ``(n_node, 6)`` or None for zero external aerodynamic forces.
        :param thrust_t: Thrust time history, ``{key: (n_tstep, )}``. If none, this will use the reference value.
        :param load_steps: Number of load steps to apply the external loads over.
        :return: Structure dataclass containing results of the dynamic analysis.
        """

        if load_steps <= 0:
            raise ValueError("load_steps must be a positive integer")

        # set thrust if not provided
        thrust_t_: dict[str, Array] = (
            thrust_t
            if thrust_t is not None
            else {k: jnp.full(n_tstep, v) for k, v in self.thrust_reference.items()}
        )

        if prescribed_dofs is None:
            # inherit prescribed dofs from initial state
            if init_state is None:
                raise ValueError("prescribed_dofs cannot be None if init_state is None")
            prescribed_dofs = init_state.prescribed_dofs

        # degrees of freedom to solve for
        prescribed_dofs_arr = self.make_prescribed_dofs_tuple(prescribed_dofs)
        solve_dofs = get_solve_dofs(
            n_dof=self.n_dof, prescribed_dofs=prescribed_dofs_arr
        )

        # check and process external forces
        def check_force(arr: Array | None, name: str) -> Array | None:
            if arr is None:
                return None
            match arr.ndim:
                case 2:
                    out_ = jnp.broadcast_to(arr[None, ...], (n_tstep, self.n_nodes, 6))
                case 3:
                    out_ = arr
                case _:
                    raise ValueError(
                        f"{name} must have shape [n_node, 6] or [n_tstep, n_node, 6]"
                    )
            check_arr_shape(out_, (n_tstep, self.n_nodes, 6), name)
            return out_

        f_ext_dead = check_force(f_ext_dead, "f_ext_dead")  # (n_tstep, n_node, 6)
        f_ext_follower = check_force(
            f_ext_follower, "f_ext_follower"
        )  # (n_tstep, n_node, 6)
        f_ext_aero = check_force(f_ext_aero, "f_ext_aero")  # (n_tstep, n_node, 6)

        # time integration parameters
        self.time_integrator = TimeIntegrator(
            spectral_radius=self.spectral_radius, dt=jnp.array(dt)
        )

        def evaluate_initial_equilibrium(
            init_state__: StructureCase,
        ) -> StructureCase:
            r"""
            Evaluates the forces for a given initial state to check whether it is in equilibrium. If not, a warning is
            raised with the maximum residual force. This is important to ensure that the time integration starts from a
            consistent state.
            :param init_state__: Structure containing the initial state to evaluate.
            :return: Structure with the forces evaluated for the initial state.
            """
            d, eps, f_ext_dead_, f_ext_aero_, f_grav, f_int, f_gyr, f_iner, f_res = (
                self.resolve_forces(
                    hg=init_state__.hg,
                    dynamic=True,
                    f_ext_dead=init_state__.f_ext_dead,
                    f_ext_aero=init_state__.f_ext_aero,
                    thrust=init_state__.thrust,
                    f_ext_follower=init_state__.f_ext_follower,
                    v=init_state__.v,
                    v_dot=init_state__.v_dot,
                )
            )

            max_res = jnp.max(jnp.abs(f_res))
            jax_print(
                "Initial state maximum residual force: {max_res:.3e}",
                max_res=max_res,
                verbose_level="normal",
            )

            f_elem = self.make_f_elem(eps=eps)

            return StructureCase(
                hg=init_state__.hg,
                conn=self.connectivity,
                o0=self.o0,
                d=d,
                eps=eps,
                varphi=init_state__.varphi,
                v=init_state__.v,
                v_dot=init_state__.v_dot,
                a=init_state__.v_dot,  # initial pseudo-acceleration set equal to initial acceleration
                f_ext_follower=init_state__.f_ext_follower,
                f_ext_dead=f_ext_dead_,
                f_ext_aero=f_ext_aero_,
                f_grav=f_grav,
                f_int=f_int,
                f_elem=f_elem,
                f_iner_gyr=f_iner + f_gyr,  # type: ignore
                f_res=f_res,
                thrust=init_state__.thrust,
                thrust_nodes=self.thrust_nodes,
                thrust_direction=self.thrust_direction,
                t=init_state__.t,
                i_ts=init_state__.i_ts,
                prescribed_dofs=prescribed_dofs_arr,
            )

        # time steps
        t = jnp.arange(n_tstep) * dt
        if init_state is not None and init_state.is_dynamic:
            if init_state.is_batched:
                t += init_state.t[0]
            else:
                t += init_state.t

        # set up initial state
        if init_state is None:
            init_state_: StructureCase = self.reference_configuration(
                use_f_aero=f_ext_aero is not None,
                use_f_ext_dead=f_ext_dead is not None,
                use_f_ext_follower=f_ext_follower is not None,
                prescribed_dofs=tuple(prescribed_dofs_arr),
            ).to_dynamic(t=None)
        elif not init_state.is_dynamic:
            init_state_ = init_state.to_dynamic(t=None)
        elif not init_state.is_batched:
            init_state_ = init_state
        else:
            raise TypeError(
                "dynamic_solve init_state cannot be a batched Structure; pass a "
                "snapshot or static state"
            )

        # check if initial state satisfies equilibrium
        init_state_eval = evaluate_initial_equilibrium(init_state_)
        dynamic_struct = StructureCase.initialise(
            initial_snapshot=init_state_eval,
            t=t,
            use_f_ext_follower=f_ext_follower is not None,
            use_f_ext_dead=f_ext_dead is not None,
            use_f_ext_aero=False,
        )
        converge_status = ConvergenceStatus(
            convergence_settings=self.struct_convergence_settings
        )

        ConvergenceStatus.print_header(dynamic=True)

        out = self.base_dynamic_solve(
            struct_case=dynamic_struct,
            struct_convergence_status=converge_status,
            t=t,
            solve_dofs=solve_dofs,
            load_steps=load_steps,
            f_ext_dead=f_ext_dead,
            f_ext_follower=f_ext_follower,
            aero_obj=None,
            aero_case=None,
            fsi_convergence_status=None,
            thrust_t=thrust_t_,
            cs_ang_t=None,
            cs_vel_t=None,
        )

        ConvergenceStatus.print_line(dynamic=True)
        return out
