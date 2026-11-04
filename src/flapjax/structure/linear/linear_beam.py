from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, Literal

from jax import Array, vmap
from jax import numpy as jnp
from jax.scipy.linalg import lu_factor, lu_solve

from flapjax.algebra.array_utils import check_arr_shape
from flapjax.algebra.se3 import exp_se3
from flapjax.structure.data_structures import StructureCase
from flapjax.structure.linear.data_structures import (
    StructureInputUnflattened,
    StructureLinearResult,
    StructureOutputUnflattened,
    StructureStateUnflattened,
)
from flapjax.structure.utils import get_solve_dofs, transform_nodal_vect
from flapjax.utils.constants import BASE_LOBATTO_ORDER
from flapjax.utils.linear import LinearComponent, LinearModel, LinearSystem, SliceEntry
from flapjax.utils.utils import make_pytree

if TYPE_CHECKING:
    from flapjax.structure.beam import BaseBeamStructure


@make_pytree
class LinearBeam(
    LinearModel[
        StructureCase,
        StructureInputUnflattened,
        StructureStateUnflattened,
        StructureOutputUnflattened,
        StructureLinearResult,
    ]
):
    r"""
    Class to represent a linearised beam system about a reference state.
    """

    _static: ClassVar[tuple[str, ...]] = (
        "free_dofs",
        "n_free_dof",
        "n_nodes",
        "modal_states",
        "modal_inputs",
        "modal_outputs",
        "_n_modes",
        "alpha_m",
        "beta_k",
        "n_inputs",
        "n_states",
        "n_outputs",
    )

    def __init__(
        self,
        beam: BaseBeamStructure,
        reference: StructureCase,
        n_modes: int | None,
        dt: float | Array,
        modal_inputs: bool = False,
        modal_outputs: bool = False,
        int_order: Literal[3, 4, 5] = BASE_LOBATTO_ORDER,
        prescribed_dofs: Sequence[int] | Array | slice | int | None = None,
    ):
        if prescribed_dofs is not None:
            prescribed_dofs = beam.make_prescribed_dofs_tuple(prescribed_dofs)
            free_dofs = get_solve_dofs(
                n_dof=beam.n_dof, prescribed_dofs=prescribed_dofs
            )
        else:
            # inherit from the reference by default
            free_dofs = reference.free_dofs
        self.free_dofs: tuple = free_dofs
        self.n_free_dof: int = len(free_dofs)
        self.n_nodes: int = beam.n_nodes
        self.modal_states: bool = n_modes is not None
        self.modal_inputs: bool = modal_inputs
        self.modal_outputs: bool = modal_outputs
        self._n_modes: int | None = n_modes

        if not self.modal_states and (self.modal_inputs or self.modal_outputs):
            raise ValueError(
                "Modal projection for inputs or outputs requires the system states to be modal."
            )

        # compute m, k, mode_shapes so that the superclass has access when initialised
        if self.modal_states:
            assert n_modes is not None
            self.m, self.k, self.mode_shapes = beam.make_modal_m_k(
                case=reference,
                int_order=int_order,
                n_modes=n_modes,
            )
        else:
            self.m, self.k = beam.make_nodal_m_k(case=reference, int_order=int_order)
            self.mode_shapes = None

        # Rayleigh damping matrix
        self.alpha_m: float = beam.alpha_m
        self.beta_k: float = beam.beta_k
        if beam.alpha_m != 0.0 or beam.beta_k != 0.0:
            self.c = beam.alpha_m * self.m + beam.beta_k * self.k
        else:
            self.c = None

        super().__init__(reference=reference, dt=dt)

        self.sys: LinearSystem = self.linearise()

    @property
    def n_modes(self) -> int:
        assert self._n_modes is not None
        return self._n_modes

    @property
    def reference(self) -> StructureCase:
        return self._reference

    def nodal_to_modal(self, q_nodal: Array) -> Array:
        r"""
        Convert a nodal property to a modal property.
        :param q_nodal: Nodal property, ``(n_free_dof, )``.
        :return: Modal property, ``(n_modes, )``.
        """

        return self.mode_shapes @ q_nodal

    def modal_to_nodal(self, q_modal: Array) -> Array:
        r"""
        Convert a modal property to a nodal property.
        :param q_modal: Mode property, ``(n_modes, )``.
        :return: Nodal property, ``(n_free_dofs, )``.
        """

        return q_modal @ self.mode_shapes

    def extract_reference_inputs(
        self,
    ) -> dict[str, Array | None]:
        # combine follower and dead reference forces into a single global-frame reference
        rmat_ref = self.reference.hg[:, :3, :3]
        follower_global = (
            transform_nodal_vect(self.reference.f_ext_follower, rmat_ref)
            if self.reference.f_ext_follower is not None
            else None
        )
        dead_global = (
            transform_nodal_vect(self.reference.f_ext_dead, rmat_ref)
            if self.reference.f_ext_dead is not None
            else None
        )
        if follower_global is None and dead_global is None:
            f_ext_ref = None
        elif follower_global is None:
            f_ext_ref = dead_global
        elif dead_global is None:
            f_ext_ref = follower_global
        else:
            f_ext_ref = follower_global + dead_global

        if not self.modal_inputs or f_ext_ref is None:
            return {"f_ext": f_ext_ref}

        # modal_inputs: project the free-dof global-frame reference onto modes
        assert self.mode_shapes is not None
        free_dofs_arr = jnp.array(self.free_dofs)
        return {"f_ext": self.nodal_to_modal(f_ext_ref.ravel()[free_dofs_arr])}

    def extract_reference_states(
        self,
    ) -> dict[str, Array | None]:
        return {
            "q": jnp.zeros(self.n_modes)
            if self.modal_states
            else jnp.zeros(len(self.reference.free_dofs)),
            "q_dot": jnp.zeros(self.n_modes)
            if self.modal_states
            else jnp.zeros(len(self.reference.free_dofs)),
        }

    def extract_reference_outputs(
        self,
    ) -> dict[str, Array | None]:
        return {
            "q": jnp.zeros(self.n_modes)
            if self.modal_outputs
            else jnp.zeros(len(self.reference.free_dofs)),
            "q_dot": jnp.zeros(self.n_modes)
            if self.modal_outputs
            else jnp.zeros(len(self.reference.free_dofs)),
        }

    @property
    def input_object(self) -> type[StructureInputUnflattened]:
        return StructureInputUnflattened

    @property
    def state_object(
        self,
    ) -> type[StructureStateUnflattened]:
        return StructureStateUnflattened

    @property
    def output_object(
        self,
    ) -> type[StructureOutputUnflattened]:
        return StructureOutputUnflattened

    def _make_input_slices(
        self,
        reference: StructureCase,
    ) -> tuple[dict[str, LinearComponent], int]:
        r"""
        Create input slices for the input vector.
        :return: InputSlices instance and total number of input elements.
        """
        slice_entries = (
            SliceEntry(
                name="f_ext",
                enabled=True,
                shapes=(self.n_modes,) if self.modal_inputs else (self.n_nodes, 6),
            ),
        )

        return self._make_slices(slice_entries)

    def _make_state_slices(
        self, reference: StructureCase
    ) -> tuple[dict[str, LinearComponent], int]:
        r"""
        Create state slices for the state vector.
        :return: StateSlices instance and total number of state elements.
        """
        slice_entries = (
            SliceEntry(
                name="q",
                enabled=True,
                shapes=(self.n_modes,) if self.modal_states else (self.n_free_dof,),
            ),
            SliceEntry(
                name="q_dot",
                enabled=True,
                shapes=(self.n_modes,) if self.modal_states else (self.n_free_dof,),
            ),
        )
        return self._make_slices(slice_entries)

    def _make_output_slices(
        self, reference: StructureCase
    ) -> tuple[dict[str, LinearComponent], int]:
        r"""
        Create output slices for the output vector.
        :return: OutputSlices instance and total number of output elements.
        """
        slice_entries = (
            SliceEntry(
                name="q",
                enabled=True,
                shapes=(self.n_modes,) if self.modal_outputs else (self.n_free_dof,),
            ),
            SliceEntry(
                name="q_dot",
                enabled=True,
                shapes=(self.n_modes,) if self.modal_outputs else (self.n_free_dof,),
            ),
        )
        return self._make_slices(slice_entries)

    def linearise(self) -> LinearSystem:
        cont_sys = self.linearise_continuous()
        return cont_sys.continuous_to_discrete(method="tustin")

    def linearise_continuous(self) -> LinearSystem:
        r"""
        Form a system of linear equations about a reference state. The system is of the form:
        :math:`\dot{\mathbf{x}} = \mathbf{A~x + B~u}, \mathbf{y} = \mathbf{C~x + D~u}`.

        The state, input and output layouts each depend on their respective ``modal_*`` flag:

        * States: nodal-free-dof :math:`[q, \dot q]` of length ``2 * n_free_dof`` when ``n_modes is None``, or modal
          :math:`[q_m, \dot q_m]` of length ``2 * n_modes`` when ``modal_states``. Modal mass and stiffness are the
          projected :math:`\Phi M \Phi^T` / :math:`\Phi K \Phi^T` with ``Phi = mode_shapes`` of shape
          ``[n_modes, n_free_dof]`` (rows are mode shapes).
        * Inputs: a single global-frame external force ``f_ext``. Non-modal inputs are per-node ``[n_nodes, 6]``. Modal
          inputs are direct modal forces of length ``n_modes``, requiring the user to have already applied the modal
          projection.
        * Outputs: :math:`[q, \dot q]` in either nodal-free-dof or modal form to match ``modal_outputs``.

        :return: Linearised continuous-time system.
        """

        # single LU factorisation of M reused for every M^{-1} product below
        m_lu = lu_factor(self.m)  # [q_state_size, q_state_size]
        q_state_size = self.n_modes if self.modal_states else self.n_free_dof

        # dynamics matrix: [q_dot; q_ddot] = A [q; q_dot]
        # includes Raleigh damping contribution if enabled
        m_inv_c = (
            lu_solve(m_lu, self.c)
            if self.c is not None
            else jnp.zeros((q_state_size, q_state_size))
        )
        a = jnp.block(
            [
                [jnp.zeros((q_state_size, q_state_size)), jnp.eye(q_state_size)],
                [-lu_solve(m_lu, self.k), -m_inv_c],
            ]
        )

        # input matrix: maps a single global-frame external force to state derivative
        if self.modal_inputs:
            assert self.modal_states, "modal_inputs requires modal_states"
            # user supplies modal forces directly
            b_bottom = lu_solve(m_lu, jnp.eye(q_state_size))  # (n_modes, n_modes)
        else:
            p_free = jnp.eye(self.n_nodes * 6)[
                jnp.array(self.free_dofs), :
            ]  # (n_free_dof, n_nodes * 6)

            if self.modal_states:
                nodal_to_state = self.nodal_to_modal(p_free)  # (n_modes, n_nodes * 6)
            else:
                nodal_to_state = p_free  # (n_free_dof, n_nodes * 6)
            b_bottom = lu_solve(
                m_lu, nodal_to_state
            )  # (n_modes | n_free_dof, n_nodes * 6)

        b = jnp.concatenate(
            [jnp.zeros((q_state_size, b_bottom.shape[1])), b_bottom], axis=0
        )

        if self.modal_states and not self.modal_outputs:
            assert self.mode_shapes is not None
            state_to_output = (
                self.mode_shapes.T
            )  # projects force to modes, (n_free_dof, n_modes)
        else:
            state_to_output = jnp.eye(
                q_state_size
            )  # direct projection, (n_free_dof, n_free_dof) or (n_modes, n_modes)

        c = jnp.block(
            [
                [state_to_output, jnp.zeros_like(state_to_output)],
                [jnp.zeros_like(state_to_output), state_to_output],
            ]
        )  # pass state (q, q_dot) to output (q, q_dot)
        d = jnp.zeros((c.shape[0], b.shape[1]))  # no feedthrough

        return LinearSystem(
            a=a, b=b, c=c, d=d, dt=self.dt, continuous_time=True, removed_u_np1=False
        )

    # noinspection PyMethodOverriding
    def run(
        self,
        u: StructureInputUnflattened,
        x0: StructureStateUnflattened | None = None,
    ) -> StructureLinearResult:
        r"""
        Run the linear system.
        :param u: Total input over time (reference + pertubation).
        :param x0: Initial state perturbations, defaults to zero state.
        :return: Linear system results.
        """

        assert (
            isinstance(self.reference, StructureCase) and not self.reference.is_dynamic
        ), "Reference structure must be a static Structure"

        # has to be specified in the input object, as the linear system does not know how many time steps to run for in
        # the case where there is no external forcing
        n_tstep = u.n_tstep

        q_state_size = self.n_modes if self.modal_states else self.n_free_dof

        # initial states
        if x0 is None:
            q0_ = None
            q0_dot_ = None
        else:
            q0_ = x0.q
            q0_dot_ = x0.q_dot

        def _project_initial(q0_arr: Array | None) -> Array:
            if q0_arr is None:
                return jnp.zeros((q_state_size,))

            match q0_arr.shape:
                case (self.n_nodes, 6):
                    q0_nodal = q0_arr.ravel()[self.free_dofs]  # remove prescribed dofs
                    if self.modal_states:
                        q0__ = self.nodal_to_modal(q0_nodal)
                    else:
                        q0__ = q0_nodal
                case (self.n_modes,):
                    if not self.modal_states:
                        raise NotImplementedError(
                            "Inputs cannot be modal for a nodal system"
                        )
                    q0__ = q0_arr
                case _:
                    raise ValueError("Invalid input for initial states")
            return q0__

        q0 = _project_initial(q0_)  # (n_free_dof | n_modes, )
        q0_dot = _project_initial(q0_dot_)  # (n_free_dof | n_modes, )

        if q0.shape != q0_dot.shape:
            raise ValueError("q0 and q0_dot must both be modal or nodal")

        ref_f_ext = self._reference_inputs["f_ext"]
        assert ref_f_ext is None or isinstance(
            ref_f_ext, Array
        )  # type narrowing as it cannot be an ArrayList

        if self.modal_inputs:
            # user-provided input is a modal force time history of shape (n_tstep, n_modes)
            if u.f_ext is None:
                delta_f_ext_t = (
                    jnp.zeros((n_tstep, self.n_modes))
                    if ref_f_ext is None
                    else -jnp.broadcast_to(ref_f_ext[None, :], (n_tstep, self.n_modes))
                )
            else:
                check_arr_shape(u.f_ext, (n_tstep, self.n_modes), name="f_ext_t")
                delta_f_ext_t = (
                    u.f_ext if ref_f_ext is None else u.f_ext - ref_f_ext[None, :]
                )
        else:
            # user-provided input is a nodal global-frame force time history of shape (n_tstep, n_nodes, 6)
            if u.f_ext is None:
                delta_f_ext_t = (
                    jnp.zeros((n_tstep, self.n_nodes, 6))
                    if ref_f_ext is None
                    else -jnp.broadcast_to(
                        ref_f_ext[None, :, :], (n_tstep, self.n_nodes, 6)
                    )
                )
            else:
                check_arr_shape(u.f_ext, (n_tstep, self.n_nodes, 6), name="f_ext_t")
                delta_f_ext_t = (
                    u.f_ext if ref_f_ext is None else u.f_ext - ref_f_ext[None, :, :]
                )

        delta_u = StructureInputUnflattened(n_tstep=n_tstep, f_ext=delta_f_ext_t)
        delta_u_vec = self._pack_input_vector_t(delta_u)

        # run linear system
        x_t, _ = self.sys.run(
            u=delta_u_vec,
            x0=jnp.concatenate((q0, q0_dot)),
        )

        # extract perturbations in displacements and velocities, reconstructing nodal form if requested
        delta_q_state = x_t[:, :q_state_size]
        delta_q_dot_state = x_t[:, q_state_size:]

        if self.modal_outputs:
            delta_q_t = delta_q_state
            delta_q_dot_t = delta_q_dot_state
            hg_t = None  # do not reconstruct coordinates
        else:
            if self.modal_states:
                assert self.mode_shapes is not None
                delta_q_t = self.modal_to_nodal(delta_q_state)  # (n_tstep, n_free_dof)
                delta_q_dot_t = self.modal_to_nodal(delta_q_dot_state)
            else:
                delta_q_t = delta_q_state
                delta_q_dot_t = delta_q_dot_state

            # add in zeros for dofs not solved for to allow for reconstructing the full configuration
            delta_q_t_full = (
                jnp.zeros((n_tstep, self.n_nodes * 6))
                .at[:, self.free_dofs]
                .set(delta_q_t)
                .reshape(n_tstep, self.n_nodes, 6)
            )

            delta_hg_t = vmap(vmap(exp_se3, 0, 0), 1, 1)(
                delta_q_t_full
            )  # (n_tstep, n_nodes, 4, 4)

            hg_t = jnp.einsum("ijk,hikl->hijl", self.reference.hg, delta_hg_t)

        return StructureLinearResult(
            reference=self.reference,
            f_ext=u.f_ext,
            delta_q=delta_q_t,
            delta_q_dot=delta_q_dot_t,
            hg=hg_t,
            t=jnp.arange(n_tstep) * self.dt,
        )
