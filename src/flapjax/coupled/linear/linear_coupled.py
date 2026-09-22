from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal

import jax
from jax import Array, vmap
from jax import numpy as jnp
from matplotlib import pyplot as plt

from flapjax.aero.frequency_flowfields import FrequencyFlowField
from flapjax.aero.linear import LinearUVLM, LinearWakeType
from flapjax.aero.linear.data_structures import (
    AeroInputUnflattened,
    AeroOutputUnflattened,
    AeroStateUnflattened,
)
from flapjax.aero.linear.linear_uvlm import LinearInputProjection
from flapjax.aero.utils import project_forcing_to_beam
from flapjax.algebra.array_utils import ArrayList, construct_named_block_jacobian
from flapjax.algebra.base import ADMode, jacrev_custom
from flapjax.algebra.se3 import exp_se3, ha_to_ha_tilde
from flapjax.coupled import AeroelasticCase
from flapjax.coupled.linear.data_structures import (
    AeroelasticInputUnflattened,
    AeroelasticLinearResult,
    AeroelasticOutputUnflattened,
    AeroelasticStateUnflattened,
)
from flapjax.plotting.modal import plot_modes_vtu
from flapjax.structure.linear.linear_beam import LinearBeam
from flapjax.structure.utils import get_solve_dofs
from flapjax.utils.constants import BASE_LOBATTO_ORDER
from flapjax.utils.linear import (
    LinearComponent,
    LinearModel,
    LinearSystem,
    SliceEntry,
    conjugate_partner_mask,
)
from flapjax.utils.print_utils import (
    jax_print,
    print_table_line,
    print_table_title,
)

if TYPE_CHECKING:
    from flapjax.coupled.coupled import BaseCoupledAeroelastic


class LinearCoupled(
    LinearModel[
        AeroelasticCase,
        AeroelasticInputUnflattened,
        AeroelasticStateUnflattened,
        AeroelasticOutputUnflattened,
        AeroelasticLinearResult,
    ]
):
    def __init__(
        self,
        case: BaseCoupledAeroelastic,
        reference: AeroelasticCase,
        batch_size: int | None,
        n_struct_modes: int | None,
        wake_type: LinearWakeType = "frozen",
        bound_upwash: bool = True,
        wake_upwash: bool = False,
        unsteady_force: bool = True,
        int_order: Literal[3, 4, 5] = BASE_LOBATTO_ORDER,
        *,
        skip_checks: bool = False,
        prescribed_dofs: Sequence[int] | Array | slice | int | None = None,
    ):
        if prescribed_dofs is not None:
            prescribed_dofs = case.structure.make_prescribed_dofs_tuple(prescribed_dofs)

        self.aero = LinearUVLM(
            case=case.aero,
            reference=reference.aero,
            wake_type=wake_type,
            bound_upwash=bound_upwash,
            wake_upwash=wake_upwash,
            unsteady_force=unsteady_force,
            skip_linearisation=True,
            skip_checks=skip_checks,
        )
        self.structure = LinearBeam(
            beam=case.structure,
            reference=reference.structure,
            dt=case.aero.dt,
            n_modes=n_struct_modes,
            int_order=int_order,
            prescribed_dofs=prescribed_dofs,
        )

        effective_prescribed = (
            prescribed_dofs
            if prescribed_dofs is not None
            else reference.structure.prescribed_dofs
        )
        self.n_beam_nodal_dof: int = case.structure.n_dof - len(effective_prescribed)
        self.n_beam_input_dof: int = (
            self.structure.n_modes
            if self.structure.modal_inputs
            else self.n_beam_nodal_dof
        )
        self.n_beam_state_dof: int = (
            self.structure.n_modes
            if self.structure.modal_states
            else self.n_beam_nodal_dof
        )
        self.n_beam_output_dof: int = (
            self.structure.n_modes
            if self.structure.modal_outputs
            else self.n_beam_nodal_dof
        )

        self.n_nodes: int = case.structure.n_nodes
        self.free_dofs: Array = jnp.array(
            get_solve_dofs(
                n_dof=case.structure.n_dof,
                prescribed_dofs=effective_prescribed,
            )
        )

        super().__init__(reference=reference, dt=case.aero.dt)

        self._case: BaseCoupledAeroelastic = case
        self.unsteady_force: bool = unsteady_force
        if batch_size is not False:
            self.sys = self.linearise(batch_size=batch_size)

    @property
    def reference(self) -> AeroelasticCase:
        return self._reference

    def extract_reference_inputs(
        self,
    ) -> dict[str, Array | ArrayList | None]:
        inputs = (
            self.aero.extract_reference_inputs()
            | self.structure.extract_reference_inputs()
        )

        # remove redundant
        inputs.pop("zeta_b")
        inputs.pop("zeta_b_dot")
        return inputs

    def extract_reference_states(
        self,
    ) -> dict[str, Array | ArrayList | None]:
        states = (
            self.aero.extract_reference_states()
            | self.structure.extract_reference_states()
        )

        # remove redundant
        states.pop("zeta_b")
        return states

    def extract_reference_outputs(
        self,
    ) -> dict[str, Array | ArrayList | None]:
        states = (
            self.aero.extract_reference_outputs()
            | self.structure.extract_reference_outputs()
        )

        # remove redundant
        states.pop("f_steady")
        states.pop("f_unsteady")
        return states

    @property
    def input_object(self) -> type[AeroelasticInputUnflattened]:
        return AeroelasticInputUnflattened

    @property
    def state_object(
        self,
    ) -> type[AeroelasticStateUnflattened]:
        return AeroelasticStateUnflattened

    @property
    def output_object(
        self,
    ) -> type[AeroelasticOutputUnflattened]:
        return AeroelasticOutputUnflattened

    def _make_input_slices(
        self, reference: AeroelasticCase
    ) -> tuple[dict[str, LinearComponent], int]:
        r"""
        Create input slices for the input vector.
        :return: InputSlices instance and total number of input elements.
        """

        slice_entries = (
            SliceEntry(
                "nu_b",
                *(
                    (True, reference.aero.zeta_b.shape)
                    if self.aero.bound_upwash
                    else (False, None)
                ),
            ),
            SliceEntry(
                "nu_w",
                *(
                    (True, reference.aero.zeta_w.shape)
                    if self.aero.wake_upwash
                    else (False, None)
                ),
            ),
            SliceEntry(
                "f_ext",
                True,
                # this also allows for input forces on prescribed degrees of freedom for consistency, even if they do
                # nothing
                (self.structure.n_modes,)
                if self.structure.modal_inputs
                else (self.n_nodes, 6),
            ),
        )
        return self._make_slices(slice_entries)

    def _make_state_slices(
        self, reference: AeroelasticCase
    ) -> tuple[dict[str, LinearComponent], int]:
        r"""
        Create state slices for the state vector.
        :return: StateSlices instance and total number of state elements.
        """
        slice_entries = (
            SliceEntry("gamma_b", True, reference.aero.gamma_b.shape),
            SliceEntry("gamma_w", True, reference.aero.gamma_w.shape),
            SliceEntry(
                "gamma_b_nm1",
                *(
                    (True, reference.aero.gamma_b.shape)
                    if self.aero.unsteady_force
                    else (False, None)
                ),
            ),
            SliceEntry(
                "zeta_w",
                *(
                    (True, reference.aero.zeta_w.shape)
                    if self.aero.prescribed_wake
                    else (False, None)
                ),
            ),
            SliceEntry("q", True, (self.n_beam_state_dof,)),
            SliceEntry("q_dot", True, (self.n_beam_state_dof,)),
        )
        return self._make_slices(slice_entries)

    def _make_output_slices(
        self, reference: AeroelasticCase
    ) -> tuple[dict[str, LinearComponent], int]:
        r"""
        Create output slices for the output vector.
        :return: OutputSlices instance and total number of output elements.
        """
        slice_entries = (
            SliceEntry("q", True, (self.n_beam_output_dof,)),
            SliceEntry("q_dot", True, (self.n_beam_output_dof,)),
        )
        return self._make_slices(slice_entries)

    def step(
        self,
        gamma_b_vec: Array | None = None,
        gamma_w_vec: Array | None = None,
        gamma_b_nm1_vec: Array | None = None,
        zeta_w_vec: Array | None = None,
        nu_b_vec: Array | None = None,
        nu_w_vec: Array | None = None,
        f_ext: Array | None = None,
        q_nodal: Array | None = None,
        q_dot_nodal: Array | None = None,
    ) -> tuple[AeroelasticStateUnflattened, AeroelasticOutputUnflattened]:
        r"""
        Step solution from states at timestep n and inputs at timestep n+1 to give states at timestep n+1 and outputs
        at timestep n.
        """
        ref = self.reference

        # unravel vector inputs, falling back to reference values when None
        gamma_b = (
            ArrayList.from_vector(vect=gamma_b_vec, shapes=ref.aero.gamma_b.shape)
            if gamma_b_vec is not None
            else ref.aero.gamma_b
        )
        gamma_w = (
            ArrayList.from_vector(vect=gamma_w_vec, shapes=ref.aero.gamma_w.shape)
            if gamma_w_vec is not None
            else ref.aero.gamma_w
        )
        gamma_b_nm1 = (
            (
                ArrayList.from_vector(
                    vect=gamma_b_nm1_vec, shapes=ref.aero.gamma_b.shape
                )
                if gamma_b_nm1_vec is not None
                else ref.aero.gamma_b
            )
            if self.aero.unsteady_force
            else None
        )
        zeta_w = (
            ArrayList.from_vector(vect=zeta_w_vec, shapes=ref.aero.zeta_w.shape)
            if zeta_w_vec is not None
            else ref.aero.zeta_w
        )
        nu_b = (
            (
                ArrayList.from_vector(vect=nu_b_vec, shapes=ref.aero.zeta_b.shape)
                if nu_b_vec is not None
                else ArrayList.zeros_like(ref.aero.zeta_b)
            )
            if self.aero.bound_upwash
            else None
        )
        nu_w = (
            (
                ArrayList.from_vector(vect=nu_w_vec, shapes=ref.aero.zeta_w.shape)
                if nu_w_vec is not None
                else ArrayList.zeros_like(ref.aero.zeta_w)
            )
            if self.aero.wake_upwash
            else None
        )
        q_nodal = (
            q_nodal if q_nodal is not None else jnp.zeros(len(self.structure.free_dofs))
        )
        q_dot_nodal = (
            q_dot_nodal
            if q_dot_nodal is not None
            else jnp.zeros(len(self.structure.free_dofs))
        )

        # fill in prescribed dofs with zeros
        q_full = (
            jnp.zeros(self.n_nodes * 6)
            .at[self.free_dofs]
            .set(q_nodal)
            .reshape(self.n_nodes, 6)
        )
        q_dot_full = (
            jnp.zeros(self.n_nodes * 6)
            .at[self.free_dofs]
            .set(q_dot_nodal)
            .reshape(self.n_nodes, 6)
        )

        # total perturbed coordinates and time derivative
        hg = jnp.einsum("ijk,ikl->ijl", ref.structure.hg, vmap(exp_se3)(q_full))
        hg_dot = jnp.einsum(
            "ijk,ikl->ijl",
            ref.structure.hg,
            vmap(ha_to_ha_tilde)(q_dot_full),
        )

        # aerodynamic grid
        zeta_b = self.aero.case.hg_to_zeta_b(hg_n=hg, cs_ang_n=self.aero.case.cs_ang0)
        zeta_b_dot = self.aero.case.hg_dot_to_zeta_b_dot(
            hg_n=hg,
            hg_dot_n=hg_dot,
            cs_ang_n=self.aero.case.cs_ang0,
            cs_vel_n=self.aero.case.cs_vel0,
        )

        # pass through aerodynamic system
        u_n_aero = AeroInputUnflattened(
            zeta_b=zeta_b, zeta_b_dot=zeta_b_dot, nu_b=nu_b, nu_w=nu_w
        )

        x_n_aero = AeroStateUnflattened(
            gamma_b=gamma_b,
            gamma_w=gamma_w,
            gamma_b_nm1=gamma_b_nm1,
            zeta_w=zeta_w if self.aero.prescribed_wake else None,
            zeta_b=zeta_b if self.aero.prescribed_wake else None,
        )

        u_n_aero_vec = self.aero.pack_input_vector(u_n_aero)
        x_n_aero_vec = self.aero.pack_state_vector(x_n_aero)

        x_np1_aero_vec, y_np1_aero_vec = self.aero.step_vec(
            x_vec=x_n_aero_vec, u_vec=u_n_aero_vec
        )

        x_np1_aero = self.aero.unpack_state_vector(x=x_np1_aero_vec)
        y_np1_aero = self.aero.unpack_output_vector(y=y_np1_aero_vec)
        assert isinstance(x_np1_aero, AeroStateUnflattened) and isinstance(
            y_np1_aero, AeroOutputUnflattened
        ), (
            "Unpacked aero state and output must be of type AeroStateUnflattened and AeroOutputUnflattened."
        )

        # total aero forces on the grid, from the aero step (returns totals)
        f_aero_np1 = y_np1_aero.f_steady
        if self.unsteady_force:
            assert y_np1_aero.f_unsteady is not None
            # don't want to mutate in place
            # noinspection augment-assignment
            f_aero_np1 = f_aero_np1 + y_np1_aero.f_unsteady

        # project total aero forces onto the beam under the current (perturbed) rotation
        rmat = hg[:, :3, :3]
        f_aero_beam_total = project_forcing_to_beam(
            f_total=f_aero_np1,
            rmat=rmat,
            dof_mapping=self.aero.case.dof_mapping,
            x0_aero=self.aero.case.zeta_b0,
            mirror_edge_low=self.aero.case.mirror_edge_low,
            mirror_edge_high=self.aero.case.mirror_edge_high,
        )

        # subtract the reference contribution so the aero forcing fed into the beam operator
        # is a pure perturbation (the beam sys.a / sys.b operate on perturbations)
        f_aero_ref_total = ref.aero.f_steady
        if self.aero.unsteady_force:
            # noinspection augment-assignment
            f_aero_ref_total = f_aero_ref_total + ref.aero.f_unsteady
        f_aero_beam_ref = project_forcing_to_beam(
            f_total=f_aero_ref_total,
            rmat=ref.structure.hg[:, :3, :3],
            dof_mapping=self.aero.case.dof_mapping,
            x0_aero=self.aero.case.zeta_b0,
            mirror_edge_low=self.aero.case.mirror_edge_low,
            mirror_edge_high=self.aero.case.mirror_edge_high,
        )
        delta_f_aero_beam = f_aero_beam_total - f_aero_beam_ref

        f_ext_: Array = f_ext if f_ext is not None else jnp.zeros(self.free_dofs.size)

        # scatter free-dof forcing to full (n_nodes * 6) vectors expected by the beam B operator, and add aero
        # forcing (global frame, perturbation) as an external force
        f_ext_full = (
            jnp.zeros(self.n_nodes * 6).at[self.free_dofs].set(f_ext_)
            + delta_f_aero_beam.ravel()
        )  # [n_nodes * 6]

        if self.structure.modal_inputs:
            f_ext_full = self.structure.nodal_to_modal(f_ext_full[self.free_dofs])

        # beam step in perturbation form (discrete-time Tustin)
        x_beam_n = jnp.concatenate(
            [
                self.structure.nodal_to_modal(q_nodal)
                if self.structure.modal_states
                else q_nodal,
                self.structure.nodal_to_modal(q_dot_nodal)
                if self.structure.modal_states
                else q_dot_nodal,
            ]
        )

        x_beam_np1 = (
            self.structure.sys.a @ x_beam_n
            + self.structure.sys.b[:, self.structure.input_slices["f_ext"].slices]
            @ f_ext_full
        )

        q_np1 = x_beam_np1[: self.n_beam_state_dof]
        q_dot_np1 = x_beam_np1[self.n_beam_state_dof :]

        state_np1 = AeroelasticStateUnflattened(
            gamma_b=x_np1_aero.gamma_b,
            gamma_w=x_np1_aero.gamma_w,
            gamma_b_nm1=x_np1_aero.gamma_b_nm1,
            zeta_w=x_np1_aero.zeta_w,
            q=q_np1,
            q_dot=q_dot_np1,
        )
        output_n = AeroelasticOutputUnflattened(q=q_np1, q_dot=q_dot_np1)

        return state_np1, output_n

    def gamma_b_step(
        self,
        gamma_b_n_vec: Array,
        gamma_w_n_vec: Array,
        q_n: Array,
        q_dot_n: Array,
        zeta_w_n_vec: Array | None = None,
        nu_b_n_vec: Array | None = None,
    ) -> Array:
        r"""
        Bound circulation at timestep n+1 as a function of states at n and inputs at n+1.
        """
        x_np1, _ = self.step(
            nu_b_vec=nu_b_n_vec,
            gamma_b_vec=gamma_b_n_vec,
            gamma_w_vec=gamma_w_n_vec,
            zeta_w_vec=zeta_w_n_vec,
            q_nodal=self.structure.modal_to_nodal(q_n)
            if self.structure.modal_states
            else q_n,
            q_dot_nodal=self.structure.modal_to_nodal(q_dot_n)
            if self.structure.modal_states
            else q_dot_n,
        )
        return x_np1.gamma_b.ravel()

    def wake_prop_step(
        self,
        gamma_b_n_vec: Array,
        gamma_w_n_vec: Array,
        q_n: Array,
        zeta_w_n_vec: Array | None = None,
        nu_w_n_vec: Array | None = None,
    ) -> tuple[Array | None, Array]:
        r"""
        Wake propagation as a function of states at n and inputs at n+1.
        """
        x_new, _ = self.step(
            nu_w_vec=nu_w_n_vec,
            gamma_b_vec=gamma_b_n_vec,
            gamma_w_vec=gamma_w_n_vec,
            zeta_w_vec=zeta_w_n_vec,
            q_nodal=self.structure.modal_to_nodal(q_n)
            if self.structure.modal_states
            else q_n,
        )
        assert not ((x_new.zeta_w is not None) ^ self.aero.prescribed_wake), (
            "zeta_w should be None only if prescribed_wake is False."
        )
        return (
            x_new.zeta_w.ravel() if x_new.zeta_w is not None else None,
            x_new.gamma_w.ravel(),
        )

    def q_step(
        self,
        q_n: Array,
        q_dot_n: Array,
        gamma_b_n_vec: Array,
        gamma_w_n_vec: Array,
        f_ext: Array | None,
        zeta_w_n_vec: Array | None = None,
        gamma_b_nm1_vec: Array | None = None,
        nu_b_n_vec: Array | None = None,
        nu_w_n_vec: Array | None = None,
    ) -> Array:
        r"""
        Beam displacement at timestep n+1 as a function of states at n and external forcing.
        """
        x_new, _ = self.step(
            f_ext=f_ext,
            gamma_b_vec=gamma_b_n_vec,
            gamma_w_vec=gamma_w_n_vec,
            gamma_b_nm1_vec=gamma_b_nm1_vec,
            zeta_w_vec=zeta_w_n_vec,
            nu_b_vec=nu_b_n_vec,
            nu_w_vec=nu_w_n_vec,
            q_nodal=self.structure.modal_to_nodal(q_n)
            if self.structure.modal_states
            else q_n,
            q_dot_nodal=self.structure.modal_to_nodal(q_dot_n)
            if self.structure.modal_states
            else q_dot_n,
        )
        return x_new.q.ravel()

    def q_dot_step(
        self,
        q_n: Array,
        q_dot_n: Array,
        gamma_b_n_vec: Array,
        gamma_w_n_vec: Array,
        f_ext: Array | None,
        zeta_w_n_vec: Array | None,
        gamma_b_nm1_vec: Array | None = None,
        nu_b_n_vec: Array | None = None,
        nu_w_n_vec: Array | None = None,
    ) -> Array:
        r"""
        Beam velocity at timestep n+1 as a function of states at n and external forcing.
        """
        x_new, _ = self.step(
            f_ext=f_ext,
            gamma_b_vec=gamma_b_n_vec,
            gamma_w_vec=gamma_w_n_vec,
            gamma_b_nm1_vec=gamma_b_nm1_vec,
            zeta_w_vec=zeta_w_n_vec,
            nu_b_vec=nu_b_n_vec,
            nu_w_vec=nu_w_n_vec,
            q_nodal=self.structure.modal_to_nodal(q_n)
            if self.structure.modal_states
            else q_n,
            q_dot_nodal=self.structure.modal_to_nodal(q_dot_n)
            if self.structure.modal_states
            else q_dot_n,
        )
        return x_new.q_dot.ravel()

    def compute_jacobians(
        self,
    ) -> tuple[
        dict[str, tuple[Callable[..., Any], dict[str, Any], Sequence[str]]],
        dict[str, dict[str, Callable[..., Array]]],
    ]:
        r"""
        :return: Tuple. First entry is dictionaries with keys being the function name (e.g., gamma_b, gamma_w), with
        each entry containing the relevant stepping function, the arguments for the function, and the name of the
        arguments for which to obtain derivatives. Second entry is a dictionary of explicit Jacobians functions that
        take the same arguments as the first output.
        """
        ref = self.reference

        # bound circulation
        gamma_b_args: dict[str, Any] = {
            "gamma_b_n_vec": ref.aero.gamma_b.ravel(),
            "gamma_w_n_vec": ref.aero.gamma_w.ravel(),
            "q_n": jnp.zeros((self.n_beam_state_dof,)),
            "q_dot_n": jnp.zeros((self.n_beam_state_dof,)),
            "zeta_w_n_vec": ref.aero.zeta_w.ravel(),
            "nu_b_n_vec": jnp.zeros(ref.aero.zeta_b.size)
            if self.aero.bound_upwash
            else None,
        }
        gamma_b_diff = ["gamma_w_n_vec", "q_n", "q_dot_n"]
        if self.aero.prescribed_wake:
            gamma_b_diff.append("zeta_w_n_vec")
        if self.aero.bound_upwash:
            gamma_b_diff.append("nu_b_n_vec")

        # wake
        wake_args: dict[str, Any] = {
            "gamma_b_n_vec": ref.aero.gamma_b.ravel(),
            "gamma_w_n_vec": ref.aero.gamma_w.ravel(),
            "q_n": jnp.zeros((self.n_beam_state_dof,)),
            "zeta_w_n_vec": ref.aero.zeta_w.ravel(),
            "nu_w_n_vec": jnp.zeros(ref.aero.zeta_w.size)
            if self.aero.wake_upwash
            else None,
        }
        gamma_w_diff = ["gamma_b_n_vec", "gamma_w_n_vec"]
        zeta_w_diff = ["q_n"]
        if self.aero.prescribed_wake:
            zeta_w_diff.append("zeta_w_n_vec")
        if self.aero.wake_upwash:
            zeta_w_diff.append("nu_w_n_vec")
        if self.aero.free_wake:
            zeta_w_diff.extend(["gamma_b_n_vec", "gamma_w_n_vec"])

        q_args: dict[str, Any] = {
            "q_n": jnp.zeros((self.n_beam_state_dof,)),
            "q_dot_n": jnp.zeros((self.n_beam_state_dof,)),
            "gamma_b_n_vec": ref.aero.gamma_b.ravel(),
            "gamma_w_n_vec": ref.aero.gamma_w.ravel(),
            "f_ext": jnp.zeros((self.n_beam_input_dof,)),
            "zeta_w_n_vec": ref.aero.zeta_w.ravel(),
            "gamma_b_nm1_vec": ref.aero.gamma_b.ravel()
            if self.aero.unsteady_force
            else None,
            "nu_b_n_vec": jnp.zeros(ref.aero.zeta_b.size)
            if self.aero.bound_upwash
            else None,
            "nu_w_n_vec": jnp.zeros(ref.aero.zeta_w.size)
            if self.aero.wake_upwash
            else None,
        }
        q_diff = [
            "q_n",
            "q_dot_n",
            "gamma_b_n_vec",
            "gamma_w_n_vec",
            "f_ext",
        ]

        if self.aero.prescribed_wake:
            q_diff.append("zeta_w_n_vec")
        if self.aero.unsteady_force:
            q_diff.append("gamma_b_nm1_vec")
        if self.aero.bound_upwash:
            q_diff.append("nu_b_n_vec")
        if self.aero.wake_upwash:
            q_diff.append("nu_w_n_vec")

        linear_args: dict[
            str, tuple[Callable[..., Any], dict[str, Any], Sequence[str]]
        ] = {
            "gamma_b": (self.gamma_b_step, gamma_b_args, gamma_b_diff),
            "gamma_w": (
                lambda *args, **kwargs: self.wake_prop_step(**kwargs)[1],
                wake_args,
                gamma_w_diff,
            ),
            "gamma_b_nm1": (
                lambda *args, **kwargs: None,
                {
                    "gamma_b_n_vec": ref.aero.gamma_b.ravel(),
                },
                ["gamma_b_n_vec"],
            ),
            "q": (self.q_step, q_args, q_diff),
            "q_dot": (self.q_dot_step, q_args, q_diff),
        }

        # add zeta_w for linearisation
        if self.aero.prescribed_wake:
            linear_args["zeta_w"] = (
                lambda *args, **kwargs: self.wake_prop_step(**kwargs)[0],
                wake_args,
                zeta_w_diff,
            )

        # define Jacobians we know nicely as jac_options
        jac_options = {
            "q": {
                "q_n": lambda *args, **kwargs: self.structure.sys.a[
                    : self.n_beam_state_dof, : self.n_beam_state_dof
                ],
                "q_dot_n": lambda *args, **kwargs: self.structure.sys.a[
                    : self.n_beam_state_dof, self.n_beam_state_dof :
                ],
            },
            "q_dot": {
                "q_n": lambda *args, **kwargs: self.structure.sys.a[
                    self.n_beam_state_dof :, : self.n_beam_state_dof
                ],
                "q_dot_n": lambda *args, **kwargs: self.structure.sys.a[
                    self.n_beam_state_dof :, self.n_beam_state_dof :
                ],
            },
        }
        if self.aero.unsteady_force:
            jac_options["gamma_b_nm1"] = {
                "gamma_b_n_vec": lambda *args, **kwargs: jnp.eye(ref.aero.gamma_b.size)
            }

        return linear_args, jac_options

    def _build_beam_projection(self) -> LinearInputProjection:
        r"""
        Build the beam kinematics projection ``(q_n, q_dot_n) -> (zeta_b_vec, zeta_b_dot_vec)``.
        """
        ref = self.reference

        def to_zeta(q_n: Array, q_dot_n: Array) -> tuple[Array, Array]:
            q_nodal = (
                self.structure.modal_to_nodal(q_n)
                if self.structure.modal_states
                else q_n
            )
            q_dot_nodal = (
                self.structure.modal_to_nodal(q_dot_n)
                if self.structure.modal_states
                else q_dot_n
            )
            q_full = (
                jnp.zeros(self.n_nodes * 6)
                .at[self.free_dofs]
                .set(q_nodal)
                .reshape(self.n_nodes, 6)
            )
            q_dot_full = (
                jnp.zeros(self.n_nodes * 6)
                .at[self.free_dofs]
                .set(q_dot_nodal)
                .reshape(self.n_nodes, 6)
            )
            hg = jnp.einsum("ijk,ikl->ijl", ref.structure.hg, vmap(exp_se3)(q_full))
            hg_dot = jnp.einsum(
                "ijk,ikl->ijl",
                ref.structure.hg,
                vmap(ha_to_ha_tilde)(q_dot_full),
            )
            zeta_b = self.aero.case.hg_to_zeta_b(
                hg_n=hg, cs_ang_n=self.aero.case.cs_ang0
            )
            zeta_b_dot = self.aero.case.hg_dot_to_zeta_b_dot(
                hg_n=hg,
                hg_dot_n=hg_dot,
                cs_ang_n=self.aero.case.cs_ang0,
                cs_vel_n=self.aero.case.cs_vel0,
            )
            return zeta_b.ravel(), zeta_b_dot.ravel()

        return LinearInputProjection(
            arg_names=("q_n", "q_dot_n"),
            arg_sizes=(self.n_beam_state_dof, self.n_beam_state_dof),
            arg_refs=(
                jnp.zeros(self.n_beam_state_dof),
                jnp.zeros(self.n_beam_state_dof),
            ),
            to_zeta=to_zeta,
        )

    def create_jacobians(
        self,
        mode: ADMode | dict[str, ADMode] = "reverse",
        batch_size: int | None = None,
        n_profile_loops: int | None = None,
        jac_options: dict[str, dict[str, Callable[..., Any] | None]] | None = None,
    ) -> tuple[
        dict[str, dict[str, Array]],
        dict[str, dict[str, float]] | None,
        dict[str, dict[str, float]] | None,
    ]:
        """
        Assemble the per-residual Jacobians for the coupled linearisation. When
        ``n_profile_loops`` is set, all Jacobian constructions are looped locally so that they can be timed.
        """
        res_args, jac_options_exp = self.compute_jacobians()
        jac_options_total: dict[str, dict[str, Callable[..., Any] | None]] = (
            jac_options if jac_options is not None else {}
        ) | jac_options_exp

        jacobians: dict[str, dict[str, Array]] = {}
        compile_time: dict[str, dict[str, float]] = {}
        run_time: dict[str, dict[str, float]] = {}

        aero_residual_names = {"gamma_b", "gamma_w", "gamma_b_nm1", "zeta_w"}
        delegate_aero = n_profile_loops is None

        for res_name, (res_func, args, diff_arg_names) in res_args.items():
            if delegate_aero and res_name in aero_residual_names:
                continue

            res_jac_options: dict[str, Callable[..., Any] | None] = {
                arg: None for arg in diff_arg_names
            }
            if res_name in jac_options_total:
                for arg, entry in jac_options_total[res_name].items():
                    if arg in res_jac_options:
                        res_jac_options[arg] = entry

            if isinstance(mode, str):
                res_mode: ADMode = mode
            elif isinstance(mode, dict):
                try:
                    res_mode = mode[res_name]
                except KeyError:
                    res_mode = "reverse"
            else:
                raise NotImplementedError

            jacs, res_compile_time, res_run_time = jacrev_custom(
                func=res_func,
                jac_options=res_jac_options,
                n_profile_loops=n_profile_loops,
                func_name=res_name,
                map_batch_size=batch_size,
                mode=res_mode,
            )(**args)

            jacobians[res_name] = jacs
            if n_profile_loops is not None:
                assert res_compile_time is not None and res_run_time is not None
                compile_time[res_name] = res_compile_time
                run_time[res_name] = res_run_time

        if delegate_aero:
            # delegate aero linearisation to the aero system, with the beam kinematics wrapped as an input projection
            beam_proj = self._build_beam_projection()
            aero_mode: ADMode | dict[str, ADMode]
            if isinstance(mode, dict):
                aero_mode = {k: mode[k] for k in aero_residual_names if k in mode}
            else:
                aero_mode = mode
            needed_aero_residuals = {"gamma_b", "gamma_w"}
            if self.aero.unsteady_force:
                needed_aero_residuals.add("gamma_b_nm1")
            if self.aero.prescribed_wake:
                needed_aero_residuals.add("zeta_w")
            aero_jacs = self.aero.create_jacobians(
                mode=aero_mode,
                batch_size=batch_size,
                input_projection=beam_proj,
                residual_names=tuple(needed_aero_residuals),
            )
            jacobians.update(aero_jacs)

        return (
            jacobians,
            compile_time if n_profile_loops is not None else None,
            run_time if n_profile_loops is not None else None,
        )

    def linearise(
        self,
        batch_size: int | None = None,
    ) -> LinearSystem:
        jacobians, _, _ = self.create_jacobians(
            n_profile_loops=None,
            jac_options=None,
            batch_size=batch_size,
            mode={"gamma_b": "forward"} if self.structure.modal_states else {},  # type: ignore
        )

        # (row-label in `jacobians`, column-argname used inside each jacobian dict, size)
        state_specs: list[tuple[str, str, int]] = [
            ("gamma_b", "gamma_b_n_vec", self.reference.aero.gamma_b.size),
            ("gamma_w", "gamma_w_n_vec", self.reference.aero.gamma_w.size),
        ]
        if self.aero.unsteady_force:
            state_specs.append(
                ("gamma_b_nm1", "gamma_b_nm1_vec", self.reference.aero.gamma_b.size)
            )
        if self.aero.prescribed_wake:
            state_specs.append(
                ("zeta_w", "zeta_w_n_vec", self.reference.aero.zeta_w.size)
            )
        state_specs.extend(
            [
                ("q", "q_n", self.n_beam_state_dof),
                ("q_dot", "q_dot_n", self.n_beam_state_dof),
            ]
        )
        state_names = [s[0] for s in state_specs]
        state_arg_names = [s[1] for s in state_specs]
        state_sizes = [s[2] for s in state_specs]

        input_specs: list[tuple[str, int]] = []
        if self.aero.bound_upwash:
            input_specs.append(("nu_b_n_vec", self.reference.aero.zeta_b.size))
        if self.aero.wake_upwash:
            input_specs.append(("nu_w_n_vec", self.reference.aero.zeta_w.size))
        input_specs.append(("f_ext", self.n_beam_input_dof))
        input_arg_names = [s[0] for s in input_specs]
        input_sizes = [s[1] for s in input_specs]

        output_sizes = [self.n_beam_output_dof, self.n_beam_output_dof]

        a = construct_named_block_jacobian(
            entries=tuple(jacobians[k] for k in state_names),
            keys=state_arg_names,
            widths=state_sizes,
            heights=state_sizes,
        )

        b = construct_named_block_jacobian(
            entries=tuple(jacobians[k] for k in state_names),
            keys=input_arg_names,
            widths=input_sizes,
            heights=state_sizes,
        )

        c_entries: tuple[dict[str, Array], ...] = (
            {
                "q_n": self.structure.sys.c[
                    : self.n_beam_output_dof, : self.n_beam_state_dof
                ]
            },
            {
                "q_dot_n": self.structure.sys.c[
                    self.n_beam_output_dof :, self.n_beam_state_dof :
                ]
            },
        )
        c = construct_named_block_jacobian(
            entries=c_entries,
            keys=state_arg_names,
            widths=state_sizes,
            heights=output_sizes,
        )
        d = jnp.zeros((sum(output_sizes), sum(input_sizes)))

        return LinearSystem(a=a, b=b, c=c, d=d, dt=self.dt, removed_u_np1=False)

    def linearise_profile(
        self,
        n_profile_loops: int = 3,
    ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
        r"""
        Profile forming the Jacobians required for the linearised model.
        :param n_profile_loops: Number of times to loop Jacobian creation for averaging.
        :return: Dictionaries of compile and run times for each sub function.
        """

        print_table_title(inner_width=95, title="Aeroelastic Adjoint Profile")

        _, compile_time, run_time = self.create_jacobians(
            n_profile_loops=n_profile_loops,
            jac_options=None,
            mode={"gamma_b": "forward"} if self.structure.modal_states else {},  # type: ignore
        )

        assert compile_time is not None and run_time is not None, (
            "No output timings passed"
        )

        print_table_line(inner_width=95)

        return compile_time, run_time

    def modal(
        self,
        n_modes: int | None = None,
        freq_range: tuple[float | Array, float | Array] = (0.0, jnp.inf),
        damp_range: tuple[float | Array, float | Array] = (-jnp.inf, jnp.inf),
        min_struct_content: float | Array = 0.0,
        remove_complex_conjugate: bool = True,
        plot_eigvals: bool = False,
        sort: Literal["frequency", "damping"] = "frequency",
        plot_xlim: tuple[float, float] = (-500.0, 50.0),
        plot_ylim: tuple[float, float] = (-400.0, 400.0),
        n_plot_vtk: int = 0,
        vtu_directory: os.PathLike | str = "./modal",
        n_phase: int = 8,
        n_interp: int = 0,
        max_disp: float = 0.2,
        max_ang: float = 0.2,
        max_gamma: float = 100.0,
    ) -> Array:
        r"""
        Compute stability eigenvalues of the linear system A matrix.
        :param n_modes: Number of modes to be kept. If None, all eigenvalues are returned.
        :param freq_range: (min, max) natural frequency window in Hz. Modes outside are pushed past the
        truncation and dropped when n_modes is set.
        :param damp_range: (min, max) damping-ratio window. Modes outside are pushed past the truncation and
        dropped when n_modes is set.
        :param min_struct_content: Minimum fraction of eigenvector energy that must live in the beam
        ``(q, q_dot)`` states in range `[0, 1]`. Modes below this threshold (typically wake convection modes) are pushed
        past the truncation.
        :param remove_complex_conjugate: If true, one mode from each complex-conjugate pair is dropped.
        :param plot_eigvals: If true, plot the eigenvalues with Matplotlib.
        :param sort: Method for sorting eigenvalues before truncation, can be either "frequency" or "damping".
        :param plot_xlim: Range of real component to be used for plotting.
        :param plot_ylim: Range of imaginary component to be used for plotting.
        :param n_plot_vtk: Number of modes (starting from the most damped) to write to VTK for visualisation. Set to 0
        to skip plotting.
        :param vtu_directory: Directory for plotting vtu files, defaults to "./modal".
        :param n_phase: Number of phase samples of the complex eigenvector to plot per mode.
        :param n_interp: Number of interpolation points to add along each beam element in the beam VTU output.
        :param max_disp: Maximum linear displacement used to normalise the plotted mode shape (in reference units).
        :param max_ang: Maximum angular displacement used to normalise the plotted mode shape.
        :param max_gamma: Maximum circulation used to normalise the plotted mode shape.
        :return: Continuous-time eigenvalues of the system A matrix, ``(n_states, )`` or ``(n_states, 2)`` if ``to_components=True``.
        """

        evals_d, evecs = jnp.linalg.eig(self.sys.a)
        evals = jnp.log(evals_d) / self.dt  # convert to continuous time

        # order from most to least damped and truncate
        omega_damped = jnp.abs(evals.imag)
        damping = -evals.real / jnp.abs(evals)
        omega_natural = omega_damped / jnp.sqrt(1.0 - damping**2)

        freq_natural_hz = omega_natural / (2.0 * jnp.pi)

        match sort:
            case "frequency":
                idx = omega_natural.argsort()
            case "damping":
                idx = damping.argsort()

        # push conjugate partners past the truncation point (indexed by original position, then re-sorted so it
        # aligns with `idx`). Stable-argsort preserves the primary sort within each group.
        if remove_complex_conjugate:
            partner = conjugate_partner_mask(
                freq_hz=freq_natural_hz, damping=damping, tiebreaker=evals.real
            )
            idx = idx[jnp.argsort(partner[idx], stable=True)]

        # fraction of eigenvector energy in the structural states — used to reject
        # aero-only modes (e.g. wake convection)
        q_slice = self.state_slices["q"].slices
        q_dot_slice = self.state_slices["q_dot"].slices
        evec_sq = jnp.abs(evecs) ** 2
        struct_content = (
            evec_sq[q_slice].sum(axis=0) + evec_sq[q_dot_slice].sum(axis=0)
        ) / evec_sq.sum(axis=0)

        # push modes outside the requested natural-frequency / damping window to the back so truncation to
        # n_modes keeps only the in-range ones.
        in_range = (
            (freq_natural_hz[idx] >= freq_range[0])
            & (freq_natural_hz[idx] <= freq_range[1])
            & (damping[idx] >= damp_range[0])
            & (damping[idx] <= damp_range[1])
            & (struct_content[idx] >= min_struct_content)
        )
        idx = idx[jnp.argsort(~in_range, stable=True)]

        if n_modes is not None:
            idx = idx[:n_modes]

        freq_damped_ordered = omega_damped[idx] / (2.0 * jnp.pi)
        freq_natural_ordered = omega_natural[idx] / (2.0 * jnp.pi)
        damping_ordered = damping[idx]

        # write to console
        if n_modes is not None:
            print_table_line(inner_width=71)
            jax_print(
                "| Mode | Damped Frequency [Hz] | Natural Frequency [Hz] | Damping Ratio |",
                verbose_level="normal",
            )
            print_table_line(inner_width=71)
            for i_mode in range(n_modes):
                jax_print(
                    "| {mode:>4d} | {freq_damped:>21.3f} | {freq_natural:>22.3f} | {damp:>13.6f} |",
                    mode=i_mode + 1,
                    freq_damped=freq_damped_ordered[i_mode],
                    freq_natural=freq_natural_ordered[i_mode],
                    damp=damping_ordered[i_mode],
                    verbose_level="normal",
                )
            print_table_line(inner_width=71)

        if plot_eigvals:
            _, ax = plt.subplots()
            ax.scatter(
                evals.real,
                evals.imag,
            )
            ax.set_xlim(*plot_xlim)
            ax.set_ylim(*plot_ylim)
            ax.set_xlabel("Re(eig) [1/s]")
            ax.set_ylabel("Im(eig) [1/s]")
            ax.set_title("Eigenvalues")
            plt.show()

        if n_plot_vtk > 0:
            evecs_ordered = evecs[:, idx[:n_plot_vtk]].T  # (m, n_states)

            q_mode = evecs_ordered[
                :, self.state_slices["q"].slices
            ]  # (m, n_free_dof | n_modes)
            if self.structure.modal_states:
                q_mode = self.structure.modal_to_nodal(q_mode)  # (m, n_free_dof)
            q_full = (
                jnp.zeros((n_plot_vtk, self.n_nodes * 6), dtype=complex)
                .at[:, self.free_dofs]
                .set(q_mode)
            )

            def _extract_array_list(name: str) -> ArrayList | None:
                component = self.state_slices[name]
                if not component.enabled:
                    return None
                return ArrayList(
                    [
                        evecs_ordered[:, s].reshape((n_plot_vtk, *shape))
                        for s, shape in zip(component.slices, component.shapes)
                    ]
                )

            gamma_b_full = _extract_array_list("gamma_b")
            gamma_w_full = _extract_array_list("gamma_w")
            zeta_w_full = _extract_array_list("zeta_w")

            plot_modes_vtu(
                reference=self.reference,
                directory=vtu_directory,
                q_full=q_full.reshape(n_plot_vtk, self.n_nodes, 6),
                freqs=freq_damped_ordered,
                dampings=damping_ordered,
                gamma_b_full=gamma_b_full,
                gamma_w_full=gamma_w_full,
                zeta_w_full=zeta_w_full,
                uvlm=self.aero.case,
                n_phase=n_phase,
                n_interp=n_interp,
                max_disp=max_disp,
                max_ang=max_ang,
                max_gamma=max_gamma,
            )

        return evals[idx]

    def modal_rescaled(
        self,
        velocity: float | Array,
        density: float | Array,
        chord: float | Array,
        n_modes: int | None = None,
        freq_range: tuple[float | Array, float | Array] = (0.0, jnp.inf),
        damp_range: tuple[float | Array, float | Array] = (-jnp.inf, jnp.inf),
        min_struct_content: float | Array = 0.0,
        remove_complex_conjugate: bool = True,
        sort: Literal["frequency", "damping"] = "frequency",
    ) -> Array:
        r"""
        Compute eigenvalues of the rescaled linear system at a new velocity, density, and chord length without
        re-linearising.
        :param velocity: Freestream velocity magnitude(s) at the new condition(s), scalar or ``(*n_points,)``.
        :param density: Flow density(s) at the new condition(s), scalar or broadcastable with ``velocity``.
        :param chord: Reference chord length(s) at the new condition(s), scalar or broadcastable with ``velocity``.
        :param n_modes: Number of modes to keep. If ``None``, all eigenvalues are returned.
        :param freq_range: (min, max) natural frequency window in Hz.
        :param damp_range: (min, max) damping-ratio window.
        :param min_struct_content: Minimum structural eigenvector energy fraction in ``[0, 1]``.
        :param remove_complex_conjugate: Drop one partner from each conjugate pair.
        :param sort: Sort eigenvalues by ``"frequency"`` or ``"damping"``.
        :return: Continuous-time eigenvalues of the rescaled system(s), ``(n_out,)`` for scalar
            inputs or ``(*n_points, n_out)`` for batched inputs.
        """
        velocity, density, chord = jnp.broadcast_arrays(
            jnp.asarray(velocity, dtype=float),
            jnp.asarray(density, dtype=float),
            jnp.asarray(chord, dtype=float),
        )
        batch_shape = velocity.shape

        def _single(velocity_: Array, density_: Array, chord_: Array) -> Array:
            r"""
            Rescale and diagonalise the linear system at a single (velocity, density, chord) point.
            """
            # reference conditions
            u_ref = self._case.aero.flowfield.u_inf_mag
            rho_ref = self._case.aero.flowfield.rho
            m_chord = self.reference.aero.gamma_b[0].shape[0]

            dt_new = chord_ / (m_chord * velocity_)

            # discretise structural system at new dt
            struct_cont = self.structure.linearise_continuous()
            n_struct = struct_cont.a.shape[0]
            eye_s = jnp.eye(n_struct)

            mat_inv_new = jnp.linalg.inv(eye_s - struct_cont.a * 0.5 * dt_new)
            a_struct_new = mat_inv_new @ (eye_s + struct_cont.a * 0.5 * dt_new)
            b_struct_new = mat_inv_new @ (struct_cont.b * dt_new)

            f_ext_slice = self.structure.input_slices["f_ext"].slices
            b_d_f_ref = self.structure.sys.b[:, f_ext_slice]
            b_d_f_new = b_struct_new[:, f_ext_slice]

            # rescaled A matrix
            a_ref = self.sys.a

            # structural and aero index ranges in the coupled state vector
            q_start = self.state_slices["q"].slices.start
            q_dot_end = self.state_slices["q_dot"].slices.stop
            struct_slice = slice(q_start, q_dot_end)

            # reference structural discrete-time A_d
            a_struct_ref = self.structure.sys.a

            # aero-induced contribution in the structural rows
            struct_rows = a_ref[struct_slice, :]
            delta = struct_rows.at[:, struct_slice].add(-a_struct_ref)

            # extract force Jacobian
            force_jac = jnp.linalg.pinv(b_d_f_ref) @ delta

            # per-column force scaling
            n_total = a_ref.shape[0]
            q_state_slice = self.state_slices["q"].slices
            base_force_scale = (density_ * velocity_) / (rho_ref * u_ref)
            force_col_scale = jnp.ones(n_total) * base_force_scale
            force_col_scale = force_col_scale.at[q_state_slice].set(
                (density_ * velocity_**2) / (rho_ref * u_ref**2)
            )
            delta_new = b_d_f_new @ (force_jac * force_col_scale[None, :])

            a_new = a_ref.at[struct_slice, :].set(delta_new)
            a_new = a_new.at[struct_slice, struct_slice].add(a_struct_new)

            vel_ratio = velocity_ / u_ref
            a_new = a_new.at[:q_start, q_state_slice].multiply(vel_ratio)

            # eigenvalue computation
            evals_d, evecs = jnp.linalg.eig(a_new)
            evals = jnp.log(evals_d) / dt_new

            omega_damped = jnp.abs(evals.imag)
            damping = -evals.real / jnp.abs(evals)
            omega_natural = omega_damped / jnp.sqrt(1.0 - damping**2)
            freq_natural_hz = omega_natural / (2.0 * jnp.pi)

            match sort:
                case "frequency":
                    idx = omega_natural.argsort()
                case "damping":
                    idx = damping.argsort()

            if remove_complex_conjugate:
                partner = conjugate_partner_mask(
                    freq_hz=freq_natural_hz, damping=damping, tiebreaker=evals.real
                )
                idx = idx[jnp.argsort(partner[idx], stable=True)]

            q_slice = self.state_slices["q"].slices
            q_dot_slice = self.state_slices["q_dot"].slices
            evec_sq = jnp.abs(evecs) ** 2
            struct_content = (
                evec_sq[q_slice].sum(axis=0) + evec_sq[q_dot_slice].sum(axis=0)
            ) / evec_sq.sum(axis=0)

            in_range = (
                (freq_natural_hz[idx] >= freq_range[0])
                & (freq_natural_hz[idx] <= freq_range[1])
                & (damping[idx] >= damp_range[0])
                & (damping[idx] <= damp_range[1])
                & (struct_content[idx] >= min_struct_content)
            )
            idx = idx[jnp.argsort(~in_range, stable=True)]

            if n_modes is not None:
                idx = idx[:n_modes]

            return evals[idx]

        if batch_shape == ():
            return _single(velocity, density, chord)

        n_points = velocity.size

        # map scaling across multiple points
        evals_flat = vmap(_single)(
            velocity.reshape(n_points),
            density.reshape(n_points),
            chord.reshape(n_points),
        )  # (n_points, n_out)
        return evals_flat.reshape(*batch_shape, evals_flat.shape[-1])

    # noinspection PyMethodOverriding
    def run(
        self,
        u: AeroelasticInputUnflattened,
        x0: AeroelasticStateUnflattened | None = None,
    ) -> AeroelasticLinearResult:
        raise NotImplementedError

    def frf(
        self,
        omega: Array,
        flowfield: FrequencyFlowField | None,
    ) -> AeroelasticOutputUnflattened:
        r"""
        Compute the frequency response function for a gust input.
        :param omega: Frequencies in rad/s, ``(n_freq,)``.
        :param flowfield: Frequency-domain turbulence spectrum. If ``None``,
            the raw transfer function is returned.
        :return: Gust FRF with ``q`` and ``q_dot`` fields, each ``(n_freq, n_dof)``.
        """
        h = self.frf_base(omega, flowfield)
        n_out = self.n_beam_output_dof
        return AeroelasticOutputUnflattened(q=h[:, :n_out], q_dot=h[:, n_out:])

    def frf_base(
        self,
        omega: Array,
        flowfield: FrequencyFlowField | None,
    ) -> Array:
        r"""
        Compute the gust FRF as a flat vector.
        :param omega: Frequencies in rad/s, ``(n_freq,)``.
        :param flowfield: Frequency-domain turbulence spectrum. If ``None``,
            the raw transfer function is returned.
        :return: Complex FRF, ``(n_freq, n_outputs)``.
        """
        from flapjax.coupled.linear_gradients.frf import gust_penetration_vector

        u_inf = (
            flowfield.u_inf
            if flowfield is not None
            else self._case.aero.flowfield.u_inf_mag
        )

        zeta_b0 = self._reference.aero.zeta_b
        vertex_x = jnp.concatenate([z[..., 0].ravel() for z in zeta_b0])
        g = gust_penetration_vector(omega=omega, vertex_x=vertex_x, u_inf=u_inf)

        a = self.sys.a
        c = self.sys.c
        n_states = a.shape[0]

        linear_upwash = LinearCoupled(
            case=self._case,
            reference=self._reference,
            batch_size=False,
            n_struct_modes=None,
            bound_upwash=True,
            skip_checks=True,
        )
        nu_b_zero = jnp.zeros(self._reference.aero.zeta_b.size)

        def _step_nu_b(nu_b_vec: Array) -> Array:
            state_np1, _ = linear_upwash.step(nu_b_vec=nu_b_vec)
            return linear_upwash.pack_state_vector(state_np1)

        def _b_g_single(g_k: Array) -> Array:
            _, bg_re = jax.jvp(_step_nu_b, (nu_b_zero,), (g_k.real,))
            _, bg_im = jax.jvp(_step_nu_b, (nu_b_zero,), (g_k.imag,))
            return bg_re + 1j * bg_im

        b_g = jax.vmap(_b_g_single)(g)

        z = jnp.exp(1j * omega * self.dt)
        eye = jnp.eye(n_states)

        def _solve_single(z_k: Array, bg_k: Array) -> Array:
            return z_k * c @ jnp.linalg.solve(z_k * eye - a, bg_k)

        h = jax.vmap(_solve_single)(z, b_g)

        if flowfield is not None:
            # scale with flowfield PSD if available
            h *= jnp.sqrt(flowfield.psd(omega))[:, None]

        return h
