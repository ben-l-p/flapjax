from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import jax.numpy as jnp
from jax import Array, vmap

from flapjax.aero.aic import compute_v_ind
from flapjax.aero.data_structures import AeroCase
from flapjax.aero.flowfields import FlowField
from flapjax.aero.gradients.data_structures import AeroFullStates
from flapjax.aero.linear.data_structures import (
    AeroInputUnflattened,
    AeroLinearResult,
    AeroOutputUnflattened,
    AeroStateUnflattened,
)
from flapjax.aero.utils import (
    KernelFunction,
    biot_savart_cutoff,
    compute_nc,
    compute_steady_forcing,
)
from flapjax.algebra.array_utils import (
    ArrayList,
    construct_named_block_jacobian,
    split_to_vertex,
)
from flapjax.algebra.base import ADMode, jacrev_custom
from flapjax.utils.linear import (
    LinearComponent,
    LinearModel,
    LinearSystem,
    SliceEntry,
)
from flapjax.utils.print_utils import warn

if TYPE_CHECKING:
    from flapjax.aero.uvlm import UVLM

type LinearWakeType = Literal["frozen", "prescribed", "free"]


@dataclass
class LinearInputProjection:
    r"""
    Projection of grid coordinates and velocities from a lower-dimensional input space (e.g. beam DOFs) to
    ``(zeta_b_np1_vec, zeta_b_dot_np1_vec)``.
    :param arg_names: Names of the projected input dofs (used as column labels).
    :param arg_sizes: Sizes of each projected input dof block.
    :param arg_refs: Reference values for the projected inputs.
    :param to_zeta: Callable ``(**{arg_name: vec}) -> (zeta_b_vec, zeta_b_dot_vec)`` returning their total values.
    """

    arg_names: tuple[str, ...]
    arg_sizes: tuple[int, ...]
    arg_refs: tuple[Array, ...]
    to_zeta: Callable[..., tuple[Array, Array]]


@dataclass
class LinearOutputProjection:
    r"""
    Projection of forces from the full aerodynamic grid to a single output space.
    :param name: Name of the projected output block (used as row label).
    :param size: Size of the projected output.
    :param to_output: Callable ``(f_steady_vec, f_unsteady_vec | None) -> projected_vec``.
    """

    name: str
    size: int
    to_output: Callable[[Array, Array | None], Array]


class LinearUVLM(
    LinearModel[
        AeroCase,
        AeroInputUnflattened,
        AeroStateUnflattened,
        AeroOutputUnflattened,
        AeroLinearResult,
    ]
):
    r"""
    Class to represent a linearised UVLM aerodynamic system about a reference state.
    """

    def __init__(
        self,
        case: UVLM,
        reference: AeroCase,
        wake_type: LinearWakeType = "frozen",
        bound_upwash: bool = True,
        wake_upwash: bool = True,
        unsteady_force: bool = True,
        *,
        skip_checks: bool = False,
        skip_linearisation: bool = False,
    ):
        r"""
        Initialise linear UVLM system about a reference state.
        :param case: UVLM case object to linearise.
        :param reference: StaticAero representing the reference state for linearisation.
        :param wake_type: Instance of LinearWakeType enum to specify wake treatment.
        :param bound_upwash: If true, include bound surface upwash velocities as inputs.
        :param wake_upwash: If true, include wake surface upwash velocities as inputs.
        :param unsteady_force: If true, include unsteady force.
        """
        # options
        self.prescribed_wake, self.free_wake = {
            "frozen": (False, False),
            "prescribed": (True, False),
            "free": (True, True),
        }[wake_type]
        self.unsteady_force: bool = unsteady_force
        self.bound_upwash: bool = bound_upwash
        self.wake_upwash: bool = wake_upwash

        # time info
        super().__init__(reference=reference, dt=case.dt)

        # check that the reference state is steady
        # whilst linearisation can be performed about unsteady states, the current implementation omits some terms
        # required for this, however, cannot see a practical use case for such a model. Warn the user if the reference
        # state appears unsteady.
        if (
            not skip_checks
            and max([jnp.abs(zbd).max() for zbd in reference.zeta_b_dot]) > 1e-6
        ):
            warn(
                "Reference bound surface velocities are non-zero. Ensure that the reference state is steady for linearisation."
            )

        if (
            not skip_checks
            and max([jnp.abs(gbd).max() for gbd in reference.gamma_b_dot]) > 1e-6
        ):
            warn(
                "Reference bound circulation time derivative is non-zero. Ensure that the reference state is steady for linearisation."
            )

        # kernels
        self.kernels_b: Sequence[KernelFunction] = reference.n_surf * [
            biot_savart_cutoff
        ]
        self.kernels_w: Sequence[KernelFunction] = reference.n_surf * [
            biot_savart_cutoff
        ]

        # wake propagation deltas
        self.case: UVLM = case

        # linear system
        if not skip_linearisation:
            self.sys = self.linearise()

    @property
    def reference(self) -> AeroCase:
        return self._reference

    def extract_reference_inputs(
        self,
    ) -> dict[str, Array | ArrayList | None]:
        return {
            "zeta_b": self.reference.zeta_b,
            "zeta_b_dot": self.reference.zeta_b_dot,
            "nu_b": ArrayList.zeros_like(self.reference.zeta_b)
            if self.bound_upwash
            else None,
            "nu_w": ArrayList.zeros_like(self.reference.zeta_w)
            if self.wake_upwash
            else None,
        }

    def extract_reference_states(
        self,
    ) -> dict[str, Array | ArrayList | None]:
        return {
            "gamma_b": self.reference.gamma_b,
            "gamma_w": self.reference.gamma_w,
            "gamma_b_nm1": self.reference.gamma_b if self.unsteady_force else None,
            "zeta_w": self.reference.zeta_w if self.prescribed_wake else None,
            "zeta_b": self.reference.zeta_b if self.prescribed_wake else None,
        }

    def extract_reference_outputs(
        self,
    ) -> dict[str, Array | ArrayList | None]:
        return {
            "f_steady": self.reference.f_steady,
            "f_unsteady": self.reference.f_unsteady if self.unsteady_force else None,
        }

    @property
    def input_object(self) -> type[AeroInputUnflattened]:
        return AeroInputUnflattened

    @property
    def state_object(
        self,
    ) -> type[AeroStateUnflattened]:
        return AeroStateUnflattened

    @property
    def output_object(
        self,
    ) -> type[AeroOutputUnflattened]:
        return AeroOutputUnflattened

    def _make_input_slices(
        self, reference: AeroCase
    ) -> tuple[dict[str, LinearComponent], int]:
        r"""
        Create input slices for the input vector.
        :return: InputSlices instance and total number of input elements.
        """
        slice_entries = (
            SliceEntry("zeta_b", True, reference.zeta_b.shape),
            SliceEntry("zeta_b_dot", True, reference.zeta_b.shape),
            SliceEntry(
                "nu_b",
                *(
                    (True, reference.zeta_b.shape)
                    if self.bound_upwash
                    else (False, None)
                ),
            ),
            SliceEntry(
                "nu_w",
                *(
                    (True, reference.zeta_w.shape)
                    if self.wake_upwash
                    else (False, None)
                ),
            ),
        )
        return self._make_slices(slice_entries)

    def _make_state_slices(
        self, reference: AeroCase
    ) -> tuple[dict[str, LinearComponent], int]:
        r"""
        Create state slices for the state vector.
        :return: StateSlices instance and total number of state elements.
        """
        slice_entries = (
            SliceEntry("gamma_b", True, reference.gamma_b.shape),
            SliceEntry("gamma_w", True, reference.gamma_w.shape),
            SliceEntry(
                "gamma_b_nm1",
                *(
                    (True, reference.gamma_b.shape)
                    if self.unsteady_force
                    else (False, None)
                ),
            ),
            SliceEntry(
                "zeta_w",
                *(
                    (True, reference.zeta_w.shape)
                    if self.prescribed_wake
                    else (False, None)
                ),
            ),
            SliceEntry(
                "zeta_b",
                *(
                    (True, reference.zeta_b.shape)
                    if self.prescribed_wake
                    else (False, None)
                ),
            ),
        )
        return self._make_slices(slice_entries)

    def _make_output_slices(
        self, reference: AeroCase
    ) -> tuple[dict[str, LinearComponent], int]:
        r"""
        Create output slices for the output vector.
        :return: OutputSlices instance and total number of output elements.
        """
        slice_entries = (
            SliceEntry("f_steady", True, reference.zeta_b.shape),
            SliceEntry(
                "f_unsteady",
                *(
                    (True, reference.zeta_b.shape)
                    if self.unsteady_force
                    else (False, None)
                ),
            ),
        )
        return self._make_slices(slice_entries)

    def step_vec(self, x_vec: Array, u_vec: Array) -> tuple[Array, Array]:
        r"""
        Combined state/output step in vector form, operating on total (reference + perturbation) quantities and
        returning perturbations relative to the reference.
        :param x_vec: State vector, ``(n_states, )``
        :param u_vec: Input vector, ``(n_inputs, )``
        :return: Tuple of (state perturbation vector, output perturbation vector).
        """

        u_np1 = self._unpack_input_vector(u_vec)
        x_n = self.unpack_state_vector(x_vec)

        assert isinstance(u_np1, AeroInputUnflattened) and isinstance(
            x_n, AeroStateUnflattened
        ), (
            "Unpacked input and state must be of type AeroInputUnflattened and AeroStateUnflattened."
        )

        x_np1, y_n = self.step(u_np1=u_np1, x_n=x_n)

        return self.pack_state_vector(x_np1), self.pack_output_vector(y_n)

    def step(
        self,
        u_np1: AeroInputUnflattened,
        x_n: AeroStateUnflattened,
    ) -> tuple[AeroStateUnflattened, AeroOutputUnflattened]:
        r"""
        From total inputs `u` at timestep n+1 and states `x` at timestep n, compute the states at timestep n+1 and
        outputs at timestep n.
        """
        ref = self.reference

        zeta_b_np1: ArrayList = u_np1.zeta_b
        zeta_dot_b_np1: ArrayList = u_np1.zeta_b_dot
        gamma_b_n: ArrayList = x_n.gamma_b
        gamma_w_n: ArrayList = x_n.gamma_w

        if self.unsteady_force:
            assert x_n.gamma_b_nm1 is not None, "gamma_b_nm1 is None"
            gamma_b_dot_n: ArrayList = (gamma_b_n - x_n.gamma_b_nm1) / self.dt
        else:
            gamma_b_dot_n = ref.gamma_b_dot

        if self.prescribed_wake:
            assert x_n.zeta_b is not None, "zeta_b is None"
            assert x_n.zeta_w is not None, "zeta_w is None"
            zeta_b_n: ArrayList = x_n.zeta_b
            zeta_w_n: ArrayList = x_n.zeta_w
        else:
            zeta_b_n = ref.zeta_b
            zeta_w_n = ref.zeta_w

        q_n = AeroFullStates(
            gamma_b=gamma_b_n,
            gamma_w=gamma_w_n,
            gamma_b_dot=ref.gamma_b_dot,
            zeta_w=zeta_w_n,
        )
        (
            _,
            _,
            gamma_b_np1,
            gamma_w_np1,
            _,
            _,
            zeta_w_np1,
            *_,
        ) = self.case.base_solve_from_grid(
            q_nm1=q_n,
            t_n=ref.t,
            zeta_b_n=zeta_b_np1,
            zeta_b_nm1=zeta_b_n,
            zeta_b_dot_n=zeta_dot_b_np1,
            static=False,
            horseshoe=False,
            linearise_variable_wake=True,
            nu_b=u_np1.nu_b,
            nu_w=u_np1.nu_w,
        )

        # the forcing needs to be computed seperately to find its dependence on the current states
        rho = ref.flowfield.rho

        def v_out_func(x_target: Array) -> Array:
            return ref.flowfield.vmap_call(x=x_target, t=ref.t) + compute_v_ind(
                cs=x_target,
                zetas=ArrayList([*zeta_b_np1, *zeta_w_n]),
                gammas=ArrayList([*gamma_b_n, *gamma_w_n]),
                kernels=[*self.kernels_b, *self.kernels_w],
                batch_size=self.case.batch_size,
                mirror_normal=self.case.mirror_normal,
                mirror_point=self.case.mirror_point,
            )

        f_steady_n = compute_steady_forcing(
            zeta_b=zeta_b_np1,
            zeta_dot_b=zeta_dot_b_np1,
            gamma_b=gamma_b_n,
            gamma_w=gamma_w_n,
            rho=rho,
            v_func=v_out_func,
            v_inputs=u_np1.nu_b if self.bound_upwash else None,
            mirror_point=self.case.mirror_point,
            mirror_normal=self.case.mirror_normal,
            mirror_edge_low=self.case.mirror_edge_low,
            mirror_edge_high=self.case.mirror_edge_high,
        )

        normals = compute_nc(zetas=zeta_b_np1)
        if self.unsteady_force:
            f_unsteady_n = ArrayList(
                [
                    split_to_vertex(
                        rho * gamma_b_dot_n[i][..., None] * normals[i], (0, 1)
                    )
                    for i in range(ref.n_surf)
                ]
            )
        else:
            f_unsteady_n = None

        x_np1 = AeroStateUnflattened(
            gamma_b=gamma_b_np1,
            gamma_w=gamma_w_np1,
            gamma_b_nm1=gamma_b_n if self.unsteady_force else None,
            zeta_w=zeta_w_np1 if self.prescribed_wake else None,
            zeta_b=zeta_b_np1 if self.prescribed_wake else None,
        )
        y_n = AeroOutputUnflattened(f_steady=f_steady_n, f_unsteady=f_unsteady_n)

        return x_np1, y_n

    def _step_from_vecs(
        self,
        gamma_b_n_vec: Array | None = None,
        gamma_w_n_vec: Array | None = None,
        gamma_b_nm1_vec: Array | None = None,
        zeta_w_n_vec: Array | None = None,
        zeta_b_n_vec: Array | None = None,
        zeta_b_np1_vec: Array | None = None,
        zeta_b_dot_np1_vec: Array | None = None,
        nu_b_np1_vec: Array | None = None,
        nu_w_np1_vec: Array | None = None,
    ) -> tuple[AeroStateUnflattened, AeroOutputUnflattened]:
        r"""
        Unravel vectors into their shaped forms and step the system.
        """
        ref = self.reference

        def _unravel(vec: Array | None, template: ArrayList) -> ArrayList:
            return (
                ArrayList.from_vector(vect=vec, shapes=template.shape)
                if vec is not None
                else template
            )

        gamma_b = _unravel(gamma_b_n_vec, ref.gamma_b)
        gamma_w = _unravel(gamma_w_n_vec, ref.gamma_w)
        gamma_b_nm1 = (
            _unravel(gamma_b_nm1_vec, ref.gamma_b) if self.unsteady_force else None
        )
        zeta_b_np1 = _unravel(zeta_b_np1_vec, ref.zeta_b)
        zeta_b_dot_np1 = _unravel(zeta_b_dot_np1_vec, ref.zeta_b_dot)
        zeta_w_n = _unravel(zeta_w_n_vec, ref.zeta_w) if self.prescribed_wake else None
        zeta_b_n = _unravel(zeta_b_n_vec, ref.zeta_b) if self.prescribed_wake else None
        nu_b = (
            _unravel(nu_b_np1_vec, ref.zeta_b)
            if self.bound_upwash and nu_b_np1_vec is not None
            else (ArrayList.zeros_like(ref.zeta_b) if self.bound_upwash else None)
        )
        nu_w = (
            _unravel(nu_w_np1_vec, ref.zeta_w)
            if self.wake_upwash and nu_w_np1_vec is not None
            else (ArrayList.zeros_like(ref.zeta_w) if self.wake_upwash else None)
        )

        u_np1 = AeroInputUnflattened(
            zeta_b=zeta_b_np1,
            zeta_b_dot=zeta_b_dot_np1,
            nu_b=nu_b,
            nu_w=nu_w,
        )
        x_n = AeroStateUnflattened(
            gamma_b=gamma_b,
            gamma_w=gamma_w,
            gamma_b_nm1=gamma_b_nm1,
            zeta_w=zeta_w_n,
            zeta_b=zeta_b_n,
        )
        return self.step(u_np1=u_np1, x_n=x_n)

    def gamma_b_step(self, **kwargs: Any) -> Array:
        x_np1, _ = self._step_from_vecs(**kwargs)
        return x_np1.gamma_b.ravel()

    def wake_prop_step(self, **kwargs: Any) -> tuple[Array | None, Array]:
        x_np1, _ = self._step_from_vecs(**kwargs)
        return (
            x_np1.zeta_w.ravel() if x_np1.zeta_w is not None else None,
            x_np1.gamma_w.ravel(),
        )

    def f_steady_step(self, **kwargs: Any) -> Array:
        _, y_n = self._step_from_vecs(**kwargs)
        return y_n.f_steady.ravel()

    def f_unsteady_step(self, **kwargs: Any) -> Array:
        _, y_n = self._step_from_vecs(**kwargs)
        assert y_n.f_unsteady is not None, "unsteady_force must be enabled"
        return y_n.f_unsteady.ravel()

    def compute_jacobians(
        self,
        input_projection: LinearInputProjection | None = None,
        output_projection: LinearOutputProjection | None = None,
        residual_names: Sequence[str] | None = None,
    ) -> dict[str, tuple[Callable[..., Any], dict[str, Any], Sequence[str]]]:
        r"""
        Build the Jacobians for the linear system.
        :param input_projection: If set, projects the inputs onto a different space.
        :param output_projection: If set, projects the forcing outputs onto a different space.
        :param residual_names: If set, restrict the returned residuals to this subset to avoid redundant computation.
        :return: Mapping of residual name to Jacobian(s).
        """
        ref = self.reference

        # bound circulation
        gamma_b_args: dict[str, Any] = {
            "gamma_b_n_vec": ref.gamma_b.ravel(),
            "gamma_w_n_vec": ref.gamma_w.ravel(),
            "zeta_b_np1_vec": ref.zeta_b.ravel(),
            "zeta_b_dot_np1_vec": ref.zeta_b_dot.ravel(),
        }
        gamma_b_diff = [
            "gamma_b_n_vec",
            "gamma_w_n_vec",
            "zeta_b_np1_vec",
            "zeta_b_dot_np1_vec",
        ]
        if self.prescribed_wake:
            gamma_b_args["zeta_w_n_vec"] = ref.zeta_w.ravel()
            gamma_b_args["zeta_b_n_vec"] = ref.zeta_b.ravel()
            gamma_b_diff.extend(["zeta_w_n_vec", "zeta_b_n_vec"])
        if self.bound_upwash:
            gamma_b_args["nu_b_np1_vec"] = jnp.zeros(ref.zeta_b.size)
            gamma_b_diff.append("nu_b_np1_vec")
        if self.wake_upwash:
            gamma_b_args["nu_w_np1_vec"] = jnp.zeros(ref.zeta_w.size)
            gamma_b_diff.append("nu_w_np1_vec")

        # wake propagation (gamma_w and optionally zeta_w)
        wake_args: dict[str, Any] = {
            "gamma_b_n_vec": ref.gamma_b.ravel(),
            "gamma_w_n_vec": ref.gamma_w.ravel(),
        }
        gamma_w_diff = ["gamma_b_n_vec", "gamma_w_n_vec"]
        zeta_w_diff: list[str] = []
        if self.prescribed_wake:
            wake_args["zeta_w_n_vec"] = ref.zeta_w.ravel()
            wake_args["zeta_b_np1_vec"] = ref.zeta_b.ravel()
            zeta_w_diff.extend(["zeta_w_n_vec", "zeta_b_np1_vec"])
        if self.wake_upwash:
            wake_args["nu_w_np1_vec"] = jnp.zeros(ref.zeta_w.size)
            zeta_w_diff.append("nu_w_np1_vec")
        if self.free_wake:
            zeta_w_diff.extend(["gamma_b_n_vec", "gamma_w_n_vec"])

        # steady forcing
        f_steady_args: dict[str, Any] = {
            "gamma_b_n_vec": ref.gamma_b.ravel(),
            "gamma_w_n_vec": ref.gamma_w.ravel(),
            "zeta_b_np1_vec": ref.zeta_b.ravel(),
            "zeta_b_dot_np1_vec": ref.zeta_b_dot.ravel(),
        }
        f_steady_diff = [
            "gamma_b_n_vec",
            "gamma_w_n_vec",
            "zeta_b_np1_vec",
            "zeta_b_dot_np1_vec",
        ]
        if self.prescribed_wake:
            f_steady_args["zeta_w_n_vec"] = ref.zeta_w.ravel()
            f_steady_diff.append("zeta_w_n_vec")
        if self.bound_upwash:
            f_steady_args["nu_b_np1_vec"] = jnp.zeros(ref.zeta_b.size)
            f_steady_diff.append("nu_b_np1_vec")

        residuals: dict[
            str, tuple[Callable[..., Any], dict[str, Any], Sequence[str]]
        ] = {
            "gamma_b": (self.gamma_b_step, gamma_b_args, gamma_b_diff),
            "gamma_w": (
                lambda **kw: self.wake_prop_step(**kw)[1],
                wake_args,
                gamma_w_diff,
            ),
        }
        if self.prescribed_wake:
            residuals["zeta_w"] = (
                lambda **kw: self.wake_prop_step(**kw)[0],
                wake_args,
                zeta_w_diff,
            )
        if self.unsteady_force:
            residuals["gamma_b_nm1"] = (
                lambda **kw: kw["gamma_b_n_vec"],
                {"gamma_b_n_vec": ref.gamma_b.ravel()},
                ["gamma_b_n_vec"],
            )
        if self.prescribed_wake:
            # zeta_b state at n+1 == zeta_b input at n+1
            residuals["zeta_b"] = (
                lambda **kw: kw["zeta_b_np1_vec"],
                {"zeta_b_np1_vec": ref.zeta_b.ravel()},
                ["zeta_b_np1_vec"],
            )
        residuals["f_steady"] = (self.f_steady_step, f_steady_args, f_steady_diff)
        if self.unsteady_force:
            f_unsteady_args: dict[str, Any] = {
                "gamma_b_n_vec": ref.gamma_b.ravel(),
                "gamma_b_nm1_vec": ref.gamma_b.ravel(),
                "zeta_b_np1_vec": ref.zeta_b.ravel(),
            }
            residuals["f_unsteady"] = (
                self.f_unsteady_step,
                f_unsteady_args,
                ["gamma_b_n_vec", "gamma_b_nm1_vec", "zeta_b_np1_vec"],
            )

        # apply I/O projections if provided
        if input_projection is not None:
            residuals = self._apply_input_projection(residuals, input_projection)
        if output_projection is not None:
            residuals = self._apply_output_projection(residuals, output_projection)

        if residual_names is not None:
            residuals = {k: v for k, v in residuals.items() if k in residual_names}

        return residuals

    @staticmethod
    def _apply_input_projection(
        residuals: dict[str, tuple[Callable[..., Any], dict[str, Any], Sequence[str]]],
        proj: LinearInputProjection,
    ) -> dict[str, tuple[Callable[..., Any], dict[str, Any], Sequence[str]]]:
        r"""
        Project residual functions onto a given input space.
        """
        zeta_args_replaced = {"zeta_b_np1_vec", "zeta_b_dot_np1_vec", "zeta_b_n_vec"}
        proj_names = tuple(proj.arg_names)

        new_residuals: dict[
            str, tuple[Callable[..., Any], dict[str, Any], Sequence[str]]
        ] = {}
        for name, (func, args, diff_arg_names) in residuals.items():
            if not any(k in args for k in zeta_args_replaced):
                new_residuals[name] = (func, args, diff_arg_names)
                continue

            new_args = {k: v for k, v in args.items() if k not in zeta_args_replaced}
            for aname, aref in zip(proj_names, proj.arg_refs, strict=True):
                new_args[aname] = aref

            new_diff = [k for k in diff_arg_names if k not in zeta_args_replaced]
            for aname in proj_names:
                if aname not in new_diff:
                    new_diff.append(aname)

            def _wrap(
                inner: Callable[..., Any],
                needs_zeta_b_np1: bool = "zeta_b_np1_vec" in args,
                needs_zeta_b_dot_np1: bool = "zeta_b_dot_np1_vec" in args,
                needs_zeta_b_n: bool = "zeta_b_n_vec" in args,
            ) -> Callable[..., Any]:
                def wrapped(**kwargs: Any) -> Any:
                    proj_kwargs = {n: kwargs.pop(n) for n in proj_names if n in kwargs}
                    zb, zbd = proj.to_zeta(**proj_kwargs)
                    if needs_zeta_b_np1:
                        kwargs["zeta_b_np1_vec"] = zb
                    if needs_zeta_b_dot_np1:
                        kwargs["zeta_b_dot_np1_vec"] = zbd
                    if needs_zeta_b_n:
                        # previous-step bound geometry uses the same projection output
                        kwargs["zeta_b_n_vec"] = zb
                    return inner(**kwargs)

                return wrapped

            new_residuals[name] = (_wrap(func), new_args, new_diff)

        return new_residuals

    @staticmethod
    def _apply_output_projection(
        residuals: dict[str, tuple[Callable[..., Any], dict[str, Any], Sequence[str]]],
        proj: LinearOutputProjection,
    ) -> dict[str, tuple[Callable[..., Any], dict[str, Any], Sequence[str]]]:
        r"""
        Project forcing outputs into a given output space.
        """
        f_steady_entry = residuals.get("f_steady")
        f_unsteady_entry = residuals.get("f_unsteady")
        assert f_steady_entry is not None

        f_steady_func, f_steady_args, f_steady_diff = f_steady_entry
        f_unsteady_func: Callable[..., Any] | None
        if f_unsteady_entry is not None:
            f_unsteady_func, f_unsteady_args, f_unsteady_diff = f_unsteady_entry
        else:
            f_unsteady_func = None
            f_unsteady_args = {}
            f_unsteady_diff = []

        combined_args = {**f_steady_args, **f_unsteady_args}
        combined_diff_ordered: list[str] = []
        for name in list(f_steady_diff) + list(f_unsteady_diff):
            if name not in combined_diff_ordered:
                combined_diff_ordered.append(name)

        f_steady_kwargs_names = set(f_steady_args.keys())
        f_unsteady_kwargs_names = set(f_unsteady_args.keys())

        def combined(
            f_unsteady_func_: Callable[..., Any] | None = f_unsteady_func,
            **kwargs: Any,
        ) -> Array:
            f_steady_kwargs = {
                k: v for k, v in kwargs.items() if k in f_steady_kwargs_names
            }
            f_steady = f_steady_func(**f_steady_kwargs)
            if f_unsteady_func_ is not None:
                f_unsteady_kwargs = {
                    k: v for k, v in kwargs.items() if k in f_unsteady_kwargs_names
                }
                f_unsteady = f_unsteady_func_(**f_unsteady_kwargs)
            else:
                f_unsteady = None
            return proj.to_output(f_steady, f_unsteady)

        new_residuals = {
            k: v for k, v in residuals.items() if k not in ("f_steady", "f_unsteady")
        }
        new_residuals[proj.name] = (combined, combined_args, combined_diff_ordered)

        return new_residuals

    def create_jacobians(
        self,
        mode: ADMode | dict[str, ADMode] = "reverse",
        batch_size: int | None = None,
        input_projection: LinearInputProjection | None = None,
        output_projection: LinearOutputProjection | None = None,
        residual_names: Sequence[str] | None = None,
    ) -> dict[str, dict[str, Array]]:
        r"""
        Compute the per-residual Jacobians using :func:`jacrev_custom`.
        """
        residuals = self.compute_jacobians(
            input_projection=input_projection,
            output_projection=output_projection,
            residual_names=residual_names,
        )

        jacobians: dict[str, dict[str, Array]] = {}
        for res_name, (res_func, args, diff_arg_names) in residuals.items():
            res_jac_options: dict[str, Callable[..., Array] | None] = {
                arg: None for arg in diff_arg_names
            }

            res_mode: ADMode = (
                mode.get(res_name, "reverse") if isinstance(mode, dict) else mode
            )

            jacs, _, _ = jacrev_custom(
                func=res_func,
                jac_options=res_jac_options,
                n_profile_loops=None,
                func_name=res_name,
                map_batch_size=batch_size,
                mode=res_mode,
            )(**args)
            jacobians[res_name] = jacs

        return jacobians

    def linearise(
        self,
        batch_size: int | None = None,
        *,
        input_projection: LinearInputProjection | None = None,
        output_projection: LinearOutputProjection | None = None,
    ) -> LinearSystem:
        r"""
        Build the linear state-space system.
        :param batch_size: If not None, batch the Jacobian passes to reduce memory.
        :param input_projection: Optional projection from lower-dim inputs (e.g. beam DOFs) to
        ``(zeta_b, zeta_b_dot)``.
        :param output_projection: Optional projection from ``(f_steady, f_unsteady)`` to a
        smaller output (e.g. beam force); when set, C/D rows correspond to the projected output.
        :return: LinearSystem object.
        """
        jacobians = self.create_jacobians(
            mode="reverse",
            batch_size=batch_size,
            input_projection=input_projection,
            output_projection=output_projection,
        )

        # (row, column, size)
        ref = self.reference
        state_specs: list[tuple[str, str, int]] = [
            ("gamma_b", "gamma_b_n_vec", ref.gamma_b.size),
            ("gamma_w", "gamma_w_n_vec", ref.gamma_w.size),
        ]
        if self.unsteady_force:
            state_specs.append(("gamma_b_nm1", "gamma_b_nm1_vec", ref.gamma_b.size))
        if self.prescribed_wake:
            state_specs.append(("zeta_w", "zeta_w_n_vec", ref.zeta_w.size))
            state_specs.append(("zeta_b", "zeta_b_n_vec", ref.zeta_b.size))
        state_names = [s[0] for s in state_specs]
        state_arg_names = [s[1] for s in state_specs]
        state_sizes = [s[2] for s in state_specs]

        if input_projection is None:
            input_specs: list[tuple[str, int]] = [
                ("zeta_b_np1_vec", ref.zeta_b.size),
                ("zeta_b_dot_np1_vec", ref.zeta_b.size),
            ]
        else:
            input_specs = list(
                zip(input_projection.arg_names, input_projection.arg_sizes)
            )
        if self.bound_upwash:
            input_specs.append(("nu_b_np1_vec", ref.zeta_b.size))
        if self.wake_upwash:
            input_specs.append(("nu_w_np1_vec", ref.zeta_w.size))
        input_arg_names = [s[0] for s in input_specs]
        input_sizes = [s[1] for s in input_specs]

        if output_projection is None:
            n_fs = sum(3 * (m + 1) * (n + 1) for (m, n) in ref.gamma_b.shape)
            output_specs: list[tuple[str, int]] = [("f_steady", n_fs)]
            if self.unsteady_force:
                output_specs.append(("f_unsteady", n_fs))
        else:
            output_specs = [(output_projection.name, output_projection.size)]
        output_names = [s[0] for s in output_specs]
        output_sizes = [s[1] for s in output_specs]

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
        c = construct_named_block_jacobian(
            entries=tuple(jacobians[k] for k in output_names),
            keys=state_arg_names,
            widths=state_sizes,
            heights=output_sizes,
        )
        d = construct_named_block_jacobian(
            entries=tuple(jacobians[k] for k in output_names),
            keys=input_arg_names,
            widths=input_sizes,
            heights=output_sizes,
        )

        return LinearSystem(a=a, b=b, c=c, d=d, dt=self.dt)

    # noinspection PyMethodOverriding
    def run(
        self,
        u: AeroInputUnflattened,
        x0: AeroStateUnflattened | None = None,
        flowfield: FlowField | None = None,
    ) -> AeroLinearResult:
        r"""
        Run the linear system.
        :param u: Total input over time (reference + pertubation).
        :param x0: Initial state perturbations, defaults to zero state.
        :param flowfield: FlowField object to provide flow velocities for bound and wake upwash, defaults to no flow.
        """
        if self.prescribed_wake and self.sys.removed_u_np1:
            warn(
                "Wake perturbations coordinates at the trailing edge are zero when removing u_np1 from the system."
            )

        if x0 is None:
            x0_vec = None
        else:
            x0_vec = self._pack_state_vector_t(x0)

        n_tstep: int = u.zeta_b[0].shape[
            0
        ]  # number of time steps from first surface, first entry
        t = self.reference.t + jnp.arange(0, n_tstep) * self.dt  # time vector

        u_tot = u

        if self.bound_upwash and flowfield is None and u_tot.nu_b is None:
            warn(
                "No flowfield or bound upwash perturbations provided. Assuming zero bound upwash perturbations."
            )
            u_tot.nu_b = ArrayList(
                [jnp.zeros((n_tstep, *zb.shape)) for zb in self.reference.zeta_b]
            )

        if self.wake_upwash and flowfield is None and u_tot.nu_w is None:
            warn(
                "No flowfield or wake upwash perturbations provided. Assuming zero wake upwash perturbations."
            )
            u_tot.nu_w = ArrayList(
                [jnp.zeros((n_tstep, *zw.shape)) for zw in self.reference.zeta_w]
            )

        # add flowfield contributions to input upwash if provided
        if flowfield is not None:
            if self.bound_upwash:
                nu_b_flow = ArrayList([])
                for i_surf in range(self.reference.n_surf):
                    nu_b_flow.append(
                        vmap(flowfield.vmap_call, in_axes=(None, 0), out_axes=0)(
                            self.reference.zeta_b[i_surf],
                            t,  # type: ignore
                        )
                        - flowfield.vmap_call(self.reference.zeta_b[i_surf], t[0])[
                            None, ...
                        ]
                    )
                if u_tot.nu_b is None:
                    u_tot.nu_b = nu_b_flow
                else:
                    u_tot.nu_b += nu_b_flow
            if self.wake_upwash:
                nu_w_flow = ArrayList([])
                for i_surf in range(self.reference.n_surf):
                    nu_w_flow.append(
                        vmap(flowfield.vmap_call, in_axes=(None, 0), out_axes=0)(
                            self.reference.zeta_w[i_surf],
                            t,  # type: ignore
                        )
                        - flowfield.vmap_call(self.reference.zeta_w[i_surf], t[0])[
                            None, ...
                        ]
                    )
                if u_tot.nu_w is None:
                    u_tot.nu_w = nu_w_flow
                else:
                    u_tot.nu_w += nu_w_flow
        u_vec = self._pack_input_vector_t(u_tot)

        # run linear system
        x_t, y_t = self.sys.run(u_vec, x0_vec)

        x_t_obj = self._unpack_state_vector_t(x_t)
        y_t_obj = self._unpack_output_vector_t(y_t)

        assert isinstance(x_t_obj, AeroStateUnflattened) and isinstance(
            y_t_obj, AeroOutputUnflattened
        ), (
            "Unpacked state and output must be of type AeroStateUnflattened and AeroOutputUnflattened."
        )

        x_t_tot_obj = self.get_total_state_t(x_t_obj)
        y_t_tot_obj = self.get_total_output_t(y_t_obj)
        u_t_tot_obj = self.get_total_input_t(u_tot)

        assert (
            isinstance(u_t_tot_obj, AeroInputUnflattened)
            and isinstance(x_t_tot_obj, AeroStateUnflattened)
            and isinstance(y_t_tot_obj, AeroOutputUnflattened)
        ), (
            "Unpacked total state and output must be of type AeroStateUnflattened and AeroOutputUnflattened."
        )

        assert isinstance(self.reference, AeroCase), (
            "Reference state must be of type AeroCase."
        )

        # save results to object
        return AeroLinearResult(
            reference=self.reference,
            u_t=u,
            x_t=x_t_obj,
            y_t=y_t_obj,
            u_t_tot=u_t_tot_obj,
            x_t_tot=x_t_tot_obj,
            y_t_tot=y_t_tot_obj,
            n_tstep=n_tstep,
            t=t,
            n_surf=self.reference.n_surf,
            surf_b_names=self.case.surf_b_names,
            surf_w_names=self.case.surf_w_names,
        )

    def reference_snapshot(self) -> AeroCase:
        r"""
        Get the reference (initial) initial_snapshot of the aerodynamic case. This will set the timestep as -1.
        :return: StaticAero at reference state
        """
        return AeroCase(
            zeta_b=self.reference.zeta_b,
            zeta_b_dot=self.reference.zeta_b_dot,
            zeta_w=self.reference.zeta_w,
            gamma_b=self.reference.gamma_b,
            gamma_b_dot=self.reference.gamma_b_dot,
            gamma_w=self.reference.gamma_w,
            f_steady=self.reference.f_steady,
            f_unsteady=self.reference.f_unsteady,
            cs_ang=self.reference.cs_ang,
            cs_vel=self.reference.cs_vel,
            surf_b_names=self.case.surf_b_names,
            surf_w_names=self.case.surf_w_names,
            i_ts=-1,
            t=jnp.array(0.0),
            c=self.reference.c,
            n=self.reference.nc,
            alpha=self.reference.alpha,
            cl=self.reference.cl,
            cd=self.reference.cd,
            cm=self.reference.cm,
            kernels=self.reference.kernels,
            mirror_normal=self.reference.mirror_normal,
            mirror_point=self.reference.mirror_point,
            mirror_edge_low=self.reference.mirror_edge_low,
            mirror_edge_high=self.reference.mirror_edge_high,
            flowfield=self.reference.flowfield,
            dof_mapping=self.reference.dof_mapping,
            free_wake=self.reference.free_wake,
            gamma_dot_relaxation=self.reference.gamma_dot_relaxation,
            static_horseshoe=self.reference.static_horseshoe,
            batch_size=self.case.batch_size,
        )

    def plot_reference(
        self, directory: os.PathLike, plot_wake: bool = True
    ) -> Sequence[Path]:
        r"""
        Plot the reference (initial) initial_snapshot of the aerodynamic case. This will set the timestep as -1.
        :param directory: File path to save the plots to
        :param plot_wake: If True, plot the wake grid
        """
        return self.reference_snapshot().plot(
            Path(directory).resolve(), index=None, plot_wake=plot_wake
        )
