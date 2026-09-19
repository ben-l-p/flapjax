from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from jax import Array
from jax import numpy as jnp

from flapjax.aero.data_structures import AeroCase
from flapjax.aero.flowfields import FlowField
from flapjax.aero.linear.data_structures import (
    AeroInputUnflattened,
    AeroStateUnflattened,
)
from flapjax.aero.linear.linear_uvlm import LinearUVLM
from flapjax.aero.utils import compute_c, compute_nc, cs_ang_to_cs_vel
from flapjax.algebra.array_utils import ArrayList, check_arr_shape
from flapjax.coupled.coupled import DEFAULT_FSI_CONVERGENCE_SETTINGS
from flapjax.coupled.data_structures import AeroelasticCase
from flapjax.structure import BeamStructure
from flapjax.structure.time_integration import TimeIntegrator
from flapjax.structure.utils import get_solve_dofs
from flapjax.utils.data_structures import ConvergenceSettings, ConvergenceStatus
from flapjax.utils.print_utils import warn
from flapjax.utils.utils import make_pytree


@make_pytree
class NonlinearBeamLinearAero:
    r"""
    Aeroelastic case coupling a nonlinear beam with a pre-built linear UVLM model. Useful for
    relatively cheap time-domain aeroelastic simulations where the aerodynamic nonlinearities are
    not significant.
    """

    _static: ClassVar[tuple[str, ...]] = (
        "fsi_convergence_settings",
        "include_unsteady_force",
    )

    def __init__(
        self,
        structure: BeamStructure,
        aero: LinearUVLM,
        fsi_convergence_settings: ConvergenceSettings = DEFAULT_FSI_CONVERGENCE_SETTINGS,
    ) -> None:
        r"""
        :param structure: Nonlinear ``BeamStructure`` with design variables set.
        :param aero: Pre-built ``LinearUVLM`` linearised about some reference state.
        :param fsi_convergence_settings: FSI iteration convergence controls.
        """
        self.structure: BeamStructure = structure
        self.aero: LinearUVLM = aero
        self.fsi_convergence_settings: ConvergenceSettings = fsi_convergence_settings
        self.include_unsteady_force: bool = aero.unsteady_force

    @property
    def zeta_b0(self) -> ArrayList:
        return self.aero.case.zeta_b0

    @property
    def dt(self) -> Array:
        return self.aero.case.dt

    @property
    def flowfield(self) -> FlowField:
        return self.aero.case.flowfield

    @property
    def cs_ang0(self) -> dict[str, Array]:
        return self.aero.reference.cs_ang

    def get_state(self, i_ts: int, case: AeroCase) -> AeroStateUnflattened:
        r"""
        Extract linear aero states at time step ``i_ts`` from the case object.
        :param i_ts: Time step index.
        :param case: Batched ``AeroCase`` containing the linear aero states.
        :return: Linear aero states at time step ``i_ts``.
        """
        gamma_b = case.gamma_b.index_all(i_ts, ...)
        gamma_w = case.gamma_w.index_all(i_ts, ...)

        if self.aero.unsteady_force:
            i_prev = jnp.maximum(i_ts - 1, 0)
            gamma_b_nm1 = case.gamma_b.index_all(i_prev, ...)
        else:
            gamma_b_nm1 = None

        if self.aero.prescribed_wake:
            assert case.zeta_w is not None
            zeta_w = case.zeta_w.index_all(i_ts, ...)
            zeta_b_state = case.zeta_b.index_all(i_ts, ...)
        else:
            zeta_w = None
            zeta_b_state = None

        return AeroStateUnflattened(
            gamma_b=gamma_b,
            gamma_w=gamma_w,
            gamma_b_nm1=gamma_b_nm1,
            zeta_w=zeta_w,
            zeta_b=zeta_b_state,
        )

    def case_solve(
        self,
        case: AeroCase,
        i_ts: int,
        hg_n: Array | None,
        hg_nm1: Array | None,
        hg_dot_n: Array | None,
        static: bool,
        horseshoe: bool,
        cs_ang_n: dict[str, Array],
        cs_ang_nm1: dict[str, Array] | None,
        cs_vel_n: dict[str, Array] | None,
    ) -> AeroCase:
        r"""
        Step the linear aero system one step and write the result into ``case``. Called by
        ``BeamStructure.base_dynamic_solve`` inside the FSI loop. Only supports dynamic solves;
        ``hg_nm1``, ``cs_ang_nm1`` and ``horseshoe`` are part of the ``DynamicAeroSolver`` protocol
        but not consumed here.
        """
        del hg_nm1, cs_ang_nm1, horseshoe
        if static:
            raise NotImplementedError(
                "case_solve(static=True) is not supported for the linear aero adapter"
            )
        assert hg_n is not None and hg_dot_n is not None

        uvlm = self.aero.case
        sys = self.aero.sys
        cs_vel_n_ = cs_vel_n if cs_vel_n is not None else {}

        zeta_b_n = uvlm.hg_to_zeta_b(hg_n=hg_n, cs_ang_n=cs_ang_n)
        zeta_b_dot_n = uvlm.hg_dot_to_zeta_b_dot(
            hg_n=hg_n,
            hg_dot_n=hg_dot_n,
            cs_ang_n=cs_ang_n,
            cs_vel_n=cs_vel_n_,
        )

        # flowfield perturbation from the linearisation reference, applied as extra input upwash
        t_n = case.t[i_ts]
        t_ref = self.aero.reference.t

        if self.aero.bound_upwash:
            nu_b_n = ArrayList.zeros_like(zeta_b_n)
            nu_b_n += self.flowfield.surf_vmap_call(
                xs=self.aero.reference.zeta_b, t=t_n
            ) - self.flowfield.surf_vmap_call(xs=self.aero.reference.zeta_b, t=t_ref)
        else:
            nu_b_n = None

        if self.aero.wake_upwash:
            nu_w_n = ArrayList.zeros_like(self.aero.reference.zeta_w)
            nu_w_n += self.flowfield.surf_vmap_call(
                xs=self.aero.reference.zeta_w, t=t_n
            ) - self.flowfield.surf_vmap_call(xs=self.aero.reference.zeta_w, t=t_ref)
        else:
            nu_w_n = None

        u_n = AeroInputUnflattened(
            zeta_b=zeta_b_n,
            zeta_b_dot=zeta_b_dot_n,
            nu_b=nu_b_n,
            nu_w=nu_w_n,
        )
        u_n_vec = self.aero.pack_input_vector(u_n)
        u_ref_vec = self.aero.pack_input_vector(self.aero.reference_inputs)
        du_n = u_n_vec - u_ref_vec

        x_nm1_unflat = self.get_state(i_ts=i_ts - 1, case=case)
        x_nm1_vec = self.aero.pack_state_vector(x_nm1_unflat)
        x_ref_vec = self.aero.pack_state_vector(self.aero.reference_states)
        dx_prev = x_nm1_vec - x_ref_vec

        dx_n = sys.a @ dx_prev + sys.b @ du_n
        dy_n = sys.c @ dx_n + sys.d @ du_n

        x_n_vec = dx_n + x_ref_vec
        y_ref_vec = self.aero.pack_output_vector(self.aero.reference_outputs)
        y_n_vec = dy_n + y_ref_vec

        x_n_unflat = self.aero.unpack_state_vector(x_n_vec)
        y_n_unflat = self.aero.unpack_output_vector(y_n_vec)

        case.set_arraylist_at_ts("zeta_b", zeta_b_n, i_ts)
        case.set_arraylist_at_ts("zeta_b_dot", zeta_b_dot_n, i_ts)
        case.set_arraylist_at_ts("c", compute_c(zeta_b_n), i_ts)
        case.set_arraylist_at_ts("nc", compute_nc(zeta_b_n), i_ts)
        case.set_arraylist_at_ts("gamma_b", x_n_unflat.gamma_b, i_ts)
        case.set_arraylist_at_ts("gamma_w", x_n_unflat.gamma_w, i_ts)
        case.set_arraylist_at_ts("f_steady", y_n_unflat.f_steady, i_ts)

        if self.aero.unsteady_force:
            assert y_n_unflat.f_unsteady is not None
            gamma_b_dot_n = ArrayList(
                [
                    (gb - gb_prev) / self.dt
                    for gb, gb_prev in zip(x_n_unflat.gamma_b, x_nm1_unflat.gamma_b)
                ]
            )
            case.set_arraylist_at_ts("gamma_b_dot", gamma_b_dot_n, i_ts)
            case.set_arraylist_at_ts("f_unsteady", y_n_unflat.f_unsteady, i_ts)

        assert case.zeta_w is not None
        if self.aero.prescribed_wake:
            assert x_n_unflat.zeta_w is not None
            case.set_arraylist_at_ts("zeta_w", x_n_unflat.zeta_w, i_ts)
        else:
            case.set_arraylist_at_ts("zeta_w", self.aero.reference.zeta_w, i_ts)

        case.t = case.t.at[i_ts].set(t_n)

        return case

    def reference_configuration(
        self,
        prescribed_dofs: Sequence[int] | Array | slice | int = (),
        use_f_ext_follower: bool = False,
        use_f_ext_dead: bool = False,
    ) -> AeroelasticCase:
        r"""
        Aeroelastic snapshot built from the beam's reference configuration and
        the aero linearisation reference state.
        :param prescribed_dofs: Prescribed DOFs for the beam structure. Defaults to no prescribed DOFs.
        :param use_f_ext_follower: Whether to include follower forces in the reference configuration.
        :param use_f_ext_dead: Whether to include dead forces in the reference configuration.
        :return: Aeroelastic snapshot at the reference configuration.
        """
        prescribed_dofs_tuple = self.structure.make_prescribed_dofs_tuple(
            prescribed_dofs
        )
        return AeroelasticCase(
            structure=self.structure.reference_configuration(
                use_f_grav=self.structure.use_gravity,
                use_f_ext_dead=use_f_ext_dead,
                use_f_ext_follower=use_f_ext_follower,
                use_f_aero=True,
                prescribed_dofs=prescribed_dofs_tuple,
            ),
            aero=self.aero.reference_snapshot(),
        )

    def dynamic_solve(
        self,
        init_case: AeroelasticCase | None,
        prescribed_dofs: Sequence[int] | Array | slice | int,
        n_tstep: int,
        f_ext_follower: Array | None = None,
        f_ext_dead: Array | None = None,
        t_init: float = 0.0,
        load_steps: int = 1,
        thrust_t: dict[str, Array] | None = None,
        cs_ang_t: dict[str, Array] | None = None,
        cs_vel_t: dict[str, Array] | None = None,
    ) -> AeroelasticCase:
        r"""
        Dynamic aeroelastic solve with nonlinear beam and linear UVLM.
        """
        ref_cs_ang = self.aero.reference.cs_ang
        if cs_ang_t is not None:
            for key, series in cs_ang_t.items():
                if key not in ref_cs_ang:
                    raise ValueError(
                        f"cs_ang_t key '{key}' not present in the linearisation reference"
                    )
                if series.shape[0] != n_tstep:
                    raise ValueError(
                        f"Inconsistent number of time steps for control surface input cs_ang_t['{key}']"
                    )

                # TODO: implement linear control surfaces
                if not jnp.allclose(series, ref_cs_ang[key]):
                    warn(f"cs_ang_t['{key}'] deviates from the linearisation reference")
        cs_ang_t_ = (
            cs_ang_t
            if cs_ang_t is not None
            else {k: jnp.full(n_tstep, v) for k, v in ref_cs_ang.items()}
        )

        if cs_vel_t is None:
            cs_vel_t_ = cs_ang_to_cs_vel(cs_ang_t=cs_ang_t_, dt=self.aero.dt)
        else:
            cs_vel_t_ = cs_vel_t

        if thrust_t is not None:
            if thrust_t.keys() != dict(self.structure.thrust_direction).keys():
                raise ValueError("Mismatch in keys for thrust")
            for k, v in thrust_t.items():
                check_arr_shape(v, (n_tstep,), name=f"thrust_t['{k}']")
            thrust_t_: dict[str, Array] = thrust_t
        else:
            thrust_t_ = {
                k: jnp.full(n_tstep, v)
                for k, v in self.structure.thrust_reference.items()
            }

        prescribed_dofs_tuple = self.structure.make_prescribed_dofs_tuple(
            prescribed_dofs
        )
        solve_dofs = get_solve_dofs(
            n_dof=self.structure.n_dof, prescribed_dofs=prescribed_dofs_tuple
        )

        t = jnp.arange(n_tstep) * self.aero.dt + t_init

        self.structure.time_integrator = TimeIntegrator(
            spectral_radius=self.structure.spectral_radius, dt=self.aero.dt
        )

        if init_case is None:
            initial_snapshot = self.reference_configuration(
                prescribed_dofs=prescribed_dofs_tuple,
                use_f_ext_follower=f_ext_follower is not None,
                use_f_ext_dead=f_ext_dead is not None,
            ).to_dynamic(t=None)
        else:
            initial_snapshot = init_case

        case = AeroelasticCase.initialise(
            initial_snapshot=initial_snapshot,
            t=t,
            use_f_ext_follower=f_ext_follower is not None,
            use_f_ext_dead=f_ext_dead is not None,
            structure=self.structure,
            x0_aero=self.aero.case.zeta_b0,
        )

        if f_ext_follower is not None and case.structure.f_ext_follower is not None:
            case.structure.f_ext_follower = case.structure.f_ext_follower.at[
                0, ...
            ].set(f_ext_follower[0, ...])
        if f_ext_dead is not None and case.structure.f_ext_dead is not None:
            case.structure.f_ext_dead = case.structure.f_ext_dead.at[0, ...].set(
                self.structure.make_f_dead_ext(
                    f_ext=f_ext_dead[0, ...], rmat=case.structure.hg[0, :, :3, :3]
                )
            )

        case.structure.prescribed_dofs = prescribed_dofs_tuple

        fsi_converge_status = ConvergenceStatus(self.fsi_convergence_settings)
        fsi_converge_status.print_header(dynamic=True)

        out = self.structure.base_dynamic_solve(
            struct_case=case.structure,
            struct_convergence_status=ConvergenceStatus(
                self.structure.struct_convergence_settings
            ),
            t=t,
            solve_dofs=solve_dofs,
            load_steps=load_steps,
            f_ext_follower=f_ext_follower,
            f_ext_dead=f_ext_dead,
            aero_obj=self,
            aero_case=case.aero,
            fsi_convergence_status=fsi_converge_status,
            thrust_t=thrust_t_,
            cs_ang_t=cs_ang_t_,
            cs_vel_t=cs_vel_t_,
        )

        fsi_converge_status.print_line(dynamic=True)
        return out
