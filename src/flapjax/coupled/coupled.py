from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, Literal

import equinox.internal as eqxi
from jax import Array, vmap
from jax import numpy as jnp

from flapjax.aero.data_structures import AeroCase
from flapjax.aero.flowfields import FlowField
from flapjax.aero.linear import LinearWakeType
from flapjax.aero.utils import cs_ang_to_cs_vel
from flapjax.algebra.array_utils import ArrayList, check_arr_shape
from flapjax.algebra.se3 import hg_to_d
from flapjax.coupled.data_structures import (
    AeroelasticCase,
    AeroelasticDesignVariables,
)
from flapjax.coupled.gradients.data_structures import AeroelasticGradsToCompute
from flapjax.coupled.linear.linear_coupled import LinearCoupled
from flapjax.structure import BeamStructure, StructureCase
from flapjax.structure.time_integration import TimeIntegrator
from flapjax.structure.utils import (
    get_solve_dofs,
    transform_nodal_vect,
)
from flapjax.utils.constants import BASE_LOBATTO_ORDER
from flapjax.utils.data_structures import ConvergenceSettings, ConvergenceStatus
from flapjax.utils.print_utils import (
    get_verbosity,
    map_verbosity_level,
    warn,
)
from flapjax.utils.utils import make_pytree

if TYPE_CHECKING:
    from flapjax.aero.uvlm import UVLM

DEFAULT_FSI_CONVERGENCE_SETTINGS = ConvergenceSettings(
    max_n_iter=25,
    rel_disp_tol=1e-3,
    abs_disp_tol=1e-5,
    rel_force_tol=1e-3,
    abs_force_tol=1e-5,
)


@make_pytree
class BaseCoupledAeroelastic:
    _static: ClassVar[tuple[str, ...]] = ("fsi_convergence_settings",)

    def __init__(
        self,
        structure: BeamStructure,
        aero: UVLM,
        fsi_convergence_settings: ConvergenceSettings = DEFAULT_FSI_CONVERGENCE_SETTINGS,
    ):
        self.structure: BeamStructure = structure
        self.aero: UVLM = aero
        self.fsi_convergence_settings: ConvergenceSettings = fsi_convergence_settings

    def set_design_variables(
        self,
        coords: Array,
        k_cs: Array,
        m_cs: Array | None,
        m_lumped: Array | None,
        dt: float | Array,
        flowfield: FlowField,
        x0_aero: ArrayList | Sequence[Array] | Array,
        delta_w: Sequence[Array | None] | Array | None = None,
        thrust_reference: dict[str, Array] | None = None,
        orientation_euler: Array | None = None,
        cs_angles_reference: dict[str, Array] | None = None,
        *,
        remove_checks: bool = False,
    ):
        self.structure.set_design_variables(
            coords=coords,
            k_cs=k_cs,
            m_cs=m_cs,
            m_lumped=m_lumped,
            thrust_reference=thrust_reference,
            orientation_euler=orientation_euler,
            remove_checks=remove_checks,
        )
        self.aero.set_design_variables(
            dt=dt,
            flowfield=flowfield,
            delta_w=delta_w,
            zeta_b0=x0_aero,
            hg0=self.structure.hg0,
            reference_cs_angles=cs_angles_reference,
        )

    def case_from_dv(self, dv: AeroelasticDesignVariables) -> BaseCoupledAeroelastic:
        return BaseCoupledAeroelastic(
            structure=self.structure.case_from_dv(dv.structure),
            aero=self.aero.case_from_dv(dv.aero),
        )

    def get_design_variables(
        self,
        case: AeroelasticCase,
        grads_to_compute: AeroelasticGradsToCompute | None,
    ) -> AeroelasticDesignVariables:
        r"""
        Obtain the design variables describing the wing.
        :param case:
        :param grads_to_compute: Data structure which describes which design variables should be obtained. If none, all
        variables are obtained.
        :return: AeroelasticDesignVariables object.
        """
        return AeroelasticDesignVariables(
            structure_dv=self.structure.get_design_variables(
                struct_case=case.structure,
                thrust_t=case.structure.thrust,
                grads_to_compute=grads_to_compute.structure
                if grads_to_compute is not None
                else None,
            ),
            aero_dv=self.aero.get_design_variables(
                cs_ang_t=case.aero.cs_ang,
                cs_vel_t=case.aero.cs_vel,
                grads_to_compute=grads_to_compute.aero
                if grads_to_compute is not None
                else None,
            ),
        )

    def reference_configuration(
        self,
        prescribed_dofs: Sequence[int] | Array | slice | int,
        horseshoe: bool = False,
        use_f_ext_follower: bool = False,
        use_f_ext_dead: bool = False,
        t_init: float | Array = 0.0,
    ) -> AeroelasticCase:
        r"""
        Obtain the static aeroelastic object describing the undeformed wing.
        :param prescribed_dofs: Prescribed dofs for the structure.
        :param horseshoe: Horseshoe flag.
        :param use_f_ext_follower: If true, allocate an array for follower forces.
        :param use_f_ext_dead: If true, allocate an array for dead forces.

        :param t_init: Initial time
        :return: Static aeroelastic object for undeformed wing
        """
        prescribed_dofs = self.structure.make_prescribed_dofs_tuple(prescribed_dofs)
        return AeroelasticCase(
            structure=self.structure.reference_configuration(
                use_f_grav=self.structure.use_gravity,
                use_f_ext_dead=use_f_ext_dead,
                use_f_ext_follower=use_f_ext_follower,
                use_f_aero=True,
                prescribed_dofs=prescribed_dofs,
            ),
            aero=self.aero.solve_static(
                t=t_init, hg=self.structure.hg0, horseshoe=horseshoe
            ),
        )

    def static_solve(
        self,
        prescribed_dofs: Sequence[int] | Array | slice | int,
        f_ext_follower: Array | None = None,
        f_ext_dead: Array | None = None,
        t: float | Array = 0.0,
        load_steps: int = 1,
        horseshoe: bool = False,
        fsi_relaxation: float = 0.6,
        fsi_omega_min: float = 1e-2,
    ) -> AeroelasticCase:
        prescribed_dofs: tuple[int, ...] = self.structure.make_prescribed_dofs_tuple(
            prescribed_dofs
        )

        if not self.aero.flowfield.relative_motion:
            warn(
                "Flow field relative motion is disabled. Static solve may not be accurate."
            )

        if not 0.0 < fsi_relaxation <= 1.0:
            raise ValueError(f"fsi_relaxation must be in (0, 1], got {fsi_relaxation}.")
        if not 0.0 < fsi_omega_min <= fsi_relaxation:
            raise ValueError(
                f"fsi_omega_min must be in (0, fsi_relaxation], got {fsi_omega_min}."
            )

        # mask out prescribed dofs from the force residual
        free_mask = jnp.ones((self.structure.n_nodes, 6), dtype=bool)
        for dof in prescribed_dofs:
            node, comp = divmod(dof, 6)
            free_mask = free_mask.at[node, comp].set(False)

        def _convergence_loop(
            converge_status_: ConvergenceStatus,
            struct_case_n: StructureCase,
            aero_case_n: AeroCase,
            omega_prev: Array,
            r_prev: Array,
        ) -> tuple[ConvergenceStatus, StructureCase, AeroCase, Array, Array]:
            f_aero_new = aero_case_n.project_forcing_to_beam(
                i_ts=0,
                rmat=struct_case_n.hg[:, :3, :3],
                x0_aero=self.aero.zeta_b0,
                include_unsteady=False,
            )  # [n_nodes_, 6], global frame

            # convert from local to global frame
            assert struct_case_n.f_ext_aero is not None
            f_applied_global = transform_nodal_vect(
                struct_case_n.f_ext_aero, struct_case_n.hg[:, :3, :3]
            )

            # unrelaxed Picard residual on unconstrained dofs
            r_curr = jnp.where(free_mask, f_aero_new - f_applied_global, 0.0)

            # Aitken delta squared update
            delta_r = r_curr - r_prev
            den = jnp.vdot(delta_r, delta_r)
            omega_aitken = (
                -omega_prev * jnp.vdot(r_prev, delta_r) / jnp.where(den > 0.0, den, 1.0)
            )
            omega_aitken = jnp.clip(omega_aitken, fsi_omega_min, 1.0)
            omega = jnp.where(
                converge_status_.i_iter >= 2,
                omega_aitken,
                jnp.array(fsi_relaxation),
            )

            # apply relaxation to the forcing, using the full forcing for the first iteration
            f_aero_n = jnp.where(
                converge_status_.i_iter == 0,
                f_aero_new,
                f_applied_global + omega * r_curr,
            )

            struct_case_np1 = self.structure.static_solve(
                f_ext_follower=f_ext_follower,
                f_ext_dead=f_ext_dead,
                f_ext_aero=f_aero_n,
                prescribed_dofs=prescribed_dofs,
                load_steps=load_steps,
                print_header=False,
            )

            # compute the full varphi delta
            delta_n_raw = vmap(hg_to_d)(
                struct_case_n.hg, struct_case_np1.hg
            )  # (n_nodes, 6)
            # delta with relaxation
            delta_n = jnp.where(
                converge_status_.i_iter == 0,
                delta_n_raw,
                delta_n_raw / omega,
            )

            tot_n = vmap(hg_to_d)(struct_case_np1.hg, self.structure.hg0)

            delta_f = -r_curr  # f_applied_global - f_aero_new
            total_f = jnp.where(free_mask, f_aero_new, 0.0)

            converge_status_.update(
                delta_disp=delta_n,
                total_disp=tot_n,
                delta_force=delta_f,
                total_force=total_f,
            )

            aero_case_np1 = self.aero.solve_static(
                t=t,
                hg=struct_case_np1.hg,
                horseshoe=horseshoe,
                cs_ang=self.aero.cs_ang0,
            )

            if map_verbosity_level(get_verbosity()) >= map_verbosity_level("normal"):
                converge_status_.print_fsi_message(i_ts=None, t=None)

            return converge_status_, struct_case_np1, aero_case_np1, omega, r_curr

        fsi_converge_status = ConvergenceStatus(self.fsi_convergence_settings)
        fsi_converge_status.print_header(dynamic=False)

        _, struct_case, aero_case, _, _ = eqxi.while_loop(
            lambda args_: ~args_[0].get_status(),
            lambda args_: _convergence_loop(*args_),  # type: ignore
            (
                fsi_converge_status,
                self.structure.reference_configuration(
                    use_f_grav=self.structure.use_gravity,
                    use_f_ext_dead=f_ext_dead is not None,
                    use_f_ext_follower=f_ext_follower is not None,
                    use_f_aero=True,
                    prescribed_dofs=prescribed_dofs,
                ),
                self.aero.solve_static(t=t, hg=self.structure.hg0, horseshoe=horseshoe),
                jnp.array(fsi_relaxation),
                jnp.zeros((self.structure.n_nodes, 6)),
            ),
            max_steps=self.fsi_convergence_settings.max_n_iter,
            kind="bounded",
        )

        fsi_converge_status.print_line(dynamic=False)

        return AeroelasticCase(structure=struct_case, aero=aero_case)

    def dynamic_solve(
        self,
        init_case: AeroelasticCase | None,
        n_tstep: int,
        prescribed_dofs: Sequence[int] | Array | slice | int | None = None,
        f_ext_follower: Array | None = None,
        f_ext_dead: Array | None = None,
        t_init: float = 0.0,
        load_steps: int = 1,
        thrust_t: dict[str, Array] | None = None,
        cs_ang_t: dict[str, Array] | None = None,
        cs_vel_t: dict[str, Array] | None = None,
    ) -> AeroelasticCase:
        # check control inputs
        if cs_ang_t is not None:
            for key in cs_ang_t:
                if cs_ang_t[key].shape[0] != n_tstep:
                    raise ValueError(
                        f"Inconsistent number of time steps for control surface input cs_ang_t['{key}']"
                    )

                if cs_vel_t is not None and key not in cs_vel_t:
                    raise ValueError(
                        f"Missing control velocity for control surface input cs_vel_t['{key}']"
                    )
        elif init_case is not None:
            # if control angles are not input, we obtain them from the initial case and keep them constant over time
            cs_ang_t = {
                k: jnp.full(n_tstep, v) for k, v in init_case.aero.cs_ang.items()
            }

        # if control velocities are not input, we obtain them from finite differences
        if cs_vel_t is None and cs_ang_t is not None:
            cs_vel_t = cs_ang_to_cs_vel(cs_ang_t=cs_ang_t, dt=self.aero.dt)

        # check thrust
        if thrust_t is not None:
            if thrust_t.keys() != dict(self.structure.thrust_direction).keys():
                raise ValueError("Mismatch in keys for thrust")

            for k, v in thrust_t.items():
                check_arr_shape(v, (n_tstep,), name=f"thrust_t['{k}']")

            thrust_t_: dict[str, Array] = {}
        else:
            thrust_t_ = {
                k: jnp.full(n_tstep, v)
                for k, v in self.structure.thrust_reference.items()
            }

        if prescribed_dofs is None:
            if init_case is None:
                raise ValueError(
                    "prescribed_dofs must be provided if no init_case is provided."
                )
            prescribed_dofs = init_case.structure.prescribed_dofs

        # degrees of freedom to constrain or solve for
        prescribed_dofs: tuple[int, ...] = self.structure.make_prescribed_dofs_tuple(
            prescribed_dofs
        )
        solve_dofs = get_solve_dofs(
            n_dof=self.structure.n_dof, prescribed_dofs=prescribed_dofs
        )

        t = jnp.arange(n_tstep) * self.aero.dt + t_init

        self.structure.time_integrator = TimeIntegrator(
            spectral_radius=self.structure.spectral_radius, dt=self.aero.dt
        )

        # initialise aeroelastic case object
        if init_case is None:
            case: AeroelasticCase = AeroelasticCase.initialise(
                initial_snapshot=self.reference_configuration(
                    prescribed_dofs=prescribed_dofs,
                    use_f_ext_follower=f_ext_follower is not None,
                    use_f_ext_dead=f_ext_dead is not None,
                ).to_dynamic(t=None),
                t=t,
                use_f_ext_follower=f_ext_follower is not None,
                use_f_ext_dead=f_ext_dead is not None,
                structure=self.structure,
                x0_aero=self.aero.zeta_b0,
            )

            # set forces at timestep 0
            # this is important as the time integration scheme refers to these values when solving the first timestep
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
        else:
            case = AeroelasticCase.initialise(
                initial_snapshot=init_case,
                t=t,
                use_f_ext_follower=f_ext_follower is not None,
                use_f_ext_dead=f_ext_dead is not None,
                structure=self.structure,
                x0_aero=self.aero.zeta_b0,
            )

        # propagate the dynamic prescribed_dofs to the case
        case.structure.prescribed_dofs = prescribed_dofs

        fsi_converge_status: ConvergenceStatus = ConvergenceStatus(
            self.fsi_convergence_settings
        )
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
            aero_obj=self.aero,
            aero_case=case.aero,
            fsi_convergence_status=fsi_converge_status,
            thrust_t=thrust_t_,
            cs_ang_t=cs_ang_t if cs_ang_t is not None else {},
            cs_vel_t=cs_vel_t if cs_vel_t is not None else {},
        )

        fsi_converge_status.print_line(dynamic=True)
        return out

    def initialise_dynamic(
        self,
        static_case: AeroelasticCase,
        prescribed_dofs: Sequence[int] | Array | slice | int,
    ) -> AeroelasticCase:
        r"""
        Initialise a dynamic aeroelastic snapshot from a static aeroelastic case. This takes a static aeroelastic case
        obtained under clamped conditions (i.e. `relative_motion` is True for the free stream), and sets the structural
        velocity to be that of the freestream.
        :param static_case: Static aeroelastic case.
        :param prescribed_dofs: Prescribed dofs. This is often useful for updating from a clamped trim to a free-flying
        dynamic case.
        :return: Dynamic aeroelastic snapshot.
        """
        u_inf = self.aero.flowfield.u_inf  # flowfield velocity to set
        self.aero.flowfield.relative_motion = (
            False  # the output will have relative motion disabled
        )

        rmat_struct = static_case.structure.hg[:, :3, :3]  # deformed rotations
        v_local = jnp.einsum(
            "ijk,j->ik", rmat_struct, -u_inf
        )  # local frame velocity, (n_nodes, 3)

        dynamic_case = static_case.to_dynamic(t=None)
        dynamic_case.structure.v = dynamic_case.structure.v.at[:, :3].set(v_local)
        dynamic_case.structure.prescribed_dofs = (
            self.structure.make_prescribed_dofs_tuple(prescribed_dofs)
        )
        dynamic_case.structure.free_dofs = get_solve_dofs(
            n_dof=self.structure.n_dof,
            prescribed_dofs=dynamic_case.structure.prescribed_dofs,
        )

        return dynamic_case

    def linearise(
        self,
        reference: AeroelasticCase,
        wake_type: LinearWakeType = "frozen",
        batch_size: int | None = 64,
        n_struct_modes: int | None = None,
        bound_upwash: bool = False,
        wake_upwash: bool = False,
        unsteady_force: bool = True,
        int_order: Literal[3, 4, 5] = BASE_LOBATTO_ORDER,
        *,
        skip_checks: bool = False,
        prescribed_dofs: Sequence[int] | Array | slice | int | None = None,
    ) -> LinearCoupled:
        return LinearCoupled(
            case=self,
            reference=reference,
            wake_type=wake_type,
            bound_upwash=bound_upwash,
            wake_upwash=wake_upwash,
            unsteady_force=unsteady_force,
            batch_size=batch_size,
            int_order=int_order,
            skip_checks=skip_checks,
            n_struct_modes=n_struct_modes,
            prescribed_dofs=prescribed_dofs,
        )
