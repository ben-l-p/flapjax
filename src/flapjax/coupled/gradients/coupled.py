from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Final, Literal

import jax
from jax import Array, vmap
from jax import numpy as jnp

from flapjax.aero.gradients.data_structures import (
    AeroDesignVariables,
    AeroFullStates,
    AeroGradsToCompute,
)
from flapjax.aero.utils import project_forcing_to_beam
from flapjax.algebra.array_utils import ArrayList
from flapjax.algebra.base import ADMode
from flapjax.algebra.se3 import exp_se3
from flapjax.coupled.coupled import BaseCoupledAeroelastic
from flapjax.coupled.data_structures import (
    AeroelasticDesignVariables,
    AeroelasticFullStates,
    AeroelasticMinimalStates,
)
from flapjax.coupled.gradients.data_structures import (
    AeroelasticGradsToCompute,
    AeroelasticJacobianApproximations,
    TrimVariables,
)
from flapjax.coupled.gradients.utils import (
    check_unique_members,
    expand_groups,
    group_key,
    parse_groups,
)
from flapjax.structure import BeamStructure, StructureDesignVariables
from flapjax.structure.constraints import HardConstraint, MultibodyHinge
from flapjax.structure.data_structures import OptionalJacobians, StructureMinimalStates
from flapjax.structure.gradients.data_structures import (
    StructureFullStates,
    StructureGradsToCompute,
)
from flapjax.structure.utils import get_solve_dofs, transform_nodal_vect
from flapjax.utils.print_utils import (
    get_verbosity,
    jax_print,
    map_verbosity_level,
    print_table_line,
    print_table_title,
    verbosity,
    warn,
)
from flapjax.utils.utils import dv_or, make_pytree, pytree_clone

if TYPE_CHECKING:
    from flapjax.coupled.coupled import AeroelasticCase

type AeroelasticObjectiveFunction = (
    Callable[[AeroelasticFullStates, AeroelasticDesignVariables, None], Array]
    | Callable[[AeroelasticFullStates, AeroelasticDesignVariables, int], Array]
)
ORIENTATION_DICT: Final[dict[str, int]] = {"x": 0, "y": 1, "z": 2}

DEFAULT_GRADS_TO_COMPUTE: Final[AeroelasticGradsToCompute] = AeroelasticGradsToCompute()
DEFAULT_OPTIONAL_JACOBIANS: Final[OptionalJacobians] = OptionalJacobians(
    True, True, True, True
)


@make_pytree
class CoupledAeroelastic(BaseCoupledAeroelastic):
    def aeroelastic_states_res_from_dv_varphi(
        self,
        dv: AeroelasticDesignVariables,
        varphi: Array,
        thrust: dict[str, Array],
        i_ts: int,
        t: Array,
        use_horseshoe: bool,
    ) -> tuple[AeroelasticFullStates, Array]:
        r"""
        Obtain useful states and forcing residual from design variables and a minimal configuration vector.
        """

        # make a copy of the structure object to prevent modifying the original states
        inner_case = pytree_clone(self)

        struct_dv = dv.structure
        aero_dv = dv.aero
        struct = self.structure
        aero = self.aero
        flowfield = (
            aero.flowfield.from_design_variables(design_variables=aero_dv.flowfield)
            if aero_dv.flowfield is not None
            else aero.flowfield
        )
        inner_case.set_design_variables(
            coords=dv_or(struct_dv.x0, struct.x0),
            k_cs=dv_or(struct_dv.k_cs, struct.k_cs),
            m_cs=dv_or(struct_dv.m_cs, struct.m_cs),
            m_lumped=dv_or(struct_dv.m_lumped, struct.m_lumped)
            if struct.use_lumped_mass
            else None,
            thrust_reference=dv_or(struct_dv.thrust_t, struct.thrust_reference),
            flowfield=flowfield,
            delta_w=aero.delta_w,
            dt=aero.dt,
            x0_aero=dv_or(aero_dv.zeta_b0, aero.zeta_b0),
            orientation_euler=dv_or(
                struct_dv.orientation_euler, struct.orientation_euler
            ),
            cs_angles_reference=dv_or(aero_dv.cs_ang_t, aero.cs_ang0),
            remove_checks=True,
        )

        exp_varphi = vmap(exp_se3)(varphi.reshape(-1, 6))  # (n_nodes, 4, 4)
        hg = jnp.einsum(
            "ijk,ikl->ijl", inner_case.structure.hg0, exp_varphi
        )  # (n_nodes, 4, 4)

        # evaluate aero forcing and project to beam nodes
        aero_sol = inner_case.aero.static_solve(hg=hg, t=t, horseshoe=use_horseshoe)
        f_ext_aero_global = aero_sol.project_forcing_to_beam(
            i_ts=0,
            rmat=hg[:, :3, :3],
            x0_aero=self.aero.zeta_b0,
            include_unsteady=False,
        )

        d = inner_case.structure.make_d(hg)
        p_d = inner_case.structure.make_p_d(d)
        eps = inner_case.structure.make_eps(d)
        f_elem = inner_case.structure.make_f_elem(eps=eps)

        if inner_case.structure.use_gravity:
            m_t = inner_case.structure.make_m_t(d)
        else:
            m_t = None

        if dv.structure.f_ext_dead is not None:
            f_ext_dead = inner_case.structure.make_f_dead_ext(
                dv.structure.f_ext_dead, hg[:, :3, :3]
            )
        else:
            f_ext_dead = None

        f_dead_total = inner_case.structure.make_f_ext_dead_tot(
            f_ext_dead, f_ext_aero_global, i_load_step=None
        )

        f_res = inner_case.structure.make_f_res(
            solve_dofs=None,
            p_d=p_d,
            eps=eps,
            hg=hg,
            f_ext_follower_n=dv.structure.f_ext_follower,
            f_ext_dead_n=f_dead_total,
            thrust_n=dv.structure.thrust_t
            if dv.structure.thrust_t is not None
            else thrust,
            dynamic=False,
            m_t=m_t,
            c_l=None,
            c_l_lumped=None,
            v=None,
            v_dot=None,
        )[0]

        struct_states = StructureFullStates(
            hg=hg,
            varphi=varphi,
            eps=eps,
            f_elem=f_elem,
            f_res=f_res.reshape(-1, 6),
            v=None,
            v_dot=None,
        )

        aero_states = aero_sol.get_states(i_ts=i_ts)

        return AeroelasticFullStates(structure=struct_states, aero=aero_states), f_res

    def minimal_states_to_full_states(
        self,
        i_ts: int,
        q: AeroelasticMinimalStates,
        dv: AeroelasticDesignVariables,
        dv_full: AeroelasticDesignVariables,
    ) -> AeroelasticFullStates:
        return AeroelasticFullStates(
            structure=self.structure.minimal_states_to_full_states(
                i_ts=i_ts, q=q.structure, dv=dv.structure, dv_full=dv_full.structure
            ),
            aero=q.aero,
        )

    @jax.jit(static_argnums=(0, 1, 2, 3, 4, 5, 6))
    def static_adjoint(
        self,
        case: AeroelasticCase,
        objective: AeroelasticObjectiveFunction,
        grads_to_compute: AeroelasticGradsToCompute = DEFAULT_GRADS_TO_COMPUTE,
        optional_jacobians: OptionalJacobians | None = DEFAULT_OPTIONAL_JACOBIANS,
        ad_mode: ADMode = "forward",
        batch_size: int | None = 32,
    ) -> tuple[AeroelasticDesignVariables, Array]:
        r"""
        Computes the static grads of the structure, which is used to compute gradients of the loss with respect to
        the structure's parameters.
        :param case: AeroelasticCase containing the current state of the aeroelastic system.
        :param objective: Objective function that takes the structure and design variables and returns an array
        :param grads_to_compute: Data structure which specifies which gradients to compute. This is used to speed up the
        adjoint solve by only computing the necessary Jacobian blocks.
        :param optional_jacobians: OptionalJacobians object specifying which Jacobians to compute.
        :param ad_mode: Optional use of either forward or reverse adjoint. For passing the initial state sensitivities
        to a dynamic solve, only forward mode can be used to give the required adjoint.
        :param batch_size: Batch size for computing p_res_p_varphi to reduce memory usage. Ignored when
        ``matrix_free`` is True.
        :return: Gradient of objective function output with respect to design variables.
        """

        if ad_mode not in ("forward", "reverse"):
            raise ValueError("ad_mode must be either 'forward' or 'reverse'")

        jax_print("Computing static adjoint", verbose_level="normal")

        solve_dofs = jnp.array(
            get_solve_dofs(
                n_dof=self.structure.n_dof,
                prescribed_dofs=case.structure.prescribed_dofs,
            )
        )

        if optional_jacobians is not None:
            self.structure.optional_jacobians = optional_jacobians

        dv = self.get_design_variables(case=case, grads_to_compute=grads_to_compute)
        states = case.get_full_states()

        # find shape of objective function output without evaluating function
        f_properties = jax.eval_shape(lambda: objective(states, dv, None))
        f_shape = f_properties.shape
        j0_shape = f_shape if len(f_shape) > 0 else (1,)
        n_f = f_properties.size
        n_x = dv.structure.n_x + dv.aero.n_x
        n_u_full = self.structure.n_dof

        varphi = case.structure.varphi

        if case.aero.static_horseshoe is None:
            raise ValueError("static_horseshoe not defined")
        static_horseshoe: bool = case.aero.static_horseshoe

        # function for computing sensitivity of objective to design variables and degrees of freedom
        # to obtain the actual Jacobian we must pull back the identity through it
        vjp_fn = self.compute_p_j0_p_x(
            case=case,
            objective=objective,
            grads_to_compute=grads_to_compute,
            horseshoe=static_horseshoe,
        )

        cot_j0 = jnp.eye(n_f).reshape(n_f, *j0_shape)  # seed for backpropogation
        p_j_p_varphi_raw, p_j_p_x_raw = jax.vmap(vjp_fn)(cot_j0)  # sensitivities

        p_j_p_varphi_flat = p_j_p_varphi_raw.reshape(n_f, -1)  # (n_f, n_dof)
        p_j_p_x_flat = AeroelasticDesignVariables(
            structure_dv=StructureDesignVariables(
                **{
                    k: getattr(p_j_p_x_raw.structure, k) for k in dv.structure.to_dict()
                },
                f_shape=(n_f,),
            ),
            aero_dv=AeroDesignVariables(
                **{k: getattr(p_j_p_x_raw.aero, k) for k in dv.aero.to_dict()},
                f_shape=(n_f,),
            ),
        ).ravel_jacobian(f_size=n_f, x_size=n_x)

        def _residual(varphi_vec: Array, dv_: AeroelasticDesignVariables) -> Array:
            r"""
            Helper function to give the static aeroelastic residual for a given deformation and design variables.
            """
            return self.aeroelastic_states_res_from_dv_varphi(
                dv=dv_,
                varphi=varphi_vec.reshape(self.structure.n_nodes, 6),
                thrust=case.structure.thrust,
                t=case.aero.t,
                i_ts=0,
                use_horseshoe=static_horseshoe,
            )[1]

        if ad_mode == "forward":
            # single joint VJP shared between p_res_p_varphi and p_res_p_x
            _, vjp_res_both = jax.vjp(_residual, varphi.ravel(), dv)
            p_res_p_varphi, p_res_p_x = jax.lax.map(
                vjp_res_both,
                jnp.eye(n_u_full),
                batch_size=batch_size,
            )
            # solve for adjoint
            adj = jnp.linalg.solve(
                p_res_p_varphi[jnp.ix_(solve_dofs, solve_dofs)],
                p_res_p_x.ravel_jacobian(f_size=n_u_full, x_size=n_x)[solve_dofs, :],
            )

            d_f_d_x_dict = dv.from_adjoint(
                f_shape,
                p_j_p_x_flat - p_j_p_varphi_flat[:, solve_dofs] @ adj,
            )
        else:
            # construct residual Jacobian
            _, vjp_res_varphi = jax.vjp(lambda v: _residual(v, dv), varphi.ravel())
            p_res_p_varphi = jax.lax.map(
                lambda cot: vjp_res_varphi(cot)[0],
                jnp.eye(n_u_full),
                batch_size=batch_size,
            )
            adj = jnp.linalg.solve(
                p_res_p_varphi[jnp.ix_(solve_dofs, solve_dofs)].T,
                p_j_p_varphi_flat[:, solve_dofs].T,
            ).T  # (n_f, n_solve_dofs)

            adj_full = (
                jnp.zeros((n_f, n_u_full), dtype=adj.dtype).at[:, solve_dofs].set(adj)
            )

            # sensitivity of residual w.r.t. design variables
            _, vjp_res_dv = jax.vjp(lambda dv_: _residual(varphi.ravel(), dv_), dv)
            (adj_p_res_p_x_raw,) = jax.vmap(vjp_res_dv)(adj_full)

            adj_p_res_p_x_flat = AeroelasticDesignVariables(
                structure_dv=StructureDesignVariables(
                    **{
                        k: getattr(adj_p_res_p_x_raw.structure, k)
                        for k in dv.structure.to_dict()
                    },
                    f_shape=(n_f,),
                ),
                aero_dv=AeroDesignVariables(
                    **{
                        k: getattr(adj_p_res_p_x_raw.aero, k) for k in dv.aero.to_dict()
                    },
                    f_shape=(n_f,),
                ),
            ).ravel_jacobian(f_size=n_f, x_size=n_x)

            d_f_d_x_dict = dv.from_adjoint(
                f_shape,
                p_j_p_x_flat - adj_p_res_p_x_flat,
            )

        return dv.split_adjoint(d_f_d_x=d_f_d_x_dict, f_shape=f_shape), adj

    def compute_p_j0_p_x(
        self,
        case: AeroelasticCase,
        objective: AeroelasticObjectiveFunction,
        grads_to_compute: AeroelasticGradsToCompute | None,
        horseshoe: bool = False,
        include_q0: bool = False,
    ) -> Callable[
        ...,
        tuple[Array, AeroelasticDesignVariables],
    ]:
        r"""
        Build the VJP of the initial-timestep objective for pertubations in the design variables, and optionally the
        initial states.
        :param case: AeroelasticCase solution for the initial timestep.
        :param objective: Objective function which takes the system full states, design variables and timestep index.
        :param grads_to_compute: Grads to compute when computing design gradient.
        :param horseshoe: Flag for using horseshoe wake.
        :param include_q0: If True, the returned VJP also propagates a cotangent through ``q0``.
        :return: VJP for cotangents of the design variables, and optionally the initial states.
        """

        # design variables with variables that we don't require gradients omitted to speed up computations.
        dv = self.get_design_variables(case=case, grads_to_compute=grads_to_compute)

        # design variables with no omissions
        dv_full = self.get_design_variables(case=case, grads_to_compute=None)

        varphi = case.structure.varphi
        n_dof = self.structure.n_dof

        @jax.checkpoint
        def objective_from_varphi(
            varphi_: Array,
            dv_: AeroelasticDesignVariables,
        ) -> Array | tuple[Array, Array]:
            inner_case = self.case_from_dv(dv=dv_)

            assert (
                dv_full.aero.cs_ang_t is not None and dv_full.aero.cs_vel_t is not None
            )

            # solve aero problem
            hg = inner_case.structure.compute_hg_from_varphi(varphi=varphi_)
            _, _, gamma_b, gamma_w, _, _, zeta_w, _, f_steady, _, _, _, _, _ = (
                inner_case.aero.base_solve(
                    q_nm1=None,
                    t_n=case.aero.t,
                    hg_n=hg,
                    hg_nm1=None,
                    hg_dot_n=None,
                    static=True,
                    horseshoe=horseshoe,
                    cs_ang_n={
                        k: jnp.atleast_1d(v)[0]
                        for k, v in (
                            dv_.aero.cs_ang_t
                            if dv_.aero.cs_ang_t is not None
                            else dv_full.aero.cs_ang_t
                        ).items()
                    },
                    cs_ang_nm1=None,
                    cs_vel_n={
                        k: jnp.atleast_1d(v)[0]
                        for k, v in (
                            dv_.aero.cs_vel_t
                            if dv_.aero.cs_vel_t is not None
                            else dv_full.aero.cs_vel_t
                        ).items()
                    },
                )
            )

            f_aero_beam_global = project_forcing_to_beam(
                f_total=f_steady,
                rmat=hg[:, :3, :3],
                dof_mapping=inner_case.aero.dof_mapping,
                x0_aero=inner_case.aero.zeta_b0,
                mirror_edge_low=inner_case.aero.mirror_edge_low,
                mirror_edge_high=inner_case.aero.mirror_edge_high,
            )

            f_aero_beam_local = transform_nodal_vect(
                vect=f_aero_beam_global, rmat=jnp.transpose(hg[:, :3, :3], (0, 2, 1))
            )

            q_aero = AeroFullStates(
                gamma_b=gamma_b,
                gamma_w=gamma_w,
                zeta_w=zeta_w,
                gamma_b_dot=ArrayList.zeros_like(gamma_b),
            )

            # assume initial velocities and accelerations are zero
            q_structure = StructureMinimalStates(
                varphi=varphi_,
                v=jnp.zeros_like(varphi_),
                v_dot=jnp.zeros_like(varphi_),
                a=jnp.zeros_like(varphi_),
                f_ext_aero=f_aero_beam_local,
            )

            q0 = AeroelasticMinimalStates(structure=q_structure, aero=q_aero).ravel()
            q0_full = self.minimal_states_to_full_states(
                i_ts=0,
                q=AeroelasticMinimalStates.from_vector(
                    vect=q0,
                    n_dof=n_dof,
                    aero_shapes=q_aero.shapes(),
                ),
                dv=dv_,
                dv_full=dv_full,
            )
            j0 = jnp.atleast_1d(objective(q0_full, dv_, 0))
            if include_q0:
                return j0, q0
            return j0

        _, vjp_fn = jax.vjp(objective_from_varphi, varphi, dv)
        return vjp_fn

    def _initial_timestep_grad_contribution(
        self,
        case: AeroelasticCase,
        objective: AeroelasticObjectiveFunction,
        grads_to_compute: AeroelasticGradsToCompute | None,
        p_varphi_p_x: Array | None,
        solve_dofs: Array,
        adj_t_p_r1_p_q0: Array,
        horseshoe: bool = False,
    ) -> AeroelasticDesignVariables:
        r"""
        Assemble the initial-timestep contribution to ``d_j_d_x``.
        :param case: AeroelasticCase object.
        :param objective: Aeroelastic objective function.
        :param grads_to_compute: Class outlining the gradients to be computed.
        :param p_varphi_p_x: Sensitivity of initial deformation to design variables.
        :param solve_dofs: Array of dofs to be solved.
        :param adj_t_p_r1_p_q0: Initial adjoint-Jacobian product adj^T @ p_r1_p_q0, which is returned from the time
        domain adjoint propagation.
        :param horseshoe: Whether to use horseshoe or not for the VLM.
        :return: Assembled initial timestep contribution.
        """
        dv = self.get_design_variables(case=case, grads_to_compute=grads_to_compute)
        vjp_fn = self.compute_p_j0_p_x(
            case=case,
            objective=objective,
            grads_to_compute=grads_to_compute,
            horseshoe=horseshoe,
            include_q0=True,
        )

        n_j = adj_t_p_r1_p_q0.shape[0]
        f_shape = (n_j,)
        adj_varphi, adj_dv_raw = jax.vmap(vjp_fn)((jnp.eye(n_j), adj_t_p_r1_p_q0))

        # direct term
        adj_dv = AeroelasticDesignVariables(
            structure_dv=StructureDesignVariables(
                **{k: getattr(adj_dv_raw.structure, k) for k in dv.structure.to_dict()},
                f_shape=f_shape,
            ),
            aero_dv=AeroDesignVariables(
                **{k: getattr(adj_dv_raw.aero, k) for k in dv.aero.to_dict()},
                f_shape=f_shape,
            ),
        )

        # indirect term
        if p_varphi_p_x is not None:
            indirect_mat = (
                adj_varphi.reshape(n_j, -1)[:, solve_dofs] @ p_varphi_p_x
            )  # [n_j, n_x]
            indirect_dict = dv.from_adjoint(f_shape, indirect_mat)
            adj_dv += AeroelasticDesignVariables(
                structure_dv=StructureDesignVariables(
                    **{k: indirect_dict[k] for k in dv.structure.to_dict()},
                    f_shape=f_shape,
                ),
                aero_dv=AeroDesignVariables(
                    **{k: indirect_dict[k] for k in dv.aero.to_dict()},
                    f_shape=f_shape,
                ),
            )

        return adj_dv

    def timestep_residual(
        self,
        i_ts: int | Array,
        t: Array,
        q_nm1: AeroelasticMinimalStates,
        q_n: AeroelasticMinimalStates,
        dv_: AeroelasticDesignVariables,
        dv_full: AeroelasticDesignVariables,
        thrust_t: dict[str, Array],
        solve_dofs: tuple[int, ...],
        approx_grads: bool,
    ) -> Array:
        r"""
        Compute the coupled aeroelastic residual vector. This is used for the matrix-free time domain case by applying
        VJP to this function.
        :param i_ts: Time step index.
        :param t: Time at step n.
        :param q_nm1: Minimal states at step n-1.
        :param q_n: Minimal states at step n.
        :param dv_: Aeroelastic design variables (may have some fields omitted).
        :param dv_full: Aeroelastic design variables without omissions.
        :param thrust_t: Thrust time history, {key: [n_tstep]}.
        :param solve_dofs: Structural degrees of freedom which are solved for.
        :param approx_grads: If True, remove some gradient terms which are generally small.
        :return: Coupled residual ``(n_adj_dof, )``.
        """
        assert (
            q_n.structure.f_ext_aero is not None
            and q_nm1.structure.f_ext_aero is not None
        )

        struct_res = self.structure.timestep_residual(
            i_ts=i_ts,
            q_nm1=q_nm1.structure,
            q_n=q_n.structure,
            dv_=dv_.structure,
            thrust_t=thrust_t,
            solve_dofs=solve_dofs,
            approx_grads=approx_grads,
        )

        # Rematerialise the aero pass to reduce memory usage
        @jax.checkpoint
        def _aero_forward(
            varphi_nm1_: Array,
            varphi_n_: Array,
            v_n_: Array,
            t_n_: Array,
            q_nm1_aero_: AeroFullStates,
            q_n_aero_: AeroFullStates,
            dv__: AeroelasticDesignVariables,
            dv_full_: AeroelasticDesignVariables,
            f_aero_beam_n_: Array,
        ) -> Array:
            return self.aero.timestep_residual(
                i_ts=i_ts,
                varphi_nm1=varphi_nm1_,
                varphi_n=varphi_n_,
                v_n=v_n_,
                t_n=t_n_,
                q_n=q_n_aero_,
                q_nm1=q_nm1_aero_,
                dv=dv__,
                dv_full=dv_full_,
                f_aero_beam_n=f_aero_beam_n_,
                struct_obj=self.structure,
                approx_grads=approx_grads,
            )

        # evaluate checkpointed function
        aero_res = _aero_forward(
            varphi_nm1_=q_nm1.structure.varphi,
            varphi_n_=q_n.structure.varphi,
            v_n_=q_n.structure.v,
            t_n_=t,
            q_nm1_aero_=q_nm1.aero,
            q_n_aero_=q_n.aero,
            dv__=dv_,
            dv_full_=dv_full,
            f_aero_beam_n_=q_n.structure.f_ext_aero,
        )

        # remove forces from degrees of freedom which are not solved for
        n_aero_states: int = q_n.aero.n_states
        solve_dofs_arr = jnp.array(solve_dofs)
        aero_res_solve = jnp.concatenate(
            (aero_res[:n_aero_states], aero_res[n_aero_states:][solve_dofs_arr])
        )

        return jnp.concatenate((struct_res, aero_res_solve))

    def timestep_residual_jacobians(
        self,
        i_ts: int | Array,
        t: Array,
        q_nm1: AeroelasticMinimalStates,
        q_n: AeroelasticMinimalStates,
        dv_: AeroelasticDesignVariables,
        dv_full: AeroelasticDesignVariables,
        thrust_t: dict[str, Array],
        solve_dofs: tuple[int, ...],
        approx_grads: bool,
        n_profile_loops: int | None,
        jac_options: dict,
        mode: ADMode = "reverse",
        map_batch_size: int | None = None,
    ) -> tuple[
        Array,
        Array,
        StructureDesignVariables,
        AeroelasticDesignVariables,
        dict[str, dict[str, float]] | None,
        dict[str, dict[str, float]] | None,
    ]:
        r"""
        Compute the required time-domain residual Jacobians for the adjoint solution.
        :param i_ts: Time step index.
        :param t: Time.
        :param q_nm1: Minimal degrees of freedom at timestep n-1.
        :param q_n: Minimal degrees of freedom at timestep n.
        :param dv_: Design variables for which to obtain gradients.
        :param dv_full: All design variables.
        :param thrust_t: Thrust at each time step.
        :param solve_dofs: Degrees of freedom to solve for.
        :param approx_grads: Whether to use approximate gradients for the structural dynamic subproblem.
        :param n_profile_loops: Number of profile loops. Used for profiling routines only.
        :param jac_options: Options for Jacobian computation, allowing for approximations to be introduced.
        :param mode: Mode for automatic differentiation, either ``forward`` or ``reverse``.
        :param map_batch_size: Batch size used for vectorising Jacobian construction.
        :return: Jacobian of residual with respect to previous degrees of freedom, Jacobian of residual with respect to
        current degrees of freedom. Jacobian of v_dot residual with respect to structural design variables, Jacobian of
        aero residual with respect to design variables. Can also include compile time and run time when profiling is
        used.
        """

        assert (
            q_n.structure.f_ext_aero is not None
            and q_nm1.structure.f_ext_aero is not None
        )

        (
            p_aero_res_p_q_aero_nm1,
            p_aero_res_p_q_aero_n,
            p_aero_res_d_dv,
            p_aero_res_p_q_struct_nm1,
            p_aero_res_p_q_struct_n,
            aero_compile_time,
            aero_run_time,
        ) = self.aero.timestep_residual_jacobians(
            i_ts=i_ts,
            varphi_nm1=q_nm1.structure.varphi,
            varphi_n=q_n.structure.varphi,
            v_n=q_n.structure.v,
            t_n=t,
            q_n=q_n.aero,
            q_nm1=q_nm1.aero,
            dv=dv_,
            dv_full=dv_full,
            f_aero_beam_n=q_n.structure.f_ext_aero,
            struct_obj=self.structure,
            approx_grads=approx_grads,
            solve_dofs=solve_dofs,
            n_profile_loops=n_profile_loops,
            jac_options=jac_options,
            mode=mode,
            map_batch_size=map_batch_size,
        )

        n_aero_dof, _ = p_aero_res_p_q_struct_nm1.shape

        (
            p_struct_res_p_q_struct_nm1,
            p_struct_res_p_q_struct_n,
            p_v_dot_res_p_struct_dv,
            p_v_dot_res_p_f_ext_nm1,
            p_v_dot_res_p_f_ext_n,
            struct_compile_time,
            struct_run_time,
        ) = self.structure.timestep_residual_jacobians(
            i_ts=i_ts,
            q_nm1=q_nm1.structure,
            q_n=q_n.structure,
            f_ext_aero_n=q_n.structure.f_ext_aero,
            f_ext_aero_nm1=q_nm1.structure.f_ext_aero,
            thrust_t=thrust_t,
            dv=dv_.structure,
            solve_dofs=solve_dofs,
            approx_grads=approx_grads,
            n_profile_loops=n_profile_loops,
            jac_options=jac_options,
            mode=mode,
        )

        assert p_v_dot_res_p_f_ext_nm1 is not None and p_v_dot_res_p_f_ext_n is not None

        # reduce aero-to-struct Jacobian columns to solve_dofs
        n_solve = len(solve_dofs)
        n_struct_res = p_struct_res_p_q_struct_nm1.shape[0]
        solve_dofs_arr = jnp.array(solve_dofs)
        struct_col_ix = jnp.concatenate(
            [solve_dofs_arr + i * self.structure.n_dof for i in range(4)]
        )
        p_aero_res_p_q_struct_nm1 = p_aero_res_p_q_struct_nm1[:, struct_col_ix]
        p_aero_res_p_q_struct_n = p_aero_res_p_q_struct_n[:, struct_col_ix]

        # create struct-to-aero cross-coupling (only v_dot residual depends on f_ext_aero)
        # f_aero block is n_solve-wide (last n_solve cols of aero state), so slice Jacobian to solve_dofs cols
        p_struct_res_p_q_aero_nm1 = jnp.zeros((n_struct_res, n_aero_dof))
        p_struct_res_p_q_aero_nm1 = p_struct_res_p_q_aero_nm1.at[
            jnp.arange(n_solve) + 2 * n_solve, -n_solve:
        ].set(p_v_dot_res_p_f_ext_nm1[:, solve_dofs_arr])

        p_struct_res_p_q_aero_n = jnp.zeros((n_struct_res, n_aero_dof))
        p_struct_res_p_q_aero_n = p_struct_res_p_q_aero_n.at[
            jnp.arange(n_solve) + 2 * n_solve, -n_solve:
        ].set(p_v_dot_res_p_f_ext_n[:, solve_dofs_arr])

        p_res_p_q_nm1 = jnp.block(
            [
                [p_struct_res_p_q_struct_nm1, p_struct_res_p_q_aero_nm1],
                [p_aero_res_p_q_struct_nm1, p_aero_res_p_q_aero_nm1],
            ]
        )

        p_res_p_q_n = jnp.block(
            [
                [p_struct_res_p_q_struct_n, p_struct_res_p_q_aero_n],
                [p_aero_res_p_q_struct_n, p_aero_res_p_q_aero_n],
            ]
        )

        if n_profile_loops is not None:
            assert (
                aero_compile_time is not None
                and aero_run_time is not None
                and struct_compile_time is not None
                and struct_run_time is not None
            )
            compile_time = aero_compile_time | struct_compile_time
            run_time = aero_run_time | struct_run_time
        else:
            compile_time = None
            run_time = None
        return (
            p_res_p_q_nm1,
            p_res_p_q_n,
            p_v_dot_res_p_struct_dv,
            p_aero_res_d_dv,
            compile_time,
            run_time,
        )

    def construct_approximate_jacobians(
        self,
        sol: AeroelasticCase,
        jacobian_approximations: AeroelasticJacobianApproximations,
    ) -> dict[str, dict[str, Callable[..., Any] | None]]:
        r"""
        Compute approximations for Jacobians which are specified in the jacobian_approximations data structure. The
        aerodynamic residual approximations are delegated to ``UVLM.construct_approximate_jacobians``, and the
        structural residual approximations to ``BeamStructure.construct_approximate_jacobians``. The two
        dictionaries are merged into a single result.
        :param sol: Solution for which approximations will be created for the initial time step.
        :param jacobian_approximations: Data structure which defines which approximations to create.
        :return: Dictionary of approximations keyed by residual name.
        """
        dv = self.get_design_variables(case=sol, grads_to_compute=None)
        solve_dofs = get_solve_dofs(
            n_dof=self.structure.n_dof,
            prescribed_dofs=sol.structure.prescribed_dofs,
        )

        aero_options = self.aero.construct_approximate_jacobians(
            aero_sol=sol.aero,
            structure_sol=sol.structure,
            struct_obj=self.structure,
            dv=dv,
            dv_full=dv,
            solve_dofs=solve_dofs,
            jacobian_approximations=jacobian_approximations.aero,
        )

        struct_options = self.structure.construct_approximate_jacobians(
            sol=sol.structure,
            jacobian_approximations=jacobian_approximations.structure,
        )

        return aero_options | struct_options

    def evaluate_dynamic_objective(
        self, case: AeroelasticCase, objective: AeroelasticObjectiveFunction
    ) -> Array:
        r"""
        Evaluate the dynamic objective for a given case.
        :param case: Dynamic aeroelastic case object.
        :param objective: Objective function to be evaluated.
        :return: Value of subobjective at every time step, [n_tstep].
        """
        n_tstep = case.structure.n_tstep

        dv = self.get_design_variables(case=case, grads_to_compute=None)

        return jax.vmap(
            lambda i_ts: jnp.atleast_1d(
                objective(case.get_full_states(i_ts=i_ts), dv, i_ts)
            )
        )(jnp.arange(n_tstep)).reshape(n_tstep, -1)

    def make_frozen_wake_preconditioner(
        self,
        case: AeroelasticCase,
        dv_full: AeroelasticDesignVariables,
        solve_dofs: tuple[int, ...],
        approx_grads: bool = False,
        precond_i_ts: int = 0,
        batch_size: int | None = 32,
    ) -> Callable[[Array], Array]:
        r"""
        Build a preconditioner for the coupled aeroelastic system which skips the wake grid and circulation. When
        applied to the matrix-free system GMRES, this found a good reduction in the number of iterations required whilst
        avoiding the large memory overhead involved in computing the full Jacobian due to the large number of wake
        states.
        :param case: Dynamic aeroelastic case object.
        :param dv_full: Full dynamic aeroelastic design variables.
        :param solve_dofs: Solve degree of freedom index.
        :param approx_grads: Approximate gradient of the coupled aeroelastic system, removing some negligible terms.
        :param precond_i_ts: Time step index for which to create the preconditioner. Defaults to 0.
        :param batch_size: Batch size for mapping the Jacobian construction on the aerodynamic system.
        :return: Preconditioner function.
        """
        precond_q_nm1 = case.get_minimal_states(i_ts=max(precond_i_ts - 1, 0))
        precond_q_n = case.get_minimal_states(i_ts=precond_i_ts)
        precond_t_n = case.structure.t[precond_i_ts]

        dv_precond = self.get_design_variables(case=case, grads_to_compute=None)

        assert precond_q_n.structure.f_ext_aero is not None
        assert precond_q_nm1.structure.f_ext_aero is not None

        # populate jac_options with the expected residual/argname keys, all mapped to None so AD is used everywhere
        jac_options = self.construct_approximate_jacobians(
            sol=case,
            jacobian_approximations=AeroelasticJacobianApproximations(),
        )

        # compute structural Jacobians
        (
            _,
            p_struct_res_p_q_struct_n,
            _,
            _,
            p_v_dot_res_p_f_ext_n,
            *_,
        ) = self.structure.timestep_residual_jacobians(
            i_ts=precond_i_ts,
            q_nm1=precond_q_nm1.structure,
            q_n=precond_q_n.structure,
            f_ext_aero_n=precond_q_n.structure.f_ext_aero,
            f_ext_aero_nm1=precond_q_nm1.structure.f_ext_aero,
            thrust_t=case.structure.thrust,
            dv=dv_precond.structure,
            solve_dofs=solve_dofs,
            approx_grads=approx_grads,
            n_profile_loops=None,
            jac_options=jac_options,
        )

        # compute aero Jacobians
        (
            _,
            p_aero_res_p_q_aero_n,
            _,
            _,
            p_aero_res_p_q_struct_n,
            *_,
        ) = self.aero.timestep_residual_jacobians(
            i_ts=precond_i_ts,
            varphi_nm1=precond_q_nm1.structure.varphi,
            varphi_n=precond_q_n.structure.varphi,
            v_n=precond_q_n.structure.v,
            t_n=precond_t_n,
            q_n=precond_q_n.aero,
            q_nm1=precond_q_nm1.aero,
            dv=dv_precond,
            dv_full=dv_full,
            f_aero_beam_n=precond_q_n.structure.f_ext_aero,
            struct_obj=self.structure,
            approx_grads=approx_grads,
            solve_dofs=solve_dofs,
            n_profile_loops=None,
            jac_options=jac_options,
            compute_wake_gradients=False,
            map_batch_size=batch_size,
        )

        assert p_v_dot_res_p_f_ext_n is not None

        solve_dofs_arr = jnp.array(solve_dofs)
        struct_col_ix = jnp.concatenate(
            [solve_dofs_arr + i * self.structure.n_dof for i in range(4)]
        )
        p_aero_res_p_q_struct_n = p_aero_res_p_q_struct_n[:, struct_col_ix]

        # assemble Jacobians
        n_solve = len(solve_dofs)
        n_struct_res = p_struct_res_p_q_struct_n.shape[0]
        n_aero_reduced = p_aero_res_p_q_aero_n.shape[0]
        p_struct_res_p_q_aero_n = jnp.zeros((n_struct_res, n_aero_reduced))
        p_struct_res_p_q_aero_n = p_struct_res_p_q_aero_n.at[
            jnp.arange(n_solve) + 2 * n_solve, -n_solve:
        ].set(p_v_dot_res_p_f_ext_n[:, solve_dofs_arr])

        p_res_p_q_n_reduced = jnp.block(
            [
                [p_struct_res_p_q_struct_n, p_struct_res_p_q_aero_n],
                [p_aero_res_p_q_struct_n, p_aero_res_p_q_aero_n],
            ]
        )

        # compute the LU decomposition for fast reuse when solving
        precond_lu = jax.scipy.linalg.lu_factor(p_res_p_q_n_reduced.T)

        # adjoint state counts and placement
        n_struct = 4 * n_solve
        n_gamma_b = int(precond_q_n.aero.gamma_b.ravel().size)
        n_gamma_w = int(precond_q_n.aero.gamma_w.ravel().size)
        n_gamma_b_dot = int(precond_q_n.aero.gamma_b_dot.ravel().size)
        n_zeta_w = int(precond_q_n.aero.zeta_w.ravel().size)

        gamma_w_start = n_struct + n_gamma_b
        gamma_w_end = gamma_w_start + n_gamma_w
        zeta_w_start = gamma_w_end + n_gamma_b_dot
        zeta_w_end = zeta_w_start + n_zeta_w

        def apply_precond(vec: Array) -> Array:
            # split the system to remove gamma_w and zeta_w. The removed blocks use the negative identity as preconditioner.
            vec_pre_gw = vec[:gamma_w_start]  # struct + gamma_b
            vec_gw = vec[gamma_w_start:gamma_w_end]  # gamma_w (identity)
            vec_mid = vec[gamma_w_end:zeta_w_start]  # gamma_b_dot
            vec_zw = vec[zeta_w_start:zeta_w_end]  # zeta_w (identity)
            vec_post_zw = vec[zeta_w_end:]  # f_ext_aero
            vec_reduced = jnp.concatenate([vec_pre_gw, vec_mid, vec_post_zw])
            x_reduced = jax.scipy.linalg.lu_solve(precond_lu, vec_reduced)

            x_pre_gw = x_reduced[:gamma_w_start]
            x_mid = x_reduced[gamma_w_start : gamma_w_start + n_gamma_b_dot]
            x_post_zw = x_reduced[gamma_w_start + n_gamma_b_dot :]

            return jnp.concatenate([x_pre_gw, -vec_gw, x_mid, -vec_zw, x_post_zw])

        return apply_precond

    @jax.jit(static_argnums=(0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17))
    def dynamic_adjoint(
        self,
        case: AeroelasticCase,
        objective: AeroelasticObjectiveFunction,
        matrix_free: bool = True,
        jacobian_approximations: AeroelasticJacobianApproximations | None = None,
        grads_to_compute: AeroelasticGradsToCompute | None = DEFAULT_GRADS_TO_COMPUTE,
        p_varphi_p_x: Array | None = None,
        save_adjoint: bool = False,
        approx_grads: bool = True,
        i_ts_adjoint_range: tuple[int | None, int | None] = (None, None),
        include_initial_state_grad: bool = True,
        gmres_mode: Literal["batched", "incremental"] = "incremental",
        gmres_warm_start: bool = True,
        gmres_precond: bool = True,
        gmres_restart: int = 50,
        i_ts_preconditioner: int = 0,
        preconditioner_batch_size: int | None = 16,
        preconditioner: Callable[[Array], Array] | None = None,
    ) -> tuple[AeroelasticDesignVariables, Array, Array | None]:
        r"""
        Compute the adjoint of a coupled dynamic aeroelastic system.
        :param case: Dynamic aeroelastic case
        :param objective: Objective function that takes the system full states, design variables and timestep index,
        and returns an array.
        :param matrix_free: If true, do not explicitly compute the residual Jacobians and instead use the VJP and GMRES
        to solve.
        :param jacobian_approximations: Data structure which specifies Jacobian approximations to use for each part of
        the problem.
        :param grads_to_compute: Specify which design variables for which to compute gradients for. If None, all
        available gradients are computed.
        :param p_varphi_p_x: Gradient of initial twists with respect to design variables. In practice, this is found
        from the static solve.
        :param save_adjoint: Whether to save the adjoint of the dynamic aeroelastic system.
        :param approx_grads: Whether to use gradient approximation or not. This removes some negligible contributions
        in the structural dynamic system.
        :param i_ts_adjoint_range: Optional ``(start, end)`` window of time steps for which to compute the adjoint.
         Either entry may be ``None`` to leave that side untruncated. When ``start > 1`` the initial-state gradient
         contribution is automatically skipped.
        :param include_initial_state_grad: If False, skip the ``_initial_timestep_grad_contribution`` call that solves
        the static adjoint at ``t = 0`` and propagates ``p_varphi_p_x``. Intended for profiling only.
        :param gmres_mode: If using matrix free, sets the mode for GMRES. Batched is preferred for GPU, whereas
        incremental may be preferred on CPU.
        :param gmres_warm_start: If True, use the previous timestep adjoint vector as the first guess for the current
        value. Otherwise, initialise with the zero vector.
        :param gmres_precond: If True and ``preconditioner`` is None, build the frozen-wake preconditioner internally.
        Ignored when ``preconditioner`` is supplied.
        :param gmres_restart: Number of times to restart the GMRES algorithm.
        :param i_ts_preconditioner: Timestep at which to build the preconditioner if requested.
        :param preconditioner_batch_size: Batch size for creating the Jacobians for the preconditioner. Ignored if
        ``preconditioner`` is supplied.
        :param preconditioner: Optional pass a prebuild preconditioner. Useful for profiling.
        :return: Gradient of sum of objective across timesteps with respect to design variables, objective at each time
        step, and optional adjoint states.
        """

        # make copies to prevent contaminating input object with tracer
        case = deepcopy(case)
        p_varphi_p_x = deepcopy(p_varphi_p_x)

        solve_dofs: tuple[int, ...] = get_solve_dofs(
            n_dof=self.structure.n_dof,
            prescribed_dofs=case.structure.prescribed_dofs,
        )

        solve_dofs_arr: Array = jnp.array(solve_dofs)

        n_tstep = case.structure.n_tstep

        dv = self.get_design_variables(case=case, grads_to_compute=grads_to_compute)

        dv_full = self.get_design_variables(case=case, grads_to_compute=None)

        assert case.aero.static_horseshoe is not None
        static_horseshoe: bool = case.aero.static_horseshoe

        full_states_init = case.get_full_states(i_ts=0)
        minimal_states_init = case.get_minimal_states(i_ts=0)

        j_properties = jax.eval_shape(
            lambda: jnp.atleast_1d(objective(full_states_init, dv, 0))
        )
        j_shape = j_properties.shape
        n_j = j_properties.size

        n_solve = len(solve_dofs)

        j_eval = jax.vmap(
            lambda i_ts: jnp.atleast_1d(
                objective(case.get_full_states(i_ts=i_ts), dv, i_ts)
            )
        )(jnp.arange(n_tstep)).reshape(n_tstep, n_j)

        jac_options = self.construct_approximate_jacobians(
            sol=case,
            jacobian_approximations=jacobian_approximations
            if jacobian_approximations is not None
            else AeroelasticJacobianApproximations(),
        )

        # define adjoint window
        i_ts_start_adj, i_ts_end_adj = i_ts_adjoint_range
        i_ts_start_adj_: int = 1 if i_ts_start_adj is None else i_ts_start_adj
        i_ts_end_adj_: int = n_tstep - 1 if i_ts_end_adj is None else i_ts_end_adj
        if i_ts_start_adj_ < 1:
            raise ValueError(
                f"i_ts_adjoint_range start must be >= 1, got {i_ts_start_adj_}"
            )
        if i_ts_end_adj_ > n_tstep - 1:
            raise ValueError(
                f"i_ts_adjoint_range end must be <= n_tstep - 1 = {n_tstep - 1}, got "
                f"{i_ts_end_adj_}"
            )
        if i_ts_end_adj_ < i_ts_start_adj_:
            raise ValueError(
                f"i_ts_adjoint_range end ({i_ts_end_adj_}) must be >= start "
                f"({i_ts_start_adj_})"
            )
        n_adj_iters: int = i_ts_end_adj_ - i_ts_start_adj_ + 1

        @jax.jit
        def objective_jacobians(
            i_ts: int, q_n: AeroelasticMinimalStates
        ) -> tuple[Array, AeroelasticDesignVariables]:
            # function to obtain the Jacobians of the objective w.r.t. the minimal states and the design variables
            p_j_n_p_q_n, p_j_n_p_x = jax.jacrev(
                lambda q_free, dv__: jnp.atleast_1d(
                    objective(
                        self.minimal_states_to_full_states(
                            i_ts=i_ts,
                            q=AeroelasticMinimalStates.from_vector(
                                vect=q_n.ravel().at[free_state_ix].set(q_free),
                                n_dof=self.structure.n_dof,
                                aero_shapes=minimal_states_init.aero.shapes(),
                            ),
                            dv=dv__,
                            dv_full=dv_full,
                        ),
                        dv__,
                        i_ts,
                    )
                ),
                argnums=(0, 1),
                allow_int=True,
            )(q_n.ravel()[free_state_ix], dv)
            return p_j_n_p_q_n, p_j_n_p_x

        assert dv_full.aero.cs_ang_t is not None and dv_full.aero.cs_vel_t is not None

        # create the initial d_j_d_x sensitivites which are accumulated though the solve process
        dv_grad_init = AeroelasticDesignVariables.zeros(
            system=self, case=case, grads_to_compute=grads_to_compute, j_shape=j_shape
        )

        n_dof: int = self.structure.n_dof
        n_aero_states: int = minimal_states_init.aero.n_states
        free_state_ix: Array = jnp.concatenate(
            [solve_dofs_arr + i * n_dof for i in range(4)]
            + [jnp.arange(5 * n_dof, 5 * n_dof + n_aero_states)]
            + [solve_dofs_arr + 4 * n_dof]
        )
        n_adj_dof = 4 * n_solve + n_aero_states + n_solve

        adj_full_init: Array | None = (
            jnp.zeros((case.structure.n_tstep + 1, n_j, n_adj_dof))
            if save_adjoint
            else None
        )

        d_j_d_x: AeroelasticDesignVariables
        if matrix_free:
            if preconditioner is not None:
                apply_precond = preconditioner
            elif gmres_precond:
                apply_precond = self.make_frozen_wake_preconditioner(
                    case=case,
                    dv_full=dv_full,
                    solve_dofs=solve_dofs,
                    approx_grads=approx_grads,
                    precond_i_ts=i_ts_preconditioner,
                    batch_size=preconditioner_batch_size,
                )
                jax_print(
                    "Built frozen-wake preconditioner",
                    verbose_level="normal",
                )
            else:
                apply_precond = None

            def matrix_free_body(
                rev_i_ts_: int,
                carry: tuple[AeroelasticDesignVariables, Array, Array, Array],
            ) -> tuple[AeroelasticDesignVariables, Array, Array, Array]:
                d_j_d_x_, adj_np1, adj_t_p_r_np1_p_q_n, adj_full_ = carry

                i_ts = i_ts_end_adj_ - rev_i_ts_
                i_ts_nm1 = jnp.maximum(i_ts - 1, 0)
                q_nm1 = case.get_minimal_states(i_ts=i_ts_nm1)
                q_n = case.get_minimal_states(i_ts=i_ts)
                t_n = case.structure.t[i_ts]

                p_j_n_p_q_n, p_j_n_p_x = objective_jacobians(i_ts=i_ts, q_n=q_n)

                def _residual_all(
                    q_n_: AeroelasticMinimalStates,
                    q_nm1_: AeroelasticMinimalStates,
                    dv_: AeroelasticDesignVariables,
                ) -> Array:
                    return self.timestep_residual(
                        i_ts=i_ts,
                        t=t_n,
                        q_nm1=q_nm1_,
                        q_n=q_n_,
                        dv_=dv_,
                        dv_full=dv_full,
                        thrust_t=case.structure.thrust,
                        solve_dofs=solve_dofs,
                        approx_grads=approx_grads,
                    )

                # single VJP shared between the GMRES matvec, the coupling term and the design-variable pull
                _, pull_all = jax.vjp(_residual_all, q_n, q_nm1, dv)

                def matvec_qn_t(v: Array) -> Array:
                    if map_verbosity_level(get_verbosity()) >= map_verbosity_level(
                        "normal"
                    ):
                        # print a dot for every GMRES iteration. Due to the jax GMRES function not returning the number
                        # of iterations, this at least allows us to count the dots!
                        def _print_gmres_dot() -> None:
                            sys.stdout.write(".")
                            sys.stdout.flush()

                        jax.debug.callback(_print_gmres_dot, ordered=True)

                    d_q: AeroelasticMinimalStates = pull_all(v)[0]
                    return d_q.to_free_dofs(solve_dofs_arr=solve_dofs_arr)

                b_rhs = -(p_j_n_p_q_n.reshape(n_j, -1) + adj_t_p_r_np1_p_q_n)

                def _solve_row(b_row: Array, x0_row: Array) -> tuple[Array, Array]:
                    # noinspection PyTypeChecker
                    x, info = jax.scipy.sparse.linalg.gmres(
                        matvec_qn_t,
                        b_row,
                        x0=x0_row if gmres_warm_start else None,
                        tol=1e-6,
                        atol=1e-6,
                        restart=gmres_restart,
                        maxiter=50,
                        M=apply_precond,
                        solve_method=gmres_mode,
                    )
                    return x, info

                adj_n, gmres_info = jax.vmap(_solve_row)(b_rhs, adj_np1)

                def _pull_row(
                    a: Array,
                ) -> tuple[Array, AeroelasticDesignVariables]:
                    q_nm1_cot: AeroelasticMinimalStates
                    _, q_nm1_cot, dv_cot = pull_all(a)

                    return q_nm1_cot.to_free_dofs(solve_dofs_arr=solve_dofs_arr), dv_cot

                adj_t_p_r_n_p_q_nm1, dv_grads = jax.vmap(_pull_row)(adj_n)

                d_j_d_x_ += dv_grads
                d_j_d_x_ += p_j_n_p_x

                jax_print(
                    "\nSolved adjoint for timestep {i_ts} (GMRES converged={converged}, max|adj|={ma:.2e})",
                    i_ts=i_ts,
                    converged=jnp.max(gmres_info) == 0,
                    ma=jnp.max(jnp.abs(adj_n)),
                    verbose_level="normal",
                )

                if save_adjoint:
                    adj_full_ = adj_full_.at[i_ts].set(adj_n)

                return d_j_d_x_, adj_n, adj_t_p_r_n_p_q_nm1, adj_full_

            d_j_d_x, _, future_row, adj_full = jax.lax.fori_loop(
                lower=0,
                upper=n_adj_iters,
                body_fun=matrix_free_body,
                init_val=(
                    dv_grad_init,
                    jnp.zeros((n_j, n_adj_dof)),
                    jnp.zeros((n_j, n_adj_dof)),
                    adj_full_init,
                ),
            )
        else:

            def step_body(
                rev_i_ts_: int,
                carry: tuple[AeroelasticDesignVariables, Array, Array, Array],
            ) -> tuple[AeroelasticDesignVariables, Array, Array, Array]:
                d_j_d_x_, adj_np1, p_r_np1_p_q_n, adj_full_ = carry

                i_ts = i_ts_end_adj_ - rev_i_ts_
                i_ts_nm1 = jnp.maximum(i_ts - 1, 0)
                q_nm1 = case.get_minimal_states(i_ts=i_ts_nm1)
                q_n = case.get_minimal_states(i_ts=i_ts)
                (
                    p_res_p_q_nm1,
                    p_res_p_q_n,
                    p_v_dot_res_p_struct_dv_,
                    p_aero_res_d_dv_,
                    *_,
                ) = self.timestep_residual_jacobians(
                    i_ts=i_ts,
                    t=case.structure.t[i_ts],
                    q_nm1=q_nm1,
                    q_n=q_n,
                    dv_=dv,
                    dv_full=dv_full,
                    thrust_t=case.structure.thrust,
                    solve_dofs=solve_dofs,
                    approx_grads=approx_grads,
                    n_profile_loops=None,
                    jac_options=jac_options,
                )
                p_j_n_p_q_n_, p_j_n_p_x_ = objective_jacobians(i_ts=i_ts, q_n=q_n)

                # solve adjoint step
                b = -(p_j_n_p_q_n_.reshape(n_j, -1) + adj_np1 @ p_r_np1_p_q_n).T
                adj_n = jnp.linalg.solve(p_res_p_q_n.T, b).T

                jax_print(
                    "Solved adjoint for timestep {i_ts}",
                    i_ts=i_ts,
                    verbose_level="normal",
                )

                # add sentitivity of aerodynamic problem through full aero residual
                d_j_d_x_ += p_aero_res_d_dv_.premultiply_adj(adj_n[:, 4 * n_solve :])

                # add sensitivity of structural problem through v_dot residual
                d_j_d_x_.structure += p_v_dot_res_p_struct_dv_.premultiply_adj(
                    adj_n[:, 2 * n_solve : 3 * n_solve]
                )
                d_j_d_x_ += p_j_n_p_x_

                if save_adjoint:
                    adj_full_ = adj_full_.at[i_ts].set(adj_n)

                return d_j_d_x_, adj_n, p_res_p_q_nm1, adj_full_

            d_j_d_x, adj_last, p_r1_p_q0, adj_full = jax.lax.fori_loop(
                lower=0,
                upper=n_adj_iters,
                body_fun=step_body,
                init_val=(
                    dv_grad_init,
                    jnp.zeros((n_j, n_adj_dof)),
                    jnp.zeros((n_adj_dof, n_adj_dof)),
                    adj_full_init,
                ),
            )
            future_row = adj_last @ p_r1_p_q0

        # solve initial timestep adjoint, as there is no r0. Skipped when the adjoint window truncates early time steps
        if include_initial_state_grad and i_ts_start_adj_ <= 1:
            future_cot_q0_full = jnp.zeros((n_j, minimal_states_init.n_states))
            if case.structure.n_tstep > 1:
                future_cot_q0_full = future_cot_q0_full.at[:, free_state_ix].set(
                    future_row
                )

            d_j_d_x += self._initial_timestep_grad_contribution(
                case=case[0].to_static(),
                objective=objective,
                grads_to_compute=grads_to_compute,
                p_varphi_p_x=p_varphi_p_x,
                solve_dofs=solve_dofs_arr,
                adj_t_p_r1_p_q0=future_cot_q0_full,
                horseshoe=static_horseshoe,
            )

        # restore original shape of j, and cut off zeros for past-end timestep and initial timestep which are always 0
        adj = (
            adj_full.reshape(adj_full.shape[0], *j_shape, *adj_full.shape[2:])[1:-1]
            if save_adjoint
            else None
        )

        d_j_d_x.mapping = dv.mapping

        return d_j_d_x, j_eval, adj

    def dynamic_adjoint_profile(
        self,
        case: AeroelasticCase,
        approx_grads: bool,
        jacobian_approximations: AeroelasticJacobianApproximations | None = None,
        grads_to_compute: AeroelasticGradsToCompute | None = None,
        i_ts: int = 1,
        n_profile_loops: int = 10,
    ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
        r"""
        Function to time evaluation of the Jacobians used for the coupled aeroelastic adjoint solution for the case
        where the full Jacobian is computed.
        :param case: Dynamic aeroelastic case from which to extract states.
        :param approx_grads: If True, neglect small gradient terms.
        :param jacobian_approximations: Define which blocks of the adjoint Jacobians will be substituted for
        approximations.
        :param grads_to_compute: AeroelasticGradsToCompute object describing which design gradients to compute. If
        None, all gradients will be computed.
        :param i_ts: Time step index where to evaluate residual Jacobians.
        :param n_profile_loops: Number of times to loop the Jacobian evaluation time for averaging the runtime.
        :return: Dictionary of {residual_name: {gradient_argument: val}} for compile time and run time respectively.
        """

        print_table_title(inner_width=95, title="Aeroelastic Adjoint Profile")

        jac_options = self.construct_approximate_jacobians(
            sol=case,
            jacobian_approximations=jacobian_approximations
            if jacobian_approximations is not None
            else AeroelasticJacobianApproximations(),
        )

        *_, compile_time, run_time = self.timestep_residual_jacobians(
            i_ts=i_ts,
            t=case.aero.t[i_ts],
            q_nm1=case.get_minimal_states(i_ts=i_ts - 1),
            q_n=case.get_minimal_states(i_ts=i_ts),
            dv_=self.get_design_variables(case=case, grads_to_compute=grads_to_compute),
            dv_full=self.get_design_variables(case=case, grads_to_compute=None),
            thrust_t=case.structure.thrust,
            solve_dofs=get_solve_dofs(
                n_dof=self.structure.n_dof,
                prescribed_dofs=case.structure.prescribed_dofs,
            ),
            approx_grads=approx_grads,
            n_profile_loops=n_profile_loops,
            jac_options=jac_options,
        )

        assert compile_time is not None and run_time is not None, (
            "No output timings passed"
        )

        print_table_line(inner_width=95)

        return compile_time, run_time

    def trim(
        self,
        prescribed_dofs: Sequence[int] | Array | slice | int,
        zero_force_dofs: Sequence[int] | Array | slice | int,
        trim_cs: Sequence[str | Sequence[str]] | str | None,
        thrust_nodes: Sequence[str | Sequence[str]] | str | None,
        trim_orientation: str | Sequence[str] | None = "x",
        trim_hinges: Sequence[str | Sequence[str]] | str | None = None,
        trim_f_abs_tolerance: float = 1e-2,
        f_ext_follower: Array | None = None,
        f_ext_dead: Array | None = None,
        t: float | Array = 0.0,
        load_steps: int = 1,
        trim_relaxation: float = 0.9,
        horseshoe: bool = False,
        method: Literal["adjoint", "finite_difference"] = "finite_difference",
        broyden_fd_step: float = 1e-3,
        max_iter: int = 100,
    ) -> tuple[AeroelasticCase, TrimVariables]:
        r"""
        Trim an aircraft such that the resulting sum of forces on the aircraft is zero without any supports.
        :param prescribed_dofs: Degrees of freedom which are clamped for the trim process.
        :param zero_force_dofs: Degrees freedom where we wish to drive the clamping force to zero. This is not
        necessarily the same as prescribed_dofs, as in some cases there are degrees of freedom we will allow to have a
        non-zero clamping force. For example in the case of a clamped cantilever wing where we wish to find the angle
        of attack that gives lift equal to the weight, there would be a nonzero pitching moment.
        :param trim_cs: Keys of control surfaces which are to be used to trim the aircraft. Each element may either be
        a single control-surface key (that surface gets its own independent deflection) or a sequence of keys (those
        surfaces are tied together and share a single deflection).
        :param thrust_nodes: Keys of thrust nodes which are to be used to trim the aircraft. Each element may either be
        a single node key (that node gets its own independent thrust value) or a sequence of node keys (those nodes are
        tied together and share a single thrust value).
        :param trim_orientation: Inertial axis (or axes if a sequence is provided) around which the aircraft is
        rotated about at the clamp to achieve trim.
        :param trim_hinges: Names of ``MultibodyHinge`` constraints (keys into the structure's named constraints)
        whose rotation should be solved as a trim variable.
        :param trim_f_abs_tolerance: Absolute maximum force residual at the clamped nodes for convergence to be achieved.
        :param f_ext_follower: External follower forces, [n_nodes, 6].
        :param f_ext_dead: external dead forces, [n_nodes, 6].
        :param t: Time at which to trim the aircraft, default zero.
        :param load_steps: Number of load steps used for the static solution.
        :param trim_relaxation: Relaxation factor for updates to degrees of freedom used to achieve trim.
        :param horseshoe: If true, use a horseshoe wake formulation.
        :param method: "adjoint" rebuilds the trim Jacobian each iteration via the adjoint method. "finite_difference"
        approximates the Jacobian with a forward finite-difference sweep. Usually the latter is faster assuming a small
        number of trim variables.
        :param broyden_fd_step: Step size used for the finite-difference bootstrap of the Broyden Jacobian.
        :param max_iter: Maximum number of trim iterations. A warning is emitted if the routine fails to converge within
        this limit.
        :return: Aeroelastic solution object for the trimmed aircraft.
        """

        # parse groups for paired thrust nodes/control surfaces/hinges
        cs_groups: list[tuple[str, ...]] = parse_groups(trim_cs, "trim_cs")
        thrust_groups: list[tuple[str, ...]] = parse_groups(
            thrust_nodes, "thrust_nodes"
        )
        hinge_groups: list[tuple[str, ...]] = parse_groups(trim_hinges, "trim_hinges")

        check_unique_members(cs_groups, "trim_cs")
        check_unique_members(thrust_groups, "thrust_nodes")
        check_unique_members(hinge_groups, "trim_hinges")

        if hinge_groups and method == "adjoint":
            raise ValueError(
                "trim_hinges is only supported with method='finite_difference'."
            )

        trim_orientation_: Sequence[str] = (
            [trim_orientation]
            if isinstance(trim_orientation, str)
            else trim_orientation
            if trim_orientation is not None
            else []
        )

        zero_force_dofs_: tuple[int, ...] = self.structure.make_prescribed_dofs_tuple(
            zero_force_dofs
        )

        prescribed_dofs_: tuple[int, ...] = self.structure.make_prescribed_dofs_tuple(
            prescribed_dofs
        )

        if not self.structure.use_gravity:
            warn("Gravity is not enabled. Trim may result in unexpected behaviour.")

        # initial set of variables. Tied members share a single value, initialised from the first member.
        trim_variables_init: TrimVariables = TrimVariables(
            cs_ang={group_key(g): self.aero.cs_ang0[g[0]] for g in cs_groups},
            thrust={
                group_key(g): self.structure.thrust_reference[g[0]]
                for g in thrust_groups
            },
            trim_angles={
                k: self.structure.orientation_euler[ORIENTATION_DICT[k]]
                for k in trim_orientation_
            },
            hinge_angle={group_key(g): jnp.array(0.0) for g in hinge_groups},
        )

        ae_sol_init = self.reference_configuration(
            horseshoe=horseshoe,
            prescribed_dofs=prescribed_dofs_,
            use_f_ext_dead=f_ext_dead is not None,
            use_f_ext_follower=f_ext_follower is not None,
        )

        inner_case = deepcopy(self)

        if method == "adjoint":

            def trim_body(
                i_iter: int,
                trim_variables_: TrimVariables,
                sol_: AeroelasticCase,
                f_clamp_: Array,
            ) -> tuple[int, TrimVariables, AeroelasticCase, Array]:
                i_iter, _, tv, sol, fc = self.trim_iter(
                    i_iter,
                    inner_case,
                    trim_variables_,
                    sol_,
                    f_clamp_,
                    prescribed_dofs=prescribed_dofs_,
                    zero_force_dofs=zero_force_dofs_,
                    f_ext_dead=f_ext_dead,
                    f_ext_follower=f_ext_follower,
                    t=jnp.array(t),
                    load_steps=load_steps,
                    horseshoe=horseshoe,
                    cs_groups=cs_groups,
                    thrust_groups=thrust_groups,
                    trim_orientation=trim_orientation_,
                    trim_relaxation=trim_relaxation,
                )
                return i_iter, tv, sol, fc

            f_clamp_init = jnp.full((len(zero_force_dofs_)), 1e10)
            print_table_title(title="Trim (Adjoint)", inner_width=104)
            trim_variables_init.print_header(f_clamp=f_clamp_init)
            _, trim_variables, ae_sol, f_clamp_final = jax.lax.while_loop(
                lambda args_: jnp.logical_and(
                    jnp.any(jnp.abs(args_[3]) >= trim_f_abs_tolerance),
                    args_[0] < max_iter,
                ),
                body_fun=lambda args_: trim_body(*args_),
                init_val=(
                    0,
                    trim_variables_init,
                    ae_sol_init,
                    f_clamp_init,
                ),
            )
            print_table_line(inner_width=104)
        elif method == "finite_difference":
            print_table_title(title="Trim (Finite Difference)", inner_width=104)

            b_approx_init, f_clamp_init, ae_sol_bootstrap = self._trim_fd_jacobian(
                inner_case=inner_case,
                trim_variables=trim_variables_init,
                fd_step=broyden_fd_step,
                prescribed_dofs=prescribed_dofs_,
                zero_force_dofs=zero_force_dofs_,
                f_ext_follower=f_ext_follower,
                f_ext_dead=f_ext_dead,
                t=jnp.array(t),
                load_steps=load_steps,
                horseshoe=horseshoe,
                cs_groups=cs_groups,
                thrust_groups=thrust_groups,
                hinge_groups=hinge_groups,
                trim_orientation=trim_orientation_,
            )

            def trim_body_broyden(
                i_iter: int,
                trim_variables_: TrimVariables,
                sol_: AeroelasticCase,
                f_clamp_: Array,
                b_approx_: Array,
            ) -> tuple[int, TrimVariables, AeroelasticCase, Array, Array]:
                i_iter, _, tv, sol, fc, b = self._trim_iter_fd(
                    i_iter,
                    inner_case,
                    trim_variables_,
                    sol_,
                    f_clamp_,
                    b_approx_,
                    prescribed_dofs=prescribed_dofs_,
                    zero_force_dofs=zero_force_dofs_,
                    f_ext_dead=f_ext_dead,
                    f_ext_follower=f_ext_follower,
                    t=jnp.array(t),
                    load_steps=load_steps,
                    horseshoe=horseshoe,
                    cs_groups=cs_groups,
                    thrust_groups=thrust_groups,
                    hinge_groups=hinge_groups,
                    trim_orientation=trim_orientation_,
                    trim_relaxation=trim_relaxation,
                )
                return i_iter, tv, sol, fc, b

            trim_variables_init.print_header(f_clamp=f_clamp_init)
            _, trim_variables, ae_sol, f_clamp_final, _ = jax.lax.while_loop(
                lambda args_: jnp.logical_and(
                    jnp.any(jnp.abs(args_[3]) >= trim_f_abs_tolerance),
                    args_[0] < max_iter,
                ),
                body_fun=lambda args_: trim_body_broyden(*args_),
                init_val=(
                    0,
                    trim_variables_init,
                    ae_sol_bootstrap,
                    f_clamp_init,
                    b_approx_init,
                ),
            )
            print_table_line(inner_width=104)
        else:
            raise ValueError(f"Unknown trim method: {method!r}.")

        if bool(jnp.any(jnp.isnan(f_clamp_final))):
            warn("Trim residual is NaN - solution diverged")
        elif bool(jnp.any(jnp.abs(f_clamp_final) >= trim_f_abs_tolerance)):
            warn(f"Trim did not converge within max_iter={max_iter} iterations ")

        new_orientation: Array = self.structure.orientation_euler
        for k, v in trim_variables.trim_angles.items():
            new_orientation = new_orientation.at[ORIENTATION_DICT[k]].set(v)

        # set solutions into case object
        self.set_design_variables(
            coords=self.structure.x0_reference,
            k_cs=self.structure.k_cs,
            m_cs=self.structure.m_cs,
            m_lumped=self.structure.m_lumped
            if self.structure.use_lumped_mass
            else None,
            dt=self.aero.dt,
            flowfield=self.aero.flowfield,
            delta_w=self.aero.delta_w,
            x0_aero=self.aero.zeta_b0,
            thrust_reference=self.structure.thrust_reference
            | expand_groups(trim_variables.thrust, thrust_groups),
            orientation_euler=new_orientation,
            cs_angles_reference=self.aero.cs_ang0
            | expand_groups(trim_variables.cs_ang, cs_groups),
            remove_checks=True,
        )

        if hinge_groups:
            # release the angle-pinning constraint used to condition the trim solve
            self._revert_hinge_trim(hinge_groups)

        return ae_sol, trim_variables

    def trim_angles_to_euler(self, trim_angles: dict[str, Array]) -> Array:
        r"""
        Find the 3 Euler angles describing the aircraft orientation. This allows for any combination to be set by
        the trim routine, with values not passed using the fixed values provided in the reference orientation.
        :param trim_angles: Dictionary of axis-angle pairs.
        :return: Euler angles describing the aircraft orientation, ``(3, )``.
        """

        orientation_euler = self.structure.orientation_euler
        for k, v in trim_angles.items():
            orientation_euler = orientation_euler.at[ORIENTATION_DICT[k]].set(v)
        return orientation_euler

    @staticmethod
    def _rebuild_hinge(
        old: MultibodyHinge, prescribed_angle: Array | float | None
    ) -> MultibodyHinge:
        """Reconstruct a MultibodyHinge with the same geometry/spring/damping but a different prescribed angle. A
        prescribed angle of None will release it."""
        return MultibodyHinge(
            node_i=old.node_i,
            node_j=old.node_j,
            axis=old.axis,
            hg_rel_ref=old.hg_rel_ref,
            spring_stiffness=old.spring_stiffness,
            damping=old.damping,
            prescribed_angle=prescribed_angle,
        )

    @staticmethod
    def _set_hinge_constraints(
        structure: BeamStructure, updates: dict[str, MultibodyHinge]
    ) -> None:
        """Substitute the named hinge constraints in place on ``structure``."""
        named = dict(structure.constraints)
        named.update(updates)
        structure.constraints = named

    @staticmethod
    def _apply_hinge_trim(
        structure: BeamStructure,
        hinge_angle: dict[str, Array],
        hinge_groups: Sequence[Sequence[str]],
    ) -> None:
        """Pin each named hinge to a given angle."""
        per_member = expand_groups(hinge_angle, hinge_groups)
        if not per_member:
            return
        updates: dict[str, MultibodyHinge] = {}
        for name, angle in per_member.items():
            old = structure.constraints.get(name)
            if not isinstance(old, MultibodyHinge):
                raise TypeError(
                    f"trim_hinges entry {name} not MultibodyHinge constraint."
                )
            updates[name] = CoupledAeroelastic._rebuild_hinge(old, angle)
        CoupledAeroelastic._set_hinge_constraints(structure, updates)

    def _revert_hinge_trim(self, hinge_groups: Sequence[Sequence[str]]) -> None:
        """Release the trimmed hinges back to free (zero-stiffness) joints once trim has converged."""
        names = [name for group in hinge_groups for name in group]
        updates: dict[str, MultibodyHinge] = {}
        for name in names:
            old = self.structure.constraints[name]
            if not isinstance(old, MultibodyHinge):
                raise TypeError(
                    f"trim_hinges entry {name} not a MultibodyHinge constraint"
                )
            updates[name] = self._rebuild_hinge(old, None)
        self._set_hinge_constraints(self.structure, updates)

    @staticmethod
    def _hinge_moment_residual(
        structure: BeamStructure,
        ae_sol: AeroelasticCase,
        hinge_groups: Sequence[Sequence[str]],
        prescribed_dofs: tuple[int, ...],
    ) -> Array:
        """
        Re-solve the converged system once more at the trimmed equilibrium to obtain the Lagrange multiplier
        on each prescribed hinge's angle-pinning constraint row.
        """
        struct_case = ae_sol.structure
        hg = struct_case.hg
        rmat = hg[:, :3, :3]

        f_dead_total: Array | None = None
        if struct_case.f_ext_dead is not None:
            f_dead_total = transform_nodal_vect(struct_case.f_ext_dead, rmat)
        if struct_case.f_ext_aero is not None:
            f_aero_global = transform_nodal_vect(struct_case.f_ext_aero, rmat)
            f_dead_total = (
                f_aero_global if f_dead_total is None else f_dead_total + f_aero_global
            )

        solve_dofs = jnp.array(
            get_solve_dofs(n_dof=structure.n_dof, prescribed_dofs=prescribed_dofs)
        )
        d = structure.make_d(hg)
        p_d = structure.make_p_d(d)
        eps = structure.make_eps(d)
        m_t = structure.make_m_t(d) if structure.use_gravity else None

        k_t_full = structure.make_k_t_full(
            d=d, p_d=p_d, eps=eps, f_ext_dead=f_dead_total, rmat=rmat, m_t=m_t
        )
        k_t_full = structure.apply_nodal_constraint_tangent(
            mat=k_t_full, hg=hg, i_ts=0, gamma_prime=None
        )
        k_t_solve = k_t_full[jnp.ix_(solve_dofs, solve_dofs)]

        f_res_solve, _ = structure.make_f_res(
            solve_dofs=solve_dofs,
            p_d=p_d,
            eps=eps,
            hg=hg,
            f_ext_follower_n=struct_case.f_ext_follower,
            f_ext_dead_n=f_dead_total,
            thrust_n=structure.thrust_reference,
            dynamic=False,
            m_t=m_t,
            c_l=None,
            c_l_lumped=None,
            v=None,
            v_dot=None,
        )
        _, lam, _ = structure.solve_constrained(
            sys_mat_solve=k_t_solve,
            f_res_solve=f_res_solve,
            hg_eval=hg,
            solve_dofs=solve_dofs,
        )

        offsets: dict[str, int] = {}
        running = 0
        for name, con in structure.constraints.items():
            if isinstance(con, HardConstraint) and con.is_holonomic:
                offsets[name] = running
                running += con.n_constraints

        def _row(name_: str) -> Array:
            con_ = structure.constraints[name_]
            assert isinstance(con_, HardConstraint)
            return lam[offsets[name_] + con_.n_constraints - 1]

        return jnp.stack([sum(_row(name) for name in group) for group in hinge_groups])

    @staticmethod
    def _trim_solve_f_clamp(
        inner_case: CoupledAeroelastic,
        trim_variables: TrimVariables,
        prescribed_dofs: tuple[int, ...],
        zero_force_dofs: tuple[int, ...],
        f_ext_follower: Array | None,
        f_ext_dead: Array | None,
        t: Array,
        load_steps: int,
        horseshoe: bool,
        cs_groups: Sequence[Sequence[str]],
        thrust_groups: Sequence[Sequence[str]],
        hinge_groups: Sequence[Sequence[str]] = (),
    ) -> tuple[AeroelasticCase, Array]:
        """Push trim variables into inner_case, run a static solve, and return ``(solution, clamp-force residual)``."""
        CoupledAeroelastic._apply_hinge_trim(
            structure=inner_case.structure,
            hinge_angle=trim_variables.hinge_angle,
            hinge_groups=hinge_groups,
        )
        inner_case.set_design_variables(
            coords=inner_case.structure.x0_reference,
            k_cs=inner_case.structure.k_cs,
            m_cs=inner_case.structure.m_cs,
            m_lumped=inner_case.structure.m_lumped
            if inner_case.structure.use_lumped_mass
            else None,
            thrust_reference=inner_case.structure.thrust_reference
            | expand_groups(trim_variables.thrust, thrust_groups),
            dt=inner_case.aero.dt,
            flowfield=inner_case.aero.flowfield,
            delta_w=inner_case.aero.delta_w,
            x0_aero=inner_case.aero.zeta_b0,
            orientation_euler=inner_case.trim_angles_to_euler(
                trim_variables.trim_angles
            ),
            cs_angles_reference=inner_case.aero.cs_ang0
            | expand_groups(trim_variables.cs_ang, cs_groups),
            remove_checks=True,
        )
        with verbosity(
            level="silent"
            if map_verbosity_level(get_verbosity()) <= map_verbosity_level("normal")
            else get_verbosity()
        ):
            ae_sol = inner_case.static_solve(
                prescribed_dofs=prescribed_dofs,
                f_ext_follower=f_ext_follower,
                f_ext_dead=f_ext_dead,
                t=t,
                load_steps=load_steps,
                horseshoe=horseshoe,
            )
        f_clamp = ae_sol.structure.f_res.ravel()[jnp.array(zero_force_dofs)]
        if hinge_groups:
            hinge_moment = CoupledAeroelastic._hinge_moment_residual(
                structure=inner_case.structure,
                ae_sol=ae_sol,
                hinge_groups=hinge_groups,
                prescribed_dofs=prescribed_dofs,
            )
            f_clamp = jnp.concatenate([f_clamp, hinge_moment])
        return ae_sol, f_clamp

    @staticmethod
    def _apply_trim_update(
        trim_variables: TrimVariables,
        trim_update: Array,
        cs_groups: Sequence[Sequence[str]],
        thrust_groups: Sequence[Sequence[str]],
        trim_orientation: Sequence[str],
        hinge_groups: Sequence[Sequence[str]] = (),
    ) -> TrimVariables:
        """Subtract the flat trim_update vector from the current variables and return a new TrimVariables."""
        idx = 0
        updated_cs_ang = {}
        for group in cs_groups:
            key = group_key(group)
            updated_cs_ang[key] = trim_variables.cs_ang[key] - trim_update[idx]
            idx += 1
        updated_thrust = {}
        for group in thrust_groups:
            key = group_key(group)
            updated_thrust[key] = trim_variables.thrust[key] - trim_update[idx]
            idx += 1
        updated_trim_angles = {}
        for k in trim_orientation:
            updated_trim_angles[k] = trim_variables.trim_angles[k] - trim_update[idx]
            idx += 1
        updated_hinge_angle = {}
        for group in hinge_groups:
            key = group_key(group)
            updated_hinge_angle[key] = (
                trim_variables.hinge_angle[key] - trim_update[idx]
            )
            idx += 1
        return TrimVariables(
            cs_ang=updated_cs_ang,
            thrust=updated_thrust,
            trim_angles=updated_trim_angles,
            hinge_angle=updated_hinge_angle,
        )

    @staticmethod
    def trim_iter(
        i_iter: int,
        inner_case: CoupledAeroelastic,
        trim_variables: TrimVariables,
        _: AeroelasticCase,
        __: Array,
        *,
        prescribed_dofs: tuple[int, ...],
        zero_force_dofs: tuple[int, ...],
        f_ext_follower: Array | None,
        f_ext_dead: Array | None,
        t: Array,
        load_steps: int,
        trim_relaxation: float,
        horseshoe: bool,
        cs_groups: Sequence[Sequence[str]],
        thrust_groups: Sequence[Sequence[str]],
        trim_orientation: Sequence[str],
    ) -> tuple[int, CoupledAeroelastic, TrimVariables, AeroelasticCase, Array]:
        ae_sol, f_clamp = CoupledAeroelastic._trim_solve_f_clamp(
            inner_case=inner_case,
            trim_variables=trim_variables,
            prescribed_dofs=prescribed_dofs,
            zero_force_dofs=zero_force_dofs,
            f_ext_follower=f_ext_follower,
            f_ext_dead=f_ext_dead,
            t=t,
            load_steps=load_steps,
            horseshoe=horseshoe,
            cs_groups=cs_groups,
            thrust_groups=thrust_groups,
        )

        with verbosity(
            level="silent"
            if map_verbosity_level(get_verbosity()) <= map_verbosity_level("normal")
            else get_verbosity()
        ):

            def objective(states: AeroelasticFullStates, *_) -> Array:
                return states.structure.f_res.ravel()[jnp.array(zero_force_dofs)]

            d_f_clamp_d_x: AeroelasticDesignVariables
            d_f_clamp_d_x, _ = inner_case.static_adjoint(
                case=ae_sol,
                objective=objective,
                grads_to_compute=AeroelasticGradsToCompute(
                    structure=StructureGradsToCompute(
                        x0=False,
                        orientation_euler=True,
                        k_cs=False,
                        m_cs=False,
                        m_lumped=False,
                        f_ext_follower=False,
                        f_ext_dead=False,
                        thrust_t=True,
                    ),
                    aero=AeroGradsToCompute(
                        x0_aero=False, flowfield=False, cs_ang_t=True, cs_vel_t=False
                    ),
                ),
            )

        assert (
            d_f_clamp_d_x.aero.cs_ang_t is not None
            and d_f_clamp_d_x.structure.thrust_t is not None
            and d_f_clamp_d_x.structure.orientation_euler is not None
        )
        n_zero_force = len(zero_force_dofs)
        d_f_clamp_d_cs_ang = (
            jnp.concatenate(
                [
                    sum(
                        d_f_clamp_d_x.aero.cs_ang_t[k].reshape(n_zero_force, -1)
                        for k in group
                    )
                    for group in cs_groups
                ],
                axis=1,
            )
            if cs_groups
            else jnp.zeros((n_zero_force, 0))
        )
        d_f_clamp_d_thrust = (
            jnp.concatenate(
                [
                    sum(
                        d_f_clamp_d_x.structure.thrust_t[k].reshape(n_zero_force, -1)
                        for k in group
                    )
                    for group in thrust_groups
                ],
                axis=1,
            )
            if thrust_groups
            else jnp.zeros((n_zero_force, 0))
        )
        d_f_clamp_d_trim_angles = (
            jnp.concatenate(
                [
                    d_f_clamp_d_x.structure.orientation_euler[:, [ORIENTATION_DICT[k]]]
                    for k in trim_orientation
                ],
                axis=1,
            )
            if trim_orientation
            else jnp.zeros((len(zero_force_dofs), 0))
        )

        d_f_clamp_d_trim_variables = jnp.concatenate(
            [d_f_clamp_d_cs_ang, d_f_clamp_d_thrust, d_f_clamp_d_trim_angles],
            axis=1,
        )  # note that this is not square (n_zero_force_dof, n_free_trim_dof)

        trim_update = (
            jnp.linalg.lstsq(a=d_f_clamp_d_trim_variables, b=f_clamp)[0]
            * trim_relaxation
        )  # subtract this from current solution to update

        trim_variables = CoupledAeroelastic._apply_trim_update(
            trim_variables=trim_variables,
            trim_update=trim_update,
            cs_groups=cs_groups,
            thrust_groups=thrust_groups,
            trim_orientation=trim_orientation,
        )

        trim_variables.print_values(i_iter=i_iter, f_clamp=f_clamp)

        return i_iter + 1, inner_case, trim_variables, ae_sol, f_clamp

    @staticmethod
    def _trim_fd_jacobian(
        inner_case: CoupledAeroelastic,
        trim_variables: TrimVariables,
        fd_step: float,
        prescribed_dofs: tuple[int, ...],
        zero_force_dofs: tuple[int, ...],
        f_ext_follower: Array | None,
        f_ext_dead: Array | None,
        t: Array,
        load_steps: int,
        horseshoe: bool,
        cs_groups: Sequence[Sequence[str]],
        thrust_groups: Sequence[Sequence[str]],
        trim_orientation: Sequence[str],
        hinge_groups: Sequence[Sequence[str]] = (),
    ) -> tuple[Array, Array, AeroelasticCase]:
        """
        Obtain the Broyden Jacobian with a forward finite-difference sweep: perturb each trim variable in turn by
        `fd_step`, solve, and build the corresponding column.
        """
        ae_sol_0, f_clamp_0 = CoupledAeroelastic._trim_solve_f_clamp(
            inner_case=inner_case,
            trim_variables=trim_variables,
            prescribed_dofs=prescribed_dofs,
            zero_force_dofs=zero_force_dofs,
            f_ext_follower=f_ext_follower,
            f_ext_dead=f_ext_dead,
            t=t,
            load_steps=load_steps,
            horseshoe=horseshoe,
            cs_groups=cs_groups,
            thrust_groups=thrust_groups,
            hinge_groups=hinge_groups,
        )

        n_trim = (
            len(cs_groups)
            + len(thrust_groups)
            + len(trim_orientation)
            + len(hinge_groups)
        )
        n_residual = len(zero_force_dofs) + len(hinge_groups)
        if n_trim == 0:
            return jnp.zeros((n_residual, 0)), f_clamp_0, ae_sol_0

        base_structure = pytree_clone(inner_case.structure)
        base_aero = pytree_clone(inner_case.aero)

        def reset_inner_case() -> None:
            inner_case.structure = pytree_clone(base_structure)
            inner_case.aero = pytree_clone(base_aero)

        def eval_perturbed(i: Array) -> Array:
            reset_inner_case()

            trim_update = jnp.zeros((n_trim,)).at[i].set(-fd_step)
            tv_p = CoupledAeroelastic._apply_trim_update(
                trim_variables=trim_variables,
                trim_update=trim_update,
                cs_groups=cs_groups,
                thrust_groups=thrust_groups,
                trim_orientation=trim_orientation,
                hinge_groups=hinge_groups,
            )
            _, f_p = CoupledAeroelastic._trim_solve_f_clamp(
                inner_case=inner_case,
                trim_variables=tv_p,
                prescribed_dofs=prescribed_dofs,
                zero_force_dofs=zero_force_dofs,
                f_ext_follower=f_ext_follower,
                f_ext_dead=f_ext_dead,
                t=t,
                load_steps=load_steps,
                horseshoe=horseshoe,
                cs_groups=cs_groups,
                thrust_groups=thrust_groups,
                hinge_groups=hinge_groups,
            )
            return (f_p - f_clamp_0) / fd_step

        columns = jax.lax.map(
            eval_perturbed, jnp.arange(n_trim)
        )  # (n_trim, n_residual)
        b_approx = columns.T

        # restore concrete state for the Broyden while_loop that follows
        reset_inner_case()

        return b_approx, f_clamp_0, ae_sol_0

    @staticmethod
    def _trim_iter_fd(
        i_iter: int,
        inner_case: CoupledAeroelastic,
        trim_variables: TrimVariables,
        _: AeroelasticCase,
        f_clamp: Array,
        b_approx: Array,
        *,
        prescribed_dofs: tuple[int, ...],
        zero_force_dofs: tuple[int, ...],
        f_ext_follower: Array | None,
        f_ext_dead: Array | None,
        t: Array,
        load_steps: int,
        trim_relaxation: float,
        horseshoe: bool,
        cs_groups: Sequence[Sequence[str]],
        thrust_groups: Sequence[Sequence[str]],
        trim_orientation: Sequence[str],
        hinge_groups: Sequence[Sequence[str]] = (),
    ) -> tuple[int, CoupledAeroelastic, TrimVariables, AeroelasticCase, Array, Array]:
        """
        One Broyden trim iteration: take a step using the current Jacobian approximation, re-solve at the new point,
        then update the Jacobian with the Broyden rank-1 formula.
        """
        trim_update = jnp.linalg.lstsq(a=b_approx, b=f_clamp)[0] * trim_relaxation
        trim_variables_new = CoupledAeroelastic._apply_trim_update(
            trim_variables=trim_variables,
            trim_update=trim_update,
            cs_groups=cs_groups,
            thrust_groups=thrust_groups,
            trim_orientation=trim_orientation,
            hinge_groups=hinge_groups,
        )

        ae_sol_new, f_clamp_new = CoupledAeroelastic._trim_solve_f_clamp(
            inner_case=inner_case,
            trim_variables=trim_variables_new,
            prescribed_dofs=prescribed_dofs,
            zero_force_dofs=zero_force_dofs,
            f_ext_follower=f_ext_follower,
            f_ext_dead=f_ext_dead,
            t=t,
            load_steps=load_steps,
            horseshoe=horseshoe,
            cs_groups=cs_groups,
            thrust_groups=thrust_groups,
            hinge_groups=hinge_groups,
        )

        s = -trim_update  # actual step taken in trim-variable space
        y = f_clamp_new - f_clamp
        b_approx_new = b_approx + jnp.outer(y - b_approx @ s, s) / jnp.dot(s, s)

        trim_variables_new.print_values(i_iter=i_iter, f_clamp=f_clamp_new)

        return (
            i_iter + 1,
            inner_case,
            trim_variables_new,
            ae_sol_new,
            f_clamp_new,
            b_approx_new,
        )
