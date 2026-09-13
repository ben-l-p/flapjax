from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import fields
from typing import Any, cast

import jax
from jax import Array, vmap
from jax import numpy as jnp

from flapjax.algebra.array_utils import construct_named_block_jacobian
from flapjax.algebra.base import (
    ADMode,
    construct_approximation,
    jacrev_custom,
)
from flapjax.algebra.se3 import exp_se3, log_se3
from flapjax.structure import OptionalJacobians, StructureCase
from flapjax.structure.beam import BaseBeamStructure
from flapjax.structure.data_structures import StructureMinimalStates
from flapjax.structure.gradients.data_structures import (
    AApprox,
    StructureDesignVariables,
    StructureFullStates,
    StructureGradsToCompute,
    StructureJacobianApproximations,
    VApprox,
    VarphiApprox,
    VDotApprox,
)
from flapjax.structure.utils import get_solve_dofs, transform_nodal_vect
from flapjax.utils.print_utils import (
    jax_print,
    print_table_line,
    print_table_title,
)
from flapjax.utils.utils import dv_or, make_pytree, pytree_clone

type StructureObjectiveFunction = (
    Callable[[StructureFullStates, StructureDesignVariables, int], Array]
    | Callable[[StructureFullStates, StructureDesignVariables, Array], Array]
    | Callable[[StructureFullStates, StructureDesignVariables, None], Array]
)


@make_pytree
class BeamStructure(BaseBeamStructure):
    def case_from_dv(self, dv: StructureDesignVariables) -> BeamStructure:
        r"""
        Obtain a structural object as a function of design variables, allowing it to have defined gradients w.r.t. design variables.
        :param dv: Design variables.
        :return: Beam structure object with the same functionality as self.
        """
        inner_case = pytree_clone(self)
        inner_case.set_design_variables(
            coords=dv_or(dv.x0, self.x0),
            k_cs=dv_or(dv.k_cs, self.k_cs),
            m_cs=dv_or(dv.m_cs, self.m_cs),
            m_lumped=dv_or(dv.m_lumped, self._m_lumped),
            remove_checks=True,
        )

        return inner_case

    def minimal_states_to_full_states(
        self,
        i_ts: int,
        q: StructureMinimalStates,
        dv: StructureDesignVariables,
        dv_full: StructureDesignVariables,
    ) -> StructureFullStates:
        r"""
        Obtain the full set of states from the minimal states and the design variables.
        :param i_ts: Index of the time step.
        :param q: Minimal dynamic structure states.
        :param dv: Design variables, where entries for gradients which aren't needed are set to None.
        :param dv_full: Design variables, without omissions. These values are fallen back to when an entry in ``dv`` is
        none, to give an equivalent with zero gradient.
        :return: Full set of structural states used inside objective function.
        """
        struct = self.case_from_dv(dv)
        hg = struct.compute_hg_from_varphi(q.varphi)
        d = struct.make_d(hg=hg)
        p_d = struct.make_p_d(d=d)
        eps = struct.make_eps(d=d)
        f_elem = struct.make_f_elem(eps=eps)

        f_ext_dead_i = (
            dv.f_ext_dead[i_ts, ...]
            if dv.f_ext_dead is not None
            else dv_full.f_ext_dead[i_ts, ...]
            if dv_full.f_ext_dead is not None
            else None
        )
        m_t = struct.make_m_t(d=d)

        # k_t_assembled is only required when Rayleigh damping is active. Skip the extra assembly when unused.
        k_t_assembled = (
            struct.make_k_t_full(
                d=d,
                p_d=p_d,
                eps=eps,
                f_ext_dead=f_ext_dead_i,
                rmat=hg[:, :3, :3],
                m_t=m_t,
            )
            if struct.beta_k != 0.0
            else None
        )

        assert dv_full.thrust_t is not None
        f_res = struct.make_f_res(
            solve_dofs=None,
            p_d=p_d,
            eps=eps,
            hg=hg,
            f_ext_follower_n=dv.f_ext_follower[i_ts, ...]
            if dv.f_ext_follower is not None
            else dv_full.f_ext_follower[i_ts, ...]
            if dv_full.f_ext_follower is not None
            else None,
            f_ext_dead_n=f_ext_dead_i,
            thrust_n={k: (v[i_ts] if v.ndim > 0 else v) for k, v in dv.thrust_t.items()}
            if dv.thrust_t is not None
            else {
                k: (v[i_ts] if v.ndim > 0 else v) for k, v in dv_full.thrust_t.items()
            },
            dynamic=True,
            m_t=m_t,
            c_l=self._make_c_t(d=d, d_dot=self._make_d_dot(p_d=p_d, v=q.v), v=q.v)[0],
            c_l_lumped=self._make_c_t_lumped(v=q.v)[0]
            if self.use_lumped_mass
            else None,
            v=q.v,
            v_dot=q.v_dot,
            i_ts=i_ts,
            k_t_assembled=k_t_assembled,
        )[0]
        return StructureFullStates(
            v=q.v,
            v_dot=q.v_dot,
            eps=eps,
            varphi=q.varphi,
            hg=hg,
            f_elem=f_elem,
            f_res=f_res,
        )

    def _structural_states_res_from_dv_varphi(
        self,
        dv: StructureDesignVariables,
        varphi: Array,
        thrust: dict[str, Array],
    ) -> StructureFullStates:
        r"""
        Obtain useful states and forcing residual from design variables and a minimal configuration vector.
        :param dv: Design variables.
        :param varphi: Twist coordinates which map from the reference configuration to the current as
        :math:`\mathbf{H} = \mathbf{H}_0 \mathrm{exp} (\varphi)`.
        :param thrust: Thrust. This is only needed when thrust is not included as a design variable. {keys, ()}.
        :return: Structural states and forcing residual.
        """

        inner_case = self.case_from_dv(dv=dv)

        hg = inner_case.compute_hg_from_varphi(varphi=varphi)  # (n_nodes, 4, 4)
        d = inner_case.make_d(hg)
        p_d = inner_case.make_p_d(d)
        eps = inner_case.make_eps(d)
        f_elem = jnp.einsum("ijk,ik->ij", inner_case.k_cs, eps)

        if inner_case.use_gravity:
            m_t = inner_case.make_m_t(d)
        else:
            m_t = None

        f_res = inner_case.make_f_res(
            solve_dofs=None,
            p_d=p_d,
            eps=eps,
            hg=hg,
            f_ext_follower_n=dv.f_ext_follower,
            f_ext_dead_n=dv.f_ext_dead,
            thrust_n=dv.thrust_t if dv.thrust_t is not None else thrust,
            dynamic=False,
            m_t=m_t,
            c_l=None,
            c_l_lumped=None,
            v=None,
            v_dot=None,
        )[0]

        return StructureFullStates(
            hg=hg,
            varphi=varphi,
            eps=eps,
            f_elem=f_elem,
            f_res=f_res,
            v=None,
            v_dot=None,
        )

    OPTIONAL_JACOBIANS_DEFAULT = OptionalJacobians(True, True, True, True)

    def static_adjoint(
        self,
        structure: StructureCase,
        objective: StructureObjectiveFunction,
        optional_jacobians: OptionalJacobians | None = OPTIONAL_JACOBIANS_DEFAULT,
        ad_mode: ADMode = "reverse",
    ) -> tuple[StructureDesignVariables, Array]:
        r"""
        Computes the static grads of the structure, which is used to compute gradients of the loss with respect to
        the structure's parameters.
        :param structure: StructureCase containing the current state of the structure.
        :param objective: Objective function that takes the structure and design variables and returns an array
        :param optional_jacobians: OptionalJacobians object specifying which Jacobians to compute.
        :param ad_mode: Flag on which to use of the forward or reverse adjoint.
        :return: Gradient of objective function output with respect to design variables, and adjoint states.
        """

        solve_dofs = jnp.array(
            get_solve_dofs(n_dof=self.n_dof, prescribed_dofs=structure.prescribed_dofs)
        )

        if optional_jacobians is not None:
            self.optional_jacobians = optional_jacobians

        # Recover original global dead force: structure.f_ext_dead is stored in local frame as
        # f_local = R^T @ f_global, so f_global = R @ f_local
        rmat = structure.hg[:, :3, :3]
        f_ext_dead_global = (
            transform_nodal_vect(structure.f_ext_dead, rmat)
            if structure.f_ext_dead is not None
            else None
        )

        # make design variables for current state of structure
        dv = StructureDesignVariables(
            x0=self.x0,
            orientation_euler=self.orientation_euler,
            k_cs=self.k_cs,
            m_cs=self._m_cs,
            m_lumped=self.m_lumped if self.use_lumped_mass else None,
            f_ext_follower=structure.f_ext_follower,
            f_ext_dead=f_ext_dead_global,
            thrust_t=structure.thrust,
            f_shape=(),
        )

        struct_states = structure.get_full_states()

        # find shape of objective function output without evaluating function
        f_properties = jax.eval_shape(lambda: objective(struct_states, dv, None))
        f_shape = f_properties.shape
        n_f = f_properties.size
        n_x = dv.n_x
        n_u = len(solve_dofs)
        n_u_full = self.n_dof

        # gradient of objective w.r.t. minimal states
        p_f_p_n, p_f_p_x = jax.jacrev(
            lambda varphi_, dv_: objective(
                self._structural_states_res_from_dv_varphi(
                    dv=dv_, varphi=varphi_, thrust=structure.thrust
                ),
                dv_,
                None,
            ),
            argnums=(0, 1),
            allow_int=True,
        )(structure.varphi, dv)

        p_f_p_n = p_f_p_n.reshape(n_f, n_u_full)[:, solve_dofs]  # (n_f, n_u)
        p_f_p_x = p_f_p_x.ravel_jacobian(n_f, n_x)  # (n_f, n_x)

        # gradient of residual w.r.t. design variables and minimal states
        p_res_p_x, p_res_p_varphi = (jax.jacfwd if n_u > n_x else jax.jacrev)(
            lambda dv_, varphi_: (
                self._structural_states_res_from_dv_varphi(
                    dv=dv_, varphi=varphi_, thrust=structure.thrust
                ).f_res
            ),
            argnums=(0, 1),
            allow_int=True,
        )(dv, structure.varphi)

        p_res_p_x = p_res_p_x.ravel_jacobian(n_u_full, n_x)[solve_dofs, :]  # (n_u, n_x)
        p_res_p_varphi = p_res_p_varphi.reshape(n_u_full, n_u_full)[
            jnp.ix_(solve_dofs, solve_dofs)
        ]  # (n_u, n_u)

        if ad_mode == "forward":
            # forward mode
            adj = jnp.linalg.solve(p_res_p_varphi, p_res_p_x)  # (n_u, n_x)
            rhs = p_f_p_n @ adj  # (n_f, n_x)
        elif ad_mode == "reverse":
            # reverse mode
            adj = jnp.linalg.solve(p_res_p_varphi.T, p_f_p_n.T).T  # (n_f, n_u)
            rhs = adj @ p_res_p_x  # (n_f, n_x)
        else:
            raise ValueError("AD mode must be either 'forward' or 'reverse'")

        return StructureDesignVariables(
            **dv.from_adjoint(f_shape, p_f_p_x - rhs), f_shape=f_shape
        ), adj

    def varphi_res_func(
        self,
        varphi_nm1: Array,
        varphi_n: Array,
        v_nm1: Array,
        a_nm1: Array,
        a_n: Array,
        solve_dofs: tuple[int, ...],
    ) -> Array:
        varphi_nm1 = varphi_nm1.reshape(-1, 6)
        varphi_n = varphi_n.reshape(-1, 6)
        v_nm1 = v_nm1.reshape(-1, 6)
        a_nm1 = a_nm1.reshape(-1, 6)
        a_n = a_n.reshape(-1, 6)

        # time integrator parameters
        dt = self.time_integrator.dt
        beta = self.time_integrator.beta

        phi_n = dt * v_nm1 + (0.5 - beta) * dt * dt * a_nm1 + beta * dt * dt * a_n
        return vmap(
            lambda vp_n, vp_nm1, phi: log_se3(
                exp_se3(-vp_n) @ exp_se3(vp_nm1) @ exp_se3(phi)
            ),
            0,
            0,
            0,
        )(varphi_n, varphi_nm1, phi_n).ravel()[jnp.array(solve_dofs)]

    def v_res_func(
        self,
        v_nm1: Array,
        v_n: Array,
        a_nm1: Array,
        a_n: Array,
        solve_dofs: tuple[int, ...],
    ) -> Array:
        # time integrator parameters
        dt = self.time_integrator.dt
        gamma = self.time_integrator.gamma

        v_res = (v_nm1 + (1.0 - gamma) * dt * a_nm1 + gamma * dt * a_n - v_n).ravel()[
            jnp.array(solve_dofs)
        ]

        # scale by gamma_prime to scale order of magnitude for a better conditioned problem
        return v_res / self.time_integrator.gamma_prime

    def a_res_func(
        self,
        v_dot_nm1: Array,
        v_dot_n: Array,
        a_nm1: Array,
        a_n: Array,
        solve_dofs: tuple[int, ...],
    ) -> Array:
        # time integrator parameters
        alpha_f = self.time_integrator.alpha_f
        alpha_m = self.time_integrator.alpha_m

        a_res = (
            ((1.0 - alpha_f) * v_dot_n + alpha_f * v_dot_nm1 - alpha_m * a_nm1)
            / (1.0 - alpha_m)
            - a_n
        ).ravel()[jnp.array(solve_dofs)]

        # scale by gamma_prime to scale order of magnitude for a better conditioned problem
        return a_res / self.time_integrator.gamma_prime

    def v_dot_res_func(
        self,
        i_ts: int | Array,
        varphi_nm1: Array,
        varphi_n: Array,
        v_nm1: Array,
        v_n: Array,
        v_dot_nm1: Array,
        v_dot_n: Array,
        dv: StructureDesignVariables,
        f_aero_nm1: Array | None,
        f_aero_n: Array | None,
        thrust_t: dict[str, Array],
        solve_dofs: tuple[int, ...],
        approx_grads: bool,
    ) -> Array:
        varphi_nm1 = varphi_nm1.reshape(-1, 6)
        varphi_n = varphi_n.reshape(-1, 6)
        v_nm1 = v_nm1.reshape(-1, 6)
        v_n = v_n.reshape(-1, 6)
        v_dot_nm1 = v_dot_nm1.reshape(-1, 6)
        v_dot_n = v_dot_n.reshape(-1, 6)
        f_aero_nm1 = f_aero_nm1.reshape(-1, 6) if f_aero_nm1 is not None else None
        f_aero_n = f_aero_n.reshape(-1, 6) if f_aero_n is not None else None

        solve_dofs: Array = jnp.array(solve_dofs)

        # time integrator parameters
        alpha_f = self.time_integrator.alpha_f

        # updates to v_dot, which are obtained from relation to other states through structural problem
        inner_case = self.case_from_dv(
            dv=dv
        )  # allows for gradients w.r.t. design variables

        varphi_alpha = inner_case.time_integrator.compute_varphi_alpha(
            varphi_nm1=varphi_nm1, varphi_n=varphi_n
        )
        v_alpha = inner_case.time_integrator.compute_v_alpha(v_nm1=v_nm1, v_n=v_n)
        v_dot_alpha = inner_case.time_integrator.compute_v_dot_alpha(
            v_dot_nm1=v_dot_nm1, v_dot_n=v_dot_n
        )

        hg_alpha = inner_case.compute_hg_from_varphi(varphi_alpha)

        # obtain forces at alpha
        f_ext_dead_alpha = (
            inner_case.time_integrator.compute_f_alpha(
                f_nm1=dv.f_ext_dead[i_ts - 1, ...], f_n=dv.f_ext_dead[i_ts, ...]
            )
            if dv.f_ext_dead is not None
            else None
        )
        f_ext_follower_alpha = (
            inner_case.time_integrator.compute_f_alpha(
                f_nm1=dv.f_ext_follower[i_ts - 1, ...],
                f_n=dv.f_ext_follower[i_ts, ...],
            )
            if dv.f_ext_follower is not None
            else None
        )

        if f_aero_n is not None and f_aero_nm1 is not None:
            f_aero_alpha = inner_case.time_integrator.compute_f_alpha(
                f_nm1=f_aero_nm1, f_n=f_aero_n
            )
        else:
            f_aero_alpha = None

        if dv.thrust_t is not None:
            thrust_t_: dict[str, Array] = dv.thrust_t
        else:
            # if not included as a design variable, use the input value
            thrust_t_ = thrust_t

        thrust_alpha: dict[str, Array] = {
            k: inner_case.time_integrator.compute_f_alpha(
                f_nm1=v[i_ts - 1], f_n=v[i_ts, ...]
            )
            for k, v in thrust_t_.items()
        }

        # find system properties at alpha
        (
            d_alpha,
            _,
            f_dead_alpha,  # dead force in local frame
            _,
            f_grav_alpha,
            f_int_alpha,
            f_gyr_alpha,
            *_,
        ) = inner_case.resolve_forces(
            hg=hg_alpha,
            dynamic=True,
            f_ext_follower=f_ext_follower_alpha,
            f_ext_dead=f_ext_dead_alpha,
            f_ext_aero=None,  # this is None here, as we already have the aero force in the local frame
            thrust=thrust_alpha,
            v=v_alpha,
            v_dot=v_dot_alpha,
            approx_gradients=approx_grads,
        )

        # use stop gradient to prevent effective stiffness contribution
        m_alpha = inner_case.assemble_matrix_from_entries(
            inner_case.make_m_t(
                d=jax.lax.stop_gradient(d_alpha) if approx_grads else d_alpha
            )
        )
        if inner_case.use_lumped_mass:
            m_alpha = inner_case.add_lumped_contributions_to_arr(
                arr=m_alpha, lumped_arr=inner_case.m_lumped
            )

        # calculate non-inertial forcing residual
        f_res_non_iner = f_int_alpha + f_gyr_alpha
        if inner_case.use_gravity:
            f_res_non_iner += f_grav_alpha
        if (
            f_dead_alpha is not None
        ):  # use the output from the resolve forces function, as it's in the local frame
            f_res_non_iner += f_dead_alpha
        if f_ext_follower_alpha is not None:
            f_res_non_iner += f_ext_follower_alpha
        if f_aero_alpha is not None:
            f_res_non_iner += f_aero_alpha

        # from forcing residual, solve for v_dot which satisfies f_res=0
        # restrict solve to free DOFs to avoid prescribed-DOF reaction forces
        m_alpha = m_alpha[jnp.ix_(solve_dofs, solve_dofs)]

        # find the v_dot residual to satisfy the structural problem
        v_dot_res = f_res_non_iner.ravel()[solve_dofs] / (1.0 - alpha_f) - m_alpha @ (
            alpha_f / (1.0 - alpha_f) * v_dot_nm1.ravel()[solve_dofs]
            + v_dot_n.ravel()[solve_dofs]
        )

        # scale by beta_prime to scale order of magnitude for a better conditioned problem
        return v_dot_res / self.time_integrator.beta_prime

    def timestep_residual(
        self,
        i_ts: int | Array,
        q_nm1: StructureMinimalStates,
        q_n: StructureMinimalStates,
        dv_: StructureDesignVariables,
        thrust_t: dict[str, Array],
        solve_dofs: tuple[int, ...],
        approx_grads: bool,
    ) -> Array:
        r"""
        Routine to compute the full residual for the structural dynamic problem.
        :param i_ts: Time step index.
        :param q_nm1: Previous minimal state.
        :param q_n: Current minimal state.
        :param dv_: Design variables.
        :param thrust_t: Thrust time history, ``{key, (n_tstep, )}``.
        :param solve_dofs: Solve degrees of freedom.
        :param approx_grads: If true, block gradients from some parts of the solution.
        :return: Residual vector, ``(4 * n_solve_dof,)``.
        """
        return jnp.stack(
            (
                self.varphi_res_func(
                    varphi_nm1=q_nm1.varphi,
                    varphi_n=q_n.varphi,
                    v_nm1=q_nm1.v,
                    a_nm1=q_nm1.a,
                    a_n=q_n.a,
                    solve_dofs=solve_dofs,
                ),
                self.v_res_func(
                    v_nm1=q_nm1.v,
                    v_n=q_n.v,
                    a_nm1=q_nm1.a,
                    a_n=q_n.a,
                    solve_dofs=solve_dofs,
                ),
                self.v_dot_res_func(
                    i_ts=i_ts,
                    varphi_nm1=q_nm1.varphi,
                    varphi_n=q_n.varphi,
                    v_nm1=q_nm1.v,
                    v_n=q_n.v,
                    v_dot_nm1=q_nm1.v_dot,
                    v_dot_n=q_n.v_dot,
                    approx_grads=approx_grads,
                    f_aero_nm1=q_nm1.f_ext_aero,
                    f_aero_n=q_n.f_ext_aero,
                    thrust_t=thrust_t,
                    dv=dv_,
                    solve_dofs=solve_dofs,
                ),
                self.a_res_func(
                    v_dot_nm1=q_nm1.v_dot,
                    v_dot_n=q_n.v_dot,
                    a_nm1=q_nm1.a,
                    a_n=q_n.a,
                    solve_dofs=solve_dofs,
                ),
            ),
            axis=0,
        ).ravel()  # [4*n_free_dof]

    def timestep_residual_jacobians(
        self,
        i_ts: int | Array,
        q_nm1: StructureMinimalStates,
        q_n: StructureMinimalStates,
        f_ext_aero_nm1: Array | None,
        f_ext_aero_n: Array | None,
        dv: StructureDesignVariables,
        thrust_t: dict[str, Array],
        solve_dofs: tuple[int, ...],
        approx_grads: bool,
        n_profile_loops: int | None,
        jac_options: dict[str, dict[str, Callable[..., Any] | None]],
        mode: ADMode = "reverse",
    ) -> tuple[
        Array,
        Array,
        StructureDesignVariables,
        Array | None,
        Array | None,
        dict[str, dict[str, float]] | None,
        dict[str, dict[str, float]] | None,
    ]:
        r"""
        Obtain the Jacobians of the structural residual with respect to the current states and previous states.
        :param i_ts: Time step index.
        :param q_nm1: Previous minimal states.
        :param q_n: Current minimal states.
        :param f_ext_aero_nm1: Optional aerodynamic forcing for previous time step, ``(n_nodes, 6)``.
        :param f_ext_aero_n: Optional aerodynamic forcing for current time step, ``(n_nodes, 6)``.
        :param dv: Design variables.
        :param thrust_t: Thrust time history, ``{key, (n_tstep, )}``.
        :param solve_dofs: Index of degrees of freedom to solve for.
        :param approx_grads: If True, remove some gradient terms which are generally small.
        :param n_profile_loops: Number of profile loops to run for timing function. If None, no profiling is done.
        :param jac_options: Input which passes functions which can be used to approximate the Jacobians. If entries are
        None, AD is used.
        :param mode: AD mode used for Jacobian construction.
        :return: Jacobians with respect to previous state and current state, gradients with respect to design variables,
        previous, and current aerodynamic forces respectively, and profiling times for compilation and run time.
        """

        compile_time: dict[str, dict[str, float]] = {}
        run_time: dict[str, dict[str, float]] = {}

        compute_f_aero_grads = f_ext_aero_nm1 is not None and f_ext_aero_n is not None
        if compute_f_aero_grads:
            jac_options["v_dot"].update({"f_aero_nm1": None, "f_aero_n": None})

        # varphi
        d_varphi, compile_time["varphi"], run_time["varphi"] = jacrev_custom(
            func=self.varphi_res_func,
            jac_options=jac_options["varphi"],
            n_profile_loops=n_profile_loops,
            func_name="varphi",
            static_argnames=("solve_dofs",),
            mode=mode,
        )(
            varphi_nm1=q_nm1.varphi.ravel(),
            varphi_n=q_n.varphi.ravel(),
            v_nm1=q_nm1.v.ravel(),
            a_nm1=q_nm1.a.ravel(),
            a_n=q_n.a.ravel(),
            solve_dofs=solve_dofs,
        )

        # velocity
        d_v, compile_time["v"], run_time["v"] = jacrev_custom(
            func=self.v_res_func,
            jac_options=jac_options["v"],
            n_profile_loops=n_profile_loops,
            func_name="v",
            static_argnames=("solve_dofs",),
            mode=mode,
        )(
            v_nm1=q_nm1.v.ravel(),
            v_n=q_n.v.ravel(),
            a_nm1=q_nm1.a.ravel(),
            a_n=q_n.a.ravel(),
            solve_dofs=solve_dofs,
        )

        # acceleration
        d_v_dot, compile_time["v_dot"], run_time["v_dot"] = jacrev_custom(
            func=self.v_dot_res_func,
            jac_options=jac_options["v_dot"],
            n_profile_loops=n_profile_loops,
            func_name="v_dot",
            static_argnames=("solve_dofs", "approx_grads"),
            mode=mode,
        )(
            i_ts=i_ts,
            varphi_nm1=q_nm1.varphi.ravel(),
            varphi_n=q_n.varphi.ravel(),
            v_nm1=q_nm1.v.ravel(),
            v_n=q_n.v.ravel(),
            v_dot_nm1=q_nm1.v_dot.ravel(),
            v_dot_n=q_n.v_dot.ravel(),
            dv=dv,
            f_aero_nm1=f_ext_aero_nm1.ravel() if compute_f_aero_grads else None,  # type: ignore
            f_aero_n=f_ext_aero_n.ravel() if compute_f_aero_grads else None,  # type: ignore
            thrust_t=thrust_t,
            solve_dofs=solve_dofs,
            approx_grads=approx_grads,
        )

        if not compute_f_aero_grads:
            # no Jacobians for aero case, but include a None to keep the keys consistent
            d_v_dot.update({"f_aero_nm1": None, "f_aero_n": None})

        # pseudo-acceleration
        d_a, compile_time["a"], run_time["a"] = jacrev_custom(
            func=self.a_res_func,
            jac_options=jac_options["a"],
            n_profile_loops=n_profile_loops,
            func_name="a",
            static_argnames=("solve_dofs",),
            mode=mode,
        )(
            v_dot_nm1=q_nm1.v_dot.ravel(),
            v_dot_n=q_n.v_dot.ravel(),
            a_nm1=q_nm1.a.ravel(),
            a_n=q_n.a.ravel(),
            solve_dofs=solve_dofs,
        )

        struct_sizes = (
            len(solve_dofs),
            len(solve_dofs),
            len(solve_dofs),
            len(solve_dofs),
        )

        nm1_keys = ("varphi_nm1", "v_nm1", "v_dot_nm1", "a_nm1")
        p_r_n_p_q_nm1 = construct_named_block_jacobian(
            entries=tuple(
                [
                    {k: v[:, solve_dofs] for k, v in jacs.items() if k in nm1_keys}
                    for jacs in (d_varphi, d_v, d_v_dot, d_a)
                ]
            ),
            keys=nm1_keys,
            widths=struct_sizes,
            heights=struct_sizes,
        )

        n_keys = ("varphi_n", "v_n", "v_dot_n", "a_n")
        p_r_n_p_q_n = construct_named_block_jacobian(
            entries=tuple(
                [
                    {k: v[:, solve_dofs] for k, v in jacs.items() if k in n_keys}
                    for jacs in (d_varphi, d_v, d_v_dot, d_a)
                ]
            ),
            keys=n_keys,
            widths=struct_sizes,
            heights=struct_sizes,
        )

        return (
            p_r_n_p_q_nm1,
            p_r_n_p_q_n,
            d_v_dot["dv"],
            d_v_dot["f_aero_nm1"],
            d_v_dot["f_aero_n"],
            compile_time if n_profile_loops is not None else None,
            run_time if n_profile_loops is not None else None,
        )

    def j_from_q_x(
        self,
        q_n_mat: Array,
        dv: StructureDesignVariables,
        dv_full: StructureDesignVariables,
        objective: StructureObjectiveFunction,
        i_ts: int,
    ) -> Array:
        r"""
        Obtain the objective as a function of the minimal states and design variables.
        :param q_n_mat: Matrix representation of the minimal states.
        :param dv: Design variables, with unwanted entries replaced with None.
        :param dv_full: Design variables which are defined for all entries.
        :param objective: Objective function.
        :param i_ts: Time step index.
        :return: Objective value.
        """
        full_states = self.minimal_states_to_full_states(
            i_ts=i_ts,
            q=StructureMinimalStates.from_mat(q_n_mat),
            dv=dv,
            dv_full=dv_full,
        )
        return jnp.atleast_1d(objective(full_states, dv, i_ts))

    @jax.jit(static_argnums=(0, 1, 3, 4))
    def p_j(
        self,
        objective: StructureObjectiveFunction,
        i_ts: int,
        dv: StructureDesignVariables,
        dv_full: StructureDesignVariables,
        q_n: StructureMinimalStates,
    ) -> tuple[Array, StructureDesignVariables]:
        r"""
        Obtains Jacobians of the objective function.
        :param objective: Objective function.
        :param i_ts: Time step index.
        :param dv: Design variables.
        :param dv_full: Design variables which are defined for all entries.
        :param q_n: Current minimal states.
        :return: Jacobian with respect to minimal states and design variables.
        """

        def _j(q_n_mat: Array, dv_: StructureDesignVariables) -> Array:
            return self.j_from_q_x(
                q_n_mat=q_n_mat, dv=dv_, dv_full=dv_full, objective=objective, i_ts=i_ts
            )

        p_j_n_p_q_n, p_j_n_p_x = jax.jacrev(_j, argnums=(0, 1), allow_int=True)(
            q_n.to_mat(), dv
        )

        return cast(Array, p_j_n_p_q_n), cast(StructureDesignVariables, p_j_n_p_x)

    def adjoint_time_loop(
        self,
        rev_i_ts: int,
        d_j_d_x_: StructureDesignVariables,
        adj_: Array,
        p_r_np1_p_q_n: Array | None,
        adj_t_p_r_np1_p_q_n: Array | None,
        q_n: StructureMinimalStates,
        structure: StructureCase,
        objective: StructureObjectiveFunction,
        dv: StructureDesignVariables,
        dv_full: StructureDesignVariables,
        thrust_t: dict[str, Array],
        solve_dofs: tuple[int, ...],
        approx_grads: bool,
        save_adjoint: bool,
        matrix_free: bool,
        n_j: int,
        jac_options: dict[str, dict[str, Callable[..., Any] | None]],
        i_ts_end: int | None = None,
    ) -> tuple[StructureDesignVariables, Array, Array, StructureMinimalStates]:
        r"""
        Function to obtain the grads states at timestep varphi, which is dependent on the grads at timestep varphi+1.
        :param rev_i_ts: Reversed timestep index. JAX loop does not allow for reverse indexing, and so this is.
        explicitly reversed within the function body to obtain i_ts.
        :param d_j_d_x_: Design gradient to accumulate.
        :param adj_: Full grads matrix which is updated inplace, ``(n_tstep, *j_shape, 5*n_dof)``.
        :param p_r_np1_p_q_n: Gradient of future step with respect to current state, used when computing the full
        Jacobian ``(5*n_dof, 5*n_dof)``.
        :param adj_t_p_r_np1_p_q_n: VJP of the future adjoint step and the Jacobian of the future residual with respect
        to the current state, ``(n_adj_dof, )``.
        :param q_n: Current minimal states.
        :param structure: Dynamic structure solution.
        :param objective: Objective function.
        :param dv: Structure design variables.
        :param dv_full: Structure design variables which are defined for all entries.
        :param thrust_t: Thrust time history, ``{key, (n_tstep, )}``.
        :param solve_dofs: Tuple of dof index to solve.
        :param approx_grads: Whether to approximate the gradient or not.
        :param save_adjoint: Whether to save the full adjoint time history.
        :param matrix_free: If False, solve the system using the residual Jacobian-vector product using GMRES.
        :param n_j: Number of objective function outputs.
        :param jac_options: Input which passes functions which can be used to approximate the Jacobians. If entries are
        None, AD is used.
        :param i_ts_end: Largest time step index for which the adjoint is computed. Defaults to
        ``structure.n_tstep - 1`` when ``None``
        :return: Updated grads matrix, gradient of current step with respect to previous state and current state.
        """

        i_ts_end_ = structure.n_tstep - 1 if i_ts_end is None else i_ts_end
        i_ts = i_ts_end_ - rev_i_ts  # index for timestep n, which decrements

        i_ts_nm1 = jnp.maximum(i_ts - 1, 0)  # index for timestep varphi-1

        solve_idx = jnp.array(solve_dofs)

        # find minimal states for timestep varphi-1
        q_nm1 = structure.get_minimal_states(i_ts_nm1)

        # Objective sensitivities
        p_j_n_p_q_n, p_j_n_p_x = self.p_j(
            objective=objective, i_ts=i_ts, dv=dv, dv_full=dv_full, q_n=q_n
        )

        if matrix_free:

            def _residual_states(
                q_n_: StructureMinimalStates, q_nm1_: StructureMinimalStates
            ):
                return self.timestep_residual(
                    i_ts=i_ts,
                    q_nm1=q_nm1_,
                    q_n=q_n_,
                    dv_=dv,
                    thrust_t=thrust_t,
                    solve_dofs=solve_dofs,
                    approx_grads=approx_grads,
                )

            # Linearise the timestep residual around (q_n, q_nm1). This single VJP returns:
            # p_r_n_dot_v(v)[0] = (p_r_n/p_q_n).T @ v, p_r_n_dot_v(v)[1] = (p_r_n/p_q_nm1).T @ v
            _, p_r_n_dot_v = jax.vjp(_residual_states, q_n, q_nm1)

            def _cot_to_solve_vec(cot: StructureMinimalStates) -> Array:
                # collapse cotangent to [n_adj_dof]
                mat = cot.to_mat()  # [4, n_nodes, 6]
                return mat.reshape(mat.shape[0], -1)[:, solve_idx].ravel()

            def matvec_qn_t(v: Array) -> Array:
                # function to compute (p_r_n/p_q_n).T @ v for some vector v
                return _cot_to_solve_vec(p_r_n_dot_v(v)[0])

            # sensitivity of objective to degrees of freedom, (n_j, n_adj_dof)
            p_j_solve = (
                p_j_n_p_q_n.reshape(n_j, 4, -1, 6)
                .reshape(n_j, 4, -1)[..., solve_idx]
                .reshape(n_j, -1)
            )
            assert adj_t_p_r_np1_p_q_n is not None, (
                "The adjoint-Jacobian product has not been passed"
            )
            b_rhs = -(p_j_solve + adj_t_p_r_np1_p_q_n)  # (n_j, n_adj_dof)

            # solve for the adjoint vector at timestep n, batched along the size of the objective.
            def _solve_row(b_row: Array) -> Array:
                # noinspection PyTypeChecker
                x, _ = jax.scipy.sparse.linalg.gmres(
                    matvec_qn_t,
                    b_row,
                    tol=1e-10,
                    atol=1e-10,
                    maxiter=50,
                    solve_method="batched",
                )
                return x

            adj_n = jax.vmap(_solve_row)(b_rhs)  # (n_j, n_adj_dof)

            # Design gradient accumulation via a separate VJP to obtain adj.T @ p_r_v_dot_n_p_dv.
            def _residual_dv(dv_: StructureDesignVariables) -> Array:
                return self.timestep_residual(
                    i_ts=i_ts,
                    q_nm1=q_nm1,
                    q_n=q_n,
                    dv_=dv_,
                    thrust_t=thrust_t,
                    solve_dofs=solve_dofs,
                    approx_grads=approx_grads,
                )

            _, pull_dv = jax.vjp(_residual_dv, dv)
            dv_grads = jax.vmap(pull_dv)(adj_n)[0]

            # accumulate with seperate statements as there is no __add__ member
            d_j_d_x_ += dv_grads
            d_j_d_x_ += p_j_n_p_x

            # compute adj_n @ p_r_n/p_q_nm1 for next iteration
            def _coupling_row(a: Array) -> Array:
                _, cot_qnm1 = p_r_n_dot_v(a)
                return _cot_to_solve_vec(cot_qnm1)

            adj_t_p_r_n_p_q_nm1 = jax.vmap(_coupling_row)(adj_n)  # (n_j, n_adj_dof)

            p_r_n_p_q_nm1: Array | None = None  # unused
        else:
            # find gradients of residual function (state Jacobians only)
            p_r_n_p_q_nm1, p_r_n_p_q_n, p_r_v_dot_n_p_dv, *_ = (
                self.timestep_residual_jacobians(
                    i_ts=i_ts,
                    q_n=q_n,
                    q_nm1=q_nm1,
                    dv=dv,
                    solve_dofs=solve_dofs,
                    approx_grads=approx_grads,
                    f_ext_aero_nm1=None,
                    f_ext_aero_n=None,
                    thrust_t=thrust_t,
                    n_profile_loops=None,
                    jac_options=jac_options,
                )
            )

            # solve for adjoint at current timestep
            prev_adjoint = adj_[i_ts + 1, ...] if save_adjoint else adj_
            b: Array = -(p_j_n_p_q_n.reshape(n_j, -1) + prev_adjoint @ p_r_np1_p_q_n).T
            adj_n = jnp.linalg.solve(p_r_n_p_q_n.T, b).T

            # accumulate design derivative
            d_j_d_x_ += p_r_v_dot_n_p_dv.premultiply_adj(
                adj_n[:, solve_idx + 2 * len(solve_dofs)]
            )

            # add on direct contribution from objective
            d_j_d_x_ += p_j_n_p_x

            adj_t_p_r_n_p_q_nm1 = None  # unused

        # update adjoint vector time history if requested
        if save_adjoint:
            adj_ = adj_.at[i_ts, ...].set(adj_n)

        # print to console
        jax_print(
            "Adjoint step: {i_ts}",
            i_ts=i_ts,
            verbose_level="normal",
        )

        if matrix_free:
            assert adj_t_p_r_n_p_q_nm1 is not None
            return d_j_d_x_, adj_ if save_adjoint else adj_n, adj_t_p_r_n_p_q_nm1, q_nm1
        else:
            assert p_r_n_p_q_nm1 is not None
            return d_j_d_x_, adj_ if save_adjoint else adj_n, p_r_n_p_q_nm1, q_nm1

    def construct_approximate_jacobians(
        self,
        sol: StructureCase,
        jacobian_approximations: StructureJacobianApproximations,
    ) -> dict[str, dict[str, Callable[..., Any] | None]]:
        r"""
        Compute approximations for Jacobians which are specified in the jacobian_approximations data structure.
        :param sol: Solution for which approximations will be created for the initial time step.
        :param jacobian_approximations: Data structure which defines which approximations to create.
        :return: Dictionary of approximations.
        """
        q_nm1 = sol.get_minimal_states(0)
        q_n = sol.get_minimal_states(1)
        dv = self.get_design_variables(
            struct_case=sol, thrust_t=sol.thrust, grads_to_compute=None
        )
        solve_dofs = tuple(
            int(i)
            for i in get_solve_dofs(
                n_dof=self.n_dof, prescribed_dofs=sol.prescribed_dofs
            )
        )
        if sol.f_ext_aero is not None:
            f_aero_nm1 = sol.f_ext_aero[0, ...].ravel()
            f_aero_n = sol.f_ext_aero[1, ...].ravel()
        else:
            f_aero_nm1 = None
            f_aero_n = None

        res_args: dict[
            str, tuple[Callable[..., Array], dict[str, Any], Sequence[str]]
        ] = {
            "varphi": (
                self.varphi_res_func,
                {
                    "varphi_nm1": q_nm1.varphi.ravel(),
                    "varphi_n": q_n.varphi.ravel(),
                    "v_nm1": q_nm1.v.ravel(),
                    "a_nm1": q_nm1.a.ravel(),
                    "a_n": q_n.a.ravel(),
                    "solve_dofs": solve_dofs,
                },
                [f.name for f in fields(VarphiApprox)],
            ),
            "v": (
                self.v_res_func,
                {
                    "v_nm1": q_nm1.v.ravel(),
                    "v_n": q_n.v.ravel(),
                    "a_nm1": q_nm1.a.ravel(),
                    "a_n": q_n.a.ravel(),
                    "solve_dofs": solve_dofs,
                },
                [f.name for f in fields(VApprox)],
            ),
            "v_dot": (
                self.v_dot_res_func,
                {
                    "i_ts": 1,
                    "varphi_nm1": q_nm1.varphi.ravel(),
                    "varphi_n": q_n.varphi.ravel(),
                    "v_nm1": q_nm1.v.ravel(),
                    "v_n": q_n.v.ravel(),
                    "v_dot_nm1": q_nm1.v_dot.ravel(),
                    "v_dot_n": q_n.v_dot.ravel(),
                    "dv": dv,
                    "f_aero_nm1": f_aero_nm1,
                    "f_aero_n": f_aero_n,
                    "thrust_t": sol.thrust,
                    "solve_dofs": solve_dofs,
                    "approx_grads": True,
                },
                [f.name for f in fields(VDotApprox)],
            ),
            "a": (
                self.a_res_func,
                {
                    "v_dot_nm1": q_nm1.v_dot.ravel(),
                    "v_dot_n": q_n.v_dot.ravel(),
                    "a_nm1": q_nm1.a.ravel(),
                    "a_n": q_n.a.ravel(),
                    "solve_dofs": solve_dofs,
                },
                [f.name for f in fields(AApprox)],
            ),
        }

        return construct_approximation(
            res_args=res_args, jacobian_approximations=jacobian_approximations
        )

    JACOBIAN_APPROXIMATIONS_DEFAULT = StructureJacobianApproximations()
    GRADS_TO_COMPUTE_DEFAULT = StructureGradsToCompute(
        x0=False,
        k_cs=True,
        m_cs=True,
        m_lumped=False,
        f_ext_follower=False,
        f_ext_dead=False,
    )

    def dynamic_adjoint(
        self,
        structure: StructureCase,
        objective: StructureObjectiveFunction,
        matrix_free: bool = False,
        jacobian_approximations: StructureJacobianApproximations = JACOBIAN_APPROXIMATIONS_DEFAULT,
        p_q0_p_x: StructureDesignVariables | None = None,
        save_adjoint: bool = False,
        approx_grads: bool = True,
        grads_to_compute: StructureGradsToCompute = GRADS_TO_COMPUTE_DEFAULT,
        i_ts_adjoint_range: tuple[int | None, int | None] = (None, None),
    ) -> tuple[StructureDesignVariables, Array | None]:
        r"""
        Dynamic structure grads problem. This computes the gradient of the objective of the dynamic response with
        respect to design variables. The objective has structure
        :math:`J = \sum_{i=1}^N \left(j(\mathbf{x}, \mathbf{y}_i)\right)` where :math:`\mathbf{x}` are the design variables
        and :math:`\mathbf{y}` are the structural states at each timestep, which depend on the design variables through
        the dynamic structure equations. The gradient is computed by first solving a backward pass to obtain the grads
        states, and then using these to compute the gradient w.r.t. design variables in a forward pass.
        :param structure: Dynamic structure solution object.
        :param objective: Objective function :math:`j(\mathbf{x}, \mathbf{y}_i)`.
        :param matrix_free: Whether to use matrix-free methods for solving the linear systems. Default is False, as
        structural problems generally do not benefit from this solve.
        :param jacobian_approximations: Data structure which specifies Jacobian approximations to use for each part of
        the problem. The value can either be None for no approximation, `constant` for the assumption that the Jacobian
        does not vary with any variables. Alternatively, it can be tuple pairs with first entry being `dense_linear` or
        `lazy_linear`, with the second entry being a sequence of argument names for which to obtain the Hessian.
        :param p_q0_p_x: Optional Jacobian used to describe the sensitivities of the initial structural degrees of
        freedom to the design variables.
        :param save_adjoint: Whether to save the full adjoint vectors.
        :param approx_grads: If true, some gradient contributions which are assumed to be near-zero are removed to
        decrease computational cost.
        :param grads_to_compute: Design variables with which to compute design gradients for.
        :param i_ts_adjoint_range: Optional ``(start, end)`` window of time step indices over which the objective
        contributes to the gradient. Either entry may be ``None`` to leave that side untruncated. Defining a start
        time step that is nonzero will  skip the initial state adjoint contribution.
        :return: Objective gradient :math:`\frac{dJ}{d\mathbf{x}}` and adjoint states
        """

        dv = self.get_design_variables(
            struct_case=structure,
            thrust_t=structure.thrust,
            grads_to_compute=grads_to_compute,
        )

        dv_full = self.get_design_variables(
            struct_case=structure, thrust_t=structure.thrust, grads_to_compute=None
        )

        struct_states_init = structure.get_full_states(i_ts=0)

        j_properties = jax.eval_shape(
            lambda: jnp.atleast_1d(objective(struct_states_init, dv, None))
        )
        j_shape = j_properties.shape
        n_j = j_properties.size

        # assemble
        solve_dofs: tuple[int, ...] = tuple(
            int(i)
            for i in get_solve_dofs(
                n_dof=self.n_dof, prescribed_dofs=structure.prescribed_dofs
            )
        )

        dv_grad_init = StructureDesignVariables(
            x0=jnp.zeros((*j_shape, *self.x0.shape)) if dv.x0 is not None else None,
            orientation_euler=jnp.zeros((*j_shape, 3))
            if dv.orientation_euler is not None
            else None,
            k_cs=jnp.zeros((*j_shape, *self.k_cs.shape))
            if dv.k_cs is not None
            else None,
            m_cs=jnp.zeros((*j_shape, *self.m_cs.shape))
            if dv.m_cs is not None
            else None,
            m_lumped=jnp.zeros((*j_shape, *self.m_lumped.shape))
            if self.use_lumped_mass and dv.m_lumped is not None
            else None,
            f_ext_dead=jnp.zeros((*j_shape, *structure.f_ext_dead.shape))
            if structure.f_ext_dead is not None and dv.f_ext_dead is not None
            else None,
            f_ext_follower=jnp.zeros((*j_shape, *structure.f_ext_follower.shape))
            if structure.f_ext_follower is not None and dv.f_ext_follower is not None
            else None,
            thrust_t={
                k: jnp.zeros((*j_shape, *v.shape)) for k, v in structure.thrust.items()
            }
            if dv.thrust_t is not None
            else None,
            f_shape=(),
        )

        n_adj_dof = 4 * (
            self.n_dof - len(structure.prescribed_dofs)
        )  # number of grads degrees of freedom

        # compute Jacobian approximations, if requested
        jac_options = self.construct_approximate_jacobians(
            sol=structure, jacobian_approximations=jacobian_approximations
        )

        # check adjoint window
        i_ts_start_adj, i_ts_end_adj = i_ts_adjoint_range
        i_ts_start_adj_: int = 1 if i_ts_start_adj is None else i_ts_start_adj
        i_ts_end_adj_: int = (
            structure.n_tstep - 1 if i_ts_end_adj is None else i_ts_end_adj
        )
        if i_ts_start_adj_ < 1:
            raise ValueError(
                f"i_ts_adjoint_range start must be >= 1, got {i_ts_start_adj_}"
            )
        if i_ts_end_adj_ > structure.n_tstep - 1:
            raise ValueError(
                f"i_ts_adjoint_range end must be <= n_tstep - 1 = "
                f"{structure.n_tstep - 1}, got {i_ts_end_adj_}"
            )
        if i_ts_end_adj_ < i_ts_start_adj_:
            raise ValueError(
                f"i_ts_adjoint_range end ({i_ts_end_adj_}) must be >= start "
                f"({i_ts_start_adj_})"
            )
        n_adj_iters: int = i_ts_end_adj_ - i_ts_start_adj_ + 1

        # wrap in a local JIT so structure/aero_dv become closure constants
        @jax.jit
        def adjoint_step(
            rev_i_ts_: int,
            d_j_d_x_: StructureDesignVariables,
            adj_: Array,
            coupling_arr: Array,
            q_n: StructureMinimalStates,
        ) -> tuple[StructureDesignVariables, Array, Array, StructureMinimalStates]:
            return self.adjoint_time_loop(
                rev_i_ts=rev_i_ts_,
                d_j_d_x_=d_j_d_x_,
                adj_=adj_,
                p_r_np1_p_q_n=None if matrix_free else coupling_arr,
                adj_t_p_r_np1_p_q_n=coupling_arr if matrix_free else None,
                q_n=q_n,
                structure=structure,
                objective=objective,
                dv=dv,
                dv_full=dv_full,
                thrust_t=structure.thrust,
                solve_dofs=solve_dofs,
                approx_grads=approx_grads,
                save_adjoint=save_adjoint,
                matrix_free=matrix_free,
                n_j=n_j,
                jac_options=jac_options,
                i_ts_end=i_ts_end_adj_,
            )

        # coupling array is either a Jacobian or a VJP depending on if using matrix free or not
        coupling_init = (
            jnp.zeros((n_j, n_adj_dof))
            if matrix_free
            else jnp.zeros((n_adj_dof, n_adj_dof))
        )

        # pass through time steps backwards to obtain adjoints
        # coupling0 is p_r1_p_q0 when matrix_free is False, and adj_1 @ p_r1_p_q0 when matrix_free is True
        d_j_d_x, adj, coupling0, _ = jax.lax.fori_loop(
            lower=0,
            upper=n_adj_iters,
            body_fun=lambda i_ts_, args: adjoint_step(i_ts_, *args),
            init_val=(
                dv_grad_init,
                jnp.zeros((structure.n_tstep + 1, n_j, n_adj_dof))
                if save_adjoint
                else jnp.zeros((n_j, n_adj_dof)),
                coupling_init,
                structure.get_minimal_states(i_ts_end_adj_),
            ),
        )

        # solve initial timestep adjoint, as there is no r0. Skipped when the adjoint window truncates early time steps
        if i_ts_start_adj_ <= 1:
            p_j0_p_q0, p_j0_p_x = self.p_j(
                objective=objective,
                i_ts=0,
                dv=dv,
                dv_full=dv_full,
                q_n=structure.get_minimal_states(0),
            )

            if matrix_free:
                adj0 = -p_j0_p_q0.reshape(n_j, -1) - coupling0
            else:
                adj0 = (
                    -p_j0_p_q0.reshape(n_j, -1)
                    - (adj[1, ...] if save_adjoint else adj) @ coupling0
                )

            # add initial direct sensitivity
            d_j_d_x += p_j0_p_x

            # include initial state sensitivity
            if p_q0_p_x is not None:
                d_j_d_x += p_q0_p_x.premultiply_adj(-adj0)
        else:
            adj0 = jnp.zeros((n_j, n_adj_dof))

        # restore original shape of j, and cut off zeros for past-end timestep
        if save_adjoint:
            adj = adj.at[0, ...].set(adj0)
            return d_j_d_x, adj.reshape(adj.shape[0], *j_shape, *adj.shape[2:])[:-1]
        else:
            return d_j_d_x, None

    def dynamic_adjoint_jacobian_profile(
        self,
        sol: StructureCase,
        approx_grads: bool,
        jacobian_approximations: StructureJacobianApproximations = JACOBIAN_APPROXIMATIONS_DEFAULT,
        grads_to_compute: StructureGradsToCompute | None = None,
        f_aero_nm1_n: tuple[Array, Array] | None = None,
        i_ts: int = 1,
        n_loop: int = 10,
        *,
        print_header: bool = True,
    ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
        r"""
        Function to time evaluation of the Jacobians used for the adjoint solution.
        :param sol: Dynamic structural solution to extract states from.
        :param approx_grads: If True, neglect small gradient terms.
        :param jacobian_approximations: Data structure which specifies Jacobian approximations to use for each part of
        the problem.
        :param grads_to_compute: StructureGradsToCompute object which describes which design gradients to compute. If
        None, all gradients will be computed.
        :param f_aero_nm1_n: Tuple of [f_aero_nm1, f_aero_n] which are passed from the aero problem. If None, no
        aerodynamic force gradients will be computed.
        :param i_ts: Time step index where to evaluate residual Jacobians.
        :param n_loop: Number of times to loop the Jacobian evaluation time for averaging the runtime.
        :param print_header: Flag used to prevent heading printer when called by the coupled profiler.
        :return: Dictionary of {residual_name: {gradient_argument: val}} for compile time and run time respectively.
        """

        if print_header:
            print_table_title(inner_width=95, title="Structure Adjoint Profile")

        # compute Jacobian approximations, if requested
        jac_options = self.construct_approximate_jacobians(
            sol=sol, jacobian_approximations=jacobian_approximations
        )

        common_kwargs = {
            "i_ts": i_ts,
            "q_nm1": sol.get_minimal_states(i_ts - 1),
            "q_n": sol.get_minimal_states(i_ts),
            "dv": self.get_design_variables(
                struct_case=sol, thrust_t=sol.thrust, grads_to_compute=grads_to_compute
            ),
            "thrust_t": sol.thrust,
            "solve_dofs": tuple(
                get_solve_dofs(n_dof=self.n_dof, prescribed_dofs=sol.prescribed_dofs)
            ),
            "approx_grads": approx_grads,
            "n_profile_loops": n_loop,
            "jac_options": jac_options,
        }

        if f_aero_nm1_n is not None:
            *_, compile_time, run_time = self.timestep_residual_jacobians(
                f_ext_aero_nm1=f_aero_nm1_n[0],
                f_ext_aero_n=f_aero_nm1_n[1],
                **common_kwargs,
            )
        else:
            *_, compile_time, run_time = self.timestep_residual_jacobians(
                f_ext_aero_nm1=None,
                f_ext_aero_n=None,
                **common_kwargs,
            )

        if print_header:
            print_table_line(inner_width=95)

        assert compile_time is not None and run_time is not None, (
            "No output timings passed"
        )

        return compile_time, run_time
