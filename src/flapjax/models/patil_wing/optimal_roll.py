"""
Patil Wing Optimal Roll Manoeuvre

Optimise a pair of control-surface velocity profiles so that a highly
flexible cantilever wing reaches a target roll angle. The optimisation uses
gradients of the roll-angle objective with respect to the control surface
velocities, obtained from the coupled dynamic adjoint, and drives an SLSQP
solver subject to deflection and rate limits.
"""

from __future__ import annotations

from pathlib import Path

import jax
import matplotlib
import numpy as np
from jax import Array
from jax import numpy as jnp
from scipy.optimize import LinearConstraint, minimize

matplotlib.use("Agg")  # no display on HPC compute nodes
from matplotlib import pyplot as plt

from flapjax.aero.gradients.data_structures import AeroGradsToCompute
from flapjax.aero.utils import cs_vel_to_cs_ang
from flapjax.coupled.data_structures import (
    AeroelasticDesignVariables,
    AeroelasticFullStates,
)
from flapjax.coupled.gradients.data_structures import AeroelasticGradsToCompute
from flapjax.models.patil_wing.patil_wing import generate_patil_wing
from flapjax.structure.gradients.data_structures import StructureGradsToCompute
from flapjax.utils.data_structures import ConvergenceSettings

# define two problems - a short one for quick evaluation, and a full one for a
# more accurate solution

# quick case - for fast local iteration
CONTROL_TIME: float = 1.2
PHYSICAL_TIME: float = 2.0  # total simulation time
N_NODES: int = 21  # number of structural nodes
M_STAR: int = 6  # wake length
N_ITER: int = 20  # number of optimisation iterations
ROLL_ANG_OBJECTIVE: float = float(jnp.deg2rad(5.0))  # desired roll angle

# full size case - relatively slow to compute; run on HPC with the dense adjoint
# (see matrix_free=False below)
# CONTROL_TIME: float = 4.0
# PHYSICAL_TIME: float = 10.0
# N_NODES: int = 49
# M_STAR: int = 20
# N_ITER: int = 75
# ROLL_ANG_OBJECTIVE: float = float(jnp.deg2rad(30.0))

MAX_CS_DEFLECTION = jnp.deg2rad(10.0)  # control angle limit
MAX_CS_VELOCITY = jnp.deg2rad(15.0)  # control velocity limit

# directory for saving outputs
OUTPUT_DIR = Path("./optimal_roll_outputs/")


def save_dat(path: Path, header: list[str], columns: list[Array]) -> None:
    data = np.column_stack([np.asarray(column) for column in columns])
    np.savetxt(path, data, header=" ".join(header))


def main() -> None:
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # create wing model
    wing = generate_patil_wing(sigma=1.1, m_star=M_STAR, n_nodes=N_NODES)
    wing.aero.batch_size = (
        256  # larger values should compute faster but require more RAM
    )
    dt = float(wing.aero.dt)
    delta = dt * wing.aero.flowfield.u_inf_mag
    delta_w = delta * jnp.logspace(0.0, 1.2, M_STAR)

    wing.aero.set_design_variables(
        dt=dt,
        flowfield=wing.aero.flowfield,
        zeta_b0=wing.aero.zeta_b0,
        hg0=wing.aero.hg_ref,
        delta_w=delta_w,
        reference_cs_angles={
            "left_aileron": jnp.zeros(1),
            "right_aileron": jnp.zeros(1),
        },
    )

    # convergence settings, made stricter than default to improve adjoint accuracy
    wing.structure.struct_convergence_settings = ConvergenceSettings(
        max_n_iter=25,
        rel_disp_tol=1e-8,
        abs_disp_tol=1e-10,
        rel_force_tol=1e-8,
        abs_force_tol=1e-10,
    )
    wing.fsi_convergence_settings = ConvergenceSettings(
        max_n_iter=25,
        rel_disp_tol=1e-5,
        abs_disp_tol=1e-7,
        rel_force_tol=1e-5,
        abs_force_tol=1e-7,
    )

    # static solve and trim: fix the middle node's degrees of freedom for trim
    # to create a well-posed problem. The pitch angle at which the wing is
    # fixed is adjusted by the trim routine until the lift equals the weight
    prescribed_dofs_static = jnp.arange(6) + (wing.structure.n_nodes - 1) * 3

    wing.static_solve(prescribed_dofs=prescribed_dofs_static, horseshoe=False)

    trimmed_sol, _trim_vars = wing.trim(
        prescribed_dofs=prescribed_dofs_static,
        zero_force_dofs=prescribed_dofs_static[2],
        trim_cs=None,
        thrust_nodes=None,
        trim_orientation="y",
        horseshoe=False,
        trim_relaxation=0.5,
        trim_f_abs_tolerance=1e-5,
    )

    # dynamic problem setup: fix 5 degrees of freedom on the middle node,
    # leaving the model free in the roll direction
    n_tstep = int(PHYSICAL_TIME // dt)
    n_ctrl_tstep = int(CONTROL_TIME // dt)
    t = jnp.arange(n_tstep) * dt

    prescribed_dofs_dynamic = prescribed_dofs_static[jnp.array((0, 1, 2, 4, 5))]

    prescribed_dofs_dynamic_list = prescribed_dofs_dynamic.tolist()
    mid_node = (wing.structure.n_nodes - 1) // 2  # central node index

    def reconstruct_dict(states: Array) -> dict[str, Array]:
        # Split vector of degrees of freedom into two dictionaries for the
        # control surfaces.
        return {
            "left_aileron": states[: n_ctrl_tstep - 1],
            "right_aileron": states[n_ctrl_tstep - 1 :],
        }

    def extend_control(cs_vel_ctrl: Array) -> Array:
        # fix velocity at zero after the control window so the angle stays
        # constant in the no-control period.
        if n_tstep > n_ctrl_tstep:
            return jnp.concatenate((cs_vel_ctrl, jnp.zeros(n_tstep - n_ctrl_tstep)))
        else:
            return cs_vel_ctrl

    def objective(
        states: AeroelasticFullStates,
        dv: AeroelasticDesignVariables,
        i_ts: int,
    ) -> Array:
        # squared error between the roll angle at the wing's centre node and
        # the target roll angle, scaled up once the control window has closed
        roll_angle = states.structure.varphi[
            mid_node, 3
        ]  # extract the centre node roll angle

        angle_error = roll_angle - ROLL_ANG_OBJECTIVE
        ang_squared_obj = (angle_error**2) / (ROLL_ANG_OBJECTIVE**2)
        assert dv.aero.cs_vel_t is not None

        angle_squared_weight_ = jax.lax.select(i_ts < n_ctrl_tstep, 1.0, 10.0)

        return ang_squared_obj * angle_squared_weight_ / n_tstep

    def control_to_full(cs_vel_ctrl_no_zero: Array) -> tuple[Array, Array]:
        # at timestep zero, we add that the velocity here must be zero
        cs_vel_ctrl_ = jnp.concatenate((jnp.zeros(1), cs_vel_ctrl_no_zero))
        cs_vel_full_ = extend_control(cs_vel_ctrl_)

        # compute angles from velocities
        cs_ang_full_ = cs_vel_to_cs_ang(cs_vel_t={"_": cs_vel_full_}, dt=dt)["_"]
        return cs_ang_full_[1:], cs_vel_full_[1:]

    @jax.jit
    def compute(cs_vel_t_vec_norm: Array):
        # compute the primal solution and the derivative of the cost function
        cs_vel_ctrl = {
            k_: jnp.concatenate((jnp.zeros(1), v_))
            for k_, v_ in reconstruct_dict(
                states=cs_vel_t_vec_norm * MAX_CS_VELOCITY
            ).items()
        }
        cs_vel_t = {k_: extend_control(v_) for k_, v_ in cs_vel_ctrl.items()}

        # integrate velocity to angle; pass both so the solver doesn't
        # recompute via finite diff
        cs_ang_t = cs_vel_to_cs_ang(cs_vel_t=cs_vel_t, dt=dt)

        # solve primal
        dynamic_sol_ = wing.dynamic_solve(
            init_case=trimmed_sol,
            prescribed_dofs=prescribed_dofs_dynamic_list,
            n_tstep=n_tstep,
            cs_ang_t=cs_ang_t,
            cs_vel_t=cs_vel_t,
        )

        # solve tangent
        grads, objective_val_, _adjoint = wing.dynamic_adjoint(
            case=dynamic_sol_,
            objective=objective,
            matrix_free=False,
            save_adjoint=False,
            grads_to_compute=AeroelasticGradsToCompute(
                structure=StructureGradsToCompute(k_cs=False, m_cs=False),
                aero=AeroGradsToCompute(zeta_b0=False, cs_ang_t=True, cs_vel_t=True),
            ),
        )

        total_obj = jnp.sum(objective_val_)

        # extract partial gradients w.r.t. the solver's independent inputs
        assert grads.aero.cs_ang_t is not None and grads.aero.cs_vel_t is not None
        p_obj_p_cs_ang_t: dict[str, Array] = {
            k_: v_.ravel()[1:] for k_, v_ in grads.aero.cs_ang_t.items()
        }
        p_obj_p_cs_vel_t: dict[str, Array] = {
            k_: v_.ravel()[1:] for k_, v_ in grads.aero.cs_vel_t.items()
        }

        # chain partial gradients through the zero-padded extension
        d_obj_d_cs_vel: dict[str, Array] = {}
        for k, value in p_obj_p_cs_vel_t.items():
            _, vjp_fn = jax.vjp(control_to_full, cs_vel_ctrl[k][1:])
            (chain_term,) = vjp_fn((p_obj_p_cs_ang_t[k], value))
            d_obj_d_cs_vel[k] = chain_term

        # flatten gradient in the same order as x0 and normalise
        grad = (
            jnp.concatenate([d_obj_d_cs_vel[k] for k in p_obj_p_cs_vel_t])
            * MAX_CS_VELOCITY
        )

        return (
            total_obj,
            grad,
            objective_val_,
            cs_ang_t,
            cs_vel_t,
            dynamic_sol_,
            p_obj_p_cs_ang_t,
            p_obj_p_cs_vel_t,
            d_obj_d_cs_vel,
        )

    # initialise the control surfaces to be at zero
    cs_vel_init = {
        "left_aileron": jnp.zeros(n_ctrl_tstep - 1),
        "right_aileron": jnp.zeros(n_ctrl_tstep - 1),
    }

    i_iter = 0

    def get_cs_grad(cs_vel_t_vec_norm: Array) -> tuple[Array, Array]:
        nonlocal i_iter

        (
            total_obj,
            grad,
            objective_val_,
            cs_ang_t,
            cs_vel_t,
            dynamic_sol_,
            p_obj_p_cs_ang_t,
            p_obj_p_cs_vel_t,
            d_obj_d_cs_vel,
        ) = compute(cs_vel_t_vec_norm)

        iter_path = output_dir.joinpath(f"iteration_{i_iter}/")
        iter_path.mkdir(parents=True, exist_ok=True)

        if not i_iter % 10:
            # plot solution to VTK files
            dynamic_sol_.plot(iter_path.joinpath("vtk"))

        # control surface angle plot
        cs_ang_keys = list(cs_ang_t.keys())
        _, ax = plt.subplots()
        ax.set_title(f"Control surface angles at iteration {i_iter}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Control angle (deg)")
        for k in cs_ang_keys:
            ax.plot(t, jnp.rad2deg(cs_ang_t[k]), label=k)
        ax.legend()
        plt.savefig(iter_path.joinpath("cs_ang.png"))
        save_dat(
            iter_path.joinpath("cs_ang.dat"),
            header=["t"] + cs_ang_keys,
            columns=[t] + [jnp.rad2deg(cs_ang_t[k]) for k in cs_ang_keys],
        )

        # control surface velocity plot
        cs_vel_keys = list(cs_vel_t.keys())
        _, ax = plt.subplots()
        ax.set_title(f"Control surface velocity at iteration {i_iter}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Control velocity (deg/s)")
        for k in cs_vel_keys:
            ax.plot(t, jnp.rad2deg(cs_vel_t[k]), label=k)
        ax.legend()
        plt.savefig(iter_path.joinpath("cs_vel.png"))
        save_dat(
            iter_path.joinpath("cs_vel.dat"),
            header=["t"] + cs_vel_keys,
            columns=[t] + [jnp.rad2deg(cs_vel_t[k]) for k in cs_vel_keys],
        )

        # cost function integrand
        _, ax = plt.subplots()
        ax.plot(t[1:], objective_val_[1:])
        ax.hlines(0.0, xmin=t[1], xmax=t[-1])
        ax.set_title(
            f"Objective integrands at iteration {i_iter}, "
            f"objective = {float(total_obj):02f}"
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Objective integrand")
        plt.savefig(iter_path.joinpath("objective.png"))
        save_dat(
            iter_path.joinpath("objective.dat"),
            header=["t", "objective_integrand"],
            columns=[t[1:], objective_val_[1:]],
        )

        # roll angle over time plot
        roll_angle_t = jnp.rad2deg(dynamic_sol_.structure.varphi[1:, mid_node, 3])
        _, ax = plt.subplots()
        ax.plot(t[1:], roll_angle_t)
        ax.hlines(jnp.rad2deg(ROLL_ANG_OBJECTIVE), xmin=t[1], xmax=t[-1])
        ax.set_title(
            f"Roll history at iteration {i_iter}, objective = {float(total_obj):02f}"
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Roll angle (deg)")
        plt.savefig(iter_path.joinpath("roll.png"))
        save_dat(
            iter_path.joinpath("roll.dat"),
            header=["t", "roll_angle_deg"],
            columns=[t[1:], roll_angle_t],
        )

        # plot partial and total derivatives
        p_cs_ang_keys = list(p_obj_p_cs_ang_t.keys())
        _, ax = plt.subplots()
        for k in p_cs_ang_keys:
            ax.plot(t[1:], p_obj_p_cs_ang_t[k], label=k)
        ax.set_title("Partial gradient of objective w.r.t. control surface angles")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Gradient")
        ax.legend()
        plt.savefig(iter_path.joinpath("p_obj_p_cs_ang_t.png"))
        save_dat(
            iter_path.joinpath("p_obj_p_cs_ang_t.dat"),
            header=["t"] + p_cs_ang_keys,
            columns=[t[1:]] + [p_obj_p_cs_ang_t[k] for k in p_cs_ang_keys],
        )

        p_cs_vel_keys = list(p_obj_p_cs_vel_t.keys())
        _, ax = plt.subplots()
        for k in p_cs_vel_keys:
            ax.plot(t[1:], p_obj_p_cs_vel_t[k], label=k)
        ax.set_title("Partial gradient of objective w.r.t. control surface velocities")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Gradient")
        ax.legend()
        plt.savefig(iter_path.joinpath("p_obj_p_cs_vel_t.png"))
        save_dat(
            iter_path.joinpath("p_obj_p_cs_vel_t.dat"),
            header=["t"] + p_cs_vel_keys,
            columns=[t[1:]] + [p_obj_p_cs_vel_t[k] for k in p_cs_vel_keys],
        )

        d_cs_vel_keys = list(d_obj_d_cs_vel.keys())
        _, ax = plt.subplots()
        for k in d_cs_vel_keys:
            ax.plot(t[1:n_ctrl_tstep], d_obj_d_cs_vel[k], label=k)
        ax.set_title("Total gradient of objective w.r.t. control surface velocities")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Gradient")
        ax.legend()
        plt.savefig(iter_path.joinpath("d_obj_d_cs_vel_t.png"))
        save_dat(
            iter_path.joinpath("d_obj_d_cs_vel_t.dat"),
            header=["t"] + d_cs_vel_keys,
            columns=[t[1:n_ctrl_tstep]] + [d_obj_d_cs_vel[k] for k in d_cs_vel_keys],
        )

        plt.close("all")

        print(f"iteration {i_iter}: objective = {float(total_obj):.6e}")
        i_iter += 1

        return total_obj, grad

    # constraints and optimisation run: impose hard constraints that limit
    # the control deflections and velocities, then run the SLSQP optimisation
    cum_sum_dt = dt * jnp.tril(jnp.ones((n_ctrl_tstep, n_ctrl_tstep)))
    ang_block = (cum_sum_dt @ jnp.eye(n_ctrl_tstep, n_ctrl_tstep - 1, k=-1)) * (
        MAX_CS_VELOCITY / MAX_CS_DEFLECTION
    )
    zero_block = jnp.zeros_like(ang_block)
    angle_constraint_jac = jnp.block([[ang_block, zero_block], [zero_block, ang_block]])
    angle_constraint = LinearConstraint(A=angle_constraint_jac, lb=-1.0, ub=1.0)

    result = minimize(  # type: ignore
        fun=get_cs_grad,
        x0=jnp.concatenate([v for v in cs_vel_init.values()]),
        jac=True,
        bounds=[(-1.0, 1.0)] * (2 * (n_ctrl_tstep - 1)),  # velocity limits
        constraints=(angle_constraint,),
        method="SLSQP",
        options={"maxiter": N_ITER, "ftol": 1e-15},
    )

    print()
    print(f"Optimisation finished: {result.message}")
    print(f"Final objective = {float(result.fun):.6e}")


if __name__ == "__main__":
    main()
