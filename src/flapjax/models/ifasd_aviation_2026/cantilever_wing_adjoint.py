import jax
from jax import Array
from jax import numpy as jnp

from flapjax.aero.data_structures import GridDiscretisation
from flapjax.aero.flowfields import OneMinusCosineFlowField
from flapjax.aero.utils import make_rectangular_grid
from flapjax.aero.uvlm import UVLM
from flapjax.coupled import CoupledAeroelastic
from flapjax.coupled.data_structures import (
    AeroelasticDesignVariables,
    AeroelasticFullStates,
)
from flapjax.structure import BeamStructure
from flapjax.utils.data_structures import ConvergenceSettings

if __name__ == "__main__":
    r"""
    Obtain the time history of the cantilever wing case, and the gradient of the wing root bending moment with respect
    to some structural and aerodynamic properties. This does not verify with finite differences.
    """
    # problem discretisation
    m = 10  # number of chordwise panels
    n = 20  # number of spanwise panels
    m_star = 25  # wake length
    n_tstep = 1000  # time step count

    c_ref = 1.0  # aerodynamic chord
    b_ref = 6.0  # semi-span
    u_inf = jnp.array((20.0, 0.0, 2.0))  # freestream velocity
    u_inf_mag = jnp.linalg.norm(u_inf)
    k_cs = jnp.diag(
        jnp.array((1e7, 1e7, 1e7, 1e5, 1e7, 2e4))
    )  # cross-sectional stiffness
    m_cs = jnp.diag(
        jnp.array((10.0, 10.0, 10.0, 10.0, 10.0, 10.0))
    )  # cross-sectional mass
    dt = c_ref / (m * u_inf_mag)  # time step length

    # variable wake discretisation spacing
    delta = dt * u_inf_mag
    delta_w = delta * jnp.logspace(0.0, 0.9, m_star)

    # beam non-design variables
    y_vector = jnp.array((0.0, 0.0, 1.0))  # local beam y-axis direction

    # connectivity matrix
    conn = jnp.zeros((n, 2), dtype=int)
    conn = conn.at[:, 0].set(jnp.arange(n))
    conn = conn.at[:, 1].set(jnp.arange(1, n + 1))
    beam = BeamStructure(
        num_nodes=n + 1,
        connectivity=conn,
        y_vector=y_vector,
        spectral_radius=1.0,  # no numerical dissipation
    )

    # aero non-design variables
    gd = GridDiscretisation(m=m, n=n, m_star=m_star)
    uvlm = UVLM(
        grid_shapes=[gd],
        dof_mapping=jnp.arange(n + 1),
        mirror_point=jnp.zeros(3),
        mirror_normal=jnp.array((0.0, 1.0, 0.0)),  # mirror plane in y
        gamma_dot_relaxation=0.7,
    )

    wing = CoupledAeroelastic(beam, uvlm)

    beam_coords = jnp.zeros((n + 1, 3)).at[:, 1].set(jnp.linspace(0, b_ref, n + 1))
    grid = make_rectangular_grid(m, n, c_ref, ea=0.2)  # set elastic axis at 0.2 chord

    flowfield = OneMinusCosineFlowField(
        u_inf=u_inf,
        rho=1.225,  # flowfield density
        relative_motion=True,
        gust_length=10.0,
        gust_amplitude=4.0,
        gust_x0=jnp.array((-25.0, 0.0, 0.0)),
    )

    wing.set_design_variables(
        coords=beam_coords,
        k_cs=k_cs,
        m_cs=m_cs,
        m_lumped=None,
        flowfield=flowfield,
        delta_w=delta_w,
        x0_aero=grid,
        dt=dt,
    )

    # set tolerance to zero, rather than none, to prevent error messages
    # this is very strict but good for testing
    wing.structure.struct_convergence_settings = ConvergenceSettings(
        max_n_iter=100,
        rel_disp_tol=0.0,
        abs_disp_tol=0.0,
        rel_force_tol=0.0,
        abs_force_tol=0.0,
    )
    wing.fsi_convergence_settings = ConvergenceSettings(
        max_n_iter=40,
        rel_disp_tol=0.0,
        abs_disp_tol=0.0,
        rel_force_tol=0.0,
        abs_force_tol=0.0,
    )

    static_sol = wing.static_solve(
        prescribed_dofs=jnp.arange(6)
    )  # remove the first 6 degrees of freedom from solve and solve static case

    # solve dynamic primal system
    dynamic_sol = wing.dynamic_solve(
        init_case=static_sol,
        prescribed_dofs=jnp.arange(6),
        n_tstep=n_tstep,
    )

    # plot the dynamic solution
    dynamic_sol.plot(directory="./cantilever_outputs/dynamic_solution")

    # obtain the index with the largest WRBM
    i_max: int = int(jnp.argmax(dynamic_sol.structure.f_elem[:, 0, 5]))

    # objective which refers to a single timestep
    def objective(
        states: AeroelasticFullStates,
        _: AeroelasticDesignVariables,
        i_ts: int | Array | None,
    ) -> Array:
        # zero everywhere except when i_ts==i_max
        return jax.lax.select(
            i_ts == i_max,
            states.structure.f_elem[0, 5],
            jnp.zeros(()),
        )

    # compute static adjoint
    static_grad, static_adj = wing.static_adjoint(
        case=static_sol, objective=objective, ad_mode="forward"
    )

    # compute dynamic adjoint with the static adjoint as input to account for initial degree of freedom gradients
    # all design variable gradients can be extracted from dynamic_grad
    dynamic_grad, objective_val, dynamic_adj = wing.dynamic_adjoint(
        case=dynamic_sol,
        objective=objective,
        p_varphi_p_x=-static_adj,
    )

    # plot the gradients for the beam and aerodynamic grids
    dynamic_grad.plot(
        case=dynamic_sol,
        directory="./cantilever_outputs/dynamic_gradient",
        i_ts=0,
    )
