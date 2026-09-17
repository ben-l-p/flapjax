from jax import numpy as jnp

from flapjax.models.simple_hale.simple_hale import generate_simple_hale


class TestHingedWingtipEquilibrium:
    @staticmethod
    def test_trim_remains_at_equilibrium():
        r"""
        A HALE aircraft with free-hinging wingtips, trimmed and released to free flight, should remain at that trim when
        time-stepped forward.
        """

        hale = generate_simple_hale(
            flare_angle=jnp.deg2rad(20.0),
        )

        static_sol, _ = hale.trim(
            prescribed_dofs=tuple(range(6)),
            zero_force_dofs=tuple(range(6)),
            trim_cs="elevator",
            thrust_nodes="thrust",
            trim_orientation="y",
            horseshoe=True,
            trim_relaxation=0.4,
            broyden_fd_step=1e-2,
            trim_hinges=["left_hinge", "right_hinge"],
        )

        dynamic_init = hale.initialise_dynamic(
            static_case=static_sol, prescribed_dofs=()
        )

        dynamic_sol = hale.dynamic_solve(
            init_case=dynamic_init, prescribed_dofs=(), n_tstep=10
        )

        # wingtip hinge angles should stay at their trimmed value
        for key in ("left_hinge", "right_hinge"):
            angle = dynamic_sol.structure.constraint_data[key]["angle"]
            drift = jnp.abs(angle - angle[0]).max()
            assert drift < 5e-4, (
                f"{key} angle drifted from trim by {float(drift):.3e} rad"
            )

        # structural strains should stay constant
        eps_drift = jnp.abs(
            dynamic_sol.structure.eps - dynamic_sol.structure.eps[[0], ...]
        ).max()
        assert eps_drift < 2e-3, (
            f"Structural strain drifted from trim by {float(eps_drift):.3e}"
        )

        # root velocity should stay constant
        v_root = dynamic_sol.structure.v[:, 0, :]
        v_root_drift = jnp.abs(v_root - v_root[0]).max()
        assert v_root_drift < 5e-2, (
            f"Root velocity drifted from trim by {float(v_root_drift):.3e}"
        )
