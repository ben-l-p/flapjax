import copy
from typing import cast

import jax
from jax import Array
from jax import numpy as jnp

from flapjax.structure import (
    BeamStructure,
    StructureCase,
    StructureDesignVariables,
    StructureFullStates,
)
from flapjax.structure.gradients.data_structures import StructureGradsToCompute


class TestClampedBeamDynamicAdjoint:
    r"""
    Regression test for ``dynamic_adjoint`` with a non-empty ``prescribed_dofs``.
    """

    @staticmethod
    def _run(matrix_free: bool) -> tuple[float, float, float, float]:
        n_nodes = 5
        conn = jnp.stack((jnp.arange(n_nodes - 1), jnp.arange(1, n_nodes)), axis=1)
        y_vect = jnp.array((0.0, 1.0, 0.0))

        beam = BeamStructure(
            num_nodes=n_nodes,
            connectivity=conn,
            y_vector=y_vect[None, :],
            spectral_radius=1.0,
        )

        coords = jnp.zeros((n_nodes, 3)).at[:, 0].set(jnp.linspace(-0.5, 0.5, n_nodes))
        m_cs0 = jnp.diag(jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))[None, ...]
        k_cs0 = jnp.diag(jnp.array([1e3, 1e3, 1e3, 1e3, 1e3, 1e3]))[None, ...]
        beam.set_design_variables(coords=coords, k_cs=k_cs0, m_cs=m_cs0, m_lumped=None)

        n_tstep = 5
        n_check = n_tstep - 1
        dt = 1e-3
        f_mag = 2.0
        prescribed_dofs = tuple(range(6))  # clamp node 0

        f_ext = jnp.zeros((n_tstep, n_nodes, 6)).at[:, -1, 2].set(f_mag)

        def run_primal(case) -> StructureCase:
            init = case.reference_configuration(
                prescribed_dofs=prescribed_dofs
            ).to_dynamic()
            return case.dynamic_solve(
                init_state=init, n_tstep=n_tstep, dt=dt, f_ext_dead=f_ext
            )

        def tip_vz(sol: StructureCase) -> float:
            assert sol.v is not None
            return float(sol.v[n_check, -1, 2])

        def objective(
            ss: StructureFullStates,
            _: StructureDesignVariables,
            i_ts: int | Array | None,
        ) -> Array:
            assert ss.v is not None
            return jax.lax.select(i_ts == n_check, ss.v[-1, 2], jnp.zeros(()))

        solution = run_primal(beam)

        grads, _ = beam.dynamic_adjoint(
            structure=solution,
            objective=objective,
            matrix_free=matrix_free,
            p_q0_p_x=None,
            approx_grads=False,
            grads_to_compute=StructureGradsToCompute(k_cs=True, m_cs=True),
        )
        adj_grad_k = float(cast(Array, grads.k_cs)[0, 0, 4, 4])
        adj_grad_m = float(cast(Array, grads.m_cs)[0, 0, 2, 2])

        def perturbed_obj(dk: float = 0.0, dm: float = 0.0) -> float:
            case = copy.deepcopy(beam)
            case.set_design_variables(
                coords=coords,
                k_cs=k_cs0.at[0, 4, 4].add(dk),
                m_cs=m_cs0.at[0, 2, 2].add(dm),
            )
            return tip_vz(run_primal(case))

        eps_k = 1e-4 * float(k_cs0[0, 4, 4])
        eps_m = 1e-3 * float(m_cs0[0, 2, 2])

        fd_grad_k = (perturbed_obj(dk=eps_k) - perturbed_obj(dk=-eps_k)) / (2 * eps_k)
        fd_grad_m = (perturbed_obj(dm=eps_m) - perturbed_obj(dm=-eps_m)) / (2 * eps_m)

        return adj_grad_k, adj_grad_m, fd_grad_k, fd_grad_m

    def test_adjoint(self):
        adj_k, adj_m, fd_k, fd_m = self._run(matrix_free=False)

        assert abs(fd_k - adj_k) / abs(adj_k) < 1e-4, (
            "Bending-stiffness gradient on a clamped beam does not match finite differences"
        )
        assert abs(fd_m - adj_m) / abs(adj_m) < 1e-2, (
            "Mass gradient on a clamped beam does not match finite differences"
        )

    def test_adjoint_matrix_free(self):
        adj_k, adj_m, fd_k, fd_m = self._run(matrix_free=True)

        assert abs(fd_k - adj_k) / abs(adj_k) < 1e-4, (
            "Bending-stiffness gradient on a clamped beam does not match finite differences"
        )
        assert abs(fd_m - adj_m) / abs(adj_m) < 1e-2, (
            "Mass gradient on a clamped beam does not match finite differences"
        )
