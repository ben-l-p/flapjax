import copy
from typing import cast

import jax
from jax import Array
from jax import numpy as jnp

from flapjax.models.double_pendulum.double_pendulum import generate_double_pendulum
from flapjax.structure import (
    StructureCase,
    StructureDesignVariables,
    StructureFullStates,
)
from flapjax.structure.gradients.data_structures import StructureGradsToCompute


class TestMultibodyHingeDynamicAdjoint:
    @staticmethod
    def _run(matrix_free: bool) -> tuple[float, float, float, float]:
        n_tstep = 5
        n_check = 3
        dt = 2e-3
        f_mag = 1.0  # lateral tip force, orthogonal to the hinge axis

        beam, _root_hinge, _mid_hinge = generate_double_pendulum(
            length=1.0,
            n_nodes_per_segment=3,
            gravity=(0.0, 0.0, 0.0),
        )
        coords = beam.x0_reference
        k_cs0 = beam.k_cs
        m_cs0 = beam.m_cs

        def make_f_ext(n_nodes: int) -> Array:
            return jnp.zeros((n_tstep, n_nodes, 6)).at[:, -1, 1].set(f_mag)

        def run_primal(case) -> StructureCase:
            init = case.reference_configuration().to_dynamic()
            return case.dynamic_solve(
                init_state=init,
                n_tstep=n_tstep,
                dt=dt,
                f_ext_dead=make_f_ext(case.n_nodes),
            )

        def tip_vy(sol: StructureCase) -> float:
            assert sol.v is not None
            return float(sol.v[n_check, -1, 1])

        def objective(
            ss: StructureFullStates,
            _: StructureDesignVariables,
            i_ts: int | Array | None,
        ) -> Array:
            assert ss.v is not None
            return jax.lax.select(
                i_ts == n_check,
                ss.v[-1, 1],
                jnp.zeros(()),
            )

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
        adj_grad_m = float(cast(Array, grads.m_cs)[0, 0, 0, 0])

        def perturbed_obj(dk: float = 0.0, dm: float = 0.0) -> float:
            case = copy.deepcopy(beam)
            case.set_design_variables(
                coords=coords,
                k_cs=k_cs0.at[0, 4, 4].add(dk),
                m_cs=m_cs0.at[0, 0, 0].add(dm),
            )
            return tip_vy(run_primal(case))

        eps_k = 1e-4 * float(k_cs0[0, 4, 4])  # relative step on bending stiffness
        eps_m = 3e-2 * float(m_cs0[0, 0, 0])  # relative step on translational mass

        fd_grad_k = (perturbed_obj(dk=eps_k) - perturbed_obj(dk=-eps_k)) / (2 * eps_k)
        fd_grad_m = (perturbed_obj(dm=eps_m) - perturbed_obj(dm=-eps_m)) / (2 * eps_m)

        return adj_grad_k, adj_grad_m, fd_grad_k, fd_grad_m

    def test_adjoint(self):
        adj_k, adj_m, fd_k, fd_m = self._run(matrix_free=False)

        assert abs(fd_k - adj_k) / abs(adj_k) < 1e-4, (
            "Bending-stiffness gradient propagated across the multibody hinge does not "
            "match finite differences"
        )
        assert abs(fd_m - adj_m) / abs(adj_m) < 1e-2, (
            "Mass gradient propagated across the multibody hinge does not match finite "
            "differences"
        )

    def test_adjoint_matrix_free(self):
        adj_k, adj_m, fd_k, fd_m = self._run(matrix_free=True)

        assert abs(fd_k - adj_k) / abs(adj_k) < 1e-4, (
            "Bending-stiffness gradient propagated across the multibody hinge does not "
            "match finite differences"
        )
        assert abs(fd_m - adj_m) / abs(adj_m) < 1e-2, (
            "Mass gradient propagated across the multibody hinge does not match finite "
            "differences"
        )
