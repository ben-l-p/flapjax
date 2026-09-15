import jax
from jax import numpy as jnp

from flapjax.coupled import CoupledAeroelastic
from flapjax.models.cantilever_wing.cantilever_wing import generate_cantilever_wing


class TestParallelCantilever:
    r"""
    Simulate many cantilever wing problems at once to ensure multi-case aeroelastic parallelism works.
    """

    @staticmethod
    def test_parallel_cantilever():
        n_case = 4
        mult = jnp.linspace(1.0, 1.1, n_case)
        n_tstep = 20

        cases = []
        for m in mult:
            case = generate_cantilever_wing(n_nodes=10, m=6, m_star=20)
            case.structure.k_cs *= m
            cases.append(case)

        stacked_case = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *cases)

        def solve(case_: CoupledAeroelastic):
            static_sol = case_.static_solve(
                prescribed_dofs=tuple(range(6)), horseshoe=True
            )

            return case_.dynamic_solve(
                init_case=static_sol,
                prescribed_dofs=tuple(range(6)),
                n_tstep=n_tstep,
            )

        solutions = jax.vmap(solve)(stacked_case)
        jax.block_until_ready(solutions)
