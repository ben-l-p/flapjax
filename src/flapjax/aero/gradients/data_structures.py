from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, Self

from jax import Array
from jax import numpy as jnp

if TYPE_CHECKING:
    from flapjax.aero.data_structures import AeroCase
from flapjax.algebra.array_utils import ArrayList, ArrayListShape, vect_to_arrs
from flapjax.plotting.aerogrid import plot_grid_to_vtk
from flapjax.utils.data_structures import DesignVariables
from flapjax.utils.print_utils import warn
from flapjax.utils.utils import make_pytree


@dataclass(frozen=True)
class AeroGradsToCompute:
    r"""
    Class which contains flags to determine which gradients are to be computed for the aerodynamic problem during the
    adjoint solve. Defaults to computing only the aerodynamic grid gradients.
    :param x0_aero: Aerodynamic grid coordinates.
    :param flowfield: Flow field parameters.
    :param cs_ang_t: Control surface deflection angle time history.
    :param cs_vel_t: Control surface velocity time history.
    """

    x0_aero: bool = True
    flowfield: bool = False
    cs_ang_t: bool = False
    cs_vel_t: bool = False


# Pycharm typing checking bug
# noinspection PyUnresolvedReferences
type ApproxJacobianEntry = (
    Literal["zero", "constant", "identity"]
    | tuple[Literal["dense_linear", "lazy_linear"], str | Sequence[str]]
    | None
)


@dataclass
class GammaBApprox:
    varphi_n: ApproxJacobianEntry = None
    v_n: ApproxJacobianEntry = None
    gamma_b_n: ApproxJacobianEntry = None
    gamma_w_n: ApproxJacobianEntry = None
    zeta_w_n: ApproxJacobianEntry = None
    dv: ApproxJacobianEntry = None


@dataclass
class GammaWApprox:
    varphi_nm1: ApproxJacobianEntry = None
    varphi_n: ApproxJacobianEntry = None
    gamma_b_nm1: ApproxJacobianEntry = None
    gamma_w_nm1: ApproxJacobianEntry = None
    gamma_w_n: ApproxJacobianEntry = None
    zeta_w_nm1: ApproxJacobianEntry = None
    zeta_w_n: ApproxJacobianEntry = None
    dv: ApproxJacobianEntry = None


@dataclass
class ZetaWApprox:
    varphi_nm1: ApproxJacobianEntry = None
    varphi_n: ApproxJacobianEntry = None
    gamma_b_nm1: ApproxJacobianEntry = None
    gamma_w_nm1: ApproxJacobianEntry = None
    gamma_w_n: ApproxJacobianEntry = None
    zeta_w_nm1: ApproxJacobianEntry = None
    zeta_w_n: ApproxJacobianEntry = None
    dv: ApproxJacobianEntry = None


@dataclass
class GammaBDotApprox:
    gamma_b_nm1: ApproxJacobianEntry = None
    gamma_b_n: ApproxJacobianEntry = None
    gamma_b_dot_nm1: ApproxJacobianEntry = None
    gamma_b_dot_n: ApproxJacobianEntry = None
    dv: ApproxJacobianEntry = None


@dataclass
class FAeroApprox:
    varphi_n: ApproxJacobianEntry = None
    v_n: ApproxJacobianEntry = None
    gamma_b_n: ApproxJacobianEntry = None
    gamma_w_n: ApproxJacobianEntry = None
    gamma_b_dot_n: ApproxJacobianEntry = None
    zeta_w_n: ApproxJacobianEntry = None
    dv: ApproxJacobianEntry = None
    f_aero_beam_n: ApproxJacobianEntry = None


@dataclass
class AeroJacobianApproximations:
    gamma_b: GammaBApprox = field(default_factory=GammaBApprox)
    gamma_w: GammaWApprox = field(default_factory=GammaWApprox)
    zeta_w: ZetaWApprox = field(default_factory=ZetaWApprox)
    gamma_b_dot: GammaBDotApprox = field(default_factory=GammaBDotApprox)
    f_aero: FAeroApprox = field(default_factory=FAeroApprox)


@make_pytree
class AeroFullStates:
    r"""
    Aerodynamic states used for the adjoint solve.
    """

    _static: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        gamma_b: ArrayList,
        gamma_w: ArrayList,
        gamma_b_dot: ArrayList,
        zeta_w: ArrayList,
    ) -> None:
        r"""
        :param gamma_b: Bound panel circulation strengths. ``(n_surf, )(m, n)``
        :param gamma_w: Wake panel circulation strengths. ``(n_surf, )(m_star, n)``
        :param gamma_b_dot: Bound grid circulation time derivatives. ``(n_surf, )(m, n)``
        :param zeta_w: Wake grid coordinates. ``(n_surf, )(m_star + 1, n + 1, 3)``
        """
        self.gamma_b: ArrayList = gamma_b
        self.gamma_w: ArrayList = gamma_w
        self.gamma_b_dot: ArrayList = gamma_b_dot
        self.zeta_w: ArrayList = zeta_w

    def shapes(self) -> OrderedDict[str, tuple[int, ...] | ArrayListShape | None]:
        r"""
        Obtain the shapes of all arrays within the data structure.
        :return: Dictionary of {name: shape} pairs of all arrays or ArrayLists within the data structure.
        """
        return OrderedDict(
            gamma_b=self.gamma_b.shape,
            gamma_w=self.gamma_w.shape,
            gamma_b_dot=self.gamma_b_dot.shape,
            zeta_w=self.zeta_w.shape,
        )

    @staticmethod
    def from_vector(
        vect: Array,
        shapes: OrderedDict[str, tuple[int, ...] | ArrayListShape | None],
    ) -> AeroFullStates:
        r"""
        Construct an AeroFullStates object from a vector of data and a corresponding dictionary of shapes, being the inverse
        of ``self.ravel()``.
        :param vect: Aerodynamic state vector.
        :param shapes: Dictionary of {name: shape} pairs of all arrays or ArrayLists within the data structure.
        :return: AeroFullStates object.
        """
        return AeroFullStates(**vect_to_arrs(vect, shapes))

    def ravel(self) -> Array:
        r"""
        Ravel the data structure to a vector, being the inverse of ``cls.from_vector()``.
        :return: Data vector containing all states.
        """

        return jnp.concatenate(
            [
                self.gamma_b.ravel(),
                self.gamma_w.ravel(),
                self.gamma_b_dot.ravel(),
                self.zeta_w.ravel(),
            ]
        )

    @property
    def n_states(self) -> int:
        r"""
        Obtain the total number of states contained within the data structure, being the size of the vector obtained
        from ``self.ravel()``.
        :return: Size of vector.
        """

        return (
            self.gamma_b.size
            + self.gamma_w.size
            + self.gamma_b_dot.size
            + self.zeta_w.size
        )


@make_pytree
class AeroDesignVariables(DesignVariables):
    r"""
    Class to hold all differentiable aerodynamic design variables.
    """

    _static: ClassVar[tuple[str, ...]] = ("f_size", "f_shape", "n_x", "shapes")

    def __init__(
        self,
        zeta_b0: ArrayList | None,
        flowfield: dict[str, Array] | None,
        cs_ang_t: dict[str, Array] | None,
        cs_vel_t: dict[str, Array] | None,
        f_shape: tuple[int, ...],
    ):
        r"""
        :param zeta_b0: Bound aerodynamic grid local reference coordinates. ``(n_surf, )(m+1, n+1, 3)``
        :param flowfield: Dictionary of flowfield variables.
        :param cs_ang_t: Control surface angle time histories, {name: ``(n_tstep, )``}.
        :param cs_vel_t: Control velocity time histories, {name: ``(n_tstep, )``}.
        :param f_shape: Shape of objective. As this class can hold both the primal design variables and the gradient of
        the objective with respect to design variables, the latter case results in arrays which have a shape which
        depends on the objective shape. In this case, all data has (*f_shape, *dv.shape) dimensionality.
        """
        super().__init__()
        self.zeta_b0: ArrayList | None = zeta_b0
        self.flowfield: dict[str, Array] | None = flowfield
        self.cs_ang_t: dict[str, Array] | None = cs_ang_t
        self.cs_vel_t: dict[str, Array] | None = cs_vel_t

        self.f_shape: tuple[int, ...] = f_shape
        self.f_size: int = prod(f_shape)

        self.shapes: dict[
            str,
            tuple[int, ...]
            | ArrayListShape
            | dict[str, tuple[int, ...] | ArrayListShape]
            | None,
        ] = self.get_shapes()
        self.mapping, self.n_x = self.make_index_mapping()

    def __iadd__(self, other: AeroDesignVariables) -> Self:
        r"""
        In-place addition, used for accumulating the gradient in the adjoint problem.
        :param other: Second instance of AeroDesignVariables with equal ``f_shape``.
        :return: Updated ``self``.
        """
        if self.zeta_b0 is not None:
            assert other.zeta_b0 is not None
            self.zeta_b0 = ArrayList(
                [self.zeta_b0[i] + other.zeta_b0[i] for i in range(len(self.zeta_b0))]
            )
        if self.flowfield is not None:
            assert other.flowfield is not None
            for k in self.flowfield:
                self.flowfield[k] += other.flowfield[k]
        if self.cs_ang_t is not None:
            assert other.cs_ang_t is not None
            for k in self.cs_ang_t:
                self.cs_ang_t[k] += other.cs_ang_t[k]
        if self.cs_vel_t is not None:
            assert other.cs_vel_t is not None
            for k in self.cs_vel_t:
                self.cs_vel_t[k] += other.cs_vel_t[k]
        return self

    def get_cs_n(
        self,
        i_ts: int | Array,
        dv_full: AeroDesignVariables,
    ) -> tuple[dict[str, Array], dict[str, Array]]:
        r"""
        Obtain the angles and velocities for all control surfaces at a given timestep.
        :param i_ts: Timestep index.
        :param dv_full: Full design variables. This is used to substitute a non-differentiable value when the control
        inputs are chosen to be omitted from the design variables.
        :return: Dictionary of {name: value} pairs for control angles and velocities.
        """

        # get control surface deflections from design variables
        assert dv_full.cs_ang_t is not None and dv_full.cs_vel_t is not None
        cs_ang_n = {
            k: v[i_ts, ...]
            for k, v in (
                self.cs_ang_t if self.cs_ang_t is not None else dv_full.cs_ang_t
            ).items()
        }

        cs_vel_n = {
            k: v[i_ts, ...]
            for k, v in (
                self.cs_vel_t if self.cs_vel_t is not None else dv_full.cs_vel_t
            ).items()
        }

        return cs_ang_n, cs_vel_n

    def premultiply_adj(self, adj: Array) -> AeroDesignVariables:
        r"""
        Premultiply all design gradients by the adjoint vector.
        :param adj: Adjoint vector.
        :return: Adjoint-Jacobian product.
        """
        return AeroDesignVariables(
            zeta_b0=ArrayList(
                [
                    jnp.einsum("ij,j...->i...", adj, self.zeta_b0[i])
                    for i in range(len(self.zeta_b0))
                ]
            )
            if self.zeta_b0 is not None
            else None,
            flowfield={
                k: jnp.einsum("ij,j...->i...", adj, v)
                for k, v in self.flowfield.items()
            }
            if self.flowfield is not None
            else None,
            cs_ang_t={
                k: jnp.einsum("ij,j...->i...", adj, v) for k, v in self.cs_ang_t.items()
            }
            if self.cs_ang_t is not None
            else None,
            cs_vel_t={
                k: jnp.einsum("ij,j...->i...", adj, v) for k, v in self.cs_vel_t.items()
            }
            if self.cs_vel_t is not None
            else None,
            f_shape=(adj.shape[1],),
        )

    def to_dict(self) -> dict[str, Array | ArrayList | dict[str, Array] | None]:
        r"""
        Extract the design variables as a dictionary.
        :return: Dictionary of design variable name and value pairs.
        """
        return {
            "zeta_b0": self.zeta_b0,
            "flowfield": self.flowfield,
            "cs_ang_t": self.cs_ang_t,
            "cs_vel_t": self.cs_vel_t,
        }

    def plot(
        self,
        case: AeroCase,
        rmat_nodal: ArrayList | None,
        directory: os.PathLike | str,
    ) -> Sequence[Path]:
        r"""
        Plot the aerodynamic grid gradient for cases with a scalar objective.
        :param case: Dynamic aerodynamic case object. This should only contain 1 time step, as the gradients for the
        grid are constant across all time steps.
        :param rmat_nodal: Rotation matrices from beam. As the aerodynamic grid gradients are given in the local frame
        (as this is where ``zeta_b0`` exists), this transformation is used to transform the gradients into the global frame
        for visualisation.
        :param directory: Directory in which to save the plots.
        :return: Sequence of paths of data created.
        """
        if self.f_size != 1:
            raise ValueError("Can only plot gradients for scalar objective functions.")

        if case.n_tstep != 1:
            raise ValueError("Can only plot gradients for singe timestep cases.")

        if self.zeta_b0 is None:
            warn(
                "Aerodynamic grid gradient not computed. Skipping grid gradient plotting."
            )
            return []

        directory_path = Path(directory)
        directory_path.mkdir(parents=True, exist_ok=True)
        paths = []

        for i_surf in range(case.n_surf):
            bound_filename = Path(directory).joinpath(
                case.surf_b_names[i_surf] + "_gradient"
            )

            if rmat_nodal is not None:
                d_x0_aero: Array = jnp.einsum(
                    "ijk,...lik->lij", rmat_nodal[i_surf], self.zeta_b0[i_surf]
                )
            else:
                d_x0_aero = self.zeta_b0[i_surf]

            paths.append(
                plot_grid_to_vtk(
                    case.zeta_b[i_surf],
                    bound_filename,
                    None,
                    node_vector_data={
                        "zeta_b0": d_x0_aero,
                    },
                    cell_scalar_data={},
                )
            )

        return paths
