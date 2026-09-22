from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from math import prod
from typing import ClassVar

import numpy as np
from jax import Array
from jax import numpy as jnp

from flapjax.algebra.array_utils import ArrayList, ArrayListShape, check_arr_shape
from flapjax.utils.print_utils import (
    jax_print,
    print_table_line,
    print_table_title,
    warn,
)
from flapjax.utils.utils import make_pytree


@dataclass
class ConvergenceSettings:
    max_n_iter: int | None
    rel_disp_tol: float | None
    abs_disp_tol: float | None
    rel_force_tol: float | None
    abs_force_tol: float | None


@make_pytree
class ConvergenceStatus:
    _static: ClassVar[tuple[str, ...]] = ("convergence_settings",)

    def __init__(self, convergence_settings: ConvergenceSettings):
        r"""
        Object to track convergence status of an iterative solver based on absolute and relative tolerances. Absolute
        convergence is measured for the delta vector, while relative convergence is measured as the ratio of the maximum
        element in the delta vector to the total vector.
        :param convergence_settings: Convergence settings object containing tolerances and maximum iteration count for
        convergence failure.
        """

        # make sure settings allow for loop to be broken
        if (
            convergence_settings.rel_disp_tol is None
            and convergence_settings.abs_disp_tol is None
            and convergence_settings.rel_force_tol is None
            and convergence_settings.abs_force_tol is None
        ):
            if convergence_settings.max_n_iter is None:
                raise ValueError(
                    "No convergence criteria provided, at least one tolerance or maximum iteration count "
                    "must be specified."
                )
            warn(
                "No convergence tolerances provided, will iterate until maximum iteration counter."
            )

        # base parameters
        self.i_iter: Array = jnp.zeros((), dtype=int)
        self.convergence_settings: ConvergenceSettings = convergence_settings

        # store residual values
        self.rel_disp_val: Array = jnp.zeros(())
        self.abs_disp_val: Array = jnp.zeros(())
        self.rel_force_val: Array = jnp.zeros(())
        self.abs_force_val: Array = jnp.zeros(())

        # convergence status
        self.converged: Array = jnp.zeros((), dtype=bool)
        self.converged_abs_disp: Array = jnp.zeros((), dtype=bool)
        self.converged_rel_disp: Array = jnp.zeros((), dtype=bool)
        self.converged_rel_force: Array = jnp.zeros((), dtype=bool)
        self.converged_abs_force: Array = jnp.zeros((), dtype=bool)

        # flags for other convergence failure modes
        self.final_iter: Array = jnp.zeros((), dtype=bool)
        self.has_nan: Array = jnp.zeros((), dtype=bool)

    def update(
        self,
        delta_disp: Array | None,
        total_disp: Array | None,
        delta_force: Array | None,
        total_force: Array | None,
    ) -> None:
        r"""
        :param delta_disp: Difference in displacement vector between current and previous iteration.
        :param total_disp: Total displacement for time step, used for relative convergence calculation, typically the
        current solution vector.
        :param delta_force: Residual force vector, used for force convergence calculation.
        :param total_force: Total force vector for time step, used for relative convergence calculation.
        """
        # update iteration counter
        self.i_iter += 1

        # check absolute displacement convergence
        if delta_disp is not None:
            self.abs_disp_val = jnp.linalg.norm(delta_disp)

            # NaNs are checked for with displacement magnitude
            self.has_nan = jnp.isnan(delta_disp).any()

        if self.convergence_settings.abs_disp_tol is not None:
            self.converged_abs_disp = (
                self.abs_disp_val < self.convergence_settings.abs_disp_tol
            )

        # check relative displacement convergence:
        if self.convergence_settings.rel_disp_tol is not None:
            if total_disp is None:
                raise ValueError("total_disp cannot be None")
            max_total_elem = jnp.linalg.norm(total_disp)
            self.rel_disp_val = self.abs_disp_val / max_total_elem
            self.converged_rel_disp = (
                jnp.nan_to_num(self.rel_disp_val, True, jnp.inf)
                < self.convergence_settings.rel_disp_tol
            )

        # check absolute force convergence
        if self.convergence_settings.abs_force_tol is not None:
            if delta_force is None:
                raise ValueError("delta_force cannot be None")
            self.abs_force_val = jnp.linalg.norm(delta_force)
            self.converged_abs_force = (
                self.abs_force_val < self.convergence_settings.abs_force_tol
            )

        # check relative force convergence:
        if self.convergence_settings.rel_force_tol is not None:
            if total_force is None:
                raise ValueError("total_force cannot be None")
            max_total_elem = jnp.abs(total_force).max()
            self.rel_force_val = self.abs_force_val / max_total_elem
            self.converged_rel_force = (
                jnp.nan_to_num(self.rel_force_val, True, jnp.inf)
                < self.convergence_settings.rel_force_tol
            )

        # find convergence status of numerics (excluding failure modes such as max iterations or nans)
        self.converged = (
            self.converged_rel_disp
            | self.converged_abs_disp
            | self.converged_rel_force
            | self.converged_abs_force
        ) & (self.i_iter > 0)

        # check for failure modes
        if self.convergence_settings.max_n_iter is not None:
            self.final_iter = self.i_iter >= self.convergence_settings.max_n_iter

    def get_status(self) -> Array:
        """Get overall convergence status."""
        return self.converged | self.has_nan | self.final_iter

    def reset_status(self) -> None:
        r"""
        Reset convergence status for next load step, setting all convergence flags to False.
        """
        false_ = jnp.zeros((), dtype=bool)
        zero_ = jnp.zeros(())
        self.converged = false_
        self.converged_abs_disp = false_
        self.converged_rel_disp = false_
        self.converged_rel_force = false_
        self.converged_abs_force = false_
        self.rel_disp_val = zero_
        self.abs_disp_val = zero_
        self.rel_force_val = zero_
        self.abs_force_val = zero_
        self.has_nan = false_
        self.final_iter = false_
        self.i_iter = jnp.zeros((), dtype=int)

    def _print_convergence_message(
        self,
        label: str,
        i_ts: int | None,
        t: Array | None,
        i_load_step: int | None,
    ) -> None:
        parts: list[str] = ["|"]
        kw: dict[str, Array | int] = {
            "i_iter": self.i_iter,
            "conv": self.converged,
            "rel_disp_val": self.rel_disp_val,
            "abs_disp_val": self.abs_disp_val,
            "rel_force_val": self.rel_force_val,
            "abs_force_val": self.abs_force_val,
        }
        if t is not None:
            parts.append("{i_ts:<8} | {t:.03e} |")
            kw["i_ts"] = i_ts  # type: ignore[assignment]
            kw["t"] = t
        if label == "Struct":
            parts.append("Struct: {i_iter:<3} |")
        else:
            parts.append("FSI: {i_iter:<3}    |")
        parts.append("{conv!s:<5} | {rel_disp_val:.03e} | {abs_disp_val:.03e} |")
        parts.append("{rel_force_val:.03e} | {abs_force_val:.03e} |")
        if i_load_step is not None:
            parts.append("{i_load_step:<2}        |")
            kw["i_load_step"] = i_load_step
        else:
            parts.append("          |")
        jax_print(" ".join(parts), verbose_level="normal", **kw)

    def print_struct_message(
        self, i_ts: int | None, t: Array | None, i_load_step: int
    ) -> None:
        """Print convergence message for structure based on status."""
        self._print_convergence_message("Struct", i_ts, t, i_load_step)

    def print_fsi_message(self, i_ts: int | None, t: Array | None) -> None:
        """Print convergence message for FSI based on status."""
        self._print_convergence_message("FSI", i_ts, t, None)

    @classmethod
    def print_header(cls, dynamic: bool) -> None:
        if dynamic:
            print_table_title(title="Dynamic Solve", inner_width=104)
            jax_print(
                "| Timestep |   Time    |    Iter     | Conv  | Rel Disp  | Abs Disp  | Rel Force | Abs Force | Load Step |",
                verbose_level="normal",
            )
        else:
            print_table_title(title="Static Solve", inner_width=81)
            jax_print(
                "|    Iter     | Conv  | Rel Disp  | Abs Disp  | Rel Force | Abs Force | Load Step |",
                verbose_level="normal",
            )
        cls.print_line(dynamic=dynamic)

    @staticmethod
    def print_line(dynamic: bool) -> None:
        print_table_line(inner_width=104 if dynamic else 81)


class DesignVariables:
    def __init__(self):
        # should only be used in child classes
        self.shapes: dict[str, tuple[int, ...] | ArrayListShape | None] = {}
        self.mapping: dict[str, Array | ArrayList | None] = {}

    def to_dict(self) -> dict[str, Array | ArrayList | dict[str, Array] | None]:
        # should only be used in child classes
        return {}

    def get_shapes(
        self,
    ) -> OrderedDict[
        str,
        tuple[int, ...]
        | ArrayListShape
        | OrderedDict[str, tuple[int, ...] | ArrayListShape]
        | None,
    ]:
        def _elem_shape(
            elem: Array | ArrayList | dict[str, Array | ArrayList] | None,
        ) -> (
            tuple[int, ...]
            | ArrayListShape
            | dict[str, tuple[int, ...] | ArrayListShape]
            | None
        ):
            if elem is None:
                return None
            elif isinstance(elem, dict):
                sub_dict = OrderedDict()
                for k_, v in sorted(elem.items()):
                    sub_dict[k_] = v.shape
                return sub_dict
            else:
                return elem.shape

        out_dict = OrderedDict()
        for k, var in self.to_dict().items():
            out_dict[k] = _elem_shape(var)

        return out_dict

    def make_index_mapping(
        self,
    ) -> tuple[
        OrderedDict[str, Array | ArrayList | OrderedDict[str, Array] | None], int
    ]:
        mapping = OrderedDict()
        cnt = 0
        for name, shape in self.shapes.items():
            if shape is not None:
                if isinstance(shape, tuple):
                    var_size = prod(shape)
                    mapping[name] = np.arange(cnt, cnt + var_size).reshape(shape)
                    cnt += var_size
                elif isinstance(shape, ArrayListShape):
                    sub_mappings = []
                    for i_arr in range(shape.n_arrays):
                        var_size = prod(shape.shapes[i_arr])
                        sub_mappings.append(
                            np.arange(cnt, cnt + var_size).reshape(shape.shapes[i_arr])
                        )
                        cnt += var_size
                    mapping[name] = ArrayList(sub_mappings)
                elif isinstance(shape, OrderedDict):
                    sub_mappings = OrderedDict()
                    for k, v in shape.items():
                        var_size = prod(v)
                        sub_mappings[k] = np.arange(cnt, cnt + var_size).reshape(v)
                        cnt += var_size
                    mapping[name] = sub_mappings
                else:
                    raise ValueError("Invalid shape type in DesignVariables.")
            else:
                mapping[name] = None
        return mapping, cnt

    def ravel_jacobian(self, f_size: int, x_size: int) -> Array:
        def _inner_ravel(
            var: Array | ArrayList | dict[str, Array | ArrayList],
        ) -> Array | None:
            if isinstance(var, Array):
                return var.reshape(f_size, -1)
            elif isinstance(var, ArrayList):
                sub_parts: list[Array] = []
                for sub_var in var:
                    sub_r = _inner_ravel(sub_var)
                    if sub_r is not None:
                        sub_parts.append(sub_r)
                return jnp.concatenate(sub_parts, axis=-1)
            elif isinstance(var, dict):
                if not var:
                    return None
                dict_parts: list[Array] = []
                for k in sorted(var.keys()):
                    dict_r = _inner_ravel(var[k])
                    if dict_r is not None:
                        dict_parts.append(dict_r)
                return jnp.concatenate(dict_parts, axis=-1)
            else:
                raise TypeError("Invalid variable type in DesignVariables.")

        parts: list[Array] = []
        for value in self.to_dict().values():
            if value is None:
                continue
            r = _inner_ravel(value)
            if r is not None:
                parts.append(r)
        arr = jnp.concatenate(parts, axis=1)
        check_arr_shape(arr, (f_size, x_size), "Internal Jacobian")
        return arr

    def ravel(self) -> Array:
        def _inner_func(var: Array | ArrayList | dict[str, Array | ArrayList]) -> Array:
            if isinstance(var, Array | ArrayList):
                return var.ravel()
            elif isinstance(var, dict):
                return jnp.concatenate([_inner_func(v) for v in var.values()])
            else:
                raise TypeError("Invalid variable type in DesignVariables.")

        return jnp.concatenate(
            [_inner_func(var) for var in self.to_dict().values() if var is not None]
        )

    def reshape(self, *args: int) -> Array:
        return self.ravel().reshape(*args)

    def from_adjoint(
        self, f_shape: tuple[int, ...], df_dx: Array
    ) -> OrderedDict[str, Array | ArrayList]:
        out_dict = OrderedDict()
        for name in self.shapes:
            if (this_mapping := self.mapping[name]) is not None:
                if isinstance((this_shapes := self.shapes[name]), tuple):
                    out_dict[name] = df_dx[:, this_mapping].reshape(
                        *f_shape, *this_shapes
                    )
                elif isinstance((this_shapes := self.shapes[name]), ArrayListShape):
                    subarrays = []
                    for i_arr in range(this_shapes.n_arrays):
                        subarrays.append(
                            df_dx[:, this_mapping[i_arr]].reshape(
                                *f_shape, *this_shapes.shapes[i_arr]
                            )
                        )
                    out_dict[name] = ArrayList(subarrays)
                elif isinstance((this_shapes := self.shapes[name]), dict):
                    subarrays = OrderedDict()
                    for k, v in this_shapes.items():
                        subarrays[k] = df_dx[:, this_mapping[k]].reshape(*f_shape, *v)
                    out_dict[name] = subarrays
                else:
                    raise ValueError("Invalid shape type in DesignVariables.")
            else:
                out_dict[name] = None
        return out_dict

    @classmethod
    def concatenate[T: DesignVariables](cls, *dvs: T) -> T:
        dict_dvs = [dvs.to_dict() for dvs in dvs]

        f_shape = 0
        out_dict = {}
        for key in dict_dvs[0]:
            if all(dict_dv[key] is None for dict_dv in dict_dvs):
                out_dict[key] = None
            elif all(isinstance(dict_dv[key], Array) for dict_dv in dict_dvs):
                out_dict[key] = jnp.concatenate(
                    [dict_dv[key] for dict_dv in dict_dvs],  # type: ignore
                    axis=0,
                )
                f_shape = out_dict[key].shape[0]
            elif all(isinstance(dict_dv[key], ArrayList) for dict_dv in dict_dvs):
                out_dict[key] = ArrayList(
                    [
                        jnp.concatenate(
                            [dict_dv[key][i_arr] for dict_dv in dict_dvs],  # type: ignore
                            axis=0,
                        )
                        for i_arr in range(len(dict_dvs[0][key]))  # type: ignore
                    ]
                )
                f_shape = out_dict[key][0].shape[0]
            elif all(isinstance(dict_dv[key], dict) for dict_dv in dict_dvs):
                out_dict[key] = {}
                for subkey in dict_dvs[0][key]:  # type: ignore
                    out_dict[key][subkey] = jnp.concatenate(
                        [dict_dv[key][subkey] for dict_dv in dict_dvs]  # type: ignore
                    )
                    f_shape = out_dict[key][subkey].shape[0]
            else:
                raise ValueError("Invalid shape type in DesignVariables.")
        return type(dvs[0])(**out_dict, f_shape=(f_shape,))  # type: ignore
