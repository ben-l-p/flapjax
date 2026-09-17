from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from jax import Array
from jax import numpy as jnp

from flapjax.aero.gradients.data_structures import (
    AeroGradsToCompute,
    AeroJacobianApproximations,
)
from flapjax.structure.gradients.data_structures import (
    StructureGradsToCompute,
    StructureJacobianApproximations,
)
from flapjax.utils.print_utils import jax_print, print_table_line
from flapjax.utils.utils import make_pytree


@dataclass(frozen=True)
class AeroelasticGradsToCompute:
    structure: StructureGradsToCompute = field(
        default_factory=StructureGradsToCompute
    )
    aero: AeroGradsToCompute = field(default_factory=AeroGradsToCompute)


@dataclass
class AeroelasticJacobianApproximations:
    structure: StructureJacobianApproximations = field(
        default_factory=StructureJacobianApproximations
    )
    aero: AeroJacobianApproximations = field(default_factory=AeroJacobianApproximations)


@make_pytree
class TrimVariables:
    _static: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        cs_ang: dict[str, Array],
        thrust: dict[str, Array],
        trim_angles: dict[str, Array],
        hinge_angle: dict[str, Array] | None = None,
    ):
        self.cs_ang: dict[str, Array] = cs_ang
        self.thrust: dict[str, Array] = thrust
        self.trim_angles: dict[str, Array] = trim_angles
        self.hinge_angle: dict[str, Array] = {} if hinge_angle is None else hinge_angle

    _INNER_WIDTH: ClassVar[int] = 104
    _ITER_W: ClassVar[int] = 5

    def _column_specs(self, f_clamp: Array | None) -> list[tuple[str, str, int, str]]:
        """Return (key, unit, value_width, format_spec) for each numeric column."""
        specs: list[tuple[str, str, int, str]] = []
        specs.extend((k, "deg", 5, ".2f") for k in self.cs_ang)
        specs.extend((k, "N", 9, ".2e") for k in self.thrust)
        specs.extend((k, "deg", 5, ".2f") for k in self.trim_angles)
        specs.extend((k, "deg", 5, ".2f") for k in self.hinge_angle)
        if f_clamp is not None:
            specs.extend((f"f{i}", "N", 9, ".2e") for i in range(f_clamp.shape[0]))
        return specs

    @staticmethod
    def _header_label(key: str, unit: str) -> str:
        return f"{key}, {unit}"

    def print_header(self, f_clamp: Array | None) -> None:
        """Print the column-header row. Call once before the iteration loop."""
        specs = self._column_specs(f_clamp)
        col_widths = [max(len(self._header_label(k, u)), vw) for k, u, vw, _ in specs]

        cells = ["iter".rjust(self._ITER_W)]
        cells += [
            self._header_label(k, u).rjust(cw)
            for (k, u, _, _), cw in zip(specs, col_widths)
        ]
        inner = " | ".join(cells)
        padding = self._INNER_WIDTH - len(inner) - 2
        jax_print("| " + inner + " " * padding + " |", verbose_level="normal")
        print_table_line(inner_width=self._INNER_WIDTH)

    def print_values(self, i_iter: int, f_clamp: Array | None) -> None:
        """Print one row of numeric values, aligned with the header."""
        specs = self._column_specs(f_clamp)
        col_widths = [max(len(self._header_label(k, u)), vw) for k, u, vw, _ in specs]

        def _scalar(x: Array) -> Array:
            return jnp.ravel(x)[0]

        values: list[Array] = []
        values.extend(jnp.rad2deg(_scalar(v)) for v in self.cs_ang.values())
        values.extend(_scalar(v) for v in self.thrust.values())
        values.extend(jnp.rad2deg(_scalar(v)) for v in self.trim_angles.values())
        values.extend(jnp.rad2deg(_scalar(v)) for v in self.hinge_angle.values())
        if f_clamp is not None:
            values.extend(_scalar(f_clamp[i]) for i in range(f_clamp.shape[0]))

        placeholders = [f"c{i}" for i in range(len(specs))]
        cells = [f"{{i_iter:>{self._ITER_W}}}"]
        cells += [
            f"{{{ph}:>{cw}{fmt}}}"
            for ph, (_, _, _, fmt), cw in zip(placeholders, specs, col_widths)
        ]
        inner = " | ".join(cells)
        rendered_len = self._ITER_W + sum(col_widths) + 3 * len(specs)
        padding = self._INNER_WIDTH - rendered_len - 2

        kwargs = {"i_iter": i_iter, **dict(zip(placeholders, values))}
        jax_print("| " + inner + " " * padding + " |", **kwargs, verbose_level="normal")
