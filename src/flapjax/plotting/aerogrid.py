from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import vtk
from jax import Array

# noinspection PyUnresolvedReferences
from vtk.util import numpy_support as vns

from flapjax.plotting.writer import (
    configure_fast_writer,
    numpy_to_vtk_scalar,
    numpy_to_vtk_vector,
    to_host,
)


def _check_planar(grid_arr: Array) -> None:
    if grid_arr.ndim != 3:
        raise ValueError(
            f"grid_arr must be a planar structured grid of shape (m, n, 3), "
            f"got {grid_arr.ndim}-D array"
        )


def swap_last_axis(arr: np.ndarray) -> np.ndarray:
    """Swap axis 0 with axis 1 (VTK expects y, x order for planar grids)."""
    return np.swapaxes(arr, 0, 1)


def make_grid_points(grid_host: np.ndarray) -> vtk.vtkPoints:
    """Build interleaved points array in VTK ordering from a host array."""
    xyz = np.stack(
        [swap_last_axis(grid_host[..., i]).ravel() for i in range(3)],
        axis=-1,
    )
    xyz = np.ascontiguousarray(xyz, dtype=np.float64)
    points = vtk.vtkPoints()
    points.SetData(vns.numpy_to_vtk(xyz, deep=True))
    return points


def create_structured_grid(grid_arr: Array) -> vtk.vtkStructuredGrid:
    r"""
    Create a VTK structured grid from a JAX array.
    :param grid_arr: Array of grid points, ``(m, n, 3)``.
    :return: VTK structured grid.
    """
    sg = vtk.vtkStructuredGrid()
    sg.SetDimensions(*grid_arr.shape[:-1], 1)

    grid_host = to_host(grid_arr)
    sg.SetPoints(make_grid_points(grid_host))
    return sg


def attach_grid_arrays(
    sg: vtk.vtkStructuredGrid,
    node_scalar_data: dict[str, Array] | None,
    node_vector_data: dict[str, Array] | None,
    cell_scalar_data: dict[str, Array] | None,
    cell_vector_data: dict[str, Array] | None,
) -> None:
    if cell_scalar_data:
        for name, arr in cell_scalar_data.items():
            arr_h = swap_last_axis(to_host(arr)).ravel()
            sg.GetCellData().AddArray(numpy_to_vtk_scalar(arr_h, name))

    if cell_vector_data:
        for name, arr in cell_vector_data.items():
            arr_h = to_host(arr)
            if arr_h.shape[-1] != 3:
                raise ValueError(
                    f"Node vector data '{name}' must have trailing dimension of size 3, got {arr_h.shape[-1]}"
                )
            xyz = np.stack(
                [swap_last_axis(arr_h[..., i]).ravel() for i in range(3)],
                axis=-1,
            )
            sg.GetCellData().AddArray(numpy_to_vtk_vector(xyz, name))

    if node_scalar_data:
        for name, arr in node_scalar_data.items():
            arr_h = swap_last_axis(to_host(arr)).ravel()
            sg.GetPointData().AddArray(numpy_to_vtk_scalar(arr_h, name))

    if node_vector_data:
        for name, arr in node_vector_data.items():
            arr_h = to_host(arr)
            if arr_h.shape[-1] != 3:
                raise ValueError(
                    f"Node vector data '{name}' must have trailing dimension of size 3, got {arr_h.shape[-1]}"
                )
            xyz = np.stack(
                [swap_last_axis(arr_h[..., i]).ravel() for i in range(3)],
                axis=-1,
            )
            sg.GetPointData().AddArray(numpy_to_vtk_vector(xyz, name))


def plot_grid_to_vtk(
    grid_arr: Array,
    filename: str | os.PathLike,
    i_ts: int | None = None,
    node_scalar_data: dict[str, Array] | None = None,
    node_vector_data: dict[str, Array] | None = None,
    cell_scalar_data: dict[str, Array] | None = None,
    cell_vector_data: dict[str, Array] | None = None,
) -> Path:
    r"""
    Plot a single timestep of grid data
    :param grid_arr: Planar structured grid array, ``(m, n, 3)``.
    :param filename: Base filename, including directory. Information on the time step number will be
    appended to this.
    :param i_ts: Timestep to write
    :param node_scalar_data: Dictionary of node scalar data
    :param node_vector_data: Dictionary of node vector data
    :param cell_scalar_data: Dictionary of cell scalar data
    :param cell_vector_data: Dictionary of cell vector data
    """
    filepath = Path(filename)
    _check_planar(grid_arr)
    sg = create_structured_grid(grid_arr)
    attach_grid_arrays(
        sg,
        node_scalar_data,
        node_vector_data,
        cell_scalar_data,
        cell_vector_data,
    )

    name = filepath.name
    if i_ts is not None:
        name += f"_ts_{i_ts}"
    filename_full = Path(filepath.parent).joinpath(name).with_suffix(".vts")

    writer = vtk.vtkXMLStructuredGridWriter()
    configure_fast_writer(writer)
    writer.SetFileName(str(filename_full))
    writer.SetInputData(sg)
    writer.Write()

    return Path(filename_full)


class GridVTSSeries:
    """Reusable structured-grid VTS writer. Grid dimensions are fixed at construction. Per-timestep call refresh the data"""

    def __init__(
        self,
        grid_arr_ref: Array,
        base_filename: str | os.PathLike,
    ) -> None:
        _check_planar(grid_arr_ref)
        self._base = Path(base_filename)

        self.sg = vtk.vtkStructuredGrid()
        self.sg.SetDimensions(*grid_arr_ref.shape[:-1], 1)

        self.sg.SetPoints(make_grid_points(to_host(grid_arr_ref)))

        self.writer = vtk.vtkXMLStructuredGridWriter()
        configure_fast_writer(self.writer)
        self.writer.SetInputData(self.sg)

    def set_base_filename(self, base_filename: str | os.PathLike) -> None:
        """Change output path without rebuilding grid topology."""
        self._base = Path(base_filename)

    def write(
        self,
        grid_arr: Array,
        i_ts: int | None,
        *,
        node_scalar_data: dict[str, Array] | None = None,
        node_vector_data: dict[str, Array] | None = None,
        cell_scalar_data: dict[str, Array] | None = None,
        cell_vector_data: dict[str, Array] | None = None,
    ) -> Path:
        r"""
        For a time step, write the grid and associated data to a VTS file. The grid dimensions must match those of the
        reference grid used to construct the object.
        """
        self.sg.SetPoints(make_grid_points(to_host(grid_arr)))
        self.sg.GetPointData().Initialize()
        self.sg.GetCellData().Initialize()
        attach_grid_arrays(
            sg=self.sg,
            node_scalar_data=node_scalar_data,
            node_vector_data=node_vector_data,
            cell_scalar_data=cell_scalar_data,
            cell_vector_data=cell_vector_data,
        )

        name = self._base.name
        if i_ts is not None:
            name += f"_ts_{i_ts}"
        filename_full = self._base.parent.joinpath(name).with_suffix(".vts")
        self.writer.SetFileName(str(filename_full))
        self.writer.Write()
        return filename_full
