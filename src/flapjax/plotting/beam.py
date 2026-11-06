from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import jax
import numpy as np
import vtk
from jax import Array
from jax import numpy as jnp

from flapjax.algebra.se3 import exp_se3, hg_inv, hg_to_d
from flapjax.plotting.writer import (
    configure_fast_writer,
    make_line_cell_array,
    make_poly_line_cell_array,
    make_vtk_points,
    numpy_to_vtk_scalar,
    numpy_to_vtk_vector,
    to_host,
)
from flapjax.utils.print_utils import warn


def interpolate_beam(
    hg1: Array, hg2: Array, o0: Array, n_interp: int, include_endpoints: bool = False
) -> Array:
    """
    Interpolate beam geometry and orientation between two nodes.
    :param hg1: SE(3) transform of node 1, ``(4, 4)``
    :param hg2: SE(3) transform of node 2, ``(4, 4)``
    :param o0: Local beam orientation transformation, ``(3, 3)``
    :param n_interp: Number of interpolation points to compute.
    :param include_endpoints: Whether to include the original nodes in the output (if False, only the interpolated points are returned)
    :return: Interpolated SE(3) transforms along the beam, ``(n_interp, 4, 4)``
    """
    s_l = (
        jnp.linspace(0.0, 1.0, n_interp)
        if include_endpoints
        else jnp.linspace(0.0, 1.0, n_interp + 2)[1:-1]
    )

    h0 = jnp.zeros((4, 4))
    h0 = h0.at[:3, :3].set(o0)
    h0 = h0.at[3, 3].set(1.0)

    hg1h0 = hg1 @ h0
    hg2h0 = hg2 @ h0

    d = hg_to_d(hg1h0, hg2h0)

    s_l_d = jnp.outer(s_l, d)

    exp_s_l_d = jax.vmap(exp_se3, 0, 0)(s_l_d)
    return jnp.einsum("ij,hjk,kl->hil", hg1h0, exp_s_l_d, hg_inv(h0))


def _interpolate_all_elements(
    hg: Array, conn: Array, o0: Array, n_interp: int
) -> Array:
    """Vectorised beam interpolation for every element."""
    return jax.vmap(
        partial(interpolate_beam, n_interp=n_interp, include_endpoints=False),
        (0, 0, 0),
        0,
    )(hg[conn[:, 0], ...], hg[conn[:, 1], ...], o0)


def _build_beam_topology(
    hg: Array, conn: Array, o0: Array, n_interp: int
) -> tuple[vtk.vtkUnstructuredGrid, np.ndarray, np.ndarray | None]:
    """Build the beam VTK grid with topology + points.

    :return: (grid, coords_host_(n_pts, 3), full_hg_host or None). ``full_hg``
        is returned only when interpolation is active, so downstream data
        attachment can pad node arrays with the interpolated slots.
    """
    conn_host = np.asarray(conn, dtype=np.int64)
    if conn_host.ndim != 2 or conn_host.shape[1] != 2:
        raise ValueError("Connectivity must be a 2D array with shape (n_elem, 2)")

    n_nodes = int(hg.shape[0])
    n_elems = int(conn_host.shape[0])
    ug = vtk.vtkUnstructuredGrid()

    if n_interp > 0:
        interp_hg = _interpolate_all_elements(hg, conn, o0, n_interp)
        interp_hg_host = to_host(interp_hg)  # (n_elem, n_interp, 4, 4)

        coords_nodes = to_host(hg[:, :3, 3])  # (n_nodes, 3)
        coords_interp = interp_hg_host[:, :, :3, 3].reshape(-1, 3)
        coords = np.concatenate((coords_nodes, coords_interp), axis=0)

        added_nodes = np.arange(
            n_nodes, n_nodes + n_interp * n_elems, dtype=np.int64
        ).reshape(n_elems, n_interp)
        interp_conns = np.concatenate(
            (conn_host[:, [0]], added_nodes, conn_host[:, [1]]), axis=1
        )  # (n_elems, 2 + n_interp)

        ug.SetPoints(make_vtk_points(coords))
        ug.SetCells(vtk.VTK_POLY_LINE, make_poly_line_cell_array(interp_conns))
        return ug, coords, interp_hg_host

    coords = to_host(hg[:, :3, 3])
    ug.SetPoints(make_vtk_points(coords))
    ug.SetCells(vtk.VTK_LINE, make_line_cell_array(conn_host))
    return ug, coords, None


def create_beam_unstructured_grid(
    hg: Array,
    conn: Array,
    o0: Array,
    n_interp: int,
) -> tuple[vtk.vtkUnstructuredGrid, Array | None, Array | None]:
    """
    Create a VTK UnstructuredGrid representing line (beam) elements.
    :param hg: Array of node SE(3) transformations, ``(n_nodes, 4, 4)``.
    :param conn: Connectivity array with shape, ``(n_elem, 2)``.
    :param o0: Array of local beam orientation transformations, ``(n_elem, 3, 3)``.
    :param n_interp: Number of interpolation points to add along each beam element (does not include endpoints).
    :return: vtkUnstructuredGrid with VTK_LINE cells, array of SE(3) transforms interpolated case, and element mapping
    array (mapping each new interpolated element to the original element index).
    """
    ug, _, interp_hg_host = _build_beam_topology(hg, conn, o0, n_interp)

    if n_interp == 0:
        return ug, None, None

    n_elems = int(conn.shape[0])
    endpoints_start = to_host(hg[conn[:, 0], ...])[:, None, :, :]
    endpoints_end = to_host(hg[conn[:, 1], ...])[:, None, :, :]
    full_hg = np.concatenate((endpoints_start, interp_hg_host, endpoints_end), axis=1)
    elem_map = np.repeat(np.arange(n_elems), 1 + n_interp)
    return ug, jnp.asarray(full_hg), jnp.asarray(elem_map)


def _pad_node_scalar(arr: np.ndarray, n_nodes: int, n_pad: int) -> np.ndarray:
    if n_pad == 0:
        return arr
    out = np.zeros(n_nodes + n_pad, dtype=arr.dtype)
    out[:n_nodes] = arr.ravel()
    return out


def _pad_node_vector(arr: np.ndarray, n_nodes: int, n_pad: int) -> np.ndarray:
    if n_pad == 0:
        return arr
    out = np.zeros((n_nodes + n_pad, 3), dtype=arr.dtype)
    out[:n_nodes, :] = arr.reshape(n_nodes, 3)
    return out


def _attach_arrays(
    ug: vtk.vtkUnstructuredGrid,
    n_nodes: int,
    n_elems: int,
    n_pad_nodes: int,
    node_scalar_data: dict[str, Array | None] | None,
    node_vector_data: dict[str, Array | None] | None,
    cell_scalar_data: dict[str, Array | None] | None,
    cell_vector_data: dict[str, Array | None] | None,
) -> None:
    if node_scalar_data:
        for name, arr in node_scalar_data.items():
            if arr is None:
                continue
            arr_h = to_host(arr)
            if arr_h.shape[0] != n_nodes:
                raise ValueError(
                    f"Node scalar '{name}' has incorrect length {arr_h.shape[0]}; expected {n_nodes}"
                )
            padded = _pad_node_scalar(arr_h, n_nodes, n_pad_nodes)
            ug.GetPointData().AddArray(numpy_to_vtk_scalar(padded, name))

    if node_vector_data:
        for name, arr in node_vector_data.items():
            if arr is None:
                continue
            arr_h = to_host(arr)
            if arr_h.shape != (n_nodes, 3):
                raise ValueError(
                    f"Node vector '{name}' must have shape {(n_nodes, 3)}, got {arr_h.shape}"
                )
            padded = _pad_node_vector(arr_h, n_nodes, n_pad_nodes)
            ug.GetPointData().AddArray(numpy_to_vtk_vector(padded, name))

    if cell_scalar_data:
        for name, arr in cell_scalar_data.items():
            if arr is None:
                continue
            arr_h = to_host(arr)
            if arr_h.shape != (n_elems,):
                raise ValueError(
                    f"Cell scalar '{name}' has incorrect shape {arr_h.shape}; expected {(n_elems,)}"
                )
            ug.GetCellData().AddArray(numpy_to_vtk_scalar(arr_h, name))

    if cell_vector_data:
        for name, arr in cell_vector_data.items():
            if arr is None:
                continue
            arr_h = to_host(arr)
            if arr_h.shape != (n_elems, 3):
                raise ValueError(
                    f"Cell vector '{name}' must have shape {(n_elems, 3)}, got {arr_h.shape}"
                )
            ug.GetCellData().AddArray(numpy_to_vtk_vector(arr_h, name))


def plot_beam_to_vtk(
    hg: Array,
    conn: Array,
    o0: Array,
    n_interp: int,
    filename: str | os.PathLike,
    i_ts: int | None = None,
    node_scalar_data: dict[str, Array | None] | None = None,
    node_vector_data: dict[str, Array | None] | None = None,
    cell_scalar_data: dict[str, Array | None] | None = None,
    cell_vector_data: dict[str, Array | None] | None = None,
) -> Path:
    """
    Write beam (line element) data to a VTU file.

    :param hg: Array of SE(3) elements, ``(n_nodes, 4, 4)``
    :param conn: Connectivity array, ``(n_elem, 2)``
    :param o0: Array of local beam orientation transformations, ``(n_elem, 3, 3)``
    :param n_interp: Number of interpolation points to add along each beam element (does not include endpoints)
    :param filename: Base filename (directory + base name); _ts_<i_ts> will be appended if i_ts provided
    :param i_ts: Optional time step index to append to filename
    :param node_scalar_data: dict of ``{name, (n_nodes)}``
    :param node_vector_data: dict of ``{name, (n_nodes, 3)}``
    :param cell_scalar_data: dict of ``{name, (n_elem)}``
    :param cell_vector_data: dict of ``{name, (n_elem, 3)}``
    :return: Path of the written VTU file
    """

    if n_interp < 0:
        warn(
            "Number of interpolation points cannot be negative; defaulting to 0 (no interpolation)"
        )
        n_interp = 0

    filepath = Path(filename)
    n_nodes = int(hg.shape[0])
    n_elems = int(conn.shape[0])

    ug, _, _ = _build_beam_topology(hg, conn, o0, n_interp)
    n_pad_nodes = n_interp * n_elems if n_interp > 0 else 0
    _attach_arrays(
        ug,
        n_nodes,
        n_elems,
        n_pad_nodes,
        node_scalar_data,
        node_vector_data,
        cell_scalar_data,
        cell_vector_data,
    )

    name = filepath.name
    if i_ts is not None:
        name += f"_ts_{i_ts}"
    filename_full = Path(filepath.parent).joinpath(name).with_suffix(".vtu")

    writer = vtk.vtkXMLUnstructuredGridWriter()
    configure_fast_writer(writer)
    writer.SetFileName(str(filename_full))
    writer.SetInputData(ug)
    writer.Write()

    return Path(filename_full)


class BeamVTUSeries:
    """
    Reusable-topology beam VTU writer.
    Build once from a reference and refresh the values for each timestep.
    """

    def __init__(
        self,
        hg_ref: Array,
        conn: Array,
        o0: Array,
        n_interp: int,
        base_filename: str | os.PathLike,
    ) -> None:
        if n_interp < 0:
            warn(
                "Number of interpolation points cannot be negative; defaulting to 0 (no interpolation)"
            )
            n_interp = 0
        self._n_interp = n_interp
        self._n_nodes = int(hg_ref.shape[0])
        self._n_elems = int(conn.shape[0])
        self._n_pad_nodes = n_interp * self._n_elems if n_interp > 0 else 0
        self._conn = np.asarray(conn, dtype=np.int64)
        self._o0 = o0
        self._base = Path(base_filename)

        self._ug, self._coords, _ = _build_beam_topology(hg_ref, conn, o0, n_interp)

        self._writer = vtk.vtkXMLUnstructuredGridWriter()
        configure_fast_writer(self._writer)
        self._writer.SetInputData(self._ug)

    def set_base_filename(self, base_filename: str | os.PathLike) -> None:
        """Change output path without rebuilding mesh topology."""
        self._base = Path(base_filename)

    def _refresh_points(self, hg: Array) -> None:
        if self._n_interp > 0:
            interp_hg = _interpolate_all_elements(
                hg, jnp.asarray(self._conn), self._o0, self._n_interp
            )
            interp_hg_host = to_host(interp_hg)
            coords_nodes = to_host(hg[:, :3, 3])
            coords_interp = interp_hg_host[:, :, :3, 3].reshape(-1, 3)
            coords = np.concatenate((coords_nodes, coords_interp), axis=0)
        else:
            coords = to_host(hg[:, :3, 3])
        self._ug.SetPoints(make_vtk_points(coords))

    def write(
        self,
        hg: Array,
        i_ts: int | None,
        *,
        node_scalar_data: dict[str, Array | None] | None = None,
        node_vector_data: dict[str, Array | None] | None = None,
        cell_scalar_data: dict[str, Array | None] | None = None,
        cell_vector_data: dict[str, Array | None] | None = None,
    ) -> Path:
        self._refresh_points(hg)

        self._ug.GetPointData().Initialize()
        self._ug.GetCellData().Initialize()
        _attach_arrays(
            self._ug,
            self._n_nodes,
            self._n_elems,
            self._n_pad_nodes,
            node_scalar_data,
            node_vector_data,
            cell_scalar_data,
            cell_vector_data,
        )

        name = self._base.name
        if i_ts is not None:
            name += f"_ts_{i_ts}"
        filename_full = self._base.parent.joinpath(name).with_suffix(".vtu")
        self._writer.SetFileName(str(filename_full))
        self._writer.Write()
        return filename_full
