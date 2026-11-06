from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from jax import Array, vmap
from jax import numpy as jnp

from flapjax.aero.data_structures import AeroCase
from flapjax.algebra.array_utils import ArrayList
from flapjax.algebra.se3 import exp_se3
from flapjax.plotting.aerogrid import GridVTSSeries
from flapjax.plotting.beam import BeamVTUSeries
from flapjax.plotting.pvd import write_pvd
from flapjax.structure import StructureCase

if TYPE_CHECKING:
    from flapjax.aero.uvlm import UVLM
    from flapjax.coupled import AeroelasticCase


def _phase_real(arr: Array, phase: Array) -> Array:
    r"""
    Broadcast a ``(n_phase, )`` complex phase factor against an array of arbitrary rank.
    :param arr: Array to modulate, any shape.
    :param phase: Complex phase factors, ``(n_phase, )``.
    :return: Real part of ``arr * phase``, broadcast to ``(n_phase, *arr.shape)``.
    """
    return (arr[None, ...] * phase.reshape((phase.shape[0],) + (1,) * arr.ndim)).real


def plot_modes_vtu(
    reference: AeroelasticCase | StructureCase,
    q_full: Array,
    freqs: Array,
    dampings: Array,
    gamma_b_full: ArrayList | None,
    gamma_w_full: ArrayList | None,
    zeta_w_full: ArrayList | None,
    uvlm: UVLM | None,
    directory: os.PathLike | str = "./modal",
    n_phase: int = 8,
    n_interp: int = 0,
    max_disp: float = 0.2,
    max_ang: float = 0.2,
    max_gamma: float = 100.0,
) -> None:
    r"""
    Plot aeroelastic modes for visualisation in paraview.
    :param reference: Reference aeroelastic model used for linearisation.
    :param directory: Directory for saving data. If not passed, will default to ./modal.
    :param q_full: Full nodal mode deflections, ``(n_modes, n_nodes, 6)``.
    :param freqs: Damped mode frequencies, Hz. ``(n_modes, )``, used for file naming.
    :param dampings: Damping ratios, ``(n_modes, )``, used for file naming.
    :param gamma_b_full: Full bound gamma modes, (n_surf)(n_modes, m, n) or None.
    :param gamma_w_full: Full wake modes, ``(n_surf, )(n_modes, m*, n)`` or None.
    :param zeta_w_full: Full wake displacement modes, ``(n_surf, )(n_modes, m_star, n, 3)`` or None.
    :param uvlm: UVLM model, used for generating the aerodynamic grid.
    :param n_phase: Number of phase points to plot the modes.
    :param n_interp: Number of interpolation points for plotting the structure.
    :param max_disp: Maximum nodal displacement for mode normalisation.
    :param max_ang: Maximum nodal angle for mode normalisation.
    :param max_gamma: Maximum gamma for mode normalisation.
    """

    if (gamma_b_full is None and gamma_w_full is not None) or (
        gamma_w_full is None and gamma_b_full is not None
    ):
        raise ValueError(
            "Both gamma_b and gamma_w must simultaneously be None or not None."
        )

    if gamma_b_full is not None and gamma_w_full is not None and uvlm is None:
        raise ValueError("UVLM model must be provided when plotting aeroelastic modes.")

    phase_circle = jnp.exp(
        1j * jnp.linspace(0.0, 2.0 * jnp.pi, n_phase, endpoint=False)
    )

    plot_dir = Path(directory)
    plot_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(reference, StructureCase):
        struct_ref: StructureCase = reference
        aero_ref: AeroCase | None = None
    else:
        struct_ref: StructureCase = reference.structure
        aero_ref: AeroCase | None = reference.aero

    conn = jnp.array(struct_ref.conn, dtype=int)
    o0 = struct_ref.o0

    # mesh topology, reference aero grid, and the deflected control-surface reference grid are all identical
    # for every mode and every phase
    beam_series = BeamVTUSeries(
        hg_ref=struct_ref.hg,
        conn=conn,
        o0=o0,
        n_interp=n_interp,
        base_filename=plot_dir.joinpath("beam"),
    )

    if uvlm is not None:
        n_surf = uvlm.n_surf

        zeta_b0_cs = uvlm.grid_func(uvlm.zeta_b0, **uvlm.cs_ang0)
        zeta_b_ref = uvlm.hg_to_zeta_b(hg_n=struct_ref.hg, zeta_b0_cs=zeta_b0_cs)
        bound_series: list[GridVTSSeries] = [
            GridVTSSeries(
                grid_arr_ref=zeta_b_ref[i_surf],
                base_filename=plot_dir.joinpath(uvlm.surf_b_names[i_surf]),
            )
            for i_surf in range(n_surf)
        ]
        if aero_ref is not None:
            wake_active = [
                aero_ref.zeta_w[i_surf].shape[0] > 1 for i_surf in range(n_surf)
            ]
            wake_series: list[GridVTSSeries | None] = [
                GridVTSSeries(
                    grid_arr_ref=aero_ref.zeta_w[i_surf],
                    base_filename=plot_dir.joinpath(uvlm.surf_w_names[i_surf]),
                )
                if wake_active[i_surf]
                else None
                for i_surf in range(n_surf)
            ]
        else:
            wake_active = [False] * n_surf
            wake_series = [None] * n_surf
    else:
        n_surf = 0
        zeta_b0_cs = None
        bound_series = []
        wake_series = []
        wake_active = []

    n_modes = q_full.shape[0]
    for i_mode in range(n_modes):
        eig_max_disp = jnp.linalg.norm(q_full[i_mode, :, :3], axis=1).max()
        eig_max_ang = jnp.linalg.norm(q_full[i_mode, :, 3:], axis=1).max()

        eig_max_gamma = (
            jnp.maximum(
                jnp.abs(gamma_b_full.index_all(i_mode, ...).ravel()).max(),
                jnp.abs(gamma_w_full.index_all(i_mode, ...).ravel()).max()
                if gamma_w_full.size
                else 0.0,
            )
            if gamma_b_full is not None and gamma_w_full is not None
            else 1e-9
        )

        # scale to whichever limit is reached first; ignore vanishing components
        scale_factor = jnp.array(
            (
                jnp.where(eig_max_disp > 0, max_disp / eig_max_disp, jnp.inf),
                jnp.where(eig_max_ang > 0, max_ang / eig_max_ang, jnp.inf),
                jnp.where(eig_max_gamma > 0, max_gamma / eig_max_gamma, jnp.inf),
            )
        ).min()

        # rescale
        delta_q_full: Array = q_full[i_mode, ...] * scale_factor

        delta_gamma_b: ArrayList | None = (
            gamma_b_full.index_all(i_mode, ...) * scale_factor
            if gamma_b_full is not None
            else None
        )
        delta_gamma_w: ArrayList | None = (
            gamma_w_full.index_all(i_mode, ...) * scale_factor
            if gamma_w_full is not None
            else None
        )
        delta_zeta_w: ArrayList | None = (
            zeta_w_full.index_all(i_mode, ...) * scale_factor
            if zeta_w_full is not None
            else None
        )

        mode_dir = plot_dir.joinpath(
            f"mode_{i_mode + 1}_f{float(freqs[i_mode]):.2f}Hz_z{float(dampings[i_mode]):+.4f}"
        )
        mode_dir.mkdir(parents=True, exist_ok=True)

        beam_series.set_base_filename(mode_dir.joinpath("beam"))
        beam_paths: list[Path] = []

        if uvlm is not None:
            for i_surf in range(n_surf):
                bound_series[i_surf].set_base_filename(
                    mode_dir.joinpath(uvlm.surf_b_names[i_surf])
                )
                ws = wake_series[i_surf]
                if ws is not None:
                    ws.set_base_filename(mode_dir.joinpath(uvlm.surf_w_names[i_surf]))
            bound_paths: list[list[Path]] = [[] for _ in range(n_surf)]
            wake_paths: list[list[Path]] = [[] for _ in range(n_surf)]
        else:
            bound_paths = []
            wake_paths = []

        # batch into one pass, leaving the Python loop to do only the VTK writes
        q_pert_full_all = _phase_real(
            delta_q_full, phase_circle
        )  # (n_phase, n_nodes, 6)
        hg_all = jnp.einsum(
            "ijk,pikl->pijl",
            struct_ref.hg,
            vmap(vmap(exp_se3))(q_pert_full_all),
        )  # (n_phase, n_nodes, 4, 4)

        gamma_b_all: list[Array] | None = (
            [
                _phase_real(delta_gamma_b[i_surf], phase_circle)
                + aero_ref.gamma_b[i_surf][None, ...]
                for i_surf in range(n_surf)
            ]
            if delta_gamma_b is not None and aero_ref is not None
            else None
        )
        gamma_w_all: list[Array] | None = (
            [
                _phase_real(delta_gamma_w[i_surf], phase_circle)
                + aero_ref.gamma_w[i_surf][None, ...]
                for i_surf in range(n_surf)
            ]
            if delta_gamma_w is not None and aero_ref is not None
            else None
        )
        zeta_w_animated = delta_zeta_w is not None and aero_ref is not None
        if delta_zeta_w is not None and aero_ref is not None:
            zeta_w_all: list[Array] | None = [
                _phase_real(delta_zeta_w[i_surf], phase_circle)
                + aero_ref.zeta_w[i_surf][None, ...]
                for i_surf in range(n_surf)
            ]
        else:
            zeta_w_all = None

        if uvlm is not None:
            zeta_b_all: ArrayList | None = vmap(
                lambda hg_, _uvlm=uvlm: _uvlm.hg_to_zeta_b(
                    hg_n=hg_, zeta_b0_cs=zeta_b0_cs
                )
            )(hg_all)
        else:
            zeta_b_all = None

        for i_phase in range(n_phase):
            beam_paths.append(beam_series.write(hg=hg_all[i_phase], i_ts=i_phase))

            if uvlm is not None:
                assert zeta_b_all is not None
                for i_surf in range(n_surf):
                    bound_paths[i_surf].append(
                        bound_series[i_surf].write(
                            grid_arr=zeta_b_all[i_surf][i_phase],
                            i_ts=i_phase,
                            cell_scalar_data={"gamma": gamma_b_all[i_surf][i_phase]}
                            if gamma_b_all is not None
                            else None,
                        )
                    )

                    ws = wake_series[i_surf]
                    if (
                        gamma_w_all is not None
                        and wake_active[i_surf]
                        and ws is not None
                    ):
                        assert aero_ref is not None
                        if zeta_w_animated:
                            assert zeta_w_all is not None
                            zeta_w_frame = zeta_w_all[i_surf][i_phase]
                        else:
                            zeta_w_frame = aero_ref.zeta_w[i_surf]
                        wake_paths[i_surf].append(
                            ws.write(
                                grid_arr=zeta_w_frame,
                                i_ts=i_phase,
                                cell_scalar_data={
                                    "gamma": gamma_w_all[i_surf][i_phase]
                                },
                            )
                        )

        times = (
            jnp.linspace(0.0, 1.0 / freqs[i_mode], n_phase, endpoint=False)
        ).tolist()

        # link timesteps with PVD
        write_pvd(
            directory=mode_dir, name="beam", file_dirs=beam_paths, times=times
        )  # always write beam data

        for i_surf in range(n_surf):
            # write for aero grid data if it exists
            if bound_paths[i_surf] and uvlm is not None:
                write_pvd(
                    directory=mode_dir,
                    name=uvlm.surf_b_names[i_surf],
                    file_dirs=bound_paths[i_surf],
                    times=times,
                )
            if wake_paths[i_surf] and uvlm is not None:
                write_pvd(
                    directory=mode_dir,
                    name=uvlm.surf_w_names[i_surf],
                    file_dirs=wake_paths[i_surf],
                    times=times,
                )
