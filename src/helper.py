"""Helper utilities for region labeling and trajectory statistics.

This module contains njit-compatible utilities used by the simulation core,
as well as fast MSD computation helpers for analysis.
"""

from numba import njit
import numpy as np

@njit
def add_box_region(region_id, region_val, imin, imax, jmin, jmax, kmin, kmax):
    """Mark a rectangular region with a region id.

    Indexing follows region_id[i, j, k] with shapes (Nx, Ny, Nz). Ranges are
    half-open [min, max) and are assumed to be pre-clamped by the caller.
    """
    for i in range(imin, imax):
        for j in range(jmin, jmax):
            for k in range(kmin, kmax):
                region_id[i, j, k] = region_val

@njit
def build_face_counts(region_id, region_val):
    """Count adjacent faces that touch a given region id.

    For each bulk voxel (id==0), counts how many of its 6 neighbors belong to
    the specified region. Used to scale Robin penalties near sinks.
    """
    nx, ny, nz = region_id.shape
    nfaces = np.zeros((nx, ny, nz), dtype=np.int8)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if region_id[i, j, k] != 0:
                    continue
                cnt = 0
                # -x
                if i > 0 and region_id[i - 1, j, k] == region_val:
                    cnt += 1
                # +x
                if i < nx - 1 and region_id[i + 1, j, k] == region_val:
                    cnt += 1
                # -y
                if j > 0 and region_id[i, j - 1, k] == region_val:
                    cnt += 1
                # +y
                if j < ny - 1 and region_id[i, j + 1, k] == region_val:
                    cnt += 1
                # -z
                if k > 0 and region_id[i, j, k - 1] == region_val:
                    cnt += 1
                # +z
                if k < nz - 1 and region_id[i, j, k + 1] == region_val:
                    cnt += 1
                nfaces[i, j, k] = cnt
    return nfaces


@njit
def check_region(position, env, which):
    """Return region id at the swimmer's position for region 1 or 2.

    Maps continuous position to grid indices with clamping to domain bounds.
    """
    x, y, z = position
    ix = int(x / env.box_size[0] * env.N_x)
    iy = int(y / env.box_size[1] * env.N_y)
    iz = int(z / env.box_size[2] * env.N_z)

    if ix < 0:
        ix = 0
    elif ix >= env.N_x:
        ix = env.N_x - 1
    if iy < 0:
        iy = 0
    elif iy >= env.N_y:
        iy = env.N_y - 1
    if iz < 0:
        iz = 0
    elif iz >= env.N_z:
        iz = env.N_z - 1

    if which == 1:
        return env.region_id_1[ix, iy, iz]
    else:
        return env.region_id_2[ix, iy, iz]
