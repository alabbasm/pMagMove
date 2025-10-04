## Collecution of helper functions 

from numba import njit
import numpy as np

@njit
def add_box_region(region_id, region_val, imin, imax, jmin, jmax, kmin, kmax):
    # mark [imin:imax), etc. Assumes bounds checked by caller.
    for i in range(imin, imax):
        for j in range(jmin, jmax):
            for k in range(kmin, kmax):
                region_id[i, j, k] = region_val

@njit
def build_face_counts(region_id, region_val):
    nx, ny, nz = region_id.shape
    nfaces = np.zeros((nx, ny, nz), dtype=np.int8)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if region_id[i, j, k] != 0:  # only bulk voxels will receive Robin penalty
                    continue
                cnt = 0
                # -x
                if i > 0 and region_id[i-1, j, k] == region_val:
                    cnt += 1
                # +x
                if i < nx-1 and region_id[i+1, j, k] == region_val:
                    cnt += 1
                # -y
                if j > 0 and region_id[i, j-1, k] == region_val:
                    cnt += 1
                # +y
                if j < ny-1 and region_id[i, j+1, k] == region_val:
                    cnt += 1
                # -z
                if k > 0 and region_id[i, j, k-1] == region_val:
                    cnt += 1
                # +z
                if k < nz-1 and region_id[i, j, k+1] == region_val:
                    cnt += 1
                nfaces[i, j, k] = cnt
    return nfaces


@njit
def check_region(position, env, which):
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


def _autocorr_fft_1d(x):
    """
    Return autocorrelation sum_{t} x[t] * x[t+tau] for tau=0..M-1.
    No mean removal (not needed for MSD formula below).
    Uses zero-padding to avoid circular wrap.
    """
    x = np.asarray(x, dtype=np.float64)
    M = x.shape[0]
    nfft = 1 << (2*M - 1).bit_length()  # pow2 >= 2M
    xp = np.zeros(nfft, dtype=np.float64)
    xp[:M] = x
    F = np.fft.rfft(xp)
    S = F * np.conj(F)
    acf = np.fft.irfft(S, nfft)[:M].real
    return acf

def msd_fft_single(r):
    """
    MSD for a single trajectory r with shape (M, d).
    Returns MSD[0..M-1] with time-origin averaging (unbiased by counts).
    """
    r = np.asarray(r, dtype=np.float64)
    if r.ndim == 1:
        r = r[:, None]
    M, d = r.shape

    # sum over dimensions of the autocorrelation r(t)·r(t+τ)
    acf_dot = np.zeros(M, dtype=np.float64)
    for k in range(d):
        acf_dot += _autocorr_fft_1d(r[:, k])

    # prefix sums of |r|^2 to get windowed sums quickly
    rsq = np.sum(r*r, axis=1)                 # (M,)
    csum = np.concatenate(([0.0], np.cumsum(rsq)))  # len M+1

    taus = np.arange(M)                       # 0..M-1
    counts = (M - taus).astype(np.float64)    # number of pairs per τ

    # sum_{t=0}^{M-τ-1} |r(t)|^2 = csum[M-τ] - csum[0]
    sum_rsq_t      = csum[M - taus] - csum[0]
    # sum_{t=0}^{M-τ-1} |r(t+τ)|^2 = csum[M] - csum[taus]
    sum_rsq_t_tau  = csum[M] - csum[taus]

    # MSD(τ) = (sum |r(t+τ)|^2 + sum |r(t)|^2 - 2 * sum r(t)·r(t+τ)) / counts
    msd = (sum_rsq_t + sum_rsq_t_tau - 2.0 * acf_dot) / counts
    return msd

def compute_msd_fft(positions, max_lag=None, stride=1, average=True):
    """
    positions: array with shape (N, M, d)  [N particles, M frames, d dims]
               (if you have (M, N, d), pass positions.swapaxes(0,1))
    max_lag:   compute up to this lag (inclusive); None -> M-1
    stride:    subsample frames to reduce M (e.g., 10 keeps every 10th frame)
    average:   if True, return particle-mean MSD; else return (N, L) per-particle.
    """
    pos = np.asarray(positions, dtype=np.float64)
    assert pos.ndim == 3, "positions must be (N, M, d)"
    N, M, d = pos.shape

    # optional temporal downsampling
    pos = pos[:, ::stride, :]
    M = pos.shape[1]

    L = M if max_lag is None else int(min(max_lag, M-1)) + 1
    out = np.empty((N, L), dtype=np.float64)
    for n in range(N):
        msd_full = msd_fft_single(pos[n])
        out[n] = msd_full[:L]
    return out.mean(axis=0) if average else out
