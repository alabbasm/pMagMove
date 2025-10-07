"""Simulation driver.

Provides the simulation loop and a CLI that reads parameters from a text file
with simple key=value pairs. See the repository `params.example.txt` for a
template of supported parameters.
"""

import argparse
import os
from numba import njit
from numba.typed import List
import numpy as np
from magmove import Swimmer, Environment
from helper import check_region


@njit
def run_brownian_swarm_numba_wchemo_storeC_strided(swimmers, dt, final_time, env, save_stride):
    """
    C_hist shape must be (n_saves, Ny, Nx, Nz)
    Returns:
      positions_hist: (n_saves, n_swimmers, 3)
      n_saves_written
    """
    n_swimmers = len(swimmers)
    n_steps = int(final_time / dt)
    C_star = env.C_star

    # how many snapshots (include t=0)
    if save_stride <= 0:
        save_stride = 1
    n_saves = n_steps // save_stride + 1

    # allocate outputs that are strided identically
    C_hist = np.zeros((n_saves, env.N_y, env.N_x, env.N_z), dtype=np.float64)
    positions_hist = np.zeros((n_saves, n_swimmers, 3), dtype=np.float64)
    orientation_hist = np.zeros((n_saves, n_swimmers, 3), dtype=np.float64)

    # working fields
    C = np.empty((env.N_y, env.N_x, env.N_z), dtype=np.float64)
    rho = np.zeros((env.N_y, env.N_x, env.N_z), dtype=np.float64)

    # init C
    for j in range(env.N_y):
        for i in range(env.N_x):
            for k in range(env.N_z):
                C[j, i, k] = env.initial_conc

    # store initial positions (t=0) into positions_hist and C_hist[0]
    for s in range(n_swimmers):
        positions_hist[0, s, :] = swimmers[s].position
    # write initial C
    C_hist[0, :, :, :] = C
    t_out = 1  # next snapshot index (we already wrote t=0)

    # main time stepping
    for t in range(1, n_steps + 1):
        # zero rho
        for j in range(env.N_y):
            for i in range(env.N_x):
                for k in range(env.N_z):
                    rho[j, i, k] = 0.0

        # accumulate swimmers into rho
        for s in range(n_swimmers):
            x, y, z = swimmers[s].position
            ix = int(x / env.box_size[0] * env.N_x)
            iy = int(y / env.box_size[1] * env.N_y)
            iz = int(z / env.box_size[2] * env.N_z)
            if ix < 0: ix = 0
            elif ix >= env.N_x: ix = env.N_x - 1
            if iy < 0: iy = 0
            elif iy >= env.N_y: iy = env.N_y - 1
            if iz < 0: iz = 0
            elif iz >= env.N_z: iz = env.N_z - 1
            rho[iy, ix, iz] += 1.0

        # update oxygen
        env.oxygen_step_ftcs_3d_with_robin_dirichlet_interior(C, rho)

        # gradient
        del_C = env.get_del_C(C)

        # advance swimmers
        for s in range(n_swimmers):
            # sample current grid cell
            x, y, z = swimmers[s].position
            ix = int(x / env.box_size[0] * env.N_x)
            iy = int(y / env.box_size[1] * env.N_y)
            iz = int(z / env.box_size[2] * env.N_z)
            if ix < 0: ix = 0
            elif ix >= env.N_x: ix = env.N_x - 1
            if iy < 0: iy = 0
            elif iy >= env.N_y: iy = env.N_y - 1
            if iz < 0: iz = 0
            elif iz >= env.N_z: iz = env.N_z - 1

            grad_here = del_C[iy, ix, iz, :]
            C_here = C[iy, ix, iz]

            # binding/unbinding BEFORE movement
            r_1 = check_region(swimmers[s].position, env, 1)
            r_2 = check_region(swimmers[s].position, env, 2)

            p_bind_1 = 1.0 - np.exp(-env.binding_rate_1 * dt)
            p_bind_2 = 1.0 - np.exp(-env.binding_rate_2 * dt)
            p_unbind_1 = 1.0 - np.exp(-env.unbinding_rate_1 * dt)
            p_unbind_2 = 1.0 - np.exp(-env.unbinding_rate_2 * dt)

            if r_1 == 1 and swimmers[s].bound_state == 0:
                if np.random.random() < p_bind_1:
                    swimmers[s].bound_state = 1
            elif r_2 == 2 and swimmers[s].bound_state == 0:
                if np.random.random() < p_bind_2:
                    swimmers[s].bound_state = 2
            else:
                if swimmers[s].bound_state == 1 and np.random.random() < p_unbind_1:
                    swimmers[s].bound_state = 0
                elif swimmers[s].bound_state == 2 and np.random.random() < p_unbind_2:
                    swimmers[s].bound_state = 0

            # step
            swimmers[s].step(grad_here, C_here)

        # strided snapshot
        if (t % save_stride) == 0:
            # bounds check in case of rounding
            if t_out < n_saves:
                # positions
                for s in range(n_swimmers):
                    positions_hist[t_out, s, :] = swimmers[s].position
                    orientation_hist[t_out, s, :] = swimmers[s].orientation
                # oxygen
                C_hist[t_out, :, :, :] = C
                t_out += 1
    
        # lightweight progress hook (integer percent every 10%)
        # Note: printing from nopython is limited; this keeps it simple
        if (t * 100) // n_steps % 10 == 0 and (t % save_stride) == 0:
            print((t * 100) // n_steps, "%")

    return positions_hist, C_hist, orientation_hist, t_out


def _as_bool(val):
    v = str(val).strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _maybe_vector(val, length=3):
    if isinstance(val, (list, tuple, np.ndarray)):
        return np.array(val, dtype=np.float64)
    if isinstance(val, str) and "," in val:
        parts = [float(x) for x in val.split(",")]
        arr = np.array(parts, dtype=np.float64)
        if arr.shape[0] != length:
            raise ValueError(f"Expected vector of length {length}, got {arr}")
        return arr
    # scalar replicated
    f = float(val)
    return np.full(length, f, dtype=np.float64)


def parse_params_file(path):
    params = {}
    with open(path, "r") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            params[k] = v
    return params


def build_sim_from_params(p):
    # defaults (match previous hard-coded ones)
    n_swimmers = int(p.get("n_swimmers", 1000))
    dt = float(p.get("dt", 0.002))
    final_time = float(p.get("final_time", 1000.0))

    v_self = float(p.get("v_self", 14.2e-6))
    gamma_t = float(p.get("gamma_t", 5.1e-8))
    gamma_r = float(p.get("gamma_r", 6.8e-20))
    T = float(p.get("T", 305.0))
    Teff = float(p.get("Teff", 4.2e4))
    M_mag = float(p.get("M_mag", 6e-16))

    strategy = int(p.get("strategy", 1))
    prev_state = int(p.get("prev_state", 1))
    state = int(p.get("state", 1))

    mean_run = float(p.get("mean_run", 1.0))
    mean_tumble = float(p.get("mean_tumble", 0.14))
    C_star = float(p.get("C_star", 10.0))

    kindofchemotaxis = int(p.get("kindofchemotaxis", 0))
    grad_ref = float(p.get("grad_ref", 25.0e4))
    Cb = float(p.get("Cb", 70.0))
    Ca = float(p.get("Ca", 10.0))

    B_arr = _maybe_vector(p.get("B", p.get("B_arr", "-35.36,-35.36,0.0")))
    F_ext_arr = _maybe_vector(p.get("F_ext", p.get("F_ext_arr", "0,0,0")))
    box_size_arr = _maybe_vector(p.get("box_size", p.get("box_size_arr", "0.002,0.0006,0.002")))

    rand_pos_flag = _as_bool(p.get("rand_pos_flag", True))
    rand_ori_flag = _as_bool(p.get("rand_ori_flag", True))

    diffusion_coeff = float(p.get("diffusion_coeff", 2.1e-9))
    consumption_rate = float(p.get("consumption_rate", 8.33e-14))
    initial_conc = float(p.get("initial_conc", 70.0))
    N_x = int(p.get("N_x", 100))
    N_y = int(p.get("N_y", 30))
    N_z = int(p.get("N_z", 100))

    imin1 = int(p.get("imin1", 20)); imax1 = int(p.get("imax1", 40))
    jmin1 = int(p.get("jmin1", 0));  jmax1 = int(p.get("jmax1", 30))
    kmin1 = int(p.get("kmin1", 20)); kmax1 = int(p.get("kmax1", 40))
    imin2 = int(p.get("imin2", 60)); imax2 = int(p.get("imax2", 80))
    jmin2 = int(p.get("jmin2", 0));  jmax2 = int(p.get("jmax2", 30))
    kmin2 = int(p.get("kmin2", 20)); kmax2 = int(p.get("kmax2", 40))

    kappa_1 = float(p.get("kappa_1", 2e-5)); kappa_2 = float(p.get("kappa_2", 2e-5))
    C_sink_1 = float(p.get("C_sink_1", 5.0)); C_sink_2 = float(p.get("C_sink_2", 15.0))
    binding_rate_1 = float(p.get("binding_rate_1", 8.0)); binding_rate_2 = float(p.get("binding_rate_2", 0.8))
    unbinding_rate_1 = float(p.get("unbinding_rate_1", 0.3)); unbinding_rate_2 = float(p.get("unbinding_rate_2", 0.3))

    save_stride = int(p.get("save_stride", 100))
    out_prefix = p.get("out_prefix", "sim")
    out_dir = p.get("out_dir", ".")

    # swimmers
    swimmers = List.empty_list(Swimmer.class_type.instance_type)
    for _ in range(n_swimmers):
        sw = Swimmer(
            v_self, gamma_t, gamma_r, T, Teff, dt,
            M_mag, B_arr, F_ext_arr, mean_run, mean_tumble,
            state,
            box_size_arr, rand_pos_flag, rand_ori_flag,
            kindofchemotaxis, grad_ref, C_star, strategy, prev_state,
        )
        swimmers.append(sw)

    env = Environment(
        box_size_arr, dt, final_time,
        diffusion_coeff, consumption_rate,
        initial_conc, N_x, N_y, N_z,
        kappa_1, kappa_2, C_sink_1, C_sink_2,
        imin1, imax1, jmin1, jmax1, kmin1, kmax1,
        imin2, imax2, jmin2, jmax2, kmin2, kmax2,
        binding_rate_1, binding_rate_2,
        unbinding_rate_1, unbinding_rate_2,
        C_star,
        Cb,
        Ca,
    )

    return (
        swimmers, dt, final_time, env, save_stride,
        n_swimmers, out_prefix, out_dir,
    )


def main():
    parser = argparse.ArgumentParser(description="Run pMagMove simulation from a params file.")
    parser.add_argument("--params", default="params.txt", help="Path to params.txt file")
    args = parser.parse_args()

    params_path = args.params
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Params file not found: {params_path}")

    p = parse_params_file(params_path)
    (
        swimmers, dt, final_time, env, save_stride,
        n_swimmers, out_prefix, out_dir,
    ) = build_sim_from_params(p)

    positions_hist, C_hist, orientation_hist, t_out = run_brownian_swarm_numba_wchemo_storeC_strided(
        swimmers, dt, final_time, env, save_stride
    )

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{out_prefix}_C_hist.npy"), C_hist)
    np.save(os.path.join(out_dir, f"{out_prefix}_positions.npy"), positions_hist)
    # np.save(os.path.join(out_dir, f"{out_prefix}_orientation.npy"), orientation_hist)

    print(f"Simulated {n_swimmers} swimmers for {final_time} s.")
    print(f"strided snapshots: {save_stride}, total: {t_out}")


if __name__ == "__main__":
    main()
