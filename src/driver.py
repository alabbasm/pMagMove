from numba import njit
from numba.typed import List
import numpy as np
from magmove import Swimmer, Environment
from helper import *


@njit
def run_brownian_swarm_numba_wchemo_storeC_strided(
    swimmers, dt, final_time, env, save_stride
):
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
    
        if (t/n_steps)*100 % 10 == 0:
            print((t/n_steps)*100, "%")

    return positions_hist, C_hist, orientation_hist, t_out


if __name__ == "__main__":
    # global
    n_swimmers = 1000
    dt = 0.002
    final_time = 1000

    # shared numeric parameters
    v_self = 14.2e-6    # m/s
    gamma_t = 5.1e-8
    gamma_r = 6.8e-20
    T = 305.0           # K
    Teff = 4.2e4        # K
    M_mag = 6e-6         # A·m

    # strategy
    strategy = 1       # 0: run-tumble, 1: run-reverse
    prev_state = 1   # start in RUN
    state = 1      # start in RUN

    # Codutti: mean run time t1_0 = 1.0 s in chemotaxis modes
    mean_run = 1.0
    mean_tumble = 0.14
    C_star = 10

    # chemotaxis mode and grad_ref
    kindofchemotaxis = 0   # 0: band, else off
    grad_ref = 25.0e4        # μM/mm 
    Cb = 70 # uM
    Ca = 10 

    B_arr = np.array([-50.0, 0.0, 0.0], dtype=np.float64) # 50 uT in -x 
    F_ext_arr = np.zeros(3, dtype=np.float64)
    box_size_arr = np.array([0.002, 0.0006, 0.002], dtype=np.float64)

    rand_pos_flag = True
    rand_ori_flag = True

    # environment parameters
    diffusion_coeff = 2.1e-9    # m^2/s (2100 μm^2/s)
    # Interpret as μM/s per cell (grid independent). Tune to match Codutti's kO2 conversion if needed.
    consumption_rate = 8.33e-14     # μM/s per cell * s
    initial_conc = 70       # uM
    N_x = 100
    N_y = 30
    N_z = 100

    imin1, imax1 = 20, 40
    jmin1, jmax1 = 0, 30
    kmin1, kmax1 = 20, 40
    imin2, imax2 = 60, 80
    jmin2, jmax2 = 0, 30
    kmin2, kmax2 = 20, 40

    kappa_1, kappa_2 = 2e-5, 2e-5
    C_sink_1, C_sink_2 = 15, 5
    binding_rate_1, binding_rate_2 = 8, 0.8
    unbinding_rate_1, unbinding_rate_2 = 0.3, 0.3

    # create swimmers
    swimmers = List.empty_list(Swimmer.class_type.instance_type)
    for _ in range(n_swimmers):
        sw = Swimmer(
            v_self, gamma_t, gamma_r, T, Teff, dt,
            M_mag, B_arr, F_ext_arr, mean_run, mean_tumble,
            state,  # start running
            box_size_arr, rand_pos_flag, rand_ori_flag,
            kindofchemotaxis, grad_ref, C_star,strategy, prev_state
        )
        swimmers.append(sw)

    # create environment
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
        Ca
    )

    save_stride = 100
    positions_hist, C_hist, orientation_hist, t_out = \
            run_brownian_swarm_numba_wchemo_storeC_strided(swimmers, dt, final_time, env,save_stride)
    # save the concentration history
    np.save("chist_rev_nobinding_neg_binding_n1000t1000_155.npy", C_hist)
    np.save("pos_rev_nobinding_neg_binding_n1000t1000_155.npy", positions_hist)
    #np.save("ori_test_rev.npy", orientation_hist)

    print(f"Simulated {n_swimmers} swimmers for {final_time} s.")
    print(f"strided snapshots: {save_stride}, total: {t_out}")
