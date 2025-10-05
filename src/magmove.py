import numpy as np
from numba.experimental import jitclass
from numba import njit
from numba.types import float64, int8, int64, boolean
from numba.typed import List
from helper import *

kB = 1.380649e-23  # J/K

# spec for numba jitclass
spec_swimmer = [
    ('v_self', float64),
    ('base_v_self', float64),
    ('gamma_t', float64),
    ('gamma_r', float64),
    ('T', float64),
    ('Teff', float64),
    ('dt', float64),
    ('M_mag', float64),
    ('mean_run', float64),
    ('mean_tumble', float64),
    ('status', int8),  # 1=run, 0=tumble, 2=reverse
    ('time_in_state', float64),
    ('next_interval', float64),
    ('kindofchemotaxis', int64),
    ('grad_ref', float64),
    ('random_position', boolean),
    ('random_orientation', boolean),
    ('position', float64[:]),       # length-3 array
    ('orientation', float64[:]),    # length-3 array
    ('B', float64[:]),              # length-3 array
    ('F_ext', float64[:]),          # length-3 array
    ('box_size', float64[:]),       # length-3 array
    ('C_star', float64),
    ('regions_state',float64),
    ('bound_state',float64),
    ('strategy', boolean),
    ('prev_state', int8)
]

# spec for numba jitclass
spec_environment = [
    ('box_size', float64[:]),
    ('dt', float64),
    ('final_time', float64),
    ('diffusion_coeff', float64),
    ('consumption_rate', float64),
    ('initial_conc', float64),
    ('N_x', int64),
    ('N_y', int64),
    ('N_z', int64),
    ('region_id_1', int8[:, :, :]),
    ('region_id_2', int8[:, :, :]),
    ('kappa_1', float64),
    ('kappa_2', float64),
    ('C_sink_1', float64),
    ('C_sink_2', float64),
    ('nfaces1', int8[:, :, :]),
    ('nfaces2', int8[:, :, :]),
    ('binding_rate_1', float64),
    ('binding_rate_2', float64),
    ('unbinding_rate_1', float64),
    ('unbinding_rate_2', float64),
    ('C_star', float64),
    ('Cb', float64),
    ('Ca', float64)
]


@jitclass(spec_environment)
class Environment:
    """
    Environment for magnetoaerotactic Brownian swimmers:
    - Holds domain parameters and oxygen field
    - Advances a diffusion-consumption PDE
    - Provides local chemical gradient for swimmers
    - Applies reflective boundaries to swimmers
    """

    def __init__(
        self, box_size, dt, final_time,
        diffusion_coeff, consumption_rate,
        initial_conc, N_x, N_y, N_z,
        kappa_1, kappa_2, C_sink_1, C_sink_2,
        imin1, imax1, jmin1, jmax1, kmin1, kmax1,
        imin2, imax2, jmin2, jmax2, kmin2, kmax2,
        binding_rate_1, binding_rate_2,
        unbinding_rate_1, unbinding_rate_2,
        C_star,
        Cb,Ca
    ):
        # Domain parameters
        self.box_size = box_size
        self.dt = dt
        self.final_time = final_time
        self.diffusion_coeff = diffusion_coeff
        self.consumption_rate = consumption_rate
        self.initial_conc = initial_conc
        self.N_x = N_x
        self.N_y = N_y
        self.N_z = N_z
        self.region_id_1 = np.zeros((N_x, N_y, N_z), dtype=np.int8)
        self.region_id_2 = np.zeros((N_x, N_y, N_z), dtype=np.int8)
        self.binding_rate_1 = binding_rate_1
        self.binding_rate_2 = binding_rate_2
        self.kappa_1 = kappa_1
        self.kappa_2 = kappa_2
        self.C_sink_1 = C_sink_1
        self.C_sink_2 = C_sink_2
        self.binding_rate_1 = binding_rate_1
        self.binding_rate_2 = binding_rate_2
        self.unbinding_rate_1 = unbinding_rate_1
        self.unbinding_rate_2 = unbinding_rate_2
        self.C_star = C_star
        add_box_region(self.region_id_1, 1, imin1, imax1, jmin1, jmax1, kmin1, kmax1)  # box 1
        add_box_region(self.region_id_2, 2, imin2, imax2, jmin2, jmax2, kmin2, kmax2)  # box 2
        self.Cb = Cb
        self.nfaces1 = build_face_counts(self.region_id_1, 1)
        self.nfaces2 = build_face_counts(self.region_id_2, 2)
        self.Ca = Ca
    def oxygen_step_ftcs_3d_with_robin_dirichlet_interior(self, C, rho):
        Ny, Nx, Nz = self.N_y, self.N_x, self.N_z
        hx = self.box_size[0] / Nx
        hy = self.box_size[1] / Ny
        hz = self.box_size[2] / Nz

        invhx2 = 1.0 / (hx * hx)
        invhy2 = 1.0 / (hy * hy)
        invhz2 = 1.0 / (hz * hz)

        D   = self.diffusion_coeff
        dt  = self.dt
        Cb  = self.Cb     # Dirichlet at z- (air/water)
        Ca  = self.Ca     # Michaelis-Menten half-saturation (μM)
        k1  = self.kappa_1
        k2  = self.kappa_2
        C1  = self.C_sink_1
        C2  = self.C_sink_2

        Cnew = np.empty_like(C)

        for j in range(Ny):
            for i in range(Nx):
                for k in range(Nz):
                    # Determine if CURRENT voxel is inside a sink
                    # region_id_* arrays are indexed [i,j,k] (Nx,Ny,Nz) while C is [j,i,k] (Ny,Nx,Nz)
                    rid_here = 1 if self.region_id_1[i, j, k] == 1 else (2 if self.region_id_2[i, j, k] == 2 else 0)

                    # If inside a sink volume: enforce Dirichlet in the NEW field and skip update
                    if rid_here == 1:
                        Cnew[j, i, k] = C1
                        continue
                    elif rid_here == 2:
                        Cnew[j, i, k] = C2
                        continue

                    # Bulk voxel update 
                    cijk = C[j, i, k]

                    # Neighbors (default interior/Neumann handling by copying self at boundary)
                    C_xm = C[j, i-1, k] if i > 0     else C[j, i, k]
                    C_xp = C[j, i+1, k] if i < Nx-1 else C[j, i, k]
                    C_ym = C[j-1, i, k] if j > 0     else C[j, i, k]
                    C_yp = C[j+1, i, k] if j < Ny-1 else C[j, i, k]
                    # z- face: Dirichlet to Cb if k == 0; z+ default Neumann
                    C_zm = Cb if k == 0 else C[j, i, k-1]
                    C_zp = C[j, i, k+1] if k < Nz-1 else C[j, i, k]

                    # Robin flux terms for faces adjacent to sink voxels 
                    robin_sum = 0.0  # [μM/s], will be subtracted

                    # x- face
                    if i > 0:
                        ridm = 1 if self.region_id_1[i-1, j, k] == 1 else (2 if self.region_id_2[i-1, j, k] == 2 else 0)
                        if ridm == 1:
                            C_xm = cijk   
                            robin_sum += k1 * (cijk - C1) / hx  
                        elif ridm == 2:
                            C_xm = cijk  
                            robin_sum += k2 * (cijk - C2) / hx  

                    # x+ face
                    if i < Nx-1:
                        ridp = 1 if self.region_id_1[i+1, j, k] == 1 else (2 if self.region_id_2[i+1, j, k] == 2 else 0)
                        if ridp == 1:
                            C_xp = cijk   
                            robin_sum += k1 * (cijk - C1) / hx  
                        elif ridp == 2:
                            C_xp = cijk   
                            robin_sum += k2 * (cijk - C2) / hx  

                    # y- face
                    if j > 0:
                        ridm = 1 if self.region_id_1[i, j-1, k] == 1 else (2 if self.region_id_2[i, j-1, k] == 2 else 0)
                        if ridm == 1:
                            C_ym = cijk   
                            robin_sum += k1 * (cijk - C1) / hy  
                        elif ridm == 2:
                            C_ym = cijk   
                            robin_sum += k2 * (cijk - C2) / hy  

                    # y+ face
                    if j < Ny-1:
                        ridp = 1 if self.region_id_1[i, j+1, k] == 1 else (2 if self.region_id_2[i, j+1, k] == 2 else 0)
                        if ridp == 1:
                            C_yp = cijk  
                            robin_sum += k1 * (cijk - C1) / hy  
                        elif ridp == 2:
                            C_yp = cijk  
                            robin_sum += k2 * (cijk - C2) / hy  

                    # z- face (only if NOT already hard Dirichlet at domain boundary)
                    if k > 0:
                        ridm = 1 if self.region_id_1[i, j, k-1] == 1 else (2 if self.region_id_2[i, j, k-1] == 2 else 0)
                        if ridm == 1:
                            C_zm = cijk  
                            robin_sum += k1 * (cijk - C1) / hz  
                        elif ridm == 2:
                            C_zm = cijk  
                            robin_sum += k2 * (cijk - C2) / hz  

                    # z+ face
                    if k < Nz-1:
                        ridp = 1 if self.region_id_1[i, j, k+1] == 1 else (2 if self.region_id_2[i, j, k+1] == 2 else 0)
                        if ridp == 1:
                            C_zp = cijk  
                            robin_sum += k1 * (cijk - C1) / hz  
                        elif ridp == 2:
                            C_zp = cijk  
                            robin_sum += k2 * (cijk - C2) / hz  

                    # Laplacian (with sink-touching faces neutralized as above)
                    lap = ((C_xp + C_xm - 2.0 * cijk) * invhx2 +
                        (C_yp + C_ym - 2.0 * cijk) * invhy2 +
                        (C_zp + C_zm - 2.0 * cijk) * invhz2)

                    # Uptake (Michaelis-Menten per cell); rho is #cells per voxel
                    uptake = 0.0
                    if cijk > 0.0:
                        voxel_vol_L = (hx * hy * hz) * 1000.0  # m^3 -> L
                        cells_per_L = (rho[j, i, k] / voxel_vol_L) if voxel_vol_L > 0.0 else 0.0
                        uptake = self.consumption_rate * (cijk / (cijk + Ca)) * cells_per_L

                    # Explicit update
                    Cval = cijk + dt * (D * lap - uptake - robin_sum)  

                    # Clamp non-negative
                    Cnew[j, i, k] = 0.0 if Cval < 0.0 else Cval

        # Write back
        C[:, :, :] = Cnew

    def get_del_C(self, C):
        # Return gradient field with shape (Ny, Nx, Nz, 3)
        del_C = np.zeros((self.N_y, self.N_x, self.N_z, 3), dtype=np.float64)

        # physical voxel sizes and inverse
        hx = self.box_size[0] / self.N_x
        hy = self.box_size[1] / self.N_y
        hz = self.box_size[2] / self.N_z

        inv2hx = 0.5 / hx
        inv2hy = 0.5 / hy
        inv2hz = 0.5 / hz

        for j in range(self.N_y):
            for i in range(self.N_x):
                for k in range(self.N_z):
                    c_xp = C[j, i+1, k] if i < self.N_x - 1 else C[j, i, k]
                    c_xm = C[j, i-1, k] if i > 0 else C[j, i, k]

                    c_yp = C[j+1, i, k] if j < self.N_y - 1 else C[j, i, k]
                    c_ym = C[j-1, i, k] if j > 0 else C[j, i, k]

                    c_zp = C[j, i, k+1] if k < self.N_z - 1 else C[j, i, k]
                    c_zm = C[j, i, k-1] if k > 0 else C[j, i, k]

                    grad_x = (c_xp - c_xm) * inv2hx
                    grad_y = (c_yp - c_ym) * inv2hy
                    grad_z = (c_zp - c_zm) * inv2hz

                    del_C[j, i, k, 0] = grad_x
                    del_C[j, i, k, 1] = grad_y
                    del_C[j, i, k, 2] = grad_z

        return del_C


# jitted swimmer class
@jitclass(spec_swimmer)
class Swimmer:
    """Swimmer class for magnetoaerotactic Brownian swimmers"""

    def __init__(
        self, v_self, gamma_t, gamma_r, T, Teff, dt,
        M_mag, B, F_ext, mean_run, mean_tumble,
        initial_state,  # int8: 1=run, 0=tumble
        box_size,       # np.ndarray(3,)
        random_position,  # bool
        random_orientation,  # bool
        kindofchemotaxis,  # int: 0 band, 1 attractant, 2 repellent, else off
        grad_ref,           # float: reference gradient magnitude (same units as |∇C|)
        C_star,
        strategy,
        prev_state,              
    ):
        # physics
        self.v_self = v_self
        self.base_v_self = v_self
        self.gamma_t = gamma_t
        self.gamma_r = gamma_r
        self.T = T
        self.Teff = Teff
        self.dt = dt

        # magnetotaxis
        self.M_mag = M_mag
        self.B = B
        self.F_ext = F_ext

        # timing parameters
        self.mean_run = mean_run
        self.mean_tumble = mean_tumble

        # state
        self.status = initial_state
        self.strategy = strategy
        self.prev_state = prev_state

        # domain + randomness flags
        self.box_size = box_size
        self.random_position = random_position
        self.random_orientation = random_orientation
        self.kindofchemotaxis = kindofchemotaxis
        self.grad_ref = grad_ref
        self.C_star = C_star

        # initialize arrays inside nopython
        self.position = self._init_position()
        self.orientation = self._init_orientation()

        # timers and next interval
        self.time_in_state = 0.0
        if self.status == 1:
            self.next_interval = np.random.exponential(self.mean_run)
        else:
            self.next_interval = np.random.exponential(self.mean_tumble)
        
        # binding/region state
        self.bound_state = 0.0
        self.regions_state = 0.0

    def _init_position(self):
        if self.random_position:
            return np.random.random(3).astype(np.float64) * self.box_size
        else:
            return np.array([0.001, 0.0003, 0.0019], dtype=np.float64)

    def _init_orientation(self):
        if self.random_orientation:
            v = np.array(
                [
                    np.random.normal(0.0, 1.0),
                    np.random.normal(0.0, 1.0),
                    np.random.normal(0.0, 1.0),
                ],
                dtype=np.float64,
            )
        else:
            v = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        norm = np.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
        return v / norm

    def _draw_interval(self, grad_C, conc_here):
        # Draw an exponential interval for the UPCOMING state based on chemotaxis mode
        t1_0 = self.mean_run
        scalar = self.orientation[0] * grad_C[0] + self.orientation[1] * grad_C[1] + self.orientation[2] * grad_C[2]
        mode = self.kindofchemotaxis
        if mode == 0:
            # chemoattractant
            if conc_here <= self.C_star:
                if scalar <= 0.0:
                    tau = t1_0
                #elif scalar <= self.grad_ref:
                #    tau = t1_0 * (1 + scalar / self.grad_ref)
                else:
                    tau = 2 * t1_0
            else:
                # chemorepellent
                if scalar >= 0.0:
                    tau = t1_0
                #elif scalar > -self.grad_ref: 
                #    tau = t1_0 * (1 - scalar / self.grad_ref)
                else:
                    tau = 2 * t1_0
        else:
            tau = t1_0
        return np.random.exponential(tau)

    def step(self, grad_C, conc_here):
        # Wiener increments
        sqrt_dt = np.sqrt(self.dt)
        dW_t = np.array(
            [
                np.random.normal(0.0, 1.0) * sqrt_dt,
                np.random.normal(0.0, 1.0) * sqrt_dt,
                np.random.normal(0.0, 1.0) * sqrt_dt,
            ],
            dtype=np.float64,
        )
        dW_r = np.array(
            [
                np.random.normal(0.0, 1.0) * sqrt_dt,
                np.random.normal(0.0, 1.0) * sqrt_dt,
                np.random.normal(0.0, 1.0) * sqrt_dt,
            ],
            dtype=np.float64,
        )

        if self.bound_state == 0.0:
            self.v_self = self.base_v_self
        else:
            self.v_self = 0.0

        # 1) position update
        # Include translational diffusion and external force in BOTH states
        # self-propulsion drift only during run (status==1)
        # reverse direction during reverse (status==2)
        if self.status != 0:
             drift = (self.v_self * self.orientation * self.dt)
        else:
            drift = np.zeros(3, dtype=np.float64)
        
        force_drift = (self.F_ext / self.gamma_t) * self.dt
        diffusion = np.sqrt(2 * kB * self.T / self.gamma_t) * dW_t
        dx = drift + force_drift + diffusion

        x = self.position
        Lx, Ly, Lz = self.box_size[0], self.box_size[1], self.box_size[2]

        # reflect on each violated face 
        x_new = x + dx

        # +X (right wall)
        if x_new[0] > Lx:
            n = np.array([-1.0, 0.0, 0.0])
            dx = dx - 2.0 * np.dot(dx, n) * n
            x_new = x + dx

        # -X (left wall)
        if x_new[0] < 0.0:
            n = np.array([+1.0, 0.0, 0.0])
            dx = dx - 2.0 * np.dot(dx, n) * n
            x_new = x + dx

        # +Y (top)
        if x_new[1] > Ly:
            n = np.array([0.0, -1.0, 0.0])
            dx = dx - 2.0 * np.dot(dx, n) * n
            x_new = x + dx

        # -Y (bottom)
        if x_new[1] < 0.0:
            n = np.array([0.0, +1.0, 0.0])
            dx = dx - 2.0 * np.dot(dx, n) * n
            x_new = x + dx

        # +Z (front)
        if x_new[2] > Lz:
            n = np.array([0.0, 0.0, -1.0])
            dx = dx - 2.0 * np.dot(dx, n) * n
            x_new = x + dx

        # -Z (back)
        if x_new[2] < 0.0:
            n = np.array([0.0, 0.0, +1.0])
            dx = dx - 2.0 * np.dot(dx, n) * n
            x_new = x + dx

        # clamp (safety against tiny overshoot)
        x_new[0] = (
            0.0 if x_new[0] < 0.0 else (Lx if x_new[0] > Lx else x_new[0])
        )
        x_new[1] = (
            0.0 if x_new[1] < 0.0 else (Ly if x_new[1] > Ly else x_new[1])
        )
        x_new[2] = (
            0.0 if x_new[2] < 0.0 else (Lz if x_new[2] > Lz else x_new[2])
        )

        self.position = x_new

        # 2) orientation update
        torque = np.cross(self.M_mag * self.orientation, self.B)
        drift_rot = (torque / self.gamma_r) * self.dt

        Tn = self.Teff if self.status == 0 else self.T
        noise_rot = np.sqrt(2 * kB * Tn / self.gamma_r) * dW_r
        Phi = drift_rot + noise_rot

        self.orientation += np.cross(Phi, self.orientation)
        norm = np.sqrt(
            self.orientation[0] * self.orientation[0] +
            self.orientation[1] * self.orientation[1] +
            self.orientation[2] * self.orientation[2]
        )
        self.orientation /= norm

        # 3) state switch via pre-drawn exponential intervals
        
        self.time_in_state += self.dt

        # run-reverse strategy
        if self.strategy == 1:
            # exiting RUN or REVERSE, immediately enter PAUSE/TUMBLE
            if self.time_in_state >= self.next_interval and self.status != 0:
                # flip state
                self.prev_state = self.status
                self.status = 0 
                self.time_in_state = 0.0
                self.next_interval = np.random.exponential(self.mean_tumble)

            elif self.time_in_state >= self.next_interval and self.prev_state == 1:
                self.status = 2  # enter REVERSE
                self.orientation = -self.orientation  # reverse direction
                self.time_in_state = 0.0
                self.next_interval = self._draw_interval(grad_C, conc_here)
                
            elif self.time_in_state >= self.next_interval and self.prev_state == 2:
                self.status = 1  # enter RUN
                self.orientation = -self.orientation  # reverse direction
                self.time_in_state = 0.0
                self.next_interval = self._draw_interval(grad_C, conc_here)
        
        # run-tumble strategy
        else:
            if self.time_in_state >= self.next_interval:
                # flip state
                self.status = 0 if self.status == 1 else 1
                self.time_in_state = 0.0
                if self.status == 1:
                    # entering RUN: draw based on chemotaxis rules
                    self.next_interval = self._draw_interval(grad_C, conc_here)
                else:
                    # entering TUMBLE: draw from mean_tumble
                    self.next_interval = np.random.exponential(self.mean_tumble)
