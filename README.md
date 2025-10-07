## pMagMove

Magnetoaerotactic swimmer simulation with oxygen diffusion and binding sinks.

### Quick start
- Install Python 3.10+ and dependencies:
  - numpy, numba
- Copy the example params and adjust as needed:
```bash
cp params.example.txt params.txt
```
- Run the simulation (from repo root):
```bash
cd src && python driver.py --params ../params.txt
```
The outputs will be written to the directory set by `out_dir` in your params (defaults to `./data_out`).


### Parameters file (params.txt)
Plain text, one `key=value` per line, comments start with `#`. See `params.example.txt` for a complete template. Common keys:
- Core
  - `n_swimmers`: number of agents (int)
  - `dt`: timestep in seconds (float)
  - `final_time`: total simulated time in seconds (float)
  - `save_stride`: save every N steps (int)
  - `out_prefix`: filename prefix (str)
  - `out_dir`: output directory (str)
- Swimmer physics
  - `v_self`, `gamma_t`, `gamma_r`, `T`, `Teff`, `M_mag`
  - `strategy`: 0=run-tumble, 1=run-reverse
  - `state`, `prev_state`: initial states (1=run)
  - `mean_run`, `mean_tumble`
  - `kindofchemotaxis`: 0=band, else off
  - `C_star`, `grad_ref`
  - `B`: magnetic field vector, e.g. `-35.36,-35.36,0.0`
  - `F_ext`: external force vector
  - `box_size`: domain size in meters, e.g. `0.002,0.0006,0.002`
  - `rand_pos_flag`, `rand_ori_flag`: true/false
- Environment (oxygen field)
  - `diffusion_coeff`, `consumption_rate`, `initial_conc`
  - `N_x`, `N_y`, `N_z`: grid resolution
  - Sink boxes (half-open indices): `imin1`, `imax1`, `jmin1`, `jmax1`, `kmin1`, `kmax1` and the same for `*2`
  - Robin/Dirichlet: `kappa_1`, `kappa_2`, `C_sink_1`, `C_sink_2`
  - Binding/unbinding: `binding_rate_1`, `binding_rate_2`, `unbinding_rate_1`, `unbinding_rate_2`
  - Boundaries: `Cb` (air/water Dirichlet at z-), `Ca` (Michaelis-Menten half-saturation)

Notes:
- Vectors are comma-separated without brackets.
- Booleans accept: true/false/1/0/yes/no.

### Outputs
Files written to `out_dir`:
- `{out_prefix}_C_hist.npy`: oxygen concentration snapshots `(n_saves, Ny, Nx, Nz)`
- `{out_prefix}_positions.npy`: swimmer positions `(n_saves, N, 3)`
  - You can compute MSD using `helper.compute_msd_fft`.

### Development
- Simulation loop and CLI: `src/driver.py`
- Core types: `src/magmove.py`
- Helpers: `src/helper.py`