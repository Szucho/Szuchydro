import numpy as np
import os
import struct

def read_meta(outdir):
    meta = {}
    with open(os.path.join(outdir, "meta.txt")) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split()
            if len(parts) == 2:
                meta[parts[0]] = float(parts[1])
    return meta

def read_grid(path):
    frames = []
    block_fmt = "<i q q d d d d Q Q"  # int32, int64*2, double*4, uint64*2
    block_size = struct.calcsize(block_fmt)
    
    with open(path, 'rb') as f:
        while True:
            t_buf = f.read(8)
            if not t_buf:
                break
            t = np.frombuffer(t_buf, dtype=np.float64)[0]
            
            nb_buf = f.read(8)
            if len(nb_buf) < 8:
                break
            nblocks = int(np.frombuffer(nb_buf, dtype=np.uint64)[0])
            
            frame_blocks = []
            for _ in range(nblocks):
                bdata = f.read(block_size)
                if len(bdata) < block_size:
                    break
                level, lx1, lx2, x1min, x1max, x2min, x2max, nx1, nx2 = struct.unpack(block_fmt, bdata)
                frame_blocks.append(dict(
                    level=int(level), lx1=int(lx1), lx2=int(lx2),
                    x1min=x1min, x1max=x1max, x2min=x2min, x2max=x2max,
                    nx1=int(nx1), nx2=int(nx2)
                ))
            
            block_arrays = []
            complete = True
            for blk in frame_blocks:
                n_cells = blk['nx1'] * blk['nx2'] * 4  # NVAR = 4
                raw = f.read(n_cells * 8)
                if len(raw) != n_cells * 8:
                    complete = False
                    break
                block_arrays.append(np.frombuffer(raw, dtype=np.float64).reshape(blk['nx1'], blk['nx2'], 4))
            
            if not complete:
                break
            frames.append((t, block_arrays, frame_blocks))
            
    return frames

def evaluate_boundary_conservation(outdir):
    meta = read_meta(outdir)
    xmin, xmax = meta['xmin'], meta['xmax']
    ymin, ymax = meta['ymin'], meta['ymax']
    
    grid_path = os.path.join(outdir, "grid.bin")
    if not os.path.exists(grid_path):
        print(f"Error: {grid_path} not found.")
        return

    frames = read_grid(grid_path)
    if not frames:
        print("No frames found to evaluate.")
        return

    def compute_global_integrals_and_boundary_flux(block_arrays, frame_blocks, dt_approx):
        total_mass = 0.0
        total_energy = 0.0
        boundary_mass_flux = 0.0
        boundary_energy_flux = 0.0
        
        for data, blk in zip(block_arrays, frame_blocks):
            dx = (blk['x1max'] - blk['x1min']) / blk['nx1']
            dy = (blk['x2max'] - blk['x2min']) / blk['nx2']
            cell_volume = dx * dy
            
            rho = data[:, :, 0]
            mx = data[:, :, 1] # x-momentum (rho * u)
            E = data[:, :, 3]
            
            # Integrate global totals
            total_mass += np.sum(rho) * cell_volume
            total_energy += np.sum(E) * cell_volume
            
            # Check if block touches physical x boundaries (xmin or xmax)
            # and accumulate outward mass/energy fluxes through face cells
            if abs(blk['x1min'] - xmin) < 1e-12:
                # Left boundary face cells (i = 0)
                u_face = mx[0, :] / rho[0, :]
                face_area = dy
                # Mass flux out: rho * u * area (negative u means leaving xmin boundary)
                boundary_mass_flux += np.sum(rho[0, :] * u_face) * face_area * dt_approx
                boundary_energy_flux += np.sum(E[0, :] * u_face) * face_area * dt_approx
                
            if abs(blk['x1max'] - xmax) < 1e-12:
                # Right boundary face cells (i = -1)
                u_face = mx[-1, :] / rho[-1, :]
                face_area = dy
                # Mass flux out: rho * u * area (positive u means leaving xmax boundary)
                boundary_mass_flux += np.sum(rho[-1, :] * u_face) * face_area * dt_approx
                boundary_energy_flux += np.sum(E[-1, :] * u_face) * face_area * dt_approx
                
        return total_mass, total_energy, boundary_mass_flux, boundary_energy_flux

    # t = 0 state
    t0, init_arrays, init_blocks = frames[0]
    mass_0, energy_0, _, _ = compute_global_integrals_and_boundary_flux(init_arrays, init_blocks, 0.0)
    
    # t = tf state (final frame)
    tf, final_arrays, final_blocks = frames[-1]
    
    # Integrate boundary fluxes iteratively across frame intervals
    cum_delta_mass_phys = 0.0
    cum_delta_energy_phys = 0.0
    
    for i in range(1, len(frames)):
        dt = frames[i][0] - frames[i-1][0]
        _, _, m_flux, e_flux = compute_global_integrals_and_boundary_flux(frames[i][1], frames[i][2], dt)
        cum_delta_mass_phys += m_flux
        cum_delta_energy_phys += e_flux

    mass_tf, energy_tf, _, _ = compute_global_integrals_and_boundary_flux(final_arrays, final_blocks, 0.0)
    
    # Actual domain change: M(t0) - M(tf) should match what left via boundaries
    mass_drop = mass_0 - mass_tf
    energy_drop = energy_0 - energy_tf
    
    print(f"\n==========================================================================")
    print(f"             PHYSICAL BOUNDARY CONSERVATION REPORT: {outdir}")
    print(f"==========================================================================")
    print(f"Initial Time t_0     = {t0:.4f}")
    print(f"Final Time   t_f     = {tf:.4f}\n")
    
    print(f"  M(t=0)             = {mass_0:.12e}")
    print(f"  E(t=0)             = {energy_0:.12e}\n")
    
    print(f"  M(t=t_f)           = {mass_tf:.12e}")
    print(f"  E(t=t_f)           = {energy_tf:.12e}\n")
    
    print(f"  |M(t0) - M(tf)|    = {abs(mass_drop):.12e}")
    print(f"  Delta M_phys(tf)   = {abs(cum_delta_mass_phys):.12e}")
    print(f"  Mass Balance Err   = {abs(abs(mass_drop) - abs(cum_delta_mass_phys)) / max(mass_0, 1e-14):.4e}\n")
    
    print(f"  |E(t0) - E(tf)|    = {abs(energy_drop):.12e}")
    print(f"  Delta E_phys(tf)   = {abs(cum_delta_energy_phys):.12e}")
    print(f"  Energy Balance Err = {abs(abs(energy_drop) - abs(cum_delta_energy_phys)) / max(energy_0, 1e-14):.4e}")
    print(f"==========================================================================\n")

if __name__ == "__main__":
    outdir = input("Output directory: ").strip()
    evaluate_boundary_conservation(outdir)
