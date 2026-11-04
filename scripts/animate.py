import numpy as np
import os
import struct
import matplotlib.pyplot as plt
import matplotlib.animation as animation


SAVE_OUTPUTS = True
FPS = 30


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
    """Returns frames: list of (t, block_arrays, frame_blocks)"""
    frames = []
    block_fmt = "<i q q d d d d Q Q"  # int32, int64*2, double*4, uint64*2
    block_size = struct.calcsize(block_fmt)
    
    with open(path, 'rb') as f:
        while True:
            # 1. Read timestamp (double)
            t_buf = f.read(8)
            if not t_buf:
                break
            t = np.frombuffer(t_buf, dtype=np.float64)[0]
            
            # 2. Read nblocks for this specific frame (uint64)
            nb_buf = f.read(8)
            if len(nb_buf) < 8:
                break
            nblocks = int(np.frombuffer(nb_buf, dtype=np.uint64)[0])
            
            # 3. Read block geometries for this frame
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
            
            # 4. Read cell data payload for this frame's active blocks
            block_arrays = []
            complete = True
            for blk in frame_blocks:
                n_cells = blk['nx1'] * blk['nx2'] * 4  # NVAR = 4
                raw = f.read(n_cells * 8)              # 8 bytes per double
                if len(raw) != n_cells * 8:
                    complete = False
                    break
                block_arrays.append(np.frombuffer(raw, dtype=np.float64).reshape(blk['nx1'], blk['nx2'], 4))
            
            if not complete:
                break
            frames.append((t, block_arrays, frame_blocks))
            
    return frames


def get_global_canvas_resolution(frames, meta):
    global_max_level = 0
    for _, _, frame_blocks in frames:
        ml = max(b['level'] for b in frame_blocks)
        if ml > global_max_level:
            global_max_level = ml
            
    # frames[0] is (t, block_arrays, frame_blocks). Index [2] is frame_blocks!
    base_blk = frames[0][2][0]
    nx1_block = base_blk['nx1']
    nx2_block = base_blk['nx2']
    
    xmin, xmax = meta['xmin'], meta['xmax']
    ymin, ymax = meta['ymin'], meta['ymax']
    
    # Locate any frame with level 0 blocks to get root block counts safely
    lvl0_any = None
    for _, _, frame_blocks in frames:
        lvl0 = [b for b in frame_blocks if b['level'] == 0]
        if lvl0:
            lvl0_any = lvl0
            break
    if lvl0_any is None:
        lvl0_any = frames[0][2]
        
    nblk1 = max(b['lx1'] for b in lvl0_any) + 1
    nblk2 = max(b['lx2'] for b in lvl0_any) + 1
    
    nx_c = nblk1 * nx1_block * (1 << global_max_level)
    ny_c = nblk2 * nx2_block * (1 << global_max_level)
    return nx_c, ny_c, global_max_level


def stitch_frame_by_coords(block_arrays, frame_blocks, xmin, xmax, ymin, ymax, nx_c, ny_c):
    canvas = np.empty((nx_c, ny_c, 4))
    canvas[:] = 0.0
    
    for data, b in zip(block_arrays, frame_blocks):
        i0 = int(round((b['x1min'] - xmin) / (xmax - xmin) * nx_c))
        i1 = int(round((b['x1max'] - xmin) / (xmax - xmin) * nx_c))
        j0 = int(round((b['x2min'] - ymin) / (ymax - ymin) * ny_c))
        j1 = int(round((b['x2max'] - ymin) / (ymax - ymin) * ny_c))
        
        i0 = max(0, min(nx_c, i0))
        i1 = max(0, min(nx_c, i1))
        j0 = max(0, min(ny_c, j0))
        j1 = max(0, min(ny_c, j1))
        
        if i1 > i0 and j1 > j0:
            target_ni = i1 - i0
            target_nj = j1 - j0
            fi = max(1, target_ni // b['nx1'])
            fj = max(1, target_nj // b['nx2'])
            
            expanded = np.repeat(np.repeat(data, fi, axis=0), fj, axis=1)
            
            ei = min(i0 + expanded.shape[0], nx_c)
            ej = min(j0 + expanded.shape[1], ny_c)
            canvas[i0:ei, j0:ej, :] = expanded[:ei-i0, :ej-j0, :]
            
    return canvas


def compute_primitives(data, gamma=1.4):
    rho = data[:, :, 0]
    u   = data[:, :, 1] / rho
    v   = data[:, :, 2] / rho
    E   = data[:, :, 3]
    p   = (gamma - 1.0) * (E - 0.5 * rho * (u**2 + v**2))
    return rho, u, v, p


outdir = input("Output directory: ").strip()
meta = read_meta(outdir)

xmin, xmax = meta['xmin'], meta['xmax']
ymin, ymax = meta['ymin'], meta['ymax']
gamma      = meta['gamma']

frames = read_grid(os.path.join(outdir, "grid.bin"))
if not frames:
    raise RuntimeError("grid.bin is empty or could not be read")

nx_c, ny_c, global_max_level = get_global_canvas_resolution(frames, meta)

times = []
prims = []
for t, block_arrays, frame_blocks in frames:
    canvas_data = stitch_frame_by_coords(block_arrays, frame_blocks, xmin, xmax, ymin, ymax, nx_c, ny_c)
    times.append(t)
    prims.append(compute_primitives(canvas_data, gamma))

# Stable global canvas dimensions
dx = (xmax - xmin) / nx_c
dy = (ymax - ymin) / ny_c
x_centers = xmin + (np.arange(nx_c) + 0.5) * dx
j_mid = ny_c // 2

x_edges = np.linspace(xmin, xmax, nx_c + 1)
y_edges = np.linspace(ymin, ymax, ny_c + 1)
X, Y = np.meshgrid(x_edges, y_edges)

# 1D slice animation
fig_1d, ax1 = plt.subplots(figsize=(9, 4.5))

line_rho, = ax1.plot([], [], 'b-',  lw=1.8, label=r'$\rho(x, y=0.5)$')
line_u,   = ax1.plot([], [], 'r--', lw=1.8, label=r'$u(x, y=0.5)$')
line_p,   = ax1.plot([], [], 'g-.', lw=1.8, label=r'$p(x, y=0.5)$')

ax1.set_xlim(xmin, xmax)
ax1.set_ylim(-0.05, 1.15)
ax1.set_xlabel('x', fontsize=11)
ax1.set_ylabel('Primitive vals', fontsize=11)
ax1.grid(True, linestyle=":", alpha=0.6)
ax1.legend(loc='center right', fontsize=11)

def update_1d(n):
    rho, u, _, p = prims[n]
    line_rho.set_data(x_centers, rho[:, j_mid])
    line_u.set_data(x_centers, u[:, j_mid])
    line_p.set_data(x_centers, p[:, j_mid])
    ax1.set_title(f"shock tube primitive vals at t~{times[n]:.4f}", fontsize=12)
    return line_rho, line_u, line_p

ani_1d = animation.FuncAnimation(fig_1d, update_1d, frames=len(prims), interval=40, blit=False)

if SAVE_OUTPUTS:
    gif_1d_path = os.path.join(outdir, "shock_tube_1d_animation.gif")
    ani_1d.save(gif_1d_path, writer='pillow', fps=FPS)
    print(f"Saved animation: {gif_1d_path}")

# Snapshot at final time
rho_f, u_f, _, p_f = prims[-1]
fig_snap, ax_snap = plt.subplots(figsize=(9, 4.5))
ax_snap.plot(x_centers, rho_f[:, j_mid], 'b-',  lw=1.8, label=r'$\rho(x, y=0.5)$')
ax_snap.plot(x_centers, u_f[:, j_mid],   'r--', lw=1.8, label=r'$u(x, y=0.5)$')
ax_snap.plot(x_centers, p_f[:, j_mid],   'g-.', lw=1.8, label=r'$p(x, y=0.5)$')

ax_snap.set_xlim(xmin, xmax)
ax_snap.set_ylim(-0.05, 1.15)
ax_snap.set_xlabel('x', fontsize=11)
ax_snap.set_ylabel(f'Primitive vals HLLC t={times[-1]:.4f}', fontsize=11)
ax_snap.grid(True, linestyle=":", alpha=0.6)
ax_snap.legend(loc='center right', fontsize=11)
ax_snap.set_title(f"shock tube snippet primitive vals at t~{times[-1]:.4f}", fontsize=12)

fig_snap.tight_layout()
if SAVE_OUTPUTS:
    snap_path = os.path.join(outdir, "shock_tube_1d_snippet.png")
    fig_snap.savefig(snap_path, dpi=300)
    print(f"Saved figure: {snap_path}")

# 2D multi-panel animation of primitive vals
fig_2d, axes = plt.subplots(2, 2, figsize=(10, 5))
fig_2d.subplots_adjust(hspace=0.4, wspace=0.3)

titles = [r"$\rho$", r"$u$", r"$v$", r"$p$"]
cmaps  = ["viridis", "coolwarm", "coolwarm", "plasma"]

limits = [
    (min(p[0].min() for p in prims), max(p[0].max() for p in prims)),
    (min(p[1].min() for p in prims), max(p[1].max() for p in prims)),
    (min(p[2].min() for p in prims), max(p[2].max() for p in prims)),
    (min(p[3].min() for p in prims), max(p[3].max() for p in prims))
]

meshes = []
for ax, field, title, cmap, lim in zip(axes.flat, prims[0], titles, cmaps, limits):
    mesh = ax.pcolormesh(X, Y, field.T, cmap=cmap, vmin=lim[0], vmax=lim[1], shading='flat')
    ax.set_aspect('auto')
    fig_2d.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    meshes.append(mesh)

suptitle = fig_2d.suptitle(f"t = {times[0]:.4f}", fontsize=12, fontweight='bold')

def update_2d(n):
    rho, u, v, p = prims[n]
    for mesh, field in zip(meshes, [rho, u, v, p]):
        mesh.set_array(field.T.ravel())
    suptitle.set_text(f"t = {times[n]:.4f}")
    return meshes + [suptitle]

ani_2d = animation.FuncAnimation(fig_2d, update_2d, frames=len(prims), interval=40, blit=False)

if SAVE_OUTPUTS:
    gif_2d_path = os.path.join(outdir, "shock_tube_2d_animation.gif")
    ani_2d.save(gif_2d_path, writer='pillow', fps=FPS)
    print(f"Saved animation: {gif_2d_path}")

plt.show()
