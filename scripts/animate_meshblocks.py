import struct
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import os
import sys

def read_grid_bin_animation(grid_bin_path):
    frames = []
    
    frame_header_fmt = "<d Q"  # double t, uint64_t nblocks
    frame_header_size = struct.calcsize(frame_header_fmt)
    
    block_fmt = "<i q q d d d d Q Q"  # int32, int64*2, double*4, uint64*2
    block_size = struct.calcsize(block_fmt)
    
    if not os.path.exists(grid_bin_path):
        raise FileNotFoundError(f"Error: {grid_bin_path} not found!")

    with open(grid_bin_path, 'rb') as f:
        while True:
            header_data = f.read(frame_header_size)
            if len(header_data) < frame_header_size:
                break  # End of file
            
            t, nblocks = struct.unpack(frame_header_fmt, header_data)
            
            blocks = []
            total_cells_per_block = 0
            for _ in range(nblocks):
                block_data = f.read(block_size)
                if len(block_data) < block_size:
                    break
                
                level, lx1, lx2, x1min, x1max, x2min, x2max, nx1, nx2 = struct.unpack(block_fmt, block_data)
                total_cells_per_block = nx1 * nx2
                
                blocks.append({
                    'level': level,
                    'lx1': lx1,
                    'lx2': lx2,
                    'x1min': x1min,
                    'x1max': x1max,
                    'x2min': x2min,
                    'x2max': x2max,
                    'nx1': nx1,
                    'nx2': nx2
                })
            
            # Skip past the raw cell data payload for this frame so we can read the next frame header
            NVAR = 4  # Adjust if your NVAR is different
            cell_data_bytes_per_block = total_cells_per_block * NVAR * 8  # 8 bytes per double
            total_payload_bytes = nblocks * cell_data_bytes_per_block
            f.seek(total_payload_bytes, 1)  # Seek forward relative to current position
            
            if blocks:
                frames.append({'time': t, 'blocks': blocks})
                
    return frames

def animate_mesh_blocks(grid_bin_path, output_mp4="mesh_evolution.mp4"):
    frames = read_grid_bin_animation(grid_bin_path)
    if not frames:
        print("No animation frames found in grid.bin!")
        return

    fig, ax = plt.subplots(figsize=(10, 7), facecolor='white')
    ax.set_facecolor('white')

    all_blocks = [b for f in frames for b in f['blocks']]
    xmin = min(b['x1min'] for b in all_blocks)
    xmax = max(b['x1max'] for b in all_blocks)
    ymin = min(b['x2min'] for b in all_blocks)
    ymax = max(b['x2max'] for b in all_blocks)

    margin_x = (xmax - xmin) * 0.03
    margin_y = (ymax - ymin) * 0.03
    ax.set_xlim(xmin - margin_x, xmax + margin_x)
    ax.set_ylim(ymin - margin_y, ymax + margin_y)

    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.3, color='gray')

    title_text = ax.set_title('', fontsize=14)

    def update(frame_data):
        # Remove existing patch rectangles safely in modern Matplotlib
        for p in list(ax.patches):
            p.remove()
            
        t = frame_data['time']
        blocks = frame_data['blocks']

        title_text.set_text(f'Dynamic AMR Mesh Evolution (t = {t:.4f}) | Blocks: {len(blocks)}')

        for b in blocks:
            x0 = b['x1min']
            y0 = b['x2min']
            width = b['x1max'] - b['x1min']
            height = b['x2max'] - b['x2min']
            lvl = b['level']

            rect = patches.Rectangle(
                (x0, y0), width, height,
                linewidth=0.8 + 0.3 * lvl,
                edgecolor='black',
                facecolor='none',
                zorder=lvl + 2
            )
            ax.add_patch(rect)
        return ax.patches, title_text

    print(f"Animating {len(frames)} frames...")
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=100, blit=False)

    try:
        ani.save(output_mp4, writer='ffmpeg', dpi=200)
        print(f"Animation successfully saved to: {output_mp4}")
    except Exception as e:
        print(f"FFmpeg writer not found or failed ({e}). Displaying interactively instead...")
        plt.show()

if __name__ == "__main__":
    grid_file = "Sod_shock_tube/grid.bin"
    if len(sys.argv) > 1:
        grid_file = sys.argv[1]

    animate_mesh_blocks(grid_file)
