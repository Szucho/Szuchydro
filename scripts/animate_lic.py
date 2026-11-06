#!/usr/bin/env python3
"""
Fig. 1-style simplified animation (Huang & Bai 2026) for Szuchydro grid.bin output:
line integral convolution (LIC) of the poloidal velocity, coloured by |v_p|/c_s
(or by density, or by azimuthal vorticity), x1 = R and x2 = z.

Only needs numpy, scipy, matplotlib and pillow (ffmpeg optional, for .mp4).

Examples
  # annotated still of the last frame (axes + colorbar) to check the look
  python3 animate_lic.py disk_out --still -1 --annotate -o fig1_still.png

  # clean background animation, skipping the spin-up phase
  python3 animate_lic.py disk_out --tmin 80 --fps 20 -o disk_background.gif disk_background.mp4

  # vorticity version (red/blue = sense of rotation of the eddies)
  python3 animate_lic.py disk_out --field vort --tmin 80 -o disk_vorticity.mp4
"""
import argparse, os, struct, sys
import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

NVAR = 4
HDR, BLK = "<dQ", "<iqqddddQQ"   #frame header (t, nblocks), block header
HS, BS = struct.calcsize(HDR), struct.calcsize(BLK)


#I/O
def read_meta(outdir):
    meta = {}
    with open(os.path.join(outdir, "meta.txt")) as f:
        for line in f:
            k, v = line.split()
            meta[k] = float(v)
    return meta


def index_frames(path):
    """Byte offset and time of every frame, without loading the data."""
    idx = []
    with open(path, "rb") as f:
        while True:
            pos = f.tell()
            h = f.read(HS)
            if len(h) < HS:
                break
            t, nb = struct.unpack(HDR, h)
            ncell = 0
            for _ in range(nb):
                b = struct.unpack(BLK, f.read(BS))
                ncell += b[7] * b[8]
            f.seek(ncell * NVAR * 8, 1)
            idx.append((pos, t))
    return idx


def load_frame(path, pos):
    """Stitch all blocks of one frame onto the finest uniform canvas.
    Returns t, x (cell centres), y, Q with shape (NVAR, nx, ny)."""
    with open(path, "rb") as f:
        f.seek(pos)
        t, nb = struct.unpack(HDR, f.read(HS))
        blocks = [struct.unpack(BLK, f.read(BS)) for _ in range(nb)]
        arrs = [np.frombuffer(f.read(b[7] * b[8] * NVAR * 8), dtype="<f8").reshape(b[7], b[8], NVAR)
                for b in blocks]
    dx = min((b[4] - b[3]) / b[7] for b in blocks)
    dy = min((b[6] - b[5]) / b[8] for b in blocks)
    x0, x1 = min(b[3] for b in blocks), max(b[4] for b in blocks)
    y0, y1 = min(b[5] for b in blocks), max(b[6] for b in blocks)
    nx, ny = int(round((x1 - x0) / dx)), int(round((y1 - y0) / dy))
    Q = np.empty((NVAR, nx, ny))
    for b, a in zip(blocks, arrs):
        r = int(round(((b[4] - b[3]) / b[7]) / dx))  #refinement ratio to finest level
        i0, j0 = int(round((b[3] - x0) / dx)), int(round((b[5] - y0) / dy))
        up = np.repeat(np.repeat(a, r, 0), r, 1) if r > 1 else a
        Q[:, i0:i0 + up.shape[0], j0:j0 + up.shape[1]] = np.moveaxis(up, 2, 0)
    x = x0 + (np.arange(nx) + 0.5) * dx
    y = y0 + (np.arange(ny) + 0.5) * dy
    return t, x, y, Q


def primitives(Q, gamma):
    rho = Q[0]
    u, v = Q[1] / rho, Q[2] / rho
    p = (gamma - 1.0) * (Q[3] - 0.5 * rho * (u * u + v * v))
    cs = np.sqrt(np.maximum(gamma * p / rho, 1e-300))
    return rho, u, v, p, cs


#LIC
def vorticity_over_omega(u, v, x, y):
    """Azimuthal vorticity omega_phi = dv_R/dz - dv_z/dR, divided by Omega_K = R^-1.5
    so eddies at all radii are comparable (raw omega scales with the orbital frequency)."""
    dudz = np.gradient(u, y, axis=1)
    dvdR = np.gradient(v, x, axis=0)
    return (dudz - dvdR) * x[:, None] ** 1.5


def resample(field, x, y, X, Y):
    """Bilinear resample of a (nx, ny) cell-centred field onto pixel coords X, Y."""
    fi = (X - x[0]) / (x[1] - x[0])
    fj = (Y - y[0]) / (y[1] - y[0])
    return map_coordinates(field, [fi, fj], order=1, mode="nearest")


def lic(vx, vy, noise, length=30, step=0.6):
    """Vectorised line integral convolution on an image grid (rows = z, cols = R).
    vx, vy: velocity in pixel units (any magnitude), noise: same shape.
    Integrates `length` midpoint (RK2) steps forward and backward and averages the
    noise along the streamline with a Hann kernel. Velocity lookups are nearest-pixel
    (the flow is smooth on pixel scales), only the noise is sampled bilinearly."""
    ny, nx = noise.shape
    mag = np.hypot(vx, vy) + 1e-30
    ux, uy = (vx / mag).ravel(), (vy / mag).ravel()

    def near(fr, fc):  #nearest-pixel flat index
        return (np.clip(np.rint(fr), 0, ny - 1).astype(np.int64) * nx
                + np.clip(np.rint(fc), 0, nx - 1).astype(np.int64))

    nz = noise.ravel()

    def bilin(fr, fc):
        r0 = np.floor(fr).astype(np.int64); c0 = np.floor(fc).astype(np.int64)
        r0 = np.clip(r0, 0, ny - 2); c0 = np.clip(c0, 0, nx - 2)
        wr, wc = fr - r0, fc - c0
        i = r0 * nx + c0
        return ((1 - wr) * ((1 - wc) * nz[i] + wc * nz[i + 1])
                + wr * ((1 - wc) * nz[i + nx] + wc * nz[i + nx + 1]))

    rows, cols = np.mgrid[0:ny, 0:nx]
    rows, cols = rows.ravel().astype(float), cols.ravel().astype(float)
    acc = nz.copy()
    wsum = 1.0
    w = 0.5 * (1 + np.cos(np.pi * np.arange(1, length + 1) / (length + 1)))  #Hann
    hs = 0.5 * step
    for sgn in (1.0, -1.0):
        r, c = rows.copy(), cols.copy()
        for k in range(length):
            i = near(r, c)
            j = near(r + sgn * hs * uy[i], c + sgn * hs * ux[i])
            r = np.clip(r + sgn * step * uy[j], 0, ny - 1)
            c = np.clip(c + sgn * step * ux[j], 0, nx - 1)
            acc += w[k] * bilin(r, c)
            wsum += w[k]
    return (acc / wsum).reshape(ny, nx)


#render
class Renderer:
    def __init__(self, x, y, width, args, gamma):
        self.args, self.gamma = args, gamma
        #crop (defaults to the full domain)
        self.xlo = args.xlim[0] if args.xlim else x[0] - 0.5 * (x[1] - x[0])
        self.xhi = args.xlim[1] if args.xlim else x[-1] + 0.5 * (x[1] - x[0])
        self.ylo = args.ylim[0] if args.ylim else y[0] - 0.5 * (y[1] - y[0])
        self.yhi = args.ylim[1] if args.ylim else y[-1] + 0.5 * (y[1] - y[0])
        self.W = width
        self.H = int(round(width * (self.yhi - self.ylo) / (self.xhi - self.xlo)))
        self.H += self.H % 2  #even size for video codecs
        px = (self.xhi - self.xlo) / self.W
        Xc = self.xlo + (np.arange(self.W) + 0.5) * px
        Yc = self.yhi - (np.arange(self.H) + 0.5) * (self.yhi - self.ylo) / self.H  #top row = +z
        self.X, self.Y = np.meshgrid(Xc, Yc)
        rng = np.random.default_rng(args.seed)
        noise = rng.random((self.H, self.W))
        #slightly blurred noise -> smoother, less sparkly streaks at video resolution
        self.noise = gaussian_filter(noise, args.noise_blur) if args.noise_blur > 0 else noise
        self.stretch = None  #fixed LIC contrast stretch, set on first frame (no flicker)
        if args.cmap is None:
            args.cmap = "RdBu_r" if args.field == "vort" else "managua_r"
        self.cmap = plt.get_cmap(args.cmap)
        self.vlim = None  #fixed symmetric vorticity range, set on first frame

    def frame(self, x, y, Q):
        a = self.args
        rho, u, v, p, cs = primitives(Q, self.gamma)
        U = resample(u, x, y, self.X, self.Y)
        V = resample(v, x, y, self.X, self.Y)
        if a.field == "vp":
            F = resample(np.hypot(u, v) / cs, x, y, self.X, self.Y)
            norm = LogNorm(a.vmin or 1e-2, a.vmax or 2.0, clip=True)
        elif a.field == "vort":
            F = resample(vorticity_over_omega(u, v, x, y), x, y, self.X, self.Y)
            if self.vlim is None:
                #range from the disk interior only (|z|/R < 0.25); otherwise the strong
                #disk-wind shear layers set the scale and the eddies look washed out
                disk = np.abs(self.Y / self.X) < 0.25
                self.vlim = a.vmax or float(np.percentile(np.abs(F[disk]), 98))
            norm = Normalize(-self.vlim, self.vlim, clip=True)
        else:  #density
            F = resample(rho, x, y, self.X, self.Y)
            norm = LogNorm(a.vmin or max(F.min(), F.max() * 1e-6), a.vmax or F.max(), clip=True)
        #image rows go downward, so flip the z velocity
        tex = lic(U, -V, self.noise, a.length, a.step)
        if self.stretch is None:
            lo, hi = np.percentile(tex, [1, 99])
            self.stretch = (lo, hi)
        lo, hi = self.stretch
        tex = np.clip((tex - lo) / (hi - lo), 0, 1)
        rgb = self.cmap(norm(F))[..., :3]
        shade = (1 - a.contrast) + a.contrast * tex  #LIC modulates brightness
        img = np.clip(rgb * shade[..., None], 0, 1)
        return img, norm

    def save_annotated(self, img, norm, t, path):
        a = self.args
        fig, ax = plt.subplots(figsize=(8, 8 * self.H / self.W + 0.6), dpi=150)
        ax.imshow(img, extent=[self.xlo, self.xhi, self.ylo, self.yhi], origin="upper",
                  interpolation="bilinear")
        ax.set_xlabel("R"); ax.set_ylabel("z")
        ax.set_title(f"t = {t:.1f}")
        sm = plt.cm.ScalarMappable(norm=norm, cmap=self.cmap)
        lbl = {"vp": r"$|v_\mathrm{p}|/c_\mathrm{s}$", "rho": r"$\rho$",
               "vort": r"$\omega_\phi/\Omega_\mathrm{K}$"}[a.field]
        fig.colorbar(sm, ax=ax, label=lbl, fraction=0.035, pad=0.02)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Fig.1-style LIC animation of Szuchydro output")
    ap.add_argument("outdir", help="simulation output dir (contains grid.bin, meta.txt)")
    ap.add_argument("-o", "--out", nargs="+", default=["disk_lic.gif"],
                    help="one or more of .gif / .mp4 (or one .png with --still)")
    ap.add_argument("--field", choices=["vp", "rho", "vort"], default="vp",
                    help="colour: |v_p|/c_s, density, or vorticity omega_phi/Omega_K")
    ap.add_argument("--cmap", default=None, help="default: managua_r (vp, rho), RdBu_r (vort)")
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None, help="for vort: symmetric limit (default: 98th pct inside the disk)")
    ap.add_argument("--width", type=int, default=960, help="output width in pixels")
    ap.add_argument("--xlim", type=float, nargs=2, default=None, help="crop in R")
    ap.add_argument("--ylim", type=float, nargs=2, default=None, help="crop in z")
    ap.add_argument("--tmin", type=float, default=0.0, help="skip frames before this time")
    ap.add_argument("--tmax", type=float, default=np.inf)
    ap.add_argument("--every", type=int, default=1, help="use every n-th frame")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--still", type=int, default=None, help="render only this frame index (e.g. -1)")
    ap.add_argument("--annotate", action="store_true", help="axes, time, colorbar (for checking)")
    ap.add_argument("--length", type=int, default=40, help="LIC half-length in steps")
    ap.add_argument("--step", type=float, default=0.6, help="LIC step in pixels")
    ap.add_argument("--contrast", type=float, default=0.75, help="0 = flat colour, 1 = full LIC")
    ap.add_argument("--noise-blur", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    path = os.path.join(a.outdir, "grid.bin")
    gamma = read_meta(a.outdir).get("gamma", 1.4)
    idx = index_frames(path)
    if not idx:
        sys.exit("no frames in " + path)
    if a.still is not None:
        sel = [idx[a.still]]
    else:
        sel = [f for f in idx if a.tmin <= f[1] <= a.tmax][::a.every]
    print(f"{len(idx)} frames in file, rendering {len(sel)}")

    t, x, y, Q = load_frame(path, sel[0][0])
    R = Renderer(x, y, a.width, a, gamma)

    if a.still is not None:
        img, norm = R.frame(x, y, Q)
        if a.annotate:
            R.save_annotated(img, norm, t, a.out[0])
        else:
            plt.imsave(a.out[0], img)
        print("wrote", a.out[0])
        return

    frames = []
    for n, (pos, _) in enumerate(sel):
        t, x, y, Q = load_frame(path, pos)
        img, norm = R.frame(x, y, Q)
        frames.append((img * 255).astype(np.uint8))
        print(f"\r  frame {n + 1}/{len(sel)}  t={t:.1f}", end="", flush=True)
    print()

    for out in a.out:
        write_movie(out, frames, a.fps, R.W, R.H)


def write_movie(out, frames, fps, W, H):
    if out.lower().endswith(".gif"):
        from PIL import Image
        ims = [Image.fromarray(f) for f in frames]
        #one shared adaptive palette -> no palette flicker between frames
        pal = ims[len(ims) // 2].quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        ims = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in ims]
        ims[0].save(out, save_all=True, append_images=ims[1:], duration=int(1000 / fps),
                    loop=0, optimize=False)
    else:  #mp4 via ffmpeg (matplotlib writer)
        from matplotlib.animation import FFMpegWriter
        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        im = ax.imshow(frames[0], interpolation="none")
        w = FFMpegWriter(fps=fps, codec="libx264", bitrate=-1,
                         extra_args=["-pix_fmt", "yuv420p", "-crf", "18"])
        with w.saving(fig, out, dpi=100):
            for f in frames:
                im.set_data(f); w.grab_frame()
        plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
