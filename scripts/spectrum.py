#!/usr/bin/env python3
"""
Kinetic energy spectra of Szuchydro output (2D, periodic, square box).

  E(k): shell-averaged spectrum of the velocity (or of sqrt(rho) v with --weight sqrtrho), in shells
        of integer |k| (units of 2 pi / L), normalized by Parseval so that sum_k E(k) = 1/2 <|v|^2>
  Helmholtz split into solenoidal (vortical) and compressive parts, E = E_sol + E_comp
  averaged over all snapshots with tmin <= t <= tmax, power-law fit over a chosen k range

2D turbulence is NOT Kolmogorov 3D turbulence (Kraichnan 1967, Leith 1968, Batchelor 1969):
  forward enstrophy cascade  E ~ k^-3    for k > k_forcing (forcing at large scales)
  inverse energy cascade     E ~ k^-5/3  for k < k_forcing (forcing at small scales + large-scale drag)
  shock-dominated flows      E ~ k^-2    (Burgers-like)

Examples:
  python3 scripts/spectrum.py turb_out --tmin 5 --fit 4 30 -o spectrum.png
  python3 scripts/spectrum.py turb_out --tmin 5 --fit 4 30 --component sol   # 2D cascades: solenoidal part
  python3 scripts/spectrum.py --selftest          # checks the script on fields with known spectra

Needs numpy; matplotlib only for the plot.
"""
import argparse, os, struct, sys
import numpy as np

NVAR = 4
HDR, BLK = "<dQ", "<iqqddddQQ"
HS, BS = struct.calcsize(HDR), struct.calcsize(BLK)


def read_frames(path, tmin=-np.inf, tmax=np.inf):
    """Snapshots with tmin <= t <= tmax as (t, x, y, Q), Q (NVAR, nx, ny) on the finest uniform grid."""
    out = []
    with open(path, "rb") as f:
        while True:
            h = f.read(HS)
            if len(h) < HS:
                break
            t, nb = struct.unpack(HDR, h)
            blocks = [struct.unpack(BLK, f.read(BS)) for _ in range(nb)]
            if not (tmin <= t <= tmax):
                f.seek(sum(b[7]*b[8] for b in blocks)*NVAR*8, 1)
                continue
            arrs = [np.frombuffer(f.read(b[7]*b[8]*NVAR*8), dtype="<f8").reshape(b[7], b[8], NVAR)
                    for b in blocks]
            dx = min((b[4] - b[3])/b[7] for b in blocks)
            dy = min((b[6] - b[5])/b[8] for b in blocks)
            x0, x1 = min(b[3] for b in blocks), max(b[4] for b in blocks)
            y0, y1 = min(b[5] for b in blocks), max(b[6] for b in blocks)
            nx, ny = int(round((x1 - x0)/dx)), int(round((y1 - y0)/dy))
            Q = np.empty((NVAR, nx, ny))
            for b, a in zip(blocks, arrs):
                r = int(round(((b[4] - b[3])/b[7])/dx))
                i0, j0 = int(round((b[3] - x0)/dx)), int(round((b[5] - y0)/dy))
                up = np.repeat(np.repeat(a, r, 0), r, 1) if r > 1 else a
                Q[:, i0:i0 + up.shape[0], j0:j0 + up.shape[1]] = np.moveaxis(up, 2, 0)
            out.append((t, x0 + (np.arange(nx) + 0.5)*dx, y0 + (np.arange(ny) + 0.5)*dy, Q))
    return out


def spectra(u, v):
    """Shell spectra of a periodic 2D vector field on an n x n grid.
    Returns k (integer shells 1..), E_total, E_solenoidal, E_compressive with
    sum(E_total) = 1/2 mean(|v - <v>|^2) exactly (Parseval)."""
    n = u.shape[0]
    if u.shape != (n, n):
        raise ValueError("spectra need a square grid")
    uk = np.fft.fft2(u - u.mean())/n**2
    vk = np.fft.fft2(v - v.mean())/n**2
    kx = np.fft.fftfreq(n, 1.0/n)[:, None]   #integer wavenumbers
    ky = np.fft.fftfreq(n, 1.0/n)[None, :]
    k2 = kx**2 + ky**2
    k2s = np.where(k2 == 0, 1.0, k2)
    dot = (kx*uk + ky*vk)/k2s            #Helmholtz: compressive part along k
    cu, cv = dot*kx, dot*ky
    su, sv = uk - cu, vk - cv
    e_tot = 0.5*(np.abs(uk)**2 + np.abs(vk)**2)
    e_comp = 0.5*(np.abs(cu)**2 + np.abs(cv)**2)
    e_sol = 0.5*(np.abs(su)**2 + np.abs(sv)**2)
    shell = np.rint(np.sqrt(k2)).astype(int)   #shells reach into the corners: up to ~n/sqrt(2)
    kmax = shell.max()
    E = np.bincount(shell.ravel(), e_tot.ravel(), kmax + 1)
    Es = np.bincount(shell.ravel(), e_sol.ravel(), kmax + 1)
    Ec = np.bincount(shell.ravel(), e_comp.ravel(), kmax + 1)
    k = np.arange(kmax + 1)
    return k[1:], E[1:], Es[1:], Ec[1:]


def fit_slope(k, E, kmin, kmax):
    """Least-squares power law E ~ k^s over kmin <= k <= kmax: slope and its standard error."""
    m = (k >= kmin) & (k <= kmax) & (E > 0)
    if m.sum() < 3:
        raise ValueError(f"fewer than 3 points with E > 0 in {kmin} <= k <= {kmax}")
    X, Y = np.log(k[m]), np.log(E[m])
    A = np.vstack([X, np.ones_like(X)]).T
    coef, res, _, _ = np.linalg.lstsq(A, Y, rcond=None)
    dof = m.sum() - 2
    sigma2 = (res[0]/dof) if (len(res) and dof > 0) else 0.0
    cov = sigma2 * np.linalg.inv(A.T @ A)
    return coef[0], np.sqrt(cov[0, 0])


def synthetic(n, slope, kind, seed):
    """Periodic field whose shell spectrum is ~ k^slope: |v_k|^2 ~ k^(slope-1) (a 2D shell holds
    ~ 2 pi k modes), purely solenoidal or purely compressive.
    Exact amplitudes with random PHASES only: random amplitudes would make each shell a sum of
    exponentially distributed energies, whose logarithm is biased low where shells hold few modes
    (small k), which flattens a fitted slope. No Nyquist modes (k = n/2 is its own partner, so a
    real field cannot be purely solenoidal or compressive there)."""
    rng = np.random.default_rng(seed)
    kx = np.fft.fftfreq(n, 1.0/n)[:, None]
    ky = np.fft.fftfreq(n, 1.0/n)[None, :]
    k = np.sqrt(kx**2 + ky**2)
    amp = np.where(k > 0, k, 1.0)**((slope - 1.0)/2.0)
    amp[0, 0] = 0.0
    amp[n//2, :] = 0.0
    amp[:, n//2] = 0.0
    phi = np.exp(2j*np.pi*rng.random((n, n)))*amp
    ks = np.where(k > 0, k, 1.0)
    if kind == "sol":                          #v = curl(psi): perpendicular to k
        uk, vk = 1j*ky/ks*phi, -1j*kx/ks*phi
    else:                                      #v = grad(phi): along k
        uk, vk = 1j*kx/ks*phi, 1j*ky/ks*phi
    return np.real(np.fft.ifft2(uk)), np.real(np.fft.ifft2(vk))


def selftest():
    ok = True
    def check(name, val, good, need):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'PASS' if good else 'FAIL'}  {name:52s} {val:12.4e}   (need {need})")
    print("spectrum.py self-test on synthetic fields with known spectra (256^2):")
    for slope, label in [(-5/3, "-5/3"), (-3.0, "-3"), (-2.0, "-2")]:
        for kind in ["sol", "comp"]:
            u, v = synthetic(256, slope, kind, seed=1)
            k, E, Es, Ec = spectra(u, v)
            s, ds = fit_slope(k, E, 4, 60)
            check(f"slope {label:>4s} ({'solenoidal' if kind == 'sol' else 'compressive'}): fitted - true",
                  s - slope, abs(s - slope) < 0.03, "|.| < 0.03")
            parseval = abs(E.sum()/(0.5*np.mean(u**2 + v**2)) - 1)
            check("  Parseval: sum E(k) vs 1/2 <|v|^2>", parseval, parseval < 1e-12, "< 1e-12")
            wrong = (Ec if kind == "sol" else Es).sum()/E.sum()
            check(f"  Helmholtz: {'compressive' if kind == 'sol' else 'solenoidal'} share of a pure field",
                  wrong, wrong < 1e-20, "< 1e-20")
    print("ALL PASSED" if ok else "SOME CHECKS FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser(description="kinetic energy spectra of Szuchydro output")
    ap.add_argument("outdir", nargs="?", help="simulation output dir (contains grid.bin)")
    ap.add_argument("--tmin", type=float, default=-np.inf, help="average snapshots from this time on")
    ap.add_argument("--tmax", type=float, default=np.inf)
    ap.add_argument("--weight", choices=["v", "sqrtrho"], default="v",
                    help="spectrum of v, or of sqrt(rho) v (integrates to the kinetic energy density)")
    ap.add_argument("--fit", type=float, nargs=2, metavar=("KMIN", "KMAX"), help="fit a power law here")
    ap.add_argument("--component", choices=["total", "sol", "comp"], default="total",
                    help="spectrum to fit: total, solenoidal or compressive. For 2D cascades use sol:\n"
                         "sound waves (compressive) can dominate the total at intermediate k")
    ap.add_argument("--gamma", type=float, default=1.4)
    ap.add_argument("-o", "--out", default=None, help="plot file (png/pdf); needs matplotlib")
    ap.add_argument("--selftest", action="store_true", help="check the script on synthetic fields")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if not a.outdir:
        ap.error("outdir is required (or use --selftest)")

    frames = read_frames(os.path.join(a.outdir, "grid.bin"), a.tmin, a.tmax)
    if not frames:
        sys.exit(f"no snapshots with {a.tmin} <= t <= {a.tmax}")
    acc = None
    for t, x, y, Q in frames:
        rho = Q[0]
        u, v = Q[1]/rho, Q[2]/rho
        if a.weight == "sqrtrho":
            u, v = np.sqrt(rho)*u, np.sqrt(rho)*v
        res = spectra(u, v)
        acc = list(res) if acc is None else [acc[0]] + [p + q for p, q in zip(acc[1:], res[1:])]
    k = acc[0]
    E, Es, Ec = (arr/len(frames) for arr in acc[1:])
    ts = [f[0] for f in frames]
    print(f"{len(frames)} snapshot(s), t = {min(ts):.3g} .. {max(ts):.3g}, grid {frames[0][3].shape[1]}^2,"
          f" weight = {a.weight}")
    print(f"compressive share of the energy: {Ec.sum()/E.sum():.3f}")
    if a.fit:
        Efit = {"total": E, "sol": Es, "comp": Ec}[a.component]
        s, ds = fit_slope(k, Efit, *a.fit)
        print(f"power-law fit of the {a.component} spectrum over {a.fit[0]:g} <= k <= {a.fit[1]:g}:"
              f"  E ~ k^({s:.3f} +- {ds:.3f})"
              f"   (2D references: -3 forward enstrophy cascade, -5/3 inverse energy cascade, -2 shocks)")
    if a.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4.8))
        m = E > 0
        ax.loglog(k[m], E[m], "k-", lw=1.8, label="total")
        ax.loglog(k[Es > 0], Es[Es > 0], "C0-", lw=1, label="solenoidal")
        ax.loglog(k[Ec > 0], Ec[Ec > 0], "C3-", lw=1, label="compressive")
        k0 = a.fit[0] if a.fit else 4
        E0 = np.interp(k0, k, {"total": E, "sol": Es, "comp": Ec}[a.component])
        kk = np.array([k0, k[m].max()])
        for p, st, lab in [(-5/3, ":", r"$k^{-5/3}$"), (-3, "--", r"$k^{-3}$"), (-2, "-.", r"$k^{-2}$")]:
            ax.loglog(kk, E0*(kk/k0)**p, "0.5", ls=st, lw=1, label=lab)
        if a.fit:
            ax.axvspan(*a.fit, color="0.9", zorder=0)
            ax.set_title(f"{a.component} spectrum, fit {a.fit[0]:g}-{a.fit[1]:g}: "
                         f"slope {s:.2f} ± {ds:.2f}")
        ax.set_xlabel(r"$k\,L/2\pi$"); ax.set_ylabel(r"$E(k)$")
        ax.set_ylim(E[m].max()*1e-9, E[m].max()*3)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout(); fig.savefig(a.out, dpi=300)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
