#!/usr/bin/env python3
"""
Velocity rms versus forcing rms for a set of forced runs (e.g. Kolmogorov flow at several kol_F).

For every run, u_rms = sqrt(< |u|^2>) of each snapshot in the steady state (from --tmin, or the second
half of the run) gives one point: the mean, with the standard deviation as the error bar (TODO: is this good?)
The forcing rms of a sine forcing a = F sin(k y) is a_rms = F/sqrt(2). A weighted least-squares fit in log-log
gives the exponent p of u_rms ~ a_rms^p, drawn with the reference slopes 1 and 1/2.

In 2D the exponent tells which mechanism balances the forcing: a linear drag gives p = 1 (u ~ a/alpha),
an eddy viscosity nu_t ~ u L gives p = 1/2 (as in 3D). The laminar regime is exactly linear (p = 1).

usage:  forcing_scaling.py RUN_DIR [RUN_DIR ...] --F F1 [F2 ...] [--tmin T] [--out plot.png]
        forcing_scaling.py --selftest
"""
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectrum import read_frames
SCALE = 0.6
FIGSIZE = (SCALE*8, SCALE*6)

def save(fig, fname):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig.savefig(fname, dpi=300)
    fig.savefig(fname.with_suffix(".pdf"))
    plt.close(fig)
    print(f"saved {fname} (+ .pdf)")


def steady_u_rms(outdir, tmin = None):
    """Mean and standard deviation of u_rms over the steady-state snapshots of one run
    (t >= tmin; by default the second half of the run). Returns (mean, std, number of snapshots)."""
    frames = read_frames(os.path.join(outdir, "grid.bin"))
    if tmin is None:
        tmin = 0.5*frames[-1][0] #get steady state
    vals = []
    for t, x, y, Q in frames:
        if t < tmin: #not in steady state -> skipped
            continue
        rho = Q[0]
        u, v = Q[1]/rho, Q[2]/rho
        vals.append(np.sqrt(np.mean(u*u + v*v))) #rms of velocity field
    vals = np.array(vals)
    return float(vals.mean()), float(vals.std()), len(vals)


def fit_exponent(a, u, du):
    """Weighted least-squares fit of log u = p log a + c. The weights come from the relative errors
    du/u; if all errors vanish (steady runs), an unweighted fit. Returns (p, error of p, c)."""
    a, u, du = np.asarray(a, float), np.asarray(u, float), np.asarray(du, float)
    x, y = np.log(a), np.log(u)
    sig = du/u
    w = 1.0/np.maximum(sig, 1e-12)**2 if np.all(sig > 0.0) else np.ones_like(x) #had to add floor
    W = np.sum(w)
    xm, ym = np.sum(w*x)/W, np.sum(w*y)/W
    sxx = np.sum(w*(x - xm)**2)
    p = np.sum(w*(x - xm)*(y - ym))/sxx
    c = ym - p*xm
    if len(x) > 2:  #error of the slope from the scatter about the fit
        res = y - (p*x + c)
        dp = float(np.sqrt(np.sum(w*res**2)/(len(x) - 2)/sxx))
    else:
        dp = float("nan")
    return float(p), dp, float(c)


def plot(a, u, du, p, c, out):
    """log-log u_rms(a_rms) with error bars, the fit and the reference slopes 1 and 1/2."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    a, u = np.asarray(a), np.asarray(u)
    fig, ax = plt.subplots(figsize = FIGSIZE)
    ax.errorbar(a, u, yerr = du, fmt = "o", capsize = 3, label = "runs (mean +- std)")
    aa = np.geomspace(a.min()/1.5, a.max()*1.5, 50)
    ax.plot(aa, np.exp(c)*aa**p, "-", label = f"fit: p = {p:.3f}")
    for q, ls in [(1.0, "--"), (0.5, ":")]:  #reference slopes through the middle point
        m = len(a)//2
        ax.plot(aa, u[m]*(aa/a[m])**q, ls, color = "gray", label = f"slope {q:g}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$a_{\rm rms}$")
    ax.set_ylabel(r"$u_{\rm rms}$")
    ax.legend()
    fig.tight_layout()
    save(fig, out)


def selftest():
    """Synthetic data u = c a^p with 2% scatter: the fit must recover p (1 and 1/2) to 0.02."""
    rng = np.random.default_rng(1)
    a = np.geomspace(0.01, 1.0, 8)
    ok = True
    for p_true in [1.0, 0.5]:
        u = 0.7*a**p_true*(1.0 + 0.02*rng.standard_normal(a.size))
        p, dp, c = fit_exponent(a, u, 0.02*u)
        good = abs(p - p_true) < 0.02
        ok = ok and good
        print(f"selftest p = {p_true}: fitted {p:.4f} +- {dp:.4f} {'ok' if good else 'FAILED'}")
    p, dp, c = fit_exponent(a, 0.3*a, np.zeros_like(a))  # exact data, no errors
    good = abs(p - 1.0) < 1e-12
    ok = ok and good
    print(f"selftest exact data (zero error bars): fitted {p:.15f} {'ok' if good else 'FAILED'}")
    print("SELFTEST PASSED" if ok else "SELFTEST FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description = "u_rms versus forcing a_rms for a set of forced runs")
    ap.add_argument("runs", nargs = "*", help = "run output directories (each contains grid.bin)")
    ap.add_argument("--F", type = float, nargs = "+", help = "forcing amplitude of each run (a_rms = F/sqrt(2))")
    ap.add_argument("--tmin", type = float, default = None, help = "steady state from this time on (default: second half)")
    ap.add_argument("--out", default = "forcing_scaling.png", help = "plot file")
    ap.add_argument("--selftest", action = "store_true", help = "check the fit on synthetic data")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    if not args.runs or not args.F or len(args.F) != len(args.runs):
        ap.error("give one --F value per run directory")
    a = np.array(args.F)/np.sqrt(2.0)
    stats = [steady_u_rms(r, args.tmin) for r in args.runs]
    u = np.array([s[0] for s in stats])
    du = np.array([s[1] for s in stats])
    for r, ai, s in zip(args.runs, a, stats):
        print(f"{r}: a_rms = {ai:.4e}  u_rms = {s[0]:.4e} +- {s[1]:.1e}  ({s[2]} snapshots)")
    p, dp, c = fit_exponent(a, u, du)
    print(f"exponent p = {p:.4f} +- {dp:.4f}   (drag-dominated: 1, eddy-viscosity-dominated: 1/2)")
    plot(a, u, du, p, c, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
