#!/usr/bin/env python3
"""
Szuchydro test suite

Physics problems checks:
    sod_uniform         Sod tube vs exact Riemann solution (L1 error) + conservation
    sod_smr             same with static refined band -> test refluxing
    sod_amr             same with adaptive refinement -> test AMR remap + refluxing
    sedov_uniform       2D Sedov-Taylor blast vs analytical solution + symmetry + conservation
    sedov_amr           same with AMR from coarse base grid (2D refinement in both directions)
    disk_equilibrium    disk_Rz without stirring/wind must stay in hydrostatic equilibrium
    disk_stirred        disk_Rz smoke test: finite, positive, sensible Mach numbers

Bit-for-bit mode (local only):
    before: python3 tests/run_tests.py --exe build/szuchydro2d --save-ref refs/
    after:  python3 tests/run_tests.py --exe build/szuchydro2d --compare-ref refs/

Usage:
    all tests: python3 tests/run_tests.py --exe build/szuchydro2d
    only names containing sod: python3 tests/run_tests.py --exe build/szuchydro2d -k sod
Exit code 0 = all passed, 1 = something failed (CI uses this)
"""

import argparse, hashlib, json, os, shutil, struct, subprocess, sys, tempfile, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = os.path.join(HERE, "cases")
NVAR = 4
HDR, BLK = "<dQ", "<iqqddddQQ" #16, 72 byte
HS, BS = struct.calcsize(HDR), struct.calcsize(BLK)


def read_frames(path):
    #all frames of grid.bin as list of (t, blocks, arrays)
    #blocks: (level, lx1, lx2, x1min, x1max, x2min, x2max, nx1, nx2)
    #arrays: (nx1, nx2, NVAR)
    frames = []
    with open(path, "rb") as f:
        while True:
            h = f.read()
            if len(h) < HS:
                break
            t, nb = struct.unpack(HDR, h)
            blocks = [struct.unpack(BLK, f.read(BS)) for _ in range(nb)]
            arrs = [np.frombuffer(f.read(b[7]*b[8]*NVAR*8), dtype="<f8").reshape(b[7], b[8], NVAR)
                    for b in blocks]
            frames.append((t, blocks, arrs))
    return frames

def totals(frame):
    #volume integrals of all conserved quantities (exact over mixed refinement levels)
    _, blocks, arrs = frame
    tot = np.zeros(NVAR)
    for b, a in zip(blocks, arrs):
        dV = (b[4] - b[3])/b[7] * (b[6] - b[5])/b[8]
        tot += a.reshape(-1, NVAR).sum(axis=0)*dV
    return tot

def stitch(frame):
    #finest-level uniform canvas: x, y (cell centres), Q (NVAR, nx, ny)
    _, blocks, arrs = frame
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
    return x0 + (np.arange(nx) + 0.5)*dx, y0 + (np.arange(ny) + 0.5)*dy, Q

def primitives(Q, gamma):
    rho = Q[0]
    u, v = Q[1]/rho, Q[2]/rho
    p = (gamma - 1.0)*(Q[3] - 0.5*rho*(u*u + v*v))
    return rho, u, v, p

def frame_hash(frame):
    h = hashlib.sha256()
    t, blocks, arrs = frame
    h.update(struct.pack("<d", t))
    for b, a in zip(blocks, arrs):
        h.update(struct.pack(BLK, *b))
        h.update(a.tobytes())
    return h.hexdigest()


#analytical solutions 
#Riemann
def exact_riemann(rl, ul, pl, rr, ur, pr, gamma, xi):
    #exact solution of the 1D Euler Riemann problem (Toro 2009, ch. 4), sampled at xi = x/t
    cl, cr = np.sqrt(gamma*pl/rl), np.sqrt(gamma*pr/rr)
    g1, g2 = (gamma - 1)/(2*gamma), (gamma + 1)/(2*gamma)

    def f(p, rk, pk, ck):
        if p > pk:  # shock
            A, B = 2/((gamma + 1)*rk), (gamma - 1)/(gamma + 1)*pk
            return (p - pk)*np.sqrt(A/(p + B)), np.sqrt(A/(B + p))*(1 - 0.5*(p - pk)/(B + p))
        return (2*ck/(gamma - 1)*((p/pk)**g1 - 1),  #rarefaction
                1/(rk*ck)*(p/pk)**(-g2))

    p = max(1e-8, 0.5*(pl + pr))
    for _ in range(100):
        fL, dL = f(p, rl, pl, cl)
        fR, dR = f(p, rr, pr, cr)
        pn = max(1e-12, p - (fL + fR + ur - ul)/(dL + dR))
        if abs(pn - p) < 1e-14*(pn + p):
            p = pn
            break
        p = pn
    fL, _ = f(p, rl, pl, cl)
    fR, _ = f(p, rr, pr, cr)
    u = 0.5*(ul + ur) + 0.5*(fR - fL)

    rho, vel, prs = np.empty_like(xi), np.empty_like(xi), np.empty_like(xi)
    for n, s in enumerate(xi):
        if s <= u:  #left of contact
            if p > pl:
                SL = ul - cl*np.sqrt(g2*p/pl + g1)
                st = (rl, ul, pl) if s <= SL else (rl*((p/pl + 
                                                  (gamma - 1)/(gamma + 1))/((gamma - 1)/(gamma + 1)*p/pl + 1)), u, p)
            else:
                cs_ = cl*(p/pl)**g1
                if s <= ul - cl:
                    st = (rl, ul, pl)
                elif s >= u - cs_:
                    st = (rl*(p/pl)**(1/gamma), u, p)
                else:
                    c = 2/(gamma + 1)*(cl + (gamma - 1)/2*(ul - s))
                    vv = 2/(gamma + 1)*(cl + (gamma - 1)/2*ul + s)
                    st = (rl*(c/cl)**(2/(gamma - 1)), vv, pl*(c/cl)**(2*gamma/(gamma - 1)))
        else:  #right of contact
            if p > pr:
                SR = ur + cr*np.sqrt(g2*p/pr + g1)
                st = (rr, ur, pr) if s >= SR else (rr*((p/pr + (gamma - 1)/(gamma + 1))/
                                                  ((gamma - 1)/(gamma + 1)*p/pr + 1)), u, p)
            else:
                cs_ = cr*(p/pr)**g1
                if s >= ur + cr:
                    st = (rr, ur, pr)
                elif s <= u + cs_:
                    st = (rr*(p/pr)**(1/gamma), u, p)
                else:
                    c = 2/(gamma + 1)*(cr - (gamma - 1)/2*(ur - s))
                    vv = 2/(gamma + 1)*(-cr + (gamma - 1)/2*ur + s)
                    st = (rr*(c/cr)**(2/(gamma - 1)), vv, pr*(c/cr)**(2*gamma/(gamma - 1)))
        rho[n], vel[n], prs[n] = st
    return rho, vel, prs

#Sedov-Taylor blast solution
def sedov_profile(j, gamma, n=20000):
    """Standard Sedov-Taylor solution for a uniform medium (omega = 0), closed form after
    Sedov (1959) / Kamm & Timmes (2007). j = 1 planar, 2 cylindrical, 3 spherical.
    Returns xi = r/r_shock and rho/rho_2, u/u_2, p/p_2 (2 = immediately behind the shock),
    plus alpha with  r_shock = (E / (alpha rho_0))**(1/(j+2)) * t**(2/(j+2))."""
    g, w = gamma, 0.0
    a = (j + 2 - w)*(g + 1)/4
    b  = (g + 1)/(g - 1)
    c = (j + 2 - w)*g/2
    d = (j + 2 - w)*(g + 1)/((j + 2 - w)*(g + 1) - 2*(2 + j*(g - 1)))
    e = (2 + j*(g - 1))/2
    a0 = 2/(j + 2 - w)
    a2 = -(g - 1)/(2*(g - 1) + j - g*w)
    a1 = (j + 2 - w)*g/(2 + j*(g - 1))*(2*(j*(2 - g) - w)/(g*(j + 2 - w)**2) - a2)
    a3 = (j - w)/(2*(g - 1) + j - g*w)
    a4 = (j + 2 - w)*(j - w)/(j*(2 - g) - w)*a1
    a5 = (w*(1 + g) - 2*j)/(j*(2 - g) - w)
    V0 = 2/((j + 2 - w)*g)          #centre (xi -> 0)
    V2 = 4/((j + 2 - w)*(g + 1))    #shock  (xi = 1)
    #cluster points towards V0 where the profile varies rapidly in V
    s = np.linspace(0, 1, n)[1:]
    V = V0 + (V2 - V0)*s**4
    x1, x2, x3, x4 = a*V, b*(c*V - 1), d*(1 - e*V), b*(1 - c*V/g)
    xi = x1**-a0*x2**-a2*x3**-a1
    rho = x1**(a0*w)*x2**(a3 + a2*w)*x3**(a4 + a1*w)*x4**a5
    u = x1*xi
    p = x1**(a0*j)*x3**(a4 + a1*(w - 2))*x4**(1 + a5)
    o = np.argsort(xi)
    xi, rho, u, p = xi[o], rho[o], u[o], p[o]
    #energy integral at t = 1, r_s = 1, rho_0 = 1: shock speed us = a0, post-shock state
    us = a0
    rho2, u2, p2 = b, 2/(g + 1)*us, 2/(g + 1)*us**2
    area = {1: 2.0, 2: 2*np.pi, 3: 4*np.pi}[j]
    eden = 0.5*rho2*rho*(u2*u)**2 + p2*p/(g - 1)
    f = eden*area*xi**(j - 1)
    alpha = np.sum(0.5*(f[1:] + f[:-1])*np.diff(xi))  #E = alpha rho0 r^(j+2) / t^2 at r=t=1
    return xi, rho, u, p, alpha

#checks
class Result:
    def __init__(self, name):
        self.name, self.checks = name, []

    def check(self, what, value, ok, limit):
        self.checks.append((what, value, ok, limit))

    @property
    def passed(self):
        return all(ok for _, _, ok, _ in self.checks)

def check_conservation(res, frames, pl, pr, tol=1e-12):
    """The waves never reach the open walls, so no mass or energy crosses them. Momentum does
    change, exactly by the pressure difference between the walls: d(Px) = (pl - pr)*Ly*t."""
    t0, t1 = totals(frames[0]), totals(frames[-1])
    for k, name in [(0, "mass"), (3, "energy")]:
        rel = abs(t1[k] - t0[k])/abs(t0[k])
        res.check(f"{name} conservation |dQ|/Q", rel, rel < tol, f"< {tol:g}")
    _, blocks, _ = frames[-1]
    Ly = max(b[6] for b in blocks) - min(b[5] for b in blocks)
    expected = (pl - pr)*Ly*(frames[-1][0] - frames[0][0])
    err = abs((t1[1] - t0[1]) - expected)/abs(expected)
    res.check("x-momentum change vs (pl-pr)*Ly*t", err, err < tol, f"< {tol:g} (relative)")

def check_sod(res, frames, gamma, l1_limit):
    t = frames[-1][0]
    x, y, Q = stitch(frames[-1])
    rho, u, v, p = primitives(Q, gamma)
    rho_ex, _, _ = exact_riemann(1.0, 0.0, 1.0, 0.125, 0.0, 0.1, gamma, x/t) #vals from test init cond
    rho_mean = rho.mean(axis=1)  #average over the periodic y direction
    l1 = np.mean(np.abs(rho_mean - rho_ex))
    res.check("L1(rho) vs exact solution", l1, l1 < l1_limit, f"< {l1_limit:g}")
    yspread = np.max(np.abs(rho - rho_mean[:, None]))
    res.check("1D symmetry: max |rho - <rho>_y|", yspread, yspread < 1e-10, "< 1e-10")
    res.check("final time reached", t, abs(t - 0.2) < 1e-12, "= 0.2")
    check_conservation(res, frames, pl=1.0, pr=0.1)

def check_sedov(res, frames, gamma, p_amb, l1_limit):
    #2D (cylindrical, j = 2) Sedov-Taylor blast centred in the domain
    xi, srho, _, _, alpha = sedov_profile(2, gamma)
    t = frames[-1][0]
    _, blocks, _ = frames[0]
    area = (max(b[4] for b in blocks) - min(b[3] for b in blocks))* \
           (max(b[6] for b in blocks) - min(b[5] for b in blocks))
    e_blast = totals(frames[0])[3] - p_amb/(gamma - 1)*area
    res.check("deposited blast energy |E - 1|", abs(e_blast - 1), abs(e_blast - 1) < 1e-10, "< 1e-10")
    rs = (e_blast/alpha)**0.25*np.sqrt(t)          # analytic shock radius
    x, y, Q = stitch(frames[-1])
    rho = Q[0]
    X, Y = np.meshgrid(x - 0.5*(x[0] + x[-1]), y - 0.5*(y[0] + y[-1]), indexing="ij")
    R = np.hypot(X, Y)
    #azimuthal average in bins of the coarsest cell size (no empty bins); only used for the front
    dxc = max((b[4] - b[3])/b[7] for b in frames[-1][1])
    edges = np.arange(0.0, R.max(), dxc)
    rc = 0.5*(edges[1:] + edges[:-1])
    idx = np.digitize(R.ravel(), edges) - 1
    prof = np.array([rho.ravel()[idx == k].mean() for k in range(len(rc))])
    #shock front: outermost radius where rho > 2 rho_0 (post-shock is 6 rho_0 for gamma = 1.4)
    r_front = rc[np.nonzero(prof > 2.0)[0].max()]
    err = abs(r_front/rs - 1)
    res.check("shock radius |r_front/r_s - 1|", err, err < 0.02, "< 0.02")
    #cell by cell against the analytic profile at each cell's radius (binning would make the
    #result depend on where the shock spike falls relative to the bin edges)
    rho2 = (gamma + 1)/(gamma - 1)
    exact = np.interp(R/rs, xi, srho*rho2, right=1.0)
    m = R < 1.2*rs
    l1 = np.sum(np.abs(rho - exact)[m])/np.sum(exact[m])
    res.check("relative L1(rho) inside 1.2 r_s", l1, l1 < l1_limit, f"< {l1_limit:g}")
    sym = float(np.abs(rho - rho.T).max())
    res.check("x<->y symmetry max|rho - rho^T|", sym, sym < 1e-10, "< 1e-10")
    t0, t1 = totals(frames[0]), totals(frames[-1])
    for k, name in [(0, "mass"), (3, "energy")]:
        rel = abs(t1[k] - t0[k])/abs(t0[k])
        res.check(f"{name} conservation |dQ|/Q", rel, rel < 1e-12, "< 1e-12")


def disk_fields(frame, gamma):
    x, y, Q = stitch(frame)
    rho, u, v, p = primitives(Q, gamma)
    X, Y = np.meshgrid(x, y, indexing="ij")
    eta = np.abs(Y/X)
    mach = np.hypot(u, v)/np.sqrt(np.maximum(gamma * p/rho, 1e-300))
    return X, Y, eta, rho, u, v, p, mach

def check_finite_positive(res, Q, gamma):
    rho, u, v, p = primitives(Q, gamma)
    ok = bool(np.isfinite(Q).all())
    res.check("all values finite", float(ok), ok, "= 1")
    res.check("min density", rho.min(), rho.min() > 0, "> 0")
    res.check("min pressure", p.min(), p.min() > 0, "> 0")

def run_case(exe, name, workdir):
    src = os.path.join(CASES, name + ".txt") #input file
    shutil.copy(src, workdir)
    t0 = time.time()
    r = subprocess.run([exe, name + ".txt"], cwd=workdir, capture_output=True, text=True) #run it
    dt = time.time() - t0
    log = r.stdout + r.stderr #we capture the output
    with open(os.path.join(workdir, name + ".log"), "w") as f:
        f.write(log)
    outdir = None
    for line in open(src):
        parts = line.split("#")[0].split()
        if len(parts) == 2 and parts[0] == "outdir":
            outdir = os.path.join(workdir, parts[1])
    return r.returncode, log, dt, outdir

def test_case(exe, name, workdir):
    res = Result(name)
    code, log, dt, outdir = run_case(exe, name, workdir)
    res.runtime = dt
    res.outdir = outdir
    res.check("exit code", code, code == 0, "= 0")
    if code != 0:
        res.log_tail = "\n".join(log.replace("\r", "\n").splitlines()[-15:])
        return res
    bad = [l for l in log.replace("\r", "\n").splitlines() if "INVALID" in l or "ERROR" in l]
    res.check("no INVALID/ERROR in log", len(bad), not bad, "= 0")
    gfile = os.path.join(outdir, "grid.bin") if outdir else ""
    frames = read_frames(gfile) if os.path.isfile(gfile) else []
    res.check("snapshots written", len(frames), len(frames) >= 2, ">= 2")
    if len(frames) < 2:
        return res
    gamma = 1.4
    x, y, Q = stitch(frames[-1])
    check_finite_positive(res, Q, gamma)

    if name == "sod_uniform":
        check_sod(res, frames, gamma, l1_limit=6e-3)
    elif name == "sod_smr":
        check_sod(res, frames, gamma, l1_limit=8e-3)
        res.check("refined blocks present", len(frames[-1][1]), len(frames[-1][1]) > 8, "> 8 blocks")
    elif name == "sod_amr":
        check_sod(res, frames, gamma, l1_limit=8e-3)
        lv = max(b[0] for b in frames[-1][1])
        res.check("AMR refined (max level)", lv, lv >= 1, ">= 1")
    elif name == "sedov_uniform":
        check_sedov(res, frames, gamma, p_amb=1e-5, l1_limit=0.18)
    elif name == "sedov_amr":
        check_sedov(res, frames, gamma, p_amb=1e-5, l1_limit=0.11)
        lv = max(b[0] for b in frames[-1][1])
        res.check("AMR refined (max level)", lv, lv >= 2, ">= 2")
    elif name == "disk_equilibrium":
        X, Y, eta, rho, u, v, p, mach = disk_fields(frames[-1], gamma)
        rho0 = stitch(frames[0])[2][0]
        dm = abs(totals(frames[-1])[0]/totals(frames[0])[0] - 1)
        res.check("|total mass change|", dm, dm < 2e-3, "< 2e-3")
        med = float(np.median(mach[eta < 0.25]))
        res.check("median Mach in disk", med, med < 5e-3, "< 5e-3")
        mid = rho[:, rho.shape[1]//2]/rho0[:, rho.shape[1]//2]
        dev = float(np.max(np.abs(mid - 1)))
        res.check("max |rho/rho0 - 1| at midplane", dev, dev < 0.05, "< 0.05")
    elif name == "disk_stirred":
        X, Y, eta, rho, u, v, p, mach = disk_fields(frames[-1], gamma)
        med = float(np.median(mach[eta < 0.25]))
        res.check("median Mach in disk", med, 0.01 < med < 0.3, "in (0.01, 0.3)")
        cor = eta > 0.5
        out = float(np.mean((np.sign(Y)*v)[cor] > 0))
        res.check("corona fraction flowing outward", out, out > 0.9, "> 0.9")
    return res

#test cases
ALL = ["sod_uniform", "sod_smr", "sod_amr", "sedov_uniform", "sedov_amr",
       "disk_equilibrium", "disk_stirred"]

def main():
    ap = argparse.ArgumentParser(description="Szuchydro test suite")
    ap.add_argument("--exe", default="build/szuchydro2d", help="path to the executable")
    ap.add_argument("-k", default="", help="only run tests whose name contains this")
    ap.add_argument("--keep", action="store_true", help="keep the run directory")
    ap.add_argument("--save-ref", metavar="DIR", help="store final-frame hashes (bit-for-bit mode)")
    ap.add_argument("--compare-ref", metavar="DIR", help="compare final-frame hashes to DIR")
    a = ap.parse_args()
    exe = os.path.abspath(a.exe)
    if not os.path.isfile(exe):
        sys.exit(f"executable not found: {exe}")

    names = [n for n in ALL if a.k in n]
    work = tempfile.mkdtemp(prefix="szuchydro_tests_")
    print(f"running {len(names)} test(s) in {work}\n")

    results, hashes = [], {}
    for n in names:
        res = test_case(exe, n, work)
        results.append(res)
        status = "PASS" if res.passed else "FAIL"
        print(f"[{status}] {n}  ({res.runtime:.1f} s)")
        for what, val, ok, lim in res.checks:
            mark = " " if ok else "x"
            v = f"{val:.3e}" if isinstance(val, (float, np.floating)) else str(val)
            print(f"    {mark} {what:38s} {v:>12s}   (need {lim})")
        if hasattr(res, "log_tail"):
            print("    --- last log lines ---\n" + res.log_tail)
        if res.outdir and os.path.isfile(os.path.join(res.outdir, "grid.bin")):
            frames = read_frames(os.path.join(res.outdir, "grid.bin"))
            if frames:  #a crashed run can leave an empty file behind
                hashes[n] = frame_hash(frames[-1])


    ok = all(r.passed for r in results) #only if every test passed
    if a.save_ref:
        os.makedirs(a.save_ref, exist_ok=True)
        with open(os.path.join(a.save_ref, "hashes.json"), "w") as f:
            json.dump(hashes, f, indent=2)
        print(f"\nsaved {len(hashes)} final-frame hashes to {a.save_ref}/hashes.json")
    if a.compare_ref:
        ref = json.load(open(os.path.join(a.compare_ref, "hashes.json")))
        print("\nbit-for-bit comparison:")
        for n in names:
            if n not in ref:
                print(f"    ? {n}: no reference")
                continue
            same = hashes.get(n) == ref[n]
            ok &= same
            print(f"    {'identical' if same else 'DIFFERENT'}  {n}")

    if not a.keep:
        shutil.rmtree(work, ignore_errors=True)
    print("\nALL PASSED" if ok else "\nSOME TESTS FAILED")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
