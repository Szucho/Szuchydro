#!/usr/bin/env python3
"""
Szuchydro test suite

Physics problems checks:
    sod_uniform         Sod tube vs exact Riemann solution (L1 error) + conservation
    sod_smr             same with static refined band -> test refluxing
    sod_amr             same with adaptive refinement -> test AMR remap + refluxing
    sod_amr_level2      AMR capped at amr_max_level 2 (tests AMR parameters)
    invalid_params      bad input must stop with 'Parameter error' and an exit code 1
    invalid_integrator  unknown integrator must stop with a "Parameter error"
    turb_solenoidal     driven turbulent box, solenoidal driving: conservation, Mach, mostly vortical, mode list
    turb_compressive    the same with compressive driving: mostly compressive flow
    invalid_turb_bc     turb with non-periodic walls must stop with 'Parameter error'
    turb_drag           exact: uniform flow without forcing decays as exp(-turb_drag t), internal energy unchanged
    wb_isothermal       well-balanced isothermal atmosphere: at rest to machine precision
    wb_tjump            the same with a tenfold temperature jump
    wb_pointmass_iso/_poly   barotropic well-balancing in a 2D potential (point mass): at rest to machine precision
    wb_pointmass_linear  the same state with the linear variant must drift (counter-check)
    wb_pointmass_integrators  the isothermal case with euler and rk2
    wb_pointmass_pert   a 1e-6 pulse in the 2D atmosphere: wave visible, far field at round-off
    invalid_wb_equilibrium  unknown wb_equilibrium must stop with 'Parameter error'
    wb_smr_hband/_vband/_tjump/_nested/_pointmass   well-balancing across static refinement levels
    wb_smr_integrators  wb_smr_hband with euler and rk2
    wb_smr_pulse        a pulse crossing a refined band: mass exactly conserved
    wb_amr_rest         AMR + well_balanced at rest: nothing refines, at rest to machine precision
    wb_amr_pulse        a pulse followed by AMR (linear and isothermal): mass exact, close to the uniform run
    wb_amr_reset_pulse  the same with a reference reset (closure for new blocks afterwards)
    wb_reset_smr / wb_reset_pointmass   a reset of an exact equilibrium with static refinement stays exact
    wb_amr_refcheck     self-check: after every regrid, every block's reference equals the initial condition
    amr_criterion_select  one AMR crit/run: pressure_gradient refines a pure pressure pulse, density/vorticity do not
    amr_crit_vorticity  the vorticity criterion refines the vortex centre, not the far field
    amr_crit_hook       the problem's geometry hook refines exactly its region
    invalid_amr_criterion  unknown criterion, contrast without well_balanced, hook without a hook: Parameter error
    hydrostatic_wall_mass_pm/_poly/_atm  pulses hitting Hydrostatic walls: mass exactly conserved (wall flux)
    hydrostatic_wall_reflection  without gravity a Hydrostatic wall reflects like a Closed wall
    visc_* / cond_* / diffusion_*   viscosity (nu, nu_b) and conduction (chi): exact decay rates, free-slip and
                                    insulating walls, convergence, conservation, refinement, well-balancing
    kol_*               Kolmogorov flow: laminar transient and steady state, momentum, cooling, heating,
                        instability, reproducibility, drag guard, energy budget, laminar u(a) scaling
    hist_*              history output (history.hst): conservation, agreement with snapshots,
                        interval independence, kinetic-energy budgets (viscous, pressure work, Kolmogorov)
    nowb_isothermal     well_balanced 0: must drift (the switch matters)
    wb_harmonic_axes    harmonic potential along y and x: at rest, x == transposed y (all 4 walls)
    wb_perturbation     a 1e-6 pressure pulse propagates; the far field stays at round-off
    invalid_wb_no_potential / invalid_hydrostatic_no_potential   must stop with 'Parameter error'
    conv_advect, conv_vortex   order of accuracy (N = 32, 64) for euler, vl2, rk2: vl2/rk2 second order
    wb_tjump_vl2/_rk2, sedov_amr_vl2, sod_amr_rk2, turb_solenoidal_vl2   the delicate parts with vl2/rk2
    sod_vanleer, sod_superbee, sod_mc, sod_primitive   Sod with each limiter / primitive reconstruction
    dirichlet_contact_x/_y  contact carried in through a Dirichlet wall
    contact_primitive   the same with primitive reconstruction: u and p uniform to round-off
    invalid_dirichlet   Dirichlet wall with a missing state component must stop with 'Parameter error'
    outflow_walls       gas streaming towards each wall (outflow): rotated/mirrored results must agree
    invalid_problem     unknown gas_profile must stop with 'Parameter error'
    invalid_number      'tf 0.2s' must stop with 'Parameter error' (no silent partial parsing)
    unused_param_warning  a misspelled key ('limitter') must run but be reported as unused
    sedov_uniform       2D Sedov-Taylor blast vs analytical solution + symmetry + conservation
    sedov_amr           same with AMR from coarse base grid (2D refinement in both directions)
    disk_equilibrium    disk_Rz without stirring/wind must stay in hydrostatic equilibrium
    disk_stirred        disk_Rz smoke test: finite, positive, sensible Mach numbers

Bit-for-bit mode (local only):
    before: python3 tests/run_tests.py --exe build/szuchydro --save-ref refs/
    after:  python3 tests/run_tests.py --exe build/szuchydro --compare-ref refs/

Usage:
    all tests: python3 tests/run_tests.py --exe build/szuchydro
    only names containing sod: python3 tests/run_tests.py --exe build/szuchydro -k sod
Exit code 0 = all passed, 1 = something failed (CI uses this)
"""

import argparse, re, hashlib, json, os, shutil, struct, subprocess, sys, tempfile, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TIMEOUT_S = 600  # per test, the slowest test takes about 10 s
CASES = os.path.join(HERE, "cases")
NVAR = 4
EXTRA = [] #set by --extra
HDR, BLK = "<dQ", "<iqqddddQQ" #16, 72 byte
HS, BS = struct.calcsize(HDR), struct.calcsize(BLK)


def read_frames(path):
    #all frames of grid.bin as list of (t, blocks, arrays)
    #blocks: (level, lx1, lx2, x1min, x1max, x2min, x2max, nx1, nx2)
    #arrays: (nx1, nx2, NVAR)
    frames = []
    with open(path, "rb") as f:
        while True:
            h = f.read(HS)
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

def save_frame_npz(path, frame):
    """final snapshot for --save-ref: time, block table and data, so a later comparison
    can say HOW different two runs are, not just whether they differ."""
    t, blocks, arrs = frame
    np.savez_compressed(path, t=np.array(t), blocks=np.array(blocks, dtype=np.float64),
                        data=np.stack(arrs))


VAR_NAMES = ["rho", "mx", "my", "E"]


def compare_frames(ref_npz, frame):
    """none if the meshes differ, else the max difference per variable relative to that
    variable's largest magnitude in the reference (absolute if the reference is all zero)."""
    ref = np.load(ref_npz)
    t, blocks, arrs = frame
    if ref["blocks"].shape != np.array(blocks).shape or \
       not np.array_equal(ref["blocks"], np.array(blocks, dtype=np.float64)):
        return None
    new, old = np.stack(arrs), ref["data"]
    diffs = []
    for k in range(NVAR):
        scale = np.max(np.abs(old[..., k]))
        d = np.max(np.abs(new[..., k] - old[..., k]))
        diffs.append(d/scale if scale > 0 else d)
    return diffs

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

def check_sod_contact_width(res, frames, max_cells):
    """width of the Sod contact (rho between 0.29 and 0.40 around x = 0.185 at t = 0.2).
    minmod with conserved reconstruction smears it over 4 cells; sharper limiters use fewer."""
    x, y, Q = stitch(frames[-1])
    rho = Q[0].mean(axis=1)
    w = int(np.sum((x > 0.05) & (x < 0.30) & (rho > 0.29) & (rho < 0.40)))
    res.check("contact width [cells]", w, w <= max_cells, f"<= {max_cells}")

def check_contact(res, frames, gamma, axis, uniform_tol):
    """Dirichlet inflow: rho = 2 enters a uniform flow rho = 1, speed 1, p = 1 along 'axis'.
    Exact: the contact sits at 0.5 at t = 0.5; u and p stay uniform; the box gains exactly
    (2 - 1)*L*t mass and (4.5 - 4.0)*L*t energy (boundary fluxes rho*u and (E + p)*u)."""
    t = frames[-1][0]
    x, y, Q = stitch(frames[-1])
    rho, u, v, p = primitives(Q, gamma)
    if axis == "x":
        coord, prof, vel, vtr = x, rho.mean(axis=1), u, v
    else:
        coord, prof, vel, vtr = y, rho.mean(axis=0), v, u
    d = coord[1] - coord[0]
    k = np.nonzero(prof < 1.5)[0][0]  # first cell where rho has dropped below the midpoint
    pos = coord[k - 1] + (1.5 - prof[k - 1])*d/(prof[k] - prof[k - 1])
    err = abs(pos - t)/d                     # contact moves at speed 1, so it should be at t
    res.check("contact position error [cells]", err, err < 0.5, "< 0.5")
    for name, f in [("max |p - 1|", p - 1.0), ("max |u_normal - 1|", vel - 1.0), ("max |u_transverse|", vtr)]:
        val = float(np.abs(f).max())
        res.check(name, val, val < uniform_tol, f"< {uniform_tol:g}")
    _, blocks, _ = frames[0]
    L = (max(b[6] for b in blocks) - min(b[5] for b in blocks)) if axis == "x" else \
        (max(b[4] for b in blocks) - min(b[3] for b in blocks))
    t0, t1 = totals(frames[0]), totals(frames[-1])
    for k_, name, flux in [(0, "mass", 1.0), (3, "energy", 0.5)]:
        rel = abs((t1[k_] - t0[k_])/(flux*L*t) - 1)
        res.check(f"{name} inflow vs exact boundary flux", rel, rel < 1e-12, "< 1e-12 (relative)")


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


TURB_DEFAULT_MODES = [(0, 1), (0, 2), (1, -2), (1, -1), (1, 0), (1, 1), (1, 2), (2, -1), (2, 0), (2, 1)]


def check_turb(res, frames, gamma, log, comp_range, mach_range):
    """Driven periodic box: no net mass or momentum change, saturated Mach number in range, and
    the compressive share of the kinetic energy (Helmholtz decomposition by FFT) in range."""
    t0, t1 = totals(frames[0]), totals(frames[-1])
    dm = abs(t1[0]/t0[0] - 1)
    res.check("mass conservation |dM|/M", dm, dm < 1e-12, "< 1e-12")
    x, y, Q = stitch(frames[-1])
    rho, u, v, p = primitives(Q, gamma)
    cs = np.sqrt(gamma*p/rho)
    vrms = np.sqrt(np.mean(rho*(u*u + v*v))/np.mean(rho))
    mom = np.hypot(t1[1], t1[2])/(t1[0]*vrms)
    res.check("net momentum |P|/(M v_rms)", mom, mom < 1e-12, "< 1e-12")
    mach = vrms/np.mean(cs)
    res.check("Mach_rms", mach, mach_range[0] < mach < mach_range[1], f"in {mach_range}")
    uk, vk = np.fft.fft2(u - u.mean()), np.fft.fft2(v - v.mean())
    kx = np.fft.fftfreq(u.shape[0])[:, None]; ky = np.fft.fftfreq(u.shape[1])[None, :]
    k2 = kx**2 + ky**2; k2[0, 0] = 1.0
    dot = (kx*uk + ky*vk)/k2
    e_comp = np.sum(np.abs(dot*kx)**2 + np.abs(dot*ky)**2)
    e_tot = np.sum(np.abs(uk)**2 + np.abs(vk)**2)
    frac = float(e_comp/e_tot)
    res.check("compressive energy fraction", frac, comp_range[0] < frac < comp_range[1], f"in {comp_range}")
    line = [l for l in log.splitlines() if l.startswith("turb:")]
    modes = [tuple(map(int, m)) for m in re.findall(r"\((-?\d+),(-?\d+)\)", line[0])] if line else []
    res.check("mode list = default shell (10 modes)", len(modes), modes == TURB_DEFAULT_MODES, "= expected list")


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
    if EXTRA: #--extra: lines appended to every case
        with open(os.path.join(workdir, name + ".txt"), "a") as f:
            f.write("\n" + "\n".join(EXTRA) + "\n")
    t0 = time.time()
    try: #run sim, hanging run (e.g. caused by Nan somewhere) becomes a clean failure
        r = subprocess.run([exe, name + ".txt"], cwd=workdir, capture_output=True, text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"") if isinstance(e.stdout, bytes) else (e.stdout or "").encode()
        r = subprocess.CompletedProcess(e.cmd, -999, out.decode(errors="replace"), 
                                        f"\nTIMEOUT: run killed after {TIMEOUT_S} s\n")
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

def run_variant(exe, base, overrides, workdir):
    """Run test case 'base' with extra lines appended (a repeated key overrides the case's value);
    the variant gets its own name and output directory."""
    tag = "_".join(v.split()[-1] for v in overrides)
    name = f"{base}__{tag}"
    with open(os.path.join(CASES, base + ".txt")) as f:
        txt = f.read()
    txt += "\n" + "\n".join(overrides) + f"\noutdir {name}_out\n"
    with open(os.path.join(workdir, name + ".txt"), "w") as f:
        f.write(txt)
    t0 = time.time()
    r = subprocess.run([exe, name + ".txt"], cwd=workdir, capture_output=True, text=True, timeout=TIMEOUT_S)
    return r.returncode, r.stdout + r.stderr, time.time() - t0, os.path.join(workdir, name + "_out")

def test_convergence(exe, name, workdir, limits, ratio):
    """Order of accuracy: L1(rho) after one full period (exact solution = initial cell averages)
    at N = 32 and 64, for each integrator. limits: {integrator: (min_order, max_order)};
    ratio: (integrator, max error relative to euler at N = 64)."""
    res = Result(name)
    res.runtime, res.outdir = 0.0, None
    err = {}
    for integ in ["euler", "vl2", "rk2"]:
        for N in (32, 64):
            code, log, dt, outdir = run_variant(exe, name, [f"Nx {N}", f"Ny {N}", f"integrator {integ}"], workdir)
            res.runtime += dt
            if code != 0:
                res.check(f"exit code ({integ}, N={N})", code, False, "= 0")
                res.log_tail = "\n".join(log.replace("\r", "\n").splitlines()[-15:])
                return res
            fr = read_frames(os.path.join(outdir, "grid.bin"))
            _, _, Q0 = stitch(fr[0])
            _, _, Q1 = stitch(fr[-1])
            err[(integ, N)] = float(np.mean(np.abs(Q1[0] - Q0[0])))
    for integ, (lo, hi) in limits.items():
        order = np.log2(err[(integ, 32)]/err[(integ, 64)])
        res.check(f"order of accuracy ({integ})", order, lo < order < hi, f"in ({lo:g}, {hi:g})")
    for integ, r in ratio:
        q = err[(integ, 64)]/err[("euler", 64)]
        res.check(f"L1 error {integ} / euler at N=64", q, q < r, f"< {r:g}")
    return res

def test_wb_pointmass_integrators(exe, workdir):
    """The isothermal point-mass atmosphere with euler and rk2 (vl2 is the default, covered by
    wb_pointmass_iso): every integrator must keep the barotropic equilibrium at rest."""
    name = "wb_pointmass_integrators"
    res = Result(name)
    res.runtime, res.outdir = 0.0, None
    for integ in ["euler", "rk2"]:
        code, log, dt, outdir = run_variant(exe, name, [f"integrator {integ}"], workdir)
        res.runtime += dt
        res.check(f"exit code ({integ})", code, code == 0, "= 0")
        if code != 0:
            res.log_tail = "\n".join(log.replace("\r", "\n").splitlines()[-15:])
            return res
        frames = read_frames(os.path.join(outdir, "grid.bin"))
        x, y, Q = stitch(frames[-1])
        rho, u, v, p = primitives(Q, 1.4)
        mach = float(np.max(np.hypot(u, v)/np.sqrt(1.4*p/rho)))
        res.check(f"max |v|/cs ({integ})", mach, mach < 1e-12, "< 1e-12")
    return res


def test_wb_smr_integrators(exe, workdir):
    """wb_smr_integrators: the refined-band atmosphere with euler and rk2 (vl2 is covered by
    wb_smr_hband): every integrator must keep it at rest across the level boundaries."""
    name = "wb_smr_integrators"
    res = Result(name)
    res.runtime, res.outdir = 0.0, None
    for integ in ["euler", "rk2"]:
        code, log, dt, outdir = run_variant(exe, name, [f"integrator {integ}"], workdir)
        res.runtime += dt
        res.check(f"exit code ({integ})", code, code == 0, "= 0")
        if code != 0:
            res.log_tail = "\n".join(log.replace("\r", "\n").splitlines()[-15:])
            return res
        frames = read_frames(os.path.join(outdir, "grid.bin"))
        x, y, Q = stitch(frames[-1])
        rho, u, v, p = primitives(Q, 1.4)
        mach = float(np.max(np.hypot(u, v)/np.sqrt(1.4*p/rho)))
        res.check(f"max |v|/cs ({integ})", mach, mach < 1e-12, "< 1e-12")
    return res

def test_wb_amr_pulse(exe, workdir):
    """Check amr run against uniform, check max refinement level and conservation"""
    name = "wb_amr_pulse"
    res = Result(name)
    res.runtime, res.outdir = 0.0, None
    for eq, mass_tol in [("linear", 1e-12), ("isothermal", 1e-10)]:
        peak = {}
        for amr in ["1", "0"]:
            code, log, dt, outdir = run_variant(exe, name, [f"wb_equilibrium {eq}", f"use_amr {amr}"], workdir)
            res.runtime += dt
            if code != 0:
                res.check(f"exit code ({eq}, use_amr {amr})", code, False, "= 0")
                res.log_tail = "\n".join(log.replace("\r", "\n").splitlines()[-15:])
                return res
            frames = read_frames(os.path.join(outdir, "grid.bin"))
            x, y, Q = stitch(frames[-1])
            rho, u, v, p = primitives(Q, 1.4)
            peak[amr] = float(np.max(np.hypot(u, v)/np.sqrt(1.4*p/rho)))
            if amr == "1":
                lv = max(b[0] for b in frames[-1][1])
                res.check(f"AMR refines the pulse ({eq}, max level)", lv, lv == 2, "= 2")
                dm = abs(totals(frames[-1])[0]/totals(frames[0])[0] - 1)
                res.check(f"mass conservation with AMR ({eq})", dm, dm < mass_tol, f"< {mass_tol:g}")
        q = peak["1"]/peak["0"]
        res.check(f"peak |v|/cs AMR / uniform ({eq})", q, 0.5 < q < 2.0, "in (0.5, 2)")
    return res

def block_distance(b, px, py):
    """Distance from the point (px, py) to a block's rectangle (0 if the point lies inside or on it)."""
    dx = max(b[3] - px, 0.0, px - b[4])
    dy = max(b[5] - py, 0.0, py - b[6])
    return float(np.hypot(dx, dy))


def test_amr_criterion_select(exe, workdir):
    """One criterion per run: on a pure pressure pulse, pressure_gradient must refine the pulse (and only
    its neighbourhood) at once, while density_gradient and vorticity must refine nothing."""
    name = "amr_criterion_select"
    res = Result(name)
    res.runtime, res.outdir = 0.0, None
    for crit, expect_refined in [("pressure_gradient", True), ("density_gradient", False), ("vorticity", False)]:
        code, log, dt, outdir = run_variant(exe, name, [f"amr_criterion {crit}"], workdir)
        res.runtime += dt
        res.check(f"exit code ({crit})", code, code == 0, "= 0")
        if code != 0:
            res.log_tail = "\n".join(log.replace("\r", "\n").splitlines()[-15:])
            return res
        blocks = read_frames(os.path.join(outdir, "grid.bin"))[-1][1]
        lv = max(b[0] for b in blocks)
        if expect_refined:
            res.check(f"{crit}: pulse refined (max level)", lv, lv == 1, "= 1")
            far = max([b[0] for b in blocks if block_distance(b, 0.5, 2.0) > 1.0] or [0])
            res.check(f"{crit}: blocks farther than 1 from the pulse (max level)", far, far == 0, "= 0")
        else:
            res.check(f"{crit}: nothing refined (max level)", lv, lv == 0, "= 0")
    return res


def test_invalid_amr_criterion(exe, workdir):
    """Invalid criterion choices must stop with a Parameter error: an unknown name, contrast without
    well_balanced (no reference), and hook for a problem that defines no refinement hook."""
    name = "invalid_amr_criterion"
    res = Result(name)
    res.runtime, res.outdir = 0.0, None
    for crit in ["steepness", "contrast", "hook"]:
        code, log, dt, outdir = run_variant(exe, name, [f"amr_criterion {crit}"], workdir)
        res.runtime += dt
        res.check(f"non-zero exit code ({crit})", code, code != 0, "!= 0")
        found = "Parameter error" in log
        res.check(f"'Parameter error' message ({crit})", int(found), found, "= 1")
    return res

def wall_speed(frame, gamma):
    """Largest |v|/c_s in the outermost two cell rows/columns: proves a pulse really reached the walls."""
    x, y, Q = stitch(frame)
    rho, u, v, p = primitives(Q, gamma)
    m = np.hypot(u, v)/np.sqrt(gamma*p/rho)
    return float(max(m[:2, :].max(), m[-2:, :].max(), m[:, :2].max(), m[:, -2:].max()))


def test_hydrostatic_wall_reflection(exe, workdir):
    """Without gravity a Hydrostatic wall is a reflecting wall like Closed (they differ only in the second
    ghost layer): the reflected pulse must agree to 1e-3 of its amplitude. A wall flux that ignored the
    velocity into the wall (pressure only) would differ at the percent level."""
    name = "hydrostatic_wall_reflection"
    res = Result(name)
    res.runtime, res.outdir = 0.0, None
    vel = {}
    for wall in ["Hydrostatic", "Closed"]:
        code, log, dt, outdir = run_variant(exe, name, [f"bc_bottom {wall}", f"bc_top {wall}"], workdir)
        res.runtime += dt
        res.check(f"exit code ({wall})", code, code == 0, "= 0")
        if code != 0:
            res.log_tail = "\n".join(log.replace("\r", "\n").splitlines()[-15:])
            return res
        x, y, Q = stitch(read_frames(os.path.join(outdir, "grid.bin"))[-1])
        rho, u, v, p = primitives(Q, 1.4)
        vel[wall] = (u, v)
    du = np.hypot(vel["Hydrostatic"][0] - vel["Closed"][0], vel["Hydrostatic"][1] - vel["Closed"][1]).max()
    amp = np.hypot(*vel["Closed"]).max()
    res.check("max |v_Hydrostatic - v_Closed| / max |v|", du/amp, du/amp < 1e-3, "< 1e-3")
    return res

def mode_amplitude(frame, field, gamma, n=1, diag=False):
    """Amplitude of the Fourier mode n of a field ('u', 'v', 'T' or 's' = p/rho^gamma) on the stitched
    grid: along y (x-averaged), or along the diagonal (phase kx x + ky y) for the diagonal test modes.
    Entropy modes are measured with 's': conduction launches weak sound waves, which carry adiabatic
    temperature fluctuations but no entropy. Sound waves are measured with 'w+' = v + dp/(rho c): with
    diffusion, the inviscid initial wave also seeds a small backward wave, whose beating would distort
    a measurement of v."""
    x, y, Q = stitch(frame)
    rho, u, v, p = primitives(Q, gamma)
    if field == "w+":  # forward acoustic characteristic: only the right-moving sound wave
        f = v + (p - p.mean())/(rho.mean()*np.sqrt(gamma*p.mean()/rho.mean()))
    else:
        f = {"u": u, "v": v, "T": p/rho, "s": p/rho**gamma}[field]
    f = f - f.mean()
    lx = x[-1] - x[0] + (x[1] - x[0])
    ly = y[-1] - y[0] + (y[1] - y[0])
    x0 = x[0] - 0.5*(x[1] - x[0])
    y0 = y[0] - 0.5*(y[1] - y[0])
    X, Y = np.meshgrid(x - x0, y - y0, indexing="ij")
    ph = 2.0*np.pi*n*(X/lx + Y/ly) if diag else 2.0*np.pi*n*Y/ly
    return float(2.0*np.hypot((f*np.sin(ph)).mean(), (f*np.cos(ph)).mean()))


def decay_rate(frames, field, gamma, n=1, diag=False):
    """Exponential decay rate of a mode between the first and the last snapshot."""
    a0 = mode_amplitude(frames[0], field, gamma, n, diag)
    a1 = mode_amplitude(frames[-1], field, gamma, n, diag)
    return float(-np.log(a1/a0)/(frames[-1][0] - frames[0][0]))


def run_frames(exe, name, extra, workdir, res):
    """Run a variant of a case; returns its frames, or None (with the failure recorded in res)."""
    code, log, dt, outdir = run_variant(exe, name, extra, workdir)
    res.runtime += dt
    if code != 0:
        res.check(f"exit code ({' '.join(extra)})", code, False, "= 0")
        res.log_tail = "\n".join(log.replace("\r", "\n").splitlines()[-15:])
        return None
    return read_frames(os.path.join(outdir, "grid.bin"))


def test_diffusion_convergence(exe, name, field, key, coeff, workdir):
    """Convergence of a diagonal mode's decay rate at N = 16, 32, 64. A grid-oblique wave is also damped
    by the hyperbolic scheme's own numerical dissipation, so at each resolution the inviscid (key = 0)
    run's decay is subtracted; the remaining diffusive part must converge at >= 1.8 (Richardson, from
    successive differences) and match the exact rate at N = 64. Block corners use one-sided tangential
    differences, so this also checks that the global order stays near 2."""
    res = Result(name)
    res.runtime, res.outdir = 0.0, None
    rates = []
    for nres in [16, 32, 64]:
        r = []
        for c in [coeff, 0.0]:
            frames = run_frames(exe, name, [f"Nx {nres}", f"Ny {nres}", f"{key} {c}"], workdir, res)
            if frames is None:
                return res
            r.append(decay_rate(frames, field, 1.4, 1, diag=True))
        rates.append(r[0] - r[1])
    order = float(np.log2(abs(rates[0] - rates[1])/abs(rates[1] - rates[2])))
    res.check("convergence order of the diffusive decay rate", order, order >= 1.8, ">= 1.8")
    err = abs(rates[2]/(coeff*2.0*(2.0*np.pi)**2) - 1)
    res.check("diffusive decay rate at N = 64 vs exact", err, err < 5e-3, "< 5e-3 (relative)")
    return res


def test_diffusion_sound_damping(exe, workdir):
    """Sound-wave damping: inviscid run, then nu, nu_b and chi in turn. The extra damping over the
    inviscid (numerical) one must match linear theory, k^2 (4/3 nu + nu_b + (gamma - 1) chi)/2."""
    name = "diffusion_sound_damping"
    res = Result(name)
    res.runtime, res.outdir = 0.0, None
    k2 = (2.0*np.pi)**2
    frames = run_frames(exe, name, [], workdir, res)
    if frames is None:
        return res
    base = decay_rate(frames, "w+", 1.4)
    for extra, theory in [("viscosity_nu 0.01", 0.5*k2*4.0/3.0*0.01), ("viscosity_bulk 0.01", 0.5*k2*0.01),
                          ("conduction_chi 0.01", 0.5*k2*0.4*0.01)]:
        frames = run_frames(exe, name, [extra], workdir, res)
        if frames is None:
            return res
        extra_rate = decay_rate(frames, "w+", 1.4) - base
        err = abs(extra_rate/theory - 1)
        res.check(f"damping from {extra.split()[0]} vs linear theory", err, err < 2e-2, "< 2e-2 (relative)")
    return res


def test_variants_rest(exe, name, variants, workdir, check):
    """Run a case with several integrator variants; check(res, frames, label) records the checks."""
    res = Result(name)
    res.runtime, res.outdir = 0.0, None
    for v in variants:
        frames = run_frames(exe, name, [f"integrator {v}"], workdir, res)
        if frames is None:
            return res
        check(res, frames, v)
    return res


def test_diffusion_wb_pulse(exe, workdir):
    """The 1e-6 pulse on the balanced atmosphere with and without viscosity: the wave must be weaker
    with nu, and the far field must stay at round-off (no spurious motion from the diffusive fluxes)."""
    name = "diffusion_wb_pulse"
    res = Result(name)
    res.runtime, res.outdir = 0.0, None
    wave = {}
    for nu in ["0.01", "0.0"]:
        frames = run_frames(exe, name, [f"viscosity_nu {nu}"], workdir, res)
        if frames is None:
            return res
        x, y, Q = stitch(frames[-1])
        rho, u, v, p = primitives(Q, 1.4)
        X, Y = np.meshgrid(x, y, indexing="ij")
        r = np.hypot(X - 1.0, Y - 2.0)
        mach = np.hypot(u, v)/np.sqrt(1.4*p/rho)
        wave[nu] = float(mach[r < 0.9].max())
        if nu == "0.01":
            far = float(mach[r > 1.7].max())  # viscosity widens the precursor tail slightly
            res.check("far field with nu: max |v|/cs for r > 1.7", far, far < 1e-12, "< 1e-12")
    res.check("wave with nu / without", wave["0.01"]/wave["0.0"], wave["0.01"] < wave["0.0"], "< 1")
    return res

def kol_amplitude(frame, gamma = 1.4):
    """Amplitude of the sin(k y) mode (k = 2 pi / Ly) of the x-averaged x-velocity."""
    x, y, Q = stitch(frame)
    u = (Q[1]/Q[0]).mean(axis = 0)
    ph = 2.0*np.pi*(y - y[0] + 0.5*(y[1] - y[0]))/(y[-1] - y[0] + (y[1] - y[0]))
    return 2.0*float((u*np.sin(ph)).mean())


def test_kol_instability(exe, workdir):
    """Kolmogorov flow in an elongated box with a seeded perturbation: at Re = U/(nu k) = 5 (above the
    critical ~sqrt(2)) the perturbation (v_rms) must grow by > 10x until t = 20, at Re = 0.5 it must decay."""
    name = "kol_instability"
    res = Result(name)
    res.runtime, res.outdir = 0.0, None
    for re_, f in [(5.0, 0.124025), (0.5, 0.0124025)]:
        frames = run_frames(exe, name, [f"kol_F {f}"], workdir, res)
        if frames is None:
            return res
        v0, v1 = [float(np.sqrt(np.mean((stitch(fr)[2][2]/stitch(fr)[2][0])**2))) for fr in (frames[0], frames[-1])]
        if re_ > 1.0:
            res.check("Re = 5: v_rms(t = 20) / v_rms(0)", v1/v0, v1/v0 > 10.0, "> 10")
        else:
            res.check("Re = 0.5: v_rms(t = 20) / v_rms(0)", v1/v0, v1/v0 < 1.0, "< 1")
    return res


def test_kol_reproducible(exe, workdir):
    """The seeded perturbation: the same seed twice must give bit-identical results, another seed must not."""
    name = "kol_reproducible"
    res = Result(name)
    res.runtime, res.outdir = 0.0, None
    q = []
    for extra in [[], [], ["kol_seed 4"]]:
        frames = run_frames(exe, name, extra, workdir, res)
        if frames is None:
            return res
        q.append(stitch(frames[-1])[2])
    same = float(np.max(np.abs(q[0] - q[1])))
    diff = float(np.max(np.abs(q[0] - q[2])))
    res.check("same seed: max |difference|", same, same == 0.0, "= 0")
    res.check("other seed: max |difference|", diff, diff > 1e-6, "> 1e-6")
    return res


def test_kol_scaling_laminar(exe, workdir):
    """Laminar Kolmogorov flow at three forcing amplitudes, analysed with scripts/forcing_scaling.py
    (steady-state u_rms, fit in log-log): in the laminar regime u_rms is proportional to a_rms, p = 1."""
    name = "kol_scaling_laminar"
    res = Result(name)
    res.runtime, res.outdir = 0.0, None
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
    import forcing_scaling
    forces, u, du = [0.005, 0.01, 0.02], [], []
    for f in forces:
        code, log, dt, outdir = run_variant(exe, name, [f"kol_F {f}"], workdir)
        res.runtime += dt
        if code != 0:
            res.check(f"exit code (kol_F {f})", code, False, "= 0")
            return res
        m, sd, n = forcing_scaling.steady_u_rms(outdir)
        u.append(m)
        du.append(sd)
    p, dp, c = forcing_scaling.fit_exponent(np.array(forces)/np.sqrt(2.0), u, du)
    res.check("exponent p of u_rms ~ a_rms^p (laminar)", p, abs(p - 1.0) < 1e-3, "1 +- 1e-3")
    return res

def read_hist(outdir):
    """history.hst of a run: column names (from the '#' header line) and the data as a dict of arrays."""
    path = os.path.join(outdir, "history.hst")
    names = None
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                names = line[1:].split()
    data = np.atleast_2d(np.loadtxt(path))
    return {n: data[:, k] for k, n in enumerate(names)}


def test_hist_interval_independent(exe, workdir):
    """Integrals are accumulated every step, independently of how often lines are written: the rows of
    a run with hist_interval 10 must be bit-identical to the rows of hist_interval 1 at the same times."""
    name = "hist_interval_independent"
    res = Result(name)
    res.runtime, res.outdir = 0.0, None
    h = {}
    for iv in [1, 10]:
        code, log, dt, outdir = run_variant(exe, name, [f"hist_interval {iv}"], workdir)
        res.runtime += dt
        if code != 0:
            res.check(f"exit code (hist_interval {iv})", code, False, "= 0")
            return res
        h[iv] = read_hist(outdir)
    rows = [int(np.nonzero(h[1]["t"] == t)[0][0]) for t in h[10]["t"] if np.any(h[1]["t"] == t)]
    res.check("rows of the sparse history found in the dense one", len(rows), len(rows) == len(h[10]["t"]), f"= {len(h[10]['t'])}")
    worst = max(float(np.max(np.abs(h[10][c] - h[1][c][rows]))) for c in h[1])
    res.check("max |difference| over all columns", worst, worst == 0.0, "= 0")
    return res

def test_outflow_walls(exe, workdir):
    """gas streams towards each wall in turn (cases outflow_left/right/bottom/top), with Outflow
    on both walls of that direction: one wall lets the gas leave, the other must block inflow.
    the four runs are the same problem rotated or mirrored, so after transforming them back into
    the frame of outflow_left they must agree to round-off.
    Catches index/sign mistakes in any wall."""
    res = Result("outflow_walls")
    res.runtime, res.outdir = 0.0, None
    fields = {}
    for case in ["left", "right", "bottom", "top"]:
        code, log, dt, outdir = run_case(exe, "outflow_" + case, workdir)
        res.runtime += dt
        res.check(f"exit code ({case})", code, code == 0, "= 0")
        if code != 0:
            res.log_tail = "\n".join(log.replace("\r", "\n").splitlines()[-15:])
            return res
        x, y, Q = stitch(read_frames(os.path.join(outdir, "grid.bin"))[-1])
        rho, u, v, p = primitives(Q, 1.4)
        # into the frame of outflow_left: flow along -x, arrays indexed [along flow, across]
        if case == "left":
            f = (rho, p, u, v)
        elif case == "right":    # mirror x
            f = (rho[::-1], p[::-1], -u[::-1], v[::-1])
        elif case == "bottom":   # swap x and y
            f = (rho.T, p.T, v.T, u.T)
        else:                    # top: swap x and y, then mirror
            f = (rho.T[::-1], p.T[::-1], -v.T[::-1], u.T[::-1])
        fields[case] = f
    ref = fields["left"]
    for case in ["right", "bottom", "top"]:
        d = max(float(np.max(np.abs(a - b))) for a, b in zip(fields[case], ref))
        res.check(f"{case} vs left (max abs difference)", d, d < 1e-12, "< 1e-12")
    # the right wall of outflow_left blocks inflow: the gas next to it is slowed down clearly
    u_wall = float(np.mean(ref[2][-1]))
    res.check("blocked wall slows the flow: |u_wall + 0.5|", abs(u_wall + 0.5), abs(u_wall + 0.5) > 0.1, "> 0.1")
    return res

def atmosphere_rest(res, frames, gamma, tol):
    """Hydrostatic equilibrium kept at rest: velocities, density change and mass change."""
    x, y, Q = stitch(frames[-1])
    rho, u, v, p = primitives(Q, gamma)
    _, _, Q0 = stitch(frames[0])
    mach = float(np.max(np.hypot(u, v)/np.sqrt(gamma*p/rho)))
    drho = float(np.max(np.abs(rho/Q0[0] - 1)))
    dm = abs(totals(frames[-1])[0]/totals(frames[0])[0] - 1)
    res.check("max |v|/cs", mach, mach < tol, f"< {tol:g}")
    res.check("max |rho/rho0 - 1|", drho, drho < tol, f"< {tol:g}")
    res.check("mass change", dm, dm < tol, f"< {tol:g}")
    return mach


def test_wb_harmonic_axes(exe, workdir):
    """Harmonic potential with a temperature jump, stratified along y and along x (Hydrostatic walls
    bottom/top resp. left/right): both at rest, and the x-run transposed equals the y-run."""
    res = Result("wb_harmonic_axes")
    res.runtime, res.outdir = 0.0, None
    fields = {}
    for case in ["y", "x"]:
        code, log, dt, outdir = run_case(exe, "wb_harmonic_" + case, workdir)
        res.runtime += dt
        res.check(f"exit code ({case})", code, code == 0, "= 0")
        if code != 0:
            res.log_tail = "\n".join(log.replace("\r", "\n").splitlines()[-15:])
            return res
        frames = read_frames(os.path.join(outdir, "grid.bin"))
        atmosphere_rest(res, frames, 1.4, 1e-12)
        fields[case] = stitch(frames[-1])[2]
    d = float(np.max(np.abs(fields["x"].transpose(0, 2, 1) - fields["y"])))
    res.check("x-run transposed vs y-run (max abs difference)", d, d < 1e-12, "< 1e-12")
    return res


def test_case(exe, name, workdir):
    if name == "conv_advect":
        return test_convergence(exe, name, workdir,
                                {"euler": (0.0, 1.3), "vl2": (1.7, 2.3), "rk2": (1.5, 2.3)},
                                [("vl2", 0.25), ("rk2", 0.25)])
    if name == "conv_vortex":
        return test_convergence(exe, name, workdir,
                                {"euler": (0.0, 1.2), "vl2": (1.8, 2.3), "rk2": (1.8, 2.3)},
                                [("vl2", 0.5), ("rk2", 0.5)])
    if name == "hist_interval_independent":
        return test_hist_interval_independent(exe, workdir)
    if name == "kol_instability":
        return test_kol_instability(exe, workdir)
    if name == "kol_reproducible":
        return test_kol_reproducible(exe, workdir)
    if name == "kol_scaling_laminar":
        return test_kol_scaling_laminar(exe, workdir)
    if name == "visc_convergence":
        return test_diffusion_convergence(exe, name, "u", "viscosity_nu", 0.01, workdir)
    if name == "cond_convergence":
        return test_diffusion_convergence(exe, name, "s", "conduction_chi", 0.01, workdir)
    if name == "diffusion_sound_damping":
        return test_diffusion_sound_damping(exe, workdir)
    if name == "diffusion_wb_pulse":
        return test_diffusion_wb_pulse(exe, workdir)
    if name == "visc_integrators":
        def chk(res, frames, v):
            err = abs(decay_rate(frames, "u", 1.4)/(0.01*(2.0*np.pi)**2) - 1)
            res.check(f"shear decay rate vs exp(-nu k^2 t) ({v})", err, err < 5e-3, "< 5e-3 (relative)")
        return test_variants_rest(exe, name, ["euler", "rk2"], workdir, chk)
    if name == "diffusion_wb_integrators":
        def chk(res, frames, v):
            x, y, Q = stitch(frames[-1])
            rho, u, v_, p = primitives(Q, 1.4)
            mach = float(np.max(np.hypot(u, v_)/np.sqrt(1.4*p/rho)))
            res.check(f"max |v|/cs ({v})", mach, mach < 1e-12, "< 1e-12")
        return test_variants_rest(exe, name, ["euler", "rk2"], workdir, chk)
    if name == "hydrostatic_wall_reflection":
        return test_hydrostatic_wall_reflection(exe, workdir)
    if name == "amr_criterion_select":
        return test_amr_criterion_select(exe, workdir)
    if name == "invalid_amr_criterion":
        return test_invalid_amr_criterion(exe, workdir)
    if name == "wb_amr_pulse":
        return test_wb_amr_pulse(exe, workdir)
    if name == "wb_smr_integrators":  # runs its case with euler and rk2
        return test_wb_smr_integrators(exe, workdir)
    if name == "wb_pointmass_integrators":  # runs its case with euler and rk2
        return test_wb_pointmass_integrators(exe, workdir)
    if name == "wb_harmonic_axes":  # runs two cases and compares them with each other
        return test_wb_harmonic_axes(exe, workdir)
    if name == "outflow_walls":  #runs four cases and compares them with each other
        return test_outflow_walls(exe, workdir)
    res = Result(name)
    code, log, dt, outdir = run_case(exe, name, workdir)
    res.runtime = dt
    res.outdir = outdir

    if name == "kol_drag_guard":  # must stop cleanly, naming the too-strong drag
        res.check("non-zero exit code", code, code != 0, "!= 0")
        found = "kol_drag dt" in log
        res.check("message names the drag limit", int(found), found, "= 1")
        return res
    if name.startswith("invalid_"): #these are supposed to fail
        res.check("non-zero exit code", code, code != 0, "!= 0")
        found = "Parameter error" in log
        res.check("'Parameter error' message", int(found), found, "= 1")
        return res

    res.check("exit code", code, code == 0, "= 0")
    if code != 0:
        res.log_tail = "\n".join(log.replace("\r", "\n").splitlines()[-15:])
        return res

    lines = log.replace("\r", "\n").splitlines()
    bad = [l for l in lines if "INVALID" in l.upper() or "ERROR" in l.upper()]

    res.check("no INVALID/ERROR in log", len(bad), not bad, "= 0")
    unused = [l for l in lines if "is not used by this setup" in l]
    if name == "unused_param_warning":  #the typo must be reported (and nothing else)
        found = any("'limitter'" in l for l in unused)
        res.check("warning names 'limitter'", int(found), found, "= 1")
        res.check("no other unused-parameter warnings",
                  len(unused) - int(found), len(unused) == int(found), "= 0")
    else:  #test cases must not contain keys their problem never reads
        res.check("no unused-parameter warnings", len(unused), not unused, "= 0")

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
    elif name in ("sod_amr", "sod_amr_rk2"):
        check_sod(res, frames, gamma, l1_limit=8e-3)
        lv = max(b[0] for b in frames[-1][1])
        res.check("AMR refined (max level)", lv, lv >= 1, ">= 1")
    elif name == "unused_param_warning":
        check_sod(res, frames, gamma, l1_limit=6e-3)  #runs normally, with the default limiter
    elif name == "sod_amr_level2":
        check_sod(res, frames, gamma, l1_limit=8e-3)
        lv = max(b[0] for b in frames[-1][1])
        res.check("max level over all snapshots", lv, lv == 2, "= 2")
    #each option must leave a fingerprint that the default 
    #cannot produce, so a silently ignored option fails the test
    elif name == "sod_vanleer":
        check_sod(res, frames, gamma, l1_limit=2.9e-3) #vl2 van leer for reference
        check_sod_contact_width(res, frames, max_cells=3)
    elif name == "sod_mc":
        check_sod(res, frames, gamma, l1_limit=2.8e-3)
        check_sod_contact_width(res, frames, max_cells=3)
    elif name == "sod_superbee":
        check_sod(res, frames, gamma, l1_limit=4e-3)
        check_sod_contact_width(res, frames, max_cells=2)
    elif name == "sod_primitive":
        check_sod(res, frames, gamma, l1_limit=4e-3)
        code, log, dt, out = run_variant(exe, name, ["reconstruct conserved"], workdir)
        _, _, Qc = stitch(read_frames(os.path.join(out, "grid.bin"))[-1])
        d = float(np.max(np.abs(Q[0] - Qc[0])))
        res.check("differs from conserved reconstruction", d, d > 1e-4, "> 1e-4")
    elif name == "dirichlet_contact_x":
        check_contact(res, frames, gamma, "x", uniform_tol=1e-8)
    elif name == "dirichlet_contact_y":
        check_contact(res, frames, gamma, "y", uniform_tol=1e-8)
    elif name == "contact_primitive":
        check_contact(res, frames, gamma, "x", uniform_tol=1e-12)
    elif name in ("wb_isothermal", "wb_tjump", "wb_tjump_vl2", "wb_tjump_rk2"):
        atmosphere_rest(res, frames, gamma, 1e-12)
    elif name in ("wb_smr_hband", "wb_smr_vband", "wb_smr_tjump", "wb_smr_nested", "wb_smr_pointmass"):
        atmosphere_rest(res, frames, gamma, 1e-12)
        lv = max(b[0] for b in frames[-1][1])
        res.check("refined blocks present (max level)", lv, lv >= 1, ">= 1")
    elif name in ("wb_amr_rest", "wb_reset_smr", "wb_reset_pointmass"):
        if name == "wb_amr_rest":
            lv = max(max(b[0] for b in f[1]) for f in frames)
            res.check("nothing refined at rest (max level, all snapshots)", lv, lv == 0, "= 0")
            if lv != 0:
                return res
        atmosphere_rest(res, frames, gamma, 1e-12)
        if "reset" in name:
            done = "reference equilibrium re-captured" in log
            res.check("the reset happened", int(done), done, "= 1")
            vals = [float(v) for v in re.findall(r"from the closure = ([0-9.eE+-]+)", log)]
            worst = max(vals) if vals else 1.0
            res.check("reset ghost cells = family closure (max rel. deviation)", worst, worst < 1e-14, "< 1e-14")
    elif name == "wb_amr_reset_pulse":
        dm = abs(totals(frames[-1])[0]/totals(frames[0])[0] - 1)
        res.check("mass conservation |dM|/M", dm, dm < 1e-12, "< 1e-12")
        lv = max(b[0] for b in frames[-1][1])
        res.check("AMR refines the pulse (max level)", lv, lv >= 1, ">= 1")
        done = "reference equilibrium re-captured" in log
        res.check("the reset happened", int(done), done, "= 1")
    elif name in ("visc_shear_decay", "visc_shear_walls", "visc_dt_limited"):
        nu = 0.2 if name == "visc_dt_limited" else 0.01  # as in the case files
        err = abs(decay_rate(frames, "u", gamma)/(nu*(2.0*np.pi)**2) - 1)
        tol = 1e-2 if name == "visc_dt_limited" else 5e-3
        res.check("shear decay rate vs exp(-nu k^2 t)", err, err < tol, f"< {tol:g} (relative)")
        if name == "visc_dt_limited":
            found = "limited by diffusion" in log
            res.check("log reports the diffusive time-step limit", int(found), found, "= 1")
    elif name.startswith("hist_"):
        h = read_hist(outdir)
        res.check("history lines written", len(h["t"]), len(h["t"]) >= 2, ">= 2")
        if name == "hist_conservation":
            for c in ["mass", "Etot"]:
                d = float(np.max(np.abs(h[c]/h[c][0] - 1.0)))
                res.check(f"{c} column constant: max relative change", d, d < 1e-12, "< 1e-12")
        elif name == "hist_matches_snapshots":
            worst = 0.0
            for fr in frames:
                k = np.nonzero(h["t"] == fr[0])[0]
                res.check(f"history line at snapshot t = {fr[0]:.4f}", len(k), len(k) == 1, "= 1")
                if len(k) != 1:
                    return res
                tot = totals(fr)
                ek = sum(float((0.5*(a[..., 1]**2 + a[..., 2]**2)/a[..., 0]).sum())*(b[4] - b[3])/b[7]*(b[6] - b[5])/b[8]
                         for b, a in zip(fr[1], fr[2]))
                for c, ref in [("mass", tot[0]), ("mx", tot[1]), ("my", tot[2]), ("Etot", tot[3]), ("Ekin", ek)]:
                    worst = max(worst, abs(h[c][k[0]] - ref)/abs(ref))
            res.check("history vs grid.bin integrals: max relative difference", worst, worst < 1e-12, "< 1e-12")
        elif name == "hist_shear_decay":
            dek = h["Ekin"][-1] - h["Ekin"][0]
            q = abs(h["int_D_visc"][-1]/dek + 1.0)
            res.check("int D_visc vs -(E_kin(t) - E_kin(0))", q, q < 1e-2, "< 1e-2 (relative)")
            # D_visc uses central gradients, the solver's viscous operator compact face gradients: for
            # u ~ sin(k y) their dissipation rates differ by (k dy)^2/4, which is exactly eps_num here
            pred = (2.0*np.pi/64.0)**2/4.0
            e = abs(h["eps_num"][-1]/dek)/pred
            res.check("|eps_num/dE_kin| vs the stencil difference (k dy)^2/4", e, abs(e - 1.0) < 0.05, "1 +- 0.05")
        elif name == "hist_pressure_work":
            amp = float(np.max(np.abs(h["Ekin"] - h["Ekin"][0])))
            res.check("kinetic energy oscillates: max |E_kin - E_kin0| / E_kin max", amp/float(np.max(h["Ekin"])),
                      amp/float(np.max(h["Ekin"])) > 0.5, "> 0.5")
            e = float(np.max(np.abs(h["eps_num"])))/amp
            res.check("max |eps_num| / oscillation amplitude", e, e < 5e-3, "< 5e-3")  # 2.6e-3; 1.1e-2 with 1st-order time integrals
        elif name == "hist_laminar_budget":
            dek = h["Ekin"][-1] - h["Ekin"][0]
            e = abs(h["eps_num"][-1]/dek)
            # bounded by the central-vs-compact stencil difference (k dy)^2/4 = 2.4e-3 (see hist_shear_decay)
            res.check("|eps_num| / kinetic-energy gain", e, e < 3e-3, "< 3e-3")
        elif name == "hist_kolmogorov_eps":
            m = h["t"] >= 24.0
            ts = h["t"][m]
            rate = (h["eps_num"][m][-1] - h["eps_num"][m][0])/(ts[-1] - ts[0])
            q = rate/float(np.mean(h["P_force"][m]))
            res.check("settled state: eps_num rate / forcing power", q, 0.05 < q < 0.3, "in (0.05, 0.3)")
    elif name in ("kol_laminar_transient", "kol_laminar_steady"):
        nu, al, f, k = 0.01, 0.1, 0.01, 2.0*np.pi
        uinf = f/(nu*k*k + al)
        if name == "kol_laminar_transient":
            worst = max(abs(kol_amplitude(fr)/(uinf*(1.0 - np.exp(-(nu*k*k + al)*fr[0]))) - 1.0) for fr in frames[1:])
            res.check("amplitude vs U (1 - exp(-(nu k^2 + alpha) t)), all snapshots",
                      worst, worst < 1e-3, "< 1e-3 (relative)")
            # total energy changes only by forcing work minus drag work (viscous heating stays inside):
            # dE/dt = (rho/2)(F A - alpha A^2) per volume, A(t) = U (1 - exp(-lam t)), rho = 1, box area 1
            lam = nu*k*k + al
            tt = np.linspace(0.0, frames[-1][0], 4001)
            amp = uinf*(1.0 - np.exp(-lam*tt))
            trapezoid = getattr(np, "trapezoid", None) or np.trapz  # NumPy >= 2 renamed trapz
            work = float(trapezoid(0.5*(f*amp - al*amp*amp), tt))
            de = totals(frames[-1])[3] - totals(frames[0])[3]
            res.check("total energy change vs forcing work - drag work",
                      abs(de/work - 1.0), abs(de/work - 1.0) < 2e-3, "< 2e-3 (relative)")
        else:
            a1 = kol_amplitude(frames[-1])
            err = abs(a1/uinf - 1.0)
            res.check("amplitude vs the laminar U = F/(nu k^2 + alpha)", err, err < 1e-3, "< 1e-3 (relative)")
            # forcing work and viscous heating vary along y (~cos^2 k y): without cooling the pressure
            # differences drive a small transverse flow; it must stay far below the flow speed
            vmax = float(np.max(np.abs(stitch(frames[-1])[2][2])))/uinf
            res.check("transverse flow: max |rho v| / U", vmax, vmax < 1e-3, "< 1e-3")
            # steady laminar flow: injection = friction + viscous loss exactly (no numerical dissipation
            # of a stationary shear flow); the check's centred gradients differ from the face-based ones
            # by ~(k dy)^2/6
            x, y, Q = stitch(frames[-1])
            rho, u, v, p = primitives(Q, gamma)
            dy = y[1] - y[0]
            a = f*np.sin(k*(y - y[0] + 0.5*dy))[None, :]
            uy = (np.roll(u, -1, 1) - np.roll(u, 1, 1))/(2.0*dy)
            inj = float((rho*u*a).mean())
            loss = float((al*rho*u*u).mean() + (rho*nu*uy*uy).mean())
            res.check("energy budget: (friction + viscous) / injection",
                      loss/inj, abs(loss/inj - 1.0) < 1e-2, "1 +- 1e-2")
    elif name == "kol_momentum":
        t0_, t1_ = totals(frames[0]), totals(frames[-1])
        err = abs(t1_[1]/t0_[1]/np.exp(-0.1*frames[-1][0]) - 1.0)
        res.check("mean flow: Px(t)/Px(0) vs exp(-alpha t)", err, err < 1e-6, "< 1e-6 (relative)")
        res.check("no y-momentum: |Py|/|Px(0)|", abs(t1_[2])/abs(t0_[1]), abs(t1_[2])/abs(t0_[1]) < 1e-12, "< 1e-12")
    elif name == "kol_cooling":
        x, y, Q = stitch(frames[-1])
        rho, u, v, p = primitives(Q, gamma)
        t = frames[-1][0]
        err = float(np.max(np.abs((p/rho - 1.0)/np.exp(-t/1e-3) - 1.0)))
        res.check("(T - T0)/(T_init - T0) vs exp(-t/tau), tau < dt", err, err < 1e-12, "< 1e-12")
    elif name == "kol_heating_balance":
        dev = max(float(np.max(np.abs(primitives(stitch(fr)[2], gamma)[3]/stitch(fr)[2][0] - 1.0))) for fr in frames)
        res.check("max |T/T0 - 1| over all snapshots", dev, dev < 1e-5, "< 1e-5")
    elif name == "kol_energy_budget":
        # in the settled state: injection rho u.a = friction alpha rho |u|^2 + viscous tau:grad u + numerical
        nu, al, f, k = 0.002, 0.1, 0.053687, 2.0*np.pi
        def budget(fr):
            x, y, Q = stitch(fr)
            rho, u, v, p = primitives(Q, gamma)
            dx, dy = x[1] - x[0], y[1] - y[0]
            a = f*np.sin(k*(y - y[0] + 0.5*dy))[None, :]
            ddx = lambda q: (np.roll(q, -1, 0) - np.roll(q, 1, 0))/(2.0*dx)
            ddy = lambda q: (np.roll(q, -1, 1) - np.roll(q, 1, 1))/(2.0*dy)
            ux, uy, vx, vy = ddx(u), ddy(u), ddx(v), ddy(v)
            div = ux + vy
            return (float((rho*u*a).mean()), float((al*rho*(u*u + v*v)).mean()),
                    float((rho*nu*(2.0*(ux**2 + vy**2) + (uy + vx)**2 - 2.0/3.0*div**2)).mean()),
                    float(np.sqrt(np.mean(v*v))/np.sqrt(np.mean(u*u + v*v))))
        # settled non-laminar state: the remainder is the scheme's numerical dissipation (~13% here)
        st = np.array([budget(fr) for fr in frames if fr[0] >= 24.0])
        closure = (st[:, 1].mean() + st[:, 2].mean())/st[:, 0].mean()
        res.check("settled state is non-laminar: v_rms/u_rms", float(st[:, 3].mean()), st[:, 3].mean() > 0.3, "> 0.3")
        res.check("settled state: (friction + viscous) / injection", float(closure),
                  0.8 < closure < 1.02, "in (0.8, 1.02)")
    elif name == "visc_heating":
        # heating rate mu (du/dy)^2 = rho nu A^2 k^2 cos^2(ky) e^(-2 nu k^2 t): its cos(2ky) part, integrated
        def eint_cos2(f):
            x, y, Q = stitch(f)
            rho, u, v, p = primitives(Q, gamma)
            e = (p/(gamma - 1.0)).mean(axis=0)
            ph = 2.0*np.pi*(y - y[0] + 0.5*(y[1] - y[0]))/(y[-1] - y[0] + (y[1] - y[0]))
            return 2.0*float((e*np.cos(2.0*ph)).mean())
        nu, k, a = 0.01, 2.0*np.pi, 0.1*np.sqrt(gamma)
        t = frames[-1][0]
        pred = 0.5*nu*a*a*k*k*(1.0 - np.exp(-2.0*nu*k*k*t))/(2.0*nu*k*k)
        q = (eint_cos2(frames[-1]) - eint_cos2(frames[0]))/pred
        res.check("viscous heating pattern: measured / predicted cos(2ky) amplitude", q, 0.8<q< 1.2, "in (0.8, 1.2)")
    elif name in ("cond_entropy_mode", "cond_walls"):
        err = abs(decay_rate(frames, "s", gamma)/(0.01*(2.0*np.pi)**2) - 1)
        res.check("entropy-mode decay rate vs exp(-chi k^2 t)", err, err < 5e-3, "< 5e-3 (relative)")
        if name == "cond_walls":
            de = abs(totals(frames[-1])[3]/totals(frames[0])[3] - 1)
            res.check("total energy conserved (insulating walls)", de, de < 1e-12, "< 1e-12")
    elif name in ("diffusion_conservation", "diffusion_smr"):
        t0, t1 = totals(frames[0]), totals(frames[-1])
        for k, label in [(0, "mass"), (3, "energy")]:
            d = abs(t1[k]/t0[k] - 1)
            res.check(f"{label} conservation |dQ|/Q", d, d < 1e-12, "< 1e-12")
        scale = abs(t0[0])*np.sqrt(1.4)
        for k, label in [(1, "x-momentum"), (2, "y-momentum")]:
            d = abs(t1[k] - t0[k])/max(abs(t0[k]), scale)
            res.check(f"{label} conservation", d, d < 1e-12, "< 1e-12 (relative)")
        if name == "diffusion_conservation":
            # conservation alone would also pass if the diffusion did nothing: its kinetic-energy loss
            # (viscous dissipation into heat) must clearly exceed the scheme's own, from an inviscid run
            def vortex_ke(f):
                x, y, Q = stitch(f)
                rho, u, v, p = primitives(Q, gamma)
                return float((0.5*rho*((u - u.mean())**2 + (v - v.mean())**2)).sum())
            loss = 1.0 - vortex_ke(frames[-1])/vortex_ke(frames[0])
            code0, log0, dt0, out0 = run_variant(exe, name, ["viscosity_nu 0", "viscosity_bulk 0", "conduction_chi 0"], workdir)
            f0 = read_frames(os.path.join(out0, "grid.bin"))
            loss0 = 1.0 - vortex_ke(f0[-1])/vortex_ke(f0[0])
            res.check("kinetic-energy loss with diffusion minus without", loss - loss0, loss - loss0 > 1e-2, "> 1e-2")
        if name == "diffusion_smr":
            err = abs(decay_rate(frames, "u", gamma)/(0.01*(2.0*np.pi)**2) - 1)
            res.check("shear decay rate with a refined band", err, err < 1e-2, "< 1e-2 (relative)")
            lv = max(b[0] for b in frames[-1][1])
            res.check("refined blocks present (max level)", lv, lv >= 1, ">= 1")
    elif name in ("diffusion_wb_rest", "diffusion_wb_tjump", "diffusion_wb_pointmass_iso",
                  "diffusion_wb_pointmass_poly", "diffusion_wb_smr", "diffusion_wb_amr"):
        if name == "diffusion_wb_amr":
            lv = max(max(b[0] for b in f[1]) for f in frames)
            res.check("nothing refined at rest (max level, all snapshots)", lv, lv == 0, "= 0")
            if lv != 0:
                return res
        atmosphere_rest(res, frames, gamma, 1e-12)
    elif name == "diffusion_wb_tjump_cond":
        x, y, Q0 = stitch(frames[0])
        x, y, Q1 = stitch(frames[-1])
        T0 = (primitives(Q0, gamma)[3]/Q0[0]).mean(axis=0)
        rho1, u1, v1, p1 = primitives(Q1, gamma)
        T1 = (p1/rho1).mean(axis=0)
        jump = y[int(np.argmax(np.abs(np.gradient(T0))))]
        dT = float(np.max(np.abs(T1/T0 - 1)))
        res.check("heat flows at the jump: max |dT/T|", dT, dT > 1e-4, "> 1e-4")
        dm = abs(totals(frames[-1])[0]/totals(frames[0])[0] - 1)
        res.check("mass conservation |dM|/M", dm, dm < 1e-12, "< 1e-12")
        mach = np.hypot(u1, v1)/np.sqrt(gamma*p1/rho1)
        # beyond the sound waves from the jump (~0.2) and the scheme's exponential precursor tail
        far = float(mach[:, np.abs(y - jump) > 1.3].max())
        res.check("far from the jump (|y - y_jump| > 1.3): max |v|/cs", far, far < 1e-12, "< 1e-12")

    elif name in ("hydrostatic_wall_mass_pm", "hydrostatic_wall_mass_poly", "hydrostatic_wall_mass_atm"):
        dm = abs(totals(frames[-1])[0]/totals(frames[0])[0] - 1)
        res.check("mass conservation |dM|/M", dm, dm < 1e-13, "< 1e-13")
        ws = max(wall_speed(f, gamma) for f in frames)
        res.check("the pulse reached the walls: max |v|/cs next to them", ws, ws > 1e-5, "> 1e-5")
    elif name == "amr_crit_vorticity":
        blocks = frames[-1][1]
        cx, cy = 0.1, 0.1  # vortex centre at t = 0.1 (starts at the origin, advected with u0 = v0 = 1)
        centre = min(b[0] for b in blocks if block_distance(b, cx, cy) == 0.0)
        res.check("blocks at the vortex centre (min level)", centre, centre == 2, "= 2")
        far = max([b[0] for b in blocks if block_distance(b, cx, cy) > 3.0] or [0])
        res.check("blocks farther than 3 from the centre (max level)", far, far == 0, "= 0")
    elif name == "amr_crit_hook":
        blocks = frames[-1][1]
        centre = min(b[0] for b in blocks if block_distance(b, 0.0, 0.0) == 0.0)
        res.check("blocks touching the mass (min level)", centre, centre == 2, "= 2")
        # the hook refines a block whose region touches pm_refine_radius, and refinement splits it into
        # all four children, so every level-2 block's PARENT (level-1 block, 0.5 wide from -2) must touch
        # the radius; level-1 buffer blocks further out come from the mesh's 2:1 level balance
        def parent(b):
            x0 = -2.0 + 0.5*np.floor((b[3] + 2.0)/0.5 + 1e-9)
            y0 = -2.0 + 0.5*np.floor((b[5] + 2.0)/0.5 + 1e-9)
            return (None, None, None, x0, x0 + 0.5, y0, y0 + 0.5)
        outside = sum(1 for b in blocks if b[0] == 2 and block_distance(parent(b), 0.0, 0.0) > 0.5 + 1e-12)
        res.check("level-2 blocks whose parent lies outside pm_refine_radius (count)", outside, outside == 0, "= 0")
        coarse = sum(1 for b in blocks if b[0] == 0)
        res.check("level-0 blocks remain (count)", coarse, coarse > 0, "> 0")
    elif name == "wb_amr_refcheck":
        vals = [float(v) for v in re.findall(r"initial condition = ([0-9.eE+-]+)", log)]
        res.check("regrids checked", len(vals), len(vals) >= 10, ">= 10")
        worst = max(vals) if vals else 1.0
        res.check("max |reference / initial condition - 1|", worst, worst < 1e-14, "< 1e-14")
    elif name == "wb_smr_pulse":
        dm = abs(totals(frames[-1])[0]/totals(frames[0])[0] - 1)
        res.check("mass conservation |dM|/M", dm, dm < 1e-12, "< 1e-12")
        _, _, Qf = stitch(frames[-1])
        rho_, u_, v_, p_ = primitives(Qf, gamma)
        mach = float(np.max(np.hypot(u_, v_)/np.sqrt(gamma*p_/rho_)))
        res.check("the pulse is dynamic: max |v|/cs", mach, mach > 1e-3, "> 1e-3")
    elif name in ("wb_pointmass_iso", "wb_pointmass_poly"):
        atmosphere_rest(res, frames, gamma, 1e-12)
    elif name == "wb_pointmass_linear":  #counter-check: linear is only second order in a 2D potential
        _, _, Qf = stitch(frames[-1])
        rho_, u_, v_, p_ = primitives(Qf, gamma)
        mach = float(np.max(np.hypot(u_, v_)/np.sqrt(gamma*p_/rho_)))
        res.check("max |v|/cs (must drift)", mach, mach > 1e-5, "> 1e-5")
    elif name == "wb_pointmass_pert":
        xx, yy, Qf = stitch(frames[-1])
        rho_, u_, v_, p_ = primitives(Qf, gamma)
        X, Y = np.meshgrid(xx, yy, indexing="ij")
        r = np.hypot(X - 0.8, Y)
        mach = np.hypot(u_, v_)/np.sqrt(gamma*p_/rho_)
        wave, far = float(mach[r < 0.6].max()), float(mach[r > 2.0].max())
        res.check("wave: max |v|/cs for r < 0.6", wave, 1e-8 < wave < 1e-6, "in (1e-8, 1e-6)")
        res.check("far field: max |v|/cs for r > 2.0", far, far < 1e-12, "< 1e-12")
    elif name == "nowb_isothermal":  # the counter-check: without balancing, the equilibrium drifts
        _, _, Q = stitch(frames[-1])
        rho, u, v, p = primitives(Q, gamma)
        mach = float(np.max(np.hypot(u, v)/np.sqrt(gamma*p/rho)))
        res.check("max |v|/cs (must drift)", mach, mach > 1e-6, "> 1e-6")
    elif name == "wb_perturbation":
        x, y, Q = stitch(frames[-1])
        rho, u, v, p = primitives(Q, gamma)
        X, Y = np.meshgrid(x, y, indexing="ij")
        r = np.hypot(X - 1.0, Y - 2.0)
        mach = np.hypot(u, v)/np.sqrt(gamma*p/rho)
        wave, far = float(mach[r < 0.9].max()), float(mach[r > 1.3].max())
        res.check("wave: max |v|/cs for r < 0.9", wave, 1e-8 < wave < 1e-6, "in (1e-8, 1e-6)")
        res.check("far field: max |v|/cs for r > 1.3", far, far < 1e-12, "< 1e-12")
    elif name == "turb_drag":
        t = frames[-1][0]
        (_, b0, a0), (_, b1, a1) = frames[0], frames[-1]
        t0, t1 = totals(frames[0]), totals(frames[-1])
        expect = np.exp(-2.0 * t)                       # turb_drag 2.0
        for k, nm in [(1, "x"), (2, "y")]:
            err = abs(t1[k]/t0[k]/expect - 1)
            res.check(f"momentum {nm}: P(t)/P0 vs exp(-alpha t)", err, err < 1e-12, "< 1e-12 (relative)")
        Q0, Q1 = np.stack(a0), np.stack(a1)
        e0 = Q0[..., 3] - 0.5*(Q0[..., 1]**2 + Q0[..., 2]**2)/Q0[..., 0]
        e1 = Q1[..., 3] - 0.5*(Q1[..., 1]**2 + Q1[..., 2]**2)/Q1[..., 0]
        de = float(np.max(np.abs(e1/e0 - 1)))
        res.check("internal energy unchanged", de, de < 1e-12, "< 1e-12 (relative)")
    elif name in ("turb_solenoidal", "turb_solenoidal_vl2"):
        check_turb(res, frames, gamma, log, comp_range=(0.0, 0.25), mach_range=(0.3, 2.0))
    elif name == "turb_compressive":
        check_turb(res, frames, gamma, log, comp_range=(0.8, 1.0), mach_range=(0.05, 0.5))
    elif name == "sedov_uniform":
        check_sedov(res, frames, gamma, p_amb=1e-5, l1_limit=0.18)
    elif name in ("sedov_amr", "sedov_amr_vl2"):
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
ALL = ["sod_uniform", "sod_smr", "sod_amr", "sod_amr_level2",
       "sod_vanleer", "sod_superbee", "sod_mc", "sod_primitive",
       "dirichlet_contact_x", "dirichlet_contact_y", "contact_primitive",
       "sedov_uniform", "sedov_amr", "disk_equilibrium", "disk_stirred",
       "outflow_walls", "unused_param_warning", "invalid_params",
       "invalid_dirichlet", "invalid_problem", "invalid_number", "invalid_integrator",
       "turb_solenoidal", "turb_compressive", "invalid_turb_bc", "turb_drag",
       "wb_isothermal", "wb_tjump", "nowb_isothermal", "wb_harmonic_axes", "wb_perturbation",
       "invalid_wb_no_potential", "invalid_hydrostatic_no_potential",
       "conv_advect", "conv_vortex", "wb_tjump_vl2", "wb_tjump_rk2", "sedov_amr_vl2", "sod_amr_rk2",
       "turb_solenoidal_vl2", "wb_pointmass_iso", "wb_pointmass_poly", "wb_pointmass_linear",
       "wb_pointmass_integrators", "wb_pointmass_pert", "invalid_wb_equilibrium",
       "wb_smr_hband", "wb_smr_vband", "wb_smr_tjump", "wb_smr_nested", "wb_smr_pointmass",
       "wb_smr_integrators", "wb_smr_pulse", "wb_amr_rest", "wb_amr_pulse", "wb_amr_reset_pulse",
       "wb_reset_smr", "wb_reset_pointmass", "wb_amr_refcheck", "amr_criterion_select",
       "amr_crit_vorticity", "amr_crit_hook", "invalid_amr_criterion", "hydrostatic_wall_mass_pm",
       "hydrostatic_wall_mass_poly", "hydrostatic_wall_mass_atm", "hydrostatic_wall_reflection",
       "visc_shear_decay", "visc_shear_walls", "visc_convergence", "diffusion_sound_damping",
       "visc_integrators", "visc_dt_limited", "cond_entropy_mode", "cond_walls",
       "cond_convergence", "diffusion_conservation", "diffusion_smr", "diffusion_wb_rest",
       "diffusion_wb_tjump", "diffusion_wb_tjump_cond", "diffusion_wb_pointmass_iso", "diffusion_wb_pointmass_poly",
       "diffusion_wb_smr", "diffusion_wb_amr", "diffusion_wb_pulse", "diffusion_wb_integrators", "visc_heating",
       "kol_laminar_transient", "kol_laminar_steady", "kol_momentum", "kol_instability", "kol_cooling",
       "kol_heating_balance", "kol_reproducible", "kol_drag_guard", "invalid_kolmogorov", "kol_energy_budget",
       "kol_scaling_laminar", "hist_conservation", "hist_matches_snapshots", "hist_interval_independent",
       "hist_shear_decay", "hist_pressure_work", "hist_laminar_budget", "hist_kolmogorov_eps",
       "invalid_hist_interval"]


def main():
    ap = argparse.ArgumentParser(description="Szuchydro test suite")
    ap.add_argument("--exe", default="build/szuchydro", help="path to the executable")
    ap.add_argument("-k", default="", help="only run tests whose name contains this")
    ap.add_argument("--keep", action="store_true", help="keep the run directory")
    ap.add_argument("--save-ref", metavar="DIR", help="store final-frame hashes (bit-for-bit mode)")
    ap.add_argument("--compare-ref", metavar="DIR", 
                    help="compare final frames to the references in DIR")
    ap.add_argument("--extra", action="append", default=[], metavar="LINE",
                    help="append this line to every test case, e.g. --extra 'integrator vl2' (repeatable)")
    ap.add_argument("--tol", type=float, default=0.0,
                    help=("with --compare-ref: accept max relative"  
                    "differences up to this (default 0 = bit-for-bit)"))

    a = ap.parse_args()
    EXTRA.extend(a.extra)
    exe = os.path.abspath(a.exe)
    if not os.path.isfile(exe):
        sys.exit(f"executable not found: {exe}")

    names = [n for n in ALL if a.k in n]
    work = tempfile.mkdtemp(prefix="szuchydro_tests_")
    print(f"running {len(names)} test(s) in {work}\n")

    results, hashes, finals = [], {}, {}
    for n in names:
        res = test_case(exe, n, work)
        results.append(res)
        status = "PASS" if res.passed else "FAIL"
        print(f"[{status}] {n}  ({res.runtime:.1f} s)")
        for what, val, ok, lim in res.checks:
            mark = " " if ok else "x"
            v = f"{val:.3e}" if isinstance(val, (float, np.floating)) else str(val)
            print(f"     {mark} {what:38s} {v:>12s}   (need {lim})")
        if hasattr(res, "log_tail"):
            print("    --- last log lines ---\n" + res.log_tail)
        if res.outdir and os.path.isfile(os.path.join(res.outdir, "grid.bin")):
            frames = read_frames(os.path.join(res.outdir, "grid.bin"))
            if frames:  #a crashed run can leave an empty file behind
                hashes[n] = frame_hash(frames[-1])
                finals[n] = frames[-1]


    ok = all(r.passed for r in results) #only if every test passed
    if a.save_ref:
        os.makedirs(a.save_ref, exist_ok=True)
        with open(os.path.join(a.save_ref, "hashes.json"), "w") as f:
            json.dump(hashes, f, indent=2)
        for n, fr in finals.items():
            save_frame_npz(os.path.join(a.save_ref, n + ".npz"), fr)
        print(f"\nsaved {len(hashes)} final-frame hashes and snapshots to {a.save_ref}/")

    if a.compare_ref:
        ref_file = os.path.join(a.compare_ref, "hashes.json")
        if not os.path.isfile(ref_file):
            sys.exit(f"\nno references in {a.compare_ref}: "
                     f"save them first with --save-ref {a.compare_ref}")
        ref = json.load(open(ref_file))
        tol_txt = f" (accepting relative differences <= {a.tol:g})" if a.tol > 0 else ""
        print(f"\nbit-for-bit comparison{tol_txt}:")
        for n in names:
            if n not in ref:
                print(f"    ? {n}: no reference")
                continue
            if hashes.get(n) == ref[n]:
                print(f"    identical  {n}")
                continue
            npz = os.path.join(a.compare_ref, n + ".npz")
            if n not in finals or not os.path.isfile(npz):  #crashed run, or old-style reference
                ok = False
                print(f"    DIFFERENT  {n}")
                continue
            diffs = compare_frames(npz, finals[n])
            if diffs is None:
                ok = False
                print(f"    DIFFERENT  {n}   (mesh structure changed)")
                continue
            within = max(diffs) <= a.tol
            ok &= within
            detail = ", ".join(f"{v} {d:.1e}" for v, d in zip(VAR_NAMES, diffs))
            print(f"    {'within tol' if within else 'DIFFERENT '}  {n} max rel. diff: {detail}")


    if not a.keep:
        shutil.rmtree(work, ignore_errors=True)
    print("\nALL PASSED" if ok else "\nSOME TESTS FAILED")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
