#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
topo_viz.py — Orthographic 3D Topology Visualizer (macOS/Linux, interactive-ready)

Two CLIs:
  • Compact (fast, flags anywhere):
      python topo_viz.py                      # list scenes
      python topo_viz.py interactive live_qwz m=0.5 Nk=81 -dpi=220
      python topo_viz.py figure -savefig=fig.png bz_torus_bulged m=0.5 Nk=85 alpha=0.12 -dpi=220

  • Argparse (typed kwargs):
      python topo_viz.py --list
      python topo_viz.py --scene qwz_curvature --kwargs "{'m':0.5,'Nk':101}" --out fig.png --dpi 240 --show

Interactivity backends (choose ONE per run, set from the shell BEFORE running):
  macOS native (retina):             MPLBACKEND=MacOSX
  Qt (recommended on Linux/macOS):   MPLBACKEND=QtAgg
  Tk (fallback):                     MPLBACKEND=TkAgg

All 3D scenes use: orthographic camera • box aspect (1,1,1) • view angles (deg(phi−1), −45°).

Keyboard in live_* scenes:
  s = save PNG   r = reset   p = spin view on/off
  ←/→ = nudge slider #1    ↑/↓ = nudge slider #2
  a/d = nudge slider #3    w/x = nudge slider #4
  h = print keyboard help

Dependencies: numpy, mpmath, matplotlib (plus optional GUI backend: PyQt6/PyQt5 or Tk)
"""

# --------------------- Imports & Backend ---------------------
from __future__ import annotations
import os, sys, math, argparse, ast, glob, zipfile, warnings
import numpy as np
import mpmath as mp

# Select backend BEFORE pyplot import if user set MPLBACKEND in the shell.
import matplotlib as mpl
# (If MPLBACKEND not set, matplotlib will pick a default; you can still save PNGs headless.)
if "MPLBACKEND" in os.environ and os.environ["MPLBACKEND"]:
    try:
        mpl.use(os.environ["MPLBACKEND"], force=True)
    except Exception as e:
        warnings.warn(f"Could not set backend {os.environ['MPLBACKEND']}: {e}")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.widgets import Slider, Button, RadioButtons

# --------------------- Camera / Utils ---------------------
phi = (1 + 5**0.5) / 2.0
ELEV = math.degrees(phi - 1.0)  # ≈ 35.4°
AZIM = -45.0

def set_ortho(ax):
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=ELEV, azim=AZIM)

def integer_snap(val: float) -> tuple[int, float]:
    ni = int(round(val))
    return ni, abs(val - ni)

def _ensure_parent_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

def _save(fig, out=None, dpi=180, show=False):
    if out:  # explicit file path
        _ensure_parent_dir(out)
        fig.savefig(out, dpi=int(dpi), bbox_inches="tight")
    elif not show:
        out = os.path.join(os.getcwd(), "figure.png")
        _ensure_parent_dir(out)
        fig.savefig(out, dpi=int(dpi), bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return out or "(shown only)"

def _parse_scalar(s):
    if isinstance(s, (int, float, bool)): return s
    if not isinstance(s, str): return s
    sl = s.lower()
    if sl in ("true", "yes", "on"): return True
    if sl in ("false", "no", "off"): return False
    try:
        if any(ch in s for ch in ".eE"):
            return float(s)
        return int(s)
    except ValueError:
        return s

# --------------------- QWZ / FHS core ---------------------
def qwz_h(kx, ky, m):
    sx = np.array([[0,1],[1,0]], dtype=complex)
    sy = np.array([[0,-1j],[1j,0]], dtype=complex)
    sz = np.array([[1,0],[0,-1]], dtype=complex)
    c1 = m + np.cos(kx) + np.cos(ky)
    sxy = np.sin(kx) - 1j*np.sin(ky)
    return np.array([[ c1, sxy ],
                     [ np.conjugate(sxy), -c1 ]], dtype=complex)

def _lowest_eigvec(H):
    w, v = np.linalg.eigh(H)
    return v[:, np.argsort(w)[0]]

def chern_fhs_2d(h, Nk=81, params=None):
    Nk = int(Nk)
    ks = np.linspace(-np.pi, np.pi, Nk, endpoint=False)
    vecs = np.empty((Nk, Nk, 2), dtype=complex)
    for i, kx in enumerate(ks):
        for j, ky in enumerate(ks):
            vecs[i, j, :] = _lowest_eigvec(h(kx, ky, **(params or {})))
    def Ux(i, j):
        z = np.vdot(vecs[i, j, :], vecs[(i+1)%Nk, j, :]); return z/np.abs(z)
    def Uy(i, j):
        z = np.vdot(vecs[i, j, :], vecs[i, (j+1)%Nk, :]); return z/np.abs(z)
    curv = np.zeros((Nk, Nk), dtype=float)
    Fsum = 0.0
    for i in range(Nk):
        for j in range(Nk):
            W = Ux(i,j)*Uy((i+1)%Nk,j)*np.conjugate(Ux(i,(j+1)%Nk))*np.conjugate(Uy(i,j))
            ang = np.angle(W)
            curv[i,j] = ang
            Fsum += ang
    ch = float(Fsum/(2*np.pi))
    return ch, curv

# --------------------- Weierstrass core ---------------------
def wp_lattice_sum(z, N=5):
    z = complex(z)
    if z == 0:
        return np.inf
    s = 1.0/(z*z)
    for m in range(-int(N), int(N)+1):
        for n in range(-int(N), int(N)+1):
            if m==0 and n==0: continue
            w = m + 1j*n
            s += 1.0/((z - w)**2) - 1.0/(w*w)
    return s

def stereographic_to_sphere(w):
    if np.isinf(w) or (isinstance(w, complex) and (abs(w) > 1e12)):
        return np.array([0.0, 0.0, 1.0])
    w = complex(w)
    r2 = (w.real*w.real + w.imag*w.imag)
    denom = r2 + 1.0
    x = 2.0*w.real/denom
    y = 2.0*w.imag/denom
    z = (r2 - 1.0)/denom
    return np.array([x, y, z])

# --------------------- Geometry helpers ---------------------
def _torus_embed_kgrid(Nk=85, R=0.7, r=0.25):
    ks = np.linspace(-np.pi, np.pi, int(Nk), endpoint=False)
    U, V = np.meshgrid(ks + np.pi, ks + np.pi, indexing='ij')
    X = (R + r*np.cos(V)) * np.cos(U)
    Y = (R + r*np.cos(V)) * np.sin(U)
    Z = r * np.sin(V)
    return X, Y, Z

def _torus_xyz(R=0.6, r=0.3, Nu=160, Nv=60):
    u = np.linspace(0, 2*np.pi, int(Nu), endpoint=True)
    v = np.linspace(0, 2*np.pi, int(Nv), endpoint=True)
    U, V = np.meshgrid(u, v, indexing='ij')
    X = (R + r*np.cos(V)) * np.cos(U)
    Y = (R + r*np.cos(V)) * np.sin(U)
    Z = r * np.sin(V)
    return U, V, X, Y, Z

# --------------------- Static Scenes ---------------------
def scene_gamma_volume(n_max=15.0, steps=601, out=None, dpi=180, show=False):
    def vol_unit_ball(n): return mp.pi**(n/2) / mp.gamma(n/2 + 1)
    n_vals = np.linspace(0.0, float(n_max), int(steps))
    V_vals = np.array([vol_unit_ball(float(n)) for n in n_vals], dtype=float)
    n_V_max = float(n_vals[np.argmax(V_vals)])
    fig = plt.figure(figsize=(7.6,4.6)); ax = fig.add_subplot(111)
    ax.plot(n_vals, V_vals, lw=1.75)
    ax.axvline(n_V_max, ls="--", lw=1.0, label=f"peak ≈ n={n_V_max:.2f}")
    ax.set_xlabel("dimension n (continuous)"); ax.set_ylabel("Unit ball volume V_n(1)")
    ax.set_title("Gamma Dial: Unit Ball Volume vs dimension"); ax.legend()
    return _save(fig, out, dpi, show)

def scene_gamma_area(n_max=15.0, steps=601, out=None, dpi=180, show=False):
    def vol_unit_ball(n): return mp.pi**(n/2) / mp.gamma(n/2 + 1)
    def area_unit_sphere(n): return n * vol_unit_ball(n)
    n_vals = np.linspace(0.0, float(n_max), int(steps))
    S_vals = np.array([area_unit_sphere(float(n)) for n in n_vals], dtype=float)
    n_S_max = float(n_vals[np.argmax(S_vals)])
    fig = plt.figure(figsize=(7.6,4.6)); ax = fig.add_subplot(111)
    ax.plot(n_vals, S_vals, lw=1.75)
    ax.axvline(n_S_max, ls="--", lw=1.0, label=f"peak ≈ n={n_S_max:.2f}")
    ax.set_xlabel("dimension n (continuous)"); ax.set_ylabel("Unit sphere area S_{n-1}(1)")
    ax.set_title("Gamma Dial: Unit Sphere Area vs dimension"); ax.legend()
    return _save(fig, out, dpi, show)

def scene_simplex_orthant(n_max=6.0, steps=301, out=None, dpi=180, show=False):
    def vol_orthant_simplex(n): return 1.0 / mp.gamma(n + 1)
    n_vals = np.linspace(0.0, float(n_max), int(steps))
    V = np.array([vol_orthant_simplex(float(n)) for n in n_vals], dtype=float)
    fig = plt.figure(figsize=(7.6,4.6)); ax = fig.add_subplot(111)
    ax.plot(n_vals, V, lw=1.75)
    ax.set_xlabel("dimension n (continuous)"); ax.set_ylabel("Volume")
    ax.set_title("Orthant Simplex Volume vs dimension")
    return _save(fig, out, dpi, show)

def scene_simplex_regular(n_max=6.0, steps=301, out=None, dpi=180, show=False):
    def vol_regular_simplex(n, s=1.0):
        return mp.sqrt(n + 1) / (mp.gamma(n + 1) * (2.0 ** (n/2))) * (s ** n)
    n_vals = np.linspace(0.0, float(n_max), int(steps))
    V = np.array([vol_regular_simplex(float(n), 1.0) for n in n_vals], dtype=float)
    fig = plt.figure(figsize=(7.6,4.6)); ax = fig.add_subplot(111)
    ax.plot(n_vals, V, lw=1.75)
    ax.set_xlabel("dimension n (continuous)"); ax.set_ylabel("Volume")
    ax.set_title("Regular Simplex Volume (side=1)")
    return _save(fig, out, dpi, show)

def scene_qwz_curvature(m=0.5, Nk=81, out=None, dpi=180, show=False):
    ch, curv = chern_fhs_2d(qwz_h, Nk=int(Nk), params={'m':float(m)})
    ci, ce = integer_snap(ch)
    fig = plt.figure(figsize=(6.2,5.5)); ax = fig.add_subplot(111)
    im = ax.imshow(curv.T, origin="lower", extent=[-np.pi, np.pi, -np.pi, np.pi], aspect="equal")
    ax.set_xlabel("k_x"); ax.set_ylabel("k_y")
    ax.set_title(f"QWZ Berry Curvature (m={m}) • Chern ≈ {ch:.6f} → {ci} (|Δ|={ce:.2e})")
    fig.colorbar(im, ax=ax, shrink=0.85)
    return _save(fig, out, dpi, show)

def scene_bz_torus_curvature_cloud(m=0.5, Nk=85, out=None, dpi=180, show=False):
    Nk = int(Nk)
    ch, curv = chern_fhs_2d(qwz_h, Nk=Nk, params={'m':float(m)})
    X, Y, Z = _torus_embed_kgrid(Nk=Nk, R=0.7, r=0.25)
    fig = plt.figure(figsize=(7.2,7.2))
    ax = fig.add_subplot(111, projection='3d'); set_ortho(ax)
    ax.plot_wireframe(X[::6,::3], Y[::6,::3], Z[::6,::3], rstride=1, cstride=1, linewidth=0.35)
    ax.scatter(X.ravel(), Y.ravel(), Z.ravel(), s=6, c=curv.ravel())
    ax.set_title(f"Brillouin Torus with QWZ Curvature • Chern ≈ {ch:.6f}")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    return _save(fig, out, dpi, show)

def scene_bz_torus_bulged(m=0.5, Nk=85, R=0.7, r=0.25, alpha=0.10, out=None, dpi=180, show=False):
    Nk = int(Nk)
    ks = np.linspace(-np.pi, np.pi, Nk, endpoint=False)
    U, V = np.meshgrid(ks + np.pi, ks + np.pi, indexing='ij')
    X = (R + r*np.cos(V)) * np.cos(U)
    Y = (R + r*np.cos(V)) * np.sin(U)
    Z = r * np.sin(V)
    Nx = np.cos(U) * np.cos(V); Ny = np.sin(U) * np.cos(V); Nz = np.sin(V)
    Nn = np.sqrt(Nx*Nx + Ny*Ny + Nz*Nz); Nx,Ny,Nz = Nx/Nn, Ny/Nn, Nz/Nn
    ch, curv = chern_fhs_2d(qwz_h, Nk=Nk, params={'m':float(m)})
    ci, ce = integer_snap(ch)
    curv_norm = curv / (np.max(np.abs(curv)) + 1e-12)
    Xb = X + alpha * curv_norm * Nx
    Yb = Y + alpha * curv_norm * Ny
    Zb = Z + alpha * curv_norm * Nz
    fig = plt.figure(figsize=(7.6,7.6))
    ax = fig.add_subplot(111, projection='3d'); set_ortho(ax)
    ax.plot_wireframe(X[::6,::3], Y[::6,::3], Z[::6,::3], rstride=1, cstride=1, linewidth=0.3)
    ax.plot_wireframe(Xb[::4,::2], Yb[::4,::2], Zb[::4,::2], rstride=1, cstride=1, linewidth=0.9)
    ax.set_title(f"Curvature-Bulged Brillouin Torus (m={m}) • Chern ≈ {ch:.6f} → {ci} (|Δ|={ce:.2e})")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    return _save(fig, out, dpi, show)

def scene_qwz_cylinder_bands(Ny=60, Nk=181, m=1.0, out=None, dpi=180, show=False):
    sx = np.array([[0,1],[1,0]], dtype=complex)
    sy = np.array([[0,-1j],[1j,0]], dtype=complex)
    sz = np.array([[1,0],[0,-1]], dtype=complex)
    kxs = np.linspace(-np.pi, np.pi, int(Nk), endpoint=True)
    evals = np.zeros((int(Nk), 2*int(Ny)))
    edge_score = np.zeros((int(Nk), 2*int(Ny)))
    for idx, kx in enumerate(kxs):
        H0 = np.sin(kx)*sx + (float(m) + np.cos(kx))*sz
        Ty = (-0.5j)*sy + 0.5*sz
        H = np.zeros((2*int(Ny), 2*int(Ny)), dtype=complex)
        for y in range(int(Ny)):
            H[2*y:2*y+2, 2*y:2*y+2] = H0
        for y in range(int(Ny)-1):
            H[2*y:2*y+2, 2*(y+1):2*(y+1)+2] = Ty
            H[2*(y+1):2*(y+1)+2, 2*y:2*y+2] = Ty.conj().T
        w, v = np.linalg.eigh(H)
        evals[idx,:] = w
        left_mask = np.zeros((int(Ny),), dtype=float); left_mask[:3] = 1.0
        right_mask = np.zeros((int(Ny),), dtype=float); right_mask[-3:] = 1.0
        for b in range(2*int(Ny)):
            psi = v[:,b].reshape(int(Ny),2)
            dens = np.sum(np.abs(psi)**2, axis=1)
            lw = float(np.dot(left_mask, dens))
            rw = float(np.dot(right_mask, dens))
            edge_score[idx,b] = lw - rw
    X = np.cos(kxs)[:,None] * np.ones_like(evals)
    Y = np.sin(kxs)[:,None] * np.ones_like(evals)
    Z = evals
    ch, _ = chern_fhs_2d(qwz_h, Nk=61, params={'m':float(m)})
    ci, ce = integer_snap(ch)
    fig = plt.figure(figsize=(8.2,8.2)); ax = fig.add_subplot(111, projection='3d'); set_ortho(ax)
    th = np.linspace(0, 2*np.pi, 180); zc = np.linspace(np.min(Z), np.max(Z), 36)
    Th, Zgrid = np.meshgrid(th, zc, indexing='ij')
    Xsc, Ysc = np.cos(Th), np.sin(Th)
    ax.plot_wireframe(Xsc[::6,::2], Ysc[::6,::2], Zgrid[::6,::2], linewidth=0.25)
    ax.plot(np.cos(th), np.sin(th), 0.0*th, linewidth=1.6)  # E=0 ring
    ax.scatter(X, Y, Z, s=3, c=edge_score)
    ax.set_title(f"QWZ Cylinder Bands (m={m}) • Bulk Chern ≈ {ch:.6f} → {ci} (|Δ|={ce:.2e})\nZero-energy isoplane; color encodes left↔right localization")
    ax.set_xlabel("cos k_x"); ax.set_ylabel("sin k_x"); ax.set_zlabel("Energy")
    return _save(fig, out, dpi, show)

def scene_wp_domain(M=60, N=5, out=None, dpi=180, show=False):
    us = np.linspace(0.0, 1.0, int(M), endpoint=False)
    vs = np.linspace(0.0, 1.0, int(M), endpoint=False)
    mag = np.zeros((int(M), int(M)))
    for i,u in enumerate(us):
        for j,v in enumerate(vs):
            w = wp_lattice_sum(u+1j*v, N=int(N))
            mag[i,j] = 0 if np.isinf(w) else min(10.0, abs(w))
    half_periods = [0+0j, 0.5+0j, 0+0.5j, 0.5+0.5j]
    fig = plt.figure(figsize=(6.4,5.6)); ax = fig.add_subplot(111)
    ax.imshow(mag.T, origin="lower", extent=[0,1,0,1], aspect="equal")
    ax.scatter([p.real for p in half_periods], [p.imag for p in half_periods], s=30)
    ax.set_xlabel("u ∈ [0,1)"); ax.set_ylabel("v ∈ [0,1)")
    ax.set_title("Weierstrass ℘ on Torus Fundamental Domain (indicative |℘(z)|)")
    return _save(fig, out, dpi, show)

def scene_wp_sphere(M=60, N=6, out=None, dpi=180, show=False):
    us = np.linspace(0.0, 1.0, int(M), endpoint=False)
    vs = np.linspace(0.0, 1.0, int(M), endpoint=False)
    pts = []
    for u in us:
        for v in vs:
            w = wp_lattice_sum(u+1j*v, N=int(N))
            pts.append(stereographic_to_sphere(w))
    pts = np.array(pts)
    half_periods = [0+0j, 0.5+0j, 0+0.5j, 0.5+0.5j]
    branch_xyz = []
    for z in half_periods:
        w = wp_lattice_sum(z + 1e-6, N=int(N)+2)
        branch_xyz.append(stereographic_to_sphere(w))
    branch_xyz = np.array(branch_xyz)
    fig = plt.figure(figsize=(6.6,6.6))
    ax = fig.add_subplot(111, projection='3d'); set_ortho(ax)
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=4)
    ax.scatter(branch_xyz[:,0], branch_xyz[:,1], branch_xyz[:,2], s=60)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("Weierstrass ℘ → Riemann Sphere (branch images highlighted)")
    return _save(fig, out, dpi, show)

def scene_wp_sphere_cuts(pairing='diag', N=7, out=None, dpi=180, show=False):
    half_periods = [0+0j, 0.5+0j, 0+0.5j, 0.5+0.5j]
    B = []
    for z in half_periods:
        w = wp_lattice_sum(z + 1e-6, N=int(N))
        B.append(stereographic_to_sphere(w))
    B = np.array(B)
    th = np.linspace(0, np.pi, 64); ph = np.linspace(0, 2*np.pi, 128)
    Th, Ph = np.meshgrid(th, ph, indexing='ij')
    Xs = np.sin(Th)*np.cos(Ph); Ys = np.sin(Th)*np.sin(Ph); Zs = np.cos(Th)
    def great_circle(a, b, num=240):
        a = a/np.linalg.norm(a); b = b/np.linalg.norm(b)
        n = np.cross(a, b); ln = np.linalg.norm(n)
        if ln < 1e-9:
            return np.column_stack([a[0]*np.ones(num), a[1]*np.ones(num), a[2]*np.ones(num)])
        n = n/ln; ang = np.arccos(np.clip(np.dot(a,b), -1.0, 1.0))
        ts = np.linspace(0, ang, num)
        u = a; v = np.cross(n, u); v = v/np.linalg.norm(v)
        P = np.outer(np.cos(ts), u) + np.outer(np.sin(ts), v)
        return P
    pairs = [(0,3),(1,2)] if pairing=='diag' else [(0,1),(2,3)]
    fig = plt.figure(figsize=(7.2,7.2)); ax = fig.add_subplot(111, projection='3d'); set_ortho(ax)
    ax.plot_wireframe(Xs[::4,::4], Ys[::4,::4], Zs[::4,::4], linewidth=0.4)
    ax.scatter(B[:,0], B[:,1], B[:,2], s=60)
    for (i,j) in pairs:
        P = great_circle(B[i], B[j], num=240)
        ax.plot3D(P[:,0], P[:,1], P[:,2], linewidth=2.0)
    ax.set_title(f"Weierstrass Branch-Cut Relocation (pairing = {pairing}) — integers unchanged")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    return _save(fig, out, dpi, show)

def scene_torus_engine(R=0.6, r=0.3, Nu=160, Nv=60, p=2, q=1, out=None, dpi=180, show=False):
    U, V, X, Y, Z = _torus_xyz(R, r, Nu, Nv)
    fig = plt.figure(figsize=(7.4,7.4)); ax = fig.add_subplot(111, projection='3d'); set_ortho(ax)
    ax.plot_wireframe(X[::6,::3], Y[::6,::3], Z[::6,::3], rstride=1, cstride=1, linewidth=0.4)
    v_idx = X.shape[1]//8; u_idx = X.shape[0]//6
    ax.plot3D(X[:,v_idx], Y[:,v_idx], Z[:,v_idx], linewidth=2)
    ax.plot3D(X[u_idx,:], Y[u_idx,:], Z[u_idx,:], linewidth=2)
    for du in range(-3,4):
        idx = (u_idx + du) % X.shape[0]
        ax.plot3D(X[idx,:], Y[idx,:], Z[idx,:], linewidth=(1.0 if du==0 else 0.6))
    t = np.linspace(0, 2*np.pi, 900, endpoint=True)
    u_path = t * int(p); v_path = t * int(q)
    Xpq = (R + r*np.cos(v_path)) * np.cos(u_path)
    Ypq = (R + r*np.cos(v_path)) * np.sin(u_path)
    Zpq = r*np.sin(v_path)
    ax.plot3D(Xpq, Ypq, Zpq, linewidth=2)
    ax.set_title("Embedded Torus: A/B cycles, thin annulus, and (p,q) loop")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    return _save(fig, out, dpi, show)

def scene_torus_defects_ribbon(R=0.7, r=0.25, out=None, dpi=180, show=False):
    def torus_xyz(u, v, R=0.7, r=0.25):
        x = (R + r*np.cos(v)) * np.cos(u)
        y = (R + r*np.cos(v)) * np.sin(u)
        z = r * np.sin(v)
        return np.array([x,y,z])
    def torus_frames(u, v, R=0.7, r=0.25):
        Nx = np.cos(u)*np.cos(v); Ny = np.sin(u)*np.cos(v); Nz = np.sin(v)
        N = np.array([Nx,Ny,Nz]); N = N / np.linalg.norm(N, axis=0)
        xu = np.array([-(R + r*np.cos(v))*np.sin(u),(R + r*np.cos(v))*np.cos(u),0.0*u+0.0])
        Tu = xu / np.linalg.norm(xu, axis=0)
        B = np.cross(Tu, N, axis=0); B = B / np.linalg.norm(B, axis=0)
        return N, Tu, B
    def tube_along_pq(p=2, q=1, eps=0.028, n_theta=10, n_t=700, R=0.7, r=0.25):
        t = np.linspace(0, 2*np.pi, n_t, endpoint=True)
        u = p * t; v = q * t
        P = torus_xyz(u, v, R, r); N, T, B = torus_frames(u, v, R, r)
        theta = np.linspace(0, 2*np.pi, n_theta, endpoint=True)
        Xs, Ys, Zs = [], [], []
        for th in theta:
            offset = eps * (np.cos(th)*B + np.sin(th)*N)
            Q = P + offset
            Xs.append(Q[0,:]); Ys.append(Q[1,:]); Zs.append(Q[2,:])
        return np.array(Xs), np.array(Ys), np.array(Zs)
    Nu,Nv = 180,60
    u_grid = np.linspace(0, 2*np.pi, Nu); v_grid = np.linspace(0, 2*np.pi, Nv)
    U, V = np.meshgrid(u_grid, v_grid, indexing='ij')
    X = (R + r*np.cos(V)) * np.cos(U); Y = (R + r*np.cos(V)) * np.sin(U); Z = r*np.sin(V)
    fig = plt.figure(figsize=(7.6,7.6)); ax = fig.add_subplot(111, projection='3d'); set_ortho(ax)
    ax.plot_wireframe(X[::8,::3], Y[::8,::3], Z[::8,::3], rstride=1, cstride=1, linewidth=0.35)
    def small_sphere(center, rad=0.04, n=16):
        th = np.linspace(0, np.pi, n); ph = np.linspace(0, 2*np.pi, 2*n)
        Th, Ph = np.meshgrid(th, ph, indexing='ij')
        xs = center[0] + rad*np.sin(Th)*np.cos(Ph)
        ys = center[1] + rad*np.sin(Th)*np.sin(Ph)
        zs = center[2] + rad*np.cos(Th)
        return xs, ys, zs
    centers_uv = [(0.8*np.pi, 0.2*np.pi), (1.6*np.pi, 1.3*np.pi)]
    for (uu, vv) in centers_uv:
        c = np.array(torus_xyz(uu, vv, R, r))
        xs, ys, zs = small_sphere(c, rad=0.04, n=16)
        ax.plot_wireframe(xs, ys, zs, rstride=1, cstride=1, linewidth=0.3)
    Xp, Yp, Zp = tube_along_pq(p=2, q=1, eps=0.028, n_theta=10, n_t=500, R=R, r=r)
    for i in range(Xp.shape[0]):
        ax.plot3D(Xp[i,:], Yp[i,:], Zp[i,:], linewidth=0.9)
    ax.set_title("Torus with Defects and a (p,q)-Wilson Ribbon")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    return _save(fig, out, dpi, show)

def scene_pump_bloch(Nk=140, Nt=140, a=0.6, b=0.8, out=None, dpi=180, show=False):
    def dvec(k, t, a=0.6, b=0.8):
        dx = np.cos(k) + a*np.cos(t)
        dy = np.sin(k)
        dz = b*np.sin(t)
        return np.array([dx, dy, dz])
    ks = np.linspace(-np.pi, np.pi, int(Nk), endpoint=False)
    ts = np.linspace(0.0, 2*np.pi, int(Nt), endpoint=False)
    D = []
    for k in ks:
        for t in ts:
            d = dvec(k,t,a=float(a),b=float(b)); n = np.linalg.norm(d)
            if n == 0: continue
            D.append(d/n)
    D = np.array(D)
    th = np.linspace(0, np.pi, 60); ph = np.linspace(0, 2*np.pi, 120)
    Th, Ph = np.meshgrid(th, ph, indexing='ij')
    Xs = np.sin(Th)*np.cos(Ph); Ys = np.sin(Th)*np.sin(Ph); Zs = np.cos(Th)
    fig = plt.figure(figsize=(7.4,7.4)); ax = fig.add_subplot(111, projection='3d'); set_ortho(ax)
    ax.plot_wireframe(Xs[::3,::3], Ys[::3,::3], Zs[::3,::3], linewidth=0.4)
    ax.scatter(D[:,0], D[:,1], D[:,2], s=3)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("Pump Image on Bloch Sphere")
    return _save(fig, out, dpi, show)

def scene_pump_loops(k0=0.3*math.pi, t0=0.7*math.pi, a=0.6, b=0.8, out=None, dpi=180, show=False):
    def d_bcast(k, t, a=0.6, b=0.8):
        k = np.asarray(k); t = np.asarray(t)
        dx = np.cos(k) + a*np.cos(t)
        dy = np.sin(k)
        dz = b*np.sin(t)
        dx = np.broadcast_to(dx, np.broadcast(k, t).shape)
        dy = np.broadcast_to(dy, np.broadcast(k, t).shape)
        dz = np.broadcast_to(dz, np.broadcast(k, t).shape)
        return np.array([dx, dy, dz])
    def normed(V):
        n = np.sqrt((V*V).sum(axis=0)); n[n==0]=1.0; return V/n
    th = np.linspace(0, np.pi, 60); ph = np.linspace(0, 2*np.pi, 120)
    Th, Ph = np.meshgrid(th, ph, indexing='ij')
    Xs = np.sin(Th)*np.cos(Ph); Ys = np.sin(Th)*np.sin(Ph); Zs = np.cos(Th)
    ts = np.linspace(0, 2*np.pi, 600, endpoint=True)
    ks = np.linspace(-np.pi, np.pi, 600, endpoint=True)
    L1 = normed(d_bcast(float(k0), ts, a=float(a), b=float(b)))
    L2 = normed(d_bcast(ks, float(t0), a=float(a), b=float(b)))
    fig = plt.figure(figsize=(7.4,7.4)); ax = fig.add_subplot(111, projection='3d'); set_ortho(ax)
    ax.plot_wireframe(Xs[::3,::3], Ys[::3,::3], Zs[::3,::3], linewidth=0.4)
    ax.plot3D(L1[0,:], L1[1,:], L1[2,:], linewidth=2.0)
    ax.plot3D(L2[0,:], L2[1,:], L2[2,:], linewidth=2.0)
    ax.set_title("Pump on Bloch Sphere: Fixed-k and Fixed-t Loops")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    return _save(fig, out, dpi, show)

def scene_pump_chern(Nk=51, Nt=51, a=0.6, b=0.8, out=None, dpi=180, show=False):
    def pump_h(k, t, a=0.6, b=0.8):
        dx = np.cos(k) + a*np.cos(t); dy = np.sin(k); dz = b*np.sin(t)
        return np.array([[ dz, dx - 1j*dy ], [ dx + 1j*dy, -dz ]], dtype=complex)
    def chern_family(Nk=41, Nt=41):
        ks = np.linspace(-np.pi, np.pi, int(Nk), endpoint=False)
        ts = np.linspace(0.0, 2*np.pi, int(Nt), endpoint=False)
        vecs = np.empty((int(Nk), int(Nt), 2), dtype=complex)
        for i, k in enumerate(ks):
            for j, t in enumerate(ts):
                w, v = np.linalg.eigh(pump_h(k, t, a=float(a), b=float(b)))
                vecs[i, j, :] = v[:, np.argsort(w)][0]
        def U_k(i, j):
            z = np.vdot(vecs[i, j, :], vecs[(i+1)%int(Nk), j, :]); return z / np.abs(z)
        def U_t(i, j):
            z = np.vdot(vecs[i, j, :], vecs[i, (j+1)%int(Nt), :]); return z / np.abs(z)
        F = 0.0
        for i in range(int(Nk)):
            for j in range(int(Nt)):
                W = U_k(i,j)*U_t((i+1)%int(Nk),j)*np.conjugate(U_k(i,(j+1)%int(Nt)))*np.conjugate(U_t(i,j))
                F += np.angle(W)
        return float(F/(2*np.pi))
    C = chern_family(Nk=int(Nk), Nt=int(Nt))
    Ci, Ce = integer_snap(C)
    fig = plt.figure(figsize=(6.0,3.0)); ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.02, 0.7, f"Pump Chern on (k,t): {C:.8f} → {Ci} (|Δ|={Ce:.2e})", fontsize=14)
    ax.text(0.02, 0.35, "Odd→Even suspension: spectral flow equals index", fontsize=10)
    return _save(fig, out, dpi, show)

def scene_hopf_link(out=None, dpi=180, show=False):
    def hopf_zs(eta, xi, phi):
        z1 = np.cos(xi/2.0) * np.exp(1j*(eta + phi)/2.0)
        z2 = np.sin(xi/2.0) * np.exp(1j*(eta - phi)/2.0)
        return z1, z2
    def stereographic_S3_to_R3(z1, z2):
        a = np.real(z1); b = np.imag(z1); c = np.real(z2); d = np.imag(z2)
        X1, X2, X3, X4 = a, b, c, d
        denom = 1.0 - X4; denom[denom==0] = 1e-12
        x = X1/denom; y = X2/denom; z = X3/denom
        return x, y, z
    def hopf_fiber_xyz(xi, phi, n=800):
        eta = np.linspace(0, 2*np.pi, n, endpoint=True)
        z1, z2 = hopf_zs(eta, xi, phi); x, y, z = stereographic_S3_to_R3(z1, z2)
        return x, y, z
    xi = np.pi/2; phi1, phi2 = 0.0, np.pi/2
    x1, y1, z1 = hopf_fiber_xyz(xi, phi1, n=800)
    x2, y2, z2 = hopf_fiber_xyz(xi, phi2, n=800)
    phis = np.linspace(0, 2*np.pi, 10, endpoint=False)
    fig = plt.figure(figsize=(7.4,7.4)); ax = fig.add_subplot(111, projection='3d'); set_ortho(ax)
    for ph in phis:
        xs, ys, zs = hopf_fiber_xyz(xi, ph, n=300)
        ax.plot3D(xs, ys, zs, linewidth=0.4)
    ax.plot3D(x1, y1, z1, linewidth=2.0)
    ax.plot3D(x2, y2, z2, linewidth=2.0)
    ax.set_title("Hopf Fibration: Linked Fibers (stereographic)")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    return _save(fig, out, dpi, show)

def scene_padic_tree(p=3, depth=5, out=None, dpi=180, show=False):
    def padic_tree_edges(p=3, depth=5):
        edges = []; levels = [[(0,0)]]; node_id = 1
        for d in range(1, int(depth)+1):
            prev = levels[-1]; level = []
            for (nid, _) in prev:
                for k in range(int(p)):
                    level.append((node_id, nid)); edges.append((nid, node_id)); node_id += 1
            levels.append(level)
        return edges, levels
    def radial_layout(levels, R0=0.1, dR=0.2):
        coords = {}
        for depth, level in enumerate(levels):
            n = len(level); R = R0 + dR*depth
            for i, (nid, _) in enumerate(level):
                theta = 2*np.pi * i / max(1, n)
                coords[nid] = (R*np.cos(theta), R*np.sin(theta))
        coords[0] = (0.0, 0.0); return coords
    edges, levels = padic_tree_edges(p=int(p), depth=int(depth)); coords = radial_layout(levels)
    fig = plt.figure(figsize=(7.2,7.2)); ax = fig.add_subplot(111)
    for (u,v) in edges:
        x0,y0 = coords[u]; x1,y1 = coords[v]; ax.plot([x0,x1],[y0,y1], linewidth=0.8)
    xs = [coords[nid][0] for lvl in levels for (nid,_) in lvl] + [0.0]
    ys = [coords[nid][1] for lvl in levels for (nid,_) in lvl] + [0.0]
    ax.scatter(xs, ys, s=12)
    rings = [0.1 + 0.2*d for d in range(len(levels))]; th = np.linspace(0, 2*np.pi, 180)
    for R in rings: ax.plot(R*np.cos(th), R*np.sin(th), linewidth=0.6)
    ax.set_aspect("equal", adjustable="box"); ax.set_title(f"p-adic Toy: Clopen Shells (p={p}, depth={depth})")
    return _save(fig, out, dpi, show)

# --------------------- LIVE / INTERACTIVE Scenes ---------------------
class SpinController:
    """Toggleable azimuth spin for 3D axes."""
    def __init__(self, fig, ax, d_az=0.6, interval=30):
        self.fig, self.ax = fig, ax
        self.running = False
        self.d_az = float(d_az)
        self.timer = fig.canvas.new_timer(interval=int(interval))
        def _tick():
            try:
                self.ax.view_init(self.ax.elev, self.ax.azim + self.d_az)
                self.fig.canvas.draw_idle()
            except Exception:
                pass
        self.timer.add_callback(_tick)
    def toggle(self):
        self.running = not self.running
        if self.running: self.timer.start()
        else: self.timer.stop()

def _help_keys():
    print(
        "Keys:  s=save  r=reset  p=spin  ←/→ slider#1  ↑/↓ slider#2  a/d slider#3  w/x slider#4  h=help",
        flush=True
    )

def _connect_keys(fig, sliders, save_cb, reset_cb, spin: SpinController|None):
    step = [getattr(s, "valstep", None) or 0.01 for s in sliders] + [0.01, 0.01, 0.01, 0.01]
    def nudge(i, sign):
        if i >= len(sliders): return
        s = sliders[i]; val = float(s.val)
        s.set_val(val + sign * step[i])
    def on_key(evt):
        k = (evt.key or "").lower()
        if k == "s": save_cb()
        elif k == "r": reset_cb()
        elif k == "p" and spin is not None: spin.toggle()
        elif k == "h": _help_keys()
        elif k == "left": nudge(0, -1)
        elif k == "right": nudge(0, +1)
        elif k == "up": nudge(1, +1)
        elif k == "down": nudge(1, -1)
        elif k == "a": nudge(2, -1)
        elif k == "d": nudge(2, +1)
        elif k == "w": nudge(3, +1)
        elif k == "x": nudge(3, -1)
    fig.canvas.mpl_connect("key_press_event", on_key)

def scene_live_qwz(m=0.5, Nk=81, out=None, dpi=180, show=True):
    m0 = float(m); Nk0 = int(Nk)
    ch, curv = chern_fhs_2d(qwz_h, Nk=Nk0, params={'m':m0})
    fig = plt.figure(figsize=(7.2,6.0)); ax = fig.add_subplot(111)
    im = ax.imshow(curv.T, origin="lower", extent=[-np.pi, np.pi, -np.pi, np.pi], aspect="equal")
    ci, ce = integer_snap(ch)
    ttl = ax.set_title(f"QWZ Curvature (m={m0:.3f}, Nk={Nk0}) • Chern≈{ch:.6f}→{ci} (|Δ|={ce:.2e})")
    ax.set_xlabel("k_x"); ax.set_ylabel("k_y"); fig.colorbar(im, ax=ax, shrink=0.85)
    # Sliders
    ax_m = plt.axes([0.15, 0.02, 0.6, 0.03])
    ax_n = plt.axes([0.15, 0.06, 0.6, 0.03])
    s_m = Slider(ax_m, "m", -3.0, 3.0, valinit=m0, valstep=0.01)
    s_n = Slider(ax_n, "Nk", 21, 201, valinit=Nk0, valstep=2)
    def _update(_):
        mm = float(s_m.val); NN = int(s_n.val)
        C, F = chern_fhs_2d(qwz_h, Nk=NN, params={'m':mm})
        im.set_data(F.T)
        ci, ce = integer_snap(C)
        ttl.set_text(f"QWZ Curvature (m={mm:.3f}, Nk={NN}) • Chern≈{C:.6f}→{ci} (|Δ|={ce:.2e})")
        fig.canvas.draw_idle()
    s_m.on_changed(_update); s_n.on_changed(_update)
    # Buttons
    ax_b = plt.axes([0.80, 0.02, 0.15, 0.07]); btn = Button(ax_b, "Save PNG")
    def _save_btn(event=None):
        mm, NN = float(s_m.val), int(s_n.val)
        path = out or f"fig_live_qwz_m{mm:+.2f}_Nk{NN}.png"
        _ensure_parent_dir(path); fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
        print("Saved:", path)
    btn.on_clicked(_save_btn)
    def _reset():
        s_m.reset(); s_n.reset()
    _connect_keys(fig, [s_m, s_n], _save_btn, _reset, spin=None)
    if show: plt.show()
    else: return _save(fig, out, dpi, show=False)

def scene_live_bz_torus_bulged(m=0.5, Nk=85, alpha=0.10, out=None, dpi=180, show=True):
    m0 = float(m); Nk0 = int(Nk); a0 = float(alpha)
    ks = np.linspace(-np.pi, np.pi, Nk0, endpoint=False)
    U, V = np.meshgrid(ks + np.pi, ks + np.pi, indexing='ij')
    X = (0.7 + 0.25*np.cos(V)) * np.cos(U)
    Y = (0.7 + 0.25*np.cos(V)) * np.sin(U)
    Z = 0.25 * np.sin(V)
    fig = plt.figure(figsize=(7.8,7.8)); ax = fig.add_subplot(111, projection='3d'); set_ortho(ax)
    ax.plot_wireframe(X[::6,::3], Y[::6,::3], Z[::6,::3], rstride=1, cstride=1, linewidth=0.3)
    ch, curv = chern_fhs_2d(qwz_h, Nk=Nk0, params={'m':m0})
    Nx = np.cos(U) * np.cos(V); Ny = np.sin(U) * np.cos(V); Nz = np.sin(V)
    Nn = np.sqrt(Nx*Nx + Ny*Ny + Nz*Nz); Nx,Ny,Nz = Nx/Nn, Ny/Nn, Nz/Nn
    curv_norm = curv / (np.max(np.abs(curv)) + 1e-12)
    Xb = X + a0 * curv_norm * Nx; Yb = Y + a0 * curv_norm * Ny; Zb = Z + a0 * curv_norm * Nz
    ax.plot_wireframe(Xb[::4,::2], Yb[::4,::2], Zb[::4,::2], rstride=1, cstride=1, linewidth=0.9)
    ci, ce = integer_snap(ch)
    ttl = ax.set_title(f"Bulged BZ Torus (m={m0:.2f}, Nk={Nk0}, α={a0:.2f}) • Chern≈{ch:.6f}→{ci} (|Δ|={ce:.2e})")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    # Sliders
    ax_m = plt.axes([0.15, 0.02, 0.46, 0.03])
    ax_n = plt.axes([0.15, 0.06, 0.46, 0.03])
    ax_a = plt.axes([0.65, 0.06, 0.29, 0.03])
    s_m = Slider(ax_m, "m", -3.0, 3.0, valinit=m0, valstep=0.01)
    s_n = Slider(ax_n, "Nk", 25, 181, valinit=Nk0, valstep=2)
    s_a = Slider(ax_a, "alpha", 0.0, 0.25, valinit=a0, valstep=0.005)
    spin = SpinController(fig, ax)
    def _redraw(mm, NN, aa):
        ks = np.linspace(-np.pi, np.pi, NN, endpoint=False)
        U, V = np.meshgrid(ks + np.pi, ks + np.pi, indexing='ij')
        X = (0.7 + 0.25*np.cos(V)) * np.cos(U)
        Y = (0.7 + 0.25*np.cos(V)) * np.sin(U)
        Z = 0.25 * np.sin(V)
        ch, curv = chern_fhs_2d(qwz_h, Nk=NN, params={'m':mm})
        Nx = np.cos(U) * np.cos(V); Ny = np.sin(U) * np.cos(V); Nz = np.sin(V)
        Nn = np.sqrt(Nx*Nx + Ny*Ny + Nz*Nz); Nx,Ny,Nz = Nx/Nn, Ny/Nn, Nz/Nn
        curv_norm = curv / (np.max(np.abs(curv)) + 1e-12)
        Xb = X + aa * curv_norm * Nx; Yb = Y + aa * curv_norm * Ny; Zb = Z + aa * curv_norm * Nz
        ax.collections.clear()
        ax.plot_wireframe(X[::6,::3], Y[::6,::3], Z[::6,::3], rstride=1, cstride=1, linewidth=0.3)
        ax.plot_wireframe(Xb[::4,::2], Yb[::4,::2], Zb[::4,::2], rstride=1, cstride=1, linewidth=0.9)
        ci, ce = integer_snap(ch)
        ttl.set_text(f"Bulged BZ Torus (m={mm:.2f}, Nk={NN}, α={aa:.2f}) • Chern≈{ch:.6f}→{ci} (|Δ|={ce:.2e})")
        fig.canvas.draw_idle()
    def _update(_):
        _redraw(float(s_m.val), int(s_n.val), float(s_a.val))
    for s in (s_m, s_n, s_a): s.on_changed(_update)
    # Save
    ax_b = plt.axes([0.65, 0.02, 0.29, 0.03]); btn = Button(ax_b, "Save PNG")
    def _save_btn(event=None):
        mm, NN, aa = float(s_m.val), int(s_n.val), float(s_a.val)
        path = out or f"fig_live_bz_bulged_m{mm:+.2f}_Nk{NN}_a{aa:.2f}.png"
        _ensure_parent_dir(path); fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
        print("Saved:", path)
    btn.on_clicked(_save_btn)
    def _reset():
        s_m.reset(); s_n.reset(); s_a.reset()
    _connect_keys(fig, [s_m, s_n, s_a], _save_btn, _reset, spin)
    if show: plt.show()
    else: return _save(fig, out, dpi, show=False)

def scene_live_pump(a=0.6, b=0.8, k0=0.9, t0=2.1, out=None, dpi=180, show=True):
    a0=float(a); b0=float(b); k00=float(k0); t00=float(t0)
    def d_bcast(k, t, a=0.6, b=0.8):
        k = np.asarray(k); t = np.asarray(t)
        dx = np.cos(k) + a*np.cos(t); dy = np.sin(k); dz = b*np.sin(t)
        dx = np.broadcast_to(dx, np.broadcast(k, t).shape)
        dy = np.broadcast_to(dy, np.broadcast(k, t).shape)
        dz = np.broadcast_to(dz, np.broadcast(k, t).shape)
        return np.array([dx, dy, dz])
    def normed(V):
        n = np.sqrt((V*V).sum(axis=0)); n[n==0]=1.0; return V/n
    th = np.linspace(0, np.pi, 64); ph = np.linspace(0, 2*np.pi, 128)
    Th, Ph = np.meshgrid(th, ph, indexing='ij')
    Xs = np.sin(Th)*np.cos(Ph); Ys = np.sin(Th)*np.sin(Ph); Zs = np.cos(Th)
    ts = np.linspace(0, 2*np.pi, 600, endpoint=True)
    ks = np.linspace(-np.pi, np.pi, 600, endpoint=True)
    fig = plt.figure(figsize=(7.6,7.6)); ax = fig.add_subplot(111, projection='3d'); set_ortho(ax)
    ax.plot_wireframe(Xs[::4,::4], Ys[::4,::4], Zs[::4,::4], linewidth=0.4)
    L1 = normed(d_bcast(k00, ts, a=a0, b=b0)); L2 = normed(d_bcast(ks, t00, a=a0, b=b0))
    l1, = ax.plot3D(L1[0,:], L1[1,:], L1[2,:], linewidth=2.0)
    l2, = ax.plot3D(L2[0,:], L2[1,:], L2[2,:], linewidth=2.0)
    ttl = ax.set_title(f"Pump Loops (a={a0:.2f}, b={b0:.2f}, k0={k00:.2f}, t0={t00:.2f})")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    # Sliders
    ax_a = plt.axes([0.13, 0.02, 0.33, 0.03])
    ax_b = plt.axes([0.13, 0.06, 0.33, 0.03])
    ax_k = plt.axes([0.53, 0.02, 0.33, 0.03])
    ax_t = plt.axes([0.53, 0.06, 0.33, 0.03])
    s_a = Slider(ax_a, "a", 0.0, 1.2, valinit=a0, valstep=0.01)
    s_b = Slider(ax_b, "b", 0.0, 1.2, valinit=b0, valstep=0.01)
    s_k = Slider(ax_k, "k0", -math.pi, math.pi, valinit=k00, valstep=0.01)
    s_t = Slider(ax_t, "t0", 0.0, 2*math.pi, valinit=t00, valstep=0.01)
    spin = SpinController(fig, ax)
    def _update(_):
        aa, bb, kk, tt = map(float, (s_a.val, s_b.val, s_k.val, s_t.val))
        L1 = normed(d_bcast(kk, ts, a=aa, b=bb))
        L2 = normed(d_bcast(ks, tt, a=aa, b=bb))
        l1.set_data_3d(L1[0,:], L1[1,:], L1[2,:])
        l2.set_data_3d(L2[0,:], L2[1,:], L2[2,:])
        ttl.set_text(f"Pump Loops (a={aa:.2f}, b={bb:.2f}, k0={kk:.2f}, t0={tt:.2f})")
        fig.canvas.draw_idle()
    for s in (s_a, s_b, s_k, s_t): s.on_changed(_update)
    ax_btn = plt.axes([0.88, 0.02, 0.09, 0.07]); btn = Button(ax_btn, "Save PNG")
    def _save_btn(event=None):
        aa, bb, kk, tt = map(float, (s_a.val, s_b.val, s_k.val, s_t.val))
        path = out or f"fig_live_pump_a{aa:.2f}_b{bb:.2f}_k{kk:.2f}_t{tt:.2f}.png"
        _ensure_parent_dir(path); fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
        print("Saved:", path)
    btn.on_clicked(_save_btn)
    def _reset():
        s_a.reset(); s_b.reset(); s_k.reset(); s_t.reset()
    _connect_keys(fig, [s_a, s_b, s_k, s_t], _save_btn, _reset, spin)
    if show: plt.show()
    else: return _save(fig, out, dpi, show=False)

def scene_live_wp(N=6, M=60, out=None, dpi=180, show=True):
    N0=int(N); M0=int(M)
    us = np.linspace(0.0, 1.0, M0, endpoint=False)
    vs = np.linspace(0.0, 1.0, M0, endpoint=False)
    mag = np.zeros((M0, M0))
    for i,u in enumerate(us):
        for j,v in enumerate(vs):
            w = wp_lattice_sum(u+1j*v, N=N0)
            mag[i,j] = 0 if np.isinf(w) else min(10.0, abs(w))
    fig = plt.figure(figsize=(7.0,6.0)); ax = fig.add_subplot(111)
    im = ax.imshow(mag.T, origin="lower", extent=[0,1,0,1], aspect="equal")
    half_periods = [0+0j, 0.5+0j, 0+0.5j, 0.5+0.5j]
    ax.scatter([p.real for p in half_periods], [p.imag for p in half_periods], s=40)
    ax.set_title(f"Weierstrass ℘ Fundamental Domain (N={N0}, M={M0})")
    ax.set_xlabel("u"); ax.set_ylabel("v")
    ax_r = plt.axes([0.78, 0.62, 0.18, 0.18]); rb = RadioButtons(ax_r, labels=("diag","adj"), active=0)
    lab = ax.text(0.02, 1.02, "pair: diag", transform=ax.transAxes)
    def _pair(label):
        lab.set_text(f"pair: {label}")
        fig.canvas.draw_idle()
    rb.on_clicked(_pair)
    ax_s = plt.axes([0.12, 0.02, 0.7, 0.03]); sN = Slider(ax_s, "sum N", 3, 12, valinit=N0, valstep=1)
    def _update(_):
        NN = int(sN.val)
        for i,u in enumerate(us):
            for j,v in enumerate(vs):
                w = wp_lattice_sum(u+1j*v, N=NN)
                mag[i,j] = 0 if np.isinf(w) else min(10.0, abs(w))
        im.set_data(mag.T); ax.set_title(f"Weierstrass ℘ Fundamental Domain (N={NN}, M={M0})")
        fig.canvas.draw_idle()
    sN.on_changed(_update)
    ax_b = plt.axes([0.83, 0.05, 0.13, 0.07]); btn = Button(ax_b, "Save PNG")
    def _save_btn(event=None):
        path = out or f"fig_live_wp_N{int(sN.val)}.png"
        _ensure_parent_dir(path); fig.savefig(path, dpi=int(dpi), bbox_inches="tight")
        print("Saved:", path)
    btn.on_clicked(_save_btn)
    def _reset(): sN.reset()
    _connect_keys(fig, [sN], _save_btn, _reset, spin=None)
    if show: plt.show()
    else: return _save(fig, out, dpi, show=False)

# --------------------- Scene registry / API ---------------------
SCENES = {
    # static
    "gamma_volume": scene_gamma_volume,
    "gamma_area": scene_gamma_area,
    "simplex_orthant": scene_simplex_orthant,
    "simplex_regular": scene_simplex_regular,
    "qwz_curvature": scene_qwz_curvature,
    "bz_torus_curvature_cloud": scene_bz_torus_curvature_cloud,
    "bz_torus_bulged": scene_bz_torus_bulged,
    "qwz_cylinder_bands": scene_qwz_cylinder_bands,
    "wp_domain": scene_wp_domain,
    "wp_sphere": scene_wp_sphere,
    "wp_sphere_cuts": scene_wp_sphere_cuts,
    "torus_engine": scene_torus_engine,
    "torus_defects_ribbon": scene_torus_defects_ribbon,
    "pump_bloch": scene_pump_bloch,
    "pump_loops": scene_pump_loops,
    "pump_chern": scene_pump_chern,
    "hopf_link": scene_hopf_link,
    "padic_tree": scene_padic_tree,
    # live
    "live_qwz": scene_live_qwz,
    "live_bz_torus_bulged": scene_live_bz_torus_bulged,
    "live_pump": scene_live_pump,
    "live_wp": scene_live_wp,
}

def list_scenes():
    return sorted(SCENES.keys())

def render(scene_name: str, out: str|None=None, dpi: int=180, show: bool=False, **kwargs):
    if scene_name not in SCENES:
        raise KeyError(f"Unknown scene '{scene_name}'. Available: {', '.join(list_scenes())}")
    fn = SCENES[scene_name]
    return fn(out=out, dpi=int(dpi), show=bool(show), **kwargs)

# --------------------- Portfolio helpers ---------------------
from matplotlib.backends.backend_pdf import PdfPages

def export_portfolio_pdf(png_glob="fig_*.png", pdf_path="Topological_Visual_Portfolio.pdf"):
    pngs = sorted(glob.glob(png_glob))
    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(8.5,11)); ax = fig.add_subplot(111); ax.axis("off")
        ax.text(0.05, 0.92, "Topological Visual Portfolio (Orthographic 3D)", fontsize=16, weight="bold")
        ax.text(0, 0.88, "Camera: orthographic • box aspect (1,1,1) • view (deg(φ−1), −45°)", fontsize=10)
        ax.text(0.05, 0.83, "Contains figures matching:", fontsize=10)
        ax.text(0.07, 0.80, png_glob, fontsize=10, style="italic")
        pdf.savefig(fig); plt.close(fig)
        for p in pngs:
            img = plt.imread(p)
            fig = plt.figure(figsize=(10,7.5)); ax = fig.add_subplot(111); ax.axis("off")
            ax.imshow(img); ax.set_title(os.path.basename(p), fontsize=10)
            pdf.savefig(fig); plt.close(fig)
    return pdf_path

def export_portfolio_zip(png_glob="fig_*.png", zip_path="Topological_Visual_Portfolio_pngs.zip"):
    pngs = sorted(glob.glob(png_glob))
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in pngs: zf.write(p, arcname=os.path.basename(p))
    return zip_path

# --------------------- Compact CLI (fast & order-agnostic) ---------------------
def _is_flag(tok: str) -> bool:
    return tok.startswith("-") or tok.startswith("--")

def _split_kv(tok: str):
    if "=" not in tok: return tok, None
    k, v = tok.split("=", 1); return k, _parse_scalar(v)

def parse_compact_cli(argv):
    """
    Grammar:
      [figure|interactive|help] [flags...] <scene> [flags...] [k=v ...]
    Recognized flags:
      -savefig=PATH | savefig=PATH    -> output filepath
      -show | --show | -pop           -> interactive window
      -dpi=INT | dpi=INT              -> DPI
    """
    if not argv: return {"mode":"list"}
    mode = "figure"; out = None; show = None; dpi = None; scene = None; kwargs = {}
    # pass 1: mode+scene
    for tok in argv:
        low = tok.lower()
        if low in ("figure","fig","interactive","help"):
            mode = "interactive" if low=="interactive" else ("figure" if low in ("figure","fig") else low)
            continue
        if not _is_flag(tok) and "=" not in tok and scene is None:
            scene = tok
    # pass 2: flags+kwargs anywhere
    for tok in argv:
        low = tok.lower()
        if low in ("figure","fig","interactive","help") or tok == scene:
            continue
        if _is_flag(tok) or ("=" in tok and low.split("=",1)[0] in ("savefig","dpi","show","out")):
            k,v = _split_kv(low.lstrip("-"))
            if k in ("savefig","out"):
                out = tok.split("=",1)[1] if "=" in tok else out
            elif k in ("show","pop"):
                show = True if v is None else bool(v)
            elif k == "dpi":
                try: dpi = int(tok.split("=",1)[1])
                except: pass
        elif "=" in tok:
            k,v = _split_kv(tok)
            if v is not None: kwargs[k]=v
    if scene is None: return {"mode":"list"}
    if show is None: show = (mode=="interactive")
    if dpi is None: dpi = 180
    return dict(mode=mode, scene=scene, out=out, show=show, dpi=dpi, kwargs=kwargs)

# --------------------- Argparse CLI ---------------------
def _parse_kwargs(s: str):
    if not s: return {}
    try:
        return ast.literal_eval(s)
    except Exception as e:
        raise SystemExit(f"Could not parse --kwargs. Use a Python dict literal, e.g. \"{{'m': 1.0, 'Nk': 81}}\". Error: {e}")

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Compact path when no --scene/--list provided
    if argv and not any(a.startswith("--scene") or a=="--list" for a in argv):
        parsed = parse_compact_cli(argv)
        if parsed.get("mode")=="list" or parsed.get("mode")=="help":
            print("Available scenes:"); [print(" -", k) for k in list_scenes()]; return 0
        out = parsed["out"] or (None if parsed["show"] else f"fig_{parsed['scene']}.png")
        path = render(parsed["scene"], out=out, dpi=parsed["dpi"], show=parsed["show"], **parsed["kwargs"])
        print("Rendered (interactive):" if parsed["show"] else "Saved:", path or "(interactive)")
        return 0

    # Argparse path
    ap = argparse.ArgumentParser(description="Orthographic 3D Topological Visualizer")
    ap.add_argument("--scene", type=str, required=False, help="Scene to render. Use --list to see options.")
    ap.add_argument("--kwargs", type=str, default="", help="Python dict of kwargs for the scene, e.g. \"{'m':1.0,'Nk':81}\"")
    ap.add_argument("--out", type=str, default="", help="Output image path (PNG). Defaults to ./fig_<scene>.png unless --show")
    ap.add_argument("--dpi", type=int, default=180, help="Output DPI (default 180)")
    ap.add_argument("--show", action="store_true", help="Also show the figure in a window")
    ap.add_argument("--list", action="store_true", help="List available scenes and exit")
    ap.add_argument("--portfolio-pdf", type=str, default="", help="Build a portfolio PDF from existing PNGs matching fig_*.png")
    ap.add_argument("--portfolio-zip", type=str, default="", help="Zip existing PNGs matching fig_*.png")
    args = ap.parse_args(argv)

    if args.list or not args.scene:
        print("Available scenes:")
        for k in list_scenes(): print(" -", k)
        if not args.scene: return 0

    kw = _parse_kwargs(args.kwargs)
    out = args.out or (None if args.show else os.path.join(os.getcwd(), f"fig_{args.scene}.png"))
    path = render(args.scene, out=out, dpi=args.dpi, show=args.show, **kw)
    print("Rendered (interactive):" if args.show else "Saved:", path or "(interactive)")

    if args.portfolio_pdf:
        pdf_path = export_portfolio_pdf(png_glob="fig_*.png", pdf_path=args.portfolio_pdf)
        print("Portfolio PDF:", pdf_path)
    if args.portfolio_zip:
        zip_path = export_portfolio_zip(png_glob="fig_*.png", zip_path=args.portfolio_zip)
        print("Portfolio ZIP:", zip_path)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
