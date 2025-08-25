#!/usr/bin/env python3
"""
morphic_3d_atlas.py

Four orthographic CGA views of the “morphic” family with intrinsic anchors.
- Equation mode: "shifted" (τ^3 − (2−k)τ − 1) or "simple" (τ^3 − kτ − 1)
- Surface: param (θ × k) → (X,Y,Z=k), colored by peak curvature per k-slice
- Overlays: special loops at k={0,1,2,3,4}, labels, centroid spine
- Anchors: discriminant plane at k*, perfect-circle plane at k_perfect
"""

import argparse, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

from morphic_core import (
    real_roots, discriminant,
    k_perfect_circle, k_discriminant_zero,
    sample_loop_xyz, curvature_peak
)

def parse_args():
    p = argparse.ArgumentParser(description="Morphic 3D Atlas (CGA)")
    p.add_argument("--mode", choices=["shifted","simple"], default="shifted",
                   help="equation family: 'shifted' = τ^3−(2−k)τ−1, 'simple' = τ^3−kτ−1")
    p.add_argument("--kmin", type=float, default=0.0, help="min k (default: 0.0)")
    p.add_argument("--kmax", type=float, default=4.0, help="max k (default: 4.0)")
    p.add_argument("--nk",   type=int,   default=81,   help="number of k slices (default: 81)")
    p.add_argument("--nt",   type=int,   default=480,  help="θ resolution (default: 480)")
    p.add_argument("--lim",  type=float, default=1.8,  help="XY half-extent (default: 1.8)")
    p.add_argument("--output","-o", default="morphic_views.png", help="output PNG file")
    p.add_argument("--transparent", action="store_true", help="transparent figure background")
    return p.parse_args()

def main():
    args = parse_args()
    mode = args.mode
    k_min, k_max = args.kmin, args.kmax
    N_k, N_t = args.nk, args.nt
    L = float(args.lim)

    # intrinsic anchors
    k_star    = k_discriminant_zero(mode)
    k_circle  = k_perfect_circle(mode)

    # grids
    k_cont = np.linspace(k_min, k_max, N_k)
    theta  = np.linspace(0, 2*np.pi, N_t, endpoint=False)
    dtheta = 2*np.pi / N_t

    # compute slices (surface rows) and peak curvature per row
    slices = []
    peaks  = []
    for k in k_cont:
        tau0 = real_roots(k, mode=mode)[0]
        xyz  = sample_loop_xyz(tau0, theta)
        slices.append((xyz, k))
        peaks.append(curvature_peak(xyz[:,:2], dtheta))
    peaks = np.array(peaks)
    max_peak = float(np.nanmax(peaks)) if np.isfinite(peaks).any() else 1.0

    # discrete overlays at k = {0,1,2,3,4} (regardless of mode)
    k_special = np.array([0,1,2,3,4], dtype=float)
    special_names = ["ϕ-Golden","ρ-Plastic","1-Unity","ψ⁻-Inv-Super-Golden","σ⁻-Inv-Super-Silver"]
    discrete = []
    for k0 in k_special:
        tau0 = real_roots(k0, mode=mode)[0]
        xyz  = sample_loop_xyz(tau0, theta)
        discrete.append((xyz, k0))

    # centroid spine
    centroids = [(xyz[:,0].mean(), xyz[:,1].mean(), k) for xyz,k in slices]

    # build surface arrays
    X = np.stack([xyz_k[:,0] for (xyz_k,_) in slices])
    Y = np.stack([xyz_k[:,1] for (xyz_k,_) in slices])
    Z = np.repeat(k_cont[:,None], N_t, axis=1)
    CV = np.repeat(peaks[:,None], N_t, axis=1)

    # colors from curvature
    cmap = plt.cm.plasma
    norm = matplotlib.colors.Normalize(0, max_peak)
    FC = cmap(norm(np.nan_to_num(CV)))

    # fig
    fig = plt.figure(figsize=(12,12), dpi=220)
    if args.transparent:
        fig.patch.set_alpha(0.0)
    gs = fig.add_gridspec(2,2, wspace=0.2, hspace=0.2)

    views = [
        (90, 0,    "Top-down",      gs[0,0]),
        (0,  0,    "Side",          gs[0,1]),
        (30, 180,  "45° Isometric", gs[1,0]),
        (np.degrees((1+5**0.5)/2 -1), 225, "Golden-angle", gs[1,1]),
    ]

    # suppress harmless warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    for elev, azim, title, cell in views:
        ax = fig.add_subplot(cell, projection="3d")
        if args.transparent:
            ax.set_facecolor("none"); ax.patch.set_alpha(0)
        ax.set_proj_type("ortho")
        ax.view_init(elev, azim)
        ax.set_box_aspect((1,1,1))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_title(f"{title}  [{mode}]", fontsize=11, pad=10)

        # bounds
        ax.set_xlim(-L, L); ax.set_ylim(-L, L)
        ax.set_zlim(k_min, k_max)

        # discriminant plane at k_star
        Xp, Yp = np.meshgrid([-L, L], [-L, L])
        Zp = np.full_like(Xp, k_star)
        ax.plot_surface(Xp, Yp, Zp, color="gold", alpha=0.10, edgecolor="none")

        # perfect-circle slice at k_circle
        # (recompute via CGA so it’s consistent with the mapping)
        tau_c = 1.0
        circ  = sample_loop_xyz(tau_c, theta)
        ax.plot(circ[:,0], circ[:,1], zs=k_circle, zdir="z",
                ls="--", color="gray", lw=1.2)

        # surface (θ×k)
        ax.plot_surface(X, Y, Z, facecolors=FC,
                        rstride=1, cstride=1, antialiased=True, shade=False)

        # overlays
        for (xyz0, k0), name in zip(discrete, special_names):
            m = (np.abs(xyz0[:,0]) <= L+0.2) & (np.abs(xyz0[:,1]) <= L+0.2)
            idx = np.where(m)[0]
            if len(idx)>1:
                ax.plot(xyz0[idx,0], xyz0[idx,1], zs=k0, zdir="z",
                        color="red", lw=2.3, label=f"{name} (k={int(k0)})")
        if title == "Top-down":
            ax.legend(loc="upper left", fontsize=8, framealpha=0.75)

        # centroid spine
        Cx, Cy, Cz = zip(*centroids)
        ax.plot(Cx, Cy, Cz, ls=":", color="black", lw=1.0)

        # critical star (max-x point on the k≈k* slice)
        i_star = int(np.argmin(np.abs(k_cont - k_star)))
        xyz_s, _ = slices[i_star]
        imax = int(xyz_s[:,0].argmax())
        sx, sy = xyz_s[imax,0], xyz_s[imax,1]
        ax.scatter([sx],[sy],[k_cont[i_star]], marker="*", s=100, color="black")

    # shared colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cax = fig.add_axes([0.92, 0.55, 0.015, 0.30])
    fig.colorbar(sm, cax=cax, label="Peak curvature |κ|")

    plt.tight_layout(rect=[0,0,0.9,1])
    fig.savefig(args.output, dpi=220, transparent=args.transparent)
    print(f"[✓] {args.output} written — mode={mode}, k∈[{k_min},{k_max}], θ={N_t}, k-slices={N_k}")

if __name__ == "__main__":
    main()
