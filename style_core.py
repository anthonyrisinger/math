# --- style_core.py ---
from __future__ import annotations
import os, math, time, dataclasses
from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt

# ---------- 0) Canonical camera (numeric, not mnemonic) ----------
# φ = (1+√5)/2; φ−1 ≈ 0.61803 rad = 35.437° (locked for reproducibility)
CAMERA_ELEV_DEG = 35.437
CAMERA_AZIM_DEG = -45.0
CAMERA_CANON: Tuple[float, float] = (CAMERA_ELEV_DEG, CAMERA_AZIM_DEG)

def set_canonical_ortho(ax, elev: float = CAMERA_ELEV_DEG, azim: float = CAMERA_AZIM_DEG):
    """Apply orthographic projection, box aspect (1,1,1), canonical view."""
    # 3D axes only; caller is responsible for projection="3d" on construction.
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    # Quiet, readable panes
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo.get("tick", {}).update(inward_factor=0.0, outward_factor=0.0)
    ax.grid(False)
    return ax

# ---------- 1) Invariant badge & residual policy (ASCII-safe) ----------
@dataclass(frozen=True)
class ResidualPolicy:
    # Snap threshold for declaring integer lock
    snap_epsilon: float = 5e-3
    # Buckets for coloring/health (you choose colors in draw layer)
    ok: float = 1e-3
    warn: float = 5e-3

def nearest_int_and_residual(x: float) -> Tuple[int, float]:
    n = int(round(x))
    return n, (x - n)

def format_badge(name: str, value: float, policy: ResidualPolicy = ResidualPolicy()) -> Tuple[str, str]:
    """
    Returns (text, level) where level is one of {'ok','warn','bad'} by |res|.
    Format is ASCII-stable and monospaced-friendly.
    """
    n, res = nearest_int_and_residual(value)
    ares = abs(res)
    if ares <= policy.ok:
        lvl = "ok"
    elif ares <= policy.warn:
        lvl = "warn"
    else:
        lvl = "bad"
    text = f"{name}: {value: .6f}  ->  {n: d}    |res|={ares:.2e}"
    return text, lvl

# ---------- 2) Health strip (backend, grid, gauge, fps) ----------
def backend_name() -> str:
    # 'module://matplotlib_inline.backend_inline' → 'inline'
    b = matplotlib.get_backend()
    return b.split('.')[-1].lower()

@dataclass
class HealthState:
    backend: str
    Nk: Optional[int] = None
    Nk_level: Optional[str] = None  # 'ok'|'warn'|'bad'
    gauge_ok: Optional[bool] = None
    fps: Optional[float] = None
    hints: Optional[str] = None     # free-form, e.g., 'QT_QPA_PLATFORM=xcb on Wayland'

def nk_level(Nk: Optional[int]) -> Optional[str]:
    if Nk is None:
        return None
    if Nk >= 121:
        return "ok"
    if Nk >= 61:
        return "warn"
    return "bad"

def format_health_strip(h: HealthState) -> str:
    parts = [f"backend={h.backend}"]
    if h.Nk is not None:
        lvl = h.Nk_level or nk_level(h.Nk)
        parts.append(f"Nk={h.Nk}({lvl})")
    if h.gauge_ok is not None:
        parts.append(f"gauge={'pass' if h.gauge_ok else 'fail'}")
    if h.fps is not None:
        parts.append(f"FPS≈{h.fps:.0f}")
    parts.append("save:s reset:r spin:p help:h")
    if h.hints:
        parts.append(h.hints)
    return " | ".join(parts)

# ---------- 3) Safe saving (never fail) ----------
def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

# ---------- 4) Simple FPS meter (optional) ----------
class FPSMeter:
    def __init__(self, avg_over: int = 15):
        self.avg_over = avg_over
        self.samples = []

    def tick(self) -> Optional[float]:
        now = time.perf_counter()
        self.samples.append(now)
        if len(self.samples) > self.avg_over:
            self.samples.pop(0)
        if len(self.samples) >= 2:
            dt = (self.samples[-1] - self.samples[0]) / (len(self.samples) - 1)
            return 1.0 / dt if dt > 0 else None
        return None

# ---------- 5) Scene style contract (programmatic) ----------
@dataclass(frozen=True)
class StyleContract:
    camera: Tuple[float, float] = CAMERA_CANON
    badge_required: bool = True
    convergence_key: Optional[str] = "Nk"   # e.g., 'Nk' or 'Nt' or None
    gauge_check: bool = True
    save_key: str = "s"
    # Optional label normalization for UI/CLI consistency
    symbols_ascii: Tuple[str, ...] = ("Nk", "Nt", "alpha", "SL(2,Z)")

def enforce_axes_style(ax) -> None:
    """Minimal axis cosmetics to keep visuals clean and consistent."""
    # Remove axis lines & panes; leave ticks to scene preference
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0, 0, 0, 0))
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

# Handy one-liners to drop in scenes:
def render_badge(ax, name: str, value: float, policy: ResidualPolicy = ResidualPolicy(), loc: Tuple[float,float]=(0.02,0.98)):
    txt, lvl = format_badge(name, value, policy)
    color = {"ok":"#0b8043", "warn":"#e37400", "bad":"#c11"}[lvl]
    ax.text(loc[0], loc[1], txt, transform=ax.figure.transFigure, ha="left", va="top",
            color=color, fontsize=10, family="monospace", bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))
    return txt, lvl

# Draw health strip overlay
def draw_health_strip(fig: plt.Figure, text: str, fontsize: float = 9):
    # Bottom-left overlay that never clips; ASCII-safe; high-contrast
    fig.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=fontsize, color="#111111", family="monospace")

# Safe saving
def savefig_safe(fig: plt.Figure, path: str, dpi: int = 200, transparent: bool = False, metadata: Optional[dict] = None):
    ensure_parent_dir(path)
    meta = dict(Title=os.path.basename(path), Software="topo_viz", Artist="topo_viz")
    if metadata:
        meta.update(metadata)
    fig.savefig(path, dpi=dpi, transparent=transparent, bbox_inches="tight", pad_inches=0.0, metadata=meta)
