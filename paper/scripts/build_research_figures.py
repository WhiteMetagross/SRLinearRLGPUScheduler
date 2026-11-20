"""Build publication-quality figures for the paper:
Forecast Guided Reinforcement Learning for Reliable and Risk Aware Quota Control in Preemptible GPU Scheduling.

Outputs PDF, SVG, and 300 DPI PNG for all figures into:
  paper/figures/pdf/
  paper/figures/svg/
  paper/figures/png/
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ============================================================================
# DIRECTORIES AND PATHS
# ============================================================================

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper" / "figures"
PDF_DIR = OUT_DIR / "pdf"
SVG_DIR = OUT_DIR / "svg"
PNG_DIR = OUT_DIR / "png"

for d in (OUT_DIR, PDF_DIR, SVG_DIR, PNG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ============================================================================
# STYLE CONFIGURATION (Matching newIMLProject and ASPLOS/ATC references)
# ============================================================================

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Liberation Serif'],
    'mathtext.fontset': 'dejavuserif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9.5,
    'ytick.labelsize': 9.5,
    'legend.fontsize': 9.5,
    'figure.titlesize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.9,
    'axes.grid': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

# Vibrant color palette (Matching newIMLProject / IMLReport style)
COLORS = {
    'primary': '#FF2D55',       # Vibrant crimson (SRLinear / SRAC)
    'vermillion': '#D55E00',    # Deep vermillion
    'purple': '#5856D6',        # Royal violet / purple
    'blue': '#0072B2',          # Deep cobalt blue
    'sky': '#56B4E9',           # Sky cyan
    'green': '#009E73',         # Emerald green
    'light_green': '#34C759',   # Vibrant lime green
    'orange': '#FF9500',        # Vibrant orange
    'amber': '#E69F00',         # Amber
    'gray': '#7F7F7F',          # Neutral gray
    'light_gray': '#D3D3D3',    # Pale gray
    'dark_gray': '#222222',     # Charcoal
    'infeasible': '#D32F2F',    # Red border for infeasibility
}

def save_all_formats(fig: plt.Figure, stem: str) -> None:
    """Save figure into dedicated pdf, svg, and png subfolders with 300 DPI."""
    fig.savefig(PDF_DIR / f"{stem}.pdf", bbox_inches='tight')
    fig.savefig(SVG_DIR / f"{stem}.svg", bbox_inches='tight')
    fig.savefig(PNG_DIR / f"{stem}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Generated {stem} in pdf/, svg/, and png/ folders")


# ============================================================================
# ARCHITECTURE DIAGRAM HELPERS AND REDESIGNED FIGURES 1-3
# ============================================================================

def draw_card(ax, x, y, w, h, title, subtitle="", body_lines=None, border_color="#334155", bg_color="#ffffff", lw=0.9, title_color="#0f172a", title_size=7.2):
    """Draw a clean publication-grade card with perfectly centered text lines."""
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.035",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=lw,
        zorder=2
    )
    ax.add_patch(box)
    
    # Collect non-empty text lines
    lines = []
    if title:
        lines.append(("title", title))
    if subtitle:
        lines.append(("subtitle", subtitle))
    if body_lines:
        for b in body_lines:
            if b:
                lines.append(("body", b))
                
    n = len(lines)
    if n == 0:
        return
        
    y_step = h / (n + 1)
    for idx, (kind, text) in enumerate(lines):
        y_pos = y + h - (idx + 1) * y_step
        if kind == "title":
            ax.text(x + w / 2.0, y_pos, text,
                    fontsize=title_size, fontweight="bold", color=title_color,
                    ha="center", va="center", zorder=3)
        elif kind == "subtitle":
            ax.text(x + w / 2.0, y_pos, text,
                    fontsize=6.5, color="#1e293b",
                    ha="center", va="center", zorder=3)
        else:
            ax.text(x + w / 2.0, y_pos, text,
                    fontsize=6.0, color="#64748b",
                    ha="center", va="center", zorder=3)

def draw_orthogonal(ax, points, color="#334155", lw=0.9, arrow=True, dashed=False, arrow_size=7):
    """Draw orthogonal lines through points, with optional arrowhead at the final point."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ls = "--" if dashed else "-"
    ax.plot(xs, ys, color=color, lw=lw, linestyle=ls, zorder=1)
    if arrow and len(points) >= 2:
        p_prev = points[-2]
        p_last = points[-1]
        ax.annotate("", xy=p_last, xytext=p_prev,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                    mutation_scale=arrow_size),
                    zorder=1)

def save_all_formats(fig, name):
    for ext, d in [("pdf", PDF_DIR), ("png", PNG_DIR), ("svg", SVG_DIR)]:
        path = d / f"{name}.{ext}"
        if ext == "png":
            fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03)
        else:
            fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"[OK] Saved {name}")

# ==============================================================================
# FIGURE 1: SRLinear Mathematical Dataflow
# ==============================================================================
def generate_fig01_srlinear_architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 2.75))
    ax.set_xlim(0, 7.1)
    ax.set_ylim(0, 2.75)
    ax.axis("off")
    
    # Column 1: Inputs (x = 0.15, w = 1.35)
    draw_card(ax, 0.15, 1.85, 1.35, 0.72, "Seasonal Anchors",
              subtitle=r"$A_k \in \mathbb{R}^{K \times C}$",
              body_lines=["24h, 168h Causal Lags"],
              border_color="#94a3b8", bg_color="#f8fafc", title_size=7.2)
              
    draw_card(ax, 0.15, 1.00, 1.35, 0.72, "Context Features",
              subtitle=r"$F_t \in \mathbb{R}^d$",
              body_lines=["Hour, Day, Trend"],
              border_color="#94a3b8", bg_color="#f8fafc", title_size=7.2)
              
    draw_card(ax, 0.15, 0.15, 1.35, 0.72, "Usage History",
              subtitle=r"$X_{1:L} \in \mathbb{R}^{L \times C}$",
              body_lines=["L=672h, 28 Orgs"],
              border_color="#94a3b8", bg_color="#f8fafc", title_size=7.2)

    # Column 2: Intermediate Estimators (x = 1.85, w = 1.55)
    draw_card(ax, 1.85, 1.85, 1.55, 0.72, "Convex Gating",
              subtitle=r"$g = \mathrm{Softmax}(W_g F_t + b_g)$",
              body_lines=["Adaptive Anchor Mix"],
              border_color="#d73027", bg_color="#fff8f8", lw=1.1, title_color="#991b1b", title_size=7.2)
              
    draw_card(ax, 1.85, 1.00, 1.55, 0.72, "Linear Residuals",
              subtitle=r"$R_{\mathrm{org}} = W_o F_t + b_o$",
              body_lines=[r"$R_{\mathrm{pool}} = W_p F_t + b_p$"],
              border_color="#d73027", bg_color="#fff8f8", lw=1.1, title_color="#991b1b", title_size=7.2)
              
    draw_card(ax, 1.85, 0.15, 1.55, 0.72, "Variance Head",
              subtitle=r"$h_o = W_v F_t + b_v$",
              body_lines=["Heteroscedastic Scale"],
              border_color="#94a3b8", bg_color="#f8fafc", title_size=7.2)

    # Column 3: Aggregation & Volatility (x = 3.80, w = 1.55)
    draw_card(ax, 3.80, 1.85, 1.55, 0.72, "Seasonal Base",
              subtitle=r"$X_{\mathrm{base}} = \sum_k g_k A_k$",
              body_lines=["Causal Anchor Mix"],
              border_color="#94a3b8", bg_color="#f8fafc", title_size=7.2)
              
    draw_card(ax, 3.80, 1.00, 1.55, 0.72, "Coherent Reconciler",
              subtitle=r"$\hat{\mu}_o = \mathrm{Reconcile}(\cdot)$",
              body_lines=["Exact Pool Hierarchy"],
              border_color="#d73027", bg_color="#fff8f8", lw=1.1, title_color="#991b1b", title_size=7.2)
              
    draw_card(ax, 3.80, 0.15, 1.55, 0.72, "Uncertainty Scale",
              subtitle=r"$\hat{\sigma}_o = \mathrm{Softplus}(h_o)$",
              body_lines=["Residual Volatility"],
              border_color="#94a3b8", bg_color="#f8fafc", title_size=7.2)

    # Column 4: Final Demand Distribution (x = 5.65, w = 1.30)
    draw_card(ax, 5.65, 0.35, 1.30, 2.05, r"Forecast $\hat{y}_o$",
              subtitle=r"$\hat{y}_o \sim \mathcal{N}(\hat{\mu}_o, \hat{\sigma}^2_o)$",
              body_lines=[r"$\hat{y}_{\mathrm{pool}} = \sum_o \hat{y}_o$",
                          "Cluster Coherent", "Gaussian Demand"],
              border_color="#334155", bg_color="#f8fafc", lw=1.1, title_size=7.2)

    # Connections
    # Anchors -> Seasonal Base (over top margin)
    draw_orthogonal(ax, [(0.825, 2.57), (0.825, 2.68), (4.575, 2.68), (4.575, 2.57)], color="#334155", lw=0.9)
    
    # Context Features -> Bus -> Gating, Residuals, Variance Head
    draw_orthogonal(ax, [(1.50, 1.36), (1.68, 1.36)], color="#334155", lw=0.9, arrow=False)
    draw_orthogonal(ax, [(1.68, 2.21), (1.68, 0.51)], color="#334155", lw=0.9, arrow=False)
    draw_orthogonal(ax, [(1.68, 2.21), (1.85, 2.21)], color="#334155", lw=0.9)
    draw_orthogonal(ax, [(1.68, 1.36), (1.85, 1.36)], color="#334155", lw=0.9)
    draw_orthogonal(ax, [(1.68, 0.51), (1.85, 0.51)], color="#334155", lw=0.9)
    
    # Column 2 -> Column 3
    draw_orthogonal(ax, [(3.40, 2.21), (3.80, 2.21)], color="#334155", lw=0.9)
    draw_orthogonal(ax, [(3.40, 1.36), (3.80, 1.36)], color="#334155", lw=0.9)
    draw_orthogonal(ax, [(3.40, 0.51), (3.80, 0.51)], color="#334155", lw=0.9)
    
    # Seasonal Base -> Coherent Reconciler
    draw_orthogonal(ax, [(4.575, 1.85), (4.575, 1.72)], color="#334155", lw=0.9)
    
    # Column 3 -> Column 4 (Pure horizontal & orthogonal)
    draw_orthogonal(ax, [(5.35, 1.36), (5.65, 1.36)], color="#334155", lw=0.9)
    draw_orthogonal(ax, [(5.35, 0.51), (5.50, 0.51), (5.50, 0.85), (5.65, 0.85)], color="#334155", lw=0.9)

    save_all_formats(fig, "fig01_srlinear_architecture")

# ==============================================================================
# FIGURE 2: SRAC Closed-Loop System Architecture
# ==============================================================================
def generate_fig02_srac_system_architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 2.95))
    ax.set_xlim(0, 7.1)
    ax.set_ylim(0, 2.95)
    ax.axis("off")

    # Module 1: Causal Forecaster (x = 0.15, w = 1.50)
    draw_card(ax, 0.15, 1.15, 1.50, 1.35, "1. Forecaster",
              subtitle="SRLinear (L=672h)",
              body_lines=[r"$\hat{\mu}_t, \hat{\sigma}_t$ Demand Forecast",
                          "Cluster Coherence"],
              border_color="#d73027", bg_color="#fff8f8", lw=1.1, title_color="#991b1b", title_size=7.4)

    # Module 2: Quota Controller (x = 1.95, w = 1.55)
    draw_card(ax, 1.95, 1.15, 1.55, 1.35, "2. Quota Policy",
              subtitle="SRAC Constrained RL",
              body_lines=[r"Adaptive: $\alpha_t, \beta_t$",
                          "Safety Shield (SLO ≤ 2%)"],
              border_color="#2563eb", bg_color="#eff6ff", lw=1.1, title_color="#1d4ed8", title_size=7.4)

    # Module 3: Scheduling Core (split into 3a and 3b)
    # 3a: HP Gang Packing (x = 4.15, w = 1.65)
    draw_card(ax, 4.15, 1.88, 1.65, 0.62, "3a. HP Gang Packing",
              subtitle="Strict Gang Allocation",
              body_lines=["0% Preemption Rate"],
              border_color="#475569", bg_color="#f8fafc", lw=0.9, title_size=7.0)

    # 3b: Spot Eviction Selector (x = 4.15, w = 1.65)
    draw_card(ax, 4.15, 1.15, 1.65, 0.62, "3b. Spot Preemption",
              subtitle=r"Min-Rollback: $\arg\min \Phi$",
              body_lines=[r"Bounded by Quota $Q_t$"],
              border_color="#475569", bg_color="#f8fafc", lw=0.9, title_size=7.0)

    # Module 4: Cluster Fabric (x = 6.20, w = 0.75)
    draw_card(ax, 6.20, 1.15, 0.75, 1.35, "Cluster",
              subtitle="Fabric",
              body_lines=["3,456 A100s", "Static Nodes"],
              border_color="#334155", bg_color="#f8fafc", lw=1.0, title_size=7.2)

    # HP Arrivals
    ax.text(4.975, 2.76, "HP Job Queue", fontsize=7.2, fontweight="bold", ha="center", va="center", color="#0f172a")
    draw_orthogonal(ax, [(4.975, 2.65), (4.975, 2.50)], color="#0f172a", lw=1.0)

    # Spot Arrivals
    ax.text(3.70, 0.88, "Spot Jobs", fontsize=6.5, ha="center", va="center", color="#475569")
    draw_orthogonal(ax, [(3.70, 0.98), (3.92, 0.98), (3.92, 1.28), (4.15, 1.28)], color="#475569", lw=0.9)

    # Module 1 -> Module 2 (Forecast: mu, sigma)
    draw_orthogonal(ax, [(1.65, 1.82), (1.95, 1.82)], color="#334155", lw=1.0)
    ax.text(1.80, 1.96, r"$\hat{\mu}, \hat{\sigma}$", fontsize=7.0, ha="center", va="center", color="#0f172a")

    # Module 2 -> Module 3b (Quota Q_t)
    draw_orthogonal(ax, [(3.50, 1.46), (4.15, 1.46)], color="#2563eb", lw=1.1)
    ax.text(3.82, 1.58, r"$Q_t$", fontsize=7.2, fontweight="bold", ha="center", va="bottom", color="#2563eb")

    # Module 3 -> Module 4 (Placements & Evictions)
    draw_orthogonal(ax, [(5.80, 2.19), (6.20, 2.19)], color="#334155", lw=0.9)
    draw_orthogonal(ax, [(5.80, 1.46), (6.20, 1.46)], color="#334155", lw=0.9)

    # Closed-Loop Telemetry & State Feedback (Bottom container)
    telemetry_box = patches.FancyBboxPatch(
        (0.25, 0.12), 6.60, 0.58,
        boxstyle="round,pad=0.015,rounding_size=0.035",
        facecolor="#f8fafc",
        edgecolor="#64748b",
        linewidth=0.9,
        linestyle="--",
        zorder=1
    )
    ax.add_patch(telemetry_box)
    ax.text(3.55, 0.48, "Closed-Loop Telemetry & State Feedback",
            fontsize=7.4, fontweight="bold", ha="center", va="center", color="#0f172a")
    ax.text(3.55, 0.28,
            r"Queue Delays $(D_{\mathrm{hp}}, D_{\mathrm{spot}})$, Evictions $(E_t)$, Rollback Losses $(\Phi_t)$, Allocation",
            fontsize=6.5, color="#475569", ha="center", va="center")

    # Feedback wiring: Fabric -> Telemetry
    draw_orthogonal(ax, [(6.575, 1.15), (6.575, 0.70)], color="#64748b", lw=0.9, dashed=True)
    
    # Telemetry -> Forecaster & Quota Controller
    draw_orthogonal(ax, [(0.90, 0.70), (0.90, 1.15)], color="#64748b", lw=0.9, dashed=True)
    draw_orthogonal(ax, [(2.725, 0.70), (2.725, 1.15)], color="#64748b", lw=0.9, dashed=True)

    save_all_formats(fig, "fig02_srac_system_architecture")

# ==============================================================================
# FIGURE 3: SRAC Constrained Recurrent RL Agent Architecture
# ==============================================================================
def generate_fig03_srac_rl_agent_architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 3.40))
    ax.set_xlim(0, 7.1)
    ax.set_ylim(0, 3.40)
    ax.axis("off")

    # State Vector s_t (x = 0.15, w = 1.15)
    draw_card(ax, 0.15, 1.40, 1.15, 1.65, r"State $s_t$",
              subtitle=r"$\mathbb{R}^{39}$ Input Space",
              body_lines=["Alloc & Capacity",
                          r"Queues $D_{\mathrm{hp}}, D_{\mathrm{sp}}$",
                          r"Forecast $\hat{\mu}, \hat{\sigma}$",
                          "Risk Budgets"],
              border_color="#94a3b8", bg_color="#f8fafc", title_size=7.4)

    # Encoder (x = 1.55, w = 0.95)
    draw_card(ax, 1.55, 1.80, 0.95, 0.85, "Encoder",
              subtitle=r"$e_t = \tanh(W_e s_t)$",
              body_lines=["Linear (Dim=64)"],
              border_color="#475569", bg_color="#ffffff", title_size=7.4)

    # GRU Core (x = 2.75, w = 1.05)
    draw_card(ax, 2.75, 1.70, 1.05, 1.05, "GRU Core",
              subtitle=r"$h_t = \mathrm{GRU}(e_t, h_{t-1})$",
              body_lines=[r"Latent $h_t \in \mathbb{R}^{64}$"],
              border_color="#5856d6", bg_color="#faf5ff", lw=1.1, title_color="#4338ca", title_size=7.4)

    # Recurrent loop arrow above GRU
    draw_orthogonal(ax, [(3.45, 2.75), (3.45, 2.95), (3.05, 2.95), (3.05, 2.75)],
                    color="#5856d6", lw=0.9)
    ax.text(3.25, 3.06, r"$h_{t-1}$", fontsize=6.8, color="#5856d6", ha="center", va="center")

    # Three Heads (x = 4.05, w = 1.20)
    # Actor Head (blue)
    draw_card(ax, 4.05, 2.45, 1.20, 0.62, r"Actor Head $\pi_\theta$",
              subtitle=r"$\alpha_t, \beta_t \in [0, 1]$",
              body_lines=["Mix & Uncertainty"],
              border_color="#2563eb", bg_color="#eff6ff", lw=1.1, title_color="#1d4ed8", title_size=7.0)

    # Reward Critic (green)
    draw_card(ax, 4.05, 1.70, 1.20, 0.62, r"Reward Critic $V_\phi$",
              subtitle=r"$V(s_t) \in \mathbb{R}$",
              body_lines=["Expected Spot Work"],
              border_color="#16a34a", bg_color="#f0fdf4", lw=1.1, title_color="#15803d", title_size=7.0)

    # Cost Critic (amber)
    draw_card(ax, 4.05, 0.95, 1.20, 0.62, r"Cost Critic $V^c_\psi$",
              subtitle=r"$V^c(s_t) \in \mathbb{R}^3$",
              body_lines=["HP, Rollback, Queue"],
              border_color="#d97706", bg_color="#fffbeb", lw=1.1, title_color="#b45309", title_size=7.0)

    # Safety Shield (x = 5.50, w = 1.35)
    draw_card(ax, 5.50, 1.50, 1.35, 1.55, "Safety Shield",
              subtitle=r"Ramp: $|\Delta \alpha|, |\Delta \beta| \leq 0.25$",
              body_lines=[r"Override: $D_{\mathrm{hp}} > 0$",
                          r"$\to \alpha=0, \beta=1$",
                          r"$Q_t = \mathrm{clip}(Q_{\mathrm{mix}} - \beta \hat{\sigma})$"],
              border_color="#d73027", bg_color="#fff8f8", lw=1.1, title_color="#991b1b", title_size=7.2)

    # Executed Quota output arrow
    draw_orthogonal(ax, [(6.85, 2.27), (7.05, 2.27)], color="#0f172a", lw=1.2)
    ax.text(6.95, 2.43, r"$Q_t$", fontsize=7.8, fontweight="bold", ha="center", va="center", color="#0f172a")

    # Inference Connections
    # State -> Encoder
    draw_orthogonal(ax, [(1.30, 2.22), (1.55, 2.22)], color="#334155", lw=0.9)
    
    # Encoder -> GRU
    draw_orthogonal(ax, [(2.50, 2.22), (2.75, 2.22)], color="#334155", lw=0.9)
    ax.text(2.625, 2.34, r"$e_t$", fontsize=6.8, ha="center", va="center", color="#334155")

    # GRU -> Latent Bus -> 3 Heads
    draw_orthogonal(ax, [(3.80, 2.22), (3.92, 2.22)], color="#5856d6", lw=0.9, arrow=False)
    draw_orthogonal(ax, [(3.92, 2.76), (3.92, 1.26)], color="#5856d6", lw=0.9, arrow=False)
    draw_orthogonal(ax, [(3.92, 2.76), (4.05, 2.76)], color="#5856d6", lw=0.9)
    draw_orthogonal(ax, [(3.92, 2.01), (4.05, 2.01)], color="#5856d6", lw=0.9)
    draw_orthogonal(ax, [(3.92, 1.26), (4.05, 1.26)], color="#5856d6", lw=0.9)

    # Actor Head -> Safety Shield
    draw_orthogonal(ax, [(5.25, 2.76), (5.38, 2.76), (5.38, 2.27), (5.50, 2.27)],
                    color="#2563eb", lw=1.0)
    ax.text(5.31, 2.89, r"$\alpha_t, \beta_t$", fontsize=6.8, color="#2563eb", ha="center", va="bottom")

    # Training Path (Offline PPO Container)
    train_box = patches.FancyBboxPatch(
        (0.15, 0.05), 6.80, 0.77,
        boxstyle="round,pad=0.015,rounding_size=0.035",
        facecolor="#fbfbfe",
        edgecolor="#7c3aed",
        linewidth=0.9,
        linestyle="--",
        zorder=1
    )
    ax.add_patch(train_box)
    ax.text(3.55, 0.71, "Constrained PPO & Lagrangian Dual Update (Offline Training Path)",
            fontsize=7.2, fontweight="bold", ha="center", va="center", color="#6d28d9")

    # Subcards inside Training Container
    # GAE Advantages
    draw_card(ax, 0.25, 0.10, 1.95, 0.48, "GAE Advantages",
              subtitle=r"$A^r_t = \mathrm{GAE}(r_t, V)$",
              body_lines=[r"$A^c_{t,k} = \mathrm{GAE}(c_{t,k}, V^c_k)$"],
              border_color="#7c3aed", bg_color="#ffffff", lw=0.7, title_size=6.8)

    # Lagrangian Multipliers
    draw_card(ax, 2.50, 0.10, 2.15, 0.48, "Lagrangian Multipliers",
              subtitle=r"$\lambda_k \leftarrow \max(0, \lambda_k + \eta(\bar{c}_k - 1))$",
              body_lines=[r"$A^{\mathrm{lag}}_t = A^r_t - \sum_k \lambda_k A^c_{t,k}$"],
              border_color="#7c3aed", bg_color="#ffffff", lw=0.7, title_size=6.8)

    # PPO Clipped Loss
    draw_card(ax, 4.95, 0.10, 1.90, 0.48, "PPO Clipped Loss",
              subtitle=r"$\mathcal{L}_{\mathrm{CLIP}}(\theta) \text{ on } A^{\mathrm{lag}}$",
              body_lines=[r"$+\; c_1 \mathcal{L}_V + c_2 \mathcal{L}_{V^c}$"],
              border_color="#7c3aed", bg_color="#ffffff", lw=0.7, title_size=6.8)

    # Intra-training arrows
    draw_orthogonal(ax, [(2.20, 0.34), (2.50, 0.34)], color="#7c3aed", lw=0.8)
    draw_orthogonal(ax, [(4.65, 0.34), (4.95, 0.34)], color="#7c3aed", lw=0.8)

    # Gradient update arrow from container top straight up to Encoder
    draw_orthogonal(ax, [(2.02, 0.82), (2.02, 1.80)],
                    color="#7c3aed", lw=0.9, dashed=True)
    ax.text(2.14, 1.25, r"$\nabla_\theta$", fontsize=7.2, color="#7c3aed", ha="left", va="center")

    # Transition feedback into PPO Loss
    draw_orthogonal(ax, [(6.95, 1.50), (6.95, 0.34), (6.85, 0.34)],
                    color="#7c3aed", lw=0.8, dashed=True)
    ax.text(6.85, 1.05, r"$(s_t, a_t, r_t, c_t)$", fontsize=6.2, color="#7c3aed", ha="right", va="center")

    save_all_formats(fig, "fig03_srac_rl_agent_architecture")


# Use the dedicated architecture renderer for Figures 1 to 3. The result
# figures below retain their independent statistical plotting code.
from research_architecture_figures import (
    generate_fig01_srlinear_architecture,
    generate_fig02_srac_system_architecture,
    generate_fig03_srac_rl_agent_architecture,
)


# ============================================================================
# FIGURE 4: FORECAST BENCHMARK AND COMPUTATIONAL EFFICIENCY
# ============================================================================

def generate_fig04_forecast_benchmark_efficiency() -> None:
    """Figure 4: Grouped bars for MAE/RMSE, Runtime vs MAE Pareto scatter, Efficiency bars."""
    fig = plt.figure(figsize=(14.2, 9.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.32, wspace=0.22)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # Panel A: Runtime vs MAE Scatter Plot (Log Runtime)
    models_scatter = [
        ("Seasonal Naive", 3.1, 265.109, 0, 0, COLORS['gray']),
        ("OrgLinear", 4300.2, 241.357, 10004, 274848, COLORS['purple']),
        ("NLinear", 93.6, 187.220, 2692, 75264, COLORS['sky']),
        ("DLinear", 153.0, 169.548, 5384, 150528, COLORS['blue']),
        ("Temporal Mamba", 1405.2, 167.161, 12900, 8555000, COLORS['orange']),
        ("PatchTST", 1179.0, 163.593, 94724, 204661000, COLORS['green']),
        ("SRLinear", 60.8, 154.964, 2052, 10372, COLORS['primary']),
    ]
    
    err_data = {
        "OrgLinear": (0, 20.229),
        "NLinear": (0, 2.509),
        "DLinear": (0, 3.974),
        "Temporal Mamba": (0, 1.710),
        "PatchTST": (0, 0.611),
        "SRLinear": (0, 0.130),
    }

    for name, rt, mae, params, macs, col in models_scatter:
        sz = 80 if params == 0 else 60 + 520 * (np.log10(params + 10) / np.log10(100000))
        yerr = err_data.get(name, (0, 0))[1]
        ax_a.errorbar(rt, mae, yerr=yerr, fmt='none', ecolor=col, elinewidth=1.2, capsize=3, alpha=0.7)
        ax_a.scatter(rt, mae, s=sz, color=col, edgecolor='black', linewidth=0.9, zorder=5, alpha=0.9)

    ax_a.set_xscale('log')
    ax_a.set_xlabel("Wall-Clock Benchmark Runtime (Seconds, Log Scale)")
    ax_a.set_ylabel("Mean Absolute Error (GPU Cards, MAE)")
    ax_a.set_title("(a) Runtime vs. Accuracy Tradeoff (Size $\\propto$ Parameters)")
    ax_a.grid(True, linestyle='--', alpha=0.3, which='both')
    ax_a.set_ylim(140, 285)
    ax_a.set_xlim(1.5, 12000)

    # Screen space offsets remain stable on the logarithmic runtime axis.
    offsets = {
        "Seasonal Naive": (10, 14),
        "OrgLinear": (-70, 24),
        "NLinear": (10, 14),
        "DLinear": (12, -24),
        "Temporal Mamba": (12, 22),
        "PatchTST": (32, -25),
        "SRLinear": (-10, 16),
    }
    for name, rt, mae, _, _, col in models_scatter:
        dx, dy = offsets[name]
        ax_a.annotate(name, xy=(rt, mae), xytext=(dx, dy),
                     textcoords='offset points', fontsize=8.5,
                     ha='left' if dx >= 0 else 'right',
                     fontweight='bold' if 'SRLinear' in name else 'normal',
                     color=COLORS['primary'] if 'SRLinear' in name else COLORS['dark_gray'],
                     arrowprops=dict(arrowstyle="-", color='#888888', lw=0.6))

    # Panel B: Grouped Bar Chart for MAE and RMSE
    models_bar = ["OrgLinear", "DLinear", "PatchTST", "SRLinear"]
    mae_vals = [241.357, 169.548, 163.593, 154.964]
    mae_stds = [20.229, 3.974, 0.611, 0.130]
    rmse_vals = [342.984, 203.589, 197.316, 198.602]
    rmse_stds = [36.747, 4.197, 0.457, 0.255]

    x = np.arange(len(models_bar))
    width = 0.35

    # Color shades per model: Dark shade for MAE, Light shade for RMSE (no diagonal hatching)
    mae_shades = ['#6a51a3', '#1f78b4', '#238b45', '#d73027']
    rmse_shades = ['#bcbddc', '#9ecae1', '#a1d99b', '#fc9272']

    ax_b.bar(x - width/2, mae_vals, width, yerr=mae_stds, capsize=3,
             label='MAE', color=mae_shades,
             edgecolor='black', linewidth=0.8)
    ax_b.bar(x + width/2, rmse_vals, width, yerr=rmse_stds, capsize=3,
             label='RMSE', color=rmse_shades,
             edgecolor='black', linewidth=0.8)

    ax_b.set_ylabel("Forecast Error (GPU Cards)")
    ax_b.set_title("(b) Point Forecast Error Comparison Across Architectures")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(models_bar, rotation=12, ha='right')
    ax_b.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax_b.set_ylim(0, 440)

    ax_b.annotate("5.27% lower MAE\nthan PatchTST",
                  xy=(3 - width/2, mae_vals[-1]), xytext=(2.25, 270),
                  ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                  color=COLORS['primary'],
                  arrowprops=dict(arrowstyle="->", color=COLORS['primary'], lw=0.9))

    p_dark = patches.Patch(facecolor='#555555', edgecolor='black', label='MAE (Darker Shade)')
    p_light = patches.Patch(facecolor='#cccccc', edgecolor='black', label='RMSE (Lighter Shade)')
    ax_b.legend(handles=[p_dark, p_light], loc='upper right', frameon=True, framealpha=0.92)

    # Panel C: Computational Complexity
    all_models = ["SRLinear", "NLinear", "DLinear", "OrgLinear", "Mamba", "PatchTST"]
    params_k = [2.052, 2.692, 5.384, 10.004, 12.900, 94.724]
    macs_k = [10.372, 75.264, 150.528, 274.848, 8555.0, 204661.0]

    x_c = np.arange(len(all_models))
    w_c = 0.36

    # Color shades for parameters (light shade) vs MACs (dark shade), no diagonal hatching
    c_params = '#6baed6'
    c_macs = '#08519c'

    ax_c.bar(x_c - w_c/2, params_k, w_c, label='Parameters (K, Light Shade)', color=c_params,
             edgecolor='black', linewidth=0.8)
    ax_c.bar(x_c + w_c/2, macs_k, w_c, label='Compute MACs (K, Dark Shade)', color=c_macs,
             edgecolor='black', linewidth=0.8)

    ax_c.set_yscale('log')
    ax_c.set_ylabel("Count (Thousands, Log Scale)")
    ax_c.set_title("(c) Parameter & Operation Count (Log Scale)")
    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(all_models)
    ax_c.grid(True, axis='y', linestyle='--', alpha=0.3, which='both')
    ax_c.legend(loc='upper left', frameon=True, framealpha=0.92)
    ax_c.set_ylim(1, 1000000)

    ax_c.annotate("19,732× fewer\nMACs than PatchTST", xy=(0 + w_c/2, 10.372), xytext=(0.5, 300),
                  fontsize=8, fontweight='bold', color=COLORS['primary'],
                  arrowprops=dict(arrowstyle="->", color=COLORS['primary'], lw=1.0))

    # Panel D: Horizon Breakdown
    horizons = ["1 Hour", "2 Hours", "3 Hours", "4 Hours"]
    h_mae = [163.377, 148.335, 156.686, 151.457]
    h_rmse = [204.582, 193.712, 202.240, 193.630]
    
    x_d = np.arange(len(horizons))
    ax_d.plot(x_d, h_mae, marker='o', markersize=6, color=COLORS['primary'], lw=2.0, label='SRLinear MAE')
    ax_d.plot(x_d, h_rmse, marker='s', markersize=6, color=COLORS['purple'], lw=2.0, linestyle='--', label='SRLinear RMSE')
    
    for i in range(len(horizons)):
        ax_d.text(x_d[i], h_mae[i] - 7, f"{h_mae[i]:.1f}", ha='center', va='top', fontsize=8.5, color=COLORS['primary'])
        ax_d.text(x_d[i], h_rmse[i] + 5, f"{h_rmse[i]:.1f}", ha='center', va='bottom', fontsize=8.5, color=COLORS['purple'])

    ax_d.set_xticks(x_d)
    ax_d.set_xticklabels(horizons)
    ax_d.set_ylabel("Error by Horizon (GPU Cards)")
    ax_d.set_title("(d) Error Across Multi-Step Horizons (1h to 4h)")
    ax_d.grid(True, linestyle='--', alpha=0.3)
    ax_d.set_ylim(130, 225)
    ax_d.legend(loc='upper right', frameon=True, framealpha=0.9)

    save_all_formats(fig, "fig04_forecast_benchmark_efficiency")


# ============================================================================
# FIGURE 5: NON-RL SCHEDULING BENCHMARK COMPARISON
# ============================================================================

def generate_fig05_non_rl_scheduling_comparison() -> None:
    """Figure 5: 14 non-RL scheduling methods evaluated under 4-thread validation trace."""
    fig = plt.figure(figsize=(14.6, 9.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.34, wspace=0.36)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # Table 5 Non-RL Methods Data
    methods = [
        "Native GFS", "FCFS", "Causal SEPT", "Fixed Residual", "Diurnal Quota",
        "Online Quantile", "Headroom Reserve", "CLQ", "YARN-CS",
        "Chronus", "Lyra", "FGD", "SRGFS", "SRMix"
    ]
    delta_work = [
        0.0000, 0.2550, 0.1064, 0.1002, 0.1916,
        -2.8172, 0.2529, 0.2726, 0.0165,
        0.0727, 0.2386, -0.0222, 0.2560, 0.2593
    ]
    lost_work = [
        329.007, 1292.174, 422.147, 319.434, 609.250,
        395.676, 604.595, 640.240, 33810.634,
        0.000, 946.913, 43354.449, 1195.832, 814.693
    ]
    spot_q = [
        193469.7, 143439.4, 174605.3, 250029.4, 84168.3,
        218313.2, 170447.1, 39883.4, 149163.2,
        96175.6, 106065.4, 180870.5, 27812.6, 64216.8
    ]
    hp_viol = [
        1.448, 1.340, 1.662, 1.555, 0.751,
        0.751, 1.609, 0.214, 0.214,
        0.322, 2.198, 0.214, 0.483, 0.858
    ]
    feasible = [True, True, True, True, True, True, True, True, True, True, False, True, True, True]

    bar_colors = []
    for m in methods:
        if m == "SRMix":
            bar_colors.append(COLORS['primary'])
        elif m == "CLQ":
            bar_colors.append(COLORS['blue'])
        elif m == "Native GFS":
            bar_colors.append(COLORS['green'])
        elif m == "SRGFS":
            bar_colors.append(COLORS['orange'])
        elif m in ["YARN-CS", "Chronus", "Lyra", "FGD"]:
            bar_colors.append(COLORS['gray'])
        else:
            bar_colors.append(COLORS['sky'])

    # Panel A: Useful Spot Work Change relative to Native GFS (%)
    y_pos = np.arange(len(methods))
    ax_a.barh(y_pos, delta_work, color=bar_colors, edgecolor='black', linewidth=0.7)
    ax_a.axvline(0, color='black', linewidth=0.9)
    ax_a.set_yticks(y_pos)
    ax_a.set_yticklabels(methods, fontsize=8.5)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Useful Spot Work $\\Delta$ vs. Native GFS (%)")
    ax_a.set_title("(a) Spot Work Gain (Proposed & Rulers in Color)")
    ax_a.grid(True, axis='x', linestyle='--', alpha=0.3)
    ax_a.set_xlim(-3.2, 0.55)

    ax_a.text(0.30, 7, "+0.27%", va='center', ha='left', fontsize=7.5, color=COLORS['blue'], fontweight='bold')
    ax_a.text(0.29, 13, "+0.26%", va='center', ha='left', fontsize=7.5, color=COLORS['primary'], fontweight='bold')

    # Panel B: Rollback Lost Work & Evictions (Log Scale)
    log_lost = [max(1.0, val) for val in lost_work]
    ax_b.barh(y_pos, log_lost, color=bar_colors, edgecolor='black', linewidth=0.7)
    ax_b.set_xscale('log')
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(methods, fontsize=8.5)
    ax_b.invert_yaxis()
    ax_b.set_xlabel("Rollback Lost GPU Hours (Log Scale)")
    ax_b.set_title("(b) Rollback Overhead (Lower is Better)")
    ax_b.grid(True, axis='x', linestyle='--', alpha=0.3, which='both')
    ax_b.set_xlim(1.0, 100000)

    # Panel C: Spot Queue Delay (GPU-Hours)
    ax_c.barh(y_pos, [q / 1000.0 for q in spot_q], color=bar_colors, edgecolor='black', linewidth=0.7)
    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels(methods, fontsize=8.5)
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Spot Queue Delay (Thousand GPU-Hours)")
    ax_c.set_title("(c) Spot Queue Reduction (Lower is Better)")
    ax_c.grid(True, axis='x', linestyle='--', alpha=0.3)

    # Panel D: HP Gang Violation Rate (%) with Feasibility Ceiling
    viol_colors = [COLORS['infeasible'] if not f else COLORS['green'] for f in feasible]
    ax_d.barh(y_pos, hp_viol, color=viol_colors, edgecolor='black', linewidth=0.7)
    ax_d.axvline(2.0, color=COLORS['infeasible'], linestyle='--', linewidth=1.5, label='2.0% Feasibility Ceiling')
    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels(methods, fontsize=8.5)
    ax_d.invert_yaxis()
    ax_d.set_xlabel("HP Gang Violation Rate (%)")
    ax_d.set_title("(d) HP SLO Reliability (Ceiling ≤ 2.0%)")
    ax_d.grid(True, axis='x', linestyle='--', alpha=0.3)
    ax_d.set_xlim(0, 2.8)
    ax_d.legend(loc='lower right', frameon=True, framealpha=0.9)

    ax_d.text(2.198 + 0.05, 10, "2.20% (Infeasible)", va='center', fontsize=8, color=COLORS['infeasible'], fontweight='bold')

    save_all_formats(fig, "fig05_non_rl_scheduling_comparison")


# ============================================================================
# FIGURE 6: REINFORCEMENT LEARNING COMPARISON & PARETO TRADEOFFS
# ============================================================================

def generate_fig06_rl_comparison_tradeoffs() -> None:
    """Figure 6: SRAC versus RL baselines and CLQ."""
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14.0, 5.4))
    fig.subplots_adjust(wspace=0.28, top=0.90, bottom=0.14, left=0.07, right=0.97)

    # Table 8 RL Data
    rl_models = [
        ("CLQ", 96408.612, 640.240, 39883.4, 3001.3, 0.2145, 245.0, 0.0, 0.0, COLORS['blue']),
        ("SRGFS", 96392.645, 1195.832, 27812.6, 3816.4, 0.4826, 314.0, 0.0, 49.7, COLORS['orange']),
        ("CLQ-PPO", 96384.645, 723.666, 41098.6, 10925.0, 1.8231, 254.0, 0.0, 0.0, COLORS['purple']),
        ("SRU-PPO", 96398.457, 1898.094, 58789.7, 5932.0, 1.0009, 395.3, 5.0, 638.5, COLORS['sky']),
        ("SRQ-PPO", 96261.064, 976.155, 143054.4, 8087.4, 1.3762, 262.7, 123.7, 378.2, COLORS['gray']),
        ("SRAC", 96361.058, 621.251, 44637.0, 8426.7, 1.5550, 233.3, 4.6, 80.4, COLORS['primary']),
    ]

    # Panel A: Useful Work vs Rollback Loss Pareto Tradeoff
    for name, work, lost, sq, hq, viol, ev, work_err, lost_err, col in rl_models:
        sz = 80 + 350 * (hq / 12000.0)
        ax_a.errorbar(work, lost, xerr=work_err, yerr=lost_err, fmt='none', ecolor=col, elinewidth=1.2, capsize=3, alpha=0.8)
        ax_a.scatter(work, lost, s=sz, color=col, edgecolor='black', linewidth=1.0, zorder=5, alpha=0.9)

    ax_a.plot([96361.058, 96408.612], [621.251, 640.240], color=COLORS['dark_gray'], linestyle='--', lw=1.5, zorder=4, label='Pareto Frontier')

    ax_a.set_xlabel("Useful Spot GPU-Hours (Higher is Better)")
    ax_a.set_ylabel("Rollback Lost GPU-Hours (Lower is Better)")
    ax_a.set_title("(a) Scheduling Tradeoff: Useful Work vs. Rollback Loss")
    ax_a.grid(True, linestyle='--', alpha=0.3)
    ax_a.set_xlim(96200, 96460)
    ax_a.set_ylim(450, 2200)

    ann_coords = {
        "CLQ": (96413, 545),
        "SRGFS": (96342, 1285),
        "CLQ-PPO": (96325, 820),
        "SRU-PPO": (96355, 2030),
        "SRQ-PPO": (96275, 1140),
        "SRAC": (96318, 520),
    }
    for name, work, lost, _, _, _, _, _, _, col in rl_models:
        if name == "SRU-PPO":
            ax_a.annotate(
                name,
                xy=(work, lost),
                xytext=(-14, 0),
                textcoords="offset points",
                fontsize=8.5,
                color=COLORS['dark_gray'],
                ha="right",
                va="center",
            )
            continue
        tx, ty = ann_coords[name]
        ax_a.annotate(name, xy=(work, lost), xytext=(tx, ty),
                     textcoords='data', fontsize=8.5, fontweight='bold' if 'SRAC' in name else 'normal',
                     color=COLORS['primary'] if 'SRAC' in name else COLORS['dark_gray'],
                     arrowprops=dict(arrowstyle="->", color='#888888', lw=0.7))

    # Panel B: HP Gang Violation Reliability vs Constraint Budget
    rl_short = ["CLQ", "SRGFS", "CLQ-PPO", "SRU-PPO", "SRQ-PPO", "SRAC"]
    viols = [0.2145, 0.4826, 1.8231, 1.0009, 1.3762, 1.5550]
    viol_err = [0.0, 0.1857, 0.0, 0.4024, 0.3650, 0.1857]

    x_b = np.arange(len(rl_short))
    bars = ax_b.bar(x_b, viols, yerr=viol_err, capsize=3,
                    color=[COLORS['blue'], COLORS['orange'], COLORS['purple'], COLORS['sky'], COLORS['gray'], COLORS['primary']],
                    edgecolor='black', linewidth=0.8, width=0.55)
    ax_b.axhline(2.0, color=COLORS['infeasible'], linestyle='--', linewidth=1.5, label='2.0% Safety Budget Ceiling')
    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(rl_short, fontsize=8.5)
    ax_b.set_ylabel("HP Gang Violation Rate (%)")
    ax_b.set_title("(b) High Priority Reliability Under Constraint Budget")
    ax_b.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax_b.set_ylim(0, 2.4)
    ax_b.legend(loc='upper left', frameon=True, framealpha=0.9)

    for i, (v, err) in enumerate(zip(viols, viol_err)):
        ax_b.text(i, v + err + 0.06, f"{v:.2f}%", ha='center', va='bottom', fontsize=8.2,
                  fontweight='bold' if 'SRAC' in rl_short[i] else 'normal',
                  color=COLORS['primary'] if 'SRAC' in rl_short[i] else COLORS['dark_gray'])

    save_all_formats(fig, "fig06_rl_comparison_tradeoffs")


# ============================================================================
# FIGURE 7: MECHANISM ABLATION GRID (Matching ablation_study_grid.png style)
# ============================================================================

def generate_fig07_mechanism_ablations() -> None:
    """Figure 7: Horizontal ablation bars with one shared, readable method axis."""
    fig, axes = plt.subplots(2, 2, figsize=(15.2, 10.4), sharey=True)
    fig.subplots_adjust(hspace=0.28, wspace=0.15, left=0.20, right=0.98,
                        bottom=0.07, top=0.96)

    ax_work, ax_queue = axes[0, 0], axes[0, 1]
    ax_lost, ax_evict = axes[1, 0], axes[1, 1]

    # Evaluated ablation methods across Tables 6 & 7
    variants = [
        "SRQGuard Quota",
        "SRQGuard Priority",
        "SRQGuard Full",
        "SRQGuard No Aging",
        "OrgQGuard",
        "SRGFS Mean",
        "SRMix Fixed",
        "SRMix",
    ]

    # Data from Table 6 & Table 7
    # Useful Spot Work Change vs Native GFS (%)
    delta_work_vals = [0.2566, 0.2511, 0.2565, 0.2590, 0.1158, 0.2272, 0.2296, 0.2593]
    # Spot Queue Delay Change vs Native GFS (%)
    delta_queue_vals = [-88.8, -86.2, -90.8, -92.4, -80.4, +11.8, -60.6, -66.8]
    # Lost Spot GPU-Hours (Rollback Loss)
    lost_work_vals = [2125.8, 1795.4, 1827.2, 2148.3, 1035.5, 1876.7, 1073.7, 814.7]
    lost_work_stds = [508.6, 453.4, 528.4, 324.4, 0.0, 219.0, 314.9, 309.7]
    # Spot Evictions Count
    eviction_vals = [340.3, 392.7, 486.7, 340.3, 233.0, 584.3, 229.0, 236.0]
    eviction_stds = [50.5, 25.1, 40.7, 24.8, 0.0, 120.4, 4.0, 6.6]

    y = np.arange(len(variants))
    h = 0.62

    # Colors: SRMix in primary crimson, SRQGuard in green, and SRGFS variants in blue.
    var_colors = [
        COLORS['green'], COLORS['green'], COLORS['green'], COLORS['light_green'], COLORS['gray'],
        COLORS['purple'], COLORS['blue'], COLORS['primary']
    ]

    # Subplot (a): Useful Spot Work Gain.
    ax_work.barh(y, delta_work_vals, h, color=var_colors, edgecolor='black', linewidth=0.8)
    ax_work.axvline(0, color='black', linewidth=0.9)
    ax_work.set_yticks(y)
    ax_work.set_yticklabels(variants, fontsize=9.0)
    ax_work.invert_yaxis()
    ax_work.set_xlabel("Useful Work Gain vs. Native GFS (%)")
    ax_work.set_title("(a) Spot Work Gain Across Mechanism Ablations")
    ax_work.grid(True, axis='x', linestyle='--', alpha=0.3)
    ax_work.set_xlim(0, 0.30)

    for i, v in enumerate(delta_work_vals):
        ax_work.text(v + 0.006, y[i], f"+{v:.2f}%", va='center', ha='left', fontsize=8.0,
                     fontweight='bold' if i == 7 else 'normal',
                     color=COLORS['primary'] if i == 7 else COLORS['dark_gray'])

    # Subplot (b): Spot Queue Delay Change.
    q_colors = [COLORS['green'] if v < 0 else COLORS['infeasible'] for v in delta_queue_vals]
    ax_queue.barh(y, delta_queue_vals, h, color=q_colors, edgecolor='black', linewidth=0.8)
    ax_queue.axvline(0, color='black', linewidth=0.9)
    ax_queue.set_yticks(y)
    ax_queue.tick_params(axis='y', labelleft=False)
    ax_queue.set_xlabel("Spot Queue Delay Change (%) (Lower is Better)")
    ax_queue.set_title("(b) Spot Queue Reduction Across Mechanism Ablations")
    ax_queue.grid(True, axis='x', linestyle='--', alpha=0.3)
    ax_queue.set_xlim(-105, 25)

    for i, v in enumerate(delta_queue_vals):
        x_pos = v - 2.0 if v < 0 else v + 1.5
        ax_queue.text(x_pos, y[i], f"{v:+.1f}%",
                      ha='right' if v < 0 else 'left', va='center', fontsize=8.0,
                      fontweight='bold' if i == 7 else 'normal',
                      color=COLORS['green'] if v < 0 else COLORS['infeasible'])

    # Subplot (c): Rollback Lost GPU Hours.
    ax_lost.barh(y, lost_work_vals, h, xerr=lost_work_stds, capsize=3,
                 color=var_colors, edgecolor='black', linewidth=0.8)
    ax_lost.set_yticks(y)
    ax_lost.set_yticklabels(variants, fontsize=9.0)
    ax_lost.set_xlabel("Rollback Lost GPU Hours (Lower is Better)")
    ax_lost.set_title("(c) Rollback Protection (Lower Loss is Superior)")
    ax_lost.grid(True, axis='x', linestyle='--', alpha=0.3)
    ax_lost.set_xlim(0, 2850)

    for i, v in enumerate(lost_work_vals):
        err = lost_work_stds[i]
        ax_lost.text(v + err + 55, y[i], f"{v:.0f}", va='center', ha='left', fontsize=8.0,
                     fontweight='bold' if i == 7 else 'normal',
                     color=COLORS['primary'] if i == 7 else COLORS['dark_gray'])

    # Subplot (d): Spot Eviction Counts.
    ax_evict.barh(y, eviction_vals, h, xerr=eviction_stds, capsize=3,
                  color=var_colors, edgecolor='black', linewidth=0.8)
    ax_evict.set_yticks(y)
    ax_evict.tick_params(axis='y', labelleft=False)
    ax_evict.set_xlabel("Spot Evictions Count (Lower is Better)")
    ax_evict.set_title("(d) Preemption Evictions (Lower Count is Superior)")
    ax_evict.grid(True, axis='x', linestyle='--', alpha=0.3)
    ax_evict.set_xlim(0, 760)

    for i, v in enumerate(eviction_vals):
        err = eviction_stds[i]
        ax_evict.text(v + err + 15, y[i], f"{v:.0f}", va='center', ha='left', fontsize=8.0,
                      fontweight='bold' if i == 7 else 'normal',
                      color=COLORS['primary'] if i == 7 else COLORS['dark_gray'])

    save_all_formats(fig, "fig07_mechanism_ablations")


# ============================================================================
# FIGURE 8: SRAC DYNAMIC POLICY BEHAVIOR UNDER HIGH-PRESSURE STRESS WINDOW
# ============================================================================

def generate_fig08_srac_policy_dynamics() -> None:
    """Figure 8: Real SRAC validation actions in the highest HP queue window."""
    fig = plt.figure(figsize=(14.8, 13.4))
    gs = fig.add_gridspec(
        8, 1,
        height_ratios=[3.1, 0.42, 3.1, 0.42, 3.1, 0.42, 3.1, 0.60],
        hspace=0.22,
        left=0.08,
        right=0.98,
        bottom=0.055,
        top=0.98,
    )
    ax_c1 = fig.add_subplot(gs[0, 0])
    ax_c2 = fig.add_subplot(gs[2, 0], sharex=ax_c1)
    ax_c3 = fig.add_subplot(gs[4, 0], sharex=ax_c1)
    ax_c4 = fig.add_subplot(gs[6, 0], sharex=ax_c1)
    legend_axes = [fig.add_subplot(gs[row, 0]) for row in (1, 3, 5, 7)]
    for legend_ax in legend_axes:
        legend_ax.axis('off')

    validation_raw = os.environ.get("SRAC_VALIDATION_JSON")
    if not validation_raw:
        raise FileNotFoundError(
            "Set SRAC_VALIDATION_JSON to one completed SRAC validation.json ledger."
        )
    validation_path = Path(validation_raw).expanduser().resolve()
    if not validation_path.is_file():
        raise FileNotFoundError(f"SRAC validation ledger not found: {validation_path}")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    actions = validation.get("actions", [])
    if len(actions) < 864:
        raise ValueError("Figure 8 requires at least 72 hours of five minute validation actions.")

    hp_queue_cost = np.asarray(
        [float(row["constraint_costs_raw"][2]) for row in actions], dtype=np.float64
    )
    window_steps = 72 * 12
    pressure = np.convolve(hp_queue_cost, np.ones(window_steps), mode="valid")
    start = int(np.argmax(pressure))
    rows = actions[start:start + window_steps]
    hours = np.arange(window_steps, dtype=np.float64) / 12.0

    def executed(name: str) -> np.ndarray:
        return np.asarray(
            [float(row["executed_action"][name]) for row in rows], dtype=np.float64
        )

    srlinear_quota = executed("srlinear_quota")
    contextual_quota = executed("contextual_quota")
    target_quota = executed("target_quota")
    uncertainty = executed("uncertainty_gpu")
    alpha_mix = executed("executed_alpha")
    beta_deduct = executed("executed_beta")
    shielded = np.asarray(
        [bool(row["executed_action"]["shielded"]) for row in rows], dtype=bool
    )
    costs = np.asarray(
        [row["constraint_costs_raw"] for row in rows], dtype=np.float64
    )
    display = slice(None, None, 2)

    # Subpanel A: causal quota endpoints supplied to the policy.
    ax_c1.plot(hours[display], contextual_quota[display], color=COLORS['dark_gray'],
               lw=1.7, linestyle=':', label='CLQ endpoint')
    ax_c1.plot(hours[display], srlinear_quota[display], color=COLORS['primary'],
               lw=1.8, label='SRLinear quota')
    ax_c1.fill_between(
        hours,
        np.maximum(0.0, srlinear_quota - uncertainty),
        srlinear_quota + uncertainty,
        color=COLORS['primary'],
        alpha=0.18,
        label='SRLinear uncertainty range',
    )
    ax_c1.set_ylabel("Quota Endpoint\n(GPUs)", fontsize=10.5)
    ax_c1.set_title("(a) Causal Quota Endpoints and SRLinear Uncertainty",
                    fontsize=11.5, pad=7)
    ax_c1.grid(True, linestyle='--', alpha=0.25)
    handles, labels = ax_c1.get_legend_handles_labels()
    legend_axes[0].legend(handles, labels, loc='center', ncol=3, fontsize=9.5,
                          frameon=False, handlelength=2.8, columnspacing=2.0)

    # Subpanel B: executed quota and its deviation from the reference endpoint.
    ax_c2.plot(hours[display], contextual_quota[display], color=COLORS['dark_gray'],
               lw=1.7, linestyle=':', label='CLQ endpoint')
    ax_c2.plot(hours[display], target_quota[display], color=COLORS['green'],
               lw=1.9, label='SRAC executed quota')
    ax_c2.fill_between(
        hours,
        target_quota,
        contextual_quota,
        where=(contextual_quota > target_quota),
        color=COLORS['green'],
        alpha=0.22,
        label='Protective quota deduction',
    )
    ax_c2.set_ylabel("Spot Quota\n(GPUs)", fontsize=10.5)
    ax_c2.set_title("(b) CLQ and Executed SRAC Spot Quotas", fontsize=11.5, pad=7)
    ax_c2.grid(True, linestyle='--', alpha=0.25)
    handles, labels = ax_c2.get_legend_handles_labels()
    legend_axes[1].legend(handles, labels, loc='center', ncol=3, fontsize=9.5,
                          frameon=False, handlelength=2.8, columnspacing=2.0)

    # Subpanel C: executed policy actions after ramp and risk protection.
    ax_c3.plot(hours[display], alpha_mix[display], color=COLORS['purple'], lw=1.8,
               label='Mixture weight $\\alpha_t$')
    ax_c3.plot(hours[display], beta_deduct[display], color=COLORS['orange'],
               lw=1.8, linestyle='-.', label='Uncertainty scale $\\beta_t$')
    ax_c3.set_ylabel("Action Scale\n$[0, 1]$", fontsize=10.5)
    ax_c3.set_title("(c) Learned Quota Mixture and Uncertainty Actions",
                    fontsize=11.5, pad=7)
    ax_c3.set_ylim(0, 1.1)
    ax_c3.grid(True, linestyle='--', alpha=0.25)
    handles, labels = ax_c3.get_legend_handles_labels()
    legend_axes[2].legend(handles, labels, loc='center', ncol=2, fontsize=9.5,
                          frameon=False, handlelength=2.8, columnspacing=2.4)

    # Subpanel D: risk events recorded by the validation environment.
    hp_violations = hours[costs[:, 0] > 0.0]
    rollback_events = hours[costs[:, 1] > 0.0]
    shield_events = hours[shielded]
    ax_c4.scatter(shield_events, np.full_like(shield_events, 2.0), marker='|', s=65,
                  color=COLORS['green'], label='Safety shield actions')
    ax_c4.scatter(hp_violations, np.full_like(hp_violations, 1.0), marker='|', s=85,
                  color=COLORS['primary'], label='HP service violations')
    ax_c4.scatter(rollback_events, np.zeros_like(rollback_events), marker='x', s=45,
                  color=COLORS['orange'], label='Rollback loss events')
    ax_c4.set_yticks([0, 1, 2])
    ax_c4.set_yticklabels(['Rollback loss', 'HP violations', 'Shield actions'], fontsize=8.5)
    ax_c4.set_ylim(-0.5, 2.5)
    ax_c4.set_xlabel("Elapsed Time in the 72 Hour Stress Window (Hours)", fontsize=10.5)
    ax_c4.set_title("(d) Validation Risk Events and Safety Shield Activity", fontsize=11.5, pad=7)
    ax_c4.grid(True, axis='x', linestyle='--', alpha=0.25)
    ax_c4.set_xlim(0, 72)
    ax_c4.set_xticks(np.arange(0, 73, 6))
    handles, labels = ax_c4.get_legend_handles_labels()
    legend_axes[3].legend(handles, labels, loc='center', ncol=3, fontsize=9.5,
                          frameon=False, handlelength=2.8, columnspacing=2.4)

    for ax in (ax_c1, ax_c2, ax_c3):
        ax.tick_params(axis='x', labelbottom=False)
    for ax in (ax_c1, ax_c2, ax_c3, ax_c4):
        ax.tick_params(axis='both', labelsize=9.5)

    save_all_formats(fig, "fig08_srac_policy_dynamics")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main() -> None:
    print("=" * 70)
    print("GENERATING PUBLICATION-QUALITY FIGURES FOR PREEMPTIBLE GPU SCHEDULING")
    print("=" * 70)
    generate_fig01_srlinear_architecture()
    generate_fig02_srac_system_architecture()
    generate_fig03_srac_rl_agent_architecture()
    generate_fig04_forecast_benchmark_efficiency()
    generate_fig05_non_rl_scheduling_comparison()
    generate_fig06_rl_comparison_tradeoffs()
    generate_fig07_mechanism_ablations()
    generate_fig08_srac_policy_dynamics()
    print("=" * 70)
    print("ALL 8 FIGURES SUCCESSFULLY PRODUCED IN PDF, SVG, AND 300 DPI PNG!")
    print("=" * 70)

if __name__ == "__main__":
    main()
