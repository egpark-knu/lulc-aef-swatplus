#!/usr/bin/env python3
"""
p2_14f_make_figures_era5.py
Generate publication figures for LULC_R1 manuscript based on unified ERA5 results.

Inputs:
  compare_orig_vs_patched_era5.json (from p2_14e run)

Outputs → /Users/eungyupark/Dropbox/Aohan/Paper/LULC_R1/
  fig_et_decomposition_era5.png      — ΔET decomposition (eplant/esoil/ecanopy)
  fig_waterbalance_era5.png          — Full WB delta bar chart
  table6_era5_updated.md             — Markdown table (copy-paste ready)
  table6_era5_updated.csv            — CSV for external editing
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path("/Users/eungyupark/Dropbox/myproj/dev_260402_LULC")
PHASE2 = PROJECT / "phase2"
CMP_JSON = PHASE2 / "data" / "continuous_native_experiment_urbanveg" / "compare_orig_vs_patched_era5.json"

OUT_DIR = Path("/Users/eungyupark/Dropbox/Aohan/Paper/LULC_R1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(CMP_JSON) as f:
    D = json.load(f)

SITES = ("hwaseong", "sejong")
SITE_LABEL = {"hwaseong": "Hwaseong (urban 41→47%)", "sejong": "Sejong (urban 14→17%)"}
SCENARIOS = ("Static-2017", "Continuous-Native", "Static-2024")
SCEN_SHORT = {"Static-2017": "S-2017", "Continuous-Native": "Continuous", "Static-2024": "S-2024"}

# ── Figure 1: ΔET decomposition ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
components = ("eplant", "esoil", "ecanopy")
comp_label = {"eplant": r"$\Delta$plant T", "esoil": r"$\Delta$soil E", "ecanopy": r"$\Delta$canopy E"}
comp_color = {"eplant": "#2E7D32", "esoil": "#C62828", "ecanopy": "#1565C0"}

for ax, site in zip(axes, SITES):
    x = np.arange(len(SCENARIOS))
    width = 0.25
    for i, comp in enumerate(components):
        vals = [D["sites"][site][s]["delta_patched_minus_orig"][comp] for s in SCENARIOS]
        ax.bar(x + (i - 1) * width, vals, width,
               label=comp_label[comp], color=comp_color[comp], edgecolor="black", linewidth=0.5)
    # Net ΔET as black dots
    dnets = [D["sites"][site][s]["delta_patched_minus_orig"]["et"] for s in SCENARIOS]
    ax.plot(x, dnets, "o-", color="black", linewidth=2, markersize=8,
            markerfacecolor="white", markeredgewidth=2, label=r"Net $\Delta$ET", zorder=10)
    for xi, dn in zip(x, dnets):
        ax.annotate(f"{dn:+.0f}", (xi, dn), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9, fontweight="bold")

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([SCEN_SHORT[s] for s in SCENARIOS])
    ax.set_title(SITE_LABEL[site], fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    if site == "hwaseong":
        ax.set_ylabel(r"$\Delta$ (mm yr$^{-1}$), patched $-$ orig")

axes[0].legend(loc="lower left", fontsize=9, framealpha=0.95)
fig.suptitle("ET partitioning shift: urban plnt_com = past_comm (patched) vs null (orig)\n"
             "Unified ERA5-Land climate, 8-year mean (2017–2024)",
             fontsize=11, y=1.02)
fig.tight_layout()
f1 = OUT_DIR / "fig_et_decomposition_era5.png"
fig.savefig(f1, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {f1}")

# ── Figure 2: Full water balance deltas ────────────────────────────────
wb_comps = ("et", "surq_gen", "latq", "perc", "wateryld")
wb_label = {"et": "ET", "surq_gen": "Surf runoff", "latq": "Lateral Q",
            "perc": "Percolation", "wateryld": "Water yield"}
wb_color = {"et": "#D84315", "surq_gen": "#1976D2", "latq": "#00838F",
            "perc": "#6A1B9A", "wateryld": "#283593"}

fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey="row")
for col, scen in enumerate(SCENARIOS):
    for row, site in enumerate(SITES):
        ax = axes[row][col]
        vals = [D["sites"][site][scen]["delta_patched_minus_orig"][c] for c in wb_comps]
        colors = [wb_color[c] for c in wb_comps]
        bars = ax.bar(wb_comps, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="gray", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    v + (2 if v >= 0 else -2),
                    f"{v:+.0f}",
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=8, fontweight="bold")
        ax.set_xticklabels([wb_label[c] for c in wb_comps], rotation=25, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        if row == 0:
            ax.set_title(SCEN_SHORT[scen], fontsize=11, fontweight="bold")
        if col == 0:
            ax.set_ylabel(f"{SITE_LABEL[site]}\n"+r"$\Delta$ (mm yr$^{-1}$)", fontsize=10)

fig.suptitle("Water-balance response to urban-pervious plant-community patch (patched $-$ orig)\n"
             "Unified ERA5-Land climate",
             fontsize=12, y=1.00)
fig.tight_layout()
f2 = OUT_DIR / "fig_waterbalance_era5.png"
fig.savefig(f2, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {f2}")

# ── Table 6 (markdown + csv) ───────────────────────────────────────────
rows = [("Site", "Scenario", "P",
         "ET_orig", "ET_patched", "ΔET",
         "Δeplant", "Δesoil", "Δecanopy",
         "Δsurq", "Δlatq", "Δperc", "Δwateryld")]
for site in SITES:
    for scen in SCENARIOS:
        b = D["sites"][site][scen]
        o = b["orig"]; p = b["patched"]; d = b["delta_patched_minus_orig"]
        rows.append((
            site.capitalize(), scen,
            f"{o['precip']:.1f}",
            f"{o['et']:.1f}", f"{p['et']:.1f}", f"{d['et']:+.1f}",
            f"{d['eplant']:+.1f}", f"{d['esoil']:+.1f}", f"{d['ecanopy']:+.1f}",
            f"{d['surq_gen']:+.1f}", f"{d['latq']:+.1f}",
            f"{d['perc']:+.1f}", f"{d['wateryld']:+.1f}",
        ))

# CSV
csv_path = OUT_DIR / "table6_era5_updated.csv"
with open(csv_path, "w") as f:
    for r in rows:
        f.write(",".join(r) + "\n")
print(f"Wrote {csv_path}")

# Markdown
md_path = OUT_DIR / "table6_era5_updated.md"
hdr = rows[0]; data = rows[1:]
md = ["# Table 6. Patched vs Orig urban-pervious template (unified ERA5-Land, 8-yr mean 2017–2024)",
      "",
      "Units: mm yr⁻¹. Δ = patched − orig.",
      "",
      "| " + " | ".join(hdr) + " |",
      "|" + "|".join(["---"] * len(hdr)) + "|"]
for r in data:
    md.append("| " + " | ".join(r) + " |")
md += ["",
       "**Climate**: unified ERA5-Land (hourly → daily aggregate via Open-Meteo API) for both Hwaseong and Sejong — replaces prior mix of KMA Seoul 108 ASOS + ERA5.",
       "",
       "**Patch**: urban LUMs (urml/urhd/ucom/utrn/uidu/urld) `plnt_com = past_comm` instead of `null` (SWAT+ editor v3.1.0 default).",
       "",
       "**Key observation**: under unified ERA5, patched template yields lower ET (−11 to −84 mm yr⁻¹) than orig in *all* scenarios. Mechanism: eplant↑ (grass cover restores transpiration) + esoil↓↓ (grass shades soil, suppresses bare-soil evap). The esoil reduction dominates, so net ET ↓ and water-yield ↑."]
md_path.write_text("\n".join(md))
print(f"Wrote {md_path}")

print("\nAll figures + table written to:", OUT_DIR)
