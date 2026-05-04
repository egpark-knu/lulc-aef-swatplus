# AEF-SWAT+ Dynamic LULC Framework

**Automated annual land-use forcing for SWAT+ watershed models via AlphaEarth Foundations embeddings**

## Overview

This repository contains the reproducibility package for:

> Jin, A., Park, E., Kim, T., & Park, J. (2026). From Satellite Embeddings to Dynamic Hydrology: An Automated AEF-SWAT+ Land Use Update Framework. *Environmental Modelling & Software* (under review).

The framework uses AlphaEarth Foundations (AEF) satellite embeddings, publicly available in Google Earth Engine at 10 m annual resolution, to estimate six-class LULC compositions and translate them into land-use updates for SWAT+ hydrological response units (HRUs). It is evaluated in two Korean basins — Sejong and Hwaseong — with contrasting urbanization trajectories (2017–2024).

This public snapshot contains the reproducibility package associated with the submission. Internal logs, raw agent scratch outputs, and full SWAT+ model files are intentionally excluded.

## Quick Start

```bash
conda create -n lulc-aef python=3.11 && conda activate lulc-aef
pip install -r requirements.txt

# Authenticate GEE (one-time)
earthengine authenticate

# Phase 1: Extract AEF embeddings and classify LULC
python poc_01_extract_embeddings.py
python classification/02_classify_sejong.py

# Phase 2: Run SWAT+ sensitivity experiments
python swat_integration/03_annual_dynamic_experiment.py
```

## Key Results

| Metric | Sejong | Hwaseong |
|---|---|---|
| Urban fraction change (2017–2024) | +2.3 pp | +5.7 pp |
| Overall classification accuracy | 98.1% | 97.7% |
| Lateral flow divergence (Dynamic vs Static-2017) | <2.5% | +14.2% |
| Total water yield difference | <1% | <1% |
| MODIS ET hierarchy confirmed | forest > cropland > urban | forest > cropland > urban |

## Framework Components

### Phase 1: AEF → LULC Classification
- AEF embedding extraction from GEE (64-dim, 10 m annual)
- Six-class LULC linear-probe classifier (logistic regression)
- Training labels: ESA WorldCover 2021 ∩ Dynamic World 2021 (stable pixels)
- Spatial block cross-validation (250/500/1,000/2,000 m blocks)
- Cross-year transfer (2017–2024)

### Phase 2: SWAT+ Integration
- Three-scenario design: Static-2017 / Annual-Dynamic / Static-2024
- HRU lu_mgt annual reassignment
- ERA5-Land climate forcing (Open-Meteo archive API, free)
- MODIS MOD16A2GF v6.1 ET external validation

## Repository Structure

```
├── classification/          # Phase 1: AEF → LULC
│   ├── 01_aoi_sampling.py   # AOI setup and stable pixel sampling
│   ├── 02_classify_sejong.py  # Sejong linear-probe classifier + spatial block CV
│   ├── 03_geumho_portability.py  # Geumho basin portability screening
│   ├── 04_lup_dat_generator.py  # SWAT+ land-use (.dat) file generator
│   ├── 05_classify_hwaseong.py  # Hwaseong classification
│   ├── aoi_config.py        # AOI geometry utilities
│   └── runtime_safety.py    # Memory-safe GEE execution utilities
│
├── swat_integration/        # Phase 2: SWAT+ scenarios
│   ├── 01_sejong_build.py   # Sejong SWAT+ model setup
│   ├── 02_hwaseong_build.py # Hwaseong full experiment
│   ├── 03_annual_dynamic_experiment.py  # Core sensitivity experiment
│   ├── 04_era5_unified_rerun.py  # ERA5-Land climate unification
│   ├── 05_make_figures.py   # Water balance figures
│   └── 06_modis_et_validation.py  # MODIS ET class-stratified comparison
│
├── figures/                 # Paper figures (Figures 1–8)
├── poc_01_extract_embeddings.py  # Proof-of-concept: embedding extraction
├── poc_02_pca_visualization.py   # PoC: PCA temporal visualization
├── poc_03_lulc_classification.py # PoC: end-to-end classification demo
├── requirements.txt
└── README.md
```

## Data Sources

| Dataset | Source | Access |
|---|---|---|
| AEF embeddings | `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` | GEE (public) |
| ESA WorldCover 2021 | `ESA/WorldCover/v200` | GEE (public) |
| Dynamic World 2021 | `GOOGLE/DYNAMICWORLD/V1` | GEE (public) |
| MODIS MOD16A2GF | `MODIS/061/MOD16A2GF` | GEE (public) |
| ERA5-Land climate | Open-Meteo archive API | Free, no key |
| SWAT+ binary | swatplus.org | Free download |

## Reproducibility Notes

- All GEE operations use `seed=42` for deterministic sampling
- Classification: logistic regression, C-grid logspace(−3, 3, 10), liblinear solver
- SWAT+ experiments run in `/tmp/` to avoid cloud sync overhead
- ERA5-Land forcing fetched via Open-Meteo API (no API key required)
- Streamflow data (WAMIS) via public API (no key required)

## License

No open-source license is granted in this public snapshot. This repository is provided for manuscript peer review and non-commercial academic reproducibility. Reuse, redistribution, or commercial use requires written permission from the authors.
