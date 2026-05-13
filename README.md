# HEC Pyrochlore Oxide Property Predictor

Machine-learning pipeline for predicting **lattice parameter** and **thermal conductivity**
of high-entropy pyrochlore oxides (A₂B₂O₇) from composition alone.

---

## Project Structure

```
HECPyrochlore/
├── data/
│   ├── raw/
│   │   ├── Sample_Properties_Safin_Feb_2026.csv   ← experimental data
│   │   ├── HECPyrochlore_latt_data_ICSD.csv       ← pulled from UTK ICSD database
│   │   ├── notebookLM_dataset.csv                 ← NotebookLM extraction
│   │   └── Jordan_pyrochlore_data.csv             ← pulled data from Jordan
│   └── processed/
│       └── comined_pyrochlore.csv
│
├── src/
│   ├── features/
│   │   └── build_features.py    ← all feature engineering
│   ├── data/
│   │   └── load_data.py         ← data loading & cleaning
│   └── build_models/
│       └── train_model.py       ← training, CV, feature importance, plots
│
└── models/                      ← EXECUTABLE SCRIPTS (run these)
    ├── predict_lattice.py        ★ Predict lattice parameter interactively
    ├── predict_thermal.py        ★ Predict thermal conductivity interactively
    ├── validate_lattice.py       ★ Full validation + feature importance report
    ├── validate_thermal.py       ★ Full validation + feature importance report
    ├── data_validation.py        ★ Add more validation output
    │
    ├── lattice_param/            ← auto-created on first run
    │   ├── lattice_param_model.pkl
    │   ├── lattice_param_scaler.pkl
    │   ├── lattice_param_metadata.json
    │   ├── lattice_parity.png
    │   ├── lattice_cv_comparison.png
    │   ├── lattice_feature_importance.png
    │   └── lattice_validation_report.txt
    │
    └── thermal_cond/             ← auto-created on first run
        ├── thermal_cond_model.pkl
        ├── thermal_cond_scaler.pkl
        ├── thermal_cond_metadata.json
        ├── thermal_parity.png
        ├── thermal_cv_comparison.png
        ├── thermal_feature_importance.png
        └── thermal_validation_report.txt
```

---

## Requirements

```
Python >= 3.9
pandas
numpy
scikit-learn
matplotlib
```

Install with:
```bash
pip install pandas numpy scikit-learn matplotlib
```

---

## Quick Start

Run all scripts from the **project root** (`HECPyrochlore/`).

### Predict Lattice Parameter

**Interactive mode** (prompts for A-site and B-site elements):
```bash
python models/predict_lattice.py
```

**CLI mode** (single prediction):
```bash
python models/predict_lattice.py --a "Pr,Sm,Gd,Ho,Lu" --b "Ti"
python models/predict_lattice.py --a "La,Gd,Lu" --b "Ti,Zr"
```

**Force re-train** before predicting:
```bash
python models/predict_lattice.py --train --a "Sm,Eu,Gd,Tb,Dy" --b "Ti"
```

### Predict Thermal Conductivity

**Interactive mode:**
```bash
python models/predict_thermal.py
```

**CLI mode** (lattice parameter is optional — auto-predicted if omitted):
```bash
python models/predict_thermal.py --a "Sm,Eu,Gd,Tb,Dy" --b "Ti" --lattice 10.185
python models/predict_thermal.py --a "La,Gd,Lu" --b "Ti,Zr"   # auto lattice
```

### Validate Models (Full Report + Plots)

```bash
python models/validate_lattice.py
python models/validate_thermal.py

# Custom number of CV folds:
python models/validate_lattice.py --splits 10
python models/validate_thermal.py --splits 5
```

Each validation script prints:
- Cross-validation R², RMSE, MAE for all 4 models
- Hold-out test-set metrics for the best model
- Top-15 features ranked by **permutation importance** and tree MDI
- Feature-group analysis (A-site vs. B-site vs. cross-site)
- Saves 3 plots + a full text report to `models/<task>/`

---

## Supported Elements

**A-site** (8-coordination, rare-earth cations):
`La Ce Pr Nd Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Y`

**B-site** (6-coordination, transition metals):
`Ti Zr Hf Sn Ir Ce Nb`

Compositions are assumed **equiatomic** unless otherwise noted.

---

## Feature Engineering

All features are derived purely from elemental properties — no DFT required.

| Feature Group | Examples | Physical Meaning |
|---|---|---|
| A-site | `a_site_entropy`, `a_site_mean_radius`, `a_site_delta` | Cation size disorder, configurational entropy |
| B-site | `b_site_entropy`, `b_site_mean_radius`, `b_site_delta` | Octahedral-site chemistry |
| Cross-site | `a_b_radius_ratio`, `total_entropy`, `phonon_scattering_factor` | Pyrochlore stability, phonon scattering |
| Lattice-derived | `lattice_parameter`, `lattice_volume`, `density_theoretical` | Used for thermal model only |

Key feature definitions:
- **δ (delta)** = `sqrt(Σ xᵢ(1 - rᵢ/r̄)²)` — lattice distortion index
- **S_config** = `-R Σ xᵢ ln(xᵢ)` — configurational entropy (J/mol·K)
- **phonon_scattering_factor** = `S_config × δ_total` — composite phonon scattering proxy
- **r_A/r_B** = A-site to B-site mean ionic radius ratio (governs pyrochlore stability)

---

## Models

Four regressors are compared in cross-validation; the best is automatically selected:

| Model | Notes |
|---|---|
| **GradientBoosting** | Usually best; balanced bias-variance |
| **RandomForest** | Robust; good uncertainty via tree variance |
| **ExtraTrees** | Faster; slightly higher variance |
| **Ridge** | Linear baseline; useful for interpretability |

---

## Model Performance (on current dataset, 28 samples)

| Target | Best Model | CV R² | Test R² | Test RMSE |
|---|---|---|---|---|
| Lattice Parameter (Å) | GradientBoosting | 0.967 ± 0.037 | 0.969 | 0.0192 Å |
| Thermal Conductivity (W/m·K) | GradientBoosting | 0.904 ± 0.050 | 0.919 | 0.155 W/m·K |

> **Note:** With only 28 samples these scores will vary across random seeds and folds.
> Adding more experimental data will substantially improve reliability.

---

## Key Findings (Feature Importance)

**Lattice Parameter** — dominated by **B-site properties**:
1. `b_site_mean_molar_mass` — heavier B-site → larger unit cell
2. `b_site_n_elements` — B-site complexity drives distortion
3. `b_site_mean_radius` — direct structural effect

**Thermal Conductivity** — dominated by **cross-site and A-site disorder**:
1. `a_b_radius_ratio` — size mismatch drives phonon scattering
2. `a_site_delta` — A-site lattice distortion index
3. `b_site_en_mean` — B-site electronegativity (bond stiffness)
4. `phonon_scattering_factor` — composite entropy × disorder metric

---

## Adding New Data

Place additional CSV files in `data/raw/` with the same column format, then
update `src/data/load_data.py` → `RAW_FILE` to point to your file, or pass
the path directly:

```python
from src.data.load_data import get_lattice_dataset
X, y, names = get_lattice_dataset(filepath='data/raw/my_new_data.csv')
```

Force re-train with the `--train` flag on any predict script after adding data.
