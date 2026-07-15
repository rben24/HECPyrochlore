"""
ML pipeline to fill missing values in pristine_pyrochlore.csv
using element_database.csv as an elemental feature source.

Gap analysis
------------
Column                        Missing   Strategy
-----------------------------  -------  ----------------------------------------
Thermal Conductivity (W/m/K)   41/82    ML  (RF + GBT ensemble)
Bulk Modulus (VRH)             41/82    ML
Shear Modulus (VRH)            41/82    ML
Youngs Modulus (VRH)           41/82    ML
Poisson Ratio                  41/82    ML
AEL Debye Temperature          41/82    ML
Thermal Expansion              41/82    ML
Ionic Radius A (Å)       2/82    Median imputation per Sample-A element
Formation Energy per Atom      82/82    Cannot train – left as NaN
Temperature                    82/82    Cannot train – left as NaN
Energy Above Hull              82/82    Cannot train – left as NaN

Outputs
-------
  pristine_pyrochlore_filled.csv   gap-filled dataset (original columns only)
  cv_summary.csv                   5-fold CV metrics per ML target
  feature_importances.csv          top-5 features per target
"""

import os, warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import shutil
from src.globals import PROJECT

warnings.filterwarnings("ignore")

DATA = PROJECT / 'data'
PRISTINE_DATA = DATA / 'processed' / 'pristine_pyrochlore.csv'
ELEMENT_DATA = DATA / 'raw' / 'element_database.csv'
OUTPUT_DIR = PROJECT / 'models' / 'pristine_fill'
OUTPUT_DIR.mkdir(exist_ok=True)

# ── 1. Load ────────────────────────────────────────────────────────────────────
pp = pd.read_csv(PRISTINE_DATA)
ed = pd.read_csv(ELEMENT_DATA)
print(f"Pyrochlore: {pp.shape}   |   Element DB: {ed.shape}")

# ── 2. Merge element-DB features for A-site and B-site ─────────────────────────
ed_feat_cols = [
    "Element", "Atomic Mass", "Electronegativity", "Atomic Radius",
    "Density", "Melting Point", "Thermal Expansion", "Thermal Conductivity",
    "Bulk Modulus", "Youngs Modulus", "Shear Modulus", "Poissons Ratio",
    "Metallic Radius", "Universal Anisotropy", "Band_Gap",
    "Is Magnetic", "Total Magnetization", "Vickers Hardness"
]
ed_feat_cols = [c for c in ed_feat_cols if c in ed.columns]  # guard against missing cols

ed_sub = ed[ed_feat_cols].copy()
ed_sub["Is Magnetic"] = pd.to_numeric(ed_sub["Is Magnetic"], errors="coerce")

ed_A = ed_sub.rename(columns={c: f"eA_{c}" for c in ed_feat_cols if c != "Element"})
ed_B = ed_sub.rename(columns={c: f"eB_{c}" for c in ed_feat_cols if c != "Element"})

pp = (pp
      .merge(ed_A, left_on="Sample A", right_on="Element", how="left").drop(columns=["Element"])
      .merge(ed_B, left_on="Sample B", right_on="Element", how="left").drop(columns=["Element"]))

print(f"After feature merge: {pp.shape}")

# ── 3. Fix Ionic Radius A (2 missing rows – Sn as A-site) ─────────────────────
missing_ir_A = pp["Ionic Radius A (Å)"].isnull()
print(f"\nRows with missing Ionic Radius A:\n{pp.loc[missing_ir_A, ['Composition','Sample A','Sample B']]}")

for elem in pp.loc[missing_ir_A, "Sample A"].unique():
    median_val = pp.loc[(pp["Sample A"] == elem) & (~missing_ir_A),
                        "Ionic Radius A (Å)"].median()
    if pd.isna(median_val):
        median_val = pp["Ionic Radius A (Å)"].median()
    pp.loc[(pp["Sample A"] == elem) & missing_ir_A, "Ionic Radius A (Å)"] = median_val
    print(f"  Ionic Radius A for {elem} → {median_val:.3f} Å  (median imputation)")

# ── 4. Feature engineering ─────────────────────────────────────────────────────
le = LabelEncoder()
pp["BGType_enc"] = le.fit_transform(pp["Band Gap Type"].astype(str))

# Compositional / structural cross-features
pp["EN_diff"]     = (pp["Electronegativity A"] - pp["Electronegativity B"]).abs()
pp["EN_sum"]      = pp["Electronegativity A"] + pp["Electronegativity B"]
pp["IR_ratio"]    = pp["Ionic Radius A (Å)"] / (pp["Ionic Radius B (Å)"] + 1e-9)
pp["IR_sum"]      = pp["Ionic Radius A (Å)"] + pp["Ionic Radius B (Å)"]
pp["IR_diff"]     = (pp["Ionic Radius A (Å)"] - pp["Ionic Radius B (Å)"]).abs()
pp["ox_sum"]      = pp["Oxidation State A"] + pp["Oxidation State B"]
pp["ox_product"]  = pp["Oxidation State A"] * pp["Oxidation State B"]
pp["lattice_vol"] = pp["Lattice Parameter (Å)"] ** 3
pp["gap_x_lat"]   = pp["Band Gap"] * pp["Lattice Parameter (Å)"]
pp["energy_gap"]  = pp["Energy per Atom"] * pp["Band Gap"]

# Elemental average / difference features
pp["mass_sum"]  = pp["eA_Atomic Mass"] + pp["eB_Atomic Mass"]
pp["mass_diff"] = (pp["eA_Atomic Mass"] - pp["eB_Atomic Mass"]).abs()
pp["melt_avg"]  = (pp["eA_Melting Point"].fillna(0) + pp["eB_Melting Point"].fillna(0)) / 2
pp["melt_diff"] = (pp["eA_Melting Point"] - pp["eB_Melting Point"]).abs()
pp["rad_diff"]  = (pp["eA_Atomic Radius"] - pp["eB_Atomic Radius"]).abs()
pp["kappa_avg"] = (pp["eA_Thermal Conductivity"].fillna(0) + pp["eB_Thermal Conductivity"].fillna(0)) / 2
pp["BM_avg"]    = (pp["eA_Bulk Modulus"].fillna(0) + pp["eB_Bulk Modulus"].fillna(0)) / 2
pp["SM_avg"]    = (pp["eA_Shear Modulus"].fillna(0) + pp["eB_Shear Modulus"].fillna(0)) / 2
pp["YM_avg"]    = (pp["eA_Youngs Modulus"].fillna(0) + pp["eB_Youngs Modulus"].fillna(0)) / 2
pp["PR_avg"]    = (pp["eA_Poissons Ratio"].fillna(0) + pp["eB_Poissons Ratio"].fillna(0)) / 2
pp["TE_avg"]    = (pp["eA_Thermal Expansion"].fillna(0) + pp["eB_Thermal Expansion"].fillna(0)) / 2
pp["MR_avg"]    = (pp["eA_Metallic Radius"].fillna(0) + pp["eB_Metallic Radius"].fillna(0)) / 2
pp["UA_sum"]    = (pp["eA_Universal Anisotropy"].fillna(0) + pp["eB_Universal Anisotropy"].fillna(0))
pp["VH_avg"]    = (pp["eA_Vickers Hardness"].fillna(0) + pp["eB_Vickers Hardness"].fillna(0)) / 2

# ── 5. Feature list ────────────────────────────────────────────────────────────
base_feats = [
    "Lattice Parameter (Å)", "lattice_vol",
    "Density", "Energy per Atom", "Enthalpy", "Magnetic Moment",
    "Band Gap", "BGType_enc", "Valence",
    "Ionic Radius A (Å)", "Ionic Radius B (Å)",
    "Electronegativity A", "Electronegativity B",
    "Oxidation State A", "Oxidation State B",
    "EN_diff", "EN_sum", "IR_ratio", "IR_sum", "IR_diff",
    "ox_sum", "ox_product", "gap_x_lat", "energy_gap"
]
elem_feats    = [c for c in pp.columns if c.startswith("eA_") or c.startswith("eB_")]
derived_feats = [
    "mass_sum", "mass_diff", "melt_avg", "melt_diff", "rad_diff",
    "kappa_avg", "BM_avg", "SM_avg", "YM_avg", "PR_avg",
    "TE_avg", "MR_avg", "UA_sum", "VH_avg"
]
all_features = base_feats + elem_feats + derived_feats
print(f"\nTotal features: {len(all_features)}")

# ── 6. Train / predict split ───────────────────────────────────────────────────
ml_targets = [
    "Thermal Conductivity (W/m/K)",
    "Bulk Modulus (GPa)",
    "Shear Modulus (GPa)",
    "Youngs Modulus (GPa)",
    "Poisson Ratio",
    "AEL Debye Temperature",
    "Thermal Expansion"
]

mask_missing = pp[ml_targets[0]].isnull()
train_idx = pp[~mask_missing].index    # 41 rows with known targets
pred_idx  = pp[mask_missing].index     # 41 rows to predict
print(f"Train rows: {len(train_idx)}   |   Predict rows: {len(pred_idx)}")

# Impute feature NaNs with column median
imputer = SimpleImputer(strategy="median")
X_all = pd.DataFrame(imputer.fit_transform(pp[all_features]),
                     columns=all_features, index=pp.index)
X_train = X_all.loc[train_idx]
X_pred  = X_all.loc[pred_idx]

# ── 7. Weighted RF + GBT ensemble per target ───────────────────────────────────
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_results  = {}
predictions = {}
importances = {}

print(f"\n{'='*72}")
print("  ENSEMBLE MODEL: RandomForest + GradientBoosting (weighted by CV R²)")
print(f"  5-fold cross-validation on {len(train_idx)} training samples")
print(f"{'='*72}")

for target in ml_targets:
    y_train = pp.loc[train_idx, target].values

    rf  = RandomForestRegressor(
        n_estimators=400, max_features="sqrt",
        min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    gbt = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.04,
        max_depth=4, subsample=0.8,
        min_samples_leaf=2, random_state=42
    )

    # Determine ensemble weights from individual CV R²
    w_rf  = max(cross_val_score(rf,  X_train, y_train, cv=kf, scoring="r2").mean(), 0.01)
    w_gbt = max(cross_val_score(gbt, X_train, y_train, cv=kf, scoring="r2").mean(), 0.01)
    w_sum = w_rf + w_gbt

    # Ensemble CV metrics
    cv_r2, cv_mae = [], []
    for tr_idx, val_idx in kf.split(X_train):
        Xtr, Xv = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        ytr, yv = y_train[tr_idx],      y_train[val_idx]
        rf.fit(Xtr, ytr);   p_rf  = rf.predict(Xv)
        gbt.fit(Xtr, ytr);  p_gbt = gbt.predict(Xv)
        p_ens = (w_rf * p_rf + w_gbt * p_gbt) / w_sum
        cv_r2.append(r2_score(yv, p_ens))
        cv_mae.append(mean_absolute_error(yv, p_ens))

    # Final fit on ALL training data → generate predictions
    rf.fit(X_train, y_train)
    gbt.fit(X_train, y_train)
    y_pred = (w_rf * rf.predict(X_pred) + w_gbt * gbt.predict(X_pred)) / w_sum

    cv_results[target] = {
        "CV R² (mean)":  round(np.mean(cv_r2),  4),
        "CV R² (std)":   round(np.std(cv_r2),   4),
        "CV MAE (mean)": round(np.mean(cv_mae), 6),
        "CV MAE (std)":  round(np.std(cv_mae),  6),
        "RF weight":     round(w_rf  / w_sum, 2),
        "GBT weight":    round(w_gbt / w_sum, 2),
    }
    predictions[target] = y_pred
    importances[target] = pd.Series(
        (rf.feature_importances_ + gbt.feature_importances_) / 2,
        index=all_features
    ).nlargest(5)

    print(f"\n  {target}")
    print(f"    CV R² : {np.mean(cv_r2):.3f} ± {np.std(cv_r2):.3f}")
    print(f"    CV MAE: {np.mean(cv_mae):.4g} ± {np.std(cv_mae):.4g}")
    print(f"    Weights → RF: {w_rf/w_sum:.2f}  |  GBT: {w_gbt/w_sum:.2f}")
    print(f"    Top features: {', '.join(importances[target].index[:4])}")

# ── 8. Write filled values back ────────────────────────────────────────────────
pp_filled = pp.copy()
for target in ml_targets:
    pp_filled.loc[pred_idx, target] = predictions[target]

# Formation Energy per Atom, Temperature, Energy Above Hull → still NaN
# (100% missing – no training data available for ML)

# ── 9. Output only original pyrochlore columns ─────────────────────────────────
original_cols = [
    "Composition", "Sample A", "Sample B",
    "Oxidation State A", "Oxidation State B",
    "Thermal Conductivity (W/m/K)", "Lattice Parameter (Å)", "Density",
    "Energy per Atom", "Formation Energy per Atom", "Enthalpy", "Magnetic Moment",
    "Band Gap", "Band Gap Type", "Valence",
    "Bulk Modulus (GPa)", "Shear Modulus (GPa)", "Youngs Modulus (GPa)",
    "Poisson Ratio", "AEL Debye Temperature", "Temperature",
    "Thermal Expansion", "Energy Above Hull",
    "Ionic Radius A (Å)", "Ionic Radius B (Å)",
    "Electronegativity A", "Electronegativity B",
    "Synthesis Method", "compound_type", "data_source"
]
out_df = pp_filled[original_cols].copy()

# ── 10. Save outputs ───────────────────────────────────────────────────────────
cv_df  = pd.DataFrame(cv_results).T
cv_df.index.name = "Target"

imp_rows = [
    {"Target": t, "Feature": f, "Avg Importance": round(v, 5)}
    for t, imp in importances.items()
    for f, v in imp.items()
]
imp_df = pd.DataFrame(imp_rows)

filled_path = OUTPUT_DIR / 'pristine_pyrochlore_filled.csv'
cv_path     = OUTPUT_DIR / 'cv_summary.csv'
imp_path    = OUTPUT_DIR / 'feature_importances.csv'

out_df.to_csv(filled_path, index=False)
cv_df.to_csv(cv_path)
imp_df.to_csv(imp_path, index=False)

# for src in [filled_path, cv_path, imp_path]:
#     shutil.copy(src, f"{OUTPUT_DIR}/{os.path.basename(src)}")

# ── Summary ────────────────────────────────────────────────────────────────────
still_missing = out_df.isnull().sum()
still_missing = still_missing[still_missing > 0]
print(f"\n{'='*72}")
print("  FILL SUMMARY")
print(f"{'='*72}")
print(f"  ML-filled (7 targets × 41 rows):   ✅")
print(f"  Ionic Radius A (2 rows):            ✅")
print(f"  Columns still NaN (no training data):")
for col, n in still_missing.items():
    print(f"    {col:<35} {n} rows")
print(f"\n  Saved: pristine_pyrochlore_filled.csv, cv_summary.csv, feature_importances.csv")
print("\n✅  Pipeline complete!")

print(f"\n{'='*72}")
print("  CROSS-VALIDATION SUMMARY")
print(f"{'='*72}")
print(cv_df[["CV R² (mean)","CV R² (std)","CV MAE (mean)","RF weight","GBT weight"]].to_string())
