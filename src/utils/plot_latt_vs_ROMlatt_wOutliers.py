import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("../../data/processed/rom_hec_latt.csv")
df = pd.DataFrame(data)

X = df[['Lattice Parameter (Angstrom)']]
y = df['ROM_Lattice_Parameter']

model_l = LinearRegression()
model_l.fit(X, y)
y_pred = model_l.predict(X)

# Calculate R² score
r2 = r2_score(y, y_pred)

# Calculate residuals to identify outliers
residuals = y - y_pred.flatten()
std_residuals = np.std(residuals)
threshold = 2 * std_residuals  # Outliers are points > 2 standard deviations from the fit

# Identify outliers
outlier_mask = np.abs(residuals) > threshold
outliers = df[outlier_mask].copy()
outliers['Residual'] = residuals[outlier_mask]
outliers['Predicted_ROM'] = y_pred[outlier_mask].flatten()

# Display outliers with composition
print(f"R² Score: {r2:.4f}")
print(f"\nOutliers (Residual > {threshold:.4f}):")
print(outliers[['Composition', 'Lattice Parameter (Angstrom)', 'ROM_Lattice_Parameter',
                'Predicted_ROM', 'Residual']])

# Plot with outliers highlighted
plt.figure(figsize=(10, 6))
plt.scatter(X[~outlier_mask], y[~outlier_mask], color='blue', label='Data Points', alpha=0.6)
plt.scatter(X[outlier_mask], y[outlier_mask], color='red', label='Outliers', s=100, marker='x')
plt.plot(X, y_pred, color='green', label='Regression Line', linewidth=2)
plt.xlabel('Lattice Parameter (Angstrom)')
plt.ylabel('ROM_Lattice_Parameter')
plt.title(f'Linear Regression Plot (R² = {r2:.4f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("../../models/lattice_param_rom/latt_vs_rom_latt.png", dpi=300, bbox_inches='tight')
plt.show()