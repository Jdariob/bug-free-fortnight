import numpy as np
from chempy import Substance, Reaction, ReactionSystem
from chempy.kinetics.rates import Arrhenius
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# 1. Define Initial Conditions and Variations
# B: Low-complexity biogenic units (mol/L)
B_substances = {
    'CH4': 0.4,  # Methane
    'NH3': 0.3,  # Ammonia
    'H2O': 0.2,  # Water
    'CO': 0.1    # Carbon monoxide
}

# H: Physical conditions with variations (original: electric discharge, new: UV radiation)
H_conditions = {
    'discharge_10kV': {'type': 'electric', 'energy': 10000, 'T': 300, 'weight': 0.5},  # 10 kV discharge
    'UV_254nm': {'type': 'radiation', 'energy': 4.88, 'T': 300, 'weight': 0.5}         # UV at 254 nm (eV)
}

# Initialize substances
substances = {k: Substance(k) for k in list(B_substances.keys()) + ['HCN', 'glycine', 'alanine']}

# 2. Define Reaction Network
reactions = [
    # CH4 + NH3 -> HCN + H2O (intermediate, energy-dependent)
    Reaction({'CH4': 1, 'NH3': 1}, {'HCN': 1, 'H2O': 1}, Arrhenius(A=1e-2, Ea=50000)),
    # HCN + H2O -> glycine (simplified)
    Reaction({'HCN': 2, 'H2O': 1}, {'glycine': 1}, Arrhenius(A=1e-3, Ea=60000)),
    # HCN + CO -> alanine (simplified)
    Reaction({'HCN': 1, 'CO': 1, 'H2O': 1}, {'alanine': 1}, Arrhenius(A=1e-3, Ea=62000))
]

rs = ReactionSystem(reactions, substances.values())

# 3. Simulate Reactions with Varied Conditions
n_iterations = 1000  # Fewer iterations for demonstration
initial_concentrations = {k: v * 1.0 for k, v in B_substances.items()}
condition_weights = [H_conditions[c]['weight'] for c in H_conditions]
conditions = np.random.choice(list(H_conditions.keys()), size=n_iterations, p=condition_weights)

# Store yields and conditions
yields = {'HCN': [], 'glycine': [], 'alanine': []}
condition_log = []

for i in range(n_iterations):
    conc = initial_concentrations.copy()
    cond = H_conditions[conditions[i]]
    T = cond['T']
    energy = cond['energy']
    # Adjust reaction rates based on condition (simplified scaling)
    rate_scale = 1.0 if cond['type'] == 'electric' else 0.8  # UV slightly less effective
    for r in reactions:
        rate = r.rate(conc, T=T) * rate_scale
        for reactant, stoich in r.reac.items():
            conc[reactant] = max(0, conc[reactant] - rate * stoich)
        for product, stoich in r.prod.items():
            conc[product] = conc.get(product, 0) + rate * stoich
    
    # Record yields and conditions
    for product in yields.keys():
        yields[product].append(conc.get(product, 0))
    condition_log.append(cond['type'])

# 4. Mass Spectrometry Simulation (Simplified)
# Simulate mass spec peaks for glycine (mw=75) and alanine (mw=89)
mass_range = np.arange(50, 100, 0.1)  # m/z range
glycine_spectrum = np.zeros_like(mass_range)
alanine_spectrum = np.zeros_like(mass_range)

for i, mz in enumerate(mass_range):
    glycine_spectrum[i] = np.mean([y for j, y in enumerate(yields['glycine']) if abs(mz - 75) < 0.5])
    alanine_spectrum[i] = np.mean([y for j, y in enumerate(yields['alanine']) if abs(mz - 89) < 0.5])

# Detect peaks
glycine_peaks, _ = find_peaks(glycine_spectrum, height=0.001)
alanine_peaks, _ = find_peaks(alanine_spectrum, height=0.001)

# 5. AI-Driven Reaction Modeling (Random Forest)
# Features: condition type (encoded), energy; Target: glycine yield
X = np.array([[1 if c == 'electric' else 0, H_conditions[conditions[i]]['energy']] for i, c in enumerate(condition_log)])
y = np.array(yields['glycine'])
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
predictions = rf.predict(X)

# Feature importance
importances = rf.feature_importances_
print(f"Feature Importances: Condition Type={importances[0]:.2f}, Energy={importances[1]:.2f}")

# 6. Validate Pullback Structure
# Check if yields are consistent across conditions (pullback holds)
glycine_discharge = np.mean([y for i, y in enumerate(yields['glycine']) if condition_log[i] == 'electric'])
glycine_uv = np.mean([y for i, y in enumerate(yields['glycine']) if condition_log[i] == 'radiation'])
print(f"Glycine Yield (Discharge): {glycine_discharge:.3f} mol/L")
print(f"Glycine Yield (UV): {glycine_uv:.3f} mol/L")
pullback_diff = abs(glycine_discharge - glycine_uv) / max(glycine_discharge, glycine_uv)
print(f"Pullback Consistency (relative diff): {pullback_diff:.2f}")

# 7. Visualization
# Yields by condition
plt.figure(figsize=(12, 6))
plt.scatter(range(n_iterations), yields['glycine'], c=['b' if c == 'electric' else 'r' for c in condition_log], 
            alpha=0.5, label='Glycine')
plt.xlabel('Iteration')
plt.ylabel('Concentration (mol/L)')
plt.title('Glycine Yields: Discharge (Blue) vs. UV (Red)')
plt.legend(['Discharge', 'UV'], loc='upper right')
plt.show()

# Mass spectrometry simulation
plt.figure(figsize=(10, 5))
plt.plot(mass_range, glycine_spectrum, label='Glycine (mw=75)')
plt.plot(mass_range, alanine_spectrum, label='Alanine (mw=89)')
plt.plot(mass_range[glycine_peaks], glycine_spectrum[glycine_peaks], 'x', color='black')
plt.plot(mass_range[alanine_peaks], alanine_spectrum[alanine_peaks], 'x', color='black')
plt.xlabel('m/z')
plt.ylabel('Intensity')
plt.title('Simulated Mass Spectrometry')
plt.legend()
plt.show()

# AI predictions vs. actual
plt.figure(figsize=(8, 4))
plt.scatter(y, predictions, alpha=0.5)
plt.plot([0, max(y)], [0, max(y)], 'r--')
plt.xlabel('Actual Glycine Yield')
plt.ylabel('Predicted Glycine Yield')
plt.title('AI Model: Actual vs. Predicted Yields')
plt.show()
