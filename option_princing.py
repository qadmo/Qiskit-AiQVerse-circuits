"""
Exemple : Extraire un circuit pour l'exécuter sur un QPU IBM

Ce script montre comment extraire un circuit du code d'option pricing
et le préparer pour une exécution sur un vrai QPU IBM.

Différence clé :
- Code d'optimisation (QAOA, VQE) : ❌ Ne peut PAS tourner sur QPU (algorithme itératif)
- Code avec circuit unique : ✅ PEUT tourner sur QPU (soumission unique)
"""

from qiskit import QuantumCircuit
from qiskit.circuit.library import LinearAmplitudeFunction
from qiskit_finance.circuit.library import LogNormalDistribution
import numpy as np

# ============================================================================
# PARTIE 1 : Construction du circuit (identique au notebook option pricing)
# ============================================================================

# Paramètres de l'option européenne
S = 2.0        # Spot price
K = 1.896      # Strike price
r = 0.05       # Risk-free rate
sigma = 0.4    # Volatility
T = 40 / 365   # Time to maturity

# Paramètres numériques
num_uncertainty_qubits = 3
low = np.array([0])
high = np.array([3])

# Construction de la distribution log-normale
mu = ((r - 0.5 * sigma**2) * T + np.log(S))
sigma_ = sigma * np.sqrt(T)
mean = np.exp(mu + sigma_**2 / 2)
variance = (np.exp(sigma_**2) - 1) * np.exp(2 * mu + sigma_**2)
stddev = np.sqrt(variance)

# Distribution
uncertainty_model = LogNormalDistribution(
    num_uncertainty_qubits,
    mu=mu,
    sigma=sigma_**2,
    bounds=tuple(zip(low, high))
)

# Fonction de payoff
c_approx = 0.25
breakpoints = [low[0], K]
slopes = [0, 1]
offsets = [0, 0]
f_min = 0
f_max = high[0] - K

european_call_objective = LinearAmplitudeFunction(
    num_uncertainty_qubits,
    slopes,
    offsets,
    domain=(low[0], high[0]),
    image=(f_min, f_max),
    breakpoints=breakpoints,
    rescaling_factor=c_approx,
)

# ============================================================================
# PARTIE 2 : Construction du circuit complet
# ============================================================================

num_qubits = european_call_objective.num_qubits
european_call = QuantumCircuit(num_qubits)
european_call.append(uncertainty_model, range(num_uncertainty_qubits))
european_call.append(european_call_objective, range(num_qubits))

print("Circuit créé avec succès!")
print(f"Nombre de qubits : {european_call.num_qubits}")
print(f"Profondeur : {european_call.depth()}")

# ============================================================================
# PARTIE 3 : Préparation pour QPU - AJOUT DES MESURES
# ============================================================================

# IMPORTANT : Pour exécuter sur un QPU, il faut ajouter des mesures
european_call_qpu = european_call.copy()
european_call_qpu.measure_all()

print("\n✅ Circuit prêt pour QPU (avec mesures)")
print(f"Nombre de bits classiques : {european_call_qpu.num_clbits}")

# ============================================================================
# PARTIE 4 : Exemple de soumission au QPU
# ============================================================================

print("\n" + "="*70)
print("CODE À COPIER DANS VOTRE PLATEFORME :")
print("="*70)

code_for_platform = '''
from qiskit import QuantumCircuit
from qiskit.circuit.library import LinearAmplitudeFunction
from qiskit_finance.circuit.library import LogNormalDistribution
import numpy as np

# Paramètres
S = 2.0
K = 1.896
r = 0.05
sigma = 0.4
T = 40 / 365
num_uncertainty_qubits = 3
low = np.array([0])
high = np.array([3])

# Distribution log-normale
mu = ((r - 0.5 * sigma**2) * T + np.log(S))
sigma_ = sigma * np.sqrt(T)
uncertainty_model = LogNormalDistribution(
    num_uncertainty_qubits,
    mu=mu,
    sigma=sigma_**2,
    bounds=tuple(zip(low, high))
)

# Fonction de payoff
c_approx = 0.25
breakpoints = [low[0], K]
slopes = [0, 1]
offsets = [0, 0]
f_min = 0
f_max = high[0] - K
european_call_objective = LinearAmplitudeFunction(
    num_uncertainty_qubits,
    slopes,
    offsets,
    domain=(low[0], high[0]),
    image=(f_min, f_max),
    breakpoints=breakpoints,
    rescaling_factor=c_approx,
)

# Construction du circuit
num_qubits = european_call_objective.num_qubits
circuit = QuantumCircuit(num_qubits)
circuit.append(uncertainty_model, range(num_uncertainty_qubits))
circuit.append(european_call_objective, range(num_qubits))

# AJOUT DES MESURES POUR QPU
circuit.measure_all()
'''

print(code_for_platform)

# ============================================================================
# PARTIE 5 : Instructions d'utilisation
# ============================================================================

print("\n" + "="*70)
print("INSTRUCTIONS POUR EXÉCUTER SUR QPU IBM :")
print("="*70)
print("""
1. Copiez le code ci-dessus dans votre plateforme QuantumOps

2. Allez sur la page du modèle → Cliquez sur "Exécuter"

3. Dans la modal, sélectionnez un backend IBM RÉEL :
   - ibm_fez (156 qubits)
   - ibm_torino (133 qubits)
   - ibm_marrakesh (156 qubits)

4. NE PAS sélectionner :
   - docker_aer_simulator (simulateur dans Docker)
   - aer_simulator (simulateur local)

5. Cliquez sur "Lancer l'exécution"

6. Le système va :
   ✅ Extraire automatiquement le circuit
   ✅ Le convertir en QASM
   ✅ Le soumettre au QPU IBM
   ✅ Afficher les résultats quand le job est terminé

REMARQUES IMPORTANTES :
- Les QPUs IBM ont des files d'attente → peut prendre plusieurs minutes
- Le circuit sera optimisé (transpilé) pour le QPU cible
- Vous verrez les counts (résultats de mesure) dans l'interface
- Pour l'option pricing complète, utilisez un simulateur (docker_aer_simulator)
  car l'amplitude estimation nécessite plusieurs circuits
""")

# ============================================================================
# PARTIE 6 : Comparaison avec code d'optimisation
# ============================================================================

print("\n" + "="*70)
print("POURQUOI LE CODE D'OPTIMISATION NE PEUT PAS TOURNER SUR QPU :")
print("="*70)
print("""
❌ CODE D'OPTIMISATION (Portfolio Optimization) :
   
   from qiskit_optimization.algorithms import SamplingVQE
   qaoa = SamplingVQE(...)
   result = qaoa.solve(qp)  # ← Appelle le QPU des dizaines/centaines de fois !
   
   Pourquoi ça ne marche pas :
   - QAOA/VQE génère des circuits différents à chaque itération
   - L'algorithme ajuste les paramètres basé sur les résultats précédents
   - Nécessite une boucle Python ↔ QPU impossible à exécuter sur IBM
   - File d'attente IBM rendrait ça impraticable (plusieurs heures)

✅ CODE AVEC CIRCUIT UNIQUE (Option Pricing) :
   
   circuit = QuantumCircuit(num_qubits)
   circuit.append(uncertainty_model, range(num_qubits))
   circuit.append(payoff_function, range(num_qubits))
   circuit.measure_all()  # ← Un seul circuit, une seule soumission !
   
   Pourquoi ça marche :
   - Un seul circuit à soumettre
   - Pas de boucle de feedback
   - Le QPU l'exécute et retourne les résultats
   - Temps d'attente acceptable (minutes, pas heures)

SOLUTION POUR L'AMPLITUDE ESTIMATION COMPLÈTE :
- Utilisez docker_aer_simulator pour l'algorithme complet
- OU extrayez juste le circuit pour voir les distributions sur QPU réel
""")

print("\n✅ Exemple créé avec succès!")
print("📁 Fichier : examples/extract_circuit_for_qpu.py")
