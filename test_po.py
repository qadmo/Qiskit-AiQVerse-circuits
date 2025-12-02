"""
Quantum Financial Model Template
Model: Quantum Portfolio Optimization
TargetType: Quantum
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np

# Number of assets/qubits
num_assets = 4

# Create quantum circuit for portfolio selection
qc = QuantumCircuit(num_assets, num_assets)

# Initialize superposition (all portfolio combinations)
for i in range(num_assets):
    qc.h(i)

# Add entanglement (correlations between assets)
for i in range(num_assets - 1):
    qc.cx(i, i + 1)

# Apply phase rotation (based on expected returns)
returns = [0.05, 0.08, 0.12, 0.06]  # Expected returns
for i, ret in enumerate(returns):
    qc.rz(ret * np.pi, i)

# Inverse entanglement
for i in range(num_assets - 2, -1, -1):
    qc.cx(i, i + 1)

# Measure all qubits
qc.measure(range(num_assets), range(num_assets))

# The circuit will be executed by IBM Quantum
# Results (counts) will be processed by the platform

# IMPORTANT: Set custom_results for additional metadata
custom_results = {
    'metrics': {
        'num_qubits': num_assets,
        'circuit_depth': qc.depth(),
        'num_gates': qc.size()
    },
    'parameters': {
        'expected_returns': returns
    }
}

print(f"Circuit created with {num_assets} qubits")
print(f"Circuit depth: {qc.depth()}")
