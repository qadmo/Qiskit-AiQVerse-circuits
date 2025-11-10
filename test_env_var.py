"""
Test des variables d'environnement
"""
import os
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

print("=== Test Variables d'Environnement ===")
print(f"QUANTUMOPS_RUN_ID: {os.environ.get('QUANTUMOPS_RUN_ID', 'NON DÉFINI')}")
print(f"QUANTUMOPS_API_URL: {os.environ.get('QUANTUMOPS_API_URL', 'NON DÉFINI')}")
print(f"QUANTUMOPS_API_TOKEN présent: {bool(os.environ.get('QUANTUMOPS_API_TOKEN'))}")
print()

# Circuit minimal
qr = QuantumRegister(2, 'q')
cr = ClassicalRegister(2, 'c')
circuit = QuantumCircuit(qr, cr)
circuit.h(qr[0])
circuit.cx(qr[0], qr[1])
circuit.measure(qr, cr)

print("Circuit créé (pour satisfaire prepare_circuit)")
print("=== Fin du test ===")
