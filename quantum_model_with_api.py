"""
Quantum Model with API - Example Script
========================================

Ce script montre comment créer un modèle quantique qui publie des résultats
enrichis vers la plateforme via l'API QuantumOps.

La plateforme injecte automatiquement ces variables d'environnement :
- QUANTUMOPS_RUN_ID : ID de l'exécution en cours
- QUANTUMOPS_API_URL : URL de l'API de la plateforme
- QUANTUMOPS_API_TOKEN : Token d'authentification pour cette exécution

Avantages :
- Publier des résultats custom (eigenvalues, métriques, visualisations)
- Format JSON flexible
- Les résultats s'affichent automatiquement dans l'UI
"""

import os
import requests
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import numpy as np

# Récupérer les variables d'environnement injectées par la plateforme
RUN_ID = os.environ.get('QUANTUMOPS_RUN_ID')
API_URL = os.environ.get('QUANTUMOPS_API_URL')
API_TOKEN = os.environ.get('QUANTUMOPS_API_TOKEN')

print(f"=== QuantumOps API Example ===")
print(f"Run ID: {RUN_ID}")
print(f"API URL: {API_URL}")
print(f"Token présent: {'Oui' if API_TOKEN else 'Non'}")
print()

# 1. Créer et exécuter un circuit quantique
print("📊 Création du circuit quantique...")
qr = QuantumRegister(3, 'q')
cr = ClassicalRegister(3, 'c')
circuit = QuantumCircuit(qr, cr)

# Circuit de test : état de Bell + mesure sur 3 qubits
circuit.h(qr[0])  # Hadamard sur q0
circuit.cx(qr[0], qr[1])  # CNOT q0->q1
circuit.cx(qr[0], qr[2])  # CNOT q0->q2
circuit.measure(qr, cr)

print("Circuit créé avec succès")
print(circuit)
print()

# 2. Exécuter le circuit
print("⚛️  Exécution du circuit...")
simulator = AerSimulator()
job = simulator.run(circuit, shots=1024)
result = job.result()
counts = result.get_counts(circuit)

print(f"Résultats bruts : {counts}")
print()

# 3. Analyser les résultats
print("🔬 Analyse des résultats...")

# Calcul de métriques custom
total_shots = sum(counts.values())
probabilities = {state: count/total_shots for state, count in counts.items()}

# Calcul d'entropie (mesure de l'aléatoire)
entropy = -sum(p * np.log2(p) for p in probabilities.values() if p > 0)

# Calcul de fidélité simulée (exemple)
ideal_states = ['000', '111']
fidelity = sum(counts.get(state, 0) for state in ideal_states) / total_shots

# Eigenvalues simulées (exemple)
eigenvalues = [1.0, 0.5, -0.3]

print(f"Entropie : {entropy:.4f}")
print(f"Fidélité : {fidelity:.4f}")
print(f"Eigenvalues : {eigenvalues}")
print()

# 4. Publier les résultats enrichis via l'API
if RUN_ID and API_URL and API_TOKEN:
    print("📡 Publication des résultats vers QuantumOps...")
    
    try:
        # Construire le payload avec résultats enrichis
        payload = {
            "counts": counts,  # Résultats standards
            "eigenvalues": eigenvalues,  # Eigenvalues calculées
            "custom_metrics": {
                "entropy": float(entropy),
                "fidelity": float(fidelity),
                "total_shots": total_shots,
                "num_qubits": 3,
                "circuit_depth": circuit.depth()
            },
            "probabilities": probabilities,
            "analysis": {
                "dominant_state": max(counts, key=counts.get),
                "state_diversity": len(counts),
                "bell_state_quality": f"{fidelity*100:.1f}%"
            }
        }
        
        # Appeler l'API avec le token d'authentification
        response = requests.post(
            f"{API_URL}/api/runs/{RUN_ID}/publish",
            headers={
                "Authorization": f"Bearer {API_TOKEN}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            print("✅ Résultats publiés avec succès !")
            result_data = response.json()
            print(f"Réponse : {result_data}")
        else:
            print(f"❌ Erreur lors de la publication : {response.status_code}")
            print(f"Message : {response.text}")
    
    except Exception as e:
        print(f"❌ Erreur de connexion à l'API : {str(e)}")
        print("Les résultats seront quand même visibles dans l'exécution standard.")

else:
    print("⚠️  Variables d'environnement API non trouvées")
    print("Ce script doit être exécuté via la plateforme QuantumOps")
    print("Les résultats bruts :")
    print(counts)

print()
print("=== Exécution terminée ===")
