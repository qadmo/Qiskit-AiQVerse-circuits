"""
Simple Quantum Bell State with API Publishing
==============================================

Script simplifié qui crée un état de Bell et publie les résultats via l'API.
Compatible avec la fonction prepare_circuit de la plateforme.
"""

import os
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

# Créer le circuit quantique (état de Bell sur 3 qubits)
qr = QuantumRegister(3, 'q')
cr = ClassicalRegister(3, 'c')
circuit = QuantumCircuit(qr, cr)

# Circuit de test : état de Bell + mesure sur 3 qubits
circuit.h(qr[0])  # Hadamard sur q0
circuit.cx(qr[0], qr[1])  # CNOT q0->q1
circuit.cx(qr[0], qr[2])  # CNOT q0->q2
circuit.measure(qr, cr)

# Note: Le reste du code (exécution, analyse, publication) 
# ne s'exécute que si les variables d'environnement sont présentes
# Cela évite les erreurs lors de prepare_circuit

if os.environ.get('QUANTUMOPS_RUN_ID'):
    # Ce bloc s'exécute seulement pendant l'exécution réelle,
    # pas pendant prepare_circuit
    
    import requests
    import numpy as np
    from qiskit_aer import AerSimulator
    
    RUN_ID = os.environ['QUANTUMOPS_RUN_ID']
    API_URL = os.environ['QUANTUMOPS_API_URL']
    API_TOKEN = os.environ['QUANTUMOPS_API_TOKEN']
    
    print(f"=== QuantumOps API Example ===")
    print(f"Run ID: {RUN_ID}")
    print()
    
    # Exécuter le circuit
    print("⚛️  Exécution du circuit...")
    simulator = AerSimulator()
    job = simulator.run(circuit, shots=1024)
    result = job.result()
    counts = result.get_counts(circuit)
    
    print(f"Résultats : {counts}")
    print()
    
    # Analyser les résultats
    print("🔬 Analyse...")
    total_shots = sum(counts.values())
    probabilities = {state: count/total_shots for state, count in counts.items()}
    
    # Calcul d'entropie
    entropy = -sum(p * np.log2(p) for p in probabilities.values() if p > 0)
    
    # Calcul de fidélité (états idéaux: 000 et 111)
    ideal_states = ['000', '111']
    fidelity = sum(counts.get(state, 0) for state in ideal_states) / total_shots
    
    # Eigenvalues simulées
    eigenvalues = [1.0, 0.5, -0.3]
    
    print(f"Entropie: {entropy:.4f}, Fidélité: {fidelity:.4f}")
    print()
    
    # Publier via l'API
    print("📡 Publication des résultats...")
    try:
        payload = {
            "counts": counts,
            "eigenvalues": eigenvalues,
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
            print(f"Réponse: {response.json()}")
        else:
            print(f"❌ Erreur {response.status_code}: {response.text}")
    
    except Exception as e:
        print(f"❌ Erreur API: {str(e)}")
    
    print()
    print("=== Terminé ===")
