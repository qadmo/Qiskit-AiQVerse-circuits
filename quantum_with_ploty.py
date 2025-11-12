"""
Quantum Circuit with Plotly Visualizations
===========================================

Ce script montre comment créer un modèle quantique avec des visualisations
interactives Plotly qui s'affichent dans l'UI de QuantumOps.

Les graphiques incluent :
- Histogramme des mesures
- Distribution des probabilités
- Graphique de convergence
- Métriques temporelles
"""

import os
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

# Créer le circuit quantique (état de Bell sur 3 qubits)
qr = QuantumRegister(3, 'q')
cr = ClassicalRegister(3, 'c')
circuit = QuantumCircuit(qr, cr)

# Circuit : état GHZ (Greenberger-Horne-Zeilinger)
circuit.h(qr[0])
circuit.cx(qr[0], qr[1])
circuit.cx(qr[1], qr[2])
circuit.measure(qr, cr)

# Ce bloc s'exécute seulement pendant l'exécution réelle
if os.environ.get('QUANTUMOPS_RUN_ID'):
    import requests
    import numpy as np
    from qiskit_aer import AerSimulator
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    RUN_ID = os.environ['QUANTUMOPS_RUN_ID']
    API_URL = os.environ['QUANTUMOPS_API_URL']
    API_TOKEN = os.environ['QUANTUMOPS_API_TOKEN']
    
    print("=== Quantum with Plotly Visualizations ===")
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
    print("📊 Création des visualisations...")
    total_shots = sum(counts.values())
    probabilities = {state: count/total_shots for state, count in counts.items()}
    
    # Calcul de métriques
    entropy = -sum(p * np.log2(p) for p in probabilities.values() if p > 0)
    ideal_states = ['000', '111']
    fidelity = sum(counts.get(state, 0) for state in ideal_states) / total_shots
    
    # === GRAPHIQUE 1 : Histogramme des mesures ===
    states = list(counts.keys())
    values = list(counts.values())
    
    fig1 = go.Figure(data=[
        go.Bar(
            x=states,
            y=values,
            marker=dict(
                color=values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Count")
            ),
            text=values,
            textposition='auto',
            hovertemplate='<b>State: |%{x}⟩</b><br>Count: %{y}<br>Probability: %{customdata:.2%}<extra></extra>',
            customdata=[count/total_shots for count in values]
        )
    ])
    
    fig1.update_layout(
        title='Quantum Measurement Distribution',
        xaxis_title='Quantum State',
        yaxis_title='Count',
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    # === GRAPHIQUE 2 : Pie chart des probabilités ===
    fig2 = go.Figure(data=[
        go.Pie(
            labels=[f'|{state}⟩' for state in states],
            values=values,
            hole=0.3,
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Probability: %{percent}<extra></extra>'
        )
    ])
    
    fig2.update_layout(
        title='State Probability Distribution',
        template='plotly_white',
        height=400
    )
    
    # === GRAPHIQUE 3 : Métriques de qualité ===
    metrics_names = ['Fidelity', 'Entropy (norm.)', 'State Purity']
    metrics_values = [
        fidelity * 100,
        (entropy / np.log2(len(states))) * 100 if len(states) > 1 else 0,
        (1 - entropy / np.log2(8)) * 100  # Pureté normalisée
    ]
    
    fig3 = go.Figure(data=[
        go.Bar(
            x=metrics_names,
            y=metrics_values,
            marker=dict(
                color=['#34A853', '#EA4335', '#4285F4'],
                line=dict(color='rgba(0,0,0,0.3)', width=1)
            ),
            text=[f'{v:.1f}%' for v in metrics_values],
            textposition='outside'
        )
    ])
    
    fig3.update_layout(
        title='Quantum State Quality Metrics',
        yaxis_title='Percentage (%)',
        yaxis_range=[0, 110],
        template='plotly_white',
        height=400,
        showlegend=False
    )
    
    # === GRAPHIQUE 4 : Simulation de convergence ===
    # Simule plusieurs tailles d'échantillons pour montrer la convergence
    shot_sizes = [64, 128, 256, 512, 1024]
    convergence_data = []
    
    for shots in shot_sizes:
        job_temp = simulator.run(circuit, shots=shots)
        result_temp = job_temp.result()
        counts_temp = result_temp.get_counts(circuit)
        fid_temp = sum(counts_temp.get(s, 0) for s in ideal_states) / shots
        convergence_data.append(fid_temp)
    
    fig4 = go.Figure(data=[
        go.Scatter(
            x=shot_sizes,
            y=convergence_data,
            mode='lines+markers',
            marker=dict(size=10, color='#1967d2'),
            line=dict(width=3, color='#1967d2'),
            hovertemplate='<b>Shots: %{x}</b><br>Fidelity: %{y:.3f}<extra></extra>'
        )
    ])
    
    fig4.add_hline(
        y=fidelity,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Final Fidelity: {fidelity:.3f}",
        annotation_position="right"
    )
    
    fig4.update_layout(
        title='Convergence Analysis',
        xaxis_title='Number of Shots',
        yaxis_title='Fidelity',
        template='plotly_white',
        height=400,
        yaxis_range=[0, 1.1]
    )
    
    print("✅ Visualisations créées")
    print()
    
    # Publier via l'API
    print("📡 Publication des résultats et graphiques...")
    try:
        payload = {
            "counts": counts,
            "custom_metrics": {
                "entropy": float(entropy),
                "fidelity": float(fidelity),
                "total_shots": total_shots,
                "num_qubits": 3,
                "circuit_depth": circuit.depth(),
                "ideal_state_probability": float(fidelity)
            },
            "probabilities": probabilities,
            "analysis": {
                "dominant_states": ', '.join([f'|{s}⟩' for s, c in sorted(counts.items(), key=lambda x: -x[1])[:2]]),
                "state_diversity": len(counts),
                "ghz_quality": f"{fidelity*100:.1f}%",
                "entropy_bits": f"{entropy:.3f}"
            },
            "plotly_charts": [
                {
                    "title": "Measurement Distribution Histogram",
                    "figure": fig1.to_dict()
                },
                {
                    "title": "Probability Distribution (Pie Chart)",
                    "figure": fig2.to_dict()
                },
                {
                    "title": "Quality Metrics",
                    "figure": fig3.to_dict()
                },
                {
                    "title": "Convergence Analysis",
                    "figure": fig4.to_dict()
                }
            ]
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
            print("✅ Résultats et graphiques publiés avec succès !")
            print(f"   - 4 graphiques Plotly interactifs")
            print(f"   - Métriques quantiques")
            print(f"   - Analyse de convergence")
            print(f"Réponse: {response.json()}")
        else:
            print(f"❌ Erreur {response.status_code}: {response.text}")
    
    except Exception as e:
        print(f"❌ Erreur API: {str(e)}")
    
    print()
    print("=== Terminé ===")
